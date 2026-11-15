"""weight-only int8 量化：语义、反向与入口验收。

三条判据，与项目的其它算子一致：

1. 参考语义是**手写公式**，不是拿自己给自己打分：RNE 舍入、[-127,127]
   饱和、逐通道 scale 广播。
2. 反向只对连续量（激活 / bias）给梯度；量化权重与 scale 是冻结常量，
   对它们求导必须显式报错，而不是返回零。
3. 图能走通完整链路：构图 → 执行 → JSON 往返 → 再执行，结果逐位相同。
"""

import unittest

import numpy as np

from aetherflow import (
    Graph,
    NoGradientError,
    ReferenceBackend,
    backward,
    dumps,
    f16,
    f32,
    i8,
    loads,
)
from aetherflow.conformance import default_cases, run_conformance
from aetherflow.models.quantize import (
    QUANT_MAX,
    build_int8_linear,
    dequantize_weight,
    quantize_weight,
)

BACKEND = ReferenceBackend()


def _run(graph, feed):
    return BACKEND.execute(graph, feed)


def _quantize_graph(shape):
    g = Graph("quantize_i8")
    w = g.input("w", f32[shape])
    s = g.input("s", f32[(shape[-1],)])
    g.output(g.add_node("quantize_i8", [w, s]), "q")
    return g


class TestQuantizeSemantics(unittest.TestCase):
    def test_rint_is_round_half_to_even(self):
        """半偶舍入：0.5→0、1.5→2、2.5→2，-0.5→0、-1.5→-2、-2.5→-2。

        四舍五入（half-away-from-zero）会在 .5 处系统性高估，量化偏差
        随层数累积，因此这里钉住的是 RNE，而不是"大致等于四舍五入"。
        """
        w = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5], dtype=np.float32)
        q = _run(_quantize_graph(w.shape), {"w": w, "s": np.ones(6, np.float32)})["q"]
        np.testing.assert_array_equal(q, np.array([0, 2, 2, 0, -2, -2], np.int8))

    def test_saturates_at_symmetric_endpoints(self):
        w = np.array([127.4, 127.6, -127.4, -127.6, 1000.0, -1000.0], dtype=np.float32)
        q = _run(_quantize_graph(w.shape), {"w": w, "s": np.ones(6, np.float32)})["q"]
        np.testing.assert_array_equal(
            q, np.array([127, 127, -127, -127, 127, -127], np.int8)
        )

    def test_scale_broadcasts_per_channel(self):
        w = np.array([[1.0, 10.0], [-2.0, 20.0], [3.0, -30.0]], dtype=np.float32)
        scale = np.array([1.0, 10.0], dtype=np.float32)
        q = _run(_quantize_graph(w.shape), {"w": w, "s": scale})["q"]
        np.testing.assert_array_equal(
            q, np.array([[1, 1], [-2, 2], [3, -3]], np.int8)
        )

    def test_non_positive_scale_is_rejected(self):
        w = np.ones((2, 3), dtype=np.float32)
        for bad in (0.0, -0.5):
            with self.subTest(scale=bad):
                with self.assertRaises(ValueError):
                    _run(_quantize_graph(w.shape), {"w": w, "s": np.full(3, bad, np.float32)})

    def test_dequantize_is_a_broadcast_multiply(self):
        q = np.array([[1, -2], [3, 4]], dtype=np.int8)
        scale = np.array([0.5, 2.0], dtype=np.float32)
        g = Graph("dequantize_i8")
        qv = g.input("q", i8[2, 2])
        sv = g.input("s", f32[2])
        g.output(g.add_node("dequantize_i8", [qv, sv]), "w")
        got = _run(g, {"q": q, "s": scale})["w"]
        np.testing.assert_allclose(got, q.astype(np.float32) * scale, rtol=0, atol=0)

    def test_quantize_requires_f32_weight(self):
        g = Graph("quantize_i8")
        w = g.input("w", f16[2, 3])
        s = g.input("s", f32[3])
        with self.assertRaises(TypeError):
            g.add_node("quantize_i8", [w, s])

    def test_scale_rank_cannot_exceed_weight(self):
        g = Graph("quantize_i8")
        w = g.input("w", f32[2, 3])
        s = g.input("s", f32[1, 2, 3])
        with self.assertRaises(ValueError):
            g.add_node("quantize_i8", [w, s])

    def test_symbolic_scale_mismatch_is_rejected_at_build_time(self):
        """符号维与静态非 1 维之间的广播 IR 表达不了，必须在构图期拒绝。"""
        g = Graph("quantize_i8")
        w = g.input("w", f32["B", 4])
        s = g.input("s", f32["S"])
        with self.assertRaises(ValueError):
            g.add_node("quantize_i8", [w, s])


class TestRoundTrip(unittest.TestCase):
    def test_error_is_bounded_by_half_a_scale(self):
        rng = np.random.default_rng(7)
        w = (rng.standard_normal((32, 12)) * 3).astype(np.float32)
        q, scale = quantize_weight(w)
        back = dequantize_weight(q, scale)
        err = np.max(np.abs(back - w), axis=0)
        self.assertTrue(np.all(err <= scale / 2 + 1e-5), (err, scale))

    def test_each_nonzero_channel_uses_full_range(self):
        rng = np.random.default_rng(8)
        w = (rng.standard_normal((16, 5)) * 2).astype(np.float32)
        q, scale = quantize_weight(w)
        self.assertEqual(q.dtype, np.int8)
        self.assertEqual(q.shape, w.shape)
        self.assertEqual(scale.shape, (5,))
        self.assertTrue(np.all(np.max(np.abs(q), axis=0) == QUANT_MAX))

    def test_zero_channel_gets_unit_scale(self):
        w = np.zeros((4, 3), dtype=np.float32)
        w[0, 1] = 2.0
        q, scale = quantize_weight(w)
        np.testing.assert_array_equal(q[:, 0], np.zeros(4, np.int8))
        np.testing.assert_array_equal(q[:, 2], np.zeros(4, np.int8))
        self.assertEqual(float(scale[0]), 1.0)
        self.assertGreater(float(scale[1]), 0.0)

    def test_helper_rejects_bad_weights(self):
        with self.assertRaises(ValueError):
            quantize_weight(np.zeros((2, 2, 2), dtype=np.float32))
        with self.assertRaises(TypeError):
            quantize_weight(np.zeros((2, 2), dtype=np.int32))
        with self.assertRaises(ValueError):
            quantize_weight(np.array([[1.0, np.nan]], dtype=np.float32))


class TestLinearI8(unittest.TestCase):
    def _graph(self, x_shape, w_shape, bias=True):
        g = Graph("linear_i8")
        x = g.input("x", f32[x_shape])
        q = g.input("q", i8[w_shape])
        s = g.input("s", f32[(w_shape[-1],)])
        args = [x, q, s]
        if bias:
            args.append(g.input("b", f32[(w_shape[-1],)]))
        g.output(g.add_node("linear_i8", args), "y")
        return g

    def test_matches_explicit_dequantize_then_matmul(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal((2, 4, 8)).astype(np.float32)
        q = rng.integers(-127, 128, size=(8, 6), dtype=np.int8)
        s = (rng.random(6).astype(np.float32) + 0.1)
        b = rng.standard_normal(6).astype(np.float32)
        g = self._graph((2, 4, 8), (8, 6))
        got = _run(g, {"x": x, "q": q, "s": s, "b": b})["y"]
        want = np.matmul(x, q.astype(np.float32) * s) + b
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)

    def test_without_bias(self):
        rng = np.random.default_rng(12)
        x = rng.standard_normal((3, 5)).astype(np.float32)
        q = rng.integers(-127, 128, size=(5, 4), dtype=np.int8)
        s = np.full(4, 0.25, np.float32)
        g = self._graph((3, 5), (5, 4), bias=False)
        got = _run(g, {"x": x, "q": q, "s": s})["y"]
        np.testing.assert_allclose(got, np.matmul(x, q.astype(np.float32) * s), rtol=1e-6)

    def test_rejects_wrong_dtypes_and_ranks(self):
        cases = [
            ("x", f16[2, 4], "q", i8[4, 3], "s", f32[3]),
            ("x", f32[2, 4], "q", f32[4, 3], "s", f32[3]),
            ("x", f32[2, 4], "q", i8[4], "s", f32[3]),
            ("x", f32[4], "q", i8[4, 3], "s", f32[3]),
            ("x", f32[2, 4], "q", i8[4, 3], "s", f32[2, 3]),
        ]
        for xn, xt, qn, qt, sn, st in cases:
            with self.subTest(x=xt, q=qt, s=st):
                g = Graph("linear_i8")
                x = g.input(xn, xt)
                q = g.input(qn, qt)
                s = g.input(sn, st)
                with self.assertRaises((TypeError, ValueError)):
                    g.add_node("linear_i8", [x, q, s])

    def test_inner_dimension_and_bias_length_are_checked(self):
        g = Graph("linear_i8")
        x = g.input("x", f32[2, 5])
        q = g.input("q", i8[4, 3])
        s = g.input("s", f32[3])
        with self.assertRaises(ValueError):
            g.add_node("linear_i8", [x, q, s])

        g = Graph("linear_i8")
        x = g.input("x", f32[2, 4])
        q = g.input("q", i8[4, 3])
        s = g.input("s", f32[3])
        b = g.input("b", f32[4])
        with self.assertRaises(ValueError):
            g.add_node("linear_i8", [x, q, s, b])

    def test_graph_survives_json_roundtrip(self):
        rng = np.random.default_rng(13)
        feed = {
            "x": rng.standard_normal((2, 8)).astype(np.float32),
            "q": rng.integers(-127, 128, size=(8, 3), dtype=np.int8),
            "s": np.array([0.1, 0.2, 0.3], np.float32),
            "b": np.zeros(3, np.float32),
        }
        g = self._graph((2, 8), (8, 3))
        want = _run(g, feed)["y"]
        again = _run(loads(dumps(g)), feed)["y"]
        np.testing.assert_array_equal(want, again)

    def test_zero_scale_fails_at_execution(self):
        g = self._graph((2, 4), (4, 3), bias=False)
        rng = np.random.default_rng(14)
        feed = {
            "x": rng.standard_normal((2, 4)).astype(np.float32),
            "q": np.ones((4, 3), np.int8),
            "s": np.array([0.1, 0.0, 0.2], np.float32),
        }
        with self.assertRaises(ValueError):
            _run(g, feed)


class TestLinearI8Gradient(unittest.TestCase):
    def _graph(self):
        g = Graph("linear_i8")
        x = g.input("x", f32[2, 4])
        q = g.input("q", i8[4, 3])
        s = g.input("s", f32[3])
        b = g.input("b", f32[3])
        g.output(g.add_node("linear_i8", [x, q, s, b]), "y")
        return g

    def test_dx_and_dbias_match_hand_derivation(self):
        rng = np.random.default_rng(21)
        feed = {
            "x": rng.standard_normal((2, 4)).astype(np.float32),
            "q": rng.integers(-127, 128, size=(4, 3), dtype=np.int8),
            "s": np.array([0.5, 0.25, 0.125], np.float32),
            "b": np.zeros(3, np.float32),
        }
        g = self._graph()
        bwd = backward(g, wrt=[g.find("x"), g.find("b")])
        dy = rng.standard_normal((2, 3)).astype(np.float32)
        out = _run(bwd, {**feed, "dy": dy})

        weight = feed["q"].astype(np.float32) * feed["s"]
        np.testing.assert_allclose(out["dx"], np.matmul(dy, weight.T), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(out["db"], dy.sum(axis=0), rtol=1e-6, atol=1e-6)

    def test_quantized_weight_has_no_gradient(self):
        g = self._graph()
        with self.assertRaises(NoGradientError):
            backward(g, wrt=[g.find("q")])
        with self.assertRaises(NoGradientError):
            backward(g, wrt=[g.find("s")])


class TestQuantizedLinearBuilder(unittest.TestCase):
    def test_build_int8_linear_cuts_weight_footprint(self):
        rng = np.random.default_rng(31)
        w = (rng.standard_normal((16, 5)) * 0.5).astype(np.float32)
        b = rng.standard_normal(5).astype(np.float32)
        x = rng.standard_normal((3, 16)).astype(np.float32)

        q, scale = quantize_weight(w)
        g = Graph("int8_linear")
        inputs, y = build_int8_linear(g, 16, 5)
        g.output(y, "y")
        got = _run(g, {"x": x, "w_i8": q, "scale": scale, "bias": b})["y"]

        # 8 bit 权重的误差量级是 scale/2；K=16 的点积把它放大到 1e-2 量级，
        # 所以这里比的是"同一个量级"，不是逐位——要求逐位就只能是假断言。
        np.testing.assert_allclose(got, x @ w + b, rtol=2e-2, atol=2e-2)
        # 权重驻留从 4 字节降到 1 字节；scale 与 bias 是每通道一份的小量
        self.assertEqual(q.nbytes * 4, w.nbytes)

    def test_builder_rejects_non_positive_features(self):
        g = Graph("int8_linear")
        with self.assertRaises(ValueError):
            build_int8_linear(g, 0, 4)


class TestConformanceCases(unittest.TestCase):
    def test_reference_backend_passes_new_cases(self):
        names = {"quantize_i8_channel", "dequantize_i8_channel", "linear_i8_bias"}
        cases = [c for c in default_cases() if c.name in names]
        self.assertEqual({c.name for c in cases}, names)
        report = run_conformance(BACKEND, cases=cases)
        self.assertTrue(report.ok, report.text())


if __name__ == "__main__":
    unittest.main()
