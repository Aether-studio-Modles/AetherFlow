"""int8 量化：weight-only 与 W8A8 的语义、反向与入口验收。

三条判据，与项目的其它算子一致：

1. 参考语义是**手写公式**，不是拿自己给自己打分：RNE 舍入、[-127,127]
   饱和、逐通道 scale 广播；W8A8 还要额外钉住"累加发生在 int32 里"。
2. 反向只对连续量（激活 / bias）给梯度；量化权重、int8 输入与 scale 是
   冻结常量，对它们求导必须显式报错，而不是返回零。
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
    build_int8a8_linear,
    dequantize_weight,
    quantize_activation,
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


class TestLinearI8A8(unittest.TestCase):
    """W8A8：int8 激活 × int8 权重先在 int32 里累加，再乘两个 scale。

    这一组要钉住的不是"和 f32 差不多"，而是**累加发生在整数域**：
    手写公式逐位相等，并且与"先反量化再 f32 matmul"确实不同。
    """

    def _graph(self, x_shape, w_shape, bias=True, x_scale_shape=None):
        g = Graph("linear_i8a8")
        q = g.input("x_q", i8[x_shape])
        sx = g.input(
            "x_scale",
            f32[x_scale_shape if x_scale_shape is not None else x_shape[:-1] + (1,)],
        )
        wq = g.input("w_q", i8[w_shape])
        sw = g.input("w_scale", f32[(w_shape[-1],)])
        args = [q, sx, wq, sw]
        if bias:
            args.append(g.input("b", f32[(w_shape[-1],)]))
        g.output(g.add_node("linear_i8a8", args), "y")
        return g

    def test_matches_hand_written_int32_accumulation(self):
        rng = np.random.default_rng(41)
        x_shape, w_shape = (2, 4, 8), (8, 6)
        xq = rng.integers(-127, 128, size=x_shape, dtype=np.int8)
        wq = rng.integers(-127, 128, size=w_shape, dtype=np.int8)
        sx = (rng.random(x_shape[:-1] + (1,)).astype(np.float32) + 0.02)
        sw = (rng.random(w_shape[-1]).astype(np.float32) + 0.05)
        b = rng.standard_normal(w_shape[-1]).astype(np.float32)

        got = _run(
            self._graph(x_shape, w_shape),
            {"x_q": xq, "x_scale": sx, "w_q": wq, "w_scale": sw, "b": b},
        )["y"]
        acc = np.matmul(xq.astype(np.int32), wq.astype(np.int32)).astype(np.float32)
        want = acc * sx * sw + b
        # 整数累加 + 三次浮点运算，没有理由不是逐位相同。
        np.testing.assert_array_equal(got, want)

    def test_accumulation_is_not_f32_dequantize_then_matmul(self):
        """K 足够大时，f32 逐项累加与整数累加的舍入不同。

        若这个算子只是 ``matmul(dequantize(x), dequantize(w))`` 的别名，
        这条会失败——那说明新增的语义没有存在价值。
        """
        shapes = ((3, 512), (512, 7))
        xq = np.random.default_rng(42).integers(-127, 128, size=shapes[0], dtype=np.int8)
        wq = np.random.default_rng(43).integers(-127, 128, size=shapes[1], dtype=np.int8)
        sx = np.full((shapes[0][0], 1), 0.013, np.float32)
        sw = np.full(shapes[1][1], 0.011, np.float32)
        got = _run(
            self._graph(shapes[0], shapes[1], bias=False),
            {"x_q": xq, "x_scale": sx, "w_q": wq, "w_scale": sw},
        )["y"]
        dequant = np.matmul(
            xq.astype(np.float32) * sx, wq.astype(np.float32) * sw
        )
        self.assertGreater(float(np.max(np.abs(got - dequant))), 0.0)

    def test_per_tensor_activation_scale(self):
        rng = np.random.default_rng(44)
        x_shape, w_shape = (2, 3, 5), (5, 4)
        feed = {
            "x_q": rng.integers(-127, 128, size=x_shape, dtype=np.int8),
            "x_scale": np.array([0.05], np.float32),
            "w_q": rng.integers(-127, 128, size=w_shape, dtype=np.int8),
            "w_scale": np.array([0.1, 0.2, 0.3, 0.4], np.float32),
            "b": np.zeros(4, np.float32),
        }
        got = _run(self._graph(x_shape, w_shape, x_scale_shape=(1,)), feed)["y"]
        acc = np.matmul(
            feed["x_q"].astype(np.int32), feed["w_q"].astype(np.int32)
        ).astype(np.float32)
        np.testing.assert_array_equal(got, acc * 0.05 * feed["w_scale"])

    def test_rejects_wrong_dtypes_ranks_and_scales(self):
        good = dict(x_shape=(2, 4), w_shape=(4, 3))
        cases = [
            ("xq", f32[2, 4], "x_scale", f32[2, 1]),
            ("xq", i8[2, 4], "x_scale", f16[2, 1]),
        ]
        for qn, qt, sn, st in cases:
            with self.subTest(x=qt, scale=st):
                g = Graph("linear_i8a8")
                q = g.input(qn, qt)
                sx = g.input(sn, st)
                wq = g.input("w_q", i8[good["w_shape"]])
                sw = g.input("w_scale", f32[3])
                with self.assertRaises((TypeError, ValueError)):
                    g.add_node("linear_i8a8", [q, sx, wq, sw])

        bad_builds = [
            ("权重不是 i8", lambda: self._graph_bad(w_dtype=f32)),
            ("激活只有 1 维", lambda: self._graph_bad(x_shape=(4,))),
            ("权重不是 2 维", lambda: self._graph_bad(w_shape=(2, 2, 3))),
            ("内维不匹配", lambda: self._graph_bad(x_shape=(2, 5))),
            (
                "激活 scale 沿 K 维变化",
                lambda: self._graph_bad(x_scale_shape=(2, 4)),
            ),
            ("权重 scale 是 2 维", lambda: self._graph_bad(w_scale_shape=(4, 3))),
            ("权重 scale 长度不匹配", lambda: self._graph_bad(w_scale_shape=(2,))),
            ("bias 长度不匹配", lambda: self._graph_bad(bias_len=2)),
            ("输入个数不足", lambda: self._graph_bad(drop=("w_scale",))),
            ("K 超过 int32 累加上界", lambda: self._graph_bad(x_shape=(2, 200000))),
        ]
        for label, build in bad_builds:
            with self.subTest(case=label):
                with self.assertRaises((TypeError, ValueError)):
                    build()

    def _graph_bad(
        self,
        x_shape=(2, 4),
        w_shape=(4, 3),
        x_scale_shape=None,
        w_scale_shape=None,
        w_dtype=i8,
        bias_len=None,
        drop=(),
    ):
        g = Graph("linear_i8a8")
        q = g.input("x_q", i8[x_shape])
        sx = g.input(
            "x_scale",
            f32[x_scale_shape if x_scale_shape is not None else x_shape[:-1] + (1,)],
        )
        wq = g.input("w_q", w_dtype[w_shape])
        sw = g.input(
            "w_scale",
            f32[w_scale_shape if w_scale_shape is not None else (w_shape[-1],)],
        )
        args = [q, sx, wq, sw]
        if bias_len is not None:
            args.append(g.input("b", f32[bias_len]))
        named = dict(zip(("x_q", "x_scale", "w_q", "w_scale", "b"), args))
        args = [v for name, v in named.items() if name not in drop]
        g.add_node("linear_i8a8", args)

    def test_non_positive_scale_fails_at_execution(self):
        rng = np.random.default_rng(45)
        feed = {
            "x_q": rng.integers(-127, 128, size=(2, 4), dtype=np.int8),
            "x_scale": np.array([[0.1], [0.0]], np.float32),
            "w_q": rng.integers(-127, 128, size=(4, 3), dtype=np.int8),
            "w_scale": np.array([0.1, 0.2, 0.3], np.float32),
        }
        with self.assertRaises(ValueError):
            _run(self._graph((2, 4), (4, 3), bias=False), feed)

        feed["x_scale"] = np.full((2, 1), 0.1, np.float32)
        feed["w_scale"] = np.array([0.1, -0.2, 0.3], np.float32)
        with self.assertRaises(ValueError):
            _run(self._graph((2, 4), (4, 3), bias=False), feed)

    def test_graph_survives_json_roundtrip(self):
        rng = np.random.default_rng(46)
        feed = {
            "x_q": rng.integers(-127, 128, size=(2, 4), dtype=np.int8),
            "x_scale": np.array([[0.1], [0.2]], np.float32),
            "w_q": rng.integers(-127, 128, size=(4, 3), dtype=np.int8),
            "w_scale": np.array([0.1, 0.2, 0.3], np.float32),
            "b": np.zeros(3, np.float32),
        }
        g = self._graph((2, 4), (4, 3))
        want = _run(g, feed)["y"]
        again = _run(loads(dumps(g)), feed)["y"]
        np.testing.assert_array_equal(want, again)


class TestQuantizeActivation(unittest.TestCase):
    def test_per_token_scale_shape_and_error_bound(self):
        rng = np.random.default_rng(51)
        x = (rng.standard_normal((2, 3, 16)) * 2).astype(np.float32)
        q, scale = quantize_activation(x)
        self.assertEqual(q.dtype, np.int8)
        self.assertEqual(q.shape, x.shape)
        self.assertEqual(scale.shape, (2, 3, 1))
        err = np.max(np.abs(q.astype(np.float32) * scale - x), axis=-1)
        self.assertTrue(np.all(err <= scale[..., 0] / 2 + 1e-5), (err, scale))

    def test_zero_token_gets_unit_scale(self):
        x = np.zeros((2, 4), dtype=np.float32)
        x[0, 1] = 3.0
        q, scale = quantize_activation(x)
        np.testing.assert_array_equal(q[1], np.zeros(4, np.int8))
        np.testing.assert_array_equal(q[0], np.array([0, 127, 0, 0], np.int8))
        self.assertEqual(float(scale[1, 0]), 1.0)
        self.assertGreater(float(scale[0, 0]), 0.0)

    def test_rejects_bad_activations(self):
        with self.assertRaises(ValueError):
            quantize_activation(np.zeros((4,), np.float32))
        with self.assertRaises(TypeError):
            quantize_activation(np.zeros((2, 4), np.int32))
        with self.assertRaises(ValueError):
            quantize_activation(np.array([[1.0, np.inf]], np.float32))


class TestQuantizedW8A8Builder(unittest.TestCase):
    def test_builder_matches_f32_within_quantization_error(self):
        rng = np.random.default_rng(61)
        w = (rng.standard_normal((16, 5)) * 0.5).astype(np.float32)
        b = rng.standard_normal(5).astype(np.float32)
        x = (rng.standard_normal((3, 16)) * 0.5).astype(np.float32)

        xq, xs = quantize_activation(x)
        wq, ws = quantize_weight(w)
        g = Graph("int8a8_linear")
        inputs, y = build_int8a8_linear(g, 16, 5)
        g.output(y, "y")
        got = _run(g, {"x_q": xq, "x_scale": xs, "w_q": wq, "w_scale": ws, "bias": b})[
            "y"
        ]
        # 激活与权重各多一份 scale/2 的舍入，K=16 的点积把它放大到 1e-2 量级。
        np.testing.assert_allclose(got, x @ w + b, rtol=5e-2, atol=5e-2)

    def test_builder_rejects_non_positive_features(self):
        g = Graph("int8a8_linear")
        with self.assertRaises(ValueError):
            build_int8a8_linear(g, 0, 4)


class TestLinearI8A8Gradient(unittest.TestCase):
    def _graph(self):
        g = Graph("linear_i8a8")
        q = g.input("x_q", i8[2, 4])
        sx = g.input("x_scale", f32[2, 1])
        wq = g.input("w_q", i8[4, 3])
        sw = g.input("w_scale", f32[3])
        b = g.input("b", f32[3])
        g.output(g.add_node("linear_i8a8", [q, sx, wq, sw, b]), "y")
        return g

    def test_only_bias_receives_gradient(self):
        rng = np.random.default_rng(71)
        feed = {
            "x_q": rng.integers(-127, 128, size=(2, 4), dtype=np.int8),
            "x_scale": np.array([[0.1], [0.2]], np.float32),
            "w_q": rng.integers(-127, 128, size=(4, 3), dtype=np.int8),
            "w_scale": np.array([0.1, 0.2, 0.3], np.float32),
            "b": np.zeros(3, np.float32),
        }
        g = self._graph()
        bwd = backward(g, wrt=[g.find("b")])
        dy = rng.standard_normal((2, 3)).astype(np.float32)
        out = _run(bwd, {**feed, "dy": dy})
        np.testing.assert_allclose(out["db"], dy.sum(axis=0), rtol=1e-6, atol=1e-6)

    def test_int8_inputs_and_scales_have_no_gradient(self):
        g = self._graph()
        for name in ("x_q", "x_scale", "w_q", "w_scale"):
            with self.subTest(wrt=name):
                with self.assertRaises(NoGradientError):
                    backward(g, wrt=[g.find(name)])


class TestFakeQuantizeI8(unittest.TestCase):
    """QAT 的假量化：f32 进、f32 出，前向是"量化-反量化"的组合，反向是 STE。

    与 ``quantize_i8`` 的差别只有两点：输出仍是 f32（因此能接进训练图），
    以及反向有一条直通估计器规则。前向必须与显式串联两个算子逐位相同——
    否则"假量化"和"真量化再反量化"就是两套语义。
    """

    def _graph(self, shape=(4, 3)):
        g = Graph("fake_quantize_i8")
        x = g.input("x", f32[shape])
        s = g.input("s", f32[(shape[-1],)])
        g.output(g.add_node("fake_quantize_i8", [x, s]), "y")
        return g

    def _two_op_graph(self, shape=(4, 3)):
        g = Graph("quantize_then_dequantize")
        x = g.input("x", f32[shape])
        s = g.input("s", f32[(shape[-1],)])
        q = g.add_node("quantize_i8", [x, s])
        g.output(g.add_node("dequantize_i8", [q, s]), "y")
        return g

    def test_forward_matches_explicit_quantize_then_dequantize(self):
        rng = np.random.default_rng(81)
        shape = (5, 4)
        feed = {
            "x": (rng.standard_normal(shape) * 3).astype(np.float32),
            "s": np.array([0.1, 0.05, 0.25, 0.5], np.float32),
        }
        got = _run(self._graph(shape), feed)["y"]
        want = _run(self._two_op_graph(shape), feed)["y"]
        np.testing.assert_array_equal(got, want)

    def test_rounding_is_rne_and_saturates(self):
        # scale=1 时格点就是整数：0.5→0、1.5→2、-0.5→0（半偶）；
        # 127.6 与 -127.6 被夹到 ±127。
        x = np.array([0.5, 1.5, 2.5, -0.5, -1.5, 127.6, -127.6], np.float32)
        s = np.ones((1,), np.float32)
        got = _run(self._graph((7, 1)), {"x": x.reshape(7, 1), "s": s})["y"]
        np.testing.assert_array_equal(
            got.reshape(-1), np.array([0.0, 2.0, 2.0, 0.0, -2.0, 127.0, -127.0], np.float32)
        )

    def test_zero_input_stays_zero_for_any_scale(self):
        x = np.zeros((2, 3), np.float32)
        s = np.array([0.1, 1.0, 7.0], np.float32)
        got = _run(self._graph((2, 3)), {"x": x, "s": s})["y"]
        np.testing.assert_array_equal(got, x)

    def test_ste_gradient_is_identity(self):
        """反向必须原样穿过量化：dw 与"没有假量化"时逐位相同。"""
        rng = np.random.default_rng(82)
        g = Graph("qat")
        x = g.input("x", f32[2, 4])
        w = g.input("w", f32[4, 3])
        s = g.input("s", f32[3])
        wq = g.add_node("fake_quantize_i8", [w, s])
        g.output(g.matmul(x, wq), "y")

        feed = {
            "x": rng.standard_normal((2, 4)).astype(np.float32),
            "w": (rng.standard_normal((4, 3)) * 2).astype(np.float32),
            "s": np.array([0.1, 0.2, 0.3], np.float32),
        }
        dy = rng.standard_normal((2, 3)).astype(np.float32)
        out = _run(backward(g, wrt=[g.find("w")]), {**feed, "dy": dy})
        np.testing.assert_array_equal(out["dw"], feed["x"].T @ dy)

    def test_quantization_scale_has_no_gradient(self):
        g = self._graph((4, 3))
        with self.assertRaises(NoGradientError):
            backward(g, wrt=[g.find("s")])

    def test_rejections(self):
        with self.subTest(case="non_f32_input"):
            g = Graph("fake_quantize_i8")
            x = g.input("x", f16[2, 3])
            s = g.input("s", f32[3])
            with self.assertRaises(TypeError):
                g.add_node("fake_quantize_i8", [x, s])
        with self.subTest(case="non_f32_scale"):
            g = Graph("fake_quantize_i8")
            x = g.input("x", f32[2, 3])
            s = g.input("s", f16[3])
            with self.assertRaises(TypeError):
                g.add_node("fake_quantize_i8", [x, s])
        with self.subTest(case="scale_rank_exceeds_input"):
            g = Graph("fake_quantize_i8")
            x = g.input("x", f32[2, 3])
            s = g.input("s", f32[1, 2, 3])
            with self.assertRaises(ValueError):
                g.add_node("fake_quantize_i8", [x, s])
        with self.subTest(case="symbolic_scale_mismatch"):
            g = Graph("fake_quantize_i8")
            x = g.input("x", f32["B", 4])
            s = g.input("s", f32["S"])
            with self.assertRaises(ValueError):
                g.add_node("fake_quantize_i8", [x, s])
        with self.subTest(case="wrong_arity"):
            g = Graph("fake_quantize_i8")
            x = g.input("x", f32[2, 3])
            with self.assertRaises(ValueError):
                g.add_node("fake_quantize_i8", [x])
        with self.subTest(case="non_positive_scale_at_runtime"):
            g = self._graph((2, 3))
            for bad in (0.0, -0.5):
                with self.assertRaises(ValueError):
                    _run(
                        g,
                        {"x": np.ones((2, 3), np.float32), "s": np.full(3, bad, np.float32)},
                    )

    def test_json_round_trip(self):
        rng = np.random.default_rng(83)
        g = self._graph((3, 2))
        feed = {
            "x": (rng.standard_normal((3, 2)) * 2).astype(np.float32),
            "s": np.array([0.125, 0.5], np.float32),
        }
        again = loads(dumps(g))
        np.testing.assert_array_equal(_run(g, feed)["y"], _run(again, feed)["y"])


class TestFakeQuantizeI8LSQ(unittest.TestCase):
    """LSQ：前向与 STE 版逐位相同，反向换成"非饱和区直通 + 可学步长"。

    这个算子存在的唯一理由是 ``scale`` 能训练。判据因此不是"梯度形状对不对"，
    而是三件事：前向不能与 ``fake_quantize_i8`` 分叉；``dx`` 必须把被夹到端点
    的元素置零；``ds`` 的符号与大小要能把步长推向"量化误差更小"的方向。
    """

    def _graph(self, shape=(4, 3)):
        g = Graph("fake_quantize_i8_lsq")
        x = g.input("x", f32[shape])
        s = g.input("s", f32[(shape[-1],)])
        g.output(g.add_node("fake_quantize_i8_lsq", [x, s]), "y")
        return g

    def test_forward_is_bit_identical_to_ste_version(self):
        """两个算子的前向必须共用一份舍入：否则 QAT 与 LSQ 会给出两套数值。"""
        rng = np.random.default_rng(91)
        shape = (5, 4)
        feed = {
            # 量级刻意跨过饱和端点：前向差异最容易在夹取处暴露。
            "x": (rng.standard_normal(shape) * 20).astype(np.float32),
            "s": np.array([0.1, 0.05, 0.25, 0.5], np.float32),
        }
        g_ste = Graph("fake_quantize_i8")
        sx = g_ste.input("x", f32[shape])
        ss = g_ste.input("s", f32[(shape[-1],)])
        g_ste.output(g_ste.add_node("fake_quantize_i8", [sx, ss]), "y")
        np.testing.assert_array_equal(
            _run(self._graph(shape), feed)["y"], _run(g_ste, feed)["y"]
        )

    def test_dx_masks_saturated_elements(self):
        """饱和列的输入梯度为 0，非饱和列与 STE 一样原样穿过（逐位）。"""
        rng = np.random.default_rng(92)
        g = Graph("qat_lsq")
        x = g.input("x", f32[2, 3])
        w = g.input("w", f32[3, 2])
        s = g.input("s", f32[2])
        wq = g.add_node("fake_quantize_i8_lsq", [w, s])
        g.output(g.matmul(x, wq), "y")

        feed = {
            "x": rng.standard_normal((2, 3)).astype(np.float32),
            # w[0, 0] = 100 配 s=0.1 → v = 1000，被夹到 127：该位置不接收梯度。
            "w": np.array([[100.0, 0.4], [0.5, 0.6], [0.7, 0.8]], np.float32),
            "s": np.array([0.1, 0.1], np.float32),
            "dy": np.ones((2, 2), np.float32),
        }
        got = _run(backward(g, wrt=[g.find("w")]), feed)["dw"]
        want = feed["x"].T @ feed["dy"]
        want[0, 0] = 0.0
        np.testing.assert_array_equal(got, want)

    def test_scale_gradient_matches_lsq_formula(self):
        """``ds = sum(g * (v_hat - v))``：逐通道归约，按公式手算对拍。"""
        rng = np.random.default_rng(93)
        g = self._graph((4, 3))
        x = np.array(
            [
                [0.31, 2.7, -300.0],  # -300 / 1.0 → 下饱和
                [-0.52, 0.13, 0.9],
                [1.05, -3.2, 0.0],
                [0.47, 0.66, 5.5],
            ],
            np.float32,
        )
        s = np.array([0.1, 0.25, 1.0], np.float32)
        dy = rng.standard_normal((4, 3)).astype(np.float32)
        got = _run(backward(g, wrt=[g.find("s")]), {"x": x, "s": s, "dy": dy})["ds"]

        v = (x / s).astype(np.float32)
        vq = np.clip(np.rint(v), -127.0, 127.0).astype(np.float32)
        want = (dy * (vq - v)).sum(axis=0)
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)

    def test_one_gradient_step_reduces_quantization_error(self):
        """沿负梯度走一步，量化误差必须下降——这是"步长真的能学"的最小证据。"""
        g = self._graph((4, 1))
        x = np.array([[100.0], [-100.0], [80.0], [-60.0]], np.float32)
        s0 = np.array([0.1], np.float32)
        dy = np.ones((4, 1), np.float32)
        ds = _run(backward(g, wrt=[g.find("s")]), {"x": x, "s": s0, "dy": dy})["ds"]
        s1 = (s0 - 0.001 * ds).astype(np.float32)
        err0 = np.abs(x - _run(g, {"x": x, "s": s0})["y"]).mean()
        err1 = np.abs(x - _run(g, {"x": x, "s": s1})["y"]).mean()
        self.assertLess(float(err1), float(err0))

    def test_backward_graph_uses_only_device_ops(self):
        """反向图必须整体落在设备算子面内，否则"设备上能训练"只是口号。"""
        from aetherflow.backends.device_shared import GPU_OPS

        g = Graph("qat_lsq")
        x = g.input("x", f32[2, 4])
        w = g.input("w", f32[4, 3])
        s = g.input("s", f32[3])
        wq = g.add_node("fake_quantize_i8_lsq", [w, s])
        g.output(g.matmul(x, wq), "y")
        bwd = backward(g, wrt=[g.find("w"), g.find("s")])
        used = {node.op.name for node in bwd.nodes}
        self.assertLessEqual(used, set(GPU_OPS), used - set(GPU_OPS))

    def test_rejections(self):
        with self.subTest(case="non_f32_input"):
            g = Graph("fake_quantize_i8_lsq")
            x = g.input("x", f16[2, 3])
            s = g.input("s", f32[3])
            with self.assertRaises(TypeError):
                g.add_node("fake_quantize_i8_lsq", [x, s])
        with self.subTest(case="non_f32_scale"):
            g = Graph("fake_quantize_i8_lsq")
            x = g.input("x", f32[2, 3])
            s = g.input("s", f16[3])
            with self.assertRaises(TypeError):
                g.add_node("fake_quantize_i8_lsq", [x, s])
        with self.subTest(case="scale_rank_exceeds_input"):
            g = Graph("fake_quantize_i8_lsq")
            x = g.input("x", f32[2, 3])
            s = g.input("s", f32[1, 2, 3])
            with self.assertRaises(ValueError):
                g.add_node("fake_quantize_i8_lsq", [x, s])
        with self.subTest(case="wrong_arity"):
            g = Graph("fake_quantize_i8_lsq")
            x = g.input("x", f32[2, 3])
            with self.assertRaises(ValueError):
                g.add_node("fake_quantize_i8_lsq", [x])
        with self.subTest(case="non_positive_scale_at_runtime"):
            g = self._graph((2, 3))
            for bad in (0.0, -0.5):
                with self.assertRaises(ValueError):
                    _run(
                        g,
                        {
                            "x": np.ones((2, 3), np.float32),
                            "s": np.full(3, bad, np.float32),
                        },
                    )

    def test_json_round_trip(self):
        rng = np.random.default_rng(94)
        g = self._graph((3, 2))
        feed = {
            "x": (rng.standard_normal((3, 2)) * 2).astype(np.float32),
            "s": np.array([0.125, 0.5], np.float32),
        }
        again = loads(dumps(g))
        np.testing.assert_array_equal(_run(g, feed)["y"], _run(again, feed)["y"])


class TestConformanceCases(unittest.TestCase):
    def test_reference_backend_passes_new_cases(self):
        names = {
            "quantize_i8_channel",
            "dequantize_i8_channel",
            "fake_quantize_i8_channel",
            "fake_quantize_i8_lsq_channel",
            "linear_i8_bias",
            "linear_i8a8_bias",
        }
        cases = [c for c in default_cases() if c.name in names]
        self.assertEqual({c.name for c in cases}, names)
        report = run_conformance(BACKEND, cases=cases)
        self.assertTrue(report.ok, report.text())


if __name__ == "__main__":
    unittest.main()
