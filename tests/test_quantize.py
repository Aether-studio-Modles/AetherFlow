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
    f8e4m3,
    f8e5m2,
    i8,
    loads,
)
from aetherflow.conformance import default_cases, run_conformance
from aetherflow.ir.ops import get_op
from aetherflow.models.quantize import (
    QUANT_MAX,
    build_int8_group_linear,
    build_int8_linear,
    build_int8_linear_zp,
    build_int8a8_linear,
    build_int8a8_linear_dynamic,
    build_int8a8_linear_qat,
    build_int8a8_linear_zp,
    dequantize_weight,
    dequantize_weight_asymmetric,
    dequantize_weight_fp8,
    quantize_activation,
    quantize_activation_asymmetric,
    quantize_activation_fp8,
    quantize_weight,
    quantize_weight_asymmetric,
    quantize_weight_fp8,
    quantize_weight_int4_group,
    quantize_weight_fp8_group,
    quantize_weight_group,
)
from aetherflow.runtime.fp8 import FP8_MAX, pack_fp8, quantize_fp8, round_fp8, unpack_fp8

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

    def test_int8_codes_have_no_gradient(self):
        """i8 码是整数域，没有连续变量；scale 与 bias 才是可导的连续量。"""
        g = self._graph()
        for name in ("x_q", "w_q"):
            with self.subTest(wrt=name):
                with self.assertRaises(NoGradientError):
                    backward(g, wrt=[g.find(name)])

    def test_scale_gradients_match_hand_derivation(self):
        """两路 scale 的梯度按各自广播轴归约，与手写公式逐位对拍。"""
        rng = np.random.default_rng(72)
        feed = {
            "x_q": rng.integers(-127, 128, size=(2, 4), dtype=np.int8),
            "x_scale": np.array([[0.1], [0.2]], np.float32),
            "w_q": rng.integers(-127, 128, size=(4, 3), dtype=np.int8),
            "w_scale": np.array([0.1, 0.2, 0.3], np.float32),
            "b": np.zeros(3, np.float32),
        }
        dy = rng.standard_normal((2, 3)).astype(np.float32)
        g = self._graph()
        out = _run(
            backward(g, wrt=[g.find("x_scale"), g.find("w_scale")]),
            {**feed, "dy": dy},
        )
        xd = feed["x_q"].astype(np.float32) * feed["x_scale"]
        wd = feed["w_q"].astype(np.float32) * feed["w_scale"]
        dact = dy @ wd.T
        dweight = xd.T @ dy
        want_xs = (dact * feed["x_q"]).sum(axis=-1, keepdims=True)
        want_ws = (dweight * feed["w_q"]).sum(axis=0)
        np.testing.assert_allclose(out["dx_scale"], want_xs, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(out["dw_scale"], want_ws, rtol=1e-6, atol=1e-6)


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


def _quantize_zp_graph(shape):
    g = Graph("quantize_i8_zp")
    x = g.input("x", f32[shape])
    s = g.input("s", f32[(shape[-1],)])
    zp = g.input("zp", i8[(shape[-1],)])
    g.output(g.add_node("quantize_i8_zp", [x, s, zp]), "q")
    return g


def _dequantize_zp_graph(shape):
    g = Graph("dequantize_i8_zp")
    q = g.input("q", i8[shape])
    s = g.input("s", f32[(shape[-1],)])
    zp = g.input("zp", i8[(shape[-1],)])
    g.output(g.add_node("dequantize_i8_zp", [q, s, zp]), "w")
    return g


class TestAsymmetricQuantizeSemantics(unittest.TestCase):
    def test_zero_point_shifts_the_integer_lattice(self):
        x = np.array([[0.0, 0.0], [1.0, -1.0]], np.float32)
        s = np.ones(2, np.float32)
        zp = np.array([-128, 0], np.int8)
        q = _run(_quantize_zp_graph((2, 2)), {"x": x, "s": s, "zp": zp})["q"]
        np.testing.assert_array_equal(q, np.array([[-128, 0], [-127, -1]], np.int8))

    def test_zero_point_is_applied_after_rounding(self):
        """``0.5 / 1 = 0.5`` 先按 RNE 舍成 0，再加 1 得 1。

        先加零点再舍入会得到 ``rint(1.5) = 2``——两条路径在 .5 处差一格，
        这正是"零点是整数格点的平移"与"零点是一个浮点偏移"的区别。
        """
        q = _run(
            _quantize_zp_graph((1, 1)),
            {
                "x": np.array([[0.5]], np.float32),
                "s": np.ones(1, np.float32),
                "zp": np.array([1], np.int8),
            },
        )["q"]
        np.testing.assert_array_equal(q, np.array([[1]], np.int8))

    def test_uses_both_endpoints(self):
        """非对称的负端点必须真的用上 -128，而不是对称版的 -127。"""
        x = np.array([[-127.0, 127.0, -1.0, 1.0]], np.float32)
        zp = np.array([-1, 0, -128, 127], np.int8)
        q = _run(
            _quantize_zp_graph((1, 4)),
            {"x": x, "s": np.ones(4, np.float32), "zp": zp},
        )["q"]
        np.testing.assert_array_equal(q, np.array([[-128, 127, -128, 127]], np.int8))

    def test_dequantize_subtracts_in_int32(self):
        """``q - zp`` 在 int32 里做：``-128 - 127 = -255``，不回绕成 1。"""
        q = np.array([[-128, 127, -128, 127]], np.int8)
        zp = np.array([127, -128, 127, -128], np.int8)
        s = np.array([0.5, 0.5, 1.0, 1.0], np.float32)
        got = _run(_dequantize_zp_graph((1, 4)), {"q": q, "s": s, "zp": zp})["w"]
        np.testing.assert_array_equal(
            got, np.array([[-127.5, 127.5, -255.0, 255.0]], np.float32)
        )

    def test_non_positive_scale_is_rejected(self):
        for bad in (0.0, -0.5):
            with self.subTest(scale=bad):
                with self.assertRaises(ValueError):
                    _run(
                        _quantize_zp_graph((2, 3)),
                        {
                            "x": np.ones((2, 3), np.float32),
                            "s": np.full(3, bad, np.float32),
                            "zp": np.zeros(3, np.int8),
                        },
                    )

    def test_build_time_rejections(self):
        with self.subTest(case="zero_point_must_be_i8"):
            g = Graph("bad_zp")
            x = g.input("x", f32[2, 3])
            s = g.input("s", f32[3])
            zp = g.input("zp", f32[3])
            with self.assertRaises(TypeError):
                g.add_node("quantize_i8_zp", [x, s, zp])
        with self.subTest(case="zero_point_rank_exceeds_input"):
            g = Graph("bad_zp")
            x = g.input("x", f32[2, 3])
            s = g.input("s", f32[3])
            zp = g.input("zp", i8[1, 2, 3])
            with self.assertRaises(ValueError):
                g.add_node("quantize_i8_zp", [x, s, zp])
        with self.subTest(case="symbolic_broadcast_unrepresentable"):
            g = Graph("bad_zp")
            x = g.input("x", f32["B", 4])
            s = g.input("s", f32[4])
            zp = g.input("zp", i8["S"])
            with self.assertRaises(ValueError):
                g.add_node("quantize_i8_zp", [x, s, zp])
        with self.subTest(case="wrong_arity"):
            g = Graph("bad_zp")
            x = g.input("x", f32[2, 3])
            with self.assertRaises(ValueError):
                g.add_node("quantize_i8_zp", [x])

    def test_json_round_trip(self):
        rng = np.random.default_rng(100)
        g = _quantize_zp_graph((3, 2))
        feed = {
            "x": (rng.standard_normal((3, 2)) * 3).astype(np.float32),
            "s": np.array([0.25, 0.5], np.float32),
            "zp": np.array([-128, 7], np.int8),
        }
        again = loads(dumps(g))
        np.testing.assert_array_equal(_run(g, feed)["q"], _run(again, feed)["q"])


class TestFP8WriteSide(unittest.TestCase):
    """fp8 写侧：scale 的口径必须与 int8 逐通道 / 逐 token 一致。

    判据不是"看起来像 fp8"，而是三条可断言的事实：

    1. 量化产物**逐位**等于 ``round_fp8(x / scale)``——量化只有一份真值。
    2. 每个通道 / token 的 amax 被顶到该格式的最大有限值（scale 定义正确）。
    3. 反量化误差落在该格式半个 ulp 的量级内（e4m3 尾数 3 位，相对误差
       上限 1/16）。
    """

    def test_weight_scale_is_per_output_channel(self):
        w = np.array([[1.0, -2.0], [3.0, 4.0]], np.float32)
        q, scale = quantize_weight_fp8(w)
        np.testing.assert_array_equal(scale, np.array([3.0 / 448.0, 4.0 / 448.0], np.float32))
        np.testing.assert_array_equal(q, round_fp8(w / scale, "e4m3"))
        np.testing.assert_allclose(np.abs(q).max(axis=0), [448.0, 448.0], rtol=0)

    def test_round_trip_relative_error_within_half_ulp(self):
        rng = np.random.default_rng(7)
        w = (rng.standard_normal((16, 8)) * 2.0).astype(np.float32)
        q, scale = quantize_weight_fp8(w)
        back = dequantize_weight_fp8(q, scale)
        np.testing.assert_array_equal(back, q * scale)
        np.testing.assert_allclose(back, w, rtol=1.0 / 16.0, atol=1e-6)

    def test_per_tensor_scale_matches_hf_weight_scale(self):
        rng = np.random.default_rng(8)
        w = (rng.standard_normal((8, 5)) * 3.0).astype(np.float32)
        q, scale = quantize_weight_fp8(w, per_tensor=True)
        self.assertEqual(scale.shape, (1,))
        self.assertAlmostEqual(float(scale[0]), float(np.abs(w).max()) / 448.0, places=7)
        self.assertLessEqual(float(np.abs(q).max()), 448.0)

    def test_zero_channel_gets_unit_scale(self):
        w = np.zeros((3, 2), np.float32)
        w[:, 1] = np.array([1.0, -2.0, 0.5], np.float32)
        q, scale = quantize_weight_fp8(w)
        self.assertEqual(float(scale[0]), 1.0)
        np.testing.assert_array_equal(q[:, 0], np.zeros(3, np.float32))

    def test_e5m2_saturates_instead_of_inf(self):
        w = np.array([[1.0e9, -1.0e9]], np.float32)
        q, scale = quantize_weight_fp8(w, kind="e5m2")
        self.assertTrue(np.all(np.isfinite(q)))
        self.assertEqual(float(np.abs(q).max()), FP8_MAX["e5m2"])

    def test_activation_scale_is_per_token(self):
        rng = np.random.default_rng(9)
        x = (rng.standard_normal((2, 3, 4)) * 5.0).astype(np.float32)
        q, scale = quantize_activation_fp8(x)
        self.assertEqual(scale.shape, (2, 3, 1))
        np.testing.assert_array_equal(q, round_fp8(x / scale, "e4m3"))
        np.testing.assert_allclose(
            np.abs(q).max(axis=-1), np.full((2, 3), 448.0), rtol=0
        )

    def test_quantize_fp8_clips_before_rounding(self):
        x = np.array([1.0e9, -1.0e9, 0.0], np.float32)
        q = quantize_fp8(x, np.array(1.0, np.float32), "e4m3")
        np.testing.assert_array_equal(q, np.array([448.0, -448.0, 0.0], np.float32))

    def test_rejections(self):
        cases = [
            ("rank", lambda: quantize_weight_fp8(np.ones(3, np.float32))),
            ("dtype", lambda: quantize_weight_fp8(np.ones((2, 2), np.int32))),
            ("nan", lambda: quantize_weight_fp8(np.array([[np.nan]], np.float32))),
            ("kind", lambda: quantize_weight_fp8(np.ones((1, 1), np.float32), "e3m2")),
            (
                "activation_rank",
                lambda: quantize_activation_fp8(np.ones(3, np.float32)),
            ),
        ]
        for name, call in cases:
            with self.subTest(case=name):
                with self.assertRaises((ValueError, TypeError)):
                    call()

    def test_quantize_fp8_rejects_non_positive_scale(self):
        for bad in (0.0, -1.0, np.nan):
            with self.subTest(scale=bad):
                with self.assertRaises(ValueError):
                    quantize_fp8(
                        np.ones((2, 3), np.float32), np.array(bad, np.float32)
                    )


class TestAsymmetricHelpers(unittest.TestCase):
    def test_weight_params_map_channel_ends_to_both_endpoints(self):
        """手算一份小权重：``-2..2`` 与 ``0..4`` 两条通道都该用满 -128..127。

        这里不写死每个格子：float32 的 ``scale`` 存不下 4/255 的精确值，
        ``-2 / scale`` 会落在 -127.4999 而不是 -127.5，RNE 对"恰好在半格上"
        的取整方向因此依赖末位。钉住端点覆盖 + 半步误差界，语义不变而断言
        不会因浮点细节抖动。
        """
        w = np.array([[-2.0, 0.0], [0.0, 2.0], [2.0, 4.0]], np.float32)
        q, scale, zp = quantize_weight_asymmetric(w)
        self.assertEqual(q.dtype, np.int8)
        self.assertEqual(zp.dtype, np.int8)
        np.testing.assert_array_equal(zp, np.array([0, -128], np.int8))
        np.testing.assert_allclose(scale, np.array([4 / 255, 4 / 255], np.float32))
        # 通道 0（值域 -2..2，跨零）：两端各被覆盖到。
        self.assertLessEqual(int(q[0, 0]), -126)
        self.assertGreaterEqual(int(q[2, 0]), 126)
        # 通道 1（值域 0..4，扩展后仍从 0 起）：-128 对应 0，127 对应 4；
        # 中间的 2.0 恰好落在半格上（2/scale = 127.5），取整方向由 float32 的
        # 末位决定，因此只要求它落在 ±1 格内。
        self.assertEqual(int(q[0, 1]), -128)
        self.assertEqual(int(q[2, 1]), 127)
        self.assertLessEqual(abs(int(q[1, 1])), 1)
        back = dequantize_weight_asymmetric(q, scale, zp)
        err = np.max(np.abs(back - w), axis=0)
        self.assertTrue(np.all(err <= scale / 2 + 1e-6), (err, scale))

    def test_round_trip_error_is_bounded_by_half_a_step(self):
        rng = np.random.default_rng(101)
        # 全正、0 起点的数据是 ReLU 之后激活的典型形态。
        w = (rng.random((32, 8)) * 5).astype(np.float32)
        q, scale, zp = quantize_weight_asymmetric(w)
        back = dequantize_weight_asymmetric(q, scale, zp)
        err = np.max(np.abs(back - w), axis=0)
        self.assertTrue(np.all(err <= scale / 2 + 1e-6), (err, scale))

    def test_zero_channel_gets_unit_scale_and_zero_value(self):
        w = np.zeros((4, 3), np.float32)
        w[0, 1] = 2.0
        q, scale, zp = quantize_weight_asymmetric(w)
        self.assertEqual(float(scale[0]), 1.0)
        np.testing.assert_array_equal(q[:, 0], np.full(4, -128, np.int8))
        np.testing.assert_array_equal(zp[0], np.int8(-128))
        np.testing.assert_array_equal(
            dequantize_weight_asymmetric(q, scale, zp)[:, 0], np.zeros(4, np.float32)
        )

    def test_zero_point_always_lands_inside_i8(self):
        """把值域扩到含 0 之后，零点对任何有限数据都不会越界。

        这是 ``_asymmetric_params`` 的核心不变量：``zp = -128 - lo'/scale``，
        ``lo' = min(lo, 0) <= 0`` 且 ``scale >= (hi'-lo')/255``，于是
        ``-128 <= zp <= 127``。不扩到 0 时全正数据的 ``zp < -128`` 会被夹死，
        正端整片饱和——这个用例就是那条老 bug 的守门人。
        """
        rng = np.random.default_rng(104)
        cases = {
            "跨零": rng.standard_normal((16, 4)).astype(np.float32),
            "全正": (rng.random((16, 4)) * 5 + 1).astype(np.float32),
            "全负": (-rng.random((16, 4)) * 5 - 1).astype(np.float32),
            "常数正": np.full((4, 2), 3.0, np.float32),
            "常数负": np.full((4, 2), -3.0, np.float32),
        }
        for name, w in cases.items():
            with self.subTest(case=name):
                q, scale, zp = quantize_weight_asymmetric(w)
                self.assertTrue(np.all(zp >= -128), zp)
                self.assertTrue(np.all(zp <= 127), zp)
                back = dequantize_weight_asymmetric(q, scale, zp)
                err = np.max(np.abs(back - w), axis=0)
                self.assertTrue(np.all(err <= scale / 2 + 1e-6), (err, scale))

    def test_constant_channel_is_represented_exactly(self):
        """常数通道在扩到含 0 之后可以被精确表示，而不是被压成 0。"""
        w = np.full((3, 1), 3.0, np.float32)
        q, scale, zp = quantize_weight_asymmetric(w)
        self.assertEqual(int(zp[0]), -128)
        np.testing.assert_array_equal(q[:, 0], np.full(3, 127, np.int8))
        np.testing.assert_allclose(
            dequantize_weight_asymmetric(q, scale, zp)[:, 0], w[:, 0], rtol=0, atol=1e-6
        )

    def test_asymmetric_beats_symmetric_on_positive_activations(self):
        rng = np.random.default_rng(102)
        x = (rng.random((64, 16)) * 10).astype(np.float32)
        q_sym, s_sym = quantize_activation(x)
        err_sym = float(np.mean(np.abs(q_sym.astype(np.float32) * s_sym - x)))
        q_asym, s_asym, zp_asym = quantize_activation_asymmetric(x)
        err_asym = float(
            np.mean(np.abs(dequantize_weight_asymmetric(q_asym, s_asym, zp_asym) - x))
        )
        self.assertLess(err_asym, err_sym * 0.7, (err_asym, err_sym))

    def test_helper_rejections(self):
        with self.assertRaises(ValueError):
            quantize_weight_asymmetric(np.zeros((2, 2, 2), np.float32))
        with self.assertRaises(TypeError):
            quantize_weight_asymmetric(np.zeros((2, 2), np.int32))
        bad = np.zeros((2, 2), np.float32)
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            quantize_weight_asymmetric(bad)
        with self.assertRaises(ValueError):
            quantize_activation_asymmetric(np.zeros(4, np.float32))
        with self.assertRaises(ValueError):
            quantize_activation_asymmetric(np.full((2, 3), np.inf, np.float32))


class TestAsymmetricLinearGraphs(unittest.TestCase):
    def test_weight_only_matches_dequantize_then_matmul(self):
        rng = np.random.default_rng(103)
        k, n = 8, 5
        w = rng.standard_normal((k, n)).astype(np.float32)
        q, s, zp = quantize_weight_asymmetric(w)
        x = rng.standard_normal((2, k)).astype(np.float32)
        b = rng.standard_normal(n).astype(np.float32)

        g = Graph("int8_zp_linear")
        inputs, out = build_int8_linear_zp(g, k, n)
        g.output(out, "y")
        got = _run(
            g,
            {
                "x": x,
                "w_i8": q,
                "scale": s,
                "zero_point": zp,
                "bias": b,
            },
        )["y"]
        want = x @ dequantize_weight_asymmetric(q, s, zp) + b
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
        self.assertEqual(set(inputs), {"x", "w_i8", "scale", "zero_point", "bias"})

    def test_w8a8_matches_int32_expansion(self):
        rng = np.random.default_rng(104)
        k, n = 8, 4
        w = rng.standard_normal((k, n)).astype(np.float32)
        x = (rng.random((3, k)) * 6).astype(np.float32)  # 全正激活
        wq, ws, wzp = quantize_weight_asymmetric(w)
        xq, xs, xzp = quantize_activation_asymmetric(x)
        b = rng.standard_normal(n).astype(np.float32)

        g = Graph("w8a8_zp_linear")
        inputs, out = build_int8a8_linear_zp(g, k, n)
        g.output(out, "y")
        got = _run(
            g,
            {
                "x_q": xq,
                "x_scale": xs,
                "x_zero_point": xzp,
                "w_q": wq,
                "w_scale": ws,
                "w_zero_point": wzp,
                "bias": b,
            },
        )["y"]

        acc = (xq.astype(np.int32) - xzp.astype(np.int32)) @ (
            wq.astype(np.int32) - wzp.astype(np.int32)
        )
        want = acc.astype(np.float32) * xs * ws + b
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-5)
        self.assertEqual(
            set(inputs),
            {"x_q", "x_scale", "x_zero_point", "w_q", "w_scale", "w_zero_point", "bias"},
        )

    def test_zero_point_changes_the_result(self):
        """同一份 int8 数据、同一组 scale，只改零点：结果必须不同。"""
        wq = np.array([[1, -2], [3, 4]], np.int8)
        ws = np.ones(2, np.float32)
        b = np.zeros(2, np.float32)
        xq = np.array([[5, 6]], np.int8)
        xs = np.ones((1, 1), np.float32)

        def _out(wzp, xzp):
            g = Graph("zp_matters")
            _, out = build_int8a8_linear_zp(g, 2, 2)
            g.output(out, "y")
            return _run(
                g,
                {
                    "x_q": xq,
                    "x_scale": xs,
                    "x_zero_point": np.array([[xzp]], np.int8),
                    "w_q": wq,
                    "w_scale": ws,
                    "w_zero_point": np.array([wzp, 0], np.int8),
                    "bias": b,
                },
            )["y"]

        np.testing.assert_array_equal(_out(0, 0), np.array([[23, 14]], np.float32))
        np.testing.assert_array_equal(_out(-3, -1), np.array([[66, 16]], np.float32))

    def test_int32_accumulation_limit_is_rejected(self):
        g = Graph("too_wide")
        xq = g.input("x_q", i8[1, 33026])
        xs = g.input("x_scale", f32[1, 1])
        xzp = g.input("x_zp", i8[1, 1])
        wq = g.input("w_q", i8[33026, 1])
        ws = g.input("w_scale", f32[1])
        wzp = g.input("w_zp", i8[1])
        with self.assertRaises(ValueError):
            g.add_node("linear_i8a8_zp", [xq, xs, xzp, wq, ws, wzp])

    def test_build_time_rejections(self):
        with self.subTest(case="activation_must_be_i8"):
            g = Graph("bad_w8a8")
            xq = g.input("x_q", f32[2, 8])
            xs = g.input("x_scale", f32[2, 1])
            xzp = g.input("x_zp", i8[2, 1])
            wq = g.input("w_q", i8[8, 4])
            ws = g.input("w_scale", f32[4])
            wzp = g.input("w_zp", i8[4])
            with self.assertRaises(TypeError):
                g.add_node("linear_i8a8_zp", [xq, xs, xzp, wq, ws, wzp])
        with self.subTest(case="activation_scale_not_constant_along_k"):
            g = Graph("bad_w8a8")
            xq = g.input("x_q", i8[2, 8])
            xs = g.input("x_scale", f32[2, 8])
            xzp = g.input("x_zp", i8[2, 1])
            wq = g.input("w_q", i8[8, 4])
            ws = g.input("w_scale", f32[4])
            wzp = g.input("w_zp", i8[4])
            with self.assertRaises(ValueError):
                g.add_node("linear_i8a8_zp", [xq, xs, xzp, wq, ws, wzp])
        with self.subTest(case="weight_scale_must_be_per_channel"):
            g = Graph("bad_w8a8")
            xq = g.input("x_q", i8[2, 8])
            xs = g.input("x_scale", f32[2, 1])
            xzp = g.input("x_zp", i8[2, 1])
            wq = g.input("w_q", i8[8, 4])
            ws = g.input("w_scale", f32[1, 4])
            wzp = g.input("w_zp", i8[4])
            with self.assertRaises(ValueError):
                g.add_node("linear_i8a8_zp", [xq, xs, xzp, wq, ws, wzp])
        with self.subTest(case="weight_only_rejects_f16_input"):
            g = Graph("bad_w")
            x = g.input("x", f16[2, 8])
            wq = g.input("w_q", i8[8, 4])
            ws = g.input("w_scale", f32[4])
            wzp = g.input("w_zp", i8[4])
            with self.assertRaises(TypeError):
                g.add_node("linear_i8_zp", [x, wq, ws, wzp])

    def test_backward_uses_zero_point_aware_weight(self):
        g = Graph("zp_bwd")
        x = g.input("x", f32[2, 3])
        q = g.input("q", i8[3, 2])
        s = g.input("s", f32[2])
        zp = g.input("zp", i8[2])
        g.output(g.add_node("linear_i8_zp", [x, q, s, zp]), "y")

        feed = {
            "x": np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]], np.float32),
            "q": np.array([[1, -2], [3, 4], [-5, 6]], np.int8),
            "s": np.array([0.5, 0.25], np.float32),
            "zp": np.array([-128, 7], np.int8),
            "dy": np.array([[1.0, -1.0], [0.5, 2.0]], np.float32),
        }
        got = _run(backward(g, wrt=[g.find("x")]), feed)["dx"]
        weight = dequantize_weight_asymmetric(feed["q"], feed["s"], feed["zp"])
        np.testing.assert_allclose(got, feed["dy"] @ weight.T, rtol=0, atol=1e-6)
        for name in ("q", "s", "zp"):
            with self.subTest(frozen=name):
                with self.assertRaises(NoGradientError):
                    backward(g, wrt=[g.find(name)])

    def test_w8a8_backward_only_bias(self):
        g = Graph("zp_w8a8_bwd")
        xq = g.input("x_q", i8[2, 3])
        xs = g.input("x_scale", f32[2, 1])
        xzp = g.input("x_zp", i8[2, 1])
        wq = g.input("w_q", i8[3, 2])
        ws = g.input("w_scale", f32[2])
        wzp = g.input("w_zp", i8[2])
        b = g.input("bias", f32[2])
        g.output(g.add_node("linear_i8a8_zp", [xq, xs, xzp, wq, ws, wzp, b]), "y")

        feed = {
            "x_q": np.array([[1, -2, 3], [-4, 5, 6]], np.int8),
            "x_scale": np.array([[0.1], [0.2]], np.float32),
            "x_zp": np.array([[-128], [0]], np.int8),
            "w_q": np.array([[1, -2], [3, 4], [-5, 6]], np.int8),
            "w_scale": np.array([0.5, 0.25], np.float32),
            "w_zp": np.array([-128, 7], np.int8),
            "bias": np.zeros(2, np.float32),
            "dy": np.array([[1.0, 2.0], [3.0, 4.0]], np.float32),
        }
        got = _run(backward(g, wrt=[g.find("bias")]), feed)["dbias"]
        np.testing.assert_array_equal(got, np.array([4.0, 6.0], np.float32))
        with self.assertRaises(NoGradientError):
            backward(g, wrt=[g.find("x_q")])


def _group_graph(shape, groups, with_zp=False):
    name = "dequantize_i8_group_zp" if with_zp else "dequantize_i8_group"
    g = Graph(name)
    q = g.input("q", i8[shape])
    s = g.input("s", f32[(groups, shape[1])])
    inputs = [q, s]
    if with_zp:
        inputs.append(g.input("z", i8[(groups, shape[1])]))
    g.output(g.add_node(name, inputs), "w")
    return g


class TestGroupDequantize(unittest.TestCase):
    """逐组（per-group）weight-only 反量化。

    组大小由 ``K / K_g`` 推出（没有额外 attr）：``K_g = 1`` 退化成逐通道，
    ``K_g = K`` 退化成逐行。组数取中间值时必须和"按行读 scale"给出不同的
    数，否则这个测试就只是在测逐通道。
    """

    def test_matches_manual_group_expansion(self):
        q = np.array([[1, -2], [3, -4], [5, -6], [7, -8]], np.int8)
        scale = np.array([[0.5, 0.25], [1.5, 2.0]], np.float32)
        got = _run(_group_graph((4, 2), 2), {"q": q, "s": scale})["w"]
        want = q.astype(np.float32) * np.repeat(scale, 2, axis=0)
        np.testing.assert_array_equal(got, want)
        # 第二组的第一行用的是 scale[1] 而不是 scale[0]：按行读会得到 5.0
        self.assertEqual(float(got[2, 0]), 7.5)

    def test_zero_point_is_applied_per_group_and_in_int32(self):
        q = np.full((4, 2), -128, np.int8)
        scale = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
        zp = np.array([[0, 1], [127, -128]], np.int8)
        got = _run(
            _group_graph((4, 2), 2, with_zp=True),
            {"q": q, "s": scale, "z": zp},
        )["w"]
        diff = q.astype(np.int32) - np.repeat(zp.astype(np.int32), 2, axis=0)
        np.testing.assert_array_equal(
            got, diff.astype(np.float32) * np.repeat(scale, 2, axis=0)
        )
        # (-128 - 127) * 3 = -765：先减再乘，减法不会在 int8 里回绕成 1
        self.assertEqual(float(got[2, 0]), -765.0)

    def test_group_of_one_matches_per_channel_dequantize(self):
        rng = np.random.default_rng(7)
        q = rng.integers(-127, 128, size=(4, 6), dtype=np.int8)
        scale = (np.abs(rng.standard_normal((1, 6))) * 0.1 + 0.01).astype(np.float32)
        g = Graph("per_channel")
        qq = g.input("q", i8[4, 6])
        ss = g.input("s", f32[(6,)])
        g.output(g.add_node("dequantize_i8", [qq, ss]), "w")
        want = _run(g, {"q": q, "s": scale[0]})["w"]
        got = _run(_group_graph((4, 6), 1), {"q": q, "s": scale})["w"]
        np.testing.assert_array_equal(got, want)

    def test_group_of_k_matches_per_row_scale(self):
        rng = np.random.default_rng(8)
        q = rng.integers(-127, 128, size=(3, 4), dtype=np.int8)
        scale = (np.abs(rng.standard_normal((3, 4))) * 0.1 + 0.01).astype(np.float32)
        got = _run(_group_graph((3, 4), 3), {"q": q, "s": scale})["w"]
        np.testing.assert_array_equal(got, q.astype(np.float32) * scale)

    def test_rejections(self):
        with self.subTest(case="k_not_divisible_by_groups"):
            g = Graph("bad_group")
            q = g.input("q", i8[5, 4])
            s = g.input("s", f32[2, 4])
            with self.assertRaises(ValueError):
                g.add_node("dequantize_i8_group", [q, s])
        with self.subTest(case="column_mismatch"):
            g = Graph("bad_group")
            q = g.input("q", i8[4, 4])
            s = g.input("s", f32[2, 3])
            with self.assertRaises(ValueError):
                g.add_node("dequantize_i8_group", [q, s])
        with self.subTest(case="scale_rank_three"):
            g = Graph("bad_group")
            q = g.input("q", i8[4, 4])
            s = g.input("s", f32[2, 2, 2])
            with self.assertRaises(ValueError):
                g.add_node("dequantize_i8_group", [q, s])
        with self.subTest(case="symbolic_k"):
            g = Graph("bad_group")
            q = g.input("q", i8["K", 4])
            s = g.input("s", f32[2, 4])
            with self.assertRaises(ValueError):
                g.add_node("dequantize_i8_group", [q, s])
        with self.subTest(case="zero_point_shape_mismatch"):
            g = Graph("bad_group_zp")
            q = g.input("q", i8[4, 4])
            s = g.input("s", f32[2, 4])
            z = g.input("z", i8[(4,)])
            with self.assertRaises(ValueError):
                g.add_node("dequantize_i8_group_zp", [q, s, z])
        with self.subTest(case="wrong_dtype"):
            g = Graph("bad_group")
            q = g.input("q", f32[4, 4])
            s = g.input("s", f32[2, 4])
            with self.assertRaises(TypeError):
                g.add_node("dequantize_i8_group", [q, s])
        with self.subTest(case="wrong_arity"):
            g = Graph("bad_group")
            q = g.input("q", i8[4, 4])
            with self.assertRaises(ValueError):
                g.add_node("dequantize_i8_group", [q])

    def test_json_round_trip(self):
        rng = np.random.default_rng(9)
        g = _group_graph((6, 4), 3, with_zp=True)
        feed = {
            "q": rng.integers(-128, 128, size=(6, 4), dtype=np.int8),
            "s": (np.abs(rng.standard_normal((3, 4))) * 0.2 + 0.01).astype(np.float32),
            "z": rng.integers(-128, 128, size=(3, 4), dtype=np.int8),
        }
        again = loads(dumps(g))
        np.testing.assert_array_equal(_run(g, feed)["w"], _run(again, feed)["w"])


class TestLinearI8Group(unittest.TestCase):
    """逐组 weight-only 线性层（5.329）。

    与 ``TestGroupDequantize`` 的分工：那一组只钉"逐组反量化"这一格，
    这里钉整条路径——组内共用 scale 的权重进入 GEMM 后，数值必须等于
    ``matmul(x, dequantize_i8_group(q, s)) + bias``，按逐通道解释会把
    第二组及之后的贡献算错。
    """

    def _graph(self, x_shape=(2, 8), w_shape=(8, 6), groups=4, bias=True):
        g = Graph("linear_i8_group")
        x = g.input("x", f32[x_shape])
        q = g.input("q", i8[w_shape])
        s = g.input("s", f32[(groups, w_shape[-1])])
        args = [x, q, s]
        if bias:
            args.append(g.input("b", f32[(w_shape[-1],)]))
        g.output(g.add_node("linear_i8_group", args), "y")
        return g

    def test_matches_group_dequantize_then_matmul(self):
        rng = np.random.default_rng(51)
        x = rng.standard_normal((2, 8)).astype(np.float32)
        q = rng.integers(-127, 128, size=(8, 6), dtype=np.int8)
        s = (np.abs(rng.standard_normal((4, 6))) * 0.2 + 0.05).astype(np.float32)
        b = rng.standard_normal(6).astype(np.float32)
        got = _run(self._graph(), {"x": x, "q": q, "s": s, "b": b})["y"]
        weight = q.astype(np.float32) * np.repeat(s, 2, axis=0)
        np.testing.assert_allclose(
            got, np.matmul(x, weight) + b, rtol=1e-6, atol=1e-6
        )

    def test_group_of_one_matches_per_channel_linear(self):
        rng = np.random.default_rng(52)
        x = rng.standard_normal((2, 8)).astype(np.float32)
        q = rng.integers(-127, 128, size=(8, 6), dtype=np.int8)
        s = (np.abs(rng.standard_normal((1, 6))) * 0.1 + 0.01).astype(np.float32)
        b = rng.standard_normal(6).astype(np.float32)
        got = _run(self._graph(groups=1), {"x": x, "q": q, "s": s, "b": b})["y"]
        np.testing.assert_array_equal(
            got, np.matmul(x, q.astype(np.float32) * s[0]) + b
        )

    def test_rejects_bad_shapes(self):
        with self.subTest(case="k_not_divisible_by_groups"):
            g = Graph("bad_group_linear")
            x = g.input("x", f32[2, 5])
            q = g.input("q", i8[5, 4])
            s = g.input("s", f32[2, 4])
            with self.assertRaises(ValueError):
                g.add_node("linear_i8_group", [x, q, s])
        with self.subTest(case="bias_length"):
            g = Graph("bad_group_linear")
            x = g.input("x", f32[2, 4])
            q = g.input("q", i8[4, 3])
            s = g.input("s", f32[2, 3])
            b = g.input("b", f32[4])
            with self.assertRaises(ValueError):
                g.add_node("linear_i8_group", [x, q, s, b])
        with self.subTest(case="x_not_f32"):
            g = Graph("bad_group_linear")
            x = g.input("x", f16[2, 4])
            q = g.input("q", i8[4, 3])
            s = g.input("s", f32[2, 3])
            with self.assertRaises(TypeError):
                g.add_node("linear_i8_group", [x, q, s])
        with self.subTest(case="weight_not_i8"):
            g = Graph("bad_group_linear")
            x = g.input("x", f32[2, 4])
            q = g.input("q", f32[4, 3])
            s = g.input("s", f32[2, 3])
            with self.assertRaises(TypeError):
                g.add_node("linear_i8_group", [x, q, s])

    def test_backward_matches_hand_derivation(self):
        rng = np.random.default_rng(53)
        g = self._graph(bias=True)
        feed = {
            "x": rng.standard_normal((2, 8)).astype(np.float32),
            "q": rng.integers(-127, 128, size=(8, 6), dtype=np.int8),
            "s": (np.abs(rng.standard_normal((4, 6))) * 0.2 + 0.05).astype(np.float32),
            "b": np.zeros(6, np.float32),
        }
        dy = rng.standard_normal((2, 6)).astype(np.float32)
        out = _run(
            backward(g, wrt=[g.find("x"), g.find("b")]), {**feed, "dy": dy}
        )
        weight = feed["q"].astype(np.float32) * np.repeat(feed["s"], 2, axis=0)
        np.testing.assert_allclose(out["dx"], dy @ weight.T, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(out["db"], dy.sum(axis=0), rtol=1e-6, atol=1e-6)

    def test_graph_survives_json_roundtrip(self):
        rng = np.random.default_rng(54)
        feed = {
            "x": rng.standard_normal((2, 8)).astype(np.float32),
            "q": rng.integers(-127, 128, size=(8, 6), dtype=np.int8),
            "s": (np.abs(rng.standard_normal((4, 6))) * 0.2 + 0.05).astype(np.float32),
            "b": np.zeros(6, np.float32),
        }
        g = self._graph()
        np.testing.assert_array_equal(
            _run(g, feed)["y"], _run(loads(dumps(g)), feed)["y"]
        )


class TestQuantizeWeightGroup(unittest.TestCase):
    def test_scale_shape_and_per_group_range(self):
        rng = np.random.default_rng(61)
        w = (rng.standard_normal((8, 3)) * 2).astype(np.float32)
        q, s = quantize_weight_group(w, 4)
        self.assertEqual(q.dtype, np.int8)
        self.assertEqual(q.shape, w.shape)
        self.assertEqual(s.shape, (2, 3))
        for gi in range(2):
            chunk = np.abs(q[gi * 4 : (gi + 1) * 4])
            np.testing.assert_array_equal(chunk.max(axis=0), np.full(3, QUANT_MAX, np.int8))

    def test_error_is_bounded_by_the_group_scale(self):
        rng = np.random.default_rng(62)
        w = (rng.standard_normal((8, 5)) * 3).astype(np.float32)
        q, s = quantize_weight_group(w, 2)
        back = q.astype(np.float32) * np.repeat(s, 2, axis=0)
        err = np.max(np.abs(back - w), axis=0)
        self.assertTrue(np.all(err <= s.max(axis=0) / 2 + 1e-5), (err, s))

    def test_grouping_beats_per_channel_when_rows_differ_in_scale(self):
        """一行放大 2000 倍：逐通道 scale 会把其余行整片舍成 0。"""
        w = np.full((8, 1), 0.01, np.float32)
        w[0, 0] = 20.0
        q_channel, _ = quantize_weight(w)
        q_group, _ = quantize_weight_group(w, 4)
        self.assertEqual(int(np.count_nonzero(q_channel[1:, 0])), 0)
        # 后四行自成一组（scale = 0.01/127），0.01 恰好落在正端点上。
        self.assertEqual(int(np.count_nonzero(q_group[1:, 0])), 4)

    def test_zero_group_gets_unit_scale(self):
        w = np.zeros((4, 2), np.float32)
        w[2, 0] = 1.0
        q, s = quantize_weight_group(w, 2)
        np.testing.assert_array_equal(q[:2], np.zeros((2, 2), np.int8))
        np.testing.assert_array_equal(s[0], np.ones(2, np.float32))
        self.assertGreater(float(s[1, 0]), 0.0)

    def test_helper_rejections(self):
        with self.assertRaises(ValueError):
            quantize_weight_group(np.zeros((2, 2, 2), np.float32), 1)
        with self.assertRaises(TypeError):
            quantize_weight_group(np.zeros((4, 2), np.int32), 2)
        with self.assertRaises(ValueError):
            quantize_weight_group(np.array([[1.0, np.nan]], np.float32), 1)
        with self.assertRaises(ValueError):
            quantize_weight_group(np.zeros((5, 2), np.float32), 2)
        with self.assertRaises(ValueError):
            quantize_weight_group(np.zeros((4, 2), np.float32), 0)


class TestGroupLinearBuilder(unittest.TestCase):
    def test_builder_matches_quantized_weight(self):
        rng = np.random.default_rng(71)
        w = (rng.standard_normal((8, 4)) * 1.5).astype(np.float32)
        q, s = quantize_weight_group(w, 2)
        g = Graph("group_builder")
        _, out = build_int8_group_linear(g, 8, 4, 2, batch=3)
        g.output(out, "y")
        x = rng.standard_normal((3, 8)).astype(np.float32)
        b = rng.standard_normal(4).astype(np.float32)
        got = _run(g, {"x": x, "w_i8": q, "scale": s, "bias": b})["y"]
        want = np.matmul(x, q.astype(np.float32) * np.repeat(s, 2, axis=0)) + b
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)

    def test_builder_rejects_indivisible_group(self):
        with self.assertRaises(ValueError):
            build_int8_group_linear(Graph("bad"), 8, 4, 3)
        with self.assertRaises(ValueError):
            build_int8_group_linear(Graph("bad"), 8, 4, 0)
        with self.assertRaises(ValueError):
            build_int8_group_linear(Graph("bad"), 0, 4, 1)


class TestW8A8QATBuilder(unittest.TestCase):
    """QAT 构建器：前向落在 int8 格点上，反向把梯度送回浮点参数。"""

    def test_forward_uses_the_same_lattice_as_deployment(self):
        rng = np.random.default_rng(81)
        x = rng.standard_normal((2, 8)).astype(np.float32)
        w = (rng.standard_normal((8, 4)) * 0.5).astype(np.float32)
        xq, xs = quantize_activation(x)
        wq, ws = quantize_weight(w)
        b = rng.standard_normal(4).astype(np.float32)
        g = Graph("qat")
        _, out = build_int8a8_linear_qat(g, 8, 4)
        g.output(out, "y")
        got = _run(g, {"x": x, "x_scale": xs, "w": w, "w_scale": ws, "bias": b})["y"]
        np.testing.assert_array_equal(
            got, np.matmul(xq.astype(np.float32) * xs, wq.astype(np.float32) * ws) + b
        )
        self.assertEqual(xq.dtype, np.int8)

    def test_ste_path_has_no_scale_gradient(self):
        g = Graph("qat")
        inputs, out = build_int8a8_linear_qat(g, 4, 3)
        g.output(out, "y")
        with self.assertRaises(NoGradientError):
            backward(g, wrt=[inputs["x_scale"]])

    def test_learn_scale_path_has_scale_gradient(self):
        g = Graph("qat_lsq")
        inputs, out = build_int8a8_linear_qat(g, 4, 3, learn_scale=True)
        g.output(out, "y")
        gw = backward(g, wrt=[inputs["x_scale"], inputs["w_scale"]])
        rng = np.random.default_rng(82)
        out_vals = _run(
            gw,
            {
                "x": rng.standard_normal((2, 4)).astype(np.float32),
                "x_scale": np.full((2, 1), 0.1, np.float32),
                "w": rng.standard_normal((4, 3)).astype(np.float32),
                "w_scale": np.full(3, 0.1, np.float32),
                "bias": np.zeros(3, np.float32),
                "dy": rng.standard_normal((2, 3)).astype(np.float32),
            },
        )
        self.assertEqual(out_vals["dx_scale"].shape, (2, 1))
        self.assertEqual(out_vals["dw_scale"].shape, (3,))

    def test_builder_rejects_non_positive_features(self):
        with self.assertRaises(ValueError):
            build_int8a8_linear_qat(Graph("bad"), 0, 4)
        with self.assertRaises(ValueError):
            build_int8a8_linear_qat(Graph("bad"), 4, -1)


class TestDynamicW8A8Builder(unittest.TestCase):
    def test_matches_host_per_token_quantization(self):
        rng = np.random.default_rng(91)
        x = rng.standard_normal((3, 8)).astype(np.float32)
        x[1] = 0.0  # 全零 token 必须走 scale=1.0 分支
        w = (rng.standard_normal((8, 4)) * 0.5).astype(np.float32)
        wq, ws = quantize_weight(w)
        b = rng.standard_normal(4).astype(np.float32)
        g = Graph("dynamic")
        _, out = build_int8a8_linear_dynamic(g, 8, 4)
        g.output(out, "y")
        got = _run(g, {"x": x, "w_q": wq, "w_scale": ws, "bias": b})["y"]
        xq, xs = quantize_activation(x)
        want = get_op("linear_i8a8").reference([xq, xs, wq, ws, b], {})
        np.testing.assert_array_equal(got, want)

    def test_scale_is_computed_per_token_not_per_tensor(self):
        x = np.array([[1.0, 2.0], [100.0, 200.0]], np.float32)
        w = np.ones((2, 1), np.float32)
        wq, ws = quantize_weight(w)
        g = Graph("dynamic")
        _, out = build_int8a8_linear_dynamic(g, 2, 1, bias=False)
        g.output(out, "y")
        got = _run(g, {"x": x, "w_q": wq, "w_scale": ws})["y"]
        xq, xs = quantize_activation(x)
        want = get_op("linear_i8a8").reference([xq, xs, wq, ws], {})
        np.testing.assert_array_equal(got, want)
        # 第一行的 scale 是 2/127 而不是全局的 200/127：若 scale 是逐张量的，
        # 1.0 会被舍成 1 格（≈1.57），sum 就只有 ≈2.57 而不是 ≈3.01。
        self.assertGreater(float(got[0, 0]), 2.9)

    def test_without_bias_and_graph_survives_roundtrip(self):
        rng = np.random.default_rng(92)
        w = (rng.standard_normal((4, 3)) * 0.5).astype(np.float32)
        wq, ws = quantize_weight(w)
        feed = {"x": rng.standard_normal((2, 4)).astype(np.float32), "w_q": wq, "w_scale": ws}
        g = Graph("dynamic")
        _, out = build_int8a8_linear_dynamic(g, 4, 3, bias=False)
        g.output(out, "y")
        np.testing.assert_array_equal(
            _run(g, feed)["y"], _run(loads(dumps(g)), feed)["y"]
        )

    def test_builder_rejects_non_positive_features(self):
        with self.assertRaises(ValueError):
            build_int8a8_linear_dynamic(Graph("bad"), 0, 4)
        with self.assertRaises(ValueError):
            build_int8a8_linear_dynamic(Graph("bad"), 4, 0)


class TestLinearFP8(unittest.TestCase):
    """fp8 线性层：载荷是 f32 载体，语义是"按 scale 反量化一次再做 f32 GEMM"。

    这一组钉住三件事：

    1. 数值等于手写的 ``q * scale`` 再 matmul，而不是把 f32 载体当普通权重；
    2. 两种格式按各自的格点解释（e5m2 的格点比 e4m3 粗）；
    3. 逐组 scale、未量化权重、错误的秩 / dtype / bias 都在构图期被拒绝。
    """

    def _weight_graph(self, x_shape, w_shape, bias=True, kind="e4m3", scale_shape=None):
        dtype = f8e4m3 if kind == "e4m3" else f8e5m2
        g = Graph("linear_fp8")
        x = g.input("x", f32[x_shape])
        q = g.input("q", dtype[w_shape])
        s = g.input(
            "s", f32[scale_shape if scale_shape is not None else (w_shape[-1],)]
        )
        args = [x, q, s]
        if bias:
            args.append(g.input("b", f32[(w_shape[-1],)]))
        g.output(g.add_node("linear_fp8", args), "y")
        return g

    def _a8_graph(self, x_shape, w_shape, bias=True, kind="e4m3"):
        dtype = f8e4m3 if kind == "e4m3" else f8e5m2
        g = Graph("linear_fp8a8")
        q = g.input("x_q", dtype[x_shape])
        sx = g.input("x_scale", f32[x_shape[:-1] + (1,)])
        wq = g.input("w_q", dtype[w_shape])
        sw = g.input("w_scale", f32[(w_shape[-1],)])
        args = [q, sx, wq, sw]
        if bias:
            args.append(g.input("b", f32[(w_shape[-1],)]))
        g.output(g.add_node("linear_fp8a8", args), "y")
        return g

    def test_matches_manual_dequantize_then_matmul(self):
        rng = np.random.default_rng(31)
        x = rng.standard_normal((2, 4, 8)).astype(np.float32)
        scale = rng.random(6).astype(np.float32) + 0.1
        q = quantize_fp8(rng.standard_normal((8, 6)).astype(np.float32) * 3.0, scale)
        b = rng.standard_normal(6).astype(np.float32)
        got = _run(self._weight_graph((2, 4, 8), (8, 6)), {"x": x, "q": q, "s": scale, "b": b})[
            "y"
        ]
        np.testing.assert_allclose(
            got, np.matmul(x, q * scale) + b, rtol=1e-6, atol=1e-6
        )

    def test_per_tensor_scale_without_bias(self):
        rng = np.random.default_rng(32)
        x = rng.standard_normal((3, 5)).astype(np.float32)
        scale = np.array([0.5], np.float32)
        q = quantize_fp8(rng.standard_normal((5, 4)).astype(np.float32) * 2.0, scale)
        got = _run(
            self._weight_graph((3, 5), (5, 4), bias=False, scale_shape=(1,)),
            {"x": x, "q": q, "s": scale},
        )["y"]
        np.testing.assert_allclose(
            got, np.matmul(x, q * np.float32(0.5)), rtol=1e-6, atol=1e-6
        )

    def test_e5m2_rounds_at_its_own_grid(self):
        """0.55 在两种格式下落在不同格点：e4m3 是 0.5625，e5m2 是 0.5。

        e4m3 的格点更密，所以"e5m2 能表示的数"在 e4m3 里也都能表示；要区分
        后端有没有按声明的格式舍入，只能看一个非格点输入被舍到哪一格。如果
        执行器把格式硬编码成 e4m3，这条会以 0.5625 失败。
        """
        self.assertEqual(float(round_fp8(np.array([0.55], np.float32), "e4m3")[0]), 0.5625)
        self.assertEqual(float(round_fp8(np.array([0.55], np.float32), "e5m2")[0]), 0.5)
        got = _run(
            self._weight_graph((1, 1), (1, 1), bias=False, kind="e5m2"),
            {
                "x": np.ones((1, 1), np.float32),
                "q": np.array([[0.55]], np.float32),
                "s": np.array([1.0], np.float32),
            },
        )["y"]
        self.assertEqual(float(got[0, 0]), 0.5)

    def test_fp8a8_matches_manual(self):
        rng = np.random.default_rng(33)
        xq, xs = quantize_activation_fp8(rng.standard_normal((2, 3, 8)).astype(np.float32))
        wq, ws = quantize_weight_fp8(rng.standard_normal((8, 6)).astype(np.float32))
        b = rng.standard_normal(6).astype(np.float32)
        got = _run(
            self._a8_graph((2, 3, 8), (8, 6)),
            {"x_q": xq, "x_scale": xs, "w_q": wq, "w_scale": ws, "b": b},
        )["y"]
        np.testing.assert_allclose(
            got, np.matmul(xq * xs, wq * ws) + b, rtol=1e-6, atol=1e-6
        )

    def test_fp8a8_rank2_with_per_token_scale(self):
        rng = np.random.default_rng(34)
        xq, _ = quantize_activation_fp8(rng.standard_normal((4, 8)).astype(np.float32))
        xs = (np.abs(rng.standard_normal((4, 1))) * 0.1 + 0.05).astype(np.float32)
        wq, ws = quantize_weight_fp8(rng.standard_normal((8, 3)).astype(np.float32))
        got = _run(
            self._a8_graph((4, 8), (8, 3), bias=False),
            {"x_q": xq, "x_scale": xs, "w_q": wq, "w_scale": ws},
        )["y"]
        np.testing.assert_allclose(
            got, np.matmul(xq * xs, wq * ws), rtol=1e-6, atol=1e-6
        )

    def test_scale_must_be_strictly_positive(self):
        x = np.ones((2, 4), np.float32)
        q = quantize_fp8(np.ones((4, 3), np.float32), np.ones(3, np.float32))
        for bad in (0.0, -0.5):
            with self.subTest(scale=bad):
                with self.assertRaises(ValueError):
                    _run(
                        self._weight_graph((2, 4), (4, 3), bias=False),
                        {"x": x, "q": q, "s": np.full(3, bad, np.float32)},
                    )

    def test_rejects_bad_dtypes_ranks_and_scales(self):
        with self.subTest(case="x_not_f32"):
            g = Graph("linear_fp8")
            x = g.input("x", f16[2, 4])
            q = g.input("q", f8e4m3[4, 3])
            s = g.input("s", f32[3])
            with self.assertRaises(TypeError):
                g.add_node("linear_fp8", [x, q, s])
        with self.subTest(case="weight_not_fp8"):
            g = Graph("linear_fp8")
            x = g.input("x", f32[2, 4])
            q = g.input("q", f32[4, 3])
            s = g.input("s", f32[3])
            with self.assertRaises(TypeError):
                g.add_node("linear_fp8", [x, q, s])
        with self.subTest(case="x_rank1"):
            g = Graph("linear_fp8")
            x = g.input("x", f32[4])
            q = g.input("q", f8e4m3[4, 3])
            s = g.input("s", f32[3])
            with self.assertRaises(ValueError):
                g.add_node("linear_fp8", [x, q, s])
        with self.subTest(case="group_scale_is_not_broadcastable"):
            # (K/G, N) 逐组 scale 要先把权重展开成 (K, N) 才有意义，
            # 这一版不接受；逐组路径留给 linear_i8_group 那一批（5.329）。
            g = Graph("linear_fp8")
            x = g.input("x", f32[2, 4])
            q = g.input("q", f8e4m3[4, 3])
            s = g.input("s", f32[2, 3])
            with self.assertRaises(ValueError):
                g.add_node("linear_fp8", [x, q, s])
        with self.subTest(case="bias_length"):
            g = Graph("linear_fp8")
            x = g.input("x", f32[2, 4])
            q = g.input("q", f8e4m3[4, 3])
            s = g.input("s", f32[3])
            b = g.input("b", f32[4])
            with self.assertRaises(ValueError):
                g.add_node("linear_fp8", [x, q, s, b])
        with self.subTest(case="mixed_formats"):
            g = Graph("linear_fp8a8")
            q = g.input("x_q", f8e4m3[2, 4])
            sx = g.input("x_scale", f32[2, 1])
            wq = g.input("w_q", f8e5m2[4, 3])
            sw = g.input("w_scale", f32[3])
            with self.assertRaises(TypeError):
                g.add_node("linear_fp8a8", [q, sx, wq, sw])
        with self.subTest(case="activation_scale_along_k"):
            g = Graph("linear_fp8a8")
            q = g.input("x_q", f8e4m3[2, 4])
            sx = g.input("x_scale", f32[2, 4])
            wq = g.input("w_q", f8e4m3[4, 3])
            sw = g.input("w_scale", f32[3])
            with self.assertRaises(ValueError):
                g.add_node("linear_fp8a8", [q, sx, wq, sw])

    def test_graph_survives_json_roundtrip(self):
        rng = np.random.default_rng(35)
        scale = np.array([0.5, 0.25, 0.125], np.float32)
        feed = {
            "x": rng.standard_normal((2, 8)).astype(np.float32),
            "q": quantize_fp8(rng.standard_normal((8, 3)).astype(np.float32), scale),
            "s": scale,
            "b": np.zeros(3, np.float32),
        }
        g = self._weight_graph((2, 8), (8, 3))
        again = loads(dumps(g))
        np.testing.assert_array_equal(_run(g, feed)["y"], _run(again, feed)["y"])


class TestLinearFP8Packed(unittest.TestCase):
    """5.328：fp8 的 1 字节存储（``pack_fp8`` / ``unpack_fp8`` / packed 线性层）。

    钉住三件事：

    1. 位模式与主机侧 ``pack_fp8`` / ``unpack_fp8`` 逐位一致（int8 载荷是
       字节，不是整数，差 1 就是格式错了）；
    2. packed 线性层与 ``matmul(x, unpack(q) * scale) + bias`` 数值一致；
    3. 设备端用不了的形态（不是 i8 位模式的权重、秩 > 1 的 scale、错 dtype）
       在构图期就拒绝。
    """

    def _pack_graph(self, n, kind="e4m3"):
        dtype = f8e4m3 if kind == "e4m3" else f8e5m2
        g = Graph("pack_fp8")
        x = g.input("x", dtype[(n,)])
        g.output(g.add_node("pack_fp8", [x], {"kind": kind}), "q")
        return g

    def _unpack_graph(self, n, kind="e4m3"):
        g = Graph("unpack_fp8")
        q = g.input("q", i8[(n,)])
        g.output(g.add_node("unpack_fp8", [q], {"kind": kind}), "y")
        return g

    def _packed_graph(
        self, x_shape, w_shape, bias=True, kind="e4m3", scale_shape=None
    ):
        g = Graph("linear_fp8_packed")
        x = g.input("x", f32[x_shape])
        q = g.input("q", i8[w_shape])
        s = g.input(
            "s", f32[scale_shape if scale_shape is not None else (w_shape[-1],)]
        )
        args = [x, q, s]
        if bias:
            args.append(g.input("b", f32[(w_shape[-1],)]))
        g.output(g.add_node("linear_fp8_packed", args, {"kind": kind}), "y")
        return g

    def test_pack_matches_host_bits(self):
        values = np.array(
            [0.0, -0.0, 1.0, -1.5, 0.3125, 1e-7, 448.0, 512.0, 1e30, 0.001953125],
            np.float32,
        )
        for kind in ("e4m3", "e5m2"):
            with self.subTest(kind=kind):
                got = _run(self._pack_graph(values.size, kind), {"x": values})["q"]
                self.assertEqual(got.dtype, np.int8)
                np.testing.assert_array_equal(got, pack_fp8(values, kind).view(np.int8))

    def test_unpack_matches_host_bits(self):
        codes = np.array(
            [0x00, 0x80, 0x01, 0x08, 0x38, 0xB8, 0x78, 0x7E, 0xFE, 0x39], np.uint8
        ).view(np.int8)
        got = _run(self._unpack_graph(codes.size), {"q": codes})["y"]
        np.testing.assert_array_equal(
            got, unpack_fp8(codes.view(np.uint8), "e4m3")
        )

    def test_pack_unpack_roundtrip_equals_cast(self):
        """pack 再 unpack 必须等于 ``cast``：这两个方向共用同一份编解码。"""
        x = np.array(
            [0.0, -0.0, 0.55, -1.234, 1e-7, 448.0, 512.0, -0.001953125], np.float32
        )
        g = Graph("roundtrip")
        src = g.input("x", f8e4m3[(x.size,)])
        q = g.add_node("pack_fp8", [src], {"kind": "e4m3"})
        g.output(g.add_node("unpack_fp8", [q], {"kind": "e4m3"}), "y")
        got = _run(g, {"x": x})["y"]
        np.testing.assert_array_equal(got, round_fp8(x, "e4m3"))

    def test_packed_linear_matches_manual(self):
        rng = np.random.default_rng(41)
        x = rng.standard_normal((2, 4, 8)).astype(np.float32)
        scale = (rng.random(6).astype(np.float32) + 0.1)
        wq = quantize_fp8(rng.standard_normal((8, 6)).astype(np.float32) * 3.0, scale)
        packed = pack_fp8(wq, "e4m3").view(np.int8)
        b = rng.standard_normal(6).astype(np.float32)
        got = _run(
            self._packed_graph((2, 4, 8), (8, 6)),
            {"x": x, "q": packed, "s": scale, "b": b},
        )["y"]
        weight = unpack_fp8(packed.view(np.uint8), "e4m3") * scale
        np.testing.assert_allclose(
            got, np.matmul(x, weight) + b, rtol=1e-6, atol=1e-6
        )

    def test_packed_linear_per_tensor_scale_e5m2(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal((3, 5)).astype(np.float32)
        scale = np.array([0.5], np.float32)
        wq = quantize_fp8(
            rng.standard_normal((5, 4)).astype(np.float32) * 2.0, scale, "e5m2"
        )
        packed = pack_fp8(wq, "e5m2").view(np.int8)
        got = _run(
            self._packed_graph(
                (3, 5), (5, 4), bias=False, kind="e5m2", scale_shape=(1,)
            ),
            {"x": x, "q": packed, "s": scale},
        )["y"]
        weight = unpack_fp8(packed.view(np.uint8), "e5m2") * np.float32(0.5)
        np.testing.assert_allclose(got, np.matmul(x, weight), rtol=1e-6, atol=1e-6)

    def test_rejects_bad_shapes_and_dtypes(self):
        with self.subTest(case="pack_input_not_fp8"):
            g = Graph("pack_fp8")
            x = g.input("x", f32[4])
            with self.assertRaises(TypeError):
                g.add_node("pack_fp8", [x], {"kind": "e4m3"})
        with self.subTest(case="pack_kind_mismatch"):
            g = Graph("pack_fp8")
            x = g.input("x", f8e5m2[4])
            with self.assertRaises(TypeError):
                g.add_node("pack_fp8", [x], {"kind": "e4m3"})
        with self.subTest(case="pack_bad_kind"):
            g = Graph("pack_fp8")
            x = g.input("x", f8e4m3[4])
            with self.assertRaises(ValueError):
                g.add_node("pack_fp8", [x], {"kind": "e3m4"})
        with self.subTest(case="unpack_not_i8"):
            g = Graph("unpack_fp8")
            q = g.input("q", f32[4])
            with self.assertRaises(TypeError):
                g.add_node("unpack_fp8", [q], {"kind": "e4m3"})
        with self.subTest(case="packed_weight_not_i8"):
            g = Graph("linear_fp8_packed")
            x = g.input("x", f32[2, 4])
            q = g.input("q", f8e4m3[4, 3])
            s = g.input("s", f32[3])
            with self.assertRaises(TypeError):
                g.add_node("linear_fp8_packed", [x, q, s], {"kind": "e4m3"})
        with self.subTest(case="packed_x_not_f32"):
            g = Graph("linear_fp8_packed")
            x = g.input("x", f16[2, 4])
            q = g.input("q", i8[4, 3])
            s = g.input("s", f32[3])
            with self.assertRaises(TypeError):
                g.add_node("linear_fp8_packed", [x, q, s], {"kind": "e4m3"})
        with self.subTest(case="packed_group_scale"):
            g = Graph("linear_fp8_packed")
            x = g.input("x", f32[2, 4])
            q = g.input("q", i8[4, 3])
            s = g.input("s", f32[2, 3])
            with self.assertRaises(ValueError):
                g.add_node("linear_fp8_packed", [x, q, s])
        with self.subTest(case="packed_bias_length"):
            g = Graph("linear_fp8_packed")
            x = g.input("x", f32[2, 4])
            q = g.input("q", i8[4, 3])
            s = g.input("s", f32[3])
            b = g.input("b", f32[4])
            with self.assertRaises(ValueError):
                g.add_node("linear_fp8_packed", [x, q, s, b])

    def test_rejects_non_positive_scale(self):
        x = np.ones((2, 4), np.float32)
        packed = pack_fp8(np.ones((4, 3), np.float32), "e4m3").view(np.int8)
        for bad in (0.0, -0.5):
            with self.subTest(scale=bad):
                with self.assertRaises(ValueError):
                    _run(
                        self._packed_graph((2, 4), (4, 3), bias=False),
                        {"x": x, "q": packed, "s": np.full(3, bad, np.float32)},
                    )

    def test_graph_survives_json_roundtrip(self):
        rng = np.random.default_rng(43)
        scale = np.array([0.5, 0.25, 0.125], np.float32)
        packed = pack_fp8(
            rng.standard_normal((8, 3)).astype(np.float32), "e4m3"
        ).view(np.int8)
        feed = {
            "x": rng.standard_normal((2, 8)).astype(np.float32),
            "q": packed,
            "s": scale,
            "b": np.zeros(3, np.float32),
        }
        g = self._packed_graph((2, 8), (8, 3))
        again = loads(dumps(g))
        np.testing.assert_array_equal(_run(g, feed)["y"], _run(again, feed)["y"])


class TestFP8GroupAndPackedActivation(unittest.TestCase):
    """5.342：fp8 的逐组 scale 与 packed 激活。

    钉住三件事：

    1. 逐组反量化的 scale 只在组内生效：手推 ``unpack(q) * repeat(scale, G)``
       对拍，组边界不能串（第 i 组串用第 i+1 组的 scale 必须能看出来）；
    2. 两端都打包的 W8A8 与"先解码再 f32 矩阵乘"数值一致，激活与权重都
       按 1 字节码读出；
    3. 用不了的形态（不是 i8 位模式、沿 K 变化的激活 scale、秩 > 1 的权重
       scale、不整除 K 的组）在构图期被拒绝。
    """

    def _group_graph(self, shape, groups, kind="e4m3"):
        g = Graph("dequantize_fp8_group")
        q = g.input("q", i8[shape])
        s = g.input("s", f32[(groups, shape[-1])])
        g.output(g.add_node("dequantize_fp8_group", [q, s], {"kind": kind}), "y")
        return g

    def _linear_group_graph(self, x_shape, w_shape, groups, kind="e4m3", bias=True):
        g = Graph("linear_fp8_group")
        x = g.input("x", f32[x_shape])
        q = g.input("q", i8[w_shape])
        s = g.input("s", f32[(groups, w_shape[-1])])
        args = [x, q, s]
        if bias:
            args.append(g.input("b", f32[(w_shape[-1],)]))
        g.output(g.add_node("linear_fp8_group", args, {"kind": kind}), "y")
        return g

    def _packed_a8_graph(self, x_shape, w_shape, kind="e4m3", bias=True):
        g = Graph("linear_fp8a8_packed")
        q = g.input("x_q", i8[x_shape])
        sx = g.input("x_scale", f32[x_shape[:-1] + (1,)])
        wq = g.input("w_q", i8[w_shape])
        sw = g.input("w_scale", f32[(w_shape[-1],)])
        args = [q, sx, wq, sw]
        if bias:
            args.append(g.input("b", f32[(w_shape[-1],)]))
        g.output(g.add_node("linear_fp8a8_packed", args, {"kind": kind}), "y")
        return g

    def test_dequantize_group_uses_scale_per_group(self):
        rng = np.random.default_rng(342)
        w = rng.standard_normal((8, 3)).astype(np.float32)
        scale = np.array([[0.5, 0.25, 0.125], [0.75, 1.5, 0.0625]], np.float32)
        carrier = quantize_fp8(w, np.repeat(scale, 4, axis=0), "e4m3")
        codes = pack_fp8(carrier, "e4m3").view(np.int8)
        got = _run(self._group_graph((8, 3), 2), {"q": codes, "s": scale})["y"]
        decoded = unpack_fp8(codes.view(np.uint8), "e4m3")
        want = decoded * np.repeat(scale, 4, axis=0)
        np.testing.assert_array_equal(got, want)
        # 组边界：前 4 行只乘第 0 组，后 4 行只乘第 1 组；串组会在这里露出来。
        np.testing.assert_array_equal(got[:4], decoded[:4] * scale[0])
        np.testing.assert_array_equal(got[4:], decoded[4:] * scale[1])

    def test_dequantize_group_e5m2_and_uneven_groups(self):
        rng = np.random.default_rng(343)
        w = rng.standard_normal((12, 2)).astype(np.float32)
        scale = np.array([[0.5, 0.25], [1.0, 0.125]], np.float32)
        carrier = quantize_fp8(w, np.repeat(scale, 6, axis=0), "e5m2")
        codes = pack_fp8(carrier, "e5m2").view(np.int8)
        got = _run(
            self._group_graph((12, 2), 2, kind="e5m2"), {"q": codes, "s": scale}
        )["y"]
        np.testing.assert_array_equal(
            got,
            unpack_fp8(codes.view(np.uint8), "e5m2") * np.repeat(scale, 6, axis=0),
        )

    def test_linear_group_matches_dequantized_matmul(self):
        rng = np.random.default_rng(344)
        x_shape, w_shape, groups = (2, 3, 8), (8, 4), 4
        x = rng.standard_normal(x_shape).astype(np.float32)
        w = rng.standard_normal(w_shape).astype(np.float32)
        scale = (np.abs(rng.standard_normal((groups, 4))).astype(np.float32) * 0.5 + 0.125)
        b = rng.standard_normal(4).astype(np.float32)
        carrier = quantize_fp8(w, np.repeat(scale, 2, axis=0), "e4m3")
        codes = pack_fp8(carrier, "e4m3").view(np.int8)
        got = _run(
            self._linear_group_graph(x_shape, w_shape, groups),
            {"x": x, "q": codes, "s": scale, "b": b},
        )["y"]
        want = (
            np.matmul(x, unpack_fp8(codes.view(np.uint8), "e4m3") * np.repeat(scale, 2, axis=0))
            + b
        )
        np.testing.assert_array_equal(got, want)

    def test_packed_activation_matches_float_path(self):
        rng = np.random.default_rng(345)
        x_shape, w_shape = (2, 3, 8), (8, 6)
        x = rng.standard_normal(x_shape).astype(np.float32)
        w = rng.standard_normal(w_shape).astype(np.float32)
        xs = (np.abs(rng.standard_normal((2, 3, 1))).astype(np.float32) * 0.25 + 0.25)
        ws = np.array([0.5, 0.25, 0.125, 1.0, 0.75, 0.0625], np.float32)
        b = rng.standard_normal(6).astype(np.float32)
        xc = pack_fp8(quantize_fp8(x, xs, "e4m3"), "e4m3").view(np.int8)
        wc = pack_fp8(quantize_fp8(w, ws, "e4m3"), "e4m3").view(np.int8)
        got = _run(
            self._packed_a8_graph(x_shape, w_shape),
            {"x_q": xc, "x_scale": xs, "w_q": wc, "w_scale": ws, "b": b},
        )["y"]
        want = (
            np.matmul(
                unpack_fp8(xc.view(np.uint8), "e4m3") * xs,
                unpack_fp8(wc.view(np.uint8), "e4m3") * ws,
            )
            + b
        )
        np.testing.assert_array_equal(got, want)

    def test_packed_activation_scalar_scales(self):
        rng = np.random.default_rng(346)
        x_shape, w_shape = (4, 8), (8, 3)
        x = rng.standard_normal(x_shape).astype(np.float32)
        w = rng.standard_normal(w_shape).astype(np.float32)
        xc = pack_fp8(quantize_fp8(x, np.float32(0.5), "e5m2"), "e5m2").view(np.int8)
        wc = pack_fp8(quantize_fp8(w, np.float32(0.25), "e5m2"), "e5m2").view(np.int8)
        g = Graph("linear_fp8a8_packed")
        q = g.input("x_q", i8[x_shape])
        sx = g.input("x_scale", f32[()])
        wq = g.input("w_q", i8[w_shape])
        sw = g.input("w_scale", f32[()])
        g.output(
            g.add_node("linear_fp8a8_packed", [q, sx, wq, sw], {"kind": "e5m2"}), "y"
        )
        got = _run(
            g,
            {
                "x_q": xc,
                "x_scale": np.float32(0.5),
                "w_q": wc,
                "w_scale": np.float32(0.25),
            },
        )["y"]
        want = np.matmul(
            unpack_fp8(xc.view(np.uint8), "e5m2") * np.float32(0.5),
            unpack_fp8(wc.view(np.uint8), "e5m2") * np.float32(0.25),
        )
        np.testing.assert_array_equal(got, want)

    def test_group_rejects_bad_shapes(self):
        with self.subTest(case="scale_does_not_divide_k"):
            with self.assertRaises(ValueError):
                self._group_graph((8, 3), 3)
        with self.subTest(case="scale_rank_one"):
            g = Graph("dequantize_fp8_group")
            q = g.input("q", i8[8, 3])
            s = g.input("s", f32[3])
            with self.assertRaises(ValueError):
                g.add_node("dequantize_fp8_group", [q, s], {"kind": "e4m3"})
        with self.subTest(case="weight_not_i8"):
            g = Graph("dequantize_fp8_group")
            q = g.input("q", f8e4m3[8, 3])
            s = g.input("s", f32[2, 3])
            with self.assertRaises(TypeError):
                g.add_node("dequantize_fp8_group", [q, s], {"kind": "e4m3"})
        with self.subTest(case="linear_x_not_f32"):
            g = Graph("linear_fp8_group")
            x = g.input("x", f16[2, 8])
            q = g.input("q", i8[8, 3])
            s = g.input("s", f32[2, 3])
            with self.assertRaises(TypeError):
                g.add_node("linear_fp8_group", [x, q, s], {"kind": "e4m3"})
        with self.subTest(case="linear_weight_is_f8_carrier"):
            g = Graph("linear_fp8_group")
            x = g.input("x", f32[2, 8])
            q = g.input("q", f8e4m3[8, 3])
            s = g.input("s", f32[2, 3])
            with self.assertRaises(TypeError):
                g.add_node("linear_fp8_group", [x, q, s], {"kind": "e4m3"})
        with self.subTest(case="linear_bias_length"):
            g = Graph("linear_fp8_group")
            x = g.input("x", f32[2, 8])
            q = g.input("q", i8[8, 3])
            s = g.input("s", f32[2, 3])
            b = g.input("b", f32[4])
            with self.assertRaises(ValueError):
                g.add_node("linear_fp8_group", [x, q, s, b], {"kind": "e4m3"})

    def test_packed_activation_rejects_bad_shapes(self):
        with self.subTest(case="activation_is_f8_carrier"):
            g = Graph("linear_fp8a8_packed")
            q = g.input("x_q", f8e4m3[2, 8])
            sx = g.input("x_scale", f32[2, 1])
            wq = g.input("w_q", i8[8, 3])
            sw = g.input("w_scale", f32[3])
            with self.assertRaises(TypeError):
                g.add_node("linear_fp8a8_packed", [q, sx, wq, sw], {"kind": "e4m3"})
        with self.subTest(case="activation_scale_along_k"):
            g = Graph("linear_fp8a8_packed")
            q = g.input("x_q", i8[2, 8])
            sx = g.input("x_scale", f32[2, 8])
            wq = g.input("w_q", i8[8, 3])
            sw = g.input("w_scale", f32[3])
            with self.assertRaises(ValueError):
                g.add_node("linear_fp8a8_packed", [q, sx, wq, sw], {"kind": "e4m3"})
        with self.subTest(case="weight_scale_rank_two"):
            g = Graph("linear_fp8a8_packed")
            q = g.input("x_q", i8[2, 8])
            sx = g.input("x_scale", f32[2, 1])
            wq = g.input("w_q", i8[8, 3])
            sw = g.input("w_scale", f32[4, 3])
            with self.assertRaises(ValueError):
                g.add_node("linear_fp8a8_packed", [q, sx, wq, sw], {"kind": "e4m3"})

    def test_rejects_non_positive_scale(self):
        rng = np.random.default_rng(347)
        codes = pack_fp8(rng.standard_normal((8, 3)).astype(np.float32), "e4m3")
        codes = codes.view(np.int8)
        for bad in (0.0, -0.5):
            with self.subTest(op="dequantize_fp8_group", scale=bad):
                with self.assertRaises(ValueError):
                    _run(
                        self._group_graph((8, 3), 2),
                        {"q": codes, "s": np.full((2, 3), bad, np.float32)},
                    )
            with self.subTest(op="linear_fp8a8_packed", scale=bad):
                g = self._packed_a8_graph((2, 8), (8, 3), bias=False)
                with self.assertRaises(ValueError):
                    _run(
                        g,
                        {
                            "x_q": codes,
                            "x_scale": np.full((2, 1), bad, np.float32),
                            "w_q": codes,
                            "w_scale": np.ones(3, np.float32),
                        },
                    )

    def test_group_helper_matches_manual(self):
        rng = np.random.default_rng(348)
        w = rng.standard_normal((8, 6)).astype(np.float32)
        carrier, scale = quantize_weight_fp8_group(w, 4, "e4m3")
        amax = np.max(np.abs(w.reshape(2, 4, 6)), axis=1)
        limit = FP8_MAX["e4m3"]
        self.assertEqual(scale.shape, (2, 6))
        np.testing.assert_allclose(scale, (amax / limit).astype(np.float32))
        # 载体全部落在格点上：再 pack / unpack 一轮不变。
        np.testing.assert_array_equal(
            carrier, unpack_fp8(pack_fp8(carrier, "e4m3").view(np.uint8), "e4m3")
        )
        # 每组都贴着端点：scale 取的是本组的 amax，不是邻组的。
        group_max = np.max(np.abs(carrier.reshape(2, 4, 6)), axis=1)
        np.testing.assert_allclose(group_max, np.full_like(group_max, limit), rtol=1e-6)

    def test_group_helper_rejects(self):
        w = np.ones((8, 3), np.float32)
        for bad in (0, -4):
            with self.subTest(group_size=bad):
                with self.assertRaises(ValueError):
                    quantize_weight_fp8_group(w, bad)
        with self.subTest(case="does_not_divide_k"):
            with self.assertRaises(ValueError):
                quantize_weight_fp8_group(w, 3)
        with self.subTest(case="not_two_dim"):
            with self.assertRaises(ValueError):
                quantize_weight_fp8_group(np.ones((8,), np.float32), 4)
        with self.subTest(case="nan"):
            bad = np.ones((8, 3), np.float32)
            bad[0, 0] = np.nan
            with self.assertRaises(ValueError):
                quantize_weight_fp8_group(bad, 4)

    def test_graph_survives_json_roundtrip(self):
        rng = np.random.default_rng(349)
        scale = np.array([[0.5, 0.25, 0.125], [1.0, 0.75, 0.0625]], np.float32)
        w = rng.standard_normal((8, 3)).astype(np.float32)
        codes = pack_fp8(
            quantize_fp8(w, np.repeat(scale, 4, axis=0), "e4m3"), "e4m3"
        ).view(np.int8)
        feed = {
            "q": codes,
            "s": scale,
            "x": rng.standard_normal((2, 8)).astype(np.float32),
            "b": np.zeros(3, np.float32),
        }
        graphs = {
            "dequantize_fp8_group": (self._group_graph((8, 3), 2), ["q", "s"]),
            "linear_fp8_group": (
                self._linear_group_graph((2, 8), (8, 3), 2),
                ["x", "q", "s", "b"],
            ),
        }
        for name, (g, keys) in graphs.items():
            with self.subTest(op=name):
                sub = {k: feed[k] for k in keys}
                again = loads(dumps(g))
                np.testing.assert_array_equal(_run(g, sub)["y"], _run(again, sub)["y"])

    def test_device_kernels_and_tables_declared(self):
        from aetherflow.backends.device_shared import (
            FP8_OPS,
            GPU_OPS,
            _DEQUANTIZE_FP8_GROUP_SRC,
            _INDEX_ARGS,
            _LINEAR_FP8A8_PACKED_SRC,
        )

        self.assertIn("af_dequantize_fp8_group", _DEQUANTIZE_FP8_GROUP_SRC)
        self.assertEqual(_DEQUANTIZE_FP8_GROUP_SRC.count("af_fp8_decode((int)"), 1)
        self.assertIn("af_linear_fp8a8_packed", _LINEAR_FP8A8_PACKED_SRC)
        self.assertEqual(_LINEAR_FP8A8_PACKED_SRC.count("af_fp8_decode((int)"), 2)
        for name in (
            "dequantize_fp8_group",
            "linear_fp8_group",
            "linear_fp8a8_packed",
        ):
            self.assertIn(name, GPU_OPS)
        self.assertNotIn("dequantize_fp8_group", FP8_OPS)
        self.assertEqual(_INDEX_ARGS["linear_fp8a8_packed"], frozenset({0, 2}))
        self.assertEqual(_INDEX_ARGS["linear_fp8_group"], frozenset({1}))


class TestInt4GroupQuantization(unittest.TestCase):
    """5.343：4 bit 逐组非对称量化（AWQ / GPTQ 的 4 bit 形态）。

    钉住四件事：

    1. pack / unpack 的**位序**：偶数下标进低 4 位、奇数下标进高 4 位，
       且 i8 位模式为负时解码必须先转无符号（0x5A / 0xA5 那组）；
    2. 逐组反量化与线性层是手推公式 ``(q - zp) * scale``，组边界不能串用
       邻组参数；
    3. 主机 helper 的误差界 ``scale / 2``、全零组退化（scale=1, zp=0）与
       拒绝面（形状 / 类型 / NaN / 不整除 K）;
    4. 图能构图 → 执行 → JSON 往返 → 再执行，结果逐位相同。
    """

    def _pack_graph(self, shape):
        g = Graph("pack_int4")
        q = g.input("q", i8[shape])
        g.output(g.add_node("pack_int4", [q]), "p")
        return g

    def _unpack_graph(self, shape):
        g = Graph("unpack_int4")
        p = g.input("p", i8[shape])
        g.output(g.add_node("unpack_int4", [p]), "q")
        return g

    def _dequantize_graph(self, shape, groups):
        g = Graph("dequantize_u4_group")
        q = g.input("q", i8[shape])
        s = g.input("s", f32[(groups, shape[-1])])
        zp = g.input("zp", f32[(groups, shape[-1])])
        g.output(g.add_node("dequantize_u4_group", [q, s, zp]), "w")
        return g

    def _linear_graph(self, x_shape, w_shape, groups, bias=True):
        g = Graph("linear_u4_group")
        x = g.input("x", f32[x_shape])
        q = g.input("q", i8[w_shape])
        s = g.input("s", f32[(groups, w_shape[-1])])
        zp = g.input("zp", f32[(groups, w_shape[-1])])
        args = [x, q, s, zp]
        if bias:
            args.append(g.input("b", f32[(w_shape[-1],)]))
        g.output(g.add_node("linear_u4_group", args), "y")
        return g

    def test_pack_bit_order_and_roundtrip(self):
        q = np.array(
            [[0, 15, 1, 14], [15, 0, 7, 8], [3, 12, 5, 10]], dtype=np.int8
        )
        packed = _run(self._pack_graph((3, 4)), {"q": q})["p"]
        # 手推位模式：低 4 位是偶数下标，高 4 位是奇数下标。
        want = np.array(
            [[0x00 | (15 << 4), 1 | (14 << 4)], [15 | (0 << 4), 7 | (8 << 4)],
             [3 | (12 << 4), 5 | (10 << 4)]],
            dtype=np.uint8,
        ).view(np.int8)
        np.testing.assert_array_equal(packed, want)
        # 高 4 位 >= 0x8 时字节是负的 i8；解码要先转无符号再取半字节。
        np.testing.assert_array_equal(
            _run(self._unpack_graph((3, 2)), {"p": packed})["q"], q
        )

    def test_unpack_handles_negative_bytes(self):
        codes = np.array([0x00, 0xFF, 0x0F, 0xF0, 0x5A, 0xA5], dtype=np.uint8)
        got = _run(
            self._unpack_graph((2, 3)), {"p": codes.view(np.int8).reshape(2, 3)}
        )["q"]
        want = np.array(
            [[0, 0, 15, 15, 15, 0], [0, 15, 10, 5, 5, 10]], dtype=np.int8
        )
        np.testing.assert_array_equal(got, want)

    def test_dequantize_group_matches_manual(self):
        rng = np.random.default_rng(3430)
        q = rng.integers(0, 16, size=(8, 3), dtype=np.int8)
        scale = np.array([[0.5, 0.25, 0.125], [1.0, 0.75, 0.0625]], np.float32)
        zp = np.array([[8, 0, 15], [3, 8, 11]], np.float32)
        got = _run(
            self._dequantize_graph((8, 3), 2), {"q": q, "s": scale, "zp": zp}
        )["w"]
        want = (q.astype(np.float32) - np.repeat(zp, 4, axis=0)) * np.repeat(
            scale, 4, axis=0
        )
        np.testing.assert_array_equal(got, want)
        # 组边界：前 4 行只减 / 乘第 0 组，后 4 行只减 / 乘第 1 组。
        np.testing.assert_array_equal(got[:4], (q[:4].astype(np.float32) - zp[0]) * scale[0])
        np.testing.assert_array_equal(got[4:], (q[4:].astype(np.float32) - zp[1]) * scale[1])

    def test_linear_matches_dequantized_matmul(self):
        rng = np.random.default_rng(3431)
        x_shape, w_shape, groups = (2, 3, 8), (8, 6), 4
        x = rng.standard_normal(x_shape).astype(np.float32)
        q = rng.integers(0, 16, size=w_shape, dtype=np.int8)
        scale = (np.abs(rng.standard_normal((groups, 6))).astype(np.float32) * 0.5 + 0.125)
        zp = rng.integers(0, 16, size=(groups, 6)).astype(np.float32)
        b = rng.standard_normal(6).astype(np.float32)
        got = _run(
            self._linear_graph(x_shape, w_shape, groups),
            {"x": x, "q": q, "s": scale, "zp": zp, "b": b},
        )["y"]
        w = (q.astype(np.float32) - np.repeat(zp, 2, axis=0)) * np.repeat(scale, 2, axis=0)
        np.testing.assert_array_equal(got, np.matmul(x, w) + b)

    def test_rejects_bad_forms(self):
        with self.subTest(case="pack_odd_last_dim"):
            with self.assertRaises(ValueError):
                self._pack_graph((2, 3))
        with self.subTest(case="pack_symbolic_last_dim"):
            g = Graph("pack_int4")
            q = g.input("q", i8[2, "N"])
            with self.assertRaises(ValueError):
                g.add_node("pack_int4", [q])
        with self.subTest(case="pack_not_i8"):
            g = Graph("pack_int4")
            q = g.input("q", f32[2, 4])
            with self.assertRaises(TypeError):
                g.add_node("pack_int4", [q])
        with self.subTest(case="unpack_not_i8"):
            g = Graph("unpack_int4")
            p = g.input("p", f32[2, 2])
            with self.assertRaises(TypeError):
                g.add_node("unpack_int4", [p])
        with self.subTest(case="group_does_not_divide_k"):
            with self.assertRaises(ValueError):
                self._dequantize_graph((8, 3), 3)
        with self.subTest(case="zero_point_shape_mismatch"):
            g = Graph("dequantize_u4_group")
            q = g.input("q", i8[8, 3])
            s = g.input("s", f32[2, 3])
            zp = g.input("zp", f32[3])
            with self.assertRaises(ValueError):
                g.add_node("dequantize_u4_group", [q, s, zp])
        with self.subTest(case="linear_x_not_f32"):
            g = Graph("linear_u4_group")
            x = g.input("x", f16[2, 8])
            q = g.input("q", i8[8, 3])
            s = g.input("s", f32[2, 3])
            zp = g.input("zp", f32[2, 3])
            with self.assertRaises(TypeError):
                g.add_node("linear_u4_group", [x, q, s, zp])
        with self.subTest(case="linear_weight_not_i8"):
            g = Graph("linear_u4_group")
            x = g.input("x", f32[2, 8])
            q = g.input("q", f32[8, 3])
            s = g.input("s", f32[2, 3])
            zp = g.input("zp", f32[2, 3])
            with self.assertRaises(TypeError):
                g.add_node("linear_u4_group", [x, q, s, zp])

    def test_rejects_bad_values(self):
        rng = np.random.default_rng(3432)
        q = rng.integers(0, 16, size=(8, 3), dtype=np.int8)
        s = np.ones((2, 3), np.float32)
        zp = np.full((2, 3), 8.0, np.float32)
        with self.subTest(case="non_positive_scale"):
            for bad in (0.0, -0.5):
                with self.assertRaises(ValueError):
                    _run(
                        self._dequantize_graph((8, 3), 2),
                        {"q": q, "s": np.full((2, 3), bad, np.float32), "zp": zp},
                    )
        with self.subTest(case="zero_point_out_of_range"):
            for bad in (-1.0, 16.0):
                with self.assertRaises(ValueError):
                    _run(
                        self._dequantize_graph((8, 3), 2),
                        {"q": q, "s": s, "zp": np.full((2, 3), bad, np.float32)},
                    )
        with self.subTest(case="q_out_of_4bit_range"):
            bad_q = q.copy()
            bad_q[0, 0] = 16
            with self.assertRaises(ValueError):
                _run(self._dequantize_graph((8, 3), 2), {"q": bad_q, "s": s, "zp": zp})

    def test_helper_error_bound_and_degenerate_groups(self):
        rng = np.random.default_rng(3433)
        w = rng.standard_normal((8, 6)).astype(np.float32)
        q, scale, zp = quantize_weight_int4_group(w, 4)
        self.assertEqual(q.dtype, np.int8)
        self.assertEqual((q.shape, scale.shape, zp.shape), ((8, 6), (2, 6), (2, 6)))
        self.assertTrue(int(q.min()) >= 0 and int(q.max()) <= 15)
        self.assertTrue(float(zp.min()) >= 0.0 and float(zp.max()) <= 15.0)
        restored = (q.astype(np.float32) - np.repeat(zp, 4, axis=0)) * np.repeat(
            scale, 4, axis=0
        )
        self.assertLessEqual(
            float(np.max(np.abs(restored - w) - np.repeat(scale / 2.0, 4, axis=0))),
            1e-6,
        )
        # 全零组退化成 scale=1、zp=0：零必须能被精确表示，且不触发非正 scale。
        zeros = np.zeros((8, 3), np.float32)
        zq, zs, zz = quantize_weight_int4_group(zeros, 4)
        np.testing.assert_array_equal(zq, np.zeros((8, 3), np.int8))
        np.testing.assert_array_equal(zs, np.ones((2, 3), np.float32))
        np.testing.assert_array_equal(zz, np.zeros((2, 3), np.float32))
        # 组内常数（非零）：扩展值域到 0 之后端点恰好落在格点上，误差为 0。
        const = np.full((8, 1), 0.5, np.float32)
        cq, cs, cz = quantize_weight_int4_group(const, 4)
        np.testing.assert_array_equal(
            (cq.astype(np.float32) - np.repeat(cz, 4, axis=0)) * np.repeat(cs, 4, axis=0),
            const,
        )

    def test_helper_rejects(self):
        w = np.ones((8, 3), np.float32)
        for bad in (0, -4):
            with self.subTest(group_size=bad):
                with self.assertRaises(ValueError):
                    quantize_weight_int4_group(w, bad)
        with self.subTest(case="does_not_divide_k"):
            with self.assertRaises(ValueError):
                quantize_weight_int4_group(w, 3)
        with self.subTest(case="not_two_dim"):
            with self.assertRaises(ValueError):
                quantize_weight_int4_group(np.ones((8,), np.float32), 4)
        with self.subTest(case="not_float"):
            with self.assertRaises(TypeError):
                quantize_weight_int4_group(np.ones((8, 3), np.int32), 4)
        with self.subTest(case="nan"):
            bad = np.ones((8, 3), np.float32)
            bad[0, 0] = np.nan
            with self.assertRaises(ValueError):
                quantize_weight_int4_group(bad, 4)

    def test_graph_survives_json_roundtrip(self):
        rng = np.random.default_rng(3434)
        q = rng.integers(0, 16, size=(8, 3), dtype=np.int8)
        feed = {
            "q": q,
            "q4": (np.arange(24, dtype=np.int8) % 16).reshape(6, 4),
            "p": np.array([[0x5A, 0xA5]], np.uint8).view(np.int8),
            "s": np.array([[0.5, 0.25, 0.125], [1.0, 0.75, 0.0625]], np.float32),
            "zp": np.array([[8, 0, 15], [3, 8, 11]], np.float32),
            "x": rng.standard_normal((2, 8)).astype(np.float32),
            "b": np.zeros(3, np.float32),
        }
        graphs = {
            "pack_int4": (self._pack_graph((6, 4)), ["q4"], "p"),
            "unpack_int4": (self._unpack_graph((1, 2)), ["p"], "q"),
            "dequantize_u4_group": (
                self._dequantize_graph((8, 3), 2), ["q", "s", "zp"], "w"
            ),
            "linear_u4_group": (
                self._linear_graph((2, 8), (8, 3), 2),
                ["x", "q", "s", "zp", "b"],
                "y",
            ),
        }
        for name, (g, keys, out_key) in graphs.items():
            with self.subTest(op=name):
                # pack 图的输入名固定是 q，这里用 q4 的载荷喂它。
                sub = {("q" if k == "q4" else k): feed[k] for k in keys}
                again = loads(dumps(g))
                np.testing.assert_array_equal(
                    _run(g, sub)[out_key], _run(again, sub)[out_key]
                )

    def test_device_kernels_and_tables_declared(self):
        from aetherflow.backends.device_shared import (
            GPU_OPS,
            _DEQUANTIZE_U4_GROUP_SRC,
            _INDEX_ARGS,
            _PACK_INT4_SRC,
            _UNPACK_INT4_SRC,
        )

        self.assertIn("af_pack_int4", _PACK_INT4_SRC)
        self.assertIn("af_unpack_int4", _UNPACK_INT4_SRC)
        self.assertIn("af_dequantize_u4_group", _DEQUANTIZE_U4_GROUP_SRC)
        self.assertIn("0x0f", _PACK_INT4_SRC)
        self.assertIn("0x0f", _UNPACK_INT4_SRC)
        for name in ("pack_int4", "unpack_int4", "dequantize_u4_group", "linear_u4_group"):
            self.assertIn(name, GPU_OPS)
        self.assertEqual(_INDEX_ARGS["pack_int4"], frozenset({0}))
        self.assertEqual(_INDEX_ARGS["unpack_int4"], frozenset({0}))
        self.assertEqual(_INDEX_ARGS["dequantize_u4_group"], frozenset({0}))
        self.assertEqual(_INDEX_ARGS["linear_u4_group"], frozenset({1}))


class TestConformanceCases(unittest.TestCase):
    def test_reference_backend_passes_new_cases(self):
        names = {
            "quantize_i8_channel",
            "dequantize_i8_channel",
            "fake_quantize_i8_channel",
            "fake_quantize_i8_lsq_channel",
            "linear_i8_bias",
            "linear_i8_group_bias",
            "linear_i8a8_bias",
            "quantize_i8_zp_channel",
            "dequantize_i8_zp_channel",
            "linear_i8_zp_bias",
            "linear_i8a8_zp_bias",
            "dequantize_i8_group",
            "dequantize_i8_group_zp",
            "linear_fp8_e4m3_bias",
            "linear_fp8_e5m2_bias",
            "linear_fp8a8_e4m3_bias",
            "linear_fp8a8_e5m2_rank2",
            "pack_fp8_e4m3",
            "pack_fp8_e5m2",
            "unpack_fp8_e4m3",
            "unpack_fp8_e5m2",
            "linear_fp8_packed_e4m3_bias",
            "linear_fp8_packed_e5m2",
            "dequantize_fp8_group_e4m3",
            "linear_fp8_group_e4m3_bias",
            "linear_fp8_group_e5m2_rank2",
            "linear_fp8a8_packed_e4m3_bias",
            "linear_fp8a8_packed_e5m2_rank2",
            "pack_int4",
            "unpack_int4",
            "dequantize_u4_group",
            "linear_u4_group_bias",
            "linear_u4_group_rank2",
        }
        cases = [c for c in default_cases() if c.name in names]
        self.assertEqual({c.name for c in cases}, names)
        report = run_conformance(BACKEND, cases=cases)
        self.assertTrue(report.ok, report.text())


if __name__ == "__main__":
    unittest.main()
