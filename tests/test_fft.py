"""复数 dtype 与 FFT 算子：参考语义是 numpy.fft，cuFFT 只是 CUDA 侧的实现。

这批用例全部是离线的：它验证 IR 的类型系统、形状推导、归一化口径与拒绝面。
cuFFT 绑定本身不做真机对拍（当前环境没有可用显卡），因此这里只测"纯函数
部分"——dtype 登记表、缩放因子、plan 键——以及"库不在时错误是否清楚"。
"""

from __future__ import annotations

import unittest

import numpy as np

from aetherflow import (
    Graph,
    NoGradientError,
    ReferenceBackend,
    backward,
    bf16,
    c64,
    c128,
    dumps,
    f16,
    f32,
    loads,
    verify,
)
from aetherflow.backends.cuda import cufft


BACKEND = ReferenceBackend()


def _run(graph, feed):
    return BACKEND.execute(graph, feed)


def _complex_input(graph, name="x", shape=(2, 8), dtype=c64):
    return graph.input(name, dtype[shape])


class TestComplexDType(unittest.TestCase):
    def test_alias_and_np_dtype(self):
        self.assertEqual(c64.value, "c64")
        self.assertEqual(c128.value, "c128")
        self.assertEqual(np.dtype(c64.np_dtype), np.dtype("complex64"))
        self.assertEqual(np.dtype(c128.np_dtype), np.dtype("complex128"))
        self.assertTrue(c64.is_complex)
        self.assertTrue(c128.is_complex)
        self.assertFalse(f32.is_complex)
        self.assertTrue(f32.is_float)
        self.assertFalse(c64.is_float)

    def test_complex_symbols_still_clash_with_dtype_names(self):
        # 新增 dtype 之后，形状符号同样不能叫 c64，否则 ``c64[c64, 4]`` 不可解析
        with self.assertRaises(ValueError):
            c64["c64", 4]

    def test_real_to_complex_input_is_lossless(self):
        g = Graph()
        x = _complex_input(g, shape=(2, 4))
        g.output(x, "y")
        out = _run(g, {"x": np.ones((2, 4), dtype=np.float32)})["y"]
        self.assertEqual(out.dtype, np.dtype("complex64"))
        np.testing.assert_array_equal(out.imag, 0.0)
        np.testing.assert_array_equal(out.real, 1.0)

    def test_complex_to_real_input_is_rejected(self):
        g = Graph()
        x = g.input("x", f32[2, 4])
        g.output(x, "y")
        with self.assertRaises(TypeError):
            _run(g, {"x": np.zeros((2, 4), dtype=np.complex64)})


class TestFftReference(unittest.TestCase):
    def test_forward_matches_numpy(self):
        rng = np.random.default_rng(0)
        x = (rng.standard_normal((3, 16)) + 1j * rng.standard_normal((3, 16))).astype(
            np.complex64
        )
        g = Graph()
        h = _complex_input(g, shape=(3, 16))
        g.output(g.add_node("fft", [h], {}), "y")
        got = _run(g, {"x": x})["y"]
        self.assertEqual(got.dtype, np.dtype("complex64"))
        np.testing.assert_allclose(got, np.fft.fft(x, axis=-1), rtol=1e-5, atol=1e-5)

    def test_all_norms(self):
        rng = np.random.default_rng(1)
        x = (rng.standard_normal(12) + 1j * rng.standard_normal(12)).astype(
            np.complex128
        )
        for norm in ("backward", "ortho", "forward"):
            with self.subTest(norm=norm):
                g = Graph()
                h = g.input("x", c128[12])
                g.output(g.add_node("fft", [h], {"norm": norm}), "y")
                got = _run(g, {"x": x})["y"]
                np.testing.assert_allclose(
                    got, np.fft.fft(x, norm=norm), rtol=1e-12, atol=1e-12
                )

    def test_inverse_round_trip(self):
        rng = np.random.default_rng(2)
        x = (rng.standard_normal((2, 9)) + 1j * rng.standard_normal((2, 9))).astype(
            np.complex128
        )
        g = Graph()
        h = g.input("x", c128[2, 9])
        f = g.add_node("fft", [h], {"norm": "ortho"})
        back = g.add_node("fft", [f], {"norm": "ortho", "inverse": True})
        g.output(back, "y")
        got = _run(g, {"x": x})["y"]
        np.testing.assert_allclose(got, x, rtol=1e-12, atol=1e-12)
        # 与 numpy 的 ifft 同义
        direct = Graph()
        d = direct.input("x", c128[2, 9])
        direct.output(direct.add_node("fft", [d], {"inverse": True}), "y")
        np.testing.assert_allclose(
            _run(direct, {"x": x})["y"], np.fft.ifft(x), rtol=1e-12, atol=1e-12
        )

    def test_rfft_shape_and_values(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal((4, 10)).astype(np.float32)
        g = Graph()
        h = g.input("x", f32[4, 10])
        g.output(g.add_node("rfft", [h], {}), "y")
        self.assertEqual(g.outputs[0].type, c64[4, 6])
        got = _run(g, {"x": x})["y"]
        np.testing.assert_allclose(
            got, np.fft.rfft(x, axis=-1), rtol=1e-5, atol=1e-5
        )

    def test_rfft_symbolic_last_dim(self):
        g = Graph()
        h = g.input("x", f32["B", "T"])
        y = g.add_node("rfft", [h], {})
        g.output(y, "y")
        # 符号维的形状是 "T // 2 + 1"，实例化后才能确定具体数字
        self.assertIn("T", str(y.type.shape[-1]))
        x = np.random.default_rng(4).standard_normal((2, 7)).astype(np.float32)
        got = _run(g, {"x": x})["y"]
        self.assertEqual(got.shape, (2, 4))

    def test_fft_has_no_vjp(self):
        g = Graph()
        h = _complex_input(g, shape=(2, 4))
        y = g.add_node("fft", [h], {})
        g.output(y, "y")
        with self.assertRaises(NoGradientError):
            backward(g, wrt=[h])

    def test_serialization_round_trip(self):
        g = Graph()
        h = g.input("x", c64[2, 8])
        g.output(g.add_node("fft", [h], {"norm": "ortho"}), "y")
        clone = loads(dumps(g))
        self.assertEqual(clone.inputs[0].type, c64[2, 8])
        self.assertEqual(clone.outputs[0].type, c64[2, 8])
        self.assertEqual(list(verify(clone)), [])
        rng = np.random.default_rng(5)
        x = (rng.standard_normal((2, 8)) + 1j * rng.standard_normal((2, 8))).astype(
            np.complex64
        )
        np.testing.assert_allclose(
            _run(clone, {"x": x})["y"], _run(g, {"x": x})["y"]
        )


class TestFftRejections(unittest.TestCase):
    def test_fft_rejects_real_input(self):
        g = Graph()
        h = g.input("x", f32[2, 4])
        with self.assertRaises(TypeError):
            g.add_node("fft", [h], {})

    def test_rfft_rejects_complex_and_half(self):
        # 复数输入不需要半谱变换：它已经是完整的谱
        g = Graph()
        with self.assertRaises(TypeError):
            g.add_node("rfft", [g.input("x", c64[2, 4])], {})
        # 半精度没有参考语义，显式拒绝而不是借用 f32 承载
        g = Graph()
        with self.assertRaises(TypeError):
            g.add_node("rfft", [g.input("z", f16[2, 4])], {})
        g = Graph()
        with self.assertRaises(TypeError):
            g.add_node("rfft", [g.input("b", bf16[2, 4])], {})

    def test_bad_norm(self):
        g = Graph()
        h = _complex_input(g)
        with self.assertRaises(ValueError):
            g.add_node("fft", [h], {"norm": "orthonormal"})
        with self.assertRaises(ValueError):
            g.add_node("rfft", [g.input("r", f32[2, 8])], {"norm": "none"})

    def test_inverse_must_be_bool(self):
        g = Graph()
        h = _complex_input(g)
        with self.assertRaises(TypeError):
            g.add_node("fft", [h], {"inverse": 1})

    def test_rank_zero_rejected(self):
        g = Graph()
        h = g.input("x", c64[()])
        with self.assertRaises(ValueError):
            g.add_node("fft", [h], {})


class TestCufftOffline(unittest.TestCase):
    def test_normalization_table(self):
        n = 8
        root = np.sqrt(n)
        self.assertEqual(cufft.normalization(n, "backward"), 1.0)
        self.assertEqual(cufft.normalization(n, "backward", True), 1.0 / n)
        self.assertEqual(cufft.normalization(n, "ortho"), 1.0 / root)
        self.assertEqual(cufft.normalization(n, "ortho", True), 1.0 / root)
        self.assertEqual(cufft.normalization(n, "forward"), 1.0 / n)
        self.assertEqual(cufft.normalization(n, "forward", True), 1.0)

    def test_normalization_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            cufft.normalization(8, "none")
        for size in (0, -4, True, 3.0):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    cufft.normalization(size, "backward")

    def test_kind_table(self):
        self.assertEqual(cufft.kind(np.complex64), 0x29)
        self.assertEqual(cufft.kind(np.complex64, True), 0x29)
        self.assertEqual(cufft.kind(np.complex128), 0x69)
        self.assertEqual(cufft.kind(np.float32), 0x2A)
        self.assertEqual(cufft.kind(np.float64), 0x6A)

    def test_kind_rejects_unsupported(self):
        for dtype in (np.float16, np.int32, np.dtype("complex64").newbyteorder("S")):
            with self.subTest(dtype=str(dtype)):
                with self.assertRaises(cufft.CufftUnsupported):
                    cufft.kind(dtype)
        # rfft 没有逆变换
        with self.assertRaises(cufft.CufftUnsupported):
            cufft.kind(np.float32, True)

    def test_plan_key_is_address_free(self):
        key = cufft.CufftPlan.plan_key(0x29, 16, 3)
        self.assertEqual(key, (0x29, 16, 3))
        # 方向不进键：同一个 plan 正反共用
        self.assertEqual(cufft.CufftPlan.plan_key(0x29, 16, 3), key)

    def test_available_never_raises(self):
        self.assertIsInstance(cufft.available(), bool)

    def test_missing_library_error_is_explicit(self):
        if cufft.available():
            self.skipTest("本机有 cuFFT，错误路径不适用")
        with self.assertRaises(cufft.CufftUnavailable):
            cufft.CufftPlan()


if __name__ == "__main__":
    unittest.main()
