"""``sparse_linear`` 的 cuSPARSE 路径（5.203）。

分两层：

- **离线**：dtype / 索引类型的登记表、库缺失时的错误、``sparse_impl`` 的
  选择逻辑、失败回落与"只试一次"。这些都不需要 GPU，也不加载库。
- **真机**：与自研核同形状对拍（含空步与重复行）、f16 / f64 的显式拒绝、
  捕获期拒绝。没有 CUDA 或没有 libcusparse 时整组跳过。
"""

import unittest
from unittest import mock

import numpy as np

from aetherflow import Graph, f16, f32, f64, i32
from aetherflow.backends.base import BackendCapabilityError
from aetherflow.backends.cuda import CUDA_AVAILABLE, CudaBackend
from aetherflow.backends.cuda import cusparse
from aetherflow.backends.cuda.cusparse import (
    CusparseSpMM,
    CusparseUnavailable,
    CusparseUnsupported,
    data_type,
    index_type,
)


class TestCusparseRegistry(unittest.TestCase):
    """登记表就是覆盖声明：表内的映射对上，表外的显式拒绝。"""

    def test_float32_maps_to_cuda_r_32f(self):
        self.assertEqual(data_type(np.float32), 0)
        self.assertEqual(data_type("float32"), 0)

    def test_float16_and_float64_are_refused(self):
        for dtype in (np.float16, np.float64):
            with self.assertRaises(CusparseUnsupported) as caught:
                data_type(dtype)
            message = str(caught.exception)
            self.assertIn("sparse_impl='custom'", message)
            self.assertIn(str(np.dtype(dtype)), message)

    def test_int32_index_maps_to_32i(self):
        self.assertEqual(index_type(np.int32), 1)

    def test_int64_index_is_refused_with_reason(self):
        with self.assertRaises(CusparseUnsupported) as caught:
            index_type(np.int64)
        self.assertIn("i64", str(caught.exception))

    def test_plan_key_includes_nnz_and_dtype(self):
        """workspace 大小随 nnz 变，键里漏掉它就会拿到装不下的缓冲。"""
        base = CusparseSpMM.plan_key(8, 64, 32, 100, np.float32)
        self.assertEqual(base, CusparseSpMM.plan_key(8, 64, 32, 100, "float32"))
        self.assertNotEqual(base, CusparseSpMM.plan_key(8, 64, 32, 101, np.float32))
        self.assertNotEqual(base, CusparseSpMM.plan_key(9, 64, 32, 100, np.float32))
        self.assertNotEqual(base, CusparseSpMM.plan_key(8, 64, 32, 100, np.float16))


class TestCusparseUnavailable(unittest.TestCase):
    """库缺失必须是可读的错误，而不是 import 期的副作用。"""

    def setUp(self):
        self.error = CusparseUnavailable("找不到 cuSPARSE 动态库")
        patcher = mock.patch.object(cusparse, "_load", side_effect=self.error)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_available_is_false(self):
        self.assertFalse(cusparse.available())

    def test_engine_raises_with_library_name(self):
        with self.assertRaises(CusparseUnavailable) as caught:
            CusparseSpMM()
        self.assertIn("cuSPARSE", str(caught.exception))


def _fake_backend(sparse_impl, *, capturing=False, broken=None, engine=None):
    """不构造 CUDA 上下文，只喂 ``_pick_cusparse`` 需要的字段。

    ``CudaBackend.__new__`` 不跑 ``__init__``，因此不碰显卡；方法体只读
    这几个属性，与真实实例走的是同一份代码。
    """
    backend = CudaBackend.__new__(CudaBackend)
    backend.sparse_impl = sparse_impl
    backend._capturing = capturing
    backend._cusparse_broken = broken
    backend._cusparse = engine
    return backend


class TestSparseImplSelection(unittest.TestCase):
    """``sparse_impl`` 的选择逻辑：默认 custom、auto 可回落、显式必须能算。"""

    def test_custom_never_uses_library(self):
        backend = _fake_backend("custom")
        self.assertFalse(
            backend._pick_cusparse(
                np.zeros(4, np.float32), np.zeros(2, np.int32), np.zeros(3, np.int32)
            )
        )

    def test_auto_falls_back_when_library_missing(self):
        backend = _fake_backend("auto", broken="找不到 cuSPARSE")
        self.assertFalse(
            backend._pick_cusparse(
                np.zeros(4, np.float32), np.zeros(2, np.int32), np.zeros(3, np.int32)
            )
        )

    def test_auto_never_tries_library_for_f16(self):
        """f16 不在登记表里：auto 直接走自研核，不去碰库。"""
        backend = _fake_backend("auto", engine=object())
        self.assertFalse(
            backend._pick_cusparse(
                np.zeros(4, np.float16), np.zeros(2, np.int32), np.zeros(3, np.int32)
            )
        )

    def test_auto_never_uses_library_while_capturing(self):
        backend = _fake_backend("auto", capturing=True, engine=object())
        self.assertFalse(
            backend._pick_cusparse(
                np.zeros(4, np.float32), np.zeros(2, np.int32), np.zeros(3, np.int32)
            )
        )

    def test_auto_uses_library_when_available(self):
        backend = _fake_backend("auto", engine=object())
        self.assertTrue(
            backend._pick_cusparse(
                np.zeros(4, np.float32), np.zeros(2, np.int32), np.zeros(3, np.int32)
            )
        )

    def test_explicit_cusparse_rejects_f64(self):
        backend = _fake_backend("cusparse", engine=object())
        with self.assertRaises(BackendCapabilityError) as caught:
            backend._pick_cusparse(
                np.zeros(4, np.float64), np.zeros(2, np.int32), np.zeros(3, np.int32)
            )
        self.assertIn("float32", str(caught.exception))

    def test_explicit_cusparse_rejects_i64_index(self):
        backend = _fake_backend("cusparse", engine=object())
        with self.assertRaises(BackendCapabilityError) as caught:
            backend._pick_cusparse(
                np.zeros(4, np.float32), np.zeros(2, np.int64), np.zeros(3, np.int32)
            )
        self.assertIn("int32", str(caught.exception))

    def test_explicit_cusparse_rejects_capture(self):
        backend = _fake_backend("cusparse", capturing=True, engine=object())
        with self.assertRaises(BackendCapabilityError) as caught:
            backend._pick_cusparse(
                np.zeros(4, np.float32), np.zeros(2, np.int32), np.zeros(3, np.int32)
            )
        self.assertIn("捕获", str(caught.exception))

    def test_explicit_cusparse_needs_the_library(self):
        """显式要库却没有库：转成能力错误，而不是在深处抛 OSError。"""
        backend = _fake_backend("cusparse")
        with mock.patch(
            "aetherflow.backends.cuda.backend.CusparseSpMM",
            side_effect=CusparseUnavailable("没有库"),
        ):
            # 选择阶段只回答"该不该走库"，建 handle 发生在真正要算的时候
            self.assertTrue(
                backend._pick_cusparse(
                    np.zeros(4, np.float32),
                    np.zeros(2, np.int32),
                    np.zeros(3, np.int32),
                )
            )
            with self.assertRaises(BackendCapabilityError) as caught:
                backend._sparse_linear_cusparse(
                    None, None, None, None, steps=1, rows=1, cols=1
                )
        self.assertIn("cuSPARSE", str(caught.exception))


class _Boom:
    """第一次 SpMM 就失败：用来验收 auto 的回落与 broken 的记账。"""

    def spmm(self, *args, **kwargs):
        raise RuntimeError("spmm 第一次就失败")


class TestSparseFallback(unittest.TestCase):
    """auto 失败回落、显式失败照抛——与 cuDNN 的 ``_conv2d_cudnn`` 同一条策略。"""

    def test_auto_records_failure_and_falls_back(self):
        backend = _fake_backend("auto", engine=_Boom())
        result = backend._sparse_linear_cusparse(
            None, None, None, None, steps=4, rows=6, cols=5
        )
        self.assertIsNone(result)
        self.assertIn("spmm 第一次就失败", backend._cusparse_broken)

    def test_explicit_reraises(self):
        backend = _fake_backend("cusparse", engine=_Boom())
        with self.assertRaises(RuntimeError):
            backend._sparse_linear_cusparse(
                None, None, None, None, steps=4, rows=6, cols=5
            )
        self.assertIsNone(backend._cusparse_broken)

    def test_broken_is_remembered_and_engine_not_retried(self):
        """记下原因之后 auto 直接走自研核，不再每步 try/except 一次。"""
        calls = []

        def engine():
            calls.append(1)
            return object()

        backend = _fake_backend("auto", broken="上次失败")
        backend._cusparse_engine = engine
        self.assertFalse(
            backend._pick_cusparse(
                np.zeros(4, np.float32), np.zeros(2, np.int32), np.zeros(3, np.int32)
            )
        )
        self.assertEqual(calls, [])


def _sparse_graph(rows, cols, steps, counts, dtype=f32):
    """按 ``counts`` 造一张 sparse_linear 图，返回 (graph, feed)。"""
    offsets = np.zeros(steps + 1, dtype=np.int32)
    for index, count in enumerate(counts):
        offsets[index + 1] = offsets[index] + count
    nnz = int(offsets[-1])
    g = Graph("sparse_linear_cusparse")
    weights = g.input("weights", dtype[rows, cols])
    indices = g.input("indices", i32[nnz])
    values = g.input("values", dtype[nnz])
    slots = g.input("offsets", i32[steps + 1])
    g.output(g.add_node("sparse_linear", [weights, indices, values, slots]), "y")
    rng = np.random.default_rng(11)
    np_dtype = {f32: np.float32, f16: np.float16, f64: np.float64}[dtype]
    # 重复行（3 出现两次）与空步都留着：前者验累加、后者验清零
    indices_np = np.array([1, 3, 3, 0, 1][:nnz], dtype=np.int32)
    if nnz > 5:  # 大形状补足事件
        indices_np = rng.integers(0, rows, nnz).astype(np.int32)
    feed = {
        "weights": rng.standard_normal((rows, cols)).astype(np_dtype),
        "indices": indices_np,
        "values": rng.standard_normal(nnz).astype(np_dtype),
        "offsets": offsets,
    }
    return g, feed


@unittest.skipUnless(CUDA_AVAILABLE, "需要 cupy 与 CUDA 设备")
class TestCudaSparseCusparse(unittest.TestCase):
    """真机：库路径与自研核在同一个形状上对拍。"""

    @classmethod
    def setUpClass(cls):
        if not cusparse.available():
            raise unittest.SkipTest("系统里没有可用的 cuSPARSE")
        cls.fast = CudaBackend(sparse_impl="cusparse")
        cls.custom = CudaBackend(sparse_impl="custom")

    def _check(self, rows, cols, steps, counts, dtype=f32, tol=1e-5):
        g, feed = _sparse_graph(rows, cols, steps, counts, dtype)
        expected = self.custom.execute(g, feed)["y"]
        got = self.fast.execute(g, feed)["y"]
        np.testing.assert_allclose(got, expected, rtol=tol, atol=tol)

    def test_matches_custom_with_empty_step_and_repeat_row(self):
        # 4 步、事件数 5：第 2 步是空步，第 1 步里 3 出现两次
        self._check(6, 5, 4, (2, 0, 2, 1))

    def test_matches_custom_when_nnz_varies(self):
        """nnz 进 workspace 键：同一张图换事件数必须重新量 workspace。"""
        for nnz_per_step in (0, 1, 3, 7):
            self._check(32, 17, 6, (nnz_per_step,) * 6, tol=1e-4)

    def test_matches_custom_on_larger_shape(self):
        self._check(512, 256, 64, (7,) * 64, tol=1e-4)

    def test_auto_matches_custom(self):
        g, feed = _sparse_graph(16, 8, 5, (2, 0, 2, 1, 3))
        auto = CudaBackend(sparse_impl="auto")
        np.testing.assert_allclose(
            auto.execute(g, feed)["y"], self.custom.execute(g, feed)["y"],
            rtol=1e-5, atol=1e-5,
        )

    def test_float16_is_refused_explicitly(self):
        g, feed = _sparse_graph(6, 5, 4, (2, 0, 2, 1), dtype=f16)
        with self.assertRaises(BackendCapabilityError) as caught:
            self.fast.execute(g, feed)
        self.assertIn("float32", str(caught.exception))

    def test_float64_is_refused_explicitly(self):
        g, feed = _sparse_graph(6, 5, 4, (2, 0, 2, 1), dtype=f64)
        with self.assertRaises(BackendCapabilityError) as caught:
            self.fast.execute(g, feed)
        self.assertIn("float32", str(caught.exception))

    def test_capture_is_refused_explicitly(self):
        g, feed = _sparse_graph(6, 5, 4, (2, 0, 2, 1))
        with self.assertRaises(BackendCapabilityError) as caught:
            self.fast.capture(g, feed)
        self.assertIn("捕获", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
