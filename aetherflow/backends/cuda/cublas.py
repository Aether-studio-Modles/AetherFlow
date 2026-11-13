"""cuBLAS 薄绑定（ctypes）。

为什么不用 ``cupy.matmul``：CuPy 在 CUDA Graph 捕获期间调用 cuBLAS 会直接抛
``NotImplementedError: calling cuBLAS API during stream capture is currently
unsupported``。图捕获是本项目运行期计划的基础，因此 matmul 必须走一条能被
捕获的下发路径。

只绑定 float32 与 float16 的**列主序** GEMM；行主序通过 ``C^T = B^T A^T`` 互换
A/B 得到，这是 cuBLAS 文档给出的标准做法，不需要额外的转置拷贝。

f16 走 ``cublasGemmEx``（计算类型固定 ``CUBLAS_COMPUTE_32F``）：输入输出都是
半精度存储，累加在 float 里完成——这与内核里"读成 float 再算"的策略一致，
不会因为换了 GEMM 就悄悄降低累加精度。

这仍然属于 ADR-004 的范畴：IR 里的 ``matmul`` 依旧是语义完整的 PORTABLE 算子，
"用 cuBLAS 实现它"是后端内部的实现选择，不产生 LIBRARY 节点。
"""

from __future__ import annotations

import ctypes
import ctypes.util

_CUBLAS_OP_N = 0

# cudaDataType_t / cublasComputeType_t / cublasGemmAlgo_t 的常量。
# 这些值在 CUDA 11 与 12 的 ABI 里是稳定的；不 include 头文件，直接按文档写。
_CUDA_R_16F = 2
_CUDA_R_32F = 0
_CUBLAS_COMPUTE_32F = 68
_CUBLAS_GEMM_DEFAULT = -1

# 优先用 find_library 找到的名字；找不到时按发行版的命名习惯逐个尝试。
_CANDIDATES = (
    "libcublas.so.12",
    "libcublas.so.11",
    "libcublas.so",
    "cublas64_12.dll",
    "cublas64_11.dll",
    "cublas.dll",
)

_STATUS = {
    0: "CUBLAS_STATUS_SUCCESS",
    1: "CUBLAS_STATUS_NOT_INITIALIZED",
    2: "CUBLAS_STATUS_ALLOC_FAILED",
    3: "CUBLAS_STATUS_INVALID_VALUE",
    4: "CUBLAS_STATUS_ARCH_MISMATCH",
    5: "CUBLAS_STATUS_MAPPING_ERROR",
    6: "CUBLAS_STATUS_EXECUTION_FAILED",
    7: "CUBLAS_STATUS_INTERNAL_ERROR",
    8: "CUBLAS_STATUS_NOT_SUPPORTED",
}


class CublasError(RuntimeError):
    """cuBLAS 返回非 SUCCESS 状态码。"""


def _load_library() -> ctypes.CDLL:
    names = []
    found = ctypes.util.find_library("cublas")
    if found:
        names.append(found)
    names.extend(_CANDIDATES)
    tried = []
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError as exc:  # 换下一个候选名
            tried.append(f"{name}: {exc}")
    raise OSError("找不到 cuBLAS 动态库，尝试过：\n  " + "\n  ".join(tried))


def _ptr(array) -> ctypes.c_void_p:
    return ctypes.c_void_p(array.data.ptr)


class Cublas:
    """一个进程内复用的 cuBLAS handle。

    handle 与 stream 的绑定做了缓存：同一 stream 上重复调用不会重复下发
    ``cublasSetStream``，否则每节点都要多付一次驱动往返。
    """

    def __init__(self) -> None:
        self._lib = _load_library()
        self._declare()
        handle = ctypes.c_void_p()
        self._check(self._lib.cublasCreate_v2(ctypes.byref(handle)), "cublasCreate_v2")
        self._handle = handle
        self._stream = None  # 当前绑定到 handle 的 stream 地址

    # ------------------------------------------------------------------
    # 符号声明
    # ------------------------------------------------------------------

    def _declare(self) -> None:
        lib = self._lib
        vp = ctypes.c_void_p
        fp = ctypes.POINTER(ctypes.c_float)
        ci = ctypes.c_int
        cll = ctypes.c_longlong

        lib.cublasCreate_v2.argtypes = [ctypes.POINTER(vp)]
        lib.cublasCreate_v2.restype = ci

        lib.cublasDestroy_v2.argtypes = [vp]
        lib.cublasDestroy_v2.restype = ci

        lib.cublasSetStream_v2.argtypes = [vp, vp]
        lib.cublasSetStream_v2.restype = ci

        lib.cublasSgemm_v2.argtypes = [
            vp, ci, ci, ci, ci, ci, fp, vp, ci, vp, ci, fp, vp, ci,
        ]
        lib.cublasSgemm_v2.restype = ci

        lib.cublasSgemmStridedBatched.argtypes = [
            vp, ci, ci, ci, ci, ci, fp, vp, ci, cll, vp, ci, cll, fp, vp, ci, cll, ci,
        ]
        lib.cublasSgemmStridedBatched.restype = ci

        # cublasGemmEx：alpha/beta 对 CUBLAS_COMPUTE_32F 是 float 指针，
        # A/B/C 用 void*，类型由 cudaDataType_t 参数给出。
        lib.cublasGemmEx.argtypes = [
            vp, ci, ci, ci, ci, ci, fp, vp, ci, ci, vp, ci, ci, fp, vp, ci, ci, ci, ci,
        ]
        lib.cublasGemmEx.restype = ci

        lib.cublasGemmStridedBatchedEx.argtypes = [
            vp, ci, ci, ci, ci, ci, fp, vp, ci, ci, cll, vp, ci, ci, cll,
            fp, vp, ci, ci, cll, ci, ci, ci,
        ]
        lib.cublasGemmStridedBatchedEx.restype = ci

    @staticmethod
    def _check(status: int, what: str) -> None:
        if status != 0:
            raise CublasError(f"{what} 失败：{_STATUS.get(status, status)}（code={status}）")

    # ------------------------------------------------------------------
    # 流绑定
    # ------------------------------------------------------------------

    def set_stream(self, ptr) -> None:
        """把 handle 绑定到给定 stream（``None`` / ``0`` 表示默认流）。"""
        ptr = int(ptr or 0)
        if ptr == self._stream:
            return
        self._check(self._lib.cublasSetStream_v2(self._handle, ctypes.c_void_p(ptr)),
                    "cublasSetStream_v2")
        self._stream = ptr

    # ------------------------------------------------------------------
    # GEMM
    # ------------------------------------------------------------------

    def sgemm(self, a, b, out) -> None:
        """行主序 ``out[M, N] = a[M, K] @ b[K, N]``。"""
        m, k = int(a.shape[-2]), int(a.shape[-1])
        n = int(out.shape[-1])
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        # 列主序视角：out^T(N,M) = b^T @ a^T，见模块文档。
        self._check(
            self._lib.cublasSgemm_v2(
                self._handle, _CUBLAS_OP_N, _CUBLAS_OP_N,
                n, m, k,
                ctypes.byref(alpha), _ptr(b), n,
                _ptr(a), k,
                ctypes.byref(beta), _ptr(out), n,
            ),
            "cublasSgemm_v2",
        )

    def sgemm_batched(self, a, b, out, batch: int,
                      stride_a: int, stride_b: int, stride_c: int) -> None:
        """行主序批量 GEMM，stride 以元素为单位；0 表示整批共用一份数据。"""
        m, k = int(a.shape[-2]), int(a.shape[-1])
        n = int(out.shape[-1])
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        self._check(
            self._lib.cublasSgemmStridedBatched(
                self._handle, _CUBLAS_OP_N, _CUBLAS_OP_N,
                n, m, k,
                ctypes.byref(alpha), _ptr(b), n, stride_b,
                _ptr(a), k, stride_a,
                ctypes.byref(beta), _ptr(out), n, stride_c,
                batch,
            ),
            "cublasSgemmStridedBatched",
        )

    def hgemm(self, a, b, out) -> None:
        """行主序 f16 GEMM：``out[M, N] = a[M, K] @ b[K, N]``，内部 f32 累加。"""
        m, k = int(a.shape[-2]), int(a.shape[-1])
        n = int(out.shape[-1])
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        self._check(
            self._lib.cublasGemmEx(
                self._handle, _CUBLAS_OP_N, _CUBLAS_OP_N,
                n, m, k,
                ctypes.byref(alpha),
                _ptr(b), _CUDA_R_16F, n,
                _ptr(a), _CUDA_R_16F, k,
                ctypes.byref(beta),
                _ptr(out), _CUDA_R_16F, n,
                _CUBLAS_COMPUTE_32F, _CUBLAS_GEMM_DEFAULT,
            ),
            "cublasGemmEx",
        )

    def hgemm_batched(self, a, b, out, batch: int,
                      stride_a: int, stride_b: int, stride_c: int) -> None:
        """行主序 f16 批量 GEMM，stride 以元素为单位；0 表示整批共用一份数据。"""
        m, k = int(a.shape[-2]), int(a.shape[-1])
        n = int(out.shape[-1])
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        self._check(
            self._lib.cublasGemmStridedBatchedEx(
                self._handle, _CUBLAS_OP_N, _CUBLAS_OP_N,
                n, m, k,
                ctypes.byref(alpha),
                _ptr(b), _CUDA_R_16F, n, stride_b,
                _ptr(a), _CUDA_R_16F, k, stride_a,
                ctypes.byref(beta),
                _ptr(out), _CUDA_R_16F, n, stride_c,
                batch,
                _CUBLAS_COMPUTE_32F, _CUBLAS_GEMM_DEFAULT,
            ),
            "cublasGemmStridedBatchedEx",
        )

    def close(self) -> None:
        if self._handle is not None:
            self._lib.cublasDestroy_v2(self._handle)
            self._handle = None
            self._stream = None
