"""hipBLAS 薄绑定（ctypes）。

与 ``cuda/cublas.py`` 一一对应：只绑 float32 / float16 的**列主序** GEMM，
行主序通过 ``C^T = B^T A^T`` 互换 A/B 得到。f16 走 ``hipblasGemmEx``，
计算类型固定 ``HIPBLAS_COMPUTE_32F``——存储半精度、累加 float，与内核里
"读成 float 再算"的策略一致。

常量是直接从 ROCm 7.2 的头文件里核对的（``hipblas-common.h`` 与
``hip/library_types.h``）：hipBLAS 的枚举值与 cuBLAS **不同**，
照抄 CUDA 的数字会静默算错或报参数错误，因此这里逐项注明来源等价物。
"""

from __future__ import annotations

import ctypes
import ctypes.util

# hipblasOperation_t：HIPBLAS_OP_N = 111（hipblas-common.h）
_HIPBLAS_OP_N = 111

# hipDataType：HIP_R_32F = 0、HIP_R_16F = 2（hip/library_types.h），与 cudaDataType 相同
_HIP_R_16F = 2
_HIP_R_32F = 0

# hipblasComputeType_t：HIPBLAS_COMPUTE_32F = 2（hipblas-common.h）
# 注意与 cuBLAS 的 CUBLAS_COMPUTE_32F = 68 不同，这不是笔误。
_HIPBLAS_COMPUTE_32F = 2

# hipblasGemmAlgo_t：HIPBLAS_GEMM_DEFAULT = 160
_HIPBLAS_GEMM_DEFAULT = 160

_CANDIDATES = (
    "libhipblas.so.3",
    "libhipblas.so.2",
    "libhipblas.so",
    "hipblas.dll",
)

_STATUS = {
    0: "HIPBLAS_STATUS_SUCCESS",
    1: "HIPBLAS_STATUS_NOT_INITIALIZED",
    2: "HIPBLAS_STATUS_ALLOC_FAILED",
    3: "HIPBLAS_STATUS_INVALID_VALUE",
    4: "HIPBLAS_STATUS_MAPPING_ERROR",
    5: "HIPBLAS_STATUS_EXECUTION_FAILED",
    6: "HIPBLAS_STATUS_INTERNAL_ERROR",
    7: "HIPBLAS_STATUS_NOT_SUPPORTED",
    8: "HIPBLAS_STATUS_ARCH_MISMATCH",
    9: "HIPBLAS_STATUS_HANDLE_IS_NULLPTR",
}


class HipblasError(RuntimeError):
    """hipBLAS 返回非 SUCCESS 状态码。"""


def _load_library() -> ctypes.CDLL:
    names = []
    found = ctypes.util.find_library("hipblas")
    if found:
        names.append(found)
    names.extend(_CANDIDATES)
    tried = []
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError as exc:
            tried.append(f"{name}: {exc}")
    raise OSError("找不到 hipBLAS 动态库，尝试过：\n  " + "\n  ".join(tried))


def _ptr(array) -> ctypes.c_void_p:
    return ctypes.c_void_p(array.data.ptr)


class Hipblas:
    """进程内复用的 hipBLAS handle；stream 绑定做缓存，避免重复下发。"""

    def __init__(self) -> None:
        self._lib = _load_library()
        self._declare()
        handle = ctypes.c_void_p()
        self._check(self._lib.hipblasCreate(ctypes.byref(handle)), "hipblasCreate")
        self._handle = handle
        self._stream = None

    def _declare(self) -> None:
        lib = self._lib
        vp = ctypes.c_void_p
        fp = ctypes.POINTER(ctypes.c_float)
        ci = ctypes.c_int
        cll = ctypes.c_longlong

        lib.hipblasCreate.argtypes = [ctypes.POINTER(vp)]
        lib.hipblasCreate.restype = ci
        lib.hipblasDestroy.argtypes = [vp]
        lib.hipblasDestroy.restype = ci
        lib.hipblasSetStream.argtypes = [vp, vp]
        lib.hipblasSetStream.restype = ci

        lib.hipblasSgemm.argtypes = [
            vp, ci, ci, ci, ci, ci, fp, vp, ci, vp, ci, fp, vp, ci,
        ]
        lib.hipblasSgemm.restype = ci

        lib.hipblasSgemmStridedBatched.argtypes = [
            vp, ci, ci, ci, ci, ci, fp, vp, ci, cll, vp, ci, cll, fp, vp, ci, cll, ci,
        ]
        lib.hipblasSgemmStridedBatched.restype = ci

        # hipblasGemmEx：alpha/beta 是 void*（指向 compute type 的标量），
        # A/B/C 的类型由 hipDataType 参数给出。
        lib.hipblasGemmEx.argtypes = [
            vp, ci, ci, ci, ci, ci, fp, vp, ci, ci, vp, ci, ci, fp, vp, ci, ci, ci, ci,
        ]
        lib.hipblasGemmEx.restype = ci

        # hipblasStride 是 int64_t；其余与 GemmEx 同序，只是多了三个 stride。
        lib.hipblasGemmStridedBatchedEx.argtypes = [
            vp, ci, ci, ci, ci, ci, fp,
            vp, ci, ci, cll,
            vp, ci, ci, cll,
            fp, vp, ci, ci, cll,
            ci, ci, ci,
        ]
        lib.hipblasGemmStridedBatchedEx.restype = ci

    @staticmethod
    def _check(status: int, what: str) -> None:
        if status != 0:
            raise HipblasError(
                f"{what} 失败：{_STATUS.get(status, status)}（code={status}）"
            )

    def set_stream(self, ptr) -> None:
        ptr = int(ptr or 0)
        if ptr == self._stream:
            return
        self._check(
            self._lib.hipblasSetStream(self._handle, ctypes.c_void_p(ptr)),
            "hipblasSetStream",
        )
        self._stream = ptr

    def sgemm(self, a, b, out) -> None:
        """行主序 ``out[M, N] = a[M, K] @ b[K, N]``。"""
        m, k = int(a.shape[-2]), int(a.shape[-1])
        n = int(out.shape[-1])
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        self._check(
            self._lib.hipblasSgemm(
                self._handle, _HIPBLAS_OP_N, _HIPBLAS_OP_N,
                n, m, k,
                ctypes.byref(alpha), _ptr(b), n,
                _ptr(a), k,
                ctypes.byref(beta), _ptr(out), n,
            ),
            "hipblasSgemm",
        )

    def sgemm_batched(self, a, b, out, batch: int,
                      stride_a: int, stride_b: int, stride_c: int) -> None:
        """行主序批量 GEMM，stride 以元素为单位；0 表示整批共用一份数据。"""
        m, k = int(a.shape[-2]), int(a.shape[-1])
        n = int(out.shape[-1])
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        self._check(
            self._lib.hipblasSgemmStridedBatched(
                self._handle, _HIPBLAS_OP_N, _HIPBLAS_OP_N,
                n, m, k,
                ctypes.byref(alpha), _ptr(b), n, stride_b,
                _ptr(a), k, stride_a,
                ctypes.byref(beta), _ptr(out), n, stride_c,
                batch,
            ),
            "hipblasSgemmStridedBatched",
        )

    def hgemm(self, a, b, out) -> None:
        """行主序 f16 GEMM：存储半精度，内部按 HIPBLAS_COMPUTE_32F 累加。"""
        m, k = int(a.shape[-2]), int(a.shape[-1])
        n = int(out.shape[-1])
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        self._check(
            self._lib.hipblasGemmEx(
                self._handle, _HIPBLAS_OP_N, _HIPBLAS_OP_N,
                n, m, k,
                ctypes.byref(alpha),
                _ptr(b), _HIP_R_16F, n,
                _ptr(a), _HIP_R_16F, k,
                ctypes.byref(beta),
                _ptr(out), _HIP_R_16F, n,
                _HIPBLAS_COMPUTE_32F, _HIPBLAS_GEMM_DEFAULT,
            ),
            "hipblasGemmEx",
        )

    def hgemm_batched(self, a, b, out, batch: int,
                      stride_a: int, stride_b: int, stride_c: int) -> None:
        """行主序 f16 批量 GEMM，stride 以元素为单位；0 表示整批共用一份数据。"""
        m, k = int(a.shape[-2]), int(a.shape[-1])
        n = int(out.shape[-1])
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        self._check(
            self._lib.hipblasGemmStridedBatchedEx(
                self._handle, _HIPBLAS_OP_N, _HIPBLAS_OP_N,
                n, m, k,
                ctypes.byref(alpha),
                _ptr(b), _HIP_R_16F, n, stride_b,
                _ptr(a), _HIP_R_16F, k, stride_a,
                ctypes.byref(beta),
                _ptr(out), _HIP_R_16F, n, stride_c,
                batch,
                _HIPBLAS_COMPUTE_32F, _HIPBLAS_GEMM_DEFAULT,
            ),
            "hipblasGemmStridedBatchedEx",
        )

    def close(self) -> None:
        if self._handle is not None:
            self._lib.hipblasDestroy(self._handle)
            self._handle = None
            self._stream = None
