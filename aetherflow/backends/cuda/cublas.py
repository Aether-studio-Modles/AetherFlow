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
_CUBLAS_OP_T = 1

# cudaDataType_t / cublasComputeType_t / cublasGemmAlgo_t 的常量。
# 这些值在 CUDA 11 与 12 的 ABI 里是稳定的；不 include 头文件，直接按文档写。
_CUDA_R_16F = 2
_CUDA_R_32F = 0
# CUDA_R_16BF：bf16 的 cudaDataType_t。设备上的 bf16 张量以 uint16 位模式
# 驻留，位布局与 CUDA_R_16BF 完全一致，因此可以直接交给 cuBLAS。
_CUDA_R_16BF = 14
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

        # 预置 workspace：CUDA Graph 捕获期间 cuBLAS 不能自己分配内部
        # workspace，否则会退到不需要 workspace 的慢算法。提前给一块常驻
        # 的，捕获与 eager 走同一条算法路径。大小由调用方决定。
        lib.cublasSetWorkspace_v2.argtypes = [vp, vp, ctypes.c_size_t]
        lib.cublasSetWorkspace_v2.restype = ci

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

    def set_workspace(self, ptr: int, size: int) -> None:
        """预置一块常驻 workspace（CUDA Graph 捕获期不能分配内部 workspace）。"""
        if size <= 0:
            raise ValueError("cuBLAS workspace 大小必须为正")
        self._check(
            self._lib.cublasSetWorkspace_v2(
                self._handle, ctypes.c_void_p(ptr), ctypes.c_size_t(size)
            ),
            "cublasSetWorkspace_v2",
        )

    def gemm_transposed(self, a, b, out, trans_a=False, trans_b=False) -> None:
        """连续二维源矩阵的逻辑转置直接交给 GEMM，不分配转置副本。"""
        m, n = (int(d) for d in out.shape)
        k = int(a.shape[-2] if trans_a else a.shape[-1])
        op_a = _CUBLAS_OP_T if trans_a else _CUBLAS_OP_N
        op_b = _CUBLAS_OP_T if trans_b else _CUBLAS_OP_N
        lda, ldb = int(a.shape[-1]), int(b.shape[-1])
        alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)
        # 行主序交换左右操作数；leading dimension 始终取物理源矩阵行宽。
        dtype = str(a.dtype)
        if dtype == "float32":
            status = self._lib.cublasSgemm_v2(
                self._handle, op_b, op_a, n, m, k,
                ctypes.byref(alpha), _ptr(b), ldb, _ptr(a), lda,
                ctypes.byref(beta), _ptr(out), n,
            )
        else:
            storage = _CUDA_R_16F if dtype == "float16" else _CUDA_R_16BF
            status = self._lib.cublasGemmEx(
                self._handle, op_b, op_a, n, m, k,
                ctypes.byref(alpha), _ptr(b), storage, ldb,
                _ptr(a), storage, lda, ctypes.byref(beta),
                _ptr(out), storage, n, _CUBLAS_COMPUTE_32F, _CUBLAS_GEMM_DEFAULT,
            )
        self._check(status, "转置 GEMM")

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

    def bgemm(self, a, b, out) -> None:
        """行主序 bf16 GEMM：``out[M, N] = a[M, K] @ b[K, N]``，内部 f32 累加。

        输入输出都是两字节 bf16 位模式（设备上是 ``uint16`` 容器），
        ``CUBLAS_COMPUTE_32F`` 表示乘加在 float 里完成——与内核里"读成 float
        再算"一致。A800 上会落到 HMMA.bf16 的 Tensor Core 路径。
        """
        m, k = int(a.shape[-2]), int(a.shape[-1])
        n = int(out.shape[-1])
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        self._check(
            self._lib.cublasGemmEx(
                self._handle, _CUBLAS_OP_N, _CUBLAS_OP_N,
                n, m, k,
                ctypes.byref(alpha),
                _ptr(b), _CUDA_R_16BF, n,
                _ptr(a), _CUDA_R_16BF, k,
                ctypes.byref(beta),
                _ptr(out), _CUDA_R_16BF, n,
                _CUBLAS_COMPUTE_32F, _CUBLAS_GEMM_DEFAULT,
            ),
            "cublasGemmEx(bf16)",
        )

    def bgemm_batched(self, a, b, out, batch: int,
                      stride_a: int, stride_b: int, stride_c: int) -> None:
        """行主序 bf16 批量 GEMM，stride 以元素为单位；0 表示整批共用一份数据。"""
        m, k = int(a.shape[-2]), int(a.shape[-1])
        n = int(out.shape[-1])
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        self._check(
            self._lib.cublasGemmStridedBatchedEx(
                self._handle, _CUBLAS_OP_N, _CUBLAS_OP_N,
                n, m, k,
                ctypes.byref(alpha),
                _ptr(b), _CUDA_R_16BF, n, stride_b,
                _ptr(a), _CUDA_R_16BF, k, stride_a,
                ctypes.byref(beta),
                _ptr(out), _CUDA_R_16BF, n, stride_c,
                batch,
                _CUBLAS_COMPUTE_32F, _CUBLAS_GEMM_DEFAULT,
            ),
            "cublasGemmStridedBatchedEx(bf16)",
        )

    def close(self) -> None:
        if self._handle is not None:
            self._lib.cublasDestroy_v2(self._handle)
            self._handle = None
            self._stream = None
