"""cuSPARSE SpMM：``sparse_linear`` 的一条可选快路径，不是唯一实现。

``sparse_linear`` 的三件套 ``(indices, values, offsets)`` 本来就是 CSR：
``offsets`` 是行起点、``indices`` 是列号、``values`` 是非零值。因此
"CSR 事件 × 稠密权重"与 cuSPARSE 的 SpMM 是同一个运算，5.173 已经在
``benchmarks/sparse_compare.py`` 里用 ``cupyx.scipy.sparse`` 做过内核级对照
（自研散射比 cuSPARSE 快 1.73×～2.38×）；本模块把那条约对照用的路接成
**生产路径的可选实现**，让"事件规模大到自研核不再占优"时有另一条路可走。

**默认仍走自研核**（``sparse_impl="custom"``）。两个理由：

1. 自研核带越界检查——``indices`` 越界在设备端记 ``err``、主机侧抛错。
   库不做逐元素检查，越界读在设备上是未定义行为。默认路径不能把这条
   检查丢掉。
2. 5.173 的实测在四个形状上都是自研核更快；把库路径设成默认，等于用
   一个"可能更慢、且少一道检查"的实现替换已验证的路径。

覆盖范围刻意收窄：**f32 值 + i32 索引 + 行主序稠密矩阵**。f16 与 f64
不登记（``CusparseUnsupported``），索引只认 i32。这三条都是"没有真机
验证过的形态不放开"，而不是"库不支持"：cuSPARSE 本身支持 f16/f64 与
i64，放开它们需要各自的对拍数据，当前的 CPU-only 环境给不出。

与 ``cudnn.py`` 同一个定位：它仍然属于 ADR-004 的范畴——IR 里的
``sparse_linear`` 依旧是语义完整的 PORTABLE 算子，"用 cuSPARSE 实现它"
是后端内部的实现选择，不产生 LIBRARY 节点。
"""

from __future__ import annotations

import ctypes
import ctypes.util
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "CusparseUnavailable",
    "CusparseUnsupported",
    "available",
    "data_type",
    "index_type",
    "CusparseSpMM",
]

# cudaDataType_t
_CUDA_R_32F = 0
# cusparseIndexType_t：16U=1 / 32I=2 / 64I=3 / 8I=4。写成 1 会被当成 uint16
# 去读 int32 的 rowOffsets，越界读在真机上就是 cudaErrorIllegalAddress。
_CUSPARSE_INDEX_32I = 2
# cusparseIndexBase_t
_CUSPARSE_INDEX_BASE_ZERO = 0
# cusparseOperation_t
_CUSPARSE_OPERATION_NON_TRANSPOSE = 0
# cusparseOrder_t
_CUSPARSE_ORDER_ROW = 2
# cusparseSpMMAlg_t：0 是跨版本都在的默认算法。CSR_ALG1 / ALG2 在新旧版本
# 之间编号有变动，选具体算法会让"同一份代码在 11.x 与 12.x 上跑不同的核"。
_CUSPARSE_SPMM_ALG_DEFAULT = 0
_CUSPARSE_STATUS_SUCCESS = 0

# 这张表就是本模块的覆盖声明：不在这里的 dtype / 索引类型显式拒绝。
_DATA_TYPES = {np.dtype("float32"): _CUDA_R_32F}
_INDEX_TYPES = {np.dtype("int32"): _CUSPARSE_INDEX_32I}

_CANDIDATES = (
    "libcusparse.so.12",
    "libcusparse.so.11",
    "libcusparse.so",
    "cusparse64_12.dll",
    "cusparse64_11.dll",
    "cusparse64_10.dll",
)

_loaded: Optional[Tuple[Any, str]] = None
_load_error: Optional[str] = None


class CusparseUnavailable(RuntimeError):
    """系统里没有可用的 cuSPARSE（或版本太旧，缺 SPMM 接口）。"""


class CusparseUnsupported(ValueError):
    """库在，但这个 dtype / 索引类型不在本模块登记的覆盖范围内。"""


def data_type(dtype: Any) -> int:
    """numpy dtype → ``cudaDataType_t``；未登记的类型显式拒绝。"""
    key = np.dtype(dtype)
    try:
        return _DATA_TYPES[key]
    except KeyError:
        raise CusparseUnsupported(
            f"cuSPARSE 快路径只登记了 {sorted(str(k) for k in _DATA_TYPES)}，"
            f"收到 {key}；请改用 sparse_impl='custom'"
        ) from None


def index_type(dtype: Any) -> int:
    """numpy dtype → ``cusparseIndexType_t``；只认 int32。"""
    key = np.dtype(dtype)
    try:
        return _INDEX_TYPES[key]
    except KeyError:
        raise CusparseUnsupported(
            f"cuSPARSE 快路径的 CSR 索引只登记了 "
            f"{sorted(str(k) for k in _INDEX_TYPES)}，收到 {key}；"
            "i64 索引需要单独的对拍数据，当前不放开"
        ) from None


def _load() -> Tuple[Any, str]:
    """加载 libcusparse；失败抛 ``CusparseUnavailable``（结果缓存，只试一次）。"""
    global _loaded, _load_error
    if _loaded is not None:
        return _loaded
    if _load_error is not None:
        raise CusparseUnavailable(_load_error)

    names = list(_CANDIDATES)
    found = ctypes.util.find_library("cusparse")
    if found:
        names.insert(0, found)
    try:  # pip install nvidia-cusparse-cu12 的包内 so 不在 ldconfig 路径里
        import nvidia.cusparse as _pkg

        for root in list(getattr(_pkg, "__path__", []) or []):
            names.append(root + "/lib/libcusparse.so.12")
    except ImportError:
        pass

    tried = []
    for name in names:
        try:
            _loaded = (ctypes.CDLL(name), name)
            return _loaded
        except OSError as exc:
            tried.append(f"{name}: {exc}")
    _load_error = "找不到 cuSPARSE 动态库，尝试过：\n  " + "\n  ".join(tried)
    raise CusparseUnavailable(_load_error)


def available() -> bool:
    """只回答"能不能加载"，不建 handle，也不碰 GPU。"""
    try:
        _load()
    except CusparseUnavailable:
        return False
    return True


class CusparseSpMM:
    """一个进程内复用的 cuSPARSE handle + workspace 缓存。

    **描述符每次调用重建，workspace 按形状缓存。** 描述符里写死的是
    ``offsets`` / ``indices`` / ``values`` / 权重 / 输出的地址，而这些数组
    每一步都可能换（同一张图换一批事件就是新数组）；``cusparseCsrSetPointers``
    能改地址，但那是另一条没有真机验证过的路径。workspace 的大小只由
    ``(行, 列, 宽, nnz, 算法)`` 决定，与地址无关，因此可以安全缓存。
    """

    def __init__(self) -> None:
        lib, self.name = _load()
        self.lib = lib
        vp = ctypes.c_void_p
        i64 = ctypes.c_int64

        required = (
            "cusparseCreate",
            "cusparseSetStream",
            "cusparseCreateCsr",
            "cusparseCreateDnMat",
            "cusparseSpMM_bufferSize",
            "cusparseSpMM",
            "cusparseDestroySpMat",
            "cusparseDestroyDnMat",
            "cusparseDestroy",
        )
        missing = [name for name in required if not hasattr(lib, name)]
        if missing:
            raise CusparseUnavailable(
                f"{self.name} 缺少 SpMM 接口 {missing}；"
                "cusparseCreateCsr / cusparseSpMM 需要 CUDA 11 及以上"
            )

        lib.cusparseCreate.argtypes = [ctypes.POINTER(vp)]
        lib.cusparseSetStream.argtypes = [vp, vp]
        lib.cusparseCreateCsr.argtypes = [
            ctypes.POINTER(vp), i64, i64, i64,
            vp, vp, vp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        lib.cusparseCreateDnMat.argtypes = [
            ctypes.POINTER(vp), i64, i64, i64, vp, ctypes.c_int, ctypes.c_int,
        ]
        lib.cusparseSpMM_bufferSize.argtypes = [
            vp, ctypes.c_int, ctypes.c_int, vp, vp, vp, vp, vp,
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.cusparseSpMM.argtypes = [
            vp, ctypes.c_int, ctypes.c_int, vp, vp, vp, vp, vp,
            ctypes.c_int, ctypes.c_int, vp,
        ]
        lib.cusparseDestroySpMat.argtypes = [vp]
        lib.cusparseDestroyDnMat.argtypes = [vp]
        lib.cusparseDestroy.argtypes = [vp]

        self.handle = vp()
        self._check(lib.cusparseCreate(ctypes.byref(self.handle)), "cusparseCreate")
        self._stream_ptr: Optional[int] = None
        # key -> 字节数；workspace 数组另存一份，避免每次 SpMM 都重新分配
        self._sizes: Dict[tuple, int] = {}
        self._workspaces: Dict[int, Any] = {}

    # ------------------------------------------------------------------

    @staticmethod
    def _check(status: int, what: str) -> None:
        if status != _CUSPARSE_STATUS_SUCCESS:
            raise RuntimeError(f"cuSPARSE {what} 失败，cusparseStatus_t={status}")

    @staticmethod
    def plan_key(steps: int, rows: int, cols: int, nnz: int, dtype: Any) -> tuple:
        """workspace 的缓存键：只含决定 workspace 大小的量。

        ``nnz`` 进键是必须的——SpMM 的 workspace 随非零数增长，同一形状的
        两张图事件数不同却命中同一个键，会拿到装不下的缓冲。
        """
        return (int(steps), int(rows), int(cols), int(nnz), np.dtype(dtype))

    def set_stream(self, ptr: int) -> None:
        """绑定 stream；与 cuBLAS / cuDNN 一样做缓存，减少驱动调用。"""
        ptr = int(ptr or 0)
        if ptr == self._stream_ptr:
            return
        self._check(
            self.lib.cusparseSetStream(self.handle, ctypes.c_void_p(ptr)),
            "cusparseSetStream",
        )
        self._stream_ptr = ptr

    def _workspace(self, width: int):
        if width == 0:
            return None
        buf = self._workspaces.get(width)
        if buf is None:
            import cupy as cp

            buf = cp.empty(width, dtype=cp.uint8)
            self._workspaces[width] = buf
        return buf

    def _descriptors(self, offsets, indices, values, weights, out, dtype: int,
                     rows: int, cols: int, steps: int, nnz: int):
        """建 A / B / C 三个描述符。调用方负责销毁。"""
        lib, vp = self.lib, ctypes.c_void_p
        a, b, c = vp(), vp(), vp()
        self._check(
            lib.cusparseCreateCsr(
                ctypes.byref(a), steps, rows, nnz,
                vp(int(offsets.data.ptr)), vp(int(indices.data.ptr)),
                vp(int(values.data.ptr)),
                _CUSPARSE_INDEX_32I, _CUSPARSE_INDEX_32I,
                _CUSPARSE_INDEX_BASE_ZERO, dtype,
            ),
            "cusparseCreateCsr",
        )
        try:
            # B 是 (rows, cols) 的行主序稠密矩阵，C 是 (steps, cols)：行主序
            # 的 ld 就是列数。用行主序而不是"转置成列主序"，是因为两个
            # 操作数都已经是 C 连续——转置只会多一次主机侧的心算。
            self._check(
                lib.cusparseCreateDnMat(
                    ctypes.byref(b), rows, cols, cols,
                    vp(int(weights.data.ptr)), dtype, _CUSPARSE_ORDER_ROW,
                ),
                "cusparseCreateDnMat(B)",
            )
            try:
                self._check(
                    lib.cusparseCreateDnMat(
                        ctypes.byref(c), steps, cols, cols,
                        vp(int(out.data.ptr)), dtype, _CUSPARSE_ORDER_ROW,
                    ),
                    "cusparseCreateDnMat(C)",
                )
            except Exception:
                self._check(lib.cusparseDestroyDnMat(b), "cusparseDestroyDnMat(B)")
                raise
        except Exception:
            self._check(lib.cusparseDestroySpMat(a), "cusparseDestroySpMat(A)")
            raise
        return a, b, c

    # ------------------------------------------------------------------

    def spmm(self, offsets, indices, values, weights, *,
             steps: int, rows: int, cols: int):
        """``out = A(steps×rows) · weights(rows×cols)``，返回新的 ``out``。

        三件套与权重必须是**已经连续**的设备数组：本模块不做布局转换，
        非连续输入在 ``.data.ptr`` 上会读错。dtype / 索引类型先过登记表，
        未登记的形态在碰驱动之前就抛 ``CusparseUnsupported``。
        """
        import cupy as cp

        dtype = data_type(values.dtype)
        index_type(indices.dtype)
        index_type(offsets.dtype)
        nnz = int(values.size)
        out = cp.empty((steps, cols), dtype=values.dtype)
        if steps == 0 or cols == 0:
            return out

        self.set_stream(int(cp.cuda.get_current_stream().ptr))
        a, b, c = self._descriptors(
            offsets, indices, values, weights, out, dtype,
            rows, cols, steps, nnz,
        )
        try:
            key = self.plan_key(steps, rows, cols, nnz, values.dtype)
            width = self._sizes.get(key)
            alpha = ctypes.c_float(1.0)
            beta = ctypes.c_float(0.0)
            if width is None:
                size = ctypes.c_size_t(0)
                self._check(
                    self.lib.cusparseSpMM_bufferSize(
                        self.handle,
                        _CUSPARSE_OPERATION_NON_TRANSPOSE,
                        _CUSPARSE_OPERATION_NON_TRANSPOSE,
                        ctypes.byref(alpha), a, b, ctypes.byref(beta), c,
                        _CUDA_R_32F, _CUSPARSE_SPMM_ALG_DEFAULT,
                        ctypes.byref(size),
                    ),
                    "cusparseSpMM_bufferSize",
                )
                width = int(size.value)
                self._sizes[key] = width
            workspace = self._workspace(width)
            self._check(
                self.lib.cusparseSpMM(
                    self.handle,
                    _CUSPARSE_OPERATION_NON_TRANSPOSE,
                    _CUSPARSE_OPERATION_NON_TRANSPOSE,
                    ctypes.byref(alpha), a, b,
                    ctypes.byref(beta), c,
                    _CUDA_R_32F, _CUSPARSE_SPMM_ALG_DEFAULT,
                    ctypes.c_void_p(0) if workspace is None
                    else ctypes.c_void_p(int(workspace.data.ptr)),
                ),
                "cusparseSpMM",
            )
        finally:
            self._check(self.lib.cusparseDestroySpMat(a), "cusparseDestroySpMat(A)")
            self._check(self.lib.cusparseDestroyDnMat(b), "cusparseDestroyDnMat(B)")
            self._check(self.lib.cusparseDestroyDnMat(c), "cusparseDestroyDnMat(C)")
        return out
