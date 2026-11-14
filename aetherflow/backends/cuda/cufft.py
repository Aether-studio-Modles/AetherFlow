"""cuFFT：``fft`` / ``rfft`` 的 CUDA 实现，不是唯一实现。

IR 里的 ``fft`` / ``rfft`` 是 PORTABLE 算子——参考后端用 ``numpy.fft`` 给出
唯一真值。本模块只回答"在 NVIDIA 上怎么算这一个节点"，与 ``cudnn.py`` /
``cusparse.py`` 同一个定位：不产生 LIBRARY 节点，也不改变 IR 的语义。

覆盖范围：

- **只做沿最后一维的一维变换**，批量维靠 ``cufftPlanMany`` 的 batch 铺开。
  IR 算子本身就只定义了最后一维，多维 FFT 需要新的轴属性与新的参考语义，
  不在这一步里顺带扩。
- dtype 只认 c64 / c128（``fft``）与 f32 / f64（``rfft``），四者分别对应
  ``CUFFT_C2C`` / ``CUFFT_Z2Z`` / ``CUFFT_R2C`` / ``CUFFT_D2Z``。未登记的
  组合显式抛 ``CufftUnsupported``。
- **输入必须连续**。``cufftPlanMany`` 的 stride/dist 虽然能表达非连续布局，
  但"哪些步长组合能高效"取决于具体架构，没有真机数据就不猜；非连续输入
  由调用方先 ``ascontiguousarray``。

归一化：cuFFT 两个方向都做未归一化变换（正向不除 N，逆向也不除 N）。因此
逆变换的结果是 numpy ``ifft`` 的 N 倍，必须由这里显式乘一个缩放因子。
缩放表与 numpy 的 ``norm`` 三档一一对应，见 ``normalization``——它是纯函数，
可以离线测试，不依赖任何驱动。
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "CufftUnavailable",
    "CufftUnsupported",
    "available",
    "kind",
    "normalization",
    "CufftPlan",
]

# cufftType_t
_CUFFT_R2C = 0x2A
_CUFFT_C2C = 0x29
_CUFFT_D2Z = 0x6A
_CUFFT_Z2Z = 0x69
# cufftDirection
_CUFFT_FORWARD = -1
_CUFFT_INVERSE = 1
_CUFFT_SUCCESS = 0

# 这张表就是本模块的覆盖声明：不在这里的 dtype 显式拒绝。
_KINDS = {
    (np.dtype("complex64"), False): _CUFFT_C2C,
    (np.dtype("complex64"), True): _CUFFT_C2C,
    (np.dtype("complex128"), False): _CUFFT_Z2Z,
    (np.dtype("complex128"), True): _CUFFT_Z2Z,
    (np.dtype("float32"), False): _CUFFT_R2C,
    (np.dtype("float64"), False): _CUFFT_D2Z,
}

_CANDIDATES = (
    "libcufft.so.11",
    "libcufft.so.10",
    "libcufft.so",
    "cufft64_11.dll",
    "cufft64_10.dll",
)

_loaded: Optional[Tuple[Any, str]] = None
_load_error: Optional[str] = None


class CufftUnavailable(RuntimeError):
    """系统里没有可用的 cuFFT（或版本太旧，缺 PlanMany 接口）。"""


class CufftUnsupported(ValueError):
    """库在，但这个 dtype / 方向不在本模块登记的覆盖范围内。"""


def kind(dtype: Any, inverse: bool = False) -> int:
    """``(numpy dtype, 是否逆变换)`` → ``cufftType_t``；未登记即拒绝。

    ``rfft`` 是实数进复数出，只有正向；传 ``inverse=True`` 会在这里被拒，
    而不是在 cuFFT 里产生一个含义不明的结果。
    """
    key = (np.dtype(dtype), bool(inverse))
    try:
        return _KINDS[key]
    except KeyError:
        raise CufftUnsupported(
            f"cuFFT 路径只登记了 c64/c128 的 FFT 与 f32/f64 的 RFFT"
            f"（逆变换只对 c64/c128 开放），收到 dtype={np.dtype(dtype)}、"
            f"inverse={bool(inverse)}"
        ) from None


def normalization(size: int, norm: str, inverse: bool = False) -> float:
    """cuFFT 结果要乘的缩放因子，使它与 numpy 同 ``norm`` 的结果一致。

    cuFFT 的逆变换是未归一化的（结果是 numpy ``ifft`` 的 N 倍），正向也是
    未归一化的（与 numpy ``norm="backward"`` 相同）。于是：

    ==========  =========  =======  =======
    norm        正向        逆变换
    ==========  =========  =======  =======
    backward    1           1/N
    ortho       1/√N        1/√N
    forward     1/N         1
    ==========  =========  =======  =======

    参数非法时显式报错：静默回落到默认档位会让"数值差了 N 倍"变得极难定位。
    """
    if norm not in ("backward", "ortho", "forward"):
        raise ValueError(f"norm 只能是 backward / ortho / forward，得到 {norm!r}")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"变换长度必须是正整数，得到 {size!r}")
    if norm == "backward":
        exponent = 1.0 if inverse else 0.0
    elif norm == "ortho":
        exponent = 0.5
    else:
        exponent = 0.0 if inverse else 1.0
    return 1.0 / (float(size) ** exponent)


def _load() -> Tuple[Any, str]:
    """加载 libcufft；失败抛 ``CufftUnavailable``（结果缓存，只试一次）。"""
    global _loaded, _load_error
    if _loaded is not None:
        return _loaded
    if _load_error is not None:
        raise CufftUnavailable(_load_error)

    names = list(_CANDIDATES)
    found = ctypes.util.find_library("cufft")
    if found:
        names.insert(0, found)
    try:  # pip install nvidia-cufft-cu12 的包内 so 不在 ldconfig 路径里
        import nvidia.cufft as _pkg

        for root in list(getattr(_pkg, "__path__", []) or []):
            names.append(root + "/lib/libcufft.so.11")
    except ImportError:
        pass

    tried = []
    for name in names:
        try:
            _loaded = (ctypes.CDLL(name), name)
            return _loaded
        except OSError as exc:
            tried.append(f"{name}: {exc}")
    _load_error = "找不到 cuFFT 动态库，尝试过：\n  " + "\n  ".join(tried)
    raise CufftUnavailable(_load_error)


def available() -> bool:
    """只回答"能不能加载"，不建 plan，也不碰 GPU。"""
    try:
        _load()
    except CufftUnavailable:
        return False
    return True


class CufftPlan:
    """一个进程内复用的 cuFFT handle 池，plan 按形状缓存。

    plan 里写死的是变换长度、batch 数与方向，**不含地址**：每次执行把新的
    输入 / 输出指针传给 ``cufftExec*`` 即可。这些量正好是"同一张图每次执行
    都不变"的部分，所以可以安全缓存；地址每个节点都可能换，不能进键。

    ``cufftGetVersion`` 只在能拿到时记录，用于诊断；拿不到不影响执行。
    """

    def __init__(self) -> None:
        lib, self.name = _load()
        self.lib = lib
        i32 = ctypes.c_int
        i64 = ctypes.c_longlong

        required = (
            "cufftPlanMany",
            "cufftSetStream",
            "cufftExecC2C",
            "cufftExecZ2Z",
            "cufftExecR2C",
            "cufftExecD2Z",
            "cufftDestroy",
        )
        missing = [name for name in required if not hasattr(lib, name)]
        if missing:
            raise CufftUnavailable(
                f"{self.name} 缺少接口 {missing}；cufftPlanMany 需要 CUDA 8 及以上"
            )

        vp = ctypes.c_void_p
        lib.cufftPlanMany.argtypes = [
            ctypes.POINTER(i32), i32, ctypes.POINTER(i32),
            ctypes.POINTER(i32), i32, i32,
            ctypes.POINTER(i32), i32, i32,
            i32, i32,
        ]
        lib.cufftSetStream.argtypes = [i32, vp]
        for fn, pointer in (
            ("cufftExecC2C", ctypes.POINTER(ctypes.c_float)),
            ("cufftExecZ2Z", ctypes.POINTER(ctypes.c_double)),
            ("cufftExecR2C", ctypes.POINTER(ctypes.c_float)),
            ("cufftExecD2Z", ctypes.POINTER(ctypes.c_double)),
        ):
            getattr(lib, fn).argtypes = [i32, pointer, pointer, i32]
        lib.cufftDestroy.argtypes = [i32]

        self._plans: Dict[tuple, int] = {}
        self._stream_ptr: Optional[int] = None
        self.version = self._version()

    def _version(self) -> Optional[int]:
        fn = getattr(self.lib, "cufftGetVersion", None)
        if fn is None:
            return None
        fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
        value = ctypes.c_int(0)
        if fn(ctypes.byref(value)) != _CUFFT_SUCCESS:
            return None
        return int(value.value)

    # ------------------------------------------------------------------

    @staticmethod
    def _check(status: int, what: str) -> None:
        if status != _CUFFT_SUCCESS:
            raise RuntimeError(f"cuFFT {what} 失败，cufftResult={status}")

    @staticmethod
    def plan_key(kind_id: int, size: int, rows: int) -> tuple:
        """plan 的缓存键：只含决定 plan 内容的量，与地址无关。

        ``rows`` 是 batch 数（把前面的维拉平），它进键是因为 ``cufftPlanMany``
        把 batch 写进了 plan。**方向不进键**：``cufftExecC2C`` 把方向作为
        执行参数，同一个 plan 正反两个方向共用。
        """
        return (int(kind_id), int(size), int(rows))

    def set_stream(self, ptr: int) -> None:
        """绑定 stream。与 cuBLAS / cuSPARSE 一样缓存，减少驱动调用。"""
        ptr = int(ptr or 0)
        if ptr == self._stream_ptr:
            return
        # stream 是 plan 级属性：换流要把已建 plan 全部重设一遍。
        for handle in self._plans.values():
            self._check(
                self.lib.cufftSetStream(handle, ctypes.c_void_p(ptr)),
                "cufftSetStream",
            )
        self._stream_ptr = ptr

    def _plan(self, kind_id: int, size: int, rows: int) -> int:
        key = self.plan_key(kind_id, size, rows)
        handle = self._plans.get(key)
        if handle is not None:
            return handle
        i32 = ctypes.c_int
        handle = i32(0)
        n = (i32 * 1)(size)
        out_size = size // 2 + 1 if kind_id in (_CUFFT_R2C, _CUFFT_D2Z) else size
        self._check(
            self.lib.cufftPlanMany(
                ctypes.byref(handle), 1, n,
                None, 1, size,          # inembed / istride / idist
                None, 1, out_size,      # onembed / ostride / odist
                kind_id, rows,
            ),
            "cufftPlanMany",
        )
        handle = int(handle.value)
        if self._stream_ptr:
            self._check(
                self.lib.cufftSetStream(handle, ctypes.c_void_p(self._stream_ptr)),
                "cufftSetStream",
            )
        self._plans[key] = handle
        return handle

    def close(self) -> None:
        for handle in self._plans.values():
            self.lib.cufftDestroy(handle)
        self._plans.clear()
        self._stream_ptr = None

    # ------------------------------------------------------------------

    def transform(self, src, *, size: int, rows: int, inverse: bool,
                  norm: str = "backward"):
        """对 ``src`` 的最后一维做变换，返回新的设备数组。

        ``src`` 必须是**连续**的设备数组；本模块不做布局转换——非连续输入
        在 ``data.ptr`` 上会被读成错误的行。dtype 与方向先过登记表，未登记
        的形态在碰驱动之前就抛 ``CufftUnsupported``。缩放因子在结果上原地
        应用一次，语义与 numpy 同 ``norm`` 一致。
        """
        import cupy as cp

        kind_id = kind(src.dtype, inverse)
        if not src.flags.c_contiguous:
            raise CufftUnsupported("cuFFT 路径要求输入连续（请先 ascontiguousarray）")
        out_size = size // 2 + 1 if kind_id in (_CUFFT_R2C, _CUFFT_D2Z) else size
        out_dtype = (
            np.dtype("complex64") if src.dtype == np.dtype("float32")
            else np.dtype("complex128") if src.dtype == np.dtype("float64")
            else np.dtype(src.dtype)
        )
        shape = tuple(src.shape[:-1]) + (out_size,)
        out = cp.empty(shape, dtype=out_dtype)
        if src.size == 0:
            return out

        self.set_stream(int(cp.cuda.get_current_stream().ptr))
        handle = self._plan(kind_id, size, rows)
        direction = _CUFFT_INVERSE if inverse else _CUFFT_FORWARD
        exec_name, pointer = {
            _CUFFT_C2C: ("cufftExecC2C", ctypes.POINTER(ctypes.c_float)),
            _CUFFT_Z2Z: ("cufftExecZ2Z", ctypes.POINTER(ctypes.c_double)),
            _CUFFT_R2C: ("cufftExecR2C", ctypes.POINTER(ctypes.c_float)),
            _CUFFT_D2Z: ("cufftExecD2Z", ctypes.POINTER(ctypes.c_double)),
        }[kind_id]
        self._check(
            getattr(self.lib, exec_name)(
                handle,
                ctypes.cast(ctypes.c_void_p(int(src.data.ptr)), pointer),
                ctypes.cast(ctypes.c_void_p(int(out.data.ptr)), pointer),
                direction,
            ),
            exec_name,
        )
        scale = normalization(size, norm, inverse)
        if scale != 1.0:
            out *= scale
        return out
