"""cuDNN 前向卷积：一条可选的快路径，不是唯一实现。

为什么不做成"唯一实现"：cuDNN 只在装了 NVIDIA 库的机器上存在，而自研分块核
是 CUDA / HIP 共用的一份 L2 描述。所以后端默认 ``conv_impl="auto"``：能加载
libcudnn 就用它，加载不到或形状不支持就静默回落——回落只换实现，不换数值
语义，测试用同形状对拍盯住。

覆盖范围刻意收窄：NCHW、groups=1、float32 / float16、任意 stride/pad/dilation。
分组卷积与 f64 仍走 im2col：把 group 语义塞进这里会多出一条只在部分机器上
被验证过的路径，收益抵不上风险。

这仍然属于 ADR-004 的范畴：IR 里的 ``conv2d`` 依旧是语义完整的 PORTABLE
算子，"用 cuDNN 实现它"是后端内部的实现选择，不产生 LIBRARY 节点。
"""

from __future__ import annotations

import ctypes
import ctypes.util
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = ["CudnnConv", "CudnnUnavailable", "available"]

_CUDNN_DATA_FLOAT = 0
_CUDNN_DATA_HALF = 2
_CUDNN_TENSOR_NCHW = 0
_CUDNN_CROSS_CORRELATION = 1
_CUDNN_STATUS_SUCCESS = 0
# cudnnConvolutionMath_t：只允许 FP32 的 FMA 算法，禁止 TF32 张量核转换。
_CUDNN_FMA_MATH = 3

_SUPPORTED = {np.dtype("float32"): _CUDNN_DATA_FLOAT, np.dtype("float16"): _CUDNN_DATA_HALF}

_CANDIDATES = ("libcudnn.so.9", "libcudnn.so", "cudnn64_9.dll", "cudnn64_8.dll")

_loaded: Optional[Tuple[Any, str]] = None
_load_error: Optional[str] = None


class CudnnUnavailable(RuntimeError):
    """系统里没有可用的 cuDNN。"""


class _FwdAlgoPerf(ctypes.Structure):
    _fields_ = [
        ("algo", ctypes.c_int),
        ("status", ctypes.c_int),
        ("time", ctypes.c_float),
        ("memory", ctypes.c_size_t),
        ("determinism", ctypes.c_int),
        ("mathType", ctypes.c_int),
        ("reserved", ctypes.c_int * 3),
    ]


def _load() -> Tuple[Any, str]:
    """加载 libcudnn；失败抛 ``CudnnUnavailable``（结果缓存，只试一次）。"""
    global _loaded, _load_error
    if _loaded is not None:
        return _loaded
    if _load_error is not None:
        raise CudnnUnavailable(_load_error)

    names = list(_CANDIDATES)
    found = ctypes.util.find_library("cudnn")
    if found:
        names.insert(0, found)
    try:  # pip install nvidia-cudnn-cu12 的包内 so 不在 ldconfig 路径里
        import nvidia.cudnn as _pkg

        for root in list(getattr(_pkg, "__path__", []) or []):
            names.append(root + "/lib/libcudnn.so.9")
    except ImportError:
        pass

    tried = []
    for name in names:
        try:
            _loaded = (ctypes.CDLL(name), name)
            return _loaded
        except OSError as exc:
            tried.append(f"{name}: {exc}")
    _load_error = "找不到 cuDNN 动态库，尝试过：\n  " + "\n  ".join(tried)
    raise CudnnUnavailable(_load_error)


def available() -> bool:
    """只回答"能不能加载"，不建 handle，也不碰 GPU。"""
    try:
        _load()
    except CudnnUnavailable:
        return False
    return True


class CudnnConv:
    """一个进程内复用的 cuDNN handle + 描述符缓存。"""

    def __init__(self) -> None:
        lib, self.name = _load()
        self.lib = lib
        vp = ctypes.c_void_p
        pi = ctypes.POINTER(ctypes.c_int)

        lib.cudnnGetVersion.restype = ctypes.c_size_t
        lib.cudnnCreate.argtypes = [ctypes.POINTER(vp)]
        lib.cudnnSetStream.argtypes = [vp, vp]
        lib.cudnnCreateTensorDescriptor.argtypes = [ctypes.POINTER(vp)]
        lib.cudnnCreateFilterDescriptor.argtypes = [ctypes.POINTER(vp)]
        lib.cudnnCreateConvolutionDescriptor.argtypes = [ctypes.POINTER(vp)]
        lib.cudnnSetTensor4dDescriptor.argtypes = [
            vp, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        lib.cudnnSetFilter4dDescriptor.argtypes = [
            vp, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        lib.cudnnSetConvolutionNdDescriptor.argtypes = [
            vp, ctypes.c_int, pi, pi, pi, ctypes.c_int, ctypes.c_int,
        ]
        lib.cudnnSetConvolutionMathType.argtypes = [vp, ctypes.c_int]
        lib.cudnnGetConvolutionForwardAlgorithm_v7.argtypes = [
            vp, vp, vp, vp, vp, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(_FwdAlgoPerf),
        ]
        lib.cudnnGetConvolutionForwardWorkspaceSize.argtypes = [
            vp, vp, vp, vp, vp, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.cudnnConvolutionForward.argtypes = [
            vp, ctypes.c_void_p, vp, ctypes.c_void_p, vp, ctypes.c_void_p, vp,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, vp,
            ctypes.c_void_p,
        ]
        lib.cudnnAddTensor.argtypes = [vp, vp, vp, vp, vp, vp, vp]

        self.version = int(lib.cudnnGetVersion())
        self.handle = vp()
        self._check(lib.cudnnCreate(ctypes.byref(self.handle)), "cudnnCreate")
        self._stream_ptr: Optional[int] = None
        self._plans: Dict[tuple, Tuple[tuple, int, int]] = {}
        self._workspaces: Dict[int, Any] = {}

    # ------------------------------------------------------------------

    @staticmethod
    def _check(status: int, what: str) -> None:
        if status != _CUDNN_STATUS_SUCCESS:
            raise RuntimeError(f"cuDNN {what} 失败，cudnnStatus_t={status}")

    def set_stream(self, ptr: int) -> None:
        """绑定 stream；与 cuBLAS 一样做缓存，避免每次派发多一次驱动调用。"""
        ptr = int(ptr or 0)
        if ptr == self._stream_ptr:
            return
        self._check(
            self.lib.cudnnSetStream(self.handle, ctypes.c_void_p(ptr)), "cudnnSetStream"
        )
        self._stream_ptr = ptr

    # ------------------------------------------------------------------
    # 描述符与计划
    # ------------------------------------------------------------------

    def _plan(self, x_shape, w_shape, dtype, pad, stride, dilation):
        key = (tuple(x_shape), tuple(w_shape), dtype, tuple(pad), tuple(stride), tuple(dilation))
        hit = self._plans.get(key)
        if hit is not None:
            return hit

        data_type = _SUPPORTED[dtype]
        lib, vp = self.lib, ctypes.c_void_p
        xd, wd, cd, yd, bd = vp(), vp(), vp(), vp(), vp()
        self._check(lib.cudnnCreateTensorDescriptor(ctypes.byref(xd)), "create x desc")
        self._check(lib.cudnnCreateFilterDescriptor(ctypes.byref(wd)), "create w desc")
        self._check(lib.cudnnCreateConvolutionDescriptor(ctypes.byref(cd)), "create conv desc")
        self._check(lib.cudnnCreateTensorDescriptor(ctypes.byref(yd)), "create y desc")
        self._check(lib.cudnnCreateTensorDescriptor(ctypes.byref(bd)), "create b desc")

        n, c, h, w = x_shape
        k, _, r, s = w_shape
        ph, pw = pad
        sh, sw = stride
        dh, dw = dilation
        ho = (h + 2 * ph - dh * (r - 1) - 1) // sh + 1
        wo = (w + 2 * pw - dw * (s - 1) - 1) // sw + 1
        for desc, dims in ((xd, (n, c, h, w)), (yd, (n, k, ho, wo))):
            self._check(
                lib.cudnnSetTensor4dDescriptor(desc, _CUDNN_TENSOR_NCHW, data_type, *dims),
                "set tensor4d",
            )
        self._check(
            lib.cudnnSetFilter4dDescriptor(
                wd, data_type, _CUDNN_TENSOR_NCHW, k, c, r, s
            ),
            "set filter4d",
        )
        self._check(
            lib.cudnnSetConvolutionNdDescriptor(
                cd,
                2,
                (ctypes.c_int * 2)(ph, pw),
                (ctypes.c_int * 2)(sh, sw),
                (ctypes.c_int * 2)(dh, dw),
                _CUDNN_CROSS_CORRELATION,
                _CUDNN_DATA_FLOAT,
            ),
            "set conv nd",
        )
        # bias 用 (1, C, 1, 1) 的 NCHW 描述符，最后与输出做一次 add。
        self._check(
            lib.cudnnSetTensor4dDescriptor(bd, _CUDNN_TENSOR_NCHW, data_type, 1, k, 1, 1),
            "set bias desc",
        )

        # f32 必须锁死为 FMA 数学：cuDNN 在 Ampere 之后默认允许卷积走 TF32
        # （尾数 10 位），实测 c_in=64/128 时与参考语义差到 5e-3——这不是实现
        # 差异，是精度语义被悄悄降级了。f16 保持默认（它本来就是半精度）。
        if dtype == np.dtype("float32"):
            self._check(
                lib.cudnnSetConvolutionMathType(cd, _CUDNN_FMA_MATH),
                "set conv math type",
            )

        perf = _FwdAlgoPerf()
        count = ctypes.c_int(0)
        self._check(
            lib.cudnnGetConvolutionForwardAlgorithm_v7(
                self.handle, xd, wd, cd, yd, 1, ctypes.byref(count), ctypes.byref(perf)
            ),
            "get fwd algo",
        )
        if count.value < 1:
            raise RuntimeError("cuDNN 没有返回可用算法")
        algo = int(perf.algo)
        workspace = ctypes.c_size_t(0)
        self._check(
            lib.cudnnGetConvolutionForwardWorkspaceSize(
                self.handle, xd, wd, cd, yd, algo, ctypes.byref(workspace)
            ),
            "workspace size",
        )

        plan = ((xd, wd, cd, yd, bd), algo, int(workspace.value))
        self._plans[key] = plan
        return plan

    def _workspace(self, width: int):
        if width == 0:
            return None
        buf = self._workspaces.get(width)
        if buf is None:
            import cupy as cp

            buf = cp.empty(width, dtype=cp.uint8)
            self._workspaces[width] = buf
        return buf

    # ------------------------------------------------------------------

    def convolution(self, x, w, bias, out, *, pad, stride, dilation) -> None:
        """``out = conv2d(x, w) (+ bias)``，原地写入 ``out``。"""
        import cupy as cp

        dtype = np.dtype(x.dtype)
        if dtype not in _SUPPORTED:
            raise CudnnUnavailable(f"cuDNN 快路径不支持 {dtype}")

        descs, algo, ws_bytes = self._plan(
            x.shape, w.shape, dtype, pad, stride, dilation
        )
        self.set_stream(int(cp.cuda.get_current_stream().ptr))
        workspace = self._workspace(ws_bytes)
        wsp = (
            ctypes.c_void_p(0)
            if workspace is None
            else ctypes.c_void_p(int(workspace.data.ptr))
        )

        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        vp = ctypes.c_void_p
        # 描述符本来就是 c_void_p；再套一层 ctypes.c_void_p() 在 Python 3.9+
        # 会抛 "cannot be converted to pointer"。旧机一直找不到 libcudnn、整类
        # 测试被跳过，所以这条路径直到换到带 cuDNN 的机器才暴露。
        xd, wd, cd, yd, bd = descs
        self._check(
            self.lib.cudnnConvolutionForward(
                self.handle, ctypes.byref(alpha), xd, vp(int(x.data.ptr)),
                wd, vp(int(w.data.ptr)), cd, algo,
                wsp, ws_bytes, ctypes.byref(beta), yd, vp(int(out.data.ptr)),
            ),
            "convolution forward",
        )
        if bias is not None:
            one = ctypes.c_float(1.0)
            self._check(
                self.lib.cudnnAddTensor(
                    self.handle, ctypes.byref(alpha), bd, vp(int(bias.data.ptr)),
                    ctypes.byref(one), yd, vp(int(out.data.ptr)),
                ),
                "add tensor (bias)",
            )
