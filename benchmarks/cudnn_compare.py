"""conv2d 与 cuDNN 的对照测量：自研分块核离工业实现有多远。

用法（需要 GPU 与 cuDNN 9）：

```bash
python -m benchmarks.cudnn_compare
python -m benchmarks.cudnn_compare --rounds 5
```

为什么不引 PyTorch：要的是「同一条卷积在 cuDNN 上的耗时」，而不是框架层的
调度开销。PyTorch 的 conv 除了 cuDNN 还套了 autograd、参数解析与 dispatcher，
把那些算进去会把差距夸大成框架差距；直调 cuDNN 才是硬件库本身的下限。这里用
ctypes 调 cuDNN 的 legacy API（v9 仍保留），只依赖系统里已有的 ``libcudnn.so.9``，
不新增 Python 依赖。

三条口径

1. ``tiled``：本项目分块直算，无 im2col 缓冲；
2. ``im2col``：本项目 im2col + cuBLAS；
3. ``cuDNN``：同一形状、同一 dtype，算法由 ``cudnnGetConvolutionForwardAlgorithm_v7``
   选第一个返回项，NCHW / f32 / cross-correlation。bias 的处理两边要一致：本项目的
   卷积核把 bias 融在里面，为对齐口径，cuDNN 侧也在计时窗口内用 cupy 做同一次加法。

数值必须三路一致（rtol/atol 1e-4），否则表里的耗时没有意义。

**只比设备侧**：本项目侧走 ``execute_device``（输出留在显存），cuDNN 侧也是
直调库、结果不出卡。若用 ``execute``，每次都会多付一次 4 MB 的 D2H（A800 上
约 1.4 ms），那是 PCIe 带宽而不是卷积速度（见 DESIGN 5.251）。

**输入也必须驻留**：``execute_device`` 对 numpy 输入仍会在每轮做 H2D，而 cuDNN
侧是一次 ``cp.asarray`` 后常驻。3x3_c64 的输入有 3.2 MB，A800 上 H2D 约 0.4 ms，
比卷积本身还大——这一条曾让 5.251 的表里混进 PCIe 带宽（5.253 修正）。所以
本项目侧先把 ``feed`` 转成设备数组，两边都只测计算。

**报稳态而不是单次**：单次 ``execute`` 的固定开销（host 提交 + sync 往返）在
A800 上是 0.076～0.086 ms，低 GFLOP 的形状里它比内核还长，单次口径测到的是
延迟下限而不是卷积速度。表里的数字是连续 launch 平摊后的稳态值，``--single``
可以切回单次口径做对照（5.253）。

**Ampere 及之后必须设 `NVIDIA_TF32_OVERRIDE=0`**：cuDNN 默认允许卷积走 TF32，
结果与 f32 差 1e-3 量级，会被下面的数值一致性检查拦下；本项目没有 TF32 路径。
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import statistics
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from aetherflow import Graph, f32
from aetherflow.backends.cuda import CudaBackend
from benchmarks.conv_compare import SHAPES, im2col_bytes

# ---- cuDNN 常量（cudnn.h） ------------------------------------------------
CUDNN_DATA_FLOAT = 0
CUDNN_TENSOR_NCHW = 0
CUDNN_CROSS_CORRELATION = 1
CUDNN_STATUS_SUCCESS = 0

# (x 描述符, w 描述符, 卷积描述符, y 描述符, 算法, workspace 字节)
_DescSet = Tuple[Tuple[int, int, int, int], int, int]


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


def _load_cudnn() -> Tuple[ctypes.CDLL, str]:
    """找 ``libcudnn.so.9``；找不到就由调用方决定跳过。

    先按动态链接器的名字找（系统装法），再找 ``pip install nvidia-cudnn-cu12``
    后的包内目录——那里放着的 so 不在 ``ldconfig`` 的搜索路径里，不主动加一条
    就要求用户自己设 ``LD_LIBRARY_PATH``。
    """
    candidates = ["libcudnn.so.9", "libcudnn.so"]
    found = ctypes.util.find_library("cudnn")
    if found:
        candidates.insert(0, found)
    try:
        import nvidia.cudnn as _pkg  # pip 装法：nvidia/cudnn/lib/libcudnn.so.9
    except ImportError:
        pass
    else:
        # 它是 namespace package，__file__ 为 None，只能从 __path__ 取目录。
        for root in list(getattr(_pkg, "__path__", []) or []):
            candidates.append(os.path.join(root, "lib", "libcudnn.so.9"))
    errors = []
    for name in candidates:
        try:
            return ctypes.CDLL(name), name
        except OSError as exc:  # 继续试下一个名字
            errors.append(f"{name}: {exc}")
    raise OSError("；".join(errors))


class CuDNNConv:
    """只够做「一次 NCHW f32 前向卷积」的最小绑定。"""

    def __init__(self) -> None:
        lib, self.name = _load_cudnn()
        self.lib = lib
        vp = ctypes.c_void_p
        lib.cudnnGetVersion.restype = ctypes.c_size_t
        lib.cudnnCreate.argtypes = [ctypes.POINTER(vp)]
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
        lib.cudnnSetConvolution2dDescriptor.argtypes = [
            vp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        lib.cudnnGetConvolutionForwardAlgorithm_v7.argtypes = [
            vp, vp, vp, vp, vp,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(_FwdAlgoPerf),
        ]
        lib.cudnnGetConvolutionForwardWorkspaceSize.argtypes = [
            vp, vp, vp, vp, vp, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.cudnnConvolutionForward.argtypes = [
            vp, ctypes.c_void_p, vp, ctypes.c_void_p, vp, ctypes.c_void_p, vp,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, vp,
            ctypes.c_void_p,
        ]
        self.version = int(lib.cudnnGetVersion())
        self.handle = vp()
        self._check(lib.cudnnCreate(ctypes.byref(self.handle)), "cudnnCreate")
        self._cache: Dict[tuple, _DescSet] = {}

    def _check(self, status: int, what: str) -> None:
        if status != CUDNN_STATUS_SUCCESS:
            raise RuntimeError(f"{what} 失败，cudnnStatus_t={status}")

    def descriptors(self, x_shape, w_shape, pad) -> _DescSet:
        key = (tuple(x_shape), tuple(w_shape), tuple(pad))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        lib, vp = self.lib, ctypes.c_void_p
        xd, wd, cd, yd = vp(), vp(), vp(), vp()
        self._check(lib.cudnnCreateTensorDescriptor(ctypes.byref(xd)), "x desc")
        self._check(lib.cudnnCreateFilterDescriptor(ctypes.byref(wd)), "w desc")
        self._check(lib.cudnnCreateConvolutionDescriptor(ctypes.byref(cd)), "conv desc")
        self._check(lib.cudnnCreateTensorDescriptor(ctypes.byref(yd)), "y desc")

        n, c, h, w = x_shape
        k, _, r, s = w_shape
        ph, pw = pad
        ho, wo = h + 2 * ph - r + 1, w + 2 * pw - s + 1
        for desc, dims in ((xd, (n, c, h, w)), (yd, (n, k, ho, wo))):
            self._check(
                lib.cudnnSetTensor4dDescriptor(
                    desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, *dims
                ),
                "set tensor4d",
            )
        self._check(
            lib.cudnnSetFilter4dDescriptor(
                wd, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, r, s
            ),
            "set filter4d",
        )
        self._check(
            lib.cudnnSetConvolution2dDescriptor(
                cd, ph, pw, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT
            ),
            "set conv2d",
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
        result: _DescSet = (
            (xd.value, wd.value, cd.value, yd.value), algo, int(workspace.value)
        )
        self._cache[key] = result
        return result

    def prepare(self, x, w, b, pad, workspace):
        """建输出张量并返回 (y, 一次前向的闭包)。"""
        import cupy as cp

        (xd, wd, cd, yd), algo, _ = self.descriptors(x.shape, w.shape, pad)
        k = w.shape[0]
        n, _, h, wd_ = x.shape
        r, s = w.shape[2], w.shape[3]
        ph, pw = pad
        y = cp.empty((n, k, h + 2 * ph - r + 1, wd_ + 2 * pw - s + 1), cp.float32)

        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        lib = self.lib
        xp, wp = ctypes.c_void_p(int(x.data.ptr)), ctypes.c_void_p(int(w.data.ptr))
        wsp = ctypes.c_void_p(int(workspace.data.ptr))
        yp = ctypes.c_void_p(int(y.data.ptr))
        xd_p, wd_p, cd_p, yd_p = (
            ctypes.c_void_p(xd), ctypes.c_void_p(wd),
            ctypes.c_void_p(cd), ctypes.c_void_p(yd),
        )

        def call() -> None:
            status = lib.cudnnConvolutionForward(
                self.handle, ctypes.byref(alpha), xd_p, xp, wd_p, wp, cd_p, algo,
                wsp, workspace.nbytes, ctypes.byref(beta), yd_p, yp,
            )
            self._check(status, "convolution forward")
            # 用 out= 而不是 y += ...：后者会让 y 变成闭包里的局部名。
            cp.add(y, b[None, :, None, None], out=y)

        return y, call


def build(x_shape, w_shape, pad):
    g = Graph("conv2d")
    x = g.input("x", f32[x_shape])
    w = g.input("w", f32[w_shape])
    b = g.input("b", f32[w_shape[0]])
    g.output(g.add_node("conv2d", [x, w, b], {"padding": pad}), "y")

    rng = np.random.default_rng(0)
    feed = {
        "x": rng.standard_normal(x_shape).astype(np.float32),
        "w": (rng.standard_normal(w_shape) * 0.1).astype(np.float32),
        "b": rng.standard_normal(w_shape[0]).astype(np.float32),
    }
    return g, feed


def measure(fn: Callable[[], None], sync, warmup: int, repeat: int,
            burst: int = 20) -> Dict[str, float]:
    """单次延迟（median）与稳态（连续 ``burst`` 次平摊）各报一份。

    稳态取 5 组里的最好值：同一实现固定顺序连测会互相污染，取最好比取中位数
    更接近"这台机器能做到的上限"（5.252 的测量方法）。
    """
    for _ in range(warmup):
        fn()
    sync()
    samples: List[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    steady: List[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(burst):
            fn()
        sync()
        t2 = time.perf_counter()
        steady.append((t2 - t0) * 1e3 / burst)
    return {
        "median_ms": round(statistics.median(samples), 4),
        "min_ms": round(samples[0], 4),
        "burst_ms": round(min(steady), 4),
    }


def run_aetherflow(impl: str, stages, rounds: int):
    backend = CudaBackend(conv_impl=impl)
    import cupy as cp

    sync = cp.cuda.get_current_stream().synchronize
    out: Dict[str, Dict[str, float]] = {}
    for name, x_shape, w_shape, pad in stages:
        g, feed = build(x_shape, w_shape, pad)
        # 驻留输入：与 cuDNN 侧一致，只测计算（见模块 docstring）
        feed_dev = {k: cp.asarray(v) for k, v in feed.items()}
        best: Optional[Dict[str, float]] = None
        for _ in range(rounds):
            stats = measure(lambda: backend.execute_device(g, feed_dev), sync, 3, 10)
            if best is None or stats["burst_ms"] < best["burst_ms"]:
                best = stats
        out[name] = best
    return out, backend


def run_cudnn(stages, rounds: int):
    conv = CuDNNConv()  # 先确认 cuDNN 在，再碰 cupy：缺库时错误更贴近原因
    import cupy as cp

    sync = cp.cuda.get_current_stream().synchronize
    out: Dict[str, Dict[str, float]] = {}
    outputs: Dict[str, np.ndarray] = {}
    for name, x_shape, w_shape, pad in stages:
        _, feed = build(x_shape, w_shape, pad)  # 与 aetherflow 侧同一份数据
        x = cp.asarray(feed["x"])
        w = cp.asarray(feed["w"])
        b = cp.asarray(feed["b"])
        _, _, workspace_bytes = conv.descriptors(x_shape, w_shape, pad)
        workspace = cp.empty(max(workspace_bytes, 1), cp.uint8)
        y, call = conv.prepare(x, w, b, pad, workspace)
        best: Optional[Dict[str, float]] = None
        for _ in range(rounds):
            stats = measure(call, sync, 3, 10)
            if best is None or stats["burst_ms"] < best["burst_ms"]:
                best = stats
        out[name] = best
        outputs[name] = cp.asnumpy(y)
        del x, w, b, y, workspace
        cp.get_default_memory_pool().free_all_blocks()
    return out, outputs, conv


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="conv2d 与 cuDNN 的对照")
    parser.add_argument("--rounds", type=int, default=3, help="每形状跑几轮取最好（默认 3）")
    parser.add_argument(
        "--single", action="store_true",
        help="报单次延迟而不是稳态（单次含 0.08 ms 级固定开销，见 5.253）",
    )
    args = parser.parse_args(argv)
    key = "median_ms" if args.single else "burst_ms"

    try:
        cudnn_ms, cudnn_out, conv = run_cudnn(SHAPES, args.rounds)
    except OSError as exc:
        print(f"跳过：找不到 cuDNN 9（{exc}）")
        return 0

    tiled_ms, tiled_backend = run_aetherflow("tiled", SHAPES, args.rounds)
    im2col_ms, im2col_backend = run_aetherflow("im2col", SHAPES, args.rounds)

    # 三路数值必须一致，否则耗时对照没有意义。跑之前必须 ``NVIDIA_TF32_OVERRIDE=0``：
    # cuDNN 在 Ampere 之后默认让卷积走 TF32，差值会到 1e-3，与实现正确性无关。
    for name, x_shape, w_shape, pad in SHAPES:
        g, feed = build(x_shape, w_shape, pad)
        ref = cudnn_out[name]
        for impl, backend in (("tiled", tiled_backend), ("im2col", im2col_backend)):
            got = backend.execute(g, feed)["y"]
            np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)

    print(f"cuDNN {conv.version}（{conv.name}），"
          f"{'单次延迟' if args.single else '稳态（连续 20 次平摊）'}，ms")
    print(
        f"{'shape':10s} {'tiled':>8s} {'im2col':>8s} {'cuDNN':>8s}"
        f" {'tiled/cudnn':>12s} {'im2col/cudnn':>13s} {'cols MB':>8s}"
    )
    for name, x_shape, w_shape, pad in SHAPES:
        t = tiled_ms[name][key]
        i = im2col_ms[name][key]
        c = cudnn_ms[name][key]
        cols = im2col_bytes(x_shape, w_shape, pad) / (1024 * 1024)
        print(
            f"{name:10s} {t:8.4f} {i:8.4f} {c:8.4f}"
            f" {t / c:11.2f}x {i / c:12.2f}x {cols:8.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
