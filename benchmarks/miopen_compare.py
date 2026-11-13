
"""把自研卷积与 MIOpen 的同一形状前向放在一起计时。

用法（需要 ROCm 与 MIOpen）：

```bash
python -m benchmarks.miopen_compare
python -m benchmarks.miopen_compare --rounds 5
```

为什么不引 PyTorch：要的是"同一条卷积在 MIOpen 上的耗时"，不是框架层的
调度开销。这里用 ctypes 直调 ``libMIOpen.so`` 的 **solution** API，只依赖
系统里已有的库，与 ``benchmarks/cudnn_compare.py`` 同一口径。legacy 的
``miopenFindConvolutionForwardAlgorithm`` + ``miopenConvolutionForward``
组合在 ROCm 7.2 上已经跑不通（Find 能返回算法，但随后的 Forward 报
"No invoker was registered"），因此这里改走
``GetSolutionCount`` / ``GetSolution`` / ``Immediate`` 这条官方路径：先列出
该形状的全部解，取库给出的估计耗时最小的那个，``CompileSolution`` 把 JIT
编译挪出计时窗口，再计时 ``Immediate``。

三条口径

1. ``tiled``：本项目分块直算，无 im2col 缓冲；
2. ``im2col``：本项目 im2col + hipBLAS；
3. ``MIOpen``：同一形状、同一 dtype，NCHW / f32 / cross-correlation。
   bias 两边都算进计时窗口：本项目的卷积核把 bias 融在里面，MIOpen 侧在
   窗口内用容器自己的逐元素加法补一次。

数值必须三路一致（rtol/atol 1e-4），否则表里的耗时没有意义。这个入口只做
测量，不接生产路径——与 cuDNN 对照一样，它回答"差多少"，不回答"怎么换"。
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import statistics
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from aetherflow import Graph, f32
from aetherflow.backends.hip import HipBackend, runtime
from benchmarks.conv_compare import SHAPES, im2col_bytes

# ---- MIOpen 常量（miopen.h） ----------------------------------------------
MIOPEN_STATUS_SUCCESS = 0
MIOPEN_FLOAT = 1
MIOPEN_CONVOLUTION = 0  # MIOpen 的 miopenConvolution 就是 cross-correlation

# (x/w/conv/y 描述符地址, 输出形状)
_DescSet = Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]


class _Solution(ctypes.Structure):
    """``miopenConvSolution_t``。

    字段顺序来自 ``miopen.h``：``float time; size_t workspace_size;
    uint64_t solution_id; miopenConvAlgorithm_t algorithm;``。按 int64 猜
    solution_id 会读到时间与 workspace 的拼接值，随后 Immediate 报
    ``invalid solution id``——这一步不能凭直觉写。
    """

    _fields_ = [
        ("time", ctypes.c_float),
        ("workspace_size", ctypes.c_size_t),
        ("solution_id", ctypes.c_uint64),
        ("algorithm", ctypes.c_int),
    ]


def _load_miopen() -> Tuple[ctypes.CDLL, str]:
    candidates = ["libMIOpen.so.1", "libMIOpen.so"]
    found = ctypes.util.find_library("MIOpen")
    if found:
        candidates.insert(0, found)
    errors = []
    for name in candidates:
        try:
            return ctypes.CDLL(name), name
        except OSError as exc:
            errors.append(f"{name}: {exc}")
    raise OSError("；".join(errors))


class MIOpenConv:
    """只够做"一次 NCHW f32 前向卷积"的最小绑定。"""

    def __init__(self) -> None:
        lib, self.name = _load_miopen()
        self.lib = lib
        vp = ctypes.c_void_p
        sz = ctypes.c_size_t
        lib.miopenGetVersion.argtypes = [
            ctypes.POINTER(sz), ctypes.POINTER(sz), ctypes.POINTER(sz)
        ]
        lib.miopenCreate.argtypes = [ctypes.POINTER(vp)]
        lib.miopenSetStream.argtypes = [vp, vp]
        lib.miopenCreateTensorDescriptor.argtypes = [ctypes.POINTER(vp)]
        lib.miopenSet4dTensorDescriptor.argtypes = [
            vp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        lib.miopenCreateConvolutionDescriptor.argtypes = [ctypes.POINTER(vp)]
        lib.miopenInitConvolutionDescriptor.argtypes = [
            vp, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        lib.miopenGetConvolutionForwardOutputDim.argtypes = [
            vp, vp, vp, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        # solution API：ROCm 7.2 上唯一能真正执行前向的入口
        lib.miopenConvolutionForwardGetSolutionCount.argtypes = [
            vp, vp, vp, vp, vp, ctypes.POINTER(sz)
        ]
        lib.miopenConvolutionForwardGetSolution.argtypes = [
            vp, vp, vp, vp, vp, sz, ctypes.POINTER(sz),
            ctypes.POINTER(_Solution),
        ]
        lib.miopenConvolutionForwardGetSolutionWorkspaceSize.argtypes = [
            vp, vp, vp, vp, vp, ctypes.c_uint64, ctypes.POINTER(sz)
        ]
        lib.miopenConvolutionForwardCompileSolution.argtypes = [
            vp, vp, vp, vp, vp, ctypes.c_uint64
        ]
        lib.miopenConvolutionForwardImmediate.argtypes = [
            vp, vp, vp, vp, vp, vp, vp, vp, vp, sz, ctypes.c_uint64
        ]

        self.handle = vp()
        self._check(lib.miopenCreate(ctypes.byref(self.handle)), "miopenCreate")
        # 与容器当前流一致；benchmark 在默认流上跑，0 即默认流
        lib.miopenSetStream(self.handle, vp(int(runtime.current_stream())))
        self._cache: Dict[tuple, _DescSet] = {}

    def _check(self, status: int, what: str) -> None:
        if status != MIOPEN_STATUS_SUCCESS:
            raise RuntimeError(f"MIOpen {what} 失败（status={status}）")

    def version(self) -> str:
        major, minor, patch = (
            ctypes.c_size_t(), ctypes.c_size_t(), ctypes.c_size_t()
        )
        self.lib.miopenGetVersion(
            ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch)
        )
        return f"{major.value}.{minor.value}.{patch.value}"

    def descriptors(self, x_shape, w_shape, pad):
        key = (x_shape, w_shape, pad)
        if key in self._cache:
            return self._cache[key]
        lib, vp = self.lib, ctypes.c_void_p
        n, c, h, w = x_shape
        k, _, r, s = w_shape
        ph, pw = pad
        xd, wd, cd, yd = vp(), vp(), vp(), vp()
        for alloc, what in (
            (lambda: lib.miopenCreateTensorDescriptor(ctypes.byref(xd)), "x desc"),
            (lambda: lib.miopenCreateTensorDescriptor(ctypes.byref(wd)), "w desc"),
            (lambda: lib.miopenCreateConvolutionDescriptor(ctypes.byref(cd)), "conv desc"),
            (lambda: lib.miopenCreateTensorDescriptor(ctypes.byref(yd)), "y desc"),
        ):
            self._check(alloc(), what)
        self._check(lib.miopenSet4dTensorDescriptor(xd, MIOPEN_FLOAT, n, c, h, w), "x")
        self._check(lib.miopenSet4dTensorDescriptor(wd, MIOPEN_FLOAT, k, c, r, s), "w")
        self._check(
            lib.miopenInitConvolutionDescriptor(
                cd, MIOPEN_CONVOLUTION, ph, pw, 1, 1, 1, 1
            ),
            "init conv",
        )
        on, oc, oh, ow = (
            ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        )
        self._check(
            lib.miopenGetConvolutionForwardOutputDim(
                cd, xd, wd, ctypes.byref(on), ctypes.byref(oc),
                ctypes.byref(oh), ctypes.byref(ow),
            ),
            "output dim",
        )
        y_shape = (on.value, oc.value, oh.value, ow.value)
        self._check(
            lib.miopenSet4dTensorDescriptor(yd, MIOPEN_FLOAT, *y_shape), "y"
        )
        result = ((xd.value, wd.value, cd.value, yd.value), y_shape)
        self._cache[key] = result
        return result

    def prepare(self, x, w, b, pad):
        """枚举解、编译选中的那个，返回 (y, 一次前向的闭包)。

        ``GetSolution`` 会给出每个解的结构（含库侧的估计耗时）；取估计最小
        的解。``CompileSolution`` 把 JIT 编译放在计时窗口之外，否则第一次
        ``Immediate`` 会把编译时间算进去。返回 ``None`` 表示该形状没有解，
        调用方跳过它而不是伪造一个数字。
        """
        (xd, wd, cd, yd), y_shape = self.descriptors(
            tuple(x.shape), tuple(w.shape), pad
        )
        lib, vp = self.lib, ctypes.c_void_p

        count = ctypes.c_size_t(0)
        self._check(
            lib.miopenConvolutionForwardGetSolutionCount(
                self.handle, vp(wd), vp(xd), vp(cd), vp(yd),
                ctypes.byref(count),
            ),
            "solution count",
        )
        if count.value < 1:
            return None
        solutions = (_Solution * count.value)()
        found = ctypes.c_size_t(0)
        self._check(
            lib.miopenConvolutionForwardGetSolution(
                self.handle, vp(wd), vp(xd), vp(cd), vp(yd),
                count.value, ctypes.byref(found), solutions,
            ),
            "get solution",
        )
        chosen = min(list(solutions)[: found.value], key=lambda s: s.time)
        solution_id = int(chosen.solution_id)

        workspace_bytes = ctypes.c_size_t(0)
        self._check(
            lib.miopenConvolutionForwardGetSolutionWorkspaceSize(
                self.handle, vp(wd), vp(xd), vp(cd), vp(yd),
                solution_id, ctypes.byref(workspace_bytes),
            ),
            "solution workspace",
        )
        self._check(
            lib.miopenConvolutionForwardCompileSolution(
                self.handle, vp(wd), vp(xd), vp(cd), vp(yd), solution_id
            ),
            "compile solution",
        )

        y = runtime.empty(y_shape, np.float32)
        # workspace 只是驱动要的一块裸内存，容器没有 uint8；i8 同样是 1 字节
        workspace = runtime.empty((max(workspace_bytes.value, 1),), np.int8)
        workspace_nbytes = workspace.nbytes
        xp, wp = vp(x.ptr), vp(w.ptr)
        yp, wsp = vp(y.ptr), vp(workspace.ptr)
        xd_p, wd_p, cd_p, yd_p = vp(xd), vp(wd), vp(cd), vp(yd)
        bias = b.reshape(1, -1, 1, 1)

        def call() -> None:
            status = lib.miopenConvolutionForwardImmediate(
                self.handle, wd_p, wp, xd_p, xp, cd_p, yd_p, yp,
                wsp, workspace_nbytes, solution_id,
            )
            self._check(status, "convolution immediate")
            # 不用 ``y += bias``：增强赋值会把 y 变成闭包里的局部名
            y.__iadd__(bias)

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


def measure(fn: Callable[[], None], sync, warmup: int, repeat: int) -> Dict[str, float]:
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
    return {
        "median_ms": round(statistics.median(samples), 4),
        "min_ms": round(samples[0], 4),
    }


def run_aetherflow(impl: str, stages, rounds: int):
    backend = HipBackend(conv_impl=impl)
    out: Dict[str, Dict[str, float]] = {}
    for name, x_shape, w_shape, pad in stages:
        g, feed = build(x_shape, w_shape, pad)
        best: Optional[Dict[str, float]] = None
        for _ in range(rounds):
            stats = measure(
                lambda: backend.execute(g, feed), runtime.synchronize, 3, 10
            )
            if best is None or stats["median_ms"] < best["median_ms"]:
                best = stats
        out[name] = best
    return out, backend


def run_miopen(stages, rounds: int):
    conv = MIOpenConv()  # 先确认 MIOpen 在，再碰设备内存
    out: Dict[str, Dict[str, float]] = {}
    outputs: Dict[str, np.ndarray] = {}
    skipped: List[str] = []
    for name, x_shape, w_shape, pad in stages:
        _, feed = build(x_shape, w_shape, pad)  # 与 aetherflow 侧同一份数据
        x = runtime.asarray(feed["x"])
        w = runtime.asarray(feed["w"])
        b = runtime.asarray(feed["b"])
        prepared = conv.prepare(x, w, b, pad)
        if prepared is None:
            skipped.append(name)
            del x, w, b
            continue
        y, call = prepared
        best: Optional[Dict[str, float]] = None
        for _ in range(rounds):
            stats = measure(call, runtime.synchronize, 3, 10)
            if best is None or stats["median_ms"] < best["median_ms"]:
                best = stats
        out[name] = best
        runtime.synchronize()
        outputs[name] = runtime.asnumpy(y)
        del x, w, b, y
    return out, outputs, conv, skipped


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="conv2d 与 MIOpen 的对照")
    parser.add_argument("--rounds", type=int, default=3, help="每形状跑几轮取最好（默认 3）")
    args = parser.parse_args(argv)

    try:
        miopen_ms, miopen_out, conv, skipped = run_miopen(SHAPES, args.rounds)
    except OSError as exc:
        print(f"跳过：找不到 MIOpen（{exc}）")
        return 0

    tiled_ms, tiled_backend = run_aetherflow("tiled", SHAPES, args.rounds)
    im2col_ms, im2col_backend = run_aetherflow("im2col", SHAPES, args.rounds)

    # 三路数值必须一致，否则耗时对照没有意义
    for name, x_shape, w_shape, pad in SHAPES:
        if name in skipped:
            continue
        g, feed = build(x_shape, w_shape, pad)
        ref = miopen_out[name]
        for impl, backend in (("tiled", tiled_backend), ("im2col", im2col_backend)):
            got = backend.execute(g, feed)["y"]
            np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)

    if skipped:
        print(f"MIOpen 没有返回可用算法、已跳过：{', '.join(skipped)}")
    print(f"MIOpen {conv.version()}（{conv.name}）")
    print(
        f"{'shape':10s} {'tiled':>8s} {'im2col':>8s} {'MIOpen':>8s}"
        f" {'tiled/mio':>11s} {'im2col/mio':>12s} {'cols MB':>8s}"
    )
    for name, x_shape, w_shape, pad in SHAPES:
        t = tiled_ms[name]["median_ms"]
        i = im2col_ms[name]["median_ms"]
        cols = im2col_bytes(x_shape, w_shape, pad) / (1024 * 1024)
        if name in skipped:
            print(f"{name:10s} {t:8.4f} {i:8.4f} {'-':>8s} {'-':>11s} {'-':>12s} {cols:8.2f}")
            continue
        m = miopen_ms[name]["median_ms"]
        print(
            f"{name:10s} {t:8.4f} {i:8.4f} {m:8.4f}"
            f" {t / m:10.2f}x {i / m:11.2f}x {cols:8.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
