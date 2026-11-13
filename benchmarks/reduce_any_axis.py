"""非末维归约的两条铺法对照：一维（线程=输出）与两维（x 沿 inner、y 切归约）。

用法（需要 CUDA 或 ROCm 上的 cupy）：

```bash
python -m benchmarks.reduce_any_axis
python -m benchmarks.reduce_any_axis --backend hip --rounds 5
```

5.153 的非末维归约是"一个线程一个输出元素、沿归约轴串行读"：``inner``
窄的时候输出平面只有几十个线程，warp 也凑不满一段连续的 ``inner``。
5.174 加了第二条路——``blockDim.x`` 沿 ``inner``、``blockDim.y`` 切归约轴，
块内沿 y 做树归约。这里把两条路摆在一起量，回答"什么时候该换"。

两条路都在同一个 ``_reduce_any_axis`` 里，按 ``inner`` / ``outer*inner``
选；基准在实例上盖掉 ``_use_tiled_reduce``（分派判据）来强制走某一条，
量完复原。输入先上传到设备，只量执行本身。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from aetherflow import Graph, f32
from benchmarks.run import _sync, backend_names, open_backend

# (名字, 形状, 归约轴)。挑的是两条路各自的最差格子与分界附近：
#   inner=1     —— 一维铺法里归约轴跨整行，线程之间隔着一整行
#   inner=8/64  —— 窄到凑不满一个 warp / 刚好两个 warp
#   inner=1024  —— 宽 inner，但输出平面只有 1024 个线程
#   inner=3     —— 奇数窄 inner，两维铺法的 tx 会向上取整到 4
#   inner=256   —— 输出平面自己就能填满一屏 SM，分派应当留在一维铺法
SHAPES = (
    ("t1024_d4096", (1024, 4096), 0),
    ("b1024_t4096_d1", (1024, 4096, 1), 1),
    ("b64_t4096_d64", (64, 4096, 64), 1),
    ("b256_t333_d8", (256, 333, 8), 1),
    ("b32_t2048_d32", (32, 2048, 32), 1),
    ("b1_t4096_d1024", (1, 4096, 1024), 1),
    ("b128_t333_d3", (128, 333, 3), 1),
    ("b64_t4096_d256", (64, 4096, 256), 1),
)


def build(shape: tuple, axis: int):
    rng = np.random.default_rng(0)
    data = rng.standard_normal(shape).astype(np.float32)
    g = Graph("reduce_any_axis_compare")
    x = g.input("x", f32[shape])
    g.output(g.add_node("reduce_sum", [x], {"axis": axis}), "y")
    return g, {"x": data}, data.sum(axis=axis)


def measure(fn, sync, warmup: int = 3, repeat: int = 10) -> Dict[str, float]:
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
    return {"median_ms": round(statistics.median(samples), 4),
            "min_ms": round(samples[0], 4)}


def run_case(backend, cp, graph, feed, rounds: int) -> Dict[str, float]:
    """把输入先放到设备上，只量执行本身。"""
    device_feed = {name: cp.asarray(value) for name, value in feed.items()}
    best: Optional[Dict[str, float]] = None
    for _ in range(rounds):
        stats = measure(
            lambda: backend.run_device(graph, device_feed, release=True),
            lambda: _sync(backend),
        )
        if best is None or stats["median_ms"] < best["median_ms"]:
            best = stats
    return best


def forced(backend, *, tiled: bool):
    """临时让分派走指定的一条路，返回复原用的旧方法。

    分派条件在 ``DeviceBackend._use_tiled_reduce`` 里；在实例上盖一个常量
    返回值即可，类本身不受影响。返回值交还调用方，在 ``finally`` 里装回去。
    """
    old = backend._use_tiled_reduce
    backend._use_tiled_reduce = lambda inner, total: tiled
    return old


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="非末维归约：一维铺法 vs 两维铺法"
    )
    parser.add_argument(
        "--backend",
        choices=tuple(n for n in backend_names() if n != "ref"),
        default="cuda",
    )
    parser.add_argument("--rounds", type=int, default=3, help="每形状跑几轮取最好")
    args = parser.parse_args(argv)

    backend = open_backend(args.backend)
    cp = backend.cp

    print(
        f"{'case':20s} {'tiled':>9s} {'one-dim':>9s} {'auto':>9s} "
        f"{'tiled/1d':>9s} {'reduce':>8s} {'inner':>6s} {'out':>8s}"
    )
    for name, shape, axis in SHAPES:
        graph, feed, expected = build(shape, axis)
        reduce_len = shape[axis]
        inner = int(np.prod(shape[axis + 1:], dtype=np.int64)) if axis + 1 < len(shape) else 1
        total = int(np.prod(shape, dtype=np.int64)) // reduce_len

        old = forced(backend, tiled=True)
        try:
            tiled = run_case(backend, cp, graph, feed, args.rounds)
        finally:
            backend._use_tiled_reduce = old
        old = forced(backend, tiled=False)
        try:
            one_dim = run_case(backend, cp, graph, feed, args.rounds)
        finally:
            backend._use_tiled_reduce = old
        auto = run_case(backend, cp, graph, feed, args.rounds)

        got = backend.execute(graph, feed)["y"]
        # 只做"算对了"的抽查：归约长度到几千时，随机数的和会强烈相消，
        # 输出里出现 0.0x 这样的值很常见，逐元素相对误差失去意义。
        np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-3)

        ratio = f"{one_dim['median_ms'] / tiled['median_ms']:.2f}x"
        print(
            f"{name:20s} {tiled['median_ms']:9.4f} {one_dim['median_ms']:9.4f} "
            f"{auto['median_ms']:9.4f} {ratio:>9s} {reduce_len:8d} {inner:6d} {total:8d}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
