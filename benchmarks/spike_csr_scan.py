"""``spike_csr`` 两种铺法的对照：行内串行 vs 块内分段扫描。

用法（需要 CUDA 或 ROCm 上的 cupy）：

```bash
python -m benchmarks.spike_csr_scan
python -m benchmarks.spike_csr_scan --backend hip --rounds 5
```

5.35 起第二趟压缩是"一个线程负责一行、沿列号串行"，理由是行内必须按列号递增
写，才能与稠密矩阵乘的累加顺序一致（``sparse_linear`` 逐事件累加）。5.176 把
它改成"一个 block 负责一行、段内做块内前缀和"：段间串行、段内并行，递增顺序
不变，但每行的并行度从 1 个线程变成 ``blockDim.x`` 个。

这个脚本在同一批形状上强制走两条路，先逐位对拍再计时。计时口径与
``reduce_any_axis.py`` 相同：输入先上传设备，只量执行；``cap`` 取实际事件数加
每行一格余量，两条路一致。
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from aetherflow import Graph, ReferenceBackend, f32
from benchmarks.run import _sync, backend_names, open_backend

# (名字, 步数, 列数, 发放率)
SHAPES: Sequence[Tuple[str, int, int, float]] = (
    ("s8192_k512_2pct", 8192, 512, 0.02),
    ("s4096_k1024_2pct", 4096, 1024, 0.02),
    ("s2048_k4096_1pct", 2048, 4096, 0.01),
    ("s256_k4096_10pct", 256, 4096, 0.10),
    ("s64_k8192_5pct", 64, 8192, 0.05),
    ("s1_k65536_2pct", 1, 65536, 0.02),
)


def build(steps: int, width: int, density: float):
    """一张固定的脉冲图：``spike_offsets`` 起点表 + ``spike_csr`` 列号。"""
    rng = np.random.default_rng(0)
    mask = rng.random((steps, width)) < density
    spikes = np.zeros((steps, width), dtype=np.float32)
    spikes[mask] = rng.standard_normal(int(mask.sum())).astype(np.float32)
    capacity = int(mask.sum()) + steps

    g = Graph("spike_csr_compare")
    x = g.input("x", f32[steps, width])
    offsets = g.add_node("spike_offsets", [x])
    indices = g.add_node("spike_csr", [x, offsets], {"capacity": capacity})
    g.output(offsets, "offsets")
    g.output(indices, "indices")
    return g, {"x": spikes}


def forced(backend, *, scan: bool):
    """在实例上盖掉 ``_use_scan_csr``，返回旧方法供 ``finally`` 装回。"""
    old = backend._use_scan_csr
    backend._use_scan_csr = lambda steps: scan
    return old


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
    return {
        "median_ms": round(statistics.median(samples), 4),
        "min_ms": round(samples[0], 4),
    }


def run_case(backend, cp, graph, feed, rounds: int) -> Dict[str, float]:
    """只量执行本身：输入先放到设备上，避免把 H2D 算进来。"""
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


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="spike_csr: serial vs block scan")
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
        f"{'case':20s} {'steps':>7s} {'width':>7s} {'nnz':>8s} "
        f"{'scan':>9s} {'serial':>9s} {'auto':>9s} {'serial/scan':>12s}"
    )
    for case, steps, width, density in SHAPES:
        graph, feed = build(steps, width, density)
        expected = ReferenceBackend().execute(graph, feed)

        scan_stats = run_case(backend, cp, graph, feed, args.rounds)
        scan_out = backend.execute(graph, feed)      # 校验用，含 H2D / D2H
        old = forced(backend, scan=False)
        try:
            serial_stats = run_case(backend, cp, graph, feed, args.rounds)
            serial_out = backend.execute(graph, feed)
        finally:
            backend._use_scan_csr = old
        auto_stats = run_case(backend, cp, graph, feed, args.rounds)

        # 先对拍：两条路的 offsets 与 indices 必须逐位相同（写的是同一批
        # 列号、同一个递增顺序，所以不是"近似相等"）
        for label, out in (("scan", scan_out), ("serial", serial_out)):
            for name in ("offsets", "indices"):
                np.testing.assert_array_equal(
                    out[name], expected[name], err_msg=f"{label}.{name}"
                )

        nnz = int((feed["x"] != 0).sum())
        ratio = serial_stats["median_ms"] / scan_stats["median_ms"]
        print(
            f"{case:20s} {steps:7d} {width:7d} {nnz:8d} "
            f"{scan_stats['median_ms']:9.4f} {serial_stats['median_ms']:9.4f} "
            f"{auto_stats['median_ms']:9.4f} {ratio:11.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
