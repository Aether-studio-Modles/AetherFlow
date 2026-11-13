"""conv2d 两种实现的对照测量：分块直算 vs im2col + GEMM。

用法（需要 GPU）：

```bash
python -m benchmarks.conv_compare
python -m benchmarks.conv_compare --rounds 5
```

为什么不放进 ``cases.py``：它的用途是回答"分块核在什么形状上划算"，
不是回归门禁。其中最大的形状用参考后端跑要几十秒，放进 CASES 会拖慢 CI
和本地无卡测试。

除了耗时，表里还给出一列 **im2col 的临时显存**——``(N, K, HO*WO)`` 的
float32 列矩阵。分块路径没有这块缓冲，这是"显存放大消失"的直接量化：
放大倍数约等于核面积，3x3 就是 9 倍输入大小。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from aetherflow import Graph, f32
from aetherflow.backends.cuda import CudaBackend

# (名字, 输入形状, 权重形状, padding)
SHAPES = (
    ("3x3_c3", (2, 3, 64, 64), (16, 3, 3, 3), (1, 1)),
    ("1x1_c32", (2, 32, 32, 32), (64, 32, 1, 1), (0, 0)),
    ("3x3_c8", (4, 8, 64, 64), (32, 8, 3, 3), (1, 1)),
    ("3x3_c16", (4, 16, 64, 64), (32, 16, 3, 3), (1, 1)),
    ("3x3_c24", (4, 24, 64, 64), (32, 24, 3, 3), (1, 1)),
    ("3x3_c32", (4, 32, 64, 64), (32, 32, 3, 3), (1, 1)),
    ("5x5_c16", (4, 16, 64, 64), (32, 16, 5, 5), (2, 2)),
    ("1x1_c256", (2, 256, 28, 28), (256, 256, 1, 1), (0, 0)),
    ("3x3_c64", (4, 64, 56, 56), (64, 64, 3, 3), (1, 1)),
    ("3x3_c128", (2, 128, 28, 28), (128, 128, 3, 3), (1, 1)),
)


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


def im2col_bytes(x_shape, w_shape, pad) -> int:
    """im2col 必须一次性分配的列矩阵字节数（理论上界，含 padding 展开）。"""
    n, c, h, w = x_shape
    _, _, kh, kw = w_shape
    ph, pw = pad
    ho = h + 2 * ph - kh + 1
    wo = w + 2 * pw - kw + 1
    return n * c * kh * kw * ho * wo * 4


def measure(backend, graph, feed, warmup: int, repeat: int) -> Dict[str, float]:
    for _ in range(warmup):
        backend.execute(graph, feed)
    backend.cp.cuda.get_current_stream().synchronize()

    samples: List[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        backend.execute(graph, feed)
        backend.cp.cuda.get_current_stream().synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return {
        "median_ms": round(statistics.median(samples), 4),
        "min_ms": round(samples[0], 4),
    }


def _run(impl: str, stages, rounds: int):
    """返回 (每个形状的最好成绩, 用到的 backend)——backend 留着重用，避免重复编译。"""
    backend = CudaBackend(conv_impl=impl)
    out: Dict[str, Dict[str, float]] = {}
    for name, x_shape, w_shape, pad in stages:
        g, feed = build(x_shape, w_shape, pad)
        best: Optional[Dict[str, float]] = None
        for _ in range(rounds):
            stats = measure(backend, g, feed, warmup=3, repeat=10)
            if best is None or stats["median_ms"] < best["median_ms"]:
                best = stats
        out[name] = best
    return out, backend


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="conv2d 实现对照")
    parser.add_argument("--rounds", type=int, default=3, help="每形状跑几轮取最好（默认 3）")
    args = parser.parse_args(argv)

    results = {}
    backends = {}
    for impl in ("tiled", "im2col"):
        results[impl], backends[impl] = _run(impl, SHAPES, args.rounds)

    # 两条路的数值必须一致；不一致时对照表没有意义
    for name, x_shape, w_shape, pad in SHAPES:
        g, feed = build(x_shape, w_shape, pad)
        a = backends["tiled"].execute(g, feed)["y"]
        b = backends["im2col"].execute(g, feed)["y"]
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)

    print(f"{'shape':10s} {'tiled':>9s} {'im2col':>9s} {'speedup':>8s} {'cols MB':>9s}")
    for name, x_shape, w_shape, pad in SHAPES:
        t = results["tiled"][name]["median_ms"]
        i = results["im2col"][name]["median_ms"]
        cols = im2col_bytes(x_shape, w_shape, pad) / (1024 * 1024)
        print(f"{name:10s} {t:9.4f} {i:9.4f} {i / t:7.2f}x {cols:9.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
