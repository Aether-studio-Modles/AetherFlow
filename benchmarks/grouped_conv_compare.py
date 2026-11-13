"""分组 / depthwise 卷积两条实现的对照：直算核 vs 逐组 im2col。

用法（需要 GPU）：

```bash
python -m benchmarks.grouped_conv_compare --backend hip
python -m benchmarks.grouped_conv_compare --backend hip --rounds 5
```

5.165 把 ``groups>1`` 的 auto 路径从"逐组 im2col + GEMM"换成了分组直算
核，当时只验证了数值，没有量性能。这个脚本回答两个问题：

1. 直算核在哪些形状上真的更快、快多少；
2. 哪些形状上 im2col 仍然值得保留（大 ``CG`` 时直算核没有 shared 复用，
   理论上可能被 GEMM 反超）。

表里另给一列 **im2col 的临时显存**：每组一份 ``(N, CG*KH*KW, HO*WO)`` 的
列矩阵，按组求和。直算路径不分配这块缓冲。

与 ``conv_compare.py`` 分开：那个只测 ``groups=1``（分块核 vs im2col），
而且写死 CUDA；这里的形状集是 depthwise / 小 ``CG``，后端从注册表挑，
HIP 上也能跑。
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

# (名字, 输入形状, 权重形状, attrs)。权重是 (O, C/groups, KH, KW)。
SHAPES = (
    ("dw3x3_32", (4, 32, 56, 56), (32, 1, 3, 3),
     {"groups": 32, "padding": (1, 1)}),
    ("dw3x3s2_64", (4, 64, 112, 112), (64, 1, 3, 3),
     {"groups": 64, "stride": (2, 2), "padding": (1, 1)}),
    ("dw5x5_64", (2, 64, 32, 32), (64, 1, 5, 5),
     {"groups": 64, "padding": (2, 2)}),
    ("dw3x3d2_32", (2, 32, 28, 28), (32, 1, 3, 3),
     {"groups": 32, "padding": (2, 2), "dilation": (2, 2)}),
    ("dw1x1_64", (4, 64, 56, 56), (64, 1, 1, 1), {"groups": 64}),
    ("g4_64", (4, 64, 56, 56), (64, 16, 3, 3),
     {"groups": 4, "padding": (1, 1)}),
    ("g8_32", (4, 32, 32, 32), (64, 4, 3, 3),
     {"groups": 8, "padding": (1, 1)}),
    ("g2_128", (2, 128, 28, 28), (128, 64, 3, 3),
     {"groups": 2, "padding": (1, 1)}),
    ("g16_256", (2, 256, 14, 14), (256, 16, 3, 3),
     {"groups": 16, "padding": (1, 1)}),
)


def build(x_shape, w_shape, attrs):
    g = Graph("grouped_conv2d")
    x = g.input("x", f32[x_shape])
    w = g.input("w", f32[w_shape])
    b = g.input("b", f32[w_shape[0]])
    g.output(g.add_node("conv2d", [x, w, b], dict(attrs)), "y")

    rng = np.random.default_rng(0)
    feed = {
        "x": rng.standard_normal(x_shape).astype(np.float32),
        "w": (rng.standard_normal(w_shape) * 0.1).astype(np.float32),
        "b": rng.standard_normal(w_shape[0]).astype(np.float32),
    }
    return g, feed


def im2col_bytes(x_shape, w_shape, attrs) -> int:
    """逐组 im2col 的列矩阵总字节数（每组一份，按组求和）。"""
    n, c, h, w = x_shape
    o, cg, kh, kw = w_shape
    groups = int(attrs.get("groups", 1))
    sh, sw = attrs.get("stride", (1, 1))
    ph, pw = attrs.get("padding", (0, 0))
    dh, dw = attrs.get("dilation", (1, 1))
    ho = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wo = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    return n * cg * kh * kw * ho * wo * 4 * groups


def measure(backend, graph, feed, warmup: int, repeat: int) -> Dict[str, float]:
    for _ in range(warmup):
        backend.execute(graph, feed)
    _sync(backend)

    samples: List[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        backend.execute(graph, feed)
        _sync(backend)
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return {
        "median_ms": round(statistics.median(samples), 4),
        "min_ms": round(samples[0], 4),
    }


def _run(name: str, impl: str, rounds: int):
    """返回 (每个形状的最好成绩, 用到的 backend)——backend 留着重用。"""
    backend = open_backend(name, conv_impl=impl)
    out: Dict[str, Dict[str, float]] = {}
    for case, x_shape, w_shape, attrs in SHAPES:
        g, feed = build(x_shape, w_shape, attrs)
        best: Optional[Dict[str, float]] = None
        for _ in range(rounds):
            stats = measure(backend, g, feed, warmup=3, repeat=10)
            if best is None or stats["median_ms"] < best["median_ms"]:
                best = stats
        out[case] = best
    return out, backend


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="分组 / depthwise 卷积实现对照")
    parser.add_argument(
        "--backend",
        choices=tuple(n for n in backend_names() if n != "ref"),
        default="cuda",
    )
    parser.add_argument("--rounds", type=int, default=3, help="每形状跑几轮取最好（默认 3）")
    args = parser.parse_args(argv)

    results = {}
    backends = {}
    # 用显式 grouped 而不是 auto：5.171 收窄之后 auto 只对 depthwise 走直算核，
    # 非 depthwise 会回落 im2col，这里要量的是"直算 vs 列矩阵"本身。
    for impl in ("grouped", "im2col"):
        results[impl], backends[impl] = _run(args.backend, impl, args.rounds)

    # 两条路的数值必须一致；不一致时对照表没有意义（累加顺序不同，不比逐位）
    for case, x_shape, w_shape, attrs in SHAPES:
        g, feed = build(x_shape, w_shape, attrs)
        a = backends["grouped"].execute(g, feed)["y"]
        b = backends["im2col"].execute(g, feed)["y"]
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)

    print(
        f"{'case':12s} {'grouped':>9s} {'im2col':>9s} "
        f"{'speedup':>8s} {'cols MB':>9s}"
    )
    for case, x_shape, w_shape, attrs in SHAPES:
        fast = results["grouped"][case]["median_ms"]
        slow = results["im2col"][case]["median_ms"]
        cols = im2col_bytes(x_shape, w_shape, attrs) / (1024 * 1024)
        print(f"{case:12s} {fast:9.4f} {slow:9.4f} {slow / fast:7.2f}x {cols:9.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
