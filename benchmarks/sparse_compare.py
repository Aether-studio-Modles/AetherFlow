"""``sparse_linear`` 与 cuSPARSE SpMM 的对照。

用法（需要 CUDA 或 ROCm 上的 cupy）：

```bash
python -m benchmarks.sparse_compare
python -m benchmarks.sparse_compare --backend hip --rounds 5
```

5.132 的 ``sparse_linear`` 是事件驱动的 ``out[s] = Σ W[indices[e]]``：
``indices`` / ``values`` / ``offsets`` 就是 CSR 的三件套，只是行内非零按列号
递增写（5.134）。因此它与"把脉冲矩阵当稀疏矩阵、乘一次权重"是同一件事，
可以直接和 cuSPARSE 的 SpMM 对比——这是本项目里唯一能用现成库对照的稀疏
算子，其它稀疏路径没有等价物。

两条路都只做一次乘法，不建稠密 ``S×K`` 脉冲矩阵。表里给出两者的列宽，
回答"自己写的散射是不是明显不如厂商库"。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from aetherflow import Graph, f32, i32
from benchmarks.run import _sync, backend_names, open_backend

# (名字, S, K, N, 每行非零比例)
SHAPES = (
    ("s512_k512_n512_2pct", 512, 512, 512, 0.02),
    ("s512_k1024_n1024_2pct", 512, 1024, 1024, 0.02),
    ("s256_k2048_n2048_05pct", 256, 2048, 2048, 0.005),
    ("s64_k4096_n1024_10pct", 64, 4096, 1024, 0.10),
)


def build_events(s: int, k: int, density: float, rng: np.random.Generator):
    """按"每行固定 nnz、列号递增"生成 CSR 三件套（与 spike_csr 的写出一致）。"""
    per_row = max(1, min(k, int(round(density * k))))
    indices = np.empty(s * per_row, dtype=np.int32)
    values = np.empty(s * per_row, dtype=np.float32)
    offsets = np.arange(0, (s + 1) * per_row, per_row, dtype=np.int32)
    for row in range(s):
        cols = np.sort(rng.choice(k, per_row, replace=False))
        lo = row * per_row
        indices[lo : lo + per_row] = cols
        values[lo : lo + per_row] = rng.standard_normal(per_row).astype(np.float32)
    return indices, values, offsets


def build(s: int, k: int, n: int, density: float):
    rng = np.random.default_rng(0)
    indices, values, offsets = build_events(s, k, density, rng)
    weights = (rng.standard_normal((k, n)) * 0.1).astype(np.float32)

    g = Graph("sparse_linear_compare")
    w = g.input("w", f32[k, n])
    idx = g.input("indices", i32[indices.size])
    val = g.input("values", f32[values.size])
    off = g.input("offsets", i32[s + 1])
    g.output(g.add_node("sparse_linear", [w, idx, val, off]), "y")

    feed = {"w": weights, "indices": indices, "values": values, "offsets": offsets}
    return g, feed, indices, values, offsets, weights


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
    """把输入先放到设备上，只量执行本身。

    ``execute`` 每次都会把 numpy 输入拷上设备；这里权重是 K×N（最大 16 MB），
    那样量到的主要是 PCIe 带宽，会把两条路的计算差异淹掉。
    """
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


def cusparse_case(cp, indices, values, offsets, weights, s, k, n, rounds):
    from cupyx.scipy.sparse import csr_matrix

    matrix = csr_matrix(
        (cp.asarray(values), cp.asarray(indices), cp.asarray(offsets)),
        shape=(s, k),
    )
    dense = cp.asarray(weights)
    sync = lambda: cp.cuda.Stream.null.synchronize()  # noqa: E731
    sparse_stats = measure(lambda: matrix.dot(dense), sync)
    expected = matrix.dot(dense).get()
    # 对照的第三条路：把脉冲矩阵真的建出来再乘一次。物化开销算在里面——
    # 这正是"稀疏路径存在的理由"，如果连它都不划算，稀疏就没有意义。
    dense_stats = measure(lambda: matrix.toarray().dot(dense), sync)
    return sparse_stats, dense_stats, expected


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="sparse_linear vs cuSPARSE SpMM")
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
        f"{'case':26s} {'ours':>9s} {'cusparse':>9s} {'dense mm':>9s} "
        f"{'vs cus':>7s} {'vs dense':>8s} {'nnz':>9s} {'sparse MB':>9s} "
        f"{'dense MB':>8s}"
    )
    for case, s, k, n, density in SHAPES:
        graph, feed, indices, values, offsets, weights = build(s, k, n, density)
        ours = run_case(backend, cp, graph, feed, args.rounds)
        theirs, dense_stats, expected = cusparse_case(
            cp, indices, values, offsets, weights, s, k, n, args.rounds
        )
        got = backend.execute(graph, feed)["y"]
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)

        nnz = indices.size
        sparse = (nnz * 8 + (s + 1) * 4) / (1024 * 1024)
        dense = s * k * 4 / (1024 * 1024)
        print(
            f"{case:26s} {ours['median_ms']:9.4f} {theirs['median_ms']:9.4f} "
            f"{dense_stats['median_ms']:9.4f} "
            f"{theirs['median_ms'] / ours['median_ms']:6.2f}x "
            f"{dense_stats['median_ms'] / ours['median_ms']:7.2f}x {nnz:9d} "
            f"{sparse:9.2f} {dense:8.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
