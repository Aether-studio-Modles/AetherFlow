"""捕获路径与逐节点执行的对照测量（CUDA Graph / hipGraph）。

用法（需要 GPU）：

```bash
python -m benchmarks.capture_compare --backend cuda
python -m benchmarks.capture_compare --backend hip --rounds 5
python -m benchmarks.capture_compare --backend hip --cases matmul,conv2d
```

为什么不并进 ``run.py``：``run.py`` 是回归门禁，一张基线 JSON 对应一条执行
路径；把捕获塞进去会让"同一个用例"有两种耗时口径，基线就失去含义。这里是
一次性回答"整图 replay 比逐节点下发快多少"，属于对照实验。

口径说明：两条路径都把主机输入拷进设备、都把输出拷回主机，差值只来自
CPU 侧的派发——``execute`` 每个节点一次 Python 调用，捕获路径只有一次
``graphLaunch``。因此这个倍数在小图上天然接近 1，在节点多的整图上才拉开，
这正是它要展示的东西。

捕获会按固定形状整图 replay，所以每个用例先做一次数值对拍：捕获路径与
逐节点执行必须一致，否则耗时对照没有意义。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .cases import CASES, Case, build_case
from .run import _sync, backend_names, open_backend

MAX_DIFF = 1e-4


def measure(call, sync, warmup: int, repeat: int) -> Dict[str, float]:
    """同步放在计时区间内：测的是"算完"的时间，不是"下发完"的时间。"""
    for _ in range(warmup):
        call()
    sync()
    samples: List[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        call()
        sync()
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return {
        "median_ms": round(statistics.median(samples), 4),
        "min_ms": round(samples[0], 4),
    }


def compare_case(backend, case: Case, rounds: int) -> Dict[str, object]:
    """返回一条对照记录：逐节点、捕获、倍数、显存。"""
    graph, feed = build_case(case, seed=0)

    def per_node():
        return backend.execute(graph, feed)

    plan = backend.capture(graph, feed)

    def captured():
        return plan.run(feed)

    # 对拍：捕获路径必须与逐节点执行给出同一份结果
    expected = per_node()
    actual = captured()
    for name, value in expected.items():
        np.testing.assert_allclose(
            actual[name], value, rtol=MAX_DIFF, atol=MAX_DIFF,
            err_msg=f"{case.name}: 捕获路径的输出 {name} 与逐节点执行不一致",
        )

    best_node: Optional[Dict[str, float]] = None
    best_cap: Optional[Dict[str, float]] = None

    def sync() -> None:
        _sync(backend)

    for _ in range(rounds):
        node = measure(per_node, sync, case.warmup, case.repeat)
        cap = measure(captured, sync, case.warmup, case.repeat)
        if best_node is None or node["median_ms"] < best_node["median_ms"]:
            best_node = node
        if best_cap is None or cap["median_ms"] < best_cap["median_ms"]:
            best_cap = cap

    return {
        "case": case.name,
        "node_ms": best_node["median_ms"],
        "capture_ms": best_cap["median_ms"],
        "speedup": round(best_node["median_ms"] / best_cap["median_ms"], 2),
        "peak_bytes": plan.peak_bytes,
        "total_bytes": plan.memory_bytes,
    }


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="捕获路径与逐节点执行的对照测量")
    parser.add_argument(
        "--backend",
        choices=[n for n in backend_names() if n != "ref"],
        default="cuda",
    )
    parser.add_argument("--rounds", type=int, default=3, help="每个用例跑几轮取最好（默认 3）")
    parser.add_argument(
        "--cases",
        default="",
        help="只跑名字里含这些子串的用例，逗号分隔（默认全部）",
    )
    args = parser.parse_args(argv)

    backend = open_backend(args.backend)
    wanted = [s for s in args.cases.split(",") if s]
    cases = [c for c in CASES if not wanted or any(s in c.name for s in wanted)]
    if not cases:
        print(f"--cases {args.cases} 没有匹配到任何用例", file=sys.stderr)
        return 1

    print(f"对照 {len(cases)} 个用例（{backend.name}，每例 {args.rounds} 轮取最好）……")
    rows: List[Dict[str, object]] = []
    skipped: List[str] = []
    for case in cases:
        try:
            row = compare_case(backend, case, args.rounds)
        except Exception as exc:  # 捕获有明确的形状/能力边界，报告而不是中断
            skipped.append(f"{case.name}: {type(exc).__name__}: {exc}")
            continue
        rows.append(row)
        print(
            f"  {row['case']:22s} {row['node_ms']:9.4f} -> {row['capture_ms']:9.4f} ms"
            f"  {row['speedup']:5.2f}x",
            flush=True,
        )

    if rows:
        print(
            f"\n{'case':22s} {'per-node':>10s} {'capture':>10s} {'speedup':>8s}"
            f" {'peak MB':>9s} {'total MB':>9s}"
        )
        for r in rows:
            print(
                f"{r['case']:22s} {r['node_ms']:10.4f} {r['capture_ms']:10.4f}"
                f" {r['speedup']:7.2f}x {r['peak_bytes'] / 2**20:9.2f}"
                f" {r['total_bytes'] / 2**20:9.2f}"
            )
    for line in skipped:
        print(f"  [skip] {line}")

    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(_main())
