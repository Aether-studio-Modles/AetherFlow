"""跨运行的吞吐 / 显存趋势。

`benchmarks.run` 回答"这一次比基线慢了吗"；这里回答的是**"连续多次运行里，
它是不是在慢慢变慢"**。两者不能互相替代：回归判定需要一个固定阈值，
而趋势看的是斜率与形状——每次 1% 的漂移永远过不了 25% 的门槛，但十次之后
就是 10%。

数据来源是 `TelemetryRecorder` 的 `RunRecord`：整图 wall、显存峰值、吞吐。
``--compare`` 可以接着上次落盘的记录往后排，这样 CI 每次只跑一小段，
汇总时仍能看到完整曲线。

用法::

    # 每个用例跑 5 次，打印趋势表
    py -3.10 -m benchmarks.trend --backend cuda --rounds 5

    # 接上历史记录，并保存合并后的结果
    py -3.10 -m benchmarks.trend --backend cuda --rounds 5 \
        --compare benchmarks/telemetry_cuda.json --save benchmarks/telemetry_cuda.json

**只做聚合，不做看板。** 输出是可以直接喂给任何绘图工具的 JSON；画图与告警
是展示层的事，放在这里只会长出一个没人维护的小工具。
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from aetherflow import ReferenceExecutor, TelemetryRecorder, load_records

from .cases import CASES, Case, build_case
from .run import backend_names, open_backend


def _select(cases: Sequence[Case], names: Sequence[str]) -> Sequence[Case]:
    if not names:
        return cases
    wanted = set(names)
    selected = [case for case in cases if case.name in wanted]
    missing = wanted - {case.name for case in selected}
    if missing:
        raise SystemExit(f"未知用例: {', '.join(sorted(missing))}")
    return selected


def _units(graph) -> Optional[dict]:
    """输出元素总数；符号维算不出来时返回 None（宁可不报，也不报错数）。"""
    total = 0
    for value in graph.outputs:
        if not value.type.is_static:
            return None
        total += value.type.numel()
    return {"elements": total} if total else None


def collect(backend, cases: Sequence[Case], rounds: int, seed: int = 0) -> TelemetryRecorder:
    """对每个用例预热一次，再连跑 ``rounds`` 次，记录全部运行。"""
    recorder = TelemetryRecorder(capacity=max(64, rounds * len(cases) * 4))
    wrapped = recorder.wrap(backend)
    preheat = getattr(backend, "execute", None) or backend.run
    for case in cases:
        graph, feed = build_case(case, seed)
        # 预热不算进记录：第一次执行包含捕获/编译/首次分配，会把趋势的第一点抬高
        preheat(graph, feed)
        units = _units(graph)
        for _ in range(rounds):
            wrapped.execute(graph, feed, units=units, label=case.name)
    return recorder


def prune(recorder: TelemetryRecorder, keep: int) -> TelemetryRecorder:
    """每张图只保留最近 ``keep`` 条记录，其余原样丢弃。

    趋势文件会在每次调度（CI / 本地定时）后追加一轮，不裁剪就是无限增长：
    一张图一年能攒出几千条，最后连 JSON 都要读半天。裁剪按**图名**分别计数，
    不是取全局最近 N 条——后者会让冷门用例被热门用例挤掉，曲线直接断掉。
    """
    if not isinstance(keep, int) or isinstance(keep, bool) or keep < 1:
        raise ValueError(f"keep 必须是 >= 1 的整数，得到 {keep!r}")

    counters: dict = {}
    flags = []
    for record in reversed(recorder.records):
        count = counters.get(record.graph, 0)
        kept = count < keep
        flags.append(kept)
        if kept:
            counters[record.graph] = count + 1
    flags.reverse()

    trimmed = TelemetryRecorder(capacity=max(64, sum(flags)))
    trimmed.extend(
        [record for record, kept in zip(recorder.records, flags) if kept]
    )
    return trimmed


def _print_table(payload: dict, window: Optional[int]) -> None:
    graphs = payload["graphs"]
    suffix = f"，每组取最近 {window} 次" if window else ""
    print(f"共 {payload['runs']} 次运行 / {len(graphs)} 张图{suffix}")
    print(f"{'graph':24s} {'n':>3s} {'median':>10s} {'first':>10s} {'last':>10s} "
          f"{'change':>9s} {'slope':>10s}")
    for name, entry in graphs.items():
        stats = entry["wall_ms"]
        change = "n/a" if stats["change_pct"] is None else f"{stats['change_pct']:+.2f}%"
        print(
            f"{name:24s} {entry['count']:3d} {stats['median']:10.4f} "
            f"{stats['first']:10.4f} {stats['last']:10.4f} {change:>9s} "
            f"{stats['slope_per_run']:+10.5f}"
        )

    peaks = [(name, entry["peak_bytes"]) for name, entry in graphs.items() if "peak_bytes" in entry]
    if peaks:
        print("\n显存峰值（字节）")
        for name, stats in peaks:
            change = "n/a" if stats["change_pct"] is None else f"{stats['change_pct']:+.2f}%"
            print(
                f"  {name:22s} median={stats['median']:>12.0f} "
                f"max={stats['max']:>12.0f} change={change}"
            )

    units = [(name, entry["throughput"]) for name, entry in graphs.items() if "throughput" in entry]
    if units:
        print("\n吞吐（单位/秒）")
        for name, table in units:
            for unit, stats in table.items():
                change = "n/a" if stats["change_pct"] is None else f"{stats['change_pct']:+.2f}%"
                print(
                    f"  {name:22s} {unit:>10s} median={stats['median']:>14.1f} change={change}"
                )


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AetherFlow 吞吐 / 显存趋势")
    parser.add_argument("--backend", choices=backend_names(), default="ref")
    parser.add_argument("--case", action="append", default=[], help="只跑指定用例（可重复）")
    parser.add_argument("--rounds", type=int, default=5, help="每个用例跑几次（默认 5）")
    parser.add_argument("--compare", action="append", default=[], help="接着这些历史记录往后排（可重复）")
    parser.add_argument("--save", help="把（含历史的）全部记录写到该 JSON")
    parser.add_argument("--window", type=int, help="每组只看最近 N 次")
    parser.add_argument(
        "--prune",
        type=int,
        metavar="N",
        help="保存时每张图只留最近 N 条（默认全留；只对 --save 生效）",
    )
    parser.add_argument("--json", action="store_true", help="打印 JSON 而不是表格")
    args = parser.parse_args(argv)

    if args.rounds < 1:
        raise SystemExit("--rounds 至少为 1")
    if args.prune is not None and not args.save:
        # 参数错误在跑基准之前就拒绝：花几分钟采完再报错没有意义。
        raise SystemExit("--prune 只对 --save 生效；请同时指定 --save")

    cases = _select(CASES, args.case)
    backend = (
        ReferenceExecutor()
        if args.backend == "ref"
        else open_backend(args.backend)
    )
    recorder = collect(backend, cases, args.rounds)

    history = TelemetryRecorder()
    for path in args.compare:
        history.extend(load_records(path))
    if history.records:
        history.extend(recorder.records)
        payload = history.trend(window=args.window)
        merged = history
    else:
        payload = recorder.trend(window=args.window)
        merged = recorder

    if args.json:
        import json

        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_table(payload, args.window)

    if args.prune is not None:
        merged = prune(merged, args.prune)

    if args.save:
        merged.save(args.save)
        print(f"\n已写入 {args.save}（共 {len(merged.records)} 条记录）")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
