"""同一批旋钮在两家厂商上是否选出同一个值（跨后端调优对照）。

用法（两段，因为 CUDA 与 HIP 通常不在同一台机器上）：

```bash
# 采集：各自机器上跑一遍，写一份对照报告
python -m benchmarks.tune_compare --backend cuda --save cuda_tune.json
python -m benchmarks.tune_compare --backend hip  --save hip_tune.json

# 对照：把两份报告拿回同一处读
python -m benchmarks.tune_compare --compare cuda_tune.json hip_tune.json
```

**为什么不并进 ``tune.py``。** ``tune.py`` 的职责是"在某台机器上搜出答案并
写进配置库"，产物要进运行时；这里只回答一个实验问题，产物是一次性报告。
把两件事塞进一个入口，会让 ``--save`` 的含义随参数组合漂移。

**为什么只比配置与各自机内的提升，不比绝对耗时。** CUDA 与 HIP 的数字来自
不同机器、不同时钟、不同驱动栈，跨机比毫秒没有意义（``run.py`` 开头也是
这条规矩）。可比的是：

1. 同一个旋钮（``block`` / ``tile_w``）在相同形状下，两边选出的值是否相同；
2. 选出来的值相对**本机默认配置**快多少——默认是调优空间的第一个候选，
   也就是"不调优时后端真正在跑的东西"。

v3 起"提升"来自交替 A/B 的配对测量（``tuning/paired.py``）：``decided`` 为假
表示赢家没有超出噪声（或前后两半方向相反），此时的提升只是"这次采样里谁靠前"，
打印时加 ``*`` 标注，不能当成收益读（5.140）。

**为什么用同一批用例。** 用例由 ``tune.TUNE_CASES`` 定义，两段采集共用同一份
定义；形状、dtype、随机种子都一样，否则"配置不同"可能只是输入不同。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from aetherflow.tuning import ORDER_ALTERNATE, ORDER_SHUFFLE, VOTE_ALL, VOTE_MAJORITY

from .run import backend_names, open_backend
from .tune import TUNE_CASES, TuneCase, _device_name, search

REPORT_FORMAT = "aetherflow-tune-compare/3"


def _canonical(config: Mapping[str, Any]) -> str:
    """配置的可比较形式：键序与 ``None``/``16`` 的书写差异都消掉。"""
    return json.dumps(dict(config), sort_keys=True, separators=(",", ":"))


def _fmt_config(config: Mapping[str, Any]) -> str:
    """``{'block': 256}`` -> ``block=256``；``tile_w=None`` 显示成 ``auto``。"""
    return " ".join(
        f"{name}={'auto' if value is None else value}"
        for name, value in sorted(config.items())
    )


def collect(
    backend,
    cases: Sequence[TuneCase] = TUNE_CASES,
    *,
    rounds: Optional[int] = None,
    paired: bool = True,
    staged: bool = False,
    repeats: int = 1,
    noise_floor: float = 0.01,
    order: str = ORDER_ALTERNATE,
    warmup_rounds: int = 0,
    max_round_seconds: Optional[float] = None,
    max_round_factor: Optional[float] = None,
    vote: str = VOTE_ALL,
    progress=None,
) -> Dict[str, Any]:
    """对每个用例搜索一次，把赢家、默认与提升打包成一份可保存的报告。

    v3 起默认用交替 A/B（``paired=True``）：报告里的 ``decided`` 与
    ``diff_*`` 直接回答"这个赢家是不是超出了噪声"。``paired=False`` 只用于
    复现 v2 的串行口径，报告会标记 ``method="serial"``，不与 v3 的配对结果
    混读（两者对"提升"的定义不同）。

    ``repeats`` > 1 时每个用例独立采多次，``vote`` 决定怎么合并各次判定：
    ``all``（默认）要求每次相同，``majority`` 超过半数同向即采纳（需要
    ``repeats >= 3``）。报告同时落盘 ``vote`` 与逐候选票数，便于复查"这个
    结论是全票还是多数票给的"（5.149）。
    """
    entries: Dict[str, Any] = {}

    def record(case: TuneCase, result) -> None:
        default = result.candidates[0]
        entry: Dict[str, Any] = {
            "op": case.op,
            "signature": dict(case.signature),
            "best": dict(result.best.config),
            # median 是典型延迟（排名依据），best 是最佳可达，p95 与 spread 是离散度
            "median_ms": result.best.median_ms,
            "best_ms": result.best.best_ms,
            "p95_ms": result.best.p95_ms,
            "spread": result.best.spread,
            "default": dict(default.config),
            "default_median_ms": default.median_ms,
            "default_best_ms": default.best_ms,
            "speedup": result.speedup,
            "best_speedup": result.best_speedup,
            "tie_epsilon": result.tie_epsilon,
            "rejected": [dict(item.config) for item in result.rejected],
            "method": getattr(result, "method", "serial"),
        }
        if entry["method"] == "paired":
            entry["decided"] = result.decided
            entry["rounds"] = result.rounds
            entry["noise_floor"] = result.noise_floor
            # 轮内跨度决定"这一轮的样本是否还共享同一段整机状态"：超限时
            # verdict 会被降级为 long-round，报告必须留下跨度本身供复查（5.142）
            entry["round_seconds"] = list(getattr(result, "round_seconds", ()))
            entry["max_round_seconds"] = getattr(result, "max_round_seconds", None)
            entry["max_round_factor"] = getattr(result, "max_round_factor", None)
            # 相对门槛算出的实际秒数：换机器后它才是"这一轮允许多长"的可比读数
            entry["round_budget_seconds"] = getattr(result, "round_budget_seconds", None)
            entry["warmup_rounds"] = getattr(result, "warmup_rounds", 0)
            entry["warmup_seconds"] = getattr(result, "warmup_seconds", 0.0)
            # 两阶段搜索的分解：每段扫了哪些键、赢家、最长轮与降级数（5.144）
            entry["stages"] = [dict(stage) for stage in getattr(result, "stages", ())]
            entry["repeats"] = getattr(result, "repeats", 1)
            entry["replicates"] = [
                dict(item) for item in getattr(result, "replicates", ())
            ]
            # 跨采集复现的合并方式与逐候选票数：report 里要能回答"这个结论是
            # 全票还是多数票给的"（5.149）
            entry["vote"] = getattr(result, "vote", None)
            entry["diff_median_pct"] = result.best.diff_median_pct
            entry["ci_low_pct"] = result.best.ci_low_pct
            entry["ci_high_pct"] = result.best.ci_high_pct
            # 默认胜出时 best 的配对差恒为 0，看不出"为什么没换"。逐候选记录
            # 配对差与判定，报告才能回答"每个候选离默认多远、是否超出噪声"。
            entry["pairs"] = [
                {
                    "config": dict(item.config),
                    "verdict": item.verdict,
                    "votes": dict(item.votes),
                    # 逐次配对差：与 ``votes`` 的一一对应，用来区分"那张 noise
                    # 是幅度不足还是方向反转"（5.150）
                    "replicate_diffs": list(item.replicate_diffs),
                    "diff_median_pct": item.diff_median_pct,
                    "diff_mean_pct": item.diff_mean_pct,
                    "ci_low_pct": item.ci_low_pct,
                    "ci_high_pct": item.ci_high_pct,
                    "first_half_pct": item.first_half_pct,
                    "second_half_pct": item.second_half_pct,
                }
                for item in result.candidates
                if item.feasible
            ]
        entries[case.name] = entry
        if progress is not None:
            progress(case, result)

    search(
        backend, cases,
        rounds=rounds, paired=paired, staged=staged, repeats=repeats,
        noise_floor=noise_floor, order=order,
        warmup_rounds=warmup_rounds,
        max_round_seconds=max_round_seconds,
        max_round_factor=max_round_factor,
        vote=vote,
        progress=record,
    )
    return {
        "format": REPORT_FORMAT,
        "backend": backend.name,
        "device": _device_name(backend),
        "method": "paired" if paired else "serial",
        "staged": bool(paired and staged),
        "repeats": repeats if paired else None,
        "vote": vote if paired and repeats > 1 else None,
        "order_mode": order if paired else None,
        "warmup_rounds": warmup_rounds if paired else None,
        "max_round_seconds": max_round_seconds if paired else None,
        "max_round_factor": max_round_factor if paired else None,
        "cases": entries,
    }


@dataclass(frozen=True)
class CompareRow:
    """一个用例的对照结果；只有一边跑过时 ``same`` 为 ``None``。"""

    case: str
    left: Optional[Mapping[str, Any]]
    right: Optional[Mapping[str, Any]]

    @property
    def same(self) -> Optional[bool]:
        if self.left is None or self.right is None:
            return None
        return _canonical(self.left["best"]) == _canonical(self.right["best"])


def compare_reports(left: Mapping[str, Any], right: Mapping[str, Any]) -> List[CompareRow]:
    """按用例名对齐两份报告，顺序是"先左后右、各自首次出现"。"""
    left_cases = left.get("cases", {})
    right_cases = right.get("cases", {})
    names: List[str] = []
    for table in (left_cases, right_cases):
        for name in table:
            if name not in names:
                names.append(name)
    return [
        CompareRow(name, left_cases.get(name), right_cases.get(name))
        for name in names
    ]


def summarize(rows: Sequence[CompareRow]) -> Dict[str, Any]:
    """一致率与各自机内平均提升（只统计两边都跑过的用例）。"""
    both = [row for row in rows if row.same is not None]
    agreed = [row for row in both if row.same]

    def mean_speedup(side: str) -> Optional[float]:
        values = [
            float(getattr(row, side)["speedup"])
            for row in both
            if getattr(row, side).get("speedup") is not None
        ]
        return round(sum(values) / len(values), 4) if values else None

    return {
        "total": len(rows),
        "both": len(both),
        "agreed": len(agreed),
        "left_mean_speedup": mean_speedup("left"),
        "right_mean_speedup": mean_speedup("right"),
    }


def load_report(path: str) -> Dict[str, Any]:
    """读报告；坏 JSON、版本不认识、形状不对都显式失败。"""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} 不是合法 JSON：{exc}") from exc

    if not isinstance(data, Mapping):
        raise ValueError(f"{path} 的顶层必须是对象，得到 {type(data).__name__}")
    if data.get("format") != REPORT_FORMAT:
        raise ValueError(
            f"{path} 的报告格式 {data.get('format')!r} 不受支持（本实现只读 "
            f"{REPORT_FORMAT}）；v3 起默认用交替 A/B 并以配对差的置信区间判定"
            "'是否真的更快'，与 v1/v2 的串行口径不可混读，请重新采集"
        )
    cases = data.get("cases")
    if not isinstance(cases, Mapping):
        raise ValueError(f"{path} 的 cases 必须是对象")
    for name, entry in cases.items():
        if not isinstance(entry, Mapping) or not isinstance(entry.get("best"), Mapping):
            raise ValueError(f"{path} 的用例 {name!r} 缺少 best 配置")
    return dict(data)


def _print_comparison(left: Mapping[str, Any], right: Mapping[str, Any], rows: Sequence[CompareRow]) -> None:
    print(
        f"对照 {left.get('backend')}（{left.get('device')}）"
        f" vs {right.get('backend')}（{right.get('device')}）"
    )
    left_method = left.get("method", "serial")
    right_method = right.get("method", "serial")
    if left_method != right_method:
        print(
            f"  [注意] 两份报告的测量口径不同（{left_method} vs {right_method}）："
            "一致率与提升只作参考，不能当成同口径对照"
        )

    def vote_label(report: Mapping[str, Any]) -> str:
        repeats = report.get("repeats") or 1
        if repeats <= 1:
            return "单次"
        return "多数票" if report.get("vote") == VOTE_MAJORITY else "全票"

    left_vote, right_vote = vote_label(left), vote_label(right)
    if left_vote != right_vote:
        print(
            f"  [注意] 两份报告的复现合并方式不同（{left_vote} vs {right_vote}）："
            "'未复现'的含义不同，一致率只作参考"
        )
    print(
        f"  {'case':22s} {'left':18s} {'right':18s} 一致   "
        f"{'left 提升':>10s} {'right 提升':>10s}"
    )

    def cell(entry: Optional[Mapping[str, Any]]) -> str:
        return "—" if entry is None else _fmt_config(entry["best"])

    def speed(entry: Optional[Mapping[str, Any]]) -> str:
        if entry is None or entry.get("speedup") is None:
            return "—"
        text = f"{entry['speedup']:.2f}x"
        if entry.get("method") == "paired" and not entry.get("decided", True):
            text += "*"
        return text

    undecided = any(
        (entry is not None)
        and entry.get("method") == "paired"
        and not entry.get("decided", True)
        for row in rows
        for entry in (row.left, row.right)
    )

    for row in rows:
        flag = "—" if row.same is None else ("是" if row.same else "否")
        print(
            f"  {row.case:22s} {cell(row.left):18s} {cell(row.right):18s} {flag:4s} "
            f"{speed(row.left):>10s} {speed(row.right):>10s}"
        )
    if undecided:
        print("  * 该侧赢家未被判定（差异在噪声内、批次不稳、跨度超限或未复现），提升不能当收益读")

    stats = summarize(rows)
    if stats["both"]:
        print(f"配置一致 {stats['agreed']}/{stats['both']} 个用例（两边都跑过的）")
    for side, key in (("left", "left_mean_speedup"), ("right", "right_mean_speedup")):
        value = stats[key]
        label = left.get("backend") if side == "left" else right.get("backend")
        if value is None:
            print(f"{label} 平均提升：—（没有可用的默认配置测量）")
        else:
            print(f"{label} 机内平均提升 {value:.4f}x（相对各例默认配置）")
    missing = stats["total"] - stats["both"]
    if missing:
        print(f"有 {missing} 个用例只有一边有结果，不计入一致率")


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="跨后端调优对照")
    parser.add_argument(
        "--backend",
        choices=tuple(n for n in backend_names() if n != "ref"),
        help="采集模式：在哪个后端上搜索",
    )
    parser.add_argument("--save", help="采集模式：把对照报告写到该 JSON")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("LEFT", "RIGHT"),
        help="对照模式：读两份报告并打印对照表",
    )
    parser.add_argument(
        "--rounds", type=int, default=None,
        help="交替 A/B 的轮数（默认 8）；--serial 时默认 2，表示每个候选连续测几次",
    )
    parser.add_argument(
        "--serial", action="store_true",
        help="退回串行测量（v2 口径），只用于对照，不产生置信区间",
    )
    parser.add_argument(
        "--noise-floor", type=float, default=0.01,
        help="配对差中位数低于该比例时判为噪声、保留默认（默认 0.01，即 1%%）",
    )
    parser.add_argument(
        "--order", choices=(ORDER_ALTERNATE, ORDER_SHUFFLE), default=ORDER_ALTERNATE,
        help="轮内顺序：alternate 逐轮反转（默认），shuffle 用固定种子每轮随机排列",
    )
    parser.add_argument(
        "--warmup-rounds", type=int, default=0, metavar="N",
        help="正式轮之前的预热轮数：预热样本丢弃、不计跨度，用来跑掉首轮冷启动（默认 0）",
    )
    parser.add_argument(
        "--max-round-seconds", type=float, default=None, metavar="SEC",
        help="轮内跨度上限：一轮连测全部候选超过该秒数时判为跨度超限、保留默认（默认不检查）",
    )
    parser.add_argument(
        "--max-round-factor", type=float, default=None, metavar="F",
        help="相对跨度上限：预热轮平均耗时的 F 倍，超过时判跨度超限（需要 --warmup-rounds >= 1）",
    )
    parser.add_argument(
        "--staged", action="store_true",
        help="两阶段搜索：先扫一个键、再在赢家下扫剩下的键，最后与默认做两候选对照",
    )
    parser.add_argument(
        "--repeats", type=int, default=1, metavar="N",
        help="跨采集复现：同一空间独立采 N 次，只有每次都判更快的候选才替换默认（默认 1）",
    )
    parser.add_argument(
        "--vote", choices=(VOTE_ALL, VOTE_MAJORITY), default=VOTE_ALL,
        help="跨采集复现的合并方式：all 要求每次判定相同（默认），"
             "majority 超过半数同向即采纳（需要 --repeats >= 3）",
    )
    parser.add_argument(
        "--cases",
        default="",
        help="只跑名字里含这些子串的用例，逗号分隔（默认全部）",
    )
    args = parser.parse_args(argv)
    if args.warmup_rounds < 0:
        parser.error("--warmup-rounds 不能为负")
    if args.repeats < 1:
        parser.error("--repeats 至少为 1")
    if args.vote != VOTE_ALL:
        if args.serial:
            parser.error("--vote 建立在交替 A/B 上：--serial 没有跨采集判定可投票")
        if args.staged:
            parser.error("--vote 建立在跨采集复现上：--staged 与 --repeats 本就互斥")
        if args.repeats < 3:
            parser.error(
                f"--vote {args.vote} 需要 --repeats >= 3：两次采集下'多数'就是全票，"
                "与 --vote all 没有区别"
            )
    if args.serial and args.repeats > 1:
        parser.error("--repeats 建立在交替 A/B 上：不能与 --serial 同时使用")
    if args.staged and args.repeats > 1:
        parser.error(
            "--staged 与 --repeats 不能同时使用：两阶段搜索的候选集可能每次不同，"
            "无法逐候选判断是否复现"
        )
    if args.serial and (args.warmup_rounds or args.max_round_seconds is not None):
        parser.error(
            "--warmup-rounds 与 --max-round-seconds 只对交替 A/B 有意义："
            "--serial 没有轮的概念"
        )
    if args.max_round_factor is not None:
        if args.max_round_seconds is not None:
            parser.error(
                "--max-round-seconds 与 --max-round-factor 互斥：绝对秒数与预热轮"
                "倍数不能同时作为跨度门槛"
            )
        if args.max_round_factor <= 0:
            parser.error("--max-round-factor 必须为正")
        if args.warmup_rounds < 1:
            parser.error("--max-round-factor 需要 --warmup-rounds >= 1：基准是预热轮耗时")
    if args.serial and args.max_round_factor is not None:
        parser.error("--max-round-factor 只对交替 A/B 有意义：--serial 没有轮的概念")
    if args.serial and args.staged:
        parser.error("--staged 建立在交替 A/B 上：不能与 --serial 同时使用")

    if args.compare:
        if args.backend or args.save:
            parser.error("--compare 不与 --backend/--save 同用")
        try:
            left, right = (load_report(path) for path in args.compare)
        except ValueError as exc:
            print(f"读取报告失败：{exc}", file=sys.stderr)
            return 1
        _print_comparison(left, right, compare_reports(left, right))
        return 0

    if not args.backend:
        parser.error("采集模式需要 --backend，或用 --compare LEFT RIGHT 读报告")
    if not args.save:
        parser.error("采集模式需要 --save，否则报告只存在内存里")

    wanted = [s for s in args.cases.split(",") if s]
    cases = [c for c in TUNE_CASES if not wanted or any(s in c.name for s in wanted)]
    if not cases:
        print(f"--cases {args.cases} 没有匹配到任何用例", file=sys.stderr)
        return 1

    backend = open_backend(args.backend)
    print(f"对照采集 {len(cases)} 个单算子用例（{backend.name} / {_device_name(backend)}）……")

    def progress(case, result):
        default = result.candidates[0]
        tail = f"，相对默认 {result.speedup:.2f}x" if result.speedup else ""
        if getattr(result, "method", "serial") == "paired":
            best = result.best
            if best.diff_median_pct is not None:
                tail += (
                    f"，配对差 {best.diff_median_pct:+.2f}%"
                    f" [{best.ci_low_pct:+.2f}, {best.ci_high_pct:+.2f}]"
                )
            unstable = sum(
                1 for item in result.candidates if item.verdict == "unstable"
            )
            if unstable:
                tail += f"，{unstable} 个候选批次不稳"
            long_round = sum(
                1 for item in result.candidates if item.verdict == "long-round"
            )
            unreplicated = sum(
                1 for item in result.candidates if item.verdict == "unreplicated"
            )
            if long_round:
                seconds = getattr(result, "round_seconds", ())
                longest = max(seconds) if seconds else 0.0
                tail += f"，轮内最长 {longest:.2f}s 超限（{long_round} 个候选）"
            if getattr(result, "stages", ()):
                tail += f"，两阶段 {len(result.stages)} 段"
            if getattr(result, "repeats", 1) > 1:
                how = "多数票" if getattr(result, "vote", None) == VOTE_MAJORITY else "全票"
                tail += f"，重复 {result.repeats} 次（{how}）"
            if unreplicated:
                tail += f"，{unreplicated} 个候选未复现"
            if result.decided:
                tail += "，已判定"
            elif long_round:
                tail += "，跨度超限保留默认"
            elif unstable:
                tail += "，批次不稳保留默认"
            elif unreplicated:
                tail += "，未复现保留默认"
            else:
                tail += "，噪声内保留默认"
        print(
            f"  {case.name:22s} 默认 {_fmt_config(default.config)} "
            f"-> {_fmt_config(result.best.config)}{tail}",
            flush=True,
        )

    report = collect(
        backend, cases,
        rounds=args.rounds,
        paired=not args.serial,
        staged=args.staged,
        repeats=args.repeats,
        noise_floor=args.noise_floor,
        order=args.order,
        warmup_rounds=args.warmup_rounds,
        max_round_seconds=args.max_round_seconds,
        max_round_factor=args.max_round_factor,
        vote=args.vote,
        progress=progress,
    )
    with open(args.save, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"已写入 {args.save}（{len(report['cases'])} 个用例）")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
