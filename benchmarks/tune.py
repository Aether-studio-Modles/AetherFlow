"""按算子搜索 block，写进配置库；也可以用库里的配置做对拍。

用法：

```bash
# 搜索：每个可调算子一个小图，候选配置逐个测
py -3.10 -m benchmarks.tune --backend cuda --save benchmarks/tuning_cuda.json

# 对拍：应用库里命中的配置，验证数值与默认完全一致
py -3.10 -m benchmarks.tune --backend cuda --apply benchmarks/tuning_cuda.json
```

**为什么是单算子小图，而不是整图。** GPU 是异步执行的，逐节点计时得到的是
派发耗时（见 ``runtime/telemetry``）；把整图变快归因到某一个算子，在多个
kernel 混合的图里基本是噪声。单算子小图让"这个配置让这个 kernel 变快/变慢"
成为唯一变量。

**为什么配置按形状记。** block 的收益取决于元素数：小张量受 launch 开销支配，
大张量受带宽与占用率支配。同一个算子在 (1024,1024) 上的最优值，未必等于
在 (4,64) 上的最优值。

**为什么对拍用严格相等。** 这些内核的每个线程只写自己的元素，block 只决定
线程怎么分组；dropout 的掩码又按全局下标生成，与 block 无关。因此换 block
必须逐位相同。若出现差异，说明某个内核偷偷依赖了 blockDim，那是缺陷，
不能被"差得不多"掩盖。

**为什么默认用交替 A/B。** 串行测量把"先测的候选"与"后测的候选"放在不同的
整机状态里：5.63 在 HIP 上量到同一轮内整机抬升 30%，而 1M 逐元素算子的四档
``block`` 真实差异 ≤3%，于是"最优 block"只在 2/9 的用例上可复现。这里的
默认测量改成每轮连测全部候选、相邻轮反转顺序，并用配对差的 bootstrap 区间
判定差异是否超出噪声；``--serial`` 保留旧测法，只用于对照。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from aetherflow import Graph, f32
from aetherflow.tuning import (
    ORDER_ALTERNATE,
    ORDER_SHUFFLE,
    VOTE_ALL,
    VOTE_MAJORITY,
    AllRejectedError,
    ConfigStore,
    TuningRecord,
    tune,
    tune_paired,
    tune_replicated,
    tune_staged,
)

from .run import backend_names, measure, open_backend


@dataclass(frozen=True)
class TuneCase:
    """一个单算子小图：``op`` 必须与后端 ``_launch_block`` 的键一致。"""

    name: str
    op: str
    build: Callable
    signature: Mapping[str, object]
    warmup: int = 3
    repeat: int = 10


def _f32(rng, shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _binary(name: str, shape, rhs_shape=None):
    rhs_shape = shape if rhs_shape is None else rhs_shape

    def build(rng):
        g = Graph(name)
        a = g.input("a", f32[shape])
        b = g.input("b", f32[rhs_shape])
        g.output(g.add_node("add", [a, b]), "y")
        return g, {"a": _f32(rng, shape), "b": _f32(rng, rhs_shape)}

    return build


def _unary(name: str, shape):
    def build(rng):
        g = Graph(name)
        x = g.input("x", f32[shape])
        g.output(g.add_node(name, [x]), "y")
        return g, {"x": _f32(rng, shape)}

    return build


def _silu_gate(shape):
    def build(rng):
        g = Graph("silu_gate")
        x = g.input("x", f32[shape])
        gate = g.input("gate", f32[shape])
        g.output(g.silu(x, gate), "y")
        return g, {"x": _f32(rng, shape), "gate": _f32(rng, shape)}

    return build


def _step(shape):
    def build(rng):
        g = Graph("step")
        x = g.input("x", f32[shape])
        g.output(g.step(x, threshold=0.0), "y")
        return g, {"x": _f32(rng, shape)}

    return build


def _dropout_mask(shape):
    def build(rng):
        g = Graph("dropout_mask")
        x = g.input("x", f32[shape])
        g.output(g.dropout_mask(x, p=0.25, seed=1234), "y")
        return g, {"x": _f32(rng, shape)}

    return build


def _conv_tiled(x_shape, w_shape):
    """分块卷积：K 与输出总量都要落在 ``conv_tiled_preferred`` 的判据内。"""

    def build(rng):
        g = Graph("conv_tiled")
        x = g.input("x", f32[x_shape])
        w = g.input("w", f32[w_shape])
        g.output(g.conv2d(x, w, padding=(1, 1)), "y")
        return g, {"x": _f32(rng, x_shape), "w": _f32(rng, w_shape, 0.1)}

    return build


def _softmax(shape):
    def build(rng):
        g = Graph("softmax")
        x = g.input("x", f32[shape])
        g.output(g.softmax(x), "y")
        return g, {"x": _f32(rng, shape)}

    return build


def _reduce_sum(shape):
    def build(rng):
        g = Graph("reduce_sum")
        x = g.input("x", f32[shape])
        g.output(g.reduce_sum(x), "y")
        return g, {"x": _f32(rng, shape)}

    return build


def _layernorm(shape):
    cols = shape[-1]

    def build(rng):
        g = Graph("layernorm")
        x = g.input("x", f32[shape])
        gamma = g.input("gamma", f32[(cols,)])
        beta = g.input("beta", f32[(cols,)])
        g.output(g.layernorm(x, gamma, beta), "y")
        return g, {
            "x": _f32(rng, shape),
            "gamma": _f32(rng, (cols,)),
            "beta": _f32(rng, (cols,)),
        }

    return build


def _sig(*shapes, dtype="f32") -> Dict[str, object]:
    return {"dtype": dtype, "inputs": [list(shape) for shape in shapes]}


TUNE_CASES: Tuple[TuneCase, ...] = (
    TuneCase("add_same_1m", "add", _binary("add", (1024, 1024)), _sig((1024, 1024), (1024, 1024))),
    TuneCase(
        "add_broadcast",
        "add",
        _binary("add_bcast", (4, 32, 64), (64,)),
        _sig((4, 32, 64), (64,)),
    ),
    TuneCase("relu_1m", "relu", _unary("relu", (1024, 1024)), _sig((1024, 1024))),
    TuneCase("gelu_1m", "gelu", _unary("gelu", (1024, 1024)), _sig((1024, 1024))),
    TuneCase("sigmoid_1m", "sigmoid", _unary("sigmoid", (1024, 1024)), _sig((1024, 1024))),
    TuneCase("silu_gate", "silu", _silu_gate((8, 128, 256)), _sig((8, 128, 256), (8, 128, 256))),
    TuneCase("step_1m", "step", _step((1024, 1024)), _sig((1024, 1024))),
    TuneCase("dropout_mask_1m", "dropout_mask", _dropout_mask((1024, 1024)), _sig((1024, 1024))),
    # 分块卷积的旋钮是 tile_w（不是 block）：K = 2*3*3 = 18 <= 32，
    # 输出 32*64*512 = 1M >= 4096，auto 会选分块核，tile_w 才是唯一变量。
    TuneCase(
        "conv2d_tiled_1m",
        "conv2d",
        _conv_tiled((1, 2, 64, 512), (32, 2, 3, 3)),
        _sig((1, 2, 64, 512), (32, 2, 3, 3)),
    ),
    # 5.160：归约类的 block 也开放了。三条用例分别压住三种骨架——带缩放的
    # softmax、无参数的 reduce_sum、带仿射参数的 layernorm；列数都远大于 1024，
    # 覆盖值必然落在"block < cols、每线程跨步多走"的分支上。
    TuneCase("softmax_last", "softmax", _softmax((512, 4096)), _sig((512, 4096))),
    TuneCase(
        "reduce_sum_last", "reduce_sum", _reduce_sum((512, 8192)), _sig((512, 8192))
    ),
    TuneCase(
        "layernorm_last",
        "layernorm",
        _layernorm((256, 2048)),
        _sig((256, 2048), (2048,), (2048,)),
    ),
)


def _device_name(backend) -> str:
    info = getattr(backend, "device_info", None)
    if info is None:
        return "unknown"
    try:
        return str(info().name)
    except Exception:  # noqa: BLE001 - 设备信息拿不到不该让调优失败
        return "unknown"


def _require_tunable(backend) -> None:
    if not hasattr(backend, "tuning_space") or not hasattr(backend, "set_tuning"):
        raise SystemExit(
            f"{getattr(backend, 'name', backend)} 没有声明可调参数："
            "自动调优目前只对实现了 tuning_space/set_tuning 的后端有意义"
        )


def search(
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
    min_same_sign: int = 0,
    progress=None,
) -> ConfigStore:
    """对每个用例搜索一次，把**每个**候选的赢家写进库（含"默认仍最优"）。

    ``paired`` 为真（默认）时用交替 A/B：``rounds`` 是交替轮数（默认 8，
    至少 2），赢家只有在配对差的置信区间整体越过噪声地板时才会替换默认。
    ``order`` 选择轮内顺序：``alternate`` 逐轮反转，``shuffle`` 用固定种子
    每轮随机排列（后者用来把轮内的位置效应打散，见 ``tuning/paired.py``）。
    ``max_round_seconds`` 是轮内跨度上限（默认 ``None`` 不检查）：一轮把全部
    候选连测一遍的耗时超过它时，这一轮的样本不再共享同一段整机状态，所有
    非默认候选记为 ``long-round``、保留默认（5.142）。
    ``warmup_rounds`` 是正式轮之前的预热轮数（默认 0）：预热轮样本丢弃、不计
    跨度，用来把首轮冷启动挪出统计（5.143）。
    ``max_round_factor`` 是相对跨度门槛（默认 ``None``）：上限 = 该倍数 ×
    预热轮平均耗时，因此跟着机器与候选数缩放；需要 ``warmup_rounds >= 1``，
    且与 ``max_round_seconds`` 互斥（5.148）。
    ``staged`` 为真时改用两阶段搜索（``tune_staged``）：先扫一个键、再在赢家
    下扫剩下的键，最后做一次"默认 vs 组合"的两候选对照。它把 conv2d 的 45
    个候选拆成 4 + 11 + 2，单轮跨度从 2.2～2.5 秒降到 0.6 秒以内（5.144）。
    ``repeats`` > 1 时改用跨采集复现（``tune_replicated``）：同一空间独立采
    ``repeats`` 次，只有每次都判更快的候选才替换默认；与 ``staged`` 互斥
    （两阶段的候选集可能每次不同，无法逐候选对齐，5.146）。
    ``vote`` 选择合并方式：``all``（默认）要求每次判定相同；``majority`` 允许
    少数次不同、超过半数同向即采纳，需要 ``repeats >= 3``（5.149）。它只对
    跨采集复现有意义，因此与 ``serial`` / ``staged`` 和 ``repeats == 1`` 互斥。
    ``paired`` 为假时退回旧测法：每个候选连续测 ``rounds`` 次（默认 2），
    按中位数排名。

    ``min_same_sign`` 是同号轮次门槛（5.160，默认 0 不检查）：设为 ``rounds``
    时要求每一轮都指向同一方向才换默认配置。它与 ``repeats`` 是两道独立的
    闸门——前者在单次采集内逐轮核对方向，后者要求结论换个时间段仍然成立；
    两者可以叠加，也可以各自单独使用。
    """
    _require_tunable(backend)
    if not paired and min_same_sign:
        raise ValueError(
            "--min-same-sign 建立在交替 A/B 的逐轮配对差上：--serial 没有配对差"
        )
    device = _device_name(backend)
    store = ConfigStore()
    if rounds is None:
        rounds = 8 if paired else 2
    if isinstance(repeats, bool) or not isinstance(repeats, int):
        raise TypeError(f"repeats 必须是整数，得到 {type(repeats).__name__}")
    if repeats < 1:
        raise ValueError(f"repeats 至少为 1，得到 {repeats}")
    if vote not in (VOTE_ALL, VOTE_MAJORITY):
        raise ValueError(f"vote 必须是 {VOTE_ALL!r} 或 {VOTE_MAJORITY!r}，得到 {vote!r}")
    if vote != VOTE_ALL:
        if not paired:
            raise ValueError("--vote 建立在交替 A/B 上：--serial 没有跨采集判定可投票")
        if staged:
            raise ValueError("--vote 建立在跨采集复现上：--staged 与 --repeats 本就互斥")
        if repeats <= 1:
            raise ValueError(
                f"--vote {vote} 需要 --repeats >= 2：只有一次采集时没有票可投"
            )
    if paired and staged and repeats > 1:
        raise ValueError(
            "--staged 与 --repeats 不能同时使用：两阶段搜索的候选集可能每次不同，"
            "无法逐候选判断是否复现"
        )
    if max_round_factor is not None:
        if max_round_seconds is not None:
            raise ValueError(
                "--max-round-seconds 与 --max-round-factor 互斥：绝对秒数与预热轮"
                "倍数不能同时作为跨度门槛"
            )
        if warmup_rounds < 1:
            raise ValueError(
                "--max-round-factor 需要 --warmup-rounds >= 1：相对门槛的基准就是"
                "预热轮的全候选连测耗时"
            )

    for case in cases:
        graph, feed = case.build(np.random.default_rng(0))
        space = backend.tuning_space(case.op)

        def measure_config(config):
            backend.set_tuning(case.op, config)
            return measure(backend, graph, feed, case.warmup, case.repeat)["median_ms"]

        try:
            if paired and repeats > 1:
                result = tune_replicated(
                    space, measure_config, op=f"{case.name}:{case.op}",
                    repeats=repeats, rounds=rounds, noise_floor=noise_floor,
                    order=order, warmup_rounds=warmup_rounds,
                    max_round_seconds=max_round_seconds,
                    max_round_factor=max_round_factor,
                    vote=vote,
                    min_same_sign=min_same_sign,
                )
            elif paired and staged:
                result = tune_staged(
                    space, measure_config, op=f"{case.name}:{case.op}",
                    rounds=rounds, noise_floor=noise_floor, order=order,
                    warmup_rounds=warmup_rounds,
                    max_round_seconds=max_round_seconds,
                    max_round_factor=max_round_factor,
                    min_same_sign=min_same_sign,
                )
            elif paired:
                result = tune_paired(
                    space, measure_config, op=f"{case.name}:{case.op}",
                    rounds=rounds, noise_floor=noise_floor, order=order,
                    warmup_rounds=warmup_rounds,
                    max_round_seconds=max_round_seconds,
                    max_round_factor=max_round_factor,
                    min_same_sign=min_same_sign,
                )
            else:
                result = tune(space, measure_config, op=f"{case.name}:{case.op}", rounds=rounds)
        except AllRejectedError:
            # 全被拒是"这个空间在当前后端一个都跑不起来"，必须冒出来而不是记一条空库
            raise
        finally:
            backend.clear_tuning(case.op)

        store.put(
            TuningRecord(
                backend=backend.name,
                device=device,
                op=case.op,
                signature=dict(case.signature),
                config=dict(result.best.config),
                median_ms=float(result.best.median_ms),
                best_ms=result.best.best_ms,
                p95_ms=result.best.p95_ms,
                samples=result.best.samples,
            )
        )
        if progress is not None:
            progress(case, result)

    return store


def apply_and_verify(
    backend,
    store: ConfigStore,
    cases: Sequence[TuneCase] = TUNE_CASES,
    *,
    progress=None,
) -> Tuple[int, List[str]]:
    """应用库里命中的配置，并用默认配置对拍数值。

    返回 ``(命中数, 问题列表)``。未命中的用例不算问题——配置库本来就可能
    只覆盖部分形状；但如果命中后数值不一致，那是必须修的问题。
    """
    _require_tunable(backend)
    device = _device_name(backend)
    problems: List[str] = []
    hits = 0

    for case in cases:
        record = store.get(backend.name, device, case.op, case.signature)
        if record is None:
            continue
        hits += 1
        graph, feed = case.build(np.random.default_rng(0))
        backend.clear_tuning(case.op)
        expected = backend.execute(graph, feed)["y"]
        default_ms = measure(backend, graph, feed, case.warmup, case.repeat)["median_ms"]

        backend.set_tuning(case.op, record.config)
        try:
            actual = backend.execute(graph, feed)["y"]
            tuned_ms = measure(backend, graph, feed, case.warmup, case.repeat)["median_ms"]
        finally:
            backend.clear_tuning(case.op)

        if not np.array_equal(np.asarray(expected), np.asarray(actual)):
            problems.append(
                f"{case.name}（{case.op}）：应用 {dict(record.config)} 后与默认结果不一致"
            )
        if progress is not None:
            progress(case, record, default_ms, tuned_ms)

    return hits, problems


_VERDICT_TEXT = {
    "faster": "更快",
    "slower": "更慢",
    "noise": "噪声内",
    "unstable": "批次不稳",
    "long-round": "跨度超限",
    "unreplicated": "未复现",
    "baseline": "基线",
    "no-baseline": "无法配对",
    "rejected": "不可行",
}


def _print_search(case: TuneCase, result) -> None:
    paired = getattr(result, "method", "serial") == "paired"
    for item in result.candidates:
        if not item.feasible:
            print(f"    {dict(item.config)!s:24s} 不可行：{item.reason}")
            continue
        line = (
            f"    {dict(item.config)!s:24s} {item.median_ms:10.4f} ms"
            f"  p95 {item.p95_ms:10.4f} ms  best {item.best_ms:10.4f} ms"
        )
        if paired and item.diff_median_pct is not None:
            line += (
                f"  Δ中位 {item.diff_median_pct:+6.2f}%"
                f"  均值 [{item.ci_low_pct:+6.2f}, {item.ci_high_pct:+6.2f}]"
                f"  {_VERDICT_TEXT.get(item.verdict, item.verdict)}"
            )
        if getattr(item, "votes", ()):
            line += "  票 " + "/".join(
                f"{_VERDICT_TEXT.get(name, name)}×{count}"
                for name, count in item.votes
            )
        if getattr(item, "replicate_diffs", ()):
            # 票数说"判了几次更快"，逐次差说"每次差多少"：两者并排看，才能
            # 分清那张 noise 是幅度不足（差小）还是方向反转（差为正），
            # 也才能判断这次多数票是稳定同向还是恰好凑数（5.150）
            line += "  逐次 " + " / ".join(
                "—" if value is None else f"{value:+.2f}%"
                for value in item.replicate_diffs
            )
        print(line)
    speedup = result.speedup
    tail = f"，相对默认 {speedup:.2f}x" if speedup else ""
    if paired:
        tail += (
            f"（{result.rounds} 轮，order={getattr(result, 'order_mode', 'alternate')}，"
        )
        if getattr(result, "warmup_rounds", 0):
            tail += f"预热 {result.warmup_rounds} 轮 {result.warmup_seconds:.2f}s，"
        if getattr(result, "repeats", 1) > 1:
            how = "多数票" if getattr(result, "vote", None) == VOTE_MAJORITY else "全票"
            tail += f"重复 {result.repeats} 次（{how}），"
        tail += f"噪声地板 {result.noise_floor:.1%}"
        seconds = getattr(result, "round_seconds", ())
        if seconds:
            tail += f"，轮内最长 {max(seconds):.2f}s"
        budget = getattr(result, "round_budget_seconds", None)
        if budget is not None:
            tail += f"，上限 {budget:.2f}s"
            factor = getattr(result, "max_round_factor", None)
            if factor is not None:
                tail += f"（{factor:g}× 预热）"
        tail += "）"
    print(f"  -> {case.name:20s} {case.op:12s} 最优 {dict(result.best.config)}{tail}")
    if paired:
        unstable = sum(
            1 for item in result.candidates if getattr(item, "verdict", "") == "unstable"
        )
        long_round = sum(
            1 for item in result.candidates if getattr(item, "verdict", "") == "long-round"
        )
        unreplicated = sum(
            1 for item in result.candidates if getattr(item, "verdict", "") == "unreplicated"
        )
        if getattr(result.best, "decided", False):
            print("     判定：配对差超出噪声，替换默认")
        elif long_round:
            limit = result.max_round_seconds
            print(
                f"     判定：{long_round} 个候选的轮内跨度超过上限 {limit:.2f}s"
                "（轮内整机状态不再可比），保留默认"
            )
        elif unstable:
            print(f"     判定：{unstable} 个候选前后两半方向相反（批次不稳），保留默认")
        elif unreplicated:
            detail = (
                "各次采集方向冲突或次数不足"
                if getattr(result, "vote", None) == VOTE_MAJORITY
                else "换一次采集就不成立"
            )
            print(
                f"     判定：{unreplicated} 个候选未通过跨采集复现"
                f"（{detail}），保留默认"
            )
        else:
            print("     判定：差异落在噪声内，保留默认")
        for stage in getattr(result, "stages", ()):
            keys = ",".join(stage["keys"])
            longest = max(stage["round_seconds"])
            print(
                f"     阶段 {keys}: {dict(stage['winner'])}"
                f"（{'已判定' if stage['decided'] else '保留默认'}，"
                f"最长 {longest:.2f}s，跨度超限 {stage['long_round']} 个）"
            )


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="按算子搜索 block 配置")
    parser.add_argument(
        "--backend",
        choices=tuple(n for n in backend_names() if n != "ref"),
        default="cuda",
    )
    parser.add_argument("--save", help="把搜索到的配置库写到该 JSON")
    parser.add_argument("--apply", help="加载配置库并做数值对拍")
    parser.add_argument(
        "--rounds", type=int, default=None,
        help="交替 A/B 的轮数（默认 8，至少 2）；--serial 时默认 2，表示每个候选连续测几次",
    )
    parser.add_argument(
        "--serial", action="store_true",
        help="退回串行测量（每个候选连续测 N 次），只用于对照，不产生置信区间",
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
        "--min-same-sign", type=int, default=0, metavar="N",
        help="同号轮次门槛：至少 N 轮与中位差同向才换默认，否则判 unstable"
             "（默认 0 不检查；典型用法 --rounds 5 --min-same-sign 5）",
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
    if args.min_same_sign:
        if args.serial:
            parser.error("--min-same-sign 建立在交替 A/B 的逐轮配对差上：--serial 没有配对差")
        if args.min_same_sign < 0:
            parser.error("--min-same-sign 不能为负")
        if args.min_same_sign == 1:
            parser.error(
                "--min-same-sign=1 没有意义：一轮谈不上同号一致性；用 0 关闭或 >= 2"
            )
        effective_rounds = args.rounds if args.rounds is not None else 8
        if args.min_same_sign > effective_rounds:
            parser.error(
                f"--min-same-sign({args.min_same_sign}) 不能超过 --rounds({effective_rounds})"
            )

    backend = open_backend(args.backend)
    print(f"调优 {len(TUNE_CASES)} 个单算子用例（{backend.name} / {_device_name(backend)}）")

    if args.apply:
        store = ConfigStore.load(args.apply)
        hits, problems = apply_and_verify(
            backend,
            store,
            progress=lambda case, record, default_ms, tuned_ms: print(
                f"  {case.name:20s} 默认 {default_ms:8.4f} ms -> "
                f"{dict(record.config)} {tuned_ms:8.4f} ms "
                f"({default_ms / tuned_ms:.2f}x)"
            ),
        )
        print(f"命中 {hits} 条配置")
        if problems:
            for item in problems:
                print(f"  [FAIL] {item}")
            return 1
        print("对拍通过：所有命中配置与默认结果逐位一致")
        return 0

    store = search(
        backend,
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
        min_same_sign=args.min_same_sign,
        progress=_print_search,
    )
    if args.save:
        store.save(args.save)
        print(f"已写入 {args.save}（{len(store)} 条）")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
