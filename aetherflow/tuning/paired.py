"""交替 A/B 配对测量：把配置差异从整机漂移里摘出来。

5.63 的实测结论是：串行测量下"最优 block"不可复现——同一台机器两次采集
只有 2/9 与 4/9 重合，HIP 上整机状态在测量过程中会抬升 30%（WSL2 + DXG
路径），而 1M 逐元素算子的四档 ``block`` 真实差异 ≤3%。串行测量把"先测的
候选"与"后测的候选"放在不同的整机状态里，于是"谁赢"跟着测量顺序走。

这里换一种测法，四件事缺一不可：

1. **轮内连测**：每一轮把所有候选连着测一遍，让它们共享同一段整机状态；
2. **逐轮反转**：相邻两轮改变顺序，使"先测者优势"在每个候选上平均抵消；
   只做 1 不做 2 的话，固定顺序会把顺序效应原样留在结果里。注意它只抵消
   **单调**的顺序效应：若一轮内的耗时曲线是"中间快、两头慢"（GPU 升频后
   又回落），反转会让"开头"和"结尾"都按慢算，位置偏置仍然存在——5.140 的
   三次 conv2d 采集给出三个不同的赢家，主因就在这里。``order="shuffle"``
   用固定种子逐轮随机排列，让每个候选在轮内的位置期望均匀；默认仍是
   ``alternate``，因为 5.140 的全部采集都按它做，换默认要先有 A/B 证据；
3. **配对差**：用**同一轮内**的 ``candidate / baseline`` 做比值。整机漂移
   在分子分母里同时出现，被约掉；跨轮台阶因此不再冒充配置差异；
4. **区间判定**：对配对差重采样给出均值差的置信区间。区间跨 0、或中位差
   落在 ``noise_floor`` 以内，就判定"差异落在噪声内"，保留默认配置。

**为什么还要查"两半是否同向"。** 5.140 的第一次实测就撞上了 bootstrap 的
盲区：同一配置（conv2d 的 ``tile_w=64 / roc=4``）在相隔几分钟的两次采集里，
一次 CI ``[-21.98, -1.89]`` 判"更快"，另一次 CI ``[+2.39, +19.43]`` 判"更慢"。
两次都"显著"，方向却相反。原因是 8 个轮次**自相关**：整机状态在一轮内缓慢
漂移，bootstrap 把轮次当独立样本重采样，区间因此过窄。于是追加一条启发式
检查：把轮次按时间切成前后两半，两半的配对差中位数必须同号，否则记为
``unstable`` 并按噪声处理。它不提供新的统计保证，只回答一个更朴素的问题——
"这个结论下次还会是同方向吗"。第三次采集里，8 个逐元素用例的候选有 13 个
被这条检查拦下（其中 add_broadcast 的 ``block=512`` 在第二次采集里还被判过
"更快"）。

**为什么用 bootstrap 而不是 t 区间。** 样本只有几轮，延迟分布又明显偏斜
（偶发的一轮停顿会把均值拖走）。bootstrap 不假设正态，只回答"把这批样本
重采样，均值能晃到哪"；代价是需要固定随机种子才能复现——``seed`` 默认 0，
同一批样本永远给出同一个区间。

**为什么赢家按中位差选，而不是均值差。** 区间用均值（可加、有明确的标准误
含义），选赢家用中位差（对单轮异常稳健）。两者都报出来，读者能看到"均值被
一轮拖走、中位数没有"这种分歧，而不是只拿到一个漂亮的数字。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .search import AllRejectedError, Measure, _percentile, _valid
from .space import TuningSpace

__all__ = [
    "ORDER_ALTERNATE",
    "ORDER_SHUFFLE",
    "VOTE_ALL",
    "VOTE_MAJORITY",
    "PairedCandidate",
    "PairedResult",
    "tune_paired",
    "tune_replicated",
    "tune_staged",
]

# 判定用词保持稳定：打印层把它们翻成中文，报告层原样落盘
FASTER = "faster"
SLOWER = "slower"
NOISE = "noise"
UNSTABLE = "unstable"
LONG_ROUND = "long-round"
UNREPLICATED = "unreplicated"
BASELINE = "baseline"
REJECTED = "rejected"
NO_BASELINE = "no-baseline"

# 轮内顺序策略
ORDER_ALTERNATE = "alternate"  # 偶正序、奇反序
ORDER_SHUFFLE = "shuffle"      # 每轮用固定种子随机排列
_ORDERS = (ORDER_ALTERNATE, ORDER_SHUFFLE)

# 跨采集复现的合并方式
VOTE_ALL = "all"            # 每次判定必须完全相同（5.146 的口径）
VOTE_MAJORITY = "majority"  # 允许少数次不同，按多数判定（5.149）
_VOTES = (VOTE_ALL, VOTE_MAJORITY)


@dataclass(frozen=True)
class PairedCandidate:
    """一个候选的配对测量结果；``median_ms`` 为 ``None`` 表示不可行。

    ``diff_*`` 是相对**默认候选**（空间里的第一个）的配对差，单位是百分比
    （负值代表更快）。``ci_low/ci_high`` 是均值差的 ``1 - alpha`` 区间；
    ``verdict`` 是判定结果，只有 ``faster`` / ``slower`` 才算"差异超出噪声"。
    """

    config: Mapping[str, Any]
    median_ms: Optional[float] = None
    best_ms: Optional[float] = None
    p95_ms: Optional[float] = None
    samples: int = 0
    reason: str = ""
    diff_median_pct: Optional[float] = None
    diff_mean_pct: Optional[float] = None
    ci_low_pct: Optional[float] = None
    ci_high_pct: Optional[float] = None
    first_half_pct: Optional[float] = None
    second_half_pct: Optional[float] = None
    # 与中位差同号的轮次数；``min_same_sign`` 为 0（未检查）时恒为 0
    same_sign: int = 0
    verdict: str = REJECTED
    # 跨采集复现时每次采集的判定票数（判定词 -> 次数），按票数降序、同票按词排序；
    # 单次采集里恒为空
    votes: Tuple[Tuple[str, int], ...] = ()
    # 跨采集复现时每次采集的配对差中位数（百分比），顺序与 ``replicates`` 对齐；
    # 那一次不可行（或没有可配对轮次）记 ``None``。单次采集里恒为空
    replicate_diffs: Tuple[Optional[float], ...] = ()

    @property
    def feasible(self) -> bool:
        return self.median_ms is not None

    @property
    def decided(self) -> bool:
        """配对差是否超出噪声：只有它成立，才谈得上"谁更快"。"""
        return self.verdict in (FASTER, SLOWER)

    @property
    def stable(self) -> bool:
        """前后两半同向、轮内跨度未超限、且跨采集复现；三者缺一都不算结论。"""
        return self.verdict not in (UNSTABLE, LONG_ROUND, UNREPLICATED)

    @property
    def spread(self) -> Optional[float]:
        """``(p95 - p50) / p50``：轮内中位数的离散程度；中位数为 0 时为 ``None``。"""
        if self.median_ms is None or not self.p95_ms or not self.median_ms:
            return None
        return round((self.p95_ms - self.median_ms) / self.median_ms, 4)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "config": dict(self.config),
            "samples": self.samples,
            "verdict": self.verdict,
        }
        if self.feasible:
            out["median_ms"] = self.median_ms
            out["best_ms"] = self.best_ms
            out["p95_ms"] = self.p95_ms
            out["spread"] = self.spread
        else:
            out["rejected"] = self.reason
        if self.diff_median_pct is not None:
            out["diff_median_pct"] = self.diff_median_pct
            out["diff_mean_pct"] = self.diff_mean_pct
            out["ci_low_pct"] = self.ci_low_pct
            out["ci_high_pct"] = self.ci_high_pct
            out["first_half_pct"] = self.first_half_pct
            out["second_half_pct"] = self.second_half_pct
        if self.same_sign:
            out["same_sign"] = self.same_sign
        if self.votes:
            out["votes"] = dict(self.votes)
        if self.replicate_diffs:
            out["replicate_diffs"] = list(self.replicate_diffs)
        return out


@dataclass(frozen=True)
class PairedResult:
    """一次配对测量的完整结果。

    字段与 :class:`~aetherflow.tuning.search.TuningResult` 保持同名同义
    （``best`` / ``candidates`` / ``speedup`` / ``rejected``），因此采集报告
    与配置库不需要知道测量方式变了；额外的 ``order`` / ``rounds`` /
    ``noise_floor`` 记录"这次凭什么下结论"。
    """

    best: PairedCandidate
    candidates: Tuple[PairedCandidate, ...]
    rounds: int
    noise_floor: float
    order: Tuple[Any, ...]
    bootstrap: int
    alpha: float
    order_mode: str = ORDER_ALTERNATE
    method: str = "paired"
    warmup_rounds: int = 0
    warmup_seconds: float = 0.0
    round_seconds: Tuple[float, ...] = ()
    max_round_seconds: Optional[float] = None
    # 相对跨度门槛：``max_round_factor`` 是"预热轮耗时的倍数"，实际算出的
    # 秒数记在 ``round_budget_seconds``（绝对门槛时就是 ``max_round_seconds``）
    max_round_factor: Optional[float] = None
    round_budget_seconds: Optional[float] = None
    # 两阶段搜索的逐阶段摘要；一次 ``tune_paired`` 的结果里恒为空
    stages: Tuple[Mapping[str, Any], ...] = ()
    # 跨采集复现的重复次数与逐次摘要；由 ``tune_replicated`` 填充
    repeats: int = 1
    replicates: Tuple[Mapping[str, Any], ...] = ()
    # 跨采集复现的合并方式（``all`` / ``majority``）；单次采集为 ``None``，
    # 因为那时没有"票"可数
    vote: Optional[str] = None
    # 同号轮次门槛；0 表示未检查（5.140～5.149 的口径）
    min_same_sign: int = 0

    @property
    def baseline(self) -> PairedCandidate:
        return self.candidates[0]

    @property
    def rejected(self) -> Tuple[PairedCandidate, ...]:
        return tuple(item for item in self.candidates if not item.feasible)

    @property
    def tie_epsilon(self) -> float:
        """兼容 ``TuningResult`` 的字段名：这里的"平局阈值"就是噪声地板。"""
        return self.noise_floor

    @property
    def decided(self) -> bool:
        """赢家是否真的区别于默认（而不是"证据不足，保留默认"）。"""
        return self.best is not self.candidates[0] and self.best.decided

    @property
    def speedup(self) -> Optional[float]:
        """赢家相对默认的典型延迟加速比；默认不可行时为 ``None``。"""
        default = self.candidates[0].median_ms
        if default is None or self.best.median_ms is None or not default:
            return None
        return round(default / self.best.median_ms, 4)

    @property
    def best_speedup(self) -> Optional[float]:
        """按最佳可达（轮内最小值）算的加速比；仅作参考，不作排名依据。"""
        default = self.candidates[0].best_ms
        best = self.best.best_ms
        if not default or not best:
            return None
        return round(default / best, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "order_mode": self.order_mode,
            "rounds": self.rounds,
            "order": list(self.order),
            "noise_floor": self.noise_floor,
            "bootstrap": self.bootstrap,
            "alpha": self.alpha,
            "warmup_rounds": self.warmup_rounds,
            "warmup_seconds": self.warmup_seconds,
            "round_seconds": list(self.round_seconds),
            "max_round_seconds": self.max_round_seconds,
            "max_round_factor": self.max_round_factor,
            "round_budget_seconds": self.round_budget_seconds,
            "stages": [dict(stage) for stage in self.stages],
            "repeats": self.repeats,
            "replicates": [dict(item) for item in self.replicates],
            "vote": self.vote,
            "min_same_sign": self.min_same_sign,
            "best": self.best.to_dict(),
            "baseline": dict(self.candidates[0].config),
            "decided": self.decided,
            "candidates": [item.to_dict() for item in self.candidates],
            "speedup": self.speedup,
            "best_speedup": self.best_speedup,
        }


# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------


def _bootstrap_ci(
    diffs: Sequence[float],
    *,
    bootstrap: int,
    alpha: float,
    seed: int,
) -> Tuple[float, float]:
    """均值差的百分位 bootstrap 区间。

    样本量很小时（<2）区间没有意义，调用方应先排除；这里仍做一次显式检查，
    避免"1 个样本也算出 ±0"这种看起来精确、实际是虚构的结果。
    """
    arr = np.asarray(diffs, dtype=np.float64)
    if arr.size < 2:
        raise ValueError(f"配对差至少需要 2 轮，得到 {arr.size}")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, arr.size, size=(bootstrap, arr.size))
    means = arr[draws].mean(axis=1)
    low, high = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(low), float(high)


def _verdict(
    median_pct: float,
    ci_low: float,
    ci_high: float,
    floor_pct: float,
    first_half: float,
    second_half: float,
) -> str:
    """区间、噪声地板与批次稳定性共同决定"这个差能不能当结论"。

    - 两半异号：整机状态在采集期间换过方向，结论不能跨采集复现，判 ``unstable``；
    - 区间整体落在 0 的另一侧（方向可复现），且中位差超过噪声地板（幅度值得
      换配置），才算更快 / 更慢。

    只满足区间条件时，0.1% 的"稳定更快"仍然不该替换默认——它省不下任何时间，
    却让配置库多了一条不可复现的记录。
    """
    if (first_half < 0.0) != (second_half < 0.0):
        return UNSTABLE
    if ci_high < 0.0 and median_pct <= -floor_pct:
        return FASTER
    if ci_low > 0.0 and median_pct >= floor_pct:
        return SLOWER
    return NOISE


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------


def tune_paired(
    space: TuningSpace,
    measure: Measure,
    *,
    op: str = "",
    rounds: int = 8,
    noise_floor: float = 0.01,
    bootstrap: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    order: str = ORDER_ALTERNATE,
    warmup_rounds: int = 0,
    max_round_seconds: Optional[float] = None,
    max_round_factor: Optional[float] = None,
    min_same_sign: int = 0,
) -> PairedResult:
    """交替 A/B 测完 ``space`` 的全部候选，返回"差异是否超出噪声"的结论。

    ``rounds`` 是**交替轮数**（每轮每个候选测一次），至少 2 轮——只有 1 轮时
    顺序效应无法抵消，那不是配对比较。``measure`` 每调用一次返回该轮的一个
    样本（通常是内部多次执行的中位数）。

    ``order`` 决定轮内顺序：

    - ``alternate``（默认）：偶数轮正序、奇数轮反序。只抵消**单调**的顺序效应；
    - ``shuffle``：每轮用固定种子随机排列。5.141 的 A/B 显示两种排列都没能让
      赢家可复现，但随机排列至少让"位置偏置"变成方差而不是固定偏置。

    ``max_round_seconds`` 是轮内跨度的上限（秒），默认 ``None`` 表示不检查。
    5.141 的实测显示 conv2d 一轮 45 个候选约 2.3 秒、逐元素一轮 4 个候选约
    0.26 秒，前者四次采集里两次判收益、两次判噪声。超限时所有非默认候选记
    ``long-round``：区间照算，但"谁更快"在这个采集里不成立。

    ``max_round_factor`` 是**相对**跨度门槛：上限 = 该倍数 × 预热轮平均耗时。
    5.147 显示同一个绝对秒数在 M40 上从不触发、在 WSL2/DXG 上 44/45 个候选
    触发，因此"多长算长"要跟着机器与候选数缩放。预热轮本身就是一次全候选
    连测，拿它当标尺不需要额外测量；代价是必须预热（``warmup_rounds >= 1``），
    否则没有可用的基准。它与 ``max_round_seconds`` 互斥，两者同给会被拒绝：
    同时用两把尺子时无法解释是哪一把拦下了结论。

    ``warmup_rounds`` 是**预热轮数**，默认 0（保持 5.140～5.142 的口径）。
    预热轮按正序把全部候选连测一遍，样本丢弃、不计入 ``order`` 与
    ``round_seconds``，也不消耗 ``shuffle`` 的排列随机流——因此同一 ``seed``
    下预热 0 轮与 1 轮的正式轮排列相同，两者只差"冷启动有没有跑掉"。
    5.142 的实测里首轮比后续轮次慢 1.3～14 倍，这段冷启动会同时污染样本与
    跨度门槛，预热轮就是把它挪出去。

    ``min_same_sign`` 是同号轮次门槛（5.160），默认 0 表示不检查、保持
    5.140～5.149 的口径。设为 ``N`` 时，只有"至少 ``N`` 轮与中位差同号"的
    候选才保留 ``faster`` / ``slower``；否则降为 ``unstable``——方向在轮次之间
    翻过，这不是噪声幅度不够，而是"下次还会不会是同一个方向"没有答案。它与
    ``unstable``（前后两半异号）互补：后者只比较两半的中位数，可能被一个
    中间翻转的轮次掩盖；同号计数逐轮核对。典型用法是 ``rounds=5,
    min_same_sign=5``——五轮全部同号才换默认配置。

    不可行的候选（抛异常 / 返回 None / NaN / 负数）立即停止测量并记为
    ``rejected``；它不参与任何配对差，也不会把别的候选的轮次挤乱。预热轮里
    抛异常的候选同样判不可行，正式轮不再尝试它。
    """
    if not callable(measure):
        raise TypeError(f"measure 必须可调用，得到 {type(measure).__name__}")
    if order not in _ORDERS:
        raise ValueError(f"order 必须是 {_ORDERS} 之一，得到 {order!r}")
    if isinstance(noise_floor, bool) or not isinstance(noise_floor, (int, float)):
        raise TypeError(f"noise_floor 必须是数，得到 {type(noise_floor).__name__}")
    if not 0.0 <= float(noise_floor) < 1.0:
        raise ValueError(f"noise_floor 必须在 [0, 1) 内，得到 {noise_floor!r}")
    if max_round_seconds is not None:
        if isinstance(max_round_seconds, bool) or not isinstance(max_round_seconds, (int, float)):
            raise TypeError(
                f"max_round_seconds 必须是正数或 None，得到 {type(max_round_seconds).__name__}"
            )
        if not float(max_round_seconds) > 0.0:
            raise ValueError(f"max_round_seconds 必须为正，得到 {max_round_seconds!r}")
    if max_round_factor is not None:
        if isinstance(max_round_factor, bool) or not isinstance(max_round_factor, (int, float)):
            raise TypeError(
                f"max_round_factor 必须是正数或 None，得到 {type(max_round_factor).__name__}"
            )
        if not float(max_round_factor) > 0.0:
            raise ValueError(f"max_round_factor 必须为正，得到 {max_round_factor!r}")
        if max_round_seconds is not None:
            raise ValueError(
                "max_round_seconds 与 max_round_factor 互斥：绝对秒数与预热轮倍数"
                "不能同时作为跨度门槛，否则无法解释是哪一把拦下了结论"
            )
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError(f"alpha 必须是数，得到 {type(alpha).__name__}")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha 必须在 (0, 1) 内，得到 {alpha!r}")
    rounds = int(rounds)
    if rounds < 2:
        raise ValueError(f"交替 A/B 至少需要 2 轮（1 轮无法抵消顺序效应），得到 {rounds}")
    if isinstance(warmup_rounds, bool) or not isinstance(warmup_rounds, int):
        raise TypeError(
            f"warmup_rounds 必须是整数，得到 {type(warmup_rounds).__name__}"
        )
    if warmup_rounds < 0:
        raise ValueError(f"warmup_rounds 不能为负，得到 {warmup_rounds!r}")
    if max_round_factor is not None and warmup_rounds < 1:
        raise ValueError(
            "max_round_factor 需要至少 1 个预热轮作为标尺（warmup_rounds >= 1）："
            "相对跨度门槛的基准就是预热轮的全候选连测耗时"
        )
    bootstrap = int(bootstrap)
    if bootstrap < 100:
        raise ValueError(f"bootstrap 至少 100 次才有区间意义，得到 {bootstrap}")
    if isinstance(min_same_sign, bool) or not isinstance(min_same_sign, int):
        raise TypeError(
            f"min_same_sign 必须是整数，得到 {type(min_same_sign).__name__}"
        )
    if min_same_sign < 0:
        raise ValueError(f"min_same_sign 不能为负，得到 {min_same_sign!r}")
    if min_same_sign == 1:
        raise ValueError(
            "min_same_sign=1 没有意义：一轮谈不上同号一致性；用 0 关闭检查、"
            "或用 >= 2"
        )
    if min_same_sign > rounds:
        raise ValueError(
            f"min_same_sign({min_same_sign}) 不能超过交替轮数 {rounds}："
            "要求比实际测过的轮次更多的一致性无法成立"
        )

    configs = [dict(item) for item in space.configs()]
    count = len(configs)
    # samples[idx][轮次] = 该轮样本；按轮存而不是按位置存，配对时才能确认"同一轮"
    samples: List[Dict[int, float]] = [dict() for _ in range(count)]
    reasons: List[str] = ["" for _ in range(count)]
    order_log: List[Any] = []
    round_seconds: List[float] = []
    # 排列与 bootstrap 用两个独立的 rng：同一个 seed 不该让"顺序"和"重采样"
    # 产生人为相关。异或常量是 fuzz.py 用过的做法（那边是 seed ^ 0xA5A5）。
    shuffle_rng = np.random.default_rng((int(seed) ^ 0x5EED) & 0xFFFFFFFF)

    def _visit(indices, turn: Optional[int]) -> None:
        """连测 ``indices`` 里的候选；``turn`` 为 ``None`` 表示预热轮、样本丢弃。"""
        for idx in indices:
            if reasons[idx]:
                continue
            try:
                value = _valid(measure(configs[idx]))
            except Exception as exc:  # noqa: BLE001 - 任何异常都意味着该配置不可行
                reasons[idx] = f"{type(exc).__name__}: {exc}"
                continue
            if value is None:
                reasons[idx] = "测量结果不是有限的非负数"
                continue
            if turn is not None:
                samples[idx][turn] = value

    # 预热轮固定正序、不写样本、不计跨度、不消耗 shuffle_rng：它唯一的作用是
    # 把首轮冷启动跑掉，不该改变正式轮的排列或统计口径。
    warmup_seconds = 0.0
    for _ in range(warmup_rounds):
        started = time.perf_counter()
        _visit(range(count), None)
        warmup_seconds += time.perf_counter() - started

    for turn in range(rounds):
        if order == ORDER_SHUFFLE:
            indices = [int(i) for i in shuffle_rng.permutation(count)]
            order_log.append(indices)
        else:
            forward = turn % 2 == 0
            indices = list(range(count)) if forward else list(range(count - 1, -1, -1))
            order_log.append("forward" if forward else "reverse")
        # 轮内跨度是采集属性：它决定"这一轮的样本是否还共享同一段整机状态"。
        # 只统计真正执行了的 measure（不可行候选会被跳过），与 verdict 的
        # 降级口径保持一致。
        started = time.perf_counter()
        _visit(indices, turn)
        round_seconds.append(time.perf_counter() - started)

    if max_round_factor is not None:
        # 预热轮本身就是一次完整的全候选连测，拿它的平均耗时当标尺：门槛
        # 因此随机器、候选数与算子成本一起缩放，而不是写死一个绝对秒数
        # （5.147：同一个 1.0 秒在 M40 上不触发、在 WSL2/DXG 上 44/45 触发）。
        round_budget = float(max_round_factor) * (warmup_seconds / warmup_rounds)
    elif max_round_seconds is not None:
        round_budget = float(max_round_seconds)
    else:
        round_budget = None
    too_long = round_budget is not None and any(
        seconds > round_budget for seconds in round_seconds
    )

    baseline_samples = samples[0]
    results: List[PairedCandidate] = []
    for idx, config in enumerate(configs):
        if reasons[idx] or not samples[idx]:
            results.append(
                PairedCandidate(
                    config=config,
                    samples=len(samples[idx]),
                    reason=reasons[idx] or "没有有效测量",
                )
            )
            continue

        ordered = sorted(samples[idx].values())
        entry: Dict[str, Any] = {
            "config": config,
            "median_ms": round(_percentile(ordered, 0.5), 6),
            "best_ms": round(ordered[0], 6),
            "p95_ms": round(_percentile(ordered, 0.95), 6),
            "samples": len(ordered),
        }

        if idx == 0:
            entry.update(
                diff_median_pct=0.0,
                diff_mean_pct=0.0,
                ci_low_pct=0.0,
                ci_high_pct=0.0,
                verdict=BASELINE,
            )
            results.append(PairedCandidate(**entry))
            continue

        if not baseline_samples:
            # 基线不可行时没有任何可配对的轮次：照实说明"无法判定"，
            # 而不是拿别的候选当基线偷偷补一个差异出来。
            entry.update(verdict=NO_BASELINE)
            results.append(PairedCandidate(**entry))
            continue

        pairs = [
            (value / baseline_samples[turn] - 1.0) * 100.0
            for turn, value in sorted(samples[idx].items())
            if turn in baseline_samples and baseline_samples[turn] > 0.0
        ]
        if len(pairs) < 2:
            entry.update(verdict=NO_BASELINE, reason="可配对的轮次不足 2 轮")
            results.append(PairedCandidate(**entry))
            continue

        low, high = _bootstrap_ci(pairs, bootstrap=bootstrap, alpha=alpha, seed=seed)
        median_pct = float(np.median(np.asarray(pairs, dtype=np.float64)))
        mean_pct = float(np.mean(np.asarray(pairs, dtype=np.float64)))
        half = len(pairs) // 2
        first_half = float(np.median(np.asarray(pairs[:half], dtype=np.float64)))
        second_half = float(np.median(np.asarray(pairs[half:], dtype=np.float64)))
        floor_pct = float(noise_floor) * 100.0
        verdict = _verdict(median_pct, low, high, floor_pct, first_half, second_half)
        # 同号轮次只在启用门槛时统计；它回答"方向每一轮都成立吗"，不是
        # 再给一次幅度判定。零差异不算同号。只有原本判出方向的候选会因为
        # 同号不足被降级——``noise`` 已经是不下结论，保留它更准确。
        same_sign = 0
        if min_same_sign:
            negative = median_pct < 0.0
            same_sign = sum(
                1
                for value in pairs
                if (value < 0.0 if negative else value > 0.0)
            )
            if verdict in (FASTER, SLOWER) and same_sign < min_same_sign:
                verdict = UNSTABLE
        if too_long:
            # 轮内跨度超限：区间照算（读者能看到"差多少"），但这个差不能当结论。
            # 一轮里若整机状态抬升过，区间会又窄又偏；判定词要如实说"这轮太长"，
            # 而不是给一个看起来精确的方向。
            verdict = LONG_ROUND
        entry.update(
            diff_median_pct=round(median_pct, 4),
            diff_mean_pct=round(mean_pct, 4),
            ci_low_pct=round(low, 4),
            ci_high_pct=round(high, 4),
            first_half_pct=round(first_half, 4),
            second_half_pct=round(second_half, 4),
            same_sign=same_sign,
            verdict=verdict,
        )
        results.append(PairedCandidate(**entry))

    candidates = tuple(results)
    feasible = [item for item in candidates if item.feasible]
    if not feasible:
        raise AllRejectedError(op or "<unknown>", candidates)

    # 只有"显著更快"的候选才有资格替换默认；证据不足时默认胜出。
    # 多个候选都显著更快时取中位差最小者，平局保留先出现者（候选顺序由空间决定）。
    if candidates[0].feasible:
        winner = candidates[0]
        faster = [
            item
            for item in candidates[1:]
            if item.verdict == FASTER and item.diff_median_pct is not None
        ]
        if faster:
            winner = min(faster, key=lambda item: item.diff_median_pct)  # type: ignore[arg-type,return-value]
    else:
        # 基线不可行时没有任何配对差可用。按绝对中位数挑一个可行候选只是
        # "眼下唯一能跑的东西"，不是"更快的配置"——所有候选的 verdict 都是
        # ``no-baseline``，``decided`` 因此为假，调用方必须显式看到这一点。
        winner = min(feasible, key=lambda item: item.median_ms)  # type: ignore[arg-type,return-value]

    return PairedResult(
        best=winner,
        candidates=candidates,
        rounds=rounds,
        noise_floor=float(noise_floor),
        order=tuple(order_log),
        bootstrap=bootstrap,
        alpha=float(alpha),
        order_mode=order,
        warmup_rounds=warmup_rounds,
        warmup_seconds=round(warmup_seconds, 6),
        round_seconds=tuple(round_seconds),
        max_round_seconds=None if max_round_seconds is None else float(max_round_seconds),
        max_round_factor=(
            None if max_round_factor is None else float(max_round_factor)
        ),
        round_budget_seconds=None if round_budget is None else round(round_budget, 6),
        min_same_sign=min_same_sign,
    )


# --------------------------------------------------------------------------
# 两阶段搜索
# --------------------------------------------------------------------------


class _FixedSpace:
    """只暴露 ``configs()`` 的最小适配器：把"固定若干键"的子空间交给配对测量。

    两阶段搜索需要测"``tile_w=64`` 下 ``roc × cb`` 的全部组合"，这些配置不是
    ``TuningSpace.build`` 能直接表达的（它只做全量笛卡尔积）。这里不引入新的
    空间类型，只把筛好的完整配置列表包一层——配对测量本来就只依赖
    ``configs()``，不需要知道空间是怎么来的。
    """

    def __init__(self, configs: Sequence[Mapping[str, Any]]) -> None:
        self._configs = tuple(dict(item) for item in configs)

    def configs(self):
        return self._configs


def _project(
    configs: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    base: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """只让 ``keys`` 变化、其余键固定为 ``base``；保留首次出现顺序。"""
    seen = set()
    out: List[Dict[str, Any]] = []
    for cfg in configs:
        merged = dict(base)
        merged.update({name: cfg[name] for name in keys})
        fingerprint = tuple(sorted(merged.items(), key=lambda item: item[0]))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        out.append(merged)
    return out


def _stage_summary(keys: Sequence[str], result: PairedResult) -> Dict[str, Any]:
    """一阶段的结论摘要：赢家、判定、跨度与不稳定候选数，全部可 JSON 落盘。"""
    return {
        "keys": list(keys),
        "winner": dict(result.best.config),
        "decided": result.decided,
        "rounds": result.rounds,
        "warmup_rounds": result.warmup_rounds,
        "warmup_seconds": result.warmup_seconds,
        "max_round_seconds": result.max_round_seconds,
        "max_round_factor": result.max_round_factor,
        "round_budget_seconds": result.round_budget_seconds,
        "round_seconds": list(result.round_seconds),
        "long_round": sum(
            1 for item in result.candidates if item.verdict == LONG_ROUND
        ),
        "unstable": sum(
            1 for item in result.candidates if item.verdict == UNSTABLE
        ),
        "speedup": result.speedup,
    }


def tune_staged(
    space: TuningSpace,
    measure: Measure,
    *,
    first: Optional[Sequence[str]] = None,
    op: str = "",
    rounds: int = 8,
    noise_floor: float = 0.01,
    bootstrap: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    order: str = ORDER_ALTERNATE,
    warmup_rounds: int = 0,
    max_round_seconds: Optional[float] = None,
    max_round_factor: Optional[float] = None,
    min_same_sign: int = 0,
) -> PairedResult:
    """把一个大空间拆成两段坐标搜索，每段都用交替 A/B 单独判定。

    5.143 的实测说明：conv2d 的 45 个候选一轮要 2.2～2.5 秒，即使预热掉冷启动，
    跨度门槛也只会把它整段拦下——"这一轮太长"和"这个空间太大"其实是同一件事。
    两阶段搜索不改变测量方式，只改变**每个空间的大小**：

    1. 先只扫 ``first``（默认取候选值最少的那个键），其余参数固定为默认值；
       阶段 1 的结论只决定这一段坐标，不和其他键混在一起；
    2. 把 ``first`` 固定为阶段 1 的赢家，再扫剩下的键；阶段 2 的基线因此是
       "阶段 1 赢家 + 其余默认"，是坐标搜索的下一步，而不是全局默认；
    3. 若组合配置与全局默认不同，再单独做一次**两候选**对照（默认 vs 组合），
       回答"整条搜索到底有没有让默认变快"。返回的就是这次对照——``speedup``
       与 ``decided`` 的口径因此与 ``tune_paired`` 完全一致，配置库不需要区分。

    两阶段的 ``PairedResult`` 摘要记在 ``stages`` 里，报告能回答"是哪一段坐标
    贡献了收益、哪一段被噪声或跨度拦下"。若 ``first`` 已经覆盖全部参数，或组合
    配置等于默认，则省掉多余的那一次对照：前者直接返回阶段 1，后者直接返回
    阶段 2。

    这不是"全局最优搜索"：坐标搜索会漏掉"两个键必须同时改才更快"的交互。
    但现在的问题是"单轮太长度不了"，不是"解析最优"——先把能测准的那部分测准，
    比在 45 个候选里继续给出不可复现的赢家更有价值。
    """
    all_configs = [dict(item) for item in space.configs()]
    if not all_configs:
        raise ValueError("调优空间为空：至少需要一个配置")
    default = dict(all_configs[0])
    names = tuple(default)

    if first is None:
        first_keys = (
            min(names, key=lambda name: len({cfg[name] for cfg in all_configs})),
        )
    else:
        if isinstance(first, (str, bytes)):
            raise TypeError("first 必须是参数名序列，不能是单个字符串")
        first_keys = tuple(first)
        if not first_keys:
            raise ValueError("first 不能为空：至少要挑一个键先扫")
    unknown = [name for name in first_keys if name not in default]
    if unknown:
        raise ValueError(f"调优空间里没有参数 {unknown}，可选：{list(names)}")

    rest_keys = tuple(name for name in names if name not in first_keys)
    kwargs = dict(
        op=op, rounds=rounds, noise_floor=noise_floor, bootstrap=bootstrap,
        seed=seed, alpha=alpha, order=order, warmup_rounds=warmup_rounds,
        max_round_seconds=max_round_seconds,
        max_round_factor=max_round_factor,
        min_same_sign=min_same_sign,
    )

    stage1 = tune_paired(_FixedSpace(_project(all_configs, first_keys, default)), measure, **kwargs)
    if not rest_keys:
        return replace(stage1, stages=(_stage_summary(first_keys, stage1),))

    fixed = {name: stage1.best.config[name] for name in first_keys}
    stage2_configs = [
        dict(cfg)
        for cfg in all_configs
        if all(cfg[name] == fixed[name] for name in first_keys)
    ]
    stage2 = tune_paired(_FixedSpace(stage2_configs), measure, **kwargs)
    stages = (
        _stage_summary(first_keys, stage1),
        _stage_summary(rest_keys, stage2),
    )

    combined = dict(stage2.best.config)
    if combined == default:
        return replace(stage2, stages=stages)

    final = tune_paired(_FixedSpace([default, combined]), measure, **kwargs)
    return replace(final, stages=stages)


# --------------------------------------------------------------------------
# 跨采集复现
# --------------------------------------------------------------------------


def _replicate_summary(index: int, result: PairedResult) -> Dict[str, Any]:
    """一次采集的结论摘要；用于回答"哪一次和别的不一样"。"""
    return {
        "index": index,
        "best": dict(result.best.config),
        "decided": result.decided,
        "speedup": result.speedup,
        "unstable": sum(
            1 for item in result.candidates if item.verdict == UNSTABLE
        ),
        "long_round": sum(
            1 for item in result.candidates if item.verdict == LONG_ROUND
        ),
        "round_seconds": list(result.round_seconds),
    }


def _vote_counts(per: Sequence[PairedCandidate]) -> Tuple[Tuple[str, int], ...]:
    """逐次采集的判定票数，按次数降序、同票按词排序（便于稳定落盘与比对）。"""
    counts: Dict[str, int] = {}
    for candidate in per:
        counts[candidate.verdict] = counts.get(candidate.verdict, 0) + 1
    return tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


# 平票时的取舍：faster/slower 会替换默认，最"激进"；unreplicated 说明换个时间
# 就不成立，最保守。平票取保守的一侧——换配置的代价高于继续用默认。
_VERDICT_RANK = {
    FASTER: 0,
    SLOWER: 0,
    BASELINE: 1,
    NOISE: 1,
    UNSTABLE: 2,
    LONG_ROUND: 2,
    UNREPLICATED: 3,
    NO_BASELINE: 3,
    REJECTED: 3,
}


def _merge_verdicts(per: Sequence[PairedCandidate], vote: str) -> str:
    """把同一候选在各次采集里的判定合并成一个结论。

    两条硬约束对两种投票方式都成立：任一次不可行就记 ``unreplicated``（可行性
    不是投票能抵消的），``faster`` 与 ``slower`` 同时出现也记 ``unreplicated``
    （方向都定不下来，多数的意义不大）。

    ``all`` 要求每次判定完全相同，这是 5.146 的口径。``majority`` 允许少数次
    不同：超过半数（``len // 2 + 1``）同向即采纳；没有任何一方过半时，取出现
    次数最多的判定，平票按 :data:`_VERDICT_RANK` 取保守的一侧。
    """
    if not all(candidate.feasible for candidate in per):
        return UNREPLICATED
    verdicts = [candidate.verdict for candidate in per]
    if vote == VOTE_ALL:
        return verdicts[0] if len(set(verdicts)) == 1 else UNREPLICATED

    faster = verdicts.count(FASTER)
    slower = verdicts.count(SLOWER)
    if faster and slower:
        return UNREPLICATED
    need = len(verdicts) // 2 + 1
    if faster >= need:
        return FASTER
    if slower >= need:
        return SLOWER

    counts: Dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict] = counts.get(verdict, 0) + 1
    top = max(counts.values())
    tied = sorted(verdict for verdict, count in counts.items() if count == top)
    if len(tied) == 1:
        return tied[0]
    return max(tied, key=lambda verdict: _VERDICT_RANK.get(verdict, 2))


def tune_replicated(
    space: TuningSpace,
    measure: Measure,
    *,
    repeats: int = 2,
    op: str = "",
    rounds: int = 8,
    noise_floor: float = 0.01,
    bootstrap: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    order: str = ORDER_ALTERNATE,
    warmup_rounds: int = 0,
    max_round_seconds: Optional[float] = None,
    max_round_factor: Optional[float] = None,
    vote: str = VOTE_ALL,
    min_same_sign: int = 0,
) -> PairedResult:
    """把同一空间独立采集 ``repeats`` 次，再按 ``vote`` 合并各次判定。

    5.145 的 16 轮实验说明：整机漂移一旦到 25%，加轮数只能减少假阳性，不能把
    漂移变成配置差异。这里换一个判据——**跨采集复现**：同一个候选必须在每一次
    独立采集里都判 ``faster``（且都超过噪声地板），才允许替换默认。

    每次重复用 ``seed + i``：bootstrap 与 shuffle 的随机流不同，但样本仍来自同一
    台机器的连续采集——所以它检验的是"换一段时间后结论还在不在"，不是严格的
    独立实验设计。代价是采集时间乘以 ``repeats``。

    某候选在两次里一次可行、一次不可行，或一次 ``faster``、一次 ``noise``，都记
    ``unreplicated``：这不是噪声太小，而是"换个时间就不成立"。基线本身不参与
    复现检查（它永远是基线）。返回结果的 ``round_seconds`` / ``order`` 来自最后
    一次采集，逐次摘要见 ``replicates``。

    ``vote`` 决定怎么合并：``all``（默认）要求每次判定完全相同，5.146 的口径；
    ``majority`` 允许少数次不同、超过半数同向即采纳（需要 ``repeats >= 3``），
    用来处理"重复次数一多，全票就必然过严"的问题。两者的硬约束相同：任一次
    不可行、或 ``faster`` 与 ``slower`` 互现，都记 ``unreplicated``。

    每个候选还记录 ``replicate_diffs``：逐次采集的配对差中位数，顺序与
    ``replicates`` 对齐，那一次不可行时记 ``None``。票数说"判了几次更快"，
    逐次差说"每次差多少"，两者合起来才能区分"幅度不足"与"方向反转"。
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int):
        raise TypeError(f"repeats 必须是整数，得到 {type(repeats).__name__}")
    if repeats < 2:
        raise ValueError(f"跨采集复现至少需要 2 次独立采集，得到 {repeats}")
    if vote not in _VOTES:
        raise ValueError(f"vote 必须是 {_VOTES} 之一，得到 {vote!r}")
    if vote == VOTE_MAJORITY and repeats < 3:
        raise ValueError(
            f"多数票需要 repeats >= 3，得到 {repeats}：两次采集下'多数'就是全票"
            "（2 > 1），与 all 没有区别，只会让参数看起来换了判据"
        )

    kwargs = dict(
        op=op, rounds=rounds, noise_floor=noise_floor, bootstrap=bootstrap,
        alpha=alpha, order=order, warmup_rounds=warmup_rounds,
        max_round_seconds=max_round_seconds,
        max_round_factor=max_round_factor,
        min_same_sign=min_same_sign,
    )
    results = [
        tune_paired(space, measure, seed=int(seed) + index, **kwargs)
        for index in range(repeats)
    ]

    base = results[-1]
    merged: List[PairedCandidate] = []
    for idx, item in enumerate(base.candidates):
        per = [result.candidates[idx] for result in results]
        merged.append(
            replace(
                item,
                verdict=_merge_verdicts(per, vote),
                votes=_vote_counts(per),
                # 逐次配对差与票数一一对应：看到 2/3 票型时，这里能回答
                # "那张 noise 是幅度不足（差小）还是方向反转（差为正）"
                replicate_diffs=tuple(
                    candidate.diff_median_pct for candidate in per
                ),
            )
        )

    baseline = merged[0]
    if baseline.feasible:
        winners = [item for item in merged[1:] if item.verdict == FASTER]
        winner = (
            min(winners, key=lambda item: item.diff_median_pct)
            if winners else baseline
        )
    else:
        feasible = [item for item in merged if item.feasible]
        if not feasible:
            raise AllRejectedError(op or "<unknown>", tuple(merged))
        winner = min(feasible, key=lambda item: item.median_ms)

    return replace(
        base,
        best=winner,
        candidates=tuple(merged),
        repeats=repeats,
        vote=vote,
        replicates=tuple(
            _replicate_summary(index, result) for index, result in enumerate(results)
        ),
    )
