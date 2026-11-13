"""网格搜索：候选配置逐个测，取最快的那个。

搜索刻意做成**最笨的网格**，理由有三条：

1. 候选空间很小（CUDA 侧目前只有 block 一维、四个取值）。贝叶斯或进化搜索
   在四个点上省不下时间，只会引入"这次为什么选它"的不可解释性。
2. 失败必须显式。某个候选在驱动层直接报错（shared memory 不够、block 越界），
   这不是"慢"，而是**不可行**；把它当成无穷大混进排名，会让"没有可行配置"
   退化成"有一个很慢的配置"。
3. 结果必须可复现。平局取先出现者、候选顺序由空间决定，因此同一份测量输入
   永远产出同一条配置。

**排名用中位数，报告同时给最小值。** 多轮测量里"取最小"回答的是"这块卡
能做到多快"，它天然偏向恰好撞上空闲窗口的那一轮；用它排名并叫它
``median_ms``，等于把择优偏差当成典型延迟。这里把两者分开：
``median_ms`` 是样本中位数（典型延迟，排名依据），``best_ms`` 是最小值
（最佳可达，仅供报告），``p95_ms`` / ``spread`` 说明离散程度。

**差距不显著时保留默认。** 两个候选只差 1%，在噪声范围内，把默认换掉
只是把"这次谁赢了"写成"谁更好"。``tie_epsilon`` 以下的差距一律不换，
让默认配置在证据不足时胜出。

``measure`` 由调用方注入（通常是"应用该配置后执行一次并取中位数"），
这一层不碰后端、不碰 GPU，因此可以用一个纯函数完整测试。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .space import TuningSpace

__all__ = [
    "CandidateResult",
    "TuningResult",
    "AllRejectedError",
    "tune",
]

Measure = Callable[[Mapping[str, Any]], Optional[float]]


def _percentile(ordered: List[float], q: float) -> float:
    """最近秩法取分位：不插值，返回样本里真实存在的一个值。

    样本只有个位数时，线性插值会造出一个"比最慢那轮还慢"或"比最快那轮
    还快"的数；这里宁可粗糙也不虚构没测到的值。``ordered`` 必须非空。
    """
    rank = max(1, math.ceil(q * len(ordered)))
    return float(ordered[min(rank, len(ordered)) - 1])


@dataclass(frozen=True)
class CandidateResult:
    """一个候选配置的测量结果。``median_ms`` 为 ``None`` 表示被拒绝。

    ``median_ms`` 是多轮样本的中位数，也是排名依据；``best_ms`` 是最小值，
    只说明"最佳可达"，不参与排名——两者混用时，一轮抖动就能让一个整体
    更慢的配置凭单次最优胜出。
    """

    config: Mapping[str, Any]
    median_ms: Optional[float] = None
    best_ms: Optional[float] = None
    p95_ms: Optional[float] = None
    samples: int = 0
    reason: str = ""

    @property
    def feasible(self) -> bool:
        return self.median_ms is not None

    @property
    def spread(self) -> Optional[float]:
        """``(p95 - p50) / p50``：相对默认中位数的离散程度；中位数为 0 时为 ``None``。"""
        if self.median_ms is None or not self.p95_ms or not self.median_ms:
            return None
        return round((self.p95_ms - self.median_ms) / self.median_ms, 4)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"config": dict(self.config), "samples": self.samples}
        if self.feasible:
            out["median_ms"] = self.median_ms
            out["best_ms"] = self.best_ms
            out["p95_ms"] = self.p95_ms
            out["spread"] = self.spread
        else:
            out["rejected"] = self.reason
        return out


@dataclass(frozen=True)
class TuningResult:
    """一次搜索的完整结果：赢家 + 全部候选 + 拒绝原因。

    保留全部候选而不只是赢家，是为了让"赢了多少"可核对。只报最优值时，
    一个慢 2% 的平局和一个慢 3 倍的错误选择看起来一模一样。
    """

    best: CandidateResult
    candidates: Tuple[CandidateResult, ...] = field(default_factory=tuple)
    # 判定"显著更快"所需的相对差距；差距小于它时保留先出现的候选
    tie_epsilon: float = 0.0

    @property
    def rejected(self) -> Tuple[CandidateResult, ...]:
        return tuple(item for item in self.candidates if not item.feasible)

    @property
    def speedup(self) -> Optional[float]:
        """相对**默认配置**（空间里的第一个候选）的典型延迟加速比；默认被拒时为 None。"""
        if not self.candidates or not self.candidates[0].feasible:
            return None
        default = self.candidates[0].median_ms
        if not default:
            return None
        return round(default / self.best.median_ms, 4)  # type: ignore[operator]

    @property
    def best_speedup(self) -> Optional[float]:
        """按"最佳可达"（多轮最小值）算的加速比；仅作参考，不作排名依据。"""
        if not self.candidates or not self.candidates[0].feasible:
            return None
        default = self.candidates[0].best_ms
        best = self.best.best_ms
        if not default or not best:
            return None
        return round(default / best, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best": self.best.to_dict(),
            "candidates": [item.to_dict() for item in self.candidates],
            "speedup": self.speedup,
            "best_speedup": self.best_speedup,
            "tie_epsilon": self.tie_epsilon,
        }


class AllRejectedError(RuntimeError):
    """全部候选都不可行。

    这不是"没找到更快的"，而是"这个空间在当前后端/形状下没有一个能跑"。
    两者必须分开：前者回落到默认配置是安全的，后者说明默认配置本身也有问题，
    继续执行只会在更远的地方炸。
    """

    def __init__(self, op: str, candidates: Tuple[CandidateResult, ...]) -> None:
        self.op = op
        self.candidates = candidates
        detail = "; ".join(
            f"{dict(item.config)} -> {item.reason}" for item in candidates
        )
        super().__init__(f"{op} 的全部 {len(candidates)} 个候选都不可行：{detail}")


def _valid(value: Any) -> Optional[float]:
    """把测量结果收敛成一个可比较的数；无效值返回 ``None``。"""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def tune(
    space: TuningSpace,
    measure: Measure,
    *,
    op: str = "",
    rounds: int = 1,
    tie_epsilon: float = 0.02,
) -> TuningResult:
    """穷举 ``space`` 的配置，返回**典型延迟**最短者。

    ``rounds > 1`` 时每个配置测多次：中位数作为排名依据，最小值与 p95
    一并写进结果，分别回答"通常多快"和"最好能多快、抖动有多大"。旧实现
    多轮取最小值却把字段叫 ``median_ms``，报出来的是"最佳可达"，不是典型
    延迟；现在两者分开，排名不再被单轮抖动带偏。

    ``tie_epsilon`` 是"显著更快"的判据：候选的中位数必须低于当前赢家
    ``tie_epsilon``（默认 2%）才算超越。差距落在噪声范围内时保留先出现的
    候选，默认配置因此不会被一次轻微的偶然领先替换掉。

    ``measure`` 抛异常、返回 ``None`` / NaN / 负数，都记为**不可行**并继续；
    只有全部候选都不可行时才抛 ``AllRejectedError``。
    """
    if not callable(measure):
        raise TypeError(f"measure 必须可调用，得到 {type(measure).__name__}")
    if isinstance(tie_epsilon, bool) or not isinstance(tie_epsilon, (int, float)):
        raise TypeError(f"tie_epsilon 必须是数，得到 {type(tie_epsilon).__name__}")
    if not 0.0 <= float(tie_epsilon) < 1.0:
        raise ValueError(f"tie_epsilon 必须在 [0, 1) 内，得到 {tie_epsilon!r}")
    rounds = max(1, int(rounds))
    tie_epsilon = float(tie_epsilon)

    results = []
    for config in space.configs():
        reason = ""
        values: List[float] = []
        try:
            for _ in range(rounds):
                value = _valid(measure(config))
                if value is None:
                    reason = "测量结果不是有限的非负数"
                    break
                values.append(value)
        except Exception as exc:  # noqa: BLE001 - 任何异常都意味着该配置不可行
            reason = f"{type(exc).__name__}: {exc}"
            values = []

        if reason or not values:
            results.append(
                CandidateResult(
                    config=dict(config),
                    samples=len(values),
                    reason=reason or "没有有效测量",
                )
            )
            continue

        ordered = sorted(values)
        results.append(
            CandidateResult(
                config=dict(config),
                median_ms=round(_percentile(ordered, 0.5), 6),
                best_ms=round(ordered[0], 6),
                p95_ms=round(_percentile(ordered, 0.95), 6),
                samples=len(values),
            )
        )

    candidates = tuple(results)
    feasible = [item for item in candidates if item.feasible]
    if not feasible:
        raise AllRejectedError(op or "<unknown>", candidates)

    winner = feasible[0]
    for item in feasible[1:]:
        # 只有差距超过 tie_epsilon 才替换：平局与噪声内的领先都保留先出现者，
        # 结果因此是确定的，也不会把"这次谁赢了"写成"谁更好"。
        if item.median_ms < winner.median_ms * (1.0 - tie_epsilon):  # type: ignore[operator]
            winner = item
    return TuningResult(best=winner, candidates=candidates, tie_epsilon=tie_epsilon)
