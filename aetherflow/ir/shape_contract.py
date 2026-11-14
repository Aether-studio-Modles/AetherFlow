"""动态形状契约：符号维在每一层的支持面。

IR 把符号维（裸符号与整数表达式）当作一等形状，但下游各层的支持面并不相同：
执行器能按输入数组的实际形状绑定符号维、同一张图跑不同 batch；反向建图则要看
具体算子——``attention`` 的反向可以带符号批维，``reduce_sum`` 的广播与
``reshape`` 的目标形状却必须在构图期拿到具体数字。没有一张统一的契约表，
用户撞上的是"某个算子内部的形状错误"，看不出"这一层不支持动态形状"。

所以每个环节给出一个明确状态，绝不用"碰巧能跑"来暗示支持：

- ``full``：符号维可用，已有用例覆盖；
- ``per-op``：支持面由算子决定，不支持的算子在构图期显式拒绝；
- ``data-only``：数据输入可以是符号，参数 / 权重必须静态；
- ``rejected``：入口统一拒绝，错误里带环节名与可操作建议。

``require_static_shapes`` 是 ``rejected`` 环节的统一守卫。它只在图输入上判定：
符号维只能从图输入进入，中间值不可能凭空多出一个符号。
"""

from __future__ import annotations

from types import MappingProxyType
from typing import List, Mapping, NamedTuple

from .graph import Graph

__all__ = [
    "StageSupport",
    "DYNAMIC_SHAPE_SUPPORT",
    "support_for",
    "symbolic_inputs",
    "require_static_shapes",
]

FULL = "full"
PER_OP = "per-op"
DATA_ONLY = "data-only"
REJECTED = "rejected"
STATUSES = frozenset((FULL, PER_OP, DATA_ONLY, REJECTED))


class StageSupport(NamedTuple):
    """一个环节对符号维的支持状态。``reason`` 必须能独立看懂。"""

    stage: str
    label: str
    dynamic: str
    reason: str


DYNAMIC_SHAPE_SUPPORT: Mapping[str, StageSupport] = MappingProxyType(
    {
        item.stage: item
        for item in (
            StageSupport(
                "graph",
                "构图",
                FULL,
                "TensorType 的维可以是裸符号或整数表达式，构图期不做具体化",
            ),
            StageSupport(
                "execute",
                "执行",
                FULL,
                "执行器按输入数组的实际形状绑定符号维，同一张图可跑不同 batch",
            ),
            StageSupport(
                "autodiff",
                "自动微分",
                PER_OP,
                "反向按算子逐个判定：attention、elementwise 等可带符号批维；"
                "reduce_sum 的广播与 reshape 的目标形状需要具体数字，在构图期拒绝",
            ),
            StageSupport(
                "trainer",
                "训练循环",
                REJECTED,
                "训练目标必经一次归约，反向的广播需要具体形状；Trainer 在构造期拒绝",
            ),
            StageSupport(
                "recompute",
                "分段重算",
                REJECTED,
                "分段重算依赖 backward，任一段的 VJP 都可能在符号维上拒绝；"
                "入口统一拒绝，避免训练中途才失败",
            ),
            StageSupport(
                "deploy",
                "部署",
                DATA_ONLY,
                "部署包里的权重必须是静态形状；数据输入仍由执行器按实际形状绑定",
            ),
            StageSupport(
                "torch-import",
                "torch 导入",
                REJECTED,
                "torch 的动态维（SymInt）在导入期落不成静态 IR 形状；"
                "请先用 mark_static 固定该维",
            ),
        )
    }
)


def support_for(stage: str) -> StageSupport:
    """查一个环节的支持状态；未知环节直接报错，不做默认放行。"""
    try:
        return DYNAMIC_SHAPE_SUPPORT[stage]
    except KeyError:
        raise ValueError(
            f"未知的动态形状环节 {stage!r}；"
            f"已声明的环节: {sorted(DYNAMIC_SHAPE_SUPPORT)}"
        ) from None


def symbolic_inputs(graph: Graph) -> List[str]:
    """图里含符号维的输入名（按图的输入顺序）。"""
    return [value.name for value in graph.inputs if not value.type.is_static]


def require_static_shapes(graph: Graph, *, stage: str, hint: str = "") -> None:
    """要求整张图的输入都是静态形状；否则抛出带环节名的 ``ValueError``。

    ``stage`` 必须是契约表里声明为 ``rejected`` 的环节：把 ``full`` 或
    ``per-op`` 环节传进来会让本来可用的图被误拦，所以这里先查表再判定。
    """
    support = support_for(stage)
    if support.dynamic != REJECTED:
        raise ValueError(
            f"环节 {stage!r} 的动态形状状态是 {support.dynamic!r}，"
            "不应调用 require_static_shapes；只有 rejected 环节需要这道守卫"
        )
    symbolic = [value for value in graph.inputs if not value.type.is_static]
    if not symbolic:
        return
    listed = "、".join(f"%{value.name}（{value.type}）" for value in symbolic)
    message = (
        f"{support.label}环节必须用静态形状，这些输入含符号维: {listed}。{support.reason}"
    )
    if hint:
        message = f"{message}。{hint}"
    raise ValueError(message)
