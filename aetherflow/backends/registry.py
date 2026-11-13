"""算子注册与调度。

后端关于"我承接哪些算子"的信息以前有两份：``capabilities()`` 返回的集合，
和方法名反射（``getattr(self, f"_op_{name}")``）。两份信息一旦分叉，失败是安静的：

- 声明里有、实现里没有 → 要等第一次推理才炸；
- 实现里有、声明里没有 → 前端明明可以用它，却绕开去做拆分或降级，白丢性能，
  而且没有任何报错说明这件事发生过。

``KernelRegistry`` 让这份信息只剩一份：**注册即声明，查表即调度**。注册时做三项
校验——算子名必须存在于 IR 算子表（防拼错）、不得重复注册（防静默覆盖）、
必须可调用。

它不负责图拆分、降级或融合。那些是前端与 IR 的决定；注册表只回答一个问题：
**这个算子交给谁来算**。

**多实现按代价选择**：同一个算子可以登记多个候选，每个候选带一个 ``cost`` 函数，
调度时取代价最小者。规则只有一条，是为了让"选择"这件事可解释：

- 单实现不写 cost，调度零开销，行为与从前完全一致；
- 一旦出现第二个候选，**全部**候选都必须带 cost。否则排名里会混进一个无法比较
  的项，实际胜出的必然是注册顺序——那是隐式优先级，不是按代价选择。

``cost`` 返回 ``None`` 或非有限值，表示"这个候选不适用于当前形状"。后端可以用它
表达能力边界（例如"softmax 只吃最后一维"、"分块内核要求 head_dim <= 64"），
而不必先在内核里抛错；全部候选都不适用时抛 ``BackendCapabilityError``，
与"不支持"走同一条契约。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Iterator, List, Mapping, Optional, Tuple

from ..ir.ops import OPS
from .base import BackendCapabilityError

__all__ = ["KernelRegistry", "Kernel", "CostFn", "Variant"]

# 内核签名统一为 (node, args, attrs) -> 后端容器值；这里只用来标注，不做运行期检查
Kernel = Callable[..., object]
# 代价函数与内核同参，返回估算代价；None / inf / nan 表示该候选不适用
CostFn = Callable[..., Optional[float]]


def _context_suffix(context: object) -> str:
    """把 ``context`` 变成错误信息里的定位后缀。

    只在失败路径上调用：热路径每节点都要过调度，提前拼字符串等于给每次
    成功的派发摊上一笔"可能报错"的税。节点对象在这里才被展开成节点名。
    """
    if not context:
        return ""
    if isinstance(context, str):
        return f"（{context}）"
    name = getattr(getattr(context, "output", None), "name", None)
    return f"（节点 %{name}）" if name else ""


@dataclass(frozen=True)
class Variant:
    """一个候选实现。

    ``cost`` 为 ``None`` 表示"没有代价模型"，只允许它是该算子的唯一候选；
    ``label`` 用于报错与自省（"attention 的两个候选：blocked / fallback"），
    同一算子下必须唯一。
    """

    kernel: Kernel
    cost: Optional[CostFn] = None
    label: str = ""

    @property
    def ranked(self) -> bool:
        """是否参与按代价排名。"""
        return self.cost is not None


class KernelRegistry:
    """``算子名 -> 候选实现列表`` 的登记表。

    ``name`` 只用于错误信息（"cuda 未注册算子 conv2d"），不参与查找。
    """

    def __init__(self, name: str = "backend") -> None:
        self.name = name
        self._kernels: Dict[str, List[Variant]] = {}

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------

    def register(
        self,
        op_name: str,
        kernel: Kernel | None = None,
        *,
        cost: CostFn | None = None,
        label: str | None = None,
    ):
        """登记一个内核；也可以省略内核，当装饰器用。

        校验的失败方式刻意不同：
        ``KeyError`` 表示算子名不在 IR 里（拼写错误），
        ``ValueError`` 表示重复注册或候选组合不合法（会静默覆盖/产生隐式优先级），
        ``TypeError`` 表示传了不可调用或不可求值的东西。
        """
        if kernel is None:
            def decorate(fn: Kernel) -> Kernel:
                self.register(op_name, fn, cost=cost, label=label)
                return fn

            return decorate

        if op_name not in OPS:
            raise KeyError(f"未知算子 {op_name!r}：它不在 IR 算子表里，注册了也不会被图用到")
        if not callable(kernel):
            raise TypeError(f"算子 {op_name} 的内核不可调用: {type(kernel).__name__}")
        if cost is not None and not callable(cost):
            raise TypeError(f"算子 {op_name} 的 cost 不可调用: {type(cost).__name__}")

        variant_label = label or getattr(kernel, "__name__", type(kernel).__name__)
        existing = self._kernels.get(op_name)
        if existing is None:
            self._kernels[op_name] = [Variant(kernel, cost, variant_label)]
            return kernel

        if cost is None:
            raise ValueError(
                f"算子 {op_name} 已注册（{existing[0].label}），不允许静默覆盖；"
                "要登记实现同一算子的另一个内核，请给候选指定不同的 cost"
            )
        unranked = [v.label for v in existing if v.cost is None]
        if unranked:
            raise ValueError(
                f"算子 {op_name} 已有不带 cost 的实现 {unranked}，不能再登记带 cost 的候选："
                "多个候选必须都能比较代价，否则胜出的是注册顺序"
            )
        if any(v.label == variant_label for v in existing):
            raise ValueError(
                f"算子 {op_name} 已存在名为 {variant_label!r} 的候选，label 必须唯一"
            )
        existing.append(Variant(kernel, cost, variant_label))
        return kernel

    def register_all(self, mapping: Mapping[str, Kernel]) -> None:
        """批量注册。部分失败时已成功的条目保留——重试只需补齐剩下的。"""
        for op_name, kernel in mapping.items():
            self.register(op_name, kernel)

    # ------------------------------------------------------------------
    # 查询与调度
    # ------------------------------------------------------------------

    def get(self, op_name: str) -> Kernel | None:
        """取内核；未注册返回 ``None``（供调用方自己决定怎么报错）。

        多实现时返回**第一个候选**，只用于自省与绑定检查，不代表调度结果；
        需要知道真实候选集合请用 ``variants()``。
        """
        variants = self._kernels.get(op_name)
        return variants[0].kernel if variants else None

    def variants(self, op_name: str) -> Tuple[Variant, ...]:
        """某个算子的全部候选，按注册顺序；未注册返回空元组。"""
        return tuple(self._kernels.get(op_name, ()))

    def capabilities(self) -> FrozenSet[str]:
        """能力声明直接由注册表派生，不可能与实现分叉。"""
        return frozenset(self._kernels)

    def dispatch(self, op_name: str, *args, context: object = "") -> object:
        """查找并调用内核。

        ``context`` 是给错误信息用的位置说明（例如节点名），不影响调度本身。
        它可以是字符串，也可以是节点对象——后者只在真正报错时才被格式化成
        文字。热路径上每个节点都要过一遍调度，所以不能为"可能不会发生"的
        报错先拼两个 f-string。
        未注册时抛 ``BackendCapabilityError``——与"不支持"的契约一致，
        不让调用方去分辨这是 NotImplementedError 还是别的什么。
        多实现时按 ``cost`` 选最小者；全部候选都不适用时同样抛
        ``BackendCapabilityError``，因为对调用方而言"当前形状下没有内核"
        与"没有这个算子"是同一件事。
        """
        variants = self._kernels.get(op_name)
        if not variants:
            raise BackendCapabilityError(
                f"{self.name} 未注册算子 {op_name}{_context_suffix(context)}"
            )

        if len(variants) == 1:
            return variants[0].kernel(*args)

        best: Optional[Variant] = None
        best_cost = math.inf
        unavailable: List[str] = []
        for variant in variants:
            value = variant.cost(*args)
            if value is None or not math.isfinite(float(value)):
                unavailable.append(variant.label)
                continue
            value = float(value)
            if value < best_cost:
                best, best_cost = variant, value

        if best is None:
            raise BackendCapabilityError(
                f"{self.name} 的算子 {op_name} 在当前形状下没有可用内核"
                f"{_context_suffix(context)}：候选 {unavailable} 均不适用"
            )
        return best.kernel(*args)

    # ------------------------------------------------------------------

    def __contains__(self, op_name: object) -> bool:
        return op_name in self._kernels

    def __len__(self) -> int:
        """算子个数（不是候选个数）。"""
        return len(self._kernels)

    def __iter__(self) -> Iterator[str]:
        return iter(self._kernels)
