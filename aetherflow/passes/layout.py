"""通道在后（NHWC）布局传播。

**为什么需要它**：卷积 / 池化的自研内核在 GPU 上按 NCHW 寻址，而通道在后时
同一个输出通道的元素在内存里连续，向量化与后续逐元素算子（relu、gelu…）
的访存都能利用这一点——逐元素算子对通道轴本来就是逐元素，布局换了也照样
正确。真实框架因此会在图里"选一段连续的可翻转区域，整段换成 NHWC，只在区域
边界付两次转置"。

**这里做的不是复制数学**：``conv2d`` / ``maxpool2d`` / ``avgpool2d`` 的
``layout="nhwc"`` 由 ``ir/ops.py`` 用同一份 NCHW 参考实现定义（先转 NCHW 算，
再把输出转回），因此翻转后的图与翻转前**逐位相同**。传播 pass 只负责改属性
与插入边界转置，数值差异不可能从这里冒出来——这也是它能被差分测试严格验证
的前提。

**边界**（与 DESIGN 5.246 一致）：

- 可翻转算子只包含 conv / maxpool / avgpool 与形状保持的一元逐元素算子。
  reshape、切片、归约、矩阵乘都不翻：前两者改变轴的含义或直接依赖维度顺序，
  后两者的分块策略与布局耦合，翻过去只会把问题推给后端。
- 一个连通区域里至少有一个 conv / pool 才值得翻，否则纯逐元素区域翻过去只是
  白付两次转置。
- 区域边界插 ``transpose``：入口 ``(0,2,3,1)``（NCHW→NHWC），出口
  ``(0,3,1,2)``（NHWC→NCHW）。区域内部的值在 NHWC 域里流动，区域之间的
  语义（形状与 dtype）与原来完全一致。
- **目标后端必须显式声明承接 NHWC**。声明里没有就抛 :class:`LayoutError`，
  不做"悄悄跳过"或"翻完再让执行期报错"——前者让优化看起来生效了，后者把
  失败推到几百个节点之后。
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from ..ir.graph import Graph, Node, Value
from .base import GraphRewriter, Pass

__all__ = ["LayoutError", "LayoutPropagation"]


class LayoutError(ValueError):
    """目标后端承接不了 NHWC 形态。"""


#: NCHW -> NHWC；逆变换是它的逆置换。
_NHWC_PERM: Tuple[int, ...] = (0, 2, 3, 1)
_NCHW_PERM: Tuple[int, ...] = (0, 3, 1, 2)

#: 真正"接收 4 维激活"的算子。它们在 NHWC 域里仍然接收 4 维激活，
#: 只是维度顺序变了——权重与 bias 不变。
_SPATIAL_OPS: FrozenSet[str] = frozenset({"conv2d", "maxpool2d", "avgpool2d"})

#: 形状保持的一元逐元素算子。它们对通道轴本来就是逐元素，翻转不影响语义。
#: 只列语义确定、输出类型等于输入类型的那些；``silu`` 只有 1 输入形态算，
#: 带门控的融合形态（2 输入）另算一个节点语义，不在这里吸收。
_ELEMENTWISE_OPS: FrozenSet[str] = frozenset(
    {
        "relu",
        "sigmoid",
        "tanh",
        "gelu",
        "silu",
        "exp",
        "log",
        "sqrt",
        "neg",
        "erf",
        "step",
        "spike",
        "sign",
        "round",
        "floor",
        "ceil",
    }
)


class LayoutPropagation(Pass):
    """把卷积 / 池化所在的连通区域整段翻成 NHWC。

    默认 ``supported_layouts`` 取参考后端的口径（``{"nchw", "nhwc"}``）。
    **设备后端必须显式传入自己的声明**，通常是 ``DeviceBackend.supported_layouts``：
    它只承接 NCHW，传进来后这个 pass 会在改图之前就拒绝，而不是翻完再让
    执行期在某个节点上失败。
    """

    name = "layout-propagation"

    def __init__(self, supported_layouts: Optional[Iterable[str]] = None) -> None:
        self.supported_layouts: FrozenSet[str] = (
            frozenset({"nchw", "nhwc"})
            if supported_layouts is None
            else frozenset(supported_layouts)
        )

    # ------------------------------------------------------------------
    # 判据
    # ------------------------------------------------------------------

    @staticmethod
    def _flippable(node: Node) -> bool:
        """这个节点能否在 NHWC 域里保持自己的语义。"""
        name = node.op.name
        if name in _SPATIAL_OPS:
            return node.inputs[0].type.rank == 4
        if name not in _ELEMENTWISE_OPS:
            return False
        if len(node.inputs) != 1:
            return False
        # 形状与 dtype 都必须原样保留；只要有一项变了，这个节点就不在
        # "逐元素搬运"的范围内（例如 later 加了带参数的激活）。
        return node.output.type == node.inputs[0].type and node.output.type.rank == 4

    @staticmethod
    def _activation_inputs(node: Node) -> Tuple[Value, ...]:
        """参与"区域连通"的输入。

        只有激活参与：卷积的 weight / bias 是参数而不是激活，若把它们也
        视为传播边，权重就会被当成 NHWC 张量去转置（语义直接错）。
        """
        return node.inputs[:1] if node.op.name in _SPATIAL_OPS else node.inputs

    # ------------------------------------------------------------------
    # 区域收集
    # ------------------------------------------------------------------

    def _collect_groups(self, graph: Graph) -> List[Set[int]]:
        """按拓扑求"可翻转节点的极大连通区域"，只保留含 conv / pool 的。

        返回值是节点集合的列表，元素是 ``node.output.id``。区域之间不连通，
        因此每个区域自己付两次边界转置。
        """
        candidates: Dict[int, Node] = {
            node.output.id: node for node in graph.nodes if self._flippable(node)
        }
        if not candidates:
            return []

        # 卷积的权重 / bias 若来自候选节点，说明那个候选的输出被当成了参数。
        # 此时不能翻这个卷积（权重会被误当激活转置），把它连同"以它为参数"的
        # 卷积一起剔除，直到稳定。剔除只会缩小候选集，因此循环必然终止。
        spatial_ids = {
            oid for oid, node in candidates.items() if node.op.name in _SPATIAL_OPS
        }
        changed = True
        while changed:
            changed = False
            for oid in list(spatial_ids):
                node = candidates.get(oid)
                if node is None:
                    continue
                if any(a.id in candidates for a in node.inputs[1:]):
                    del candidates[oid]
                    spatial_ids.discard(oid)
                    changed = True

        parent: Dict[int, int] = {oid: oid for oid in candidates}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for oid, node in candidates.items():
            for a in self._activation_inputs(node):
                if a.id in candidates:
                    union(oid, a.id)

        groups: Dict[int, Set[int]] = {}
        for oid in candidates:
            groups.setdefault(find(oid), set()).add(oid)

        return [
            group
            for group in groups.values()
            if any(candidates[oid].op.name in _SPATIAL_OPS for oid in group)
        ]

    # ------------------------------------------------------------------
    # 重写
    # ------------------------------------------------------------------

    def run(self, graph: Graph) -> Graph:
        groups = self._collect_groups(graph)
        if not groups:
            return graph

        if "nhwc" not in self.supported_layouts:
            raise LayoutError(
                "目标后端未声明承接 nhwc 布局"
                f"（声明为 {sorted(self.supported_layouts)}），"
                f"但图里有 {len(groups)} 段卷积 / 池化区域可以翻成 NHWC。"
                "要么换一个声明了 nhwc 的后端（参考后端就是），"
                "要么对这个后端关闭布局传播——不做静默降级"
            )

        flipped: Set[int] = set().union(*groups)
        output_ids = {v.id for v in graph.outputs}
        consumers: Dict[int, Set[int]] = {}
        for node in graph.nodes:
            for a in node.inputs:
                consumers.setdefault(a.id, set()).add(node.output.id)

        rw = GraphRewriter(graph)
        rw.copy_inputs()
        nhwc_value: Dict[int, Value] = {}
        entry_cache: Dict[int, Value] = {}

        def to_nhwc(v: Value) -> Value:
            """区域外的 NCHW 值 -> 区域内的 NHWC 值（同一来源只转一次）。"""
            cached = entry_cache.get(v.id)
            if cached is not None:
                return cached
            nv = rw.new.add_node("transpose", [rw.map[v.id]], {"perm": list(_NHWC_PERM)})
            entry_cache[v.id] = nv
            return nv

        for node in graph.nodes:
            oid = node.output.id
            if oid not in flipped:
                rw.emit(node)
                continue

            # 只有激活（卷积的第 0 个输入）需要进 NHWC 域；weight / bias 是
            # 参数，维度顺序不随激活布局变化，直接沿用原值。
            args = []
            for i, a in enumerate(node.inputs):
                if a.id in flipped:
                    args.append(nhwc_value[a.id])
                elif i == 0 or node.op.name not in _SPATIAL_OPS:
                    args.append(to_nhwc(a))
                else:
                    args.append(rw.map[a.id])
            attrs = dict(node.attrs)
            if node.op.name in _SPATIAL_OPS:
                # 只有卷积 / 池化带 layout 属性；逐元素算子的语义与维度顺序
                # 无关，不给它们加算子表之外的属性。
                attrs["layout"] = "nhwc"
            nv = rw.new.add_node(node.op.name, args, attrs)
            nhwc_value[oid] = nv

            # 区域外只要有消费者（或它就是图输出），就必须转回 NCHW；
            # 只在区域内部流动的值不转，这正是"整段翻转"省下的开销。
            leaves = oid in output_ids or any(
                c not in flipped for c in consumers.get(oid, ())
            )
            if leaves:
                rw.map[oid] = rw.new.add_node(
                    "transpose", [nv], {"perm": list(_NCHW_PERM)}
                )
            else:
                rw.map[oid] = nv

        return rw.finish()
