"""公共子表达式消除。

签名 = (算子名, 输入的新值 id, 属性的冻结表示)。
签名相同的节点只保留一次，后续出现直接复用第一个结果。

这是图级优化里"库做不到"的典型例子：cuDNN 只负责单个卷积，
它看不见"图上两条完全相同的子计算"。

**只合并纯算子。** 声明了 ``effects`` 的节点（RNG / STATE / IO）即使签名
相同也不能并成一个：两次采样、两次状态写入不是同一次计算，合并会改变语义。

``CommutativeCSE`` 是同一套消除的可选变体：``add`` / ``mul`` 的签名按输入 id
排序后再比较，因此 ``add(a, b)`` 与 ``add(b, a)`` 视为同一次计算。它不进默认
管线——反向图才是它的主场景（cotangent 的累加顺序由遍历顺序决定，同一个值的
两份贡献很容易以互换形态出现）。
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

from ..ir.graph import Graph, Node
from .base import GraphRewriter, Pass

# 满足交换律的算子：签名可以按输入排序，合并后数值逐位不变。
# 只列两个：浮点加法与乘法在 IEEE 754 下对有限值与 inf 逐位可交换；其他算子
# 要么不满足交换律（sub / div），要么语义未定义到可以互换（比较类）。
COMMUTATIVE_OPS = ("add", "mul")


def _freeze_attrs(attrs: dict) -> Tuple:
    return tuple(sorted((k, repr(v)) for k, v in attrs.items()))


class CommonSubexpressionElimination(Pass):
    name = "cse"
    commutative = False

    def signature(self, node: Node, arg_ids: Sequence[int]) -> Tuple:
        ids = tuple(arg_ids)
        if self.commutative and node.op.name in COMMUTATIVE_OPS:
            ids = tuple(sorted(ids))
        return (node.op.name, ids, _freeze_attrs(node.attrs))

    def run(self, graph: Graph) -> Graph:
        rw = GraphRewriter(graph)
        rw.copy_inputs()
        seen: Dict[Tuple, object] = {}
        for node in graph.nodes:
            if not node.op.pure:
                rw.emit(node)
                continue
            key = self.signature(node, [rw.map[a.id].id for a in node.inputs])
            if key in seen:
                rw.remap(node, seen[key])
                continue
            value = rw.emit(node)
            seen[key] = value
        return rw.finish()

    def duplicates(self, graph: Graph) -> int:
        """诊断用：按签名统计可被合并的节点数（不含带效果的节点）。"""
        seen = set()
        dup = 0
        for node in graph.nodes:
            if not node.op.pure:
                continue
            key = self.signature(node, [a.id for a in node.inputs])
            if key in seen:
                dup += 1
            else:
                seen.add(key)
        return dup


class CommutativeCSE(CommonSubexpressionElimination):
    """交换律规范化的 CSE：``add(a, b)`` 与 ``add(b, a)`` 合并成一次计算。

    合并保留**先出现**的那个节点（输入顺序不动），因此图的结构与数值都比
    ``cse`` 更保守：只是把后出现的等价节点映射过去。反向图是主场景，但它是
    通用 pass——普通图里手工写出的互换加法同样收益。
    """

    name = "cse-commutative"
    commutative = True
