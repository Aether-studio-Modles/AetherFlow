"""公共子表达式消除。

签名 = (算子名, 输入的新值 id, 属性的冻结表示)。
签名相同的节点只保留一次，后续出现直接复用第一个结果。

这是图级优化里"库做不到"的典型例子：cuDNN 只负责单个卷积，
它看不见"图上两条完全相同的子计算"。

**只合并纯算子。** 声明了 ``effects`` 的节点（RNG / STATE / IO）即使签名
相同也不能并成一个：两次采样、两次状态写入不是同一次计算，合并会改变语义。
"""

from __future__ import annotations

from typing import Dict, Tuple

from ..ir.graph import Graph
from .base import GraphRewriter, Pass


def _freeze_attrs(attrs: dict) -> Tuple:
    return tuple(sorted((k, repr(v)) for k, v in attrs.items()))


class CommonSubexpressionElimination(Pass):
    name = "cse"

    def run(self, graph: Graph) -> Graph:
        rw = GraphRewriter(graph)
        rw.copy_inputs()
        seen: Dict[Tuple, object] = {}
        for node in graph.nodes:
            if not node.op.pure:
                rw.emit(node)
                continue
            key = (
                node.op.name,
                tuple(rw.map[a.id].id for a in node.inputs),
                _freeze_attrs(node.attrs),
            )
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
            key = (node.op.name, tuple(a.id for a in node.inputs), _freeze_attrs(node.attrs))
            if key in seen:
                dup += 1
            else:
                seen.add(key)
        return dup
