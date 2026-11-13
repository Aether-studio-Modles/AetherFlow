"""死代码消除。

从图输出反向标记可达节点，未被标记的节点与未被使用的图输入一并丢弃。
它是所有优化管线里最后执行的一步：前面的 pass（CSE、形状化简）
会让一部分节点失去使用者，需要统一清理。

**带效果的节点是根。** RNG / STATE / IO 的效果不体现在数据流上：一个没有
使用者的 ``log`` 节点一旦被删，可观察行为就变了。所以它们的输出与图输出
一样作为可达性起点，连同上游依赖一起保留。
"""

from __future__ import annotations

from typing import Set

from ..ir.graph import Graph
from .base import GraphRewriter, Pass


class DeadCodeElimination(Pass):
    name = "dce"

    def run(self, graph: Graph) -> Graph:
        reachable = self._reachable(graph)
        rw = GraphRewriter(graph)
        rw.copy_inputs(keep=lambda v: v.id in reachable)
        for node in graph.nodes:
            if node.output.id in reachable:
                rw.emit(node)
        return rw.finish()

    @staticmethod
    def _roots(graph: Graph) -> list:
        roots = list(graph.outputs)
        # 未使用的带效果节点不能被删：它的效果不在数据流里。
        roots.extend(n.output for n in graph.nodes if not n.op.pure)
        return roots

    @classmethod
    def _reachable(cls, graph: Graph) -> Set[int]:
        reachable: Set[int] = set()
        stack = cls._roots(graph)
        while stack:
            value = stack.pop()
            if value.id in reachable:
                continue
            reachable.add(value.id)
            if value.producer is not None:
                stack.extend(value.producer.inputs)
        return reachable

    def saved(self, graph: Graph) -> int:
        """诊断用：该图能被删掉的节点数（带效果的节点不计入）。"""
        reachable = self._reachable(graph)
        return sum(
            1 for n in graph.nodes if n.op.pure and n.output.id not in reachable
        )
