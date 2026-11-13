"""形状与类型噪声消除。

只处理"语义上什么都没做"的节点：
- reshape 到与输入相同的形状
- 恒等置换的 transpose
- cast 到与输入相同的 dtype

这类节点在前端降级过程中会大量产生（尤其是多头拆分/合并附近），
它们不影响正确性，但会白白占掉带宽和调度开销。
"""

from __future__ import annotations

from ..ir.graph import Graph
from .base import GraphRewriter, Pass


class SimplifyShapeOps(Pass):
    name = "simplify-shape-ops"

    def run(self, graph: Graph) -> Graph:
        rw = GraphRewriter(graph)
        rw.copy_inputs()
        for node in graph.nodes:
            if self._is_noop(node):
                rw.remap(node, rw.map[node.inputs[0].id])
            else:
                rw.emit(node)
        return rw.finish()

    @staticmethod
    def _is_noop(node) -> bool:
        if not node.inputs:  # constant 等零输入节点没有可化简的形状算子
            return False
        src = node.inputs[0].type
        if node.op.name == "reshape":
            return tuple(node.attrs["shape"]) == src.shape
        if node.op.name == "transpose":
            return tuple(node.attrs["perm"]) == tuple(range(src.rank))
        if node.op.name == "cast":
            return node.attrs["dtype"] == src.dtype.value
        return False
