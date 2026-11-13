"""IR 层算子融合。

融合的边界（与 ADR-003 / ADR-010 一致）：

- 融合结果必须是**语义完整的 IR 节点**，而不是后端私有技巧。
  这里的做法是扩展已有算子的输入，而不是发明新算子：
  ``softmax(mul(x, s))`` 变成 ``softmax(x, s)``，
  ``layernorm(add(x, r), g, b)`` 变成 ``layernorm(x, g, b, r)``，
  ``mul(a, silu(b))`` 变成 ``silu(b, a)``。
- 因此融合后的图仍能被参考执行器执行，也能被任何承接该算子的后端执行。
  不需要"拆解 pass"，也就不存在"融合完没人能执行"的问题。
- 代价是融合规则必须逐条对应到真实算子的语义，不能任意组合；
  换来的是**融合结果始终可被参考语义做差分验证**。

为什么不在后端内部偷偷融合：那样优化不可移植，也无法用唯一真值验证。
"""

from __future__ import annotations

from typing import Dict, Set, Tuple

from ..ir.graph import Graph, Value
from ..ir.types import TensorType
from .base import GraphRewriter, Pass, PassManager

__all__ = [
    "FuseSoftmaxScale",
    "FuseLayerNormResidual",
    "FuseSiLU",
    "fused_pipeline",
]


def _is_scalar(t: TensorType) -> bool:
    """静态且元素个数为 1。符号维无法在编译期确定，因此不参与融合。"""
    return t.is_static and t.numel() == 1


def _consumers(graph: Graph) -> Dict[int, int]:
    consumers: Dict[int, int] = {}
    for node in graph.nodes:
        for v in node.inputs:
            consumers[v.id] = consumers.get(v.id, 0) + 1
    return consumers


class FuseSoftmaxScale(Pass):
    """把 ``softmax(mul(x, s))`` 融合成 ``softmax(x, s)``，s 为标量。

    判据（必须同时满足）：

    1. softmax 只有单一输入，且其生产者是 mul
    2. mul 的其中一侧是标量（元素个数为 1）
    3. mul 的输出只被这个 softmax 消费（fanout == 1）
    4. mul 的输出不是图输出

    第 3、4 条是为了保证语义不变：只要 mul 的结果还被别处使用，
    就不能把它吸收进 softmax。
    """

    name = "fuse_softmax_scale"

    def run(self, graph: Graph) -> Graph:
        consumers = _consumers(graph)
        outputs = {v.id for v in graph.outputs}

        merged: Dict[int, Tuple[Value, Value]] = {}
        absorbed: Set[int] = set()

        for node in graph.nodes:
            if node.op.name != "softmax" or len(node.inputs) != 1:
                continue
            producer = node.inputs[0].producer
            if producer is None or producer.op.name != "mul":
                continue
            if consumers.get(producer.output.id, 0) != 1:
                continue
            if producer.output.id in outputs:
                continue
            a, b = producer.inputs
            if _is_scalar(b.type):
                merged[node.output.id] = (a, b)
            elif _is_scalar(a.type):
                merged[node.output.id] = (b, a)
            else:
                continue
            absorbed.add(producer.output.id)

        if not merged:
            return graph

        rw = GraphRewriter(graph)
        rw.copy_inputs()
        for node in graph.nodes:
            if node.output.id in absorbed:
                continue
            pair = merged.get(node.output.id)
            if pair is not None:
                x, scale = pair
                rw.map[node.output.id] = rw.new.add_node(
                    "softmax", [rw.map[x.id], rw.map[scale.id]], node.attrs
                )
                continue
            rw.emit(node)
        return rw.finish()


class FuseLayerNormResidual(Pass):
    """把 ``layernorm(add(a, b), g, beta)`` 融合成 ``layernorm(a, g, beta, b)``。

    这是 ADR-010 所说的"整段融合"：残差的加法被搬进 LayerNorm 节点，
    ``add`` 的输出再也不物化——LayerNorm 内核在归约前直接在寄存器里做加法。
    语义等价由 ``ir/ops.py`` 的参考实现定义：``LayerNorm(x + residual)``。

    判据（必须同时满足）：

    1. layernorm 恰好是 3 输入形态（已经是融合形态就不再处理）
    2. 它的第 0 个输入由一个 ``add`` 生产
    3. add 的输出只被这个 layernorm 消费（fanout == 1）
    4. add 的输出不是图输出
    5. add 的两个输入与 layernorm 的第 0 个输入**同类型**（不吸收广播）

    第 5 条是严格限制：若 add 依赖广播（例如 ``(B,S,D) + (D,)``），
    把两个张量都塞进 layernorm 就需要在内核里复刻广播规则，得不偿失；
    参考语义仍然正确，但没有对应的内核承接，融合结果会变成"没人能执行"。

    注意真实 pre-norm Transformer 里 ``x1 = add(x, attn)`` 同时喂给
    ``layernorm(x1)`` 与最终残差 ``add(x1, ff)``，fanout 是 2，因此**不会**
    命中这条规则——残差本身仍然要被物化，融合只剩"少读一遍"，反而不划算。
    这条规则服务的是残差只有一个使用者的图（单分支 / post-norm / 子块）。
    """

    name = "fuse_layernorm_residual"

    def run(self, graph: Graph) -> Graph:
        consumers = _consumers(graph)
        outputs = {v.id for v in graph.outputs}

        merged: Dict[int, Tuple[Value, Value]] = {}
        absorbed: Set[int] = set()

        for node in graph.nodes:
            if node.op.name != "layernorm" or len(node.inputs) != 3:
                continue
            producer = node.inputs[0].producer
            if producer is None or producer.op.name != "add":
                continue
            if consumers.get(producer.output.id, 0) != 1:
                continue
            if producer.output.id in outputs:
                continue
            a, b = producer.inputs
            src = node.inputs[0].type
            if a.type != src or b.type != src:
                continue
            merged[node.output.id] = (a, b)
            absorbed.add(producer.output.id)

        if not merged:
            return graph

        rw = GraphRewriter(graph)
        rw.copy_inputs()
        for node in graph.nodes:
            if node.output.id in absorbed:
                continue
            pair = merged.get(node.output.id)
            if pair is not None:
                a, b = pair
                gamma, beta = node.inputs[1], node.inputs[2]
                rw.map[node.output.id] = rw.new.add_node(
                    "layernorm",
                    [rw.map[a.id], rw.map[gamma.id], rw.map[beta.id], rw.map[b.id]],
                    node.attrs,
                )
                continue
            rw.emit(node)
        return rw.finish()


class FuseSiLU(Pass):
    """把 ``mul(a, silu(b))`` 融合成 ``silu(b, a)``，即 SwiGLU 的门控乘法。

    判据（必须同时满足）：

    1. mul 两侧有一侧由 ``silu`` 生产，且 silu 是 1 输入形态
    2. silu 的输出只被这个 mul 消费（fanout == 1），且不是图输出
    3. 另一侧与 silu 的输出同类型（不吸收广播）

    收益是实的：原图要写一次 ``silu(b)`` 的中间张量、再读一次，
    融合后这条链的访存从 5 遍降到 3 遍，还少一次 launch。
    """

    name = "fuse_silu_gate"

    def run(self, graph: Graph) -> Graph:
        consumers = _consumers(graph)
        outputs = {v.id for v in graph.outputs}

        merged: Dict[int, Tuple[Value, Value]] = {}  # mul 输出 -> (x, gate)
        absorbed: Set[int] = set()

        for node in graph.nodes:
            if node.op.name != "mul":
                continue
            left, right = node.inputs
            for x, gate in ((left, right), (right, left)):
                producer = x.producer
                if producer is None or producer.op.name != "silu":
                    continue
                if len(producer.inputs) != 1:
                    continue
                if consumers.get(producer.output.id, 0) != 1:
                    continue
                if producer.output.id in outputs:
                    continue
                if gate.type != producer.output.type:
                    continue
                merged[node.output.id] = (producer.inputs[0], gate)
                absorbed.add(producer.output.id)
                break

        if not merged:
            return graph

        rw = GraphRewriter(graph)
        rw.copy_inputs()
        for node in graph.nodes:
            if node.output.id in absorbed:
                continue
            pair = merged.get(node.output.id)
            if pair is not None:
                x, gate = pair
                rw.map[node.output.id] = rw.new.add_node(
                    "silu", [rw.map[x.id], rw.map[gate.id]], node.attrs
                )
                continue
            rw.emit(node)
        return rw.finish()


def fused_pipeline() -> PassManager:
    """带融合的优化管线。

    融合会让节点多出可选输入，因此要求目标后端承接这些形态；
    参考后端与 CUDA 后端都已支持。默认管线（无硬件假设）刻意不含它。
    """
    from .cse import CommonSubexpressionElimination
    from .dce import DeadCodeElimination
    from .fold import ConstantFolding
    from .simplify import SimplifyShapeOps

    return PassManager(
        [
            ConstantFolding(),
            SimplifyShapeOps(),
            CommonSubexpressionElimination(),
            FuseSoftmaxScale(),
            FuseLayerNormResidual(),
            FuseSiLU(),
            DeadCodeElimination(),
        ]
    )
