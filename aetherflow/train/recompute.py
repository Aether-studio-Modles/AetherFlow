"""激活重算（梯度检查点）：把一张静态图切成几段，前向只留段与段之间的边界。

为什么需要它：``ir.grad`` 造出的反向图是自包含的——它会把前向**整张重放**
一遍，因此不需要常驻激活。但一次 ``execute`` 期间执行器不会释放节点输出，
所以反向峰值等于"整张反向图里所有值之和"，图一深就由这个和决定，而不是由
最宽的那一刻决定。

这一层给出的东西很窄：**把前向按节点顺序切成 k 段，前向只保留每段产出的
边界值**。反向从最后一段往前跑，每段自带一张反向图（同样会重放这一段的前
向），于是任一时刻的存活集合 = 全部边界 + 一段的中间值。

三条设计约束：

1. **不改 IR、不改执行器。** 段就是一张普通的 ``Graph``：外部值变成带
   ``ck_`` 前缀的图输入，段内产出的边界值变成图输出。因此它走同一套
   ``Backend.execute``，换后端不需要新代码。
2. **切分是纯拓扑的。** 按节点顺序等分，不猜"哪个算子便宜"。切点落在哪都
   不影响正确性：段内用到的、段外定义的值一律提成输入。
3. **梯度按段倒序累加。** 段内由 ``ir.grad`` 按逆拓扑累加；跨段的同一份
   cotangent 按"后段先加"并入，与单张反向图的处理顺序一致。边界值被多个
   段使用时，浮点加法的**结合顺序**可能与整图反向不同（末位级别），这是
   分段重算的固有代价：链式模型（边界只有一个消费者段）可以做到逐位一致。
4. **段内不能有不可重放的效果。** 重算会把每段前向再跑一遍，声明了
   ``STATE`` / ``IO`` 的算子会被执行两次；构造期直接拒绝。``RNG`` 允许——
   dropout 的随机性由显式 key 给出，同一个 key 重放得到同一位模式。

边界情况：某个边界值不参与损失时，反向图仍然需要它的种子输入，这里喂零。
这与整图反向"跳过该分支"不同——零梯度只会在结果里加进精确的 ``0``，但若
那个分支的前向里出现 ``inf``/``nan``，``0 * inf`` 会污染结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..ir.grad import backward
from ..ir.graph import Graph, Node, Value
from ..ir.ops import UNREPLAYABLE_EFFECTS
from ..ir.shape_contract import require_static_shapes
from ..ir.types import TensorType

__all__ = ["Segment", "CheckpointPlan"]


@dataclass(frozen=True)
class Segment:
    """一段前向子图与它自己的反向图。

    ``inputs`` / ``outputs`` 存 ``(原图里的 SSA 名, 子图里的名字)``：子图的
    输入统一带 ``ck_`` 前缀，输出沿用子图内部的 SSA 名，因此两套名字不会
    互相覆盖（``Graph`` 按名字索引值）。
    """

    graph: Graph
    backward: Graph
    inputs: Tuple[Tuple[str, str], ...]
    outputs: Tuple[Tuple[str, str], ...]
    types: Mapping[str, TensorType]


class CheckpointPlan:
    """一张静态前向图的"分段前向 + 分段反向"计划。

    ``wrt`` 是需要梯度的值（通常是全部参数）；其余外部值（数据输入）也会
    在段反向里被求出来，然后丢弃。``outputs`` 是图的输出名，仅用于校验，
    真正的种子由调用方通过 ``backward(seeds=...)`` 给。
    """

    def __init__(
        self,
        graph: Graph,
        wrt: Sequence[Value],
        segments: int = 2,
        output: Optional[str] = None,
    ) -> None:
        if not isinstance(graph, Graph):
            raise TypeError(f"graph 必须是 Graph，得到 {type(graph).__name__}")
        require_static_shapes(
            graph,
            stage="recompute",
            hint="请用固定形状构图，或关闭重算（segments=None）",
        )
        if isinstance(segments, bool) or not isinstance(segments, int) or segments < 1:
            raise ValueError(f"segments 必须是正整数，得到 {segments!r}")
        nodes = graph.nodes
        if not nodes:
            raise ValueError("空图没有可切分的节点")
        if segments > len(nodes):
            raise ValueError(
                f"段数 {segments} 超过节点数 {len(nodes)}；每段至少要有一个节点"
            )
        # 分段重算会把每段前向再跑一遍。纯算子与显式 key 的 RNG 可以重放，
        # 但 STATE / IO 会被执行两次、效果翻倍，这类图必须在构造期拒绝，
        # 而不是等到训练中途才发现副作用重复（AF-32 第三点）。
        for index, node in enumerate(nodes):
            blocked = node.op.effects & UNREPLAYABLE_EFFECTS
            if blocked:
                names = sorted(e.value for e in blocked)
                raise ValueError(
                    f"节点 #{index} {node.op.name} 声明了不可重放的效果 {names}；"
                    "分段重算会重复执行整段前向，STATE / IO 不能被重放"
                )

        self.graph = graph
        self.output = output
        self._wrt = tuple(v.name for v in wrt)
        self._params = frozenset(self._wrt)

        last_use: Dict[str, int] = {}
        for index, node in enumerate(nodes):
            for a in node.inputs:
                last_use[a.name] = index
        exported_globally = {v.name for v in graph.outputs}

        # ``output`` 之外的图输出不参与反向：沿它做一次逆拓扑可达性，把
        # 到不了它的值排除在切段之外。段反向图因此不会为这些值建种子——
        # 给它们喂零种子代替不了剪枝：``0/x`` 这类项在 x=0 处仍然是 NaN。
        live: Optional[set] = None
        if output is not None:
            by_alias = dict(zip(graph.output_names, graph.outputs))
            if output not in by_alias:
                raise KeyError(
                    f"output={output!r} 不是图输出：{list(graph.output_names)}"
                )
            live = {by_alias[output].name}
            for node in reversed(nodes):
                if node.output.name in live:
                    for a in node.inputs:
                        live.add(a.name)

        def exports_of(lo: int, hi: int) -> List[str]:
            """段 ``[lo, hi)`` 里必须留给后续的值（含图输出）。"""
            names = []
            for node in nodes[lo:hi]:
                name = node.output.name
                if live is not None and name not in live:
                    continue
                if name in exported_globally or last_use.get(name, -1) >= hi:
                    names.append(name)
            return names

        # 等分；一段如果没有任何产出就并进邻段——没有产出的段既无法被后续
        # 读取，也无法给反向图提供种子。
        size = -(-len(nodes) // segments)
        spans: List[Tuple[int, int]] = [
            (lo, min(lo + size, len(nodes)))
            for lo in range(0, len(nodes), size)
        ]
        merged: List[Tuple[int, int]] = []
        for span in spans:
            if merged and not exports_of(*merged[-1]):
                merged[-1] = (merged[-1][0], span[1])
            else:
                merged.append(span)
        # 尾段也可能没有任何导出：它可能整段都在未激活的分支上（例如末尾
        # 挂着诊断输出）。没有导出的段反向图连种子输入都没有，只能并回去。
        while len(merged) > 1 and not exports_of(*merged[-1]):
            merged[-2] = (merged[-2][0], merged[-1][1])
            merged.pop()
        if len(merged) > 1 and not exports_of(*merged[0]):
            merged[0:2] = [(merged[0][0], merged[1][1])]

        self._segments: Tuple[Segment, ...] = tuple(
            self._build(nodes, lo, hi, index, exports_of(lo, hi))
            for index, (lo, hi) in enumerate(merged)
        )

    # -- 构造 ---------------------------------------------------------

    def _build(
        self,
        nodes: Tuple[Node, ...],
        lo: int,
        hi: int,
        index: int,
        exports: Sequence[str],
    ) -> Segment:
        chunk = nodes[lo:hi]
        sub = Graph(f"{self.graph.name}_seg{index}")
        produced_names = {node.output.name for node in chunk}
        declared: Dict[str, Value] = {}
        inputs: List[Tuple[str, str]] = []
        for node in chunk:
            for a in node.inputs:
                if a.name in produced_names or a.name in declared:
                    continue
                local = "ck_" + a.name
                declared[a.name] = sub.input(local, a.type)
                inputs.append((a.name, local))

        produced: Dict[str, Value] = {}
        for node in chunk:
            args = [
                produced[a.name] if a.name in produced else declared[a.name]
                for a in node.inputs
            ]
            produced[node.output.name] = sub.add_node(
                node.op.name, args, dict(node.attrs)
            )

        outputs: List[Tuple[str, str]] = []
        types: Dict[str, TensorType] = {}
        for name in exports:
            value = produced[name]
            sub.output(value, None)
            outputs.append((name, value.name))
            types[name] = value.type

        # 只对浮点输入求导：整数索引（embedding 的 ids）、布尔掩码这些不是
        # 可微量，把它们塞进 wrt 会让 ``backward`` 因为"embedding 对 ids 没有
        # 梯度规则"而整段构造失败。段反向图只管算梯度，多算一个整数输入的
        # 梯度没有意义。
        differentiable = tuple(
            v for v in declared.values()
            if np.dtype(v.type.dtype.np_dtype).kind == "f"
        )
        # 段的所有导出都已在切段时按"损失可达"过滤过，因此这里可以不喂零
        # 种子、直接让反向图只认这几个导出（别名取子图自己的名字）。
        bwd = backward(
            sub,
            wrt=differentiable,
            name=f"{sub.name}_backward",
            active=list(sub.output_names),
        )
        return Segment(
            graph=sub,
            backward=bwd,
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            types=types,
        )

    # -- 访问 ---------------------------------------------------------

    @property
    def segments(self) -> Tuple[Segment, ...]:
        return self._segments

    # -- 执行 ---------------------------------------------------------

    def _feed_of(self, segment: Segment, feed, acts) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for orig, local in segment.inputs:
            if orig in feed:
                out[local] = feed[orig]
            elif orig in acts:
                out[local] = acts[orig]
            else:
                raise KeyError(
                    f"段 {segment.graph.name} 需要的 %{orig} 既不在输入里，"
                    "也不是前面某段的产出（切分出错）"
                )
        return out

    def forward(self, backend, feed: Mapping[str, Any]) -> Dict[str, Any]:
        """逐段前向，返回"边界值 + 图输出"（原名 -> 容器值）。"""
        acts: Dict[str, Any] = {}
        for segment in self._segments:
            result = backend.execute(segment.graph, self._feed_of(segment, feed, acts))
            for orig, local in segment.outputs:
                acts[orig] = result[local]
        return acts

    def backward(
        self,
        backend,
        feed: Mapping[str, Any],
        acts: Mapping[str, Any],
        seeds: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """逐段反向，返回 ``wrt`` 的梯度（原名 -> 数组）。

        ``feed`` 是前向时的输入（参数与数据），``acts`` 是 ``forward`` 的产出。
        ``seeds`` 按**原图值的名字**给：图输出如果不止一个，不参与损失的那些
        由调用方喂零（与整图反向的约定一致）。
        """
        leftovers: Dict[str, Any] = dict(seeds or {})  # 还没被消费的种子
        cot: Dict[str, Any] = {}  # 边界值的 cotangent 累加
        param_grads: Dict[str, Any] = {}
        for segment in reversed(self._segments):
            seg_feed = self._feed_of(segment, feed, acts)
            for orig, local in segment.outputs:
                # 边界值可能同时是"外部给了种子"和"后段传回了 cotangent"。
                # 两者必须**相加**：外部种子（例如辅助输出被喂零，或调用方对
                # 某个输出另有目标）与下游梯度是同一条边上的两份贡献，任一
                # 份都不能覆盖另一份。早先写成二选一，会让辅助输出的零种子
                # 把整条后段梯度抹掉。
                total = None
                for bucket in (leftovers, cot):
                    part = bucket.pop(orig, None)
                    if part is None:
                        continue
                    total = part if total is None else total + part
                if total is None:
                    type_ = segment.types[orig]
                    total = np.zeros(
                        tuple(int(d) for d in type_.shape),
                        dtype=type_.dtype.np_dtype,
                    )
                seg_feed["d" + local] = total
            result = backend.execute(segment.backward, seg_feed)
            for orig, local in segment.inputs:
                key = "d" + local
                if key not in result:
                    continue  # 非浮点输入（整数索引 / 掩码）没有梯度节点
                contribution = result[key]
                bucket = param_grads if orig in self._params else cot
                if orig in bucket:
                    bucket[orig] = bucket[orig] + contribution
                else:
                    bucket[orig] = contribution

        missing = [name for name in self._wrt if name not in param_grads]
        if missing:
            raise KeyError(f"这些求导目标没有梯度: {missing}")
        return param_grads
