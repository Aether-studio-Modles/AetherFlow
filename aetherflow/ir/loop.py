"""循环级 IR：把 ``WhileLoop`` 变成可分析、可重写的循环对象。

5.201 的控制流把循环放在"图与图之间"：``WhileLoop.run`` 每轮在主机上拼
``feeds``；5.245 的设备循环把条件与体各捕获一次。两者都直接吃 ``WhileLoop``，
循环内部的结构没有被任何一层分析过——于是"循环体里哪些计算每轮重算、结果却
不变"这件事无从回答，编译器也就没有做循环级优化的落点。

本层补的是**循环这一级 IR 对象**：

- ``LoopIR``：循环的规范形态（条件图 + 体图 + carried + ``max_steps``）。
  只接受单张 ``Graph``：嵌套控制区域没有可静态分析的单图形态，显式拒绝并
  指回 ``WhileLoop``，不假装能分析。
- ``LoopIR.invariant_nodes()``：体图中不依赖 carried 状态的纯节点。
- ``LoopIR.split_invariants()``：把这些节点搬进 preheader 图，体只保留逐轮
  变化的计算；``LoopSplit.run`` 先跑 preheader 再跑循环，数值与原
  ``WhileLoop`` 逐位相同。
- ``loop_to_spec`` / ``loop_from_spec``：循环级 IR 的 JSON 往返。

**不改变执行语义。** 只有 ``Op.effects`` 为空、且输入不依赖 carried 状态的
节点才外提；``RNG`` / ``STATE`` / ``IO`` 的节点每轮都必须执行，留在体内。
条件图不参与外提——它必须每轮求值，否则"循环什么时候退出"就不再由数据
决定。

**与捕获路径的关系。** ``LoopIR.to_control()`` 返回等价的 ``WhileLoop``，
所以 ``runtime.device_control.DeviceWhileLoop`` 与 ``train.loop.Trainer``
不需要认识本层：先 ``split_invariants()``，再把 ``split.loop.to_control()``
交给它们，就能既少算不变部分、又保持原有执行形态。反过来，宿主路径
``LoopSplit.run`` 也只在 preheader 上多执行一次图。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .control import GraphRegion, WhileLoop
from .graph import Graph, Node, Value
from .serialize import from_dict, to_dict
from .types import TensorType
from .verifier import ERROR, VerifierError, verify

__all__ = [
    "LoopIRError",
    "LoopIR",
    "LoopSplit",
    "loop_to_spec",
    "loop_from_spec",
    "optimize_loop",
]


class LoopIRError(RuntimeError):
    """这张循环进不了循环级 IR。

    与 ``TypeError`` 区分：参数类型没错，缺的是"能静态分析的单图形态"。
    调用方该做的是继续用 ``WhileLoop.run``（宿主路径）或
    ``DeviceWhileLoop``（捕获路径），而不是改参数重试。
    """


def _backend_or_default(backend):
    if backend is not None:
        return backend
    from ..backends.reference_backend import ReferenceBackend

    return ReferenceBackend()


class LoopIR:
    """循环的规范形态：条件图、体图、携带量与步数上限。

    构造期把契约全部跑一遍——直接复用 ``WhileLoop`` 的检查，不另写一份：
    carried 是否封闭、条件是否是单标量输出、类型是否一致，只有一处事实来源。
    """

    def __init__(
        self,
        condition: Graph,
        body: Graph,
        *,
        carried: Mapping[str, str],
        max_steps: int,
        name: str = "while",
    ) -> None:
        if not isinstance(condition, Graph):
            raise TypeError(
                f"LoopIR 的 condition 必须是 Graph，得到 {type(condition).__name__}；"
                "嵌套控制区域没有可静态分析的单图形态"
            )
        if not isinstance(body, Graph):
            raise TypeError(
                f"LoopIR 的 body 必须是 Graph，得到 {type(body).__name__}；"
                "嵌套控制区域没有可静态分析的单图形态"
            )
        self.condition = condition
        self.body = body
        self._control = WhileLoop(
            GraphRegion(condition),
            GraphRegion(body),
            carried=carried,
            max_steps=max_steps,
            name=name,
        )
        self.carried: Dict[str, str] = dict(self._control.carried)
        self.max_steps: int = self._control.max_steps
        self.name: str = self._control.name

    # -- 往返 ---------------------------------------------------------

    @classmethod
    def from_control(cls, loop: WhileLoop) -> "LoopIR":
        """从 ``WhileLoop`` 提取规范形态；嵌套区域显式拒绝。"""
        if not isinstance(loop, WhileLoop):
            raise TypeError(f"from_control 需要 WhileLoop，得到 {type(loop).__name__}")
        regions = []
        for where, region in (("条件", loop.condition), ("循环体", loop.body)):
            if not isinstance(region, GraphRegion):
                raise LoopIRError(
                    f"WhileLoop({loop.name}) 的{where}是 {type(region).__name__}："
                    "循环级 IR 需要单张 Graph 才能做静态分析；"
                    "嵌套控制区域请继续走 WhileLoop.run（或 DeviceWhileLoop 之外的宿主路径）"
                )
            regions.append(region.graph)
        return cls(
            regions[0],
            regions[1],
            carried=loop.carried,
            max_steps=loop.max_steps,
            name=loop.name,
        )

    def to_control(self) -> WhileLoop:
        """还原成等价的 ``WhileLoop``，供既有宿主 / 捕获路径使用。"""
        return self._control

    # -- 只读面 -------------------------------------------------------

    @property
    def inputs(self) -> Tuple[Tuple[str, TensorType], ...]:
        return self._control.inputs

    @property
    def outputs(self) -> Tuple[Tuple[str, TensorType], ...]:
        return self._control.outputs

    def run(
        self, feeds: Mapping[str, np.ndarray], backend: Any = None
    ) -> Dict[str, np.ndarray]:
        """宿主参考执行；与 ``WhileLoop.run`` 是同一条路径。"""
        return self._control.run(feeds, backend)

    # -- 分析 ---------------------------------------------------------

    def _must_run(self) -> set:
        """必须留在循环体里的值 id：依赖 carried、或带可观察效果的节点。

        不纯节点（``RNG`` / ``STATE`` / ``IO``）即使不读状态也不能外提：
        把"每轮重新采样"改成"只采样一次"会改变数值。它的消费者同样不能
        外提，否则新体会引用一个留在体内的值。
        """
        must_run = {value.id for value in self.body.inputs if value.name in self.carried}
        for node in self.body.nodes:
            if not node.op.pure or any(arg.id in must_run for arg in node.inputs):
                must_run.add(node.output.id)
        return must_run

    def invariant_nodes(self) -> Tuple[Node, ...]:
        """体图中每轮重算但结果不变的纯计算（保持构图顺序）。

        ``constant`` 不算：它本身几乎零成本，搬进 preheader 只会多一次
        每轮都要传的输入。常量仍会被复制进 preheader（当它被外提节点依赖
        时），只是不作为 preheader 的输出。
        """
        must_run = self._must_run()
        return tuple(
            node
            for node in self.body.nodes
            if node.output.id not in must_run and node.op.name != "constant"
        )

    # -- 重写 ---------------------------------------------------------

    def split_invariants(self) -> "LoopSplit":
        """把不变计算搬进 preheader，返回新的循环与前置图。

        没有可外提节点时原样返回 ``self``（``preheader`` 为 None）。重写后
        两张图都过一遍 verifier：外提把值从一个图搬到另一个图，最容易出的
        事故是"新图少了一条边"，而它看起来仍然是一张合法 DAG。
        """
        must_run = self._must_run()
        nodes = self.body.nodes

        # 被体内实际消费的不变值：从"必须留在体内的节点输入"与图输出出发，
        # 沿生产者向上收集。没有被任何人使用的不变节点既不进 preheader，
        # 也不进新体——它是死代码，留着只会让两张图都更胖。
        needed: set = set()
        frontier: List[Value] = []
        for node in nodes:
            if node.output.id in must_run:
                frontier.extend(arg for arg in node.inputs if arg.id not in must_run)
        frontier.extend(
            value for value in self.body.outputs if value.id not in must_run
        )
        while frontier:
            value = frontier.pop()
            if value.id in needed or value.id in must_run:
                continue
            needed.add(value.id)
            if value.producer is not None:
                frontier.extend(
                    arg for arg in value.producer.inputs if arg.id not in must_run
                )

        promoted = [
            node.output
            for node in nodes
            if node.output.id in needed and node.op.name != "constant"
        ]
        if not promoted:
            return LoopSplit(loop=self, preheader=None, hoisted=())
        internal = [node.output for node in nodes if node.output.id in needed]

        preheader_inputs = self._preheader_inputs(promoted)
        preheader = self._build_preheader(internal, promoted, preheader_inputs)
        body = self._build_body(must_run, promoted)
        errors = [
            issue
            for graph in (preheader, body)
            for issue in verify(graph)
            if issue.level == ERROR
        ]
        if errors:
            raise VerifierError(errors)
        loop = LoopIR(
            self.condition,
            body,
            carried=self.carried,
            max_steps=self.max_steps,
            name=self.name,
        )
        return LoopSplit(
            loop=loop,
            preheader=preheader,
            hoisted=tuple(value.name for value in promoted),
        )

    def _preheader_inputs(self, promoted: Sequence[Value]) -> Tuple[Value, ...]:
        """preheader 需要的自由输入，按体输入顺序；carried 不可能出现在这里。"""
        used: set = set()
        stack = [value for value in promoted]
        while stack:
            value = stack.pop()
            if value.producer is None:
                used.add(value.id)
                continue
            stack.extend(value.producer.inputs)
        result = []
        for value in self.body.inputs:
            if value.id in used:
                if value.name in self.carried:
                    raise LoopIRError(
                        f"LoopIR({self.name}) 的不变计算依赖了循环状态 %{value.name}："
                        "这说明依赖分析把变化值当成了不变量"
                    )
                result.append(value)
        return tuple(result)

    def _build_preheader(
        self,
        internal: Sequence[Value],
        outputs: Sequence[Value],
        inputs: Sequence[Value],
    ) -> Graph:
        """复制整条不变计算链，但只把 ``outputs`` 暴露给循环体。

        ``internal`` 里可以有 ``constant``：它必须跟着复制进来，否则外提
        节点少一条边。它不出现在输出里——让一个常量每轮作为输入传一遍，
        比留在体内更贵。
        """
        graph = Graph(f"{self.name}__preheader")
        mapping: Dict[int, Value] = {}
        for value in inputs:
            mapping[value.id] = graph.input(value.name, value.type)
        internal_ids = {value.id for value in internal}
        for node in self.body.nodes:
            if node.output.id not in internal_ids:
                continue
            args = [mapping[arg.id] for arg in node.inputs]
            mapping[node.output.id] = graph.add_node(
                node.op.name, args, node.attrs, name=node.output.name
            )
        for value in outputs:
            graph.output(mapping[value.id])
        return graph

    def _build_body(self, must_run: set, promoted: Sequence[Value]) -> Graph:
        """新体：原自由输入 + 外提值作输入，只保留逐轮变化的节点。

        体内直接引用的常量（例如 ``add(s, 1.0)``）就地复制一份：把它也
        变成循环输入等于每轮多传一次常量，而常量节点的执行成本可以忽略。
        """
        graph = Graph(f"{self.name}__body")
        mapping: Dict[int, Value] = {}
        for value in self.body.inputs:
            mapping[value.id] = graph.input(value.name, value.type)

        def resolve(arg: Value) -> Value:
            if arg.id in mapping:
                return mapping[arg.id]
            node = arg.producer
            if node is not None and node.op.name == "constant" and node.op.pure:
                copied = graph.add_node("constant", [], node.attrs)
                mapping[arg.id] = copied
                return copied
            raise LoopIRError(
                f"LoopIR({self.name}) 重写后缺少值 %{arg.name} 的来源："
                "它既不依赖循环状态，也没有被外提；这是依赖分析的不变量被破坏"
            )

        for value in promoted:
            # 名字沿用原 SSA 名：carried 映射、输出别名都不需要改，
            # 调用方也从 preheader 的输出名直接对上。
            mapping[value.id] = graph.input(value.name, value.type)
        for node in self.body.nodes:
            if node.output.id not in must_run:
                continue
            args = [resolve(arg) for arg in node.inputs]
            mapping[node.output.id] = graph.add_node(
                node.op.name, args, node.attrs, name=node.output.name
            )
        for value, alias in zip(self.body.outputs, self.body.output_names):
            target = resolve(value)
            graph.output(target, None if alias == target.name else alias)
        return graph

    def __repr__(self) -> str:
        return (
            f"LoopIR({self.name!r}, carried={list(self.carried)}, "
            f"max_steps={self.max_steps}, nodes={len(self.body.nodes)})"
        )


@dataclass(frozen=True)
class LoopSplit:
    """一次外提的结果：前置图 + 新循环 + 被外提的值名。

    ``preheader`` 为 None 表示体里没有可外提的纯计算，``loop`` 就是原循环。
    ``hoisted`` 是 preheader 的输出名，也是新循环的自由输入名。
    """

    loop: LoopIR
    preheader: Optional[Graph]
    hoisted: Tuple[str, ...]

    def run(
        self, feeds: Mapping[str, np.ndarray], backend: Any = None
    ) -> Dict[str, np.ndarray]:
        """先跑前置图（一次），再跑循环；数值与未外提时逐位相同。"""
        backend = _backend_or_default(backend)
        scope = dict(feeds)
        if self.preheader is not None:
            missing = [
                value.name for value in self.preheader.inputs if value.name not in scope
            ]
            if missing:
                raise KeyError(
                    f"LoopSplit 的前置图缺少输入 {missing}：外提计算需要这些自由输入"
                )
            produced = backend.execute(
                self.preheader,
                {value.name: scope[value.name] for value in self.preheader.inputs},
            )
            missing = [name for name in self.hoisted if name not in produced]
            if missing:
                raise LoopIRError(
                    f"前置图没有产生 {missing}：外提值的输出名与循环的输入名对不上"
                )
            scope.update(produced)
        return self.loop.run(scope, backend)

    @property
    def hoisted_nodes(self) -> int:
        return 0 if self.preheader is None else len(self.preheader.nodes)

    def __repr__(self) -> str:
        return (
            f"LoopSplit(hoisted={list(self.hoisted)}, "
            f"preheader_nodes={self.hoisted_nodes}, loop={self.loop!r})"
        )


def optimize_loop(control: WhileLoop) -> LoopSplit:
    """``WhileLoop`` → 外提后的 ``LoopSplit``；嵌套区域显式拒绝。"""
    return LoopIR.from_control(control).split_invariants()


# --------------------------------------------------------------------------
# 序列化：循环级 IR 与普通图的存档分开，图本身仍走 ir.serialize
# --------------------------------------------------------------------------


def loop_to_spec(loop: LoopIR) -> Dict[str, Any]:
    """导出成纯 JSON 结构；子图走 ``ir.serialize.to_dict``，不重复定义格式。"""
    if not isinstance(loop, LoopIR):
        raise TypeError(f"loop_to_spec 需要 LoopIR，得到 {type(loop).__name__}")
    return {
        "kind": "loop",
        "name": loop.name,
        "condition": to_dict(loop.condition),
        "body": to_dict(loop.body),
        "carried": dict(loop.carried),
        "max_steps": loop.max_steps,
    }


def loop_from_spec(spec: Mapping[str, Any]) -> LoopIR:
    """从 JSON 还原；构造期的契约检查会重新跑一遍。"""
    if not isinstance(spec, Mapping):
        raise TypeError(f"spec 必须是映射，得到 {type(spec).__name__}")
    kind = spec.get("kind")
    if kind != "loop":
        raise ValueError(
            f"不是循环级 IR 存档：kind={kind!r}；循环级 IR 用 ir.loop，"
            "普通控制流用 ir.control.from_spec"
        )
    for key in ("condition", "body", "carried", "max_steps"):
        if key not in spec:
            raise KeyError(f"循环级 IR 存档缺少字段 {key!r}")
    return LoopIR(
        from_dict(spec["condition"]),
        from_dict(spec["body"]),
        carried=dict(spec["carried"]),
        max_steps=spec["max_steps"],
        name=spec.get("name", "while"),
    )
