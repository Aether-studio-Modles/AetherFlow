"""设备端控制流：把控制区域内的每张图各捕获一次，状态常驻显存。

``DeviceWhileLoop`` 处理循环，``DeviceConditional`` 处理分支；两者都不改
5.201 的控制流语义，只换执行形态。

5.201 的控制流表达只解决"语义"：``WhileLoop.run`` 每轮把状态从显存拷回主机、
在主机上拼 ``feeds``、再交给 ``execute`` 拷上去。数值是对的，代价是每轮两次
整份往返；显存里的状态因此从来没有"留在显存里过"。

这一层补的是**执行形态**，不改控制流语义：

- 条件图与循环体各 ``capture`` 一次（同一条流），之后每轮只 ``replay``；
  循环跑多少轮，捕获次数都是 2。
- 循环携带量（``carried``）不再回主机：体 replay 的输出直接当下一轮的输入，
  由计划在自己的流上做一次 D2D 拷进静态缓冲。计划不重建、不重捕获。
- 每轮只有**一次同步、一次主机往返**：读谓词标量。这是数据相关分支绕不过的
  一次同步——没有 device-side 循环，也没有 CUDA Graph 的条件节点；把它藏
  起来只会让"看起来零同步"变成一句假话。体的 replay 是异步的，它的完成由
  "下一轮条件的 replay 在同一条流上 + ``run_device`` 的同步"共同保证。
- 自由输入只在建计划时上传一次；``run(feeds)`` 给了新值才重传。

``DeviceConditional`` 是它的对偶：谓词与两个分支各捕获一次，每次只 replay
谓词与**被选中的那一支**。未选中的分支一次都不执行——与 ``Conditional.run``
的惰性一致；分支输出留在显存里，``run_device`` 不做任何 D2H。

为什么放在 ``runtime`` 而不是 ``ir/control.py``
-----------------------------------------------
控制流是 IR 层的语义（``ir`` 不依赖任何后端能力，``WhileLoop.run`` 连
``capture`` 都不认识）。设备形态需要的是"后端有 capture / new_stream /
to_array / to_host"这组运行期能力，与 ``train/loop.py`` 的设备态训练是同一套
前置。把它写进 ``ir`` 会让 IR 层多一个只有设备后端才满足的契约。

契约（与宿主路径逐条对齐，且全部显式失败）
------------------------------------------
- 只支持**谓词、分支、条件与循环体都是单张 ``Graph``**（``GraphRegion``）的
  形态。嵌套的 ``Conditional`` / ``WhileLoop`` 仍走宿主路径（``Conditional.run``
  / ``WhileLoop.run``）：嵌套区域的输出不是一张图，没有可捕获的 launch 序列，
  假装支持只会得到"捕获了半个控制结构"。
- ``max_steps`` 用尽仍抛 ``ControlFlowError``，消息与宿主路径同一个形状；
  设备路径不会因为"已经捕获了"就静默截断。
- 计划必须提供 ``replay`` / ``device_outputs``（异步执行的一面）。只有
  ``run_device``（带同步）的实现不接受：那等于每轮多一次同步，本层的时序
  前提也就变了。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from ..ir.control import Conditional, ControlFlowError, GraphRegion, WhileLoop

__all__ = ["DeviceLoopError", "DeviceWhileLoop", "DeviceConditional"]

# 与 ``Trainer(device_state=True)`` 同一组前置能力，外加"能开一条自己的流"。
_REQUIRED = ("capture", "new_stream", "to_array", "to_host")
# 计划的异步执行面：缺了它只能退回每轮两次同步的形态，本层不做这种降级。
_PLAN_API = ("replay", "device_outputs")


class DeviceLoopError(RuntimeError):
    """设备端控制流不接受的形态（例如嵌套控制区域）。

    与 ``ControlFlowError`` 区分：后者是"图合法但这次执行没收敛"，这里是
    "这个控制结构没有设备形态"——构图期就该知道的事。
    """


def _require_device_backend(backend) -> None:
    missing = [name for name in _REQUIRED if not callable(getattr(backend, name, None))]
    if missing:
        raise TypeError(
            "设备端控制流需要支持整图捕获的后端（capture/new_stream/to_array/to_host），"
            f"{type(backend).__name__} 缺少 {missing}"
        )


class DeviceWhileLoop:
    """一次捕获、多轮 replay 的 ``WhileLoop``。

    建计划时就完成上传与两次捕获；``run`` / ``run_device`` 可以重复调用，
    每次从调用方给的初值重新开始（状态指针复位到初值缓冲，不重新上传）。
    """

    def __init__(self, loop: WhileLoop, feeds: Mapping[str, Any], backend) -> None:
        if not isinstance(loop, WhileLoop):
            raise TypeError(f"DeviceWhileLoop 需要 WhileLoop，得到 {type(loop).__name__}")
        _require_device_backend(backend)
        for where, region in (("条件", loop.condition), ("循环体", loop.body)):
            if not isinstance(region, GraphRegion):
                raise DeviceLoopError(
                    f"DeviceWhileLoop({loop.name}) 的{where}不是单张 Graph："
                    f"{type(region).__name__}。嵌套控制区域没有可捕获的 launch 序列，"
                    "请继续用 WhileLoop.run（每次判定回主机）"
                )

        self.loop = loop
        self.backend = backend
        self.captures = 0
        self.iterations = 0
        self.syncs = 0

        carried = tuple(loop.carried)
        free = tuple(name for name, _ in loop.inputs)
        needed = free + carried
        unknown = sorted(set(feeds) - set(needed))
        if unknown:
            raise KeyError(f"这些名字不是 WhileLoop({loop.name}) 的输入: {unknown}")
        missing_state = [state for state in carried if state not in feeds]
        if missing_state:
            raise ValueError(
                f"WhileLoop({loop.name}) 缺少初始状态 {missing_state}；"
                "循环携带量必须由调用方给出初值"
            )
        missing_free = [name for name in free if name not in feeds]
        if missing_free:
            raise KeyError(f"缺少图输入: {missing_free}")

        # 每张图的两个输入面各有一块常驻缓冲；形状/dtype 以建计划时为准。
        self._buffers: Dict[str, Any] = {
            name: backend.to_array(np.asarray(feeds[name])) for name in needed
        }
        self._shapes: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {
            name: (tuple(buf.shape), np.dtype(buf.dtype))
            for name, buf in self._buffers.items()
        }
        self._initial = {state: self._buffers[state] for state in carried}

        # 条件与体必须同一条流：体的 replay 是异步的，下一轮条件的 replay
        # 只有排在同一条流上才有"体已完成"这个前提。
        self._stream = backend.new_stream()
        condition = loop.condition.graph
        body = loop.body.graph
        self._cond_graph = condition
        self._body_graph = body
        self._cond_feed = self._feed(condition, "条件")
        self._body_feed = self._feed(body, "循环体")
        self._cond_plan = backend.capture(
            condition, self._cond_feed,
            shared_buffers=self._cond_feed, stream=self._stream,
        )
        self.captures += 1
        self._body_plan = backend.capture(
            body, self._body_feed,
            shared_buffers=self._body_feed, stream=self._stream,
        )
        self.captures += 1
        for plan, where in ((self._cond_plan, "条件"), (self._body_plan, "循环体")):
            missing = [name for name in _PLAN_API if not callable(getattr(plan, name, None))]
            if missing:
                raise DeviceLoopError(
                    f"DeviceWhileLoop({loop.name}) 的{where}计划缺少 {missing}："
                    "设备端循环要求计划提供异步 replay 面"
                )
        self._predicate = loop.condition.outputs[0][0]

    # -- 结构 ---------------------------------------------------------

    @property
    def inputs(self) -> Tuple[Tuple[str, Any], ...]:
        return self.loop.inputs

    @property
    def outputs(self) -> Tuple[Tuple[str, Any], ...]:
        return self.loop.outputs

    def _feed(self, graph, where: str) -> Dict[str, Any]:
        feed: Dict[str, Any] = {}
        for value in graph.inputs:
            buf = self._buffers.get(value.name)
            if buf is None:
                raise KeyError(
                    f"{where}读到了没有缓冲的输入 %{value.name}："
                    "设备端循环要求自由输入与循环携带量覆盖两张图的全部输入"
                )
            feed[value.name] = buf
        return feed

    def _check(self, name: str, value: Any) -> Any:
        """形状/dtype 对齐到声明值；主机数组顺手上传成设备数组。"""
        if value is self._buffers[name]:
            return value
        if isinstance(value, np.ndarray):
            value = self.backend.to_array(value)
        shape = tuple(np.shape(value))
        dtype = np.dtype(value.dtype)
        if (shape, dtype) != self._shapes[name]:
            raise ValueError(
                f"输入 %{name} 的形状/dtype 是 {shape}/{dtype}，"
                f"与建计划时的 {self._shapes[name][0]}/{self._shapes[name][1]} 不一致："
                "一张捕获图只对应一组固定形状"
            )
        return value

    def _reset(self, feeds: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        """每次 run 开头：状态回到初值指针，显式给的新自由输入写回常驻缓冲。"""
        state = dict(self._initial)
        if feeds is None:
            return state
        unknown = sorted(set(feeds) - set(self._buffers))
        if unknown:
            raise KeyError(f"这些名字不是 WhileLoop({self.loop.name}) 的输入: {unknown}")
        for name, value in feeds.items():
            if name in state:
                state[name] = self._check(name, value)
            else:
                self._buffers[name][...] = self._check(name, value)
        return state

    def _state_feed(self, graph, state: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            value.name: state.get(value.name, self._buffers[value.name])
            for value in graph.inputs
        }

    # -- 执行 ---------------------------------------------------------

    def run(self, feeds: Optional[Mapping[str, Any]] = None) -> Dict[str, np.ndarray]:
        """跑完循环，返回主机侧的状态（每个状态一次 D2H，收尾统一做）。"""
        state = self.run_device(feeds)
        return {
            name: np.asarray(self.backend.to_host(state[name]))
            for name in self.loop.carried
        }

    def run_device(
        self, feeds: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        """跑完循环，返回仍在显存里的状态。

        每轮只有一次主机往返（谓词标量）与一次同步（条件计划的 replay）；
        状态的 D2H/H2D 一次都不发生。
        """
        state = self._reset(feeds)
        self.iterations = 0
        self.syncs = 0
        name = self.loop.name
        for step in range(self.loop.max_steps):
            produced = self._cond_plan.run_device(
                self._state_feed(self._cond_graph, state)
            )
            self.syncs += 1
            if self._predicate not in produced:
                raise ControlFlowError(
                    f"WhileLoop({name}) 的条件输出 %{self._predicate} 没有出现在执行结果里"
                )
            flag = np.asarray(self.backend.to_host(produced[self._predicate]))
            if flag.size != 1:
                raise ControlFlowError(
                    f"WhileLoop({name}) 的条件输出 %{self._predicate} 运行期形状为 "
                    f"{flag.shape}，与 IR 声明的 0 维标量不一致"
                )
            if not bool(flag.reshape(()).item()):
                return {key: state[key] for key in self.loop.carried}

            # 异步 replay：不等它完成。体与条件是同一条流，下一轮条件的
            # replay 必然排在它后面，那里的同步就覆盖了这次体。
            self._body_plan.replay(self._state_feed(self._body_graph, state))
            produced = self._body_plan.device_outputs()
            state = {}
            for key, out in self.loop.carried.items():
                if out not in produced:
                    raise ControlFlowError(
                        f"WhileLoop({name}) 第 {step} 轮的循环体没有产生 %{out}"
                    )
                state[key] = produced[out]
            self.iterations += 1
        raise ControlFlowError(
            f"WhileLoop({name}) 达到 max_steps={self.loop.max_steps} 仍未退出条件；"
            "请调大 max_steps，或检查条件是否真的会变为假"
        )

    def __repr__(self) -> str:
        return (
            f"DeviceWhileLoop({self.loop.name!r}, carried={list(self.loop.carried)}, "
            f"max_steps={self.loop.max_steps}, captures={self.captures})"
        )


class DeviceConditional:
    """一次捕获、按谓词 replay 一支的 ``Conditional``。

    谓词与两个分支各捕获一次（同一条流）。每次 ``run`` 只 replay 谓词与
    **被选中的那一支**；未选中的分支一次都不执行，这与宿主路径的惰性一致。
    分支输出留在显存里，``run_device`` 不做任何 D2H。
    """

    def __init__(
        self, conditional: Conditional, feeds: Mapping[str, Any], backend
    ) -> None:
        if not isinstance(conditional, Conditional):
            raise TypeError(
                f"DeviceConditional 需要 Conditional，得到 "
                f"{type(conditional).__name__}"
            )
        _require_device_backend(backend)
        regions = (
            ("谓词", conditional.predicate),
            ("then 分支", conditional.then),
            ("otherwise 分支", conditional.otherwise),
        )
        for where, region in regions:
            if not isinstance(region, GraphRegion):
                raise DeviceLoopError(
                    f"DeviceConditional({conditional.name}) 的{where}不是单张 Graph："
                    f"{type(region).__name__}。嵌套控制区域没有可捕获的 launch 序列，"
                    "请继续用 Conditional.run（每次判定回主机）"
                )

        self.conditional = conditional
        self.backend = backend
        self.captures = 0
        self.syncs = 0
        self.branches = 0

        needed = tuple(name for name, _ in conditional.inputs)
        unknown = sorted(set(feeds) - set(needed))
        if unknown:
            raise KeyError(
                f"这些名字不是 Conditional({conditional.name}) 的输入: {unknown}"
            )
        missing = [name for name in needed if name not in feeds]
        if missing:
            raise KeyError(f"缺少图输入: {missing}")

        self._buffers: Dict[str, Any] = {
            name: backend.to_array(np.asarray(feeds[name])) for name in needed
        }
        self._shapes: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {
            name: (tuple(buf.shape), np.dtype(buf.dtype))
            for name, buf in self._buffers.items()
        }

        # 三张图共用一条流：分支的 replay 是异步的，但调用方紧接着读
        # device_outputs 或下一次 run 的谓词；同一条流让顺序有唯一解释。
        self._stream = backend.new_stream()
        self._graphs = [region.graph for _, region in regions]
        self._plans = []
        for where, region in regions:
            feed = self._feed(region.graph, where)
            plan = backend.capture(
                region.graph, feed, shared_buffers=feed, stream=self._stream
            )
            self.captures += 1
            missing_api = [
                name for name in _PLAN_API if not callable(getattr(plan, name, None))
            ]
            if missing_api:
                raise DeviceLoopError(
                    f"DeviceConditional({conditional.name}) 的{where}计划缺少 "
                    f"{missing_api}：设备端分支要求计划提供异步 replay 面"
                )
            self._plans.append(plan)
        self._predicate = conditional.predicate.outputs[0][0]
        self._outputs = conditional.outputs

    # -- 结构 ---------------------------------------------------------

    @property
    def inputs(self) -> Tuple[Tuple[str, Any], ...]:
        return self.conditional.inputs

    @property
    def outputs(self) -> Tuple[Tuple[str, Any], ...]:
        return self.conditional.outputs

    def _feed(self, graph, where: str) -> Dict[str, Any]:
        feed: Dict[str, Any] = {}
        for value in graph.inputs:
            buf = self._buffers.get(value.name)
            if buf is None:
                raise KeyError(
                    f"{where}读到了没有缓冲的输入 %{value.name}："
                    "设备端分支要求 feeds 覆盖三张图的全部输入"
                )
            feed[value.name] = buf
        return feed

    def _check(self, name: str, value: Any) -> Any:
        """形状/dtype 对齐到声明值；主机数组顺手上传成设备数组。"""
        if value is self._buffers[name]:
            return value
        if isinstance(value, np.ndarray):
            value = self.backend.to_array(value)
        shape = tuple(np.shape(value))
        dtype = np.dtype(value.dtype)
        if (shape, dtype) != self._shapes[name]:
            raise ValueError(
                f"输入 %{name} 的形状/dtype 是 {shape}/{dtype}，"
                f"与建计划时的 {self._shapes[name][0]}/{self._shapes[name][1]} 不一致："
                "一张捕获图只对应一组固定形状"
            )
        return value

    def _apply(self, feeds: Optional[Mapping[str, Any]]) -> None:
        """显式给的新输入写回常驻缓冲；没给就沿用上一次的值。"""
        if feeds is None:
            return
        unknown = sorted(set(feeds) - set(self._buffers))
        if unknown:
            raise KeyError(
                f"这些名字不是 Conditional({self.conditional.name}) 的输入: {unknown}"
            )
        for name, value in feeds.items():
            self._buffers[name][...] = self._check(name, value)

    # -- 执行 ---------------------------------------------------------

    def run(self, feeds: Optional[Mapping[str, Any]] = None) -> Dict[str, np.ndarray]:
        """按谓词走一支，返回主机侧输出（每个输出一次 D2H，收尾统一做）。"""
        state = self.run_device(feeds)
        return {
            name: np.asarray(self.backend.to_host(state[name]))
            for name, _ in self._outputs
        }

    def run_device(
        self, feeds: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        """按谓词走一支，返回仍在显存里的输出。

        每次只有一次主机往返（谓词标量）与一次同步（谓词计划的 replay）；
        分支输出的 D2H/H2D 一次都不发生，未选中的分支不执行。
        """
        self._apply(feeds)
        self.syncs = 0
        self.branches = 0
        produced = self._plans[0].run_device(
            self._feed(self._graphs[0], "谓词")
        )
        self.syncs += 1
        name = self.conditional.name
        if self._predicate not in produced:
            raise ControlFlowError(
                f"Conditional({name}) 的谓词输出 %{self._predicate} "
                "没有出现在执行结果里"
            )
        flag = np.asarray(self.backend.to_host(produced[self._predicate]))
        if flag.size != 1:
            raise ControlFlowError(
                f"Conditional({name}) 的谓词输出 %{self._predicate} 运行期形状为 "
                f"{flag.shape}，与 IR 声明的 0 维标量不一致"
            )
        index = 1 if bool(flag.reshape(()).item()) else 2
        where = "then 分支" if index == 1 else "otherwise 分支"
        # 异步 replay：只跑被选中的一支，未选中的计划连作业都不提交。
        self._plans[index].replay(self._feed(self._graphs[index], where))
        self.branches += 1
        produced = self._plans[index].device_outputs()
        result: Dict[str, Any] = {}
        for output, _ in self._outputs:
            if output not in produced:
                raise ControlFlowError(
                    f"Conditional({name}) 的{where}没有产生 %{output}"
                )
            result[output] = produced[output]
        return result

    def __repr__(self) -> str:
        return (
            f"DeviceConditional({self.conditional.name!r}, "
            f"outputs={[n for n, _ in self._outputs]}, captures={self.captures})"
        )
