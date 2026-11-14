"""GPipe 式流水：把一张图按层切成若干段，每段常驻一块卡。

数据并行（5.190）解决"一批数据摊到多块卡"，解决不了"**模型本身**一块卡放
不下"。这一层把整图按用户给的切点拆成有序的段：

    x --[段 0]--> h0 --[段 1]--> h1 ... --[段 n-1]--> loss

每段是一张独立的 ``Graph``（用户自己切、自己给名字），段间只通过一个激活
张量相连；每段的参数常驻该段所在的卡。训练一步是：

    每个 micro-batch:
        前向：段 0 → 段 n-1，段间激活跨卡（主机中转）
        反向：段 n-1 → 段 0，``backward`` 的自定义种子把 dout 传回上一段
    主机侧把各 micro-batch 的梯度按权重相加 → optimizer.step
    每段参数写回本段所在卡

**这不是重叠式流水。** 后端是单进程串行切卡（5.190 的边界），所以"前向的
micro-batch 1 与反向的 micro-batch 0 并发"做不到；本实现给的是 **GPipe 的
调度骨架**：micro-batch 逐个走完全部段，再逆序反向。收益是**容量**（每段
只占一块卡的显存/参数），不是吞吐。如实写在接口上，不假装能重叠。

**数值上不要求逐位相同。** 整批一次算与 micro-batch 分别算再相加，矩阵乘
内部的归约顺序不同，浮点结果会有 1e-7 量级的差异。判据因此是"与单卡整图
在 f32 容差内一致"，而 ZeRO 分片那条"逐位一致"的强判据在这里不成立。

**损失必须是按样本求均值**（与 5.190 同一条约定），且 micro-batch 只做**等分**：
``batch % micro_batches == 0``。只有等分时"各 micro-batch 梯度求和再除以份数"
才等于"整批均值"；不等分要先按样本数加权，那套权重在静态图里表达不了
（每份的样本数是运行期才知道的）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..ir.grad import backward
from ..ir.graph import Graph
from .optim import Optimizer
from .parallel import (
    _REDUCTIONS,
    _check_device_index,
    _check_multi_device_backend,
    copy_across_devices,
)
from .params import ParameterStore

__all__ = ["PipelineStage", "PipelineTrainer"]

#: 三种调度。``sequential`` 是既有行为：一份走完前向与反向，再下一份；
#: ``gpipe`` 是经典批式：全部份前向完再全部反向（驻留 = 份数）；``1f1b``
#: 预热"段数-1"份后交替"反向一份、前向下一份"（驻留 = 段数-1）。
#: 三者数值等价，差别只在"有几份段间激活同时活着"。
_SCHEDULES = ("sequential", "gpipe", "1f1b")


@dataclass(frozen=True)
class PipelineStage:
    """流水的一段：一张图 + 它独占的参数 + 它与上一段的连接。

    - ``params``：本段参数名（必须是本段图的输入）。各段的参数名不能重叠，
      并集必须覆盖 ``ParameterStore`` 的全部名字。
    - ``inputs``：段间激活在本段图里的输入名。第 0 段必须为空；其余段必须
      恰好一个，且等于上一段的 ``output``。
    - ``output``：本段输出名（图输出别名）。最后一段的输出是标量损失。
    """

    graph: Graph
    params: Tuple[str, ...]
    inputs: Tuple[str, ...] = ()
    output: Optional[str] = None


class PipelineTrainer:
    """把一段流水跑起来：每段一块卡，micro-batch 顺序流过。

    与 ``Trainer`` / ``DataParallelTrainer`` 的关系：三者不叠加。本类不做
    设备态捕获、混合精度与梯度累积，只用后端最基本的 ``execute`` /
    ``to_array`` / ``to_host`` / ``set_device``。
    """

    def __init__(
        self,
        stages: Sequence[PipelineStage],
        store: ParameterStore,
        optimizer: Optimizer,
        *,
        backend: Any,
        devices: Sequence[int],
        reduction: str = "mean",
        schedule: str = "sequential",
    ) -> None:
        if not isinstance(store, ParameterStore):
            raise TypeError(f"store 必须是 ParameterStore，得到 {type(store).__name__}")
        if not isinstance(optimizer, Optimizer):
            raise TypeError(
                f"optimizer 必须是 Optimizer，得到 {type(optimizer).__name__}"
            )
        _check_multi_device_backend(backend)
        if reduction not in _REDUCTIONS:
            raise ValueError(f"reduction 必须是 {_REDUCTIONS} 之一，收到 {reduction!r}")
        if schedule not in _SCHEDULES:
            raise ValueError(
                f"schedule 必须是 {_SCHEDULES} 之一，收到 {schedule!r}；"
                "sequential 逐份走完全部段再逆序反向，gpipe 先全部前向后全部反向，"
                "1f1b 在稳态里交替反向与下一份前向"
            )
        if isinstance(stages, (str, bytes)) or not isinstance(stages, Sequence):
            raise TypeError(f"stages 必须是 PipelineStage 序列，得到 {type(stages).__name__}")
        if not stages:
            raise ValueError("stages 不能为空：流水至少需要一段")
        for index, stage in enumerate(stages):
            if not isinstance(stage, PipelineStage):
                raise TypeError(
                    f"第 {index} 段的类型是 {type(stage).__name__}，不是 PipelineStage"
                )
            if not isinstance(stage.graph, Graph):
                raise TypeError(f"第 {index} 段的 graph 不是 Graph")
            if not stage.params:
                raise ValueError(f"第 {index} 段没有参数：空段没有存在意义")

        if isinstance(devices, (str, bytes)) or not isinstance(devices, Sequence):
            raise TypeError(f"devices 必须是设备索引序列，得到 {type(devices).__name__}")
        picked: List[int] = []
        for index in devices:
            value = _check_device_index(backend, index, label="设备索引")
            if value in picked:
                raise ValueError(f"devices 里有重复的设备索引 {value}")
            picked.append(value)
        if len(picked) != len(stages):
            raise ValueError(
                f"设备数 {len(picked)} 与段数 {len(stages)} 不一致；"
                "每段必须独占一块卡（一段多卡不在这一层的表达范围内）"
            )

        self._stages = tuple(stages)
        self.store = store
        self.optimizer = optimizer
        self.backend = backend
        self.devices = tuple(picked)
        self.reduction = reduction
        self.schedule = schedule
        self.name = f"pipeline_{len(stages)}"
        self.steps = 0
        # 最近一步里"前向已完成、反向还没做"的 micro-batch 峰值数。这是
        # 1F1B 相对 GPipe 唯一的结构性收益：段间激活不随 micro-batch 数线性堆积。
        self.last_resident_peak = 0

        self._check_stages()
        self._build_backward()
        self._params_on: Dict[int, Dict[str, Any]] = {}

    # -- 校验 ---------------------------------------------------------

    def _check_stages(self) -> None:
        """段间接口、参数归属、静态形状三件事一次查完。"""
        seen: Dict[str, int] = {}
        previous: Optional[PipelineStage] = None
        for index, stage in enumerate(self._stages):
            graph = stage.graph
            inputs = {v.name: v for v in graph.inputs}

            for name in stage.params:
                if name not in inputs:
                    raise ValueError(
                        f"第 {index} 段的参数 {name!r} 不是该图的输入"
                    )
                if name in seen:
                    raise ValueError(
                        f"参数 {name!r} 同时出现在第 {seen[name]} 段与第 {index} 段；"
                        "流水各段的参数不能重叠"
                    )
                seen[name] = index
                if not inputs[name].type.is_static:
                    raise ValueError(
                        f"第 {index} 段的参数 {name!r} 含符号维 {inputs[name].type.shape}；"
                        "训练图必须是静态形状"
                    )

            if previous is None:
                if stage.inputs:
                    raise ValueError(
                        f"第 0 段不能有段间输入，得到 {stage.inputs}；"
                        "流水的入口只有数据"
                    )
            else:
                if len(stage.inputs) != 1:
                    raise ValueError(
                        f"第 {index} 段必须恰好有一个段间输入，得到 {stage.inputs}"
                    )
                incoming = stage.inputs[0]
                if incoming not in inputs:
                    raise ValueError(
                        f"第 {index} 段的段间输入 {incoming!r} 不是该图的输入"
                    )
                if previous.output is None or incoming != previous.output:
                    raise ValueError(
                        f"第 {index} 段的段间输入 {incoming!r} 与第 {index - 1} 段的"
                        f"输出 {previous.output!r} 不一致；段间靠同名张量相连"
                    )
                expected = previous.graph.outputs[
                    previous.graph.output_names.index(previous.output)
                ].type
                actual = inputs[incoming].type
                if expected != actual:
                    raise ValueError(
                        f"第 {index} 段的输入 {incoming!r} 类型是 {actual}，"
                        f"但第 {index - 1} 段的同名输出是 {expected}"
                    )

            if stage.output is None:
                raise ValueError(f"第 {index} 段没有指定 output")
            if stage.output not in graph.output_names:
                raise ValueError(
                    f"第 {index} 段的输出 {stage.output!r} 不在图输出里: "
                    f"{list(graph.output_names)}"
                )
            previous = stage

        last = self._stages[-1]
        target = last.graph.outputs[last.graph.output_names.index(last.output)]
        if tuple(target.type.shape) != ():
            raise ValueError(
                f"最后一段的输出 {last.output!r} 必须是标量损失，"
                f"得到形状 {target.type.shape}"
            )
        self._loss_dtype = target.type.dtype.np_dtype

        covered = {name for stage in self._stages for name in stage.params}
        missing = sorted(set(self.store.names) - covered)
        unknown = sorted(covered - set(self.store.names))
        if missing or unknown:
            raise ValueError(
                "流水的参数与 ParameterStore 不一致："
                f"store 里多 {missing}，流水里多 {unknown}"
            )

        # 数据输入：段图里既不是参数、也不是段间输入的输入。标签通常直接喂给
        # 最后一段，所以数据输入允许出现在任意段，但**同名张量在各段的形状
        # 必须一致**（否则同一个 batch 切出来的块喂不进去）。
        data: Dict[str, Tuple[int, ...]] = {}
        stage_data: List[set] = []
        for index, stage in enumerate(self._stages):
            declared = {v.name: v for v in stage.graph.inputs}
            names = set(declared) - set(stage.params) - set(stage.inputs)
            stage_data.append(names)
            for name in names:
                shape = tuple(int(d) for d in declared[name].type.shape)
                if name in data and data[name] != shape:
                    raise ValueError(
                        f"数据输入 {name!r} 在第 {index} 段的形状是 {shape}，"
                        f"与前面的 {data[name]} 不一致"
                    )
                data[name] = shape
        if not data:
            raise ValueError("流水没有任何数据输入：训练必须从某个 batch 开始")
        self._data = data
        self._stage_data = stage_data

        # 每段的图输入必须**恰好**是"参数 + 段间输入 + 数据输入"。
        # 多一个输入在构图时看不出来，执行时才会炸成"缺输入"，所以在这里查。
        for index, stage in enumerate(self._stages):
            declared = {v.name for v in stage.graph.inputs}
            expected = set(stage.params) | set(stage.inputs) | stage_data[index]
            if declared != expected:
                raise ValueError(
                    f"第 {index} 段的图输入是 {sorted(declared)}，但流水的接口只"
                    f"允许 {sorted(expected)}；段图里不能有额外的输入"
                )

    def _build_backward(self) -> None:
        """每段一张反向图：参数梯度 + 段间输入梯度。"""
        self._bwd: List[Graph] = []
        self._input_name: List[Optional[str]] = []
        for stage in self._stages:
            wrt = [stage.graph.find(n) for n in stage.params]
            if stage.inputs:
                wrt.append(stage.graph.find(stage.inputs[0]))
            self._bwd.append(
                backward(stage.graph, wrt=wrt, active=[stage.output])
            )
            self._input_name.append(stage.inputs[0] if stage.inputs else None)

    # -- 内部 ---------------------------------------------------------

    def _replicate(self) -> None:
        for index, stage in enumerate(self._stages):
            self.backend.set_device(self.devices[index])
            self._params_on[index] = {
                name: self.backend.to_array(self.store[name]) for name in stage.params
            }

    def _split(self, batch: Mapping[str, Any], micro_batches: int) -> int:
        if isinstance(micro_batches, bool) or not isinstance(micro_batches, int):
            raise TypeError(f"micro_batches 必须是 int，收到 {micro_batches!r}")
        if micro_batches < 1:
            raise ValueError(f"micro_batches 必须 >= 1，收到 {micro_batches}")
        if not isinstance(batch, Mapping):
            raise TypeError(f"batch 必须是名字到数组的映射，得到 {type(batch).__name__}")

        data = {name: np.asarray(value) for name, value in batch.items()}
        missing = sorted(set(self._data) - set(data))
        if missing:
            raise KeyError(f"这些图输入没有提供数据: {missing}")
        unknown = sorted(set(data) - set(self._data))
        if unknown:
            raise KeyError(f"这些名字不是数据输入: {unknown}")
        for name, value in data.items():
            if value.ndim == 0:
                raise ValueError(f"数据输入 {name!r} 至少要有 1 维（按第 0 维切分）")
            if value.shape[1:] != self._data[name][1:]:
                raise ValueError(
                    f"数据输入 {name!r} 的形状 {value.shape[1:]} 与图声明的 "
                    f"{self._data[name][1:]} 不一致"
                )
        sizes = {name: int(value.shape[0]) for name, value in data.items()}
        if len(set(sizes.values())) != 1:
            raise ValueError(f"数据输入的第 0 维必须一致，得到 {sizes}")
        total = next(iter(sizes.values()))
        if total % micro_batches:
            raise ValueError(
                f"batch {total} 不能被 micro_batches {micro_batches} 整除；"
                "本层只支持等分（不等分需要按样本数加权，静态图表达不了）"
            )
        chunk = total // micro_batches
        for name, declared in self._data.items():
            if declared[0] != chunk:
                raise ValueError(
                    f"数据输入 {name!r} 每个 micro-batch 的第 0 维是 {chunk}，"
                    f"但图的静态形状是 {declared}；流水图要按 micro-batch 构图"
                )
        return chunk

    # -- 单步 ---------------------------------------------------------

    def step(self, batch: Mapping[str, Any], micro_batches: int = 1) -> float:
        """跑一步流水，返回各 micro-batch 损失的均值。"""
        chunk = self._split(batch, micro_batches)
        data = {name: np.asarray(value) for name, value in batch.items()}
        origin = self.backend.current_device()
        if not self._params_on:
            self._replicate()

        totals: Dict[str, np.ndarray] = {
            name: np.zeros_like(self.store[name]) for name in self.store.names
        }
        losses: List[float] = []
        # 种子权重：把"每份样本均值"的梯度换成"整批均值 / 整批和"的梯度。
        # micro-batch 是等分的，所以均值口径下每份的权重就是 1 / 份数。
        seed = (
            1.0 / micro_batches if self.reduction == "mean" else float(chunk)
        )
        resident = _ResidentTracker()
        try:
            if self.schedule == "sequential":
                for mb in range(micro_batches):
                    state = self._forward_micro(mb, chunk, data, resident)
                    losses.append(self._backward_micro(state, totals, seed, resident))
            elif self.schedule == "gpipe":
                states = [
                    self._forward_micro(mb, chunk, data, resident)
                    for mb in range(micro_batches)
                ]
                for state in states:
                    losses.append(self._backward_micro(state, totals, seed, resident))
            else:
                losses.extend(
                    self._run_1f1b(chunk, data, micro_batches, totals, seed, resident)
                )
            # 种子已经带上了权重，这里不再做第二次缩放。
            self.optimizer.step(totals)
            self._replicate()
        finally:
            self.backend.set_device(origin)

        self.last_resident_peak = resident.peak
        self.steps += 1
        return float(np.mean(losses))

    def _run_1f1b(
        self,
        chunk: int,
        data: Mapping[str, np.ndarray],
        micro_batches: int,
        totals: Dict[str, np.ndarray],
        seed: float,
        resident: "_ResidentTracker",
    ) -> List[float]:
        """1F1B：预热 ``min(段数-1, 份数)`` 份，之后每反向一份就前向下一份。

        单进程串行切卡，"前向的 micro-batch 1 与反向的 micro-batch 0 **真正
        并发**"做不到——这里交换的是**顺序**，不是线程。收益是可验证的两条：

        - 数值与 GPipe 在同一容差内一致（顺序不改变归约结果，只是改变了
          "哪一份的中间值还在内存里"）；
        - 同时驻留的段间激活从"份数"降到 ``min(段数-1, 份数)``，
          用 ``last_resident_peak`` 记账，不靠估算。
        """
        stages = len(self._stages)
        losses: List[float] = []
        if stages == 1:
            # 只有一段时没有段间交替，1F1B 退化成逐份前向 + 反向
            for mb in range(micro_batches):
                state = self._forward_micro(mb, chunk, data, resident)
                losses.append(self._backward_micro(state, totals, seed, resident))
            return losses
        warmup = min(stages - 1, micro_batches)
        waiting: Dict[int, dict] = {}
        for mb in range(warmup):
            waiting[mb] = self._forward_micro(mb, chunk, data, resident)

        for mb in range(micro_batches):
            state = waiting.pop(mb)
            losses.append(self._backward_micro(state, totals, seed, resident))
            nxt = mb + warmup
            if nxt < micro_batches:
                waiting[nxt] = self._forward_micro(nxt, chunk, data, resident)
        return losses

    def _forward_micro(
        self,
        index: int,
        chunk: int,
        data: Mapping[str, np.ndarray],
        resident: "_ResidentTracker",
    ) -> dict:
        """一个 micro-batch 走完全部段的前向，返回留给反向的状态。

        段间激活留在**产生它的那张卡**上，反向到该段时直接用，不再从主机
        重新上传——否则每段要两头往返一趟，而这段激活本来就是这张卡算出来的。
        """
        start = index * chunk
        piece = {
            name: self.backend.to_array(value[start : start + chunk])
            for name, value in data.items()
        }
        count = len(self._stages)
        activations: List[Any] = [None] * count
        loss = 0.0
        for stage_index, stage in enumerate(self._stages):
            self.backend.set_device(self.devices[stage_index])
            if stage_index == 0:
                feed = dict(self._params_on[0])
            else:
                feed = dict(self._params_on[stage_index])
                feed[stage.inputs[0]] = copy_across_devices(
                    self.backend,
                    activations[stage_index - 1],
                    self.devices[stage_index],
                    src_device=self.devices[stage_index - 1],
                )
            feed.update({name: piece[name] for name in self._stage_data[stage_index]})
            out = self.backend.execute(stage.graph, feed)[stage.output]
            if stage_index == count - 1:
                loss = float(self.backend.to_host(out))
            else:
                activations[stage_index] = out
        state = {"index": index, "piece": piece, "activations": activations, "loss": loss}
        resident.add(state)
        return state

    def _backward_micro(
        self,
        state: dict,
        totals: Dict[str, np.ndarray],
        seed: float,
        resident: "_ResidentTracker",
    ) -> float:
        """逆序反向，dout 沿段间边界传回去；结束后立刻释放该份的激活。"""
        activations = state["activations"]
        piece = state["piece"]
        count = len(self._stages)
        dout: Any = None
        for stage_index in reversed(range(count)):
            stage = self._stages[stage_index]
            self.backend.set_device(self.devices[stage_index])
            feed = dict(self._params_on[stage_index])
            if stage_index:
                feed[stage.inputs[0]] = activations[stage_index - 1]
            feed.update({name: piece[name] for name in self._stage_data[stage_index]})
            if stage_index == count - 1:
                # 标量损失的种子：等分权重直接写进种子，后面不再缩放梯度。
                feed["d" + stage.output] = np.asarray(
                    seed, dtype=self._loss_dtype
                )
            else:
                feed["d" + stage.output] = copy_across_devices(
                    self.backend,
                    dout,
                    self.devices[stage_index],
                    src_device=self.devices[stage_index + 1],
                )
            result = self.backend.execute(self._bwd[stage_index], feed)
            for name in stage.params:
                totals[name] += self.backend.to_host(result["d" + name])
            if stage_index > 0:
                dout = result["d" + stage.inputs[0]]
        # 这一份的段间激活到这里就没人再读了：马上放掉，峰值才有意义。
        state["activations"] = []
        resident.remove(state)
        return state["loss"]

    def __repr__(self) -> str:
        return (
            f"PipelineTrainer(stages={len(self._stages)}, devices={self.devices}, "
            f"params={len(self.store)}, reduction={self.reduction!r}, "
            f"schedule={self.schedule!r})"
        )


class _ResidentTracker:
    """记"前向已完成、反向还没做"的 micro-batch 数。

    不是显存字节数：同一段激活的形状由图的静态形状决定，而**有几份同时
    活着**才是调度决定的量。GPipe 的峰值等于份数，1F1B 的上限是段数减一。
    """

    __slots__ = ("live", "peak")

    def __init__(self) -> None:
        self.live = 0
        self.peak = 0

    def add(self, state: dict) -> None:
        self.live += 1
        if self.live > self.peak:
            self.peak = self.live

    def remove(self, state: dict) -> None:
        self.live -= 1
