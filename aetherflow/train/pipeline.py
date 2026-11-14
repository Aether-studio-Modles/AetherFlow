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
        self.name = f"pipeline_{len(stages)}"
        self.steps = 0

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
        try:
            for mb in range(micro_batches):
                losses.append(self._micro_step(mb, chunk, data, totals, seed))
            # 种子已经带上了权重，这里不再做第二次缩放。
            self.optimizer.step(totals)
            self._replicate()
        finally:
            self.backend.set_device(origin)

        self.steps += 1
        return float(np.mean(losses))

    def _micro_step(
        self,
        index: int,
        chunk: int,
        data: Mapping[str, np.ndarray],
        totals: Dict[str, np.ndarray],
        seed: float,
    ) -> float:
        """一个 micro-batch 的前向 + 反向，梯度累加进 ``totals``。

        前向的段间激活留在**产生它的那张卡**上；反向到该段时直接用，
        不再从主机重新上传——否则每段要两头往返一趟，而这段激活本来就是
        这张卡算出来的。
        """
        start = index * chunk
        piece = {
            name: self.backend.to_array(value[start : start + chunk])
            for name, value in data.items()
        }

        # 1) 前向：``activations[i]`` 是第 i 段的输出，也就是第 i+1 段的输入。
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

        # 2) 反向：逆序，dout 沿段间边界传回去。
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
        return loss

    def __repr__(self) -> str:
        return (
            f"PipelineTrainer(stages={len(self._stages)}, devices={self.devices}, "
            f"params={len(self.store)}, reduction={self.reduction!r})"
        )
