"""多卡：跨卡拷贝与数据并行的最小闭环。

这一层**不新增执行机制**，也不新增内核。它只回答两个此前空缺的问题：

1. **一块卡上的数组怎么到另一块卡上。** 后端契约（``DeviceManagerContract``）
   只有枚举、查询与切换，没有拷贝——5.169 明确把"跨卡复用"留给调用方。
   这里给出唯一的入口 :func:`copy_across_devices`：默认经主机中转
   （D2H → H2D），对 CUDA / HIP 的语义完全一致；不提供隐式的 peer-to-peer，
   因为那要求两块卡可互访、且失败时的回退会掩盖"哪一步降级了"。
2. **一次前向反向怎么摊到多块卡上。** :class:`DataParallelTrainer` 把一批
   数据按第 0 维切成 ``world_size`` 份，每块卡上放一份**独立的参数副本**，
   各跑一次前向与反向，梯度回到主机做求和 / 平均，再由优化器更新主机侧的
   ``ParameterStore``，最后把新参数写回每块卡的副本。

**这不是性能并行。** 同一个进程里依次切卡执行，收益只有"这批数据能用上多块
卡的显存与算力总量"，没有并发重叠；真正的并行需要多进程 / 多流，属于后续
工作。之所以先做串行闭环，是因为它把"多卡训练是否正确"与"通信是否高效"
拆开了：数值对了，再谈重叠。

**集合通信到哪一级，写在声明里。** :func:`collective_support` 返回
``CollectiveSupport``：参考 / CUDA / HIP 后端都没有接入设备侧归约
（NCCL / RCCL），跨卡归约一律经主机中转；``overlap`` 恒为 False。请求
``DataParallelTrainer(collective="device")`` 而进程里没有实现时，抛
``DeviceCollectiveUnavailable`` 并带上"为什么不行"——不静默降级回主机路径，
否则调用方会以为通信留在设备上。想要设备侧归约，注入 ``device_collective``
（契约见 ``DeviceAllReduce``）：归约留在设备上，只把结果拉回主机一次，
D2H 的字节数从"每卡一份整梯度"降到"一份整梯度"。

**代价如实写在接口上。** ``set_device`` 是丢弃而不是迁移旧卡缓存（5.169），
所以每切一次卡，旧卡上的编译产物与捕获计划都会作废。这个循环因此不适合
小模型；它适合"一步的计算量远大于重新编译 / 重新上传"的规模。参数副本也
是每步整份重传，不做原地增量写——正确性优先，等真机数据再谈优化。

**损失必须是按样本求均值。** 数据并行在主机侧只做"逐卡损失取平均"与
"逐卡梯度取平均"，这两步与"合并 batch 跑一次"逐位等价的前提是：每块卡拿到
的样本数相同，且图内损失对 batch 求的是均值（``mse`` / ``softmax_cross_entropy``
都是）。batch 不能被 ``world_size`` 整除时显式报错，不丢尾样本。

**图按每卡 batch 构图。** 执行器只认静态形状，所以 ``graph`` 的第 0 维是
"分片后的大小"，总 batch 才是 ``world_size`` 倍。例如两块卡、图声明
``x[B, K]`` 时，``step`` 要收到 ``2B`` 条样本；分片后正好 ``B`` 条，与图的
静态形状对上。写成"图声明总 batch、运行期再切"不可能：静态图不允许执行期
改形状。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..backends.contracts import DeviceManagerContract
from ..ir.grad import backward
from ..ir.graph import Graph
from .optim import Optimizer
from .params import ParameterStore
from .sharded import ShardedOptimizer

__all__ = [
    "copy_across_devices",
    "allreduce",
    "allreduce_grads",
    "collective_support",
    "CollectiveSupport",
    "DeviceAllReduce",
    "DeviceCollectiveUnavailable",
    "DataParallelTrainer",
]

_REDUCTIONS = ("mean", "sum")


#: 设备侧归约契约：``provider(per_device, target_device, reduction=...)``。
#: ``per_device`` 是"设备号 → {参数名: **留在设备上的**数组}"；返回值必须
#: 恰好覆盖同一组名字，且每个数组都落在 ``target_device`` 上。调用方只把
#: 归约结果拉回主机一次（而不是每卡各拉一次），通信量的差额就在这里。
DeviceAllReduce = Callable[..., Mapping[str, Any]]


class DeviceCollectiveUnavailable(NotImplementedError):
    """请求了设备侧集合通信，但当前进程没有可用实现。

    刻意继承 ``NotImplementedError``：缺的是**能力**，不是参数写错。调用方
    该做的是退回主机路径（显式改成 ``collective="host"``）或接入 NCCL / RCCL，
    而不是去改参数重试。
    """


@dataclass(frozen=True)
class CollectiveSupport:
    """后端在集合通信上的能力声明。

    ``device_allreduce`` 为 False 时，多卡路径只能走主机中转归约；``overlap``
    目前恒为 False——执行器是同步的、切卡是串行的，进程内没有可重叠的窗口。
    ``reason`` 是给人读的"为什么不行"，空字符串只出现在全部能力都具备时。
    """

    device_allreduce: bool
    overlap: bool
    provider: Optional[str]
    reason: str


def collective_support(
    backend: Any, *, provider: Optional[DeviceAllReduce] = None
) -> CollectiveSupport:
    """查询后端能提供到哪一级集合通信。

    参考后端与 CUDA / HIP 后端都没有接入设备侧集合通信（NCCL / RCCL）：
    跨卡归约一律经主机中转。把这件事变成**可查询的声明**，而不是让调用方
    从性能曲线里猜。注入 ``provider`` 时声明随之变化——这条路径由调用方
    保证实现，本层只负责把契约校验住。
    """
    _check_multi_device_backend(backend)
    if provider is not None:
        if not callable(provider):
            raise TypeError(
                f"provider 必须是可调用对象，得到 {type(provider).__name__}"
            )
        name = getattr(provider, "name", type(provider).__name__)
        return CollectiveSupport(
            device_allreduce=True,
            overlap=False,
            provider=str(name),
            reason=(
                "设备侧归约由注入的 provider 提供；计算/通信重叠仍未实现："
                "执行器同步、切卡串行，进程内没有重叠窗口"
            ),
        )
    return CollectiveSupport(
        device_allreduce=False,
        overlap=False,
        provider=None,
        reason=(
            "设备侧集合通信（NCCL / RCCL）未接入：跨卡归约一律经主机中转"
            "（D2H → 主机求和 → H2D）；计算/通信重叠也未实现"
        ),
    )


def _check_multi_device_backend(backend: Any) -> None:
    """多卡路径要求后端实现设备管理契约与三个数组方法。

    分开检查而不是只靠 ``isinstance``：契约只查方法存在性，缺 ``to_array``
    的实现在这里才会以 ``AttributeError`` 的形式炸在循环深处。
    """
    if not isinstance(backend, DeviceManagerContract):
        raise TypeError(
            "多卡路径需要实现 DeviceManagerContract 的后端"
            "（enumerate_devices / current_device / set_device），"
            f"{type(backend).__name__} 没有提供"
        )
    for name in ("execute", "to_array", "to_host"):
        if not callable(getattr(backend, name, None)):
            raise TypeError(
                f"多卡路径需要后端提供 {name}()，{type(backend).__name__} 没有"
            )


def _check_device_index(backend: Any, index: Any, *, label: str) -> int:
    """校验设备索引：必须是 int、不越界，且当场报错。"""
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError(f"{label} 必须是 int，收到 {index!r}")
    count = len(backend.enumerate_devices())
    if not 0 <= index < count:
        raise ValueError(
            f"{label} {index} 越界：进程可见 {count} 块设备（0..{count - 1}）"
        )
    return index


def copy_across_devices(
    backend: Any,
    value: Any,
    dst_device: int,
    *,
    src_device: Any = None,
) -> Any:
    """把设备数组拷到另一块卡，返回目标卡上的新数组。

    ``src_device`` 缺省取 ``backend.current_device()``；显式给出时，拷贝前会
    先切到那块卡再读。为什么要这个参数：HIP 的 ``DeviceArray`` 不记录自己
    属于哪块卡，而 D2H 走的是"当前设备"的上下文——不显式切过去，读到的
    可能是另一块卡的地址。CUDA 的 cupy 数组自带 device id，但两条路径用
    同一份显式语义，测试才有可比性。

    中转经主机（D2H → H2D）：跨厂商通用，不要求两块卡可互访。代价是一次
    主机往返；大张量的跨卡拷贝因此不适合高频调用——这正是"默认不做隐式
    peer-to-peer"的原因。目标卡与源卡相同时原样返回，不做无意义的往返。
    """
    _check_multi_device_backend(backend)
    dst = _check_device_index(backend, dst_device, label="目标设备索引")
    source = (
        backend.current_device()
        if src_device is None
        else _check_device_index(backend, src_device, label="源设备索引")
    )
    if source == dst:
        return value

    origin = backend.current_device()
    try:
        if source != origin:
            backend.set_device(source)
        host = backend.to_host(value)
        backend.set_device(dst)
        return backend.to_array(host)
    finally:
        backend.set_device(origin)


def allreduce(
    values: Sequence[Any],
    *,
    reduction: str = "mean",
) -> np.ndarray:
    """主机侧归约：把各卡的同一个数组求和或取平均。

    只接受浮点数组，且要求形状与 dtype 完全一致。不做隐式类型提升——两张
    卡的梯度精度不同本身就是配置错误，静默升成 f64 只会让"某块卡走了别的
    路径"更难被发现。返回的是新数组，不改动传入值。
    """
    if reduction not in _REDUCTIONS:
        raise ValueError(
            f"reduction 必须是 {_REDUCTIONS} 之一，收到 {reduction!r}"
        )
    arrays = [np.asarray(value) for value in values]
    if not arrays:
        raise ValueError("allreduce 至少需要一个数组")
    reference = arrays[0]
    if not np.issubdtype(reference.dtype, np.floating):
        raise TypeError(
            f"allreduce 只接受浮点数组，收到 {reference.dtype}"
            "（整数取平均会被静默截断）"
        )
    for index, array in enumerate(arrays[1:], 1):
        if array.shape != reference.shape:
            raise ValueError(
                f"第 {index} 个数组的形状是 {array.shape}，"
                f"与第 0 个的 {reference.shape} 不一致"
            )
        if array.dtype != reference.dtype:
            raise TypeError(
                f"第 {index} 个数组的 dtype 是 {array.dtype}，"
                f"与第 0 个的 {reference.dtype} 不一致"
            )
    total = np.array(reference, copy=True)
    for array in arrays[1:]:
        total += array
    if reduction == "mean":
        total /= len(arrays)
    return total


def allreduce_grads(
    per_device: Mapping[int, Mapping[str, Any]],
    *,
    reduction: str = "mean",
) -> Dict[str, np.ndarray]:
    """按名字归约各卡梯度。

    每块卡必须给出**同一组**名字：少一个说明某张卡的反向图没跑全，多一个
    说明参数表已经分叉，两种都不能靠"取交集"掩盖过去。返回顺序沿用第一个
    设备给出的键顺序（与 ``store.names`` 一致），优化器不必再排序。
    """
    if not isinstance(per_device, Mapping) or not per_device:
        raise ValueError("per_device 不能为空：至少需要一块设备的梯度")
    items = list(per_device.items())
    for device, grads in items:
        if isinstance(device, bool) or not isinstance(device, int):
            raise TypeError(f"设备索引必须是 int，收到 {device!r}")
        if not isinstance(grads, Mapping):
            raise TypeError(
                f"设备 {device} 的梯度必须是名字到数组的映射，"
                f"得到 {type(grads).__name__}"
            )
    order = list(items[0][1])
    expected = set(order)
    for device, grads in items[1:]:
        if set(grads) != expected:
            raise KeyError(
                f"设备 {device} 的梯度名字与设备 {items[0][0]} 不一致："
                f"多 {sorted(set(grads) - expected)}，"
                f"少 {sorted(expected - set(grads))}"
            )
    return {
        name: allreduce([grads[name] for _, grads in items], reduction=reduction)
        for name in order
    }


class DataParallelTrainer:
    """数据并行的最小闭环：一份主机参数 + 每卡一份副本 + 主机侧梯度归约。

    ``devices`` 是参与计算的设备索引序列（不能重复，不能越界）。每步执行：

        for 每块卡: 切卡 → 上传本卡的数据切片 → 前向 → 反向 → 梯度回主机
        主机侧 allreduce（默认逐卡取平均）
        optimizer.step(grads)           # 原地更新 store 里的主机参数
        把新参数写回每块卡的副本

    与 ``Trainer`` 的关系：后者负责单卡的主机态 / 设备态细节（捕获、分段重算、
    混合精度、梯度累积），本类只负责"把一步摊到多卡"。两者不叠加——需要
    设备态或混合精度时，仍应先用 ``Trainer`` 把单卡语义跑稳。
    """

    def __init__(
        self,
        graph: Graph,
        store: ParameterStore,
        optimizer: Optimizer,
        *,
        backend: Any,
        devices: Sequence[int],
        output: Optional[str] = None,
        reduction: str = "mean",
        collective: str = "host",
        device_collective: Optional[DeviceAllReduce] = None,
    ) -> None:
        if not isinstance(graph, Graph):
            raise TypeError(f"graph 必须是 Graph，得到 {type(graph).__name__}")
        if not isinstance(store, ParameterStore):
            raise TypeError(f"store 必须是 ParameterStore，得到 {type(store).__name__}")
        if not isinstance(optimizer, (Optimizer, ShardedOptimizer)):
            raise TypeError(
                f"optimizer 必须是 Optimizer 或 ShardedOptimizer，"
                f"得到 {type(optimizer).__name__}"
            )
        _check_multi_device_backend(backend)
        if reduction not in _REDUCTIONS:
            raise ValueError(
                f"reduction 必须是 {_REDUCTIONS} 之一，收到 {reduction!r}"
            )
        if collective not in ("host", "device"):
            raise ValueError(
                f"collective 必须是 'host' 或 'device'，收到 {collective!r}"
            )
        if device_collective is not None and not callable(device_collective):
            raise TypeError(
                f"device_collective 必须是可调用对象，"
                f"得到 {type(device_collective).__name__}"
            )
        if collective == "host" and device_collective is not None:
            raise ValueError(
                "注入了 device_collective 却选了 collective='host'："
                "两种归约路径只能选一条，不猜"
            )
        if collective == "device" and device_collective is None:
            support = collective_support(backend)
            raise DeviceCollectiveUnavailable(
                f"collective='device' 需要设备侧归约实现，当前没有：{support.reason}；"
                "请显式改用 collective='host'（经主机中转），或注入 device_collective"
            )

        if isinstance(devices, (str, bytes)) or not isinstance(devices, Sequence):
            raise TypeError(f"devices 必须是设备索引序列，得到 {type(devices).__name__}")
        picked: List[int] = []
        for index in devices:
            value = _check_device_index(backend, index, label="设备索引")
            if value in picked:
                raise ValueError(f"devices 里有重复的设备索引 {value}")
            picked.append(value)
        if not picked:
            raise ValueError("devices 不能为空：数据并行至少需要一块设备")

        symbolic = [v.name for v in graph.inputs if not v.type.is_static]
        if symbolic:
            raise ValueError(
                f"训练图必须用静态形状，这些输入含符号维: {symbolic}"
            )
        inputs = {value.name for value in graph.inputs}
        missing = [name for name in store.names if name not in inputs]
        if missing:
            raise ValueError(f"这些参数不是图的输入: {missing}")

        outputs = list(graph.output_names)
        if not outputs:
            raise ValueError("图没有输出，无法定义训练目标")
        chosen = outputs[0] if output is None else output
        if chosen not in outputs:
            raise ValueError(f"输出 {output!r} 不在图输出里: {outputs}")
        target = graph.outputs[outputs.index(chosen)]
        if tuple(target.type.shape) != ():
            raise ValueError(
                f"训练目标必须是标量，输出 {chosen!r} 的形状是 {target.type.shape}"
            )

        self.graph = graph
        self.store = store
        self.optimizer = optimizer
        self.backend = backend
        self.devices = tuple(picked)
        self.reduction = reduction
        self.collective = collective
        self.device_collective = device_collective
        self.output = chosen
        self.steps = 0

        self._loss_value = target
        self._seed_name = "d" + chosen
        self._loss_dtype = target.type.dtype.np_dtype
        self._param_names = frozenset(store.names)
        self._data_names = frozenset(inputs - self._param_names)
        # 每个数据输入的静态形状：分片后的第 0 维必须与它相等，否则执行期
        # 报的是"静态维不匹配"，看不出"图应该按每卡 batch 构图"这条前提。
        self._declared = {
            value.name: tuple(int(d) for d in value.type.shape)
            for value in graph.inputs
            if value.name in self._data_names
        }
        self._params = [graph.find(name) for name in store.names]
        # 只对损失求导：其余输出不参与反向，避免喂零种子污染梯度。
        self._bwd: Graph = backward(graph, wrt=self._params, active=[chosen])
        # 每块卡一份参数副本；首次 step 时建立，之后每步末尾整份刷新。
        self._params_on: Dict[int, Dict[str, Any]] = {}

    # -- 内部 ---------------------------------------------------------

    def _replicate(self) -> None:
        """把主机参数整份写到每块卡（丢弃旧副本）。"""
        for device in self.devices:
            self.backend.set_device(device)
            self._params_on[device] = {
                name: self.backend.to_array(self.store[name])
                for name in self.store.names
            }

    def _split(self, batch: Dict[str, np.ndarray]) -> int:
        """校验数据输入并返回每块卡的样本数。

        名字要严格对齐：少传、多传、或把参数名当数据传都当场报错——它们
        都会让"某块卡少喂了一路输入"延后到执行期才暴露。
        """
        missing = sorted(self._data_names - set(batch))
        if missing:
            raise KeyError(f"这些图输入没有提供数据: {missing}")
        unknown = sorted(set(batch) - self._data_names)
        if unknown:
            raise KeyError(f"这些名字不是数据输入: {unknown}")
        sizes = {}
        for name, value in batch.items():
            array = np.asarray(value)
            if array.ndim == 0:
                raise ValueError(f"数据输入 {name!r} 至少要有 1 维（按第 0 维切片）")
            sizes[name] = int(array.shape[0])
        if len(set(sizes.values())) != 1:
            raise ValueError(f"数据输入的第 0 维必须一致，得到 {sizes}")
        total = next(iter(sizes.values()))
        world = len(self.devices)
        if total % world:
            raise ValueError(
                f"batch {total} 不能被设备数 {world} 整除；"
                "数据并行不丢弃尾样本，请补齐或换 batch（DataLoader(drop_last=True)）"
            )
        chunk = total // world
        # 图是按**每卡** batch 构的（执行器只认静态形状），分片大小必须正好
        # 等于那个值。不提前说清楚的话，报错会是一句"静态维不匹配"。
        for name, declared in self._declared.items():
            if int(declared[0]) != chunk:
                raise ValueError(
                    f"数据输入 {name!r} 分片后的第 0 维是 {chunk}，"
                    f"但图的静态形状是 {declared}；"
                    "数据并行的图要按每卡 batch 构图"
                    "（总 batch = 设备数 × 图的静态 batch）"
                )
        return chunk

    # -- 单步 ---------------------------------------------------------

    def _host_grads(self, reduced: Any) -> Dict[str, np.ndarray]:
        """校验设备侧归约结果并转成主机数组。

        provider 的返回值是**外部输入**：名字少一个、多一个、形状或 dtype
        变了，都必须当场报错。否则"某块卡悄悄少归约了一个参数"要等到几十步
        之后才以"损失不降"的形式暴露。
        """
        if not isinstance(reduced, Mapping):
            raise TypeError(
                f"device_collective 必须返回名字到设备数组的映射，"
                f"得到 {type(reduced).__name__}"
            )
        missing = [name for name in self.store.names if name not in reduced]
        unknown = [name for name in reduced if name not in self._param_names]
        if missing or unknown:
            raise KeyError(
                f"device_collective 返回的名字与参数表不一致："
                f"多 {sorted(unknown)}，少 {sorted(missing)}"
            )
        host: Dict[str, np.ndarray] = {}
        for name in self.store.names:
            array = np.asarray(self.backend.to_host(reduced[name]))
            expected = tuple(self.store[name].shape)
            if array.shape != expected:
                raise ValueError(
                    f"device_collective 返回的 {name} 形状是 {array.shape}，"
                    f"参数形状是 {expected}"
                )
            if array.dtype != self.store[name].dtype:
                raise TypeError(
                    f"device_collective 返回的 {name} dtype 是 {array.dtype}，"
                    f"参数 dtype 是 {self.store[name].dtype}"
                )
            host[name] = array
        return host

    def step(self, batch: Mapping[str, Any]) -> float:
        """跑一步数据并行，返回各卡损失的均值。"""
        if not isinstance(batch, Mapping):
            raise TypeError(f"batch 必须是名字到数组的映射，得到 {type(batch).__name__}")
        data = {name: np.asarray(value) for name, value in batch.items()}
        chunk = self._split(data)
        # 先记下调用前的设备，再去做任何切卡（首次 step 的副本上传也会切卡）。
        origin = self.backend.current_device()
        if not self._params_on:
            self._replicate()

        losses: List[float] = []
        # 名字固定为 "设备号 → 名字 → 梯度"：主机路径存主机数组，设备路径存
        # 设备数组。两种路径的**采样与反向完全相同**，只在归约这一步分叉。
        grads_on: Dict[int, Dict[str, Any]] = {}
        try:
            for rank, device in enumerate(self.devices):
                self.backend.set_device(device)
                feed: Dict[str, Any] = dict(self._params_on[device])
                for name, array in data.items():
                    start = rank * chunk
                    feed[name] = self.backend.to_array(array[start : start + chunk])
                out = self.backend.execute(self.graph, feed)[self.output]
                losses.append(float(self.backend.to_host(out)))
                seed = np.asarray(1.0, dtype=self._loss_dtype)
                result = self.backend.execute(self._bwd, {**feed, self._seed_name: seed})
                if self.collective == "device":
                    grads_on[device] = {
                        name: result["d" + name] for name in self.store.names
                    }
                else:
                    grads_on[device] = {
                        name: self.backend.to_host(result["d" + name])
                        for name in self.store.names
                    }
            if self.collective == "device":
                # 归约留在设备侧，只把结果拉回主机一次——主机路径是每卡各拉
                # 一份整梯度再在主机上求和，D2H 的字节数是这里的 world_size 倍。
                self.backend.set_device(self.devices[0])
                reduced = self.device_collective(
                    grads_on, self.devices[0], reduction=self.reduction
                )
                self.optimizer.step(self._host_grads(reduced))
            else:
                self.optimizer.step(allreduce_grads(grads_on, reduction=self.reduction))
            # 优化器改的是主机参数；各卡副本必须整份跟上，否则下一步读的是旧权重。
            self._replicate()
        finally:
            self.backend.set_device(origin)

        self.steps += 1
        return float(np.mean(losses))

    def __repr__(self) -> str:
        return (
            f"DataParallelTrainer(devices={self.devices}, "
            f"params={len(self.store)}, reduction={self.reduction!r}, "
            f"collective={self.collective!r})"
        )
