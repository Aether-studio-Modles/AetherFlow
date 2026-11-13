"""SNN 的稀疏表示与事件内存池。

脉冲是 0/1 张量，密度通常只有几个百分点，稠密存储意味着绝大部分读写都是零
——SNN 真正的性能来源就是"只处理发放的神经元"。这一模块提供两件基础设施，
都停在**数据侧**，不新增 IR 算子、不改变图语义：

1. ``SpikeEvents``：脉冲序列的 CSR 变体表示。``offsets[t]`` 是第 t 步的第一个
   事件在 ``indices`` 里的位置，``counts[t]`` 是该步事件数；两者合起来就能从
   扁平下标还原出 (步, 尾维下标) 对，不需要 T×N 的稠密张量。
2. ``EventPool``：复用 ``indices`` 缓冲的池，并给出分配 / 扩容 / 峰值统计。

**为什么不做成 IR 算子。** 压缩与还原对值是**无损的布局变换**，在自动微分里
对应恒等映射；做成算子只会让反向图多一条恒等边，不会让任何后端变快。真正的
收益在"用稀疏表示做计算"（稀疏 matmul / 事件驱动卷积），那是后端 kernel 的
事，当前没有实现——缺口写在 DESIGN 5.35。

**多后端。** 这里不 import cupy：数组属于哪个库由类型模块名判断，numpy 与
cupy 的公共 API（``reshape`` / ``nonzero`` / ``repeat`` / ``bincount``）足够
写出同一份实现，因此同一套代码既能在主机侧做数据预处理，也能直接跑在显存上。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

__all__ = [
    "SpikeEvents",
    "EventPool",
    "EventPoolStats",
    "dense_to_events",
    "events_to_dense",
    "event_mean_rate",
]


def _array_module(a):
    """取数组所属的数值库：numpy 或 cupy。

    按类型模块名判断而不是 ``isinstance``，是为了让 runtime 层不 import cupy
    ——cupy 是可选依赖，没装 GPU 的机器也应该能 import 这个模块。
    """
    if type(a).__module__.split(".")[0] == "cupy":
        import cupy  # 延迟导入：只在真的拿到加速数组时才需要它

        return cupy
    return np


def _prod(xs) -> int:
    n = 1
    for x in xs:
        n *= int(x)
    return n


def _check_spikes(spikes) -> Tuple[int, Tuple[int, ...], int]:
    """校验脉冲形状与取值，返回 ``(steps, tail_shape, size)``。

    取值只接受 0/1：事件表示丢弃的是"零"，如果输入混着别的数值
    （比如膜电位），压缩会静默改变语义。这是**系统边界**上的校验，
    宁可多一次 pass 也不做隐式解释。
    """
    shape = tuple(int(d) for d in spikes.shape)
    if len(shape) < 2:
        raise ValueError(
            f"脉冲张量至少要有 (T, 尾...) 两个维度，得到 {shape}；"
            "只有时间维时无法区分“哪一步”与“哪个元素”"
        )
    steps, tail = shape[0], shape[1:]
    size = _prod(tail)
    flat = spikes.reshape(steps, size)
    nonzero = flat != 0
    not_one = flat != 1
    if bool((nonzero & not_one).any()):
        raise ValueError("dense_to_events 只接受 0/1 脉冲，得到其它取值")
    return steps, tail, size


@dataclass(frozen=True)
class SpikeEvents:
    """脉冲序列的稀疏表示（事件的 CSR 变体）。

    ``offsets`` 与 ``counts`` 都是长度 T 的整数数组，第 t 步的事件是
    ``indices[offsets[t] : offsets[t] + counts[t]]``——尾维的下标，升序排列。
    这里刻意不存 ``offsets[T] = nnz`` 的哨兵：末尾总量由 ``nnz`` 表达，
    少一次 ``concatenate`` 就少一次对库函数的依赖。
    """

    steps: int
    shape: Tuple[int, ...]
    counts: Any
    offsets: Any
    indices: Any

    @property
    def size(self) -> int:
        return _prod(self.shape)

    @property
    def nnz(self) -> int:
        """事件总数。cupy 数组在这里会同步一次（只取一个标量）。"""
        total = self.counts.sum()
        # cupy 14 起禁止 ``np.asarray(cupy_array)`` 的隐式拷贝，必须显式 ``.get()``
        return int(total.get() if hasattr(total, "get") else total)

    @property
    def density(self) -> float:
        total = self.steps * self.size
        return self.nnz / total if total else 0.0

    @property
    def nbytes(self) -> int:
        return int(self.counts.nbytes + self.offsets.nbytes + self.indices.nbytes)

    def step_indices(self, t: int) -> Any:
        """第 t 步的事件下标（``indices`` 上的切片视图，不拷贝）。"""
        start = int(self.offsets[t])
        return self.indices[start : start + int(self.counts[t])]


def dense_to_events(spikes, pool: Optional["EventPool"] = None) -> SpikeEvents:
    """把 0/1 脉冲张量 ``(T, 尾...)`` 压成 ``SpikeEvents``。

    传入 ``pool`` 时事件下标会拷进池缓冲（见 ``EventPool.pack`` 的复用语义）。
    """
    steps, tail, size = _check_spikes(spikes)
    flat = spikes.reshape(steps, size)
    mask = flat != 0

    counts = mask.sum(axis=1).astype(np.int64)
    offsets = counts.cumsum() - counts
    # 尾维下标用 int32：一次前向的事件数远小于 2^31；尾维本身可能更大，
    # 那时退回 int64，避免溢出成一个静默的错值。
    index_dtype = np.int32 if size < 2**31 else np.int64
    _, cols = mask.nonzero()
    indices = cols.astype(index_dtype)

    if pool is not None:
        indices = pool._adopt(indices)
    return SpikeEvents(steps=steps, shape=tail, counts=counts, offsets=offsets, indices=indices)


def events_to_dense(events: SpikeEvents, dtype=np.float32):
    """把事件还原成稠密 0/1 张量 ``(T, 尾...)``。

    这是**验证与回退**路径（稀疏 kernel 还没实现时用它接常规算子），
    也是往返一致性的判据：它对每个事件写一次 1，不使用任何隐含的填充值。
    """
    xp = _array_module(events.indices)
    dense = xp.zeros((events.steps, events.size), dtype=dtype)
    if events.nnz:
        rows = xp.repeat(xp.arange(events.steps), events.counts)
        dense[rows, events.indices] = 1
    return dense.reshape((events.steps, *events.shape))


def event_mean_rate(events: SpikeEvents, dtype=np.float32):
    """只读事件、不物化稠密张量的平均发放率（``(T, 尾...) -> (尾...)``）。

    与 ``models.snn.mean_rate`` 的结果必须一致；区别在于它一次都不需要
    ``T×N`` 的稠密缓冲——这正是稀疏表示要换来的东西。
    """
    xp = _array_module(events.indices)
    if events.steps == 0 or events.size == 0:
        return xp.zeros(events.shape, dtype=dtype)
    total = xp.bincount(events.indices, minlength=events.size)
    rate = total.astype(dtype) / dtype(events.steps)
    return rate.reshape(events.shape)


@dataclass(frozen=True)
class EventPoolStats:
    capacity: int
    nnz: int
    peak_nnz: int
    allocations: int
    growths: int
    bytes: int


class EventPool:
    """复用事件下标缓冲的内存池。

    池只固定**一块**容量为 ``capacity`` 的下标缓冲：``pack`` 时把新算出的
    事件拷进去，容量不够按 2 倍扩容。因此

    * ``pack`` 返回的 ``SpikeEvents.indices`` 是池缓冲上的**切片视图**，
      下一次 ``pack`` 会覆盖它；需要跨调用保留就自己复制一份。
    * 单次 ``pack`` 仍会产出一个临时的 ``nonzero`` 结果（库不提供写入现有
      缓冲的接口）；池消除的是**跨时间步、跨前向**的反复分配，而不是这一次。

    这两条都是刻意的取舍：把"零分配"写进注释容易，做到却需要自己写
    scatter kernel，那是后端该做的事。
    """

    def __init__(self, capacity: int = 0):
        if int(capacity) < 0:
            raise ValueError(f"事件池容量不能为负，得到 {capacity}")
        self._capacity = int(capacity)
        self._buffer = None
        self._nnz = 0
        self._peak = 0
        self._allocations = 0
        self._growths = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def pack(self, spikes) -> SpikeEvents:
        """把脉冲压成事件，下标写进池缓冲。"""
        return dense_to_events(spikes, pool=self)

    def _adopt(self, indices):
        """把新算出的下标拷进池缓冲（必要时扩容），返回池内视图。"""
        xp = _array_module(indices)
        nnz = int(indices.shape[0])
        self._nnz = nnz
        self._peak = max(self._peak, nnz)

        # 库或下标宽度变了就必须重建：池里只留一块缓冲，不做多桶缓存。
        usable = (
            self._buffer is not None
            and type(self._buffer).__module__ == type(indices).__module__
            and self._buffer.dtype == indices.dtype
        )
        if not usable or nnz > self._capacity:
            grew = nnz > self._capacity
            target = max(nnz, 1) if not usable else max(nnz, max(1, self._capacity) * 2)
            self._buffer = xp.empty(int(target), dtype=indices.dtype)
            self._capacity = int(target)
            self._allocations += 1
            if grew:
                self._growths += 1
        if nnz:
            self._buffer[:nnz] = indices
        return self._buffer[:nnz]

    def stats(self) -> EventPoolStats:
        bytes_ = 0 if self._buffer is None else int(self._buffer.nbytes)
        return EventPoolStats(
            capacity=self._capacity,
            nnz=self._nnz,
            peak_nnz=self._peak,
            allocations=self._allocations,
            growths=self._growths,
            bytes=bytes_,
        )
