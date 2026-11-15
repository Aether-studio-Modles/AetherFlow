"""跨进程 ZeRO-2：梯度也只保留本片，通信从一次 AllReduce 换成收发各一半。

5.229 的 ZeRO-1 把**优化器槽位**按参数名分片，但每步仍要 AllReduce 一整份
梯度、再把每个参数广播一遍。两处浪费：

1. 归约后的整份梯度在每个 rank 上都存在，可只有本片会被优化器用到；
2. 参数广播逐个参数发 W-1 次小消息，消息数随参数个数线性增长。

ZeRO-2 的做法是把"分片"从槽位推到梯度：

    ReduceScatter(整份梯度)  →  每个 rank 只收到自己那一段的求和结果
    本地 optimizer.step(本段) →  只更新本段参数
    AllGather(本段参数)       →  每个 rank 拼回完整参数，供下一步前向使用

**通信量。** 每步发送 ``总元素数``、接收 ``总元素数``（ReduceScatter 与
AllGather 各一半），总量与 ZeRO-1 的 AllReduce + Broadcast 相同；省下来的
是**常驻内存**：归约结果只有 ``1/W``，梯度峰值从 N 降到 N/W + N/W。

**为什么这里能按元素切、而 ZeRO-1 不能。** NCCL 要求 ReduceScatter 各段等长。
ZeRO-1 按参数名分片，各片大小不同，所以只能退回 AllReduce。ZeRO-2 仍按参数
**边界**分段（一个参数不会跨两段，否则优化器要处理半张量），但把每段**补零
到同一长度** ``pad``，通信缓冲长度统一为 ``pad × W``。代价是一点带宽，收益是
"参数边界"与"等长段"能同时成立。

**数值。** 归约顺序由 NCCL 决定，因此不承诺与 ZeRO-1 逐位相等；进程内的
``LoopbackCollective`` 按 rank 序号累加，离线用例可以逐位对拍。

**边界。** 段内参数仍按全局顺序排列，所以各 rank 用同一份参数表构造
（键的顺序也要一致）才能得到同一套分段；参数个数少于 ``world_size`` 时会有
空段，这是合法状态。参数本身仍是每个 rank 一份完整副本——那是 ZeRO-3 的
范围，这里不做参数驻留拆分。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from .optim import Optimizer
from .sharded import shard_params
from .zero import _load_shard, _same_array, _save_shard

__all__ = ["Zero2Rank", "plan_segments"]

#: 一段 = (段内参数名, 段内元素数)。段内名字按全局参数顺序排。
Segment = Tuple[List[str], int]


def plan_segments(
    params: Mapping[str, Any], world_size: int
) -> Tuple[List[Segment], int]:
    """把参数切成 ``world_size`` 段，并给出补零后的段长。

    分段复用 ``shard_params`` 的贪心（按元素数尽量均衡、结果只由参数顺序与
    形状决定）。段内名字按**全局顺序**重排，保证"同一份参数表在任何 rank 上
    都得到同一套分段"——这是 ReduceScatter 能对上的前提。

    返回 ``(segments, pad)``，``pad`` 是最大段长；通信缓冲长度一律取
    ``pad × world_size``，短段尾部补零。
    """
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise TypeError(f"world_size 必须是 int，收到 {world_size!r}")
    if world_size < 1:
        raise ValueError(f"world_size 必须 >= 1，收到 {world_size}")
    names = list(params)
    if not names:
        raise ValueError("参数表为空：没有可分片的梯度与优化器状态")

    sizes = {
        name: int(np.prod(np.shape(params[name]), dtype=np.int64))
        for name in names
    }
    buckets = shard_params(params, world_size)
    order = {name: index for index, name in enumerate(names)}
    segments: List[Segment] = []
    for bucket in buckets:
        inside = sorted(bucket, key=order.__getitem__)
        segments.append((inside, sum(sizes[name] for name in inside)))
    pad = max((size for _, size in segments), default=0)
    if pad <= 0:
        raise RuntimeError("分段长度为 0：参数表里没有元素")
    return segments, pad


class Zero2Rank:
    """一个 rank 上的 ZeRO-2 训练步骤。

    构造参数与 :class:`aetherflow.train.zero.ZeroRank` 相同：``params`` 是完整
    参数表（前向要用），``factory`` 只接收**本段**的参数子表并返回优化器。
    校验项也一致（同一块内存、名字集合、单一 dtype），另加一条：每个 rank 的
    分段必须由同一份参数顺序推出——这里不做跨进程核对，只在文档与用例里
    固定"同一模型定义顺序"这个前提。
    """

    def __init__(
        self,
        params: MutableMapping[str, np.ndarray],
        factory: Callable[[MutableMapping[str, np.ndarray]], Optimizer],
        *,
        rank: int,
        world_size: int,
        collective: Optional[Any] = None,
    ) -> None:
        if not isinstance(params, MutableMapping):
            raise TypeError(
                f"params 必须是可写映射（优化器要原地更新），"
                f"得到 {type(params).__name__}"
            )
        if not callable(factory):
            raise TypeError(f"factory 必须可调用，收到 {type(factory).__name__}")
        if isinstance(world_size, bool) or not isinstance(world_size, int):
            raise TypeError(f"world_size 必须是 int，收到 {world_size!r}")
        if world_size < 1:
            raise ValueError(f"world_size 必须 >= 1，收到 {world_size}")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError(f"rank 必须是 int，收到 {rank!r}")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} 越界：world_size={world_size}")

        self._params = params
        self._rank = rank
        self._world_size = world_size
        self._order = list(params)
        self._segments, self._pad = plan_segments(params, world_size)
        self._my_names = list(self._segments[rank][0])

        dtypes = {np.asarray(params[name]).dtype for name in self._order}
        if len(dtypes) != 1:
            raise TypeError(
                f"跨进程 ZeRO-2 要求全部参数同一 dtype（梯度与参数都要打包成"
                f"等长段），得到 {sorted(str(d) for d in dtypes)}；"
                "混合精度请先统一到主权重精度"
            )
        self._dtype = next(iter(dtypes))

        sub = {name: params[name] for name in self._my_names}
        shard = factory(sub)
        if not isinstance(shard, Optimizer):
            raise TypeError(f"factory 返回了 {type(shard).__name__}，不是 Optimizer")
        if set(shard.params) != set(sub):
            raise KeyError(
                f"本段优化器的参数名与分段不一致："
                f"多 {sorted(set(shard.params) - set(sub))}，"
                f"少 {sorted(set(sub) - set(shard.params))}"
            )
        for name, array in shard.params.items():
            if not _same_array(array, params[name]):
                raise ValueError(
                    f"本段优化器里的 {name} 不是参数表里的同一个数组；"
                    "factory 复制了子表，原地更新不会落到参数上"
                )
        self._optimizer = shard
        self._collective: Optional[Any] = None
        if collective is not None:
            self.attach(collective)

    # -- 通信面 -------------------------------------------------------

    def attach(self, collective: Any) -> None:
        """接上通信后端；需要 ``reduce_scatter`` 与 ``all_gather``。"""
        for name in ("reduce_scatter", "all_gather"):
            if not callable(getattr(collective, name, None)):
                raise TypeError(
                    f"ZeRO-2 的 collective 需要 {name}()，"
                    f"{type(collective).__name__} 没有"
                )
        rank = getattr(collective, "rank", self._rank)
        world = getattr(collective, "world_size", self._world_size)
        if int(rank) != self._rank:
            raise ValueError(
                f"collective 的 rank {rank} 与 Zero2Rank 的 {self._rank} 不一致"
            )
        if int(world) != self._world_size:
            raise ValueError(
                f"collective 的 world_size {world} 与 Zero2Rank 的 "
                f"{self._world_size} 不一致"
            )
        self._collective = collective

    # -- 只读面 -------------------------------------------------------

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def segments(self) -> Sequence[Segment]:
        return tuple((list(names), size) for names, size in self._segments)

    @property
    def assignments(self) -> Mapping[str, int]:
        owners: Dict[str, int] = {}
        for index, (names, _) in enumerate(self._segments):
            for name in names:
                owners[name] = index
        return owners

    @property
    def my_names(self) -> Sequence[str]:
        return tuple(self._my_names)

    @property
    def pad(self) -> int:
        """补零后的段长（通信缓冲长度 = ``pad × world_size``）。"""
        return self._pad

    @property
    def steps(self) -> int:
        """已经走过的步数。唯一真相是优化器的 ``step_count``：另存一份计数器
        会在 ``load_shard`` 之后与槽位不一致。"""
        return int(self._optimizer.step_count)

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @property
    def params(self) -> MutableMapping[str, np.ndarray]:
        return self._params

    def owned_bytes(self) -> int:
        """本进程常驻的优化器槽位字节数（参数仍是完整一份）。"""
        slots = 0
        for arrays in self._optimizer._slots().values():
            for array in arrays.values():
                slots += int(np.asarray(array).nbytes)
        return slots

    def communication_elements(self) -> int:
        """每步收发各多少元素：ReduceScatter 与 AllGather 各 ``pad × W``。"""
        return self._pad * self._world_size

    # -- 打包 / 拆包 --------------------------------------------------

    def _pack(self, arrays: Mapping[str, Any]) -> np.ndarray:
        """按段序把各参数拼成一条 ``pad × W`` 的缓冲，短段尾部补零。"""
        buffer = np.zeros(self._pad * self._world_size, dtype=self._dtype)
        base = 0
        for names, _ in self._segments:
            position = base
            for name in names:
                piece = np.ascontiguousarray(np.asarray(arrays[name])).ravel()
                buffer[position : position + piece.size] = piece
                position += piece.size
            base += self._pad
        return buffer

    def _pack_own(self) -> np.ndarray:
        """只填本段，其余留零——AllGather 的发送侧只需要本段。"""
        buffer = np.zeros(self._pad, dtype=self._dtype)
        names, _ = self._segments[self._rank]
        position = 0
        for name in names:
            piece = np.ascontiguousarray(np.asarray(self._params[name])).ravel()
            buffer[position : position + piece.size] = piece
            position += piece.size
        return buffer

    def _unpack_own(self, received: np.ndarray) -> Dict[str, np.ndarray]:
        """把本段的归约结果切回参数形状（只保留本段参数）。"""
        result: Dict[str, np.ndarray] = {}
        names, _ = self._segments[self._rank]
        position = 0
        for name in names:
            shape = np.shape(self._params[name])
            size = int(np.prod(shape, dtype=np.int64))
            result[name] = received[position : position + size].reshape(shape)
            position += size
        return result

    def _write_back(self, gathered: np.ndarray) -> None:
        """把 AllGather 的结果按段写回完整参数表（原地，不换对象）。"""
        base = 0
        for names, _ in self._segments:
            position = base
            for name in names:
                array = np.asarray(self._params[name])
                size = int(array.size)
                array[...] = gathered[position : position + size].reshape(
                    array.shape
                )
                position += size
            base += self._pad

    def _checked_grads(self, grads: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        """校验梯度覆盖全部参数、形状与 dtype 一致（不补零、不跳过）。"""
        missing = [name for name in self._order if name not in grads]
        if missing:
            raise KeyError(
                f"这些参数没有梯度: {missing}。"
                "通常意味着损失不依赖它们；请检查构图"
            )
        unknown = [name for name in grads if name not in self._params]
        if unknown:
            raise KeyError(f"梯度里出现了不认识的参数: {unknown}")
        checked: Dict[str, np.ndarray] = {}
        for name in self._order:
            array = np.asarray(grads[name])
            param = self._params[name]
            if array.shape != np.shape(param):
                raise ValueError(
                    f"{name} 的梯度形状 {array.shape} 与参数形状 "
                    f"{np.shape(param)} 不一致"
                )
            if array.dtype != self._dtype:
                raise TypeError(
                    f"{name} 的梯度 dtype {array.dtype} 与参数的 {self._dtype} "
                    "不一致；打包成等长段要求同一 dtype"
                )
            checked[name] = array
        return checked

    # -- 单步 ---------------------------------------------------------

    def step(self, grads: Mapping[str, np.ndarray], *, reduction: str = "mean") -> None:
        """ReduceScatter 梯度 → 只更新本段 → AllGather 参数。"""
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction 必须是 mean / sum，收到 {reduction!r}")
        if self._collective is None:
            raise RuntimeError(
                "还没有接上集合通信：请先 attach(collective) 或构造时传入"
            )
        send = self._pack(self._checked_grads(grads))
        received = np.empty(self._pad, dtype=self._dtype)
        reduced = np.asarray(
            self._collective.reduce_scatter(send, received, op="sum")
        )
        if reduced.shape != (self._pad,):
            raise ValueError(
                f"collective.reduce_scatter 返回的形状 {reduced.shape} "
                f"与段的 {self._pad} 不一致"
            )
        if reduction == "mean":
            reduced = reduced / float(self._world_size)

        local = self._unpack_own(reduced)
        if local:
            self._optimizer.step(local)

        gathered = np.empty(
            self._pad * self._world_size, dtype=self._dtype
        )
        merged = np.asarray(
            self._collective.all_gather(self._pack_own(), gathered)
        )
        if merged.shape != gathered.shape:
            raise ValueError(
                f"collective.all_gather 返回的形状 {merged.shape} "
                f"与期望的 {gathered.shape} 不一致"
            )
        self._write_back(merged)

    # -- 存档 ---------------------------------------------------------

    def shard_state(self) -> Dict[str, Any]:
        """本 rank 的存档内容：步数 + 超参 + 本段槽位 + 归属表。"""
        state = self._optimizer.state_dict()
        return {
            "rank": self._rank,
            "world_size": self._world_size,
            "step": state["step"],
            "hyper": state["hyper"],
            "slots": state["slots"],
            "assignments": {
                name: owner
                for name, owner in self.assignments.items()
                if owner == self._rank
            },
        }

    def save_shard(self, path: Any) -> str:
        """只写本段的槽位；n 个进程各写一个文件，合起来是一份完整状态。"""
        return _save_shard(self.shard_state(), path)

    def load_shard(self, path: Any) -> None:
        """装回本段槽位；片号 / 分片数 / 归属表不符都是显式错误。"""
        _load_shard(
            path,
            rank=self._rank,
            world_size=self._world_size,
            owners=self.assignments,
            optimizer=self._optimizer,
        )

    def __repr__(self) -> str:
        return (
            f"Zero2Rank(rank={self._rank}/{self._world_size}, "
            f"pad={self._pad}, owned={self._my_names}, steps={self.steps})"
        )
