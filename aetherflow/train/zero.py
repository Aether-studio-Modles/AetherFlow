"""跨进程 ZeRO-1：每个进程只持有自己那片的优化器槽位与参数更新权。

5.205 / 5.222 的分片优化器仍然把所有片都造在**同一个进程**里：内存账面上
"分片省了槽位"，但常驻的还是整份。本模块把"哪片归哪个进程"这件事真正
落地：

    rank 0: 前向/反向 → 整份梯度 → 设备侧 AllReduce
    rank 0: 只更新自己那片的参数（优化器槽位也只有这一片）
    rank 0: 把自己那片的**更新结果**广播给其余 rank，使各进程的参数一致

**通信量与数据并行相同。** ZeRO-1 省的是优化器状态的常驻内存，不是带宽：
梯度仍然要在全部 rank 上求和，参数更新后也要让每个 rank 拿到新值。因此这
里用一次整份梯度的 AllReduce（``sum``）而不是 ReduceScatter——后者要求各段
等长，而"按参数名分片"的各片大小不同（见 ``train/nccl.py`` 的说明）。要
减半带宽需要按元素切分，那会改变分片布局，属于另一层。

**参数更新只发生在本片。** 每个 rank 的 ``params`` 仍是完整的一份（前向要
用），但只有片主对自己那片执行 ``optimizer.step``；其余 rank 的那片参数
是旧值，随后由广播覆盖。这不是"每进程只存 1/W 的参数"（那是 ZeRO-3），
ZeRO-1 的边界就在这里。

**通信面是可注入的。** ``ZeroRank`` 只依赖 ``rank`` / ``world_size`` /
``all_reduce`` / ``broadcast`` 四个成员。真机上是 ``NcclHostCollective``
（H2D → NCCL → D2H）；单测里是 ``LoopbackCollective``，在进程内模拟世界，
用来验证"数值与不分片逐位相同"这条判据。把两层分开，是为了让"分片语义错
了"与"NCCL 调用错了"不会混在同一个失败里。
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from .optim import Optimizer
from .sharded import shard_params

__all__ = [
    "ZeroRank",
    "LoopbackWorld",
    "LoopbackCollective",
    "NcclHostCollective",
]

_META_KEY = "__meta__"


def _same_array(left: Any, right: Any) -> bool:
    """两个名字指向的是不是同一块内存（参数更新必须原地生效）。"""
    if left is right:
        return True
    try:
        return int(np.asarray(left).__array_interface__["data"][0]) == int(
            np.asarray(right).__array_interface__["data"][0]
        )
    except (AttributeError, KeyError, TypeError):
        return False


def _ordered_assignments(
    params: Mapping[str, Any], world_size: int
) -> Dict[str, int]:
    """参数名 → 片号；顺序沿用 ``shard_params`` 的确定性划分。"""
    names = list(params)
    buckets = shard_params(params, world_size)
    owners: Dict[str, int] = {}
    for index, shard in enumerate(buckets):
        for name in shard:
            owners[name] = index
    if set(owners) != set(names):
        raise RuntimeError("分片结果没有覆盖全部参数")
    return owners


class ZeroRank:
    """一个 rank 上的 ZeRO-1 训练步骤。

    ``params`` 是完整参数表（各进程一份，前向 / 反向都要用）；``factory``
    接收**本片**的参数子表并返回优化器。构造时逐项核对：

    - 子表里的数组必须与 ``params`` 里的是**同一块内存**——优化器原地更新
      子表，如果 factory 复制了一份，更新就不会落到参数上；
    - 优化器返回的参数名必须与本片完全一致，且是 ``Optimizer`` 实例。

    ``collective`` 可以构造后再 ``attach``：真机上 communicator 的初始化要
    等全部 rank 到齐，通常晚于参数表的构造。
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
        if not params:
            raise ValueError("参数表为空：没有可分片的优化器状态")
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
        self._owners = _ordered_assignments(params, world_size)
        self._my_names = [name for name in self._order if self._owners[name] == rank]

        dtypes = {np.asarray(params[name]).dtype for name in self._order}
        if len(dtypes) != 1:
            raise TypeError(
                f"跨进程 ZeRO 要求全部参数同一 dtype（梯度要打包成一条缓冲做 "
                f"AllReduce），得到 {sorted(str(d) for d in dtypes)}；"
                "混合精度请先统一到主权重精度"
            )
        self._dtype = next(iter(dtypes))

        sub = {name: params[name] for name in self._my_names}
        shard = factory(sub)
        if not isinstance(shard, Optimizer):
            raise TypeError(
                f"factory 返回了 {type(shard).__name__}，不是 Optimizer"
            )
        if set(shard.params) != set(sub):
            raise KeyError(
                f"本片优化器的参数名与分片不一致："
                f"多 {sorted(set(shard.params) - set(sub))}，"
                f"少 {sorted(set(sub) - set(shard.params))}"
            )
        for name, array in shard.params.items():
            if not _same_array(array, params[name]):
                raise ValueError(
                    f"本片优化器里的 {name} 不是参数表里的同一个数组；"
                    "factory 复制了子表，原地更新不会落到参数上"
                )
        self._optimizer = shard
        self._collective: Optional[Any] = None
        self.steps = 0
        if collective is not None:
            self.attach(collective)

    # -- 通信面 -------------------------------------------------------

    def attach(self, collective: Any) -> None:
        """接上通信后端；rank / world_size 必须与构造时一致。"""
        for name in ("all_reduce", "broadcast"):
            if not callable(getattr(collective, name, None)):
                raise TypeError(
                    f"collective 需要提供 {name}()，"
                    f"{type(collective).__name__} 没有"
                )
        rank = getattr(collective, "rank", self._rank)
        world = getattr(collective, "world_size", self._world_size)
        if int(rank) != self._rank:
            raise ValueError(
                f"collective 的 rank {rank} 与 ZeroRank 的 {self._rank} 不一致"
            )
        if int(world) != self._world_size:
            raise ValueError(
                f"collective 的 world_size {world} 与 ZeroRank 的 "
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
    def assignments(self) -> Mapping[str, int]:
        return dict(self._owners)

    @property
    def my_names(self) -> Sequence[str]:
        """本 rank 负责更新（也是唯一持有优化器槽位）的参数名。"""
        return tuple(self._my_names)

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @property
    def params(self) -> MutableMapping[str, np.ndarray]:
        return self._params

    def owned_bytes(self) -> int:
        """本进程常驻的参数与槽位字节数。

        ``params`` 是完整的一份（ZeRO-1 的边界），真正随分片缩小的是槽位；
        存档与内存核算都只看 ``slots`` 这一项。
        """
        slots = 0
        for arrays in self._optimizer._slots().values():
            for array in arrays.values():
                slots += int(np.asarray(array).nbytes)
        return slots

    # -- 单步 ---------------------------------------------------------

    def _flat_grads(self, grads: Mapping[str, Any]) -> np.ndarray:
        """校验并把整份梯度按参数顺序打包成一条连续缓冲。"""
        missing = [name for name in self._order if name not in grads]
        if missing:
            raise KeyError(
                f"这些参数没有梯度: {missing}。"
                "通常意味着损失不依赖它们；请检查构图"
            )
        unknown = [name for name in grads if name not in self._params]
        if unknown:
            raise KeyError(f"梯度里出现了不认识的参数: {unknown}")
        pieces = []
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
                    "不一致；打包成一条缓冲要求同一 dtype"
                )
            pieces.append(np.ascontiguousarray(array).ravel())
        return np.concatenate(pieces)

    def _slice_grads(self, flat: np.ndarray) -> Dict[str, np.ndarray]:
        """按参数顺序把归约后的缓冲切回各参数（只保留本片）。"""
        result: Dict[str, np.ndarray] = {}
        offset = 0
        for name in self._order:
            size = int(np.prod(np.shape(self._params[name]), dtype=np.int64))
            if self._owners[name] == self._rank:
                piece = flat[offset : offset + size]
                result[name] = piece.reshape(np.shape(self._params[name]))
            offset += size
        return result

    def step(self, grads: Mapping[str, np.ndarray], *, reduction: str = "mean") -> None:
        """归约整份梯度 → 只更新本片 → 广播本片参数。

        归约是**一次**整份缓冲的 AllReduce（见模块说明）；``reduction="mean"``
        在归约之后统一除以 world_size，与 ``allreduce_grads`` 的顺序一致。
        """
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction 必须是 mean / sum，收到 {reduction!r}")
        if self._collective is None:
            raise RuntimeError(
                "还没有接上集合通信：请先 attach(collective) 或构造时传入"
            )
        flat = self._flat_grads(grads)
        reduced = np.asarray(
            self._collective.all_reduce(flat, op="sum")
        )
        if reduced.shape != flat.shape:
            raise ValueError(
                f"collective.all_reduce 返回的形状 {reduced.shape} "
                f"与传入的 {flat.shape} 不一致"
            )
        if reduction == "mean":
            reduced = reduced / float(self._world_size)

        local = self._slice_grads(reduced)
        if local:
            self._optimizer.step(local)

        for name in self._order:
            owner = self._owners[name]
            array = np.asarray(self._params[name])
            updated = np.asarray(
                self._collective.broadcast(
                    np.ascontiguousarray(array), root=owner
                )
            )
            if updated.shape != array.shape:
                raise ValueError(
                    f"collective.broadcast 返回的 {name} 形状 "
                    f"{updated.shape} 与参数的 {array.shape} 不一致"
                )
            if updated.dtype != array.dtype:
                raise TypeError(
                    f"collective.broadcast 返回的 {name} dtype "
                    f"{updated.dtype} 与参数的 {array.dtype} 不一致"
                )
            array[...] = updated
        self.steps += 1

    # -- 存档 ---------------------------------------------------------

    def shard_state(self) -> Dict[str, Any]:
        """本 rank 的存档内容：步数 + 超参 + 本片槽位 + 归属表。"""
        state = self._optimizer.state_dict()
        return {
            "rank": self._rank,
            "world_size": self._world_size,
            "step": state["step"],
            "hyper": state["hyper"],
            "slots": state["slots"],
            "assignments": {
                name: owner
                for name, owner in self._owners.items()
                if owner == self._rank
            },
        }

    def save_shard(self, path: Any) -> str:
        """只写本片的槽位；n 个进程各写一个文件，合起来是一份完整状态。"""
        target = str(path)
        if not target.endswith(".npz"):
            target += ".npz"
        state = self.shard_state()
        arrays: Dict[str, np.ndarray] = {}
        for slot, by_name in state["slots"].items():
            for name, array in by_name.items():
                arrays[f"{slot}::{name}"] = np.asarray(array)
        arrays[_META_KEY] = np.array(
            json.dumps(
                {
                    "rank": state["rank"],
                    "world_size": state["world_size"],
                    "step": state["step"],
                    "hyper": state["hyper"],
                    "assignments": state["assignments"],
                },
                ensure_ascii=False,
            )
        )
        np.savez(target, **arrays)
        return target

    def load_shard(self, path: Any) -> None:
        """装回本片槽位；片号 / 分片数 / 归属表不符都是显式错误。"""
        target = str(path)
        if not target.endswith(".npz"):
            target += ".npz"
        with np.load(target, allow_pickle=False) as data:
            if _META_KEY not in data.files:
                raise ValueError(f"{target} 不是分片存档：缺少 {_META_KEY}")
            meta = json.loads(str(data[_META_KEY]))
            saved_rank = int(meta.get("rank", -1))
            if saved_rank != self._rank:
                raise ValueError(
                    f"{target} 属于 rank {saved_rank}，不能装进 rank {self._rank}"
                )
            saved_world = int(meta.get("world_size", 0))
            if saved_world != self._world_size:
                raise ValueError(
                    f"存档的分片数 {saved_world} 与当前 world_size "
                    f"{self._world_size} 不一致"
                )
            saved = {
                name: int(owner)
                for name, owner in dict(meta["assignments"]).items()
            }
            current = {
                name: owner
                for name, owner in self._owners.items()
                if owner == self._rank
            }
            if saved != current:
                raise ValueError(
                    f"rank {self._rank} 的参数归属与存档不同：存档 {saved}，"
                    f"当前 {current}"
                )
            slots: Dict[str, Dict[str, np.ndarray]] = {}
            for key in data.files:
                if key == _META_KEY:
                    continue
                slot, _, name = key.partition("::")
                if not name:
                    raise ValueError(f"{target} 里的键 {key!r} 不是 slot::name 形式")
                slots.setdefault(slot, {})[name] = np.array(data[key], copy=True)
            state = {
                "step": int(meta["step"]),
                "hyper": meta["hyper"],
                "slots": slots,
            }
        self._optimizer.load_state_dict(state)

    def __repr__(self) -> str:
        return (
            f"ZeroRank(rank={self._rank}/{self._world_size}, "
            f"owned={self._my_names}, steps={self.steps})"
        )


class LoopbackCollective:
    """进程内的假 world：多个 rank 用线程并发执行，集合调用按序配对。

    它不是通信实现，只用来在没有 GPU 的机器上验证分片语义。语义刻意与
    NCCL 对齐：每个集合调用都要等齐全部 rank（``Condition`` 屏障），
    ``all_reduce`` 按 rank 序号累加——与 ``allreduce_grads`` 的顺序一致，
    因此可以逐位对拍。真机上的 NCCL 不保证这个顺序，真机用例的判据是容差。

    每个 rank 一个实例，共享同一个 ``_LoopbackWorld``；各实例各自维护调用
    序号，因此**各 rank 必须按相同顺序发起集合调用**（``ZeroRank.step``
    天然满足）。
    """

    def __init__(self, world: "LoopbackWorld", rank: int) -> None:
        if not 0 <= rank < world.world_size:
            raise ValueError(f"rank {rank} 越界：world_size={world.world_size}")
        self._world = world
        self.rank = rank
        self.world_size = world.world_size
        self._seq = 0

    def all_reduce(self, array: Any, *, op: str = "sum") -> np.ndarray:
        if op != "sum":
            raise ValueError(f"LoopbackCollective 只模拟 sum，收到 {op!r}")
        seq = self._next()
        return self._world._collect(seq, self.rank, np.asarray(array), None)

    def broadcast(self, array: Any, *, root: int) -> np.ndarray:
        if not 0 <= root < self.world_size:
            raise ValueError(f"root {root} 越界：world_size={self.world_size}")
        seq = self._next()
        return self._world._collect(seq, self.rank, np.asarray(array), root)

    def _next(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq


class LoopbackWorld:
    """线程屏障 + 调用配对：同一 ``seq`` 的集合调用收齐后一起算结果。"""

    def __init__(self, world_size: int) -> None:
        if isinstance(world_size, bool) or not isinstance(world_size, int):
            raise TypeError(f"world_size 必须是 int，收到 {world_size!r}")
        if world_size < 1:
            raise ValueError(f"world_size 必须 >= 1，收到 {world_size}")
        self.world_size = world_size
        self._cond = threading.Condition()
        self._calls: Dict[int, Dict[int, np.ndarray]] = {}
        self._timeout = 30.0

    def collective(self, rank: int) -> LoopbackCollective:
        return LoopbackCollective(self, rank)

    def _collect(
        self, seq: int, rank: int, array: np.ndarray, root: Optional[int]
    ) -> np.ndarray:
        with self._cond:
            entry = self._calls.setdefault(seq, {})
            if rank in entry:
                raise RuntimeError(
                    f"rank {rank} 在集合调用 {seq} 上提交了两次"
                )
            entry[rank] = np.array(array, copy=True)
            self._cond.notify_all()
            deadline = time.time() + self._timeout
            while len(entry) < self.world_size:
                remaining = deadline - time.time()
                if remaining <= 0:
                    missing = sorted(set(range(self.world_size)) - set(entry))
                    raise RuntimeError(
                        f"集合调用 {seq} 等不到这些 rank: {missing}"
                    )
                self._cond.wait(remaining)
            if root is None:
                total = np.array(entry[0], copy=True)
                for other in range(1, self.world_size):
                    total += entry[other]
                result = total
            else:
                result = np.array(entry[root], copy=True)
            # 不在这里删除 entry：先返回的 rank 可能还在等其余 rank 醒来，
            # 删掉共享状态会让它们的计算变成"少了一个 rank"。这是进程内的
            # 测试替身，调用次数有限，保留到 world 被回收即可。
            return np.array(result, copy=True)


class NcclHostCollective:
    """把 NCCL 的设备侧集合通信包成"主机数组进、主机数组出"。

    每个 rank 一张卡（``device`` 缺省等于 ``rank``）。每次调用做
    H2D → NCCL → 同步 → D2H；这是"正确性优先"的形态，跨进程 ZeRO 的每步
    通信量本来就是一份梯度 + 一份参数，主机往返不是瓶颈的主要来源。
    broadcast 时非 root 不必上传数据，只分配一块同形状的空缓冲。
    """

    def __init__(self, comm: Any, *, device: Optional[int] = None) -> None:
        self.comm = comm
        self.rank = int(getattr(comm, "rank"))
        self.world_size = int(getattr(comm, "world_size"))
        self.device = int(self.rank if device is None else device)

    def _buffer(self, array: Any, *, copy: bool):
        import cupy as cp

        with cp.cuda.Device(self.device):
            if copy:
                return cp.asarray(array)
            return cp.empty(tuple(np.shape(array)), dtype=np.dtype(array.dtype).name)

    def all_reduce(self, array: Any, *, op: str = "sum") -> np.ndarray:
        import cupy as cp

        with cp.cuda.Device(self.device):
            buf = self._buffer(array, copy=True)
            self.comm.all_reduce(buf, op=op)
            self.comm.synchronize()
            return cp.asnumpy(buf)

    def broadcast(self, array: Any, *, root: int) -> np.ndarray:
        import cupy as cp

        with cp.cuda.Device(self.device):
            buf = self._buffer(array, copy=(self.rank == int(root)))
            self.comm.broadcast(buf, root=int(root))
            self.comm.synchronize()
            return cp.asnumpy(buf)
