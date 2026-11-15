"""跨进程 ZeRO-3：参数本身也拆开，用时 AllGather、用完即释放。

到 5.231 为止，参数在每个 rank 上仍是**一整份**：ZeRO-1 切了优化器槽位，
ZeRO-2 又切了梯度，但前向与反向要读权重，所以每个进程都驻留完整参数。参数
一多，"每张卡一份副本"就成了常驻内存的下限——这是 ZeRO-3 要拆的东西。

**做法。** 每个参数按元素均分成 ``world_size`` 块，rank ``r`` 只常驻第 ``r``
块（一维、定长 ``chunk = ceil(n / W)``，尾部补零）。参数的真身从此只是一堆
分片；要算的时候临时拼回来：

    materialize(names)  →  每个名字做一次 AllGather，拼回完整形状（进缓存）
    前向 / 反向          →  用拼回来的那份算
    release(names)      →  丢掉拼回来的副本，回到只持分片的状态
    step(grads)         →  每个参数 ReduceScatter 出本块的梯度、只更本块

**常驻内存** 从"参数 + 槽位"各一份，降到各 ``1/W``；代价是每步前向/反向期间
被借用的那部分参数会短暂存在。

**为什么按参数均分、而不是把所有参数拍成一条大缓冲再切。** 拍平能让一步
只发一条消息，但参数边界会与块边界错位——某个参数横跨两块时，优化器就要
在两块上各算半张量，超参（权重衰减、逐参数学习率）也会被劈开。按参数均分
没有这个问题：每块都属于唯一一个参数，优化器看到的仍是一张"完整形状"的
参数（只是被切成了 1/W 长的片段），数值语义与不分片完全相同。

**通信量。** 拼回一个参数：本块发出 ``chunk``、收回 ``chunk × W``（约等于
``n``）；一步里把所有参数各拼一次，就是"每个参数一条 AllGather"。消息数随
参数个数线性增长，与 ZeRO-1 的参数广播同量级——把多个参数合并成一条消息需要
"等长段 + 边界对齐"，那正是 5.231 的补零做法，这里刻意不做，避免一个参数
被切成两半。

**数值。** 每个参数的更新仍是逐元素的，分块不改变任何一个元素的运算序列，
因此与不分片逐位相等（进程内 ``LoopbackCollective`` 的用例判据就是逐位）。
真机上 AllGather 只搬数据、不做归约，所以拼回来的参数也是逐位的；只有
``step`` 里的 ReduceScatter 求和顺序由拓扑决定。

**边界。**
- 参数表必须在**任何** rank 上都以同一份定义顺序构造，否则块与参数对不上；
  这里不跨进程核对，只把"同一模型定义顺序"当前提写下来。
- ``materialize`` 拿到的数组是**副本**：改它不会改分片。要改参数只能走
  ``step``（这是"一份真相"的代价：拼回来的副本只是只读用品）。
- 与 ZeRO-1/2 不同，ZeRO-3 的存档里**必须同时**保存参数分片与槽位，否则
  装回来时参数是空的。
- 释放的时机由调用方决定：``release`` 早了会让后续算子读到已释放的副本
  （这里用显式异常拦下，不做静默重新拼）。
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from .optim import Optimizer

__all__ = ["Zero3Rank", "plan_chunks"]

_META_KEY = "__meta__"

#: 一个参数的分片信息：``(形状, 元素数, 块长)``。
Chunk = Tuple[Tuple[int, ...], int, int]


def plan_chunks(
    params: Mapping[str, Any], world_size: int
) -> Dict[str, Chunk]:
    """给每个参数算出 ``(形状, 元素数, 块长)``。

    块长取 ``ceil(n / W)``：最后一块的真实元素可能不足一块，缺口补零。补零
    只影响通信缓冲，不影响数值——补出来的位置梯度恒为 0，优化器更新后仍是 0。
    """
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise TypeError(f"world_size 必须是 int，收到 {world_size!r}")
    if world_size < 1:
        raise ValueError(f"world_size 必须 >= 1，收到 {world_size}")
    names = list(params)
    if not names:
        raise ValueError("参数表为空：没有可分片的参数")
    plan: Dict[str, Chunk] = {}
    for name in names:
        shape = tuple(int(dim) for dim in np.shape(params[name]))
        size = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if size <= 0:
            raise ValueError(f"{name} 是空张量（形状 {shape}），没有可分片的元素")
        plan[name] = (shape, size, -(-size // world_size))
    return plan


class Zero3Rank:
    """一个 rank 上的 ZeRO-3 参数驻留 + 训练步骤。

    ``params`` 传**完整参数表**（用来取材），构造后只有本块留在进程里，传入
    的那份表不再被引用：调用方要真正省内存，构造完就该丢掉它。``factory``
    收到的是"本块"的参数子表——每个参数都是长度 ``chunk`` 的一维数组，优化器
    的槽位因此只有整份的 ``1/W``。
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
        if not isinstance(params, Mapping):
            raise TypeError(f"params 必须是映射，得到 {type(params).__name__}")
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

        self._order = list(params)
        self._plan = plan_chunks(params, world_size)
        self._rank = rank
        self._world_size = world_size

        dtypes = {np.asarray(params[name]).dtype for name in self._order}
        if len(dtypes) != 1:
            raise TypeError(
                f"ZeRO-3 要求全部参数同一 dtype（分片与通信缓冲都要同型），"
                f"得到 {sorted(str(d) for d in dtypes)}；"
                "混合精度请先统一到主权重精度"
            )
        self._dtype = next(iter(dtypes))

        # 只取本块：从完整参数里切一维、复制出来，构造完就不再引用原表。
        self._shards: Dict[str, np.ndarray] = {}
        for name in self._order:
            shape, size, chunk = self._plan[name]
            flat = np.ascontiguousarray(np.asarray(params[name])).ravel()
            piece = np.zeros(chunk, dtype=self._dtype)
            start = rank * chunk
            stop = min(start + chunk, size)
            if start < stop:
                piece[: stop - start] = flat[start:stop]
            self._shards[name] = piece

        shard = factory(self._shards)
        if not isinstance(shard, Optimizer):
            raise TypeError(f"factory 返回了 {type(shard).__name__}，不是 Optimizer")
        if set(shard.params) != set(self._shards):
            raise KeyError(
                f"factory 造出的优化器参数名与本块不一致："
                f"多 {sorted(set(shard.params) - set(self._shards))}，"
                f"少 {sorted(set(self._shards) - set(shard.params))}"
            )
        for name, array in shard.params.items():
            if np.asarray(array) is not self._shards[name]:
                raise ValueError(
                    f"优化器里的 {name} 不是本块那个数组；"
                    "factory 复制了子表，原地更新不会落到分片上"
                )
        self._optimizer = shard
        self._cache: Dict[str, np.ndarray] = {}
        self._collective: Optional[Any] = None
        if collective is not None:
            self.attach(collective)

    # -- 通信面 -------------------------------------------------------

    def attach(self, collective: Any) -> None:
        """接上通信后端；需要 ``all_gather`` 与 ``reduce_scatter``。"""
        for name in ("all_gather", "reduce_scatter"):
            if not callable(getattr(collective, name, None)):
                raise TypeError(
                    f"ZeRO-3 的 collective 需要 {name}()，"
                    f"{type(collective).__name__} 没有"
                )
        rank = getattr(collective, "rank", self._rank)
        world = getattr(collective, "world_size", self._world_size)
        if int(rank) != self._rank:
            raise ValueError(
                f"collective 的 rank {rank} 与 Zero3Rank 的 {self._rank} 不一致"
            )
        if int(world) != self._world_size:
            raise ValueError(
                f"collective 的 world_size {world} 与 Zero3Rank 的 "
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
    def param_names(self) -> Sequence[str]:
        return tuple(self._order)

    def shapes(self) -> Mapping[str, Tuple[int, ...]]:
        return {name: self._plan[name][0] for name in self._order}

    def chunk_size(self, name: str) -> int:
        """本参数每块的元素数（通信缓冲的最小单位）。"""
        try:
            return self._plan[name][2]
        except KeyError:
            raise KeyError(f"不认识的参数: {name!r}") from None

    @property
    def steps(self) -> int:
        """已经走过的步数；唯一真相是优化器的 ``step_count``。"""
        return int(self._optimizer.step_count)

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @property
    def shards(self) -> Mapping[str, np.ndarray]:
        """本 rank 常驻的参数分片（只读用途；改参数请走 ``step``）。"""
        return dict(self._shards)

    @property
    def resident(self) -> Mapping[str, np.ndarray]:
        """当前拼回内存的参数副本；空表示已全部释放。"""
        return dict(self._cache)

    def owned_bytes(self) -> int:
        """本进程常驻的参数字节数（不含槽位、不含拼回来的副本）。"""
        return sum(int(array.nbytes) for array in self._shards.values())

    def full_bytes(self) -> int:
        """完整一份参数的字节数，用来对照 ``owned_bytes()``。"""
        return sum(
            self._plan[name][1] * np.dtype(self._dtype).itemsize
            for name in self._order
        )

    def materialize_elements(self, names: Optional[Sequence[str]] = None) -> int:
        """把 ``names`` 全部拼一次，总共收发多少元素。"""
        chosen = self._checked_names(names)
        return sum(self._plan[name][2] * self._world_size for name in chosen)

    # -- 参数拼回 / 释放 ----------------------------------------------

    def _checked_names(self, names: Optional[Sequence[str]]) -> List[str]:
        if names is None:
            return list(self._order)
        if isinstance(names, str):
            raise TypeError("names 必须是参数名的序列，不是单个字符串")
        chosen = list(names)
        for name in chosen:
            if name not in self._plan:
                raise KeyError(f"不认识的参数: {name!r}")
        return chosen

    def materialize(self, names: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
        """拼回完整参数（AllGather 每块），返回**副本**。

        已经拼过的名字直接复用缓存，直到 ``release`` 为止。返回的数组与分片
        不共享内存：它只供前向 / 反向读，改了也不会回写。
        """
        if self._collective is None:
            raise RuntimeError(
                "还没有接上集合通信：请先 attach(collective) 或构造时传入"
            )
        chosen = self._checked_names(names)
        for name in chosen:
            if name in self._cache:
                continue
            shape, size, chunk = self._plan[name]
            received = np.empty(chunk * self._world_size, dtype=self._dtype)
            merged = np.asarray(
                self._collective.all_gather(self._shards[name], received)
            )
            if merged.shape != received.shape:
                raise ValueError(
                    f"collective.all_gather 返回的形状 {merged.shape} "
                    f"与期望的 {received.shape} 不一致（参数 {name}）"
                )
            self._cache[name] = np.array(merged[:size], dtype=self._dtype).reshape(
                shape
            )
        return {name: self._cache[name] for name in chosen}

    def release(self, names: Optional[Sequence[str]] = None) -> None:
        """丢掉拼回来的副本；``None`` 表示全部释放。"""
        chosen = self._checked_names(names)
        for name in chosen:
            self._cache.pop(name, None)

    # -- 单步 ---------------------------------------------------------

    def _checked_grads(self, grads: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        missing = [name for name in self._order if name not in grads]
        if missing:
            raise KeyError(
                f"这些参数没有梯度: {missing}。"
                "通常意味着损失不依赖它们；请检查构图"
            )
        unknown = [name for name in grads if name not in self._plan]
        if unknown:
            raise KeyError(f"梯度里出现了不认识的参数: {unknown}")
        checked: Dict[str, np.ndarray] = {}
        for name in self._order:
            array = np.asarray(grads[name])
            shape = self._plan[name][0]
            if array.shape != shape:
                raise ValueError(
                    f"{name} 的梯度形状 {array.shape} 与参数形状 {shape} 不一致"
                )
            if array.dtype != self._dtype:
                raise TypeError(
                    f"{name} 的梯度 dtype {array.dtype} 与参数的 {self._dtype} "
                    "不一致；分片与归约缓冲要求同一 dtype"
                )
            checked[name] = array
        return checked

    def step(self, grads: Mapping[str, np.ndarray], *, reduction: str = "mean") -> None:
        """每个参数 ReduceScatter 出本块的梯度，只更新本块分片。"""
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction 必须是 mean / sum，收到 {reduction!r}")
        if self._collective is None:
            raise RuntimeError(
                "还没有接上集合通信：请先 attach(collective) 或构造时传入"
            )
        checked = self._checked_grads(grads)
        local: Dict[str, np.ndarray] = {}
        for name in self._order:
            shape, size, chunk = self._plan[name]
            send = np.zeros(chunk * self._world_size, dtype=self._dtype)
            send[:size] = checked[name].ravel()
            received = np.empty(chunk, dtype=self._dtype)
            reduced = np.asarray(
                self._collective.reduce_scatter(send, received, op="sum")
            )
            if reduced.shape != (chunk,):
                raise ValueError(
                    f"collective.reduce_scatter 返回的形状 {reduced.shape} "
                    f"与块长 {chunk} 不一致（参数 {name}）"
                )
            if reduction == "mean":
                reduced = reduced / float(self._world_size)
            local[name] = reduced
        self._optimizer.step(local)
        # 参数变了，之前拼回来的副本立刻过期：留着就是两份真相。
        self._cache.clear()

    # -- 存档 ---------------------------------------------------------

    def shard_state(self) -> Dict[str, Any]:
        """本 rank 的存档内容：步数 + 超参 + 本块参数 + 本块槽位。"""
        state = self._optimizer.state_dict()
        return {
            "rank": self._rank,
            "world_size": self._world_size,
            "step": state["step"],
            "hyper": state["hyper"],
            "params": {
                name: np.array(self._shards[name], copy=True) for name in self._order
            },
            "slots": state["slots"],
        }

    def save_shard(self, path: Any) -> str:
        """只写本块：参数分片与槽位各一份，n 个进程合起来是完整状态。"""
        state = self.shard_state()
        target = str(path)
        if not target.endswith(".npz"):
            target += ".npz"
        arrays: Dict[str, np.ndarray] = {}
        for name, array in state["params"].items():
            arrays[f"param::{name}"] = array
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
                    "plan": {
                        name: [list(self._plan[name][0]), self._plan[name][2]]
                        for name in self._order
                    },
                },
                ensure_ascii=False,
            )
        )
        np.savez(target, **arrays)
        return target

    def load_shard(self, path: Any) -> None:
        """装回本块参数与槽位；片号 / 分片数 / 参数表不符都是显式错误。"""
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
                    f"{self._world_size} 不一致；块布局变了必须先合并再重新分片"
                )
            saved_plan = {
                name: (tuple(int(dim) for dim in shape), int(chunk))
                for name, (shape, chunk) in dict(meta["plan"]).items()
            }
            current_plan = {
                name: (self._plan[name][0], self._plan[name][2])
                for name in self._order
            }
            if saved_plan != current_plan:
                diff = sorted(
                    name
                    for name in set(saved_plan) | set(current_plan)
                    if saved_plan.get(name) != current_plan.get(name)
                )
                raise ValueError(
                    f"这些参数的形状或块长与存档不同: {diff}；"
                    "块布局变了，装进去会错位"
                )
            params: Dict[str, np.ndarray] = {}
            slots: Dict[str, Dict[str, np.ndarray]] = {}
            for key in data.files:
                if key == _META_KEY:
                    continue
                kind, _, name = key.partition("::")
                if not name:
                    raise ValueError(f"{target} 里的键 {key!r} 不是 kind::name 形式")
                if kind == "param":
                    params[name] = np.array(data[key], copy=True)
                else:
                    slots.setdefault(kind, {})[name] = np.array(data[key], copy=True)
            if set(params) != set(self._order):
                raise ValueError(
                    f"{target} 里的参数分片与本 rank 不一致："
                    f"多 {sorted(set(params) - set(self._order))}，"
                    f"少 {sorted(set(self._order) - set(params))}"
                )
            for name, array in params.items():
                expected = self._plan[name][2]
                if array.shape != (expected,):
                    raise ValueError(
                        f"{target} 里 {name} 的分片形状 {array.shape} "
                        f"与块长 ({expected},) 不一致"
                    )
                if array.dtype != self._dtype:
                    raise TypeError(
                        f"{target} 里 {name} 的分片 dtype {array.dtype} "
                        f"与当前的 {self._dtype} 不一致"
                    )
            state = {
                "step": int(meta["step"]),
                "hyper": meta["hyper"],
                "slots": slots,
            }
        # 先全部校验完再提交：槽位装到一半失败，参数与步数就会互相矛盾。
        self._optimizer.load_state_dict(state)
        for name, array in params.items():
            self._shards[name][...] = array
        self._cache.clear()

    def __repr__(self) -> str:
        return (
            f"Zero3Rank(rank={self._rank}/{self._world_size}, "
            f"params={len(self._order)}, resident={len(self._cache)}, "
            f"steps={self.steps})"
        )
