"""ZeRO 第 1 级：把优化器状态按参数名分片。

数据并行（5.190）让每块卡都有一整份参数副本，但**优化器状态**（Adam 的一阶 /
二阶矩、SGD 的动量）在每个训练进程里仍是一整份。ZeRO 的观察很简单：槽位与
参数的形状一一对应、更新也是逐参数的，所以槽位可以按参数切开，每个 rank 只
维护自己那一片；算完再把参数同步回整份。

这一层刻意**只在主机侧**做分片：参数表仍是一整份 ``dict``，各分片的优化器拿到
的是**同一批 ndarray 对象**的子集，因此 ``param -= ...`` 原地改的就是共享的
那份数据——不需要额外的"写回"步骤，也不存在两份副本不一致的窗口。

**为什么分片不影响数值。** Adam / SGD 的更新是逐参数的：某个参数的 ``m`` / ``v``
只与它自己的梯度、以及共享的步数有关。分片只是把"对全部参数循环"换成"对
分片循环"，每个参数内部的运算序列一个字都没变，所以结果**逐位相等**，不是
"在一个容差内相等"。这条性质是测试的判据。

**边界。** 分片按**名字**切，不按元素切——拆单个张量要改优化器的槽位布局，
而收益只在"一个参数就大到一片装不下"时才出现。按参数字节数做贪心均衡，各片
最多相差一个参数的大小；世界大小大于参数个数时会出现空片，这是合法状态，
不报错（空片仍要参与 ``step_count`` 的推进，否则各片步数会分叉）。真机多进程
没有做：这里省的是同一进程的主机内存，不是显存；要省显存需要"每个进程只上传
自己那片参数"，那属于运行时改造（与 5.190 的边界一致）。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from .optim import Optimizer

__all__ = ["shard_params", "ShardedOptimizer", "reduce_scatter_grads", "Transport"]

#: 归约传输：``(片号, 本片梯度) -> 本片梯度``。真机上是 Reduce-Scatter 的
#: 接收侧；默认实现是进程内直算，只用于验证语义与统计通信量。
Transport = Callable[[int, Mapping[str, np.ndarray]], Mapping[str, np.ndarray]]


def shard_params(
    params: Mapping[str, Any], world_size: int
) -> List[List[str]]:
    """把参数名切成 ``world_size`` 片，按字节数尽量均衡。

    用"每次放进当前最轻的一片"的贪心：结果只由参数名的顺序与形状决定，
    同样的输入永远得到同样的分片（可复现比"更精确的划分"重要——否则两次
    运行的槽位归属不同，存档就对不上了）。

    参数表为空、``world_size`` 非法（不是正整数）都当场报错；参数个数少于
    片数时返回的列表里会有空片，这是合法的，调用方不得据此推断"配置错了"。
    """
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise TypeError(f"world_size 必须是 int，收到 {world_size!r}")
    if world_size < 1:
        raise ValueError(f"world_size 必须 >= 1，收到 {world_size}")
    names = list(params)
    if not names:
        raise ValueError("参数表为空：没有可分片的优化器状态")

    buckets: List[List[str]] = [[] for _ in range(world_size)]
    weights = [0] * world_size
    for name in names:
        size = int(np.prod(np.shape(params[name]), dtype=np.int64))
        # 平手时取下标最小的片，保证确定性。
        target = min(range(world_size), key=lambda i: (weights[i], i))
        buckets[target].append(name)
        weights[target] += size
    return buckets


def reduce_scatter_grads(
    grads_per_rank: Sequence[Mapping[str, np.ndarray]],
    params: Mapping[str, Any],
    *,
    world_size: Optional[int] = None,
    reduction: str = "mean",
    transport: Optional[Transport] = None,
) -> List[Dict[str, np.ndarray]]:
    """ZeRO 的通信面：用 Reduce-Scatter 取代 All-Reduce。

    数据并行里每个 rank 都算出**整份**梯度，但优化器只消费自己那片参数的
    梯度。All-Reduce 会把归约后的整份梯度发给每个 rank（每 rank 收 N 字节，
    全网 W·N）；Reduce-Scatter 只把每片交给它的主人（每 rank 收 N/W 字节，
    全网 N）。这份差额就是 ZeRO 省下的通信量，而**数值上完全等价**——每个
    参数仍然是在全部 rank 上求和 / 求平均，只是结果不再广播给不关心的 rank。

    约定：

    - ``grads_per_rank[i]`` 是第 ``i`` 个 rank 在本步算出的梯度；名字必须与
      ``params`` 完全一致（少一个说明反向没跑全，多一个说明参数表分叉）。
    - 归约顺序沿用 ``allreduce`` 的"从第 0 个依次累加"，因此结果与
      ``allreduce_grads`` **逐位相同**，不是"在容差内接近"。
    - ``transport`` 拿到的是"这一片、已经归约好的梯度"，返回值必须恰好覆盖
      同一组名字；默认 ``None`` 表示进程内直算。空片不调用 ``transport``，
      也不会出现在结果里——世界大小大于参数个数是合法状态（与 ``shard_params``
      一致）。
    - 参数本身不做 all-gather：本层象 ``ShardedOptimizer`` 一样假定各 rank
      共享同一份主机参数，跨进程的参数同步是运行时的事。

    返回按片号排列的列表，每项是该 rank 需要的那几个参数的梯度。
    """
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction 必须是 mean / sum，收到 {reduction!r}")
    if isinstance(grads_per_rank, (str, bytes)) or not isinstance(
        grads_per_rank, Sequence
    ):
        raise TypeError(
            f"grads_per_rank 必须是序列，得到 {type(grads_per_rank).__name__}"
        )
    ranks = len(grads_per_rank)
    if ranks == 0:
        raise ValueError("grads_per_rank 不能为空：至少需要一个 rank 的梯度")
    if world_size is None:
        world_size = ranks
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise TypeError(f"world_size 必须是 int，收到 {world_size!r}")
    if world_size != ranks:
        raise ValueError(
            f"world_size={world_size} 与给出的梯度份数 {ranks} 不一致；"
            "每一片都必须有自己的本地梯度"
        )

    names = list(params)
    if not names:
        raise ValueError("参数表为空：没有可归约的梯度")
    expected = set(names)
    for index, grads in enumerate(grads_per_rank):
        if not isinstance(grads, Mapping):
            raise TypeError(
                f"第 {index} 个 rank 的梯度必须是映射，得到 {type(grads).__name__}"
            )
        if set(grads) != expected:
            raise KeyError(
                f"第 {index} 个 rank 的梯度名字与参数表不一致："
                f"多 {sorted(set(grads) - expected)}，"
                f"少 {sorted(expected - set(grads))}"
            )
        for name in names:
            if np.shape(grads[name]) != np.shape(params[name]):
                raise ValueError(
                    f"第 {index} 个 rank 的 {name} 梯度形状 {np.shape(grads[name])} "
                    f"与参数形状 {np.shape(params[name])} 不一致"
                )

    assignments = shard_params(params, world_size)
    result: List[Dict[str, np.ndarray]] = []
    for index, shard_names in enumerate(assignments):
        local: Dict[str, np.ndarray] = {}
        for name in shard_names:
            first = grads_per_rank[0][name]
            # 与 allreduce 相同的累加顺序，保证逐位一致
            total = np.array(first, copy=True)
            for other in grads_per_rank[1:]:
                array = other[name]
                if array.dtype != total.dtype:
                    raise TypeError(
                        f"{name} 在各 rank 上的 dtype 不一致："
                        f"{total.dtype} vs {array.dtype}"
                    )
                total += array
            if reduction == "mean":
                total /= ranks
            local[name] = total
        if transport is not None and local:
            carried = transport(index, local)
            if set(carried) != set(shard_names):
                raise KeyError(
                    f"transport 为第 {index} 片返回的名字不对："
                    f"多 {sorted(set(carried) - set(shard_names))}，"
                    f"少 {sorted(set(shard_names) - set(carried))}"
                )
            local = {name: np.asarray(carried[name]) for name in shard_names}
        result.append(local)
    return result


class ShardedOptimizer:
    """按参数名分片的优化器：每片一个独立实例，槽位只覆盖本片。

    ``factory`` 是"给一份参数子表、造一个优化器"的可调用对象，例如

        ShardedOptimizer(lambda p: Adam(p, lr=1e-3), store.arrays(), world_size=2)

    每片的超参必须完全相同——分片是**存储布局**，不是"每片用不同学习率"。
    构造时逐片核对 ``hyperparams()``，不一致就报错。

    为什么用工厂而不是复制已有优化器：优化器的槽位是按名字懒分配的，直接
    复制一个实例会把整份槽位也复制过来，分片就失去意义了。
    """

    def __init__(
        self,
        factory: Callable[[MutableMapping[str, np.ndarray]], Optimizer],
        params: MutableMapping[str, np.ndarray],
        *,
        world_size: int,
    ) -> None:
        if not callable(factory):
            raise TypeError(f"factory 必须可调用，收到 {type(factory).__name__}")
        if not isinstance(params, MutableMapping):
            raise TypeError(
                f"params 必须是可写映射（优化器要原地更新），"
                f"得到 {type(params).__name__}"
            )
        self._params = params
        self._world_size = world_size
        self._names = shard_params(params, world_size)
        self._shards: List[Optimizer] = []
        for index, names in enumerate(self._names):
            sub = {name: params[name] for name in names}
            shard = factory(sub)
            if not isinstance(shard, Optimizer):
                raise TypeError(
                    f"第 {index} 片 factory 返回了 {type(shard).__name__}，"
                    "不是 Optimizer"
                )
            if set(shard.params) != set(names):
                raise ValueError(
                    f"第 {index} 片 factory 造出的优化器参数名与分片不一致："
                    f"多 {sorted(set(shard.params) - set(names))}，"
                    f"少 {sorted(set(names) - set(shard.params))}"
                )
            self._shards.append(shard)
        self._check_hyper()

    # -- 校验 ---------------------------------------------------------

    def _check_hyper(self) -> None:
        """各片超参必须一致：分片不能顺带改更新语义。"""
        if not self._shards:
            return
        reference = self._shards[0].hyperparams()
        for index, shard in enumerate(self._shards[1:], 1):
            current = shard.hyperparams()
            if current != reference:
                raise ValueError(
                    f"第 {index} 片的超参与第 0 片不一致：{current} vs {reference}；"
                    "分片只切存储，不允许每片用不同超参"
                )

    # -- 属性 ---------------------------------------------------------

    @property
    def params(self) -> MutableMapping[str, np.ndarray]:
        return self._params

    @property
    def assignments(self) -> Mapping[str, int]:
        """参数名 → 所属分片号。存档与调试都用它核对归属。"""
        return {
            name: index for index, names in enumerate(self._names) for name in names
        }

    @property
    def shards(self) -> Sequence[Optimizer]:
        return tuple(self._shards)

    @property
    def step_count(self) -> int:
        return self._shards[0].step_count if self._shards else 0

    @property
    def lr(self) -> float:
        return self._shards[0].lr if self._shards else 0.0

    @lr.setter
    def lr(self, value: float) -> None:
        """学习率调度要能作用到全部分片；逐片赋值，任一片拒绝就整体报错。"""
        if value <= 0.0:
            raise ValueError(f"学习率必须为正，得到 {value}")
        for shard in self._shards:
            shard.lr = float(value)

    # -- 单步 ---------------------------------------------------------

    def step(self, grads: Mapping[str, np.ndarray]) -> None:
        """每片只消费自己那几个参数的梯度。

        与 ``Optimizer.step`` 同一条契约：**先校验、再更新**。这里先确认梯度
        恰好覆盖全部参数（缺名字说明某层的反向没跑；多名字说明参数表分叉了），
        再逐片更新，避免"前几片已经走了一步、后面才报错"的半程状态。
        """
        missing = [name for name in self._params if name not in grads]
        if missing:
            raise KeyError(
                f"这些参数没有梯度: {missing}。"
                "通常意味着损失不依赖它们；请检查构图，而不是让优化器跳过不动的参数"
            )
        unknown = [name for name in grads if name not in self._params]
        if unknown:
            raise KeyError(f"梯度里出现了不认识的参数: {unknown}")

        # 先让每一片都把校验与整理做完，再统一提交：任何一片的梯度不合法
        # 都不该让前面的片先走半步（各片步数会就此分叉）。
        prepared = [
            shard.prepare_step({name: grads[name] for name in names})
            for names, shard in zip(self._names, self._shards)
        ]
        for shard, batch in zip(self._shards, prepared):
            shard.apply_step(batch)

    # -- 状态 ---------------------------------------------------------

    def hyperparams(self) -> dict:
        return dict(self._shards[0].hyperparams()) if self._shards else {}

    def state_dict(self) -> dict:
        """按片存档，并带上归属表。

        只存"每片的槽位"是不够的：恢复时如果参数的归属变了（比如改了
        ``world_size``），槽位会被装到错的地方，而形状全都对得上——归属表
        让这件事变成显式错误。
        """
        return {
            "world_size": self._world_size,
            "assignments": dict(self.assignments),
            "shards": [shard.state_dict() for shard in self._shards],
        }

    def load_state_dict(self, state: Mapping, *, restore_lr: bool = False) -> None:
        world = int(state.get("world_size", 0))
        if world != self._world_size:
            raise ValueError(
                f"存档的分片数 {world} 与当前 world_size {self._world_size} 不一致；"
                "分片布局变了必须先把状态合并回整份再重新分片"
            )
        saved = dict(state.get("assignments") or {})
        current = dict(self.assignments)
        if saved != current:
            bad = sorted(
                name
                for name in set(saved) | set(current)
                if saved.get(name) != current.get(name)
            )
            raise ValueError(
                f"这些参数的归属与存档不同: {bad}；"
                "归属不同意味着槽位会被装到别的片上"
            )
        shards = list(state.get("shards") or [])
        if len(shards) != len(self._shards):
            raise ValueError(
                f"存档有 {len(shards)} 片，当前有 {len(self._shards)} 片"
            )
        # 先全部校验一遍再提交：任一片的存档不合法时，前面几片不能已经换掉，
        # 否则各片的步数与槽位会互相矛盾（续训第一发就对不上）。
        prepared = [
            shard._prepare_state_dict(payload, restore_lr=restore_lr)
            for shard, payload in zip(self._shards, shards)
        ]
        for shard, batch in zip(self._shards, prepared):
            shard._commit_state(*batch)

    def __repr__(self) -> str:
        sizes = [len(names) for names in self._names]
        return (
            f"ShardedOptimizer(world_size={self._world_size}, "
            f"params={len(self._params)}, shards={sizes}, "
            f"steps={self.step_count})"
        )
