"""分布式存档的合并与重分片：N 个分片 ↔ 一份完整状态。

ZeRO-1/2/3 的 ``save_shard`` 都只写"本 rank 那一份"：单卡数续训够用，但两件
事做不了——

1. **合并**：把 N 个分片合成一份完整权重 / 优化器状态（推理、发布、单卡续训）；
2. **换 world_size**：8 卡训到一半换成 4 卡（或反过来），分片数与归属全变了，
   直接 ``load_shard`` 会被"分片数不一致"拦下。

本模块补这两件。它**不碰通信**：全是文件读写的纯函数，任何进程都能调，也不
需要六个 rank 同时在线。换 world_size 靠的是"先拼回完整状态、再按新布局重新
切"——而不是让两个不同规模的 world 直接对话。

**两种布局。**

- ``names``（ZeRO-1 / ZeRO-2）：按参数名分片，存档里**只有槽位**，参数在模型
  里。合并掉的只是各片槽位的并集；要写出完整 checkpoint，必须把模型权重作为
  ``params=`` 一起给进来。
- ``elements``（ZeRO-3）：按元素切块，存档里**有参数分片**，合并后能原样拼回
  完整参数，不需要额外权重。

合并产物的判据只有一条：**能被既有读取路径读回**。``save()`` 写的是
``checkpoint.py`` 的同构格式，``load_checkpoint`` 直接可读、参数与槽位逐位
相等，才算真的合并对了。

**为什么元素布局要 reshape 回完整形状而不是留一维。** 合并后的槽位是给"整份
参数"的优化器用的，``Optimizer.load_state_dict`` 校验槽位形状等于参数形状；
留在 flatten 后的样子会被当成"形状不符"拒绝。切分时再 ravel，等价往返。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .checkpoint import _HYPER, _PARAM, _SLOT, _STEP, _atomic_savez, _payload_from
from .sharded import shard_params
from .zero3 import plan_chunks

__all__ = ["ShardedCheckpoint", "merge_shards", "split_checkpoint"]

_META_KEY = "__meta__"

#: 按参数名分片（ZeRO-1 / ZeRO-2）。
_NAMES = "names"
#: 按元素切块（ZeRO-3）。
_ELEMENTS = "elements"


def _with_suffix(path: Any) -> str:
    target = str(path)
    return target if target.endswith(".npz") else target + ".npz"


def _read_shard_file(path: Any) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """读一份分片：meta 解析成字典，数组按 ``kind::name`` 原样取出。"""
    target = _with_suffix(path)
    arrays: Dict[str, np.ndarray] = {}
    with np.load(target, allow_pickle=False) as data:
        if _META_KEY not in data.files:
            raise ValueError(f"{target} 不是分片存档：缺少 {_META_KEY}")
        meta = json.loads(str(data[_META_KEY]))
        for key in data.files:
            if key == _META_KEY:
                continue
            _, sep, name = key.partition("::")
            if not sep or not name:
                raise ValueError(f"{target} 里的键 {key!r} 不是 kind::name 形式")
            arrays[key] = np.array(data[key], copy=True)
    if "plan" in meta:
        meta["layout"] = _ELEMENTS
    elif "assignments" in meta:
        meta["layout"] = _NAMES
    else:
        raise ValueError(
            f"{target} 既没有 plan 也没有 assignments，无法判断分片布局；"
            "这个文件不是 ZeRO 的分片存档"
        )
    return meta, arrays


def _check_common(metas: Sequence[Mapping[str, Any]]) -> int:
    """跨分片的一致性：片数、片号、步数、超参、布局。返回 world_size。"""
    worlds = {int(meta.get("world_size", 0)) for meta in metas}
    if len(worlds) != 1:
        raise ValueError(f"这些分片的 world_size 不一致: {sorted(worlds)}")
    world_size = worlds.pop()
    ranks = sorted(int(meta.get("rank", -1)) for meta in metas)
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"分片里的 rank 有重复: {ranks}")
    if ranks != list(range(world_size)):
        missing = sorted(set(range(world_size)) - set(ranks))
        extra = sorted(set(ranks) - set(range(world_size)))
        raise ValueError(
            f"分片不齐：world_size={world_size} 需要 rank 0..{world_size - 1}，"
            f"缺少 {missing}，越界 {extra}"
        )
    steps = {int(meta.get("step", 0)) for meta in metas}
    if len(steps) != 1:
        raise ValueError(f"这些分片的步数不一致: {sorted(steps)}")
    hypers = {json.dumps(meta.get("hyper") or {}, sort_keys=True) for meta in metas}
    if len(hypers) != 1:
        raise ValueError("这些分片的优化器超参不一致；它们不属于同一条训练轨迹")
    layouts = {str(meta["layout"]) for meta in metas}
    if len(layouts) != 1:
        raise ValueError(f"这些分片的布局不一致: {sorted(layouts)}；不能混合布局合并")
    return world_size


def _slice_elements(
    name: str,
    array: np.ndarray,
    *,
    shape: Tuple[int, ...],
    size: int,
    chunk: int,
    rank: int,
    what: str,
) -> np.ndarray:
    """把一个参数的完整数组按 ``rank`` 切出第 ``rank`` 块（尾部补零）。"""
    flat = np.ascontiguousarray(np.asarray(array)).ravel()
    if flat.size != size:
        raise ValueError(
            f"{what} {name!r} 有 {flat.size} 个元素，与形状 {shape} 的 {size} 个不符"
        )
    piece = np.zeros(chunk, dtype=flat.dtype)
    start = rank * chunk
    stop = min(start + chunk, size)
    if start < stop:
        piece[: stop - start] = flat[start:stop]
    return piece


def _join_elements(
    pieces: Sequence[np.ndarray],
    *,
    shape: Tuple[int, ...],
    size: int,
    chunk: int,
    name: str,
    what: str,
) -> np.ndarray:
    """把 ``world_size`` 块按 rank 顺序拼回完整数组（丢掉尾部补零）。"""
    flat = np.concatenate([np.asarray(piece).ravel() for piece in pieces])
    if flat.size < size:
        raise ValueError(
            f"{what} {name!r} 的分片总长 {flat.size} 小于参数元素数 {size}"
        )
    return flat[:size].reshape(shape)


class ShardedCheckpoint:
    """合并后的完整状态：参数 + 槽位 + 步数 + 超参，以及它是从哪种布局来的。

    ``params`` 在 ``elements`` 布局下必然存在（分片里就带参数）；``names`` 布局
    下只有当调用方把模型权重传给 ``merge_shards(params=...)`` 时才有。
    """

    def __init__(
        self,
        *,
        layout: str,
        world_size: int,
        step: int,
        hyper: Mapping[str, Any],
        params: Optional[Mapping[str, np.ndarray]] = None,
        slots: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None,
        assignments: Optional[Mapping[str, int]] = None,
        plan: Optional[Mapping[str, Tuple[Tuple[int, ...], int]]] = None,
    ) -> None:
        if layout not in (_NAMES, _ELEMENTS):
            raise ValueError(f"不认识的布局 {layout!r}，只支持 {_NAMES!r} / {_ELEMENTS!r}")
        self.layout = layout
        self.world_size = int(world_size)
        self.step = int(step)
        self.hyper: Dict[str, Any] = dict(hyper)
        self.params: Dict[str, np.ndarray] = {
            name: np.array(array, copy=True) for name, array in (params or {}).items()
        }
        self.slots: Dict[str, Dict[str, np.ndarray]] = {
            slot: {name: np.array(array, copy=True) for name, array in by_name.items()}
            for slot, by_name in (slots or {}).items()
        }
        self.assignments: Dict[str, int] = {
            name: int(owner) for name, owner in (assignments or {}).items()
        }
        self.plan: Dict[str, Tuple[Tuple[int, ...], int]] = {
            name: (tuple(int(dim) for dim in shape), int(chunk))
            for name, (shape, chunk) in (plan or {}).items()
        }

    # -- 输出 ---------------------------------------------------------

    def save(self, path: Any, *, meta: Optional[Mapping[str, object]] = None) -> str:
        """写成一份完整 checkpoint（与 ``save_checkpoint`` 同构）。

        没有参数时报错：``names`` 布局的存档本来只存槽位，"合并出权重"这件事
        必须由调用方把模型权重一起交进来——这里不做"用零填上"的猜测。
        """
        if not self.params:
            raise ValueError(
                "这份合并结果里没有参数：按名分片的存档（ZeRO-1/2）只存槽位，"
                "请把模型权重作为 params= 传给 merge_shards()；"
                "只有槽位的话别用 save()，用 write_shards() 重新分片"
            )
        target = str(path)
        if not target.endswith(".npz"):
            raise ValueError(f"存档后缀必须是 .npz，得到 {target!r}")
        optimizer_state = {
            "step": self.step,
            "hyper": dict(self.hyper),
            "slots": self.slots,
        }
        payload = _payload_from(self.params, optimizer_state, None, meta)
        _atomic_savez(target, payload)
        return target

    def split(
        self, world_size: Optional[int] = None, *, layout: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """按目标 ``world_size`` / ``layout`` 切成每 rank 一份的存档内容。

        返回的每个字典与 ``save_shard`` 写入的内容同构，既能直接交给
        ``_write_state`` 落盘，也能被调用方读走做别的事。
        """
        target_world = self.world_size if world_size is None else world_size
        if isinstance(target_world, bool) or not isinstance(target_world, int):
            raise TypeError(f"world_size 必须是 int，收到 {target_world!r}")
        if target_world < 1:
            raise ValueError(f"world_size 必须 >= 1，收到 {target_world}")
        target_layout = self.layout if layout is None else layout
        if target_layout not in (_NAMES, _ELEMENTS):
            raise ValueError(f"不认识的布局 {target_layout!r}")
        if target_layout == _ELEMENTS:
            return self._split_elements(target_world)
        return self._split_names(target_world)

    def write_shards(
        self,
        pattern: str,
        world_size: Optional[int] = None,
        *,
        layout: Optional[str] = None,
    ) -> List[str]:
        """把每个 rank 的分片写到 ``pattern``（必须含 ``{rank}``），返回路径。"""
        if "{rank}" not in str(pattern):
            raise ValueError(
                f"命名模板 {pattern!r} 里没有 {{rank}}：每个 rank 会覆盖同一个文件"
            )
        states = self.split(world_size, layout=layout)
        return [_write_state(state, str(pattern).format(rank=state["rank"])) for state in states]

    # -- 内部 ---------------------------------------------------------

    def _split_names(self, world_size: int) -> List[Dict[str, Any]]:
        """按参数名重新分片（ZeRO-1 / ZeRO-2 的布局）。"""
        if self.params:
            buckets = shard_params(self.params, world_size)
            owners = {name: index for index, bucket in enumerate(buckets) for name in bucket}
        elif world_size == self.world_size and self.assignments:
            owners = dict(self.assignments)
        else:
            raise ValueError(
                "换 world_size 重新按名分片需要完整参数表（按字节均衡的归属由形状"
                "决定），请把模型权重作为 params= 传给 merge_shards()"
            )
        states: List[Dict[str, Any]] = []
        for rank in range(world_size):
            owned = [name for name, owner in owners.items() if owner == rank]
            slots: Dict[str, Dict[str, np.ndarray]] = {}
            for slot, by_name in self.slots.items():
                missing = [name for name in owned if name not in by_name]
                if missing:
                    raise KeyError(
                        f"槽位 {slot!r} 缺少这些参数的数组: {missing}；"
                        "分片存档不完整，不能重新分片"
                    )
                slots[slot] = {name: np.array(by_name[name], copy=True) for name in owned}
            states.append(
                {
                    "layout": _NAMES,
                    "rank": rank,
                    "world_size": world_size,
                    "step": self.step,
                    "hyper": dict(self.hyper),
                    "slots": slots,
                    "assignments": {name: rank for name in owned},
                }
            )
        return states

    def _split_elements(self, world_size: int) -> List[Dict[str, Any]]:
        """按元素重新切块（ZeRO-3 的布局）。"""
        if not self.params:
            raise ValueError(
                "元素级分片必须有完整参数表：分片合并出的参数缺失，"
                "请把模型权重作为 params= 传给 merge_shards()"
            )
        plan = plan_chunks(self.params, world_size)
        states: List[Dict[str, Any]] = []
        for rank in range(world_size):
            params: Dict[str, np.ndarray] = {}
            slots: Dict[str, Dict[str, np.ndarray]] = {}
            for name, (shape, size, chunk) in plan.items():
                params[name] = _slice_elements(
                    name,
                    self.params[name],
                    shape=shape,
                    size=size,
                    chunk=chunk,
                    rank=rank,
                    what="参数",
                )
                for slot, by_name in self.slots.items():
                    if name not in by_name:
                        raise KeyError(
                            f"槽位 {slot!r} 缺少参数 {name!r} 的数组；分片存档不完整"
                        )
                    slots.setdefault(slot, {})[name] = _slice_elements(
                        name,
                        by_name[name],
                        shape=shape,
                        size=size,
                        chunk=chunk,
                        rank=rank,
                        what=f"槽位 {slot!r}",
                    )
            states.append(
                {
                    "layout": _ELEMENTS,
                    "rank": rank,
                    "world_size": world_size,
                    "step": self.step,
                    "hyper": dict(self.hyper),
                    "params": params,
                    "slots": slots,
                    "plan": {
                        name: [list(plan[name][0]), plan[name][2]] for name in plan
                    },
                }
            )
        return states


def _write_state(state: Mapping[str, Any], path: Any) -> str:
    """把一份"每 rank 的存档内容"写成 npz（格式与 ``save_shard`` 完全一致）。"""
    target = _with_suffix(path)
    arrays: Dict[str, np.ndarray] = {}
    if state["layout"] == _ELEMENTS:
        for name, array in state["params"].items():
            arrays[f"param::{name}"] = np.asarray(array)
        meta = {
            "rank": state["rank"],
            "world_size": state["world_size"],
            "step": state["step"],
            "hyper": state["hyper"],
            "plan": state["plan"],
        }
    else:
        meta = {
            "rank": state["rank"],
            "world_size": state["world_size"],
            "step": state["step"],
            "hyper": state["hyper"],
            "assignments": state["assignments"],
        }
    for slot, by_name in state["slots"].items():
        for name, array in by_name.items():
            arrays[f"{slot}::{name}"] = np.asarray(array)
    arrays[_META_KEY] = np.array(json.dumps(meta, ensure_ascii=False))
    _atomic_savez(target, arrays)
    return target


def merge_shards(
    paths: Sequence[Any],
    *,
    params: Optional[Mapping[str, np.ndarray]] = None,
) -> ShardedCheckpoint:
    """把 N 个分片合成一份完整状态。

    ``params`` 是可选的模型权重：``elements`` 布局会拿它当"分片拼出来的参数
    是否对得上"的 oracle（不一致说明分片来自另一份权重）；``names`` 布局则必
    须靠它才能写出完整 checkpoint（按名分片的存档不含参数）。
    """
    targets = [_with_suffix(path) for path in paths]
    if not targets:
        raise ValueError("没有给任何分片文件")
    loaded = [_read_shard_file(target) for target in targets]
    metas = [meta for meta, _ in loaded]
    world_size = _check_common(metas)
    layout = str(metas[0]["layout"])
    step = int(metas[0]["step"])
    hyper = dict(metas[0].get("hyper") or {})
    if layout == _ELEMENTS:
        return _merge_elements(metas, [arrays for _, arrays in loaded], world_size, step, hyper, params)
    return _merge_names(metas, [arrays for _, arrays in loaded], world_size, step, hyper, params)


def _merge_names(
    metas: Sequence[Mapping[str, Any]],
    arrays: Sequence[Mapping[str, np.ndarray]],
    world_size: int,
    step: int,
    hyper: Mapping[str, Any],
    params: Optional[Mapping[str, np.ndarray]],
) -> ShardedCheckpoint:
    assignments: Dict[str, int] = {}
    for meta in metas:
        rank = int(meta["rank"])
        for name, owner in dict(meta["assignments"]).items():
            if int(owner) != rank:
                raise ValueError(
                    f"rank {rank} 的存档里把 {name!r} 记在了 rank {owner} 名下；"
                    "分片自相矛盾，不能合并"
                )
            if name in assignments:
                raise ValueError(
                    f"参数 {name!r} 同时出现在多个分片的归属表里"
                    f"（rank {assignments[name]} 与 rank {rank}）"
                )
            assignments[name] = rank

    slots: Dict[str, Dict[str, np.ndarray]] = {}
    for rank, by_key in enumerate(arrays):
        for key, array in by_key.items():
            slot, sep, name = key.partition("::")
            if not sep:
                raise ValueError(f"分片 {rank} 里的键 {key!r} 不完整")
            if slot == "param":
                raise ValueError(
                    f"分片 {rank} 里有参数数组 {name!r}：按名分片的存档不该带参数；"
                    "它可能是元素布局（ZeRO-3）的存档，请勿混用"
                )
            if name not in assignments:
                raise KeyError(
                    f"分片 {rank} 的槽位 {slot!r} 里有 {name!r}，但它不在归属表里"
                )
            if name in slots.get(slot, {}):
                raise ValueError(f"槽位 {slot!r} 的 {name!r} 在两个分片里都有")
            slots.setdefault(slot, {})[name] = array

    for slot, by_name in slots.items():
        missing = sorted(set(assignments) - set(by_name))
        if missing:
            raise KeyError(f"槽位 {slot!r} 缺少这些参数的数组: {missing}；分片不完整")

    full_params: Dict[str, np.ndarray] = {}
    if params is not None:
        given = {name: np.asarray(array) for name, array in params.items()}
        if set(given) != set(assignments):
            raise KeyError(
                f"传入的参数表与分片不一致：多 {sorted(set(given) - set(assignments))}，"
                f"少 {sorted(set(assignments) - set(given))}"
            )
        for slot, by_name in slots.items():
            for name, array in by_name.items():
                if tuple(np.shape(array)) != tuple(given[name].shape):
                    raise ValueError(
                        f"槽位 {slot!r} 的 {name!r} 形状 {np.shape(array)} 与参数 "
                        f"{given[name].shape} 不一致"
                    )
        full_params = given

    return ShardedCheckpoint(
        layout=_NAMES,
        world_size=world_size,
        step=step,
        hyper=hyper,
        params=full_params,
        slots=slots,
        assignments=assignments,
    )


def _merge_elements(
    metas: Sequence[Mapping[str, Any]],
    arrays: Sequence[Mapping[str, np.ndarray]],
    world_size: int,
    step: int,
    hyper: Mapping[str, Any],
    params: Optional[Mapping[str, np.ndarray]],
) -> ShardedCheckpoint:
    plan: Dict[str, Tuple[Tuple[int, ...], int]] = {}
    for meta in metas:
        current = {
            name: (tuple(int(dim) for dim in shape), int(chunk))
            for name, (shape, chunk) in dict(meta["plan"]).items()
        }
        if not plan:
            plan = current
        elif current != plan:
            diff = sorted(
                name
                for name in set(plan) | set(current)
                if plan.get(name) != current.get(name)
            )
            raise ValueError(
                f"这些分片的块布局不一致（{diff}）；它们不是同一次训练写下的"
            )

    merged_params: Dict[str, np.ndarray] = {}
    for name, (shape, chunk) in plan.items():
        size = int(np.prod(shape, dtype=np.int64)) if shape else 1
        pieces = []
        dtype = None
        for rank, by_key in enumerate(arrays):
            key = f"param::{name}"
            if key not in by_key:
                raise KeyError(f"rank {rank} 的分片里没有参数 {name!r}")
            piece = by_key[key]
            if piece.shape != (chunk,):
                raise ValueError(
                    f"rank {rank} 的 {name!r} 分片形状 {piece.shape} 与块长 ({chunk},) 不一致"
                )
            if dtype is None:
                dtype = piece.dtype
            elif piece.dtype != dtype:
                raise TypeError(
                    f"{name!r} 的分片 dtype 不一致：{dtype} 与 {piece.dtype}"
                )
            pieces.append(piece)
        merged_params[name] = _join_elements(
            pieces, shape=shape, size=size, chunk=chunk, name=name, what="参数"
        )

    slots: Dict[str, Dict[str, np.ndarray]] = {}
    slot_names = {
        key.partition("::")[0]
        for by_key in arrays
        for key in by_key
        if key.partition("::")[0] != "param"
    }
    for rank, by_key in enumerate(arrays):
        present = {key.partition("::")[0] for key in by_key if key.partition("::")[0] != "param"}
        if present != slot_names:
            raise ValueError(
                f"rank {rank} 的槽位集合 {sorted(present)} 与其它分片 "
                f"{sorted(slot_names)} 不一致"
            )
    for slot in sorted(slot_names):
        merged: Dict[str, np.ndarray] = {}
        for name, (shape, chunk) in plan.items():
            size = int(np.prod(shape, dtype=np.int64)) if shape else 1
            pieces = []
            for rank, by_key in enumerate(arrays):
                key = f"{slot}::{name}"
                if key not in by_key:
                    raise KeyError(f"rank {rank} 的槽位 {slot!r} 里没有参数 {name!r}")
                piece = by_key[key]
                if piece.shape != (chunk,):
                    raise ValueError(
                        f"rank {rank} 的槽位 {slot!r} / {name!r} 形状 {piece.shape} "
                        f"与块长 ({chunk},) 不一致"
                    )
                pieces.append(piece)
            merged[name] = _join_elements(
                pieces, shape=shape, size=size, chunk=chunk, name=name, what=f"槽位 {slot!r}"
            )
        slots[slot] = merged

    if params is not None:
        given = {name: np.asarray(array) for name, array in params.items()}
        if set(given) != set(merged_params):
            raise KeyError(
                f"传入的参数表与分片不一致：多 {sorted(set(given) - set(merged_params))}，"
                f"少 {sorted(set(merged_params) - set(given))}"
            )
        for name, array in merged_params.items():
            if tuple(given[name].shape) != tuple(array.shape):
                raise ValueError(
                    f"传入的 {name!r} 形状 {given[name].shape} 与分片拼出的 "
                    f"{array.shape} 不一致"
                )
            if not np.array_equal(given[name], array):
                raise ValueError(
                    f"传入的 {name!r} 与分片拼出的数值不同：分片来自另一份权重，"
                    "别把它们合并到一起"
                )

    return ShardedCheckpoint(
        layout=_ELEMENTS,
        world_size=world_size,
        step=step,
        hyper=hyper,
        params=merged_params,
        slots=slots,
        plan=plan,
    )


def _read_full_checkpoint(path: Any) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]], int, Dict[str, Any]]:
    """读一份普通 checkpoint 的参数 / 槽位 / 步数 / 超参（忽略流式状态与 meta）。"""
    source = str(path)
    params: Dict[str, np.ndarray] = {}
    slots: Dict[str, Dict[str, np.ndarray]] = {}
    hyper: Dict[str, Any] = {}
    step = 0
    with np.load(source) as data:
        for key in data.files:
            if key.startswith(_PARAM):
                params[key[len(_PARAM):]] = np.array(data[key], copy=True)
            elif key.startswith(_SLOT):
                rest = key[len(_SLOT):]
                slot, sep, name = rest.partition("/")
                if not sep or not name:
                    raise KeyError(f"存档里的槽位条目 {key!r} 不完整")
                slots.setdefault(slot, {})[name] = np.array(data[key], copy=True)
            elif key == _STEP:
                step = int(data[key])
            elif key.startswith(_HYPER):
                hyper[key[len(_HYPER):]] = data[key].item()
    if not params:
        raise ValueError(f"{source} 里没有参数条目（{_PARAM}*），不是一份完整 checkpoint")
    return params, slots, step, hyper


def split_checkpoint(
    path: Any,
    world_size: int,
    pattern: str,
    *,
    layout: str = _ELEMENTS,
) -> List[str]:
    """把一份完整 checkpoint 按 ``world_size`` 切成每 rank 一个分片文件。

    ``layout="elements"`` 对应 ZeRO-3（写出的文件带参数分片），
    ``layout="names"`` 对应 ZeRO-1/2（只切槽位，参数仍由各 rank 自己的模型提供）。
    两种写出的文件都能被对应类的 ``load_shard`` 直接读回。
    """
    if os.path.isdir(str(path)):
        raise ValueError(f"{path} 是目录，不是 checkpoint 文件")
    params, slots, step, hyper = _read_full_checkpoint(path)
    checkpoint = ShardedCheckpoint(
        layout=layout,
        world_size=int(world_size),
        step=step,
        hyper=hyper,
        params=params,
        slots=slots,
        # 归属/块布局都由目标 world_size 现算，于是"换了卡数也能接上"。
        assignments=None,
        plan=None,
    )
    return checkpoint.write_shards(pattern, world_size, layout=layout)
