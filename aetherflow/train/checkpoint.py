"""权重与训练状态的存档（``.npz``）。

三条刻意的约束：

1. **参数与优化器状态分开存，但存在同一个文件里。** 参数属于模型，
   动量/一阶矩/二阶矩属于"这次训练走到哪了"。分成两个文件会让
   "只存权重"变成"忘了存步数"，而 Adam 的偏差修正恰恰依赖步数：
   少存一个 ``step``，续训第一发的更新量会被放大一个量级。
2. **加载是原地覆盖，不是替换引用。** 优化器持有的是 ``store.arrays()``
   返回的同一批数组；只要有一个参数在加载时被换成新对象，优化器就会
   继续更新那份没人用的旧数组，训练看起来在跑、实际参数永远停在存档值。
3. **对不上就报错。** 参数集合、形状、dtype 任何一处不一致都直接失败，
   不做"能恢复多少恢复多少"的尽力而为——那种静默降级最难查。

文件布局（键是字符串，``/`` 只作层级分隔）：``param/<名字>`` 存参数，
``slot/<槽位>/<名字>`` 存优化器槽位，``optim/step`` 存步数，
``state/<名字>`` 存流式推理状态（可选的 Mamba 卷积窗口 / SSM 状态），
``meta/<键>`` 存调用方自己的标量（epoch、验证集指标、配置摘要等）。

**流式状态为什么和权重存在同一个文件里**：状态只在"这份权重"下有定义——
`h` 的通道布局、`conv` 的窗口长度都由模型形状决定。分开存，现场就有机会把
A 模型的状态喂给 B 模型的权重，而形状检查只在解码第一步才炸；存一起则
``.npz`` 里参数与状态天然同源。四类条目的加载规则对称（见 ``load_checkpoint``）：
传了目标却缺存档就报错，没传目标则忽略存档里的那一类——"只加载权重"始终是合法场景。
"""

from __future__ import annotations

import os
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Mapping, Optional

import numpy as np

from ._arrays import array_module, to_host
from .optim import Optimizer
from .params import ParameterStore

__all__ = ["save_checkpoint", "load_checkpoint", "inspect_checkpoint", "TrainingSnapshot"]


_PARAM = "param/"
_SLOT = "slot/"
_STATE = "state/"
_META = "meta/"
_STEP = "optim/step"
_HYPER = "optim/hyper/"
_AUX = "aux/"


def _scalar_of(key: str, value) -> np.ndarray:
    """把 meta 值收成 0 维数组：存档只保留能跨语言读回的标量。"""
    arr = np.asarray(value)
    if arr.ndim != 0:
        raise TypeError(
            f"meta[{key!r}] 必须是标量（epoch / 指标 / 超参），得到形状 {arr.shape}；"
            "列表请自行展开成多个键"
        )
    if arr.dtype == object:
        raise TypeError(f"meta[{key!r}] 是 Python 对象（{type(value).__name__}），无法安全序列化")
    return arr


def _atomic_savez(target: str, payload: Mapping[str, np.ndarray]) -> None:
    """写同目录的临时文件，落盘后再原子替换目标。

    ``np.savez(target)`` 直接写目标：写到一半被 kill / 断电，留下的是一个
    打不开的 ``.npz``——名字却和上一份好存档一样，下次加载才发现权重没了。
    临时文件必须放在**同一目录**：``os.replace`` 只在同一文件系统内原子，
    跨盘会退化成"先删后拷"，等于没有保护。
    """
    folder = os.path.dirname(os.path.abspath(target))
    fd, tmp = tempfile.mkstemp(prefix=".ckpt-", suffix=".npz", dir=folder)
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez(fh, **payload)
            fh.flush()
            os.fsync(fh.fileno())  # 数据先落地，再换名字；否则断电后是空文件
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _payload_from(
    params: Mapping[str, np.ndarray],
    optimizer_state: Optional[Mapping] = None,
    state: Optional[Mapping[str, np.ndarray]] = None,
    meta: Optional[Mapping[str, object]] = None,
    aux: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """把"已落回主机的参数 + 优化器状态 + 流式状态 + 附加数组 + meta"拼成 npz 载荷。

    ``save_checkpoint`` 与 ``TrainingSnapshot.save`` 共用这一段：两条路
    唯一的差别是"参数从哪里来"（前者读 ``store``，后者读快照），存的格式
    必须逐字节同构，否则同一份训练状态会有两种读法。

    ``aux`` 是"属于这次训练、但不属于参数/槽位/流式状态"的数组，目前只有
    梯度累积窗口里尚未提交的累加和。它单独占一个前缀，是为了让"只加载权重"
    与"续接训练窗口"成为两件可以分别选择的事。
    """
    payload: Dict[str, np.ndarray] = {}
    for name, arr in params.items():
        payload[_PARAM + name] = arr

    if optimizer_state is not None:
        payload[_STEP] = np.asarray(int(optimizer_state["step"]), dtype=np.int64)
        for slot, arrays in optimizer_state["slots"].items():
            for name, arr in arrays.items():
                payload[f"{_SLOT}{slot}/{name}"] = arr
        for key, value in (optimizer_state.get("hyper") or {}).items():
            payload[_HYPER + str(key)] = _scalar_of(str(key), value)

    for name, arr in (state or {}).items():
        key = str(name)
        if not key:
            raise ValueError("流式状态的键不能为空")
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"流式状态 {key!r} 必须是 numpy 数组，得到 {type(arr).__name__}"
            )
        payload[_STATE + key] = arr

    for name, arr in (aux or {}).items():
        key = str(name)
        if not key:
            raise ValueError("附加数组的键不能为空")
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"附加数组 {key!r} 必须是 numpy 数组，得到 {type(arr).__name__}"
            )
        payload[_AUX + key] = arr

    for key, value in (meta or {}).items():
        payload[_META + str(key)] = _scalar_of(str(key), value)
    return payload


def save_checkpoint(
    path,
    store: ParameterStore,
    optimizer: Optional[Optimizer] = None,
    meta: Optional[Mapping[str, object]] = None,
    state: Optional[Mapping[str, np.ndarray]] = None,
) -> None:
    """把参数（可选的优化器状态、流式状态与 meta）写成一个 ``.npz`` 文件。

    ``optimizer`` 为 ``None`` 时不写槽位，只写参数；之后可以只加载权重
    做推理，但**不能**用它续训。

    ``state`` 是流式推理的跨调用状态（Mamba 每层一对 ``conv`` / ``h``），
    键用**图输入名**（``l0_conv0`` / ``l0_h0``，即
    ``MambaModelSpec.state_shapes``），这样加载后可以不经转换直接
    ``feed.update(state)``。值为 ``numpy`` 数组；形状与 dtype 在加载时由
    调用方给的模板校验，保存端不做猜测。
    """
    target = str(path)
    if not target.endswith(".npz"):
        raise ValueError(f"存档后缀必须是 .npz，得到 {target!r}")

    # 存档格式与设备无关：设备数组在这里落回主机。
    # 注意设备态训练（``Trainer(device_state=True)``）期间 ``store`` 是上传
    # 时的旧值，存档前请先 ``trainer.sync_to_host()``——或者直接用
    # ``trainer.save_checkpoint(...)``，它在 step 边界取快照，不要求先同步。
    params = {name: to_host(arr) for name, arr in store.arrays().items()}
    optimizer_state = optimizer.state_dict() if optimizer is not None else None
    _atomic_savez(target, _payload_from(params, optimizer_state, state, meta))


# 写盘线程只有一条：存档是 I/O 密集、彼此独立，但同一时刻开多条只会让
# 磁盘排队，反而把"主线程继续训练"的时间窗口挤回去。
_SNAPSHOT_WRITER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aetherflow-ckpt")


def _host_copy_of(arr) -> np.ndarray:
    """快照里的数组必须是主机副本：设备数组同理，先 D2H 再拷一次。"""
    return np.array(to_host(arr), copy=True)


class TrainingSnapshot:
    """某一 step 边界上的训练状态，全部是主机副本。

    为什么需要这个中间对象：设备态训练期间参数与优化器槽位都在显存里，
    直接调 ``save_checkpoint`` 要求先 ``sync_to_host()``——而那一步会覆盖
    ``store``（训练期间它按契约是旧值），主机态下也不安全：后台线程写盘
    时主线程仍在原地更新参数。

    快照在 step 边界把两类数组一次性 D2H + 拷贝，之后写盘与训练不再共享
    任何一块内存。**异步的是写盘，不是拷贝**：拷贝必须发生在调用方选定的
    时刻（通常是一步结束），否则拿到的会是一份跨若干步拼出来的权重。
    """

    def __init__(
        self,
        params: Mapping[str, np.ndarray],
        optimizer_state: Optional[Mapping] = None,
        *,
        steps: Optional[int] = None,
        meta: Optional[Mapping[str, object]] = None,
        aux: Optional[Mapping[str, np.ndarray]] = None,
    ) -> None:
        if not params:
            raise ValueError("快照里没有参数")
        self.params: Dict[str, np.ndarray] = {
            name: _host_copy_of(arr) for name, arr in params.items()
        }
        self._has_optimizer = optimizer_state is not None
        state = dict(optimizer_state or {})
        self.step = int(state.get("step", 0))
        self.slots: Dict[str, Dict[str, np.ndarray]] = {
            slot: {name: _host_copy_of(arr) for name, arr in arrays.items()}
            for slot, arrays in (state.get("slots") or {}).items()
        }
        self.hyper: Dict[str, object] = dict(state.get("hyper") or {})
        self.steps = None if steps is None else int(steps)
        self.meta: Dict[str, object] = dict(meta or {})
        # 附加数组（梯度累积窗口）已经在 ``state_dict`` 里拷过一次主机副本，
        # 这里再拷一次是为了与参数/槽位共享同一条"快照与训练不共享内存"的契约。
        self.aux: Dict[str, np.ndarray] = {
            name: _host_copy_of(arr) for name, arr in (aux or {}).items()
        }

    def state_dict(self) -> Optional[dict]:
        """回放给 ``_payload_from`` 的优化器部分；没带优化器时是 ``None``。"""
        if not self._has_optimizer:
            return None
        return {"step": self.step, "hyper": self.hyper, "slots": self.slots}

    def save(self, path) -> None:
        """同步落盘。格式与 ``save_checkpoint`` 完全一致。"""
        target = str(path)
        if not target.endswith(".npz"):
            raise ValueError(f"存档后缀必须是 .npz，得到 {target!r}")
        meta = dict(self.meta)
        # 训练步数与优化器步数可以不等（``grad_hook`` 返回 None 的那些步照常
        # 前进但不更新参数）。两个都记下来，续训时才知道"这一步到底更没更"。
        if self.steps is not None and "trainer_steps" not in meta:
            meta["trainer_steps"] = self.steps
        _atomic_savez(
            target, _payload_from(self.params, self.state_dict(), None, meta, self.aux)
        )

    def save_async(self, path) -> Future:
        """把 ``save`` 交给后台线程，立刻返回 ``Future``。

        主线程可以继续 ``step``：快照已经拷完，后台线程读的是自己的副本。
        失败（目录不存在、磁盘满）不会在这里抛，而是落在返回的 ``Future``
        上——``future.result()`` 取异常，``future.done()`` 轮询完成。
        """
        return _SNAPSHOT_WRITER.submit(self.save, path)

    def __repr__(self) -> str:
        return (
            f"TrainingSnapshot(step={self.step}, steps={self.steps}, "
            f"params={len(self.params)})"
        )


def inspect_checkpoint(path) -> Dict[str, object]:
    """只读地取出存档的 meta 与附加数组键名，不触碰任何目标对象。

    续训的配置校验必须在**提交之前**完成：``load_checkpoint`` 一旦通过，
    参数与优化器状态就已经被覆盖。若"累积窗口长度对不上"之类的检查排在它
    之后，现场会留下"权重已是新存档、训练状态还是旧的"半加载状态。这里
    只读 ``meta/`` 与 ``aux/`` 两类条目，让调用方先判断"这是不是同一条轨迹"。

    返回 ``{"meta": {...}, "aux": [键名, ...]}``；键名不含 ``aux/`` 前缀。
    """
    meta: Dict[str, object] = {}
    aux: list = []
    with np.load(str(path)) as data:
        for key in data.files:
            if key.startswith(_META):
                meta[key[len(_META):]] = data[key].item()
            elif key.startswith(_AUX):
                name = key[len(_AUX):]
                if not name:
                    raise KeyError(f"存档里的附加数组条目 {key!r} 不完整")
                aux.append(name)
    return {"meta": meta, "aux": sorted(aux)}


def load_checkpoint(
    path,
    store: ParameterStore,
    optimizer: Optional[Optimizer] = None,
    state: Optional[Mapping[str, np.ndarray]] = None,
    extra: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, object]:
    """从 ``.npz`` 读回参数（可选的优化器状态、流式状态），返回 meta 字典。

    不传 ``optimizer`` 时槽位被忽略（只加载权重做推理是合法场景）；
    传了 ``optimizer`` 但存档里没有槽位，说明存档不是续训用的，直接报错。

    ``extra`` 是附加数组的**模板**（目前只有梯度累积窗口的累加和）。与
    ``state`` 不同：存档里没有 ``aux/`` 条目是**正常**的（多数存档落在窗口
    边界上），这时模板原样保留；有则要求键集合与形状完全一致，并原地写入。
    传了 ``extra`` 却遇到残缺条目会报错，不会静默从空窗口开始。

    ``state`` 是一份**模板**：键与形状取自 ``MambaModelSpec.state_shapes``，
    通常就是全零的状态字典（

        ``{name: np.zeros(shape, np.float32) for name, shape in spec.state_shapes.items()}``

    ）。加载时原地写入这份模板并校验形状 / dtype，返回值不带状态——调用方手里
    的字典就是续接用的 feed，不需要再多一次拷贝或改名。同理，传了 ``state``
    但存档里没有状态条目会报错（否则会静默从零开始，历史白白丢掉）；
    不传则忽略存档里的状态。
    """
    source = str(path)
    params: Dict[str, str] = {}
    slots: Dict[str, Dict[str, np.ndarray]] = {}
    saved_state: Dict[str, np.ndarray] = {}
    saved_aux: Dict[str, np.ndarray] = {}
    hyper: Dict[str, object] = {}
    meta: Dict[str, object] = {}
    step = 0
    has_step = False

    with np.load(source) as data:
        for key in data.files:
            if key.startswith(_PARAM):
                params[key[len(_PARAM):]] = key
            elif key.startswith(_SLOT):
                rest = key[len(_SLOT):]
                slot, sep, name = rest.partition("/")
                if not sep or not name:
                    raise KeyError(f"存档里的槽位条目 {key!r} 不完整")
                # 必须在 with 内读成数组：NpzFile 关闭后再取会失败
                slots.setdefault(slot, {})[name] = np.array(data[key], copy=True)
            elif key.startswith(_STATE):
                name = key[len(_STATE):]
                if not name:
                    raise KeyError(f"存档里的状态条目 {key!r} 不完整")
                saved_state[name] = np.array(data[key], copy=True)
            elif key == _STEP:
                step = int(data[key])
                has_step = True
            elif key.startswith(_HYPER):
                name = key[len(_HYPER):]
                if not name:
                    raise KeyError(f"存档里的超参条目 {key!r} 不完整")
                hyper[name] = data[key].item()
            elif key.startswith(_AUX):
                name = key[len(_AUX):]
                if not name:
                    raise KeyError(f"存档里的附加数组条目 {key!r} 不完整")
                saved_aux[name] = np.array(data[key], copy=True)
            elif key.startswith(_META):
                meta[key[len(_META):]] = data[key].item()
            else:
                raise KeyError(f"存档里出现未知条目 {key!r}")

        saved = set(params)
        have = set(store.names)
        if saved != have:
            missing = sorted(have - saved)
            extra = sorted(saved - have)
            raise KeyError(
                f"存档与当前模型的参数集合不一致：缺少 {missing}，多出 {extra}。"
                "通常是模型结构或输入名改了；请确认这是同一个模型再加载"
            )

        # 先把所有参数读进内存并校验，**不写回**。参数写入必须放在所有检查
        # （含优化器状态）之后：否则一个坏存档会先把权重改掉、再在优化器
        # 检查处抛错，留下"参数已变、训练状态没变"的半加载现场。
        loaded_params: Dict[str, np.ndarray] = {}
        for name, key in params.items():
            target = store[name]
            loaded = np.array(data[key], copy=True)
            if loaded.shape != target.shape:
                raise ValueError(
                    f"参数 {name} 的形状 {loaded.shape} 与当前模型 {target.shape} 不一致"
                )
            if loaded.dtype != target.dtype:
                raise TypeError(
                    f"参数 {name} 的存档类型 {loaded.dtype} 与当前模型 {target.dtype} 不一致；"
                    "静默转换会悄悄改变数值精度"
                )
            loaded_params[name] = loaded

    checked_state: Dict[str, np.ndarray] = {}
    if state is not None:
        if not saved_state:
            raise ValueError(
                "存档里没有流式状态（保存时未传 state），无法从这段历史续接；"
                "只加载权重请把 state 留空"
            )
        saved_names = set(saved_state)
        want_names = set(state)
        if saved_names != want_names:
            raise KeyError(
                "存档与当前状态的键集合不一致：缺少 "
                f"{sorted(want_names - saved_names)}，多出 "
                f"{sorted(saved_names - want_names)}。通常是层数或状态命名改了"
            )
        for name, loaded in saved_state.items():
            target = state[name]
            if not isinstance(target, np.ndarray):
                raise TypeError(
                    f"状态模板 {name!r} 必须是 numpy 数组，得到 {type(target).__name__}"
                )
            if loaded.shape != target.shape:
                raise ValueError(
                    f"状态 {name} 的形状 {loaded.shape} 与当前模型 {target.shape} 不一致"
                )
            if loaded.dtype != target.dtype:
                raise TypeError(
                    f"状态 {name} 的存档类型 {loaded.dtype} 与当前模型 {target.dtype} 不一致；"
                    "静默转换会悄悄改变数值精度"
                )
            checked_state[name] = loaded

    checked_extra: Dict[str, np.ndarray] = {}
    if extra is not None and saved_aux:
        if set(saved_aux) != set(extra):
            raise KeyError(
                "存档里的附加数组与当前模板的键集合不一致：缺少 "
                f"{sorted(set(extra) - set(saved_aux))}，多出 "
                f"{sorted(set(saved_aux) - set(extra))}"
            )
        for name, loaded in saved_aux.items():
            target = extra[name]
            if not isinstance(target, np.ndarray):
                raise TypeError(
                    f"附加数组模板 {name!r} 必须是 numpy 数组，得到 {type(target).__name__}"
                )
            if loaded.shape != target.shape:
                raise ValueError(
                    f"附加数组 {name} 的形状 {loaded.shape} 与当前模板 {target.shape} 不一致"
                )
            if loaded.dtype != target.dtype:
                raise TypeError(
                    f"附加数组 {name} 的存档类型 {loaded.dtype} 与当前模板 {target.dtype} 不一致"
                )
            checked_extra[name] = loaded

    # 待提交的数组也在提交前全部准备好：设备态下 ``asarray`` 要上传到显存，
    # 那一步同样可能失败。把它挪到这里，提交之后只剩不会失败的原地赋值。
    prepared_params = {
        name: array_module(store[name]).asarray(loaded, dtype=store[name].dtype)
        for name, loaded in loaded_params.items()
    }
    prepared_state = {
        name: array_module(state[name]).asarray(loaded, dtype=state[name].dtype)
        for name, loaded in checked_state.items()
    }

    # 优化器的提交必须排在**所有校验之后**：它自己虽然也是"先校验后提交"，
    # 但如果它先提交、随后流式状态校验才失败，现场就变成"优化器已恢复、
    # 参数与状态还是旧的"——续训第一发的动量与权重对不上，比不加载更难查。
    if optimizer is not None:
        if not has_step:
            raise ValueError(
                "存档里没有优化器状态（保存时未传 optimizer），无法续训；"
                "只加载权重请把 optimizer 留空"
            )
        optimizer.load_state_dict(
            {"step": step, "slots": slots, **({"hyper": hyper} if hyper else {})}
        )

    for name, prepared in prepared_params.items():
        target = store[name]
        # 原地写入：优化器手里是这个数组本身，替换对象会让它更新一份没人用的副本
        target[...] = prepared

    for name, prepared in prepared_state.items():
        # 同样原地写：调用方手里的状态字典直接就是下一次 decode 的 feed
        state[name][...] = prepared

    for name, loaded in checked_extra.items():
        extra[name][...] = loaded
    return meta
