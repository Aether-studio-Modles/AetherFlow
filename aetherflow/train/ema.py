"""参数滑动平均（EMA）。

为什么放在训练层而不是优化器里：EMA 不改"参数怎么走一步"，它维护的是**另一份
用于推理的权重**。塞进优化器会让"一个优化器对应一组参数"变成"对应两组数组"，
存档时也就分不清哪份才是要接着训的。

两条约束：

1. **初始化即复制，不用零初始化。** 零初始化的影子在前 ``1/(1-decay)`` 步里
   严重偏向 0（decay=0.999 时是前一千步），必须配偏差修正才能用；直接复制参数
   则从第一刻起就无偏，也不需要额外的步数修正。
2. **``update`` 必须在 ``optimizer.step`` 之后。** EMA 跟踪的是"更新后的参数"；
   顺序反了就会稳定地落后一步，而且这个偏差不会随训练消失。

影子用 float32 存：f16 参数做指数平均时，``(1-decay)*param`` 这种小量在 f16 上
会被舍入归零，平均结果退化成几乎不动。
"""

from __future__ import annotations

from typing import Dict, Mapping, MutableMapping, Optional

import numpy as np

from ._arrays import array_module, to_host

__all__ = ["EMA"]


class EMA:
    """指数滑动平均：``shadow = decay * shadow + (1 - decay) * param``。"""

    def __init__(
        self,
        params: Mapping[str, np.ndarray],
        decay: float = 0.999,
        dtype=np.float32,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay 必须落在 [0, 1)，得到 {decay}")
        if not params:
            raise ValueError("EMA 至少需要一个参数")
        self.params = params
        self.decay = float(decay)
        self.dtype = np.dtype(dtype)
        # 影子跟着参数所在设备走：设备态训练里参数在显存，影子也该在显存，
        # 否则每步 update 都要把参数拷回主机
        self._shadow: Dict[str, np.ndarray] = {
            name: array_module(arr).array(arr, dtype=self.dtype, copy=True)
            for name, arr in params.items()
        }
        self.num_updates = 0

    # -- 更新 ---------------------------------------------------------

    def update(self, params: Optional[Mapping[str, np.ndarray]] = None) -> None:
        """用当前参数更新影子。``params`` 缺省时用构造时传入的那一组。"""
        source = self.params if params is None else params
        self._check(source)
        decay = self.decay
        for name, arr in source.items():
            shadow = self._shadow[name]
            xp = array_module(shadow)
            shadow *= decay
            shadow += (1.0 - decay) * (
                arr if isinstance(arr, xp.ndarray) else xp.asarray(arr, dtype=self.dtype)
            )
        self.num_updates += 1

    # -- 使用 ---------------------------------------------------------

    def shadow(self) -> Dict[str, np.ndarray]:
        """影子权重的副本（推理用；改动它不会影响内部状态）。

        设备态下返回的仍是设备副本——需要主机值时用 ``_arrays.to_host``，
        不要在训练循环里隐式 ``np.asarray``。
        """
        return {name: arr.copy() for name, arr in self._shadow.items()}

    def copy_to(self, params: Optional[MutableMapping[str, np.ndarray]] = None) -> None:
        """把影子原地写回参数表。

        原地写而不是替换字典：优化器与参数表持有的是数组引用，
        替换成新数组会让优化器继续更新一批没人再读的旧内存。
        """
        target = self.params if params is None else params
        self._check(target)
        for name, arr in target.items():
            shadow = self._shadow[name]
            if array_module(shadow) is array_module(arr):
                arr[...] = shadow.astype(arr.dtype, copy=False)
            else:  # 影子与目标不在同一侧（主机 <-> 显存）：过一次主机
                arr[...] = to_host(shadow).astype(arr.dtype, copy=False)

    # -- 存档 ---------------------------------------------------------

    def state_dict(self) -> dict:
        """存档永远是主机 numpy：设备态训练里影子在显存，也要先拷回来。"""
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": {name: np.array(to_host(arr), copy=True) for name, arr in self._shadow.items()},
        }

    def load_state_dict(self, state: Mapping) -> None:
        """恢复影子与计数。名字、形状任何一处不匹配都直接报错。

        不"能恢复多少恢复多少"：少一个参数的影子，推理权重就混进了训练权重，
        而这种偏差在指标上表现为"稍微差一点"，很难归因。
        """
        shadow = state.get("shadow") or {}
        if set(shadow) != set(self._shadow):
            raise KeyError(
                f"存档里的影子参数 {sorted(set(shadow) - set(self._shadow))} 当前不存在；"
                f"当前缺少 {sorted(set(self._shadow) - set(shadow))}"
            )
        for name, arr in shadow.items():
            if np.shape(arr) != self._shadow[name].shape:
                raise ValueError(
                    f"影子 {name} 的形状 {np.shape(arr)} 与参数形状 "
                    f"{self._shadow[name].shape} 不一致"
                )
            self._shadow[name] = array_module(self._shadow[name]).array(
                arr, dtype=self.dtype, copy=True
            )
        self.num_updates = int(state.get("num_updates", 0))

    # ------------------------------------------------------------------

    def _check(self, params: Mapping[str, np.ndarray]) -> None:
        if set(params) != set(self._shadow):
            missing = sorted(set(self._shadow) - set(params))
            extra = sorted(set(params) - set(self._shadow))
            raise KeyError(f"参数集合不一致：缺 {missing}，多 {extra}")
        for name, arr in params.items():
            if np.shape(arr) != self._shadow[name].shape:
                raise ValueError(
                    f"参数 {name} 的形状 {np.shape(arr)} 与影子 "
                    f"{self._shadow[name].shape} 不一致"
                )

    def __repr__(self) -> str:
        return f"EMA(decay={self.decay:g}, updates={self.num_updates}, {len(self._shadow)} 个参数)"
