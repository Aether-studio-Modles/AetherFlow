"""执行期调度策略：把配置库里的答案接到真实 launch 上。

配置库（``store.ConfigStore``）回答"某个后端、某张卡、某个算子、某个形状
下实测最快的是什么"，但它不会自己生效——没有这一层，库只是一份报告。
本模块负责把四段键补齐（后端名、设备名、算子名、形状签名）、查库、并给
调用方一个"合并后"的配置。

**优先级写在调用方（后端），不在这里。** 这里的 ``lookup`` 只回答"库里有
没有"，返回 ``None`` 就是没有；``overlay`` 只做"在默认值之上覆盖库里给的
键"，不决定"谁该赢"。后端据此把显式 ``set_tuning`` 与构造参数排在前面，
库排在自动推导之前。把优先级塞进策略层，会让"为什么这次没按我设的跑"
变成两层都要读的问题。

**未命中要能看见。** 统计 ``lookup / hit / miss / cached`` 四个数：只报
"调优已启用"而说不清命中率的接口，等于把"调优没生效"藏起来。``unknown``
记录被忽略的未知参数名——旧记录多带一个键只作提示，值不合法则直接抛错。

**缓存与失效。** 查库要把签名序列化成 JSON 键，逐次 launch 都做一遍没有
意义；本层按 ``(op, 规范签名)`` 缓存结果（含"没有记录"这个结果）。缓存
只在 ``reset_stats`` 之外的显式重建（新建策略）时失效——配置库在执行期
被改写属于"训练中更新配置"，那要重建策略，不做隐式热更新。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional

from .space import canonical_signature
from .store import ConfigStore

__all__ = ["SchedulePolicy"]


def _key_of(op: str, signature: Mapping[str, Any]) -> str:
    return op + "\x00" + json.dumps(
        dict(signature), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


class SchedulePolicy:
    """一个后端 + 一个设备 + 一个配置库的查表器。

    构造后字段只读：``backend`` / ``device`` 参与四段键，运行中换掉会让
    一半记录命中、一半不命中，且没有任何提示。要换就换一个策略对象。
    """

    def __init__(
        self,
        store: ConfigStore,
        *,
        backend: str,
        device: str,
        enabled: bool = True,
    ) -> None:
        if not isinstance(store, ConfigStore):
            raise TypeError(
                f"SchedulePolicy 需要 ConfigStore，得到 {type(store).__name__}"
            )
        if not isinstance(backend, str) or not backend:
            raise ValueError(f"backend 必须是非空字符串，收到 {backend!r}")
        if not isinstance(device, str) or not device:
            raise ValueError(f"device 必须是非空字符串，收到 {device!r}")
        self._store = store
        self._backend = backend
        self._device = device
        self._enabled = bool(enabled)
        self._cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self.reset_stats()

    @classmethod
    def from_backend(
        cls,
        backend: object,
        store: ConfigStore,
        *,
        device: Optional[str] = None,
        enabled: bool = True,
    ) -> "SchedulePolicy":
        """按后端自身的名字与设备名构造；``device`` 显式给出时优先。

        包装层（如上报后端）用 ``inner`` 剥掉：键里的后端名必须是真正干活的
        那一个，否则离线写库的脚本与执行期算出的键永远不相等。
        """
        target = getattr(backend, "inner", backend)
        name = getattr(target, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError(f"后端没有可用的 name，得到 {name!r}")
        if device is None:
            info = getattr(target, "device_info", None)
            if info is None:
                device = "unknown"
            else:
                try:
                    value = info()
                except Exception:  # noqa: BLE001 - 拿不到就按"不知道"记
                    device = "unknown"
                else:
                    device = str(getattr(value, "name", value) or "unknown")
        return cls(store, backend=name, device=device, enabled=enabled)

    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def device(self) -> str:
        return self._device

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = bool(value)

    def lookup(self, op: str, signature: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """查库；未命中返回 ``None``。返回值是配置的副本，可被调用方改写。

        禁用时直接返回 ``None``，且不计入统计——"没开"与"没命中"是两件事。
        """
        if not self._enabled:
            return None
        if not isinstance(signature, Mapping) or not signature:
            raise ValueError("签名必须是非空映射")
        normalized = canonical_signature(**dict(signature))
        key = _key_of(op, normalized)
        self._stats["lookup"] += 1
        if key in self._cache:
            self._stats["cached"] += 1
            hit = self._cache[key]
            return None if hit is None else dict(hit)
        record = self._store.get(self._backend, self._device, op, normalized)
        if record is None:
            self._cache[key] = None
            self._stats["miss"] += 1
            return None
        config = dict(record.config)
        self._cache[key] = config
        self._stats["hit"] += 1
        return dict(config)

    def overlay(
        self, config: Mapping[str, Any], current: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """把 ``config`` 里**当前认识的键**盖到 ``current`` 上。

        不认识的键（旧版本记录或别的算子的配置）被忽略并计入 ``unknown``：
        记录可能来自更早的参数集，键多一点不该让整份配置作废。

        值不合法则**抛错**，与 ``set_tuning`` 同一口径：库里的记录本质上是
        "预置的调优覆盖"，写坏了就该在应用它的那一刻暴露，而不是静默退回
        默认值、让人以为调优没生效。``None`` 是合法值，表示该项仍由计划层
        按形状推导。
        """
        if not isinstance(config, Mapping):
            raise TypeError(f"config 必须是映射，得到 {type(config).__name__}")
        merged = dict(current)
        for name, value in config.items():
            if name not in merged:
                self._stats["unknown"] += 1
                continue
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(
                    f"配置库中 {name} 的值 {value!r} 不合法：必须是正整数或 None"
                )
            merged[name] = value
        return merged

    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        """``lookup`` / ``hit`` / ``miss`` / ``cached`` / ``unknown``。

        ``lookup`` 是调用总次数，``hit + miss`` 只统计真正查过库的那部分，
        缓存命中的次数单列在 ``cached`` 里。
        """
        return dict(self._stats)

    def reset_stats(self) -> None:
        self._stats: Dict[str, int] = {
            "lookup": 0,
            "hit": 0,
            "miss": 0,
            "cached": 0,
            "unknown": 0,
        }

    def __len__(self) -> int:
        return len(self._store)
