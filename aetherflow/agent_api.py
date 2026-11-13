"""智能体接口：给外部智能体/工具看的**只读**查询面。

它回答四类问题，全部返回可直接 JSON 序列化的字典：

1. 这个后端是什么、承接哪些算子（``describe_backend``）；
2. 某个算子有几种实现、能不能调（``describe_operator``）；
3. 最近跑得怎么样（``performance_summary``）；
4. 当前形状下推荐哪条配置（``recommend_config``）。

**边界写死在这一层：不执行图、不接受代码、不改配置。**

- 没有任何"执行"入口。要看数据，先由调用方在正常流程里跑一遍并挂上
  ``TelemetryRecorder``；这一层只读记录。
- 不接受任意表达式或脚本字符串。查询对象只有"后端、算子名、形状、签名"，
  参数校验失败抛 ``AgentQueryError``，而不是把字符串送进 ``eval`` 之类的路径。
- 不写配置库。``recommend_config`` 只查；更新配置是 tuning 的事，且必须
  先有实测。

为什么值得单独一层而不是让智能体直接 ``dir(backend)``：反射拿到的是
Python 对象，不是"系统承诺的查询面"。接口一旦稳定，内部结构可以自由改，
而智能体看到的字段语义不变；反过来，任何"顺手加个执行入口"的改动都会
在这里被评审看到。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

from .backends.contracts import implemented_contracts
from .runtime.telemetry import TelemetryRecorder
from .tuning.space import canonical_signature
from .tuning.store import ConfigStore

__all__ = [
    "AgentQueryError",
    "describe_backend",
    "describe_backends",
    "describe_operator",
    "performance_summary",
    "recommend_config",
    "snapshot",
]


class AgentQueryError(ValueError):
    """查询不成立（算子不存在、后端不可调、签名不合法）。"""


def _target(backend: object) -> object:
    """剥掉上报包装层：智能体问的是后端本身的能力。"""
    return getattr(backend, "inner", backend)


def _device(backend: object) -> Optional[Dict[str, Any]]:
    info = getattr(backend, "device_info", None)
    if info is None:
        return None
    try:
        value = info()
    except Exception:  # noqa: BLE001 - 拿不到就报 None，不编造
        return None
    return {
        "name": getattr(value, "name", str(value)),
        "runtime": getattr(value, "runtime", None),
        "total_memory_bytes": getattr(value, "total_memory_bytes", None),
    }


def _tunable_ops(backend: object) -> list:
    names = getattr(backend, "tunable_ops", None)
    if names is None:
        return []
    try:
        return sorted(str(name) for name in names())
    except Exception:  # noqa: BLE001 - 能力查询失败等价于"没有可调项"
        return []


def _variants(backend: object, op: str) -> list:
    registry = getattr(backend, "kernels", None)
    if registry is None:
        return []
    try:
        variants = registry.variants(op)
    except Exception:  # noqa: BLE001
        return []
    return [
        {"label": variant.label, "cost_ranked": variant.cost is not None}
        for variant in variants
    ]


def describe_operator(backend: object, op: str) -> Dict[str, Any]:
    """某个算子的实现候选与可调参数；算子未承接时 ``implemented=False``。"""
    if not isinstance(op, str) or not op:
        raise AgentQueryError(f"算子名必须是非空字符串，得到 {op!r}")
    target = _target(backend)
    capabilities = getattr(target, "capabilities", None)
    implemented = bool(capabilities and op in capabilities())

    tunable = None
    if implemented:
        space_of = getattr(target, "tuning_space", None)
        if space_of is not None and op in _tunable_ops(target):
            try:
                tunable = space_of(op).to_dict()
            except Exception as exc:  # noqa: BLE001
                raise AgentQueryError(f"查询 {op} 的调优空间失败：{exc}") from exc

    return {
        "op": op,
        "implemented": implemented,
        "variants": _variants(target, op) if implemented else [],
        "tunable": tunable,
    }


def describe_backend(backend: object) -> Dict[str, Any]:
    """一个后端的完整静态画像。"""
    target = _target(backend)
    capabilities = getattr(target, "capabilities", None)
    ops = sorted(capabilities()) if capabilities else []
    return {
        "name": getattr(backend, "name", "unknown"),
        "capabilities": ops,
        "contracts": list(implemented_contracts(target)),
        "device": _device(target),
        "tunable_ops": _tunable_ops(target),
        # 只列变体，不重复 capabilities 里已有的算子名；单点查询用 describe_operator
        "operators": {op: {"variants": _variants(target, op)} for op in ops},
    }


def describe_backends(backends: Iterable[object]) -> Dict[str, Any]:
    """多个后端一起描述；重名直接拒绝，否则结果里会静默少一个。"""
    out: Dict[str, Any] = {}
    for backend in backends:
        item = describe_backend(backend)
        name = item["name"]
        if name in out:
            raise AgentQueryError(f"后端名 {name!r} 重复，无法唯一定位")
        out[name] = item
    return out


def performance_summary(recorder: TelemetryRecorder) -> Dict[str, Any]:
    """运行时上报的聚合摘要；没有记录时返回 ``runs=0`` 而不是抛错。"""
    if not isinstance(recorder, TelemetryRecorder):
        raise AgentQueryError(
            f"performance_summary 需要 TelemetryRecorder，得到 {type(recorder).__name__}"
        )
    return recorder.summary()


def recommend_config(
    store: ConfigStore,
    backend: object,
    op: str,
    signature: Mapping[str, Any],
) -> Dict[str, Any]:
    """查配置库；未命中时给后端的默认配置，并标明来源。

    ``source`` 只有两个取值：``"store"``（实测过的最优）与 ``"default"``
    （空间里第一个候选）。没有第三个——"猜一个"不是一个来源。
    """
    if not isinstance(store, ConfigStore):
        raise AgentQueryError(f"recommend_config 需要 ConfigStore，得到 {type(store).__name__}")
    if not isinstance(signature, Mapping) or not signature:
        raise AgentQueryError("signature 必须是非空映射")
    target = _target(backend)
    name = str(getattr(target, "name", "unknown"))
    device = (_device(target) or {}).get("name") or "unknown"
    normalized = canonical_signature(**dict(signature))

    record = store.get(name, device, op, normalized)
    if record is not None:
        return {
            "op": op,
            "source": "store",
            "config": dict(record.config),
            "median_ms": record.median_ms,
            "best_ms": record.best_ms,
            "p95_ms": record.p95_ms,
            "samples": record.samples,
            "device": device,
        }

    space_of = getattr(target, "tuning_space", None)
    if space_of is None or op not in _tunable_ops(target):
        raise AgentQueryError(f"{name} 的算子 {op} 没有可调参数，也没有配置库记录")
    try:
        space = space_of(op)
    except Exception as exc:  # noqa: BLE001
        raise AgentQueryError(f"查询 {op} 的调优空间失败：{exc}") from exc
    return {
        "op": op,
        "source": "default",
        "config": space.default(),
        "median_ms": None,
        "best_ms": None,
        "p95_ms": None,
        "samples": 0,
        "device": device,
    }


def snapshot(
    backends: Iterable[object],
    recorder: Optional[TelemetryRecorder] = None,
    store: Optional[ConfigStore] = None,
) -> Dict[str, Any]:
    """一次取全：后端画像 + 运行摘要 + 配置库规模。"""
    out: Dict[str, Any] = {"backends": describe_backends(backends)}
    if recorder is not None:
        out["performance"] = performance_summary(recorder)
    if store is not None:
        if not isinstance(store, ConfigStore):
            raise AgentQueryError(f"store 必须是 ConfigStore，得到 {type(store).__name__}")
        out["config_store"] = {
            "records": len(store),
            "entries": [
                {
                    "op": record.op,
                    "device": record.device,
                    "config": dict(record.config),
                    "median_ms": record.median_ms,
                    "best_ms": record.best_ms,
                    "p95_ms": record.p95_ms,
                    "samples": record.samples,
                }
                for record in store.records()
            ],
        }
    return out
