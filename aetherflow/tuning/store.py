"""调优配置库：把"在某个设备、某种形状下测出来的最优配置"落到文件。

**为什么要有库，而不是每次跑之前重新搜一遍。** 搜索的代价是几倍于正常执行
的测量开销，而硬件的答案在"同一台机器 + 同一形状"内是稳定的。库把一次搜索
摊到之后所有运行上；同时留下 `median_ms` / `best_ms` / `p95_ms` 与 `samples`，
让"这条配置有没有被真正测过、测的时候抖不抖"可核对，而不是只看到一个孤零零
的参数值。

**为什么版本升到 2。** v1 把"多轮最小值"写进 `median_ms`，那个数回答的是
"这块卡最好能做到多快"，不是典型延迟。两个口径混用会让"加速比"看起来比
实际更漂亮，也无法判断一条记录能不能复现。v2 明确：`median_ms` 是中位数，
`best_ms` 是最小值（可选，旧口径的值不能猜着塞进这里）。

**键里为什么必须有 device。** 同一份配置在 2080 Ti 上最优，换一张卡可能
更慢。只有 ``backend + device + op + signature`` 全一致才允许命中；device
取不到时写 ``"unknown"``——那是一个具体的"不知道"，不会与真实设备名混淆。

**读文件失败必须显式。** 坏 JSON、版本不认识、条目字段缺失，全部抛
``ValueError``，绝不"跳过坏条目继续加载"：配置库一旦悄悄丢掉条目，表现是
"调优没生效"，排查方向会完全跑偏。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

__all__ = ["TuningRecord", "ConfigStore", "STORE_VERSION"]

STORE_VERSION = 2


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _normalize_device(device: str) -> str:
    return " ".join(str(device).split())


@dataclass(frozen=True)
class TuningRecord:
    """一条"某个算子在某个形状下的最优配置"。

    ``median_ms`` 是多轮样本的中位数（典型延迟），``best_ms`` 是最小值
    （最佳可达，可能没采到），``p95_ms`` 给出抖动上限，``samples`` 是有效
    测量轮数。四者一起构成"这条记录可信到什么程度"的证据；只有 ``samples``
    与中位数时，看不出它是稳定复现还是一次侥幸。
    """

    backend: str
    device: str
    op: str
    signature: Mapping[str, Any]
    config: Mapping[str, Any]
    median_ms: float
    best_ms: Optional[float] = None
    p95_ms: Optional[float] = None
    samples: int = 1
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("调优记录的 backend 不能为空")
        if not self.op:
            raise ValueError("调优记录的 op 不能为空")
        if not isinstance(self.config, Mapping) or not self.config:
            raise ValueError("调优记录的 config 必须是非空映射")
        for name in ("median_ms", "best_ms", "p95_ms"):
            value = getattr(self, name)
            if value is None:
                continue
            if value < 0 or value != value:  # NaN 也拒绝
                raise ValueError(f"调优记录的 {name} 必须是非负数，得到 {value!r}")
        # 口径不能混着填：把"最小值"写进 median_ms 正是 v1 的问题所在，
        # 这里直接拒绝，而不是在报告里悄悄把它当中位数用。
        if self.best_ms is not None and self.best_ms > self.median_ms:
            raise ValueError(
                f"调优记录的 best_ms({self.best_ms}) 不能大于 median_ms"
                f"({self.median_ms})：best 是最小值，两者口径不同"
            )
        if self.p95_ms is not None and self.p95_ms < self.median_ms:
            raise ValueError(
                f"调优记录的 p95_ms({self.p95_ms}) 不能小于 median_ms"
                f"({self.median_ms})：p95 是最慢一档，口径不同"
            )
        if self.samples < 1:
            raise ValueError(f"调优记录的 samples 至少为 1，得到 {self.samples!r}")
        object.__setattr__(self, "device", _normalize_device(self.device))
        object.__setattr__(self, "signature", dict(self.signature))
        object.__setattr__(self, "config", dict(self.config))
        if not self.created_at:
            object.__setattr__(
                self,
                "created_at",
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )

    def key(self) -> Tuple[str, str, str, str]:
        """命中判据：四段全等。形状/属性按规范 JSON 比较。"""
        return (self.backend, self.device, self.op, _canonical(dict(self.signature)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "device": self.device,
            "op": self.op,
            "signature": dict(self.signature),
            "config": dict(self.config),
            "median_ms": self.median_ms,
            # None 是"没采到"，照实写出来；缺失与被省略不能靠读者猜
            "best_ms": self.best_ms,
            "p95_ms": self.p95_ms,
            "samples": self.samples,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TuningRecord":
        required = ("backend", "device", "op", "signature", "config", "median_ms")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"调优记录缺少字段 {missing}")
        return cls(
            backend=_require_str(data, "backend"),
            device=_require_str(data, "device"),
            op=_require_str(data, "op"),
            signature=_require_mapping(data, "signature"),
            config=_require_mapping(data, "config"),
            median_ms=float(data["median_ms"]),
            best_ms=_optional_float(data, "best_ms"),
            p95_ms=_optional_float(data, "p95_ms"),
            samples=int(data.get("samples", 1)),
            created_at=str(data.get("created_at", "")),
        )


def _require_str(data: Mapping[str, Any], name: str) -> str:
    value = data[name]
    if not isinstance(value, str):
        raise ValueError(f"调优记录的 {name} 必须是字符串，得到 {type(value).__name__}")
    return value


def _require_mapping(data: Mapping[str, Any], name: str) -> Dict[str, Any]:
    value = data[name]
    if not isinstance(value, Mapping):
        raise ValueError(f"调优记录的 {name} 必须是映射，得到 {type(value).__name__}")
    return dict(value)


def _optional_float(data: Mapping[str, Any], name: str) -> Optional[float]:
    """可选的数值字段：缺失或 ``None`` 表示"没采到"，其余必须是数。"""
    value = data.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"调优记录的 {name} 必须是数或 null，得到 {type(value).__name__}")
    return float(value)


class ConfigStore:
    """``TuningRecord`` 的集合，按 ``key()`` 去重。

    同键再 ``put`` 是**覆盖**：最新一次测量胜出。配置库记录的是"当前这台机器
    现在的答案"，不是历史归档——要比较历史请另存文件。
    """

    def __init__(self, records: Optional[Iterable[TuningRecord]] = None) -> None:
        self._records: Dict[Tuple[str, str, str, str], TuningRecord] = {}
        for record in records or ():
            self._records[record.key()] = record

    # ------------------------------------------------------------------

    def put(self, record: TuningRecord) -> TuningRecord:
        if not isinstance(record, TuningRecord):
            raise TypeError(f"ConfigStore 只接受 TuningRecord，得到 {type(record).__name__}")
        self._records[record.key()] = record
        return record

    def get(
        self,
        backend: str,
        device: str,
        op: str,
        signature: Mapping[str, Any],
    ) -> Optional[TuningRecord]:
        """按四段键查；未命中返回 ``None``（不返回默认值，调用方自己决定用什么）。"""
        key = (
            str(backend),
            _normalize_device(device),
            str(op),
            _canonical(dict(signature)),
        )
        return self._records.get(key)

    def remove(
        self, backend: str, device: str, op: str, signature: Mapping[str, Any]
    ) -> bool:
        key = (
            str(backend),
            _normalize_device(device),
            str(op),
            _canonical(dict(signature)),
        )
        return self._records.pop(key, None) is not None

    def records(self) -> Tuple[TuningRecord, ...]:
        """按插入顺序返回全部记录。"""
        return tuple(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records.values())

    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "store_version": STORE_VERSION,
            "records": [record.to_dict() for record in self._records.values()],
        }

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConfigStore":
        if not isinstance(data, Mapping):
            raise ValueError(f"配置库顶层必须是对象，得到 {type(data).__name__}")
        version = data.get("store_version")
        if version != STORE_VERSION:
            raise ValueError(
                f"配置库版本 {version!r} 不受支持（本实现只读 {STORE_VERSION}）；"
                "v1 的 median_ms 实际是'多轮最小值'，与 v2 的'中位数'语义不同，"
                "不猜旧值属于哪个口径，请重新采集"
            )
        raw = data.get("records")
        if not isinstance(raw, list):
            raise ValueError("配置库的 records 必须是数组")
        return cls(TuningRecord.from_dict(item) for item in raw)

    @classmethod
    def loads(cls, text: str) -> "ConfigStore":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"配置库不是合法 JSON：{exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ConfigStore":
        try:
            with open(path, encoding="utf-8") as fh:
                return cls.loads(fh.read())
        except json.JSONDecodeError as exc:  # 理论上 loads 已转成 ValueError
            raise ValueError(f"{path} 不是合法 JSON：{exc}") from exc
        except ValueError as exc:
            raise ValueError(f"无法加载配置库 {path}：{exc}") from exc

    def save(self, path: str | os.PathLike) -> None:
        """原子写：先写临时文件再 ``os.replace``。

        直接覆盖目标文件时，进程若在写一半时退出，留下的是一个**截断的 JSON**，
        下次加载会失败——而失败点离"上一次保存"很远。临时文件 + 原子替换把
        这个窗口压到零。
        """
        target = os.fspath(path)
        tmp = f"{target}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(self.dumps())
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
