"""执行侧：参考语义解释器 + SNN 的稀疏事件表示 + 运行时数据上报。"""

from .events import (
    EventPool,
    EventPoolStats,
    SpikeEvents,
    dense_to_events,
    event_mean_rate,
    events_to_dense,
)
from .reference import ReferenceExecutor, ShapeMismatchError
from .telemetry import (
    InstrumentedBackend,
    NodeTiming,
    RunRecord,
    TelemetryRecorder,
    load_records,
)

__all__ = [
    "ReferenceExecutor",
    "ShapeMismatchError",
    "NodeTiming",
    "RunRecord",
    "TelemetryRecorder",
    "InstrumentedBackend",
    "load_records",
    "SpikeEvents",
    "EventPool",
    "EventPoolStats",
    "dense_to_events",
    "events_to_dense",
    "event_mean_rate",
]
