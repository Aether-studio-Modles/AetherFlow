"""执行侧：参考语义解释器 + SNN 的稀疏事件表示 + 运行时数据上报。"""

from .events import (
    EventPool,
    EventPoolStats,
    SpikeEvents,
    dense_to_events,
    event_mean_rate,
    events_to_dense,
)
from .paged_cache import PagedCacheFullError, PagedKVCache
from .reference import ReferenceExecutor, ShapeMismatchError
from .telemetry import (
    InstrumentedBackend,
    NodeTiming,
    RunRecord,
    TelemetryRecorder,
    load_records,
)
from .trace import build_trace, trace_events, write_trace

__all__ = [
    "ReferenceExecutor",
    "ShapeMismatchError",
    "NodeTiming",
    "RunRecord",
    "TelemetryRecorder",
    "InstrumentedBackend",
    "load_records",
    "trace_events",
    "build_trace",
    "write_trace",
    "SpikeEvents",
    "EventPool",
    "EventPoolStats",
    "dense_to_events",
    "events_to_dense",
    "event_mean_rate",
    "PagedKVCache",
    "PagedCacheFullError",
]
