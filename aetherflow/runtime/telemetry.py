"""运行时数据上报：一次执行留下一条结构化记录。

记录三类东西——**整图耗时**、**逐节点的派发耗时**、**显存统计**（若后端能报）。
三者一起才够回答"慢在哪"：整图慢了 3 倍，是某个 kernel 退化，还是节点数变多，
还是分配器多占了一倍显存？只有总时长的记录回答不了这个问题。

**``dispatch_ms`` 是派发耗时，不是 GPU 内核时间。**

    t0 = perf_counter()
    inner.invoke(...)      # CUDA 是异步下发，函数返回时 kernel 往往还没算完
    dispatch_ms = ...

所以 ``dispatch_ms`` 只表示"CPU 侧把这个节点发出去花了多久"，它对
"Python 派发是否成为瓶颈"是有效证据（DESIGN 5.4 里小规模 CUDA 反而更慢
就是这一类），对"这个 kernel 在 GPU 上跑多久"无效。``wall_ms`` 覆盖
``execute`` 全程，含结尾的 ``to_host`` 同步，因此它是真实的完成时间。

**5.161 起另有一条不打断异步的路**：后端若提供 ``gpu_clock``，包装器在每个
节点前后各记一个计时事件，整次执行完只同步一次，再逐个读 ``elapsed``，得到
``NodeTiming.gpu_ms``。逐节点插同步会把测量对象换成同步开销本身，事件对不会。
这个观测只在能把事件排在当前流上的后端存在（CUDA / HIP），参考后端恒为
``None``；``None`` 是"没测"，不是 0。

**不做的两件事**：不改图、不改配置。上报只读；基于数据决定配置是 tuning 的事。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..backends.contracts import DeviceInfo, MemoryStats
from .executor import Executor

__all__ = [
    "NodeTiming",
    "RunRecord",
    "TelemetryRecorder",
    "InstrumentedBackend",
    "load_records",
]


@dataclass(frozen=True)
class NodeTiming:
    """一个节点的耗时。``name`` 是输出值名，用来区分同算子的多个实例。

    ``dispatch_ms`` 是 CPU 侧下发耗时，``gpu_ms`` 是这一节点在两个计时事件
    之间占用的 GPU 墙钟时间（5.161）。后者只在后端提供 ``gpu_clock`` 且
    调用方没有关掉计时时才有值；``None`` 表示"没有这个观测"，不是 0。
    """

    op: str
    name: str
    dispatch_ms: float
    gpu_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {"op": self.op, "name": self.name, "dispatch_ms": self.dispatch_ms}
        if self.gpu_ms is not None:
            out["gpu_ms"] = self.gpu_ms
        return out

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NodeTiming":
        gpu = payload.get("gpu_ms")
        return cls(
            op=str(payload["op"]),
            name=str(payload["name"]),
            dispatch_ms=float(payload["dispatch_ms"]),
            gpu_ms=None if gpu is None else float(gpu),
        )


@dataclass(frozen=True)
class RunRecord:
    """一次 ``execute`` 的快照。

    ``device`` / ``memory`` 允许为 ``None``——参考后端既没有设备可问，
    也没有分层分配器可报。``None`` 与 0 是两件事：前者是"没有这个观测"，
    后者是"观测到 0"，报告里不能混用。
    """

    graph: str
    backend: str
    wall_ms: float
    nodes: Tuple[NodeTiming, ...] = ()
    device: Optional[str] = None
    memory: Optional[MemoryStats] = None
    units: Optional[Mapping[str, float]] = None
    label: Optional[str] = None
    """调用方给的稳定标签（如基准用例名）。

    ``graph`` 是构图时写下的名字，同名图（不同规模的两个 ``matmul``）会撞在
    一起；趋势按 ``label or graph`` 分组，才能分清"256 的 matmul 变慢了"和
    "1024 的 matmul 变慢了"。
    """

    def __post_init__(self) -> None:
        if self.wall_ms < 0 or self.wall_ms != self.wall_ms:
            raise ValueError(f"wall_ms 必须是非负数，得到 {self.wall_ms!r}")
        if any(item.dispatch_ms < 0 for item in self.nodes):
            raise ValueError("节点派发耗时不能为负")
        if any(item.gpu_ms is not None and item.gpu_ms < 0 for item in self.nodes):
            raise ValueError("节点 GPU 耗时不能为负")

    def throughput(self) -> Dict[str, float]:
        """每个单位的每秒吞吐；没有单位或耗时为 0 时返回空字典。

        单位由调用方给（``{"elements": 1_048_576}``），框架不猜——"这个图算了
        多少 FLOPs"需要算子级模型，拍一个数字出来只会得到一条看起来精确的假数据。
        """
        if not self.units or self.wall_ms <= 0:
            return {}
        seconds = self.wall_ms / 1e3
        return {name: round(float(value) / seconds, 4) for name, value in self.units.items()}

    def node_totals(self) -> Tuple[Tuple[str, int, float], ...]:
        """按算子聚合 ``(op, 调用次数, 累计派发耗时)``，按耗时降序。"""
        calls: Dict[str, int] = {}
        total: Dict[str, float] = {}
        for item in self.nodes:
            calls[item.op] = calls.get(item.op, 0) + 1
            total[item.op] = total.get(item.op, 0.0) + item.dispatch_ms
        rows = [(op, calls[op], round(total[op], 6)) for op in total]
        rows.sort(key=lambda row: (-row[2], row[0]))
        return tuple(rows)

    def node_gpu_totals(self) -> Tuple[Tuple[str, int, float], ...]:
        """按算子聚合 ``(op, 有 GPU 观测的调用次数, 累计 GPU 耗时)``。

        与 ``node_totals`` 分开：没有 ``gpu_clock`` 的后端每次都是空元组，
        把它混进派发表会让"0 ms"和"没测"分不开。
        """
        calls: Dict[str, int] = {}
        total: Dict[str, float] = {}
        for item in self.nodes:
            if item.gpu_ms is None:
                continue
            calls[item.op] = calls.get(item.op, 0) + 1
            total[item.op] = total.get(item.op, 0.0) + item.gpu_ms
        rows = [(op, calls[op], round(total[op], 6)) for op in total]
        rows.sort(key=lambda row: (-row[2], row[0]))
        return tuple(rows)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "graph": self.graph,
            "backend": self.backend,
            "device": self.device,
            "wall_ms": self.wall_ms,
            "nodes": [item.to_dict() for item in self.nodes],
            "node_totals": [
                {"op": op, "calls": calls, "dispatch_ms": ms}
                for op, calls, ms in self.node_totals()
            ],
            "throughput": self.throughput(),
        }
        gpu_rows = self.node_gpu_totals()
        if gpu_rows:
            out["node_gpu_totals"] = [
                {"op": op, "calls": calls, "gpu_ms": ms}
                for op, calls, ms in gpu_rows
            ]
        if self.memory is not None:
            out["memory"] = {
                "live_bytes": self.memory.live_bytes,
                "peak_bytes": self.memory.peak_bytes,
                "total_bytes": self.memory.total_bytes,
                "requested_bytes": self.memory.requested_bytes,
            }
        if self.units is not None:
            out["units"] = dict(self.units)
        if self.label is not None:
            out["label"] = self.label
        return out

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunRecord":
        """从 ``to_dict()`` 的结构还原。``node_totals`` / ``throughput`` 是派生字段，忽略。"""
        memory = payload.get("memory")
        stats = MemoryStats(**memory) if memory is not None else None
        units = payload.get("units")
        return cls(
            graph=str(payload["graph"]),
            backend=str(payload["backend"]),
            wall_ms=float(payload["wall_ms"]),
            nodes=tuple(NodeTiming.from_dict(item) for item in payload.get("nodes", ())),
            device=payload.get("device"),
            memory=stats,
            units=dict(units) if units is not None else None,
            label=payload.get("label"),
        )


class InstrumentedBackend(Executor):
    """包一层 ``invoke`` 计时，其余全部委托给内层后端。

    用 Executor 子类而不是 monkeypatch：``run`` / ``run_device`` 的固定流程
    （形状校验、符号维绑定、cast）仍然只有一份实现，包装器只插入计时。
    契约方法按内层**是否存在**逐项绑定，而不是无条件定义——否则参考后端被
    包一层之后会突然"声称"实现了 ``MemoryContract``，而它只能返回 None。
    """

    def __init__(
        self,
        inner: Executor,
        recorder: "TelemetryRecorder",
        gpu_timing: bool = True,
    ) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", "unknown")
        self._recorder = recorder
        self._nodes: List[NodeTiming] = []
        # 逐节点 GPU 计时（5.161）：只有内层能提供 gpu_clock 才启用。
        self._gpu = None
        self._gpu_pairs: List[Tuple[Any, Any]] = []
        if gpu_timing:
            factory = getattr(inner, "gpu_clock", None)
            if callable(factory):
                self._gpu = factory()
        if hasattr(inner, "device_info"):
            self.device_info = inner.device_info  # type: ignore[assignment]
        if hasattr(inner, "memory_stats"):
            self.memory_stats = inner.memory_stats  # type: ignore[assignment]
        # 5.89 之后这两项也是契约的一部分：分配器与流。少转发一个，包装器就会
        # 静默"少实现一个契约"，而这正是这条按存在性绑定的意义。
        if hasattr(inner, "allocator"):
            self.allocator = inner.allocator  # type: ignore[assignment]
        if hasattr(inner, "new_stream"):
            self.new_stream = inner.new_stream  # type: ignore[assignment]
        # 5.163 的编译期 / 调优面与 5.164 的多设备也是按存在性判定的契约。漏转
        # 一项不会报错，只会让包装后的后端"静默少认一项"——基准会把多卡看成
        # 单卡、调优工具会以为这台机器没有旋钮。逐项按内层是否存在绑定。
        for name in (
            "kernel",
            "tunable_ops",
            "tuning_space",
            "set_tuning",
            "clear_tuning",
            "tuning_config",
            "enumerate_devices",
            "current_device",
            "set_device",
        ):
            if hasattr(inner, name):
                setattr(self, name, getattr(inner, name))

    @property
    def resident(self) -> Mapping[str, Any]:
        """转发驻留参数表；内层没有这个概念时抛 ``AttributeError``（与内层一致）。"""
        if not hasattr(self.inner, "resident"):
            raise AttributeError("resident")
        return self.inner.resident

    # ---- 容器钩子 ----

    def to_array(self, value: Any) -> Any:
        return self.inner.to_array(value)

    def cast(self, value: Any, dtype: Any) -> Any:
        return self.inner.cast(value, dtype)

    def to_host(self, value: Any) -> Any:
        return self.inner.to_host(value)

    def dispose(self, value: Any) -> None:
        self.inner.dispose(value)

    def capabilities(self):
        return self.inner.capabilities()

    # ---- 计时 ----

    def invoke(self, node, args, attrs):
        start = self._gpu.begin() if self._gpu is not None else None
        t0 = perf_counter()
        out = self.inner.invoke(node, args, attrs)
        elapsed = (perf_counter() - t0) * 1e3
        stop = self._gpu.end() if self._gpu is not None else None
        self._nodes.append(
            NodeTiming(node.op.name, node.output.name, round(elapsed, 6))
        )
        if start is not None:
            # 与 ``_nodes`` 一一对应；收尾时统一同步再读 elapsed
            self._gpu_pairs.append((start, stop))
        return out

    def execute(
        self,
        graph,
        inputs: Mapping[str, Any],
        units: Optional[Mapping[str, float]] = None,
        label: Optional[str] = None,
    ):
        """执行并记录一次。``units`` 只用于算吞吐，``label`` 只用于分组，都不参与计算。"""
        resident = getattr(self.inner, "resident", None)
        if resident:
            merged = dict(resident)
            merged.update(inputs)
        else:
            merged = inputs

        self._nodes = []
        self._gpu_pairs = []
        t0 = perf_counter()
        try:
            out = self.run(graph, merged)
        finally:
            wall_ms = (perf_counter() - t0) * 1e3
            self._collect_gpu()

        record = RunRecord(
            graph=getattr(graph, "name", "") or "<anonymous>",
            backend=self.name,
            wall_ms=round(wall_ms, 6),
            nodes=tuple(self._nodes),
            device=self._device_label(),
            memory=self._memory_snapshot(),
            units=units,
            label=label,
        )
        self._recorder.record(record)
        return out

    def _collect_gpu(self) -> None:
        """同步一次，把每个节点的 GPU 耗时填进 ``_nodes``。

        起点/终点事件是异步排进流的，读 ``elapsed`` 之前必须让设备完成。只在
        收尾同步一次，所以测到的是 kernel 的 GPU 时间，而不是逐节点同步开销。
        观测失败（驱动不给读数、事件没记上）不影响执行：``gpu_ms`` 留 ``None``。
        """
        if self._gpu is None or not self._gpu_pairs:
            return
        pairs, self._gpu_pairs = self._gpu_pairs, []
        values: Optional[List[float]] = None
        try:
            self._gpu.synchronize()
            values = [round(self._gpu.elapsed(a, b), 6) for a, b in pairs]
        except Exception:  # noqa: BLE001 - 观测失败不该让执行失败
            values = None
        finally:
            for start, stop in pairs:
                self._gpu.release(start, stop)
        if values is not None and len(values) == len(self._nodes):
            self._nodes = [
                replace(item, gpu_ms=value)
                for item, value in zip(self._nodes, values)
            ]

    # ---- 统计信息（失败不影响执行） ----

    def _device_label(self) -> Optional[str]:
        info = getattr(self.inner, "device_info", None)
        if info is None:
            return None
        try:
            value = info()
        except Exception:  # noqa: BLE001 - 拿不到设备信息不该让执行失败
            return None
        if isinstance(value, DeviceInfo):
            return value.name
        return str(value)

    def _memory_snapshot(self) -> Optional[MemoryStats]:
        stats = getattr(self.inner, "memory_stats", None)
        if stats is None:
            return None
        try:
            return stats()
        except Exception:  # noqa: BLE001
            return None


def _series_stats(values: Tuple[float, ...]) -> Dict[str, Any]:
    """一条时间序列的摘要：首末、极值、中位数、变化率与逐次斜率。

    ``slope_per_run`` 是最小二乘拟合的斜率，用来回答"慢是慢慢累积的，还是
    某一次跳变"——只看首末会被一次抖动带偏。``first == 0`` 时变化率没有定义，
    返回 ``None`` 而不是编一个 0。
    """
    n = len(values)
    first, last = values[0], values[-1]
    ordered = sorted(values)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    numerator = sum((i - mean_x) * (value - mean_y) for i, value in enumerate(values))
    denominator = sum((i - mean_x) ** 2 for i in range(n))
    slope = numerator / denominator if denominator else 0.0
    return {
        "first": round(first, 6),
        "last": round(last, 6),
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
        "median": round(median, 6),
        "change_pct": None if first == 0 else round((last - first) / first * 100.0, 3),
        "slope_per_run": round(slope, 6),
    }


class TelemetryRecorder:
    """收集 ``RunRecord``，提供摘要与落盘。

    ``capacity`` 是保留的最大条数（默认 1000），超出后丢最旧的：上报是给
    趋势用的，不是审计日志；不设上限的话，长跑训练会把整段历史留在内存里。
    """

    def __init__(self, capacity: int = 1000) -> None:
        if capacity < 1:
            raise ValueError(f"capacity 至少为 1，得到 {capacity!r}")
        self.capacity = int(capacity)
        self._records: List[RunRecord] = []

    def record(self, record: RunRecord) -> RunRecord:
        if not isinstance(record, RunRecord):
            raise TypeError(f"只接受 RunRecord，得到 {type(record).__name__}")
        self._records.append(record)
        if len(self._records) > self.capacity:
            del self._records[: len(self._records) - self.capacity]
        return record

    def wrap(self, backend: Executor, gpu_timing: bool = True) -> InstrumentedBackend:
        """包一层后端；已经包过就原样返回，避免重复计时。

        只接受 ``Executor`` 子类：逐节点计时依赖 ``invoke``。``ReferenceBackend``
        这类只提供 ``execute`` 的外观对象请改用 ``ReferenceExecutor``——
        明确报错好过"包上了但没有节点明细"。

        ``gpu_timing`` 只在后端提供 ``gpu_clock`` 时有效；关掉它只省下每节点
        两个事件，``dispatch_ms`` 与 ``wall_ms`` 照旧。
        """
        if isinstance(backend, InstrumentedBackend):
            return backend
        if not isinstance(backend, Executor):
            raise TypeError(
                f"TelemetryRecorder 需要 Executor 子类（逐节点计时依赖 invoke），"
                f"得到 {type(backend).__name__}"
            )
        return InstrumentedBackend(backend, self, gpu_timing=gpu_timing)

    @property
    def records(self) -> Tuple[RunRecord, ...]:
        return tuple(self._records)

    @property
    def last(self) -> Optional[RunRecord]:
        return self._records[-1] if self._records else None

    def clear(self) -> None:
        self._records.clear()

    def extend(self, records: Iterable[RunRecord]) -> None:
        """把另一份记录（例如上次 ``save`` 的产物）接着往后排。

        跨进程/跨提交的趋势就是这么拼起来的：每次 CI 只产出自己那段，
        汇总时按时间顺序 ``extend``，容量与截断规则仍由本实例决定。
        """
        for record in records:
            self.record(record)

    def trend(self, *, graph: Optional[str] = None, window: Optional[int] = None) -> Dict[str, Any]:
        """按图名分组的跨运行趋势。

        ``graph`` 只看某一张图；``window`` 只取每组最近 N 次——趋势关心的是
        "最近是不是在变慢"，把几百次历史全画进统计会被早已修好的旧版本拖偏。
        单位换算与派生指标（节点耗时、吞吐）仍由 ``RunRecord`` 负责，这里不重复算。
        """
        if window is not None and window < 1:
            raise ValueError(f"window 至少为 1，得到 {window!r}")

        groups: Dict[str, List[RunRecord]] = {}
        for record in self._records:
            key = record.label or record.graph
            if graph is not None and key != graph:
                continue
            groups.setdefault(key, []).append(record)

        graphs: Dict[str, Any] = {}
        for name, items in groups.items():
            selected = items[-window:] if window else items
            wall = tuple(record.wall_ms for record in selected)
            entry: Dict[str, Any] = {
                "graph": selected[-1].graph,
                "backend": selected[-1].backend,
                "count": len(selected),
                "wall_ms": _series_stats(wall),
                "series": {"wall_ms": list(wall)},
            }
            peaks = tuple(
                record.memory.peak_bytes for record in selected if record.memory is not None
            )
            if peaks:
                entry["peak_bytes"] = _series_stats(peaks)
                entry["series"]["peak_bytes"] = list(peaks)
            units: Dict[str, List[float]] = {}
            for record in selected:
                for unit, value in record.throughput().items():
                    units.setdefault(unit, []).append(value)
            if units:
                entry["throughput"] = {
                    unit: _series_stats(tuple(values)) for unit, values in units.items()
                }
                entry["series"]["throughput"] = units
            graphs[name] = entry

        return {"runs": sum(item["count"] for item in graphs.values()), "graphs": graphs}

    def summary(self) -> Dict[str, Any]:
        """聚合摘要：运行次数、总耗时、按算子的累计派发耗时、最大显存峰值。"""
        calls: Dict[str, int] = {}
        total: Dict[str, float] = {}
        peak: Optional[int] = None
        for record in self._records:
            for op, count, ms in record.node_totals():
                calls[op] = calls.get(op, 0) + count
                total[op] = total.get(op, 0.0) + ms
            if record.memory is not None:
                value = record.memory.peak_bytes
                peak = value if peak is None else max(peak, value)
        ops = [
            {
                "op": op,
                "calls": calls[op],
                "dispatch_ms": round(total[op], 6),
                "share": round(total[op] / sum(total.values()), 4) if sum(total.values()) else 0.0,
            }
            for op in total
        ]
        ops.sort(key=lambda row: (-row["dispatch_ms"], row["op"]))
        out: Dict[str, Any] = {
            "runs": len(self._records),
            "wall_ms_total": round(sum(r.wall_ms for r in self._records), 6),
            "operators": ops,
        }
        if peak is not None:
            out["peak_bytes_max"] = peak
        if self._records:
            last = self._records[-1]
            out["last"] = {
                "graph": last.graph,
                "backend": last.backend,
                "wall_ms": last.wall_ms,
            }
        return out

    def save(self, path: str | os.PathLike) -> None:
        """把全部记录写成 JSON（含摘要）。这是上报，不是配置库，所以整体覆盖。"""
        payload = {
            "summary": self.summary(),
            "records": [record.to_dict() for record in self._records],
        }
        target = os.fspath(path)
        tmp = f"{target}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)


def load_records(path: str | os.PathLike) -> Tuple[RunRecord, ...]:
    """读回 ``TelemetryRecorder.save()`` 写出的记录，用于跨运行的汇总。

    ``summary`` 是派生产物，不读——它可能来自不同版本的实现，而记录本身
    结构稳定。文件不是这个格式（缺 ``records``）时明确失败，不猜。
    """
    with open(os.fspath(path), encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError(f"{path} 不是 TelemetryRecorder.save() 的产物：缺少 records")
    return tuple(RunRecord.from_dict(item) for item in payload["records"])
