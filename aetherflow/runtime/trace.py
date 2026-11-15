"""把 ``RunRecord`` 导出成 Chrome Trace Event Format（可直接拖进 chrome://tracing / Perfetto）。

**为什么是"导出"而不是内置 profiler。** 观测数据在 5.161 / 5.243 已经采齐：
每个节点的派发时长、可选的 GPU 时长、以及相对 ``execute`` 起点的派发时刻。
剩下的只是换一种表示法——浏览器能读的那种。真做一个采样 profiler 要抓栈、
做符号化、处理多线程聚合，那是另一件事，而眼下要回答的问题（"这 12ms 里
时间花在哪些节点、中间空了多少"）用已有数据就能回答。

**时间轴是真的，间隔也是真的。** ``NodeTiming.start_ms`` 由包装器在
``invoke`` 前用 ``perf_counter`` 记下，零点与 ``wall_ms`` 相同。因此图上
节点之间的空白是"CPU 没在派发"（在等设备、在 Python 里算形状、在同步），
不是画图造出来的。旧记录没有 ``start_ms`` 时按顺序累加补齐，
``otherData.synthesized_nodes`` 会写明补了几个。

**GPU 轨道是"累计占位"，不是真实时刻。** 事件对只给逐节点时长，不给
绝对时刻（见 ``telemetry`` 模块文档），所以这里按顺序把它们首尾相接排开，
线程名直接写成"GPU 内核（累计占位）"。它能读占比与顺序，不能读"哪个 kernel
和 CPU 的哪一段重叠"。

单位：Chrome Trace 的 ``ts`` / ``dur`` 是微秒，记录里是毫秒；``displayTimeUnit``
设成 ``ms``，界面上仍按毫秒显示。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .telemetry import NodeTiming, RunRecord

__all__ = [
    "CPU_TRACK",
    "GPU_TRACK",
    "trace_events",
    "build_trace",
    "write_trace",
]

CPU_TRACK = "CPU 派发"
GPU_TRACK = "GPU 内核（累计占位）"

_TID_CPU = 1
_TID_GPU = 2


def _node_end(item: NodeTiming, start: float) -> float:
    return start + item.dispatch_ms


def _cpu_starts(nodes: Sequence[NodeTiming]) -> Tuple[List[float], int]:
    """每一行的派发起点（毫秒）；返回 ``(起点, 补齐条数)``。

    有 ``start_ms`` 就用实测值；没有时接在上一个节点的结束处（间隔 0）。
    混合记录逐行判断：只有缺的那些行被补，实测行保持原样。
    """
    starts: List[float] = []
    synthesized = 0
    cursor = 0.0
    for item in nodes:
        if item.start_ms is None:
            starts.append(cursor)
            synthesized += 1
        else:
            starts.append(float(item.start_ms))
        cursor = _node_end(item, starts[-1])
    return starts, synthesized


def _event(
    *,
    name: str,
    cat: str,
    pid: int,
    tid: int,
    start_ms: float,
    dur_ms: float,
    args: Optional[Dict[str, Any]] = None,
    offset_ms: float = 0.0,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ph": "X",
        "name": name,
        "cat": cat,
        "pid": pid,
        "tid": tid,
        "ts": round((offset_ms + start_ms) * 1e3, 3),
        "dur": round(dur_ms * 1e3, 3),
    }
    if args:
        payload["args"] = args
    return payload


def _metadata(pid: int, title: str) -> List[Dict[str, Any]]:
    return [
        {"ph": "M", "name": "process_name", "pid": pid, "args": {"name": title}},
        {"ph": "M", "name": "thread_name", "pid": pid, "tid": _TID_CPU, "args": {"name": CPU_TRACK}},
        {"ph": "M", "name": "thread_name", "pid": pid, "tid": _TID_GPU, "args": {"name": GPU_TRACK}},
        {"ph": "M", "name": "thread_sort_index", "pid": pid, "tid": _TID_CPU, "args": {"sort_index": 1}},
        {"ph": "M", "name": "thread_sort_index", "pid": pid, "tid": _TID_GPU, "args": {"sort_index": 2}},
    ]


def trace_events(
    records: Iterable[RunRecord] | RunRecord,
    *,
    start_pid: int = 1,
    offset_ms: float = 0.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """展开成 ``(事件列表, 附注)``。

    一份记录一个"进程"（``pid`` 递增），进程名取 ``label or graph`` 加上后端，
    这样同一份图在不同后端上的两次执行在界面上是两行。多次执行之间按
    ``offset_ms`` 顺序往后排——它们是不同时刻的两次 ``execute``，不能叠加。
    """
    if isinstance(records, RunRecord):
        ordered: Tuple[RunRecord, ...] = (records,)
    else:
        ordered = tuple(records)
        for item in ordered:
            if not isinstance(item, RunRecord):
                raise TypeError(
                    f"只接受 RunRecord，得到 {type(item).__name__}"
                )
    if not isinstance(start_pid, int) or isinstance(start_pid, bool) or start_pid < 1:
        raise ValueError(f"start_pid 必须是正整数，得到 {start_pid!r}")
    if offset_ms < 0:
        raise ValueError(f"offset_ms 不能为负，得到 {offset_ms!r}")

    events: List[Dict[str, Any]] = []
    synthesized_total = 0
    cursor = offset_ms
    for index, record in enumerate(ordered):
        pid = start_pid + index
        title = record.label or record.graph
        if record.device:
            title = f"{title} @ {record.device}"
        events.extend(_metadata(pid, f"{title} [{record.backend}]"))

        starts, synthesized = _cpu_starts(record.nodes)
        synthesized_total += synthesized
        for item, start in zip(record.nodes, starts):
            events.append(
                _event(
                    name=f"{item.op} ({item.name})",
                    cat="dispatch",
                    pid=pid,
                    tid=_TID_CPU,
                    start_ms=start,
                    dur_ms=item.dispatch_ms,
                    offset_ms=cursor,
                    args={"op": item.op, "output": item.name},
                )
            )

        # GPU 轨道：同一次执行内按顺序首尾相接（见模块文档，不是真实时刻）。
        gpu_cursor = 0.0
        gpu_started = False
        for item, start in zip(record.nodes, starts):
            if item.gpu_ms is None:
                continue
            if not gpu_started:
                gpu_cursor = start
                gpu_started = True
            events.append(
                _event(
                    name=f"{item.op} ({item.name})",
                    cat="gpu",
                    pid=pid,
                    tid=_TID_GPU,
                    start_ms=gpu_cursor,
                    dur_ms=item.gpu_ms,
                    offset_ms=cursor,
                    args={"op": item.op, "gpu_ms": item.gpu_ms, "approximate": True},
                )
            )
            gpu_cursor += item.gpu_ms

        cursor += max(record.wall_ms, 0.0)

    notes = {
        "records": len(ordered),
        "events": len(events),
        "synthesized_nodes": synthesized_total,
        "gpu_timeline": "sequential",
    }
    return events, notes


def build_trace(
    records: Iterable[RunRecord] | RunRecord,
    *,
    start_pid: int = 1,
    offset_ms: float = 0.0,
) -> Dict[str, Any]:
    """包成 Chrome Trace 的顶层结构（可直接 ``json.dump``）。"""
    events, notes = trace_events(records, start_pid=start_pid, offset_ms=offset_ms)
    return {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "otherData": notes,
    }


def write_trace(
    path: str | os.PathLike,
    records: Iterable[RunRecord] | RunRecord,
    *,
    start_pid: int = 1,
    offset_ms: float = 0.0,
) -> Dict[str, Any]:
    """写到 ``path`` 并返回顶层结构（便于调用方打印附注）。

    与 ``TelemetryRecorder.save`` 一样先写临时文件再 ``os.replace``：trace 文件
    常在 CI 里边生成边被 artifact 上传，半个 JSON 比没有更糟。
    """
    payload = build_trace(records, start_pid=start_pid, offset_ms=offset_ms)
    target = os.fspath(path)
    tmp = f"{target}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)
    return payload


def _main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m aetherflow.runtime.trace telemetry.json [-o trace.json]``。

    输入是 ``TelemetryRecorder.save()`` 的产物，输出可以直接拖进
    chrome://tracing 或 Perfetto。默认输出名是在输入名后加 ``.trace.json``。
    """
    import argparse

    from .telemetry import load_records

    parser = argparse.ArgumentParser(description="把运行时上报导出成 Chrome Trace")
    parser.add_argument("telemetry", help="TelemetryRecorder.save() 写出的 JSON")
    parser.add_argument("-o", "--out", help="输出路径（默认 <输入>.trace.json）")
    args = parser.parse_args(argv)

    records = load_records(args.telemetry)
    out = args.out or f"{args.telemetry}.trace.json"
    payload = write_trace(out, records)
    notes = payload["otherData"]
    print(
        f"{out}: {notes['events']} 个事件 / {notes['records']} 次执行"
        f"（补齐时刻 {notes['synthesized_nodes']} 个节点）"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - 命令行入口
    raise SystemExit(_main())
