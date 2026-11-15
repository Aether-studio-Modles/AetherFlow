"""Chrome Trace 导出的本地测试（5.243）。

要钉住的是"图上的时间是真的"：``ts`` / ``dur`` 必须来自记录里的
``start_ms`` / ``dispatch_ms``（换算成微秒），缺时刻时按顺序补齐并在附注里
计数；GPU 轨道是首尾相接的占位，不能伪装成真实时刻。
"""

import json
import os
import tempfile
import unittest

from aetherflow import Graph, f32
from aetherflow.runtime import ReferenceExecutor
from aetherflow.runtime.telemetry import (
    NodeTiming,
    RunRecord,
    TelemetryRecorder,
)
from aetherflow.runtime.trace import (
    CPU_TRACK,
    GPU_TRACK,
    build_trace,
    trace_events,
    write_trace,
)
import numpy as np


def _record(nodes, *, wall_ms=2.0, graph="g", label=None, device=None):
    return RunRecord(
        graph=graph,
        backend="reference",
        wall_ms=wall_ms,
        nodes=tuple(nodes),
        label=label,
        device=device,
    )


def _kinds(events):
    return {event["cat"] for event in events if event["ph"] == "X"}


class TestCpuTimeline(unittest.TestCase):
    def test_measured_start_becomes_event_timestamp(self):
        record = _record(
            [
                NodeTiming("add", "v1", 0.25, start_ms=0.5),
                NodeTiming("relu", "v2", 0.5, start_ms=1.0),
            ]
        )
        events, notes = trace_events(record)
        spans = [event for event in events if event["ph"] == "X"]
        self.assertEqual([event["ts"] for event in spans], [500.0, 1000.0])
        self.assertEqual([event["dur"] for event in spans], [250.0, 500.0])
        self.assertEqual(notes["synthesized_nodes"], 0)
        self.assertEqual(notes["records"], 1)

    def test_missing_start_is_filled_by_cumulative_sum(self):
        record = _record(
            [
                NodeTiming("add", "v1", 0.2),
                NodeTiming("relu", "v2", 0.3),
            ]
        )
        events, notes = trace_events(record)
        spans = [event for event in events if event["ph"] == "X"]
        self.assertEqual([event["ts"] for event in spans], [0.0, 200.0])
        self.assertEqual(notes["synthesized_nodes"], 2)

    def test_only_missing_rows_are_filled(self):
        record = _record(
            [
                NodeTiming("add", "v1", 0.2, start_ms=1.0),
                NodeTiming("relu", "v2", 0.3),
                NodeTiming("mul", "v3", 0.1, start_ms=3.0),
            ]
        )
        events, notes = trace_events(record)
        spans = [event for event in events if event["ph"] == "X"]
        self.assertEqual([event["ts"] for event in spans], [1000.0, 1200.0, 3000.0])
        self.assertEqual(notes["synthesized_nodes"], 1)

    def test_event_name_and_args_identify_the_node(self):
        record = _record([NodeTiming("matmul", "y", 0.5, start_ms=0.0)])
        events, _ = trace_events(record)
        span = next(event for event in events if event["ph"] == "X")
        self.assertEqual(span["name"], "matmul (y)")
        self.assertEqual(span["args"], {"op": "matmul", "output": "y"})
        self.assertEqual(span["tid"], 1)


class TestGpuTrack(unittest.TestCase):
    def test_gpu_events_are_sequential_from_first_reading(self):
        record = _record(
            [
                NodeTiming("add", "v1", 0.2, gpu_ms=1.0, start_ms=2.0),
                NodeTiming("relu", "v2", 0.2, start_ms=3.0),
                NodeTiming("mul", "v3", 0.2, gpu_ms=0.5, start_ms=4.0),
            ]
        )
        events, _ = trace_events(record)
        gpu = [event for event in events if event.get("cat") == "gpu"]
        self.assertEqual([event["ts"] for event in gpu], [2000.0, 3000.0])
        self.assertEqual([event["dur"] for event in gpu], [1000.0, 500.0])
        self.assertTrue(all(event["args"]["approximate"] for event in gpu))

    def test_without_gpu_readings_there_is_no_gpu_track(self):
        record = _record([NodeTiming("add", "v1", 0.2, start_ms=0.0)])
        events, _ = trace_events(record)
        self.assertEqual(_kinds(events), {"dispatch"})

    def test_metadata_names_both_tracks(self):
        record = _record([NodeTiming("add", "v1", 0.2, start_ms=0.0)])
        events, _ = trace_events(record)
        names = {
            event["args"]["name"]
            for event in events
            if event["ph"] == "M" and event["name"] == "thread_name"
        }
        self.assertEqual(names, {CPU_TRACK, GPU_TRACK})


class TestRecordsAndLayout(unittest.TestCase):
    def test_each_record_gets_its_own_process_and_offset(self):
        first = _record([NodeTiming("add", "v1", 0.4, start_ms=0.0)], wall_ms=1.0, label="tiny")
        second = _record([NodeTiming("add", "v1", 0.4, start_ms=0.0)], wall_ms=2.0, label="big")
        events, notes = trace_events([first, second])
        spans = [event for event in events if event["ph"] == "X"]
        self.assertEqual([event["pid"] for event in spans], [1, 2])
        self.assertEqual([event["ts"] for event in spans], [0.0, 1000.0])
        self.assertEqual(notes["records"], 2)
        titles = {
            event["pid"]: event["args"]["name"]
            for event in events
            if event["name"] == "process_name"
        }
        self.assertEqual(titles, {1: "tiny [reference]", 2: "big [reference]"})

    def test_device_joins_the_process_title(self):
        record = _record([], label="tiny", device="RTX 2080 Ti")
        events, _ = trace_events(record)
        title = next(event for event in events if event["name"] == "process_name")
        self.assertEqual(title["args"]["name"], "tiny @ RTX 2080 Ti [reference]")

    def test_empty_record_still_produces_valid_trace(self):
        payload = build_trace(_record([]))
        self.assertEqual(payload["traceEvents"][0]["ph"], "M")
        self.assertEqual(payload["displayTimeUnit"], "ms")
        self.assertEqual(payload["otherData"]["events"], len(payload["traceEvents"]))

    def test_repeated_export_is_stable(self):
        record = _record([NodeTiming("add", "v1", 0.2, start_ms=0.1)])
        self.assertEqual(trace_events(record), trace_events(record))


class TestArgumentChecks(unittest.TestCase):
    def test_non_record_is_rejected(self):
        with self.assertRaises(TypeError):
            trace_events([object()])

    def test_negative_offset_is_rejected(self):
        with self.assertRaises(ValueError):
            trace_events(_record([]), offset_ms=-1.0)

    def test_bad_pid_is_rejected(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    trace_events(_record([]), start_pid=value)

    def test_negative_start_ms_is_rejected(self):
        with self.assertRaises(ValueError):
            _record([NodeTiming("add", "v1", 0.2, start_ms=-0.5)])


class TestSerializationAndFiles(unittest.TestCase):
    def test_start_ms_survives_json_round_trip(self):
        record = _record([NodeTiming("add", "v1", 0.2, start_ms=0.75)])
        payload = record.to_dict()
        self.assertEqual(payload["nodes"][0]["start_ms"], 0.75)
        again = RunRecord.from_dict(json.loads(json.dumps(payload)))
        self.assertEqual(again, record)

    def test_missing_start_ms_is_absent_from_json(self):
        record = _record([NodeTiming("add", "v1", 0.2)])
        self.assertNotIn("start_ms", record.to_dict()["nodes"][0])
        self.assertEqual(RunRecord.from_dict(record.to_dict()), record)

    def test_write_trace_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trace.json")
            payload = write_trace(path, _record([NodeTiming("add", "v1", 0.2, start_ms=0.0)]))
            self.assertTrue(os.path.exists(path))
            self.assertEqual(os.listdir(tmp), ["trace.json"])
            with open(path, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh), payload)

    def test_cli_converts_a_telemetry_file(self):
        from aetherflow.runtime.trace import _main

        graph = Graph("add_relu")
        a = graph.input("a", f32[4, 8])
        b = graph.input("b", f32[4, 8])
        graph.output(graph.relu(graph.add(a, b)), "y")
        rng = np.random.default_rng(5)
        feed = {
            "a": rng.standard_normal((4, 8)).astype(np.float32),
            "b": rng.standard_normal((4, 8)).astype(np.float32),
        }
        recorder = TelemetryRecorder()
        recorder.wrap(ReferenceExecutor()).execute(graph, feed)

        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "telemetry.json")
            out = os.path.join(tmp, "trace.json")
            recorder.save(source)
            self.assertEqual(_main([source, "-o", out]), 0)
            with open(out, encoding="utf-8") as fh:
                payload = json.load(fh)
        self.assertEqual(payload["otherData"]["synthesized_nodes"], 0)
        self.assertEqual(_kinds(payload["traceEvents"]), {"dispatch"})


class TestWrappedBackendTimeline(unittest.TestCase):
    """包装器采到的时刻必须能让整条时间轴自洽：非负、单调不减。"""

    def test_every_node_gets_a_monotonic_start(self):
        graph = Graph("add_relu")
        a = graph.input("a", f32[4, 8])
        b = graph.input("b", f32[4, 8])
        graph.output(graph.relu(graph.add(a, b)), "y")
        rng = np.random.default_rng(7)
        feed = {
            "a": rng.standard_normal((4, 8)).astype(np.float32),
            "b": rng.standard_normal((4, 8)).astype(np.float32),
        }
        recorder = TelemetryRecorder()
        recorder.wrap(ReferenceExecutor()).execute(graph, feed)
        record = recorder.last

        starts = [item.start_ms for item in record.nodes]
        self.assertTrue(all(value is not None for value in starts))
        self.assertGreaterEqual(starts[0], 0.0)
        self.assertEqual(starts, sorted(starts))
        self.assertLessEqual(starts[-1], record.wall_ms)

        events, notes = trace_events(record)
        self.assertEqual(notes["synthesized_nodes"], 0)
        spans = [event for event in events if event["ph"] == "X"]
        self.assertEqual(len(spans), len(record.nodes))


if __name__ == "__main__":
    unittest.main()
