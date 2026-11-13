"""运行时上报的本地测试：记录结构、委托语义、摘要与落盘。

两台机器上都要成立的三条：

1. 包装**不改变结果**——同一张图包不包上报，输出必须逐位相同；
2. 没有观测就是 ``None``，不是 0（参考执行器没有设备、没有分层分配器）；
3. 上报不改图、不改配置，只读。
"""

import json
import os
import tempfile
import unittest

import numpy as np

from aetherflow import Graph, f32
from aetherflow.backends.contracts import (
    CONTRACTS,
    DeviceInfo,
    MemoryStats,
    implemented_contracts,
)
from aetherflow.runtime import ReferenceExecutor
from aetherflow.runtime.telemetry import (
    InstrumentedBackend,
    NodeTiming,
    RunRecord,
    TelemetryRecorder,
    load_records,
)


def _small_graph():
    g = Graph("add_relu")
    a = g.input("a", f32[4, 8])
    b = g.input("b", f32[4, 8])
    g.output(g.relu(g.add(a, b)), "y")
    rng = np.random.default_rng(3)
    return g, {
        "a": rng.standard_normal((4, 8)).astype(np.float32),
        "b": rng.standard_normal((4, 8)).astype(np.float32),
    }


class _FullContractExecutor(ReferenceExecutor):
    """六项可选契约全齐的假后端：验证包装器逐项转发（5.163 / 5.164）。"""

    name = "full"

    def device_info(self):
        return DeviceInfo(name="fake", runtime=1, total_memory_bytes=1 << 30)

    def enumerate_devices(self):
        return (self.device_info(),)

    def current_device(self):
        return int(getattr(self, "device", 0))

    def set_device(self, index):
        self.device = index

    def kernel(self, key, source, func):
        return (key, source, func)

    def tunable_ops(self):
        return frozenset({"relu"})

    def tuning_space(self, op):
        return {"block": (None,)}

    def set_tuning(self, op, config):
        self.tuning = {op: dict(config)}

    def clear_tuning(self, op=None):
        if op is None:
            self.tuning = {}
        else:
            self.tuning.pop(op, None)

    def tuning_config(self):
        return dict(getattr(self, "tuning", {}))

    def memory_stats(self):
        return MemoryStats(live_bytes=1, peak_bytes=2, total_bytes=3, requested_bytes=4)

    def allocator(self):
        return object()

    def new_stream(self):
        return object()


class TestRunRecord(unittest.TestCase):
    def test_negative_wall_time_is_rejected(self):
        with self.assertRaises(ValueError):
            RunRecord(graph="g", backend="ref", wall_ms=-1.0)

    def test_non_finite_wall_time_is_rejected(self):
        with self.assertRaises(ValueError):
            RunRecord(graph="g", backend="ref", wall_ms=float("nan"))

    def test_throughput_requires_units_and_time(self):
        record = RunRecord(graph="g", backend="ref", wall_ms=0.0, units={"elements": 100})
        self.assertEqual(record.throughput(), {})
        record = RunRecord(graph="g", backend="ref", wall_ms=10.0, units={"elements": 100})
        self.assertEqual(record.throughput(), {"elements": 10000.0})

    def test_node_totals_aggregate_by_operator(self):
        record = RunRecord(
            graph="g",
            backend="ref",
            wall_ms=1.0,
            nodes=(
                NodeTiming("add", "v1", 0.2),
                NodeTiming("relu", "v2", 0.5),
                NodeTiming("add", "v3", 0.1),
            ),
        )
        self.assertEqual(
            record.node_totals(),
            (("relu", 1, 0.5), ("add", 2, 0.3)),
        )

    def test_to_dict_omits_missing_observations(self):
        payload = RunRecord(graph="g", backend="ref", wall_ms=1.0).to_dict()
        self.assertIsNone(payload["device"])
        self.assertNotIn("memory", payload)
        self.assertNotIn("units", payload)


class TestInstrumentedBackend(unittest.TestCase):
    def test_outputs_are_identical_and_every_node_is_timed(self):
        graph, feed = _small_graph()
        expected = ReferenceExecutor().run(graph, feed)

        recorder = TelemetryRecorder()
        wrapped = recorder.wrap(ReferenceExecutor())
        actual = wrapped.execute(graph, feed, units={"elements": 32})

        np.testing.assert_array_equal(actual["y"], expected["y"])
        record = recorder.last
        self.assertIsNotNone(record)
        self.assertEqual(record.graph, "add_relu")
        self.assertEqual(record.backend, "reference")
        self.assertEqual(len(record.nodes), len(graph.nodes))
        self.assertEqual({item.op for item in record.nodes}, {"add", "relu"})
        self.assertGreaterEqual(record.wall_ms, 0.0)

    def test_missing_device_and_memory_stay_none(self):
        graph, feed = _small_graph()
        recorder = TelemetryRecorder()
        recorder.wrap(ReferenceExecutor()).execute(graph, feed)
        self.assertIsNone(recorder.last.device)
        self.assertIsNone(recorder.last.memory)

    def test_wrapper_does_not_claim_optional_contracts(self):
        """参考执行器没有设备/内存契约，包一层之后也不能凭空"声称"有。"""
        wrapped = TelemetryRecorder().wrap(ReferenceExecutor())
        self.assertEqual(implemented_contracts(wrapped), ())

    def test_wrapper_forwards_every_optional_contract(self):
        """内层六项契约全有，包装后必须一项不少（5.163 的编译期/调优、5.164 的多设备）。

        漏转不会报错，只会让包装后的后端"静默少认一项"：基准会把多卡看成
        单卡、调优工具会以为这台机器没有旋钮。所以按 ``CONTRACTS`` 整表断言。
        """
        wrapped = TelemetryRecorder().wrap(_FullContractExecutor())
        self.assertEqual(implemented_contracts(wrapped), tuple(CONTRACTS))

    def test_forwarded_calls_reach_the_inner_backend(self):
        inner = _FullContractExecutor()
        wrapped = TelemetryRecorder().wrap(inner)
        self.assertEqual(wrapped.current_device(), 0)
        wrapped.set_device(1)
        self.assertEqual(inner.device, 1)
        self.assertEqual(wrapped.enumerate_devices(), inner.enumerate_devices())
        wrapped.set_tuning("relu", {"block": 256})
        self.assertEqual(inner.tuning_config(), {"relu": {"block": 256}})
        wrapped.clear_tuning()
        self.assertEqual(wrapped.tuning_config(), {})

    def test_partial_contracts_are_not_promoted(self):
        """只有 ``device_info`` 的内层，包装后仍然只认 ``device`` 一项。"""

        class _DeviceOnly(ReferenceExecutor):
            name = "device_only"

            def device_info(self):
                return DeviceInfo(name="fake", runtime=1, total_memory_bytes=1)

        wrapped = TelemetryRecorder().wrap(_DeviceOnly())
        self.assertEqual(implemented_contracts(wrapped), ("device",))

    def test_double_wrap_is_idempotent(self):
        recorder = TelemetryRecorder()
        wrapped = recorder.wrap(ReferenceExecutor())
        self.assertIs(recorder.wrap(wrapped), wrapped)

    def test_non_executor_is_rejected(self):
        with self.assertRaises(TypeError):
            TelemetryRecorder().wrap(object())

    def test_capacity_keeps_recent_records(self):
        graph, feed = _small_graph()
        recorder = TelemetryRecorder(capacity=2)
        wrapped = recorder.wrap(ReferenceExecutor())
        for _ in range(3):
            wrapped.execute(graph, feed)
        self.assertEqual(len(recorder.records), 2)
        self.assertEqual(recorder.summary()["runs"], 2)

    def test_invalid_capacity_is_rejected(self):
        with self.assertRaises(ValueError):
            TelemetryRecorder(capacity=0)

    def test_summary_and_save(self):
        graph, feed = _small_graph()
        recorder = TelemetryRecorder()
        wrapped = recorder.wrap(ReferenceExecutor())
        wrapped.execute(graph, feed)
        wrapped.execute(graph, feed)

        summary = recorder.summary()
        self.assertEqual(summary["runs"], 2)
        ops = {item["op"]: item for item in summary["operators"]}
        self.assertEqual(ops["add"]["calls"], 2)
        self.assertEqual(ops["relu"]["calls"], 2)
        self.assertIn("last", summary)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "telemetry.json")
            recorder.save(path)
            self.assertFalse(os.path.exists(f"{path}.tmp"))
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertEqual(payload["summary"]["runs"], 2)
            self.assertEqual(len(payload["records"]), 2)
            self.assertEqual(payload["records"][0]["graph"], "add_relu")

    def test_recording_does_not_mutate_the_graph(self):
        graph, feed = _small_graph()
        before = [(node.op.name, tuple(node.attrs)) for node in graph.nodes]
        recorder = TelemetryRecorder()
        recorder.wrap(ReferenceExecutor()).execute(graph, feed)
        after = [(node.op.name, tuple(node.attrs)) for node in graph.nodes]
        self.assertEqual(before, after)


class _FakeClock:
    """按调用顺序返回固定耗时的假时钟（5.161）。"""

    def __init__(self, values):
        self._values = list(values)
        self.released = []
        self.synchronized = 0

    def begin(self):
        return object()

    def end(self):
        return object()

    def synchronize(self):
        self.synchronized += 1

    def elapsed(self, start, stop):
        return self._values.pop(0)

    def release(self, start, stop):
        self.released.append((start, stop))


class TestGpuNodeTiming(unittest.TestCase):
    """5.161：逐节点 GPU 计时只在后端提供 ``gpu_clock`` 时出现。

    参考执行器没有这个时钟，``gpu_ms`` 必须保持 ``None``；换成带时钟的替身
    后，每个节点拿到一个读数、整次执行只同步一次、事件在收尾时释放。
    """

    @staticmethod
    def _with_clock(values, **kwargs):
        graph, feed = _small_graph()
        executor = ReferenceExecutor()
        clock = _FakeClock(values)
        executor.gpu_clock = lambda: clock
        recorder = TelemetryRecorder()
        out = recorder.wrap(executor, **kwargs).execute(graph, feed)
        return out, recorder.last, clock

    def test_each_node_gets_a_gpu_reading(self):
        _, record, clock = self._with_clock([0.5, 1.25])
        self.assertEqual([item.gpu_ms for item in record.nodes], [0.5, 1.25])
        self.assertEqual(clock.synchronized, 1, "只应在收尾同步一次")
        self.assertEqual(len(clock.released), 2, "每对事件都要释放")

    def test_totals_and_json_carry_gpu_readings(self):
        _, record, _ = self._with_clock([0.5, 1.25])
        self.assertEqual(
            record.node_gpu_totals(), (("relu", 1, 1.25), ("add", 1, 0.5))
        )
        payload = record.to_dict()
        self.assertEqual(payload["nodes"][0]["gpu_ms"], 0.5)
        self.assertIn("node_gpu_totals", payload)
        again = RunRecord.from_dict(json.loads(json.dumps(payload)))
        self.assertEqual([item.gpu_ms for item in again.nodes], [0.5, 1.25])

    def test_without_clock_every_gpu_field_is_absent(self):
        graph, feed = _small_graph()
        recorder = TelemetryRecorder()
        recorder.wrap(ReferenceExecutor()).execute(graph, feed)
        record = recorder.last
        self.assertTrue(all(item.gpu_ms is None for item in record.nodes))
        self.assertEqual(record.node_gpu_totals(), ())
        self.assertNotIn("gpu_ms", record.to_dict()["nodes"][0])
        self.assertNotIn("node_gpu_totals", record.to_dict())

    def test_gpu_timing_can_be_switched_off(self):
        _, record, clock = self._with_clock([9.0, 9.0], gpu_timing=False)
        self.assertTrue(all(item.gpu_ms is None for item in record.nodes))
        self.assertEqual(clock.synchronized, 0)

    def test_failed_reading_keeps_execution_alive(self):
        graph, feed = _small_graph()
        executor = ReferenceExecutor()

        class _Broken(_FakeClock):
            def elapsed(self, start, stop):
                raise RuntimeError("驱动不给读数")

        clock = _Broken([])
        executor.gpu_clock = lambda: clock
        recorder = TelemetryRecorder()
        out = recorder.wrap(executor).execute(graph, feed)
        self.assertIn("y", out)
        self.assertTrue(all(item.gpu_ms is None for item in recorder.last.nodes))
        self.assertEqual(len(clock.released), 2, "读数失败也必须释放事件")

    def test_negative_gpu_reading_is_rejected(self):
        graph, feed = _small_graph()
        executor = ReferenceExecutor()
        executor.gpu_clock = lambda: _FakeClock([-1.0, 1.0])
        recorder = TelemetryRecorder()
        with self.assertRaises(ValueError):
            recorder.wrap(executor).execute(graph, feed)


class TestTrend(unittest.TestCase):
    def _record(self, wall_ms, *, graph="g", label=None, peak=None, units=None):
        memory = MemoryStats(1, peak, 2, 3) if peak is not None else None
        return RunRecord(
            graph=graph,
            backend="reference",
            wall_ms=wall_ms,
            memory=memory,
            units=units,
            label=label,
        )

    def test_from_dict_round_trip(self):
        record = self._record(1.5, label="tiny", peak=2048, units={"elements": 32})
        self.assertEqual(RunRecord.from_dict(record.to_dict()), record)

    def test_save_and_load_round_trip(self):
        recorder = TelemetryRecorder()
        recorder.record(self._record(1.0, label="tiny"))
        recorder.record(self._record(2.0, label="tiny", peak=64))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "telemetry.json")
            recorder.save(path)
            loaded = load_records(path)
        self.assertEqual(loaded, recorder.records)

    def test_load_rejects_foreign_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "foreign.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"hello": 1}, fh)
            with self.assertRaises(ValueError):
                load_records(path)

    def test_extend_appends_in_order(self):
        first = TelemetryRecorder()
        first.record(self._record(1.0))
        second = TelemetryRecorder()
        second.record(self._record(2.0))
        second.extend(first.records)
        self.assertEqual([item.wall_ms for item in second.records], [2.0, 1.0])

    def test_label_separates_same_named_graphs(self):
        recorder = TelemetryRecorder()
        recorder.record(self._record(1.0, graph="matmul", label="matmul_256"))
        recorder.record(self._record(9.0, graph="matmul", label="matmul_1024"))
        trend = recorder.trend()
        self.assertEqual(set(trend["graphs"]), {"matmul_256", "matmul_1024"})
        self.assertEqual(trend["graphs"]["matmul_1024"]["graph"], "matmul")

    def test_trend_slope_and_change(self):
        recorder = TelemetryRecorder()
        for wall in (1.0, 2.0, 3.0):
            recorder.record(self._record(wall))
        stats = recorder.trend()["graphs"]["g"]["wall_ms"]
        self.assertEqual(stats["median"], 2.0)
        self.assertEqual(stats["slope_per_run"], 1.0)
        self.assertEqual(stats["change_pct"], 200.0)

    def test_window_keeps_recent_runs(self):
        recorder = TelemetryRecorder()
        for wall in (1.0, 2.0, 3.0):
            recorder.record(self._record(wall))
        entry = recorder.trend(window=2)["graphs"]["g"]
        self.assertEqual(entry["count"], 2)
        self.assertEqual(entry["series"]["wall_ms"], [2.0, 3.0])

    def test_trend_carries_peak_and_throughput(self):
        recorder = TelemetryRecorder()
        recorder.record(self._record(10.0, peak=100, units={"elements": 1000}))
        recorder.record(self._record(20.0, peak=300, units={"elements": 1000}))
        entry = recorder.trend()["graphs"]["g"]
        self.assertEqual(entry["peak_bytes"]["max"], 300.0)
        self.assertAlmostEqual(entry["throughput"]["elements"]["median"], 75000.0, places=2)
        self.assertEqual(entry["series"]["throughput"]["elements"][0], 100000.0)

    def test_filter_and_invalid_window(self):
        recorder = TelemetryRecorder()
        recorder.record(self._record(1.0, graph="a"))
        recorder.record(self._record(2.0, graph="b"))
        self.assertEqual(set(recorder.trend(graph="a")["graphs"]), {"a"})
        with self.assertRaises(ValueError):
            recorder.trend(window=0)

    def test_empty_trend_is_empty(self):
        self.assertEqual(TelemetryRecorder().trend(), {"runs": 0, "graphs": {}})


if __name__ == "__main__":
    unittest.main()
