"""智能体接口的测试：只读、可序列化、查询不成立时明确失败。

这一层最容易被"顺手加个执行入口"侵蚀，所以测试同时钉住**能力**和**边界**：
能查到什么，以及它不该能做什么。
"""

import json
import unittest

from aetherflow import DeviceInfo, Graph, ReferenceBackend, f32
from aetherflow import agent_api
from aetherflow.agent_api import (
    AgentQueryError,
    describe_backend,
    describe_backends,
    describe_operator,
    performance_summary,
    recommend_config,
    snapshot,
)
from aetherflow.runtime import ReferenceExecutor
from aetherflow.runtime.telemetry import TelemetryRecorder
from aetherflow.tuning import ConfigStore, TuningRecord, TuningSpace


class _FakeTunable:
    """一个最小的可调后端外观：只实现接口，不执行任何东西。"""

    name = "fake"

    def capabilities(self):
        return frozenset({"add"})

    def device_info(self):
        return DeviceInfo(name="fake-gpu", runtime=12090, total_memory_bytes=1 << 30)

    def tunable_ops(self):
        return frozenset({"add"})

    def tuning_space(self, op):
        if op != "add":
            raise ValueError(op)
        return TuningSpace.build({"block": (64, 128)})

    def set_tuning(self, op, config):
        return None


class TestDescribe(unittest.TestCase):
    def test_reference_backend_profile(self):
        profile = describe_backend(ReferenceBackend())
        self.assertEqual(profile["name"], "reference")
        self.assertIn("add", profile["capabilities"])
        self.assertEqual(profile["contracts"], [])
        self.assertIsNone(profile["device"])
        self.assertEqual(profile["tunable_ops"], [])
        self.assertEqual(profile["operators"]["add"]["variants"], [])

    def test_operator_profile(self):
        self.assertTrue(describe_operator(ReferenceBackend(), "relu")["implemented"])
        missing = describe_operator(ReferenceBackend(), "not_an_op")
        self.assertFalse(missing["implemented"])
        self.assertIsNone(missing["tunable"])

    def test_empty_operator_name_is_rejected(self):
        with self.assertRaises(AgentQueryError):
            describe_operator(ReferenceBackend(), "")

    def test_tunable_operator_reports_space(self):
        profile = describe_operator(_FakeTunable(), "add")
        self.assertTrue(profile["implemented"])
        self.assertEqual(profile["tunable"], {"block": [64, 128]})

    def test_device_is_reported_when_available(self):
        device = describe_backend(_FakeTunable())["device"]
        self.assertEqual(device["name"], "fake-gpu")
        self.assertEqual(device["total_memory_bytes"], 1 << 30)

    def test_duplicate_backend_names_are_rejected(self):
        with self.assertRaises(AgentQueryError):
            describe_backends([ReferenceBackend(), ReferenceBackend()])

    def test_unwraps_instrumented_backend(self):
        wrapped = TelemetryRecorder().wrap(ReferenceExecutor())
        self.assertEqual(describe_backend(wrapped)["name"], "reference")


class TestPerformance(unittest.TestCase):
    def test_summary_of_no_runs(self):
        self.assertEqual(performance_summary(TelemetryRecorder())["runs"], 0)

    def test_summary_after_a_run(self):
        g = Graph("one")
        x = g.input("x", f32[2, 2])
        g.output(g.relu(x), "y")
        recorder = TelemetryRecorder()
        recorder.wrap(ReferenceExecutor()).execute(
            g, {"x": [[1.0, -1.0], [2.0, -2.0]]}
        )
        summary = performance_summary(recorder)
        self.assertEqual(summary["runs"], 1)
        self.assertEqual(summary["operators"][0]["op"], "relu")
        json.dumps(summary)

    def test_wrong_type_is_rejected(self):
        with self.assertRaises(AgentQueryError):
            performance_summary(object())


class TestRecommend(unittest.TestCase):
    def test_default_when_store_misses(self):
        result = recommend_config(
            ConfigStore(), _FakeTunable(), "add", {"dtype": "f32", "inputs": [[4, 8]]}
        )
        self.assertEqual(result["source"], "default")
        self.assertEqual(result["config"], {"block": 64})
        self.assertIsNone(result["median_ms"])
        self.assertIsNone(result["best_ms"])
        self.assertIsNone(result["p95_ms"])

    def test_store_hit(self):
        store = ConfigStore()
        record = TuningRecord(
            backend="fake",
            device="fake-gpu",
            op="add",
            signature={"dtype": "f32", "inputs": [[4, 8]]},
            config={"block": 128},
            median_ms=0.5,
            best_ms=0.45,
            p95_ms=0.55,
            samples=2,
        )
        store.put(record)
        result = recommend_config(
            store, _FakeTunable(), "add", {"inputs": [[4, 8]], "dtype": "f32"}
        )
        self.assertEqual(result["source"], "store")
        self.assertEqual(result["config"], {"block": 128})
        # 中位数与最佳可达分开报：前者是典型延迟，后者不是
        self.assertEqual(result["median_ms"], 0.5)
        self.assertEqual(result["best_ms"], 0.45)
        self.assertEqual(result["p95_ms"], 0.55)
        self.assertEqual(result["samples"], 2)

    def test_no_tuning_and_no_record(self):
        with self.assertRaises(AgentQueryError):
            recommend_config(ConfigStore(), ReferenceBackend(), "relu", {"dtype": "f32"})

    def test_bad_signature_is_rejected(self):
        with self.assertRaises(AgentQueryError):
            recommend_config(ConfigStore(), _FakeTunable(), "add", {})

    def test_wrong_store_type_is_rejected(self):
        with self.assertRaises(AgentQueryError):
            recommend_config(object(), _FakeTunable(), "add", {"dtype": "f32"})


class TestSnapshot(unittest.TestCase):
    def test_snapshot_is_json_serializable(self):
        store = ConfigStore()
        store.put(
            TuningRecord(
                backend="fake",
                device="fake-gpu",
                op="add",
                signature={"dtype": "f32"},
                config={"block": 128},
                median_ms=0.5,
            )
        )
        payload = snapshot([ReferenceBackend(), _FakeTunable()], TelemetryRecorder(), store)
        restored = json.loads(json.dumps(payload))
        self.assertEqual(set(restored["backends"]), {"reference", "fake"})
        self.assertEqual(restored["performance"]["runs"], 0)
        self.assertEqual(restored["config_store"]["records"], 1)
        entry = restored["config_store"]["entries"][0]
        self.assertIsNone(entry["best_ms"])
        self.assertIsNone(entry["p95_ms"])

    def test_only_read_only_entry_points_exist(self):
        """边界：这一层没有执行入口，也没有接收代码的入口。"""
        for name in ("execute", "run", "eval", "compile", "apply_config"):
            self.assertFalse(hasattr(agent_api, name), name)
        self.assertTrue(issubclass(AgentQueryError, ValueError))


if __name__ == "__main__":
    unittest.main()
