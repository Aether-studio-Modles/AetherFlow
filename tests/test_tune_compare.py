"""跨后端调优对照的本地测试：报告对齐、一致率、坏输入。

采集本身要真机，但"两份报告怎么对齐、缺了一边算不算不一致、平均提升怎么算"
是纯逻辑，必须在没有 GPU 的机器上钉住。
"""

import contextlib
import io
import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace

import numpy as np

from aetherflow import Graph, f32
from aetherflow.tuning import TuningSpace
from benchmarks.tune_compare import (
    REPORT_FORMAT,
    _fmt_config,
    _print_comparison,
    collect,
    compare_reports,
    load_report,
    summarize,
)
from benchmarks.tune import TuneCase


def _report(backend, cases):
    return {"format": REPORT_FORMAT, "backend": backend, "device": f"{backend} 卡", "cases": cases}


class TestCompareReports(unittest.TestCase):
    def test_alignment_and_agreement(self):
        left = _report(
            "cuda",
            {
                "same": {"best": {"block": 256}, "speedup": 1.1},
                "diff": {"best": {"block": 128}, "speedup": 1.0},
                "only_left": {"best": {"block": 512}, "speedup": 1.2},
            },
        )
        right = _report(
            "hip",
            {
                "same": {"best": {"block": 256}, "speedup": 1.0},
                "diff": {"best": {"block": 512}, "speedup": 1.3},
                "only_right": {"best": {"tile_w": None}, "speedup": 1.4},
            },
        )
        rows = compare_reports(left, right)
        self.assertEqual(
            [row.case for row in rows],
            ["same", "diff", "only_left", "only_right"],
        )
        self.assertEqual([row.same for row in rows], [True, False, None, None])

    def test_key_order_does_not_matter(self):
        left = _report("cuda", {"x": {"best": {"block": 256, "tile_w": None}, "speedup": 1.0}})
        right = _report("hip", {"x": {"best": {"tile_w": None, "block": 256}, "speedup": 1.0}})
        self.assertTrue(compare_reports(left, right)[0].same)

    def test_summary_skips_one_sided_and_missing_speedup(self):
        left = _report(
            "cuda",
            {
                "a": {"best": {"block": 256}, "speedup": 1.2},
                "b": {"best": {"block": 128}, "speedup": None},
                "c": {"best": {"block": 512}, "speedup": 1.0},
            },
        )
        right = _report(
            "hip",
            {
                "a": {"best": {"block": 256}, "speedup": 1.0},
                "b": {"best": {"block": 128}, "speedup": None},
                "d": {"best": {"block": 512}, "speedup": 1.4},
            },
        )
        stats = summarize(compare_reports(left, right))
        self.assertEqual(stats["total"], 4)
        self.assertEqual(stats["both"], 2)
        self.assertEqual(stats["agreed"], 2)
        # b 的默认配置没测出结果，不参与平均，也不把它当 1.0
        self.assertAlmostEqual(stats["left_mean_speedup"], 1.2)
        self.assertAlmostEqual(stats["right_mean_speedup"], 1.0)


class TestFormat(unittest.TestCase):
    def test_tile_auto_is_readable(self):
        self.assertEqual(_fmt_config({"tile_w": None}), "tile_w=auto")
        self.assertEqual(_fmt_config({"block": 256}), "block=256")


class TestLoadReport(unittest.TestCase):
    def _write(self, tmp, payload):
        path = os.path.join(tmp, "report.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(payload if isinstance(payload, str) else json.dumps(payload))
        return path

    def test_roundtrip(self):
        payload = _report("cuda", {"a": {"best": {"block": 256}, "speedup": 1.0}})
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_report(self._write(tmp, payload))
        self.assertEqual(loaded["backend"], "cuda")

    def test_bad_json_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_report(self._write(tmp, "{not json"))

    def test_unknown_format_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_report(self._write(tmp, {"cases": {}}))

    def test_missing_best_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_report(self._write(tmp, _report("cuda", {"a": {"speedup": 1.0}})))


class _FakeBackend:
    """只够 ``collect`` 用：声明一个两候选空间，按时长区分快慢。"""

    name = "fake"

    def __init__(self):
        self._tuning = {}

    def tuning_space(self, op):
        return TuningSpace.build({"block": (256, 128)})

    def set_tuning(self, op, config):
        self._tuning[op] = dict(config)

    def clear_tuning(self, op=None):
        self._tuning.clear()

    def device_info(self):
        return SimpleNamespace(name="Fake GPU")

    def execute(self, graph, feed):
        # 128 的忙等 2ms 远大于计时的分辨率：两候选的差异稳定超出噪声，
        # 报告里的"已判定"才可断言（空循环在快机器上可能只有几十微秒）。
        block = self._tuning.get("relu", {}).get("block", 256)
        if block == 128:
            deadline = time.perf_counter() + 0.002
            while time.perf_counter() < deadline:
                pass
        return {"y": np.zeros((4, 8), np.float32)}


def _case():
    def build(rng):
        g = Graph("relu")
        x = g.input("x", f32[(4, 8)])
        g.output(g.add_node("relu", [x]), "y")
        return g, {"x": np.zeros((4, 8), np.float32)}

    return TuneCase(
        "relu_small",
        "relu",
        build,
        {"dtype": "f32", "inputs": [[4, 8]]},
        warmup=1,
        repeat=2,
    )


class TestCollect(unittest.TestCase):
    def test_report_has_default_and_winner(self):
        backend = _FakeBackend()
        report = collect(backend, [_case()], rounds=2)
        self.assertEqual(report["format"], REPORT_FORMAT)
        self.assertEqual(report["backend"], "fake")
        self.assertEqual(report["device"], "Fake GPU")
        self.assertEqual(report["method"], "paired")
        entry = report["cases"]["relu_small"]
        self.assertEqual(entry["default"], {"block": 256})
        self.assertEqual(entry["best"], {"block": 256})
        self.assertEqual(entry["method"], "paired")
        # 128 明显更慢：配对区间整体在 0 以上，但赢家仍是默认，decided=False
        self.assertFalse(entry["decided"])
        slow = [pair for pair in entry["pairs"] if pair["config"] == {"block": 128}][0]
        self.assertEqual(slow["verdict"], "slower")
        self.assertGreater(slow["ci_low_pct"], 1.0)
        self.assertEqual(entry["signature"], {"dtype": "f32", "inputs": [[4, 8]]})
        self.assertIsNotNone(entry["speedup"])
        json.dumps(report)  # 报告必须能落盘

    def test_report_records_round_span_and_threshold(self):
        backend = _FakeBackend()
        report = collect(backend, [_case()], rounds=2, max_round_seconds=60.0)
        self.assertEqual(report["max_round_seconds"], 60.0)
        entry = report["cases"]["relu_small"]
        self.assertEqual(entry["max_round_seconds"], 60.0)
        self.assertEqual(len(entry["round_seconds"]), 2)
        self.assertTrue(all(seconds > 0.0 for seconds in entry["round_seconds"]))
        # 60 秒远大于实际轮内跨度：判定不受影响
        slow = [pair for pair in entry["pairs"] if pair["config"] == {"block": 128}][0]
        self.assertEqual(slow["verdict"], "slower")

    def test_report_records_relative_budget(self):
        backend = _FakeBackend()
        report = collect(
            backend, [_case()], rounds=2, warmup_rounds=1, max_round_factor=1000.0
        )
        self.assertEqual(report["max_round_factor"], 1000.0)
        entry = report["cases"]["relu_small"]
        self.assertEqual(entry["max_round_factor"], 1000.0)
        self.assertIsNotNone(entry["round_budget_seconds"])
        self.assertGreater(entry["warmup_seconds"], 0.0)

    def test_long_round_rejects_conclusions(self):
        backend = _FakeBackend()
        report = collect(backend, [_case()], rounds=2, max_round_seconds=0.0001)
        entry = report["cases"]["relu_small"]
        self.assertFalse(entry["decided"])
        self.assertEqual(entry["best"], {"block": 256})
        slow = [pair for pair in entry["pairs"] if pair["config"] == {"block": 128}][0]
        self.assertEqual(slow["verdict"], "long-round")

    def test_report_records_warmup(self):
        backend = _FakeBackend()
        report = collect(backend, [_case()], rounds=2, warmup_rounds=1)
        self.assertEqual(report["warmup_rounds"], 1)
        entry = report["cases"]["relu_small"]
        self.assertEqual(entry["warmup_rounds"], 1)
        # 预热轮测了 128 的忙等分支：耗时必须为正，且不进入 round_seconds
        self.assertGreater(entry["warmup_seconds"], 0.0)
        self.assertEqual(len(entry["round_seconds"]), 2)

    def test_report_records_staged_stages(self):
        backend = _FakeBackend()
        report = collect(backend, [_case()], rounds=2, staged=True)
        self.assertTrue(report["staged"])
        entry = report["cases"]["relu_small"]
        # 单键空间只有一段；分两段的情况由 test_tuning 的 TestTuneStaged 钉住
        self.assertEqual(len(entry["stages"]), 1)
        self.assertEqual(entry["stages"][0]["keys"], ["block"])

    def test_report_records_replicates(self):
        backend = _FakeBackend()
        report = collect(backend, [_case()], rounds=2, repeats=2)
        self.assertEqual(report["repeats"], 2)
        entry = report["cases"]["relu_small"]
        self.assertEqual(entry["repeats"], 2)
        self.assertEqual(len(entry["replicates"]), 2)
        slow = [pair for pair in entry["pairs"] if pair["config"] == {"block": 128}][0]
        # 两次同向，保留 slower（不是 unreplicated）
        self.assertEqual(slow["verdict"], "slower")

    def test_report_records_vote(self):
        backend = _FakeBackend()
        report = collect(backend, [_case()], rounds=2, repeats=3, vote="majority")
        self.assertEqual(report["vote"], "majority")
        entry = report["cases"]["relu_small"]
        self.assertEqual(entry["vote"], "majority")
        slow = [pair for pair in entry["pairs"] if pair["config"] == {"block": 128}][0]
        # 三次都更慢：多数票与全票给出同一个判定，但票数要落盘
        self.assertEqual(slow["verdict"], "slower")
        self.assertEqual(slow["votes"], {"slower": 3})
        # 逐次配对差与票数一一对应：三次都正（更慢），长度等于采集次数
        self.assertEqual(len(slow["replicate_diffs"]), 3)
        self.assertTrue(all(value > 0.0 for value in slow["replicate_diffs"]))

    def test_single_collection_has_no_replicate_diffs(self):
        backend = _FakeBackend()
        report = collect(backend, [_case()], rounds=2)
        slow = [pair for pair in report["cases"]["relu_small"]["pairs"]
                if pair["config"] == {"block": 128}][0]
        self.assertEqual(slow["replicate_diffs"], [])

    def test_single_collection_has_no_vote(self):
        backend = _FakeBackend()
        report = collect(backend, [_case()], rounds=2)
        self.assertIsNone(report["vote"])
        entry = report["cases"]["relu_small"]
        self.assertIsNone(entry["vote"])
        slow = [pair for pair in entry["pairs"] if pair["config"] == {"block": 128}][0]
        self.assertEqual(slow["votes"], {})

    def test_vote_conflicts_are_rejected(self):
        backend = _FakeBackend()
        with self.assertRaises(ValueError):
            collect(backend, [_case()], rounds=2, vote="majority")
        with self.assertRaises(ValueError):
            collect(backend, [_case()], rounds=2, paired=False, repeats=2, vote="majority")
        with self.assertRaises(ValueError):
            collect(backend, [_case()], rounds=2, staged=True, repeats=2, vote="majority")
        with self.assertRaises(ValueError):
            collect(backend, [_case()], rounds=2, repeats=2, vote="best-of")

    def test_serial_mode_is_marked_and_has_no_interval(self):
        backend = _FakeBackend()
        report = collect(backend, [_case()], rounds=1, paired=False)
        self.assertEqual(report["method"], "serial")
        entry = report["cases"]["relu_small"]
        self.assertEqual(entry["method"], "serial")
        self.assertNotIn("decided", entry)
        self.assertNotIn("ci_low_pct", entry)


class TestPrintComparison(unittest.TestCase):
    def test_undecided_winner_gets_a_star(self):
        left = _report(
            "cuda",
            {"relu": {"best": {"block": 256}, "speedup": 1.09,
                      "method": "paired", "decided": False}},
        )
        right = _report(
            "hip",
            {"relu": {"best": {"block": 256}, "speedup": 1.2,
                      "method": "paired", "decided": True}},
        )
        rows = compare_reports(left, right)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            _print_comparison(left, right, rows)
        text = buffer.getvalue()
        self.assertIn("1.09x*", text)
        self.assertIn("1.20x", text)
        self.assertIn("未被判定", text)


if __name__ == "__main__":
    unittest.main()
