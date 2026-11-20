"""基准框架自身的测试：不需要 GPU。

性能数字在这里不重要，重要的是**判定逻辑**不能永远返回通过——
与 conformance 用"坏后端"验证套件是同一个思路。
"""

import json
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from aetherflow import ReferenceBackend
from benchmarks import cudnn_compare
from benchmarks.cases import CASES, build_case
from benchmarks.run import (
    _main,
    _sync,
    backend_names,
    check_regression,
    machine_note,
    measure,
    open_backend,
    prepare_case,
    recheck_slow,
    run_benchmark,
)


class TestBenchmarkCases(unittest.TestCase):
    def test_all_cases_run_on_reference_backend(self):
        """每个固定用例都必须能在参考后端跑出非空结果。"""
        backend = ReferenceBackend()
        for case in CASES:
            with self.subTest(case=case.name):
                graph, feed = build_case(case)
                out = backend.execute(graph, feed)
                self.assertTrue(out)
                for value in out.values():
                    self.assertGreater(value.size, 0)

    def test_fixed_seed_is_reproducible(self):
        """同一 seed 必须构建出同一份输入，否则两端测的不是同一件事。"""
        graph_a, feed_a = build_case(CASES[0], seed=7)
        graph_b, feed_b = build_case(CASES[0], seed=7)
        self.assertEqual(list(feed_a), list(feed_b))
        for name in feed_a:
            self.assertTrue((feed_a[name] == feed_b[name]).all())

    def test_resident_cases_split_weights(self):
        """常驻用例必须拆开权重与激活；没有驻留能力的后端仍拿完整输入。"""
        case = next(c for c in CASES if c.name == "transformer_block_resident")
        _, feed = build_case(case)
        params, activations = case.split(feed)
        self.assertEqual(set(activations), {"x"})
        self.assertEqual(set(params) | {"x"}, set(feed))

        class _FakeResident:
            def __init__(self):
                self.calls = []

            def release_parameters(self):
                self.calls.append("release")

            def resident_parameters(self, graph, values):
                self.calls.append(set(values))

        fake = _FakeResident()
        _, prepared = prepare_case(fake, case)
        self.assertEqual(set(prepared), {"x"})
        self.assertEqual(fake.calls, ["release", set(params)])

        # 参考后端：原样返回完整 feed，否则会"缺少图输入"
        _, prepared = prepare_case(ReferenceBackend(), case)
        self.assertEqual(set(prepared), set(feed))

    def test_resident_case_runs_without_resident_backend(self):
        report = run_benchmark(
            ReferenceBackend(),
            cases=[c for c in CASES if c.name == "cnn_lenet_resident"],
            rounds=1,
            warmup_seconds=0.0,
        )
        self.assertIn("cnn_lenet_resident", report["cases"])


class TestMeasureAndCheck(unittest.TestCase):
    def _tiny_report(self, median_ms, backend="ref", name="toy"):
        return {
            "backend": backend,
            "device": "cpu",
            "cases": {name: {"median_ms": median_ms, "p90_ms": median_ms, "min_ms": median_ms}},
        }

    def test_measure_returns_ordered_statistics(self):
        backend = ReferenceBackend()
        graph, feed = build_case(CASES[0])
        stats = measure(backend, graph, feed, warmup=1, repeat=5)
        self.assertLessEqual(stats["min_ms"], stats["median_ms"])
        self.assertLessEqual(stats["median_ms"], stats["p90_ms"])

    def test_report_survives_json_roundtrip(self):
        report = run_benchmark(ReferenceBackend(), cases=CASES[:2])
        restored = json.loads(json.dumps(report))
        self.assertEqual(restored["backend"], report["backend"])
        self.assertEqual(set(restored["cases"]), set(report["cases"]))

    def test_no_regression_within_tolerance(self):
        base = self._tiny_report(10.0)
        cur = self._tiny_report(11.0)
        self.assertEqual(check_regression(base, cur, 0.25), [])

    def test_slowdown_is_flagged(self):
        base = self._tiny_report(10.0)
        cur = self._tiny_report(13.0)
        problems = check_regression(base, cur, 0.25)
        self.assertEqual([p["kind"] for p in problems], ["slower"])
        self.assertAlmostEqual(problems[0]["slowdown"], 0.3)

    def test_missing_case_is_flagged(self):
        base = self._tiny_report(10.0)
        cur = {"backend": "ref", "device": "cpu", "cases": {}}
        self.assertEqual(
            [p["kind"] for p in check_regression(base, cur)], ["missing"]
        )

    def test_new_case_is_not_a_regression(self):
        base = {"backend": "ref", "device": "cpu", "cases": {}}
        cur = self._tiny_report(10.0)
        self.assertEqual(check_regression(base, cur), [])

    def test_backend_mismatch_is_flagged(self):
        base = self._tiny_report(10.0, backend="cuda")
        cur = self._tiny_report(10.0, backend="ref")
        kinds = [p["kind"] for p in check_regression(base, cur)]
        self.assertIn("backend_mismatch", kinds)

    def test_throughput_mode_mismatch_is_flagged(self):
        """回传与不回传的两种口径差一倍以上，混比必须直接判不可比。"""
        base = {**self._tiny_report(10.0), "device_outputs": False}
        cur = {**self._tiny_report(10.0), "device_outputs": True}
        kinds = [p["kind"] for p in check_regression(base, cur)]
        self.assertIn("throughput_mode_mismatch", kinds)

    def test_device_outputs_rejected_for_reference_backend(self):
        """参考后端没有设备内存，--device-outputs 必须在开跑前就拒绝。"""
        with redirect_stdout(StringIO()):
            with self.assertRaises(SystemExit):
                _main(["--backend", "ref", "--device-outputs", "--rounds", "1"])

    def test_machine_note_silent_on_same_machine(self):
        base = self._tiny_report(10.0)
        base["machine"] = {"cpu": "Xeon", "cores": 8}
        cur = self._tiny_report(10.0)
        cur["machine"] = {"cpu": "Xeon", "cores": 8}
        self.assertEqual(machine_note(base, cur), "")

    def test_machine_note_flags_different_host_cpu(self):
        """换一台宿主 CPU 更弱的实例，几乎只有指纹能说清"为什么全体变慢"。"""
        base = self._tiny_report(10.0)
        base["machine"] = {"cpu": "Xeon Platinum 8255C", "cores": 8}
        cur = self._tiny_report(26.0)
        cur["machine"] = {"cpu": "Xeon E5-2678 v3", "cores": 8}
        note = machine_note(base, cur)
        self.assertIn("不是同一台机器", note)
        self.assertIn("E5-2678", note)

    def test_machine_note_silent_without_fingerprint(self):
        """5.89 之前的基线没有指纹：无法判断，就不要编一个结论。"""
        base = self._tiny_report(10.0)
        cur = self._tiny_report(26.0)
        cur["machine"] = {"cpu": "Xeon", "cores": 8}
        self.assertEqual(machine_note(base, cur), "")

    def test_multiple_rounds_keep_the_fastest(self):
        """多轮取最好：第一轮被"整机变吵"拖慢，不能污染结果。"""

        class SlowFirstRound:
            name = "fake"

            def __init__(self, slow_calls):
                self.slow_calls = slow_calls
                self.calls = 0

            def execute(self, graph, feed):
                self.calls += 1
                if self.calls <= self.slow_calls:
                    time.sleep(0.005)
                return {"y": np.zeros(1, np.float32)}

        case = CASES[0].__class__(
            name="toy", build=CASES[0].build, warmup=1, repeat=3, note=""
        )
        per_round = case.warmup + case.repeat

        one = run_benchmark(SlowFirstRound(per_round), cases=[case], rounds=1)
        drifty = run_benchmark(SlowFirstRound(per_round), cases=[case], rounds=2)
        self.assertLess(
            drifty["cases"]["toy"]["median_ms"], one["cases"]["toy"]["median_ms"]
        )
        self.assertEqual(drifty["cases"]["toy"]["rounds"], 2)

    def test_slow_case_is_rechecked_before_failing(self):
        """瞬时干扰：首轮慢、重测正常 → 不算回归；两次都慢 → 才算。"""

        class Drift:
            name = "fake"

            def __init__(self, slow_calls):
                self.slow_calls = slow_calls
                self.calls = 0

            def execute(self, graph, feed):
                self.calls += 1
                if self.calls <= self.slow_calls:
                    time.sleep(0.005)
                return {"y": np.zeros(1, np.float32)}

        case = CASES[0].__class__(
            name="toy", build=CASES[0].build, warmup=1, repeat=3, note=""
        )
        per_round = case.warmup + case.repeat

        # 首轮慢、重测快：基线取"正常速度的四分之一"，慢轮必然超阈值
        backend = Drift(per_round)
        report = run_benchmark(backend, cases=[case], rounds=1)
        baseline = {
            "backend": "fake",
            "device": "cpu",
            "cases": {"toy": {"median_ms": report["cases"]["toy"]["median_ms"] / 4}},
        }
        problems, retried = recheck_slow(backend, baseline, report, cases=[case])
        self.assertEqual(retried, 1)
        self.assertEqual(problems, [])

        # 一直慢：重测也救不回来
        backend = Drift(10**6)
        report = run_benchmark(backend, cases=[case], rounds=1)
        baseline = {
            "backend": "fake",
            "device": "cpu",
            "cases": {"toy": {"median_ms": report["cases"]["toy"]["median_ms"] / 4}},
        }
        problems, retried = recheck_slow(backend, baseline, report, cases=[case])
        self.assertEqual(retried, 1)
        self.assertEqual([p["kind"] for p in problems], ["slower"])


class TestSync(unittest.TestCase):
    """计时区间里的同步必须对两种容器都成立。

    cupy 与 HIP 的替身模块长得像但不同：前者有 ``cp.cuda`` 假命名空间，
    后者没有（见 ``hip/runtime.py::current_stream``）。只认 ``cp.cuda``
    会让 HIP 上的所有基准在第一次 ``measure`` 就抛 AttributeError。
    """

    def test_cupy_style_backend_uses_current_stream(self):
        calls = []

        class _Stream:
            def synchronize(self):
                calls.append("stream")

        class _Cuda:
            @staticmethod
            def get_current_stream():
                return _Stream()

        class _Cp:
            cuda = _Cuda()

            def synchronize(self):
                calls.append("module")

        _sync(SimpleNamespace(cp=_Cp()))
        self.assertEqual(calls, ["stream"])

    def test_hip_style_backend_falls_back_to_module_sync(self):
        calls = []

        class _Cp:
            def synchronize(self):
                calls.append("module")

        _sync(SimpleNamespace(cp=_Cp()))
        self.assertEqual(calls, ["module"])

    def test_backend_without_cp_is_noop(self):
        _sync(SimpleNamespace())


class TestBackendSelection(unittest.TestCase):
    """三个基准 CLI 共用一份后端工厂；``--backend`` 的选项也由它给出。

    重点不是"能不能造出 CUDA 后端"，而是两条**失败语义**：明确指定一个
    装不起来的后端时必须失败，而不是悄悄退回 CPU 产出一组假性能数字。
    """

    def test_names_cover_all_builtins_without_duplicates(self):
        names = backend_names()
        for expected in ("ref", "cuda", "hip"):
            self.assertIn(expected, names)
        # ``ref`` 是 ``reference`` 的短名，不能两个都出现在选项里
        self.assertNotIn("reference", names)
        self.assertEqual(len(names), len(set(names)))

    def test_ref_is_a_cli_alias_only(self):
        """``ref`` 只是命令行短名；误传给工厂必须报错，不能当成真名去猜。"""
        with self.assertRaises(LookupError):
            open_backend("ref")

    def test_hip_is_either_built_or_refused(self):
        """有 ROCm 就返回 HIP，没有就明确拒绝——绝不允许退回参考后端。"""
        try:
            backend = open_backend("hip")
        except SystemExit as exc:
            self.assertIn("hip", str(exc))
        else:
            self.assertEqual(backend.name, "hip")

    def test_unknown_backend_is_a_lookup_error(self):
        with self.assertRaises(LookupError):
            open_backend("no_such_backend")


class TestCuDNNCompare(unittest.TestCase):
    """cuDNN 对照是可选能力：没装库时必须明确跳过，而不是崩在调用上。"""

    def test_missing_cudnn_is_a_skip_not_a_failure(self):
        buf = StringIO()
        with patch.object(
            cudnn_compare, "_load_cudnn", side_effect=OSError("没有 libcudnn.so.9")
        ):
            with redirect_stdout(buf):
                code = cudnn_compare._main([])
        self.assertEqual(code, 0)
        self.assertIn("跳过", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
