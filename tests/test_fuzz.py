"""IR 差分模糊测试的本地用例。

这里不重复测具体算子的数学（那些有各自的专项用例），只测模糊测试自己的
三条约定：

1. 同一种子必须得到同一张图——否则失败无法复现；
2. 生成的图必须合法，且优化前后逐位一致；
3. 非法属性必须在构图期被拒绝，而不是被接受或崩溃。
"""

import unittest
from unittest import mock

import numpy as np

from aetherflow import (
    FuzzFailure,
    FuzzReport,
    Graph,
    ReferenceExecutor,
    f32,
    random_case,
    run_fuzz,
)
from aetherflow import fuzz
from aetherflow.fuzz import _mutate_rejected
from aetherflow.ir.serialize import dumps, loads, to_dict
from aetherflow.ir.verifier import ERROR, verify


class TestRandomCase(unittest.TestCase):
    def test_same_seed_is_reproducible(self):
        left = random_case(11)
        right = random_case(11)
        self.assertEqual(to_dict(left.graph), to_dict(right.graph))
        self.assertEqual(set(left.feed), set(right.feed))
        for name in left.feed:
            np.testing.assert_array_equal(left.feed[name], right.feed[name])

    def test_different_seeds_produce_different_graphs(self):
        self.assertNotEqual(dumps(random_case(1).graph), dumps(random_case(2).graph))

    def test_generated_graphs_are_valid_and_serializable(self):
        for seed in range(12):
            case = random_case(seed)
            errors = [item for item in verify(case.graph) if item.level == ERROR]
            self.assertEqual(errors, [], f"seed={seed}: {errors}")
            self.assertEqual(to_dict(loads(dumps(case.graph))), to_dict(case.graph))

    def test_generated_graphs_are_static_and_have_outputs(self):
        for seed in range(6):
            graph = random_case(seed).graph
            self.assertTrue(graph.outputs, f"seed={seed} 没有输出")
            self.assertTrue(all(value.type.is_static for value in graph.values))


class TestMutation(unittest.TestCase):
    def test_illegal_attributes_are_rejected(self):
        verdicts = [_mutate_rejected(random_case(seed)) for seed in range(24)]
        self.assertNotIn(False, verdicts, "存在被接受的非法属性")
        self.assertIn(True, verdicts, "24 个种子里没有一个可用变异")


class TestRunFuzz(unittest.TestCase):
    def test_short_run_is_clean(self):
        report = run_fuzz(20)
        self.assertTrue(report.ok, report.text())
        self.assertEqual(report.checked, 20)
        self.assertGreater(report.mutated, 0)

    def test_coverage_reaches_the_shape_and_value_family(self):
        report = run_fuzz(40)
        for name in ("add", "relu", "matmul", "softmax", "reshape", "transpose", "concat"):
            self.assertIn(name, report.ops)
        self.assertGreaterEqual(len(report.ops), 15)

    def test_coverage_reaches_the_geometry_family(self):
        report = run_fuzz(40)
        for name in ("conv2d", "maxpool2d", "avgpool2d", "attention", "scan_linear", "embedding"):
            self.assertIn(name, report.ops)
        self.assertGreaterEqual(len(report.ops), 30)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            run_fuzz(0)
        with self.assertRaises(ValueError):
            run_fuzz(1, start=-1)

    def test_report_text(self):
        good = FuzzReport(checked=1, mutated=1, skipped=0, failures=[], ops=frozenset({"add"}))
        self.assertIn("全部通过", good.text())
        bad = FuzzReport(
            checked=1,
            mutated=0,
            skipped=0,
            failures=[FuzzFailure(seed=3, stage="verify", detail="boom")],
            ops=frozenset(),
        )
        self.assertFalse(bad.ok)
        self.assertIn("1 条失败", bad.text())
        self.assertIn("seed=3", bad.text())


class TestSemanticMutations(unittest.TestCase):
    """合法属性名 + 非法值：必须在构图期被拒，而不是等到执行。"""

    def test_semantic_violations_are_rejected(self):
        report = run_fuzz(15)
        self.assertTrue(report.ok, report.text())
        self.assertGreaterEqual(report.semantic, 30)
        self.assertLessEqual(report.semantic_skipped, report.checked)

    def test_text_reports_semantic_count(self):
        report = FuzzReport(
            checked=1,
            mutated=1,
            skipped=0,
            failures=[],
            ops=frozenset({"add"}),
            semantic=3,
        )
        self.assertIn("语义非法节点拒绝 3 例", report.text())


class TestRuntimeSemantics(unittest.TestCase):
    """图合法、数据非法：越界下标与除零只能在执行期判，且必须显式报错。"""

    def test_every_runtime_case_behaves_as_declared(self):
        failures = []
        passed = fuzz._check_runtime(failures, 0)
        self.assertEqual(failures, [])
        self.assertEqual(passed, len(fuzz._runtime_cases()))

    def test_short_run_includes_runtime_checks(self):
        report = run_fuzz(5)
        self.assertTrue(report.ok, report.text())
        self.assertEqual(report.runtime, len(fuzz._runtime_cases()))
        self.assertEqual(report.runtime, 8)
        self.assertIn("运行期语义错误拒绝 8 例", report.text())

    def test_silently_accepted_invalid_data_is_a_failure(self):
        """把期望反过来：声明"应当报错"的用例被静默接受时必须记成失败。"""

        def run_ok():
            g = Graph("runtime_ok")
            x = g.input("x", f32[2])
            g.output(g.add_node("neg", [x]), "y")
            return ReferenceExecutor().run(g, {"x": np.ones(2, np.float32)})["y"]

        with mock.patch.object(fuzz, "_runtime_cases", return_value=(("假用例", run_ok, True),)):
            failures = []
            self.assertEqual(fuzz._check_runtime(failures, 0), 0)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].stage, "runtime")
        self.assertIn("被静默接受", failures[0].detail)

    def test_unexpected_exception_is_a_failure(self):
        """``IndexError`` 之类不是约定的出口：它说明错误没被翻译成 ``ValueError``。"""

        def run_boom():
            raise IndexError("下标越界")

        with mock.patch.object(fuzz, "_runtime_cases", return_value=(("炸", run_boom, True),)):
            failures = []
            self.assertEqual(fuzz._check_runtime(failures, 0), 0)
        self.assertEqual(len(failures), 1)
        self.assertIn("IndexError", failures[0].detail)


class TestBackendAgreement(unittest.TestCase):
    """随机图也要跨后端对拍：固定图集只能覆盖作者想到的组合。"""

    def _available(self, name: str) -> bool:
        from aetherflow.backends import BackendUnavailableError, get_backend

        try:
            get_backend(name)
        except (BackendUnavailableError, LookupError):
            return False
        return True

    def _check(self, name: str):
        if not self._available(name):
            self.skipTest(f"{name} 后端在这台机器上不可用")
        report = run_fuzz(10, backends=(name,))
        self.assertTrue(report.ok, report.text())
        self.assertIn(name, report.backends)
        self.assertGreater(report.compared[name], 0, report.text())

    def test_hip_matches_reference_on_random_graphs(self):
        self._check("hip")

    def test_cuda_matches_reference_on_random_graphs(self):
        self._check("cuda")

    def test_unknown_backend_is_a_failure_not_a_silent_skip(self):
        report = run_fuzz(1, backends=("nope",))
        self.assertFalse(report.ok)
        self.assertTrue(any("nope" in item.detail for item in report.failures))

    def test_report_text_shows_backend_counts(self):
        report = FuzzReport(
            checked=2,
            mutated=1,
            skipped=0,
            failures=[],
            ops=frozenset({"add"}),
            backends=("hip",),
            compared={"hip": 3},
            refused={"hip": 2},
        )
        text = report.text()
        self.assertIn("hip 对拍 3 例", text)
        self.assertIn("拒绝 2 例", text)


if __name__ == "__main__":
    unittest.main()
