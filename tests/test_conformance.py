"""一致性套件的自测。

这一组测试要证明套件本身是有效的——用"故意做错的后端"验证它确实能抓住问题。
只验证"参考后端通过"是不够的：一个永远返回 PASS 的套件也能通过那条断言。

三类假后端分别对应三种失控方式：
- 少声明（能力协商被信任，但套件要确认它确实拒绝）
- 说谎（声明支持却算错）
- 绕道（用别的异常代替 BackendCapabilityError、或结果不确定）
"""

import unittest

import numpy as np

from aetherflow import (
    BackendCapabilityError,
    ReferenceBackend,
    default_cases,
    run_conformance,
    unsupported_ops,
)
from aetherflow.backends.cuda import CUDA_AVAILABLE, CudaBackend
from aetherflow.ir.ops import OPS


class _Base:
    """假后端的公共部分：统一用参考执行器算，只改变"是否被允许算"。"""

    name = "fake"

    def __init__(self):
        self._ref = ReferenceBackend()

    def capabilities(self):
        raise NotImplementedError

    def execute(self, graph, inputs):
        raise NotImplementedError

    def _guard(self, graph):
        missing = unsupported_ops(graph, self.capabilities())
        if missing:
            raise BackendCapabilityError(f"不支持 {missing}")


class SubsetBackend(_Base):
    """只声明一部分算子，其余必须显式拒绝。"""

    name = "subset"

    def __init__(self, ops=("add", "relu")):
        super().__init__()
        self._ops = frozenset(ops)

    def capabilities(self):
        return self._ops

    def execute(self, graph, inputs):
        self._guard(graph)
        return self._ref.execute(graph, inputs)


class ZeroBackend(_Base):
    """声明全部、输出全零：数值对不上。"""

    name = "zero"

    def capabilities(self):
        return frozenset(OPS)

    def execute(self, graph, inputs):
        self._guard(graph)
        return {k: np.zeros_like(v) for k, v in self._ref.execute(graph, inputs).items()}


class CrashBackend(_Base):
    """声明全部，但用 RuntimeError 代替 BackendCapabilityError。"""

    name = "crash"

    def capabilities(self):
        return frozenset(OPS)

    def execute(self, graph, inputs):
        raise RuntimeError("内部实现没准备好")


class NoisyBackend(_Base):
    """数值在容差内但每次不同：确定性检查必须抓住它。"""

    name = "noisy"

    def capabilities(self):
        return frozenset({"relu"})

    def execute(self, graph, inputs):
        self._guard(graph)
        out = self._ref.execute(graph, inputs)
        rng = np.random.default_rng()
        # 1e-6 在 f32 上是可分辨的，同时远小于 atol=1e-4：数值检查会放过，确定性不能
        return {
            k: (v + rng.normal(0.0, 1e-6, v.shape)).astype(v.dtype)
            for k, v in out.items()
        }


class TestReferenceConformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_conformance(ReferenceBackend())

    def test_passes(self):
        self.assertTrue(self.report.ok, self.report.text())

    def test_covers_all_operators(self):
        self.assertEqual(set(self.report.covered_ops), set(OPS))

    def test_no_refusals(self):
        """参考后端承接全部 PORTABLE 算子，不该拒绝任何用例。"""
        self.assertEqual(self.report.refused, [])

    def test_report_is_readable(self):
        text = self.report.text()
        for section in ("backend", "checks", "cases", "result"):
            self.assertIn(section, text)
        self.assertIn("PASS", text)
        self.assertIn("coverage", text)

    def test_every_case_executed(self):
        self.assertEqual(len(self.report.executed), len(default_cases()))

    def test_determinism_can_be_disabled(self):
        report = run_conformance(ReferenceBackend(), check_determinism=False)
        self.assertTrue(report.ok, report.text())


class TestSuiteCatchesBadBackends(unittest.TestCase):
    def test_subset_backend_passes_and_refuses_rest(self):
        report = run_conformance(SubsetBackend())
        self.assertTrue(report.ok, report.text())
        self.assertEqual(set(report.covered_ops), {"add", "relu"})
        self.assertGreater(len(report.refused), 0)
        # 拒绝必须点名是哪个算子没声明
        self.assertTrue(any("未声明算子" in c.detail for c in report.refused))

    def test_wrong_numbers_fail(self):
        report = run_conformance(ZeroBackend(), check_determinism=False)
        self.assertFalse(report.ok)
        self.assertGreater(len(report.failed_cases), 0)
        self.assertIn("参考语义不一致", report.failed_cases[0].detail)

    def test_wrong_exception_type_fails(self):
        report = run_conformance(CrashBackend(), check_determinism=False)
        self.assertFalse(report.ok)
        explicit = next(c for c in report.checks if c.name == "explicit_failure")
        self.assertFalse(explicit.ok)
        self.assertIn("RuntimeError", explicit.detail)

    def test_nondeterminism_fails(self):
        report = run_conformance(NoisyBackend())
        self.assertFalse(report.ok)
        self.assertTrue(
            any("两次执行结果不同" in c.detail for c in report.failed_cases),
            report.text(),
        )

    def test_unknown_capability_fails(self):
        report = run_conformance(SubsetBackend(("relu", "conv3d")), check_determinism=False)
        self.assertFalse(report.ok)
        contract = next(c for c in report.checks if c.name == "contract")
        self.assertFalse(contract.ok)
        self.assertIn("conv3d", contract.detail)

    def test_declared_but_never_executed_fails(self):
        """声明了 softmax 却把所有用例都拒掉，覆盖度检查必须抓住。"""

        class RefuseAll(_Base):
            name = "refuse_all"

            def capabilities(self):
                return frozenset(OPS)

            def execute(self, graph, inputs):
                raise BackendCapabilityError("一律拒绝")

        report = run_conformance(RefuseAll(), check_determinism=False)
        self.assertFalse(report.ok)
        coverage = next(c for c in report.checks if c.name == "coverage")
        self.assertFalse(coverage.ok)


class TestCustomCases(unittest.TestCase):
    def test_subset_of_cases(self):
        """调用方可以只跑自己关心的用例（例如对手写的 JSON 样例）。"""
        case = next(c for c in default_cases() if c.name == "matmul_2d")
        report = run_conformance(ReferenceBackend(), cases=[case])
        self.assertTrue(report.ok, report.text())
        self.assertEqual(len(report.cases), 1)


@unittest.skipUnless(CUDA_AVAILABLE, "需要 cupy 与 CUDA 设备")
class TestCudaConformance(unittest.TestCase):
    """CUDA 后端必须自证：声明支持的都对齐，边界必须显式拒绝。"""

    @classmethod
    def setUpClass(cls):
        cls.report = run_conformance(CudaBackend())

    def test_passes(self):
        self.assertTrue(self.report.ok, self.report.text())

    def test_known_limits_are_explicit_refusals(self):
        refused = {c.name for c in self.report.refused}
        for name in (
            "f64_input_relu",
        ):
            self.assertIn(name, refused, self.report.text())

    def test_all_declared_operators_exercised(self):
        """声明的算子必须全部跑通；未实现的算子只能以拒绝出现。"""
        self.assertEqual(set(self.report.covered_ops), set(self.report.capabilities))
        self.assertIn("conv2d", self.report.capabilities)
        self.assertNotIn("conv2d_3x3_pad1", {c.name for c in self.report.refused})

    def test_numerics_align(self):
        for case in self.report.executed:
            self.assertLess(case.max_error, 1e-3, f"{case.name}: {case.max_error}")


if __name__ == "__main__":
    unittest.main()
