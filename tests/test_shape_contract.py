"""动态形状契约的验收测试。

判据分三层：

1. 契约表本身：每个环节都有明确状态与非空理由，未知环节不默认放行；
2. ``rejected`` 环节在入口显式拒绝：训练循环与分段重算都在构造期报错，
   错误里带环节名、具体输入与可操作建议；
3. ``full`` / ``per-op`` / ``data-only`` 的声明不是空话：同一张符号图能被
   执行器按不同 batch 跑、部署包只要求权重静态、reshape 的 ``(B,4) -> (4*B,)``
   在构图期被拒绝。
"""

import tempfile
import unittest

import numpy as np

from aetherflow import Graph, f32
from aetherflow.backends import ReferenceBackend
from aetherflow.deploy import BundleError, load_bundle, save_bundle
from aetherflow.frontend.torch_export import UnsupportedOperatorError, _shape_of
from aetherflow.ir import (
    DYNAMIC_SHAPE_SUPPORT,
    require_static_shapes,
    support_for,
    symbolic_inputs,
)
from aetherflow.ir.types import sym_mul
from aetherflow.train import ParameterStore, SGD, Trainer
from aetherflow.train.recompute import CheckpointPlan


def _symbolic_graph():
    """批维是符号的最简训练图：x[B,4] @ w[4,1] + b -> reduce_sum -> loss。"""
    g = Graph("sym")
    x = g.input("x", f32["B", 4])
    t = g.input("t", f32["B", 1])
    w = g.input("w", f32[4, 1])
    b = g.input("b", f32[1])
    diff = g.sub(g.add(g.matmul(x, w), b), t)
    loss = g.reduce_sum(diff, axis=0, keepdims=False)
    g.output(g.reduce_sum(loss, axis=0, keepdims=False), "loss")
    return g


def _static_graph():
    g = Graph("static")
    x = g.input("x", f32[4])
    g.output(g.reduce_sum(g.mul(x, x), axis=0, keepdims=False), "loss")
    return g


class TestContractTable(unittest.TestCase):
    def test_every_stage_has_status_and_reason(self):
        expected = {
            "graph",
            "execute",
            "autodiff",
            "trainer",
            "recompute",
            "deploy",
            "torch-import",
        }
        self.assertEqual(set(DYNAMIC_SHAPE_SUPPORT), expected)
        for stage, item in DYNAMIC_SHAPE_SUPPORT.items():
            with self.subTest(stage=stage):
                self.assertEqual(item.stage, stage)
                self.assertIn(item.dynamic, ("full", "per-op", "data-only", "rejected"))
                self.assertTrue(item.reason, f"{stage} 缺少理由")
                self.assertTrue(item.label, f"{stage} 缺少中文标签")

    def test_table_is_read_only(self):
        with self.assertRaises(TypeError):
            DYNAMIC_SHAPE_SUPPORT["execute"] = None

    def test_unknown_stage_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            support_for("serving")
        self.assertIn("未知的动态形状环节", str(ctx.exception))

    def test_static_guard_only_applies_to_rejected_stages(self):
        """把 full / per-op 环节交给守卫会误拦本来可用的图，必须直接拒绝。"""
        graph = _static_graph()
        for stage in ("execute", "autodiff"):
            with self.subTest(stage=stage):
                with self.assertRaises(ValueError) as ctx:
                    require_static_shapes(graph, stage=stage)
                self.assertIn("不应调用 require_static_shapes", str(ctx.exception))


class TestSymbolicInputs(unittest.TestCase):
    def test_only_symbolic_inputs_are_listed(self):
        g = Graph("mixed")
        g.input("static", f32[4])
        g.input("dynamic", f32["B", 4])
        g.input("expr", f32[sym_mul("B", 2)])
        self.assertEqual(symbolic_inputs(g), ["dynamic", "expr"])

    def test_static_graph_passes_the_guard(self):
        require_static_shapes(_static_graph(), stage="trainer")


class TestRejectedStages(unittest.TestCase):
    def test_trainer_rejects_symbolic_input_with_stage_name(self):
        g = _symbolic_graph()
        store = ParameterStore.from_graph(g, exclude=("x", "t"))
        with self.assertRaises(ValueError) as ctx:
            Trainer(g, store, SGD(store.arrays(), lr=0.1))
        message = str(ctx.exception)
        self.assertIn("训练循环环节", message)
        self.assertIn("符号维", message)
        self.assertIn("%x", message)
        self.assertIn("DataLoader(drop_last=True)", message)

    def test_checkpoint_plan_rejects_symbolic_graph(self):
        g = _symbolic_graph()
        with self.assertRaises(ValueError) as ctx:
            CheckpointPlan(g, wrt=[g.find("w")], segments=1)
        self.assertIn("分段重算环节", str(ctx.exception))

    def test_checkpoint_plan_still_accepts_static_graph(self):
        plan = CheckpointPlan(_static_graph(), wrt=[], segments=1)
        self.assertEqual(plan.graph.name, "static")


class TestDeclaredSupport(unittest.TestCase):
    def test_executor_binds_one_symbolic_graph_to_several_batches(self):
        """``execute`` 声明为 full：一次构图，多形状执行。"""
        g = Graph("bind")
        x = g.input("x", f32["B", 4])
        g.output(g.silu(x), "y")
        backend = ReferenceBackend()
        for batch in (1, 3, 7):
            out = backend.execute(g, {"x": np.ones((batch, 4), dtype=np.float32)})["y"]
            self.assertEqual(out.shape, (batch, 4))

    def test_deploy_requires_static_weights_but_allows_symbolic_data(self):
        """``deploy`` 声明为 data-only：权重必须静态，数据输入仍可绑定。"""
        g = Graph("deploy")
        x = g.input("x", f32["B", 4])
        w = g.input("w", f32[4])
        g.output(g.mul(x, w), "y")
        weights = {"w": np.arange(4, dtype=np.float32)}
        with tempfile.TemporaryDirectory() as folder:
            path = save_bundle(f"{folder}/dyn.afpkg", g, weights)
            deployment = load_bundle(path)
            out = deployment.run({"x": np.ones((3, 4), dtype=np.float32)})
            self.assertEqual(out["y"].shape, (3, 4))

        symbolic_weight = Graph("bad")
        symbolic_weight.input("w", f32["N"])
        symbolic_weight.output(symbolic_weight.silu(symbolic_weight.find("w")), "y")
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(BundleError) as ctx:
                save_bundle(
                    f"{folder}/bad.afpkg",
                    symbolic_weight,
                    {"w": np.ones(2, dtype=np.float32)},
                )
            self.assertIn("静态形状", str(ctx.exception))

    def test_reshape_cannot_express_a_symbol_expression(self):
        """``(B,4) -> (4*B,)`` 需要表达式目标，reshape 在构图期拒绝。"""
        g = Graph("reshape")
        x = g.input("x", f32["B", 4])
        with self.assertRaises(ValueError) as ctx:
            g.reshape(x, shape=(sym_mul(4, "B"),))
        self.assertIn("符号表达式", str(ctx.exception))

    def test_torch_import_rejects_dynamic_dims(self):
        """torch 的动态维不能靠 int() 静默具体化。"""

        class _SymInt:
            def __repr__(self):
                return "s0"

        class _Value:
            shape = (_SymInt(), 4)

        with self.assertRaises(UnsupportedOperatorError) as ctx:
            _shape_of(_Value())
        self.assertIn("动态维", str(ctx.exception))

        class _Static:
            shape = (2, 4)

        self.assertEqual(_shape_of(_Static()), (2, 4))


if __name__ == "__main__":
    unittest.main()
