"""动态形状契约的验收测试。

判据分三层：

1. 契约表本身：每个环节都有明确状态与非空理由，未知环节不默认放行；
2. ``rejected`` 环节在入口显式拒绝：分段重算在构造期报错，设备态训练拒绝
   符号输入，错误里带环节名、具体输入与可操作建议；
3. ``full`` / ``per-op`` / ``data-only`` 的声明不是空话：同一张符号图能被
   执行器按不同 batch 跑、部署包只要求权重静态、reshape 的 ``(B,4) -> (4*B,)``
   在构图期被拒绝。
"""

import tempfile
import unittest

import numpy as np

from aetherflow import Graph, backward, f32
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
        for stage in ("execute", "autodiff", "trainer"):
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
        require_static_shapes(_static_graph(), stage="recompute")


class TestRejectedStages(unittest.TestCase):
    def test_host_state_trainer_runs_symbolic_batches(self):
        """``trainer`` 声明为 per-op：主机态的符号 batch 能真的训练一步。

        一次构图、多形状执行；损失对符号维求平均时，反向的计数按每步的实际
        形状算。参数形状与 batch 无关，优化器看不出差别。
        """
        g = _symbolic_graph()
        store = ParameterStore.from_graph(g, exclude=("x", "t"))
        trainer = Trainer(g, store, SGD(store.arrays(), lr=0.05))
        before = store["w"].copy()
        for batch in (1, 3, 5):
            loss = trainer.step(
                {
                    "x": np.ones((batch, 4), dtype=np.float32),
                    "t": np.zeros((batch, 1), dtype=np.float32),
                }
            )
            self.assertIsInstance(loss, float)
            self.assertTrue(np.isfinite(loss))
        self.assertFalse(np.array_equal(store["w"], before))

    def test_device_state_trainer_rejects_symbolic_input(self):
        """设备态的捕获计划把维度写死在图里，符号批维必须在构造期拒绝。"""
        g = _symbolic_graph()
        store = ParameterStore.from_graph(g, exclude=("x", "t"))
        with self.assertRaises(ValueError) as ctx:
            Trainer(g, store, SGD(store.arrays(), lr=0.1), device_state=True)
        message = str(ctx.exception)
        self.assertIn("设备态训练要求静态形状", message)
        self.assertIn("%x", message)
        self.assertIn("device_state=False", message)

    def test_data_parallel_rejects_symbolic_input_with_shape_hint(self):
        """数据并行按每卡固定 batch 切分：符号批维没有"该切多少"。"""
        g = _symbolic_graph()
        store = ParameterStore.from_graph(g, exclude=("x", "t"))
        trainer = Trainer(g, store, SGD(store.arrays(), lr=0.1))
        with self.assertRaises(ValueError) as ctx:
            trainer.data_input_shapes()
        self.assertIn("含符号维", str(ctx.exception))

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

    def test_reshape_accepts_symbol_expression_target(self):
        """``(B,4) -> (4*B,)`` 的目标可以是表达式：构图期只查符号是否已定义，
        具体取值在执行期按绑定求值。"""
        g = Graph("reshape")
        x = g.input("x", f32["B", 4])
        y = g.reshape(x, shape=(sym_mul(4, "B"),))
        g.output(y, "y")
        self.assertEqual(y.type.shape, (sym_mul(4, "B"),))

    def test_symbolic_reshape_runs_on_several_batches(self):
        """执行器把表达式目标解析成实际整数：同一张图跑不同 batch。"""
        g = Graph("reshape-run")
        x = g.input("x", f32["B", 4])
        g.output(g.reshape(x, shape=(sym_mul(4, "B"),)), "y")
        backend = ReferenceBackend()
        for batch in (1, 3, 7):
            data = np.arange(batch * 4, dtype=np.float32).reshape(batch, 4)
            out = backend.execute(g, {"x": data})["y"]
            self.assertEqual(out.shape, (4 * batch,))
            np.testing.assert_array_equal(out, data.reshape(-1))

    def test_symbolic_reshape_rejects_unknown_symbol(self):
        g = Graph("reshape-bad")
        x = g.input("x", f32["B", 4])
        with self.assertRaises(ValueError) as ctx:
            g.reshape(x, shape=(sym_mul(4, "S"),))
        self.assertIn("未定义的符号维", str(ctx.exception))

    def test_symbolic_reduce_mean_backward(self):
        """归约轴静态、batch 维符号：反向的计数在运行期按形状算。

        对 ``(B,4)`` 沿最后一维求均值，正确梯度是每元素 1/4。静态实现会把
        ``1/count`` 写成常量；符号 batch 走的是 broadcast_to + reduce_sum
        的运行期计数，两条路径都要给出同一个值。
        """
        g = Graph("dyn-mean")
        x = g.input("x", f32["B", 4])
        g.output(g.reduce_mean(x, axis=(-1,), keepdims=False), "y")
        bwd = backward(g, wrt=[x])
        backend = ReferenceBackend()
        for batch in (1, 3, 7):
            data = np.arange(batch * 4, dtype=np.float32).reshape(batch, 4)
            seed = np.ones((batch,), dtype=np.float32)
            grads = backend.execute(bwd, {"x": data, "dy": seed})
            np.testing.assert_allclose(
                grads["dx"], np.full((batch, 4), 0.25, dtype=np.float32), rtol=1e-6, atol=1e-6
            )

    def test_symbolic_reduce_sum_backward(self):
        g = Graph("dyn-sum")
        x = g.input("x", f32["B", 4])
        g.output(g.reduce_sum(x, axis=(-1,), keepdims=False), "y")
        bwd = backward(g, wrt=[x])
        backend = ReferenceBackend()
        for batch in (1, 3, 7):
            data = np.ones((batch, 4), dtype=np.float32)
            grads = backend.execute(bwd, {"x": data, "dy": np.ones((batch,), np.float32)})
            np.testing.assert_array_equal(grads["dx"], np.ones((batch, 4), dtype=np.float32))

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
