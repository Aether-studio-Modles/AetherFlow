"""中层优化 pass 测试。

重点不是"节点变少了"，而是两件事同时成立：
1. 结构按预期被化简
2. 优化前后在相同输入下输出数值等价（差分测试）

只验证第 1 点的 pass 测试是不安全的——很多 bug 表现为"图更小了，结果错了"。
"""

import unittest

import numpy as np

from aetherflow import Graph, ReferenceBackend, dumps, f32, loads, verify
from aetherflow.ir.ops import OPS, Effect, Op
from aetherflow.models import BlockSpec, build_transformer_block
from aetherflow.passes import (
    CommonSubexpressionElimination,
    ConstantFolding,
    DeadCodeElimination,
    FuseLayerNormResidual,
    FuseSiLU,
    FuseSoftmaxScale,
    PassManager,
    SimplifyShapeOps,
    default_pipeline,
    fused_pipeline,
)


class TestDeadCodeElimination(unittest.TestCase):
    def test_removes_unreachable_branch(self):
        g = Graph()
        x = g.input("x", f32[4, 4])
        a = g.relu(x)
        b = g.silu(x)
        g.add(a, b)  # 无用分支
        g.output(a, "y")

        out = DeadCodeElimination().run(g)
        self.assertEqual([n.op.name for n in out.nodes], ["relu"])

    def test_removes_unused_input(self):
        g = Graph()
        x = g.input("x", f32[4])
        g.input("unused", f32[8])
        g.output(g.relu(x), "y")

        out = DeadCodeElimination().run(g)
        self.assertEqual([v.name for v in out.inputs], ["x"])

    def test_keeps_all_nodes_when_everything_is_used(self):
        g = Graph()
        x = g.input("x", f32[4])
        g.output(g.relu(g.silu(x)), "y")
        out = DeadCodeElimination().run(g)
        self.assertEqual([n.op.name for n in out.nodes], ["silu", "relu"])

    def test_saved_report(self):
        g = Graph()
        x = g.input("x", f32[4])
        a = g.relu(x)
        g.add(a, g.silu(x))
        g.output(a, "y")
        self.assertEqual(DeadCodeElimination().saved(g), 2)


class TestCommonSubexpressionElimination(unittest.TestCase):
    def test_merges_identical_nodes(self):
        g = Graph()
        x = g.input("x", f32[8, 8])
        a = g.matmul(x, x)
        b = g.matmul(x, x)
        g.output(g.add(a, b), "y")

        self.assertEqual(CommonSubexpressionElimination().duplicates(g), 1)
        out = CommonSubexpressionElimination().run(g)
        self.assertEqual([n.op.name for n in out.nodes], ["matmul", "add"])

        xv = np.random.default_rng(0).standard_normal((8, 8)).astype(np.float32)
        got = ReferenceBackend().execute(out, {"x": xv})["y"]
        np.testing.assert_allclose(got, 2.0 * (xv @ xv), rtol=1e-5, atol=1e-5)

    def test_different_attrs_are_not_merged(self):
        g = Graph()
        x = g.input("x", f32[8, 8])
        a = g.softmax(x, axis=-1)
        b = g.softmax(x, axis=0)
        g.output(g.add(a, b), "y")
        out = CommonSubexpressionElimination().run(g)
        self.assertEqual(len(out.nodes), 3)

    def test_two_outputs_merged_into_one_value(self):
        g = Graph()
        x = g.input("x", f32[8, 8])
        a = g.matmul(x, x)
        b = g.matmul(x, x)
        g.output(a, "a")
        g.output(b, "b")

        out = CommonSubexpressionElimination().run(g)
        xv = np.random.default_rng(1).standard_normal((8, 8)).astype(np.float32)
        res = ReferenceBackend().execute(out, {"x": xv})
        self.assertEqual(sorted(res), ["a", "b"])
        np.testing.assert_allclose(res["a"], res["b"])


class TestSimplifyShapeOps(unittest.TestCase):
    def test_removes_noop_reshape_transpose_cast(self):
        g = Graph()
        x = g.input("x", f32["B", 64])
        r = g.reshape(x, shape=("B", 64))
        t = g.transpose(r, perm=(0, 1))
        c = g.cast(x, dtype="f32")
        g.output(g.add(t, c), "y")

        out = SimplifyShapeOps().run(g)
        self.assertEqual([n.op.name for n in out.nodes], ["add"])

    def test_keeps_meaningful_shape_ops(self):
        g = Graph()
        x = g.input("x", f32["B", 4, 8])
        t = g.transpose(x, perm=(0, 2, 1))
        g.output(t, "y")
        out = SimplifyShapeOps().run(g)
        self.assertEqual([n.op.name for n in out.nodes], ["transpose"])


class TestPipeline(unittest.TestCase):
    spec = BlockSpec(d_model=32, num_heads=4, d_ff=64)

    def _build(self):
        g = Graph("block")
        inputs, out = build_transformer_block(g, self.spec)
        g.output(out, "y")
        return g, inputs

    def _feed(self):
        rng = np.random.default_rng(7)
        feed = {"x": rng.standard_normal((2, 5, self.spec.d_model)).astype(np.float32)}
        for name, shape in self.spec.param_shapes.items():
            if name == "scale":
                feed[name] = np.array([1.0 / np.sqrt(self.spec.head_dim)], dtype=np.float32)
            else:
                feed[name] = (rng.standard_normal(shape) * 0.1).astype(np.float32)
        return feed

    def test_pipeline_preserves_numerics(self):
        g, _ = self._build()
        opt = default_pipeline().run(g)
        feed = self._feed()
        backend = ReferenceBackend()
        np.testing.assert_allclose(
            backend.execute(opt, feed)["y"],
            backend.execute(g, feed)["y"],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_pipeline_output_is_valid_ir(self):
        g, _ = self._build()
        opt = default_pipeline().run(g)
        self.assertEqual([str(i) for i in verify(opt)], [])

    def test_pipeline_is_idempotent(self):
        g, _ = self._build()
        once = default_pipeline().run(g)
        twice = default_pipeline().run(once)
        self.assertEqual([n.op.name for n in once.nodes], [n.op.name for n in twice.nodes])

    def test_pipeline_never_grows_graph(self):
        g, _ = self._build()
        opt = default_pipeline().run(g)
        self.assertLessEqual(len(opt.nodes), len(g.nodes))
        self.assertLessEqual(len(opt.inputs), len(g.inputs))

    def test_pass_manager_validates_intermediate_graphs(self):
        g, _ = self._build()
        pm = PassManager([SimplifyShapeOps(), CommonSubexpressionElimination(), DeadCodeElimination()])
        self.assertEqual(len(pm.run(g).nodes), len(default_pipeline().run(g).nodes))


class TestFuseSoftmaxScale(unittest.TestCase):
    """IR 层融合：softmax(mul(x, s)) -> softmax(x, s)，s 为标量。"""

    def _graph(self, left_scalar=False):
        g = Graph()
        x = g.input("x", f32[3, 5])
        s = g.input("s", f32[1])
        scaled = g.mul(s, x) if left_scalar else g.mul(x, s)
        g.output(g.softmax(scaled, axis=-1), "y")
        return g

    def _feed(self):
        rng = np.random.default_rng(3)
        return {
            "x": rng.standard_normal((3, 5)).astype(np.float32),
            "s": np.array([2.5], dtype=np.float32),
        }

    def test_structure(self):
        out = FuseSoftmaxScale().run(self._graph())
        self.assertEqual(out.op_names(), ["softmax"])
        self.assertEqual(len(out.nodes[0].inputs), 2)

    def test_numerics_preserved(self):
        g = self._graph()
        fused = FuseSoftmaxScale().run(g)
        ref, feed = ReferenceBackend(), self._feed()
        np.testing.assert_allclose(
            ref.execute(g, feed)["y"],
            ref.execute(fused, feed)["y"],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_scalar_on_left_is_fused(self):
        out = FuseSoftmaxScale().run(self._graph(left_scalar=True))
        self.assertEqual(out.op_names(), ["softmax"])

    def test_fanout_blocks_fusion(self):
        """mul 的结果还被别处使用，就不能被 softmax 吸收。"""
        g = Graph()
        x = g.input("x", f32[3, 5])
        s = g.input("s", f32[1])
        scaled = g.mul(x, s)
        g.output(g.softmax(scaled, axis=-1), "y")
        g.output(g.relu(scaled), "r")
        self.assertIn("mul", FuseSoftmaxScale().run(g).op_names())

    def test_graph_output_blocks_fusion(self):
        g = Graph()
        x = g.input("x", f32[3, 5])
        s = g.input("s", f32[1])
        scaled = g.mul(x, s)
        g.output(g.softmax(scaled, axis=-1), "y")
        g.output(scaled, "z")
        self.assertIn("mul", FuseSoftmaxScale().run(g).op_names())

    def test_non_scalar_is_not_fused(self):
        """softmax(x * y) 里的 y 不是标量，不能折叠成 scale。"""
        g = Graph()
        x = g.input("x", f32[3, 5])
        y = g.input("y", f32[3, 5])
        g.output(g.softmax(g.mul(x, y), axis=-1), "z")
        self.assertIn("mul", FuseSoftmaxScale().run(g).op_names())


class TestFusedPipelineOnBlock(unittest.TestCase):
    """真实模型上的融合：Transformer Block 的 attention 缩放被吸收进 softmax。"""

    spec = BlockSpec(d_model=16, num_heads=4, d_ff=32)

    def _build(self):
        g = Graph("block")
        inputs, out = build_transformer_block(g, self.spec)
        g.output(out, "y")
        rng = np.random.default_rng(7)
        feed = {"x": rng.standard_normal((2, 5, self.spec.d_model)).astype(np.float32)}
        for name, shape in self.spec.param_shapes.items():
            if name == "scale":
                feed[name] = np.array([1.0 / np.sqrt(self.spec.head_dim)], np.float32)
            else:
                feed[name] = (rng.standard_normal(shape) * 0.1).astype(np.float32)
        return g, feed

    def test_attention_scaling_is_fused(self):
        g, _ = self._build()
        fused = fused_pipeline().run(g)
        self.assertEqual([n.op.name for n in fused.nodes].count("mul"), 0)
        self.assertLess(len(fused.nodes), len(g.nodes))

    def test_numerics_preserved(self):
        g, feed = self._build()
        fused = fused_pipeline().run(g)
        ref = ReferenceBackend()
        np.testing.assert_allclose(
            ref.execute(g, feed)["y"],
            ref.execute(fused, feed)["y"],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_fused_graph_is_valid_ir(self):
        g, _ = self._build()
        self.assertEqual([str(i) for i in verify(fused_pipeline().run(g))], [])


class TestConstantFolding(unittest.TestCase):
    """常量子图在编译期算掉，结果与参考语义逐位一致。"""

    def test_folds_all_constant_chain(self):
        g = Graph()
        a = g.constant(2.0, shape=(3,))
        b = g.constant(3.0, shape=(3,))
        g.output(g.mul(a, b), "y")

        out = default_pipeline().run(g)
        self.assertEqual(out.op_names(), ["constant"])
        np.testing.assert_allclose(ReferenceBackend().execute(out, {})["y"], [6.0, 6.0, 6.0])

    def test_non_uniform_constant_is_preserved(self):
        """广播后的结果不是标量填充，必须用 data 承载而不是丢掉。"""
        g = Graph()
        a = g.constant(data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], shape=(2, 3))
        b = g.constant(10.0, shape=(3,))
        g.output(g.add(a, b), "y")

        out = default_pipeline().run(g)
        self.assertEqual(out.op_names(), ["constant"])
        np.testing.assert_allclose(
            ReferenceBackend().execute(out, {})["y"],
            np.array([[11.0, 12.0, 13.0], [14.0, 15.0, 16.0]], dtype=np.float32),
        )

    def test_folded_graph_roundtrips_json(self):
        g = Graph()
        c = g.constant(1.5, shape=(2, 2))
        g.output(g.transpose(c, perm=(1, 0)), "y")

        out = ConstantFolding().run(g)
        restored = loads(dumps(out))
        ref = ReferenceBackend()
        np.testing.assert_allclose(
            ref.execute(out, {})["y"], ref.execute(restored, {})["y"]
        )

    def test_input_dependent_nodes_are_kept(self):
        g = Graph()
        x = g.input("x", f32[4])
        c = g.constant(2.0, shape=(4,))
        g.output(g.mul(x, c), "y")

        out = ConstantFolding().run(g)
        self.assertEqual(out.op_names(), ["constant", "mul"])

    def test_symbolic_output_is_not_folded(self):
        g = Graph()
        c = g.constant(1.0, shape=(4,))
        g.output(g.broadcast_to(c, shape=("B", 4)), "y")

        out = ConstantFolding().run(g)
        self.assertEqual(out.op_names(), ["constant", "broadcast_to"])

    def test_random_op_is_not_folded(self):
        """dropout_mask 由 seed 决定，但语义是"每次重新采样"，不能固化。"""
        g = Graph()
        c = g.constant(1.0, shape=(4,))
        g.output(g.dropout_mask(c, p=0.5, seed=7), "y")

        out = ConstantFolding().run(g)
        self.assertEqual(out.op_names(), ["constant", "dropout_mask"])

    def test_size_guard(self):
        g = Graph()
        a = g.constant(2.0, shape=(32,))
        b = g.constant(3.0, shape=(32,))
        g.output(g.mul(a, b), "y")

        self.assertEqual(ConstantFolding(max_elements=16).run(g).op_names(), ["constant", "constant", "mul"])
        self.assertEqual(default_pipeline().run(g).op_names(), ["constant"])

    def test_numerics_preserved(self):
        g = Graph()
        a = g.constant(0.5, shape=(3, 4))
        b = g.constant(-1.25, shape=(4,))
        y = g.sigmoid(g.add(g.mul(a, b), a))
        g.output(y, "y")

        out = ConstantFolding().run(g)
        ref = ReferenceBackend()
        np.testing.assert_allclose(
            ref.execute(g, {})["y"], ref.execute(out, {})["y"], rtol=0, atol=0
        )

    def test_only_boundary_results_are_materialised(self):
        """折叠链的中间结果不该各发一个 constant（5.161）。

        ``c1*c2 + c3`` 三步全可折，但只有 ``+`` 的结果被 ``mul(x, mask)`` 读到；
        中间两个结果的使用者也都在折叠集合里。旧实现给每个中间节点都发一个
        constant，节点数一点没少，执行期还多出两次上传。这里跑完整流水线：
        折叠只留边界，原始常量再被 DCE 收走。"""
        g = Graph()
        x = g.input("x", f32[4])
        c1 = g.constant(2.0, shape=(4,))
        c2 = g.constant(3.0, shape=(4,))
        c3 = g.constant(1.0, shape=(4,))
        mask = g.add(g.mul(c1, c2), c3)
        g.output(g.mul(x, mask), "y")

        out = default_pipeline().run(g)
        self.assertEqual(out.op_names(), ["constant", "mul"])
        ref = ReferenceBackend()
        feed = {"x": np.arange(4, dtype=np.float32)}
        np.testing.assert_allclose(
            ref.execute(g, feed)["y"], ref.execute(out, feed)["y"], rtol=0, atol=0
        )

    def test_count_report(self):
        g = Graph()
        a = g.constant(2.0, shape=(2,))
        b = g.constant(3.0, shape=(2,))
        g.output(g.add(g.mul(a, b), a), "y")
        self.assertEqual(ConstantFolding().count(g), 2)

    def test_folding_is_idempotent(self):
        g = Graph()
        a = g.constant(2.0, shape=(2,))
        b = g.constant(3.0, shape=(2,))
        g.output(g.mul(a, b), "y")
        once = ConstantFolding().run(g)
        twice = ConstantFolding().run(once)
        self.assertEqual(once.op_names(), twice.op_names())
        self.assertEqual(once.nodes[0].attrs, twice.nodes[0].attrs)

    def test_default_pipeline_folds(self):
        g = Graph()
        a = g.constant(2.0, shape=(3,))
        b = g.constant(3.0, shape=(3,))
        g.output(g.mul(a, b), "y")
        self.assertEqual(default_pipeline().run(g).op_names(), ["constant"])


class TestFuseLayerNormResidual(unittest.TestCase):
    """layernorm(add(a, r)) -> layernorm(a, gamma, beta, r)。"""

    def _graph(self):
        g = Graph()
        a = g.input("a", f32[3, 5])
        r = g.input("r", f32[3, 5])
        gamma = g.input("g", f32[5])
        beta = g.input("b", f32[5])
        g.output(g.layernorm(g.add(a, r), gamma, beta, axes=(-1,)), "y")
        return g

    def _feed(self):
        rng = np.random.default_rng(11)
        return {
            "a": rng.standard_normal((3, 5)).astype(np.float32),
            "r": rng.standard_normal((3, 5)).astype(np.float32),
            "g": rng.standard_normal(5).astype(np.float32),
            "b": rng.standard_normal(5).astype(np.float32),
        }

    def test_structure(self):
        out = FuseLayerNormResidual().run(self._graph())
        self.assertEqual(out.op_names(), ["layernorm"])
        self.assertEqual(len(out.nodes[0].inputs), 4)

    def test_numerics_preserved(self):
        g = self._graph()
        fused = FuseLayerNormResidual().run(g)
        ref, feed = ReferenceBackend(), self._feed()
        np.testing.assert_allclose(
            ref.execute(g, feed)["y"], ref.execute(fused, feed)["y"], rtol=1e-6, atol=1e-6
        )

    def test_two_fanouts_block_fusion(self):
        """残差还被别处使用（pre-norm Block 的真实形态），就不能被吸收。"""
        g = Graph()
        a = g.input("a", f32[3, 5])
        r = g.input("r", f32[3, 5])
        gamma = g.input("g", f32[5])
        beta = g.input("b", f32[5])
        s = g.add(a, r)
        g.output(g.layernorm(s, gamma, beta, axes=(-1,)), "y")
        g.output(g.relu(s), "r2")
        out = FuseLayerNormResidual().run(g)
        ln = [n for n in out.nodes if n.op.name == "layernorm"][0]
        self.assertEqual(len(ln.inputs), 3)

    def test_graph_output_blocks_fusion(self):
        g = Graph()
        a = g.input("a", f32[3, 5])
        r = g.input("r", f32[3, 5])
        gamma = g.input("g", f32[5])
        beta = g.input("b", f32[5])
        s = g.add(a, r)
        g.output(g.layernorm(s, gamma, beta, axes=(-1,)), "y")
        g.output(s, "s")
        out = FuseLayerNormResidual().run(g)
        ln = [n for n in out.nodes if n.op.name == "layernorm"][0]
        self.assertEqual(len(ln.inputs), 3)

    def test_broadcast_is_not_absorbed(self):
        """add 依赖广播时，内核没有对应的承接形态，不融合。"""
        g = Graph()
        a = g.input("a", f32[3, 5])
        bias = g.input("bias", f32[5])
        gamma = g.input("g", f32[5])
        beta = g.input("b", f32[5])
        g.output(g.layernorm(g.add(a, bias), gamma, beta, axes=(-1,)), "y")
        out = FuseLayerNormResidual().run(g)
        self.assertEqual(out.op_names(), ["add", "layernorm"])

    def test_idempotent(self):
        once = FuseLayerNormResidual().run(self._graph())
        twice = FuseLayerNormResidual().run(once)
        self.assertEqual(once.op_names(), twice.op_names())
        self.assertEqual(len(twice.nodes[0].inputs), 4)

    def test_gradient_matches_reference(self):
        """融合形态必须仍可求导，且梯度与未融合的图一致。"""
        from aetherflow import backward

        g = self._graph()
        fused = FuseLayerNormResidual().run(g)
        feed = self._feed()
        seed = {"dy": np.ones((3, 5), dtype=np.float32)}
        bwd_ref = backward(g)
        bwd_fused = backward(fused)
        ref = ReferenceBackend()
        for name in ("da", "dr", "dg", "db"):
            np.testing.assert_allclose(
                ref.execute(bwd_fused, {**feed, **seed})[name],
                ref.execute(bwd_ref, {**feed, **seed})[name],
                rtol=1e-5,
                atol=1e-5,
            )


class TestFuseSiLU(unittest.TestCase):
    """mul(a, silu(b)) -> silu(b, a)，SwiGLU 的门控乘法。"""

    def _graph(self, reverse=False):
        g = Graph()
        a = g.input("a", f32[3, 5])
        b = g.input("b", f32[3, 5])
        s = g.silu(b)
        g.output(g.mul(s, a) if reverse else g.mul(a, s), "y")
        return g

    def _feed(self):
        rng = np.random.default_rng(13)
        return {
            "a": rng.standard_normal((3, 5)).astype(np.float32),
            "b": rng.standard_normal((3, 5)).astype(np.float32),
        }

    def test_structure(self):
        out = FuseSiLU().run(self._graph())
        self.assertEqual(out.op_names(), ["silu"])
        self.assertEqual(len(out.nodes[0].inputs), 2)

    def test_either_side(self):
        out = FuseSiLU().run(self._graph(reverse=True))
        self.assertEqual(out.op_names(), ["silu"])

    def test_numerics_preserved(self):
        g = self._graph()
        fused = FuseSiLU().run(g)
        ref, feed = ReferenceBackend(), self._feed()
        np.testing.assert_allclose(
            ref.execute(g, feed)["y"], ref.execute(fused, feed)["y"], rtol=1e-6, atol=1e-6
        )

    def test_fanout_blocks_fusion(self):
        g = Graph()
        a = g.input("a", f32[3, 5])
        b = g.input("b", f32[3, 5])
        s = g.silu(b)
        g.output(g.mul(a, s), "y")
        g.output(s, "s")
        self.assertEqual(FuseSiLU().run(g).op_names(), ["silu", "mul"])

    def test_broadcast_gate_blocks_fusion(self):
        g = Graph()
        a = g.input("a", f32[3, 5])
        s = g.silu(g.input("b", f32[5]))
        g.output(g.mul(a, s), "y")
        self.assertEqual(FuseSiLU().run(g).op_names(), ["silu", "mul"])


class TestFusedPipelineOnBlockExtended(unittest.TestCase):
    """融合管线在真实模型上的命中情况。"""

    spec = BlockSpec(d_model=16, num_heads=4, d_ff=32)

    def _build(self):
        g = Graph("block")
        inputs, out = build_transformer_block(g, self.spec)
        g.output(out, "y")
        rng = np.random.default_rng(7)
        feed = {"x": rng.standard_normal((2, 5, self.spec.d_model)).astype(np.float32)}
        for name, shape in self.spec.param_shapes.items():
            if name == "scale":
                feed[name] = np.array([1.0 / np.sqrt(self.spec.head_dim)], np.float32)
            else:
                feed[name] = (rng.standard_normal(shape) * 0.1).astype(np.float32)
        return g, feed

    def test_pre_norm_residual_does_not_fuse(self):
        """x1 同时喂 layernorm 与最终残差，fanout=2，不该融合。

        这条测试固定的是一个**否定结论**：融合规则不能为了"命中"而放开
        fanout 限制——那样残差仍要物化，融合反而多读一遍。
        """
        g, _ = self._build()
        fused = fused_pipeline().run(g)
        lns = [n for n in fused.nodes if n.op.name == "layernorm"]
        self.assertTrue(lns)
        self.assertTrue(all(len(n.inputs) == 3 for n in lns))

    def test_pipeline_still_valid_and_equal(self):
        g, feed = self._build()
        fused = fused_pipeline().run(g)
        self.assertEqual([str(i) for i in verify(fused)], [])
        ref = ReferenceBackend()
        np.testing.assert_allclose(
            ref.execute(g, feed)["y"], ref.execute(fused, feed)["y"], rtol=1e-5, atol=1e-5
        )


class TestEffects(unittest.TestCase):
    """副作用声明如何约束各优化 pass（AF-32 第三点）。

    这组用例固定的是**否定结论**：效果系统存在的意义就是让某几类化简
    在某些节点上不发生。如果哪天 pass 又"顺手"合并/删除/折叠了采样节点，
    这里必须红。
    """

    def test_only_sampling_declares_effects(self):
        with_effects = {name: op.effects for name, op in OPS.items() if op.effects}
        self.assertEqual(set(with_effects), {"dropout_mask"})
        self.assertEqual(with_effects["dropout_mask"], frozenset({Effect.RNG}))

    def test_pure_is_the_default(self):
        self.assertTrue(OPS["add"].pure)
        self.assertFalse(OPS["dropout_mask"].pure)

    def test_effects_are_normalised(self):
        add = OPS["add"]
        op = Op(
            name="probe",
            infer=add.infer,
            reference=add.reference,
            effects=[Effect.STATE],
        )
        self.assertIsInstance(op.effects, frozenset)
        self.assertFalse(op.pure)

    def test_effects_reject_plain_strings(self):
        add = OPS["add"]
        with self.assertRaises(TypeError):
            Op(name="probe", infer=add.infer, reference=add.reference, effects={"rng"})

    def test_cse_does_not_merge_sampling(self):
        g = Graph()
        x = g.input("x", f32[4, 4])
        a = g.dropout_mask(x, p=0.5, seed=1)
        b = g.dropout_mask(x, p=0.5, seed=1)
        g.output(g.add(a, b), "y")

        cse = CommonSubexpressionElimination()
        self.assertEqual(cse.duplicates(g), 0)
        self.assertEqual(
            cse.run(g).op_names(), ["dropout_mask", "dropout_mask", "add"]
        )

    def test_dce_keeps_unused_sampling_and_its_inputs(self):
        g = Graph()
        x = g.input("x", f32[4, 4])
        h = g.relu(x)
        g.dropout_mask(h, p=0.5, seed=2)  # 结果没有任何使用者
        g.output(h, "y")

        dce = DeadCodeElimination()
        self.assertEqual(dce.saved(g), 0)
        self.assertEqual(dce.run(g).op_names(), ["relu", "dropout_mask"])

    def test_folding_skips_sampling(self):
        g = Graph()
        c = g.constant(data=[1.0, 2.0, 3.0, 4.0], shape=(4,))
        g.output(g.dropout_mask(c, p=0.5, seed=3), "y")

        fold = ConstantFolding()
        self.assertEqual(fold.count(g), 0)
        self.assertEqual(fold.run(g).op_names(), ["constant", "dropout_mask"])

    def test_pipeline_keeps_sampling(self):
        """默认管线跑完，采样节点仍在且只出现一次。"""
        g = Graph()
        x = g.input("x", f32[4, 4])
        m = g.dropout_mask(x, p=0.5, seed=4)
        g.output(g.mul(x, m), "y")

        out = default_pipeline().run(g)
        self.assertEqual([n.op.name for n in out.nodes].count("dropout_mask"), 1)


if __name__ == "__main__":
    unittest.main()
