"""激活重算（梯度检查点）的验收测试。

三条判据：

1. **数值**：分段反向与整图反向给出同样的梯度——链式模型要求**逐位**一致；
2. **静态预算**：分段后的峰值（全部边界 + 最宽一段）必须小于整张反向图的
   总字节数，且不需要 GPU 就能算；
3. **契约**：段数不合法时明确报错，单段与整图等价。
"""

import unittest

import numpy as np

from aetherflow import Graph, ReferenceBackend, backward, f32
from aetherflow.ir.liveness import plan_memory
from aetherflow.ir.ops import OPS, UNREPLAYABLE_EFFECTS, Effect, Op
from aetherflow.plugin import register_operator, unregister_operator
from aetherflow.train import SGD, ParameterStore, Trainer
from aetherflow.train.recompute import CheckpointPlan


BACKEND = ReferenceBackend()


def _chain(depth=4, width=16, batch=8):
    """一条链式 MLP：每层的激活只被下一层使用（边界只有一个消费者段）。"""
    g = Graph("mlp")
    x = g.input("x", f32[batch, width])
    h = x
    names = []
    for i in range(depth):
        w = g.input(f"w{i}", f32[width, width])
        h = g.relu(g.matmul(h, w))
        names.append(w.name)
    w = g.input("wout", f32[width, 1])
    names.append(w.name)
    # 两次归约把 (batch, 1) 收成标量（IR 里没有 0 维类型字面量）
    g.output(g.reduce_mean(g.reduce_mean(g.matmul(h, w))), "loss")
    return g, names


def _store(g):
    return ParameterStore.from_graph(g, exclude=("x",), seed=3)


def _batch(rng, shape=(8, 16)):
    return {"x": rng.standard_normal(shape).astype(np.float32)}


class TestCheckpointGradients(unittest.TestCase):
    def test_segmented_matches_monolithic_bitwise(self):
        g, names = _chain()
        ref_store = _store(g)
        seg_store = _store(g)
        rng = np.random.default_rng(5)

        ref = Trainer(g, ref_store, SGD(ref_store.arrays(), lr=0.05), backend=BACKEND)
        seg = Trainer(
            g, seg_store, SGD(seg_store.arrays(), lr=0.05), backend=BACKEND, segments=3
        )
        for _ in range(4):
            batch = _batch(rng)
            self.assertEqual(ref.step(batch), seg.step(batch))
            for name in names:
                np.testing.assert_array_equal(ref_store[name], seg_store[name])

    def test_single_segment_equals_the_whole_graph(self):
        g, names = _chain(depth=3)
        ref_store = _store(g)
        seg_store = _store(g)
        ref = Trainer(g, ref_store, SGD(ref_store.arrays(), lr=0.1), backend=BACKEND)
        seg = Trainer(
            g, seg_store, SGD(seg_store.arrays(), lr=0.1), backend=BACKEND, segments=1
        )
        batch = _batch(np.random.default_rng(9))
        ref.step(batch)
        seg.step(batch)
        for name in names:
            np.testing.assert_array_equal(ref_store[name], seg_store[name])

    def test_more_segments_than_nodes_is_rejected(self):
        g, names = _chain(depth=1, width=4, batch=2)
        store = _store(g)
        with self.assertRaises(ValueError):
            Trainer(g, store, SGD(store.arrays()), backend=BACKEND, segments=10)

    def test_segment_count_must_be_positive_int(self):
        g, _ = _chain(depth=1, width=4, batch=2)
        store = _store(g)
        for bad in (0, -1, 2.0, True):
            with self.assertRaises(ValueError):
                Trainer(g, store, SGD(store.arrays()), backend=BACKEND, segments=bad)


class TestInactiveOutputPruning(unittest.TestCase):
    def test_inactive_output_is_pruned_from_segments(self):
        """AF-04：分段路径同样按损失输出剪枝。

        ``loss=sum(x*x)`` 与诊断输出 ``sum(log(x))``，x=0：给诊断段喂零种子
        仍会算出 ``0/0``。``output="loss"`` 时诊断分支既不出现在段的导出里，
        也不出现在段反向图里。
        """
        g = Graph("diag")
        x = g.input("x", f32[2])
        g.output(g.reduce_sum(g.mul(x, x), axis=0, keepdims=False), "loss")
        g.output(
            g.reduce_sum(g.add_node("log", [x]), axis=0, keepdims=False), "diag"
        )
        plan = CheckpointPlan(g, wrt=[x], segments=2, output="loss")
        exported = {name for seg in plan.segments for name, _ in seg.outputs}
        self.assertNotIn(g.outputs[1].name, exported)

        feed = {"x": np.zeros(2, np.float32)}
        acts = plan.forward(BACKEND, feed)
        with np.errstate(invalid="ignore", divide="ignore"):
            grads = plan.backward(
                BACKEND,
                feed,
                acts,
                seeds={g.outputs[0].name: np.ones((), np.float32)},
            )
        self.assertTrue(np.array_equal(grads["x"], np.zeros(2, np.float32)))

    def test_unknown_output_is_rejected(self):
        g = Graph("two")
        x = g.input("x", f32[2])
        g.output(g.reduce_sum(g.mul(x, x), axis=0, keepdims=False), "loss")
        with self.assertRaises(KeyError):
            CheckpointPlan(g, wrt=[x], segments=1, output="nope")


class TestCheckpointBudget(unittest.TestCase):
    def test_peak_shrinks_against_the_whole_backward_graph(self):
        """静态口径：整图反向一次 execute 不释放任何中间值，分段把这个和拆开。"""
        g, names = _chain(depth=6, width=32, batch=16)
        params = [g.find(name) for name in names]
        mono = plan_memory(backward(g, wrt=params)).total_bytes

        plan = CheckpointPlan(g, params, segments=4)
        self.assertEqual(len(plan.segments), 4)
        held = {orig for seg in plan.segments for orig, _ in seg.outputs}
        held_bytes = sum(
            g.find(name).type.numel()
            * np.dtype(g.find(name).type.dtype.np_dtype).itemsize
            for name in held
        )
        worst = 0
        for seg in plan.segments:
            budget = plan_memory(seg.backward)
            # 边界是跨段存活的，别在每一段里重复计一遍
            local = {alias for orig, alias in seg.inputs if orig in held}
            reused = sum(f.bytes for f in budget.values if f.name in local)
            worst = max(worst, budget.total_bytes - reused)
        self.assertLess(held_bytes + worst, mono)

    def test_plan_covers_every_node_exactly_once(self):
        g, names = _chain(depth=5, width=8, batch=4)
        plan = CheckpointPlan(g, [g.find(n) for n in names], segments=3)
        replayed = [seg.graph.op_names() for seg in plan.segments]
        flat = [op for ops in replayed for op in ops]
        self.assertEqual(flat, g.op_names())
        # 每个边界值由且只由一个段产出
        produced = [orig for seg in plan.segments for orig, _ in seg.outputs]
        self.assertEqual(len(produced), len(set(produced)))

    def test_auxiliary_output_seed_adds_to_downstream_cotangent(self):
        """辅助输出被喂零种子时，不能覆盖后段传回的非零梯度。

        图是 ``a=x*x``、``y=a*a``，两者都是图输出。对 y 求导、a 的种子给 0，
        在 x=2 处 dy/dx = 4*a*x = 32。早先的写法让零种子把 a 的下游 cotangent
        整个吃掉，分段结果变成 0。
        """
        g = Graph("aux")
        x = g.input("x", f32[1])
        a = g.mul(x, x)
        y = g.mul(a, a)
        g.output(a, "aux")
        g.output(y, "y")

        feed = {"x": np.array([2.0], np.float32)}
        plan = CheckpointPlan(g, wrt=[x], segments=2)
        acts = plan.forward(BACKEND, feed)
        grads = plan.backward(
            BACKEND,
            feed,
            acts,
            seeds={
                a.name: np.zeros(1, np.float32),
                y.name: np.ones(1, np.float32),
            },
        )
        self.assertAlmostEqual(float(grads["x"][0]), 32.0, places=5)

        # 整图反向是同一件事的 oracle
        ref = BACKEND.execute(
            backward(g, wrt=[x]),
            {**feed, "daux": np.zeros(1, np.float32), "dy": np.ones(1, np.float32)},
        )
        np.testing.assert_allclose(grads["x"], ref["dx"], rtol=1e-6, atol=1e-6)

    def test_integer_inputs_are_not_differentiated(self):
        """段反向只对浮点输入求导：embedding 的 ids 不是可微量。"""
        from aetherflow import i32

        g = Graph("embed")
        table = g.input("table", f32[4, 3])
        ids = g.input("ids", i32[2])
        h = g.embedding(table, ids)
        g.output(g.reduce_sum(g.reduce_sum(h, axis=1), axis=0), "loss")

        plan = CheckpointPlan(g, wrt=[table], segments=1)
        feed = {
            "table": np.arange(12, dtype=np.float32).reshape(4, 3),
            "ids": np.array([1, 3], np.int32),
        }
        acts = plan.forward(BACKEND, feed)
        grads = plan.backward(
            BACKEND, feed, acts, seeds={g.outputs[0].name: np.ones((), np.float32)}
        )
        want = np.zeros((4, 3), np.float32)
        want[1] = 1.0
        want[3] = 1.0
        np.testing.assert_allclose(grads["table"], want, rtol=0, atol=0)


class TestCheckpointEffects(unittest.TestCase):
    """分段重算会重放整段前向，因此只接受可重放的效果（AF-32 第三点）。"""

    def test_sampling_is_replayable(self):
        g = Graph("drop")
        x = g.input("x", f32[4, 4])
        m = g.dropout_mask(x, p=0.5, seed=9)
        g.output(g.reduce_mean(g.reduce_mean(g.mul(x, m))), "loss")
        plan = CheckpointPlan(g, wrt=[x], segments=2)

        feed = {"x": np.ones((4, 4), np.float32)}
        loss = g.outputs[0].name
        first = plan.forward(BACKEND, feed)[loss]
        second = plan.forward(BACKEND, feed)[loss]
        np.testing.assert_array_equal(first, second)

    def test_state_and_io_effects_are_rejected(self):
        relu = OPS["relu"]
        for effect in (Effect.STATE, Effect.IO):
            name = f"probe.{effect.value}"
            register_operator(
                Op(
                    name=name,
                    infer=relu.infer,
                    reference=relu.reference,
                    effects=frozenset({effect}),
                ),
                namespace="probe",
            )
            try:
                g = Graph("blocked")
                x = g.input("x", f32[2])
                g.output(g.add_node(name, [x]), "y")
                with self.assertRaises(ValueError) as ctx:
                    CheckpointPlan(g, wrt=[x], segments=1)
                self.assertIn(effect.value, str(ctx.exception))
            finally:
                unregister_operator(name)

    def test_rng_is_not_unreplayable(self):
        self.assertNotIn(Effect.RNG, UNREPLAYABLE_EFFECTS)


if __name__ == "__main__":
    unittest.main()
