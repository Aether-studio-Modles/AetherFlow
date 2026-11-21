"""训练循环封装的验收测试。

判据与 ``test_train.py`` 一致：不看形状，看"真的在学"。另外几条属于封装
自身的契约——损失必须是标量、参数必须是图输入、训练图必须是静态形状。
"""

import os
import tempfile
import unittest

import numpy as np

from aetherflow import Graph, f32, i32
from aetherflow.backends import ReferenceBackend
from aetherflow.train.loop import _derive_rng_key
from aetherflow.train import (
    DataLoader,
    GradScaler,
    GradientAccumulator,
    ParameterStore,
    SGD,
    StepDecay,
    Trainer,
    TrainingSnapshot,
    clip_global_norm,
    load_checkpoint,
    mse,
)


def _data(seed=0, n=32):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    true_w = np.array([[1.0], [-2.0], [0.5], [3.0]], dtype=np.float32)
    true_b = np.array([0.7], dtype=np.float32)
    return x, (x @ true_w + true_b).astype(np.float32)


def _graph(batch=8):
    g = Graph("lin")
    x = g.input("x", f32[batch, 4])
    t = g.input("t", f32[batch, 1])
    w = g.input("w", f32[4, 1])
    b = g.input("b", f32[1])
    g.output(mse(g, g.add(g.matmul(x, w), b), t), "loss")
    return g


class TestTrainer(unittest.TestCase):
    def test_converges_on_linear_regression(self):
        x, t = _data()
        g = _graph()
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        trainer = Trainer(g, store, SGD(store.arrays(), lr=0.1))
        losses = trainer.run(DataLoader({"x": x, "t": t}, 8), epochs=50)

        self.assertEqual(len(losses), 200)
        self.assertLess(losses[-1], losses[0])
        self.assertLess(losses[-1], 1e-4)
        np.testing.assert_allclose(store["w"].ravel(), [1.0, -2.0, 0.5, 3.0], atol=1e-2)
        np.testing.assert_allclose(store["b"], [0.7], atol=1e-2)
        self.assertEqual(trainer.steps, 200)

    def test_step_returns_the_graph_loss_and_updates_in_place(self):
        x, t = _data(n=8)
        g = _graph()
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        before = store["w"].copy()
        trainer = Trainer(g, store, SGD(store.arrays(), lr=0.1))

        expected = float(np.mean((x @ before + store["b"] - t) ** 2))
        loss = trainer.step({"x": x, "t": t})

        self.assertAlmostEqual(loss, expected, places=5)
        self.assertFalse(np.allclose(store["w"], before))
        self.assertEqual(trainer.steps, 1)

    def test_non_scalar_output_is_rejected(self):
        g = Graph("vec")
        x = g.input("x", f32[4])
        g.output(g.mul(x, x), "y")
        store = ParameterStore({"x": np.ones(4, dtype=np.float32)})
        with self.assertRaises(ValueError):
            Trainer(g, store, SGD(store.arrays(), lr=0.1)).step({})

    def test_unknown_output_is_rejected(self):
        g = _graph()
        store = ParameterStore.from_graph(g, exclude=("x", "t"))
        with self.assertRaises(ValueError):
            Trainer(g, store, SGD(store.arrays(), lr=0.1), output="nope")

    def test_output_can_be_selected(self):
        g = Graph("two")
        x = g.input("x", f32[4])
        g.output(g.mul(x, x), "squared")
        g.output(g.reduce_sum(g.mul(x, x), axis=0, keepdims=False), "loss")
        store = ParameterStore({"x": np.ones(4, dtype=np.float32)})
        trainer = Trainer(g, store, SGD(store.arrays(), lr=0.1), output="loss")
        self.assertAlmostEqual(trainer.step({}), 4.0, places=6)

    def test_inactive_output_does_not_poison_the_gradient(self):
        """AF-04 的现场：未选输出不能靠喂零种子处理。

        诊断输出 ``sum(log(x))`` 在 x=0 处，零种子仍会算出 ``0/0``，把对
        ``loss=sum(x*x)`` 的梯度污染成 NaN。反向图里根本不该有这条分支。
        """
        g = Graph("diagnostic")
        x = g.input("x", f32[2])
        g.output(g.reduce_sum(g.mul(x, x), axis=0, keepdims=False), "loss")
        g.output(
            g.reduce_sum(g.add_node("log", [x]), axis=0, keepdims=False), "diag"
        )
        store = ParameterStore({"x": np.zeros(2, dtype=np.float32)})
        trainer = Trainer(g, store, SGD(store.arrays(), lr=0.1), output="loss")
        with np.errstate(invalid="ignore", divide="ignore"):
            loss = trainer.step({})
        self.assertEqual(loss, 0.0)
        self.assertTrue(np.all(np.isfinite(store["x"])))
        self.assertTrue(np.array_equal(store["x"], np.zeros(2, dtype=np.float32)))

    def test_store_names_must_be_graph_inputs(self):
        g = _graph()
        other = ParameterStore({"zzz": np.ones(2, dtype=np.float32)})
        with self.assertRaises(ValueError):
            Trainer(g, other, SGD(other.arrays(), lr=0.1))

    def test_symbolic_input_is_rejected_with_a_clear_error(self):
        """反向图的归约需要静态形状：这条边界要在构造期讲清楚。"""
        g = Graph("sym")
        x = g.input("x", f32["B", 4])
        t = g.input("t", f32["B", 1])
        w = g.input("w", f32[4, 1])
        b = g.input("b", f32[1])
        diff = g.sub(g.add(g.matmul(x, w), b), t)
        sq = g.mul(diff, diff)
        loss = g.reduce_sum(g.reduce_sum(sq, axis=0, keepdims=False), axis=0, keepdims=False)
        g.output(loss, "loss")
        store = ParameterStore.from_graph(g, exclude=("x", "t"))
        with self.assertRaises(ValueError) as ctx:
            Trainer(g, store, SGD(store.arrays(), lr=0.1))
        self.assertIn("符号维", str(ctx.exception))

    def test_on_step_receives_global_indices(self):
        x, t = _data(n=8)
        g = _graph(batch=4)
        store = ParameterStore.from_graph(g, exclude=("x", "t"))
        trainer = Trainer(g, store, SGD(store.arrays(), lr=0.01))
        seen = []
        trainer.run(DataLoader({"x": x, "t": t}, 4), epochs=2, on_step=lambda i, v: seen.append(i))
        self.assertEqual(seen, [0, 1, 2, 3])

    def test_epochs_must_be_positive(self):
        g = _graph()
        store = ParameterStore.from_graph(g, exclude=("x", "t"))
        trainer = Trainer(g, store, SGD(store.arrays(), lr=0.1))
        for bad in (0, -1, True):
            with self.subTest(epochs=bad):
                with self.assertRaises(ValueError):
                    trainer.run([], epochs=bad)

    def test_grad_hook_can_skip_and_clip(self):
        """钩子返回 None 时参数不动；返回裁剪结果时位移受 max_norm 约束。"""
        x, t = _data(n=8)
        g = _graph()
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        before = store["w"].copy()
        trainer = Trainer(
            g, store, SGD(store.arrays(), lr=0.1), grad_hook=lambda grads: None
        )

        trainer.step({"x": x, "t": t})
        self.assertEqual(trainer.steps, 1)
        self.assertEqual(trainer.optimizer.step_count, 0)
        np.testing.assert_array_equal(store["w"], before)

        trainer.grad_hook = lambda grads: clip_global_norm(grads, 0.5)[0]
        trainer.step({"x": x, "t": t})
        self.assertEqual(trainer.optimizer.step_count, 1)
        # 全局范数 <= max_norm 时每个分量都 <= max_norm，位移不超过 lr*max_norm
        self.assertLessEqual(float(np.abs(store["w"] - before).max()), 0.1 * 0.5 + 1e-6)

    def test_grad_hook_accumulates_micro_batches(self):
        """两个 4 样本微批的累积，必须等价于一个 8 样本批次的更新。"""
        x, t = _data(n=8)

        big = _graph(batch=8)
        big_store = ParameterStore.from_graph(big, exclude=("x", "t"), seed=0)
        Trainer(big, big_store, SGD(big_store.arrays(), lr=0.1)).step({"x": x, "t": t})

        small = _graph(batch=4)
        small_store = ParameterStore.from_graph(small, exclude=("x", "t"), seed=0)
        acc = GradientAccumulator(2)

        def hook(grads):
            return acc.flush() if acc.add(grads) else None

        trainer = Trainer(small, small_store, SGD(small_store.arrays(), lr=0.1), grad_hook=hook)
        init = ParameterStore.from_graph(small, exclude=("x", "t"), seed=0)
        trainer.step({"x": x[:4], "t": t[:4]})
        # 窗口没满：梯度只进了累积缓冲，参数一个数都没动
        np.testing.assert_array_equal(small_store["w"], init["w"])
        trainer.step({"x": x[4:], "t": t[4:]})

        self.assertEqual(trainer.steps, 2)
        self.assertEqual(trainer.optimizer.step_count, 1)
        for name in small_store.names:
            np.testing.assert_allclose(
                small_store[name], big_store[name], rtol=1e-5, atol=1e-6
            )


class TestTrainerRandomState(unittest.TestCase):
    """AF-20：dropout 的随机源必须是显式状态，并由 Trainer 逐步推进。

    图里 ``dropout_mask(x, rng)`` 的 rng 是一个 i32[1] 输入。Trainer 每步
    先写 step 号再跑前向/反向，所以同一 step 的前向、重算、反向共用同一个
    key，下一步换新 key；同一份图 + 同一个 key 仍然逐位可复现。
    """

    def _build(self):
        g = Graph("drop_rng")
        x = g.input("x", f32[4, 3])
        rng = g.input("rng", i32[1])
        w = g.input("w", f32[3])
        # 两个 dropout 站点共用一个 key、不同 offset：同形状也不会撞掩码
        h = g.mul(g.mul(x, w), g.dropout_mask(x, rng, p=0.5, offset=0))
        h = g.mul(h, g.dropout_mask(h, rng, p=0.5, offset=1))
        g.output(g.reduce_sum(g.reshape(h, shape=(12,)), axis=0, keepdims=False), "loss")
        return g

    def _trainer(self, g):
        store = ParameterStore.from_graph(g, exclude=("x", "rng"))
        return Trainer(
            g, store, SGD(store.arrays(), lr=0.1), rng_inputs=("rng",)
        )

    def test_key_advances_between_steps(self):
        g = self._build()
        trainer = self._trainer(g)
        x = np.ones((4, 3), dtype=np.float32)
        # 钩子返回 None = 不更新参数；损失的变化只可能来自换 key
        trainer.grad_hook = lambda grads: None
        a = trainer.step({"x": x})
        b = trainer.step({"x": x})
        self.assertNotAlmostEqual(a, b)

    def test_same_step_key_is_reproducible(self):
        x = np.ones((4, 3), dtype=np.float32)
        g1, g2 = self._build(), self._build()
        a = self._trainer(g1).step({"x": x})
        b = self._trainer(g2).step({"x": x})
        self.assertAlmostEqual(a, b, places=6)

    def test_segmented_recompute_uses_the_same_key(self):
        """重算若用别的 key，掩码就变了，分段梯度会与整图梯度对不上。"""
        x = np.ones((4, 3), dtype=np.float32)
        whole = self._build()
        whole_trainer = self._trainer(whole)
        seen_whole = {}
        whole_trainer.grad_hook = lambda g: seen_whole.update(g) or None
        whole_trainer.step({"x": x})

        split = self._build()
        store = ParameterStore.from_graph(split, exclude=("x", "rng"))
        split_trainer = Trainer(
            split,
            store,
            SGD(store.arrays(), lr=0.1),
            segments=2,
            rng_inputs=("rng",),
        )
        seen_split = {}
        split_trainer.grad_hook = lambda g: seen_split.update(g) or None
        split_trainer.step({"x": x})

        for name in store.names:
            np.testing.assert_allclose(
                seen_split[name], seen_whole[name], rtol=1e-6, atol=1e-7
            )

    def test_unknown_rng_input_is_rejected(self):
        g = self._build()
        store = ParameterStore.from_graph(g, exclude=("x", "rng"))
        with self.assertRaises(ValueError):
            Trainer(g, store, SGD(store.arrays(), lr=0.1), rng_inputs=("nope",))

    def test_parameter_cannot_be_a_rng_input(self):
        g = self._build()
        store = ParameterStore.from_graph(g, exclude=("x", "rng"))
        with self.assertRaises(ValueError):
            Trainer(g, store, SGD(store.arrays(), lr=0.1), rng_inputs=("w",))

    def test_derived_keys_are_i32_and_distinct(self):
        """64 个 (step, rank) 组合必须给出 64 个不同的 i32 正数。"""
        keys = {
            _derive_rng_key(0, step, rank)
            for step in range(8)
            for rank in range(8)
        }
        self.assertEqual(len(keys), 64)
        self.assertTrue(all(0 <= key < 2 ** 31 for key in keys))

    def test_key_depends_on_seed_and_rank(self):
        a = _derive_rng_key(11, 0, 0)
        self.assertEqual(a, _derive_rng_key(11, 0, 0))
        self.assertNotEqual(a, _derive_rng_key(12, 0, 0))
        self.assertNotEqual(a, _derive_rng_key(11, 0, 1))

    def test_ranks_draw_different_masks(self):
        """同参数、同数据、同 step：不同 rank 的 dropout 掩码必须不同。"""
        x = np.ones((4, 3), dtype=np.float32)
        losses = []
        for rank in range(2):
            g = self._build()
            store = ParameterStore.from_graph(g, exclude=("x", "rng"))
            trainer = Trainer(
                g,
                store,
                SGD(store.arrays(), lr=0.1),
                rng_inputs=("rng",),
                rng_rank=rank,
                rng_world_size=2,
            )
            trainer.grad_hook = lambda grads: None
            losses.append(trainer.step({"x": x}))
        self.assertNotAlmostEqual(losses[0], losses[1])

    def test_invalid_rank_or_seed_is_rejected(self):
        g = self._build()
        store = ParameterStore.from_graph(g, exclude=("x", "rng"))
        bad = (
            {"rng_rank": 2, "rng_world_size": 2},
            {"rng_rank": -1},
            {"rng_seed": -1},
            {"rng_seed": 2 ** 31},
            {"rng_world_size": 0},
        )
        for kwargs in bad:
            with self.assertRaises(ValueError):
                Trainer(
                    g,
                    store,
                    SGD(store.arrays(), lr=0.1),
                    rng_inputs=("rng",),
                    **kwargs,
                )


class _FakePlan:
    """设备态计划的替身：把"共享缓冲 + replay"用 numpy 如实演一遍。

    它不模拟显存，只模拟两件事：图读的是**捕获时那块**缓冲（所以写入
    ``run`` 之外的输入不会改变结果），以及每次 replay 把同一批输出缓冲
    重新算出来。有了它，"设备态 + 分段"的调度逻辑（清零、累加、段序）
    在没有任何 GPU 的机器上也能验收。
    """

    def __init__(self, backend, graph, buffers):
        self.backend = backend
        self.graph = graph
        self.buffers = buffers
        self.outputs = {}

    def run_device(self, inputs):
        for name, buf in self.buffers.items():
            value = inputs[name]
            if value is not buf:
                buf[...] = value
        result = self.backend.execute(self.graph, self.buffers)
        self.outputs = {name: result[name] for name in self.graph.output_names}
        return dict(self.outputs)


class _FakeDeviceBackend(ReferenceBackend):
    """把 numpy 数组当"显存数组"：只补 capture / to_array / to_host / 流。"""

    def to_array(self, value):
        return np.array(value, copy=True)

    def to_host(self, value):
        return np.asarray(value)

    def new_stream(self):
        return object()

    def capture(self, graph, sample_inputs, reuse_memory=True, shared_buffers=None, stream=None):
        shared = dict(shared_buffers or {})
        buffers = {}
        for value in graph.inputs:
            name = value.name
            if name in shared:
                buffers[name] = shared[name]
            else:
                buffers[name] = np.array(sample_inputs[name], copy=True)
        return _FakePlan(self, graph, buffers)


class _FakeDeviceTrainer(Trainer):
    """绕过"必须是真的设备数组"这道门（替身的数组模块就是 numpy）。"""

    def _upload_params(self):
        source = self._resident if self._resident is not None else self.store.arrays()
        return {name: np.array(source[name], copy=True) for name in self.store.names}

    def _upload_master(self, resident):
        # 替身没有真正的显存数组，主副本也只是另一份 numpy 拷贝。
        if self._resident is None:
            return resident
        return {name: np.array(self.store[name], copy=True) for name in self.store.names}


def _deep_graph(batch=4):
    """两段之间有一条非参数的边界（relu 的输出），反向必须靠它重放前向。"""
    g = Graph("deep")
    x = g.input("x", f32[batch, 6])
    t = g.input("t", f32[batch, 1])
    w1 = g.input("w1", f32[6, 8])
    b1 = g.input("b1", f32[8])
    w2 = g.input("w2", f32[8, 8])
    b2 = g.input("b2", f32[8])
    w3 = g.input("w3", f32[8, 1])
    b3 = g.input("b3", f32[1])
    h = g.relu(g.add(g.matmul(x, w1), b1))
    h = g.relu(g.add(g.matmul(h, w2), b2))
    g.output(mse(g, g.add(g.matmul(h, w3), b3), t), "loss")
    return g


class TestDeviceStateSegments(unittest.TestCase):
    """设备态与激活重算叠加时，调度必须与主机态分段训练给出同一条轨迹。"""

    def _run(self, device_state):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(4, 6)).astype(np.float32)
        t = rng.normal(size=(4, 1)).astype(np.float32)
        g = _deep_graph()
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=3)
        backend = _FakeDeviceBackend() if device_state else None
        cls = _FakeDeviceTrainer if device_state else Trainer
        trainer = cls(
            g,
            store,
            SGD(store.arrays(), lr=0.05),
            backend=backend,
            segments=2,
            device_state=device_state,
        )
        losses = [trainer.step({"x": x, "t": t}) for _ in range(3)]
        trainer.sync_to_host()
        return trainer, losses, {name: store[name].copy() for name in store.names}

    def test_matches_host_segments(self):
        trainer, device_losses, device_params = self._run(True)
        _, host_losses, host_params = self._run(False)

        np.testing.assert_allclose(device_losses, host_losses, rtol=1e-6, atol=1e-7)
        for name, expected in host_params.items():
            np.testing.assert_allclose(device_params[name], expected, rtol=1e-6, atol=1e-7)

        # 每段各有前后向两张计划；边界值与 cotangent 必须是两组缓冲
        self.assertEqual(len(trainer._seg_fwd), len(trainer._plan.segments))
        self.assertEqual(len(trainer._seg_bwd), len(trainer._plan.segments))
        self.assertTrue(trainer._dev_cots)
        for name, cot in trainer._dev_cots.items():
            self.assertIsNot(cot, trainer._dev_acts[name])

    def test_loss_is_read_from_the_boundary_buffer(self):
        """损失只 D2H 一次，且读的就是最后一段写进边界缓冲的那个值。"""
        trainer, losses, _ = self._run(True)
        self.assertEqual(len(losses), 3)
        boundary = trainer._dev_acts[trainer._loss_value.name]
        self.assertAlmostEqual(
            losses[-1], float(trainer.backend.to_host(boundary)), places=6
        )


class TestTrainingStateAndSnapshot(unittest.TestCase):
    """AF-29：训练状态的版本凭证，与"拷贝在同步点、写盘在后台"的快照。

    设备态训练的收益建立在一条隐式契约上——优化器原地更新的数组、捕获计划
    里写死的缓冲、驻留参数表是同一块显存。这里既验证"正常时它成立"，也验证
    "被换掉一份数组时能被查出来"，否则测试只是在重复实现的假设。
    """

    def _trainer(self, device_state=False, steps=2):
        x, t = _data(n=8)
        g = _graph(batch=8)
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        backend = _FakeDeviceBackend() if device_state else None
        cls = _FakeDeviceTrainer if device_state else Trainer
        trainer = cls(
            g,
            store,
            SGD(store.arrays(), lr=0.1),
            backend=backend,
            device_state=device_state,
        )
        for _ in range(steps):
            trainer.step({"x": x, "t": t})
        return trainer, store

    def test_host_state_is_always_synced(self):
        trainer, _ = self._trainer()
        state = trainer.training_state()
        self.assertFalse(state.device)
        self.assertTrue(state.host_in_sync)
        self.assertEqual(state.steps, 2)
        self.assertEqual(state.optimizer_step, 2)

    def test_device_state_marks_the_host_copy_stale_until_sync(self):
        trainer, store = self._trainer(device_state=True)
        state = trainer.training_state()
        self.assertTrue(state.device)
        self.assertFalse(state.host_in_sync)

        trainer.sync_to_host()
        self.assertTrue(trainer.training_state().host_in_sync)
        for name in store.names:
            np.testing.assert_array_equal(
                store[name], trainer.backend.to_host(trainer._dev_params[name])
            )

    def test_verify_device_state_passes_when_everything_is_one_version(self):
        trainer, _ = self._trainer(device_state=True)
        report = trainer.verify_device_state()
        self.assertEqual(report["plans"], ["fwd", "bwd"])
        self.assertEqual(report["params"], 2)
        self.assertEqual(report["steps"], 2)
        self.assertEqual(report["optimizer_step"], 2)

    def test_verify_device_state_catches_a_detached_optimizer(self):
        """换一份"名字形状都对、地址全错"的参数表：训练会跑，参数不动。"""
        trainer, store = self._trainer(device_state=True)
        trainer.optimizer.rebind(
            {name: np.array(trainer._dev_params[name], copy=True) for name in store.names}
        )
        with self.assertRaises(RuntimeError) as caught:
            trainer.verify_device_state()
        self.assertIn("优化器手里的", str(caught.exception))

    def test_verify_device_state_catches_a_replaced_capture_buffer(self):
        """捕获计划里换成一份新数组：图此后读的是另一块，训练全白做。"""
        trainer, store = self._trainer(device_state=True)
        name = store.names[0]
        trainer._fwd_plan.buffers[name] = np.array(trainer._dev_params[name], copy=True)
        with self.assertRaises(RuntimeError) as caught:
            trainer.verify_device_state()
        self.assertIn(f"fwd 捕获时用的 {name}", str(caught.exception))

    def test_verify_before_the_first_step_is_rejected(self):
        g = _graph(batch=8)
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        trainer = _FakeDeviceTrainer(
            g,
            store,
            SGD(store.arrays(), lr=0.1),
            backend=_FakeDeviceBackend(),
            device_state=True,
        )
        with self.assertRaises(RuntimeError):
            trainer.verify_device_state()

    def test_step_async_matches_step_bitwise(self):
        """``step_async`` 只推迟损失的取用，不改变任何数值。"""
        x, t = _data(n=8)

        def run(async_):
            g = _graph(batch=8)
            store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
            trainer = _FakeDeviceTrainer(
                g,
                store,
                SGD(store.arrays(), lr=0.1),
                backend=_FakeDeviceBackend(),
                device_state=True,
            )
            losses = []
            for _ in range(3):
                if async_:
                    losses.append(trainer.step_async({"x": x, "t": t}).value())
                else:
                    losses.append(trainer.step({"x": x, "t": t}))
            trainer.sync_to_host()
            return np.array(losses), {name: store[name].copy() for name in store.names}

        want_loss, want_params = run(False)
        got_loss, got_params = run(True)
        np.testing.assert_array_equal(got_loss, want_loss)
        for name in want_params:
            np.testing.assert_array_equal(got_params[name], want_params[name])

    def test_step_async_handle_can_be_read_twice(self):
        """句柄的 ``value()`` 只同步一次，重复读拿到同一份结果。"""
        x, t = _data(n=8)
        g = _graph(batch=8)
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        trainer = _FakeDeviceTrainer(
            g,
            store,
            SGD(store.arrays(), lr=0.1),
            backend=_FakeDeviceBackend(),
            device_state=True,
        )
        handle = trainer.step_async({"x": x, "t": t})
        first = handle.value()
        self.assertEqual(float(handle), float(first))
        self.assertIs(handle.value(), first)

    def test_snapshot_does_not_alias_live_parameters(self):
        x, t = _data(n=8)
        trainer, store = self._trainer()
        snapshot = trainer.snapshot()
        self.assertIsInstance(snapshot, TrainingSnapshot)
        before = {name: snapshot.params[name].copy() for name in store.names}

        for _ in range(3):
            trainer.step({"x": x, "t": t})
        for name in store.names:
            np.testing.assert_array_equal(snapshot.params[name], before[name])

    def test_device_snapshot_reads_the_resident_parameters(self):
        trainer, store = self._trainer(device_state=True)
        snapshot = trainer.snapshot()
        self.assertEqual(snapshot.steps, 2)
        self.assertEqual(snapshot.step, trainer.optimizer.step_count)
        for name in store.names:
            np.testing.assert_array_equal(
                snapshot.params[name],
                trainer.backend.to_host(trainer._dev_params[name]),
            )
        # store 还是上传时的旧值，快照不能来自它
        self.assertFalse(np.array_equal(snapshot.params["w"], store["w"]))

    def test_snapshot_roundtrips_through_checkpoint(self):
        trainer, store = self._trainer()
        snapshot = trainer.snapshot(meta={"epoch": 1})
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "snap.npz")
            snapshot.save(path)

            fresh = ParameterStore.from_graph(
                _graph(batch=8), exclude=("x", "t"), seed=0
            )
            optimizer = SGD(fresh.arrays(), lr=0.1)
            meta = load_checkpoint(path, fresh, optimizer)

            self.assertEqual(meta["epoch"], 1)
            self.assertEqual(meta["trainer_steps"], trainer.steps)
            self.assertEqual(optimizer.step_count, trainer.optimizer.step_count)
            for name in store.names:
                np.testing.assert_array_equal(fresh[name], snapshot.params[name])

    def test_async_checkpoint_writes_without_alias(self):
        trainer, store = self._trainer()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "async.npz")
            future = trainer.save_checkpoint(path, blocking=False)
            future.result(timeout=30)
            self.assertTrue(os.path.exists(path))

            fresh = ParameterStore.from_graph(
                _graph(batch=8), exclude=("x", "t"), seed=0
            )
            load_checkpoint(path, fresh, SGD(fresh.arrays(), lr=0.1))
            for name in store.names:
                np.testing.assert_array_equal(fresh[name], store[name])

    def test_async_checkpoint_reports_failure_on_the_future(self):
        """写到不存在的目录：异常落在 Future 上，不该打断主线程的训练。"""
        trainer, _ = self._trainer()
        future = trainer.save_checkpoint(
            os.path.join("no_such_dir", "x.npz"), blocking=False
        )
        with self.assertRaises(OSError):
            future.result(timeout=30)

    def test_snapshot_separates_trainer_steps_from_optimizer_steps(self):
        """``grad_hook`` 返回 None 的那些步只前进、不更新，两个计数必须分开。"""
        x, t = _data(n=8)
        g = _graph(batch=8)
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        trainer = Trainer(g, store, SGD(store.arrays(), lr=0.1))
        trainer.grad_hook = lambda grads: None
        trainer.step({"x": x, "t": t})
        trainer.step({"x": x, "t": t})

        snapshot = trainer.snapshot()
        self.assertEqual(snapshot.steps, 2)
        self.assertEqual(snapshot.step, 0)
        self.assertEqual(trainer.training_state().optimizer_step, 0)


class TestDeviceStateMasterWeights(unittest.TestCase):
    """设备态 master weights：优化器写 f32 主副本，f16 驻留每步被覆盖。

    替身把 numpy 当显存：判据不是"真的省了多少显存"，而是三处契约——
    捕获图读 f16 那块、优化器持 f32 那块、每步更新后前者等于后者的 cast。
    """

    def _trainer(self, steps=3):
        x, t = _data(n=8)
        g = _graph(batch=8)
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        trainer = _FakeDeviceTrainer(
            g,
            store,
            SGD(store.arrays(), lr=0.05),
            backend=_FakeDeviceBackend(),
            device_state=True,
            mixed_precision="f16",
            master_weights=True,
        )
        losses = [trainer.step({"x": x, "t": t}) for _ in range(steps)]
        return trainer, store, losses

    def test_capture_reads_f16_and_optimizer_writes_f32(self):
        trainer, store, _ = self._trainer()
        for name in store.names:
            self.assertEqual(trainer._dev_params[name].dtype, np.float16)
            self.assertEqual(trainer._dev_master[name].dtype, np.float32)
            self.assertIs(trainer.optimizer.params[name], trainer._dev_master[name])
            self.assertIs(trainer._fwd_plan.buffers[name], trainer._dev_params[name])

    def test_each_update_is_written_back_to_the_resident_copy(self):
        trainer, store, _ = self._trainer()
        for name in store.names:
            np.testing.assert_array_equal(
                trainer._dev_params[name],
                trainer._dev_master[name].astype(np.float16),
            )
        self.assertEqual(trainer.verify_device_state()["params"], 2)

    def test_loss_decreases_and_sync_writes_the_master(self):
        trainer, store, losses = self._trainer(steps=20)
        self.assertLess(losses[-1], losses[0])
        trainer.sync_to_host()
        for name in store.names:
            np.testing.assert_array_equal(store[name], trainer._dev_master[name])


class TestTrainerSchedule(unittest.TestCase):
    """调度器接进循环后，"第几步"必须指真实更新次数。"""

    def _trainer(self, **kwargs):
        g = _graph(batch=8)
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        return Trainer(g, store, SGD(store.arrays(), lr=0.1), **kwargs), store

    def test_lr_is_written_before_each_update(self):
        x, t = _data(n=8)
        # step_size=2 的阶梯：第 0/1 次更新用 1.0，第 2/3 次用 0.5
        trainer, _ = self._trainer(
            schedule=StepDecay(base_lr=1.0, step_size=2, gamma=0.5)
        )
        seen = []
        for _ in range(4):
            trainer.step({"x": x, "t": t})
            seen.append(trainer.optimizer.lr)

        # 构造时的 0.1 在第 1 次更新前就被调度器覆盖了
        self.assertEqual(seen, [1.0, 1.0, 0.5, 0.5])

    def test_accumulation_window_does_not_burn_warmup_steps(self):
        x, t = _data(n=8)
        trainer, _ = self._trainer(
            accumulate_steps=2,
            schedule=StepDecay(base_lr=1.0, step_size=1, gamma=0.5),
        )
        seen = []
        for _ in range(4):
            trainer.step({"x": x, "t": t})
            seen.append(trainer.optimizer.lr)

        # 窗口内的第 1、3 步不更新，lr 停在上一轮的值；两次更新各取一步
        self.assertEqual(seen, [0.1, 1.0, 1.0, 0.5])
        self.assertEqual(trainer.optimizer.step_count, 2)

    def test_scheduled_lr_resumes_from_the_checkpoint(self):
        """带调度器时存档里的 lr 是当前位置，必须恢复而不是拿构造值（F06）。"""
        x, t = _data(n=8)
        schedule = StepDecay(base_lr=1.0, step_size=2, gamma=0.5)
        trainer, store = self._trainer(schedule=schedule)
        for _ in range(3):
            trainer.step({"x": x, "t": t})
        self.assertEqual(trainer.optimizer.lr, 0.5)

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "sched.npz")
            trainer.save_checkpoint(path)

            g = _graph(batch=8)
            fresh = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
            resumed = Trainer(
                g,
                fresh,
                SGD(fresh.arrays(), lr=0.1),
                schedule=StepDecay(base_lr=1.0, step_size=2, gamma=0.5),
            )
            resumed.load_training_state(path)
            self.assertEqual(resumed.optimizer.lr, 0.5)
            self.assertEqual(resumed.optimizer.step_count, 3)

            trainer.step({"x": x, "t": t})
            resumed.step({"x": x, "t": t})

        for name in store.names:
            np.testing.assert_allclose(
                fresh[name], store[name], rtol=1e-6, atol=1e-7
            )


class TestTrainerAccumulation(unittest.TestCase):
    """梯度累积的判据是"与等效大 batch 走一步逐位一致"，不是"步数对"。"""

    def test_four_micro_batches_equal_one_big_batch(self):
        x, t = _data(n=8)

        big = _graph(batch=8)
        big_store = ParameterStore.from_graph(big, exclude=("x", "t"), seed=0)
        Trainer(big, big_store, SGD(big_store.arrays(), lr=0.1)).step({"x": x, "t": t})

        small = _graph(batch=2)
        small_store = ParameterStore.from_graph(small, exclude=("x", "t"), seed=0)
        trainer = Trainer(
            small,
            small_store,
            SGD(small_store.arrays(), lr=0.1),
            accumulate_steps=4,
        )
        for start in (0, 2, 4, 6):
            trainer.step({"x": x[start : start + 2], "t": t[start : start + 2]})

        self.assertEqual(trainer.steps, 4)
        self.assertEqual(trainer.optimizer.step_count, 1)
        for name in big_store.names:
            np.testing.assert_allclose(
                small_store[name], big_store[name], rtol=1e-5, atol=1e-6
            )

    def test_an_incomplete_window_leaves_the_parameters_alone(self):
        x, t = _data(n=8)
        g = _graph(batch=2)
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        before = {name: store[name].copy() for name in store.names}
        trainer = Trainer(
            g, store, SGD(store.arrays(), lr=0.1), accumulate_steps=4
        )

        for start in (0, 2):
            trainer.step({"x": x[start : start + 2], "t": t[start : start + 2]})

        self.assertEqual(trainer.steps, 2)
        self.assertEqual(trainer.optimizer.step_count, 0)
        for name in store.names:
            np.testing.assert_array_equal(store[name], before[name])

    def test_resume_restores_the_pending_window(self):
        """审计 P0：中途存档必须带上窗口里**尚未提交**的梯度。

        判据：A 一口气跑完 4 个微批；B 跑 2 个后存档、重建、加载，再跑
        后 2 个。两者的参数与优化器步数逐位一致。丢掉窗口的话，B 的第二
        次更新只累加了后两批——等效 batch 悄悄减半。
        """
        x, t = _data(n=8)
        g = _graph(batch=2)
        store_a = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        trainer_a = Trainer(
            g, store_a, SGD(store_a.arrays(), lr=0.1), accumulate_steps=4
        )
        for start in (0, 2, 4, 6):
            trainer_a.step({"x": x[start : start + 2], "t": t[start : start + 2]})

        store_b = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        trainer_b = Trainer(
            g, store_b, SGD(store_b.arrays(), lr=0.1), accumulate_steps=4
        )
        for start in (0, 2):
            trainer_b.step({"x": x[start : start + 2], "t": t[start : start + 2]})

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "window.npz")
            trainer_b.save_checkpoint(path)

            store_c = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
            trainer_c = Trainer(
                g, store_c, SGD(store_c.arrays(), lr=0.1), accumulate_steps=4
            )
            meta = trainer_c.load_training_state(path)
            self.assertEqual(int(meta["accum_pending"]), 2)
            self.assertEqual(trainer_c.steps, 2)
            self.assertEqual(trainer_c.optimizer.step_count, 0)

            for start in (4, 6):
                trainer_c.step({"x": x[start : start + 2], "t": t[start : start + 2]})

        self.assertEqual(trainer_c.optimizer.step_count, 1)
        for name in store_a.names:
            np.testing.assert_allclose(
                store_c[name], store_a[name], rtol=1e-6, atol=1e-7
            )

    def test_a_full_window_resumes_without_pending_gradients(self):
        """满窗存档时没有未提交梯度，加载后不应凭空多出累加和。"""
        x, t = _data(n=8)
        g = _graph(batch=2)
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        trainer = Trainer(
            g, store, SGD(store.arrays(), lr=0.1), accumulate_steps=2
        )
        for start in (0, 2):
            trainer.step({"x": x[start : start + 2], "t": t[start : start + 2]})

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "full.npz")
            trainer.save_checkpoint(path)
            fresh = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
            resumed = Trainer(
                g, fresh, SGD(fresh.arrays(), lr=0.1), accumulate_steps=2
            )
            meta = resumed.load_training_state(path)
        self.assertEqual(int(meta["accum_pending"]), 0)
        self.assertEqual(resumed.optimizer.step_count, 1)

    def test_optimizer_hyperparameters_are_bound_to_the_checkpoint(self):
        """换了 lr 再加载旧存档，轨迹就不是续训了——必须报错而不是照跑。"""
        x, t = _data(n=8)
        g = _graph(batch=2)
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        trainer = Trainer(g, store, SGD(store.arrays(), lr=0.1))
        trainer.step({"x": x[:2], "t": t[:2]})

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "hyper.npz")
            trainer.save_checkpoint(path)
            fresh = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
            other = Trainer(g, fresh, SGD(fresh.arrays(), lr=0.05))
            with self.assertRaisesRegex(ValueError, "超参"):
                other.load_training_state(path)

    def test_window_length_and_rng_convention_are_validated(self):
        x, t = _data(n=8)
        g = _graph(batch=2)
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        trainer = Trainer(
            g, store, SGD(store.arrays(), lr=0.1), accumulate_steps=2
        )
        trainer.step({"x": x[:2], "t": t[:2]})

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "conv.npz")
            trainer.save_checkpoint(path)

            fresh = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
            longer = Trainer(
                g, fresh, SGD(fresh.arrays(), lr=0.1), accumulate_steps=4
            )
            with self.assertRaisesRegex(ValueError, "累积窗口长度"):
                longer.load_training_state(path)

            fresh2 = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
            reseeded = Trainer(
                g, fresh2, SGD(fresh2.arrays(), lr=0.1),
                accumulate_steps=2, rng_seed=7,
            )
            with self.assertRaisesRegex(ValueError, "rng_seed"):
                reseeded.load_training_state(path)

    def test_failed_validation_leaves_parameters_untouched(self):
        """配置校验必须先于提交（F05）。

        先更新过一个窗口、再留下 1 个未提交微批存档；拿窗口长度不同的训练器
        去加载必须报错，且**参数与优化器步数都不能变**——否则现场是"权重已
        换成存档、累积窗口还是旧的"，续训的动量与梯度对不上。
        """
        x, t = _data(n=8)
        g = _graph(batch=2)
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        trainer = Trainer(
            g, store, SGD(store.arrays(), lr=0.1), accumulate_steps=2
        )
        for start in (0, 2, 4):
            trainer.step({"x": x[start : start + 2], "t": t[start : start + 2]})
        self.assertEqual(trainer.optimizer.step_count, 1)

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "pending.npz")
            trainer.save_checkpoint(path)

            fresh = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
            before = {name: fresh[name].copy() for name in fresh.names}
            longer = Trainer(
                g, fresh, SGD(fresh.arrays(), lr=0.1), accumulate_steps=4
            )
            with self.assertRaisesRegex(ValueError, "累积窗口长度"):
                longer.load_training_state(path)
            self.assertEqual(longer.optimizer.step_count, 0)
            self.assertEqual(longer.steps, 0)
            for name in fresh.names:
                np.testing.assert_array_equal(fresh[name], before[name])

    def test_inconsistent_window_metadata_is_rejected_before_committing(self):
        """缺 ``accum_weight`` 或累加和键对不上时，先报错、不落任何写入。"""
        x, t = _data(n=8)
        g = _graph(batch=2)
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        trainer = Trainer(
            g, store, SGD(store.arrays(), lr=0.1), accumulate_steps=2
        )
        trainer.step({"x": x[:2], "t": t[:2]})  # 窗口 1/2

        with tempfile.TemporaryDirectory() as folder:
            missing = os.path.join(folder, "missing.npz")
            snapshot = trainer.snapshot()
            snapshot.meta.pop("accum_weight")
            snapshot.save(missing)

            fresh = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
            before = {name: fresh[name].copy() for name in fresh.names}
            resumed = Trainer(
                g, fresh, SGD(fresh.arrays(), lr=0.1), accumulate_steps=2
            )
            with self.assertRaisesRegex(KeyError, "accum_weight"):
                resumed.load_training_state(missing)
            for name in fresh.names:
                np.testing.assert_array_equal(fresh[name], before[name])

            bogus = os.path.join(folder, "bogus.npz")
            snapshot = trainer.snapshot()
            snapshot.aux["grad/not_a_param"] = np.zeros(1, np.float32)
            snapshot.save(bogus)
            with self.assertRaisesRegex(KeyError, "参数集合不一致"):
                resumed.load_training_state(bogus)
            for name in fresh.names:
                np.testing.assert_array_equal(fresh[name], before[name])

    def test_hook_runs_once_per_window(self):
        """钩子在 flush 之后、更新之前：窗口内看不到梯度，满窗才看到一次。"""
        x, t = _data(n=8)
        g = _graph(batch=2)
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        calls = []

        def hook(grads):
            calls.append(sorted(grads))
            return grads

        trainer = Trainer(
            g,
            store,
            SGD(store.arrays(), lr=0.1),
            accumulate_steps=3,
            grad_hook=hook,
        )
        for start in (0, 2, 4, 6):
            trainer.step({"x": x[start : start + 2], "t": t[start : start + 2]})

        # 4 个微批、窗口 3：第 3 步满窗更新一次，第 4 步开始新窗口
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], sorted(store.names))
        self.assertEqual(trainer.optimizer.step_count, 1)


class TestWeightedAccumulation(unittest.TestCase):
    """审计 P1：累积按微批等权，变长样本会让"每 token"的梯度偏心率变化。

    ``weight`` 必须是这一批的有效 token 数；窗口平均于是精确等于按总有效
    token 的均值。这里用线性回归的解析梯度验收，不依赖反向图。
    """

    def _analytic(self, x, t, w, b):
        n = x.shape[0]
        residual = x @ w + b - t
        return (
            (2.0 / n) * (x.T @ residual),  # dw，形状 (4, 1)
            (2.0 / n) * residual.sum(axis=0),  # db，形状 (1,)
        )

    def test_weights_produce_a_token_weighted_mean_gradient(self):
        x, t = _data(n=8)
        g = _graph(batch=4)
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        seen = {}

        def capture(grads):
            seen.update({name: np.array(value) for name, value in grads.items()})
            return grads

        wa, wb = 3.0, 5.0
        w0 = store["w"].copy()
        b0 = store["b"].copy()
        trainer = Trainer(
            g,
            store,
            SGD(store.arrays(), lr=0.1),  # 钩子在更新之前，参数改动不影响比较
            accumulate_steps=2,
            grad_hook=capture,
        )
        trainer.step({"x": x[:4], "t": t[:4]}, weight=wa)
        trainer.step({"x": x[4:], "t": t[4:]}, weight=wb)

        dw_a, db_a = self._analytic(x[:4], t[:4], w0, b0)
        dw_b, db_b = self._analytic(x[4:], t[4:], w0, b0)
        np.testing.assert_allclose(
            seen["w"], (wa * dw_a + wb * dw_b) / (wa + wb), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            seen["b"], (wa * db_a + wb * db_b) / (wa + wb), rtol=1e-5, atol=1e-6
        )

    def test_default_weight_is_the_old_micro_batch_mean(self):
        x, t = _data(n=8)
        g = _graph(batch=4)
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        seen = {}

        def capture(grads):
            seen.update({name: np.array(value) for name, value in grads.items()})
            return grads

        w0, b0 = store["w"].copy(), store["b"].copy()
        trainer = Trainer(
            g, store, SGD(store.arrays(), lr=0.1),
            accumulate_steps=2, grad_hook=capture,
        )
        trainer.step({"x": x[:4], "t": t[:4]})
        trainer.step({"x": x[4:], "t": t[4:]})

        dw_a, db_a = self._analytic(x[:4], t[:4], w0, b0)
        dw_b, db_b = self._analytic(x[4:], t[4:], w0, b0)
        np.testing.assert_allclose(seen["w"], (dw_a + dw_b) / 2.0, rtol=1e-5)
        np.testing.assert_allclose(seen["b"], (db_a + db_b) / 2.0, rtol=1e-5)

    def test_invalid_weights_are_rejected_before_accumulating(self):
        acc = GradientAccumulator(2)
        grads = {"w": np.zeros((4, 1), np.float32)}
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "有限正数"):
                acc.add(grads, bad)
        self.assertEqual(acc.pending, 0)

    def test_discard_resets_the_weight(self):
        """作废窗口必须连权重一起清零（F03）。

        只清累加和与计数、留下旧的权重和，下一个窗口的 flush 会除以一个
        包含了已作废微批的分母，梯度整体被缩小。
        """
        acc = GradientAccumulator(2)
        grads = {"w": np.ones((2, 1), np.float32)}

        acc.add(grads, 3.0)
        self.assertEqual(acc.weight, 3.0)
        self.assertEqual(acc.discard(), 1)
        self.assertEqual(acc.pending, 0)
        self.assertEqual(acc.weight, 0.0)

        # 新窗口只该按自己的 2 + 2 归一，不该带着被丢掉的 3
        acc.add(grads, 2.0)
        acc.add(grads, 2.0)
        np.testing.assert_allclose(acc.flush()["w"], np.ones((2, 1), np.float32))

    def test_weighted_window_survives_the_checkpoint(self):
        """存档必须带上窗口的权重和，否则续训的分母变成微批数（F04）。

        窗口里两批权重 3 与 1：存档前只有第一批，恢复后补上第二批，flush
        的梯度必须是 ``(3*dw_a + 1*dw_b) / 4``，不是 ``(dw_a + dw_b) / 2``。
        """
        x, t = _data(n=8)
        g = _graph(batch=4)
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        seen = {}

        def capture(grads):
            seen.update({name: np.array(value) for name, value in grads.items()})
            return grads

        w0, b0 = store["w"].copy(), store["b"].copy()
        trainer = Trainer(
            g, store, SGD(store.arrays(), lr=0.1),
            accumulate_steps=2, grad_hook=capture,
        )
        trainer.step({"x": x[:4], "t": t[:4]}, weight=3.0)

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "weighted.npz")
            trainer.save_checkpoint(path)
            fresh = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
            resumed = Trainer(
                g, fresh, SGD(fresh.arrays(), lr=0.1),
                accumulate_steps=2, grad_hook=capture,
            )
            meta = resumed.load_training_state(path)
            self.assertEqual(float(meta["accum_weight"]), 3.0)
            self.assertEqual(resumed._accumulator.weight, 3.0)
            resumed.step({"x": x[4:], "t": t[4:]}, weight=1.0)

        dw_a, db_a = self._analytic(x[:4], t[:4], w0, b0)
        dw_b, db_b = self._analytic(x[4:], t[4:], w0, b0)
        np.testing.assert_allclose(
            seen["w"], (3.0 * dw_a + 1.0 * dw_b) / 4.0, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            seen["b"], (3.0 * db_a + 1.0 * db_b) / 4.0, rtol=1e-5, atol=1e-6
        )


class TestTrainerAmp(unittest.TestCase):
    """GradScaler 接进循环：种子随 scale 变，更新量与不缩放时相同。"""

    def _pair(self, batch=8, **kwargs):
        g = _graph(batch=batch)
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        return g, store, Trainer(g, store, SGD(store.arrays(), lr=0.1), **kwargs)

    def test_scaling_does_not_change_the_update(self):
        """scale 乘在种子上、又原样除掉，参数轨迹必须与不缩放逐位一致。"""
        x, t = _data(n=8)
        _, plain_store, plain = self._pair()
        _, scaled_store, scaled = self._pair(
            scaler=GradScaler(init_scale=4.0, growth_interval=100)
        )
        for _ in range(3):
            a = plain.step({"x": x, "t": t})
            b = scaled.step({"x": x, "t": t})
            self.assertAlmostEqual(a, b, places=6)

        for name in plain_store.names:
            np.testing.assert_allclose(
                scaled_store[name], plain_store[name], rtol=1e-6, atol=1e-7
            )
        self.assertEqual(scaled.scaler.skipped, 0)
        self.assertEqual(scaled.scaler.scale, 4.0)

    def test_nonfinite_gradients_skip_the_update_and_halve_the_scale(self):
        x, t = _data(n=8)
        _, store, trainer = self._pair(
            scaler=GradScaler(init_scale=8.0, growth_interval=100)
        )
        before = {name: store[name].copy() for name in store.names}
        trainer.grad_hook = lambda grads: {
            name: np.full_like(grad, np.inf) for name, grad in grads.items()
        }

        trainer.step({"x": x, "t": t})

        self.assertEqual(trainer.optimizer.step_count, 0)
        self.assertEqual(trainer.scaler.skipped, 1)
        self.assertEqual(trainer.scaler.scale, 4.0)
        for name in store.names:
            np.testing.assert_array_equal(store[name], before[name])

        # 去掉坏梯度之后同一步能正常更新，scale 不会继续退
        trainer.grad_hook = None
        trainer.step({"x": x, "t": t})
        self.assertEqual(trainer.optimizer.step_count, 1)
        self.assertEqual(trainer.scaler.scale, 4.0)

    def test_overflow_inside_a_window_discards_the_whole_window(self):
        """被污染的累加和没法只减掉出错那一批：整窗作废，参数不动。"""
        x, t = _data(n=8)
        _, store, trainer = self._pair(
            batch=2,
            accumulate_steps=2,
            scaler=GradScaler(init_scale=8.0, growth_interval=100),
        )
        before = {name: store[name].copy() for name in store.names}
        scaler = trainer.scaler
        bad = {
            name: np.full(store[name].shape, np.inf, dtype=np.float32)
            for name in store.names
        }

        # 直接喂内部入口：图本身很难稳定地产生 inf 梯度，而这条路径判的是
        # "某一微批溢出后窗口怎么办"，与图无关。
        good = {name: np.ones(store[name].shape, dtype=np.float32) for name in store.names}
        self.assertIsNone(trainer._accumulate(good))  # 窗口 1/2
        self.assertEqual(trainer._accumulator.pending, 1)

        self.assertIsNone(trainer._accumulate(bad))
        self.assertEqual(scaler.skipped, 1)
        self.assertEqual(scaler.scale, 4.0)
        # 已经攒下的那一批也必须一起作废，否则下一批会和它"平均"出坏梯度
        self.assertEqual(trainer._accumulator.pending, 0)

        self.assertIsNone(trainer._accumulate(good))  # 新窗口 1/2
        self.assertEqual(trainer._accumulator.pending, 1)
        self.assertEqual(trainer.optimizer.step_count, 0)
        for name in store.names:
            np.testing.assert_array_equal(store[name], before[name])

    def test_scaler_state_survives_the_checkpoint(self):
        x, t = _data(n=8)
        _, _, trainer = self._pair(
            scaler=GradScaler(init_scale=8.0, growth_interval=3)
        )
        # 两步正常更新：增长计数走到 2，还没到增长点
        trainer.step({"x": x, "t": t})
        trainer.step({"x": x, "t": t})
        self.assertEqual(trainer.scaler.state_dict()["growth"], 2)
        snapshot = trainer.snapshot()

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "amp.npz")
            snapshot.save(path)
            fresh = ParameterStore.from_graph(
                _graph(batch=8), exclude=("x", "t"), seed=0
            )
            meta = load_checkpoint(path, fresh, SGD(fresh.arrays(), lr=0.1))

        self.assertIn("amp_scale", meta)
        restored = GradScaler(init_scale=1.0)
        restored.load_state_dict(
            {key: meta[f"amp_{key}"] for key in ("scale", "growth", "skipped")}
        )
        self.assertEqual(restored.scale, 8.0)
        self.assertEqual(restored.state_dict()["growth"], 2)

    def test_load_amp_state_needs_a_scaler(self):
        _, _, trainer = self._pair()
        with self.assertRaises(ValueError):
            trainer.load_amp_state({"amp_scale": 1.0})

    def test_grad_hook_sees_unscaled_gradients(self):
        """钩子必须在反缩放之后运行（F02）。

        否则裁剪的 ``max_norm`` 量的是 scale 倍的梯度，scale 变大时同一个
        ``max_norm`` 会突然开始"裁"，训练轨迹随 AMP 的缩放历史漂移。
        """
        x, t = _data(n=8)
        plain_seen, scaled_seen = {}, {}
        _, _, plain = self._pair(
            grad_hook=lambda grads: plain_seen.update(
                {k: np.array(v) for k, v in grads.items()}
            )
            or grads
        )
        _, _, scaled = self._pair(
            scaler=GradScaler(init_scale=8.0, growth_interval=100),
            grad_hook=lambda grads: scaled_seen.update(
                {k: np.array(v) for k, v in grads.items()}
            )
            or grads,
        )
        plain.step({"x": x, "t": t})
        scaled.step({"x": x, "t": t})

        for name in plain_seen:
            np.testing.assert_allclose(
                scaled_seen[name], plain_seen[name], rtol=1e-6, atol=1e-7
            )

    def test_overflow_does_not_advance_the_schedule(self):
        """溢出跳过的步不能消耗一次调度（F02）。

        调度用的是"真实更新次数"，若在溢出判定之前就 ``apply_lr``，被跳过
        的那一步会白白把 lr 推一格，续训的曲线与未溢出时对不上。
        """
        _, _, trainer = self._pair(
            scaler=GradScaler(init_scale=8.0, growth_interval=100),
            schedule=StepDecay(base_lr=1.0, step_size=1, gamma=0.5),
        )
        before = trainer.optimizer.lr
        bad = {
            name: np.full(shape.shape, np.inf, dtype=np.float32)
            for name, shape in trainer.store.arrays().items()
        }
        self.assertFalse(trainer._apply(bad, already_unscaled=False))
        self.assertEqual(trainer.optimizer.lr, before)
        self.assertEqual(trainer.optimizer.step_count, 0)
        self.assertEqual(trainer.scaler.skipped, 1)
        self.assertEqual(trainer.scaler.scale, 4.0)

    def test_amp_mismatch_fails_before_committing(self):
        """不带 scaler 的训练器加载 AMP 存档：报错且参数不动（F05）。"""
        x, t = _data(n=8)
        _, _, trainer = self._pair(
            scaler=GradScaler(init_scale=4.0, growth_interval=100)
        )
        trainer.step({"x": x, "t": t})

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "amp.npz")
            trainer.save_checkpoint(path)
            fresh = ParameterStore.from_graph(
                _graph(batch=8), exclude=("x", "t"), seed=0
            )
            before = {name: fresh[name].copy() for name in fresh.names}
            plain = Trainer(_graph(batch=8), fresh, SGD(fresh.arrays(), lr=0.1))
            with self.assertRaisesRegex(ValueError, "GradScaler"):
                plain.load_training_state(path)
            self.assertEqual(plain.optimizer.step_count, 0)
            for name in fresh.names:
                np.testing.assert_array_equal(fresh[name], before[name])

    def test_load_amp_state_rejects_a_checkpoint_without_scaling(self):
        _, _, trainer = self._pair(scaler=GradScaler())
        with self.assertRaises(KeyError):
            trainer.load_amp_state({"trainer_steps": 3})


if __name__ == "__main__":
    unittest.main()
