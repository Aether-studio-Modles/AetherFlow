"""CUDA 上的训练闭环：``Trainer`` + ``CudaBackend``。

本地没有 GPU 时整组跳过。这一组回答的是"能不能在 GPU 上训练一个模型"，
而不只是"每个反向算子算得对"——后者由 ``test_cuda_backend`` 的逐算子对拍
覆盖。两者缺一不可：算子在 CUDA 上对，不等于整条训练路径能跑。

为什么单独一个文件：训练路径依赖 ``train/`` 的调度（反向图构建、优化器原地
更新、dataloader 切批），把它塞进后端测试会让"后端契约"和"训练契约"混在一起。
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from aetherflow import Graph, f32
from aetherflow.backends.cuda import CUDA_AVAILABLE, CudaBackend
from aetherflow.train import Adam, DataLoader, ParameterStore, SGD, Trainer, mse


def _linear(batch=8):
    g = Graph("lin")
    x = g.input("x", f32[batch, 4])
    t = g.input("t", f32[batch, 1])
    w = g.input("w", f32[4, 1])
    b = g.input("b", f32[1])
    g.output(mse(g, g.add(g.matmul(x, w), b), t), "loss")
    return g


def _linear_data(seed=0, n=32):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    true_w = np.array([[1.0], [-2.0], [0.5], [3.0]], dtype=np.float32)
    true_b = np.array([0.7], dtype=np.float32)
    return x, (x @ true_w + true_b).astype(np.float32)


def _cnn(batch=2):
    """一个小 LeNet：两次 conv+pool 加一层全连接，覆盖卷积反向与池化反向。"""
    g = Graph("cnn_train")
    x = g.input("x", f32[batch, 1, 16, 16])
    t = g.input("t", f32[batch, 4])
    w1 = g.input("w1", f32[4, 1, 3, 3])
    b1 = g.input("b1", f32[4])
    w2 = g.input("w2", f32[8, 4, 3, 3])
    b2 = g.input("b2", f32[8])
    fc = g.input("fc", f32[2 * 2 * 8, 4])
    fcb = g.input("fcb", f32[4])

    h = g.maxpool2d(g.relu(g.conv2d(x, w1, b1)), kernel=2)
    h = g.maxpool2d(g.relu(g.conv2d(h, w2, b2)), kernel=2)
    h = g.reshape(h, shape=(batch, 2 * 2 * 8))
    g.output(mse(g, g.add(g.matmul(h, fc), fcb), t), "loss")
    return g


@unittest.skipUnless(CUDA_AVAILABLE, "需要 cupy 与 CUDA 设备")
class TestTrainOnCuda(unittest.TestCase):
    def test_linear_regression_converges(self):
        """最朴素的判据：真的在学，而不是只把梯度算对。"""
        x, t = _linear_data()
        g = _linear()
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        trainer = Trainer(
            g, store, SGD(store.arrays(), lr=0.1), backend=CudaBackend()
        )
        losses = trainer.run(DataLoader({"x": x, "t": t}, 8), epochs=50)

        self.assertLess(losses[-1], losses[0])
        self.assertLess(losses[-1], 1e-4)
        np.testing.assert_allclose(store["w"].ravel(), [1.0, -2.0, 0.5, 3.0], atol=1e-2)
        np.testing.assert_allclose(store["b"], [0.7], atol=1e-2)

    def test_cnn_step_matches_reference(self):
        """同一份初值走一步：损失与更新后的参数必须与参考后端一致。

        这一步同时把 conv2d 的输入/权重反向、池化反向、reshape 反向都走了一遍。
        """
        rng = np.random.default_rng(3)
        x = rng.normal(size=(2, 1, 16, 16)).astype(np.float32)
        t = rng.normal(size=(2, 4)).astype(np.float32)

        g = _cnn()
        results = []
        for backend in (None, CudaBackend()):
            store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=7)
            trainer = Trainer(g, store, SGD(store.arrays(), lr=0.05), backend=backend)
            loss = trainer.step({"x": x, "t": t})
            results.append((loss, {name: store[name].copy() for name in store.names}))

        (ref_loss, ref_params), (cuda_loss, cuda_params) = results
        self.assertAlmostEqual(cuda_loss, ref_loss, places=3)
        for name, expected in ref_params.items():
            np.testing.assert_allclose(cuda_params[name], expected, rtol=1e-3, atol=1e-4)

    def test_cnn_loss_decreases_over_a_few_steps(self):
        """几十步内损失要往下走；只对一步的数值一致还不够。"""
        rng = np.random.default_rng(5)
        x = rng.normal(size=(2, 1, 16, 16)).astype(np.float32)
        t = rng.normal(size=(2, 4)).astype(np.float32)

        g = _cnn()
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=1)
        trainer = Trainer(g, store, SGD(store.arrays(), lr=0.02), backend=CudaBackend())

        losses = [trainer.step({"x": x, "t": t}) for _ in range(30)]
        self.assertLess(losses[-1], losses[0])
        self.assertTrue(np.isfinite(losses).all())

    def test_auto_graph_training_matches_eager(self):
        """训练 step 也能吃 CUDA Graph：前向、反向各一张图，各自捕获一次。

        权重每步都在 numpy 里原地更新。若计划没有把当前输入重新拷进静态缓冲，
        第二步就会拿旧权重算出另一个损失——这里会立刻对不上。
        """
        x, t = _linear_data(n=8)
        g = _linear()
        runs = {}
        for mode in ("off", "auto"):
            backend = CudaBackend(graph_mode=mode)
            store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=11)
            trainer = Trainer(g, store, SGD(store.arrays(), lr=0.05), backend=backend)
            losses = [trainer.step({"x": x, "t": t}) for _ in range(10)]
            runs[mode] = (
                losses,
                {name: store[name].copy() for name in store.names},
                backend,
            )

        off_losses, off_params, _ = runs["off"]
        auto_losses, auto_params, auto_backend = runs["auto"]
        np.testing.assert_allclose(auto_losses, off_losses, rtol=1e-4, atol=1e-5)
        for name, expected in off_params.items():
            np.testing.assert_allclose(
                auto_params[name], expected, rtol=1e-4, atol=1e-5
            )
        self.assertEqual(
            len(auto_backend._graph_plans), 2, "前向与反向各应有一条捕获计划"
        )

    def test_device_state_matches_host(self):
        """参数常驻显存：10 步轨迹与主机态一致，且两张计划一次都没重捕获。

        判据分三层：损失与参数要跟主机态对齐（说明梯度链路对）；优化器手里
        必须是设备数组、主机表在 ``sync_to_host`` 之前保持初值（说明参数真的
        留在了显存）；前后向共用同一块权重缓冲（说明每步既没有 H2D 也没有
        因为权重换代而让捕获作废）。
        """
        x, t = _linear_data(n=8)
        g = _linear()

        host_store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=11)
        host_trainer = Trainer(
            g,
            host_store,
            SGD(host_store.arrays(), lr=0.05),
            backend=CudaBackend(graph_mode="off"),
        )
        host_losses = [host_trainer.step({"x": x, "t": t}) for _ in range(10)]

        init = ParameterStore.from_graph(g, exclude=("x", "t"), seed=11)
        backend = CudaBackend(graph_mode="off")
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=11)
        trainer = Trainer(
            g,
            store,
            SGD(store.arrays(), lr=0.05),
            backend=backend,
            device_state=True,
        )
        losses = [trainer.step({"x": x, "t": t}) for _ in range(10)]

        np.testing.assert_allclose(losses, host_losses, rtol=1e-4, atol=1e-5)
        for name in store.names:
            self.assertIsInstance(trainer.optimizer.params[name], backend.cp.ndarray)
            np.testing.assert_array_equal(store[name], init[name])
        self.assertIs(trainer._fwd_plan._buffers["w"], trainer.optimizer.params["w"])
        self.assertIs(trainer._bwd_plan._buffers["w"], trainer.optimizer.params["w"])

        trainer.sync_to_host()
        for name in store.names:
            np.testing.assert_allclose(store[name], host_store[name], rtol=1e-4, atol=1e-5)

    def test_device_state_adam_matches_host(self):
        """Adam 的一阶/二阶矩同样留在显存：更新轨迹必须与主机态一致。"""
        x, t = _linear_data(n=8)
        g = _linear()
        runs = {}
        for device_state in (False, True):
            store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=5)
            trainer = Trainer(
                g,
                store,
                Adam(store.arrays(), lr=0.02),
                backend=CudaBackend(graph_mode="off"),
                device_state=device_state,
            )
            losses = [trainer.step({"x": x, "t": t}) for _ in range(5)]
            trainer.sync_to_host()
            runs[device_state] = (losses, {n: store[n].copy() for n in store.names})

        host_losses, host_params = runs[False]
        dev_losses, dev_params = runs[True]
        np.testing.assert_allclose(dev_losses, host_losses, rtol=1e-4, atol=1e-5)
        for name, expected in host_params.items():
            np.testing.assert_allclose(dev_params[name], expected, rtol=1e-4, atol=1e-5)

    def test_device_state_with_segments_matches_host(self):
        """设备态叠加激活重算：逐段捕获、逐段 replay，轨迹与主机态一致。

        判据：损失与参数跟"主机态 + 同样的分段"对齐；每段各有前后向两张
        计划；边界值与 cotangent 是两组缓冲（段反向要靠边界值重放前向）。
        """
        rng = np.random.default_rng(3)
        x = rng.normal(size=(2, 1, 16, 16)).astype(np.float32)
        t = rng.normal(size=(2, 4)).astype(np.float32)
        g = _cnn(batch=2)

        runs = {}
        for device_state in (False, True):
            store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=7)
            trainer = Trainer(
                g,
                store,
                SGD(store.arrays(), lr=0.02),
                backend=CudaBackend(graph_mode="off"),
                segments=2,
                device_state=device_state,
            )
            losses = [trainer.step({"x": x, "t": t}) for _ in range(4)]
            trainer.sync_to_host()
            runs[device_state] = (losses, {n: store[n].copy() for n in store.names})

        host_losses, host_params = runs[False]
        dev_losses, dev_params = runs[True]
        np.testing.assert_allclose(dev_losses, host_losses, rtol=1e-4, atol=1e-5)
        for name, expected in host_params.items():
            np.testing.assert_allclose(dev_params[name], expected, rtol=1e-4, atol=1e-5)
        self.assertEqual(len(trainer._seg_fwd), len(trainer._plan.segments))
        self.assertEqual(len(trainer._seg_bwd), len(trainer._plan.segments))
        self.assertIsNot(trainer._dev_acts, trainer._dev_cots)

    def test_device_state_resume_restores_pending_accumulation(self):
        """设备态续训：存档里带未提交的累积窗口。

        ``load_training_state`` 在显存建立之前把累加和恢复成主机数组；续训后
        第一个微批的梯度是 cupy 数组，必须先把累加和搬到显存——不能对设备
        数组做隐式主机转换，否则第一个微批就会抛 TypeError。
        """
        x, t = _linear_data(n=8)
        data = {"x": x, "t": t}
        g = _linear()

        def new_trainer():
            store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=5)
            trainer = Trainer(
                g,
                store,
                Adam(store.arrays(), lr=0.02),
                backend=CudaBackend(graph_mode="off"),
                device_state=True,
                accumulate_steps=2,
            )
            return trainer, store

        ref, ref_store = new_trainer()
        ref_losses = [ref.step(data) for _ in range(4)]
        ref.sync_to_host()

        cut, _ = new_trainer()
        for _ in range(3):  # 窗口里剩一个未提交的微批
            cut.step(data)

        workdir = tempfile.mkdtemp(prefix="af_dev_resume_")
        path = os.path.join(workdir, "trainer.npz")
        try:
            cut.save_checkpoint(path)
            resumed, store = new_trainer()
            resumed.load_training_state(path)
            tail = resumed.step(data)  # 补齐窗口并提交一次更新
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        resumed.sync_to_host()
        self.assertEqual(resumed.optimizer.step_count, ref.optimizer.step_count)
        np.testing.assert_allclose(tail, ref_losses[3], rtol=1e-6, atol=1e-7)
        for name in store.names:
            np.testing.assert_allclose(
                store[name], ref_store[name], rtol=1e-5, atol=1e-7
            )
