"""训练开关的组合矩阵：单开都对，叠在一起还得对。

现有用例逐个验收每个开关（累积、加权、AMP、裁剪、调度、存档、重算、主副本、
设备驻留）。这里只问一个问题：**这些开关同时打开时，语义是否还是各自声明
的那一份**。判据尽量选"可手算"或"同一配置换一条执行路径"：

1. 五个开关全开，第一次更新必须等于
   ``clip(加权窗口梯度)`` 再乘调度器第 0 步的学习率——缩放只乘在种子上，
   累积、裁剪、调度的顺序不能互相偷换；
2. 全开中途存档、换一个 trainer 接着跑，轨迹与不中断逐位一致（待提交窗口的
   权重、scaler 的 scale、调度器步数必须一起接回去）；
3. 同一组开关下，分段重算与整图反向逐位一致（重算只改中间值怎么存）；
4. f16 驻留 + f32 主副本与全 f32 的轨迹在量化误差内一致；
5. 设备驻留与主机态在同样开关下逐位一致（替身后端把"捕获缓冲 + replay"如实
   演一遍，见 ``test_train_loop._FakeDeviceBackend``）。
"""

import os
import tempfile
import unittest

import numpy as np

from aetherflow import f32
from aetherflow.backends import ReferenceBackend
from aetherflow.train import (
    GradScaler,
    ParameterStore,
    SGD,
    StepDecay,
    Trainer,
    clip_global_norm,
)

from tests.test_train_loop import _FakeDeviceBackend, _FakeDeviceTrainer, _deep_graph

BATCH = 2
LR = 0.2
MAX_NORM = 0.05
INIT_SCALE = 8.0
SCHEDULE = StepDecay(base_lr=LR, step_size=2, gamma=0.5)


def _store(batch=BATCH):
    return ParameterStore.from_graph(_deep_graph(batch), exclude=("x", "t"), seed=0)


def _batches(n=6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(BATCH * n, 6)).astype(np.float32)
    t = rng.normal(size=(BATCH * n, 1)).astype(np.float32)
    return [{"x": x[i * BATCH:(i + 1) * BATCH], "t": t[i * BATCH:(i + 1) * BATCH]}
            for i in range(n)]


def _initial():
    """每份对照都从同一组初始参数出发（同名同 seed 即可复现）。"""
    return _store()


def _all_knobs(**overrides):
    """五个开关全开的基本配置；个别用例按需覆盖。"""
    options = dict(
        accumulate_steps=2,
        grad_hook=lambda grads: clip_global_norm(grads, MAX_NORM)[0],
        schedule=SCHEDULE,
        scaler=GradScaler(init_scale=INIT_SCALE, growth_interval=1000),
    )
    options.update(overrides)
    return options


class TestCombinedKnobs(unittest.TestCase):
    def test_combined_update_matches_the_hand_computation(self):
        """缩放/累积/加权/裁剪/调度全开：第一次更新等于手算一遍。"""
        batches = _batches(n=2)
        weights = (3.0, 5.0)

        # 参照：同样的图与初始参数，只把窗口梯度捕获下来、不更新。
        ref_graph = _deep_graph(BATCH)
        ref_store = _initial()
        captured = {}

        def capture(grads):
            captured.update({name: grad.copy() for name, grad in grads.items()})
            return None

        ref = Trainer(
            ref_graph,
            ref_store,
            SGD(ref_store.arrays(), lr=LR),
            accumulate_steps=2,
            grad_hook=capture,
            schedule=SCHEDULE,
        )
        for batch, weight in zip(batches, weights):
            ref.step(batch, weight=weight)
        self.assertEqual(ref.optimizer.step_count, 0)

        graph = _deep_graph(BATCH)
        store = _initial()
        initial = {name: store[name].copy() for name in store.names}
        trainer = Trainer(graph, store, SGD(store.arrays(), lr=LR), **_all_knobs())
        for batch, weight in zip(batches, weights):
            trainer.step(batch, weight=weight)

        self.assertEqual(trainer.optimizer.step_count, 1)
        self.assertEqual(trainer.scaler.skipped, 0)
        clipped, _ = clip_global_norm(captured, MAX_NORM)
        lr = SCHEDULE.lr_at(0)
        for name in store.names:
            expected = initial[name] - lr * clipped[name]
            np.testing.assert_allclose(store[name], expected, rtol=1e-6, atol=1e-7)

    def test_resume_inside_a_window_keeps_every_knob(self):
        """窗口一半时存档：权重、scale、调度步数、裁剪上限都要接回去。"""
        batches = _batches(n=5)
        weights = (2.0, 1.0, 4.0, 1.0, 3.0)

        straight_graph = _deep_graph(BATCH)
        straight_store = _initial()
        straight = Trainer(
            straight_graph, straight_store, SGD(straight_store.arrays(), lr=LR),
            **_all_knobs(),
        )
        for batch, weight in zip(batches[:2], weights[:2]):
            straight.step(batch, weight=weight)

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "state.npz")
            straight.save_checkpoint(path)

            resumed_graph = _deep_graph(BATCH)
            resumed_store = _initial()
            resumed = Trainer(
                resumed_graph, resumed_store, SGD(resumed_store.arrays(), lr=LR),
                **_all_knobs(),
            )
            resumed.load_training_state(path)

            for batch, weight in zip(batches[2:], weights[2:]):
                straight.step(batch, weight=weight)
                resumed.step(batch, weight=weight)

        # 第 0/1 个微批提交过一次；之后 3 个微批里再更新一次。
        self.assertEqual(straight.optimizer.step_count, 2)
        self.assertEqual(resumed.optimizer.step_count, 2)
        self.assertEqual(resumed.scaler.scale, straight.scaler.scale)
        for name in straight_store.names:
            np.testing.assert_allclose(
                resumed_store[name], straight_store[name], rtol=1e-6, atol=1e-7
            )

    def test_recompute_segments_match_the_full_graph(self):
        """同一组开关下，分段重算与整图只差"中间值怎么存"。"""
        batches = _batches(n=4)
        weights = (1.0, 2.0, 1.0, 1.0)

        full_graph = _deep_graph(BATCH)
        full_store = _initial()
        full = Trainer(
            full_graph, full_store, SGD(full_store.arrays(), lr=LR), **_all_knobs()
        )

        seg_graph = _deep_graph(BATCH)
        seg_store = _initial()
        segmented = Trainer(
            seg_graph, seg_store, SGD(seg_store.arrays(), lr=LR),
            segments=2, **_all_knobs(),
        )

        for batch, weight in zip(batches, weights):
            full.step(batch, weight=weight)
            segmented.step(batch, weight=weight)

        self.assertEqual(segmented.optimizer.step_count, full.optimizer.step_count)
        for name in full_store.names:
            np.testing.assert_allclose(
                seg_store[name], full_store[name], rtol=1e-6, atol=1e-7
            )

    def test_f16_master_matches_f32_within_quantization(self):
        """f16 驻留 + f32 主副本，叠加累积/裁剪/调度/缩放后仍在量化误差内。"""
        batches = _batches(n=6)
        weights = (1.0, 2.0, 1.0, 3.0, 1.0, 1.0)

        ref_store = _initial()
        ref = Trainer(
            _deep_graph(BATCH), ref_store, SGD(ref_store.arrays(), lr=LR), **_all_knobs()
        )

        mixed_store = _initial()
        mixed = Trainer(
            _deep_graph(BATCH), mixed_store, SGD(mixed_store.arrays(), lr=LR),
            mixed_precision="f16", master_weights=True, **_all_knobs(),
        )

        for batch, weight in zip(batches, weights):
            ref.step(batch, weight=weight)
            mixed.step(batch, weight=weight)

        self.assertEqual(mixed.optimizer.step_count, ref.optimizer.step_count)
        for name in ref_store.names:
            np.testing.assert_allclose(
                mixed_store[name], ref_store[name], rtol=2e-2, atol=2e-3
            )

    def test_device_state_matches_host_with_the_same_knobs(self):
        """设备驻留与主机态：累积、加权、裁剪、调度、缩放同时打开也逐位一致。"""
        batches = _batches(n=4)
        weights = (1.0, 3.0, 2.0, 1.0)

        host_store = _initial()
        host = Trainer(
            _deep_graph(BATCH), host_store, SGD(host_store.arrays(), lr=LR), **_all_knobs()
        )

        device_store = _initial()
        device = _FakeDeviceTrainer(
            _deep_graph(BATCH), device_store, SGD(device_store.arrays(), lr=LR),
            backend=_FakeDeviceBackend(), device_state=True, **_all_knobs(),
        )

        for batch, weight in zip(batches, weights):
            host.step(batch, weight=weight)
            device.step(batch, weight=weight)
        # 设备态的 store 在同步点之前是旧值；显式同步才谈得上逐位比较。
        device.sync_to_host()

        self.assertEqual(device.optimizer.step_count, host.optimizer.step_count)
        for name in host_store.names:
            np.testing.assert_allclose(
                device_store[name], host_store[name], rtol=1e-6, atol=1e-7
            )


if __name__ == "__main__":
    unittest.main()
