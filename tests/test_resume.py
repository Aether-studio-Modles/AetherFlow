"""续训的两块契约：数据迭代位置与训练配置凭证。

审计指出"完整续训"缺口：权重、优化器、AMP、随机约定都能接上，但
**数据顺序**只到轮次边界，**模型图与调度器配置**没有任何凭证。这里各补一条
可判等的协议：

- ``DataLoader.state_dict`` 记录"哪一轮的第几批之后"，恢复后从同一批继续；
  配置（seed / batch_size / size / shuffle / drop_last）变了就拒绝——同一个
  游标在另一套配置下指向的是另一批样本。
- ``Trainer.snapshot`` 记录前向图指纹与调度器类型 / 超参，加载前校验。
  参数名与形状相同、图或衰减曲线不同，仍然是另一条训练轨迹。
"""

import os
import tempfile
import unittest

import numpy as np

from aetherflow import Graph, ReferenceBackend, f32
from aetherflow.train import (
    Adam,
    CosineDecay,
    DataLoader,
    ParameterStore,
    StepDecay,
    Trainer,
    Warmup,
    mse,
)

BACKEND = ReferenceBackend()


def _data(n=10):
    return {"x": np.arange(n * 2, dtype=np.float32).reshape(n, 2)}


def _loss_graph(kind="tanh"):
    g = Graph("fit")
    x = g.input("x", f32[4, 2])
    t = g.input("t", f32[4, 1])
    w = g.input("w", f32[2, 1])
    b = g.input("b", f32[1, 1])
    pred = g.add(g.matmul(x, w), b)
    pred = g.add_node(kind, [pred])
    g.output(mse(g, pred, t), "loss")
    return g


def _trainer(kind="tanh", schedule=None):
    g = _loss_graph(kind)
    store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
    trainer = Trainer(g, store, Adam(store.arrays(), lr=0.01), schedule=schedule)
    return trainer, store


def _batch():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 2)).astype(np.float32)
    t = rng.normal(size=(4, 1)).astype(np.float32)
    return {"x": x, "t": t}


class TestDataLoaderResume(unittest.TestCase):
    def test_state_dict_resumes_mid_epoch(self):
        reference = list(DataLoader(_data(), 3, shuffle=True, seed=7))
        self.assertEqual(len(reference), 4)

        loader = DataLoader(_data(), 3, shuffle=True, seed=7)
        it = iter(loader)
        for _ in range(2):
            next(it)
        state = loader.state_dict()
        self.assertEqual(state["epoch"], 1)
        self.assertEqual(state["batch"], 2)

        restored = DataLoader(_data(), 3, shuffle=True, seed=7)
        restored.load_state_dict(state)
        tail = list(restored)
        self.assertEqual(len(tail), 2)
        for got, want in zip(tail, reference[2:]):
            np.testing.assert_array_equal(got["x"], want["x"])

    def test_resume_keeps_the_next_epoch_order(self):
        plain = DataLoader(_data(), 3, shuffle=True, seed=7)
        first = list(plain)
        second = list(plain)

        loader = DataLoader(_data(), 3, shuffle=True, seed=7)
        it = iter(loader)
        for _ in range(2):
            next(it)
        restored = DataLoader(_data(), 3, shuffle=True, seed=7)
        restored.load_state_dict(loader.state_dict())
        tail = list(restored)
        for got, want in zip(tail, first[2:]):
            np.testing.assert_array_equal(got["x"], want["x"])
        for got, want in zip(list(restored), second):
            np.testing.assert_array_equal(got["x"], want["x"])

    def test_complete_epoch_returns_to_the_boundary(self):
        loader = DataLoader(_data(), 3)
        list(loader)
        self.assertEqual(loader.cursor, 0)
        self.assertEqual(loader.epoch, 1)
        state = loader.state_dict()
        self.assertEqual((state["epoch"], state["batch"]), (1, 0))

    def test_partial_iteration_keeps_the_cursor(self):
        loader = DataLoader(_data(), 3, shuffle=True, seed=1)
        it = iter(loader)
        next(it)
        # 提前放弃迭代器：游标停在"这一批已经吃过"，而不是回到轮首
        self.assertEqual(loader.cursor, 1)
        self.assertEqual(loader.state_dict()["batch"], 1)

    def test_config_mismatch_is_rejected_before_any_change(self):
        loader = DataLoader(_data(), 3, shuffle=True, seed=7)
        state = loader.state_dict()
        other = DataLoader(_data(), 3, shuffle=True, seed=8)
        with self.assertRaisesRegex(ValueError, "seed"):
            other.load_state_dict(state)
        self.assertEqual((other.epoch, other.cursor), (0, 0))

        bigger = DataLoader(_data(), 5, shuffle=True, seed=7)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            bigger.load_state_dict(state)

    def test_missing_keys_and_bad_positions_are_rejected(self):
        loader = DataLoader(_data(), 3)
        state = loader.state_dict()
        state.pop("batch")
        with self.assertRaisesRegex(KeyError, "batch"):
            loader.load_state_dict(state)

        good = DataLoader(_data(), 3).state_dict()
        good["epoch"] = 1
        good["batch"] = len(loader) + 1
        with self.assertRaisesRegex(ValueError, "batch"):
            loader.load_state_dict(good)
        good["batch"] = 1
        good["epoch"] = 0
        with self.assertRaisesRegex(ValueError, "epoch"):
            loader.load_state_dict(good)


class TestResumeCredentials(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def _path(self, name="ckpt.npz"):
        return os.path.join(self.dir.name, name)

    def test_fingerprint_tracks_the_graph_not_just_params(self):
        first, _ = _trainer("tanh")
        second, _ = _trainer("tanh")
        other, _ = _trainer("sigmoid")
        self.assertEqual(first.graph_fingerprint(), second.graph_fingerprint())
        self.assertNotEqual(first.graph_fingerprint(), other.graph_fingerprint())

    def test_roundtrip_records_credentials(self):
        trainer, _ = _trainer(schedule=StepDecay(base_lr=0.01, step_size=2))
        trainer.save_checkpoint(self._path())
        fresh, _ = _trainer(schedule=StepDecay(base_lr=0.01, step_size=2))
        meta = fresh.load_training_state(self._path())
        self.assertEqual(meta["graph_fingerprint"], trainer.graph_fingerprint())
        self.assertEqual(meta["schedule_kind"], "StepDecay")
        self.assertIn("step_size", str(meta["schedule_config"]))

    def test_graph_change_is_rejected(self):
        trainer, _ = _trainer("tanh")
        trainer.save_checkpoint(self._path())
        fresh, _ = _trainer("sigmoid")
        with self.assertRaisesRegex(ValueError, "图指纹"):
            fresh.load_training_state(self._path())

    def test_schedule_kind_change_is_rejected(self):
        trainer, _ = _trainer(schedule=StepDecay(base_lr=0.01, step_size=2))
        trainer.save_checkpoint(self._path())
        fresh, _ = _trainer(schedule=CosineDecay(base_lr=0.01, total_steps=10))
        with self.assertRaisesRegex(ValueError, "调度器是"):
            fresh.load_training_state(self._path())

    def test_schedule_hyperparams_change_is_rejected(self):
        trainer, _ = _trainer(schedule=StepDecay(base_lr=0.01, step_size=2))
        trainer.save_checkpoint(self._path())
        fresh, _ = _trainer(schedule=StepDecay(base_lr=0.01, step_size=3))
        with self.assertRaisesRegex(ValueError, "调度器配置"):
            fresh.load_training_state(self._path())

    def test_warmup_inner_schedule_is_recorded(self):
        outer = Warmup(CosineDecay(base_lr=0.01, total_steps=10), 3, 0.001)
        other = Warmup(CosineDecay(base_lr=0.01, total_steps=20), 3, 0.001)
        trainer, _ = _trainer(schedule=outer)
        trainer.save_checkpoint(self._path())
        fresh, _ = _trainer(schedule=other)
        with self.assertRaisesRegex(ValueError, "调度器配置"):
            fresh.load_training_state(self._path())

    def test_schedule_presence_change_is_rejected(self):
        trainer, _ = _trainer()
        trainer.save_checkpoint(self._path())
        fresh, _ = _trainer(schedule=StepDecay(base_lr=0.01, step_size=2))
        with self.assertRaisesRegex(ValueError, "调度器是"):
            fresh.load_training_state(self._path())

    def test_resume_reaches_the_same_trajectory(self):
        trainer, store = _trainer(schedule=StepDecay(base_lr=0.01, step_size=2))
        batch = _batch()
        for _ in range(3):
            trainer.step(batch)
        trainer.save_checkpoint(self._path())

        fresh, fresh_store = _trainer(schedule=StepDecay(base_lr=0.01, step_size=2))
        fresh.load_training_state(self._path())
        for _ in range(2):
            trainer.step(batch)
            fresh.step(batch)
        for name in store.names:
            np.testing.assert_allclose(fresh_store[name], store[name], rtol=0, atol=0)


def _pairs(n=8):
    rng = np.random.default_rng(11)
    return {
        "x": rng.normal(size=(n, 2)).astype(np.float32),
        "t": rng.normal(size=(n, 1)).astype(np.float32),
    }


class TestLoaderHandoff(unittest.TestCase):
    """训练存档与数据迭代位置必须一起走（5.225）。

    审计指出"完整续训"的最后一环：权重、优化器、AMP、图与调度器都能校验，
    但"下一批是哪批"仍要调用方手工管理。这里把 loader 交给 Trainer 的存档
    与恢复入口，位置与权重同进同出。
    """

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def _path(self, name="handoff.npz"):
        return os.path.join(self.dir.name, name)

    def test_run_loader_advances_the_loader_position(self):
        loader = DataLoader(_pairs(8), 4, shuffle=True, seed=5)
        trainer, _ = _trainer()
        losses = trainer.run_loader(loader)
        self.assertEqual(len(losses), 2)
        self.assertEqual((loader.epoch, loader.cursor), (1, 0))
        self.assertEqual(trainer.steps, 2)

    def test_checkpoint_restores_the_same_next_batch(self):
        data = _pairs(8)
        loader = DataLoader(data, 4, shuffle=True, seed=3)
        trainer, store = _trainer()
        trainer.step(next(iter(loader)))  # 吃掉第一批，游标停在第 1 批
        trainer.save_checkpoint(self._path(), loader=loader)

        fresh, fresh_store = _trainer()
        resumed = DataLoader(data, 4, shuffle=True, seed=3)
        fresh.load_training_state(self._path(), loader=resumed)
        self.assertEqual(resumed.state_dict(), loader.state_dict())
        for name in store.names:
            np.testing.assert_array_equal(fresh_store[name], store[name])
        # 中断点之后的第一批与原 loader 的下一批完全相同
        np.testing.assert_array_equal(
            next(iter(resumed))["x"], next(iter(loader))["x"]
        )

    def test_loader_config_change_is_rejected_before_any_write(self):
        data = _pairs(8)
        loader = DataLoader(data, 4, shuffle=True, seed=3)
        trainer, _ = _trainer()
        trainer.step(next(iter(loader)))
        trainer.save_checkpoint(self._path(), loader=loader)

        fresh, fresh_store = _trainer()
        before = {
            name: np.array(fresh_store[name], copy=True) for name in fresh_store.names
        }
        wrong = DataLoader(data, 2, shuffle=True, seed=3)
        with self.assertRaisesRegex(ValueError, "配置不一致"):
            fresh.load_training_state(self._path(), loader=wrong)
        # 检查失败不能留下"权重换了、位置没换"的半现场
        for name in before:
            np.testing.assert_array_equal(fresh_store[name], before[name])
        self.assertEqual(wrong.state_dict()["batch"], 0)

    def test_missing_or_extra_loader_is_rejected(self):
        trainer, _ = _trainer()
        trainer.save_checkpoint(self._path())
        fresh, _ = _trainer()
        with self.assertRaisesRegex(KeyError, "没有数据迭代位置"):
            fresh.load_training_state(self._path(), loader=DataLoader(_pairs(8), 4))

        loader = DataLoader(_pairs(8), 4)
        trainer.save_checkpoint(self._path("with.npz"), loader=loader)
        with self.assertRaisesRegex(KeyError, "没有传 loader"):
            fresh.load_training_state(self._path("with.npz"))

    def test_invalid_arguments_are_rejected(self):
        loader = DataLoader(_pairs(8), 4)
        trainer, _ = _trainer()
        with self.assertRaises(TypeError):
            trainer.save_checkpoint(self._path(), loader="loader")
        with self.assertRaisesRegex(ValueError, "互相覆盖"):
            trainer.save_checkpoint(self._path(), meta={"data": {}}, loader=loader)
        with self.assertRaises(ValueError):
            trainer.run_loader(loader, epochs=0)
        with self.assertRaises(TypeError):
            trainer.run_loader([])


if __name__ == "__main__":
    unittest.main()
