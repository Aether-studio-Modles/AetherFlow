"""训练态 BatchNorm 的 running stats 闭环验收。

判据有三层：EMA 数值与手写公式一致；参数没更新时统计量照样推进；训练完
取出的 ``(mean, var)`` 能直接接进推理图（``bn_infer``）算出 eval 结果，且
存档/续训能把这对状态原样接回去。
"""

import os
import tempfile
import unittest

import numpy as np

from aetherflow import Graph, f32
from aetherflow.backends import ReferenceBackend
from aetherflow.train import ParameterStore, SGD, Trainer
from aetherflow.train.bn import bn_sites, build_stats_graph

BATCH = 4
CH = 3
SIDE = 2
COUNT = BATCH * SIDE * SIDE
EPS = 1e-5


def _graph():
    g = Graph("bn")
    x = g.input("x", f32[BATCH, CH, SIDE, SIDE])
    gamma = g.input("gamma", f32[CH])
    beta = g.input("beta", f32[CH])
    y = g.batch_norm(x, gamma, beta, axis=1, eps=EPS)
    g.output(g.reduce_sum(g.mul(y, y), axis=(0, 1, 2, 3), keepdims=False), "loss")
    return g


def _store():
    return ParameterStore(
        {
            "gamma": np.ones(CH, dtype=np.float32),
            "beta": np.zeros(CH, dtype=np.float32),
        }
    )


def _batches(seed=0, n=3):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(BATCH, CH, SIDE, SIDE)).astype(np.float32) for _ in range(n)]


def _expected(batches, momentum):
    """手写 EMA：方差用无偏估计（``n / (n - 1)``），与 torch 一致。"""
    mean = np.zeros(CH, dtype=np.float32)
    var = np.ones(CH, dtype=np.float32)
    for x in batches:
        batch_mean = x.mean(axis=(0, 2, 3))
        batch_var = x.var(axis=(0, 2, 3)) * (COUNT / (COUNT - 1.0))
        mean = (1.0 - momentum) * mean + momentum * batch_mean
        var = (1.0 - momentum) * var + momentum * batch_var
    return mean, var


def _site_name():
    return bn_sites(_graph())[0].name


class TestRunningStats(unittest.TestCase):
    def test_running_stats_follow_momentum(self):
        g = _graph()
        store = _store()
        batches = _batches()
        trainer = Trainer(g, store, SGD(store.arrays(), lr=0.1), bn_momentum=0.2)
        for x in batches:
            trainer.step({"x": x})

        mean, var = trainer.bn_running_stats()[_site_name()]
        expected_mean, expected_var = _expected(batches, 0.2)
        np.testing.assert_allclose(mean, expected_mean, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(var, expected_var, rtol=1e-5, atol=1e-6)

    def test_stats_update_even_when_parameters_do_not(self):
        """累积窗口没满时参数不动，但 BN 观察到的数据照样进 EMA。"""
        g = _graph()
        store = _store()
        x = _batches(n=1)[0]
        trainer = Trainer(
            g, store, SGD(store.arrays(), lr=0.1), accumulate_steps=2, bn_momentum=0.5
        )
        before = store["gamma"].copy()
        trainer.step({"x": x})

        self.assertTrue(np.array_equal(store["gamma"], before))
        mean, _ = trainer.bn_running_stats()[_site_name()]
        self.assertFalse(np.allclose(mean, np.zeros(CH)))

    def test_running_stats_feed_the_inference_graph(self):
        """训练完的 running stats 直接喂 ``bn_infer``，得到 eval 输出。"""
        from aetherflow.models import bn_infer

        g = _graph()
        store = _store()
        x = _batches(n=1)[0]
        trainer = Trainer(g, store, SGD(store.arrays(), lr=0.1), bn_momentum=0.5)
        trainer.step({"x": x})
        mean, var = trainer.bn_running_stats()[_site_name()]

        gi = Graph("bn_infer")
        xi = gi.input("x", f32[BATCH, CH, SIDE, SIDE])
        mi = gi.input("m", f32[CH])
        vi = gi.input("v", f32[CH])
        gi.input("gamma", f32[CH])
        bi = gi.input("beta", f32[CH])
        gi.output(bn_infer(gi, xi, mi, vi, gi.find("gamma"), bi, eps=EPS), "y")
        got = ReferenceBackend().execute(
            gi, {"x": x, "m": mean, "v": var, "gamma": store["gamma"], "beta": store["beta"]}
        )["y"]

        shape = (1, CH, 1, 1)
        expected = (x - mean.reshape(shape)) / np.sqrt(var.reshape(shape) + EPS)
        expected = expected * store["gamma"].reshape(shape) + store["beta"].reshape(shape)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_stats_graph_only_covers_bn_inputs(self):
        """统计子图只复制到 BN 输入，不重跑分类头/损失那一段。"""
        g = _graph()
        stats_graph = build_stats_graph(g, bn_sites(g))
        self.assertNotIn("reduce_sum", stats_graph.op_names())
        self.assertIn("batch_norm_stats", stats_graph.op_names())

    def test_bn_momentum_none_disables_maintenance(self):
        g = _graph()
        store = _store()
        trainer = Trainer(g, store, SGD(store.arrays(), lr=0.1), bn_momentum=None)
        trainer.step({"x": _batches(n=1)[0]})
        with self.assertRaises(ValueError):
            trainer.bn_running_stats()

    def test_invalid_momentum_is_rejected(self):
        g = _graph()
        store = _store()
        for bad in (1.5, -0.1, True):
            with self.assertRaises(ValueError):
                Trainer(g, store, SGD(store.arrays(), lr=0.1), bn_momentum=bad)

    def test_device_state_with_bn_is_rejected(self):
        g = _graph()
        store = _store()
        with self.assertRaises(ValueError) as ctx:
            Trainer(
                g, store, SGD(store.arrays(), lr=0.1), device_state=True, bn_momentum=0.1
            )
        self.assertIn("设备态", str(ctx.exception))

    def test_symbolic_channel_is_rejected(self):
        g = Graph("dyn")
        x = g.input("x", f32["B", CH, SIDE, SIDE])
        g.output(g.batch_norm(x, g.input("gamma", f32[CH]), g.input("beta", f32[CH])), "y")
        with self.assertRaises(ValueError):
            bn_sites(g)

    def test_checkpoint_roundtrip(self):
        g = _graph()
        batches = _batches()
        site = _site_name()

        store1 = _store()
        t1 = Trainer(g, store1, SGD(store1.arrays(), lr=0.1), bn_momentum=0.3)
        for x in batches[:2]:
            t1.step({"x": x})
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "state.npz")
            t1.save_checkpoint(path)

            store2 = _store()
            t2 = Trainer(g, store2, SGD(store2.arrays(), lr=0.1), bn_momentum=0.3)
            t2.load_training_state(path)
            for a, b in zip(t1.bn_running_stats()[site], t2.bn_running_stats()[site]):
                np.testing.assert_allclose(b, a, rtol=1e-6, atol=1e-7)

            t1.step({"x": batches[2]})
            t2.step({"x": batches[2]})
            for a, b in zip(t1.bn_running_stats()[site], t2.bn_running_stats()[site]):
                np.testing.assert_allclose(b, a, rtol=1e-6, atol=1e-7)

            store3 = _store()
            t3 = Trainer(g, store3, SGD(store3.arrays(), lr=0.1), bn_momentum=0.5)
            with self.assertRaises(ValueError):
                t3.load_training_state(path)


if __name__ == "__main__":
    unittest.main()
