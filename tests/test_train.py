"""训练闭环的验收测试。

判据不是"能跑"，而是"真能学"：

1. **收敛性**：线性回归与 XOR 必须真的把损失降下来。
   只测形状的测试无法发现"梯度符号反了""参数没被原地更新"这类缺陷。
2. **正确性**：交叉熵与数值梯度对齐；优化器的单步更新与手算公式一致。
3. **契约**：缺梯度必须报错；初始化必须可复现；参数与数据不能同名。

这些测试刻意都用参考后端：训练闭环与后端无关，换后端不应改变结论。
"""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from aetherflow import Graph, ReferenceBackend, backward, f16, f32
from aetherflow.train import (
    SGD,
    Adam,
    DataLoader,
    ParameterStore,
    init_array,
    l2_penalty,
    load_checkpoint,
    mean,
    mse,
    save_checkpoint,
    softmax_cross_entropy,
)


BACKEND = ReferenceBackend()


def _grads_of(graph, backward_graph, feed, names):
    """跑一次反向图，取出各参数的梯度。

    种子形状必须取自**前向输出**的类型：反向图的输入顺序是
    "前向输入 + 种子"，种子并不固定在第一位。
    """
    seed = "d" + graph.output_names[0]
    full = dict(feed)
    full[seed] = np.ones(graph.outputs[0].type.shape, dtype=np.float32)
    result = BACKEND.execute(backward_graph, full)
    return {name: result[f"d{name}"] for name in names}


def _forward(graph, feed):
    return BACKEND.execute(graph, feed)[graph.output_names[0]]


def _train(graph, store, opt, batch, steps, on_step=None):
    """通用训练循环：反向图只构一次，每步只换数据。"""
    names = store.names
    bwd = backward(graph, wrt=[graph.find(n) for n in names])
    losses = []
    for i in range(steps):
        feed = store.feed(batch())
        loss = _forward(graph, feed)
        losses.append(float(loss))
        opt.step(_grads_of(graph, bwd, feed, names))
        if on_step is not None:
            on_step(i, loss)
    return losses


def _run_steps(graph, store, opt, batch, steps):
    """只走参数更新、不算损失：验证"续训轨迹一致"时不需要损失值。"""
    names = store.names
    bwd = backward(graph, wrt=[graph.find(n) for n in names])
    for _ in range(steps):
        feed = store.feed(batch)
        opt.step(_grads_of(graph, bwd, feed, names))


# --------------------------------------------------------------------------
# 参数容器
# --------------------------------------------------------------------------


class TestParameterStore(unittest.TestCase):
    def test_default_initializers_follow_name_and_rank(self):
        g = Graph("g")
        w = g.input("w", f32[3, 2])
        b = g.input("b", f32[2])
        gamma = g.input("ln_g", f32[2])
        g.output(g.relu(g.add(w, g.add(b, gamma))), "y")

        store = ParameterStore.from_graph(g)
        self.assertEqual(store["b"].tolist(), [0.0, 0.0])
        self.assertEqual(store["ln_g"].tolist(), [1.0, 1.0])
        # Xavier 的幅度由扇入扇出决定：limit = sqrt(6/(3+2))
        self.assertTrue(np.all(np.abs(store["w"]) <= np.sqrt(6.0 / 5.0) + 1e-6))
        self.assertGreater(float(np.abs(store["w"]).max()), 1e-6)

    def test_same_seed_reproduces_exactly(self):
        g = Graph("g")
        w = g.input("w", f32[5, 4])
        g.output(g.relu(w), "y")
        a = ParameterStore.from_graph(g, seed=7)
        b = ParameterStore.from_graph(g, seed=7)
        c = ParameterStore.from_graph(g, seed=8)
        np.testing.assert_array_equal(a["w"], b["w"])
        self.assertFalse(np.array_equal(a["w"], c["w"]))

    def test_exclude_marks_data_inputs(self):
        g = Graph("g")
        x = g.input("x", f32[2, 3])
        w = g.input("w", f32[3, 3])
        g.output(g.matmul(x, w), "y")
        store = ParameterStore.from_graph(g, exclude=("x",))
        self.assertEqual(store.names, ("w",))

    def test_non_float_and_symbolic_inputs_are_rejected(self):
        from aetherflow import i32

        g = Graph("g")
        ids = g.input("ids", i32[4])
        g.output(ids, "y")
        with self.assertRaises(TypeError):
            ParameterStore.from_graph(g)

        g2 = Graph("g2")
        t = g2.input("t", f32["B", 4])
        g2.output(t, "y")
        with self.assertRaises(ValueError):
            ParameterStore.from_graph(g2)

    def test_feed_rejects_name_collision(self):
        g = Graph("g")
        x = g.input("x", f32[2])
        g.output(x, "y")
        store = ParameterStore.from_graph(g)
        with self.assertRaises(KeyError):
            store.feed({"x": np.zeros(2, dtype=np.float32)})
        self.assertEqual(set(store.feed({"data": np.zeros(2, np.float32)})), {"x", "data"})


# --------------------------------------------------------------------------
# 优化器
# --------------------------------------------------------------------------


class TestOptimizers(unittest.TestCase):
    def test_sgd_plain_step(self):
        p = {"w": np.array([1.0, -2.0], dtype=np.float32)}
        opt = SGD(p, lr=0.5)
        opt.step({"w": np.array([2.0, 4.0], dtype=np.float32)})
        np.testing.assert_allclose(p["w"], [0.0, -4.0])

    def test_sgd_momentum_matches_manual_formula(self):
        w0 = 1.0
        p = {"w": np.array([w0], dtype=np.float32)}
        opt = SGD(p, lr=0.1, momentum=0.9)
        g = np.array([2.0], dtype=np.float32)
        # 第一步 buf = g，更新 w -= lr * g
        opt.step({"w": g})
        np.testing.assert_allclose(p["w"], [w0 - 0.1 * 2.0], rtol=1e-6)
        # 第二步 buf = 0.9*2 + 2 = 3.8
        opt.step({"w": g})
        np.testing.assert_allclose(p["w"], [w0 - 0.2 - 0.1 * 3.8], rtol=1e-6)

    def test_adam_first_step_is_sign_like(self):
        p = {"w": np.array([1.0, -1.0], dtype=np.float32)}
        opt = Adam(p, lr=0.1)
        opt.step({"w": np.array([5.0, -0.5], dtype=np.float32)})
        # 偏差修正后 m_hat = g, v_hat = g^2 ⇒ 步长约等于 lr * sign(g)
        np.testing.assert_allclose(p["w"], [0.9, -0.9], atol=1e-5)

    def test_missing_gradient_raises(self):
        p = {"w": np.zeros(2, dtype=np.float32)}
        with self.assertRaises(KeyError):
            SGD(p, lr=0.1).step({})

    def test_weight_decay_shrinks_without_gradient(self):
        p = {"w": np.array([1.0], dtype=np.float32)}
        opt = SGD(p, lr=0.1, weight_decay=0.5)
        opt.step({"w": np.zeros(1, dtype=np.float32)})
        np.testing.assert_allclose(p["w"], [0.95], rtol=1e-6)

    def test_rebind_switches_target_arrays(self):
        """换表之后更新只落在新表上；名字或形状不一致直接拒绝。"""
        host = {"w": np.ones(2, dtype=np.float32)}
        opt = SGD(host, lr=0.5)
        device = {"w": np.ones(2, dtype=np.float32)}
        opt.rebind(device)
        opt.step({"w": np.full(2, 2.0, dtype=np.float32)})
        np.testing.assert_allclose(device["w"], [0.0, 0.0])
        np.testing.assert_allclose(host["w"], [1.0, 1.0], err_msg="旧表不该再被更新")

        with self.assertRaises(KeyError):
            opt.rebind({"other": np.ones(2, dtype=np.float32)})
        with self.assertRaises(ValueError):
            opt.rebind({"w": np.ones(3, dtype=np.float32)})

    def test_bad_gradient_leaves_no_half_step(self):
        """任何一个参数的梯度不合法，整步都不提交。

        逐个"校验一个更新一个"时，先被校验的参数已经走了一步、步数却还是
        旧的；续训会从这个自相矛盾的状态继续。
        """
        params = {
            "a": np.ones(2, dtype=np.float32),
            "b": np.ones(2, dtype=np.float32),
        }
        opt = SGD(params, lr=0.1)
        with self.assertRaises(ValueError):
            opt.step({
                "a": np.ones(2, dtype=np.float32),
                "b": np.ones(3, dtype=np.float32),
            })
        np.testing.assert_array_equal(params["a"], [1.0, 1.0])
        np.testing.assert_array_equal(params["b"], [1.0, 1.0])
        self.assertEqual(opt.step_count, 0)

    def test_adam_rejects_half_precision_parameters(self):
        """f16 参数 + eps=1e-8：零梯度会得到 0/0。

        ``eps`` 在 f16 里下溢成 0，``v=0`` 时更新量是 ``0 / (sqrt(0) + 0)``，
        一步零梯度就能把参数写成 NaN。入口直接拒绝，并指向 f32 master。
        """
        params = {"w": np.ones(2, dtype=np.float16)}
        with self.assertRaises(TypeError):
            Adam(params)
        # f32 参数不受影响
        Adam({"w": np.ones(2, dtype=np.float32)})


# --------------------------------------------------------------------------
# 损失
# --------------------------------------------------------------------------


class TestLosses(unittest.TestCase):
    def test_mse_matches_numpy(self):
        g = Graph("mse")
        a = g.input("a", f32[3, 4])
        b = g.input("b", f32[3, 4])
        g.output(mse(g, a, b), "loss")
        rng = np.random.default_rng(0)
        feed = {"a": rng.standard_normal((3, 4)).astype(np.float32),
                "b": rng.standard_normal((3, 4)).astype(np.float32)}
        want = float(np.mean((feed["a"] - feed["b"]) ** 2))
        self.assertAlmostEqual(float(_forward(g, feed)), want, places=6)

    def test_cross_entropy_matches_numpy(self):
        g = Graph("ce")
        logits = g.input("logits", f32[2, 5])
        labels = g.input("labels", f32[2, 5])
        g.output(softmax_cross_entropy(g, logits, labels), "loss")
        rng = np.random.default_rng(1)
        z = rng.standard_normal((2, 5)).astype(np.float32)
        lab = np.eye(5, dtype=np.float32)[[3, 0]]
        want = float(-np.mean(np.sum(lab * np.log(_softmax(z)), axis=-1)))
        self.assertAlmostEqual(float(_forward(g, {"logits": z, "labels": lab})), want, places=5)

    def test_mean_requires_static_shape(self):
        g = Graph("m")
        x = g.input("x", f32["B", 4])
        with self.assertRaises(ValueError):
            mean(g, x)

    def test_cross_entropy_survives_saturated_logits(self):
        """置信度极端的错分样本必须仍有满幅梯度。

        ``[0, -100]`` 目标为第二类时标准交叉熵是 100、梯度是 ``[1, -1]``。
        旧写法 ``-log(softmax(p) + 1e-12)`` 会把损失钉在 27.63 附近，并把梯度
        乘上 ``p_k / (p_k + eps) ≈ 3.8e-32``——最需要纠正的样本反而学不动。
        """
        for dtype in (f32, f16):
            g = Graph("ce_sat")
            logits = g.input("logits", dtype[1, 2])
            labels = g.input("labels", dtype[1, 2])
            g.output(softmax_cross_entropy(g, logits, labels), "loss")
            np_dtype = np.dtype(dtype.np_dtype)
            feed = {
                "logits": np.array([[0.0, -100.0]], np_dtype),
                "labels": np.array([[0.0, 1.0]], np_dtype),
            }
            loss = float(_forward(g, feed))
            self.assertTrue(np.isfinite(loss), f"{dtype.value} loss={loss}")
            self.assertAlmostEqual(loss, 100.0, places=2)

            bwd = backward(g, wrt=[logits])
            res = BACKEND.execute(bwd, {**feed, "dloss": np.ones((), np_dtype)})
            np.testing.assert_allclose(
                res["dlogits"].astype(np.float64),
                np.array([[1.0, -1.0]]),
                rtol=1e-3,
                atol=1e-3,
            )

    def test_cross_entropy_matches_numpy_over_a_wide_range(self):
        """logit 间隔从 0 到 1e4：loss 与梯度都要对照独立公式。"""
        g = Graph("ce_wide")
        logits = g.input("logits", f32[2, 3])
        labels = g.input("labels", f32[2, 3])
        g.output(softmax_cross_entropy(g, logits, labels), "loss")
        z = np.array([[0.0, -1e4, 3.0], [-2.0, 0.5, 0.0]], np.float32)
        lab = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], np.float32)
        feed = {"logits": z, "labels": lab}

        def log_softmax(a):
            a = a.astype(np.float64)
            m = a.max(-1, keepdims=True)
            return a - m - np.log(np.exp(a - m).sum(-1, keepdims=True))

        want = float(-np.mean(np.sum(lab * log_softmax(z), axis=-1)))
        self.assertAlmostEqual(float(_forward(g, feed)), want, places=3)

        res = BACKEND.execute(
            backward(g, wrt=[logits]), {**feed, "dloss": np.ones((), np.float32)}
        )
        p = np.exp(log_softmax(z))
        np.testing.assert_allclose(
            res["dlogits"], (p - lab) / z.shape[0], rtol=1e-4, atol=1e-5
        )

    def test_l2_penalty_value_and_gradient(self):
        g = Graph("l2")
        w = g.input("w", f32[3])
        loss = l2_penalty(g, [w], weight=0.5)
        g.output(loss, "loss")
        feed = {"w": np.array([1.0, -2.0, 3.0], dtype=np.float32)}
        self.assertAlmostEqual(float(_forward(g, feed)), 0.25 * 14.0, places=5)
        bwd = backward(g, wrt=[w])
        res = BACKEND.execute(bwd, {**feed, "dloss": np.ones((), np.float32)})
        np.testing.assert_allclose(res["dw"], 0.5 * feed["w"], rtol=1e-6)


def _softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# --------------------------------------------------------------------------
# 收敛性
# --------------------------------------------------------------------------


class TestConvergence(unittest.TestCase):
    def test_linear_regression_reaches_optimum(self):
        g = Graph("linreg")
        x = g.input("x", f32[8, 3])
        w = g.input("w", f32[3, 1])
        b = g.input("b", f32[1])
        target = g.input("target", f32[8, 1])
        g.output(mse(g, g.add(g.matmul(x, w), b), target), "loss")

        rng = np.random.default_rng(0)
        xs = rng.standard_normal((8, 3)).astype(np.float32)
        true_w = np.array([[1.0], [-2.0], [0.5]], dtype=np.float32)
        true_b = np.array([0.3], dtype=np.float32)
        ys = xs @ true_w + true_b

        store = ParameterStore.from_graph(g, exclude=("x", "target"), seed=0)
        opt = Adam(store.arrays(), lr=0.1)
        losses = _train(g, store, opt, lambda: {"x": xs, "target": ys}, 400)

        self.assertLess(losses[-1], 1e-6, f"损失没有收敛: {losses[0]} -> {losses[-1]}")
        np.testing.assert_allclose(store["w"], true_w, atol=2e-3)
        np.testing.assert_allclose(store["b"], true_b, atol=2e-3)

    def test_mlp_learns_xor(self):
        g = Graph("xor")
        x = g.input("x", f32[4, 2])
        w1 = g.input("w1", f32[2, 8])
        b1 = g.input("b1", f32[8])
        w2 = g.input("w2", f32[8, 1])
        b2 = g.input("b2", f32[1])
        target = g.input("target", f32[4, 1])
        h = g.tanh(g.add(g.matmul(x, w1), b1))
        g.output(mse(g, g.add(g.matmul(h, w2), b2), target), "loss")

        xs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        ys = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32)

        store = ParameterStore.from_graph(g, exclude=("x", "target"), seed=3)
        opt = SGD(store.arrays(), lr=0.1, momentum=0.9)
        losses = _train(g, store, opt, lambda: {"x": xs, "target": ys}, 2000)

        self.assertLess(losses[-1], 0.01, f"XOR 没学会: {losses[0]} -> {losses[-1]}")

        # 拟合结果用参数直接前算（与图里的表达式完全一致），核对四个样本的符号
        pred = np.tanh(xs @ store["w1"] + store["b1"]) @ store["w2"] + store["b2"]
        np.testing.assert_array_equal(pred > 0.5, ys > 0.5)

    def test_gradient_descent_reduces_loss_monotonically_in_expectation(self):
        """小学习率的全批梯度下降：损失必须持续下降（不是只降一点点）。"""
        g = Graph("quad")
        w = g.input("w", f32[2])
        target = g.input("target", f32[2])
        g.output(mse(g, w, target), "loss")

        store = ParameterStore.from_graph(g, exclude=("target",), seed=0)
        target_arr = np.array([2.0, -3.0], dtype=np.float32)
        opt = SGD(store.arrays(), lr=0.3)
        losses = _train(g, store, opt, lambda: {"target": target_arr}, 100)

        self.assertLess(losses[-1], losses[0] * 1e-6)
        np.testing.assert_allclose(store["w"], target_arr, atol=1e-4)

    def test_training_graph_backward_roundtrip(self):
        """训练用的反向图也是普通 IR：JSON 往返后逐步结果完全一致。"""
        from aetherflow import dumps, loads

        g = Graph("rt")
        w = g.input("w", f32[3])
        t = g.input("t", f32[3])
        g.output(mse(g, w, t), "loss")
        bwd = backward(g, wrt=[w])

        restored = loads(dumps(bwd))
        feed = {
            "w": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "t": np.array([0.5, 2.5, -1.0], dtype=np.float32),
            "dloss": np.ones((), dtype=np.float32),
        }
        np.testing.assert_array_equal(
            BACKEND.execute(restored, feed)["dw"], BACKEND.execute(bwd, feed)["dw"]
        )

    def test_init_array_unknown_strategy(self):
        with self.assertRaises(ValueError):
            init_array("w", (2, 2), init="magic")


# --------------------------------------------------------------------------
# 存档
# --------------------------------------------------------------------------


class TestCheckpoint(unittest.TestCase):
    """存档的验收点不是"文件能写出来"，而是"续训和没中断过一样"。"""

    def test_roundtrip_restores_parameters_slots_and_meta(self):
        store = ParameterStore({
            "w": np.arange(6, dtype=np.float32).reshape(3, 2),
            "b": np.zeros(2, dtype=np.float32),
        })
        opt = Adam(store.arrays(), lr=0.1)
        opt.step({
            "w": np.ones((3, 2), dtype=np.float32),
            "b": np.ones(2, dtype=np.float32),
        })
        snapshot = {n: np.array(store[n], copy=True) for n in store.names}
        want_m = {n: np.array(arr, copy=True) for n, arr in opt._m.items()}

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ck.npz")
            save_checkpoint(
                path, store, opt,
                meta={"epoch": 3, "best": 0.25, "tag": "run-a", "done": False},
            )

            fresh = ParameterStore({
                "w": np.zeros((3, 2), dtype=np.float32),
                "b": np.zeros(2, dtype=np.float32),
            })
            fresh_opt = Adam(fresh.arrays(), lr=0.1)
            alias = fresh["w"]
            meta = load_checkpoint(path, fresh, fresh_opt)

        for name in store.names:
            np.testing.assert_array_equal(fresh[name], snapshot[name])
        self.assertEqual(meta, {"epoch": 3, "best": 0.25, "tag": "run-a", "done": False})
        self.assertEqual(fresh_opt.step_count, opt.step_count)
        for name in want_m:
            np.testing.assert_array_equal(fresh_opt._m[name], want_m[name])
        # 原地覆盖：优化器手里还是同一个数组对象，不会被换成没人用的副本
        self.assertIs(fresh_opt.params["w"], alias)
        np.testing.assert_array_equal(alias, snapshot["w"])

    def test_resumed_training_matches_uninterrupted(self):
        """存档前 3 步 + 续训 3 步，必须与连续 6 步逐位一致。"""
        g = Graph("traj")
        x = g.input("x", f32[8, 3])
        w = g.input("w", f32[3, 1])
        b = g.input("b", f32[1])
        target = g.input("target", f32[8, 1])
        g.output(mse(g, g.add(g.matmul(x, w), b), target), "loss")

        rng = np.random.default_rng(0)
        batch = {
            "x": rng.standard_normal((8, 3)).astype(np.float32),
            "target": rng.standard_normal((8, 1)).astype(np.float32),
        }

        straight = ParameterStore.from_graph(g, exclude=("x", "target"), seed=5)
        opt = Adam(straight.arrays(), lr=0.05)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ck.npz")
            _run_steps(g, straight, opt, batch, 3)
            save_checkpoint(path, straight, opt, meta={"epoch": 3})
            _run_steps(g, straight, opt, batch, 3)

            resumed = ParameterStore.from_graph(g, exclude=("x", "target"), seed=999)
            resumed_opt = Adam(resumed.arrays(), lr=0.05)
            self.assertEqual(load_checkpoint(path, resumed, resumed_opt)["epoch"], 3)
            _run_steps(g, resumed, resumed_opt, batch, 3)

        for name in straight.names:
            np.testing.assert_array_equal(
                resumed[name], straight[name],
                err_msg=f"参数 {name} 的续训轨迹与连续训练不一致",
            )

    def test_parameter_set_mismatch_is_rejected(self):
        store = ParameterStore({"w": np.zeros(2, np.float32)})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ck.npz")
            save_checkpoint(path, store)
            other = ParameterStore({
                "w": np.zeros(2, np.float32),
                "b": np.zeros(2, np.float32),
            })
            with self.assertRaises(KeyError):
                load_checkpoint(path, other)

    def test_shape_mismatch_is_rejected(self):
        store = ParameterStore({"w": np.zeros((3, 2), np.float32)})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ck.npz")
            save_checkpoint(path, store)
            other = ParameterStore({"w": np.zeros((2, 3), np.float32)})
            with self.assertRaises(ValueError):
                load_checkpoint(path, other)

    def test_path_must_end_with_npz(self):
        store = ParameterStore({"w": np.zeros(2, np.float32)})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_checkpoint(os.path.join(tmp, "ck.bin"), store)

    def test_loading_optimizer_state_from_weights_only_file_fails(self):
        store = ParameterStore({"w": np.zeros(2, np.float32)})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ck.npz")
            save_checkpoint(path, store)  # 没传 optimizer
            fresh = ParameterStore({"w": np.zeros(2, np.float32)})
            with self.assertRaises(ValueError):
                load_checkpoint(path, fresh, SGD(fresh.arrays(), lr=0.1))
            # 只加载权重是合法场景
            self.assertEqual(load_checkpoint(path, fresh), {})

    def test_meta_must_be_scalar(self):
        store = ParameterStore({"w": np.zeros(2, np.float32)})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TypeError):
                save_checkpoint(
                    os.path.join(tmp, "ck.npz"), store, meta={"history": [1.0, 2.0]}
                )

    def test_incomplete_slots_are_rejected(self):
        """``{'step': 100, 'slots': {}}`` 不能把旧动量原样留下。

        只检查"存档里出现的槽位名"，会把残缺存档当成合法的；结果是步数跳到
        100、m/v 还是旧的，偏差修正直接算错。
        """
        params = {"w": np.ones(2, dtype=np.float32)}
        opt = Adam(params, lr=0.1)
        opt.step({"w": np.full(2, 1.0, dtype=np.float32)})
        before_m = {n: np.array(a, copy=True) for n, a in opt._m.items()}

        with self.assertRaises(KeyError):
            opt.load_state_dict({"step": 100, "slots": {}})
        self.assertEqual(opt.step_count, 1)
        for name, arr in before_m.items():
            np.testing.assert_array_equal(opt._m[name], arr)

        # 只给一个槽位也算残缺
        with self.assertRaises(KeyError):
            opt.load_state_dict({"step": 100, "slots": {"m": dict(before_m)}})
        self.assertEqual(opt.step_count, 1)

    def test_failed_load_keeps_parameters_intact(self):
        """坏存档不能先把权重改掉、再在优化器检查处报错。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "weights.npz")
            saved = ParameterStore({"w": np.full(2, 9.0, np.float32)})
            save_checkpoint(path, saved)  # 只有权重，没有优化器状态

            fresh = ParameterStore({"w": np.ones(2, np.float32)})
            opt = Adam(fresh.arrays(), lr=0.1)
            with self.assertRaises(ValueError):
                load_checkpoint(path, fresh, opt)
            np.testing.assert_array_equal(fresh["w"], [1.0, 1.0])

    def test_stream_state_roundtrip(self):
        """流式状态与权重同源存取：原地写回模板，调用方直接拿去当 feed。"""
        store = ParameterStore({"w": np.zeros((2,), np.float32)})
        state = {
            "l0_conv0": np.arange(6, dtype=np.float32).reshape(2, 1, 3),
            "l0_h0": np.full((2, 3, 4), 0.5, dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ck.npz")
            save_checkpoint(path, store, meta={"step": 7}, state=state)

            fresh = ParameterStore({"w": np.ones((2,), np.float32)})
            template = {name: np.zeros_like(arr) for name, arr in state.items()}
            alias = template["l0_h0"]
            meta = load_checkpoint(path, fresh, state=template)

        self.assertEqual(meta, {"step": 7})
        for name, want in state.items():
            np.testing.assert_array_equal(template[name], want)
        # 原地写入：模板对象没被换掉，调用方手里的字典就是下一次 decode 的 feed
        self.assertIs(template["l0_h0"], alias)

    def test_stream_state_missing_from_file_is_rejected(self):
        """旧格式存档没有状态条目：要续接就必须报错，不能静默从零开始。"""
        store = ParameterStore({"w": np.zeros(2, np.float32)})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ck.npz")
            save_checkpoint(path, store)
            template = {"l0_h0": np.zeros((1, 2, 3), np.float32)}
            with self.assertRaises(ValueError):
                load_checkpoint(path, store, state=template)
            # 不传 state 时旧存档照常可读——只加载权重始终是合法场景
            self.assertEqual(load_checkpoint(path, store), {})

    def test_stream_state_mismatch_leaves_template_intact(self):
        store = ParameterStore({"w": np.zeros(2, np.float32)})
        saved = {"l0_h0": np.zeros((2, 3, 4), np.float32)}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ck.npz")
            save_checkpoint(path, store, state=saved)

            wrong_shape = {"l0_h0": np.ones((3, 2, 4), np.float32)}
            with self.assertRaises(ValueError):
                load_checkpoint(path, store, state=wrong_shape)
            np.testing.assert_array_equal(wrong_shape["l0_h0"], 1.0)

            with self.assertRaises(KeyError):
                load_checkpoint(path, store, state={"l0_conv0": np.zeros((2, 3), np.float32)})

    def test_failed_stream_state_check_keeps_optimizer_untouched(self):
        """流式状态校验失败时，优化器不能被提前提交。

        优化器自己的加载是"先校验后提交"，但它排在状态校验之前的话，坏模板
        会留下"步数与动量已恢复、参数还是旧的"的反向半步——续训第一发的
        更新方向就与中断前对不上，而这种偏差在损失曲线上几乎看不出来。
        """
        store = ParameterStore({"w": np.zeros((2,), np.float32)})
        opt = Adam(store.arrays(), lr=0.1)
        opt.step({"w": np.ones((2,), np.float32)})
        saved = {"l0_h0": np.zeros((2, 3, 4), np.float32)}

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ck.npz")
            save_checkpoint(path, store, opt, state=saved)

            fresh = ParameterStore({"w": np.zeros((2,), np.float32)})
            fresh_opt = Adam(fresh.arrays(), lr=0.1)
            wrong = {"l0_h0": np.ones((3, 2, 4), np.float32)}
            with self.assertRaises(ValueError):
                load_checkpoint(path, fresh, fresh_opt, state=wrong)

        self.assertEqual(fresh_opt.step_count, 0)
        self.assertEqual(fresh_opt._m, {})
        np.testing.assert_array_equal(fresh["w"], 0.0)
        np.testing.assert_array_equal(wrong["l0_h0"], 1.0)

    def test_save_replaces_atomically_and_leaves_no_temp_file(self):
        """保存走"临时文件 + 原子替换"，目录里不能留下半成品。"""
        store = ParameterStore({"w": np.zeros(2, np.float32)})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ck.npz")
            save_checkpoint(path, store)
            self.assertEqual(os.listdir(tmp), ["ck.npz"])

            store["w"][...] = 3.0  # 覆盖已存在的存档
            save_checkpoint(path, store)
            self.assertEqual(os.listdir(tmp), ["ck.npz"])

            fresh = ParameterStore({"w": np.zeros(2, np.float32)})
            load_checkpoint(path, fresh)
            np.testing.assert_array_equal(fresh["w"], [3.0, 3.0])

    def test_interrupted_save_keeps_previous_file(self):
        """写到一半失败：旧存档原样保留，目标路径不会变成半个文件。"""

        def _broken_savez(file, **payload):
            # 模拟"已经写出半截数据，然后失败"：这是原子替换要挡住的真实故障
            data = b"PK\x03\x04partial"
            if hasattr(file, "write"):
                file.write(data)
            else:
                with open(file, "wb") as fh:
                    fh.write(data)
            raise OSError("磁盘满")

        store = ParameterStore({"w": np.zeros(2, np.float32)})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ck.npz")
            save_checkpoint(path, store)
            with open(path, "rb") as fh:
                before = fh.read()

            store["w"][...] = 7.0
            with mock.patch(
                "aetherflow.train.checkpoint.np.savez", side_effect=_broken_savez
            ):
                with self.assertRaises(OSError):
                    save_checkpoint(path, store)

            with open(path, "rb") as fh:
                self.assertEqual(fh.read(), before)
            self.assertEqual(os.listdir(tmp), ["ck.npz"])


# --------------------------------------------------------------------------
# 数据迭代
# --------------------------------------------------------------------------


class TestDataLoader(unittest.TestCase):
    def test_sequential_batches_cover_every_sample(self):
        data = {"x": np.arange(10, dtype=np.float32).reshape(10, 1)}
        loader = DataLoader(data, batch_size=4)
        batches = list(loader)
        self.assertEqual([b["x"].shape for b in batches], [(4, 1), (4, 1), (2, 1)])
        np.testing.assert_array_equal(
            np.concatenate([b["x"] for b in batches]), data["x"]
        )
        self.assertEqual(len(loader), 3)
        self.assertEqual(loader.size, 10)

    def test_shuffle_is_reproducible_and_advances_per_epoch(self):
        data = {"x": np.arange(8, dtype=np.float32)}
        first = DataLoader(data, batch_size=4, shuffle=True, seed=11)
        second = DataLoader(data, batch_size=4, shuffle=True, seed=11)
        epoch0 = [b["x"] for b in first]
        epoch1 = [b["x"] for b in first]
        # 两轮都覆盖全部样本，只是顺序不同
        np.testing.assert_array_equal(np.sort(np.concatenate(epoch0)), data["x"])
        np.testing.assert_array_equal(np.sort(np.concatenate(epoch1)), data["x"])
        # 两个 loader 的同一轮顺序完全一致
        for a, b in zip(epoch0, [b["x"] for b in second]):
            np.testing.assert_array_equal(a, b)
        # 不同轮次的顺序必须不同（否则"洗牌"根本没生效）
        self.assertFalse(np.array_equal(np.concatenate(epoch0), np.concatenate(epoch1)))
        # reset 后重放第一轮
        first.reset()
        self.assertEqual(first.epoch, 0)
        for a, b in zip([b["x"] for b in first], epoch0):
            np.testing.assert_array_equal(a, b)

    def test_shuffle_differs_from_sequential(self):
        data = {"x": np.arange(16, dtype=np.float32)}
        shuffled = np.concatenate([b["x"] for b in DataLoader(data, 4, shuffle=True, seed=1)])
        self.assertFalse(np.array_equal(shuffled, data["x"]))
        np.testing.assert_array_equal(np.sort(shuffled), data["x"])

    def test_drop_last_drops_incomplete_batch(self):
        data = {"x": np.arange(10, dtype=np.float32)}
        loader = DataLoader(data, batch_size=4, drop_last=True)
        self.assertEqual(len(loader), 2)
        self.assertTrue(all(b["x"].shape[0] == 4 for b in loader))

    def test_mismatched_sample_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            DataLoader({"x": np.zeros((4, 2)), "y": np.zeros((3, 1))}, batch_size=2)
        with self.assertRaises(ValueError):
            DataLoader({"x": np.zeros((4, 2))}, batch_size=0)
        with self.assertRaises(ValueError):
            DataLoader({"x": np.float32(1.0)}, batch_size=1)

    def test_training_loop_consumes_batches(self):
        """DataLoader 产出的键必须能直接喂 store.feed：这是它与训练闭环的接口。

        批大小必须与图的静态批维一致——``mse`` 要静态元素个数来算平均，
        所以这里用 4 条一批、8 条样本两个批的配置。
        """
        g = Graph("dl")
        x = g.input("x", f32[4, 2])
        w = g.input("w", f32[2, 1])
        target = g.input("target", f32[4, 1])
        g.output(mse(g, g.matmul(x, w), target), "loss")

        rng = np.random.default_rng(2)
        xs = rng.standard_normal((8, 2)).astype(np.float32)
        true_w = np.array([[1.5], [-0.5]], dtype=np.float32)
        data = {"x": xs, "target": xs @ true_w}

        store = ParameterStore.from_graph(g, exclude=("x", "target"), seed=0)
        opt = Adam(store.arrays(), lr=0.1)
        loader = DataLoader(data, batch_size=4, shuffle=True, seed=1)
        bwd = backward(g, wrt=[g.find(n) for n in store.names])

        epoch_losses = []
        for _ in range(40):
            total, count = 0.0, 0
            for batch in loader:
                self.assertEqual(set(batch), {"x", "target"})
                feed = store.feed(batch)
                loss = float(_forward(g, feed))
                total += loss
                count += 1
                opt.step(_grads_of(g, bwd, feed, store.names))
            epoch_losses.append(total / count)

        self.assertEqual(loader.epoch, 40)
        self.assertEqual(len(loader), 2)
        self.assertLess(
            epoch_losses[-1], epoch_losses[0] * 0.05,
            f"用 DataLoader 喂数据没学会: {epoch_losses[0]} -> {epoch_losses[-1]}",
        )


if __name__ == "__main__":
    unittest.main()
