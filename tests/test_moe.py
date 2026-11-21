"""MoE 构件的验收测试。

验收口径与其它模型构件一致：IR 可移植、与独立 numpy 实现数值一致、
可反向、可序列化，另外补两条 MoE 特有的判据：

- **门控归一化**：所有专家参数相同时，输出必须等于单个专家的输出。
  这条不依赖"路由选得对不对"，只依赖"选中概率之和为 1"，能把
  "忘记重新归一化"或"掩码乘错位置"直接暴露出来；
- **负载均衡辅助损失**：``E * Σ f_e * P_e`` 手算一遍，避免把公式抄成
  "看起来像"的另一种写法（例如漏掉 top-k 的 1/k 折算）。

梯度判据用中心差分对拍路由矩阵与专家权重；路由选择是离散的，梯度只经
softmax 概率那条边回传，这与 PyTorch 的 top-k 门控一致。
"""

from __future__ import annotations

import unittest

import numpy as np

from aetherflow import (
    Graph,
    ReferenceBackend,
    backward,
    dumps,
    f32,
    loads,
    verify,
    verify_portable_only,
)
from aetherflow.models import (
    EXPERT_PARALLEL_SUPPORTED,
    MoESpec,
    build_moe,
    require_expert_parallel,
)

_DATA_INPUTS = ("x",)


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------


def _feed(inputs, seed: int = 0):
    rng = np.random.default_rng(seed)
    feed = {}
    for name, value in inputs.items():
        shape = tuple(int(d) for d in value.type.shape)
        feed[name] = (rng.standard_normal(shape) * 0.5).astype(np.float32)
    return feed


def _run(graph, feed, alias: str = "y"):
    return ReferenceBackend().execute(graph, feed)[alias]


def _assert_ir_ok(test: unittest.TestCase, graph) -> None:
    test.assertEqual(verify(graph), [])
    test.assertEqual(verify_portable_only(graph), [])


def _grads(graph, inputs, feed):
    wrt = [v for name, v in inputs.items() if name not in _DATA_INPUTS]
    bwd = backward(graph, wrt=wrt)
    out_shape = tuple(int(d) for d in graph.outputs[0].type.shape)
    bfeed = dict(feed)
    bfeed["dy"] = np.ones(out_shape, dtype=np.float32)
    res = ReferenceBackend().execute(bwd, bfeed)
    names = {v.name for v in wrt}
    return {k[1:]: v for k, v in res.items() if k.startswith("d") and k[1:] in names}, bwd


def _numeric_grad(graph, feed, param: str, eps: float = 1e-3) -> np.ndarray:
    arr = feed[param]
    flat = arr.reshape(-1)
    grad = np.zeros(flat.size, dtype=np.float64)
    for i in range(flat.size):
        orig = float(flat[i])
        flat[i] = orig + eps
        hi = float(np.sum(_run(graph, feed), dtype=np.float64))
        flat[i] = orig - eps
        lo = float(np.sum(_run(graph, feed), dtype=np.float64))
        flat[i] = orig
        grad[i] = (hi - lo) / (2 * eps)
    return grad.reshape(arr.shape)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _act(x, name):
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "silu":
        return x * _sigmoid(x)
    if name == "gelu":
        return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))
    return np.tanh(x)


def _numpy_moe(x, p, spec):
    """逐专家显式循环的参考实现；刻意与构件代码写法不同。"""
    x = np.asarray(x, dtype=np.float64)
    p = {k: np.asarray(v, dtype=np.float64) for k, v in p.items()}

    logits = x @ p["w_router"]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    order = np.argsort(-probs, axis=1, kind="stable")[:, : spec.top_k]
    mask = np.zeros_like(probs)
    np.put_along_axis(mask, order, 1.0, axis=1)

    selected = probs * mask
    gates = selected / selected.sum(axis=1, keepdims=True)

    y = np.zeros_like(x)
    for i in range(spec.experts):
        h = _act(x @ p[f"w1_{i}"] + p[f"b1_{i}"], spec.activation)
        out = h @ p[f"w2_{i}"] + p[f"b2_{i}"]
        y += gates[:, i : i + 1] * out

    f_mean = mask.mean(axis=0) / spec.top_k
    p_mean = probs.mean(axis=0)
    aux = float(spec.experts) * spec.load_balance_weight * float(np.sum(f_mean * p_mean))
    return y, aux


class TestMoE(unittest.TestCase):
    def _build(self, spec=None, with_aux=True):
        spec = spec or MoESpec(d_model=3, d_ff=4, experts=3, top_k=2, batch=4)
        g = Graph("moe")
        inputs, y, aux = build_moe(g, spec)
        g.output(y, "y")
        if with_aux:
            g.output(aux, "aux")
        return g, inputs, spec, y, aux

    def test_ir_is_portable(self):
        g, _, _, _, _ = self._build()
        _assert_ir_ok(self, g)
        self.assertIn("topk_mask", g.op_names())
        self.assertEqual(g.op_names().count("softmax"), 1)

    def test_matches_numpy(self):
        g, inputs, spec, _, _ = self._build()
        feed = _feed(inputs, seed=1)
        expected_y, expected_aux = _numpy_moe(feed["x"], feed, spec)
        np.testing.assert_allclose(_run(g, feed), expected_y, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(
            float(_run(g, feed, "aux")), expected_aux, rtol=1e-5, atol=1e-6
        )

    def test_gates_sum_to_one(self):
        """所有专家共享同一组参数：输出必须等于单个专家的输出。"""
        spec = MoESpec(d_model=3, d_ff=4, experts=3, top_k=2, batch=4)
        g, inputs, spec, _, _ = self._build(spec)
        feed = _feed(inputs, seed=2)
        for e in range(1, spec.experts):
            for suffix in ("w1_", "b1_", "w2_", "b2_"):
                feed[f"{suffix}{e}"] = feed[f"{suffix}0"]
        h = _act(
            feed["x"].astype(np.float64) @ feed["w1_0"].astype(np.float64)
            + feed["b1_0"].astype(np.float64),
            spec.activation,
        )
        single = h @ feed["w2_0"].astype(np.float64) + feed["b2_0"].astype(np.float64)
        np.testing.assert_allclose(_run(g, feed), single, rtol=1e-4, atol=1e-4)

    def test_top1_routes_to_argmax(self):
        """top_k=1 时输出必须是最大概率专家的输出（并列按下标小者优先）。"""
        spec = MoESpec(d_model=3, d_ff=4, experts=3, top_k=1, batch=4)
        g, inputs, spec, _, _ = self._build(spec)
        feed = _feed(inputs, seed=3)
        x = feed["x"].astype(np.float64)
        logits = x @ feed["w_router"].astype(np.float64)
        chosen = np.argmax(logits, axis=1)
        expected = np.zeros_like(x)
        for i in range(spec.experts):
            h = _act(
                x @ feed[f"w1_{i}"].astype(np.float64) + feed[f"b1_{i}"].astype(np.float64),
                spec.activation,
            )
            out = h @ feed[f"w2_{i}"].astype(np.float64) + feed[f"b2_{i}"].astype(np.float64)
            expected[chosen == i] = out[chosen == i]
        np.testing.assert_allclose(_run(g, feed), expected, rtol=1e-4, atol=1e-4)

    def test_load_balance_weight_zero_disables_aux(self):
        spec = MoESpec(d_model=2, d_ff=2, experts=2, top_k=1, batch=3, load_balance_weight=0.0)
        g, inputs, _, _, _ = self._build(spec)
        np.testing.assert_array_equal(float(_run(g, _feed(inputs, 4), "aux")), 0.0)

    def test_gradient_matches_numeric(self):
        spec = MoESpec(d_model=2, d_ff=2, experts=2, top_k=1, batch=2)
        g, inputs, spec, _, _ = self._build(spec, with_aux=False)
        feed = _feed(inputs, seed=5)
        analytic, bwd = _grads(g, inputs, feed)
        _assert_ir_ok(self, bwd)
        for name in ("w_router", "w1_0", "w1_1", "w2_1", "b2_0"):
            numeric = _numeric_grad(g, feed, name)
            np.testing.assert_allclose(
                analytic[name], numeric, rtol=2e-2, atol=2e-2,
                err_msg=f"参数 {name} 的解析梯度与中心差分不一致",
            )

    def test_trains_with_aux_loss(self):
        """接上 ParameterStore + Adam：任务损失与负载均衡项都必须下降。"""
        from aetherflow.train import Adam, ParameterStore

        spec = MoESpec(d_model=2, d_ff=4, experts=2, top_k=1, batch=4,
                       load_balance_weight=0.1)
        g = Graph("moe_train")
        inputs, y, aux = build_moe(g, spec)
        g.output(y, "y")
        g.output(aux, "aux")

        store = ParameterStore.from_graph(g, exclude=_DATA_INPUTS, seed=0)
        opt = Adam(store.arrays(), lr=0.05)
        bwd = backward(g, wrt=[g.find(n) for n in store.names])

        rng = np.random.default_rng(6)
        x = rng.standard_normal((spec.batch, spec.d_model)).astype(np.float32)
        target = (rng.standard_normal((spec.batch, spec.d_model)) * 0.5).astype(np.float32)

        backend = ReferenceBackend()
        losses = []
        for _ in range(60):
            feed = store.feed({"x": x})
            pred = backend.execute(g, feed)["y"]
            diff = pred - target
            loss = float(np.mean(diff * diff)) + float(backend.execute(g, feed)["aux"])
            losses.append(loss)
            bfeed = dict(feed)
            bfeed["dy"] = (2.0 / target.size) * diff
            bfeed["daux"] = np.ones((), dtype=np.float32)
            res = backend.execute(bwd, bfeed)
            opt.step({name: res["d" + name] for name in store.names})
        self.assertLess(losses[-1], losses[0], f"loss 未下降: {losses[0]} -> {losses[-1]}")

    def test_serialization_roundtrip(self):
        g, inputs, _, _, _ = self._build()
        feed = _feed(inputs, seed=7)
        restored = loads(dumps(g))
        self.assertEqual(restored.op_names(), g.op_names())
        np.testing.assert_array_equal(_run(restored, feed), _run(g, feed))
        np.testing.assert_array_equal(
            _run(restored, feed, "aux"), _run(g, feed, "aux")
        )

    def test_expert_parallel_is_rejected(self):
        self.assertFalse(EXPERT_PARALLEL_SUPPORTED)
        with self.assertRaises(NotImplementedError) as ctx:
            MoESpec(expert_parallel=2)
        self.assertIn("专家并行", str(ctx.exception))
        require_expert_parallel(1)  # 单设备显式允许
        with self.assertRaises(NotImplementedError):
            require_expert_parallel(8)
        with self.assertRaises(ValueError):
            require_expert_parallel(0)
        with self.assertRaises(TypeError):
            require_expert_parallel("2")

    def test_invalid_spec(self):
        with self.assertRaises(ValueError):
            MoESpec(experts=2, top_k=3)
        with self.assertRaises(ValueError):
            MoESpec(experts=0)
        with self.assertRaises(ValueError):
            MoESpec(activation="swish")
        with self.assertRaises(ValueError):
            MoESpec(load_balance_weight=-0.1)
        with self.assertRaises(ValueError):
            MoESpec(load_balance_weight=float("inf"))
        with self.assertRaises(TypeError):
            MoESpec(top_k=True)
        with self.assertRaises(TypeError):
            MoESpec(top_k=1.5)


class TestTopKMask(unittest.TestCase):
    def _graph(self, shape, k):
        g = Graph("topk")
        x = g.input("x", f32[shape])
        g.output(g.add_node("topk_mask", [x], {"k": k, "axis": -1}), "y")
        return g, x

    def test_mask_selects_largest_with_stable_ties(self):
        g, _ = self._graph((2, 4), 2)
        x = np.array([[1.0, 3.0, 3.0, 2.0], [5.0, 4.0, 0.0, 1.0]], np.float32)
        expected = np.array([[0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]], np.float32)
        np.testing.assert_array_equal(_run(g, {"x": x}), expected)

    def test_k_equal_to_dim_selects_all(self):
        g, _ = self._graph((3, 2), 2)
        got = _run(g, {"x": np.random.default_rng(0).standard_normal((3, 2)).astype(np.float32)})
        np.testing.assert_array_equal(got, np.ones((3, 2), np.float32))

    def test_invalid_attrs(self):
        with self.assertRaises(ValueError):
            self._graph((2, 3), 4)
        with self.assertRaises(ValueError):
            self._graph((2, 3), 0)
        with self.assertRaises(TypeError):
            self._graph((2, 3), 1.5)


if __name__ == "__main__":
    unittest.main()
