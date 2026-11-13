"""Transformer Block 端到端验证。

验证目标不是"跑通"，而是三件事同时成立：
1. IR 能完整表达一个真实模型，且全部节点是 PORTABLE（无硬件泄漏）
2. 参考后端的输出与一份独立写的 numpy 实现数值一致
3. 同一份 IR 在多个符号形状下都能执行，无需重建图
"""

import unittest

import numpy as np

from aetherflow import Graph, ReferenceBackend, f32, verify, verify_portable_only
from aetherflow.models import BlockSpec, build_transformer_block
from aetherflow.runtime.reference import ReferenceExecutor


# ---- 独立实现的 numpy 参照（刻意不共享任何 IR 代码）----------------------


def _ln(a, g, b, eps):
    m = a.mean(-1, keepdims=True)
    v = a.var(-1, keepdims=True)
    return (a - m) / np.sqrt(v + eps) * g + b


def _softmax(a):
    m = a.max(-1, keepdims=True)
    e = np.exp(a - m)
    return e / e.sum(-1, keepdims=True)


def _silu(a):
    return a / (1.0 + np.exp(-a))


def numpy_transformer_block(x, p, spec):
    d, h, dh = spec.d_model, spec.num_heads, spec.head_dim
    b, s, _ = x.shape

    h1 = _ln(x, p["ln1_g"], p["ln1_b"], spec.eps)
    q, k, v = h1 @ p["wq"], h1 @ p["wk"], h1 @ p["wv"]

    qh = q.reshape(b, s, h, dh).transpose(0, 2, 1, 3)
    kh = k.reshape(b, s, h, dh).transpose(0, 2, 1, 3)
    vh = v.reshape(b, s, h, dh).transpose(0, 2, 1, 3)

    scores = (qh @ kh.transpose(0, 1, 3, 2)) * p["scale"]
    ctx = (_softmax(scores) @ vh).transpose(0, 2, 1, 3).reshape(b, s, d)

    x1 = x + ctx @ p["wo"]
    h2 = _ln(x1, p["ln2_g"], p["ln2_b"], spec.eps)
    return x1 + _silu(h2 @ p["w1"]) @ p["w2"]


# ---- 测试 ----------------------------------------------------------------


class TestTransformerBlock(unittest.TestCase):
    spec = BlockSpec(d_model=32, num_heads=4, d_ff=64)

    def _params(self, inputs, rng):
        return {
            name: rng.standard_normal(shape).astype(np.float32) * 0.1
            for name, shape in self.spec.param_shapes.items()
            if name != "scale"
        }

    def _build(self):
        g = Graph("transformer_block")
        inputs, out = build_transformer_block(g, self.spec)
        g.output(out, "y")
        return g, inputs

    def test_ir_is_fully_portable(self):
        g, _ = self._build()
        self.assertEqual(verify_portable_only(g), [], "共享路径上出现了不可移植节点")

    def test_ir_verifies_cleanly(self):
        g, _ = self._build()
        self.assertEqual([str(i) for i in verify(g)], [])

    def test_numerical_equivalence_with_numpy(self):
        g, inputs = self._build()
        rng = np.random.default_rng(0)
        b, s = 2, 5
        feed = {"x": rng.standard_normal((b, s, self.spec.d_model)).astype(np.float32)}
        feed.update(self._params(inputs, rng))
        feed["scale"] = np.array([1.0 / np.sqrt(self.spec.head_dim)], dtype=np.float32)

        got = ReferenceBackend().execute(g, feed)["y"]
        want = numpy_transformer_block(feed["x"], feed, self.spec)

        self.assertEqual(got.shape, (b, s, self.spec.d_model))
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

    def test_same_graph_multiple_shapes(self):
        """一次构图，多个 (batch, seq) 组合复用，验证符号维没有退化成静态维。"""
        g, inputs = self._build()
        rng = np.random.default_rng(1)
        params = self._params(inputs, rng)
        params["scale"] = np.array([1.0 / np.sqrt(self.spec.head_dim)], dtype=np.float32)

        backend = ReferenceBackend()
        for b, s in ((1, 1), (3, 7)):
            feed = dict(params)
            feed["x"] = rng.standard_normal((b, s, self.spec.d_model)).astype(np.float32)
            got = backend.execute(g, feed)["y"]
            want = numpy_transformer_block(feed["x"], feed, self.spec)
            np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

    def test_attention_scores_shape(self):
        g, _ = self._build()
        attn = [v for v in g.values if v.producer and v.producer.op.name == "softmax"][0]
        self.assertEqual(str(attn.type), "f32[B, 4, S, S]")


class TestReleaseLiveness(unittest.TestCase):
    """捕获用的"用完即回收"只能改变环境里留了谁，不能改变数值。

    这是 arena 能复用的前提：不摘掉已死的中间值，全部中间张量会一起活到
    图结束，分配器只能一路向上要新块（实测多占一倍显存）。
    """

    def _graph(self):
        g = Graph()
        x = g.input("x", f32[2, 3])
        w = g.input("w", f32[3])
        g.output(g.relu(g.mul(x, w)), "z")
        return g

    def test_release_matches_and_drops_dead_values(self):
        g = self._graph()
        rng = np.random.default_rng(0)
        feed = {
            "x": rng.standard_normal((2, 3)).astype(np.float32),
            "w": rng.standard_normal(3).astype(np.float32),
        }
        executor = ReferenceExecutor()
        full = executor.run_device(g, feed)
        lean = executor.run_device(g, feed, release=True)

        np.testing.assert_array_equal(lean[g.outputs[0].id], full[g.outputs[0].id])

        mid = [v for v in g.values if v.producer is not None and v not in g.outputs][0]
        self.assertIn(mid.id, full)  # 默认路径按老样子保留全部中间值
        self.assertNotIn(mid.id, lean)  # 释放路径只剩输入与输出
        for v in list(g.inputs) + list(g.outputs):
            self.assertIn(v.id, lean)


class TestBlockSerialization(unittest.TestCase):
    """真实模型图的序列化往返。

    5.8 的格式用例只覆盖了 5 个算子；Transformer Block 覆盖 14 个算子在真实
    用法下的组合（符号维、批量 matmul、广播 add、slice 切片、融合后的 softmax）。
    """

    def test_roundtrip_runs_identically(self):
        from aetherflow import dumps, loads

        spec = BlockSpec(d_model=32, num_heads=4, d_ff=64)
        g = Graph("block")
        _, out = build_transformer_block(g, spec)
        g.output(out, "y")

        restored = loads(dumps(g))
        self.assertEqual(restored.op_names(), g.op_names())
        self.assertEqual(restored.output_names, g.output_names)

        rng = np.random.default_rng(2)
        params = {}
        for name, shape in spec.param_shapes.items():
            params[name] = (
                np.array([1.0 / np.sqrt(spec.head_dim)], dtype=np.float32)
                if name == "scale"
                else (rng.standard_normal(shape) * 0.1).astype(np.float32)
            )
        feed = {"x": rng.standard_normal((2, 5, spec.d_model)).astype(np.float32), **params}

        backend = ReferenceBackend()
        np.testing.assert_array_equal(
            backend.execute(restored, feed)["y"], backend.execute(g, feed)["y"]
        )


if __name__ == "__main__":
    unittest.main()
