"""融合 attention 的语义与反向验收。

三条判据：

1. **前向**与一份独立写的 numpy 实现（``softmax(q k^T * scale [+ mask]) @ v``）
   逐点一致，覆盖批维、交叉注意力（S != T）、加性 mask 与 dropout。
2. **反向**与中心差分一致。用 f64 构图：差分的噪声随精度平方增长，f32 下
   "规则写错"和"精度不够"会混在一起。
3. **构件层**：``fused_attention=True`` 的 Block / decoder 与手工拼的
   ``matmul + softmax + dropout`` 逐位对齐——融合必须是实现替换，
   不能夹带语义漂移。
"""

import unittest
from dataclasses import replace

import numpy as np

from aetherflow import (
    Graph,
    ReferenceBackend,
    backward,
    f32,
    f64,
    verify,
    verify_portable_only,
)
from aetherflow.models import (
    BlockSpec,
    TransformerSpec,
    build_transformer,
    build_transformer_block,
    causal_mask_bias,
)

BACKEND = ReferenceBackend()


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------


def _run(graph, feed):
    return BACKEND.execute(graph, feed)


def _numpy_attention(q, k, v, scale=1.0, mask=None):
    """独立写的语义定义：逐行实现，不借用被测代码的任何中间量。"""
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    if mask is not None:
        scores = scores + mask
    m = np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores - m)
    p = e / np.sum(e, axis=-1, keepdims=True)
    return np.matmul(p, v)


def _numeric_grad(graph, feed, name, eps=1e-6):
    arr = feed[name]
    flat = arr.reshape(-1)
    grad = np.zeros(flat.size, dtype=np.float64)
    for i in range(flat.size):
        orig = float(flat[i])
        flat[i] = orig + eps
        hi = float(np.sum(list(_run(graph, feed).values())[0]))
        flat[i] = orig - eps
        lo = float(np.sum(list(_run(graph, feed).values())[0]))
        grad[i] = (hi - lo) / (2 * eps)
        flat[i] = orig
    return grad.reshape(arr.shape)


def _analytic_grad(graph, feed, names):
    bwd = backward(graph, wrt=[graph.find(n) for n in names])
    seed = "d" + graph.output_names[0]
    full = dict(feed)
    full[seed] = np.ones(
        tuple(int(d) for d in graph.outputs[0].type.shape), dtype=np.float64
    )
    out = _run(bwd, full)
    return [out[f"d{n}"] for n in names]


# --------------------------------------------------------------------------
# 前向
# --------------------------------------------------------------------------


class TestAttentionForward(unittest.TestCase):
    def _build(self, q_shape, k_shape, v_shape, *, mask_shape=None, scale=1.0,
               dropout=0.0, seed=0, dtype=f64):
        g = Graph("attn")
        q = g.input("q", dtype[q_shape])
        k = g.input("k", dtype[k_shape])
        v = g.input("v", dtype[v_shape])
        args = [q, k, v]
        if mask_shape is not None:
            args.append(g.input("mask", dtype[mask_shape]))
        g.output(g.attention(*args, scale=scale, dropout=dropout, seed=seed), "y")
        return g

    def _feed(self, seed=0, **extra):
        rng = np.random.default_rng(seed)
        feed = {
            "q": rng.standard_normal((2, 2, 3, 2)),
            "k": rng.standard_normal((2, 2, 3, 2)),
            "v": rng.standard_normal((2, 2, 3, 2)),
        }
        feed.update(extra)
        return feed

    def test_matches_numpy(self):
        g = self._build((2, 2, 3, 2), (2, 2, 3, 2), (2, 2, 3, 2), scale=0.5)
        feed = self._feed(1)
        want = _numpy_attention(feed["q"], feed["k"], feed["v"], scale=0.5)
        np.testing.assert_allclose(_run(g, feed)["y"], want, rtol=1e-12)

    def test_cross_attention_lengths(self):
        """q 的序列长度与 k/v 可以不同，输出长度跟着 q 走。"""
        g = self._build((2, 2, 3, 2), (2, 2, 5, 2), (2, 2, 5, 2), scale=0.5)
        rng = np.random.default_rng(2)
        feed = {
            "q": rng.standard_normal((2, 2, 3, 2)),
            "k": rng.standard_normal((2, 2, 5, 2)),
            "v": rng.standard_normal((2, 2, 5, 2)),
        }
        want = _numpy_attention(feed["q"], feed["k"], feed["v"], scale=0.5)
        self.assertEqual(want.shape, (2, 2, 3, 2))
        np.testing.assert_allclose(_run(g, feed)["y"], want, rtol=1e-12)

    def test_value_dim_may_differ(self):
        g = self._build((2, 2, 3, 2), (2, 2, 3, 2), (2, 2, 3, 4), scale=0.5)
        rng = np.random.default_rng(3)
        feed = {
            "q": rng.standard_normal((2, 2, 3, 2)),
            "k": rng.standard_normal((2, 2, 3, 2)),
            "v": rng.standard_normal((2, 2, 3, 4)),
        }
        want = _numpy_attention(feed["q"], feed["k"], feed["v"], scale=0.5)
        self.assertEqual(want.shape, (2, 2, 3, 4))
        np.testing.assert_allclose(_run(g, feed)["y"], want, rtol=1e-12)

    def test_zero_mask_changes_nothing(self):
        plain_g = self._build((2, 2, 3, 2), (2, 2, 3, 2), (2, 2, 3, 2), scale=0.5)
        masked_g = self._build(
            (2, 2, 3, 2), (2, 2, 3, 2), (2, 2, 3, 2),
            mask_shape=(2, 2, 3, 3), scale=0.5,
        )
        feed = self._feed(4)
        masked_feed = dict(feed, mask=np.zeros((2, 2, 3, 3)))
        np.testing.assert_allclose(
            _run(masked_g, masked_feed)["y"], _run(plain_g, feed)["y"], rtol=1e-14
        )

    def test_causal_mask_matches_numpy(self):
        g = self._build(
            (2, 2, 3, 2), (2, 2, 3, 2), (2, 2, 3, 2),
            mask_shape=(1, 1, 3, 3), scale=0.5,
        )
        feed = self._feed(5, mask=causal_mask_bias(3).astype(np.float64))
        want = _numpy_attention(
            feed["q"], feed["k"], feed["v"], scale=0.5, mask=feed["mask"]
        )
        np.testing.assert_allclose(_run(g, feed)["y"], want, rtol=1e-12)

    def test_causal_mask_blocks_future_values(self):
        """改最后一个位置的 v，前面位置的输出必须逐位不变。"""
        g = self._build(
            (2, 2, 3, 2), (2, 2, 3, 2), (2, 2, 3, 2),
            mask_shape=(1, 1, 3, 3), scale=0.5,
        )
        feed = self._feed(6, mask=causal_mask_bias(3).astype(np.float64))
        base = _run(g, feed)["y"]
        perturbed = dict(feed)
        perturbed["v"] = feed["v"].copy()
        perturbed["v"][:, :, -1] += 5.0
        np.testing.assert_array_equal(_run(g, perturbed)["y"][:, :, :-1], base[:, :, :-1])

    def test_dropout_is_deterministic(self):
        g = self._build((2, 2, 3, 2), (2, 2, 3, 2), (2, 2, 3, 2), dropout=0.5, seed=7)
        feed = self._feed(7)
        np.testing.assert_array_equal(_run(g, feed)["y"], _run(g, feed)["y"])

    def test_dropout_changes_result(self):
        plain = self._build((2, 2, 3, 2), (2, 2, 3, 2), (2, 2, 3, 2))
        drop = self._build(
            (2, 2, 3, 2), (2, 2, 3, 2), (2, 2, 3, 2), dropout=0.5, seed=7
        )
        feed = self._feed(8)
        self.assertFalse(np.allclose(_run(plain, feed)["y"], _run(drop, feed)["y"]))

    def test_symbolic_batch_builds_and_differentiates(self):
        """批维是符号时也能构图、也能展开反向：attention 的反向没有广播归约。"""
        g = Graph("attn_sym")
        q = g.input("q", f32["B", 2, 3, 2])
        k = g.input("k", f32["B", 2, 3, 2])
        v = g.input("v", f32["B", 2, 3, 2])
        y = g.attention(q, k, v, scale=0.5)
        self.assertEqual(y.type, f32["B", 2, 3, 2])
        g.output(y, "y")
        bwd = backward(g)
        self.assertEqual(verify(bwd), [])


# --------------------------------------------------------------------------
# 反向
# --------------------------------------------------------------------------


class TestAttentionBackward(unittest.TestCase):
    def _build(self, q_shape, k_shape, v_shape, *, mask_shape=None, scale=0.5,
               dropout=0.0, seed=0):
        g = Graph("attn")
        q = g.input("q", f64[q_shape])
        k = g.input("k", f64[k_shape])
        v = g.input("v", f64[v_shape])
        args = [q, k, v]
        if mask_shape is not None:
            args.append(g.input("mask", f64[mask_shape]))
        g.output(g.attention(*args, scale=scale, dropout=dropout, seed=seed), "y")
        return g

    def test_gradient_matches_numeric(self):
        g = self._build((2, 2, 3, 2), (2, 2, 3, 2), (2, 2, 3, 2))
        rng = np.random.default_rng(10)
        feed = {
            "q": rng.standard_normal((2, 2, 3, 2)),
            "k": rng.standard_normal((2, 2, 3, 2)),
            "v": rng.standard_normal((2, 2, 3, 2)),
        }
        dq, dk, dv = _analytic_grad(g, feed, ["q", "k", "v"])
        for name, got in (("q", dq), ("k", dk), ("v", dv)):
            np.testing.assert_allclose(got, _numeric_grad(g, feed, name), atol=1e-7)

    def test_cross_attention_gradient(self):
        g = self._build((2, 2, 3, 2), (2, 2, 4, 2), (2, 2, 4, 2))
        rng = np.random.default_rng(11)
        feed = {
            "q": rng.standard_normal((2, 2, 3, 2)),
            "k": rng.standard_normal((2, 2, 4, 2)),
            "v": rng.standard_normal((2, 2, 4, 2)),
        }
        dq, dk, dv = _analytic_grad(g, feed, ["q", "k", "v"])
        for name, got in (("q", dq), ("k", dk), ("v", dv)):
            np.testing.assert_allclose(got, _numeric_grad(g, feed, name), atol=1e-7)

    def test_mask_with_matching_shape_gets_gradient(self):
        g = self._build(
            (2, 2, 3, 2), (2, 2, 3, 2), (2, 2, 3, 2), mask_shape=(2, 2, 3, 3)
        )
        rng = np.random.default_rng(12)
        feed = {
            "q": rng.standard_normal((2, 2, 3, 2)),
            "k": rng.standard_normal((2, 2, 3, 2)),
            "v": rng.standard_normal((2, 2, 3, 2)),
            "mask": rng.standard_normal((2, 2, 3, 3)) * 0.1,
        }
        dmask, = _analytic_grad(g, feed, ["mask"])
        np.testing.assert_allclose(dmask, _numeric_grad(g, feed, "mask"), atol=1e-7)

    def test_broadcast_mask_gradient_sums_over_batch(self):
        """典型因果掩码（1,1,S,S）广播到 (B,H,S,S)：梯度要沿广播轴求和。

        旧规则用"形状完全相同才给梯度"，把这个输入当成常量返回 None；但加性
        偏置的数学梯度是确定的，能不能训练该由调用者决定（不放进优化器即可），
        不该由算子替它判断角色。
        """
        g = self._build(
            (2, 2, 3, 2), (2, 2, 3, 2), (2, 2, 3, 2), mask_shape=(1, 1, 3, 3)
        )
        rng = np.random.default_rng(13)
        feed = {
            "q": rng.standard_normal((2, 2, 3, 2)),
            "k": rng.standard_normal((2, 2, 3, 2)),
            "v": rng.standard_normal((2, 2, 3, 2)),
            "mask": causal_mask_bias(3).astype(np.float64),
        }
        dmask, = _analytic_grad(g, feed, ["mask"])
        self.assertEqual(dmask.shape, (1, 1, 3, 3))
        np.testing.assert_allclose(dmask, _numeric_grad(g, feed, "mask"), atol=1e-7)

        # 默认 wrt 现在包含 mask，不该再抛 NoGradientError
        self.assertIn("dmask", backward(g).output_names)

        dq, dk, dv = _analytic_grad(g, feed, ["q", "k", "v"])
        for name, got in (("q", dq), ("k", dk), ("v", dv)):
            np.testing.assert_allclose(got, _numeric_grad(g, feed, name), atol=1e-7)

    def test_dropout_gradient_matches_numeric(self):
        g = self._build(
            (2, 2, 3, 2), (2, 2, 3, 2), (2, 2, 3, 2), dropout=0.5, seed=3
        )
        rng = np.random.default_rng(14)
        feed = {
            "q": rng.standard_normal((2, 2, 3, 2)),
            "k": rng.standard_normal((2, 2, 3, 2)),
            "v": rng.standard_normal((2, 2, 3, 2)),
        }
        dq, dk, dv = _analytic_grad(g, feed, ["q", "k", "v"])
        for name, got in (("q", dq), ("k", dk), ("v", dv)):
            np.testing.assert_allclose(got, _numeric_grad(g, feed, name), atol=1e-7)


# --------------------------------------------------------------------------
# 构图期校验
# --------------------------------------------------------------------------


class TestAttentionValidation(unittest.TestCase):
    def _inputs(self, q_shape=(2, 2, 3, 2), k_shape=(2, 2, 3, 2), v_shape=(2, 2, 3, 2)):
        g = Graph("bad")
        return g, g.input("q", f32[q_shape]), g.input("k", f32[k_shape]), g.input("v", f32[v_shape])

    def test_batch_dims_must_match(self):
        g, q, k, v = self._inputs(q_shape=(2, 2, 3, 2), k_shape=(3, 2, 3, 2))
        with self.assertRaises(ValueError):
            g.attention(q, k, v)

    def test_head_dim_must_match(self):
        g, q, k, v = self._inputs(k_shape=(2, 2, 3, 4))
        with self.assertRaises(ValueError):
            g.attention(q, k, v)

    def test_kv_length_must_match(self):
        g, q, k, v = self._inputs(v_shape=(2, 2, 4, 2))
        with self.assertRaises(ValueError):
            g.attention(q, k, v)

    def test_dtype_must_match(self):
        g = Graph("bad")
        q = g.input("q", f32[2, 2, 3, 2])
        k = g.input("k", f64[2, 2, 3, 2])
        v = g.input("v", f32[2, 2, 3, 2])
        with self.assertRaises(TypeError):
            g.attention(q, k, v)

    def test_mask_dtype_must_match(self):
        g, q, k, v = self._inputs()
        mask = g.input("mask", f64[1, 1, 3, 3])
        with self.assertRaises(TypeError):
            g.attention(q, k, v, mask)

    def test_mask_last_dim_must_match(self):
        g, q, k, v = self._inputs()
        mask = g.input("mask", f32[1, 1, 3, 4])
        with self.assertRaises(ValueError):
            g.attention(q, k, v, mask)

    def test_dropout_range(self):
        g, q, k, v = self._inputs()
        with self.assertRaises(ValueError):
            g.attention(q, k, v, dropout=1.0)

    def test_input_count(self):
        g, q, k, v = self._inputs()
        with self.assertRaises(ValueError):
            g.attention(q, k)


# --------------------------------------------------------------------------
# 构件层：融合必须与手工拼的图逐位一致
# --------------------------------------------------------------------------


def _fill(inputs, seed):
    rng = np.random.default_rng(seed)
    feed = {}
    for name, value in inputs.items():
        shape = tuple(int(d) for d in value.type.shape)
        feed[name] = (rng.standard_normal(shape) * 0.3).astype(np.float32)
    return feed


class TestFusedBlock(unittest.TestCase):
    def _build(self, fused, dropout=0.0):
        spec = BlockSpec(
            d_model=8, num_heads=2, d_ff=8, batch=2, seq=3,
            dropout=dropout, dropout_seed=11, fused_attention=fused,
        )
        g = Graph("blk")
        inputs, out = build_transformer_block(g, spec)
        g.output(out, "y")
        return g, inputs, spec

    def _paired_feed(self, inputs_plain, spec, seed=20):
        feed = _fill(inputs_plain, seed)
        feed["x"] = (
            np.random.default_rng(seed).standard_normal((2, 3, spec.d_model)) * 0.3
        ).astype(np.float32)
        feed["scale"] = np.array([1.0 / np.sqrt(spec.head_dim)], dtype=np.float32)
        return feed

    def test_fused_matches_plain(self):
        g_plain, inputs_plain, spec = self._build(False)
        g_fused, inputs_fused, _ = self._build(True)
        feed = self._paired_feed(inputs_plain, spec)
        fused_feed = {k: v for k, v in feed.items() if k != "scale"}
        np.testing.assert_allclose(
            _run(g_fused, fused_feed)["y"], _run(g_plain, feed)["y"],
            rtol=1e-5, atol=1e-6,
        )

    def test_fused_dropout_matches_plain(self):
        g_plain, inputs_plain, spec = self._build(False, dropout=0.25)
        g_fused, _, _ = self._build(True, dropout=0.25)
        feed = self._paired_feed(inputs_plain, spec, seed=21)
        fused_feed = {k: v for k, v in feed.items() if k != "scale"}
        np.testing.assert_allclose(
            _run(g_fused, fused_feed)["y"], _run(g_plain, feed)["y"],
            rtol=1e-5, atol=1e-6,
        )

    def test_fused_has_no_scale_parameter(self):
        _, inputs_fused, _ = self._build(True)
        self.assertNotIn("scale", inputs_fused)

    def test_fused_is_portable(self):
        g, _, _ = self._build(True)
        self.assertEqual(verify(g), [])
        self.assertEqual(verify_portable_only(g), [])


class TestFusedDecoder(unittest.TestCase):
    def _spec(self, fused, dropout=0.0):
        return TransformerSpec(
            d_model=8, num_heads=2, d_ff=8, num_layers=2, seq=3, vocab=0,
            batch=2, dropout=dropout, dropout_seed=13, fused_attention=fused,
        )

    def _build(self, fused, dropout=0.0):
        spec = self._spec(fused, dropout)
        g = Graph("lm")
        inputs, out = build_transformer(g, spec)
        g.output(out, "y")
        return g, inputs, spec

    def _feed(self, inputs, seed=30):
        feed = _fill(inputs, seed)
        feed["x"] = (
            np.random.default_rng(seed).standard_normal((2, 3, self.d_model)) * 0.3
        ).astype(np.float32)
        feed["mask"] = causal_mask_bias(3)
        return feed

    d_model = 8

    def test_fused_matches_plain(self):
        g_plain, inputs_plain, _ = self._build(False)
        g_fused, _, _ = self._build(True)
        feed = self._feed(inputs_plain)
        np.testing.assert_allclose(
            _run(g_fused, feed)["y"], _run(g_plain, feed)["y"],
            rtol=1e-5, atol=1e-5,
        )

    def test_fused_gradients_match_plain(self):
        g_plain, inputs_plain, spec = self._build(False)
        g_fused, inputs_fused, _ = self._build(True)
        feed = self._feed(inputs_plain)
        names = [n for n in spec.param_shapes if n in inputs_plain]

        def grads(graph, inputs):
            bwd = backward(graph, wrt=[inputs[n] for n in names])
            bfeed = dict(feed)
            bfeed["dy"] = np.ones((2, 3, spec.d_model), dtype=np.float32)
            out = _run(bwd, bfeed)
            return [out[f"d{n}"] for n in names]

        got_plain = grads(g_plain, inputs_plain)
        got_fused = grads(g_fused, inputs_fused)
        for name, a, b in zip(names, got_plain, got_fused):
            np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-6, err_msg=name)


if __name__ == "__main__":
    unittest.main()
