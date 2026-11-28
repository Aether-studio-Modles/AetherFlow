"""Llama / Qwen2 构建器的自证测试。

判据不是"图能跑通"，而是**逐点等于一份独立写的 HF 口径参考实现**：
参考实现用 numpy 的 float64 重写 RoPE、GQA、SwiGLU 与因果 mask，不复用
``aetherflow`` 的任何构件——如果构建器把"前后两半配对"写成"相邻两维配对"，
或把 GQA 的组顺序搞反，这里会直接以数值差暴露。
"""

from __future__ import annotations

import unittest

import numpy as np

from aetherflow.backends.reference_backend import ReferenceBackend
from aetherflow.ir.types import f32
from aetherflow.models.llama import (
    LlamaSpec,
    build_llama,
    llama_inv_freq,
    llama_param_shapes,
    llama_param_transposes,
)
from aetherflow.models.transformer import causal_mask_bias


def _small(**overrides) -> LlamaSpec:
    base = dict(
        vocab_size=97,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=64,
        seq=8,
        batch=2,
        qkv_bias=True,
    )
    base.update(overrides)
    return LlamaSpec(**base)


def _random_params(spec: LlamaSpec, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    params = {}
    for name, shape in llama_param_shapes(spec).items():
        params[name] = (rng.standard_normal(shape) * 0.1).astype(np.float32)
    return params


def _rmsnorm(x, w, eps):
    scale = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return x * scale * w


def _rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _repeat_kv(x, group):
    return np.repeat(x, group, axis=1)


def _attention(q, k, v, scale, mask):
    scores = np.einsum("bhqd,bhkd->bhqk", q, k) * scale
    if mask is not None:
        scores = scores + mask
    scores = scores - scores.max(-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", probs, v)


def reference_forward(spec: LlamaSpec, params: dict, tokens: np.ndarray) -> np.ndarray:
    """HF 口径的独立实现（float64）。"""
    hd = spec.head_dim
    H = spec.num_heads
    KV = spec.num_kv_heads
    G = spec.group_size

    x = params["model.embed_tokens.weight"][tokens].astype(np.float64)

    pos = np.arange(spec.seq, dtype=np.float64)[:, None]
    inv = llama_inv_freq(spec).astype(np.float64)[None, :]
    freqs = pos * inv
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos, sin = np.cos(emb), np.sin(emb)

    mask = causal_mask_bias(spec.seq) if spec.causal else None
    scale = spec.attn_scale

    for i in range(spec.num_layers):
        p = f"model.layers.{i}"
        h = _rmsnorm(x, params[f"{p}.input_layernorm.weight"], spec.norm_eps)

        q = h @ params[f"{p}.self_attn.q_proj.weight"]
        k = h @ params[f"{p}.self_attn.k_proj.weight"]
        v = h @ params[f"{p}.self_attn.v_proj.weight"]
        if spec.qkv_bias:
            q = q + params[f"{p}.self_attn.q_proj.bias"]
            k = k + params[f"{p}.self_attn.k_proj.bias"]
            v = v + params[f"{p}.self_attn.v_proj.bias"]

        b, s = x.shape[:2]
        q = q.reshape(b, s, H, hd).transpose(0, 2, 1, 3)
        k = k.reshape(b, s, KV, hd).transpose(0, 2, 1, 3)
        v = v.reshape(b, s, KV, hd).transpose(0, 2, 1, 3)

        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        if G > 1:
            k = _repeat_kv(k, G)
            v = _repeat_kv(v, G)

        ctx = _attention(q, k, v, scale, mask)
        merged = ctx.transpose(0, 2, 1, 3).reshape(b, s, H * hd)
        x = x + merged @ params[f"{p}.self_attn.o_proj.weight"]

        h = _rmsnorm(x, params[f"{p}.post_attention_layernorm.weight"], spec.norm_eps)
        gate = h @ params[f"{p}.mlp.gate_proj.weight"]
        up = h @ params[f"{p}.mlp.up_proj.weight"]
        x = x + (_silu(gate) * up) @ params[f"{p}.mlp.down_proj.weight"]

    h = _rmsnorm(x, params["model.norm.weight"], spec.norm_eps)
    return h @ params["lm_head.weight"]


class TestSpec(unittest.TestCase):
    def test_rejects_symbolic_seq(self):
        with self.assertRaisesRegex(ValueError, "静态整数"):
            LlamaSpec(seq="S")

    def test_rejects_odd_head_dim(self):
        with self.assertRaisesRegex(ValueError, "偶数"):
            LlamaSpec(d_model=36, num_heads=4)

    def test_rejects_indivisible_heads(self):
        with self.assertRaisesRegex(ValueError, "整除"):
            LlamaSpec(d_model=32, num_heads=3)

    def test_rejects_uneven_gqa_groups(self):
        with self.assertRaisesRegex(ValueError, "整数倍"):
            LlamaSpec(d_model=32, num_heads=4, num_kv_heads=3)

    def test_qwen2_5_0_5b_official_shape(self):
        spec = LlamaSpec.qwen2_5_0_5b()
        self.assertEqual((spec.d_model, spec.num_heads, spec.num_kv_heads), (896, 14, 2))
        self.assertEqual(spec.head_dim, 64)
        self.assertEqual(spec.kv_dim, 128)
        self.assertEqual(spec.group_size, 7)
        self.assertTrue(spec.qkv_bias)


class TestParamLayout(unittest.TestCase):
    def test_linear_weights_are_transposed_for_ir(self):
        spec = _small()
        shapes = llama_param_shapes(spec)
        d = spec.d_model
        self.assertEqual(shapes["model.embed_tokens.weight"], (spec.vocab_size, d))
        self.assertEqual(shapes["model.layers.0.self_attn.q_proj.weight"], (d, d))
        self.assertEqual(
            shapes["model.layers.0.self_attn.k_proj.weight"],
            (d, spec.kv_dim),
        )
        self.assertEqual(shapes["model.layers.0.mlp.gate_proj.weight"], (d, spec.d_ff))
        self.assertEqual(shapes["model.layers.0.mlp.down_proj.weight"], (spec.d_ff, d))
        self.assertEqual(shapes["lm_head.weight"], (d, spec.vocab_size))

    def test_transposes_cover_every_linear_but_not_embedding_or_norm(self):
        spec = _small()
        transposed = llama_param_transposes(spec)
        self.assertIn("lm_head.weight", transposed)
        self.assertIn("model.layers.0.self_attn.o_proj.weight", transposed)
        self.assertNotIn("model.embed_tokens.weight", transposed)
        self.assertNotIn("model.norm.weight", transposed)
        self.assertNotIn("model.layers.0.self_attn.q_proj.bias", transposed)

    def test_graph_inputs_cover_all_params(self):
        spec = _small()
        _, inputs = build_llama(spec)
        self.assertEqual(set(inputs) - {"tokens"}, set(llama_param_shapes(spec)))


class TestForward(unittest.TestCase):
    def test_matches_hf_reference(self):
        spec = _small()
        params = _random_params(spec, seed=1)
        tokens = np.random.default_rng(2).integers(0, spec.vocab_size, size=(2, spec.seq))
        tokens = tokens.astype(np.int32)

        graph, inputs = build_llama(spec)
        feeds = dict(params)
        feeds["tokens"] = tokens
        got = ReferenceBackend().execute(graph, feeds)["logits"]
        want = reference_forward(spec, params, tokens)

        self.assertEqual(got.shape, (2, spec.seq, spec.vocab_size))
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

    def test_mha_path_when_kv_heads_equal_query_heads(self):
        spec = _small(num_kv_heads=4)
        params = _random_params(spec, seed=3)
        tokens = np.zeros((2, spec.seq), dtype=np.int32)
        graph, _ = build_llama(spec)
        got = ReferenceBackend().execute(graph, {**params, "tokens": tokens})["logits"]
        want = reference_forward(spec, params, tokens)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

    def test_without_logits_returns_hidden(self):
        spec = _small()
        graph, _ = build_llama(spec, with_logits=False)
        self.assertEqual(len(graph.outputs), 1)
        self.assertEqual(graph.outputs[0].type, f32[2, spec.seq, spec.d_model])

    def test_causal_mask_hides_future_tokens(self):
        """改动最后一个 token 的输入，不得影响前面位置（含位置 0）的输出。"""
        spec = _small(batch=1)
        params = _random_params(spec, seed=4)
        base = np.zeros((1, spec.seq), dtype=np.int32)
        changed = base.copy()
        changed[0, -1] = 7

        graph, _ = build_llama(spec)
        backend = ReferenceBackend()
        a = backend.execute(graph, {**params, "tokens": base})["logits"]
        b = backend.execute(graph, {**params, "tokens": changed})["logits"]

        np.testing.assert_allclose(a[:, :-1], b[:, :-1], rtol=1e-6, atol=1e-6)
        self.assertGreater(np.abs(a[:, -1] - b[:, -1]).max(), 1e-6)

    def test_no_bias_llama_variant(self):
        spec = _small(qkv_bias=False)
        params = _random_params(spec, seed=5)
        tokens = np.ones((2, spec.seq), dtype=np.int32)
        graph, inputs = build_llama(spec)
        self.assertFalse(any(name.endswith(".bias") for name in inputs))
        got = ReferenceBackend().execute(graph, {**params, "tokens": tokens})["logits"]
        want = reference_forward(spec, params, tokens)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
