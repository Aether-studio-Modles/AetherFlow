"""增量注意力解码 + 分页 KV cache 的数值验收。

判据只有一条，但足够硬：**逐位置等于稠密图**。

同一个 ``TransformerSpec``（``vocab=0``）与同一份随机权重下，
``cached_forward``（逐 token、K/V 走分页缓存）的最后隐状态必须与
``build_transformer`` 整段前向的对应位置一致。页宽只改分配粒度，不该改数值。
"""

import unittest
from dataclasses import replace

import numpy as np

from aetherflow import Graph, ReferenceBackend
from aetherflow.models import (
    TransformerSpec,
    build_transformer,
    cached_forward,
    causal_mask_bias,
)

BACKEND = ReferenceBackend()


def _spec(**overrides):
    base = dict(
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        seq=8,
        vocab=0,
        batch=1,
        dropout=0.0,
    )
    base.update(overrides)
    return TransformerSpec(**base)


def _params(spec, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name, shape in spec.param_shapes.items():
        if name == "pos":
            out[name] = np.zeros(shape, dtype=np.float32)
        elif name.endswith("_g"):
            out[name] = np.ones(shape, dtype=np.float32)
        elif name.endswith("_b"):
            out[name] = np.zeros(shape, dtype=np.float32)
        else:
            fan_in = shape[0] if shape else 1
            out[name] = (
                rng.standard_normal(shape) / np.sqrt(fan_in)
            ).astype(np.float32)
    return out


def _dense(xs, params, spec):
    seq = xs.shape[0]
    dense_spec = replace(spec, seq=seq)
    g = Graph("dense_reference")
    inputs, out = build_transformer(g, dense_spec)
    g.output(out, "y")
    feed = {
        inputs["x"].name: xs[None],
        inputs["pos"].name: np.zeros((1, seq, spec.d_model), dtype=np.float32),
        inputs["mask"].name: causal_mask_bias(seq),
    }
    for name in dense_spec.param_shapes:
        if name == "pos":
            continue  # pos 已按实际长度喂入
        feed[inputs[name].name] = params[name]
    return BACKEND.execute(g, feed)["y"]


class TestCachedForward(unittest.TestCase):
    def _check(self, spec, steps, block_size):
        rng = np.random.default_rng(steps * 10 + block_size)
        xs = rng.standard_normal((steps, spec.d_model)).astype(np.float32) * 0.5
        params = _params(spec)
        want = _dense(xs, params, spec)
        got = cached_forward(BACKEND, params, spec, xs, block_size=block_size)
        self.assertEqual(got.shape, (1, steps, spec.d_model))
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)

    def test_matches_dense_for_many_page_widths(self):
        spec = _spec()
        for steps in (1, 3, 5):
            for block_size in (1, 2, 3, 4, 8):
                with self.subTest(steps=steps, block_size=block_size):
                    self._check(spec, steps, block_size)

    def test_matches_dense_with_fused_attention(self):
        spec = _spec(fused_attention=True)
        self._check(spec, 4, 3)

    def test_single_layer_and_single_head(self):
        spec = _spec(num_layers=1, num_heads=1, d_model=4, d_ff=8)
        self._check(spec, 5, 2)

    def test_rejections(self):
        spec = _spec()
        params = _params(spec)
        xs = np.zeros((3, spec.d_model), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "vocab=0"):
            cached_forward(BACKEND, params, _spec(vocab=16), xs)
        with self.assertRaisesRegex(ValueError, "dropout"):
            cached_forward(BACKEND, params, _spec(dropout=0.1), xs)
        with self.assertRaisesRegex(ValueError, "rng_input"):
            cached_forward(BACKEND, params, _spec(rng_input="key"), xs)
        with self.assertRaisesRegex(ValueError, "batch=1"):
            cached_forward(BACKEND, params, _spec(batch=2), xs)
        with self.assertRaisesRegex(TypeError, "batch"):
            cached_forward(BACKEND, params, _spec(batch="B"), xs)
        with self.assertRaisesRegex(ValueError, "xs 形状"):
            cached_forward(BACKEND, params, spec, np.zeros((3, 3), np.float32))
        with self.assertRaisesRegex(ValueError, "至少要有一个位置"):
            cached_forward(BACKEND, params, spec, np.zeros((0, spec.d_model), np.float32))
        with self.assertRaisesRegex(ValueError, "block_size"):
            cached_forward(BACKEND, params, spec, xs, block_size=0)
        with self.assertRaisesRegex(TypeError, "block_size"):
            cached_forward(BACKEND, params, spec, xs, block_size=True)
        with self.assertRaisesRegex(KeyError, "缺少权重"):
            cached_forward(BACKEND, {}, spec, xs)


if __name__ == "__main__":
    unittest.main()
