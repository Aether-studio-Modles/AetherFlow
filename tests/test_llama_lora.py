"""LoRA adapter 的自证测试。

三条判据：

1. **恒等性**：``lora_b`` 为零时，带 adapter 的图与基座逐位相同——旁路必须
   真的乘出 0，而不是"差不多为 0"。
2. **冻结**：``Trainer(trainable=...)`` 跑若干步后，基座参数逐位不变，只有
   adapter 被优化器改写。
3. **装载边界**：基座权重文件里没有 adapter，``load_llama_weights(skip=...)``
   既不能要求文件提供它，也不能覆盖它；忘了 skip 必须显式报错。
"""

from __future__ import annotations

import unittest

import numpy as np

from aetherflow.backends.reference_backend import ReferenceBackend
from aetherflow.ir.types import f32, i32
from aetherflow.models import (
    LlamaSpec,
    LoraSpec,
    build_llama,
    llama_param_shapes,
    llama_param_transposes,
    load_llama_weights,
    lora_param_shapes,
    masked_lm_loss_indexed,
)
from aetherflow.train import Adam, ParameterStore, Trainer
from aetherflow.weights import SafeTensorsError


def _spec(**overrides) -> LlamaSpec:
    base = dict(
        vocab_size=97,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=64,
        seq=8,
        batch=2,
    )
    base.update(overrides)
    return LlamaSpec(**base)


def _loss_graph(spec: LlamaSpec, lora: LoraSpec):
    g, _ = build_llama(spec, lora=lora)
    targets = g.input("targets", i32[spec.batch, spec.seq])
    mask = g.input("mask", f32[spec.batch, spec.seq])
    g.output(masked_lm_loss_indexed(g, g.outputs[0], targets, mask), "loss")
    return g


def _feed(spec: LlamaSpec, seed: int = 0):
    rng = np.random.default_rng(seed)
    tokens = rng.integers(1, spec.vocab_size, size=(spec.batch, spec.seq)).astype(np.int32)
    labels = rng.integers(0, spec.vocab_size, size=(spec.batch, spec.seq))
    positions = np.arange(spec.batch)[:, None] * spec.seq + np.arange(spec.seq)[None, :]
    targets = (positions * spec.vocab_size + labels).astype(np.int32)
    return {
        "tokens": tokens,
        "targets": targets,
        "mask": np.ones((spec.batch, spec.seq), dtype=np.float32),
    }


def _base_params(spec: LlamaSpec, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        name: (rng.standard_normal(shape) * 0.05).astype(np.float32)
        for name, shape in llama_param_shapes(spec).items()
    }


class LoraShapesTest(unittest.TestCase):
    def test_default_targets_are_q_and_v(self):
        spec = _spec()
        shapes = lora_param_shapes(spec, LoraSpec(rank=4))
        self.assertEqual(len(shapes), spec.num_layers * 2 * 2)
        for i in range(spec.num_layers):
            q_stem = f"model.layers.{i}.self_attn.q_proj"
            v_stem = f"model.layers.{i}.self_attn.v_proj"
            self.assertEqual(shapes[f"{q_stem}.lora_a"], (spec.d_model, 4))
            self.assertEqual(shapes[f"{q_stem}.lora_b"], (4, spec.d_model))
            self.assertEqual(shapes[f"{v_stem}.lora_a"], (spec.d_model, 4))
            self.assertEqual(shapes[f"{v_stem}.lora_b"], (4, spec.kv_dim))
            self.assertNotIn(f"model.layers.{i}.mlp.gate_proj.lora_a", shapes)

    def test_all_targets_follow_each_projection_shape(self):
        spec = _spec()
        lora = LoraSpec(
            rank=3,
            targets=("q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"),
        )
        shapes = lora_param_shapes(spec, lora)
        kv = spec.kv_dim
        self.assertEqual(shapes["model.layers.0.self_attn.k_proj.lora_b"], (3, kv))
        self.assertEqual(shapes["model.layers.0.self_attn.o_proj.lora_a"], (spec.d_model, 3))
        self.assertEqual(shapes["model.layers.0.mlp.down_proj.lora_a"], (spec.d_ff, 3))
        self.assertEqual(shapes["model.layers.0.mlp.down_proj.lora_b"], (3, spec.d_model))

    def test_targets_are_validated(self):
        with self.assertRaises(ValueError):
            LoraSpec(targets=("q_proj", "nope"))
        with self.assertRaises(ValueError):
            LoraSpec(rank=0)
        with self.assertRaises(ValueError):
            LoraSpec(alpha=0.0)
        with self.assertRaises(ValueError):
            LoraSpec(targets=("q_proj", "q_proj"))


class LoraForwardTest(unittest.TestCase):
    def test_zero_b_matches_base(self):
        spec = _spec()
        lora = LoraSpec(rank=4, alpha=8.0)
        base_graph, _ = build_llama(spec)
        lora_graph, _ = build_llama(spec, lora=lora)

        base = _base_params(spec, seed=1)
        rng = np.random.default_rng(2)
        adapter = {
            name: (rng.standard_normal(shape) * 0.05).astype(np.float32)
            for name, shape in lora_param_shapes(spec, lora).items()
        }
        for name in adapter:
            if name.endswith(".lora_b"):
                adapter[name][...] = 0.0

        feed = _feed(spec, seed=3)
        backend = ReferenceBackend()
        expected = backend.execute(base_graph, {**base, **feed})["logits"]
        actual = backend.execute(lora_graph, {**base, **adapter, **feed})["logits"]
        self.assertTrue(np.array_equal(expected, actual))

    def test_nonzero_b_changes_output(self):
        spec = _spec()
        lora = LoraSpec(rank=4, alpha=8.0)
        base_graph, _ = build_llama(spec)
        lora_graph, _ = build_llama(spec, lora=lora)

        base = _base_params(spec, seed=1)
        rng = np.random.default_rng(2)
        adapter = {
            name: (rng.standard_normal(shape) * 0.05).astype(np.float32)
            for name, shape in lora_param_shapes(spec, lora).items()
        }

        feed = _feed(spec, seed=3)
        backend = ReferenceBackend()
        expected = backend.execute(base_graph, {**base, **feed})["logits"]
        actual = backend.execute(lora_graph, {**base, **adapter, **feed})["logits"]
        self.assertFalse(np.allclose(expected, actual))


class LoraTrainerTest(unittest.TestCase):
    def test_frozen_base_stays_bitwise_and_adapter_moves(self):
        spec = _spec()
        lora = LoraSpec(rank=4, alpha=8.0)
        graph = _loss_graph(spec, lora)
        store = ParameterStore.from_graph(
            graph, exclude=("tokens", "targets", "mask"), seed=5
        )
        for name in store.names:
            if name.endswith(".lora_b"):
                store[name][...] = 0.0

        adapter = tuple(name for name in store.names if ".lora_" in name)
        base = tuple(name for name in store.names if ".lora_" not in name)
        frozen_before = {name: store[name].copy() for name in base}
        adapter_before = {name: store[name].copy() for name in adapter}

        trainer = Trainer(
            graph,
            store,
            Adam({name: store[name] for name in adapter}, lr=1e-2),
            output="loss",
            trainable=adapter,
        )
        for _ in range(2):
            loss = trainer.step(_feed(spec, seed=7))
            self.assertTrue(np.isfinite(loss))

        for name in base:
            self.assertTrue(
                np.array_equal(store[name], frozen_before[name]),
                f"基座参数 {name} 被改动了",
            )
        for name in adapter:
            self.assertFalse(
                np.array_equal(store[name], adapter_before[name]),
                f"adapter 参数 {name} 没有更新",
            )

    def test_trainable_must_match_optimizer(self):
        spec = _spec()
        lora = LoraSpec(rank=2)
        graph = _loss_graph(spec, lora)
        store = ParameterStore.from_graph(
            graph, exclude=("tokens", "targets", "mask"), seed=5
        )
        adapter = tuple(name for name in store.names if ".lora_" in name)
        with self.assertRaises(ValueError):
            Trainer(
                graph,
                store,
                Adam(store.arrays(), lr=1e-3),
                output="loss",
                trainable=adapter,
            )
        with self.assertRaises(KeyError):
            Trainer(
                graph,
                store,
                Adam({name: store[name] for name in adapter}, lr=1e-3),
                output="loss",
                trainable=adapter + ("no.such.param",),
            )
        with self.assertRaises(ValueError):
            Trainer(
                graph,
                store,
                Adam({name: store[name] for name in adapter}, lr=1e-3),
                output="loss",
                trainable=(),
            )


class LoraLoadTest(unittest.TestCase):
    def test_skip_keeps_adapter_and_loads_base(self):
        spec = _spec()
        lora = LoraSpec(rank=4)
        graph, _ = build_llama(spec, lora=lora)
        store = ParameterStore.from_graph(
            graph, exclude=("tokens",), seed=9
        )
        adapter = tuple(name for name in store.names if ".lora_" in name)
        adapter_before = {name: store[name].copy() for name in adapter}

        transposed = llama_param_transposes(spec)
        tensors = {
            name: np.ones(shape, dtype=np.float32).T
            if name in transposed
            else np.ones(shape, dtype=np.float32)
            for name, shape in llama_param_shapes(spec).items()
        }
        unused = load_llama_weights(spec, store, tensors, skip=adapter)
        self.assertEqual(unused, ())
        for name in llama_param_shapes(spec):
            self.assertTrue(np.all(store[name] == 1.0), name)
        for name in adapter:
            self.assertTrue(np.array_equal(store[name], adapter_before[name]), name)

    def test_forgetting_skip_is_an_error(self):
        spec = _spec()
        lora = LoraSpec(rank=4)
        graph, _ = build_llama(spec, lora=lora)
        store = ParameterStore.from_graph(
            graph, exclude=("tokens",), seed=9
        )
        tensors = {
            name: np.ones(shape, dtype=np.float32)
            for name, shape in llama_param_shapes(spec).items()
        }
        with self.assertRaises(SafeTensorsError):
            load_llama_weights(spec, store, tensors)
        with self.assertRaises(SafeTensorsError):
            load_llama_weights(spec, store, tensors, skip=("no.such.param",))


if __name__ == "__main__":
    unittest.main()
