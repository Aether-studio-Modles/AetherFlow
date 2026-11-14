"""BF16 的真实数值语义、两字节存储与训练闭环。

四条判据：

- 量化：f32 -> bf16 是 RNE，tie 进位到偶数；inf / nan 保留，f32 最大有限值
  进位成 inf；
- 范围：bf16 借用 f32 的指数，``1e30`` 不会像 f16 那样溢出；
- 执行：bf16 类型的输入与节点输出都被量化，``matmul`` 的结果等于"量化输入
  在 f32 里累加、输出再量化"；
- 训练：``mixed_precision="bf16"`` 能收敛，``master_weights`` 的驻留缓冲
  确实是 2 字节，且不需要（也不接受）``GradScaler``。
"""

import os
import tempfile
import unittest

import numpy as np

from aetherflow import Graph, ReferenceBackend, bf16, f16, f32
from aetherflow.passes import Autocast
from aetherflow.runtime.bf16 import (
    STORAGE_BYTES,
    pack_bf16,
    round_bf16,
    unpack_bf16,
)
from aetherflow.train import Adam, GradScaler, ParameterStore, Trainer, mse

BACKEND = ReferenceBackend()


def _mlp():
    """``softmax(x @ w + b)``：一条"该降 + 该留"的最小链。"""
    g = Graph("mlp")
    x = g.input("x", f32[4, 3])
    w = g.input("w", f32[3, 5])
    b = g.input("b", f32[1, 5])
    z = g.add_node("matmul", [x, w])
    y = g.add_node("softmax", [g.add(z, b)], {"axis": -1})
    g.output(y, "y")
    return g


def _feed():
    rng = np.random.default_rng(0)
    return {
        "x": rng.normal(size=(4, 3)).astype(np.float32),
        "w": rng.normal(size=(3, 5)).astype(np.float32),
        "b": rng.normal(size=(1, 5)).astype(np.float32),
    }


class TestBf16Codec(unittest.TestCase):
    def test_rounding_is_nearest_even(self):
        step = 2.0 ** -7
        cases = {
            1.0: 1.0,                        # 精确值不动
            1.0 + 2.0 ** -8: 1.0,            # 恰好一半，进位到偶数（尾数 0）
            1.0 + 3 * step / 2: 1.0 + 2 * step,  # 恰好一半，进位到偶数（尾数 2）
            1.0 + 2.0 ** -8 + 2.0 ** -20: 1.0 + step,  # 略多于一半，向上
            -1.0 - 2.0 ** -8: -1.0,          # 负数同样按 RNE
        }
        keys = np.array(list(cases), np.float32)
        got = round_bf16(keys)
        np.testing.assert_array_equal(got, np.array(list(cases.values()), np.float32))

    def test_non_finite_and_overflow_follow_ieee(self):
        max_bf16 = 3.3895314e38
        x = np.array([np.inf, -np.inf, np.nan, max_bf16, -max_bf16,
                      np.finfo(np.float32).max], np.float32)
        got = round_bf16(x)
        self.assertTrue(np.isinf(got[0]) and got[0] > 0)
        self.assertTrue(np.isinf(got[1]) and got[1] < 0)
        self.assertTrue(np.isnan(got[2]))
        self.assertEqual(got[3], np.float32(max_bf16))
        self.assertEqual(got[4], np.float32(-max_bf16))
        # 最大有限 f32 在 RNE 下进位成 inf，这是位宽决定的语义。
        self.assertTrue(np.isinf(got[5]))

    def test_nan_payload_survives_quantization(self):
        # 尾数全 1 的 NaN 加上 RNE 偏置会进位到指数甚至回绕，早先会变成 ±0。
        bits = np.array([0x7FFFFFFF, 0xFFFFFFFF, 0x7F80FFFF], np.uint32)
        x = bits.view(np.float32)
        got = round_bf16(x)
        self.assertTrue(np.isnan(got).all())
        # 尾数截断后为 0 的 NaN 补 quiet 位，不能被读成 inf。
        self.assertEqual(got.view(np.uint32)[2], np.uint32(0x7FC00000))
        self.assertTrue(np.isnan(unpack_bf16(pack_bf16(x))).all())

    def test_pack_is_two_bytes_and_roundtrip_is_exact(self):
        x = np.array([[1.0 + 2.0 ** -9, 1e30], [1e-30, -2.5]], np.float32)
        packed = pack_bf16(x)
        self.assertEqual(packed.dtype, np.uint16)
        self.assertEqual(packed.itemsize * packed.size, STORAGE_BYTES * x.size)
        np.testing.assert_array_equal(unpack_bf16(packed), round_bf16(x))

    def test_pack_rejects_non_uint16_bits(self):
        with self.assertRaises(TypeError):
            unpack_bf16(np.array([16256], np.int32))

    def test_bf16_range_is_wider_than_f16(self):
        x = np.array([1e30, 1e-30], np.float32)
        got = round_bf16(x)
        self.assertTrue(np.isfinite(got).all())
        # f16 的指数只有 5 位：大值溢出成 inf，小值下溢成 0。
        with np.errstate(all="ignore"):
            half = x.astype(np.float16)
        self.assertTrue(np.isinf(half[0]))
        self.assertEqual(half[1], np.float16(0))


class TestBf16Execution(unittest.TestCase):
    def test_bf16_input_is_quantized_at_the_boundary(self):
        g = Graph("in")
        x = g.input("x", bf16[2])
        g.output(g.add(x, x), "y")
        out = BACKEND.execute(
            g, {"x": np.array([1.0 + 2.0 ** -9, 1.1], np.float32)}
        )["y"]
        # 1 + 2^-9 低于半个 bf16 间隔，量化成 1.0；1.1 落到最近的格点。
        self.assertEqual(out[0], np.float32(2.0))
        self.assertEqual(float(out[1]), float(np.float32(round_bf16(
            np.array([1.1], np.float32)
        )[0]) * 2))

    def test_cast_to_bf16_keeps_the_ir_label_and_quantizes_values(self):
        g = Graph("cast")
        x = g.input("x", f32[2])
        y = g.cast(x, dtype=bf16)
        g.output(y, "y")
        out = BACKEND.execute(
            g, {"x": np.array([1.0 + 2.0 ** -8, 1.3333], np.float32)}
        )["y"]
        self.assertEqual(y.type.dtype, bf16)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(
            out, round_bf16(np.array([1.0 + 2.0 ** -8, 1.3333], np.float32))
        )

    def test_matmul_rounds_inputs_and_output(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(4, 5)).astype(np.float32)
        b = rng.normal(size=(5, 2)).astype(np.float32)
        g = Graph("gemm")
        x = g.input("a", bf16[4, 5])
        w = g.input("b", bf16[5, 2])
        g.output(g.add_node("matmul", [x, w]), "y")
        got = BACKEND.execute(g, {"a": a, "b": b})["y"]
        want = round_bf16(round_bf16(a) @ round_bf16(b))
        np.testing.assert_array_equal(got, want)

    def test_f32_graph_output_is_restored(self):
        g = Graph("out")
        x = g.input("x", f32[2])
        g.output(g.cast(x, dtype=bf16), "y")
        out = BACKEND.execute(g, {"x": np.array([1.2, 2.4], np.float32)})["y"]
        self.assertEqual(out.dtype, np.float32)


class TestAutocastBf16(unittest.TestCase):
    def test_whitelist_moves_to_bf16_and_sensitive_ops_stay_f32(self):
        out = Autocast(dtype="bf16").run(_mlp())
        dtypes = {n.op.name: [a.type.dtype.value for a in n.inputs] for n in out.nodes}
        self.assertEqual(dtypes["matmul"], ["bf16", "bf16"])
        self.assertIn("f32", dtypes["softmax"])
        self.assertEqual({v.name: v.type.dtype for v in out.inputs}["x"], f32)
        self.assertEqual({name: v.type.dtype for v, name in
                          zip(out.outputs, out.output_names)}["y"], f32)

    def test_forward_matches_f32_within_bf16_precision(self):
        feed = _feed()
        ref = BACKEND.execute(_mlp(), feed)["y"]
        got = BACKEND.execute(Autocast(dtype="bf16").run(_mlp()), feed)["y"]
        self.assertEqual(got.dtype, np.float32)
        np.testing.assert_allclose(got, ref, rtol=5e-2, atol=5e-3)

    def test_low_precision_inputs_declare_bf16(self):
        out = Autocast(dtype="bf16", low_precision_inputs=("w",)).run(_mlp())
        dtypes = {v.name: v.type.dtype for v in out.inputs}
        self.assertEqual(dtypes["w"], bf16)
        self.assertEqual(dtypes["x"], f32)

    def test_is_idempotent(self):
        once = Autocast(dtype="bf16", low_precision_inputs=("w",)).run(_mlp())
        twice = Autocast(dtype="bf16", low_precision_inputs=("w",)).run(once)
        self.assertEqual(
            [(n.op.name, [a.type.dtype for a in n.inputs]) for n in once.nodes],
            [(n.op.name, [a.type.dtype for a in n.inputs]) for n in twice.nodes],
        )


def _lin_graph():
    g = Graph("lin")
    x = g.input("x", f32[8, 4])
    t = g.input("t", f32[8, 1])
    w = g.input("w", f32[4, 1])
    b = g.input("b", f32[1])
    g.output(mse(g, g.add(g.matmul(x, w), b), t), "loss")
    return g


def _build(mixed="bf16", master=True):
    g = _lin_graph()
    store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
    trainer = Trainer(
        g, store, Adam(store.arrays(), lr=0.05),
        mixed_precision=mixed,
        master_weights=master,
    )
    return trainer, store


def _batches(n=240):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 4)).astype(np.float32)
    true_w = np.array([[1.0], [-2.0], [0.5], [3.0]], np.float32)
    t = (x @ true_w + 0.7).astype(np.float32)
    return [
        {"x": x[(i * 8) % 32:(i * 8) % 32 + 8],
         "t": t[(i * 8) % 32:(i * 8) % 32 + 8]}
        for i in range(n)
    ]


class TestBf16Trainer(unittest.TestCase):
    def test_master_weights_reside_in_two_bytes(self):
        trainer, store = _build()
        dtypes = {v.name: v.type.dtype for v in trainer.graph.inputs}
        self.assertEqual(dtypes["w"], bf16)
        self.assertEqual(dtypes["x"], f32)
        for name in store.names:
            self.assertEqual(store[name].dtype, np.float32)
            self.assertEqual(trainer._resident[name].dtype, np.uint16)
            self.assertEqual(trainer._resident[name].nbytes * 2, store[name].nbytes)
        self.assertEqual(trainer.optimizer.params["w"].dtype, np.float32)

    def test_training_converges_and_matches_f32_within_bf16_precision(self):
        mixed, store = _build()
        ref, ref_store = _build(mixed=None, master=False)
        for batch in _batches():
            mixed.step(batch)
            ref.step(batch)

        for name in store.names:
            scale = max(float(np.abs(ref_store[name]).max()), 1e-12)
            rel = float(np.abs(store[name] - ref_store[name]).max() / scale)
            self.assertLess(rel, 2e-2, f"{name} 的权重相对误差 {rel:.3e}")
        np.testing.assert_allclose(
            store["w"].ravel(), [1.0, -2.0, 0.5, 3.0], atol=5e-2
        )

    def test_resident_is_refreshed_from_the_master_copy(self):
        trainer, store = _build()
        store["w"][...] = np.float32(0.25)
        trainer.refresh_resident()
        np.testing.assert_array_equal(
            unpack_bf16(trainer._resident["w"]), np.full((4, 1), 0.25, np.float32)
        )

    def test_without_master_weights_the_parameters_stay_f32(self):
        trainer, store = _build(master=False)
        dtypes = {v.name: v.type.dtype for v in trainer.graph.inputs}
        self.assertEqual(dtypes["w"], f32)
        self.assertEqual(store["w"].dtype, np.float32)
        for batch in _batches(8):
            loss = trainer.step(batch)
        self.assertTrue(np.isfinite(loss))

    def test_snapshot_records_the_precision_and_stays_f32(self):
        trainer, store = _build()
        trainer.step(_batches(1)[0])
        snap = trainer.snapshot()
        self.assertEqual(snap.meta["mixed_precision"], "bf16")
        for name in store.names:
            self.assertEqual(snap.params[name].dtype, np.float32)
            np.testing.assert_array_equal(snap.params[name], store[name])

    def test_resuming_with_another_precision_is_rejected(self):
        trainer, _ = _build()
        trainer.step(_batches(1)[0])
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "bf16.npz")
            trainer.save_checkpoint(path)
            other, other_store = _build(mixed="f16", master=True)
            before = {name: other_store[name].copy() for name in other_store.names}
            with self.assertRaisesRegex(ValueError, "mixed_precision"):
                other.load_training_state(path)
            for name, value in before.items():
                np.testing.assert_array_equal(other_store[name], value)


class TestBf16Rejects(unittest.TestCase):
    def _graph_and_store(self):
        g = _lin_graph()
        return g, ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)

    def test_scaler_is_rejected_for_bf16(self):
        g, store = self._graph_and_store()
        with self.assertRaisesRegex(ValueError, "GradScaler"):
            Trainer(
                g, store, Adam(store.arrays(), lr=0.05),
                mixed_precision="bf16", scaler=GradScaler(),
            )

    def test_device_state_is_rejected_for_bf16(self):
        g, store = self._graph_and_store()
        with self.assertRaisesRegex(ValueError, "设备态"):
            Trainer(
                g, store, Adam(store.arrays(), lr=0.05),
                mixed_precision="bf16", device_state=True,
            )

    def test_master_weights_without_a_low_precision_target_is_rejected(self):
        g, store = self._graph_and_store()
        with self.assertRaisesRegex(ValueError, "master_weights"):
            Trainer(g, store, Adam(store.arrays(), lr=0.05), master_weights=True)


if __name__ == "__main__":
    unittest.main()
