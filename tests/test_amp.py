"""f16 与梯度缩放的验收测试。

三条判据：

1. **f16 图在语义上就是 f32 图的低精度版本**：同一份数据下前向与反向
   都只差在量化误差里，没有额外的行为差异。
2. **缩放确实把下溢的梯度救回来**：小于 f16 次正规下限的值，缩放后可表示，
   反缩放后还原；反之在 f16 上直接相除会再次变成 0。
3. **溢出必须跳过**：inf/nan 梯度不能交给优化器；跳过的步数也不计入
   ``step_count``，否则 Adam 的偏差修正会算错。
"""

import unittest

import numpy as np

from aetherflow import Graph, ReferenceBackend, backward, f16, f32, f64
from aetherflow.models.lm import LMSpec, build_language_model, masked_lm_loss
from aetherflow.passes.autocast import Autocast
from aetherflow.train import (
    SGD,
    Adam,
    GradScaler,
    ParameterStore,
    Trainer,
    init_array,
    lm_batch,
    mse,
)


BACKEND = ReferenceBackend()


class TestGradScaler(unittest.TestCase):
    def test_backward_seed_is_scalar_of_requested_dtype(self):
        scaler = GradScaler(init_scale=1024.0)
        seed = scaler.backward_seed(np.float16)
        self.assertEqual(seed.shape, ())
        self.assertEqual(seed.dtype, np.float16)
        self.assertEqual(float(seed), 1024.0)

    def test_backward_seed_rejects_scale_beyond_dtype(self):
        # f16 上限 65504，2^16 = 65536 装不下
        scaler = GradScaler(init_scale=2.0 ** 16, max_scale=2.0 ** 16)
        with self.assertRaises(ValueError):
            scaler.backward_seed(np.float16)
        self.assertEqual(float(scaler.backward_seed(np.float32)), 65536.0)

    def test_unscale_promotes_to_f32_before_dividing(self):
        scaler = GradScaler(init_scale=2.0 ** 14)
        true_grad = 1e-8                      # 低于 f16 次正规下限，直接表示会是 0
        scaled = np.float16(true_grad * scaler.scale)
        self.assertNotEqual(float(scaled), 0.0)

        out = scaler.unscale({"w": scaled})
        self.assertEqual(out["w"].dtype, np.float32)
        self.assertAlmostEqual(float(out["w"]), true_grad, delta=true_grad * 0.02)
        self.assertFalse(scaler.found_nonfinite)

        # 在 f16 上直接相除会二次下溢——这正是"先升精度再除"的理由
        self.assertEqual(float(np.float16(scaled / np.float16(scaler.scale))), 0.0)

    def test_step_applies_unscaled_gradient(self):
        params = {"w": np.array([1.0], dtype=np.float32)}
        opt = SGD(params, lr=0.1)
        scaler = GradScaler(init_scale=256.0)
        # 缩放后的梯度 512 对应真实梯度 2
        self.assertTrue(scaler.step(opt, {"w": np.array([512.0], dtype=np.float16)}))
        np.testing.assert_allclose(params["w"], [0.8], rtol=1e-6)
        self.assertEqual(opt.step_count, 1)

    def test_step_skips_nonfinite_gradient(self):
        for bad in (np.inf, -np.inf, np.nan):
            params = {"w": np.array([1.0], dtype=np.float32)}
            opt = SGD(params, lr=0.1)
            scaler = GradScaler(init_scale=1024.0)
            ok = scaler.step(opt, {"w": np.array([bad], dtype=np.float16)})
            self.assertFalse(ok)
            self.assertTrue(scaler.found_nonfinite)
            np.testing.assert_array_equal(params["w"], [1.0])
            self.assertEqual(opt.step_count, 0)   # 跳过的步不计入，偏差修正才不会错
            self.assertEqual(scaler.scale, 512.0)
            self.assertEqual(scaler.skipped, 1)

    def test_scale_grows_after_interval_and_is_capped(self):
        params = {"w": np.zeros(1, dtype=np.float32)}
        opt = SGD(params, lr=0.1)
        scaler = GradScaler(init_scale=8.0, growth_interval=2, max_scale=16.0)
        zero = np.zeros(1, dtype=np.float16)
        for _ in range(6):
            self.assertTrue(scaler.step(opt, {"w": zero}))
        # 8 -> 16 -> 16 -> 16：封顶后不会再涨，否则种子会超出 f16 范围
        self.assertEqual(scaler.scale, 16.0)

    def test_invalid_hyperparameters(self):
        for kwargs in (
            {"init_scale": 0.0},
            {"growth_factor": 1.0},
            {"backoff_factor": 1.0},
            {"growth_interval": 0},
            {"max_scale": 1.0},
        ):
            with self.assertRaises(ValueError):
                GradScaler(**kwargs)

    def test_state_dict_roundtrip(self):
        scaler = GradScaler(init_scale=8.0, growth_interval=2)
        opt = SGD({"w": np.zeros(1, dtype=np.float32)}, lr=0.1)
        scaler.step(opt, {"w": np.zeros(1, dtype=np.float16)})

        state = scaler.state_dict()
        restored = GradScaler(init_scale=1.0)
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)

    def test_load_state_dict_rejects_bad_values(self):
        scaler = GradScaler()
        for state in (
            {"scale": 0.0, "growth": 0, "skipped": 0},
            {"scale": float("inf"), "growth": 0, "skipped": 0},
            {"scale": 1.0, "growth": -1, "skipped": 0},
            {"scale": 1.0, "growth": 0, "skipped": -1},
            {"scale": 1.0, "growth": 0},
        ):
            with self.assertRaises((ValueError, KeyError)):
                scaler.load_state_dict(state)
        # 校验失败不能留下半更新：一个坏值都不该改到当前状态
        self.assertEqual(scaler.state_dict(), {"scale": 2.0 ** 14, "growth": 0, "skipped": 0})


class TestF16Graph(unittest.TestCase):
    def test_f16_forward_and_backward_match_f32(self):
        def build(dtype):
            g = Graph("m")
            x = g.input("x", dtype[4, 3])
            w = g.input("w", dtype[3, 2])
            b = g.input("b", dtype[2])
            g.output(g.tanh(g.add(g.matmul(x, w), b)), "y")
            return g

        rng = np.random.default_rng(0)
        base = {
            "x": rng.standard_normal((4, 3)).astype(np.float16),
            "w": rng.standard_normal((3, 2)).astype(np.float16),
            "b": rng.standard_normal(2).astype(np.float16),
        }
        base32 = {k: v.astype(np.float32) for k, v in base.items()}

        g16, g32 = build(f16), build(f32)
        y16 = BACKEND.execute(g16, base)["y"]
        y32 = BACKEND.execute(g32, base32)["y"]
        self.assertEqual(y16.dtype, np.float16)
        np.testing.assert_allclose(y16.astype(np.float32), y32, rtol=1e-2, atol=1e-3)

        bwd16 = backward(g16, wrt=[g16.find("x"), g16.find("w")])
        bwd32 = backward(g32, wrt=[g32.find("x"), g32.find("w")])
        d16 = BACKEND.execute(bwd16, {**base, "dy": np.ones((4, 2), np.float16)})
        d32 = BACKEND.execute(bwd32, {**base32, "dy": np.ones((4, 2), np.float32)})
        for name in ("dx", "dw"):
            self.assertEqual(d16[name].dtype, np.float16)
            np.testing.assert_allclose(
                d16[name].astype(np.float32), d32[name], rtol=5e-2, atol=1e-3
            )

    def test_f16_training_converges_with_scaler(self):
        """f16 图 + f32 master 参数 + 缩放：真实训练循环能收敛。"""
        g = Graph("amp")
        x = g.input("x", f16[8, 3])
        w = g.input("w", f16[3, 1])
        b = g.input("b", f16[1])
        target = g.input("target", f16[8, 1])
        g.output(mse(g, g.add(g.matmul(x, w), b), target), "loss")

        rng = np.random.default_rng(1)
        xs = rng.standard_normal((8, 3)).astype(np.float16)
        true_w = np.array([[1.0], [-2.0], [0.5]], dtype=np.float16)
        ys = (xs @ true_w + np.float16(0.3)).astype(np.float16)

        # 参数留在 f32：f16 的更新量会被量化吞掉，master 权重必须高精度
        store = ParameterStore({
            "w": init_array("w", (3, 1), seed=0),
            "b": init_array("b", (1,), seed=0),
        })
        opt = Adam(store.arrays(), lr=0.05)
        scaler = GradScaler(init_scale=2.0 ** 10, growth_interval=50)
        bwd = backward(g, wrt=[g.find(n) for n in store.names])

        def feed():
            data = {n: store[n].astype(np.float16) for n in store.names}
            data.update({"x": xs, "target": ys})
            return data

        losses = []
        for _ in range(300):
            raw = BACKEND.execute(
                bwd, {**feed(), "dloss": scaler.backward_seed(np.float16)}
            )
            # 反向图按 d<输入名> 命名；优化器要的是参数名
            grads = {n: raw[f"d{n}"] for n in store.names}
            scaler.step(opt, grads)
            losses.append(float(BACKEND.execute(g, feed())["loss"]))

        self.assertEqual(scaler.skipped, 0, f"这个规模不该溢出: {scaler}")
        self.assertLess(losses[-1], losses[0] * 0.05, f"f16 训练没收敛: {losses[0]} -> {losses[-1]}")


class TestF16BackwardQuantization(unittest.TestCase):
    """f16 反向的量化：真实 LM 图（embedding + Mamba + masked loss）。

    5.121 只验证过 ``cast`` 的 VJP 能透传，没有量化半精度梯度在乘法链上的
    累积误差。这里用同一份参数与数据对比 f32 反向与 f16（Autocast + 缩放）
    反向：前向损失只差量化误差，每个参数梯度的相对误差都在 1e-2 内、方向
    余弦大于 0.999，且没有非有限值。

    缩放必须开：f16 种子乘 2^14 再除回来，比较的才是同一个物理量。
    """

    def test_long_chain_matches_f32_within_quantization_error(self):
        spec = LMSpec(
            vocab_size=258,
            d_model=16,
            num_layers=1,
            d_state=8,
            d_conv=4,
            steps=8,
            batch=2,
        )
        g = Graph("lm")
        _inputs, logits = build_language_model(g, spec)
        targets = g.input("targets", f32[(spec.batch, spec.steps, spec.vocab_size)])
        mask = g.input("mask", f32[(spec.batch, spec.steps)])
        g.output(masked_lm_loss(g, logits, targets, mask), "loss")

        params = {
            name: np.asarray(value, np.float32)
            for name, value in spec.init_params(seed=0).items()
        }
        rng = np.random.default_rng(1)
        ids = rng.integers(
            2, spec.vocab_size, size=(spec.batch, spec.steps), dtype=np.int32
        )
        data = lm_batch([row.tolist() for row in ids], spec.vocab_size, steps=spec.steps)
        feed = {**params, **data}

        loss32 = float(BACKEND.execute(g, feed)["loss"])
        bwd32 = backward(g, wrt=[g.find(name) for name in params], active=["loss"])
        grads32 = {
            name: np.asarray(
                BACKEND.execute(bwd32, {**feed, "dloss": np.array(1.0, np.float32)})[
                    "d" + name
                ]
            )
            for name in params
        }

        g16 = Autocast(dtype="f16").run(g)
        loss16 = float(BACKEND.execute(g16, {**params, **data})["loss"])
        bwd16 = backward(g16, wrt=[g16.find(name) for name in params], active=["loss"])
        scale = float(2 ** 14)
        grads16 = {
            name: np.asarray(
                BACKEND.execute(
                    bwd16, {**params, **data, "dloss": np.array(scale, np.float16)}
                )["d" + name],
                dtype=np.float64,
            )
            / scale
            for name in params
        }

        self.assertLess(abs(loss16 - loss32) / abs(loss32), 1e-3)
        for name, ref in grads32.items():
            got = grads16[name]
            self.assertTrue(np.isfinite(got).all(), f"{name} 的 f16 梯度含非有限值")
            ref = ref.astype(np.float64)
            ref_norm = np.linalg.norm(ref)
            got_norm = np.linalg.norm(got)
            rel = np.linalg.norm(ref - got) / (ref_norm + 1e-30)
            cos = float((ref * got).sum() / (ref_norm * got_norm + 1e-30))
            self.assertLess(rel, 1e-2, f"{name} 的相对误差 {rel:.3e}")
            self.assertGreater(cos, 0.999, f"{name} 的方向余弦 {cos:.6f}")


class TestMasterWeights(unittest.TestCase):
    """f32 主副本 + f16 驻留：参数只在更新时量化一次，而不是每步 cast。

    与 ``mixed_precision="f16"``（参数仍 f32、图里每步 cast）的差别只有一处：
    图输入本身声明成 f16。判据因此是三条——驻留 dtype、matmul 直接读参数
    输入（没有 cast）、训练轨迹与全 f32 在量化误差内一致。
    """

    def _build(self, mixed="f16", master=True, scaler=True):
        g = Graph("lin")
        x = g.input("x", f32[8, 4])
        t = g.input("t", f32[8, 1])
        w = g.input("w", f32[4, 1])
        b = g.input("b", f32[1])
        g.output(mse(g, g.add(g.matmul(x, w), b), t), "loss")
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        trainer = Trainer(
            g,
            store,
            Adam(store.arrays(), lr=0.05),
            mixed_precision=mixed,
            master_weights=master,
            scaler=(
                GradScaler(init_scale=2.0 ** 8, growth_interval=1000)
                if scaler
                else None
            ),
        )
        return trainer, store

    @staticmethod
    def _batches(n=240):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(32, 4)).astype(np.float32)
        true_w = np.array([[1.0], [-2.0], [0.5], [3.0]], np.float32)
        t = (x @ true_w + 0.7).astype(np.float32)
        out = []
        for i in range(n):
            j = (i * 8) % 32
            out.append({"x": x[j:j + 8], "t": t[j:j + 8]})
        return out

    def test_parameters_are_f16_resident_with_f32_master(self):
        trainer, store = self._build()
        dtypes = {v.name: v.type.dtype for v in trainer.graph.inputs}
        self.assertEqual(dtypes["w"], f16)
        self.assertEqual(dtypes["b"], f16)
        self.assertEqual(dtypes["x"], f32)   # 数据输入不参与降精度
        for name in store.names:
            self.assertEqual(trainer._resident[name].dtype, np.float16)
            self.assertEqual(store[name].dtype, np.float32)
            self.assertEqual(trainer.optimizer.params[name].dtype, np.float32)

    def test_parameter_feeds_the_matmul_without_a_cast(self):
        trainer, _ = self._build()
        (mm,) = [n for n in trainer.graph.nodes if n.op.name == "matmul"]
        self.assertTrue(any(a.producer is None and a.name == "w" for a in mm.inputs))

        plain, _ = self._build(master=False)
        (mm2,) = [n for n in plain.graph.nodes if n.op.name == "matmul"]
        self.assertTrue(
            any(
                a.producer is not None
                and a.producer.op.name == "cast"
                and a.producer.inputs[0].name == "w"
                for a in mm2.inputs
            )
        )

    def test_training_converges_and_matches_f32_within_quantization(self):
        mixed, store = self._build()
        ref, ref_store = self._build(mixed=None, master=False, scaler=False)
        batches = self._batches()
        for batch in batches:
            mixed.step(batch)
            ref.step(batch)

        for name in store.names:
            scale = max(float(np.abs(ref_store[name]).max()), 1e-12)
            rel = float(np.abs(store[name] - ref_store[name]).max() / scale)
            self.assertLess(rel, 1e-3, f"{name} 的权重相对误差 {rel:.3e}")
        np.testing.assert_allclose(store["w"].ravel(), [1.0, -2.0, 0.5, 3.0], atol=5e-2)

    def test_refresh_resident_picks_up_an_external_store_change(self):
        trainer, store = self._build()
        store["w"][...] = np.float32(0.25)
        trainer.refresh_resident()
        np.testing.assert_array_equal(
            trainer._resident["w"], np.full((4, 1), np.float16(0.25))
        )

        plain, _ = self._build(master=False)
        with self.assertRaises(ValueError):
            plain.refresh_resident()

    def test_snapshot_stores_the_f32_master(self):
        trainer, store = self._build()
        for batch in self._batches(4):
            trainer.step(batch)
        snapshot = trainer.snapshot()
        for name in store.names:
            self.assertEqual(snapshot.params[name].dtype, np.float32)
            np.testing.assert_array_equal(snapshot.params[name], store[name])

    def test_master_weights_requires_f16_and_f32_params(self):
        g = Graph("lin")
        x = g.input("x", f32[2, 2])
        w = g.input("w", f32[2, 1])
        g.output(mse(g, g.matmul(x, w), g.input("t", f32[2, 1])), "loss")
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        with self.assertRaises(ValueError):
            Trainer(g, store, SGD(store.arrays(), lr=0.1), master_weights=True)
        with self.assertRaises(ValueError):
            Trainer(
                g, store, SGD(store.arrays(), lr=0.1),
                mixed_precision="bf16", master_weights=True,
            )

        g64 = Graph("lin64")
        x64 = g64.input("x", f64[2, 2])
        w64 = g64.input("w", f64[2, 1])
        g64.output(mse(g64, g64.matmul(x64, w64), g64.input("t", f64[2, 1])), "loss")
        store64 = ParameterStore.from_graph(g64, exclude=("x", "t"), seed=0)
        with self.assertRaises(TypeError):
            Trainer(
                g64, store64, SGD(store64.arrays(), lr=0.1),
                mixed_precision="f16", master_weights=True,
            )


if __name__ == "__main__":
    unittest.main()
