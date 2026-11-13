"""训练辅助：学习率调度、梯度裁剪、梯度累积、EMA。

这一组测试固定的是几处"看起来能跑、其实悄悄错了"的地方：

- warmup 取 0 会让第一步空转，所以起点必须显式给；
- 梯度累积的最后一个窗口不足时，除以窗口大小会把梯度整体缩小；
- EMA 若零初始化，前一千步严重偏向 0，必须初始化即复制；
- 全局范数裁剪只能缩尺度，不能改方向。
"""

import unittest

import numpy as np

from aetherflow.train import (
    CosineDecay,
    EMA,
    ExponentialDecay,
    GradientAccumulator,
    SGD,
    StepDecay,
    Warmup,
    apply_lr,
    clip_global_norm,
    global_norm,
)


class TestSchedules(unittest.TestCase):
    def test_cosine_endpoints_and_monotonicity(self):
        s = CosineDecay(base_lr=0.1, total_steps=100, min_lr=0.001)
        self.assertAlmostEqual(s.lr_at(0), 0.1)
        self.assertAlmostEqual(s.lr_at(50), (0.1 + 0.001) / 2)
        self.assertAlmostEqual(s.lr_at(100), 0.001)
        # 超界停在下界，不报错：训练跑长了不该崩
        self.assertAlmostEqual(s.lr_at(1000), 0.001)
        values = [s.lr_at(i) for i in range(101)]
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(values, values[1:])))

    def test_cosine_rejects_bad_params(self):
        with self.assertRaises(ValueError):
            CosineDecay(base_lr=0.0, total_steps=10)
        with self.assertRaises(ValueError):
            CosineDecay(base_lr=0.1, total_steps=0)
        with self.assertRaises(ValueError):
            CosineDecay(base_lr=0.1, total_steps=10, min_lr=0.2)

    def test_step_decay(self):
        s = StepDecay(base_lr=1.0, step_size=2, gamma=0.5, min_lr=0.1)
        self.assertAlmostEqual(s.lr_at(0), 1.0)
        self.assertAlmostEqual(s.lr_at(1), 1.0)
        self.assertAlmostEqual(s.lr_at(2), 0.5)
        self.assertAlmostEqual(s.lr_at(4), 0.25)
        self.assertAlmostEqual(s.lr_at(100), 0.1)

    def test_exponential_decay(self):
        s = ExponentialDecay(base_lr=1.0, gamma=0.5, min_lr=0.05)
        self.assertAlmostEqual(s.lr_at(0), 1.0)
        self.assertAlmostEqual(s.lr_at(1), 0.5)
        self.assertAlmostEqual(s.lr_at(10), 0.05)

    def test_warmup_is_continuous_and_hands_over(self):
        inner = CosineDecay(base_lr=0.1, total_steps=100)
        s = Warmup(inner, steps=10, start_lr=0.001)
        self.assertAlmostEqual(s.lr_at(0), 0.001)
        self.assertAlmostEqual(s.lr_at(5), (0.001 + inner.lr_at(10)) / 2)
        # 交接点与内层严格对齐，之后直接委托（按绝对步数）
        self.assertAlmostEqual(s.lr_at(10), inner.lr_at(10))
        self.assertAlmostEqual(s.lr_at(50), inner.lr_at(50))

    def test_warmup_rejects_descending(self):
        with self.assertRaises(ValueError):
            Warmup(CosineDecay(base_lr=0.1, total_steps=10), steps=5, start_lr=0.5)

    def test_apply_lr_uses_optimizer_step_count(self):
        opt = SGD({"w": np.zeros(2, dtype=np.float32)}, lr=0.1)
        schedule = StepDecay(base_lr=0.1, step_size=50, gamma=0.1)
        self.assertAlmostEqual(apply_lr(opt, schedule), 0.1)
        opt.step_count = 50
        self.assertAlmostEqual(apply_lr(opt, schedule), 0.01)
        self.assertAlmostEqual(opt.lr, 0.01)


class TestClipping(unittest.TestCase):
    def test_global_norm(self):
        grads = {"a": np.array([3.0, 0.0], np.float32), "b": np.array([4.0], np.float32)}
        self.assertAlmostEqual(global_norm(grads), 5.0, places=6)

    def test_no_clip_when_below_threshold(self):
        grads = {"w": np.array([0.3, 0.4], np.float32)}
        clipped, total = clip_global_norm(grads, 1.0)
        np.testing.assert_allclose(clipped["w"], grads["w"])
        self.assertAlmostEqual(total, 0.5, places=6)

    def test_clip_preserves_direction_and_hits_norm(self):
        grads = {"w": np.array([3.0, 4.0], np.float32)}
        clipped, total = clip_global_norm(grads, 1.0)
        self.assertAlmostEqual(total, 5.0, places=6)
        self.assertAlmostEqual(global_norm(clipped), 1.0, places=5)
        np.testing.assert_allclose(
            clipped["w"] / np.linalg.norm(clipped["w"]),
            grads["w"] / np.linalg.norm(grads["w"]),
            atol=1e-6,
        )

    def test_returns_copies(self):
        grads = {"w": np.array([3.0, 4.0], np.float32)}
        clipped, _ = clip_global_norm(grads, 1.0)
        self.assertIsNot(clipped["w"], grads["w"])
        np.testing.assert_allclose(grads["w"], [3.0, 4.0])

    def test_non_finite_rejected(self):
        grads = {"w": np.array([np.inf, 0.0], np.float32)}
        with self.assertRaises(ValueError):
            clip_global_norm(grads, 1.0)

    def test_bad_max_norm(self):
        with self.assertRaises(ValueError):
            clip_global_norm({"w": np.zeros(1, np.float32)}, 0.0)


class TestGradientAccumulator(unittest.TestCase):
    def test_average_of_micro_batches(self):
        acc = GradientAccumulator(2)
        self.assertFalse(acc.add({"w": np.ones(3, np.float32)}))
        self.assertTrue(acc.add({"w": np.full(3, 3.0, np.float32)}))
        out = acc.flush()
        np.testing.assert_allclose(out["w"], np.full(3, 2.0))
        self.assertEqual(acc.pending, 0)

    def test_partial_window_divides_by_actual_count(self):
        """最后一个窗口不足时除以实际累积数，否则梯度会被整体缩小。"""
        acc = GradientAccumulator(4)
        acc.add({"w": np.full(2, 2.0, np.float32)})
        out = acc.flush()
        np.testing.assert_allclose(out["w"], np.full(2, 2.0))

    def test_add_after_full_rejected(self):
        acc = GradientAccumulator(1)
        self.assertTrue(acc.add({"w": np.zeros(2, np.float32)}))
        with self.assertRaises(RuntimeError):
            acc.add({"w": np.zeros(2, np.float32)})

    def test_empty_flush(self):
        self.assertIsNone(GradientAccumulator(2).flush())

    def test_shape_mismatch_rejected(self):
        acc = GradientAccumulator(2)
        acc.add({"w": np.zeros(2, np.float32)})
        with self.assertRaises(ValueError):
            acc.add({"w": np.zeros(3, np.float32)})

    def test_name_set_mismatch_rejected(self):
        acc = GradientAccumulator(2)
        acc.add({"w": np.zeros(2, np.float32)})
        with self.assertRaises(KeyError):
            acc.add({"b": np.zeros(2, np.float32)})

    def test_f16_inputs_accumulate_in_f32(self):
        acc = GradientAccumulator(2)
        acc.add({"w": np.ones(2, np.float16)})
        acc.add({"w": np.ones(2, np.float16)})
        out = acc.flush()
        self.assertEqual(out["w"].dtype, np.float32)
        np.testing.assert_allclose(out["w"], np.ones(2))


class TestEMA(unittest.TestCase):
    def test_shadow_starts_as_copy(self):
        params = {"w": np.array([1.0, 2.0], np.float32)}
        ema = EMA(params, decay=0.5)
        np.testing.assert_allclose(ema.shadow()["w"], [1.0, 2.0])
        np.testing.assert_allclose(params["w"], [1.0, 2.0])

    def test_update_formula(self):
        params = {"w": np.array([1.0, 2.0], np.float32)}
        ema = EMA(params, decay=0.5)
        params["w"][...] = 3.0
        ema.update()
        np.testing.assert_allclose(ema.shadow()["w"], [2.0, 2.5])
        self.assertEqual(ema.num_updates, 1)

    def test_copy_to_writes_in_place(self):
        params = {"w": np.array([1.0, 2.0], np.float32)}
        ema = EMA(params, decay=0.5)
        params["w"][...] = 3.0
        ema.update()
        original = params["w"]
        ema.copy_to()
        self.assertIs(params["w"], original)  # 原地写，不替换数组
        np.testing.assert_allclose(params["w"], [2.0, 2.5])

    def test_state_roundtrip(self):
        params = {"w": np.array([1.0, 2.0], np.float32)}
        ema = EMA(params, decay=0.9)
        params["w"][...] = 5.0
        ema.update()
        state = ema.state_dict()

        other = EMA({"w": np.zeros(2, np.float32)}, decay=0.9)
        other.load_state_dict(state)
        np.testing.assert_allclose(other.shadow()["w"], ema.shadow()["w"])
        self.assertEqual(other.num_updates, ema.num_updates)

    def test_state_shape_mismatch_rejected(self):
        ema = EMA({"w": np.zeros(2, np.float32)}, decay=0.5)
        with self.assertRaises(ValueError):
            ema.load_state_dict({"shadow": {"w": np.zeros(3, np.float32)}})

    def test_bad_decay(self):
        with self.assertRaises(ValueError):
            EMA({"w": np.zeros(1, np.float32)}, decay=1.0)


if __name__ == "__main__":
    unittest.main()
