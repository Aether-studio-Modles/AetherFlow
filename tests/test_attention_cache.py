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
    PagedDecodeScheduler,
    TransformerSpec,
    build_transformer,
    cached_forward,
    causal_mask_bias,
)
from aetherflow.runtime import PagedCacheFullError

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


class TestPagedDecodeScheduler(unittest.TestCase):
    """多序列交错解码：池容量与换出次数都不该改数值。

    判据：

    1. 两条序列逐步交错，每步隐状态与各自单独跑稠密图逐位置一致；
    2. 池只装得下一部分时，挂起 / 恢复会自动换出 / 换入，恢复后数据连续；
    3. 正在解码（active）的序列被 pin 住，池满时宁可报错也不换出它；
    4. 失败的那一步不写回缓存——长度不变，重试不会重复计数。
    """

    def _scheduler(self, spec, params, **overrides):
        options = dict(num_blocks=8, block_size=2, max_steps=5)
        options.update(overrides)
        return PagedDecodeScheduler(BACKEND, params, spec, **options)

    def _stream(self, scheduler, rid, xs):
        return np.stack([scheduler.step(rid, x) for x in xs])

    def test_interleaved_matches_dense(self):
        spec = _spec()
        params = _params(spec, seed=3)
        steps = 5
        rng = np.random.default_rng(7)
        xs_a = rng.standard_normal((steps, spec.d_model)).astype(np.float32) * 0.5
        xs_b = rng.standard_normal((steps, spec.d_model)).astype(np.float32) * 0.5
        sched = self._scheduler(spec, params, max_steps=steps)
        sched.submit(1)
        sched.submit(2)
        got_a, got_b = [], []
        for t in range(steps):
            got_a.append(sched.step(1, xs_a[t]))
            got_b.append(sched.step(2, xs_b[t]))
        np.testing.assert_allclose(
            np.stack(got_a), _dense(xs_a, params, spec)[0], rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            np.stack(got_b), _dense(xs_b, params, spec)[0], rtol=1e-4, atol=1e-5
        )
        self.assertEqual(sched.length(1), steps)
        self.assertEqual(sched.length(2), steps)
        self.assertEqual(sched.stats()["swaps_out"], 0)
        self.assertEqual(sched.stats()["active_sequences"], 2)

    def test_suspend_evicts_and_resume_continues(self):
        spec = _spec()
        params = _params(spec, seed=5)
        steps = 4
        rng = np.random.default_rng(11)
        xs_a = rng.standard_normal((steps, spec.d_model)).astype(np.float32) * 0.5
        xs_b = rng.standard_normal((3, spec.d_model)).astype(np.float32) * 0.5
        # 池容量 = 2 页 × 2 token = 4 个 token，装不下两条序列的 7 个 token
        sched = self._scheduler(spec, params, num_blocks=2, block_size=2, max_steps=steps)
        sched.submit(1)
        got_a = list(self._stream(sched, 1, xs_a[:3]))
        sched.suspend(1)
        self.assertEqual(sched.stats()["suspended_sequences"], 1)

        sched.submit(2)
        got_b = list(self._stream(sched, 2, xs_b))
        self.assertTrue(sched.is_swapped(1))  # 页不够，挂起的 1 被换出
        self.assertFalse(sched.is_swapped(2))
        sched.suspend(2)

        self.assertEqual(sched.resume(1), 2)
        self.assertTrue(sched.is_swapped(2))
        got_a.append(sched.step(1, xs_a[3]))
        np.testing.assert_allclose(
            np.stack(got_a), _dense(xs_a, params, spec)[0], rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            np.stack(got_b), _dense(xs_b, params, spec)[0][:3], rtol=1e-4, atol=1e-5
        )
        info = sched.stats()
        self.assertEqual(info["swaps_out"], 2)
        self.assertEqual(info["swaps_in"], 1)
        self.assertEqual(info["used_blocks"], 2)

    def test_active_sequences_are_pinned(self):
        spec = _spec()
        params = _params(spec, seed=9)
        rng = np.random.default_rng(21)
        xs = rng.standard_normal((4, spec.d_model)).astype(np.float32) * 0.5
        sched = self._scheduler(spec, params, num_blocks=2, block_size=2, max_steps=4)
        sched.submit(1)
        self._stream(sched, 1, xs[:3])  # 2 页占满，仍然 active
        sched.submit(2)
        with self.assertRaises(PagedCacheFullError):
            sched.step(2, xs[0])
        self.assertEqual(sched.length(2), 0)
        self.assertEqual(sched.length(1), 3)
        self.assertFalse(sched.is_swapped(1))
        got = sched.step(1, xs[3])
        np.testing.assert_allclose(
            got, _dense(xs, params, spec)[0][3], rtol=1e-4, atol=1e-5
        )

    def test_suspend_with_evict_releases_pages_immediately(self):
        spec = _spec()
        params = _params(spec, seed=13)
        rng = np.random.default_rng(31)
        xs = rng.standard_normal((4, spec.d_model)).astype(np.float32) * 0.5
        sched = self._scheduler(spec, params, num_blocks=4, block_size=2, max_steps=4)
        sched.submit(1)
        got = list(self._stream(sched, 1, xs[:3]))
        self.assertTrue(sched.suspend(1, evict=True))
        self.assertTrue(sched.is_swapped(1))
        self.assertEqual(sched.stats()["used_blocks"], 0)
        self.assertFalse(sched.suspend(1))

        self.assertEqual(sched.resume(1), 2)
        self.assertFalse(sched.is_swapped(1))
        got.append(sched.step(1, xs[3]))
        np.testing.assert_allclose(
            np.stack(got), _dense(xs, params, spec)[0], rtol=1e-4, atol=1e-5
        )

    def test_fork_shares_prefix_then_diverges(self):
        spec = _spec()
        params = _params(spec, seed=17)
        rng = np.random.default_rng(41)
        prefix = rng.standard_normal((2, spec.d_model)).astype(np.float32) * 0.5
        tail_a = rng.standard_normal((3, spec.d_model)).astype(np.float32) * 0.5
        tail_b = rng.standard_normal((3, spec.d_model)).astype(np.float32) * 0.5
        xs_a = np.concatenate([prefix, tail_a])
        xs_b = np.concatenate([prefix, tail_b])
        sched = self._scheduler(spec, params, num_blocks=8, block_size=2, max_steps=5)
        sched.submit(1)
        got_prefix = self._stream(sched, 1, prefix)
        used = sched.stats()["used_blocks"]
        sched.fork(1, 2)
        info = sched.stats()
        self.assertEqual(info["used_blocks"], used)  # 分叉不额外占页
        self.assertEqual(info["shared_blocks"], used)
        self.assertEqual(info["forks"], 1)
        self.assertEqual(info["active_sequences"], 2)
        self.assertEqual(sched.length(2), 2)
        got_a, got_b = [], []
        for t in range(3):
            got_a.append(sched.step(1, tail_a[t]))
            got_b.append(sched.step(2, tail_b[t]))
        np.testing.assert_allclose(
            np.concatenate([got_prefix, np.stack(got_a)]),
            _dense(xs_a, params, spec)[0],
            rtol=1e-4,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.concatenate([got_prefix, np.stack(got_b)]),
            _dense(xs_b, params, spec)[0],
            rtol=1e-4,
            atol=1e-5,
        )
        self.assertEqual(sched.length(1), 5)
        self.assertEqual(sched.length(2), 5)

    def test_fork_rejections(self):
        spec = _spec()
        params = _params(spec)
        sched = self._scheduler(spec, params, num_blocks=4, block_size=2, max_steps=4)
        with self.assertRaises(KeyError):
            sched.fork(9, 1)
        sched.submit(1)
        sched.submit(3)
        with self.assertRaisesRegex(ValueError, "已经提交"):
            sched.fork(1, 3)
        rng = np.random.default_rng(3)
        xs = rng.standard_normal((2, spec.d_model)).astype(np.float32) * 0.5
        self._stream(sched, 1, xs)
        sched.suspend(1, evict=True)
        with self.assertRaisesRegex(ValueError, "resume"):
            sched.fork(1, 2)

    def test_rejections_and_id_reuse(self):
        spec = _spec()
        params = _params(spec)
        with self.assertRaisesRegex(ValueError, "vocab=0"):
            PagedDecodeScheduler(
                BACKEND, params, _spec(vocab=8), num_blocks=4, max_steps=4
            )
        with self.assertRaisesRegex(ValueError, "batch=1"):
            PagedDecodeScheduler(
                BACKEND, params, _spec(batch=2), num_blocks=4, max_steps=4
            )
        with self.assertRaisesRegex(TypeError, "num_blocks"):
            PagedDecodeScheduler(
                BACKEND, params, spec, num_blocks=True, max_steps=4
            )
        with self.assertRaisesRegex(ValueError, "block_size"):
            PagedDecodeScheduler(
                BACKEND, params, spec, num_blocks=4, block_size=0, max_steps=4
            )
        with self.assertRaisesRegex(ValueError, "max_steps"):
            PagedDecodeScheduler(
                BACKEND, params, spec, num_blocks=4, max_steps=0
            )
        with self.assertRaisesRegex(KeyError, "缺少权重"):
            PagedDecodeScheduler(BACKEND, {}, spec, num_blocks=4, max_steps=4)

        sched = self._scheduler(spec, params, num_blocks=2, max_steps=2)
        with self.assertRaisesRegex(KeyError, "没有提交"):
            sched.step(1, np.zeros(spec.d_model, np.float32))
        sched.submit(1)
        with self.assertRaisesRegex(ValueError, "已经提交"):
            sched.submit(1)
        xs = np.zeros((2, spec.d_model), np.float32)
        sched.step(1, xs[0])
        with self.assertRaisesRegex(ValueError, "x 形状"):
            sched.step(1, np.zeros((3,), np.float32))
        sched.step(1, xs[1])
        with self.assertRaisesRegex(ValueError, "max_steps"):
            sched.step(1, xs[0])
        sched.suspend(1)
        with self.assertRaisesRegex(ValueError, "挂起状态"):
            sched.step(1, xs[0])
        sched.resume(1)
        with self.assertRaisesRegex(KeyError, "没有提交"):
            sched.finish(9)
        sched.finish(1)
        self.assertNotIn(1, sched)
        sched.submit(1)  # 同一个 id 可以复用
        self.assertEqual(sched.length(1), 0)


if __name__ == "__main__":
    unittest.main()
