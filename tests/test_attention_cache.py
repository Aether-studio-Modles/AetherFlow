"""增量注意力解码 + 分页 KV cache 的数值验收。

判据只有一条，但足够硬：**逐位置等于稠密图**。

同一个 ``TransformerSpec``（``vocab=0``）与同一份随机权重下，
``cached_forward``（逐 token、K/V 走分页缓存）的最后隐状态必须与
``build_transformer`` 整段前向的对应位置一致。页宽只改分配粒度，不该改数值。
"""

import math
import unittest
from dataclasses import replace

import numpy as np

from aetherflow import Graph, ReferenceBackend
from aetherflow.models import (
    PagedDecodeScheduler,
    PagedGenerator,
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


def _vocab_spec(**overrides):
    base = dict(
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        seq=12,
        vocab=16,
        batch=1,
        dropout=0.0,
    )
    base.update(overrides)
    return TransformerSpec(**base)


def _vocab_params(spec, seed=0):
    """在 ``_params`` 之上把位置编码改成非零：零位置表测不出位置真的用上了。"""
    params = _params(spec, seed)
    rng = np.random.default_rng(seed + 1000)
    params["pos"] = (
        rng.standard_normal(spec.param_shapes["pos"]) * 0.1
    ).astype(np.float32)
    return params


def _dense_next(params, spec, tokens):
    """稠密图参照：整段前向一遍，取最后位置的 argmax。"""
    seq = len(tokens)
    dense_spec = replace(spec, seq=seq)
    g = Graph("dense_next")
    inputs, logits = build_transformer(g, dense_spec)
    g.output(logits, "logits")
    feed = {
        inputs["ids"].name: np.array([tokens], dtype=np.int32),
        inputs["mask"].name: causal_mask_bias(seq),
    }
    for name in dense_spec.param_shapes:
        feed[inputs[name].name] = (
            params["pos"][:, :seq] if name == "pos" else params[name]
        )
    out = BACKEND.execute(g, feed)["logits"]
    return int(np.argmax(out[0, -1]))


def _dense_greedy(params, spec, prompt, max_new_tokens, eos_id=None):
    tokens = list(prompt)
    out = []
    for _ in range(max_new_tokens):
        nxt = _dense_next(params, spec, tokens)
        out.append(nxt)
        if eos_id is not None and nxt == eos_id:
            break
        tokens.append(nxt)
    return out


class TestPagedGenerator(unittest.TestCase):
    """分页 KV 上的生成闭环：判据仍是"逐 token 等于稠密图"。

    生成层只是补上 token → 嵌入 / 位置编码、隐状态 → 词表头、以及停止条件；
    分页、换出换入与共享前缀都来自 :class:`PagedDecodeScheduler`，所以这里
    重点验三件事：贪心 token 与稠密图一致、``fork`` 不额外占页且数值不变、
    挂起换出之后续写的 token 不变。
    """

    def _generator(self, spec, params, **overrides):
        # pad_to 是 max_steps，所以池容量（blocks × block_size）必须装得下它
        options = dict(
            num_blocks=max(8, math.ceil(spec.seq / 2)), block_size=2, max_steps=spec.seq
        )
        options.update(overrides)
        return PagedGenerator(BACKEND, params, spec, **options)

    def test_generate_matches_dense_greedy(self):
        spec = _vocab_spec()
        params = _vocab_params(spec, seed=2)
        prompt = [1, 4, 7]
        for block_size in (1, 2, 3):
            with self.subTest(block_size=block_size):
                gen = self._generator(
                    spec,
                    params,
                    block_size=block_size,
                    num_blocks=math.ceil(spec.seq / block_size),
                )
                got = gen.generate(prompt, max_new_tokens=5, rid=1)
                self.assertEqual(
                    got, _dense_greedy(params, spec, prompt, 5)
                )
                # 最后一个生成的 token 不必再喂回缓存：长度 = prompt + 4
                self.assertEqual(gen.length(1), len(prompt) + 4)
                self.assertEqual(gen.tokens(1), prompt + got[:-1])

    def test_generate_stops_at_eos(self):
        spec = _vocab_spec()
        params = _vocab_params(spec, seed=4)
        prompt = [3, 5]
        want = _dense_greedy(params, spec, prompt, 5)
        eos = None
        for index in range(1, len(want)):
            if want[index] not in want[:index]:
                eos = want[index]
                break
        self.assertIsNotNone(eos, f"参照序列 {want} 全重复，换一组权重")
        stop = want.index(eos) + 1
        gen = self._generator(spec, params)
        got = gen.generate(prompt, max_new_tokens=5, eos_id=eos, rid=1)
        self.assertEqual(got, want[:stop])

    def test_fork_shares_prefix_and_matches_independent(self):
        spec = _vocab_spec()
        params = _vocab_params(spec, seed=6)
        prefix = [1, 4, 7]
        parent = self._generator(spec, params)
        parent.generate(prefix, max_new_tokens=0, rid=1)
        used = parent.stats()["used_blocks"]
        parent.fork(1, 2)
        info = parent.stats()
        self.assertEqual(info["used_blocks"], used)  # 分叉不额外占页
        self.assertEqual(info["shared_blocks"], used)
        self.assertEqual(info["forks"], 1)
        self.assertEqual(parent.tokens(2), prefix)

        # 子只喂多出来的一个 token，之后与独立实例逐 token 对拍
        shared_tail = [2]
        nxt = parent.prefill(2, shared_tail)
        got = []
        for _ in range(3):
            got.append(nxt)
            nxt = parent.feed(2, nxt)
        other = self._generator(spec, params)
        want = other.generate(prefix + shared_tail, max_new_tokens=3, rid=1)
        self.assertEqual(got, want)
        # 父仍然停在原来的前缀上，没被子序列写脏
        self.assertEqual(parent.length(1), len(prefix))
        self.assertEqual(parent.tokens(1), prefix)

    def test_suspend_evict_and_resume_keep_tokens(self):
        spec = _vocab_spec(seq=8)
        params = _vocab_params(spec, seed=8)
        prompt = [1, 6, 9]
        gen = self._generator(spec, params, num_blocks=4, block_size=2, max_steps=8)
        gen.submit(1)
        got = []
        nxt = gen.prefill(1, prompt)
        for index in range(5):
            got.append(nxt)
            if index == 2:
                gen.suspend(1, evict=True)
                self.assertTrue(gen.is_swapped(1))
                self.assertEqual(gen.stats()["swaps_out"], 1)
                gen.resume(1)
                self.assertFalse(gen.is_swapped(1))
            if index < 4:
                nxt = gen.feed(1, nxt)
        # 中途整段换出再换入，续写的 token 必须与不换出时完全一致
        self.assertEqual(got, _dense_greedy(params, spec, prompt, 5))
        self.assertEqual(gen.stats()["swaps_in"], 1)

    def test_prefill_feed_and_token_history(self):
        spec = _vocab_spec()
        params = _vocab_params(spec, seed=10)
        gen = self._generator(spec, params)
        gen.submit(1)
        nxt = gen.prefill(1, [2, 3, 4])
        self.assertEqual(nxt, _dense_greedy(params, spec, [2, 3, 4], 1)[0])
        self.assertEqual(gen.tokens(1), [2, 3, 4])
        history = gen.tokens(1)
        history.append(99)
        self.assertEqual(gen.tokens(1), [2, 3, 4])  # 返回的是副本

        gen.finish(1)
        self.assertNotIn(1, gen)
        gen.submit(1)  # 同一个 id 可以复用
        self.assertEqual(gen.length(1), 0)
        self.assertEqual(gen.tokens(1), [])

    def test_rejections(self):
        spec = _vocab_spec()
        params = _vocab_params(spec, seed=12)
        with self.assertRaisesRegex(ValueError, "vocab>=1"):
            self._generator(_vocab_spec(vocab=0), params)
        with self.assertRaisesRegex(ValueError, "位置编码表长度"):
            self._generator(spec, params, max_steps=spec.seq + 1)
        bad_pos = dict(params)
        bad_pos["pos"] = params["pos"][0]
        with self.assertRaisesRegex(ValueError, "params\\['pos'\\]"):
            self._generator(spec, bad_pos)
        bad_emb = dict(params)
        bad_emb["tok_emb"] = params["tok_emb"][:, :1]
        with self.assertRaisesRegex(ValueError, "tok_emb"):
            self._generator(spec, bad_emb)
        missing = dict(params)
        del missing["w_head"]
        with self.assertRaisesRegex(KeyError, "w_head"):
            self._generator(spec, missing)

        gen = self._generator(spec, params)
        with self.assertRaisesRegex(ValueError, "至少要有一个 token"):
            gen.generate([], max_new_tokens=2)
        with self.assertRaisesRegex(ValueError, "越界"):
            gen.generate([spec.vocab], max_new_tokens=2)
        with self.assertRaisesRegex(TypeError, "整数 token id"):
            gen.generate([1.5], max_new_tokens=2)
        with self.assertRaisesRegex(TypeError, "整数 token id"):
            gen.generate([True], max_new_tokens=2)
        with self.assertRaisesRegex(TypeError, "prompt 必须是 token id 序列"):
            gen.generate("abc", max_new_tokens=2)
        with self.assertRaisesRegex(TypeError, "max_new_tokens"):
            gen.generate([1], max_new_tokens=True)
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            gen.generate([1], max_new_tokens=-1)
        with self.assertRaisesRegex(ValueError, "超过 max_steps"):
            gen.generate([1, 2, 3, 4], max_new_tokens=9)
        with self.assertRaisesRegex(ValueError, "越界"):
            gen.generate([1], max_new_tokens=1, eos_id=spec.vocab)
        with self.assertRaisesRegex(KeyError, "没有提交"):
            gen.feed(9, 1)
        with self.assertRaisesRegex(KeyError, "没有提交"):
            gen.finish(9)
        with self.assertRaisesRegex(KeyError, "没有提交"):
            gen.generate([1], max_new_tokens=1, fork_from=9)

        gen.submit(1)
        gen.prefill(1, [1, 2])
        with self.assertRaisesRegex(ValueError, "前缀不一致"):
            gen.generate([1, 3], max_new_tokens=1, fork_from=1, rid=2)
        with self.assertRaisesRegex(ValueError, "多至少一个 token"):
            gen.generate([1, 2], max_new_tokens=1, fork_from=1, rid=2)
        with self.assertRaisesRegex(ValueError, "已经提交"):
            gen.submit(1)
        with self.assertRaisesRegex(ValueError, "已经提交"):
            gen.fork(1, 1)
        with self.assertRaisesRegex(ValueError, "达到 max_steps"):
            gen2 = self._generator(spec, params, max_steps=3)
            gen2.submit(1)
            gen2.prefill(1, [1, 2, 3])
            gen2.feed(1, 4)


class TestPrefixCache(unittest.TestCase):
    """5.335 前缀缓存：自动挑最长前缀，命中结果与独立生成逐 token 相同。

    缓存只省 prefill，不改变数学——所以每条命中用例都拿同一个参数集的
    独立实例（不共享任何页）做参照。
    """

    def _generator(self, spec, params, **overrides):
        options = dict(
            num_blocks=max(8, math.ceil(spec.seq / 2)), block_size=2, max_steps=spec.seq
        )
        options.update(overrides)
        return PagedGenerator(BACKEND, params, spec, **options)

    def test_hit_reuses_prefix_and_matches_independent(self):
        spec = _vocab_spec()
        params = _vocab_params(spec, seed=12)
        gen = self._generator(spec, params)
        gen.generate([1, 4, 7], max_new_tokens=0, rid=1)

        prompt = [1, 4, 7, 2, 5]
        got = gen.generate(prompt, max_new_tokens=3, rid=2)
        self.assertEqual(got, _dense_greedy(params, spec, prompt, 3))
        info = gen.stats()
        self.assertEqual(info["prefix_hits"], 1)
        self.assertEqual(info["prefix_reused_tokens"], 3)
        self.assertEqual(info["prefix_misses"], 1)  # 第一条 prompt 没命中

    def test_hit_shares_blocks_instead_of_copying(self):
        spec = _vocab_spec()
        params = _vocab_params(spec, seed=14)
        gen = self._generator(spec, params)
        gen.generate([1, 4, 7], max_new_tokens=0, rid=1)
        used = gen.stats()["used_blocks"]

        gen.generate([1, 4, 7, 2], max_new_tokens=0, rid=2)
        info = gen.stats()
        # 子序列只多一个 token：共享父的两页，至多因写尾页多一页（写时复制）
        self.assertLessEqual(info["used_blocks"] - used, 1)
        self.assertGreater(info["shared_blocks"], 0)
        self.assertEqual(info["forks"], 1)

    def test_longest_prefix_wins(self):
        spec = _vocab_spec()
        params = _vocab_params(spec, seed=16)
        gen = self._generator(spec, params)
        gen.generate([1, 4], max_new_tokens=0, rid=1)
        gen.generate([1, 4, 7], max_new_tokens=0, rid=2)  # 命中 rid=1 的 2 个
        gen.generate([1, 4, 7, 9, 2], max_new_tokens=2, rid=3)  # 命中 rid=2 的 3 个

        info = gen.stats()
        self.assertEqual(info["prefix_hits"], 2)
        self.assertEqual(info["prefix_reused_tokens"], 5)
        self.assertEqual(
            gen.generate([1, 4, 7, 9, 2, 5], max_new_tokens=1, rid=4),
            _dense_greedy(params, spec, [1, 4, 7, 9, 2, 5], 1),
        )

    def test_equal_length_and_swapped_are_not_reused(self):
        spec = _vocab_spec(seq=8)
        params = _vocab_params(spec, seed=18)

        swapped = self._generator(spec, params, num_blocks=4, block_size=2)
        swapped.generate([1, 4, 7], max_new_tokens=0, rid=1)
        swapped.suspend(1, evict=True)
        got = swapped.generate([1, 4, 7, 2], max_new_tokens=1, rid=2)
        self.assertEqual(swapped.stats()["prefix_hits"], 0)
        self.assertEqual(got, _dense_greedy(params, spec, [1, 4, 7, 2], 1))

        equal = self._generator(spec, params)
        equal.generate([1, 4, 7], max_new_tokens=0, rid=1)
        # 等长：没有"尚未解码的位置"，不能从父序列的末尾续写
        equal.generate([1, 4, 7], max_new_tokens=1, rid=2)
        self.assertEqual(equal.stats()["prefix_hits"], 0)

    def test_disabled_cache_keeps_sequences_independent(self):
        spec = _vocab_spec()
        params = _vocab_params(spec, seed=20)
        gen = self._generator(spec, params, prefix_cache=False)
        gen.generate([1, 4, 7], max_new_tokens=0, rid=1)
        used = gen.stats()["used_blocks"]

        gen.generate([1, 4, 7, 2], max_new_tokens=1, rid=2)
        info = gen.stats()
        self.assertEqual(info["prefix_hits"], 0)
        self.assertEqual(info["prefix_misses"], 0)
        self.assertEqual(info["shared_blocks"], 0)
        self.assertGreater(info["used_blocks"], used)


if __name__ == "__main__":
    unittest.main()
