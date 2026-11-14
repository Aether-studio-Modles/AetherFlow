"""SSM 语言模型的闭环验收（审计 AF-34 的图内部分）。

四条判据，都是可执行的：

1. **前向**：logits 形状 ``(B, T, V)``、数值有限、参数表与 ``param_shapes`` 一致。
2. **流式恒等式**：prefill 整段 == 分块续接；逐 token decode == 整段前向。
3. **损失**：masked 交叉熵只对被选中的 token 归一，全 padding 的一批是 0 而不是 NaN。
4. **采样与停止**：贪心 == argmax、top-k 不越界、EOS 立即停止、``max_new_tokens`` 生效。

tokenizer 与 chat template 不在这一轮：它们是数据/协议层，不该借"多一个算子"
来假装已经具备（见 DESIGN 5.119）。
"""

import unittest

import numpy as np

from aetherflow import Graph, ReferenceBackend, backward, f16, f32, i32
from aetherflow.models import (
    LMSpec,
    LMRunner,
    ServingEngine,
    build_language_model,
    build_language_model_stateful,
    generate,
    generate_batch,
    generate_beam,
    masked_lm_loss,
    masked_lm_loss_indexed,
    sample_next_token,
)
from aetherflow.train import lm_batch

BACKEND = ReferenceBackend()


def _small(**overrides):
    base = dict(
        vocab_size=16,
        d_model=8,
        num_layers=1,
        d_state=4,
        d_conv=3,
        expand=2,
        steps=6,
        batch=2,
        tie_weights=True,
        pad_id=0,
        eos_id=1,
    )
    base.update(overrides)
    return LMSpec(**base)


def _ids(spec, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, spec.vocab_size, (spec.batch, spec.steps)).astype(np.int32)


def _zero_states(spec):
    return {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in spec.state_shapes.items()
    }


def _feed_with_states(params, spec, ids):
    feed = dict(params)
    feed["ids"] = ids
    feed.update(_zero_states(spec))
    return feed


class TestLanguageModelForward(unittest.TestCase):
    def test_logits_shape_and_finiteness(self):
        spec = _small()
        g = Graph("lm")
        inputs, logits = build_language_model(g, spec)
        g.output(logits, "logits")

        params = spec.init_params(seed=0)
        out = BACKEND.execute(g, _feed_with_states(params, spec, _ids(spec)))["logits"]
        self.assertEqual(out.shape, (spec.batch, spec.steps, spec.vocab_size))
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertEqual(set(inputs) - {"ids"}, set(params))

    def test_untied_head_has_its_own_parameters(self):
        spec = _small(tie_weights=False)
        g = Graph("lm")
        inputs, _ = build_language_model(g, spec)
        params = spec.init_params(seed=0)
        self.assertIn("head_w", inputs)
        self.assertIn("head_b", inputs)
        self.assertEqual(set(inputs) - {"ids"}, set(params))
        self.assertEqual(params["head_w"].shape, (spec.d_model, spec.vocab_size))

    def test_prompt_prefix_does_not_collide(self):
        """带 prefix 时输入名与参数名一起加前缀，否则两张图会互相覆盖。"""
        spec = _small()
        g = Graph("lm")
        inputs, _ = build_language_model(g, spec, prefix="m_")
        self.assertIn("m_ids", inputs)
        self.assertIn("m_l0_w_in", inputs)
        self.assertNotIn("ids", inputs)

    def test_init_params_match_the_declared_f32(self):
        """审计 P0：嵌入表曾是 f64，f32 master 的 Trainer 会直接拒绝这组参数。"""
        for name, value in _small().init_params(seed=0).items():
            self.assertEqual(value.dtype, np.dtype("float32"), name)

    def test_reset_is_an_optional_input_on_both_entry_points(self):
        g = Graph("lm_reset")
        inputs, _ = build_language_model(g, _small(), with_reset=True)
        self.assertIn("reset", inputs)

        g2 = Graph("lm_reset_stream")
        inputs2, _, _ = build_language_model_stateful(g2, _small(), with_reset=True)
        self.assertIn("reset", inputs2)


class TestStreamingIdentities(unittest.TestCase):
    def _whole(self, spec, ids, params):
        g = Graph("whole")
        _, logits = build_language_model(g, spec)
        g.output(logits, "logits")
        return BACKEND.execute(g, _feed_with_states(params, spec, ids))["logits"]

    def test_stateful_chunks_match_the_whole_sequence(self):
        """分块续接 == 整段：prefill 的状态足以代表全部历史。"""
        spec = _small(steps=6)
        params = spec.init_params(seed=1)
        ids = _ids(spec, seed=2)
        whole = self._whole(spec, ids, params)

        g1 = Graph("chunk1")
        spec1 = spec.with_steps(3)
        in1, logits1, state1 = build_language_model_stateful(g1, spec1)
        g1.output(logits1, "logits")
        for sname, value in state1.items():
            g1.output(value, sname)
        out1 = BACKEND.execute(g1, _feed_with_states(params, spec1, ids[:, :3]))

        g2 = Graph("chunk2")
        spec2 = spec.with_steps(3)
        in2, logits2, state2 = build_language_model_stateful(g2, spec2)
        g2.output(logits2, "logits")
        for sname, value in state2.items():
            g2.output(value, sname)
        feed2 = dict(params)
        feed2["ids"] = ids[:, 3:]
        feed2.update(spec1.carry_state(out1))
        out2 = BACKEND.execute(g2, feed2)

        np.testing.assert_allclose(out1["logits"], whole[:, :3], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out2["logits"], whole[:, 3:], rtol=1e-5, atol=1e-6)

    def test_one_token_at_a_time_matches_prefill(self):
        spec = _small(batch=1, steps=5)
        params = spec.init_params(seed=3)
        ids = _ids(spec, seed=4)
        whole = self._whole(spec, ids, params)

        decode_spec = spec.with_steps(1)
        g = Graph("decode")
        _, logits, state_out = build_language_model_stateful(g, decode_spec)
        g.output(logits, "logits")
        for sname, value in state_out.items():
            g.output(value, sname)

        state = _zero_states(spec)
        rows = []
        for t in range(spec.steps):
            feed = dict(params)
            feed["ids"] = ids[:, t : t + 1]
            feed.update(state)
            out = BACKEND.execute(g, feed)
            rows.append(out["logits"][:, 0, :])
            state = decode_spec.carry_state(out)
        got = np.stack(rows, axis=1)
        np.testing.assert_allclose(got, whole, rtol=1e-5, atol=1e-6)


class TestLeftPaddingIsolation(unittest.TestCase):
    """审计 P0：左填充必须真正隔离 pad，而不是只遮住损失。

    判据有三条，全部可执行：

    1. 左侧补 pad 的行，真实位置上的 logits 与末状态等于该行单独跑；
    2. 改 pad embedding 行，真实位置的 logits/状态逐位不变（pad 的值进不了
       卷积窗口与递归状态）；
    3. 改 pad embedding 行，损失与全部参数梯度逐位不变（pad 位置不贡献梯度）。
    """

    vocab, steps, rows = 16, 6, [[5, 6, 7], [8, 9]]

    def _spec(self, batch=2, steps=None, tie_weights=True):
        return _small(
            batch=batch,
            steps=self.steps if steps is None else steps,
            d_model=8,
            num_layers=1,
            d_conv=3,
            tie_weights=tie_weights,
        )

    def _feed(self, inputs, spec, ids, reset, params):
        feed = dict(params)
        feed[inputs["ids"].name] = ids
        feed[inputs["reset"].name] = reset
        for name, shape in spec.state_shapes.items():
            if name in inputs:
                feed[inputs[name].name] = np.zeros(shape, dtype=np.float32)
        return feed

    def test_padded_rows_match_individual_runs(self):
        spec = self._spec()
        params = spec.init_params(seed=5)
        data = lm_batch(self.rows, self.vocab, steps=self.steps, pad_id=0)

        g = Graph("lm_pad")
        inputs, logits, state = build_language_model_stateful(
            g, spec, with_reset=True
        )
        g.output(logits, "logits")
        for sname, value in state.items():
            g.output(value, sname)
        out = BACKEND.execute(
            g, self._feed(inputs, spec, data["ids"], data["reset"], params)
        )
        got_state = spec.carry_state({k: v for k, v in out.items() if k != "logits"})

        for index, row in enumerate(self.rows):
            one = self._spec(batch=1, steps=len(row))
            g1 = Graph("lm_one")
            in1, logits1, state1 = build_language_model_stateful(
                g1, one, with_reset=True
            )
            g1.output(logits1, "logits")
            for sname, value in state1.items():
                g1.output(value, sname)
            ref = BACKEND.execute(
                g1,
                self._feed(
                    in1,
                    one,
                    np.asarray([row], dtype=np.int32),
                    np.zeros((1, len(row)), dtype=np.float32),
                    params,
                ),
            )
            start = self.steps - len(row)
            np.testing.assert_allclose(
                out["logits"][index, start:], ref["logits"][0],
                rtol=1e-5, atol=1e-6,
                err_msg=f"第 {index} 行（有效长度 {len(row)}）的 logits 与单独跑不一致",
            )
            ref_state = one.carry_state(
                {k: v for k, v in ref.items() if k != "logits"}
            )
            for name in spec.state_shapes:
                np.testing.assert_allclose(
                    got_state[name][index], ref_state[name][0],
                    rtol=1e-5, atol=1e-6,
                    err_msg=f"第 {index} 行的 {name} 与单独跑不一致",
                )

    def test_pad_embedding_does_not_reach_real_positions(self):
        # 解开权重绑定：否则 embed 同时是词表头，改它必然改 logits 的第 0 列
        spec = self._spec(tie_weights=False)
        params = spec.init_params(seed=6)
        data = lm_batch(self.rows, self.vocab, steps=self.steps, pad_id=0)
        altered = {name: np.array(value, copy=True) for name, value in params.items()}
        altered["embed"][0] += 5.0  # pad_id 行整行挪走

        g = Graph("lm_pad_probe")
        inputs, logits, _ = build_language_model_stateful(g, spec, with_reset=True)
        g.output(logits, "logits")

        feed = self._feed(inputs, spec, data["ids"], data["reset"], params)
        feed2 = self._feed(inputs, spec, data["ids"], data["reset"], altered)
        a = BACKEND.execute(g, feed)["logits"]
        b = BACKEND.execute(g, feed2)["logits"]
        for index, row in enumerate(self.rows):
            start = self.steps - len(row)
            np.testing.assert_array_equal(
                a[index, start:], b[index, start:],
                err_msg=f"第 {index} 行：pad embedding 影响了真实位置",
            )

    def test_pad_embedding_gets_no_loss_and_no_gradient(self):
        spec = self._spec(tie_weights=False)
        params = spec.init_params(seed=7)
        data = lm_batch(self.rows, self.vocab, steps=self.steps, pad_id=0)
        altered = {name: np.array(value, copy=True) for name, value in params.items()}
        altered["embed"][0] -= 3.0

        g = Graph("lm_pad_loss")
        inputs, logits = build_language_model(g, spec, with_reset=True)
        targets = g.input("targets", f32[(spec.batch, spec.steps, self.vocab)])
        mask = g.input("mask", f32[(spec.batch, spec.steps)])
        g.output(masked_lm_loss(g, logits, targets, mask), "loss")

        def run(weights):
            feed = self._feed(inputs, spec, data["ids"], data["reset"], weights)
            feed["targets"] = data["targets"]
            feed["mask"] = data["mask"]
            loss = BACKEND.execute(g, feed)["loss"]
            bwd = backward(g, wrt=[inputs["embed"]], active=["loss"])
            grad = BACKEND.execute(bwd, {**feed, "dloss": np.float32(1.0)})[
                "d" + inputs["embed"].name
            ]
            return float(loss), np.array(grad, copy=True)

        loss_a, grad_a = run(params)
        loss_b, grad_b = run(altered)
        self.assertAlmostEqual(loss_a, loss_b, places=6)
        np.testing.assert_array_equal(grad_a, grad_b)
        self.assertTrue(np.all(np.isfinite(grad_a)))


class TestMaskedLoss(unittest.TestCase):
    def _loss(self, logits, targets, mask):
        g = Graph("loss")
        lg = g.input("logits", f32[tuple(int(d) for d in logits.shape)])
        tg = g.input("targets", f32[tuple(int(d) for d in targets.shape)])
        mk = g.input("mask", f32[tuple(int(d) for d in mask.shape)])
        g.output(masked_lm_loss(g, lg, tg, mk), "loss")
        return float(
            BACKEND.execute(g, {"logits": logits, "targets": targets, "mask": mask})[
                "loss"
            ]
        )

    def _one_hot(self, ids, vocab):
        out = np.zeros(ids.shape + (vocab,), dtype=np.float32)
        np.put_along_axis(out, ids[..., None], 1.0, axis=-1)
        return out

    def test_padding_rows_do_not_contribute(self):
        rng = np.random.default_rng(0)
        logits = rng.standard_normal((1, 3, 4)).astype(np.float32)
        ids = np.array([[1, 2, 3]])
        targets = self._one_hot(ids, 4)
        full = np.ones((1, 3), dtype=np.float32)
        masked = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)

        # 把被 mask 的那一行目标改成一个完全不同的词：损失必须不变
        flipped = targets.copy()
        flipped[0, 2] = 0.0
        flipped[0, 2, 0] = 1.0

        np.testing.assert_allclose(
            self._loss(logits, targets, masked),
            self._loss(logits, flipped, masked),
            rtol=1e-6,
        )
        self.assertNotAlmostEqual(
            self._loss(logits, targets, full),
            self._loss(logits, targets, masked),
            places=6,
        )

    def test_fully_masked_batch_is_zero_not_nan(self):
        logits = np.zeros((1, 2, 4), dtype=np.float32)
        targets = self._one_hot(np.array([[1, 2]]), 4)
        value = self._loss(logits, targets, np.zeros((1, 2), dtype=np.float32))
        self.assertTrue(np.isfinite(value))
        self.assertAlmostEqual(value, 0.0)

    def test_f16_fully_masked_batch_is_finite_zero_with_finite_grad(self):
        """f16 下 ``1e-12`` 会下溢成 0；损失内部必须先把输入抬到 f32。"""
        g = Graph("loss16")
        lg = g.input("logits", f16[1, 2, 4])
        tg = g.input("targets", f16[1, 2, 4])
        mk = g.input("mask", f16[1, 2])
        g.output(masked_lm_loss(g, lg, tg, mk), "loss")
        feed = {
            "logits": np.zeros((1, 2, 4), dtype=np.float16),
            "targets": np.zeros((1, 2, 4), dtype=np.float16),
            "mask": np.zeros((1, 2), dtype=np.float16),
        }
        value = float(BACKEND.execute(g, feed)["loss"])
        self.assertTrue(np.isfinite(value))
        self.assertAlmostEqual(value, 0.0)

        bwd = backward(g, wrt=[lg], active=["loss"])
        grad = BACKEND.execute(bwd, {**feed, "dloss": np.float32(1.0)})["dlogits"]
        self.assertTrue(np.all(np.isfinite(grad)))

    def test_matches_uniform_logits_entropy(self):
        """均匀 logits 下交叉熵应正好是 log(V)。"""
        logits = np.zeros((1, 2, 4), dtype=np.float32)
        targets = self._one_hot(np.array([[1, 2]]), 4)
        self.assertAlmostEqual(
            self._loss(logits, targets, np.ones((1, 2), dtype=np.float32)),
            float(np.log(4.0)),
            places=5,
        )


class TestIndexedMaskedLoss(unittest.TestCase):
    """整数标签路径：与 one-hot 是同一损失的另一种编码。

    one-hot 要把 ``B*T*V`` 个 float32 带进图里，V=32K 时根本传不动。整数标签
    只有 ``B*T`` 个 int32。判据是：损失**逐位相等**、梯度在 1 ulp 内一致、
    下标编码能被解回 token，且无标签位不影响结果。
    """

    vocab = 6
    steps = 5
    rows = [[3, 4, 5], [1, 2]]

    def _logits(self):
        rng = np.random.default_rng(7)
        return rng.standard_normal((2, self.steps, self.vocab)).astype(np.float32)

    def _graph(self, loss_fn, target_type):
        g = Graph("indexed")
        lg = g.input("logits", f32[2, self.steps, self.vocab])
        tg = g.input("targets", target_type)
        mk = g.input("mask", f32[2, self.steps])
        g.output(loss_fn(g, lg, tg, mk), "loss")
        return g

    def test_index_round_trips_to_the_one_hot_token(self):
        one = lm_batch(self.rows, self.vocab, steps=self.steps)
        idx = lm_batch(self.rows, self.vocab, steps=self.steps, labels="index")
        self.assertEqual(idx["targets"].dtype, np.int32)
        self.assertEqual(idx["targets"].shape, (2, self.steps))

        flat = one["targets"].reshape(-1, self.vocab)
        labeled = idx["mask"].astype(bool)
        np.testing.assert_array_equal(
            idx["targets"].reshape(-1) % self.vocab, np.argmax(flat, axis=-1)
        )
        # 有标签的位置必须落在自己的 (b, t) 行；无标签位填 0，不参与断言。
        np.testing.assert_array_equal(
            (idx["targets"] // self.vocab)[labeled],
            np.arange(2 * self.steps).reshape(2, self.steps)[labeled],
        )
        # 无标签位填 0，也必须是合法下标：越界会被 embedding 的检查拦下。
        self.assertTrue(np.all(idx["targets"] >= 0))
        self.assertTrue(np.all(idx["targets"] < 2 * self.steps * self.vocab))

    def test_loss_is_bit_identical_and_gradient_matches(self):
        one = lm_batch(self.rows, self.vocab, steps=self.steps)
        idx = lm_batch(self.rows, self.vocab, steps=self.steps, labels="index")
        logits = self._logits()

        g_one = self._graph(masked_lm_loss, f32[2, self.steps, self.vocab])
        g_idx = self._graph(masked_lm_loss_indexed, i32[2, self.steps])
        feed_one = {
            "logits": logits,
            "targets": one["targets"],
            "mask": one["mask"],
        }
        feed_idx = {
            "logits": logits,
            "targets": idx["targets"],
            "mask": idx["mask"],
        }
        self.assertEqual(
            BACKEND.execute(g_one, feed_one)["loss"],
            BACKEND.execute(g_idx, feed_idx)["loss"],
        )

        bwd_one = backward(g_one, wrt=[g_one.find("logits")], active=["loss"])
        bwd_idx = backward(g_idx, wrt=[g_idx.find("logits")], active=["loss"])
        grad_one = BACKEND.execute(bwd_one, {**feed_one, "dloss": np.float32(1.0)})[
            "dlogits"
        ]
        grad_idx = BACKEND.execute(bwd_idx, {**feed_idx, "dloss": np.float32(1.0)})[
            "dlogits"
        ]
        np.testing.assert_allclose(grad_idx, grad_one, rtol=1e-6, atol=1e-7)
        self.assertTrue(np.all(np.isfinite(grad_idx)))

    def test_masked_position_does_not_change_the_loss(self):
        idx = lm_batch(self.rows, self.vocab, steps=self.steps, labels="index")
        logits = self._logits()
        mask = np.zeros((2, self.steps), dtype=np.float32)
        mask[:, 1] = 1.0
        moved = idx["targets"].copy()
        # 只动无标签位的下标：它不该影响损失
        for (row, position) in ((0, 0), (1, 0)):
            self.assertEqual(mask[row, position], 0.0)
            moved[row, position] = (row * self.steps + position) * self.vocab + 5

        def _run(targets):
            g = self._graph(masked_lm_loss_indexed, i32[2, self.steps])
            return float(
                BACKEND.execute(g, {"logits": logits, "targets": targets, "mask": mask})[
                    "loss"
                ]
            )

        self.assertEqual(_run(idx["targets"]), _run(moved))

    def test_fully_masked_batch_is_zero(self):
        idx = lm_batch(self.rows, self.vocab, steps=self.steps, labels="index")
        g = self._graph(masked_lm_loss_indexed, i32[2, self.steps])
        value = float(
            BACKEND.execute(
                g,
                {
                    "logits": self._logits(),
                    "targets": idx["targets"],
                    "mask": np.zeros((2, self.steps), dtype=np.float32),
                },
            )["loss"]
        )
        self.assertTrue(np.isfinite(value))
        self.assertAlmostEqual(value, 0.0)

    def test_wrong_target_shape_is_rejected(self):
        g = Graph("bad")
        lg = g.input("logits", f32[2, self.steps, self.vocab])
        tg = g.input("targets", i32[2, self.steps + 1])
        mk = g.input("mask", f32[2, self.steps])
        with self.assertRaises(ValueError):
            masked_lm_loss_indexed(g, lg, tg, mk)

    def test_float_targets_are_rejected(self):
        g = Graph("bad")
        lg = g.input("logits", f32[2, self.steps, self.vocab])
        tg = g.input("targets", f32[2, self.steps])
        mk = g.input("mask", f32[2, self.steps])
        with self.assertRaises(TypeError):
            masked_lm_loss_indexed(g, lg, tg, mk)

    def test_unknown_labels_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            lm_batch(self.rows, self.vocab, labels="ids")

    def test_index_overflow_is_rejected(self):
        # 检查发生在分配之前：这一批的 one-hot 会是 16 GB，不能先建再报错。
        with self.assertRaises(ValueError):
            lm_batch([[1]], 10**6, steps=4096, labels="index")


class TestSampling(unittest.TestCase):
    def test_greedy_is_argmax(self):
        logits = np.array([[0.1, 3.0, -1.0, 2.0]])
        self.assertEqual(int(sample_next_token(logits, temperature=0.0)[0]), 1)

    def test_top_k_never_samples_outside(self):
        rng = np.random.default_rng(0)
        logits = np.zeros((64, 5), dtype=np.float32)
        logits[:, 4] = -1e6  # 极端劣势的一列，top_k=2 必须完全排除
        picks = sample_next_token(logits, top_k=2, rng=rng)
        self.assertFalse(np.any(picks == 4))

    def test_same_seed_gives_the_same_draw(self):
        logits = np.zeros((8, 6), dtype=np.float32)
        a = sample_next_token(logits, rng=np.random.default_rng(7))
        b = sample_next_token(logits, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)

    def test_non_finite_logits_are_rejected(self):
        logits = np.array([[0.0, np.nan, 1.0]])
        with self.assertRaises(ValueError):
            sample_next_token(logits)

    def test_tiny_temperature_does_not_produce_nan(self):
        """审计 P1：先除温度再减最大值时，``x/T`` 溢出成 inf，``inf-inf`` 是 NaN。"""
        logits = np.array([[0.0, 5.0, -3.0]], dtype=np.float32)
        got = sample_next_token(logits, temperature=1e-300, rng=np.random.default_rng(0))
        self.assertEqual(int(got[0]), 1)

    def test_non_finite_temperature_is_rejected(self):
        logits = np.zeros((1, 3), dtype=np.float32)
        for bad in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "temperature"):
                sample_next_token(logits, temperature=bad)


class _ScriptedBackend:
    """只按脚本返回 logits 与零状态——用来单测 generate 的控制流。"""

    def __init__(self, picks):
        self.picks = list(picks)
        self.calls = 0

    def execute(self, graph, feed):
        out = {}
        for out_name, value in zip(graph.output_names, graph.outputs):
            shape = tuple(int(d) for d in value.type.shape)
            if out_name == "logits":
                buf = np.full(shape, -50.0, dtype=np.float32)
                pick = self.picks[min(self.calls, len(self.picks) - 1)]
                if isinstance(pick, (list, tuple)):
                    # 逐行脚本：第 i 个值只抬高第 i 行，用来验证各行独立停止
                    for row, token in enumerate(pick):
                        buf[row, ..., token] = 50.0
                else:
                    buf[..., pick] = 50.0
                out["logits"] = buf
            else:
                out[out_name] = np.zeros(shape, dtype=np.float32)
        self.calls += 1
        return out


class TestLastPositionHead(unittest.TestCase):
    """推理只需要下一个 token 的分布，不该物化 ``(B, T, V)``。"""

    def _run(self, spec, params, ids, head):
        g = Graph(f"head_{head}")
        inputs, logits, state = build_language_model_stateful(g, spec, head=head)
        self.assertEqual(inputs["ids"].name, "ids")
        if logits is not None:
            g.output(logits, "logits")
        for sname, value in state.items():
            g.output(value, sname)
        feed = dict(params)
        feed["ids"] = ids
        feed.update(_zero_states(spec))
        return BACKEND.execute(g, feed)

    def test_last_row_matches_the_full_projection(self):
        spec = _small()
        params = spec.init_params(seed=5)
        ids = _ids(spec, seed=6)
        full = self._run(spec, params, ids, "all")
        last = self._run(spec, params, ids, "last")

        self.assertEqual(last["logits"].shape, (spec.batch, 1, spec.vocab_size))
        np.testing.assert_allclose(
            last["logits"][:, 0], full["logits"][:, -1], rtol=1e-6, atol=1e-6
        )
        for name in spec.state_shapes:
            np.testing.assert_allclose(
                last[name[:-1]], full[name[:-1]], rtol=1e-6, atol=1e-6
            )

    def test_none_head_carries_state_without_logits(self):
        spec = _small()
        params = spec.init_params(seed=7)
        ids = _ids(spec, seed=8)
        full = self._run(spec, params, ids, "all")
        bare = self._run(spec, params, ids, "none")

        self.assertNotIn("logits", bare)
        for name in spec.state_shapes:
            np.testing.assert_allclose(
                bare[name[:-1]], full[name[:-1]], rtol=1e-6, atol=1e-6
            )

    def test_unknown_head_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "head"):
            build_language_model_stateful(Graph("bad_head"), _small(), head="first")


class TestFixedChunkPrefill(unittest.TestCase):
    """固定块预填充：块长可以远小于 prompt，结果必须与整段一致。"""

    def _spec(self, steps):
        return _small(batch=1, steps=steps)

    def test_chunked_prefill_matches_single_shot(self):
        prompt = [3, 4, 5, 6, 7, 8, 9]
        params = self._spec(7).init_params(seed=9)
        whole = generate(
            BACKEND, params, prompt, self._spec(7), max_new_tokens=4
        )
        for chunk in (2, 3, 4):
            got = generate(
                BACKEND, params, prompt, self._spec(4),
                max_new_tokens=4, chunk_size=chunk,
            )
            self.assertEqual(list(got), list(whole), f"chunk={chunk}")

    def test_exact_multiple_of_the_chunk_size(self):
        prompt = [3, 4, 5, 6, 7, 8]
        params = self._spec(6).init_params(seed=10)
        whole = generate(BACKEND, params, prompt, self._spec(6), max_new_tokens=3)
        got = generate(
            BACKEND, params, prompt, self._spec(3), max_new_tokens=3, chunk_size=3
        )
        self.assertEqual(list(got), list(whole))

    def test_prompt_longer_than_steps_requires_chunking(self):
        spec = self._spec(2)
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            generate(BACKEND, spec.init_params(), [3, 4, 5], spec)

    def test_invalid_chunk_size_is_rejected(self):
        spec = self._spec(4)
        for bad in (0, -1, 5):
            with self.assertRaisesRegex(ValueError, "chunk_size"):
                generate(
                    BACKEND, spec.init_params(), [3, 4], spec,
                    max_new_tokens=1, chunk_size=bad,
                )


class _ResidentRecorder:
    """包一层参考后端，记录哪些输入真的由调用方每步喂进来。"""

    def __init__(self, inner):
        self.inner = inner
        self.resident = {}
        self.fed = []

    def resident_parameters(self, graph, values):
        declared = {value.name for value in graph.inputs}
        unknown = set(values) - declared
        if unknown:
            raise KeyError(sorted(unknown))
        self.resident = dict(values)

    def release_parameters(self):
        self.resident = {}

    def execute(self, graph, feed):
        self.fed.append(set(feed))
        merged = dict(self.resident)
        merged.update(feed)
        return self.inner.execute(graph, merged)


class TestResidentWeights(unittest.TestCase):
    def test_resident_matches_host_feed_and_releases(self):
        spec = _small(batch=1, steps=4)
        params = spec.init_params(seed=11)
        prompt = [3, 4, 5, 6, 7]
        want = generate(
            BACKEND, params, prompt, spec, max_new_tokens=3, chunk_size=2
        )
        rec = _ResidentRecorder(BACKEND)
        got = generate(
            rec, params, prompt, spec, max_new_tokens=3, chunk_size=2, resident=True
        )
        self.assertEqual(list(got), list(want))
        self.assertEqual(rec.resident, {})  # 用完必须释放
        for fed in rec.fed:
            self.assertNotIn("embed", fed)
            self.assertIn("ids", fed)

    def test_non_empty_resident_table_is_rejected(self):
        spec = _small(batch=1, steps=4)
        rec = _ResidentRecorder(BACKEND)
        rec.resident = {"embed": np.zeros((spec.vocab_size, spec.d_model), np.float32)}
        with self.assertRaisesRegex(ValueError, "驻留"):
            generate(
                rec, spec.init_params(), [3], spec,
                max_new_tokens=1, resident=True,
            )

    def test_backend_without_residency_is_rejected(self):
        spec = _small(batch=1, steps=4)
        with self.assertRaisesRegex(ValueError, "resident_parameters"):
            generate(
                _ScriptedBackend([2]), {}, [3], spec,
                max_new_tokens=1, resident=True,
            )


class TestGenerate(unittest.TestCase):
    def _spec(self):
        return _small(batch=1, steps=4)

    def test_eos_stops_immediately(self):
        spec = self._spec()
        backend = _ScriptedBackend([spec.eos_id])
        got = generate(backend, {}, [3], spec, max_new_tokens=5)
        self.assertEqual(list(got), [])
        self.assertEqual(backend.calls, 1)  # 只跑了 prefill

    def test_zero_new_tokens_returns_nothing(self):
        """审计 P1：``max_new_tokens=0`` 曾仍然吐出一个 token。"""
        spec = self._spec()
        backend = _ScriptedBackend([2])
        self.assertEqual(list(generate(backend, {}, [3], spec, max_new_tokens=0)), [])

    def test_max_new_tokens_is_respected(self):
        spec = self._spec()
        backend = _ScriptedBackend([5, 6, 7, 8])
        got = generate(backend, {}, [3], spec, max_new_tokens=2)
        self.assertEqual(list(got), [5, 6])

    def test_eos_after_some_tokens(self):
        spec = self._spec()
        backend = _ScriptedBackend([5, spec.eos_id, 7])
        got = generate(backend, {}, [3], spec, max_new_tokens=5)
        self.assertEqual(list(got), [5])

    def test_non_unit_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            generate(_ScriptedBackend([1]), {}, [3], _small(batch=2), max_new_tokens=1)


class TestBatchGeneration(unittest.TestCase):
    """批量解码：左填充对齐 + 逐行独立停止，每行结果与单条调用一致。"""

    def _spec(self, batch):
        return _small(batch=batch, steps=4)

    def test_ragged_prompts_match_single_row_runs(self):
        prompts = [[3], [3, 4, 5, 6, 7], [3, 4]]
        spec = self._spec(batch=3)
        params = spec.init_params(seed=21)
        want = [
            list(generate(
                BACKEND, params, row, self._spec(batch=1),
                max_new_tokens=3, chunk_size=2,
            ))
            for row in prompts
        ]
        got = generate_batch(
            BACKEND, params, prompts, spec, max_new_tokens=3, chunk_size=2
        )
        self.assertEqual(got, want)

    def test_batch_one_matches_generate_with_sampling(self):
        """同样的 seed 下，batch=1 的批量入口与单条入口必须逐 token 一致。"""
        spec = self._spec(batch=1)
        params = spec.init_params(seed=22)
        prompt = [3, 4, 5, 6, 7]
        kwargs = dict(
            max_new_tokens=4, temperature=0.8, top_k=5, seed=7, chunk_size=2
        )
        want = list(generate(BACKEND, params, prompt, spec, **kwargs))
        got = generate_batch(BACKEND, params, [prompt], spec, **kwargs)
        self.assertEqual(got, [want])

    def test_each_row_stops_on_its_own_eos(self):
        spec = self._spec(batch=2)
        backend = _ScriptedBackend([[5, spec.eos_id], [6, 9], [7, 9]])
        got = generate_batch(backend, {}, [[3], [3, 4]], spec, max_new_tokens=3)
        self.assertEqual(got, [[5, 6, 7], []])

    def test_each_row_respects_max_new_tokens(self):
        spec = self._spec(batch=2)
        backend = _ScriptedBackend([[5, 7], [6, spec.eos_id]])
        got = generate_batch(backend, {}, [[3], [3, 4]], spec, max_new_tokens=2)
        self.assertEqual(got, [[5, 6], [7]])

    def test_zero_new_tokens_returns_empty_rows(self):
        spec = self._spec(batch=2)
        got = generate_batch(BACKEND, spec.init_params(), [[3], [3, 4]], spec,
                             max_new_tokens=0)
        self.assertEqual(got, [[], []])

    def test_resident_batch_matches_and_releases(self):
        spec = self._spec(batch=2)
        params = spec.init_params(seed=23)
        prompts = [[3], [3, 4, 5, 6, 7]]
        want = generate_batch(BACKEND, params, prompts, spec,
                              max_new_tokens=2, chunk_size=2)
        rec = _ResidentRecorder(BACKEND)
        got = generate_batch(rec, params, prompts, spec,
                             max_new_tokens=2, chunk_size=2, resident=True)
        self.assertEqual(got, want)
        self.assertEqual(rec.resident, {})
        for fed in rec.fed:
            self.assertNotIn("embed", fed)

    def test_constraint_is_per_row(self):
        """约束按行生效：受约束行只吐许可 token，未约束行与单条生成一致。"""
        spec = self._spec(batch=2)
        params = spec.init_params(seed=24)
        prompts = [[3], [3, 4]]
        got = generate_batch(
            BACKEND, params, prompts, spec, max_new_tokens=2,
            constraint=[_OnlyToken(3, spec.vocab_size), None],
        )
        self.assertEqual(got[0], [3, 3])
        free = list(generate(
            BACKEND, params, prompts[1], self._spec(batch=1), max_new_tokens=2
        ))
        self.assertEqual(got[1], free)

    def test_row_constraint_sequence_must_match_batch(self):
        spec = self._spec(batch=2)
        params = spec.init_params()
        with self.assertRaisesRegex(ValueError, "受约束"):
            generate_batch(BACKEND, params, [[3], [3, 4]], spec,
                           max_new_tokens=1,
                           constraint=[_OnlyToken(3, spec.vocab_size)])
        with self.assertRaisesRegex(ValueError, "受约束"):
            generate_batch(BACKEND, params, [[3], [3, 4]], spec,
                           max_new_tokens=1,
                           constraint=[_OnlyToken(3, spec.vocab_size), object()])

    def test_unfinished_row_constraint_is_an_error(self):
        spec = self._spec(batch=2)
        params = spec.init_params()
        with self.assertRaisesRegex(ValueError, "第 1 行"):
            generate_batch(
                BACKEND, params, [[3], [3, 4]], spec, max_new_tokens=1,
                constraint=[_OnlyToken(3, spec.vocab_size), _NeverDone()],
            )

    def test_invalid_batch_requests_are_rejected(self):
        spec = self._spec(batch=2)
        params = spec.init_params()
        with self.assertRaisesRegex(ValueError, "prompts"):
            generate_batch(BACKEND, params, [], spec, max_new_tokens=1)
        with self.assertRaisesRegex(ValueError, "spec.batch"):
            generate_batch(BACKEND, params, [[3]], spec, max_new_tokens=1)
        with self.assertRaisesRegex(ValueError, "prompt"):
            generate_batch(BACKEND, params, [[3], []], spec, max_new_tokens=1)
        with self.assertRaisesRegex(ValueError, "受约束"):
            generate_batch(BACKEND, params, [[3], [3, 4]], spec,
                           max_new_tokens=1, constraint=object())
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            generate_batch(BACKEND, params, [[3], [3, 4]], spec,
                           max_new_tokens=1, chunk_size=0)


class TestLMRunner(unittest.TestCase):
    """会话复用：数值与 generate 逐位一致，只是图与驻留不重新建。"""

    def _setup(self):
        spec = _small(batch=1, steps=4)
        return spec, spec.init_params(seed=13), [3, 4, 5, 6, 7]

    def test_matches_generate_and_reuses_graphs(self):
        spec, params, prompt = self._setup()
        want = generate(BACKEND, params, prompt, spec, max_new_tokens=3, chunk_size=2)
        runner = LMRunner(BACKEND, params, spec)
        first = runner.generate(prompt, max_new_tokens=3, chunk_size=2)
        cached = runner.graph_count
        second = runner.generate(prompt, max_new_tokens=3, chunk_size=2)

        self.assertEqual(list(first), list(want))
        self.assertEqual(list(second), list(want))
        # chunk=2、prompt=5：满块（无词表头）与尾块各一张；decode 复用尾块那张
        self.assertEqual(cached, 2)
        self.assertEqual(runner.graph_count, cached)
        # 换一个长度才需要新建尾块图，已有两张仍留在缓存里
        runner.generate([3, 4, 2, 6], max_new_tokens=1, chunk_size=2)
        self.assertEqual(runner.graph_count, 3)
        runner.close()
        self.assertEqual(runner.graph_count, 0)

    def test_graph_cache_is_bounded(self):
        spec, params, _ = self._setup()
        runner = LMRunner(BACKEND, params, spec, max_graphs=2)
        for prompt in ([3], [3, 4], [3, 4, 5], [3, 4, 5, 6]):
            runner.generate(prompt, max_new_tokens=1, chunk_size=1)
            self.assertLessEqual(runner.graph_count, 2)
        runner.close()

    def test_resident_survives_requests_and_is_released_on_close(self):
        spec, params, prompt = self._setup()
        want = generate(BACKEND, params, prompt, spec, max_new_tokens=2, chunk_size=2)
        rec = _ResidentRecorder(BACKEND)
        with LMRunner(rec, params, spec, resident=True) as runner:
            self.assertEqual(
                list(runner.generate(prompt, max_new_tokens=2, chunk_size=2)), list(want)
            )
            self.assertEqual(
                list(runner.generate(prompt, max_new_tokens=2, chunk_size=2)), list(want)
            )
            self.assertNotEqual(rec.resident, {})
            for fed in rec.fed:
                self.assertNotIn("embed", fed)
        # 退出上下文必须释放自己登记的那一份
        self.assertEqual(rec.resident, {})

    def test_zero_new_tokens_builds_no_graph(self):
        spec, params, _ = self._setup()
        runner = LMRunner(BACKEND, params, spec)
        self.assertEqual(
            list(runner.generate([3, 4, 5], max_new_tokens=0, chunk_size=2)), []
        )
        self.assertEqual(runner.graph_count, 0)
        runner.close()

    def test_close_blocks_further_use(self):
        spec, params, prompt = self._setup()
        runner = LMRunner(BACKEND, params, spec)
        runner.generate(prompt, max_new_tokens=1, chunk_size=2)
        runner.close()
        runner.close()  # 可重复调用
        with self.assertRaisesRegex(ValueError, "关闭"):
            runner.generate(prompt, max_new_tokens=1, chunk_size=2)

    def test_rejects_bad_configuration(self):
        spec, params, prompt = self._setup()
        with self.assertRaises(ValueError):
            LMRunner(BACKEND, params, spec, max_graphs=0)
        with self.assertRaises(TypeError):
            LMRunner(BACKEND, params, spec, max_graphs=True)
        with self.assertRaisesRegex(ValueError, "resident_parameters"):
            LMRunner(_ScriptedBackend([2]), {}, spec, resident=True)
        rec = _ResidentRecorder(BACKEND)
        rec.resident = {"embed": np.zeros((spec.vocab_size, spec.d_model), np.float32)}
        with self.assertRaisesRegex(ValueError, "驻留"):
            LMRunner(rec, params, spec, resident=True)

    def test_batch_session_matches_and_reuses_graphs(self):
        spec = _small(batch=2, steps=4)
        params = spec.init_params(seed=17)
        prompts = [[3], [3, 4, 5, 6, 7]]
        want = generate_batch(
            BACKEND, params, prompts, spec, max_new_tokens=2, chunk_size=2
        )
        runner = LMRunner(BACKEND, params, spec)
        first = runner.generate_batch(prompts, max_new_tokens=2, chunk_size=2)
        cached = runner.graph_count
        second = runner.generate_batch(prompts, max_new_tokens=2, chunk_size=2)
        self.assertEqual(first, want)
        self.assertEqual(second, want)
        # width=5、chunk=2：满块（无词表头）与尾块各一张，decode 复用尾块图
        self.assertEqual(cached, 2)
        self.assertEqual(runner.graph_count, cached)
        # 单条入口在非 1 的会话上显式拒绝，并指向正确的入口
        with self.assertRaisesRegex(ValueError, "generate_batch"):
            runner.generate([3], max_new_tokens=1)
        runner.close()

    def test_batch_session_row_constraints_match_plain_call(self):
        """会话复用不改变逐行约束的结果：与无状态入口逐 token 相同。"""
        spec = _small(batch=2, steps=4)
        params = spec.init_params(seed=18)
        prompts = [[3], [3, 4]]
        held = [_OnlyToken(3, spec.vocab_size), None]
        want = generate_batch(
            BACKEND, params, prompts, spec, max_new_tokens=2, constraint=held
        )
        runner = LMRunner(BACKEND, params, spec)
        got = runner.generate_batch(
            prompts, max_new_tokens=2,
            constraint=[_OnlyToken(3, spec.vocab_size), None],
        )
        self.assertEqual(got, want)
        runner.close()

    def test_invalid_request_is_rejected_before_any_graph(self):
        spec, params, _ = self._setup()
        runner = LMRunner(BACKEND, params, spec)
        with self.assertRaises(ValueError):
            runner.generate([], max_new_tokens=1)
        with self.assertRaises(ValueError):
            runner.generate([3, 4], chunk_size=0)
        self.assertEqual(runner.graph_count, 0)
        runner.close()


class _CountingBackend:
    """数每次 execute 真正喂进去的 token 数，用来证明前缀没有被重算。"""

    def __init__(self, inner):
        self.inner = inner
        self.tokens = []

    def execute(self, graph, feed):
        self.tokens.append(int(np.asarray(feed["ids"]).size))
        return self.inner.execute(graph, feed)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class TestSessionPrefix(unittest.TestCase):
    """会话前缀缓存：prime 一次，generate 只算新增 token，结果必须不变。"""

    def _setup(self, seed=31):
        spec = _small(batch=1, steps=8)
        return spec, spec.init_params(seed=seed)

    def test_prime_then_generate_matches_full_prefill(self):
        spec, params = self._setup()
        prompt = [3, 4, 5, 6, 7]
        want = list(generate(
            BACKEND, params, prompt, spec, max_new_tokens=3, chunk_size=2
        ))
        runner = LMRunner(BACKEND, params, spec)
        self.assertEqual(runner.prime(prompt[:3], chunk_size=2), 3)
        got = list(runner.generate(prompt, max_new_tokens=3, chunk_size=2))
        self.assertEqual(got, want)
        runner.close()

    def test_prime_with_sampling_matches(self):
        spec, params = self._setup(seed=35)
        prompt = [3, 4, 5, 6, 7]
        kwargs = dict(max_new_tokens=3, temperature=0.5, top_k=5, chunk_size=2)
        want = list(generate(BACKEND, params, prompt, spec, **kwargs))
        runner = LMRunner(BACKEND, params, spec)
        runner.prime(prompt[:4], chunk_size=2)
        self.assertEqual(list(runner.generate(prompt, **kwargs)), want)
        runner.close()

    def test_generate_on_the_primed_prompt_uses_cached_logits(self):
        """prompt 与缓存前缀等长：不该再喂任何 token，直接采样末位置。"""
        spec, params = self._setup(seed=36)
        prompt = [3, 4, 5, 6]
        want = list(generate(
            BACKEND, params, prompt, spec, max_new_tokens=2, chunk_size=2
        ))
        counting = _CountingBackend(BACKEND)
        runner = LMRunner(counting, params, spec)
        runner.prime(prompt, chunk_size=2)
        counting.tokens.clear()
        self.assertEqual(list(runner.generate(prompt, max_new_tokens=2,
                                              chunk_size=2)), want)
        # 只有 decode 的一步，没有重复预填充
        self.assertEqual(counting.tokens, [1])
        runner.close()

    def test_prime_only_fills_the_increment(self):
        spec, params = self._setup(seed=32)
        prefix = [3, 4, 5, 6]
        counting = _CountingBackend(BACKEND)
        runner = LMRunner(counting, params, spec)
        runner.prime(prefix, chunk_size=2)
        self.assertEqual(sum(counting.tokens), len(prefix))
        counting.tokens.clear()
        runner.generate(prefix + [7], max_new_tokens=2, chunk_size=2)
        # 增量只有 1 个 token；之后只有 1 次 decode
        self.assertEqual(counting.tokens, [1, 1])
        graphs = runner.graph_count
        runner.reset()
        self.assertEqual(runner.graph_count, graphs)  # reset 不动图缓存
        runner.close()

    def test_prefix_must_remain_a_true_prefix(self):
        spec, params = self._setup(seed=33)
        runner = LMRunner(BACKEND, params, spec)
        runner.prime([3, 4, 5], chunk_size=2)
        with self.assertRaisesRegex(ValueError, "前缀"):
            runner.prime([3, 4], chunk_size=2)
        with self.assertRaisesRegex(ValueError, "前缀"):
            runner.generate([3, 4, 9, 10], max_new_tokens=1)
        runner.reset()
        want = list(generate(
            BACKEND, params, [3, 4, 9, 10], spec, max_new_tokens=1
        ))
        self.assertEqual(
            list(runner.generate([3, 4, 9, 10], max_new_tokens=1)), want
        )
        runner.close()

    def test_zero_new_tokens_keeps_the_cache(self):
        spec, params = self._setup(seed=37)
        prompt = [3, 4, 5, 6, 7]
        want = list(generate(
            BACKEND, params, prompt, spec, max_new_tokens=2, chunk_size=2
        ))
        runner = LMRunner(BACKEND, params, spec)
        runner.prime(prompt[:3], chunk_size=2)
        self.assertEqual(
            list(runner.generate(prompt, max_new_tokens=0, chunk_size=2)), []
        )
        self.assertEqual(
            list(runner.generate(prompt, max_new_tokens=2, chunk_size=2)), want
        )
        runner.close()

    def test_prefix_is_batch_one_only(self):
        spec2 = _small(batch=2, steps=8)
        runner2 = LMRunner(BACKEND, spec2.init_params(seed=34), spec2)
        with self.assertRaisesRegex(ValueError, "batch=1"):
            runner2.prime([3, 4], chunk_size=2)
        runner2.close()

        spec, params = self._setup(seed=34)
        runner = LMRunner(BACKEND, params, spec)
        runner.prime([3, 4], chunk_size=2)
        with self.assertRaisesRegex(ValueError, "reset"):
            runner.generate_batch([[3, 4]], max_new_tokens=1)
        runner.close()

    def test_prime_and_reset_follow_close(self):
        spec, params = self._setup(seed=38)
        runner = LMRunner(BACKEND, params, spec)
        runner.close()
        with self.assertRaisesRegex(ValueError, "关闭"):
            runner.prime([3, 4])
        with self.assertRaisesRegex(ValueError, "关闭"):
            runner.reset()


class TestNucleusAndPenalty(unittest.TestCase):
    """采样面：top-p 截断与重复惩罚，都只动主机侧 logits。"""

    def test_top_p_keeps_only_the_nucleus(self):
        # probs ≈ (.645, .237, .087, .032)；top_p=.7 只应留下前两个
        logits = np.tile(np.array([[3.0, 2.0, 1.0, 0.0]]), (256, 1))
        picks = sample_next_token(logits, top_p=0.7, rng=np.random.default_rng(0))
        self.assertTrue(np.all(picks <= 1), f"采到了核外 token：{np.unique(picks)}")

    def test_top_p_one_is_a_no_op(self):
        logits = np.zeros((16, 5), dtype=np.float32)
        a = sample_next_token(logits, top_p=1.0, rng=np.random.default_rng(3))
        b = sample_next_token(logits, rng=np.random.default_rng(3))
        np.testing.assert_array_equal(a, b)

    def test_top_p_keeps_the_argmax_even_below_its_own_mass(self):
        logits = np.array([[0.0, 0.0, 10.0]])
        for u in (0.01, 0.05, 0.5):
            got = sample_next_token(logits, top_p=u, rng=np.random.default_rng(1))
            self.assertEqual(int(got[0]), 2)

    def test_repetition_penalty_lowers_a_seen_token(self):
        logits = np.array([[0.0, 4.0, 3.0]])
        plain = sample_next_token(logits, temperature=0.0, top_p=1.0)
        self.assertEqual(int(plain[0]), 1)
        seeded = sample_next_token(
            logits, temperature=0.0, repetition_penalty=4.0, recent=[1]
        )
        # 4/4 = 1 < 3，贪心改选 2；乘性惩罚对负 logit 同样生效
        self.assertEqual(int(seeded[0]), 2)

    def test_repetition_penalty_requires_history(self):
        with self.assertRaisesRegex(ValueError, "recent"):
            sample_next_token(np.zeros((1, 4)), repetition_penalty=2.0)

    def test_invalid_sampling_parameters_are_rejected(self):
        logits = np.zeros((1, 4))
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "top_p"):
                sample_next_token(logits, top_p=bad)
        for bad in (0.0, -2.0, float("nan")):
            with self.assertRaisesRegex(ValueError, "repetition_penalty"):
                sample_next_token(logits, repetition_penalty=bad, recent=[0])
        with self.assertRaisesRegex(ValueError, "越界"):
            sample_next_token(logits, repetition_penalty=2.0, recent=[9])


class _TableBackend:
    """按"本次喂进去的最后一个 token"查表返回 logits；状态恒零。

    这样就能构造一个确定性模型："贪心第一步更好、但第二步走进死胡同"，
    用来验证束搜索真的在搜索而不是逐位贪心。
    """

    def __init__(self, table):
        self.table = {int(k): np.asarray(v, dtype=np.float32) for k, v in table.items()}

    def execute(self, graph, feed):
        ids = next(value for name, value in feed.items() if name.endswith("ids"))
        out = {}
        for out_name, value in zip(graph.output_names, graph.outputs):
            shape = tuple(int(d) for d in value.type.shape)
            if out_name == "logits":
                buf = np.zeros(shape, dtype=np.float32)
                for row in range(shape[0]):
                    buf[row, -1] = self.table[int(ids[row, -1])]
                out["logits"] = buf
            else:
                out[out_name] = np.zeros(shape, dtype=np.float32)
        return out


class TestBeamSearch(unittest.TestCase):
    """束搜索：选路规则、退化到贪心、以及非法参数。"""

    # 3 之后：2 稍好于 4；但 2 之后只能拿到 0，而 4 之后 0 的概率高得多。
    TABLE = {
        3: [0.0, 0.0, 2.0, 0.0, 1.5, 0.0],
        2: [0.0, 0.0, 0.0, 0.0, 0.0, -5.0],
        4: [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        0: [0.0] * 6,
        1: [0.0] * 6,
        5: [0.0] * 6,
    }

    def _spec(self, **overrides):
        return _small(batch=1, steps=2, vocab_size=6, **overrides)

    def test_beam_finds_a_better_path_than_greedy(self):
        spec = self._spec()
        backend = _TableBackend(self.TABLE)
        params = spec.init_params(seed=0)
        greedy = generate(backend, params, [3], spec, max_new_tokens=2)
        got = generate_beam(
            backend, params, [3], spec, num_beams=2, max_new_tokens=2,
            length_penalty=0.0,
        )
        self.assertEqual(list(greedy), [2, 0])
        self.assertEqual(list(got), [4, 0])

    def test_one_beam_matches_greedy_decode(self):
        spec = self._spec()
        backend = _TableBackend(self.TABLE)
        params = spec.init_params(seed=0)
        for limit in (1, 2, 3):
            greedy = generate(backend, params, [3], spec, max_new_tokens=limit)
            beam = generate_beam(
                backend, params, [3], spec, num_beams=1, max_new_tokens=limit
            )
            self.assertEqual(list(beam), list(greedy), f"max_new_tokens={limit}")

    def test_eos_is_not_returned_and_early_stopping_works(self):
        table = {
            3: [0.0, 4.0, 1.0, 0.0, 0.0, 0.0],  # 第一个位置就倾向 EOS
            0: [0.0] * 6,
            1: [0.0] * 6,
            2: [0.0] * 6,
            4: [0.0] * 6,
            5: [0.0] * 6,
        }
        spec = self._spec()
        backend = _TableBackend(table)
        params = spec.init_params(seed=0)
        got = generate_beam(
            backend, params, [3], spec, num_beams=3, max_new_tokens=2
        )
        self.assertEqual(list(got), [])

    def test_zero_new_tokens_returns_nothing(self):
        spec = self._spec()
        backend = _TableBackend(self.TABLE)
        got = generate_beam(
            backend, spec.init_params(seed=0), [3], spec,
            num_beams=2, max_new_tokens=0,
        )
        self.assertEqual(list(got), [])

    def test_invalid_beam_arguments_are_rejected(self):
        spec = self._spec()
        backend = _TableBackend(self.TABLE)
        params = spec.init_params(seed=0)
        with self.assertRaisesRegex(ValueError, "num_beams"):
            generate_beam(backend, params, [3], spec, num_beams=0)
        with self.assertRaisesRegex(TypeError, "num_beams"):
            generate_beam(backend, params, [3], spec, num_beams=True)
        with self.assertRaisesRegex(TypeError, "num_beams"):
            generate_beam(backend, params, [3], spec, num_beams=2.5)
        with self.assertRaisesRegex(ValueError, "length_penalty"):
            generate_beam(backend, params, [3], spec, length_penalty=-1.0)
        with self.assertRaisesRegex(ValueError, "repetition_penalty"):
            generate_beam(backend, params, [3], spec, repetition_penalty=0.0)
        with self.assertRaisesRegex(ValueError, "prompt"):
            generate_beam(backend, params, [], spec)
        with self.assertRaisesRegex(ValueError, "batch=1"):
            generate_beam(backend, params, [3], _small(batch=2), num_beams=2)

    def test_resident_beam_matches_and_releases(self):
        spec = self._spec()
        backend = _TableBackend(self.TABLE)
        backend.resident_parameters = lambda graph, params: None
        backend.release_parameters = lambda: None
        backend.resident = False
        params = spec.init_params(seed=0)
        plain = generate_beam(
            backend, params, [3], spec, num_beams=2, max_new_tokens=2,
            length_penalty=0.0,
        )
        kept = generate_beam(
            backend, params, [3], spec, num_beams=2, max_new_tokens=2,
            length_penalty=0.0, resident=True,
        )
        self.assertEqual(list(plain), list(kept))

    def test_runner_beam_matches_module_level_and_shares_the_cache(self):
        spec = self._spec()
        backend = _TableBackend(self.TABLE)
        backend.resident_parameters = lambda graph, params: None
        backend.release_parameters = lambda: None
        backend.resident = False
        params = spec.init_params(seed=0)
        expected = generate_beam(
            backend, params, [3], spec, num_beams=2, max_new_tokens=2,
            length_penalty=0.0,
        )
        runner = LMRunner(backend, params, spec)
        self.assertEqual(
            list(runner.generate_beam(
                [3], num_beams=2, max_new_tokens=2, length_penalty=0.0
            )),
            list(expected),
        )
        before = runner.graph_count
        self.assertEqual(
            list(runner.generate_beam(
                [3], num_beams=2, max_new_tokens=2, length_penalty=0.0
            )),
            list(expected),
        )
        self.assertEqual(runner.graph_count, before)
        # 同一会话里 batch 不同的图共存，互不覆盖
        kind = _TableBackend(self.TABLE)
        self.assertEqual(list(runner.generate([3], max_new_tokens=2)),
                         list(generate(kind, params, [3], spec, max_new_tokens=2)))
        runner.close()

    def test_beam_with_full_width_matches_exhaustive_search(self):
        """束宽 >= 词表、步数 2 时束搜索是穷举：分数必须与暴力搜索一致。

        这条用例真正走的是**状态重排**：宽度 16 的 batch 里每步都有 k 行
        带着各自不同的 SSM 状态，父下标取错时分数立刻对不上。
        """
        vocab, prompt, limit = 16, [3, 4], 2
        spec = _small(batch=1, steps=len(prompt) + limit, vocab_size=vocab)
        params = spec.init_params(seed=11)
        beam = generate_beam(
            BACKEND, params, prompt, spec, num_beams=vocab,
            max_new_tokens=limit, length_penalty=0.0, eos_id=-1,
        )
        self.assertEqual(len(beam), limit)

        def score(tokens):
            g = Graph("score")
            _, logits = build_language_model(g, spec)
            g.output(logits, "logits")
            ids = np.asarray([prompt + list(tokens)], dtype=np.int32)
            out = BACKEND.execute(g, _feed_with_states(params, spec, ids))["logits"][0]
            got = 0.0
            for index, token in enumerate(tokens):
                row = np.asarray(out[len(prompt) - 1 + index], dtype=np.float64)
                row = row - np.max(row)
                row = row - np.log(np.sum(np.exp(row)))
                got += float(row[token])
            return got

        best = max(score([a, b]) for a in range(vocab) for b in range(vocab))
        self.assertAlmostEqual(score(beam), best, places=5)
        greedy = generate(
            BACKEND, params, prompt, spec, max_new_tokens=limit, eos_id=-1
        )
        self.assertLessEqual(score(greedy), score(beam) + 1e-9)


class TestGenerationPenalty(unittest.TestCase):
    """generate/generate_batch 把惩罚历史按行维护：prompt + 该行生成的 token。"""

    TABLE = {
        3: [0.0, 0.0, 1.0, 0.0, 0.9, 0.0],
        2: [0.0, 0.0, 2.0, 0.0, 0.5, 0.0],
        4: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        0: [0.0] * 6,
        1: [0.0] * 6,
        5: [0.0] * 6,
    }

    def test_generate_penalises_its_own_history(self):
        spec = _small(batch=1, steps=2, vocab_size=6)
        backend = _TableBackend(self.TABLE)
        params = spec.init_params(seed=0)
        plain = generate(backend, params, [3], spec, max_new_tokens=2)
        self.assertEqual(list(plain), [2, 2])
        penalised = generate(
            backend, params, [3], spec, max_new_tokens=2, repetition_penalty=5.0
        )
        self.assertEqual(list(penalised), [2, 4])

    def test_batch_rows_keep_separate_histories(self):
        spec = _small(batch=2, steps=2, vocab_size=6)
        backend = _TableBackend(self.TABLE)
        params = spec.init_params(seed=0)
        rows = generate_batch(
            backend, params, [[3], [3]], spec,
            max_new_tokens=2, repetition_penalty=5.0,
        )
        self.assertEqual([list(row) for row in rows], [[2, 4], [2, 4]])


class TestBeamRunnerGuards(unittest.TestCase):
    """LMRunner.generate_beam 的显式拒绝：多 batch 会话、已有缓存前缀。"""

    def test_beam_rejects_a_batch_session(self):
        spec = _small(batch=2, steps=2)
        runner = LMRunner(BACKEND, spec.init_params(seed=0), spec)
        with self.assertRaisesRegex(ValueError, "batch=1"):
            runner.generate_beam([3, 4], num_beams=2)
        runner.close()

    def test_beam_rejects_a_primed_prefix(self):
        spec = _small(batch=1, steps=4)
        runner = LMRunner(BACKEND, spec.init_params(seed=0), spec)
        runner.prime([3, 4])
        with self.assertRaisesRegex(ValueError, "reset"):
            runner.generate_beam([3, 4], num_beams=2)
        runner.close()


class _OnlyToken:
    """只允许一个 token 的玩具约束：验证约束是**按请求**生效的。

    ``feed`` 之后把 ``done`` 置真：与 ``JsonGrammar`` 一样，走完文法才允许
    结束；批量路径会逐行检查这一位。
    """

    def __init__(self, token, vocab):
        self.token = int(token)
        self.vocab = int(vocab)
        self.done = False

    def allowed(self):
        return np.asarray([self.token], dtype=np.int64)

    def feed(self, token):
        if int(token) != self.token:
            raise AssertionError(f"约束被违反：{token}")
        self.done = True


class _NeverDone:
    """永不完成的玩具文法：批量路径必须把"没走完"当成错误而不是半截结果。"""

    done = False

    def allowed(self):
        return np.asarray([3], dtype=np.int64)

    def feed(self, token):
        pass


class TestServingEngine(unittest.TestCase):
    """请求队列 + 连续批处理：调度不改变输出，只改变谁和谁一起跑。"""

    def _spec(self, **overrides):
        return _small(batch=1, steps=6, **overrides)

    def test_outputs_match_generate_at_every_capacity(self):
        """容量 1 / 2 / 4 下逐位相同，且都等于逐个 generate。"""
        spec = self._spec()
        params = spec.init_params(seed=7)
        prompts = [[2], [4, 5], [6, 7, 8]]
        want = [
            list(generate(
                BACKEND, params, prompt, spec,
                max_new_tokens=4, temperature=1.0, seed=index,
            ))
            for index, prompt in enumerate(prompts)
        ]
        for capacity in (1, 2, 4):
            with self.subTest(capacity=capacity):
                engine = ServingEngine(BACKEND, params, spec, capacity=capacity)
                ids = [
                    engine.submit(
                        prompt, max_new_tokens=4, temperature=1.0, seed=index
                    )
                    for index, prompt in enumerate(prompts)
                ]
                got = dict(engine.run())
                self.assertEqual([got[rid] for rid in ids], want)
                self.assertEqual(engine.pending, 0)
                self.assertEqual(engine.active, 0)
                engine.close()

    def test_capacity_caps_concurrency_and_admits_mid_flight(self):
        spec = self._spec()
        params = spec.init_params(seed=8)
        engine = ServingEngine(BACKEND, params, spec, capacity=2)
        first = engine.submit([2], max_new_tokens=5, temperature=0.0)
        engine.step()  # 预填充，还没出 token
        self.assertEqual(engine.active, 1)
        self.assertEqual(engine.pending, 0)
        second = engine.submit([4, 5], max_new_tokens=1, temperature=0.0)
        self.assertEqual(engine.pending, 1)
        completed = []
        for _ in range(40):
            completed.extend(engine.step())
            self.assertLessEqual(engine.active, 2)
            if not engine.pending and engine.active == 0:
                break
        ids = [rid for rid, _ in completed]
        self.assertIn(first, ids)
        self.assertIn(second, ids)
        self.assertEqual(dict(completed)[second],
                         list(generate(BACKEND, params, [4, 5], spec,
                                       max_new_tokens=1)))
        engine.close()

    def test_zero_tokens_finishes_without_building_a_graph(self):
        spec = self._spec()
        engine = ServingEngine(BACKEND, spec.init_params(seed=9), spec)
        rid = engine.submit([2], max_new_tokens=0)
        self.assertEqual(engine.pending, 0)
        self.assertEqual(engine.active, 0)
        self.assertEqual(engine.step(), [(rid, [])])
        self.assertEqual(engine.step(), [])
        engine.close()

    def test_eos_stops_only_its_own_request(self):
        spec = self._spec()
        params = spec.init_params(seed=10)
        prompts = [[2], [4, 5]]
        want = [
            list(generate(BACKEND, params, prompt, spec, max_new_tokens=5))
            for prompt in prompts
        ]
        engine = ServingEngine(BACKEND, params, spec, capacity=2)
        ids = [engine.submit(prompt, max_new_tokens=5) for prompt in prompts]
        got = dict(engine.run())
        self.assertEqual([got[rid] for rid in ids], want)
        for row in want:
            self.assertNotIn(spec.eos_id, row)
        engine.close()

    def test_constraint_is_per_request(self):
        spec = self._spec(vocab_size=8)
        params = spec.init_params(seed=11)
        engine = ServingEngine(BACKEND, params, spec, capacity=2)
        constrained = engine.submit(
            [2], max_new_tokens=3, constraint=_OnlyToken(3, spec.vocab_size)
        )
        free = engine.submit([2], max_new_tokens=3)
        got = dict(engine.run())
        self.assertEqual(got[constrained], [3, 3, 3])
        self.assertEqual(got[free],
                         list(generate(BACKEND, params, [2], spec, max_new_tokens=3)))
        engine.close()

    def test_resident_matches_plain(self):
        spec = self._spec()
        params = spec.init_params(seed=12)
        want = list(generate(BACKEND, params, [2, 3], spec, max_new_tokens=3))
        plain = ServingEngine(BACKEND, params, spec)
        rid = plain.submit([2, 3], max_new_tokens=3)
        self.assertEqual(dict(plain.run())[rid], want)
        plain.close()

        rec = _ResidentRecorder(BACKEND)
        kept = ServingEngine(rec, params, spec, resident=True)
        rid = kept.submit([2, 3], max_new_tokens=3)
        self.assertEqual(dict(kept.run())[rid], want)
        self.assertNotEqual(rec.resident, {})
        kept.close()
        self.assertEqual(rec.resident, {})

    def test_invalid_configuration_is_rejected(self):
        spec = self._spec()
        params = spec.init_params(seed=13)
        with self.assertRaisesRegex(ValueError, "capacity"):
            ServingEngine(BACKEND, params, spec, capacity=0)
        with self.assertRaisesRegex(TypeError, "capacity"):
            ServingEngine(BACKEND, params, spec, capacity=True)
        with self.assertRaisesRegex(TypeError, "capacity"):
            ServingEngine(BACKEND, params, spec, capacity=2.5)
        with self.assertRaisesRegex(ValueError, "batch=1"):
            ServingEngine(BACKEND, params, _small(batch=2, steps=4))
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            ServingEngine(BACKEND, params, spec, chunk_size=0)
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            ServingEngine(BACKEND, params, spec, chunk_size=spec.steps + 1)

        engine = ServingEngine(BACKEND, params, spec)
        with self.assertRaisesRegex(ValueError, "prompt"):
            engine.submit([])
        with self.assertRaisesRegex(ValueError, "top_p"):
            engine.submit([2], top_p=0.0)
        with self.assertRaisesRegex(ValueError, "repetition_penalty"):
            engine.submit([2], repetition_penalty=-1.0)
        engine.close()
        with self.assertRaisesRegex(ValueError, "已关闭"):
            engine.submit([2])
        with self.assertRaisesRegex(ValueError, "已关闭"):
            engine.step()


if __name__ == "__main__":
    unittest.main()
