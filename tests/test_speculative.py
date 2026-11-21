"""投机解码的闭环验收。

核心判据只有一条：**草稿再差也改不了结果**。

对任意草稿（完美的、恒定的、故意猜错的、返回停止符的），
``SpeculativeDecoder.generate`` 的输出都必须与 ``generate(temperature=0)``
逐位相同。另外验证：

- 草稿与目标同构时验收应当一路通过（``accepted == proposed``）；
- 快照 / 回滚在所有页宽（这里是 ``draft_tokens`` 与 ``spec.steps`` 的组合）
  下都不留状态漂移；
- 驻留权重、图缓存、关闭后的重用行为与 ``LMRunner`` 同一套口径。
"""

import unittest


from aetherflow import ReferenceBackend
from aetherflow.models import (
    LMSpec,
    SpeculativeDecoder,
    generate,
    greedy_draft,
)

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
        batch=1,
        tie_weights=True,
        pad_id=0,
        eos_id=1,
    )
    base.update(overrides)
    return LMSpec(**base)


class _ResidentRecorder:
    """包一层参考后端，记录驻留权重与每步实际喂进来的输入名。"""

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


class TestSpeculativeMatchesGreedy(unittest.TestCase):
    def _setup(self, **overrides):
        spec = _small(**overrides)
        return spec, spec.init_params(seed=21)

    def test_perfect_draft_matches_generate_and_gets_accepted(self):
        spec, params = self._setup()
        for prompt in ([3], [3, 4], [5, 6, 7]):
            want = list(generate(BACKEND, params, prompt, spec, max_new_tokens=5))
            with SpeculativeDecoder(BACKEND, params, spec) as decoder:
                got = decoder.generate(
                    prompt,
                    greedy_draft(BACKEND, params, spec, chunk_size=spec.steps),
                    max_new_tokens=5,
                    draft_tokens=3,
                )
            with self.subTest(prompt=prompt):
                self.assertEqual(got, want)
                if spec.eos_id not in want:
                    self.assertGreater(decoder.last_stats["accepted"], 0)
                    self.assertEqual(
                        decoder.last_stats["accepted"],
                        decoder.last_stats["proposed"],
                    )
                    # 一次前向确认多个 token：轮数少于产出 token 数
                    self.assertLess(decoder.last_stats["rounds"], 5)

    def test_bad_draft_cannot_change_the_result(self):
        spec, params = self._setup()
        prompt = [3, 4]
        want = list(generate(BACKEND, params, prompt, spec, max_new_tokens=6))

        def constant(token):
            return lambda history, k: [token] * k

        for draft in (constant(0), constant(spec.vocab_size - 1), lambda h, k: [h[-1]]):
            with self.subTest(draft=draft):
                with SpeculativeDecoder(BACKEND, params, spec) as decoder:
                    got = decoder.generate(
                        prompt, draft, max_new_tokens=6, draft_tokens=2
                    )
                self.assertEqual(got, want)

    def test_draft_of_stop_token_does_not_leak_into_output(self):
        spec, params = self._setup()
        prompt = [3]
        want = list(generate(BACKEND, params, prompt, spec, max_new_tokens=4))

        def stop_draft(history, k):
            return [spec.eos_id] * k

        with SpeculativeDecoder(BACKEND, params, spec) as decoder:
            got = decoder.generate(prompt, stop_draft, max_new_tokens=4)
        self.assertEqual(got, want)

    def test_empty_draft_falls_back_to_the_target_token(self):
        spec, params = self._setup()
        prompt = [3, 4]
        want = list(generate(BACKEND, params, prompt, spec, max_new_tokens=4))
        with SpeculativeDecoder(BACKEND, params, spec) as decoder:
            got = decoder.generate(
                prompt, lambda h, k: [], max_new_tokens=4, draft_tokens=3
            )
            self.assertEqual(decoder.last_stats["proposed"],
                             decoder.last_stats["rounds"])
        self.assertEqual(got, want)

    def test_draft_tokens_and_steps_combinations(self):
        spec, params = self._setup(steps=3)
        prompt = [5, 6]
        want = list(
            generate(BACKEND, params, prompt, spec, max_new_tokens=5, chunk_size=3)
        )
        for draft_tokens in (1, 2, 3, 5):
            for draft in (
                greedy_draft(BACKEND, params, spec, chunk_size=3),
                lambda h, k: [0] * k,
            ):
                with self.subTest(draft_tokens=draft_tokens):
                    with SpeculativeDecoder(BACKEND, params, spec) as decoder:
                        got = decoder.generate(
                            prompt,
                            draft,
                            max_new_tokens=5,
                            draft_tokens=draft_tokens,
                            chunk_size=3,
                        )
                    self.assertEqual(got, want)

    def test_long_prompt_with_chunked_prefill(self):
        spec, params = self._setup(steps=3)
        prompt = [5, 6, 7, 8, 9]
        want = list(
            generate(BACKEND, params, prompt, spec, max_new_tokens=4, chunk_size=2)
        )
        with SpeculativeDecoder(BACKEND, params, spec) as decoder:
            got = decoder.generate(
                prompt,
                greedy_draft(BACKEND, params, spec, chunk_size=2),
                max_new_tokens=4,
                draft_tokens=2,
                chunk_size=2,
            )
        self.assertEqual(got, want)

    def test_zero_tokens_builds_no_graph(self):
        spec, params = self._setup()
        with SpeculativeDecoder(BACKEND, params, spec) as decoder:
            got = decoder.generate(
                [3], lambda h, k: [0], max_new_tokens=0
            )
            self.assertEqual(got, [])
            self.assertEqual(decoder.graph_count, 0)

    def test_resident_matches_plain_and_releases(self):
        spec, params = self._setup()
        prompt = [3, 4]
        want = list(generate(BACKEND, params, prompt, spec, max_new_tokens=3))
        recorder = _ResidentRecorder(BACKEND)
        decoder = SpeculativeDecoder(recorder, params, spec, resident=True)
        rid_draft = greedy_draft(recorder, params, spec, chunk_size=spec.steps)
        got = decoder.generate(prompt, rid_draft, max_new_tokens=3, draft_tokens=2)
        self.assertEqual(got, want)
        self.assertNotEqual(recorder.resident, {})
        decoder.close()
        self.assertEqual(recorder.resident, {})
        decoder.close()  # 幂等
        with self.assertRaisesRegex(ValueError, "已关闭"):
            decoder.generate(prompt, rid_draft)

    def test_rejections(self):
        spec, params = self._setup()
        prompt = [3]
        with self.assertRaisesRegex(ValueError, "batch=1"):
            SpeculativeDecoder(BACKEND, params, _small(batch=2))
        with self.assertRaisesRegex(ValueError, "max_graphs"):
            SpeculativeDecoder(BACKEND, params, spec, max_graphs=0)
        with self.assertRaisesRegex(TypeError, "max_graphs"):
            SpeculativeDecoder(BACKEND, params, spec, max_graphs=True)
        with self.assertRaisesRegex(ValueError, "resident"):
            SpeculativeDecoder(BACKEND, {}, spec, resident=True)

        with SpeculativeDecoder(BACKEND, params, spec) as decoder:
            with self.assertRaisesRegex(TypeError, "draft"):
                decoder.generate(prompt, object())
            with self.assertRaisesRegex(ValueError, "draft_tokens"):
                decoder.generate(prompt, lambda h, k: [0], draft_tokens=0)
            with self.assertRaisesRegex(TypeError, "draft_tokens"):
                decoder.generate(prompt, lambda h, k: [0], draft_tokens=True)
            with self.assertRaisesRegex(ValueError, "超过要求"):
                decoder.generate(
                    prompt, lambda h, k: [0] * (k + 1), draft_tokens=2
                )
            with self.assertRaisesRegex(TypeError, "必须是 int"):
                decoder.generate(prompt, lambda h, k: [0.5])
            with self.assertRaisesRegex(ValueError, "越界"):
                decoder.generate(
                    prompt, lambda h, k: [spec.vocab_size]
                )
            with self.assertRaisesRegex(ValueError, "chunk_size"):
                decoder.generate(prompt, lambda h, k: [0], chunk_size=0)

    def test_greedy_draft_matches_generate(self):
        spec, params = self._setup()
        prompt = [3, 4]
        draft = greedy_draft(BACKEND, params, spec)
        self.assertEqual(
            list(draft(prompt, 3)),
            list(generate(BACKEND, params, prompt, spec, max_new_tokens=3)),
        )

    def test_state_does_not_drift_across_sessions(self):
        """同一 prompt 用不同草稿跑两次，结果一致；会话间没有共享状态。"""
        spec, params = self._setup()
        prompt = [4, 5]
        want = list(generate(BACKEND, params, prompt, spec, max_new_tokens=4))
        outputs = []
        for draft in (
            greedy_draft(BACKEND, params, spec, chunk_size=spec.steps),
            lambda h, k: [spec.eos_id - 1] * k,
        ):
            with SpeculativeDecoder(BACKEND, params, spec) as decoder:
                outputs.append(
                    decoder.generate(prompt, draft, max_new_tokens=4, draft_tokens=2)
                )
        self.assertEqual(outputs[0], want)
        self.assertEqual(outputs[1], want)


if __name__ == "__main__":
    unittest.main()
