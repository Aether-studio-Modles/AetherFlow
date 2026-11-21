"""AF-34 产品层的判据：词表双射、模板拼接、batch 的 mask 与配置绑定。

这四个都是"错了不会报错、只会安静地错位"的东西：词表错一位，文本就乱；模板
顺序错，模型学到的是另一种对话格式；mask 错，梯度被 padding 稀释；配置没绑
定，同名参数能装进另一个形状的模型。所以每一条都用一个可执行的不变量来钉。
"""

import unittest

import numpy as np

from aetherflow.models import ByteTokenizer, ChatTemplate
from aetherflow.train import lm_batch


class TestByteTokenizer(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()

    def test_roundtrips_arbitrary_utf8(self):
        for text in ("hello", "你好，世界", "emoji 🙂🙃", "line1\nline2\tend", ""):
            self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_ids_follow_the_spec_convention(self):
        spec = self.tok.spec(d_model=8, steps=4, batch=1)
        self.assertEqual(spec.vocab_size, 258)
        self.assertEqual(spec.pad_id, self.tok.pad_id)
        self.assertEqual(spec.eos_id, self.tok.eos_id)
        self.assertEqual(spec.vocab_size, self.tok.vocab_size)

    def test_bytes_start_above_the_specials(self):
        ids = self.tok.encode("A")
        self.assertEqual(ids, [2 + ord("A")])
        self.assertNotIn(self.tok.pad_id, ids)
        self.assertNotIn(self.tok.eos_id, ids)

    def test_decode_skips_padding_and_stops_at_eos(self):
        body = self.tok.encode("ok")
        self.assertEqual(self.tok.decode([0, 0, *body, self.tok.eos_id, 2 + 33]), "ok")

    def test_decode_rejects_out_of_vocab(self):
        with self.assertRaises(ValueError):
            self.tok.decode([self.tok.vocab_size + 10])

    def test_decode_rejects_a_truncated_utf8_sequence(self):
        raw = "你".encode("utf-8")
        truncated = [2 + byte for byte in raw[:2]]
        with self.assertRaises(ValueError):
            self.tok.decode(truncated)

    def test_spec_cannot_override_the_vocabulary(self):
        for key in ("vocab_size", "pad_id", "eos_id"):
            with self.assertRaises(ValueError):
                self.tok.spec(**{key: 3})


class TestChatTemplate(unittest.TestCase):
    def test_turns_are_rendered_in_order(self):
        text = ChatTemplate().render(
            [("system", "be brief"), ("user", "hi")], add_generation_prompt=False
        )
        self.assertEqual(text, "<|system|>\nbe brief\n<|user|>\nhi\n")

    def test_generation_prompt_opens_the_reply(self):
        text = ChatTemplate().render([("user", "hi")])
        self.assertTrue(text.endswith("<|assistant|>\n"))

    def test_unknown_role_is_rejected(self):
        with self.assertRaises(ValueError):
            ChatTemplate().render([("tool", "x")])

    def test_empty_content_is_rejected(self):
        with self.assertRaises(ValueError):
            ChatTemplate().render([("user", "")])

    def test_empty_message_list_is_rejected(self):
        with self.assertRaises(ValueError):
            ChatTemplate().render([])

    def test_rendered_text_survives_the_tokenizer(self):
        tok = ByteTokenizer()
        text = ChatTemplate().render(
            [("system", "你是助手"), ("user", "你好")]
        )
        self.assertEqual(tok.decode(tok.encode(text)), text)


class TestLanguageModelBatch(unittest.TestCase):
    def test_left_padding_keeps_the_real_tokens_at_the_end(self):
        out = lm_batch([[5, 6, 7], [8]], 16, steps=4)
        np.testing.assert_array_equal(
            out["ids"], np.array([[0, 5, 6, 7], [0, 0, 0, 8]], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            out["mask"], np.array([[0, 1, 1, 0], [0, 0, 0, 0]], dtype=np.float32)
        )

    def test_targets_are_the_next_token_as_one_hot(self):
        out = lm_batch([[5, 6, 7]], 16, steps=4)
        np.testing.assert_array_equal(
            np.argmax(out["targets"][0], axis=-1),
            np.array([0, 6, 7, 0]),
        )
        self.assertEqual(out["targets"].shape, (1, 4, 16))
        self.assertEqual(float(out["targets"][0].sum()), 2.0)

    def test_mask_covers_exactly_where_targets_are_set(self):
        out = lm_batch([[3, 4, 5], [6, 7]], 8, steps=5)
        has_target = out["targets"].sum(axis=-1) > 0
        np.testing.assert_array_equal(has_target, out["mask"] > 0)

    def test_reset_marks_the_first_real_token_of_each_row(self):
        """reset 只标"真实区间的起点"，pad 位不标：这是左填充隔离的入口。"""
        out = lm_batch([[5, 6, 7], [8]], 16, steps=4)
        np.testing.assert_array_equal(
            out["reset"],
            np.array([[0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32),
        )
        # 没有 padding 的一行也在 0 处标 reset：图从零状态开始，语义上确实是新序列
        out = lm_batch([[3, 4]], 8)
        np.testing.assert_array_equal(out["reset"], np.array([[1, 0]], np.float32))

    def test_too_long_sequence_is_rejected(self):
        with self.assertRaises(ValueError):
            lm_batch([[1, 2, 3]], 8, steps=2)

    def test_pad_inside_a_sequence_is_rejected(self):
        with self.assertRaises(ValueError):
            lm_batch([[1, 0, 2]], 8, steps=4)

    def test_out_of_vocab_token_is_rejected(self):
        with self.assertRaises(ValueError):
            lm_batch([[9]], 8, steps=2)

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            lm_batch([], 8)
        with self.assertRaises(ValueError):
            lm_batch([[]], 8)


class TestSpecBinding(unittest.TestCase):
    def _spec(self, **kw):
        return ByteTokenizer().spec(d_model=8, steps=4, batch=1, **kw)

    def test_meta_roundtrips(self):
        spec = self._spec()
        spec.check_meta(spec.meta())  # 不抛就是通过

    def test_changed_shape_is_rejected(self):
        spec = self._spec()
        meta = dict(spec.meta(), lm_d_model=16)
        with self.assertRaises(ValueError) as caught:
            spec.check_meta(meta)
        self.assertIn("d_model", str(caught.exception))

    def test_bool_fields_are_not_confused_with_integers(self):
        spec = self._spec(tie_weights=False)
        meta = dict(spec.meta(), lm_tie_weights=0)
        spec.check_meta(meta)  # 0 与 False 等价，应当通过
        with self.assertRaises(ValueError):
            spec.check_meta(dict(spec.meta(), lm_tie_weights=1))

    def test_unbound_archive_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self._spec().check_meta({})
        self.assertIn("没有绑定", str(caught.exception))

    def test_meta_is_json_friendly_scalars(self):
        for value in self._spec().meta().values():
            self.assertIsInstance(value, (int, float, bool))


if __name__ == "__main__":
    unittest.main()
