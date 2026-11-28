"""ByteLevel BPE 的判据：字节双射、合并优先级、切分边界、特殊 token 与拒绝面。

这个模块的每一处"差不多"都会安静地错：切分边界偏一位、合并顺序换一下、
特殊 token 被拆成字节，都不会抛异常，只会在下游表现为"模型有点笨"。所以这里
逐条钉住可执行的不变量，而不是"encode 出来看起来像"。

基础词表刻意让 ``id == 字节值``（0..255），这样断言里能直接写 ASCII 的 id；
``<|endoftext|>`` 占 256，其余合并 token 从 257 起。
"""

import json
import os
import tempfile
import unittest

from aetherflow.models import BpeTokenizer, BpeTokenizerError
from aetherflow.models.lm_bpe import _BYTE_ENCODER

EOS_ID = 256


def _char(byte: int) -> str:
    """某个字节在 GPT-2 字节映射里的可见字符。"""
    return _BYTE_ENCODER[byte]


def _data(merges=(), merged=(), added_tokens=None, vocab_extra=None, **overrides):
    vocab = {ch: byte for byte, ch in _BYTE_ENCODER.items()}
    for offset, token in enumerate(merged):
        vocab[token] = 257 + offset
    vocab.update(vocab_extra or {})
    data = {
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": [list(pair) for pair in merges],
        },
        "added_tokens": [
            {"content": "<|endoftext|>", "id": EOS_ID, "special": True},
            *(added_tokens or []),
        ],
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "decoder": {"type": "ByteLevel"},
    }
    data.update(overrides)
    return data


def _tokenizer(**kwargs) -> BpeTokenizer:
    return BpeTokenizer.from_dict(_data(**kwargs))


class TestRoundTrip(unittest.TestCase):
    def test_byte_mapping_roundtrips_arbitrary_utf8(self):
        tok = _tokenizer()
        for text in ("hello", "你好，世界", "emoji 🙂", "a\tb\nc", ""):
            ids = tok.encode(text)
            self.assertEqual(tok.decode(ids), text)

    def test_chinese_is_three_tokens_per_character(self):
        tok = _tokenizer()
        ids = tok.encode("你")
        self.assertEqual(ids, list("你".encode("utf-8")))
        self.assertEqual(tok.decode(ids), "你")

    def test_multibyte_ids_are_merged_before_utf8_decode(self):
        tok = _tokenizer()
        # 逐字节解会得到三个替换字符；实现必须先把连续字节拼起来。
        self.assertEqual(tok.decode([0xE4, 0xBD, 0xA0]), "你")

    def test_out_of_vocab_id_is_rejected(self):
        tok = _tokenizer()
        with self.assertRaises(BpeTokenizerError):
            tok.decode([tok.vocab_size + 10])

    def test_truncated_utf8_is_rejected(self):
        tok = _tokenizer()
        with self.assertRaises(BpeTokenizerError):
            tok.decode(list("你".encode("utf-8"))[:2])

    def test_non_bytelevel_token_is_rejected(self):
        tok = _tokenizer(vocab_extra={"Ω": 300})
        with self.assertRaises(BpeTokenizerError):
            tok.decode([300])


class TestMerges(unittest.TestCase):
    def test_single_pair_merge(self):
        tok = _tokenizer(merges=[("a", "b")], merged=("ab",))
        self.assertEqual(tok.encode("ab"), [257])
        self.assertEqual(tok.decode([257]), "ab")

    def test_lowest_rank_wins_not_left_to_right(self):
        tok = _tokenizer(merges=[("b", "c"), ("a", "b")], merged=("bc",))
        self.assertEqual(tok.encode("abc"), [97, 257])

    def test_rank_order_changes_the_result(self):
        tok = _tokenizer(merges=[("a", "b"), ("b", "c")], merged=("ab",))
        self.assertEqual(tok.encode("abc"), [257, 99])

    def test_merges_accept_the_string_form(self):
        data = _data()
        data["model"]["merges"] = ["a b"]
        data["model"]["vocab"]["ab"] = 257
        tok = BpeTokenizer.from_dict(data)
        self.assertEqual(tok.encode("ab"), [257])

    def test_inconsistent_merges_are_rejected_at_encode(self):
        tok = _tokenizer(merges=[("a", "b")])
        with self.assertRaises(BpeTokenizerError):
            tok.encode("ab")


class TestPreTokenization(unittest.TestCase):
    def test_leading_space_binds_to_the_word(self):
        tok = _tokenizer(merges=[(_char(32), "a")], merged=(_char(32) + "a",))
        self.assertEqual(tok.encode("a a"), [97, 257])

    def test_contraction_suffix_is_its_own_piece(self):
        tok = _tokenizer(merges=[("i", "t")], merged=("it",))
        # "it's" → ["it", "'s"]；若整段处理会先合并出 "it"，后两个字节仍分开。
        self.assertEqual(tok.encode("it's"), [257, 39, 115])

    def test_digits_and_punctuation_do_not_join_letters(self):
        tok = _tokenizer(merges=[("a", "1")], merged=("a1",))
        self.assertEqual(tok.encode("a1"), [97, 49])

    def test_whitespace_run_leaves_the_last_space_to_the_next_piece(self):
        tok = _tokenizer(merges=[(_char(32), _char(32))], merged=(_char(32) * 2,))
        self.assertEqual(tok.encode("  hello"), [32, 32, 104, 101, 108, 108, 111])

    def test_trailing_whitespace_stays_one_piece(self):
        tok = _tokenizer(merges=[(_char(32), _char(32))], merged=(_char(32) * 2,))
        self.assertEqual(tok.encode("hello  "), [104, 101, 108, 108, 111, 257])

    def test_use_regex_false_keeps_the_whole_text(self):
        pre = {"type": "ByteLevel", "use_regex": False, "add_prefix_space": False}
        tok = _tokenizer(
            merges=[("a", _char(32))], merged=("a" + _char(32),), pre_tokenizer=pre
        )
        self.assertEqual(tok.encode("a a"), [257, 97])

    def test_use_regex_true_splits_at_the_space(self):
        tok = _tokenizer(merges=[("a", _char(32))], merged=("a" + _char(32),))
        self.assertEqual(tok.encode("a a"), [97, 32, 97])

    def test_add_prefix_space(self):
        pre = {"type": "ByteLevel", "use_regex": True, "add_prefix_space": True}
        tok = _tokenizer(pre_tokenizer=pre)
        self.assertEqual(tok.encode("hi"), [32, 104, 105])

    def test_normalizer_is_applied_before_splitting(self):
        tok = _tokenizer(normalizer={"type": "NFC"})
        self.assertEqual(tok.encode("e\u0301"), tok.encode("\u00e9"))


class TestAddedTokens(unittest.TestCase):
    def setUp(self):
        self.added = [
            {"content": "<s>", "id": 257, "special": True},
            {"content": "<s><s>", "id": 258, "special": True},
        ]
        self.tok = _tokenizer(added_tokens=self.added)

    def test_longest_match_wins(self):
        self.assertEqual(self.tok.encode("<s><s>"), [258])

    def test_added_token_is_not_split_into_bytes(self):
        self.assertEqual(self.tok.encode("<s>"), [257])

    def test_decode_outputs_the_original_text(self):
        self.assertEqual(self.tok.decode([258]), "<s><s>")
        self.assertEqual(self.tok.decode([257, 257]), "<s><s>")

    def test_skip_special_tokens(self):
        self.assertEqual(self.tok.decode([257, 97, 258], skip_special_tokens=True), "a")
        self.assertEqual(self.tok.decode([257, 97, 258]), "<s>a<s><s>")

    def test_non_special_added_token_decodes_as_text(self):
        tok = _tokenizer(added_tokens=[{"content": "<|x|>", "id": 257}])
        self.assertEqual(tok.encode("<|x|>"), [257])
        self.assertEqual(tok.decode([257]), "<|x|>")

    def test_text_around_added_tokens_still_goes_through_bpe(self):
        self.assertEqual(self.tok.encode("a<s>b"), [97, 257, 98])


class TestVocabulary(unittest.TestCase):
    def test_vocab_size_covers_the_highest_added_id(self):
        tok = _tokenizer(added_tokens=[{"content": "<s>", "id": 300, "special": True}])
        self.assertEqual(tok.vocab_size, 301)

    def test_spec_follows_the_tokenizer(self):
        tok = _tokenizer()
        spec = tok.spec(d_model=8, steps=4, batch=1)
        self.assertEqual(spec.vocab_size, tok.vocab_size)
        self.assertEqual(spec.pad_id, tok.pad_id)
        self.assertEqual(spec.eos_id, tok.eos_id)

    def test_spec_cannot_override_the_vocabulary(self):
        tok = _tokenizer()
        for key in ("vocab_size", "pad_id", "eos_id"):
            with self.assertRaises(ValueError):
                tok.spec(**{key: 3})

    def test_pad_defaults_to_eos_and_eos_is_inferred(self):
        tok = _tokenizer()
        self.assertEqual(tok.eos_id, EOS_ID)
        self.assertEqual(tok.pad_id, EOS_ID)

    def test_explicit_eos_and_pad(self):
        tok = BpeTokenizer.from_dict(_data(), eos_token="<|endoftext|>", pad_token=_char(0))
        self.assertEqual(tok.eos_id, EOS_ID)
        self.assertEqual(tok.pad_id, 0)

    def test_unknown_eos_or_pad_is_rejected(self):
        with self.assertRaises(BpeTokenizerError):
            BpeTokenizer.from_dict(_data(), eos_token="</s>")
        with self.assertRaises(BpeTokenizerError):
            BpeTokenizer.from_dict(_data(), pad_token="</s>")

    def test_eos_stops_decoding_even_when_it_is_also_pad(self):
        tok = _tokenizer()
        self.assertEqual(tok.decode([EOS_ID, 97]), "")

    def test_pad_is_skipped_when_it_differs_from_eos(self):
        tok = BpeTokenizer.from_dict(_data(), pad_token=_char(0))
        self.assertEqual(tok.decode([0, 97, 0]), "a")


class TestRejections(unittest.TestCase):
    def _bad(self, **changes):
        data = _data()
        for key, value in changes.items():
            data[key] = value
        return data

    def test_non_bpe_model(self):
        data = _data()
        data["model"]["type"] = "WordPiece"
        with self.assertRaises(BpeTokenizerError):
            BpeTokenizer.from_dict(data)

    def test_sentencepiece_switches(self):
        for key in ("byte_fallback", "end_of_word_suffix", "continuing_subword_prefix"):
            data = _data()
            data["model"][key] = True if key == "byte_fallback" else "_"
            with self.assertRaises(BpeTokenizerError):
                BpeTokenizer.from_dict(data)

    def test_bpe_dropout(self):
        data = _data()
        data["model"]["dropout"] = 0.1
        with self.assertRaises(BpeTokenizerError):
            BpeTokenizer.from_dict(data)

    def test_merges_shape(self):
        data = _data()
        data["model"]["merges"] = "not-a-list"
        with self.assertRaises(BpeTokenizerError):
            BpeTokenizer.from_dict(data)
        data["model"]["merges"] = [["only-one"]]
        with self.assertRaises(BpeTokenizerError):
            BpeTokenizer.from_dict(data)

    def test_missing_byte_characters(self):
        vocab = {ch: byte for byte, ch in _BYTE_ENCODER.items() if byte != 0}
        with self.assertRaises(BpeTokenizerError):
            BpeTokenizer(vocab=vocab, ranks={})

    def test_empty_vocab(self):
        with self.assertRaises(BpeTokenizerError):
            BpeTokenizer(vocab={}, ranks={})

    def test_duplicate_ids(self):
        vocab = {ch: byte for byte, ch in _BYTE_ENCODER.items()}
        vocab["x"] = 0
        with self.assertRaises(BpeTokenizerError):
            BpeTokenizer(vocab=vocab, ranks={})

    def test_added_id_clashes_with_vocab(self):
        with self.assertRaises(BpeTokenizerError):
            _tokenizer(added_tokens=[{"content": "<s>", "id": 97, "special": True}])

    def test_two_added_tokens_share_one_id(self):
        with self.assertRaises(BpeTokenizerError):
            _tokenizer(
                added_tokens=[
                    {"content": "<s>", "id": 257, "special": True},
                    {"content": "<t>", "id": 257, "special": True},
                ]
            )

    def test_added_token_reuses_its_vocab_id(self):
        with self.assertRaises(BpeTokenizerError):
            _tokenizer(added_tokens=[{"content": "<s>", "id": 257}], vocab_extra={"<s>": 300})

    def test_pad_and_eos_must_be_in_range(self):
        vocab = {ch: byte for byte, ch in _BYTE_ENCODER.items()}
        with self.assertRaises(BpeTokenizerError):
            BpeTokenizer(vocab=vocab, ranks={}, pad_id=999)
        with self.assertRaises(BpeTokenizerError):
            BpeTokenizer(vocab=vocab, ranks={}, eos_id=999)

    def test_unsupported_normalizer(self):
        with self.assertRaises(BpeTokenizerError):
            _tokenizer(normalizer={"type": "Precompiled", "precompiled_charsmap": ""})

    def test_unsupported_pre_tokenizers(self):
        with self.assertRaises(BpeTokenizerError):
            _tokenizer(pre_tokenizer={"type": "Metaspace"})
        with self.assertRaises(BpeTokenizerError):
            _tokenizer(pre_tokenizer={"type": "Split", "pattern": {"Regex": "\\d"}})

    def test_sequence_with_conflicting_bytelevel_settings(self):
        pre = {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "ByteLevel", "use_regex": True},
                {"type": "ByteLevel", "use_regex": False},
            ],
        }
        with self.assertRaises(BpeTokenizerError):
            _tokenizer(pre_tokenizer=pre)

    def test_unsupported_decoder(self):
        with self.assertRaises(BpeTokenizerError):
            _tokenizer(decoder={"type": "WordPiece", "prefix": "##"})

    def test_added_token_with_strip_flags(self):
        with self.assertRaises(BpeTokenizerError):
            _tokenizer(added_tokens=[{"content": "<s>", "id": 257, "lstrip": True}])

    def test_bad_added_token_entries(self):
        with self.assertRaises(BpeTokenizerError):
            _tokenizer(added_tokens=[{"content": "", "id": 257}])
        with self.assertRaises(BpeTokenizerError):
            _tokenizer(added_tokens=[{"content": "<s>", "id": -1}])

    def test_top_level_must_be_an_object(self):
        with self.assertRaises(BpeTokenizerError):
            BpeTokenizer.from_dict([])


class TestFromFile(unittest.TestCase):
    def test_reads_a_file_and_reports_bad_json(self):
        with tempfile.TemporaryDirectory() as folder:
            good = os.path.join(folder, "tokenizer.json")
            with open(good, "w", encoding="utf-8") as fh:
                json.dump(_data(), fh)
            self.assertEqual(BpeTokenizer.from_file(good).eos_id, EOS_ID)

            bad = os.path.join(folder, "broken.json")
            with open(bad, "w", encoding="utf-8") as fh:
                fh.write("{not json")
            with self.assertRaises(BpeTokenizerError):
                BpeTokenizer.from_file(bad)


if __name__ == "__main__":
    unittest.main()
