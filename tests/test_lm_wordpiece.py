"""WordPiece（BERT 谱系）词表的语义与拒绝面。"""

import json
import os
import tempfile
import unittest

from aetherflow.models import WordPieceTokenizer, WordPieceTokenizerError


VOCAB = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[MASK]": 4,
    "un": 5,
    "##aff": 6,
    "##able": 7,
    "hello": 8,
    ",": 9,
    "world": 10,
    "!": 11,
    "你": 12,
    "好": 13,
    "don": 14,
    "'": 15,
    "t": 16,
    "cafe": 17,
    "play": 18,
    "##ing": 19,
    "steam": 20,
    "##boat": 21,
}


class TestWordPieceEncode(unittest.TestCase):
    def setUp(self):
        self.tokenizer = WordPieceTokenizer(vocab=VOCAB)

    def test_basic_tokenizer_splits_punctuation_and_lowercases(self):
        self.assertEqual(self.tokenizer.encode("Hello, world!"), [8, 9, 10, 11])
        self.assertEqual(self.tokenizer.encode("HELLO"), [8])

    def test_chinese_is_split_per_character(self):
        self.assertEqual(self.tokenizer.encode("你好"), [12, 13])
        self.assertEqual(self.tokenizer.encode("你好 world"), [12, 13, 10])

    def test_apostrophe_is_its_own_token(self):
        self.assertEqual(self.tokenizer.encode("don't"), [14, 15, 16])

    def test_greedy_longest_match_uses_continuation_prefix(self):
        self.assertEqual(self.tokenizer.encode("unaffable"), [5, 6, 7])
        self.assertEqual(self.tokenizer.encode("steamboat"), [20, 21])

    def test_longest_piece_wins_over_shorter_segmentation(self):
        longest = WordPieceTokenizer(vocab={
            "[UNK]": 0, "[SEP]": 1, "un": 2, "unaff": 3, "##aff": 4, "##able": 5,
        })
        self.assertEqual(longest.encode("unaffable"), [3, 5])

    def test_accents_are_stripped_by_default(self):
        self.assertEqual(self.tokenizer.encode("café"), [17])
        keep = WordPieceTokenizer(vocab=VOCAB, strip_accents=False)
        self.assertEqual(keep.encode("café"), [1])

    def test_unknown_pieces_fall_back_to_unk(self):
        self.assertEqual(self.tokenizer.encode("zzz"), [1])
        self.assertEqual(self.tokenizer.encode("unzzz"), [1])

    def test_max_input_chars_per_word_rejects_long_words(self):
        short = WordPieceTokenizer(vocab=VOCAB, max_input_chars_per_word=3)
        self.assertEqual(short.encode("unaffable"), [1])
        self.assertEqual(short.encode("un"), [5])

    def test_whitespace_is_a_separator_and_control_chars_are_dropped(self):
        self.assertEqual(self.tokenizer.encode("hello\tworld"), [8, 10])
        self.assertEqual(self.tokenizer.encode("hello\r\nworld"), [8, 10])
        # \x00 是控制字符，被删掉后两个词拼在一起，词表里没有 -> [UNK]
        self.assertEqual(self.tokenizer.encode("hello\x00world"), [1])

    def test_chinese_handling_can_be_turned_off(self):
        raw = WordPieceTokenizer(vocab=VOCAB, handle_chinese_chars=False)
        self.assertEqual(raw.encode("你好"), [1])

    def test_case_sensitive_vocab(self):
        cased = WordPieceTokenizer(
            vocab=VOCAB, do_lower_case=False, strip_accents=False
        )
        self.assertEqual(cased.encode("hello"), [8])
        self.assertEqual(cased.encode("Hello"), [1])


class TestWordPieceDecode(unittest.TestCase):
    def setUp(self):
        self.tokenizer = WordPieceTokenizer(vocab=VOCAB)

    def test_continuation_pieces_are_joined(self):
        self.assertEqual(self.tokenizer.decode([5, 6, 7]), "unaffable")
        self.assertEqual(self.tokenizer.decode([8, 9, 10, 11]), "hello , world !")

    def test_stops_at_eos_and_skips_pad(self):
        self.assertEqual(self.tokenizer.decode([2, 8, 3, 0, 0]), "[CLS] hello")

    def test_skip_special_tokens(self):
        self.assertEqual(
            self.tokenizer.decode([2, 8, 3], skip_special_tokens=True), "hello"
        )

    def test_ids_outside_vocab_are_rejected(self):
        with self.assertRaisesRegex(WordPieceTokenizerError, "不在词表内"):
            self.tokenizer.decode([8, 99])

    def test_eos_and_pad_can_share_an_id(self):
        shared = WordPieceTokenizer(vocab=VOCAB, pad_id=3, eos_id=3)
        self.assertEqual(shared.decode([8, 3, 0]), "hello")


class TestWordPieceLoading(unittest.TestCase):
    def _write(self, text, suffix=".txt"):
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=suffix, delete=False, newline=""
        )
        try:
            handle.write(text)
        finally:
            handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_vocab_file_line_number_is_id(self):
        path = self._write("[PAD]\n[UNK]\n[SEP]\nhello\n")
        tokenizer = WordPieceTokenizer.from_file(path)
        self.assertEqual(tokenizer.vocab, {"[PAD]": 0, "[UNK]": 1, "[SEP]": 2, "hello": 3})
        self.assertEqual(tokenizer.encode("hello"), [3])
        self.assertEqual(tokenizer.eos_id, 2)
        self.assertEqual(tokenizer.pad_id, 0)

    def test_blank_line_in_the_middle_is_rejected(self):
        path = self._write("[PAD]\n[UNK]\n\nhello\n")
        with self.assertRaisesRegex(WordPieceTokenizerError, "行是空的"):
            WordPieceTokenizer.from_file(path)

    def test_duplicate_token_is_rejected(self):
        path = self._write("[PAD]\n[UNK]\nhello\nhello\n")
        with self.assertRaisesRegex(WordPieceTokenizerError, "出现两次"):
            WordPieceTokenizer.from_file(path)

    def test_tokenizer_json_wordpiece(self):
        data = {
            "model": {
                "type": "WordPiece",
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
                "vocab": VOCAB,
            },
            "normalizer": {
                "type": "BertNormalizer",
                "clean_text": True,
                "handle_chinese_chars": True,
                "lowercase": True,
            },
            "pre_tokenizer": {"type": "BertPreTokenizer"},
            "added_tokens": [{"id": 0, "content": "[PAD]", "special": True}],
        }
        path = self._write(json.dumps(data), suffix=".json")
        tokenizer = WordPieceTokenizer.from_file(path)
        self.assertEqual(tokenizer.encode("Hello, world!"), [8, 9, 10, 11])
        self.assertEqual(tokenizer.spec().vocab_size, len(VOCAB))

    def test_other_model_types_are_rejected(self):
        with self.assertRaisesRegex(WordPieceTokenizerError, "只吃 WordPiece"):
            WordPieceTokenizer.from_dict({"model": {"type": "BPE", "vocab": VOCAB}})

    def test_other_normalizers_are_rejected(self):
        with self.assertRaisesRegex(WordPieceTokenizerError, "BertNormalizer"):
            WordPieceTokenizer.from_dict({
                "model": {"type": "WordPiece", "vocab": VOCAB},
                "normalizer": {"type": "NFC"},
            })

    def test_other_pre_tokenizers_are_rejected(self):
        with self.assertRaisesRegex(WordPieceTokenizerError, "BertPreTokenizer"):
            WordPieceTokenizer.from_dict({
                "model": {"type": "WordPiece", "vocab": VOCAB},
                "pre_tokenizer": {"type": "Metaspace"},
            })

    def test_added_token_must_exist_in_vocab(self):
        with self.assertRaisesRegex(WordPieceTokenizerError, "不在 model.vocab"):
            WordPieceTokenizer.from_dict({
                "model": {"type": "WordPiece", "vocab": VOCAB},
                "added_tokens": [{"id": 99, "content": "[NEW]"}],
            })


class TestWordPieceValidation(unittest.TestCase):
    def test_vocab_and_unk_are_checked(self):
        with self.assertRaisesRegex(WordPieceTokenizerError, "不能为空"):
            WordPieceTokenizer(vocab={})
        with self.assertRaisesRegex(WordPieceTokenizerError, "unk_token"):
            WordPieceTokenizer(vocab={"a": 0})

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(WordPieceTokenizerError, "id 重复"):
            WordPieceTokenizer(vocab={"[UNK]": 0, "a": 0})

    def test_prefix_and_length_are_checked(self):
        with self.assertRaisesRegex(WordPieceTokenizerError, "continuing_subword_prefix"):
            WordPieceTokenizer(vocab=VOCAB, continuing_subword_prefix="")
        with self.assertRaisesRegex(WordPieceTokenizerError, "max_input_chars_per_word"):
            WordPieceTokenizer(vocab=VOCAB, max_input_chars_per_word=0)

    def test_spec_requires_eos(self):
        tokenizer = WordPieceTokenizer(vocab={"[UNK]": 0, "a": 1})
        self.assertIsNone(tokenizer.eos_id)
        with self.assertRaisesRegex(WordPieceTokenizerError, "\\[SEP\\]"):
            tokenizer.spec()

    def test_spec_reflects_vocab(self):
        spec = WordPieceTokenizer(vocab=VOCAB).spec()
        self.assertEqual(spec.vocab_size, len(VOCAB))
        self.assertEqual(spec.pad_id, 0)
        self.assertEqual(spec.eos_id, 3)


if __name__ == "__main__":
    unittest.main()
