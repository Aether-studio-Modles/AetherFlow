"""Unigram 词表（tokenizer.json）的切分、回退与 precompiled 字符映射。"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import unittest

from aetherflow.models import UnigramTokenizer, UnigramTokenizerError

# 一张够用的小词表：分数是 log 概率，负数，Viterbi 取总和最大。
SCORES = {
    "<pad>": 0.0,
    "</s>": 0.0,
    "<unk>": 0.0,
    "\u2581": -2.0,
    "a": -0.5,
    "b": -0.5,
    "ab": -1.0,
    "x": -0.4,
    "y": -0.4,
    "xy": -3.0,
    "hello": -3.0,
    "\u2581hello": -1.5,
    "\u2581world": -1.5,
    "\u4f60": -2.0,  # 你
    "X": -2.0,
    "\u597d": -2.0,  # 好
    "\u4f60\u597d": -3.0,  # 你好
}


def _build(extra=None, **kwargs) -> UnigramTokenizer:
    scores = dict(SCORES)
    scores.update(extra or {})
    vocab = {piece: index for index, piece in enumerate(scores)}
    kwargs.setdefault("unk_id", vocab["<unk>"])
    return UnigramTokenizer(vocab=vocab, scores=scores, **kwargs)


def _with_added(extra, name) -> UnigramTokenizer:
    base = _build(extra)
    return UnigramTokenizer(
        vocab=base.vocab,
        scores=base.scores,
        unk_id=base.unk_id,
        added={name: base.vocab[name]},
    )


def _toy_charsmap() -> str:
    """手工拼一张 darts-clone 双数组：A -> 你、B -> X。

    units[0] 是根，offset=128；键 A(65) / B(66) 落在 193 / 194，偏移分别指向
    192 / 198 两个叶子单元。值的语义是"替换串里的字节偏移"。
    """
    units = [0] * 256
    units[0] = 128 << 10  # 根：base = 128
    units[128 ^ ord("A")] = (1 << 10) | (1 << 8) | ord("A")  # 叶子在 193 ^ 1
    units[128 ^ ord("B")] = (4 << 10) | (1 << 8) | ord("B")  # 叶子在 194 ^ 4
    units[193 ^ 1] = 0
    units[194 ^ 4] = 6
    body = b"".join(unit.to_bytes(4, "little") for unit in units)
    normalized = b"\xe4\xbd\xa0\x00\x00\x00X\x00"  # 0:"你" 6:"X"
    blob = len(body).to_bytes(4, "little") + body + normalized
    return base64.b64encode(blob).decode("ascii")


class TestViterbi(unittest.TestCase):
    def test_two_short_pieces_can_beat_one_long(self):
        # "xy" 单片段 -3.0，x+y 合计 -0.8：贪心最长匹配会选错，Viterbi 不会。
        tokenizer = _build()
        self.assertEqual(
            tokenizer.encode("xy"),
            [tokenizer.vocab["x"], tokenizer.vocab["y"]],
        )

    def test_one_long_piece_can_beat_two_short(self):
        # "ab" -1.0，a+b 合计 -1.0 打平；先写入的更长片段胜出（与 HF 一致）。
        tokenizer = _build()
        self.assertEqual(tokenizer.encode("ab"), [tokenizer.vocab["ab"]])

    def test_longer_piece_wins_when_cheaper(self):
        tokenizer = _build({"ab": -0.1})
        self.assertEqual(tokenizer.encode("ab"), [tokenizer.vocab["ab"]])

    def test_multi_char_piece_beats_two_chars(self):
        tokenizer = _build({"\u4f60\u597d": -3.5})
        self.assertEqual(tokenizer.encode("\u4f60\u597d"), [tokenizer.vocab["\u4f60\u597d"]])

    def test_empty_text_has_no_ids(self):
        self.assertEqual(_build().encode(""), [])


class TestUnknown(unittest.TestCase):
    def test_unknown_char_falls_back_to_unk(self):
        tokenizer = _build()
        self.assertEqual(tokenizer.encode("z"), [tokenizer.unk_id])

    def test_fuse_unk_merges_consecutive_unknowns(self):
        tokenizer = _build()
        self.assertEqual(tokenizer.encode("zz"), [tokenizer.unk_id])

    def test_fuse_unk_can_be_disabled(self):
        tokenizer = _build(fuse_unk=False)
        self.assertEqual(tokenizer.encode("zz"), [tokenizer.unk_id, tokenizer.unk_id])

    def test_unk_splits_known_pieces(self):
        tokenizer = _build()
        self.assertEqual(
            tokenizer.encode("aza"),
            [tokenizer.vocab["a"], tokenizer.unk_id, tokenizer.vocab["a"]],
        )

    def test_byte_fallback_expands_unknown_bytes(self):
        tokenizer = _build(
            {"<0x7A>": -5.0},
            byte_fallback=True,
        )
        self.assertEqual(tokenizer.encode("z"), [tokenizer.vocab["<0x7A>"]])

    def test_byte_fallback_needs_all_byte_tokens(self):
        tokenizer = _build(byte_fallback=True)
        with self.assertRaisesRegex(UnigramTokenizerError, "byte_fallback"):
            tokenizer.encode("z")

    def test_byte_fallback_expands_one_char_into_three_bytes(self):
        # 词表里没有"你"，Viterbi 无真词可选，只能按字节拆成三个节点。
        scores = {
            **SCORES,
            "<0xE4>": -5.0,
            "<0xBD>": -5.0,
            "<0xA0>": -5.0,
        }
        for piece in ("\u4f60", "\u4f60\u597d"):
            scores.pop(piece)
        vocab = {piece: index for index, piece in enumerate(scores)}
        tokenizer = UnigramTokenizer(
            vocab=vocab, scores=scores, unk_id=vocab["<unk>"], byte_fallback=True
        )
        self.assertEqual(
            tokenizer.encode("\u4f60"),
            [vocab["<0xE4>"], vocab["<0xBD>"], vocab["<0xA0>"]],
        )

    def test_byte_fallback_keeps_real_piece_when_cheaper(self):
        tokenizer = _build(
            {"<0xE4>": -5.0, "<0xBD>": -5.0, "<0xA0>": -5.0},
            byte_fallback=True,
        )
        tokenizer = UnigramTokenizer(
            vocab=tokenizer.vocab,
            scores={**tokenizer.scores, "\u4f60": -9.0},
            unk_id=tokenizer.unk_id,
            byte_fallback=True,
        )
        self.assertEqual(tokenizer.encode("\u4f60"), [tokenizer.vocab["\u4f60"]])


class TestMetaspace(unittest.TestCase):
    METASPACE = {
        "type": "Metaspace",
        "replacement": "\u2581",
        "add_prefix_space": True,
        "prepend_scheme": "always",
        "split": True,
    }

    def test_prefix_and_split(self):
        tokenizer = _build(pre_tokenizer=self.METASPACE)
        self.assertEqual(
            tokenizer.encode("hello world"),
            [tokenizer.vocab["\u2581hello"], tokenizer.vocab["\u2581world"]],
        )

    def test_never_prepend(self):
        tokenizer = _build(
            pre_tokenizer={
                "type": "Metaspace",
                "replacement": "\u2581",
                "prepend_scheme": "never",
                "split": True,
            }
        )
        self.assertEqual(tokenizer.encode("hello"), [tokenizer.vocab["hello"]])

    def test_no_split_keeps_one_chunk(self):
        tokenizer = _build(
            pre_tokenizer={
                "type": "Metaspace",
                "replacement": "\u2581",
                "prepend_scheme": "always",
                "split": False,
            }
        )
        self.assertEqual(
            tokenizer.encode("hello world"),
            [tokenizer.vocab["\u2581hello"], tokenizer.vocab["\u2581world"]],
        )

    def test_multiple_spaces_become_markers(self):
        tokenizer = _build(pre_tokenizer=self.METASPACE)
        ids = tokenizer.encode("hello  world")
        self.assertEqual(ids[1], tokenizer.vocab["\u2581"])


class TestAddedTokens(unittest.TestCase):
    def test_special_token_is_matched_whole(self):
        tokenizer = _with_added({"<extra_id_0>": 0.0}, "<extra_id_0>")
        self.assertEqual(
            tokenizer.encode("a<extra_id_0>b"),
            [tokenizer.vocab["a"], tokenizer.vocab["<extra_id_0>"], tokenizer.vocab["b"]],
        )

    def test_added_token_must_exist_in_vocab(self):
        tokenizer = _build()
        with self.assertRaisesRegex(UnigramTokenizerError, "added token"):
            UnigramTokenizer(
                vocab=tokenizer.vocab,
                scores=tokenizer.scores,
                unk_id=tokenizer.unk_id,
                added={"<nope>": 999},
            )


class TestPrecompiled(unittest.TestCase):
    def test_charsmap_rewrites_before_viterbi(self):
        tokenizer = _build(
            normalizer={"type": "Precompiled", "precompiled_charsmap": _toy_charsmap()}
        )
        self.assertEqual(
            tokenizer.encode("AB"),
            [tokenizer.vocab["\u4f60"], tokenizer.vocab["X"]],
        )

    def test_charsmap_leaves_unmatched_text(self):
        tokenizer = _build(
            normalizer={"type": "Precompiled", "precompiled_charsmap": _toy_charsmap()}
        )
        self.assertEqual(tokenizer.encode("\u4f60"), [tokenizer.vocab["\u4f60"]])

    def test_charsmap_size_is_validated(self):
        body = b"\x00" * 1024
        blob = (1024).to_bytes(4, "little") + body[:-8]
        payload = base64.b64encode(blob).decode("ascii")
        with self.assertRaisesRegex(UnigramTokenizerError, "超出载荷"):
            _build(normalizer={"type": "Precompiled", "precompiled_charsmap": payload})

    def test_other_normalizers_are_rejected(self):
        with self.assertRaisesRegex(UnigramTokenizerError, "BertNormalizer"):
            _build(normalizer={"type": "BertNormalizer"})


class TestDecode(unittest.TestCase):
    METASPACE = {
        "type": "Metaspace",
        "replacement": "\u2581",
        "add_prefix_space": True,
        "prepend_scheme": "always",
        "split": True,
    }

    def test_round_trip_with_metaspace(self):
        tokenizer = _build(pre_tokenizer=self.METASPACE, decoder=self.METASPACE)
        ids = tokenizer.encode("hello world")
        self.assertEqual(tokenizer.decode(ids), "hello world")

    def test_decode_stops_at_eos_and_skips_pad(self):
        tokenizer = _build()
        ids = [
            tokenizer.vocab["a"],
            tokenizer.vocab["<pad>"],
            tokenizer.vocab["</s>"],
            tokenizer.vocab["b"],
        ]
        self.assertEqual(tokenizer.decode(ids), "a")

    def test_decode_byte_fallback(self):
        tokenizer = _build(
            {"<0xE4>": -5.0, "<0xBD>": -5.0, "<0xA0>": -5.0},
            decoder={"type": "ByteFallback"},
        )
        ids = [
            tokenizer.vocab["<0xE4>"],
            tokenizer.vocab["<0xBD>"],
            tokenizer.vocab["<0xA0>"],
        ]
        self.assertEqual(tokenizer.decode(ids), "\u4f60")

    def test_decode_unknown_id_raises(self):
        tokenizer = _build()
        with self.assertRaisesRegex(UnigramTokenizerError, "不在词表内"):
            tokenizer.decode([tokenizer.vocab_size + 5])

    def test_skip_special_tokens(self):
        base = _build({"<extra_id_0>": 0.0})
        tokenizer = UnigramTokenizer(
            vocab=base.vocab,
            scores=base.scores,
            unk_id=base.unk_id,
            added={"<extra_id_0>": base.vocab["<extra_id_0>"]},
            special_tokens=["<extra_id_0>"],
        )
        ids = [tokenizer.vocab["a"], tokenizer.vocab["<extra_id_0>"]]
        self.assertEqual(tokenizer.decode(ids), "a<extra_id_0>")
        self.assertEqual(tokenizer.decode(ids, skip_special_tokens=True), "a")


class TestConstruction(unittest.TestCase):
    def test_spec_uses_vocab_ids(self):
        tokenizer = _build()
        spec = tokenizer.spec()
        self.assertEqual(spec.vocab_size, tokenizer.vocab_size)
        self.assertEqual(spec.eos_id, tokenizer.vocab["</s>"])
        self.assertEqual(spec.pad_id, tokenizer.vocab["<pad>"])

    def test_missing_score_is_rejected(self):
        with self.assertRaisesRegex(UnigramTokenizerError, "没有分数"):
            UnigramTokenizer(vocab={"a": 0}, scores={}, unk_id=0)

    def test_duplicate_piece_is_rejected(self):
        with self.assertRaisesRegex(UnigramTokenizerError, "出现两次"):
            UnigramTokenizer.from_dict(
                {"model": {"type": "Unigram", "unk_id": 0, "vocab": [["a", -1.0], ["a", -2.0]]}}
            )

    def test_unk_out_of_range(self):
        with self.assertRaisesRegex(UnigramTokenizerError, "超出词表"):
            UnigramTokenizer(vocab={"a": 0}, scores={"a": -1.0}, unk_id=5)

    def test_bpe_and_wordpiece_point_to_their_own_class(self):
        with self.assertRaisesRegex(UnigramTokenizerError, "BpeTokenizer"):
            UnigramTokenizer.from_dict({"model": {"type": "BPE"}})
        with self.assertRaisesRegex(UnigramTokenizerError, "WordPieceTokenizer"):
            UnigramTokenizer.from_dict({"model": {"type": "WordPiece"}})

    def test_from_dict_requires_unk_id(self):
        with self.assertRaisesRegex(UnigramTokenizerError, "unk_id"):
            UnigramTokenizer.from_dict(
                {"model": {"type": "Unigram", "vocab": [["<unk>", 0.0]]}}
            )

    def test_byte_level_pre_tokenizer_is_rejected(self):
        with self.assertRaisesRegex(UnigramTokenizerError, "Metaspace"):
            _build(pre_tokenizer={"type": "ByteLevel"})

    def test_from_file_rejects_sentencepiece_binary(self):
        path = os.path.join(tempfile.mkdtemp(), "spiece.model")
        with open(path, "wb") as handle:
            handle.write(b"\x00\x01")
        try:
            with self.assertRaisesRegex(UnigramTokenizerError, "tokenizer.json"):
                UnigramTokenizer.from_file(path)
        finally:
            os.remove(path)
            os.rmdir(os.path.dirname(path))

    def test_from_dict_reads_list_vocab(self):
        data = {
            "model": {
                "type": "Unigram",
                "unk_id": 1,
                "byte_fallback": False,
                "vocab": [["<pad>", 0.0], ["<unk>", 0.0], ["\u2581hi", -1.0]],
            },
            "pre_tokenizer": {
                "type": "Metaspace",
                "replacement": "\u2581",
                "prepend_scheme": "always",
                "split": True,
            },
            "decoder": {"type": "Metaspace", "replacement": "\u2581"},
        }
        tokenizer = UnigramTokenizer.from_dict(data)
        ids = tokenizer.encode("hi")
        self.assertEqual(ids, [2])
        self.assertEqual(tokenizer.decode(ids), "hi")

    def test_from_file_reads_json(self):
        data = {
            "model": {"type": "Unigram", "unk_id": 0, "vocab": [["<unk>", 0.0], ["ab", -1.0]]},
        }
        root = tempfile.mkdtemp()
        path = os.path.join(root, "tokenizer.json")
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
            tokenizer = UnigramTokenizer.from_file(path)
            self.assertEqual(tokenizer.encode("ab"), [1])
        finally:
            os.remove(path)
            os.rmdir(root)


if __name__ == "__main__":
    unittest.main()
