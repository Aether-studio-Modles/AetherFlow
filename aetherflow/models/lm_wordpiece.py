"""WordPiece 词表（BERT 谱系）：``vocab.txt`` 或 ``tokenizer.json``。

为什么另一份实现：ByteLevel BPE（``lm_bpe``）先把整段文本映射成字节再合并，
WordPiece 则是"先做 BasicTokenizer 预切分，再对每个词做贪心最长前缀匹配"。
两者的切分边界与词表形态都不同，硬塞进一处只会让两边都变糊。

支持的范围（刻意窄，越界显式拒绝）：

- ``vocab.txt``：一行一个 token，行号即 id（BERT 发布物的形态）；
- ``tokenizer.json``：``model.type == "WordPiece"``，``normalizer`` 为 null 或
  ``BertNormalizer``，``pre_tokenizer`` 为 null 或 ``BertPreTokenizer``；
- BasicTokenizer 的四件事：清控制字符 / 中文按字加空格 / 小写 / 去重音，
  然后按空白与标点切分；
- WordPiece 的三件事：``##`` 续接前缀、超过 ``max_input_chars_per_word`` 的
  词直接给 ``[UNK]``、匹配不上也给 ``[UNK]``。

拒绝而不是近似：``Unigram``（T5 / Llama）、``SentencePiece`` 的 ``Metaspace``
与 ``byte_fallback``、自定义 ``Split`` 预分词——它们的切分边界与 BERT 不同，
近似实现产出的 id 每个都合法，但不是训练时那串。

特殊 token 的判定沿用 BERT 的形态：形如 ``[PAD]`` / ``[CLS]`` 的 token
（``[`` 开头 ``]`` 结尾）视为特殊 token，``decode`` 在 ``skip_special_tokens``
时跳过它们。
"""

from __future__ import annotations

import json
import os
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

from .lm import LMSpec

__all__ = ["WordPieceTokenizerError", "WordPieceTokenizer"]


class WordPieceTokenizerError(ValueError):
    """词表不是受支持的 WordPiece 形态，或用的 id 不在词表内。"""


def _is_whitespace(char: str) -> bool:
    if char in " \t\n\r\u2028\u2029":
        return True
    return unicodedata.category(char) == "Zs"


def _is_control(char: str) -> bool:
    if char in "\t\n\r":
        return False
    return unicodedata.category(char).startswith("C")


def _is_punctuation(char: str) -> bool:
    cp = ord(char)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(char).startswith("P")


def _is_cjk(char: str) -> bool:
    cp = ord(char)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2B820 <= cp <= 0x2CEAF
        or 0xF900 <= cp <= 0xFAFF
        or 0x2F800 <= cp <= 0x2FA1F
    )


def _clean_text(text: str) -> str:
    out: List[str] = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xFFFD or _is_control(char):
            continue
        out.append(" " if _is_whitespace(char) else char)
    return "".join(out)


def _tokenize_chinese_chars(text: str) -> str:
    out: List[str] = []
    for char in text:
        if _is_cjk(char):
            out.append(" ")
            out.append(char)
            out.append(" ")
        else:
            out.append(char)
    return "".join(out)


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def _split_on_punctuation(token: str) -> List[str]:
    chars = list(token)
    out: List[str] = []
    start = 0
    for index, char in enumerate(chars):
        if not _is_punctuation(char):
            continue
        if start < index:
            out.append("".join(chars[start:index]))
        out.append(char)
        start = index + 1
    if start < len(chars):
        out.append("".join(chars[start:]))
    return out


@dataclass
class WordPieceTokenizer:
    """一份 WordPiece 词表；``encode`` / ``decode`` 与 ``ByteTokenizer`` 同签名。"""

    vocab: Mapping[str, int]
    unk_token: str = "[UNK]"
    continuing_subword_prefix: str = "##"
    max_input_chars_per_word: int = 100
    do_lower_case: bool = True
    strip_accents: Optional[bool] = None
    clean_text: bool = True
    handle_chinese_chars: bool = True
    pad_id: Optional[int] = None
    eos_id: Optional[int] = None
    special_tokens: Optional[Iterable[str]] = None

    def __post_init__(self) -> None:
        if not self.vocab:
            raise WordPieceTokenizerError("词表不能为空")
        seen: Dict[int, str] = {}
        for token, index in self.vocab.items():
            if not isinstance(token, str) or not token:
                raise WordPieceTokenizerError(f"词表里有非法 token：{token!r}")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise WordPieceTokenizerError(f"token {token!r} 的 id 必须是非负整数")
            if index in seen:
                raise WordPieceTokenizerError(
                    f"id {index} 同时属于 {seen[index]!r} 与 {token!r}：词表 id 重复"
                )
            seen[index] = token
        if not isinstance(self.unk_token, str) or not self.unk_token:
            raise WordPieceTokenizerError(f"unk_token 非法：{self.unk_token!r}")
        if self.unk_token not in self.vocab:
            raise WordPieceTokenizerError(
                f"unk_token={self.unk_token!r} 不在词表里：匹配不上时无处可退"
            )
        if not isinstance(self.continuing_subword_prefix, str) or not self.continuing_subword_prefix:
            raise WordPieceTokenizerError(
                f"continuing_subword_prefix 必须是非空字符串，得到 {self.continuing_subword_prefix!r}"
            )
        if not isinstance(self.max_input_chars_per_word, int) or self.max_input_chars_per_word <= 0:
            raise WordPieceTokenizerError(
                "max_input_chars_per_word 必须是正整数，"
                f"得到 {self.max_input_chars_per_word!r}"
            )
        for name in ("do_lower_case", "clean_text", "handle_chinese_chars"):
            if not isinstance(getattr(self, name), bool):
                raise WordPieceTokenizerError(f"{name} 必须是 bool")
        if self.strip_accents is not None and not isinstance(self.strip_accents, bool):
            raise WordPieceTokenizerError("strip_accents 必须是 bool 或 None")
        total = self.vocab_size
        if self.pad_id is None:
            self.pad_id = self.vocab.get("[PAD]")
        if self.eos_id is None:
            self.eos_id = self.vocab.get("[SEP]")
        for name in ("pad_id", "eos_id"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise WordPieceTokenizerError(f"{name} 必须是整数或 None，得到 {value!r}")
            if not 0 <= value < total:
                raise WordPieceTokenizerError(f"{name}={value} 超出词表 [0, {total})")
        if self.special_tokens is None:
            names = {token for token in self.vocab if token.startswith("[") and token.endswith("]")}
        else:
            names = set(self.special_tokens)
            missing = [name for name in names if name not in self.vocab]
            if missing:
                raise WordPieceTokenizerError(f"特殊 token 不在词表里：{missing[0]!r}")
        object.__setattr__(self, "vocab", dict(self.vocab))
        object.__setattr__(self, "special_tokens", frozenset(names))
        object.__setattr__(self, "_id_map", {index: token for token, index in self.vocab.items()})
        object.__setattr__(
            self,
            "_special_ids",
            frozenset(self.vocab[token] for token in names),
        )

    # -- 构造 ---------------------------------------------------------

    @classmethod
    def from_vocab_file(cls, path, **kwargs) -> "WordPieceTokenizer":
        """读 ``vocab.txt``：一行一个 token，行号即 id。

        文件末尾的换行不算空行；**中间**的空行会被拒绝——它会让后面所有
        token 的 id 整体偏移，而这类错位不会在解码时报错。
        """
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as handle:
                text = handle.read()
        except OSError as exc:
            raise WordPieceTokenizerError(f"读不了词表 {os.fspath(path)!r}：{exc}") from exc
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        vocab: Dict[str, int] = {}
        for index, line in enumerate(lines):
            token = line[:-1] if line.endswith("\r") else line
            if not token:
                raise WordPieceTokenizerError(f"词表第 {index + 1} 行是空的：行号即 id，不能跳行")
            if token in vocab:
                raise WordPieceTokenizerError(
                    f"词表里 {token!r} 出现两次（第 {vocab[token] + 1} 行与第 {index + 1} 行）"
                )
            vocab[token] = index
        return cls(vocab=vocab, **kwargs)

    @classmethod
    def from_dict(cls, data: Mapping, **kwargs) -> "WordPieceTokenizer":
        """读 HF ``tokenizer.json`` 里的 WordPiece 段。"""
        if not isinstance(data, Mapping):
            raise WordPieceTokenizerError(
                f"tokenizer.json 的顶层必须是对象，得到 {type(data).__name__}"
            )
        model = data.get("model")
        if not isinstance(model, Mapping):
            raise WordPieceTokenizerError("缺少 model 段")
        kind = model.get("type")
        if kind != "WordPiece":
            raise WordPieceTokenizerError(
                f"model.type={kind!r}：这里只吃 WordPiece，BPE 请用 BpeTokenizer"
            )
        vocab = model.get("vocab")
        if not isinstance(vocab, Mapping):
            raise WordPieceTokenizerError("model.vocab 必须是 token → id 的对象")
        options = dict(kwargs)
        for key, field_name in (
            ("unk_token", "unk_token"),
            ("continuing_subword_prefix", "continuing_subword_prefix"),
            ("max_input_chars_per_word", "max_input_chars_per_word"),
        ):
            if model.get(key) is not None and field_name not in options:
                options[field_name] = model[key]
        normalizer = data.get("normalizer")
        if normalizer is not None:
            options.update(_normalizer_options(normalizer))
        pre_tokenizer = data.get("pre_tokenizer")
        if pre_tokenizer is not None:
            _check_pre_tokenizer(pre_tokenizer)
        for entry in data.get("added_tokens") or []:
            if not isinstance(entry, Mapping):
                raise WordPieceTokenizerError(f"added_tokens 里有一项不是对象：{entry!r}")
            content = entry.get("content")
            if not isinstance(content, str) or not content:
                raise WordPieceTokenizerError(f"added token 的 content 非法：{content!r}")
            if content not in vocab:
                raise WordPieceTokenizerError(
                    f"added token {content!r} 不在 model.vocab 里：它无法参与编码"
                )
        return cls(vocab=vocab, **options)

    @classmethod
    def from_file(cls, path, **kwargs) -> "WordPieceTokenizer":
        """按扩展名分派：``.json`` 走 tokenizer.json，其它按 ``vocab.txt``。"""
        suffix = os.path.splitext(os.fspath(path))[1].lower()
        if suffix == ".json":
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except OSError as exc:
                raise WordPieceTokenizerError(
                    f"读不了 {os.fspath(path)!r}：{exc}"
                ) from exc
            except json.JSONDecodeError as exc:
                raise WordPieceTokenizerError(
                    f"{os.fspath(path)!r} 不是合法 JSON：{exc}"
                ) from exc
            return cls.from_dict(data, **kwargs)
        return cls.from_vocab_file(path, **kwargs)

    # -- 属性 ---------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        highest = max(self.vocab.values())
        return highest + 1

    def spec(self, **overrides) -> LMSpec:
        for key in ("vocab_size", "pad_id", "eos_id"):
            if key in overrides:
                raise ValueError(f"spec() 不接受覆盖 {key}：它由词表决定")
        if self.eos_id is None:
            raise WordPieceTokenizerError(
                "词表里没有 [SEP]，eos_id 无从推断；请显式传 eos_id=..."
            )
        return LMSpec(
            vocab_size=self.vocab_size,
            pad_id=int(self.pad_id if self.pad_id is not None else 0),
            eos_id=int(self.eos_id),
            **overrides,
        )

    # -- 编码 ---------------------------------------------------------

    def _basic_tokenize(self, text: str) -> List[str]:
        if self.clean_text:
            text = _clean_text(text)
        if self.handle_chinese_chars:
            text = _tokenize_chinese_chars(text)
        strip = self.do_lower_case if self.strip_accents is None else self.strip_accents
        if self.do_lower_case:
            text = text.lower()
        if strip:
            text = _strip_accents(text)
        out: List[str] = []
        for token in text.split():
            out.extend(_split_on_punctuation(token))
        return out

    def _wordpiece(self, token: str) -> List[str]:
        if len(token) > self.max_input_chars_per_word:
            return [self.unk_token]
        pieces: List[str] = []
        start = 0
        while start < len(token):
            end = len(token)
            match: Optional[str] = None
            while start < end:
                candidate = token[start:end]
                if start > 0:
                    candidate = self.continuing_subword_prefix + candidate
                if candidate in self.vocab:
                    match = candidate
                    break
                end -= 1
            if match is None:
                return [self.unk_token]
            pieces.append(match)
            start = end
        return pieces

    def encode(self, text: str) -> List[int]:
        """编码为 id 序列；匹配不上的片段落到 ``[UNK]``。"""
        ids: List[int] = []
        for token in self._basic_tokenize(str(text)):
            for piece in self._wordpiece(token):
                ids.append(self.vocab[piece])
        return ids

    # -- 解码 ---------------------------------------------------------

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        """把 id 还原成文本：``##`` 续接片段拼回前一个 token，eos 处停止。

        先判 eos 再判 pad：BERT 里两者通常同 id（``[SEP]`` 与 ``[PAD]`` 不同，
        但推理侧常把 ``[SEP]`` 当句末），此时"停止"比"跳过"更接近原意。
        """
        pieces: List[str] = []
        for raw in ids:
            index = int(raw)
            if self.eos_id is not None and index == self.eos_id:
                break
            if self.pad_id is not None and index == self.pad_id:
                continue
            token = self._id_map.get(index)
            if token is None:
                raise WordPieceTokenizerError(
                    f"id {index} 不在词表内（vocab_size={self.vocab_size}）"
                )
            if skip_special_tokens and index in self._special_ids:
                continue
            pieces.append(token)
        text = " ".join(pieces)
        if self.continuing_subword_prefix:
            text = text.replace(" " + self.continuing_subword_prefix, "")
        return text


def _normalizer_options(normalizer: Mapping) -> Dict[str, object]:
    if not isinstance(normalizer, Mapping):
        raise WordPieceTokenizerError(
            f"normalizer 必须是对象或 null，得到 {type(normalizer).__name__}"
        )
    if normalizer.get("type") != "BertNormalizer":
        raise WordPieceTokenizerError(
            f"normalizer.type={normalizer.get('type')!r}：WordPiece 只吃 BertNormalizer"
            "（NFC / 自定义 Normalizer 会改切分边界）"
        )
    options: Dict[str, object] = {
        "clean_text": bool(normalizer.get("clean_text", True)),
        "handle_chinese_chars": bool(normalizer.get("handle_chinese_chars", True)),
    }
    lowercase = normalizer.get("lowercase")
    if lowercase is not None:
        options["do_lower_case"] = bool(lowercase)
    strip = normalizer.get("strip_accents")
    if strip is not None:
        options["strip_accents"] = bool(strip)
    return options


def _check_pre_tokenizer(pre_tokenizer: object) -> None:
    if isinstance(pre_tokenizer, Mapping):
        kind = pre_tokenizer.get("type")
        if kind == "BertPreTokenizer":
            return
        if kind == "Sequence":
            for part in pre_tokenizer.get("pretokenizers") or []:
                _check_pre_tokenizer(part)
            return
        raise WordPieceTokenizerError(
            f"pre_tokenizer.type={kind!r}：WordPiece 只吃 BertPreTokenizer"
        )
    if isinstance(pre_tokenizer, (list, tuple)):
        for part in pre_tokenizer:
            _check_pre_tokenizer(part)
        return
    raise WordPieceTokenizerError(
        f"pre_tokenizer 形态不认识：{type(pre_tokenizer).__name__}"
    )
