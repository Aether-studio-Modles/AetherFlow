"""Unigram 词表（T5 / ALBERT / mBART 谱系）：``tokenizer.json`` 的 ``Unigram`` 模型。

为什么又是一份实现：ByteLevel BPE 按合并表逐个吞并，WordPiece 对每个词做贪心
最长前缀匹配，而 Unigram 把分词看成一个**词表上的最优路径问题**——每个候选
片段带一个 log 概率，取总分最大的那条切分（Viterbi）。三者的切分边界互不相同，
共用一份代码只会让谁都说不清。

支持的范围（刻意窄，越界显式拒绝）：

- ``model.type == "Unigram"``，``vocab`` 是 ``[[piece, score], ...]`` 或
  ``piece -> score``；``unk_id`` 必填，``byte_fallback`` / ``fuse_unk`` 按语义实现；
- ``normalizer`` 为 null、``Precompiled``（SentencePiece 的
  ``precompiled_charsmap``：base64 里是"4 字节长度 + darts-clone 双数组 +
  NUL 分隔的替换串"），或它们的 ``Sequence``，以及 NFC / NFD / NFKC / NFKD；
- ``pre_tokenizer`` 为 null 或 ``Metaspace``（``replacement`` / ``split`` /
  ``add_prefix_space`` / ``prepend_scheme``），即把空格换成 ``▁`` 并可选地
  按它切块；
- ``decoder`` 为 null、``Metaspace``、``ByteFallback``、``Fuse`` 及其
  ``Sequence``。

拒绝而不是近似：``BertPreTokenizer`` / ``ByteLevel`` 预分词、``BertNormalizer``、
``Replace``（正则）、``Strip``——它们与 Unigram 的切分边界不同，近似实现产出的
id 每个都合法，但不是训练时那串。``sentencepiece.model`` 是二进制 protobuf，
这里只吃 ``tokenizer.json``。

``Precompiled`` 必须按原样实现的原因：它承载的是 NFKC 一类的字符映射，映射后
的等价类会改变 Viterbi 能走的路（例如全角字符被折成半角，才落进词表）。
只做 ``unicodedata.normalize`` 会得到另一串 id，而且**不会报错**。
"""

from __future__ import annotations

import base64
import json
import os
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .lm import LMSpec

__all__ = ["UnigramTokenizerError", "UnigramTokenizer"]

# 未知片段的代价：SentencePiece 取 ``min_score - 10``，保证它总比任何真词差，
# 但当整段都没有真词时仍能凑出一条路。
_UNK_PENALTY = 10.0
# 空格替换符 U+2581 LOWER ONE EIGHTH BLOCK（SentencePiece 的 ``▁``）。
_SPACE_MARKER = "\u2581"
# trie 的终止键：字节键都是 int，用 None 不会与它们冲突。
_TERM = None


class UnigramTokenizerError(ValueError):
    """词表不是受支持的 Unigram 形态，或 id / 分数不自洽。"""


# -- UTF-8 与 darts-clone 双数组 ------------------------------------------


def _utf8_step(raw: bytes, pos: int) -> int:
    """返回从 ``pos`` 开始的一个完整 UTF-8 字符占几个字节；非法或截断返回 0。"""
    lead = raw[pos]
    if lead < 0x80:
        size = 1
    elif 0xC2 <= lead <= 0xDF:
        size = 2
    elif 0xE0 <= lead <= 0xEF:
        size = 3
    elif 0xF0 <= lead <= 0xF4:
        size = 4
    else:
        return 0
    if pos + size > len(raw):
        return 0
    try:
        raw[pos:pos + size].decode("utf-8")
    except UnicodeDecodeError:
        return 0
    return size


def _has_leaf(unit: int) -> bool:
    return ((unit >> 8) & 1) == 1


def _unit_value(unit: int) -> int:
    return unit & 0x7FFFFFFF


def _unit_label(unit: int) -> int:
    return unit & 0x800000FF


def _unit_offset(unit: int) -> int:
    return (unit >> 10) << ((unit & 0x200) >> 6)


def _decode_charsmap(blob: bytes) -> Tuple[List[int], bytes]:
    """把 ``precompiled_charsmap`` 拆成 (double-array units, 替换串)。

    头部是 4 字节小端长度，后面紧跟双数组本体，再后面是以 NUL 结尾的替换串。
    ``trie`` 的大小必须是 1024 的整数倍——这是 darts-clone 建表器的约定，
    SentencePiece 的读取端也这么校验。
    """
    if len(blob) <= 4:
        raise UnigramTokenizerError("precompiled_charsmap 太短：连 4 字节头部都不完整")
    size = int.from_bytes(blob[:4], "little")
    if size < 1024 or (size & 0x3FF) != 0:
        raise UnigramTokenizerError(
            f"precompiled_charsmap 的 trie 长度 {size} 不是 1024 的整数倍"
        )
    if size + 4 >= len(blob):
        raise UnigramTokenizerError(
            f"precompiled_charsmap 声明的 trie 长度 {size} 超出载荷 {len(blob) - 4}"
        )
    body = blob[4:4 + size]
    units = [
        int.from_bytes(body[index:index + 4], "little")
        for index in range(0, size, 4)
    ]
    normalized = blob[4 + size:]
    if not normalized or normalized[-1:] != b"\x00":
        raise UnigramTokenizerError("precompiled_charsmap 的替换串必须以 NUL 结尾")
    return units, normalized


class _Precompiled:
    """SentencePiece 的字符映射：最长前缀匹配 + 替换串。"""

    def __init__(self, blob: bytes) -> None:
        self.units, self.normalized = _decode_charsmap(blob)

    def _matches(self, raw: bytes, pos: int) -> List[Tuple[int, int]]:
        """darts-clone ``commonPrefixSearch``：返回 (匹配字节数, 替换串偏移)。"""
        units = self.units
        total = len(units)
        node = 0
        node ^= _unit_offset(units[node])
        out: List[Tuple[int, int]] = []
        index = pos
        while index < len(raw):
            byte = raw[index]
            node ^= byte
            if node < 0 or node >= total:
                return out
            unit = units[node]
            if _unit_label(unit) != byte:
                return out
            node ^= _unit_offset(unit)
            if node < 0 or node >= total:
                return out
            if _has_leaf(unit):
                out.append((index - pos + 1, _unit_value(units[node])))
            index += 1
        return out

    def transform(self, text: str) -> str:
        raw = text.encode("utf-8")
        out = bytearray()
        pos = 0
        while pos < len(raw):
            longest = 0
            value = 0
            for length, offset in self._matches(raw, pos):
                if length > longest:
                    longest = length
                    value = offset
            if longest and value < len(self.normalized):
                end = self.normalized.find(b"\x00", value)
                out += self.normalized[value:end]
                pos += longest
                continue
            step = _utf8_step(raw, pos)
            if step == 0:
                # 非法 UTF-8：消费 1 字节、输出替换字符（与 SentencePiece 一致）。
                out += b"\xef\xbf\xbd"
                pos += 1
            else:
                out += raw[pos:pos + step]
                pos += step
        return out.decode("utf-8")


def _normalizer_ops(spec: Optional[Mapping]) -> Tuple[object, ...]:
    """把 ``normalizer`` 段展开成一串可调用对象（_Precompiled 或归一化形式名）。"""
    if spec is None:
        return ()
    if isinstance(spec, (list, tuple)):
        ops: List[object] = []
        for sub in spec:
            ops.extend(_normalizer_ops(sub))
        return tuple(ops)
    if not isinstance(spec, Mapping):
        raise UnigramTokenizerError(
            f"normalizer 必须是对象、数组或 null，得到 {type(spec).__name__}"
        )
    kind = spec.get("type")
    if kind == "Sequence":
        ops = []
        for sub in spec.get("normalizers") or []:
            ops.extend(_normalizer_ops(sub))
        return tuple(ops)
    if kind == "Precompiled":
        payload = spec.get("precompiled_charsmap")
        if not isinstance(payload, str) or not payload:
            raise UnigramTokenizerError("Precompiled 缺少 precompiled_charsmap 字符串")
        try:
            blob = base64.b64decode(payload, validate=True)
        except (ValueError, TypeError) as exc:
            raise UnigramTokenizerError(f"precompiled_charsmap 不是合法 base64：{exc}") from exc
        return (_Precompiled(blob),)
    if kind in ("NFC", "NFD", "NFKC", "NFKD"):
        return (kind,)
    raise UnigramTokenizerError(
        f"normalizer.type={kind!r}：Unigram 只吃 Precompiled / NFC / NFD / NFKC / NFKD"
        " 及其 Sequence（BertNormalizer / Replace / Strip 会改切分边界）"
    )


def _apply_normalizers(text: str, ops: Sequence[object]) -> str:
    for op in ops:
        if isinstance(op, _Precompiled):
            text = op.transform(text)
        else:
            text = unicodedata.normalize(str(op), text)
    return text


def _read_pre_tokenizer(spec: Optional[Mapping]) -> Optional[Dict[str, object]]:
    """读出 Metaspace 的四个开关；null 表示不做预分词。"""
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        merged: Optional[Dict[str, object]] = None
        for sub in spec:
            current = _read_pre_tokenizer(sub)
            if current is None:
                continue
            if merged is None:
                merged = current
            elif merged != current:
                raise UnigramTokenizerError(
                    f"pre_tokenizer.Sequence 里 Metaspace 配置不一致：{merged} 与 {current}"
                )
        return merged
    if not isinstance(spec, Mapping):
        raise UnigramTokenizerError(
            f"pre_tokenizer 必须是对象、数组或 null，得到 {type(spec).__name__}"
        )
    kind = spec.get("type")
    if kind == "Sequence":
        merged = None
        for sub in spec.get("pretokenizers") or []:
            current = _read_pre_tokenizer(sub)
            if current is None:
                continue
            if merged is None:
                merged = current
            elif merged != current:
                raise UnigramTokenizerError(
                    f"pre_tokenizer.Sequence 里 Metaspace 配置不一致：{merged} 与 {current}"
                )
        return merged
    if kind != "Metaspace":
        raise UnigramTokenizerError(
            f"pre_tokenizer.type={kind!r}：Unigram 只吃 Metaspace（ByteLevel / "
            "BertPreTokenizer / Split 都会改切分边界）"
        )
    replacement = spec.get("replacement", _SPACE_MARKER)
    if not isinstance(replacement, str) or not replacement:
        raise UnigramTokenizerError(f"Metaspace.replacement 非法：{replacement!r}")
    add_prefix_space = spec.get("add_prefix_space", True)
    if not isinstance(add_prefix_space, bool):
        raise UnigramTokenizerError("Metaspace.add_prefix_space 必须是 bool")
    scheme = spec.get("prepend_scheme")
    if scheme is None:
        scheme = "always" if add_prefix_space else "never"
    if scheme not in ("always", "never", "first"):
        raise UnigramTokenizerError(
            f"Metaspace.prepend_scheme={scheme!r}：只认 always / never / first"
        )
    split = spec.get("split", True)
    if not isinstance(split, bool):
        raise UnigramTokenizerError("Metaspace.split 必须是 bool")
    return {
        "replacement": replacement,
        "prepend": scheme,
        "split": split,
    }


def _read_decoder(spec: Optional[Mapping]) -> Dict[str, object]:
    """读出 decoder 的两件事：要不要把 ``▁`` 还原成空格、要不要按字节拼回。"""
    result: Dict[str, object] = {
        "metaspace": False,
        "byte_fallback": False,
        "replacement": _SPACE_MARKER,
        "add_prefix_space": True,
    }
    if spec is None:
        return result
    if isinstance(spec, (list, tuple)):
        for sub in spec:
            result.update(_read_decoder(sub))
        return result
    if not isinstance(spec, Mapping):
        raise UnigramTokenizerError(
            f"decoder 必须是对象、数组或 null，得到 {type(spec).__name__}"
        )
    kind = spec.get("type")
    if kind == "Sequence":
        for sub in spec.get("decoders") or []:
            result.update(_read_decoder(sub))
        return result
    if kind == "Metaspace":
        replacement = spec.get("replacement", _SPACE_MARKER)
        if not isinstance(replacement, str) or not replacement:
            raise UnigramTokenizerError(f"decoder.Metaspace.replacement 非法：{replacement!r}")
        add_prefix_space = spec.get("add_prefix_space", True)
        if not isinstance(add_prefix_space, bool):
            raise UnigramTokenizerError("decoder.Metaspace.add_prefix_space 必须是 bool")
        result.update(
            {
                "metaspace": True,
                "replacement": replacement,
                "add_prefix_space": add_prefix_space,
            }
        )
        return result
    if kind == "ByteFallback":
        result["byte_fallback"] = True
        return result
    if kind == "Fuse":
        # 拼接本来就是默认行为，显式声明一次不改变结果。
        return result
    raise UnigramTokenizerError(
        f"decoder.type={kind!r}：只支持 Metaspace / ByteFallback / Fuse 及其 Sequence"
    )


@dataclass
class UnigramTokenizer:
    """一份 Unigram 词表；``encode`` / ``decode`` 与 ``ByteTokenizer`` 同签名。"""

    vocab: Mapping[str, int]
    scores: Mapping[str, float]
    unk_id: int
    normalizer: Optional[Mapping] = None
    pre_tokenizer: Optional[Mapping] = None
    decoder: Optional[Mapping] = None
    byte_fallback: bool = False
    fuse_unk: bool = True
    added: Mapping[str, int] = field(default_factory=dict)
    special_ids: frozenset = frozenset()
    pad_id: Optional[int] = None
    eos_id: Optional[int] = None
    special_tokens: Optional[Iterable[str]] = None

    def __post_init__(self) -> None:
        if not self.vocab:
            raise UnigramTokenizerError("词表不能为空")
        seen: Dict[int, str] = {}
        for piece, index in self.vocab.items():
            if not isinstance(piece, str) or not piece:
                raise UnigramTokenizerError(f"词表里有非法 token：{piece!r}")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise UnigramTokenizerError(f"token {piece!r} 的 id 必须是非负整数")
            if index in seen:
                raise UnigramTokenizerError(
                    f"id {index} 同时属于 {seen[index]!r} 与 {piece!r}：词表 id 重复"
                )
            seen[index] = piece
        missing = [piece for piece in self.vocab if piece not in self.scores]
        if missing:
            raise UnigramTokenizerError(
                f"token {missing[0]!r} 没有分数：Unigram 需要每个片段的 log 概率"
            )
        extra = [piece for piece in self.scores if piece not in self.vocab]
        if extra:
            raise UnigramTokenizerError(f"scores 里的 {extra[0]!r} 不在词表中")
        clean_scores: Dict[str, float] = {}
        for piece, score in self.scores.items():
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise UnigramTokenizerError(f"token {piece!r} 的分数必须是实数，得到 {score!r}")
            if score != score or score in (float("inf"), float("-inf")):
                raise UnigramTokenizerError(f"token {piece!r} 的分数不是有限值：{score!r}")
            clean_scores[piece] = float(score)
        total = self.vocab_size
        if not isinstance(self.unk_id, int) or isinstance(self.unk_id, bool):
            raise UnigramTokenizerError(f"unk_id 必须是整数，得到 {self.unk_id!r}")
        if not 0 <= self.unk_id < total:
            raise UnigramTokenizerError(f"unk_id={self.unk_id} 超出词表 [0, {total})")
        for name in ("byte_fallback", "fuse_unk"):
            if not isinstance(getattr(self, name), bool):
                raise UnigramTokenizerError(f"{name} 必须是 bool")
        added_seen: Dict[int, str] = {}
        for name, index in self.added.items():
            if not isinstance(name, str) or not name:
                raise UnigramTokenizerError(f"added token 的名字非法：{name!r}")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise UnigramTokenizerError(f"added token {name!r} 的 id 必须是非负整数")
            if self.vocab.get(name) != index:
                raise UnigramTokenizerError(
                    f"added token {name!r} 不在词表里或 id 不一致：它会与 Viterbi 的结果冲突"
                )
            taken = added_seen.get(index)
            if taken is not None and taken != name:
                raise UnigramTokenizerError(
                    f"added tokens {taken!r} 与 {name!r} 占用同一个 id {index}"
                )
            added_seen[index] = name
        if self.pad_id is None:
            self.pad_id = self.vocab.get("<pad>")
        if self.eos_id is None:
            self.eos_id = self.vocab.get("</s>", self.vocab.get("<eos>"))
        for name in ("pad_id", "eos_id"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise UnigramTokenizerError(f"{name} 必须是整数或 None，得到 {value!r}")
            if not 0 <= value < total:
                raise UnigramTokenizerError(f"{name}={value} 超出词表 [0, {total})")
        for index in self.special_ids:
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < total:
                raise UnigramTokenizerError(f"special_ids 里的 {index!r} 超出词表 [0, {total})")
        if self.special_tokens is None:
            # 没显式给名字时，优先用 tokenizer.json 的 special 标记；
            # 标记为空才把所有 added token 当特殊 token。
            special = set(self.special_ids)
            if not special:
                special = {self.vocab[name] for name in self.added}
        else:
            names = set(self.special_tokens)
            missing_names = [name for name in names if name not in self.vocab]
            if missing_names:
                raise UnigramTokenizerError(f"特殊 token 不在词表里：{missing_names[0]!r}")
            special = {self.vocab[name] for name in names}
        object.__setattr__(self, "vocab", dict(self.vocab))
        object.__setattr__(self, "scores", clean_scores)
        object.__setattr__(self, "added", dict(self.added))
        object.__setattr__(self, "special_ids", frozenset(self.special_ids))
        table = {index: piece for piece, index in self.vocab.items()}
        object.__setattr__(self, "_id_map", table)
        object.__setattr__(self, "_special_ids", frozenset(special))
        # added tokens 按首字符分桶、桶内按长度降序：每步只试同首字符的那几个。
        by_first: Dict[str, List[str]] = {}
        for name in sorted(self.added, key=len, reverse=True):
            by_first.setdefault(name[0], []).append(name)
        object.__setattr__(self, "_added_by_first", {k: tuple(v) for k, v in by_first.items()})
        object.__setattr__(self, "_min_score", min(clean_scores.values()))
        object.__setattr__(self, "_unk_score", self._min_score - _UNK_PENALTY)
        object.__setattr__(self, "_trie", _build_trie(self.vocab, clean_scores))
        object.__setattr__(self, "_normalizers", _normalizer_ops(self.normalizer))
        metaspace = _read_pre_tokenizer(self.pre_tokenizer)
        object.__setattr__(self, "_metaspace", metaspace)
        decoder = _read_decoder(self.decoder)
        object.__setattr__(self, "_decoder", decoder)

    # -- 构造 ---------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping, **kwargs) -> "UnigramTokenizer":
        """读 HF ``tokenizer.json`` 里的 Unigram 段。"""
        if not isinstance(data, Mapping):
            raise UnigramTokenizerError(
                f"tokenizer.json 的顶层必须是对象，得到 {type(data).__name__}"
            )
        model = data.get("model")
        if not isinstance(model, Mapping):
            raise UnigramTokenizerError("缺少 model 段")
        kind = model.get("type")
        if kind == "BPE":
            raise UnigramTokenizerError("model.type='BPE'：请用 BpeTokenizer")
        if kind == "WordPiece":
            raise UnigramTokenizerError("model.type='WordPiece'：请用 WordPieceTokenizer")
        if kind != "Unigram":
            raise UnigramTokenizerError(f"model.type={kind!r}：这里只吃 Unigram")
        raw_vocab = model.get("vocab")
        vocab: Dict[str, int] = {}
        scores: Dict[str, float] = {}
        if isinstance(raw_vocab, Mapping):
            for piece, score in raw_vocab.items():
                if not isinstance(piece, str):
                    raise UnigramTokenizerError(f"model.vocab 的键必须是字符串：{piece!r}")
                scores[piece] = score
                vocab[piece] = len(vocab)
        elif isinstance(raw_vocab, (list, tuple)):
            for entry in raw_vocab:
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    raise UnigramTokenizerError(
                        f"model.vocab 的每一项应该是 [piece, score]，得到 {entry!r}"
                    )
                piece, score = entry
                if not isinstance(piece, str) or not piece:
                    raise UnigramTokenizerError(f"model.vocab 里有非法 token：{piece!r}")
                if piece in vocab:
                    raise UnigramTokenizerError(f"model.vocab 里 {piece!r} 出现两次")
                vocab[piece] = len(vocab)
                scores[piece] = score
        else:
            raise UnigramTokenizerError("model.vocab 必须是 [piece, score] 数组或对象")
        unk_id = model.get("unk_id")
        if unk_id is None:
            raise UnigramTokenizerError("缺少 model.unk_id：未知片段需要回退目标")
        byte_fallback = model.get("byte_fallback", False)
        if not isinstance(byte_fallback, bool):
            raise UnigramTokenizerError("model.byte_fallback 必须是 bool")
        added: Dict[str, int] = {}
        special: set = set()
        for entry in data.get("added_tokens") or []:
            if not isinstance(entry, Mapping):
                raise UnigramTokenizerError(f"added_tokens 里有一项不是对象：{entry!r}")
            content = entry.get("content")
            index = entry.get("id")
            if not isinstance(content, str) or not content:
                raise UnigramTokenizerError(f"added token 的 content 非法：{content!r}")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise UnigramTokenizerError(f"added token {content!r} 的 id 非法：{index!r}")
            added[content] = index
            if entry.get("special"):
                special.add(index)
        options = dict(kwargs)
        options.setdefault("added", added)
        options.setdefault("special_ids", frozenset(special))
        return cls(
            vocab=vocab,
            scores=scores,
            unk_id=unk_id,
            byte_fallback=byte_fallback,
            normalizer=data.get("normalizer"),
            pre_tokenizer=data.get("pre_tokenizer"),
            decoder=data.get("decoder"),
            **options,
        )

    @classmethod
    def from_file(cls, path, **kwargs) -> "UnigramTokenizer":
        """读 ``tokenizer.json``；``sentencepiece.model`` 是二进制 protobuf，拒绝。"""
        suffix = os.path.splitext(os.fspath(path))[1].lower()
        if suffix != ".json":
            raise UnigramTokenizerError(
                f"{os.fspath(path)!r} 不是 tokenizer.json：SentencePiece 的 .model 是"
                "二进制 protobuf，这里只吃 HuggingFace 导出的 JSON"
            )
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except OSError as exc:
            raise UnigramTokenizerError(f"读不了 {os.fspath(path)!r}：{exc}") from exc
        except json.JSONDecodeError as exc:
            raise UnigramTokenizerError(
                f"{os.fspath(path)!r} 不是合法 JSON：{exc}"
            ) from exc
        return cls.from_dict(data, **kwargs)

    # -- 属性 ---------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return max(self.vocab.values()) + 1

    def spec(self, **overrides) -> LMSpec:
        for key in ("vocab_size", "pad_id", "eos_id"):
            if key in overrides:
                raise ValueError(f"spec() 不接受覆盖 {key}：它由词表决定")
        if self.eos_id is None:
            raise UnigramTokenizerError(
                "词表里没有 </s> 或 <eos>，eos_id 无从推断；请显式传 eos_id=..."
            )
        return LMSpec(
            vocab_size=self.vocab_size,
            pad_id=int(self.pad_id if self.pad_id is not None else 0),
            eos_id=int(self.eos_id),
            **overrides,
        )

    # -- 编码 ---------------------------------------------------------

    def _pre_tokenize(self, text: str) -> List[str]:
        spec = self._metaspace
        if spec is None:
            return [text] if text else []
        marker = str(spec["replacement"])
        if spec["prepend"] in ("always", "first") and not text.startswith(marker):
            text = marker + text
        text = text.replace(" ", marker)
        if not spec["split"]:
            return [text] if text else []
        chunks: List[str] = []
        for position, part in enumerate(text.split(marker)):
            if position == 0:
                if part:
                    chunks.append(part)
            else:
                chunks.append(marker + part)
        return chunks

    def _encode_chunk(self, chunk: str) -> List[int]:
        raw = chunk.encode("utf-8")
        size = len(raw)
        best: List[Optional[float]] = [None] * (size + 1)
        back: List[Optional[Tuple[int, Optional[int]]]] = [None] * (size + 1)
        best[0] = 0.0
        pos = 0
        while pos < size:
            step = _utf8_step(raw, pos) or 1
            score = best[pos]
            if score is not None:
                has_single = False
                for length, (piece, piece_score, piece_id) in _trie_matches(
                    self._trie, raw, pos
                ):
                    end = pos + length
                    candidate = score + piece_score
                    if best[end] is None or candidate > best[end]:
                        best[end] = candidate
                        back[end] = (pos, piece_id)
                    if length == step:
                        has_single = True
                if not has_single:
                    if self.byte_fallback:
                        # 一个字节一个节点，这样未知字符也能被拆成 <0xXX>。
                        start = pos
                        running = score
                        for index in range(step):
                            end = start + 1
                            name = f"<0x{raw[start]:02X}>"
                            piece_id = self.vocab.get(name)
                            if piece_id is None:
                                raise UnigramTokenizerError(
                                    f"byte_fallback 需要 {name!r}，但词表里没有："
                                    "这份词表不能按字节回退"
                                )
                            candidate = running + self._unk_score
                            if best[end] is None or candidate > best[end]:
                                best[end] = candidate
                                back[end] = (start, piece_id)
                            running = best[end]
                            start = end
                    else:
                        end = pos + step
                        candidate = score + self._unk_score
                        if best[end] is None or candidate > best[end]:
                            best[end] = candidate
                            back[end] = (pos, None)
            pos += step
        if best[size] is None:
            raise UnigramTokenizerError("Viterbi 没能走到句尾：词表与分数不自洽")
        path: List[Tuple[int, int, Optional[int]]] = []
        end = size
        while end > 0:
            entry = back[end]
            if entry is None:
                raise UnigramTokenizerError(f"位置 {end} 没有回边：Viterbi 表不自洽")
            start, piece_id = entry
            path.append((start, end, piece_id))
            end = start
        path.reverse()
        # 保留 piece_id：byte_fallback 的片段已经有 id（<0xXX>），不能再按 unk 处理。
        # 路径里的字节不可单独解码成文本（byte_fallback 会切在 UTF-8 中间），
        # 所以这里只留 id，未命中即为 None。
        results: List[Optional[int]] = [piece_id for _, _, piece_id in path]
        if self.fuse_unk:
            fused: List[Optional[int]] = []
            in_unk = False
            for piece_id in results:
                if piece_id is None:
                    if not in_unk:
                        fused.append(None)
                        in_unk = True
                    continue
                in_unk = False
                fused.append(piece_id)
            results = fused
        return [self.unk_id if piece_id is None else piece_id for piece_id in results]

    def _encode_plain(self, text: str) -> List[int]:
        ids: List[int] = []
        for chunk in self._pre_tokenize(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def encode(self, text: str) -> List[int]:
        """编码为 id 序列；未知片段落到 ``unk_id``（可选按字节回退）。"""
        text = _apply_normalizers(str(text), self._normalizers)
        if not self.added:
            return self._encode_plain(text)
        ids: List[int] = []
        plain: List[str] = []
        index = 0
        while index < len(text):
            hit = None
            for candidate in self._added_by_first.get(text[index], ()):
                if text.startswith(candidate, index):
                    hit = candidate
                    break
            if hit is None:
                plain.append(text[index])
                index += 1
                continue
            if plain:
                ids.extend(self._encode_plain("".join(plain)))
                plain = []
            ids.append(self.added[hit])
            index += len(hit)
        if plain:
            ids.extend(self._encode_plain("".join(plain)))
        return ids

    # -- 解码 ---------------------------------------------------------

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        """把 id 还原成文本：``▁`` 还原为空格、``<0xXX>`` 先拼回字节。

        先判 eos 再判 pad：两者常常不同 id，但推理侧常把 eos 当句末，此时
        "停止"比"跳过"更接近原意。
        """
        decoder = self._decoder
        pieces: List[str] = []
        pending = bytearray()
        for raw in ids:
            index = int(raw)
            if self.eos_id is not None and index == self.eos_id:
                break
            if self.pad_id is not None and index == self.pad_id:
                continue
            piece = self._id_map.get(index)
            if piece is None:
                raise UnigramTokenizerError(
                    f"id {index} 不在词表内（vocab_size={self.vocab_size}）"
                )
            if skip_special_tokens and index in self._special_ids:
                continue
            if (
                decoder["byte_fallback"]
                and len(piece) == 6
                and piece.startswith("<0x")
                and piece.endswith(">")
            ):
                try:
                    pending.append(int(piece[3:5], 16))
                except ValueError as exc:
                    raise UnigramTokenizerError(
                        f"token {piece!r}（id {index}）不是合法的字节回退标记"
                    ) from exc
                continue
            if pending:
                pieces.append(bytes(pending).decode("utf-8", "replace"))
                pending.clear()
            pieces.append(piece)
        if pending:
            pieces.append(bytes(pending).decode("utf-8", "replace"))
        text = "".join(pieces)
        if decoder["metaspace"]:
            marker = str(decoder["replacement"])
            text = text.replace(marker, " ")
            if decoder["add_prefix_space"] and text.startswith(" "):
                text = text[1:]
        return text


def _build_trie(
    vocab: Mapping[str, int], scores: Mapping[str, float]
) -> Dict[object, object]:
    """把片段按 UTF-8 字节建一棵嵌套字典树；终止节点挂在 ``None`` 上。"""
    root: Dict[object, object] = {}
    for piece, index in vocab.items():
        node = root
        for byte in piece.encode("utf-8"):
            nxt = node.get(byte)
            if nxt is None:
                nxt = {}
                node[byte] = nxt
            node = nxt  # type: ignore[assignment]
        node[_TERM] = (piece, scores[piece], index)
    return root


def _trie_matches(
    trie: Mapping[object, object], raw: bytes, pos: int
) -> Iterable[Tuple[int, Tuple[str, float, int]]]:
    """从 ``pos`` 开始的所有词表前缀，按长度递增（短片段先被看到，平局时保留它）。"""
    node = trie
    index = pos
    while index < len(raw):
        nxt = node.get(raw[index])
        if nxt is None:
            return
        node = nxt  # type: ignore[assignment]
        index += 1
        hit = node.get(_TERM)  # type: ignore[union-attr]
        if hit is not None:
            yield index - pos, hit  # type: ignore[misc]
