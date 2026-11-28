"""HF ``tokenizer.json`` 的 ByteLevel BPE 读取器。

为什么自己读：与 ``weights.py`` 同一条理由——运行期只依赖 numpy，而分词器是
"权重装进来之后立刻要用的下一件事"。一个模型换进来，如果词表对不上，权重再
正确也只会输出错位的文本，而且**不会报错**。所以这里的判据是逐 token / 逐字节
的往返，不是"看起来差不多"。

支持的范围（刻意窄，越界显式拒绝）：

- ``model.type == "BPE"``，且词表覆盖 ByteLevel 的 256 个字节映射——这是 GPT-2
  谱系（GPT-2 / Qwen2 / Qwen3 的 tokenizer.json）的共同形态；
- ``normalizer`` 为 null，或 NFC / NFD / NFKC / NFKD 及其 Sequence；
- ``pre_tokenizer`` 为 null 或 ByteLevel 及其 Sequence，并尊重 ``use_regex``
  （False 时整段不切）与 ``add_prefix_space``；
- ``decoder`` 为 null / ByteLevel / Sequence（其中 Replace 不参与解码，ByteLevel
  已经能还原字节）。

拒绝而不是近似：WordPiece / Unigram（``model.type`` 不是 BPE）、SentencePiece
的 ``byte_fallback`` / ``end_of_word_suffix`` / ``continuing_subword_prefix``、
``Metaspace`` 预分词、``Split`` 预分词（它的 ``pattern`` 会改切分边界，例如
Llama-3 把数字按 1～3 位一组切；忽略 pattern 等于换了一份分词器），以及不认识
的组件类型。近似实现会安静地产出错 token，这类错误不会在解码时报错，只会在
下游表现为"模型有点笨"。

预分词的实现用 ``unicodedata.category`` 的 L* / N* 手写扫描，而不是把 GPT-2 的
``\\p{L}`` 正则塞进 ``re``——Python 的 ``re`` 不支持 Unicode 属性类，用
``[^\\W\\d_]`` 之类的近似会在组合符号（变音符、印度语系）边界上偏移。
"""

from __future__ import annotations

import json
import os
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .lm import LMSpec

__all__ = ["BpeTokenizerError", "BpeTokenizer"]


class BpeTokenizerError(ValueError):
    """tokenizer.json 不是受支持的 ByteLevel BPE，或用的 id 不在词表内。"""


def _bytes_to_unicode() -> Tuple[Dict[int, str], Dict[str, int]]:
    """GPT-2 的字节 → 可见字符映射（可打印字符保持不变，其余平移到 256 之后）。"""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    mapping = {b: chr(b) for b in printable}
    next_char = 256
    for byte in range(256):
        if byte not in mapping:
            mapping[byte] = chr(next_char)
            next_char += 1
    return mapping, {ch: b for b, ch in mapping.items()}


_BYTE_ENCODER, _BYTE_DECODER = _bytes_to_unicode()

# GPT-2 的英文缩略后缀；正则里是区分大小写的字面量。
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")

_EOS_CANDIDATES = (
    "<|endoftext|>",
    "<|end_of_text|>",
    "<|eos|>",
    "</s>",
    "<eos>",
)


def _is_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L")


def _is_number(ch: str) -> bool:
    return unicodedata.category(ch).startswith("N")


def _pre_tokenize(text: str) -> List[str]:
    """按 GPT-2 的规则切分，返回原始（未做字节映射）的片段。

    规则顺序：缩略后缀 → `` ?`` + 连续字母 / 连续数字 / 连续其它非空白 →
    连续空白。前导空格只吃一个，多出来的会落到空白分支——与 GPT-2 正则里
    ``\\s+(?!\\S)`` / ``\\s+`` 两条分支的合并结果一致。
    """
    pieces: List[str] = []
    i, total = 0, len(text)
    while i < total:
        chunk = None
        for suffix in _CONTRACTIONS:
            if text.startswith(suffix, i):
                chunk = suffix
                break
        if chunk is None:
            start = i + 1 if text[i] == " " else i
            if start < total and _is_letter(text[start]):
                end = start + 1
                while end < total and _is_letter(text[end]):
                    end += 1
                chunk = text[i:end]
            elif start < total and _is_number(text[start]):
                end = start + 1
                while end < total and _is_number(text[end]):
                    end += 1
                chunk = text[i:end]
            elif start < total and not text[start].isspace():
                end = start + 1
                while (
                    end < total
                    and not text[end].isspace()
                    and not _is_letter(text[end])
                    and not _is_number(text[end])
                ):
                    end += 1
                chunk = text[i:end]
        if chunk is None:
            end = i + 1
            while end < total and text[end].isspace():
                end += 1
            # GPT-2 的 ``\s+(?!\S)`` 在空白串后还有内容时会留下最后一个空白，
            # 让它并入下一片段的 `` ?`` 前缀；只有单独一个空白时才整体成 token。
            if end < total and end - i >= 2:
                end -= 1
            chunk = text[i:end]
        pieces.append(chunk)
        i += len(chunk)
    return pieces


def _split(text: str, use_regex: bool) -> List[str]:
    """按预分词开关切分：``use_regex=False`` 时整段作为一个片段。"""
    if use_regex:
        return _pre_tokenize(text)
    return [text] if text else []


def _apply_normalizer(text: str, spec: Optional[Mapping]) -> str:
    if spec is None:
        return text
    if not isinstance(spec, Mapping):
        raise BpeTokenizerError("normalizer 必须是对象或 null")
    kind = spec.get("type")
    if kind in ("NFC", "NFD", "NFKC", "NFKD"):
        return unicodedata.normalize(kind, text)
    if kind == "Sequence":
        for sub in spec.get("normalizers") or []:
            text = _apply_normalizer(text, sub)
        return text
    raise BpeTokenizerError(
        f"不支持的 normalizer 类型 {kind!r}：忽略它会让 encode 与训练时的分词不一致"
    )


def _read_pre_tokenizer(spec: Optional[Mapping]) -> Tuple[bool, bool]:
    """读出 ``(add_prefix_space, use_regex)``；不支持的形态直接拒绝。

    ``pre_tokenizer`` 为 null 表示"不预分词"，等价于整段不切。``Sequence`` 里
    多项配置必须一致，否则说明我们没有理解这份分词器。
    """
    if spec is None:
        return False, False
    if not isinstance(spec, Mapping):
        raise BpeTokenizerError("pre_tokenizer 必须是对象或 null")
    kind = spec.get("type")
    if kind == "Sequence":
        result: Optional[Tuple[bool, bool]] = None
        for sub in spec.get("pretokenizers") or []:
            current = _read_pre_tokenizer(sub)
            if result is None:
                result = current
            elif result != current:
                raise BpeTokenizerError(
                    f"pre_tokenizer.Sequence 里配置不一致：{result} 与 {current}"
                )
        return result if result is not None else (False, False)
    if kind == "ByteLevel":
        return (
            bool(spec.get("add_prefix_space", False)),
            bool(spec.get("use_regex", True)),
        )
    raise BpeTokenizerError(
        f"不支持的 pre_tokenizer 类型 {kind!r}：只支持 ByteLevel 及其 Sequence"
        "（Split 的 pattern 会改切分边界，不读它就是换了一份分词器）"
    )


def _check_decoder(spec: Optional[Mapping]) -> None:
    if spec is None:
        return
    if not isinstance(spec, Mapping):
        raise BpeTokenizerError("decoder 必须是对象或 null")
    kind = spec.get("type")
    if kind == "Sequence":
        for sub in spec.get("decoders") or []:
            _check_decoder(sub)
        return
    if kind not in ("ByteLevel", "Replace"):
        raise BpeTokenizerError(
            f"不支持的 decoder 类型 {kind!r}：本实现只按 ByteLevel 还原字节"
        )


def _parse_merges(raw: object) -> List[Tuple[str, str]]:
    if not isinstance(raw, list):
        raise BpeTokenizerError("model.merges 必须是列表")
    merges: List[Tuple[str, str]] = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            parts = item.split(" ")
        elif isinstance(item, list):
            parts = [str(part) for part in item]
        else:
            raise BpeTokenizerError(f"merges[{index}] 既不是字符串也不是数组：{item!r}")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise BpeTokenizerError(
                f"merges[{index}] 必须是两个非空片段（第一条通常是 'Ġ t'），得到 {item!r}"
            )
        merges.append((parts[0], parts[1]))
    return merges


@dataclass(frozen=True)
class BpeTokenizer:
    """一份 ByteLevel BPE 词表；``encode`` / ``decode`` 与 ``ByteTokenizer`` 同签名。"""

    vocab: Mapping[str, int]
    ranks: Mapping[Tuple[str, str], int]
    added: Mapping[str, int] = field(default_factory=dict)
    special_ids: frozenset = frozenset()
    normalizer: Optional[Mapping] = None
    add_prefix_space: bool = False
    use_regex: bool = True
    pad_id: int = 0
    eos_id: int = 1

    def __post_init__(self) -> None:
        if not self.vocab:
            raise BpeTokenizerError("词表不能为空")
        seen: Dict[int, str] = {}
        for token, index in self.vocab.items():
            if not isinstance(token, str) or not token:
                raise BpeTokenizerError(f"词表里有非法 token：{token!r}")
            if not isinstance(index, int) or index < 0:
                raise BpeTokenizerError(f"token {token!r} 的 id 必须是非负整数")
            if index in seen:
                raise BpeTokenizerError(
                    f"id {index} 同时属于 {seen[index]!r} 与 {token!r}：词表 id 重复"
                )
            seen[index] = token
        missing = [ch for ch in _BYTE_ENCODER.values() if ch not in self.vocab]
        if missing:
            raise BpeTokenizerError(
                "词表缺少 ByteLevel 的 256 个字节字符（缺 "
                f"{len(missing)} 个，例如 {missing[0]!r}）：这不是 ByteLevel BPE 词表，"
                "可能来自 SentencePiece"
            )
        added_seen: Dict[int, str] = {}
        for name, index in self.added.items():
            if not isinstance(name, str) or not name:
                raise BpeTokenizerError(f"added token 的名字非法：{name!r}")
            if not isinstance(index, int) or index < 0:
                raise BpeTokenizerError(f"added token {name!r} 的 id 必须是非负整数")
            clash = self.vocab.get(name)
            if clash is not None and clash != index:
                raise BpeTokenizerError(
                    f"added token {name!r} 想占 id {index}，但词表里它已经是 {clash}"
                )
            taken = seen.get(index)
            if taken is not None and taken != name:
                raise BpeTokenizerError(
                    f"added token {name!r} 的 id {index} 与词表 token {taken!r} 冲突"
                )
            previous = added_seen.get(index)
            if previous is not None and previous != name:
                raise BpeTokenizerError(
                    f"added tokens {previous!r} 与 {name!r} 占用同一个 id {index}"
                )
            added_seen[index] = name
        total = self.vocab_size
        if not 0 <= int(self.pad_id) < total:
            raise BpeTokenizerError(f"pad_id={self.pad_id} 超出词表 [0, {total})")
        if not 0 <= int(self.eos_id) < total:
            raise BpeTokenizerError(f"eos_id={self.eos_id} 超出词表 [0, {total})")
        object.__setattr__(self, "vocab", dict(self.vocab))
        object.__setattr__(self, "ranks", dict(self.ranks))
        object.__setattr__(self, "added", dict(self.added))
        object.__setattr__(self, "special_ids", frozenset(self.special_ids))
        # 合并结果按词形缓存；词形数量随语料增长，给一个上限避免长跑时无界。
        object.__setattr__(self, "_cache", {})
        table = {index: token for token, index in self.vocab.items()}
        table.update({index: token for token, index in self.added.items()})
        object.__setattr__(self, "_id_map", table)
        # added tokens 按首字符分桶、桶内按长度降序：编码时每步只试同首字符的那几个。
        by_first: Dict[str, List[str]] = {}
        for name in sorted(self.added, key=len, reverse=True):
            by_first.setdefault(name[0], []).append(name)
        object.__setattr__(self, "_added_by_first", {k: tuple(v) for k, v in by_first.items()})

    # -- 构造 ---------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping, *, eos_token: Optional[str] = None,
                  pad_token: Optional[str] = None) -> "BpeTokenizer":
        if not isinstance(data, Mapping):
            raise BpeTokenizerError(f"tokenizer.json 的顶层必须是对象，得到 {type(data).__name__}")
        model = data.get("model")
        if not isinstance(model, Mapping):
            raise BpeTokenizerError("缺少 model 段")
        kind = model.get("type")
        if kind != "BPE":
            raise BpeTokenizerError(
                f"model.type={kind!r}：本实现只支持 BPE（WordPiece / Unigram 请先用"
                "对应工具转成 ByteLevel BPE）"
            )
        for key in ("byte_fallback", "end_of_word_suffix", "continuing_subword_prefix"):
            if model.get(key):
                raise BpeTokenizerError(
                    f"model.{key}={model.get(key)!r} 是 SentencePiece 语义，"
                    "与 ByteLevel BPE 不兼容"
                )
        if model.get("dropout"):
            raise BpeTokenizerError("model.dropout 非空：BPE dropout 是不确定的分词，不支持")

        vocab = model.get("vocab")
        if not isinstance(vocab, Mapping):
            raise BpeTokenizerError("model.vocab 必须是 token → id 的对象")
        ranks = {pair: index for index, pair in enumerate(_parse_merges(model.get("merges")))}

        _apply_normalizer("", data.get("normalizer"))  # 只做类型校验
        add_prefix_space, use_regex = _read_pre_tokenizer(data.get("pre_tokenizer"))
        _check_decoder(data.get("decoder"))

        added: Dict[str, int] = {}
        special: set = set()
        for entry in data.get("added_tokens") or []:
            if not isinstance(entry, Mapping):
                raise BpeTokenizerError(f"added_tokens 里有一项不是对象：{entry!r}")
            content = entry.get("content")
            index = entry.get("id")
            if not isinstance(content, str) or not content:
                raise BpeTokenizerError(f"added token 的 content 非法：{content!r}")
            if not isinstance(index, int) or index < 0:
                raise BpeTokenizerError(f"added token {content!r} 的 id 非法：{index!r}")
            if entry.get("lstrip") or entry.get("rstrip"):
                raise BpeTokenizerError(
                    f"added token {content!r} 带 lstrip / rstrip：本实现按原文整体匹配，"
                    "不裁剪两侧空白"
                )
            added[content] = index
            if entry.get("special"):
                special.add(index)

        tokenizer = cls(
            vocab={str(k): int(v) for k, v in vocab.items()},
            ranks=ranks,
            added=added,
            special_ids=special,
            normalizer=data.get("normalizer"),
            add_prefix_space=add_prefix_space,
            use_regex=use_regex,
            pad_id=0,
            eos_id=0,
        )
        eos_id = _resolve_id(tokenizer, eos_token, _EOS_CANDIDATES, "eos")
        pad_id = eos_id if pad_token is None else _resolve_id(tokenizer, pad_token, (), "pad")
        object.__setattr__(tokenizer, "eos_id", eos_id)
        object.__setattr__(tokenizer, "pad_id", pad_id)
        return tokenizer

    @classmethod
    def from_file(cls, path, **kwargs) -> "BpeTokenizer":
        with open(os.fspath(path), "r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError as exc:
                raise BpeTokenizerError(f"{path} 不是合法的 JSON（{exc}）") from exc
        return cls.from_dict(data, **kwargs)

    # -- 词表约定 -----------------------------------------------------

    @property
    def vocab_size(self) -> int:
        highest = max(self.vocab.values())
        if self.added:
            highest = max(highest, max(self.added.values()))
        return highest + 1

    def spec(self, **overrides) -> LMSpec:
        for key in ("vocab_size", "pad_id", "eos_id"):
            if key in overrides:
                raise ValueError(f"spec() 不接受覆盖 {key}：它由词表决定")
        return LMSpec(
            vocab_size=self.vocab_size,
            pad_id=self.pad_id,
            eos_id=self.eos_id,
            **overrides,
        )

    # -- 编码 ---------------------------------------------------------

    def _merge(self, word: str) -> Tuple[str, ...]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        symbols = list(word)
        while len(symbols) > 1:
            best: Optional[Tuple[str, str]] = None
            best_rank: Optional[int] = None
            for pair in zip(symbols, symbols[1:]):
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = pair, rank
            if best is None:
                break
            merged: List[str] = []
            index = 0
            while index < len(symbols):
                if (
                    index + 1 < len(symbols)
                    and symbols[index] == best[0]
                    and symbols[index + 1] == best[1]
                ):
                    merged.append(best[0] + best[1])
                    index += 2
                else:
                    merged.append(symbols[index])
                    index += 1
            symbols = merged
        result = tuple(symbols)
        if len(self._cache) >= 8192:
            self._cache.clear()
        self._cache[word] = result
        return result

    def _encode_plain(self, text: str) -> List[int]:
        ids: List[int] = []
        for piece in _split(text, self.use_regex):
            mapped = "".join(_BYTE_ENCODER[byte] for byte in piece.encode("utf-8"))
            for symbol in self._merge(mapped):
                index = self.vocab.get(symbol)
                if index is None:
                    raise BpeTokenizerError(
                        f"合并后的片段 {symbol!r} 不在词表里：merges 与 vocab 不自洽"
                    )
                ids.append(index)
        return ids

    def encode(self, text: str) -> List[int]:
        """编码为 id 序列。

        先按 added token 的原文整体匹配（同首字符的里长的优先），剩余片段才走
        BPE——这与 HF 的行为一致：特殊 token 不参与合并，否则 ``<|im_start|>``
        会被拆成一堆字节。
        """
        text = _apply_normalizer(str(text), self.normalizer)
        if self.add_prefix_space and not text.startswith(" "):
            text = " " + text
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
        """把 id 还原成文本；跳过 pad、在第一个 eos 处停止。

        连续的字节片段要**先拼起来再解 UTF-8**：一个汉字在字节级词表里是三个
        token，逐个解会得到三个替换字符。added token（含特殊 token）按原文输出，
        不参与字节还原。先判 eos 再判 pad：两者常常同 id（如 Qwen 的
        ``pad = <|endoftext|>``），此时"停止"比"跳过"更接近原意。
        """
        out: List[str] = []
        pending = bytearray()
        for raw in ids:
            index = int(raw)
            if index == self.eos_id:
                break
            if index == self.pad_id:
                continue
            token = self._id_map.get(index)
            if token is None:
                raise BpeTokenizerError(f"id {index} 不在词表内（vocab_size={self.vocab_size}）")
            if index in self.special_ids and skip_special_tokens:
                continue
            if index in self.special_ids or token in self.added:
                if pending:
                    out.append(_decode_utf8(bytes(pending)))
                    pending.clear()
                out.append(token)
                continue
            for ch in token:
                byte = _BYTE_DECODER.get(ch)
                if byte is None:
                    raise BpeTokenizerError(
                        f"token {token!r}（id {index}）含非 ByteLevel 字符，无法还原字节"
                    )
                pending.append(byte)
        if pending:
            out.append(_decode_utf8(bytes(pending)))
        return "".join(out)


def _decode_utf8(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BpeTokenizerError(
            f"这段 id 不是完整的 UTF-8 序列（可能被截断或来自另一份词表）：{exc}"
        ) from exc


def _resolve_id(
    tokenizer: BpeTokenizer, name: Optional[str], candidates: Sequence[str], role: str
) -> int:
    if name is not None:
        index = tokenizer.added.get(name, tokenizer.vocab.get(name))
        if index is None:
            raise BpeTokenizerError(f"{role}_token={name!r} 不在词表或 added tokens 里")
        return int(index)
    for candidate in candidates:
        index = tokenizer.added.get(candidate, tokenizer.vocab.get(candidate))
        if index is not None:
            return int(index)
    raise BpeTokenizerError(
        f"无法从词表推断 {role}_id：added tokens 里没有 {', '.join(candidates)}。"
        f"请显式传 {role}_token=..."
    )
