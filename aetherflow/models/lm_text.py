"""字节级 tokenizer 与最小 chat 模板（审计 AF-34 的产品层第一块）。

这两件事的共同点是：**它们决定 id 与文本的双射**。一旦与训练时的约定不一致，
模型不会报错，只会输出错位的文本——所以判据只能是 roundtrip，而不是"看起来
差不多"。

- 词表 = 256 个字节 + 两个特殊 token（``<pad>`` / ``<eos>``），没有 BPE 合并表。
  因此 ``decode(encode(s)) == s`` 对任意 UTF-8 文本成立；代价是任何语言都按
  字节计长度，一个汉字是 3 个 token。
- chat 模板是**纯文本拼接**，不引入新的特殊 token。新增特殊 token 会改词表，
  让旧 checkpoint 的嵌入表整体错位；模板只负责"把消息排成一段文本"，不负责
  截断。工具调用的受约束格式在 ``lm_tools.py`` 里——那需要语法层（哪些字节
  在当前位置合法），与"怎么把消息拼成文本"是两件事。

两者都不依赖 numpy 或图：它们是数据层，能在任何后端上先被验收。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .lm import LMSpec

__all__ = ["ByteTokenizer", "ChatTemplate", "BYTE_OFFSET"]

# 与 ``LMSpec`` 的默认约定一致：0 是 padding、1 是 EOS，字节从 2 开始。
# ``BYTE_OFFSET`` 是公开的：任何"从字节到 token"的映射（受约束解码就是）
# 都要用它，不该各自复制一份常量。
PAD_ID = 0
EOS_ID = 1
BYTE_OFFSET = 2
_BYTE_OFFSET = BYTE_OFFSET
_BYTE_COUNT = 256


@dataclass(frozen=True)
class ByteTokenizer:
    """字节级词表：``<pad>=0``、``<eos>=1``、字节 ``b`` 的 id 是 ``2 + b``。"""

    pad: str = "<pad>"
    eos: str = "<eos>"

    def __post_init__(self) -> None:
        if not self.pad or not self.eos:
            raise ValueError("pad / eos 的名字不能为空")
        if self.pad == self.eos:
            raise ValueError(f"pad 与 eos 不能同名: {self.pad!r}")

    @property
    def vocab_size(self) -> int:
        return _BYTE_OFFSET + _BYTE_COUNT

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    def encode(self, text: str) -> List[int]:
        """按 UTF-8 字节编码；不做规整化，也不丢弃任何字符。"""
        return [_BYTE_OFFSET + byte for byte in str(text).encode("utf-8")]

    def decode(self, ids: Sequence[int]) -> str:
        """跳过 pad、在第一个 eos 处停止；其余 id 必须落在字节区间内。

        ``errors="strict"`` 是刻意的：解出非法 UTF-8 说明 id 序列被拼错或
        来自另一份词表，这时给一段乱码比给一句"编码不完整"更有害。
        """
        raw = bytearray()
        for token in ids:
            value = int(token)
            if value == PAD_ID:
                continue
            if value == EOS_ID:
                break
            if not _BYTE_OFFSET <= value < self.vocab_size:
                raise ValueError(
                    f"token {value} 不在词表内（合法字节区间是 "
                    f"[{_BYTE_OFFSET}, {self.vocab_size})）"
                )
            raw.append(value - _BYTE_OFFSET)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "这串 id 不是一个完整的 UTF-8 序列（可能被截断或拼错）："
                f"{exc}"
            ) from exc

    def spec(self, **overrides) -> LMSpec:
        """给出与这份词表对齐的 ``LMSpec``；词表大小与 pad/eos 不允许覆盖。"""
        for key in ("vocab_size", "pad_id", "eos_id"):
            if key in overrides:
                raise ValueError(f"spec() 不接受覆盖 {key}：它由词表约定决定")
        return LMSpec(
            vocab_size=self.vocab_size,
            pad_id=self.pad_id,
            eos_id=self.eos_id,
            **overrides,
        )


_ROLE_PREFIX = (
    ("system", "<|system|>\n"),
    ("user", "<|user|>\n"),
    ("assistant", "<|assistant|>\n"),
)


@dataclass(frozen=True)
class ChatTemplate:
    """把 ``[(role, content), ...]`` 拼成一段纯文本。

    ``add_generation_prompt=True`` 时在末尾补一个 assistant 前缀——这正是
    prefill 的输入，模型从这里续写。模板不校验轮次是否交替（那属于具体产品
    的会话规则），只拒绝空内容与未知 role。
    """

    prefixes: Tuple[Tuple[str, str], ...] = _ROLE_PREFIX
    turn_suffix: str = "\n"

    def __post_init__(self) -> None:
        roles = [role for role, _ in self.prefixes]
        if len(set(roles)) != len(roles):
            raise ValueError(f"role 前缀重复: {roles}")
        if not self.turn_suffix:
            raise ValueError("turn_suffix 不能为空")

    @property
    def roles(self) -> Tuple[str, ...]:
        return tuple(role for role, _ in self.prefixes)

    def _prefix(self, role: str) -> str:
        for name, prefix in self.prefixes:
            if name == role:
                return prefix
        raise ValueError(f"未知 role {role!r}，支持的是 {self.roles}")

    def render(
        self,
        messages: Sequence[Tuple[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> str:
        if not messages:
            raise ValueError("messages 不能为空")
        chunks: List[str] = []
        for role, content in messages:
            prefix = self._prefix(role)
            text = str(content)
            if not text:
                raise ValueError(f"{role} 的内容不能为空")
            chunks.append(prefix + text + self.turn_suffix)
        if add_generation_prompt:
            chunks.append(self._prefix("assistant"))
        return "".join(chunks)
