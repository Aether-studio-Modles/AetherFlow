"""GGUF 权重读取：llama.cpp 生态的容器格式与量化块解码。

为什么要自己读：与 ``weights.py`` 同一条理由——运行期只依赖 numpy。而且 GGUF
跟 safetensors 的差别不在"又一个容器"，在于**张量本身不是浮点数**：一个
``Q4_K`` 张量存的是 256 个元素一组的块（缩放、最小值和 4 bit 码点），不按块
解码出来的只是一串噪声。这里把容器解析与块解码放在一起，判据是"解出来能逐元素
对上"，不是"能读出个数组"。

格式（v2 / v3，均小端）：

- 头：magic ``GGUF`` + version(u32) + tensor_count(u64) + metadata_kv_count(u64)；
- metadata：key(string) + value_type(u32) + value；value 可以是标量、字符串或
  同类型数组；
- tensor info：name + n_dims(u32) + dims(u64 × n_dims) + type(u32) + offset(u64)；
  ``offset`` 相对数据区；
- 数据区起点 = 头结束位置按 ``general.alignment`` 向上取整（缺省 32）。

**维度顺序**：ggml 的 ``ne[0]`` 是最内层（变化最快）的维，numpy 的最内层是最后
一维，所以 ``info.shape`` 存的是 ``dims[::-1]``，``tensor()`` 直接按它 reshape。

**只做能逐元素对上的解码。** F32 / F16 / BF16 / Q4_0 / Q4_1 / Q5_0 / Q5_1 /
Q8_0 / Q8_1 / Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / Q8_K 解成 float32（fp8 不在
GGUF 里）；I8 / I16 / I32 / I64 原样返回整数；F64 原样返回。IQ 系（格点码本）、
TQ 系（三值）与 v1 头显式拒绝——不认识的码本"猜一个"会安静地给出错权重，
比读不出来更糟。

**先校验后读。** 头部长度、字符串长度、数组元素数、张量字节范围全部在切片之前
检查；解析失败的错误里带具体字段（哪个张量、哪个偏移、期望多少字节）。
"""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

__all__ = [
    "GGUFError",
    "GGUFTensorInfo",
    "GGUFFile",
    "read_gguf",
    "GGML_TYPES",
]

_MAGIC = 0x46554747  # "GGUF" 小端
_SUPPORTED_VERSIONS = (2, 3)

# 头 / 元数据里各类计的防爆上限。正常模型 metadata 只有几十项，张量数最多几千；
# 一个声称"有 40 亿个张量"的文件只可能是在骗读取方先分配内存。
_MAX_TENSORS = 1 << 20
_MAX_KV = 1 << 20
_MAX_STRING_BYTES = 64 << 20
_MAX_ARRAY_ELEMENTS = 1 << 28

# GGUF 的元数据值类型（键 → struct 格式）。STRING / ARRAY 单独处理。
_VALUE_TYPES = {
    0: ("UINT8", "<B", 1),
    1: ("INT8", "<b", 1),
    2: ("UINT16", "<H", 2),
    3: ("INT16", "<h", 2),
    4: ("UINT32", "<I", 4),
    5: ("INT32", "<i", 4),
    6: ("FLOAT32", "<f", 4),
    7: ("BOOL", "<B", 1),
    10: ("UINT64", "<Q", 8),
    11: ("INT64", "<q", 8),
    12: ("FLOAT64", "<d", 8),
}
_VALUE_STRING = 8
_VALUE_ARRAY = 9

# ggml 张量类型：名字 → (type id, block_size, block_bytes)。block_size=1 表示
# 逐元素存储，block_bytes 就是单个元素字节数。只列能逐元素解出来的那些。
GGML_TYPES: Dict[str, Tuple[int, int, int]] = {
    "F32": (0, 1, 4),
    "F16": (1, 1, 2),
    "Q4_0": (2, 32, 18),
    "Q4_1": (3, 32, 20),
    "Q5_0": (6, 32, 22),
    "Q5_1": (7, 32, 24),
    "Q8_0": (8, 32, 34),
    "Q8_1": (9, 32, 36),
    "Q2_K": (10, 256, 84),
    "Q3_K": (11, 256, 110),
    "Q4_K": (12, 256, 144),
    "Q5_K": (13, 256, 176),
    "Q6_K": (14, 256, 210),
    "Q8_K": (15, 256, 292),
    "I8": (24, 1, 1),
    "I16": (25, 1, 2),
    "I32": (26, 1, 4),
    "I64": (27, 1, 8),
    "F64": (28, 1, 8),
    "BF16": (30, 1, 2),
}

_BY_ID: Dict[int, str] = {spec[0]: name for name, spec in GGML_TYPES.items()}


class GGUFError(ValueError):
    """GGUF 文件不可读、不合规，或用了本实现不支持的块类型。"""


# ---------------------------------------------------------------------------
# 块结构：与 ggml 的 block_* 布局逐字节一致（小端）。
# ---------------------------------------------------------------------------

_BLOCK_LAYOUT: Dict[str, List[Tuple[str, object]]] = {
    "Q4_0": [("d", "<f2"), ("qs", ("u1", 16))],
    "Q4_1": [("d", "<f2"), ("m", "<f2"), ("qs", ("u1", 16))],
    "Q5_0": [("d", "<f2"), ("qh", ("u1", 4)), ("qs", ("u1", 16))],
    "Q5_1": [("d", "<f2"), ("m", "<f2"), ("qh", ("u1", 4)), ("qs", ("u1", 16))],
    "Q8_0": [("d", "<f2"), ("qs", ("i1", 32))],
    "Q8_1": [("d", "<f2"), ("s", "<f2"), ("qs", ("i1", 32))],
    "Q2_K": [("scales", ("u1", 16)), ("qs", ("u1", 64)), ("d", "<f2"), ("dmin", "<f2")],
    "Q3_K": [("hmask", ("u1", 32)), ("qs", ("u1", 64)), ("scales", ("u1", 12)), ("d", "<f2")],
    "Q4_K": [("d", "<f2"), ("dmin", "<f2"), ("scales", ("u1", 12)), ("qs", ("u1", 128))],
    "Q5_K": [
        ("d", "<f2"),
        ("dmin", "<f2"),
        ("scales", ("u1", 12)),
        ("qh", ("u1", 32)),
        ("qs", ("u1", 128)),
    ],
    "Q6_K": [("ql", ("u1", 128)), ("qh", ("u1", 64)), ("scales", ("i1", 16)), ("d", "<f2")],
    "Q8_K": [("d", "<f4"), ("qs", ("i1", 256)), ("bsums", ("<i2", 16))],
}

_BLOCK_DTYPE: Dict[str, np.dtype] = {
    name: np.dtype([(fname, ftype) if isinstance(ftype, str) else (fname, ftype[0], (ftype[1],))
                    for fname, ftype in layout])
    for name, layout in _BLOCK_LAYOUT.items()
}

for _name, _spec in GGML_TYPES.items():
    if _name in _BLOCK_DTYPE:
        _actual = _BLOCK_DTYPE[_name].itemsize
        if _actual != _spec[2]:
            raise AssertionError(f"{_name} 块结构 {_actual} 字节与类型表 {_spec[2]} 不一致")


def _f16(blocks: np.ndarray, name: str) -> np.ndarray:
    """半精度字段 → (块数, 1) 的 float32，便于广播。"""
    return blocks[name].astype(np.float32)[:, None]


def _dq_q4_0(raw: bytes, count: int) -> np.ndarray:
    b = np.frombuffer(raw, dtype=_BLOCK_DTYPE["Q4_0"], count=count)
    q = b["qs"].astype(np.int32)
    out = np.empty((count, 32), dtype=np.float32)
    out[:, :16] = (q & 0xF) - 8
    out[:, 16:] = (q >> 4) - 8
    return (out * _f16(b, "d")).reshape(-1)


def _dq_q4_1(raw: bytes, count: int) -> np.ndarray:
    b = np.frombuffer(raw, dtype=_BLOCK_DTYPE["Q4_1"], count=count)
    q = b["qs"].astype(np.int32)
    out = np.empty((count, 32), dtype=np.float32)
    out[:, :16] = q & 0xF
    out[:, 16:] = q >> 4
    return (out * _f16(b, "d") + _f16(b, "m")).reshape(-1)


def _qh_bits(blocks: np.ndarray) -> np.ndarray:
    """Q5_0 / Q5_1 的 4 字节 ``qh`` → 32 个低位标志的 ``(块数, 16)``。"""
    bytes_ = blocks["qh"].astype(np.uint32)
    word = (
        bytes_[:, 0]
        | (bytes_[:, 1] << 8)
        | (bytes_[:, 2] << 16)
        | (bytes_[:, 3] << 24)
    )
    low = np.arange(16, dtype=np.uint32)
    x0 = ((word[:, None] >> low) & 1).astype(np.int32) << 4
    x1 = ((word[:, None] >> (low + 16)) & 1).astype(np.int32) << 4
    return np.concatenate([x0, x1], axis=1)  # (块数, 32)


def _q5_codes(blocks: np.ndarray) -> np.ndarray:
    q = blocks["qs"].astype(np.int32)
    bits = _qh_bits(blocks)
    low = (q & 0xF) | bits[:, :16]
    high = (q >> 4) | bits[:, 16:]
    return np.concatenate([low, high], axis=1)  # 前 16 是低 4 位，后 16 是高 4 位


def _dq_q5_0(raw: bytes, count: int) -> np.ndarray:
    b = np.frombuffer(raw, dtype=_BLOCK_DTYPE["Q5_0"], count=count)
    return (_q5_codes(b).astype(np.float32) * _f16(b, "d")).reshape(-1)


def _dq_q5_1(raw: bytes, count: int) -> np.ndarray:
    b = np.frombuffer(raw, dtype=_BLOCK_DTYPE["Q5_1"], count=count)
    return (_q5_codes(b).astype(np.float32) * _f16(b, "d") + _f16(b, "m")).reshape(-1)


def _dq_q8_0(raw: bytes, count: int) -> np.ndarray:
    b = np.frombuffer(raw, dtype=_BLOCK_DTYPE["Q8_0"], count=count)
    return (b["qs"].astype(np.float32) * _f16(b, "d")).reshape(-1)


def _dq_q8_1(raw: bytes, count: int) -> np.ndarray:
    b = np.frombuffer(raw, dtype=_BLOCK_DTYPE["Q8_1"], count=count)
    return (b["qs"].astype(np.float32) * _f16(b, "d")).reshape(-1)


def _dq_q8_k(raw: bytes, count: int) -> np.ndarray:
    b = np.frombuffer(raw, dtype=_BLOCK_DTYPE["Q8_K"], count=count)
    return (b["qs"].astype(np.float32) * b["d"].astype(np.float32)[:, None]).reshape(-1)


def _k4_scale_min(scales: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Q4_K / Q5_K 的 12 字节 6 bit 缩放与最小值 → 两组各 8 个 (缩放, 最小值)。"""
    q = scales.astype(np.int32)
    counts = q.shape[0]
    ds = np.empty((counts, 8), dtype=np.int32)
    ms = np.empty((counts, 8), dtype=np.int32)
    for j in range(4):
        ds[:, j] = q[:, j] & 63
        ms[:, j] = q[:, j + 4] & 63
    for j in range(4, 8):
        ds[:, j] = (q[:, j + 4] & 0xF) | ((q[:, j - 4] >> 6) << 4)
        ms[:, j] = (q[:, j + 4] >> 4) | ((q[:, j] >> 6) << 4)
    return ds, ms


def _dq_q4_k(raw: bytes, count: int) -> np.ndarray:
    b = np.frombuffer(raw, dtype=_BLOCK_DTYPE["Q4_K"], count=count)
    ds, ms = _k4_scale_min(b["scales"])
    q = b["qs"].astype(np.int32)
    nib = np.stack([q & 0xF, (q >> 4) & 0xF], axis=2)  # (块数, 128, 2)
    vals = nib.reshape(count, 4, 32, 2).transpose(0, 1, 3, 2).reshape(count, 256)
    scale = np.repeat(ds, 32, axis=1).astype(np.float32)
    minimum = np.repeat(ms, 32, axis=1).astype(np.float32)
    return (vals.astype(np.float32) * scale * _f16(b, "d") - minimum * _f16(b, "dmin")).reshape(-1)


def _dq_q5_k(raw: bytes, count: int) -> np.ndarray:
    b = np.frombuffer(raw, dtype=_BLOCK_DTYPE["Q5_K"], count=count)
    ds, ms = _k4_scale_min(b["scales"])
    q = b["qs"].astype(np.int32)
    qh = b["qh"].astype(np.int32)
    low = np.empty((count, 4, 32), dtype=np.int32)
    high = np.empty((count, 4, 32), dtype=np.int32)
    for g in range(4):
        group = q[:, g * 32 : (g + 1) * 32]
        u1 = 1 << (2 * g)
        u2 = 1 << (2 * g + 1)
        low[:, g] = (group & 0xF) + np.where((qh & u1) != 0, 16, 0)
        high[:, g] = (group >> 4) + np.where((qh & u2) != 0, 16, 0)
    vals = np.stack([low, high], axis=2).transpose(0, 1, 3, 2).reshape(count, 256)
    scale = np.repeat(ds, 32, axis=1).astype(np.float32)
    minimum = np.repeat(ms, 32, axis=1).astype(np.float32)
    return (vals.astype(np.float32) * scale * _f16(b, "d") - minimum * _f16(b, "dmin")).reshape(-1)


def _q3_scales(raw: np.ndarray) -> np.ndarray:
    """Q3_K 的 12 字节 6 bit 缩放 → (块数, 16) 的 int32。"""
    counts = raw.shape[0]
    aux = np.zeros((counts, 4), dtype=np.uint32)
    for word in range(3):
        for byte in range(4):
            aux[:, word] |= raw[:, 4 * word + byte].astype(np.uint32) << (8 * byte)
    tmp = aux[:, 2].copy()
    k1 = np.uint32(0x03030303)
    k2 = np.uint32(0x0F0F0F0F)
    a0 = (aux[:, 0] & k2) | (((tmp >> np.uint32(0)) & k1) << np.uint32(4))
    a1 = (aux[:, 1] & k2) | (((tmp >> np.uint32(2)) & k1) << np.uint32(4))
    a2 = ((aux[:, 0] >> np.uint32(4)) & k2) | (((tmp >> np.uint32(4)) & k1) << np.uint32(4))
    a3 = ((aux[:, 1] >> np.uint32(4)) & k2) | (((tmp >> np.uint32(6)) & k1) << np.uint32(4))
    packed = np.stack([a0, a1, a2, a3], axis=1)  # (块数, 4)
    bytes_ = np.stack([(packed >> np.uint32(8 * k)) & 0xFF for k in range(4)], axis=2)
    signed = bytes_.reshape(counts, 16).astype(np.int32)
    return np.where(signed >= 128, signed - 256, signed)


def _dq_q2_k(raw: bytes, count: int) -> np.ndarray:
    b = np.frombuffer(raw, dtype=_BLOCK_DTYPE["Q2_K"], count=count)
    sc = b["scales"].astype(np.int32)
    q = b["qs"].astype(np.int32)
    d = _f16(b, "d")
    dmin = _f16(b, "dmin")
    out = np.empty((count, 256), dtype=np.float32)
    for n in range(2):
        qn = q[:, n * 32 : (n + 1) * 32]
        for j in range(4):
            shift = 2 * j
            idx = n * 8 + 2 * j
            d0 = d * (sc[:, idx] & 0xF)[:, None]
            m0 = dmin * (sc[:, idx] >> 4)[:, None]
            d1 = d * (sc[:, idx + 1] & 0xF)[:, None]
            m1 = dmin * (sc[:, idx + 1] >> 4)[:, None]
            base = n * 128 + j * 32
            out[:, base : base + 16] = ((qn[:, :16] >> shift) & 3) * d0 - m0
            out[:, base + 16 : base + 32] = ((qn[:, 16:] >> shift) & 3) * d1 - m1
    return out.reshape(-1)


def _dq_q3_k(raw: bytes, count: int) -> np.ndarray:
    b = np.frombuffer(raw, dtype=_BLOCK_DTYPE["Q3_K"], count=count)
    hm = b["hmask"].astype(np.int32)
    q = b["qs"].astype(np.int32)
    sc = _q3_scales(b["scales"])
    d = _f16(b, "d")
    out = np.empty((count, 256), dtype=np.float32)
    for n in range(2):
        qn = q[:, n * 32 : (n + 1) * 32]
        for j in range(4):
            shift = 2 * j
            mask = 1 << (4 * n + j)
            base = n * 128 + j * 32
            dl = d * (sc[:, 2 * (4 * n + j)] - 32)[:, None]
            dl2 = d * (sc[:, 2 * (4 * n + j) + 1] - 32)[:, None]
            low = ((qn[:, :16] >> shift) & 3) - np.where((hm[:, :16] & mask) != 0, 0, 4)
            high = ((qn[:, 16:] >> shift) & 3) - np.where((hm[:, 16:] & mask) != 0, 0, 4)
            out[:, base : base + 16] = low * dl
            out[:, base + 16 : base + 32] = high * dl2
    return out.reshape(-1)


def _dq_q6_k(raw: bytes, count: int) -> np.ndarray:
    b = np.frombuffer(raw, dtype=_BLOCK_DTYPE["Q6_K"], count=count)
    ql = b["ql"].astype(np.int32)
    qh = b["qh"].astype(np.int32)
    sc = b["scales"].astype(np.int32)
    d = _f16(b, "d")
    half = np.arange(32) // 16  # 每 16 个元素换一个缩放
    out = np.empty((count, 256), dtype=np.float32)
    for n in range(2):
        ql_n = ql[:, n * 64 : (n + 1) * 64]
        qh_n = qh[:, n * 32 : (n + 1) * 32]
        sc_n = sc[:, n * 8 : (n + 1) * 8]
        q1 = ((ql_n[:, :32] & 0xF) | (((qh_n >> 0) & 3) << 4)) - 32
        q2 = ((ql_n[:, 32:] & 0xF) | (((qh_n >> 2) & 3) << 4)) - 32
        q3 = ((ql_n[:, :32] >> 4) | (((qh_n >> 4) & 3) << 4)) - 32
        q4 = ((ql_n[:, 32:] >> 4) | (((qh_n >> 6) & 3) << 4)) - 32
        base = n * 128
        out[:, base : base + 32] = q1 * sc_n[:, half] * d
        out[:, base + 32 : base + 64] = q2 * sc_n[:, half + 2] * d
        out[:, base + 64 : base + 96] = q3 * sc_n[:, half + 4] * d
        out[:, base + 96 : base + 128] = q4 * sc_n[:, half + 6] * d
    return out.reshape(-1)


_DEQUANT = {
    "Q4_0": _dq_q4_0,
    "Q4_1": _dq_q4_1,
    "Q5_0": _dq_q5_0,
    "Q5_1": _dq_q5_1,
    "Q8_0": _dq_q8_0,
    "Q8_1": _dq_q8_1,
    "Q2_K": _dq_q2_k,
    "Q3_K": _dq_q3_k,
    "Q4_K": _dq_q4_k,
    "Q5_K": _dq_q5_k,
    "Q6_K": _dq_q6_k,
    "Q8_K": _dq_q8_k,
}


# ---------------------------------------------------------------------------
# 容器解析
# ---------------------------------------------------------------------------


class _Reader:
    """带边界检查的顺序读取器；任何越界都立刻报错，而不是让 numpy 猜。"""

    def __init__(self, data: Union[bytes, mmap.mmap], limit: int) -> None:
        self.data = data
        self.pos = 0
        self.limit = limit

    @property
    def remaining(self) -> int:
        return self.limit - self.pos

    def take(self, n: int, what: str) -> bytes:
        if n < 0:
            raise GGUFError(f"{what} 的长度为负：{n}")
        if n > self.remaining:
            raise GGUFError(
                f"{what} 越过文件末尾：需要 {n} 字节，从偏移 {self.pos} 起只剩 "
                f"{self.remaining} 字节"
            )
        start = self.pos
        self.pos += n
        return bytes(self.data[start : start + n])

    def scalar(self, fmt: str, what: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.take(size, what))[0]

    def u32(self, what: str) -> int:
        return self.scalar("<I", what)

    def u64(self, what: str) -> int:
        return self.scalar("<Q", what)

    def string(self, what: str) -> str:
        length = self.u64(f"{what} 的长度")
        if length > _MAX_STRING_BYTES:
            raise GGUFError(f"{what} 声称有 {length} 字节，超过上限 {_MAX_STRING_BYTES}")
        raw = self.take(length, what)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFError(f"{what} 不是合法 UTF-8：{exc}") from exc

    def value(self, value_type: int, what: str, depth: int = 0):
        if value_type == _VALUE_STRING:
            return self.string(what)
        if value_type == _VALUE_ARRAY:
            if depth >= 4:
                raise GGUFError(f"{what} 的数组嵌套超过 4 层：本实现不接受嵌套数组")
            element_type = self.u32(f"{what} 的元素类型")
            count = self.u64(f"{what} 的元素数")
            if element_type == _VALUE_ARRAY:
                raise GGUFError(f"{what} 是二维数组：GGUF 的元数据数组只允许一维")
            if count > _MAX_ARRAY_ELEMENTS:
                raise GGUFError(f"{what} 声称有 {count} 个元素，超过上限 {_MAX_ARRAY_ELEMENTS}")
            if element_type in _VALUE_TYPES:
                size = _VALUE_TYPES[element_type][2]
                if count * size > self.remaining:
                    raise GGUFError(
                        f"{what} 需要 {count}×{size} 字节，越过文件末尾"
                    )
            elif element_type != _VALUE_STRING:
                raise GGUFError(f"{what} 的元素类型 {element_type} 不支持")
            return [self.value(element_type, f"{what}[{i}]", depth + 1) for i in range(count)]
        spec = _VALUE_TYPES.get(value_type)
        if spec is None:
            raise GGUFError(f"{what} 的元数据值类型 {value_type} 不受支持")
        raw = self.scalar(spec[1], what)
        return bool(raw) if spec[0] == "BOOL" else raw


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class GGUFTensorInfo:
    """一个张量的头信息；``shape`` 已换成 numpy 的维序（最内层在最后）。"""

    name: str
    dims: Tuple[int, ...]
    shape: Tuple[int, ...]
    type_name: str
    type_id: int
    offset: int
    n_elements: int
    n_bytes: int


@dataclass
class GGUFFile:
    """解析好的 GGUF 文件；``tensor()`` 按需解码，不预先展开整份权重。"""

    path: Optional[str]
    version: int
    metadata: Dict[str, object]
    tensors: Dict[str, GGUFTensorInfo]
    alignment: int
    data_offset: int
    file_size: int
    _buffer: Union[bytes, mmap.mmap] = field(repr=False, default=b"")
    _closed: bool = field(repr=False, default=False)

    def __len__(self) -> int:
        return len(self.tensors)

    def __contains__(self, name: object) -> bool:
        return name in self.tensors

    def __enter__(self) -> "GGUFFile":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if isinstance(self._buffer, mmap.mmap):
            self._buffer.close()
        self._buffer = b""

    def tensor_info(self, name: str) -> GGUFTensorInfo:
        info = self.tensors.get(name)
        if info is None:
            raise GGUFError(f"文件里没有张量 {name!r}（共有 {len(self.tensors)} 个）")
        return info

    def tensor(self, name: str) -> np.ndarray:
        """解码一个张量。

        量化类型 / F16 / BF16 统一返回 float32；I8 / I16 / I32 / I64 与 F64
        保持原生 dtype。返回的是新数组，与文件缓冲区不共享内存。
        """
        if self._closed:
            raise GGUFError("GGUFFile 已关闭，不能再读张量")
        info = self.tensor_info(name)
        start = self.data_offset + info.offset
        end = start + info.n_bytes
        raw = self._buffer[start:end]
        if len(raw) != info.n_bytes:  # 解析期已校验，这里只防缓冲区被提前释放
            raise GGUFError(
                f"张量 {name!r} 需要 {info.n_bytes} 字节，实际只取到 {len(raw)} 字节"
            )
        return _decode_tensor(info, bytes(raw))

    def names(self) -> List[str]:
        return list(self.tensors)


def _decode_tensor(info: GGUFTensorInfo, raw: bytes) -> np.ndarray:
    name = info.type_name
    block_size = GGML_TYPES[name][1]
    count = info.n_elements // block_size
    if name == "F32":
        array = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    elif name == "F16":
        array = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif name == "BF16":
        # bf16 就是 f32 的高 16 位：左移 16 位后按 f32 位模式解释，是精确的。
        wide = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        array = wide.view(np.float32)
    elif name == "F64":
        return np.frombuffer(raw, dtype="<f8").astype(np.float64).reshape(info.shape)
    elif name == "I8":
        array = np.frombuffer(raw, dtype="i1")
    elif name == "I16":
        array = np.frombuffer(raw, dtype="<i2")
    elif name == "I32":
        array = np.frombuffer(raw, dtype="<i4")
    elif name == "I64":
        array = np.frombuffer(raw, dtype="<i8")
    else:
        dequant = _DEQUANT.get(name)
        if dequant is None:  # 解析期已拒绝，这里防御性兜底
            raise GGUFError(f"张量类型 {name} 没有解码器")
        array = dequant(raw, count)
    if array.size != info.n_elements:
        raise GGUFError(
            f"张量 {info.name!r} 解码出 {array.size} 个元素，头部声明 {info.n_elements} 个"
        )
    return array.reshape(info.shape)


def read_gguf(source: Union[str, os.PathLike, bytes, bytearray]) -> GGUFFile:
    """读一个 GGUF 文件（路径或内存字节）。

    路径用 ``mmap`` 打开，``tensor()`` 按需解码，不把整份权重读进内存；用完
    调 ``close()``，或写成 ``with read_gguf(path) as f``。
    """
    if isinstance(source, (bytes, bytearray)):
        buffer: Union[bytes, mmap.mmap] = bytes(source)
        path = None
    else:
        path = os.fspath(source)
        try:
            handle = open(path, "rb")
        except OSError as exc:
            raise GGUFError(f"打不开 GGUF 文件 {path!r}：{exc}") from exc
        try:
            size = os.fstat(handle.fileno()).st_size
            if size < 24:
                raise GGUFError(f"文件 {path!r} 只有 {size} 字节，连 GGUF 头都放不下")
            buffer = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        except OSError as exc:
            raise GGUFError(f"映射 GGUF 文件 {path!r} 失败：{exc}") from exc
        finally:
            handle.close()
    size = len(buffer)

    reader = _Reader(buffer, size)
    magic = reader.u32("magic")
    if magic != _MAGIC:
        raise GGUFError(f"不是 GGUF 文件：magic=0x{magic:08X}，期望 0x{_MAGIC:08X}")
    version = reader.u32("version")
    if version not in _SUPPORTED_VERSIONS:
        raise GGUFError(f"GGUF 版本 {version} 不受支持：只读 v2 / v3")
    tensor_count = reader.u64("tensor_count")
    kv_count = reader.u64("metadata_kv_count")
    if tensor_count > _MAX_TENSORS:
        raise GGUFError(f"tensor_count={tensor_count} 超过上限 {_MAX_TENSORS}")
    if kv_count > _MAX_KV:
        raise GGUFError(f"metadata_kv_count={kv_count} 超过上限 {_MAX_KV}")
    # 每条 KV 至少 8 字节 key 长度 + 4 字节类型；每个张量头至少 24 字节。
    if kv_count * 12 > reader.remaining:
        raise GGUFError(
            f"metadata_kv_count={kv_count} 与剩余 {reader.remaining} 字节对不上"
        )

    metadata: Dict[str, object] = {}
    for _ in range(kv_count):
        key = reader.string("metadata key")
        if key in metadata:
            raise GGUFError(f"metadata key {key!r} 重复出现")
        value_type = reader.u32(f"metadata[{key!r}] 的类型")
        metadata[key] = reader.value(value_type, f"metadata[{key!r}]")

    alignment = 32
    raw_alignment = metadata.get("general.alignment")
    if raw_alignment is not None:
        if not isinstance(raw_alignment, int) or raw_alignment <= 0:
            raise GGUFError(f"general.alignment={raw_alignment!r} 必须是正整数")
        if raw_alignment & (raw_alignment - 1):
            raise GGUFError(f"general.alignment={raw_alignment} 不是 2 的幂")
        alignment = raw_alignment

    infos: Dict[str, GGUFTensorInfo] = {}
    if tensor_count * 24 > reader.remaining:
        raise GGUFError(f"tensor_count={tensor_count} 与剩余 {reader.remaining} 字节对不上")
    for index in range(tensor_count):
        try:
            name = reader.string(f"tensor[{index}] 的名字")
            if name in infos:
                raise GGUFError(f"张量名 {name!r} 重复")
            n_dims = reader.u32(f"tensor[{name!r}] 的维数")
            if not 1 <= n_dims <= 4:
                raise GGUFError(f"张量 {name!r} 的维数 {n_dims} 不在 1..4：ggml 最多四维")
            dims = tuple(reader.u64(f"tensor[{name!r}].ne[{d}]") for d in range(n_dims))
            if any(d == 0 for d in dims):
                raise GGUFError(f"张量 {name!r} 的维度里含 0：{dims}")
            type_id = reader.u32(f"tensor[{name!r}] 的类型")
            type_name = _BY_ID.get(type_id)
            if type_name is None:
                raise GGUFError(
                    f"张量 {name!r} 用了类型 id={type_id}：IQ 系格点码本 / TQ 系三值 / "
                    "分块向量布局本实现不解码，拒绝而不是猜"
                )
            offset = reader.u64(f"tensor[{name!r}] 的偏移")
        except GGUFError as exc:
            # GGUF 没有张量表长度字段，tensor_count 说谎时只能在解析途中发现。
            # 这里补上"第几个、声明了几个"，否则现场只看到一条莫名其妙的维度错误。
            raise GGUFError(
                f"第 {index + 1} 个张量信息解析失败（头里声明 {tensor_count} 个，"
                f"与文件内容对不上）：{exc}"
            ) from exc
        n_elements = 1
        for d in dims:
            n_elements *= d
        block_size, block_bytes = GGML_TYPES[type_name][1], GGML_TYPES[type_name][2]
        if n_elements % block_size != 0:
            raise GGUFError(
                f"张量 {name!r} 有 {n_elements} 个元素，不是 {type_name} 块大小 "
                f"{block_size} 的整数倍"
            )
        n_bytes = n_elements // block_size * block_bytes
        if offset % alignment != 0:
            raise GGUFError(
                f"张量 {name!r} 的偏移 {offset} 没有按 alignment={alignment} 对齐"
            )
        infos[name] = GGUFTensorInfo(
            name=name,
            dims=dims,
            shape=tuple(reversed(dims)),
            type_name=type_name,
            type_id=type_id,
            offset=offset,
            n_elements=n_elements,
            n_bytes=n_bytes,
        )

    data_offset = _align_up(reader.pos, alignment)
    if data_offset > size:
        raise GGUFError(f"数据区起点 {data_offset} 越过文件末尾 {size}")
    for info in infos.values():
        end = data_offset + info.offset + info.n_bytes
        if end > size:
            raise GGUFError(
                f"张量 {info.name!r} 的数据 [ {data_offset + info.offset}, {end} ) "
                f"越过文件末尾 {size}"
            )

    return GGUFFile(
        path=path,
        version=version,
        metadata=metadata,
        tensors=infos,
        alignment=alignment,
        data_offset=data_offset,
        file_size=size,
        _buffer=buffer,
    )
