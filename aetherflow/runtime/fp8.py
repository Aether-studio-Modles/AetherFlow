"""fp8 的数值语义与单字节存储编码。

numpy 没有 fp8 标量类型，参考语义与 bf16 一样借用 float32 作**载体**：每次
``f8e4m3`` / ``f8e5m2`` 边界都把数值量化到该格式能表示的格点上。本模块同时
给出 1 字节的位模式（``pack_fp8`` / ``unpack_fp8``），供读取 fp8 检查点这类
存储路径使用。

两种格式都是"1 位符号 + 指数 + 尾数"，区别在于指数与尾数各占几位：

- ``e4m3``：4 位指数（偏置 7）、3 位尾数。没有 inf——指数全 1 且尾数全 1 的
  位模式是 NaN，其余指数全 1 的位模式仍是正规值，最大有限值 448。输入超出
  这个范围时**饱和**到 ±448（与主流推理框架的 fp8 量化一致），NaN 保持 NaN。
- ``e5m2``：5 位指数（偏置 15）、2 位尾数。遵循 IEEE：指数全 1、尾数 0 是
  inf，尾数非 0 是 NaN；最大有限值 57344，溢出按 RNE 舍入到 ±inf。

舍入是 round-to-nearest-even（RNE）：先按数值所在的 binade 取 ulp 做
``rint``，若进位跨到下一个 binade（例如 7.5 个 ulp 进位到 8 个），再按新的
指数重算一次。次正规区的 ulp 固定在 ``2 ** (e_min - mantissa_bits)``，所以
最小可表示值分别是 ``2 ** -9``（e4m3）与 ``2 ** -16``（e5m2）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "round_fp8",
    "pack_fp8",
    "unpack_fp8",
    "quantize_fp8",
    "dequantize_fp8",
    "FP8_KINDS",
    "FP8_MAX",
    "STORAGE_BYTES",
]

#: fp8 的存储宽度（字节）。参考后端的载体是 f32，打包表示才是 1 字节。
STORAGE_BYTES = 1

FP8_KINDS = ("e4m3", "e5m2")


@dataclass(frozen=True)
class _Format:
    kind: str
    mantissa_bits: int
    exp_bits: int
    bias: int
    e_min: int
    e_max: int
    max_value: float
    #: 超出最大有限值时是饱和（e4m3）还是舍入到 inf（e5m2）。
    saturate: bool

    @property
    def exp_all(self) -> int:
        return (1 << self.exp_bits) - 1

    @property
    def nan_code(self) -> int:
        # 两种格式的 0x7F 都是 NaN：e4m3 是指数全 1 + 尾数全 1，
        # e5m2 是指数全 1 + 尾数非 0。
        return 0x7F

    @property
    def inf_code(self) -> int:
        # e4m3 没有 inf；这个值只在饱和分支不参与时兜底，按最大有限值处理。
        return 0x7E if self.kind == "e4m3" else (self.exp_all << self.mantissa_bits)


_E4M3 = _Format(
    kind="e4m3",
    mantissa_bits=3,
    exp_bits=4,
    bias=7,
    e_min=-6,
    e_max=8,
    max_value=448.0,
    saturate=True,
)
_E5M2 = _Format(
    kind="e5m2",
    mantissa_bits=2,
    exp_bits=5,
    bias=15,
    e_min=-14,
    e_max=15,
    max_value=57344.0,
    saturate=False,
)

_FORMATS = {"e4m3": _E4M3, "e5m2": _E5M2}

#: 每种格式的最大有限值；带 scale 的量化把它当作缩放后的值域端点。
FP8_MAX = {kind: fmt.max_value for kind, fmt in _FORMATS.items()}


def _format(kind: str) -> _Format:
    if kind not in _FORMATS:
        raise ValueError(f"未知的 fp8 格式 {kind!r}；支持 {FP8_KINDS}")
    return _FORMATS[kind]


def _floor_log2(value: np.ndarray) -> np.ndarray:
    """``floor(log2(x))``，要求 ``x > 0``（调用方保证）。"""
    _, exponent = np.frexp(np.asarray(value, dtype=np.float64))
    return exponent.astype(np.int64) - 1


def _round_abs(value: np.ndarray, fmt: _Format) -> np.ndarray:
    """把非负的 f32 值量化到 ``fmt`` 的格点（仍以 float64 返回）。

    0 / inf / NaN 原样返回，由编码阶段决定它们的位模式。
    """
    exact = np.asarray(value, dtype=np.float64)
    finite = np.isfinite(exact) & (exact > 0)
    safe = np.where(finite, exact, 1.0)

    exponent = np.maximum(_floor_log2(safe), fmt.e_min)
    ulp = np.ldexp(1.0, (exponent - fmt.mantissa_bits).astype(np.int64))
    first = np.rint(safe / ulp) * ulp

    # 进位可能跨 binade（7.5 ulp -> 8 ulp 会进入下一个指数），用进位后的
    # 指数再舍入一次。两次就够：第二次的 ulp 一定不小于真实格点。
    exponent2 = np.maximum(_floor_log2(np.where(first > 0, first, 1.0)), fmt.e_min)
    ulp2 = np.ldexp(1.0, (exponent2 - fmt.mantissa_bits).astype(np.int64))
    second = np.rint(safe / ulp2) * ulp2
    return np.where(finite, second, exact)


def _encode(value: np.ndarray, fmt: _Format) -> np.ndarray:
    """把已量化到格点的非负值编码成 8 位模式（不含符号位）。"""
    mantissa = fmt.mantissa_bits
    work = np.asarray(value, dtype=np.float64)
    if fmt.saturate:
        # e4m3 没有 inf：超出 448 一律饱和到最大有限值。
        work = np.where(np.isinf(work) | (work > fmt.max_value), fmt.max_value, work)
    # 非饱和格式（e5m2）保留 inf，让它落到下面的溢出分支。

    finite_positive = np.isfinite(work) & (work > 0)
    safe = np.where(finite_positive, work, 1.0)
    exponent = _floor_log2(safe)

    # 次正规：n = rint(x * 2^(M - e_min))；n == 2^M 进位成最小正规数。
    sub = finite_positive & (exponent < fmt.e_min)
    sub_raw = np.rint(np.where(sub, work, 0.0) * np.ldexp(1.0, mantissa - fmt.e_min))
    sub_carry = sub & (sub_raw >= float(1 << mantissa))
    sub_code = np.where(sub, sub_raw.astype(np.int64), 0)

    # 正规：尾数 = rint(x / 2^(e - M)) - 2^M；进位时指数加一、尾数清零。
    normal = finite_positive & ~sub
    normal_exp = np.maximum(exponent, fmt.e_min)
    raw = np.rint(safe / np.ldexp(1.0, (normal_exp - mantissa).astype(np.int64)))
    carry = normal & (raw >= float(1 << (mantissa + 1)))
    final_exp = np.where(carry, normal_exp + 1, normal_exp)
    mant = np.where(
        normal, np.where(carry, 0.0, raw - (1 << mantissa)), 0.0
    ).astype(np.int64)

    # e5m2 的溢出：指数位跑到全 1、尾数 0，正好是 inf。
    overflow = normal & (final_exp > fmt.e_max)
    normal_code = np.where(
        overflow,
        fmt.inf_code,
        ((final_exp + fmt.bias) << mantissa) | mant,
    )

    code = np.where(normal, normal_code, 0)
    code = np.where(sub & ~sub_carry, sub_code, code)
    code = np.where(sub_carry, 1 << mantissa, code)
    code = np.where(finite_positive, code, 0)

    # 特殊值单独覆盖：NaN、inf、零（-0 的符号位由调用方补）。
    code = np.where(np.isnan(value), fmt.nan_code, code)
    code = np.where(np.isinf(value), fmt.inf_code, code)
    code = np.where(value == 0, 0, code)
    return code.astype(np.uint8)


def pack_fp8(value: Any, kind: str = "e4m3") -> np.ndarray:
    """把 f32 值量化并编码成 1 字节位模式（``uint8``）。

    返回的是 fp8 的位模式，不是整数数值：``pack_fp8([1.0]) == [0x38]``
    （e4m3）。
    """
    fmt = _format(kind)
    array = np.array(value, dtype=np.float32, copy=True, order="C", subok=False)
    bits = array.view(np.uint32)
    negative = ((bits >> np.uint32(31)) & np.uint32(1)).astype(np.uint8)
    code = _encode(_round_abs(np.abs(array), fmt), fmt)
    return (code | (negative << np.uint8(7))).astype(np.uint8)


def unpack_fp8(bits: Any, kind: str = "e4m3") -> np.ndarray:
    """把 1 字节位模式解码成 f32 载体；fp8 -> f32 是精确的。"""
    fmt = _format(kind)
    code = np.asarray(bits)
    if code.dtype != np.uint8:
        raise TypeError(
            f"fp8 位模式数组必须是 uint8，得到 {code.dtype}；"
            "这里收的是位模式而不是整数数值"
        )
    mantissa = fmt.mantissa_bits
    exponent = ((code >> np.uint8(mantissa)) & np.uint8(fmt.exp_all)).astype(np.int64)
    mant = (code & np.uint8((1 << mantissa) - 1)).astype(np.int64)

    all_one = exponent == fmt.exp_all
    subnormal = exponent == 0
    regular = ~subnormal

    value = np.ldexp(1.0, np.where(regular, exponent - fmt.bias, 0))
    value = np.where(
        regular,
        value * (1.0 + mant / float(1 << mantissa)),
        mant * float(2.0 ** (fmt.e_min - mantissa)),
    )

    nan_mask = all_one & (
        mant == (1 << mantissa) - 1 if fmt.kind == "e4m3" else mant != 0
    )
    inf_mask = all_one & (mant == 0) & (fmt.kind != "e4m3")
    value = np.where(nan_mask, np.nan, value)
    value = np.where(inf_mask, np.inf, value)

    signed = np.where(((code >> np.uint8(7)) & np.uint8(1)) != 0, -value, value)
    return signed.astype(np.float32)


def round_fp8(value: Any, kind: str = "e4m3") -> np.ndarray:
    """把 f32 载体量化到 fp8 的格点，返回仍是 f32 的数组。

    只做舍入，不改 dtype：IR 里 fp8 的类型标签由调用方持有。实现是
    ``unpack_fp8(pack_fp8(x))``——编码与解码只有一份真值，量化结果必然落
    在一个可表示的位模式上。
    """
    return unpack_fp8(pack_fp8(value, kind), kind)


def quantize_fp8(value: Any, scale: Any, kind: str = "e4m3") -> np.ndarray:
    """带 scale 的 fp8 量化：``round_fp8(clip(x / scale, ±max))``。

    与 int8 的 ``quantize_i8`` 同构——先除以 scale、落到格点，产物是
    **缩放后的** f8 载体（数量级在 ``±max`` 内），反量化时再乘回 scale。
    这样 GEMM 可以先把整段值域压进 fp8 的可表示范围，而不要求权重本身
    落在 ``[-448, 448]``。

    为什么显式 clip：e4m3 在 encode 阶段自己饱和，但 e5m2 溢出会舍到
    ``±inf``；而量化权重里出现 inf 会让整条推理链变成 inf。统一先夹到
    格式的最大有限值，两种格式的语义就一致了。

    scale 必须能广播到 ``value``（逐通道 / 逐 token 时形状分别是 ``(N,)`` 与
    ``(..., 1)``），且有限、严格为正——与 int8 的约束同源。
    """
    fmt = _format(kind)
    x = np.asarray(value, dtype=np.float32)
    s = np.asarray(scale, dtype=np.float32)
    if not np.all(np.isfinite(s)) or np.any(s <= 0.0):
        raise ValueError(
            f"quantize_fp8 的 scale 必须是有限且严格为正的数，得到 {s.tolist()}"
        )
    scaled = np.clip(x / s, -fmt.max_value, fmt.max_value)
    return round_fp8(scaled, kind).astype(np.float32)


def dequantize_fp8(value: Any, scale: Any) -> np.ndarray:
    """把缩放后的 f8 载体乘回 scale，得到真实数值（仍是 f32 载体）。

    与 ``dequantize_i8`` 同一口径：fp8 张量在 IR 里是 f32 载体，所以这里
    只是一次乘法，不需要解码位模式。
    """
    product = np.asarray(value, dtype=np.float32) * np.asarray(scale, dtype=np.float32)
    return product.astype(np.float32)
