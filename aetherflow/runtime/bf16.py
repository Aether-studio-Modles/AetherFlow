"""BF16 的数值语义与两字节存储编码。

numpy 没有 bfloat16 标量类型，所以参考语义用 float32 作**载体**。但"载体是
f32"不等于"bf16 只是个标签"：本模块把每次 bf16 边界都真的量化到 bf16 精度，
并给出 2 字节的打包表示，供"参数以 bf16 驻留"这类存储路径使用。

bf16 与 f16 的取舍不同：bf16 保留 f32 的 8 位指数（范围相同，不会溢出到
inf），尾数只有 7 位（f16 是 10 位）。因此 bf16 不需要梯度缩放——下溢不是
它的问题，相对精度更差才是。

舍入采用 IEEE 754 的 round-to-nearest-even（RNE），与硬件 ``cvt`` 指令一致：

- 尾数第 16 位（bf16 的 LSB）为 1 时加 1，实现"恰好一半时进位到偶数"；
- 指数全 1 的 inf / nan **不参与进位**，只截断尾数：偏置加到尾数全 1 的
  NaN 上会进位到指数、甚至回绕，把 NaN 变成 0；截断后尾数为 0 的 NaN 再补
  一个 quiet 位，避免被读成 inf；
- 最大有限值加偏置后进位成 inf——这正是 RNE 的溢出语义。
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["round_bf16", "pack_bf16", "unpack_bf16", "STORAGE_BYTES"]

#: bf16 的存储宽度（字节）。参考后端的载体是 f32，但打包表示确实是 2 字节。
STORAGE_BYTES = 2

_MASK = np.uint32(0xFFFF0000)
_HALF = np.uint32(0x7FFF)
_ONE = np.uint32(1)
_SHIFT = np.uint32(16)
_EXP_MASK = np.uint32(0x7F800000)
_MANT_MASK = np.uint32(0x007FFFFF)
_QUIET = np.uint32(0x00400000)


def round_bf16(value: Any) -> np.ndarray:
    """把 f32 载体量化到 bf16 精度，返回仍是 f32 的数组。

    只做舍入，不改 dtype：IR 里 bf16 的类型标签由调用方持有，这里只保证
    数值确实落在一个 bf16 能表示的格点上。
    """
    # ``view`` 需要连续内存，所以先复制一份连续的 f32。
    arr = np.array(value, dtype=np.float32, copy=True, order="C", subok=False)
    bits = arr.view(np.uint32)
    # 先看尾数第 16 位（bf16 的 LSB）：为 1 时加 1 才会把"恰好一半"进位到
    # 偶数，这就是 RNE 与"四舍五入"的唯一区别。
    lsb = (bits >> _SHIFT) & _ONE
    rounded = (bits + _HALF + lsb) & _MASK
    # inf / nan 的指数全 1，加偏置会让尾数全 1 的 NaN 进位到指数（NaN 变 0）
    # 甚至回绕。它们本来就落在 bf16 的格点上，只截断尾数即可。
    special = (bits & _EXP_MASK) == _EXP_MASK
    if special.any():
        kept = bits & _MASK
        # 截断后尾数为 0 的 NaN 会被读成 inf：补一个 quiet 位保住 NaN。
        quiet = (
            special
            & ((bits & _MANT_MASK) != 0)
            & ((kept & _MANT_MASK) == 0)
        )
        rounded = np.where(
            special, kept | np.where(quiet, _QUIET, np.uint32(0)), rounded
        )
    return rounded.view(np.float32)


def pack_bf16(value: Any) -> np.ndarray:
    """打包成 2 字节的 ``uint16``（存的是 bf16 的位模式，不是整数数值）。"""
    return (round_bf16(value).view(np.uint32) >> _SHIFT).astype(np.uint16)


def unpack_bf16(value: Any) -> np.ndarray:
    """把 ``uint16`` 位模式还原成 f32 载体；bf16 -> f32 是精确的。"""
    bits = np.asarray(value)
    if bits.dtype != np.uint16:
        raise TypeError(
            f"bf16 打包数组必须是 uint16，得到 {bits.dtype}；"
            "这里收的是位模式而不是整数数值"
        )
    return (bits.astype(np.uint32) << _SHIFT).view(np.float32)
