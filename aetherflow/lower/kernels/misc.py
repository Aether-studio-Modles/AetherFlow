"""拼接、补零与 dropout 掩码这类没有独立算子族的杂项内核。"""

from __future__ import annotations

from ..kernel import (
    Bin,
    Call,
    Cast,
    Cond,
    Index,
    Kernel,
    Let,
    Num,
    Paren,
    Scalar,
    Store,
    Tensor,
    Var,
)
from .common import _flat_loop


def concat_copy() -> Kernel:
    """拼接 / 补零共用的分段拷贝：把输入整段搬到输出偏移处。

    按 ``(outer, axis, inner)`` 看张量，拼接就是"这一段落到 axis 的某个偏移"，
    于是不需要"输出下标 -> 输入编号"的索引表，也不用把输入个数写成内核参数。

    这是纯搬运：值本来就是存储类型，读写都不套 ``af_ld`` / ``af_st``，
    所以 f16 下是逐位复制，不会经过 float 再舍入一次。
    """
    # ((o * out_axis) + offset + s) * inner + j
    dst = Bin(
        "+",
        Bin(
            "*",
            Paren(
                Bin(
                    "+",
                    Bin(
                        "+",
                        Paren(Bin("*", Var("o"), Var("out_axis"))),
                        Var("offset"),
                    ),
                    Var("s"),
                )
            ),
            Var("inner"),
        ),
        Var("j"),
    )
    return Kernel(
        name="concat_copy",
        params=(
            Tensor("x"),
            Tensor("out", read_only=False),
            Scalar("long long", "len"),
            Scalar("long long", "inner"),
            Scalar("long long", "out_axis"),
            Scalar("long long", "offset"),
            Scalar("long long", "total"),
        ),
        body=_flat_loop(
            (
                Let("long long", "j", Bin("%", Var("idx"), Var("inner"))),
                Let("long long", "t", Bin("/", Var("idx"), Var("inner"))),
                Let("long long", "s", Bin("%", Var("t"), Var("len"))),
                Let("long long", "o", Bin("/", Var("t"), Var("len"))),
                Store("out", dst, Index("x", Var("idx")), raw=True),
            )
        ),
    )


_SPLITMIX64 = r"""
__device__ __forceinline__ unsigned long long splitmix64(unsigned long long z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
"""


def dropout_mask() -> Kernel:
    """生成缩放后的 dropout 掩码。

    随机源刻意不用 cuRAND：参考实现（numpy）与内核必须生成**逐位相同**的掩码，
    否则这个算子只能用"统计上近似"来验证。SplitMix64 只做整数乘法与异或，
    两边跑同一套位运算即可对齐；掩码只取决于 ``(seed, 下标)``，与块的几何无关。
    """
    return Kernel(
        name="dropout_mask",
        params=(
            Tensor("out", read_only=False),
            Scalar("long long", "n"),
            Scalar("float", "p"),
            Scalar("float", "scale"),
            Scalar("unsigned long long", "seed"),
        ),
        body=_flat_loop(
            (
                Let(
                    "unsigned long long",
                    "z",
                    Call(
                        "splitmix64",
                        (
                            Bin(
                                "+",
                                Var("seed"),
                                Bin(
                                    "*",
                                    Cast("unsigned long long", Var("i")),
                                    Num("0x9E3779B97F4A7C15ULL"),
                                ),
                            ),
                        ),
                    ),
                ),
                # 取高 24 位映射到 [0, 1)：这正好是 float32 的尾数宽度，
                # numpy 与 CUDA 会落在同一个 float 上，不需要额外舍入。
                Let(
                    "float",
                    "u",
                    Bin(
                        "*",
                        Cast(
                            "float",
                            Paren(
                                Bin(
                                    "&",
                                    Paren(Bin(">>", Var("z"), Num("40"))),
                                    Num("0xFFFFFFULL"),
                                )
                            ),
                        ),
                        Paren(Bin("/", Num("1.0f"), Num("16777216.0f"))),
                    ),
                ),
                Store(
                    "out",
                    Var("i"),
                    Cond(
                        Paren(Bin(">=", Var("u"), Var("p"))),
                        Var("scale"),
                        Num("0.0f"),
                    ),
                ),
            ),
            index="i",
            count="n",
        ),
        helpers=_SPLITMIX64,
    )
