"""融合注意力前向：在线 softmax + 分块点积。"""

from __future__ import annotations

from ..kernel import (
    Assign,
    AugAssign,
    Axis,
    Bin,
    Call,
    Cast,
    Cond,
    For,
    If,
    Index,
    Kernel,
    Let,
    Load,
    Num,
    Paren,
    Ptr,
    Scalar,
    Shared,
    Store,
    Sync,
    Tensor,
    Var,
)


_ATTENTION_RAND = r"""
__device__ __forceinline__ unsigned long long attention_rand(unsigned long long seed,
                                                             long long idx) {
    unsigned long long z =
        seed + (unsigned long long)idx * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
"""


def attention_fwd() -> Kernel:
    """融合注意力前向：一个 block 负责一个 (batch*head, query 行)。

    key 轴按 ``blockDim.x`` 分块流式处理，配合在线 softmax，``(S, T)`` 的
    分数矩阵从不落盘；跨块只保留行最大值、归一化因子与输出累加器，三者都在
    共享内存里，所以块内的 ``block_reduce_*`` 天然对全体线程广播同一个值。

    dropout 乘在归一化后的权重上，不改进分母——与 ``ir/ops.py`` 里"dropout
    作用在 softmax 概率上"的定义一致。掩码由 ``(seed, (bh*S+s)*T+t)`` 决定，
    与块几何无关，因此与参考实现逐位相同。

    ``mask_shared=1`` 表示所有 ``(b, h)`` 共用一份 ``(S, T)``（因果 mask 的
    常见形态）：此时跳过 ``bh`` 偏移，主机侧不必把它复制成 ``(BH, S, T)``。
    """
    # q / k / v 的行偏移始终按真实 (b, h) 算
    row = Bin("+", Bin("*", Var("bh"), Var("S")), Var("s"))
    # (bh * S + s) * D / T / E
    row_stride = lambda dim: Bin("*", Paren(row), Var(dim))
    # mask 的行偏移：(mask_shared ? 0 : bh) * S + s
    mask_index = Bin(
        "+",
        Bin("*", Paren(Cond(Var("mask_shared"), Num("0"), Var("bh"))), Var("S")),
        Var("s"),
    )
    return Kernel(
        name="attention_fwd",
        params=(
            Tensor("q"),
            Tensor("k"),
            Tensor("v"),
            Tensor("mask"),
            Tensor("out", read_only=False),
            Scalar("long long", "S"),
            Scalar("long long", "T"),
            Scalar("long long", "D"),
            Scalar("long long", "E"),
            Scalar("float", "scale"),
            Scalar("float", "p_drop"),
            Scalar("float", "drop_scale"),
            Scalar("unsigned long long", "seed"),
            Scalar("int", "has_mask"),
            Scalar("int", "mask_shared"),
        ),
        body=(
            Shared("float", "smem", extern=True),
            Let("float*", "s_w", Var("smem")),
            Let("float*", "s_red", Bin("+", Var("smem"), Axis("bdim"))),
            Let(
                "float*",
                "s_acc",
                Bin("+", Var("smem"), Bin("*", Num("2"), Axis("bdim"))),
            ),
            Let("const long long", "bh", Axis("bid", "y")),
            Let("const long long", "s", Axis("bid", "x")),
            For(
                "e",
                Axis("tid"),
                Var("E"),
                Axis("bdim"),
                (Assign(Index("s_acc", Var("e")), Num("0.f")),),
            ),
            Sync(),
            Ptr("qs", "q", row_stride("D")),
            Ptr(
                "vbase",
                "v",
                Bin("*", Bin("*", Var("bh"), Var("T")), Var("E")),
            ),
            Let(
                "const %T%*",
                "mrow",
                Cond(
                    Var("has_mask"),
                    Paren(
                        Bin(
                            "+",
                            Var("mask"),
                            Bin("*", Paren(mask_index), Var("T")),
                        )
                    ),
                    Var("nullptr"),
                ),
            ),
            Let("float", "m", Num("-3.4e38f")),
            Let("float", "l", Num("0.f")),
            For(
                "t0",
                Num("0"),
                Var("T"),
                Axis("bdim"),
                (
                    Let(
                        "const long long",
                        "t",
                        Bin("+", Var("t0"), Axis("tid")),
                    ),
                    Let("float", "sc", Num("-3.4e38f")),
                    If(
                        Bin("<", Var("t"), Var("T")),
                        (
                            Let(
                                "const %T%*",
                                "kt",
                                Bin(
                                    "+",
                                    Var("k"),
                                    Bin(
                                        "*",
                                        Paren(
                                            Bin(
                                                "+",
                                                Bin("*", Var("bh"), Var("T")),
                                                Var("t"),
                                            )
                                        ),
                                        Var("D"),
                                    ),
                                ),
                            ),
                            Let("float", "dot", Num("0.f")),
                            For(
                                "d",
                                Num("0"),
                                Var("D"),
                                Num("1"),
                                (
                                    AugAssign(
                                        Var("dot"),
                                        "+",
                                        Bin(
                                            "*",
                                            Load("qs", Var("d")),
                                            Load("kt", Var("d")),
                                        ),
                                    ),
                                ),
                                incr="++",
                                type_name="long long",
                            ),
                            Assign(
                                Var("sc"),
                                Bin("*", Var("dot"), Var("scale")),
                            ),
                            If(
                                Var("has_mask"),
                                (
                                    AugAssign(
                                        Var("sc"),
                                        "+",
                                        Load("mrow", Var("t")),
                                    ),
                                ),
                            ),
                        ),
                    ),
                    Let(
                        "const float",
                        "blk_max",
                        Call("block_reduce_max", (Var("sc"), Var("s_red"))),
                    ),
                    Let(
                        "const float",
                        "m_new",
                        Cond(
                            Bin(">", Var("m"), Var("blk_max")),
                            Var("m"),
                            Var("blk_max"),
                        ),
                    ),
                    Let(
                        "const float",
                        "corr",
                        Call("expf", (Bin("-", Var("m"), Var("m_new")),)),
                    ),
                    Let("float", "p", Num("0.f")),
                    Let("float", "w", Num("0.f")),
                    If(
                        Bin("<", Var("t"), Var("T")),
                        (
                            Assign(
                                Var("p"),
                                Call("expf", (Bin("-", Var("sc"), Var("m_new")),)),
                            ),
                            Assign(Var("w"), Var("p")),
                            If(
                                Bin(">", Var("p_drop"), Num("0.f")),
                                (
                                    Let(
                                        "const unsigned long long",
                                        "z",
                                        Call(
                                            "attention_rand",
                                            (
                                                Var("seed"),
                                                Bin(
                                                    "+",
                                                    Bin("*", Paren(row), Var("T")),
                                                    Var("t"),
                                                ),
                                            ),
                                        ),
                                    ),
                                    Let(
                                        "const float",
                                        "u",
                                        Bin(
                                            "*",
                                            Cast(
                                                "float",
                                                Paren(
                                                    Bin(
                                                        "&",
                                                        Paren(
                                                            Bin(
                                                                ">>",
                                                                Var("z"),
                                                                Num("40"),
                                                            )
                                                        ),
                                                        Num("0xFFFFFFULL"),
                                                    )
                                                ),
                                            ),
                                            Paren(
                                                Bin(
                                                    "/",
                                                    Num("1.0f"),
                                                    Num("16777216.0f"),
                                                )
                                            ),
                                        ),
                                    ),
                                    Assign(
                                        Var("w"),
                                        Cond(
                                            Paren(
                                                Bin(
                                                    ">=",
                                                    Var("u"),
                                                    Var("p_drop"),
                                                )
                                            ),
                                            Bin("*", Var("p"), Var("drop_scale")),
                                            Num("0.f"),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                    Assign(Index("s_w", Axis("tid")), Var("w")),
                    Let(
                        "const float",
                        "blk_sum",
                        Call("block_reduce_sum", (Var("p"), Var("s_red"))),
                    ),
                    Assign(
                        Var("l"),
                        Bin(
                            "+",
                            Bin("*", Var("l"), Var("corr")),
                            Var("blk_sum"),
                        ),
                    ),
                    For(
                        "e",
                        Axis("tid"),
                        Var("E"),
                        Axis("bdim"),
                        (
                            Let(
                                "float",
                                "acc",
                                Bin("*", Index("s_acc", Var("e")), Var("corr")),
                            ),
                            For(
                                "j",
                                Num("0"),
                                Bin(
                                    "&&",
                                    Axis("bdim"),
                                    Bin("<", Bin("+", Var("t0"), Var("j")), Var("T")),
                                ),
                                Num("1"),
                                (
                                    AugAssign(
                                        Var("acc"),
                                        "+",
                                        Bin(
                                            "*",
                                            Index("s_w", Var("j")),
                                            Load(
                                                "vbase",
                                                Bin(
                                                    "+",
                                                    Bin(
                                                        "*",
                                                        Paren(
                                                            Bin(
                                                                "+",
                                                                Var("t0"),
                                                                Var("j"),
                                                            )
                                                        ),
                                                        Var("E"),
                                                    ),
                                                    Var("e"),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                                incr="++",
                                type_name="long long",
                            ),
                            Assign(Index("s_acc", Var("e")), Var("acc")),
                        ),
                    ),
                    Sync(),
                    Assign(Var("m"), Var("m_new")),
                ),
            ),
            Let(
                "const float",
                "inv",
                Cond(
                    Bin(">", Var("l"), Num("0.f")),
                    Bin("/", Num("1.f"), Var("l")),
                    Num("0.f"),
                ),
            ),
            Ptr("orow", "out", row_stride("E"), const=False),
            For(
                "e",
                Axis("tid"),
                Var("E"),
                Axis("bdim"),
                (
                    Store(
                        "orow",
                        Var("e"),
                        Bin("*", Index("s_acc", Var("e")), Var("inv")),
                    ),
                ),
            ),
        ),
        helpers=_ATTENTION_RAND,
    )
