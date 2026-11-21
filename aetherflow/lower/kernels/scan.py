"""线性递推与前缀和：整段扫描、分块扫描及其反向。"""

from __future__ import annotations

from ..kernel import (
    Assign,
    Bin,
    Cond,
    For,
    Index,
    Kernel,
    Let,
    Load,
    Num,
    Paren,
    Scalar,
    Store,
    Tensor,
    Var,
)
from ..indexing import Layout
from .common import _flat_loop, grid_stride


def scan_linear() -> Kernel:
    """线性递推 ``h_t = a_t * h_{t-1} + b_t``（``h_{-1} = h0``）。

    一个线程负责一个尾元素、沿时间轴顺序走。递推本身串行，并行度来自尾部
    （batch / channel / state），所以尾维被展平，秩 >= 2 都适用。
    """
    idx = _T_ELEMS.offset(("t", "i"), cast=None)
    return Kernel(
        name="scan_linear",
        params=(
            Tensor("a"),
            Tensor("b"),
            Tensor("h0"),
            Tensor("out", read_only=False),
            Scalar("long long", "steps"),
            Scalar("long long", "elems"),
        ),
        body=_flat_loop(
            (
                Let("float", "h", Load("h0", Var("i"))),
                For(
                    "t",
                    Num("0"),
                    Var("steps"),
                    Num("1"),
                    (
                        Let("const long long", "idx", idx),
                        Assign(
                            Var("h"),
                            Bin(
                                "+",
                                Bin("*", Load("a", Var("idx")), Var("h")),
                                Load("b", Var("idx")),
                            ),
                        ),
                        Store("out", Var("idx"), Var("h")),
                    ),
                    incr="++",
                ),
            ),
            index="i",
            count="elems",
            stride="stride",
        ),
    )


def scan_linear_backward() -> Kernel:
    """线性递推的反向：``dh_t = dy_t + a_{t+1} * dh_{t+1}``，``dh_T = 0``。

    同样的"一个线程一个尾元素"，只是倒着走。``a[idx + elems]`` 是 ``a_{t+1}``；
    最后一步没有后继，进位被丢弃。
    """
    idx = _T_ELEMS.offset(("t", "i"), cast=None)
    return Kernel(
        name="scan_linear_backward",
        params=(
            Tensor("a"),
            Tensor("dy"),
            Tensor("out", read_only=False),
            Scalar("long long", "steps"),
            Scalar("long long", "elems"),
        ),
        body=_flat_loop(
            (
                Let("float", "dh", Num("0.f")),
                For(
                    "t",
                    Bin("-", Var("steps"), Num("1")),
                    Num("0"),
                    Num("1"),
                    (
                        Let("const long long", "idx", idx),
                        Let(
                            "const float",
                            "carry",
                            Cond(
                                Paren(
                                    Bin(
                                        "<",
                                        Bin("+", Var("t"), Num("1")),
                                        Var("steps"),
                                    )
                                ),
                                Bin(
                                    "*",
                                    Load(
                                        "a",
                                        Bin("+", Var("idx"), Var("elems")),
                                    ),
                                    Var("dh"),
                                ),
                                Num("0.f"),
                            ),
                        ),
                        Assign(
                            Var("dh"),
                            Bin("+", Load("dy", Var("idx")), Var("carry")),
                        ),
                        Store("out", Var("idx"), Var("dh")),
                    ),
                    cmp=">=",
                    incr="--",
                ),
            ),
            index="i",
            count="elems",
            stride="stride",
        ),
    )


def cumsum() -> Kernel:
    """沿单个轴的前缀和：一个线程负责一个 ``(outer, inner)`` 位置。

    把张量看成 ``(outer, steps, inner)``，前缀和只沿 ``steps`` 走，所以并行度
    来自 ``outer * inner``（batch / 其余维），与 ``scan_linear`` 的铺法同构。
    累加用 ``float``：f16 逐项累加会在长序列上先溢出再失真，而前缀和每一步
    都是最终输出，没有"最后再 cast 一次"能补救。
    """
    return Kernel(
        name="cumsum",
        params=(
            Tensor("x"),
            Tensor("out", read_only=False),
            Scalar("long long", "work"),
            Scalar("long long", "steps"),
            Scalar("long long", "inner"),
        ),
        body=_flat_loop(
            (
                Let("const long long", "o", Bin("/", Var("idx"), Var("inner"))),
                Let("const long long", "j", Bin("%", Var("idx"), Var("inner"))),
                Let(
                    "const long long",
                    "base",
                    Bin(
                        "+",
                        Bin(
                            "*",
                            Var("o"),
                            Bin("*", Var("steps"), Var("inner")),
                        ),
                        Var("j"),
                    ),
                ),
                Let("float", "acc", Num("0.f")),
                For(
                    "t",
                    Num("0"),
                    Var("steps"),
                    Num("1"),
                    (
                        Let(
                            "const long long",
                            "pos",
                            Bin(
                                "+",
                                Var("base"),
                                Bin("*", Var("t"), Var("inner")),
                            ),
                        ),
                        Assign(
                            Var("acc"),
                            Bin("+", Var("acc"), Load("x", Var("pos"))),
                        ),
                        Store("out", Var("pos"), Var("acc")),
                    ),
                    incr="++",
                ),
            ),
            index="idx",
            count="work",
            stride="step",
        ),
    )


def tril() -> Kernel:
    """最后两维的下三角掩码：一个线程一个元素。

    ``idx`` 是 C 布局的线性下标，所以 ``c = idx % cols``、
    ``r = (idx / cols) % rows``；前导维自然被 ``% rows`` 吃掉，不需要单独
    传 batch。条件外写 0 而不是跳过，输出才不会带着未初始化值。
    """
    return Kernel(
        name="tril",
        params=(
            Tensor("x"),
            Tensor("out", read_only=False),
            Scalar("long long", "n"),
            Scalar("long long", "rows"),
            Scalar("long long", "cols"),
            Scalar("long long", "diagonal"),
        ),
        # grid_stride 是可变参数（``*body``）：这里必须平铺传入，
        # 包成一个 tuple 会被当成"一条语句"，渲染器不认识嵌套 tuple。
        body=grid_stride(
            Var("n"),
            Let("long long", "c", Bin("%", Var("i"), Var("cols"))),
            Let(
                "long long",
                "r",
                Bin(
                    "%",
                    Bin("/", Var("i"), Var("cols")),
                    Var("rows"),
                ),
            ),
            Store(
                "out",
                Var("i"),
                Cond(
                    Bin(
                        "<=",
                        Var("c"),
                        Bin("+", Var("r"), Var("diagonal")),
                    ),
                    Load("x", Var("i")),
                    Num("0.f"),
                ),
            ),
        ),
    )


def _chunk_coords() -> tuple:
    """把展平下标 ``j`` 拆成 ``(块号 k, 尾元素 i)``。

    分块扫描的两个趟都用 ``chunks * elems`` 个线程，块号在前的布局与
    ``prod`` / ``sum`` 的 ``(chunks, elems)`` 形状一致，所以下标就是
    ``k * elems + i``。
    """
    return (
        Let("const long long", "i", Bin("%", Var("j"), Var("elems"))),
        Let("const long long", "k", Bin("/", Var("j"), Var("elems"))),
    )


def _chunk_bounds(prefix: str = "t") -> tuple:
    """``t0 = k * chunk``、``t1 = min(t0 + chunk, steps)``（按时间轴方向）。"""
    start = Bin("*", Var("k"), Var("chunk"))
    end = Bin("+", start, Var("chunk"))
    return (
        Let("const long long", prefix + "0", start),
        Let(
            "const long long",
            prefix + "1",
            Cond(
                Paren(Bin("<", end, Var("steps"))),
                end,
                Var("steps"),
            ),
        ),
    )


def _chunk_prefix_scan() -> tuple:
    """读 ``prod`` / ``sum`` 求本块入口状态 ``h_in``。

    块 k 的入口是前 k 块递推的复合：``h <- prod_c * h + sum_c``。这是块间
    唯一的串行部分，长度是 ``chunks`` 而不是 ``steps``；每个线程各自重算
    一遍，省掉一趟 launch 与一份 scratch，块数远小于 T 时这比再写一趟便宜。

    ``prod`` / ``sum`` 固定用 float 存储（不随 dtype 变成 f16）：块内递推
    本来就在 float 里做，把聚合值降成 f16 会额外舍入两次，误差比串行版大
    一个数量级。
    """
    return (
        Let("float", "h", Load("h0", Var("i"))),
        For(
            "c",
            Num("0"),
            Var("k"),
            Num("1"),
            (
                Let(
                    "const long long",
                    "cidx",
                    _C_ELEMS.offset(("c", "i"), cast=None),
                ),
                Assign(
                    Var("h"),
                    Bin(
                        "+",
                        Bin(
                            "*",
                            Index("prod", Var("cidx")),
                            Var("h"),
                        ),
                        Index("sum", Var("cidx")),
                    ),
                ),
            ),
            incr="++",
        ),
    )


def scan_linear_chunk_sum() -> Kernel:
    """分块扫描第一趟：每块算复合 ``(prod, sum)``，即 ``h_out = prod*h_in + sum``。

    块内仍然逐 token 顺序递推（``s <- a*s + b``、``p <- a*p``），只是初值取
    中性元 ``s=0, p=1``，所以块和与块入口状态无关、可以完全并行。并行度从
    "尾元素数"变成"尾元素数 × 块数"——这正是分块扫描比串行版快的原因。
    """
    return Kernel(
        name="scan_linear_chunk_sum",
        params=(
            Tensor("a"),
            Tensor("b"),
            Tensor("prod", read_only=False, type_name="float"),
            Tensor("sum", read_only=False, type_name="float"),
            Scalar("long long", "steps"),
            Scalar("long long", "elems"),
            Scalar("long long", "chunk"),
            Scalar("long long", "total"),
        ),
        body=_flat_loop(
            _chunk_coords()
            + _chunk_bounds()
            + (
                Let("float", "p", Num("1.f")),
                Let("float", "s", Num("0.f")),
                For(
                    "t",
                    Var("t0"),
                    Var("t1"),
                    Num("1"),
                    (
                        Let(
                            "const long long",
                            "idx",
                            _T_ELEMS.offset(("t", "i"), cast=None),
                        ),
                        Let("const float", "av", Load("a", Var("idx"))),
                        Assign(
                            Var("s"),
                            Bin(
                                "+",
                                Bin("*", Var("av"), Var("s")),
                                Load("b", Var("idx")),
                            ),
                        ),
                        Assign(Var("p"), Bin("*", Var("av"), Var("p"))),
                    ),
                    incr="++",
                ),
                Store("prod", Var("j"), Var("p"), raw=True),
                Store("sum", Var("j"), Var("s"), raw=True),
            ),
            index="j",
            count="total",
            stride="stride",
        ),
    )


def scan_linear_chunk() -> Kernel:
    """分块扫描第二趟：先求块前缀 ``h_in``，再块内顺序写出 ``out``。

    与 ``scan_linear`` 的区别只在**块间**：块内递推逐字相同，因此
    ``chunk >= steps`` 时两者逐位一致；只要跨了块，块入口就是"先乘再加"
    的复合，舍入路径不同，误差量级约 1e-7（f32）。
    """
    return Kernel(
        name="scan_linear_chunk",
        params=(
            Tensor("a"),
            Tensor("b"),
            Tensor("h0"),
            Tensor("prod", type_name="float"),
            Tensor("sum", type_name="float"),
            Tensor("out", read_only=False),
            Scalar("long long", "steps"),
            Scalar("long long", "elems"),
            Scalar("long long", "chunk"),
            Scalar("long long", "total"),
        ),
        body=_flat_loop(
            _chunk_coords()
            + _chunk_bounds()
            + _chunk_prefix_scan()
            + (
                For(
                    "t",
                    Var("t0"),
                    Var("t1"),
                    Num("1"),
                    (
                        Let(
                            "const long long",
                            "idx",
                            _T_ELEMS.offset(("t", "i"), cast=None),
                        ),
                        Assign(
                            Var("h"),
                            Bin(
                                "+",
                                Bin("*", Load("a", Var("idx")), Var("h")),
                                Load("b", Var("idx")),
                            ),
                        ),
                        Store("out", Var("idx"), Var("h")),
                    ),
                    incr="++",
                ),
            ),
            index="j",
            count="total",
            stride="stride",
        ),
    )


def scan_linear_chunk_sum_backward() -> Kernel:
    """分块反向第一趟：沿倒序时间轴按块聚合。

    把反向递推 ``dh_t = dy_t + a_{t+1} * dh_{t+1}`` 看成正向扫描
    ``s <- A*s + B``，其中 ``A = a_{t+1}``（``t = steps-1`` 时为 0）、
    ``B = dy_t``。倒序的第 ``r`` 步对应 ``t = steps-1-r``，块也按 ``r`` 切。
    """
    return Kernel(
        name="scan_linear_chunk_sum_backward",
        params=(
            Tensor("a"),
            Tensor("dy"),
            Tensor("prod", read_only=False, type_name="float"),
            Tensor("sum", read_only=False, type_name="float"),
            Scalar("long long", "steps"),
            Scalar("long long", "elems"),
            Scalar("long long", "chunk"),
            Scalar("long long", "total"),
        ),
        body=_flat_loop(
            _chunk_coords()
            + _chunk_bounds()
            + (
                Let("float", "p", Num("1.f")),
                Let("float", "s", Num("0.f")),
                For(
                    "r",
                    Var("t0"),
                    Var("t1"),
                    Num("1"),
                    (
                        Let(
                            "const long long",
                            "t",
                            Bin("-", Bin("-", Var("steps"), Num("1")), Var("r")),
                        ),
                        Let(
                            "const long long",
                            "idx",
                            _T_ELEMS.offset(("t", "i"), cast=None),
                        ),
                        Let(
                            "const float",
                            "av",
                            Cond(
                                Paren(
                                    Bin(
                                        "<",
                                        Bin("+", Var("t"), Num("1")),
                                        Var("steps"),
                                    )
                                ),
                                Load(
                                    "a",
                                    Bin("+", Var("idx"), Var("elems")),
                                ),
                                Num("0.f"),
                            ),
                        ),
                        Assign(
                            Var("s"),
                            Bin(
                                "+",
                                Bin("*", Var("av"), Var("s")),
                                Load("dy", Var("idx")),
                            ),
                        ),
                        Assign(Var("p"), Bin("*", Var("av"), Var("p"))),
                    ),
                    incr="++",
                ),
                Store("prod", Var("j"), Var("p"), raw=True),
                Store("sum", Var("j"), Var("s"), raw=True),
            ),
            index="j",
            count="total",
            stride="stride",
        ),
    )


def scan_linear_chunk_backward() -> Kernel:
    """分块反向第二趟：块前缀 + 块内倒序写出。"""
    return Kernel(
        name="scan_linear_chunk_backward",
        params=(
            Tensor("a"),
            Tensor("dy"),
            Tensor("prod", type_name="float"),
            Tensor("sum", type_name="float"),
            Tensor("out", read_only=False),
            Scalar("long long", "steps"),
            Scalar("long long", "elems"),
            Scalar("long long", "chunk"),
            Scalar("long long", "total"),
        ),
        body=_flat_loop(
            _chunk_coords()
            + _chunk_bounds()
            + (
                # 反向的初值是 0（dh_T = 0），块前缀从 0 起扫。
                Let("float", "h", Num("0.f")),
                For(
                    "c",
                    Num("0"),
                    Var("k"),
                    Num("1"),
                    (
                        Let(
                            "const long long",
                            "cidx",
                            _C_ELEMS.offset(("c", "i"), cast=None),
                        ),
                        Assign(
                            Var("h"),
                            Bin(
                                "+",
                                Bin("*", Index("prod", Var("cidx")), Var("h")),
                                Index("sum", Var("cidx")),
                            ),
                        ),
                    ),
                    incr="++",
                ),
                For(
                    "r",
                    Var("t0"),
                    Var("t1"),
                    Num("1"),
                    (
                        Let(
                            "const long long",
                            "t",
                            Bin("-", Bin("-", Var("steps"), Num("1")), Var("r")),
                        ),
                        Let(
                            "const long long",
                            "idx",
                            _T_ELEMS.offset(("t", "i"), cast=None),
                        ),
                        Let(
                            "const float",
                            "av",
                            Cond(
                                Paren(
                                    Bin(
                                        "<",
                                        Bin("+", Var("t"), Num("1")),
                                        Var("steps"),
                                    )
                                ),
                                Load(
                                    "a",
                                    Bin("+", Var("idx"), Var("elems")),
                                ),
                                Num("0.f"),
                            ),
                        ),
                        Assign(
                            Var("h"),
                            Bin(
                                "+",
                                Load("dy", Var("idx")),
                                Bin("*", Var("av"), Var("h")),
                            ),
                        ),
                        Store("out", Var("idx"), Var("h")),
                    ),
                    incr="++",
                ),
            ),
            index="j",
            count="total",
            stride="stride",
        ),
    )


def scan_linear_last() -> Kernel:
    """只写末状态：一个线程一个尾元素，走完 T 步后只写一次。

    与 ``scan_linear`` 的计算完全相同，差别只在**写**：``T-1`` 次全局写与
    对应的 ``T-1`` 次读改写都被省掉。流式推理只需要 ``(tail...)`` 的末状态
    时（stateful 模型逐 token 推进），这是唯一被需要的输出。
    """
    return Kernel(
        name="scan_linear_last",
        params=(
            Tensor("a"),
            Tensor("b"),
            Tensor("h0"),
            Tensor("out", read_only=False),
            Scalar("long long", "steps"),
            Scalar("long long", "elems"),
        ),
        body=_flat_loop(
            (
                Let("float", "h", Load("h0", Var("i"))),
                For(
                    "t",
                    Num("0"),
                    Var("steps"),
                    Num("1"),
                    (
                        Let(
                            "const long long",
                            "idx",
                            _T_ELEMS.offset(("t", "i"), cast=None),
                        ),
                        Assign(
                            Var("h"),
                            Bin(
                                "+",
                                Bin("*", Load("a", Var("idx")), Var("h")),
                                Load("b", Var("idx")),
                            ),
                        ),
                    ),
                    incr="++",
                ),
                Store("out", Var("i"), Var("h")),
            ),
            index="i",
            count="elems",
            stride="stride",
        ),
    )


def scan_linear_chunk_last() -> Kernel:
    """分块版的末状态：读块和一次性复合，O(chunks) 步写一个数。

    与 ``scan_linear_chunk_sum`` 配对使用：第一趟按块聚合，这一趟一个线程
    一个尾元素，沿块串行走 ``chunks`` 步。不需要再读 ``a`` / ``b``。
    """
    return Kernel(
        name="scan_linear_chunk_last",
        params=(
            Tensor("prod", type_name="float"),
            Tensor("sum", type_name="float"),
            Tensor("h0"),
            Tensor("out", read_only=False),
            Scalar("long long", "chunks"),
            Scalar("long long", "elems"),
        ),
        body=_flat_loop(
            (
                Let("float", "h", Load("h0", Var("i"))),
                For(
                    "c",
                    Num("0"),
                    Var("chunks"),
                    Num("1"),
                    (
                        Let(
                            "const long long",
                            "cidx",
                            _C_ELEMS.offset(("c", "i"), cast=None),
                        ),
                        Assign(
                            Var("h"),
                            Bin(
                                "+",
                                Bin("*", Index("prod", Var("cidx")), Var("h")),
                                Index("sum", Var("cidx")),
                            ),
                        ),
                    ),
                    incr="++",
                ),
                Store("out", Var("i"), Var("h")),
            ),
            index="i",
            count="elems",
            stride="stride",
        ),
    )


_T_ELEMS = Layout(("t", "i"), (None, "elems"))


_C_ELEMS = Layout(("c", "i"), (None, "elems"))
