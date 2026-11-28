"""卷积内核：im2col、直接分块、分组、隐式 GEMM 与 Winograd。"""

from __future__ import annotations

from ..kernel import (
    ArrayDecl,
    Assign,
    AugAssign,
    Axis,
    Bin,
    Cast,
    Cond,
    Continue,
    Expr,
    For,
    If,
    Index,
    Kernel,
    Let,
    Load,
    Not,
    Num,
    Paren,
    Ptr,
    Return,
    Scalar,
    Shared,
    Store,
    Sync,
    Tensor,
    Var,
)
from .common import _flat_loop, grid_stride
from .image import _split_output_index, _tap


def _conv_tap(b: str, ch: str, ch_dim: str, y: str, y_dim: str, x: str, x_dim: str) -> Expr:
    """``((long long)(b * ch_dim + ch) * y_dim + y) * x_dim + x``"""
    return Bin(
        "+",
        Bin(
            "*",
            Paren(
                Bin(
                    "+",
                    Bin(
                        "*",
                        Cast(
                            "long long",
                            Paren(
                                Bin(
                                    "+",
                                    Bin("*", Var(b), Var(ch_dim)),
                                    Var(ch),
                                )
                            ),
                        ),
                        Var(y_dim),
                    ),
                    Var(y),
                )
            ),
            Var(x_dim),
        ),
        Var(x),
    )


def conv2d_input_grad() -> Kernel:
    """卷积对输入求梯度（gather 形式）：一个线程负责一个 ``dx`` 元素。

    固定输入格 ``(ih, iw)`` 后，贡献它的输出格满足
    ``ho*SH - PH + ki*DH == ih``，即 ``ho = (ih + PH - ki*DH)/SH`` 且必须整除。
    用 gather 而不是 scatter 就不需要原子加：每个 dx 元素只被一个线程写。
    """
    return Kernel(
        name="conv2d_input_grad",
        params=(
            Tensor("dy"),
            Tensor("weight"),
            Tensor("dx", read_only=False),
            Scalar("int", "C"),
            Scalar("int", "H"),
            Scalar("int", "W"),
            Scalar("int", "O"),
            Scalar("int", "KH"),
            Scalar("int", "KW"),
            Scalar("int", "HO"),
            Scalar("int", "WO"),
            Scalar("int", "SH"),
            Scalar("int", "SW"),
            Scalar("int", "PH"),
            Scalar("int", "PW"),
            Scalar("int", "DH"),
            Scalar("int", "DW"),
            Scalar("long long", "total"),
        ),
        body=_flat_loop(
            _split_output_index("iw", "ih", "W", "H")
            + (
                Let("float", "acc", Num("0.f")),
                For(
                    "oi",
                    Num("0"),
                    Var("O"),
                    Num("1"),
                    (
                        For(
                            "ki",
                            Num("0"),
                            Var("KH"),
                            Num("1"),
                            (
                                Let(
                                    "int",
                                    "num",
                                    Bin(
                                        "-",
                                        Bin("+", Var("ih"), Var("PH")),
                                        Bin("*", Var("ki"), Var("DH")),
                                    ),
                                ),
                                If(
                                    Bin(
                                        "||",
                                        Bin("<", Var("num"), Num("0")),
                                        Bin(
                                            "!=",
                                            Bin("%", Var("num"), Var("SH")),
                                            Num("0"),
                                        ),
                                    ),
                                    (Continue(),),
                                ),
                                Let("int", "ho", Bin("/", Var("num"), Var("SH"))),
                                If(
                                    Bin(">=", Var("ho"), Var("HO")),
                                    (Continue(),),
                                ),
                                For(
                                    "kj",
                                    Num("0"),
                                    Var("KW"),
                                    Num("1"),
                                    (
                                        Let(
                                            "int",
                                            "numw",
                                            Bin(
                                                "-",
                                                Bin(
                                                    "+",
                                                    Var("iw"),
                                                    Var("PW"),
                                                ),
                                                Bin("*", Var("kj"), Var("DW")),
                                            ),
                                        ),
                                        If(
                                            Bin(
                                                "||",
                                                Bin("<", Var("numw"), Num("0")),
                                                Bin(
                                                    "!=",
                                                    Bin(
                                                        "%",
                                                        Var("numw"),
                                                        Var("SW"),
                                                    ),
                                                    Num("0"),
                                                ),
                                            ),
                                            (Continue(),),
                                        ),
                                        Let(
                                            "int",
                                            "wo",
                                            Bin("/", Var("numw"), Var("SW")),
                                        ),
                                        If(
                                            Bin(">=", Var("wo"), Var("WO")),
                                            (Continue(),),
                                        ),
                                        AugAssign(
                                            Var("acc"),
                                            "+",
                                            Bin(
                                                "*",
                                                Load(
                                                    "dy",
                                                    _conv_tap(
                                                        "n",
                                                        "oi",
                                                        "O",
                                                        "ho",
                                                        "HO",
                                                        "wo",
                                                        "WO",
                                                    ),
                                                ),
                                                Load(
                                                    "weight",
                                                    _conv_tap(
                                                        "oi",
                                                        "c",
                                                        "C",
                                                        "ki",
                                                        "KH",
                                                        "kj",
                                                        "KW",
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                    incr="++",
                                    type_name="int",
                                ),
                            ),
                            incr="++",
                            type_name="int",
                        ),
                    ),
                    incr="++",
                    type_name="int",
                ),
                Store("dx", Var("idx"), Var("acc")),
            )
        ),
    )


def conv2d_weight_grad() -> Kernel:
    """卷积对权重求梯度（gather 形式）：一个线程负责一个 ``dw`` 元素。

    沿整个 batch 与所有输出格累加，所以小 batch 训练时这里是计算瓶颈，
    与卷积前向同阶。同样用 gather 保持结果可复现（不用 float 原子加）。
    """
    return Kernel(
        name="conv2d_weight_grad",
        params=(
            Tensor("x"),
            Tensor("dy"),
            Tensor("dw", read_only=False),
            Scalar("int", "N"),
            Scalar("int", "C"),
            Scalar("int", "H"),
            Scalar("int", "W"),
            Scalar("int", "O"),
            Scalar("int", "KH"),
            Scalar("int", "KW"),
            Scalar("int", "HO"),
            Scalar("int", "WO"),
            Scalar("int", "SH"),
            Scalar("int", "SW"),
            Scalar("int", "PH"),
            Scalar("int", "PW"),
            Scalar("int", "DH"),
            Scalar("int", "DW"),
            Scalar("long long", "total"),
        ),
        body=_flat_loop(
            (
                Let(
                    "int",
                    "kj",
                    Cast("int", Paren(Bin("%", Var("idx"), Var("KW")))),
                ),
                Let("long long", "t", Bin("/", Var("idx"), Var("KW"))),
                Let(
                    "int",
                    "ki",
                    Cast("int", Paren(Bin("%", Var("t"), Var("KH")))),
                ),
                AugAssign(Var("t"), "/", Var("KH")),
                Let(
                    "int",
                    "ci",
                    Cast("int", Paren(Bin("%", Var("t"), Var("C")))),
                ),
                Let(
                    "int",
                    "oi",
                    Cast("int", Paren(Bin("/", Var("t"), Var("C")))),
                ),
                Let("float", "acc", Num("0.f")),
                For(
                    "ni",
                    Num("0"),
                    Var("N"),
                    Num("1"),
                    (
                        For(
                            "ho",
                            Num("0"),
                            Var("HO"),
                            Num("1"),
                            (
                                Let(
                                    "int",
                                    "ih",
                                    Bin(
                                        "+",
                                        Bin(
                                            "-",
                                            Bin("*", Var("ho"), Var("SH")),
                                            Var("PH"),
                                        ),
                                        Bin("*", Var("ki"), Var("DH")),
                                    ),
                                ),
                                If(
                                    Bin(
                                        "||",
                                        Bin("<", Var("ih"), Num("0")),
                                        Bin(">=", Var("ih"), Var("H")),
                                    ),
                                    (Continue(),),
                                ),
                                For(
                                    "wo",
                                    Num("0"),
                                    Var("WO"),
                                    Num("1"),
                                    (
                                        Let(
                                            "int",
                                            "iw",
                                            Bin(
                                                "+",
                                                Bin(
                                                    "-",
                                                    Bin(
                                                        "*",
                                                        Var("wo"),
                                                        Var("SW"),
                                                    ),
                                                    Var("PW"),
                                                ),
                                                Bin(
                                                    "*",
                                                    Var("kj"),
                                                    Var("DW"),
                                                ),
                                            ),
                                        ),
                                        If(
                                            Bin(
                                                "||",
                                                Bin("<", Var("iw"), Num("0")),
                                                Bin(">=", Var("iw"), Var("W")),
                                            ),
                                            (Continue(),),
                                        ),
                                        AugAssign(
                                            Var("acc"),
                                            "+",
                                            Bin(
                                                "*",
                                                Load(
                                                    "dy",
                                                    _conv_tap(
                                                        "ni",
                                                        "oi",
                                                        "O",
                                                        "ho",
                                                        "HO",
                                                        "wo",
                                                        "WO",
                                                    ),
                                                ),
                                                Load(
                                                    "x",
                                                    _conv_tap(
                                                        "ni",
                                                        "ci",
                                                        "C",
                                                        "ih",
                                                        "H",
                                                        "iw",
                                                        "W",
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                    incr="++",
                                    type_name="int",
                                ),
                            ),
                            incr="++",
                            type_name="int",
                        ),
                    ),
                    incr="++",
                    type_name="long long",
                ),
                Store("dw", Var("idx"), Var("acc")),
            )
        ),
    )


def _plane_prefix(*, taint: bool) -> tuple:
    """``(long long)n * C + c`` 及其括号形式 —— 图像类内核共用的通道偏移。"""
    head = Bin("+", Bin("*", Cast("long long", Var("n")), Var("C")), Var("c"))
    return (Paren(head),) if taint else (head,)


def im2col() -> Kernel:
    """卷积展开：``NCHW`` -> ``(N, C*KH*KW, HO*WO)`` 的列矩阵。

    列矩阵按 batch 分段（不合并成一个大矩阵），于是 GEMM 的输出已经是
    ``(N, O, HO*WO)``，既不需要 col2im 也不需要额外转置。

    越界采样点写 0：填充在语义上等价于在图像外补零，而不是跳过该列。
    """
    # ((long long)n * C + c) * H * W + (long long)ih * W + iw
    src = Bin(
        "+",
        Bin("*", *_plane_prefix(taint=True), Bin("*", Var("H"), Var("W"))),
        Bin(
            "+",
            Bin("*", Cast("long long", Var("ih")), Var("W")),
            Var("iw"),
        ),
    )
    body = (
        Let(
            "int",
            "wo",
            Cast("int", Paren(Bin("%", Var("idx"), Var("WO")))),
        ),
        Let("long long", "t", Bin("/", Var("idx"), Var("WO"))),
        Let("int", "ho", Cast("int", Paren(Bin("%", Var("t"), Var("HO"))))),
        AugAssign(Var("t"), "/", Var("HO")),
        Let("int", "k", Cast("int", Paren(Bin("%", Var("t"), Var("per_n"))))),
        Let("long long", "n", Bin("/", Var("t"), Var("per_n"))),
        Let("int", "kj", Bin("%", Var("k"), Var("KW"))),
        Let(
            "int",
            "ki",
            Bin("%", Paren(Bin("/", Var("k"), Var("KW"))), Var("KH")),
        ),
        Let("int", "c", Bin("/", Var("k"), Var("plane"))),
        Let(
            "int",
            "ih",
            Bin(
                "+",
                Bin("-", Bin("*", Var("ho"), Var("SH")), Var("PH")),
                Bin("*", Var("ki"), Var("DH")),
            ),
        ),
        Let(
            "int",
            "iw",
            Bin(
                "+",
                Bin("-", Bin("*", Var("wo"), Var("SW")), Var("PW")),
                Bin("*", Var("kj"), Var("DW")),
            ),
        ),
        Let("float", "v", Num("0.f")),
        If(
            Bin(
                "&&",
                Bin(
                    "&&",
                    Bin("&&", Bin(">=", Var("ih"), Num("0")), Bin("<", Var("ih"), Var("H"))),
                    Bin(">=", Var("iw"), Num("0")),
                ),
                Bin("<", Var("iw"), Var("W")),
            ),
            (Assign(Var("v"), Load("x", src)),),
        ),
        Store("cols", Var("idx"), Var("v")),
    )
    return Kernel(
        name="im2col",
        params=(
            Tensor("x"),
            Tensor("cols", read_only=False),
            Scalar("int", "C"),
            Scalar("int", "H"),
            Scalar("int", "W"),
            Scalar("int", "KH"),
            Scalar("int", "KW"),
            Scalar("int", "HO"),
            Scalar("int", "WO"),
            Scalar("int", "SH"),
            Scalar("int", "SW"),
            Scalar("int", "PH"),
            Scalar("int", "PW"),
            Scalar("int", "DH"),
            Scalar("int", "DW"),
            Scalar("long long", "total"),
        ),
        body=(
            Let(
                "const long long",
                "step",
                Bin("*", Cast("long long", Axis("gdim")), Axis("bdim")),
            ),
            Let("const int", "plane", Bin("*", Var("KH"), Var("KW"))),
            Let("const int", "per_n", Bin("*", Var("C"), Var("plane"))),
            For(
                "idx",
                Bin(
                    "+",
                    Bin("*", Cast("long long", Axis("bid")), Axis("bdim")),
                    Axis("tid"),
                ),
                Var("total"),
                Var("step"),
                body,
            ),
        ),
    )


def _tiled_weight() -> Expr:
    """``w[(long long)ocg * K + (long long)(c0 + cc) * khkw + t]``。"""
    return Index(
        "w",
        Bin(
            "+",
            Bin(
                "+",
                Bin("*", Cast("long long", Var("ocg")), Var("K")),
                Bin(
                    "*",
                    Cast(
                        "long long",
                        Paren(Bin("+", Var("c0"), Var("cc"))),
                    ),
                    Var("khkw"),
                ),
            ),
            Var("t"),
        ),
    )


def _column_index(index: Expr, scale) -> Expr:
    """横向坐标的寻址项：NCHW 原样（``scale=None``），NHWC 乘通道数。

    NHWC 下相邻列差 ``C`` 个元素（同一像素的通道连续），因此列下标要先乘
    ``C`` 再加到行指针上；NCHW 下相邻列本就在内存里相邻，乘 1 等于没乘。
    """
    return index if scale is None else Bin("*", Paren(index), scale)


def _nhwc_output_ptr() -> Expr:
    """``out + ((long long)n * HO + ho) * WO * O + wo * O + oc``

    NHWC 的输出像素步长是 ``O``（通道连续），批次步长是 ``HO*WO*O``；
    同一像素内 ``ROC`` 个累加器只差 1。NCHW 的写法见 ``conv2d_tiled`` 里
    的分支——两种布局的输出指针形状不同，但都以 ``OC`` 为最内层。
    """
    return Bin(
        "+",
        Var("out"),
        Bin(
            "+",
            Bin(
                "+",
                Bin(
                    "*",
                    Paren(
                        Bin(
                            "+",
                            Bin("*", Cast("long long", Var("n")), Var("HO")),
                            Var("ho"),
                        )
                    ),
                    Bin("*", Var("WO"), Var("O")),
                ),
                Bin("*", Var("wo"), Var("O")),
            ),
            Var("oc"),
        ),
    )


def _output_writeback(layout: str = "nchw") -> tuple:
    """``dst`` 指针 + 写回 ``ROC`` 个累加器的循环（含 bias 融合）。

    两种布局只差两处：``dst`` 怎么算（见 :func:`_nhwc_output_ptr`），以及
    相邻输出通道之间的步长——NCHW 是 ``HO*WO``，NHWC 是 1。
    """
    nhwc = layout == "nhwc"
    dst = (
        _nhwc_output_ptr()
        if nhwc
        else Bin(
            "+",
            Var("out"),
            Bin(
                "+",
                Bin(
                    "*",
                    Paren(
                        Bin(
                            "+",
                            Bin("*", Cast("long long", Var("n")), Var("O")),
                            Var("oc"),
                        )
                    ),
                    Bin("*", Var("HO"), Var("WO")),
                ),
                Bin(
                    "+",
                    Bin("*", Cast("long long", Var("ho")), Var("WO")),
                    Var("wo"),
                ),
            ),
        )
    )
    step = (
        Cast("long long", Var("j"))
        if nhwc
        else Bin("*", Cast("long long", Var("j")), Bin("*", Var("HO"), Var("WO")))
    )
    return (
        Let("float*", "dst", dst),
        For(
            "j",
            Num("0"),
            Var("ROC"),
            Num("1"),
            (
                If(
                    Bin("<", Bin("+", Var("oc"), Var("j")), Var("O")),
                    (
                        Assign(
                            Index("dst", step),
                            Bin(
                                "+",
                                Index("acc", Var("j")),
                                Paren(
                                    Cond(
                                        Var("has_bias"),
                                        Index("b", Bin("+", Var("oc"), Var("j"))),
                                        Num("0.f"),
                                    )
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            incr="++",
            type_name="int",
        ),
    )


def _tiled_accumulate() -> tuple:
    """内层：``acc[j] += v * wb[off + j * cb * khkw + kj];``"""
    return (
        For(
            "j",
            Num("0"),
            Var("ROC"),
            Num("1"),
            (
                AugAssign(
                    Index("acc", Var("j")),
                    "+",
                    Bin(
                        "*",
                        Var("v"),
                        Index(
                            "wb",
                            Bin(
                                "+",
                                Bin(
                                    "+",
                                    Var("off"),
                                    Bin(
                                        "*",
                                        Bin("*", Var("j"), Var("cb")),
                                        Var("khkw"),
                                    ),
                                ),
                                Var("kj"),
                            ),
                        ),
                    ),
                ),
            ),
            incr="++",
            type_name="int",
        ),
    )


def conv2d_tiled(*, layout: str = "nchw") -> Kernel:
    """直接分块卷积（互相关）：不物化列矩阵的那条路径。

    两级复用：块内 ``TOC`` 个输出通道的权重按 ``CB`` 个输入通道分批暂存到
    共享内存（每批一次屏障，而不是每个通道一次）；一个输入值喂 ``ROC`` 个
    累加器，把全局读次数降到 ``1/ROC``。

    落界的线程不提前退出，而是走带边界的慢路径——``__syncthreads()`` 必须
    被整个 block 到达，提前 ``return`` 会让屏障死锁。

    bias 在写回时融合：``acc + (has_bias ? b[oc] : 0)``。这样调用方不必为
    一次 ``(1, O, 1, 1)`` 的广播加再发一个逐元素核；没有 bias 时传任意有效
    指针并把 ``has_bias`` 置 0，内核不会解引用它（与 attention 的 mask 同法）。

    全程 ``float`` 计算：这个内核不做 dtype 参数化，f16 走 im2col + GEMM。

    ``layout="nhwc"``（5.334）只改三处寻址：通道偏移从 ``* H * W`` 变成
    ``* 1``、行的步长从 ``W`` 变成 ``W*C``、列的步长从 ``1`` 变成 ``C``；
    写回时同一输出像素的 ``O`` 个通道相邻。权重仍是 ``(O, C, KH, KW)``，
    分块与共享内存的复用方式完全不变——两种布局共用这一份描述。
    """
    if layout not in ("nchw", "nhwc"):
        raise ValueError(f"layout 只能是 'nchw' 或 'nhwc'，收到 {layout!r}")
    nhwc = layout == "nhwc"

    def _channel_offset(index):
        """``(long long)(c0+cc)`` 在通道维上的字节偏移倍率。"""
        if nhwc:
            return Cast("long long", Paren(index))
        return Bin(
            "*",
            Bin("*", Cast("long long", Paren(index)), Var("H")),
            Var("W"),
        )

    row_stride = Bin("*", Var("W"), Var("C")) if nhwc else Var("W")
    col_scale = Var("C") if nhwc else None

    grad = Bin(
        "-",
        Bin("*", Var("ho"), Var("SH")),
        Var("PH"),
    )
    inside = Bin(
        "&&",
        Bin(
            "&&",
            Bin(
                "&&",
                Bin(
                    "&&",
                    Var("active"),
                    Bin(">=", Var("ih0"), Num("0")),
                ),
                Bin(
                    "<",
                    Bin(
                        "+",
                        Var("ih0"),
                        Bin(
                            "*",
                            Paren(Bin("-", Var("KH"), Num("1"))),
                            Var("DH"),
                        ),
                    ),
                    Var("H"),
                ),
            ),
            Bin(">=", Var("iw0"), Num("0")),
        ),
        Bin(
            "<",
            Bin(
                "+",
                Var("iw0"),
                Bin(
                    "*",
                    Paren(Bin("-", Var("KW"), Num("1"))),
                    Var("DW"),
                ),
            ),
            Var("W"),
        ),
    )
    return Kernel(
        name="conv2d_tiled_nhwc" if nhwc else "conv2d_tiled",
        params=(
            Tensor("x", type_name="float"),
            Tensor("w", type_name="float"),
            Tensor("out", read_only=False, type_name="float"),
            Tensor("b", type_name="float"),
            Scalar("int", "C"),
            Scalar("int", "H"),
            Scalar("int", "W"),
            Scalar("int", "O"),
            Scalar("int", "KH"),
            Scalar("int", "KW"),
            Scalar("int", "HO"),
            Scalar("int", "WO"),
            Scalar("int", "SH"),
            Scalar("int", "SW"),
            Scalar("int", "PH"),
            Scalar("int", "PW"),
            Scalar("int", "DH"),
            Scalar("int", "DW"),
            Scalar("int", "TOC"),
            Scalar("int", "ROC"),
            Scalar("int", "CB"),
            Scalar("int", "oc_tiles"),
            Scalar("int", "has_bias"),
        ),
        body=(
            Shared("float", "sw", extern=True),
            Let("const int", "nx", Axis("bdim")),
            Let("const int", "nt", Bin("*", Var("nx"), Axis("bdim", "y"))),
            Let(
                "const int",
                "tid",
                Bin("+", Bin("*", Axis("tid", "y"), Var("nx")), Axis("tid")),
            ),
            Let("const int", "khkw", Bin("*", Var("KH"), Var("KW"))),
            Let("const int", "K", Bin("*", Var("C"), Var("khkw"))),
            Let("const int", "tile", Axis("bid", "z")),
            Let(
                "const int",
                "oc0",
                Bin(
                    "*",
                    Paren(Bin("%", Var("tile"), Var("oc_tiles"))),
                    Var("TOC"),
                ),
            ),
            Let("const int", "n", Bin("/", Var("tile"), Var("oc_tiles"))),
            Let("const int", "ho", Axis("bid", "y")),
            Let(
                "const int",
                "wo",
                Bin("+", Bin("*", Axis("bid"), Var("nx")), Axis("tid")),
            ),
            Let(
                "const int",
                "oc",
                Bin(
                    "+",
                    Var("oc0"),
                    Bin("*", Axis("tid", "y"), Var("ROC")),
                ),
            ),
            Let(
                "const bool",
                "active",
                Bin(
                    "&&",
                    Paren(Bin("<", Var("wo"), Var("WO"))),
                    Paren(Bin("<", Var("oc"), Var("O"))),
                ),
            ),
            Let("const int", "ih0", grad),
            Let(
                "const int",
                "iw0",
                Bin("-", Bin("*", Var("wo"), Var("SW")), Var("PW")),
            ),
            Let("const bool", "inside", inside),
            Let(
                "const float*",
                "xn",
                Bin(
                    "+",
                    Var("x"),
                    Bin(
                        "*",
                        Bin("*", Cast("long long", Var("n")), Var("C")),
                        Bin("*", Var("H"), Var("W")),
                    ),
                ),
            ),
            ArrayDecl("float", "acc", 8, None),
            For(
                "j",
                Num("0"),
                Var("ROC"),
                Num("1"),
                (Assign(Index("acc", Var("j")), Num("0.f")),),
                incr="++",
                type_name="int",
            ),
            For(
                "c0",
                Num("0"),
                Var("C"),
                Var("CB"),
                (
                    Sync(),
                    Let(
                        "const int",
                        "cb",
                        Cond(
                            Paren(
                                Bin(
                                    "<",
                                    Bin("-", Var("C"), Var("c0")),
                                    Var("CB"),
                                )
                            ),
                            Paren(Bin("-", Var("C"), Var("c0"))),
                            Var("CB"),
                        ),
                    ),
                    For(
                        "i",
                        Var("tid"),
                        Bin(
                            "*",
                            Bin("*", Var("TOC"), Var("cb")),
                            Var("khkw"),
                        ),
                        Var("nt"),
                        (
                            Let("const int", "t", Bin("%", Var("i"), Var("khkw"))),
                            Let("const int", "rest", Bin("/", Var("i"), Var("khkw"))),
                            Let("const int", "cc", Bin("%", Var("rest"), Var("cb"))),
                            Let(
                                "const int",
                                "ocg",
                                Bin("+", Var("oc0"), Bin("/", Var("rest"), Var("cb"))),
                            ),
                            Store(
                                "sw",
                                Var("i"),
                                Cond(
                                    Bin("<", Var("ocg"), Var("O")),
                                    _tiled_weight(),
                                    Num("0.f"),
                                ),
                                raw=True,
                            ),
                        ),
                        type_name="int",
                    ),
                    Sync(),
                    If(Not(Var("active")), (Continue(),)),
                    If(
                        Var("inside"),
                        (
                            For(
                                "cc",
                                Num("0"),
                                Var("cb"),
                                Num("1"),
                                (
                                    Let(
                                        "const float*",
                                        "xc",
                                        Bin(
                                            "+",
                                            Var("xn"),
                                            _channel_offset(
                                                Bin(
                                                    "+",
                                                    Var("c0"),
                                                    Var("cc"),
                                                )
                                            ),
                                        ),
                                    ),
                                    Let(
                                        "const float*",
                                        "wb",
                                        Bin(
                                            "+",
                                            Var("sw"),
                                            Bin("*", Var("cc"), Var("khkw")),
                                        ),
                                    ),
                                    For(
                                        "ki",
                                        Num("0"),
                                        Var("KH"),
                                        Num("1"),
                                        (
                                            Let(
                                                "const float*",
                                                "xr",
                                                Bin(
                                                    "+",
                                                    Var("xc"),
                                                    Bin(
                                                        "*",
                                                        Cast(
                                                            "long long",
                                                            Paren(
                                                                Bin(
                                                                    "+",
                                                                    Var("ih0"),
                                                                    Bin(
                                                                        "*",
                                                                        Var("ki"),
                                                                        Var("DH"),
                                                                    ),
                                                                )
                                                            ),
                                                        ),
                                                        row_stride,
                                                    ),
                                                ),
                                            ),
                                            Let(
                                                "const int",
                                                "off",
                                                Bin(
                                                    "+",
                                                    Bin(
                                                        "*",
                                                        Bin(
                                                            "*",
                                                            Bin(
                                                                "*",
                                                                Axis("tid", "y"),
                                                                Var("ROC"),
                                                            ),
                                                            Var("cb"),
                                                        ),
                                                        Var("khkw"),
                                                    ),
                                                    Bin("*", Var("ki"), Var("KW")),
                                                ),
                                            ),
                                            For(
                                                "kj",
                                                Num("0"),
                                                Var("KW"),
                                                Num("1"),
                                                (
                                                    Let(
                                                        "const float",
                                                        "v",
                                                        Index(
                                                            "xr",
                                                            _column_index(
                                                                Bin(
                                                                    "+",
                                                                    Var("iw0"),
                                                                    Bin(
                                                                        "*",
                                                                        Var("kj"),
                                                                        Var("DW"),
                                                                    ),
                                                                ),
                                                                col_scale,
                                                            ),
                                                        ),
                                                    ),
                                                )
                                                + _tiled_accumulate(),
                                                incr="++",
                                                type_name="int",
                                            ),
                                        ),
                                        incr="++",
                                        type_name="int",
                                    ),
                                ),
                                incr="++",
                                type_name="int",
                            ),
                        ),
                        (
                            For(
                                "cc",
                                Num("0"),
                                Var("cb"),
                                Num("1"),
                                (
                                    Let(
                                        "const float*",
                                        "xc",
                                        Bin(
                                            "+",
                                            Var("xn"),
                                            Bin(
                                                "*",
                                                Bin(
                                                    "*",
                                                    Cast(
                                                        "long long",
                                                        Paren(
                                                            Bin(
                                                                "+",
                                                                Var("c0"),
                                                                Var("cc"),
                                                            )
                                                        ),
                                                    ),
                                                    Var("H"),
                                                ),
                                                Var("W"),
                                            ),
                                        ),
                                    ),
                                    Let(
                                        "const float*",
                                        "wb",
                                        Bin(
                                            "+",
                                            Var("sw"),
                                            Bin("*", Var("cc"), Var("khkw")),
                                        ),
                                    ),
                                    For(
                                        "ki",
                                        Num("0"),
                                        Var("KH"),
                                        Num("1"),
                                        (
                                            Let(
                                                "const int",
                                                "ih",
                                                Bin(
                                                    "+",
                                                    Var("ih0"),
                                                    Bin(
                                                        "*",
                                                        Var("ki"),
                                                        Var("DH"),
                                                    ),
                                                ),
                                            ),
                                            If(
                                                Bin(
                                                    "||",
                                                    Bin("<", Var("ih"), Num("0")),
                                                    Bin(">=", Var("ih"), Var("H")),
                                                ),
                                                (Continue(),),
                                            ),
                                            Let(
                                                "const float*",
                                                "xr",
                                                Bin(
                                                    "+",
                                                    Var("xc"),
                                                    Bin(
                                                        "*",
                                                        Cast(
                                                            "long long",
                                                            Var("ih"),
                                                        ),
                                                        row_stride,
                                                    ),
                                                ),
                                            ),
                                            Let(
                                                "const int",
                                                "off",
                                                Bin(
                                                    "+",
                                                    Bin(
                                                        "*",
                                                        Bin(
                                                            "*",
                                                            Bin(
                                                                "*",
                                                                Axis("tid", "y"),
                                                                Var("ROC"),
                                                            ),
                                                            Var("cb"),
                                                        ),
                                                        Var("khkw"),
                                                    ),
                                                    Bin("*", Var("ki"), Var("KW")),
                                                ),
                                            ),
                                            For(
                                                "kj",
                                                Num("0"),
                                                Var("KW"),
                                                Num("1"),
                                                (
                                                    Let(
                                                        "const int",
                                                        "iw",
                                                        Bin(
                                                            "+",
                                                            Var("iw0"),
                                                            Bin(
                                                                "*",
                                                                Var("kj"),
                                                                Var("DW"),
                                                            ),
                                                        ),
                                                    ),
                                                    If(
                                                        Bin(
                                                            "||",
                                                            Bin("<", Var("iw"), Num("0")),
                                                            Bin(">=", Var("iw"), Var("W")),
                                                        ),
                                                        (Continue(),),
                                                    ),
                                                    Let(
                                                        "const float",
                                                        "v",
                                                        Index(
                                                            "xr",
                                                            _column_index(
                                                                Var("iw"), col_scale
                                                            ),
                                                        ),
                                                    ),
                                                )
                                                + _tiled_accumulate(),
                                                incr="++",
                                                type_name="int",
                                            ),
                                        ),
                                        incr="++",
                                        type_name="int",
                                    ),
                                ),
                                incr="++",
                                type_name="int",
                            ),
                        ),
                    ),
                ),
                type_name="int",
            ),
            If(Not(Var("active")), (Return(),)),
            *_output_writeback(layout),
        ),
    )


def conv2d_grouped() -> Kernel:
    """分组卷积（含 depthwise）：一个线程一个输出元素，权重直接从全局读。

    5.165 之前 groups>1 是"逐组切片 + im2col + GEMM"：每个组物化一份
    ``(N, CG*KH*KW, HO*WO)`` 的列矩阵，再发一次极瘦的 GEMM。depthwise 下
    GEMM 是 ``(N*HO*WO, KH*KW) × (KH*KW, 1)``，cuBLAS 的效率接近最坏情况，
    而列矩阵的读写量却是真正的数倍。

    这里不再物化任何中间张量：每个线程按 ``(n, o, ho, wo)`` 取出自己那一个
    累加器，只遍历它所属组的 ``CG * KH * KW`` 个乘加。同一 block 的相邻线程
    共享输出通道与权重，L1 会替我们完成复用；分组卷积的 ``CG`` 通常很小
    （depthwise 是 1），没有必要为它再搭一层 shared staging。

    参数里的 ``C`` 是**输出**通道数（``NCHW`` 的输出张量），输入通道数由
    ``GROUPS * CG`` 给出；权重布局是 ``(O, CG, KH, KW)``，组内输出通道的下标
    是 ``o - (o / OG) * OG``，权重偏移因此只用 ``o * CG`` 就能算对。

    bias 同样在写回时融合；没有 bias 时传任意有效指针并把 ``has_bias`` 置 0。

    全程 ``float``：f16 仍走 im2col + hgemm（与分块核、隐式核的边界一致）。
    """
    return Kernel(
        name="conv2d_grouped",
        params=(
            Tensor("x", type_name="float"),
            Tensor("w", type_name="float"),
            Tensor("out", read_only=False, type_name="float"),
            Tensor("b", type_name="float"),
            Scalar("int", "C"),
            Scalar("int", "H"),
            Scalar("int", "W"),
            Scalar("int", "KH"),
            Scalar("int", "KW"),
            Scalar("int", "HO"),
            Scalar("int", "WO"),
            Scalar("int", "SH"),
            Scalar("int", "SW"),
            Scalar("int", "PH"),
            Scalar("int", "PW"),
            Scalar("int", "DH"),
            Scalar("int", "DW"),
            Scalar("int", "CG"),
            Scalar("int", "OG"),
            Scalar("int", "GROUPS"),
            Scalar("int", "has_bias"),
            Scalar("long long", "total"),
        ),
        body=_flat_loop(
            (
                Let(
                    "int",
                    "wo",
                    Cast(
                        "int",
                        Paren(Bin("%", Var("idx"), Var("WO"))),
                    ),
                ),
                Let("long long", "t", Bin("/", Var("idx"), Var("WO"))),
                Let(
                    "int",
                    "ho",
                    Cast(
                        "int",
                        Paren(Bin("%", Var("t"), Var("HO"))),
                    ),
                ),
                AugAssign(Var("t"), "/", Var("HO")),
                Let(
                    "int",
                    "o",
                    Cast(
                        "int",
                        Paren(Bin("%", Var("t"), Var("C"))),
                    ),
                ),
                Let("long long", "n", Bin("/", Var("t"), Var("C"))),
                Let("int", "g", Bin("/", Var("o"), Var("OG"))),
                Let("int", "cstart", Bin("*", Var("g"), Var("CG"))),
                Let(
                    "float",
                    "acc",
                    Cond(
                        Var("has_bias"),
                        Load("b", Var("o")),
                        Num("0.f"),
                    ),
                ),
                For(
                    "cg",
                    Num("0"),
                    Var("CG"),
                    Num("1"),
                    (
                        Ptr(
                            "base",
                            "x",
                            Bin(
                                "*",
                                Paren(
                                    Bin(
                                        "+",
                                        Bin(
                                            "*",
                                            Cast("long long", Var("n")),
                                            Bin("*", Var("GROUPS"), Var("CG")),
                                        ),
                                        Bin("+", Var("cstart"), Var("cg")),
                                    )
                                ),
                                Bin("*", Cast("long long", Var("H")), Var("W")),
                            ),
                        ),
                        Ptr(
                            "wb",
                            "w",
                            Bin(
                                "*",
                                Paren(
                                    Bin(
                                        "+",
                                        Bin(
                                            "*",
                                            Cast("long long", Var("o")),
                                            Var("CG"),
                                        ),
                                        Var("cg"),
                                    )
                                ),
                                Bin("*", Cast("long long", Var("KH")), Var("KW")),
                            ),
                        ),
                        For(
                            "ki",
                            Num("0"),
                            Var("KH"),
                            Num("1"),
                            (
                                Let(
                                    "int",
                                    "ih",
                                    Bin(
                                        "+",
                                        Bin(
                                            "-",
                                            Bin(
                                                "*",
                                                Var("ho"),
                                                Var("SH"),
                                            ),
                                            Var("PH"),
                                        ),
                                        Bin("*", Var("ki"), Var("DH")),
                                    ),
                                ),
                                If(
                                    Bin(
                                        "||",
                                        Bin("<", Var("ih"), Num("0")),
                                        Bin(">=", Var("ih"), Var("H")),
                                    ),
                                    (Continue(),),
                                ),
                                For(
                                    "kj",
                                    Num("0"),
                                    Var("KW"),
                                    Num("1"),
                                    (
                                        Let(
                                            "int",
                                            "iw",
                                            Bin(
                                                "+",
                                                Bin(
                                                    "-",
                                                    Bin(
                                                        "*",
                                                        Var("wo"),
                                                        Var("SW"),
                                                    ),
                                                    Var("PW"),
                                                ),
                                                Bin("*", Var("kj"), Var("DW")),
                                            ),
                                        ),
                                        If(
                                            Bin(
                                                "||",
                                                Bin("<", Var("iw"), Num("0")),
                                                Bin(">=", Var("iw"), Var("W")),
                                            ),
                                            (Continue(),),
                                        ),
                                        Let(
                                            "float",
                                            "v",
                                            Load("base", _tap().index),
                                        ),
                                        AugAssign(
                                            Var("acc"),
                                            "+",
                                            Bin(
                                                "*",
                                                Var("v"),
                                                Load(
                                                    "wb",
                                                    Bin(
                                                        "+",
                                                        Bin(
                                                            "*",
                                                            Cast(
                                                                "long long",
                                                                Var("ki"),
                                                            ),
                                                            Var("KW"),
                                                        ),
                                                        Var("kj"),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                    incr="++",
                                    type_name="int",
                                ),
                            ),
                            incr="++",
                            type_name="int",
                        ),
                    ),
                    incr="++",
                    type_name="int",
                ),
                Store("out", Var("idx"), Var("acc")),
            )
        ),
    )


def conv2d_implicit() -> Kernel:
    """隐式 GEMM 卷积：输入窗口与权重一起进 shared，累加阶段不再读全局。

    与 ``conv2d_tiled`` 只有两处不同：

    1. 每个输入通道的 ``KH x SPAN`` 窗口在累加前搬进 shared，块内 ``tile_w``
       个输出列共享这段窗口。分块直算每算一个输出像素要沿 K 读一遍全局，
       全局读次数正比于 ``tile_w``；这里降成每个 block 每通道只读一次窗口，
       K 越大差距越大（DESIGN 5.16 里 K=576 慢 54% 的那一段）；
    2. 边界不在累加里逐点判断，而是在 staging 时把越界位置填 0。内层循环
       因此没有 ``if``——K 大时这几百次分支本身也是成本。

    两条路径共用 ``_tiled_weight`` / ``_tiled_accumulate``，累加顺序完全一致，
    所以换实现不改变浮点结果（同一批乘加按同一顺序累加）。
    """
    # 输入窗口里第 j 列的全局列号：iw_base + j
    window_iw = Bin("+", Var("iw_base"), Var("j"))
    # 该列的全局行号：ih0 + ki * DH
    window_ih = Bin("+", Var("ih0"), Bin("*", Var("ki"), Var("DH")))
    inside_window = Bin(
        "&&",
        Paren(
            Bin(
                "&&",
                Paren(Bin(">=", window_ih, Num("0"))),
                Paren(Bin("<", window_ih, Var("H"))),
            )
        ),
        Paren(
            Bin(
                "&&",
                Paren(Bin(">=", window_iw, Num("0"))),
                Paren(Bin("<", window_iw, Var("W"))),
            )
        ),
    )
    source = Index(
        "xn",
        Bin(
            "+",
            Bin(
                "*",
                Cast(
                    "long long",
                    Paren(Bin("+", Var("c0"), Var("cc"))),
                ),
                Bin("*", Var("H"), Var("W")),
            ),
            Bin(
                "+",
                Bin("*", Cast("long long", Paren(window_ih)), Var("W")),
                window_iw,
            ),
        ),
    )
    return Kernel(
        name="conv2d_implicit",
        params=(
            Tensor("x", type_name="float"),
            Tensor("w", type_name="float"),
            Tensor("out", read_only=False, type_name="float"),
            Tensor("b", type_name="float"),
            Scalar("int", "C"),
            Scalar("int", "H"),
            Scalar("int", "W"),
            Scalar("int", "O"),
            Scalar("int", "KH"),
            Scalar("int", "KW"),
            Scalar("int", "HO"),
            Scalar("int", "WO"),
            Scalar("int", "SH"),
            Scalar("int", "SW"),
            Scalar("int", "PH"),
            Scalar("int", "PW"),
            Scalar("int", "DH"),
            Scalar("int", "DW"),
            Scalar("int", "TOC"),
            Scalar("int", "ROC"),
            Scalar("int", "CB"),
            Scalar("int", "oc_tiles"),
            Scalar("int", "has_bias"),
            Scalar("int", "SPAN"),
        ),
        body=(
            Shared("float", "sw", extern=True),
            Let("const int", "nx", Axis("bdim")),
            Let("const int", "nt", Bin("*", Var("nx"), Axis("bdim", "y"))),
            Let(
                "const int",
                "tid",
                Bin("+", Bin("*", Axis("tid", "y"), Var("nx")), Axis("tid")),
            ),
            Let("const int", "khkw", Bin("*", Var("KH"), Var("KW"))),
            Let("const int", "K", Bin("*", Var("C"), Var("khkw"))),
            Let("const int", "tile", Axis("bid", "z")),
            Let(
                "const int",
                "oc0",
                Bin(
                    "*",
                    Paren(Bin("%", Var("tile"), Var("oc_tiles"))),
                    Var("TOC"),
                ),
            ),
            Let("const int", "n", Bin("/", Var("tile"), Var("oc_tiles"))),
            Let("const int", "ho", Axis("bid", "y")),
            Let(
                "const int",
                "wo",
                Bin("+", Bin("*", Axis("bid"), Var("nx")), Axis("tid")),
            ),
            Let(
                "const int",
                "oc",
                Bin("+", Var("oc0"), Bin("*", Axis("tid", "y"), Var("ROC"))),
            ),
            Let(
                "const bool",
                "active",
                Bin(
                    "&&",
                    Paren(Bin("<", Var("wo"), Var("WO"))),
                    Paren(Bin("<", Var("oc"), Var("O"))),
                ),
            ),
            Let(
                "const int",
                "ih0",
                Bin("-", Bin("*", Var("ho"), Var("SH")), Var("PH")),
            ),
            Let(
                "const int",
                "iw_base",
                Bin(
                    "-",
                    Bin("*", Bin("*", Axis("bid"), Var("nx")), Var("SW")),
                    Var("PW"),
                ),
            ),
            Let(
                "const float*",
                "xn",
                Bin(
                    "+",
                    Var("x"),
                    Bin(
                        "*",
                        Bin("*", Cast("long long", Var("n")), Var("C")),
                        Bin("*", Var("H"), Var("W")),
                    ),
                ),
            ),
            # 输入窗口紧跟在权重之后：shared 只声明一块，偏移由 TOC/CB/khkw 算
            Let(
                "float*",
                "sx",
                Bin(
                    "+",
                    Var("sw"),
                    Bin("*", Bin("*", Var("TOC"), Var("CB")), Var("khkw")),
                ),
            ),
            ArrayDecl("float", "acc", 8, None),
            For(
                "j",
                Num("0"),
                Var("ROC"),
                Num("1"),
                (Assign(Index("acc", Var("j")), Num("0.f")),),
                incr="++",
                type_name="int",
            ),
            For(
                "c0",
                Num("0"),
                Var("C"),
                Var("CB"),
                (
                    Sync(),
                    Let(
                        "const int",
                        "cb",
                        Cond(
                            Paren(
                                Bin(
                                    "<",
                                    Bin("-", Var("C"), Var("c0")),
                                    Var("CB"),
                                )
                            ),
                            Paren(Bin("-", Var("C"), Var("c0"))),
                            Var("CB"),
                        ),
                    ),
                    # ---- 权重 staging：与分块直算逐字相同 ----
                    For(
                        "i",
                        Var("tid"),
                        Bin(
                            "*",
                            Bin("*", Var("TOC"), Var("cb")),
                            Var("khkw"),
                        ),
                        Var("nt"),
                        (
                            Let("const int", "t", Bin("%", Var("i"), Var("khkw"))),
                            Let("const int", "rest", Bin("/", Var("i"), Var("khkw"))),
                            Let("const int", "cc", Bin("%", Var("rest"), Var("cb"))),
                            Let(
                                "const int",
                                "ocg",
                                Bin("+", Var("oc0"), Bin("/", Var("rest"), Var("cb"))),
                            ),
                            Store(
                                "sw",
                                Var("i"),
                                Cond(
                                    Bin("<", Var("ocg"), Var("O")),
                                    _tiled_weight(),
                                    Num("0.f"),
                                ),
                                raw=True,
                            ),
                        ),
                        type_name="int",
                    ),
                    # ---- 输入窗口 staging：越界填 0，累加阶段不再判边界 ----
                    For(
                        "i",
                        Var("tid"),
                        Bin("*", Bin("*", Var("cb"), Var("KH")), Var("SPAN")),
                        Var("nt"),
                        (
                            Let("const int", "j", Bin("%", Var("i"), Var("SPAN"))),
                            Let("const int", "rest", Bin("/", Var("i"), Var("SPAN"))),
                            Let("const int", "ki", Bin("%", Var("rest"), Var("KH"))),
                            Let("const int", "cc", Bin("/", Var("rest"), Var("KH"))),
                            Assign(
                                Index("sx", Var("i")),
                                Cond(inside_window, source, Num("0.f")),
                            ),
                        ),
                        type_name="int",
                    ),
                    Sync(),
                    If(Not(Var("active")), (Continue(),)),
                    For(
                        "cc",
                        Num("0"),
                        Var("cb"),
                        Num("1"),
                        (
                            Let(
                                "const float*",
                                "wb",
                                Bin(
                                    "+",
                                    Var("sw"),
                                    Bin("*", Var("cc"), Var("khkw")),
                                ),
                            ),
                            For(
                                "ki",
                                Num("0"),
                                Var("KH"),
                                Num("1"),
                                (
                                    Let(
                                        "const float*",
                                        "sr",
                                        Bin(
                                            "+",
                                            Bin(
                                                "+",
                                                Var("sx"),
                                                Bin(
                                                    "*",
                                                    Paren(
                                                        Bin(
                                                            "+",
                                                            Bin(
                                                                "*",
                                                                Var("cc"),
                                                                Var("KH"),
                                                            ),
                                                            Var("ki"),
                                                        )
                                                    ),
                                                    Var("SPAN"),
                                                ),
                                            ),
                                            Bin("*", Axis("tid"), Var("SW")),
                                        ),
                                    ),
                                    Let(
                                        "const int",
                                        "off",
                                        Bin(
                                            "+",
                                            Bin(
                                                "*",
                                                Bin(
                                                    "*",
                                                    Bin(
                                                        "*",
                                                        Axis("tid", "y"),
                                                        Var("ROC"),
                                                    ),
                                                    Var("cb"),
                                                ),
                                                Var("khkw"),
                                            ),
                                            Bin("*", Var("ki"), Var("KW")),
                                        ),
                                    ),
                                    For(
                                        "kj",
                                        Num("0"),
                                        Var("KW"),
                                        Num("1"),
                                        (
                                            Let(
                                                "const float",
                                                "v",
                                                Index(
                                                    "sr",
                                                    Bin("*", Var("kj"), Var("DW")),
                                                ),
                                            ),
                                        )
                                        + _tiled_accumulate(),
                                        incr="++",
                                        type_name="int",
                                    ),
                                ),
                                incr="++",
                                type_name="int",
                            ),
                        ),
                        incr="++",
                        type_name="int",
                    ),
                ),
                type_name="int",
            ),
            If(Not(Var("active")), (Return(),)),
            *_output_writeback(),
        ),
    )


# ---------------------------------------------------------------------------
# Winograd F(2x2, 3x3)
# ---------------------------------------------------------------------------
#
# 三个内核合起来是一次 F(2x2,3x3) 卷积：输入变换 ``V = B^T d B``、16 个
# ``(O,K)x(K,T)`` 的批量 GEMM（走 cuBLAS，不在这里）、输出变换
# ``Y = A^T M A``。逐元素乘那一步不写内核，只是 16 个 GEMM 的 batch 维，
# 所以这里的两个内核都按 ``(16, C, T)`` / ``(16, O, T)`` 的布局写：
# 变换位置 ``p`` 是最外维，同一次 GEMM 的 batch 各是一块连续的 ``(C, T)``。
#
# 索引约定（三处一致，改一处就要改三处）：
# - ``t`` 是 tile 的全局序号，``n = t / (TH*TW)``、``r = t % (TH*TW)``、
#   ``th = r / TW``、``tw = r % TW``；
# - 输入 / 输出的 ``idx`` 分别是 ``c * NT + t`` 与 ``o * NT + t``；
# - ``p = ii * 4 + jj`` 按行优先编号 4x4 变换位置。
#
# 只覆盖 f32、stride=1、dilation=1、groups=1、3x3 核。判据与派发在
# ``backends/device_backend.py::_conv2d_winograd``。


def _wino_inside(ih: Expr, iw: Expr) -> Expr:
    """窗口内的采样点：``0 <= ih < H && 0 <= iw < W``（打印器不推断优先级）。"""
    return Bin(
        "&&",
        Paren(
            Bin(
                "&&",
                Paren(Bin(">=", ih, Num("0"))),
                Paren(Bin("<", ih, Var("H"))),
            )
        ),
        Paren(
            Bin(
                "&&",
                Paren(Bin(">=", iw, Num("0"))),
                Paren(Bin("<", iw, Var("W"))),
            )
        ),
    )


def _wino_tile_lets(outer: str) -> tuple:
    """``i = outer * NT + t`` 拆成 ``(outer, t, n, th, tw)``。

    ``outer`` 在输入变换里是输入通道 ``c``、在输出变换里是输出通道 ``o``。
    依赖 ``grid_stride`` 声明的循环变量 ``i``、标量 ``NT`` 与
    ``per_n = TH*TW``。
    """
    return (
        Let(
            "const int",
            outer,
            Cast("int", Paren(Bin("/", Var("i"), Var("NT")))),
        ),
        Let(
            "const int",
            "t",
            Cast("int", Paren(Bin("%", Var("i"), Var("NT")))),
        ),
        Let("const int", "n", Bin("/", Var("t"), Var("per_n"))),
        Let("const int", "r", Bin("%", Var("t"), Var("per_n"))),
        Let("const int", "th", Bin("/", Var("r"), Var("TW"))),
        Let("const int", "tw", Bin("%", Var("r"), Var("TW"))),
    )


def _wino_cell(name: str, row: Expr, col: Expr) -> Expr:
    """4x4 局部数组的 ``name[row*4 + col]``。"""
    return Index(name, Bin("+", Bin("*", row, Num("4")), col))


def _wino_pos(row: Expr, col: Expr) -> Expr:
    """变换位置 ``p = row * 4 + col``（0..15）。"""
    return Bin("+", Bin("*", row, Num("4")), col)


def winograd_input_transform() -> Kernel:
    """输入变换：``V = B^T d B``，一个线程一个 ``(c, tile)``。

    ``d`` 是 4x4 输入窗口（stride=1、dilation=1、3x3 核），把输出 tile 的
    起点映射回输入后越界的位置读 0——与 im2col 的"填充即补零"一致。两次
    4x4 乘法的系数只有 ``±1``，全部展开成加减，没有乘法。

    写回是 ``(16, C, NT)`` 里的 16 个独立位置：相邻线程的 ``t`` 连续，
    所以每一路写都是合并的；读也是（同一 warp 读同一行相邻 tile）。
    """
    ih = Bin("-", Bin("+", Bin("*", Var("th"), Num("2")), Var("ii")), Var("PH"))
    iw = Bin("-", Bin("+", Bin("*", Var("tw"), Num("2")), Var("jj")), Var("PW"))
    src = Bin(
        "+",
        Var("base"),
        Bin("+", Bin("*", Cast("long long", Paren(ih)), Var("W")), iw),
    )
    to_v = lambda p: Bin("+", Bin("*", Cast("long long", Paren(p)), Var("plane")), Var("i"))
    plane = Bin("*", Cast("long long", Var("C")), Var("NT"))
    body = (
        Let("const int", "NT", Bin("*", Bin("*", Var("N"), Var("TH")), Var("TW"))),
        Let("const int", "per_n", Bin("*", Var("TH"), Var("TW"))),
        Let("const long long", "plane", plane),
        *_wino_tile_lets("c"),
        Let(
            "const long long",
            "base",
            Bin(
                "*",
                Paren(
                    Bin(
                        "+",
                        Bin("*", Cast("long long", Var("n")), Var("C")),
                        Var("c"),
                    )
                ),
                Bin("*", Cast("long long", Var("H")), Var("W")),
            ),
        ),
        ArrayDecl("float", "d", 16, None),
        ArrayDecl("float", "tmp", 16, None),
        For(
            "ii",
            Num("0"),
            Num("4"),
            Num("1"),
            (
                For(
                    "jj",
                    Num("0"),
                    Num("4"),
                    Num("1"),
                    (
                        Let("const int", "ih", ih),
                        Let("const int", "iw", iw),
                        Assign(
                            _wino_cell("d", Var("ii"), Var("jj")),
                            Cond(
                                _wino_inside(Var("ih"), Var("iw")),
                                Index("x", src),
                                Num("0.f"),
                            ),
                        ),
                    ),
                    type_name="int",
                ),
            ),
            type_name="int",
        ),
        # 列方向：tmp = B^T d
        For(
            "jj",
            Num("0"),
            Num("4"),
            Num("1"),
            (
                Assign(
                    _wino_cell("tmp", Num("0"), Var("jj")),
                    Bin(
                        "-",
                        _wino_cell("d", Num("0"), Var("jj")),
                        _wino_cell("d", Num("2"), Var("jj")),
                    ),
                ),
                Assign(
                    _wino_cell("tmp", Num("1"), Var("jj")),
                    Bin(
                        "+",
                        _wino_cell("d", Num("1"), Var("jj")),
                        _wino_cell("d", Num("2"), Var("jj")),
                    ),
                ),
                Assign(
                    _wino_cell("tmp", Num("2"), Var("jj")),
                    Bin(
                        "-",
                        _wino_cell("d", Num("2"), Var("jj")),
                        _wino_cell("d", Num("1"), Var("jj")),
                    ),
                ),
                Assign(
                    _wino_cell("tmp", Num("3"), Var("jj")),
                    Bin(
                        "-",
                        _wino_cell("d", Num("1"), Var("jj")),
                        _wino_cell("d", Num("3"), Var("jj")),
                    ),
                ),
            ),
            type_name="int",
        ),
        # 行方向：V = tmp B，直接写回，不需要第三块数组
        For(
            "ii",
            Num("0"),
            Num("4"),
            Num("1"),
            (
                Store(
                    "v",
                    to_v(_wino_pos(Var("ii"), Num("0"))),
                    Bin(
                        "-",
                        _wino_cell("tmp", Var("ii"), Num("0")),
                        _wino_cell("tmp", Var("ii"), Num("2")),
                    ),
                    raw=True,
                ),
                Store(
                    "v",
                    to_v(_wino_pos(Var("ii"), Num("1"))),
                    Bin(
                        "+",
                        _wino_cell("tmp", Var("ii"), Num("1")),
                        _wino_cell("tmp", Var("ii"), Num("2")),
                    ),
                    raw=True,
                ),
                Store(
                    "v",
                    to_v(_wino_pos(Var("ii"), Num("2"))),
                    Bin(
                        "-",
                        _wino_cell("tmp", Var("ii"), Num("2")),
                        _wino_cell("tmp", Var("ii"), Num("1")),
                    ),
                    raw=True,
                ),
                Store(
                    "v",
                    to_v(_wino_pos(Var("ii"), Num("3"))),
                    Bin(
                        "-",
                        _wino_cell("tmp", Var("ii"), Num("1")),
                        _wino_cell("tmp", Var("ii"), Num("3")),
                    ),
                    raw=True,
                ),
            ),
            type_name="int",
        ),
    )
    return Kernel(
        name="winograd_input",
        params=(
            Tensor("x", type_name="float"),
            Tensor("v", read_only=False, type_name="float"),
            Scalar("int", "N"),
            Scalar("int", "C"),
            Scalar("int", "H"),
            Scalar("int", "W"),
            Scalar("int", "TH"),
            Scalar("int", "TW"),
            Scalar("int", "PH"),
            Scalar("int", "PW"),
            Scalar("long long", "total"),
        ),
        body=grid_stride(Var("total"), *body),
    )


def winograd_weight_transform() -> Kernel:
    """权重变换：``U = G g G^T``，一个线程一个 ``(o, c)``。

    ``G`` 的中间两行带 ``1/2``，所以这一步有乘法；但规模只有 ``O*C`` 个
    线程、每个 9 读 16 写，与 GEMM 相比可以忽略。

    输出布局 ``(16, O, C)`` 与输入变换的 ``(16, C, NT)`` 对齐，两者一起
    喂给一次 ``batch=16`` 的 strided GEMM。
    """
    g = lambda row, col: Index(
        "w",
        Bin(
            "+",
            Bin("*", Cast("long long", Var("i")), Num("9")),
            Bin("+", Bin("*", row, Num("3")), col),
        ),
    )
    to_u = lambda p: Bin("+", Bin("*", Cast("long long", Paren(p)), Var("oplane")), Var("i"))
    body = (
        Let(
            "const long long",
            "oplane",
            Bin("*", Cast("long long", Var("O")), Var("C")),
        ),
        ArrayDecl("float", "t", 16, None),
        # 行方向：tmp = G g，只占前 4 行（G 是 4x3）
        For(
            "jj",
            Num("0"),
            Num("3"),
            Num("1"),
            (
                Assign(
                    _wino_cell("t", Num("0"), Var("jj")),
                    g(Num("0"), Var("jj")),
                ),
                Assign(
                    _wino_cell("t", Num("1"), Var("jj")),
                    Bin(
                        "*",
                        Num("0.5f"),
                        Paren(
                            Bin(
                                "+",
                                Bin(
                                    "+",
                                    g(Num("0"), Var("jj")),
                                    g(Num("1"), Var("jj")),
                                ),
                                g(Num("2"), Var("jj")),
                            )
                        ),
                    ),
                ),
                Assign(
                    _wino_cell("t", Num("2"), Var("jj")),
                    Bin(
                        "*",
                        Num("0.5f"),
                        Paren(
                            Bin(
                                "+",
                                Bin(
                                    "-",
                                    g(Num("0"), Var("jj")),
                                    g(Num("1"), Var("jj")),
                                ),
                                g(Num("2"), Var("jj")),
                            )
                        ),
                    ),
                ),
                Assign(
                    _wino_cell("t", Num("3"), Var("jj")),
                    g(Num("2"), Var("jj")),
                ),
            ),
            type_name="int",
        ),
        # 列方向：U = t G^T
        For(
            "ii",
            Num("0"),
            Num("4"),
            Num("1"),
            (
                Store(
                    "u",
                    to_u(_wino_pos(Var("ii"), Num("0"))),
                    _wino_cell("t", Var("ii"), Num("0")),
                    raw=True,
                ),
                Store(
                    "u",
                    to_u(_wino_pos(Var("ii"), Num("1"))),
                    Bin(
                        "*",
                        Num("0.5f"),
                        Paren(
                            Bin(
                                "+",
                                Bin(
                                    "+",
                                    _wino_cell("t", Var("ii"), Num("0")),
                                    _wino_cell("t", Var("ii"), Num("1")),
                                ),
                                _wino_cell("t", Var("ii"), Num("2")),
                            )
                        ),
                    ),
                    raw=True,
                ),
                Store(
                    "u",
                    to_u(_wino_pos(Var("ii"), Num("2"))),
                    Bin(
                        "*",
                        Num("0.5f"),
                        Paren(
                            Bin(
                                "+",
                                Bin(
                                    "-",
                                    _wino_cell("t", Var("ii"), Num("0")),
                                    _wino_cell("t", Var("ii"), Num("1")),
                                ),
                                _wino_cell("t", Var("ii"), Num("2")),
                            )
                        ),
                    ),
                    raw=True,
                ),
                Store(
                    "u",
                    to_u(_wino_pos(Var("ii"), Num("3"))),
                    _wino_cell("t", Var("ii"), Num("2")),
                    raw=True,
                ),
            ),
            type_name="int",
        ),
    )
    return Kernel(
        name="winograd_weight",
        params=(
            Tensor("w", type_name="float"),
            Tensor("u", read_only=False, type_name="float"),
            Scalar("int", "O"),
            Scalar("int", "C"),
            Scalar("long long", "total"),
        ),
        body=grid_stride(Var("total"), *body),
    )


def winograd_output_transform(epilogue=None, extra=()) -> Kernel:
    """输出变换：``Y = A^T M A`` + bias，一个线程一个 ``(o, tile)``。

    读 16 个 ``M``（每路相隔 ``O*NT``，同一 warp 内 ``t`` 连续所以每一路
    都是合并读），写 4 个输出像素。``A^T`` 的系数只有 ``±1``，两次 4x4
    乘法同样展开成加减。

    越界的输出位置（``HO`` / ``WO`` 为奇数时最后一个 tile 的右 / 下沿）
    不写——那些位置根本不存在，算出来也只是溢出。
    """
    m = lambda row, col: Index(
        "m",
        Bin(
            "+",
            Bin(
                "*",
                Cast("long long", Paren(_wino_pos(row, col))),
                Var("oplane"),
            ),
            Var("i"),
        ),
    )
    to_y = lambda oh, ow: Bin(
        "+",
        Var("obase"),
        Bin(
            "+",
            Bin("*", Cast("long long", Paren(oh)), Var("WO")),
            ow,
        ),
    )
    y00 = lambda row: Bin(
        "+",
        Bin(
            "+",
            _wino_cell("tmp", row, Num("0")),
            _wino_cell("tmp", row, Num("1")),
        ),
        _wino_cell("tmp", row, Num("2")),
    )
    y01 = lambda row: Bin(
        "-",
        Bin(
            "-",
            _wino_cell("tmp", row, Num("1")),
            _wino_cell("tmp", row, Num("2")),
        ),
        _wino_cell("tmp", row, Num("3")),
    )

    def epi(value, channel):
        """写回前的 epilogue：把后继逐元素链（BN / 激活）内联在这里。

        默认 ``None`` 时原样返回，生成的源码与没有这个参数时逐字相同。
        """
        return value if epilogue is None else epilogue(value, channel)

    body = (
        Let("const int", "per_n", Bin("*", Var("TH"), Var("TW"))),
        Let(
            "const long long",
            "oplane",
            Bin("*", Cast("long long", Var("O")), Var("NT")),
        ),
        *_wino_tile_lets("o"),
        Let(
            "const long long",
            "obase",
            Bin(
                "*",
                Paren(
                    Bin(
                        "+",
                        Bin("*", Cast("long long", Var("n")), Var("O")),
                        Var("o"),
                    )
                ),
                Bin("*", Cast("long long", Var("HO")), Var("WO")),
            ),
        ),
        Let(
            "const float",
            "bv",
            Cond(Var("has_bias"), Index("b", Var("o")), Num("0.f")),
        ),
        ArrayDecl("float", "tmp", 16, None),
        # 列方向：tmp = A^T M（只有前两行）
        For(
            "jj",
            Num("0"),
            Num("4"),
            Num("1"),
            (
                Assign(
                    _wino_cell("tmp", Num("0"), Var("jj")),
                    Bin(
                        "+",
                        Bin(
                            "+",
                            m(Num("0"), Var("jj")),
                            m(Num("1"), Var("jj")),
                        ),
                        m(Num("2"), Var("jj")),
                    ),
                ),
                Assign(
                    _wino_cell("tmp", Num("1"), Var("jj")),
                    Bin(
                        "-",
                        Bin(
                            "-",
                            m(Num("1"), Var("jj")),
                            m(Num("2"), Var("jj")),
                        ),
                        m(Num("3"), Var("jj")),
                    ),
                ),
            ),
            type_name="int",
        ),
        # 行方向：Y = tmp A，越界不写
        For(
            "ii",
            Num("0"),
            Num("2"),
            Num("1"),
            (
                For(
                    "jj",
                    Num("0"),
                    Num("2"),
                    Num("1"),
                    (
                        Let(
                            "const int",
                            "oh",
                            Bin(
                                "+",
                                Bin("*", Var("th"), Num("2")),
                                Var("ii"),
                            ),
                        ),
                        Let(
                            "const int",
                            "ow",
                            Bin(
                                "+",
                                Bin("*", Var("tw"), Num("2")),
                                Var("jj"),
                            ),
                        ),
                        If(
                            Bin(
                                "&&",
                                Paren(Bin("<", Var("oh"), Var("HO"))),
                                Paren(Bin("<", Var("ow"), Var("WO"))),
                            ),
                            (
                                Store(
                                    "y",
                                    to_y(Var("oh"), Var("ow")),
                                    epi(
                                        Bin(
                                            "+",
                                            Paren(
                                                Cond(
                                                    Bin("==", Var("jj"), Num("0")),
                                                    Paren(y00(Var("ii"))),
                                                    Paren(y01(Var("ii"))),
                                                )
                                            ),
                                            Var("bv"),
                                        ),
                                        Var("o"),
                                    ),
                                    raw=True,
                                ),
                            ),
                        ),
                    ),
                    type_name="int",
                ),
            ),
            type_name="int",
        ),
    )
    return Kernel(
        name="winograd_output",
        params=(
            Tensor("m", type_name="float"),
            Tensor("b", type_name="float"),
            Tensor("y", read_only=False, type_name="float"),
            Scalar("int", "N"),
            Scalar("int", "O"),
            Scalar("int", "HO"),
            Scalar("int", "WO"),
            Scalar("int", "TH"),
            Scalar("int", "TW"),
            Scalar("int", "NT"),
            Scalar("int", "has_bias"),
            Scalar("long long", "total"),
            *(Tensor(name, type_name="float") for name, _ in extra),
        ),
        body=grid_stride(Var("total"), *body),
    )


def winograd_fused(epilogue=None, extra=()) -> Kernel:
    """融合版 F(2x2,3x3)：一次 launch 走完输入变换、乘加与输出变换。

    4 段版（``winograd_input`` + 16 个 cuBLAS GEMM + ``winograd_output``）
    在 A800 上被固定开销支配：c3/c8/c16/c24 四档的稳态几乎都是 0.103 ms，
    与通道数无关——四段各付一次 kernel 延迟（每段 0.015～0.019 ms）与一次
    主机提交。融合版把这三段压进同一个内核，且 ``V`` 只留在共享内存里
    （``(16, C, TT)``，TT 是每块的 tile 数），不再物化 ``(16, O, NT)`` 的
    ``M``。实测 3x3_c8 的 device 时间从 0.1015 ms 降到 0.0199 ms。

    线程分工：

    - 第一阶段整个 block 协作算 ``V``：一个线程一个 ``(c, tile)``，写进
      ``sv[(p*C + c)*TT + lt]``。``TT`` 个 tile 相邻，所以每个 ``p`` 的写
      是连续的；
    - 第二阶段一个线程一个 ``(o, tile)``，持 16 个累加器沿 ``C`` 累加
      ``U[p][o][c] * V[p][c][lt]``，然后做 ``A^T M A`` 并写回 4 个像素。

    ``U`` 由 ``winograd_weight`` 事先算好（布局 ``(16, O, C)``），本内核只读。
    这个内核不覆盖大 ``C``：``U`` 沿 ``C`` 的读次数是 ``16*C`` 每线程，
    c64 起就超过共享内存与 L2 的承受范围（device 0.246 ms），那几档仍走
    四段版。判据见 ``plan.conv_winograd_fused_preferred``。

    边界与四段版逐条一致：越界采样读 0、``HO`` / ``WO`` 为奇数时最后一个
    tile 的右 / 下沿不写。``TT`` 是运行期标量，网格与线程数由调用方按
    ``TT * O`` 铺。
    """
    cell = _wino_cell
    acc = lambda row, col: Index("acc", _wino_pos(row, col))
    sv = lambda p, c, lt: Bin(
        "+",
        Bin(
            "*",
            Paren(Bin("+", Bin("*", Var(p), Var("C")), Var(c))),
            Var("TT"),
        ),
        Var(lt),
    )
    ih = Bin(
        "-",
        Bin("+", Bin("*", Var("th"), Num("2")), Var("ii")),
        Var("PH"),
    )
    iw = Bin(
        "-",
        Bin("+", Bin("*", Var("tw"), Num("2")), Var("jj")),
        Var("PW"),
    )
    src = Bin(
        "+",
        Var("base"),
        Bin("+", Bin("*", Cast("long long", Paren(ih)), Var("W")), iw),
    )

    def epi(value, channel):
        """写回前的 epilogue：把后继逐元素链（BN / 激活）内联在这里。

        默认 ``None`` 时原样返回，生成的源码与没有这个参数时逐字相同。
        ``channel`` 是当前输出通道变量（``oo``），逐通道参数的索引由它给出。
        """
        return value if epilogue is None else epilogue(value, channel)

    body = (
        Shared("float", "sv", extern=True),
        Let("const int", "per_n", Bin("*", Var("TH"), Var("TW"))),
        Let("const int", "t0", Bin("*", Axis("bid"), Var("TT"))),
        # ---- 1. 输入变换：V = B^T d B，一个线程一个 (c, tile) ----
        For(
            "idx",
            Axis("tid"),
            Bin("*", Var("C"), Var("TT")),
            Axis("bdim"),
            (
                Let("const int", "c", Bin("/", Var("idx"), Var("TT"))),
                Let("const int", "lt", Bin("%", Var("idx"), Var("TT"))),
                Let("const int", "t", Bin("+", Var("t0"), Var("lt"))),
                Let("const bool", "active", Bin("<", Var("t"), Var("NT"))),
                Let("const int", "n", Bin("/", Var("t"), Var("per_n"))),
                Let("const int", "r", Bin("%", Var("t"), Var("per_n"))),
                Let("const int", "th", Bin("/", Var("r"), Var("TW"))),
                Let("const int", "tw", Bin("%", Var("r"), Var("TW"))),
                Let(
                    "const long long",
                    "base",
                    Bin(
                        "*",
                        Cast(
                            "long long",
                            Paren(
                                Bin(
                                    "+",
                                    Bin("*", Var("n"), Var("C")),
                                    Var("c"),
                                )
                            ),
                        ),
                        Bin("*", Var("H"), Var("W")),
                    ),
                ),
                ArrayDecl("float", "d", 16, None),
                ArrayDecl("float", "tmp", 16, None),
                ArrayDecl("float", "vv", 16, None),
                For(
                    "ii",
                    Num("0"),
                    Num("4"),
                    Num("1"),
                    (
                        For(
                            "jj",
                            Num("0"),
                            Num("4"),
                            Num("1"),
                            (
                                Let("const int", "ih", ih),
                                Let("const int", "iw", iw),
                                Assign(
                                    cell("d", Var("ii"), Var("jj")),
                                    Cond(
                                        Bin(
                                            "&&",
                                            Var("active"),
                                            _wino_inside(Var("ih"), Var("iw")),
                                        ),
                                        Index("x", src),
                                        Num("0.f"),
                                    ),
                                ),
                            ),
                            type_name="int",
                        ),
                    ),
                    type_name="int",
                ),
                # tmp = B^T d（列方向）
                For(
                    "jj",
                    Num("0"),
                    Num("4"),
                    Num("1"),
                    (
                        Assign(
                            cell("tmp", Num("0"), Var("jj")),
                            Bin(
                                "-",
                                cell("d", Num("0"), Var("jj")),
                                cell("d", Num("2"), Var("jj")),
                            ),
                        ),
                        Assign(
                            cell("tmp", Num("1"), Var("jj")),
                            Bin(
                                "+",
                                cell("d", Num("1"), Var("jj")),
                                cell("d", Num("2"), Var("jj")),
                            ),
                        ),
                        Assign(
                            cell("tmp", Num("2"), Var("jj")),
                            Bin(
                                "-",
                                cell("d", Num("2"), Var("jj")),
                                cell("d", Num("1"), Var("jj")),
                            ),
                        ),
                        Assign(
                            cell("tmp", Num("3"), Var("jj")),
                            Bin(
                                "-",
                                cell("d", Num("1"), Var("jj")),
                                cell("d", Num("3"), Var("jj")),
                            ),
                        ),
                    ),
                    type_name="int",
                ),
                # vv = tmp B（行方向）
                For(
                    "ii",
                    Num("0"),
                    Num("4"),
                    Num("1"),
                    (
                        Assign(
                            cell("vv", Var("ii"), Num("0")),
                            Bin(
                                "-",
                                cell("tmp", Var("ii"), Num("0")),
                                cell("tmp", Var("ii"), Num("2")),
                            ),
                        ),
                        Assign(
                            cell("vv", Var("ii"), Num("1")),
                            Bin(
                                "+",
                                cell("tmp", Var("ii"), Num("1")),
                                cell("tmp", Var("ii"), Num("2")),
                            ),
                        ),
                        Assign(
                            cell("vv", Var("ii"), Num("2")),
                            Bin(
                                "-",
                                cell("tmp", Var("ii"), Num("2")),
                                cell("tmp", Var("ii"), Num("1")),
                            ),
                        ),
                        Assign(
                            cell("vv", Var("ii"), Num("3")),
                            Bin(
                                "-",
                                cell("tmp", Var("ii"), Num("1")),
                                cell("tmp", Var("ii"), Num("3")),
                            ),
                        ),
                    ),
                    type_name="int",
                ),
                For(
                    "p",
                    Num("0"),
                    Num("16"),
                    Num("1"),
                    (
                        Store(
                            "sv",
                            sv("p", "c", "lt"),
                            Index("vv", Var("p")),
                            raw=True,
                        ),
                    ),
                    type_name="int",
                ),
            ),
            type_name="int",
        ),
        Sync(),
        # ---- 2. 逐点乘 + 沿 C 求和 ----
        Let("const int", "lt", Bin("%", Axis("tid"), Var("TT"))),
        Let("const int", "oo", Bin("/", Axis("tid"), Var("TT"))),
        Let("const int", "t", Bin("+", Var("t0"), Var("lt"))),
        If(Bin(">=", Var("oo"), Var("O")), (Return(),)),
        If(Bin(">=", Var("t"), Var("NT")), (Return(),)),
        ArrayDecl("float", "acc", 16, None),
        For(
            "p",
            Num("0"),
            Num("16"),
            Num("1"),
            (Assign(Index("acc", Var("p")), Num("0.f")),),
            type_name="int",
        ),
        For(
            "c",
            Num("0"),
            Var("C"),
            Num("1"),
            (
                Let(
                    "const float*",
                    "up",
                    Bin(
                        "+",
                        Var("w"),
                        Bin(
                            "+",
                            Bin("*", Var("oo"), Var("C")),
                            Var("c"),
                        ),
                    ),
                ),
                For(
                    "p",
                    Num("0"),
                    Num("16"),
                    Num("1"),
                    (
                        AugAssign(
                            Index("acc", Var("p")),
                            "+",
                            Bin(
                                "*",
                                Index(
                                    "up",
                                    Bin(
                                        "*",
                                        Var("p"),
                                        Bin("*", Var("O"), Var("C")),
                                    ),
                                ),
                                Index("sv", sv("p", "c", "lt")),
                            ),
                        ),
                    ),
                    type_name="int",
                ),
            ),
            type_name="int",
        ),
        # ---- 3. 输出变换：Y = A^T M A + bias ----
        ArrayDecl("float", "m2", 8, None),
        For(
            "jj",
            Num("0"),
            Num("4"),
            Num("1"),
            (
                Assign(
                    Index("m2", Var("jj")),
                    Bin(
                        "+",
                        Bin(
                            "+",
                            acc(Num("0"), Var("jj")),
                            acc(Num("1"), Var("jj")),
                        ),
                        acc(Num("2"), Var("jj")),
                    ),
                ),
                Assign(
                    Index("m2", Bin("+", Num("4"), Var("jj"))),
                    Bin(
                        "-",
                        Bin(
                            "-",
                            acc(Num("1"), Var("jj")),
                            acc(Num("2"), Var("jj")),
                        ),
                        acc(Num("3"), Var("jj")),
                    ),
                ),
            ),
            type_name="int",
        ),
        Let(
            "const float",
            "y00",
            Bin(
                "+",
                Bin("+", Index("m2", Num("0")), Index("m2", Num("1"))),
                Index("m2", Num("2")),
            ),
        ),
        Let(
            "const float",
            "y01",
            Bin(
                "-",
                Bin("-", Index("m2", Num("1")), Index("m2", Num("2"))),
                Index("m2", Num("3")),
            ),
        ),
        Let(
            "const float",
            "y10",
            Bin(
                "+",
                Bin("+", Index("m2", Num("4")), Index("m2", Num("5"))),
                Index("m2", Num("6")),
            ),
        ),
        Let(
            "const float",
            "y11",
            Bin(
                "-",
                Bin("-", Index("m2", Num("5")), Index("m2", Num("6"))),
                Index("m2", Num("7")),
            ),
        ),
        Let("const int", "n", Bin("/", Var("t"), Var("per_n"))),
        Let("const int", "rr", Bin("%", Var("t"), Var("per_n"))),
        Let("const int", "th", Bin("/", Var("rr"), Var("TW"))),
        Let("const int", "tw", Bin("%", Var("rr"), Var("TW"))),
        Let(
            "const float",
            "bv",
            Cond(Var("has_bias"), Index("b", Var("oo")), Num("0.f")),
        ),
        Let(
            "const long long",
            "obase",
            Bin(
                "*",
                Paren(
                    Bin(
                        "+",
                        Bin("*", Cast("long long", Var("n")), Var("O")),
                        Var("oo"),
                    )
                ),
                Bin("*", Cast("long long", Var("HO")), Var("WO")),
            ),
        ),
        Let("const int", "oy", Bin("*", Var("th"), Num("2"))),
        Let("const int", "ox", Bin("*", Var("tw"), Num("2"))),
        Let("const int", "row1", Bin("+", Var("oy"), Num("1"))),
        Let("const int", "col1", Bin("+", Var("ox"), Num("1"))),
        If(
            Bin("<", Var("row1"), Var("HO")),
            (
                If(
                    Bin("<", Var("col1"), Var("WO")),
                    (
                        Store(
                            "out",
                            Bin(
                                "+",
                                Var("obase"),
                                Bin(
                                    "+",
                                    Bin("*", Var("row1"), Var("WO")),
                                    Var("col1"),
                                ),
                            ),
                            epi(Bin("+", Var("y11"), Var("bv")), Var("oo")),
                        ),
                    ),
                ),
                Store(
                    "out",
                    Bin(
                        "+",
                        Var("obase"),
                        Bin(
                            "+",
                            Bin("*", Var("row1"), Var("WO")),
                            Var("ox"),
                        ),
                    ),
                    epi(Bin("+", Var("y10"), Var("bv")), Var("oo")),
                ),
            ),
        ),
        If(
            Bin("<", Var("col1"), Var("WO")),
            (
                Store(
                    "out",
                    Bin(
                        "+",
                        Var("obase"),
                        Bin(
                            "+",
                            Bin("*", Var("oy"), Var("WO")),
                            Var("col1"),
                        ),
                    ),
                    epi(Bin("+", Var("y01"), Var("bv")), Var("oo")),
                ),
            ),
        ),
        Store(
            "out",
            Bin(
                "+",
                Var("obase"),
                Bin("+", Bin("*", Var("oy"), Var("WO")), Var("ox")),
            ),
            epi(Bin("+", Var("y00"), Var("bv")), Var("oo")),
        ),
    )
    return Kernel(
        name="winograd_fused",
        params=(
            Tensor("x", type_name="float"),
            Tensor("w", type_name="float"),
            Tensor("b", type_name="float"),
            Tensor("out", read_only=False, type_name="float"),
            Scalar("int", "C"),
            Scalar("int", "H"),
            Scalar("int", "W"),
            Scalar("int", "O"),
            Scalar("int", "TH"),
            Scalar("int", "TW"),
            Scalar("int", "HO"),
            Scalar("int", "WO"),
            Scalar("int", "PH"),
            Scalar("int", "PW"),
            Scalar("int", "NT"),
            Scalar("int", "TT"),
            Scalar("int", "has_bias"),
            *(Tensor(name, type_name="float") for name, _ in extra),
        ),
        body=body,
    )
