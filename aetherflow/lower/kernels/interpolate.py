"""二维空间插值：最近邻与双线性，前向与反向。

坐标口径与 ``ir/ops.py`` 的参考实现逐字对应（整数分子/分母，不是浮点
scale）：

- nearest：``ih = (long long)oh * H / OH``；
- bilinear / align_corners=False：``src = ((2*dst+1)*in - out) / (2*out)``，
  再截断到 ``[0, in-1]``；
- bilinear / align_corners=True：``src = dst * (in-1) / (out-1)``。

``align_corners`` 是构图期属性，因此在这里展开成两套语句，执行期不分支。

反向用原子加：多个输出格可能命中同一个输入格（最近邻的重复采样、双线性在
in/out 不成比例时的重叠），覆盖写会丢梯度。
"""

from __future__ import annotations

from ..kernel import (
    Assign,
    AtomicAdd,
    Bin,
    Cast,
    Cond,
    Expr,
    If,
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
from .common import _flat_loop
from .image import _image_slice, _split_output_index, _tap_at


def _params() -> tuple:
    return (
        Tensor("x"),
        Tensor("out", read_only=False),
        Scalar("int", "C"),
        Scalar("int", "H"),
        Scalar("int", "W"),
        Scalar("int", "OH"),
        Scalar("int", "OW"),
        Scalar("long long", "total"),
    )


def _backward_params() -> tuple:
    return (
        Tensor("dy"),
        Tensor("dx", read_only=False),
        Scalar("int", "C"),
        Scalar("int", "H"),
        Scalar("int", "W"),
        Scalar("int", "OH"),
        Scalar("int", "OW"),
        Scalar("long long", "total"),
    )


def _index_expr(ih: Expr, iw: Expr) -> Expr:
    """``(long long)ih * W + iw``——与 :func:`_tap_at` 的偏移公式相同。"""
    return Bin("+", Bin("*", Cast("long long", ih), Var("W")), iw)


def _nearest_index(prefix: str, out_var: str, in_dim: str, out_dim: str) -> Let:
    """``int <prefix> = (long long)<o> * <in> / <out>``（整数除法即下取整）。"""
    return Let(
        "int",
        prefix,
        Cast(
            "int",
            Paren(
                Bin(
                    "/",
                    Paren(Bin("*", Cast("long long", Var(out_var)), Var(in_dim))),
                    Var(out_dim),
                )
            ),
        ),
    )


def _bilinear_index(
    prefix: str,
    out_var: str,
    in_dim: str,
    out_dim: str,
    *,
    align_corners: bool,
) -> tuple:
    """一个轴的 ``num/den -> i0/rem/lam/i1`` 语句序列。

    分子分母都是整数：``src = num / den``，``i0 = num // den``，
    ``lam = (num - i0*den) / den``。这样参考实现（numpy 整数除法）与设备
    内核（``long long`` 除法）得到逐位相同的坐标，不受浮点 scale 的边界抖动
    影响。
    """
    if align_corners:
        num = Let(
            "long long",
            prefix + "num",
            Bin(
                "*",
                Cast("long long", Var(out_var)),
                Paren(Bin("-", Var(in_dim), Num("1"))),
            ),
        )
        den = Let(
            "int",
            prefix + "den",
            Cond(
                Bin(">", Var(out_dim), Num("1")),
                Bin("-", Var(out_dim), Num("1")),
                Num("1"),
            ),
        )
        clamps: tuple = ()
    else:
        num = Let(
            "long long",
            prefix + "num",
            Bin(
                "-",
                Bin(
                    "*",
                    Paren(
                        Bin(
                            "+",
                            Bin("*", Cast("long long", Var(out_var)), Num("2")),
                            Num("1"),
                        )
                    ),
                    Var(in_dim),
                ),
                Var(out_dim),
            ),
        )
        # 下界对应 torch 的 src<0 -> 0；上界是双线性插值的标准定义。H==1 时
        # 上界为 0，整轴退化成取唯一的像素。
        cap = Bin(
            "*",
            Cast(
                "long long",
                Paren(Bin("-", Var(in_dim), Num("1"))),
            ),
            Paren(Bin("*", Num("2"), Var(out_dim))),
        )
        clamps = (
            Let("long long", prefix + "cap", cap),
            If(
                Bin("<", Var(prefix + "num"), Num("0")),
                (Assign(Var(prefix + "num"), Num("0")),),
            ),
            If(
                Bin(">", Var(prefix + "num"), Var(prefix + "cap")),
                (Assign(Var(prefix + "num"), Var(prefix + "cap")),),
            ),
        )
        den = Let(
            "int", prefix + "den", Bin("*", Num("2"), Var(out_dim))
        )

    return (num,) + clamps + (
        den,
        Let(
            "int",
            prefix + "0",
            Cast(
                "int",
                Paren(
                    Bin("/", Var(prefix + "num"), Var(prefix + "den"))
                ),
            ),
        ),
        Let(
            "int",
            prefix + "rem",
            Cast(
                "int",
                Paren(
                    Bin(
                        "-",
                        Var(prefix + "num"),
                        Bin(
                            "*",
                            Cast("long long", Var(prefix + "0")),
                            Var(prefix + "den"),
                        ),
                    )
                ),
            ),
        ),
        Let(
            "float",
            prefix + "lam",
            Bin(
                "/",
                Cast("float", Var(prefix + "rem")),
                Cast("float", Var(prefix + "den")),
            ),
        ),
        Let(
            "int",
            prefix + "1",
            Bin(
                "+",
                Var(prefix + "0"),
                Paren(
                    Cond(
                        Bin(
                            "<",
                            Paren(Bin("+", Var(prefix + "0"), Num("1"))),
                            Var(in_dim),
                        ),
                        Num("1"),
                        Num("0"),
                    )
                ),
            ),
        ),
    )


def interpolate_nearest() -> Kernel:
    """一个线程一个输出格，取最近邻源点。"""
    return Kernel(
        name="interpolate_nearest",
        params=_params(),
        body=_flat_loop(
            _split_output_index("ow", "oh", "OW", "OH")
            + (
                _image_slice("x"),
                _nearest_index("ih", "oh", "H", "OH"),
                _nearest_index("iw", "ow", "W", "OW"),
                Store("out", Var("idx"), _tap_at(Var("ih"), Var("iw"))),
            )
        ),
    )


def interpolate_bilinear(*, align_corners: bool) -> Kernel:
    """一个线程一个输出格，四角加权。

    权重按 ``lh0*lw0, lh0*lw, lh*lw0, lh*lw`` 展开；``i1`` 已截断到
    ``in-1``，落点越界时对应权重为 0，不会读出界。
    """
    axis_h = _bilinear_index(
        "hi", "oh", "H", "OH", align_corners=align_corners
    )
    axis_w = _bilinear_index(
        "wi", "ow", "W", "OW", align_corners=align_corners
    )
    acc = Bin(
        "+",
        Bin(
            "+",
            Bin(
                "*",
                Var("a"),
                _tap_at(Var("hi0"), Var("wi0")),
            ),
            Bin(
                "*",
                Var("b"),
                _tap_at(Var("hi0"), Var("wi1")),
            ),
        ),
        Bin(
            "+",
            Bin(
                "*",
                Var("c"),
                _tap_at(Var("hi1"), Var("wi0")),
            ),
            Bin(
                "*",
                Var("d"),
                _tap_at(Var("hi1"), Var("wi1")),
            ),
        ),
    )
    return Kernel(
        name="interpolate_bilinear",
        params=_params(),
        body=_flat_loop(
            _split_output_index("ow", "oh", "OW", "OH")
            + (_image_slice("x"),)
            + axis_h
            + axis_w
            + (
                Let(
                    "float",
                    "lh0",
                    Bin("-", Num("1.f"), Var("hilam")),
                ),
                Let(
                    "float",
                    "lw0",
                    Bin("-", Num("1.f"), Var("wilam")),
                ),
                Let("float", "a", Bin("*", Var("lh0"), Var("lw0"))),
                Let("float", "b", Bin("*", Var("lh0"), Var("wilam"))),
                Let("float", "c", Bin("*", Var("hilam"), Var("lw0"))),
                Let("float", "d", Bin("*", Var("hilam"), Var("wilam"))),
                Store("out", Var("idx"), acc),
            )
        ),
    )


def interpolate_nearest_backward() -> Kernel:
    """把每个 dy 原子加回它的源点。"""
    return Kernel(
        name="interpolate_nearest_backward",
        params=_backward_params(),
        body=_flat_loop(
            _split_output_index("ow", "oh", "OW", "OH")
            + (
                _image_slice("dx", const=False),
                _nearest_index("ih", "oh", "H", "OH"),
                _nearest_index("iw", "ow", "W", "OW"),
                AtomicAdd(
                    "base",
                    _index_expr(Var("ih"), Var("iw")),
                    Load("dy", Var("idx")),
                ),
            )
        ),
    )


def interpolate_bilinear_backward(*, align_corners: bool) -> Kernel:
    """把 dy 按四角权重原子加回源点。"""
    axis_h = _bilinear_index(
        "hi", "oh", "H", "OH", align_corners=align_corners
    )
    axis_w = _bilinear_index(
        "wi", "ow", "W", "OW", align_corners=align_corners
    )
    return Kernel(
        name="interpolate_bilinear_backward",
        params=_backward_params(),
        body=_flat_loop(
            _split_output_index("ow", "oh", "OW", "OH")
            + (_image_slice("dx", const=False),)
            + axis_h
            + axis_w
            + (
                Let(
                    "float",
                    "lh0",
                    Bin("-", Num("1.f"), Var("hilam")),
                ),
                Let(
                    "float",
                    "lw0",
                    Bin("-", Num("1.f"), Var("wilam")),
                ),
                Let("float", "g", Load("dy", Var("idx"))),
                Let("float", "ga", Bin("*", Var("g"), Bin("*", Var("lh0"), Var("lw0")))),
                Let("float", "gb", Bin("*", Var("g"), Bin("*", Var("lh0"), Var("wilam")))),
                Let("float", "gc", Bin("*", Var("g"), Bin("*", Var("hilam"), Var("lw0")))),
                Let("float", "gd", Bin("*", Var("g"), Bin("*", Var("hilam"), Var("wilam")))),
                AtomicAdd("base", _index_expr(Var("hi0"), Var("wi0")), Var("ga")),
                AtomicAdd("base", _index_expr(Var("hi0"), Var("wi1")), Var("gb")),
                AtomicAdd("base", _index_expr(Var("hi1"), Var("wi0")), Var("gc")),
                AtomicAdd("base", _index_expr(Var("hi1"), Var("wi1")), Var("gd")),
            )
        ),
    )
