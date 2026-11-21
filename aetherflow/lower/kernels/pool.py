"""池化内核：最大 / 平均 / 自适应平均，含反向与取值位置。"""

from __future__ import annotations

from ..kernel import (
    Assign,
    AtomicAdd,
    AugAssign,
    Bin,
    Call,
    Cast,
    Cond,
    Continue,
    Expr,
    For,
    If,
    Kernel,
    Let,
    LetMulti,
    Load,
    Num,
    Paren,
    Scalar,
    Store,
    Tensor,
    Var,
)
from .common import _flat_loop
from .image import _image_slice, _split_output_index, _tap


def _pool_params(*, adaptive: bool, extra: tuple = ()) -> tuple:
    """池化内核的公共参数：``C/H/W`` 加上窗口或输出尺寸。"""
    if adaptive:
        dims = (Scalar("int", "OH"), Scalar("int", "OW"))
    else:
        dims = (
            Scalar("int", "KH"),
            Scalar("int", "KW"),
            Scalar("int", "HO"),
            Scalar("int", "WO"),
            Scalar("int", "SH"),
            Scalar("int", "SW"),
            Scalar("int", "PH"),
            Scalar("int", "PW"),
        )
    return (
        (
            Scalar("int", "C"),
            Scalar("int", "H"),
            Scalar("int", "W"),
        )
        + dims
        + (Scalar("long long", "total"),)
        + extra
    )


def _window_axis(k: str, i: str, stride: str, pad: str, o: str, lo: Expr, hi: Expr) -> tuple:
    """``int i = o * stride - pad + k; if (i < lo || i >= hi) { continue; }``"""
    return (
        Let(
            "int",
            i,
            Bin(
                "+",
                Bin("-", Bin("*", Var(o), Var(stride)), Var(pad)),
                Var(k),
            ),
        ),
        If(
            Bin("||", Bin("<", Var(i), lo), Bin(">=", Var(i), hi)),
            (Continue(),),
        ),
    )


def _window_scan(body: tuple, *, lo_h=None, hi_h=None, lo_w=None, hi_w=None) -> For:
    """``for ki ... { ...; for kj ... { ...; <body> } }``——越界的采样点直接跳过。

    默认边界是输入本身 ``[0, H)`` / ``[0, W)``。``avgpool2d`` 在
    ``count_include_pad=True`` 时把边界放宽到 ``[-PH, H + PH)``：那一档要数
    "窗口落在 padded 输入里"的格数，padding 的零计入分母但值为 0，因此循环
    会进入这些格子，取值却要自己再判一次（见 ``_in_image``）。
    """
    lo_h = Num("0") if lo_h is None else lo_h
    hi_h = Var("H") if hi_h is None else hi_h
    lo_w = Num("0") if lo_w is None else lo_w
    hi_w = Var("W") if hi_w is None else hi_w
    return For(
        "ki",
        Num("0"),
        Var("KH"),
        Num("1"),
        _window_axis("ki", "ih", "SH", "PH", "ho", lo_h, hi_h)
        + (
            For(
                "kj",
                Num("0"),
                Var("KW"),
                Num("1"),
                _window_axis("kj", "iw", "SW", "PW", "wo", lo_w, hi_w) + body,
                incr="++",
                type_name="int",
            ),
        ),
        incr="++",
        type_name="int",
    )


def _in_image(ih: str = "ih", iw: str = "iw") -> Expr:
    """``ih >= 0 && ih < H && iw >= 0 && iw < W``——这一格是不是真实输入。"""
    return Bin(
        "&&",
        Bin(
            "&&",
            Bin(">=", Var(ih), Num("0")),
            Bin("<", Var(ih), Var("H")),
        ),
        Bin(
            "&&",
            Bin(">=", Var(iw), Num("0")),
            Bin("<", Var(iw), Var("W")),
        ),
    )


def maxpool2d() -> Kernel:
    """2D 最大池化：一个线程一个输出元素，越界采样点不参与比较。"""
    return Kernel(
        name="maxpool2d",
        params=(Tensor("x"), Tensor("out", read_only=False)) + _pool_params(adaptive=False),
        body=_flat_loop(
            _split_output_index("wo", "ho", "WO", "HO")
            + (
                _image_slice("x"),
                Let(
                    "float",
                    "m",
                    Call("__int_as_float", (Num("0xff800000"),)),
                ),
                _window_scan(
                    (
                        Let("float", "v", Load("base", _tap().index)),
                        Assign(
                            Var("m"),
                            Cond(
                                Bin(">", Var("v"), Var("m")),
                                Var("v"),
                                Var("m"),
                            ),
                        ),
                    )
                ),
                Store("out", Var("idx"), Var("m")),
            )
        ),
    )


def maxpool2d_backward() -> Kernel:
    """最大池化反向：把 dy 路由到每个窗口的第一个最大值位置。

    平局取行优先扫描里的**第一个**最大值——与 IR 参考实现一致；把比较改成
    ``>=`` 会在高原上悄悄改变语义。重叠窗口可能落到同一个输入单元，所以
    必须原子加。
    """
    return Kernel(
        name="maxpool2d_backward",
        params=(
            Tensor("x"),
            Tensor("dy"),
            Tensor("dx", read_only=False),
        )
        + _pool_params(adaptive=False),
        body=_flat_loop(
            _split_output_index("wo", "ho", "WO", "HO")
            + (
                _image_slice("x"),
                Let("float", "best", Num("-3.4e38f")),
                LetMulti("int", (("best_ih", Num("-1")), ("best_iw", Num("-1")))),
                _window_scan(
                    (
                        Let("float", "v", Load("base", _tap().index)),
                        If(
                            Bin(
                                "||",
                                Bin("<", Var("best_ih"), Num("0")),
                                Bin(">", Var("v"), Var("best")),
                            ),
                            (
                                Assign(Var("best"), Var("v")),
                                Assign(Var("best_ih"), Var("ih")),
                                Assign(Var("best_iw"), Var("iw")),
                            ),
                        ),
                    )
                ),
                If(
                    Bin(">=", Var("best_ih"), Num("0")),
                    (
                        _image_slice("dx", "dst", const=False),
                        AtomicAdd(
                            "dst",
                            Bin(
                                "+",
                                Bin(
                                    "*",
                                    Cast("long long", Var("best_ih")),
                                    Var("W"),
                                ),
                                Var("best_iw"),
                            ),
                            Load("dy", Var("idx")),
                        ),
                    ),
                ),
            )
        ),
    )


def maxpool2d_indices() -> Kernel:
    """最大池化的取值位置：``out[idx] = best_ih * W + best_iw``（i32）。

    与 ``maxpool2d_backward`` 用同一套定位规则——行优先扫描、严格大于才更新，
    所以并列最大值三个算子选中同一个元素。窗口全在 padding 里时写 -1（
    ``ceil_mode`` 的起点修正保证正常配置不会出现，这里只是不留未定义值）。
    """
    return Kernel(
        name="maxpool2d_indices",
        params=(Tensor("x"), Tensor("out", read_only=False, type_name="int"))
        + _pool_params(adaptive=False),
        body=_flat_loop(
            _split_output_index("wo", "ho", "WO", "HO")
            + (
                _image_slice("x"),
                Let("float", "best", Num("-3.4e38f")),
                LetMulti("int", (("best_ih", Num("-1")), ("best_iw", Num("-1")))),
                _window_scan(
                    (
                        Let("float", "v", Load("base", _tap().index)),
                        If(
                            Bin(
                                "||",
                                Bin("<", Var("best_ih"), Num("0")),
                                Bin(">", Var("v"), Var("best")),
                            ),
                            (
                                Assign(Var("best"), Var("v")),
                                Assign(Var("best_ih"), Var("ih")),
                                Assign(Var("best_iw"), Var("iw")),
                            ),
                        ),
                    )
                ),
                Store("out", Var("idx"), Num("-1"), raw=True),
                If(
                    Bin(">=", Var("best_ih"), Num("0")),
                    (
                        Store(
                            "out",
                            Var("idx"),
                            Bin(
                                "+",
                                Bin("*", Var("best_ih"), Var("W")),
                                Var("best_iw"),
                            ),
                            raw=True,
                        ),
                    ),
                ),
            )
        ),
    )


def _avgpool_pad_bounds() -> tuple:
    """``lo_h / hi_h / lo_w / hi_w``——分母计数的坐标区间。

    ``count_include_pad`` 为真时 padding 计入分母，区间放宽到
    ``[-P, size + P)``；否则只数真实输入。写成四个 ``Let`` 而不是把三元表达式
    直接嵌进比较里：打印器不做优先级推断，``a < (c ? x : y)`` 不带括号会被
    解析成 ``(a < c) ? ...``。
    """

    def bound(off: str, size: str, sign: str):
        return Cond(
            Bin("!=", Var("count_include_pad"), Num("0")),
            (Bin("*", Num("-1"), Var(off)) if sign == "-" else Bin("+", Var(size), Var(off))),
            Num("0") if sign == "-" else Var(size),
        )

    return (
        Let("int", "lo_h", bound("PH", "H", "-")),
        Let("int", "hi_h", bound("PH", "H", "+")),
        Let("int", "lo_w", bound("PW", "W", "-")),
        Let("int", "hi_w", bound("PW", "W", "+")),
    )


def avgpool2d() -> Kernel:
    """2D 平均池化：分母是窗口落在 **padded 输入** 里的格数。

    ``count_include_pad=True`` 时 padding 的位置计入分母（值按 0 算），边界
    取 ``[-PH, H + PH)``；``False`` 时只数窗口与真实输入的重叠，边界取
    ``[0, H)``。两者共用一份源码，由 ``count_include_pad`` 这个标量选边界。

    为什么不能像早先那样"固定除以 ``KH * KW``"：``ceil_mode`` 允许最后一格
    越过 padded 输入（PyTorch 只要求窗口**起点**落在左 padding 或输入内），
    越界的那几格既不计值也不进分母。所以即便
    ``count_include_pad=True``，角落窗口的分母也可能小于 ``KH * KW``；在
    前端的 ``F.avg_pool2d(..., ceil_mode=True)`` 上这直接差出 1.5 倍。
    """
    return Kernel(
        name="avgpool2d",
        params=(Tensor("x"), Tensor("out", read_only=False))
        + _pool_params(
            adaptive=False,
            extra=(Scalar("int", "count_include_pad"),),
        ),
        body=_flat_loop(
            _split_output_index("wo", "ho", "WO", "HO")
            + _avgpool_pad_bounds()
            + (
                _image_slice("x"),
                Let("float", "acc", Num("0.f")),
                Let("int", "cnt", Num("0")),
                _window_scan(
                    (
                        AugAssign(Var("cnt"), "+", Num("1")),
                        AugAssign(
                            Var("acc"),
                            "+",
                            Cond(
                                _in_image(),
                                Load("base", _tap().index),
                                Num("0.f"),
                            ),
                        ),
                    ),
                    lo_h=Var("lo_h"),
                    hi_h=Var("hi_h"),
                    lo_w=Var("lo_w"),
                    hi_w=Var("hi_w"),
                ),
                Let(
                    "float",
                    "den",
                    Cond(
                        Bin(">", Var("cnt"), Num("0")),
                        Bin("/", Num("1.f"), Cast("float", Var("cnt"))),
                        Num("1.f"),
                    ),
                ),
                Store("out", Var("idx"), Bin("*", Var("acc"), Var("den"))),
            )
        ),
    )


def avgpool2d_backward() -> Kernel:
    """平均池化反向：把 ``dy * 分母倒数`` 散射回窗口内的真实输入位置。

    重叠窗口会碰到同一个输入单元，所以累加必须原子——普通写入只会留下
    最后一个窗口的贡献。分母与 ``avgpool2d`` 前向逐格对应：先按同一套边界
    （含 ``ceil_mode`` 越界与 ``count_include_pad``）数一遍格数，再只把梯度
    加到真实输入上；padding 那一侧只参与分母，不接收梯度。
    """
    return Kernel(
        name="avgpool2d_backward",
        params=(
            Tensor("dy"),
            Tensor("dx", read_only=False),
        )
        + _pool_params(
            adaptive=False,
            extra=(Scalar("int", "count_include_pad"),),
        ),
        body=_flat_loop(
            _split_output_index("wo", "ho", "WO", "HO")
            + _avgpool_pad_bounds()
            + (
                Let("int", "cnt", Num("0")),
                _window_scan(
                    (AugAssign(Var("cnt"), "+", Num("1")),),
                    lo_h=Var("lo_h"),
                    hi_h=Var("hi_h"),
                    lo_w=Var("lo_w"),
                    hi_w=Var("hi_w"),
                ),
                Let(
                    "float",
                    "den",
                    Cond(
                        Bin(">", Var("cnt"), Num("0")),
                        Bin("/", Num("1.f"), Cast("float", Var("cnt"))),
                        Num("1.f"),
                    ),
                ),
                Let("float", "g", Bin("*", Load("dy", Var("idx")), Var("den"))),
                _image_slice("dx", const=False),
                _window_scan(
                    (
                        If(
                            _in_image(),
                            (AtomicAdd("base", _tap().index, Var("g")),),
                        ),
                    ),
                    lo_h=Var("lo_h"),
                    hi_h=Var("hi_h"),
                    lo_w=Var("lo_w"),
                    hi_w=Var("hi_w"),
                ),
            )
        ),
    )


def _adaptive_bounds(y_var: str, x_var: str) -> tuple:
    """``hs/he/ws/we``——PyTorch 的 floor/ceil 窗口规则，用整数除法避免边界抖动。"""
    return (
        Let(
            "int",
            "hs",
            Cast(
                "int",
                Paren(
                    Bin(
                        "/",
                        Paren(Bin("*", Cast("long long", Var(y_var)), Var("H"))),
                        Var("OH"),
                    )
                ),
            ),
        ),
        Let(
            "int",
            "he",
            Cast(
                "int",
                Paren(
                    Bin(
                        "/",
                        Paren(
                            Bin(
                                "-",
                                Bin(
                                    "+",
                                    Paren(
                                        Bin(
                                            "*",
                                            Cast(
                                                "long long",
                                                Paren(
                                                    Bin(
                                                        "+",
                                                        Var(y_var),
                                                        Num("1"),
                                                    )
                                                ),
                                            ),
                                            Var("H"),
                                        )
                                    ),
                                    Var("OH"),
                                ),
                                Num("1"),
                            )
                        ),
                        Var("OH"),
                    )
                ),
            ),
        ),
        Let(
            "int",
            "ws",
            Cast(
                "int",
                Paren(
                    Bin(
                        "/",
                        Paren(Bin("*", Cast("long long", Var(x_var)), Var("W"))),
                        Var("OW"),
                    )
                ),
            ),
        ),
        Let(
            "int",
            "we",
            Cast(
                "int",
                Paren(
                    Bin(
                        "/",
                        Paren(
                            Bin(
                                "-",
                                Bin(
                                    "+",
                                    Paren(
                                        Bin(
                                            "*",
                                            Cast(
                                                "long long",
                                                Paren(
                                                    Bin(
                                                        "+",
                                                        Var(x_var),
                                                        Num("1"),
                                                    )
                                                ),
                                            ),
                                            Var("W"),
                                        )
                                    ),
                                    Var("OW"),
                                ),
                                Num("1"),
                            )
                        ),
                        Var("OW"),
                    )
                ),
            ),
        ),
    )


def _adaptive_window(body: tuple) -> For:
    """``for (int ih = hs; ih < he; ++ih) { for (int iw = ws; iw < we; ++iw) { ... } }``"""
    return For(
        "ih",
        Var("hs"),
        Var("he"),
        Num("1"),
        (
            For(
                "iw",
                Var("ws"),
                Var("we"),
                Num("1"),
                body,
                incr="++",
                type_name="int",
            ),
        ),
        incr="++",
        type_name="int",
    )


def adaptive_avg_pool2d() -> Kernel:
    """自适应平均池化：窗口是 ``[floor(o*H/OH), ceil((o+1)*H/OH))``，除数是窗口自身面积。"""
    return Kernel(
        name="adaptive_avg_pool2d",
        params=(Tensor("x"), Tensor("out", read_only=False)) + _pool_params(adaptive=True),
        body=_flat_loop(
            _split_output_index("ow", "oh", "OW", "OH")
            + (_image_slice("x"),)
            + _adaptive_bounds("oh", "ow")
            + (
                Let("float", "acc", Num("0.f")),
                _adaptive_window((AugAssign(Var("acc"), "+", Load("base", _tap().index)),)),
                Store(
                    "out",
                    Var("idx"),
                    Bin(
                        "/",
                        Var("acc"),
                        Cast(
                            "float",
                            Paren(
                                Bin(
                                    "*",
                                    Paren(Bin("-", Var("he"), Var("hs"))),
                                    Paren(Bin("-", Var("we"), Var("ws"))),
                                )
                            ),
                        ),
                    ),
                ),
            )
        ),
    )


def adaptive_avg_pool2d_backward() -> Kernel:
    """自适应平均池化反向：把 ``dy / 窗口面积`` 散射回窗口内的单元（原子加）。"""
    return Kernel(
        name="adaptive_avg_pool2d_backward",
        params=(Tensor("dy"), Tensor("dx", read_only=False)) + _pool_params(adaptive=True),
        body=_flat_loop(
            _split_output_index("ow", "oh", "OW", "OH")
            + _adaptive_bounds("oh", "ow")
            + (
                Let(
                    "float",
                    "share",
                    Bin(
                        "/",
                        Load("dy", Var("idx")),
                        Cast(
                            "float",
                            Paren(
                                Bin(
                                    "*",
                                    Paren(Bin("-", Var("he"), Var("hs"))),
                                    Paren(Bin("-", Var("we"), Var("ws"))),
                                )
                            ),
                        ),
                    ),
                ),
                _image_slice("dx", const=False),
                _adaptive_window((AtomicAdd("base", _tap().index, Var("share")),)),
            )
        ),
    )
