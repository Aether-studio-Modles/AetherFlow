"""归一化内核：LayerNorm / RMSNorm / BatchNorm 及其统计量。"""

from __future__ import annotations

from ..kernel import (
    Assign,
    AugAssign,
    Axis,
    Bin,
    Call,
    Cast,
    Expr,
    For,
    If,
    Kernel,
    Let,
    Load,
    Narrow,
    Num,
    Paren,
    Ptr,
    Return,
    Scalar,
    Shared,
    Stmt,
    Store,
    Tensor,
    Var,
    Widen,
)
from ..plan import REDUCE_SCRATCH


def layernorm_last() -> Kernel:
    """沿最后一维的 LayerNorm：一个 block 一行，均值与方差各一次归约。

    方差由第二次扫描现场算出，而不是用 ``E[x^2] - mean^2``——后者在
    均值远大于方差时会丢掉有效位（差分抵消），归一化结果就不再可信。
    """

    def _scan(body: tuple) -> For:
        return For("i", Axis("tid"), Var("cols"), Axis("bdim"), body)

    def _centered() -> Expr:
        """``%LD%(xr[i]) - mean``"""
        return Bin("-", Load("xr", Var("i")), Var("mean"))

    def _inv_std(var: Expr) -> Expr:
        """``rsqrtf(var / (float)cols + eps)``"""
        return Call(
            "rsqrtf",
            (Bin("+", Bin("/", var, Cast("float", Var("cols"))), Var("eps")),),
        )

    return Kernel(
        name="layernorm_last",
        params=(
            Tensor("x"),
            Tensor("gamma"),
            Tensor("beta"),
            Tensor("out", read_only=False),
            Scalar("long long", "rows"),
            Scalar("long long", "cols"),
            Scalar("float", "eps"),
        ),
        body=(
            Let("long long", "row", Axis("bid")),
            If(Bin(">=", Var("row"), Var("rows")), (Return(),)),
            Ptr("xr", "x", Bin("*", Var("row"), Var("cols"))),
            Ptr(
                "orow",
                "out",
                Bin("*", Var("row"), Var("cols")),
                const=False,
            ),
            Shared("float", "smem", REDUCE_SCRATCH),
            Let("float", "s", Num("0.f")),
            _scan((AugAssign(Var("s"), "+", Load("xr", Var("i"))),)),
            Assign(Var("s"), Call("block_reduce_sum", (Var("s"), Var("smem")))),
            Let(
                "float",
                "mean",
                Bin("/", Var("s"), Cast("float", Var("cols"))),
            ),
            Let("float", "v", Num("0.f")),
            _scan(
                (
                    Let("float", "d", _centered()),
                    AugAssign(Var("v"), "+", Bin("*", Var("d"), Var("d"))),
                )
            ),
            Assign(Var("v"), Call("block_reduce_sum", (Var("v"), Var("smem")))),
            Let("float", "inv", _inv_std(Var("v"))),
            _scan(
                (
                    Store(
                        "orow",
                        Var("i"),
                        Bin(
                            "+",
                            Bin(
                                "*",
                                Bin("*", Paren(_centered()), Var("inv")),
                                Load("gamma", Var("i")),
                            ),
                            Load("beta", Var("i")),
                        ),
                    ),
                )
            ),
        ),
    )


def layernorm_residual_last() -> Kernel:
    """融合残差的 LayerNorm：``x + res`` 不落成中间张量。

    每次用到 ``xv`` 都重新算一遍 ``x + res``，并且**先舍入回存储类型**——
    f16 的参考实现（numpy 两个 f16 相加）也是这样，先加再窄化，而不是
    一路用 float 累加到最后才舍入。
    """

    def _scan(body: tuple) -> For:
        return For("i", Axis("tid"), Var("cols"), Axis("bdim"), body)

    def _xv() -> Stmt:
        """``%T% xv = %ST%(%LD%(xr[i]) + %LD%(rr[i]));``"""
        return Let(
            None,
            "xv",
            Narrow(Bin("+", Load("xr", Var("i")), Load("rr", Var("i")))),
        )

    def _xv_f() -> Expr:
        """``%LD%(xv)``"""
        return Widen(Var("xv"))

    def _centered() -> Expr:
        return Bin("-", _xv_f(), Var("mean"))

    def _inv_std(var: Expr) -> Expr:
        return Call(
            "rsqrtf",
            (Bin("+", Bin("/", var, Cast("float", Var("cols"))), Var("eps")),),
        )

    return Kernel(
        name="layernorm_residual",
        params=(
            Tensor("x"),
            Tensor("res"),
            Tensor("gamma"),
            Tensor("beta"),
            Tensor("out", read_only=False),
            Scalar("long long", "rows"),
            Scalar("long long", "cols"),
            Scalar("float", "eps"),
        ),
        body=(
            Let("long long", "row", Axis("bid")),
            If(Bin(">=", Var("row"), Var("rows")), (Return(),)),
            Ptr("xr", "x", Bin("*", Var("row"), Var("cols"))),
            Ptr("rr", "res", Bin("*", Var("row"), Var("cols"))),
            Ptr(
                "orow",
                "out",
                Bin("*", Var("row"), Var("cols")),
                const=False,
            ),
            Shared("float", "smem", REDUCE_SCRATCH),
            Let("float", "s", Num("0.f")),
            _scan((_xv(), AugAssign(Var("s"), "+", _xv_f()))),
            Assign(Var("s"), Call("block_reduce_sum", (Var("s"), Var("smem")))),
            Let(
                "float",
                "mean",
                Bin("/", Var("s"), Cast("float", Var("cols"))),
            ),
            Let("float", "v", Num("0.f")),
            _scan(
                (
                    _xv(),
                    Let("float", "d", _centered()),
                    AugAssign(Var("v"), "+", Bin("*", Var("d"), Var("d"))),
                )
            ),
            Assign(Var("v"), Call("block_reduce_sum", (Var("v"), Var("smem")))),
            Let("float", "inv", _inv_std(Var("v"))),
            _scan(
                (
                    _xv(),
                    Store(
                        "orow",
                        Var("i"),
                        Bin(
                            "+",
                            Bin(
                                "*",
                                Bin("*", Paren(_centered()), Var("inv")),
                                Load("gamma", Var("i")),
                            ),
                            Load("beta", Var("i")),
                        ),
                    ),
                )
            ),
        ),
    )


def rmsnorm_last() -> Kernel:
    """沿最后一维的 RMSNorm：只做一次归约（均方），不减均值、没有 beta。

    这正是它与 LayerNorm 的全部差别：少了中心化，就少了一次扫描，
    也因此少了一个内核参数。
    """

    def _scan(body: tuple) -> For:
        return For("i", Axis("tid"), Var("cols"), Axis("bdim"), body)

    return Kernel(
        name="rmsnorm_last",
        params=(
            Tensor("x"),
            Tensor("gamma"),
            Tensor("out", read_only=False),
            Scalar("long long", "rows"),
            Scalar("long long", "cols"),
            Scalar("float", "eps"),
        ),
        body=(
            Let("long long", "row", Axis("bid")),
            If(Bin(">=", Var("row"), Var("rows")), (Return(),)),
            Ptr("xr", "x", Bin("*", Var("row"), Var("cols"))),
            Ptr(
                "orow",
                "out",
                Bin("*", Var("row"), Var("cols")),
                const=False,
            ),
            Shared("float", "smem", REDUCE_SCRATCH),
            Let("float", "s", Num("0.f")),
            _scan(
                (
                    Let("float", "v", Load("xr", Var("i"))),
                    AugAssign(Var("s"), "+", Bin("*", Var("v"), Var("v"))),
                )
            ),
            Assign(Var("s"), Call("block_reduce_sum", (Var("s"), Var("smem")))),
            Let(
                "float",
                "inv",
                Call(
                    "rsqrtf",
                    (
                        Bin(
                            "+",
                            Bin("/", Var("s"), Cast("float", Var("cols"))),
                            Var("eps"),
                        ),
                    ),
                ),
            ),
            _scan(
                (
                    Store(
                        "orow",
                        Var("i"),
                        Bin(
                            "*",
                            Bin("*", Load("xr", Var("i")), Var("inv")),
                            Load("gamma", Var("i")),
                        ),
                    ),
                )
            ),
        ),
    )


def _batch_scan(body: tuple) -> For:
    """``for (long long i = threadIdx.x; i < inner; i += blockDim.x)``"""
    return For("i", Axis("tid"), Var("inner"), Axis("bdim"), body)


def _outer_scan(body: tuple) -> For:
    """``for (long long o = 0; o < outer; ++o)``"""
    return For("o", Num("0"), Var("outer"), Num("1"), body, incr="++")


def _channel_slice(buf: str, name: str = "base", *, const: bool = True) -> Ptr:
    """``x + o * stride + c * inner``——(outer, channels, inner) 布局下的通道切片。"""
    return Ptr(
        name,
        buf,
        Bin(
            "+",
            Bin("*", Var("o"), Var("stride")),
            Bin("*", Var("c"), Var("inner")),
        ),
        const=const,
    )


def _batch_norm_prefix() -> tuple:
    """两个批归一化内核共用的前四步：通道定位、共享内存、stride 与元素数。"""
    return (
        Let("const long long", "c", Axis("bid")),
        If(Bin(">=", Var("c"), Var("channels")), (Return(),)),
        Shared("float", "smem", REDUCE_SCRATCH),
        Let(
            "const long long",
            "stride",
            Bin("*", Var("channels"), Var("inner")),
        ),
        Let(
            "const float",
            "m",
            Cast("float", Paren(Bin("*", Var("outer"), Var("inner")))),
        ),
    )


def _batch_norm_reductions() -> tuple:
    """均值与方差两次扫描（两遍法，与 numpy.var 的参考实现一致）。"""
    return (
        Let("float", "s", Num("0.f")),
        _outer_scan(
            (
                _channel_slice("x"),
                _batch_scan((AugAssign(Var("s"), "+", Load("base", Var("i"))),)),
            )
        ),
        Assign(Var("s"), Call("block_reduce_sum", (Var("s"), Var("smem")))),
        Let("const float", "mean", Bin("/", Var("s"), Var("m"))),
        Let("float", "v", Num("0.f")),
        _outer_scan(
            (
                _channel_slice("x"),
                _batch_scan(
                    (
                        Let(
                            "const float",
                            "d",
                            Bin("-", Load("base", Var("i")), Var("mean")),
                        ),
                        AugAssign(Var("v"), "+", Bin("*", Var("d"), Var("d"))),
                    )
                ),
            )
        ),
        Assign(Var("v"), Call("block_reduce_sum", (Var("v"), Var("smem")))),
    )


def batch_norm() -> Kernel:
    """训练态批归一化：一个 block 一个通道，x 视为 ``(outer, channels, inner)``。

    这样同一个内核既能覆盖 NCHW，也能覆盖任意"某一维是通道"的秩——
    outer / inner 由调用方按 axis 折算，内核不认识原始布局。
    """
    return Kernel(
        name="batch_norm",
        params=(
            Tensor("x"),
            Tensor("gamma"),
            Tensor("beta"),
            Tensor("out", read_only=False),
            Scalar("long long", "outer"),
            Scalar("long long", "channels"),
            Scalar("long long", "inner"),
            Scalar("float", "eps"),
        ),
        body=_batch_norm_prefix()
        + _batch_norm_reductions()
        + (
            Let(
                "const float",
                "inv",
                Call(
                    "rsqrtf",
                    (Bin("+", Bin("/", Var("v"), Var("m")), Var("eps")),),
                ),
            ),
            Let("const float", "g", Load("gamma", Var("c"))),
            Let("const float", "b", Load("beta", Var("c"))),
            _outer_scan(
                (
                    _channel_slice("x", "src"),
                    _channel_slice("out", "dst", const=False),
                    _batch_scan(
                        (
                            Store(
                                "dst",
                                Var("i"),
                                Bin(
                                    "+",
                                    Bin(
                                        "*",
                                        Bin(
                                            "*",
                                            Paren(
                                                Bin(
                                                    "-",
                                                    Load("src", Var("i")),
                                                    Var("mean"),
                                                )
                                            ),
                                            Var("inv"),
                                        ),
                                        Var("g"),
                                    ),
                                    Var("b"),
                                ),
                            ),
                        )
                    ),
                )
            ),
        ),
    )


def batch_norm_stats() -> Kernel:
    """批归一化的统计量：同样的两次归约，但把 mean / var 写出（``(2, channels)``）。"""
    return Kernel(
        name="batch_norm_stats",
        params=(
            Tensor("x"),
            Tensor("out", read_only=False),
            Scalar("long long", "outer"),
            Scalar("long long", "channels"),
            Scalar("long long", "inner"),
        ),
        body=_batch_norm_prefix()
        + _batch_norm_reductions()
        + (
            If(
                Bin("==", Axis("tid"), Num("0")),
                (
                    Store("out", Var("c"), Var("mean")),
                    Store(
                        "out",
                        Bin("+", Var("channels"), Var("c")),
                        Bin("/", Var("v"), Var("m")),
                    ),
                ),
            ),
        ),
    )
