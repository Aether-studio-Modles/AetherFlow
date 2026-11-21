"""归约与 softmax：末维归约、任意轴归约、argmax 与两段扫描。"""

from __future__ import annotations

from typing import Tuple
from ..kernel import (
    Assign,
    AtomicMin,
    AugAssign,
    Axis,
    Bin,
    Call,
    Cast,
    Cond,
    Expr,
    For,
    If,
    Index,
    Kernel,
    Let,
    Load,
    Num,
    Paren,
    Ptr,
    Return,
    Scalar,
    Shared,
    Stmt,
    Store,
    Sync,
    Tensor,
    Var,
)
from ..plan import REDUCE_SCRATCH


def _reduce_last(
    name: str, reduce_fn: str, *, mean: bool = False, init: str = "0.f", combine: str = "add"
) -> Kernel:
    """末维归约内核的骨架：一个 block 负责一行，块内归约。

    ``reduce_fn`` 是方言 prelude 提供的块归约（``block_reduce_sum`` /
    ``block_reduce_max``）；``mean`` 在归约后除以列数，``init`` 是累加/取最大
    的初值，``combine`` 决定逐元素那一步是求和还是取较大。四者是归约族之间
    唯一的差别，其余（网格步进、越界退出、共享内存声明、只让 0 号线程写回）
    逐字相同。
    """
    load = Load("xr", Var("i"))
    if combine == "add":
        step: Tuple[Stmt, ...] = (AugAssign(Var("s"), "+", load),)
    elif combine == "max":
        step = (
            Let("float", "v", load),
            If(Bin(">", Var("v"), Var("s")), (Assign(Var("s"), Var("v")),)),
        )
    else:
        raise ValueError(f"未知归约方式: {combine}")
    result: Expr = Call(reduce_fn, (Var("s"), Var("smem")))
    if mean:
        result = Bin("/", result, Cast("float", Var("cols")))
    return Kernel(
        name=name,
        params=(
            Tensor("x"),
            Tensor("out", read_only=False),
            Scalar("long long", "rows"),
            Scalar("long long", "cols"),
        ),
        body=(
            Let("long long", "row", Axis("bid")),
            If(Bin(">=", Var("row"), Var("rows")), (Return(),)),
            Ptr("xr", "x", Bin("*", Var("row"), Var("cols"))),
            Shared("float", "smem", REDUCE_SCRATCH),
            Let("float", "s", Num(init)),
            For("i", Axis("tid"), Var("cols"), Axis("bdim"), step),
            Assign(Var("s"), result),
            If(
                Bin("==", Axis("tid"), Num("0")),
                (Store("out", Var("row"), Var("s")),),
            ),
        ),
    )


def reduce_sum_last() -> Kernel:
    """沿最后一维求和：一个 block 负责一行，块内归约。"""
    return _reduce_last("reduce_sum_last", "block_reduce_sum")


def reduce_max_last() -> Kernel:
    """沿最后一维求最大值。"""
    return _reduce_last("reduce_max_last", "block_reduce_max", init="-3.4e38f", combine="max")


def reduce_mean_last() -> Kernel:
    """沿最后一维求均值：归约后除以列数。

    除法在 float 里做（f16 也先读成 float），只有写回时降回存储类型。
    """
    return _reduce_last("reduce_mean_last", "block_reduce_sum", mean=True)


_AXIS_REDUCE_INIT = {
    "sum": "0.f",
    "mean": "0.f",
    "max": "-3.4e38f",
    "min": "3.4e38f",
}


def reduce_axis(kind: str) -> Kernel:
    """任意单轴归约：一个线程负责一个输出元素。

    末维归约走 ``reduce_*_last``（一个 block 一行、块内归约），快得多；非末维
    没有连续性可利用——同一行的元素跨了 ``inner`` 步，块内归约也拿不到合并
    访问。这里按 ``(outer, reduce, inner)`` 三段式直接步进：线程 ``idx`` 对应
    ``(o, i)``，沿 reduce 维串行读。它是"只支持最后一维"这条能力限制的兜底
    实现；``inner`` 窄到凑不满一个 warp、或输出平面本身填不满机器时，
    ``reduce_axis_tiled`` 会把 reduce 轴的并行度借过来。

    读入统一转 float 再累加，与末维内核一致：f16 只省存储，不省计算精度。
    """
    if kind not in _AXIS_REDUCE_INIT:
        raise ValueError(f"未知的归约类型 {kind!r}；已支持: {sorted(_AXIS_REDUCE_INIT)}")
    init = Num(_AXIS_REDUCE_INIT[kind])
    load = Cast(
        "float",
        Load(
            "x",
            Bin(
                "+",
                Bin(
                    "*",
                    Paren(
                        Bin(
                            "+",
                            Bin("*", Var("o"), Var("reduce_len")),
                            Var("r"),
                        )
                    ),
                    Var("inner"),
                ),
                Var("i"),
            ),
        ),
    )
    if kind in ("sum", "mean"):
        combine = Bin("+", Var("acc"), load)
    elif kind == "max":
        combine = Cond(Bin(">", load, Var("acc")), load, Var("acc"))
    else:
        combine = Cond(Bin("<", load, Var("acc")), load, Var("acc"))
    result = Var("acc")
    if kind == "mean":
        result = Bin("/", Var("acc"), Cast("float", Var("reduce_len")))
    return Kernel(
        name=f"reduce_{kind}_axis",
        params=(
            Tensor("x"),
            Tensor("out", read_only=False),
            Scalar("long long", "total"),
            Scalar("long long", "reduce_len"),
            Scalar("long long", "inner"),
        ),
        body=(
            Let(
                "long long",
                "idx",
                Bin(
                    "+",
                    Bin("*", Cast("long long", Axis("bid")), Axis("bdim")),
                    Axis("tid"),
                ),
            ),
            If(Bin(">=", Var("idx"), Var("total")), (Return(),)),
            Let("long long", "o", Bin("/", Var("idx"), Var("inner"))),
            Let("long long", "i", Bin("%", Var("idx"), Var("inner"))),
            Let("float", "acc", init),
            For(
                "r",
                Num("0"),
                Var("reduce_len"),
                Num("1"),
                (Assign(Var("acc"), combine),),
                type_name="long long",
            ),
            Store("out", Var("idx"), result),
        ),
    )


_AXIS_REDUCE_TILE_FN = {
    "sum": "block_reduce_sum_y",
    "mean": "block_reduce_sum_y",
    "max": "block_reduce_max_y",
    "min": "block_reduce_min_y",
}


def reduce_axis_tiled(kind: str) -> Kernel:
    """非末维归约的两维铺法：x 沿 inner、y 切 reduce 轴。

    一维铺法（``reduce_axis``）的瓶颈不是访存合并，而是并行度：输出只有
    ``outer * inner`` 个元素，reduce 轴完全串行。这里把 block 摊成
    ``(blockDim.x = inner 方向, blockDim.y = reduce 方向)``：同一 warp 里
    ``threadIdx.x`` 连续的线程读连续 ``inner`` 元素，归约由 ``blockDim.y``
    个线程分摊，最后沿 y 做一次块内树归约（``block_reduce_*_y``）。两维的
    尺寸（``tx`` / ``ty``）由 ``lower/plan.py`` 按形状给。

    网格是 ``(outer, ceil(inner / blockDim.x))``；每块只写自己那一列，
    越界线程（``i >= inner``）照常参与树归约，只是带着累加单位元。
    """
    if kind not in _AXIS_REDUCE_INIT:
        raise ValueError(f"未知的归约类型 {kind!r}；已支持: {sorted(_AXIS_REDUCE_INIT)}")
    init = Num(_AXIS_REDUCE_INIT[kind])
    load = Cast(
        "float",
        Load(
            "x",
            Bin(
                "+",
                Bin("*", Paren(Bin("+", Var("row"), Var("r"))), Var("inner")),
                Var("i"),
            ),
        ),
    )
    if kind in ("sum", "mean"):
        combine = Bin("+", Var("acc"), load)
    elif kind == "max":
        combine = Cond(Bin(">", load, Var("acc")), load, Var("acc"))
    else:
        combine = Cond(Bin("<", load, Var("acc")), load, Var("acc"))
    result = Var("acc")
    if kind == "mean":
        result = Bin("/", Var("acc"), Cast("float", Var("reduce_len")))
    return Kernel(
        name=f"reduce_{kind}_axis_tiled",
        params=(
            Tensor("x"),
            Tensor("out", read_only=False),
            Scalar("long long", "reduce_len"),
            Scalar("long long", "inner"),
        ),
        body=(
            Let("long long", "o", Cast("long long", Axis("bid", "x"))),
            Let("long long", "nx", Cast("long long", Axis("bdim", "x"))),
            Let(
                "long long",
                "i",
                Bin(
                    "+",
                    Bin("*", Cast("long long", Axis("bid", "y")), Var("nx")),
                    Cast("long long", Axis("tid", "x")),
                ),
            ),
            Let("long long", "row", Bin("*", Var("o"), Var("reduce_len"))),
            Shared("float", "smem", REDUCE_SCRATCH),
            Let("float", "acc", init),
            If(
                Bin("<", Var("i"), Var("inner")),
                (
                    For(
                        "r",
                        Cast("long long", Axis("tid", "y")),
                        Var("reduce_len"),
                        Axis("bdim", "y"),
                        (Assign(Var("acc"), combine),),
                        type_name="long long",
                    ),
                ),
            ),
            Assign(
                Var("acc"),
                Call(_AXIS_REDUCE_TILE_FN[kind], (Var("acc"), Var("smem"))),
            ),
            If(
                Bin("==", Axis("tid", "y"), Num("0")),
                (
                    If(
                        Bin("<", Var("i"), Var("inner")),
                        (
                            Store(
                                "out",
                                Bin("+", Bin("*", Var("o"), Var("inner")), Var("i")),
                                result,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def softmax_axis() -> Kernel:
    """任意单轴 softmax：一个线程负责一行，三遍扫描。

    末维走 ``softmax_last``（一个 block 一行，块内两次归约）。非末维的行在
    内存里不连续，块内归约也换不来合并访问，因此这里一个线程串行扫一整行：
    先求最大值、再求 ``exp`` 之和、最后写回。``exp`` 因此要算两遍——换来的是
    不需要 shared、也不需要行内同步。

    与末维内核一致，读入与中间量都是 float，只有 ``out`` 按存储类型写出；
    ``scale`` 是融合 pass 塞进来的标量乘子（``softmax(x, scale)``）。
    """
    base = Bin(
        "+",
        Bin("*", Bin("*", Var("o"), Var("reduce_len")), Var("inner")),
        Var("i"),
    )

    def at(offset: Expr) -> Expr:
        return Bin("+", base, Bin("*", Var("r"), Var("inner")))

    def scaled(offset: Expr) -> Expr:
        return Bin("*", Load("x", at(offset)), Var("sc"))

    return Kernel(
        name="softmax_axis",
        params=(
            Tensor("x"),
            Tensor("out", read_only=False),
            Scalar("long long", "total"),
            Scalar("long long", "reduce_len"),
            Scalar("long long", "inner"),
            Tensor("scale", type_name="float"),
        ),
        body=(
            Let(
                "long long",
                "idx",
                Bin(
                    "+",
                    Bin("*", Cast("long long", Axis("bid")), Axis("bdim")),
                    Axis("tid"),
                ),
            ),
            If(Bin(">=", Var("idx"), Var("total")), (Return(),)),
            Let("long long", "o", Bin("/", Var("idx"), Var("inner"))),
            Let("long long", "i", Bin("%", Var("idx"), Var("inner"))),
            Let("const float", "sc", Index("scale", Num("0"))),
            Let("float", "m", Num("-3.4e38f")),
            For(
                "r",
                Num("0"),
                Var("reduce_len"),
                Num("1"),
                (
                    Let("float", "v", scaled(Num("0"))),
                    Assign(
                        Var("m"),
                        Cond(Bin(">", Var("v"), Var("m")), Var("v"), Var("m")),
                    ),
                ),
                type_name="long long",
            ),
            Let("float", "s", Num("0.f")),
            For(
                "r",
                Num("0"),
                Var("reduce_len"),
                Num("1"),
                (
                    AugAssign(
                        Var("s"),
                        "+",
                        Call(
                            "expf",
                            (Bin("-", scaled(Num("0")), Var("m")),),
                        ),
                    ),
                ),
                type_name="long long",
            ),
            Let("const float", "inv", Bin("/", Num("1.f"), Var("s"))),
            For(
                "r",
                Num("0"),
                Var("reduce_len"),
                Num("1"),
                (
                    Store(
                        "out",
                        at(Num("0")),
                        Bin(
                            "*",
                            Call(
                                "expf",
                                (Bin("-", scaled(Num("0")), Var("m")),),
                            ),
                            Var("inv"),
                        ),
                    ),
                ),
                type_name="long long",
            ),
        ),
    )


def argmax_last() -> Kernel:
    """沿最后一维取最大值的下标（i32）。

    两段式：先做一次块内最大值归约，再让"自己的最大值等于全局最大值"的线程
    用 ``atomicMin`` 竞争最小下标。取最小下标是 numpy/PyTorch 的约定——并列
    最大值时返回**第一次出现**的位置。
    """
    return Kernel(
        name="argmax_last",
        params=(
            Tensor("x"),
            Tensor("out", read_only=False, type_name="int"),
            Scalar("long long", "rows"),
            Scalar("long long", "cols"),
        ),
        body=(
            Let("long long", "row", Axis("bid")),
            If(Bin(">=", Var("row"), Var("rows")), (Return(),)),
            Ptr("xr", "x", Bin("*", Var("row"), Var("cols"))),
            Shared("float", "smem", REDUCE_SCRATCH),
            Shared("int", "ibuf", 1),
            Let("float", "best", Num("-3.4e38f")),
            Let("int", "best_i", Num("-1")),
            For(
                "i",
                Axis("tid"),
                Var("cols"),
                Axis("bdim"),
                (
                    Let("float", "v", Load("xr", Var("i"))),
                    If(
                        Bin(">", Var("v"), Var("best")),
                        (
                            Assign(Var("best"), Var("v")),
                            Assign(Var("best_i"), Cast("int", Var("i"))),
                        ),
                    ),
                ),
            ),
            Let(
                "float",
                "m",
                Call("block_reduce_max", (Var("best"), Var("smem"))),
            ),
            If(
                Bin("==", Axis("tid"), Num("0")),
                (Store("ibuf", Num("0"), Cast("int", Var("cols")), raw=True),),
            ),
            Sync(),
            If(
                Bin("==", Var("best"), Var("m")),
                (AtomicMin("ibuf", Num("0"), Var("best_i")),),
            ),
            Sync(),
            If(
                Bin("==", Axis("tid"), Num("0")),
                (Store("out", Var("row"), Index("ibuf", Num("0")), raw=True),),
            ),
        ),
    )


def softmax_last() -> Kernel:
    """沿最后一维的 softmax：一个 block 负责一行，两段归约 + 三段重扫。"""

    def _scan(body: tuple) -> For:
        return For("i", Axis("tid"), Var("cols"), Axis("bdim"), body)

    def _scaled(i: Expr) -> Expr:
        """``%LD%(xr[i]) * sc``"""
        return Bin("*", Load("xr", i), Var("sc"))

    return Kernel(
        name="softmax_last",
        params=(
            Tensor("x"),
            Tensor("out", read_only=False),
            Scalar("long long", "rows"),
            Scalar("long long", "cols"),
            Tensor("scale", type_name="float"),
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
            Let("const float", "sc", Index("scale", Num("0"))),
            Shared("float", "smem", REDUCE_SCRATCH),
            Let("float", "m", Num("-3.4e38f")),
            _scan(
                (
                    Let("float", "v", _scaled(Var("i"))),
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
            Assign(Var("m"), Call("block_reduce_max", (Var("m"), Var("smem")))),
            Let("float", "s", Num("0.f")),
            _scan(
                (
                    AugAssign(
                        Var("s"),
                        "+",
                        Call(
                            "expf",
                            (Bin("-", _scaled(Var("i")), Var("m")),),
                        ),
                    ),
                )
            ),
            Assign(Var("s"), Call("block_reduce_sum", (Var("s"), Var("smem")))),
            Let("float", "inv", Bin("/", Num("1.f"), Var("s"))),
            _scan(
                (
                    Store(
                        "orow",
                        Var("i"),
                        Bin(
                            "*",
                            Call(
                                "expf",
                                (
                                    Bin(
                                        "-",
                                        _scaled(Var("i")),
                                        Var("m"),
                                    ),
                                ),
                            ),
                            Var("inv"),
                        ),
                    ),
                )
            ),
        ),
    )
