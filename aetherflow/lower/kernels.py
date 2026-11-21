"""参考内核的 L2 描述。

``ew_binary`` 是最简单的并行映射（网格步进 + 一维索引），``reduce_sum_last``
引入归约、共享内存与提前退出；``ew_unary`` / ``ew_step`` 覆盖三元条件与
一元负号，``ew_broadcast`` 覆盖定长数组与倒序循环，``softmax_last`` 把两段
归约与三段重扫合在同一个内核里。这些合起来正好覆盖 L2 目前需要的全部节点，
也正好对应 ``cuda/codegen.py`` 里的手写模板，便于用"去空白后逐字相等"验收。

matmul 不在这里：它走 cuBLAS / hipBLAS，没有内核模板，也就没有可下降的内核。
线性代数的可移植性由库契约保证，不是 L2 的职责。
"""

from __future__ import annotations

from .kernel import (
    ArrayDecl,
    Assign,
    AtomicAdd,
    AtomicMin,
    AugAssign,
    Axis,
    Bin,
    Call,
    Cast,
    Cond,
    Continue,
    Expr,
    For,
    If,
    Index,
    Kernel,
    Let,
    LetMulti,
    Load,
    Neg,
    Not,
    Narrow,
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
    Widen,
)
from .indexing import Layout
from .plan import CSR_SCAN_SLOTS, REDUCE_SCRATCH


__all__ = [
    "grid_stride",
    "ew_binary",
    "ew_unary",
    "ew_step",
    "ew_broadcast",
    "reduce_sum_last",
    "reduce_max_last",
    "reduce_mean_last",
    "reduce_axis",
    "reduce_axis_tiled",
    "argmax_last",
    "softmax_last",
    "layernorm_last",
    "layernorm_residual_last",
    "rmsnorm_last",
    "silu_gate",
    "batch_norm",
    "batch_norm_stats",
    "concat_copy",
    "dropout_mask",
    "spike_offsets",
    "spike_csr",
    "spike_csr_serial",
    "sparse_linear",
    "sparse_linear_weight_grad",
    "spike_linear",
    "spike_linear_weight_grad",
    "scan_linear",
    "scan_linear_backward",
    "scan_linear_chunk_sum",
    "scan_linear_chunk",
    "scan_linear_chunk_sum_backward",
    "scan_linear_chunk_backward",
    "scan_linear_last",
    "scan_linear_chunk_last",
    "conv2d_input_grad",
    "conv2d_weight_grad",
    "maxpool2d",
    "maxpool2d_backward",
    "maxpool2d_indices",
    "avgpool2d",
    "avgpool2d_backward",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool2d_backward",
    "im2col",
    "conv2d_tiled",
    "conv2d_grouped",
    "conv2d_implicit",
    "winograd_weight_transform",
    "winograd_input_transform",
    "winograd_output_transform",
    "winograd_fused",
    "attention_fwd",
]


BINARY_OP = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def grid_stride(limit: Expr, *body: Stmt) -> tuple:
    """一维网格步进的块内索引骨架，全部逐元素内核共用。

    ``i = blockIdx*blockDim + tid``、``stride = gridDim*blockDim``，
    然后 ``for (; i < limit; i += stride)``。三个量缺一不可：

    - 索引用 64 位：``blockIdx*blockDim`` 在超大张量上会越过 32 位；
    - stride 必须是运行期的网格大小——块数被 ``max_blocks`` 截断时，
      多出来的元素只能靠步进补上，写成 ``blockDim`` 会漏算；
    - ``limit`` 交给调用方：标量路径是元素数 ``n``，向量路径是
      ``n / 4``（索引是向量下标）。

    ``elementwise.build_chain`` 用的是同一个骨架，所以融合链与单算子内核
    的循环形状逐字相同。
    """
    return (
        Let(
            "long long",
            "i",
            Bin(
                "+",
                Bin("*", Cast("long long", Axis("bid")), Axis("bdim")),
                Axis("tid"),
            ),
        ),
        Let(
            "long long",
            "stride",
            Bin("*", Cast("long long", Axis("gdim")), Axis("bdim")),
        ),
        For("i", None, limit, Var("stride"), tuple(body)),
    )


def ew_binary(kind: str, *, scalar_a: bool = False, scalar_b: bool = False) -> Kernel:
    """逐元素二元内核（网格步进）。"""
    if kind not in BINARY_OP:
        raise ValueError(f"未知的二元运算 {kind!r}；已支持: {sorted(BINARY_OP)}")

    # 标量一侧仍然收设备指针、在内核里取 [0]：传主机 float 会强制同步，
    # 也会让整张图无法被捕获。
    a_index = Num("0") if scalar_a else Var("i")
    b_index = Num("0") if scalar_b else Var("i")
    # 括号显式写进 L2（模板原本是 "(A) op (B)"），打印器不推断优先级。
    value = Bin(
        BINARY_OP[kind],
        Paren(Load("a", a_index)),
        Paren(Load("b", b_index)),
    )

    return Kernel(
        name="ew_bin",
        params=(
            Tensor("a"),
            Tensor("b"),
            Tensor("out", read_only=False),
            Scalar("long long", "n"),
        ),
        body=grid_stride(
            Var("n"),
            Store("out", Var("i"), value),
        ),
    )


def _reduce_last(name: str, reduce_fn: str, *, mean: bool = False,
                 init: str = "0.f", combine: str = "add") -> Kernel:
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
    return _reduce_last(
        "reduce_max_last", "block_reduce_max", init="-3.4e38f", combine="max"
    )


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


# 一元算子的表达式。``x`` 是内核里先读成 float 的局部变量。
# 与 ``cuda/codegen.py`` 的 ``UNARY_EXPR`` 逐 token 对齐：那边的字符串是
# 手写对照物，这里是 L2 的结构化版本。
#
# 每个构造器接受一个操作数表达式，而不是写死 ``Var("x")``：逐元素融合
# （``elementwise.py``）把同一张表当作表达式模板复用，只是操作数换成了
# 前一步的结果。括号由调用方按需加（``Paren``），构造器不加——否则单算子
# 内核就会多出一层与手写模板不对齐的括号。
UNARY_EXPR = {
    "relu": lambda x: Cond(
        Bin(">", x, Num("0.f")), x, Num("0.f")
    ),
    "silu": lambda x: Bin(
        "/",
        x,
        Paren(Bin("+", Num("1.f"), Call("expf", (Neg(x),)))),
    ),
    "gelu": lambda x: Bin(
        "*",
        Bin("*", Num("0.5f"), x),
        Paren(
            Bin(
                "+",
                Num("1.f"),
                Call(
                    "tanhf",
                    (
                        Bin(
                            "*",
                            Num("0.7978845608028654f"),
                            Paren(
                                Bin(
                                    "+",
                                    x,
                                    Bin(
                                        "*",
                                        Bin(
                                            "*",
                                            Bin("*", Num("0.044715f"), x),
                                            x,
                                        ),
                                        x,
                                    ),
                                )
                            ),
                        ),
                    ),
                ),
            )
        ),
    ),
    "exp": lambda x: Call("expf", (x,)),
    "log": lambda x: Call("logf", (x,)),
    "sqrt": lambda x: Call("sqrtf", (x,)),
    "neg": lambda x: Neg(x),
    "tanh": lambda x: Call("tanhf", (x,)),
    "sigmoid": lambda x: Bin(
        "/",
        Num("1.f"),
        Paren(Bin("+", Num("1.f"), Call("expf", (Neg(x),)))),
    ),
    "erf": lambda x: Call("erff", (x,)),
}


def ew_unary(kind: str) -> Kernel:
    """逐元素一元内核：先把输入读成 float，算完再写回存储类型。"""
    try:
        build = UNARY_EXPR[kind]
    except KeyError:
        raise ValueError(
            f"未知的一元运算 {kind!r}；已支持: {sorted(UNARY_EXPR)}"
        ) from None

    return Kernel(
        name="ew_un",
        params=(
            Tensor("in"),
            Tensor("out", read_only=False),
            Scalar("long long", "n"),
        ),
        body=grid_stride(
            Var("n"),
            Let("float", "x", Load("in", Var("i"))),
            Store("out", Var("i"), build(Var("x"))),
        ),
    )


def ew_cast(out_type: str, out_store: Optional[str] = None) -> Kernel:
    """跨存储宽度的元素类型转换：读按当前方言，写按 ``out_type``。

    这是唯一一个"输入与输出存储类型不同"的内核，所以两端的类型分别给出：
    ``in`` 跟随渲染时指定的 dtype，``out`` 由 ``out_type`` 钉死。

    ``out_store`` 是写回时的舍入函数名（方言提供的 ``af_st_h`` 之类）。为 ``None``
    表示"直接裸赋值"——目标比源宽时数值本来就是从窄类型升上来的 float，
    再套一层本类型的 store helper 只会把值先舍回窄类型。
    """
    value = Var("x")
    written = Call(out_store, (value,)) if out_store else value
    return Kernel(
        name="ew_cast",
        params=(
            Tensor("in"),
            Tensor("out", read_only=False, type_name=out_type),
            Scalar("long long", "n"),
        ),
        body=grid_stride(
            Var("n"),
            Let("float", "x", Load("in", Var("i"))),
            Store("out", Var("i"), written, raw=True),
        ),
    )


def ew_step() -> Kernel:
    """阶跃：大于阈值写 1，否则写 0。比其它一元算子多一个标量参数。"""
    return Kernel(
        name="ew_step",
        params=(
            Tensor("in"),
            Tensor("out", read_only=False),
            Scalar("long long", "n"),
            Scalar("float", "thr"),
        ),
        body=grid_stride(
            Var("n"),
            Store(
                "out",
                Var("i"),
                Cond(
                    Bin(">", Load("in", Var("i")), Var("thr")),
                    Num("1.f"),
                    Num("0.f"),
                ),
            )
        ),
    )


def ew_clamp() -> Kernel:
    """按标量上下界的逐元素截断。

    上下界用两个 ``int`` 开关而不是把 ``None`` 编进源码：生成期的分支会把
    同一个算子的四种形态（仅 min / 仅 max / 双侧 / 无界）拆成四份内核，
    而"无界"其实不合法。运行期开关只多两次比较，源码只有一份。

    用 ``cond ? bound : value`` 而不用 ``fminf`` / ``fmaxf``：后者对 NaN
    有"返回另一侧"的语义（``fminf(NaN, 1) == 1``），而 ``clamp(NaN)`` 必须
    仍是 NaN。比较为假正好把 NaN 原样透传。
    """
    x = Var("x")
    # 先夹上界，再在结果上夹下界；两侧的顺序不影响结果（区间已保证 lo <= hi）。
    upper = Cond(Bin("&&", Var("has_hi"), Bin(">", x, Var("hi"))), Var("hi"), x)
    lower = Cond(Bin("&&", Var("has_lo"), Bin("<", Var("v"), Var("lo"))), Var("lo"), Var("v"))
    return Kernel(
        name="ew_clamp",
        params=(
            Tensor("in"),
            Tensor("out", read_only=False),
            Scalar("long long", "n"),
            Scalar("int", "has_lo"),
            Scalar("int", "has_hi"),
            Scalar("float", "lo"),
            Scalar("float", "hi"),
        ),
        body=grid_stride(
            Var("n"),
            Let("float", "x", Load("in", Var("i"))),
            Let("float", "v", upper),
            Store("out", Var("i"), lower),
        ),
    )


def ew_mod_int(*, scalar_a: bool = False, scalar_b: bool = False) -> Kernel:
    """i32 取模，且必须是 **floor 语义**（C 的 ``%`` 是截断取余）。

    修正只有一句：``r != 0 && (r < 0) != (y < 0)`` 时 ``r += y``。三种符号
    组合都要成立——``-7 % 3`` 要给 2 而不是 -1，``7 % -3`` 要给 -2 而不是 1。
    这与参考实现的 ``np.mod`` 逐位一致，是 Engram 哈希表下标正确的前提。

    整数内核不经过 dtype 的存取 helper（``af_ld_f`` / ``af_st_h`` 都是浮点
    语义），所以参数用 ``type_name="int"`` 钉死，读写走裸 ``Index`` 与
    ``raw=True`` 的 ``Store``；打印时仍需要一个 dtype 参数，取 F32 只为拿到
    prelude，不影响整数表达。
    """
    a_index = Num("0") if scalar_a else Var("i")
    b_index = Num("0") if scalar_b else Var("i")
    r = Var("r")
    y = Var("y")
    return Kernel(
        name="ew_mod",
        params=(
            Tensor("a", type_name="int"),
            Tensor("b", type_name="int"),
            Tensor("out", read_only=False, type_name="int"),
            Scalar("long long", "n"),
        ),
        body=grid_stride(
            Var("n"),
            Let("int", "x", Index("a", a_index)),
            Let("int", "y", Index("b", b_index)),
            Let("int", "r", Bin("%", Var("x"), y)),
            If(
                Bin(
                    "&&",
                    Bin("!=", r, Num("0")),
                    Bin(
                        "!=",
                        Paren(Bin("<", r, Num("0"))),
                        Paren(Bin("<", y, Num("0"))),
                    ),
                ),
                (Assign(r, Bin("+", r, y)),),
            ),
            Store("out", Var("i"), r, raw=True),
        ),
    )


def silu_gate() -> Kernel:
    """门控 SiLU（SwiGLU）：``silu(x) * gate``，一个内核算完、不落中间张量。"""
    xv = Var("xv")
    return Kernel(
        name="silu_gate",
        params=(
            Tensor("x"),
            Tensor("gate"),
            Tensor("out", read_only=False),
            Scalar("long long", "n"),
        ),
        body=grid_stride(
            Var("n"),
            Let("float", "xv", Load("x", Var("i"))),
            Store(
                "out",
                Var("i"),
                Bin(
                    "*",
                    Bin(
                        "/",
                        xv,
                        Paren(Bin("+", Num("1.f"), Call("expf", (Neg(xv),)))),
                    ),
                    Load("gate", Var("i")),
                ),
            ),
        ),
    )


def ew_broadcast(kind: str) -> Kernel:
    """支持 numpy 多维广播的逐元素二元内核。

    广播靠"对齐后的 stride"：被广播的维度 stride 为 0，同一个输入元素会被
    多个输出位置复用。维度上限 4，超过时由后端拒绝。
    """
    if kind not in BINARY_OP:
        raise ValueError(f"未知的二元运算 {kind!r}；已支持: {sorted(BINARY_OP)}")

    value = Bin(
        BINARY_OP[kind],
        Paren(Load("a", Var("oa"))),
        Paren(Load("b", Var("ob"))),
    )

    def _dims(prefix: str) -> tuple:
        return tuple(Var(f"{prefix}{k}") for k in range(4))

    params = [Tensor("a"), Tensor("b"), Tensor("out", read_only=False)]
    params.append(Scalar("long long", "n"))
    params.append(Scalar("int", "ndim"))
    for prefix in ("d", "a", "b"):
        params.extend(Scalar("long long", v.name) for v in _dims(prefix))

    return Kernel(
        name="ew_bcast",
        params=tuple(params),
        body=(
            ArrayDecl("long long", "shape", 4, _dims("d")),
            ArrayDecl("long long", "sa", 4, _dims("a")),
            ArrayDecl("long long", "sb", 4, _dims("b")),
            Let(
                "long long",
                "i",
                Bin(
                    "+",
                    Bin("*", Cast("long long", Axis("bid")), Axis("bdim")),
                    Axis("tid"),
                ),
            ),
            Let(
                "long long",
                "step",
                Bin("*", Cast("long long", Axis("gdim")), Axis("bdim")),
            ),
            For(
                "i",
                None,
                Var("n"),
                Var("step"),
                (
                    LetMulti(
                        "long long",
                        (
                            ("rem", Var("i")),
                            ("oa", Num("0")),
                            ("ob", Num("0")),
                        ),
                    ),
                    For(
                        "d",
                        Bin("-", Var("ndim"), Num("1")),
                        Num("0"),
                        Num("1"),
                        (
                            Let(
                                "long long",
                                "c",
                                Bin(
                                    "%",
                                    Var("rem"),
                                    Index("shape", Var("d")),
                                ),
                            ),
                            AugAssign(
                                Var("rem"), "/", Index("shape", Var("d"))
                            ),
                            AugAssign(
                                Var("oa"),
                                "+",
                                Bin(
                                    "*",
                                    Var("c"),
                                    Index("sa", Var("d")),
                                ),
                            ),
                            AugAssign(
                                Var("ob"),
                                "+",
                                Bin(
                                    "*",
                                    Var("c"),
                                    Index("sb", Var("d")),
                                ),
                            ),
                        ),
                        cmp=">=",
                        incr="--",
                        type_name="int",
                    ),
                    Store("out", Var("i"), value),
                ),
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
            Assign(
                Var("m"), Call("block_reduce_max", (Var("m"), Var("smem")))
            ),
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
            Assign(
                Var("s"), Call("block_reduce_sum", (Var("s"), Var("smem")))
            ),
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
            (
                Bin("+", Bin("/", var, Cast("float", Var("cols"))), Var("eps")),
            ),
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
            Assign(
                Var("s"), Call("block_reduce_sum", (Var("s"), Var("smem")))
            ),
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
            Assign(
                Var("v"), Call("block_reduce_sum", (Var("v"), Var("smem")))
            ),
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
            (
                Bin("+", Bin("/", var, Cast("float", Var("cols"))), Var("eps")),
            ),
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
            Assign(
                Var("s"), Call("block_reduce_sum", (Var("s"), Var("smem")))
            ),
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
            Assign(
                Var("v"), Call("block_reduce_sum", (Var("v"), Var("smem")))
            ),
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
            Assign(
                Var("s"), Call("block_reduce_sum", (Var("s"), Var("smem")))
            ),
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
        Assign(
            Var("v"), Call("block_reduce_sum", (Var("v"), Var("smem")))
        ),
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
        Scalar("int", "C"),
        Scalar("int", "H"),
        Scalar("int", "W"),
    ) + dims + (Scalar("long long", "total"),) + extra


def _flat_loop(
    body: tuple, *, index: str = "idx", count: str = "total", stride: str = "step"
) -> tuple:
    """``step = ...; for (long long idx = ...; idx < total; idx += step)``"""
    return (
        Let(
            "const long long",
            stride,
            Bin("*", Cast("long long", Axis("gdim")), Axis("bdim")),
        ),
        For(
            index,
            Bin(
                "+",
                Bin("*", Cast("long long", Axis("bid")), Axis("bdim")),
                Axis("tid"),
            ),
            Var(count),
            Var(stride),
            body,
        ),
    )


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


def _clamp_index(var: str, limit: str) -> Expr:
    """把 ``var`` 夹到 ``[0, limit - 1]``。

    只在越界已经记进 ``err`` 之后使用：夹取是为了让这次执行不非法访存，
    真正让主机侧抛异常的是那个标志，不是这里的结果。
    """
    return Cond(
        Bin("<", Var(var), Num("0")),
        Num("0"),
        Cond(
            Bin(">=", Var(var), Var(limit)),
            Bin("-", Var(limit), Num("1")),
            Var(var),
        ),
    )


def _event_segment() -> Tuple[Stmt, ...]:
    """切出第 ``s`` 步的事件区间 ``[lo, hi)``，越界时记 ``err`` 并夹回。

    捕获期不能把标量读回主机（读回会打断捕获），所以错误统一记在设备端的
    ``err[0]`` 上，由后端 ``_check_ids`` 在需要时同步并报错——与 embedding
    的越界检查走同一条通路。
    """
    lo, hi = Var("lo"), Var("hi")
    return (
        Let("long long", "lo", Index("offsets", Var("s"))),
        Let("long long", "hi", Index("offsets", Bin("+", Var("s"), Num("1")))),
        If(
            Bin(
                "||",
                Bin("||", Bin("<", lo, Num("0")), Bin(">", hi, Var("nnz"))),
                Bin("<", hi, lo),
            ),
            (Store("err", Num("0"), Num("1"), raw=True),),
        ),
        If(Bin("<", lo, Num("0")), (Assign(lo, Num("0")),)),
        If(Bin(">", hi, Var("nnz")), (Assign(hi, Var("nnz")),)),
        If(Bin("<", hi, lo), (Assign(hi, lo),)),
    )


def _event_row_check() -> Tuple[Stmt, ...]:
    """检查事件下标 ``r`` 落在 ``[0, rows)``，越界记 ``err`` 并夹回。"""
    row = Var("r")
    return (
        Let("long long", "r", Index("indices", Var("e"))),
        If(
            Bin("||", Bin("<", row, Num("0")), Bin(">=", row, Var("rows"))),
            (
                Store("err", Num("0"), Num("1"), raw=True),
                Assign(row, _clamp_index("r", "rows")),
            ),
        ),
    )


def spike_offsets() -> Kernel:
    """统计每行非零数并写 CSR 起点表（长度 ``T+1``）。

    两阶段放在**同一个 block** 里：先让每个线程跨步数若干行的非零数、写进
    ``offsets[t+1]``，``__syncthreads()`` 之后由 0 号线程串行做前缀和。前缀和
    因此只有 ``T`` 步，而不是旧实现里"单线程扫完 ``T×K`` 再累加"——大 ``T`` 下
    差的正是 ``K`` 倍的串行读。

    语义没有变：起点表的定义、越界行为与之前完全一致，``offsets`` 仍是唯一
    输出、仍然单输出，因此不需要新的 L2 能力。
    """
    hit = Bin(
        "!=",
        Load("spikes", _T_WIDTH.offset(("t", "k"), cast=None)),
        Num("0.f"),
    )
    return Kernel(
        name="spike_offsets",
        params=(
            Tensor("spikes"),
            Tensor("offsets", read_only=False, type_name="int"),
            Scalar("long long", "width"),
            Scalar("long long", "steps"),
        ),
        body=(
            # 阶段 1：并行数每行的非零数，先放在 offsets[t + 1] 上
            For(
                "t",
                Axis("tid"),
                Var("steps"),
                Axis("bdim"),
                (
                    Let("long long", "c", Num("0")),
                    For(
                        "k",
                        Num("0"),
                        Var("width"),
                        Num("1"),
                        (
                            If(
                                hit,
                                (
                                    Assign(
                                        Var("c"),
                                        Bin("+", Var("c"), Num("1")),
                                    ),
                                ),
                            ),
                        ),
                        incr="++",
                    ),
                    Store(
                        "offsets", Bin("+", Var("t"), Num("1")), Var("c"), raw=True
                    ),
                ),
            ),
            Sync(),
            # 阶段 2：0 号线程把 T 个计数就地变成前缀和
            If(
                Bin("==", Axis("tid"), Num("0")),
                (
                    Let("long long", "acc", Num("0")),
                    For(
                        "t",
                        Num("0"),
                        Var("steps"),
                        Num("1"),
                        (
                            Let(
                                "long long",
                                "cur",
                                Index(
                                    "offsets", Bin("+", Var("t"), Num("1"))
                                ),
                            ),
                            Store("offsets", Var("t"), Var("acc"), raw=True),
                            Assign(Var("acc"), Bin("+", Var("acc"), Var("cur"))),
                        ),
                        incr="++",
                    ),
                    Store("offsets", Var("steps"), Var("acc"), raw=True),
                ),
            ),
        ),
    )


def spike_csr_serial() -> Kernel:
    """``spike_csr`` 的旧铺法：一个线程负责一行，沿列号串行。

    保留只为对照（5.176 的 ``benchmarks/spike_csr_scan.py`` 在实例上盖掉
    ``_use_scan_csr`` 后走这里），生产路径恒为 :func:`spike_csr`。数值语义
    两者相同——都是按列号递增写同一个 ``indices``。
    """
    hit = Bin(
        "!=",
        Load("spikes", _T_WIDTH.offset(("t", "k"), cast=None)),
        Num("0.f"),
    )
    lo, hi = Var("lo"), Var("hi")
    return Kernel(
        name="spike_csr_serial",
        params=(
            Tensor("spikes"),
            Tensor("offsets", type_name="int"),
            Tensor("indices", read_only=False, type_name="int"),
            Scalar("long long", "capacity"),
            Scalar("long long", "width"),
            Scalar("long long", "steps"),
            Tensor("err", read_only=False, type_name="int"),
        ),
        body=grid_stride(
            Var("steps"),
            *(
                Let("const long long", "t", Var("i")),
                Let("long long", "lo", Index("offsets", Var("t"))),
                Let(
                    "long long",
                    "hi",
                    Index("offsets", Bin("+", Var("t"), Num("1"))),
                ),
                If(
                    Bin(
                        "||",
                        Bin("||", Bin("<", lo, Num("0")), Bin(">", hi, Var("capacity"))),
                        Bin("<", hi, lo),
                    ),
                    (Store("err", Num("0"), Num("1"), raw=True),),
                ),
                If(Bin("<", lo, Num("0")), (Assign(lo, Num("0")),)),
                If(
                    Bin(">", hi, Var("capacity")),
                    (Assign(hi, Var("capacity")),),
                ),
                If(Bin("<", hi, lo), (Assign(hi, lo),)),
                Let("long long", "at", Var("lo")),
                For(
                    "k",
                    Num("0"),
                    Var("width"),
                    Num("1"),
                    (
                        If(
                            hit,
                            (
                                If(
                                    Bin("<", Var("at"), Var("hi")),
                                    (
                                        Store(
                                            "indices", Var("at"), Var("k"), raw=True
                                        ),
                                        Assign(
                                            Var("at"),
                                            Bin("+", Var("at"), Num("1")),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                    incr="++",
                ),
            ),
        ),
    )


def spike_csr() -> Kernel:
    """按起点表把每行的非零列号写进 ``indices[capacity]``。

    一个 **block** 负责一行（5.176）：行内必须按列号递增写，才能与稠密矩阵
    乘的累加顺序一致（``sparse_linear`` 逐事件累加），所以并行化不能改成
    "一个线程一个列号"。做法是分段扫——每段 ``blockDim.x`` 个列号，段内做一次
    块内排他前缀和（``block_scan_sum``），第 ``tid`` 个线程拿到"本段排在自己
    前面的命中数"，写位置是 ``lo + 已写 + 前缀``。段间串行、段内并行，递增
    顺序因此天然保持；段数 ``ceil(width / blockDim.x)`` 对全块一致，不会出现
    一半线程跳过 ``__syncthreads`` 的情况。

    起点表越界（倒序或超过容量）记 ``err`` 并夹回：夹取只保证不非法访存，
    主机侧仍会抛错——与 ``sparse_linear`` 的越界检查同一条通路。
    """
    t = Var("t")
    tid = Axis("tid")
    bdim = Axis("bdim")
    lo, hi = Var("lo"), Var("hi")
    k = Var("k")
    segment = Var("seg")
    prefix = Var("pre")
    written = Var("written")

    # ``? :`` 的分支在 C 里是短路的：``k`` 越界时不读 ``spikes``，所以这里
    # 不必先夹取索引。命中记 1、未命中记 0，恰好就是前缀和的被扫描值。
    counted = Cond(
        Bin("<", k, Var("width")),
        Cond(
            Bin(
                "!=",
                Load("spikes", Bin("+", Bin("*", t, Var("width")), k)),
                Num("0.f"),
            ),
            Num("1"),
            Num("0"),
        ),
        Num("0"),
    )
    position = Bin("+", Bin("+", lo, written), prefix)
    total_slot = Index("smem", bdim)
    taken = Var("taken")
    written_hit = Bin("==", taken, Num("1"))

    return Kernel(
        name="spike_csr",
        params=(
            Tensor("spikes"),
            Tensor("offsets", type_name="int"),
            Tensor("indices", read_only=False, type_name="int"),
            Scalar("long long", "capacity"),
            Scalar("long long", "width"),
            Scalar("long long", "steps"),
            Tensor("err", read_only=False, type_name="int"),
        ),
        body=(
            Let("const long long", "t", Cast("long long", Axis("bid"))),
            Shared("int", "smem", CSR_SCAN_SLOTS),
            Let("long long", "lo", Index("offsets", t)),
            Let("long long", "hi", Index("offsets", Bin("+", t, Num("1")))),
            If(
                Bin(
                    "||",
                    Bin(
                        "||",
                        Bin("<", lo, Num("0")),
                        Bin(">", hi, Var("capacity")),
                    ),
                    Bin("<", hi, lo),
                ),
                (Store("err", Num("0"), Num("1"), raw=True),),
            ),
            If(Bin("<", lo, Num("0")), (Assign(lo, Num("0")),)),
            If(
                Bin(">", hi, Var("capacity")),
                (Assign(hi, Var("capacity")),),
            ),
            If(Bin("<", hi, lo), (Assign(hi, lo),)),
            Let(
                "long long",
                "segments",
                # 打印器不推断优先级：分子的加法必须自带括号，否则会变成
                # ``width + blockDim.x - 1 / blockDim.x``。
                Bin(
                    "/",
                    Paren(Bin("-", Bin("+", Var("width"), bdim), Num("1"))),
                    bdim,
                ),
            ),
            Let("long long", "written", Num("0")),
            For(
                "seg",
                Num("0"),
                Var("segments"),
                Num("1"),
                (
                    Let(
                        "long long",
                        "k",
                        Bin("+", Bin("*", segment, bdim), Cast("long long", tid)),
                    ),
                    Let("int", "taken", counted),
                    Let(
                        "int",
                        "pre",
                        Call("block_scan_sum", (taken, Var("smem"))),
                    ),
                    If(
                        Bin("&&", written_hit, Bin("<", position, hi)),
                        (Store("indices", position, Cast("int", k), raw=True),),
                    ),
                    If(
                        Bin("&&", written_hit, Bin(">=", position, hi)),
                        (Store("err", Num("0"), Num("1"), raw=True),),
                    ),
                    Assign(written, Bin("+", written, total_slot)),
                ),
                incr="++",
                type_name="long long",
            ),
            # 参考实现要求"本行命中数 == hi - lo"；多了在段内已经记过，少了
            # 只能等全部段扫完才知道。两侧都对上才算 offsets 与脉冲一致。
            If(
                Bin("!=", Bin("+", lo, written), hi),
                (Store("err", Num("0"), Num("1"), raw=True),),
            ),
        ),
    )


def sparse_linear() -> Kernel:
    """事件驱动稀疏线性：``out[s, n] = Σ_e values[e] * W[indices[e], n]``。

    ``e`` 扫的是第 ``s`` 步**真正发放的事件**（CSR 的一段），而不是全部 K 个
    神经元——这就是"事件驱动"的全部含义：稠密的 ``(S, K)`` 脉冲矩阵从不
    物化，读的事件数只与发放率成正比。

    输出本身是稠密的 ``(S, N)``，所以一个线程负责一个输出元素；选"按输出
    收集"而不是"按事件散射"，是为了每个输出只写一次、不需要原子加。
    """
    weight_at = _ROW_COLS.offset(("r", "n"), cast=None)
    return Kernel(
        name="sparse_linear",
        params=(
            Tensor("weights"),
            Tensor("indices", type_name="int"),
            Tensor("values"),
            Tensor("offsets", type_name="int"),
            Tensor("out", read_only=False),
            Scalar("long long", "cols"),
            Scalar("long long", "rows"),
            Scalar("long long", "nnz"),
            Scalar("long long", "total"),
            Tensor("err", read_only=False, type_name="int"),
        ),
        body=grid_stride(
            Var("total"),
            *(
                Let("const long long", "s", Bin("/", Var("i"), Var("cols"))),
                Let("const long long", "n", Bin("%", Var("i"), Var("cols"))),
                *_event_segment(),
                Let("float", "acc", Num("0.f")),
                For(
                    "e",
                    Var("lo"),
                    Var("hi"),
                    Num("1"),
                    (
                        *_event_row_check(),
                        Assign(
                            Var("acc"),
                            Bin(
                                "+",
                                Var("acc"),
                                Bin(
                                    "*",
                                    Load("values", Var("e")),
                                    Load("weights", weight_at),
                                ),
                            ),
                        ),
                    ),
                    incr="++",
                ),
                Store("out", Var("i"), Var("acc")),
            ),
        ),
    )


def sparse_linear_weight_grad() -> Kernel:
    """``sparse_linear`` 对权重的梯度：把 ``values[e] * dout[s]`` 散射回 ``idx[e]`` 行。

    同一行会被多个事件命中（同一神经元在 batch 内多次发放、或同一 batch 里
    重复的 index），写回必须原子加；输出缓冲由后端先清零。
    """
    dst = _ROW_COLS.offset(("r", "n"), cast=None)
    return Kernel(
        name="sparse_linear_weight_grad",
        params=(
            Tensor("dout"),
            Tensor("indices", type_name="int"),
            Tensor("values"),
            Tensor("offsets", type_name="int"),
            Tensor("dweights", read_only=False),
            Scalar("long long", "cols"),
            Scalar("long long", "rows"),
            Scalar("long long", "nnz"),
            Scalar("long long", "total"),
            Tensor("err", read_only=False, type_name="int"),
        ),
        body=grid_stride(
            Var("total"),
            *(
                Let("const long long", "s", Bin("/", Var("i"), Var("cols"))),
                Let("const long long", "n", Bin("%", Var("i"), Var("cols"))),
                *_event_segment(),
                Let("const float", "g", Load("dout", Var("i"))),
                For(
                    "e",
                    Var("lo"),
                    Var("hi"),
                    Num("1"),
                    (
                        *_event_row_check(),
                        AtomicAdd(
                            "dweights",
                            dst,
                            Bin("*", Load("values", Var("e")), Var("g")),
                        ),
                    ),
                    incr="++",
                ),
            ),
        ),
    )


def sparse_linear_values_grad() -> Kernel:
    """``sparse_linear`` 对事件幅度的梯度：``dvalues[e] = dout[s] · W[r]``。

    与权重梯度"按输出散射、重复下标要原子加"不同，这里**按步并行**：第
    ``i`` 个线程独占第 ``i`` 步的全部事件，逐个算一个长度 ``cols`` 的点积，
    直接写 ``dvalues[e]``。每个事件只被一个线程写一次，因此不需要原子加。

    并行度取步数而不是事件数，是为了避免"事件 → 步"的映射：那需要在设备上
    对 ``offsets`` 做二分查找（L2 没有 while，只能展开成定长 for），而这里
    ``offsets`` 只需顺序读一遍。代价是同一 batch 内步数少时并行度偏低、且
    空步的线程直接空转——两点都比多一层二分简单。
    """
    dout_at = Bin("+", Bin("*", Var("i"), Var("cols")), Var("n"))
    weight_at = _ROW_COLS.offset(("r", "n"), cast=None)
    return Kernel(
        name="sparse_linear_values_grad",
        params=(
            Tensor("dout"),
            Tensor("weights"),
            Tensor("indices", type_name="int"),
            Tensor("offsets", type_name="int"),
            Tensor("dvalues", read_only=False),
            Scalar("long long", "cols"),
            Scalar("long long", "rows"),
            Scalar("long long", "nnz"),
            Scalar("long long", "steps"),
            Tensor("err", read_only=False, type_name="int"),
        ),
        body=grid_stride(
            Var("steps"),
            *(
                Let("const long long", "s", Var("i")),
                *_event_segment(),
                For(
                    "e",
                    Var("lo"),
                    Var("hi"),
                    Num("1"),
                    (
                        *_event_row_check(),
                        Let("float", "acc", Num("0.f")),
                        For(
                            "n",
                            Num("0"),
                            Var("cols"),
                            Num("1"),
                            (
                                Assign(
                                    Var("acc"),
                                    Bin(
                                        "+",
                                        Var("acc"),
                                        Bin(
                                            "*",
                                            Load("dout", dout_at),
                                            Load("weights", weight_at),
                                        ),
                                    ),
                                ),
                            ),
                            incr="++",
                        ),
                        Store("dvalues", Var("e"), Var("acc")),
                    ),
                    incr="++",
                ),
            ),
        ),
    )


def spike_linear() -> Kernel:
    """事件驱动的脉冲线性：``out[s, n] = Σ_e spikes[s, indices[e]] * W[indices[e], n]``。

    与 ``sparse_linear`` 的唯一差别是事件幅度直接读 ``spikes[s, r]``，而不是
    读单独的 ``values`` 缓冲。这一步省掉的正是"把脉冲值再拷一份到 capacity
    长度"的缓冲，也让反向可以把脉冲当成普通稠密矩阵来求导。

    ``e`` 仍只扫第 ``s`` 步真正发放的区间：稠密的 ``(S, K)`` 乘积从不物化。
    """
    weight_at = _ROW_COLS.offset(("r", "n"), cast=None)
    spike_at = _S_ROWS.offset(("s", "r"), cast=None)
    return Kernel(
        name="spike_linear",
        params=(
            Tensor("spikes"),
            Tensor("weights"),
            Tensor("indices", type_name="int"),
            Tensor("offsets", type_name="int"),
            Tensor("out", read_only=False),
            Scalar("long long", "cols"),
            Scalar("long long", "rows"),
            Scalar("long long", "nnz"),
            Scalar("long long", "total"),
            Tensor("err", read_only=False, type_name="int"),
        ),
        body=grid_stride(
            Var("total"),
            *(
                Let("const long long", "s", Bin("/", Var("i"), Var("cols"))),
                Let("const long long", "n", Bin("%", Var("i"), Var("cols"))),
                *_event_segment(),
                Let("float", "acc", Num("0.f")),
                For(
                    "e",
                    Var("lo"),
                    Var("hi"),
                    Num("1"),
                    (
                        *_event_row_check(),
                        Assign(
                            Var("acc"),
                            Bin(
                                "+",
                                Var("acc"),
                                Bin(
                                    "*",
                                    Load("spikes", spike_at),
                                    Load("weights", weight_at),
                                ),
                            ),
                        ),
                    ),
                    incr="++",
                ),
                Store("out", Var("i"), Var("acc")),
            ),
        ),
    )


def spike_linear_weight_grad() -> Kernel:
    """``spike_linear`` 对权重的梯度：把 ``spikes[s, r] * dout[s]`` 散射回 ``r`` 行。

    值由脉冲矩阵现场读取，因此不需要 values 缓冲；同一行被多个事件命中时
    必须原子累加，输出缓冲由后端先清零。
    """
    dst = _ROW_COLS.offset(("r", "n"), cast=None)
    spike_at = _S_ROWS.offset(("s", "r"), cast=None)
    return Kernel(
        name="spike_linear_weight_grad",
        params=(
            Tensor("dout"),
            Tensor("spikes"),
            Tensor("indices", type_name="int"),
            Tensor("offsets", type_name="int"),
            Tensor("dweights", read_only=False),
            Scalar("long long", "cols"),
            Scalar("long long", "rows"),
            Scalar("long long", "nnz"),
            Scalar("long long", "total"),
            Tensor("err", read_only=False, type_name="int"),
        ),
        body=grid_stride(
            Var("total"),
            *(
                Let("const long long", "s", Bin("/", Var("i"), Var("cols"))),
                Let("const long long", "n", Bin("%", Var("i"), Var("cols"))),
                *_event_segment(),
                Let("const float", "g", Load("dout", Var("i"))),
                For(
                    "e",
                    Var("lo"),
                    Var("hi"),
                    Num("1"),
                    (
                        *_event_row_check(),
                        AtomicAdd(
                            "dweights",
                            dst,
                            Bin("*", Load("spikes", spike_at), Var("g")),
                        ),
                    ),
                    incr="++",
                ),
            ),
        ),
    )


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


# 布局常量：生产内核里反复出现的两维行主序线性化，统一由 ``Layout`` 生成。
# ``cast=None`` 关掉首因子到 64 位的提升，生成的表达式与原先手写的
# ``a * stride + b`` 逐 token 相同（5.209 的判据是对渲染源码取哈希比对）。
_ROW_COLS = Layout(("r", "n"), (None, "cols"))
_T_ELEMS = Layout(("t", "i"), (None, "elems"))
_C_ELEMS = Layout(("c", "i"), (None, "elems"))
_S_ROWS = Layout(("s", "r"), (None, "rows"))
_T_WIDTH = Layout(("t", "k"), (None, "width"))

_IMAGE_LAYOUT = Layout(("n", "c", "h", "w"), (None, "C", "H", "W"))
_TAP_LAYOUT = Layout(("h", "w"), (None, "W"))


def _split_output_index(x_var: str, y_var: str, x_dim: str, y_dim: str
) -> tuple:
    """把展平的输出下标拆成 ``(n, c, y, x)``（按 ``(N, C, Y, X)`` 行优先）。

    生成的名字是**约定**而不是参数：中间量固定叫 ``t``、通道叫 ``c``、批叫
    ``n``，因为紧随其后的 ``_image_slice`` 就按这几个名字取。卷积权重梯度
    不能用它——那里的末轴是"输出通道 ``oi``"，语义上是纯除法、类型是
    ``int``，与本骨架的最后两行不同（见 ``conv2d_weight_grad``）。
    """
    return Layout(
        ("n", "c", y_var, x_var),
        (None, Var("C"), Var(y_dim), Var(x_dim)),
    ).unravel_steps(Var("idx"), ("n", "c", y_var, x_var))


def _image_slice(buf: str, name: str = "base", *, const: bool = True) -> Ptr:
    """``x + ((long long)n * C + c) * (long long)H * W``"""
    return Ptr(name, buf, _IMAGE_LAYOUT.offset(("n", "c")), const=const)


def _tap(buf: str = "base") -> Expr:
    """``buf[(long long)ih * W + iw]``"""
    return Index(buf, _TAP_LAYOUT.offset(("ih", "iw")))


def _window_axis(
    k: str, i: str, stride: str, pad: str, o: str, lo: Expr, hi: Expr
) -> tuple:
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


def _window_scan(
    body: tuple, *, lo_h=None, hi_h=None, lo_w=None, hi_w=None
) -> For:
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
        params=(Tensor("x"), Tensor("out", read_only=False))
        + _pool_params(adaptive=False),
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
            (
                Bin("*", Num("-1"), Var(off))
                if sign == "-"
                else Bin("+", Var(size), Var(off))
            ),
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
                        Paren(
                            Bin("*", Cast("long long", Var(y_var)), Var("H"))
                        ),
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
                        Paren(
                            Bin("*", Cast("long long", Var(x_var)), Var("W"))
                        ),
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
        params=(Tensor("x"), Tensor("out", read_only=False))
        + _pool_params(adaptive=True),
        body=_flat_loop(
            _split_output_index("ow", "oh", "OW", "OH")
            + (
                _image_slice("x"),
            )
            + _adaptive_bounds("oh", "ow")
            + (
                Let("float", "acc", Num("0.f")),
                _adaptive_window(
                    (AugAssign(Var("acc"), "+", Load("base", _tap().index)),)
                ),
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


def _conv_tap(
    b: str, ch: str, ch_dim: str, y: str, y_dim: str, x: str, x_dim: str
) -> Expr:
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
                                            Bin(
                                                "/", Var("numw"), Var("SW")
                                            ),
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
                                            Bin(
                                                "*", Var("ho"), Var("SH")
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


def adaptive_avg_pool2d_backward() -> Kernel:
    """自适应平均池化反向：把 ``dy / 窗口面积`` 散射回窗口内的单元（原子加）。"""
    return Kernel(
        name="adaptive_avg_pool2d_backward",
        params=(Tensor("dy"), Tensor("dx", read_only=False))
        + _pool_params(adaptive=True),
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


def conv2d_tiled() -> Kernel:
    """直接分块卷积（NCHW，互相关）：不物化列矩阵的那条路径。

    两级复用：块内 ``TOC`` 个输出通道的权重按 ``CB`` 个输入通道分批暂存到
    共享内存（每批一次屏障，而不是每个通道一次）；一个输入值喂 ``ROC`` 个
    累加器，把全局读次数降到 ``1/ROC``。

    落界的线程不提前退出，而是走带边界的慢路径——``__syncthreads()`` 必须
    被整个 block 到达，提前 ``return`` 会让屏障死锁。

    bias 在写回时融合：``acc + (has_bias ? b[oc] : 0)``。这样调用方不必为
    一次 ``(1, O, 1, 1)`` 的广播加再发一个逐元素核；没有 bias 时传任意有效
    指针并把 ``has_bias`` 置 0，内核不会解引用它（与 attention 的 mask 同法）。

    全程 ``float`` 计算：这个内核不做 dtype 参数化，f16 走 im2col + GEMM。
    """
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
        name="conv2d_tiled",
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
                                                        Var("W"),
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
                                                        Var("W"),
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
                                                        Index("xr", Var("iw")),
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
            Let(
                "float*",
                "dst",
                Bin(
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
                ),
            ),
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
                                Index(
                                    "dst",
                                    Bin(
                                        "*",
                                        Cast("long long", Var("j")),
                                        Bin("*", Var("HO"), Var("WO")),
                                    ),
                                ),
                                Bin(
                                    "+",
                                    Index("acc", Var("j")),
                                    Paren(
                                        Cond(
                                            Var("has_bias"),
                                            Index(
                                                "b",
                                                Bin("+", Var("oc"), Var("j")),
                                            ),
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
            Let(
                "float*",
                "dst",
                Bin(
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
                ),
            ),
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
                                Index(
                                    "dst",
                                    Bin(
                                        "*",
                                        Cast("long long", Var("j")),
                                        Bin("*", Var("HO"), Var("WO")),
                                    ),
                                ),
                                Bin(
                                    "+",
                                    Index("acc", Var("j")),
                                    Paren(
                                        Cond(
                                            Var("has_bias"),
                                            Index(
                                                "b",
                                                Bin("+", Var("oc"), Var("j")),
                                            ),
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
    to_v = lambda p: Bin(
        "+", Bin("*", Cast("long long", Paren(p)), Var("plane")), Var("i")
    )
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
    to_u = lambda p: Bin(
        "+", Bin("*", Cast("long long", Paren(p)), Var("oplane")), Var("i")
    )
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


def winograd_output_transform() -> Kernel:
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
        ),
        body=grid_stride(Var("total"), *body),
    )


def winograd_fused() -> Kernel:
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
                            Bin("+", Var("y11"), Var("bv")),
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
                    Bin("+", Var("y10"), Var("bv")),
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
                    Bin("+", Var("y01"), Var("bv")),
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
            Bin("+", Var("y00"), Var("bv")),
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
        ),
        body=body,
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
