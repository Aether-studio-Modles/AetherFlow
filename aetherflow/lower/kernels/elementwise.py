"""逐元素内核：二元、一元、类型转换、阶跃、截断与广播。"""

from __future__ import annotations

from typing import Optional
from ..kernel import (
    ArrayDecl,
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
    LetMulti,
    Load,
    Neg,
    Num,
    Paren,
    Scalar,
    Store,
    Tensor,
    Var,
)
from .common import BINARY_CALL, BINARY_OP, COMPARE_OP, grid_stride


def _binary_value(kind: str, a, b, *, compare: bool = False):
    """二元表达式的统一入口：比较 / 中缀算术 / 函数调用三种形态。

    ``a`` / ``b`` 是已经取好值的操作数表达式。中缀形态显式加 ``Paren``
    （打印器不推断优先级），函数调用的实参本来就是自定界的，不加括号。
    """
    if compare:
        return Bin(COMPARE_OP[kind], Paren(a), Paren(b))
    if kind in BINARY_OP:
        return Bin(BINARY_OP[kind], Paren(a), Paren(b))
    if kind in BINARY_CALL:
        return BINARY_CALL[kind](a, b)
    label = "比较" if compare else "二元"
    table = COMPARE_OP if compare else BINARY_OP
    known = sorted(set(table) | set(BINARY_CALL))
    raise ValueError(f"未知的{label}运算 {kind!r}；已支持: {known}")


def ew_binary(
    kind: str,
    *,
    scalar_a: bool = False,
    scalar_b: bool = False,
    compare: bool = False,
) -> Kernel:
    """逐元素二元内核（网格步进）。

    ``compare`` 为真时是**比较**内核：算子从 ``COMPARE_OP`` 取，输出容器是
    1 字节的 bool。算术与比较共用同一份循环骨架，差异只有这两处。
    """
    table = COMPARE_OP if compare else BINARY_OP
    if kind not in table and kind not in BINARY_CALL:
        label = "比较" if compare else "二元"
        raise ValueError(f"未知的{label}运算 {kind!r}；已支持: {sorted(table)}")

    # 标量一侧仍然收设备指针、在内核里取 [0]：传主机 float 会强制同步，
    # 也会让整张图无法被捕获。
    a_index = Num("0") if scalar_a else Var("i")
    b_index = Num("0") if scalar_b else Var("i")
    value = _binary_value(
        kind, Load("a", a_index), Load("b", b_index), compare=compare
    )

    out = Tensor("out", read_only=False, type_name="bool") if compare else Tensor(
        "out", read_only=False
    )
    return Kernel(
        name="ew_cmp" if compare else "ew_bin",
        params=(
            Tensor("a"),
            Tensor("b"),
            out,
            Scalar("long long", "n"),
        ),
        body=grid_stride(
            Var("n"),
            # 比较结果是 C 的 0/1，直接裸写进 1 字节容器；套 ``af_st_q``
            # 只会多一次"非零归一"，而那一步在 0/1 上是恒等的。
            Store("out", Var("i"), value, raw=compare),
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
    "relu": lambda x: Cond(Bin(">", x, Num("0.f")), x, Num("0.f")),
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
    # 三角族：RoPE 的 sin / cos 表由这两个算子按位置生成。
    "sin": lambda x: Call("sinf", (x,)),
    "cos": lambda x: Call("cosf", (x,)),
    "neg": lambda x: Neg(x),
    "tanh": lambda x: Call("tanhf", (x,)),
    "sigmoid": lambda x: Bin(
        "/",
        Num("1.f"),
        Paren(Bin("+", Num("1.f"), Call("expf", (Neg(x),)))),
    ),
    "erf": lambda x: Call("erff", (x,)),
    # 符号：三路选择。用嵌套 Cond 而不是 Copysign/Sign 函数，是因为语义
    # 在 0 与 NaN 上都要精确：0 → 0，NaN 两个比较都为假 → 原样透传。
    "sign": lambda x: Cond(
        Bin(">", x, Num("0.f")),
        Num("1.f"),
        Cond(Bin("<", x, Num("0.f")), Num("-1.f"), x),
    ),
    # nearbyintf 默认按当前舍入模式（round-half-to-even），与 np.round /
    # torch.round 一致；roundf 是 half-away-from-zero，会差一。
    "round": lambda x: Call("nearbyintf", (x,)),
    "floor": lambda x: Call("floorf", (x,)),
    "ceil": lambda x: Call("ceilf", (x,)),
    # 逻辑非：只用于 bool 输入（``af_ld_q`` 取出的是 0/1 的 float），
    # 非零写 0、零写 1。写回经 ``af_st_q`` 归一，与参考端的 np.logical_not
    # 在"任意非零即真"的口径上一致。
    "lnot": lambda x: Cond(Bin("!=", x, Num("0.f")), Num("0.f"), Num("1.f")),
}


def ew_unary(kind: str) -> Kernel:
    """逐元素一元内核：先把输入读成 float，算完再写回存储类型。"""
    try:
        build = UNARY_EXPR[kind]
    except KeyError:
        raise ValueError(f"未知的一元运算 {kind!r}；已支持: {sorted(UNARY_EXPR)}") from None

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
            ),
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


def ew_where_same(*, raw_store: bool = False) -> Kernel:
    """``where(cond, a, b)`` 同形扁平内核。

    ``cond`` 是 1 字节 bool，裸索引即可（不需要 load helper）。输出跟随
    a / b 的存储类型：浮点时走 ``af_st_*`` 舍入，bool 时（两侧都是掩码）
    ``raw_store`` 为真，直接写 0/1。
    """
    return Kernel(
        name="ew_where",
        params=(
            Tensor("cond", type_name="bool"),
            Tensor("a"),
            Tensor("b"),
            Tensor("out", read_only=False),
            Scalar("long long", "n"),
        ),
        body=grid_stride(
            Var("n"),
            Store(
                "out",
                Var("i"),
                Cond(
                    Index("cond", Var("i")),
                    Load("a", Var("i")),
                    Load("b", Var("i")),
                ),
                raw=raw_store,
            ),
        ),
    )


def ew_where_bcast(*, raw_store: bool = False) -> Kernel:
    """``where(cond, a, b)`` 的三输入广播内核。

    骨架与 ``ew_broadcast`` 相同，只是多一组 ``oc`` 坐标：cond 被广播的
    维度 stride 为 0，因此同一个条件可以覆盖多个输出位置。
    """
    value = Cond(
        Index("cond", Var("oc")),
        Load("a", Var("oa")),
        Load("b", Var("ob")),
    )

    def _dims(prefix: str) -> tuple:
        return tuple(Var(f"{prefix}{k}") for k in range(4))

    params = [
        Tensor("cond", type_name="bool"),
        Tensor("a"),
        Tensor("b"),
        Tensor("out", read_only=False),
        Scalar("long long", "n"),
        Scalar("int", "ndim"),
    ]
    for prefix in ("d", "c", "a", "b"):
        params.extend(Scalar("long long", v.name) for v in _dims(prefix))

    decomp = (
        Let("long long", "r", Bin("%", Var("rem"), Index("shape", Var("d")))),
        AugAssign(Var("rem"), "/", Index("shape", Var("d"))),
    )

    decls = [
        ArrayDecl("long long", "shape", 4, _dims("d")),
        ArrayDecl("long long", "sc", 4, _dims("c")),
        ArrayDecl("long long", "sa", 4, _dims("a")),
        ArrayDecl("long long", "sb", 4, _dims("b")),
    ]

    return Kernel(
        name="ew_where_bcast",
        params=tuple(params),
        body=(
            *decls,
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
                            ("oc", Num("0")),
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
                            *decomp,
                            AugAssign(
                                Var("oc"), "+", Bin("*", Var("r"), Index("sc", Var("d")))
                            ),
                            AugAssign(
                                Var("oa"), "+", Bin("*", Var("r"), Index("sa", Var("d")))
                            ),
                            AugAssign(
                                Var("ob"), "+", Bin("*", Var("r"), Index("sb", Var("d")))
                            ),
                        ),
                        cmp=">=",
                        incr="--",
                        type_name="int",
                    ),
                    Store("out", Var("i"), value, raw=raw_store),
                ),
            ),
        ),
    )


def ew_broadcast(kind: str, *, compare: bool = False) -> Kernel:
    """支持 numpy 多维广播的逐元素二元内核。

    广播靠"对齐后的 stride"：被广播的维度 stride 为 0，同一个输入元素会被
    多个输出位置复用。维度上限 4，超过时由后端拒绝。

    坐标还原用 64 位 ``%`` / ``/``。曾经尝试过"全维是 2 的幂时改用掩码 +
    右移"的形态，实测在 V100 上反而慢 14~15%（多传的 ``sh`` 局部数组被
    动态下标寻址，落到 local memory），去掉数组改用 ``__clz`` 只追平，
    所以那条路径已经删除，见 DESIGN 5.300。
    """
    table = COMPARE_OP if compare else BINARY_OP
    if kind not in table and kind not in BINARY_CALL:
        label = "比较" if compare else "二元"
        raise ValueError(f"未知的{label}运算 {kind!r}；已支持: {sorted(table)}")

    value = _binary_value(
        kind, Load("a", Var("oa")), Load("b", Var("ob")), compare=compare
    )

    def _dims(prefix: str) -> tuple:
        return tuple(Var(f"{prefix}{k}") for k in range(4))

    out = Tensor("out", read_only=False, type_name="bool") if compare else Tensor(
        "out", read_only=False
    )
    params = [Tensor("a"), Tensor("b"), out]
    params.append(Scalar("long long", "n"))
    params.append(Scalar("int", "ndim"))
    for prefix in ("d", "a", "b"):
        params.extend(Scalar("long long", v.name) for v in _dims(prefix))

    decomp = (
        Let("long long", "c", Bin("%", Var("rem"), Index("shape", Var("d")))),
        AugAssign(Var("rem"), "/", Index("shape", Var("d"))),
    )

    decls = [
        ArrayDecl("long long", "shape", 4, _dims("d")),
        ArrayDecl("long long", "sa", 4, _dims("a")),
        ArrayDecl("long long", "sb", 4, _dims("b")),
    ]

    return Kernel(
        name="ew_cmp_bcast" if compare else "ew_bcast",
        params=tuple(params),
        body=(
            *decls,
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
                            *decomp,
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
                    Store("out", Var("i"), value, raw=compare),
                ),
            ),
        ),
    )
