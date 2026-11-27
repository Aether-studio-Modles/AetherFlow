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
from .common import BINARY_OP, COMPARE_OP, grid_stride


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
    if kind not in table:
        label = "比较" if compare else "二元"
        raise ValueError(f"未知的{label}运算 {kind!r}；已支持: {sorted(table)}")

    # 标量一侧仍然收设备指针、在内核里取 [0]：传主机 float 会强制同步，
    # 也会让整张图无法被捕获。
    a_index = Num("0") if scalar_a else Var("i")
    b_index = Num("0") if scalar_b else Var("i")
    # 括号显式写进 L2（模板原本是 "(A) op (B)"），打印器不推断优先级。
    value = Bin(
        table[kind],
        Paren(Load("a", a_index)),
        Paren(Load("b", b_index)),
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


def ew_broadcast(kind: str, *, pow2: bool = False, compare: bool = False) -> Kernel:
    """支持 numpy 多维广播的逐元素二元内核。

    广播靠"对齐后的 stride"：被广播的维度 stride 为 0，同一个输入元素会被
    多个输出位置复用。维度上限 4，超过时由后端拒绝。

    ``pow2`` 为真表示**所有维都是 2 的幂**（主机侧判定），于是坐标还原可以
    用掩码 + 右移代替 64 位 div/mod：``rem % shape`` → ``rem & (shape-1)``、
    ``rem / shape`` → ``rem >> log2(shape)``。``shape`` 是运行期参数、编译器
    不知道它是不是 2 的幂，所以这个信息只能由主机侧多传一组 ``s0..s3`` 给
    内核；两者都按"取模"语义成立，数值语义不变。
    """
    table = COMPARE_OP if compare else BINARY_OP
    if kind not in table:
        label = "比较" if compare else "二元"
        raise ValueError(f"未知的{label}运算 {kind!r}；已支持: {sorted(table)}")

    value = Bin(
        table[kind],
        Paren(Load("a", Var("oa"))),
        Paren(Load("b", Var("ob"))),
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
    if pow2:
        # 尾部未用的维由主机传 dim=1 / shift=0：掩码为 0、右移 0 位，
        # 正好把 c 还原成 0 且 rem 不动，不需要在内核里分支。
        params.extend(Scalar("int", v.name) for v in _dims("s"))

    if pow2:
        decomp = (
            Let(
                "long long",
                "c",
                Bin(
                    "&",
                    Var("rem"),
                    Paren(Bin("-", Index("shape", Var("d")), Num("1"))),
                ),
            ),
            AugAssign(Var("rem"), ">>", Index("sh", Var("d"))),
        )
    else:
        decomp = (
            Let("long long", "c", Bin("%", Var("rem"), Index("shape", Var("d")))),
            AugAssign(Var("rem"), "/", Index("shape", Var("d"))),
        )

    decls = [
        ArrayDecl("long long", "shape", 4, _dims("d")),
        ArrayDecl("long long", "sa", 4, _dims("a")),
        ArrayDecl("long long", "sb", 4, _dims("b")),
    ]
    if pow2:
        decls.append(ArrayDecl("int", "sh", 4, _dims("s")))

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
