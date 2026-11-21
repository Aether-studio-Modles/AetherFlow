"""语义级 L2：逐元素算子的表达式树、融合链与 4 通道向量化。

语法级 L2（``kernel.py`` / ``kernels.py``）一条语句对一条语句地描述"一个
算子一个内核"：它可检视、可双路打印，但每个算子都要落一次中间张量，每次
访存也只是 32 位。这一层往上走一步，回答两个问题：

- **一串同形的逐元素算子能不能只写一次内存？** 能，把它们描述成一棵
  表达式树，中间值不声明、不物化，只有链首的输入与链尾的输出真正读写。
- **一个线程能不能一次算四个元素？** 能，前提是长度能被 4 整除、每个指针
  16 字节对齐、dtype 是 f32。满足时每链首输入做一次 ``af_ld4_f`` 得到
  ``float4``，四条通道各算一遍**逐字相同**的标量表达式，最后 ``af_st4_f``
  一次写回。

关键的一点是：向量化不改变算术。四条通道的表达式只有成员名（``.x`` /
``.y`` / ``z`` / ``w``）不同，每个元素的浮点运算顺序与标量路径完全一致，
所以"向量化版"与"标量版"可以互相作为对照，而不是两种精度。

这一层仍然不认识 Graph：``ElemStep`` 里的名字只是局部名字，图里的哪条边
对应哪个名字由后端决定（见 ``backends/cuda/backend.py`` 的 ``_invoke_chain``）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

from ..ir.types import DType
from .kernel import (
    Bin,
    Expr,
    Kernel,
    Let,
    Load,
    Member,
    Narrow,
    Num,
    Paren,
    Scalar,
    Store,
    Tensor,
    Var,
    VecLoad,
    VecStore,
    Widen,
)
from .kernels import BINARY_OP, UNARY_EXPR, grid_stride

__all__ = [
    "ElemInput", "ElemStep", "CHAIN_OPS", "apply_op", "build_chain",
    "channel_broadcast",
]

# 可以进链的算子：恰好是 L2 已经有一元/二元表达式表的那些。step / spike
# 不在里面——它们多一个标量参数（阈值），融进来会让"参数就是张量"这条
# 简单规则出现例外。
CHAIN_OPS = frozenset(BINARY_OP) | frozenset(UNARY_EXPR)

_LANES = ("x", "y", "z", "w")

# float4 存取的两个设备函数。放在 ``Kernel.helpers`` 里而不是方言 prelude 里：
# 只有向量化的链才会用到它们，塞进 prelude 会让每个逐元素内核（包括第三方
# 方言的对照源码）都多出两段没人调用的函数。NOTE: the C source below goes
# through NVRTC/hipRTC -- keep the comments inside it ASCII.
_VEC4_HELPERS = r"""
// One 128-bit access per four elements. The host only picks this path when
// every pointer is 16-byte aligned and the element count is a multiple of 4;
// a misaligned float4 dereference would be an illegal address, not a slow path.
__device__ __forceinline__ float4 af_ld4_f(const float* p, long long i) {
    return *reinterpret_cast<const float4*>(p + i);
}

__device__ __forceinline__ void af_st4_f(float* p, long long i,
                                         float x, float y, float z, float w) {
    float4 v;
    v.x = x; v.y = y; v.z = z; v.w = w;
    *reinterpret_cast<float4*>(p + i) = v;
}
"""


@dataclass(frozen=True)
class ElemInput:
    """链首的一个张量输入。

    ``scalar`` 表示它是"实际只有 1 个元素、在所有位置复用"的那一侧（例如
    ``x * scale`` 里的 scale）：整个内核只读一次 ``[0]``，四条通道共用。

    ``channel`` 是 ``(C, H*W)`` 两个整数，表示这个输入是逐通道广播
    （``(1,C,1,1)`` 或 ``(C,1,1)``）：第 ``i`` 个展平输出元素取
    ``(i / (H*W)) % C`` 号元素。它是 BN 的常态，也是"同形或单元素"之外
    唯一还值得单独开口子的形状——其余广播一律不融。
    """

    name: str
    scalar: bool = False
    channel: Tuple[int, int] = ()


def channel_broadcast(
    shape: Sequence[int], out_shape: Sequence[int]
) -> Optional[Tuple[int, int]]:
    """逐通道广播的参数 ``(C, H*W)``；不是这种形状就返回 ``None``。

    只认 NCHW 上"只有通道维非 1"的形状，且允许省略前导维：
    ``(1,C,1,1)`` 与 ``(C,1,1)`` 都返回同一个 ``(C, H*W)``。``(B,C,1,1)``
    这类随 batch 变化的掩码不在此列，仍然不融。
    """
    if len(out_shape) != 4 or len(shape) not in (3, 4):
        return None
    aligned = (1,) * (4 - len(shape)) + tuple(shape)
    _, c, h, w = (int(d) for d in out_shape)
    if c <= 0 or h <= 0 or w <= 0:
        return None
    if aligned[1] != c or aligned[0] != 1 or aligned[2] != 1 or aligned[3] != 1:
        return None
    if c == 1:
        return None  # 单通道等价于标量，交给 scalar 路径
    return (c, h * w)


@dataclass(frozen=True)
class ElemStep:
    """链里的一步：``op(args) -> out``。

    ``out`` 在标量路径里是一个真实的局部变量；在向量路径里它只是一棵
    表达式的名字，不会单独声明——四条通道各自把整棵树内联一遍。
    """

    out: str
    op: str
    args: Tuple[str, ...]


def _atom(e: Expr) -> Expr:
    """操作数在复合表达式里需要括号，原子表达式不需要。

    括号是显式的（``Paren`` 是节点），所以"哪里需要括号"由这里决定，而不是
    交给打印器推断优先级。少一层括号让单算子链的源码更干净，多一层则保证
    ``a + b`` 作为乘法的左操作数时不会改变结合方式。
    """
    return e if isinstance(e, (Var, Num, Member)) else Paren(e)


def apply_op(op: str, args: Sequence[Expr]) -> Expr:
    """把一个逐元素算子作用在给定的操作数上，返回表达式。

    与 ``kernels.ew_binary`` / ``ew_unary`` 共用同一张表：新加一个逐元素
    算子时只需要在 ``kernels.py`` 里写一次表达式，融合与向量化自动获得。
    """
    if op in BINARY_OP:
        if len(args) != 2:
            raise ValueError(f"{op} 需要 2 个操作数，收到 {len(args)}")
        return Bin(BINARY_OP[op], _atom(args[0]), _atom(args[1]))
    try:
        build = UNARY_EXPR[op]
    except KeyError:
        raise ValueError(
            f"未知的逐元素算子 {op!r}；已支持: {sorted(CHAIN_OPS)}"
        ) from None
    if len(args) != 1:
        raise ValueError(f"{op} 需要 1 个操作数，收到 {len(args)}")
    return build(_atom(args[0]))


def build_chain(
    steps: Iterable[ElemStep],
    inputs: Iterable[ElemInput],
    *,
    vec: int = 1,
    dtype: DType = DType.F32,
    kernel_name: str = "ew_chain",
    out: str = "out",
) -> Kernel:
    """把一条逐元素链编成一个 L2 内核。

    ``vec=1``：每个线程一个元素，每一步物化成一个 ``float`` 局部量。
    ``vec=4``：每个线程四个元素，只有链首输入与链尾输出做向量访存，中间
    步骤内联进四条通道的标量表达式。两者除访存宽度外没有差别。

    ``vec=4`` 只在 f32 上可用：f16 的存取靠 PTX 的 16 位转换，攒成向量需要
    另一套写法（见 ``cstyle.py`` 的 ``_vec_load_fn``）。这里直接拒绝，而不是
    悄悄退回标量——调用方需要知道自己拿到的是哪条路径。
    """
    steps = tuple(steps)
    inputs = tuple(inputs)
    if not steps:
        raise ValueError("逐元素链至少要有一步")
    if vec not in (1, 4):
        raise ValueError(f"vec 只支持 1 或 4，收到 {vec}")
    if vec == 4 and dtype is not DType.F32:
        raise ValueError(f"向量化目前只支持 f32，收到 {dtype.value}")

    leaves: Dict[str, ElemInput] = {}
    for inp in inputs:
        if inp.name in leaves:
            raise ValueError(f"输入 {inp.name!r} 重复")
        leaves[inp.name] = inp

    defs: Dict[str, ElemStep] = {}
    for step in steps:
        if step.out in leaves or step.out in defs:
            raise ValueError(f"值 {step.out!r} 被重复定义")
        if step.op not in CHAIN_OPS:
            raise ValueError(f"算子 {step.op!r} 不能进逐元素链")
        for a in step.args:
            if a not in leaves and a not in defs:
                raise ValueError(f"步 {step.out!r} 引用了未定义的值 {a!r}")
        defs[step.out] = step

    last = steps[-1].out
    # f16 的链每一步都必须先舍入回存储类型，再进下一步：不融合的图里每个
    # 算子都是一次真实的 f16 往返，融合若一路在 float 里算，结果会与参考
    # 实现不同（而且差得不是一点）。f32 的往返是恒等映射，不插这一层。
    round_steps = dtype is DType.F16 and vec == 1

    def expr_for(name: str, lane: str) -> Expr:
        """按通道展开一个值的表达式（向量路径：整棵树内联）。"""
        if name in leaves:
            inp = leaves[name]
            if inp.channel:
                # 四个连续元素可能跨通道边界，不能把逐通道输入一次读成
                # float4；按 lane 单独取索引，正确性优先，带宽由 L2 兜住。
                if not lane:
                    return Load(inp.name, Num("0"))
                at = Bin("+", index, Num(str(_LANES.index(lane))))
                channel, plane = inp.channel
                # ``at`` 是复合表达式：不包一层括号的话，打印出来会是
                # ``i * 4 + 1 / plane``，lane 的偏移被吞进除法里。
                return Load(
                    inp.name,
                    Bin(
                        "%",
                        Bin("/", _atom(at), Num(str(plane))),
                        Num(str(channel)),
                    ),
                )
            local = Var(f"v_{inp.name}")
            if inp.scalar or not lane:
                return local
            return Member(local, lane)
        step = defs[name]
        return apply_op(step.op, tuple(expr_for(a, lane) for a in step.args))

    def consume(name: str) -> Expr:
        """标量路径里"读一个已定义的值"：中间步在 f16 下要先读成 float。"""
        e: Expr = Var(f"v_{name}")
        return Widen(e) if round_steps and name in defs else e

    params = [Tensor(inp.name) for inp in inputs]
    params.append(Tensor(out, read_only=False))
    params.append(Scalar("long long", "n"))

    # 向量路径里每个线程负责 4 个连续元素，循环上界与索引都按"向量"计。
    # 16 字节对齐的前提是元素索引是 4 的倍数，所以 ``i * 4`` 不是可选项。
    limit: Expr = Var("n") if vec == 1 else Bin("/", Var("n"), Num("4"))
    index: Expr = Var("i") if vec == 1 else Bin("*", Var("i"), Num("4"))

    body: list = []
    for inp in inputs:
        local = f"v_{inp.name}"
        if inp.scalar:
            body.append(Let("float", local, Load(inp.name, Num("0"))))
        elif inp.channel:
            # 向量路径不在这里加载：四个 lane 的通道号可能不同，见 expr_for。
            if vec == 1:
                channel, plane = inp.channel
                at = Bin("%", Bin("/", Var("i"), Num(str(plane))), Num(str(channel)))
                body.append(Let("float", local, Load(inp.name, at)))
        elif vec == 1:
            body.append(Let("float", local, Load(inp.name, Var("i"))))
        else:
            body.append(Let("float4", local, VecLoad(inp.name, index, vec)))

    if vec == 1:
        for step in steps:
            value = apply_op(step.op, tuple(consume(a) for a in step.args))
            local = f"v_{step.out}"
            if round_steps and step is not steps[-1]:
                # type_name=None 表示用方言的存储类型（f16 下是 unsigned short）
                body.append(Let(None, local, Narrow(value)))
            else:
                body.append(Let("float", local, value))
        body.append(Store(out, Var("i"), Var(f"v_{last}")))
    else:
        body.append(
            VecStore(
                out,
                index,
                tuple(expr_for(last, lane) for lane in _LANES),
                vec,
            )
        )

    return Kernel(
        name=kernel_name,
        params=tuple(params),
        body=grid_stride(limit, *body),
        helpers=_VEC4_HELPERS if vec == 4 else "",
    )
