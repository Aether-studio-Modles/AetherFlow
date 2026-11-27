"""每个算子的反向规则（VJP）。

约定的两点，写在这里免得每条规则重复解释：

1. **规则返回"已经归约到输入形状"的梯度。** 广播（``add(x, b)`` 里 b 是
   ``(D,)``）的反向必须把梯度沿被广播的轴求和回去，这件事由 ``sum_to`` 统一做，
   规则里不各写一份——否则漏一条就是"形状对不上"或更糟的"悄悄算错"。
2. **只用已有语义构图。** 反向节点全部是前向也能用的 PORTABLE 算子
   （``mul`` / ``reduce_sum`` / ``broadcast_to`` / ``sqrt`` …），因此反向图
   在任何一个能跑前向的后端上都能跑，不需要"反向专用内核"这一层。

唯一的例外是卷积与最大池化：它们的反向不是几个逐元素算子的组合能表达的
（``dx`` 要散射、``dw`` 要对 batch 归约），所以它们是**语义完整的 PORTABLE
算子**（``conv2d_input_grad`` / ``conv2d_weight_grad`` / ``maxpool2d_backward``），
有独立的参考实现，而不是"某个后端的反向内核"。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional, Sequence, Tuple

from .ops import OPS, NoGradientError
from .types import DType

RULES: Dict[str, Any] = {}


def vjp(name: str):
    def deco(fn):
        RULES[name] = fn
        return fn

    return deco


# --------------------------------------------------------------------------
# 形状工具
# --------------------------------------------------------------------------


def _prod(xs) -> int:
    n = 1
    for x in xs:
        n *= int(x)
    return n


def _static(value, what: str) -> Tuple[int, ...]:
    """取静态形状；含符号维时明确报错。

    为什么要求静态：这些规则要靠 ``reshape`` 把若干维合并/拆开，而合并符号维
    需要一套运行期形状表达式，IR 现在没有。只靠转置 + ``keepdims`` 就能完成
    的求和已经放开（见 :func:`sum_to`），其余仍在构图期拒绝。
    """
    if not value.type.is_static:
        raise NoGradientError(
            f"{what} 的反向需要静态形状，但 %{value.name} 的类型是 {value.type}。"
            "含符号维的广播/归约反向需要运行期形状表达式，第一版不支持"
        )
    return tuple(int(d) for d in value.type.shape)


def _sum_axis_keep(b, g, axis: int):
    """沿 ``axis`` 求和并保留该维（长度变 1）。

    用 ``transpose`` 把轴换到最后，是因为 CUDA 后端的 ``reduce_sum`` 只实现了
    最后一维；这样这条规则在两个后端上都成立，而不是"参考后端能跑"。

    只需要秩，不需要各维的具体数字：``reduce_sum`` 的 ``keepdims=True`` 本身
    就接受符号维，转置也只依赖下标。所以符号批维上求和的这一条路是通的。
    """
    r = g.type.rank
    if r == 1:
        return b.add_node("reduce_sum", [g], {"axis": 0, "keepdims": True})
    perm = [i for i in range(r) if i != axis] + [axis]
    gt = b.add_node("transpose", [g], {"perm": perm})
    gt = b.add_node("reduce_sum", [gt], {"axis": -1, "keepdims": True})
    inv = [0] * r
    for pos, src in enumerate(perm):
        inv[src] = pos
    return b.add_node("transpose", [gt], {"perm": inv})


def sum_axes(b, g, axes: Sequence[int]):
    """沿 ``axes`` 求和并**去掉**这些维。"""
    shape = _static(g, "归约")
    r = len(shape)
    norm = tuple(sorted({int(a) % r for a in axes}))
    if not norm:
        return g
    keep = [i for i in range(r) if i not in norm]
    perm = keep + list(norm)
    if perm != list(range(r)):
        g = b.add_node("transpose", [g], {"perm": perm})
    keep_shape = tuple(shape[i] for i in keep)
    g = b.add_node(
        "reshape", [g], {"shape": [_prod(keep_shape), _prod(shape[i] for i in norm)]}
    )
    g = b.add_node("reduce_sum", [g], {"axis": -1, "keepdims": False})
    return b.add_node("reshape", [g], {"shape": list(keep_shape)})


def sum_to(b, g, target_shape: Sequence[int]):
    """把 ``g`` 归约到 ``target_shape``（广播的反向）。

    形状相同直接返回：这一步不需要任何具体数字，符号批维也走这里。

    需要真正求和时分成两条路。全是具体数字时保留原来的"合并前导维再一次
    归约"，数值与算子顺序都不变；含符号维时前导乘积无法在构图期算出来，
    改成"前导维转置到最后 -> ``reduce_sum``（``keepdims=True``）-> reshape
    去掉最后那个 1"。reshape 的目标只有保留维（本来就已定义），不引入新符号。
    """
    gs = tuple(g.type.shape)
    ts = tuple(target_shape)
    if len(ts) > len(gs):
        raise NoGradientError(f"梯度形状 {gs} 比目标形状 {ts} 秩更小，无法反向广播")
    if gs == ts:
        return g

    lead = len(gs) - len(ts)
    # 先处理对齐部分的内部广播维（这些是"形状里有 1、被广播成 n"的轴）
    for i, (gd, td) in enumerate(zip(gs[lead:], ts)):
        if td == 1 and gd != 1:
            g = _sum_axis_keep(b, g, lead + i)

    if not lead:
        return g

    if all(isinstance(d, int) for d in gs) and all(isinstance(d, int) for d in ts):
        # 多出来的前导维整体求和：转置到 (目标维..., 前导维...)，二维化后归约。
        r = len(gs)
        perm = list(range(lead, r)) + list(range(lead))
        g = b.add_node("transpose", [g], {"perm": perm})
        knum = _prod(ts)
        lnum = _prod(gs[:lead])
        if lnum != 1:
            g = b.add_node("reshape", [g], {"shape": [knum, lnum]})
            g = b.add_node("reduce_sum", [g], {"axis": -1, "keepdims": False})
        return b.add_node("reshape", [g], {"shape": list(ts)})

    r = len(gs)
    perm = list(range(lead, r)) + list(range(lead))
    g = b.add_node("transpose", [g], {"perm": perm})
    g = b.add_node("reduce_sum", [g], {"axis": -1, "keepdims": True})
    return b.add_node("reshape", [g], {"shape": list(ts)})


def _const(b, value: float, dtype):
    return b.constant(value, dtype=dtype)


def _expand_to(b, value, shape):
    """广播到目标形状；形状已相同就不插节点。"""
    if tuple(value.type.shape) == tuple(shape):
        return value
    return b.add_node("broadcast_to", [value], {"shape": list(shape)})


def _transpose_last_two(b, value):
    r = value.type.rank
    perm = tuple(range(r - 2)) + (r - 1, r - 2)
    return b.add_node("transpose", [value], {"perm": perm})


# --------------------------------------------------------------------------
# 逐元素
# --------------------------------------------------------------------------


@vjp("add")
def _add(ctx):
    b, g = ctx.b, ctx.g
    return (
        sum_to(b, g, ctx.input(0).type.shape),
        sum_to(b, g, ctx.input(1).type.shape),
    )


@vjp("sub")
def _sub(ctx):
    b, g = ctx.b, ctx.g
    return (
        sum_to(b, g, ctx.input(0).type.shape),
        sum_to(b, b.add_node("neg", [g]), ctx.input(1).type.shape),
    )


@vjp("where")
def _where(ctx):
    """d/dx = where(cond, g, 0)，d/dy = where(cond, 0, g)。

    广播由 where 自己负责：标量零与 (1,) 常量都能直接参与，不需要先
    expand。bool 的两侧本来不可导（掩码之间选择），返回 None 让自动微分
    给出"不在可导路径上"的明确结论。
    """
    b, g = ctx.b, ctx.g
    cond = ctx.input(0)
    x, y = ctx.input(1), ctx.input(2)
    dx = dy = None
    if x.type.dtype is not DType.BOOL:
        zero = _const(b, 0.0, x.type.dtype)
        dx = sum_to(b, b.add_node("where", [cond, g, zero]), x.type.shape)
        dy = sum_to(b, b.add_node("where", [cond, zero, g]), y.type.shape)
    return (None, dx, dy)


@vjp("mul")
def _mul(ctx):
    b, g = ctx.b, ctx.g
    x, y = ctx.input(0), ctx.input(1)
    return (
        sum_to(b, b.add_node("mul", [g, y]), x.type.shape),
        sum_to(b, b.add_node("mul", [g, x]), y.type.shape),
    )


@vjp("div")
def _div(ctx):
    b, g = ctx.b, ctx.g
    x, y = ctx.input(0), ctx.input(1)
    dx = sum_to(b, b.add_node("div", [g, y]), x.type.shape)
    # d(x/y)/dy = -x/y^2
    dy = b.add_node("div", [g, b.add_node("mul", [y, y])])
    dy = b.add_node("neg", [b.add_node("mul", [x, dy])])
    return (dx, sum_to(b, dy, y.type.shape))


@vjp("neg")
def _neg(ctx):
    return (ctx.b.add_node("neg", [ctx.g]),)


@vjp("clamp")
def _clamp(ctx):
    """区间门：``dx = g * [min <= x <= max]``。

    边界点按 PyTorch 的约定**保留**（闭区间导数），与 ``relu`` 在 0 点取 0
    的约定不同——两者各自的语义就是这么定义的，不要统一。门用
    ``1 - step(lo - x)`` 构造而不是 ``step(x - lo)``：后者在 ``x == lo``
    处返回 0，会把边界点错误地排除掉。
    """
    b, g = ctx.b, ctx.g
    x = ctx.input(0)
    dt = x.type.dtype
    one = _const(b, 1.0, dt)

    def ge(value, bound):
        # x >= bound  <=>  not (bound - x > 0)
        return b.add_node(
            "sub",
            [one, b.add_node("step", [b.add_node("sub", [bound, value])], {"threshold": 0.0})],
        )

    def le(value, bound):
        # x <= bound  <=>  not (x - bound > 0)
        return b.add_node(
            "sub",
            [one, b.add_node("step", [b.add_node("sub", [value, bound])], {"threshold": 0.0})],
        )

    lo, hi = ctx.attrs.get("min"), ctx.attrs.get("max")
    mask = None
    if lo is not None:
        mask = ge(x, _const(b, float(lo), dt))
    if hi is not None:
        part = le(x, _const(b, float(hi), dt))
        mask = part if mask is None else b.add_node("mul", [mask, part])
    return (b.add_node("mul", [g, mask]),)


@vjp("relu")
def _relu(ctx):
    # 0 点导数取 0（与 PyTorch 的约定一致），所以用严格大于的 step。
    return (ctx.b.add_node("mul", [ctx.g, ctx.b.add_node("step", [ctx.input(0)])]),)


@vjp("silu")
def _silu(ctx):
    # d/dx [x*s] = s * (1 + x*(1-s))，s = sigmoid(x)
    b, g, x = ctx.b, ctx.g, ctx.input(0)
    s = b.add_node("sigmoid", [x])
    one = _const(b, 1.0, x.type.dtype)
    t = b.add_node("add", [one, b.add_node("mul", [x, b.add_node("sub", [one, s])])])
    dx = b.add_node("mul", [b.add_node("mul", [g, s]), t])
    if len(ctx.node.inputs) == 1:
        return (dx,)
    # 门控形态 y = silu(x) * gate：对 x 的梯度再乘 gate，对 gate 的梯度是 g*silu(x)。
    gate = ctx.input(1)
    return (b.add_node("mul", [dx, gate]), b.add_node("mul", [g, b.add_node("silu", [x])]))


@vjp("gelu")
def _gelu(ctx):
    """tanh 近似的导数。

    逐个因子复算（而不缓存前向中间值）是反向图的既定做法：``GradCtx`` 只暴露
    输入、输出与 attrs，规则必须能只靠这些重建局部量。代价是重复计算，
    收益是反向图自包含、可单独序列化。
    """
    b, g, x = ctx.b, ctx.g, ctx.input(0)
    dt = x.type.dtype
    c = _const(b, 0.7978845608028654, dt)
    k = _const(b, 0.044715, dt)
    one = _const(b, 1.0, dt)
    half = _const(b, 0.5, dt)

    x2 = b.add_node("mul", [x, x])
    inner = b.add_node(
        "mul",
        [c, b.add_node("add", [x, b.add_node("mul", [k, b.add_node("mul", [x2, x])])])],
    )
    t = b.add_node("tanh", [inner])

    # d/dx = 0.5(1+t) + 0.5x(1-t^2) * c(1+3k x^2)
    first = b.add_node("mul", [half, b.add_node("add", [one, t])])
    du = b.add_node(
        "mul",
        [c, b.add_node("add", [one, b.add_node("mul", [_const(b, 3.0 * 0.044715, dt), x2])])],
    )
    one_minus_t2 = b.add_node("sub", [one, b.add_node("mul", [t, t])])
    second = b.add_node(
        "mul", [half, b.add_node("mul", [x, b.add_node("mul", [one_minus_t2, du])])]
    )
    return (b.add_node("mul", [g, b.add_node("add", [first, second])]),)


@vjp("exp")
def _exp(ctx):
    return (ctx.b.add_node("mul", [ctx.g, ctx.output]),)


@vjp("log")
def _log(ctx):
    return (ctx.b.add_node("div", [ctx.g, ctx.input(0)]),)


@vjp("sqrt")
def _sqrt(ctx):
    b = ctx.b
    two = _const(b, 2.0, ctx.output.type.dtype)
    return (b.add_node("div", [ctx.g, b.add_node("mul", [two, ctx.output])]),)


@vjp("erf")
def _erf(ctx):
    """d/dx erf(x) = 2/sqrt(pi) * exp(-x^2)。"""
    b, g = ctx.b, ctx.g
    x = ctx.input(0)
    scale = _const(b, 1.1283791670955126, x.type.dtype)
    square = b.add_node("mul", [x, x])
    decay = b.add_node("exp", [b.add_node("neg", [square])])
    return (b.add_node("mul", [g, b.add_node("mul", [scale, decay])]),)


@vjp("tanh")
def _tanh(ctx):
    b, g = ctx.b, ctx.g
    one = _const(b, 1.0, ctx.output.type.dtype)
    return (
        b.add_node(
            "mul", [g, b.add_node("sub", [one, b.add_node("mul", [ctx.output, ctx.output])])]
        ),
    )


@vjp("sigmoid")
def _sigmoid(ctx):
    b, g = ctx.b, ctx.g
    one = _const(b, 1.0, ctx.output.type.dtype)
    return (
        b.add_node(
            "mul", [g, b.add_node("mul", [ctx.output, b.add_node("sub", [one, ctx.output])])]
        ),
    )


@vjp("step")
def _step(ctx):
    # 阶跃函数几乎处处导数为 0；它出现在 relu 的反向里，本身不该被继续求导。
    return (None,)


@vjp("sign")
def _sign(ctx):
    # 几乎处处导数为 0（0 点不可导）。与 step 同口径：返回 None 表示该输入
    # 不接收梯度，而不是伪造一个零梯度节点。
    return (None,)


@vjp("round")
def _round(ctx):
    # 阶梯函数，几乎处处导数为 0；PyTorch 同样把梯度定义为 0。
    return (None,)


@vjp("floor")
def _floor(ctx):
    return (None,)


@vjp("ceil")
def _ceil(ctx):
    return (None,)


@vjp("spike")
def _spike(ctx):
    """替代梯度（surrogate gradient）。

    前向是阶跃，真导数几乎处处为零——直接用会让 SNN 一点也学不到。这里用
    三角窗：峰值 ``1/alpha`` 落在阈值上，向两侧线性衰减，``|x-thr| >= alpha``
    时为 0。它是一条**定义**，不是对阶跃的近似（就像直通估计器定义了量化算子的
    梯度），所以前向数值与 ``step`` 完全一致，只改反向。

    为什么用 ``relu(d) + relu(neg(d))`` 而不是 ``sqrt(d*d)`` 表达 ``|d|``：
    后者在 ``d = 0``（恰好等于阈值）处反向是 ``g / (2*sqrt(0))``，会算出 NaN，
    而 x == thr 恰恰是 SNN 最常停留的位置。前者在 0 处取 0，不会污染梯度。
    """
    b = ctx.b
    x = ctx.input(0)
    thr = float(ctx.attrs.get("threshold", 1.0))
    alpha = float(ctx.attrs.get("alpha", 1.0))
    if alpha <= 0:
        raise ValueError(f"spike 的 alpha 必须为正，得到 {alpha}")
    dtype = x.type.dtype
    d = b.add_node("sub", [x, _const(b, thr, dtype)])
    absd = b.add_node(
        "add",
        [b.relu(d), b.relu(b.add_node("neg", [d]))],
    )
    q = b.add_node("mul", [absd, _const(b, 1.0 / alpha, dtype)])
    w = b.relu(b.add_node("sub", [_const(b, 1.0, dtype), q]))
    return (b.add_node("mul", [w, _const(b, 1.0 / alpha, dtype)]),)


@vjp("scan_linear")
def _scan_linear(ctx):
    """线性递归的反向。

    前向：``h_t = a_t * h_{t-1} + b_t``。
    伴随：``dh_t = dy_t + a_{t+1} * dh_{t+1}``，它本身也是一次递归，因此委托给
    ``scan_linear_backward`` 这个语义算子，而不是展开成 T 个节点。

    拿到 ``dh`` 之后，三个输入的梯度都是**已有算子**的组合：

        db   = dh
        da_t = dh_t * h_{t-1}      # h_{-1} = h0，靠 pad + slice 平移得到
        dh0  = a_0 * dh_0          # h_0 = a_0 * h0 + b_0，别漏掉 a_0

    平移之所以用 ``pad + slice`` 而不是引入 concat：pad 与 slice 都是已交付、
    两个后端都承接的 PORTABLE 算子，反向图因此不需要新原语。
    """
    b, g = ctx.b, ctx.g
    a = ctx.input(0)
    h = ctx.output
    h0 = ctx.input(2)
    shape = _static(a, "scan_linear")
    steps, tail = int(shape[0]), tuple(shape[1:])
    if steps < 1:
        raise NoGradientError(f"scan_linear 的时间维必须 >= 1，得到 {steps}")

    dh = b.add_node("scan_linear_backward", [a, g])

    # h_prev = [h0, h_0, h_1, ..., h_{T-2}]
    shifted = b.add_node(
        "slice",
        [b.add_node("pad", [h], {"dim": 0, "before": 1, "after": 0})],
        {"dim": 0, "start": 0, "stop": steps},
    )
    init = b.add_node(
        "pad",
        [b.add_node("reshape", [h0], {"shape": [1, *tail]})],
        {"dim": 0, "before": 0, "after": steps - 1},
    )
    h_prev = b.add_node("add", [shifted, init])

    da = b.add_node("mul", [dh, h_prev])
    db = dh
    dh0 = b.add_node(
        "reshape",
        [
            b.add_node(
                "mul",
                [
                    b.add_node("slice", [dh], {"dim": 0, "start": 0, "stop": 1}),
                    b.add_node("slice", [a], {"dim": 0, "start": 0, "stop": 1}),
                ],
            )
        ],
        {"shape": list(tail)},
    )
    return (da, db, dh0)


@vjp("cumsum")
def _cumsum(ctx):
    """含端点前缀和的反向：``dx_i = Σ_{j >= i} g_j``（后缀和）。

    没有 ``flip`` 原语，用恒等式 ``后缀和 = 总和 - 不含端点前缀和`` 拼出来：
    ``exclusive = shift(cumsum(g))``，其中 shift 是 ``pad + slice``——与
    ``scan_linear`` 反向里的平移同一个手段，不引入新原语。

    ``axis`` 的长度必须静态：pad 的补零宽度要写进输出类型。这与
    ``scan_linear`` 要求静态时间维是同一条约束，报错点也一致。
    """
    b, g = ctx.b, ctx.g
    raw = ctx.attrs.get("axis", -1)
    axis = int(raw[0] if isinstance(raw, (list, tuple)) else raw)
    shape = _static(g, "cumsum")
    axis %= len(shape)
    steps = shape[axis]
    total = b.add_node("reduce_sum", [g], {"axis": axis, "keepdims": True})
    inc = b.add_node("cumsum", [g], {"axis": axis})
    shifted = b.add_node("pad", [inc], {"dim": axis, "before": 1, "after": 0})
    excl = b.add_node(
        "slice", [shifted], {"dim": axis, "start": 0, "stop": steps}
    )
    full = b.add_node("broadcast_to", [total], {"shape": list(shape)})
    return (b.add_node("sub", [full, excl]),)


@vjp("tril")
def _tril(ctx):
    """下三角是逐元素 0/1 掩码的乘法，掩码自伴：``dx = tril(g)``。"""
    return (ctx.b.add_node("tril", [ctx.g], {"diagonal": int(ctx.attrs.get("diagonal", 0))}),)


@vjp("constant")
def _constant(ctx):
    return ()


# --------------------------------------------------------------------------
# 矩阵 / 卷积 / 池化
# --------------------------------------------------------------------------


@vjp("matmul")
def _matmul(ctx):
    b, g = ctx.b, ctx.g
    a, w = ctx.input(0), ctx.input(1)
    da = b.add_node("matmul", [g, _transpose_last_two(b, w)])
    dw = b.add_node("matmul", [_transpose_last_two(b, a), g])
    return (sum_to(b, da, a.type.shape), sum_to(b, dw, w.type.shape))


@vjp("linear_i8")
def _linear_i8(ctx):
    """weight-only int8 线性层的反向：只对激活与 bias 求导。

    量化权重与 scale 是**推理期常量**：int8 上不存在"沿量化误差求导"的
    语义（那里没有连续变量），scale 若也可学就等价于重新训练量化参数，
    属于 QAT 范畴，不在这一版。返回 ``None`` 让反向图把它当作"不接收梯度"，
    而不是造一张零张量——后者会把"权重是冻结的"藏进图里。
    """
    b, g = ctx.b, ctx.g
    x = ctx.input(0)
    weight = b.add_node("dequantize_i8", [ctx.input(1), ctx.input(2)])
    dx = b.add_node("matmul", [g, _transpose_last_two(b, weight)])
    if len(ctx.node.inputs) < 4:
        return (sum_to(b, dx, x.type.shape), None, None)
    db = sum_axes(b, g, tuple(range(g.type.rank - 1)))
    return (sum_to(b, dx, x.type.shape), None, None, db)


def _conv_attrs(attrs: dict) -> dict:
    return {k: attrs[k] for k in ("stride", "padding", "dilation", "groups") if k in attrs}


def _require_nchw_backward(op: str, attrs: dict) -> None:
    """NHWC 前向没有反向规则，这里显式拒绝而不是静默按 NCHW 算。

    反向算子（``conv2d_input_grad`` 等）的语义与内核都是 NCHW 的；若把 NHWC
    的形状原样喂进去，梯度会按错误的轴归约，结果看着有值但全错。因此宁可
    在这里失败：布局传播是**推理期**优化（DESIGN 5.246），训练图保持 NCHW。
    """
    layout = attrs.get("layout", "nchw")
    if layout != "nchw":
        raise NoGradientError(
            f"{op} 的 layout={layout!r} 没有反向规则：布局传播只覆盖前向"
            "（DESIGN 5.246），梯度定义在 NCHW 上；训练图请保持 NCHW"
        )


@vjp("conv2d")
def _conv2d(ctx):
    _require_nchw_backward("conv2d", ctx.attrs)
    b, g = ctx.b, ctx.g
    x, w = ctx.input(0), ctx.input(1)
    geo = _conv_attrs(ctx.attrs)
    # input_shape 允许带符号 batch：它是属性，执行时会随符号绑定一起解析。
    dx = b.add_node(
        "conv2d_input_grad", [g, w], {"input_shape": list(x.type.shape), **geo}
    )
    dw = b.add_node(
        "conv2d_weight_grad", [x, g], {"weight_shape": list(w.type.shape), **geo}
    )
    if len(ctx.node.inputs) < 3:
        return (dx, dw)
    db = sum_axes(b, g, (0, 2, 3))
    return (dx, dw, db)


@vjp("maxpool2d")
def _maxpool2d(ctx):
    _require_nchw_backward("maxpool2d", ctx.attrs)
    attrs = {k: ctx.attrs[k] for k in ("kernel", "stride", "padding") if k in ctx.attrs}
    return (ctx.b.add_node("maxpool2d_backward", [ctx.input(0), ctx.g], attrs),)


@vjp("avgpool2d")
def _avgpool2d(ctx):
    _require_nchw_backward("avgpool2d", ctx.attrs)
    x = ctx.input(0)
    attrs = {k: ctx.attrs[k] for k in ("kernel", "stride", "padding") if k in ctx.attrs}
    return (
        ctx.b.add_node(
            "avgpool2d_backward", [ctx.g], {"input_shape": list(x.type.shape), **attrs}
        ),
    )


@vjp("adaptive_avg_pool2d")
def _adaptive_avg_pool2d(ctx):
    x = ctx.input(0)
    attrs = {k: ctx.attrs[k] for k in ("output_size",) if k in ctx.attrs}
    return (
        ctx.b.add_node(
            "adaptive_avg_pool2d_backward",
            [ctx.g],
            {"input_shape": list(x.type.shape), **attrs},
        ),
    )


# conv / pool 的反向算子只有一阶规则。给它们留 None 会让"对反向图再求导"
# 变成静默的二阶零；这里不写 vjp（=缺规则），引擎会明确报 NoGradientError。


# --------------------------------------------------------------------------
# 归一化 / 归约
# --------------------------------------------------------------------------


@vjp("softmax")
def _softmax(ctx):
    b, g, y = ctx.b, ctx.g, ctx.output
    axis = ctx.attrs.get("axis", -1)
    s = b.add_node(
        "reduce_sum", [b.add_node("mul", [g, y])], {"axis": axis, "keepdims": True}
    )
    dx = b.add_node("mul", [y, b.add_node("sub", [g, s])])
    if len(ctx.node.inputs) > 1:
        # 第二输入是融合 pass 产生的标量缩放：d/dx 要多乘一次 scale。
        # 对 scale 本身不给梯度——它是常量属性被提升成输入的结果，
        # 训练时不应更新它（ADR-010 的融合只发生在推理图里）。
        return (b.add_node("mul", [dx, ctx.input(1)]), None)
    return (dx,)


def _sum_to_mask(b, ds, mask):
    """把加性偏置的梯度沿被广播的轴求和回 ``mask`` 的形状。

    ``mask`` 的语义是"加在分数上的偏置"，不是"结构性常量"：可学习的相对位置
    偏置就是这一路的正常用法，所以数学上必须给梯度。只有形状不可静态判定
    （``ds`` 含符号维、或需要增加维度）时才返回 None——那需要运行期形状表达式，
    与 conv 拒绝符号维是同一条理由；返回 None 而不是猜一个形状，是因为
    "这个输入不该训练"是调用者的角色判断，不该由数学算子替它决定。
    """
    shape = tuple(ds.type.shape)
    want = tuple(mask.type.shape)
    if len(shape) != len(want):
        return None
    axes = []
    for i, (d, w) in enumerate(zip(shape, want)):
        if w == d:
            continue
        if w == 1:
            axes.append(i)
            continue
        return None
    if not axes:
        return ds
    if not ds.type.is_static:
        return None
    for axis in axes:
        ds = _sum_axis_keep(b, ds, axis)
    return ds


ATTENTION_BACKWARD_CHUNK_ROWS = 128
"""反向重算时沿 query 序列维一次算多少行。

反向图是"复用已有算子重算中间量"，所以峰值不取决于总计算量，而取决于
**同时活着**的 ``(...,S,T)`` 缓冲数：整块重算时分数、概率、``dp``、``ds``
在一段区间里并存（5.117 量到 3.4 份）。按 query 行切块后，这些缓冲缩成
``(块行数, T)``，块贡献完就被 liveness 回收，峰值降到"每块一份"量级。

切块只影响乘法顺序，不改变语义：softmax 沿 key 维归一化，各行互相独立，
所以任意切分都逐位等价（同一后端上）。
"""


def _slice_rows(b, value, start: int, stop: int):
    """沿倒数第二维（序列维）切出一段行。"""
    return b.add_node("slice", [value], {"dim": -2, "start": start, "stop": stop})


def _attention_block(b, q, k, v, g, mask, scale, p, seed, dtype, kt, vt):
    """一个 query 行块上的反向重算。

    返回 ``(dq, dk, dv, dmask)``；``dmask`` 在 mask 不需要梯度时为 None。
    ``kt`` / ``vt`` 是 k、v 的转置，由调用方建一次共用——切块后若每块都
    再转置一次，同一份转置会被算 N 遍。
    """
    scores = b.add_node("matmul", [q, kt])
    scores = b.add_node("mul", [scores, _const(b, scale, dtype)])
    if mask is not None:
        scores = b.add_node("add", [scores, mask])
    probs = b.add_node("softmax", [scores], {"axis": -1})

    applied, keep = probs, None
    if p > 0.0:
        keep = b.add_node("dropout_mask", [probs], {"p": p, "seed": seed})
        applied = b.add_node("mul", [probs, keep])

    dv = b.add_node("matmul", [_transpose_last_two(b, applied), g])
    dp = b.add_node("matmul", [g, vt])
    if keep is not None:
        dp = b.add_node("mul", [dp, keep])

    # softmax 的 VJP：ds = p * (dp - sum(dp * p))
    total = b.add_node(
        "reduce_sum",
        [b.add_node("mul", [dp, probs])],
        {"axis": -1, "keepdims": True},
    )
    ds = b.add_node("mul", [probs, b.add_node("sub", [dp, total])])

    # 缩放是乘在分数上的常数，链式法则要再乘一次
    scaled = b.add_node("mul", [ds, _const(b, scale, dtype)])
    dq = b.add_node("matmul", [scaled, k])
    dk = b.add_node("matmul", [_transpose_last_two(b, scaled), q])
    dmask = _sum_to_mask(b, ds, mask) if mask is not None else None
    return dq, dk, dv, dmask


def _attention_chunk_rows(q) -> Optional[int]:
    """返回按行切块的行数；不该切块（或切不了）时返回 ``None``。

    切不了的情况只有符号序列维：``slice`` 的区间要静态整数，而运行期
    形状表达式 IR 还没有（与 conv 拒绝符号维同一条理由）。这类图退回整块
    重算，语义完全一致，只是峰值按 5.117 的口径。
    """
    shape = q.type.shape
    if not isinstance(shape[-2], int):
        return None
    if int(shape[-2]) <= ATTENTION_BACKWARD_CHUNK_ROWS:
        return None
    return ATTENTION_BACKWARD_CHUNK_ROWS


@vjp("attention")
def _attention(ctx):
    """融合注意力的反向：完全由已有 PORTABLE 算子组合出来。

    前向是 ``softmax(q k^T * scale [+ mask]) @ v``，中间量（分数、概率）在
    反向图里按同样步骤重算——这是"反向图自包含"的既定做法：代价是重复
    计算，收益是 attention 不需要反向专用内核，随便哪个能跑 ``matmul`` /
    ``softmax`` 的后端都能跑它的反向。

    链式展开四步：
    ``dv = p^T dy``、``dp = dy v^T``、``ds = p * (dp - sum(dp*p))``、
    ``dq = (ds*scale) k``、``dk = (ds*scale)^T q``。

    dropout 作用在概率上，反向要按同一 ``(p, seed)`` 重建掩码；
    ``dropout_mask`` 对输入没有梯度，被乘掉的那一份由乘法那条边负责。

    序列维够长时按行分块重算（见 ``ATTENTION_BACKWARD_CHUNK_ROWS``）：
    ``dq`` 与 mask 梯度沿行维 ``concat`` 回去，``dk`` / ``dv`` 沿块累加。
    行之间互不影响，所以分块与整块在同一后端上逐位等价。
    """
    b, g = ctx.b, ctx.g
    q, k, v = ctx.input(0), ctx.input(1), ctx.input(2)
    dtype = q.type.dtype
    scale = float(ctx.attrs.get("scale", 1.0))
    mask = ctx.input(3) if len(ctx.node.inputs) > 3 else None
    p = float(ctx.attrs.get("dropout", 0.0))
    seed = int(ctx.attrs.get("seed", 0))

    kt = _transpose_last_two(b, k)
    vt = _transpose_last_two(b, v)

    size = _attention_chunk_rows(q)
    if p > 0.0:
        # dropout 掩码按"输入的形状 + seed"重建：切块后每块的掩码是另一组
        # 采样，与整块图不再给出同一结果。"同 seed 同结果"是 5.29 的约定，
        # 不能让一次显存优化悄悄换掉它——所以带 dropout 的反向退回整块。
        size = None
    rows = int(q.type.shape[-2]) if size is not None else 0
    mask_rows = None
    if size is not None and mask is not None:
        # mask 的序列维要么等于 S（逐行不同），要么是 1（广播）：前者按行切，
        # 后者整块复用；其余形状 _sum_to_mask 本来就给不出梯度，走整块路径
        # 让那条规则自己判断。
        if mask.type.is_static and int(mask.type.shape[-2]) in (1, rows):
            mask_rows = int(mask.type.shape[-2])
        else:
            size = None

    if size is None:
        dq, dk, dv, dmask = _attention_block(
            b, q, k, v, g, mask, scale, p, seed, dtype, kt, vt
        )
    else:
        dq_parts, dmask_parts = [], []
        dk = dv = None
        for start in range(0, rows, size):
            stop = min(start + size, rows)
            m_c = None
            if mask is not None:
                m_c = mask if mask_rows == 1 else _slice_rows(b, mask, start, stop)
            dq_c, dk_c, dv_c, dm_c = _attention_block(
                b,
                _slice_rows(b, q, start, stop),
                k,
                v,
                _slice_rows(b, g, start, stop),
                m_c,
                scale,
                p,
                seed,
                dtype,
                kt,
                vt,
            )
            dq_parts.append(dq_c)
            dk = dk_c if dk is None else b.add_node("add", [dk, dk_c])
            dv = dv_c if dv is None else b.add_node("add", [dv, dv_c])
            if m_c is not None:
                dmask_parts.append(dm_c)
        if len(dq_parts) == 1:
            dq = dq_parts[0]
        else:
            dq = b.add_node("concat", dq_parts, {"axis": -2})
        if not dmask_parts:
            dmask = None
        elif any(part is None for part in dmask_parts):
            dmask = None
        elif mask_rows == rows and len(dmask_parts) > 1:
            dmask = b.add_node("concat", dmask_parts, {"axis": -2})
        else:
            # 广播 mask：每块贡献形状都是 (...,1,T)，按块相加
            dmask = dmask_parts[0]
            for part in dmask_parts[1:]:
                dmask = b.add_node("add", [dmask, part])

    # q / k / v 的批维在 attention 里不允许广播（见 _infer_attention），
    # 三个梯度的形状因此天然等于输入形状，不需要再归约。
    grads = [dq, dk, dv]
    if mask is not None:
        grads.append(dmask)
    return tuple(grads)


@vjp("layernorm")
def _layernorm(ctx):
    b, g = ctx.b, ctx.g
    x0, gamma, beta = ctx.input(0), ctx.input(1), ctx.input(2)
    has_residual = len(ctx.node.inputs) == 4
    # 融合形态 layernorm(x, gamma, beta, residual) 的归一化对象是 x + residual；
    # 反向图里重算这一次加法，规则因此仍然只依赖输入（与 gelu 的"复算"同一条约定）。
    x = b.add_node("add", [x0, ctx.input(3)]) if has_residual else x0
    raw_axes = ctx.attrs.get("axes", (-1,))
    axes = (raw_axes,) if isinstance(raw_axes, int) else tuple(raw_axes)
    eps = float(ctx.attrs.get("eps", 1e-5))

    shape = _static(x, "layernorm")
    r = len(shape)
    norm = tuple(a % r for a in axes)
    count = _prod(shape[a] for a in norm)
    inv_n = _const(b, 1.0 / count, x.type.dtype)
    one = _const(b, 1.0, x.type.dtype)

    mean = b.add_node(
        "mul", [b.add_node("reduce_sum", [x], {"axis": list(norm), "keepdims": True}), inv_n]
    )
    xc = b.add_node("sub", [x, mean])
    var = b.add_node(
        "mul",
        [
            b.add_node(
                "reduce_sum",
                [b.add_node("mul", [xc, xc])],
                {"axis": list(norm), "keepdims": True},
            ),
            inv_n,
        ],
    )
    inv_std = b.add_node(
        "div", [one, b.add_node("sqrt", [b.add_node("add", [var, _const(b, eps, x.type.dtype)])])]
    )
    xhat = b.add_node("mul", [xc, inv_std])

    # gamma / beta 的形状是"归一化轴上的长度"，所以要对**其余所有轴**求和。
    # 参数本身可能被广播（标量 gamma/beta 也合法）：最后必须归约到它的
    # 实际形状，否则梯度是 (C,) 而参数是 ()，求和/更新会直接报形状不一致。
    rest = tuple(i for i in range(r) if i not in norm)
    dbeta = sum_to(b, sum_axes(b, g, rest), beta.type.shape)
    dgamma = sum_to(b, sum_axes(b, b.add_node("mul", [g, xhat]), rest), gamma.type.shape)

    dhat = b.add_node("mul", [g, gamma])
    m1 = b.add_node(
        "mul",
        [
            b.add_node("reduce_sum", [dhat], {"axis": list(norm), "keepdims": True}),
            inv_n,
        ],
    )
    m2 = b.add_node(
        "mul",
        [
            b.add_node(
                "reduce_sum",
                [b.add_node("mul", [dhat, xhat])],
                {"axis": list(norm), "keepdims": True},
            ),
            inv_n,
        ],
    )
    dx = b.add_node(
        "mul",
        [inv_std, b.add_node("sub", [b.add_node("sub", [dhat, m1]), b.add_node("mul", [xhat, m2])])],
    )
    if has_residual:
        # y = LayerNorm(x + r)：dL/dx 与 dL/dr 相同。
        return (dx, dgamma, dbeta, dx)
    return (dx, dgamma, dbeta)


@vjp("batch_norm")
def _batch_norm(ctx):
    """BN 的反向：统计量由输入重算，不缓存前向的 mean/var。

    与 gelu / layernorm 同一条约定：规则只依赖输入与 attrs，不依赖前向的
    运行期中间值，反向图因此是自包含、可序列化的。代价是再走两遍归约。

    所有归约都经 ``sum_axes``（transpose + 二维化 + 最后一维求和），
    这样 CUDA 的 ``reduce_sum`` 只需支持最后一维就能承接整张反向图。
    含符号维时会由 ``_static`` 明确报错——BN 的反向需要知道每通道的元素数。
    """
    b, g = ctx.b, ctx.g
    x, gamma = ctx.input(0), ctx.input(1)
    shape = _static(x, "batch_norm")
    r = len(shape)
    axis = int(ctx.attrs.get("axis", 1)) % r
    eps = float(ctx.attrs.get("eps", 1e-5))
    red = tuple(i for i in range(r) if i != axis)
    count = _prod(shape[i] for i in red)
    dtype = x.type.dtype
    inv_m = _const(b, 1.0 / count, dtype)
    one = _const(b, 1.0, dtype)
    keep_shape = [1 if i in red else d for i, d in enumerate(shape)]

    def red_keep(value):
        return b.add_node("reshape", [sum_axes(b, value, red)], {"shape": keep_shape})

    mean = b.add_node("mul", [red_keep(x), inv_m])
    xc = b.add_node("sub", [x, mean])
    var = b.add_node("mul", [red_keep(b.add_node("mul", [xc, xc])), inv_m])
    inv_std = b.add_node(
        "div", [one, b.add_node("sqrt", [b.add_node("add", [var, _const(b, eps, dtype)])])]
    )
    xhat = b.add_node("mul", [xc, inv_std])

    # gamma/beta 是 (C,)，先广播到通道轴；这与 CUDA 内核只接 (C,) 的约束一致
    gshape = [1] * r
    gshape[axis] = shape[axis]
    gamma_b = b.add_node("reshape", [gamma], {"shape": gshape})

    param_shape = list(gamma.type.shape)
    dbeta = b.add_node("reshape", [sum_axes(b, g, red)], {"shape": param_shape})
    dgamma = b.add_node(
        "reshape", [sum_axes(b, b.add_node("mul", [g, xhat]), red)], {"shape": param_shape}
    )

    dxhat = b.add_node("mul", [g, gamma_b])
    sum_dxhat = red_keep(dxhat)
    sum_dxhat_xhat = red_keep(b.add_node("mul", [dxhat, xhat]))
    t = b.add_node(
        "sub",
        [
            b.add_node("sub", [b.add_node("mul", [_const(b, float(count), dtype), dxhat]), sum_dxhat]),
            b.add_node("mul", [xhat, sum_dxhat_xhat]),
        ],
    )
    dx = b.add_node("mul", [t, b.add_node("mul", [inv_std, inv_m])])
    return (dx, dgamma, dbeta)


@vjp("rmsnorm")
def _rmsnorm(ctx):
    """RMSNorm 的导数。

    记 ``s = mean(x^2)``、``r = sqrt(s + eps)``、``n`` 为归一化轴上的元素数：

        dgamma_i = sum_{其余轴} g_i * x_i / r
        dx_i     = (g_i * gamma_i) / r
                   - x_i * (sum_j g_j gamma_j x_j) / (n * r^3)

    第二项来自 ``r`` 本身依赖全部 x：``d(1/r)/dx_j = -x_j / (n * r^3)``。
    少了它梯度形状仍然正确，但数值会偏——这类错误只有数值对拍能抓到。
    """
    b, g = ctx.b, ctx.g
    x, gamma = ctx.input(0), ctx.input(1)
    raw_axes = ctx.attrs.get("axes", (-1,))
    axes = (raw_axes,) if isinstance(raw_axes, int) else tuple(raw_axes)
    eps = float(ctx.attrs.get("eps", 1e-6))

    shape = _static(x, "rmsnorm")
    r = len(shape)
    norm = tuple(a % r for a in axes)
    count = _prod(shape[a] for a in norm)
    dt = x.type.dtype
    one = _const(b, 1.0, dt)

    ms = b.add_node(
        "mul",
        [
            b.add_node(
                "reduce_sum",
                [b.add_node("mul", [x, x])],
                {"axis": list(norm), "keepdims": True},
            ),
            _const(b, 1.0 / count, dt),
        ],
    )
    inv_r = b.add_node(
        "div", [one, b.add_node("sqrt", [b.add_node("add", [ms, _const(b, eps, dt)])])]
    )

    rest = tuple(i for i in range(r) if i not in norm)
    # 同 layernorm：gamma 可以被广播成标量，梯度要归约回它的实际形状。
    dgamma = sum_to(
        b,
        sum_axes(b, b.add_node("mul", [b.add_node("mul", [g, x]), inv_r]), rest),
        gamma.type.shape,
    )

    dgx = b.add_node("mul", [g, gamma])
    dx_first = b.add_node("mul", [inv_r, dgx])
    total = b.add_node(
        "reduce_sum",
        [b.add_node("mul", [dgx, x])],
        {"axis": list(norm), "keepdims": True},
    )
    denom = b.add_node(
        "mul",
        [_const(b, float(count), dt), b.add_node("add", [ms, _const(b, eps, dt)])],
    )
    coef = b.add_node("div", [inv_r, denom])
    dx = b.add_node(
        "sub", [dx_first, b.add_node("mul", [b.add_node("mul", [x, total]), coef])]
    )
    return (dx, dgamma)


@vjp("reduce_sum")
def _reduce_sum(ctx):
    b, g, x = ctx.b, ctx.g, ctx.input(0)
    # 归约的反向只需要"把梯度广播回原形状"，逐维保持原样即可：静态维与符号维
    # 走同一条路。reshape 的目标可以含裸符号，broadcast_to 也已接受符号目标。
    shape = tuple(x.type.shape)
    raw_axes = ctx.attrs.get("axis", -1)
    axes = (raw_axes,) if isinstance(raw_axes, int) else tuple(raw_axes)
    norm = {a % len(shape) for a in axes}
    if not ctx.attrs.get("keepdims", False):
        mid = [1 if i in norm else d for i, d in enumerate(shape)]
        g = b.add_node("reshape", [g], {"shape": mid})
    return (_expand_to(b, g, shape),)


def _reduce_info(ctx):
    """归约类规则共用的 (输入形状, 归一化轴, keepdims)。形状允许含符号维。"""
    shape = tuple(ctx.input(0).type.shape)
    raw_axes = ctx.attrs.get("axis", -1)
    axes = (raw_axes,) if isinstance(raw_axes, int) else tuple(raw_axes)
    norm = tuple(sorted({a % len(shape) for a in axes}))
    return shape, norm, bool(ctx.attrs.get("keepdims", False))


def _keepdims_shape(shape, norm):
    return [1 if i in norm else d for i, d in enumerate(shape)]


@vjp("reduce_mean")
def _reduce_mean(ctx):
    b, g, x = ctx.b, ctx.g, ctx.input(0)
    shape, norm, keepdims = _reduce_info(ctx)
    mid = _keepdims_shape(shape, norm)
    if not keepdims:
        g = b.add_node("reshape", [g], {"shape": mid})

    if all(isinstance(shape[a], int) for a in norm):
        count = 1
        for a in norm:
            count *= shape[a]
        scaled = b.add_node("mul", [g, _const(b, 1.0 / count, x.type.dtype)])
    else:
        # 归约轴含符号维：计数只能在执行期算。用"把标量 1 广播成 mid 形状再
        # 沿归约轴求和"得到与梯度同形的计数张量——它只取决于形状，与输入值
        # 无关（不能用 exp(x-x) 之类的技巧：x 含 inf 时会污染计数）。
        one = _const(b, 1.0, x.type.dtype)
        ones = b.add_node("broadcast_to", [one], {"shape": list(mid)})
        count = b.add_node(
            "reduce_sum", [ones], {"axis": list(norm), "keepdims": True}
        )
        scaled = b.add_node("div", [g, count])
    return (_expand_to(b, scaled, shape),)


@vjp("reduce_max")
def _reduce_max(ctx):
    b, g, x = ctx.b, ctx.g, ctx.input(0)
    shape, norm, keepdims = _reduce_info(ctx)
    if keepdims:
        m = ctx.output
    else:
        mid = _keepdims_shape(shape, norm)
        g = b.add_node("reshape", [g], {"shape": mid})
        m = b.add_node("reshape", [ctx.output], {"shape": mid})
    # max 的梯度是"均分给所有取到最大值的元素"（PyTorch 的约定）。
    # mask 由 ``1 - step(m - x)`` 得到：step 是严格大于，所以只有差为 0
    # 的位置留下 1，既不需要阈值，也不引入近似。
    one = _const(b, 1.0, x.type.dtype)
    mask = b.add_node(
        "sub", [one, b.add_node("step", [b.add_node("sub", [m, x])])]
    )
    count = b.add_node(
        "reduce_sum", [mask], {"axis": list(norm), "keepdims": True}
    )
    dx = b.add_node("mul", [b.add_node("div", [g, count]), mask])
    return (_expand_to(b, dx, shape),)


# --------------------------------------------------------------------------
# 形状 / 类型 / 索引
# --------------------------------------------------------------------------


@vjp("reshape")
def _reshape(ctx):
    target = list(ctx.input(0).type.shape)
    return (ctx.b.add_node("reshape", [ctx.g], {"shape": target}),)


@vjp("transpose")
def _transpose(ctx):
    perm = tuple(int(p) for p in ctx.attrs["perm"])
    inv = [0] * len(perm)
    for pos, src in enumerate(perm):
        inv[src] = pos
    return (ctx.b.add_node("transpose", [ctx.g], {"perm": inv}),)


@vjp("broadcast_to")
def _broadcast_to(ctx):
    return (sum_to(ctx.b, ctx.g, ctx.input(0).type.shape),)


@vjp("cast")
def _cast(ctx):
    target = ctx.input(0).type.dtype
    if ctx.g.type.dtype == target:
        # 上游梯度已经是输入的类型：这一步 cast 是恒等。建成节点只会在
        # 反向图里多一次 kernel launch（混精里成片出现），直接透传。
        return (ctx.g,)
    return (ctx.b.add_node("cast", [ctx.g], {"dtype": target.value}),)


@vjp("slice")
def _slice(ctx):
    b = ctx.b
    x = ctx.input(0)
    shape = _static(x, "slice")
    dim = int(ctx.attrs["dim"]) % len(shape)
    start = int(ctx.attrs.get("start", 0))
    stop = ctx.attrs.get("stop")
    stop = shape[dim] if stop is None else int(stop)
    return (
        b.add_node(
            "pad",
            [ctx.g],
            {"dim": dim, "before": start, "after": shape[dim] - stop},
        ),
    )


@vjp("pad")
def _pad(ctx):
    b = ctx.b
    g = ctx.g
    shape = _static(g, "pad")
    dim = int(ctx.attrs["dim"]) % len(shape)
    before = int(ctx.attrs.get("before", 0))
    after = int(ctx.attrs.get("after", 0))
    stop = shape[dim] - after
    return (
        b.add_node("slice", [g], {"dim": dim, "start": before, "stop": stop}),
    )


@vjp("concat")
def _concat(ctx):
    """按各段长度切片回去。

    不需要新的"split"算子：concat 的每一段长度在构图期已知（infer 强制拼接维
    静态），``slice`` 恰好是它的逆运算，两个方向都已经是 PORTABLE 语义。
    """
    b = ctx.b
    shape = _static(ctx.output, "concat")
    ax = int(ctx.attrs.get("axis", 0)) % len(shape)
    grads = []
    start = 0
    for i in range(len(ctx.node.inputs)):
        size = int(ctx.input(i).type.shape[ax])
        grads.append(
            b.add_node("slice", [ctx.g], {"dim": ax, "start": start, "stop": start + size})
        )
        start += size
    return tuple(grads)


@vjp("embedding")
def _embedding(ctx):
    table = ctx.input(0)
    shape = _static(table, "embedding")
    return (
        ctx.b.add_node(
            "embedding_backward", [ctx.g, ctx.input(1)], {"table_shape": list(shape)}
        ),
        None,
    )


@vjp("gather")
def _gather(ctx):
    """``gather`` 只对 x 求导：下标是整型，梯度沿被选中的位置散射回去。

    ``x_shape`` 必须静态：反向节点的属性要把每一维写死（与 embedding 的
    ``table_shape`` 同一条理由），含符号维时直接报"没有定义梯度"，而不是
    在运行期拿绑定后的形状去拼一个与图不符的属性。
    """
    shape = _static(ctx.input(0), "gather")
    dim = int(ctx.attrs.get("dim", -1))
    if dim < 0:
        dim += len(shape)
    return (
        ctx.b.add_node(
            "gather_backward",
            [ctx.g, ctx.input(1)],
            {"x_shape": list(shape), "dim": dim},
        ),
        None,
    )


@vjp("cross_entropy")
def _cross_entropy(ctx):
    """交叉熵对 logits 的梯度是 ``(softmax - onehot) * dloss``。

    专门的反向算子而不是"softmax + 散射 + 逐元素"拼出来的图：拼法要么
    物化一份与 logits 同大的 softmax，要么走一次 gather_backward 的散射。
    融合实现两次扫过 logits，中间量只有每行的 max / sum。
    """
    return (
        ctx.b.add_node(
            "cross_entropy_backward", [ctx.input(0), ctx.input(1), ctx.g]
        ),
        None,
    )


@vjp("sparse_linear")
def _sparse_linear(ctx):
    """事件驱动稀疏线性：``weights`` 与 ``values`` 都有梯度。

    前向对 ``values`` 是线性的（``out[s] = Σ_e values[e] * W[idx[e]]``），所以

        dvalues[e] = dout[s(e), :] · W[idx[e], :]

    每个事件只出现一次，这条梯度是**逐事件独立**的，不需要 ``add.at`` 式的
    重复下标累加。它与权重梯度的方向正好相反：权重是"把事件散射回行"（会撞
    行、要原子加），这里是"把每行收回来"（不撞事件、直接写）。

    为什么不做成"散射进稠密 ``(S, K)`` 的 dspikes"：那需要 ``(S, K)`` 的缓冲与
    原子加，正好把前向省下来的稠密脉冲矩阵又在反向造回来。事件形态的梯度
    长度就是 ``E``，交给 ``spike`` 之类的上游算子按事件处理即可。

    ``indices`` / ``offsets`` 是整数索引，没有梯度。``rows`` 只能由属性带过来：
    反向图里没有 ``weights`` 这个输入，而散射内核必须知道行数才能检查越界。
    """
    shape = _static(ctx.input(0), "sparse_linear")
    dvalues = ctx.b.add_node(
        "sparse_linear_values_grad",
        [ctx.g, ctx.input(0), ctx.input(1), ctx.input(3)],
    )
    return (
        ctx.b.add_node(
            "sparse_linear_weight_grad",
            [ctx.g, ctx.input(1), ctx.input(2), ctx.input(3)],
            {"rows": int(shape[0])},
        ),
        None,
        dvalues,
        None,
    )


@vjp("spike_linear")
def _spike_linear(ctx):
    """融合的事件驱动脉冲线性：前向稀疏，反向对激活**不稀疏**。

    这是刻意的，也是它能与稠密 ``build_snn_network`` 对拍的原因：脉冲的
    ``surrogate`` 梯度在未发放位置同样非零（只要膜电位落在窗内），若为了
    "稀疏"把梯度只散回非零事件，那些位置的梯度就永久丢失——初始层一次都不
    发放时甚至完全学不动。因此 ``dspikes`` 走与 ``matmul`` 完全相同的构图，
    只有权重梯度保持事件散射（``spike_linear_weight_grad``）。
    """
    b = ctx.b
    spikes, weights, indices, offsets = (ctx.input(i) for i in range(4))
    dspikes = b.add_node("matmul", [ctx.g, _transpose_last_two(b, weights)])
    dweights = b.add_node(
        "spike_linear_weight_grad", [ctx.g, spikes, indices, offsets]
    )
    return (
        sum_to(b, dspikes, spikes.type.shape),
        dweights,
        None,
        None,
    )


@vjp("dropout_mask")
def _dropout_mask(ctx):
    """掩码只由 ``(seed, 扁平索引)`` 决定，与 x 的数值无关，所以 x 没有梯度。

    运行期 key 输入同理：它只参与随机源，不是被微分的量。

    返回 None 而不是零张量：``mul(x, mask)`` 里 x 的梯度已经由乘法那条边
    完整给出，这里再造一个零张量只会让反向图多一次分配，也让"DCE 之后
    什么都不剩"这种情形看不出来。
    """
    return (None,) * len(ctx.node.inputs)


@vjp("topk_mask")
def _topk_mask(ctx):
    """掩码是"选第几个"的离散结果，与输入数值之间没有可用的导数。

    返回 None 而不是零张量：``mul(prob, mask)`` 里 prob 的梯度已经由乘法
    那条边完整给出，这里再造零只会让反向图多一次分配（与 ``dropout_mask``
    同一条理由）。掩码本身有消费者时仍会得到它的梯度，不受这里影响。
    """
    return (None,) * len(ctx.node.inputs)


@vjp("topk_values")
def _topk_values(ctx):
    """``topk_values`` 对 x 求导：把 dy 散射回被选中的位置。

    下标是离散比较的结果，不能对 x 求导，但"哪个位置被选中"本身是常量，
    所以 values 这条路径上的梯度是良定义的——与 ``gather`` 同一个形状：
    ``gather_backward`` 用重复下标累加。反向重算一遍 ``topk_indices`` 而不是
    在前向保存下标，代价是排序算两次，换来的是反向图不依赖前向中间量。

    ``x`` 的形状必须静态（``x_shape`` 要写进反向节点的属性），与 ``gather``
    的 ``x_shape`` 同一条理由；含符号维时明确报"没有定义梯度"。
    """
    shape = _static(ctx.input(0), "topk_values")
    axis = int(ctx.attrs.get("axis", -1))
    if axis < 0:
        axis += len(shape)
    k = int(ctx.attrs["k"])
    idx = ctx.b.add_node("topk_indices", [ctx.input(0)], {"k": k, "axis": axis})
    return (
        ctx.b.add_node(
            "gather_backward", [ctx.g, idx], {"x_shape": list(shape), "dim": axis}
        ),
    )


@vjp("topk_indices")
def _topk_indices(ctx):
    """下标是离散结果，不可导；输出 i32，本来也不会进浮点反传。"""
    return (None,) * len(ctx.node.inputs)


# --------------------------------------------------------------------------
# 安装
# --------------------------------------------------------------------------


def install() -> None:
    """把规则挂到 ``OPS`` 上。

    ``Op`` 是 frozen dataclass，所以用 ``replace`` 生成新实例换进表里。
    没有反向规则的算子保持 ``vjp=None``：``backward`` 遇到它会明确报错，
    而不是把"忘了写"当成"梯度为零"。
    """
    for name, fn in RULES.items():
        op = OPS[name]
        OPS[name] = replace(op, vjp=fn)


install()
