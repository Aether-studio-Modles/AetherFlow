"""脉冲神经网络（SNN）的模型级 IR 构件。

SNN 与普通前馈网络只差两件事，而这两件都不需要新的 IR 机制：

1. **有时间步。** 静态 IR 里没有循环，所以把时间**展开进图**：T 步就是 T 份
   子图，膜电位从上一步显式传到下一步。展开后的图与普通图完全一样，因此
   verifier、JSON 序列化、CUDA Graph 捕获、任意后端都照常适用。
2. **神经元是"积分—发放—重置"。** 发放是阶跃，真导数处处为零；反向用
   ``spike`` 算子的替代梯度（三角窗，见 5.18）。这是 SNN 能训练的全部秘密，
   而它落在一条 ``vjp`` 规则里，不在引擎里。

这一层刻意只做三件事：一步 LIF、T 步展开、脉冲序列的平均发放率。分类头、
损失、优化器都用普通图与 ``backward`` 组合，不另起一套。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..ir.graph import Graph, Value
from ..ir.types import f32

__all__ = [
    "LIFSpec",
    "lif_step",
    "build_snn",
    "mean_rate",
    "SNNSpec",
    "constant_current_encode",
    "build_snn_network",
    "build_snn_events",
]


@dataclass(frozen=True)
class LIFSpec:
    """漏积分—发放（LIF）神经元的离散参数。

    ``decay``
        膜电位泄漏系数（``v <- decay * v + x``）。必须落在 ``[0, 1)``：
        1.0 表示完全不泄漏，电位会无界累积。
    ``threshold``
        发放阈值。前向只比较这个值，具体取多少不影响可微性。
    ``alpha``
        替代梯度的窗宽。峰值梯度 ``1/alpha`` 落在阈值上；窗外的梯度为 0。
        调大更平滑但更"糊"，调小更接近真实阶跃但更容易梯度消失。
    ``reset``
        ``"subtract"``：发放后减去阈值（最常见的硬重置）；
        ``"none"``：不重置，电位继续累积。
    """

    decay: float = 0.9
    threshold: float = 1.0
    alpha: float = 1.0
    reset: str = "subtract"

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay < 1.0:
            raise ValueError(f"decay 必须落在 [0, 1)，得到 {self.decay}")
        if self.threshold <= 0.0:
            raise ValueError(f"threshold 必须为正，得到 {self.threshold}")
        if self.alpha <= 0.0:
            raise ValueError(f"alpha 必须为正，得到 {self.alpha}")
        if self.reset not in ("subtract", "none"):
            raise ValueError(f"reset 只能是 'subtract' 或 'none'，得到 {self.reset!r}")


def lif_step(g: Graph, x: Value, v: Value, spec: LIFSpec = LIFSpec()) -> Tuple[Value, Value]:
    """一步 LIF。返回 ``(脉冲 s, 新膜电位 v_next)``。

    方程（与 spikingjelly / Lava 的离散形式一致）：

        v_pre  = decay * v + x
        s      = spike(v_pre - 0, threshold, alpha)      # 即 v_pre > threshold
        v_next = v_pre - threshold * s                   # reset == "subtract"

    注意 ``s`` 用的是 ``spike`` 而不是 ``step``：两者前向逐位相同，但 ``spike``
    带替代梯度，``step`` 明确不带（它出现在 relu 的反向里，不该被再求导）。
    """
    if x.type != v.type:
        raise TypeError(
            f"LIF 的输入与膜电位必须同类型，得到 {x.type} 与 {v.type}；"
            "请先在图上 cast 到同一 dtype"
        )
    dtype = x.type.dtype

    v_pre = g.add(g.mul(v, g.constant(spec.decay, dtype=dtype)), x)
    s = g.add_node(
        "spike",
        [v_pre],
        {"threshold": spec.threshold, "alpha": spec.alpha},
    )
    if spec.reset == "none":
        return s, v_pre

    v_next = g.sub(v_pre, g.mul(s, g.constant(spec.threshold, dtype=dtype)))
    return s, v_next


def build_snn(
    g: Graph,
    x: Value,
    spec: LIFSpec = LIFSpec(),
    init: Optional[Value] = None,
) -> Tuple[List[Value], Value]:
    """把 ``x`` 的 T 个时间步展开进图，返回 ``(每步脉冲, 末态膜电位)``。

    ``x`` 形状 ``(T, ...)``，**T 必须是静态维**：展开发生在构图期，图里要有
    确定数量的节点，符号维在这里没有意义（与 conv / 广播反向拒绝符号维同理）。
    """
    shape = tuple(x.type.shape)
    if not shape:
        raise ValueError("SNN 输入至少要有时间维，得到 0 维张量")
    if not x.type.is_static:
        raise ValueError(
            f"SNN 第一版要求整个输入形状静态，得到 {x.type}。"
            "原因不是时间维本身：初始膜电位是一个 zero 常量节点，"
            "常量的形状写在属性里，无法随符号维在运行期变化。"
        )
    steps = int(shape[0])
    step_shape = shape[1:]

    v = init if init is not None else g.constant(0.0, dtype=x.type.dtype, shape=step_shape)
    if v.type.dtype != x.type.dtype or tuple(v.type.shape) != step_shape:
        raise ValueError(
            f"初始膜电位应为 {x.type.dtype.value}{step_shape}，得到 {v.type}"
        )

    spikes: List[Value] = []
    for t in range(steps):
        xt = g.slice(x, dim=0, start=t, stop=t + 1)
        xt = g.reshape(xt, shape=step_shape)
        s, v = lif_step(g, xt, v, spec)
        spikes.append(s)
    return spikes, v


def mean_rate(g: Graph, spikes: List[Value]) -> Value:
    """脉冲序列的平均发放率：``sum(spikes) / T``。

    这是最常用的读出头——把 (T, ...) 的稀疏脉冲压成一个 (...)，再接到普通
    分类头上。不用 ``reduce_sum`` 沿时间轴归约，是因为 CUDA 的 ``reduce_sum``
    只支持最后一维，而时间维在最前面；逐元素相加在两个后端上都成立。
    """
    if not spikes:
        raise ValueError("empty spike train")
    acc = spikes[0]
    for s in spikes[1:]:
        acc = g.add(acc, s)
    dtype = spikes[0].type.dtype
    return g.mul(acc, g.constant(1.0 / len(spikes), dtype=dtype))


# --------------------------------------------------------------------------
# 多层 SNN
# --------------------------------------------------------------------------


def constant_current_encode(g: Graph, x: Value, steps: int) -> List[Value]:
    """把静态输入复制成 ``steps`` 个时间步的恒定电流。

    这不是"复制节点"：每个时间步直接引用同一个 ``Value``，图里只多出 T 条
    边而不多出计算。脉冲神经网络真正的编码器（泊松/延迟）需要随机数或
    时间比较，那些属于数据侧，不该在模型构件里假装成确定性的图。
    """
    if steps < 1:
        raise ValueError(f"steps 必须为正，得到 {steps}")
    return [x] * steps


@dataclass(frozen=True)
class SNNSpec:
    """多层全连接 SNN。

    每层是 ``linear + LIF``，时间维展开进图；读出用最后一层的平均发放率，
    可选接一个线性分类头。``batch`` 与 ``steps`` 必须静态（膜电位初值是
    零常量节点，展开发生在构图期）。
    """

    input_dim: int = 16
    hidden: Tuple[int, ...] = (32, 32)
    steps: int = 8
    batch: int = 4
    num_classes: int = 0
    lif: LIFSpec = LIFSpec()

    def __post_init__(self) -> None:
        if self.input_dim < 1 or self.steps < 1 or self.batch < 1:
            raise ValueError("input_dim / steps / batch 都必须为正")
        if not self.hidden:
            raise ValueError("hidden 至少需要一层")
        if any(int(h) < 1 for h in self.hidden):
            raise ValueError(f"各层宽度必须为正，得到 {self.hidden}")
        if self.num_classes < 0:
            raise ValueError(f"num_classes 不能为负，得到 {self.num_classes}")

    @property
    def param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        shapes: Dict[str, Tuple[int, ...]] = {}
        fan_in = self.input_dim
        for i, width in enumerate(self.hidden):
            shapes[f"l{i}_w"] = (fan_in, int(width))
            shapes[f"l{i}_b"] = (int(width),)
            fan_in = int(width)
        if self.num_classes:
            shapes["head_w"] = (fan_in, self.num_classes)
            shapes["head_b"] = (self.num_classes,)
        return shapes


def _lif_layer(
    g: Graph,
    seq: Sequence[Value],
    w: Value,
    b: Value,
    lif: LIFSpec,
    batch: int,
    out_dim: int,
) -> List[Value]:
    dtype = seq[0].type.dtype
    v = g.constant(0.0, dtype=dtype, shape=(batch, out_dim))
    spikes: List[Value] = []
    for x in seq:
        pre = g.add(g.matmul(x, w), b)
        s, v = lif_step(g, pre, v, lif)
        spikes.append(s)
    return spikes


def _event_layer(
    g: Graph,
    seq: Sequence[Value],
    w: Value,
    b: Value,
    lif: LIFSpec,
    batch: int,
    width: int,
    out_dim: int,
) -> List[Value]:
    """一层事件驱动的 ``linear + LIF``。

    与 ``_lif_layer`` 的唯一差别是矩阵乘：这里先把 T 步脉冲压成 CSR，再用
    ``spike_linear`` 逐事件累加。权重、偏置、膜电位与读出完全同构，因此
    两条路径前向逐位相同、反向也对拍到 1e-6。

    压缩拆成 ``spike_offsets`` 与 ``spike_csr`` 两个单输出节点，而不是打包成
    一个 i32 张量再切片：IR 没有多输出节点，而 GPU 后端的 ``slice`` 不支持
    int32（不做隐式降级）。拆开后下游直接拿到起点表与列号，零拷贝。

    容量取 ``T×batch×width``：这是非零元素数的上界，静态图必须一次分配。
    实际发放远少于它时，CSR 只写前 ``nnz`` 个位置，多余容量不占计算。

    ``spike_linear`` 的值直接从脉冲矩阵按列号读，不物化 ``values`` 缓冲；
    反向对脉冲走普通 ``matmul``（见 ``grad_rules._spike_linear``），未发放
    位置的 surrogate 梯度因此不会丢。
    """
    dtype = seq[0].type.dtype
    steps = len(seq)
    capacity = steps * batch * width
    flat = g.reshape(g.concat(*seq, axis=0), shape=(steps * batch, width))
    offsets = g.add_node("spike_offsets", [flat])
    indices = g.add_node("spike_csr", [flat, offsets], {"capacity": capacity})
    pre_all = g.add_node("spike_linear", [flat, w, indices, offsets])
    pre_all = g.reshape(pre_all, shape=(steps, batch, out_dim))

    v = g.constant(0.0, dtype=dtype, shape=(batch, out_dim))
    spikes: List[Value] = []
    for t in range(steps):
        pre = g.reshape(
            g.slice(pre_all, dim=0, start=t, stop=t + 1),
            shape=(batch, out_dim),
        )
        s, v = lif_step(g, g.add(pre, b), v, lif)
        spikes.append(s)
    return spikes


def build_snn_events(
    g: Graph, spec: SNNSpec | None = None
) -> Tuple[Dict[str, Value], Value]:
    """事件驱动的多层 SNN，输入/参数/输出与 ``build_snn_network`` 完全一致。

    第 0 层的输入是连续电流（还没发放过，压不出稀疏），仍走稠密 ``matmul``；
    之后每一层都先把上一层 T 步脉冲压成 CSR，再交给 ``spike_linear``。这样
    稀疏性真正作用在计算上：一层只读它实际发放的事件，而不是 ``K`` 个神经元。

    可训练：``spike_linear`` 的反向对权重走事件散射、对脉冲走普通 ``matmul``，
    因此梯度与 ``build_snn_network`` 一致（前向逐位相同，反向差在浮点累加
    顺序的 1e-6 量级）。未发放位置的 surrogate 梯度不会被丢掉——那正是
    "初始层一次都不发放"时仍能学起来的前提。
    """
    spec = spec or SNNSpec()
    inputs: Dict[str, Value] = {
        "x": g.input("x", f32[spec.steps, spec.batch, spec.input_dim])
    }
    for name, shape in spec.param_shapes.items():
        inputs[name] = g.input(name, f32[shape])

    seq: Sequence[Value] = [
        g.reshape(
            g.slice(inputs["x"], dim=0, start=t, stop=t + 1),
            shape=(spec.batch, spec.input_dim),
        )
        for t in range(spec.steps)
    ]

    for i, width in enumerate(spec.hidden):
        width = int(width)
        if i == 0:
            seq = _lif_layer(
                g, seq, inputs["l0_w"], inputs["l0_b"], spec.lif, spec.batch, width
            )
            continue
        seq = _event_layer(
            g,
            seq,
            inputs[f"l{i}_w"],
            inputs[f"l{i}_b"],
            spec.lif,
            spec.batch,
            int(spec.hidden[i - 1]),
            width,
        )

    rate = mean_rate(g, list(seq))
    if spec.num_classes:
        return inputs, g.add(g.matmul(rate, inputs["head_w"]), inputs["head_b"])
    return inputs, rate


def build_snn_network(
    g: Graph, spec: SNNSpec | None = None
) -> Tuple[Dict[str, Value], Value]:
    """构建多层 SNN，返回 ``(输入表, 输出 Value)``。

    输入 ``x`` 形状 ``(steps, batch, input_dim)``；输出是最后一层的平均发放率
    ``(batch, hidden[-1])``，若 ``num_classes > 0`` 则再接线性头给出 logits。
    """
    spec = spec or SNNSpec()
    inputs: Dict[str, Value] = {
        "x": g.input("x", f32[spec.steps, spec.batch, spec.input_dim])
    }
    for name, shape in spec.param_shapes.items():
        inputs[name] = g.input(name, f32[shape])

    seq: Sequence[Value] = [
        g.reshape(
            g.slice(inputs["x"], dim=0, start=t, stop=t + 1),
            shape=(spec.batch, spec.input_dim),
        )
        for t in range(spec.steps)
    ]

    for i, width in enumerate(spec.hidden):
        seq = _lif_layer(g, seq, inputs[f"l{i}_w"], inputs[f"l{i}_b"], spec.lif,
                         spec.batch, int(width))

    rate = mean_rate(g, list(seq))
    if spec.num_classes:
        return inputs, g.add(g.matmul(rate, inputs["head_w"]), inputs["head_b"])
    return inputs, rate
