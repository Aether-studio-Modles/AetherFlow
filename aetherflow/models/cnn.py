"""CNN 构件：卷积块、残差块、一个小型分类网络。

三个和 Transformer 不同的地方，都值得写下来：

1. **空间维必须是静态的。** 池化与卷积的输出尺寸要写进类型里，而 IR 没有
   "尺寸表达式"。所以 ``image`` 是静态整数，batch 反而可以符号化——
   批量变化不影响任何形状推导。
2. **全局平均池化用 ``avgpool2d`` 表达。** kernel 等于最终特征图尺寸，
   输出 ``(B, C, 1, 1)``，再 reshape 成 ``(B, C)`` 接分类头。这样就不需要
   "global_avg_pool" 这种专用算子。
3. **残差块的 1x1 投影只在 stride != 1 时出现。** 通道数变化本来就需要
   投影，把它显式写在参数表里，好过藏在"自动分支"里。
4. **BN 与 dropout 是构图期开关，不是运行期 state。** ``batch_norm=True``
   往图里插 ``batch_norm`` 算子（训练态）；``dropout>0`` 往图里插
   seed 化的 ``dropout_mask``。两者关掉时图里**没有**对应节点，因此
   推理图就是同一份代码、不同参数构建出来的另一张图——不需要"模式切换"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..ir.graph import Graph, Value
from ..ir.types import f32

__all__ = ["CNNSpec", "conv_block", "bn_conv_block", "bn_infer", "residual_block", "build_cnn"]


def conv_block(g: Graph, x: Value, w: Value, b: Value, stride: int = 1) -> Value:
    """``relu(conv2d(x, w, b, stride, padding=1))``，即 3x3 "same" 卷积块。"""
    y = g.add_node("conv2d", [x, w, b], {"stride": (stride, stride), "padding": (1, 1)})
    return g.relu(y)


def bn_conv_block(
    g: Graph,
    x: Value,
    w: Value,
    b: Value,
    gamma: Value,
    beta: Value,
    stride: int = 1,
    eps: float = 1e-5,
) -> Value:
    """``relu(batch_norm(conv2d(x)))``：卷积 → 训练态 BN → 激活。

    BN 放在 conv 与 relu 之间：``batch_norm`` 的语义就是"沿通道轴以外的所有轴
    统计"，所以 NCHW 布局下不需要任何 reshape。
    """
    y = g.add_node("conv2d", [x, w, b], {"stride": (stride, stride), "padding": (1, 1)})
    y = g.batch_norm(y, gamma, beta, axis=1, eps=eps)
    return g.relu(y)


def bn_infer(
    g: Graph,
    x: Value,
    mean: Value,
    var: Value,
    gamma: Value,
    beta: Value,
    eps: float = 1e-5,
) -> Value:
    """推理态 BN：用 running stats 做一次仿射，**不需要新算子**。

    训练图的 ``batch_norm`` 统计的是当前 batch；部署时统计量已经固定，
    于是 ``(x - mean) / sqrt(var + eps) * gamma + beta`` 就是全部语义。
    把 running stats 作为图输入（而不是 attrs）意味着换一组统计量不需要重构图。
    """
    if x.type.rank < 2:
        raise ValueError(f"BN 需要秩 >= 2，得到 {x.type.rank}")
    shape = [1] * x.type.rank
    shape[1] = x.type.shape[1]
    bshape = tuple(shape)
    m = g.reshape(mean, shape=bshape)
    v = g.reshape(var, shape=bshape)
    std = g.sqrt(g.add(v, g.constant(eps)))
    return g.add(g.mul(g.div(g.sub(x, m), std), g.reshape(gamma, shape=bshape)),
                 g.reshape(beta, shape=bshape))


def residual_block(
    g: Graph,
    x: Value,
    w1: Value,
    b1: Value,
    w2: Value,
    b2: Value,
    stride: int = 1,
    shortcut: Optional[Tuple[Value, Value]] = None,
) -> Value:
    """标准 ResNet 基本块：``relu(shortcut + conv(relu(conv(x))))``。

    ``stride != 1``（或通道数变化）时，必须有 ``shortcut=(w, b)`` 的 1x1
    卷积把残差投影到同形状；缺它就在构图期报错，而不是让 ``add`` 在运行期
    因为形状不匹配而失败——后者会把"配置错误"伪装成"后端问题"。
    """
    y = g.relu(g.add_node("conv2d", [x, w1, b1], {"stride": (stride, stride), "padding": (1, 1)}))
    y = g.add_node("conv2d", [y, w2, b2], {"padding": (1, 1)})

    if shortcut is None:
        if stride != 1:
            raise ValueError(f"stride={stride} 的残差块必须提供 1x1 投影 shortcut")
        if x.type.shape[1] != y.type.shape[1]:
            raise ValueError(
                f"残差两侧通道数不同（{x.type.shape[1]} 与 {y.type.shape[1]}），必须提供 shortcut"
            )
        skip = x
    else:
        sw, sb = shortcut
        skip = g.add_node("conv2d", [x, sw, sb], {"stride": (stride, stride), "padding": (0, 0)})

    return g.relu(g.add(skip, y))


@dataclass(frozen=True)
class CNNSpec:
    """小型卷积分类网络。

    结构：``stem`` → 若干 ``conv+relu``（相邻宽度之间各跟一次 2x2 最大池化）
    → 全局平均池化 → 线性分类头。``widths`` 的首项是 stem 输出通道。

    ``batch_norm=True`` 时每个卷积块变成 ``conv → BN → relu``，参数表相应多出
    ``*_bn_g`` / ``*_bn_b``。``dropout>0`` 时在分类头之前插一个 seed 化的
    ``dropout_mask``；``dropout=0``（默认）不插节点，图与原来逐字相同。
    """

    in_channels: int = 3
    image: int = 32
    num_classes: int = 10
    widths: Tuple[int, ...] = (16, 32, 64)
    batch: str = "B"
    eps: float = 1e-5
    batch_norm: bool = False
    dropout: float = 0.0
    dropout_seed: int = 0

    def __post_init__(self) -> None:
        if self.in_channels < 1 or self.image < 1 or self.num_classes < 1:
            raise ValueError("in_channels / image / num_classes 都必须为正")
        if len(self.widths) < 1:
            raise ValueError("widths 至少需要一个通道数")
        if any(int(w) < 1 for w in self.widths):
            raise ValueError(f"通道数必须为正，得到 {self.widths}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout 必须在 [0, 1) 内，得到 {self.dropout}")
        stages = len(self.widths) - 1
        if self.image % (2 ** stages) != 0:
            raise ValueError(
                f"image={self.image} 不能被 2^{stages} 整除；每级池化减半，空间维必须是静态整数"
            )

    @property
    def pooled_image(self) -> int:
        return self.image // (2 ** (len(self.widths) - 1))

    @property
    def param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        first = int(self.widths[0])
        shapes: Dict[str, Tuple[int, ...]] = {
            "stem_w": (first, self.in_channels, 3, 3),
            "stem_b": (first,),
        }
        if self.batch_norm:
            shapes["stem_bn_g"] = (first,)
            shapes["stem_bn_b"] = (first,)
        for i in range(len(self.widths) - 1):
            shapes[f"stage{i}_w"] = (int(self.widths[i + 1]), int(self.widths[i]), 3, 3)
            shapes[f"stage{i}_b"] = (int(self.widths[i + 1]),)
            if self.batch_norm:
                shapes[f"stage{i}_bn_g"] = (int(self.widths[i + 1]),)
                shapes[f"stage{i}_bn_b"] = (int(self.widths[i + 1]),)
        shapes["head_w"] = (int(self.widths[-1]), self.num_classes)
        shapes["head_b"] = (self.num_classes,)
        return shapes


def build_cnn(
    g: Graph, spec: CNNSpec | None = None
) -> Tuple[Dict[str, Value], Value]:
    """构建卷积分类网络，返回 ``(输入表, logits)``，logits 形状 ``(B, num_classes)``。"""
    spec = spec or CNNSpec()
    p: Dict[str, Value] = {
        "x": g.input("x", f32[spec.batch, spec.in_channels, spec.image, spec.image])
    }
    for name, shape in spec.param_shapes.items():
        p[name] = g.input(name, f32[shape])

    def conv(h: Value, w: Value, b: Value, stride: int, tag: str) -> Value:
        if spec.batch_norm:
            return bn_conv_block(
                g, h, w, b, p[tag + "_bn_g"], p[tag + "_bn_b"], stride=stride, eps=spec.eps
            )
        return conv_block(g, h, w, b, stride=stride)

    h = conv(p["x"], p["stem_w"], p["stem_b"], 1, "stem")
    for i in range(len(spec.widths) - 1):
        h = conv(h, p[f"stage{i}_w"], p[f"stage{i}_b"], 1, f"stage{i}")
        h = g.maxpool2d(h, kernel=(2, 2), stride=(2, 2))

    # 全局平均池化：kernel 等于最终特征图尺寸，输出 (B, C, 1, 1)
    side = spec.pooled_image
    h = g.avgpool2d(h, kernel=(side, side), stride=(1, 1))
    flat = g.reshape(h, shape=(spec.batch, int(spec.widths[-1])))
    if spec.dropout > 0.0:
        flat = g.mul(flat, g.dropout_mask(flat, p=spec.dropout, seed=spec.dropout_seed))
    return p, g.add(g.matmul(flat, p["head_w"]), p["head_b"])
