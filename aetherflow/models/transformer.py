"""Transformer Block 的 IR 表达。

这不是一个"模型库"，而是一个检验器：用它来回答
"当前的语义 IR 能不能自然、无泄漏地表达一个真实模型"。

刻意保留了以下结构，因为它们最容易暴露 IR 设计缺陷：
- 多头拆分依赖 reshape + transpose 的组合，考验形状系统的符号维处理
- 注意力分数需要 broadcast 缩放，考验广播规则
- 残差需要 fanout（同一个 Value 被多个节点消费）
- FFN 使用 SiLU，考验逐元素算子的可组合性

文件里有两个层次，刻意分开：

``BlockSpec`` / ``build_transformer_block``
    单个 pre-norm 块，序列长度是符号维。用于检验形状系统本身。
``TransformerSpec`` / ``build_transformer``
    一个可训练的 decoder-only 语言模型：词嵌入 + 可学习位置编码 + 因果掩码
    + 多层堆叠 + 最终归一化 + LM head。序列长度是**静态**的，因为位置编码
    与因果掩码的内容随长度变化，图里必须能写死一个确定长度。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..ir.graph import Graph, Value
from ..ir.types import f32, i32

__all__ = [
    "BlockSpec",
    "build_transformer_block",
    "TransformerSpec",
    "causal_mask_bias",
    "build_transformer",
]


@dataclass(frozen=True)
class BlockSpec:
    d_model: int = 32
    num_heads: int = 4
    d_ff: int = 64
    eps: float = 1e-5
    batch: str = "B"
    seq: str = "S"
    dropout: float = 0.0
    dropout_seed: int = 0
    fused_attention: bool = False
    # 非空时建一个 i32[1] 的图输入当 dropout 的运行时 key：训练循环每步
    # 往里写新值，掩码才真的逐步采样。为空则沿用 dropout_seed 常量。
    rng_input: Optional[str] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout 必须在 [0, 1) 内，得到 {self.dropout}")
        if self.rng_input and self.fused_attention and self.dropout > 0.0:
            raise ValueError(
                "fused_attention 的 dropout 随机源写在算子 attrs 里（常量 seed），"
                "无法跟随运行时 key；请关掉 fused_attention，或把 dropout 设为 0"
            )

    @property
    def head_dim(self) -> int:
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model={self.d_model} 不能被 num_heads={self.num_heads} 整除")
        return self.d_model // self.num_heads

    @property
    def param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        d, dff = self.d_model, self.d_ff
        shapes: Dict[str, Tuple[int, ...]] = {
            "ln1_g": (d,),
            "ln1_b": (d,),
            "wq": (d, d),
            "wk": (d, d),
            "wv": (d, d),
            "wo": (d, d),
            "ln2_g": (d,),
            "ln2_b": (d,),
            "w1": (d, dff),
            "w2": (dff, d),
        }
        if not self.fused_attention:
            # fused 路径的缩放是 attention 的 attrs（构图期固定 1/sqrt(d_head)），
            # 不再是可训练参数；留在表里只会多一个永远没有梯度的死输入。
            shapes["scale"] = (1,)
        return shapes


def _dropout(
    g: Graph,
    x: Value,
    p: float,
    seed: int,
    key: Optional[Value] = None,
    offset: int = 0,
) -> Value:
    """``p=0`` 时**不插节点**：推理图因此与训练图是两张各自干净的图。

    dropout 的随机源有两种写法：``seed``（写进 attrs，图自包含、可复现）
    或 ``key``（运行时 i32 输入）。前者适合推理/对拍与"同图同结果"的确定性
    场景；后者是训练用的——每步换 key 才真的重新采样，而且捕获成 CUDA Graph
    之后仍然有效（seed 走 attrs 时会被捕获按值冻结）。

    ``offset`` 让同一张图里的多个站点共用一个 key 输入却拿到不同掩码。
    """
    if p <= 0.0:
        return x
    if key is not None:
        return g.mul(x, g.dropout_mask(x, key, p=p, offset=offset))
    return g.mul(x, g.dropout_mask(x, p=p, seed=seed))


def build_transformer_block(
    g: Graph, spec: BlockSpec | None = None
) -> Tuple[Dict[str, Value], Value]:
    """在 g 上构建一个 pre-norm Transformer Block，返回 (输入表, 输出 Value)。"""
    spec = spec or BlockSpec()
    d, h, dh = spec.d_model, spec.num_heads, spec.head_dim

    inputs: Dict[str, Value] = {
        "x": g.input("x", f32[spec.batch, spec.seq, d]),
    }
    for name, shape in spec.param_shapes.items():
        inputs[name] = g.input(name, f32[shape])
    key = None
    if spec.rng_input and spec.dropout > 0.0:
        # 运行时 key：训练循环每步写一个新值，掩码才逐步采样
        key = inputs[spec.rng_input] = g.input(spec.rng_input, i32[1])

    x = inputs["x"]

    # 注意力
    h1 = g.layernorm(x, inputs["ln1_g"], inputs["ln1_b"], axes=(-1,), eps=spec.eps)
    q = g.matmul(h1, inputs["wq"])
    k = g.matmul(h1, inputs["wk"])
    v = g.matmul(h1, inputs["wv"])

    head_shape = (spec.batch, spec.seq, h, dh)
    qh = g.transpose(g.reshape(q, shape=head_shape), perm=(0, 2, 1, 3))
    kh = g.transpose(g.reshape(k, shape=head_shape), perm=(0, 2, 1, 3))
    vh = g.transpose(g.reshape(v, shape=head_shape), perm=(0, 2, 1, 3))

    if spec.fused_attention:
        # 一个 attention 节点代替"缩放 + softmax + dropout + 第二个 matmul"四步；
        # 语义完全相同，但中间分数矩阵不再进入图，反向也只有一条规则。
        ctx = g.attention(
            qh,
            kh,
            vh,
            scale=1.0 / math.sqrt(dh),
            dropout=spec.dropout,
            seed=spec.dropout_seed,
        )
    else:
        scores = g.matmul(qh, g.transpose(kh, perm=(0, 1, 3, 2)))
        attn = g.softmax(g.mul(scores, inputs["scale"]), axis=-1)
        ctx = g.matmul(
            _dropout(g, attn, spec.dropout, spec.dropout_seed, key, 0), vh
        )

    merged = g.reshape(g.transpose(ctx, perm=(0, 2, 1, 3)), shape=(spec.batch, spec.seq, d))
    x1 = g.add(x, g.matmul(merged, inputs["wo"]))

    # 前馈
    h2 = g.layernorm(x1, inputs["ln2_g"], inputs["ln2_b"], axes=(-1,), eps=spec.eps)
    act = g.silu(g.matmul(h2, inputs["w1"]))
    ff = g.matmul(
        _dropout(g, act, spec.dropout, spec.dropout_seed + 1, key, 1), inputs["w2"]
    )

    return inputs, g.add(x1, ff)


# --------------------------------------------------------------------------
# 可训练的 decoder-only 语言模型
# --------------------------------------------------------------------------


def causal_mask_bias(seq: int) -> np.ndarray:
    """因果掩码的加性偏置 ``(1, 1, seq, seq)``：禁止位取 -1e9，允许位取 0。

    用 ``-1e9`` 而不是 ``-inf``：softmax 里 ``exp(x - max)`` 对 -1e9 已经是 0，
    但 ``-inf`` 参与后续乘法/反向时会产生 NaN（``0 * inf``）。有限的大负数
    在有界图里更安全，且与"禁止"在数值上无区别（概率 < 1e-300）。
    """
    if seq <= 0:
        raise ValueError(f"seq 必须为正，得到 {seq}")
    mask = np.tril(np.ones((seq, seq), dtype=np.float32))
    bias = np.where(mask > 0, 0.0, -1e9).astype(np.float32)
    return bias.reshape(1, 1, seq, seq)


@dataclass(frozen=True)
class TransformerSpec:
    """decoder-only 语言模型的形状与层数。

    ``seq`` 与 ``vocab`` 都是静态整数：位置编码是长度 ``seq`` 的参数，
    LM head 的输出宽度是 ``vocab``，两者都要在构图期确定。``vocab=0`` 表示
    不做词嵌入与 LM head，直接吃 ``(B, S, d)`` 的隐状态（纯编码器用法）。
    """

    d_model: int = 64
    num_heads: int = 4
    d_ff: int = 128
    num_layers: int = 2
    seq: int = 16
    vocab: int = 0
    eps: float = 1e-5
    batch: str = "B"
    dropout: float = 0.0
    dropout_seed: int = 0
    fused_attention: bool = False
    # 与 ``BlockSpec.rng_input`` 同义：所有层的 dropout 共用一个 i32[1] key，
    # 各站点靠 offset 区分。
    rng_input: Optional[str] = None

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} 不能被 num_heads={self.num_heads} 整除"
            )
        if self.num_layers < 1:
            raise ValueError(f"num_layers 至少为 1，得到 {self.num_layers}")
        if self.seq < 1:
            raise ValueError(f"seq 至少为 1，得到 {self.seq}")
        if self.vocab < 0:
            raise ValueError(f"vocab 不能为负，得到 {self.vocab}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout 必须在 [0, 1) 内，得到 {self.dropout}")
        if self.rng_input and self.fused_attention and self.dropout > 0.0:
            raise ValueError(
                "fused_attention 的 dropout 随机源写在算子 attrs 里（常量 seed），"
                "无法跟随运行时 key；请关掉 fused_attention，或把 dropout 设为 0"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        d, dff, seq = self.d_model, self.d_ff, self.seq
        shapes: Dict[str, Tuple[int, ...]] = {}
        if self.vocab:
            shapes["tok_emb"] = (self.vocab, d)
        shapes["pos"] = (1, seq, d)
        for i in range(self.num_layers):
            p = f"l{i}_"
            shapes.update(
                {
                    p + "ln1_g": (d,),
                    p + "ln1_b": (d,),
                    p + "wq": (d, d),
                    p + "wk": (d, d),
                    p + "wv": (d, d),
                    p + "wo": (d, d),
                    p + "ln2_g": (d,),
                    p + "ln2_b": (d,),
                    p + "w1": (d, dff),
                    p + "w2": (dff, d),
                }
            )
        shapes["final_g"] = (d,)
        shapes["final_b"] = (d,)
        if self.vocab:
            shapes["w_head"] = (d, self.vocab)
        return shapes


def _decoder_block(
    g: Graph,
    x: Value,
    params: Dict[str, Value],
    spec: TransformerSpec,
    mask: Value,
    prefix: str,
    scale: Value | None,
    scale_value: float,
    layer: int,
    key: Value | None = None,
) -> Value:
    d, h, dh, seq = spec.d_model, spec.num_heads, spec.head_dim, spec.seq
    batch = spec.batch
    # 每层两个 dropout 位置：注意力概率与 FFN 激活。seed 随层号偏移，
    # 否则整栈会复用同一张掩码，等于把"逐层独立"退化成"逐层相同"。
    seed_attn = spec.dropout_seed + layer * 2
    seed_ffn = seed_attn + 1

    h1 = g.layernorm(x, params[prefix + "ln1_g"], params[prefix + "ln1_b"], axes=(-1,), eps=spec.eps)
    q = g.matmul(h1, params[prefix + "wq"])
    k = g.matmul(h1, params[prefix + "wk"])
    v = g.matmul(h1, params[prefix + "wv"])

    head_shape = (batch, seq, h, dh)
    qh = g.transpose(g.reshape(q, shape=head_shape), perm=(0, 2, 1, 3))
    kh = g.transpose(g.reshape(k, shape=head_shape), perm=(0, 2, 1, 3))
    vh = g.transpose(g.reshape(v, shape=head_shape), perm=(0, 2, 1, 3))

    if spec.fused_attention:
        # 因果掩码作为 attention 的第 4 个输入（加性偏置），缩放变成算子属性。
        ctx = g.attention(
            qh,
            kh,
            vh,
            mask,
            scale=scale_value,
            dropout=spec.dropout,
            seed=seed_attn,
        )
    else:
        scores = g.matmul(qh, g.transpose(kh, perm=(0, 1, 3, 2)))
        # 因果掩码是加性偏置：加在缩放之后、softmax 之前，禁止位的概率严格为 0
        scores = g.add(g.mul(scores, scale), mask)
        attn = g.softmax(scores, axis=-1)
        ctx = g.matmul(
            _dropout(g, attn, spec.dropout, seed_attn, key, layer * 2), vh
        )
    merged = g.reshape(g.transpose(ctx, perm=(0, 2, 1, 3)), shape=(batch, seq, d))
    x1 = g.add(x, g.matmul(merged, params[prefix + "wo"]))

    h2 = g.layernorm(x1, params[prefix + "ln2_g"], params[prefix + "ln2_b"], axes=(-1,), eps=spec.eps)
    act = g.gelu(g.matmul(h2, params[prefix + "w1"]))
    ff = g.matmul(
        _dropout(g, act, spec.dropout, seed_ffn, key, layer * 2 + 1),
        params[prefix + "w2"],
    )
    return g.add(x1, ff)


def build_transformer(
    g: Graph, spec: TransformerSpec | None = None
) -> Tuple[Dict[str, Value], Value]:
    """构建多层 decoder-only Transformer，返回 ``(输入表, 输出 Value)``。

    输入表含 ``pos``（可学习位置编码）与 ``mask``（因果偏置，由
    :func:`causal_mask_bias` 生成）。``mask`` 不是参数，训练时不要把它交给
    优化器——把它放进 ``ParameterStore.from_graph`` 的 ``exclude`` 里。

    ``dropout>0`` 时在每层插入两处 ``dropout_mask``（注意力概率、FFN 激活），
    seed 按层号偏移；``dropout=0`` 时图里没有对应节点，这就是推理图。
    """
    spec = spec or TransformerSpec()
    d = spec.d_model

    inputs: Dict[str, Value] = {}
    if spec.vocab:
        inputs["ids"] = g.input("ids", i32[spec.batch, spec.seq])
        inputs["tok_emb"] = g.input("tok_emb", f32[spec.vocab, d])
    else:
        inputs["x"] = g.input("x", f32[spec.batch, spec.seq, d])

    inputs["pos"] = g.input("pos", f32[1, spec.seq, d])
    inputs["mask"] = g.input("mask", f32[1, 1, spec.seq, spec.seq])
    for name, shape in spec.param_shapes.items():
        if name not in inputs:
            inputs[name] = g.input(name, f32[shape])

    # 输入统一在构图前声明，读起来更清楚；但这不是序列化的硬约束——
    # 节点输出带显式 SSA 名，输入与节点交错声明也能无损回读。
    key = None
    if spec.rng_input and spec.dropout > 0.0:
        key = inputs[spec.rng_input] = g.input(spec.rng_input, i32[1])
    x = g.embedding(inputs["tok_emb"], inputs["ids"]) if spec.vocab else inputs["x"]

    x = g.add(x, inputs["pos"])
    scale_value = 1.0 / math.sqrt(spec.head_dim)
    # fused 路径把缩放写进 attention 的 attrs，不需要图里的常量节点。
    scale = None if spec.fused_attention else g.constant(scale_value, dtype=f32, shape=(1,))
    for i in range(spec.num_layers):
        x = _decoder_block(
            g, x, inputs, spec, inputs["mask"], f"l{i}_", scale, scale_value, i, key
        )

    x = g.layernorm(x, inputs["final_g"], inputs["final_b"], axes=(-1,), eps=spec.eps)
    if spec.vocab:
        return inputs, g.matmul(x, inputs["w_head"])
    return inputs, x
