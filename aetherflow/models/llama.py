"""Llama / Qwen2 系 decoder-only 语言模型的 IR 构建器。

这是"框架能不能真的训一个开源 LLM"的第一块拼图：把一个真实架构完整地表达成
IR 图，而不是靠新增算子。当前覆盖的结构（Qwen2.5-0.5B 与 Llama 同构）：

    embedding → [ rmsnorm → qkv → RoPE → attention(GQA) → o_proj → 残差
                  → rmsnorm → gate / up → silu(gate) * up → down → 残差 ] × L
              → final rmsnorm → lm_head

**只依赖已有算子**：``embedding`` / ``rmsnorm`` / ``matmul`` / ``add`` / ``mul`` /
``silu`` / ``attention`` / ``reshape`` / ``transpose`` / ``slice`` / ``concat`` /
``broadcast_to`` / ``sin`` / ``cos``。没有为某个架构开后门，也没有把"这几层拼
起来"写成新的融合算子——融合是后端的事，IR 只负责语义。

### 与 HuggingFace 的口径对齐（对拍时逐项可查）

- RMSNorm：``x / sqrt(mean(x^2) + eps) * w``，``eps`` 默认 1e-6，归一化最后一维。
- RoPE：与 HF 的 ``rotate_half`` 完全一致——把 head_dim 的**前后两半**配对
  （不是相邻两维配对）：``rot = cat(-x2, x1)``，``out = x * cos + rot * sin``；
  频率表 ``freqs = outer(position, inv_freq)``，``inv_freq = theta^(-2i/d)``，
  ``emb = cat(freqs, freqs)``。
- attention：``softmax(q·k^T * scale + mask) · v``，``scale`` 默认 ``head_dim**-0.5``；
  mask 是加性偏置（见 ``causal_mask_bias``）。
- SwiGLU：``down(silu(gate(x)) * up(x))``。
- Qwen2 的 q/k/v 投影带 bias，Llama 不带 → ``qkv_bias``。

### 两个刻意的取舍

1. **``seq`` 必须静态**。RoPE 的频率表与因果掩码的内容都随序列长度变化，IR 里
   没有 ``arange`` 这类"按长度生成"的算子（前端会把 ``arange`` 折成常量），
   符号化的 ``seq`` 无法表达这两张表。代价是每换一个长度要重建图；收益是
   decode 之外的全部训练/预填场景都能一张图跑完。
2. **GQA 用 ``reshape + broadcast_to + reshape`` 扩头**，不引入 repeat 算子。
   语义上等价于 HF 的 ``repeat_kv``：``(B, KV, 1, S, D) → (B, KV, G, S, D)``
   再摊平。它比先物化一份 ``(B, H, S, D)`` 的副本更诚实——"复制"这件事发生在
   广播里，后端可以自由决定要不要真的复制。

### 参数布局

参数名与 HuggingFace 的 ``state_dict`` **同名**，但形状是 IR 的约定：
``matmul`` 的权重是 ``(in, out)``，而 ``torch.nn.Linear.weight`` 是 ``(out, in)``。
所以所有 2D 投影权重在 IR 里都是转置后的形状，加载时用
``load_into_store(..., transpose=llama_param_transposes(spec))`` 一次性转好，
**图里不做转置**——``lm_head`` 是 ``(d, vocab)``，每步搬一次 136M 元素的转置
会把训练拖成内存瓶颈。

``tie_weights``（Qwen2.5-0.5B 为 True）在这里表达为"``lm_head.weight`` 独立声明"：
图需要的是一个 ``(d, vocab)`` 的边，而 ``embed_tokens`` 是 ``(vocab, d)``，
图内转置太贵。加载时把 ``embed_tokens.weight`` 转置后同时填给两者即可；训练期
是否合并两份梯度是训练器的事，不属于构建器。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, FrozenSet, Mapping, Optional, Tuple

import numpy as np

from ..ir.graph import Graph, Value
from ..ir.types import Dim, f32, i32
from .transformer import causal_mask_bias

if TYPE_CHECKING:  # 只为类型注解：weights 会反过来导入本包的运行期模块
    from ..train.params import ParameterStore

__all__ = [
    "LlamaSpec",
    "build_llama",
    "load_llama_weights",
    "llama_inv_freq",
    "llama_param_shapes",
    "llama_param_transposes",
]


@dataclass(frozen=True)
class LlamaSpec:
    """Llama / Qwen2 的形状与超参。

    默认值是一组"能在一张卡上跑通闭环"的小配置，真实模型用
    :meth:`qwen2_5_0_5b` 或直接填字段。
    """

    vocab_size: int = 1024
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2
    d_ff: int = 256
    seq: int = 16
    batch: Dim = "B"
    norm_eps: float = 1e-6
    rope_theta: float = 1e6
    qkv_bias: bool = False
    causal: bool = True
    scale: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.seq, int):
            raise ValueError(
                f"seq 必须是静态整数，得到 {self.seq!r}：RoPE 频率表与因果掩码"
                "的内容随长度变化，IR 里没有按符号长度生成它们的算子"
            )
        if self.seq <= 0:
            raise ValueError(f"seq 必须为正，得到 {self.seq}")
        for name in ("vocab_size", "d_model", "num_layers", "num_heads",
                     "num_kv_heads", "d_ff"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数，得到 {value!r}")
        if self.d_model % self.num_heads:
            raise ValueError(
                f"d_model ({self.d_model}) 必须能被 num_heads ({self.num_heads}) 整除"
            )
        if self.head_dim % 2:
            raise ValueError(
                f"head_dim ({self.head_dim}) 必须是偶数：RoPE 把最后一维按前后两半配对"
            )
        if self.num_heads % self.num_kv_heads:
            raise ValueError(
                f"num_heads ({self.num_heads}) 必须是 num_kv_heads "
                f"({self.num_kv_heads}) 的整数倍（GQA 要求每组大小相同）"
            )
        if self.scale is not None and not self.scale > 0:
            raise ValueError(f"scale 必须为正，得到 {self.scale}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def group_size(self) -> int:
        """每个 KV 头被多少个 query 头共享。1 表示 MHA。"""
        return self.num_heads // self.num_kv_heads

    @property
    def attn_scale(self) -> float:
        return self.scale if self.scale is not None else self.head_dim ** -0.5

    @classmethod
    def qwen2_5_0_5b(cls, **overrides) -> "LlamaSpec":
        """Qwen2.5-0.5B 的官方结构超参（``seq`` 等仍可覆盖）。"""
        base = dict(
            vocab_size=151936,
            d_model=896,
            num_layers=24,
            num_heads=14,
            num_kv_heads=2,
            d_ff=4864,
            seq=128,
            norm_eps=1e-6,
            rope_theta=1e6,
            qkv_bias=True,
        )
        base.update(overrides)
        return cls(**base)


def llama_inv_freq(spec: LlamaSpec) -> np.ndarray:
    """RoPE 的逆频率 ``theta^(-2i/head_dim)``，长度 ``head_dim / 2``。

    与 HF 的 ``Qwen2RotaryEmbedding`` 一致：``i`` 取 ``0, 2, 4, ...``。
    单独暴露出来是为了让对拍脚本能在主机上复算同一张表，而不是"相信构建器"。
    """
    index = np.arange(0, spec.head_dim, 2, dtype=np.float64)
    return (1.0 / (spec.rope_theta ** (index / spec.head_dim))).astype(np.float32)


def llama_param_shapes(spec: LlamaSpec) -> Dict[str, Tuple[int, ...]]:
    """IR 参数名 → 形状。名字与 HF ``state_dict`` 相同，形状已转置成 IR 约定。"""
    d = spec.d_model
    hd = spec.head_dim
    q_dim = spec.num_heads * hd
    kv_dim = spec.kv_dim
    ff = spec.d_ff
    vocab = spec.vocab_size

    shapes: Dict[str, Tuple[int, ...]] = {"model.embed_tokens.weight": (vocab, d)}
    for i in range(spec.num_layers):
        p = f"model.layers.{i}"
        shapes[f"{p}.self_attn.q_proj.weight"] = (d, q_dim)
        shapes[f"{p}.self_attn.k_proj.weight"] = (d, kv_dim)
        shapes[f"{p}.self_attn.v_proj.weight"] = (d, kv_dim)
        shapes[f"{p}.self_attn.o_proj.weight"] = (q_dim, d)
        shapes[f"{p}.input_layernorm.weight"] = (d,)
        shapes[f"{p}.post_attention_layernorm.weight"] = (d,)
        shapes[f"{p}.mlp.gate_proj.weight"] = (d, ff)
        shapes[f"{p}.mlp.up_proj.weight"] = (d, ff)
        shapes[f"{p}.mlp.down_proj.weight"] = (ff, d)
        if spec.qkv_bias:
            shapes[f"{p}.self_attn.q_proj.bias"] = (q_dim,)
            shapes[f"{p}.self_attn.k_proj.bias"] = (kv_dim,)
            shapes[f"{p}.self_attn.v_proj.bias"] = (kv_dim,)
    shapes["model.norm.weight"] = (d,)
    shapes["lm_head.weight"] = (d, vocab)
    return shapes


def llama_param_transposes(spec: LlamaSpec) -> FrozenSet[str]:
    """加载 HF 权重时需要转置的参数名（``nn.Linear.weight`` 是 ``(out, in)``）。

    ``embed_tokens`` 与 norm 的权重天然就是 IR 形状，不在此列。
    """
    names = {"lm_head.weight"}
    for i in range(spec.num_layers):
        p = f"model.layers.{i}"
        names.update(
            {
                f"{p}.self_attn.q_proj.weight",
                f"{p}.self_attn.k_proj.weight",
                f"{p}.self_attn.v_proj.weight",
                f"{p}.self_attn.o_proj.weight",
                f"{p}.mlp.gate_proj.weight",
                f"{p}.mlp.up_proj.weight",
                f"{p}.mlp.down_proj.weight",
            }
        )
    return frozenset(names)


def build_llama(
    spec: LlamaSpec,
    *,
    name: str = "llama",
    with_logits: bool = True,
) -> Tuple[Graph, Dict[str, Value]]:
    """构建一张完整的 Llama / Qwen2 前向图。

    返回 ``(graph, inputs)``：``inputs`` 里同时有数据输入 ``tokens`` 与全部参数
    的 :class:`Value`，训练器用 ``ParameterStore.from_graph(graph, exclude=("tokens",))``
    即可把参数与数据分开。
    """
    shapes = llama_param_shapes(spec)
    g = Graph(name)

    tokens = g.input("tokens", i32[spec.batch, spec.seq])
    params = {key: g.input(key, f32[shape]) for key, shape in shapes.items()}

    x = g.embedding(params["model.embed_tokens.weight"], tokens)

    cos, sin = _rope_tables(g, spec)
    mask = (
        g.constant(
            data=causal_mask_bias(spec.seq),
            shape=(1, 1, spec.seq, spec.seq),
        )
        if spec.causal
        else None
    )

    for i in range(spec.num_layers):
        x = _block(g, x, params, spec, i, cos, sin, mask)

    x = g.rmsnorm(
        x, params["model.norm.weight"], axes=(-1,), eps=spec.norm_eps
    )
    if with_logits:
        g.output(g.matmul(x, params["lm_head.weight"]), "logits")
    else:
        g.output(x, "hidden")
    g.seal()
    return g, {"tokens": tokens, **params}


def load_llama_weights(
    spec: LlamaSpec,
    store: "ParameterStore",
    tensors: Mapping[str, np.ndarray],
    *,
    tie_weights: Optional[bool] = None,
    strict: bool = True,
    allow_downcast: bool = False,
) -> Tuple[str, ...]:
    """把 HF 形态的张量装进 ``ParameterStore``，返回没用到的张量名。

    与 ``load_into_store`` 的区别只有一处：**`tie_weights`**。Qwen2.5-0.5B 的
    ``lm_head`` 与 ``embed_tokens`` 共享权重，而两者在 IR 里是形状不同的两个
    参数（``(d, V)`` 与 ``(V, d)``）。HF 存 safetensors 时会把 tied 的那份去掉，
    文件里只剩 ``model.embed_tokens.weight``；这里把它映射给 ``lm_head.weight``，
    转置仍由 ``llama_param_transposes`` 统一处理——两个参数共用一个来源数组，
    依次转置成各自的形状。

    ``tie_weights=None`` 时按文件内容判断（缺 ``lm_head.weight`` 即视为 tied），
    因为"文件里有什么"是唯一不会骗人的依据；显式传布尔值则覆盖该判断。
    """
    from ..weights import load_into_store

    tied = "lm_head.weight" not in tensors if tie_weights is None else bool(tie_weights)
    mapping: Dict[str, str] = {}
    if tied:
        if "model.embed_tokens.weight" not in tensors:
            raise KeyError(
                "按 tied 权重加载，但张量里既没有 'lm_head.weight' 也没有 "
                "'model.embed_tokens.weight'：无法确定 lm_head 的来源"
            )
        mapping["lm_head.weight"] = "model.embed_tokens.weight"
    return load_into_store(
        store,
        tensors,
        mapping=mapping,
        transpose=llama_param_transposes(spec),
        strict=strict,
        allow_downcast=allow_downcast,
    )


# --------------------------------------------------------------------------
# 图内构件
# --------------------------------------------------------------------------


def _rope_tables(g: Graph, spec: LlamaSpec) -> Tuple[Value, Value]:
    """``(cos, sin)``，各 ``(seq, head_dim)``，由 IR 的 ``cos`` / ``sin`` 算出。

    主机只算 ``outer(position, inv_freq)`` 这一步（纯乘加），三角函数本身留给
    图：这样 sin/cos 算子的语义、精度与后端实现都要在同一次前向里被真的用到，
    而不是"构建器在 numpy 里算完、后端缺这两个算子也发现不了"。

    表只建一次、全层复用（24 层共享同一对 Value，CSE 之外的手工复用）。
    """
    half = spec.head_dim // 2
    position = np.arange(spec.seq, dtype=np.float32)[:, None]
    freqs = (position * llama_inv_freq(spec)[None, :]).astype(np.float32)
    emb = g.concat(
        g.constant(data=freqs, shape=(spec.seq, half)),
        g.constant(data=freqs, shape=(spec.seq, half)),
        axis=-1,
    )
    return g.cos(emb), g.sin(emb)


def _rope(g: Graph, x: Value, cos: Value, sin: Value, spec: LlamaSpec) -> Value:
    """HF ``apply_rotary_pos_emb``：前后两半配对旋转。"""
    half = spec.head_dim // 2
    first = g.slice(x, dim=-1, start=0, stop=half)
    second = g.slice(x, dim=-1, start=half, stop=spec.head_dim)
    rotated = g.concat(g.neg(second), first, axis=-1)
    return g.add(g.mul(x, cos), g.mul(rotated, sin))


def _repeat_kv(g: Graph, x: Value, spec: LlamaSpec) -> Value:
    """把 KV 头广播到 query 头数；``group_size == 1`` 时原样返回。"""
    group = spec.group_size
    if group == 1:
        return x
    expanded = g.reshape(
        x, shape=(spec.batch, spec.num_kv_heads, 1, spec.seq, spec.head_dim)
    )
    expanded = g.broadcast_to(
        expanded, shape=(spec.batch, spec.num_kv_heads, group, spec.seq, spec.head_dim)
    )
    return g.reshape(
        expanded, shape=(spec.batch, spec.num_heads, spec.seq, spec.head_dim)
    )


def _block(
    g: Graph,
    x: Value,
    params: Dict[str, Value],
    spec: LlamaSpec,
    index: int,
    cos: Value,
    sin: Value,
    mask: Optional[Value],
) -> Value:
    p = f"model.layers.{index}"
    batch, seq = spec.batch, spec.seq
    q_dim = spec.num_heads * spec.head_dim

    h = g.rmsnorm(x, params[f"{p}.input_layernorm.weight"], axes=(-1,), eps=spec.norm_eps)
    q = g.matmul(h, params[f"{p}.self_attn.q_proj.weight"])
    k = g.matmul(h, params[f"{p}.self_attn.k_proj.weight"])
    v = g.matmul(h, params[f"{p}.self_attn.v_proj.weight"])
    if spec.qkv_bias:
        q = g.add(q, params[f"{p}.self_attn.q_proj.bias"])
        k = g.add(k, params[f"{p}.self_attn.k_proj.bias"])
        v = g.add(v, params[f"{p}.self_attn.v_proj.bias"])

    q = g.transpose(g.reshape(q, shape=(batch, seq, spec.num_heads, spec.head_dim)),
                    perm=(0, 2, 1, 3))
    k = g.transpose(g.reshape(k, shape=(batch, seq, spec.num_kv_heads, spec.head_dim)),
                    perm=(0, 2, 1, 3))
    v = g.transpose(g.reshape(v, shape=(batch, seq, spec.num_kv_heads, spec.head_dim)),
                    perm=(0, 2, 1, 3))

    q = _rope(g, q, cos, sin, spec)
    k = _rope(g, k, cos, sin, spec)
    k = _repeat_kv(g, k, spec)
    v = _repeat_kv(g, v, spec)

    args = [q, k, v] if mask is None else [q, k, v, mask]
    ctx = g.attention(*args, scale=spec.attn_scale)
    merged = g.reshape(
        g.transpose(ctx, perm=(0, 2, 1, 3)), shape=(batch, seq, q_dim)
    )
    x = g.add(x, g.matmul(merged, params[f"{p}.self_attn.o_proj.weight"]))

    h = g.rmsnorm(
        x, params[f"{p}.post_attention_layernorm.weight"], axes=(-1,), eps=spec.norm_eps
    )
    gate = g.matmul(h, params[f"{p}.mlp.gate_proj.weight"])
    up = g.matmul(h, params[f"{p}.mlp.up_proj.weight"])
    x = g.add(
        x, g.matmul(g.mul(g.silu(gate), up), params[f"{p}.mlp.down_proj.weight"])
    )
    return x
