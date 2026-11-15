"""张量并行：列并行 / 行并行线性层、attention 头切分与 TP 构建器。

这一层回答的问题是"一块卡放不下的权重，怎么按列 / 行切开后仍然得到**同一个
结果**"，而不是"怎么在多卡上跑得更快"。它不新增执行机制，也不新增内核：

- 切分后的每一份子图仍然由同一份 IR 表达，交给同一个后端执行；
- 唯一的跨卡动作是 **求和归约**（行并行的部分和），走调用方注入的集合通信
  对象（``LoopbackWorld`` / ``NcclHostCollective`` / 任何带 ``all_reduce`` 的
  对象），与 ZeRO 的注入方式一致；
- 数值判据是"逐位或容差内等于不分片的同一张图"，不是"看起来在并行"。

三条结构性事实：

1. **列并行不需要通信。** 权重 ``(K, N)`` 按输出维切成 ``(K, N/W)``，各 rank
   算出的是一段**独立的输出特征**，下游按列拼接即可。``linear_i8`` 之外的
   线性层都适用。
2. **行并行必须归约。** 权重 ``(K, N)`` 按输入维切成 ``(K/W, N)``，各 rank
   算出的是**部分和**，只有求和之后才是真值。因此 :func:`row_parallel_linear`
   返回的是"部分和"，注释和命名都写明这一点——静默当成完整输出会得到
   ``1/W`` 的结果，且不会报错。
3. **attention 的头切分是"列并行 + 头维对齐"。** q/k/v 的输出维按 head 分组
   （``N = num_heads * head_dim``，按 head 顺序排列），所以"按列切"与"按头切"
   是同一个操作，前提是 ``num_heads`` 能被 ``world_size`` 整除；``wo`` 的行维
   同样按 head 分组，于是它天然是行并行。

**集合通信只能在图外发生。** IR 里没有 collective 节点（加了它就得为每个
后端定义设备侧语义，而参考后端没有"卡"的概念）。所以本层把 block 拆成两张图：
``attn`` 段与 ``ff`` 段，段间由 :class:`TensorParallelBlock` 在主机侧做一次
``all_reduce``，再拼上残差。图外面只剩两次加法，其余（LayerNorm、matmul、
softmax、SiLU）全部在 IR 里。

**这一层只做前向。** 反向的集合通信（行并行输入梯度求和、列并行权重梯度按列
拼接）与优化器状态分片属于 Zero / FSDP 的范畴，未接入；不要把它当成可训练的
张量并行实现。切分后的参数仍是普通 numpy 数组，交给 ``ParameterStore`` 的
只是"每个 rank 各自的那一份"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..backends.reference_backend import ReferenceBackend
from ..ir.graph import Graph, Value
from ..ir.types import f32
from ..models.transformer import BlockSpec

__all__ = [
    "shard_size",
    "split_axis",
    "merge_axis",
    "column_parallel_linear",
    "row_parallel_linear",
    "build_tensor_parallel_attention",
    "TensorParallelBlock",
]


# --------------------------------------------------------------------------
# 切分工具
# --------------------------------------------------------------------------


def shard_size(full: int, world_size: int, *, what: str = "维度") -> int:
    """``full`` 能被 ``world_size`` 整除时返回每份大小，否则显式报错。

    不补零、不丢尾：tensor parallel 的每一份都要参与同一组矩阵乘，长度不齐
    意味着两个 rank 的图形状不同，这在静态 IR 里是不可表达的。
    """
    if isinstance(full, bool) or not isinstance(full, int):
        raise TypeError(f"{what} 必须是 int，得到 {type(full).__name__}")
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise TypeError(f"world_size 必须是 int，得到 {type(world_size).__name__}")
    if world_size < 1:
        raise ValueError(f"world_size 必须 >= 1，得到 {world_size}")
    if full < 1:
        raise ValueError(f"{what} 必须为正，得到 {full}")
    if full % world_size != 0:
        raise ValueError(
            f"{what}={full} 不能被 world_size={world_size} 整除；"
            "tensor parallel 的每份必须等长，否则各 rank 的静态图形状不同"
        )
    return full // world_size


def split_axis(
    array: np.ndarray,
    axis: int,
    rank: int,
    world_size: int,
    *,
    what: str = "张量",
) -> np.ndarray:
    """把 ``array`` 沿 ``axis`` 等切成 ``world_size`` 份，取第 ``rank`` 份。

    返回的是**切片视图的副本**（``np.array(..., copy=True)``）：参数会被
    优化器原地更新，共享视图会让一个 rank 的更新悄悄改到另一份上。
    """
    values = np.asarray(array)
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise TypeError(f"world_size 必须是 int，得到 {type(world_size).__name__}")
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError(f"rank 必须是 int，得到 {type(rank).__name__}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank={rank} 越界：world_size={world_size}")
    ndim = values.ndim
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis={axis} 越界：{what} 的秩是 {ndim}")
    norm = axis % ndim
    part = shard_size(values.shape[norm], world_size, what=f"{what} 的第 {norm} 维")
    start = rank * part
    index = [slice(None)] * ndim
    index[norm] = slice(start, start + part)
    return np.array(values[tuple(index)], copy=True)


def merge_axis(parts: Sequence[np.ndarray], axis: int, *, what: str = "张量") -> np.ndarray:
    """按 rank 顺序把 :func:`split_axis` 的产物拼回完整张量。"""
    if isinstance(parts, (str, bytes)) or not isinstance(parts, Sequence):
        raise TypeError(f"parts 必须是张量序列，得到 {type(parts).__name__}")
    if not parts:
        raise ValueError("parts 不能为空")
    arrays = [np.asarray(p) for p in parts]
    shape = arrays[0].shape
    for index, item in enumerate(arrays[1:], start=1):
        if item.ndim != len(shape):
            raise ValueError(
                f"第 {index} 份的秩 {item.ndim} 与第 0 份的秩 {len(shape)} 不一致"
            )
        for dim, (got, want) in enumerate(zip(item.shape, shape)):
            if dim == axis % len(shape):
                continue
            if got != want:
                raise ValueError(
                    f"第 {index} 份在非切分维 {dim} 上长度 {got} 与第 0 份的 {want} 不一致"
                )
    return np.concatenate(arrays, axis=axis)


def _static_dim(value: Value, axis: int) -> Optional[int]:
    shape = value.type.shape
    if not -len(shape) <= axis < len(shape):
        return None
    dim = shape[axis % len(shape)]
    return dim if isinstance(dim, int) else None


def _check_matmul_dims(x: Value, weight: Value, *, what: str) -> None:
    if weight.type.rank != 2:
        raise ValueError(f"{what} 的权重必须是 2 维，得到 {weight.type}")
    if x.type.rank < 2:
        raise ValueError(f"{what} 的输入至少 2 维，得到 {x.type}")
    inner_x = _static_dim(x, -1)
    inner_w = _static_dim(weight, 0)
    if inner_x is not None and inner_w is not None and inner_x != inner_w:
        raise ValueError(
            f"{what} 的内维不匹配：输入最后一维 {inner_x}，权重第 0 维 {inner_w}"
        )


# --------------------------------------------------------------------------
# 列并行 / 行并行线性层
# --------------------------------------------------------------------------


def column_parallel_linear(
    g: Graph, x: Value, weight: Value, bias: Optional[Value] = None
) -> Value:
    """列并行线性层：``weight`` 是**本 rank 的列切片** ``(K, N/W)``。

    输出形状是 ``x.shape[:-1] + (N/W,)``：一段独立的输出特征，下游不需要
    任何通信就能继续算。``bias`` 同理是本 rank 的 ``(N/W,)`` 切片。
    """
    _check_matmul_dims(x, weight, what="列并行线性层")
    if bias is not None and bias.type.rank != 1:
        raise ValueError(f"列并行线性层的 bias 必须是 1 维，得到 {bias.type}")
    out = g.matmul(x, weight)
    if bias is not None:
        out = g.add(out, bias)
    return out


def row_parallel_linear(
    g: Graph, x: Value, weight: Value, bias: Optional[Value] = None
) -> Value:
    """行并行线性层：``weight`` 是本 rank 的行切片 ``(K/W, N)``。

    ``x`` 的最后一维必须已经是 ``K/W``（通常是上游列并行 + 头切分的结果）。
    返回的是**部分和**：只有把各 rank 的结果按元素求和之后才是完整输出。
    接口不隐式归约，因为归约需要集合通信，而 IR 里没有 collective 节点。
    ``bias`` 在这里**不加**——加了就会在 ``W`` 个 rank 上各加一次，归约之后
    变成 ``W * bias``。偏置由归约后的调用方加一次。
    """
    _check_matmul_dims(x, weight, what="行并行线性层")
    if bias is not None:
        raise ValueError(
            "行并行线性层不接受 bias：它会被每个 rank 各加一次，归约后放大 "
            "world_size 倍；请在 all_reduce 之后加一次"
        )
    return g.matmul(x, weight)


# --------------------------------------------------------------------------
# attention 头切分
# --------------------------------------------------------------------------


def build_tensor_parallel_attention(
    g: Graph,
    h: Value,
    wq: Value,
    wk: Value,
    wv: Value,
    wo: Value,
    *,
    num_heads: int,
    d_model: int,
    batch: Any,
    seq: Any,
    rank: int,
    world_size: int,
    scale: Optional[Value] = None,
    fused_attention: bool = False,
    out_name: str = "attn",
) -> Value:
    """在本 rank 上构建"只算自己那几个头"的 attention，输出**部分和**。

    ``wq`` / ``wk`` / ``wv`` 是列切片 ``(d_model, heads_local * head_dim)``，
    ``wo`` 是行切片 ``(heads_local * head_dim, d_model)``。头在输出维上按
    ``head * head_dim + i`` 排列，所以"沿列等切"与"按头分组"是同一个切片；
    这也是必须 ``num_heads % world_size == 0`` 的原因。

    返回的是 ``wo`` 之后的部分和，需要调用方 ``all_reduce``；未 fused 时必须
    给 ``scale``（``(1,)`` 的图输入或常量）。
    """
    heads_local = shard_size(num_heads, world_size, what="num_heads")
    if d_model % num_heads != 0:
        raise ValueError(f"d_model={d_model} 不能被 num_heads={num_heads} 整除")
    head_dim = d_model // num_heads
    local_width = heads_local * head_dim
    if _static_dim(wq, -1) not in (None, local_width):
        raise ValueError(
            f"wq 的列数应为 heads_local * head_dim = {local_width}，"
            f"得到 {_static_dim(wq, -1)}"
        )
    if _static_dim(wo, 0) not in (None, local_width):
        raise ValueError(
            f"wo 的行数应为 heads_local * head_dim = {local_width}，"
            f"得到 {_static_dim(wo, 0)}"
        )
    if not fused_attention and scale is None:
        raise ValueError("非 fused 的 attention 需要 scale 输入（1/sqrt(head_dim)）")

    q = column_parallel_linear(g, h, wq)
    k = column_parallel_linear(g, h, wk)
    v = column_parallel_linear(g, h, wv)

    head_shape = (batch, seq, heads_local, head_dim)
    qh = g.transpose(g.reshape(q, shape=head_shape), perm=(0, 2, 1, 3))
    kh = g.transpose(g.reshape(k, shape=head_shape), perm=(0, 2, 1, 3))
    vh = g.transpose(g.reshape(v, shape=head_shape), perm=(0, 2, 1, 3))

    if fused_attention:
        ctx = g.attention(qh, kh, vh, scale=1.0 / np.sqrt(head_dim), dropout=0.0, seed=0)
    else:
        scores = g.matmul(qh, g.transpose(kh, perm=(0, 1, 3, 2)))
        attn = g.softmax(g.mul(scores, scale), axis=-1)
        ctx = g.matmul(attn, vh)

    merged = g.reshape(
        g.transpose(ctx, perm=(0, 2, 1, 3)), shape=(batch, seq, local_width)
    )
    return row_parallel_linear(g, merged, wo)


# --------------------------------------------------------------------------
# TP 构建器
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorParallelBlock:
    """一个 pre-norm Transformer block 的张量并行版本。

    与前缀并行地：参数按"列 / 行 / 复制"三类切好；前向拆成两张图
    （``attn`` 段与 ``ff`` 段），段间做一次 ``all_reduce``。数值上与
    ``build_transformer_block`` 完全一致（残差、LayerNorm、SiLU 都在图内）。

    约束（都是"不满足就不能静默降级"的那类）：

    - ``num_heads % world_size == 0``（头切分）；
    - ``d_ff % world_size == 0``（前馈列切分）；
    - ``dropout == 0``：随机掩码按形状采样，各 rank 的头宽度不同，同一 seed
      采不到一致的掩码；要训练请关掉 dropout 或先实现 broadcast 掩码。
    """

    spec: BlockSpec
    rank: int
    world_size: int

    def __post_init__(self) -> None:
        spec = self.spec
        if not isinstance(spec, BlockSpec):
            raise TypeError(f"spec 必须是 BlockSpec，得到 {type(spec).__name__}")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError(f"rank 必须是 int，得到 {type(self.rank).__name__}")
        if isinstance(self.world_size, bool) or not isinstance(self.world_size, int):
            raise TypeError(
                f"world_size 必须是 int，得到 {type(self.world_size).__name__}"
            )
        if self.world_size < 1:
            raise ValueError(f"world_size 必须 >= 1，得到 {self.world_size}")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank={self.rank} 越界：world_size={self.world_size}")
        shard_size(spec.num_heads, self.world_size, what="num_heads")
        shard_size(spec.d_ff, self.world_size, what="d_ff")
        if spec.dropout > 0.0:
            raise ValueError(
                "张量并行的 block 不支持 dropout：掩码按形状采样，各 rank 的头"
                "宽度不同，同一 seed 采不到一致的掩码；请把 dropout 设为 0"
            )

    # -- 形状 ---------------------------------------------------------

    @property
    def heads_per_rank(self) -> int:
        return shard_size(self.spec.num_heads, self.world_size, what="num_heads")

    @property
    def head_dim(self) -> int:
        return self.spec.head_dim

    @property
    def local_head_width(self) -> int:
        return self.heads_per_rank * self.head_dim

    @property
    def shard_axes(self) -> Dict[str, int]:
        """参数名 → 切分维。不在表里的参数是**复制**的（LayerNorm / scale）。"""
        return {
            "wq": 1,
            "wk": 1,
            "wv": 1,
            "wo": 0,
            "w1": 1,
            "w2": 0,
        }

    @property
    def param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """本 rank 的参数形状（与 ``BlockSpec.param_shapes`` 同键）。"""
        d, dff = self.spec.d_model, self.spec.d_ff
        width = self.local_head_width
        local_ff = shard_size(dff, self.world_size, what="d_ff")
        shapes: Dict[str, Tuple[int, ...]] = {
            "ln1_g": (d,),
            "ln1_b": (d,),
            "wq": (d, width),
            "wk": (d, width),
            "wv": (d, width),
            "wo": (width, d),
            "ln2_g": (d,),
            "ln2_b": (d,),
            "w1": (d, local_ff),
            "w2": (local_ff, d),
        }
        if not self.spec.fused_attention:
            shapes["scale"] = (1,)
        return shapes

    def split_dense(self, params: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """把一份完整（不分片）参数切成"本 rank 的那一份"。

        ``params`` 的键必须与 ``spec.param_shapes`` 一致；形状按完整形状校验，
        不猜、不补零。复制类参数（LayerNorm 的 ``g`` / ``b``、``scale``）必须
        拷贝，否则优化器原地更新会改到别的 rank 的同一块内存上。
        """
        if not isinstance(params, Mapping):
            raise TypeError(f"params 必须是映射，得到 {type(params).__name__}")
        full = self.spec.param_shapes
        missing = sorted(set(full) - set(params))
        extra = sorted(set(params) - set(full))
        if missing:
            raise ValueError(f"缺少参数：{missing}")
        if extra:
            raise ValueError(f"多出参数：{extra}")
        local: Dict[str, np.ndarray] = {}
        for name, shape in full.items():
            array = np.asarray(params[name])
            if tuple(array.shape) != tuple(shape):
                raise ValueError(
                    f"参数 {name!r} 的形状 {tuple(array.shape)} 与声明的 "
                    f"{tuple(shape)} 不一致"
                )
            axis = self.shard_axes.get(name)
            if axis is None:
                local[name] = np.array(array, copy=True)
            else:
                local[name] = split_axis(
                    array, axis, self.rank, self.world_size, what=f"参数 {name}"
                )
        return local

    # -- 构图 ---------------------------------------------------------

    def build(self) -> Tuple[Graph, Graph]:
        """构建两段图：``attn``（输出部分和）与 ``ff``（输出部分和）。"""
        spec = self.spec
        shapes = self.param_shapes

        g_attn = Graph("tp_attn")
        x = g_attn.input("x", f32[spec.batch, spec.seq, spec.d_model])
        attn_inputs: Dict[str, Value] = {"x": x}
        for name in ("ln1_g", "ln1_b", "wq", "wk", "wv", "wo"):
            attn_inputs[name] = g_attn.input(name, f32[shapes[name]])
        scale = None
        if not spec.fused_attention:
            scale = attn_inputs["scale"] = g_attn.input("scale", f32[shapes["scale"]])
        h1 = g_attn.layernorm(
            x, attn_inputs["ln1_g"], attn_inputs["ln1_b"], axes=(-1,), eps=spec.eps
        )
        attn_partial = build_tensor_parallel_attention(
            g_attn,
            h1,
            attn_inputs["wq"],
            attn_inputs["wk"],
            attn_inputs["wv"],
            attn_inputs["wo"],
            num_heads=spec.num_heads,
            d_model=spec.d_model,
            batch=spec.batch,
            seq=spec.seq,
            rank=self.rank,
            world_size=self.world_size,
            scale=scale,
            fused_attention=spec.fused_attention,
        )
        g_attn.output(attn_partial, "attn")

        g_ff = Graph("tp_ff")
        x1 = g_ff.input("x1", f32[spec.batch, spec.seq, spec.d_model])
        ff_inputs: Dict[str, Value] = {"x1": x1}
        for name in ("ln2_g", "ln2_b", "w1", "w2"):
            ff_inputs[name] = g_ff.input(name, f32[shapes[name]])
        h2 = g_ff.layernorm(
            x1, ff_inputs["ln2_g"], ff_inputs["ln2_b"], axes=(-1,), eps=spec.eps
        )
        act = g_ff.silu(column_parallel_linear(g_ff, h2, ff_inputs["w1"]))
        ff_partial = row_parallel_linear(g_ff, act, ff_inputs["w2"])
        g_ff.output(ff_partial, "ff")

        return g_attn, g_ff

    # -- 前向 ---------------------------------------------------------

    def run(
        self,
        collective: Any,
        x: np.ndarray,
        params: Mapping[str, np.ndarray],
        *,
        backend: Any = None,
        graphs: Optional[Tuple[Graph, Graph]] = None,
    ) -> np.ndarray:
        """本 rank 的前向：两段图 + 两次 ``all_reduce``（求和语义）。

        ``params`` 是本 rank 的切片（``split_dense`` 或等价产物）。所有 rank
        必须用同一个 ``collective``、按相同顺序各调用一次本方法，否则
        ``all_reduce`` 会永远等不齐。
        """
        backend = backend or ReferenceBackend()
        g_attn, g_ff = graphs or self.build()
        values = np.asarray(x)
        if values.ndim != 3:
            raise ValueError(f"输入 x 必须是 3 维 (batch, seq, d_model)，得到 {values.shape}")
        if values.shape[-1] != self.spec.d_model:
            raise ValueError(
                f"输入 x 的最后一维应为 d_model={self.spec.d_model}，得到 {values.shape[-1]}"
            )
        full = self.param_shapes
        missing = sorted(set(full) - set(params))
        if missing:
            raise ValueError(f"缺少本 rank 的参数：{missing}")

        feed = {"x": values}
        for name in ("ln1_g", "ln1_b", "wq", "wk", "wv", "wo", "scale"):
            if name in full:
                feed[name] = np.asarray(params[name])
        attn_partial = backend.execute(g_attn, feed)["attn"]
        attn_full = _all_reduce(collective, attn_partial, what="attention 输出")
        x1 = values + attn_full

        feed_ff = {"x1": x1}
        for name in ("ln2_g", "ln2_b", "w1", "w2"):
            feed_ff[name] = np.asarray(params[name])
        ff_partial = backend.execute(g_ff, feed_ff)["ff"]
        ff_full = _all_reduce(collective, ff_partial, what="前馈输出")
        return x1 + ff_full

    def __repr__(self) -> str:
        return (
            f"TensorParallelBlock(rank={self.rank}/{self.world_size}, "
            f"heads/rank={self.heads_per_rank}, d_ff/rank="
            f"{shard_size(self.spec.d_ff, self.world_size, what='d_ff')})"
        )


def _all_reduce(collective: Any, array: np.ndarray, *, what: str) -> np.ndarray:
    """把部分和按 ``sum`` 归约，并校验形状 / 类型没有在通信里变样。"""
    if collective is None or not hasattr(collective, "all_reduce"):
        raise TypeError(
            "collective 必须提供 all_reduce(array, op='sum')；"
            "没有集合通信时张量并行无法完成行并行的求和"
        )
    values = np.asarray(array)
    reduced = np.asarray(collective.all_reduce(values, op="sum"))
    if reduced.shape != values.shape:
        raise ValueError(
            f"{what} 归约后的形状 {reduced.shape} 与部分和 {values.shape} 不一致"
        )
    if reduced.dtype != values.dtype:
        raise ValueError(
            f"{what} 归约后的 dtype {reduced.dtype} 与部分和 {values.dtype} 不一致"
        )
    return reduced
