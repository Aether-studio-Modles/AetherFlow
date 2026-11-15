"""带 KV cache 的增量注意力解码：把分页缓存里的 K/V 送进同一套 IR。

这一层回答的问题是"已经有 ``t`` 个 token 的 K/V，怎么只算当前这一个 token"，
而不是"怎么把 attention 写得更快"。它不新增算子：

- 图输入是当前 token 的隐状态 ``(B, 1, d)``、每层**已经取回**的 K/V
  ``(B, cache_steps, H, D)`` 与一个加性掩码；
- 图内把当前 token 的 k/v 拼到缓存末尾（``concat``），做一次单查询注意力，
  再走残差、FFN 与末端归一化；
- 图输出是当前 token 的隐状态与**本步新算出的 k/v**，由调用方写回缓存。

数值判据是"与 :func:`aetherflow.models.transformer.build_transformer` 在同样
权重、同样输入下逐位置一致"，不是"看起来像增量解码"。

三条边界：

1. **只做推理。** ``dropout`` 必须为 0：增量图里没有"重采样掩码"的位置，
   训练图与推理图是两张各自干净的图。
2. **只吃隐状态。** 位置编码由调用方在进图前加好；词表投影不在这里做。
   ``TransformerSpec.vocab`` 必须为 0，否则同名的 ``w_head`` 会被静默忽略。
3. **缓存布局由 :class:`aetherflow.runtime.paged_cache.PagedKVCache` 决定。**
   这里拿到的永远是"按块表拼好的连续副本"，它不知道页号，也不该知道。
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from ..ir.graph import Graph, Value
from ..ir.types import f32
from ..runtime.paged_cache import PagedKVCache
from .transformer import TransformerSpec

__all__ = [
    "build_cached_decoder_step",
    "cached_forward",
]


def _layer_weight_names() -> Tuple[str, ...]:
    """每层参与增量计算的权重名（与 ``TransformerSpec.param_shapes`` 对齐）。"""
    return (
        "ln1_g",
        "ln1_b",
        "wq",
        "wk",
        "wv",
        "wo",
        "ln2_g",
        "ln2_b",
        "w1",
        "w2",
    )


def _check_spec(spec: TransformerSpec) -> None:
    if spec.vocab != 0:
        raise ValueError(
            f"增量解码只输出隐状态，要求 vocab=0，得到 vocab={spec.vocab}；"
            "词表投影请在拿到隐状态之后自己接"
        )
    if spec.dropout != 0.0:
        raise ValueError(
            f"增量解码只支持 dropout=0（推理图），得到 {spec.dropout}；"
            "训练请用 build_transformer"
        )
    if spec.rng_input is not None:
        raise ValueError(
            "增量解码不接受 rng_input：它只在 dropout>0 时有意义，而这里只做推理"
        )


def build_cached_decoder_step(
    g: Graph, spec: TransformerSpec, *, cache_steps: int
) -> Tuple[Dict[str, Value], Value, Dict[str, Value]]:
    """构建单步增量解码图，返回 ``(inputs, y, new_kv)``。

    ``cache_steps`` 是缓存里**已保留的固定宽度**（含补零位），当前 token 的
    k/v 会拼在它末尾，因此注意力长度是 ``cache_steps + 1``。

    输入表：

    ``x``
        当前 token 的隐状态 ``(B, 1, d)``（位置编码已由调用方加好）。
    ``mask``
        加性偏置 ``(1, 1, 1, cache_steps + 1)``，0 表示可见、``-1e9`` 表示屏蔽。
        补零位与未来位都靠它屏蔽。
    ``l{i}_k`` / ``l{i}_v``
        第 ``i`` 层已缓存（或补零）的 K/V ``(B, cache_steps, H, D)``。
    ``l{i}_*``
        与 :class:`TransformerSpec` 同名的层权重。

    输出表：``y`` 与 ``l{i}_k_out`` / ``l{i}_v_out``（本步新算出的
    ``(B, 1, H, D)``，调用方负责写回缓存；名字带 ``_out`` 是因为图里输入与
    输出不能同名）。
    """
    if isinstance(cache_steps, bool) or not isinstance(cache_steps, int):
        raise TypeError(f"cache_steps 必须是 int，得到 {type(cache_steps).__name__}")
    if cache_steps < 1:
        raise ValueError(f"cache_steps 必须 >= 1，得到 {cache_steps}")
    _check_spec(spec)

    d, h, dh = spec.d_model, spec.num_heads, spec.head_dim
    batch = spec.batch
    scale_value = 1.0 / math.sqrt(dh)

    inputs: Dict[str, Value] = {
        "x": g.input("x", f32[batch, 1, d]),
        "mask": g.input("mask", f32[1, 1, 1, cache_steps + 1]),
    }
    for layer in range(spec.num_layers):
        for base in _layer_weight_names():
            inputs[f"l{layer}_{base}"] = g.input(
                f"l{layer}_{base}", f32[spec.param_shapes[f"l{layer}_{base}"]]
            )
        inputs[f"l{layer}_k"] = g.input(
            f"l{layer}_k", f32[batch, cache_steps, h, dh]
        )
        inputs[f"l{layer}_v"] = g.input(
            f"l{layer}_v", f32[batch, cache_steps, h, dh]
        )
    for base in ("final_g", "final_b"):
        inputs[base] = g.input(base, f32[spec.param_shapes[base]])

    scale = None
    if not spec.fused_attention:
        scale = g.constant(scale_value, dtype=f32, shape=(1,))

    x = inputs["x"]
    new_kv: Dict[str, Value] = {}
    for layer in range(spec.num_layers):
        prefix = f"l{layer}_"
        h1 = g.layernorm(
            x, inputs[prefix + "ln1_g"], inputs[prefix + "ln1_b"], axes=(-1,), eps=spec.eps
        )
        q = g.matmul(h1, inputs[prefix + "wq"])
        k = g.matmul(h1, inputs[prefix + "wk"])
        v = g.matmul(h1, inputs[prefix + "wv"])

        qh = g.transpose(
            g.reshape(q, shape=(batch, 1, h, dh)), perm=(0, 2, 1, 3)
        )
        kh = g.transpose(
            g.reshape(k, shape=(batch, 1, h, dh)), perm=(0, 2, 1, 3)
        )
        vh = g.transpose(
            g.reshape(v, shape=(batch, 1, h, dh)), perm=(0, 2, 1, 3)
        )
        # 缓存里的 (B, T, H, D) 换成 (B, H, T, D) 再与当前这一个位置拼接
        past_k = g.transpose(inputs[prefix + "k"], perm=(0, 2, 1, 3))
        past_v = g.transpose(inputs[prefix + "v"], perm=(0, 2, 1, 3))
        k_all = g.concat(past_k, kh, axis=2)
        v_all = g.concat(past_v, vh, axis=2)

        if spec.fused_attention:
            ctx = g.attention(
                qh,
                k_all,
                v_all,
                inputs["mask"],
                scale=scale_value,
                dropout=0.0,
                seed=0,
            )
        else:
            scores = g.matmul(qh, g.transpose(k_all, perm=(0, 1, 3, 2)))
            scores = g.add(g.mul(scores, scale), inputs["mask"])
            attn = g.softmax(scores, axis=-1)
            ctx = g.matmul(attn, v_all)

        merged = g.reshape(g.transpose(ctx, perm=(0, 2, 1, 3)), shape=(batch, 1, d))
        x = g.add(x, g.matmul(merged, inputs[prefix + "wo"]))

        h2 = g.layernorm(
            x, inputs[prefix + "ln2_g"], inputs[prefix + "ln2_b"], axes=(-1,), eps=spec.eps
        )
        act = g.gelu(g.matmul(h2, inputs[prefix + "w1"]))
        x = g.add(x, g.matmul(act, inputs[prefix + "w2"]))

        # 输出名必须与输入名（缓存里的 ``l{i}_k``）区分开：图里同名只能是
        # 同一个 Value，输入与输出不能共用一个名字。
        new_kv[prefix + "k_out"] = g.transpose(kh, perm=(0, 2, 1, 3))
        new_kv[prefix + "v_out"] = g.transpose(vh, perm=(0, 2, 1, 3))

    y = g.layernorm(x, inputs["final_g"], inputs["final_b"], axes=(-1,), eps=spec.eps)
    return inputs, y, new_kv


def _causal_step_mask(cache_steps: int, valid: int) -> np.ndarray:
    """第 ``valid`` 步的加性掩码：前 ``valid`` 个缓存位与末尾的当前位可见。"""
    mask = np.full((1, 1, 1, cache_steps + 1), -1e9, dtype=np.float32)
    mask[:, :, :, :valid] = 0.0
    mask[:, :, :, cache_steps] = 0.0
    return mask


def _host(backend, value) -> np.ndarray:
    """把（可能驻留在设备上的）数组取回主机；与 ``lm._host_array`` 同义。"""
    if isinstance(value, np.ndarray):
        return value
    convert = getattr(backend, "to_host", None)
    return np.asarray(convert(value) if convert is not None else value)


def cached_forward(
    backend,
    params: Dict[str, np.ndarray],
    spec: TransformerSpec,
    xs,
    *,
    block_size: int = 4,
) -> np.ndarray:
    """逐 token 走增量解码，返回 ``(1, T, d)`` 的隐状态序列。

    ``xs`` 是 ``(T, d)`` 的输入隐状态（位置编码请自己加好）。每一步都从
    :class:`PagedKVCache` 按块表取回该层的 K/V，执行一张固定形状的增量图，
    再把新算出的 k/v 写回缓存——与"整段跑一遍稠密图"逐位置一致。

    页宽 ``block_size`` 只影响分配粒度，不影响数值；``T`` 个 token 至少需要
    ``ceil(T / block_size)`` 页，这里按需分配，用完不回收（调用方持有返回值，
    缓存生命周期就是这次调用）。
    """
    _check_spec(spec)
    if isinstance(spec.batch, bool) or not isinstance(spec.batch, int):
        raise TypeError(
            f"cached_forward 需要静态的整数 batch，得到 {spec.batch!r}；"
            "符号 batch 无法在这里分配缓存"
        )
    if spec.batch != 1:
        raise ValueError(
            f"cached_forward 目前只支持 batch=1，得到 {spec.batch}；"
            "多序列请每行各自一份缓存"
        )
    values = np.asarray(xs, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != spec.d_model:
        raise ValueError(
            f"xs 形状必须是 (T, {spec.d_model})，得到 {values.shape}"
        )
    steps = values.shape[0]
    if steps < 1:
        raise ValueError("xs 至少要有一个位置")
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise TypeError(f"block_size 必须是 int，得到 {type(block_size).__name__}")
    if block_size < 1:
        raise ValueError(f"block_size 必须 >= 1，得到 {block_size}")

    cache = PagedKVCache(
        num_blocks=(steps + block_size - 1) // block_size,
        block_size=block_size,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        head_dim=spec.head_dim,
    )
    rid = 0
    cache.allocate(rid)

    g = Graph("cached_decoder_step")
    inputs, y, new_kv = build_cached_decoder_step(g, spec, cache_steps=steps)
    g.output(y, "y")
    for name, value in new_kv.items():
        g.output(value, name)

    layer_names = tuple(
        f"l{layer}_{base}"
        for layer in range(spec.num_layers)
        for base in _layer_weight_names()
    ) + ("final_g", "final_b")

    outs = []
    for step in range(steps):
        feed: Dict[str, np.ndarray] = {}
        for name in layer_names:
            if name not in params:
                raise KeyError(f"params 缺少权重 {name!r}")
            feed[inputs[name].name] = np.asarray(params[name], dtype=np.float32)
        feed[inputs["x"].name] = values[step][None, None, :]
        feed[inputs["mask"].name] = _causal_step_mask(steps, step)
        for layer in range(spec.num_layers):
            keys, vals = cache.gather(rid, layer, pad_to=steps)
            feed[inputs[f"l{layer}_k"].name] = keys[None]
            feed[inputs[f"l{layer}_v"].name] = vals[None]
        out = backend.execute(g, feed)
        outs.append(_host(backend, out["y"]).reshape(1, spec.d_model))
        keys_out = [
            _host(backend, out[f"l{layer}_k_out"])[0]
            for layer in range(spec.num_layers)
        ]
        values_out = [
            _host(backend, out[f"l{layer}_v_out"])[0]
            for layer in range(spec.num_layers)
        ]
        cache.append(rid, keys_out, values_out)
    return np.stack(outs, axis=1)
