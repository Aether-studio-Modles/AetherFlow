"""weight-only int8 量化的构图与离线工具。

这里没有任何新的数值语义：量化与反量化的定义都在 ``ir/ops.py``
（``quantize_i8`` / ``dequantize_i8`` / ``linear_i8``），参考实现在
``ir/ops.py`` 的 reference 函数里。本模块只做两件事：

1. **离线量化一份权重**（``quantize_weight``）：从 f32 的 ``(K, N)`` 权重算出
   逐输出通道的 scale 与 int8 权重。它直接调用算子的 reference 函数，
   因此"训练后导出"与"图里跑一遍"用的是同一份语义，不会出现两套舍入规则。
2. **把量化线性层表达成 IR**（``build_int8_linear``）：返回输入表与输出
   Value，形状约定与 ``models/mlp.py`` 一致——权重是 ``(in, out)``，
   不是 PyTorch 的 ``(out, in)``。

边界：这是 **weight-only** 量化。激活仍是 f32，GEMM 在 f32 上累加，
收益是权重占用与读带宽；精度损失来自权重的舍入，与激活无关。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..ir.graph import Graph, Value
from ..ir.ops import get_op

__all__ = [
    "QUANT_MAX",
    "quantize_weight",
    "dequantize_weight",
    "build_int8_linear",
]

#: 对称 int8 的正端点。负端点是 -127 而不是 -128——见 ``ir/ops.py`` 的说明。
QUANT_MAX = 127


def quantize_weight(weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """把 ``(K, N)`` 的 f32 权重按**输出通道**量化成 int8。

    逐输出通道（``axis=0``，即每个 ``N`` 一个 scale）比逐张量量化精度高得多，
    也是 serving 里最常见的形态：不同输出通道的量级差一个数量级时，
    全局 scale 会让小通道整片被舍成 0。

    全零的通道给 ``scale = 1.0``：零权重在任何 scale 下都量化为 0，
    但 ``scale = 0`` 会触发"非正 scale"拒绝，所以这里显式取 1。
    """
    w = np.asarray(weight)
    if w.ndim != 2:
        raise ValueError(f"quantize_weight 只接受 (K, N) 的线性层权重，得到 {w.shape}")
    if w.dtype.kind != "f":
        raise TypeError(f"quantize_weight 需要浮点权重，得到 {w.dtype}")
    if not np.all(np.isfinite(w)):
        raise ValueError("权重里有 NaN / Inf：量化会把这些值传播到整条推理链")

    amax = np.max(np.abs(w), axis=0)
    # amax / 127 让最大绝对值恰好落在正端点，negative 端点到 -127 为止。
    scale = np.where(amax > 0, amax / float(QUANT_MAX), 1.0).astype(np.float32)
    q = get_op("quantize_i8").reference([w.astype(np.float32), scale], {})
    return q, scale


def dequantize_weight(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """``(K, N)`` 的 int8 权重还原成 f32（与 ``dequantize_i8`` 同一份语义）。"""
    return get_op("dequantize_i8").reference([np.asarray(q), np.asarray(scale)], {})


def build_int8_linear(
    g: Graph,
    in_features: int,
    out_features: int,
    batch: str = "B",
    bias: bool = True,
) -> Tuple[Dict[str, Value], Value]:
    """在 ``g`` 上构一个 weight-only int8 线性层。

    返回 ``(输入表, 输出 Value)``；输入表含 ``x`` / ``w_i8`` / ``scale``
    （``bias=True`` 时还有 ``bias``）。参数量是 ``K*N`` 个 int8 加 ``N`` 个
    f32——与 f32 权重相比，权重占用降到约四分之一。
    """
    from ..ir.types import f32, i8

    k, n = int(in_features), int(out_features)
    if k <= 0 or n <= 0:
        raise ValueError(f"in/out features 必须为正，得到 {k} 与 {n}")

    inputs: Dict[str, Value] = {
        "x": g.input("x", f32[batch, k]),
        "w_i8": g.input("w_i8", i8[k, n]),
        "scale": g.input("scale", f32[n]),
    }
    args = [inputs["x"], inputs["w_i8"], inputs["scale"]]
    if bias:
        inputs["bias"] = g.input("bias", f32[n])
        args.append(inputs["bias"])
    return inputs, g.add_node("linear_i8", args)
