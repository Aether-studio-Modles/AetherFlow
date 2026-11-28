"""int8 / fp8 量化的构图与离线工具。

这里没有任何新的数值语义：量化与反量化的定义都在 ``ir/ops.py``
（``quantize_i8`` / ``dequantize_i8`` / ``fake_quantize_i8`` / ``linear_i8`` /
``linear_i8a8``，以及带零点的 ``*_zp`` 族），参考实现在同一处的 reference
函数里。fp8 的数值语义（含带 scale 的量化）在 ``runtime/fp8.py``。本模块
只做三件事：

1. **离线量化一份权重**（``quantize_weight`` / ``quantize_weight_asymmetric``）：
   从 f32 的 ``(K, N)`` 权重算出逐输出通道的量化参数与 int8 权重。它直接调用
   算子的 reference 函数，因此"训练后导出"与"图里跑一遍"用的是同一份语义，
   不会出现两套舍入规则。
2. **按 token 量化一份激活**（``quantize_activation`` /
   ``quantize_activation_asymmetric``）：沿最后一维取 amax（对称）或
   min/max（非对称），得到每个 token 一组量化参数；与权重共用同一份 RNE 舍入。
3. **把量化线性层表达成 IR**：``build_int8_linear`` 是 weight-only，
   ``build_int8a8_linear`` 是激活也量化的 W8A8；带 ``_zp`` 的两个 builder 是
   对应的非对称版本；``build_int8_group_linear`` 是逐组 scale 的 weight-only，
   ``build_int8a8_linear_qat`` 是训练期的假量化 STE / LSQ，
   ``build_int8a8_linear_dynamic`` 在图内按 token 现算激活 scale。权重形状
   约定与 ``models/mlp.py`` 一致——权重是 ``(in, out)``，不是 PyTorch 的
   ``(out, in)``。

fp8 的一节只覆盖**写侧**（把 f32 权重 / 激活压到 fp8 格点并给出 scale）：
``quantize_weight_fp8`` / ``quantize_activation_fp8`` / ``dequantize_weight_fp8``。
IR 里的 fp8 张量是 f32 载体（``DType.F8E4M3`` 的 ``np_dtype`` 就是 float32），
所以"量化"产出的是缩放后的 f32 数组，值全部落在 fp8 格点上；配套的 scale
与 int8 的逐通道 / 逐 token 口径一致，可直接用于 HF 风格的
``weight_scale`` / ``input_scale`` 约定。

两套语义的差别写清楚：

- **weight-only**（``linear_i8`` / ``linear_i8_zp``）：权重反量化后在 f32 上做
  GEMM。收益是权重占用与读带宽，精度损失只来自权重的舍入。
- **W8A8**（``linear_i8a8`` / ``linear_i8a8_zp``）：int8 激活与 int8 权重先在
  int32 里整数累加，再乘两个 scale 的积。它才对应 int8 矩阵乘的累加语义；
  激活量化多引入一份舍入误差，收益是算力与读带宽。

对称与非对称的差别：

- 对称把 0 钉在 0 上（``-127..127``），适合近对称的权重。
- 非对称用满 ``-128..127`` 两级，0 不必落在格点上（``q = rint(x/s) + zp``），
  主要给"全为正"的激活（如 ReLU 之后）用：对称量化会把一半格子分给永远不会
  出现的负数。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..ir.graph import Graph, Value
from ..ir.ops import get_op
from ..runtime.fp8 import FP8_KINDS, FP8_MAX, dequantize_fp8, quantize_fp8

__all__ = [
    "QUANT_MAX",
    "QUANT_MIN_ASYMMETRIC",
    "FP8_KINDS",
    "quantize_weight",
    "dequantize_weight",
    "quantize_activation",
    "quantize_weight_asymmetric",
    "dequantize_weight_asymmetric",
    "quantize_activation_asymmetric",
    "build_int8_linear",
    "build_int8a8_linear",
    "build_int8_linear_zp",
    "build_int8a8_linear_zp",
    "quantize_weight_group",
    "quantize_weight_int4_group",
    "build_int8_group_linear",
    "build_int8a8_linear_qat",
    "build_int8a8_linear_dynamic",
    "quantize_weight_fp8",
    "quantize_weight_fp8_group",
    "dequantize_weight_fp8",
    "quantize_activation_fp8",
]

#: 对称 int8 的正端点。负端点是 -127 而不是 -128——见 ``ir/ops.py`` 的说明。
QUANT_MAX = 127

#: 非对称量化把整段 256 级都用满，负端点是 -128。
QUANT_MIN_ASYMMETRIC = -128


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


def _asymmetric_params(
    lo: np.ndarray, hi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """由每组 ``(min, max)`` 算 ``(scale, zero_point)``：``lo`` 落到 -128。

    值域用满整段 256 级（``scale = (hi - lo) / 255``），零点按
    ``zp = rint(-128 - lo / scale)`` 反解，再夹回 i8。

    **值域先扩到包含 0**：零点必须落在 ``[-128, 127]``，而
    ``zp = -128 - lo/scale`` 只有在 ``lo <= 0 <= hi`` 时才不越界。全正的数据
    （ReLU 之后的激活）若直接用 ``lo > 0``，零点会小于 -128 被夹死，正端
    则整片饱和——比对称量化更差。扩到 0 之后，ReLU 激活的 ``lo'=0``、
    ``zp=-128``，一格不浪费；这是"0 必须能被精确表示"这条约束的直接推论。

    若 ``hi == lo``（扩展后只剩常数 ``c`` 或纯 0），没有值域可用，退化成
    ``scale = 1``：此时该组只能表示 0，与对称量化把常数通道压成 0 是同一种
    取舍。
    """
    lo64 = np.minimum(lo, 0.0).astype(np.float64)
    hi64 = np.maximum(hi, 0.0).astype(np.float64)
    span = hi64 - lo64
    scale64 = np.where(span > 0.0, span / 255.0, 1.0)
    zp = np.rint(float(QUANT_MIN_ASYMMETRIC) - lo64 / scale64)
    return (
        scale64.astype(np.float32),
        np.clip(zp, QUANT_MIN_ASYMMETRIC, QUANT_MAX).astype(np.int8),
    )


def quantize_weight_asymmetric(
    weight: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 ``(K, N)`` 的 f32 权重按**输出通道**做非对称量化。

    返回 ``(q, scale, zero_point)``：每组量化参数形状都是 ``(N,)``。
    校验与 :func:`quantize_weight` 完全一致（二维、浮点、无 NaN/Inf），
    差别只在参数计算：这里用每通道的 min/max 而不是 abs-max。
    """
    w = np.asarray(weight)
    if w.ndim != 2:
        raise ValueError(
            f"quantize_weight_asymmetric 只接受 (K, N) 的线性层权重，得到 {w.shape}"
        )
    if w.dtype.kind != "f":
        raise TypeError(
            f"quantize_weight_asymmetric 需要浮点权重，得到 {w.dtype}"
        )
    if not np.all(np.isfinite(w)):
        raise ValueError("权重里有 NaN / Inf：量化会把这些值传播到整条推理链")

    scale, zp = _asymmetric_params(np.min(w, axis=0), np.max(w, axis=0))
    q = get_op("quantize_i8_zp").reference([w.astype(np.float32), scale, zp], {})
    return q, scale, zp


def dequantize_weight_asymmetric(
    q: np.ndarray, scale: np.ndarray, zero_point: np.ndarray
) -> np.ndarray:
    """``(K, N)`` 的 int8 权重按非对称口径还原成 f32。"""
    return get_op("dequantize_i8_zp").reference(
        [np.asarray(q), np.asarray(scale), np.asarray(zero_point)], {}
    )


def quantize_activation_asymmetric(
    x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 f32 激活按 **token**（最后一维）做非对称量化。

    返回 ``(q, scale, zero_point)``，后两者形状都是 ``x.shape[:-1] + (1,)``
    且都是逐 token 一组，可以直接喂给 :func:`build_int8a8_linear_zp`。
    激活全为正（ReLU 之后）时，非对称把负半轴让给实际值域，而对称量化
    会把一半格子浪费在不可能出现的负数上。
    """
    a = np.asarray(x)
    if a.ndim < 2:
        raise ValueError(f"quantize_activation_asymmetric 至少需要 2 维，得到 {a.shape}")
    if a.dtype.kind != "f":
        raise TypeError(f"quantize_activation_asymmetric 需要浮点激活，得到 {a.dtype}")
    if not np.all(np.isfinite(a)):
        raise ValueError("激活里有 NaN / Inf：量化会把这些值传播到整条推理链")

    lo = np.min(a, axis=-1, keepdims=True)
    hi = np.max(a, axis=-1, keepdims=True)
    scale, zp = _asymmetric_params(lo, hi)
    q = get_op("quantize_i8_zp").reference([a.astype(np.float32), scale, zp], {})
    return q, scale, zp


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


def quantize_activation(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """把 f32 激活按 **token**（最后一维）量化为 int8，返回 ``(q, scale)``。

    ``scale`` 的形状是 ``x.shape[:-1] + (1,)``：每个 token 一个 scale，
    与 ``quantize_i8`` 的广播规则一致，因此可以直接喂给
    :func:`build_int8a8_linear` 的 ``x_q`` / ``x_scale`` 输入。

    为什么是最后一维：``linear_i8a8`` 沿 ``K`` 做 int32 累加，scale 要能提到
    求和号外面；沿 K 逐元素变化的 scale 无法用整数累加表达（算子会显式拒绝）。

    全零的 token 给 ``scale = 1.0``，理由与 ``quantize_weight`` 相同。
    """
    a = np.asarray(x)
    if a.ndim < 2:
        raise ValueError(f"quantize_activation 至少需要 2 维，得到 {a.shape}")
    if a.dtype.kind != "f":
        raise TypeError(f"quantize_activation 需要浮点激活，得到 {a.dtype}")
    if not np.all(np.isfinite(a)):
        raise ValueError("激活里有 NaN / Inf：量化会把这些值传播到整条推理链")

    amax = np.max(np.abs(a), axis=-1, keepdims=True)
    scale = np.where(amax > 0, amax / float(QUANT_MAX), 1.0).astype(np.float32)
    q = get_op("quantize_i8").reference([a.astype(np.float32), scale], {})
    return q, scale


def build_int8a8_linear(
    g: Graph,
    in_features: int,
    out_features: int,
    batch: str = "B",
    bias: bool = True,
) -> Tuple[Dict[str, Value], Value]:
    """在 ``g`` 上构一个 W8A8 线性层（激活与权重都是 int8）。

    返回 ``(输入表, 输出 Value)``；输入表含 ``x_q`` / ``x_scale`` / ``w_q`` /
    ``w_scale``（``bias=True`` 时还有 ``bias``）。``x_scale`` 的形状是
    ``(batch, 1)``：每个 token 一个 scale；``w_scale`` 是 ``(N,)``：每个输出
    通道一个。

    数值上等价于 ``(int32(x_q) @ int32(w_q)).f32() * x_scale * w_scale [+ b]``，
    与 ``linear_i8`` 的"先反量化再 f32 GEMM"不是同一份累加。用
    :func:`quantize_activation` 与 :func:`quantize_weight` 准备输入。
    """
    from ..ir.types import f32, i8

    k, n = int(in_features), int(out_features)
    if k <= 0 or n <= 0:
        raise ValueError(f"in/out features 必须为正，得到 {k} 与 {n}")

    inputs: Dict[str, Value] = {
        "x_q": g.input("x_q", i8[batch, k]),
        "x_scale": g.input("x_scale", f32[batch, 1]),
        "w_q": g.input("w_q", i8[k, n]),
        "w_scale": g.input("w_scale", f32[n]),
    }
    args = [inputs["x_q"], inputs["x_scale"], inputs["w_q"], inputs["w_scale"]]
    if bias:
        inputs["bias"] = g.input("bias", f32[n])
        args.append(inputs["bias"])
    return inputs, g.add_node("linear_i8a8", args)


def build_int8_linear_zp(
    g: Graph,
    in_features: int,
    out_features: int,
    batch: str = "B",
    bias: bool = True,
) -> Tuple[Dict[str, Value], Value]:
    """在 ``g`` 上构一个**非对称** weight-only int8 线性层。

    返回 ``(输入表, 输出 Value)``；输入表含 ``x`` / ``w_i8`` / ``scale`` /
    ``zero_point``（``bias=True`` 时还有 ``bias``）。``scale`` 与
    ``zero_point`` 的形状都是 ``(N,)``：每个输出通道一组。用
    :func:`quantize_weight_asymmetric` 准备权重侧的两个参数。
    """
    from ..ir.types import f32, i8

    k, n = int(in_features), int(out_features)
    if k <= 0 or n <= 0:
        raise ValueError(f"in/out features 必须为正，得到 {k} 与 {n}")

    inputs: Dict[str, Value] = {
        "x": g.input("x", f32[batch, k]),
        "w_i8": g.input("w_i8", i8[k, n]),
        "scale": g.input("scale", f32[n]),
        "zero_point": g.input("zero_point", i8[n]),
    }
    args = [
        inputs["x"],
        inputs["w_i8"],
        inputs["scale"],
        inputs["zero_point"],
    ]
    if bias:
        inputs["bias"] = g.input("bias", f32[n])
        args.append(inputs["bias"])
    return inputs, g.add_node("linear_i8_zp", args)


def build_int8a8_linear_zp(
    g: Graph,
    in_features: int,
    out_features: int,
    batch: str = "B",
    bias: bool = True,
) -> Tuple[Dict[str, Value], Value]:
    """在 ``g`` 上构一个**非对称** W8A8 线性层（激活与权重都带零点）。

    返回 ``(输入表, 输出 Value)``；输入表含 ``x_q`` / ``x_scale`` /
    ``x_zero_point`` / ``w_q`` / ``w_scale`` / ``w_zero_point``（``bias=True``
    时还有 ``bias``）。激活侧的量化参数形状是 ``(batch, 1)``（每个 token
    一组），权重侧是 ``(N,)``（每个输出通道一组）。

    数值上等价于
    ``((x_q - x_zp) @ (w_q - w_zp)).f32() * x_scale * w_scale [+ b]``，
    减法与累加都在 int32 里完成。用 :func:`quantize_activation_asymmetric`
    与 :func:`quantize_weight_asymmetric` 准备输入。
    """
    from ..ir.types import f32, i8

    k, n = int(in_features), int(out_features)
    if k <= 0 or n <= 0:
        raise ValueError(f"in/out features 必须为正，得到 {k} 与 {n}")

    inputs: Dict[str, Value] = {
        "x_q": g.input("x_q", i8[batch, k]),
        "x_scale": g.input("x_scale", f32[batch, 1]),
        "x_zero_point": g.input("x_zero_point", i8[batch, 1]),
        "w_q": g.input("w_q", i8[k, n]),
        "w_scale": g.input("w_scale", f32[n]),
        "w_zero_point": g.input("w_zero_point", i8[n]),
    }
    args = [
        inputs["x_q"],
        inputs["x_scale"],
        inputs["x_zero_point"],
        inputs["w_q"],
        inputs["w_scale"],
        inputs["w_zero_point"],
    ]
    if bias:
        inputs["bias"] = g.input("bias", f32[n])
        args.append(inputs["bias"])
    return inputs, g.add_node("linear_i8a8_zp", args)


def quantize_weight_group(
    weight: np.ndarray, group_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """把 ``(K, N)`` 的 f32 权重按 **K 方向每 ``group_size`` 行一组**量化。

    返回 ``(q, scale)``：``q`` 与权重同形，``scale`` 是
    ``(K // group_size, N)``。这是 GPTQ / AWQ 的 ``group_size`` 形态——
    比逐通道多出 ``K/G`` 倍的 f32 量化参数，换来更细的动态范围：组内共用
    一份 scale，某一行的离群值不会把同通道其它行整片压小。

    组大小必须整除 ``K``；全零的组给 ``scale = 1.0``（理由同
    :func:`quantize_weight`）。量化本身仍走 ``quantize_i8`` 的参考实现，
    因此与图里"再乘回去"共享同一份 RNE 与饱和规则。
    """
    w = np.asarray(weight)
    if w.ndim != 2:
        raise ValueError(
            f"quantize_weight_group 只接受 (K, N) 的线性层权重，得到 {w.shape}"
        )
    if w.dtype.kind != "f":
        raise TypeError(f"quantize_weight_group 需要浮点权重，得到 {w.dtype}")
    if not np.all(np.isfinite(w)):
        raise ValueError("权重里有 NaN / Inf：量化会把这些值传播到整条推理链")

    gs = int(group_size)
    if gs <= 0:
        raise ValueError(f"group_size 必须为正，得到 {group_size}")
    k, n = w.shape
    if k % gs != 0:
        raise ValueError(f"group_size={gs} 不能整除 K={k}：逐组量化要求整组切分")

    grouped = w.astype(np.float32).reshape(k // gs, gs, n)
    amax = np.max(np.abs(grouped), axis=1)
    scale = np.where(amax > 0, amax / float(QUANT_MAX), 1.0).astype(np.float32)
    # (K/G, 1, N) 与 (K/G, G, N) 广播：组内共用同一份 scale。
    q = get_op("quantize_i8").reference([grouped, scale[:, None, :]], {})
    return q.reshape(k, n), scale


def quantize_weight_int4_group(
    weight: np.ndarray, group_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 ``(K, N)`` 的 f32 权重按组量化成 **4 bit**（AWQ / GPTQ 的 4 bit 形态）。

    返回 ``(q, scale, zero_point)``：``q`` 与权重同形、值域 ``0..15``（仍由
    ``i8`` 承载，可直接交给 ``pack_int4`` 压成一半存储）；``scale`` /
    ``zero_point`` 都是 ``(K // group_size, N)``、按组共用。

    口径与 ``dequantize_u4_group`` 一致：``w ≈ (q - zp) * scale``。组内先取
    min/max 并**扩到包含 0**（理由同 :func:`_asymmetric_params`：4 bit 的零点
    必须落在 ``[0, 15]``，全正的组若不舍入到 0 附近会让负端格子永远用不上），
    再令 ``scale = (hi - lo) / 15``、``zp = clip(rint(-lo / scale), 0, 15)``。
    全零组给 ``scale = 1, zp = 0``：零在任意 scale 下都量化为 zp，但
    ``scale = 0`` 会触发"非正 scale"拒绝。

    舍入用 RNE（``np.rint``，与其它量化入口同一口径），非饱和点的误差不超过
    ``scale / 2``；两端点落在 ``lo`` / ``hi`` 本身，误差同样不超过半步。
    """
    w = np.asarray(weight)
    if w.ndim != 2:
        raise ValueError(
            f"quantize_weight_int4_group 只接受 (K, N) 的线性层权重，得到 {w.shape}"
        )
    if w.dtype.kind != "f":
        raise TypeError(f"quantize_weight_int4_group 需要浮点权重，得到 {w.dtype}")
    if not np.all(np.isfinite(w)):
        raise ValueError("权重里有 NaN / Inf：量化会把这些值传播到整条推理链")

    gs = int(group_size)
    if gs <= 0:
        raise ValueError(f"group_size 必须为正，得到 {group_size}")
    k, n = w.shape
    if k % gs != 0:
        raise ValueError(f"group_size={gs} 不能整除 K={k}：逐组量化要求整组切分")

    grouped = w.astype(np.float32).reshape(k // gs, gs, n)
    lo = np.minimum(np.min(grouped, axis=1), 0.0)
    hi = np.maximum(np.max(grouped, axis=1), 0.0)
    span = (hi - lo).astype(np.float32)
    scale = np.where(span > 0, span / 15.0, 1.0).astype(np.float32)
    zero = np.clip(np.rint(-lo / scale), 0.0, 15.0).astype(np.float32)
    q = np.rint(grouped / scale[:, None, :] + zero[:, None, :])
    q = np.clip(q, 0.0, 15.0).astype(np.int8)
    return q.reshape(k, n), scale, zero


def build_int8_group_linear(
    g: Graph,
    in_features: int,
    out_features: int,
    group_size: int,
    batch: str = "B",
    bias: bool = True,
) -> Tuple[Dict[str, Value], Value]:
    """在 ``g`` 上构一个**逐组** weight-only int8 线性层。

    返回 ``(输入表, 输出 Value)``；输入表含 ``x`` / ``w_i8`` / ``scale``
    （形状 ``(K // group_size, N)``）与可选的 ``bias``。权重占用与
    :func:`build_int8_linear` 相同（``K*N`` 个 int8），多出来的是
    ``K/G * N`` 个 f32 的量化参数；换来的是更细的量化粒度。

    用 :func:`quantize_weight_group` 准备 ``w_i8`` 与 ``scale``。
    """
    from ..ir.types import f32, i8

    k, n = int(in_features), int(out_features)
    if k <= 0 or n <= 0:
        raise ValueError(f"in/out features 必须为正，得到 {k} 与 {n}")
    gs = int(group_size)
    if gs <= 0 or k % gs != 0:
        raise ValueError(
            f"group_size={gs} 必须为正且整除 in_features={k}，否则组切分不完整"
        )

    inputs: Dict[str, Value] = {
        "x": g.input("x", f32[batch, k]),
        "w_i8": g.input("w_i8", i8[k, n]),
        "scale": g.input("scale", f32[k // gs, n]),
    }
    args = [inputs["x"], inputs["w_i8"], inputs["scale"]]
    if bias:
        inputs["bias"] = g.input("bias", f32[n])
        args.append(inputs["bias"])
    return inputs, g.add_node("linear_i8_group", args)


def build_int8a8_linear_qat(
    g: Graph,
    in_features: int,
    out_features: int,
    batch: str = "B",
    bias: bool = True,
    learn_scale: bool = False,
) -> Tuple[Dict[str, Value], Value]:
    """在 ``g`` 上构一个 **QAT** 的 W8A8 线性层（假量化 + STE）。

    输入表含 ``x`` / ``x_scale`` / ``w`` / ``w_scale``（可选 ``bias``），
    全部是 f32。前向把两端假量化到 int8 格点后做 f32 GEMM——**图里没有 i8
    张量**：训练要对浮点参数求导，int8 码在反向里没有连续变量。

    ``learn_scale=True`` 时走 ``fake_quantize_i8_lsq``：反向把量化误差的
    归约和送给 scale，量化步长本身可学；否则是标准 STE，scale 冻结。
    导出的权重可以用 :func:`quantize_weight` / :func:`quantize_weight_group`
    转成推理期的 int8 图，前向格点与训练时逐位相同。
    """
    from ..ir.types import f32

    k, n = int(in_features), int(out_features)
    if k <= 0 or n <= 0:
        raise ValueError(f"in/out features 必须为正，得到 {k} 与 {n}")

    fake = "fake_quantize_i8_lsq" if learn_scale else "fake_quantize_i8"
    inputs: Dict[str, Value] = {
        "x": g.input("x", f32[batch, k]),
        "x_scale": g.input("x_scale", f32[batch, 1]),
        "w": g.input("w", f32[k, n]),
        "w_scale": g.input("w_scale", f32[n]),
    }
    xq = g.add_node(fake, [inputs["x"], inputs["x_scale"]])
    wq = g.add_node(fake, [inputs["w"], inputs["w_scale"]])
    out = g.add_node("matmul", [xq, wq])
    if bias:
        inputs["bias"] = g.input("bias", f32[n])
        out = g.add_node("add", [out, inputs["bias"]])
    return inputs, out


def build_int8a8_linear_dynamic(
    g: Graph,
    in_features: int,
    out_features: int,
    batch: str = "B",
    bias: bool = True,
) -> Tuple[Dict[str, Value], Value]:
    """在 ``g`` 上构一个 **per-token 动态 scale** 的 W8A8 线性层。

    输入表含 ``x`` / ``w_q`` / ``w_scale``（可选 ``bias``）：激活的 scale
    不再是图输入，而是在图里按每个 token 的 amax 现算
    （``amax / 127``，全零 token 给 1.0）。这是 serving 里 "dynamic
    per-token quantization" 的形态：输入分布变了也不需要重算激活 scale，
    代价是每个 token 多一次 ``reduce_max``。

    只由 ``reduce_max`` / ``where`` / ``gt`` / ``ge`` / ``neg`` / ``div`` /
    ``quantize_i8`` 这些既有算子组成，设备端只要承接它们就能跑，没有新增
    内核；量化格点与 :func:`quantize_activation` 完全同口径。
    """
    from ..ir.types import f32, i8

    k, n = int(in_features), int(out_features)
    if k <= 0 or n <= 0:
        raise ValueError(f"in/out features 必须为正，得到 {k} 与 {n}")

    inputs: Dict[str, Value] = {
        "x": g.input("x", f32[batch, k]),
        "w_q": g.input("w_q", i8[k, n]),
        "w_scale": g.input("w_scale", f32[n]),
    }
    zero = g.constant(0.0)
    one = g.constant(1.0)
    limit = g.constant(float(QUANT_MAX))
    # |x| = where(x >= 0, x, -x)
    absx = g.add_node(
        "where",
        [
            g.add_node("ge", [inputs["x"], zero]),
            inputs["x"],
            g.add_node("neg", [inputs["x"]]),
        ],
    )
    amax = g.add_node("reduce_max", [absx], {"axis": -1, "keepdims": True})
    x_scale = g.add_node(
        "where",
        [
            g.add_node("gt", [amax, zero]),
            g.add_node("div", [amax, limit]),
            one,
        ],
    )
    x_q = g.add_node("quantize_i8", [inputs["x"], x_scale])
    args = [x_q, x_scale, inputs["w_q"], inputs["w_scale"]]
    if bias:
        inputs["bias"] = g.input("bias", f32[n])
        args.append(inputs["bias"])
    return inputs, g.add_node("linear_i8a8", args)


# --------------------------------------------------------------------------
# fp8 写侧（e4m3 / e5m2）
# --------------------------------------------------------------------------
#
# 与 int8 的差别只有格点：fp8 的格点由格式决定（e4m3 最大 448、e5m2 最大
# 57344），量化参数仍然只有 scale——fp8 的 0 精确可表示，非对称量化没有收益，
# 所以这里既不提供 zero_point，也不接受它。
#
# scale 的口径刻意与 int8 一致：权重逐输出通道、激活逐 token。这样同一条
# linear 只需要替换"权重容器 + 算子"，缩放约定不用跟着换；HF 的 fp8 检查点
# 常用 per-tensor 的 ``weight_scale``，对应 ``per_tensor=True``。
#
# 全零的通道 / token 给 scale = 1.0，理由与 ``quantize_weight`` 相同：零在
# 任何 scale 下都量化为 0，但 scale = 0 会触发"非正 scale"拒绝。


def _fp8_limit(kind: str) -> float:
    if kind not in FP8_KINDS:
        raise ValueError(f"未知的 fp8 格式 {kind!r}；支持 {FP8_KINDS}")
    return float(FP8_MAX[kind])


def quantize_weight_fp8(
    weight: np.ndarray, kind: str = "e4m3", per_tensor: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """把 ``(K, N)`` 的 f32 权重压到 fp8 格点，返回 ``(w_f8, weight_scale)``。

    ``w_f8`` 与输入同形状、仍是 f32 载体（值全部落在 fp8 格点上，数量级在
    ``±max`` 内）；``weight_scale`` 是逐输出通道的 ``(N,)``，``per_tensor=True``
    时退化成 ``(1,)`` 的整张量一个 scale（HF 的 ``weight_scale`` 形态）。
    反量化是 ``w_f8 * weight_scale``（:func:`dequantize_weight_fp8`）。
    """
    limit = _fp8_limit(kind)
    w = np.asarray(weight)
    if w.ndim != 2:
        raise ValueError(f"quantize_weight_fp8 只接受 (K, N) 的线性层权重，得到 {w.shape}")
    if w.dtype.kind != "f":
        raise TypeError(f"quantize_weight_fp8 需要浮点权重，得到 {w.dtype}")
    if not np.all(np.isfinite(w)):
        raise ValueError("权重里有 NaN / Inf：量化会把这些值传播到整条推理链")

    if per_tensor:
        amax = np.array([np.max(np.abs(w))], dtype=np.float32)
    else:
        amax = np.max(np.abs(w), axis=0)
    scale = np.where(amax > 0, amax / limit, 1.0).astype(np.float32)
    return quantize_fp8(w.astype(np.float32), scale, kind), scale


def quantize_weight_fp8_group(
    weight: np.ndarray, group_size: int, kind: str = "e4m3"
) -> Tuple[np.ndarray, np.ndarray]:
    """把 ``(K, N)`` 权重按**输入维分组**压到 fp8 格点，返回 ``(w_f8, scale)``。

    ``scale`` 的形状是 ``(K/G, N)``：沿 K 每 ``group_size`` 行共用一份，
    与 GPTQ / AWQ 的 group_size 形态、以及 :class:`weights.FP8Weight` 的
    per-group 口径一致。``w_f8`` 仍是 f32 载体，值全部落在 fp8 格点上。

    分组能降低量化误差：一个离群值只抬高它所在那一组的 scale，不再拖累整
    列。代价是 scale 的驻留从 ``N`` 个 float 变成 ``K/G * N`` 个。
    """
    limit = _fp8_limit(kind)
    w = np.asarray(weight)
    if w.ndim != 2:
        raise ValueError(
            f"quantize_weight_fp8_group 只接受 (K, N) 的线性层权重，得到 {w.shape}"
        )
    if w.dtype.kind != "f":
        raise TypeError(f"quantize_weight_fp8_group 需要浮点权重，得到 {w.dtype}")
    if not np.all(np.isfinite(w)):
        raise ValueError("权重里有 NaN / Inf：量化会把这些值传播到整条推理链")

    k, n = w.shape
    gs = int(group_size)
    if gs <= 0:
        raise ValueError(f"group_size 必须是正整数，得到 {group_size}")
    if k % gs != 0:
        raise ValueError(f"group_size={gs} 不能整除 K={k}")

    amax = np.max(np.abs(w.reshape(k // gs, gs, n)), axis=1)
    scale = np.where(amax > 0, amax / limit, 1.0).astype(np.float32)
    repeated = np.repeat(scale, gs, axis=0)
    return quantize_fp8(w.astype(np.float32), repeated, kind), scale


def dequantize_weight_fp8(w: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """``(K, N)`` 的 f8 载体还原成真实数值：``w * scale``。"""
    return dequantize_fp8(w, scale)


def quantize_activation_fp8(
    x: np.ndarray, kind: str = "e4m3"
) -> Tuple[np.ndarray, np.ndarray]:
    """把 f32 激活按 **token**（最后一维）压到 fp8 格点，返回 ``(x_f8, scale)``。

    ``scale`` 的形状是 ``x.shape[:-1] + (1,)``：每个 token 一个 scale，与
    ``quantize_activation`` 的 int8 版本完全同口径。静态激活 scale 需要的是
    一个标量，可以自己做 ``x.reshape(-1, k).max()`` 后手算，或直接用
    :func:`quantize_fp8` 配自己的 scale。
    """
    limit = _fp8_limit(kind)
    a = np.asarray(x)
    if a.ndim < 2:
        raise ValueError(f"quantize_activation_fp8 至少需要 2 维，得到 {a.shape}")
    if a.dtype.kind != "f":
        raise TypeError(f"quantize_activation_fp8 需要浮点激活，得到 {a.dtype}")
    if not np.all(np.isfinite(a)):
        raise ValueError("激活里有 NaN / Inf：量化会把这些值传播到整条推理链")

    amax = np.max(np.abs(a), axis=-1, keepdims=True)
    scale = np.where(amax > 0, amax / limit, 1.0).astype(np.float32)
    return quantize_fp8(a.astype(np.float32), scale, kind), scale

