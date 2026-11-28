"""设备后端的共享层：算子面声明、索引参数约定、纯函数 helper 与内核句柄。

这些定义不依赖任何实例状态，被 ``device_backend`` 与各算子族 mixin 共同引用。
拆出来是为了让 mixin 文件能导入它们，而不必反向导入 ``device_backend`` 造成循环。
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Tuple


from ..ir.graph import Graph, Node
from ..ir.types import DType
from ..lower.elementwise import (
    CHAIN_OPS,
    channel_broadcast,
)
from ..lower.kernels import BINARY_OP
from ..lower.plan import (
    CONV_CB_MAX,
    CONV_ROC_MAX,
    CONV_TILE_OPTIONS,
)
from .base import BackendCapabilityError


#: fp8 在 IR 里的载体也是 f32（``np_dtype``）；1 字节容器是 **i8 位模式**
#: （``pack_fp8`` / ``unpack_fp8``，见 5.328），不是独立的 fp8 dtype。
#: ``cast`` 由 ``_CAST_FP8_SRC`` 在设备上把数值舍入到格式格点（见 5.325）；
#: 承接 fp8 的算子只有 ``cast``、两个线性层与 pack/unpack（5.327 / 5.328），
#: 其余算子遇到 fp8 仍必须显式拒绝——否则会被 f32 载体悄悄当成 f32 计算。
#: 放在共享层是因为两条路径都要用它：逐节点的 ``invoke`` 与计划构建期的
#: ``_planned_kernel``（后者预绑定内核后就不再经过 ``invoke`` 的检查）。
FP8_DTYPES: FrozenSet[DType] = frozenset({DType.F8E4M3, DType.F8E5M2})

#: 允许输入 / 输出带 fp8 标量的算子。fp8 的载体就是 f32，所以它们不需要
#: 新内核，只需要让 ``invoke`` 与 ``_planned_kernel`` 不再按"fp8 一律拒绝"
#: 处理。刻意写成白名单而不是"排除几个"：新算子默认仍然不承接 fp8。
#: ``linear_fp8_packed`` 不在这里：它的权重是 i8 位模式，图里根本没有
#: fp8 标量（位模式即存储形态）。
FP8_OPS: FrozenSet[str] = frozenset(
    {
        "cast",
        "linear_fp8",
        "linear_fp8a8",
        "pack_fp8",
        "unpack_fp8",
    }
)


# 同一批自研内核模板（NVRTC / hipRTC 各编译一份）承接的算子面。它是声明，
# 与当前接入了哪家硬件无关；各后端在自己的文件里把它取成自己的名字。
# 存在的唯一理由：**不建任何设备上下文也能读到**。
GPU_OPS: FrozenSet[str] = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "where",
        "relu",
        "silu",
        "exp",
        "log",
        "sqrt",
        "neg",
        "tanh",
        "sigmoid",
        "erf",
        "step",
        "spike",
        "sign",
        "round",
        "floor",
        "ceil",
        "constant",
        "broadcast_to",
        "logical_and",
        "logical_or",
        "logical_not",
        "matmul",
        "quantize_i8",
        "quantize_i8_zp",
        "dequantize_i8",
        "dequantize_i8_zp",
        "dequantize_i8_group",
        "dequantize_i8_group_zp",
        "fake_quantize_i8",
        "fake_quantize_i8_lsq",
        "linear_i8",
        "linear_i8_group",
        "linear_i8_zp",
        "linear_i8a8",
        "linear_i8a8_zp",
        # fp8 线性层不新增内核：载体是 f32，反量化就是一次 ``mul``，
        # 之后复用 sgemm（见 DESIGN 5.327）。声明在这里是因为能力面
        # 与注册表必须逐项对账。
        "linear_fp8",
        "linear_fp8a8",
        # 5.328 的 1 字节存储：pack / unpack 是逐元素位模式搬运，
        # ``linear_fp8_packed`` 直接读 i8 字节、在寄存器里解码。
        "pack_fp8",
        "unpack_fp8",
        "linear_fp8_packed",
        # 5.342：逐组 scale 的反量化 / 线性层，以及两端都打包的 W8A8。
        "dequantize_fp8_group",
        "linear_fp8_group",
        "linear_fp8a8_packed",
        "conv2d",
        "maxpool2d",
        "maxpool2d_indices",
        "softmax",
        "attention",
        "layernorm",
        "reduce_sum",
        "reduce_max",
        "reduce_mean",
        # 5.341：bool 归约，按"数真值个数"求和后判定，输出仍是 bool。
        "reduce_any",
        "reduce_all",
        "argmax",
        "topk_values",
        "topk_indices",
        "topk_mask",
        "reshape",
        "transpose",
        "cast",
        "slice",
        "gelu",
        "rmsnorm",
        "avgpool2d",
        "avgpool2d_backward",
        "adaptive_avg_pool2d",
        "adaptive_avg_pool2d_backward",
        "concat",
        "embedding",
        "embedding_backward",
        "gather",
        "gather_backward",
        "cross_entropy",
        "cross_entropy_backward",
        "pad",
        "maxpool2d_backward",
        "conv2d_input_grad",
        "conv2d_weight_grad",
        "dropout_mask",
        "batch_norm",
        "batch_norm_stats",
        "scan_linear",
        "scan_linear_backward",
        "cumsum",
        "tril",
        "clamp",
        "mod",
        "sparse_linear",
        "sparse_linear_weight_grad",
        "sparse_linear_values_grad",
        "spike_linear",
        "spike_linear_weight_grad",
        "spike_offsets",
        "spike_csr",
    }
)

# 这些算子的某个参数是**整数输入**（下标或随机 key），不是浮点激活值。dtype
# 的通用检查必须给它们开口子，否则 embedding 会在进入内核之前就被"只支持浮点"
# 挡下来，dropout_mask 的运行期 key 也一样。
_INDEX_ARGS: Dict[str, FrozenSet[int]] = {
    "embedding": frozenset({1}),
    "embedding_backward": frozenset({1}),
    "gather": frozenset({1}),
    "gather_backward": frozenset({1}),
    "cross_entropy": frozenset({1}),
    "cross_entropy_backward": frozenset({1}),
    "dropout_mask": frozenset({1}),
    "sparse_linear": frozenset({1, 3}),
    "sparse_linear_weight_grad": frozenset({1, 3}),
    "sparse_linear_values_grad": frozenset({2, 3}),
    "spike_linear": frozenset({2, 3}),
    "spike_linear_weight_grad": frozenset({2, 3}),
    "spike_csr": frozenset({1}),
    # 量化：i8 是**数据**而不是下标，但它同样不能走"只支持浮点"的默认检查。
    "quantize_i8_zp": frozenset({2}),
    "dequantize_i8": frozenset({0}),
    "dequantize_i8_zp": frozenset({0, 2}),
    # 逐组反量化：q 与 L2 的 zero_point 是 i8 **数据**；scale 仍是 f32。
    "dequantize_i8_group": frozenset({0}),
    "dequantize_i8_group_zp": frozenset({0, 2}),
    "linear_i8": frozenset({1}),
    # 逐组版只有权重是 1 字节；scale 是二维 f32，仍按 dtype 推断字节数。
    "linear_i8_group": frozenset({1}),
    # W8A8：激活与权重都是 i8 数据，scale / bias 仍是 f32。
    "linear_i8a8": frozenset({0, 2}),
    # 非对称版：多出来的是两处 zero_point（都是 i8 数据）。
    "linear_i8_zp": frozenset({1, 3}),
    "linear_i8a8_zp": frozenset({0, 2, 3, 5}),
    # 1 字节 fp8：pack 的输入是 f32 载体，unpack 的输入与 packed 线性层
    # 的权重都是 i8 位模式（不是整数数据，但同样不能走"只支持浮点"）。
    "unpack_fp8": frozenset({0}),
    "linear_fp8_packed": frozenset({1}),
    # 5.342：逐组反量化的 q 是 i8 位模式；逐组线性层与 W8A8 packed 的
    # 激活 / 权重同样是 1 字节码，scale 是 f32。
    "dequantize_fp8_group": frozenset({0}),
    "linear_fp8_group": frozenset({1}),
    "linear_fp8a8_packed": frozenset({0, 2}),
    # 整数取模：两侧都是**整数数据**，不是下标。
    "mod": frozenset({0, 1}),
}

# 整数下标允许的 dtype。与 ir/ops.py 的 infer 保持一致：i32 是前端导入的
# 常态，i8 是压缩标签 / 量化权重的形态。
_INDEX_DTYPES = ("int32", "int8")

# 允许 bool 输入的算子。比较算子两侧都可以是 bool（``ne(mask, 0)`` 这类
# "掩码再取反"的写法很常见），它们的设备存储是 1 字节 ``unsigned char``，
# 走独立的 ``BOOL_DIALECT``。逻辑算子是纯 bool 算子；``cast`` 是两个方向
# 都合法的显式桥（f32 -> bool 就是"非零即真"，bool -> f32 是 0/1）。
# 算术算子不在此列：true + true = 2 再归一成 1
# 没有意义，IR 的 infer 与 verifier 都已拒绝。
_BOOL_OPS: FrozenSet[str] = frozenset(
    {
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "where",
        "logical_and",
        "logical_or",
        "logical_not",
        # 5.341：bool 归约输入输出都是 bool（1 字节存储，见 BOOL_DIALECT）。
        "reduce_any",
        "reduce_all",
        "cast",
    }
)

# 带 ``layout`` 属性的算子。5.334 起设备端对这三个算子声明了 NHWC：
# conv2d 只有分块直算承接（其它实现入口显式拒绝），池化只有前向承接
# （反向与取值位置仍按 NCHW 寻址）。
_LAYOUT_OPS: FrozenSet[str] = frozenset({"conv2d", "maxpool2d", "avgpool2d"})

# 量化只有两个内核：读权重、写权重。两份源码同时喂给 NVRTC 与 hipRTC——
# "半偶舍入 + 端点饱和"是语义的一部分，不能让两个后端各写一次近似。
# ``per_row`` 为 1 时按 ``s[i % cols]`` 取与输出通道对应的 scale，避免把
# 逐通道 scale 物化成与权重同大的 f32。
_QUANTIZE_I8_SRC = r"""
extern "C" __global__ void af_quantize_i8(
    const float* __restrict__ w,
    const float* __restrict__ s,
    signed char* __restrict__ out,
    long long n, long long cols, long long per_row) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float sv = per_row ? s[i % cols] : s[i];
  float q = rintf(w[i] / sv);
  q = q > 127.0f ? 127.0f : (q < -127.0f ? -127.0f : q);
  out[i] = (signed char)q;
}
"""

_DEQUANTIZE_I8_SRC = r"""
extern "C" __global__ void af_dequantize_i8(
    const signed char* __restrict__ q,
    const float* __restrict__ s,
    float* __restrict__ out,
    long long n, long long cols, long long per_row) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float sv = per_row ? s[i % cols] : s[i];
  out[i] = (float)q[i] * sv;
}
"""

# 非对称版本：零点作为 i8 张量进图，**先舍入到格点再平移**——零点不是值域里
# 的浮点偏移，而是整数格点的平移，两种写法在 x/scale 非整数时差一个舍入。
# 饱和两端都少用一格对称性：负端点取 -128，把 256 级全部用上。
_QUANTIZE_I8_ZP_SRC = r"""
extern "C" __global__ void af_quantize_i8_zp(
    const float* __restrict__ x,
    const float* __restrict__ s,
    const signed char* __restrict__ zp,
    signed char* __restrict__ out,
    long long n, long long cols, long long per_row) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long long si = per_row ? (i % cols) : i;
  float q = rintf(x[i] / s[si]) + (float)zp[si];
  q = q > 127.0f ? 127.0f : (q < -128.0f ? -128.0f : q);
  out[i] = (signed char)q;
}
"""

# 非对称反量化：减法留在 int32 里做（-128 减 127 是 -255，超出 i8 范围），
# 与参考实现的 ``(int32)q - (int32)zp`` 顺序一致，因此逐位相同。
_DEQUANTIZE_I8_ZP_SRC = r"""
extern "C" __global__ void af_dequantize_i8_zp(
    const signed char* __restrict__ q,
    const float* __restrict__ s,
    const signed char* __restrict__ zp,
    float* __restrict__ out,
    long long n, long long cols, long long per_row) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long long si = per_row ? (i % cols) : i;
  out[i] = (float)((int)q[i] - (int)zp[si]) * s[si];
}
"""

# 逐组（per-group）weight-only 反量化：q 是 (K, N)，scale 是 (K/G, N)。
# 组大小 G 不另设 attr，由 ``K / scale.shape[0]`` 推出（见 DESIGN 5.317）；
# 因此内核只需要知道列数 N 与组行数 G，下标按 ``row / G`` 折回 scale 的行。
# 组内共用一份 scale，和 GPTQ / AWQ 的 group_size 形态一致。
_DEQUANTIZE_I8_GROUP_SRC = r"""
extern "C" __global__ void af_dequantize_i8_group(
    const signed char* __restrict__ q,
    const float* __restrict__ s,
    float* __restrict__ out,
    long long n, long long cols, long long group) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long long row = i / cols;
  long long col = i - row * cols;
  long long si = (row / group) * cols + col;
  out[i] = (float)q[i] * s[si];
}
"""

# 非对称版本：减法留在 int32 里做，再乘 scale——与参考实现的
# ``(int32)q - (int32)zp`` 顺序一致，逐位相同。zero_point 与 scale 同形。
_DEQUANTIZE_I8_GROUP_ZP_SRC = r"""
extern "C" __global__ void af_dequantize_i8_group_zp(
    const signed char* __restrict__ q,
    const float* __restrict__ s,
    const signed char* __restrict__ zp,
    float* __restrict__ out,
    long long n, long long cols, long long group) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long long row = i / cols;
  long long col = i - row * cols;
  long long si = (row / group) * cols + col;
  out[i] = (float)((int)q[i] - (int)zp[si]) * s[si];
}
"""

# fp8 的编解码：f32 载体 ↔ fp8 格点。算法逐位复刻 runtime/fp8.py 的
# ``pack_fp8`` / ``unpack_fp8``：double 精度下的 RNE、e4m3 饱和到 ±448（无
# inf）、e5m2 溢出到 ±inf。全部用 double，是为了和主机侧 numpy float64 的
# 参考实现逐位一致；``fabs / isinf`` 都靠 f32 位模式判断，避免依赖设备端
# libm 的符号。
#
# 5.325 的 ``cast`` 只做"落到格点上"（输出仍是 f32、不省带宽）；5.328 的
# pack / unpack 与融合线性层共用同一份编解码，区别只在落盘宽度（1 字节）。
_FP8_COMMON_SRC = r"""
__device__ __forceinline__ unsigned int af_fp8_u32(float f) {
  union { float f; unsigned int u; } v;
  v.f = f;
  return v.u;
}

__device__ __forceinline__ float af_fp8_f32(unsigned int u) {
  union { float f; unsigned int u; } v;
  v.u = u;
  return v.f;
}

__device__ __forceinline__ double af_fp8_pow2(int e) {
  union { double d; unsigned long long u; } v;
  v.u = (unsigned long long)(e + 1023) << 52;
  return v.d;
}

__device__ __forceinline__ int af_fp8_ilog2(double x) {
  union { double d; unsigned long long u; } v;
  v.d = x;
  return (int)((v.u >> 52) & 0x7ffull) - 1023;
}

__device__ __forceinline__ double af_fp8_nan() {
  union { double d; unsigned long long u; } v;
  v.u = 0x7ff8000000000000ull;
  return v.d;
}

__device__ __forceinline__ double af_fp8_inf() {
  union { double d; unsigned long long u; } v;
  v.u = 0x7ff0000000000000ull;
  return v.d;
}

/* Round-half-to-even for non-negative doubles; call-site quotients are below
   32, so the integer cast is safe. */
__device__ __forceinline__ double af_fp8_rint(double x) {
  unsigned long long q = (unsigned long long)x;
  double frac = x - (double)q;
  if (frac > 0.5) return (double)(q + 1ull);
  if (frac < 0.5) return (double)q;
  return (double)(q + (q & 1ull));
}

/* 把 f32 载体编码成 fp8 字节（含符号位）。规则与 runtime/fp8.py 的
   ``pack_fp8`` 逐位一致：double 下的两次 RNE、e4m3 饱和到 ±448（无 inf）、
   e5m2 溢出到 inf。 */
__device__ __forceinline__ int af_fp8_encode(float f, int kind) {
  int M = (kind == 1) ? 3 : 2;
  int exp_bits = (kind == 1) ? 4 : 5;
  int bias = (kind == 1) ? 7 : 15;
  int e_min = (kind == 1) ? -6 : -14;
  int e_max = (kind == 1) ? 8 : 15;
  double max_value = (kind == 1) ? 448.0 : 57344.0;
  int saturate = (kind == 1);
  int exp_all = (1 << exp_bits) - 1;
  int nan_code = 0x7f;
  int inf_code = (kind == 1) ? 0x7e : (exp_all << M);

  unsigned int bits = af_fp8_u32(f);
  int negative = (int)(bits >> 31);
  double exact = (double)af_fp8_f32(bits & 0x7fffffffu);

  /* runtime/fp8.py's _round_abs: round twice inside the binade, and redo the
     carry with the new exponent when it crosses a binade. 0 / inf / NaN pass
     through unchanged. */
  int finite = (exact > 0.0) && (exact < 1e308);
  double safe = finite ? exact : 1.0;
  int e = af_fp8_ilog2(safe);
  if (e < e_min) e = e_min;
  double ulp = af_fp8_pow2(e - M);
  double first = af_fp8_rint(safe / ulp) * ulp;
  int e2 = af_fp8_ilog2(first > 0.0 ? first : 1.0);
  if (e2 < e_min) e2 = e_min;
  double ulp2 = af_fp8_pow2(e2 - M);
  double value = finite ? af_fp8_rint(safe / ulp2) * ulp2 : exact;

  /* _encode: saturate, then subnormal / normal, then special-value overrides. */
  double work = value;
  if (saturate && ((work > 1e308) || (work > max_value))) work = max_value;
  int finite_pos = (work > 0.0) && (work < 1e308);
  double wsafe = finite_pos ? work : 1.0;
  int exponent = af_fp8_ilog2(wsafe);

  int sub = finite_pos && (exponent < e_min);
  double sub_raw = sub ? af_fp8_rint(work * af_fp8_pow2(M - e_min)) : 0.0;
  int sub_carry = sub && (sub_raw >= (double)(1 << M));
  int sub_code = sub ? (int)sub_raw : 0;

  int normal = finite_pos && !sub;
  int normal_exp = exponent < e_min ? e_min : exponent;
  double raw = normal ? af_fp8_rint(wsafe / af_fp8_pow2(normal_exp - M)) : 0.0;
  int carry = normal && (raw >= (double)(1 << (M + 1)));
  int final_exp = carry ? normal_exp + 1 : normal_exp;
  int mant = normal ? (carry ? 0 : ((int)raw - (1 << M))) : 0;
  int overflow = normal && (final_exp > e_max);
  int normal_code = overflow ? inf_code : (((final_exp + bias) << M) | mant);

  int code = normal ? normal_code : 0;
  if (sub && !sub_carry) code = sub_code;
  if (sub_carry) code = (1 << M);
  if (!finite_pos) code = 0;
  if (value != value) code = nan_code;
  else if (value > 1e308) code = inf_code;
  if (value == 0.0) code = 0;

  return negative ? (code | 0x80) : code;
}

/* 解码：fp8 字节 -> f32 载体。每个编码恰好对应一个 f32 值（含 ±inf / NaN），
   与 runtime/fp8.py 的 ``unpack_fp8`` 逐位一致。 */
__device__ __forceinline__ float af_fp8_decode(int byte, int kind) {
  int M = (kind == 1) ? 3 : 2;
  int bias = (kind == 1) ? 7 : 15;
  int e_min = (kind == 1) ? -6 : -14;
  int exp_all = (kind == 1) ? 15 : 31;

  int negative = (byte >> 7) & 1;
  int code = byte & 0x7f;
  int ue = (code >> M) & exp_all;
  int um = code & ((1 << M) - 1);
  double uval;
  if (ue == 0) {
    uval = (double)um * af_fp8_pow2(e_min - M);
  } else {
    uval = af_fp8_pow2(ue - bias) * (1.0 + (double)um / (double)(1 << M));
  }
  int all_one = (ue == exp_all);
  int u_nan = all_one && ((kind == 1) ? (um == (1 << M) - 1) : (um != 0));
  int u_inf = all_one && (um == 0) && (kind != 1);
  if (u_nan) uval = af_fp8_nan();
  else if (u_inf) uval = af_fp8_inf();
  float result = (float)uval;
  return negative ? -result : result;
}

"""

# 5.325 的语义：f32 载体 -> 落到格式格点上的 f32 载体（不省带宽）。
_CAST_FP8_SRC = _FP8_COMMON_SRC + r"""
extern "C" __global__ void af_cast_fp8(
    const float* __restrict__ in,
    float* __restrict__ out,
    long long n, int kind) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = af_fp8_decode(af_fp8_encode(in[i], kind), kind);
}
"""

# 5.328 的 1 字节容器：fp8 本身存成 i8 字节（IR 没有 u8，``signed char``
# 正好 8 位）。两个方向和 ``cast`` 共用同一份编解码，所以 pack/unpack 往返
# 恒等，且 pack 的数值与 ``cast`` 相同——只是把结果落成 1 字节。
_PACK_FP8_SRC = _FP8_COMMON_SRC + r"""
extern "C" __global__ void af_pack_fp8(
    const float* __restrict__ in,
    signed char* __restrict__ out,
    long long n, int kind) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = (signed char)(af_fp8_encode(in[i], kind) & 0xff);
}
"""

_UNPACK_FP8_SRC = _FP8_COMMON_SRC + r"""
extern "C" __global__ void af_unpack_fp8(
    const signed char* __restrict__ in,
    float* __restrict__ out,
    long long n, int kind) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = af_fp8_decode((int)(unsigned char)in[i], kind);
}
"""

# weight-only fp8 的融合内核：权重按 1 字节读出、在寄存器里解码再乘 scale，
# 不物化 f32 权重。结构照 ``_LINEAR_I8A8_SRC``（每线程一个输出元素、无分块），
# 交付的是"权重驻留与读带宽降到 1/4"，**不是**算力：没有 fp8 tensor core，
# 累加仍是 f32。``ws_stride`` 为 0 表示逐张量 scale（scale 只有 1 个）。
_LINEAR_FP8_PACKED_SRC = _FP8_COMMON_SRC + r"""
extern "C" __global__ void af_linear_fp8_packed(
    const float* __restrict__ x,
    const signed char* __restrict__ w,
    const float* __restrict__ ws,
    float* __restrict__ out,
    long long m, long long k, long long n, long long ws_stride, int kind) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= m * n) return;
  long long row = idx / n;
  long long col = idx - row * n;
  const float* xr = x + row * k;
  const signed char* wc = w + col;
  float s = ws_stride ? ws[col] : ws[0];
  float acc = 0.0f;
  for (long long kk = 0; kk < k; ++kk) {
    float wv = af_fp8_decode((int)(unsigned char)wc[kk * n], kind);
    acc += xr[kk] * (wv * s);
  }
  out[idx] = acc;
}
"""

# 逐组 fp8 反量化（5.342）：组内共用一份 scale，结构照
# ``_DEQUANTIZE_I8_GROUP_SRC``，只多一步寄存器里的解码。乘法顺序是
# ``decode(q) * s``，与参考实现的 ``unpack_fp8(q) * repeat(scale)`` 逐位一致。
_DEQUANTIZE_FP8_GROUP_SRC = _FP8_COMMON_SRC + r"""
extern "C" __global__ void af_dequantize_fp8_group(
    const signed char* __restrict__ q,
    const float* __restrict__ s,
    float* __restrict__ out,
    long long n, long long cols, long long group, int kind) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long long row = i / cols;
  long long col = i - row * cols;
  long long si = (row / group) * cols + col;
  out[i] = af_fp8_decode((int)(unsigned char)q[i], kind) * s[si];
}
"""

# 两端都打包的 W8A8（5.342）：激活与权重各按 1 字节读出、在寄存器里解码，
# 累加仍是 f32（没有 fp8 tensor core）。``xs_stride`` / ``ws_stride`` 为 0
# 表示逐张量 scale；参考语义是 ``matmul(unpack(x)*xs, unpack(w)*ws)``，这里
# 按同样的乘法顺序逐项累加，差异只来自累加顺序（BLAS 分块 vs 顺序）。
_LINEAR_FP8A8_PACKED_SRC = _FP8_COMMON_SRC + r"""
extern "C" __global__ void af_linear_fp8a8_packed(
    const signed char* __restrict__ x,
    const float* __restrict__ xs,
    const signed char* __restrict__ w,
    const float* __restrict__ ws,
    float* __restrict__ out,
    long long m, long long k, long long n,
    long long xs_stride, long long ws_stride, int kind) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= m * n) return;
  long long row = idx / n;
  long long col = idx - row * n;
  const signed char* xr = x + row * k;
  const signed char* wc = w + col;
  float a = xs_stride ? xs[row] : xs[0];
  float b = ws_stride ? ws[col] : ws[0];
  float acc = 0.0f;
  for (long long kk = 0; kk < k; ++kk) {
    float xv = af_fp8_decode((int)(unsigned char)xr[kk], kind);
    float wv = af_fp8_decode((int)(unsigned char)wc[kk * n], kind);
    acc += (xv * a) * (wv * b);
  }
  out[idx] = acc;
}
"""

# W8A8 的矩阵乘：i8 输入、int32 累加、按行 / 按列缩放。累加必须留在整数域，
# 这正是它和 ``linear_i8``（反量化后走 f32 GEMM）的区别——后者的累加在浮点。
# 两份源码只差“怎么算 4 路点积”：CUDA 走 ``__dp4a``（sm_61+，V100 是 70），
# HIP 走标量乘加。两者都是精确整数运算，结果逐位相同，所以跨后端对拍成立。
_LINEAR_I8A8_SRC = r"""
extern "C" __global__ void af_linear_i8a8(
    const signed char* __restrict__ x,
    const float* __restrict__ xs,
    const signed char* __restrict__ w,
    const float* __restrict__ ws,
    float* __restrict__ out,
    long long m, long long k, long long n) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long total = m * n;
  if (idx >= total) return;
  long long row = idx / n;
  long long col = idx - row * n;
  const signed char* xr = x + row * k;
  const signed char* wc = w + col;
  int acc = 0;
  for (long long kk = 0; kk < k; ++kk) {
    acc += (int)xr[kk] * (int)wc[kk * n];
  }
  out[idx] = (float)acc * xs[row] * ws[col];
}
"""

_LINEAR_I8A8_DP4A_SRC = r"""
extern "C" __global__ void af_linear_i8a8(
    const signed char* __restrict__ x,
    const float* __restrict__ xs,
    const signed char* __restrict__ w,
    const float* __restrict__ ws,
    float* __restrict__ out,
    long long m, long long k, long long n) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long total = m * n;
  if (idx >= total) return;
  long long row = idx / n;
  long long col = idx - row * n;
  const signed char* xr = x + row * k;
  const signed char* wc = w + col;
  int acc = 0;
  long long kk = 0;
  for (; kk + 4 <= k; kk += 4) {
    const signed char* xp = xr + kk;
    const signed char* wp = wc + kk * n;
    int xv = (int)((unsigned int)(unsigned char)xp[0]
                 | ((unsigned int)(unsigned char)xp[1] << 8)
                 | ((unsigned int)(unsigned char)xp[2] << 16)
                 | ((unsigned int)(unsigned char)xp[3] << 24));
    int wv = (int)((unsigned int)(unsigned char)wp[0]
                 | ((unsigned int)(unsigned char)wp[n] << 8)
                 | ((unsigned int)(unsigned char)wp[2 * n] << 16)
                 | ((unsigned int)(unsigned char)wp[3 * n] << 24));
    acc = __dp4a(xv, wv, acc);
  }
  for (; kk < k; ++kk) acc += (int)xr[kk] * (int)wc[kk * n];
  out[idx] = (float)acc * xs[row] * ws[col];
}
"""


# 非对称 W8A8：每个乘子先减自己的零点再相乘，累加仍在 int32，最后乘两路
# scale。**不能**先按对称版算 acc 再在 f32 里补偏置——展开后的
# ``-zx*colsum - zw*rowsum + K*zx*zw`` 需要额外的归约，且与逐项相减的整数
# 累加不等价（参考实现 5.315 就是这样定的）。零点减法会打散字节对齐，所以
# 这一版不走 ``__dp4a``，用标量乘加；两条后端共用同一份源码。
_LINEAR_I8A8_ZP_SRC = r"""
// 逐项减零点的整数累加。抽成 device 函数只为一件事：无 bias / 带 bias
// 两个入口必须共用同一份累加体，否则"融合 epilogue"会悄悄变成第二条语义。
__device__ __forceinline__ int af_i8a8_zp_acc(
    const signed char* __restrict__ xr,
    const signed char* __restrict__ wc,
    int zx, int zw, long long k, long long n) {
  int acc = 0;
  for (long long kk = 0; kk < k; ++kk) {
    acc += ((int)xr[kk] - zx) * ((int)wc[kk * n] - zw);
  }
  return acc;
}

extern "C" __global__ void af_linear_i8a8_zp(
    const signed char* __restrict__ x,
    const signed char* __restrict__ xzp,
    const float* __restrict__ xs,
    const signed char* __restrict__ w,
    const signed char* __restrict__ wzp,
    const float* __restrict__ ws,
    float* __restrict__ out,
    long long m, long long k, long long n) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long total = m * n;
  if (idx >= total) return;
  long long row = idx / n;
  long long col = idx - row * n;
  const signed char* xr = x + row * k;
  const signed char* wc = w + col;
  int acc = af_i8a8_zp_acc(xr, wc, (int)xzp[row], (int)wzp[col], k, n);
  out[idx] = (float)acc * xs[row] * ws[col];
}

// 带 bias 的入口：把 ``add`` 融进来，省一次 kernel launch 与一块 m*n 中间张量。
// 两次乘法用 ``__fmul_rn``、加法用 ``__fadd_rn``：不融合时 bias 是在另一颗
// 内核里做 f32 加法，各自按 RN 舍入；若这里让编译器发 FFMA，末位会差 1 ulp，
// "融合前后逐位相同"就不成立了。
extern "C" __global__ void af_linear_i8a8_zp_bias(
    const signed char* __restrict__ x,
    const signed char* __restrict__ xzp,
    const float* __restrict__ xs,
    const signed char* __restrict__ w,
    const signed char* __restrict__ wzp,
    const float* __restrict__ ws,
    const float* __restrict__ bias,
    float* __restrict__ out,
    long long m, long long k, long long n) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long total = m * n;
  if (idx >= total) return;
  long long row = idx / n;
  long long col = idx - row * n;
  const signed char* xr = x + row * k;
  const signed char* wc = w + col;
  int acc = af_i8a8_zp_acc(xr, wc, (int)xzp[row], (int)wzp[col], k, n);
  float v = __fmul_rn((float)acc, xs[row]);
  v = __fmul_rn(v, ws[col]);
  out[idx] = __fadd_rn(v, bias[col]);
}
"""


def _step_params(node: Node) -> tuple:
    """逐元素链一步的编译期常量参数。

    - ``clamp``：``(min, max)``，缺省一侧为 ``None``；
    - ``step`` / ``spike``：``(threshold,)``，默认值分别与 IR 的参考实现一致
      （0.0 / 1.0，见 ``ir/ops.py``）。

    IR 的 infer 已经保证 clamp 至少一侧存在且 ``min <= max``，所以这里不再
    重复校验；参数会进源码缓存键，边界/阈值不同的两个节点不会共用内核。
    """
    name = node.op.name
    if name == "clamp":
        return (node.attrs.get("min"), node.attrs.get("max"))
    if name == "step":
        return (node.attrs.get("threshold", 0.0),)
    if name == "spike":
        return (node.attrs.get("threshold", 1.0),)
    return ()


def _chainable(node: Node) -> bool:
    """这个节点能不能作为逐元素链的一环（不看它与邻居的关系）。"""
    name = node.op.name
    if name not in CHAIN_OPS:
        return False
    arity = 2 if name in BINARY_OP else 1
    if len(node.inputs) != arity:
        return False
    out_type = node.output.type
    if out_type.dtype not in (DType.F32, DType.F16, DType.BF16):
        return False
    for value in node.inputs:
        t = value.type
        if t.dtype != out_type.dtype:
            return False
        if t.shape == out_type.shape:
            continue
        # 只有"恰好一个元素"的一侧可以当标量；符号维无法静态确认大小
        if t.is_static:
            size = 1
            for d in t.shape:
                size *= d
            if size == 1:
                continue
            # 逐通道广播 (1,C,1,1) / (C,1,1)：BN 的常态，索引可由静态形状
            # 完全确定，因此可以融（见 lower.elementwise.channel_broadcast）。
            if out_type.is_static and channel_broadcast(t.shape, out_type.shape):
                continue
        return False
    return True


def _group_elementwise(graph: Graph) -> List[tuple]:
    """把相邻的同形逐元素算子合并成一组（纯静态计划，不碰设备）。

    判据全部是**静态**的，任何一条不满足就退回"一算子一内核"：

    - 算子必须有一元/二元表达式表（``CHAIN_OPS``），且参数个数与表一致
      （``silu`` 在门控形态下是二元的，语义是 ``silu(x) * gate``，不能当成
      一元链的一步）；
    - 相邻两节点之间那条边只被链内下一个节点使用，且不是图输出——否则
      中间值必须物化，融合就改变了可观测行为；
    - 形状与 dtype 在 IR 上相同（符号维按名字比较）；大小为 1 的一侧可以
      当标量，其余广播形态不融。

    长度 1 的组不合并：单算子仍走原来的专用内核，源码与数值行为逐字不变。
    运行期还会再查一次（是否连续、指针对齐、长度能否被 4 整除），见
    ``DeviceBackend._try_chain``；静态计划乐观不会算错，最多白算一次。
    """
    nodes = graph.nodes
    users: Dict[int, List[int]] = {}
    for idx, node in enumerate(nodes):
        for a in node.inputs:
            users.setdefault(a.id, []).append(idx)
    out_ids = {v.id for v in graph.outputs}

    groups: List[tuple] = []
    i = 0
    while i < len(nodes):
        j = i
        while j + 1 < len(nodes) and _chainable(nodes[j]) and _chainable(nodes[j + 1]):
            mid = nodes[j].output
            if mid.id in out_ids or users.get(mid.id) != [j + 1]:
                break
            j += 1
        groups.append(tuple(nodes[i : j + 1]))
        i = j + 1
    return groups


def _pair(value) -> tuple:
    """IR 属性里的 int 或二元组（JSON 往返后是 list）统一成 (h, w)。"""
    if isinstance(value, int):
        return (value, value)
    return (int(value[0]), int(value[1]))


def _pool_out_dim(size: int, kernel: int, stride: int, padding: int, ceil_mode: bool) -> int:
    """池化单维输出长度，与 ``ir/ops.py`` 的 ``_pool_out_dim`` 逐条一致。

    ``ceil_mode`` 的两步（向上取整 + 去掉起点越界的多余窗口）必须和 IR 层同步：
    这里算出的 ``HO`` / ``WO`` 决定内核的网格，差一格就是越界写或漏算。
    """
    if ceil_mode:
        out = -((int(size) + 2 * padding - kernel) // -stride) + 1
        if (out - 1) * stride >= int(size) + padding:
            out -= 1
    else:
        out = (int(size) + 2 * padding - kernel) // stride + 1
    return out


def _pool_flags(attrs: dict, op: str) -> tuple:
    """池化的布尔属性 ``(ceil_mode, count_include_pad)``；类型不对就显式报错。"""
    ceil_mode = attrs.get("ceil_mode", False)
    if not isinstance(ceil_mode, bool):
        raise BackendCapabilityError(f"{op} 的 ceil_mode 必须是 bool，收到 {ceil_mode!r}")
    count_include_pad = attrs.get("count_include_pad", True)
    if not isinstance(count_include_pad, bool):
        raise BackendCapabilityError(
            f"{op} 的 count_include_pad 必须是 bool，收到 {count_include_pad!r}"
        )
    return ceil_mode, count_include_pad


def _mask_shared_over_heads(mask_shape, s_len: int, t_len: int) -> bool:
    """``mask`` 能否被所有 ``(b, h)`` 共用一份 ``(S, T)``。

    条件是最后两维正好等于 ``(S, T)``、其余维全为 1——因果 mask 的
    ``(1,1,S,T)`` 与直接的 ``(S,T)`` 都落在这里。任何随 batch 或 head 变化的
    mask 仍要广播，不能共用。
    """
    shape = tuple(int(dim) for dim in mask_shape)
    return shape[-2:] == (int(s_len), int(t_len)) and all(dim == 1 for dim in shape[:-2])


def _mask_target_shape(mask_shape, keep: int) -> Optional[Tuple[int, ...]]:
    """去掉 ``mask`` 多余前导维后的目标形状；前导维不全是 1 时返回 ``None``。

    只做形状推导，不碰数组本身：调用方据此 ``reshape`` 降维。之所以不用
    ``mask[0]``，是因为天数（Iluvatar）的 ``DeviceArray.__getitem__`` 只接受
    切片、拒绝降维索引，整数索引会抛 ``TypeError``；在 numpy / cupy 上
    ``reshape`` 与降维索引等价，语义因此不分叉。
    """
    shape = tuple(int(dim) for dim in mask_shape)
    extra = len(shape) - keep
    if extra <= 0:
        return shape
    if any(dim != 1 for dim in shape[:extra]):
        return None
    return shape[extra:]


# conv2d 分块核在 W 方向的线程数、shared 预算与分块阈值都在 ``lower/plan.py``：
# 它们是与设备无关的纯算术，放在后端里既挡住了离线单测，也容易和 HIP 侧分叉。

_CONV_IMPLS = ("auto", "im2col", "tiled", "implicit", "winograd")

# ``sparse_linear`` 的实现选择。基类只给 "custom" 与 "auto"——"cusparse"
# 属于 NVIDIA 的库，由 ``CudaBackend`` 扩展；HIP 侧构造期就会拒绝它，
# 而不是等到第一次稀疏乘才发现。默认 "custom" 而不是 "auto"：自研核带
# 下标越界检查，库路径没有；默认值不能悄悄换掉一道检查（见 5.203）。
_SPARSE_IMPLS = ("custom", "auto")

# 逐元素类：每个线程只处理一个元素，block 大小只影响占用率与合并访存，
# 不改变 shared memory 需求，也不改变数值语义。
_ELEMENTWISE_TUNABLE_OPS: FrozenSet[str] = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "where",
        "relu",
        "silu",
        "exp",
        "log",
        "sqrt",
        "neg",
        "tanh",
        "sigmoid",
        "erf",
        "gelu",
        "step",
        "spike",
        "dropout_mask",
    }
)

# 归约类：一个 block 负责一行（或一个通道），块内归约的暂存区是固定的
# ``REDUCE_SCRATCH``（1024 个 float = 4 KB），**不随 block 变化**；内核沿
# ``blockDim.x`` 跨步读输入，所以 block 只改变归约树深度与占用率。改 block
# 因此既不改变 shared 预算，也不改变数值语义（块内求和顺序会变，但归约本身
# 就是浮点非结合的操作，逐位一致只在同一 block 下成立）。
#
# 此前这批被排除在搜索之外，理由是"shared 与 blockDim 绑定"；5.160 核对内核
# 后确认该前提不成立（``Shared("float", "smem", REDUCE_SCRATCH)`` 是常量），
# 于是开放给调优面。``batch_norm`` 的两个内核共用同一套骨架，一起放开。
_REDUCE_TUNABLE_OPS: FrozenSet[str] = frozenset(
    {
        "softmax",
        "layernorm",
        "rmsnorm",
        "reduce_sum",
        "reduce_max",
        "reduce_mean",
        "argmax",
        "topk_values",
        "topk_indices",
        "topk_mask",
        "batch_norm",
        "batch_norm_stats",
    }
)

_TUNABLE_OPS: FrozenSet[str] = _ELEMENTWISE_TUNABLE_OPS | _REDUCE_TUNABLE_OPS

# conv2d 的旋钮不是 block，而是分块核的线程铺法，见 ``_CONV_KNOBS``。
_CONV_TILE_OP = "conv2d"

# 分块核的另两个旋钮的候选值：2 的幂，到计划层的上限为止（5.16 把上限本身
# 定成了最优值，所以候选没有理由越过它）。
_CONV_ROC_OPTIONS = tuple(v for v in (1, 2, 4) if v <= CONV_ROC_MAX)
_CONV_CB_OPTIONS = tuple(v for v in (1, 2, 4, 8) if v <= CONV_CB_MAX)

# 参数名 -> (实例字段, 候选值)。``None`` 是"按形状自动推导"，它随形状变化、
# 钉不成常数，所以永远在候选里；当前值排第一位，即 ``TuningSpace`` 的默认值。
_CONV_KNOBS: Dict[str, Tuple[str, Tuple[int, ...]]] = {
    "tile_w": ("_conv_tile", CONV_TILE_OPTIONS),
    "roc": ("_conv_roc", _CONV_ROC_OPTIONS),
    "cb": ("_conv_cb", _CONV_CB_OPTIONS),
}

# "有没有旋钮"的总集合。逐元素类拧 block，conv2d 拧分块核的线程铺法，
# 具体有哪些参数由 ``tuning_space`` 回答。
_ALL_TUNABLE_OPS: FrozenSet[str] = _TUNABLE_OPS | frozenset({_CONV_TILE_OP})

# ``execute`` 的图执行方式：off = 逐节点下发（默认）；auto = 首次遇到一组
# 形状时捕获成整图，之后只 replay。默认保持 off，因为捕获与 replay 的语义
# 差异（随机掩码、主机同步）需要使用者知情；基准与训练循环可显式打开。
_GRAPH_MODES = ("off", "auto")

# 执行期内存策略。``container`` 是历史行为（容器自己的池）；``tiered`` 由
# 后端提供分层分配器，见 ``DeviceBackend._memory_scope``。
_MEMORY_MODES = ("container", "tiered")


class _KernelHandle:
    """``kernel()`` 的稳定返回面：把后端各自的启动约定收成一个 ``launch``。

    后端内部仍直接用各自的句柄（CUDA 是 cupy 的 ``Function``，靠 ``__call__``
    启动；HIP 是模块函数，靠 ``launch``）。公开入口如果不收口，"打印/编译内核"
    这类跨后端工具就得写两套启动代码——5.168 在 CUDA 真机上正是因为测试按
    HIP 的形态断言 ``handle.launch`` 而失败，才发现两边公开面并不一致。
    """

    __slots__ = ("_fn",)

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def launch(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<kernel handle {self._fn!r}>"
