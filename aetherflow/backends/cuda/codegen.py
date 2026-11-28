"""CUDA C 源码生成。

这里生成的是**人类可读的 CUDA C**，不是 PTX，再交给 NVRTC 在运行时编译。
选择 NVRTC 而不是提前用 nvcc 编译的原因：不需要在目标机器上安装 CUDA toolkit，
插件只要拿到驱动即可工作。

按算子分别生成 f32 与 f16 两种存储类型。f16 的算术仍在 ``float`` 里完成
（累加、归约、指数都不降精度），只有存储宽度减半——否则误差会以"悄悄降级"
的形式出现，而不是以明确的实现边界出现。

模板里的占位符：``%PRELUDE%`` 由 ``render()`` 按 dtype 展开成"归约辅助 +
类型转换"，``%T%`` 是存储类型，``%LD%`` / ``%ST%`` 是"按存储类型读成 float /
把 float 写回存储类型"，``%ATOMIC%`` 是"把 float 累加进存储类型"的原子加。
"""

from __future__ import annotations

import numpy as np

from ...ir.types import DType
from ...lower import emit
from ...lower.kernel import Kernel
from ...lower.kernels import (
    adaptive_avg_pool2d,
    adaptive_avg_pool2d_backward,
    argmax_last,
    argmin_last,
    attention_fwd,
    avgpool2d,
    avgpool2d_backward,
    batch_norm,
    batch_norm_stats,
    concat_copy,
    conv2d_input_grad,
    conv2d_grouped,
    conv2d_implicit,
    conv2d_tiled,
    conv2d_weight_grad,
    cumsum,
    cumprod,
    dropout_mask,
    ew_binary,
    ew_broadcast,
    ew_cast,
    ew_clamp,
    interpolate_bilinear,
    interpolate_bilinear_backward,
    interpolate_nearest,
    interpolate_nearest_backward,
    ew_mod_int,
    ew_step,
    ew_unary,
    ew_where_bcast,
    ew_where_same,
    flip,
    im2col,
    layernorm_last,
    layernorm_residual_last,
    logsumexp_axis,
    maxpool2d,
    maxpool2d_backward,
    maxpool2d_indices,
    reduce_all_last,
    reduce_any_last,
    reduce_max_last,
    reduce_mean_last,
    reduce_sum_last,
    reduce_sum_rows,
    reduce_axis,
    reduce_axis_tiled,
    rmsnorm_last,
    roll,
    repeat_interleave,
    scan_linear,
    softmax_axis,
    scan_linear_backward,
    scan_linear_chunk,
    scan_linear_chunk_backward,
    scan_linear_chunk_last,
    scan_linear_chunk_sum,
    scan_linear_chunk_sum_backward,
    scan_linear_last,
    silu_gate,
    softmax_last,
    tril,
    spike_csr,
    spike_csr_serial,
    spike_linear,
    spike_linear_weight_grad,
    spike_offsets,
    sparse_linear,
    sparse_linear_values_grad,
    sparse_linear_weight_grad,
    winograd_input_transform,
    winograd_output_transform,
    winograd_weight_transform,
    winograd_fused,
)

_REDUCE = r"""
__device__ __forceinline__ float block_reduce_max(float v, float* smem) {
    smem[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float other = smem[threadIdx.x + s];
            smem[threadIdx.x] = smem[threadIdx.x] > other ? smem[threadIdx.x] : other;
        }
        __syncthreads();
    }
    float r = smem[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
    smem[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) { smem[threadIdx.x] += smem[threadIdx.x + s]; }
        __syncthreads();
    }
    float r = smem[0];
    __syncthreads();
    return r;
}

// Reduce over threadIdx.x for each fixed threadIdx.y: the mirror image of
// block_reduce_*_y. Used when the reduced dimension is narrow (d_state,
// n_scales, head_dim ...) so one warp-sized block per row wastes the machine:
// the block's y dimension packs many rows instead, and each row keeps the
// exact halving tree block_reduce_sum uses (bit-identical results).
__device__ __forceinline__ float block_reduce_sum_x(float v, float* smem) {
    const int nx = blockDim.x;
    const int base = threadIdx.y * nx;
    smem[base + threadIdx.x] = v;
    __syncthreads();
    for (int s = nx >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[base + threadIdx.x] += smem[base + threadIdx.x + s];
        }
        __syncthreads();
    }
    float r = smem[base];
    __syncthreads();
    return r;
}

// Two-dimensional block reductions: reduce over threadIdx.y for each fixed
// threadIdx.x. A strided (outer, reduce, inner) reduction reads consecutive
// inner elements with threadIdx.x, so the reduce axis has to be split across
// threadIdx.y; the flat block_reduce_* above cannot do that.
__device__ __forceinline__ float block_reduce_sum_y(float v, float* smem) {
    const int nx = blockDim.x;
    const int tid = threadIdx.y * nx + threadIdx.x;
    smem[tid] = v;
    __syncthreads();
    for (int s = blockDim.y >> 1; s > 0; s >>= 1) {
        if (threadIdx.y < s) { smem[tid] += smem[tid + s * nx]; }
        __syncthreads();
    }
    float r = smem[tid];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float block_reduce_max_y(float v, float* smem) {
    const int nx = blockDim.x;
    const int tid = threadIdx.y * nx + threadIdx.x;
    smem[tid] = v;
    __syncthreads();
    for (int s = blockDim.y >> 1; s > 0; s >>= 1) {
        if (threadIdx.y < s) {
            float other = smem[tid + s * nx];
            smem[tid] = smem[tid] > other ? smem[tid] : other;
        }
        __syncthreads();
    }
    float r = smem[tid];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float block_reduce_min_y(float v, float* smem) {
    const int nx = blockDim.x;
    const int tid = threadIdx.y * nx + threadIdx.x;
    smem[tid] = v;
    __syncthreads();
    for (int s = blockDim.y >> 1; s > 0; s >>= 1) {
        if (threadIdx.y < s) {
            float other = smem[tid + s * nx];
            smem[tid] = smem[tid] < other ? smem[tid] : other;
        }
        __syncthreads();
    }
    float r = smem[tid];
    __syncthreads();
    return r;
}

/* Exclusive prefix sum across the block: returns how many lanes before tid
   carried a non-zero value. The block total is left in smem[blockDim.x], one
   slot past the scan area, so the caller reads it after the trailing barrier.
   Used by spike_csr to write each row's columns in ascending order. */
__device__ __forceinline__ int block_scan_sum(int v, int* smem) {
    const int tid = threadIdx.x;
    const int n = blockDim.x;
    smem[tid] = v;
    __syncthreads();
    for (int off = 1; off < n; off <<= 1) {
        const int prev = (tid >= off) ? smem[tid - off] : 0;
        __syncthreads();
        smem[tid] += prev;
        __syncthreads();
    }
    const int total = smem[n - 1];
    const int pre = smem[tid] - v;
    if (tid == 0) { smem[n] = total; }
    __syncthreads();
    return pre;
}
"""

# 表达式级辅助函数：不是"按存储宽度读写"，而是内核表达式里直接调用的函数，
# 与 dtype 无关（实参与返回值都是 float）。**不**拼进下面三个 dtype 块，
# 而由使用方按"每个生成单元恰好一次"追加：``CUDA_DIALECT`` 的每个 dtype 各
# 追加一次，``_CAST_HELPERS`` / ``_BOOL_HELPERS`` 这类把三块拼在一起的组合
# 只追加一次。三块各自带一份会让组合方言里 ``af_rem`` 重复定义（NVRTC 报
# "has already been defined"）。
# NOTE: ASCII only -- this text goes to NVRTC.
_EXPR_HELPERS = r"""
// Floor-semantics modulo for floats. fmodf truncates, so its result takes the
// sign of the dividend; numpy.mod / torch.remainder take the sign of the
// divisor. Adding one divisor when the signs differ fixes that. NaN and a zero
// divisor propagate as-is (fmodf returns NaN), matching the IEEE result numpy
// produces.
__device__ __forceinline__ float af_rem(float a, float b) {
    float r = fmodf(a, b);
    if (r != 0.f && ((r < 0.f) != (b < 0.f))) { r += b; }
    return r;
}
"""

# f32 的四个辅助函数是恒等映射：模板只写一份，两种 dtype 共用同一份源码结构。
_F32_HELPERS = r"""
__device__ __forceinline__ float af_ld_f(float v) { return v; }
__device__ __forceinline__ float af_st_f(float v) { return v; }
__device__ __forceinline__ float af_add_f(float* p, float v) { return atomicAdd(p, v); }
"""

# f16 存储为 unsigned short。刻意不 include cuda_fp16.h：目标机可能只装了驱动
# 而没有 toolkit，NVRTC 的 include 路径不可靠；PTX 的 cvt 指令是所有 NVRTC
# 版本都认识的。算术一律读成 float 再算，只有存取时降回 f16。
# NOTE: the C source below is fed to NVRTC, which only accepts ASCII -- the
# comments inside the raw string must stay ASCII even though the Python
# comments around it are Chinese.
_F16_HELPERS = r"""
__device__ __forceinline__ float af_ld_h(unsigned short h) {
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h));
    return r;
}

__device__ __forceinline__ unsigned short af_st_h(float v) {
    unsigned short r;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(v));
    return r;
}

// 16-bit has no native atomicAdd; a CAS loop implements it without pulling in
// cuda_fp16.h. Contention only happens on overlapping windows or repeated ids.
// sm_52 (Maxwell) also lacks atomicCAS(unsigned short*), which appears only from
// sm_60 on: widen the CAS to the aligned 32-bit word and mask the target half.
// The compare value is built from ``assumed`` (not from a fresh load of the
// word) on purpose: the caller's retry loop adds ``v`` again on a mismatch, so a
// CAS that succeeded while returning a different value would double-count.
__device__ __forceinline__ unsigned short af_cas_h(unsigned short* p,
                                                   unsigned short assumed,
                                                   unsigned short v) {
#if __CUDA_ARCH__ >= 600
    return atomicCAS(p, assumed, v);
#else
    unsigned long long addr = (unsigned long long)p;
    unsigned int* base = (unsigned int*)(addr & ~2ULL);
    unsigned int shift = (addr & 2ULL) ? 16u : 0u;
    unsigned int mask = (addr & 2ULL) ? 0xFFFF0000u : 0x0000FFFFu;
    unsigned int other = *base & ~mask;
    for (;;) {
        unsigned int assumed32 = other | (((unsigned int)assumed) << shift);
        unsigned int desired = other | (((unsigned int)v) << shift);
        unsigned int prev = atomicCAS(base, assumed32, desired);
        if (prev == assumed32) return assumed;
        unsigned short got = (unsigned short)((prev & mask) >> shift);
        // Another half of the word changed: keep going. The target half moved:
        // report it so the caller recomputes from the fresh value.
        if (got != assumed) return got;
        other = prev & ~mask;
    }
#endif
}

__device__ __forceinline__ float af_add_h(unsigned short* p, float v) {
    unsigned short old = *p;
    unsigned short assumed;
    do {
        assumed = old;
        old = af_cas_h(p, assumed, af_st_h(af_ld_h(assumed) + v));
    } while (assumed != old);
    return af_ld_h(old);
}
"""

# bf16 同样存成 unsigned short，但存取是**纯位操作**：bf16 就是 f32 的高 16
# 位，加宽是左移、收窄是高半字的 RNE 舍入。刻意不用 PTX 的 cvt 指令，也不用
# cuda_bf16.h——cvt 的 bf16 形式要 sm_80，而按位实现从 Maxwell 起都能编；
# 舍入规则与主机侧 ``runtime.bf16.round_bf16`` 逐条一致（尾数 LSB 为 1 时进位，
# inf / nan 只截断，截断后尾数清零的 NaN 补 quiet 位）。
_BF16_HELPERS = r"""
__device__ __forceinline__ float af_ld_b(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

__device__ __forceinline__ unsigned short af_st_b(float v) {
    unsigned int bits = __float_as_uint(v);
    unsigned int lsb = (bits >> 16) & 1u;
    unsigned int kept = (bits + 0x7FFFu + lsb) >> 16;
    if ((bits & 0x7F800000u) == 0x7F800000u) {
        // inf / nan: adding the bias would carry into the exponent, or wrap a
        // mantissa of all ones into zero. They are already on the bf16 grid.
        kept = bits >> 16;
        if ((bits & 0x007FFFFFu) != 0 && (kept & 0x007Fu) == 0) {
            kept |= 0x0040u;
        }
    }
    return (unsigned short)kept;
}

__device__ __forceinline__ unsigned short af_cas_b(unsigned short* p,
                                                   unsigned short assumed,
                                                   unsigned short v) {
#if __CUDA_ARCH__ >= 600
    return atomicCAS(p, assumed, v);
#else
    unsigned long long addr = (unsigned long long)p;
    unsigned int* base = (unsigned int*)(addr & ~2ULL);
    unsigned int shift = (addr & 2ULL) ? 16u : 0u;
    unsigned int mask = (addr & 2ULL) ? 0xFFFF0000u : 0x0000FFFFu;
    unsigned int other = *base & ~mask;
    for (;;) {
        unsigned int assumed32 = other | (((unsigned int)assumed) << shift);
        unsigned int desired = other | (((unsigned int)v) << shift);
        unsigned int prev = atomicCAS(base, assumed32, desired);
        if (prev == assumed32) return assumed;
        unsigned short got = (unsigned short)((prev & mask) >> shift);
        if (got != assumed) return got;
        other = prev & ~mask;
    }
#endif
}

__device__ __forceinline__ float af_add_b(unsigned short* p, float v) {
    unsigned short old = *p;
    unsigned short assumed;
    do {
        assumed = old;
        old = af_cas_b(p, assumed, af_st_b(af_ld_b(assumed) + v));
    } while (assumed != old);
    return af_ld_b(old);
}
"""

# 占位符：render() 会把它替换成 _REDUCE + 对应 dtype 的辅助函数。
PRELUDE = "%PRELUDE%"

_STORAGE = {
    "f32": ("float", "af_ld_f", "af_st_f", "af_add_f"),
    "f16": ("unsigned short", "af_ld_h", "af_st_h", "af_add_h"),
    "bf16": ("unsigned short", "af_ld_b", "af_st_b", "af_add_b"),
    "bool": ("unsigned char", "af_ld_q", "af_st_q", "af_add_q"),
}

# boolean is stored as one byte with values in {0, 1}: reads widen to float and
# writes normalise any non-zero value back to 1. That lets the L2 layer emit a
# comparison result as an ordinary float expression. ``af_add_q`` is a stub --
# no kernel reduces into a bool -- it only keeps the four-slot storage tuple
# self-consistent.
_BOOL_HELPERS = r"""
__device__ __forceinline__ float af_ld_q(unsigned char v) { return v ? 1.f : 0.f; }
__device__ __forceinline__ unsigned char af_st_q(float v) { return v != 0.f ? 1 : 0; }
__device__ __forceinline__ float af_add_q(unsigned char* p, float v) { return 0.f; }
"""

_HELPERS = {
    "f32": _F32_HELPERS + _EXPR_HELPERS,
    "f16": _F16_HELPERS + _EXPR_HELPERS,
    "bf16": _BF16_HELPERS + _EXPR_HELPERS,
    "bool": _BOOL_HELPERS,
}


def _dtype_key(dtype) -> str:
    # ``np.dtype`` 实例有 ``.name``，标量**类型**（``cp.float32``）没有：
    # 它的 ``str`` 是 ``<class 'numpy.float32'>``。两种写法都要认，否则
    # 同一份调用换个写法就编不出内核。
    name = getattr(dtype, "name", None)
    if not isinstance(name, str):
        name = np.dtype(dtype).name
    if name in ("float32", "f32"):
        return "f32"
    if name in ("float16", "f16"):
        return "f16"
    # bf16 在设备上的容器是 uint16（位模式）；numpy 侧只可能是这两个写法。
    if name in ("uint16", "bfloat16", "bf16"):
        return "bf16"
    if name in ("bool", "bool_"):
        return "bool"
    raise ValueError(f"CUDA codegen 只支持 f32/f16/bf16/bool，收到 {name}")


def _ir_dtype(dtype) -> DType:
    """运行期的 numpy dtype -> IR 的 DType（L2 用后者认类型）。"""
    key = _dtype_key(dtype)
    return {
        "f32": DType.F32,
        "f16": DType.F16,
        "bf16": DType.BF16,
        "bool": DType.BOOL,
    }[key]


def render(template, dtype, *, f16_helpers: str | None = None, dialect=None) -> str:
    """把内核展开成可交给 NVRTC 的源码。

    接受两种输入：

    - ``str``：按 dtype 参数化的手写模板（历史路径，仍用于对照与回退）；
    - ``Kernel``：L2 描述（5.42 之后的生产路径），由 ``lower.emit`` 打印。

    ``f16_helpers`` / ``dialect`` 是给**别的后端复用同一套内核**留的方言注入点：
    HIP 的 clang 不认 PTX 的 ``cvt`` 约束，必须换掉 f16 的读写实现。
    """
    if isinstance(template, Kernel):
        if dialect is None:
            from .dialect import BOOL_DIALECT, CUDA_DIALECT

            # bool 的存储是 1 字节 ``unsigned char``，存取走 ``af_ld_q`` /
            # ``af_st_q``；``CUDA_DIALECT`` 里没有这两项。一元逻辑非
            # （``logical_not``）是唯一一个"只拿 dtype 就够"的 bool 内核，
            # 在这里按 dtype 自动选方言，避免每个调用点各写一遍。
            dialect = BOOL_DIALECT if _dtype_key(dtype) == "bool" else CUDA_DIALECT
        return emit(template, dialect, _ir_dtype(dtype))

    key = _dtype_key(dtype)
    storage, ld, st, add = _STORAGE[key]
    if key == "f16" and f16_helpers is not None:
        helpers = f16_helpers
    else:
        helpers = _HELPERS[key]
    return (
        template.replace("%PRELUDE%", _REDUCE + helpers)
        .replace("%T%", storage)
        .replace("%LD%", ld)
        .replace("%ST%", st)
        .replace("%ATOMIC%", add)
    )


BINARY_FN = {
    "add": "({}) + ({})",
    "sub": "({}) - ({})",
    "mul": "({}) * ({})",
    "div": "({}) / ({})",
}

UNARY_EXPR = {
    "relu": "x > 0.f ? x : 0.f",
    "silu": "x / (1.f + expf(-x))",
    # tanh 近似的 GELU：与 ir/ops.py 的参考实现逐位同式，
    # 常数就是 sqrt(2/pi) 与 0.044715（GPT/BERT 实际使用的定义）
    "gelu": "0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)))",
    # 自动微分需要的初等函数。与 relu/silu 走同一个一元模板：
    # 反向规则只用这些基本语义组合，所以反向图不需要单独的 kernel。
    "exp": "expf(x)",
    "log": "logf(x)",
    "sqrt": "sqrtf(x)",
    "sin": "sinf(x)",
    "cos": "cosf(x)",
    "neg": "-x",
    "tanh": "tanhf(x)",
    "sigmoid": "1.f / (1.f + expf(-x))",
    "erf": "erff(x)",
    # 嵌套三目与 L2 的 Cond 逐个 token 对齐；0 → 0、NaN 两个比较都为假 → 原样透传。
    "sign": "x > 0.f ? 1.f : x < 0.f ? -1.f : x",
    "round": "nearbyintf(x)",
    "floor": "floorf(x)",
    "ceil": "ceilf(x)",
}

# step 比其他一元算子多一个 kernel 参数（阈值），因此不进 UNARY_EXPR 模板。
STEP_SRC = (
    PRELUDE
    + r"""
extern "C" __global__ void ew_step(const %T%* __restrict__ in,
                                   %T%* __restrict__ out,
                                   long long n,
                                   float thr) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) { out[i] = %ST%(%LD%(in[i]) > thr ? 1.f : 0.f); }
}
"""
)

_BINARY_TMPL = (
    PRELUDE
    + r"""
extern "C" __global__ void ew_bin(%A_DECL%,
                                  %B_DECL%,
                                  %T%* __restrict__ out,
                                  long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) { out[i] = %ST%(%EXPR%); }
}
"""
)

_UNARY_TMPL = (
    PRELUDE
    + r"""
extern "C" __global__ void ew_un(const %T%* __restrict__ in,
                                 %T%* __restrict__ out,
                                 long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) { float x = %LD%(in[i]); out[i] = %ST%(%EXPR%); }
}
"""
)

_BCAST_TMPL = (
    PRELUDE
    + r"""
extern "C" __global__ void ew_bcast(const %T%* __restrict__ a,
                                    const %T%* __restrict__ b,
                                    %T%* __restrict__ out,
                                    long long n,
                                    int ndim,
                                    long long d0, long long d1, long long d2, long long d3,
                                    long long a0, long long a1, long long a2, long long a3,
                                    long long b0, long long b1, long long b2, long long b3) {
    long long shape[4] = {d0, d1, d2, d3};
    long long sa[4] = {a0, a1, a2, a3};
    long long sb[4] = {b0, b1, b2, b3};
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += step) {
        long long rem = i, oa = 0, ob = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            long long c = rem % shape[d];
            rem /= shape[d];
            oa += c * sa[d];
            ob += c * sb[d];
        }
        out[i] = %ST%(%EXPR%);
    }
}
"""
)

SOFTMAX_SRC = (
    PRELUDE
    + r"""
extern "C" __global__ void softmax_last(const %T%* __restrict__ x,
                                        %T%* __restrict__ out,
                                        long long rows, long long cols,
                                        const float* __restrict__ scale) {
    long long row = blockIdx.x;
    if (row >= rows) { return; }
    const %T%* xr = x + row * cols;
    %T%* orow = out + row * cols;
    const float sc = scale[0];
    __shared__ float smem[1024];

    float m = -3.4e38f;
    for (long long i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = %LD%(xr[i]) * sc;
        m = v > m ? v : m;
    }
    m = block_reduce_max(m, smem);

    float s = 0.f;
    for (long long i = threadIdx.x; i < cols; i += blockDim.x) { s += expf(%LD%(xr[i]) * sc - m); }
    s = block_reduce_sum(s, smem);
    float inv = 1.f / s;

    for (long long i = threadIdx.x; i < cols; i += blockDim.x) {
        orow[i] = %ST%(expf(%LD%(xr[i]) * sc - m) * inv);
    }
}
"""
)

# Fused attention forward. One block per (batch*head, query row); the key axis
# is streamed in chunks of blockDim.x with an online softmax, so the (S, T)
# score matrix is never materialized. Only the running row max, the running
# normalizer and the output accumulator live across chunks.
#
# Layout: q (BH, S, D), k (BH, T, D), v (BH, T, E), out (BH, S, E). The mask is
# either (BH, S, T) or a single (S, T) shared by every (b, h) when mask_shared=1 —
# the common causal case, which must not be copied into a (BH, S, T) buffer.
# When has_mask is 0 the pointer is a dummy and never dereferenced.
#
# Dropout multiplies the normalized weights, not the denominator:
#   out = sum_i exp(s_i - m) * keep_i * v_i / sum_i exp(s_i - m)
# which matches ir/ops.py, where dropout acts on the softmax probabilities.
# The mask is drawn from (seed, (bh*S + s)*T + t) -- the flat index of the
# probability matrix -- so it is bit-identical to the reference backend.
ATTENTION_SRC = (
    PRELUDE
    + r"""
__device__ __forceinline__ unsigned long long attention_rand(unsigned long long seed,
                                                             long long idx) {
    unsigned long long z =
        seed + (unsigned long long)idx * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

extern "C" __global__ void attention_fwd(const %T%* __restrict__ q,
                                         const %T%* __restrict__ k,
                                         const %T%* __restrict__ v,
                                         const %T%* __restrict__ mask,
                                         %T%* __restrict__ out,
                                         long long S, long long T,
                                         long long D, long long E,
                                         float scale,
                                         float p_drop, float drop_scale,
                                         unsigned long long seed,
                                         int has_mask,
                                         int mask_shared) {
    // Dynamic shared memory: [w | reduce scratch | accumulator].
    extern __shared__ float smem[];
    float* s_w = smem;
    float* s_red = smem + blockDim.x;
    float* s_acc = smem + 2 * blockDim.x;

    const long long bh = blockIdx.y;
    const long long s = blockIdx.x;

    for (long long e = threadIdx.x; e < E; e += blockDim.x) { s_acc[e] = 0.f; }
    __syncthreads();

    const %T%* qs = q + (bh * S + s) * D;
    const %T%* vbase = v + bh * T * E;
    const %T%* mrow =
        has_mask ? (mask + ((mask_shared ? 0 : bh) * S + s) * T) : nullptr;

    // Every thread keeps the same m and l: block_reduce_* broadcasts smem[0].
    float m = -3.4e38f;
    float l = 0.f;

    for (long long t0 = 0; t0 < T; t0 += blockDim.x) {
        const long long t = t0 + threadIdx.x;
        float sc = -3.4e38f;
        if (t < T) {
            const %T%* kt = k + (bh * T + t) * D;
            float dot = 0.f;
            for (long long d = 0; d < D; ++d) { dot += %LD%(qs[d]) * %LD%(kt[d]); }
            sc = dot * scale;
            if (has_mask) { sc += %LD%(mrow[t]); }
        }

        const float blk_max = block_reduce_max(sc, s_red);
        const float m_new = m > blk_max ? m : blk_max;
        // Rescale factor for everything accumulated so far. On the first
        // chunk m is -FLT_MAX, so corr underflows to 0 as intended.
        const float corr = expf(m - m_new);

        float p = 0.f;
        float w = 0.f;
        if (t < T) {
            p = expf(sc - m_new);
            w = p;
            if (p_drop > 0.f) {
                const unsigned long long z =
                    attention_rand(seed, (bh * S + s) * T + t);
                const float u =
                    (float)((z >> 40) & 0xFFFFFFULL) * (1.0f / 16777216.0f);
                w = (u >= p_drop) ? p * drop_scale : 0.f;
            }
        }
        s_w[threadIdx.x] = w;

        const float blk_sum = block_reduce_sum(p, s_red);
        l = l * corr + blk_sum;

        // Threads switch roles here: each one owns a slice of the output
        // channels and walks the whole chunk. Consecutive threads read
        // consecutive v elements, so the loads stay coalesced.
        for (long long e = threadIdx.x; e < E; e += blockDim.x) {
            float acc = s_acc[e] * corr;
            for (long long j = 0; j < blockDim.x && t0 + j < T; ++j) {
                acc += s_w[j] * %LD%(vbase[(t0 + j) * E + e]);
            }
            s_acc[e] = acc;
        }
        // s_w is rewritten by the next chunk: everyone must be done reading.
        __syncthreads();
        m = m_new;
    }

    const float inv = l > 0.f ? 1.f / l : 0.f;
    %T%* orow = out + (bh * S + s) * E;
    for (long long e = threadIdx.x; e < E; e += blockDim.x) {
        orow[e] = %ST%(s_acc[e] * inv);
    }
}
"""
)

LAYERNORM_SRC = (
    PRELUDE
    + r"""
extern "C" __global__ void layernorm_last(const %T%* __restrict__ x,
                                          const %T%* __restrict__ gamma,
                                          const %T%* __restrict__ beta,
                                          %T%* __restrict__ out,
                                          long long rows, long long cols, float eps) {
    long long row = blockIdx.x;
    if (row >= rows) { return; }
    const %T%* xr = x + row * cols;
    %T%* orow = out + row * cols;
    __shared__ float smem[1024];

    float s = 0.f;
    for (long long i = threadIdx.x; i < cols; i += blockDim.x) { s += %LD%(xr[i]); }
    s = block_reduce_sum(s, smem);
    float mean = s / (float)cols;

    float v = 0.f;
    for (long long i = threadIdx.x; i < cols; i += blockDim.x) {
        float d = %LD%(xr[i]) - mean;
        v += d * d;
    }
    v = block_reduce_sum(v, smem);
    float inv = rsqrtf(v / (float)cols + eps);

    for (long long i = threadIdx.x; i < cols; i += blockDim.x) {
        orow[i] = %ST%((%LD%(xr[i]) - mean) * inv * %LD%(gamma[i]) + %LD%(beta[i]));
    }
}
"""
)

# LayerNorm with a fused residual: the add is never materialized.
# The sum is rounded back to the storage type first, so f16 matches the
# reference semantics (numpy also adds two f16 arrays in f16).
LAYERNORM_RESIDUAL_SRC = (
    PRELUDE
    + r"""
extern "C" __global__ void layernorm_residual(const %T%* __restrict__ x,
                                              const %T%* __restrict__ res,
                                              const %T%* __restrict__ gamma,
                                              const %T%* __restrict__ beta,
                                              %T%* __restrict__ out,
                                              long long rows, long long cols, float eps) {
    long long row = blockIdx.x;
    if (row >= rows) { return; }
    const %T%* xr = x + row * cols;
    const %T%* rr = res + row * cols;
    %T%* orow = out + row * cols;
    __shared__ float smem[1024];

    float s = 0.f;
    for (long long i = threadIdx.x; i < cols; i += blockDim.x) {
        %T% xv = %ST%(%LD%(xr[i]) + %LD%(rr[i]));
        s += %LD%(xv);
    }
    s = block_reduce_sum(s, smem);
    float mean = s / (float)cols;

    float v = 0.f;
    for (long long i = threadIdx.x; i < cols; i += blockDim.x) {
        %T% xv = %ST%(%LD%(xr[i]) + %LD%(rr[i]));
        float d = %LD%(xv) - mean;
        v += d * d;
    }
    v = block_reduce_sum(v, smem);
    float inv = rsqrtf(v / (float)cols + eps);

    for (long long i = threadIdx.x; i < cols; i += blockDim.x) {
        %T% xv = %ST%(%LD%(xr[i]) + %LD%(rr[i]));
        orow[i] = %ST%((%LD%(xv) - mean) * inv * %LD%(gamma[i]) + %LD%(beta[i]));
    }
}
"""
)

# Gated SiLU (SwiGLU): silu(x) * gate in one pass, no intermediate tensor.
SILU_GATE_SRC = (
    PRELUDE
    + r"""
extern "C" __global__ void silu_gate(const %T%* __restrict__ x,
                                     const %T%* __restrict__ gate,
                                     %T%* __restrict__ out,
                                     long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        float xv = %LD%(x[i]);
        out[i] = %ST%(xv / (1.f + expf(-xv)) * %LD%(gate[i]));
    }
}
"""
)

# batch_norm / batch_norm_stats：一个 block 处理一个通道。
# x 在这里被看成 (outer, channels, inner)，所以同一个内核既能覆盖 NCHW，
# 也能覆盖任意"某一维是通道"的秩；outer/inner 由调用方按 axis 折算。
BATCH_NORM_SRC = (
    PRELUDE
    + r"""
extern "C" __global__ void batch_norm(const %T%* __restrict__ x,
                                      const %T%* __restrict__ gamma,
                                      const %T%* __restrict__ beta,
                                      %T%* __restrict__ out,
                                      long long outer, long long channels,
                                      long long inner, float eps) {
    const long long c = blockIdx.x;
    if (c >= channels) { return; }
    __shared__ float smem[1024];
    const long long stride = channels * inner;
    const float m = (float)(outer * inner);

    // Two passes rather than sumsq - mean^2: the reference implementation
    // (numpy.var) is two-pass, and matching it keeps the comparison tight.
    float s = 0.f;
    for (long long o = 0; o < outer; ++o) {
        const %T%* base = x + o * stride + c * inner;
        for (long long i = threadIdx.x; i < inner; i += blockDim.x) { s += %LD%(base[i]); }
    }
    s = block_reduce_sum(s, smem);
    const float mean = s / m;

    float v = 0.f;
    for (long long o = 0; o < outer; ++o) {
        const %T%* base = x + o * stride + c * inner;
        for (long long i = threadIdx.x; i < inner; i += blockDim.x) {
            const float d = %LD%(base[i]) - mean;
            v += d * d;
        }
    }
    v = block_reduce_sum(v, smem);
    const float inv = rsqrtf(v / m + eps);
    const float g = %LD%(gamma[c]);
    const float b = %LD%(beta[c]);

    for (long long o = 0; o < outer; ++o) {
        const %T%* src = x + o * stride + c * inner;
        %T%* dst = out + o * stride + c * inner;
        for (long long i = threadIdx.x; i < inner; i += blockDim.x) {
            dst[i] = %ST%((%LD%(src[i]) - mean) * inv * g + b);
        }
    }
}

// Same reduction as above, but writes mean/var out (flat layout (2, channels)).
extern "C" __global__ void batch_norm_stats(const %T%* __restrict__ x,
                                            %T%* __restrict__ out,
                                            long long outer, long long channels,
                                            long long inner) {
    const long long c = blockIdx.x;
    if (c >= channels) { return; }
    __shared__ float smem[1024];
    const long long stride = channels * inner;
    const float m = (float)(outer * inner);

    float s = 0.f;
    for (long long o = 0; o < outer; ++o) {
        const %T%* base = x + o * stride + c * inner;
        for (long long i = threadIdx.x; i < inner; i += blockDim.x) { s += %LD%(base[i]); }
    }
    s = block_reduce_sum(s, smem);
    const float mean = s / m;

    float v = 0.f;
    for (long long o = 0; o < outer; ++o) {
        const %T%* base = x + o * stride + c * inner;
        for (long long i = threadIdx.x; i < inner; i += blockDim.x) {
            const float d = %LD%(base[i]) - mean;
            v += d * d;
        }
    }
    v = block_reduce_sum(v, smem);

    if (threadIdx.x == 0) {
        out[c] = %ST%(mean);
        out[channels + c] = %ST%(v / m);
    }
}
"""
)

REDUCE_SUM_SRC = (
    PRELUDE
    + r"""
extern "C" __global__ void reduce_sum_last(const %T%* __restrict__ x,
                                           %T%* __restrict__ out,
                                           long long rows, long long cols) {
    long long row = blockIdx.x;
    if (row >= rows) { return; }
    const %T%* xr = x + row * cols;
    __shared__ float smem[1024];
    float s = 0.f;
    for (long long i = threadIdx.x; i < cols; i += blockDim.x) { s += %LD%(xr[i]); }
    s = block_reduce_sum(s, smem);
    if (threadIdx.x == 0) { out[row] = %ST%(s); }
}
"""
)


REDUCE_MAX_SRC = (
    PRELUDE
    + r"""
extern "C" __global__ void reduce_max_last(const %T%* __restrict__ x,
                                           %T%* __restrict__ out,
                                           long long rows, long long cols) {
    long long row = blockIdx.x;
    if (row >= rows) { return; }
    const %T%* xr = x + row * cols;
    __shared__ float smem[1024];
    float s = -3.4e38f;
    for (long long i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = %LD%(xr[i]);
        if (v > s) { s = v; }
    }
    s = block_reduce_max(s, smem);
    if (threadIdx.x == 0) { out[row] = %ST%(s); }
}
"""
)


REDUCE_MEAN_SRC = (
    PRELUDE
    + r"""
extern "C" __global__ void reduce_mean_last(const %T%* __restrict__ x,
                                            %T%* __restrict__ out,
                                            long long rows, long long cols) {
    long long row = blockIdx.x;
    if (row >= rows) { return; }
    const %T%* xr = x + row * cols;
    __shared__ float smem[1024];
    float s = 0.f;
    for (long long i = threadIdx.x; i < cols; i += blockDim.x) { s += %LD%(xr[i]); }
    s = block_reduce_sum(s, smem) / (float)cols;
    if (threadIdx.x == 0) { out[row] = %ST%(s); }
}
"""
)


# bool 归约（5.341）：存储是 0/1 的 1 字节，求和等价于"数真值个数"。
# ``any`` 看和是否非零，``all`` 看和是否等于列数；判定放在块内归约之后。
REDUCE_ANY_SRC = (
    PRELUDE
    + r"""
extern "C" __global__ void reduce_any_last(const %T%* __restrict__ x,
                                           %T%* __restrict__ out,
                                           long long rows, long long cols) {
    long long row = blockIdx.x;
    if (row >= rows) { return; }
    const %T%* xr = x + row * cols;
    __shared__ float smem[1024];
    float s = 0.f;
    for (long long i = threadIdx.x; i < cols; i += blockDim.x) { s += %LD%(xr[i]); }
    s = block_reduce_sum(s, smem) != 0.f ? 1.f : 0.f;
    if (threadIdx.x == 0) { out[row] = %ST%(s); }
}
"""
)


REDUCE_ALL_SRC = (
    PRELUDE
    + r"""
extern "C" __global__ void reduce_all_last(const %T%* __restrict__ x,
                                           %T%* __restrict__ out,
                                           long long rows, long long cols) {
    long long row = blockIdx.x;
    if (row >= rows) { return; }
    const %T%* xr = x + row * cols;
    __shared__ float smem[1024];
    float s = 0.f;
    for (long long i = threadIdx.x; i < cols; i += blockDim.x) { s += %LD%(xr[i]); }
    s = block_reduce_sum(s, smem) == (float)cols ? 1.f : 0.f;
    if (threadIdx.x == 0) { out[row] = %ST%(s); }
}
"""
)


ARGMAX_SRC = (
    PRELUDE
    + r"""
// argmax over the last dim: block-wide max, then smallest matching index via
// atomicMin so ties resolve to the first occurrence (numpy/PyTorch convention).
// NOTE: NVRTC requires ASCII source -- keep every comment in this file ASCII.
extern "C" __global__ void argmax_last(const %T%* __restrict__ x,
                                       int* __restrict__ out,
                                       long long rows, long long cols) {
    long long row = blockIdx.x;
    if (row >= rows) { return; }
    const %T%* xr = x + row * cols;
    __shared__ float smem[1024];
    __shared__ int ibuf[1];
    float best = -3.4e38f;
    int best_i = -1;
    for (long long i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = %LD%(xr[i]);
        if (v > best) { best = v; best_i = (int)i; }
    }
    float m = block_reduce_max(best, smem);
    if (threadIdx.x == 0) { ibuf[0] = (int)cols; }
    __syncthreads();
    if (best == m) { atomicMin(&ibuf[0], best_i); }
    __syncthreads();
    if (threadIdx.x == 0) { out[row] = ibuf[0]; }
}
"""
)


TOPK_SRC = (
    PRELUDE
    + r"""
// topk over the last dim: k rounds of (block-wide max, then smallest index via
// atomicMin). Ties resolve to the smaller index and NaN ranks as +inf, matching
// the IR reference (torch.topk convention). `sel` is dynamic shared memory with
// room for k ints; it doubles as the buffer for the chosen index and as the
// already-taken list, so the kernel never allocates per-row scratch.
// NOTE: NVRTC requires ASCII source -- keep every comment in this file ASCII.
__device__ __forceinline__ void topk_select_row(const %T%* __restrict__ xr,
                                                long long cols, int k, int mode,
                                                %T%* __restrict__ vals,
                                                int* __restrict__ ids,
                                                float* smem, int* sel) {
    for (int rank = 0; rank < k; ++rank) {
        float best = __int_as_float(0xff800000);   // -inf
        int best_i = 0x7fffffff;
        for (long long j = threadIdx.x; j < cols; j += blockDim.x) {
            int taken = 0;
            for (int t = 0; t < rank; ++t) {
                if (sel[t] == (int)j) { taken = 1; break; }
            }
            if (taken) { continue; }
            float v = %LD%(xr[j]);
            if (v != v) { v = __int_as_float(0x7f800000); }   // +inf
            if (v > best || (v == best && (int)j < best_i)) {
                best = v;
                best_i = (int)j;
            }
        }
        float m = block_reduce_max(best, smem);
        if (threadIdx.x == 0) { sel[rank] = (int)cols; }
        __syncthreads();
        if (best == m) { atomicMin(&sel[rank], best_i); }
        __syncthreads();
        if (threadIdx.x == 0) {
            int at = sel[rank];
            if (mode == 0) { vals[(long long)rank] = xr[at]; }
            else if (mode == 1) { ids[(long long)rank] = at; }
            else { vals[at] = %ST%(1.f); }
        }
        __syncthreads();
    }
}

extern "C" __global__ void topk_values(const %T%* __restrict__ x,
                                      void* __restrict__ outp,
                                      long long rows, long long cols, int k) {
    long long row = blockIdx.x;
    if (row >= rows) { return; }
    extern __shared__ int sel_dyn[];
    __shared__ float smem[1024];
    topk_select_row(x + row * cols, cols, k, 0,
                    (%T%*)outp + row * (long long)k, 0, smem, sel_dyn);
}

extern "C" __global__ void topk_indices(const %T%* __restrict__ x,
                                       void* __restrict__ outp,
                                       long long rows, long long cols, int k) {
    long long row = blockIdx.x;
    if (row >= rows) { return; }
    extern __shared__ int sel_dyn[];
    __shared__ float smem[1024];
    topk_select_row(x + row * cols, cols, k, 1,
                    0, (int*)outp + row * (long long)k, smem, sel_dyn);
}

extern "C" __global__ void topk_mask_select(const %T%* __restrict__ x,
                                            void* __restrict__ outp,
                                            long long rows, long long cols, int k) {
    long long row = blockIdx.x;
    if (row >= rows) { return; }
    extern __shared__ int sel_dyn[];
    __shared__ float smem[1024];
    topk_select_row(x + row * cols, cols, k, 2,
                    (%T%*)outp + row * cols, 0, smem, sel_dyn);
}
"""
)


IM2COL_SRC = (
    PRELUDE
    + r"""
// im2col: NCHW -> (N, C*KH*KW, HO*WO) column matrix for batched GEMM.
// Segments are kept per batch (not merged into one big matrix) so the GEMM
// output is already (N, O, HO*WO) and no transpose/col2im is needed.
// NOTE: NVRTC requires ASCII source -- keep every comment in this file ASCII.
extern "C" __global__ void im2col(const %T%* __restrict__ x,
                                  %T%* __restrict__ cols,
                                  int C, int H, int W,
                                  int KH, int KW, int HO, int WO,
                                  int SH, int SW, int PH, int PW,
                                  int DH, int DW,
                                  long long total) {
    const long long step = (long long)gridDim.x * blockDim.x;
    const int plane = KH * KW;
    const int per_n = C * plane;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += step) {
        int wo = (int)(idx % WO);
        long long t = idx / WO;
        int ho = (int)(t % HO);
        t /= HO;
        int k = (int)(t % per_n);
        long long n = t / per_n;
        int kj = k % KW;
        int ki = (k / KW) % KH;
        int c = k / plane;
        int ih = ho * SH - PH + ki * DH;
        int iw = wo * SW - PW + kj * DW;
        float v = 0.f;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            v = %LD%(x[((long long)n * C + c) * H * W + (long long)ih * W + iw]);
        }
        cols[idx] = %ST%(v);
    }
}
"""
)


MAXPOOL2D_SRC = (
    PRELUDE
    + r"""
// maxpool2d: one thread per output element, grid-stride loop.
// Cells that fall into the padded border are skipped, so they never compete
// with real values in the max (matches the IR reference, which pads with -inf).
// NOTE: NVRTC requires ASCII source -- keep every comment in this file ASCII.
extern "C" __global__ void maxpool2d(const %T%* __restrict__ x,
                                     %T%* __restrict__ out,
                                     int C, int H, int W,
                                     int KH, int KW, int HO, int WO,
                                     int SH, int SW, int PH, int PW,
                                     long long total) {
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += step) {
        int wo = (int)(idx % WO);
        long long t = idx / WO;
        int ho = (int)(t % HO);
        t /= HO;
        int c = (int)(t % C);
        long long n = t / C;
        const %T%* base = x + ((long long)n * C + c) * (long long)H * W;
        float m = __int_as_float(0xff800000);  // -inf; NVRTC has no <math.h> macros
        for (int ki = 0; ki < KH; ++ki) {
            int ih = ho * SH - PH + ki;
            if (ih < 0 || ih >= H) { continue; }
            for (int kj = 0; kj < KW; ++kj) {
                int iw = wo * SW - PW + kj;
                if (iw < 0 || iw >= W) { continue; }
                float v = %LD%(base[(long long)ih * W + iw]);
                m = v > m ? v : m;
            }
        }
        out[idx] = %ST%(m);
    }
}
"""
)


CONV2D_TILED_SRC = (
    PRELUDE
    + r"""
// Direct tiled conv2d (NCHW, cross-correlation, no kernel flip).
// One thread computes ROC output channels at one spatial position. Two kinds of
// reuse make this competitive with im2col + GEMM without materializing the
// (N, K, HO*WO) column buffer:
//   1. weights for the block's TOC channels are staged in shared memory CB
//      input channels at a time (TOC*CB*KH*KW floats), so shared memory stays
//      small enough to keep occupancy high while the barrier cost is amortized
//      over CB channels of work;
//   2. each input value fetched from global memory feeds ROC accumulators,
//      cutting global input reads by a factor of ROC.
// Threads that fall outside the output stay in the loop so every
// __syncthreads() is reached by the whole block.
// The bias is folded into the store: acc + (has_bias ? b[oc] : 0). Callers
// without a bias pass any valid pointer and has_bias = 0; it is never read.
// NOTE: NVRTC requires ASCII source -- keep every comment in this file ASCII.
extern "C" __global__ void conv2d_tiled(const float* __restrict__ x,
                                        const float* __restrict__ w,
                                        float* __restrict__ out,
                                        const float* __restrict__ b,
                                        int C, int H, int W,
                                        int O, int KH, int KW,
                                        int HO, int WO,
                                        int SH, int SW, int PH, int PW,
                                        int DH, int DW,
                                        int TOC, int ROC, int CB, int oc_tiles,
                                        int has_bias) {
    extern __shared__ float sw[];
    const int nx = blockDim.x;
    const int nt = nx * blockDim.y;
    const int tid = threadIdx.y * nx + threadIdx.x;
    const int khkw = KH * KW;
    const int K = C * khkw;

    const int tile = blockIdx.z;
    const int oc0 = (tile % oc_tiles) * TOC;
    const int n = tile / oc_tiles;
    const int ho = blockIdx.y;
    const int wo = blockIdx.x * nx + threadIdx.x;
    const int oc = oc0 + threadIdx.y * ROC;

    const bool active = (wo < WO) && (oc < O);
    // Fast path: the whole KH*KW window lies inside the image, so the inner
    // loops need no bounds checks. Only border threads take the slow path.
    const int ih0 = ho * SH - PH;
    const int iw0 = wo * SW - PW;
    const bool inside = active && ih0 >= 0 && ih0 + (KH - 1) * DH < H &&
                        iw0 >= 0 && iw0 + (KW - 1) * DW < W;
    const float* xn = x + (long long)n * C * H * W;
    float acc[8];
    for (int j = 0; j < ROC; ++j) { acc[j] = 0.f; }

    for (int c0 = 0; c0 < C; c0 += CB) {
        __syncthreads();
        const int cb = (C - c0 < CB) ? (C - c0) : CB;
        for (int i = tid; i < TOC * cb * khkw; i += nt) {
            const int t = i % khkw;
            const int rest = i / khkw;
            const int cc = rest % cb;
            const int ocg = oc0 + rest / cb;
            sw[i] = ocg < O ? w[(long long)ocg * K + (long long)(c0 + cc) * khkw + t]
                            : 0.f;
        }
        __syncthreads();
        if (!active) { continue; }

        if (inside) {
            // Fast path: the whole KH*KW window is inside the image, so the
            // inner loops carry no bounds checks at all.
            for (int cc = 0; cc < cb; ++cc) {
                const float* xc = xn + (long long)(c0 + cc) * H * W;
                const float* wb = sw + cc * khkw;
                for (int ki = 0; ki < KH; ++ki) {
                    const float* xr = xc + (long long)(ih0 + ki * DH) * W;
                    const int off = threadIdx.y * ROC * cb * khkw + ki * KW;
                    for (int kj = 0; kj < KW; ++kj) {
                        const float v = xr[iw0 + kj * DW];
                        for (int j = 0; j < ROC; ++j) {
                            acc[j] += v * wb[off + j * cb * khkw + kj];
                        }
                    }
                }
            }
        } else {
            for (int cc = 0; cc < cb; ++cc) {
                const float* xc = xn + (long long)(c0 + cc) * H * W;
                const float* wb = sw + cc * khkw;
                for (int ki = 0; ki < KH; ++ki) {
                    const int ih = ih0 + ki * DH;
                    if (ih < 0 || ih >= H) { continue; }
                    const float* xr = xc + (long long)ih * W;
                    const int off = threadIdx.y * ROC * cb * khkw + ki * KW;
                    for (int kj = 0; kj < KW; ++kj) {
                        const int iw = iw0 + kj * DW;
                        if (iw < 0 || iw >= W) { continue; }
                        const float v = xr[iw];
                        for (int j = 0; j < ROC; ++j) {
                            acc[j] += v * wb[off + j * cb * khkw + kj];
                        }
                    }
                }
            }
        }
    }

    if (!active) { return; }
    float* dst = out + ((long long)n * O + oc) * HO * WO + (long long)ho * WO + wo;
    for (int j = 0; j < ROC; ++j) {
        if (oc + j < O) {
            dst[(long long)j * HO * WO] = acc[j] + (has_bias ? b[oc + j] : 0.f);
        }
    }
}
"""
)


RMSNORM_SRC = (
    PRELUDE
    + r"""
// rmsnorm: one block per row, reduce mean(x^2) over the last dim.
// No mean-centering and no beta -- that is what separates RMSNorm from
// LayerNorm, and it is why the reference only needs one reduction.
// NOTE: NVRTC requires ASCII source -- keep every comment in this file ASCII.
extern "C" __global__ void rmsnorm_last(const %T%* __restrict__ x,
                                        const %T%* __restrict__ gamma,
                                        %T%* __restrict__ out,
                                        long long rows, long long cols, float eps) {
    long long row = blockIdx.x;
    if (row >= rows) { return; }
    const %T%* xr = x + row * cols;
    %T%* orow = out + row * cols;
    __shared__ float smem[1024];

    float s = 0.f;
    for (long long i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = %LD%(xr[i]);
        s += v * v;
    }
    s = block_reduce_sum(s, smem);
    float inv = rsqrtf(s / (float)cols + eps);

    for (long long i = threadIdx.x; i < cols; i += blockDim.x) {
        orow[i] = %ST%(%LD%(xr[i]) * inv * %LD%(gamma[i]));
    }
}
"""
)

AVGPOOL2D_SRC = (
    PRELUDE
    + r"""
// avgpool2d: one thread per output element, grid-stride loop.
// The divisor is the number of taps that landed inside the *padded* input:
// with count_include_pad the padding zeros count towards it (bounds
// [-P, size + P)), otherwise only real samples do (bounds [0, size)).
// ceil_mode lets the last window start inside the image but run past the
// padded input; those overhanging taps count for neither the sum nor the
// divisor, which is why the divisor cannot be a fixed KH*KW.
// NOTE: NVRTC requires ASCII source -- keep every comment in this file ASCII.
extern "C" __global__ void avgpool2d(const %T%* __restrict__ x,
                                     %T%* __restrict__ out,
                                     int C, int H, int W,
                                     int KH, int KW, int HO, int WO,
                                     int SH, int SW, int PH, int PW,
                                     long long total,
                                     int count_include_pad) {
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += step) {
        int wo = (int)(idx % WO);
        long long t = idx / WO;
        int ho = (int)(t % HO);
        t /= HO;
        int c = (int)(t % C);
        long long n = t / C;
        int lo_h = count_include_pad != 0 ? -1 * PH : 0;
        int hi_h = count_include_pad != 0 ? H + PH : H;
        int lo_w = count_include_pad != 0 ? -1 * PW : 0;
        int hi_w = count_include_pad != 0 ? W + PW : W;
        const %T%* base = x + ((long long)n * C + c) * (long long)H * W;
        float acc = 0.f;
        int cnt = 0;
        for (int ki = 0; ki < KH; ++ki) {
            int ih = ho * SH - PH + ki;
            if (ih < lo_h || ih >= hi_h) { continue; }
            for (int kj = 0; kj < KW; ++kj) {
                int iw = wo * SW - PW + kj;
                if (iw < lo_w || iw >= hi_w) { continue; }
                cnt += 1;
                acc += ih >= 0 && ih < H && iw >= 0 && iw < W ? %LD%(base[(long long)ih * W + iw]) : 0.f;
            }
        }
        float den = cnt > 0 ? 1.f / (float)cnt : 1.f;
        out[idx] = %ST%(acc * den);
    }
}
"""
)

AVGPOOL2D_BACKWARD_SRC = (
    PRELUDE
    + r"""
// avgpool2d backward: scatter dy/divisor back to every real input position.
// The divisor matches the forward pass tap for tap: count_include_pad widens
// the counting bounds to [-P, size + P), otherwise they stay [0, size).
// Overlapping windows touch the same input cell, so the accumulation must be
// atomic -- a plain store would keep only the last window's contribution.
// Padding only contributes to the divisor; it never receives a gradient.
// dx is zero-initialised by the caller.
// NOTE: NVRTC requires ASCII source -- keep every comment in this file ASCII.
extern "C" __global__ void avgpool2d_backward(const %T%* __restrict__ dy,
                                              %T%* __restrict__ dx,
                                              int C, int H, int W,
                                              int KH, int KW, int HO, int WO,
                                              int SH, int SW, int PH, int PW,
                                              long long total,
                                              int count_include_pad) {
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += step) {
        int wo = (int)(idx % WO);
        long long t = idx / WO;
        int ho = (int)(t % HO);
        t /= HO;
        int c = (int)(t % C);
        long long n = t / C;
        int lo_h = count_include_pad != 0 ? -1 * PH : 0;
        int hi_h = count_include_pad != 0 ? H + PH : H;
        int lo_w = count_include_pad != 0 ? -1 * PW : 0;
        int hi_w = count_include_pad != 0 ? W + PW : W;
        int cnt = 0;
        for (int ki = 0; ki < KH; ++ki) {
            int ih = ho * SH - PH + ki;
            if (ih < lo_h || ih >= hi_h) { continue; }
            for (int kj = 0; kj < KW; ++kj) {
                int iw = wo * SW - PW + kj;
                if (iw < lo_w || iw >= hi_w) { continue; }
                cnt += 1;
            }
        }
        float den = cnt > 0 ? 1.f / (float)cnt : 1.f;
        float g = %LD%(dy[idx]) * den;
        %T%* base = dx + ((long long)n * C + c) * (long long)H * W;
        for (int ki = 0; ki < KH; ++ki) {
            int ih = ho * SH - PH + ki;
            if (ih < lo_h || ih >= hi_h) { continue; }
            for (int kj = 0; kj < KW; ++kj) {
                int iw = wo * SW - PW + kj;
                if (iw < lo_w || iw >= hi_w) { continue; }
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    %ATOMIC%(&base[(long long)ih * W + iw], g);
                }
            }
        }
    }
}
"""
)

ADAPTIVE_AVGPOOL2D_SRC = (
    PRELUDE
    + r"""
// adaptive_avg_pool2d: one thread per output element.
// The window of output cell (oh, ow) is [floor(oh*H/OH), ceil((oh+1)*H/OH))
// by [floor(ow*W/OW), ceil((ow+1)*W/OW)); for H not divisible by OH adjacent
// windows overlap, and the divisor is this window's own area, not a constant.
// The integer form of floor/ceil is used so the window edges match the IR
// reference exactly (no float rounding on the boundary).
// NOTE: NVRTC requires ASCII source -- keep every comment in this file ASCII.
extern "C" __global__ void adaptive_avg_pool2d(const %T%* __restrict__ x,
                                               %T%* __restrict__ out,
                                               int C, int H, int W,
                                               int OH, int OW,
                                               long long total) {
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += step) {
        int ow = (int)(idx % OW);
        long long t = idx / OW;
        int oh = (int)(t % OH);
        t /= OH;
        int c = (int)(t % C);
        long long n = t / C;
        const %T%* base = x + ((long long)n * C + c) * (long long)H * W;
        int hs = (int)(((long long)oh * H) / OH);
        int he = (int)((((long long)(oh + 1) * H) + OH - 1) / OH);
        int ws = (int)(((long long)ow * W) / OW);
        int we = (int)((((long long)(ow + 1) * W) + OW - 1) / OW);
        float acc = 0.f;
        for (int ih = hs; ih < he; ++ih) {
            for (int iw = ws; iw < we; ++iw) {
                acc += %LD%(base[(long long)ih * W + iw]);
            }
        }
        out[idx] = %ST%(acc / (float)((he - hs) * (we - ws)));
    }
}
"""
)

ADAPTIVE_AVGPOOL2D_BACKWARD_SRC = (
    PRELUDE
    + r"""
// adaptive_avg_pool2d backward: scatter dy/window_area to the window's cells.
// Windows may overlap when the input size is not divisible by the output size,
// so the same input cell can receive several contributions -- atomic add is
// required (dx is zero-initialised by the caller), exactly like avgpool2d.
// NOTE: NVRTC requires ASCII source -- keep every comment in this file ASCII.
extern "C" __global__ void adaptive_avg_pool2d_backward(const %T%* __restrict__ dy,
                                                        %T%* __restrict__ dx,
                                                        int C, int H, int W,
                                                        int OH, int OW,
                                                        long long total) {
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += step) {
        int ow = (int)(idx % OW);
        long long t = idx / OW;
        int oh = (int)(t % OH);
        t /= OH;
        int c = (int)(t % C);
        long long n = t / C;
        int hs = (int)(((long long)oh * H) / OH);
        int he = (int)((((long long)(oh + 1) * H) + OH - 1) / OH);
        int ws = (int)(((long long)ow * W) / OW);
        int we = (int)((((long long)(ow + 1) * W) + OW - 1) / OW);
        float share = %LD%(dy[idx]) / (float)((he - hs) * (we - ws));
        %T%* base = dx + ((long long)n * C + c) * (long long)H * W;
        for (int ih = hs; ih < he; ++ih) {
            for (int iw = ws; iw < we; ++iw) {
                %ATOMIC%(&base[(long long)ih * W + iw], share);
            }
        }
    }
}
"""
)

CONCAT_COPY_SRC = (
    PRELUDE
    + r"""
// concat: one strided copy per input segment, launched once per input.
// Viewing the tensor as (outer, axis, inner) turns concat into "copy this
// segment to a shifted axis offset"; no index table over the input list is
// needed, and every copy stays coalesced along the inner dimension.
// NOTE: NVRTC requires ASCII source -- keep every comment in this file ASCII.
extern "C" __global__ void concat_copy(const %T%* __restrict__ x,
                                       %T%* __restrict__ out,
                                       long long len, long long inner,
                                       long long out_axis, long long offset,
                                       long long total) {
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += step) {
        long long j = idx % inner;
        long long t = idx / inner;
        long long s = t % len;
        long long o = t / len;
        out[((o * out_axis) + offset + s) * inner + j] = x[idx];
    }
}
"""
)

MAXPOOL2D_BACKWARD_SRC = (
    PRELUDE
    + r"""
// maxpool2d backward: route dy to the argmax cell of each window.
// Ties pick the FIRST maximum in row-major scan order, which is exactly what
// the IR reference does -- flipping the comparison to >= would silently change
// the semantics on plateaus. Overlapping windows can land on the same input
// cell, so the scatter must be atomic.
// NOTE: NVRTC requires ASCII source -- keep every comment in this file ASCII.
extern "C" __global__ void maxpool2d_backward(const %T%* __restrict__ x,
                                              const %T%* __restrict__ dy,
                                              %T%* __restrict__ dx,
                                              int C, int H, int W,
                                              int KH, int KW, int HO, int WO,
                                              int SH, int SW, int PH, int PW,
                                              long long total) {
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += step) {
        int wo = (int)(idx % WO);
        long long t = idx / WO;
        int ho = (int)(t % HO);
        t /= HO;
        int c = (int)(t % C);
        long long n = t / C;
        const %T%* base = x + ((long long)n * C + c) * (long long)H * W;
        float best = -3.4e38f;
        int best_ih = -1, best_iw = -1;
        for (int ki = 0; ki < KH; ++ki) {
            int ih = ho * SH - PH + ki;
            if (ih < 0 || ih >= H) { continue; }
            for (int kj = 0; kj < KW; ++kj) {
                int iw = wo * SW - PW + kj;
                if (iw < 0 || iw >= W) { continue; }
                float v = %LD%(base[(long long)ih * W + iw]);
                if (best_ih < 0 || v > best) {
                    best = v;
                    best_ih = ih;
                    best_iw = iw;
                }
            }
        }
        if (best_ih >= 0) {
            %T%* dst = dx + ((long long)n * C + c) * (long long)H * W;
            %ATOMIC%(&dst[(long long)best_ih * W + best_iw], %LD%(dy[idx]));
        }
    }
}
"""
)

MAXPOOL2D_INDICES_SRC = (
    PRELUDE
    + r"""
// maxpool2d indices: one thread per output element.
// Writes ih*W + iw of the window maximum (i32), matching PyTorch's
// max_pool2d(return_indices=True). Ties pick the FIRST maximum in row-major
// scan order, exactly like maxpool2d_backward, so the three pooling ops agree
// on which element they mean. A window entirely inside padding would leave
// nothing to report; -1 is written first so the output is never undefined.
// NOTE: NVRTC requires ASCII source -- keep every comment in this file ASCII.
extern "C" __global__ void maxpool2d_indices(const %T%* __restrict__ x,
                                             int* __restrict__ out,
                                             int C, int H, int W,
                                             int KH, int KW, int HO, int WO,
                                             int SH, int SW, int PH, int PW,
                                             long long total) {
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += step) {
        int wo = (int)(idx % WO);
        long long t = idx / WO;
        int ho = (int)(t % HO);
        t /= HO;
        int c = (int)(t % C);
        long long n = t / C;
        const %T%* base = x + ((long long)n * C + c) * (long long)H * W;
        float best = -3.4e38f;
        int best_ih = -1, best_iw = -1;
        for (int ki = 0; ki < KH; ++ki) {
            int ih = ho * SH - PH + ki;
            if (ih < 0 || ih >= H) { continue; }
            for (int kj = 0; kj < KW; ++kj) {
                int iw = wo * SW - PW + kj;
                if (iw < 0 || iw >= W) { continue; }
                float v = %LD%(base[(long long)ih * W + iw]);
                if (best_ih < 0 || v > best) {
                    best = v;
                    best_ih = ih;
                    best_iw = iw;
                }
            }
        }
        out[idx] = -1;
        if (best_ih >= 0) {
            out[idx] = best_ih * W + best_iw;
        }
    }
}
"""
)

CONV2D_INPUT_GRAD_SRC = (
    PRELUDE
    + r"""
// conv2d input gradient (gather form): one thread per dx element.
// For a fixed input cell (ih, iw) the contributing output cells are those with
// ho*SH - PH + ki*DH == ih, so ho = (ih + PH - ki*DH)/SH and must be an exact
// non-negative division. Gathering avoids atomics entirely: every dx element
// is written by exactly one thread, and each thread sums over (o, ki, kj).
// NOTE: NVRTC requires ASCII source -- keep every comment in this file ASCII.
extern "C" __global__ void conv2d_input_grad(const %T%* __restrict__ dy,
                                             const %T%* __restrict__ weight,
                                             %T%* __restrict__ dx,
                                             int C, int H, int W,
                                             int O, int KH, int KW, int HO, int WO,
                                             int SH, int SW, int PH, int PW,
                                             int DH, int DW,
                                             long long total) {
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += step) {
        int iw = (int)(idx % W);
        long long t = idx / W;
        int ih = (int)(t % H);
        t /= H;
        int c = (int)(t % C);
        long long n = t / C;

        float acc = 0.f;
        for (int oi = 0; oi < O; ++oi) {
            for (int ki = 0; ki < KH; ++ki) {
                int num = ih + PH - ki * DH;
                if (num < 0 || num % SH != 0) { continue; }
                int ho = num / SH;
                if (ho >= HO) { continue; }
                for (int kj = 0; kj < KW; ++kj) {
                    int numw = iw + PW - kj * DW;
                    if (numw < 0 || numw % SW != 0) { continue; }
                    int wo = numw / SW;
                    if (wo >= WO) { continue; }
                    acc += %LD%(dy[((long long)(n * O + oi) * HO + ho) * WO + wo])
                         * %LD%(weight[((long long)(oi * C + c) * KH + ki) * KW + kj]);
                }
            }
        }
        dx[idx] = %ST%(acc);
    }
}
"""
)

CONV2D_WEIGHT_GRAD_SRC = (
    PRELUDE
    + r"""
// conv2d weight gradient (gather form): one thread per dw element.
// Accumulates over the whole batch and every output cell, so small batch
// training is compute-bound here -- same order as the forward convolution.
// Gathering again keeps the result deterministic (no float atomics).
// NOTE: NVRTC requires ASCII source -- keep every comment in this file ASCII.
extern "C" __global__ void conv2d_weight_grad(const %T%* __restrict__ x,
                                              const %T%* __restrict__ dy,
                                              %T%* __restrict__ dw,
                                              int N, int C, int H, int W,
                                              int O, int KH, int KW, int HO, int WO,
                                              int SH, int SW, int PH, int PW,
                                              int DH, int DW,
                                              long long total) {
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += step) {
        int kj = (int)(idx % KW);
        long long t = idx / KW;
        int ki = (int)(t % KH);
        t /= KH;
        int ci = (int)(t % C);
        int oi = (int)(t / C);

        float acc = 0.f;
        for (long long ni = 0; ni < N; ++ni) {
            for (int ho = 0; ho < HO; ++ho) {
                int ih = ho * SH - PH + ki * DH;
                if (ih < 0 || ih >= H) { continue; }
                for (int wo = 0; wo < WO; ++wo) {
                    int iw = wo * SW - PW + kj * DW;
                    if (iw < 0 || iw >= W) { continue; }
                    acc += %LD%(dy[((long long)(ni * O + oi) * HO + ho) * WO + wo])
                         * %LD%(x[((long long)(ni * C + ci) * H + ih) * W + iw]);
                }
            }
        }
        dw[idx] = %ST%(acc);
    }
}
"""
)

# dropout 的掩码生成。随机源刻意不用 cuRAND：参考实现（numpy）与内核必须
# 生成**逐位相同**的掩码，否则这个算子无法做跨后端对拍，只剩"统计上近似"
# 这一种弱验证。SplitMix64 只用整数乘法与异或，两边用同一套位运算即可。
DROPOUT_MASK_SRC = (
    PRELUDE
    + r"""
__device__ __forceinline__ unsigned long long splitmix64(unsigned long long z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// One thread per element. The flat index doubles as the counter, so the mask
// depends only on (seed, index) and never on the launch geometry: the same
// seed yields the same mask on any device and with any block size.
extern "C" __global__ void dropout_mask(%T%* __restrict__ out,
                                        long long n,
                                        float p,
                                        float scale,
                                        unsigned long long seed) {
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += step) {
        unsigned long long z =
            splitmix64(seed + (unsigned long long)i * 0x9E3779B97F4A7C15ULL);
        // Top 24 bits map to [0, 1); that is exactly the float32 mantissa width,
        // so numpy and CUDA land on the same float without extra rounding.
        float u = (float)((z >> 40) & 0xFFFFFFULL) * (1.0f / 16777216.0f);
        out[i] = %ST%((u >= p) ? scale : 0.0f);
    }
}
"""
)

# 运行期 key 版：seed 从设备内存读，捕获成 CUDA Graph 之后仍能换掩码。
# L2 描述的参数类型只有浮点张量与标量，表达不了"整数指针"参数，所以这一版
# 是手写模板；常量 seed 版继续走 L2（流水线没有分叉，只是这一个变体多一条路）。
DROPOUT_MASK_KEYED_SRC = (
    PRELUDE
    + r"""
__device__ __forceinline__ unsigned long long splitmix64(unsigned long long z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Same bit-exact mapping as the constant-seed kernel, but the seed is read
// from device memory: a captured CUDA Graph replays the same launch, so a
// scalar kernel argument would freeze the mask forever. `offset` separates
// dropout sites that share one key input.
extern "C" __global__ void dropout_mask_keyed(%T%* __restrict__ out,
                                              long long n,
                                              float p,
                                              float scale,
                                              const int* __restrict__ key,
                                              unsigned long long offset) {
    const unsigned long long seed =
        (unsigned long long)(long long)(*key) + offset;
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += step) {
        unsigned long long z =
            splitmix64(seed + (unsigned long long)i * 0x9E3779B97F4A7C15ULL);
        float u = (float)((z >> 40) & 0xFFFFFFULL) * (1.0f / 16777216.0f);
        out[i] = %ST%((u >= p) ? scale : 0.0f);
    }
}
"""
)

_EMBEDDING_TMPL = (
    PRELUDE
    + r"""
// embedding: gather row ids[i] from the table. The table is viewed as
// (vocab, row) so any trailing shape works without extra arithmetic.
// An out-of-range id cannot be raised from inside a kernel: instead of a
// silent clamp we flag it in *err, and the host turns that into a
// BackendCapabilityError. The clamp only prevents an illegal memory access
// on the way to that error.
// NOTE: NVRTC requires ASCII source -- keep every comment in this file ASCII.
extern "C" __global__ void embedding(const %T%* __restrict__ table,
                                     const %IDS_T%* __restrict__ ids,
                                     %T%* __restrict__ out,
                                     long long n, long long row, long long vocab,
                                     int* __restrict__ err, long long total) {
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += step) {
        long long j = idx % row;
        long long i = idx / row;
        long long v = (long long)ids[i];
        if (v < 0 || v >= vocab) { *err = 1; v = 0; }
        out[idx] = table[v * row + j];
    }
}
"""
)

_EMBEDDING_BACKWARD_TMPL = (
    PRELUDE
    + r"""
// embedding backward: scatter-add each dy row back to its table row.
// Repeated ids are the normal case (the same token appears many times in a
// batch), so the accumulation must be atomic; assignment would lose all but
// the last occurrence.
// NOTE: NVRTC requires ASCII source -- keep every comment in this file ASCII.
extern "C" __global__ void embedding_backward(const %T%* __restrict__ dy,
                                              const %IDS_T%* __restrict__ ids,
                                              %T%* __restrict__ dtable,
                                              long long n, long long row, long long vocab,
                                              int* __restrict__ err, long long total) {
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += step) {
        long long j = idx % row;
        long long i = idx / row;
        long long v = (long long)ids[i];
        if (v < 0 || v >= vocab) { *err = 1; continue; }
        %ATOMIC%(&dtable[v * row + j], %LD%(dy[idx]));
    }
}
"""
)


# gather: out[p] = x[(o * x_dim + v) * inner + i]. idx / out are viewed as
# (outer, idx_dim, inner) and x as (outer, x_dim, inner) -- the IR requires
# the same rank and equal non-dim extents, so this three-axis form is exact.
# Out-of-range ids are flagged in *err instead of being wrapped; the clamp to
# 0 only prevents an illegal access on the way to the host-side error.
# NOTE: NVRTC requires ASCII source -- keep every comment in this file ASCII.
_GATHER_TMPL = (
    PRELUDE
    + r"""
extern "C" __global__ void gather(const %T%* __restrict__ x,
                                  const %IDS_T%* __restrict__ idx,
                                  %T%* __restrict__ out,
                                  long long total, long long inner,
                                  long long idx_dim, long long x_dim,
                                  int* __restrict__ err) {
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         p < total; p += step) {
        const long long i = p % inner;
        const long long t = p / inner;
        const long long o = t / idx_dim;
        long long v = (long long)idx[p];
        if (v < 0 || v >= x_dim) { *err = 1; v = 0; }
        out[p] = x[(o * x_dim + v) * inner + i];
    }
}
"""
)

# gather backward: scatter-add dy[p] to the selected slot of x. Repeated ids
# are expected (the same row can be selected many times), so the add is
# atomic; a plain store would keep only the last contribution.
_GATHER_BACKWARD_TMPL = (
    PRELUDE
    + r"""
extern "C" __global__ void gather_backward(const %T%* __restrict__ dy,
                                           const %IDS_T%* __restrict__ idx,
                                           %T%* __restrict__ dx,
                                           long long total, long long inner,
                                           long long idx_dim, long long x_dim,
                                           int* __restrict__ err) {
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         p < total; p += step) {
        const long long i = p % inner;
        const long long t = p / inner;
        const long long o = t / idx_dim;
        long long v = (long long)idx[p];
        if (v < 0 || v >= x_dim) { *err = 1; continue; }
        %ATOMIC%(&dx[(o * x_dim + v) * inner + i], %LD%(dy[p]));
    }
}
"""
)

# cross_entropy: one block per row, two block-wide reductions (max, then
# exp-sum). The softmax probabilities are never materialised: only the row
# statistics and the single target logit are touched, which is the whole
# point of fusing when V is 32k and there are B*T rows.
# Shared scratch is REDUCE_SCRATCH (1024 floats), matching the block cap.
_CROSS_ENTROPY_TMPL = (
    PRELUDE
    + r"""
extern "C" __global__ void cross_entropy(const %T%* __restrict__ logits,
                                         const %IDS_T%* __restrict__ target,
                                         %T%* __restrict__ out,
                                         long long rows, long long vocab,
                                         int* __restrict__ err) {
    __shared__ float smem[1024];
    const long long row = (long long)blockIdx.x;
    if (row >= rows) { return; }
    const %T%* xr = logits + row * vocab;
    float m = -3.4e38f;
    for (long long j = threadIdx.x; j < vocab; j += blockDim.x) {
        const float v = %LD%(xr[j]);
        m = v > m ? v : m;
    }
    m = block_reduce_max(m, smem);
    float s = 0.f;
    for (long long j = threadIdx.x; j < vocab; j += blockDim.x) {
        s += expf(%LD%(xr[j]) - m);
    }
    s = block_reduce_sum(s, smem);
    if (threadIdx.x == 0) {
        long long t = (long long)target[row];
        if (t < 0 || t >= vocab) { *err = 1; t = 0; }
        out[row] = %ST%(m + logf(s) - %LD%(xr[t]));
    }
}
"""
)

# cross_entropy backward: (softmax - onehot) * dloss, computed row by row.
# It re-derives the row statistics instead of caching probabilities from the
# forward pass: caching would need a [rows, vocab] buffer, exactly the
# materialisation the fused forward avoids.
_CROSS_ENTROPY_BACKWARD_TMPL = (
    PRELUDE
    + r"""
extern "C" __global__ void cross_entropy_backward(
    const %T%* __restrict__ logits,
    const %IDS_T%* __restrict__ target,
    const %T%* __restrict__ dloss,
    %T%* __restrict__ dlogits,
    long long rows, long long vocab,
    int* __restrict__ err) {
    __shared__ float smem[1024];
    const long long row = (long long)blockIdx.x;
    if (row >= rows) { return; }
    const %T%* xr = logits + row * vocab;
    %T%* dr = dlogits + row * vocab;
    float m = -3.4e38f;
    for (long long j = threadIdx.x; j < vocab; j += blockDim.x) {
        const float v = %LD%(xr[j]);
        m = v > m ? v : m;
    }
    m = block_reduce_max(m, smem);
    float s = 0.f;
    for (long long j = threadIdx.x; j < vocab; j += blockDim.x) {
        s += expf(%LD%(xr[j]) - m);
    }
    s = block_reduce_sum(s, smem);
    const float inv = 1.f / s;
    const float g = %LD%(dloss[row]);
    long long t = (long long)target[row];
    if (t < 0 || t >= vocab) { *err = 1; t = 0; }
    for (long long j = threadIdx.x; j < vocab; j += blockDim.x) {
        float p = expf(%LD%(xr[j]) - m) * inv;
        if (j == t) { p -= 1.f; }
        dr[j] = %ST%(p * g);
    }
}
"""
)


# scan_linear: h_t = a_t * h_{t-1} + b_t, h_{-1} = h0.
# One thread owns one tail element and walks the time axis in order. The
# recurrence is inherently sequential, so the parallelism comes from the tail
# (batch / channel / state), not from T; the tail is flattened, so any rank
# >= 2 works without extra index math. A chunked (Blelloch) scan would need a
# second pass and is left for a later round -- this kernel is the semantics,
# not a bandwidth-optimal implementation.
# NOTE: NVRTC requires ASCII source -- keep every comment in this file ASCII.
SCAN_LINEAR_SRC = (
    PRELUDE
    + r"""
extern "C" __global__ void scan_linear(const %T%* __restrict__ a,
                                       const %T%* __restrict__ b,
                                       const %T%* __restrict__ h0,
                                       %T%* __restrict__ out,
                                       long long steps, long long elems) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < elems; i += stride) {
        float h = %LD%(h0[i]);
        for (long long t = 0; t < steps; ++t) {
            const long long idx = t * elems + i;
            h = %LD%(a[idx]) * h + %LD%(b[idx]);
            out[idx] = %ST%(h);
        }
    }
}
"""
)

# scan_linear_backward: dh_t = dy_t + a_{t+1} * dh_{t+1}, dh_T = 0.
# Same owner-per-tail-element layout, walked backwards. a[idx + elems] is
# a_{t+1}; the last step has no successor, so its carry is dropped.
SCAN_LINEAR_BACKWARD_SRC = (
    PRELUDE
    + r"""
extern "C" __global__ void scan_linear_backward(const %T%* __restrict__ a,
                                                const %T%* __restrict__ dy,
                                                %T%* __restrict__ out,
                                                long long steps, long long elems) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < elems; i += stride) {
        float dh = 0.f;
        for (long long t = steps - 1; t >= 0; --t) {
            const long long idx = t * elems + i;
            const float carry = (t + 1 < steps) ? %LD%(a[idx + elems]) * dh : 0.f;
            dh = %LD%(dy[idx]) + carry;
            out[idx] = %ST%(dh);
        }
    }
}
"""
)


# ids 的 numpy dtype -> CUDA C 类型。不能直接把 "int32" 填进源码：
# CUDA C 没有这个类型名，NVRTC 会在编译期报 "identifier int32 is undefined"。
_IDS_C_TYPE = {"int32": "int", "int8": "signed char"}


def embedding_source(ids_dtype) -> str:
    """按 ids 的实际类型生成查表 kernel。

    i32 与 i8 是两个真实的来源（前端导入的 token 既可能是 int32 索引，
    也可能是 int8 压缩过的标签），所以按类型生成而不是在 kernel 里做
    有符号扩展——后者在 NVRTC 下会把类型问题推迟成运行期的静默错值。
    """
    name = str(ids_dtype)
    if name not in _IDS_C_TYPE:
        raise ValueError(f"embedding 的 ids 只支持 int32/int8，收到 {ids_dtype}")
    return _EMBEDDING_TMPL.replace("%IDS_T%", _IDS_C_TYPE[name])


def embedding_backward_source(ids_dtype) -> str:
    name = str(ids_dtype)
    if name not in _IDS_C_TYPE:
        raise ValueError(f"embedding_backward 的 ids 只支持 int32/int8，收到 {ids_dtype}")
    return _EMBEDDING_BACKWARD_TMPL.replace("%IDS_T%", _IDS_C_TYPE[name])


def _ids_template(template: str, ids_dtype, op: str) -> str:
    """把 ``%IDS_T%`` 换成真实的 C 类型；只接受 IR 允许的两种下标类型。"""
    name = str(ids_dtype)
    if name not in _IDS_C_TYPE:
        raise ValueError(f"{op} 的 idx 只支持 int32/int8，收到 {ids_dtype}")
    return template.replace("%IDS_T%", _IDS_C_TYPE[name])


def gather_source(ids_dtype) -> str:
    return _ids_template(_GATHER_TMPL, ids_dtype, "gather")


def gather_backward_source(ids_dtype) -> str:
    return _ids_template(_GATHER_BACKWARD_TMPL, ids_dtype, "gather_backward")


def cross_entropy_source(ids_dtype) -> str:
    return _ids_template(_CROSS_ENTROPY_TMPL, ids_dtype, "cross_entropy")


def cross_entropy_backward_source(ids_dtype) -> str:
    return _ids_template(
        _CROSS_ENTROPY_BACKWARD_TMPL, ids_dtype, "cross_entropy_backward"
    )


def binary_template(kind: str, scalar_a: bool = False, scalar_b: bool = False) -> str:
    """手写模板——**对照物**，不参与生产。

    5.42 之后生产路径走 ``binary_source`` 的 L2 描述；这里保留一份逐字等价的
    手写版本，测试拿它当基准：L2 打印器一旦漂移，去空白后的 diff 就会失败。
    等 L2 覆盖全部算子并稳定一轮，这份对照物可以删。
    """
    a_ref = "%LD%(a[0])" if scalar_a else "%LD%(a[i])"
    b_ref = "%LD%(b[0])" if scalar_b else "%LD%(b[i])"
    src = _BINARY_TMPL
    src = src.replace("%A_DECL%", "const %T%* __restrict__ a")
    src = src.replace("%B_DECL%", "const %T%* __restrict__ b")
    src = src.replace("%EXPR%", BINARY_FN[kind].format(a_ref, b_ref))
    return src


def binary_source(kind: str, scalar_a: bool = False, scalar_b: bool = False) -> Kernel:
    """逐元素二元内核的 L2 描述（生产路径，5.42）。

    ``scalar_a`` / ``scalar_b`` 表示对应侧按标量处理（用于 (1,) 广播）。
    这不是"优化"，是语义需要：Transformer 里的 ``x * scale`` 就是标量广播。

    标量一侧仍然收**设备指针**（kernel 内取 ``a[0]``），而不是主机 float：
    传主机 float 需要先 ``float(array)`` 把值拷回 CPU，那是一次强制同步，
    既拖慢每次 execute，也让整张图无法被 CUDA Graph 捕获。
    """
    return ew_binary(kind, scalar_a=scalar_a, scalar_b=scalar_b)


def reduce_sum_source() -> Kernel:
    """末维求和的 L2 描述（生产路径，5.42）。"""
    return reduce_sum_last()


def reduce_sum_rows_source() -> Kernel:
    """窄末维求和的 L2 描述：一个 block 打包多行（每行一个 warp 宽）。

    只在 ``cols <= 32`` 时启用（见 ``device_reduce._reduce_last_dim``）：
    这种形状下行数往往上百万，"一 block 一行"的调度开销远大于算术。
    """
    return reduce_sum_rows()


def reduce_max_source() -> Kernel:
    """末维最大值的 L2 描述（生产路径）。"""
    return reduce_max_last()


def reduce_mean_source() -> Kernel:
    """末维均值的 L2 描述（生产路径）。"""
    return reduce_mean_last()


def reduce_any_source() -> Kernel:
    """末维逻辑或的 L2 描述：bool 输入，求和后判非零（5.341）。

    输出也是 bool，写回由 ``af_st_q`` 归一成 0/1——``render`` 按 dtype 自动
    选 1 字节容器的方言，调用点只需要把 bool 张量传进来。
    """
    return reduce_any_last()


def reduce_all_source() -> Kernel:
    """末维逻辑与的 L2 描述：bool 输入，和等于列数才算全真（5.341）。"""
    return reduce_all_last()


def reduce_axis_source(kind: str, pred: str | None = None) -> Kernel:
    """非末维单轴归约的 L2 描述：一个线程一个输出元素（5.153）。

    ``kind`` 取 ``sum`` / ``max`` / ``min`` / ``mean``，四者的内核骨架相同，
    只有初值、合并方式与是否除以长度不同。``pred`` 是 bool 归约的判定方式
    （``any`` / ``all``，见 5.341），与 ``reduce_any_source`` 同一口径。
    """
    return reduce_axis(kind, pred)


def reduce_axis_tiled_source(kind: str, pred: str | None = None) -> Kernel:
    """非末维归约的两维铺法：x 沿 inner、y 切 reduce 轴（5.174）。"""
    return reduce_axis_tiled(kind, pred)


def softmax_axis_source() -> Kernel:
    """非末维 softmax 的 L2 描述：一个线程一行，三遍扫描（5.153）。"""
    return softmax_axis()


def argmax_source() -> Kernel:
    """末维 argmax 的 L2 描述（生产路径）。

    输出参数是 ``int*``（i32），不随输入 dtype 变化——下标不是浮点激活值。
    """
    return argmax_last()


def argmin_source() -> Kernel:
    """末维 argmin 的 L2 描述（生产路径，5.349）。

    与 ``argmax_source`` 同构，只换比较方向；并列同样取第一次出现。
    """
    return argmin_last()


def logsumexp_source() -> Kernel:
    """任意单轴 logsumexp 的 L2 描述（5.349）。

    铺法与 ``softmax_axis`` 相同（一个线程扫完一行），输出比输入少一维，
    每行一个标量。
    """
    return logsumexp_axis()


def topk_source() -> str:
    """末维 topk 的手写内核源码（values / indices / mask 三个入口共用）。

    没有做成 L2 描述：每轮需要一段运行期长度（``k``）的动态共享内存，
    而 L2 的 ``Shared`` 只能声明静态大小。选择规则与 ``argmax_last`` 一致
    （最大值、并列取小下标），NaN 按 +inf 参与比较。
    """
    return TOPK_SRC


def broadcast_binary_source(kind: str) -> Kernel:
    """支持 numpy 多维广播的逐元素二元内核的 L2 描述（生产路径）。

    广播通过"对齐后的 stride"实现：被广播的维度 stride 置 0，
    于是同一份输入元素会被多个输出位置复用。维度上限 4，
    超过时由后端拒绝（而不是悄悄换实现）。

    ``pow2`` 的掩码/右移变体已删除：V100 上实测反而更慢，理由见
    ``ew_broadcast`` 与 DESIGN 5.300。
    """
    return ew_broadcast(kind)


def compare_source(
    kind: str,
    *,
    scalar_a: bool = False,
    scalar_b: bool = False,
    broadcast: bool = False,
    dtype=None,
    dialect=None,
) -> str:
    """比较内核源码：输出容器固定是 1 字节 bool。

    ``broadcast`` 为假表示两侧同形（扁平网格），为真表示走多维广播版。
    ``dtype`` 是**输入**的存储类型：参数声明、读取 helper
    与 prelude 都由它决定，输出则钉死成 bool 并裸写（比较结果本身就是 0/1）。
    bool 输入（``ne(mask, 0)``）需要 1 字节容器的方言，因此单独走
    ``BOOL_DIALECT``。
    """
    if not broadcast:
        template = ew_binary(kind, scalar_a=scalar_a, scalar_b=scalar_b, compare=True)
    else:
        template = ew_broadcast(kind, compare=True)
    ir_dtype = _ir_dtype(np.dtype("float32") if dtype is None else dtype)
    if dialect is None:
        if ir_dtype is DType.BOOL:
            from .dialect import BOOL_DIALECT

            dialect = BOOL_DIALECT
        else:
            from .dialect import CUDA_DIALECT

            dialect = CUDA_DIALECT
    return emit(template, dialect, ir_dtype)


def where_source(*, broadcast: bool = False, dtype=None, dialect=None) -> str:
    """``where(cond, a, b)`` 源码。

    ``broadcast`` 为假表示三者同形（扁平网格），为真表示三输入广播版。``dtype``
    是 a / b 的存储类型，**输出跟随它**：浮点时正常走 ``af_st_*``；两侧本身
    就是 bool（掩码之间选择）时输出也是 1 字节，此时裸写并换用
    ``BOOL_DIALECT``。``cond`` 无论如何都是 1 字节 bool，裸索引读取。
    """
    ir_dtype = _ir_dtype(np.dtype("float32") if dtype is None else dtype)
    is_bool = ir_dtype is DType.BOOL
    if not broadcast:
        template = ew_where_same(raw_store=is_bool)
    else:
        template = ew_where_bcast(raw_store=is_bool)
    if dialect is None:
        if is_bool:
            from .dialect import BOOL_DIALECT

            dialect = BOOL_DIALECT
        else:
            from .dialect import CUDA_DIALECT

            dialect = CUDA_DIALECT
    return emit(template, dialect, ir_dtype)


def broadcast_template(kind: str) -> str:
    """手写模板——对照物，不参与生产（同 ``binary_template``）。"""
    return _BCAST_TMPL.replace(
        "%EXPR%", BINARY_FN[kind].format("%LD%(a[oa])", "%LD%(b[ob])")
    )


def unary_source(kind: str) -> Kernel:
    """逐元素一元内核的 L2 描述（生产路径）。"""
    return ew_unary(kind)


def cast_fn(src, dst, dialect=None) -> str:
    """跨存储宽度的转换源码（f32 / f16 / bf16 三者任意方向）。

    这是唯一一个"两侧类型不一致"的内核：读取按**源** dtype 渲染（决定用
    ``af_ld_f`` / ``af_ld_h`` / ``af_ld_b``），写回的目标类型由张量参数自己
    声明。方言换成 ``CAST_DIALECT``，让它同时带上三套存取 helper。HIP 侧传入
    自己的一份（f16 的 helper 换成 clang 版），L2 描述与其余逻辑逐字共用。
    """
    if dialect is None:
        from .dialect import CAST_DIALECT as dialect

    src_key, dst_key = _dtype_key(src), _dtype_key(dst)
    if src_key == dst_key:
        raise ValueError(f"cast_fn 只处理跨宽度转换，收到 {src_key} -> {dst_key}")
    out_type = _STORAGE[dst_key][0]
    # f32 是恒等 store，裸赋值即可；窄类型必须走各自的 RNE store helper。
    out_store = None if dst_key == "f32" else _STORAGE[dst_key][2]
    return emit(ew_cast(out_type, out_store), dialect, _ir_dtype(src))


def unary_template(kind: str) -> str:
    """手写模板——对照物，不参与生产。"""
    return _UNARY_TMPL.replace("%EXPR%", UNARY_EXPR[kind])


def step_source() -> Kernel:
    """阶跃内核的 L2 描述（生产路径）。"""
    return ew_step()


def step_template() -> str:
    """手写模板——对照物，不参与生产。"""
    return STEP_SRC


def clamp_source() -> Kernel:
    """按标量上下界截断的 L2 描述（生产路径）。"""
    return ew_clamp()


def mod_source(*, scalar_a: bool = False, scalar_b: bool = False) -> Kernel:
    """i32 floor 取模的 L2 描述（生产路径）。

    整数参数在 L2 里用 ``type_name="int"`` 钉死，不跟随渲染 dtype；调用方
    传 f32 只是为了拿到方言 prelude（见 ``ew_mod_int`` 的说明）。
    """
    return ew_mod_int(scalar_a=scalar_a, scalar_b=scalar_b)


def softmax_source() -> Kernel:
    """末维 softmax 的 L2 描述（生产路径）。"""
    return softmax_last()


def softmax_template() -> str:
    """手写模板——对照物，不参与生产。"""
    return SOFTMAX_SRC


def layernorm_source() -> Kernel:
    """末维 LayerNorm 的 L2 描述（生产路径）。"""
    return layernorm_last()


def layernorm_template() -> str:
    """手写模板——对照物，不参与生产。"""
    return LAYERNORM_SRC


def layernorm_residual_source() -> Kernel:
    """融合残差的末维 LayerNorm 的 L2 描述（生产路径）。"""
    return layernorm_residual_last()


def layernorm_residual_template() -> str:
    """手写模板——对照物，不参与生产。"""
    return LAYERNORM_RESIDUAL_SRC


def rmsnorm_source() -> Kernel:
    """末维 RMSNorm 的 L2 描述（生产路径）。"""
    return rmsnorm_last()


def rmsnorm_template() -> str:
    """手写模板——对照物，不参与生产。"""
    return RMSNORM_SRC


def silu_gate_source() -> Kernel:
    """门控 SiLU 的 L2 描述（生产路径）。"""
    return silu_gate()


def silu_gate_template() -> str:
    """手写模板——对照物，不参与生产。"""
    return SILU_GATE_SRC


def batch_norm_source() -> Kernel:
    """训练态批归一化的 L2 描述（生产路径）。"""
    return batch_norm()


def batch_norm_stats_source() -> Kernel:
    """批归一化统计量的 L2 描述（生产路径）。"""
    return batch_norm_stats()


def batch_norm_template() -> str:
    """手写模板——对照物，不参与生产。

    这一份字符串里有 ``batch_norm`` 与 ``batch_norm_stats`` 两个内核（原先
    一起交给 NVRTC 编译），拆成两个 ``Kernel`` 后对照时按名字截取。
    """
    return BATCH_NORM_SRC


def concat_copy_source() -> Kernel:
    """拼接 / 补零的分段拷贝的 L2 描述（生产路径）。"""
    return concat_copy()


def concat_copy_template() -> str:
    """手写模板——对照物，不参与生产。"""
    return CONCAT_COPY_SRC


def dropout_mask_source() -> Kernel:
    """缩放 dropout 掩码的 L2 描述（生产路径）。"""
    return dropout_mask()


def sparse_linear_source() -> Kernel:
    """事件驱动稀疏线性的 L2 描述（生产路径）。

    没有手写模板对照物：内核只做一次 CSR 段内的加权求和，形状与语义都
    足够简单，再抄一份 C 只会多一处会漂移的事实。
    """
    return sparse_linear()


def spike_offsets_source() -> Kernel:
    """脉冲压缩第一趟（统计每行非零数并写起点表）的 L2 描述。"""
    return spike_offsets()


def spike_csr_source() -> Kernel:
    """脉冲压缩第二趟（按起点表写列号）的 L2 描述。"""
    return spike_csr()


def spike_csr_serial_source() -> Kernel:
    """同一趟的旧铺法：一个线程一行、沿列号串行（仅 5.176 的基准对照）。"""
    return spike_csr_serial()


def sparse_linear_weight_grad_source() -> Kernel:
    """事件驱动稀疏线性对权重梯度的 L2 描述（生产路径）。"""
    return sparse_linear_weight_grad()


def sparse_linear_values_grad_source() -> Kernel:
    """事件驱动稀疏线性对事件幅度梯度的 L2 描述（生产路径）。

    与权重梯度相反，这里按步并行、逐事件算点积，不需要原子加——事件之间
    不存在重复下标。没有手写模板对照物：骨架就是"一段事件 + 一段列号"，
    与 ``sparse_linear`` 的前向同形。
    """
    return sparse_linear_values_grad()


def spike_linear_source() -> Kernel:
    """融合事件线性（脉冲值直接从 CSR 结构读）的 L2 描述（生产路径）。

    没有手写模板对照物：它与 ``sparse_linear`` 共用同一套逐事件累加骨架，
    差别只在事件幅度取自脉冲矩阵本身。
    """
    return spike_linear()


def spike_linear_weight_grad_source() -> Kernel:
    """融合事件线性对权重梯度的 L2 描述（生产路径）。"""
    return spike_linear_weight_grad()


def dropout_mask_template() -> str:
    """手写模板——对照物，不参与生产。"""
    return DROPOUT_MASK_SRC


def dropout_mask_keyed_template() -> str:
    """运行期 key 版的手写模板（捕获后仍能推进掩码）。"""
    return DROPOUT_MASK_KEYED_SRC


def scan_linear_source() -> Kernel:
    """线性递推的 L2 描述（生产路径）。"""
    return scan_linear()


def scan_linear_template() -> str:
    """手写模板——对照物，不参与生产。"""
    return SCAN_LINEAR_SRC


def scan_linear_backward_source() -> Kernel:
    """线性递推反向的 L2 描述（生产路径）。"""
    return scan_linear_backward()


def scan_linear_backward_template() -> str:
    """手写模板——对照物，不参与生产。"""
    return SCAN_LINEAR_BACKWARD_SRC


def scan_linear_chunk_sum_source() -> Kernel:
    """分块扫描第一趟（块内复合）的 L2 描述。

    这些内核没有手写对照模板：它们不是 ``scan_linear`` 的逐字等价物
    （块间前缀的舍入路径不同），"逐 token 相等"的判据在这里不成立。
    验收方式是数值对拍与开关默认关闭时逐位不变。
    """
    return scan_linear_chunk_sum()


def scan_linear_chunk_source() -> Kernel:
    """分块扫描第二趟（块前缀 + 块内读出）的 L2 描述。"""
    return scan_linear_chunk()


def scan_linear_chunk_sum_backward_source() -> Kernel:
    """分块反向第一趟的 L2 描述。"""
    return scan_linear_chunk_sum_backward()


def scan_linear_chunk_backward_source() -> Kernel:
    """分块反向第二趟的 L2 描述。"""
    return scan_linear_chunk_backward()


def scan_linear_last_source() -> Kernel:
    """只写末状态的 L2 描述。"""
    return scan_linear_last()


def scan_linear_chunk_last_source() -> Kernel:
    """分块末状态的 L2 描述。"""
    return scan_linear_chunk_last()


def cumsum_source() -> Kernel:
    """前缀和的 L2 描述（生产路径）。

    没有手写模板对照物：它与 ``scan_linear`` 的铺法同构（一个线程一个尾元素
    沿时间轴走），差别只在没有 ``a_t`` 乘法，抄一份模板只会多一处漂移点。
    """
    return cumsum()


def cumprod_source() -> Kernel:
    """前缀积的 L2 描述（5.349）。

    与 ``cumsum_source`` 同一份铺法，只有累加器换成累乘、初值换成 1。
    """
    return cumprod()


def flip_source() -> Kernel:
    """单轴反转的 L2 描述（5.349）：一个线程搬一个元素。"""
    return flip()


def roll_source() -> Kernel:
    """单轴循环移位的 L2 描述（5.349）：一个线程搬一个元素，``shift`` 由 host 归一化。"""
    return roll()


def repeat_interleave_source() -> Kernel:
    """单轴逐元素重复的 L2 描述（5.350）：输出轴长是输入的 ``repeats`` 倍。"""
    return repeat_interleave()


def tril_source() -> Kernel:
    """下三角掩码的 L2 描述（生产路径）。

    没有手写模板对照物：单趟逐元素写回，结构由 ``grid_stride`` 定死。
    """
    return tril()


def conv2d_input_grad_source() -> Kernel:
    """卷积输入梯度的 L2 描述（生产路径）。"""
    return conv2d_input_grad()


def conv2d_input_grad_template() -> str:
    """手写模板——对照物，不参与生产。"""
    return CONV2D_INPUT_GRAD_SRC


def conv2d_weight_grad_source() -> Kernel:
    """卷积权重梯度的 L2 描述（生产路径）。"""
    return conv2d_weight_grad()


def conv2d_weight_grad_template() -> str:
    """手写模板——对照物，不参与生产。"""
    return CONV2D_WEIGHT_GRAD_SRC


def maxpool2d_source(layout: str = "nchw") -> Kernel:
    """2D 最大池化的 L2 描述（生产路径）。"""
    return maxpool2d(layout=layout)


def maxpool2d_template() -> str:
    """手写模板——对照物，不参与生产。"""
    return MAXPOOL2D_SRC


def maxpool2d_backward_source() -> Kernel:
    """2D 最大池化反向的 L2 描述（生产路径）。"""
    return maxpool2d_backward()


def maxpool2d_indices_source() -> Kernel:
    """2D 最大池化取值位置的 L2 描述（i32 平面内下标，生产路径）。"""
    return maxpool2d_indices()


def maxpool2d_backward_template() -> str:
    """手写模板——对照物，不参与生产。"""
    return MAXPOOL2D_BACKWARD_SRC


def maxpool2d_indices_template() -> str:
    """手写模板——对照物，不参与生产。"""
    return MAXPOOL2D_INDICES_SRC


def avgpool2d_source(layout: str = "nchw") -> Kernel:
    """2D 平均池化的 L2 描述（生产路径）。"""
    return avgpool2d(layout=layout)


def avgpool2d_template() -> str:
    """手写模板——对照物，不参与生产。"""
    return AVGPOOL2D_SRC


def avgpool2d_backward_source() -> Kernel:
    """2D 平均池化反向的 L2 描述（生产路径）。"""
    return avgpool2d_backward()


def avgpool2d_backward_template() -> str:
    """手写模板——对照物，不参与生产。"""
    return AVGPOOL2D_BACKWARD_SRC


def adaptive_avg_pool2d_source() -> Kernel:
    """自适应平均池化的 L2 描述（生产路径）。"""
    return adaptive_avg_pool2d()


def adaptive_avg_pool2d_template() -> str:
    """手写模板——对照物，不参与生产。"""
    return ADAPTIVE_AVGPOOL2D_SRC


def adaptive_avg_pool2d_backward_source() -> Kernel:
    """自适应平均池化反向的 L2 描述（生产路径）。"""
    return adaptive_avg_pool2d_backward()


def adaptive_avg_pool2d_backward_template() -> str:
    """手写模板——对照物，不参与生产。"""
    return ADAPTIVE_AVGPOOL2D_BACKWARD_SRC


def interpolate_nearest_source() -> Kernel:
    """最近邻插值的 L2 描述（生产路径）。"""
    return interpolate_nearest()


def interpolate_bilinear_source(align_corners: bool = False) -> Kernel:
    """双线性插值的 L2 描述（生产路径）；``align_corners`` 决定坐标公式。"""
    return interpolate_bilinear(align_corners=align_corners)


def interpolate_nearest_backward_source() -> Kernel:
    """最近邻插值反向的 L2 描述（生产路径）。"""
    return interpolate_nearest_backward()


def interpolate_bilinear_backward_source(align_corners: bool = False) -> Kernel:
    """双线性插值反向的 L2 描述（生产路径）。"""
    return interpolate_bilinear_backward(align_corners=align_corners)


def im2col_source() -> Kernel:
    """卷积展开的 L2 描述（生产路径）。"""
    return im2col()


def im2col_template() -> str:
    """手写模板——对照物，不参与生产。"""
    return IM2COL_SRC


def conv2d_tiled_source(layout: str = "nchw") -> Kernel:
    """直接分块卷积的 L2 描述（生产路径）。

    这个内核不做 dtype 参数化（全程 ``float``），模板里的 ``%T%`` 从未出现，
    因此两种 dtype 打印出的源码相同。``layout="nhwc"`` 只改寻址公式
    （通道在末维，见 5.334），分块几何与累加顺序不变。
    """
    return conv2d_tiled(layout=layout)


def conv2d_tiled_template() -> str:
    """手写模板——对照物，不参与生产。"""
    return CONV2D_TILED_SRC


def conv2d_grouped_source() -> Kernel:
    """分组 / depthwise 卷积的 L2 描述（生产路径，5.165）。

    与分块核、隐式核一样不做 dtype 参数化：全程 ``float``，f16 走 im2col。
    """
    return conv2d_grouped()


def conv2d_implicit_source() -> Kernel:
    """隐式 GEMM 卷积的 L2 描述：输入窗口也进 shared。

    与 :func:`conv2d_tiled_source` 一样不做 dtype 参数化，模板里的 ``%T%``
    不出现，两种 dtype 打印出的源码相同。
    """
    return conv2d_implicit()


def winograd_weight_source() -> Kernel:
    """Winograd F(2x2,3x3) 的权重变换 ``U = G g G^T``（生产路径）。

    三个 Winograd 内核都只做 f32：变换的中间量（``1/2`` 缩放后的和）没有
    对应的 f16 写法，且 f16 卷积已经走 im2col + hgemm。
    """
    return winograd_weight_transform()


def winograd_input_source() -> Kernel:
    """Winograd F(2x2,3x3) 的输入变换 ``V = B^T d B``（生产路径）。"""
    return winograd_input_transform()


def winograd_output_source(epilogue=None, extra=()) -> Kernel:
    """Winograd F(2x2,3x3) 的输出变换 ``Y = A^T M A`` + bias（生产路径）。

    ``epilogue`` 非空时把后继逐元素链（BN / 激活）内联进写回点：``Y`` 不再
    落一次全局内存再被链读回来。默认 ``None`` 生成的源码逐字不变。
    """
    return winograd_output_transform(epilogue=epilogue, extra=extra)


def winograd_fused_source(epilogue=None, extra=()) -> Kernel:
    """融合版 Winograd：输入变换 + 乘加 + 输出变换，一次 launch。

    与上面三个内核只差"在哪算"：``V`` 不进全局缓冲而是留在共享内存，
    ``M`` 不进全局缓冲而是留在寄存器。数值与四段版逐位一致（同一组
    ``B`` / ``A`` / ``G`` 系数、同样的求和顺序）。
    """
    return winograd_fused(epilogue=epilogue, extra=extra)


def attention_source() -> Kernel:
    """融合注意力前向的 L2 描述（生产路径）。"""
    return attention_fwd()


def attention_template() -> str:
    """手写模板——对照物，不参与生产。"""
    return ATTENTION_SRC
