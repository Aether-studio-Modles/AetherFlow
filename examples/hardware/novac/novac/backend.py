"""Nova C 示例后端：第三方接入的全部代码。

Nova C 是一个虚构的加速器（与 CUDA 共用一套 C 语法），用来演示"换一块硬件
只需要填方言"。这里没有为框架改任何一行：L2 内核 IR 与打印器都在
``aetherflow.lower`` 里，硬件包只提供**数据**与**胶水**：

- 数据：存储类型、存取函数名、辅助函数源码（``NOVAC_DIALECT``）；
- 胶水：一个只有 ``name`` / ``capabilities`` / ``execute`` 的后端类。

真实硬件要把自己的"编译 → 加载 → 启动"接到 ``execute``；示例用参考执行器
代替，因此没有该硬件的机器也能完整跑通"安装 → 识别 → 出源码 → 跑图"。

**为什么辅助函数叫 ``block_reduce_sum`` 而不是 ``nv_block_reduce_sum``**：
归约辅助函数名是 L2 词汇表的一部分（``lower/kernels.py`` 里写死了调用），
方言只决定它们的实现。这条边界在接入真实硬件时同样成立——要么沿用这些
名字，要么先在 L2 里把名字参数化。
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Mapping

import numpy as np

from aetherflow.ir.types import DType
from aetherflow.lower import CStyleDialect, emit

__all__ = ["NOVAC_DIALECT", "NovaCBackend", "nova_source"]

# 归约辅助函数：名字由 L2 决定，实现在方言语境里编译。
_NOVAC_REDUCE = r"""
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

// Reduce over threadIdx.y for each fixed threadIdx.x: the strided
// (outer, reduce, inner) reductions split the reduce axis across threadIdx.y.
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

# f32 的四个辅助函数是恒等映射；f16 存储为 unsigned short，算术在 float 里做。
# NOTE: 交给设备编译器的源码必须是 ASCII，所以 raw string 里的注释也用英文。
_NOVAC_F32_HELPERS = r"""
__device__ __forceinline__ float nv_ld_f(float v) { return v; }
__device__ __forceinline__ float nv_st_f(float v) { return v; }
__device__ __forceinline__ float nv_add_f(float* p, float v) { return atomicAdd(p, v); }
"""

# 假定 Nova C 的工具链提供 __half 与 cuda_fp16.h 同名的转换函数。
_NOVAC_F16_HELPERS = r"""
__device__ __forceinline__ float nv_ld_h(unsigned short h) {
    return __half2float(__ushort_as_half(h));
}
__device__ __forceinline__ unsigned short nv_st_h(float v) {
    return __half_as_ushort(__float2half_rn(v));
}
__device__ __forceinline__ float nv_add_h(unsigned short* p, float v) {
    unsigned short old = *p;
    unsigned short assumed;
    do {
        assumed = old;
        old = atomicCAS(p, assumed, nv_st_h(nv_ld_h(assumed) + v));
    } while (assumed != old);
    return nv_ld_h(old);
}
"""


NOVAC_DIALECT = CStyleDialect(
    "novac",
    storage={DType.F32: "float", DType.F16: "unsigned short"},
    load={DType.F32: "nv_ld_f", DType.F16: "nv_ld_h"},
    store={DType.F32: "nv_st_f", DType.F16: "nv_st_h"},
    helpers={
        DType.F32: _NOVAC_F32_HELPERS,
        DType.F16: _NOVAC_F16_HELPERS,
    },
    reduce_helpers=_NOVAC_REDUCE,
)


def _ir_dtype(dtype) -> DType:
    name = str(getattr(dtype, "name", dtype))
    if name in ("float32", "f32"):
        return DType.F32
    if name in ("float16", "f16"):
        return DType.F16
    raise ValueError(f"Nova C 只支持 f32/f16，收到 {name}")


def nova_source(kernel, dtype="float32") -> str:
    """把 L2 内核打印成 Nova C 源码。"""
    return emit(kernel, NOVAC_DIALECT, _ir_dtype(dtype))


class NovaCBackend:
    """最小后端：能力声明 + 执行入口，其余都是方言数据。"""

    name = "novac"

    def __init__(self, **kwargs) -> None:
        self.options = dict(kwargs)

    def capabilities(self) -> FrozenSet[str]:
        # 示例借用参考后端的能力集合：能跑的算子就是 IR 里可移植的那些。
        from aetherflow.backends.reference_backend import ReferenceBackend

        return ReferenceBackend().capabilities()

    def source_for(self, kernel, dtype="float32") -> str:
        """把一个 L2 内核编译成 Nova C 源码（真实硬件在这里交给驱动）。"""
        return nova_source(kernel, dtype)

    def execute(
        self, graph, inputs: Mapping[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        # 真实硬件的接入点：在这里编译内核、分配设备内存、启动 grid。
        # 示例没有驱动，用参考执行器保证"装上就能跑"。
        from aetherflow.backends.reference_backend import ReferenceBackend

        return ReferenceBackend().execute(graph, inputs)
