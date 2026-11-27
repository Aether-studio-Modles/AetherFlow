"""HIP C 源码生成：复用 CUDA 的模板，只替换 f16 的存取实现。

模板本身（逐元素 / 广播 / softmax / attention / 卷积 / 池化 / 扫描……）是
纯 C++ 加 ``__global__`` 语义，hipRTC 完全接受——探针把全部 57 个 f32 源
交给 hipRTC 编译，零失败。唯一不通用的是 f16 的 PTX 内联汇编：
``asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h))`` 的约束在 AMD clang 上
直接报 ``invalid output constraint``。

因此这里只做一件事：换掉 PRELUDE 里的三个 f16 辅助函数，其余常量与
构造函数原样透传（``__getattr__``）。这样做的好处是两种后端**共用一份内核
语义**——模板改了，HIP 自动跟上，不存在"CUDA 修了、HIP 忘了"的分叉。
"""

from __future__ import annotations

from ..cuda import codegen as _cuda

# AMD clang 的 _Float16 与 numpy 的 float16 是同一套 IEEE 754 半精度，
# 逐位布局一致，所以 unsigned short 与 _Float16 之间用 __builtin_bit_cast
# 互转即可，不需要 include hip_fp16.h（hipRTC 默认没有 include 路径）。
#
# 16 位原子加仍然用 CAS 循环：HIP 的 atomicAdd 没有 unsigned short 重载
# （探针已确认），atomicCAS 有。
_HIP_F16_HELPERS = r"""
__device__ __forceinline__ float af_ld_h(unsigned short h) {
    return (float)__builtin_bit_cast(_Float16, h);
}
__device__ __forceinline__ unsigned short af_st_h(float v) {
    return __builtin_bit_cast(unsigned short, (_Float16)v);
}
__device__ __forceinline__ float af_add_h(unsigned short* p, float v) {
    unsigned short old = *p;
    unsigned short assumed;
    do {
        assumed = old;
        old = atomicCAS(p, assumed, af_st_h(af_ld_h(assumed) + v));
    } while (assumed != old);
    return af_ld_h(old);
}
"""


def render(template: str, dtype) -> str:
    """展开模板；f16 的读写与原子加换成 HIP 方言。"""
    # 延迟导入：hip.dialect 依赖 cuda.codegen，顶层导入会成环。
    from .dialect import HIP_DIALECT

    return _cuda.render(
        template, dtype, f16_helpers=_HIP_F16_HELPERS, dialect=HIP_DIALECT
    )


def cast_fn(src, dst) -> str:
    """``f32 <-> f16`` 的转换源码。

    L2 描述与 CUDA 逐字共用，只把方言换成 HIP 的那一份：5.33 那批整图 f16 的
    用例都直接声明 f16 输入、从不产生 ``cast`` 节点，所以这条路径以前没人走到。
    """
    # 延迟导入：hip.dialect 顶层就 import 了本模块。
    from .dialect import CAST_DIALECT

    return _cuda.cast_fn(src, dst, dialect=CAST_DIALECT)


def compare_source(kind: str, *, scalar_a: bool = False, scalar_b: bool = False,
                   pow2=None, dtype=None) -> str:
    """比较内核源码：L2 与 CUDA 逐字共用，只换方言。

    ``dtype`` 是输入的存储类型：bool 输入（``ne(mask, 0)``）要 HIP 侧的
    ``BOOL_DIALECT``，其余仍是 ``HIP_DIALECT``（L2 与 CUDA 同一份）。
    """
    # 延迟导入：hip.dialect 顶层就 import 了本模块。
    from .dialect import BOOL_DIALECT, HIP_DIALECT

    is_bool = dtype is not None and _cuda._dtype_key(dtype) == "bool"
    return _cuda.compare_source(
        kind,
        scalar_a=scalar_a,
        scalar_b=scalar_b,
        pow2=pow2,
        dtype=dtype,
        dialect=BOOL_DIALECT if is_bool else HIP_DIALECT,
    )


def where_source(*, pow2=None, dtype=None) -> str:
    """``where`` 源码：L2 与 CUDA 逐字共用，只换方言。

    ``dtype`` 是 a / b（也是输出）的存储类型；bool 两侧（掩码选掩码）需要
    1 字节容器的 ``BOOL_DIALECT``，其余走 ``HIP_DIALECT``。
    """
    # 延迟导入：hip.dialect 顶层就 import 了本模块。
    from .dialect import BOOL_DIALECT, HIP_DIALECT

    is_bool = dtype is not None and _cuda._dtype_key(dtype) == "bool"
    return _cuda.where_source(
        pow2=pow2,
        dtype=dtype,
        dialect=BOOL_DIALECT if is_bool else HIP_DIALECT,
    )


def __getattr__(name: str):
    """未定义的常量/函数一律来自 CUDA 侧的模板库（单一事实来源）。"""
    return getattr(_cuda, name)
