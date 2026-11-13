"""CUDA 方言：C 语法 + PTX 版 f16 存取。

方言是**数据**：类型映射、存取函数名、辅助函数源码。打印逻辑在
``aetherflow/lower/cstyle.py``，因此新增一家硬件只需要在这里换数据，
不用碰 L2 的节点，也不用碰打印器。
"""

from __future__ import annotations

from ...ir.types import DType
from ...lower.cstyle import CStyleDialect
from . import codegen

__all__ = ["CUDA_DIALECT", "CAST_DIALECT"]

CUDA_DIALECT = CStyleDialect(
    "cuda",
    storage={DType.F32: "float", DType.F16: "unsigned short"},
    load={DType.F32: "af_ld_f", DType.F16: "af_ld_h"},
    store={DType.F32: "af_st_f", DType.F16: "af_st_h"},
    atomic={DType.F32: "af_add_f", DType.F16: "af_add_h"},
    helpers={
        DType.F32: codegen._F32_HELPERS,
        DType.F16: codegen._F16_HELPERS,
    },
    reduce_helpers=codegen._REDUCE,
)

# ``cast`` 是唯一"输入与输出存储类型不同"的内核：按源 dtype 渲染决定用哪一套
# 读取 helper，写回的目标类型由张量参数自己声明。因此它需要两套 helper 同时在场。
# 只给这一个内核用，不并进 ``CUDA_DIALECT``——那会让所有内核的源码多出几十行，
# 而 5.42 起有"生产源码与手写模板逐 token 相等"的对照用例盯着每一个字。
CAST_DIALECT = CStyleDialect(
    "cuda",
    storage=CUDA_DIALECT.storage,
    load=CUDA_DIALECT.load,
    store=CUDA_DIALECT.store,
    atomic=CUDA_DIALECT.atomic,
    helpers={
        DType.F32: codegen._F32_HELPERS + codegen._F16_HELPERS,
        DType.F16: codegen._F16_HELPERS + codegen._F32_HELPERS,
    },
    reduce_helpers=CUDA_DIALECT.reduce_helpers,
)
