"""CUDA 方言：C 语法 + bf16 / f16 的存取（f16 走 PTX，bf16 走按位）。

方言是**数据**：类型映射、存取函数名、辅助函数源码。打印逻辑在
``aetherflow/lower/cstyle.py``，因此新增一家硬件只需要在这里换数据，
不用碰 L2 的节点，也不用碰打印器。
"""

from __future__ import annotations

from ...ir.types import DType
from ...lower.cstyle import CStyleDialect
from . import codegen

__all__ = ["CUDA_DIALECT", "CAST_DIALECT", "BOOL_DIALECT"]

CUDA_DIALECT = CStyleDialect(
    "cuda",
    storage={
        DType.F32: "float",
        DType.F16: "unsigned short",
        DType.BF16: "unsigned short",
    },
    load={DType.F32: "af_ld_f", DType.F16: "af_ld_h", DType.BF16: "af_ld_b"},
    store={DType.F32: "af_st_f", DType.F16: "af_st_h", DType.BF16: "af_st_b"},
    atomic={DType.F32: "af_add_f", DType.F16: "af_add_h", DType.BF16: "af_add_b"},
    helpers={
        DType.F32: codegen._F32_HELPERS + codegen._EXPR_HELPERS,
        DType.F16: codegen._F16_HELPERS + codegen._EXPR_HELPERS,
        DType.BF16: codegen._BF16_HELPERS + codegen._EXPR_HELPERS,
    },
    reduce_helpers=codegen._REDUCE,
)

# ``cast`` 是唯一"输入与输出存储类型不同"的内核：按源 dtype 渲染决定用哪一套
# 读取 helper，写回的目标类型由张量参数自己声明。因此它需要所有可能用到的
# helper 同时在场（f32 / f16 / bf16 任意方向）。只给这一个内核用，不并进
# ``CUDA_DIALECT``——那会让所有内核的源码多出几十行，而 5.42 起有"生产源码
# 与手写模板逐 token 相等"的对照用例盯着每一个字。
_CAST_HELPERS = (
    codegen._F32_HELPERS
    + codegen._F16_HELPERS
    + codegen._BF16_HELPERS
    # 表达式级 helper（af_rem）与 dtype 无关：三块拼起来只追加一次，
    # 否则同一个生成单元里会重复定义。
    + codegen._EXPR_HELPERS
)

CAST_DIALECT = CStyleDialect(
    "cuda",
    storage=CUDA_DIALECT.storage,
    load=CUDA_DIALECT.load,
    store=CUDA_DIALECT.store,
    atomic=CUDA_DIALECT.atomic,
    helpers={
        DType.F32: _CAST_HELPERS,
        DType.F16: _CAST_HELPERS,
        DType.BF16: _CAST_HELPERS,
    },
    reduce_helpers=CUDA_DIALECT.reduce_helpers,
)

# 比较这类"输入数值、输出 bool"的内核同时需要两套 helper：源 dtype 的读取
# 与 bool 的读出/写回。与 CAST_DIALECT 同理，只给这一族内核用，不并进
# CUDA_DIALECT——否则每个普通内核都会多带一份用不到的 bool 存取。
_BOOL_HELPERS = _CAST_HELPERS + codegen._BOOL_HELPERS

BOOL_DIALECT = CStyleDialect(
    "cuda",
    storage={**CUDA_DIALECT.storage, DType.BOOL: "unsigned char"},
    load={**CUDA_DIALECT.load, DType.BOOL: "af_ld_q"},
    store={**CUDA_DIALECT.store, DType.BOOL: "af_st_q"},
    atomic={**CUDA_DIALECT.atomic, DType.BOOL: "af_add_q"},
    helpers={
        DType.F32: _BOOL_HELPERS,
        DType.F16: _BOOL_HELPERS,
        DType.BF16: _BOOL_HELPERS,
        DType.BOOL: _BOOL_HELPERS,
    },
    reduce_helpers=CUDA_DIALECT.reduce_helpers,
)
