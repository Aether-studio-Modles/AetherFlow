"""HIP 方言：与 CUDA 同一套 C 语法，只换 f16 存取。

差异的原因见 5.40：AMD 的 clang 不认 PTX 的 ``cvt`` 约束，也不提供
``atomicAdd(unsigned short*)`` 的重载。归约辅助函数与 f32 辅助函数逐字共用
——这也说明"方言"的粒度就是这两处，其余都是同一层语法。
"""

from __future__ import annotations

from ...ir.types import DType
from ...lower.cstyle import CStyleDialect
from ..cuda import codegen as _cuda
from . import codegen

__all__ = ["HIP_DIALECT", "CAST_DIALECT", "BOOL_DIALECT"]

HIP_DIALECT = CStyleDialect(
    "hip",
    storage={DType.F32: "float", DType.F16: "unsigned short"},
    load={DType.F32: "af_ld_f", DType.F16: "af_ld_h"},
    store={DType.F32: "af_st_f", DType.F16: "af_st_h"},
    atomic={DType.F32: "af_add_f", DType.F16: "af_add_h"},
    helpers={
        DType.F32: _cuda._F32_HELPERS,
        DType.F16: codegen._HIP_F16_HELPERS,
    },
    reduce_helpers=_cuda._REDUCE,
)

# 与 CUDA 侧的 ``CAST_DIALECT`` 同构：``cast`` 的输入与输出存储宽度不同，两套
# helper 必须同时在源码里。f16 那一套是 clang 版（AMD 不认 PTX 的 ``cvt``）。
CAST_DIALECT = CStyleDialect(
    "hip",
    storage=HIP_DIALECT.storage,
    load=HIP_DIALECT.load,
    store=HIP_DIALECT.store,
    atomic=HIP_DIALECT.atomic,
    helpers={
        DType.F32: _cuda._F32_HELPERS + codegen._HIP_F16_HELPERS,
        DType.F16: codegen._HIP_F16_HELPERS + _cuda._F32_HELPERS,
    },
    reduce_helpers=HIP_DIALECT.reduce_helpers,
)

# 比较内核同时需要源 dtype 的读取与 bool 的读出/写回（同 CUDA 侧）。
_BOOL_HELPERS = _cuda._F32_HELPERS + codegen._HIP_F16_HELPERS + _cuda._BOOL_HELPERS

BOOL_DIALECT = CStyleDialect(
    "hip",
    storage={**HIP_DIALECT.storage, DType.BOOL: "unsigned char"},
    load={**HIP_DIALECT.load, DType.BOOL: "af_ld_q"},
    store={**HIP_DIALECT.store, DType.BOOL: "af_st_q"},
    atomic={**HIP_DIALECT.atomic, DType.BOOL: "af_add_q"},
    helpers={
        DType.F32: _BOOL_HELPERS,
        DType.F16: _BOOL_HELPERS,
        DType.BOOL: _BOOL_HELPERS,
    },
    reduce_helpers=HIP_DIALECT.reduce_helpers,
)
