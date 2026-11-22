"""前端接入：把外部模型格式降到本 IR，或把构图写得更顺手。

两条路，互不影响：

- ``torch_export``：把别人的模型翻译成 IR。这里的定位是**入口适配器**，
  不是内核依赖：torch 只允许出现在该文件（ADR-005），删除它、卸载 torch 后，
  IR / 运行时 / 各后端必须仍然可用。遇到不认识的算子**直接报错**，不做静默
  fallback。之所以必须有它：大纲里的"存量 CUDA 模型无需改写接入"要求能读别人的图。
- ``eager``：给本框架自己的构图加语法糖（运算符重载、标量自动转常量），
  让手写图写起来像动态图，但产出物仍是同一份静态 IR——不引入任何后端依赖。
"""

from .compat import (
    TORCH_MATRIX,
    TorchCompatibility,
    UnsupportedTorchVersion,
    check_torch,
    matrix_row,
    parse_torch_version,
)
from .eager import EagerGraph, Tensor
from .module import Module, Parameter
from .torch_export import ImportedModel, UnsupportedOperatorError, import_exported

__all__ = [
    "EagerGraph",
    "Module",
    "Parameter",
    "Tensor",
    "ImportedModel",
    "UnsupportedOperatorError",
    "import_exported",
    "TORCH_MATRIX",
    "TorchCompatibility",
    "UnsupportedTorchVersion",
    "check_torch",
    "matrix_row",
    "parse_torch_version",
]
