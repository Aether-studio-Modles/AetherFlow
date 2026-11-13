"""中层图优化 IR。

默认管线（无硬件假设）的顺序：``SimplifyShapeOps -> ConstantFolding -> CSE -> DCE``。

    1. 形状与类型噪声消除（语义为空的节点）
    2. 常量折叠（输入全是常量的子图在编译期算掉）
    3. 公共子表达式消除（图上重复的子计算）
    4. 死代码消除（清理前三步留下的无使用者节点）

融合（``fused_pipeline``）单独提供：它会改变算子的输入形态，
需要目标后端承接，因此不放进"无硬件假设"的默认管线。

两条管线的名字序列与"单独跑某个 pass"的入口都收在 ``registry`` 里，
不再由各模块各写一份——名字与顺序只有一处定义。
"""

from .base import GraphRewriter, Pass, PassManager
from .autocast import Autocast
from .cse import CommonSubexpressionElimination
from .dce import DeadCodeElimination
from .fold import ConstantFolding
from .fusion import FuseLayerNormResidual, FuseSiLU, FuseSoftmaxScale
from .registry import (
    DEFAULT_PASS_NAMES,
    FUSED_PASS_NAMES,
    available_passes,
    build_pipeline,
    default_pipeline,
    fused_pipeline,
    get_pass,
    register_pass,
    run_pass,
    run_passes,
    unregister_pass,
)
from .simplify import SimplifyShapeOps

__all__ = [
    "Pass",
    "PassManager",
    "GraphRewriter",
    "Autocast",
    "DeadCodeElimination",
    "CommonSubexpressionElimination",
    "SimplifyShapeOps",
    "ConstantFolding",
    "FuseSoftmaxScale",
    "FuseLayerNormResidual",
    "FuseSiLU",
    "DEFAULT_PASS_NAMES",
    "FUSED_PASS_NAMES",
    "available_passes",
    "build_pipeline",
    "default_pipeline",
    "fused_pipeline",
    "get_pass",
    "register_pass",
    "run_pass",
    "run_passes",
    "unregister_pass",
]
