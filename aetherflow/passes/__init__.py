"""中层图优化 IR。

默认管线（无硬件假设）的顺序：``SimplifyShapeOps -> ConstantFolding -> CSE -> DCE``。

    1. 形状与类型噪声消除（语义为空的节点）
    2. 常量折叠（输入全是常量的子图在编译期算掉）
    3. 公共子表达式消除（图上重复的子计算）
    4. 死代码消除（清理前三步留下的无使用者节点）

融合（``fused_pipeline``）单独提供：它会改变算子的输入形态，
需要目标后端承接，因此不放进"无硬件假设"的默认管线。
"""

from .base import GraphRewriter, Pass, PassManager
from .autocast import Autocast
from .cse import CommonSubexpressionElimination
from .dce import DeadCodeElimination
from .fold import ConstantFolding
from .fusion import FuseLayerNormResidual, FuseSiLU, FuseSoftmaxScale, fused_pipeline
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
    "default_pipeline",
    "fused_pipeline",
]


def default_pipeline() -> PassManager:
    """无硬件假设的默认优化管线。"""
    return PassManager(
        [
            SimplifyShapeOps(),
            ConstantFolding(),
            CommonSubexpressionElimination(),
            DeadCodeElimination(),
        ]
    )
