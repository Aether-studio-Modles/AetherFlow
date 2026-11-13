"""MLP 构件。

结构上没有任何新机制——它就是 ``matmul + add + 激活`` 的堆叠。放进模型层
的理由是"完整"：一个只能表达 Transformer 的 IR 无法回答"最简单的模型能不能
自然表达"，而恰恰是 MLP 暴露了两个容易被忽略的约定：

- 权重布局是 ``(in, out)``，与 ``x @ W`` 一致，而不是 PyTorch 的 ``(out, in)``；
  这个选择贯穿全项目，混用会让所有形状检查在语义上对但数值上错。
- 偏置用 ``add`` 的尾部广播，不需要 ``broadcast_to``：广播规则本身就是
  前端能力的一部分，能少一次物化就少一次。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from ..ir.graph import Graph, Value

__all__ = ["MLPSpec", "build_mlp"]

_ACTIVATIONS = ("relu", "silu", "gelu", "tanh")


@dataclass(frozen=True)
class MLPSpec:
    """``sizes`` 是各层宽度，首项为输入特征数、末项为输出特征数。"""

    sizes: Tuple[int, ...] = (32, 64, 32, 10)
    activation: str = "gelu"
    output_activation: str = ""
    batch: str = "B"

    def __post_init__(self) -> None:
        if len(self.sizes) < 2:
            raise ValueError(f"MLP 至少需要输入与输出两层，得到 {self.sizes}")
        if any(int(s) <= 0 for s in self.sizes):
            raise ValueError(f"各层宽度必须为正，得到 {self.sizes}")
        if self.activation not in _ACTIVATIONS:
            raise ValueError(
                f"未知激活 {self.activation!r}；可用: {', '.join(_ACTIVATIONS)}"
            )
        if self.output_activation and self.output_activation not in _ACTIVATIONS:
            raise ValueError(f"未知输出激活 {self.output_activation!r}")

    @property
    def in_features(self) -> int:
        return int(self.sizes[0])

    @property
    def out_features(self) -> int:
        return int(self.sizes[-1])

    @property
    def param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        shapes: Dict[str, Tuple[int, ...]] = {}
        for i in range(len(self.sizes) - 1):
            shapes[f"w{i}"] = (int(self.sizes[i]), int(self.sizes[i + 1]))
            shapes[f"b{i}"] = (int(self.sizes[i + 1]),)
        return shapes


def build_mlp(
    g: Graph, spec: MLPSpec | None = None
) -> Tuple[Dict[str, Value], Value]:
    """构建 MLP，返回 ``(输入表, 输出 Value)``。输入表含 ``x`` 与全部参数。"""
    from ..ir.types import f32

    spec = spec or MLPSpec()
    inputs: Dict[str, Value] = {
        "x": g.input("x", f32[spec.batch, spec.in_features])
    }
    for name, shape in spec.param_shapes.items():
        inputs[name] = g.input(name, f32[shape])

    h = inputs["x"]
    last = len(spec.sizes) - 2
    for i in range(last + 1):
        h = g.add(g.matmul(h, inputs[f"w{i}"]), inputs[f"b{i}"])
        if i < last:
            h = g.add_node(spec.activation, [h])
        elif spec.output_activation:
            h = g.add_node(spec.output_activation, [h])
    return inputs, h
