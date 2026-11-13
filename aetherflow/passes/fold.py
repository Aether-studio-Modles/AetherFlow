"""常量折叠。

图里只要存在"输入全是常量"的节点，它的结果就是编译期确定的。
在构图期用参考语义把它算掉，后续的 CSE / DCE 面对的节点更少，
后端也不再为一次确定的计算分配缓冲、发起 kernel。

三条边界，都是"折了会改变语义或得不偿失"的情形：

1. **符号维不折**。没有运行期绑定就算不出元素数，硬折会把符号图退化成
   一张只能跑一种形状的图。
2. **带效果的算子不折**（``effects`` 非空，当前是 ``dropout_mask``）。它的
   结果虽然由 seed 完全决定，但语义是"每次训练重新采样"；把掩码固化成常量
   会让同一张训练图永远复现同一份噪声。折叠只对纯函数成立，这是算子表里的
   声明，不是这里维护的白名单。
3. **超大的常量不折**。折叠结果会写进节点属性并随图序列化，超过
   ``max_elements`` 后图就不再"可读可传"，而且这类子图本来就该在别处优化。

求值统一走 ``OPS[name].reference``——它是全项目唯一的真值。因此折叠后的
图与折叠前在参考后端上逐位一致；在其他后端上，折叠结果等价于"参考语义那
一侧"，仍落在允许的数值误差内。

折叠只物化**存活边界**：被非折叠节点引用、或是图输出的那些结果。中间结果
的使用者也都在折叠集合里，物化它们不会减少任何启动次数，反而多出几次主机
到设备的拷贝（5.161 的实测见 ``DESIGN``）。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..ir.graph import Graph, Node
from ..ir.ops import OPS, Portability
from ..ir.types import DType, TensorType
from .base import GraphRewriter, Pass

__all__ = ["ConstantFolding"]


def _constant_array(node: Node) -> Optional[np.ndarray]:
    """把 constant 节点求值成 numpy 数组；形状含符号维时返回 None。"""
    if not node.output.type.is_static:
        return None
    return OPS["constant"].reference([], node.attrs)


def _attrs_for(type_: TensorType, array: np.ndarray) -> dict:
    """把求值结果写成 constant 节点属性。

    数据类型回到 IR 的声明类型（reference 的 numpy dtype 是它的承载类型，
    不是位宽承诺）；浮点值经 ``tolist`` 往返不会改变 f16/f32 的位模式。
    """
    dtype = DType(type_.dtype)
    cast = np.asarray(array, dtype=dtype.np_dtype).reshape(tuple(type_.shape))
    return {"dtype": dtype.value, "shape": list(type_.shape), "data": cast.tolist()}


class ConstantFolding(Pass):
    """把全常量子图算成 ``constant`` 节点。"""

    name = "constant-folding"

    def __init__(self, max_elements: int = 1 << 16) -> None:
        self.max_elements = max_elements

    def run(self, graph: Graph) -> Graph:
        known: Dict[int, np.ndarray] = {}
        folded: Dict[int, dict] = {}

        for node in graph.nodes:
            if node.op.name == "constant":
                value = _constant_array(node)
                if value is not None:
                    known[node.output.id] = value
                continue
            if not self._foldable(node):
                continue
            inputs: List[np.ndarray] = []
            for v in node.inputs:
                array = known.get(v.id)
                if array is None:
                    inputs = []
                    break
                inputs.append(array)
            if not inputs:
                continue

            result = node.op.reference(inputs, node.attrs)
            result = np.asarray(result)
            if tuple(result.shape) != tuple(node.output.type.shape):
                raise AssertionError(
                    f"常量折叠 {node.op.name} 时 reference 返回形状 {result.shape}，"
                    f"与类型推导的 {node.output.type.shape} 不一致"
                )
            cast = np.asarray(result, dtype=DType(node.output.type.dtype).np_dtype)
            known[node.output.id] = cast
            folded[node.output.id] = _attrs_for(node.output.type, cast)

        if not folded:
            return graph

        # 只有被"非折叠节点"引用、或是图输出的折叠结果才物化成 constant。
        # 中间结果的使用者本身也在 folded 里，物化它们只会让节点数不变甚至
        # 变多——5.161 实测过这一版：9 个节点折完还是 9 个，执行期每个中间
        # constant 都要上传一次，反而比原图慢（mask 用例 +161%）。折叠的收益
        # 必须体现为"节点变少"，所以这里只保留存活边界。
        boundary = {value.id for value in graph.outputs}
        for node in graph.nodes:
            if node.output.id in folded:
                continue
            for value in node.inputs:
                if value.id in folded:
                    boundary.add(value.id)

        rw = GraphRewriter(graph)
        rw.copy_inputs()
        for node in graph.nodes:
            attrs = folded.get(node.output.id)
            if attrs is None:
                rw.emit(node)
                continue
            if node.output.id not in boundary:
                continue
            value = rw.new.add_node("constant", [], attrs)
            if value.type != node.output.type:
                raise AssertionError(
                    f"常量折叠 {node.op.name} 后类型由 {node.output.type} 变成 {value.type}"
                )
            rw.map[node.output.id] = value
        return rw.finish()

    def _foldable(self, node: Node) -> bool:
        if node.op.nullary or node.op.portability is not Portability.PORTABLE:
            return False
        if not node.op.pure:
            # 带效果的算子（如 RNG）结果确定也不能固化，见模块说明第 2 条。
            return False
        type_ = node.output.type
        if not type_.is_static:
            return False
        return type_.numel() <= self.max_elements

    def count(self, graph: Graph) -> int:
        """诊断用：该图可被折叠的节点数（不含 constant 自身）。"""
        known = set()
        folded = 0
        for node in graph.nodes:
            if node.op.name == "constant":
                if node.output.type.is_static:
                    known.add(node.output.id)
                continue
            if not self._foldable(node):
                continue
            if all(v.id in known for v in node.inputs):
                known.add(node.output.id)
                folded += 1
        return folded
