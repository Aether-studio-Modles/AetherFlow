"""自动混合精度：把"哪一层留 f32"从调用方手里接过来。

5.28 的 f16 是**整图 f16**：要么全 f16，要么全 f32。想让某些层留 f32，
得由调用方自己在图里插 ``cast``（DESIGN 5.33 记的就是这条限制）。真实训练
要的是混合精度——矩阵乘与卷积吃 f16 的吞吐，归约、归一化、softmax 留 f32，
其余逐元素算子跟着输入走。

这个 pass 把那句话写成两条规则：

- ``LOW_PRECISION``（低精度收益）：工作精度改成目标 dtype；
- ``FORCE_F32``（数值敏感）：工作精度拉回 f32；
- 其余：按**最宽输入**走——有 f32 输入就 f32，全是 f16 才 f16。

第三条不是偷懒，正是残差连接 ``x + matmul(x, w)`` 要的行为：matmul 在 f16
里算，加法回到 f32。少了它，f16 的 matmul 结果会直接撞上一个 f32 的 x，
图就不再满足"同一节点输入 dtype 一致"。

**只降 f32，不碰 f64 与整数。** f64 是调用方明确要求的高精度，整数
（argmax 的下标、dropout 掩码）根本不该参与浮点提升。

**``dtype`` 可以是 ``f16`` 或 ``bf16``。** 两者的规则完全相同，差别只在
数值：bf16 的指数范围与 f32 一样（不需要梯度缩放），但尾数只有 7 位，
同样的图在 bf16 下的相对误差会比 f16 大一个量级左右。参考执行器会把每个
bf16 类型的值量化到 bf16 精度（``runtime/bf16.py``），所以这里的"降到
bf16"是数值承诺，不是换个标签。

**图输出一律恢复原图的 dtype。** 打开 autocast 不改变对外接口：调用方交进
交出的仍是同一份 f32 张量，只有图内部的中间值变窄。这也是它幂等的原因——
对已经转换过的图再跑一次，所有输入都已在对的位置，一个 ``cast`` 都不会新增。

``low_precision_inputs`` 把**指定的图输入**也降到目标 dtype，用于"参数以低
精度驻留"（``Trainer(master_weights=True)``）：参数缓冲减半，前向 / 反向读
权重时带宽也减半。注意这**不是**"减少 cast 的开关"——被低精度算子直接消费的
参数（matmul / conv 的权重）省掉一次 down-cast，只被 f32-only 算子（归一化、
归约）消费的参数反而在入口多一次 up-cast，净变化取决于模型结构。
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

from ..ir.graph import Graph, Value
from ..ir.types import DType
from .base import Pass

__all__ = ["Autocast", "LOW_PRECISION_OPS", "FORCE_F32_OPS"]

# 已知能吃低精度收益的算子：这两个族都走 tensor core 的张量核路径。
# 反向算子不列：反向图是从已转换的前向图展开的，梯度天然跟着 f16 走，
# 最后由 cast 节点把参数梯度带回 f32。
LOW_PRECISION_OPS: Tuple[str, ...] = ("matmul", "matmul_batched", "conv2d")

# 数值敏感：累加项多、动态范围大，或者输出直接进损失。
FORCE_F32_OPS: Tuple[str, ...] = (
    "softmax",
    "layernorm",
    "rmsnorm",
    "batch_norm",
    "batch_norm_stats",
    "reduce_sum",
    "reduce_mean",
    "reduce_max",
)

# 浮点宽窄次序：值越大越宽。这是一个**实用性**排序，不是 IEEE 位宽表：
# bf16 的指数范围与 f32 相同（不会像 f16 那样溢出到 inf），但尾数只有 7 位，
# 所以计算精度仍然低于 f32；把它排在 f16 与 f32 之间，既能让 "f32 + bf16"
# 提升到 f32，也不会让 bf16 与 f16 混用时把范围更宽的 bf16 降回 f16。
# f64 不参与降精度，只作为"最宽输入"参与提升。
# 键用 IR 里的短名（``DType.value``），与 ``cast`` 的 ``dtype`` 属性一致。
_FLOAT_ORDER = {"f16": 0, "bf16": 1, "f32": 2, "f64": 3}


class Autocast(Pass):
    """按白名单把 f32 计算降到 ``dtype``，边界自动插 ``cast``。

    ``low_precision`` / ``force_f32`` 的 ``None`` 表示用默认集；显式传空集
    就是"一个都不降 / 一个都不强制"——``force_f32=()`` 是"整图都不留 f32"
    的唯一写法，若用 ``or`` 回落默认集，这个意思表达不出来。
    """

    name = "autocast"

    def __init__(
        self,
        dtype: str = "f16",
        low_precision: Optional[Iterable[str]] = None,
        force_f32: Optional[Iterable[str]] = None,
        low_precision_inputs: Optional[Iterable[str]] = None,
    ) -> None:
        if dtype not in _FLOAT_ORDER:
            raise ValueError(
                f"autocast 的目标 dtype 必须是浮点类型，得到 {dtype!r}；"
                f"可选 {sorted(_FLOAT_ORDER)}"
            )
        if dtype == "f64":
            raise ValueError("autocast 的目标 dtype 不能是 f64（它是最宽的浮点类型）")
        self.dtype = dtype
        self.low_precision = frozenset(
            LOW_PRECISION_OPS if low_precision is None else low_precision
        )
        self.force_f32 = frozenset(
            FORCE_F32_OPS if force_f32 is None else force_f32
        )
        self.low_precision_inputs = frozenset(low_precision_inputs or ())

    def run(self, graph: Graph) -> Graph:
        unknown = self.low_precision_inputs - {v.name for v in graph.inputs}
        if unknown:
            raise KeyError(
                f"low_precision_inputs 里有不是图输入的名字: {sorted(unknown)}；"
                "拼错的名字会让参数降精度静默失效"
            )
        out = Graph(graph.name)
        mapping = {}
        for value in graph.inputs:
            if (
                value.name in self.low_precision_inputs
                and value.type.dtype.value == "f32"
            ):
                # 只降 f32：f64 是调用方明确要求的高精度，与其余规则一致。
                mapping[value.id] = out.input(
                    value.name, value.type.with_dtype(DType(self.dtype))
                )
            else:
                mapping[value.id] = out.input(value.name, value.type)

        for node in graph.nodes:
            args = [mapping[a.id] for a in node.inputs]
            work = self._work_dtype(node.op.name, [a.type.dtype.value for a in args])
            aligned = [self._align(out, a, work) for a in args]
            mapping[node.output.id] = out.add_node(node.op.name, aligned, node.attrs)

        for value, name in zip(graph.outputs, graph.output_names):
            nv = mapping[value.id]
            if nv.type.dtype != value.type.dtype:
                nv = out.add_node("cast", [nv], {"dtype": value.type.dtype.value})
            # 未命名输出的 ``output_names`` 会退化成原值名（``v<id>``），不能拿它
            # 跟重写后的新值名比：插了 cast 之后编号会错位，旧名会与自动名撞车。
            # 用原值的 SSA 名判断"有没有显式别名"，未命名就继续不命名。
            out.output(nv, None if name == value.name else name)
        return out

    # -- 规则 ---------------------------------------------------------

    def _work_dtype(self, op_name: str, dtypes) -> Optional[str]:
        """本节点的计算精度；``None`` 表示"原样搬运，不做对齐"。

        ``cast`` / ``constant`` 一律 ``None``：前者若参与对齐会把自己套进新一层
        cast，后者没有输入也无需对齐。
        """
        if op_name in ("cast", "constant"):
            return None
        floats = [d for d in dtypes if d in _FLOAT_ORDER]
        if not floats:
            return None  # 全整数：不参与浮点提升
        widest = max(floats, key=_FLOAT_ORDER.__getitem__)

        if op_name in self.force_f32:
            # f64 已经比 f32 宽，不下压；f16 / bf16 都要提到 f32——数值敏感
            # 算子的工作精度不能因为目标 dtype 换了就跟着变窄。
            return "f32" if _FLOAT_ORDER[widest] < _FLOAT_ORDER["f32"] else widest
        if op_name in self.low_precision:
            # 只降 f32；f64 不动
            return self.dtype if widest == "f32" else widest
        return widest

    @staticmethod
    def _align(graph: Graph, value: Value, work: Optional[str]) -> Value:
        # 整数参数（embedding 的 ids、gather 的下标、dropout 掩码）不参与
        # 浮点对齐：``work`` 是"这个节点的浮点工作精度"，不是"所有参数的
        # dtype"。把它们 cast 成 f32 会让 embedding 直接构图失败。
        if work is None or value.type.dtype.value not in _FLOAT_ORDER:
            return value
        if value.type.dtype.value == work:
            return value
        return graph.add_node("cast", [value], {"dtype": work})

    def __repr__(self) -> str:
        return f"<Autocast({self.dtype})>"
