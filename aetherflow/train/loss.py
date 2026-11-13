"""损失函数：全部在**图内**构图，因此梯度由 ``backward`` 自动给出。

为什么不写成 numpy 侧的函数：损失一旦在图外算，反传就只能靠手写或另一套
AD，那等于把"这套 IR 能训模型"这件事外包出去。这里的每个损失都是普通
PORTABLE 节点的组合，任何能跑前向的后端都能跑它的反向图。

一个必须说清的限制：``mean`` 需要**元素个数静态已知**。归约到标量要求
把形状折成常量，而符号维的乘积在构图期不可知。训练时 batch 通常是静态的；
若需要符号 batch，请先用静态形状构图，或自行改为 sum（等价于按 batch
大小缩放学习率）。
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..ir.graph import Graph, Value

__all__ = ["mean", "mse", "softmax_cross_entropy", "l2_penalty"]


def _const(g: Graph, value: float, dtype) -> Value:
    return g.constant(float(value), dtype=dtype, shape=())


def _sum_all(g: Graph, value: Value) -> Value:
    """把所有维依次归约成标量，不需要知道具体尺寸。

    逐轴归约而不是 reshape 成一维：reshape 需要元素总数，符号维下拿不到；
    而"秩减去一"永远是可用的信息。
    """
    while value.type.rank > 0:
        value = g.reduce_sum(value, axis=value.type.rank - 1, keepdims=False)
    return value


def mean(g: Graph, value: Value, count: Optional[int] = None) -> Value:
    """全元素平均。``count`` 缺省时用静态形状推断。"""
    if count is None:
        if not value.type.is_static:
            raise ValueError(
                f"mean 需要静态形状来计算元素个数，得到 {value.type}；"
                "请给静态 batch 维度，或显式传入 count"
            )
        count = value.type.numel()
    count = int(count)
    if count <= 0:
        raise ValueError(f"mean 的元素个数必须为正，得到 {count}")
    total = _sum_all(g, value)
    return g.mul(total, _const(g, 1.0 / count, value.type.dtype))


def mse(g: Graph, pred: Value, target: Value) -> Value:
    """均方误差：(1/N) * sum((pred - target)^2)。"""
    diff = g.sub(pred, target)
    return mean(g, g.mul(diff, diff))


def softmax_cross_entropy(g: Graph, logits: Value, labels: Value) -> Value:
    """多分类交叉熵，``labels`` 为 one-hot（与 logits 同形状同 dtype）。

    **先减最大值得 log-softmax，再取 log**：``log_softmax(z) = z - max(z)
    - log(sum(exp(z - max(z))))``。整条路径里只有非正数进 ``exp``，概率再小
    也不会先下溢成 0 再去取 ``log``。

    早先的写法是 ``softmax`` 之后再 ``log(probs + eps)``。那不只是数值技巧：
    它把损失改成了 ``-log(p + eps)``，对 one-hot 目标的梯度是
    ``p_k / (p_k + eps) * (p_j - y_j)``。当模型非常确信错误答案（``p_k`` 远
    小于 ``eps``）时，这个系数趋近于零——最需要纠正的样本反而几乎没有梯度，
    而损失也被钉死在 ``-log(eps)`` 附近。f16 下 ``1e-12`` 直接下溢成 0，
    ``log(0)`` 与反向除零还会给出 inf / NaN。
    """
    if labels.type.shape != logits.type.shape:
        raise ValueError(
            f"cross_entropy 的 labels 形状必须与 logits 一致，得到 {labels.type.shape} 与 {logits.type.shape}"
        )
    shift = g.reduce_max(logits, axis=-1, keepdims=True)
    shifted = g.sub(logits, shift)
    log_z = g.log(g.reduce_sum(g.exp(shifted), axis=-1, keepdims=True))
    logp = g.sub(shifted, log_z)
    per_row = g.reduce_sum(g.mul(logp, labels), axis=-1, keepdims=False)
    nll = g.mul(per_row, _const(g, -1.0, logits.type.dtype))
    return mean(g, nll)


def l2_penalty(g: Graph, params: Sequence[Value], weight: float = 1.0) -> Value:
    """L2 正则项：(weight / 2) * sum_i ||p_i||^2。

    返回的是一个**标量 Value**，调用方用 ``add`` 把它并进总损失；
    图里的常量因此是静态的，梯度会额外给每个参数一份 ``weight * p``。
    """
    params = list(params)
    if not params:
        raise ValueError("l2_penalty 至少需要一个参数")
    dtype = params[0].type.dtype
    total: Optional[Value] = None
    for p in params:
        if p.type.dtype != dtype:
            raise TypeError(
                f"l2_penalty 要求所有参数 dtype 一致，得到 {dtype.value} 与 {p.type.dtype.value}"
            )
        sq = _sum_all(g, g.mul(p, p))
        total = sq if total is None else g.add(total, sq)
    assert total is not None
    return g.mul(total, _const(g, 0.5 * float(weight), dtype))
