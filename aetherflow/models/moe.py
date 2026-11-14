"""MoE（混合专家）构件。

这里的重点是"用已有 IR 把路由闭环表达完整"，而不是追求稀疏算力：

- **路由**：``softmax(x @ w_router)`` 得到每个 token 对每个专家的概率；
- **选择**：``topk_mask`` 标出前 k 个专家，概率乘掩码后按行重新归一化，
  得到门控权重；
- **计算与组合**：每个专家都是一次两层 MLP，所有 token 都流经所有专家，
  再按门控权重加权求和。这与"把 token 按专家分组再算"在数值上等价，
  区别只在算力：静态图里 token 的分组大小随数据变化，要把它们真正
  打包成变长批次需要动态形状与 gather/scatter，超出当前 IR 的表达范围。
  因此这里是**稠密参考语义**，作为稀疏 dispatch 的正确性基准；
- **负载均衡**：``aux = E * Σ_e f_e * P_e``，其中 ``f_e`` 是被分到专家 e 的
  token 比例（top-k 时按 k 折算）、``P_e`` 是路由概率的批均值。它只依赖
  概率与掩码，可微且与专家输出无关；
- **专家并行**：显式拒绝。跨设备 all-to-all 的 dispatch/combine 需要设备
  集合通信，本机不具备，见 ``require_expert_parallel``。

参数命名与其它构件一致：权重布局是 ``(in, out)``，输入表同时含数据输入
``x`` 与全部参数，训练时把 ``x`` 传给 ``ParameterStore.from_graph(exclude=...)``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

from ..ir.graph import Graph, Value

__all__ = [
    "MoESpec",
    "EXPERT_PARALLEL_SUPPORTED",
    "build_moe",
    "require_expert_parallel",
]

_ACTIVATIONS = ("relu", "silu", "gelu", "tanh")

#: 专家并行是否可用。当前只有单设备稠密参考语义，跨设备 all-to-all 未实现。
EXPERT_PARALLEL_SUPPORTED = False


def require_expert_parallel(devices: int = 1) -> None:
    """专家并行的显式拒绝面。

    不做"设备数大于 1 时偷偷退回单设备"的降级：专家并行的语义是把每个
    token 只发给被选中的专家所在的设备，降级到单设备会改变通信拓扑与
    显存分布，调用方必须显式知道这件事。
    """
    if isinstance(devices, bool) or not isinstance(devices, int):
        raise TypeError(f"专家并行的设备数必须是整数，得到 {devices!r}")
    if devices < 1:
        raise ValueError(f"专家并行的设备数必须为正，得到 {devices}")
    if devices > 1:
        raise NotImplementedError(
            f"专家并行需要 {devices} 张设备与跨设备 all-to-all 的 token "
            "dispatch/combine，当前实现只有单设备稠密参考语义；"
            "请使用 devices=1，或改用 DataParallelTrainer 的设备数据并行"
        )


@dataclass(frozen=True)
class MoESpec:
    """``d_model`` 是 token 宽度、``d_ff`` 是专家隐层宽度。"""

    d_model: int = 8
    d_ff: int = 16
    experts: int = 4
    top_k: int = 2
    batch: int = 4
    activation: str = "gelu"
    load_balance_weight: float = 0.01
    expert_parallel: int = 1

    def __post_init__(self) -> None:
        for name in ("d_model", "d_ff", "experts", "top_k", "batch"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"MoESpec.{name} 必须是整数，得到 {value!r}")
            if value <= 0:
                raise ValueError(f"MoESpec.{name} 必须为正，得到 {value}")
        if self.top_k > self.experts:
            raise ValueError(
                f"top_k={self.top_k} 不能超过专家数 {self.experts}；"
                "每个 token 最多只能选中一个专家一次"
            )
        if self.activation not in _ACTIVATIONS:
            raise ValueError(
                f"未知专家激活 {self.activation!r}；可用: {', '.join(_ACTIVATIONS)}"
            )
        weight = self.load_balance_weight
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError(f"load_balance_weight 必须是数值，得到 {weight!r}")
        if not math.isfinite(float(weight)) or float(weight) < 0.0:
            raise ValueError(f"load_balance_weight 必须是非负有限数，得到 {weight}")
        require_expert_parallel(self.expert_parallel)

    @property
    def param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        shapes: Dict[str, Tuple[int, ...]] = {
            "w_router": (self.d_model, self.experts),
        }
        for e in range(self.experts):
            shapes[f"w1_{e}"] = (self.d_model, self.d_ff)
            shapes[f"b1_{e}"] = (self.d_ff,)
            shapes[f"w2_{e}"] = (self.d_ff, self.d_model)
            shapes[f"b2_{e}"] = (self.d_model,)
        return shapes


def build_moe(
    g: Graph, spec: MoESpec | None = None
) -> Tuple[Dict[str, Value], Value, Value]:
    """构建 MoE，返回 ``(输入表, 输出 y, 负载均衡损失 aux)``。

    ``y`` 的形状是 ``(batch, d_model)``；``aux`` 是标量（``f32[()]``），
    已经乘上 ``load_balance_weight``，训练时加到任务损失上即可。
    """
    from ..ir.types import f32

    spec = spec or MoESpec()
    inputs: Dict[str, Value] = {
        "x": g.input("x", f32[spec.batch, spec.d_model])
    }
    for name, shape in spec.param_shapes.items():
        inputs[name] = g.input(name, f32[shape])

    logits = g.matmul(inputs["x"], inputs["w_router"])
    probs = g.add_node("softmax", [logits], {"axis": -1})

    mask = g.add_node("topk_mask", [probs], {"k": spec.top_k, "axis": -1})
    selected = g.mul(probs, mask)
    # top-k 的概率和小于 1，按被选中的概率重新归一化，门控权重才是"条件概率"
    gates = g.div(selected, g.reduce_sum(selected, axis=-1, keepdims=True))

    y: Value | None = None
    for e in range(spec.experts):
        hidden = g.add(g.matmul(inputs["x"], inputs[f"w1_{e}"]), inputs[f"b1_{e}"])
        hidden = g.add_node(spec.activation, [hidden])
        out = g.add(g.matmul(hidden, inputs[f"w2_{e}"]), inputs[f"b2_{e}"])
        # 只取第 e 列门控，形状 (batch, 1)：与专家输出按行广播相乘
        gate_e = g.add_node("slice", [gates], {"dim": 1, "start": e, "stop": e + 1})
        contrib = g.mul(out, gate_e)
        y = contrib if y is None else g.add(y, contrib)
    assert y is not None

    # 负载均衡：f_e 是被选中比例（top-k 下除以 k，使 Σ_e f_e = 1），
    # P_e 是路由概率的批均值。两项都只依赖概率/掩码，梯度不会穿过专家输出。
    p_mean = g.reduce_mean(probs, axis=0)
    f_mean = g.mul(
        g.reduce_mean(mask, axis=0), g.constant(1.0 / spec.top_k, f32)
    )
    aux = g.mul(
        g.reduce_sum(g.mul(p_mean, f_mean), axis=0, keepdims=False),
        g.constant(float(spec.experts) * float(spec.load_balance_weight), f32),
    )
    return inputs, y, aux
