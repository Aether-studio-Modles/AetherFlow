"""训练态 BatchNorm 的 running stats 维护。

IR 的 ``batch_norm`` 是训练态语义：用**当前 batch** 的均值/方差归一化，
图里没有"running stats 更新"这一步——那属于训练策略，不是张量语义
（见 ``ir/ops.py`` 的算子说明与 DESIGN 5.24）。于是"训练输出与 torch 对齐"
并不自动意味着"训练后 eval 状态也对齐"：eval 用的是整段训练累积下来的
running mean/var，没人维护它，导出图拿去推理就会拿初始值（0 / 1）算。

这里补的是那一步，且不要求模型改构图：

1. 扫描图里的 ``batch_norm`` 节点，记下通道轴、通道数与每个通道的样本数；
2. 以这些节点的**依赖闭包**重建一张统计子图，输出各自的 ``batch_norm_stats``
   （值为 ``(2, C)`` 的 mean/var）；
3. 每步前向之后按 ``running = (1 - momentum) * running + momentum * batch``
   更新。方差用无偏估计（``n / (n - 1)``），与 torch 的 ``BatchNorm`` 一致；
   ``n == 1`` 时退回有偏值，避免除零。

统计子图只复制到 BN 输入为止的节点，不重跑整张网络；它是一次额外的
``Backend.execute``，因此只在显式开启 ``Trainer(bn_momentum=...)`` 时构建。
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Tuple

import numpy as np

from ..ir.graph import Graph, Value

__all__ = ["BNSite", "bn_sites", "build_stats_graph", "RunningStats"]


class BNSite(NamedTuple):
    """图里一个训练态 BN 的位置。"""

    name: str
    """``batch_norm`` 节点的 SSA 名（``node.output.name``），作为状态键。"""

    axis: int
    """通道轴（已按输入秩规范化）。"""

    channels: int
    """通道数，即 running stats 的长度。"""

    count: int
    """每个通道的元素个数（无偏方差修正用）。"""

    x_id: int
    """BN 输入的 Value id，统计子图按它找对应的复制值。"""

    output: str
    """统计子图里该站点 ``batch_norm_stats`` 的输出名。"""


def bn_sites(graph: Graph) -> List[BNSite]:
    """找出图里全部训练态 BN 站点；没有时返回空列表。

    形状必须静态：无偏修正要用到"每个通道多少个样本"，符号维算不出来。
    训练图本来就要求静态形状（见 ``Trainer`` 的构造期检查），这里只是把
    错误提前到 BN 站点这一层，并指出是哪个节点。
    """
    sites: List[BNSite] = []
    for node in graph.nodes:
        if node.op.name != "batch_norm":
            continue
        x = node.inputs[0]
        axis = int(node.attrs.get("axis", 1)) % x.type.rank
        shape = x.type.shape
        channels = shape[axis]
        if not isinstance(channels, int):
            raise ValueError(
                f"BN 节点 {node.output.name} 的通道维含符号 {channels!r}，"
                "running stats 的形状无法静态确定"
            )
        count = 1
        for i, dim in enumerate(shape):
            if i == axis:
                continue
            if not isinstance(dim, int):
                raise ValueError(
                    f"BN 节点 {node.output.name} 的第 {i} 维含符号 {dim!r}，"
                    "无偏方差修正需要静态样本数"
                )
            count *= dim
        sites.append(
            BNSite(
                name=node.output.name,
                axis=axis,
                channels=int(channels),
                count=int(count),
                x_id=int(x.id),
                output=f"bn_stats/{node.output.name}",
            )
        )
    return sites


def build_stats_graph(graph: Graph, sites: List[BNSite]) -> Graph:
    """按 BN 站点的依赖闭包重建一张只算统计量的子图。

    复制节点时保留原 SSA 名，常量节点按原 attrs 重建（含 ``data``）。
    图输入全量复制：调用方手里本来就有完整的 feed，少复制几个并不能省下
    一次上传，反而让"用哪个输入"多一条要维护的规则。
    """
    if not sites:
        raise ValueError("图里没有训练态 BN 节点，不需要统计子图")
    g = Graph(f"{graph.name}_bn_stats")
    env: Dict[int, Value] = {v.id: g.input(v.name, v.type) for v in graph.inputs}

    # 依赖闭包：从每个 BN 节点反向 BFS。``graph.nodes`` 是拓扑序，复制时
    # 按它遍历即可保证输入先于消费者。
    needed = set()
    stack = [node for node in graph.nodes if node.op.name == "batch_norm"]
    while stack:
        node = stack.pop()
        if node.output.id in needed:
            continue
        needed.add(int(node.output.id))
        for value in node.inputs:
            if value.producer is not None:
                stack.append(value.producer)

    for node in graph.nodes:
        if node.output.id not in needed:
            continue
        args = [env[int(value.id)] for value in node.inputs]
        env[int(node.output.id)] = g.add_node(
            node.op.name, args, dict(node.attrs), name=node.output.name
        )

    for site in sites:
        stats = g.add_node(
            "batch_norm_stats",
            [env[site.x_id]],
            {"axis": site.axis},
            name=f"bn_stats__{site.name}",
        )
        g.output(stats, site.output)
    return g


class RunningStats:
    """按站点维护的 running mean / var。

    初值与 torch 一致（mean=0、var=1）。更新是 EMA，方差用无偏估计：
    归一化用的是有偏方差（``np.var``），但累积到 running_var 的应是
    "对总体方差的无偏估计"，否则 eval 的缩放系数会系统性偏小。
    """

    def __init__(self, sites: List[BNSite]) -> None:
        self._sites = {site.name: site for site in sites}
        self._mean: Dict[str, np.ndarray] = {
            site.name: np.zeros(site.channels, dtype=np.float32) for site in sites
        }
        self._var: Dict[str, np.ndarray] = {
            site.name: np.ones(site.channels, dtype=np.float32) for site in sites
        }
        self.updates = 0

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(self._sites)

    def update(self, name: str, stats: np.ndarray, momentum: float) -> None:
        """用一个 batch 的 ``(2, C)`` 统计量推进 EMA。"""
        if name not in self._sites:
            raise KeyError(f"没有 BN 站点 {name!r}")
        site = self._sites[name]
        batch = np.asarray(stats, dtype=np.float32)
        if batch.shape != (2, site.channels):
            raise ValueError(
                f"站点 {name} 的统计量形状应为 {(2, site.channels)}，得到 {batch.shape}"
            )
        mean = batch[0]
        var = batch[1]
        if site.count > 1:
            var = var * (site.count / (site.count - 1.0))
        self._mean[name][...] = (1.0 - momentum) * self._mean[name] + momentum * mean
        self._var[name][...] = (1.0 - momentum) * self._var[name] + momentum * var
        self.updates += 1

    def get(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """某站点的 ``(mean, var)`` 副本。"""
        if name not in self._sites:
            raise KeyError(f"没有 BN 站点 {name!r}")
        return self._mean[name].copy(), self._var[name].copy()

    def arrays(self) -> Dict[str, np.ndarray]:
        """全部站点的 ``(mean, var)`` 副本，键是站点名。"""
        return {name: self.get(name) for name in self._sites}

    def state_dict(self) -> Dict[str, np.ndarray]:
        """存档用的扁平字典（``bn/mean/<站点>``、``bn/var/<站点>``）。"""
        state: Dict[str, np.ndarray] = {}
        for name in self._sites:
            state[f"bn/mean/{name}"] = self._mean[name].copy()
            state[f"bn/var/{name}"] = self._var[name].copy()
        return state

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        """从存档恢复；键集合与形状必须完全一致。"""
        want = {f"bn/mean/{name}" for name in self._sites} | {
            f"bn/var/{name}" for name in self._sites
        }
        if set(state) != want:
            raise KeyError(
                "存档里的 BN running stats 与当前图不一致：缺少 "
                f"{sorted(want - set(state))}，多出 {sorted(set(state) - want)}"
            )
        for name in self._sites:
            for kind, target in (("mean", self._mean), ("var", self._var)):
                arr = np.asarray(state[f"bn/{kind}/{name}"], dtype=np.float32)
                if arr.shape != target[name].shape:
                    raise ValueError(
                        f"站点 {name} 的 {kind} 形状 {arr.shape} 与当前 "
                        f"{target[name].shape} 不一致"
                    )
                target[name][...] = arr

    def template(self) -> Dict[str, np.ndarray]:
        """``load_checkpoint`` 的附加数组模板：形状对、值不重要。"""
        return {
            key: np.zeros_like(arr) for key, arr in self.state_dict().items()
        }
