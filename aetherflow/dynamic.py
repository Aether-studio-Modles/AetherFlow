"""动态执行：图按**当前输入形状**构建，形状见过就复用。

为什么单独一层而不改静态路径：5.25 已列过静态图的三个不可替代之处——构图期
就能预检后端能力、反向图可序列化独立分发、CUDA Graph 捕获要求路径固定。真正
的 define-by-run（每次运算即时执行、张量带值）会把这三条一起放弃，所以这里
不替换它们，只补静态图缺的那一类需求：**形状随数据变化**。

用法是"分步执行"，而不是把控制流塞进图里：

    runner = DynamicRunner(build)          # build(shapes) -> Graph
    h = runner.run({"x": batch_a})["y"]    # 第一次：按 batch_a 的形状构建
    if h.mean() > 0:                       # Python 里读回真实值做分支
        h = runner.run({"x": batch_b})["y"]

**``build`` 只能依赖形状，不能读数据。** 缓存键就是输入形状（与 dtype）；
如果 build 里按数组内容分支，形状相同而结果不同时就会命中错误缓存。数据相关
的分支必须发生在两次 ``run`` 之间——这也正是"define-by-run 的执行手感"真正
需要的东西，而不是在构图函数里埋隐藏状态。

**它不是完整的 define-by-run。** 一次 ``run`` 内部仍是静态图；没有逐算子
即时求值，也没有"中途读中间值再决定下一个算子"。这些留给后续，别在这里
假装已经做了。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from .backends.base import Backend
from .backends.reference_backend import ReferenceBackend
from .ir.graph import Graph

__all__ = ["DynamicRunner"]

ShapeKey = Tuple[Tuple[str, Tuple[int, ...], str], ...]


class DynamicRunner:
    """按输入形状缓存已构建的图，每次 ``run`` 只换数据。

    ``capacity`` 是缓存的不同形状数；超了按最久未用淘汰。设成 1 就是"每次都
    重建"，设大一点适合变长 batch 反复出现（如长度分桶后的几档）。
    """

    def __init__(
        self,
        build: Callable[[Mapping[str, Tuple[int, ...]]], Graph],
        *,
        backend: Optional[Backend] = None,
        capacity: int = 8,
    ) -> None:
        if not callable(build):
            raise TypeError(f"build 必须可调用，得到 {type(build).__name__}")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError(f"capacity 必须是正整数，得到 {capacity!r}")
        self.build = build
        self.backend = backend if backend is not None else ReferenceBackend()
        self.capacity = capacity
        self.builds = 0
        self._cache: "OrderedDict[ShapeKey, Graph]" = OrderedDict()

    @property
    def shapes(self) -> Tuple[ShapeKey, ...]:
        """缓存里的形状键，最近用过的在后面（调试用）。"""
        return tuple(self._cache)

    def run(self, inputs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """按输入形状取（或构建）一张图并执行，返回主机侧输出。"""
        if not isinstance(inputs, Mapping) or not inputs:
            raise ValueError("inputs 必须是非空映射")
        key = _shape_key(inputs)
        graph = self._cache.get(key)
        if graph is None:
            graph = self.build({name: shape for name, shape, _ in key})
            if not isinstance(graph, Graph):
                raise TypeError(
                    f"build 必须返回 Graph，得到 {type(graph).__name__}"
                )
            self.builds += 1
            self._remember(key, graph)
        else:
            self._cache.move_to_end(key)
        return self.backend.execute(graph, inputs)

    def _remember(self, key: ShapeKey, graph: Graph) -> None:
        self._cache[key] = graph
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)


def _shape_key(inputs: Mapping[str, np.ndarray]) -> ShapeKey:
    """形状 + dtype 的规范化键；顺序与输入顺序无关。"""
    entries = []
    for name in sorted(inputs):
        value = inputs[name]
        try:
            shape = tuple(int(d) for d in value.shape)
        except AttributeError:
            raise TypeError(
                f"输入 {name!r} 必须是数组（需要 .shape），得到 {type(value).__name__}"
            ) from None
        dtype = str(getattr(value, "dtype", "unknown"))
        entries.append((name, shape, dtype))
    return tuple(entries)
