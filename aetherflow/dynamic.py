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

**trace-once：声明动态轴，一次构图、多种形状重放。** 默认模式按完整形状区分
缓存，batch 从 2 变到 3 就要重建一张图。把变化的轴声明出来，构图只做一次：

    runner = DynamicRunner(build, dynamic_axes={"x": {0: "B"}})
    runner.run({"x": np.ones((2, 4), np.float32)})   # 用 f32[B, 4] 构图
    runner.run({"x": np.ones((7, 4), np.float32)})   # 复用同一张图，绑定 B=7

``build`` 收到的是**形状模板**：被声明的轴换成符号名，其余轴仍是具体整数。
符号由执行器在每次运行时按输入数组的实际形状绑定，因此一条图能跑任意 batch。
同一符号出现在多个输入上就表示"这两维必须相等"——执行器会对不一致的绑定报
冲突，而不是悄悄挑一个。

这是"一次 trace、多次重放"，不是"控制流进图"：一次 ``run`` 内部仍是静态图。
构图函数必须真正使用传进来的模板（声明为符号的轴不能写成整数），否则第一次
运行能过、换形状就撞静态维不匹配——这种情况在构图后立刻显式报错，不留到下一
次运行。

**它不是完整的 define-by-run。** 一次 ``run`` 内部仍是静态图；没有逐算子
即时求值，也没有"中途读中间值再决定下一个算子"。这些留给后续，别在这里
假装已经做了。
"""

from __future__ import annotations

from collections import OrderedDict
from types import MappingProxyType
from typing import Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from .backends.base import Backend
from .backends.reference_backend import ReferenceBackend
from .ir.graph import Graph
from .ir.types import DType, Dim, TensorType

__all__ = ["DynamicRunner"]

ShapeKey = Tuple[Tuple[str, Tuple[Dim, ...], str], ...]
DynamicAxes = Mapping[str, Mapping[int, str]]


class DynamicRunner:
    """按输入形状缓存已构建的图，每次 ``run`` 只换数据。

    ``capacity`` 是缓存的不同形状数；超了按最久未用淘汰。设成 1 就是"每次都
    重建"，设大一点适合变长 batch 反复出现（如长度分桶后的几档）。

    ``dynamic_axes`` 给出「输入名 → {轴: 符号名}」的声明。声明过的轴不进缓存
    键：形状只在这几维上不同的两次运行共用同一张图，由执行器把符号绑到实际
    数值。未声明的轴（以及 dtype）变化仍会触发重建。
    """

    def __init__(
        self,
        build: Callable[[Mapping[str, Tuple[Dim, ...]]], Graph],
        *,
        dynamic_axes: Optional[DynamicAxes] = None,
        backend: Optional[Backend] = None,
        capacity: int = 8,
    ) -> None:
        if not callable(build):
            raise TypeError(f"build 必须可调用，得到 {type(build).__name__}")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError(f"capacity 必须是正整数，得到 {capacity!r}")
        self.build = build
        self.dynamic_axes = _check_dynamic_axes(dynamic_axes)
        self.backend = backend if backend is not None else ReferenceBackend()
        self.capacity = capacity
        self.builds = 0
        self._cache: "OrderedDict[ShapeKey, Graph]" = OrderedDict()

    @property
    def shapes(self) -> Tuple[ShapeKey, ...]:
        """缓存里的形状键，最近用过的在后面（调试用）。

        trace-once 模式下键里的动态轴是符号名而不是数字——这正是"同一张图
        覆盖了哪些形状"的答案。
        """
        return tuple(self._cache)

    def run(self, inputs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """按输入形状取（或构建）一张图并执行，返回主机侧输出。"""
        if not isinstance(inputs, Mapping) or not inputs:
            raise ValueError("inputs 必须是非空映射")
        key, declared = _shape_key(inputs, self.dynamic_axes)
        graph = self._cache.get(key)
        if graph is None:
            graph = self.build({name: shape for name, shape, _ in key})
            if not isinstance(graph, Graph):
                raise TypeError(
                    f"build 必须返回 Graph，得到 {type(graph).__name__}"
                )
            _require_traced_inputs(graph, declared)
            self.builds += 1
            self._remember(key, graph)
        else:
            self._cache.move_to_end(key)
        return self.backend.execute(graph, inputs)

    def _remember(self, key: ShapeKey, graph: Graph) -> None:
        self._cache[key] = graph
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)


def _check_dynamic_axes(dynamic_axes: Optional[DynamicAxes]) -> DynamicAxes:
    """校验并冻结动态轴声明；``None`` 表示关闭 trace-once。"""
    if dynamic_axes is None:
        return MappingProxyType({})
    if not isinstance(dynamic_axes, Mapping):
        raise TypeError(
            f"dynamic_axes 必须是映射，得到 {type(dynamic_axes).__name__}"
        )
    frozen = {}
    for name, axes in dynamic_axes.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"dynamic_axes 的输入名必须是非空字符串，得到 {name!r}")
        if not isinstance(axes, Mapping) or not axes:
            raise ValueError(
                f"输入 %{name} 的动态轴声明必须是非空映射「轴: 符号名」，"
                f"得到 {axes!r}"
            )
        declared = {}
        for axis, symbol in axes.items():
            if isinstance(axis, bool) or not isinstance(axis, int):
                raise ValueError(
                    f"输入 %{name} 的动态轴必须是整数，得到 {axis!r}"
                )
            if not isinstance(symbol, str):
                raise ValueError(
                    f"输入 %{name} 的动态轴 {axis} 的符号名必须是字符串，"
                    f"得到 {symbol!r}"
                )
            # 复用 TensorType 的维校验：合法标识符、且不能与 dtype 名撞车。
            try:
                TensorType(DType.F32, (symbol,))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"输入 %{name} 的动态轴 {axis} 的符号名非法: {exc}"
                ) from None
            declared[axis] = symbol
        frozen[name] = MappingProxyType(declared)
    return MappingProxyType(frozen)


def _shape_key(
    inputs: Mapping[str, np.ndarray], dynamic_axes: DynamicAxes
) -> Tuple[ShapeKey, Dict[str, Dict[int, str]]]:
    """形状 + dtype 的规范化键；声明过的轴替换成符号。

    同时返回规范化后的动态轴声明（负轴已换算成正轴），供构图后的校验使用。
    """
    missing = [name for name in dynamic_axes if name not in inputs]
    if missing:
        raise ValueError(
            f"dynamic_axes 声明了这些输入，但本次 run 没有提供: {missing}"
        )
    entries = []
    declared: Dict[str, Dict[int, str]] = {}
    for name in sorted(inputs):
        value = inputs[name]
        try:
            shape = tuple(int(d) for d in value.shape)
        except AttributeError:
            raise TypeError(
                f"输入 {name!r} 必须是数组（需要 .shape），得到 {type(value).__name__}"
            ) from None
        axes = dynamic_axes.get(name)
        if axes:
            shape, declared[name] = _symbolic_shape(name, shape, axes)
        dtype = str(getattr(value, "dtype", "unknown"))
        entries.append((name, shape, dtype))
    return tuple(entries), declared


def _symbolic_shape(
    name: str, shape: Tuple[int, ...], axes: Mapping[int, str]
) -> Tuple[Tuple[Dim, ...], Dict[int, str]]:
    """把声明过的轴换成符号，其余维保持具体整数。"""
    rank = len(shape)
    resolved: Dict[int, str] = {}
    for axis, symbol in axes.items():
        index = axis if axis >= 0 else axis + rank
        if not 0 <= index < rank:
            raise ValueError(
                f"输入 %{name} 声明了动态轴 {axis}，但本次形状 {shape} "
                f"的秩是 {rank}"
            )
        if index in resolved:
            raise ValueError(
                f"输入 %{name} 的动态轴 {index} 被重复声明"
                f"（{resolved[index]!r} 与 {symbol!r}）"
            )
        resolved[index] = symbol
    template = list(shape)
    for index, symbol in resolved.items():
        template[index] = symbol
    return tuple(template), resolved


def _require_traced_inputs(
    graph: Graph, declared: Mapping[str, Mapping[int, str]]
) -> None:
    """构图后确认声明的轴真的是符号。

    写在模板上的符号如果被 ``build`` 换成了整数，第一次运行照样能过（数值恰好
    相同），下一次换形状才会以"静态维不匹配"失败。那种错误离原因太远，所以在
    第一次构图后就当场拒绝。
    """
    if not declared:
        return
    types = {value.name: value.type for value in graph.inputs}
    for name, axes in declared.items():
        type_ = types.get(name)
        if type_ is None:
            raise ValueError(
                f"dynamic_axes 声明了输入 %{name}，但 build 返回的图里没有这个输入；"
                f"图的实际输入是 {sorted(types)}"
            )
        for axis, symbol in axes.items():
            dim = type_.shape[axis]
            if isinstance(dim, int):
                raise ValueError(
                    f"dynamic_axes 把 %{name} 的第 {axis} 维声明为符号 {symbol!r}，"
                    f"但图里的输入类型是 {type_}。build 必须使用传入的形状模板，"
                    "否则这张图无法在换形状后重放"
                )
