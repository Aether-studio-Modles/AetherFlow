"""模型定义：把「声明参数 + 前向构图」收成一个类。

本框架的图是静态的：参数是图输入，节点在构图期一次建好，之后靠重放执行。
``Module`` 只做三件事，不引入第二套执行机制：

1. 声明参数（图输入 + 主机数组），名字带前缀、按名字稳定初始化；
2. 第一次调用时执行 ``forward`` 构图，并把返回值登记为图输出；
3. 提供 ``feed()`` / ``arrays()`` / ``state_dict()``，直接交给 ``Trainer``。

因此这里**不是** define-by-run：第二次调用不会重新构图，而是报错并提示用
``run()`` 重放。需要换形状或改结构时新建一个 Module 实例即可。

用法::

    class MLP(Module):
        def __init__(self, hidden, out):
            super().__init__()
            self.w1 = self.param("w1", (8, hidden), init="he")
            self.b1 = self.param("b1", (hidden,))
            self.w2 = self.param("w2", (hidden, out), init="xavier")

        def forward(self, x):
            return self.op("relu", self.op("add", self.op("matmul", x, self.w1), self.b1)) @ self.w2

    model = MLP(16, 4)
    y = model(x)                        # 构图一次
    trainer = Trainer(model.graph, ParameterStore(model.arrays()), ...)
    result = model.run({"x": batch})    # 用同一张图重放

参数命名是扁平字符串，嵌套结构用点号前缀表达（``blocks.0.wq``）。这样
HelixLM 这类多层模型不必先有「子模块」抽象也能写；真正的嵌套模块等 P1
再补，避免在 P0 引入两套参数命名规则。
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..ir.types import DType, TensorType
from ..train.params import init_array
from .eager import EagerGraph, Tensor

__all__ = ["Module", "Parameter"]

_DIM = Any


class Parameter(Tensor):
    """图输入 + 主机数组。

    值是构图期的 ``Value``，``data`` 是执行期喂进去的数组。两者分开是静态图
    的前提：图只认名字与形状，数组可以每步被优化器原地改写。

    ``data`` 的 setter 只做原地写入，不替换数组对象——``ParameterStore`` 与
    优化器持有的是同一个引用，换对象会让它们更新到旧数组上。
    """

    __slots__ = ("_data",)

    def __init__(
        self,
        value,
        graph: EagerGraph,
        data: np.ndarray,
        *,
        requires_grad: bool = True,
    ) -> None:
        super().__init__(value, graph)
        self._data = data
        self.requires_grad = bool(requires_grad)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value) -> None:
        array = np.asarray(value)
        if tuple(array.shape) != tuple(self._data.shape):
            raise ValueError(
                f"参数 {self.name!r} 的形状是 {tuple(self._data.shape)}，"
                f"收到 {tuple(array.shape)}"
            )
        self._data[...] = array


class Module:
    """一组参数与一次前向构图。

    子类在 ``__init__`` 里用 ``self.param(...)`` 声明参数，在 ``forward`` 里
    构图。``Module`` 不持有后端：执行由 ``run()`` 或 ``Trainer`` 负责。
    """

    def __init__(self, name: str = "") -> None:
        self._name = str(name)
        self._graph = EagerGraph()
        self._params: Dict[str, Parameter] = {}
        self._built = False

    # -- 描述 ---------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def graph(self) -> EagerGraph:
        return self._graph

    @property
    def built(self) -> bool:
        """是否已经完成一次构图。"""
        return self._built

    def _qualify(self, name: str) -> str:
        name = str(name)
        if not name:
            raise ValueError("参数名不能为空")
        return f"{self._name}.{name}" if self._name else name

    def param(
        self,
        name: str,
        shape: Sequence[_DIM],
        dtype: DType = DType.F32,
        *,
        init: Optional[str] = None,
        seed: int = 0,
        value: float = 0.0,
        requires_grad: bool = True,
    ) -> Parameter:
        """声明一个参数：图输入 + 按策略初始化的主机数组。

        ``shape`` 必须是静态整数：参数不参与符号形状推导，带符号维的参数会
        让初始化与优化器状态都没有确定大小。
        """
        full = self._qualify(name)
        if any(not isinstance(d, (int, np.integer)) for d in shape):
            raise ValueError(
                f"参数 {full!r} 的形状必须是静态整数，收到 {tuple(shape)}"
            )
        dims = tuple(int(d) for d in shape)
        try:
            dtype = DType(dtype)
        except ValueError as exc:
            raise ValueError(f"参数 {full!r} 的 dtype 不认识: {dtype!r}") from exc
        data = init_array(full, dims, dtype, init=init, seed=seed, value=value)
        base = self._graph.input(full, TensorType(dtype, dims))
        parameter = Parameter(
            base.value, self._graph, data, requires_grad=requires_grad
        )
        # 让图的前端记录也指向 Parameter：backward 缺省取
        # _input_tensors 里 requires_grad 的张量，漏换就会"训练不动"。
        self._graph._replace_tensor(base, parameter)
        self._params[full] = parameter
        return parameter

    # -- 构图 ---------------------------------------------------------

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError(f"{type(self).__name__} 必须实现 forward")

    def __call__(self, *args, **kwargs) -> Any:
        """构图一次，并把 ``forward`` 的返回值登记为图输出。

        第二次调用直接报错：静态图不会为第二次调用重建节点，静默复用上一次
        的输出只会让人以为参数变了、结果没变。
        """
        if self._built:
            raise RuntimeError(
                f"{type(self).__name__} 已经构图；静态图不会为第二次调用重建。"
                "请用 run(inputs) 重放，或新建一个实例。"
            )
        self._built = True
        result = self.forward(*args, **kwargs)
        outputs = result if isinstance(result, (tuple, list)) else (result,)
        if not outputs:
            raise RuntimeError(f"{type(self).__name__}.forward 没有返回任何张量")
        for item in outputs:
            if not isinstance(item, Tensor):
                raise TypeError(
                    f"{type(self).__name__}.forward 只能返回 Tensor，"
                    f"收到 {type(item).__name__}"
                )
            if item.graph is not self._graph:
                raise ValueError("forward 返回的张量属于另一张图")
            if item.value not in self._graph.outputs:
                self._graph.output(item)
        return result

    def op(self, name: str, *args, **attrs) -> Tensor:
        """加一个图节点。``self.op("relu", x)`` 等价于手写 ``add_node``。"""
        return self._graph.add_node(name, args, attrs or None)

    def p(self, name: str):
        """参数的构图值，供 ``Graph`` 的 DSL 方法直接使用。

        ``self.p("w")`` 与 ``self.w.value`` 等价；保留它是因为已有的构图代码
        （如 HelixLM）习惯把参数当 ``Value`` 传给 ``g.matmul``。
        """
        full = self._qualify(name)
        try:
            return self._params[full].value
        except KeyError as exc:
            raise KeyError(f"没有声明参数 {full!r}") from exc

    def adopt(self, name: str, value, data, *, requires_grad: bool = True) -> Parameter:
        """接管本图上已经建好的输入，登记为参数。

        迁移路径用：已有的构图函数自己 ``graph.input(...)`` 声明参数，先让它
        把图建完，再用 ``adopt`` 把输入包成 ``Parameter``，从而拿到统一的
        ``state_dict`` / 优化器视图。``name`` 是图里的完整名字，不再加前缀。
        传入的 ``value`` 可以是 ``Value``，也可以是对应的 ``Tensor``。
        """
        if name in self._params:
            raise ValueError(f"参数 {name!r} 已经登记")
        if isinstance(value, Tensor):
            if value.graph is not self._graph:
                raise ValueError("传入的张量属于另一张图")
            value = value.value
        try:
            existing = self._graph.find(name)
        except KeyError as exc:
            raise ValueError(f"图里没有名为 {name!r} 的输入") from exc
        if existing is not value:
            raise ValueError(f"图里的 {name!r} 与传入的值不是同一个对象")
        registered = self._graph._tensors.get(value.id)
        if registered is None:
            raise ValueError(
                f"%{name} 不是通过本图 input() 建的张量，无法登记为参数"
            )
        # 必须拷贝：调用方常把别的框架的张量直接 .numpy() 传进来，那是共享
        # 内存的视图。不拷贝就会"两个框架悄悄写同一份权重"，双方的优化器
        # 各自走一步，参数却只前进了一步（甚至相互抵消），对拍时极难定位。
        array = np.array(data, dtype=existing.type.dtype.np_dtype, copy=True)
        want = tuple(existing.type.shape)
        if tuple(array.shape) != want:
            raise ValueError(
                f"参数 {name!r} 的形状是 {want}，收到 {tuple(array.shape)}"
            )
        parameter = Parameter(value, self._graph, array, requires_grad=requires_grad)
        self._graph._replace_tensor(registered, parameter)
        self._params[name] = parameter
        return parameter

    # -- 参数视图 -----------------------------------------------------

    def named_parameters(self) -> Iterator[Tuple[str, Parameter]]:
        yield from self._params.items()

    def parameters(self) -> Iterator[Parameter]:
        for _name, parameter in self._params.items():
            yield parameter

    def arrays(self) -> Dict[str, np.ndarray]:
        """``{参数名: 数组引用}``，可直接交给 ``ParameterStore``。

        返回的是引用而不是副本：优化器原地更新后，下一次重放自动用新值。
        """
        return {name: parameter.data for name, parameter in self._params.items()}

    def state_dict(self) -> Dict[str, np.ndarray]:
        """参数的深拷贝，用于保存与对拍。"""
        return {name: parameter.data.copy() for name, parameter in self._params.items()}

    def load_state_dict(
        self, state: Mapping[str, np.ndarray], *, strict: bool = True
    ) -> None:
        """把外部参数写回 ``data``（原地），并校验键与形状。"""
        missing = sorted(set(self._params) - set(state))
        unexpected = sorted(set(state) - set(self._params))
        if strict and (missing or unexpected):
            raise KeyError(
                f"参数不匹配：缺少 {missing}，多出 {unexpected}"
            )
        for name in set(self._params) & set(state):
            self._params[name].data = state[name]

    def zero_grad(self) -> None:
        for parameter in self._params.values():
            parameter._grad = None

    def feed(self, inputs: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """执行用的输入表：外部输入 + 当前参数值。

        同名冲突直接报错：参数覆盖是评测时才有意为之的行为，不该由一次
        手滑的键名重复静默完成。
        """
        values = dict(inputs or {})
        clash = sorted(set(values) & set(self._params))
        if clash:
            raise ValueError(f"输入与参数重名: {clash}")
        values.update(self.arrays())
        return values

    def run(self, inputs: Optional[Mapping[str, Any]] = None, *, backend=None):
        """用已构好的图重放一次，返回输出数组（单个输出时直接返回数组）。"""
        if not self._built:
            raise RuntimeError(
                f"{type(self).__name__} 还没有构图：先调用一次 model(*inputs)"
            )
        from ..backends.reference_backend import ReferenceBackend

        engine = backend if backend is not None else ReferenceBackend()
        arrays = engine.execute(self._graph, self.feed(inputs))
        names = list(self._graph.output_names)
        if not names:
            raise RuntimeError("图没有输出")
        if len(names) == 1:
            return arrays[names[0]]
        return [arrays[name] for name in names]

    def __repr__(self) -> str:
        inner = ", ".join(
            f"{name}={tuple(parameter.shape)}"
            for name, parameter in self._params.items()
        )
        return f"{type(self).__name__}({inner})"
