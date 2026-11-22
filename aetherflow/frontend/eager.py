"""eager 前端：把"写起来像动态图"接在静态 IR 上。

这个模块**不改变执行模型**。图仍然是静态的、可序列化的、能在构图期做
后端能力预检的；变的只是写图的手感：重载运算符、标量自动转常量、
按 Python 的 for/if（只看 Python 值的那部分）叠加层。

为什么不做真正的 define-by-run 执行：三件事只有静态图能给——

1. 构图期就能回答"目标后端接不接得住"，而不是跑到一半抛错；
2. 反向图必须自包含、可序列化，才能独立分发与回放；
3. CUDA Graph 捕获要求执行路径与形状固定（派发开销 800μs → 8μs）。

所以缺的是前端手感，不是执行模型。补前端，不动 IR。
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..ir.graph import Graph, Value
from ..ir.ops import OPS
from ..ir.types import DType, TensorType


class Tensor:
    """构图期的张量代理。

    只持有 ``Value`` 与所属图：没有数据，也没有设备。所有运算都直接往图里
    加节点，所以"前向"就是构图过程本身。

    ``requires_grad`` / ``grad`` 是给"写一行、立刻拿梯度"用的：把输入标成
    需要梯度，调一次 ``backward``，梯度就落回这个对象上。求导本身仍是
    "再构一张反向图并执行"（``ir/grad.py``），没有偷偷引入第二套机制。
    """

    __slots__ = ("_value", "_graph", "_grad", "_requires_grad")

    def __init__(self, value: Value, graph: "EagerGraph") -> None:
        self._value = value
        self._graph = graph
        self._grad: Optional[np.ndarray] = None
        self._requires_grad = False

    # -- 只读属性 -----------------------------------------------------

    @property
    def value(self) -> Value:
        return self._value

    @property
    def graph(self) -> "EagerGraph":
        return self._graph

    @property
    def name(self) -> str:
        return self._value.name

    @property
    def type(self) -> TensorType:
        return self._value.type

    @property
    def shape(self) -> Tuple:
        return self._value.type.shape

    @property
    def dtype(self) -> DType:
        return self._value.type.dtype

    # -- 梯度 ---------------------------------------------------------

    @property
    def grad(self) -> Optional[np.ndarray]:
        """上一次 ``backward`` 写回的梯度；还没算过就是 ``None``。"""
        return self._grad

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, flag: bool) -> None:
        self._requires_grad = bool(flag)

    def backward(
        self,
        feed: Optional[Mapping[str, np.ndarray]] = None,
        wrt: Optional[Sequence["Tensor"]] = None,
        backend=None,
    ) -> Dict[str, np.ndarray]:
        """以本张量（必须是图输出）为损失，立即求导并写回 ``grad``。"""
        return self._graph.backward(self, feed=feed, wrt=wrt, backend=backend)

    # -- 运算 ---------------------------------------------------------

    def _coerce(self, other) -> Value:
        """把另一侧操作数统一成 Value。

        只接受 Python 标量与同图 Tensor。numpy/cupy 数组刻意不接受：
        它们的形状不会进 IR，等于绕过了类型推导与后端能力预检。
        """
        if isinstance(other, Tensor):
            if other.graph is not self._graph:
                raise ValueError("两个张量属于不同的图：eager 前端不支持跨图运算")
            return other.value
        if isinstance(other, bool) or not isinstance(other, (int, float)):
            raise TypeError(
                f"只支持与 Python 标量或同图张量运算，收到 {type(other).__name__}；"
                "numpy / cupy 数组请先显式建成图输入"
            )
        return self._graph.constant(float(other), dtype=self.dtype).value

    def __add__(self, other):
        return self._graph.add_node("add", [self._value, self._coerce(other)])

    __radd__ = __add__

    def __sub__(self, other):
        return self._graph.add_node("sub", [self._value, self._coerce(other)])

    def __rsub__(self, other):
        return self._graph.add_node("sub", [self._coerce(other), self._value])

    def __mul__(self, other):
        return self._graph.add_node("mul", [self._value, self._coerce(other)])

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._graph.add_node("div", [self._value, self._coerce(other)])

    def __rtruediv__(self, other):
        return self._graph.add_node("div", [self._coerce(other), self._value])

    def __matmul__(self, other):
        return self._graph.add_node("matmul", [self._value, self._coerce(other)])

    def __rmatmul__(self, other):
        return self._graph.add_node("matmul", [self._coerce(other), self._value])

    def __neg__(self):
        return self._graph.add_node("neg", [self._value])

    def __getattr__(self, name: str):
        """算子方法：``x.relu()`` 等价于 ``g.add_node("relu", [x])``。

        签名与 ``Graph`` 的构图语法糖完全一致：位置参数是算子输入，关键字
        参数是属性（``x.reshape(shape=(...))``、``x.transpose(perm=(0, 2, 1))``）。
        不另造一套 PyTorch 风格签名，避免同一个算子出现两种参数约定。
        """
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in OPS:
            raise AttributeError(f"Tensor 没有属性 {name!r}，也不是已知算子")
        op_name = name

        def call(*args, **attrs):
            return self._graph.add_node(op_name, (self, *args), attrs or None)

        return call

    # -- 明确的拒绝 ---------------------------------------------------

    def __bool__(self):
        """构图期没有张量值，任何依赖它的分支都是错的，必须当场停下。

        静默取一个"编译期真值"会让控制流悄悄走错分支，而这种错误在
        数值对拍里看不出来——所以这里报错，而不是猜。
        """
        raise TypeError(
            "eager 前端不支持依赖张量值的控制流（if / while / and / or）。"
            "构图期没有张量值；请改用 Python 常量，或把分支留到图外用两次执行表达"
        )

    __nonzero__ = __bool__

    def __getitem__(self, index):
        raise TypeError(
            "Tensor 不支持下标切片：IR 的 slice 需要显式维与范围，"
            "请用 g.slice(x, dim=..., start=..., stop=...)"
        )

    def __repr__(self) -> str:
        return f"<Tensor {self._value.type} %{self._value.name}>"


class EagerGraph(Graph):
    """``Graph`` 的语法糖版本：输入与节点都返回 ``Tensor``。

    除此之外与 ``Graph`` 完全等价——IR、verifier、自动微分、序列化都不区分
    这张图是手写的还是用运算符写出来的。

    另加一条 ``backward``：以某个输出为损失，**立即**建反向图并执行，把梯度
    写回参数 ``Tensor`` 的 ``.grad``。它只是把 ``ir.grad.backward`` 与一次
    ``Backend.execute`` 串起来，并缓存反向图；执行模型仍然是静态图。
    """

    def input(self, name: str, type: TensorType) -> Tensor:
        tensor = Tensor(super().input(name, type), self)
        self._remember(tensor)
        self._input_tensors.append(tensor)
        return tensor

    def add_node(
        self,
        op_name: str,
        args: Sequence,
        attrs: Optional[dict] = None,
    ) -> Tensor:
        values = [a.value if isinstance(a, Tensor) else a for a in args]
        tensor = Tensor(super().add_node(op_name, values, attrs), self)
        self._remember(tensor)
        return tensor

    def constant(
        self, value: float = 0.0, dtype: Optional[DType] = None, shape=(), data=None
    ) -> Tensor:
        # Graph.constant 内部走的是 self.add_node，即本类重载后的版本，
        # 已经返回 Tensor；这里若再包一层，_value 会变成 Tensor 而不是 Value。
        return super().constant(value, dtype, shape, data)

    def output(self, value, name: Optional[str] = None) -> None:
        if isinstance(value, Tensor):
            if value.graph is not self:
                raise ValueError("输出张量属于另一张图")
            value = value.value
        super().output(value, name)

    # -- 即时求导 -----------------------------------------------------

    def backward(
        self,
        loss: Tensor,
        feed: Optional[Mapping[str, np.ndarray]] = None,
        wrt: Optional[Sequence[Tensor]] = None,
        backend=None,
    ) -> Dict[str, np.ndarray]:
        """以 ``loss``（必须是本图输出）为损失求导，返回 ``{参数名: 梯度}``。

        ``wrt`` 缺省取所有 ``requires_grad=True`` 的图输入；一个都没有时
        直接报错——静默返回空字典会让"忘了标参数"变成"训练不动"的谜题。

        ``feed`` 必须给出全部图输入的数值（参数也在里面）：反向图会重放
        前向，缺一个输入就少一块值。
        """
        from ..backends.reference_backend import ReferenceBackend

        if not isinstance(loss, Tensor):
            raise TypeError(f"loss 必须是 Tensor，得到 {type(loss).__name__}")
        if loss.graph is not self:
            raise ValueError("loss 属于另一张图")
        aliases = self.output_names
        if loss.value not in self.outputs:
            raise ValueError(
                f"%{loss.name} 不是图输出，无法作为损失；请先调用 g.output({loss.name})"
            )
        if tuple(loss.shape) != ():
            # 种子是标量 1。非标量损失要先自己归约，否则这里要么形状对不上
            # （晦涩的后端报错），要么语义上等于"对每个元素各求一次梯度"。
            raise ValueError(
                f"损失必须是标量，%{loss.name} 的形状是 {loss.shape}；请先归约成标量"
            )
        if wrt is None:
            targets: List[Tensor] = [t for t in self._input_tensors if t.requires_grad]
            if not targets:
                raise ValueError(
                    "没有任何图输入标记 requires_grad=True，求导目标为空；"
                    "请给参数张量设置 requires_grad"
                )
        else:
            targets = list(wrt)
        for tensor in targets:
            if not isinstance(tensor, Tensor) or tensor.graph is not self:
                raise ValueError(f"求导目标 {tensor!r} 不是本图的张量")

        bwd = self._backward_graph(targets)
        values: Dict[str, np.ndarray] = {
            key: np.asarray(value) for key, value in dict(feed or {}).items()
        }
        # 与 Trainer 同一条规则：反向图为每个输出都开了种子，不参与损失的
        # 输出喂零，调用方不必自己补。
        for value, alias in zip(self.outputs, aliases):
            seed = "d" + alias
            if seed not in values:
                values[seed] = np.zeros(
                    tuple(int(d) for d in value.type.shape),
                    dtype=value.type.dtype.np_dtype,
                )
        values["d" + aliases[self.outputs.index(loss.value)]] = np.asarray(
            1.0, dtype=loss.dtype.np_dtype
        )

        engine = backend if backend is not None else ReferenceBackend()
        result = engine.execute(bwd, values)
        grads: Dict[str, np.ndarray] = {}
        for tensor in targets:
            grad = result["d" + tensor.name]
            tensor._grad = grad
            grads[tensor.name] = grad
        return grads

    def _backward_graph(self, targets: Sequence[Tensor]) -> Graph:
        """按 (图版本, 求导目标) 缓存反向图：形状不变时只建一次。"""
        from ..ir.grad import backward

        key = (self.revision, tuple(t.value.id for t in targets))
        cache = getattr(self, "_bwd_cache", None)
        if cache is None:
            cache = (None, None)
        if cache[0] == key:
            return cache[1]
        bwd = backward(self, wrt=[t.value for t in targets])
        self._bwd_cache = (key, bwd)
        return bwd

    def _replace_tensor(self, old: Tensor, new: Tensor) -> None:
        """把记录里的前端张量换成子类实例（``Parameter`` 用）。

        IR 不变：``new.value`` 必须就是 ``old.value``。由 ``input()`` 造出的
        张量同时进 ``_tensors`` 与 ``_input_tensors``，两处都要换，否则
        ``backward`` 的缺省求导目标会漏掉参数。
        """
        if new.value is not old.value:
            raise ValueError("替换张量必须共用同一个 Value")
        self._tensors[new.value.id] = new
        for index, tensor in enumerate(self._input_tensors):
            if tensor is old:
                self._input_tensors[index] = new
                return
        raise ValueError(f"%{old.name} 不在图输入记录里")

    def _remember(self, tensor: Tensor) -> None:
        """记住前端 Tensor，供 ``backward`` 回写梯度。

        这个名字不能叫 ``_inputs``：``Graph`` 用它存 IR 里的输入 ``Value``。
        """
        tensors = getattr(self, "_tensors", None)
        if tensors is None:
            tensors = {}
            self._tensors = tensors
            self._input_tensors = []
        tensors[tensor.value.id] = tensor

