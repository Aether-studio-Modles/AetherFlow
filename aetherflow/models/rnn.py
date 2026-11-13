"""循环网络构件：RNN / LSTM / GRU 的时间展开。

为什么是"展开"而不是新算子：IR 是静态图，没有循环控制流。把 T 步展开成
T 份子图之后，递归网络与普通前馈网络在 IR 里毫无区别——反向、序列化、
后端承接全都沿用同一套机制。代价是 **T 与 batch 必须静态**：初始隐状态是
一个零常量节点，常量形状写在属性里。

序列在构件内部用"时间步的 ``Value`` 列表"表示，不引入 stack 算子：
``concat`` 能拼，但需要先把 T 个张量变成参数个数的输入，反而更绕；
调用方按需自行拼接（例如取最后一步做分类）。

三种 cell 的参数布局都按 PyTorch 的语义：

- RNN : ``h' = tanh(x Wx + h Wh + b)``
- LSTM: 门顺序 ``i, f, g, o``，``c' = f*c + i*tanh(g)``，``h' = o*tanh(c')``
- GRU : 门顺序 ``z, r, n``，``n = tanh(x Wx_n + (r*h) Wh_n + b_n)``，
  ``h' = (1-z)*n + z*h``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from ..ir.graph import Graph, Value
from ..ir.types import f32

__all__ = ["RNNSpec", "build_rnn"]

_CELLS = ("rnn", "lstm", "gru")


def _split(g: Graph, value: Value, parts: int, axis: int = 1) -> List[Value]:
    """沿 ``axis`` 切成等长 ``parts`` 段。

    用 ``slice`` 而不是引入 split 算子：门控网络里的段数在构图期已知，
    切片就是它最直接的语义。
    """
    dim = value.type.shape[axis]
    if not isinstance(dim, int) or dim % parts != 0:
        raise ValueError(f"无法把第 {axis} 维 {dim} 等分成 {parts} 段")
    step = dim // parts
    return [
        g.slice(value, dim=axis, start=i * step, stop=(i + 1) * step)
        for i in range(parts)
    ]


@dataclass(frozen=True)
class RNNSpec:
    input_size: int = 8
    hidden_size: int = 16
    num_layers: int = 1
    steps: int = 12
    batch: int = 4
    cell: str = "lstm"

    def __post_init__(self) -> None:
        for name in ("input_size", "hidden_size", "num_layers", "steps", "batch"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} 必须为正，得到 {getattr(self, name)}")
        if self.cell not in _CELLS:
            raise ValueError(f"未知 cell {self.cell!r}；可用: {', '.join(_CELLS)}")

    @property
    def gates(self) -> int:
        return {"rnn": 1, "lstm": 4, "gru": 3}[self.cell]

    @property
    def param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        h, g = self.hidden_size, self.gates
        shapes: Dict[str, Tuple[int, ...]] = {}
        for i in range(self.num_layers):
            fan_in = self.input_size if i == 0 else h
            p = f"l{i}_"
            shapes[p + "wx"] = (fan_in, h * g)
            shapes[p + "wh"] = (h, h * g)
            shapes[p + "b"] = (h * g,)
        return shapes


def _init_state(g: Graph, spec: RNNSpec) -> Value:
    return g.constant(0.0, dtype=f32, shape=(spec.batch, spec.hidden_size))


def _step(
    g: Graph,
    cell: str,
    x: Value,
    h: Value,
    c: Value | None,
    p: Dict[str, Value],
    prefix: str,
) -> Tuple[Value, Value | None]:
    pre = g.add(
        g.add(g.matmul(x, p[prefix + "wx"]), g.matmul(h, p[prefix + "wh"])),
        p[prefix + "b"],
    )

    if cell == "rnn":
        return g.tanh(pre), None

    if cell == "lstm":
        i, f, gg, o = _split(g, pre, 4)
        assert c is not None
        c_next = g.add(g.mul(f, c), g.mul(i, g.tanh(gg)))
        return g.mul(o, g.tanh(c_next)), c_next

    # GRU：z/r 用 h 的投影，n 的 h 部分必须先乘 r，所以拆成 gx 与 gh 两半
    gx = g.add(g.matmul(x, p[prefix + "wx"]), p[prefix + "b"])
    gh = g.matmul(h, p[prefix + "wh"])
    gx_z, gx_r, gx_n = _split(g, gx, 3)
    gh_z, gh_r, gh_n = _split(g, gh, 3)
    z = g.sigmoid(g.add(gx_z, gh_z))
    r = g.sigmoid(g.add(gx_r, gh_r))
    n = g.tanh(g.add(gx_n, g.mul(r, gh_n)))
    one = g.constant(1.0, dtype=f32, shape=())
    return g.add(g.mul(g.sub(one, z), n), g.mul(z, h)), None


def build_rnn(
    g: Graph, spec: RNNSpec | None = None
) -> Tuple[Dict[str, Value], List[Value], Dict[str, Value]]:
    """构建 T 步展开的循环网络。

    返回 ``(输入表, 每步输出列表, 末态字典)``。末态字典的键为
    ``l{i}_h`` 与（仅 LSTM）``l{i}_c``。
    """
    spec = spec or RNNSpec()
    inputs: Dict[str, Value] = {
        "x": g.input("x", f32[spec.batch, spec.steps, spec.input_size])
    }
    for name, shape in spec.param_shapes.items():
        inputs[name] = g.input(name, f32[shape])

    seq: Sequence[Value] = [
        g.reshape(
            g.slice(inputs["x"], dim=1, start=t, stop=t + 1),
            shape=(spec.batch, spec.input_size),
        )
        for t in range(spec.steps)
    ]

    state: Dict[str, Value] = {}
    for layer in range(spec.num_layers):
        prefix = f"l{layer}_"
        h = _init_state(g, spec)
        c = _init_state(g, spec) if spec.cell == "lstm" else None
        outs: List[Value] = []
        for x in seq:
            h, c = _step(g, spec.cell, x, h, c, inputs, prefix)
            outs.append(h)
        state[prefix + "h"] = h
        if c is not None:
            state[prefix + "c"] = c
        seq = outs
    return inputs, list(seq), state
