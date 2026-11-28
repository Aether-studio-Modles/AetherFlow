"""torch.export -> AetherFlow IR。

只做"翻译"，不做任何优化；优化属于中层 IR 的职责。

设计约束：
1. torch 只在模块内延迟导入，缺 torch 时本模块导入不报错，调用时报明确错误。
2. 只翻译已知的 aten 算子。未知算子抛 ``UnsupportedOperatorError``，
   绝不静默跳过或用近似算子代替——那会让"对齐失败"变成不可解释的问题。
3. 参数一律降级为图输入，运行时由 ``ImportedModel.feed`` 组装；``torch.export``
   提升出来的常量张量（``lifted_tensor`` / ``get_attr``）落成 IR 的 ``constant``
   节点，不混进用户输入——否则调用方会被要求喂一个本该写在图里的值（5.224）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..ir.graph import Graph, Value
from ..ir.types import DType, TensorType, bf16, boolean, f16, f32, f64, i32
from .compat import TorchCompatibility, check_torch

try:  # 可选依赖：没有 torch 的机器上内核仍可用
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


class UnsupportedOperatorError(NotImplementedError):
    """目标算子在当前 adapter 里没有对应降级规则。"""


_TORCH_DTYPE = {
    "float32": f32,
    "float64": f64,
    "float16": f16,
    "bfloat16": bf16,
    # 整数只有 i32 一种载体（IR 没有 i64）：int32 无损，int64 收窄。
    # 收窄不是"静默截断"——执行器入口的 ``_check_input_dtype`` 会先验范围，
    # 越界值在入口就报错（见 5.181）。
    "int32": i32,
    "int64": i32,
    # bool 是掩码载体：比较 / logical_* / where / masked_fill 的条件，不参与
    # 算术；显式 cast（``to(dtype=torch.bool)`` / ``x.bool()`` / ``type_as``）
    # 可以双向进出，落成 IR 的 ``cast``（见 5.340）。
    "bool": boolean,
}

#: 导入时允许落成 ``constant`` 节点的最大元素数。与常量折叠的上限同量级：
#: 再大的"常量"往 attrs 里塞，序列化体积和反序列化开销都会失控，应该改成
#: 参数或显式输入。
MAX_IMPORTED_CONSTANT_ELEMENTS = 65536

#: IR dtype -> torch dtype 名。用于把常量构造（zeros/full/arange）落成张量；
#: i32 的载体在 torch 里用 int32（IR 没有 i64）。
_IR_TO_TORCH = {
    "f16": "float16",
    "bf16": "bfloat16",
    "f32": "float32",
    "f64": "float64",
    "i32": "int32",
    "bool": "bool",
}


@dataclass
class ImportedModel:
    """一次导入的产物：一张 IR 图 + 参数取值 + 组装 feed 的方法。"""

    graph: Graph
    parameters: Dict[str, np.ndarray] = field(default_factory=dict)
    user_inputs: List[str] = field(default_factory=list)

    def feed(self, **user_values: np.ndarray) -> Dict[str, np.ndarray]:
        for name in user_values:
            if name not in self.user_inputs:
                raise KeyError(f"{name} 不是该模型的用户输入，可选: {self.user_inputs}")
        missing = [n for n in self.user_inputs if n not in user_values]
        if missing:
            raise KeyError(f"缺少用户输入: {missing}")
        merged = dict(self.parameters)
        merged.update({k: np.asarray(v) for k, v in user_values.items()})
        return merged

    def describe(self) -> str:
        lines = [f"图 {self.graph.name}: {len(self.graph.nodes)} 个节点"]
        lines.append(f"  参数 {len(self.parameters)} 个，用户输入 {self.user_inputs}")
        for name, arr in self.parameters.items():
            lines.append(f"    {name:24s} {tuple(arr.shape)} {arr.dtype}")
        return "\n".join(lines)


def _require_torch() -> TorchCompatibility:
    """确认 torch 可用**且**在兼容矩阵内，返回对应的矩阵行。

    版本不在矩阵里时抛 ``UnsupportedTorchVersion``（见 :mod:`.compat`）：
    ``torch.export`` 的 IR 细节会随版本变，没验证过的版本不该被当成能用。
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("需要 torch 才能导入 torch.export 的模型；torch 是可选依赖")
    return check_torch(torch)


def _to_dtype(dtype: Any, row: Optional[TorchCompatibility] = None) -> DType:
    key = str(dtype).replace("torch.", "")
    if key not in _TORCH_DTYPE:
        supported = "、".join(sorted(_TORCH_DTYPE))
        raise UnsupportedOperatorError(
            f"不支持的 dtype: {dtype}；矩阵内的 dtype 是 {supported}"
            "（int64 会按 i32 承载，越界值在入口拒绝）"
        )
    if row is not None and not row.accepts_dtype(key):
        raise UnsupportedOperatorError(
            f"dtype {key} 不在 torch 兼容矩阵的 {row.span} 行里"
            f"（该行声明 {list(row.dtypes)}）；声明与入口共用同一份事实来源"
        )
    return _TORCH_DTYPE[key]


def _constant_node(
    g: Graph, value: Any, source: str, row: Optional[TorchCompatibility] = None
) -> Value:
    """把 torch 常量张量落成 IR 的 ``constant`` 节点。

    常量写进 attrs（``data``），所以图仍然自包含、可序列化。超过上限就显式
    拒绝：再大的值应该改成参数或显式输入，而不是塞进图属性里。
    """
    if not hasattr(value, "detach"):
        raise UnsupportedOperatorError(
            f"{source} 不是常量张量（{type(value).__name__}），无法落成 constant 节点"
        )
    tensor = value.detach().cpu()
    dtype = _to_dtype(tensor.dtype, row)
    # numpy 没有 bfloat16：先转 float32 取位；执行器在节点边界按 bf16 量化。
    if str(tensor.dtype) == "torch.bfloat16":
        tensor = tensor.float()
    arr = tensor.numpy()
    if arr.size > MAX_IMPORTED_CONSTANT_ELEMENTS:
        raise UnsupportedOperatorError(
            f"{source} 有 {arr.size} 个元素，超过导入上限 "
            f"{MAX_IMPORTED_CONSTANT_ELEMENTS}；请改成显式输入或参数"
        )
    if dtype is i32 and arr.size and (
        int(arr.min()) < -(2 ** 31) or int(arr.max()) > 2 ** 31 - 1
    ):
        # 直接落成 constant 会绕过执行器入口的范围检查：在这里拒绝，
        # 而不是让 numpy 构造时溢出或静默截断。
        raise UnsupportedOperatorError(
            f"{source} 的整数值超出 i32 范围（IR 只有 i32 载体）；"
            "请改成显式输入，让执行器入口报出具体越界值"
        )
    attrs = {
        "dtype": dtype,
        "shape": tuple(int(dim) for dim in arr.shape),
        "data": arr.tolist(),
    }
    return g.add_node("constant", [], attrs)


def _shape_of(value: Any) -> tuple:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise UnsupportedOperatorError(f"无法从 {value!r} 取得形状")
    dims = []
    for dim in shape:
        # 静态导出里 shape 的元素是 int；torch 的动态维是 SymInt，直接 int()
        # 会触发一次 guard 把它具体化成一个"碰巧这次成立"的数字，导出的图
        # 却不再对别的形状成立。这里显式拒绝，让调用方先固定该维。
        if not isinstance(dim, int):
            raise UnsupportedOperatorError(
                f"torch 导入只接受静态形状，遇到动态维 {dim!r}（{type(dim).__name__}）；"
                "请先用 torch._dynamo.mark_static 固定该维，或按具体输入尺寸导出"
            )
        dims.append(int(dim))
    return tuple(dims)


def _norm_dim(dim: int, rank: int) -> int:
    return dim + rank if dim < 0 else dim


def _int_pair(value: Any, name: str) -> Optional[tuple]:
    """把 aten 的 ``int[2]`` 属性规范成二元组；空列表表示"用默认值"。"""
    if value is None:
        return None
    if isinstance(value, bool):
        raise UnsupportedOperatorError(f"{name} 不能是 bool")
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        if len(value) == 1:
            return (int(value[0]), int(value[0]))
        if len(value) == 2:
            return (int(value[0]), int(value[1]))
    raise UnsupportedOperatorError(f"{name} 必须是 int 或长度 2 的序列，收到 {value!r}")


def import_exported(
    exported_program: Any, name: str = "imported", param_prefix: str = "p_"
) -> ImportedModel:
    """把 ``torch.export.ExportedProgram`` 翻译成 IR 图。"""
    row = _require_torch()

    gm = exported_program.graph_module
    sig = exported_program.graph_signature
    state = {k: v.detach().cpu().numpy() for k, v in exported_program.state_dict.items()}
    param_by_input: Dict[str, str] = dict(getattr(sig, "inputs_to_parameters", {}) or {})
    # buffers（如 BN 的 running_mean / running_var）同样是图输入，但它们不是
    # 可训练参数，只出现在 inputs_to_buffers 里。两者都要从 state_dict 取。
    buffer_by_input: Dict[str, str] = dict(getattr(sig, "inputs_to_buffers", {}) or {})
    state_by_input = {**param_by_input, **buffer_by_input}
    # ``torch.export`` 会把 forward 里的字面量张量提升成 placeholder（名字常见
    # ``c_lifted_tensor_0``）。它们既不是参数也不是缓冲，但同样**不是**用户输入：
    # 按原样走会要求调用方喂一个本该写在图里的常量。这里只认图签名明确登记的
    # 提升常量，其余 placeholder 仍按用户输入处理（不靠名字猜）。
    const_by_input: Dict[str, str] = dict(
        getattr(sig, "inputs_to_lifted_tensor_constants", {}) or {}
    )
    lifted_constants = getattr(exported_program, "tensor_constants", None)
    if lifted_constants is None:
        lifted_constants = getattr(exported_program, "constants", {}) or {}

    g = Graph(name)
    env: Dict[str, Any] = {}
    parameters: Dict[str, np.ndarray] = {}
    user_inputs: List[str] = []

    # torch.export 会把所有 buffer 提升为 placeholder，包括与推理无关的
    # ``num_batches_tracked``（int64 标量）。它们不进 IR 的 dtype 体系，
    # 也不该变成用户必须喂的输入——先算出真正被引用的 placeholder。
    #
    # 训练态 BatchNorm 还会在 SSA 图里留下 ``add_(num_batches_tracked, 1)``：
    # 输出 0 使用者，唯一可观测的效果是改那个 buffer。IR 的值不可变、也没有
    # "图外状态"，这类节点只能跳过（running stats 由训练循环维护，5.26 /
    # 5.208）。只放行"目标确实是 buffer"的节点：写到用户输入或参数上的原地
    # 写仍然交给翻译器明确拒绝，而不是悄悄丢掉。
    #
    # ``ep.run_decompositions()`` 之后还会多出取 save_mean / save_invstd 的
    # ``getitem``：同样是 0 使用者。它们没有可观测效果，跳过即可，否则会在
    # "只保留第 0 个输出"的约定上报错。
    no_op_mutations: set = set()
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.users or not node.args:
            continue
        if getattr(node.target, "__name__", "") == "getitem":
            no_op_mutations.add(node.name)
            continue
        first = node.args[0]
        if getattr(first, "name", None) in buffer_by_input:
            no_op_mutations.add(node.name)

    # 图输出也分两类：用户输出，以及 torch.export 附带的 buffer 更新 / save
    # 中间量（训练态 BN 的 num_batches_tracked、save_mean / save_invstd）。
    # 后者不属于张量语义——它们甚至可能是 IR 不支持的 int64——连同只服务它们
    # 的那部分子图一起不翻译。用户输出的可达闭包用来决定哪些 placeholder
    # 真正需要、哪些节点要翻译。
    def _closure(roots):
        seen: Dict[str, Any] = {}
        stack = list(roots)
        while stack:
            v = stack.pop()
            if v.name in seen:
                continue
            seen[v.name] = v
            stack.extend(getattr(v, "all_input_nodes", ()))
        return seen

    user_outs = set(getattr(sig, "user_outputs", ()) or ())
    output_node = next(n for n in gm.graph.nodes if n.op == "output")
    slots = list(output_node.args[0])
    if user_outs:
        kept_slots = [v for v in slots if v.name in user_outs]
        dropped_slots = [v for v in slots if v.name not in user_outs]
    else:
        kept_slots, dropped_slots = slots, []
    live = _closure(kept_slots)
    dead = {k: v for k, v in _closure(dropped_slots).items() if k not in live}

    used: set = set(live)

    for node in gm.graph.nodes:
        if node.name in dead:
            continue
        if node.op == "placeholder":
            if node.name not in used:
                continue
            if node.name in const_by_input:
                key = const_by_input[node.name]
                if key not in lifted_constants:
                    raise KeyError(
                        f"提升常量 {key!r}（节点 {node.name}）不在导出的常量表里"
                    )
                env[node.name] = _constant_node(
                    g, lifted_constants[key], f"常量 {key!r}", row
                )
                continue
            val = node.meta["val"]
            t = TensorType(_to_dtype(val.dtype, row), _shape_of(val))
            env[node.name] = g.input(node.name, t)
            if node.name in state_by_input:
                key = state_by_input[node.name]
                if key not in state:
                    raise KeyError(f"参数 {key} 不在 state_dict 中")
                parameters[node.name] = state[key]
            else:
                user_inputs.append(node.name)
        elif node.op == "call_function":
            if node.name in no_op_mutations:
                continue  # buffer 的原地自增/写回：图外语义，见上面的说明
            env[node.name] = _translate(g, node, env)
        elif node.op == "get_attr":
            # 有些导出形态把常量留在 GraphModule 的属性上，而不是提升成
            # placeholder。值与 placeholder 常量同源，落成同一个 constant 节点。
            raw = getattr(gm, str(node.target), None)
            if raw is None:
                raise UnsupportedOperatorError(
                    f"节点 {node.name} 引用的属性 {node.target!r} 不在 GraphModule 上"
                )
            env[node.name] = _constant_node(g, raw, f"属性 {node.target!r}", row)
        elif node.op == "output":
            names = list(getattr(sig, "user_outputs", ()) or ())
            for i, v in enumerate(kept_slots):
                out_name = names[i] if i < len(names) else f"out{i}"
                g.output(env[v.name], out_name)
        else:
            raise UnsupportedOperatorError(f"未知节点类型 {node.op}（{node.name}）")

    return ImportedModel(graph=g, parameters=parameters, user_inputs=user_inputs)


def _conv2d(g: Graph, node: Any, args: List[Any], a, n) -> Value:
    """``aten.conv2d`` / ``aten.convolution`` -> IR 的 ``conv2d``。

    两者的参数表不同：``convolution`` 在 dilation 之后还带 ``transposed`` /
    ``output_padding``，且 groups 在最后。转置卷积不引入第二套卷积语义，
    直接落到 ``conv2d_input_grad``（见 ``_conv_transpose2d``）。
    """
    if str(node.target) == "aten.convolution.default":
        transposed = bool(n(6))
        output_padding = _int_pair(n(7), "output_padding") if len(args) > 7 else None
        stride, padding, dilation, groups = n(3), n(4), n(5), int(n(8))
        if transposed:
            return _conv_transpose2d(
                g, node, args, a, stride, padding, dilation,
                output_padding or (0, 0), groups,
            )
        if output_padding is not None and any(output_padding):
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 output_padding={output_padding} 只对转置卷积有意义"
            )
    else:
        stride = n(3) if len(args) > 3 else None
        padding = n(4) if len(args) > 4 else None
        dilation = n(5) if len(args) > 5 else None
        groups = int(n(6)) if len(args) > 6 else 1

    x = a(0)
    w = a(1)
    attrs: Dict[str, Any] = {"groups": groups}
    for key, value in (
        ("stride", _int_pair(stride, "stride")),
        ("padding", _int_pair(padding, "padding")),
        ("dilation", _int_pair(dilation, "dilation")),
    ):
        if value is not None:
            attrs[key] = value

    inputs = [x, w]
    if len(args) > 2 and args[2] is not None:
        inputs.append(a(2))
    return g.add_node("conv2d", inputs, attrs)


def _conv_transpose2d(
    g: Graph, node: Any, args: List[Any], a, stride, padding, dilation,
    output_padding: tuple, groups: int,
) -> Value:
    """``aten.convolution``（transposed=True）-> ``conv2d_input_grad`` + bias。

    转置卷积与"conv2d 对输入的梯度"是同一条散射关系：``y[n,co,oh,ow]`` 收到
    ``x[n,ci,ih,iw]`` 的贡献当且仅当 ``oh = ih*sh - ph + k*dh``。IR 里已经有
    这个语义（``conv2d_input_grad``），所以这里只做布局换算，不新增算子、
    也不引入第二套卷积定义。

    两个细节：

    - ``output_padding`` 只改变输出尺寸（多出的行/列没有输入贡献，恒为 0），
      因此它体现在 ``input_shape`` 上。CUDA 内核按 ``dy`` 形状反推输入尺寸，
      要求 ``output_padding < stride``；更宽的情形显式拒绝，不做近似。
    - ``weight`` 是 ``(C_in, C_out/groups, kH, kW)``，正好等于
      ``conv2d_input_grad`` 期望的 ``(O, C/groups, kH, kW)``——转置卷积的
      "输出梯度通道"就是它的输入通道，不需要转置。
    """
    x, w = a(0), a(1)
    if x.type.rank != 4 or w.type.rank != 4:
        raise UnsupportedOperatorError(
            f"节点 {node.name} 的转置卷积要求 x/weight 都是 NCHW 秩 4，"
            f"得到 {x.type.rank} 与 {w.type.rank}"
        )
    sh, sw = _int_pair(stride, "stride") or (1, 1)
    ph, pw = _int_pair(padding, "padding") or (0, 0)
    dh, dw = _int_pair(dilation, "dilation") or (1, 1)
    oph, opw = output_padding
    if oph >= sh or opw >= sw:
        raise UnsupportedOperatorError(
            f"节点 {node.name} 的 output_padding={(oph, opw)} 不小于 stride={(sh, sw)}；"
            "CUDA 的 conv2d_input_grad 按 dy 形状反推输入尺寸，这种情形会算错"
        )

    n_ = x.type.shape[0]
    cout, kh, kw = w.type.shape[1], w.type.shape[2], w.type.shape[3]
    hin, win = x.type.shape[2], x.type.shape[3]
    for name, value in (("H_in", hin), ("W_in", win), ("kH", kh), ("kW", kw),
                        ("C_out/groups", cout)):
        if not isinstance(value, int):
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的转置卷积需要静态 {name}，收到 {value!r}"
            )
    hout = (hin - 1) * sh - 2 * ph + dh * (kh - 1) + oph + 1
    wout = (win - 1) * sw - 2 * pw + dw * (kw - 1) + opw + 1
    if hout <= 0 or wout <= 0:
        raise UnsupportedOperatorError(
            f"节点 {node.name} 的转置卷积输出尺寸非法: {(hout, wout)}"
        )

    attrs: Dict[str, Any] = {
        "input_shape": (n_, cout * groups, hout, wout),
        "stride": (sh, sw),
        "padding": (ph, pw),
        "dilation": (dh, dw),
        "groups": groups,
    }
    y = g.add_node("conv2d_input_grad", [x, w], attrs)
    if len(args) > 2 and args[2] is not None:
        y = g.add(y, _channel_reshape(g, a(2), 4))
    return y


def _max_pool2d(g: Graph, node: Any, args: List[Any], a, n, with_indices: bool):
    """``aten.max_pool2d`` -> IR 的 ``maxpool2d``。

    ``_with_indices`` 变体会返回 ``(values, indices)``：values 正常降级，
    indices 槽位放 ``None``，下游一旦真去用就会得到明确的报错，
    而不是一个看起来正常、实际错位的张量。

    ``ceil_mode`` 直接进属性（5.154）。此前这里把输入右边界复制几列/行，
    用 floor 模式凑出同样的输出尺寸——对 max 数值上恰好等价，但会改写图结构
    （多两个 concat），也让"用户写的是什么"在 IR 里看不出来。现在 IR 的
    ``maxpool2d`` 与 ``avgpool2d`` 都自己算 ceil 几何，前端只做搬运。
    """
    def opt(i: int, default=None):
        return args[i] if len(args) > i else default

    kernel = _int_pair(opt(1), "kernel")
    if kernel is None:
        raise UnsupportedOperatorError(f"节点 {node.name} 的 maxpool2d 缺少 kernel")
    stride = _int_pair(opt(2), "stride")
    padding = _int_pair(opt(3), "padding") or (0, 0)
    dilation = _int_pair(opt(4), "dilation") or (1, 1)
    ceil_mode = bool(opt(5, False))
    if dilation != (1, 1):
        raise UnsupportedOperatorError(
            f"节点 {node.name} 的 dilation={dilation}，IR 的 maxpool2d 暂不支持膨胀"
        )

    attrs: Dict[str, Any] = {"kernel": kernel, "padding": padding}
    if stride is not None:
        attrs["stride"] = stride
    if ceil_mode:
        attrs["ceil_mode"] = True
    values = g.add_node("maxpool2d", [a(0)], attrs)
    return (values, None) if with_indices else values


def _adaptive_avg_pool2d(
    g: Graph, node: Any, args: List[Any], a, n, kwargs: dict
) -> Value:
    """``aten.adaptive_avg_pool2d`` -> IR 的 ``adaptive_avg_pool2d``。

    ``output_size`` 可能是 int 或长度 1/2 的序列，IR 统一存成 ``[OH, OW]``；
    ``output_size=(1, 1)`` 就是全局平均池化。
    """
    raw = args[1] if len(args) > 1 else kwargs.get("output_size")
    if isinstance(raw, int) and not isinstance(raw, bool):
        size = [int(raw), int(raw)]
    elif isinstance(raw, (list, tuple)) and len(raw) in (1, 2):
        size = [int(raw[0]), int(raw[-1])]
    else:
        raise UnsupportedOperatorError(
            f"节点 {node.name} 的 adaptive_avg_pool2d 的 output_size 无法解析：{raw!r}"
        )
    return g.add_node("adaptive_avg_pool2d", [a(0)], {"output_size": size})


def _scalar(g: Graph, value: float, dtype: DType) -> Value:
    """标量常量节点；形状为空，按 numpy 广播规则参与任意秩的运算。"""
    return g.constant(float(value), dtype=dtype, shape=())


def _mul_alpha(g: Graph, y: Value, alpha, node: Any) -> Value:
    """``add`` / ``sub`` 的 ``alpha``：``x ± alpha*y`` 里的那一次缩放。

    ``alpha`` 是 ATen 的 Python 标量，不是图里的值，所以可以直接折成常量；
    ``alpha == 1``（默认值）时不加节点，保持导出图与手写图一致。
    """
    if isinstance(alpha, (int, float, np.integer, np.floating)):
        value = float(alpha)
        if value == 1.0:
            return y
        return g.mul(y, _scalar(g, value, y.type.dtype))
    raise UnsupportedOperatorError(
        f"节点 {node.name} 的 alpha={alpha!r} 不是 Python 标量，无法在构图期折叠"
    )


def _channel_reshape(g: Graph, value: Value, rank: int) -> Value:
    """把 ``(C,)`` 的通道参数 reshape 成 ``(1, C, 1, ...)``。

    不能直接依赖广播：``(N,C,H,W) * (C,)`` 会被 numpy 对齐到最后一维 W，
    这不是 BatchNorm 想要的通道维。必须先把通道参数放到第 1 维。
    """
    channels = value.type.shape[0]
    if not isinstance(channels, int):
        raise UnsupportedOperatorError(f"通道参数形状含符号维: {value.type.shape}")
    shape = [1] * rank
    shape[1] = channels
    return g.reshape(value, shape=tuple(shape))


def _batch_norm_inference(
    g: Graph, node: Any, args: List[Any], a, n, eps_index: int = 6
) -> Value:
    """推理态 BN -> 用已有原语展开的仿射变换。

    ``aten._native_batch_norm_legit_no_training``（eps 在第 6 个参数）与
    ``aten.batch_norm``（training=False，eps 在第 7 个参数）语义相同，只是
    schema 多一个 ``training`` 与 ``cudnn_enabled``，所以用 ``eps_index`` 区分。

    IR 的 ``batch_norm`` 是训练态（用当前 batch 的统计量），而推理图必须用
    权重里冻结的 ``running_mean`` / ``running_var``。这两者语义不同，不能
    直接复用；但推理态是仿射变换：

        scale = weight / sqrt(var + eps)
        shift = bias - mean * scale
        y     = x * scale + shift

    全部由 ``sqrt`` / ``div`` / ``mul`` / ``sub`` / ``add`` 与常量组成，
    不需要新增算子。返回单元素 tuple，保持与 aten 多输出节点一致的槽位约定。
    """
    x, weight, bias, mean, var = a(0), a(1), a(2), a(3), a(4)
    eps = (
        float(n(eps_index))
        if len(args) > eps_index and args[eps_index] is not None
        else 1e-5
    )
    if x.type.rank < 2:
        raise UnsupportedOperatorError(f"节点 {node.name} 的 BN 输入秩必须 >= 2")
    dtype = x.type.dtype
    channels = x.type.shape[1]
    if not isinstance(channels, int):
        raise UnsupportedOperatorError(
            f"节点 {node.name} 的 BN 通道维含符号 {channels!r}，无法静态 reshape"
        )
    for name, value in (("weight", weight), ("bias", bias), ("mean", mean), ("var", var)):
        if value.type.rank != 1 or value.type.shape[0] != channels:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 BN {name} 形状应为 ({channels},)，得到 {value.type.shape}"
            )
    inv = g.div(_scalar(g, 1.0, dtype), g.sqrt(g.add(var, _scalar(g, eps, dtype))))
    scale = _channel_reshape(g, g.mul(weight, inv), x.type.rank)
    shift = _channel_reshape(g, g.sub(bias, g.mul(mean, g.mul(weight, inv))), x.type.rank)
    return (g.add(g.mul(x, scale), shift),)


def _batch_norm_training(g: Graph, node: Any, args: List[Any], a, n) -> Value:
    """训练态 BN -> IR 的 ``batch_norm``（用当前 batch 的统计量，axis=1）。

    ``momentum`` 与 ``running_mean`` / ``running_var`` 不进图：IR 没有"图外
    状态"，BN 的 running stats 由训练循环用 ``batch_norm_stats`` 自己维护
    （5.26 的边界，5.208 把导入口打开）。这不是静默降级——训练态 BN 的输出
    只依赖 batch 统计量、``weight``、``bias``、``eps``，这四样都在图里；
    导出图里本来就没有 stats 更新那一步。

    导入后的图交给 ``Trainer`` 时，``bn_momentum`` 会扫描这些 ``batch_norm``
    节点、每步执行一张统计子图并按 EMA 维护 running mean/var
    （``train/bn.py``），训练完可以用 ``Trainer.bn_running_stats()`` 取出来
    喂给推理图（``models/cnn.py:bn_infer``）。也就是说"训练输出对齐"与
    "训练后 eval 状态对齐"现在是同一套闭环。

    ``torch.export`` 会在图里留下 ``add_(num_batches_tracked, 1)`` 这类对
    buffer 的原地写，由 :func:`import_exported` 统一跳过（0 使用者且目标
    是 buffer），不经过这里。
    """
    x, weight, bias = a(0), a(1), a(2)
    eps = float(n(7)) if len(args) > 7 and args[7] is not None else 1e-5
    if x.type.rank < 2:
        raise UnsupportedOperatorError(f"节点 {node.name} 的 BN 输入秩必须 >= 2")
    channels = x.type.shape[1]
    if not isinstance(channels, int):
        raise UnsupportedOperatorError(
            f"节点 {node.name} 的 BN 通道维含符号 {channels!r}，"
            "IR 的 batch_norm 只接受静态通道维"
        )
    for label, value in (("weight", weight), ("bias", bias)):
        if value.type.rank != 1 or value.type.shape[0] != channels:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 BN {label} 形状应为 ({channels},)，"
                f"得到 {value.type.shape}"
            )
    return g.add_node("batch_norm", [x, weight, bias], {"axis": 1, "eps": eps})


def _clamp(g: Graph, x: Value, lo: Optional[float], hi: Optional[float]) -> Value:
    """``clamp(x, lo, hi) = lo + relu(x-lo) - relu(x-hi)``。

    只有 ``relu`` 没有 ``min`` / ``max``，但这个恒等式是精确的（没有近似），
    且三段区间逐一可验证。``None`` 表示该侧不设限。
    """
    if lo is None and hi is None:
        return x
    dtype = x.type.dtype
    if lo is not None and hi is not None:
        low = _scalar(g, lo, dtype)
        return g.sub(g.add(low, g.relu(g.sub(x, low))), g.relu(g.sub(x, _scalar(g, hi, dtype))))
    if lo is not None:
        low = _scalar(g, lo, dtype)
        return g.add(low, g.relu(g.sub(x, low)))
    high = _scalar(g, hi, dtype)
    return g.sub(high, g.relu(g.sub(high, x)))


def _const_like(g: Graph, x: Value, value: Any, node: Any, what: str) -> Value:
    """把一个 Python 标量折成与 ``x`` 同 dtype 的常量节点。

    IR 的逐元素算子只按 numpy 广播规则组合张量，没有"标量操作数"这种
    第二种输入形态；标量必须先变成形状 ``()`` 的常量。整型张量遇到非整数
    标量时拒绝——``int(3.5)`` 的静默截断会把语义改掉。
    """
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise UnsupportedOperatorError(
            f"节点 {node.name} 的 {what}={value!r} 不是 Python 标量，无法在构图期折叠"
        )
    dtype = x.type.dtype
    if not dtype.is_float and float(value) != int(value):
        raise UnsupportedOperatorError(
            f"节点 {node.name} 的 {what}={value!r} 不是整数，但输入 dtype 是 {dtype.value}；"
            "整数张量上的小数标量没有无损表示"
        )
    return _scalar(g, float(value) if dtype.is_float else int(value), dtype)


def _pow_scalar(g: Graph, x: Value, exponent: Any, node: Any) -> Value:
    """``x ** exponent``（指数是 Python 标量）用已有原语精确展开。

    IR 没有通用 ``pow``：浮点指数只能通过 ``exp(y*log(x))`` 实现，而那条路
    在 ``x <= 0`` 时不是原式（定义域不同），不能当成"等价降级"。因此只接
    可精确展开的指数：0 / 0.5 / 整数 / -1，其余显式拒绝。
    """
    if isinstance(exponent, bool) or not isinstance(
        exponent, (int, float, np.integer, np.floating)
    ):
        raise UnsupportedOperatorError(
            f"节点 {node.name} 的 pow 指数不是 Python 标量（{exponent!r}）；"
            "运行期指数在 IR 里没有对应算子"
        )
    value = float(exponent)
    dtype = x.type.dtype
    one = _const_like(g, x, 1.0, node, "pow 的常量")

    def _int_pow(n: int) -> Value:
        # 平方求幂：层数与乘法次数都只取决于静态指数
        result: Optional[Value] = None
        base = x
        while n:
            if n & 1:
                result = base if result is None else g.mul(result, base)
            n >>= 1
            if n:
                base = g.mul(base, base)
        assert result is not None
        return result

    if value == 0.0:
        if not all(isinstance(d, int) for d in x.type.shape):
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 pow(0) 需要把标量 1 广播成静态形状"
            )
        return g.broadcast_to(one, shape=x.type.shape)
    if value == 0.5:
        return g.sqrt(x)
    if value == 1.0:
        return x
    if value == 2.0:
        return g.mul(x, x)
    if value == -1.0:
        return g.div(one, x)
    if value == int(value) and abs(int(value)) <= 16:
        n = int(value)
        if not dtype.is_float and n < 0:
            # 负整数指数在整数张量上是真分数，IR 没有这种语义
            raise UnsupportedOperatorError(
                f"节点 {node.name} 在 {dtype.value} 张量上使用了负指数 {n}"
            )
        return _int_pow(n) if n > 0 else g.div(one, _int_pow(-n))
    raise UnsupportedOperatorError(
        f"节点 {node.name} 的 pow 指数是 {exponent!r}，无法用已有算子精确展开；"
        "IR 没有通用 pow（exp/log 在非正输入上不等价）。请把该运算改写为 "
        "sqrt / 整数次幂，或在建模侧用 exp(y*log(x)) 并自行处理定义域"
    )


def _softplus(g: Graph, x: Value, beta: float, threshold: float, node: Any) -> Value:
    """``softplus(x) = (1/beta) * log(1 + exp(beta*x))`` 的稳定恒等式展开。

    用 ``max(z,0) + log(1 + exp(-|z|))``：指数项恒 ``<= 1``，不会溢出，
    且对任意 ``z`` 都是精确等式（不是近似）。torch 的 ``threshold`` 只是
    溢出保护：超过阈值时直接返回 ``x``，与精确值相差 ``exp(-threshold)``。
    非默认阈值会改变 torch 的输出，因此显式拒绝而不是忽略。
    """
    if beta <= 0:
        raise UnsupportedOperatorError(f"节点 {node.name} 的 softplus beta 必须为正，得到 {beta}")
    if threshold != 20.0:
        raise UnsupportedOperatorError(
            f"节点 {node.name} 的 softplus threshold={threshold} 不是默认值 20；"
            "该参数决定 torch 在哪一段直接返回 x，换值会改变输出"
        )
    dtype = x.type.dtype
    if not dtype.is_float:
        raise UnsupportedOperatorError(
            f"节点 {node.name} 的 softplus 需要浮点 dtype，得到 {dtype.value}"
        )
    z = x if beta == 1.0 else g.mul(x, _const_like(g, x, beta, node, "beta"))
    absz = g.add(g.relu(z), g.relu(g.neg(z)))
    body = g.log(g.add(_scalar(g, 1.0, dtype), g.exp(g.neg(absz))))
    out = g.add(g.relu(z), body)
    return out if beta == 1.0 else g.div(out, _const_like(g, x, beta, node, "beta"))


def _mean_dim(g: Graph, x: Value, dims, keepdim: bool) -> Value:
    """``mean`` 沿若干维 -> ``reduce_sum`` 再除以元素数（标量，可广播）。"""
    if not isinstance(dims, (list, tuple)):
        dims = [dims]
    rank = x.type.rank
    axes = tuple(_norm_dim(int(d), rank) for d in dims)
    count = 1
    for axis in axes:
        size = x.type.shape[axis]
        if not isinstance(size, int):
            raise UnsupportedOperatorError(
                f"mean 的归约维含符号 {size!r}，除数无法静态确定"
            )
        count *= size
    reduced = g.reduce_sum(x, axis=axes, keepdims=keepdim)
    return g.div(reduced, _scalar(g, float(count), x.type.dtype))


def _aten_arg(args: List[Any], kwargs: Dict[str, Any], index: int, key: str, default):
    """按 ATen schema 取一个参数：先看关键字，再看位置，最后才用默认值。

    kw-only 参数（``add`` 的 ``alpha``、SDPA 的 ``scale`` / ``enable_gqa``）在
    FX 节点里只会出现在 ``kwargs``，按位置下标去读永远读不到——会把用户显式
    传的值静默换成默认值。这里两端都看，避免依赖"export 一定按哪个顺序填"。
    """
    if key in kwargs and kwargs[key] is not None:
        return kwargs[key]
    if len(args) > index and args[index] is not None:
        return args[index]
    return default


def _reject_unknown_kwargs(node: Any, kwargs: Dict[str, Any], known) -> None:
    """出现了降级规则不认识的参数就报错，而不是当它不存在。

    静默忽略一个关键字，等于在"接受 import"的同时改变语义；这里的策略与
    未支持的算子一致：能解释的显式降级，不能解释的明确失败。
    """
    extra = sorted(set(kwargs) - set(known))
    if extra:
        raise UnsupportedOperatorError(
            f"节点 {node.name} 的 {node.target} 带有未处理的参数 {extra}；"
            "请在降级规则里消费它，或明确拒绝"
        )


def _translate(g: Graph, node: Any, env: Dict[str, Any]) -> Any:
    target_op = node.target
    target = str(target_op)
    args = list(node.args)
    kwargs = dict(node.kwargs)

    def a(i: int) -> Value:
        value = env[args[i].name]
        if value is None:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的第 {i} 个输入来自未支持的多输出槽位（None）"
            )
        if not isinstance(value, Value):
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的第 {i} 个参数是拆分结果，需要先 getitem"
            )
        return value

    def n(i: int):
        return args[i]

    def t(i: int, key: str):
        """张量参数：位置或关键字都接受，缺省返回 None。

        位置参数里的张量由 ``a`` 查；写成关键字的（合成 FX 图会这么写）要从
        ``kwargs`` 拿到节点名再查 ``env``——只按位置查会把它当成"没传"。
        """
        if key in kwargs and kwargs[key] is not None:
            raw = kwargs[key]
            value = env.get(getattr(raw, "name", ""))
            if not isinstance(value, Value):
                raise UnsupportedOperatorError(
                    f"节点 {node.name} 的 {key} 不是图内的值"
                )
            return value
        if len(args) > i and args[i] is not None:
            return a(i)
        return None

    if getattr(target_op, "__name__", "") == "getitem":
        container = env[args[0].name]
        if not isinstance(container, tuple):
            raise UnsupportedOperatorError(
                f"getitem 只支持拆分结果，节点 {node.name} 收到 {type(container).__name__}"
            )
        index = int(args[1])
        try:
            return container[index]
        except IndexError:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 取了多输出 {args[0].name} 的第 {index} 个槽位；"
                "本适配器只保留第 0 个输出（其余是 running stats 更新，不属于张量语义）"
            ) from None

    if target == "aten.split.Tensor":
        # chunk / split 都会落到这里：沿 dim 按 split_size 切块
        x = a(0)
        split_size = int(n(1))
        dim = _norm_dim(int(n(2)) if len(args) > 2 else 0, x.type.rank)
        total = x.type.shape[dim]
        if not isinstance(total, int):
            raise UnsupportedOperatorError(f"split 需要静态维度，收到 {total!r}")
        parts = []
        start = 0
        while start < total:
            stop = min(start + split_size, total)
            parts.append(g.slice(x, dim=dim, start=start, stop=stop))
            start = stop
        return tuple(parts)

    if target == "aten.chunk.default":
        # 新版 torch.export 会保留 chunk（旧版直接降解为 split.Tensor）。
        # torch 的切法：split_size = ceil(size / chunks)，因此实际块数可能少于 chunks。
        x = a(0)
        chunks = int(n(1))
        dim = _norm_dim(int(n(2)) if len(args) > 2 else 0, x.type.rank)
        total = x.type.shape[dim]
        if not isinstance(total, int):
            raise UnsupportedOperatorError(f"chunk 需要静态维度，收到 {total!r}")
        if chunks <= 0:
            raise UnsupportedOperatorError(f"chunk 的 chunks 必须为正，收到 {chunks}")
        if total == 0:
            raise UnsupportedOperatorError("chunk 不支持长度为 0 的维")
        step = -(-total // chunks)  # ceil
        parts: list = []
        start = 0
        while start < total:
            stop = min(start + step, total)
            parts.append(g.slice(x, dim=dim, start=start, stop=stop))
            start = stop
        # FX 节点固定输出 chunks 个位置，未产生的槽位保持 None（与真实语义一致）
        parts.extend([None] * (chunks - len(parts)))
        return tuple(parts)

    if target in (
        "aten.clone.default",
        "aten.alias.default",
        "aten.detach.default",
        # ``detach_`` / ``lift_fresh*`` 是 torch.export 给提升常量套的包装：
        # 输出与输入同值。IR 是 SSA、值不可变，这些包装在语义上就是恒等。
        "aten.detach_.default",
        "aten.lift_fresh.default",
        "aten.lift_fresh_copy.default",
    ):
        # IR 是 SSA、值不可变，clone/alias/detach 在语义上就是恒等
        return a(0)

    if target == "aten.index.Tensor":
        # ``table[ids]``（单个整数索引张量）正是 IR 的 embedding 语义。
        # 多索引 / 带 None 的通用高级索引没有对应的 IR 算子：只取第一个索引
        # 会把语义悄悄改掉，必须显式拒绝。
        raw = args[1] if len(args) > 1 else None
        if not isinstance(raw, (list, tuple)) or len(raw) != 1 or raw[0] is None:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 只支持单个整数索引张量（table[ids]）；"
                f"收到索引 {raw!r}，其余高级索引形态没有 IR 语义"
            )
        table = a(0)
        ids = env.get(getattr(raw[0], "name", ""))
        if not isinstance(ids, Value):
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的索引不是图内的值"
            )
        return g.add_node("embedding", [table, ids])

    if target == "aten.embedding.default":
        # (weight, indices, padding_idx=None, scale_grad_by_freq=False, sparse=False)
        padding_idx = _aten_arg(args, kwargs, 2, "padding_idx", None)
        if isinstance(padding_idx, Value):
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 padding_idx 是图内值；只支持静态缺省（None / -1）"
            )
        if padding_idx is not None and int(padding_idx) >= 0:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 带 padding_idx={padding_idx}：IR 的 embedding 是"
                "普通查表，没有“该行梯度恒为零”的语义；显式拒绝而不是算成普通查表"
            )
        if _aten_arg(args, kwargs, 3, "scale_grad_by_freq", False):
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 scale_grad_by_freq=True 没有 IR 语义"
            )
        if _aten_arg(args, kwargs, 4, "sparse", False):
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 sparse=True 没有 IR 语义"
            )
        return g.add_node("embedding", [a(0), a(1)])

    if target == "aten._native_batch_norm_legit_no_training.default":
        return _batch_norm_inference(g, node, args, a, n)

    if target == "aten._native_batch_norm_legit_functional.default":
        # ``ep.run_decompositions()`` 之后的形态：返回 (out, save_mean,
        # save_invstd)，我们只保留第 0 槽；后两者是 running stats 的中间量，
        # 不属于张量语义（取它们的 getitem 若真是死节点已在入口跳过）。
        training = bool(n(5)) if len(args) > 5 and args[5] is not None else False
        if training:
            return (_batch_norm_training(g, node, args, a, n),)
        return _batch_norm_inference(g, node, args, a, n, eps_index=7)

    if target == "aten.batch_norm.default":
        # 新版 torch.export 对 eval 态的 BatchNorm 直接保留 aten.batch_norm：
        # (input, weight, bias, running_mean, running_var, training, momentum,
        #  eps, cudnn_enabled)。推理态必须用冻结的 running stats，走下面的
        # 仿射展开；训练态用当前 batch 的统计量，与 IR 的 batch_norm 语义一致，
        # 直接映射（running stats 的维护不在张量语义里，见 5.26 / 5.208）。
        training = bool(n(5)) if len(args) > 5 and args[5] is not None else False
        if training:
            return _batch_norm_training(g, node, args, a, n)
        return _batch_norm_inference(g, node, args, a, n, eps_index=7)[0]

    if target == "aten.avg_pool2d.default":
        x = a(0)
        kernel = _int_pair(n(1), "avg_pool2d 的 kernel")
        stride = _int_pair(n(2), "avg_pool2d 的 stride") if len(args) > 2 else None
        padding = _int_pair(n(3), "avg_pool2d 的 padding") if len(args) > 3 else None
        ceil_mode = bool(n(4)) if len(args) > 4 and args[4] is not None else False
        count_include_pad = bool(n(5)) if len(args) > 5 and args[5] is not None else True
        divisor = n(6) if len(args) > 6 else None
        if kernel is None:
            raise UnsupportedOperatorError(f"节点 {node.name} 的 avg_pool2d 缺少 kernel")
        if divisor is not None:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 avg_pool2d 使用了 divisor_override，"
                "IR 的分母只有 kernel 面积与有效点数两种"
            )
        attrs: Dict[str, Any] = {"kernel": kernel, "padding": padding or (0, 0)}
        if stride is not None:
            attrs["stride"] = stride
        if ceil_mode:
            attrs["ceil_mode"] = True
        if not count_include_pad:
            attrs["count_include_pad"] = False
        return g.add_node("avgpool2d", [x], attrs)

    if target == "aten.cat.default":
        items = args[0]
        if not isinstance(items, (list, tuple)):
            raise UnsupportedOperatorError(f"节点 {node.name} 的 cat 输入不是序列")
        values: List[Value] = []
        for item in items:
            value = env.get(getattr(item, "name", None))
            if not isinstance(value, Value):
                raise UnsupportedOperatorError(
                    f"节点 {node.name} 的 cat 含非张量输入 {item!r}"
                )
            values.append(value)
        if not values:
            raise UnsupportedOperatorError(f"节点 {node.name} 的 cat 没有输入")
        if len(values) == 1:
            return values[0]
        axis = int(n(1)) if len(args) > 1 else 0
        return g.concat(*values, axis=axis)

    if target == "aten.dropout.default":
        train = bool(n(2)) if len(args) > 2 else False
        if train:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 dropout 处于训练态；导入要求确定性，"
                "随机掩码请用 dropout_mask 显式构图"
            )
        return a(0)

    # ------------------------------------------------------------------
    # 恒等包装与 dtype 转换
    # ------------------------------------------------------------------
    if target == "aten._assert_tensor_metadata.default":
        # torch.export 在 to()/layout 之后插入的元数据断言：没有张量语义。
        return a(0)

    if target == "aten.contiguous.default":
        # IR 的张量没有步长概念，连续性不是语义的一部分。
        return a(0)

    if target in ("aten.to.dtype", "aten._to_copy.default", "aten.type_as.default"):
        x = a(0)
        if target == "aten.type_as.default":
            target_dtype = a(1).type.dtype
        else:
            raw = kwargs.get("dtype")
            if raw is None and len(args) > 1:
                raw = args[1]
            if raw is None:
                # 纯 device / layout 拷贝：IR 没有设备与步长语义，值不变。
                return x
            target_dtype = _to_dtype(raw)
        return g.cast(x, dtype=target_dtype) if target_dtype != x.type.dtype else x

    if target == "aten.copy.default":
        # 非原地 copy：结果是 src 按 self 的 dtype 落盘。原地 `copy_` 不在此列
        # （它要求"写回旧值"，IR 没有图外状态）。
        src = a(1)
        dtype = a(0).type.dtype
        return g.cast(src, dtype=dtype) if src.type.dtype != dtype else src

    if target.startswith("aten.copy_."):
        raise UnsupportedOperatorError(
            f"节点 {node.name} 是原地 copy_；IR 的值不可变，也没有图外状态，"
            "无法表达「写回后旧读者仍看到旧值」的语义"
        )

    # ------------------------------------------------------------------
    # 一元数学族
    # ------------------------------------------------------------------
    if target in ("aten.tanh.default", "aten.exp.default", "aten.log.default",
                  "aten.neg.default", "aten.neg_.default", "aten.sqrt.default",
                  "aten.erf.default", "aten.sign.default", "aten.round.default",
                  "aten.floor.default", "aten.ceil.default"):
        return g.add_node(target.split(".")[1].rstrip("_"), [a(0)])

    if target in ("aten.abs.default", "aten.abs_.default"):
        # abs(x) = relu(x) + relu(-x)：两段互补，合起来精确等于 |x|
        x = a(0)
        return g.add(g.relu(x), g.relu(g.neg(x)))

    if target == "aten.rsqrt.default":
        x = a(0)
        return g.div(_scalar(g, 1.0, x.type.dtype), g.sqrt(x))

    if target == "aten.reciprocal.default":
        x = a(0)
        return g.div(_scalar(g, 1.0, x.type.dtype), x)

    if target == "aten.square.default":
        x = a(0)
        return g.mul(x, x)

    if target == "aten.log1p.default":
        # log(1+x)：IR 没有 log1p 的专门内核，展开是同一个数学函数。
        x = a(0)
        return g.log(g.add(x, _const_like(g, x, 1.0, node, "log1p 的常量")))

    if target == "aten.expm1.default":
        x = a(0)
        return g.sub(g.exp(x), _const_like(g, x, 1.0, node, "expm1 的常量"))

    # ------------------------------------------------------------------
    # 激活函数：用已有原语逐段展开，每一段都能独立验证
    # ------------------------------------------------------------------
    if target in ("aten.leaky_relu.default", "aten.leaky_relu_.default"):
        slope = float(_aten_arg(args, kwargs, 1, "negative_slope", 0.01))
        x = a(0)
        # leaky_relu(x) = relu(x) - slope*relu(-x)：负半轴要回退 slope*x，
        # 符号必须相减（relu(-x) 是正的 "距离"，不是 x 本身）。
        return g.sub(
            g.relu(x),
            g.mul(_const_like(g, x, slope, node, "negative_slope"), g.relu(g.neg(x))),
        )

    if target in ("aten.elu.default", "aten.elu_.default"):
        alpha = float(_aten_arg(args, kwargs, 1, "alpha", 1.0))
        scale = float(_aten_arg(args, kwargs, 2, "scale", 1.0))
        input_scale = float(_aten_arg(args, kwargs, 3, "input_scale", 1.0))
        x = a(0)
        z = x if input_scale == 1.0 else g.mul(
            x, _const_like(g, x, input_scale, node, "input_scale")
        )
        # elu(z) = relu(z) + alpha*(exp(-relu(-z)) - 1)：正段是 z，负段是
        # alpha*(exp(z)-1)，两段在 z=0 处都取 0。
        negative = g.relu(g.neg(z))
        body = g.add(
            g.relu(z),
            g.mul(
                _const_like(g, x, alpha, node, "alpha"),
                g.sub(g.exp(g.neg(negative)), _scalar(g, 1.0, x.type.dtype)),
            ),
        )
        return body if scale == 1.0 else g.mul(
            body, _const_like(g, x, scale, node, "scale")
        )

    if target in ("aten.hardsigmoid.default", "aten.hardsigmoid_.default"):
        x = a(0)
        shifted = g.add(x, _const_like(g, x, 3.0, node, "hardsigmoid 的常量"))
        return g.div(_clamp(g, shifted, 0.0, 6.0), _const_like(g, x, 6.0, node, "hardsigmoid 的常量"))

    if target in ("aten.hardswish.default", "aten.hardswish_.default"):
        x = a(0)
        shifted = g.add(x, _const_like(g, x, 3.0, node, "hardswish 的常量"))
        gate = g.div(_clamp(g, shifted, 0.0, 6.0), _const_like(g, x, 6.0, node, "hardswish 的常量"))
        return g.mul(x, gate)

    if target == "aten.softplus.default":
        beta = float(_aten_arg(args, kwargs, 1, "beta", 1.0))
        threshold = float(_aten_arg(args, kwargs, 2, "threshold", 20.0))
        return _softplus(g, a(0), beta, threshold, node)

    if target == "aten.log_softmax.int":
        dim = int(_aten_arg(args, kwargs, 1, "dim", -1))
        return g.log(g.softmax(a(0), axis=dim))

    # ------------------------------------------------------------------
    # 截断与标量二元运算
    # ------------------------------------------------------------------
    if target in ("aten.clamp.default", "aten.clamp_min.default",
                  "aten.clamp_max.default", "aten.relu6.default"):
        x = a(0)
        if target == "aten.relu6.default":
            return _clamp(g, x, 0.0, 6.0)
        if target == "aten.clamp_min.default":
            lo, hi = _aten_arg(args, kwargs, 1, "min", None), None
        elif target == "aten.clamp_max.default":
            lo, hi = None, _aten_arg(args, kwargs, 1, "max", None)
        else:
            lo = _aten_arg(args, kwargs, 1, "min", None)
            hi = _aten_arg(args, kwargs, 2, "max", None)
        for name, value in (("min", lo), ("max", hi)):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise UnsupportedOperatorError(
                    f"节点 {node.name} 的 clamp {name}={value!r} 不是常量；"
                    "IR 的 clamp 边界写在属性里，运行期边界需要先改成输入再另建算子"
                )
        return _clamp(
            g, x,
            None if lo is None else float(lo),
            None if hi is None else float(hi),
        )

    if target in ("aten.add.Scalar", "aten.sub.Scalar", "aten.mul.Scalar",
                  "aten.div.Scalar", "aten.rsub.Scalar"):
        x = a(0)
        other = _aten_arg(args, kwargs, 1, "other", None)
        if target == "aten.mul.Scalar":
            return g.mul(x, _const_like(g, x, other, node, "乘数"))
        if target == "aten.div.Scalar":
            return g.div(x, _const_like(g, x, other, node, "除数"))
        alpha = _aten_arg(args, kwargs, 2, "alpha", 1.0)
        scaled = _const_like(g, x, float(other) * float(alpha), node, "other*alpha")
        if target == "aten.rsub.Scalar":
            return g.sub(scaled, x)
        if target == "aten.sub.Scalar":
            return g.sub(x, scaled)
        return g.add(x, scaled)

    if target in ("aten.pow.Tensor_Scalar", "aten.pow.Scalar"):
        x = a(0)
        return _pow_scalar(g, x, _aten_arg(args, kwargs, 1, "exponent", None), node)

    if target == "aten.hardtanh.default":
        lo = n(1) if len(args) > 1 else -1.0
        hi = n(2) if len(args) > 2 else 1.0
        return _clamp(
            g,
            a(0),
            None if lo is None else float(lo),
            None if hi is None else float(hi),
        )

    # ------------------------------------------------------------------
    # 归约
    # ------------------------------------------------------------------
    if target == "aten.mean.default":
        x = a(0)
        _reject_unknown_kwargs(node, kwargs, {"dtype"})
        dtype_arg = _aten_arg(args, kwargs, 1, "dtype", None)
        if dtype_arg is not None:
            x = g.cast(x, dtype=_to_dtype(dtype_arg))
        axes = tuple(range(x.type.rank))
        if not axes:
            raise UnsupportedOperatorError(f"节点 {node.name} 对秩 0 张量做 mean")
        return g.reduce_mean(x, axis=axes, keepdims=False)

    if target == "aten.sum.default":
        x = a(0)
        _reject_unknown_kwargs(node, kwargs, {"dtype"})
        dtype_arg = _aten_arg(args, kwargs, 1, "dtype", None)
        if dtype_arg is not None:
            x = g.cast(x, dtype=_to_dtype(dtype_arg))
        axes = tuple(range(x.type.rank))
        if not axes:
            raise UnsupportedOperatorError(f"节点 {node.name} 对秩 0 张量做 sum")
        return g.reduce_sum(x, axis=axes, keepdims=False)

    if target == "aten.sum.dim_IntList":
        x = a(0)
        _reject_unknown_kwargs(node, kwargs, {"dim", "keepdim", "dtype"})
        dim = _aten_arg(args, kwargs, 1, "dim", None)
        keepdim = bool(_aten_arg(args, kwargs, 2, "keepdim", False))
        dtype_arg = _aten_arg(args, kwargs, 3, "dtype", None)
        if dtype_arg is not None:
            x = g.cast(x, dtype=_to_dtype(dtype_arg))
        if dim is None or (isinstance(dim, (list, tuple)) and not dim):
            axes = tuple(range(x.type.rank))
        else:
            axes = tuple(
                _norm_dim(int(d), x.type.rank)
                for d in (dim if isinstance(dim, (list, tuple)) else (dim,))
            )
        return g.reduce_sum(x, axis=axes, keepdims=keepdim)

    if target == "aten.amax.default":
        x = a(0)
        _reject_unknown_kwargs(node, kwargs, {"dim", "keepdim"})
        dim = _aten_arg(args, kwargs, 1, "dim", None)
        keepdim = bool(_aten_arg(args, kwargs, 2, "keepdim", False))
        if dim is None or (isinstance(dim, (list, tuple)) and not dim):
            axes = tuple(range(x.type.rank))
        else:
            axes = tuple(
                _norm_dim(int(d), x.type.rank)
                for d in (dim if isinstance(dim, (list, tuple)) else (dim,))
            )
        return g.reduce_max(x, axis=axes, keepdims=keepdim)

    if target == "aten.max.dim":
        # (values, indices)：两个槽位都用 IR 已有算子精确给出，而不是把 indices
        # 填 None——那样下游一取下标就变成"未支持的多输出槽位"。
        x = a(0)
        _reject_unknown_kwargs(node, kwargs, {"dim", "keepdim"})
        dim = _norm_dim(int(_aten_arg(args, kwargs, 1, "dim", 0)), x.type.rank)
        keepdim = bool(_aten_arg(args, kwargs, 2, "keepdim", False))
        values = g.reduce_max(x, axis=(dim,), keepdims=keepdim)
        indices = g.argmax(x, axis=dim, keepdims=keepdim)
        return (values, indices)

    if target == "aten.topk.default":
        # (values, indices) 拆成两个 IR 节点：IR 的节点是单输出，一个算子
        # 只返回一个张量；拆开之后两者都能被 DCE 单独剪掉（只取 values 的
        # 模型不会白算一遍下标）。
        x = a(0)
        _reject_unknown_kwargs(node, kwargs, {"k", "dim", "largest", "sorted"})
        k = _aten_arg(args, kwargs, 1, "k", None)
        if isinstance(k, bool) or not isinstance(k, int):
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 topk 需要静态整数 k，得到 {k!r}；"
                "动态 k 会让输出形状随输入变化，IR 的形状推断无法表达"
            )
        dim = _norm_dim(int(_aten_arg(args, kwargs, 2, "dim", -1)), x.type.rank)
        if not bool(_aten_arg(args, kwargs, 3, "largest", True)):
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 topk 只支持 largest=True；"
                "取最小的 k 个请先 neg、跑完 topk 再 neg 回来"
            )
        # sorted=False 只是允许"顺序不保证"，返回有序结果是合法子集，
        # 所以照常落到稳定降序的实现上。
        attrs = {"k": int(k), "axis": dim}
        return (
            g.topk_values(x, **attrs),
            g.topk_indices(x, **attrs),
        )

    if target == "aten.argmax.default":
        x = a(0)
        _reject_unknown_kwargs(node, kwargs, {"dim", "keepdim"})
        dim = _aten_arg(args, kwargs, 1, "dim", None)
        keepdim = bool(_aten_arg(args, kwargs, 2, "keepdim", False))
        if dim is None:
            # 全归约下标：IR 的 argmax 只接受单轴，先摊平到秩 1
            if not all(isinstance(d, int) for d in x.type.shape):
                raise UnsupportedOperatorError(
                    f"节点 {node.name} 的全局 argmax 需要静态形状，得到 {x.type.shape}"
                )
            flat = g.reshape(x, shape=(x.type.numel(),))
            return g.argmax(flat, axis=0, keepdims=False)
        return g.argmax(x, axis=_norm_dim(int(dim), x.type.rank), keepdims=keepdim)

    # ------------------------------------------------------------------
    # 掩码、前缀和与补齐
    # ------------------------------------------------------------------
    if target == "aten.tril.default":
        x = a(0)
        diagonal = int(_aten_arg(args, kwargs, 1, "diagonal", 0))
        return g.tril(x, diagonal=diagonal)

    if target == "aten.triu.default":
        # triu(x, k) = tril(x^T, -k)^T：转置把上下三角互换，不新增算子。
        x = a(0)
        diagonal = int(_aten_arg(args, kwargs, 1, "diagonal", 0))
        rank = x.type.rank
        if rank < 2:
            raise UnsupportedOperatorError(f"节点 {node.name} 的 triu 需要秩 >= 2")
        perm = list(range(rank))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        flipped = g.transpose(x, perm=tuple(perm))
        return g.transpose(g.tril(flipped, diagonal=-diagonal), perm=tuple(perm))

    if target == "aten.cumsum.default":
        x = a(0)
        _reject_unknown_kwargs(node, kwargs, {"dim", "dtype"})
        dim = _norm_dim(int(_aten_arg(args, kwargs, 1, "dim", 0)), x.type.rank)
        dtype_arg = _aten_arg(args, kwargs, 2, "dtype", None)
        if dtype_arg is not None:
            x = g.cast(x, dtype=_to_dtype(dtype_arg))
        return g.cumsum(x, axis=dim)

    if target == "aten.pad.default":
        x = a(0)
        _reject_unknown_kwargs(node, kwargs, {"pad", "mode", "value"})
        pad = _aten_arg(args, kwargs, 1, "pad", None)
        mode = str(_aten_arg(args, kwargs, 2, "mode", "constant"))
        value = _aten_arg(args, kwargs, 3, "value", None)
        if not isinstance(pad, (list, tuple)) or len(pad) % 2:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 pad 参数必须是成对的整数列表，得到 {pad!r}"
            )
        if mode != "constant":
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 pad mode={mode!r}；IR 只有零填充（constant）"
            )
        if value is not None and float(value) != 0.0:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 pad value={value}；IR 的 pad 只补零"
            )
        y = x
        for pair_index in range(len(pad) // 2):
            before, after = int(pad[2 * pair_index]), int(pad[2 * pair_index + 1])
            if before < 0 or after < 0:
                raise UnsupportedOperatorError(
                    f"节点 {node.name} 的负 pad（裁剪）={(before, after)}；"
                    "请改用 slice 显式表达裁剪"
                )
            if before == 0 and after == 0:
                continue
            dim = y.type.rank - 1 - pair_index
            if dim < 0:
                raise UnsupportedOperatorError(
                    f"节点 {node.name} 的 pad 指定了比输入秩更多的维度"
                )
            y = g.pad(y, dim=dim, before=before, after=after)
        return y

    # ------------------------------------------------------------------
    # 形状与拼接
    # ------------------------------------------------------------------
    if target == "aten.expand_as.default":
        x = a(0)
        other = a(1)
        if not all(isinstance(d, int) for d in other.type.shape):
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 expand_as 目标形状含符号维 {other.type.shape}"
            )
        return g.broadcast_to(x, shape=other.type.shape)

    if target == "aten.stack.default":
        items = args[0]
        if not isinstance(items, (list, tuple)) or not items:
            raise UnsupportedOperatorError(f"节点 {node.name} 的 stack 输入不是非空序列")
        values: List[Value] = []
        for item in items:
            value = env.get(getattr(item, "name", None))
            if not isinstance(value, Value):
                raise UnsupportedOperatorError(
                    f"节点 {node.name} 的 stack 含非张量输入 {item!r}"
                )
            values.append(value)
        dim = int(_aten_arg(args, kwargs, 1, "dim", 0))
        rank = values[0].type.rank + 1
        dim = _norm_dim(dim, rank)
        lifted = []
        for value in values:
            shape = list(value.type.shape)
            shape.insert(dim, 1)
            lifted.append(g.reshape(value, shape=tuple(shape)))
        return g.concat(*lifted, axis=dim)

    if target == "aten.split_with_sizes.default":
        x = a(0)
        _reject_unknown_kwargs(node, kwargs, {"split_sizes", "dim"})
        sizes = _aten_arg(args, kwargs, 1, "split_sizes", None)
        dim = _norm_dim(int(_aten_arg(args, kwargs, 2, "dim", 0)), x.type.rank)
        if not isinstance(sizes, (list, tuple)) or not sizes:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 split_with_sizes 需要非空的尺寸序列，得到 {sizes!r}"
            )
        total = x.type.shape[dim]
        if not isinstance(total, int):
            raise UnsupportedOperatorError(f"节点 {node.name} 的切分维含符号 {total!r}")
        if sum(int(s) for s in sizes) != total:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 split_with_sizes 尺寸之和 "
                f"{sum(int(s) for s in sizes)} 与第 {dim} 维长度 {total} 不符"
            )
        parts: list = []
        start = 0
        for size in sizes:
            stop = start + int(size)
            parts.append(g.slice(x, dim=dim, start=start, stop=stop))
            start = stop
        return tuple(parts)

    if target == "aten.narrow.default":
        x = a(0)
        dim = _norm_dim(int(n(1)), x.type.rank)
        start = int(n(2))
        length = int(n(3))
        total = x.type.shape[dim]
        if not isinstance(total, int):
            raise UnsupportedOperatorError(f"节点 {node.name} 的 narrow 需要静态维")
        if start < 0:
            start += total
        if start < 0 or length < 0 or start + length > total:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 narrow 区间 [{start}, {start + length}) "
                f"越界（该维长度 {total}）"
            )
        return g.slice(x, dim=dim, start=start, stop=start + length)

    if target == "aten.unbind.int":
        x = a(0)
        dim = _norm_dim(int(_aten_arg(args, kwargs, 1, "dim", 0)), x.type.rank)
        total = x.type.shape[dim]
        if not isinstance(total, int):
            raise UnsupportedOperatorError(f"节点 {node.name} 的 unbind 需要静态维")
        shape = list(x.type.shape)
        shape.pop(dim)
        parts = []
        for index in range(total):
            one = g.slice(x, dim=dim, start=index, stop=index + 1)
            parts.append(g.reshape(one, shape=tuple(shape)))
        return tuple(parts)

    if target == "aten.squeeze.default":
        x = a(0)
        shape = tuple(d for d in x.type.shape if d != 1)
        return g.reshape(x, shape=shape)

    if target == "aten.flatten.default":
        x = a(0)
        start = _norm_dim(int(_aten_arg(args, kwargs, 1, "start_dim", 0)), x.type.rank)
        end = _norm_dim(int(_aten_arg(args, kwargs, 2, "end_dim", -1)), x.type.rank)
        shape = list(x.type.shape)
        merged = 1
        for d in shape[start : end + 1]:
            if not isinstance(d, int):
                raise UnsupportedOperatorError(
                    f"节点 {node.name} 的 flatten 区间含符号维 {d!r}"
                )
            merged *= d
        return g.reshape(x, shape=tuple(shape[:start]) + (merged,) + tuple(shape[end + 1 :]))

    if target == "aten.t.default":
        x = a(0)
        if x.type.rank != 2:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 t() 要求秩 2，得到秩 {x.type.rank}"
            )
        return g.transpose(x, perm=(1, 0))

    if target == "aten.mm.default":
        return g.matmul(a(0), a(1))

    if target == "aten.mv.default":
        # IR 的 matmul 只接受秩 >= 2：把向量补成 (K,1)，乘完再去掉尾维。
        x, v = a(0), a(1)
        if x.type.rank != 2 or v.type.rank != 1:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 mv 要求 (M,K) 与 (K,)，"
                f"得到秩 {x.type.rank} 与 {v.type.rank}"
            )
        k = v.type.shape[0]
        column = g.reshape(v, shape=(k, 1))
        return g.reshape(g.matmul(x, column), shape=(x.type.shape[0],))

    if target == "aten.dot.default":
        x, y = a(0), a(1)
        if x.type.rank != 1 or y.type.rank != 1:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 dot 要求两个秩 1 张量，得到秩 {x.type.rank} 与 {y.type.rank}"
            )
        return g.reduce_sum(g.mul(x, y), axis=(0,), keepdims=False)

    if target == "aten.addmm.default":
        bias, x, y = a(0), a(1), a(2)
        beta = float(_aten_arg(args, kwargs, 3, "beta", 1.0))
        alpha = float(_aten_arg(args, kwargs, 4, "alpha", 1.0))
        product = g.matmul(x, y)
        if alpha != 1.0:
            product = g.mul(product, _const_like(g, product, alpha, node, "alpha"))
        scaled_bias = bias
        if beta != 1.0:
            scaled_bias = g.mul(bias, _const_like(g, bias, beta, node, "beta"))
        return g.add(scaled_bias, product)

    # ------------------------------------------------------------------
    # 常量构造：尺寸与取值都是静态的，落成 ``constant`` 节点（有元素上限）
    # ------------------------------------------------------------------
    if target in ("aten.zeros.default", "aten.ones.default", "aten.full.default"):
        size = n(0)
        if not isinstance(size, (list, tuple)):
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 size 不是静态序列：{size!r}"
            )
        shape = tuple(int(d) for d in size)
        dtype_arg = kwargs.get("dtype")
        dtype = _to_dtype(dtype_arg) if dtype_arg is not None else f32
        fill: Any = 0.0 if target == "aten.zeros.default" else 1.0
        if target == "aten.full.default":
            fill = _aten_arg(args, kwargs, 1, "fill_value", None)
            if hasattr(fill, "op") and hasattr(fill, "name"):
                raise UnsupportedOperatorError(
                    f"节点 {node.name} 的 fill_value 是图内值；IR 的常量填充要求静态取值"
                )
        return _constant_node(
            g,
            torch.full(shape, fill, dtype=getattr(torch, _IR_TO_TORCH[dtype.value])),
            f"节点 {node.name} 的常量",
        )

    if target in ("aten.zeros_like.default", "aten.ones_like.default",
                  "aten.full_like.default"):
        x = a(0)
        dtype_arg = kwargs.get("dtype")
        dtype = _to_dtype(dtype_arg) if dtype_arg is not None else x.type.dtype
        fill: Any = 0.0 if target == "aten.zeros_like.default" else 1.0
        if target == "aten.full_like.default":
            fill = _aten_arg(args, kwargs, 1, "fill_value", None)
            if hasattr(fill, "op") and hasattr(fill, "name"):
                raise UnsupportedOperatorError(
                    f"节点 {node.name} 的 fill_value 是图内值；IR 的常量填充要求静态取值"
                )
        if not all(isinstance(d, int) for d in x.type.shape):
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的形状含符号维 {x.type.shape}"
            )
        return _constant_node(
            g,
            torch.full(x.type.shape, fill, dtype=getattr(torch, _IR_TO_TORCH[dtype.value])),
            f"节点 {node.name} 的常量",
        )

    if target in ("aten.arange.default", "aten.arange.start", "aten.arange.start_step"):
        if target == "aten.arange.default":
            start, end, step = 0, n(0), 1
        elif target == "aten.arange.start":
            start, end, step = n(0), n(1), 1
        else:
            start, end, step = n(0), n(1), n(2)
        if any(hasattr(v, "name") for v in (start, end, step)):
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 arange 边界是图内值；IR 的常量构造要求静态取值"
            )
        if float(step) == 0.0:
            raise UnsupportedOperatorError(f"节点 {node.name} 的 arange step 为 0")
        dtype_arg = kwargs.get("dtype")
        if dtype_arg is None:
            dtype = f32 if any(isinstance(v, float) for v in (start, end, step)) else i32
        else:
            dtype = _to_dtype(dtype_arg)
        return _constant_node(
            g,
            torch.arange(
                start, end, step, dtype=getattr(torch, _IR_TO_TORCH[dtype.value])
            ),
            f"节点 {node.name} 的常量",
        )

    if target == "aten.scalar_tensor.default":
        value = n(0)
        if hasattr(value, "name"):
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 scalar_tensor 取值是图内值；IR 的常量构造要求静态取值"
            )
        dtype_arg = kwargs.get("dtype")
        dtype = _to_dtype(dtype_arg) if dtype_arg is not None else f32
        return _constant_node(
            g,
            torch.tensor(value, dtype=getattr(torch, _IR_TO_TORCH[dtype.value])),
            f"节点 {node.name} 的常量",
        )

    if target == "aten.mean.dim":
        dims = n(1)
        keepdim = bool(n(2)) if len(args) > 2 else False
        return _mean_dim(g, a(0), dims, keepdim)

    if target == "aten.sigmoid.default":
        return g.sigmoid(a(0))

    if target == "aten.gelu.default":
        approximate = kwargs.get("approximate")
        if approximate is None:
            approximate = n(1) if len(args) > 1 else "none"
        approximate = "none" if approximate is None else str(approximate)
        x = a(0)
        if approximate == "tanh":
            return g.gelu(x)
        if approximate != "none":
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 gelu approximate={approximate!r} 不是 tanh / none"
            )
        # 精确式 0.5x(1+erf(x/sqrt(2)))：erf 是 IR 的一元算子，与 tanh 近似
        # 是两条不同的定义，这里按导出属性走精确那条，不做静默替换。
        dtype = x.type.dtype
        half = _scalar(g, 0.5, dtype)
        one = _scalar(g, 1.0, dtype)
        inv_sqrt2 = _scalar(g, 1.0 / math.sqrt(2.0), dtype)
        inner = g.erf(g.mul(x, inv_sqrt2))
        return g.mul(g.mul(half, x), g.add(one, inner))

    if target == "aten.silu.default":
        return g.silu(a(0))

    if target in ("aten.relu.default", "aten.relu_.default"):
        # ``relu_`` 是 inplace 版本。export 出来的图是 SSA 的，后续读者拿到的是
        # 这次调用的返回值，原地写回对它们不可见，因此降级成同一个 ``relu`` 即可。
        return g.relu(a(0))

    def operand(i: int, ref: Value, what: str) -> Value:
        """取第 i 个操作数：图内值原样取，Python 标量折成与 ``ref`` 同 dtype 的常量。

        ``x + 1.5`` / ``x * 2`` 在 FX 图里是 ``aten.add.Tensor`` 的第二个
        参数为 float/int 字面量，而不是提升出来的常量张量——只按「参数一定是
        节点」读会直接 AttributeError。
        """
        raw = args[i] if len(args) > i else None
        if hasattr(raw, "name"):
            return a(i)
        return _const_like(g, ref, raw, node, what)

    def branch(raw, ref: Value, what: str) -> Value:
        """where 的某个分支：图内值原样取，Python 标量折成与另一分支同 dtype。

        与 ``operand`` 的差别是分支本身可能就是标量（``torch.where(c, x, 0.0)``），
        此时 dtype 参照物只能是另一个分支，而不是被选中的那个张量。
        """
        if hasattr(raw, "name"):
            return env[raw.name]
        return _const_like(g, ref, raw, node, what)

    compare_kind = {
        "aten.eq": "eq",
        "aten.ne": "ne",
        "aten.lt": "lt",
        "aten.le": "le",
        "aten.gt": "gt",
        "aten.ge": "ge",
    }.get(target.rsplit(".", 1)[0])

    if compare_kind is not None:
        _reject_unknown_kwargs(node, kwargs, set())
        x = a(0)
        y = operand(1, x, "比较的右操作数")
        if x.type.dtype is not y.type.dtype:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 {compare_kind} 两侧 dtype 不同"
                f"（{x.type.dtype.value} vs {y.type.dtype.value}）；"
                "IR 的比较要求同 dtype，请在模型里显式 cast"
            )
        return getattr(g, compare_kind)(x, y)

    # ------------------------------------------------------------------
    # 逻辑算子：IR 的 logical_* 只接受 bool（与比较算子的产出同一载体）。
    # ``&`` / ``|`` / ``~`` 在 FX 图里是 ``aten.__and__`` 等位运算重载，bool
    # 上的语义与 logical_* 相同；整数位运算 IR 没有对应算子，一并拒绝。
    # ------------------------------------------------------------------
    logical_kind = {
        "aten.logical_and": "logical_and",
        "aten.bitwise_and": "logical_and",
        "aten.__and__": "logical_and",
        "aten.logical_or": "logical_or",
        "aten.bitwise_or": "logical_or",
        "aten.__or__": "logical_or",
    }.get(target.rsplit(".", 1)[0])

    if logical_kind is not None:
        _reject_unknown_kwargs(node, kwargs, set())
        x = a(0)
        y = operand(1, x, "逻辑运算的右操作数")
        for value in (x, y):
            if value.type.dtype is not DType.BOOL:
                raise UnsupportedOperatorError(
                    f"节点 {node.name} 的 {target} 输入 dtype 是 "
                    f"{value.type.dtype.value}；IR 的逻辑 / 位运算只接受 bool"
                    "（浮点请先用 ne(x, 0) 落成 bool，整数位运算没有对应算子）"
                )
        return getattr(g, logical_kind)(x, y)

    if target.rsplit(".", 1)[0] in (
        "aten.logical_not",
        "aten.bitwise_not",
        "aten.__invert__",
    ):
        _reject_unknown_kwargs(node, kwargs, set())
        x = a(0)
        if x.type.dtype is not DType.BOOL:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 {target} 输入 dtype 是 "
                f"{x.type.dtype.value}；IR 的逻辑 / 位运算只接受 bool"
                "（浮点请先用 ne(x, 0) 落成 bool，整数位运算没有对应算子）"
            )
        return g.logical_not(x)

    if target.startswith("aten.where."):
        _reject_unknown_kwargs(node, kwargs, {"condition", "input", "other"})
        cond = t(0, "condition")
        if cond is None:
            raise UnsupportedOperatorError(f"节点 {node.name} 的 where 缺少 condition")
        if cond.type.dtype is not DType.BOOL:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 where 条件 dtype 是 {cond.type.dtype.value}；"
                "IR 的 where 只接受 bool 条件（浮点条件请先显式比较）"
            )
        def raw_branch(index: int, key: str):
            if len(args) > index:
                return args[index]
            return kwargs.get(key)

        raw_x = raw_branch(1, "input")
        raw_y = raw_branch(2, "other")
        if raw_x is None or raw_y is None:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 where 缺少 input / other 分支"
            )
        # 标量分支的 dtype 参照物只能取另一个（张量）分支。
        if hasattr(raw_x, "name"):
            x = env[raw_x.name]
            y = branch(raw_y, x, "where 的 other 分支")
        else:
            if not hasattr(raw_y, "name"):
                raise UnsupportedOperatorError(
                    f"节点 {node.name} 的 where 两个分支都是 Python 标量，"
                    "推不出 dtype；请把其中一个写成张量"
                )
            y = env[raw_y.name]
            x = _const_like(g, y, raw_x, node, "where 的 input 分支")
        return g.where(cond, x, y)

    if target in (
        "aten.masked_fill.Scalar",
        "aten.masked_fill_.Scalar",
        "aten.masked_fill.Tensor",
        "aten.masked_fill_.Tensor",
    ):
        # masked_fill(x, mask, v) 的语义是"mask 为真的位置填 v"，即
        # where(mask, v, x)。没有独立的 masked_fill 内核：掩码选择本来就是
        # where 的本义，多一个算子只会多一套需要校验的形状规则。
        x = a(0)
        mask = a(1)
        if mask.type.dtype is not DType.BOOL:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 masked_fill 掩码 dtype 是 "
                f"{mask.type.dtype.value}；IR 只接受 bool 掩码"
            )
        value = branch(args[2], x, "masked_fill 的填充值")
        return g.where(mask, value, x)

    if target in ("aten.add.Tensor", "aten.add_.Tensor"):
        # ``add_`` 同理：export 的图是 SSA 的，残差的 ``out += identity`` 只需按
        # ``add`` 求值，原地写回不会漏到别的读者那里。
        _reject_unknown_kwargs(node, kwargs, {"alpha"})
        x = a(0)
        y = operand(1, x, "加数")
        alpha = _aten_arg(args, kwargs, 2, "alpha", 1.0)
        return g.add(x, _mul_alpha(g, y, alpha, node))

    if target == "aten.sub.Tensor":
        _reject_unknown_kwargs(node, kwargs, {"alpha"})
        x = a(0)
        y = operand(1, x, "减数")
        alpha = _aten_arg(args, kwargs, 2, "alpha", 1.0)
        return g.sub(x, _mul_alpha(g, y, alpha, node))

    if target == "aten.mul.Tensor":
        x = a(0)
        return g.mul(x, operand(1, x, "乘数"))

    if target == "aten.div.Tensor":
        x = a(0)
        return g.div(x, operand(1, x, "除数"))

    if target in ("aten.matmul.default", "aten.bmm.default"):
        return g.matmul(a(0), a(1))

    if target == "aten.linear.default":
        x, w = a(0), a(1)
        bias = a(2) if len(args) > 2 and args[2] is not None else None
        y = g.matmul(x, g.transpose(w, perm=(1, 0)))
        return g.add(y, bias) if bias is not None else y

    if target in ("aten.conv2d.default", "aten.convolution.default"):
        return _conv2d(g, node, args, a, n)

    if target == "aten.conv_transpose2d.input":
        # 参数表与 convolution 不同：bias 之后是 stride/padding/output_padding，
        # groups 在 dilation 之前。
        return _conv_transpose2d(
            g, node, args, a,
            stride=n(3) if len(args) > 3 else None,
            padding=n(4) if len(args) > 4 else None,
            dilation=n(7) if len(args) > 7 else None,
            output_padding=(
                _int_pair(n(5), "output_padding")
                if len(args) > 5 and args[5] is not None else (0, 0)
            ),
            groups=int(n(6)) if len(args) > 6 else 1,
        )

    if target in ("aten.max_pool2d.default", "aten.max_pool2d_with_indices.default"):
        return _max_pool2d(g, node, args, a, n, with_indices="with_indices" in target)

    if target in ("aten.adaptive_avg_pool2d.default", "aten._adaptive_avg_pool2d.default"):
        return _adaptive_avg_pool2d(g, node, args, a, n, kwargs)

    if target == "aten.layer_norm.default":
        # eps 是第 5 个**位置**参数（kw-only 只有 None）；只读 kwargs 会让
        # ``layer_norm(x, shape, w, b, 0.1)`` 静默退回 1e-5。
        _reject_unknown_kwargs(
            node, kwargs, {"normalized_shape", "eps", "cudnn_enabled"}
        )
        x = a(0)
        shape_arg = _aten_arg(args, kwargs, 1, "normalized_shape", None)
        if shape_arg is None:
            raise UnsupportedOperatorError(f"节点 {node.name} 的 layer_norm 缺少 normalized_shape")
        normalized_shape = tuple(int(d) for d in shape_arg)
        weight = a(2) if len(args) > 2 and args[2] is not None else None
        bias = a(3) if len(args) > 3 and args[3] is not None else None
        eps = float(_aten_arg(args, kwargs, 4, "eps", 1e-5))
        if weight is None or bias is None:
            raise UnsupportedOperatorError("layer_norm 缺少 weight/bias 的降级规则")
        if len(normalized_shape) != 1:
            raise UnsupportedOperatorError(
                f"暂只支持最后一维的 layer_norm，收到 normalized_shape={normalized_shape}"
            )
        return g.layernorm(x, weight, bias, axes=(-1,), eps=eps)

    if target in ("aten.softmax.int", "aten._softmax.default"):
        dim = int(n(1))
        return g.softmax(a(0), axis=dim)

    if target in ("aten.view.default", "aten.reshape.default", "aten._unsafe_view.default"):
        shape = tuple(int(d) for d in n(1))
        return g.reshape(a(0), shape=shape)

    if target == "aten.flatten.using_ints":
        # CNN 的展平：把 [start, end] 区间合并成一维。IR 的 reshape 需要静态形状，
        # 所以符号维落在这个区间里时明确拒绝，而不是猜一个。
        x = a(0)
        rank = x.type.rank
        start = _norm_dim(int(n(1)) if len(args) > 1 else 0, rank)
        end = _norm_dim(int(n(2)) if len(args) > 2 else -1, rank)
        shape = list(x.type.shape)
        merged = 1
        for d in shape[start : end + 1]:
            if not isinstance(d, int):
                raise UnsupportedOperatorError(
                    f"节点 {node.name} 的 flatten 区间含符号维 {d!r}，IR 的 reshape 需要静态形状"
                )
            merged *= d
        return g.reshape(x, shape=tuple(shape[:start]) + (merged,) + tuple(shape[end + 1 :]))

    if target == "aten.expand.default":
        x = a(0)
        target_shape = [int(d) for d in n(1)]
        src = list(x.type.shape)
        if len(target_shape) < len(src):
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 expand 目标秩 {len(target_shape)} 小于输入秩 {len(src)}"
            )
        out: List[int] = []
        for i, dim in enumerate(target_shape):
            origin = i - (len(target_shape) - len(src))
            if dim == -1:
                if origin < 0:
                    raise UnsupportedOperatorError(
                        f"节点 {node.name} 的 expand 在新增维上使用 -1"
                    )
                size = src[origin]
                if not isinstance(size, int):
                    raise UnsupportedOperatorError(
                        f"节点 {node.name} 的 expand 对应维含符号 {size!r}"
                    )
                out.append(size)
            else:
                out.append(dim)
        return g.broadcast_to(x, shape=tuple(out))

    if target == "aten.unsqueeze.default":
        x = a(0)
        rank = x.type.rank + 1
        dim = _norm_dim(int(n(1)), rank)
        shape = list(x.type.shape)
        shape.insert(dim, 1)
        return g.reshape(x, shape=tuple(shape))

    if target == "aten.squeeze.dim":
        x = a(0)
        dim = _norm_dim(int(n(1)), x.type.rank)
        shape = list(x.type.shape)
        if shape[dim] == 1:
            shape.pop(dim)
        return g.reshape(x, shape=tuple(shape))

    if target == "aten.select.int":
        x = a(0)
        dim = _norm_dim(int(n(1)), x.type.rank)
        size = x.type.shape[dim]
        if not isinstance(size, int):
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 select 需要静态维，第 {dim} 维为 {size!r}"
            )
        index = int(n(2))
        if index < 0:
            index += size
        if not 0 <= index < size:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 select 下标 {index} 越界（该维长度 {size}）"
            )
        one = g.slice(x, dim=dim, start=index, stop=index + 1)
        shape = list(x.type.shape)
        shape.pop(dim)
        return g.reshape(one, shape=tuple(shape))

    if target == "aten.slice.Tensor":
        x = a(0)
        dim = _norm_dim(int(n(1)), x.type.rank)
        size = x.type.shape[dim]
        if not isinstance(size, int):
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 slice 需要静态维，第 {dim} 维为 {size!r}"
            )
        step = int(n(4)) if len(args) > 4 and args[4] is not None else 1
        if step != 1:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 slice step={step}，IR 的 slice 只支持 step=1"
            )
        start = int(n(2)) if len(args) > 2 and args[2] is not None else 0
        if start < 0:
            start += size
        start = max(0, min(start, size))
        end = n(3) if len(args) > 3 and args[3] is not None else size
        end = int(end)
        if end < 0:
            end += size
        end = max(0, min(end, size))
        if end <= start:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 slice 区间为空 [{start}, {end})，IR 不支持零长维"
            )
        return g.slice(x, dim=dim, start=start, stop=end)

    if target == "aten.scaled_dot_product_attention.default":
        # ``scale`` 与 ``enable_gqa`` 是 kw-only（schema 里在 ``*`` 之后），
        # 按 ``args[6]`` / ``args[7]`` 读永远读不到：用户显式传的 scale 会被
        # 静默换成 1/sqrt(D)，``enable_gqa=True`` 的拒绝分支也永远不会触发。
        _reject_unknown_kwargs(
            node,
            kwargs,
            {"attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa"},
        )
        q, k, v = a(0), a(1), a(2)
        mask = t(3, "attn_mask")
        dropout = float(_aten_arg(args, kwargs, 4, "dropout_p", 0.0))
        is_causal = bool(_aten_arg(args, kwargs, 5, "is_causal", False))
        scale = _aten_arg(args, kwargs, 6, "scale", None)
        gqa = bool(_aten_arg(args, kwargs, 7, "enable_gqa", False))
        if gqa:
            raise UnsupportedOperatorError(f"节点 {node.name} 使用了 enable_gqa，IR 暂不支持")
        if dropout != 0.0:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 attention dropout={dropout}，导入要求确定性"
            )
        if is_causal:
            raise UnsupportedOperatorError(
                f"节点 {node.name} 使用 is_causal=True；IR 需要显式 mask，"
                "请在建模侧物化 causal mask 后再导出"
            )
        if scale is None:
            head_dim = q.type.shape[-1]
            if not isinstance(head_dim, int):
                raise UnsupportedOperatorError(
                    f"节点 {node.name} 的 attention 头维含符号 {head_dim!r}"
                )
            scale = 1.0 / math.sqrt(head_dim)
        elif not isinstance(scale, (int, float, np.integer, np.floating)):
            raise UnsupportedOperatorError(
                f"节点 {node.name} 的 attention scale={scale!r} 不是 Python 标量；"
                "运行期标量需要先变成 IR 输入"
            )
        inputs = [q, k, v] + ([mask] if mask is not None else [])
        return g.add_node(
            "attention", inputs, {"scale": float(scale), "dropout": 0.0, "seed": 0}
        )

    if target == "aten.permute.default":
        perm = tuple(int(d) for d in n(1))
        return g.transpose(a(0), perm=perm)

    if target == "aten.transpose.int":
        x = a(0)
        d0, d1 = int(n(1)), int(n(2))
        rank = x.type.rank
        d0, d1 = _norm_dim(d0, rank), _norm_dim(d1, rank)
        perm = list(range(rank))
        perm[d0], perm[d1] = perm[d1], perm[d0]
        return g.transpose(x, perm=tuple(perm))

    raise UnsupportedOperatorError(
        f"未支持的 aten 算子: {target}（节点 {node.name}）。"
        "请在这里补降级规则，而不是让上层静默跳过。"
    )
