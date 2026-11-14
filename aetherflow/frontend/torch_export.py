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
from ..ir.types import DType, TensorType, bf16, f16, f32, f64, i32
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
}

#: 导入时允许落成 ``constant`` 节点的最大元素数。与常量折叠的上限同量级：
#: 再大的"常量"往 attrs 里塞，序列化体积和反序列化开销都会失控，应该改成
#: 参数或显式输入。
MAX_IMPORTED_CONSTANT_ELEMENTS = 65536


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


def _to_dtype(dtype: Any) -> DType:
    key = str(dtype).replace("torch.", "")
    if key not in _TORCH_DTYPE:
        supported = "、".join(sorted(_TORCH_DTYPE))
        raise UnsupportedOperatorError(
            f"不支持的 dtype: {dtype}；矩阵内的 dtype 是 {supported}"
            "（bool 没有对应的张量语义；int64 会按 i32 承载，越界值在入口拒绝）"
        )
    return _TORCH_DTYPE[key]


def _constant_node(g: Graph, value: Any, source: str) -> Value:
    """把 torch 常量张量落成 IR 的 ``constant`` 节点。

    常量写进 attrs（``data``），所以图仍然自包含、可序列化。超过上限就显式
    拒绝：再大的值应该改成参数或显式输入，而不是塞进图属性里。
    """
    if not hasattr(value, "detach"):
        raise UnsupportedOperatorError(
            f"{source} 不是常量张量（{type(value).__name__}），无法落成 constant 节点"
        )
    tensor = value.detach().cpu()
    dtype = _to_dtype(tensor.dtype)
    # numpy 没有 bfloat16：先转 float32 取位；执行器在节点边界按 bf16 量化。
    if str(tensor.dtype) == "torch.bfloat16":
        tensor = tensor.float()
    arr = tensor.numpy()
    if arr.size > MAX_IMPORTED_CONSTANT_ELEMENTS:
        raise UnsupportedOperatorError(
            f"{source} 有 {arr.size} 个元素，超过导入上限 "
            f"{MAX_IMPORTED_CONSTANT_ELEMENTS}；请改成显式输入或参数"
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
    _require_torch()

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
                    g, lifted_constants[key], f"常量 {key!r}"
                )
                continue
            val = node.meta["val"]
            t = TensorType(_to_dtype(val.dtype), _shape_of(val))
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
            env[node.name] = _constant_node(g, raw, f"属性 {node.target!r}")
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

    if target == "aten.hardtanh.default":
        lo = n(1) if len(args) > 1 else -1.0
        hi = n(2) if len(args) > 2 else 1.0
        return _clamp(
            g,
            a(0),
            None if lo is None else float(lo),
            None if hi is None else float(hi),
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

    if target in ("aten.add.Tensor", "aten.add_.Tensor"):
        # ``add_`` 同理：export 的图是 SSA 的，残差的 ``out += identity`` 只需按
        # ``add`` 求值，原地写回不会漏到别的读者那里。
        _reject_unknown_kwargs(node, kwargs, {"alpha"})
        x, y = a(0), a(1)
        alpha = _aten_arg(args, kwargs, 2, "alpha", 1.0)
        return g.add(x, _mul_alpha(g, y, alpha, node))

    if target == "aten.sub.Tensor":
        _reject_unknown_kwargs(node, kwargs, {"alpha"})
        x, y = a(0), a(1)
        alpha = _aten_arg(args, kwargs, 2, "alpha", 1.0)
        return g.sub(x, _mul_alpha(g, y, alpha, node))

    if target == "aten.mul.Tensor":
        return g.mul(a(0), a(1))

    if target == "aten.div.Tensor":
        return g.div(a(0), a(1))

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
