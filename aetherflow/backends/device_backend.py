"""设备无关的后端基类：执行计划、算子派发与调优面。

CUDA 与 HIP 以前是"HIP 继承 ``CudaBackend``"的关系：HIP 复用了近三千行
执行计划，代价是这些代码里的每一个 CUDA 假设都成了 HIP 的隐含前提。实际
已经暴露过的三处是 ``dispose`` 按 ``cupy.ndarray`` 判类型、``transpose``
读视图的 ``.base``、``matmul`` 直接取 ``cp.cuda.get_current_stream()``；
HIP 之所以能跑，是因为 ``hip/runtime.py`` 额外假装出了一个 ``cuda`` 命名
空间。这是审计 AF-32 第四点指出的"共有逻辑少了代码，也把假设扩散了"。

这里收的是两者**真正共同**的部分：分块与网格、融合分组、调优旋钮、图执行
计划缓存、全部 ``_op_*`` 派发。它们只依赖容器协议里的一小组方法。厂商差异
收成钩子，由子类各自实现：

- ``cp``：容器模块（cupy 或 ``hip.runtime``），需提供 ``ndarray`` / ``empty``
  / ``ones`` / ``zeros`` / ``asarray`` / ``asnumpy`` / ``reshape`` /
  ``transpose`` / ``ascontiguousarray``；
- ``codegen``：内核源码方言（NVRTC 或 hipRTC）；
- ``sm_count``：构造前由子类查好；
- ``_kernel`` / ``_cublas`` / ``current_stream_ptr`` / ``new_stream`` /
  ``allocator`` / ``capture``：编译、库句柄、流、分配器与图捕获；
- ``_OPS`` / ``_F16_OPS`` / ``CONV_IMPLS``：能力面与实现选择，子类覆盖。

**没搬过来的**：cuDNN 卷积、cupy 的 pinned 缓冲、内存池统计。它们只对一家
成立，留在各自文件里比在基类里加 ``if`` 更清楚。
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple
from weakref import ref

import numpy as np

from ..ir.graph import Graph, Node
from ..ir.types import DType
from ..lower.elementwise import CHAIN_OPS, ElemInput, ElemStep, build_chain
from ..lower.kernels import BINARY_OP
from ..lower.plan import (
    BLOCK_DEFAULT,
    BLOCK_OPTIONS,
    CONV_CB_MAX,
    CONV_ROC_MAX,
    CONV_SHARED_LIMIT,
    CONV_TILE_OPTIONS,
    CONV_TILED_MAX_K,
    SHARED_LIMIT,
    VEC4_ALIGN,
    attention_block,
    auto_block,
    check_block,
    cdiv,
    conv_oc_tile,
    conv_implicit_fits,
    conv_tiled_preferred,
    grid_for,
    plan_conv_implicit,
    plan_conv_tiled,
    plan_csr_scan,
    plan_csr_scan_enabled,
    plan_elementwise,
    plan_reduce,
    plan_reduce_tiled,
    REDUCE_TILE_X,
    scan_chunk,
)
from ..runtime.executor import Executor
from ..tuning.space import TuningSpace
from .base import BackendCapabilityError
from .contracts import MemoryStats
from .registry import KernelRegistry

# 同一批自研内核模板（NVRTC / hipRTC 各编译一份）承接的算子面。它是声明，
# 与当前接入了哪家硬件无关；各后端在自己的文件里把它取成自己的名字。
# 存在的唯一理由：**不建任何设备上下文也能读到**。
GPU_OPS: FrozenSet[str] = frozenset(
{
    "add",
    "sub",
    "mul",
    "div",
    "relu",
    "silu",
    "exp",
    "log",
    "sqrt",
    "neg",
    "tanh",
    "sigmoid",
    "erf",
    "step",
    "spike",
    "constant",
    "broadcast_to",
    "matmul",
    "quantize_i8",
    "dequantize_i8",
    "linear_i8",
    "conv2d",
    "maxpool2d",
    "maxpool2d_indices",
    "softmax",
    "attention",
    "layernorm",
    "reduce_sum",
    "reduce_max",
    "reduce_mean",
    "argmax",
    "reshape",
    "transpose",
    "cast",
    "slice",
    "gelu",
    "rmsnorm",
    "avgpool2d",
    "avgpool2d_backward",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool2d_backward",
    "concat",
    "embedding",
    "embedding_backward",
    "pad",
    "maxpool2d_backward",
    "conv2d_input_grad",
    "conv2d_weight_grad",
    "dropout_mask",
    "batch_norm",
    "batch_norm_stats",
    "scan_linear",
    "scan_linear_backward",
    "cumsum",
    "tril",
    "clamp",
    "mod",
    "sparse_linear",
    "sparse_linear_weight_grad",
    "sparse_linear_values_grad",
    "spike_linear",
    "spike_linear_weight_grad",
    "spike_offsets",
    "spike_csr",
}
)

# 这些算子的某个参数是**整数输入**（下标或随机 key），不是浮点激活值。dtype
# 的通用检查必须给它们开口子，否则 embedding 会在进入内核之前就被"只支持浮点"
# 挡下来，dropout_mask 的运行期 key 也一样。
_INDEX_ARGS: Dict[str, FrozenSet[int]] = {
    "embedding": frozenset({1}),
    "embedding_backward": frozenset({1}),
    "dropout_mask": frozenset({1}),
    "sparse_linear": frozenset({1, 3}),
    "sparse_linear_weight_grad": frozenset({1, 3}),
    "sparse_linear_values_grad": frozenset({2, 3}),
    "spike_linear": frozenset({2, 3}),
    "spike_linear_weight_grad": frozenset({2, 3}),
    "spike_csr": frozenset({1}),
    # 量化：i8 是**数据**而不是下标，但它同样不能走"只支持浮点"的默认检查。
    "dequantize_i8": frozenset({0}),
    "linear_i8": frozenset({1}),
    # 整数取模：两侧都是**整数数据**，不是下标。
    "mod": frozenset({0, 1}),
}

# 整数下标允许的 dtype。与 ir/ops.py 的 infer 保持一致：i32 是前端导入的
# 常态，i8 是压缩标签 / 量化权重的形态。
_INDEX_DTYPES = ("int32", "int8")

# 带 ``layout`` 属性的算子。这些算子的自研内核只按 NCHW 寻址；NHWC 形态
# 必须显式拒绝（DESIGN 5.246：布局传播由参考后端承接，设备端等待原生内核）。
_LAYOUT_OPS: FrozenSet[str] = frozenset({"conv2d", "maxpool2d", "avgpool2d"})

# 量化只有两个内核：读权重、写权重。两份源码同时喂给 NVRTC 与 hipRTC——
# "半偶舍入 + 端点饱和"是语义的一部分，不能让两个后端各写一次近似。
# ``per_row`` 为 1 时按 ``s[i % cols]`` 取与输出通道对应的 scale，避免把
# 逐通道 scale 物化成与权重同大的 f32。
_QUANTIZE_I8_SRC = r"""
extern "C" __global__ void af_quantize_i8(
    const float* __restrict__ w,
    const float* __restrict__ s,
    signed char* __restrict__ out,
    long long n, long long cols, long long per_row) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float sv = per_row ? s[i % cols] : s[i];
  float q = rintf(w[i] / sv);
  q = q > 127.0f ? 127.0f : (q < -127.0f ? -127.0f : q);
  out[i] = (signed char)q;
}
"""

_DEQUANTIZE_I8_SRC = r"""
extern "C" __global__ void af_dequantize_i8(
    const signed char* __restrict__ q,
    const float* __restrict__ s,
    float* __restrict__ out,
    long long n, long long cols, long long per_row) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float sv = per_row ? s[i % cols] : s[i];
  out[i] = (float)q[i] * sv;
}
"""


def _chainable(node: Node) -> bool:
    """这个节点能不能作为逐元素链的一环（不看它与邻居的关系）。"""
    name = node.op.name
    if name not in CHAIN_OPS:
        return False
    arity = 2 if name in BINARY_OP else 1
    if len(node.inputs) != arity:
        return False
    out_type = node.output.type
    if out_type.dtype not in (DType.F32, DType.F16):
        return False
    for value in node.inputs:
        t = value.type
        if t.dtype != out_type.dtype:
            return False
        if t.shape == out_type.shape:
            continue
        # 只有"恰好一个元素"的一侧可以当标量；符号维无法静态确认大小
        if t.is_static:
            size = 1
            for d in t.shape:
                size *= d
            if size == 1:
                continue
        return False
    return True


def _group_elementwise(graph: Graph) -> List[tuple]:
    """把相邻的同形逐元素算子合并成一组（纯静态计划，不碰设备）。

    判据全部是**静态**的，任何一条不满足就退回"一算子一内核"：

    - 算子必须有一元/二元表达式表（``CHAIN_OPS``），且参数个数与表一致
      （``silu`` 在门控形态下是二元的，语义是 ``silu(x) * gate``，不能当成
      一元链的一步）；
    - 相邻两节点之间那条边只被链内下一个节点使用，且不是图输出——否则
      中间值必须物化，融合就改变了可观测行为；
    - 形状与 dtype 在 IR 上相同（符号维按名字比较）；大小为 1 的一侧可以
      当标量，其余广播形态不融。

    长度 1 的组不合并：单算子仍走原来的专用内核，源码与数值行为逐字不变。
    运行期还会再查一次（是否连续、指针对齐、长度能否被 4 整除），见
    ``DeviceBackend._try_chain``；静态计划乐观不会算错，最多白算一次。
    """
    nodes = graph.nodes
    users: Dict[int, List[int]] = {}
    for idx, node in enumerate(nodes):
        for a in node.inputs:
            users.setdefault(a.id, []).append(idx)
    out_ids = {v.id for v in graph.outputs}

    groups: List[tuple] = []
    i = 0
    while i < len(nodes):
        j = i
        while (
            j + 1 < len(nodes)
            and _chainable(nodes[j])
            and _chainable(nodes[j + 1])
        ):
            mid = nodes[j].output
            if mid.id in out_ids or users.get(mid.id) != [j + 1]:
                break
            j += 1
        groups.append(tuple(nodes[i : j + 1]))
        i = j + 1
    return groups


def _pair(value) -> tuple:
    """IR 属性里的 int 或二元组（JSON 往返后是 list）统一成 (h, w)。"""
    if isinstance(value, int):
        return (value, value)
    return (int(value[0]), int(value[1]))


def _pool_out_dim(size: int, kernel: int, stride: int, padding: int,
                  ceil_mode: bool) -> int:
    """池化单维输出长度，与 ``ir/ops.py`` 的 ``_pool_out_dim`` 逐条一致。

    ``ceil_mode`` 的两步（向上取整 + 去掉起点越界的多余窗口）必须和 IR 层同步：
    这里算出的 ``HO`` / ``WO`` 决定内核的网格，差一格就是越界写或漏算。
    """
    if ceil_mode:
        out = -((int(size) + 2 * padding - kernel) // -stride) + 1
        if (out - 1) * stride >= int(size) + padding:
            out -= 1
    else:
        out = (int(size) + 2 * padding - kernel) // stride + 1
    return out


def _pool_flags(attrs: dict, op: str) -> tuple:
    """池化的布尔属性 ``(ceil_mode, count_include_pad)``；类型不对就显式报错。"""
    ceil_mode = attrs.get("ceil_mode", False)
    if not isinstance(ceil_mode, bool):
        raise BackendCapabilityError(f"{op} 的 ceil_mode 必须是 bool，收到 {ceil_mode!r}")
    count_include_pad = attrs.get("count_include_pad", True)
    if not isinstance(count_include_pad, bool):
        raise BackendCapabilityError(
            f"{op} 的 count_include_pad 必须是 bool，收到 {count_include_pad!r}"
        )
    return ceil_mode, count_include_pad


def _mask_shared_over_heads(mask_shape, s_len: int, t_len: int) -> bool:
    """``mask`` 能否被所有 ``(b, h)`` 共用一份 ``(S, T)``。

    条件是最后两维正好等于 ``(S, T)``、其余维全为 1——因果 mask 的
    ``(1,1,S,T)`` 与直接的 ``(S,T)`` 都落在这里。任何随 batch 或 head 变化的
    mask 仍要广播，不能共用。
    """
    shape = tuple(int(dim) for dim in mask_shape)
    return shape[-2:] == (int(s_len), int(t_len)) and all(
        dim == 1 for dim in shape[:-2]
    )


# conv2d 分块核在 W 方向的线程数、shared 预算与分块阈值都在 ``lower/plan.py``：
# 它们是与设备无关的纯算术，放在后端里既挡住了离线单测，也容易和 HIP 侧分叉。

_CONV_IMPLS = ("auto", "im2col", "tiled", "implicit")

# ``sparse_linear`` 的实现选择。基类只给 "custom" 与 "auto"——"cusparse"
# 属于 NVIDIA 的库，由 ``CudaBackend`` 扩展；HIP 侧构造期就会拒绝它，
# 而不是等到第一次稀疏乘才发现。默认 "custom" 而不是 "auto"：自研核带
# 下标越界检查，库路径没有；默认值不能悄悄换掉一道检查（见 5.203）。
_SPARSE_IMPLS = ("custom", "auto")

# 逐元素类：每个线程只处理一个元素，block 大小只影响占用率与合并访存，
# 不改变 shared memory 需求，也不改变数值语义。
_ELEMENTWISE_TUNABLE_OPS: FrozenSet[str] = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "relu",
        "silu",
        "exp",
        "log",
        "sqrt",
        "neg",
        "tanh",
        "sigmoid",
        "erf",
        "gelu",
        "step",
        "spike",
        "dropout_mask",
    }
)

# 归约类：一个 block 负责一行（或一个通道），块内归约的暂存区是固定的
# ``REDUCE_SCRATCH``（1024 个 float = 4 KB），**不随 block 变化**；内核沿
# ``blockDim.x`` 跨步读输入，所以 block 只改变归约树深度与占用率。改 block
# 因此既不改变 shared 预算，也不改变数值语义（块内求和顺序会变，但归约本身
# 就是浮点非结合的操作，逐位一致只在同一 block 下成立）。
#
# 此前这批被排除在搜索之外，理由是"shared 与 blockDim 绑定"；5.160 核对内核
# 后确认该前提不成立（``Shared("float", "smem", REDUCE_SCRATCH)`` 是常量），
# 于是开放给调优面。``batch_norm`` 的两个内核共用同一套骨架，一起放开。
_REDUCE_TUNABLE_OPS: FrozenSet[str] = frozenset(
    {
        "softmax",
        "layernorm",
        "rmsnorm",
        "reduce_sum",
        "reduce_max",
        "reduce_mean",
        "argmax",
        "batch_norm",
        "batch_norm_stats",
    }
)

_TUNABLE_OPS: FrozenSet[str] = _ELEMENTWISE_TUNABLE_OPS | _REDUCE_TUNABLE_OPS

# conv2d 的旋钮不是 block，而是分块核的线程铺法，见 ``_CONV_KNOBS``。
_CONV_TILE_OP = "conv2d"

# 分块核的另两个旋钮的候选值：2 的幂，到计划层的上限为止（5.16 把上限本身
# 定成了最优值，所以候选没有理由越过它）。
_CONV_ROC_OPTIONS = tuple(v for v in (1, 2, 4) if v <= CONV_ROC_MAX)
_CONV_CB_OPTIONS = tuple(v for v in (1, 2, 4, 8) if v <= CONV_CB_MAX)

# 参数名 -> (实例字段, 候选值)。``None`` 是"按形状自动推导"，它随形状变化、
# 钉不成常数，所以永远在候选里；当前值排第一位，即 ``TuningSpace`` 的默认值。
_CONV_KNOBS: Dict[str, Tuple[str, Tuple[int, ...]]] = {
    "tile_w": ("_conv_tile", CONV_TILE_OPTIONS),
    "roc": ("_conv_roc", _CONV_ROC_OPTIONS),
    "cb": ("_conv_cb", _CONV_CB_OPTIONS),
}

# "有没有旋钮"的总集合。逐元素类拧 block，conv2d 拧分块核的线程铺法，
# 具体有哪些参数由 ``tuning_space`` 回答。
_ALL_TUNABLE_OPS: FrozenSet[str] = _TUNABLE_OPS | frozenset({_CONV_TILE_OP})

# ``execute`` 的图执行方式：off = 逐节点下发（默认）；auto = 首次遇到一组
# 形状时捕获成整图，之后只 replay。默认保持 off，因为捕获与 replay 的语义
# 差异（随机掩码、主机同步）需要使用者知情；基准与训练循环可显式打开。
_GRAPH_MODES = ("off", "auto")

# 执行期内存策略。``container`` 是历史行为（容器自己的池）；``tiered`` 由
# 后端提供分层分配器，见 ``DeviceBackend._memory_scope``。
_MEMORY_MODES = ("container", "tiered")


class _KernelHandle:
    """``kernel()`` 的稳定返回面：把后端各自的启动约定收成一个 ``launch``。

    后端内部仍直接用各自的句柄（CUDA 是 cupy 的 ``Function``，靠 ``__call__``
    启动；HIP 是模块函数，靠 ``launch``）。公开入口如果不收口，"打印/编译内核"
    这类跨后端工具就得写两套启动代码——5.168 在 CUDA 真机上正是因为测试按
    HIP 的形态断言 ``handle.launch`` 而失败，才发现两边公开面并不一致。
    """

    __slots__ = ("_fn",)

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def launch(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<kernel handle {self._fn!r}>"


class DeviceBackend(Executor):
    """CUDA / HIP 共享的执行计划层。

    子类必须在调用 ``__init__`` **之前**设置 ``cp``（容器模块）、``codegen``
    （内核方言）与 ``sm_count``（设备的多处理器数量）。这三项是构造期唯一
    需要问硬件的部分，问法两家不同，因此不放在基类。
    """

    name = ""
    # 报错信息里称呼自己的名字。
    _LABEL = "后端"
    # 类级能力声明。子类必须覆盖，构造期会与注册表对账。
    _OPS: FrozenSet[str] = frozenset()
    #: 布局声明：自研卷积 / 池化内核只按 NCHW 寻址。NHWC 的图在
    #: ``invoke`` 入口被显式拒绝，而不是落进内核按错误步长读内存
    #: （DESIGN 5.246：布局传播目前只有参考后端承接）。
    supported_layouts: FrozenSet[str] = frozenset({"nchw"})
    # 允许 f16 的算子集；``None`` = 从 ``_OPS`` 推出。attention 也在其中：
    # 它的分块内核是混合精度（存储 f16、点积与在线 softmax 在 float 里累加，
    # 写回时舍入一次），shared 布局不随 dtype 变化。
    _F16_OPS: Optional[FrozenSet[str]] = None
    # conv2d 的实现选择。子类可以收窄（HIP 没有 cuDNN）。
    CONV_IMPLS: Tuple[str, ...] = _CONV_IMPLS
    # auto 是否允许在"K 超过分块阈值、核面积大于 1"时改走隐式 GEMM。
    # 默认 False：M40 上 5.151 的 A/B 里隐式在 K >= 288 输给 im2col 1.37～2.32×，
    # 不该由"规模大"自动触发。HIP 覆盖成 True，依据见 5.238 的实测。
    CONV_AUTO_IMPLICIT = False
    # sparse_linear 的实现选择。子类可以扩展（CUDA 多一个 "cusparse"）。
    SPARSE_IMPLS: Tuple[str, ...] = _SPARSE_IMPLS

    cp: Any
    codegen: Any
    sm_count: int

    def __init__(
        self,
        block: Optional[int] = None,
        max_blocks: int = 1024,
        conv_impl: str = "auto",
        graph_mode: str = "off",
        max_graph_plans: int = 4,
        min_graph_nodes: int = 2,
        chunked_scan: Optional[int] = None,
        memory_mode: str = "container",
        sparse_impl: str = "custom",
    ) -> None:
        if conv_impl not in self.CONV_IMPLS:
            raise ValueError(
                f"conv_impl 只能是 {self.CONV_IMPLS} 之一，收到 {conv_impl!r}"
            )
        if sparse_impl not in self.SPARSE_IMPLS:
            raise ValueError(
                f"sparse_impl 只能是 {self.SPARSE_IMPLS} 之一，收到 {sparse_impl!r}"
            )
        if graph_mode not in _GRAPH_MODES:
            raise ValueError(
                f"graph_mode 只能是 {_GRAPH_MODES} 之一，收到 {graph_mode!r}"
            )
        if max_graph_plans < 1:
            raise ValueError(f"max_graph_plans 至少为 1，收到 {max_graph_plans!r}")
        if min_graph_nodes < 1:
            raise ValueError(f"min_graph_nodes 至少为 1，收到 {min_graph_nodes!r}")
        if chunked_scan is not None:
            if (
                isinstance(chunked_scan, bool)
                or not isinstance(chunked_scan, int)
                or chunked_scan < 0
            ):
                raise ValueError(
                    f"chunked_scan 必须为非负整数或 None（自动），收到 {chunked_scan!r}"
                )
        if memory_mode not in _MEMORY_MODES:
            raise ValueError(
                f"memory_mode 只能是 {_MEMORY_MODES} 之一，收到 {memory_mode!r}"
            )
        if block is not None:
            check_block(block)
        # None = 按规模自动选（5.65）；显式给值 = 固定这个 block。
        self.block = block
        self.max_blocks = max_blocks
        self.conv_impl = conv_impl
        self.sparse_impl = sparse_impl
        # 执行期内存策略：``container`` = 容器自己的池（默认，行为与之前一致）；
        # ``tiered`` = 子类提供的分层分配器（小对象池 / 中块 best-fit / 大块
        # 直调），并在执行结束后写入 ``memory_stats()``。
        self.memory_mode = memory_mode
        # 分块扫描的块长。``None``（默认）= 按 T 与尾维规模自动选（5.155）：
        # 长 T、小尾维走两趟分块，短 T 或尾维已经喂饱带宽时保持串行；``0`` =
        # 强制串行（逐位可复现）；正数 = 显式块长。分块后块间前缀走了复合，
        # 结果**不逐位**等于串行版，这正是默认值仍可显式关掉的原因。
        self.chunked_scan = None if chunked_scan is None else int(chunked_scan)
        self._modules: Dict[Any, Any] = {}
        self._functions: Dict[Any, Any] = {}
        # 设备上的常数 1（softmax 未给 scale 时使用），常驻以避免每次运行都分配
        self._one_dev = self.cp.ones(1, dtype=self.cp.float32)
        # 设备驻留表：图输入名 -> 已上传到显存的数组（通常是权重）
        self._resident: Dict[str, Any] = {}
        # BLAS handle 惰性创建：图里没有 matmul 就不必加载库、建 handle
        self._blas: Any = None
        # 捕获期正在使用的 arena；非捕获期为空，dispose 因此是零成本的
        self._active_arena: Any = None
        # 最近一次捕获使用的 arena；MemoryContract 从它取统计
        self._last_arena: Any = None
        # 是否正处在图捕获期。捕获期不允许任何 D2H 同步，embedding 的越界
        # 检查因此只登记、不回读（见 _check_ids）
        self._capturing = False
        # 捕获期登记的越界标志：[(设备标量, 算子名, 允许范围), ...]
        self._capture_guards: Optional[List[Tuple[Any, str, str]]] = None
        # 算子注册表：注册即声明，capabilities() 与 invoke 都只认这一份
        self.kernels = KernelRegistry(self.name)
        self._register_kernels()
        # 按算子覆盖的 block（自动调优的产物）
        self._blocks: Dict[str, int] = {}
        # (op, work, vec) -> (block, grid)。铺法是纯函数，同一形状重复执行时
        # 不必每步重算；set_tuning 改动 block 覆盖时整体清空。
        self._launch_cache: Dict[Tuple[str, int, int], Tuple[int, int]] = {}
        # conv2d 分块核的三个旋钮（None = 计划层的默认推导）
        self._conv_tile: Optional[int] = None
        self._conv_roc: Optional[int] = None
        self._conv_cb: Optional[int] = None
        # f16 支持面：子类声明的集合优先，否则就是全部已实现算子
        self._f16_ops: FrozenSet[str] = (
            self._OPS if self._F16_OPS is None else self._F16_OPS
        )
        # 自动图执行：auto 时 execute 会按"图 + 结构版本 + 输入签名"缓存捕获
        # 好的计划，命中就只 replay。键里带 id 与弱引用：图被回收后 id 可能
        # 被复用，命中前必须确认弱引用还指向同一张图。
        self.graph_mode = graph_mode
        self._max_graph_plans = int(max_graph_plans)
        # 低于这个节点数不捕获：单算子图本来就只有一次 launch，replay 只多出
        # 簿记成本，还固定了 cuBLAS 的算法（5.79 实测 matmul_batched 慢 8%）。
        self._min_graph_nodes = int(min_graph_nodes)
        self._graph_plans: "OrderedDict[Any, Tuple[Any, Any]]" = OrderedDict()
        # 捕获失败按图隔离：id(graph) -> (弱引用, 图版本, 原因)。一张图失败
        # （例如图里有主机同步）不代表别的图也捕不了；旧实现把整个后端
        # 关掉，一次失败就让之后所有图都退回逐节点，且原因再也查不到。
        self._graph_failures: Dict[int, Tuple[Any, int, str]] = {}
        # 调优配置/驻留权重的版本。任何让"已捕获计划"或"线程铺法缓存"失效
        # 的变更都 +1，并被写进计划键：只清某一层时，其余层会带着旧前提
        # 继续命中（block 变了、replay 的却是旧 launch 序列）。
        self._config_epoch = 0
        # 公开 kernel() 的包装句柄，按编译键缓存：同一个 key 必须返回同一个
        # 对象（契约要求），而包装器每次新建就会破坏这一点。与厂商侧的
        # _functions 生命周期相同，调优不会让编译产物失效。
        self._kernel_handles: Dict[Any, _KernelHandle] = {}

    # ------------------------------------------------------------------
    # 子类必须实现
    # ------------------------------------------------------------------

    def kernel(self, key, source, func: str):
        """编译（或取缓存）一个内核，返回可调用句柄。

        这是 ``contracts.CodegenContract`` 的公开入口；真正实现在厂商类里
        （CUDA 走 NVRTC + ``cupy.RawModule``，HIP 走 ``hiprtc`` + 模块函数）。
        公开这一层而不是让工具直接调 ``_kernel``：契约探测只看方法名，工具
        需要一条不依赖下划线命名的稳定路径。

        返回值统一包成 ``_KernelHandle``：两种后端都能 ``launch(...)``，也保留
        ``__call__`` 转发。同一 ``key`` 返回同一个包装对象。
        """
        handle = self._kernel_handles.get(key)
        if handle is None:
            handle = _KernelHandle(self._kernel(key, source, func))
            self._kernel_handles[key] = handle
        return handle

    def _kernel(self, key, source, func: str):
        """编译（或取缓存）一个内核函数。"""
        raise NotImplementedError(f"{self._LABEL}没有实现内核编译")

    def _cublas(self):
        """惰性取得 BLAS 句柄。"""
        raise NotImplementedError(f"{self._LABEL}没有实现 BLAS")

    def current_stream_ptr(self) -> int:
        """当前流的设备地址，供 BLAS 绑定。"""
        raise NotImplementedError(f"{self._LABEL}没有实现当前流查询")

    def new_stream(self) -> Any:
        """一条非阻塞流（捕获与 replay 都跑在它上面）。"""
        raise NotImplementedError(f"{self._LABEL}没有实现流")

    def memory_stats(self) -> Optional[MemoryStats]:
        """``MemoryContract``：最近一次捕获所用分层分配器的统计。

        只有 ``reuse_memory=True`` 的捕获走 ``Arena``；普通执行与不走复用的
        捕获都用容器自己的池，而池不提供峰值。那种情况下返回 ``None``——
        不编一个数字出来，也不拿"当前已分配的字节数"冒充峰值。
        """
        arena = self._last_arena
        return arena.stats() if arena is not None else None

    def allocator(self) -> Any:
        """新建一个捕获期分配器。"""
        raise NotImplementedError(f"{self._LABEL}没有实现分配器")

    def capture(
        self,
        graph,
        sample_inputs: Mapping[str, Any],
        reuse_memory: bool = True,
        shared_buffers: Optional[Mapping[str, Any]] = None,
        stream: Optional[Any] = None,
    ):
        """把整张图捕获成一个可重复 replay 的计划。"""
        raise NotImplementedError(f"{self._LABEL}没有实现图捕获")

    # ------------------------------------------------------------------
    # 可选的设备相关行为（默认无操作）
    # ------------------------------------------------------------------

    def _pick_cudnn(self, dtype, groups: int) -> bool:
        """是否走 cuDNN 卷积。默认没有这项加速。"""
        return False

    def dispose(self, value: Any) -> None:
        """把中间值的地址立刻还给分配器。默认无操作。"""
        return None

    def _alias_view(self, owner) -> None:
        """登记一次视图引用。默认无操作。"""
        return None

    def _shares_storage(self, result, source) -> bool:
        """``result`` 是否与 ``source`` 共用同一段设备内存。

        默认按"视图"判断：``transpose`` 的恒等置换在两家容器上都返回视图，
        只有真的重排才会新分配。``result is source`` 覆盖 HIP 的
        ``ascontiguousarray``——它对已连续的输入直接原样返回。
        """
        if result is source:
            return True
        base = getattr(result, "base", None)
        if base is None:
            return False
        return base is source or getattr(base, "base", None) is source


    # ------------------------------------------------------------------
    # 契约
    # ------------------------------------------------------------------

    def capabilities(self) -> FrozenSet[str]:
        return self.kernels.capabilities()



    def _register_kernels(self) -> None:
        """显式登记全部内核，然后与类级声明对账。

        不用 ``getattr(self, f"_op_{name}")`` 反射：反射把"声明了哪些算子"与
        "实际有哪些内核"隔着一层字符串拼接，拼错要到第一次推理才暴露。
        显式登记后，漏写、写错、重复都会在**构造时**失败。
        """
        r = self.kernels
        r.register("add", self._op_add)
        r.register("sub", self._op_sub)
        r.register("mul", self._op_mul)
        r.register("div", self._op_div)
        r.register("relu", self._op_relu)
        r.register("silu", self._op_silu)
        r.register("exp", self._op_exp)
        r.register("log", self._op_log)
        r.register("sqrt", self._op_sqrt)
        r.register("neg", self._op_neg)
        r.register("tanh", self._op_tanh)
        r.register("sigmoid", self._op_sigmoid)
        r.register("erf", self._op_erf)
        r.register("step", self._op_step)
        r.register("spike", self._op_spike)
        r.register("constant", self._op_constant)
        r.register("broadcast_to", self._op_broadcast_to)
        r.register("matmul", self._op_matmul)
        r.register("quantize_i8", self._op_quantize_i8)
        r.register("dequantize_i8", self._op_dequantize_i8)
        r.register("linear_i8", self._op_linear_i8)
        r.register("conv2d", self._op_conv2d)
        r.register("maxpool2d", self._op_maxpool2d)
        r.register("maxpool2d_indices", self._op_maxpool2d_indices)
        r.register("softmax", self._op_softmax)
        r.register("attention", self._op_attention)
        r.register("layernorm", self._op_layernorm)
        r.register("reduce_sum", self._op_reduce_sum)
        r.register("reduce_max", self._op_reduce_max)
        r.register("reduce_mean", self._op_reduce_mean)
        r.register("argmax", self._op_argmax)
        r.register("reshape", self._op_reshape)
        r.register("transpose", self._op_transpose)
        r.register("cast", self._op_cast)
        r.register("slice", self._op_slice)
        r.register("gelu", self._op_gelu)
        r.register("rmsnorm", self._op_rmsnorm)
        r.register("avgpool2d", self._op_avgpool2d)
        r.register("avgpool2d_backward", self._op_avgpool2d_backward)
        r.register("adaptive_avg_pool2d", self._op_adaptive_avg_pool2d)
        r.register(
            "adaptive_avg_pool2d_backward", self._op_adaptive_avg_pool2d_backward
        )
        r.register("concat", self._op_concat)
        r.register("embedding", self._op_embedding)
        r.register("embedding_backward", self._op_embedding_backward)
        r.register("pad", self._op_pad)
        r.register("maxpool2d_backward", self._op_maxpool2d_backward)
        r.register("conv2d_input_grad", self._op_conv2d_input_grad)
        r.register("conv2d_weight_grad", self._op_conv2d_weight_grad)
        r.register("dropout_mask", self._op_dropout_mask)
        r.register("batch_norm", self._op_batch_norm)
        r.register("batch_norm_stats", self._op_batch_norm_stats)
        r.register("scan_linear", self._op_scan_linear)
        r.register("scan_linear_backward", self._op_scan_linear_backward)
        r.register("cumsum", self._op_cumsum)
        r.register("tril", self._op_tril)
        r.register("clamp", self._op_clamp)
        r.register("mod", self._op_mod)
        r.register("sparse_linear", self._op_sparse_linear)
        r.register("sparse_linear_weight_grad", self._op_sparse_linear_weight_grad)
        r.register("sparse_linear_values_grad", self._op_sparse_linear_values_grad)
        r.register("spike_linear", self._op_spike_linear)
        r.register("spike_linear_weight_grad", self._op_spike_linear_weight_grad)
        r.register("spike_offsets", self._op_spike_offsets)
        r.register("spike_csr", self._op_spike_csr)

        if r.capabilities() != self._OPS:
            raise RuntimeError(
                "CUDA 插件的能力声明与注册的内核不一致："
                f"声明 {sorted(self._OPS - r.capabilities())} 未注册，"
                f"注册 {sorted(r.capabilities() - self._OPS)} 未声明"
            )



    def to_array(self, value: Any) -> Any:
        cp = self.cp
        # 已在显存且连续：零拷贝复用，这是设备驻留能省掉 H2D 的前提
        if isinstance(value, cp.ndarray):
            arr = value
        else:
            arr = cp.asarray(value)
        # cupy 的 ascontiguousarray 会把 0 维数组变成 shape (1,)。标量本身天然
        # 连续，直接返回——反向图的种子输入就是 0 维，否则整条训练路径在
        # 形状绑定处就报"秩不匹配"。
        if arr.ndim == 0:
            return arr
        return arr if arr.flags.c_contiguous else cp.ascontiguousarray(arr)



    def cast(self, value: Any, dtype: np.dtype) -> Any:
        """按需转换元素类型。

        **f32 <-> f16 不走 cupy 的 ``astype``。** cupy 的 f16 转换内核需要
        ``__half`` 的构造函数，而 NVRTC 只能拿到声明、链接不到定义，于是编译期
        报 ``Unresolved extern function '_ZN6__halfC1Ef'``——这条路径以前没人走过，
        因为整图 f16 的用例都是直接声明 f16 输入，从不产生 ``cast`` 节点。
        改用本地的 ``cast_source``（PTX 的 ``cvt`` 指令，不依赖 ``cuda_fp16.h``）。
        """
        target = np.dtype(dtype)
        if value.dtype == target:
            return value
        src_key = self._kind(value)
        dst_key = {np.dtype(np.float16): "f16", np.dtype(np.float32): "f32"}.get(target)
        if dst_key is None or dst_key == src_key:
            # 与设备存储无关的类型（f64 等）：交给 cupy，走不到 GPU 计算路径
            return value.astype(target, copy=False)
        return self._cast_kernel(value, target, src_key, dst_key)



    def _cast_kernel(self, value, target, src_key: str, dst_key: str):
        cp = self.cp
        out = cp.empty(value.shape, target)
        n = int(value.size)
        if n == 0:
            return out
        kernel = self._kernel(
            ("cast", src_key, dst_key),
            lambda: self.codegen.cast_fn(value.dtype, target),
            "ew_cast",
        )
        block, grid = self._grid("cast", n)
        kernel((grid,), (block,), (value, out, np.int64(n)))
        return out



    def _contig(self, a):
        """已经连续就原样返回；只有真的不连续才拷一份。

        ``cp.ascontiguousarray`` 对连续数组是零拷贝，但每次调用仍要穿过
        cupy 的 Python 包装（约 5 µs）。它出现在每个算子的参数入口上，
        24 节点的图里一跑就是十几万次/秒这种量级，是纯常数开销。
        """
        return a if a.flags.c_contiguous else self.cp.ascontiguousarray(a)



    def _alias_view(self, owner) -> None:
        """捕获期登记一次视图引用，让源地址在视图死掉之前不被复用。

        非捕获期是空操作：普通执行靠 cupy 的引用计数，源数组只要还被视图
        的 ``base`` 指着就不会被回收。
        """
        arena = self._active_arena
        if arena is not None:
            arena.alias(owner.data.ptr)

    def _shared_view(self, out, src):
        """``out`` 是共享 ``src`` 存储的另一个对象时，给源地址记一次引用。

        "零拷贝算子"不止 ``reshape``：``slice`` 与退化成恒等的
        ``broadcast_to`` 也返回视图。执行器按"最后一次被读"回收的是**源
        数组**，源先死、视图还在环境里等着被读，arena 就可能把这段地址
        分给下一个算子。每个返视图的算子各写一次判断，正是 5.83 里漏掉
        ``slice`` 的那条路——统一从这里走。

        ``out is src`` 也要记：整段切片 / 同形状 reshape 返回的就是源数组
        本身，但执行器看到的是两条 Value——源那条在算子执行后立刻归零，
        地址会被下一个分配拿走，而"输出"那条还要读。记账之后两次 dispose
        各扣一次，归零才还块。
        """
        if self._shares_storage(out, src):
            self._alias_view(src)
        return out



    def to_host(self, value: Any) -> np.ndarray:
        return self.cp.asnumpy(value)


    # ------------------------------------------------------------------
    # 设备驻留
    # ------------------------------------------------------------------

    def resident_parameters(
        self, graph, values: Mapping[str, Any]
    ) -> None:
        """把参数一次性上传到显存并登记为常驻。

        之后 ``execute`` 不必再传这些输入，H2D 只发生在激活值上。
        显式传入同名值时以传入值为准（便于更新权重或做对照实验），
        这样不会出现"以为用了新权重、其实用了旧驻留值"的隐蔽问题。
        """
        declared = {v.name: v.type for v in graph.inputs}
        unknown = sorted(set(values) - set(declared))
        if unknown:
            raise KeyError(f"这些名字不是图输入: {unknown}")
        binding: Dict[str, int] = {}
        for name, data in values.items():
            arr = self.to_array(data)
            # 提前做形状校验：让"驻留了错的权重"在驻留时失败，而不是第一次推理时
            self._bind(declared[name], tuple(arr.shape), f"常驻参数 %{name}", binding)
            self._resident[name] = arr
        # 已捕获的计划把旧权重写死在静态缓冲里：不清掉，下一次 execute 会
        # 悄悄用旧权重算出"看起来正常"的结果。线程铺法与失败记录同样依赖
        # 这次变更，一并作废。
        self._invalidate_plans()



    def release_parameters(self) -> None:
        """释放全部驻留参数。"""
        self._resident.clear()
        self._invalidate_plans()

    @property
    def resident(self) -> Mapping[str, Any]:
        """当前驻留表的只读快照。"""
        return dict(self._resident)



    def _memory_scope(self):
        """执行期内存作用域。基类什么都不做（用容器自己的池）。

        子类在 ``tiered`` 模式下把驱动/容器的分配器换成分层分配器，退出时
        还原并把统计写进 ``_last_arena``——``memory_stats()`` 因此不需要
        区分"捕获期的统计"与"执行期的统计"，它读到的永远是最近一次。
        """
        return nullcontext()

    def execute(self, graph, inputs: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        with self._memory_scope():
            return self._execute(graph, inputs)

    def execute_device(self, graph, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        """执行整张图，但**不把输出拷回主机**。

        与 ``execute`` 的唯一差别是省掉结尾的 ``to_host``：一次 4 MB 的 D2H
        在这台 A800 上要 1.4 ms，而它的内核只要 0.05 ms——逐图回传会把性能
        测量变成 PCIe 带宽测量（见 DESIGN 5.251）。基准与设备侧流水线应当
        用这条路径；需要 numpy 结果的调用方仍用 ``execute``。

        **与 ``execute`` 一样吃 ``graph_mode="auto"``**：节点数达标且捕获成功时
        走 ``plan.run_device``，不再逐节点过 Python。少了这一段，多节点图在
        A800 上会慢 3～5 倍（5.251 的实测），测出来的就不是流水线的吞吐。

        返回的数组不一定归调用方：命中捕获计划时是计划持有的输出缓冲，下一次
        replay 会覆盖它。要跨轮保存请自己 ``copy()``。
        """
        with self._memory_scope():
            if self._resident:
                merged = dict(self._resident)
                merged.update(inputs)
            else:
                merged = inputs
            if self.graph_mode == "auto" and self._graph_failure(graph) is None:
                if len(graph.nodes) >= self._min_graph_nodes:
                    plan = self._graph_plan(graph, inputs, merged)
                    if plan is not None:
                        return plan.run_device(inputs)
            env = self.run_device(graph, merged, release=True)
            return {
                name: env[v.id]
                for v, name in zip(graph.outputs, graph.output_names)
            }

    def _execute(self, graph, inputs: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        if self._resident:
            merged = dict(self._resident)
            merged.update(inputs)  # 显式传入的覆盖驻留值
        else:
            merged = inputs
        if self.graph_mode == "auto" and self._graph_failure(graph) is None:
            # 节点太少时不值得捕获：省下的 launch 次数抵不过 replay 的簿记，
            # 而且捕获会把 cuBLAS / cuDNN 的算法固定成捕获时选中的那个。
            if len(graph.nodes) >= self._min_graph_nodes:
                plan = self._graph_plan(graph, inputs, merged)
                if plan is not None:
                    # 计划只认它捕获时记为可变的那些输入，即 ``inputs`` 的键；
                    # 驻留权重已被写进静态缓冲，不需要每步重传。
                    return plan.run(inputs)
        return self.run(graph, merged)



    def _graph_plan(self, graph, inputs: Mapping[str, Any], merged: Mapping[str, Any]):
        """取（必要时捕获）这张图在当前输入签名下的计划。

        签名包含合并后的全部输入：驻留权重换了形状时不能沿用旧计划。
        ``inputs`` 单独传给 ``capture``——其中的名字才会被记为可变输入，
        只出现在驻留表里的权重则应固定在图里，否则每步都要重拷一遍。
        """
        # 签名必须同时覆盖"合并后的全部输入"和"这次显式传入的输入集合"：
        # ``capture`` 用后者决定哪些缓冲可变，其余（驻留权重）被写死进静态
        # 缓冲。只按 merged 的 (名字, 形状, dtype) 做键时，先驻留 w 用 {x}
        # 捕获、之后显式传 {x, w_new}（形状类型不变）会命中旧计划，而旧计划
        # 只把 x 当可变输入，新 w 根本不会被拷进去。
        key = (
            self._config_epoch,
            id(graph),
            graph.revision,
            tuple(sorted(inputs)),
            self._signature(merged),
        )
        entry = self._graph_plans.get(key)
        if entry is not None:
            if entry[0]() is graph:
                self._graph_plans.move_to_end(key)
                return entry[1]
            # id 被另一张图复用了：旧计划对新图毫无意义，直接丢掉
            del self._graph_plans[key]
        try:
            plan = self.capture(graph, inputs)
        except Exception as exc:  # noqa: BLE001 - 捕获期的任何失败都只影响这张图
            # 捕获期比逐节点执行严格得多（不允许主机同步、cuDNN algo 查询等）。
            # 回落是语义正确的路径；失败只记在这张图的名下，其他图照常尝试，
            # 原因保留下来供 ``graph_capture_failure`` 查询。
            self._graph_failures = {
                ident: item
                for ident, item in self._graph_failures.items()
                if item[0]() is not None  # 顺手清掉已被回收的图
            }
            self._graph_failures[id(graph)] = (
                ref(graph),
                graph.revision,
                f"{type(exc).__name__}: {exc}",
            )
            return None
        self._graph_plans[key] = (ref(graph), plan)
        # 按形状缓存多个计划，但必须封顶：每个计划的 arena 都是常驻显存。
        while len(self._graph_plans) > self._max_graph_plans:
            self._graph_plans.popitem(last=False)
        return plan

    def _graph_failure(self, graph) -> Optional[str]:
        """这张图（当前结构版本）的捕获失败原因；没有失败过返回 ``None``。"""
        entry = self._graph_failures.get(id(graph))
        if entry is None:
            return None
        holder, revision, reason = entry
        if holder() is not graph or revision != graph.revision:
            # id 被复用，或图结构已经变了：那条失败记录与现在这张图无关
            del self._graph_failures[id(graph)]
            return None
        return reason

    def graph_capture_failure(self, graph) -> Optional[str]:
        """查询这张图上一次捕获失败的原因（``"异常类型: 消息"``）。

        失败的作用域是"这张图 + 这个结构版本"：换一张图、改图（``revision``
        变化）、换调优配置或换驻留权重都会让它重新尝试捕获。查询返回 ``None``
        只说明"没有记录"，不保证下一次捕获一定成功。
        """
        return self._graph_failure(graph)

    def _invalidate_plans(self) -> None:
        """让所有由调优配置/驻留权重派生的缓存一起失效。

        ``_launch_cache``（线程铺法）、``_graph_plans``（已捕获的 launch 序列）
        与 ``_graph_failures``（按图的捕获失败）都以"当前配置"为前提。只清
        其中一层，其余层会带着旧前提继续命中；``_config_epoch`` 同时被写进
        计划键，即使某条路径忘了清表，旧计划也不会被复用。
        """
        self._config_epoch += 1
        self._launch_cache.clear()
        self._graph_plans.clear()
        self._graph_failures.clear()

    @staticmethod
    def _signature(inputs: Mapping[str, Any]) -> tuple:
        """输入的 ``(名字, 形状, dtype)`` 摘要，用于区分不同的捕获计划。"""
        parts = []
        for name, value in inputs.items():
            shape = getattr(value, "shape", None)
            dtype = getattr(value, "dtype", None)
            if shape is None:
                arr = np.asarray(value)
                shape, dtype = arr.shape, arr.dtype
            parts.append((name, tuple(shape), np.dtype(dtype).str))
        parts.sort()
        return tuple(parts)


    # ------------------------------------------------------------------
    # 自动调优：可调参数由插件自己声明，搜索器在 tuning 包里
    # ------------------------------------------------------------------

    def tunable_ops(self) -> FrozenSet[str]:
        """哪些算子有可调参数。

        两类旋钮：``block``（逐元素类与归约类共用同一个参数名），以及 conv2d
        分块核的线程铺法。归约类的 shared 暂存区是固定大小（``REDUCE_SCRATCH``），
        不随 block 变化，因此与逐元素类一起开放；两者的差别只在默认值怎么来——
        逐元素按元素数自动选，归约按列数取 2 的幂。这个集合只回答"有没有旋钮"，
        具体参数用 ``tuning_space``。
        """
        return _ALL_TUNABLE_OPS



    def tuning_space(self, op: str) -> TuningSpace:
        """``op`` 的可调参数与候选值；当前生效值排在第一位。

        首候选就是空间里的默认值，所以"未调优"与"默认配置"是同一件事：
        搜索器报 ``speedup`` 时的基准，以及智能体在配置库未命中时给出的建议，
        都和后端此刻真正在跑的东西一致。若把固定候选表的第一项当默认，
        在一个自动选 block 的后端上就会推荐 128，凭空制造一个"默认值"。

        conv2d 的空间是三个参数，且每个的首候选都可能是 ``None``：分块核的
        ``tile_w`` / ``roc`` / ``cb`` 的默认值都随形状变化（W 小于 tile 时会被
        截断、``toc`` 随 OC 与核面积变化），而调优空间不知道形状，所以"自动"
        本身必须是一个候选。

        逐元素类的 ``block`` 同理：默认（``block=None``）是按规模自动选，
        一个形状一个值，钉不成常数，因此首候选也是 ``None``。归约类共用同一个
        参数名，默认同样是 ``None``（由 ``plan_reduce`` 按列数推导）。

        构造参数 ``self.block`` 非空时，它才是首候选：那是调用方明确要求的
        固定值，``_launch_block`` / ``_reduce_block`` 在没有按算子覆盖时都用它。
        """
        if op == _CONV_TILE_OP:
            options: Dict[str, Tuple[Any, ...]] = {}
            for name, (field, candidates) in _CONV_KNOBS.items():
                current = getattr(self, field)
                options[name] = (current,) + tuple(
                    v for v in candidates if v != current
                )
            return TuningSpace.build(options)
        self._require_tunable(op)
        current = self._blocks.get(op, self.block)
        options = (current,) + tuple(v for v in BLOCK_OPTIONS if v != current)
        return TuningSpace.build({"block": options})



    def set_tuning(self, op: str, config: Mapping[str, Any]) -> None:
        """应用一条配置。非法参数明确拒绝，不静默取近似值。

        ``block=None`` 表示"回到自动"：它不是一个能存下来的覆盖，而是把该
        算子的覆盖撤掉——因此 ``tuning_config()`` 里不会出现这一项。
        """
        if op == _CONV_TILE_OP:
            self._set_conv_tuning(config)
            return
        self._require_tunable(op)
        unknown = sorted(set(config) - {"block"})
        if unknown:
            raise ValueError(
                f"{op} 不认识调优参数 {unknown}；它只接受 block（收到 {dict(config)}）"
            )
        if "block" not in config:
            raise ValueError(f"{op} 的配置缺少 block（收到 {dict(config)}）")
        block = config["block"]
        if block is None:
            self._blocks.pop(op, None)
            self._invalidate_plans()
            return
        check_block(block)
        self._blocks[op] = block
        # 覆盖变了：线程铺法与已捕获的 launch 序列都以旧 block 为前提，
        # 必须一起作废，否则下一次 execute replay 的仍是旧铺法。
        self._invalidate_plans()



    def _set_conv_tuning(self, config: Mapping[str, Any]) -> None:
        """conv2d 的覆盖：``tile_w`` / ``roc`` / ``cb``，各自为正整数或 ``None``。

        三个旋钮可以只给其中任意一部分；``None`` 表示撤掉该项覆盖、回到计划层
        的按形状推导。这里只做"值本身合法"的判断——``tile_w * block_y`` 是否超
        block 上限、``roc`` 能否整除 OC tile、``cb`` 能否放进 shared 预算，都
        取决于具体卷积的形状，由 ``plan.plan_conv_tiled`` 在 launch 前判定。
        """
        unknown = sorted(set(config) - set(_CONV_KNOBS))
        if unknown:
            raise ValueError(
                f"conv2d 不认识调优参数 {unknown}；"
                f"它只接受 {sorted(_CONV_KNOBS)}（收到 {dict(config)}）"
            )
        if not config:
            raise ValueError(
                f"conv2d 的配置至少要有一个参数：{sorted(_CONV_KNOBS)}"
            )
        # 先整份校验再落库：中间某个值非法时，不能留下半应用的配置。
        for name, value in config.items():
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} 必须是正整数或 None，收到 {value!r}")
        for name, value in config.items():
            setattr(self, _CONV_KNOBS[name][0], value)
        # 三个旋钮同样决定 launch 铺法：已捕获的计划不能留着
        self._invalidate_plans()



    def clear_tuning(self, op: Optional[str] = None) -> None:
        """清掉一条或全部覆盖；之后回到 ``self.block`` 或按规模的自动选择。

        ``set_tuning`` 收到 ``block=None`` 时走的就是这里：自动不是一个能存下来的
        覆盖，撤掉覆盖才是它的准确表示。
        """
        if op is None:
            self._blocks.clear()
            for field, _ in _CONV_KNOBS.values():
                setattr(self, field, None)
            self._invalidate_plans()
            return
        if op == _CONV_TILE_OP:
            for field, _ in _CONV_KNOBS.values():
                setattr(self, field, None)
            self._invalidate_plans()
            return
        self._require_tunable(op)
        self._blocks.pop(op, None)
        self._invalidate_plans()



    def tuning_config(self) -> Dict[str, Dict[str, Any]]:
        """当前生效的按算子覆盖（只读快照）；没有覆盖时为空字典。

        每项是一个参数字典，而不是单个 block：两类旋钮的参数名不同，
        用同一种形状表示才不需要调用方按算子名分支。
        """
        out: Dict[str, Dict[str, Any]] = {
            op: {"block": block} for op, block in self._blocks.items()
        }
        conv: Dict[str, Any] = {}
        for name, (field, _) in _CONV_KNOBS.items():
            value = getattr(self, field)
            # None 不是"能存下来的覆盖"，而是"回到自动"，因此不出现在快照里。
            if value is not None:
                conv[name] = value
        if conv:
            out[_CONV_TILE_OP] = conv
        return out



    def _require_tunable(self, op: str) -> None:
        """``op`` 是否接受按算子的 block 覆盖。

        conv2d 不在这里：它拧的是分块核的线程铺法，走 ``_set_conv_tuning``。
        """
        if op not in _TUNABLE_OPS:
            raise ValueError(
                f"{op!r} 不接受 block 覆盖；可拧 block 的算子：{sorted(_TUNABLE_OPS)}。"
                f"conv2d 拧的是分块核的线程铺法，用 set_tuning('conv2d', ...)"
            )



    def _launch_block(self, op: str, work: int, vec: int = 1) -> int:
        """该算子这一次 launch 用多大 block。

        优先级：按算子的调优覆盖 → 构造参数给的固定值 → 按规模自动选。
        前两者都是"人定的"，自动只在没人说话时接手。
        """
        override = self._blocks.get(op)
        if override is not None:
            return override
        if self.block is not None:
            return self.block
        return auto_block(
            work, vec=vec, preferred=BLOCK_DEFAULT, target_blocks=self.sm_count
        )



    def _reduce_block(self, op: str, cols: int) -> int:
        """归约类这一次 launch 的 block：覆盖 → 构造参数 → 按列数推导。

        与 ``_launch_block`` 的优先级一致，但默认值来自 ``plan_reduce``：末维
        归约是"一个 block 一行"，block 应当不小于列数，让块内归约一次做完；
        列数小的时候少派线程。覆盖值可以故意小于列数——内核沿 ``blockDim.x``
        跨步读，正确性不受影响，只是每个线程多走几步。

        构造参数 ``self.block`` 在这里也生效，与逐元素类同一口径：它是"全局固定
        block"，不是"只对逐元素生效"。
        """
        override = self._blocks.get(op)
        if override is not None:
            return override
        if self.block is not None:
            return self.block
        return plan_reduce(cols)



    def _grid(self, op: str, work: int, vec: int = 1) -> Tuple[int, int]:
        """``(block, grid)``：扁平内核的线程铺法一次算完，减少漏改一处的机会。"""
        key = (op, work, vec)
        hit = self._launch_cache.get(key)
        if hit is not None:
            return hit
        block = self._launch_block(op, work, vec=vec)
        value = (
            block,
            grid_for(work, block=block, vec=vec, max_blocks=self.max_blocks),
        )
        self._launch_cache[key] = value
        return value

    # ------------------------------------------------------------------
    # 分发
    # ------------------------------------------------------------------

    def invoke(self, node: Node, args: List[Any], attrs: dict) -> Any:
        name = node.op.name
        # 先判注册，再判 dtype：未实现的算子无论输入什么类型，原因都是"未实现"
        if name not in self.kernels:
            raise BackendCapabilityError(
                f"{self._LABEL}未实现算子 {name}（节点 %{node.output.name}）"
            )
        if name in _LAYOUT_OPS and attrs.get("layout", "nchw") != "nchw":
            raise BackendCapabilityError(
                f"{self._LABEL}的 {name} 只承接 NCHW 布局，收到 layout="
                f"{attrs.get('layout')!r}；自研内核按 NCHW 寻址，"
                "NHWC 需要显式的布局转换（见 DESIGN 5.246）"
            )
        index_args = _INDEX_ARGS.get(name, frozenset())
        for i, a in enumerate(args):
            if i in index_args:
                if str(a.dtype) not in _INDEX_DTYPES:
                    raise BackendCapabilityError(
                        f"{self._LABEL}的 {name} 的整数下标必须是 i32/i8，收到 {a.dtype}"
                    )
                continue
            if a.dtype == self.cp.float32:
                continue
            if a.dtype == self.cp.float16 and name in self._f16_ops:
                continue
            hint = ""
            if a.dtype in (self.cp.float16, self.cp.float64):
                accept = "float16 / float32" if name in self._f16_ops else "float32"
                hint = f"（{name} 目前只支持 {accept}）"
            raise BackendCapabilityError(
                f"{self._LABEL}的 {name} 不支持 {a.dtype}{hint}。"
                "不做隐式降级：请在前端显式插入 cast。"
            )
        return self.kernels.dispatch(name, node, args, attrs, context=node)

    # ------------------------------------------------------------------
    # 批量派发：把逐节点的校验挪到计划构建期
    # ------------------------------------------------------------------

    def _prepare_plan(self, plan) -> None:
        """为每个执行组预绑定内核函数（``invoke`` 的检查只做一次）。

        设备端每个节点都要过一遍"算子是否注册 + 每个输入 dtype 是否受支持"，
        24 节点的图每步因此扔掉几十微秒。这些判定只取决于 IR：图输入在入口
        已 cast 成声明类型，组输出在骨架里也按声明 cast，所以运行期看到的
        dtype 恒等于声明 dtype。判定一次就够了。

        任何一条不满足就把整组留在原路径上（``kernels`` 保持 ``None``），
        这样"未实现算子 / 不支持 dtype"仍在第一次执行时按原来的时机和文案
        抛出，不会因为预判而提前或消失。
        """
        for group in plan:
            fns = []
            for node in group.nodes:
                fn = self._planned_kernel(node)
                if fn is None:
                    fns = None
                    break
                fns.append(fn)
            if fns is not None:
                group.kernels = tuple(fns)
            # 常量节点的值只写在属性里、与运行期输入无关，因此在计划期上传
            # 一次就够：每次执行都 H2D 一份，正是 5.161 里"折叠产物大而原
            # 计算便宜时反而倒退"（mask 用例逐节点 GPU 时间 +22.4%）的根因。
            # 计划随图的版本号缓存，设备数组因此与图同寿命；切卡 / 关闭
            # arena 时由 ``_clear_plan_values`` 清掉。
            if len(group.nodes) == 1 and group.nodes[0].op.name == "constant":
                node = group.nodes[0]
                if node.output.type.dtype in (DType.F32, DType.F16):
                    group.constants[node.output.id] = self._op_constant(
                        node, (), node.attrs
                    )

    def _planned_kernel(self, node: Node):
        """按 IR 声明预判能否跳过 ``invoke`` 的校验；能就返回内核函数。

        与 ``invoke`` 的检查逐条对应：注册表里有这个算子；整数下标的位置
        （``_INDEX_ARGS``）声明为 i32/i8；其余输入声明为 f32，或 f16 且该
        算子在 f16 支持面内。返回 ``None`` 表示"这条不能预判"，走原路径。
        """
        name = node.op.name
        fn = self.kernels.get(name)
        if fn is None:
            return None
        if name in _LAYOUT_OPS and node.attrs.get("layout", "nchw") != "nchw":
            # 布局不匹配要留给 ``invoke`` 报错，不能在这里返回内核——
            # 否则快路径会拿 NCHW 内核去读 NHWC 的缓冲区。
            return None
        index_args = _INDEX_ARGS.get(name, frozenset())
        for i, value in enumerate(node.inputs):
            dtype = value.type.dtype
            if i in index_args:
                if str(np.dtype(dtype.np_dtype)) not in _INDEX_DTYPES:
                    return None
                continue
            if dtype == DType.F32:
                continue
            if dtype == DType.F16 and name in self._f16_ops:
                continue
            return None
        return fn

    # ------------------------------------------------------------------
    # 逐元素融合（语义级 L2）
    # ------------------------------------------------------------------

    def group_nodes(self, graph):
        """相邻的同形逐元素算子交给一个内核（计划见 ``_group_elementwise``）。"""
        return _group_elementwise(graph)



    def invoke_group(self, group, args, attrs, plan=None):
        """一组节点：能融合就一次内核算完，否则逐节点执行。

        运行期还要再查一次（形状是否真的相同、是否连续、指针是否 16 字节
        对齐、长度能否被 4 整除），因此静态计划即使乐观也不会算错——最多
        白算一次计划，然后走回退路径。

        融合失败后交给父类：``plan.kernels`` 有值时走批量派发快路径，
        否则仍是原来的逐节点 ``invoke``。
        """
        if len(group) > 1:
            fused = self._try_chain(group, args)
            if fused is not None:
                return fused
        return super().invoke_group(group, args, attrs, plan=plan)



    def _try_chain(self, group, args):
        """尝试把一组逐元素节点编成一个内核；不满足运行期前提时返回 None。"""
        cp = self.cp
        internal = {node.output.id for node in group[:-1]}
        entries: List[tuple] = []  # [(value_id, array)]，按参数顺序
        param_of: Dict[int, str] = {}
        local_of: Dict[int, str] = {}
        steps: List[ElemStep] = []

        for idx, (node, node_args) in enumerate(zip(group, args)):
            arg_names = []
            for value, arr in zip(node.inputs, node_args):
                if value.id in internal:
                    arg_names.append(local_of[value.id])
                    continue
                name = param_of.get(value.id)
                if name is None:
                    name = f"p{len(param_of)}"
                    param_of[value.id] = name
                    entries.append((value.id, arr))
                arg_names.append(name)
            local = f"t{idx}"
            local_of[node.output.id] = local
            steps.append(ElemStep(out=local, op=node.op.name, args=tuple(arg_names)))

        arrays = [arr for _, arr in entries]
        if not arrays:
            return None
        dtype = arrays[0].dtype
        if dtype not in (cp.float32, cp.float16):
            return None
        if any(a.dtype != dtype or not a.flags.c_contiguous for a in arrays):
            return None

        shapes = [tuple(a.shape) for a in arrays]
        try:
            out_shape = tuple(np.broadcast_shapes(*shapes))
        except ValueError:
            return None

        specs = []
        for vid, arr in entries:
            if tuple(arr.shape) == out_shape:
                specs.append(ElemInput(param_of[vid], scalar=False))
            elif arr.size == 1:
                specs.append(ElemInput(param_of[vid], scalar=True))
            else:
                return None

        n = int(np.prod(out_shape))
        if n == 0:
            return None

        out = cp.empty(out_shape, dtype=dtype)
        ir_dtype = DType.F32 if dtype == cp.float32 else DType.F16
        aligned = int(out.data.ptr) % VEC4_ALIGN == 0 and all(
            int(a.data.ptr) % VEC4_ALIGN == 0 for a in arrays
        )
        launch = plan_elementwise(
            n,
            block=self.block,
            max_blocks=self.max_blocks,
            dtype=ir_dtype.value,
            aligned=aligned,
            target_blocks=self.sm_count,
        )
        vec = launch.vec

        # 缓存键必须覆盖**完整的数据流**：每步的算子和它的参数引用（``p0`` /
        # ``p1`` / ``t0`` 以及顺序）都决定生成的源码。只记 ``(op, 参数个数)``
        # 会让 ``t=x+y; out=t-z`` 与 ``t=x+y; out=z-t`` 撞同一个键——两者
        # 算子序列相同、参数个数相同，但相减方向相反。叶子是否为标量也必须
        # 跟着它在 specs 里的名字一起进键。
        key = (
            "chain",
            tuple((s.out, s.op, s.args) for s in steps),
            tuple((s.name, s.scalar) for s in specs),
            vec,
            self._kind(arrays[0]),
        )
        kernel = self._kernel(
            key,
            lambda: self.codegen.render(
                build_chain(
                    steps, specs, vec=vec, dtype=ir_dtype, kernel_name="ew_chain"
                ),
                dtype,
            ),
            "ew_chain",
        )
        kernel((launch.grid,), (launch.block,), (*arrays, out, np.int64(n)))
        return out

    # ------------------------------------------------------------------
    # 逐元素
    # ------------------------------------------------------------------

    def _op_add(self, node, args, attrs):
        return self._binary("add", args)



    def _op_sub(self, node, args, attrs):
        return self._binary("sub", args)



    def _op_mul(self, node, args, attrs):
        return self._binary("mul", args)



    def _op_div(self, node, args, attrs):
        return self._binary("div", args)



    def _op_clamp(self, node, args, attrs):
        """按属性里的标量上下界截断；缺省的一侧由 ``has_lo`` / ``has_hi`` 关掉。

        标量通过内核参数传入而不是写进源码：同一张图里 ``clamp(min=-80)``
        与 ``clamp(0, 1)`` 很常见，按边界值生成源码会让编译缓存按数值增长。
        """
        cp = self.cp
        x = args[0]
        out = cp.empty_like(x)
        n = int(x.size)
        if n == 0:
            return out
        lo = attrs.get("min")
        hi = attrs.get("max")
        kernel = self._kernel(
            ("clamp", self._kind(x)),
            lambda: self.codegen.render(self.codegen.clamp_source(), x.dtype),
            "ew_clamp",
        )
        block, grid = self._grid("clamp", n)
        kernel(
            (grid,),
            (block,),
            (
                x,
                out,
                np.int64(n),
                np.int32(lo is not None),
                np.int32(hi is not None),
                np.float32(0.0 if lo is None else lo),
                np.float32(0.0 if hi is None else hi),
            ),
        )
        return out



    def _op_mod(self, node, args, attrs):
        """整数取模（floor 语义）。

        只承接同形与标量广播：整数逐元素内核没有做多维 stride 广播的版本，
        而 Engram 的哈希两侧同形、模数是标量。其它广播形态显式拒绝，避免
        落到"按 f32 解释整数缓冲"的错误路径上。
        """
        a, b = args
        try:
            out_shape = tuple(np.broadcast_shapes(a.shape, b.shape))
        except ValueError as exc:
            raise BackendCapabilityError(
                f"{self._LABEL}的 mod 无法广播 {a.shape} 与 {b.shape}"
            ) from exc
        if a.shape == b.shape:
            return self._mod_kernel(a, b, mode=0)
        if a.size == 1 and out_shape == b.shape:
            return self._mod_kernel(a, b, mode=1)
        if b.size == 1 and out_shape == a.shape:
            return self._mod_kernel(a, b, mode=2)
        raise BackendCapabilityError(
            f"{self._LABEL}的 mod 只承接同形与标量广播，收到 {a.shape} 与 {b.shape}"
        )



    def _mod_kernel(self, a, b, mode: int):
        cp = self.cp
        tensor = b if mode == 1 else a
        out = cp.empty_like(tensor)
        n = int(tensor.size)
        if n == 0:
            return out
        kernel = self._kernel(
            ("mod", mode),
            lambda: self.codegen.render(
                self.codegen.mod_source(scalar_a=mode == 1, scalar_b=mode == 2),
                # 整数内核把参数类型钉死成 int，dtype 只为拿到方言 prelude；
                # 必须传 np.dtype 实例（codegen 按 ``.name`` 取 f32 / f16），
                # 传 DType 枚举或 ``np.float32`` 类都会被拒绝。
                np.dtype("float32"),
            ),
            "ew_mod",
        )
        lhs = a.reshape(1) if mode == 1 else a
        rhs = b.reshape(1) if mode == 2 else b
        block, grid = self._grid("mod", n)
        kernel((grid,), (block,), (lhs, rhs, out, np.int64(n)))
        return out



    def _op_relu(self, node, args, attrs):
        return self._unary("relu", args)



    def _op_silu(self, node, args, attrs):
        if len(args) == 1:
            return self._unary("silu", args)
        cp = self.cp
        x, gate = args
        if tuple(x.shape) != tuple(gate.shape) or x.dtype != gate.dtype:
            raise BackendCapabilityError(
                f"silu 的 gate 必须与 x 同形状、同 dtype，收到 {x.shape} 与 {gate.shape}"
            )
        out = cp.empty_like(x)
        n = int(x.size)
        if n == 0:
            return out
        kernel = self._kernel(
            ("silu_gate", self._kind(x)),
            lambda: self.codegen.render(self.codegen.silu_gate_source(), x.dtype),
            "silu_gate",
        )
        block, grid = self._grid("silu", n)
        kernel((grid,), (block,), (x, gate, out, np.int64(n)))
        return out



    def _op_exp(self, node, args, attrs):
        return self._unary("exp", args)



    def _op_log(self, node, args, attrs):
        return self._unary("log", args)



    def _op_sqrt(self, node, args, attrs):
        return self._unary("sqrt", args)



    def _op_neg(self, node, args, attrs):
        return self._unary("neg", args)



    def _op_tanh(self, node, args, attrs):
        return self._unary("tanh", args)



    def _op_sigmoid(self, node, args, attrs):
        return self._unary("sigmoid", args)



    def _op_erf(self, node, args, attrs):
        """误差函数；精确 GELU 的组成部分，走一元模板。"""
        return self._unary("erf", args)



    def _op_gelu(self, node, args, attrs):
        """tanh 近似 GELU，与 ir/ops.py 的参考式逐位同式，走一元模板。"""
        return self._unary("gelu", args)



    def _op_step(self, node, args, attrs):
        return self._step(args, attrs, "step")



    def _op_spike(self, node, args, attrs):
        """前向与 step 逐位相同（都是"越过阈值发 1"）。

        差异只在反向：`spike` 的 vjp 是替代梯度，而反向图由 IR 层的规则组合，
        后端不需要知道这件事。所以这里直接复用同一个 kernel。
        调优键仍需分开：两者是不同的 IR 算子，各自可以被单独调参。
        """
        return self._step(args, attrs, "spike")



    def _step(self, args, attrs, op: str):
        cp = self.cp
        x = args[0]
        out = cp.empty_like(x)
        n = int(x.size)
        if n == 0:
            return out
        kind = self._kind(x)
        kernel = self._kernel(
            ("step", kind), lambda: self.codegen.render(self.codegen.step_source(), x.dtype), "ew_step"
        )
        block, grid = self._grid(op, n)
        kernel(
            (grid,),
            (block,),
            (x, out, np.int64(n), np.float32(float(attrs.get("threshold", 0.0)))),
        )
        return out



    def _op_dropout_mask(self, node, args, attrs):
        """生成缩放后的 dropout 掩码（保留 ``1/(1-p)``，丢弃 0）。

        与参考实现共用 SplitMix64 与"取高 24 位"的映射，所以两边可以逐位比较。
        输入 x 的数值不读，只取形状——它在图里仍是真实的边，这样输出的形状
        能跟着符号 batch 维走，而不必把形状写进 attrs。
        """
        cp = self.cp
        x = args[0]
        p = float(attrs.get("p", 0.5))
        if not 0.0 <= p < 1.0:
            raise BackendCapabilityError(f"dropout 的 p 必须在 [0, 1) 内，收到 {p}")
        out = cp.empty(x.shape, dtype=x.dtype)
        n = int(out.size)
        if n == 0:
            return out
        offset = int(attrs.get("offset", 0)) & 0xFFFFFFFFFFFFFFFF
        kind = self._kind(x)
        if len(args) > 1:
            # 运行期 key：随机源在设备内存里。标量参数会被 CUDA Graph 按值
            # 冻结，捕获后的 replay 就再也换不了掩码——所以 key 必须是指针。
            key = cp.ascontiguousarray(args[1]).reshape(1)
            kernel = self._kernel(
                ("dropout_mask_keyed", kind),
                lambda: self.codegen.render(
                    self.codegen.dropout_mask_keyed_template(), x.dtype
                ),
                "dropout_mask_keyed",
            )
            block, grid = self._grid("dropout_mask", n)
            kernel(
                (grid,),
                (block,),
                (
                    out,
                    np.int64(n),
                    np.float32(p),
                    np.float32(1.0 / (1.0 - p)),
                    key,
                    np.uint64(offset),
                ),
            )
            return out
        seed = (int(attrs.get("seed", 0)) + int(offset)) & 0xFFFFFFFFFFFFFFFF
        kernel = self._kernel(
            ("dropout_mask", kind),
            lambda: self.codegen.render(self.codegen.dropout_mask_source(), x.dtype),
            "dropout_mask",
        )
        block, grid = self._grid("dropout_mask", n)
        kernel(
            (grid,),
            (block,),
            (
                out,
                np.int64(n),
                np.float32(p),
                np.float32(1.0 / (1.0 - p)),
                np.uint64(seed),
            ),
        )
        return out



    def _op_constant(self, node, args, attrs):
        """常量节点：值在属性里，直接物化一份设备张量。

        零输入节点绕过了 ``invoke`` 里对输入 dtype 的检查，所以这里自己查一次，
        保持「GPU 后端只支持 f32/f16」这条规则没有例外。
        """
        dtype = DType(attrs["dtype"])
        if dtype not in (DType.F32, DType.F16):
            raise BackendCapabilityError(
                f"{self._LABEL}只支持 float32/float16，constant 收到 {dtype.value}"
            )
        np_dtype = np.float16 if dtype is DType.F16 else np.float32
        shape = tuple(int(d) for d in attrs["shape"])
        data = attrs.get("data")
        if data is None:
            return self.cp.full(shape, np_dtype(attrs["value"]), dtype=np_dtype)
        # 常量折叠的产物是任意张量，不一定是同值填充；先按主机 numpy 构造
        # 再整块拷到设备，避免在 GPU 上逐元素赋值。
        host = np.asarray(data, dtype=np_dtype).reshape(shape)
        return self.cp.array(host, dtype=np_dtype)



    def _op_broadcast_to(self, node, args, attrs):
        """显式广播。物化成连续内存：后续 kernel 都要求输入连续。"""
        target = tuple(int(d) for d in attrs["shape"])
        out = self.cp.ascontiguousarray(self.cp.broadcast_to(args[0], target))
        # 目标形状与输入相同时广播退化成恒等，拿到的是同一个对象或视图
        return self._shared_view(out, args[0])



    def _binary(self, kind: str, args: List[Any]):
        a, b = args
        try:
            out_shape = tuple(np.broadcast_shapes(a.shape, b.shape))
        except ValueError as exc:
            raise BackendCapabilityError(
                f"无法广播的二元运算：{kind} 收到 {a.shape} 与 {b.shape}"
            ) from exc

        if a.shape == b.shape:
            return self._binary_kernel(kind, a, b, mode=0)
        # 标量快速路径：只有当广播结果形状恰好等于另一侧形状时才能直接用
        # （例如 (5,) 与 (1,1) 的广播结果是 (1,5)，不能走这条路径）
        if a.size == 1 and out_shape == b.shape:
            return self._binary_kernel(kind, a, b, mode=1)  # a 是标量
        if b.size == 1 and out_shape == a.shape:
            return self._binary_kernel(kind, a, b, mode=2)  # b 是标量
        return self._binary_broadcast(kind, a, b, out_shape)



    def _binary_kernel(self, kind: str, a, b, mode: int):
        cp = self.cp
        tensor = b if mode == 1 else a
        out = cp.empty_like(tensor)
        n = int(tensor.size)
        if n == 0:
            return out
        scalar_a, scalar_b = mode == 1, mode == 2
        dtype_key = self._kind(tensor)
        kernel = self._kernel(
            ("bin", kind, dtype_key, mode),
            lambda: self.codegen.render(
                self.codegen.binary_source(kind, scalar_a, scalar_b), tensor.dtype
            ),
            "ew_bin",
        )
        # 标量侧直接传设备指针，不在主机上取值：避免一次强制同步（见 codegen 注释）
        lhs = a.reshape(1) if scalar_a else a
        rhs = b.reshape(1) if scalar_b else b
        block, grid = self._grid(kind, n)
        kernel((grid,), (block,), (lhs, rhs, out, np.int64(n)))
        return out



    def _binary_broadcast(self, kind: str, a, b, out_shape: tuple):
        """多维广播：把两侧 stride 对齐到输出形状，广播维 stride 置 0。

        广播的维度会被折叠（相邻且两侧 stride 都连续的维合并成一个），
        通常 (B,S,D) 与 (D,) 折叠后只剩 2 维。折叠后仍超过 4 维则明确拒绝。
        """
        cp = self.cp
        a = self._contig(a)
        b = self._contig(b)
        out = cp.empty(out_shape, dtype=a.dtype)
        n = int(out.size)
        if n == 0:
            return out

        groups = self._collapse_dims(
            out_shape,
            self._elem_strides(a.shape, out_shape),
            self._elem_strides(b.shape, out_shape),
        )
        if len(groups) > 4:
            raise BackendCapabilityError(
                f"{self._LABEL}的广播最多支持 4 个不可折叠维度，"
                f"{kind} 收到 {a.shape} 与 {b.shape}（折叠后 {len(groups)} 维）"
            )

        dims = [1, 1, 1, 1]
        sa = [0, 0, 0, 0]
        sb = [0, 0, 0, 0]
        for i, (dim, step_a, step_b) in enumerate(groups):
            dims[i], sa[i], sb[i] = dim, step_a, step_b

        dtype_key = self._kind(a)
        kernel = self._kernel(
            ("bcast", kind, dtype_key),
            lambda: self.codegen.render(self.codegen.broadcast_binary_source(kind), a.dtype),
            "ew_bcast",
        )
        block, grid = self._grid(kind, n)
        kernel(
            (grid,),
            (block,),
            (
                a,
                b,
                out,
                np.int64(n),
                np.int32(len(groups)),
                *[np.int64(v) for v in dims],
                *[np.int64(v) for v in sa],
                *[np.int64(v) for v in sb],
            ),
        )
        return out



    def _unary(self, kind: str, args: List[Any]):
        cp = self.cp
        x = args[0]
        out = cp.empty_like(x)
        n = int(x.size)
        if n == 0:
            return out
        dtype_key = self._kind(x)
        kernel = self._kernel(
            ("un", kind, dtype_key),
            lambda: self.codegen.render(self.codegen.unary_source(kind), x.dtype),
            "ew_un",
        )
        block, grid = self._grid(kind, n)
        kernel((grid,), (block,), (x, out, np.int64(n)))
        return out

    # ------------------------------------------------------------------
    # 矩阵
    # ------------------------------------------------------------------

    def _op_matmul(self, node, args, attrs):
        cp = self.cp
        a, b = args
        if a.ndim < 2 or b.ndim < 2:
            raise BackendCapabilityError(
                f"{self._LABEL}的 matmul 需要秩 >= 2，收到 {a.shape} 与 {b.shape}"
            )
        a = self._contig(a)
        b = self._contig(b)
        if a.dtype != b.dtype:
            raise BackendCapabilityError(
                f"matmul 两侧 dtype 必须一致，收到 {a.dtype} 与 {b.dtype}；"
                "请显式插入 cast"
            )
        m, k = int(a.shape[-2]), int(a.shape[-1])
        k2, n = int(b.shape[-2]), int(b.shape[-1])
        if k != k2:
            raise BackendCapabilityError(
                f"matmul 内维不匹配：{a.shape} @ {b.shape}"
            )
        try:
            batch_shape = tuple(np.broadcast_shapes(a.shape[:-2], b.shape[:-2]))
        except ValueError as exc:
            raise BackendCapabilityError(
                f"matmul 的批量维无法广播：{a.shape} @ {b.shape}"
            ) from exc

        out = cp.empty(batch_shape + (m, n), dtype=a.dtype)
        if out.size == 0:
            return out

        blas = self._cublas()
        blas.set_stream(self.current_stream_ptr())

        # f16 走 GemmEx（f32 累加），f32 走 Sgemm：两条路径的累加精度一致，
        # 差别只在输入输出的存储宽度。
        f16 = a.dtype == cp.float16
        if not batch_shape:
            if f16:
                blas.hgemm(a, b, out)
            else:
                blas.sgemm(a, b, out)
            return out

        batch = int(np.prod(batch_shape))
        stride_a = self._batch_stride(a.shape[:-2], batch_shape, m * k, "左操作数")
        stride_b = self._batch_stride(b.shape[:-2], batch_shape, k * n, "右操作数")
        if f16:
            blas.hgemm_batched(a, b, out, batch, stride_a, stride_b, m * n)
        else:
            blas.sgemm_batched(a, b, out, batch, stride_a, stride_b, m * n)
        return out

    @staticmethod
    def _batch_stride(shape: tuple, out_shape: tuple, dense: int, what: str) -> int:
        """批量 GEMM 的步长：形状一致用密集步长，整体只有 1 个 batch 时用 0。

        部分广播（例如 (2,1) 对 (2,3)）无法用一个标量 stride 表达，
        明确拒绝而不是悄悄算错。
        """
        if shape == out_shape:
            return dense
        if not shape or all(d == 1 for d in shape):
            return 0
        raise BackendCapabilityError(
            f"matmul {what}的批量维 {shape} 与输出批量维 {out_shape} 不匹配："
            "cuBLAS 的 strided batched 只能表达“完全一致”或“整体为 1”"
        )

    # ------------------------------------------------------------------
    # 量化（weight-only int8）
    # ------------------------------------------------------------------

    def _quant_scale_arg(self, scale, shape):
        """把 scale 变成内核能吃的连续 f32 数组。

        返回 ``(数组, cols, per_row)``。``per_row = 1`` 时内核按
        ``s[i % cols]`` 取缩放系数，这样逐输出通道量化**不会**把 scale 物化成
        与权重同大的 f32（那会让"省 3/4 权重"变成"多 1 倍临时量"）。
        其余可广播形态（逐张量、或与权重同形）走广播拷贝；IR 已经保证
        scale 能广播到权重，这里只是选择用哪条读法。
        """
        if not scale.flags.c_contiguous:
            scale = self.cp.ascontiguousarray(scale)
        if scale.ndim == 1 and int(scale.size) == int(shape[-1]):
            return scale, int(shape[-1]), 1
        if tuple(scale.shape) == tuple(shape):
            return scale, 0, 0
        return (
            self.cp.ascontiguousarray(self.cp.broadcast_to(scale, shape)),
            0,
            0,
        )

    def _op_quantize_i8(self, node, args, attrs):
        """对称 int8 量化：``rintf(w / s)`` 后饱和到 [-127, 127]。

        用一段自研内核而不是 cupy 的逐元素原语：CUDA 侧的 ``cp`` 是 cupy，
        HIP 侧是同一套接口的自家实现，而 HIP 那份没有 ``rint`` / ``clip``。
        半偶舍入是语义的一部分，不能在两个后端各写一次近似——所以两边共用
        这一份源码（NVRTC 与 hipRTC 都直接吃它）。

        非正 scale 的检查走 ``to_host``：scale 只有 N 个元素，代价可忽略；
        量化是离线步骤（导出权重时跑一次），不在每步推理的热路径上。
        """
        cp = self.cp
        weight, scale = args
        if weight.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 quantize_i8 只接受 f32 权重，收到 {weight.dtype}"
            )
        if weight.ndim < 1:
            raise BackendCapabilityError(
                f"{self._LABEL}的 quantize_i8 要求权重至少 1 维，收到 {weight.shape}"
            )
        if float(np.min(self.to_host(scale))) <= 0.0:
            raise BackendCapabilityError(
                "quantize_i8 的 scale 必须严格为正；非正 scale 的量化为未定义"
            )
        w = self._contig(weight)
        s, cols, per_row = self._quant_scale_arg(scale, w.shape)
        out = cp.empty(w.shape, np.int8)
        n = int(w.size)
        if n == 0:
            return out
        kernel = self._kernel(("quantize_i8", per_row), _QUANTIZE_I8_SRC, "af_quantize_i8")
        block, grid = self._grid("quantize_i8", n)
        kernel(
            (grid,),
            (block,),
            (w, s, out, np.int64(n), np.int64(cols), np.int64(per_row)),
        )
        return out

    def _op_dequantize_i8(self, node, args, attrs):
        """i8 -> f32 的放大。输出必须连续：``linear_i8`` 要把它交给 GEMM。"""
        cp = self.cp
        q, scale = args
        if q.dtype != cp.int8:
            raise BackendCapabilityError(
                f"{self._LABEL}的 dequantize_i8 第一个输入必须是 i8，收到 {q.dtype}"
            )
        if scale.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 dequantize_i8 的 scale 必须是 f32，收到 {scale.dtype}"
            )
        qq = self._contig(q)
        s, cols, per_row = self._quant_scale_arg(scale, qq.shape)
        out = cp.empty(qq.shape, np.float32)
        n = int(qq.size)
        if n == 0:
            return out
        kernel = self._kernel(
            ("dequantize_i8", per_row), _DEQUANTIZE_I8_SRC, "af_dequantize_i8"
        )
        block, grid = self._grid("dequantize_i8", n)
        kernel(
            (grid,),
            (block,),
            (qq, s, out, np.int64(n), np.int64(cols), np.int64(per_row)),
        )
        return out

    def _op_linear_i8(self, node, args, attrs):
        """weight-only int8 线性层：反量化权重后走既有 f32 GEMM。

        "int8" 在这里只体现在**权重驻留与读带宽**上：读出 i8、就地放大成
        f32、再交给 cuBLAS / hipBLAS。这就是 weight-only 量化的全部收益，
        不宣称算力提升——真正的 int8 矩阵乘需要另一套累加语义。
        """
        cp = self.cp
        x, qweight, scale = args[0], args[1], args[2]
        if x.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_i8 要求 x 是 f32，收到 {x.dtype}"
            )
        if x.ndim < 2 or qweight.ndim != 2:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_i8 要求 x 秩 >= 2、权重秩 2，"
                f"收到 {x.shape} 与 {qweight.shape}"
            )
        weight = self._op_dequantize_i8(node, [qweight, scale], attrs)
        out = self._op_matmul(node, [x, weight], {})
        if len(args) > 3:
            out = self._binary("add", [out, args[3]])
        return out

    # ------------------------------------------------------------------
    # 卷积：分块直算 / im2col + cuBLAS
    # ------------------------------------------------------------------

    def _op_conv2d(self, node, args, attrs):
        """NCHW conv2d，两条实现路径语义完全一致（交叉相关、不翻转核）。

        - 分块直算（``_conv2d_tiled``）：不再物化列矩阵，权重缓存在 shared
          memory 里按行复用，适合 K 较小、输出较大的常见 CNN 形状。
        - im2col + GEMM（``_conv2d_im2col``）：把卷积变成一次批量 GEMM，
          K 很大时 GEMM 更吃得饱，也是分块核放不下时的兜底。

        具体走哪条由 ``_pick_conv_impl`` 决定；选择依据写在那里，不散落在此。
        """
        cp = self.cp
        x, weight = args[0], args[1]
        bias = args[2] if len(args) > 2 else None
        if x.ndim != 4 or weight.ndim != 4:
            raise BackendCapabilityError(
                f"{self._LABEL}的 conv2d 要求 x/weight 都是 NCHW 秩 4，收到 {x.shape} 与 {weight.shape}"
            )
        groups = int(attrs.get("groups", 1))
        if groups < 1:
            raise BackendCapabilityError(f"conv2d 的 groups 必须 >= 1，得到 {groups}")

        sh, sw = _pair(attrs.get("stride", (1, 1)))
        ph, pw = _pair(attrs.get("padding", (0, 0)))
        dh, dw = _pair(attrs.get("dilation", (1, 1)))

        x = self._contig(x)
        weight = self._contig(weight)
        n, c, h, w = (int(v) for v in x.shape)
        o, cw, kh, kw = (int(v) for v in weight.shape)
        if c % groups != 0 or o % groups != 0:
            raise BackendCapabilityError(
                f"conv2d 的通道数必须能被 groups={groups} 整除，收到 C={c}、O={o}"
            )
        if cw * groups != c:
            raise BackendCapabilityError(
                f"conv2d 输入通道 {c} 与权重通道 {cw} * groups={groups} 不一致"
            )

        ho = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        wo = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        if ho <= 0 or wo <= 0:
            raise BackendCapabilityError(
                f"conv2d 输出尺寸非法：输入 {h}x{w}、核 {kh}x{kw}、"
                f"padding {(ph, pw)}、stride {(sh, sw)}、dilation {(dh, dw)}"
            )
        if bias is not None and tuple(bias.shape) != (o,):
            raise BackendCapabilityError(
                f"conv2d 的 bias 必须是 ({o},)，收到 {bias.shape}"
            )

        cg, og = c // groups, o // groups
        k = cg * kh * kw
        if n * o * ho * wo == 0:
            return cp.empty((n, o, ho, wo), dtype=x.dtype)

        dims = dict(
            n=n, c=cg, h=h, w=w, o=og, kh=kh, kw=kw, ho=ho, wo=wo,
            sh=sh, sw=sw, ph=ph, pw=pw, dh=dh, dw=dw, k=k,
        )
        # auto 模式下能加载 cuDNN 就优先用它：同一形状下卷积本体比自研分块核
        # 快数倍（5.69 的实测）。加载不到或这一步失败都回落到下面的自研路径。
        if self._pick_cudnn(x.dtype, groups):
            fast = self._conv2d_cudnn(x, weight, bias, **dims)
            if fast is not None:
                return fast
        if x.dtype == cp.float16:
            # 分块核与隐式 GEMM 核的 shared memory 与累加器都是 float，只覆盖
            # f32 路径。f16 走 im2col + hgemm：GEMM 内部同样是 f32 累加，
            # 语义不变。
            if self.conv_impl in ("tiled", "implicit"):
                raise BackendCapabilityError(
                    f"conv2d 的 {self.conv_impl} 内核只支持 float32；"
                    "f16 请用 conv_impl='auto' 或 'im2col'"
                )
            kernel_kind = "im2col"
        elif groups > 1:
            # 分组卷积的"每个输出通道看全部输入通道"不成立，分块核与隐式
            # 核的 shared staging 都不适用（显式要求时明确失败）。
            #
            # auto 只在 depthwise（每组恰好一个输入通道）时走 5.165 的
            # 直算核：这个形状在 CUDA 与 HIP 的全部实测形状上都更快（省掉
            # 列矩阵，也不再有极瘦 GEMM）。非 depthwise 分组不赌——5.171 在
            # M40 上测到 cuBLAS 的 im2col 路径反超（g2 0.47×、g4 0.81×），
            # 而 HIP 上同一形状是直算核更快；跨厂商没有统一结论，想省列矩阵
            # 就显式指定 conv_impl="grouped"。im2col 仍是默认回落路径。
            if self.conv_impl in ("tiled", "implicit"):
                raise BackendCapabilityError(
                    f"conv2d 的 {self.conv_impl} 内核只支持 groups=1；"
                    "分组卷积请用 conv_impl='auto'、'grouped' 或 'im2col'"
                )
            if self.conv_impl == "grouped" or (
                self.conv_impl == "auto" and groups == c
            ):
                kernel_kind = "grouped"
            else:
                kernel_kind = "im2col"
        else:
            kernel_kind = self._pick_conv_kernel(**dims)
        bias_fused = False
        if kernel_kind == "grouped":
            # 分组直算在写回时融合 bias，与分块 / 隐式核一致
            out = self._conv2d_grouped(x, weight, bias, groups=groups, **dims)
            bias_fused = True
        elif groups == 1:
            if kernel_kind == "tiled":
                # 分块核在写回时就地把 bias 加进累加器，省掉一次 broadcast 核
                out = self._conv2d_tiled(x, weight, bias, **dims)
                bias_fused = True
            elif kernel_kind == "implicit":
                # 隐式 GEMM 同样在写回时加 bias，理由与分块核一致
                out = self._conv2d_implicit(x, weight, bias, **dims)
                bias_fused = True
            else:
                out = self._conv2d_im2col(x, weight, **dims)
        else:
            # 逐组跑 im2col + GEMM，再沿输出通道拼回。depthwise
            # （groups == C == O）走的就是这条路径：正确性优先，
            # 没有为它单独做逐通道向量化内核（性能未优化，见 DESIGN）。
            out = cp.empty((n, o, ho, wo), dtype=x.dtype)
            for gi in range(groups):
                xs = cp.ascontiguousarray(x[:, gi * cg : (gi + 1) * cg])
                ws = weight[gi * og : (gi + 1) * og]
                out[:, gi * og : (gi + 1) * og] = self._conv2d_im2col(xs, ws, **dims)

        if bias is not None and not bias_fused:
            out = self._binary("add", [out, bias.reshape(1, o, 1, 1)])
        return out



    def _use_tiled_conv(self, *, n, o, k, kh, kw, ho, wo, **_ignored) -> bool:
        """是否走分块核。

        三条规则，每条都对应一个具体的失败模式：

        - ``conv_impl="im2col"``：人工强制，用于对照实验与排障。
        - K 很大（深通道 / 大核）或输出总量太小：判据在 ``plan.conv_tiled_preferred``，
          数字来自 DESIGN 5.16 的实测。
        - shared 预算放不下一个输出通道：明确报错，不静默换实现。

        强制 ``conv_impl="tiled"`` 时跳过规模判断，只校验 shared 预算。
        """
        if self.conv_impl == "im2col":
            return False
        if self.conv_impl == "auto" and not conv_tiled_preferred(
            k, n * o * ho * wo
        ):
            return False
        if conv_oc_tile(kh * kw, o) is None:
            raise BackendCapabilityError(
                f"conv2d 分块核的 shared memory 预算放不下一个输出通道"
                f"（核 {kh}x{kw} 需要 {kh * kw * 4} B，预算 {CONV_SHARED_LIMIT} B）；"
                "请用 conv_impl='auto' 或 'im2col'"
            )
        return True



    def _pick_conv_kernel(self, *, n, o, k, kh, kw, ho, wo, c, sw, dw,
                          **_ignored) -> str:
        """在 tiled / implicit / im2col 之间选一个实现。

        ``auto`` 只在分块直算的甜点区选 tiled，其余交给 im2col。隐式 GEMM
        **不进 CUDA 的 auto**：5.151 在 M40 上做了 7 个形状的同形状 A/B（K 从
        27 到 4608），K >= 288 的 6 个形状全部输给 im2col（cuBLAS 分块 GEMM），
        慢 1.37×（K=288）到 2.32×（K=864）；唯一的例外是 K=27，那一档分块直算
        本来就更快（1.69 ms vs 隐式 1.80 ms），轮不到隐式接管。它仍是显式
        ``conv_impl="implicit"`` 的对照路径（数值与 im2col 一致，相对误差
        ≤ 1.7e-6），但不该由"规模大"自动触发，否则等于用生产流量做实验。

        HIP 的结论相反（5.238 实测：中等通道 3x3/5x5 上隐式比 im2col 快
        1.33～1.63×，3x3_c128 快 1.33×），由子类的 ``CONV_AUTO_IMPLICIT``
        打开。两条额外限制都来自实测：``kh*kw > 1``（1x1 没有窗口重叠，
        implicit 反而慢 6%）、``k > CONV_TILED_MAX_K``（小 K 的分块直算仍更快）。
        """
        if self.conv_impl == "implicit":
            return "implicit"
        if (
            self.conv_impl == "auto"
            and self.CONV_AUTO_IMPLICIT
            and kh * kw > 1
            and k > CONV_TILED_MAX_K
            and conv_implicit_fits(
                kh=kh, kw=kw, c=c, o=o, wo=wo, sw=sw, dw=dw,
                tile_w=self._conv_tile,
            )
        ):
            return "implicit"
        if self._use_tiled_conv(n=n, o=o, k=k, kh=kh, kw=kw, ho=ho, wo=wo):
            return "tiled"
        return "im2col"

    def _conv2d_implicit(self, x, weight, bias, *, n, c, h, w, o, kh, kw, ho, wo,
                         sh, sw, ph, pw, dh, dw, k):
        """隐式 GEMM 卷积：输入窗口与权重一起 staging 进 shared。

        与 :meth:`_conv2d_tiled` 的差别只在 shared 的用法（见
        ``lower/kernels.py::conv2d_implicit``）与多传一个 ``SPAN``。线程铺法
        来自 ``plan.plan_conv_implicit``：它把权重与输入窗口两块预算一起算，
        只按权重挑 ``toc`` 会在宽核上把 shared 撑爆。

        ``bias`` 的处理与分块核一致：传一个有效指针占位，由 ``has_bias`` 决定
        是否读它，写回时就地累加。
        """
        cp = self.cp
        tiling = plan_conv_implicit(
            kh=kh, kw=kw, c=c, o=o, wo=wo, sw=sw, dw=dw,
            tile_w=self._conv_tile, roc=self._conv_roc, cb=self._conv_cb,
        )

        out = cp.empty((n, o, ho, wo), dtype=x.dtype)
        kernel = self._kernel(
            ("conv2d_implicit", self._kind(x)),
            lambda: self.codegen.render(self.codegen.conv2d_implicit_source(), x.dtype),
            "conv2d_implicit",
        )
        bp = x if bias is None else self._contig(bias)
        kernel(
            (cdiv(wo, tiling.tile_w), ho, n * tiling.oc_tiles),
            tiling.block,
            (
                x, weight, out, bp,
                np.int32(c), np.int32(h), np.int32(w),
                np.int32(o), np.int32(kh), np.int32(kw),
                np.int32(ho), np.int32(wo),
                np.int32(sh), np.int32(sw), np.int32(ph), np.int32(pw),
                np.int32(dh), np.int32(dw),
                np.int32(tiling.toc), np.int32(tiling.roc), np.int32(tiling.cb),
                np.int32(tiling.oc_tiles),
                np.int32(0 if bias is None else 1),
                np.int32(tiling.span),
            ),
            shared_mem=tiling.shared_bytes,
        )
        return out

    def _conv2d_grouped(self, x, weight, bias, *, n, c, h, w, o, kh, kw, ho, wo,
                        sh, sw, ph, pw, dh, dw, k, groups):
        """分组 / depthwise 直算：一个线程一个输出元素（5.165）。

        ``c`` / ``o`` 是**组内**的输入 / 输出通道数（``C/groups`` 与
        ``O/groups``），与其它卷积入口的 ``dims`` 同义；内核自己用
        ``GROUPS`` 还原总通道数。

        不物化列矩阵、也不按组分次 launch：一次内核覆盖全部组，输出直接
        按 ``(n, o, ho, wo)`` 写回。f16 不走这里（内核全程 float），
        调用方已在派发处把它留给 im2col。
        """
        cp = self.cp
        total_out = o * groups
        out = cp.empty((n, total_out, ho, wo), dtype=x.dtype)
        total = int(out.size)
        block, grid = self._grid("conv2d", total)
        kernel = self._kernel(
            ("conv2d_grouped", self._kind(x)),
            lambda: self.codegen.render(
                self.codegen.conv2d_grouped_source(), x.dtype
            ),
            "conv2d_grouped",
        )
        bp = x if bias is None else self._contig(bias)
        kernel(
            (grid,),
            (block,),
            (
                x, weight, out, bp,
                np.int32(total_out), np.int32(h), np.int32(w),
                np.int32(kh), np.int32(kw), np.int32(ho), np.int32(wo),
                np.int32(sh), np.int32(sw), np.int32(ph), np.int32(pw),
                np.int32(dh), np.int32(dw),
                np.int32(c), np.int32(o), np.int32(groups),
                np.int32(0 if bias is None else 1),
                np.int64(total),
            ),
        )
        return out

    def _conv2d_tiled(self, x, weight, bias, *, n, c, h, w, o, kh, kw, ho, wo,
                      sh, sw, ph, pw, dh, dw, k):
        """分块直算：一个线程算 ROC 个输出通道，权重按输入通道流式 staging。

        grid = (W 方向 tile, HO, N * OC tile)，block = (tile_w, toc / ROC)。
        显存只多出输出本身，不再有 ``(N, K, HO*WO)`` 的列矩阵。

        线程铺法全部来自 ``plan.plan_conv_tiled``：那里把一个 block 的权重
        tile、ROC、CB 与 shared 预算一起定下来，这里只负责开缓冲与 launch。
        三个旋钮（``tile_w`` / ``roc`` / ``cb``）都可以被调优覆盖，为 ``None``
        的项仍由计划推导。

        ``bias`` 可以传 ``None``：内核只用 ``has_bias`` 决定要不要读那个指针，
        所以这里传一个有效的已有指针占位即可，不需要额外分配。
        """
        cp = self.cp
        tiling = plan_conv_tiled(
            kh=kh, kw=kw, c=c, o=o, ho=ho, wo=wo,
            tile_w=self._conv_tile, roc=self._conv_roc, cb=self._conv_cb,
        )

        out = cp.empty((n, o, ho, wo), dtype=x.dtype)
        kernel = self._kernel(
            ("conv2d_tiled", self._kind(x)),
            lambda: self.codegen.render(self.codegen.conv2d_tiled_source(), x.dtype),
            "conv2d_tiled",
        )
        bp = x if bias is None else self._contig(bias)
        kernel(
            (cdiv(wo, tiling.tile_w), ho, n * tiling.oc_tiles),
            tiling.block,
            (
                x, weight, out, bp,
                np.int32(c), np.int32(h), np.int32(w),
                np.int32(o), np.int32(kh), np.int32(kw),
                np.int32(ho), np.int32(wo),
                np.int32(sh), np.int32(sw), np.int32(ph), np.int32(pw),
                np.int32(dh), np.int32(dw),
                np.int32(tiling.toc), np.int32(tiling.roc), np.int32(tiling.cb),
                np.int32(tiling.oc_tiles),
                np.int32(0 if bias is None else 1),
            ),
            shared_mem=tiling.shared_bytes,
        )
        return out



    def _conv2d_im2col(self, x, weight, *, n, c, h, w, o, kh, kw, ho, wo,
                       sh, sw, ph, pw, dh, dw, k):
        """im2col 展开 + 逐 batch 的 GEMM。

        布局是刻意的：列矩阵按 **batch 分段** 成 ``(N, K, HO*WO)``，
        权重 ``(O, K)`` 在所有 batch 间共用（步长 0），于是 GEMM 输出就是
        ``(N, O, HO*WO)``，直接 reshape 成 NCHW，不需要 col2im 或转置。
        """
        cp = self.cp
        out = cp.empty((n, o, ho * wo), dtype=x.dtype)
        cols = cp.empty((n, k, ho * wo), dtype=x.dtype)
        total = int(cols.size)
        dtype_key = self._kind(x)
        kernel = self._kernel(
            ("im2col", dtype_key),
            lambda: self.codegen.render(self.codegen.im2col_source(), x.dtype),
            "im2col",
        )
        block, grid = self._grid("conv2d", total)
        kernel(
            (grid,),
            (block,),
            (
                x,
                cols,
                np.int32(c), np.int32(h), np.int32(w),
                np.int32(kh), np.int32(kw), np.int32(ho), np.int32(wo),
                np.int32(sh), np.int32(sw), np.int32(ph), np.int32(pw),
                np.int32(dh), np.int32(dw),
                np.int64(total),
            ),
        )

        blas = self._cublas()
        blas.set_stream(self.current_stream_ptr())
        # stride_a=0：同一份权重服务所有 batch
        if x.dtype == cp.float16:
            blas.hgemm_batched(
                weight.reshape(o, k), cols, out, n, 0, k * ho * wo, o * ho * wo
            )
        else:
            blas.sgemm_batched(
                weight.reshape(o, k), cols, out, n, 0, k * ho * wo, o * ho * wo
            )
        return out.reshape((n, o, ho, wo))



    def _op_maxpool2d(self, node, args, attrs):
        """NCHW 最大池化：一个输出元素一个线程，纯索引运算。

        与参考语义一致，padding 区域不参与取 max。空窗口填 ``-INFINITY``，
        而不是 0——否则负值输入会在边界处被"抬"到 0。
        """
        cp = self.cp
        x = self._contig(args[0])
        if x.ndim != 4:
            raise BackendCapabilityError(
                f"{self._LABEL}的 maxpool2d 要求 NCHW 秩 4，收到 {x.shape}"
            )
        if "kernel" not in attrs:
            raise BackendCapabilityError(f"{self._LABEL}的 maxpool2d 缺少 kernel 属性")

        kh, kw = _pair(attrs["kernel"])
        raw_stride = attrs.get("stride")
        sh, sw = _pair(attrs["kernel"] if raw_stride is None else raw_stride)
        ph, pw = _pair(attrs.get("padding", (0, 0)))
        if kh < 1 or kw < 1 or sh < 1 or sw < 1:
            raise BackendCapabilityError(
                f"maxpool2d 的 kernel/stride 必须 >= 1，得到 {(kh, kw)} 与 {(sh, sw)}"
            )
        if ph < 0 or pw < 0:
            raise BackendCapabilityError(f"maxpool2d 的 padding 必须 >= 0，得到 {(ph, pw)}")

        n, c, h, w = (int(v) for v in x.shape)
        ceil_mode, _ = _pool_flags(attrs, "maxpool2d")
        ho = _pool_out_dim(h, kh, sh, ph, ceil_mode)
        wo = _pool_out_dim(w, kw, sw, pw, ceil_mode)
        if ho <= 0 or wo <= 0:
            raise BackendCapabilityError(
                f"maxpool2d 输出尺寸非法：输入 {h}x{w}、核 {kh}x{kw}、"
                f"padding {(ph, pw)}、stride {(sh, sw)}、ceil_mode={ceil_mode}"
            )

        out = cp.empty((n, c, ho, wo), dtype=x.dtype)
        total = int(out.size)
        if total == 0:
            return out

        dtype_key = self._kind(x)
        kernel = self._kernel(
            ("maxpool2d", dtype_key),
            lambda: self.codegen.render(self.codegen.maxpool2d_source(), x.dtype),
            "maxpool2d",
        )
        block, grid = self._grid("maxpool2d", total)
        kernel(
            (grid,),
            (block,),
            (
                x,
                out,
                np.int32(c), np.int32(h), np.int32(w),
                np.int32(kh), np.int32(kw), np.int32(ho), np.int32(wo),
                np.int32(sh), np.int32(sw), np.int32(ph), np.int32(pw),
                np.int64(total),
            ),
        )
        return out



    def _op_maxpool2d_indices(self, node, args, attrs):
        """最大池化的取值位置：i32，平面内展平下标 ``ih * W + iw``。

        与 ``maxpool2d`` 共用窗口几何、与 ``maxpool2d_backward`` 共用定位规则
        （行优先、严格大于才更新，所以并列最大值选中同一个元素）。输出固定
        int32——索引不是激活值，f16 输入也不改变这一点。
        """
        cp = self.cp
        x = self._contig(args[0])
        if x.ndim != 4:
            raise BackendCapabilityError(
                f"{self._LABEL}的 maxpool2d_indices 要求 NCHW 秩 4，收到 {x.shape}"
            )
        (kh, kw), (sh, sw), (ph, pw) = self._pool_geometry(attrs, "maxpool2d_indices")
        ceil_mode, _ = _pool_flags(attrs, "maxpool2d_indices")
        n, c, h, w = (int(v) for v in x.shape)
        ho = _pool_out_dim(h, kh, sh, ph, ceil_mode)
        wo = _pool_out_dim(w, kw, sw, pw, ceil_mode)
        if ho <= 0 or wo <= 0:
            raise BackendCapabilityError(
                f"maxpool2d_indices 输出尺寸非法：输入 {h}x{w}、核 {kh}x{kw}、"
                f"padding {(ph, pw)}、stride {(sh, sw)}、ceil_mode={ceil_mode}"
            )

        out = cp.empty((n, c, ho, wo), dtype=cp.int32)
        total = int(out.size)
        if total == 0:
            return out

        dtype_key = self._kind(x)
        kernel = self._kernel(
            ("maxpool2d_indices", dtype_key),
            lambda: self.codegen.render(
                self.codegen.maxpool2d_indices_source(), x.dtype
            ),
            "maxpool2d_indices",
        )
        block, grid = self._grid("maxpool2d_indices", total)
        kernel(
            (grid,),
            (block,),
            (
                x,
                out,
                np.int32(c), np.int32(h), np.int32(w),
                np.int32(kh), np.int32(kw), np.int32(ho), np.int32(wo),
                np.int32(sh), np.int32(sw), np.int32(ph), np.int32(pw),
                np.int64(total),
            ),
        )
        return out

    def _op_avgpool2d(self, node, args, attrs):
        """NCHW 平均池化：与 maxpool2d 同一套索引，差别在"累加并按分母除"。

        分母是窗口落在 padded 输入里的格数，由内核按 ``count_include_pad``
        与 ``ceil_mode`` 现算（见 ``lower/kernels.py::avgpool2d``）——``ceil``
        的最后一格会越过 padded 输入，固定除以 ``kh * kw`` 会让角落偏小。
        """
        cp = self.cp
        x = self._contig(args[0])
        if x.ndim != 4:
            raise BackendCapabilityError(
                f"{self._LABEL}的 avgpool2d 要求 NCHW 秩 4，收到 {x.shape}"
            )
        (kh, kw), (sh, sw), (ph, pw) = self._pool_geometry(attrs, "avgpool2d")
        ceil_mode, count_include_pad = _pool_flags(attrs, "avgpool2d")

        n, c, h, w = (int(v) for v in x.shape)
        ho = _pool_out_dim(h, kh, sh, ph, ceil_mode)
        wo = _pool_out_dim(w, kw, sw, pw, ceil_mode)
        if ho <= 0 or wo <= 0:
            raise BackendCapabilityError(
                f"avgpool2d 输出尺寸非法：输入 {h}x{w}、核 {kh}x{kw}、"
                f"padding {(ph, pw)}、stride {(sh, sw)}、ceil_mode={ceil_mode}"
            )

        out = cp.empty((n, c, ho, wo), dtype=x.dtype)
        total = int(out.size)
        if total == 0:
            return out

        dtype_key = self._kind(x)
        kernel = self._kernel(
            ("avgpool2d", dtype_key),
            lambda: self.codegen.render(self.codegen.avgpool2d_source(), x.dtype),
            "avgpool2d",
        )
        block, grid = self._grid("avgpool2d", total)
        kernel(
            (grid,),
            (block,),
            (
                x,
                out,
                np.int32(c), np.int32(h), np.int32(w),
                np.int32(kh), np.int32(kw), np.int32(ho), np.int32(wo),
                np.int32(sh), np.int32(sw), np.int32(ph), np.int32(pw),
                np.int64(total), np.int32(1 if count_include_pad else 0),
            ),
        )
        return out



    def _op_avgpool2d_backward(self, node, args, attrs):
        """平均池化反向：``dy/分母`` 散射回真实输入。

        窗口重叠时同一个输入格会被多个输出格写，因此必须 atomicAdd；
        输出先清零，且**不能用空内核**——atomicAdd 的初值就是 0。分母与前向
        逐格一致（内含 ``ceil_mode`` 越界与 ``count_include_pad``），padding
        那一侧只参与分母，不接收梯度。
        """
        cp = self.cp
        dy = self._contig(args[0])
        shape = tuple(int(d) for d in attrs["input_shape"])
        if len(shape) != 4:
            raise BackendCapabilityError(
                f"{self._LABEL}的 avgpool2d_backward 的 input_shape 必须是 NCHW 四元组，"
                f"收到 {shape}"
            )
        (kh, kw), (sh, sw), (ph, pw) = self._pool_geometry(attrs, "avgpool2d_backward")
        ceil_mode, count_include_pad = _pool_flags(attrs, "avgpool2d_backward")
        n, c, h, w = shape
        ho = _pool_out_dim(h, kh, sh, ph, ceil_mode)
        wo = _pool_out_dim(w, kw, sw, pw, ceil_mode)
        if ho <= 0 or wo <= 0:
            raise BackendCapabilityError(
                f"avgpool2d_backward 的窗口尺寸非法：输入 {h}x{w}、核 {kh}x{kw}、"
                f"padding {(ph, pw)}、stride {(sh, sw)}、ceil_mode={ceil_mode}"
            )
        if tuple(dy.shape) != (n, c, ho, wo):
            raise BackendCapabilityError(
                f"avgpool2d_backward 的 dy 形状应为 {(n, c, ho, wo)}，收到 {dy.shape}"
            )

        dx = cp.zeros(shape, dtype=dy.dtype)
        total = int(dy.size)
        if total == 0:
            return dx

        dtype_key = self._kind(dy)
        kernel = self._kernel(
            ("avgpool2d_backward", dtype_key),
            lambda: self.codegen.render(self.codegen.avgpool2d_backward_source(), dy.dtype),
            "avgpool2d_backward",
        )
        block, grid = self._grid("avgpool2d_backward", total)
        kernel(
            (grid,),
            (block,),
            (
                dy,
                dx,
                np.int32(c), np.int32(h), np.int32(w),
                np.int32(kh), np.int32(kw), np.int32(ho), np.int32(wo),
                np.int32(sh), np.int32(sw), np.int32(ph), np.int32(pw),
                np.int64(total), np.int32(1 if count_include_pad else 0),
            ),
        )
        return dx



    def _op_adaptive_avg_pool2d(self, node, args, attrs):
        """自适应平均池化：一个输出元素一个线程，窗口端点按整数公式算。

        ``start = i*H//OH``、``end = ceil((i+1)*H/OH)``，与 ``ir/ops.py`` 的
        ``_adaptive_window`` 逐字对齐——窗口端点差一格会直接导致对拍失败。
        ``output_size=(1, 1)`` 就是全局平均池化，不需要单独的算子。
        """
        cp = self.cp
        x = self._contig(args[0])
        if x.ndim != 4:
            raise BackendCapabilityError(
                f"{self._LABEL}的 adaptive_avg_pool2d 要求 NCHW 秩 4，收到 {x.shape}"
            )
        oh, ow = self._adaptive_output_size(attrs, "adaptive_avg_pool2d")
        n, c, h, w = (int(v) for v in x.shape)

        out = cp.empty((n, c, oh, ow), dtype=x.dtype)
        total = int(out.size)
        if total == 0:
            return out

        dtype_key = self._kind(x)
        kernel = self._kernel(
            ("adaptive_avg_pool2d", dtype_key),
            lambda: self.codegen.render(self.codegen.adaptive_avg_pool2d_source(), x.dtype),
            "adaptive_avg_pool2d",
        )
        block, grid = self._grid("adaptive_avg_pool2d", total)
        kernel(
            (grid,),
            (block,),
            (
                x,
                out,
                np.int32(c), np.int32(h), np.int32(w),
                np.int32(oh), np.int32(ow),
                np.int64(total),
            ),
        )
        return out



    def _op_adaptive_avg_pool2d_backward(self, node, args, attrs):
        """自适应平均池化反向：``dy/窗口面积`` 散射回窗口内的每个输入格。

        输入尺寸不是输出尺寸的整数倍时窗口会重叠，同一个输入格会被多个
        输出格写，因此必须 atomicAdd；dx 先清零，且不能用空内核。
        """
        cp = self.cp
        dy = self._contig(args[0])
        shape = tuple(int(d) for d in attrs["input_shape"])
        if len(shape) != 4:
            raise BackendCapabilityError(
                f"{self._LABEL}的 adaptive_avg_pool2d_backward 的 input_shape "
                f"必须是 NCHW 四元组，收到 {shape}"
            )
        oh, ow = self._adaptive_output_size(attrs, "adaptive_avg_pool2d_backward")
        n, c, h, w = shape
        if tuple(dy.shape) != (n, c, oh, ow):
            raise BackendCapabilityError(
                f"adaptive_avg_pool2d_backward 的 dy 形状应为 {(n, c, oh, ow)}，"
                f"收到 {dy.shape}"
            )

        dx = cp.zeros(shape, dtype=dy.dtype)
        total = int(dy.size)
        if total == 0:
            return dx

        dtype_key = self._kind(dy)
        kernel = self._kernel(
            ("adaptive_avg_pool2d_backward", dtype_key),
            lambda: self.codegen.render(self.codegen.adaptive_avg_pool2d_backward_source(), dy.dtype),
            "adaptive_avg_pool2d_backward",
        )
        block, grid = self._grid("adaptive_avg_pool2d_backward", total)
        kernel(
            (grid,),
            (block,),
            (
                dy,
                dx,
                np.int32(c), np.int32(h), np.int32(w),
                np.int32(oh), np.int32(ow),
                np.int64(total),
            ),
        )
        return dx


    # ------------------------------------------------------------------
    # 归约：只支持最后一维
    # ------------------------------------------------------------------

    def _op_softmax(self, node, args, attrs):
        cp = self.cp
        x = args[0]
        # 可选的第二输入是标量缩放（融合 pass 把 softmax(mul(x, s)) 合并成 softmax(x, s)）。
        # 一律以设备指针传给 kernel：不取值到主机，否则每次 execute 都会同步一次。
        scale = (
            args[1].reshape(1).astype(cp.float32, copy=False)
            if len(args) > 1
            else self._one()
        )
        rows, cols = self._last_dim_2d(x, "softmax")
        axis = int(attrs.get("axis", -1)) % x.ndim
        if axis != x.ndim - 1:
            return self._softmax_any_axis(x, axis, scale)
        out = cp.empty_like(x)
        if x.size == 0:
            return out
        dtype_key = self._kind(x)
        kernel = self._kernel(
            ("softmax", dtype_key),
            lambda: self.codegen.render(self.codegen.softmax_source(), x.dtype),
            "softmax_last",
        )
        block = self._reduce_block("softmax", cols)
        kernel(
            (rows,),
            (block,),
            (x, out, np.int64(rows), np.int64(cols), scale),
        )
        return out



    def _softmax_any_axis(self, x, axis: int, scale):
        """非末维 softmax：reshape 成 ``(outer, reduce, inner)``，一线程一行。

        行内元素跨 ``inner`` 步，块内归约拿不到合并访问，所以这里直接串行扫：
        最大值、``exp`` 求和、写回各一遍。``exp`` 因此算两遍——换来的是不需要
        shared 与行内同步。末维仍走 ``softmax_last``。

        沿空轴归一化没有定义（numpy 给整行 nan）：与其返回 nan，不如在边界拒绝。
        """
        cp = self.cp
        x = self._contig(x)
        if x.ndim == 0:
            raise BackendCapabilityError(f"{self._LABEL}的 softmax 需要至少一维")
        shape = tuple(int(dim) for dim in x.shape)
        outer = 1
        for dim in shape[:axis]:
            outer *= dim
        reduce_len = shape[axis]
        inner = 1
        for dim in shape[axis + 1:]:
            inner *= dim
        if reduce_len == 0:
            raise BackendCapabilityError(
                f"{self._LABEL}的 softmax 不能沿空轴归一化（形状 {shape}，axis={axis}）"
            )
        out = cp.empty_like(x)
        if out.size == 0:
            return out
        total = outer * inner
        kernel = self._kernel(
            ("softmax_axis", self._kind(x)),
            lambda: self.codegen.render(self.codegen.softmax_axis_source(), x.dtype),
            "softmax_axis",
        )
        block = self._reduce_block("softmax", total)
        kernel(
            (cdiv(total, block),),
            (block,),
            (x, out, np.int64(total), np.int64(reduce_len), np.int64(inner), scale),
        )
        return out

    def _op_attention(self, node, args, attrs):
        """融合注意力前向：分块 online softmax，不物化 ``(S, T)`` 分数矩阵。

        与参考语义的唯一区别是实现方式：这里按 key 分块、边算边重缩放，
        分数矩阵从不落盘。mask 分两种：按 ``(b, h)`` 可共享的（因果 mask 的
        ``(1,1,S,T)``）只保留一份 ``(S, T)``，由内核按 ``mask_shared`` 寻址；
        只有真正随 batch/head 变化的 mask 才广播成 ``(BH, S, T)``。广播规则
        因此仍与 numpy 一致，但共享情形不再多出 ``BH-1`` 份二次内存。

        f16 是**混合精度**：q/k/v/mask/out 用 f16 存储，点积、在线 softmax
        的行统计与输出累加器全在 float 里做，只在写回 ``out`` 时舍入一次。
        共享内存布局因此不随 dtype 变化（``smem`` 恒为 float）。
        """
        cp = self.cp
        q, k, v = args[0], args[1], args[2]
        s_len = int(q.shape[-2])
        d = int(q.shape[-1])
        t_len = int(k.shape[-2])
        e = int(v.shape[-1])
        batch = tuple(int(x) for x in q.shape[:-2])
        bh = int(np.prod(batch)) if batch else 1
        out = cp.empty(batch + (s_len, e), dtype=q.dtype)
        if out.size == 0:
            return out
        if t_len == 0:
            # 空序列的 softmax 没有定义；与其返回一行 nan，不如在边界处拒绝。
            raise BackendCapabilityError(
                f"attention 的 k/v 序列长度为 0（节点 %{node.output.name}），softmax 无定义"
            )
        if len(args) > 3 and args[3].dtype != q.dtype:
            # IR 的 infer 已经要求 mask 与 q 同 dtype；这里是运行期的最后一道
            # 边界——内核按同一个存储类型读 q 与 mask，混用会静默读错位宽。
            raise BackendCapabilityError(
                f"attention 的 mask 必须与 q 同 dtype，收到 {args[3].dtype} 与 "
                f"{q.dtype}（节点 %{node.output.name}）"
            )
        q3 = cp.ascontiguousarray(q).reshape(bh, s_len, d)
        k3 = cp.ascontiguousarray(k).reshape(bh, t_len, d)
        v3 = cp.ascontiguousarray(v).reshape(bh, t_len, e)
        if len(args) > 3:
            raw_mask = args[3]
            # 与参考实现一致：先去掉长度 1 的多余前导维（如 (1,1,S,T)）。
            # cupy 的 broadcast_to 不允许目标秩比输入小，直接传会报
            # "more dimensions than allowed"。
            while raw_mask.ndim > len(batch) + 2:
                if raw_mask.shape[0] != 1:
                    raise BackendCapabilityError(
                        f"attention 的 mask 前导维必须是 1，得到 {tuple(raw_mask.shape)}"
                        f"（节点 %{node.output.name}）"
                    )
                raw_mask = raw_mask[0]
            # 共享形态：(S,T) 或前导维全为 1。此时每个 (b,h) 读同一份，复制成
            # (BH,S,T) 只会多出 BH-1 份 S*T 的显存。
            if _mask_shared_over_heads(raw_mask.shape, s_len, t_len):
                m3 = cp.ascontiguousarray(raw_mask).reshape(s_len, t_len)
                mask_shared = 1
            else:
                try:
                    mask = cp.broadcast_to(raw_mask, batch + (s_len, t_len))
                except ValueError as exc:
                    raise BackendCapabilityError(
                        f"attention 的 mask 形状 {tuple(args[3].shape)} 无法广播到 "
                        f"{batch + (s_len, t_len)}（节点 %{node.output.name}）"
                    ) from exc
                m3 = cp.ascontiguousarray(mask).reshape(bh, s_len, t_len)
                mask_shared = 0
            has_mask = 1
        else:
            # has_mask=0 时内核不会解引用这个指针，但也不能传 None。
            m3 = q3
            has_mask = 0
            mask_shared = 0

        scale = np.float32(float(attrs.get("scale", 1.0)))
        p_drop = float(attrs.get("dropout", 0.0))
        # 缩放因子在主机侧算好：内核里每个元素都要用它，没必要每线程重算。
        drop_scale = (
            np.float32(1.0 / (1.0 - p_drop)) if p_drop > 0.0 else np.float32(1.0)
        )
        seed = np.uint64(int(attrs.get("seed", 0)) & 0xFFFFFFFFFFFFFFFF)

        block = attention_block(t_len)
        smem = 2 * block * 4 + e * 4
        if smem > SHARED_LIMIT:
            raise BackendCapabilityError(
                f"attention 的 shared memory 需求 {smem} 字节超过上限 "
                f"{SHARED_LIMIT}（E={e}，block={block}）"
            )
        kernel = self._kernel(
            ("attention", self._kind(q)),
            lambda: self.codegen.render(self.codegen.attention_source(), q.dtype),
            "attention_fwd",
        )
        kernel(
            (s_len, bh),
            (block,),
            (
                q3,
                k3,
                v3,
                m3,
                out.reshape(bh, s_len, e),
                np.int64(s_len),
                np.int64(t_len),
                np.int64(d),
                np.int64(e),
                scale,
                np.float32(p_drop),
                drop_scale,
                seed,
                np.int32(has_mask),
                np.int32(mask_shared),
            ),
            shared_mem=smem,
        )
        return out



    def _op_layernorm(self, node, args, attrs):
        cp = self.cp
        x, gamma, beta = args[0], args[1], args[2]
        residual = args[3] if len(args) > 3 else None
        prep, alloc, finish, rows, cols, _norm_shape, norm_axes = self._norm_layout(
            x, attrs.get("axes", (-1,)), "layernorm"
        )
        x_shape = tuple(int(dim) for dim in x.shape)
        gamma = self._norm_param(gamma, cols, x_shape, norm_axes, "gamma", "layernorm")
        beta = self._norm_param(beta, cols, x_shape, norm_axes, "beta", "layernorm")
        if gamma.dtype != x.dtype or beta.dtype != x.dtype:
            raise BackendCapabilityError(
                f"layernorm 的 gamma/beta 必须与 x 同 dtype，收到 "
                f"{gamma.dtype} / {beta.dtype} 与 {x.dtype}"
            )
        if residual is not None:
            if tuple(residual.shape) != tuple(x.shape) or residual.dtype != x.dtype:
                raise BackendCapabilityError(
                    f"layernorm 的 residual 必须与 x 同形状、同 dtype，收到 "
                    f"{residual.shape} 与 {x.shape}"
                )
        eps = float(attrs.get("eps", 1e-5))
        if x.size == 0:
            return cp.empty_like(x)
        xf = prep(x)
        rf = prep(residual) if residual is not None else None
        full, of = alloc()
        dtype_key = self._kind(x)
        block = self._reduce_block("layernorm", cols)
        if residual is None:
            kernel = self._kernel(
                ("layernorm", dtype_key),
                lambda: self.codegen.render(self.codegen.layernorm_source(), x.dtype),
                "layernorm_last",
            )
            kernel(
                (rows,),
                (block,),
                (xf, gamma, beta, of, np.int64(rows), np.int64(cols), np.float32(eps)),
            )
        else:
            kernel = self._kernel(
                ("layernorm_residual", dtype_key),
                lambda: self.codegen.render(
                    self.codegen.layernorm_residual_source(), x.dtype
                ),
                "layernorm_residual",
            )
            kernel(
                (rows,),
                (block,),
                (
                    xf,
                    rf,
                    gamma,
                    beta,
                    of,
                    np.int64(rows),
                    np.int64(cols),
                    np.float32(eps),
                ),
            )
        return finish(full, of)



    def _batch_norm_plan(self, x, attrs, name: str):
        """把 x 看成 ``(outer, channels, inner)``——BN 内核只认这三元组。

        这样同一个内核既能处理 NCHW（axis=1, inner=H*W），也能处理任意秩、
        任意通道轴，而不必为每种布局写一份模板。
        """
        if x.ndim < 2:
            raise BackendCapabilityError(f"CUDA 的 {name} 要求秩 >= 2，收到 {x.ndim}")
        axis = int(attrs.get("axis", 1)) % x.ndim
        shape = [int(d) for d in x.shape]
        outer = 1
        for d in shape[:axis]:
            outer *= d
        inner = 1
        for d in shape[axis + 1 :]:
            inner *= d
        return outer, shape[axis], inner



    def _op_batch_norm(self, node, args, attrs):
        cp = self.cp
        x = self._contig(args[0])
        gamma = self._contig(args[1])
        beta = self._contig(args[2])
        outer, channels, inner = self._batch_norm_plan(x, attrs, "batch_norm")
        if tuple(gamma.shape) != (channels,) or tuple(beta.shape) != (channels,):
            raise BackendCapabilityError(
                f"batch_norm 的 gamma/beta 必须为 ({channels},)，收到 "
                f"{tuple(gamma.shape)} 与 {tuple(beta.shape)}"
            )
        eps = float(attrs.get("eps", 1e-5))
        out = cp.empty(x.shape, dtype=x.dtype)
        if x.size == 0:
            return out
        dtype_key = self._kind(x)
        kernel = self._kernel(
            ("batch_norm", dtype_key),
            lambda: self.codegen.render(self.codegen.batch_norm_source(), x.dtype),
            "batch_norm",
        )
        block = self._reduce_block("batch_norm", outer * inner)
        kernel(
            (channels,),
            (block,),
            (
                x,
                gamma,
                beta,
                out,
                np.int64(outer),
                np.int64(channels),
                np.int64(inner),
                np.float32(eps),
            ),
        )
        return out



    def _op_batch_norm_stats(self, node, args, attrs):
        cp = self.cp
        x = self._contig(args[0])
        outer, channels, inner = self._batch_norm_plan(x, attrs, "batch_norm_stats")
        out = cp.empty((2, channels), dtype=x.dtype)
        if x.size == 0:
            return out
        dtype_key = self._kind(x)
        kernel = self._kernel(
            ("batch_norm_stats", dtype_key),
            lambda: self.codegen.render(self.codegen.batch_norm_stats_source(), x.dtype),
            "batch_norm_stats",
        )
        block = self._reduce_block("batch_norm_stats", outer * inner)
        kernel(
            (channels,),
            (block,),
            (x, out, np.int64(outer), np.int64(channels), np.int64(inner)),
        )
        return out

    # ------------------------------------------------------------------
    # 序列扫描（SSM 的线性递归）
    # ------------------------------------------------------------------

    def _op_scan_linear(self, node, args, attrs):
        return self._scan(args[0], args[1], args[2], backward=False)



    def _op_scan_linear_backward(self, node, args, attrs):
        return self._scan(args[0], args[1], None, backward=True)

    # ------------------------------------------------------------------
    # 前缀和 / 三角掩码（分块 SSM 的块内权重）
    # ------------------------------------------------------------------

    def _op_cumsum(self, node, args, attrs):
        """沿单个轴的前缀和：一个线程一条 ``(outer, inner)`` 链。

        张量按 ``(outer, steps, inner)`` 切分，内核里的 ``work`` 就是并行链数
        ``outer * inner``，grid 因此按它铺——与 ``scan_linear`` 是同一套铺法，
        区别只在没有 ``a_t`` 乘法。axis 为负时先归一化再折维度；轴越界在设备
        端也拦一次，避免快路径绕过 verifier 时把错误轴当成合法轴。
        """
        cp = self.cp
        x = self._contig(args[0])
        axis = attrs.get("axis", -1)
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise BackendCapabilityError(
                f"{self._LABEL}的 cumsum 的 axis 必须是整数，收到 {axis!r}"
            )
        axis = axis + x.ndim if axis < 0 else axis
        if not 0 <= axis < x.ndim:
            raise BackendCapabilityError(
                f"{self._LABEL}的 cumsum 轴越界: {attrs.get('axis')!r}，秩为 {x.ndim}"
            )
        shape = tuple(int(d) for d in x.shape)
        steps = shape[axis]
        outer = int(np.prod(shape[:axis], dtype=np.int64))
        inner = int(np.prod(shape[axis + 1 :], dtype=np.int64))
        out = cp.empty_like(x)
        work = outer * inner
        if work == 0 or steps == 0:
            return out
        kernel = self._kernel(
            ("cumsum", self._kind(x)),
            lambda: self.codegen.render(self.codegen.cumsum_source(), x.dtype),
            "cumsum",
        )
        block, grid = self._grid("cumsum", work)
        kernel(
            (grid,),
            (block,),
            (x, out, np.int64(work), np.int64(steps), np.int64(inner)),
        )
        return out

    def _op_tril(self, node, args, attrs):
        """最后两维保留下三角、其余置零：一个线程一个元素。

        C 布局下线性下标满足 ``c = idx % cols``、``r = (idx / cols) % rows``，
        前导维被 ``% rows`` 自然吸收，所以不需要额外传 batch；条件外写 0
        而不是跳过，输出才不会带着未初始化值。
        """
        cp = self.cp
        x = self._contig(args[0])
        if x.ndim < 2:
            raise BackendCapabilityError(
                f"{self._LABEL}的 tril 至少需要秩 2，收到 {tuple(x.shape)}"
            )
        diagonal = attrs.get("diagonal", 0)
        if isinstance(diagonal, bool) or not isinstance(diagonal, int):
            raise BackendCapabilityError(
                f"{self._LABEL}的 tril 的 diagonal 必须是整数，收到 {diagonal!r}"
            )
        rows, cols = int(x.shape[-2]), int(x.shape[-1])
        out = cp.empty_like(x)
        n = int(x.size)
        if n == 0 or rows == 0 or cols == 0:
            return out
        kernel = self._kernel(
            ("tril", self._kind(x)),
            lambda: self.codegen.render(self.codegen.tril_source(), x.dtype),
            "tril",
        )
        block, grid = self._grid("tril", n)
        kernel(
            (grid,),
            (block,),
            (
                x,
                out,
                np.int64(n),
                np.int64(rows),
                np.int64(cols),
                np.int64(diagonal),
            ),
        )
        return out



    def _op_spike_offsets(self, node, args, attrs):
        """统计每个时间步的非零数，写出 CSR 起点表 ``[T+1]``。

        与 ``spike_csr`` 拆成两个节点，而不是打包进一个 i32 张量：IR 没有
        多输出节点，打包后下游必须 ``slice`` 切整数缓冲，而 GPU 后端的
        ``slice`` 明确不支持 int32（不做隐式降级）。两趟内核各自单输出，
        下游 ``sparse_linear`` 直接拿到起点与列号，无需任何整数切片。
        """
        cp = self.cp
        (spikes,) = args
        spikes = self._contig(spikes)
        steps, width = int(spikes.shape[0]), int(spikes.shape[1])
        offsets = cp.zeros((steps + 1,), dtype=cp.int32)
        kernel = self._kernel(
            ("spike_offsets", self._kind(spikes)),
            lambda: self.codegen.render(
                self.codegen.spike_offsets_source(), spikes.dtype
            ),
            "spike_offsets",
        )
        # 单 block：阶段 1 并行数行，阶段 2 由 0 号线程做 T 步前缀和。
        # 不能用 _grid：那会铺多个 block，块间无法 __syncthreads。
        block = max(1, min(256, steps))
        kernel(
            (1,),
            (block,),
            (spikes, offsets, np.int64(width), np.int64(steps)),
        )
        return offsets

    def _use_scan_csr(self, steps: int) -> bool:
        """``spike_csr`` 是否走块内扫描（5.176）。

        判据是 ``plan_csr_scan_enabled``：只有行数少到串行铺法填不满机器时才
        换铺法——行数多时串行版靠每线程的长循环就能藏住访存延迟，并行版反而
        多付块内同步。单独抽成方法只为让基准能强制走某一条做对照
        （``benchmarks/spike_csr_scan.py`` 在实例上盖掉它），与 5.174 的
        ``_use_tiled_reduce`` 是同一个理由。
        """
        return plan_csr_scan_enabled(steps, self.sm_count)

    def _op_spike_csr(self, node, args, attrs):
        """按起点表把每行的非零列号写进 ``indices[capacity]``。

        容量固定是静态图的前提；溢出在设备端记 ``err``，主机侧抛错——
        与 ``sparse_linear`` 的越界检查同一条通路，捕获期也成立。
        """
        cp = self.cp
        spikes, offsets = args
        spikes = self._contig(spikes)
        offsets = self._contig(offsets)
        steps, width = int(spikes.shape[0]), int(spikes.shape[1])
        capacity = int(attrs["capacity"])
        indices = cp.zeros((capacity,), dtype=cp.int32)
        err = cp.zeros(1, dtype=cp.int32)
        kind = self._kind(spikes)

        scan = self._use_scan_csr(steps)
        kernel_name = "spike_csr" if scan else "spike_csr_serial"
        kernel = self._kernel(
            (kernel_name, kind),
            lambda: self.codegen.render(
                (
                    self.codegen.spike_csr_source()
                    if scan
                    else self.codegen.spike_csr_serial_source()
                ),
                spikes.dtype,
            ),
            kernel_name,
        )
        # 一个 block 负责一行（5.176）：块内要用 __syncthreads 做前缀和，
        # 所以网格不能像一维逐元素那样被 _grid 截断到 max_blocks——截断会把
        # 后面的行整行漏掉。块大小只跟列数走；串行铺法是一个线程一行，按行数铺。
        if scan:
            block = plan_csr_scan(width)
            grid = max(1, steps)
        else:
            block, grid = self._grid("spike_csr_serial", steps)
        kernel(
            (grid,),
            (block,),
            (
                spikes,
                offsets,
                indices,
                np.int64(capacity),
                np.int64(width),
                np.int64(steps),
                err,
            ),
        )
        self._check_ids(err, "spike_csr", f"[0, {capacity}]（事件数超过容量）")
        return indices

    def _pick_cusparse(self, values, indices, offsets) -> bool:
        """这一次 ``sparse_linear`` 走不走厂商库；默认永不启用。

        CUDA 覆盖它（5.203）。放在基类而不是写成 ``hasattr`` 探测：HIP 的
        ``SPARSE_IMPLS`` 里没有 ``"cusparse"``，构造期就不会接受这个值，
        这里只需回答"没有库路径"。
        """
        return False

    def _sparse_linear_cusparse(self, weights, indices, values, offsets, *,
                                steps: int, rows: int, cols: int):
        """厂商库的 SpMM；默认没有，返回 ``None``（调用方回落自研核）。"""
        return None

    def _op_sparse_linear(self, node, args, attrs):
        """事件驱动的稀疏线性：CSR 事件 × 稠密权重。

        ``indices`` / ``values`` 的长度是事件数，可以是符号维——它们只决定
        内核读多少事件，不影响输出形状。``offsets`` 的长度决定输出步数，
        ``infer`` 已要求它是静态的。

        计算量是 ``S × N`` 个输出元素各自扫自己那一段事件；稀疏脉冲矩阵
        的 ``T × K`` 稠密形态从不物化，读的事件总量等于真正发放的次数。
        """
        cp = self.cp
        weights, indices, values, offsets = args
        weights = self._contig(weights)
        values = self._contig(values)
        indices = cp.ascontiguousarray(indices).reshape(-1)
        offsets = cp.ascontiguousarray(offsets).reshape(-1)
        if int(indices.size) != int(values.size):
            raise BackendCapabilityError(
                f"{self._LABEL}的 sparse_linear 要求 indices 与 values 等长，"
                f"收到 {int(indices.size)} 与 {int(values.size)}"
            )
        steps = int(offsets.size) - 1
        rows, cols = int(weights.shape[0]), int(weights.shape[1])
        total = steps * cols
        if total == 0:
            return cp.zeros((steps, cols), dtype=values.dtype)
        # 5.203：库路径只在显式要求（或 auto 且库可用）时走。默认 custom
        # 保留越界检查；库路径不做下标检查，这是它不设默认值的直接理由。
        if self._pick_cusparse(values, indices, offsets):
            fast = self._sparse_linear_cusparse(
                weights, indices, values, offsets,
                steps=steps, rows=rows, cols=cols,
            )
            if fast is not None:
                return fast
        out = cp.zeros((steps, cols), dtype=values.dtype)
        kernel = self._kernel(
            ("sparse_linear", self._kind(values)),
            lambda: self.codegen.render(
                self.codegen.sparse_linear_source(), values.dtype
            ),
            "sparse_linear",
        )
        err = cp.zeros(1, dtype=cp.int32)
        block, grid = self._grid("sparse_linear", total)
        kernel(
            (grid,),
            (block,),
            (
                weights,
                indices,
                values,
                offsets,
                out,
                np.int64(cols),
                np.int64(rows),
                np.int64(int(values.size)),
                np.int64(total),
                err,
            ),
        )
        self._check_ids(err, "sparse_linear", f"[0, {rows})")
        return out

    def _op_sparse_linear_weight_grad(self, node, args, attrs):
        """``sparse_linear`` 对权重的梯度：把事件贡献散射回各自的行。

        输出先清零再原子累加——同一行在一个 batch 内被命中多次是常态。
        ``rows`` 只能从属性读：反向图里没有 ``weights`` 这个输入。
        """
        cp = self.cp
        dout, indices, values, offsets = args
        dout = self._contig(dout)
        values = self._contig(values)
        indices = cp.ascontiguousarray(indices).reshape(-1)
        offsets = cp.ascontiguousarray(offsets).reshape(-1)
        if int(indices.size) != int(values.size):
            raise BackendCapabilityError(
                f"{self._LABEL}的 sparse_linear_weight_grad 要求 indices 与 "
                f"values 等长，收到 {int(indices.size)} 与 {int(values.size)}"
            )
        rows = int(attrs["rows"])
        steps, cols = int(dout.shape[0]), int(dout.shape[1])
        dweights = cp.zeros((rows, cols), dtype=dout.dtype)
        total = steps * cols
        if total == 0:
            return dweights
        kernel = self._kernel(
            ("sparse_linear_weight_grad", self._kind(dout)),
            lambda: self.codegen.render(
                self.codegen.sparse_linear_weight_grad_source(), dout.dtype
            ),
            "sparse_linear_weight_grad",
        )
        err = cp.zeros(1, dtype=cp.int32)
        block, grid = self._grid("sparse_linear_weight_grad", total)
        kernel(
            (grid,),
            (block,),
            (
                dout,
                indices,
                values,
                offsets,
                dweights,
                np.int64(cols),
                np.int64(rows),
                np.int64(int(values.size)),
                np.int64(total),
                err,
            ),
        )
        self._check_ids(err, "sparse_linear_weight_grad", f"[0, {rows})")
        return dweights

    def _op_sparse_linear_values_grad(self, node, args, attrs):
        """``sparse_linear`` 对事件幅度的梯度：每个事件与所属步的 ``dout`` 做点积。

        按步并行——一个线程包办该步的全部事件，直接写 ``dvalues[e]``。事件之间
        没有重复下标，所以不像权重梯度那样需要"先清零再原子累加"。代价是
        ``steps`` 少时并行度低，但省掉了在设备上把事件号映射回步号的二分查找。
        """
        cp = self.cp
        dout, weights, indices, offsets = args
        dout = self._contig(dout)
        weights = self._contig(weights)
        indices = cp.ascontiguousarray(indices).reshape(-1)
        offsets = cp.ascontiguousarray(offsets).reshape(-1)
        steps = int(offsets.size) - 1
        rows, cols = int(weights.shape[0]), int(weights.shape[1])
        nnz = int(indices.size)
        dvalues = cp.zeros((nnz,), dtype=dout.dtype)
        if steps == 0:
            return dvalues
        kernel = self._kernel(
            ("sparse_linear_values_grad", self._kind(dout)),
            lambda: self.codegen.render(
                self.codegen.sparse_linear_values_grad_source(), dout.dtype
            ),
            "sparse_linear_values_grad",
        )
        err = cp.zeros(1, dtype=cp.int32)
        block, grid = self._grid("sparse_linear_values_grad", steps)
        kernel(
            (grid,),
            (block,),
            (
                dout,
                weights,
                indices,
                offsets,
                dvalues,
                np.int64(cols),
                np.int64(rows),
                np.int64(nnz),
                np.int64(steps),
                err,
            ),
        )
        self._check_ids(err, "sparse_linear_values_grad", f"[0, {rows})")
        return dvalues

    def _op_spike_linear(self, node, args, attrs):
        """事件驱动的脉冲线性：事件幅度直接从脉冲矩阵按列号读取。

        与 ``sparse_linear`` 的结构完全相同，只是把 ``values[e]`` 换成
        ``spikes[s, indices[e]]``。省掉 values 缓冲之后，反向对脉冲就是一次
        普通 GEMM，未发放位置的 surrogate 梯度不会被丢掉。
        """
        cp = self.cp
        spikes, weights, indices, offsets = args
        spikes = self._contig(spikes)
        weights = self._contig(weights)
        indices = cp.ascontiguousarray(indices).reshape(-1)
        offsets = cp.ascontiguousarray(offsets).reshape(-1)
        steps = int(offsets.size) - 1
        rows, cols = int(weights.shape[0]), int(weights.shape[1])
        out = cp.zeros((steps, cols), dtype=spikes.dtype)
        total = steps * cols
        if total == 0:
            return out
        kernel = self._kernel(
            ("spike_linear", self._kind(spikes)),
            lambda: self.codegen.render(
                self.codegen.spike_linear_source(), spikes.dtype
            ),
            "spike_linear",
        )
        err = cp.zeros(1, dtype=cp.int32)
        block, grid = self._grid("spike_linear", total)
        kernel(
            (grid,),
            (block,),
            (
                spikes,
                weights,
                indices,
                offsets,
                out,
                np.int64(cols),
                np.int64(rows),
                np.int64(int(indices.size)),
                np.int64(total),
                err,
            ),
        )
        self._check_ids(err, "spike_linear", f"[0, {rows})")
        return out

    def _op_spike_linear_weight_grad(self, node, args, attrs):
        """``spike_linear`` 对权重的梯度：事件贡献散射回各自的行。

        行号来自 CSR 的 ``indices``，幅度来自脉冲矩阵本身；输出先清零再原子
        累加——同一行被多个事件命中是常态。
        """
        cp = self.cp
        dout, spikes, indices, offsets = args
        dout = self._contig(dout)
        spikes = self._contig(spikes)
        indices = cp.ascontiguousarray(indices).reshape(-1)
        offsets = cp.ascontiguousarray(offsets).reshape(-1)
        rows = int(spikes.shape[1])
        steps, cols = int(dout.shape[0]), int(dout.shape[1])
        dweights = cp.zeros((rows, cols), dtype=dout.dtype)
        total = steps * cols
        if total == 0:
            return dweights
        kernel = self._kernel(
            ("spike_linear_weight_grad", self._kind(dout)),
            lambda: self.codegen.render(
                self.codegen.spike_linear_weight_grad_source(), dout.dtype
            ),
            "spike_linear_weight_grad",
        )
        err = cp.zeros(1, dtype=cp.int32)
        block, grid = self._grid("spike_linear_weight_grad", total)
        kernel(
            (grid,),
            (block,),
            (
                dout,
                spikes,
                indices,
                offsets,
                dweights,
                np.int64(cols),
                np.int64(rows),
                np.int64(int(indices.size)),
                np.int64(total),
                err,
            ),
        )
        self._check_ids(err, "spike_linear_weight_grad", f"[0, {rows})")
        return dweights

    def _scan(self, a, b, h0, backward: bool):
        """线性递归（及其伴随）的 CUDA 承接。

        分工是**元素并行、时间串行**：一个线程独占一个尾元素（把
        ``a.shape[1:]`` 压平后的一格），沿 T 顺序推进。递推本身没有结合律
        之外的并行空间可用，而这里的并行度来自尾维（batch × 通道 × 状态）。

        串行版在"长 T + 小尾维"时是纯延迟：设备填不满，瓶颈就是那一串
        逐时间步的依赖。``scan_chunk`` 按 ``(steps, elems)`` 决定是否换成
        两趟分块（块内复合 + 块前缀/读出），把并行度放大到
        ``elems * chunks``；短 T 或尾维已经喂饱带宽时仍走串行，见
        ``lower/plan.py`` 的 ``SCAN_CHUNK_*``。

        为什么可以压平尾维：a/b 的布局是 (T, 尾...)，展平后第 t 步的偏移
        就是 ``t * elems + i``，与尾维的具体阶数无关，因此 2 维、3 维、
        4 维输入共用同一个 kernel。
        """
        cp = self.cp
        a = self._contig(a)
        dy_or_b = self._contig(b)
        if tuple(dy_or_b.shape) != tuple(a.shape):
            raise BackendCapabilityError(
                f"{self._LABEL}的 scan 要求 a 与第二输入同形，"
                f"收到 {tuple(a.shape)} 与 {tuple(dy_or_b.shape)}"
            )
        if a.ndim < 2:
            raise BackendCapabilityError(
                f"{self._LABEL}的 scan 至少要求 (T, 尾...) 两个维度，收到 {tuple(a.shape)}"
            )
        out = cp.empty_like(a)
        steps = int(a.shape[0])
        if steps == 0 or int(a.size) == 0:
            return out
        elems = int(a.size) // steps

        if not backward:
            if tuple(h0.shape) != tuple(a.shape[1:]):
                raise BackendCapabilityError(
                    f"scan_linear 的 h0 形状应为 {tuple(a.shape[1:])}，"
                    f"收到 {tuple(h0.shape)}"
                )
            h0 = cp.ascontiguousarray(h0)

        chunk = scan_chunk(self.chunked_scan, steps, elems)
        if chunk:
            return self._scan_chunked(
                a,
                dy_or_b,
                h0,
                out,
                chunk=chunk,
                backward=backward,
                steps=steps,
                elems=elems,
            )

        dtype_key = self._kind(a)
        kernel = self._kernel(
            ("scan_linear_backward", dtype_key) if backward else ("scan_linear", dtype_key),
            lambda: self.codegen.render(
                self.codegen.scan_linear_backward_source()
                if backward
                else self.codegen.scan_linear_source(),
                a.dtype,
            ),
            "scan_linear_backward" if backward else "scan_linear",
        )
        block, grid = self._grid(
            "scan_linear_backward" if backward else "scan_linear", elems
        )
        args = [a, dy_or_b] if backward else [a, dy_or_b, h0]
        kernel(
            (grid,),
            (block,),
            (*args, out, np.int64(steps), np.int64(elems)),
        )
        return out

    def _scan_chunk_sums(self, a, b, steps, elems, chunk, *, backward):
        """分块扫描第一趟：每块算复合 ``(prod, sum)``。

        scratch 固定 float32（不随 dtype 变 f16）：块内递推本来就在 float
        里做，把聚合值降成 f16 会多两次舍入，误差比串行路径大一个数量级。
        """
        cp = self.cp
        chunks = (steps + chunk - 1) // chunk
        total = chunks * elems
        prod = cp.empty(total, dtype=cp.float32)
        acc = cp.empty(total, dtype=cp.float32)
        name = (
            "scan_linear_chunk_sum_backward"
            if backward
            else "scan_linear_chunk_sum"
        )
        kernel = self._kernel(
            (name, self._kind(a)),
            lambda: self.codegen.render(
                self.codegen.scan_linear_chunk_sum_backward_source()
                if backward
                else self.codegen.scan_linear_chunk_sum_source(),
                a.dtype,
            ),
            name,
        )
        block, grid = self._grid(name, total)
        kernel(
            (grid,),
            (block,),
            (
                a,
                b,
                prod,
                acc,
                np.int64(steps),
                np.int64(elems),
                np.int64(chunk),
                np.int64(total),
            ),
        )
        return prod, acc, chunks

    def _scan_chunked(self, a, b, h0, out, *, chunk, backward, steps, elems):
        """分块扫描第二趟：块前缀 + 块内顺序读出。

        并行度从 ``elems`` 变成 ``elems * chunks``；块内递推与串行版逐字
        相同，只有跨块的入口状态走了"先乘再加"的复合，因此结果**不逐位**
        等于串行版。块长由 :func:`scan_chunk` 解析（自动或显式），传进来而
        不是在这里重算：第一趟的 scratch 形状必须与第二趟的块长一致。
        """
        prod, acc, chunks = self._scan_chunk_sums(
            a, b, steps, elems, chunk, backward=backward
        )
        total = chunks * elems
        name = "scan_linear_chunk_backward" if backward else "scan_linear_chunk"
        kernel = self._kernel(
            (name, self._kind(a)),
            lambda: self.codegen.render(
                self.codegen.scan_linear_chunk_backward_source()
                if backward
                else self.codegen.scan_linear_chunk_source(),
                a.dtype,
            ),
            name,
        )
        block, grid = self._grid(name, total)
        args = [a, b, prod, acc, out] if backward else [a, b, h0, prod, acc, out]
        kernel(
            (grid,),
            (block,),
            (
                *args,
                np.int64(steps),
                np.int64(elems),
                np.int64(chunk),
                np.int64(total),
            ),
        )
        return out

    def scan_last_state(self, a, b, h0):
        """只返回 ``(tail...)`` 的末状态，不物化整段 ``out``。

        语义与 ``scan_linear`` 的最后一步相同，省掉 ``T-1`` 次全局写。
        与图无关的入口：流式推理每步只要末状态，没必要先把 ``(T, tail)``
        写出来再切片。
        """
        cp = self.cp
        a = self._contig(a)
        b = self._contig(b)
        if tuple(a.shape) != tuple(b.shape):
            raise BackendCapabilityError(
                f"{self._LABEL}的 scan 要求 a 与第二输入同形，"
                f"收到 {tuple(a.shape)} 与 {tuple(b.shape)}"
            )
        if a.ndim < 2:
            raise BackendCapabilityError(
                f"{self._LABEL}的 scan 至少要求 (T, 尾...) 两个维度，收到 {tuple(a.shape)}"
            )
        tail = tuple(a.shape[1:])
        if tuple(h0.shape) != tail:
            raise BackendCapabilityError(
                f"scan_last_state 的 h0 形状应为 {tail}，收到 {tuple(h0.shape)}"
            )
        h0 = cp.ascontiguousarray(h0)
        out = cp.empty(tail, dtype=a.dtype)
        steps = int(a.shape[0])
        elems = int(a.size) // steps if steps else 0
        if steps == 0 or elems == 0:
            return out

        chunk = scan_chunk(self.chunked_scan, steps, elems, last=True)
        if chunk:
            prod, acc, chunks = self._scan_chunk_sums(
                a, b, steps, elems, chunk, backward=False
            )
            name = "scan_linear_chunk_last"
            kernel = self._kernel(
                (name, self._kind(a)),
                lambda: self.codegen.render(
                    self.codegen.scan_linear_chunk_last_source(), a.dtype
                ),
                name,
            )
            block, grid = self._grid(name, elems)
            kernel(
                (grid,),
                (block,),
                (prod, acc, h0, out, np.int64(chunks), np.int64(elems)),
            )
            return out

        name = "scan_linear_last"
        kernel = self._kernel(
            (name, self._kind(a)),
            lambda: self.codegen.render(
                self.codegen.scan_linear_last_source(), a.dtype
            ),
            name,
        )
        block, grid = self._grid(name, elems)
        kernel(
            (grid,),
            (block,),
            (a, b, h0, out, np.int64(steps), np.int64(elems)),
        )
        return out



    def _op_rmsnorm(self, node, args, attrs):
        """RMSNorm：与 layernorm 同一个"一行一个 block"的归约形态。

        归一化轴的处理与 layernorm 完全一致（见 ``_norm_layout``）：``axes`` 是
        最后一维时零拷贝，其余轴集合先置换再算。gamma 的可接受形态也一致：只要
        能广播到 ``x`` 且不随非归一化维变化（``(cols,)``、归一化维的自然形状、
        带 1 前缀的形态都算），就展平成 ``(cols,)`` 交给内核。
        """
        cp = self.cp
        x, gamma = args
        prep, alloc, finish, rows, cols, _norm_shape, norm_axes = self._norm_layout(
            x, attrs.get("axes", (-1,)), "rmsnorm"
        )
        x_shape = tuple(int(dim) for dim in x.shape)
        gamma = self._norm_param(gamma, cols, x_shape, norm_axes, "gamma", "rmsnorm")
        if gamma.dtype != x.dtype:
            raise BackendCapabilityError(
                f"rmsnorm 的 gamma 必须与 x 同 dtype，收到 {gamma.dtype} 与 {x.dtype}"
            )
        eps = float(attrs.get("eps", 1e-6))
        if x.size == 0:
            return cp.empty_like(x)
        xf = prep(x)
        full, of = alloc()
        dtype_key = self._kind(x)
        kernel = self._kernel(
            ("rmsnorm", dtype_key),
            lambda: self.codegen.render(self.codegen.rmsnorm_source(), x.dtype),
            "rmsnorm_last",
        )
        block = self._reduce_block("rmsnorm", cols)
        kernel(
            (rows,),
            (block,),
            (xf, gamma, of, np.int64(rows), np.int64(cols), np.float32(eps)),
        )
        return finish(full, of)

    def _use_tiled_reduce(self, inner: int, total: int) -> bool:
        """非末维归约该不该走两维铺法（5.174）。

        判据是"输出平面能不能自己把机器填满，并且访存能不能合并"：
        ``inner`` 不足一个 warp 时一个线程一个输出连半条 cache line 都读不满；
        ``outer * inner`` 不足一屏 SM 时连第一件事都做不到。两个条件都不满足
        就留在一维铺法——它的并行度是 ``outer * inner``，比按归约轴切块更高。

        单独抽成方法是为了让基准能强制走某一条路做对照（见
        ``benchmarks/reduce_any_axis.py``）；生产路径只经这里分派。
        """
        return inner < REDUCE_TILE_X or total < self.sm_count * BLOCK_DEFAULT

    def _reduce_any_axis(self, node, args, axis: int, attrs, op: str):
        """非末维归约：reshape 成 ``(outer, reduce, inner)`` 交给通用内核。

        两条路，按"输出平面能不能自己把活干完"选：

        - ``inner < REDUCE_TILE_X`` 或 ``outer * inner`` 填不满一屏 SM 时，走
          ``reduce_axis_tiled``：``blockDim.x`` 沿 inner（相邻线程读相邻元素），
          ``blockDim.y`` 切 reduce 轴，块内沿 y 做树归约。这是 5.174 加的
          快路径——一维铺法在这些形状上既没有足够并行度，warp 也凑不满一段
          连续的 inner；
        - 其余情况仍用一维铺法（``reduce_axis``）：此时相邻线程的 ``i`` 已经
          连续，访存合并成立，而 ``outer * inner`` 个线程的并行度比按 reduce
          轴切块更高。

        两条路的数值语义相同：读入转 float、f16 只在写回时舍入一次。

        输入先取连续视图：通用内核按展平下标读，转置视图会让下标语义错位。
        """
        cp = self.cp
        x = self._contig(args[0])
        if x.ndim == 0:
            raise BackendCapabilityError(f"{self._LABEL}的 {op} 需要至少一维")
        keepdims = bool(attrs.get("keepdims", False))
        shape = tuple(int(dim) for dim in x.shape)
        outer = 1
        for dim in shape[:axis]:
            outer *= dim
        reduce_len = shape[axis]
        inner = 1
        for dim in shape[axis + 1:]:
            inner *= dim
        total = outer * inner
        target = shape[:axis] + ((1,) if keepdims else ()) + shape[axis + 1:]
        out = cp.empty(total, dtype=x.dtype)
        if out.size == 0:
            return out.reshape(target)
        kind = {
            "reduce_sum": "sum",
            "reduce_max": "max",
            "reduce_mean": "mean",
        }[op]
        if self._use_tiled_reduce(inner, total):
            tx, ty = plan_reduce_tiled(inner, reduce_len)
            kernel = self._kernel(
                (f"reduce_{kind}_axis_tiled", self._kind(x)),
                lambda: self.codegen.render(
                    self.codegen.reduce_axis_tiled_source(kind), x.dtype
                ),
                f"reduce_{kind}_axis_tiled",
            )
            kernel(
                (outer, cdiv(inner, tx)),
                (tx, ty),
                (x, out, np.int64(reduce_len), np.int64(inner)),
            )
            return out.reshape(target)
        kernel = self._kernel(
            (f"reduce_{kind}_axis", self._kind(x)),
            lambda: self.codegen.render(self.codegen.reduce_axis_source(kind), x.dtype),
            f"reduce_{kind}_axis",
        )
        block = self._reduce_block(op, total)
        kernel(
            (cdiv(total, block),),
            (block,),
            (x, out, np.int64(total), np.int64(reduce_len), np.int64(inner)),
        )
        return out.reshape(target)

    def _reduce_multi_axis(self, node, args, axes, attrs, op, source_fn, func):
        """多轴归约：``[keep..., reduce...]`` 置换后合并成末维，复用单轴内核。

        归约与轴顺序无关，所以"先把要归约的轴挪到最后、乘成一个轴"是等价
        变换；代价是一次转置拷贝（输入本身已连续且轴本来就在末尾时为零）。
        空归约轴按 numpy 的语义返回单位元：sum 是 0、mean 是 NaN（numpy 会
        同时发 warning）、max 没有单位元因而拒绝。
        """
        cp = self.cp
        x = self._contig(args[0])
        shape = tuple(int(d) for d in x.shape)
        keepdims = bool(attrs.get("keepdims", False))
        keep = tuple(a for a in range(x.ndim) if a not in axes)
        rows = 1
        for a in keep:
            rows *= shape[a]
        cols = 1
        for a in axes:
            cols *= shape[a]
        target = (
            tuple(1 if i in axes else shape[i] for i in range(x.ndim))
            if keepdims
            else tuple(shape[i] for i in keep)
        )
        if cols == 0:
            if op == "reduce_max":
                raise BackendCapabilityError(
                    f"{self._LABEL}的 {op} 不能对空轴归约（形状 {x.shape}，轴 {axes}）"
                )
            if op == "reduce_sum":
                return cp.zeros(rows, dtype=x.dtype).reshape(target)
            return cp.full(rows, np.nan, dtype=x.dtype).reshape(target)
        perm = keep + tuple(axes)
        if perm != tuple(range(x.ndim)):
            x = cp.ascontiguousarray(cp.transpose(x, perm))
        flat = x.reshape(rows, cols)
        out = self._reduce_last_dim(
            node, [flat], {"axis": -1, "keepdims": False}, op, source_fn, func
        )
        return out.reshape(target)

    def _reduce_last_dim(self, node, args, attrs, op: str, source_fn, func: str):
        """归约的公共骨架：末维走块内归约，其余轴走通用路径，多轴先合并。

        sum / max / mean 只在"用哪个内核"上不同；keepdims 的形状恢复、空输入
        这几件事完全一致，所以合成一处，改动只需要改一遍。

        多个轴先**置换到末尾再合并成一个轴**（``_reduce_multi_axis``）：归约
        结果与轴顺序无关，而合并后就能复用同一份单轴内核，不需要第二套内核。
        """
        cp = self.cp
        x = args[0]
        axis = attrs.get("axis", -1)
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        if x.ndim == 0:
            raise BackendCapabilityError(f"{self._LABEL}的 {op} 需要至少一维")
        if not axes:
            raise BackendCapabilityError(
                f"{self._LABEL}的 {op} 的 axis 不能为空；要恒等请直接删掉这个节点"
            )
        norm = tuple(sorted(int(a) % x.ndim for a in axes))
        if len(set(norm)) != len(norm):
            raise BackendCapabilityError(f"{self._LABEL}的 {op} 有重复的轴: {axes}")
        if len(norm) > 1:
            return self._reduce_multi_axis(node, args, norm, attrs, op, source_fn, func)
        axis = norm[0]
        if axis != x.ndim - 1:
            return self._reduce_any_axis(node, args, axis, attrs, op)
        rows, cols = self._last_dim_2d(x, op)
        keepdims = bool(attrs.get("keepdims", False))
        target = x.shape[:-1] + ((1,) if keepdims else ())
        out = cp.empty(rows, dtype=x.dtype)
        if x.size == 0:
            return out.reshape(target)
        dtype_key = self._kind(x)
        kernel = self._kernel(
            (op, dtype_key), lambda: self.codegen.render(source_fn(), x.dtype), func
        )
        block = self._reduce_block(op, cols)
        kernel((rows,), (block,), (x, out, np.int64(rows), np.int64(cols)))
        return out.reshape(target)



    def _op_reduce_sum(self, node, args, attrs):
        return self._reduce_last_dim(
            node, args, attrs, "reduce_sum", self.codegen.reduce_sum_source,
            "reduce_sum_last",
        )



    def _op_reduce_max(self, node, args, attrs):
        return self._reduce_last_dim(
            node, args, attrs, "reduce_max", self.codegen.reduce_max_source,
            "reduce_max_last",
        )



    def _op_reduce_mean(self, node, args, attrs):
        return self._reduce_last_dim(
            node, args, attrs, "reduce_mean", self.codegen.reduce_mean_source,
            "reduce_mean_last",
        )



    def _op_argmax(self, node, args, attrs):
        """任意单轴的 argmax：输出 i32，不是浮点激活值。

        非末轴先置换到最后再复用同一个末维内核，结果按原轴序恢复；下标
        语义是**该轴上的位置**（与 numpy 一致），置换不会改变它。

        空轴时 numpy 会抛错；这里也拒绝，而不是返回一个没有定义的 ``-1``。
        """
        x = args[0]
        axis = attrs.get("axis", -1)
        if isinstance(axis, (tuple, list)):
            if len(axis) != 1:
                raise BackendCapabilityError(
                    f"{self._LABEL}的 argmax 一次只支持一个轴，收到 {axis}"
                )
            axis = axis[0]
        if x.ndim == 0:
            raise BackendCapabilityError(f"{self._LABEL}的 argmax 需要至少一维")
        axis = int(axis) % x.ndim
        keepdims = bool(attrs.get("keepdims", False))
        if axis == x.ndim - 1:
            return self._argmax_last(x, keepdims)
        shape = tuple(int(d) for d in x.shape)
        perm = tuple(i for i in range(x.ndim) if i != axis) + (axis,)
        flat = self.cp.ascontiguousarray(
            self.cp.transpose(self._contig(x), perm)
        )
        out = self._argmax_last(flat, False)
        target = (
            tuple(1 if i == axis else shape[i] for i in range(x.ndim))
            if keepdims
            else tuple(shape[i] for i in range(x.ndim) if i != axis)
        )
        return out.reshape(target)

    def _argmax_last(self, x, keepdims: bool):
        cp = self.cp
        rows, cols = self._last_dim_2d(x, "argmax")
        if cols == 0:
            raise BackendCapabilityError(
                f"{self._LABEL}的 argmax 不能对空轴归约（形状 {x.shape}）"
            )
        target = x.shape[:-1] + ((1,) if keepdims else ())
        out = cp.empty(rows, dtype=cp.int32)
        if out.size == 0:
            return out.reshape(target)
        dtype_key = self._kind(x)
        kernel = self._kernel(
            ("argmax", dtype_key),
            lambda: self.codegen.render(self.codegen.argmax_source(), x.dtype),
            "argmax_last",
        )
        block = self._reduce_block("argmax", cols)
        kernel((rows,), (block,), (x, out, np.int64(rows), np.int64(cols)))
        return out.reshape(target)

    # ------------------------------------------------------------------
    # 布局 / 类型
    # ------------------------------------------------------------------

    def _op_reshape(self, node, args, attrs):
        target = tuple(int(d) for d in attrs["shape"])
        src = self._contig(args[0])
        out = self.cp.reshape(src, target)
        # reshape 是零拷贝视图，和 src 共用同一段地址。src 的回收点早于 out 的
        # 最后一次读取，不记账就会被 arena 拿去复用（详见 ``Arena.alias``）。
        return self._shared_view(out, src)



    def _op_transpose(self, node, args, attrs):
        perm = tuple(int(p) for p in attrs["perm"])
        out = self.cp.ascontiguousarray(self.cp.transpose(args[0], perm))
        # 恒等置换不拷贝，返回的是输入的视图
        return self._shared_view(out, args[0])



    def _op_cast(self, node, args, attrs):
        # 执行骨架也会按 IR 推导的 dtype 调 ``cast`` 钩子（正常路径）；这里同样
        # 走那条实现，保证"单独测这个算子"与"整图执行"是同一条代码路径。
        return self.cast(args[0], DType(attrs["dtype"]).np_dtype)



    def _op_slice(self, node, args, attrs):
        x = args[0]
        dim = int(attrs["dim"]) % x.ndim
        start = int(attrs.get("start", 0))
        stop = attrs.get("stop", x.shape[dim])
        stop = x.shape[dim] if stop is None else int(stop)
        index = [slice(None)] * x.ndim
        index[dim] = slice(start, stop)
        # 物化：切片结果是视图，后续 kernel 要求连续内存。沿某一维整段切片
        # （如 ``x[0:2]``）本来就是连续的，``ascontiguousarray`` 会原样返回
        # 这个视图——它仍然指向 x 的地址，必须记账。
        out = self.cp.ascontiguousarray(x[tuple(index)])
        return self._shared_view(out, x)

    # ------------------------------------------------------------------
    # 索引类：拼接 / 查表
    # ------------------------------------------------------------------

    def _op_concat(self, node, args, attrs):
        """沿 axis 拼接：每个输入一段 strided copy，按段偏移写入。

        ``(outer, axis, inner)`` 视角把拼接变成"把这一段搬到 axis 的某个偏移处"，
        因此不需要一张"输出下标 -> 输入编号"的表，也不用把输入个数写成
        内核参数。非拼接维的一致性由 IR 的 infer 保证，这里只需取自己的长度。
        """
        cp = self.cp
        ndim = args[0].ndim
        axis = int(attrs.get("axis", 0)) % ndim
        inner = int(np.prod(args[0].shape[axis + 1 :])) if axis + 1 < ndim else 1
        out_axis = sum(int(a.shape[axis]) for a in args)

        out_shape = list(args[0].shape)
        out_shape[axis] = out_axis
        out = cp.empty(tuple(out_shape), dtype=args[0].dtype)
        if out.size == 0:
            return out

        dtype_key = self._kind(args[0])
        kernel = self._kernel(
            ("concat", dtype_key),
            lambda: self.codegen.render(self.codegen.concat_copy_source(), args[0].dtype),
            "concat_copy",
        )
        offset = 0
        for a in args:
            # 每个输入都物化成连续内存：copy kernel 按 C 布局线性索引
            a = cp.ascontiguousarray(a)
            total = int(a.size)
            seg = int(a.shape[axis])
            if total:
                block, grid = self._grid("concat", total)
                kernel(
                    (grid,),
                    (block,),
                    (
                        a,
                        out,
                        np.int64(seg),
                        np.int64(inner),
                        np.int64(out_axis),
                        np.int64(offset),
                        np.int64(total),
                    ),
                )
            offset += seg
        return out



    def _op_embedding(self, node, args, attrs):
        """按 ids 查表；表按 ``(vocab, row)`` 视角处理，尾随维任意。

        越界下标不能在内核里抛 Python 异常，所以内核把它记进一个设备标量，
        由 ``_check_ids`` 在主机侧转成 ``BackendCapabilityError``。
        内核里同时做 clamp，是为了走向这条报错路径时不发生非法访存。
        """
        cp = self.cp
        table, ids = args
        vocab = int(table.shape[0])
        tail = tuple(int(d) for d in table.shape[1:])
        row = int(np.prod(tail)) if tail else 1
        ids_flat = cp.ascontiguousarray(ids).reshape(-1)
        out = cp.empty(tuple(ids.shape) + tail, dtype=table.dtype)
        total = int(ids_flat.size) * row
        if total == 0:
            return out

        dtype_key = self._kind(table)
        kernel = self._kernel(
            ("embedding", str(ids.dtype), dtype_key),
            lambda: self.codegen.render(self.codegen.embedding_source(ids.dtype), table.dtype),
            "embedding",
        )
        err = cp.zeros(1, dtype=cp.int32)
        block, grid = self._grid("embedding", total)
        kernel(
            (grid,),
            (block,),
            (
                cp.ascontiguousarray(table),
                ids_flat,
                out,
                np.int64(ids_flat.size),
                np.int64(row),
                np.int64(vocab),
                err,
                np.int64(total),
            ),
        )
        self._check_ids(err, "embedding", f"[0, {vocab})")
        return out



    def _op_embedding_backward(self, node, args, attrs):
        """查表反向：把每条 dy 散射累加回对应表行。

        重复下标必须累加（同一个 token 在一个 batch 里出现多次是常态），
        这里是 atomicAdd 存在的最直接理由。
        """
        cp = self.cp
        dy, ids = args
        shape = tuple(int(d) for d in attrs["table_shape"])
        vocab = shape[0]
        tail = shape[1:]
        row = int(np.prod(tail)) if tail else 1
        expected = tuple(ids.shape) + tail
        if tuple(dy.shape) != expected:
            raise BackendCapabilityError(
                f"embedding_backward 的 dy 形状应为 {expected}，收到 {dy.shape}"
            )
        ids_flat = cp.ascontiguousarray(ids).reshape(-1)
        dtable = cp.zeros(shape, dtype=dy.dtype)
        total = int(ids_flat.size) * row
        if total == 0:
            return dtable

        dtype_key = self._kind(dy)
        kernel = self._kernel(
            ("embedding_backward", str(ids.dtype), dtype_key),
            lambda: self.codegen.render(self.codegen.embedding_backward_source(ids.dtype), dy.dtype),
            "embedding_backward",
        )
        err = cp.zeros(1, dtype=cp.int32)
        block, grid = self._grid("embedding_backward", total)
        kernel(
            (grid,),
            (block,),
            (
                cp.ascontiguousarray(dy),
                ids_flat,
                dtable,
                np.int64(ids_flat.size),
                np.int64(row),
                np.int64(vocab),
                err,
                np.int64(total),
            ),
        )
        self._check_ids(err, "embedding_backward", f"[0, {vocab})")
        return dtable

    # ------------------------------------------------------------------
    # 卷积 / 池化的反向
    # ------------------------------------------------------------------

    def _op_pad(self, node, args, attrs):
        """沿单个维度两侧补零：等价于"把整块搬到偏移 before 处"。

        直接复用 concat 的 strided copy——两者都是"输出比输入大、把输入段
        按偏移写进去"，没必要为 pad 再写一个内核。
        """
        cp = self.cp
        x = self._contig(args[0])
        dim = int(attrs["dim"]) % x.ndim
        before = int(attrs.get("before", 0))
        after = int(attrs.get("after", 0))
        if before < 0 or after < 0:
            raise BackendCapabilityError(
                f"pad 的 before/after 必须 >= 0，收到 {before} 与 {after}"
            )
        out_shape = list(x.shape)
        out_shape[dim] = int(out_shape[dim]) + before + after
        out = cp.zeros(tuple(out_shape), dtype=x.dtype)
        total = int(x.size)
        if total == 0:
            return out

        inner = int(np.prod(x.shape[dim + 1 :])) if dim + 1 < x.ndim else 1
        dtype_key = self._kind(x)
        kernel = self._kernel(
            ("concat", dtype_key),
            lambda: self.codegen.render(self.codegen.concat_copy_source(), x.dtype),
            "concat_copy",
        )
        block, grid = self._grid("concat", total)
        kernel(
            (grid,),
            (block,),
            (
                x,
                out,
                np.int64(x.shape[dim]),
                np.int64(inner),
                np.int64(out_shape[dim]),
                np.int64(before),
                np.int64(total),
            ),
        )
        return out



    def _op_maxpool2d_backward(self, node, args, attrs):
        """最大池化反向：dy 散射到窗口 argmax 处。

        并列最大值必须取**第一个**（行优先），与参考实现严格一致；
        重叠窗口可能命中同一个输入格，所以累加要走 atomicAdd。
        """
        cp = self.cp
        x = self._contig(args[0])
        dy = cp.ascontiguousarray(args[1])
        if x.ndim != 4:
            raise BackendCapabilityError(
                f"{self._LABEL}的 maxpool2d_backward 要求 NCHW 秩 4，收到 {x.shape}"
            )
        (kh, kw), (sh, sw), (ph, pw) = self._pool_geometry(attrs, "maxpool2d_backward")
        ceil_mode, _ = _pool_flags(attrs, "maxpool2d_backward")
        n, c, h, w = (int(v) for v in x.shape)
        ho = _pool_out_dim(h, kh, sh, ph, ceil_mode)
        wo = _pool_out_dim(w, kw, sw, pw, ceil_mode)
        if ho <= 0 or wo <= 0:
            raise BackendCapabilityError(
                f"maxpool2d_backward 的窗口尺寸非法：输入 {h}x{w}、核 {kh}x{kw}、"
                f"padding {(ph, pw)}、stride {(sh, sw)}、ceil_mode={ceil_mode}"
            )
        if tuple(dy.shape) != (n, c, ho, wo):
            raise BackendCapabilityError(
                f"maxpool2d_backward 的 dy 形状应为 {(n, c, ho, wo)}，收到 {dy.shape}"
            )

        dx = cp.zeros(x.shape, dtype=x.dtype)
        total = int(dy.size)
        if total == 0:
            return dx

        dtype_key = self._kind(x)
        kernel = self._kernel(
            ("maxpool2d_backward", dtype_key),
            lambda: self.codegen.render(self.codegen.maxpool2d_backward_source(), x.dtype),
            "maxpool2d_backward",
        )
        block, grid = self._grid("maxpool2d_backward", total)
        kernel(
            (grid,),
            (block,),
            (
                x,
                dy,
                dx,
                np.int32(c), np.int32(h), np.int32(w),
                np.int32(kh), np.int32(kw), np.int32(ho), np.int32(wo),
                np.int32(sh), np.int32(sw), np.int32(ph), np.int32(pw),
                np.int64(total),
            ),
        )
        return dx



    def _op_conv2d_input_grad(self, node, args, attrs):
        """卷积对输入的梯度（gather 形式）：每个 dx 元素由一个线程独占。

        给定 (ih, iw)，贡献它的输出格必须满足 ``ih + PH - ki*DH`` 能被 stride
        整除，所以不需要 atomicAdd，也不需要物化 col2im 的中间矩阵。
        代价是每个线程要遍历 O*KH*KW——第一版先要正确与确定性，性能留给
        后来的 GEMM 路径。
        """
        cp = self.cp
        dy = self._contig(args[0])
        weight = cp.ascontiguousarray(args[1])
        groups = int(attrs.get("groups", 1))
        if groups < 1:
            raise BackendCapabilityError(
                f"conv2d_input_grad 的 groups 必须 >= 1，得到 {groups}"
            )
        shape = tuple(int(d) for d in attrs["input_shape"])
        if len(shape) != 4:
            raise BackendCapabilityError(
                f"conv2d_input_grad 的 input_shape 必须是 NCHW 四元组，收到 {shape}"
            )
        n, c, h, w = shape
        o, cw, kh, kw = (int(v) for v in weight.shape)
        if c % groups != 0 or o % groups != 0:
            raise BackendCapabilityError(
                f"conv2d_input_grad 的通道数必须能被 groups={groups} 整除，收到 C={c}、O={o}"
            )
        if cw * groups != c:
            raise BackendCapabilityError(
                f"conv2d_input_grad 的权重通道 {cw} * groups={groups} 与 input_shape 的通道 {c} 不一致"
            )
        (sh, sw), (ph, pw), (dil_h, dil_w) = self._conv_geometry(
            attrs, "conv2d_input_grad"
        )
        ho = (h + 2 * ph - dil_h * (kh - 1) - 1) // sh + 1
        wo = (w + 2 * pw - dil_w * (kw - 1) - 1) // sw + 1
        if tuple(dy.shape) != (n, o, ho, wo):
            raise BackendCapabilityError(
                f"conv2d_input_grad 的 dy 形状应为 {(n, o, ho, wo)}，收到 {dy.shape}"
            )

        dx = cp.empty(shape, dtype=dy.dtype)
        total = n * c * h * w
        if total == 0:
            return dx

        dtype_key = self._kind(dy)
        kernel = self._kernel(
            ("conv2d_input_grad", dtype_key),
            lambda: self.codegen.render(self.codegen.conv2d_input_grad_source(), dy.dtype),
            "conv2d_input_grad",
        )
        if groups == 1:
            groups_desc = ((dy, weight, dx, c),)
            o_g = o
        else:
            # 每组一份 (dy, weight, dx) 视图：通道数换成 C/groups、O/groups，
            # 内核本身不需要知道分组——它只看到一组普通的卷积。
            o_g = o // groups
            groups_desc = [
                (
                    cp.ascontiguousarray(dy[:, gi * o_g : (gi + 1) * o_g]),
                    weight[gi * o_g : (gi + 1) * o_g],
                    cp.empty((n, cw, h, w), dtype=dy.dtype),
                    cw,
                )
                for gi in range(groups)
            ]
        for dy_g, w_g, dx_g, c_g in groups_desc:
            total_g = n * c_g * h * w
            block, grid = self._grid("conv2d_input_grad", total_g)
            kernel(
                (grid,),
                (block,),
                (
                    dy_g,
                    w_g,
                    dx_g,
                    np.int32(c_g), np.int32(h), np.int32(w),
                    np.int32(o_g),
                    np.int32(kh), np.int32(kw), np.int32(ho), np.int32(wo),
                    np.int32(sh), np.int32(sw), np.int32(ph), np.int32(pw),
                    np.int32(dil_h), np.int32(dil_w),
                    np.int64(total_g),
                ),
            )
        if groups > 1:
            for gi in range(groups):
                dx[:, gi * cw : (gi + 1) * cw] = groups_desc[gi][2]
        return dx



    def _op_conv2d_weight_grad(self, node, args, attrs):
        """卷积对权重的梯度（gather 形式）：每个 dw 元素独占一个线程。

        遍历整个 batch 与所有输出格，量级与前向卷积相同。gather 同样避免
        了浮点 atomicAdd，结果逐位确定。
        """
        cp = self.cp
        x = self._contig(args[0])
        dy = cp.ascontiguousarray(args[1])
        groups = int(attrs.get("groups", 1))
        if groups < 1:
            raise BackendCapabilityError(
                f"conv2d_weight_grad 的 groups 必须 >= 1，得到 {groups}"
            )
        wshape = tuple(int(d) for d in attrs["weight_shape"])
        if len(wshape) != 4:
            raise BackendCapabilityError(
                f"conv2d_weight_grad 的 weight_shape 必须是四元组，收到 {wshape}"
            )
        if x.ndim != 4 or dy.ndim != 4:
            raise BackendCapabilityError(
                f"conv2d_weight_grad 要求 x/dy 都是 NCHW 秩 4，收到 {x.shape} 与 {dy.shape}"
            )
        n, c, h, w = (int(v) for v in x.shape)
        o, cw, kh, kw = wshape
        if c % groups != 0 or o % groups != 0:
            raise BackendCapabilityError(
                f"conv2d_weight_grad 的通道数必须能被 groups={groups} 整除，收到 C={c}、O={o}"
            )
        if cw * groups != c:
            raise BackendCapabilityError(
                f"conv2d_weight_grad 的 x 通道 {c} 与 weight_shape 通道 {cw} * groups={groups} 不一致"
            )
        (sh, sw), (ph, pw), (dil_h, dil_w) = self._conv_geometry(
            attrs, "conv2d_weight_grad"
        )
        ho = (h + 2 * ph - dil_h * (kh - 1) - 1) // sh + 1
        wo = (w + 2 * pw - dil_w * (kw - 1) - 1) // sw + 1
        if tuple(dy.shape) != (n, o, ho, wo):
            raise BackendCapabilityError(
                f"conv2d_weight_grad 的 dy 形状应为 {(n, o, ho, wo)}，收到 {dy.shape}"
            )

        dw = cp.empty(wshape, dtype=x.dtype)
        total = o * c * kh * kw
        if total == 0:
            return dw

        dtype_key = self._kind(x)
        kernel = self._kernel(
            ("conv2d_weight_grad", dtype_key),
            lambda: self.codegen.render(self.codegen.conv2d_weight_grad_source(), x.dtype),
            "conv2d_weight_grad",
        )
        if groups == 1:
            groups_desc = [(x, dy, dw, c, o)]
        else:
            # 每组只拥有自己那 cw 个输入通道与 og 个输出通道，
            # 因此把两边的通道切片交给同一个内核即可。
            og = o // groups
            groups_desc = [
                (
                    cp.ascontiguousarray(x[:, gi * cw : (gi + 1) * cw]),
                    cp.ascontiguousarray(dy[:, gi * og : (gi + 1) * og]),
                    dw[gi * og : (gi + 1) * og],
                    cw,
                    og,
                )
                for gi in range(groups)
            ]
        for x_g, dy_g, dw_g, c_g, o_g in groups_desc:
            total_g = o_g * c_g * kh * kw
            block, grid = self._grid("conv2d_weight_grad", total_g)
            kernel(
                (grid,),
                (block,),
                (
                    x_g,
                    dy_g,
                    dw_g,
                    np.int32(n), np.int32(c_g), np.int32(h), np.int32(w),
                    np.int32(o_g),
                    np.int32(kh), np.int32(kw), np.int32(ho), np.int32(wo),
                    np.int32(sh), np.int32(sw), np.int32(ph), np.int32(pw),
                    np.int32(dil_h), np.int32(dil_w),
                    np.int64(total_g),
                ),
            )
        return dw



    def _check_ids(self, err, op: str, allowed: str) -> None:
        """把内核记下的越界标志转成后端能力错误。

        捕获期不能读回：CUDA Graph 里不允许出现设备到主机的同步点。所以
        捕获期只把 ``(标量, 算子, 范围)`` 登记进 ``_capture_guards``，由
        ``CudaGraphPlan`` 在每次 replay 同步之后复查。捕获轮的下标与预热轮
        相同，越界在那里已经抛过；登记的那块标量就是图里写死的地址，
        replay 用新下标执行时会重新置位，因此捕获后的越界同样会报错。
        """
        if self._capturing:
            self._capture_guards.append((err, op, allowed))
            return
        if int(err.item()) != 0:
            self._raise_bad_ids(op, allowed)



    def _raise_bad_ids(self, op: str, allowed: str) -> None:
        """越界下标统一出口：普通执行与 Graph replay 复查共用一条消息。"""
        raise BackendCapabilityError(
            f"{self._LABEL}的 {op} 收到越界下标：ids 必须落在 {allowed} 之内。"
            "越界是宿主侧错误，内核只做 clamp 以保证不非法访存"
        )

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @classmethod
    def _pool_geometry(cls, attrs: dict, op: str):
        """池化的 ``(kernel, stride, padding)``，stride 缺省等于 kernel。

        与 ir/ops.py 的 ``_pool_geometry`` 逐条对齐（包括报错条件），
        这样"参考实现接受、CUDA 拒绝"的差异只会出现在真正的实现边界上。
        """
        if "kernel" not in attrs:
            raise BackendCapabilityError(f"{cls._LABEL}的 {op} 缺少 kernel 属性")
        kh, kw = _pair(attrs["kernel"])
        raw_stride = attrs.get("stride")
        sh, sw = _pair(attrs["kernel"] if raw_stride is None else raw_stride)
        ph, pw = _pair(attrs.get("padding", (0, 0)))
        if kh < 1 or kw < 1 or sh < 1 or sw < 1:
            raise BackendCapabilityError(
                f"{op} 的 kernel/stride 必须 >= 1，得到 {(kh, kw)} 与 {(sh, sw)}"
            )
        if ph < 0 or pw < 0:
            raise BackendCapabilityError(f"{op} 的 padding 必须 >= 0，得到 {(ph, pw)}")
        return (kh, kw), (sh, sw), (ph, pw)

    @classmethod
    def _adaptive_output_size(cls, attrs: dict, op: str):
        """自适应池化的 ``(OH, OW)``，与 ``ir/ops.py`` 的解析规则一致。"""
        if "output_size" not in attrs:
            raise BackendCapabilityError(f"{cls._LABEL}的 {op} 缺少 output_size 属性")
        raw = attrs["output_size"]
        if isinstance(raw, int) and not isinstance(raw, bool):
            oh = ow = int(raw)
        elif isinstance(raw, (tuple, list)) and len(raw) in (1, 2):
            oh, ow = int(raw[0]), int(raw[-1])
        else:
            raise BackendCapabilityError(
                f"{op} 的 output_size 必须是 int 或长度 1/2 的序列，收到 {raw!r}"
            )
        if oh < 1 or ow < 1:
            raise BackendCapabilityError(
                f"{op} 的 output_size 必须 >= 1，得到 {(oh, ow)}"
            )
        return oh, ow

    @staticmethod
    def _conv_geometry(attrs: dict, op: str):
        """卷积的 ``(stride, padding, dilation)``，三者都可缺省。

        与 ir/ops.py 的 ``_conv_geometry`` 同名同义：stride 缺省 1、padding
        缺省 0、dilation 缺省 1；反向算子用同一组几何，才能和前向严格互逆。
        """
        sh, sw = _pair(attrs.get("stride", (1, 1)))
        ph, pw = _pair(attrs.get("padding", (0, 0)))
        dh, dw = _pair(attrs.get("dilation", (1, 1)))
        if sh < 1 or sw < 1 or dh < 1 or dw < 1:
            raise BackendCapabilityError(
                f"{op} 的 stride/dilation 必须 >= 1，得到 {(sh, sw)} 与 {(dh, dw)}"
            )
        if ph < 0 or pw < 0:
            raise BackendCapabilityError(f"{op} 的 padding 必须 >= 0，得到 {(ph, pw)}")
        return (sh, sw), (ph, pw), (dh, dw)



    def _last_dim_2d(self, x, op: str):
        if x.ndim == 0:
            raise BackendCapabilityError(f"{op} 不支持 0 维输入")
        rows = int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1
        return rows, int(x.shape[-1])

    def _norm_layout(self, x, axes, name: str):
        """把归一化轴挪到最后，返回布局工具。

        ``axes`` 是任意轴集合（layernorm / rmsnorm 的 ``axes`` 属性）：参考实现
        按 numpy 广播算出 ``(x - mean) / std * gamma + beta``，后端不做第二份
        数值路径，只在**布局**上做等价变换——归一化轴正好是最后若干维时取一个
        零拷贝的二维视图，否则按 ``[keep..., norm...]`` 置换一次再转连续。

        返回 ``(prep, alloc, finish, rows, cols, norm_shape, norm_axes)``：

        - ``prep``：把与 ``x`` 同形的张量（``x`` 自己、``residual``）变成
          ``(rows, cols)``；
        - ``alloc``：分配输出，返回 ``(原形状数组, (rows, cols) 的写入视图)``。
          归一化轴就是最后若干维时按 ``x`` 的形状分配——与改造前只有一份缓冲，
          峰值不变；
        - ``finish``：把写好的二维视图换回原形状与原轴序。
        """
        cp = self.cp
        rank = x.ndim
        if rank == 0:
            raise BackendCapabilityError(f"{name} 不支持 0 维输入")
        norm = tuple(sorted(int(a) % rank for a in axes))
        if not norm:
            raise BackendCapabilityError(f"{name} 的 axes 不能为空")
        if len(set(norm)) != len(norm):
            raise BackendCapabilityError(f"{name} 的 axes 有重复轴: {axes}")
        shape = tuple(int(dim) for dim in x.shape)
        keep = tuple(a for a in range(rank) if a not in norm)
        rows = 1
        for a in keep:
            rows *= shape[a]
        cols = 1
        for a in norm:
            cols *= shape[a]
        norm_shape = tuple(shape[a] for a in norm)
        if norm == tuple(range(len(keep), rank)):
            def prep(t):
                return self._contig(t).reshape(rows, cols)

            def alloc():
                out = cp.empty_like(x)
                return out, out.reshape(rows, cols)

            def finish(full, flat):
                return full

            return prep, alloc, finish, rows, cols, norm_shape, norm
        perm = keep + norm
        permuted = tuple(shape[a] for a in perm)
        inverse = tuple(perm.index(a) for a in range(rank))

        def prep(t):
            return cp.ascontiguousarray(cp.transpose(t, perm)).reshape(rows, cols)

        def alloc():
            out = cp.empty((rows, cols), dtype=x.dtype)
            return out, out

        def finish(full, flat):
            return cp.ascontiguousarray(cp.transpose(flat.reshape(permuted), inverse))

        return prep, alloc, finish, rows, cols, norm_shape, norm

    def _norm_param(self, p, cols: int, x_shape: tuple, norm_axes: tuple,
                    name: str, op: str):
        """归一化参数 ``gamma`` / ``beta`` 统一展平成 ``(cols,)``。

        内核一行一份参数，所以参数必须"对每一行都一样"：右对齐到 ``x`` 的秩
        后，非归一化轴全为 1，归一化轴等于 ``x`` 的对应维或 1。归一化轴上的 1
        是广播——沿该轴复用同一个系数，这里用 ``broadcast_to`` 展开成与 ``x``
        同形再展平，因此结果与 numpy 广播逐元素相同。

        非归一化轴上不是 1 的形态（例如 ``gamma`` 随 batch 变化）会在运行时逐行
        不同，内核装不下，显式拒绝。
        """
        cp = self.cp
        shape = tuple(int(dim) for dim in p.shape)
        if len(shape) > len(x_shape):
            raise BackendCapabilityError(
                f"{op} 的 {name} 的秩不能超过 x，收到 {shape} 与 {x_shape}"
            )
        padded = (1,) * (len(x_shape) - len(shape)) + shape
        if any(padded[a] != 1 for a in range(len(x_shape)) if a not in norm_axes):
            raise BackendCapabilityError(
                f"{op} 的 {name} 不能随非归一化维变化，收到 {shape}"
                f"（x 为 {x_shape}，归一化轴为 {norm_axes}）"
            )
        src = tuple(padded[a] for a in norm_axes)
        target = tuple(x_shape[a] for a in norm_axes)
        if any(s != 1 and s != t for s, t in zip(src, target)):
            raise BackendCapabilityError(
                f"{op} 的 {name} 在归一化轴上必须与 x 相同或为 1，"
                f"收到 {src}，x 的归一化维为 {target}"
            )
        view = p if shape == src else cp.reshape(p, src)
        if src != target:
            view = cp.ascontiguousarray(cp.broadcast_to(view, target))
        return cp.reshape(view, (cols,))

    @staticmethod
    def _elem_strides(shape: tuple, out_shape: tuple) -> List[int]:
        """按 C 连续布局、右对齐到 out_shape 后，逐维的元素步长。

        被广播的维（源形状为 1）步长置 0，这样索引会一直停在同一个元素上。
        """
        pad = len(out_shape) - len(shape)
        padded = (1,) * pad + tuple(shape)
        raw = [0] * len(padded)
        acc = 1
        for d in range(len(padded) - 1, -1, -1):
            raw[d] = acc
            acc *= padded[d]
        return [0 if dim == 1 else raw[d] for d, dim in enumerate(padded)]

    @staticmethod
    def _collapse_dims(shape: tuple, sa: List[int], sb: List[int]) -> List[tuple]:
        """把相邻且两侧都连续的维度合并，返回 [(dim, stride_a, stride_b)]（由外到内）。

        合并的前提是两侧的 stride 都恰好等于"内层总长度"，否则合并会改变索引。
        """
        groups: List[tuple] = []
        for d in range(len(shape) - 1, -1, -1):
            if shape[d] == 1:
                continue
            if groups:
                inner_dim, inner_a, inner_b = groups[-1]
                if sa[d] == inner_a * inner_dim and sb[d] == inner_b * inner_dim:
                    groups[-1] = (inner_dim * shape[d], inner_a, inner_b)
                    continue
            groups.append((shape[d], sa[d], sb[d]))
        groups.reverse()
        return groups



    def _kind(self, a) -> str:
        """内核源码与缓存的 dtype 键（``f32`` / ``f16``）。

        只区分这两种：其它 dtype 在 ``invoke`` 就已拒绝，走不到这里。
        """
        return "f16" if a.dtype == self.cp.float16 else "f32"



    def _one(self):
        """设备上的常数 1（形状 (1,)），供 softmax 的默认缩放使用。"""
        return self._one_dev

