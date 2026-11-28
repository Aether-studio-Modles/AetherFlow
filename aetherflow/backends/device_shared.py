"""设备后端的共享层：算子面声明、索引参数约定、纯函数 helper 与内核句柄。

这些定义不依赖任何实例状态，被 ``device_backend`` 与各算子族 mixin 共同引用。
拆出来是为了让 mixin 文件能导入它们，而不必反向导入 ``device_backend`` 造成循环。
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Tuple


from ..ir.graph import Graph, Node
from ..ir.types import DType
from ..lower.elementwise import (
    CHAIN_OPS,
    channel_broadcast,
)
from ..lower.kernels import BINARY_OP
from ..lower.plan import (
    CONV_CB_MAX,
    CONV_ROC_MAX,
    CONV_TILE_OPTIONS,
)
from .base import BackendCapabilityError


#: 设备端还没有 fp8 的存储容器与内核：bf16 走 uint16 位模式，fp8 连位模式
#: 都还没接。参考语义已覆盖（runtime/fp8.py），设备端遇到这些类型必须显式
#: 拒绝——否则会被 ``np_dtype`` 的 f32 载体悄悄当成 f32 计算。
#: 放在共享层是因为两条路径都要用它：逐节点的 ``invoke`` 与计划构建期的
#: ``_planned_kernel``（后者预绑定内核后就不再经过 ``invoke`` 的检查）。
FP8_DTYPES: FrozenSet[DType] = frozenset({DType.F8E4M3, DType.F8E5M2})


# 同一批自研内核模板（NVRTC / hipRTC 各编译一份）承接的算子面。它是声明，
# 与当前接入了哪家硬件无关；各后端在自己的文件里把它取成自己的名字。
# 存在的唯一理由：**不建任何设备上下文也能读到**。
GPU_OPS: FrozenSet[str] = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "where",
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
        "sign",
        "round",
        "floor",
        "ceil",
        "constant",
        "broadcast_to",
        "matmul",
        "quantize_i8",
        "dequantize_i8",
        "fake_quantize_i8",
        "fake_quantize_i8_lsq",
        "linear_i8",
        "linear_i8a8",
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
        "topk_values",
        "topk_indices",
        "topk_mask",
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
        "gather",
        "gather_backward",
        "cross_entropy",
        "cross_entropy_backward",
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
    "gather": frozenset({1}),
    "gather_backward": frozenset({1}),
    "cross_entropy": frozenset({1}),
    "cross_entropy_backward": frozenset({1}),
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
    # W8A8：激活与权重都是 i8 数据，scale / bias 仍是 f32。
    "linear_i8a8": frozenset({0, 2}),
    # 整数取模：两侧都是**整数数据**，不是下标。
    "mod": frozenset({0, 1}),
}

# 整数下标允许的 dtype。与 ir/ops.py 的 infer 保持一致：i32 是前端导入的
# 常态，i8 是压缩标签 / 量化权重的形态。
_INDEX_DTYPES = ("int32", "int8")

# 允许 bool 输入的算子。比较算子两侧都可以是 bool（``ne(mask, 0)`` 这类
# "掩码再取反"的写法很常见），它们的设备存储是 1 字节 ``unsigned char``，
# 走独立的 ``BOOL_DIALECT``。算术算子不在此列：true + true = 2 再归一成 1
# 没有意义，IR 的 verifier 也已拒绝。
_BOOL_OPS: FrozenSet[str] = frozenset(
    {"eq", "ne", "lt", "le", "gt", "ge", "where"}
)

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

# W8A8 的矩阵乘：i8 输入、int32 累加、按行 / 按列缩放。累加必须留在整数域，
# 这正是它和 ``linear_i8``（反量化后走 f32 GEMM）的区别——后者的累加在浮点。
# 两份源码只差“怎么算 4 路点积”：CUDA 走 ``__dp4a``（sm_61+，V100 是 70），
# HIP 走标量乘加。两者都是精确整数运算，结果逐位相同，所以跨后端对拍成立。
_LINEAR_I8A8_SRC = r"""
extern "C" __global__ void af_linear_i8a8(
    const signed char* __restrict__ x,
    const float* __restrict__ xs,
    const signed char* __restrict__ w,
    const float* __restrict__ ws,
    float* __restrict__ out,
    long long m, long long k, long long n) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long total = m * n;
  if (idx >= total) return;
  long long row = idx / n;
  long long col = idx - row * n;
  const signed char* xr = x + row * k;
  const signed char* wc = w + col;
  int acc = 0;
  for (long long kk = 0; kk < k; ++kk) {
    acc += (int)xr[kk] * (int)wc[kk * n];
  }
  out[idx] = (float)acc * xs[row] * ws[col];
}
"""

_LINEAR_I8A8_DP4A_SRC = r"""
extern "C" __global__ void af_linear_i8a8(
    const signed char* __restrict__ x,
    const float* __restrict__ xs,
    const signed char* __restrict__ w,
    const float* __restrict__ ws,
    float* __restrict__ out,
    long long m, long long k, long long n) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long total = m * n;
  if (idx >= total) return;
  long long row = idx / n;
  long long col = idx - row * n;
  const signed char* xr = x + row * k;
  const signed char* wc = w + col;
  int acc = 0;
  long long kk = 0;
  for (; kk + 4 <= k; kk += 4) {
    const signed char* xp = xr + kk;
    const signed char* wp = wc + kk * n;
    int xv = (int)((unsigned int)(unsigned char)xp[0]
                 | ((unsigned int)(unsigned char)xp[1] << 8)
                 | ((unsigned int)(unsigned char)xp[2] << 16)
                 | ((unsigned int)(unsigned char)xp[3] << 24));
    int wv = (int)((unsigned int)(unsigned char)wp[0]
                 | ((unsigned int)(unsigned char)wp[n] << 8)
                 | ((unsigned int)(unsigned char)wp[2 * n] << 16)
                 | ((unsigned int)(unsigned char)wp[3 * n] << 24));
    acc = __dp4a(xv, wv, acc);
  }
  for (; kk < k; ++kk) acc += (int)xr[kk] * (int)wc[kk * n];
  out[idx] = (float)acc * xs[row] * ws[col];
}
"""


def _step_params(node: Node) -> tuple:
    """逐元素链一步的编译期常量参数。

    - ``clamp``：``(min, max)``，缺省一侧为 ``None``；
    - ``step`` / ``spike``：``(threshold,)``，默认值分别与 IR 的参考实现一致
      （0.0 / 1.0，见 ``ir/ops.py``）。

    IR 的 infer 已经保证 clamp 至少一侧存在且 ``min <= max``，所以这里不再
    重复校验；参数会进源码缓存键，边界/阈值不同的两个节点不会共用内核。
    """
    name = node.op.name
    if name == "clamp":
        return (node.attrs.get("min"), node.attrs.get("max"))
    if name == "step":
        return (node.attrs.get("threshold", 0.0),)
    if name == "spike":
        return (node.attrs.get("threshold", 1.0),)
    return ()


def _chainable(node: Node) -> bool:
    """这个节点能不能作为逐元素链的一环（不看它与邻居的关系）。"""
    name = node.op.name
    if name not in CHAIN_OPS:
        return False
    arity = 2 if name in BINARY_OP else 1
    if len(node.inputs) != arity:
        return False
    out_type = node.output.type
    if out_type.dtype not in (DType.F32, DType.F16, DType.BF16):
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
            # 逐通道广播 (1,C,1,1) / (C,1,1)：BN 的常态，索引可由静态形状
            # 完全确定，因此可以融（见 lower.elementwise.channel_broadcast）。
            if out_type.is_static and channel_broadcast(t.shape, out_type.shape):
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
        while j + 1 < len(nodes) and _chainable(nodes[j]) and _chainable(nodes[j + 1]):
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


def _pool_out_dim(size: int, kernel: int, stride: int, padding: int, ceil_mode: bool) -> int:
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
    return shape[-2:] == (int(s_len), int(t_len)) and all(dim == 1 for dim in shape[:-2])


def _mask_target_shape(mask_shape, keep: int) -> Optional[Tuple[int, ...]]:
    """去掉 ``mask`` 多余前导维后的目标形状；前导维不全是 1 时返回 ``None``。

    只做形状推导，不碰数组本身：调用方据此 ``reshape`` 降维。之所以不用
    ``mask[0]``，是因为天数（Iluvatar）的 ``DeviceArray.__getitem__`` 只接受
    切片、拒绝降维索引，整数索引会抛 ``TypeError``；在 numpy / cupy 上
    ``reshape`` 与降维索引等价，语义因此不分叉。
    """
    shape = tuple(int(dim) for dim in mask_shape)
    extra = len(shape) - keep
    if extra <= 0:
        return shape
    if any(dim != 1 for dim in shape[:extra]):
        return None
    return shape[extra:]


# conv2d 分块核在 W 方向的线程数、shared 预算与分块阈值都在 ``lower/plan.py``：
# 它们是与设备无关的纯算术，放在后端里既挡住了离线单测，也容易和 HIP 侧分叉。

_CONV_IMPLS = ("auto", "im2col", "tiled", "implicit", "winograd")

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
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "where",
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
        "topk_values",
        "topk_indices",
        "topk_mask",
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
