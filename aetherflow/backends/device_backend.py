"""设备后端主体：执行计划缓存、调优旋钮与厂商钩子的公共实现。

算子实现按族拆在 ``device_<family>.py`` 的 mixin 里；本模块保留类属性、
构造、执行入口、调优与几何 helper。模块级常量与纯函数在 ``device_shared``。
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple
from ..ir.graph import Graph as Graph, Node
from ..ir.types import DType

#: 设备端还没有 fp8 的 1 字节存储容器与专用内核（见 device_shared.FP8_DTYPES）。
#: IR 里 fp8 的载体是 f32，只有白名单算子（``FP8_OPS``：cast 与两个线性层）
#: 承接；其余算子遇到这些类型必须显式拒绝，否则会被 f32 载体悄悄当成 f32 算。
#: 参考语义在 runtime/fp8.py。
from ..lower.elementwise import (
    CHAIN_OPS as CHAIN_OPS,
    ElemInput as ElemInput,
    ElemStep as ElemStep,
    apply_op as apply_op,
    build_chain as build_chain,
    channel_broadcast as channel_broadcast,
)
from ..lower.kernel import Index as Index, Num as Num
from ..lower.kernels import BINARY_OP as BINARY_OP
from ..lower.plan import (
    BLOCK_DEFAULT,
    BLOCK_OPTIONS,
    CONV_CB_MAX as CONV_CB_MAX,
    CONV_ROC_MAX as CONV_ROC_MAX,
    CONV_SHARED_LIMIT as CONV_SHARED_LIMIT,
    CONV_TILED_MAX_K as CONV_TILED_MAX_K,
    CONV_TILE_OPTIONS as CONV_TILE_OPTIONS,
    CONV_WINOGRAD_FUSED_MIN_C as CONV_WINOGRAD_FUSED_MIN_C,
    CONV_WINOGRAD_FUSED_RESIDENT_MIN_C as CONV_WINOGRAD_FUSED_RESIDENT_MIN_C,
    CONV_WINOGRAD_TILE as CONV_WINOGRAD_TILE,
    REDUCE_TILE_X as REDUCE_TILE_X,
    SHARED_LIMIT as SHARED_LIMIT,
    VEC4_ALIGN as VEC4_ALIGN,
    attention_block as attention_block,
    auto_block,
    cdiv as cdiv,
    check_block,
    conv_implicit_fits as conv_implicit_fits,
    conv_implicit_preferred as conv_implicit_preferred,
    conv_oc_tile as conv_oc_tile,
    conv_tiled_preferred as conv_tiled_preferred,
    conv_winograd_fused_preferred as conv_winograd_fused_preferred,
    conv_winograd_preferred as conv_winograd_preferred,
    conv_winograd_supported as conv_winograd_supported,
    grid_for,
    plan_conv_implicit as plan_conv_implicit,
    plan_conv_tiled as plan_conv_tiled,
    plan_csr_scan as plan_csr_scan,
    plan_csr_scan_enabled as plan_csr_scan_enabled,
    plan_elementwise as plan_elementwise,
    plan_reduce,
    plan_reduce_tiled as plan_reduce_tiled,
    scan_chunk as scan_chunk,
)
from ..runtime.bf16 import pack_bf16 as pack_bf16, unpack_bf16
from ..tuning.space import TuningSpace
from .base import BackendCapabilityError
from .contracts import MemoryStats
from .registry import KernelRegistry
from collections import OrderedDict
from contextlib import nullcontext
from weakref import ref
import numpy as np
from .device_shared import (
    FP8_DTYPES as _FP8_DTYPES,
    FP8_OPS as _FP8_OPS,
    GPU_OPS as GPU_OPS,
    _CAST_FP8_SRC as _CAST_FP8_SRC,
    _ALL_TUNABLE_OPS as _ALL_TUNABLE_OPS,
    _BOOL_OPS as _BOOL_OPS,
    _CONV_CB_OPTIONS as _CONV_CB_OPTIONS,
    _CONV_IMPLS as _CONV_IMPLS,
    _CONV_KNOBS as _CONV_KNOBS,
    _CONV_ROC_OPTIONS as _CONV_ROC_OPTIONS,
    _CONV_TILE_OP as _CONV_TILE_OP,
    _DEQUANTIZE_I8_SRC as _DEQUANTIZE_I8_SRC,
    _ELEMENTWISE_TUNABLE_OPS as _ELEMENTWISE_TUNABLE_OPS,
    _GRAPH_MODES as _GRAPH_MODES,
    _INDEX_ARGS as _INDEX_ARGS,
    _INDEX_DTYPES as _INDEX_DTYPES,
    _KernelHandle as _KernelHandle,
    _LAYOUT_OPS as _LAYOUT_OPS,
    _MEMORY_MODES as _MEMORY_MODES,
    _QUANTIZE_I8_SRC as _QUANTIZE_I8_SRC,
    _REDUCE_TUNABLE_OPS as _REDUCE_TUNABLE_OPS,
    _SPARSE_IMPLS as _SPARSE_IMPLS,
    _TUNABLE_OPS as _TUNABLE_OPS,
    _chainable as _chainable,
    _group_elementwise as _group_elementwise,
    _mask_shared_over_heads as _mask_shared_over_heads,
    _pair as _pair,
    _pool_flags as _pool_flags,
    _pool_out_dim as _pool_out_dim,
    _step_params as _step_params,
)
from ..runtime.executor import Executor
from .device_plan import _PlanMixin
from .device_elementwise import _ElementwiseMixin
from .device_linear import _LinearMixin
from .device_conv import _ConvMixin
from .device_pool import _PoolMixin
from .device_attention import _AttentionMixin
from .device_norm import _NormMixin
from .device_scan import _ScanMixin
from .device_sparse import _SparseMixin
from .device_reduce import _ReduceMixin
from .device_shape import _ShapeMixin
from .device_index import _IndexMixin
from .device_loss import _LossMixin
from .device_misc import _MiscMixin


class DeviceBackend(
    _PlanMixin,
    _ElementwiseMixin,
    _LinearMixin,
    _ConvMixin,
    _PoolMixin,
    _AttentionMixin,
    _NormMixin,
    _ScanMixin,
    _SparseMixin,
    _ReduceMixin,
    _ShapeMixin,
    _IndexMixin,
    _LossMixin,
    _MiscMixin,
    Executor,
):
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
    # 允许 bf16 的算子集。``None`` = 这个后端不支持 bf16（HIP 目前如此）。
    # bf16 在设备上的容器是 ``uint16`` 位模式（与 f16 同为两字节，但没有
    # 原生 dtype），存取由 codegen 的 ``af_ld_b`` / ``af_st_b`` 按位完成，
    # matmul 单独走 cuBLAS 的 CUDA_R_16BF。
    _BF16_OPS: Optional[FrozenSet[str]] = None
    # conv2d 的实现选择。子类可以收窄（HIP 没有 cuDNN）。
    CONV_IMPLS: Tuple[str, ...] = _CONV_IMPLS
    # auto 是否允许在"K 超过分块阈值、核面积大于 1"时改走隐式 GEMM。
    # 默认 False：M40 上 5.151 的 A/B 里隐式在 K >= 288 输给 im2col 1.37～2.32×，
    # 不该由"规模大"自动触发。HIP 覆盖成 True，依据见 5.238 的实测。
    CONV_AUTO_IMPLICIT = False
    # auto 是否允许在中等 K 的甜点区改走隐式 GEMM（判据见 plan.conv_implicit_preferred）。
    # 默认 False：这份阈值是 A800 上 16 个形状测出来的，只有 CUDA 覆盖成 True。
    CONV_AUTO_IMPLICIT_MID = False
    # auto 是否允许 3x3/stride1/dilation1 走 Winograd（判据见
    # plan.conv_winograd_preferred）。默认 False：阈值来自 A800 的七个形状，
    # 且 HIP 侧没有实测过这条路径，只有 CUDA 覆盖成 True。
    CONV_AUTO_WINOGRAD = False
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
            raise ValueError(f"conv_impl 只能是 {self.CONV_IMPLS} 之一，收到 {conv_impl!r}")
        if sparse_impl not in self.SPARSE_IMPLS:
            raise ValueError(f"sparse_impl 只能是 {self.SPARSE_IMPLS} 之一，收到 {sparse_impl!r}")
        if graph_mode not in _GRAPH_MODES:
            raise ValueError(f"graph_mode 只能是 {_GRAPH_MODES} 之一，收到 {graph_mode!r}")
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
            raise ValueError(f"memory_mode 只能是 {_MEMORY_MODES} 之一，收到 {memory_mode!r}")
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
        # 融合 Winograd 实际执行 **权重变换内核** 的次数。常量权重应该只在
        # 第一次执行（或计划重建后）加一，图输入权重则每次执行都加一——
        # 基准与测试因此可以直接验证"驻留有没有真的生效"，不必靠计时猜。
        self.conv_weight_transforms = 0
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
        # 捕获期正在记录的图输出 id。这些 Value 的零拷贝视图要物化成
        # 独立缓冲（见 ``_shared_view``），否则源被回收后地址会被复用。
        self._capture_outputs: frozenset = frozenset()
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
        self._f16_ops: FrozenSet[str] = self._OPS if self._F16_OPS is None else self._F16_OPS
        # bf16 支持面：默认空集，子类显式声明才开启（见 ``_BF16_OPS``）
        self._bf16_ops: FrozenSet[str] = frozenset() if self._BF16_OPS is None else self._BF16_OPS
        # 自动图执行：auto 时 execute 会按"图 + 结构版本 + 输入签名"缓存捕获
        # 好的计划，命中就只 replay。键里带 id 与弱引用：图被回收后 id 可能
        # 被复用，命中前必须确认弱引用还指向同一张图。
        self.graph_mode = graph_mode
        self._max_graph_plans = int(max_graph_plans)
        # 低于这个节点数不捕获：单算子图本来就只有一次 launch，replay 只多出
        # 簿记成本，还固定了 cuBLAS 的算法（5.79 实测 matmul_batched 慢 8%）。
        self._min_graph_nodes = int(min_graph_nodes)
        self._graph_plans: "OrderedDict[Any, Tuple[Any, Any]]" = OrderedDict()
        # 捕获 / 命中计数：只用于观测（基准脚本据此判断到底走没走到重放）。
        # 与 ``_graph_plans`` 同生命周期，失效时不清零——它记录的是累计次数。
        self.graph_captures = 0
        self.graph_replays = 0
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

    def _pick_cudnn(self, dtype, groups: int, flops: float = 0.0) -> bool:
        """是否走 cuDNN 卷积。默认没有这项加速。"""
        return False

    def dispose(self, value: Any) -> None:
        """把中间值的地址立刻还给分配器。默认无操作。"""
        return None

    def _storage_root(self, arr):
        """视图的底层持有者；非视图返回自身。

        cupy 的 ``base`` 可以是一条链（视图的视图）：``_op_reshape`` 对一个
        已经是 ``slice`` 结果的数组再 reshape，``out.base`` 就指向**最底层
        数组**而不是中间那个视图。记账与回收必须都落在同一个对象上，所以
        统一追到底（``dispose`` 里的 ``while base.base`` 是同一个约定）。
        """
        base = getattr(arr, "base", None)
        if base is None:
            return arr
        while getattr(base, "base", None) is not None:
            base = base.base
        return base

    def _shares_storage(self, result, source) -> bool:
        """``result`` 是否与 ``source`` 共用同一段设备内存。

        判据是**底层持有者同一**，不是 ``base is source``：当 ``source``
        本身是视图时（SSM 状态被连续 slice / reshape），``result.base``
        指向最底层数组，与 ``source`` 的恒等比较必然失败，于是源被当成
        "没人再用"提前还给 arena，地址立刻分给后面的算子——下游读到的
        是别人写进去的数据（A800 实测：捕获图与逐节点 execute 首步分叉）。
        """
        if result is source:
            return True
        if getattr(result, "base", None) is None:
            return False
        return self._storage_root(result) is self._storage_root(source)

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
        r.register("eq", self._op_eq)
        r.register("ne", self._op_ne)
        r.register("lt", self._op_lt)
        r.register("le", self._op_le)
        r.register("gt", self._op_gt)
        r.register("ge", self._op_ge)
        r.register("where", self._op_where)
        r.register("sign", self._op_sign)
        r.register("round", self._op_round)
        r.register("floor", self._op_floor)
        r.register("ceil", self._op_ceil)
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
        r.register("quantize_i8_zp", self._op_quantize_i8_zp)
        r.register("dequantize_i8", self._op_dequantize_i8)
        r.register("dequantize_i8_zp", self._op_dequantize_i8_zp)
        r.register("dequantize_i8_group", self._op_dequantize_i8_group)
        r.register("dequantize_i8_group_zp", self._op_dequantize_i8_group_zp)
        r.register("fake_quantize_i8", self._op_fake_quantize_i8)
        # LSQ 版与 STE 版只差反向规则；设备端前向共用同一对内核。
        r.register("fake_quantize_i8_lsq", self._op_fake_quantize_i8_lsq)
        r.register("linear_i8", self._op_linear_i8)
        r.register("linear_i8_zp", self._op_linear_i8_zp)
        r.register("linear_i8a8", self._op_linear_i8a8)
        r.register("linear_i8a8_zp", self._op_linear_i8a8_zp)
        # fp8 线性层没有专用内核：载体是 f32，反量化是一次 ``mul``，之后复用
        # sgemm（见 DESIGN 5.327）。5.328 的 pack / unpack 与融合线性层有
        # 自己的内核：前者是位模式搬运，后者按 1 字节读权重、寄存器里解码。
        r.register("linear_fp8", self._op_linear_fp8)
        r.register("linear_fp8a8", self._op_linear_fp8a8)
        r.register("pack_fp8", self._op_pack_fp8)
        r.register("unpack_fp8", self._op_unpack_fp8)
        r.register("linear_fp8_packed", self._op_linear_fp8_packed)
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
        r.register("topk_values", self._op_topk_values)
        r.register("topk_indices", self._op_topk_indices)
        r.register("topk_mask", self._op_topk_mask)
        r.register("reshape", self._op_reshape)
        r.register("transpose", self._op_transpose)
        r.register("cast", self._op_cast)
        r.register("slice", self._op_slice)
        r.register("gelu", self._op_gelu)
        r.register("rmsnorm", self._op_rmsnorm)
        r.register("avgpool2d", self._op_avgpool2d)
        r.register("avgpool2d_backward", self._op_avgpool2d_backward)
        r.register("adaptive_avg_pool2d", self._op_adaptive_avg_pool2d)
        r.register("adaptive_avg_pool2d_backward", self._op_adaptive_avg_pool2d_backward)
        r.register("concat", self._op_concat)
        r.register("embedding", self._op_embedding)
        r.register("embedding_backward", self._op_embedding_backward)
        r.register("gather", self._op_gather)
        r.register("gather_backward", self._op_gather_backward)
        r.register("cross_entropy", self._op_cross_entropy)
        r.register("cross_entropy_backward", self._op_cross_entropy_backward)
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

        bf16 同理：它的设备容器是 ``uint16`` 位模式，``astype`` 只会把位模式
        当整数做数值转换。三条跨宽度路径（f32/f16/bf16 任意方向）都走同一个
        ``cast_fn``，由 codegen 按源与目标的存储类型各自展开。
        """
        target = np.dtype(dtype)
        if value.dtype == target:
            return value
        src_key = self._kind(value)
        dst_key = {
            np.dtype(np.float16): "f16",
            np.dtype(np.float32): "f32",
            np.dtype(np.uint16): "bf16",
        }.get(target)
        if dst_key is None:
            # 与设备存储无关的类型（f64 等）：交给 cupy，走不到 GPU 计算路径
            return value.astype(target, copy=False)
        if dst_key == src_key:
            return value
        return self._cast_kernel(value, target, src_key, dst_key)

    def device_dtype(self, dtype: DType) -> np.dtype:
        """bf16 在本后端以 ``uint16`` 位模式驻留（与 f16 同为两字节）。

        其余类型就是 numpy 载体。HIP 没有声明 bf16 支持面，因此保持默认行为，
        在 ``invoke`` 处明确拒绝。
        """
        if dtype is DType.BF16 and self._bf16_ops:
            return np.dtype(np.uint16)
        return np.dtype(dtype.np_dtype)

    def quantize(self, value: Any, dtype: DType) -> Any:
        """节点边界：bf16 / fp8 都舍入到格式的格点上。

        bf16 的载体在设备上是两字节位模式，打包用 ``cast``（f32 -> bf16 的
        store helper 就是 RNE）；fp8 的载体仍是 f32，用 ``_fp8_round`` 在设备
        上重做一遍 pack/unpack 的量化（见 5.325）。两者都不做 D2H 同步。
        """
        if dtype in _FP8_DTYPES:
            cp = self.cp
            if isinstance(value, cp.ndarray):
                kind = "e4m3" if dtype is DType.F8E4M3 else "e5m2"
                return self._fp8_round(value, kind)
            return super().quantize(value, dtype)
        if dtype is not DType.BF16:
            return value
        cp = self.cp
        if isinstance(value, cp.ndarray):
            if value.dtype == cp.uint16:
                return value
            return self.cast(value, np.dtype(np.uint16))
        return super().quantize(value, dtype)

    def _fp8_round(self, value, kind: str):
        """把 f32 载体量化到 fp8 的格点（输出仍是 f32）。

        IR 里 fp8 的 ``np_dtype`` 就是 f32，所以"cast 到 fp8"在设备上只是
        把数值舍入到格式能表示的格点上；``_CAST_FP8_SRC`` 与 runtime/fp8.py
        的 RNE / 饱和 / 溢出规则逐位一致。
        """
        cp = self.cp
        src = self._contig(value)
        out = cp.empty(src.shape, cp.float32)
        n = int(src.size)
        if n == 0:
            return out
        kernel = self._kernel(("cast_fp8", kind), _CAST_FP8_SRC, "af_cast_fp8")
        block, grid = self._grid("cast", n)
        kernel(
            (grid,),
            (block,),
            (src, out, np.int64(n), np.int32(1 if kind == "e4m3" else 2)),
        )
        return out

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

        登记的是**底层持有者**的地址，不是 ``owner`` 自己的 ``data.ptr``：
        后者在 ``owner`` 是视图时是偏移地址，而 ``dispose`` 追到最底层后
        用根地址扣账，两边对不上就一条也匹配不到。
        """
        arena = self._active_arena
        if arena is not None:
            arena.alias(self._storage_root(owner).data.ptr)

    def _shared_view(self, out, src, node=None):
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

        捕获期多一条：**图输出必须独占一块**。捕获图把地址写死，输出若与
        源共用一段，源按"最后一个读者"回收后地址会分给后面的算子，图里
        就出现两个输出写同一块（A800 实测：``logits`` 与 ``l0_h`` 同址，
        设备态解码首步之后 token 全偏）。物化一次只发生在捕获期，replay
        时的地址仍然固定。
        """
        if node is not None and node.output.id in self._capture_outputs:
            dst = self.cp.empty(out.shape, dtype=out.dtype)
            dst[...] = out
            return dst
        if self._shares_storage(out, src):
            self._alias_view(src)
        return out

    def to_host(self, value: Any) -> np.ndarray:
        """回到主机：bf16 的 ``uint16`` 容器解包成 f32 载体（与 IR 的载体一致）。"""
        if isinstance(value, self.cp.ndarray) and value.dtype == self.cp.uint16:
            return unpack_bf16(self.cp.asnumpy(value))
        return self.cp.asnumpy(value)

    def to_host_async(self, value: Any, stream: Any = None):
        """pinned 异步 D2H；后端给不出异步路径时退回同步。

        解包 bf16（``uint16`` 容器）必须在主机侧做，所以那一类永远走同步：
        设备端没有"解包成 f32"的拷贝，硬塞进异步只会读到还没转换的位模式。
        """
        if isinstance(value, self.cp.ndarray) and value.dtype != self.cp.uint16:
            handle = self._host_copy(value, stream)
            if handle is not None:
                return handle
        return super().to_host_async(value, stream)

    def _host_copy(self, value: Any, stream: Any) -> Any:
        """后端自定义的 pinned D2H；返回 ``None`` 表示只能同步拷。"""
        return None

    # ------------------------------------------------------------------
    # 设备驻留
    # ------------------------------------------------------------------

    def resident_parameters(self, graph, values: Mapping[str, Any]) -> None:
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
            return {name: env[v.id] for v, name in zip(graph.outputs, graph.output_names)}

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
                self.graph_replays += 1
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
        self.graph_captures += 1
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

    def graph_profile_stats(self, graph) -> Optional[Dict[str, float]]:
        """这张图捕获计划的累计分段耗时（需 ``AETHERFLOW_GRAPH_PROFILE=1``）。

        键：``upload`` / ``launch`` / ``sync`` / ``guards``（秒）与 ``steps``。
        ``launch`` 只计提交，GPU 执行时间落在 ``sync`` 里。
        """
        for holder, plan in self._graph_plans.values():
            if holder() is graph:
                stats = getattr(plan, "profile_stats", None)
                return dict(stats) if stats else None
        return None

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
                options[name] = (current,) + tuple(v for v in candidates if v != current)
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
            raise ValueError(f"conv2d 的配置至少要有一个参数：{sorted(_CONV_KNOBS)}")
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
        return auto_block(work, vec=vec, preferred=BLOCK_DEFAULT, target_blocks=self.sm_count)

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
        # fp8 的载体是 f32，但语义要求节点边界落在格式格点上；设备端承接
        # fp8 的只有 ``cast``（内核在设备上重做 RNE 量化，见 5.325）、两个
        # 线性层（按 f32 载体读出、反量化后走既有 GEMM，见 5.327）与
        # pack / unpack（位模式与载体互转，见 5.328）。其它算子遇到 fp8
        # 必须显式拒绝，否则会被 f32 载体当成 f32 算掉。
        if name not in _FP8_OPS:
            for item in (*node.inputs, node.output):
                if item.type.dtype in _FP8_DTYPES:
                    raise BackendCapabilityError(
                        f"{self._LABEL}的 {name} 不承接 fp8（节点 "
                        f"%{node.output.name} 涉及 {item.type.dtype.value}）；"
                        "目前只有 cast / pack_fp8 / unpack_fp8 与两个 fp8 "
                        "线性层承接，见 DESIGN 5.325 / 5.328"
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
            if a.dtype == self.cp.bool_ and name in _BOOL_OPS:
                continue
            if a.dtype == self.cp.float32:
                continue
            if a.dtype == self.cp.float16 and name in self._f16_ops:
                continue
            if a.dtype == self.cp.uint16 and name in self._bf16_ops:
                continue
            hint = ""
            if a.dtype in (self.cp.float16, self.cp.float64):
                accept = "float16 / float32" if name in self._f16_ops else "float32"
                hint = f"（{name} 目前只支持 {accept}）"
            elif a.dtype == self.cp.uint16:
                hint = (
                    "（bf16 支持面不含这个算子，见 _BF16_OPS）"
                    if self._bf16_ops
                    else "（本后端未声明 bf16 支持）"
                )
            raise BackendCapabilityError(
                f"{self._LABEL}的 {name} 不支持 {a.dtype}{hint}。"
                "不做隐式降级：请在前端显式插入 cast。"
            )
        return self.kernels.dispatch(name, node, args, attrs, context=node)

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
            raise BackendCapabilityError(f"{op} 的 output_size 必须 >= 1，得到 {(oh, ow)}")
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

    def _norm_param(self, p, cols: int, x_shape: tuple, norm_axes: tuple, name: str, op: str):
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
    def _collapse_dims(shape: tuple, *strides: List[int]) -> List[tuple]:
        """把相邻且每个输入都连续的维度合并，返回 ``[(dim, stride...)]``（由外到内）。

        合并的前提是每个输入的 stride 都恰好等于"内层总长度"，否则合并会改变索引。
        二元算子是两组 stride，``where`` 是三组。
        """
        groups: List[tuple] = []
        for d in range(len(shape) - 1, -1, -1):
            if shape[d] == 1:
                continue
            if groups:
                inner = groups[-1]
                inner_dim = inner[0]
                if all(
                    s[d] == inner[k + 1] * inner_dim for k, s in enumerate(strides)
                ):
                    groups[-1] = (inner_dim * shape[d],) + inner[1:]
                    continue
            groups.append((shape[d],) + tuple(s[d] for s in strides))
        groups.reverse()
        return groups

    def _kind(self, a) -> str:
        """内核源码与缓存的 dtype 键（``f32`` / ``f16`` / ``bf16`` / ``bool``）。

        ``uint16`` 只可能是 bf16 的位模式容器（IR 没有 u16 类型）；``bool``
        只出现在比较结果与 ``where`` 的 cond 上，由 ``_BOOL_OPS`` 放行。
        """
        if a.dtype == self.cp.bool_:
            return "bool"
        if a.dtype == self.cp.uint16:
            return "bf16"
        return "f16" if a.dtype == self.cp.float16 else "f32"

    def _one(self):
        """设备上的常数 1（形状 (1,)），供 softmax 的默认缩放使用。"""
        return self._one_dev
