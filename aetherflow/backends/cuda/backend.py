"""CUDA 后端插件：只放厂商相关的部分。

执行计划（分块、融合分组、调优、``_op_*`` 派发）在 ``..device_backend``，
HIP 与 CUDA 是兄弟关系而不是父子关系。这里留下的是只有 NVIDIA 才成立的
东西：cuBLAS、cuDNN、cupy 的 pinned 缓冲与内存池、CUDA Graph 捕获。

``CUDA_OPS`` 仍然定义在这里而不是基类：能力面属于厂商声明，HIP 将来若要
收窄，改自己的那份即可，不必动这里的代码。
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple

import numpy as np

from ...ir.graph import Node
from ...lower.plan import CONV_CUDNN_MIN_GFLOPS
from ..base import BackendCapabilityError
from ..contracts import DeviceInfo
from ..device_backend import DeviceBackend, GPU_OPS
from . import codegen
from .cublas import Cublas
from .cudnn import CudnnConv, CudnnUnavailable
from .cufft import CufftPlan, CufftUnavailable, available as cufft_available
from .cusparse import CusparseSpMM, CusparseUnavailable
from .memory import Arena

# 类级能力声明。它存在的唯一理由：**不建 CUDA 上下文也能读到**。构图的工具、
# 预检脚本、"这张图这个后端接不接得住"的判断，都不该为了问一句能力而初始化
# 一块显卡。运行期的事实来源是 ``self.kernels``；构造时两者对账，不一致直接
# 抛错（见 ``_register_kernels``），所以不存在"声明与实现分叉"的中间状态。
CUDA_OPS: FrozenSet[str] = GPU_OPS

_CONV_IMPLS = (
    "auto", "im2col", "tiled", "implicit", "winograd", "winograd_fused",
    "cudnn", "grouped",
)
_SPARSE_IMPLS = ("custom", "auto", "cusparse")

# fft / rfft 由 cuFFT 承接。它们不在 ``CUDA_OPS`` 里：那张表是"自研内核模板
# 承接面"的声明，而 FFT 没有自研内核——它是厂商库能力，与 cuDNN 一样只作为
# 实现细节存在。``invoke`` 在 dtype 门禁之前拦下这两个名字。
_FFT_OPS: FrozenSet[str] = frozenset({"fft", "rfft"})


class CudaBackend(DeviceBackend):
    """NVIDIA 上的后端。构造失败时抛出带原因的错误，不静默退回 CPU。"""

    name = "cuda"
    _LABEL = "CUDA 后端"
    _OPS = CUDA_OPS
    CONV_IMPLS = _CONV_IMPLS
    SPARSE_IMPLS = _SPARSE_IMPLS
    # 中等 K 的 3x3/5x5 交给隐式 GEMM：A800 上比 im2col 快 7～13%（5.252）。
    # 与 HIP 的整体开关不同，这里的甜点区有物化规模与 K 上界，具体判据在
    # ``plan.conv_implicit_preferred``。
    CONV_AUTO_IMPLICIT_MID = True
    # 3x3/stride1/dilation1 且 K 超过分块阈值时改走 Winograd：A800 上
    # 0.133～0.136 ms，同形状 im2col 0.16～0.31 ms（5.255）。判据在
    # ``plan.conv_winograd_preferred``。
    CONV_AUTO_WINOGRAD = True

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
        device: int = 0,
    ) -> None:
        import cupy as cp  # 延迟导入：没有 CUDA 的机器上仍可 import 本模块

        self.cp = cp
        # 方言由实例持有：HIP 插件复用同一批模板，只把 f16 的存取换成
        # clang 的 `_Float16`。放在实例上而不是模块全局，两个后端才能同时
        # 存在而不互相污染。
        self.codegen = codegen
        # 先选卡再问属性：SM 数是**当前设备**的属性，顺序反了就会拿 0 号卡
        # 的数值去铺别的卡。越界索引在这里显式报错，不留到第一次分配。
        count = int(cp.cuda.runtime.getDeviceCount())
        if isinstance(device, bool) or not isinstance(device, int):
            raise BackendCapabilityError(f"设备索引必须是 int，收到 {device!r}")
        if not 0 <= device < count:
            raise BackendCapabilityError(
                f"设备索引 {device} 越界：进程可见 {count} 块设备（0..{count - 1}）"
            )
        cp.cuda.Device(device).use()
        # 自动 block 的目标块数：一块 SM 至少排到一个块。查设备属性是一次
        # 主机侧查询，构造时就做完，捕获期不再查。
        self.sm_count = max(
            1,
            int(
                cp.cuda.runtime.getDeviceProperties(device)["multiProcessorCount"]
            ),
        )
        # pinned 暂存池：捕获后的 replay 靠它把 H2D 变成真异步（见 pinned_array）
        self._pinned: Dict[Tuple[str, Tuple[int, ...], np.dtype], np.ndarray] = {}
        # 执行期分层分配器；只在显式 memory_mode="tiered" 时惰性创建
        self._tiered: Any = None
        # cuDNN 惰性创建；_cudnn_broken 存"第一次失败的原因"，非空就不再重试
        self._cudnn: Any = None
        self._cudnn_broken: Optional[str] = None
        # cuSPARSE 同样惰性创建；失败原因只记一次，auto 下不反复 try/except
        self._cusparse: Any = None
        self._cusparse_broken: Optional[str] = None
        # cuFFT 同样是惰性创建 + 失败只记一次
        self._cufft: Any = None
        self._cufft_broken: Optional[str] = None
        super().__init__(
            block=block,
            max_blocks=max_blocks,
            conv_impl=conv_impl,
            graph_mode=graph_mode,
            max_graph_plans=max_graph_plans,
            min_graph_nodes=min_graph_nodes,
            chunked_scan=chunked_scan,
            memory_mode=memory_mode,
            sparse_impl=sparse_impl,
        )

    def _memory_scope(self):
        """执行期分层内存：把 cupy 的分配器临时换成本后端的 ``Arena``。

        ``Arena`` 本来就是捕获期用的分层分配器——chunk 切分、best-fit、相邻
        块合并，返回值是 cupy 分配器接口要求的 ``MemoryPointer``（底层的
        ``UnownedMemory`` 正是为此留的）。执行期复用它有两个直接后果：

        - ``run`` 固定用 ``release=True``，``dispose`` 会把不再被读的中间值
          立刻还给 arena，峰值因此接近"这张图的显存下界"，而不是"所有中间值
          同时活着"；
        - 统计口径与捕获期完全一致：``memory_stats()`` 读到的是最近一次的
          arena 快照，不需要区分"这是哪一段的峰值"。

        进程级的 ``set_allocator`` 必须严格成对还原——捕获期用的是同一套
        机制（``capture`` 里 install/restore）。
        """
        if self.memory_mode != "tiered":
            return nullcontext()
        cp = self.cp
        if self._tiered is None:
            self._tiered = Arena(cp)
        previous = cp.cuda.get_allocator()

        @contextmanager
        def scope():
            cp.cuda.set_allocator(self._tiered.malloc)
            self._active_arena = self._tiered
            try:
                yield
            finally:
                self._active_arena = None
                cp.cuda.set_allocator(previous)
                self._last_arena = self._tiered

        return scope()

    def pinned_array(self, key: str, arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
        """把主机数组拷进一块复用的 pinned 缓冲，返回它的 numpy 视图。

        直接 ``buf.set(pageable_array)`` 会退化成同步拷贝：主机要等这次 H2D
        真正完成才能继续，CUDA Graph 省下来的下发时间又被这一次等待吃掉。
        先拷进 pinned 内存（几微秒的 CPU 拷贝），H2D 才是真异步的。

        缓冲按 ``(key, 形状, dtype)`` 复用：训练里的输入形状固定，种类有限。
        调用方要保证上一轮拷贝已经同步过（``run`` / ``run_device`` 都会同步），
        否则会覆盖正在传输的内容。
        """
        shape = tuple(arr.shape)
        ident = (key, shape, np.dtype(dtype))
        count = int(np.prod(shape, dtype=np.int64))
        view = self._pinned.get(ident)
        if view is None:
            nbytes = max(count * np.dtype(dtype).itemsize, 1)
            mem = self.cp.cuda.alloc_pinned_memory(nbytes)
            view = np.frombuffer(mem, dtype=dtype, count=count).reshape(shape)
            self._pinned[ident] = view
        np.copyto(view, arr, casting="unsafe")
        return view

    def dispose(self, value: Any) -> None:
        """把中间值的地址立刻还给 arena（只在捕获期有效）。

        cupy 14 的 ``MemoryPointer`` 析构时不会回调自定义分配器，所以
        arena 的回收点只能由执行器显式给出。非捕获期是空操作。

        ``reshape`` / ``transpose`` / ``slice`` 的输出是共享底层缓冲的视图。
        视图不拥有内存，所以这里不能按地址还块：源数组的回收要等最后一个
        视图死掉。两个方向都靠 ``Arena`` 的引用计数兜住——创建视图时
        ``alias`` 记账，这里 ``unalias`` 扣账，源数组的 ``free`` 记账成"待回收"。

        扣账必须用**最底层源数组**的地址：``alias`` 登记的是它，而视图的
        ``data.ptr`` 是"源地址 + 偏移"——slice 从第 2 行开始时两者不等，
        按视图自己的地址扣账会一条也匹配不上，源地址就永久挂住。
        """
        arena = self._active_arena
        if arena is None or not isinstance(value, self.cp.ndarray):
            return
        base = value.base
        if base is not None:
            while base.base is not None:
                base = base.base
            arena.unalias(base.data)
            return
        if arena.pending(value.data):
            # 整段切片 / 同形状 reshape 返回的就是源数组本身，但执行器看到
            # 两条 Value：第一次 dispose 已把地址记成待回收，这一次扣掉自己
            # 那份视图登记，而不是把 free 再走一遍（那样永远不归零）。
            arena.unalias(value.data)
            return
        arena.free(value.data, value.nbytes)

    # ------------------------------------------------------------------
    # 设备驻留
    # ------------------------------------------------------------------

    def device_info(self) -> DeviceInfo:
        """``DeviceContract``：这块设备是什么。

        只报三项：名字、CUDA 运行时版本、显存总量。多设备选择、stream 管理
        在 ``DeviceManagerContract`` 与 ``StreamContract`` 里，不混进来。
        """
        cp = self.cp
        index = int(cp.cuda.Device().id)
        props = cp.cuda.runtime.getDeviceProperties(index)
        name = props["name"]
        return DeviceInfo(
            name=name.decode() if isinstance(name, bytes) else name,
            runtime=cp.cuda.runtime.runtimeGetVersion(),
            total_memory_bytes=cp.cuda.Device(index).mem_info[1],
        )

    def enumerate_devices(self) -> Tuple[DeviceInfo, ...]:
        """``DeviceManagerContract``：按索引顺序列出全部设备。

        ``mem_info`` 只认当前设备，所以每块卡都在自己的上下文里读；``with``
        退出时会还原到调用前的设备，枚举本身不改变"现在用哪块卡"。
        """
        cp = self.cp
        version = cp.cuda.runtime.runtimeGetVersion()
        devices = []
        for index in range(int(cp.cuda.runtime.getDeviceCount())):
            props = cp.cuda.runtime.getDeviceProperties(index)
            name = props["name"]
            with cp.cuda.Device(index):
                total = cp.cuda.Device(index).mem_info[1]
            devices.append(
                DeviceInfo(
                    name=name.decode() if isinstance(name, bytes) else name,
                    runtime=version,
                    total_memory_bytes=total,
                )
            )
        return tuple(devices)

    def current_device(self) -> int:
        return int(self.cp.cuda.Device().id)

    def set_device(self, index: int) -> None:
        """切换当前设备，并让自动 block 的 SM 数跟着换。"""
        cp = self.cp
        count = int(cp.cuda.runtime.getDeviceCount())
        if isinstance(index, bool) or not isinstance(index, int):
            raise BackendCapabilityError(f"设备索引必须是 int，收到 {index!r}")
        if not 0 <= index < count:
            raise BackendCapabilityError(
                f"设备索引 {index} 越界：进程可见 {count} 块设备（0..{count - 1}）"
            )
        previous = int(cp.cuda.Device().id)
        if index != previous and self._tiered is not None:
            # 分层池的块属于旧卡：必须在旧 context 里 free，切走之后
            # runtime.free 会落到新卡的上下文上。
            self._tiered.close()
            self._tiered = None
        cp.cuda.Device(index).use()
        self.sm_count = max(
            1,
            int(cp.cuda.runtime.getDeviceProperties(index)["multiProcessorCount"]),
        )
        if index == previous:
            return
        # 换卡之后旧设备上的一切都不能再用：NVRTC 编译出的模块绑在编译时的
        # context，cuBLAS handle 绑在设备，驻留数组带 device id，捕获计划里的
        # 地址属于旧卡的内存池。不清掉就会"在新卡上执行旧计划"，报错点离
        # 真正的原因很远。驻留参数直接清空：它们本来就在旧卡的显存里，
        # 保留一个跨设备的数组只会把错误推迟到下一次 execute。
        self._modules.clear()
        self._functions.clear()
        self._kernel_handles.clear()
        self._blas = None
        # cuSPARSE handle 与 workspace 都绑在旧卡的 context 上，和 cuBLAS 同理
        self._cusparse = None
        # cuFFT 的 plan 绑在旧卡的 context 上
        if self._cufft is not None:
            self._cufft.close()
        self._cufft = None
        self._resident.clear()
        self._invalidate_plans()
        # 计划里缓存的常量设备副本也属于旧卡（5.172）：计划结构照旧有效，
        # 但里面的地址不能再用，清掉值，下次执行在计划重建时重新上传。
        self._clear_plan_values()

    def allocator(self) -> Arena:
        """``MemoryContract``：新建一个捕获期分配器（``AllocatorContract``）。

        每次捕获都要一个新的：arena 里切出来的地址被写死进图，两张图共用
        一个 arena 就会让各自的中间值互相踩。所以这里是工厂，不是缓存。
        """
        return Arena(self.cp)

    def _cudnn_engine(self):
        """惰性建 cuDNN handle；失败一次就不再重试。

        "失败只试一次"很关键：每步都 try/except 会把派发常数抬回去，而缺少
        cuDNN 的机器上这个异常是**必然**发生的。失败原因留在 ``_cudnn_broken``
        里，显式 ``conv_impl="cudnn"`` 时用它报一条能看懂的错。
        """
        if self._cudnn is not None:
            return self._cudnn
        if self._cudnn_broken:
            return None
        try:
            self._cudnn = CudnnConv()
        except (CudnnUnavailable, OSError, AttributeError) as exc:
            # AttributeError 是"库在、符号没了"：cuDNN 版本不匹配时
            # ``CudnnConv.__init__`` 里取 legacy 符号（如
            # ``cudnnGetConvolutionForwardAlgorithm_v7``）会抛它。它和"库不在"
            # 是同一类事实（这台机器用不了这条快路径），auto 必须回落；
            # 漏在这里会让默认 ``conv_impl="auto"`` 在构造期就整体失败。
            if self.conv_impl == "cudnn":
                raise BackendCapabilityError(
                    f"{self._LABEL} 的 conv_impl='cudnn' 需要可用的 cuDNN：{exc}"
                ) from exc
            self._cudnn_broken = f"{type(exc).__name__}: {exc}"
            return None
        return self._cudnn

    def _pick_cudnn(self, dtype, groups: int, flops: float = 0.0) -> bool:
        """这一层 conv 走不走 cuDNN。

        只覆盖 groups=1 的 f32/f16：分组卷积与 f64 仍走自研路径，不为此多开
        一条只在部分机器上验证过的分支。

        auto 还要过规模门槛：1 GFLOP 以下 cuDNN 的固定开销（描述符、算法查询、
        独立 bias 核）比省下的卷积时间更贵（5.252）。
        """
        if groups != 1:
            return False
        supported = np.dtype(dtype) in (np.float32, np.float16)
        if self.conv_impl == "cudnn":
            if not supported:
                raise BackendCapabilityError(
                    f"conv_impl='cudnn' 只支持 float32 / float16，收到 {np.dtype(dtype)}"
                )
            if self._capturing:
                raise BackendCapabilityError(
                    "CUDA Graph 捕获期不支持 conv_impl='cudnn'（algo 查询是 host API）；"
                    "捕获请用 conv_impl='auto' 或 'tiled'"
                )
            return True
        if self.conv_impl != "auto" or not supported:
            return False
        if flops < CONV_CUDNN_MIN_GFLOPS:
            return False
        if self._capturing:
            # 捕获期内不换实现：cuDNN 的 algo 查询是 host API，在捕获里调用
            # 不安全，而且"平时走 cuDNN、捕获时走自研核"会让两条路径的数值
            # 与显存布局不一致。捕获要的是路径固定，这里就保持固定。
            return False
        if self._cudnn_broken is not None:
            return False
        return self._cudnn_engine() is not None

    def _conv2d_cudnn(self, x, weight, bias, *, n, c, h, w, o, kh, kw, ho, wo,
                      sh, sw, ph, pw, dh, dw, k):
        """cuDNN 前向；返回 ``None`` 表示"这一步没走成，请回落"。

        auto 模式下任何 cuDNN 异常都降级而不是冒泡：使用者要的是"能算对"，
        不是"必须用 cuDNN"。显式 ``conv_impl="cudnn"`` 时异常照常抛出，
        否则排障时会被静默换实现误导。
        """
        cp = self.cp
        engine = self._cudnn_engine()
        if engine is None:
            return None
        out = cp.empty((n, o, ho, wo), dtype=x.dtype)
        try:
            engine.convolution(
                self._contig(x),
                self._contig(weight),
                None if bias is None else self._contig(bias),
                out,
                pad=(ph, pw),
                stride=(sh, sw),
                dilation=(dh, dw),
            )
        except Exception as exc:  # noqa: BLE001 —— 见上面的降级说明
            if self.conv_impl == "cudnn":
                raise
            self._cudnn_broken = f"{type(exc).__name__}: {exc}"
            return None
        return out

    def _cusparse_engine(self):
        """惰性建 cuSPARSE handle；失败一次就不再重试（与 cuDNN 同一条理由）。"""
        if self._cusparse is not None:
            return self._cusparse
        if self._cusparse_broken:
            return None
        try:
            self._cusparse = CusparseSpMM()
        except (CusparseUnavailable, OSError) as exc:
            if self.sparse_impl == "cusparse":
                raise BackendCapabilityError(
                    f"{self._LABEL} 的 sparse_impl='cusparse' 需要可用的 "
                    f"cuSPARSE：{exc}"
                ) from exc
            self._cusparse_broken = str(exc)
            return None
        return self._cusparse

    def _pick_cusparse(self, values, indices, offsets) -> bool:
        """这一次 ``sparse_linear`` 走不走 cuSPARSE。

        只覆盖登记过的形态：f32 值 + i32 的 ``indices`` / ``offsets``。
        ``auto`` 在捕获期与库缺失时回落到自研核；显式 ``cusparse`` 则把
        这两种情况变成明确错误，而不是"悄悄换了个实现"。

        **不覆盖越界检查**：自研核在设备端记 ``err`` 后由主机侧抛错，库不做
        逐元素检查。因此显式走库路径时，下标合法性回到调用方责任——这也是
        默认值仍是 ``custom`` 的原因。
        """
        supported = (
            np.dtype(values.dtype) == np.dtype("float32")
            and np.dtype(indices.dtype) == np.dtype("int32")
            and np.dtype(offsets.dtype) == np.dtype("int32")
        )
        if self.sparse_impl == "cusparse":
            if not supported:
                raise BackendCapabilityError(
                    "sparse_impl='cusparse' 只支持 float32 值 + int32 索引，"
                    f"收到 values={np.dtype(values.dtype)}、"
                    f"indices={np.dtype(indices.dtype)}、"
                    f"offsets={np.dtype(offsets.dtype)}"
                )
            if self._capturing:
                raise BackendCapabilityError(
                    "CUDA Graph 捕获期不支持 sparse_impl='cusparse'"
                    "（SpMM 的 workspace 查询是 host API）；"
                    "捕获请用 sparse_impl='auto' 或 'custom'"
                )
            return True
        if self.sparse_impl != "auto" or not supported:
            return False
        if self._capturing or self._cusparse_broken is not None:
            return False
        return self._cusparse_engine() is not None

    def _sparse_linear_cusparse(self, weights, indices, values, offsets, *,
                                steps: int, rows: int, cols: int):
        """cuSPARSE SpMM；返回 ``None`` 表示"没走成，请回落"。

        与 cuDNN 的 ``_conv2d_cudnn`` 同一条策略：auto 模式下任何异常都
        降级（先把 ``_cusparse_broken`` 记下来，后续步骤不再重试），显式
        ``cusparse`` 时异常照常抛出。失败时已经写进 ``out`` 的内容不会被
        使用——调用方会重新走自研核，那条路先把输出清零。
        """
        engine = self._cusparse_engine()
        if engine is None:
            return None
        try:
            return engine.spmm(
                offsets, indices, values, weights,
                steps=steps, rows=rows, cols=cols,
            )
        except Exception as exc:  # noqa: BLE001 —— 见上面的降级说明
            if self.sparse_impl == "cusparse":
                raise
            self._cusparse_broken = f"{type(exc).__name__}: {exc}"
            return None

    # ------------------------------------------------------------------
    # cuFFT
    # ------------------------------------------------------------------

    def _cufft_engine(self) -> CufftPlan:
        """惰性建 cuFFT plan 池；失败一次就不再重试（与 cuDNN 同一条理由）。

        ``fft`` / ``rfft`` 没有自研内核可回落，所以这里不设"auto 降级"：
        库不可用就是明确错误。捕获期单独拒绝——``cufftPlanMany`` 是 host API，
        在 CUDA Graph 捕获里调用不安全。
        """
        if self._cufft is not None:
            return self._cufft
        if self._cufft_broken:
            raise BackendCapabilityError(
                f"{self._LABEL} 的 fft/rfft 需要可用的 cuFFT：{self._cufft_broken}"
            )
        if self._capturing:
            raise BackendCapabilityError(
                "CUDA Graph 捕获期不支持 fft/rfft（cufftPlanMany 是 host API）；"
                "含 FFT 的图请用 graph_mode='off'"
            )
        try:
            self._cufft = CufftPlan()
        except (CufftUnavailable, OSError) as exc:
            self._cufft_broken = str(exc)
            raise BackendCapabilityError(
                f"{self._LABEL} 的 fft/rfft 需要可用的 cuFFT：{exc}"
            ) from exc
        return self._cufft

    def invoke(self, node: Node, args: List[Any], attrs: dict) -> Any:
        """``fft`` / ``rfft`` 走 cuFFT，其余交给基类。

        这两个算子的输入 dtype（c64 / c128 / f32 / f64）不在基类的"f32 或
        f16"门禁里，必须在门禁之前拦下。dtype 与形状仍按 IR 声明核对：
        入口已在 ``run_device`` 对齐过，这里再确认一次是为了让"直接调
        ``invoke``"的路径同样严格。
        """
        name = node.op.name
        if name not in _FFT_OPS:
            return super().invoke(node, args, attrs)
        x = self._contig(args[0])
        if x.dtype != np.dtype(node.inputs[0].type.dtype.np_dtype):
            raise BackendCapabilityError(
                f"{self._LABEL} 的 {name} 输入 dtype {x.dtype} 与 IR 声明的 "
                f"{node.inputs[0].type.dtype.value} 不一致"
            )
        size = int(x.shape[-1])
        rows = int(x.size // size) if size else 0
        inverse = bool(attrs.get("inverse", False)) if name == "fft" else False
        return self._cufft_engine().transform(
            x,
            size=size,
            rows=rows,
            inverse=inverse,
            norm=attrs.get("norm", "backward"),
        )

    def capabilities(self) -> FrozenSet[str]:
        """自研内核 + 可用的 cuFFT。

        ``fft`` / ``rfft`` 不是内核表里的条目，但能力声明必须如实反映"这个
        后端能不能算"。cuFFT 缺失时**不声明**：宁可让一致性套件把它记为
        "未声明算子"从而拒绝，也不要声明一个执行时才失败的算子。
        """
        caps = super().capabilities()
        if cufft_available():
            return caps | _FFT_OPS
        return caps

    # ------------------------------------------------------------------
    # 编译与 BLAS
    # ------------------------------------------------------------------

    def _kernel(self, key: Any, source, func: str):
        # 连函数句柄一起缓存：``get_function`` 本身有可观的 Python 开销，
        # 而这一段在每次 execute 的每个节点上都会走到。
        #
        # ``source`` 允许是"返回源码的函数"：源码生成（L2 → C）比编译便宜，
        # 但同样在每一步重复——conv2d 一次 0.4 ms，逐元素链上叠得更多；
        # 传函数进来，命中缓存时连源码都不会生成了。
        fn = self._functions.get(key)
        if fn is None:
            text = source() if callable(source) else source
            module = self.cp.RawModule(code=text, options=("-std=c++11",))
            fn = module.get_function(func)
            self._modules[key] = module
            self._functions[key] = fn
        return fn

    def current_stream_ptr(self) -> int:
        """当前流的地址，供 cuBLAS 绑定。"""
        return int(self.cp.cuda.get_current_stream().ptr)

    def gpu_clock(self) -> "_CudaGpuClock":
        """逐节点 GPU 计时的时钟（见 ``runtime/telemetry.py``）。"""
        return _CudaGpuClock(self)

    def _cublas(self) -> Cublas:
        if self._blas is None:
            try:
                self._blas = Cublas()
            except OSError as exc:
                raise BackendCapabilityError(f"{self._LABEL}无法加载 cuBLAS：{exc}") from exc
        return self._blas

    # ------------------------------------------------------------------
    # 运行期计划：CUDA Graph
    # ------------------------------------------------------------------

    def new_stream(self):
        """一条非阻塞流：捕获必须发生在非默认流上，replay 也跑在它上面。"""
        return self.cp.cuda.Stream(non_blocking=True)

    def capture(
        self,
        graph,
        sample_inputs: Mapping[str, Any],
        reuse_memory: bool = True,
        shared_buffers: Optional[Mapping[str, Any]] = None,
        stream: Optional[Any] = None,
    ) -> "CudaGraphPlan":
        """把整张图捕获成一个 CUDA Graph，返回可重复 replay 的计划。

        ``sample_inputs`` 只用于确定形状并写入静态缓冲；捕获后再调
        ``plan.run(inputs)`` 只更新输入缓冲、replay 整图，CPU 侧只剩一次
        下发，不再逐节点走 Python。

        规则与 ``execute`` 一致：名字出现在 ``sample_inputs`` 里的输入视为
        可变（每次 run 都要传），只从设备驻留表取到的输入视为固定（权重）。

        ``reuse_memory`` 打开时用 ``Arena`` 代替 cupy 池：图里的地址是写死的，
        中间张量的复用必须在捕获期完成，否则每个节点都独占一块显存。

        ``shared_buffers`` 直接把这些显存数组当作静态输入缓冲，而不是另申请
        一块再拷一遍。设备态训练靠它让前向、反向两张图共用同一块权重：优化器
        原地改的就是图里写死的那段地址，每步既不需要 H2D 也不需要重捕获。
        计划**不持有**这些数组的所有权，调用方必须让它们活到计划释放为止。

        ``stream`` 缺省时新建一条非阻塞流。同一组图传同一条流时，replay 之间
        天然有序，不必再建跨流事件依赖（激活重算的分段模式就是这样用的）。
        """
        cp = self.cp
        declared = [v.name for v in graph.inputs]
        shared = dict(shared_buffers or {})
        unknown = sorted(set(shared) - set(declared))
        if unknown:
            raise KeyError(f"shared_buffers 里这些名字不是图输入: {unknown}")

        buffers: Dict[str, Any] = {}
        mutable: List[str] = []
        pending: List[Tuple[Any, Any]] = []
        for name in declared:
            if name in sample_inputs:
                src = self.to_array(sample_inputs[name])
                mutable.append(name)
            elif name in self._resident:
                src = self._resident[name]
            else:
                raise KeyError(f"capture 缺少图输入: {name}")
            buf = shared.get(name)
            if buf is None:
                # 静态缓冲：捕获记录的是固定地址，之后每次 run 往这里拷新值。
                # 拷贝本身留到预热用的那条流上做——``to_array`` 出来的是主机
                # 数组，在默认流上异步 H2D，非阻塞流上的预热会先读到未就绪的
                # 内存（表现为 embedding 拿随机 ids）。共享缓冲由调用方负责
                # 填好，这里不重复拷。
                buf = cp.empty(src.shape, dtype=src.dtype)
                pending.append((buf, src))
            elif tuple(buf.shape) != tuple(src.shape) or buf.dtype != src.dtype:
                raise ValueError(
                    f"共享缓冲 %{name} 的形状/dtype 是 {tuple(buf.shape)}/{buf.dtype}，"
                    f"与传入值的 {tuple(src.shape)}/{src.dtype} 不一致"
                )
            buffers[name] = buf

        allocator = self.allocator() if reuse_memory else cp.cuda.MemoryPool()
        previous_allocator = cp.cuda.get_allocator()
        stream = stream if stream is not None else self.new_stream()
        # 静态缓冲的填充必须在换分配器之前完成：``to_array`` 出来的是主机
        # 数组，H2D 在默认流上异步进行，而预热跑在非阻塞流上——不建立这条
        # 依赖，预热会先读到未就绪的内存（表现为 embedding 拿随机 ids）。
        # 放在换分配器之前也让这次 H2D 的临时分配不落进捕获 arena，
        # ``capture_reports_memory_budget`` 的预算因此只反映图本身。
        for buf, src in pending:
            buf[...] = src
        ready = cp.cuda.Event()
        ready.record()
        stream.wait_event(ready)
        cp.cuda.set_allocator(allocator.malloc)
        self._active_arena = allocator if isinstance(allocator, Arena) else None
        self._last_arena = self._active_arena
        try:
            # 预热：NVRTC 编译与 cuBLAS workspace 分配都不能发生在捕获期间。
            # 两轮都要按"用完即回收"跑：否则预热结束时 arena 里全是活着的块，
            # 捕获阶段拿不到可复用地址，图里就会给每个中间值各留一块显存。
            with stream:
                self.run_device(graph, buffers, release=True)
            stream.synchronize()
            if isinstance(allocator, Arena):
                allocator.reset_stats()
            # 常量副本在预热轮上传进了这次捕获的 arena；捕获轮要把它们重新
            # 上传一遍，让图里记录的是本轮地址（HIP 的 recycle 会收回预热
            # 期地址，CUDA 这里同样按"捕获轮的地址才算数"处理，两家一致）。
            self._drop_plan(graph)
            # 重建必须在 ``begin_capture`` **之前**：重建会走 ``_prepare_plan``
            # → ``_op_constant``，那一遍是 ``cudaMemcpyAsync`` 的 H2D，而捕获
            # 流上任何 H2D 都会被驱动拒绝（"H2D transfers are disallowed"），
            # 整张图随即退回逐节点执行。这里让它在默认流上完成，再让捕获流
            # 等一次事件，图里读到的就是本轮上传的那份常量。
            self._plan(graph)
            ensure = cp.cuda.Event()
            ensure.record()
            stream.wait_event(ensure)
            with stream:
                stream.begin_capture()
                # 捕获期禁止任何 D2H 同步（embedding 的越界回读会破坏捕获），
                # 因此这里只登记标志，replay 之后由计划复查
                guards: List[Tuple[Any, str, str]] = []
                self._capture_guards = guards
                self._capturing = True
                try:
                    env = self.run_device(graph, buffers, release=True)
                finally:
                    self._capturing = False
                    self._capture_guards = None
                cg = stream.end_capture()
        finally:
            self._active_arena = None
            cp.cuda.set_allocator(previous_allocator)

        outputs = {
            name: env[v.id] for v, name in zip(graph.outputs, graph.output_names)
        }
        return CudaGraphPlan(
            self, cg, stream, buffers, outputs, mutable, allocator, guards
        )

class CudaGraphPlan:
    """一次捕获、多次 replay 的运行期计划。

    ``allocator`` 必须被计划持有：图里写死的是捕获时那些显存的地址，
    arena 一旦被回收，replay 就会写到已经属于别人的内存上。
    """

    def __init__(
        self,
        backend: CudaBackend,
        graph,
        stream,
        buffers: Dict[str, Any],
        outputs: Dict[str, Any],
        mutable: List[str],
        allocator,
        guards: Optional[List[Tuple[Any, str, str]]] = None,
    ) -> None:
        self.backend = backend
        self._cg = graph
        self._stream = stream
        self._buffers = buffers
        self._outputs = outputs
        self._mutable = tuple(mutable)
        self._allocator = allocator
        # 捕获期登记的越界标志（embedding 一类按下标访存的算子）。
        # 图里写死的是这些设备标量的地址，每次 replay 都会重新置位。
        self._guards = tuple(guards or ())

    @property
    def buffers(self) -> Dict[str, Any]:
        """图里写死的输入缓冲（名字 → 数组对象）。

        返回的是字典的副本、数组本身的引用：调用方拿它做**身份**校验
        （"优化器改的是不是图读的那一块"），而不是拿来替换缓冲。
        """
        return dict(self._buffers)

    @property
    def input_shapes(self) -> Dict[str, tuple]:
        return {name: tuple(buf.shape) for name, buf in self._buffers.items()}

    @property
    def mutable_inputs(self) -> tuple:
        return self._mutable

    @property
    def memory_bytes(self) -> int:
        """这张图实际占用的显存（中间缓冲 + 输出）。"""
        if isinstance(self._allocator, Arena):
            return self._allocator.total_bytes
        return self._allocator.total_bytes()

    @property
    def peak_bytes(self) -> int:
        """捕获期同时存活的峰值字节，即这张图绕不过去的显存下界。

        与 ``memory_bytes`` 的差额就是分配器本身的浪费（块内碎片、对齐）。
        """
        if isinstance(self._allocator, Arena):
            return self._allocator.peak_live_bytes
        return self.memory_bytes

    def run(self, inputs: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        """更新可变输入后 replay，返回主机侧输出。"""
        self.replay(inputs)
        # 必须显式同步捕获流：输出缓冲由该流上的图写入，
        # 而 to_host 走的是默认流，不建依赖就会读到上一轮的结果。
        self._stream.synchronize()
        self._check_guards()
        return {name: self.backend.to_host(v) for name, v in self._outputs.items()}

    def run_device(self, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        """replay 后同步，返回**设备上**的输出，不做 D2H。

        设备态训练用它把梯度留在显存直接交给优化器：一步只把损失标量搬回
        主机，权重与梯度的往返彻底消失。
        """
        self.replay(inputs)
        self._stream.synchronize()
        self._check_guards()
        return dict(self._outputs)

    def _check_guards(self) -> None:
        """复查捕获期登记的设备标量，把 replay 里的越界下标变成异常。

        读回是 D2H 同步——捕获早已结束，这条约束不再适用，但必须等
        ``synchronize()`` 之后才读得到本次 replay 的结果。同一个计划还会被
        继续使用，所以报错前先把标量清零，否则下一次合法 replay 会误报。
        """
        for err, op, allowed in self._guards:
            if int(err.item()) == 0:
                continue
            with self._stream:
                err[...] = 0
            self.backend._raise_bad_ids(op, allowed)

    def replay(self, inputs: Mapping[str, Any]) -> None:
        """只下拷输入并 replay，不等 GPU 结束（输出留在设备上）。"""
        # 捕获流是非阻塞流，和默认流（优化器原地更新权重的地方）之间没有
        # 隐式同步。进流之前记下当前流，显式建一条依赖，保证 replay 读到的
        # 是主机侧刚刚更新过的权重。
        previous = self.backend.cp.cuda.get_current_stream()
        with self._stream:
            self._stream.wait_event(previous.record())
            for name in self._mutable:
                if name not in inputs:
                    raise KeyError(f"缺少图输入: {name}")
                buf = self._buffers[name]
                value = inputs[name]
                if tuple(np.shape(value)) != tuple(buf.shape):
                    raise ValueError(
                        f"输入 %{name} 形状 {tuple(np.shape(value))} 与捕获时的 "
                        f"{tuple(buf.shape)} 不一致：一张 CUDA Graph 只对应一组固定形状"
                    )
                if value is buf:
                    # 共享缓冲：调用方（设备态训练）已经把它写在这块地址上了
                    continue
                if isinstance(value, np.ndarray):
                    # 主机数组：先进 pinned 暂存再异步 H2D，主机不必等这次拷贝完
                    buf.set(
                        self.backend.pinned_array(name, value, buf.dtype),
                        stream=self._stream,
                    )
                else:  # 已在显存：设备内拷贝，当前流由上面的 with 保证
                    buf[...] = value
            self._cg.launch(stream=self._stream)

    def device_outputs(self) -> Dict[str, Any]:
        """当前设备上的输出（下一次 replay 会被覆盖）。"""
        return dict(self._outputs)


class _CudaGpuClock:
    """逐节点 GPU 计时：起点/终点各记一个 cupy 事件。

    ``elapsed`` 只在两个事件都完成之后才成立，所以调用方必须在读取前统一
    同步一次。逐节点插同步是另一回事：那样测到的是同步开销本身。
    """

    def __init__(self, backend: "CudaBackend") -> None:
        self._backend = backend
        self._cp = backend.cp

    def _mark(self):
        event = self._cp.cuda.Event()
        event.record()
        return event

    def begin(self):
        return self._mark()

    def end(self):
        return self._mark()

    def synchronize(self) -> None:
        self._cp.cuda.Device().synchronize()

    def elapsed(self, start, stop) -> float:
        return float(start.elapsed_time(stop))

    def release(self, start, stop) -> None:
        for event in (start, stop):
            close = getattr(event, "close", None)
            if close is not None:
                close()
