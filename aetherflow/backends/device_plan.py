"""执行计划：把内核函数预绑定到执行组，逐元素链与 conv 后融合的静态展开。"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from ..ir.graph import Graph, Node
from ..ir.types import DType
from ..lower.elementwise import ElemInput, ElemStep, apply_op, build_chain, channel_broadcast
from ..lower.kernel import Index, Num
from ..lower.plan import (
    CONV_SHARED_LIMIT,
    CONV_WINOGRAD_TILE,
    VEC4_ALIGN,
    cdiv,
    conv_winograd_supported,
    plan_elementwise,
)
import numpy as np
from .device_shared import (
    FP8_DTYPES,
    _INDEX_ARGS,
    _INDEX_DTYPES,
    _LAYOUT_OPS,
    _chainable,
    _group_elementwise,
    _pair,
    _step_params,
)


class _PlanMixin:
    """执行计划：把内核函数预绑定到执行组，逐元素链与 conv 后融合的静态展开。"""

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

        两遍扫描：常量副本要先全部上传，第二遍才知道哪些 conv 的权重来自
        常量（见 :meth:`_holds_constant_weight`）。
        """
        constant_ids = set()
        constant_values: Dict[int, Any] = {}
        for group in plan:
            # 常量节点的值只写在属性里、与运行期输入无关，因此在计划期上传
            # 一次就够：每次执行都 H2D 一份，正是 5.161 里"折叠产物大而原
            # 计算便宜时反而倒退"（mask 用例逐节点 GPU 时间 +22.4%）的根因。
            # 计划随图的版本号缓存，设备数组因此与图同寿命；切卡 / 关闭
            # arena 时由 ``_clear_plan_values`` 清掉。
            if len(group.nodes) != 1:
                continue
            node = group.nodes[0]
            if node.op.name == "constant":
                # i32 也在其中：它是 embedding / gather 的索引表，同样与运行期
                # 无关。漏掉它，捕获轮会在 ``begin_capture`` 之后才第一次
                # 物化，驱动直接拒绝捕获流上的 H2D（"H2D transfers are
                # disallowed"），整张图退回逐节点执行。
                if node.output.type.dtype in (DType.F32, DType.F16, DType.I32):
                    value = self._op_constant(node, (), node.attrs)
                    group.constants[node.output.id] = value
                    constant_ids.add(node.output.id)
                    constant_values[node.output.id] = value
                continue
            # 常量的一次 reshape 只是一次视图：目标形状与元素个数在 IR 上已经
            # 定死，运行期再派发一次没有任何数据工作可做（见
            # :meth:`_fold_constant_reshape`）。
            folded = self._fold_constant_reshape(node, constant_values)
            if folded is not None:
                group.constants[node.output.id] = folded
                constant_ids.add(node.output.id)
                constant_values[node.output.id] = folded
        for group in plan:
            fns = []
            for node in group.nodes:
                fn = self._planned_kernel(node)
                if fn is None:
                    fns = None
                    break
                if self._holds_constant_weight(node, constant_ids):
                    # 权重是常量：Winograd 的 ``U = G g G^T`` 只与权重有关，
                    # 可以跟着计划一起驻留（见 ``_conv2d_winograd_fused``）。
                    # 静态 3x3 的整条 launch 还能在计划期展开：形状、线程铺法、
                    # 标量参数都不随执行变化，运行期只剩分配与一次内核调用。
                    prepared = self._planned_winograd_fused(node)
                    if prepared is None:
                        prepared = self._planned_winograd(node)
                    fn = prepared if prepared is not None else self._resident_conv2d(fn)
                fns.append(fn)
            if fns is not None:
                group.kernels = tuple(fns)
                if len(group.nodes) > 1:
                    if group.nodes[0].op.name == "conv2d":
                        # conv + epilogue：链首不是逐元素算子，不能走
                        # ``_planned_chain``；绑定一个整组闭包，把链内联进
                        # conv 内核的写回点（见 ``_planned_conv_epilogue``）。
                        group.fused = self._planned_conv_epilogue(group)
                    elif group.nodes[0].op.name == "transpose":
                        group.fused = self._planned_transpose_matmul(group)
                    else:
                        group.chain = self._planned_chain(group)

    def _fold_constant_reshape(self, node: Node, values: Dict[int, Any]):
        """常量上的 ``reshape`` 在计划期折成视图；不是这种情形返回 ``None``。

        ``reshape`` 不读也不写数据，只换元数据。输入是计划期已上传的常量时，
        "运行期再 reshape 一次"与"计划期 reshape 好、运行期直接取用"逐位相同，
        但前者每层要付一次派发 / 形状比对 / 写回环境。8 层 conv-bn-relu 的
        ``channel`` 形态（BN 参数以 ``(C,)`` 给出、再 reshape 成 ``(1,C,1,1)``）
        因此比 ``channel4d`` 多约 0.1 ms，折掉这两处就追平。

        只折"目标形状全是字面整数"的 reshape：含符号维时目标随绑定变化，
        必须留给运行期按本次绑定解析。返回值仍是视图（与 ``_op_reshape``
        同一条数据路径）；不登记 arena 别名，因为常量在计划存活期内不会被
        回收，源数组由 ``constant_values`` 与 ``group.constants`` 双双持有。
        """
        if node.op.name != "reshape" or len(node.inputs) != 1:
            return None
        src = values.get(node.inputs[0].id)
        if src is None:
            return None
        target = node.attrs.get("shape") if isinstance(node.attrs, dict) else None
        if not target or not all(isinstance(d, int) for d in target):
            return None
        return self.cp.reshape(src, tuple(int(d) for d in target))

    @staticmethod
    def _holds_constant_weight(node: Node, constant_ids) -> bool:
        """conv 的权重是不是一个常量节点的输出。

        只有这种情况才允许缓存权重变换的结果。判据是 IR 结构（权重由
        ``constant`` 节点产生），不是"指针看起来没变"——后者在权重是图
        输入时会把原地更新的新值算成旧值，且没有任何可观测的迹象。
        """
        if node.op.name != "conv2d" or len(node.inputs) < 2:
            return False
        if node.inputs[0].type.dtype is not DType.F32:
            return False
        return node.inputs[1].id in constant_ids

    @staticmethod
    def _resident_conv2d(fn):
        """给 conv 内核挂一个"权重变换结果"的盒子（惰性、随计划存活）。

        盒子在第一次真正走融合内核时才填：计划期还不知道 cuDNN 会不会
        接管这一层，白算一次 ``U`` 只是浪费。盒子由 ``group.kernels``
        持有，因此与执行计划同寿命；计划作废（``_clear_plan_values`` /
        ``_drop_plan``）时盒子一起消失，不会跨图误用。
        """
        box: List[Any] = []

        def call(node, args, attrs):
            return fn(node, args, attrs, resident=box)

        return call

    def _winograd_static(self, node: Node, kinds: Tuple[str, ...]):
        """静态 3x3 conv 的解析结果 ``(dims, has_bias)``；不适用时返回 ``None``。

        ``kinds`` 是调用方愿意在计划期展开的实现名（融合版 / 四段版）。
        形状校验、cuDNN 让位判据与 ``_pick_conv_kernel`` 的调用都只写这一处，
        两份展开因此共用同一套阈值。显式 ``conv_impl`` 直接就是实现名，
        只有 ``auto`` 才去问判据。
        """
        if self.conv_impl not in ("auto",) + tuple(kinds):
            return None
        try:
            x_type = node.inputs[0].type
            w_type = node.inputs[1].type
            b_type = node.inputs[2].type if len(node.inputs) > 2 else None
            if not x_type.is_static or not w_type.is_static:
                return None
            if b_type is not None and not b_type.is_static:
                return None
            if len(x_type.shape) != 4 or len(w_type.shape) != 4:
                return None
            if x_type.dtype is not DType.F32:
                return None
            n, c, h, w = (int(v) for v in x_type.shape)
            o, cw, kh, kw = (int(v) for v in w_type.shape)
            attrs = node.attrs
            if int(attrs.get("groups", 1)) != 1 or cw != c:
                return None
            sh, sw = _pair(attrs.get("stride", (1, 1)))
            ph, pw = _pair(attrs.get("padding", (0, 0)))
            dh, dw = _pair(attrs.get("dilation", (1, 1)))
            if not conv_winograd_supported(kh=kh, kw=kw, sh=sh, sw=sw, dh=dh, dw=dw):
                return None
            ho = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
            wo = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
            if ho <= 0 or wo <= 0:
                return None
            has_bias = b_type is not None
            if has_bias and tuple(int(v) for v in b_type.shape) != (o,):
                return None
            k = c * kh * kw
            dims = dict(
                n=n,
                c=c,
                h=h,
                w=w,
                o=o,
                kh=kh,
                kw=kw,
                ho=ho,
                wo=wo,
                sh=sh,
                sw=sw,
                ph=ph,
                pw=pw,
                dh=dh,
                dw=dw,
                k=k,
            )
            flops = 2.0 * n * o * ho * wo * k / 1e9
            if self._pick_cudnn(self.cp.float32, 1, flops):
                return None
            kind = (
                self.conv_impl
                if self.conv_impl != "auto"
                else self._pick_conv_kernel(weight_resident=True, **dims)
            )
            if kind not in kinds:
                return None
            return dims, has_bias
        except Exception:  # noqa: BLE001 - 预解析失败只影响这条优化
            return None

    def _planned_winograd_fused(self, node: Node, epilogue=None, extra=(), epi_key=None):
        """静态 3x3 conv 的融合 Winograd launch，在计划期展开成闭包。

        运行期每次重新走 ``_op_conv2d`` 的代价不在内核：形状解包、``_pair``、
        ``_pick_conv_kernel``、``_kind``、``_kernel`` 查表与十几个 ``np.int32``
        的打包，在 A800 上合计约 15～20 µs，已经和 c16 的内核本身同量级。
        这些量在静态图上**只取决于 IR**，与绑定值无关，因此可以提前算好。

        只覆盖 ``auto`` / 显式 ``winograd_fused`` 且判据确实选中融合版的形状：
        实现选择仍由 ``_pick_conv_kernel`` 这一份判据给出，不在这里复制阈值。
        任何一条不满足就返回 ``None``，运行期走原来的 ``_op_conv2d``——报错
        时机与文案因此完全不变。

        权重必须来自常量节点（调用方已保证）：``U`` 由闭包持有的 ``box``
        驻留，与执行计划同寿命；形状或 dtype 变了就回落原路径重算，不会拿
        旧 ``U`` 算新权重。
        """
        parsed = self._winograd_static(node, ("winograd_fused",))
        if parsed is None:
            return None
        dims, has_bias = parsed
        n, c, h, w, o = dims["n"], dims["c"], dims["h"], dims["w"], dims["o"]
        ho, wo, ph, pw = dims["ho"], dims["wo"], dims["ph"], dims["pw"]
        try:
            tt = CONV_WINOGRAD_TILE
            shared = 16 * c * tt * 4
            if shared > CONV_SHARED_LIMIT:
                return None
            th, tw = cdiv(ho, 2), cdiv(wo, 2)
            nt = n * th * tw
            # 权重变换的线程铺法：与 ``_conv2d_winograd_fused`` 的 ``_grid``
            # 同一个键，缓存命中后只做一次字典查找。
            self._grid("conv2d", o * c)
            static = (
                np.int32(c),
                np.int32(h),
                np.int32(w),
                np.int32(o),
                np.int32(th),
                np.int32(tw),
                np.int32(ho),
                np.int32(wo),
                np.int32(ph),
                np.int32(pw),
                np.int32(nt),
                np.int32(tt),
                np.int32(1 if has_bias else 0),
            )
            out_shape = (n, o, ho, wo)
            x_shape = (n, c, h, w)
            grid = (cdiv(nt, tt),)
            block = (tt * o,)
            cp = self.cp
            box: List[Any] = []

            def call(node, args, attrs):
                x = args[0]
                # 图输入可能以不同形状 / 非连续视图进来：这两种情况要求
                # 重新解析与可能的一次 ascontiguousarray，交回原路径。
                if x.shape != x_shape or x.dtype != cp.float32 or not x.flags.c_contiguous:
                    if epilogue is not None:
                        return None
                    return self._op_conv2d(node, args, attrs, resident=box)
                u = box[0] if box else None
                if u is None:
                    # 首次：让原实现算 U 并填进盒子，之后的执行才走快路径。
                    return self._conv2d_winograd_fused(
                        x,
                        args[1],
                        args[2] if has_bias else None,
                        resident=box,
                        epilogue=epilogue,
                        extra=extra,
                        extra_args=tuple(args[len(node.inputs) :]),
                        **dims,
                    )
                bias = args[2] if has_bias else None
                out = cp.empty(out_shape, dtype=x.dtype)
                bp = x if bias is None else self._contig(bias)
                fused_key = ("winograd_fused", "f32")
                if epi_key is not None:
                    fused_key += ((epi_key,),)
                self._kernel(
                    fused_key,
                    lambda: self.codegen.render(
                        self.codegen.winograd_fused_source(epilogue=epilogue, extra=extra),
                        cp.float32,
                    ),
                    "winograd_fused",
                )(
                    grid,
                    block,
                    (x, u, bp, out) + static + tuple(args[len(node.inputs) :]),
                    shared_mem=shared,
                )
                return out

            return call
        except Exception:  # noqa: BLE001 - 预解析失败只影响这条优化
            return None

    def _planned_winograd(self, node: Node, epilogue=None, extra=(), epi_key=None):
        """静态 3x3 conv 的四段 Winograd，在计划期展开成闭包。

        ``epilogue`` 非空时，闭包把后继逐元素链内联进输出变换内核：``Y``
        不再物化一次再被链读回来。``args`` 的前 ``len(node.inputs)`` 项是
        conv 自己的输入，之后是按 ``extra`` 顺序排的额外张量（由调用方的
        融合闭包拼上来）。

        ``overhead_compare`` 在 c64 上的分解：burst 0.113 / kernel 0.085 /
        三段内核 + GEMM 合计 0.044 ms。缺口里有三块中间缓冲的 ``cp.empty``、
        二十多个 ``np.int32`` 与三次 ``_grid`` / ``_kernel`` 查表——全都只取决于
        IR。展开后运行期只剩四次分配与四次提交。

        实现选择仍由 ``_pick_conv_kernel`` 给出（与融合版共用
        ``_winograd_static``）。形状 / dtype / 连续性不符时回落 ``_op_conv2d``，
        报错时机与文案不变。

        权重驻留：这条展开只在"权重是 IR 常量"的分支里被采用（见
        ``_prepare_plan``），所以权重变换 ``U`` 首次执行算一次就随计划常驻，
        每轮只做输入变换 + GEMM + 输出变换。与融合版的 ``resident`` 同一语义。
        """
        parsed = self._winograd_static(node, ("winograd",))
        if parsed is None:
            return None
        dims, has_bias = parsed
        n, c, h, w, o = dims["n"], dims["c"], dims["h"], dims["w"], dims["o"]
        ho, wo, ph, pw = dims["ho"], dims["wo"], dims["ph"], dims["pw"]
        try:
            th, tw = cdiv(ho, 2), cdiv(wo, 2)
            nt = n * th * tw
            u_shape, v_shape, m_shape = (16, o, c), (16, c, nt), (16, o, nt)
            out_shape = (n, o, ho, wo)
            x_shape = (n, c, h, w)
            w_grid, w_block = self._grid("conv2d", o * c)
            i_grid, i_block = self._grid("conv2d", c * nt)
            o_grid, o_block = self._grid("conv2d", o * nt)
            w_static = (np.int32(o), np.int32(c), np.int64(o * c))
            i_static = (
                np.int32(n),
                np.int32(c),
                np.int32(h),
                np.int32(w),
                np.int32(th),
                np.int32(tw),
                np.int32(ph),
                np.int32(pw),
                np.int64(c * nt),
            )
            o_static = (
                np.int32(n),
                np.int32(o),
                np.int32(ho),
                np.int32(wo),
                np.int32(th),
                np.int32(tw),
                np.int32(nt),
                np.int32(1 if has_bias else 0),
                np.int64(o * nt),
            )
            cp = self.cp
            dtype = np.dtype(cp.float32)
            holder: List[Any] = [None, None]  # [内核三元组, 驻留的 U]
            blas = self._cublas()
            gemm = (16, o * c, c * nt, o * nt)

            def call(node, args, attrs):
                x = args[0]
                if x.shape != x_shape or x.dtype != cp.float32 or not x.flags.c_contiguous:
                    if epilogue is not None:
                        # 形状 / 连续性不符时无法复用这份展开。带 epilogue 的
                        # 情形不能回落到 ``_op_conv2d``（那会丢掉链），交回
                        # 逐节点执行才是正确的。
                        return None
                    return self._op_conv2d(node, args, attrs)
                kernels = holder[0]
                if kernels is None:
                    out_key = ("winograd_output", "f32")
                    if epi_key is not None:
                        out_key += ((epi_key,),)
                    kernels = holder[0] = (
                        self._kernel(
                            ("winograd_weight", "f32"),
                            lambda: self.codegen.render(
                                self.codegen.winograd_weight_source(), dtype
                            ),
                            "winograd_weight",
                        ),
                        self._kernel(
                            ("winograd_input", "f32"),
                            lambda: self.codegen.render(
                                self.codegen.winograd_input_source(), dtype
                            ),
                            "winograd_input",
                        ),
                        self._kernel(
                            out_key,
                            lambda: self.codegen.render(
                                self.codegen.winograd_output_source(epilogue=epilogue, extra=extra),
                                dtype,
                            ),
                            "winograd_output",
                        ),
                    )
                k_weight, k_input, k_output = kernels
                # ``U = G g G^T`` 只依赖权重，而这条展开只在权重是常量的分支里
                # 被采用（调用点即判据）：首次执行算一次、随计划驻留，之后每轮
                # 直接读。``v`` / ``m`` 每轮都被写满且是内核输出，不驻留——
                # 复用它们会让同一张图的并发执行互相覆盖。
                u = holder[1]
                if u is None:
                    u = holder[1] = cp.empty(u_shape, dtype=x.dtype)
                    k_weight((w_grid,), (w_block,), (args[1], u) + w_static)
                v = cp.empty(v_shape, dtype=x.dtype)
                m = cp.empty(m_shape, dtype=x.dtype)
                out = cp.empty(out_shape, dtype=x.dtype)
                k_input((i_grid,), (i_block,), (x, v) + i_static)
                blas.set_stream(self.current_stream_ptr())
                blas.sgemm_batched(
                    u.reshape(u_shape), v.reshape(v_shape), m.reshape(m_shape), *gemm
                )
                bp = x if not has_bias else self._contig(args[2])
                k_output(
                    (o_grid,), (o_block,), (m, bp, out) + o_static + tuple(args[len(node.inputs) :])
                )
                return out

            return call
        except Exception:  # noqa: BLE001 - 预解析失败只影响这条优化
            return None

    def _planned_kernel(self, node: Node):
        """按 IR 声明预判能否跳过 ``invoke`` 的校验；能就返回内核函数。

        与 ``invoke`` 的检查逐条对应：注册表里有这个算子；fp8 只出现在
        ``cast`` 上（其它算子遇到 fp8 要留给 ``invoke`` 报错）；整数下标的
        位置（``_INDEX_ARGS``）声明为 i32/i8；其余输入声明为 f32，或 f16 /
        bf16 且该算子在该支持面内。返回 ``None`` 表示"这条不能预判"，走原路径。

        ``cast`` 是唯一承接 fp8 的算子（5.325）：fp8 的输入 / 输出都是 f32
        载体，快路径末尾会按 ``out_dtype`` 调 ``quantize`` 把数值落回格点。
        """
        name = node.op.name
        fn = self.kernels.get(name)
        if fn is None:
            return None
        if node.output.type.dtype in FP8_DTYPES and name != "cast":
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
            if dtype in FP8_DTYPES and name == "cast":
                continue
            if dtype == DType.F16 and name in self._f16_ops:
                continue
            if dtype == DType.BF16 and name in self._bf16_ops:
                continue
            return None
        return fn

    # ------------------------------------------------------------------
    # 逐元素融合（语义级 L2）
    # ------------------------------------------------------------------

    def group_nodes(self, graph):
        """相邻的同形逐元素算子交给一个内核（计划见 ``_group_elementwise``）。"""
        groups = self._merge_conv_epilogue(graph, _group_elementwise(graph))
        return self._merge_transpose_matmul(graph, groups)

    def _merge_transpose_matmul(self, graph, groups):
        """只合并相邻、单消费者的二维转置；其他布局与 HIP 保持原路径。"""
        if not getattr(self, "GEMM_TRANSPOSE", False):
            return groups
        users = {}
        for node in graph.nodes:
            for value in node.inputs:
                users.setdefault(value.id, []).append(node)
        outputs = {value.id for value in graph.outputs}
        merged = []
        for group in groups:
            if len(group) == 1 and group[0].op.name == "matmul":
                matmul = group[0]
                selected = []
                wanted = {value.id for value in matmul.inputs}
                while merged and len(selected) < 2:
                    prev = merged[-1]
                    node = prev[0]
                    if not (
                        len(prev) == 1 and node.op.name == "transpose"
                        and tuple(node.attrs["perm"]) == (1, 0)
                        and node.output.id in wanted
                        and node.output.id not in outputs
                        and users.get(node.output.id) == [matmul]
                        and all(v.type.is_static and len(v.type.shape) == 2
                                and all(int(d) > 0 for d in v.type.shape)
                                and v.type.dtype in (DType.F32, DType.F16, DType.BF16)
                                for v in matmul.inputs)
                    ):
                        break
                    selected.insert(0, merged.pop()[0])
                if selected:
                    group = tuple(selected) + group
            merged.append(group)
        return merged

    def _planned_transpose_matmul(self, group):
        """用源矩阵和 GEMM 转置位替代组内转置副本。"""
        nodes = group.nodes
        positions = {node.output.id: i for i, node in enumerate(nodes[:-1])}
        matmul = nodes[-1]
        sources = tuple(positions.get(value.id) for value in matmul.inputs)
        shape = tuple(int(d) for d in matmul.output.type.shape)

        def call(args):
            arrays = [args[-1][i] if pos is None else args[pos][0]
                      for i, pos in enumerate(sources)]
            if any(a.ndim != 2 or not a.flags.c_contiguous for a in arrays):
                return None
            a, b = arrays
            if a.dtype != b.dtype or a.dtype not in (
                self.cp.float32, self.cp.float16, self.cp.uint16
            ):
                return None
            out = self.cp.empty(shape, dtype=a.dtype)
            blas = self._cublas()
            blas.set_stream(self.current_stream_ptr())
            blas.gemm_transposed(a, b, out, sources[0] is not None, sources[1] is not None)
            return out

        return call

    def _merge_conv_epilogue(self, graph: Graph, groups: List[tuple]) -> List[tuple]:
        """把单个 conv 节点与紧随其后的逐元素链并成一组。

        这是 epilogue 融合的**静态前提**，所有条件都必须在这里成立，因为
        "合并之后又融合不了"会让这条链退回逐节点执行（比不合并更慢）：

        - conv 的输出只被链首使用，且不是图输出（否则中间值必须可见）；
        - 链里每个节点都在逐元素表里、形状与 dtype 与 conv 输出一致；
        - 链的额外输入只能是标量（大小 1）或逐通道 ``(1,C,1,1)``：conv
          内核按输出通道 ``o`` 索引它们，同形的第二张大张量不存在（conv
          只产出一个输出），也无法在写回点内联。

        目前只对 f32 生效：f16 / bf16 的图里 conv 输出会先舍入到存储类型，
        融合后一路在 float 里算会改变数值。
        """
        if not groups:
            return groups
        users: Dict[int, List[Node]] = {}
        for node in graph.nodes:
            for a in node.inputs:
                users.setdefault(a.id, []).append(node)
        out_ids = {v.id for v in graph.outputs}

        merged: List[tuple] = []
        i = 0
        while i < len(groups):
            group = groups[i]
            if (
                i + 1 < len(groups)
                and len(group) == 1
                and group[0].op.name == "conv2d"
                and group[0].output.type.dtype is DType.F32
            ):
                conv = group[0]
                nxt = groups[i + 1]
                mid = conv.output
                if (
                    mid.id not in out_ids
                    and users.get(mid.id) == [nxt[0]]
                    and all(_chainable(n) for n in nxt)
                    and all(
                        n.output.type.dtype is DType.F32 and n.output.type.shape == mid.type.shape
                        for n in nxt
                    )
                    and self._epilogue_inputs_ok(conv, nxt)
                ):
                    merged.append(group + nxt)
                    i += 2
                    continue
            merged.append(group)
            i += 1
        return merged

    @staticmethod
    def _epilogue_inputs_ok(conv: Node, chain: tuple) -> bool:
        """链的额外输入只能是标量或逐通道广播（见 ``_merge_conv_epilogue``）。"""
        internal = {node.output.id for node in chain}
        internal.add(conv.output.id)
        out_shape = conv.output.type.shape
        for node in chain:
            for value in node.inputs:
                if value.id in internal:
                    continue
                t = value.type
                if not t.is_static:
                    return False
                size = 1
                for d in t.shape:
                    size *= int(d)
                if size == 1:
                    continue
                if channel_broadcast(t.shape, out_shape) is None:
                    return False
        return True

    def invoke_group(self, group, args, attrs, plan=None):
        """一组节点：能融合就一次内核算完，否则逐节点执行。

        运行期还要再查一次（形状是否真的相同、是否连续、指针是否 16 字节
        对齐、长度能否被 4 整除），因此静态计划即使乐观也不会算错——最多
        白算一次计划，然后走回退路径。

        融合失败后交给父类：``plan.kernels`` 有值时走批量派发快路径，
        否则仍是原来的逐节点 ``invoke``。
        """
        if len(group) > 1:
            if group[0].op.name in ("conv2d", "transpose"):
                # conv 后融合与转置矩阵乘都不是逐元素链，不能交给
                # ``_try_chain``。``fused`` 为 ``None``（计划期判定不成立）或
                # 运行期前提不满足时，逐节点执行仍是正确的回退路径。
                fused = plan.fused if plan is not None else None
                if fused is not None:
                    out = fused(args)
                    if out is not None:
                        return out
                return super().invoke_group(group, args, attrs, plan=plan)
            chain = plan.chain if plan is not None else None
            if chain is not None:
                out = chain(group, args)
                if out is not None:
                    return out
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
            steps.append(
                ElemStep(
                    out=local,
                    op=node.op.name,
                    args=tuple(arg_names),
                    params=_step_params(node),
                )
            )

        arrays = [arr for _, arr in entries]
        if not arrays:
            return None
        dtype = arrays[0].dtype
        if dtype not in (cp.float32, cp.float16, cp.uint16):
            return None
        if dtype == cp.uint16 and (
            not self._bf16_ops or any(node.op.name not in self._bf16_ops for node in group)
        ):
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
            shape = tuple(arr.shape)
            if shape == out_shape:
                specs.append(ElemInput(param_of[vid], scalar=False))
            elif arr.size == 1:
                specs.append(ElemInput(param_of[vid], scalar=True))
            else:
                chan = channel_broadcast(shape, out_shape)
                if chan is None:
                    return None
                specs.append(ElemInput(param_of[vid], channel=chan))

        n = int(np.prod(out_shape))
        if n == 0:
            return None

        out = cp.empty(out_shape, dtype=dtype)
        if dtype == cp.float32:
            ir_dtype = DType.F32
        elif dtype == cp.float16:
            ir_dtype = DType.F16
        else:
            ir_dtype = DType.BF16
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
            tuple((s.out, s.op, s.args, s.params) for s in steps),
            # channel 必须进键：同一个链形状在不同 H*W / C 下生成的索引
            # 表达式不同，只按名字与 scalar 复用会拿错内核。
            tuple((s.name, s.scalar, s.channel) for s in specs),
            vec,
            self._kind(arrays[0]),
        )
        kernel = self._kernel(
            key,
            lambda: self.codegen.render(
                build_chain(steps, specs, vec=vec, dtype=ir_dtype, kernel_name="ew_chain"),
                dtype,
            ),
            "ew_chain",
        )
        kernel((launch.grid,), (launch.block,), (*arrays, out, np.int64(n)))
        return out

    def _planned_conv_epilogue(self, group):
        """把 conv 后继的逐元素链内联进 conv 内核的写回点。

        条件由 ``_merge_conv_epilogue`` 在分组时保证；这里只负责把链翻译成
        表达式并绑定到 Winograd 的输出变换（四段版或融合版）。链的**组外**
        输入按形状分成两类：

        - 大小 1 的标量：``e0[0]``，四条通道共用；
        - ``(1,C,1,1)`` / ``(C,1,1)`` 的逐通道参数：``e0[oo]``，用 conv 内核
          里已有的输出通道变量索引，不需要再按展平下标反算通道。

        实现选择与 ``_op_conv2d`` 走同一份判据（``_winograd_static`` 里的
        ``_pick_conv_kernel``）：任何一条不满足就返回 ``None``，整组退回逐
        节点执行。回退是**正确性优先**的：conv 先落一次全局内存，链再读回来，
        与没有这个优化时逐位相同。
        """
        conv = group.nodes[0]
        internal = {node.output.id for node in group.nodes[:-1]}
        extra: List[Tuple[str, str]] = []
        slots: List[Tuple[int, int]] = []
        named: Dict[int, Tuple[str, str]] = {}
        # 只收**链**的组外输入：conv 自己的输入（x / w / bias）不是
        # epilogue 的操作数，收进来只会给内核多加几个用不到的指针。
        for j, node in enumerate(group.nodes[1:], start=1):
            for k, value in enumerate(node.inputs):
                if value.id in internal or value.id in named:
                    continue
                t = value.type
                size = 1
                for d in t.shape:
                    size *= int(d)
                kind = "scalar" if size == 1 else "channel"
                name = f"e{len(extra)}"
                extra.append((name, kind))
                slots.append((j, k))
                named[value.id] = (name, kind)

        conv_out = conv.output.id

        def build(base, channel):
            local = {conv_out: base}
            for node in group.nodes[1:]:
                args = []
                for value in node.inputs:
                    if value.id in local:
                        args.append(local[value.id])
                        continue
                    name, kind = named[value.id]
                    args.append(Index(name, Num("0")) if kind == "scalar" else Index(name, channel))
                local[node.output.id] = apply_op(node.op.name, args, _step_params(node))
            return local[group.nodes[-1].output.id]

        epi_key = (
            tuple((node.op.name, _step_params(node)) for node in group.nodes[1:]),
            tuple(extra),
        )
        prepared = self._planned_winograd(conv, build, tuple(extra), epi_key)
        if prepared is None:
            prepared = self._planned_winograd_fused(conv, build, tuple(extra), epi_key)
        if prepared is None:
            return None

        def fused(group_args):
            extras = [group_args[j][k] for j, k in slots]
            return prepared(conv, tuple(group_args[0]) + tuple(extras), None)

        return fused

    def _planned_chain(self, group):
        """静态逐元素组在计划期展开，运行期只剩分配与一次内核调用。

        ``_try_chain`` 每次执行都要重新推导参数顺序、广播形状、线程铺法，
        再组装源码键——8 层 conv-bn-relu 上实测约 60 µs/层，比它自己的内核
        长一个数量级（DESIGN 5.261）。这些量与绑定值无关，可以一次算好。

        运行期只核对每个外部数组的形状、dtype、连续性与 16 字节对齐；任一
        不符就返回 ``None``，调用方回落 ``_try_chain``（它会按实际情况重选
        vec）。因此这里的乐观假设不会算错，最多白算一次。

        内核本身仍是惰性获取：计划期只构造键与源码生成器，第一次真正执行
        时才编译，之后把句柄留在闭包里，不再查表。
        """
        nodes = group.nodes
        out_type = nodes[-1].output.type
        if not out_type.is_static:
            return None
        out_shape = tuple(int(v) for v in out_type.shape)
        if not out_shape:
            return None
        ir_dtype = out_type.dtype
        if ir_dtype not in (DType.F32, DType.F16, DType.BF16):
            return None
        if ir_dtype is DType.BF16 and (
            not self._bf16_ops or any(node.op.name not in self._bf16_ops for node in nodes)
        ):
            return None
        internal = {node.output.id for node in nodes[:-1]}
        order: List[int] = []
        param_of: Dict[int, str] = {}
        local_of: Dict[int, str] = {}
        steps: List[ElemStep] = []
        specs: List[ElemInput] = []
        want: List[Tuple[int, ...]] = []
        for idx, node in enumerate(nodes):
            names = []
            for value in node.inputs:
                if value.id in internal:
                    names.append(local_of[value.id])
                    continue
                name = param_of.get(value.id)
                if name is None:
                    name = f"p{len(param_of)}"
                    param_of[value.id] = name
                    value_type = value.type
                    if not value_type.is_static:
                        return None
                    shape = tuple(int(v) for v in value_type.shape)
                    if shape == out_shape:
                        specs.append(ElemInput(name, scalar=False))
                    elif int(np.prod(shape)) == 1:
                        specs.append(ElemInput(name, scalar=True))
                    else:
                        chan = channel_broadcast(shape, out_shape)
                        if chan is None:
                            return None
                        specs.append(ElemInput(name, channel=chan))
                    order.append(value.id)
                    want.append(shape)
                names.append(name)
            local = f"t{idx}"
            local_of[node.output.id] = local
            steps.append(
                ElemStep(
                    out=local,
                    op=node.op.name,
                    args=tuple(names),
                    params=_step_params(node),
                )
            )
        if not order:
            return None
        n = 1
        for d in out_shape:
            n *= d
        if n == 0:
            return None

        cp = self.cp
        if ir_dtype is DType.F32:
            cp_dtype = cp.float32
        elif ir_dtype is DType.F16:
            cp_dtype = cp.float16
        else:
            cp_dtype = cp.uint16
        # ``codegen.render`` 要的是 ``numpy.dtype`` 实例；``cp.float32`` 是类型
        # 对象，直接传进去会被当成"未知 dtype"拒绝。
        render_dtype = np.dtype(cp_dtype)
        kind = {DType.F32: "f32", DType.F16: "f16", DType.BF16: "bf16"}[ir_dtype]
        # 对齐只有运行期才知道：先按对齐选铺法，闭包入口再核对指针。
        launch = plan_elementwise(
            n,
            block=self.block,
            max_blocks=self.max_blocks,
            dtype=ir_dtype.value,
            aligned=True,
            target_blocks=self.sm_count,
        )
        key = (
            "chain",
            tuple((s.out, s.op, s.args, s.params) for s in steps),
            # channel 必须进键：同一个链形状在不同 H*W / C 下生成的索引
            # 表达式不同，只按名字与 scalar 复用会拿错内核。
            tuple((s.name, s.scalar, s.channel) for s in specs),
            launch.vec,
            kind,
        )
        grid = (launch.grid,)
        block = (launch.block,)
        n_arg = np.int64(n)
        holder: List[Any] = [None]

        def call(group, args):
            arrays = []
            seen = set()
            for node, node_args in zip(group, args):
                for value, arr in zip(node.inputs, node_args):
                    if value.id in internal or value.id in seen:
                        continue
                    seen.add(value.id)
                    arrays.append(arr)
            if len(arrays) != len(want):
                return None
            for arr, shape in zip(arrays, want):
                if tuple(arr.shape) != shape or arr.dtype != cp_dtype:
                    return None
                if not arr.flags.c_contiguous or int(arr.data.ptr) % VEC4_ALIGN:
                    return None
            kernel = holder[0]
            if kernel is None:
                kernel = holder[0] = self._kernel(
                    key,
                    lambda: self.codegen.render(
                        build_chain(
                            steps,
                            specs,
                            vec=launch.vec,
                            dtype=ir_dtype,
                            kernel_name="ew_chain",
                        ),
                        render_dtype,
                    ),
                    "ew_chain",
                )
            out = cp.empty(out_shape, dtype=cp_dtype)
            kernel(grid, block, (*arrays, out, n_arg))
            return out

        return call
