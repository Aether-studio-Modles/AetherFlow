"""卷积：im2col / tiled / implicit / 分组 / Winograd 五条实现路径。"""

from __future__ import annotations

from ..lower.plan import (
    CONV_SHARED_LIMIT,
    CONV_TILED_MAX_K,
    CONV_WINOGRAD_FUSED_MIN_C,
    CONV_WINOGRAD_FUSED_RESIDENT_MIN_C,
    CONV_WINOGRAD_TILE,
    cdiv,
    conv_implicit_fits,
    conv_implicit_preferred,
    conv_oc_tile,
    conv_tiled_preferred,
    conv_winograd_fused_preferred,
    conv_winograd_preferred,
    conv_winograd_supported,
    plan_conv_implicit,
    plan_conv_tiled,
)
from .base import BackendCapabilityError
from ..tuning.signature import conv2d_signature
import numpy as np
from .device_shared import _pair


class _ConvMixin:
    """卷积：im2col / tiled / implicit / 分组 / Winograd 五条实现路径。"""

    # ------------------------------------------------------------------
    # 卷积：分块直算 / im2col + cuBLAS
    # ------------------------------------------------------------------

    def _op_conv2d(self, node, args, attrs, *, resident=None):
        """NCHW conv2d，两条实现路径语义完全一致（交叉相关、不翻转核）。

        - 分块直算（``_conv2d_tiled``）：不再物化列矩阵，权重缓存在 shared
          memory 里按行复用，适合 K 较小、输出较大的常见 CNN 形状。
        - im2col + GEMM（``_conv2d_im2col``）：把卷积变成一次批量 GEMM，
          K 很大时 GEMM 更吃得饱，也是分块核放不下时的兜底。

        具体走哪条由 ``_pick_conv_impl`` 决定；选择依据写在那里，不散落在此。

        ``resident`` 是执行计划给融合 Winograd 提供的权重变换缓存盒子
        （见 ``_prepare_plan``）；只有权重来自常量节点的 conv 才有，其余
        情况下为 ``None``，权重变换照旧每次重算。
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
            raise BackendCapabilityError(f"conv2d 的 bias 必须是 ({o},)，收到 {bias.shape}")

        cg, og = c // groups, o // groups
        k = cg * kh * kw
        if n * o * ho * wo == 0:
            return cp.empty((n, o, ho, wo), dtype=x.dtype)

        dims = dict(
            n=n,
            c=cg,
            h=h,
            w=w,
            o=og,
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
        # auto 模式下规模够大才用 cuDNN：卷积本体比自研核快数倍（5.69），但
        # 走框架要付描述符 / 算法查询 / 独立 bias 核的固定开销，1 GFLOP 以下
        # 反而更慢（5.252 的同形状 A/B）。加载不到或这一步失败都回落到自研路径。
        flops = 2.0 * n * o * ho * wo * k / 1e9
        if self._pick_cudnn(x.dtype, groups, flops):
            fast = self._conv2d_cudnn(x, weight, bias, **dims)
            if fast is not None:
                return fast
        if x.dtype == cp.float16:
            # 分块核与隐式 GEMM 核的 shared memory 与累加器都是 float，只覆盖
            # f32 路径。f16 走 im2col + hgemm：GEMM 内部同样是 f32 累加，
            # 语义不变。
            if self.conv_impl in ("tiled", "implicit", "winograd", "winograd_fused"):
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
            if self.conv_impl in ("tiled", "implicit", "winograd", "winograd_fused"):
                raise BackendCapabilityError(
                    f"conv2d 的 {self.conv_impl} 内核只支持 groups=1；"
                    "分组卷积请用 conv_impl='auto'、'grouped' 或 'im2col'"
                )
            if self.conv_impl == "grouped" or (self.conv_impl == "auto" and groups == c):
                kernel_kind = "grouped"
            else:
                kernel_kind = "im2col"
        elif self.conv_impl == "winograd":
            # Winograd 的系数矩阵只对 3x3 / stride=1 / dilation=1 成立，
            # 显式指定时不支持就明确失败，不静默换成别的实现。
            if not conv_winograd_supported(kh=kh, kw=kw, sh=sh, sw=sw, dh=dh, dw=dw):
                raise BackendCapabilityError(
                    "conv2d 的 winograd 内核只支持 3x3、stride=1、dilation=1，"
                    f"收到核 {kh}x{kw}、stride {(sh, sw)}、dilation {(dh, dw)}；"
                    "请用 conv_impl='auto' 或 'im2col'"
                )
            kernel_kind = "winograd"
        elif self.conv_impl == "winograd_fused":
            # 融合版的适用面与四段版相同（同一组系数），额外受 shared 预算
            # 约束：``V`` 要整块留在共享内存里，``16*C*TT*4`` 字节。
            if not conv_winograd_supported(kh=kh, kw=kw, sh=sh, sw=sw, dh=dh, dw=dw):
                raise BackendCapabilityError(
                    "conv2d 的 winograd_fused 内核只支持 3x3、stride=1、dilation=1，"
                    f"收到核 {kh}x{kw}、stride {(sh, sw)}、dilation {(dh, dw)}；"
                    "请用 conv_impl='auto' 或 'im2col'"
                )
            need = 16 * c * CONV_WINOGRAD_TILE * 4
            if need > CONV_SHARED_LIMIT:
                raise BackendCapabilityError(
                    f"conv2d 的 winograd_fused 内核需要 {need} B 共享内存"
                    f"（C={c}、tile={CONV_WINOGRAD_TILE}），超过预算 "
                    f"{CONV_SHARED_LIMIT} B；请用 conv_impl='auto' 或 'winograd'"
                )
            kernel_kind = "winograd_fused"
        else:
            kernel_kind = self._pick_conv_kernel(weight_resident=resident is not None, **dims)
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
            elif kernel_kind == "winograd":
                # Winograd 的输出变换同样融 bias：最后一个内核写回时就加
                out = self._conv2d_winograd(x, weight, bias, **dims)
                bias_fused = True
            elif kernel_kind == "winograd_fused":
                # 融合版的写回同样融 bias；常量权重可跨执行复用权重变换
                out = self._conv2d_winograd_fused(x, weight, bias, resident=resident, **dims)
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
        if self.conv_impl == "auto" and not conv_tiled_preferred(k, n * o * ho * wo):
            return False
        if conv_oc_tile(kh * kw, o) is None:
            raise BackendCapabilityError(
                f"conv2d 分块核的 shared memory 预算放不下一个输出通道"
                f"（核 {kh}x{kw} 需要 {kh * kw * 4} B，预算 {CONV_SHARED_LIMIT} B）；"
                "请用 conv_impl='auto' 或 'im2col'"
            )
        return True

    def _pick_conv_kernel(
        self, *, n, o, k, kh, kw, ho, wo, c, sw, dw, weight_resident: bool = False, **_ignored
    ) -> str:
        """在 winograd / tiled / implicit / im2col 之间选一个实现。

        ``auto`` 先看 Winograd 的适用面，再看隐式 GEMM 的甜点区，再看分块
        直算，最后回落到 im2col。

        Winograd（``CONV_AUTO_WINOGRAD``，CUDA，5.255）：3x3 / stride=1 /
        dilation=1 且 ``k > CONV_TILED_MAX_K``
        （``plan.conv_winograd_preferred``）。A800 稳态上 0.133～0.136 ms，
        同形状 im2col 0.16～0.31 ms；3x3_c3（K=27）仍留给分块直算
        （0.077 ms）。

        隐式有两条互斥的入口：

        - ``CONV_AUTO_IMPLICIT_MID``（CUDA，5.252）：K 在 (32, 400] 的
          3x3/5x5，且 im2col 的物化缓冲 ``n*ho*wo*K`` 不超过 3.2M 元素
          （``plan.conv_implicit_preferred``）。A800 上这一档比 im2col 快
          7～13%，与同形状的分块直算持平或略快；
        - ``CONV_AUTO_IMPLICIT``（HIP，5.238）：K 超过分块阈值就交给隐式，
          5.238 的 A/B 显示中等通道 3x3/5x5 快 1.33～1.63×。

        CUDA 的旧结论（5.151，M40）是"隐式在 K >= 288 输给 im2col 1.37×"，
        它只在物化缓冲超过 12.8 MB 或 K > 400 时成立，所以甜点区之外仍交给
        im2col——隐式仍是显式 ``conv_impl="implicit"`` 的对照路径（数值与
        im2col 一致，相对误差 ≤ 1.7e-6）。

        两条入口都要过 ``kh*kw > 1``（1x1 没有窗口重叠，隐式反而慢 6%）、
        ``k > CONV_TILED_MAX_K``（小 K 的分块直算仍更快）与 shared 预算。

        ``weight_resident`` 表示这一层的权重来自常量节点、``U`` 可以驻留。
        它只改融合版的下限（c3 也划算，见 5.257），其余判据不变——图输入
        权重无法证明"值没变"，仍按 8 起。
        """
        if self.conv_impl == "implicit":
            return "implicit"
        if (
            self.conv_impl == "auto"
            and self.CONV_AUTO_WINOGRAD
            and conv_winograd_fused_preferred(
                c=c,
                kh=kh,
                kw=kw,
                sw=sw,
                dw=dw,
                o=o,
                work=n * o * ho * wo,
                # 权重是常量时权重变换那次 launch 不进每轮成本，下限放到 c3
                min_c=(
                    CONV_WINOGRAD_FUSED_RESIDENT_MIN_C
                    if weight_resident
                    else CONV_WINOGRAD_FUSED_MIN_C
                ),
            )
        ):
            return "winograd_fused"
        if (
            self.conv_impl == "auto"
            and self.CONV_AUTO_WINOGRAD
            and conv_winograd_preferred(k=k, kh=kh, kw=kw, sw=sw, dw=dw, work=n * o * ho * wo)
        ):
            return "winograd"
        if (
            self.conv_impl == "auto"
            and self.CONV_AUTO_IMPLICIT_MID
            and conv_implicit_preferred(n=n, k=k, kh=kh, kw=kw, ho=ho, wo=wo)
            and conv_implicit_fits(
                kh=kh,
                kw=kw,
                c=c,
                o=o,
                wo=wo,
                sw=sw,
                dw=dw,
                tile_w=self._conv_tile,
            )
        ):
            return "implicit"
        if (
            self.conv_impl == "auto"
            and self.CONV_AUTO_IMPLICIT
            and kh * kw > 1
            and k > CONV_TILED_MAX_K
            and conv_implicit_fits(
                kh=kh,
                kw=kw,
                c=c,
                o=o,
                wo=wo,
                sw=sw,
                dw=dw,
                tile_w=self._conv_tile,
            )
        ):
            return "implicit"
        if self._use_tiled_conv(n=n, o=o, k=k, kh=kh, kw=kw, ho=ho, wo=wo):
            return "tiled"
        return "im2col"

    def _conv2d_implicit(
        self, x, weight, bias, *, n, c, h, w, o, kh, kw, ho, wo, sh, sw, ph, pw, dh, dw, k
    ):
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
            kh=kh,
            kw=kw,
            c=c,
            o=o,
            wo=wo,
            sw=sw,
            dw=dw,
            tile_w=self._conv_tile,
            roc=self._conv_roc,
            cb=self._conv_cb,
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
                x,
                weight,
                out,
                bp,
                np.int32(c),
                np.int32(h),
                np.int32(w),
                np.int32(o),
                np.int32(kh),
                np.int32(kw),
                np.int32(ho),
                np.int32(wo),
                np.int32(sh),
                np.int32(sw),
                np.int32(ph),
                np.int32(pw),
                np.int32(dh),
                np.int32(dw),
                np.int32(tiling.toc),
                np.int32(tiling.roc),
                np.int32(tiling.cb),
                np.int32(tiling.oc_tiles),
                np.int32(0 if bias is None else 1),
                np.int32(tiling.span),
            ),
            shared_mem=tiling.shared_bytes,
        )
        return out

    def _conv2d_winograd(
        self,
        x,
        weight,
        bias,
        *,
        n,
        c,
        h,
        w,
        o,
        kh,
        kw,
        ho,
        wo,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        k,
        epilogue=None,
        extra=(),
        extra_args=(),
    ):
        """Winograd F(2x2,3x3)：用 16 个小 GEMM 换掉 9 倍的乘法。

        三次自研 launch + 一次批量 GEMM：

        1. ``winograd_weight``：``U = G g G^T``，布局 ``(16, O, C)``；
        2. ``winograd_input``：``V = B^T d B``，布局 ``(16, C, NT)``，
           其中 ``NT = N * ceil(HO/2) * ceil(WO/2)``；
        3. 16 个 ``(O, C) x (C, NT)`` 的 strided GEMM，一次 cuBLAS 提交；
        4. ``winograd_output``：``Y = A^T M A``，写回时融合 bias。

        三条缓冲的量级（相对 im2col 的 ``N*K*HO*WO = 9*N*C*HO*WO``）：
        ``16*C*NT ≈ 4*N*C*HO*WO``、``16*O*NT ≈ 4*N*O*HO*WO``，比 3x3 的
        列矩阵小一半多。代价是精度：变换系数里的加减会放大舍入误差，
        与 cuDNN 的容差要按 1e-4 单独验（见 ``benchmarks/cudnn_compare.py``）。

        ``kh`` / ``kw`` / ``sh`` / ``sw`` / ``dh`` / ``dw`` 只用来断言——
        调用方已在派发处拒绝非 3x3 stride=1 dilation=1 的形状，这里不重复
        写判据，但保留参数是为了与其它卷积入口同签名。
        """
        cp = self.cp
        th = cdiv(ho, 2)
        tw = cdiv(wo, 2)
        nt = n * th * tw
        u = cp.empty((16, o, c), dtype=x.dtype)
        v = cp.empty((16, c, nt), dtype=x.dtype)
        m = cp.empty((16, o, nt), dtype=x.dtype)
        out = cp.empty((n, o, ho, wo), dtype=x.dtype)
        dtype_key = self._kind(x)

        def launch(name: str, total: int, args, epi=None, epi_extra=()) -> None:
            block, grid = self._grid("conv2d", total)
            # epilogue 改变了输出变换的源码与参数表，内核缓存键必须带上它，
            # 否则同一张图里"带 BN 的 conv"与"不带 BN 的 conv"会共用内核。
            key = (name, dtype_key)
            if epi is not None:
                key += ((tuple(epi_extra),),)
            kernel = self._kernel(
                key,
                lambda: self.codegen.render(
                    getattr(self.codegen, f"{name}_source")(epilogue=epi, extra=epi_extra)
                    if epi is not None
                    else getattr(self.codegen, f"{name}_source")(),
                    x.dtype,
                ),
                name,
            )
            kernel((grid,), (block,), args)

        # ---- 权重变换：U[p][o][c] = (G g G^T)[p] ----
        total = o * c
        launch(
            "winograd_weight",
            total,
            (weight, u, np.int32(o), np.int32(c), np.int64(total)),
        )
        # ---- 输入变换：V[p][c][t] = (B^T d B)[p] ----
        total = c * nt
        launch(
            "winograd_input",
            total,
            (
                x,
                v,
                np.int32(n),
                np.int32(c),
                np.int32(h),
                np.int32(w),
                np.int32(th),
                np.int32(tw),
                np.int32(ph),
                np.int32(pw),
                np.int64(total),
            ),
        )
        # ---- 逐元素乘并沿输入通道求和：16 个 (O, C) x (C, NT) ----
        blas = self._cublas()
        blas.set_stream(self.current_stream_ptr())
        blas.sgemm_batched(
            u.reshape(16, o, c),
            v.reshape(16, c, nt),
            m.reshape(16, o, nt),
            16,
            o * c,
            c * nt,
            o * nt,
        )
        # ---- 输出变换：Y = A^T M A + bias ----
        total = o * nt
        bp = x if bias is None else self._contig(bias)
        launch(
            "winograd_output",
            total,
            (
                m,
                bp,
                out,
                np.int32(n),
                np.int32(o),
                np.int32(ho),
                np.int32(wo),
                np.int32(th),
                np.int32(tw),
                np.int32(nt),
                np.int32(0 if bias is None else 1),
                np.int64(total),
            )
            + tuple(extra_args),
            epi=epilogue,
            epi_extra=extra,
        )
        return out

    def _conv2d_winograd_fused(
        self,
        x,
        weight,
        bias,
        *,
        n,
        c,
        h,
        w,
        o,
        kh,
        kw,
        ho,
        wo,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        k,
        resident=None,
        epilogue=None,
        extra=(),
        extra_args=(),
    ):
        """融合版 F(2x2,3x3)：两次 launch（权重变换 + 融合内核）。

        与 :meth:`_conv2d_winograd` 的差别只在中间量落在哪：``V`` 留在共享
        内存、``M`` 留在寄存器，因此不再有 ``(16, C, NT)`` / ``(16, O, NT)``
        两块全局缓冲，也没有 cuBLAS 的那次提交。数值与四段版逐位一致
        （同一组系数、同样的求和顺序，实测相对 cuDNN ≤ 2e-7）。

        ``U`` 的驻留分两种情况，判据是 IR 结构而不是"指针没变"：

        - 权重来自**常量节点**（``torch.export`` 的 ``get_attr`` / lifted
          张量、图里显式 ``constant``）：执行计划在 ``_prepare_plan`` 里给了
          ``resident`` 盒子，第一次算完就留在盒子里，之后每次执行只跑融合
          内核。权重是常量意味着它的设备副本由计划独占、与计划同寿命，
          "值没变"因此是可证的而不是猜的。
        - 权重是**图输入**（基准脚本的常见形态，可以用新权重覆盖）：没有
          盒子，每次执行重算一次 ``U``。这一次 launch 约 0.008～0.015 ms，
          换来的是"绝不会拿旧权重算新输入"。

        盒子里的 ``U`` 形状/dtype 与本次不符时同样重算并替换：那说明权重
        本身变了，旧 ``U`` 不能再用。
        """
        cp = self.cp
        th = cdiv(ho, 2)
        tw = cdiv(wo, 2)
        nt = n * th * tw
        tt = CONV_WINOGRAD_TILE
        dtype_key = self._kind(x)

        u = resident[0] if resident else None
        if u is None or u.shape != (16, o, c) or u.dtype != x.dtype:
            u = cp.empty((16, o, c), dtype=x.dtype)
            self.conv_weight_transforms += 1
            total = o * c
            block, grid = self._grid("conv2d", total)
            weight_kernel = self._kernel(
                ("winograd_weight", dtype_key),
                lambda: self.codegen.render(self.codegen.winograd_weight_source(), x.dtype),
                "winograd_weight",
            )
            weight_kernel(
                (grid,),
                (block,),
                (weight, u, np.int32(o), np.int32(c), np.int64(total)),
            )
            if resident is not None:
                # 盒子由执行计划持有（``group.kernels`` → 闭包），因此这块
                # 显存与计划同寿命；计划作废时盒子一起消失。
                if resident:
                    resident[0] = u
                else:
                    resident.append(u)

        out = cp.empty((n, o, ho, wo), dtype=x.dtype)
        key = ("winograd_fused", dtype_key)
        if epilogue is not None:
            key += ((tuple(extra),),)
        kernel = self._kernel(
            key,
            lambda: self.codegen.render(
                self.codegen.winograd_fused_source(epilogue=epilogue, extra=extra),
                x.dtype,
            ),
            "winograd_fused",
        )
        bp = x if bias is None else self._contig(bias)
        kernel(
            (cdiv(nt, tt),),
            (tt * o,),
            (
                x,
                u,
                bp,
                out,
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
                np.int32(0 if bias is None else 1),
            )
            + tuple(extra_args),
            shared_mem=16 * c * tt * 4,
        )
        return out

    def _conv2d_grouped(
        self, x, weight, bias, *, n, c, h, w, o, kh, kw, ho, wo, sh, sw, ph, pw, dh, dw, k, groups
    ):
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
            lambda: self.codegen.render(self.codegen.conv2d_grouped_source(), x.dtype),
            "conv2d_grouped",
        )
        bp = x if bias is None else self._contig(bias)
        kernel(
            (grid,),
            (block,),
            (
                x,
                weight,
                out,
                bp,
                np.int32(total_out),
                np.int32(h),
                np.int32(w),
                np.int32(kh),
                np.int32(kw),
                np.int32(ho),
                np.int32(wo),
                np.int32(sh),
                np.int32(sw),
                np.int32(ph),
                np.int32(pw),
                np.int32(dh),
                np.int32(dw),
                np.int32(c),
                np.int32(o),
                np.int32(groups),
                np.int32(0 if bias is None else 1),
                np.int64(total),
            ),
        )
        return out

    def _conv2d_tiled(
        self, x, weight, bias, *, n, c, h, w, o, kh, kw, ho, wo, sh, sw, ph, pw, dh, dw, k
    ):
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
        tile_w, roc, cb = self._conv_tile, self._conv_roc, self._conv_cb
        # 三个旋钮都还是"自动"时才查配置库：调用方手动调过其中任意一项，
        # 说明正在做 A/B，此时再叠库里的值会分不清"这次改的到底起没起作用"。
        policy = self._schedule_policy
        if policy is not None and tile_w is None and roc is None and cb is None:
            cfg = self._scheduled_config(
                "conv2d",
                conv2d_signature(
                    x.shape,
                    weight.shape,
                    stride=(sh, sw),
                    padding=(ph, pw),
                    dilation=(dh, dw),
                    dtype=x.dtype,
                ),
            )
            if cfg is not None:
                merged = policy.overlay(cfg, {"tile_w": None, "roc": None, "cb": None})
                tile_w, roc, cb = merged["tile_w"], merged["roc"], merged["cb"]
        tiling = plan_conv_tiled(
            kh=kh,
            kw=kw,
            c=c,
            o=o,
            ho=ho,
            wo=wo,
            tile_w=tile_w,
            roc=roc,
            cb=cb,
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
                x,
                weight,
                out,
                bp,
                np.int32(c),
                np.int32(h),
                np.int32(w),
                np.int32(o),
                np.int32(kh),
                np.int32(kw),
                np.int32(ho),
                np.int32(wo),
                np.int32(sh),
                np.int32(sw),
                np.int32(ph),
                np.int32(pw),
                np.int32(dh),
                np.int32(dw),
                np.int32(tiling.toc),
                np.int32(tiling.roc),
                np.int32(tiling.cb),
                np.int32(tiling.oc_tiles),
                np.int32(0 if bias is None else 1),
            ),
            shared_mem=tiling.shared_bytes,
        )
        return out

    def _conv2d_im2col(
        self, x, weight, *, n, c, h, w, o, kh, kw, ho, wo, sh, sw, ph, pw, dh, dw, k
    ):
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
                np.int32(c),
                np.int32(h),
                np.int32(w),
                np.int32(kh),
                np.int32(kw),
                np.int32(ho),
                np.int32(wo),
                np.int32(sh),
                np.int32(sw),
                np.int32(ph),
                np.int32(pw),
                np.int32(dh),
                np.int32(dw),
                np.int64(total),
            ),
        )

        blas = self._cublas()
        blas.set_stream(self.current_stream_ptr())
        # stride_a=0：同一份权重服务所有 batch
        if x.dtype == cp.float16:
            blas.hgemm_batched(weight.reshape(o, k), cols, out, n, 0, k * ho * wo, o * ho * wo)
        else:
            blas.sgemm_batched(weight.reshape(o, k), cols, out, n, 0, k * ho * wo, o * ho * wo)
        return out.reshape((n, o, ho, wo))
