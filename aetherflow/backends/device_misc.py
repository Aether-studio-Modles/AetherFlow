"""杂项：补零与卷积反向（对输入 / 对权重）。"""

from __future__ import annotations

from .base import BackendCapabilityError
import numpy as np


class _MiscMixin:
    """杂项：补零与卷积反向（对输入 / 对权重）。"""

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
            raise BackendCapabilityError(f"pad 的 before/after 必须 >= 0，收到 {before} 与 {after}")
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
            raise BackendCapabilityError(f"conv2d_input_grad 的 groups 必须 >= 1，得到 {groups}")
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
        (sh, sw), (ph, pw), (dil_h, dil_w) = self._conv_geometry(attrs, "conv2d_input_grad")
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
                    np.int32(c_g),
                    np.int32(h),
                    np.int32(w),
                    np.int32(o_g),
                    np.int32(kh),
                    np.int32(kw),
                    np.int32(ho),
                    np.int32(wo),
                    np.int32(sh),
                    np.int32(sw),
                    np.int32(ph),
                    np.int32(pw),
                    np.int32(dil_h),
                    np.int32(dil_w),
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
            raise BackendCapabilityError(f"conv2d_weight_grad 的 groups 必须 >= 1，得到 {groups}")
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
        (sh, sw), (ph, pw), (dil_h, dil_w) = self._conv_geometry(attrs, "conv2d_weight_grad")
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
                    np.int32(n),
                    np.int32(c_g),
                    np.int32(h),
                    np.int32(w),
                    np.int32(o_g),
                    np.int32(kh),
                    np.int32(kw),
                    np.int32(ho),
                    np.int32(wo),
                    np.int32(sh),
                    np.int32(sw),
                    np.int32(ph),
                    np.int32(pw),
                    np.int32(dil_h),
                    np.int32(dil_w),
                    np.int64(total_g),
                ),
            )
        return dw
