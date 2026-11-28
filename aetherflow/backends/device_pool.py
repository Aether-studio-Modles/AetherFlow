"""池化：最大 / 平均 / 自适应平均，含反向与取值位置。"""

from __future__ import annotations

from .base import BackendCapabilityError
import numpy as np
from .device_shared import _pair, _pool_flags, _pool_out_dim


def _interp_geometry(attrs, where: str) -> tuple:
    """``(mode, align_corners, OH, OW)``；与 ``ir/ops.py`` 的解析逐字一致。"""
    mode = attrs.get("mode", "nearest")
    if mode not in ("nearest", "bilinear"):
        raise BackendCapabilityError(
            f"{where} 只支持 mode='nearest' / 'bilinear'，收到 {mode!r}"
        )
    align = attrs.get("align_corners", False)
    if not isinstance(align, bool):
        raise BackendCapabilityError(
            f"{where} 的 align_corners 必须是 bool，收到 {align!r}"
        )
    if mode == "nearest" and align:
        raise BackendCapabilityError(
            f"{where} 的 nearest 模式不接受 align_corners=True"
        )
    raw = attrs.get("size")
    if isinstance(raw, int) and not isinstance(raw, bool):
        oh = ow = int(raw)
    elif isinstance(raw, (tuple, list)) and len(raw) in (1, 2):
        oh = int(raw[0])
        ow = int(raw[-1])
    else:
        raise BackendCapabilityError(
            f"{where} 的 size 必须是 int 或长度 1/2 的序列，收到 {raw!r}"
        )
    if oh < 1 or ow < 1:
        raise BackendCapabilityError(f"{where} 的 size 必须 >= 1，收到 {(oh, ow)}")
    return mode, align, oh, ow


class _PoolMixin:
    """池化：最大 / 平均 / 自适应平均，含反向与取值位置。"""

    def _op_maxpool2d(self, node, args, attrs):
        """最大池化：一个输出元素一个线程，纯索引运算。

        与参考语义一致，padding 区域不参与取 max。空窗口填 ``-INFINITY``，
        而不是 0——否则负值输入会在边界处被"抬"到 0。

        ``layout="nhwc"``（5.334）只换下标公式：输入 ``(N,H,W,C)``、
        输出 ``(N,HO,WO,C)``，窗口语义与 NCHW 完全一致。
        """
        cp = self.cp
        layout = attrs.get("layout", "nchw")
        if layout not in ("nchw", "nhwc"):
            raise BackendCapabilityError(
                f"maxpool2d 的 layout 只能是 'nchw' 或 'nhwc'，收到 {layout!r}"
            )
        x = self._contig(args[0])
        if x.ndim != 4:
            raise BackendCapabilityError(
                f"{self._LABEL}的 maxpool2d 要求 NCHW/NHWC 秩 4，收到 {x.shape}"
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

        if layout == "nhwc":
            n, h, w, c = (int(v) for v in x.shape)
        else:
            n, c, h, w = (int(v) for v in x.shape)
        ceil_mode, _ = _pool_flags(attrs, "maxpool2d")
        ho = _pool_out_dim(h, kh, sh, ph, ceil_mode)
        wo = _pool_out_dim(w, kw, sw, pw, ceil_mode)
        if ho <= 0 or wo <= 0:
            raise BackendCapabilityError(
                f"maxpool2d 输出尺寸非法：输入 {h}x{w}、核 {kh}x{kw}、"
                f"padding {(ph, pw)}、stride {(sh, sw)}、ceil_mode={ceil_mode}"
            )

        suffix = "_nhwc" if layout == "nhwc" else ""
        out = cp.empty(
            (n, ho, wo, c) if layout == "nhwc" else (n, c, ho, wo), dtype=x.dtype
        )
        total = int(out.size)
        if total == 0:
            return out

        dtype_key = self._kind(x)
        kernel = self._kernel(
            ("maxpool2d" + suffix, dtype_key),
            lambda: self.codegen.render(self.codegen.maxpool2d_source(layout), x.dtype),
            "maxpool2d" + suffix,
        )
        block, grid = self._grid("maxpool2d", total)
        kernel(
            (grid,),
            (block,),
            (
                x,
                out,
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
            lambda: self.codegen.render(self.codegen.maxpool2d_indices_source(), x.dtype),
            "maxpool2d_indices",
        )
        block, grid = self._grid("maxpool2d_indices", total)
        kernel(
            (grid,),
            (block,),
            (
                x,
                out,
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
                np.int64(total),
            ),
        )
        return out

    def _op_avgpool2d(self, node, args, attrs):
        """平均池化：与 maxpool2d 同一套索引，差别在"累加并按分母除"。

        分母是窗口落在 padded 输入里的格数，由内核按 ``count_include_pad``
        与 ``ceil_mode`` 现算（见 ``lower/kernels.py::avgpool2d``）——``ceil``
        的最后一格会越过 padded 输入，固定除以 ``kh * kw`` 会让角落偏小。

        ``layout="nhwc"``（5.334）只换下标公式：输入 ``(N,H,W,C)``、
        输出 ``(N,HO,WO,C)``。
        """
        cp = self.cp
        layout = attrs.get("layout", "nchw")
        if layout not in ("nchw", "nhwc"):
            raise BackendCapabilityError(
                f"avgpool2d 的 layout 只能是 'nchw' 或 'nhwc'，收到 {layout!r}"
            )
        x = self._contig(args[0])
        if x.ndim != 4:
            raise BackendCapabilityError(
                f"{self._LABEL}的 avgpool2d 要求 NCHW/NHWC 秩 4，收到 {x.shape}"
            )
        (kh, kw), (sh, sw), (ph, pw) = self._pool_geometry(attrs, "avgpool2d")
        ceil_mode, count_include_pad = _pool_flags(attrs, "avgpool2d")

        if layout == "nhwc":
            n, h, w, c = (int(v) for v in x.shape)
        else:
            n, c, h, w = (int(v) for v in x.shape)
        ho = _pool_out_dim(h, kh, sh, ph, ceil_mode)
        wo = _pool_out_dim(w, kw, sw, pw, ceil_mode)
        if ho <= 0 or wo <= 0:
            raise BackendCapabilityError(
                f"avgpool2d 输出尺寸非法：输入 {h}x{w}、核 {kh}x{kw}、"
                f"padding {(ph, pw)}、stride {(sh, sw)}、ceil_mode={ceil_mode}"
            )

        suffix = "_nhwc" if layout == "nhwc" else ""
        out = cp.empty(
            (n, ho, wo, c) if layout == "nhwc" else (n, c, ho, wo), dtype=x.dtype
        )
        total = int(out.size)
        if total == 0:
            return out

        dtype_key = self._kind(x)
        kernel = self._kernel(
            ("avgpool2d" + suffix, dtype_key),
            lambda: self.codegen.render(self.codegen.avgpool2d_source(layout), x.dtype),
            "avgpool2d" + suffix,
        )
        block, grid = self._grid("avgpool2d", total)
        kernel(
            (grid,),
            (block,),
            (
                x,
                out,
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
                np.int64(total),
                np.int32(1 if count_include_pad else 0),
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
                np.int64(total),
                np.int32(1 if count_include_pad else 0),
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
                np.int32(c),
                np.int32(h),
                np.int32(w),
                np.int32(oh),
                np.int32(ow),
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
                f"adaptive_avg_pool2d_backward 的 dy 形状应为 {(n, c, oh, ow)}，收到 {dy.shape}"
            )

        dx = cp.zeros(shape, dtype=dy.dtype)
        total = int(dy.size)
        if total == 0:
            return dx

        dtype_key = self._kind(dy)
        kernel = self._kernel(
            ("adaptive_avg_pool2d_backward", dtype_key),
            lambda: self.codegen.render(
                self.codegen.adaptive_avg_pool2d_backward_source(), dy.dtype
            ),
            "adaptive_avg_pool2d_backward",
        )
        block, grid = self._grid("adaptive_avg_pool2d_backward", total)
        kernel(
            (grid,),
            (block,),
            (
                dy,
                dx,
                np.int32(c),
                np.int32(h),
                np.int32(w),
                np.int32(oh),
                np.int32(ow),
                np.int64(total),
            ),
        )
        return dx

    def _op_interpolate(self, node, args, attrs):
        """二维空间插值：一个输出元素一个线程。

        坐标口径与 ``ir/ops.py`` 的参考实现完全相同（整数分子/分母），因此
        ``nearest`` 只是一次取值，``bilinear`` 是四角加权；``align_corners``
        是构图期属性，两套公式各编译一份，执行期不分支。
        """
        cp = self.cp
        x = self._contig(args[0])
        if x.ndim != 4:
            raise BackendCapabilityError(
                f"{self._LABEL}的 interpolate 要求 NCHW 秩 4，收到 {x.shape}"
            )
        mode, align, oh, ow = _interp_geometry(attrs, "interpolate")
        n, c, h, w = (int(v) for v in x.shape)

        out = cp.empty((n, c, oh, ow), dtype=x.dtype)
        total = int(out.size)
        if total == 0:
            return out

        if mode == "nearest":
            name = "interpolate_nearest"
            source = self.codegen.interpolate_nearest_source
            render = lambda: self.codegen.render(source(), x.dtype)  # noqa: E731
        else:
            name = "interpolate_bilinear" + ("_ac" if align else "_half")
            render = lambda: self.codegen.render(  # noqa: E731
                self.codegen.interpolate_bilinear_source(align), x.dtype
            )

        dtype_key = self._kind(x)
        kernel = self._kernel((name, dtype_key), render, name)
        block, grid = self._grid(name, total)
        kernel(
            (grid,),
            (block,),
            (
                x,
                out,
                np.int32(c),
                np.int32(h),
                np.int32(w),
                np.int32(oh),
                np.int32(ow),
                np.int64(total),
            ),
        )
        return out

    def _op_interpolate_backward(self, node, args, attrs):
        """插值反向：dy 按源位置散射回 dx。

        最近邻一个源点、双线性四个角；多个输出格可能命中同一输入格，因此
        dx 先清零、写回走 atomicAdd（空内核会留下未初始化结果，不能省）。
        """
        cp = self.cp
        dy = self._contig(args[0])
        mode, align, oh, ow = _interp_geometry(attrs, "interpolate_backward")
        shape = tuple(int(d) for d in attrs["input_shape"])
        if len(shape) != 4:
            raise BackendCapabilityError(
                f"{self._LABEL}的 interpolate_backward 的 input_shape "
                f"必须是 NCHW 四元组，收到 {shape}"
            )
        n, c, h, w = shape
        if tuple(dy.shape) != (n, c, oh, ow):
            raise BackendCapabilityError(
                f"interpolate_backward 的 dy 形状应为 {(n, c, oh, ow)}，收到 {dy.shape}"
            )

        dx = cp.zeros(shape, dtype=dy.dtype)
        total = int(dy.size)
        if total == 0:
            return dx

        if mode == "nearest":
            name = "interpolate_nearest_backward"
            render = lambda: self.codegen.render(  # noqa: E731
                self.codegen.interpolate_nearest_backward_source(), dy.dtype
            )
        else:
            name = "interpolate_bilinear_backward" + ("_ac" if align else "_half")
            render = lambda: self.codegen.render(  # noqa: E731
                self.codegen.interpolate_bilinear_backward_source(align), dy.dtype
            )

        dtype_key = self._kind(dy)
        kernel = self._kernel((name, dtype_key), render, name)
        block, grid = self._grid(name, total)
        kernel(
            (grid,),
            (block,),
            (
                dy,
                dx,
                np.int32(c),
                np.int32(h),
                np.int32(w),
                np.int32(oh),
                np.int32(ow),
                np.int64(total),
            ),
        )
        return dx

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
                np.int64(total),
            ),
        )
        return dx
