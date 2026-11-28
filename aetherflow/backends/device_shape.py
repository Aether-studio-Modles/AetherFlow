"""形状与拼接：reshape、transpose、cast、slice、concat。"""

from __future__ import annotations

from ..ir.types import DType
from .device_shared import FP8_DTYPES
import numpy as np


class _ShapeMixin:
    """形状与拼接：reshape、transpose、cast、slice、concat。"""

    # ------------------------------------------------------------------
    # 布局 / 类型
    # ------------------------------------------------------------------

    def _op_reshape(self, node, args, attrs):
        target = tuple(int(d) for d in attrs["shape"])
        src = self._contig(args[0])
        out = self.cp.reshape(src, target)
        # reshape 是零拷贝视图，和 src 共用同一段地址。src 的回收点早于 out 的
        # 最后一次读取，不记账就会被 arena 拿去复用（详见 ``Arena.alias``）。
        return self._shared_view(out, src, node)

    def _op_transpose(self, node, args, attrs):
        perm = tuple(int(p) for p in attrs["perm"])
        out = self.cp.ascontiguousarray(self.cp.transpose(args[0], perm))
        # 恒等置换不拷贝，返回的是输入的视图
        return self._shared_view(out, args[0], node)

    def _op_cast(self, node, args, attrs):
        # 执行骨架也会按 IR 推导的 dtype 调 ``cast`` 钩子（正常路径）；这里同样
        # 走那条实现，保证"单独测这个算子"与"整图执行"是同一条代码路径。
        # bf16 / fp8 的载体都是 f32：``cast`` 只按 numpy dtype 看会"不需要转换"，
        # 真正落格点的是随后的 ``quantize``（bf16 打包成 uint16；fp8 在设备上
        # 重做 RNE 量化，见 5.325）。
        target = DType(attrs["dtype"])
        out = self.cast(args[0], target.np_dtype)
        if target is DType.BF16 or target in FP8_DTYPES:
            return self.quantize(out, target)
        return out

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
        return self._shared_view(out, x, node)

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
