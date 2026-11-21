"""归约：末维 / 非末维 / 多轴，以及 argmax。"""

from __future__ import annotations

from ..lower.plan import BLOCK_DEFAULT, REDUCE_TILE_X, cdiv, plan_reduce_tiled
from .base import BackendCapabilityError
import numpy as np


class _ReduceMixin:
    """归约：末维 / 非末维 / 多轴，以及 argmax。"""

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
        for dim in shape[axis + 1 :]:
            inner *= dim
        total = outer * inner
        target = shape[:axis] + ((1,) if keepdims else ()) + shape[axis + 1 :]
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
                lambda: self.codegen.render(self.codegen.reduce_axis_tiled_source(kind), x.dtype),
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
            node,
            args,
            attrs,
            "reduce_sum",
            self.codegen.reduce_sum_source,
            "reduce_sum_last",
        )

    def _op_reduce_max(self, node, args, attrs):
        return self._reduce_last_dim(
            node,
            args,
            attrs,
            "reduce_max",
            self.codegen.reduce_max_source,
            "reduce_max_last",
        )

    def _op_reduce_mean(self, node, args, attrs):
        return self._reduce_last_dim(
            node,
            args,
            attrs,
            "reduce_mean",
            self.codegen.reduce_mean_source,
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
        flat = self.cp.ascontiguousarray(self.cp.transpose(self._contig(x), perm))
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
            raise BackendCapabilityError(f"{self._LABEL}的 argmax 不能对空轴归约（形状 {x.shape}）")
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
