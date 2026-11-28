"""归约：末维 / 非末维 / 多轴，以及 argmax / topk。"""

from __future__ import annotations

from ..lower.plan import BLOCK_DEFAULT, REDUCE_TILE_X, cdiv, plan_reduce_tiled
from .base import BackendCapabilityError
import numpy as np


class _ReduceMixin:
    """归约：末维 / 非末维 / 多轴，以及 argmax / topk。"""

    def _check_bool_reduce_len(self, op: str, length: int) -> None:
        """bool 归约的轴长上限（5.341）。

        ``all`` 的判定是"和恰好等于长度"：bool 元素是 0/1，求和必须精确，
        而 float 只能精确表示 2^24 以内的整数。超长时明确拒绝，而不是给出
        一个依赖累加顺序的答案。``any`` 只看和是否非零，本可以放宽，但与
        ``all`` 共用同一内核与同一份说明，保持口径一致。
        """
        if op in ("reduce_any", "reduce_all") and length > (1 << 24):
            raise BackendCapabilityError(
                f"{self._LABEL}的 {op} 一次最多归约 {1 << 24} 个元素，收到 {length}"
            )

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

    def _reduce_any_axis(self, node, args, axis: int, attrs, op: str, pred=None):
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

        ``pred`` 是 bool 归约（``any`` / ``all``）的判定方式（5.341）：此时
        ``op`` 仍是求和，判定由内核在累加之后做。超长轴拒绝——``all`` 要求
        "和恰好等于长度"，float 只能精确表示 2^24 以内的整数。
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
        self._check_bool_reduce_len(op, reduce_len)
        target = shape[:axis] + ((1,) if keepdims else ()) + shape[axis + 1 :]
        out = cp.empty(total, dtype=x.dtype)
        if out.size == 0:
            return out.reshape(target)
        kind = {
            "reduce_sum": "sum",
            "reduce_max": "max",
            "reduce_mean": "mean",
            # bool 归约按"数真值个数"求和，判定交给 pred（5.341）。
            "reduce_any": "sum",
            "reduce_all": "sum",
        }[op]
        if self._use_tiled_reduce(inner, total):
            tx, ty = plan_reduce_tiled(inner, reduce_len)
            kernel = self._kernel(
                (f"reduce_{kind}_axis_tiled", self._kind(x), pred),
                lambda: self.codegen.render(
                    self.codegen.reduce_axis_tiled_source(kind, pred), x.dtype
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
            (f"reduce_{kind}_axis", self._kind(x), pred),
            lambda: self.codegen.render(
                self.codegen.reduce_axis_source(kind, pred), x.dtype
            ),
            f"reduce_{kind}_axis",
        )
        block = self._reduce_block(op, total)
        kernel(
            (cdiv(total, block),),
            (block,),
            (x, out, np.int64(total), np.int64(reduce_len), np.int64(inner)),
        )
        return out.reshape(target)

    def _reduce_multi_axis(self, node, args, axes, attrs, op, source_fn, func, pred=None):
        """多轴归约：``[keep..., reduce...]`` 置换后合并成末维，复用单轴内核。

        归约与轴顺序无关，所以"先把要归约的轴挪到最后、乘成一个轴"是等价
        变换；代价是一次转置拷贝（输入本身已连续且轴本来就在末尾时为零）。
        空归约轴按 numpy 的语义返回单位元：sum 是 0、mean 是 NaN（numpy 会
        同时发 warning）、max 没有单位元因而拒绝；bool 归约里 any 是 False、
        all 是 True。
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
            if op in ("reduce_sum", "reduce_any"):
                return cp.zeros(rows, dtype=x.dtype).reshape(target)
            if op == "reduce_all":
                return cp.ones(rows, dtype=x.dtype).reshape(target)
            return cp.full(rows, np.nan, dtype=x.dtype).reshape(target)
        perm = keep + tuple(axes)
        if perm != tuple(range(x.ndim)):
            x = cp.ascontiguousarray(cp.transpose(x, perm))
        flat = x.reshape(rows, cols)
        out = self._reduce_last_dim(
            node, [flat], {"axis": -1, "keepdims": False}, op, source_fn, func, pred
        )
        return out.reshape(target)

    def _reduce_last_dim(self, node, args, attrs, op: str, source_fn, func: str, pred=None):
        """归约的公共骨架：末维走块内归约，其余轴走通用路径，多轴先合并。

        sum / max / mean 只在"用哪个内核"上不同；keepdims 的形状恢复、空输入
        这几件事完全一致，所以合成一处，改动只需要改一遍。

        多个轴先**置换到末尾再合并成一个轴**（``_reduce_multi_axis``）：归约
        结果与轴顺序无关，而合并后就能复用同一份单轴内核，不需要第二套内核。

        ``pred`` 只用于 bool 归约（``any`` / ``all``，5.341）：这几个算子的
        骨架与 sum 完全相同（数真值个数），差别只在累加后的判定。
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
            return self._reduce_multi_axis(
                node, args, norm, attrs, op, source_fn, func, pred
            )
        axis = norm[0]
        if axis != x.ndim - 1:
            return self._reduce_any_axis(node, args, axis, attrs, op, pred)
        rows, cols = self._last_dim_2d(x, op)
        self._check_bool_reduce_len(op, cols)
        keepdims = bool(attrs.get("keepdims", False))
        target = x.shape[:-1] + ((1,) if keepdims else ())
        out = cp.empty(rows, dtype=x.dtype)
        if x.size == 0:
            # 空轴（cols == 0 且还有输出行）：sum 的 0 已经在 empty 的语义之外，
            # 这里按 numpy 给 bool 归约的单位元——any 是 False、all 是 True。
            if rows and op == "reduce_any":
                return cp.zeros(rows, dtype=x.dtype).reshape(target)
            if rows and op == "reduce_all":
                return cp.ones(rows, dtype=x.dtype).reshape(target)
            return out.reshape(target)
        dtype_key = self._kind(x)
        if op == "reduce_sum" and cols <= 32:
            # 窄末维（d_state / n_scales / head_dim 与长度 1 的轴）：行数常在
            # 十万到百万级，"一 block 一行"等于把 1～32 个线程配一个 block，
            # 实测 29μs/次里几乎全是 block 调度（5.297）。改成 block 的 y 维
            # 打包多行：blockDim.x 仍是 plan_reduce 给出的折半树宽度，结果
            # 与旧内核逐位一致，只是 block 数少 blockDim.y 倍。
            nx = self._reduce_block(op, cols)
            ny = max(1, min(256 // max(nx, 1), 32))
            kernel = self._kernel(
                (op, dtype_key, "rows"),
                lambda: self.codegen.render(
                    self.codegen.reduce_sum_rows_source(), x.dtype
                ),
                "reduce_sum_rows",
            )
            grid = ((rows + ny - 1) // ny,)
            kernel(grid, (nx, ny), (x, out, np.int64(rows), np.int64(cols)))
            return out.reshape(target)
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

    def _op_reduce_any(self, node, args, attrs):
        """bool 归约：任一为真即真（5.341）。

        与 sum 共用内核骨架（数真值个数），判定放在块内归约之后；输出是
        bool，写回由 ``af_st_q`` 归一成 0/1。
        """
        return self._reduce_last_dim(
            node,
            args,
            attrs,
            "reduce_any",
            self.codegen.reduce_any_source,
            "reduce_any_last",
            "any",
        )

    def _op_reduce_all(self, node, args, attrs):
        """bool 归约：全为真才真（5.341）。

        判定是"和恰好等于长度"，因此轴长超过 2^24 时拒绝（float 的整数
        精确区间），见 ``_check_bool_reduce_len``。
        """
        return self._reduce_last_dim(
            node,
            args,
            attrs,
            "reduce_all",
            self.codegen.reduce_all_source,
            "reduce_all_last",
            "all",
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

    def _op_argmin(self, node, args, attrs):
        """任意单轴的 argmin：输出 i32，与 ``argmax`` 逐条同构，只换比较方向。

        非末轴先置换到最后复用末维内核；并列取**第一次出现**的位置（numpy /
        PyTorch 的约定）。空轴同样拒绝，而不是返回没有定义的 ``-1``。
        """
        x = args[0]
        axis = attrs.get("axis", -1)
        if isinstance(axis, (tuple, list)):
            if len(axis) != 1:
                raise BackendCapabilityError(
                    f"{self._LABEL}的 argmin 一次只支持一个轴，收到 {axis}"
                )
            axis = axis[0]
        if x.ndim == 0:
            raise BackendCapabilityError(f"{self._LABEL}的 argmin 需要至少一维")
        axis = int(axis) % x.ndim
        keepdims = bool(attrs.get("keepdims", False))
        if axis == x.ndim - 1:
            return self._argmin_last(x, keepdims)
        shape = tuple(int(d) for d in x.shape)
        perm = tuple(i for i in range(x.ndim) if i != axis) + (axis,)
        flat = self.cp.ascontiguousarray(self.cp.transpose(self._contig(x), perm))
        out = self._argmin_last(flat, False)
        target = (
            tuple(1 if i == axis else shape[i] for i in range(x.ndim))
            if keepdims
            else tuple(shape[i] for i in range(x.ndim) if i != axis)
        )
        return out.reshape(target)

    def _argmin_last(self, x, keepdims: bool):
        cp = self.cp
        rows, cols = self._last_dim_2d(x, "argmin")
        if cols == 0:
            raise BackendCapabilityError(
                f"{self._LABEL}的 argmin 不能对空轴归约（形状 {x.shape}）"
            )
        target = x.shape[:-1] + ((1,) if keepdims else ())
        out = cp.empty(rows, dtype=cp.int32)
        if out.size == 0:
            return out.reshape(target)
        dtype_key = self._kind(x)
        kernel = self._kernel(
            ("argmin", dtype_key),
            lambda: self.codegen.render(self.codegen.argmin_source(), x.dtype),
            "argmin_last",
        )
        block = self._reduce_block("argmin", cols)
        kernel((rows,), (block,), (x, out, np.int64(rows), np.int64(cols)))
        return out.reshape(target)

    # -- topk ---------------------------------------------------------
    #
    # 三个算子共用一份内核（``topk_source``）：一个 block 处理一行，``k`` 轮
    # "块内最大值 + atomicMin 取小下标"。与 argmax 一样，非末轴先换到末维再
    # 复用同一个内核，输出最后换回原轴序。

    def _topk_axis_k(self, x, attrs, op: str):
        """解析并防御性校验 ``axis`` / ``k``（IR 已验证过，这里守住后端入口）。"""
        if x.ndim == 0:
            raise BackendCapabilityError(f"{self._LABEL}的 {op} 需要至少一维")
        axis = attrs.get("axis", -1)
        if isinstance(axis, (tuple, list)):
            if len(axis) != 1:
                raise BackendCapabilityError(
                    f"{self._LABEL}的 {op} 一次只支持一个轴，收到 {axis}"
                )
            axis = axis[0]
        axis = int(axis) % x.ndim
        k = attrs.get("k")
        if isinstance(k, bool) or not isinstance(k, int):
            raise BackendCapabilityError(
                f"{self._LABEL}的 {op} 的 k 必须是整数，收到 {k!r}"
            )
        if k < 1:
            raise BackendCapabilityError(f"{self._LABEL}的 {op} 的 k 必须为正，收到 {k}")
        cols = int(x.shape[axis])
        if k > cols:
            raise BackendCapabilityError(
                f"{self._LABEL}的 {op} 的 k={k} 超过轴 {axis} 的长度 {cols}"
            )
        return axis, k

    def _topk_flatten(self, x, axis: int):
        """把 ``axis`` 换到末维并转连续；返回 ``(flat, perm, perm_shape)``。

        ``perm`` 为 ``None`` 表示本来就是末维（不拷贝）。
        """
        cp = self.cp
        shape = tuple(int(d) for d in x.shape)
        if axis == x.ndim - 1:
            return self._contig(x), None, shape
        perm = tuple(i for i in range(x.ndim) if i != axis) + (axis,)
        flat = cp.ascontiguousarray(cp.transpose(self._contig(x), perm))
        return flat, perm, tuple(shape[i] for i in perm)

    def _topk_launch(self, op: str, flat, out, rows: int, cols: int, k: int, func: str):
        dtype_key = self._kind(flat)
        kernel = self._kernel(
            (op, dtype_key),
            lambda: self.codegen.render(self.codegen.topk_source(), flat.dtype),
            func,
        )
        block = self._reduce_block(op, cols)
        # 静态 smem 是 4 KiB（block_reduce_max 的 scratch），动态部分是 k 个 int。
        sel_bytes = 4 * k
        if 4096 + sel_bytes > 48 * 1024:
            raise BackendCapabilityError(
                f"{self._LABEL}的 {op} 需要 {sel_bytes} 字节动态共享内存，"
                f"超过每块 48 KiB 的上限（k={k} 过大）"
            )
        kernel(
            (rows,),
            (block,),
            (flat, out, np.int64(rows), np.int64(cols), np.int32(k)),
            shared_mem=sel_bytes,
        )

    def _topk_result(self, out_perm, perm, target):
        """把置换布局（末维是 k 或 cols）的输出换回原轴序。"""
        if perm is None:
            return out_perm.reshape(target)
        inv = tuple(sorted(range(len(perm)), key=lambda i: perm[i]))
        return self.cp.ascontiguousarray(out_perm.transpose(inv))

    def _op_topk_values(self, node, args, attrs):
        cp = self.cp
        x = args[0]
        axis, k = self._topk_axis_k(x, attrs, "topk_values")
        flat, perm, perm_shape = self._topk_flatten(x, axis)
        rows, cols = self._last_dim_2d(flat, "topk_values")
        out = cp.empty(perm_shape[:-1] + (k,), dtype=flat.dtype)
        if rows:
            self._topk_launch(
                "topk_values", flat, out, rows, cols, k, "topk_values"
            )
        target = tuple(k if i == axis else int(d) for i, d in enumerate(x.shape))
        return self._topk_result(out, perm, target)

    def _op_topk_indices(self, node, args, attrs):
        cp = self.cp
        x = args[0]
        axis, k = self._topk_axis_k(x, attrs, "topk_indices")
        flat, perm, perm_shape = self._topk_flatten(x, axis)
        rows, cols = self._last_dim_2d(flat, "topk_indices")
        out = cp.empty(perm_shape[:-1] + (k,), dtype=cp.int32)
        if rows:
            self._topk_launch(
                "topk_indices", flat, out, rows, cols, k, "topk_indices"
            )
        target = tuple(k if i == axis else int(d) for i, d in enumerate(x.shape))
        return self._topk_result(out, perm, target)

    def _op_topk_mask(self, node, args, attrs):
        cp = self.cp
        x = args[0]
        axis, k = self._topk_axis_k(x, attrs, "topk_mask")
        flat, perm, perm_shape = self._topk_flatten(x, axis)
        rows, cols = self._last_dim_2d(flat, "topk_mask")
        out = cp.zeros(perm_shape, dtype=flat.dtype)
        if rows:
            self._topk_launch(
                "topk_mask", flat, out, rows, cols, k, "topk_mask_select"
            )
        return self._topk_result(out, perm, tuple(int(d) for d in x.shape))
