"""线性递推与前缀和：整段扫描、分块扫描、cumsum、tril。"""

from __future__ import annotations

from ..lower.plan import scan_chunk
from .base import BackendCapabilityError
import numpy as np


class _ScanMixin:
    """线性递推与前缀和：整段扫描、分块扫描、cumsum、tril。"""

    # ------------------------------------------------------------------
    # 序列扫描（SSM 的线性递归）
    # ------------------------------------------------------------------

    def _op_scan_linear(self, node, args, attrs):
        return self._scan(args[0], args[1], args[2], backward=False)

    def _op_scan_linear_backward(self, node, args, attrs):
        return self._scan(args[0], args[1], None, backward=True)

    # ------------------------------------------------------------------
    # 前缀和 / 三角掩码（分块 SSM 的块内权重）
    # ------------------------------------------------------------------

    def _op_cumsum(self, node, args, attrs):
        """沿单个轴的前缀和：一个线程一条 ``(outer, inner)`` 链。

        张量按 ``(outer, steps, inner)`` 切分，内核里的 ``work`` 就是并行链数
        ``outer * inner``，grid 因此按它铺——与 ``scan_linear`` 是同一套铺法，
        区别只在没有 ``a_t`` 乘法。axis 为负时先归一化再折维度；轴越界在设备
        端也拦一次，避免快路径绕过 verifier 时把错误轴当成合法轴。
        """
        cp = self.cp
        x = self._contig(args[0])
        axis = attrs.get("axis", -1)
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise BackendCapabilityError(
                f"{self._LABEL}的 cumsum 的 axis 必须是整数，收到 {axis!r}"
            )
        axis = axis + x.ndim if axis < 0 else axis
        if not 0 <= axis < x.ndim:
            raise BackendCapabilityError(
                f"{self._LABEL}的 cumsum 轴越界: {attrs.get('axis')!r}，秩为 {x.ndim}"
            )
        shape = tuple(int(d) for d in x.shape)
        steps = shape[axis]
        outer = int(np.prod(shape[:axis], dtype=np.int64))
        inner = int(np.prod(shape[axis + 1 :], dtype=np.int64))
        out = cp.empty_like(x)
        work = outer * inner
        if work == 0 or steps == 0:
            return out
        kernel = self._kernel(
            ("cumsum", self._kind(x)),
            lambda: self.codegen.render(self.codegen.cumsum_source(), x.dtype),
            "cumsum",
        )
        block, grid = self._grid("cumsum", work)
        kernel(
            (grid,),
            (block,),
            (x, out, np.int64(work), np.int64(steps), np.int64(inner)),
        )
        return out

    def _op_cumprod(self, node, args, attrs):
        """沿单个轴的前缀积：与 ``cumsum`` 同一套 ``(outer, steps, inner)`` 铺法。

        累乘换掉累加，内核初值 1；其它（轴归一化、链数、边界）逐条同 ``cumsum``。
        """
        cp = self.cp
        x = self._contig(args[0])
        axis = self._axis_of(attrs, x.ndim, "cumprod")
        shape = tuple(int(d) for d in x.shape)
        steps = shape[axis]
        outer = int(np.prod(shape[:axis], dtype=np.int64))
        inner = int(np.prod(shape[axis + 1 :], dtype=np.int64))
        out = cp.empty_like(x)
        work = outer * inner
        if work == 0 or steps == 0:
            return out
        kernel = self._kernel(
            ("cumprod", self._kind(x)),
            lambda: self.codegen.render(self.codegen.cumprod_source(), x.dtype),
            "cumprod",
        )
        block, grid = self._grid("cumprod", work)
        kernel(
            (grid,),
            (block,),
            (x, out, np.int64(work), np.int64(steps), np.int64(inner)),
        )
        return out

    def _axis_of(self, attrs, ndim: int, op: str) -> int:
        """``axis`` 归一化 + 越界检查；``flip`` / ``roll`` / ``cumprod`` 共用。"""
        axis = attrs.get("axis", -1)
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise BackendCapabilityError(
                f"{self._LABEL}的 {op} 的 axis 必须是整数，收到 {axis!r}"
            )
        axis = axis + ndim if axis < 0 else axis
        if not 0 <= axis < ndim:
            raise BackendCapabilityError(
                f"{self._LABEL}的 {op} 轴越界: {attrs.get('axis')!r}，秩为 {ndim}"
            )
        return axis

    def _reorder(self, source, kernel_name: str, x, axis: int, shift):
        """沿单轴的元素搬运（``flip`` / ``roll`` 共用）：一个线程搬一个元素。

        与 ``tril`` / 逐元素族同一套扁平铺法：``work`` 是整个张量的元素数，
        内核把线性下标拆成 ``(o, j, i)`` 后只改 ``j``。``shift`` 在这里归一化到
        ``[0, steps)``——C 的 ``%`` 对负数向零截断，交给内核会得到负下标。
        """
        cp = self.cp
        shape = tuple(int(d) for d in x.shape)
        steps = shape[axis]
        inner = int(np.prod(shape[axis + 1 :], dtype=np.int64))
        work = int(np.prod(shape, dtype=np.int64))
        out = cp.empty_like(x)
        if work == 0 or steps == 0:
            return out
        extra = () if shift is None else (np.int64(int(shift) % steps),)
        kernel = self._kernel(
            (kernel_name, self._kind(x)),
            lambda: self.codegen.render(source(), x.dtype),
            kernel_name,
        )
        block, grid = self._grid(kernel_name, work)
        kernel(
            (grid,),
            (block,),
            (x, out, np.int64(work), np.int64(steps), np.int64(inner)) + extra,
        )
        return out

    def _op_flip(self, node, args, attrs):
        """沿单个轴反转：读写下标互为镜像，不需要额外缓冲。"""
        x = self._contig(args[0])
        axis = self._axis_of(attrs, x.ndim, "flip")
        return self._reorder(self.codegen.flip_source, "flip_axis", x, axis, None)

    def _op_roll(self, node, args, attrs):
        """沿单个轴循环右移；``shift`` 取模后一定是非负的。"""
        x = self._contig(args[0])
        axis = self._axis_of(attrs, x.ndim, "roll")
        shift = attrs.get("shift", 0)
        if isinstance(shift, bool) or not isinstance(shift, int):
            raise BackendCapabilityError(
                f"{self._LABEL}的 roll 的 shift 必须是整数，收到 {shift!r}"
            )
        return self._reorder(self.codegen.roll_source, "roll_axis", x, axis, shift)

    def _op_repeat_interleave(self, node, args, attrs):
        """沿单轴逐元素重复 ``repeats`` 次：该轴长度变成 ``dim * repeats``。

        铺法与 ``flip`` / ``roll`` 相同（一个线程搬一个元素），差别只在下标：
        ``steps`` / ``inner`` 取自**输入**，而 ``work`` 与线性下标的拆分按
        **输出**形状（内核里算 ``steps * repeats``）。
        """
        cp = self.cp
        x = self._contig(args[0])
        axis = self._axis_of(attrs, x.ndim, "repeat_interleave")
        repeats = attrs.get("repeats")
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
            raise BackendCapabilityError(
                f"{self._LABEL}的 repeat_interleave 的 repeats 必须是正整数，"
                f"收到 {repeats!r}"
            )
        shape = tuple(int(d) for d in x.shape)
        steps = shape[axis]
        inner = int(np.prod(shape[axis + 1 :], dtype=np.int64))
        out_shape = shape[:axis] + (steps * repeats,) + shape[axis + 1 :]
        out = cp.empty(out_shape, dtype=x.dtype)
        work = int(np.prod(out_shape, dtype=np.int64))
        if work == 0 or steps == 0:
            return out
        kernel = self._kernel(
            ("repeat_interleave", self._kind(x)),
            lambda: self.codegen.render(
                self.codegen.repeat_interleave_source(), x.dtype
            ),
            "repeat_interleave",
        )
        block, grid = self._grid("repeat_interleave", work)
        kernel(
            (grid,),
            (block,),
            (
                x,
                out,
                np.int64(work),
                np.int64(steps),
                np.int64(inner),
                np.int64(repeats),
            ),
        )
        return out


    def _op_tril(self, node, args, attrs):
        """最后两维保留下三角、其余置零：一个线程一个元素。

        C 布局下线性下标满足 ``c = idx % cols``、``r = (idx / cols) % rows``，
        前导维被 ``% rows`` 自然吸收，所以不需要额外传 batch；条件外写 0
        而不是跳过，输出才不会带着未初始化值。
        """
        cp = self.cp
        x = self._contig(args[0])
        if x.ndim < 2:
            raise BackendCapabilityError(
                f"{self._LABEL}的 tril 至少需要秩 2，收到 {tuple(x.shape)}"
            )
        diagonal = attrs.get("diagonal", 0)
        if isinstance(diagonal, bool) or not isinstance(diagonal, int):
            raise BackendCapabilityError(
                f"{self._LABEL}的 tril 的 diagonal 必须是整数，收到 {diagonal!r}"
            )
        rows, cols = int(x.shape[-2]), int(x.shape[-1])
        out = cp.empty_like(x)
        n = int(x.size)
        if n == 0 or rows == 0 or cols == 0:
            return out
        kernel = self._kernel(
            ("tril", self._kind(x)),
            lambda: self.codegen.render(self.codegen.tril_source(), x.dtype),
            "tril",
        )
        block, grid = self._grid("tril", n)
        kernel(
            (grid,),
            (block,),
            (
                x,
                out,
                np.int64(n),
                np.int64(rows),
                np.int64(cols),
                np.int64(diagonal),
            ),
        )
        return out

    def _scan(self, a, b, h0, backward: bool):
        """线性递归（及其伴随）的 CUDA 承接。

        分工是**元素并行、时间串行**：一个线程独占一个尾元素（把
        ``a.shape[1:]`` 压平后的一格），沿 T 顺序推进。递推本身没有结合律
        之外的并行空间可用，而这里的并行度来自尾维（batch × 通道 × 状态）。

        串行版在"长 T + 小尾维"时是纯延迟：设备填不满，瓶颈就是那一串
        逐时间步的依赖。``scan_chunk`` 按 ``(steps, elems)`` 决定是否换成
        两趟分块（块内复合 + 块前缀/读出），把并行度放大到
        ``elems * chunks``；短 T 或尾维已经喂饱带宽时仍走串行，见
        ``lower/plan.py`` 的 ``SCAN_CHUNK_*``。

        为什么可以压平尾维：a/b 的布局是 (T, 尾...)，展平后第 t 步的偏移
        就是 ``t * elems + i``，与尾维的具体阶数无关，因此 2 维、3 维、
        4 维输入共用同一个 kernel。
        """
        cp = self.cp
        a = self._contig(a)
        dy_or_b = self._contig(b)
        if tuple(dy_or_b.shape) != tuple(a.shape):
            raise BackendCapabilityError(
                f"{self._LABEL}的 scan 要求 a 与第二输入同形，"
                f"收到 {tuple(a.shape)} 与 {tuple(dy_or_b.shape)}"
            )
        if a.ndim < 2:
            raise BackendCapabilityError(
                f"{self._LABEL}的 scan 至少要求 (T, 尾...) 两个维度，收到 {tuple(a.shape)}"
            )
        out = cp.empty_like(a)
        steps = int(a.shape[0])
        if steps == 0 or int(a.size) == 0:
            return out
        elems = int(a.size) // steps

        if not backward:
            if tuple(h0.shape) != tuple(a.shape[1:]):
                raise BackendCapabilityError(
                    f"scan_linear 的 h0 形状应为 {tuple(a.shape[1:])}，收到 {tuple(h0.shape)}"
                )
            h0 = cp.ascontiguousarray(h0)

        chunk = scan_chunk(self.chunked_scan, steps, elems)
        if chunk:
            return self._scan_chunked(
                a,
                dy_or_b,
                h0,
                out,
                chunk=chunk,
                backward=backward,
                steps=steps,
                elems=elems,
            )

        dtype_key = self._kind(a)
        kernel = self._kernel(
            ("scan_linear_backward", dtype_key) if backward else ("scan_linear", dtype_key),
            lambda: self.codegen.render(
                self.codegen.scan_linear_backward_source()
                if backward
                else self.codegen.scan_linear_source(),
                a.dtype,
            ),
            "scan_linear_backward" if backward else "scan_linear",
        )
        block, grid = self._grid("scan_linear_backward" if backward else "scan_linear", elems)
        args = [a, dy_or_b] if backward else [a, dy_or_b, h0]
        kernel(
            (grid,),
            (block,),
            (*args, out, np.int64(steps), np.int64(elems)),
        )
        return out

    def _scan_chunk_sums(self, a, b, steps, elems, chunk, *, backward):
        """分块扫描第一趟：每块算复合 ``(prod, sum)``。

        scratch 固定 float32（不随 dtype 变 f16）：块内递推本来就在 float
        里做，把聚合值降成 f16 会多两次舍入，误差比串行路径大一个数量级。
        """
        cp = self.cp
        chunks = (steps + chunk - 1) // chunk
        total = chunks * elems
        prod = cp.empty(total, dtype=cp.float32)
        acc = cp.empty(total, dtype=cp.float32)
        name = "scan_linear_chunk_sum_backward" if backward else "scan_linear_chunk_sum"
        kernel = self._kernel(
            (name, self._kind(a)),
            lambda: self.codegen.render(
                self.codegen.scan_linear_chunk_sum_backward_source()
                if backward
                else self.codegen.scan_linear_chunk_sum_source(),
                a.dtype,
            ),
            name,
        )
        block, grid = self._grid(name, total)
        kernel(
            (grid,),
            (block,),
            (
                a,
                b,
                prod,
                acc,
                np.int64(steps),
                np.int64(elems),
                np.int64(chunk),
                np.int64(total),
            ),
        )
        return prod, acc, chunks

    def _scan_chunked(self, a, b, h0, out, *, chunk, backward, steps, elems):
        """分块扫描第二趟：块前缀 + 块内顺序读出。

        并行度从 ``elems`` 变成 ``elems * chunks``；块内递推与串行版逐字
        相同，只有跨块的入口状态走了"先乘再加"的复合，因此结果**不逐位**
        等于串行版。块长由 :func:`scan_chunk` 解析（自动或显式），传进来而
        不是在这里重算：第一趟的 scratch 形状必须与第二趟的块长一致。
        """
        prod, acc, chunks = self._scan_chunk_sums(a, b, steps, elems, chunk, backward=backward)
        total = chunks * elems
        name = "scan_linear_chunk_backward" if backward else "scan_linear_chunk"
        kernel = self._kernel(
            (name, self._kind(a)),
            lambda: self.codegen.render(
                self.codegen.scan_linear_chunk_backward_source()
                if backward
                else self.codegen.scan_linear_chunk_source(),
                a.dtype,
            ),
            name,
        )
        block, grid = self._grid(name, total)
        args = [a, b, prod, acc, out] if backward else [a, b, h0, prod, acc, out]
        kernel(
            (grid,),
            (block,),
            (
                *args,
                np.int64(steps),
                np.int64(elems),
                np.int64(chunk),
                np.int64(total),
            ),
        )
        return out

    def scan_last_state(self, a, b, h0):
        """只返回 ``(tail...)`` 的末状态，不物化整段 ``out``。

        语义与 ``scan_linear`` 的最后一步相同，省掉 ``T-1`` 次全局写。
        与图无关的入口：流式推理每步只要末状态，没必要先把 ``(T, tail)``
        写出来再切片。
        """
        cp = self.cp
        a = self._contig(a)
        b = self._contig(b)
        if tuple(a.shape) != tuple(b.shape):
            raise BackendCapabilityError(
                f"{self._LABEL}的 scan 要求 a 与第二输入同形，"
                f"收到 {tuple(a.shape)} 与 {tuple(b.shape)}"
            )
        if a.ndim < 2:
            raise BackendCapabilityError(
                f"{self._LABEL}的 scan 至少要求 (T, 尾...) 两个维度，收到 {tuple(a.shape)}"
            )
        tail = tuple(a.shape[1:])
        if tuple(h0.shape) != tail:
            raise BackendCapabilityError(
                f"scan_last_state 的 h0 形状应为 {tail}，收到 {tuple(h0.shape)}"
            )
        h0 = cp.ascontiguousarray(h0)
        out = cp.empty(tail, dtype=a.dtype)
        steps = int(a.shape[0])
        elems = int(a.size) // steps if steps else 0
        if steps == 0 or elems == 0:
            return out

        chunk = scan_chunk(self.chunked_scan, steps, elems, last=True)
        if chunk:
            prod, acc, chunks = self._scan_chunk_sums(a, b, steps, elems, chunk, backward=False)
            name = "scan_linear_chunk_last"
            kernel = self._kernel(
                (name, self._kind(a)),
                lambda: self.codegen.render(self.codegen.scan_linear_chunk_last_source(), a.dtype),
                name,
            )
            block, grid = self._grid(name, elems)
            kernel(
                (grid,),
                (block,),
                (prod, acc, h0, out, np.int64(chunks), np.int64(elems)),
            )
            return out

        name = "scan_linear_last"
        kernel = self._kernel(
            (name, self._kind(a)),
            lambda: self.codegen.render(self.codegen.scan_linear_last_source(), a.dtype),
            name,
        )
        block, grid = self._grid(name, elems)
        kernel(
            (grid,),
            (block,),
            (a, b, h0, out, np.int64(steps), np.int64(elems)),
        )
        return out
