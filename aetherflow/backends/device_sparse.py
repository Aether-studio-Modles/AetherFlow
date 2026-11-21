"""事件驱动稀疏与脉冲：CSR 起点表、稀疏线性及其梯度。"""

from __future__ import annotations

from ..lower.plan import plan_csr_scan, plan_csr_scan_enabled
from .base import BackendCapabilityError
import numpy as np


class _SparseMixin:
    """事件驱动稀疏与脉冲：CSR 起点表、稀疏线性及其梯度。"""

    def _op_spike_offsets(self, node, args, attrs):
        """统计每个时间步的非零数，写出 CSR 起点表 ``[T+1]``。

        与 ``spike_csr`` 拆成两个节点，而不是打包进一个 i32 张量：IR 没有
        多输出节点，打包后下游必须 ``slice`` 切整数缓冲，而 GPU 后端的
        ``slice`` 明确不支持 int32（不做隐式降级）。两趟内核各自单输出，
        下游 ``sparse_linear`` 直接拿到起点与列号，无需任何整数切片。
        """
        cp = self.cp
        (spikes,) = args
        spikes = self._contig(spikes)
        steps, width = int(spikes.shape[0]), int(spikes.shape[1])
        offsets = cp.zeros((steps + 1,), dtype=cp.int32)
        kernel = self._kernel(
            ("spike_offsets", self._kind(spikes)),
            lambda: self.codegen.render(self.codegen.spike_offsets_source(), spikes.dtype),
            "spike_offsets",
        )
        # 单 block：阶段 1 并行数行，阶段 2 由 0 号线程做 T 步前缀和。
        # 不能用 _grid：那会铺多个 block，块间无法 __syncthreads。
        block = max(1, min(256, steps))
        kernel(
            (1,),
            (block,),
            (spikes, offsets, np.int64(width), np.int64(steps)),
        )
        return offsets

    def _use_scan_csr(self, steps: int) -> bool:
        """``spike_csr`` 是否走块内扫描（5.176）。

        判据是 ``plan_csr_scan_enabled``：只有行数少到串行铺法填不满机器时才
        换铺法——行数多时串行版靠每线程的长循环就能藏住访存延迟，并行版反而
        多付块内同步。单独抽成方法只为让基准能强制走某一条做对照
        （``benchmarks/spike_csr_scan.py`` 在实例上盖掉它），与 5.174 的
        ``_use_tiled_reduce`` 是同一个理由。
        """
        return plan_csr_scan_enabled(steps, self.sm_count)

    def _op_spike_csr(self, node, args, attrs):
        """按起点表把每行的非零列号写进 ``indices[capacity]``。

        容量固定是静态图的前提；溢出在设备端记 ``err``，主机侧抛错——
        与 ``sparse_linear`` 的越界检查同一条通路，捕获期也成立。
        """
        cp = self.cp
        spikes, offsets = args
        spikes = self._contig(spikes)
        offsets = self._contig(offsets)
        steps, width = int(spikes.shape[0]), int(spikes.shape[1])
        capacity = int(attrs["capacity"])
        indices = cp.zeros((capacity,), dtype=cp.int32)
        err = cp.zeros(1, dtype=cp.int32)
        kind = self._kind(spikes)

        scan = self._use_scan_csr(steps)
        kernel_name = "spike_csr" if scan else "spike_csr_serial"
        kernel = self._kernel(
            (kernel_name, kind),
            lambda: self.codegen.render(
                (
                    self.codegen.spike_csr_source()
                    if scan
                    else self.codegen.spike_csr_serial_source()
                ),
                spikes.dtype,
            ),
            kernel_name,
        )
        # 一个 block 负责一行（5.176）：块内要用 __syncthreads 做前缀和，
        # 所以网格不能像一维逐元素那样被 _grid 截断到 max_blocks——截断会把
        # 后面的行整行漏掉。块大小只跟列数走；串行铺法是一个线程一行，按行数铺。
        if scan:
            block = plan_csr_scan(width)
            grid = max(1, steps)
        else:
            block, grid = self._grid("spike_csr_serial", steps)
        kernel(
            (grid,),
            (block,),
            (
                spikes,
                offsets,
                indices,
                np.int64(capacity),
                np.int64(width),
                np.int64(steps),
                err,
            ),
        )
        self._check_ids(err, "spike_csr", f"[0, {capacity}]（事件数超过容量）")
        return indices

    def _pick_cusparse(self, values, indices, offsets) -> bool:
        """这一次 ``sparse_linear`` 走不走厂商库；默认永不启用。

        CUDA 覆盖它（5.203）。放在基类而不是写成 ``hasattr`` 探测：HIP 的
        ``SPARSE_IMPLS`` 里没有 ``"cusparse"``，构造期就不会接受这个值，
        这里只需回答"没有库路径"。
        """
        return False

    def _sparse_linear_cusparse(
        self, weights, indices, values, offsets, *, steps: int, rows: int, cols: int
    ):
        """厂商库的 SpMM；默认没有，返回 ``None``（调用方回落自研核）。"""
        return None

    def _op_sparse_linear(self, node, args, attrs):
        """事件驱动的稀疏线性：CSR 事件 × 稠密权重。

        ``indices`` / ``values`` 的长度是事件数，可以是符号维——它们只决定
        内核读多少事件，不影响输出形状。``offsets`` 的长度决定输出步数，
        ``infer`` 已要求它是静态的。

        计算量是 ``S × N`` 个输出元素各自扫自己那一段事件；稀疏脉冲矩阵
        的 ``T × K`` 稠密形态从不物化，读的事件总量等于真正发放的次数。
        """
        cp = self.cp
        weights, indices, values, offsets = args
        weights = self._contig(weights)
        values = self._contig(values)
        indices = cp.ascontiguousarray(indices).reshape(-1)
        offsets = cp.ascontiguousarray(offsets).reshape(-1)
        if int(indices.size) != int(values.size):
            raise BackendCapabilityError(
                f"{self._LABEL}的 sparse_linear 要求 indices 与 values 等长，"
                f"收到 {int(indices.size)} 与 {int(values.size)}"
            )
        steps = int(offsets.size) - 1
        rows, cols = int(weights.shape[0]), int(weights.shape[1])
        total = steps * cols
        if total == 0:
            return cp.zeros((steps, cols), dtype=values.dtype)
        # 5.203：库路径只在显式要求（或 auto 且库可用）时走。默认 custom
        # 保留越界检查；库路径不做下标检查，这是它不设默认值的直接理由。
        if self._pick_cusparse(values, indices, offsets):
            fast = self._sparse_linear_cusparse(
                weights,
                indices,
                values,
                offsets,
                steps=steps,
                rows=rows,
                cols=cols,
            )
            if fast is not None:
                return fast
        out = cp.zeros((steps, cols), dtype=values.dtype)
        kernel = self._kernel(
            ("sparse_linear", self._kind(values)),
            lambda: self.codegen.render(self.codegen.sparse_linear_source(), values.dtype),
            "sparse_linear",
        )
        err = cp.zeros(1, dtype=cp.int32)
        block, grid = self._grid("sparse_linear", total)
        kernel(
            (grid,),
            (block,),
            (
                weights,
                indices,
                values,
                offsets,
                out,
                np.int64(cols),
                np.int64(rows),
                np.int64(int(values.size)),
                np.int64(total),
                err,
            ),
        )
        self._check_ids(err, "sparse_linear", f"[0, {rows})")
        return out

    def _op_sparse_linear_weight_grad(self, node, args, attrs):
        """``sparse_linear`` 对权重的梯度：把事件贡献散射回各自的行。

        输出先清零再原子累加——同一行在一个 batch 内被命中多次是常态。
        ``rows`` 只能从属性读：反向图里没有 ``weights`` 这个输入。
        """
        cp = self.cp
        dout, indices, values, offsets = args
        dout = self._contig(dout)
        values = self._contig(values)
        indices = cp.ascontiguousarray(indices).reshape(-1)
        offsets = cp.ascontiguousarray(offsets).reshape(-1)
        if int(indices.size) != int(values.size):
            raise BackendCapabilityError(
                f"{self._LABEL}的 sparse_linear_weight_grad 要求 indices 与 "
                f"values 等长，收到 {int(indices.size)} 与 {int(values.size)}"
            )
        rows = int(attrs["rows"])
        steps, cols = int(dout.shape[0]), int(dout.shape[1])
        dweights = cp.zeros((rows, cols), dtype=dout.dtype)
        total = steps * cols
        if total == 0:
            return dweights
        kernel = self._kernel(
            ("sparse_linear_weight_grad", self._kind(dout)),
            lambda: self.codegen.render(
                self.codegen.sparse_linear_weight_grad_source(), dout.dtype
            ),
            "sparse_linear_weight_grad",
        )
        err = cp.zeros(1, dtype=cp.int32)
        block, grid = self._grid("sparse_linear_weight_grad", total)
        kernel(
            (grid,),
            (block,),
            (
                dout,
                indices,
                values,
                offsets,
                dweights,
                np.int64(cols),
                np.int64(rows),
                np.int64(int(values.size)),
                np.int64(total),
                err,
            ),
        )
        self._check_ids(err, "sparse_linear_weight_grad", f"[0, {rows})")
        return dweights

    def _op_sparse_linear_values_grad(self, node, args, attrs):
        """``sparse_linear`` 对事件幅度的梯度：每个事件与所属步的 ``dout`` 做点积。

        按步并行——一个线程包办该步的全部事件，直接写 ``dvalues[e]``。事件之间
        没有重复下标，所以不像权重梯度那样需要"先清零再原子累加"。代价是
        ``steps`` 少时并行度低，但省掉了在设备上把事件号映射回步号的二分查找。
        """
        cp = self.cp
        dout, weights, indices, offsets = args
        dout = self._contig(dout)
        weights = self._contig(weights)
        indices = cp.ascontiguousarray(indices).reshape(-1)
        offsets = cp.ascontiguousarray(offsets).reshape(-1)
        steps = int(offsets.size) - 1
        rows, cols = int(weights.shape[0]), int(weights.shape[1])
        nnz = int(indices.size)
        dvalues = cp.zeros((nnz,), dtype=dout.dtype)
        if steps == 0:
            return dvalues
        kernel = self._kernel(
            ("sparse_linear_values_grad", self._kind(dout)),
            lambda: self.codegen.render(
                self.codegen.sparse_linear_values_grad_source(), dout.dtype
            ),
            "sparse_linear_values_grad",
        )
        err = cp.zeros(1, dtype=cp.int32)
        block, grid = self._grid("sparse_linear_values_grad", steps)
        kernel(
            (grid,),
            (block,),
            (
                dout,
                weights,
                indices,
                offsets,
                dvalues,
                np.int64(cols),
                np.int64(rows),
                np.int64(nnz),
                np.int64(steps),
                err,
            ),
        )
        self._check_ids(err, "sparse_linear_values_grad", f"[0, {rows})")
        return dvalues

    def _op_spike_linear(self, node, args, attrs):
        """事件驱动的脉冲线性：事件幅度直接从脉冲矩阵按列号读取。

        与 ``sparse_linear`` 的结构完全相同，只是把 ``values[e]`` 换成
        ``spikes[s, indices[e]]``。省掉 values 缓冲之后，反向对脉冲就是一次
        普通 GEMM，未发放位置的 surrogate 梯度不会被丢掉。
        """
        cp = self.cp
        spikes, weights, indices, offsets = args
        spikes = self._contig(spikes)
        weights = self._contig(weights)
        indices = cp.ascontiguousarray(indices).reshape(-1)
        offsets = cp.ascontiguousarray(offsets).reshape(-1)
        steps = int(offsets.size) - 1
        rows, cols = int(weights.shape[0]), int(weights.shape[1])
        out = cp.zeros((steps, cols), dtype=spikes.dtype)
        total = steps * cols
        if total == 0:
            return out
        kernel = self._kernel(
            ("spike_linear", self._kind(spikes)),
            lambda: self.codegen.render(self.codegen.spike_linear_source(), spikes.dtype),
            "spike_linear",
        )
        err = cp.zeros(1, dtype=cp.int32)
        block, grid = self._grid("spike_linear", total)
        kernel(
            (grid,),
            (block,),
            (
                spikes,
                weights,
                indices,
                offsets,
                out,
                np.int64(cols),
                np.int64(rows),
                np.int64(int(indices.size)),
                np.int64(total),
                err,
            ),
        )
        self._check_ids(err, "spike_linear", f"[0, {rows})")
        return out

    def _op_spike_linear_weight_grad(self, node, args, attrs):
        """``spike_linear`` 对权重的梯度：事件贡献散射回各自的行。

        行号来自 CSR 的 ``indices``，幅度来自脉冲矩阵本身；输出先清零再原子
        累加——同一行被多个事件命中是常态。
        """
        cp = self.cp
        dout, spikes, indices, offsets = args
        dout = self._contig(dout)
        spikes = self._contig(spikes)
        indices = cp.ascontiguousarray(indices).reshape(-1)
        offsets = cp.ascontiguousarray(offsets).reshape(-1)
        rows = int(spikes.shape[1])
        steps, cols = int(dout.shape[0]), int(dout.shape[1])
        dweights = cp.zeros((rows, cols), dtype=dout.dtype)
        total = steps * cols
        if total == 0:
            return dweights
        kernel = self._kernel(
            ("spike_linear_weight_grad", self._kind(dout)),
            lambda: self.codegen.render(self.codegen.spike_linear_weight_grad_source(), dout.dtype),
            "spike_linear_weight_grad",
        )
        err = cp.zeros(1, dtype=cp.int32)
        block, grid = self._grid("spike_linear_weight_grad", total)
        kernel(
            (grid,),
            (block,),
            (
                dout,
                spikes,
                indices,
                offsets,
                dweights,
                np.int64(cols),
                np.int64(rows),
                np.int64(int(indices.size)),
                np.int64(total),
                err,
            ),
        )
        self._check_ids(err, "spike_linear_weight_grad", f"[0, {rows})")
        return dweights
