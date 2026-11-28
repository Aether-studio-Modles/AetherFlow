"""归一化：LayerNorm、BatchNorm（含统计量）与 RMSNorm。"""

from __future__ import annotations

from .base import BackendCapabilityError
import numpy as np


class _NormMixin:
    """归一化：LayerNorm、BatchNorm（含统计量）与 RMSNorm。"""

    def _op_layernorm(self, node, args, attrs):
        cp = self.cp
        x, gamma, beta = args[0], args[1], args[2]
        residual = args[3] if len(args) > 3 else None
        prep, alloc, finish, rows, cols, _norm_shape, norm_axes = self._norm_layout(
            x, attrs.get("axes", (-1,)), "layernorm"
        )
        x_shape = tuple(int(dim) for dim in x.shape)
        gamma = self._norm_param(gamma, cols, x_shape, norm_axes, "gamma", "layernorm")
        beta = self._norm_param(beta, cols, x_shape, norm_axes, "beta", "layernorm")
        if gamma.dtype != x.dtype or beta.dtype != x.dtype:
            raise BackendCapabilityError(
                f"layernorm 的 gamma/beta 必须与 x 同 dtype，收到 "
                f"{gamma.dtype} / {beta.dtype} 与 {x.dtype}"
            )
        if residual is not None:
            if tuple(residual.shape) != tuple(x.shape) or residual.dtype != x.dtype:
                raise BackendCapabilityError(
                    f"layernorm 的 residual 必须与 x 同形状、同 dtype，收到 "
                    f"{residual.shape} 与 {x.shape}"
                )
        eps = float(attrs.get("eps", 1e-5))
        if x.size == 0:
            return cp.empty_like(x)
        xf = prep(x)
        rf = prep(residual) if residual is not None else None
        full, of = alloc()
        dtype_key = self._kind(x)
        block = self._reduce_block("layernorm", cols)
        # 每线程缓存的元素数：cols 能被 block 整除时第一遍扫描就把值留在
        # 寄存器里，后两遍不再重读显存（见 ``_reduce_cache``）。
        cache = self._reduce_cache(cols, block)
        if residual is None:
            kernel = self._kernel(
                ("layernorm", dtype_key, cache),
                lambda: self.codegen.render(
                    self.codegen.layernorm_source(cache), x.dtype
                ),
                "layernorm_last",
            )
            kernel(
                (rows,),
                (block,),
                (xf, gamma, beta, of, np.int64(rows), np.int64(cols), np.float32(eps)),
            )
        else:
            kernel = self._kernel(
                ("layernorm_residual", dtype_key),
                lambda: self.codegen.render(self.codegen.layernorm_residual_source(), x.dtype),
                "layernorm_residual",
            )
            kernel(
                (rows,),
                (block,),
                (
                    xf,
                    rf,
                    gamma,
                    beta,
                    of,
                    np.int64(rows),
                    np.int64(cols),
                    np.float32(eps),
                ),
            )
        return finish(full, of)

    def _batch_norm_plan(self, x, attrs, name: str):
        """把 x 看成 ``(outer, channels, inner)``——BN 内核只认这三元组。

        这样同一个内核既能处理 NCHW（axis=1, inner=H*W），也能处理任意秩、
        任意通道轴，而不必为每种布局写一份模板。
        """
        if x.ndim < 2:
            raise BackendCapabilityError(f"CUDA 的 {name} 要求秩 >= 2，收到 {x.ndim}")
        axis = int(attrs.get("axis", 1)) % x.ndim
        shape = [int(d) for d in x.shape]
        outer = 1
        for d in shape[:axis]:
            outer *= d
        inner = 1
        for d in shape[axis + 1 :]:
            inner *= d
        return outer, shape[axis], inner

    def _op_batch_norm(self, node, args, attrs):
        cp = self.cp
        x = self._contig(args[0])
        gamma = self._contig(args[1])
        beta = self._contig(args[2])
        outer, channels, inner = self._batch_norm_plan(x, attrs, "batch_norm")
        if tuple(gamma.shape) != (channels,) or tuple(beta.shape) != (channels,):
            raise BackendCapabilityError(
                f"batch_norm 的 gamma/beta 必须为 ({channels},)，收到 "
                f"{tuple(gamma.shape)} 与 {tuple(beta.shape)}"
            )
        eps = float(attrs.get("eps", 1e-5))
        out = cp.empty(x.shape, dtype=x.dtype)
        if x.size == 0:
            return out
        dtype_key = self._kind(x)
        kernel = self._kernel(
            ("batch_norm", dtype_key),
            lambda: self.codegen.render(self.codegen.batch_norm_source(), x.dtype),
            "batch_norm",
        )
        block = self._reduce_block("batch_norm", outer * inner)
        kernel(
            (channels,),
            (block,),
            (
                x,
                gamma,
                beta,
                out,
                np.int64(outer),
                np.int64(channels),
                np.int64(inner),
                np.float32(eps),
            ),
        )
        return out

    def _op_batch_norm_stats(self, node, args, attrs):
        cp = self.cp
        x = self._contig(args[0])
        outer, channels, inner = self._batch_norm_plan(x, attrs, "batch_norm_stats")
        out = cp.empty((2, channels), dtype=x.dtype)
        if x.size == 0:
            return out
        dtype_key = self._kind(x)
        kernel = self._kernel(
            ("batch_norm_stats", dtype_key),
            lambda: self.codegen.render(self.codegen.batch_norm_stats_source(), x.dtype),
            "batch_norm_stats",
        )
        block = self._reduce_block("batch_norm_stats", outer * inner)
        kernel(
            (channels,),
            (block,),
            (x, out, np.int64(outer), np.int64(channels), np.int64(inner)),
        )
        return out

    def _op_rmsnorm(self, node, args, attrs):
        """RMSNorm：与 layernorm 同一个"一行一个 block"的归约形态。

        归一化轴的处理与 layernorm 完全一致（见 ``_norm_layout``）：``axes`` 是
        最后一维时零拷贝，其余轴集合先置换再算。gamma 的可接受形态也一致：只要
        能广播到 ``x`` 且不随非归一化维变化（``(cols,)``、归一化维的自然形状、
        带 1 前缀的形态都算），就展平成 ``(cols,)`` 交给内核。
        """
        cp = self.cp
        x, gamma = args
        prep, alloc, finish, rows, cols, _norm_shape, norm_axes = self._norm_layout(
            x, attrs.get("axes", (-1,)), "rmsnorm"
        )
        x_shape = tuple(int(dim) for dim in x.shape)
        gamma = self._norm_param(gamma, cols, x_shape, norm_axes, "gamma", "rmsnorm")
        if gamma.dtype != x.dtype:
            raise BackendCapabilityError(
                f"rmsnorm 的 gamma 必须与 x 同 dtype，收到 {gamma.dtype} 与 {x.dtype}"
            )
        eps = float(attrs.get("eps", 1e-6))
        if x.size == 0:
            return cp.empty_like(x)
        xf = prep(x)
        full, of = alloc()
        dtype_key = self._kind(x)
        block = self._reduce_block("rmsnorm", cols)
        cache = self._reduce_cache(cols, block)
        kernel = self._kernel(
            ("rmsnorm", dtype_key, cache),
            lambda: self.codegen.render(
                self.codegen.rmsnorm_source(cache), x.dtype
            ),
            "rmsnorm_last",
        )
        kernel(
            (rows,),
            (block,),
            (xf, gamma, of, np.int64(rows), np.int64(cols), np.float32(eps)),
        )
        return finish(full, of)
