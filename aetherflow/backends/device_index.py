"""索引类算子：embedding、gather 及其反向。"""

from __future__ import annotations

from .base import BackendCapabilityError
import numpy as np


class _IndexMixin:
    """索引类算子：embedding、gather 及其反向。"""

    def _op_embedding(self, node, args, attrs):
        """按 ids 查表；表按 ``(vocab, row)`` 视角处理，尾随维任意。

        越界下标不能在内核里抛 Python 异常，所以内核把它记进一个设备标量，
        由 ``_check_ids`` 在主机侧转成 ``BackendCapabilityError``。
        内核里同时做 clamp，是为了走向这条报错路径时不发生非法访存。
        """
        cp = self.cp
        table, ids = args
        vocab = int(table.shape[0])
        tail = tuple(int(d) for d in table.shape[1:])
        row = int(np.prod(tail)) if tail else 1
        ids_flat = cp.ascontiguousarray(ids).reshape(-1)
        out = cp.empty(tuple(ids.shape) + tail, dtype=table.dtype)
        total = int(ids_flat.size) * row
        if total == 0:
            return out

        dtype_key = self._kind(table)
        kernel = self._kernel(
            ("embedding", str(ids.dtype), dtype_key),
            lambda: self.codegen.render(self.codegen.embedding_source(ids.dtype), table.dtype),
            "embedding",
        )
        err = cp.zeros(1, dtype=cp.int32)
        block, grid = self._grid("embedding", total)
        kernel(
            (grid,),
            (block,),
            (
                cp.ascontiguousarray(table),
                ids_flat,
                out,
                np.int64(ids_flat.size),
                np.int64(row),
                np.int64(vocab),
                err,
                np.int64(total),
            ),
        )
        self._check_ids(err, "embedding", f"[0, {vocab})")
        return out

    def _op_embedding_backward(self, node, args, attrs):
        """查表反向：把每条 dy 散射累加回对应表行。

        重复下标必须累加（同一个 token 在一个 batch 里出现多次是常态），
        这里是 atomicAdd 存在的最直接理由。
        """
        cp = self.cp
        dy, ids = args
        shape = tuple(int(d) for d in attrs["table_shape"])
        vocab = shape[0]
        tail = shape[1:]
        row = int(np.prod(tail)) if tail else 1
        expected = tuple(ids.shape) + tail
        if tuple(dy.shape) != expected:
            raise BackendCapabilityError(
                f"embedding_backward 的 dy 形状应为 {expected}，收到 {dy.shape}"
            )
        ids_flat = cp.ascontiguousarray(ids).reshape(-1)
        dtable = cp.zeros(shape, dtype=dy.dtype)
        total = int(ids_flat.size) * row
        if total == 0:
            return dtable

        dtype_key = self._kind(dy)
        kernel = self._kernel(
            ("embedding_backward", str(ids.dtype), dtype_key),
            lambda: self.codegen.render(
                self.codegen.embedding_backward_source(ids.dtype), dy.dtype
            ),
            "embedding_backward",
        )
        err = cp.zeros(1, dtype=cp.int32)
        block, grid = self._grid("embedding_backward", total)
        kernel(
            (grid,),
            (block,),
            (
                cp.ascontiguousarray(dy),
                ids_flat,
                dtable,
                np.int64(ids_flat.size),
                np.int64(row),
                np.int64(vocab),
                err,
                np.int64(total),
            ),
        )
        self._check_ids(err, "embedding_backward", f"[0, {vocab})")
        return dtable

    def _gather_dim(self, attrs: dict, rank: int, op: str) -> int:
        """负轴归一化 + 越界显式报错（与 ``ir/ops.py`` 的 ``_norm_axis`` 一致）。"""
        raw = attrs.get("dim", -1)
        dim = int(raw)
        if dim < 0:
            dim += rank
        if not 0 <= dim < rank:
            raise BackendCapabilityError(
                f"{self._LABEL}的 {op} 的 dim 越界：得到 {raw}，秩为 {rank}"
            )
        return dim

    def _gather_shapes(self, x_shape, idx_shape, dim: int, op: str) -> None:
        """秩相同、除 dim 外每一维相等——与 IR 的 ``_check_gather_shapes`` 同义。"""
        if len(x_shape) != len(idx_shape):
            raise BackendCapabilityError(
                f"{self._LABEL}的 {op} 的 idx 秩必须与 x 相同"
                f"（{len(x_shape)}），收到 {len(idx_shape)}"
            )
        for d, (a, b) in enumerate(zip(x_shape, idx_shape)):
            if d != dim and int(a) != int(b):
                raise BackendCapabilityError(
                    f"{self._LABEL}的 {op} 要求除 dim={dim} 外每一维相等：第 {d} 维是 {a} 与 {b}"
                )

    def _op_gather(self, node, args, attrs):
        """沿任意一维按下标取值；idx / out 视为 ``(outer, idx_dim, inner)``。

        收紧后的形状约束（除 dim 外与 x 相等）让偏移只依赖三个运行期长度，
        不必把整张 shape 表传进内核。
        """
        cp = self.cp
        x, idx = args
        if x.ndim < 1:
            raise BackendCapabilityError(
                f"{self._LABEL}的 gather 需要 x 至少 1 维，收到形状 {tuple(x.shape)}"
            )
        dim = self._gather_dim(attrs, x.ndim, "gather")
        self._gather_shapes(x.shape, idx.shape, dim, "gather")
        out = cp.empty(tuple(int(d) for d in idx.shape), dtype=x.dtype)
        total = int(idx.size)
        if total == 0:
            return out
        inner = int(np.prod(x.shape[dim + 1 :])) if dim + 1 < x.ndim else 1
        idx_dim = int(idx.shape[dim])
        x_dim = int(x.shape[dim])
        if x_dim < 1:
            raise BackendCapabilityError(
                f"{self._LABEL}的 gather 的 dim={dim} 长度为 0（形状 {tuple(x.shape)}）"
            )

        ids = cp.ascontiguousarray(idx).reshape(-1)
        dtype_key = self._kind(x)
        kernel = self._kernel(
            ("gather", str(idx.dtype), dtype_key),
            lambda: self.codegen.render(self.codegen.gather_source(idx.dtype), x.dtype),
            "gather",
        )
        err = cp.zeros(1, dtype=cp.int32)
        block, grid = self._grid("gather", total)
        kernel(
            (grid,),
            (block,),
            (
                self._contig(x),
                ids,
                out,
                np.int64(total),
                np.int64(inner),
                np.int64(idx_dim),
                np.int64(x_dim),
                err,
            ),
        )
        self._check_ids(err, "gather", f"[0, {x_dim})")
        return out

    def _op_gather_backward(self, node, args, attrs):
        """把 dy 散射累加回 x 的被选位置；重复下标必须累加。"""
        cp = self.cp
        dy, idx = args
        shape = tuple(int(d) for d in attrs["x_shape"])
        dim = self._gather_dim(attrs, len(shape), "gather_backward")
        if tuple(int(d) for d in dy.shape) != tuple(int(d) for d in idx.shape):
            raise BackendCapabilityError(
                f"{self._LABEL}的 gather_backward 的 dy 形状必须等于 idx 形状 "
                f"{tuple(idx.shape)}，收到 {tuple(dy.shape)}"
            )
        self._gather_shapes(shape, dy.shape, dim, "gather_backward")
        dx = cp.zeros(shape, dtype=dy.dtype)
        total = int(dy.size)
        if total == 0:
            return dx
        inner = int(np.prod(shape[dim + 1 :])) if dim + 1 < len(shape) else 1
        idx_dim = int(dy.shape[dim])
        x_dim = shape[dim]
        if x_dim < 1:
            raise BackendCapabilityError(
                f"{self._LABEL}的 gather_backward 的 x_shape[{dim}] 为 0（{shape}）"
            )

        ids = cp.ascontiguousarray(idx).reshape(-1)
        dtype_key = self._kind(dy)
        kernel = self._kernel(
            ("gather_backward", str(idx.dtype), dtype_key),
            lambda: self.codegen.render(self.codegen.gather_backward_source(idx.dtype), dy.dtype),
            "gather_backward",
        )
        err = cp.zeros(1, dtype=cp.int32)
        block, grid = self._grid("gather_backward", total)
        kernel(
            (grid,),
            (block,),
            (
                self._contig(dy),
                ids,
                dx,
                np.int64(total),
                np.int64(inner),
                np.int64(idx_dim),
                np.int64(x_dim),
                err,
            ),
        )
        self._check_ids(err, "gather_backward", f"[0, {x_dim})")
        return dx
