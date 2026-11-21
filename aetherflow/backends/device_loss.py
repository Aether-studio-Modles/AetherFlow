"""损失：融合交叉熵及其反向。"""

from __future__ import annotations

from .base import BackendCapabilityError
import numpy as np


class _LossMixin:
    """损失：融合交叉熵及其反向。"""

    def _ce_geometry(self, logits, target, op: str):
        """交叉熵的 ``(rows, vocab)``，并把形状不一致挡在内核之前。"""
        if logits.ndim < 1:
            raise BackendCapabilityError(
                f"{self._LABEL}的 {op} 需要 logits 至少 1 维，收到形状 {tuple(logits.shape)}"
            )
        vocab = int(logits.shape[-1])
        if vocab < 1:
            raise BackendCapabilityError(
                f"{self._LABEL}的 {op} 的词表维必须 >= 1，收到形状 {tuple(logits.shape)}"
            )
        rows = int(logits.size) // vocab
        if int(target.size) != rows:
            raise BackendCapabilityError(
                f"{self._LABEL}的 {op} 的 target 元素数应为 {rows}，"
                f"收到 {int(target.size)}（形状 {tuple(target.shape)}）"
            )
        return rows, vocab

    def _op_cross_entropy(self, node, args, attrs):
        """融合交叉熵：一个 block 一行，两遍扫描求 logsumexp。

        只读 logits 两遍（max 与 exp 和），**不物化** ``(rows, vocab)`` 的
        概率矩阵。``out`` 是每行损失，形状为 ``logits.shape[:-1]``；训练里
        再接一个 ``reduce_mean`` 就是标量损失。
        """
        cp = self.cp
        logits, target = args
        rows, vocab = self._ce_geometry(logits, target, "cross_entropy")
        out = cp.empty(tuple(int(d) for d in logits.shape[:-1]), dtype=logits.dtype)
        if rows == 0:
            return out

        dtype_key = self._kind(logits)
        kernel = self._kernel(
            ("cross_entropy", str(target.dtype), dtype_key),
            lambda: self.codegen.render(
                self.codegen.cross_entropy_source(target.dtype), logits.dtype
            ),
            "cross_entropy",
        )
        block = self._reduce_block("cross_entropy", vocab)
        err = cp.zeros(1, dtype=cp.int32)
        kernel(
            (rows,),
            (block,),
            (
                self._contig(logits),
                cp.ascontiguousarray(target).reshape(-1),
                out,
                np.int64(rows),
                np.int64(vocab),
                err,
            ),
        )
        self._check_ids(err, "cross_entropy", f"[0, {vocab})")
        return out

    def _op_cross_entropy_backward(self, node, args, attrs):
        """``(softmax - onehot) * dloss``：重算行统计，不缓存前向概率。"""
        cp = self.cp
        logits, target, dloss = args
        rows, vocab = self._ce_geometry(logits, target, "cross_entropy_backward")
        head = tuple(int(d) for d in logits.shape[:-1])
        if tuple(int(d) for d in dloss.shape) != head:
            raise BackendCapabilityError(
                f"{self._LABEL}的 cross_entropy_backward 的 dloss 形状应为 {head}，"
                f"收到 {tuple(dloss.shape)}"
            )
        dlogits = cp.empty(tuple(int(d) for d in logits.shape), dtype=logits.dtype)
        if rows == 0:
            return dlogits

        dtype_key = self._kind(logits)
        kernel = self._kernel(
            ("cross_entropy_backward", str(target.dtype), dtype_key),
            lambda: self.codegen.render(
                self.codegen.cross_entropy_backward_source(target.dtype), logits.dtype
            ),
            "cross_entropy_backward",
        )
        block = self._reduce_block("cross_entropy", vocab)
        err = cp.zeros(1, dtype=cp.int32)
        kernel(
            (rows,),
            (block,),
            (
                self._contig(logits),
                cp.ascontiguousarray(target).reshape(-1),
                self._contig(dloss).reshape(-1),
                dlogits,
                np.int64(rows),
                np.int64(vocab),
                err,
            ),
        )
        self._check_ids(err, "cross_entropy_backward", f"[0, {vocab})")
        return dlogits
