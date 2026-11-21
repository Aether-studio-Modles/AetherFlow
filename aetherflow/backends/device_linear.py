"""线性代数与量化：matmul、int8 量化 / 反量化、weight-only 线性层。"""

from __future__ import annotations

from .base import BackendCapabilityError
import numpy as np
from .device_shared import _DEQUANTIZE_I8_SRC, _QUANTIZE_I8_SRC


class _LinearMixin:
    """线性代数与量化：matmul、int8 量化 / 反量化、weight-only 线性层。"""

    # ------------------------------------------------------------------
    # 矩阵
    # ------------------------------------------------------------------

    def _op_matmul(self, node, args, attrs):
        cp = self.cp
        a, b = args
        if a.ndim < 2 or b.ndim < 2:
            raise BackendCapabilityError(
                f"{self._LABEL}的 matmul 需要秩 >= 2，收到 {a.shape} 与 {b.shape}"
            )
        a = self._contig(a)
        b = self._contig(b)
        if a.dtype != b.dtype:
            raise BackendCapabilityError(
                f"matmul 两侧 dtype 必须一致，收到 {a.dtype} 与 {b.dtype}；请显式插入 cast"
            )
        m, k = int(a.shape[-2]), int(a.shape[-1])
        k2, n = int(b.shape[-2]), int(b.shape[-1])
        if k != k2:
            raise BackendCapabilityError(f"matmul 内维不匹配：{a.shape} @ {b.shape}")
        try:
            batch_shape = tuple(np.broadcast_shapes(a.shape[:-2], b.shape[:-2]))
        except ValueError as exc:
            raise BackendCapabilityError(f"matmul 的批量维无法广播：{a.shape} @ {b.shape}") from exc

        out = cp.empty(batch_shape + (m, n), dtype=a.dtype)
        if out.size == 0:
            return out

        blas = self._cublas()
        blas.set_stream(self.current_stream_ptr())

        # f16 / bf16 走 GemmEx（f32 累加），f32 走 Sgemm：三条路径的累加
        # 精度一致，差别只在输入输出的存储宽度。bf16 的存储是 uint16 位模式，
        # cuBLAS 按 CUDA_R_16BF 解释同一段内存，因此在 A800/sm_80 上会落到
        # Tensor Core；HIP 没有声明 bf16 支持面，这个分支到不了。
        f16 = a.dtype == cp.float16
        bf16 = a.dtype == cp.uint16
        if not batch_shape:
            if f16:
                blas.hgemm(a, b, out)
            elif bf16:
                blas.bgemm(a, b, out)
            else:
                blas.sgemm(a, b, out)
            return out

        batch = int(np.prod(batch_shape))
        stride_a = self._batch_stride(a.shape[:-2], batch_shape, m * k, "左操作数")
        stride_b = self._batch_stride(b.shape[:-2], batch_shape, k * n, "右操作数")
        if f16:
            blas.hgemm_batched(a, b, out, batch, stride_a, stride_b, m * n)
        elif bf16:
            blas.bgemm_batched(a, b, out, batch, stride_a, stride_b, m * n)
        else:
            blas.sgemm_batched(a, b, out, batch, stride_a, stride_b, m * n)
        return out

    @staticmethod
    def _batch_stride(shape: tuple, out_shape: tuple, dense: int, what: str) -> int:
        """批量 GEMM 的步长：形状一致用密集步长，整体只有 1 个 batch 时用 0。

        部分广播（例如 (2,1) 对 (2,3)）无法用一个标量 stride 表达，
        明确拒绝而不是悄悄算错。
        """
        if shape == out_shape:
            return dense
        if not shape or all(d == 1 for d in shape):
            return 0
        raise BackendCapabilityError(
            f"matmul {what}的批量维 {shape} 与输出批量维 {out_shape} 不匹配："
            "cuBLAS 的 strided batched 只能表达“完全一致”或“整体为 1”"
        )

    # ------------------------------------------------------------------
    # 量化（weight-only int8）
    # ------------------------------------------------------------------

    def _quant_scale_arg(self, scale, shape):
        """把 scale 变成内核能吃的连续 f32 数组。

        返回 ``(数组, cols, per_row)``。``per_row = 1`` 时内核按
        ``s[i % cols]`` 取缩放系数，这样逐输出通道量化**不会**把 scale 物化成
        与权重同大的 f32（那会让"省 3/4 权重"变成"多 1 倍临时量"）。
        其余可广播形态（逐张量、或与权重同形）走广播拷贝；IR 已经保证
        scale 能广播到权重，这里只是选择用哪条读法。
        """
        if not scale.flags.c_contiguous:
            scale = self.cp.ascontiguousarray(scale)
        if scale.ndim == 1 and int(scale.size) == int(shape[-1]):
            return scale, int(shape[-1]), 1
        if tuple(scale.shape) == tuple(shape):
            return scale, 0, 0
        return (
            self.cp.ascontiguousarray(self.cp.broadcast_to(scale, shape)),
            0,
            0,
        )

    def _op_quantize_i8(self, node, args, attrs):
        """对称 int8 量化：``rintf(w / s)`` 后饱和到 [-127, 127]。

        用一段自研内核而不是 cupy 的逐元素原语：CUDA 侧的 ``cp`` 是 cupy，
        HIP 侧是同一套接口的自家实现，而 HIP 那份没有 ``rint`` / ``clip``。
        半偶舍入是语义的一部分，不能在两个后端各写一次近似——所以两边共用
        这一份源码（NVRTC 与 hipRTC 都直接吃它）。

        非正 scale 的检查走 ``to_host``：scale 只有 N 个元素，代价可忽略；
        量化是离线步骤（导出权重时跑一次），不在每步推理的热路径上。
        """
        cp = self.cp
        weight, scale = args
        if weight.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 quantize_i8 只接受 f32 权重，收到 {weight.dtype}"
            )
        if weight.ndim < 1:
            raise BackendCapabilityError(
                f"{self._LABEL}的 quantize_i8 要求权重至少 1 维，收到 {weight.shape}"
            )
        if float(np.min(self.to_host(scale))) <= 0.0:
            raise BackendCapabilityError(
                "quantize_i8 的 scale 必须严格为正；非正 scale 的量化为未定义"
            )
        w = self._contig(weight)
        s, cols, per_row = self._quant_scale_arg(scale, w.shape)
        out = cp.empty(w.shape, np.int8)
        n = int(w.size)
        if n == 0:
            return out
        kernel = self._kernel(("quantize_i8", per_row), _QUANTIZE_I8_SRC, "af_quantize_i8")
        block, grid = self._grid("quantize_i8", n)
        kernel(
            (grid,),
            (block,),
            (w, s, out, np.int64(n), np.int64(cols), np.int64(per_row)),
        )
        return out

    def _op_dequantize_i8(self, node, args, attrs):
        """i8 -> f32 的放大。输出必须连续：``linear_i8`` 要把它交给 GEMM。"""
        cp = self.cp
        q, scale = args
        if q.dtype != cp.int8:
            raise BackendCapabilityError(
                f"{self._LABEL}的 dequantize_i8 第一个输入必须是 i8，收到 {q.dtype}"
            )
        if scale.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 dequantize_i8 的 scale 必须是 f32，收到 {scale.dtype}"
            )
        qq = self._contig(q)
        s, cols, per_row = self._quant_scale_arg(scale, qq.shape)
        out = cp.empty(qq.shape, np.float32)
        n = int(qq.size)
        if n == 0:
            return out
        kernel = self._kernel(("dequantize_i8", per_row), _DEQUANTIZE_I8_SRC, "af_dequantize_i8")
        block, grid = self._grid("dequantize_i8", n)
        kernel(
            (grid,),
            (block,),
            (qq, s, out, np.int64(n), np.int64(cols), np.int64(per_row)),
        )
        return out

    def _op_linear_i8(self, node, args, attrs):
        """weight-only int8 线性层：反量化权重后走既有 f32 GEMM。

        "int8" 在这里只体现在**权重驻留与读带宽**上：读出 i8、就地放大成
        f32、再交给 cuBLAS / hipBLAS。这就是 weight-only 量化的全部收益，
        不宣称算力提升——真正的 int8 矩阵乘需要另一套累加语义。
        """
        cp = self.cp
        x, qweight, scale = args[0], args[1], args[2]
        if x.dtype != cp.float32:
            raise BackendCapabilityError(f"{self._LABEL}的 linear_i8 要求 x 是 f32，收到 {x.dtype}")
        if x.ndim < 2 or qweight.ndim != 2:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_i8 要求 x 秩 >= 2、权重秩 2，"
                f"收到 {x.shape} 与 {qweight.shape}"
            )
        weight = self._op_dequantize_i8(node, [qweight, scale], attrs)
        out = self._op_matmul(node, [x, weight], {})
        if len(args) > 3:
            out = self._binary("add", [out, args[3]])
        return out
