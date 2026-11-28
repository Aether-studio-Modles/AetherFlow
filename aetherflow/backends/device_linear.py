"""线性代数与量化：matmul、int8 量化 / 反量化、weight-only 与 W8A8 线性层。"""

from __future__ import annotations

from .base import BackendCapabilityError
import numpy as np
from .device_shared import (
    _DEQUANTIZE_I8_GROUP_SRC,
    _DEQUANTIZE_I8_GROUP_ZP_SRC,
    _DEQUANTIZE_I8_SRC,
    _DEQUANTIZE_I8_ZP_SRC,
    _LINEAR_I8A8_DP4A_SRC,
    _LINEAR_I8A8_SRC,
    _QUANTIZE_I8_SRC,
    _QUANTIZE_I8_ZP_SRC,
)


class _LinearMixin:
    """线性代数与量化：matmul、int8 量化 / 反量化、weight-only 与 W8A8 线性层。"""

    #: W8A8 的点积是否走 ``__dp4a``（sm_61+ 的 4 路 int8 点积指令）。CUDA 侧
    #: 覆盖成 True；HIP 的 hipRTC 不带 ``hip_runtime.h``，走标量乘加。两条路
    #: 都是精确整数运算，累加结果逐位相同，只是指令条数不同。
    _I8A8_DOT4 = False

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

    def _zp_args(self, shape, scale, zp):
        """把可广播的 scale 与 zero_point 归一成内核能吃的形态。

        返回 ``(scale, zp, cols, per_row)``。两者都恰好是"最后一维长度 = N"
        的一维张量时走 ``[i % cols]`` 的复用读法——这是逐通道量化的常态，
        避免把参数物化成与数据同大的数组。其余广播形态（逐张量、逐元素、
        或两者形态不一致）一次性物化：量化是离线步骤，反量化的参数通常也
        只有 N 个，代价可忽略，而"任意广播都能正确"比省这份临时量重要。
        """
        cp = self.cp
        n = int(shape[-1]) if len(shape) else 0
        if (
            scale.ndim == 1
            and zp.ndim == 1
            and int(scale.size) == n
            and int(zp.size) == n
        ):
            return self._contig(scale), self._contig(zp), n, 1
        return (
            cp.ascontiguousarray(cp.broadcast_to(scale, shape)),
            cp.ascontiguousarray(cp.broadcast_to(zp, shape)),
            0,
            0,
        )

    def _op_quantize_i8_zp(self, node, args, attrs):
        """非对称 int8 量化：``clip(rint(x / scale) + zp, -128, 127)``。

        与对称版只差零点与负端点：零点在**舍入之后**加到整数格点上，饱和
        下界用满 -128。非正 scale 的检查与对称版同一口径（走 ``to_host``）。
        """
        cp = self.cp
        x, scale, zp = args
        if x.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 quantize_i8_zp 只接受 f32 输入，收到 {x.dtype}"
            )
        if scale.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 quantize_i8_zp 的 scale 必须是 f32，收到 {scale.dtype}"
            )
        if zp.dtype != cp.int8:
            raise BackendCapabilityError(
                f"{self._LABEL}的 quantize_i8_zp 的 zero_point 必须是 i8，"
                f"收到 {zp.dtype}"
            )
        if x.ndim < 1:
            raise BackendCapabilityError(
                f"{self._LABEL}的 quantize_i8_zp 至少需要 1 维输入，收到 {x.shape}"
            )
        if float(np.min(self.to_host(scale))) <= 0.0:
            raise BackendCapabilityError(
                "quantize_i8_zp 的 scale 必须严格为正；非正 scale 的量化为未定义"
            )
        xx = self._contig(x)
        s, z, cols, per_row = self._zp_args(xx.shape, scale, zp)
        out = cp.empty(xx.shape, np.int8)
        n = int(xx.size)
        if n == 0:
            return out
        kernel = self._kernel(
            ("quantize_i8_zp", per_row), _QUANTIZE_I8_ZP_SRC, "af_quantize_i8_zp"
        )
        block, grid = self._grid("quantize_i8_zp", n)
        kernel(
            (grid,),
            (block,),
            (xx, s, z, out, np.int64(n), np.int64(cols), np.int64(per_row)),
        )
        return out

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

    def _op_dequantize_i8_zp(self, node, args, attrs):
        """非对称反量化：``f32((int32)q - (int32)zp) * scale``。

        减法留在 int32 里做（-128 减 127 等于 -255，超出 i8），与参考实现
        逐位相同。
        """
        cp = self.cp
        q, scale, zp = args
        if q.dtype != cp.int8 or zp.dtype != cp.int8:
            raise BackendCapabilityError(
                f"{self._LABEL}的 dequantize_i8_zp 要求 q 与 zero_point 都是 i8，"
                f"收到 {q.dtype} 与 {zp.dtype}"
            )
        if scale.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 dequantize_i8_zp 的 scale 必须是 f32，"
                f"收到 {scale.dtype}"
            )
        qq = self._contig(q)
        s, z, cols, per_row = self._zp_args(qq.shape, scale, zp)
        out = cp.empty(qq.shape, np.float32)
        n = int(qq.size)
        if n == 0:
            return out
        kernel = self._kernel(
            ("dequantize_i8_zp", per_row), _DEQUANTIZE_I8_ZP_SRC, "af_dequantize_i8_zp"
        )
        block, grid = self._grid("dequantize_i8_zp", n)
        kernel(
            (grid,),
            (block,),
            (qq, s, z, out, np.int64(n), np.int64(cols), np.int64(per_row)),
        )
        return out

    def _group_shape(self, name: str, q, scale) -> tuple:
        """校验逐组反量化的形状，返回 ``(cols, rows_per_group)``。

        IR 的 infer 已保证这些条件；设备端再判一次是因为图可以来自反序列化，
        不能假设上游一定跑过 verifier——形状对不上时越界读显存不会报错，
        只会悄悄算错。
        """
        if q.ndim != 2:
            raise BackendCapabilityError(
                f"{self._LABEL}的 {name} 要求 q 是二维 (K, N)，收到 {q.shape}"
            )
        if scale.ndim != 2 or int(scale.shape[1]) != int(q.shape[1]):
            raise BackendCapabilityError(
                f"{self._LABEL}的 {name} 要求 scale 形如 (K/G, N)，"
                f"收到 {scale.shape} 与 q {q.shape}"
            )
        rows = int(scale.shape[0])
        k = int(q.shape[0])
        if rows <= 0 or k % rows != 0:
            raise BackendCapabilityError(
                f"{self._LABEL}的 {name} 的组数 {rows} 不能整除 K={k}"
            )
        return int(q.shape[1]), k // rows

    def _op_dequantize_i8_group(self, node, args, attrs):
        """逐组对称反量化：q (K, N) 沿 K 每 ``K/G`` 行共用一份 scale。

        组大小由形状推出（G = K / scale.shape[0]），不另设 attr（DESIGN 5.317）；
        内核按 ``row / G`` 折回 scale 的行，组内共用同一份系数。输出必须连续，
        下游通常直接接 GEMM。
        """
        cp = self.cp
        q, scale = args
        if q.dtype != cp.int8:
            raise BackendCapabilityError(
                f"{self._LABEL}的 dequantize_i8_group 第一个输入必须是 i8，收到 {q.dtype}"
            )
        if scale.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 dequantize_i8_group 的 scale 必须是 f32，收到 {scale.dtype}"
            )
        cols, group = self._group_shape("dequantize_i8_group", q, scale)
        qq = self._contig(q)
        ss = self._contig(scale)
        out = cp.empty(qq.shape, np.float32)
        n = int(qq.size)
        if n == 0:
            return out
        kernel = self._kernel(
            ("dequantize_i8_group",), _DEQUANTIZE_I8_GROUP_SRC, "af_dequantize_i8_group"
        )
        block, grid = self._grid("dequantize_i8_group", n)
        kernel(
            (grid,),
            (block,),
            (qq, ss, out, np.int64(n), np.int64(cols), np.int64(group)),
        )
        return out

    def _op_dequantize_i8_group_zp(self, node, args, attrs):
        """逐组非对称反量化：``(int32)q - (int32)zp`` 再乘 scale。

        减法留在整数里做，与参考实现的 ``(int32)q - (int32)zp`` 顺序一致，
        因此逐位相同（不会因为先转 f32 而丢低位）。
        """
        cp = self.cp
        q, scale, zp = args
        if q.dtype != cp.int8 or zp.dtype != cp.int8:
            raise BackendCapabilityError(
                f"{self._LABEL}的 dequantize_i8_group_zp 要求 q 与 zero_point 都是 i8，"
                f"收到 {q.dtype} 与 {zp.dtype}"
            )
        if scale.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 dequantize_i8_group_zp 的 scale 必须是 f32，"
                f"收到 {scale.dtype}"
            )
        if tuple(zp.shape) != tuple(scale.shape):
            raise BackendCapabilityError(
                f"{self._LABEL}的 dequantize_i8_group_zp 的 zero_point 必须与 scale "
                f"同形，收到 {zp.shape} 与 {scale.shape}"
            )
        cols, group = self._group_shape("dequantize_i8_group_zp", q, scale)
        qq = self._contig(q)
        ss = self._contig(scale)
        zz = self._contig(zp)
        out = cp.empty(qq.shape, np.float32)
        n = int(qq.size)
        if n == 0:
            return out
        kernel = self._kernel(
            ("dequantize_i8_group_zp",),
            _DEQUANTIZE_I8_GROUP_ZP_SRC,
            "af_dequantize_i8_group_zp",
        )
        block, grid = self._grid("dequantize_i8_group_zp", n)
        kernel(
            (grid,),
            (block,),
            (qq, ss, zz, out, np.int64(n), np.int64(cols), np.int64(group)),
        )
        return out

    def _op_fake_quantize_i8(self, node, args, attrs):
        """QAT 假量化：量化到 i8 再原样放大回 f32。

        不新增内核——直接复用 ``_op_quantize_i8`` 与 ``_op_dequantize_i8``。
        前向与"图上显式串联这两个算子"逐位相同（舍入与端点饱和只有一份
        实现）；独立成算子的唯一理由在反向：IR 给它挂了一条直通估计器
        （STE）规则，训练图才能穿过量化继续更新 f32 权重。
        """
        q = self._op_quantize_i8(node, [args[0], args[1]], attrs)
        return self._op_dequantize_i8(node, [q, args[1]], attrs)

    def _op_fake_quantize_i8_lsq(self, node, args, attrs):
        """LSQ 版前向：与 STE 版逐位相同，只有反向规则不同。

        保留独立方法而不是在注册表里指同一个函数，是为了让"声明即实现"的
        反射检查（内核方法名 ↔ ``CUDA_OPS``）继续成立；也把"设备端没有为
        可学步长新增内核"这件事写在代码里。
        """
        return self._op_fake_quantize_i8(node, args, attrs)

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

    def _op_linear_i8a8(self, node, args, attrs):
        """W8A8 线性层：i8 输入、**int32 累加**，最后才乘两路 scale。

        与 ``linear_i8`` 的分界：那边把权重反量化成 f32 再交给 cuBLAS，累加
        在浮点；这里累加全程 int32，只有缩放落到 f32。代价是 scale 必须提到
        求和号外面——``x_scale`` 每个 token 一个、``w_scale`` 每个输出通道
        一个，IR 已拒绝沿压缩轴变化的形态，所以设备端只需按行 / 按列取。

        逐位口径与参考实现一致：``(float)acc * xs * ws + bias`` 的乘加顺序
        与 numpy 的 ``acc.astype(f32) * row * ws (+ bias)`` 相同。int32 不会
        溢出，因为 IR 把 K 限制在 ``(2**31 - 1) / 127**2`` 以内。
        """
        cp = self.cp
        xq, xs, wq, ws = args[0], args[1], args[2], args[3]
        if xq.dtype != cp.int8 or wq.dtype != cp.int8:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_i8a8 要求激活与权重都是 i8，"
                f"收到 {xq.dtype} 与 {wq.dtype}"
            )
        if xs.dtype != cp.float32 or ws.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_i8a8 的 scale 必须是 f32，"
                f"收到 {xs.dtype} 与 {ws.dtype}"
            )
        if xq.ndim < 2 or wq.ndim != 2:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_i8a8 要求激活秩 >= 2、权重秩 2，"
                f"收到 {xq.shape} 与 {wq.shape}"
            )
        if (
            xs.ndim != xq.ndim
            or int(xs.shape[-1]) != 1
            or tuple(xs.shape[:-1]) != tuple(xq.shape[:-1])
        ):
            raise BackendCapabilityError(
                f"linear_i8a8 的激活 scale 形状必须是激活的前导维加 1，"
                f"收到 {xs.shape} 对 {xq.shape}"
            )
        k = int(xq.shape[-1])
        n = int(wq.shape[-1])
        if int(wq.shape[0]) != k:
            raise BackendCapabilityError(
                f"linear_i8a8 的内维不匹配：{xq.shape} @ {wq.shape}"
            )
        if int(ws.size) != n:
            raise BackendCapabilityError(
                f"linear_i8a8 的权重 scale 长度必须是 {n}，收到 {ws.shape}"
            )
        if len(args) > 4:
            bias = args[4]
            if bias.dtype != cp.float32 or int(bias.size) != n:
                raise BackendCapabilityError(
                    f"linear_i8a8 的 bias 必须是长度 {n} 的 f32，收到 "
                    f"{bias.dtype} {bias.shape}"
                )

        m = int(np.prod(xq.shape[:-1]))
        out = cp.empty((m, n), cp.float32)
        if out.size:
            x = self._contig(xq).reshape(m, k)
            w = self._contig(wq)
            sxa = self._contig(xs).reshape(m)
            swa = self._contig(ws).reshape(n)
            dot4 = self._I8A8_DOT4
            src = _LINEAR_I8A8_DP4A_SRC if dot4 else _LINEAR_I8A8_SRC
            kernel = self._kernel(("linear_i8a8", dot4), src, "af_linear_i8a8")
            block, grid = self._grid("linear_i8a8", m * n)
            kernel(
                (grid,),
                (block,),
                (
                    x,
                    sxa,
                    w,
                    swa,
                    out,
                    np.int64(m),
                    np.int64(k),
                    np.int64(n),
                ),
            )
        out = out.reshape(xq.shape[:-1] + (n,))
        if len(args) > 4:
            out = self._binary("add", [out, args[4]])
        return out

