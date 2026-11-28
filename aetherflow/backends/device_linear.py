"""线性代数与量化：matmul、i8 / fp8 量化与反量化、weight-only 与 W8A8 线性层。"""

from __future__ import annotations

from .base import BackendCapabilityError
import numpy as np
from .device_shared import (
    _DEQUANTIZE_FP8_GROUP_SRC,
    _DEQUANTIZE_I8_GROUP_SRC,
    _DEQUANTIZE_I8_GROUP_ZP_SRC,
    _DEQUANTIZE_I8_SRC,
    _DEQUANTIZE_I8_ZP_SRC,
    _DEQUANTIZE_U4_GROUP_SRC,
    _LINEAR_FP8A8_PACKED_SRC,
    _LINEAR_FP8_PACKED_SRC,
    _LINEAR_I8A8_DP4A_SRC,
    _LINEAR_I8A8_SRC,
    _LINEAR_I8A8_ZP_SRC,
    _PACK_FP8_SRC,
    _PACK_INT4_SRC,
    _QUANTIZE_I8_SRC,
    _QUANTIZE_I8_ZP_SRC,
    _UNPACK_FP8_SRC,
    _UNPACK_INT4_SRC,
)


class _LinearMixin:
    """线性代数与量化：matmul、i8 / fp8 量化与反量化、weight-only 与 W8A8 线性层。"""

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
        a, stride_a = self._batch_operand(a, batch_shape, m, k)
        b, stride_b = self._batch_operand(b, batch_shape, k, n)
        if f16:
            blas.hgemm_batched(a, b, out, batch, stride_a, stride_b, m * n)
        elif bf16:
            blas.bgemm_batched(a, b, out, batch, stride_a, stride_b, m * n)
        else:
            blas.sgemm_batched(a, b, out, batch, stride_a, stride_b, m * n)
        return out

    def _batch_operand(self, arr, out_shape: tuple, rows: int, cols: int):
        """批量 GEMM 的操作数：返回能用一个标量 stride 表达的数组与步长。

        形状与输出批量维一致 → 密集步长；整批只有 1 个 batch（包括没有批量维）
        → 步长 0，整批复用同一份数据。部分广播（例如 ``(2,1)`` 对 ``(2,3)``）
        无法用一个标量 stride 表达，cuBLAS 的 strided batched 也没有"逐 batch
        偏移表"这种入参，于是把广播视角物化成完整批次再走同一条批量路径：
        代价是 batch 份副本，正确性优先于省这份内存。
        """
        shape = tuple(arr.shape[:-2])
        dense = rows * cols
        if shape == out_shape:
            return arr, dense
        if not shape or all(d == 1 for d in shape):
            return arr, 0
        full = self.cp.broadcast_to(arr, out_shape + (rows, cols))
        return self.cp.ascontiguousarray(full), dense

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

    def _op_linear_i8_group(self, node, args, attrs):
        """逐组 weight-only int8 线性层：按组反量化权重后走既有 f32 GEMM。

        与 ``linear_i8`` 的唯一差别是权重侧 scale 的形状（``(K/G, N)`` 而不是
        ``(N,)``）；反量化仍在浮点里做，收益还是"权重占 1 字节"，不是算力。
        """
        cp = self.cp
        x, qweight, scale = args[0], args[1], args[2]
        if x.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_i8_group 要求 x 是 f32，收到 {x.dtype}"
            )
        if x.ndim < 2 or qweight.ndim != 2 or scale.ndim != 2:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_i8_group 要求 x 秩 >= 2、权重秩 2、"
                f"scale 秩 2，收到 {x.shape} / {qweight.shape} / {scale.shape}"
            )
        weight = self._op_dequantize_i8_group(node, [qweight, scale], attrs)
        out = self._op_matmul(node, [x, weight], {})
        if len(args) > 3:
            out = self._binary("add", [out, args[3]])
        return out

    def _op_pack_int4(self, node, args, attrs):
        """i8 承载的 4 bit 值 (..., N) -> (..., N/2) 个打包字节。

        只搬位、不做数值转换：偶数下标进低 4 位、奇数下标进高 4 位。每个线程
        产出一个字节，线程数取输出元素数（= 输入元素数 / 2）。
        """
        cp = self.cp
        q = args[0]
        if q.dtype != cp.int8:
            raise BackendCapabilityError(
                f"{self._LABEL}的 pack_int4 要求输入是 i8（承载 0..15），收到 {q.dtype}"
            )
        if int(q.shape[-1]) % 2 != 0:
            raise BackendCapabilityError(
                f"{self._LABEL}的 pack_int4 要求最后一维为偶数，收到 {q.shape}"
            )
        qq = self._contig(q)
        out = cp.empty(qq.shape[:-1] + (int(qq.shape[-1]) // 2,), np.int8)
        n = int(out.size)
        if n == 0:
            return out
        kernel = self._kernel(("pack_int4",), _PACK_INT4_SRC, "af_pack_int4")
        block, grid = self._grid("pack_int4", n)
        kernel((grid,), (block,), (qq, out, np.int64(n)))
        return out

    def _op_unpack_int4(self, node, args, attrs):
        """(..., N/2) 个打包字节 -> i8 承载的 4 bit 值 (..., N)。"""
        cp = self.cp
        p = args[0]
        if p.dtype != cp.int8:
            raise BackendCapabilityError(
                f"{self._LABEL}的 unpack_int4 要求输入是 i8 位模式，收到 {p.dtype}"
            )
        pp = self._contig(p)
        out = cp.empty(pp.shape[:-1] + (2 * int(pp.shape[-1]),), np.int8)
        n = int(out.size)
        if n == 0:
            return out
        kernel = self._kernel(("unpack_int4",), _UNPACK_INT4_SRC, "af_unpack_int4")
        block, grid = self._grid("unpack_int4", n)
        kernel((grid,), (block,), (pp, out, np.int64(n)))
        return out

    def _op_dequantize_u4_group(self, node, args, attrs):
        """逐组非对称 4 bit 反量化：``(q - zero_point) * scale``。

        q 是 4 bit 值（i8 承载，0..15），zero_point / scale 同形、都是 f32。
        减法留在 f32 里做：q 与 zp 都是精确整数，与参考实现的
        ``float(q) - zp`` 顺序逐位一致。非正 scale 与越界 zero_point 读回
        主机判定，与 fp8 逐组反量化同一口径。
        """
        cp = self.cp
        q, scale, zp = args
        if q.dtype != cp.int8:
            raise BackendCapabilityError(
                f"{self._LABEL}的 dequantize_u4_group 要求 q 是 i8（承载 0..15），"
                f"收到 {q.dtype}"
            )
        if scale.dtype != cp.float32 or zp.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 dequantize_u4_group 要求 scale 与 zero_point "
                f"都是 f32，收到 {scale.dtype} 与 {zp.dtype}"
            )
        if tuple(zp.shape) != tuple(scale.shape):
            raise BackendCapabilityError(
                f"{self._LABEL}的 dequantize_u4_group 的 zero_point 必须与 scale "
                f"同形，收到 {zp.shape} 与 {scale.shape}"
            )
        cols, group = self._group_shape("dequantize_u4_group", q, scale)
        if float(np.min(self.to_host(scale))) <= 0.0:
            raise BackendCapabilityError(
                "dequantize_u4_group 的 scale 必须严格为正；"
                "非正 scale 的反量化为未定义"
            )
        zp_host = self.to_host(zp)
        if float(np.min(zp_host)) < 0.0 or float(np.max(zp_host)) > 15.0:
            raise BackendCapabilityError(
                f"dequantize_u4_group 的 zero_point 必须在 [0, 15] 内，收到 "
                f"[{float(np.min(zp_host))}, {float(np.max(zp_host))}]"
            )
        qq = self._contig(q)
        ss = self._contig(scale)
        zz = self._contig(zp)
        out = cp.empty(qq.shape, np.float32)
        n = int(qq.size)
        if n == 0:
            return out
        kernel = self._kernel(
            ("dequantize_u4_group",), _DEQUANTIZE_U4_GROUP_SRC, "af_dequantize_u4_group"
        )
        block, grid = self._grid("dequantize_u4_group", n)
        kernel(
            (grid,),
            (block,),
            (qq, ss, zz, out, np.int64(n), np.int64(cols), np.int64(group)),
        )
        return out

    def _op_linear_u4_group(self, node, args, attrs):
        """逐组 weight-only 4 bit 线性层：按组反量化权重后走既有 f32 GEMM。

        与 ``linear_i8_group`` 同构，只多了 zero_point、权重只有 16 个格点。
        反量化会物化一份 f32 权重副本，收益是"权重驻留 0.5 字节 + 更细的
        量化粒度"，**不是**算力：int4 tensor core 不在本版声称范围内。
        """
        cp = self.cp
        x, qweight, scale, zp = args[0], args[1], args[2], args[3]
        if x.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_u4_group 要求 x 是 f32，收到 {x.dtype}"
            )
        if qweight.dtype != cp.int8:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_u4_group 要求权重是 i8（承载 0..15），"
                f"收到 {qweight.dtype}"
            )
        if scale.dtype != cp.float32 or zp.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_u4_group 要求 scale 与 zero_point "
                f"都是 f32，收到 {scale.dtype} 与 {zp.dtype}"
            )
        if x.ndim < 2 or qweight.ndim != 2 or scale.ndim != 2:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_u4_group 要求 x 秩 >= 2、权重秩 2、"
                f"scale 秩 2，收到 {x.shape} / {qweight.shape} / {scale.shape}"
            )
        if int(qweight.shape[0]) != int(x.shape[-1]):
            raise BackendCapabilityError(
                f"linear_u4_group 的内维不匹配：{x.shape} @ {qweight.shape}"
            )
        if tuple(zp.shape) != tuple(scale.shape):
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_u4_group 的 zero_point 必须与 scale 同形，"
                f"收到 {zp.shape} 与 {scale.shape}"
            )
        weight = self._op_dequantize_u4_group(node, [qweight, scale, zp], attrs)
        out = self._op_matmul(node, [x, weight], {})
        if len(args) > 4:
            bias = args[4]
            if bias.dtype != cp.float32 or int(bias.size) != int(qweight.shape[-1]):
                raise BackendCapabilityError(
                    f"linear_u4_group 的 bias 必须是长度 {int(qweight.shape[-1])} 的 f32，"
                    f"收到 {bias.dtype} {bias.shape}"
                )
            out = self._binary("add", [out, bias])
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

    def _op_linear_fp8(self, node, args, attrs):
        """weight-only fp8 线性层：反量化权重后走既有 f32 GEMM。

        fp8 在设备上的载荷就是 f32 载体（数值已在写入侧落到格式格点上），所以
        反量化只是一次逐元素乘，既不解码位模式、也不需要新内核：乘完直接交给
        cuBLAS / hipBLAS 的 sgemm。收益只在"权重占用与读带宽"（真正的 1 字节
        存储见 5.328），不是算力——fp8 tensor core 需要另一套缩放与累加语义，
        不在本版声称范围内。

        设备端不复查"值是否在格点上"：那需要把权重读回主机，与"零同步"的
        设备路径直接冲突；格点校验由写入侧（``weights.load_fp8_linear``）负责。
        """
        cp = self.cp
        x, qweight, scale = args[0], args[1], args[2]
        if x.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_fp8 要求 x 是 f32，收到 {x.dtype}"
            )
        if qweight.dtype != cp.float32 or scale.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_fp8 要求权重（f32 载体）与 scale 都是 f32，"
                f"收到 {qweight.dtype} 与 {scale.dtype}"
            )
        if x.ndim < 2 or qweight.ndim != 2:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_fp8 要求 x 秩 >= 2、权重秩 2，"
                f"收到 {x.shape} 与 {qweight.shape}"
            )
        weight = self._binary("mul", [qweight, scale])
        out = self._op_matmul(node, [x, weight], {})
        if len(args) > 3:
            out = self._binary("add", [out, args[3]])
        return out

    def _op_linear_fp8a8(self, node, args, attrs):
        """W8A8 fp8 线性层：两路载荷各按自己的 scale 反量化，再走 f32 GEMM。

        与 ``linear_i8a8`` 的分界：那边累加在 int32 里、scale 必须提到求和号
        外面，因此 IR 限定"沿 K 恒定"；这里反量化先落成 f32 张量，累加本来
        就在浮点，所以设备端不需要对 scale 形状做额外假设——只要它按 numpy
        规则广播到载荷即可（IR 已把形态收窄成逐 token / 逐输出通道）。

        没有任何 fp8 专用内核，也不承诺 fp8 tensor core：收益同 ``linear_fp8``，
        仍只在驻留与读带宽（1 字节存储见 5.328）。
        """
        cp = self.cp
        xq, xs, wq, ws = args[0], args[1], args[2], args[3]
        for label, arr in (("激活", xq), ("激活 scale", xs), ("权重", wq), ("权重 scale", ws)):
            if arr.dtype != cp.float32:
                raise BackendCapabilityError(
                    f"{self._LABEL}的 linear_fp8a8 要求{label}是 f32"
                    f"（fp8 的载荷也是 f32 载体），收到 {arr.dtype}"
                )
        if xq.ndim < 2 or wq.ndim != 2:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_fp8a8 要求激活秩 >= 2、权重秩 2，"
                f"收到 {xq.shape} 与 {wq.shape}"
            )
        k = int(xq.shape[-1])
        n = int(wq.shape[-1])
        if int(wq.shape[0]) != k:
            raise BackendCapabilityError(
                f"linear_fp8a8 的内维不匹配：{xq.shape} @ {wq.shape}"
            )
        if len(args) > 4:
            bias = args[4]
            if bias.dtype != cp.float32 or int(bias.size) != n:
                raise BackendCapabilityError(
                    f"linear_fp8a8 的 bias 必须是长度 {n} 的 f32，收到 "
                    f"{bias.dtype} {bias.shape}"
                )
        x = self._binary("mul", [xq, xs])
        weight = self._binary("mul", [wq, ws])
        out = self._op_matmul(node, [x, weight], {})
        if len(args) > 4:
            out = self._binary("add", [out, args[4]])
        return out

    # ------------------------------------------------------------------
    # fp8 的 1 字节存储（5.328）
    # ------------------------------------------------------------------

    def _fp8_kind(self, attrs, name: str) -> int:
        """attr ``kind`` -> 内核整数码。与 ``cast`` 同口径：1 = e4m3，2 = e5m2。"""
        kind = attrs.get("kind")
        if kind == "e4m3":
            return 1
        if kind == "e5m2":
            return 2
        raise BackendCapabilityError(
            f"{self._LABEL}的 {name} 需要 attr kind='e4m3' 或 'e5m2'，收到 {kind!r}"
        )

    def _op_pack_fp8(self, node, args, attrs):
        """f8 载体 -> i8 位模式：每元素 1 字节，数值与 ``cast`` 完全相同。

        收益是驻留与读带宽（4 字节降到 1 字节），不是算力。参考语义按位
        转成 uint8，所以这里不经手任何数值转换，只把编码落成 8 位存储。
        """
        cp = self.cp
        x = args[0]
        if x.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 pack_fp8 要求输入是 f32 载体，收到 {x.dtype}"
            )
        kind = self._fp8_kind(attrs, "pack_fp8")
        xx = self._contig(x)
        out = cp.empty(xx.shape, np.int8)
        n = int(xx.size)
        if n == 0:
            return out
        kernel = self._kernel(("pack_fp8", kind), _PACK_FP8_SRC, "af_pack_fp8")
        block, grid = self._grid("pack_fp8", n)
        kernel((grid,), (block,), (xx, out, np.int64(n), np.int32(kind)))
        return out

    def _op_unpack_fp8(self, node, args, attrs):
        """i8 位模式 -> f8 载体。逐位精确（含 ±inf / NaN 与次正规）。"""
        cp = self.cp
        q = args[0]
        if q.dtype != cp.int8:
            raise BackendCapabilityError(
                f"{self._LABEL}的 unpack_fp8 要求输入是 i8 位模式，收到 {q.dtype}"
            )
        kind = self._fp8_kind(attrs, "unpack_fp8")
        qq = self._contig(q)
        out = cp.empty(qq.shape, cp.float32)
        n = int(qq.size)
        if n == 0:
            return out
        kernel = self._kernel(("unpack_fp8", kind), _UNPACK_FP8_SRC, "af_unpack_fp8")
        block, grid = self._grid("unpack_fp8", n)
        kernel((grid,), (block,), (qq, out, np.int64(n), np.int32(kind)))
        return out

    def _op_linear_fp8_packed(self, node, args, attrs):
        """weight-only fp8 线性层，权重以 1 字节位模式驻留。

        与 ``linear_fp8`` 的数值语义相同（反量化后 f32 累加），区别在读法：
        这里不把权重反量化成 f32 张量，而是让内核按 ``unsigned char`` 读入、
        在寄存器里解码，因此权重的显存占用与读流量都是 1/4。代价是放弃
        cuBLAS / hipBLAS（``_LINEAR_FP8_PACKED_SRC`` 是朴素每线程一个输出
        元素、无分块），**交付的是存储形态而不是速度**；fp8 tensor core 仍
        不在本版范围内。

        scale 只接逐张量 ``(1,)`` 与逐输出通道 ``(N,)``：内核按列取 scale，
        逐组 / 逐元素 scale 需要把 K 方向的折返也带进内核（IR 已拒绝，留待
        后续）；非正 scale 与其它量化算子同一口径（读回主机判定）。
        """
        cp = self.cp
        x, qweight, scale = args[0], args[1], args[2]
        if x.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_fp8_packed 要求 x 是 f32，收到 {x.dtype}"
            )
        if qweight.dtype != cp.int8:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_fp8_packed 要求权重是 i8 位模式"
                f"（用 pack_fp8 生成），收到 {qweight.dtype}；"
                "已解码的 f8 载体请用 linear_fp8"
            )
        if scale.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_fp8_packed 要求 scale 是 f32，收到 {scale.dtype}"
            )
        if x.ndim < 2 or qweight.ndim != 2:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_fp8_packed 要求 x 秩 >= 2、权重秩 2，"
                f"收到 {x.shape} 与 {qweight.shape}"
            )
        k = int(x.shape[-1])
        n = int(qweight.shape[-1])
        if int(qweight.shape[0]) != k:
            raise BackendCapabilityError(
                f"linear_fp8_packed 的内维不匹配：{x.shape} @ {qweight.shape}"
            )
        if scale.ndim > 1 or int(scale.size) not in (1, n):
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_fp8_packed 的 scale 只能是逐张量 (1,) 或"
                f"逐输出通道 ({n},)，收到 {scale.shape}"
            )
        if float(np.min(self.to_host(scale))) <= 0.0:
            raise BackendCapabilityError(
                "linear_fp8_packed 的 scale 必须严格为正；非正 scale 的反量化为未定义"
            )
        if len(args) > 3:
            bias = args[3]
            if bias.dtype != cp.float32 or int(bias.size) != n:
                raise BackendCapabilityError(
                    f"linear_fp8_packed 的 bias 必须是长度 {n} 的 f32，收到 "
                    f"{bias.dtype} {bias.shape}"
                )
        kind = self._fp8_kind(attrs, "linear_fp8_packed")
        xx = self._contig(x)
        w = self._contig(qweight)
        s = self._contig(scale)
        m = int(np.prod(xx.shape[:-1]))
        out = cp.empty(tuple(xx.shape[:-1]) + (n,), cp.float32)
        if m * n == 0:
            return out
        ws_stride = 0 if int(scale.size) == 1 else 1
        kernel = self._kernel(
            ("linear_fp8_packed", kind), _LINEAR_FP8_PACKED_SRC, "af_linear_fp8_packed"
        )
        block, grid = self._grid("linear_fp8_packed", m * n)
        kernel(
            (grid,),
            (block,),
            (
                xx,
                w,
                s,
                out,
                np.int64(m),
                np.int64(k),
                np.int64(n),
                np.int64(ws_stride),
                np.int32(kind),
            ),
        )
        if len(args) > 3:
            out = self._binary("add", [out, args[3]])
        return out

    def _op_dequantize_fp8_group(self, node, args, attrs):
        """逐组 fp8 反量化：i8 位模式 (K, N)，scale (K/G, N) 组内共用。

        与 ``dequantize_i8_group`` 同形：组大小由 ``K / scale.shape[0]`` 推出。
        区别只在读出后要先按 fp8 格点解码，乘法顺序与参考实现的
        ``unpack_fp8(q) * repeat(scale)`` 相同，因此逐位一致。
        """
        cp = self.cp
        q, scale = args
        if q.dtype != cp.int8:
            raise BackendCapabilityError(
                f"{self._LABEL}的 dequantize_fp8_group 第一个输入必须是 i8 位模式"
                f"（用 pack_fp8 生成），收到 {q.dtype}"
            )
        if scale.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 dequantize_fp8_group 的 scale 必须是 f32，"
                f"收到 {scale.dtype}"
            )
        cols, group = self._group_shape("dequantize_fp8_group", q, scale)
        if float(np.min(self.to_host(scale))) <= 0.0:
            raise BackendCapabilityError(
                "dequantize_fp8_group 的 scale 必须严格为正；"
                "非正 scale 的反量化为未定义"
            )
        kind = self._fp8_kind(attrs, "dequantize_fp8_group")
        qq = self._contig(q)
        ss = self._contig(scale)
        out = cp.empty(qq.shape, np.float32)
        n = int(qq.size)
        if n == 0:
            return out
        kernel = self._kernel(
            ("dequantize_fp8_group", kind),
            _DEQUANTIZE_FP8_GROUP_SRC,
            "af_dequantize_fp8_group",
        )
        block, grid = self._grid("dequantize_fp8_group", n)
        kernel(
            (grid,),
            (block,),
            (
                qq,
                ss,
                out,
                np.int64(n),
                np.int64(cols),
                np.int64(group),
                np.int32(kind),
            ),
        )
        return out

    def _op_linear_fp8_group(self, node, args, attrs):
        """逐组 weight-only fp8 线性层：反量化后走既有 f32 GEMM。

        与 ``linear_fp8_packed`` 的分界是 scale 的形状：那边只有逐张量 /
        逐输出通道，这里沿 K 每 ``K/G`` 行共用一份（GPTQ / AWQ 的 group_size
        形态）。做法与 ``linear_i8_group`` 同构——反量化会物化一份 f32 权重
        副本，收益仍是"权重驻留 1 字节 + 更细的量化粒度"，不是算力。
        """
        cp = self.cp
        x, qweight, scale = args[0], args[1], args[2]
        if x.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_fp8_group 要求 x 是 f32，收到 {x.dtype}"
            )
        if qweight.dtype != cp.int8:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_fp8_group 要求权重是 i8 位模式"
                f"（用 pack_fp8 生成），收到 {qweight.dtype}"
            )
        if scale.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_fp8_group 的 scale 必须是 f32，"
                f"收到 {scale.dtype}"
            )
        if x.ndim < 2 or qweight.ndim != 2 or scale.ndim != 2:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_fp8_group 要求 x 秩 >= 2、权重秩 2、"
                f"scale 秩 2，收到 {x.shape} / {qweight.shape} / {scale.shape}"
            )
        if int(qweight.shape[0]) != int(x.shape[-1]):
            raise BackendCapabilityError(
                f"linear_fp8_group 的内维不匹配：{x.shape} @ {qweight.shape}"
            )
        n = int(qweight.shape[-1])
        if len(args) > 3:
            bias = args[3]
            if bias.dtype != cp.float32 or int(bias.size) != n:
                raise BackendCapabilityError(
                    f"linear_fp8_group 的 bias 必须是长度 {n} 的 f32，收到 "
                    f"{bias.dtype} {bias.shape}"
                )
        weight = self._op_dequantize_fp8_group(node, [qweight, scale], attrs)
        out = self._op_matmul(node, [x, weight], {})
        if len(args) > 3:
            out = self._binary("add", [out, args[3]])
        return out

    def _op_linear_fp8a8_packed(self, node, args, attrs):
        """两端都打包的 W8A8：激活与权重各以 1 字节码驻留，寄存器里解码。

        与 ``linear_fp8a8`` 的数值语义相同（f32 累加、按行 / 按列缩放），
        区别是激活也不物化 f32 载体：内核按 ``unsigned char`` 读入并解码。
        这正是 5.329 遗留的 "packed 激活"，驻留从 4 字节降到 1 字节。

        累加仍是 f32（没有 fp8 tensor core），所以不宣称算力。激活 scale
        允许标量或 ``x.shape[:-1] + (1,)``，与 ``linear_fp8a8`` 同口径；
        ``xs_stride = 0`` 表示逐张量。
        """
        cp = self.cp
        xq, xs, wq, ws = args[0], args[1], args[2], args[3]
        if xq.dtype != cp.int8 or wq.dtype != cp.int8:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_fp8a8_packed 要求激活与权重都是 i8 位模式，"
                f"收到 {xq.dtype} 与 {wq.dtype}"
            )
        if xs.dtype != cp.float32 or ws.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_fp8a8_packed 的 scale 必须是 f32，"
                f"收到 {xs.dtype} 与 {ws.dtype}"
            )
        if xq.ndim < 2 or wq.ndim != 2:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_fp8a8_packed 要求激活秩 >= 2、权重秩 2，"
                f"收到 {xq.shape} 与 {wq.shape}"
            )
        if int(wq.shape[0]) != int(xq.shape[-1]):
            raise BackendCapabilityError(
                f"linear_fp8a8_packed 的内维不匹配：{xq.shape} @ {wq.shape}"
            )
        n = int(wq.shape[-1])
        if xs.ndim == 0:
            if int(xs.size) != 1:
                raise BackendCapabilityError(
                    f"linear_fp8a8_packed 的标量激活 scale 只能有 1 个元素，"
                    f"收到 {xs.shape}"
                )
        elif (
            int(xs.shape[-1]) != 1
            or tuple(xs.shape[:-1]) != tuple(xq.shape[:-1])
        ):
            raise BackendCapabilityError(
                f"linear_fp8a8_packed 的激活 scale 形状必须是激活的前导维加 1，"
                f"收到 {xs.shape} 对 {xq.shape}"
            )
        if ws.ndim > 1 or int(ws.size) not in (1, n):
            raise BackendCapabilityError(
                f"linear_fp8a8_packed 的权重 scale 只能是逐张量标量或"
                f"逐输出通道 ({n},)，收到 {ws.shape}"
            )
        if float(np.min(self.to_host(xs))) <= 0.0:
            raise BackendCapabilityError(
                "linear_fp8a8_packed 的激活 scale 必须严格为正；"
                "非正 scale 的反量化为未定义"
            )
        if float(np.min(self.to_host(ws))) <= 0.0:
            raise BackendCapabilityError(
                "linear_fp8a8_packed 的权重 scale 必须严格为正；"
                "非正 scale 的反量化为未定义"
            )
        if len(args) > 4:
            bias = args[4]
            if bias.dtype != cp.float32 or int(bias.size) != n:
                raise BackendCapabilityError(
                    f"linear_fp8a8_packed 的 bias 必须是长度 {n} 的 f32，收到 "
                    f"{bias.dtype} {bias.shape}"
                )
        kind = self._fp8_kind(attrs, "linear_fp8a8_packed")
        xx = self._contig(xq).reshape(-1, int(xq.shape[-1]))
        ww = self._contig(wq)
        xs_stride = 0 if int(xs.size) == 1 else 1
        xs_flat = self._contig(xs).reshape(-1)
        ws_stride = 0 if int(ws.size) == 1 else 1
        ws_flat = self._contig(ws).reshape(-1)
        m = int(xx.shape[0])
        # 输出按激活的前导维 + N 分配，而不是 (m, n)：IR 对秩 >= 3 的激活推导
        # 出 (..., N)，执行器按这个形状校验写回；内核只按线性索引写 m*n 个数，
        # 形状本身不影响它。
        out = cp.empty(tuple(xq.shape[:-1]) + (n,), cp.float32)
        if m * n == 0:
            return out
        flat = out.reshape(m, n)
        kernel = self._kernel(
            ("linear_fp8a8_packed", kind),
            _LINEAR_FP8A8_PACKED_SRC,
            "af_linear_fp8a8_packed",
        )
        block, grid = self._grid("linear_fp8a8_packed", m * n)
        kernel(
            (grid,),
            (block,),
            (
                xx,
                xs_flat,
                ww,
                ws_flat,
                flat,
                np.int64(m),
                np.int64(int(xq.shape[-1])),
                np.int64(n),
                np.int64(xs_stride),
                np.int64(ws_stride),
                np.int32(kind),
            ),
        )
        if len(args) > 4:
            out = self._binary("add", [out, args[4]])
        return out

    def _op_linear_i8_zp(self, node, args, attrs):
        """非对称 weight-only int8 线性层：先反量化权重，再走既有 f32 GEMM。

        与 ``linear_i8`` 的唯一差别是读出权重时多减一个零点（5.322 的
        ``dequantize_i8_zp``）；GEMM 与 bias 完全复用。因此收益仍只在
        "i8 驻留与读带宽"，不宣称算力提升。
        """
        cp = self.cp
        x, qweight, scale, zp = args[0], args[1], args[2], args[3]
        if x.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_i8_zp 要求 x 是 f32，收到 {x.dtype}"
            )
        if x.ndim < 2 or qweight.ndim != 2:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_i8_zp 要求 x 秩 >= 2、权重秩 2，"
                f"收到 {x.shape} 与 {qweight.shape}"
            )
        weight = self._op_dequantize_i8_zp(node, [qweight, scale, zp], {})
        out = self._op_matmul(node, [x, weight], {})
        if len(args) > 4:
            out = self._binary("add", [out, args[4]])
        return out

    def _op_linear_i8a8_zp(self, node, args, attrs):
        """非对称 W8A8：两侧各减零点后 int32 累加，最后乘两路 scale。

        逐位口径与参考实现一致：``(float)acc * xs * ws (+ bias)``，acc 的每一项
        都是 ``(xq - xzp) * (wq - wzp)``。不能先按对称版算完 acc 再在 f32 里补
        偏置——展开后的 ``-zx*colsum - zw*rowsum + K*zx*zw`` 与逐项相减的整数
        累加不等价（5.315 的语义）。int32 不溢出由 IR 的 ``K * 255 * 255``
        上界保证。

        零点减法会打散字节对齐，所以这里不走 ``__dp4a``，用标量乘加；两条
        后端共用同一份源码，结果逐位相同。bias 直接融进内核的 epilogue
        （``af_linear_i8a8_zp_bias``），不再单独发一次 ``add``：省掉一次
        launch 与一块 ``m*n`` 的中间张量。融合用的是 ``__fmul_rn`` /
        ``__fadd_rn``，与"分开两次内核各自 RN 舍入"逐位一致。

        dp4a / IMMA 仍然欠着：零点相减后的差值不是 int8，``__dp4a`` 的
        4 路点积吃不下；要复用整数点积只能做下面这条展开——

            sum((x - zx)*(w - zw)) = P - zx*Sw - zw*Sx + K*zx*zw

        其中 ``Sw`` / ``Sx`` 是逐行 / 逐列整型和。展开在数学上与逐项相减
        等价，但需要额外汇总轴并在跨后端验证整数溢出边界（``P`` 的上界
        与 ``K*255*255`` 同量级），所以留到有真机可验证时再做。
        """
        cp = self.cp
        xq, xs, xzp = args[0], args[1], args[2]
        wq, ws, wzp = args[3], args[4], args[5]
        if xq.dtype != cp.int8 or wq.dtype != cp.int8:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_i8a8_zp 要求激活与权重都是 i8，"
                f"收到 {xq.dtype} 与 {wq.dtype}"
            )
        if xzp.dtype != cp.int8 or wzp.dtype != cp.int8:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_i8a8_zp 的 zero_point 必须是 i8，"
                f"收到 {xzp.dtype} 与 {wzp.dtype}"
            )
        if xs.dtype != cp.float32 or ws.dtype != cp.float32:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_i8a8_zp 的 scale 必须是 f32，"
                f"收到 {xs.dtype} 与 {ws.dtype}"
            )
        if xq.ndim < 2 or wq.ndim != 2:
            raise BackendCapabilityError(
                f"{self._LABEL}的 linear_i8a8_zp 要求激活秩 >= 2、权重秩 2，"
                f"收到 {xq.shape} 与 {wq.shape}"
            )
        rows = tuple(xq.shape[:-1]) + (1,)
        if tuple(xs.shape) != rows or tuple(xzp.shape) != rows:
            raise BackendCapabilityError(
                f"linear_i8a8_zp 的激活 scale / zero_point 形状必须是激活的"
                f"前导维加 1，收到 {xs.shape} / {xzp.shape} 对 {xq.shape}"
            )
        k = int(xq.shape[-1])
        n = int(wq.shape[-1])
        if int(wq.shape[0]) != k:
            raise BackendCapabilityError(
                f"linear_i8a8_zp 的内维不匹配：{xq.shape} @ {wq.shape}"
            )
        if int(ws.size) != n or int(wzp.size) != n:
            raise BackendCapabilityError(
                f"linear_i8a8_zp 的权重 scale / zero_point 长度必须是 {n}，"
                f"收到 {ws.shape} / {wzp.shape}"
            )
        if len(args) > 6:
            bias = args[6]
            if bias.dtype != cp.float32 or int(bias.size) != n:
                raise BackendCapabilityError(
                    f"linear_i8a8_zp 的 bias 必须是长度 {n} 的 f32，收到 "
                    f"{bias.dtype} {bias.shape}"
                )

        m = int(np.prod(xq.shape[:-1]))
        out = cp.empty((m, n), cp.float32)
        has_bias = len(args) > 6
        if out.size:
            x = self._contig(xq).reshape(m, k)
            w = self._contig(wq)
            zxa = self._contig(xzp).reshape(m)
            sxa = self._contig(xs).reshape(m)
            zwa = self._contig(wzp).reshape(n)
            swa = self._contig(ws).reshape(n)
            kernel = self._kernel(
                ("linear_i8a8_zp", has_bias),
                _LINEAR_I8A8_ZP_SRC,
                "af_linear_i8a8_zp_bias" if has_bias else "af_linear_i8a8_zp",
            )
            block, grid = self._grid("linear_i8a8_zp", m * n)
            operands = [x, zxa, sxa, w, zwa, swa]
            if has_bias:
                operands.append(self._contig(args[6]).reshape(n))
            operands.extend((out, np.int64(m), np.int64(k), np.int64(n)))
            kernel((grid,), (block,), tuple(operands))
        out = out.reshape(xq.shape[:-1] + (n,))
        return out

