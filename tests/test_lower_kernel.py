"""L2 内核 IR 的验收。

判据不是"看起来对"：同一份 L2 描述打印出的源码，必须与手写模板在**去掉空白
与注释后逐字符相等**。括号在 L2 里是显式的（``Paren`` 节点），所以这条比较是
严格的——打印器一旦悄悄改变结合方式，测试立刻失败。

第二个判据是方言差异的边界：f32 不涉及 f16 存取，CUDA 与 HIP 必须一字不差；
f16 只允许 prelude 不同，主体必须相同。
"""

import re
import unittest

import numpy as np

from aetherflow.backends.cuda import codegen as cuda_codegen
from aetherflow.backends.cuda.dialect import CUDA_DIALECT
from aetherflow.backends.hip import codegen as hip_codegen
from aetherflow.backends.hip.dialect import HIP_DIALECT
from aetherflow.ir.types import DType
from aetherflow.lower import (
    adaptive_avg_pool2d,
    adaptive_avg_pool2d_backward,
    attention_fwd,
    avgpool2d,
    avgpool2d_backward,
    batch_norm,
    batch_norm_stats,
    build_chain,
    concat_copy,
    conv2d_input_grad,
    conv2d_grouped,
    conv2d_tiled,
    conv2d_weight_grad,
    dropout_mask,
    ElemInput,
    ElemStep,
    emit,
    ew_binary,
    ew_broadcast,
    ew_step,
    ew_unary,
    im2col,
    layernorm_last,
    layernorm_residual_last,
    maxpool2d,
    maxpool2d_indices,
    maxpool2d_backward,
    argmax_last,
    reduce_max_last,
    reduce_mean_last,
    reduce_sum_last,
    rmsnorm_last,
    scan_linear,
    scan_linear_backward,
    scan_linear_chunk,
    scan_linear_chunk_backward,
    scan_linear_chunk_last,
    scan_linear_chunk_sum,
    scan_linear_chunk_sum_backward,
    scan_linear_last,
    silu_gate,
    softmax_last,
)

# 生产路径的 codegen 按 numpy dtype 认类型（它面对的是运行期张量），
# L2 按 IR 的 DType 认类型。两边在这里各用各的，比较的是产出的源码。
_F32 = np.dtype("float32")
_F16 = np.dtype("float16")


def _squash(src: str) -> str:
    """去掉注释与所有空白，只留 token 序列。"""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"//[^\n]*", "", src)
    return re.sub(r"\s+", "", src)


def _kernel_part(src: str) -> str:
    """从内核签名开始的部分（跳过 prelude）。"""
    return src[src.index('extern "C"') :]


def _named_kernel(src: str, name: str) -> str:
    """从一份可能含多个内核的源码里截出指定 kernel（含签名与函数体）。

    ``BATCH_NORM_SRC`` 这一份模板里有 ``batch_norm`` 与 ``batch_norm_stats``
    两个内核；L2 把它们拆成两个 ``Kernel``，对照时按名字各取一段。
    """
    start = src.index(f'extern "C" __global__ void {name}(')
    depth = 0
    for offset, ch in enumerate(src[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return src[start : offset + 1]
    raise AssertionError(f"源码里 {name} 的花括号不配对")


class TestEwBinary(unittest.TestCase):
    def test_matches_handwritten_template_f32(self):
        expected = cuda_codegen.render(cuda_codegen.binary_template("add"), _F32)
        actual = emit(ew_binary("add"), CUDA_DIALECT, DType.F32)
        self.assertEqual(_squash(actual), _squash(expected))

    def test_matches_handwritten_template_f16(self):
        expected = cuda_codegen.render(cuda_codegen.binary_template("mul"), _F16)
        actual = emit(ew_binary("mul"), CUDA_DIALECT, DType.F16)
        self.assertEqual(_squash(actual), _squash(expected))

    def test_all_four_operators(self):
        for kind in ("add", "sub", "mul", "div"):
            expected = cuda_codegen.render(cuda_codegen.binary_template(kind), _F32)
            actual = emit(ew_binary(kind), CUDA_DIALECT, DType.F32)
            self.assertEqual(_squash(actual), _squash(expected), kind)

    def test_scalar_side_reads_index_zero(self):
        src = emit(ew_binary("mul", scalar_b=True), CUDA_DIALECT, DType.F32)
        self.assertIn("af_ld_f(b[0])", src)
        self.assertIn("af_ld_f(a[i])", src)

    def test_unknown_operator_is_refused(self):
        with self.assertRaises(ValueError):
            ew_binary("pow")


class TestReduceSum(unittest.TestCase):
    def test_matches_handwritten_template_f32(self):
        expected = cuda_codegen.render(cuda_codegen.REDUCE_SUM_SRC, _F32)
        actual = emit(reduce_sum_last(), CUDA_DIALECT, DType.F32)
        self.assertEqual(_squash(actual), _squash(expected))

    def test_matches_handwritten_template_f16(self):
        expected = cuda_codegen.render(cuda_codegen.REDUCE_SUM_SRC, _F16)
        actual = emit(reduce_sum_last(), CUDA_DIALECT, DType.F16)
        self.assertEqual(_squash(actual), _squash(expected))


class TestReduceFamily(unittest.TestCase):
    """max / mean / argmax 与 sum 共用骨架，但各自的模板必须独立对齐。"""

    def _check(self, kernel, template, np_dtype, ir_dtype):
        # 模板用 numpy dtype 渲染（codegen 面对运行期张量），L2 用 IR 的 DType。
        expected = cuda_codegen.render(template, np_dtype)
        actual = emit(kernel, CUDA_DIALECT, ir_dtype)
        self.assertEqual(_squash(actual), _squash(expected))

    def test_reduce_max_f32(self):
        self._check(reduce_max_last(), cuda_codegen.REDUCE_MAX_SRC, _F32, DType.F32)

    def test_reduce_max_f16(self):
        self._check(reduce_max_last(), cuda_codegen.REDUCE_MAX_SRC, _F16, DType.F16)

    def test_reduce_mean_f32(self):
        self._check(reduce_mean_last(), cuda_codegen.REDUCE_MEAN_SRC, _F32, DType.F32)

    def test_reduce_mean_f16(self):
        self._check(reduce_mean_last(), cuda_codegen.REDUCE_MEAN_SRC, _F16, DType.F16)

    def test_argmax_f32(self):
        self._check(argmax_last(), cuda_codegen.ARGMAX_SRC, _F32, DType.F32)

    def test_argmax_f16(self):
        self._check(argmax_last(), cuda_codegen.ARGMAX_SRC, _F16, DType.F16)

    def test_argmax_output_is_int_regardless_of_dtype(self):
        """下标不是浮点激活值：两种 dtype 下 out 参数都必须是 int*。"""
        for dtype in (DType.F32, DType.F16):
            src = emit(argmax_last(), CUDA_DIALECT, dtype)
            self.assertIn("int* __restrict__ out", src)


class TestReduceAxisTiled(unittest.TestCase):
    """非末维归约的两维铺法（5.174）：沿 ``threadIdx.y`` 归约，两种方言同源。"""

    def test_sum_uses_the_y_reduction_helper(self):
        src = cuda_codegen.render(cuda_codegen.reduce_axis_tiled_source("sum"), _F32)
        self.assertIn("block_reduce_sum_y(acc, smem)", src)
        self.assertIn(
            "for (long long r = (long long)threadIdx.y; r < reduce_len; "
            "r += blockDim.y)",
            src,
        )
        self.assertIn(
            "(long long)blockIdx.y * nx + (long long)threadIdx.x", src
        )

    def test_each_kind_picks_its_helper(self):
        cases = {
            "sum": "block_reduce_sum_y",
            "mean": "block_reduce_sum_y",
            "max": "block_reduce_max_y",
            "min": "block_reduce_min_y",
        }
        for kind, fn in cases.items():
            with self.subTest(kind=kind):
                src = cuda_codegen.render(
                    cuda_codegen.reduce_axis_tiled_source(kind), _F32
                )
                self.assertIn(f"{fn}(acc, smem)", src)

    def test_float16_only_rounds_on_store(self):
        src = cuda_codegen.render(cuda_codegen.reduce_axis_tiled_source("sum"), _F16)
        self.assertIn("out[o * inner + i] = af_st_h(acc)", src)

    def test_hip_reuses_the_same_kernel(self):
        src = hip_codegen.render(cuda_codegen.reduce_axis_tiled_source("sum"), _F16)
        self.assertIn("block_reduce_sum_y", src)
        self.assertIn("_Float16", src)

    def test_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            cuda_codegen.reduce_axis_tiled_source("prod")


class TestDialectsDifferOnlyWhereExpected(unittest.TestCase):
    def test_f32_is_identical_across_vendors(self):
        """f32 不碰 f16 方言，两家生成的源码必须一字不差。"""
        self.assertEqual(
            emit(ew_binary("add"), CUDA_DIALECT, DType.F32),
            emit(ew_binary("add"), HIP_DIALECT, DType.F32),
        )

    def test_f16_differs_only_in_prelude(self):
        src_cuda = emit(ew_binary("add"), CUDA_DIALECT, DType.F16)
        src_hip = emit(ew_binary("add"), HIP_DIALECT, DType.F16)
        self.assertNotEqual(src_cuda, src_hip)
        self.assertEqual(_kernel_part(src_cuda), _kernel_part(src_hip))
        self.assertIn("cvt.f32.f16", src_cuda)
        self.assertIn("_Float16", src_hip)

    def test_hip_f16_matches_hip_render(self):
        """HIP 侧的生产路径（render + 注入）与 L2 输出的也必须是同一份源码。"""
        expected = hip_codegen.render(cuda_codegen.binary_template("add"), _F16)
        actual = emit(ew_binary("add"), HIP_DIALECT, DType.F16)
        self.assertEqual(_squash(actual), _squash(expected))


class TestProductionPath(unittest.TestCase):
    """生产路径必须真的走 L2，而不是"测试里走、生产里仍用字符串"。"""

    def test_binary_source_returns_l2_kernel(self):
        from aetherflow.lower.kernel import Kernel

        self.assertIsInstance(cuda_codegen.binary_source("add"), Kernel)

    def test_reduce_sum_source_returns_l2_kernel(self):
        from aetherflow.lower.kernel import Kernel

        self.assertIsInstance(cuda_codegen.reduce_sum_source(), Kernel)

    def test_render_of_l2_kernel_equals_render_of_template(self):
        self.assertEqual(
            _squash(cuda_codegen.render(cuda_codegen.binary_source("add"), _F32)),
            _squash(cuda_codegen.render(cuda_codegen.binary_template("add"), _F32)),
        )

    def test_hip_render_of_l2_kernel_uses_hip_dialect(self):
        src = hip_codegen.render(cuda_codegen.binary_source("add"), _F16)
        self.assertIn("_Float16", src)

    def test_new_sources_return_l2_kernels(self):
        from aetherflow.lower.kernel import Kernel

        for kernel in (
            cuda_codegen.unary_source("relu"),
            cuda_codegen.step_source(),
            cuda_codegen.broadcast_binary_source("mul"),
            cuda_codegen.softmax_source(),
            cuda_codegen.layernorm_source(),
            cuda_codegen.layernorm_residual_source(),
            cuda_codegen.rmsnorm_source(),
            cuda_codegen.silu_gate_source(),
            cuda_codegen.batch_norm_source(),
            cuda_codegen.batch_norm_stats_source(),
            cuda_codegen.maxpool2d_source(),
            cuda_codegen.maxpool2d_backward_source(),
            cuda_codegen.avgpool2d_source(),
            cuda_codegen.avgpool2d_backward_source(),
            cuda_codegen.adaptive_avg_pool2d_source(),
            cuda_codegen.adaptive_avg_pool2d_backward_source(),
            cuda_codegen.im2col_source(),
            cuda_codegen.conv2d_tiled_source(),
            cuda_codegen.attention_source(),
        ):
            self.assertIsInstance(kernel, Kernel)


class TestEwUnary(unittest.TestCase):
    def test_all_operators_match_handwritten_template(self):
        for kind in cuda_codegen.UNARY_EXPR:
            expected = cuda_codegen.render(cuda_codegen.unary_template(kind), _F32)
            actual = emit(ew_unary(kind), CUDA_DIALECT, DType.F32)
            self.assertEqual(_squash(actual), _squash(expected), kind)

    def test_f16_matches_handwritten_template(self):
        expected = cuda_codegen.render(cuda_codegen.unary_template("gelu"), _F16)
        actual = emit(ew_unary("gelu"), CUDA_DIALECT, DType.F16)
        self.assertEqual(_squash(actual), _squash(expected))

    def test_unknown_operator_is_refused(self):
        with self.assertRaises(ValueError):
            ew_unary("pow")


class TestEwStep(unittest.TestCase):
    def test_matches_handwritten_template_f32(self):
        expected = cuda_codegen.render(cuda_codegen.step_template(), _F32)
        actual = emit(ew_step(), CUDA_DIALECT, DType.F32)
        self.assertEqual(_squash(actual), _squash(expected))

    def test_matches_handwritten_template_f16(self):
        expected = cuda_codegen.render(cuda_codegen.step_template(), _F16)
        actual = emit(ew_step(), CUDA_DIALECT, DType.F16)
        self.assertEqual(_squash(actual), _squash(expected))


class TestEwBroadcast(unittest.TestCase):
    def test_all_operators_match_handwritten_template(self):
        for kind in ("add", "sub", "mul", "div"):
            expected = cuda_codegen.render(
                cuda_codegen.broadcast_template(kind), _F32
            )
            actual = emit(ew_broadcast(kind), CUDA_DIALECT, DType.F32)
            self.assertEqual(_squash(actual), _squash(expected), kind)

    def test_f16_matches_handwritten_template(self):
        expected = cuda_codegen.render(
            cuda_codegen.broadcast_template("mul"), _F16
        )
        actual = emit(ew_broadcast("mul"), CUDA_DIALECT, DType.F16)
        self.assertEqual(_squash(actual), _squash(expected))


class TestSoftmax(unittest.TestCase):
    def test_matches_handwritten_template_f32(self):
        expected = cuda_codegen.render(cuda_codegen.softmax_template(), _F32)
        actual = emit(softmax_last(), CUDA_DIALECT, DType.F32)
        self.assertEqual(_squash(actual), _squash(expected))

    def test_matches_handwritten_template_f16(self):
        expected = cuda_codegen.render(cuda_codegen.softmax_template(), _F16)
        actual = emit(softmax_last(), CUDA_DIALECT, DType.F16)
        self.assertEqual(_squash(actual), _squash(expected))


class TestLayernorm(unittest.TestCase):
    def test_matches_handwritten_template_f32(self):
        expected = cuda_codegen.render(cuda_codegen.layernorm_template(), _F32)
        actual = emit(layernorm_last(), CUDA_DIALECT, DType.F32)
        self.assertEqual(_squash(actual), _squash(expected))

    def test_matches_handwritten_template_f16(self):
        expected = cuda_codegen.render(cuda_codegen.layernorm_template(), _F16)
        actual = emit(layernorm_last(), CUDA_DIALECT, DType.F16)
        self.assertEqual(_squash(actual), _squash(expected))


class TestLayernormResidual(unittest.TestCase):
    """融合残差要在 L2 里显式表达"先舍入回存储类型"。"""

    def test_matches_handwritten_template_f32(self):
        expected = cuda_codegen.render(
            cuda_codegen.layernorm_residual_template(), _F32
        )
        actual = emit(layernorm_residual_last(), CUDA_DIALECT, DType.F32)
        self.assertEqual(_squash(actual), _squash(expected))

    def test_matches_handwritten_template_f16(self):
        expected = cuda_codegen.render(
            cuda_codegen.layernorm_residual_template(), _F16
        )
        actual = emit(layernorm_residual_last(), CUDA_DIALECT, DType.F16)
        self.assertEqual(_squash(actual), _squash(expected))

    def test_f16_keeps_the_storage_rounding(self):
        src = emit(layernorm_residual_last(), CUDA_DIALECT, DType.F16)
        self.assertIn("af_st_h(af_ld_h(xr[i]) + af_ld_h(rr[i]))", src)
        self.assertIn("unsigned short xv =", src)


class TestRmsnorm(unittest.TestCase):
    def test_matches_handwritten_template_f32(self):
        expected = cuda_codegen.render(cuda_codegen.rmsnorm_template(), _F32)
        actual = emit(rmsnorm_last(), CUDA_DIALECT, DType.F32)
        self.assertEqual(_squash(actual), _squash(expected))

    def test_matches_handwritten_template_f16(self):
        expected = cuda_codegen.render(cuda_codegen.rmsnorm_template(), _F16)
        actual = emit(rmsnorm_last(), CUDA_DIALECT, DType.F16)
        self.assertEqual(_squash(actual), _squash(expected))


class TestSiluGate(unittest.TestCase):
    def test_matches_handwritten_template_f32(self):
        expected = cuda_codegen.render(cuda_codegen.silu_gate_template(), _F32)
        actual = emit(silu_gate(), CUDA_DIALECT, DType.F32)
        self.assertEqual(_squash(actual), _squash(expected))

    def test_matches_handwritten_template_f16(self):
        expected = cuda_codegen.render(cuda_codegen.silu_gate_template(), _F16)
        actual = emit(silu_gate(), CUDA_DIALECT, DType.F16)
        self.assertEqual(_squash(actual), _squash(expected))


class TestBatchNorm(unittest.TestCase):
    """模板里两个内核共用一份源码，L2 拆成两个，按名字各对照一段。"""

    def _expected(self, name, dtype):
        return _named_kernel(
            cuda_codegen.render(cuda_codegen.batch_norm_template(), dtype), name
        )

    def test_batch_norm_matches_handwritten_template_f32(self):
        actual = emit(batch_norm(), CUDA_DIALECT, DType.F32)
        self.assertEqual(
            _squash(_kernel_part(actual)),
            _squash(self._expected("batch_norm", _F32)),
        )

    def test_batch_norm_matches_handwritten_template_f16(self):
        actual = emit(batch_norm(), CUDA_DIALECT, DType.F16)
        self.assertEqual(
            _squash(_kernel_part(actual)),
            _squash(self._expected("batch_norm", _F16)),
        )

    def test_batch_norm_stats_matches_handwritten_template_f32(self):
        actual = emit(batch_norm_stats(), CUDA_DIALECT, DType.F32)
        self.assertEqual(
            _squash(_kernel_part(actual)),
            _squash(self._expected("batch_norm_stats", _F32)),
        )

    def test_batch_norm_stats_matches_handwritten_template_f16(self):
        actual = emit(batch_norm_stats(), CUDA_DIALECT, DType.F16)
        self.assertEqual(
            _squash(_kernel_part(actual)),
            _squash(self._expected("batch_norm_stats", _F16)),
        )


class TestPooling(unittest.TestCase):
    """池化家族：引入 ``AtomicAdd``（重叠窗口散射）与 ``Continue``（越界跳过）。"""

    def _cases(self):
        return (
            (maxpool2d, cuda_codegen.maxpool2d_template),
            (maxpool2d_backward, cuda_codegen.maxpool2d_backward_template),
            (maxpool2d_indices, cuda_codegen.maxpool2d_indices_template),
            (avgpool2d, cuda_codegen.avgpool2d_template),
            (avgpool2d_backward, cuda_codegen.avgpool2d_backward_template),
            (adaptive_avg_pool2d, cuda_codegen.adaptive_avg_pool2d_template),
            (
                adaptive_avg_pool2d_backward,
                cuda_codegen.adaptive_avg_pool2d_backward_template,
            ),
        )

    def test_matches_handwritten_template(self):
        for dtype, np_dtype in ((DType.F32, _F32), (DType.F16, _F16)):
            for build, template in self._cases():
                actual = emit(build(), CUDA_DIALECT, dtype)
                expected = cuda_codegen.render(template(), np_dtype)
                self.assertEqual(
                    _squash(_kernel_part(actual)),
                    _squash(_kernel_part(expected)),
                    f"{build.__name__} {dtype.value}",
                )

    def test_backward_scatter_is_atomic(self):
        # 重叠窗口会命中同一个输入单元，普通写入只留最后一个窗口的贡献。
        for build in (
            maxpool2d_backward,
            avgpool2d_backward,
            adaptive_avg_pool2d_backward,
        ):
            self.assertIn(
                "af_add_f(&",
                emit(build(), CUDA_DIALECT, DType.F32),
                build.__name__,
            )

    def test_f16_backward_uses_the_16bit_atomic(self):
        self.assertIn(
            "af_add_h(&",
            emit(avgpool2d_backward(), CUDA_DIALECT, DType.F16),
        )


class TestConcat(unittest.TestCase):
    """拼接 / 补零的分段拷贝：只做搬运，读写都不经过 float。"""

    def test_matches_handwritten_template(self):
        for dtype, np_dtype in ((DType.F32, _F32), (DType.F16, _F16)):
            actual = emit(concat_copy(), CUDA_DIALECT, dtype)
            expected = cuda_codegen.render(
                cuda_codegen.concat_copy_template(), np_dtype
            )
            self.assertEqual(
                _squash(_kernel_part(actual)),
                _squash(_kernel_part(expected)),
                dtype.value,
            )

    def test_copy_is_a_bitwise_move(self):
        # f16 的拷贝必须逐位搬运：套一层 af_st 会多一次"读成 float 再舍入"。
        body = _kernel_part(emit(concat_copy(), CUDA_DIALECT, DType.F16))
        self.assertNotIn("af_ld_", body)
        self.assertNotIn("af_st_", body)


class TestDropoutMask(unittest.TestCase):
    """dropout 掩码：随机源是内核私有的设备函数，但掩码本身必须逐位可复现。"""

    def test_matches_handwritten_template(self):
        for dtype, np_dtype in ((DType.F32, _F32), (DType.F16, _F16)):
            actual = emit(dropout_mask(), CUDA_DIALECT, dtype)
            expected = cuda_codegen.render(
                cuda_codegen.dropout_mask_template(), np_dtype
            )
            self.assertEqual(
                _squash(_kernel_part(actual)),
                _squash(_kernel_part(expected)),
                dtype.value,
            )

    def test_device_helper_is_private_to_the_kernel(self):
        self.assertIn("splitmix64", emit(dropout_mask(), CUDA_DIALECT, DType.F32))
        self.assertNotIn(
            "splitmix64", emit(ew_binary("add"), CUDA_DIALECT, DType.F32)
        )


class TestScanLinear(unittest.TestCase):
    """线性递推（正/反向）：一个线程一个尾元素，沿时间轴串行。"""

    def _cases(self):
        return (
            (scan_linear, cuda_codegen.scan_linear_template),
            (scan_linear_backward, cuda_codegen.scan_linear_backward_template),
        )

    def test_matches_handwritten_template(self):
        for dtype, np_dtype in ((DType.F32, _F32), (DType.F16, _F16)):
            for build, template in self._cases():
                actual = emit(build(), CUDA_DIALECT, dtype)
                expected = cuda_codegen.render(template(), np_dtype)
                self.assertEqual(
                    _squash(_kernel_part(actual)),
                    _squash(_kernel_part(expected)),
                    f"{build.__name__} {dtype.value}",
                )


class TestChunkedScan(unittest.TestCase):
    """分块扫描：没有手写对照模板，钉住结构与 scratch 的存储宽度。

    它们不是 ``scan_linear`` 的逐字等价物（块间前缀改变舍入），"逐 token 相等"
    在这里不成立；数值验收在 ``TestCudaChunkedScan``，这里只保证离线能渲染、
    且 f16 下块和仍是 float32——降成半精度会多两次舍入。
    """

    BUILDS = (
        (scan_linear_chunk_sum, "scan_linear_chunk_sum"),
        (scan_linear_chunk, "scan_linear_chunk"),
        (scan_linear_chunk_sum_backward, "scan_linear_chunk_sum_backward"),
        (scan_linear_chunk_backward, "scan_linear_chunk_backward"),
        (scan_linear_last, "scan_linear_last"),
        (scan_linear_chunk_last, "scan_linear_chunk_last"),
    )

    def test_renders_both_dtypes(self):
        for build, name in self.BUILDS:
            for dtype in (DType.F32, DType.F16):
                src = emit(build(), CUDA_DIALECT, dtype)
                self.assertIn(f'void {name}(', src, f"{name} {dtype.value}")
                self.assertIn("__global__", src, f"{name} {dtype.value}")

    def test_scratch_stays_float32_in_f16(self):
        for build in (
            scan_linear_chunk_sum,
            scan_linear_chunk_sum_backward,
        ):
            src = emit(build(), CUDA_DIALECT, DType.F16)
            self.assertIn("unsigned short* __restrict__ a", src, build.__name__)
            self.assertIn("float* __restrict__ prod", src, build.__name__)
            self.assertIn("float* __restrict__ sum", src, build.__name__)
            self.assertIn("prod[j] = p;", src, build.__name__)

    def test_first_chunk_has_no_prefix_scan(self):
        """``k == 0`` 时前缀循环体不执行：块内递推与串行版逐字相同。"""
        src = emit(scan_linear_chunk(), CUDA_DIALECT, DType.F32)
        self.assertIn("for (long long c = 0; c < k; ++c)", src)

    def test_last_state_writes_one_value_per_element(self):
        src = _kernel_part(emit(scan_linear_last(), CUDA_DIALECT, DType.F32))
        self.assertNotIn("out[idx]", src)
        self.assertIn("out[i]", src)


class TestConv2dGrads(unittest.TestCase):
    """卷积的输入/权重梯度：gather 形式，每个输出元素只被一个线程写。"""

    def _cases(self):
        return (
            (conv2d_input_grad, cuda_codegen.conv2d_input_grad_template),
            (conv2d_weight_grad, cuda_codegen.conv2d_weight_grad_template),
        )

    def test_matches_handwritten_template(self):
        for dtype, np_dtype in ((DType.F32, _F32), (DType.F16, _F16)):
            for build, template in self._cases():
                actual = emit(build(), CUDA_DIALECT, dtype)
                expected = cuda_codegen.render(template(), np_dtype)
                self.assertEqual(
                    _squash(_kernel_part(actual)),
                    _squash(_kernel_part(expected)),
                    f"{build.__name__} {dtype.value}",
                )

    def test_gather_needs_no_atomics(self):
        for build in (conv2d_input_grad, conv2d_weight_grad):
            self.assertNotIn(
                "af_add_f(&",
                _kernel_part(emit(build(), CUDA_DIALECT, DType.F32)),
                build.__name__,
            )


class TestConv2dIm2col(unittest.TestCase):
    """im2col 与直接分块卷积：两条互不物化/物化列矩阵的路径。"""

    def test_im2col_matches_handwritten_template(self):
        for dtype, np_dtype in ((DType.F32, _F32), (DType.F16, _F16)):
            actual = emit(im2col(), CUDA_DIALECT, dtype)
            expected = cuda_codegen.render(
                cuda_codegen.im2col_template(), np_dtype
            )
            self.assertEqual(
                _squash(_kernel_part(actual)),
                _squash(_kernel_part(expected)),
                dtype.value,
            )

    def test_conv2d_tiled_matches_handwritten_template(self):
        # 这个内核不按 dtype 参数化（全程 float），所以两种 dtype 的期望值
        # 是同一份源码——打印器里一旦出现"存储类型"泄漏，这里会立刻失败。
        for dtype in (DType.F32, DType.F16):
            actual = emit(conv2d_tiled(), CUDA_DIALECT, dtype)
            expected = cuda_codegen.render(
                cuda_codegen.conv2d_tiled_template(), _F32
            )
            self.assertEqual(
                _squash(_kernel_part(actual)),
                _squash(_kernel_part(expected)),
                dtype.value,
            )

    def test_grouped_uses_dilation_and_group_offsets(self):
        """分组直算核的三处索引必须各自对：dilation、组起点、权重偏移。"""
        src = _squash(_kernel_part(emit(conv2d_grouped(), CUDA_DIALECT, DType.F32)))
        self.assertIn("ih=ho*SH-PH+ki*DH;", src)
        self.assertIn("iw=wo*SW-PW+kj*DW;", src)
        self.assertIn("intg=o/OG;", src)
        self.assertIn("intcstart=g*CG;", src)
        self.assertIn("base=x+((longlong)n*GROUPS*CG+cstart+cg)", src)
        self.assertIn("wb=w+((longlong)o*CG+cg)", src)
        # 与分块核一样不按 dtype 参数化：f16 下指针仍是 float
        f16 = _squash(
            _kernel_part(emit(conv2d_grouped(), CUDA_DIALECT, DType.F16))
        )
        self.assertIn("constfloat*__restrict__x", f16)

    def test_border_threads_still_reach_the_barrier(self):
        # 越界线程必须在屏障处走 continue，提前 return 会让 __syncthreads 死锁；
        # 唯一允许的 return 在 c0 循环之后。
        src = _squash(_kernel_part(emit(conv2d_tiled(), CUDA_DIALECT, DType.F32)))
        self.assertLess(
            src.index("if(!active){continue;}"),
            src.index("if(!active){return;}"),
        )


class TestAttention(unittest.TestCase):
    """融合注意力前向：动态共享内存 + 在线 softmax，只在 f32 上生产。"""

    def test_matches_handwritten_template(self):
        actual = emit(attention_fwd(), CUDA_DIALECT, DType.F32)
        expected = cuda_codegen.render(cuda_codegen.attention_template(), _F32)
        self.assertEqual(
            _squash(_kernel_part(actual)),
            _squash(_kernel_part(expected)),
        )

    def test_device_helper_is_private_to_the_kernel(self):
        self.assertIn(
            "attention_rand", emit(attention_fwd(), CUDA_DIALECT, DType.F32)
        )
        self.assertNotIn(
            "attention_rand", emit(ew_binary("add"), CUDA_DIALECT, DType.F32)
        )

    def test_shared_memory_is_split_by_run_time_block_size(self):
        # 三段共享内存都从 blockDim.x 算出偏移，所以内核里不能出现静态尺寸。
        src = _kernel_part(emit(attention_fwd(), CUDA_DIALECT, DType.F32))
        self.assertIn("extern __shared__ float smem[];", src)
        self.assertIn("float* s_acc = smem + 2 * blockDim.x;", src)

    def test_f16_is_mixed_precision(self):
        """f16 下只有存储是半精度：指针类型、读、写都跟着 dtype 走，
        行最大值、归一化因子与输出累加器仍在 float 里。"""
        src = _squash(_kernel_part(emit(attention_fwd(), CUDA_DIALECT, DType.F16)))
        self.assertIn("constunsignedshort*__restrict__q", src)
        self.assertIn("constunsignedshort*qs=q+", src)
        self.assertIn("dot+=af_ld_h(qs[d])*af_ld_h(kt[d]);", src)
        self.assertIn("sc+=af_ld_h(mrow[t]);", src)
        self.assertIn("acc+=s_w[j]*af_ld_h(vbase[", src)
        self.assertIn("unsignedshort*orow=out+", src)
        self.assertIn("orow[e]=af_st_h(", src)
        # 累加器与共享内存不随 dtype 变宽/变窄
        self.assertIn("extern__shared__floatsmem[];", src)
        self.assertIn("floatacc=s_acc[e]*corr;", src)


class TestNewKernelsAcrossVendors(unittest.TestCase):
    """新增内核族同样要满足"f32 一字不差、f16 只差 prelude"。"""

    def _kernels(self):
        return (
            ("unary", ew_unary("silu")),
            ("step", ew_step()),
            ("broadcast", ew_broadcast("div")),
            ("softmax", softmax_last()),
            ("layernorm", layernorm_last()),
            ("layernorm_residual", layernorm_residual_last()),
            ("rmsnorm", rmsnorm_last()),
            ("silu_gate", silu_gate()),
            ("batch_norm", batch_norm()),
            ("batch_norm_stats", batch_norm_stats()),
            ("concat_copy", concat_copy()),
            ("dropout_mask", dropout_mask()),
            ("scan_linear", scan_linear()),
            ("scan_linear_backward", scan_linear_backward()),
            ("conv2d_input_grad", conv2d_input_grad()),
            ("conv2d_weight_grad", conv2d_weight_grad()),
            ("maxpool2d", maxpool2d()),
            ("maxpool2d_backward", maxpool2d_backward()),
            ("avgpool2d", avgpool2d()),
            ("avgpool2d_backward", avgpool2d_backward()),
            ("adaptive_avg_pool2d", adaptive_avg_pool2d()),
            ("adaptive_avg_pool2d_backward", adaptive_avg_pool2d_backward()),
            ("im2col", im2col()),
            ("conv2d_tiled", conv2d_tiled()),
            ("attention_fwd", attention_fwd()),
        )

    def test_f32_is_identical_across_vendors(self):
        for name, kernel in self._kernels():
            self.assertEqual(
                emit(kernel, CUDA_DIALECT, DType.F32),
                emit(kernel, HIP_DIALECT, DType.F32),
                name,
            )

    def test_f16_differs_only_in_prelude(self):
        for name, kernel in self._kernels():
            src_cuda = emit(kernel, CUDA_DIALECT, DType.F16)
            src_hip = emit(kernel, HIP_DIALECT, DType.F16)
            self.assertNotEqual(src_cuda, src_hip, name)
            self.assertEqual(
                _kernel_part(src_cuda), _kernel_part(src_hip), name
            )


class TestElementwiseChain(unittest.TestCase):
    """语义级 L2：一串逐元素算子编成一个内核，可选 4 通道向量化。

    这一组不看数值（数值由 CUDA 真机对拍），只看**结构**：中间值不物化、
    每步都在、向量路径只有链首与链尾做 128 位访存。
    """

    def _steps(self):
        return (
            ElemStep("t0", "add", ("p0", "p1")),
            ElemStep("t1", "relu", ("t0",)),
            ElemStep("t2", "mul", ("t1", "p2")),
        )

    def _inputs(self):
        return (ElemInput("p0"), ElemInput("p1"), ElemInput("p2", scalar=True))

    def test_scalar_path_materializes_each_step(self):
        src = _kernel_part(
            emit(build_chain(self._steps(), self._inputs()), CUDA_DIALECT, DType.F32)
        )
        self.assertIn("float v_p0 = af_ld_f(p0[i]);", src)
        self.assertIn("float v_p2 = af_ld_f(p2[0]);", src)
        self.assertIn("float v_t0 = v_p0 + v_p1;", src)
        self.assertIn("float v_t1 = v_t0 > 0.f ? v_t0 : 0.f;", src)
        self.assertIn("float v_t2 = v_t1 * v_p2;", src)
        self.assertIn("out[i] = af_st_f(v_t2);", src)

    def test_vector_path_reads_four_lanes(self):
        src = _kernel_part(
            emit(
                build_chain(self._steps(), self._inputs(), vec=4),
                CUDA_DIALECT,
                DType.F32,
            )
        )
        self.assertIn("float4 v_p0 = af_ld4_f(p0, i * 4);", src)
        self.assertIn("float4 v_p1 = af_ld4_f(p1, i * 4);", src)
        # 标量一侧仍然只读一个元素，四条通道共用
        self.assertIn("float v_p2 = af_ld_f(p2[0]);", src)
        self.assertIn("af_st4_f(out, i * 4,", src)
        # 每个线程负责 4 个元素：循环上界是向量数，索引必须回到 4 的倍数
        self.assertIn("i < n / 4", src)
        for lane in "xyzw":
            self.assertIn(f"v_p0.{lane}", src)
        # 中间值不落内存：向量路径里没有 float v_t0 / v_t1
        self.assertNotIn("float v_t0", src)
        self.assertNotIn("float v_t1", src)

    def test_vector_helpers_are_private_to_the_kernel(self):
        vector = emit(
            build_chain(self._steps(), self._inputs(), vec=4),
            CUDA_DIALECT,
            DType.F32,
        )
        self.assertIn("af_ld4_f", vector)
        self.assertNotIn(
            "af_ld4_f", emit(ew_binary("add"), CUDA_DIALECT, DType.F32)
        )

    def test_f16_rounds_every_step(self):
        """不融合的图里每步都是一次 f16 往返；融合不能把中间结果留在 float。"""
        src = _kernel_part(
            emit(
                build_chain(self._steps(), self._inputs(), dtype=DType.F16),
                CUDA_DIALECT,
                DType.F16,
            )
        )
        self.assertIn("unsigned short v_t0 = af_st_h(v_p0 + v_p1);", src)
        self.assertIn("af_ld_h(v_t0)", src)
        self.assertIn("out[i] = af_st_h(v_t2);", src)

    def test_vectorization_is_f32_only(self):
        with self.assertRaises(ValueError) as ctx:
            build_chain(self._steps(), self._inputs(), vec=4, dtype=DType.F16)
        self.assertIn("f32", str(ctx.exception))

    def test_unknown_operator_is_refused(self):
        with self.assertRaises(ValueError):
            build_chain((ElemStep("t0", "pow", ("p0",)),), (ElemInput("p0"),))

    def test_empty_chain_is_refused(self):
        with self.assertRaises(ValueError):
            build_chain((), (ElemInput("p0"),))

    def test_undefined_value_is_refused(self):
        with self.assertRaises(ValueError):
            build_chain((ElemStep("t0", "relu", ("nope",)),), (ElemInput("p0"),))

    def test_vector_width_must_be_one_or_four(self):
        with self.assertRaises(ValueError):
            build_chain(self._steps(), self._inputs(), vec=2)

    def test_f32_chain_is_identical_across_vendors(self):
        kernel = build_chain(self._steps(), self._inputs(), vec=4)
        self.assertEqual(
            emit(kernel, CUDA_DIALECT, DType.F32),
            emit(kernel, HIP_DIALECT, DType.F32),
        )


class TestUnsupportedDtype(unittest.TestCase):
    def test_f64_is_refused_with_reason(self):
        with self.assertRaises(ValueError) as ctx:
            emit(ew_binary("add"), CUDA_DIALECT, DType.F64)
        message = str(ctx.exception)
        self.assertIn("f64", message)
        self.assertIn("cuda", message)


class TestSpikeCsrScan(unittest.TestCase):
    """``spike_csr`` 的块内扫描（5.176）：一 block 一行，段内前缀和保序。"""

    def _src(self, dtype=_F32):
        return cuda_codegen.render(cuda_codegen.spike_csr_source(), dtype)

    def test_one_block_per_row(self):
        src = self._src()
        self.assertIn("const long long t = (long long)blockIdx.x;", src)
        # 网格不再步进：每行由一个 block 独占，块内要同步
        self.assertNotIn("gridDim", src)

    def test_row_scan_uses_block_prefix_sum(self):
        src = self._src()
        self.assertIn("block_scan_sum(taken, smem)", src)
        self.assertIn("__shared__ int smem[1026];", src)
        self.assertIn("(width + blockDim.x - 1) / blockDim.x", src)
        self.assertIn("indices[lo + written + pre] = (int)k;", src)
        self.assertIn("written = written + smem[blockDim.x];", src)

    def test_hit_is_not_read_past_the_row(self):
        """最后一段的列号可能越过 width：读 ``spikes`` 前必须短路。"""
        src = self._src()
        self.assertIn("int taken = k < width ? ", src)

    def test_hit_count_must_match_offsets(self):
        """参考实现要求"命中数 == hi - lo"，设备侧也要把两侧失配记成错误。"""
        src = self._src()
        self.assertIn("if (lo + written != hi) {", src)

    def test_hip_f16_reuses_the_same_kernel(self):
        src = hip_codegen.render(cuda_codegen.spike_csr_source(), _F16)
        self.assertIn("block_scan_sum(taken, smem)", src)
        self.assertIn("_Float16", src)


if __name__ == "__main__":
    unittest.main()
