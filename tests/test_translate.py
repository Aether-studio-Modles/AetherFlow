"""转译模式的判据：只改登记过的 token、改得对、改不了就当场拒绝。

这组用例盯三件事：

1. **改对了**：运行期 API、头文件、枚举与类型名在两家之间逐个对上；
2. **没改错**：注释、字符串、未登记的标识符原样透传——转译器不是"整篇
   文本替换"，把注释里的 ``cudaMemcpy`` 也改掉就是一份看不出问题的坏源码；
3. **改不了就说**：带掩码的 warp 内建、PTX 内联汇编、``_Float16`` 这些
   "换个名字就变语义"的东西必须抛错并带行号，而不是产出一份编译期才炸的
   源码。
"""

import unittest

from aetherflow.backends.cuda.dialect import CUDA_DIALECT
from aetherflow.backends.hip.codegen import _HIP_F16_HELPERS
from aetherflow.backends.hip.dialect import HIP_DIALECT
from aetherflow.ir.types import DType
from aetherflow.lower import (
    SUPPORTED_DIALECTS,
    TranslationError,
    compile_translated,
    emit,
    ew_binary,
    translate,
)

_LAUNCH = """
#include <cuda_runtime.h>

extern "C" __global__ void add(const float* __restrict__ a,
                               float* __restrict__ b) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1024) b[i] = a[i] + 1.0f;
}
"""


class FakeBackend:
    """只记录 ``kernel()`` 收到的东西；用来验证缓存键与方言校验。"""

    def __init__(self, label="HIP 后端"):
        self._LABEL = label
        self.calls = []

    def kernel(self, key, source, func):
        self.calls.append((key, source, func))
        return ("handle", key)


class TestRewrite(unittest.TestCase):
    def test_include_and_runtime_api_are_rewritten(self):
        source = (
            "#include <cuda_runtime.h>\n"
            "cudaError_t rc = cudaMalloc(&p, n);\n"
            "cudaMemcpy(p, h, n, cudaMemcpyHostToDevice);\n"
            "cudaDeviceSynchronize();\n"
        )
        out = translate(source, source_dialect="cuda", target_dialect="hip")
        self.assertIn("#include <hip/hip_runtime.h>", out)
        self.assertIn("hipError_t rc = hipMalloc(&p, n);", out)
        self.assertIn("hipMemcpy(p, h, n, hipMemcpyHostToDevice);", out)
        self.assertIn("hipDeviceSynchronize();", out)
        self.assertNotIn("cuda", out)

    def test_reverse_direction(self):
        source = "#include <hip/hip_runtime.h>\nhipFree(p);\nhipSuccess;\n"
        out = translate(source, source_dialect="hip", target_dialect="cuda")
        self.assertIn("#include <cuda_runtime.h>", out)
        self.assertIn("cudaFree(p);", out)
        self.assertIn("cudaSuccess;", out)

    def test_comments_and_strings_are_not_code(self):
        source = (
            '// cudaMalloc 在这里只是提一句\n'
            '/* cudaMemcpy 同上 */\n'
            'const char* name = "cudaMalloc";\n'
            'char quote = \'c\';\n'
            'cudaFree(p);\n'
        )
        out = translate(source, source_dialect="cuda", target_dialect="hip")
        self.assertIn("// cudaMalloc 在这里只是提一句", out)
        self.assertIn("/* cudaMemcpy 同上 */", out)
        self.assertIn('const char* name = "cudaMalloc";', out)
        self.assertIn("char quote = 'c';", out)
        self.assertIn("hipFree(p);", out)

    def test_unregistered_identifiers_pass_through(self):
        source = "my_kernel_launcher(x, cudaFree);\n"  # cudaFree 是已登记的
        out = translate(source, source_dialect="cuda", target_dialect="hip")
        self.assertIn("my_kernel_launcher(x, hipFree);", out)

    def test_unregistered_header_stays(self):
        source = "#include <math.h>\n"
        out = translate(source, source_dialect="cuda", target_dialect="hip")
        self.assertEqual(out, source)

    def test_same_dialect_is_identity_even_with_asm(self):
        source = 'asm volatile("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h));\n'
        self.assertEqual(translate(source, source_dialect="cuda", target_dialect="cuda"), source)

    def test_builtins_shared_by_both_sides_are_untouched(self):
        source = "__syncthreads(); atomicAdd(&p[i], v); threadIdx.x;\n"
        translate(source, source_dialect="cuda", target_dialect="hip")
        # 只验证不抛错：这些名字两边同名，替换表里是恒等项。
        self.assertEqual(
            translate(source, source_dialect="cuda", target_dialect="hip"), source
        )


class TestRefusal(unittest.TestCase):
    def test_ptx_asm_is_refused(self):
        source = 'x = 1;\nasm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h));\n'
        with self.assertRaises(TranslationError) as ctx:
            translate(source, source_dialect="cuda", target_dialect="hip")
        self.assertIn("第 2 行", str(ctx.exception))
        self.assertIn("asm", str(ctx.exception))

    def test_masked_warp_intrinsic_is_refused(self):
        source = "v = __shfl_down_sync(0xffffffff, v, 1);\n"
        with self.assertRaises(TranslationError) as ctx:
            translate(source, source_dialect="cuda", target_dialect="hip")
        self.assertIn("__shfl_down_sync", str(ctx.exception))
        self.assertIn("掩码", str(ctx.exception))

    def test_half_type_is_refused_in_both_directions(self):
        with self.assertRaises(TranslationError):
            translate("__half h;\n", source_dialect="cuda", target_dialect="hip")
        with self.assertRaises(TranslationError):
            translate("_Float16 h;\n", source_dialect="hip", target_dialect="cuda")

    def test_hip_f16_helpers_cannot_be_translated_to_cuda(self):
        """HIP 的 f16 辅助函数用了 __builtin_bit_cast；转译必须拒绝。

        这也正是 ``backends/hip/codegen.py`` 必须另写一份 f16 辅助函数、
        而不能靠转译得到的原因。
        """
        with self.assertRaises(TranslationError) as ctx:
            translate(_HIP_F16_HELPERS, source_dialect="hip", target_dialect="cuda")
        self.assertIn("__builtin_bit_cast", str(ctx.exception))

    def test_cuda_f16_prelude_is_refused(self):
        from aetherflow.backends.cuda.codegen import _F16_HELPERS

        with self.assertRaises(TranslationError):
            translate(_F16_HELPERS, source_dialect="cuda", target_dialect="hip")


class TestArguments(unittest.TestCase):
    def test_unknown_dialect_names_are_rejected(self):
        with self.assertRaises(ValueError):
            translate("x;\n", source_dialect="rocm", target_dialect="hip")
        with self.assertRaises(ValueError):
            translate("x;\n", source_dialect="cuda", target_dialect="vulkan")

    def test_source_must_be_text(self):
        with self.assertRaises(TypeError):
            translate(b"x;\n", source_dialect="cuda", target_dialect="hip")

    def test_supported_dialects_are_the_two_vendors(self):
        self.assertEqual(SUPPORTED_DIALECTS, ("cuda", "hip"))


class TestAgreesWithNativeCodegen(unittest.TestCase):
    def test_f32_sources_are_identical_so_translation_is_identity(self):
        """f32 的两份方言逐字相同（5.40）；转译不该改动任何一个字符。"""
        cuda_source = emit(ew_binary("add"), CUDA_DIALECT, DType.F32)
        hip_source = emit(ew_binary("add"), HIP_DIALECT, DType.F32)
        self.assertEqual(cuda_source, hip_source)
        self.assertEqual(
            translate(cuda_source, source_dialect="cuda", target_dialect="hip"),
            hip_source,
        )

    def test_f16_sources_differ_only_where_translation_refuses(self):
        """f16 的差异恰好落在转译拒收项上：两条路径互相印证。"""
        cuda_source = emit(ew_binary("add"), CUDA_DIALECT, DType.F16)
        hip_source = emit(ew_binary("add"), HIP_DIALECT, DType.F16)
        self.assertNotEqual(cuda_source, hip_source)
        with self.assertRaises(TranslationError):
            translate(cuda_source, source_dialect="cuda", target_dialect="hip")
        with self.assertRaises(TranslationError):
            translate(hip_source, source_dialect="hip", target_dialect="cuda")


class TestCompileTranslated(unittest.TestCase):
    def test_target_dialect_is_inferred_from_the_backend_label(self):
        backend = FakeBackend("HIP 后端")
        handle, key = compile_translated(
            backend, _LAUNCH, "add", source_dialect="cuda"
        )
        self.assertEqual(handle, "handle")
        self.assertEqual(backend.calls[0][2], "add")
        self.assertIn("#include <hip/hip_runtime.h>", backend.calls[0][1])
        self.assertNotIn("cuda", backend.calls[0][1])

    def test_cache_key_depends_on_the_original_source(self):
        backend = FakeBackend("HIP 后端")
        compile_translated(backend, _LAUNCH, "add", source_dialect="cuda")
        compile_translated(backend, _LAUNCH, "add", source_dialect="cuda")
        compile_translated(backend, _LAUNCH + "\n", "add", source_dialect="cuda")
        keys = [call[0] for call in backend.calls]
        self.assertEqual(keys[0], keys[1])
        self.assertNotEqual(keys[0], keys[2])
        self.assertEqual(keys[0][0], "translate")
        self.assertEqual(len(keys[0][3]), 64)  # sha256 十六进制

    def test_mismatched_backend_is_rejected(self):
        backend = FakeBackend("CUDA 后端")
        with self.assertRaises(ValueError) as ctx:
            compile_translated(
                backend, _LAUNCH, "add", source_dialect="hip", target_dialect="hip"
            )
        self.assertIn("目标方言", str(ctx.exception))
        self.assertEqual(backend.calls, [])

    def test_same_vendor_source_passes_through(self):
        """CUDA 后端拿 CUDA 源码：不转译，仍走同一个编译缓存入口。"""
        backend = FakeBackend("CUDA 后端")
        compile_translated(backend, "asm volatile(\"nop;\");\n", "f", source_dialect="cuda")
        self.assertEqual(backend.calls[0][1], "asm volatile(\"nop;\");\n")

    def test_backend_without_a_label_needs_an_explicit_target(self):
        backend = FakeBackend("参考后端")
        with self.assertRaises(ValueError):
            compile_translated(backend, "x;\n", "f", source_dialect="cuda")
        handle, _ = compile_translated(
            backend, "cudaFree(p);\n", "f", source_dialect="cuda", target_dialect="hip"
        )
        self.assertEqual(handle, "handle")
        self.assertEqual(backend.calls[0][1], "hipFree(p);\n")

    def test_backend_without_kernel_is_rejected(self):
        class NoCodegen:
            _LABEL = "HIP 后端"

        with self.assertRaises(TypeError):
            compile_translated(NoCodegen(), "x;\n", "f", source_dialect="cuda")


if __name__ == "__main__":
    unittest.main()
