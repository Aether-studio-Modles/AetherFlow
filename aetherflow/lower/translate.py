"""代码生成的第三条路：把**已经写好的**一份厂商源码换成另一家的源码。

此前只有两条路：

- **原生生成**：L1 → 智能体 / L2 内核 IR → 方言 → 源码 → NVRTC / hipRTC。
  这是生产路径，覆盖全部算子；但它要求源码是**本框架打印出来的**。
- **调库复用**：cuBLAS / cuDNN / hipBLAS / MIOpen。它只覆盖库已有的算子。

缺的是第三条：手上有一份别人写好的内核（论文附录、旧项目里的 ``.cu``、某家
库的示例），想拿到另一块卡上跑。没有这条路，唯一的选择是"读懂它再用 L2 重
写一遍"——而重写恰恰是最容易改错语义的一步。

**转译的边界（这是本模块最重要的部分）**
--------------------------------------
只做**逐 token 可证的**改写：一个标识符换成另一个标识符，一个头文件名换成
另一个头文件名。凡是"换个名字就变了语义"的地方，一律显式拒绝，并在错误
里说清楚为什么：

- ``__shfl_down_sync(mask, v, d)`` → HIP 的 ``__shfl_down(v, d)`` 要去掉
  第一个掩码参数。这不是改名，是改参数列表——本模块拒绝，而不是产出一份
  参数个数不对、只能在编译期才炸的源码。
- ``(_Float16)v`` → CUDA 的 ``__half`` 需要 ``__float2half_rn`` 才会有正确
  语义；直接改名会生成一份"能编译、但把 float 的位截一半"的源码。拒绝。
- ``__builtin_bit_cast`` 是 Clang 内建，NVRTC 没有。拒绝。
- PTX 内联汇编（``asm("cvt.f32.f16 ...")``）在 AMD clang 上没有对应。拒绝。
  —— 这正是 ``backends/hip/codegen.py`` 里 f16 辅助函数必须另写一份、而不能
  靠转译得到的原因；两条路径在这里互相印证。

**为什么不做完整 C 解析器**
--------------------------
转译的目的不是做编译器，而是"把这一份能跑的源码搬到另一块卡上"。一个能证明
"只有这些 token 变了"的替换表，比一个半吊子的 C 前端更容易审计：读这张表就
知道哪些东西会被改，表上没有的标识符原样透传，表上被拒绝的当场失败。
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, Iterable, Mapping, Optional, Tuple

__all__ = [
    "TranslationError",
    "SUPPORTED_DIALECTS",
    "translate",
    "compile_translated",
]

SUPPORTED_DIALECTS = ("cuda", "hip")

_INCLUDE = re.compile(r'^\s*#\s*include\s*[<"](?P<name>[^>"]+)[>"]')

# 注释、字符串、字符字面量先整体吞掉，只在标识符上做替换：注释是散文，
# 不该被改；字符串里的 ``"cudaMemcpy"`` 是数据，改了就是另一份程序。
_TOKEN = re.compile(
    r"""
      (?P<comment>//[^\n]*|/\*.*?\*/)
    | (?P<string>"(?:\\.|[^"\\])*")
    | (?P<char>'(?:\\.|[^'\\])*')
    | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE | re.DOTALL,
)


class TranslationError(ValueError):
    """这份源码不能被逐 token 安全地换成目标方言。

    与 ``ValueError`` 的区别只是语义：调用方通常要据此回退到"自己重写"，
    所以错误信息里带行号与原因，而不是一句"不支持"。
    """


class _Rules:
    """一个方向上的三张表：头文件、标识符、拒收项。"""

    __slots__ = ("headers", "identifiers", "refuse")

    def __init__(
        self,
        headers: Mapping[str, str],
        identifiers: Mapping[str, str],
        refuse: Mapping[str, str],
    ) -> None:
        self.headers = dict(headers)
        self.identifiers = dict(identifiers)
        self.refuse = dict(refuse)


_CUDA_TO_HIP = _Rules(
    headers={
        "cuda_runtime.h": "hip/hip_runtime.h",
        "cuda_runtime_api.h": "hip/hip_runtime_api.h",
        "device_launch_parameters.h": "hip/hip_runtime.h",
        "cuda_fp16.h": "hip/hip_fp16.h",
        "cuda_bf16.h": "hip/hip_bf16.h",
        "cublas_v2.h": "hipblas/hipblas.h",
        "cublasLt.h": "hipblaslt/hipblaslt.h",
        "cudnn.h": "miopen/miopen.h",
        "curand_kernel.h": "hiprand_kernel.h",
    },
    identifiers={
        # 运行期 API：HIP 的对应函数逐字同名，只有前缀不同。
        "cudaMalloc": "hipMalloc",
        "cudaMallocAsync": "hipMallocAsync",
        "cudaFree": "hipFree",
        "cudaFreeAsync": "hipFreeAsync",
        "cudaMemcpy": "hipMemcpy",
        "cudaMemcpyAsync": "hipMemcpyAsync",
        "cudaMemset": "hipMemset",
        "cudaMemsetAsync": "hipMemsetAsync",
        "cudaDeviceSynchronize": "hipDeviceSynchronize",
        "cudaDeviceReset": "hipDeviceReset",
        "cudaStreamCreate": "hipStreamCreate",
        "cudaStreamDestroy": "hipStreamDestroy",
        "cudaStreamSynchronize": "hipStreamSynchronize",
        "cudaStreamWaitEvent": "hipStreamWaitEvent",
        "cudaEventCreate": "hipEventCreate",
        "cudaEventRecord": "hipEventRecord",
        "cudaEventSynchronize": "hipEventSynchronize",
        "cudaEventElapsedTime": "hipEventElapsedTime",
        "cudaGetLastError": "hipGetLastError",
        "cudaPeekAtLastError": "hipPeekAtLastError",
        "cudaGetErrorString": "hipGetErrorString",
        "cudaHostAlloc": "hipHostMalloc",
        "cudaFreeHost": "hipHostFree",
        "cudaDeviceSetLimit": "hipDeviceSetLimit",
        "cudaGraphCreate": "hipGraphCreate",
        "cudaGraphInstantiate": "hipGraphInstantiate",
        "cudaGraphLaunch": "hipGraphLaunch",
        "cudaGraphDestroy": "hipGraphDestroy",
        "cudaStreamBeginCapture": "hipStreamBeginCapture",
        "cudaStreamEndCapture": "hipStreamEndCapture",
        # 枚举与类型名。
        "cudaSuccess": "hipSuccess",
        "cudaError_t": "hipError_t",
        "cudaStream_t": "hipStream_t",
        "cudaEvent_t": "hipEvent_t",
        "cudaMemcpyHostToDevice": "hipMemcpyHostToDevice",
        "cudaMemcpyDeviceToHost": "hipMemcpyDeviceToHost",
        "cudaMemcpyDeviceToDevice": "hipMemcpyDeviceToDevice",
        "cudaStreamCaptureModeGlobal": "hipStreamCaptureModeGlobal",
        # 核函数与内建：两边同名或只剩类型名差异。
        "atomicAdd": "atomicAdd",
        "atomicMin": "atomicMin",
        "atomicCAS": "atomicCAS",
        "__syncthreads": "__syncthreads",
        "__ldg": "__ldg",
        "__forceinline__": "__forceinline__",
        "__launch_bounds__": "__launch_bounds__",
        "threadIdx": "threadIdx",
        "blockIdx": "blockIdx",
        "blockDim": "blockDim",
        "gridDim": "gridDim",
        "warpSize": "warpSize",
    },
    refuse={
        "asm": "PTX 内联汇编在 AMD clang 上没有对应写法；请改用 amdgcn 内建或 C++ 表达式",
        "__shfl_down_sync": "HIP 的 __shfl_down 不带掩码参数，改名会产出参数个数不对的源码",
        "__shfl_up_sync": "HIP 的 __shfl_up 不带掩码参数，改名会产出参数个数不对的源码",
        "__shfl_xor_sync": "HIP 的 __shfl_xor 不带掩码参数，改名会产出参数个数不对的源码",
        "__shfl_sync": "HIP 的 __shfl 不带掩码参数，改名会产出参数个数不对的源码",
        "__ballot_sync": "HIP 的 __ballot 不带掩码参数，改名会产出参数个数不对的源码",
        "__any_sync": "HIP 的 __any 不带掩码参数，改名会产出参数个数不对的源码",
        "__all_sync": "HIP 的 __all 不带掩码参数，改名会产出参数个数不对的源码",
        "__activemask": "HIP 没有 __activemask；warp 宽度与掩码语义都不同，需要重写",
        "__half": "CUDA 的 __half 与 HIP 的 _Float16 转换规则不同，改名会改变舍入语义",
        "__half2": "HIP 没有 __half2 包装类型，向量半精度需要另写",
        "__half2float": "HIP 没有 __half2float，半精度转换需要显式类型转换",
        "__float2half_rn": "HIP 没有 __float2half_rn，半精度转换需要显式类型转换",
        "cooperative_groups": "HIP 的 cooperative groups 接口不同，需要按目标平台重写",
        "cudaLaunchCooperativeKernel": "HIP 的协作启动接口不同，需要按目标平台重写",
    },
)

_HIP_TO_CUDA = _Rules(
    headers={
        "hip/hip_runtime.h": "cuda_runtime.h",
        "hip/hip_runtime_api.h": "cuda_runtime_api.h",
        "hip/hip_fp16.h": "cuda_fp16.h",
        "hip/hip_bf16.h": "cuda_bf16.h",
        "hipblas/hipblas.h": "cublas_v2.h",
        "miopen/miopen.h": "cudnn.h",
        "hiprand_kernel.h": "curand_kernel.h",
    },
    identifiers={
        "hipMalloc": "cudaMalloc",
        "hipMallocAsync": "cudaMallocAsync",
        "hipFree": "cudaFree",
        "hipFreeAsync": "cudaFreeAsync",
        "hipMemcpy": "cudaMemcpy",
        "hipMemcpyAsync": "cudaMemcpyAsync",
        "hipMemset": "cudaMemset",
        "hipMemsetAsync": "cudaMemsetAsync",
        "hipDeviceSynchronize": "cudaDeviceSynchronize",
        "hipDeviceReset": "cudaDeviceReset",
        "hipStreamCreate": "cudaStreamCreate",
        "hipStreamDestroy": "cudaStreamDestroy",
        "hipStreamSynchronize": "cudaStreamSynchronize",
        "hipStreamWaitEvent": "cudaStreamWaitEvent",
        "hipEventCreate": "cudaEventCreate",
        "hipEventRecord": "cudaEventRecord",
        "hipEventSynchronize": "cudaEventSynchronize",
        "hipEventElapsedTime": "cudaEventElapsedTime",
        "hipGetLastError": "cudaGetLastError",
        "hipPeekAtLastError": "cudaPeekAtLastError",
        "hipGetErrorString": "cudaGetErrorString",
        "hipHostMalloc": "cudaHostAlloc",
        "hipHostFree": "cudaFreeHost",
        "hipDeviceSetLimit": "cudaDeviceSetLimit",
        "hipGraphCreate": "cudaGraphCreate",
        "hipGraphInstantiate": "cudaGraphInstantiate",
        "hipGraphLaunch": "cudaGraphLaunch",
        "hipGraphDestroy": "cudaGraphDestroy",
        "hipStreamBeginCapture": "cudaStreamBeginCapture",
        "hipStreamEndCapture": "cudaStreamEndCapture",
        "hipSuccess": "cudaSuccess",
        "hipError_t": "cudaError_t",
        "hipStream_t": "cudaStream_t",
        "hipEvent_t": "cudaEvent_t",
        "hipMemcpyHostToDevice": "cudaMemcpyHostToDevice",
        "hipMemcpyDeviceToHost": "cudaMemcpyDeviceToHost",
        "hipMemcpyDeviceToDevice": "cudaMemcpyDeviceToDevice",
        "hipStreamCaptureModeGlobal": "cudaStreamCaptureModeGlobal",
        "atomicAdd": "atomicAdd",
        "atomicMin": "atomicMin",
        "atomicCAS": "atomicCAS",
        "__syncthreads": "__syncthreads",
        "__ldg": "__ldg",
        "__forceinline__": "__forceinline__",
        "__launch_bounds__": "__launch_bounds__",
        "threadIdx": "threadIdx",
        "blockIdx": "blockIdx",
        "blockDim": "blockDim",
        "gridDim": "gridDim",
        "warpSize": "warpSize",
    },
    refuse={
        "_Float16": "CUDA 的半精度是 __half 包装类型，赋值要走 __float2half_rn；"
                    "把 _Float16 改名成 __half 会改变舍入语义",
        "__builtin_bit_cast": "Clang 内建，NVRTC 没有；半精度的位改写要改用 __half 的接口",
        "__builtin_amdgcn_wavefrontsize": "AMD 专有内建，CUDA 没有对应",
        "__builtin_amdgcn_s_barrier": "AMD 专有内建，CUDA 没有对应",
        "__hip_bfloat16": "HIP 的 bfloat16 包装类型与 CUDA 的 __nv_bfloat16 接口不同，需要重写",
        "__device_builtin__": "AMD 专有的声明修饰，CUDA 没有对应",
    },
)

_RULES: Dict[Tuple[str, str], _Rules] = {
    ("cuda", "hip"): _CUDA_TO_HIP,
    ("hip", "cuda"): _HIP_TO_CUDA,
}


def _rewrite_headers(source: str, rules: _Rules) -> str:
    """换 ``#include`` 里的头文件名；未登记的头文件原样保留。"""
    if "#" not in source:
        return source
    lines = source.splitlines(keepends=True)
    for index, line in enumerate(lines):
        match = _INCLUDE.match(line)
        if not match:
            continue
        name = match.group("name")
        replacement = rules.headers.get(name)
        if replacement is not None:
            lines[index] = line.replace(name, replacement, 1)
    return "".join(lines)


def _line_of(source: str, offset: int, line_starts: Iterable[int]) -> int:
    return sum(1 for start in line_starts if start <= offset)


def translate(
    source: str,
    *,
    source_dialect: str,
    target_dialect: str,
) -> str:
    """把 ``source`` 从 ``source_dialect`` 换成 ``target_dialect`` 的源码。

    - 同方言：原样返回（不扫描、不改写）——"复制到另一台机器"不该因为
      源码里有 ``asm`` 而失败。
    - 跨方言：只替换登记过的标识符与头文件名；遇到拒收项立刻抛
      ``TranslationError``，错误里带行号与原因。
    - 注释、字符串、字符字面量一律不改：它们不是代码。
    """
    if not isinstance(source, str):
        raise TypeError(f"source 必须是字符串，得到 {type(source).__name__}")
    for label, value in (("source_dialect", source_dialect), ("target_dialect", target_dialect)):
        if value not in SUPPORTED_DIALECTS:
            raise ValueError(
                f"{label} 必须是 {SUPPORTED_DIALECTS} 之一，得到 {value!r}"
            )
    if source_dialect == target_dialect:
        return source

    rules = _RULES[(source_dialect, target_dialect)]
    text = _rewrite_headers(source, rules)
    line_starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            line_starts.append(index + 1)

    pieces = []
    position = 0
    for match in _TOKEN.finditer(text):
        kind = match.lastgroup
        if kind in ("comment", "string", "char"):
            continue
        token = match.group()
        reason = rules.refuse.get(token)
        if reason is not None:
            line = _line_of(text, match.start(), line_starts)
            raise TranslationError(
                f"第 {line} 行的 {token} 不能被转译到 {target_dialect}：{reason}"
            )
        replacement = rules.identifiers.get(token)
        if replacement is None or replacement == token:
            continue
        pieces.append(text[position : match.start()])
        pieces.append(replacement)
        position = match.end()
    if not pieces:
        return text
    pieces.append(text[position:])
    return "".join(pieces)


def _backend_dialect(backend) -> Optional[str]:
    """从后端的显示标签推断方言；推断不出就返回 ``None``（不做校验）。"""
    label = getattr(backend, "_LABEL", "")
    for name, token in (("cuda", "CUDA"), ("hip", "HIP")):
        if token in label:
            return name
    return None


def compile_translated(
    backend,
    source: str,
    func: str,
    *,
    source_dialect: str,
    target_dialect: Optional[str] = None,
    key: Optional[object] = None,
):
    """转译后走后端自己的编译缓存。

    编译缓存键包含**原文**的 sha256 与两个方言名：源码改一个字符就会重新
    编译，而同一份原文转译两次命中同一个句柄。``target_dialect`` 缺省按后端
    的显示标签推断；显式给出时若与后端不符，当场报错——把 HIP 源码交给
    NVRTC 只会得到一句来自编译器的、看不出根因的失败。
    """
    resolved = _backend_dialect(backend) if target_dialect is None else target_dialect
    if resolved is None:
        raise ValueError(
            f"无法从 {type(backend).__name__} 推断目标方言，请显式给出 target_dialect"
        )
    actual = _backend_dialect(backend)
    if actual is not None and actual != resolved:
        raise ValueError(
            f"目标方言是 {resolved!r}，但 {type(backend).__name__} 是 {actual!r} 后端；"
            "跨厂商源码只能交给对应厂商的后端编译"
        )
    handle = getattr(backend, "kernel", None)
    if not callable(handle):
        raise TypeError(f"{type(backend).__name__} 没有实现 CodegenContract.kernel()")
    translated = translate(
        source, source_dialect=source_dialect, target_dialect=resolved
    )
    cache_key = (
        "translate",
        source_dialect,
        resolved,
        hashlib.sha256(source.encode("utf-8")).hexdigest(),
        func,
    )
    if key is not None:
        cache_key = (cache_key, key)
    return backend.kernel(cache_key, translated, func)
