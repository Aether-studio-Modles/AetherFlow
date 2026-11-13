"""hipRTC 编译（ctypes）。

与 CUDA 侧的 NVRTC 完全对应：目标机只需要驱动，不需要装整套编译器。
ROCm 把这个能力放在 ``libhiprtc`` 里，编译产物是一段 AMDGPU 的 ELF
code object，再由 ``hipModuleLoadData`` 装进当前上下文。

这里只做"源码 -> code object"，不碰设备与内存：模块加载、函数查询、
参数打包都在 ``runtime.py``。分开的理由很实际——编译可以在没有 GPU 的
机器上跑（只要有 hipRTC 与目标架构），测试因此能覆盖语法与链接。
"""

from __future__ import annotations

import ctypes
import ctypes.util

_LIB_CANDIDATES = (
    "libhiprtc.so.7",
    "libhiprtc.so.6",
    "libhiprtc.so",
)

# 默认编译选项。``--offload-arch`` 由调用方按设备填；这里只放语言标准。
_BASE_OPTIONS = ("-std=c++17",)


class HipCompileError(RuntimeError):
    """hipRTC 编译失败。消息里带编译器日志的尾部，便于定位。"""


def _load_library() -> ctypes.CDLL:
    names = []
    found = ctypes.util.find_library("hiprtc")
    if found:
        names.append(found)
    names.extend(_LIB_CANDIDATES)
    tried = []
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError as exc:
            tried.append(f"{name}: {exc}")
    raise OSError("找不到 hipRTC 动态库，尝试过：\n  " + "\n  ".join(tried))


def available() -> bool:
    """进程里能否拿到 hipRTC（不需要 GPU）。"""
    try:
        _load_library()
    except OSError:
        return False
    return True


class Rtc:
    """一个进程内复用的 hipRTC 会话。

    hipRTC 的 program 对象是"一次编译"的单位，编译失败后不能复用
    （重复编译同一个 program 会重复链接，报 duplicate symbol），
    因此每次 :meth:`compile` 都新建一个 program 并立刻销毁。
    """

    def __init__(self) -> None:
        self._lib = _load_library()
        vp = ctypes.c_void_p
        cp = ctypes.POINTER(ctypes.c_char_p)
        sz = ctypes.POINTER(ctypes.c_size_t)
        self._lib.hiprtcCreateProgram.argtypes = [
            ctypes.POINTER(vp), ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_int, cp, cp,
        ]
        self._lib.hiprtcCreateProgram.restype = ctypes.c_int
        self._lib.hiprtcCompileProgram.argtypes = [vp, ctypes.c_int, cp]
        self._lib.hiprtcCompileProgram.restype = ctypes.c_int
        self._lib.hiprtcGetProgramLogSize.argtypes = [vp, sz]
        self._lib.hiprtcGetProgramLogSize.restype = ctypes.c_int
        self._lib.hiprtcGetProgramLog.argtypes = [vp, ctypes.c_char_p]
        self._lib.hiprtcGetProgramLog.restype = ctypes.c_int
        self._lib.hiprtcGetCodeSize.argtypes = [vp, sz]
        self._lib.hiprtcGetCodeSize.restype = ctypes.c_int
        self._lib.hiprtcGetCode.argtypes = [vp, ctypes.c_char_p]
        self._lib.hiprtcGetCode.restype = ctypes.c_int
        self._lib.hiprtcDestroyProgram.argtypes = [ctypes.POINTER(vp)]
        self._lib.hiprtcDestroyProgram.restype = ctypes.c_int

    def compile(self, source: str, name: str, arch: str | None = None) -> bytes:
        """把源码编译成 code object 字节串。

        ``arch`` 为空时交给 hipRTC 自己决定（它会按驱动报告的默认架构编译）；
        显式给出 ``gfx1100`` 这类名字更可控，也让跨架构错误在编译期暴露。
        """
        lib = self._lib
        prog = ctypes.c_void_p()
        rc = lib.hiprtcCreateProgram(
            ctypes.byref(prog), source.encode(), f"{name}.hip".encode(), 0, None, None
        )
        if rc != 0:
            raise HipCompileError(f"hiprtcCreateProgram 失败（code={rc}）")
        try:
            options = list(_BASE_OPTIONS)
            if arch:
                options.append(f"--offload-arch={arch}")
            arr = (ctypes.c_char_p * len(options))(*[o.encode() for o in options])
            rc = lib.hiprtcCompileProgram(prog, len(options), arr)
            if rc != 0:
                raise HipCompileError(
                    f"hipRTC 编译 {name} 失败（code={rc}）：\n{self._log(prog)}"
                )
            size = ctypes.c_size_t()
            if lib.hiprtcGetCodeSize(prog, ctypes.byref(size)) != 0:
                raise HipCompileError(f"hiprtcGetCodeSize 失败（{name}）")
            code = ctypes.create_string_buffer(size.value)
            if lib.hiprtcGetCode(prog, code) != 0:
                raise HipCompileError(f"hiprtcGetCode 失败（{name}）")
            return code.raw
        finally:
            lib.hiprtcDestroyProgram(ctypes.byref(prog))

    def _log(self, prog, limit: int = 4000) -> str:
        size = ctypes.c_size_t()
        self._lib.hiprtcGetProgramLogSize(prog, ctypes.byref(size))
        buf = ctypes.create_string_buffer(size.value + 1)
        self._lib.hiprtcGetProgramLog(prog, buf)
        text = buf.value.decode(errors="replace")
        # 编译器日志可能几千行，只保留尾部：错误的上下文通常在最后
        return text[-limit:]
