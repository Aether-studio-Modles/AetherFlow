"""HIP 后端插件（ROCm）。

``ctypes`` 是标准库：本模块层面不加载任何第三方包，所以没有 ROCm 的机器上
``import aetherflow.backends.hip`` 仍然可用，只是 ``HipBackend()`` 会带着
明确原因失败。
"""

from __future__ import annotations

from . import runtime
from .backend import HIP_OPS, HipBackend

HIP_AVAILABLE = runtime.available()

__all__ = ["HipBackend", "HIP_OPS", "HIP_AVAILABLE"]
