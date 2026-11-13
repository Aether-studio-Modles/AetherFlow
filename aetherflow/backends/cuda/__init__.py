"""CUDA 后端插件。

``cupy`` 是可选依赖：在本模块层面不导入它，
所以没有 CUDA 的机器上 ``import aetherflow.backends.cuda`` 仍然可用，
只是 ``CudaBackend()`` 会带着明确原因失败。
"""

from __future__ import annotations

import importlib.util

from .backend import CUDA_OPS, CudaBackend

CUDA_AVAILABLE = importlib.util.find_spec("cupy") is not None

__all__ = ["CudaBackend", "CUDA_OPS", "CUDA_AVAILABLE"]
