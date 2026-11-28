"""示例硬件包的端到端验收。

验收的不是"示例能跑"，而是"接入只需要填方言"：

1. zip 装进目录后，**新进程**仅靠目录扫描就能识别；
2. 后端拿到的内核与 CUDA 方言打印的结构逐字一致，差异只在存取函数名；
3. 换 f16 时只有 prelude 里的存取实现跟着变，主体不变；
4. 图能真的跑出结果（示例用参考执行器代替驱动）。
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aetherflow import (
    Graph,
    f32,
    get_backend,
    install_hardware,
    registered_backends,
    uninstall_hardware,
)
from aetherflow.backends.cuda.dialect import CUDA_DIALECT
from aetherflow.ir.types import DType
from aetherflow.lower import emit, ew_binary

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE = _ROOT / "examples" / "hardware" / "novac"


def _squash(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"//[^\n]*", "", src)
    return re.sub(r"\s+", "", src)


def _normalize(src: str) -> str:
    """把示例方言的存取函数名换回 CUDA 的名字，再比较结构。"""
    for new, old in (
        ("nv_ld_f", "af_ld_f"),
        ("nv_st_f", "af_st_f"),
        ("nv_add_f", "af_add_f"),
        ("nv_rem", "af_rem"),
    ):
        src = src.replace(new, old)
    return _squash(src)


def _build_zip(dest: Path) -> Path:
    """按固定清单打包示例目录（不把输出 zip 自己打进去）。"""
    files = sorted(
        path
        for path in _EXAMPLE.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(_EXAMPLE).as_posix())
    return dest


class TestExampleHardware(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.home = self.dir / "hardware"
        self.home.mkdir()
        patcher = patch.dict(
            os.environ, {"AETHERFLOW_HARDWARE_DIR": str(self.home)}
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._cleanup_modules)
        install_hardware(_build_zip(self.dir / "novac.zip"))
        self.addCleanup(uninstall_hardware, "novac")

    def _cleanup_modules(self):
        root = str(self.dir)
        for module in list(sys.modules.values()):
            origin = getattr(module, "__file__", "") or ""
            if origin.startswith(root):
                sys.modules.pop(module.__name__, None)
        sys.path[:] = [entry for entry in sys.path if not entry.startswith(root)]

    def test_package_is_discovered(self):
        self.assertIn("novac", registered_backends())

    def test_factory_builds_backend(self):
        backend = get_backend("novac", block=256)
        self.assertEqual(backend.name, "novac")
        self.assertEqual(backend.options, {"block": 256})

    def test_new_process_finds_package_by_directory_scan(self):
        """安装不写任何注册表：新进程只靠扫目录就要认得它。"""
        code = (
            "import json;"
            "from aetherflow import registered_backends;"
            "print(json.dumps(sorted(registered_backends())))"
        )
        env = dict(os.environ)
        env["AETHERFLOW_HARDWARE_DIR"] = str(self.home)
        env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("novac", json.loads(result.stdout))

    def test_only_the_dialect_changes(self):
        """同一份 L2 内核，示例方言与 CUDA 的差别只在函数名。"""
        actual = get_backend("novac").source_for(ew_binary("add"), "float32")
        expected = emit(ew_binary("add"), CUDA_DIALECT, DType.F32)
        self.assertEqual(_normalize(actual), _squash(expected))

    def test_f16_uses_the_dialect_helpers(self):
        src = get_backend("novac").source_for(ew_binary("mul"), "float16")
        self.assertIn("__half2float", src)
        self.assertIn("nv_ld_h", src)
        self.assertIn("nv_st_h", src)
        self.assertNotIn("cvt.f32.f16", src)

    def test_execute_runs_a_graph(self):
        backend = get_backend("novac")
        g = Graph("demo")
        a = g.input("a", f32[4])
        b = g.input("b", f32[4])
        g.output(g.add(a, b), "y")

        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = backend.execute(g, {"a": x, "b": x * 2.0})
        np.testing.assert_allclose(out["y"], x * 3.0)


if __name__ == "__main__":
    unittest.main()
