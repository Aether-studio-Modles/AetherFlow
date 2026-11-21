"""打包元数据与代码内声明的一致性。

版本号与 torch 兼容矩阵都写在两处（``pyproject.toml`` 与代码）。这类漂移
最晚才会被发现：wheel 的版本与 ``aetherflow.__version__`` 不一致，或者安装
约束放行了运行时会显式拒绝的 torch。两条都值得在提交前钉住。
"""

import re
import unittest
from pathlib import Path

from aetherflow import __api_version__, __version__
from aetherflow.frontend.compat import TORCH_MATRIX

try:  # Python 3.11+
    import tomllib
except ImportError:  # pragma: no cover - 取决于解释器版本
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(tomllib is not None, "需要 tomllib（3.11+）或 tomli")
class TestPackaging(unittest.TestCase):
    def setUp(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.data = tomllib.loads(text)

    def test_version_is_dynamic_and_matches_code(self):
        project = self.data["project"]
        # 版本只允许有一处事实来源：pyproject 里不能同时写死 version。
        self.assertIn("version", project["dynamic"])
        self.assertNotIn("version", project)
        self.assertRegex(__version__, r"^\d+\.\d+\.\d+")

    def test_api_version_is_a_major_string(self):
        self.assertTrue(__api_version__.isdigit())

    def test_torch_extra_matches_compat_matrix(self):
        """安装约束与运行时矩阵必须同一个区间。

        ``frontend = ["torch>=2.9,<2.14"]`` 对应矩阵的 2.9–2.13：上界开区间
        要写成 max+1，否则安装能装进来、导入却抛 UnsupportedTorchVersion。
        """
        deps = self.data["project"]["optional-dependencies"]["frontend"]
        spec = next(dep for dep in deps if dep.startswith("torch"))
        bounds = re.findall(r"(\d+)\.(\d+)", spec)
        self.assertEqual(len(bounds), 2, spec)
        row = TORCH_MATRIX[0]
        self.assertEqual(
            (int(bounds[0][0]), int(bounds[0][1])),
            row.min_version,
        )
        self.assertEqual(
            (int(bounds[1][0]), int(bounds[1][1])),
            (row.max_version[0], row.max_version[1] + 1),
        )
