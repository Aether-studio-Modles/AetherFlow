"""torch 兼容矩阵：把"哪些 torch 版本 / dtype 验证过"写成可查询的声明。

前端的对齐测试一直只在**一个** torch 版本上跑过；版本一换，``torch.export``
的 IR 细节（动态维的表示、dtype 名、算子拆分方式）就可能变。与其在调用方
那里留下一句"应该也支持吧"，不如把矩阵写死在代码里：

- 版本落在矩阵的某一行内才允许导出，否则抛 :class:`UnsupportedTorchVersion`
  并说明支持范围；
- dtype 不在该行声明的集合里就显式拒绝，不做"能跑就行"的猜测；
- ``verified`` 记录真正跑过测试的版本，``notes`` 写清楚哪个区间只是声明。

矩阵之外的版本**不是**"大概率能用"，而是"没有验证过"——两者的区别正是这张
表存在的意义。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Tuple

__all__ = [
    "TorchCompatibility",
    "TORCH_MATRIX",
    "UnsupportedTorchVersion",
    "parse_torch_version",
    "matrix_row",
    "check_torch",
]

_VERSION = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?")


class UnsupportedTorchVersion(RuntimeError):
    """安装的 torch 版本不在兼容矩阵内。

    刻意继承 ``RuntimeError``：这是**环境**不匹配，不是参数写错。调用方该做
    的是换版本，或者在矩阵里显式登记并补上验证。
    """


@dataclass(frozen=True)
class TorchCompatibility:
    """矩阵的一行：一个 torch 版本区间 + 该区间下已验证的能力。"""

    min_version: Tuple[int, int]
    max_version: Tuple[int, int]
    dtypes: Tuple[str, ...]
    verified: Tuple[str, ...]
    notes: str

    def covers(self, version: Tuple[int, int, int]) -> bool:
        return self.min_version <= version[:2] <= self.max_version

    def accepts_dtype(self, name: str) -> bool:
        return name in self.dtypes

    @property
    def span(self) -> str:
        low = ".".join(str(part) for part in self.min_version)
        high = ".".join(str(part) for part in self.max_version)
        return low if low == high else f"{low}–{high}"


#: 目前唯一的验证环境是 torch 2.13.0（py3.14 / CPU / Windows）。2.9 的导出
#: 形态曾经用来做过一次手工核对（5.152），但没有留下可复跑的用例，因此只写进
#: 范围、不标 verified——"跑过"和"记着跑过"不是一回事。
TORCH_MATRIX: Tuple[TorchCompatibility, ...] = (
    TorchCompatibility(
        min_version=(2, 9),
        max_version=(2, 13),
        dtypes=("float16", "float32", "float64", "bfloat16", "int32", "int64", "bool"),
        verified=("2.13.0",),
        notes=(
            "verified 列出的版本由 tests/test_frontend.py 实跑；"
            "区间内其余版本只有声明，没有逐版本回归。int64 按 IR 的 i32 承载"
            "（IR 没有 i64），越界值在执行器入口拒绝"
        ),
    ),
)


def parse_torch_version(text: str) -> Tuple[int, int, int]:
    """把 ``2.13.0+cpu`` / ``2.9.0a0`` 之类的版本串解析成三元组。"""
    match = _VERSION.match(str(text).strip())
    if match is None:
        raise ValueError(f"无法解析 torch 版本号：{text!r}")
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3) or 0),
    )


def matrix_row(version: Tuple[int, int, int]) -> TorchCompatibility:
    """返回覆盖该版本的矩阵行；没有覆盖行时显式报错。"""
    for row in TORCH_MATRIX:
        if row.covers(version):
            return row
    spans = "、".join(row.span for row in TORCH_MATRIX)
    raise UnsupportedTorchVersion(
        f"torch {'.'.join(str(part) for part in version)} 不在兼容矩阵内："
        f"已验证的区间是 {spans}；请换到区间内的版本，或在 TORCH_MATRIX 里"
        "登记新版本并补上回归用例"
    )


def check_torch(module: Optional[Any] = None) -> TorchCompatibility:
    """校验（安装的）torch 版本，返回它对应的矩阵行。"""
    if module is None:
        try:
            import torch as module  # noqa: PLC0415 - 可选依赖，延迟导入
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise ImportError(
                "需要 torch 才能做前端导入导出；torch 是可选依赖"
            ) from exc
    version = parse_torch_version(getattr(module, "__version__", ""))
    return matrix_row(version)
