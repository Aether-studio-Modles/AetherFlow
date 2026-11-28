"""执行期与离线搜索共用的算子签名。

配置库（``tuning.store.ConfigStore``）的命中判据是四段键，第四段是"这条
配置在什么形状 / 属性下测出来的"。如果离线脚本拼一套键、执行期拼另一套，
两边永远不相等——库里有记录、运行时不命中，看起来就像"调优没有生效"。
本模块把签名收敛成唯一入口：

- :func:`conv2d_signature`：输入 / 权重形状、步长、填充、膨胀、分组、dtype；
- :func:`block_signature`：逐元素 / 归约类按 launch 规模选 block 的签名。

字段只包含**影响内核选择或线程铺法**的量：设备指针、随机种子、时间戳这类
每次运行都变的东西一旦进签名，配置就再也不会命中了。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from .space import canonical_signature

__all__ = ["conv2d_signature", "block_signature"]


def _pair(name: str, value: Sequence[int]) -> list:
    if isinstance(value, (str, bytes)) or len(tuple(value)) != 2:
        raise ValueError(f"{name} 必须是两个整数，收到 {value!r}")
    return [int(item) for item in value]


def conv2d_signature(
    x_shape: Sequence[int],
    weight_shape: Sequence[int],
    *,
    stride: Sequence[int] = (1, 1),
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    groups: int = 1,
    dtype: Optional[Any] = None,
) -> Dict[str, Any]:
    """卷积的规范签名；形状必须已是静态的 4 维。

    分组数参与签名：同样的 ``x`` / ``weight`` 形状，分组不同意味着不同的
    内核路径（逐组累加 vs 全额乘加），配置不能跨组复用。
    """
    x = tuple(int(value) for value in x_shape)
    weight = tuple(int(value) for value in weight_shape)
    if len(x) != 4:
        raise ValueError(f"conv2d 的输入必须是 4 维，收到 {x}")
    if len(weight) != 4:
        raise ValueError(f"conv2d 的权重必须是 4 维，收到 {weight}")
    if isinstance(groups, bool) or not isinstance(groups, int) or groups < 1:
        raise ValueError(f"groups 必须是正整数，收到 {groups!r}")
    return canonical_signature(
        dtype=None if dtype is None else str(dtype),
        x=list(x),
        weight=list(weight),
        stride=_pair("stride", stride),
        padding=_pair("padding", padding),
        dilation=_pair("dilation", dilation),
        groups=groups,
    )


def block_signature(
    op: str, *, work: int, vec: int = 1, dtype: Optional[Any] = None
) -> Dict[str, Any]:
    """按 launch 规模拧 block 的签名：``work`` 是元素数 / 归约列数。

    不含 ``block`` 本身：签名回答"在什么规模下测的"，配置才是"测出来的
    block 是多少"。把结果放进签名会让每个候选配置各自形成一条记录。
    """
    if not isinstance(op, str) or not op:
        raise ValueError(f"算子名必须是非空字符串，收到 {op!r}")
    if isinstance(work, bool) or not isinstance(work, int) or work < 0:
        raise ValueError(f"work 必须是非负整数，收到 {work!r}")
    if isinstance(vec, bool) or not isinstance(vec, int) or vec < 1:
        raise ValueError(f"vec 必须是正整数，收到 {vec!r}")
    return canonical_signature(
        dtype=None if dtype is None else str(dtype),
        work=work,
        vec=vec,
    )
