"""NCHW 图像寻址的公共片段：展平下标拆分、通道切片与窗口取样。"""

from __future__ import annotations

from ..kernel import Expr, Index, Ptr, Var
from ..indexing import Layout


_IMAGE_LAYOUT = Layout(("n", "c", "h", "w"), (None, "C", "H", "W"))


_TAP_LAYOUT = Layout(("h", "w"), (None, "W"))


def _split_output_index(x_var: str, y_var: str, x_dim: str, y_dim: str) -> tuple:
    """把展平的输出下标拆成 ``(n, c, y, x)``（按 ``(N, C, Y, X)`` 行优先）。

    生成的名字是**约定**而不是参数：中间量固定叫 ``t``、通道叫 ``c``、批叫
    ``n``，因为紧随其后的 ``_image_slice`` 就按这几个名字取。卷积权重梯度
    不能用它——那里的末轴是"输出通道 ``oi``"，语义上是纯除法、类型是
    ``int``，与本骨架的最后两行不同（见 ``conv2d_weight_grad``）。
    """
    return Layout(
        ("n", "c", y_var, x_var),
        (None, Var("C"), Var(y_dim), Var(x_dim)),
    ).unravel_steps(Var("idx"), ("n", "c", y_var, x_var))


def _image_slice(buf: str, name: str = "base", *, const: bool = True) -> Ptr:
    """``x + ((long long)n * C + c) * (long long)H * W``"""
    return Ptr(name, buf, _IMAGE_LAYOUT.offset(("n", "c")), const=const)


def _tap(buf: str = "base") -> Expr:
    """``buf[(long long)ih * W + iw]``"""
    return Index(buf, _TAP_LAYOUT.offset(("ih", "iw")))
