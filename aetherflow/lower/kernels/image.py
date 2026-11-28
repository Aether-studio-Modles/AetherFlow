"""图像寻址的公共片段：展平下标拆分、通道切片与窗口取样。

两套布局共用同一组名字（``n`` / ``c`` / ``ih`` / ``iw`` / ``base``）：

- NCHW（默认）：``((n*C + c)*H + h)*W + w``，通道是第二维，同通道的相邻
  像素在内存里连续；
- NHWC（5.334）：``((n*H + h)*W + w)*C + c``，通道是末维，同一像素的相邻
  通道连续——卷积内层循环沿 ``cc`` 走时因此变成连续访存，这正是 NHWC 的
  收益来源。

两套函数都只生成下标表达式，不含任何布局判断；选哪套由内核描述在构造时
决定，执行期不再分支。
"""

from __future__ import annotations

from ..kernel import Bin, Expr, Index, Ptr, Var
from ..indexing import Layout


_IMAGE_LAYOUT = Layout(("n", "c", "h", "w"), (None, "C", "H", "W"))


_TAP_LAYOUT = Layout(("h", "w"), (None, "W"))


# NHWC：批的步长是 H*W*C，像素内通道偏移就是 c 本身（通道连续）。
_IMAGE_LAYOUT_NHWC = Layout(("n", "h", "w", "c"), (None, "H", "W", "C"))


_TAP_LAYOUT_NHWC = Layout(("h", "w", "c"), (None, "W", "C"))


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


def _split_output_index_nhwc(x_var: str, y_var: str, x_dim: str, y_dim: str) -> tuple:
    """把展平的输出下标拆成 ``(n, y, x, c)``（按 ``(N, Y, X, C)`` 行优先）。

    名字约定与 :func:`_split_output_index` 相同，只是 ``c`` 挪到了**最后**
    一段：``idx -> c`` 取余，``idx / C -> (n, y, x)``。调用方后面的
    ``_image_slice_nhwc`` / ``_tap_nhwc`` 依赖同样的名字。
    """
    return Layout(
        ("n", y_var, x_var, "c"),
        (None, Var(y_dim), Var(x_dim), Var("C")),
    ).unravel_steps(Var("idx"), ("n", y_var, x_var, "c"))


def _image_slice_nhwc(buf: str, name: str = "base", *, const: bool = True) -> Ptr:
    """``x + (long long)n * H * W * C + c``

    ``n`` 的步长是 ``H*W*C``（``_IMAGE_LAYOUT_NHWC.offset(("n",))``），通道
    偏移直接加 ``c``——它不是下一维，而是一个与像素偏移可交换的加数，所以
    这里的指针与 NCHW 版一样只依赖 ``(n, c)``，窗口内的 h/w 偏移由 tap 提供。
    """
    return Ptr(
        name,
        buf,
        Bin("+", _IMAGE_LAYOUT_NHWC.offset(("n",)), Var("c")),
        const=const,
    )


def _tap_nhwc(buf: str = "base") -> Expr:
    """``buf[((long long)ih * W + iw) * C]``——同一像素的通道连续存放。"""
    return Index(buf, _TAP_LAYOUT_NHWC.offset(("ih", "iw")))
