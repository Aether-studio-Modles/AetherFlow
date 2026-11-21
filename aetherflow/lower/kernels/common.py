"""全部内核共用的骨架：网格步进、展平循环与二元算符表。"""

from __future__ import annotations

from ..kernel import Axis, Bin, Cast, Expr, For, Let, Stmt, Var


BINARY_OP = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def grid_stride(limit: Expr, *body: Stmt) -> tuple:
    """一维网格步进的块内索引骨架，全部逐元素内核共用。

    ``i = blockIdx*blockDim + tid``、``stride = gridDim*blockDim``，
    然后 ``for (; i < limit; i += stride)``。三个量缺一不可：

    - 索引用 64 位：``blockIdx*blockDim`` 在超大张量上会越过 32 位；
    - stride 必须是运行期的网格大小——块数被 ``max_blocks`` 截断时，
      多出来的元素只能靠步进补上，写成 ``blockDim`` 会漏算；
    - ``limit`` 交给调用方：标量路径是元素数 ``n``，向量路径是
      ``n / 4``（索引是向量下标）。

    ``elementwise.build_chain`` 用的是同一个骨架，所以融合链与单算子内核
    的循环形状逐字相同。
    """
    return (
        Let(
            "long long",
            "i",
            Bin(
                "+",
                Bin("*", Cast("long long", Axis("bid")), Axis("bdim")),
                Axis("tid"),
            ),
        ),
        Let(
            "long long",
            "stride",
            Bin("*", Cast("long long", Axis("gdim")), Axis("bdim")),
        ),
        For("i", None, limit, Var("stride"), tuple(body)),
    )


def _flat_loop(
    body: tuple, *, index: str = "idx", count: str = "total", stride: str = "step"
) -> tuple:
    """``step = ...; for (long long idx = ...; idx < total; idx += step)``"""
    return (
        Let(
            "const long long",
            stride,
            Bin("*", Cast("long long", Axis("gdim")), Axis("bdim")),
        ),
        For(
            index,
            Bin(
                "+",
                Bin("*", Cast("long long", Axis("bid")), Axis("bdim")),
                Axis("tid"),
            ),
            Var(count),
            Var(stride),
            body,
        ),
    )
