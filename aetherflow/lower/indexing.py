"""语义级 L2 的索引层：把「多维坐标 ↔ 扁平下标」收成纯函数。

5.42 的 L2 是语法级的：``Index`` 只认一维下标，多维张量怎么线性化由每个内核
手写 ``Bin("*")``。手写本身没有错，但同一套行主序规则在十个内核里重复，而
打印器**不推断优先级**——漏一个 ``Paren`` 就会打印成 ``a + b * c``，语义测试
未必接得住。

这一层只做三件事：

- ``Layout`` 描述一个行主序张量：维度名 + 每维长度，最高维长度可以省略
  （它只参与「商」，不参与乘法）；
- ``Layout.offset(coords)`` 生成「坐标 -> 扁平下标」的表达式；
- ``Layout.unravel_steps(flat, names)`` 生成它的逆：把扁平下标拆成坐标的一串
  ``Let`` / ``AugAssign``。

**它不决定布局**——分块、向量化、合并访存仍由 ``plan.py`` 与各内核自己决定。
这里只保证同一套坐标打印出同一条下标表达式，且括号显式。

一个刻意的边界：乘积链只在**首因子**上做 64 位提升。现有内核
（``_image_slice`` / ``_tap``）就是这么写的，而 ``int * int`` 在
``(long long)H * W`` 这种形状上会溢出。调用方要换类型就传 ``cast=None``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

from .kernel import AugAssign, Bin, Cast, Expr, Let, Num, Paren, Stmt, Var

__all__ = ["Layout"]

Coord = Union[str, Expr]


def _expr(value: Coord) -> Expr:
    """允许调用方用维度名代替 ``Var``——尺寸与坐标都是运行期标量。"""
    return Var(value) if isinstance(value, str) else value


def _checked(value: Coord) -> Expr:
    """常量尺寸必须是正整数。运行期尺寸（``Var``）无法检查，交给内核。"""
    e = _expr(value)
    if isinstance(e, Num) and e.text.isdigit() and int(e.text) < 1:
        raise ValueError(f"布局的维度长度必须是正整数，收到 {e.text}")
    return e


@dataclass(frozen=True)
class Layout:
    """行主序、元素为单位的连续布局。

    ``sizes[i]`` 是第 ``i`` 维的长度，最高维（``i == 0``）可以给 ``None``：
    线性化只需要 ``sizes[1:]``，最高维只出现在解线性化的最后一次除法里，
    而那里除的是第二维的长度。
    """

    dims: Tuple[str, ...]
    sizes: Tuple[Optional[Coord], ...]

    def __post_init__(self) -> None:
        if not self.dims:
            raise ValueError("布局至少要有一维")
        if len(self.dims) != len(self.sizes):
            raise ValueError(
                f"维度名有 {len(self.dims)} 个，长度有 {len(self.sizes)} 个"
            )
        for i, size in enumerate(self.sizes):
            if i == 0:
                continue
            if size is None:
                raise ValueError(f"第 {i} 维（{self.dims[i]}）缺少长度")
            _checked(size)

    def _size(self, i: int) -> Expr:
        size = self.sizes[i]
        if size is None:
            raise ValueError(f"第 {i} 维（{self.dims[i]}）没有长度，不能参与线性化")
        return _expr(size)

    @staticmethod
    def _cast(e: Expr, cast: Optional[str]) -> Expr:
        """乘积链的首因子提升到 64 位；已经是常量或转换过的不再套一层。"""
        if cast is None or isinstance(e, (Num, Cast)):
            return e
        return Cast(cast, e)

    def _product(self, start: int, cast: Optional[str]) -> Expr:
        """``sizes[start] * sizes[start+1] * ...``——左结合，首因子提升。"""
        acc = self._cast(self._size(start), cast)
        for i in range(start + 1, len(self.dims)):
            acc = Bin("*", acc, self._size(i))
        return acc

    def offset(self, coords: Sequence[Coord], *, cast: Optional[str] = "long long") -> Expr:
        """把前 ``len(coords)`` 维的坐标折成扁平下标。

        折叠形态是 ``((c0 * S1 + c1) * S2 + c2)``；坐标少于维数时，剩下的
        维度当作 0，末尾再乘它们的长度乘积（``_image_slice`` 的
        ``(n * C + c) * H * W`` 就是这一形态）。
        """
        points = tuple(_expr(c) for c in coords)
        if not points:
            raise ValueError("至少需要一个坐标")
        if len(points) > len(self.dims):
            raise ValueError(
                f"坐标有 {len(points)} 个，布局只有 {len(self.dims)} 维"
            )
        acc = self._cast(points[0], cast)
        for i in range(1, len(points)):
            acc = Bin("+", Bin("*", acc, self._size(i)), points[i])
        if len(points) < len(self.dims):
            # 折叠后的和要整体乘剩余维的乘积，括号必须显式写出来：
            # 打印器不推断优先级，`a + b * c` 与 `(a + b) * c` 是两回事。
            acc = Bin("*", Paren(acc), self._product(len(points), cast))
        return acc

    def unravel_steps(
        self,
        flat: Coord,
        names: Sequence[str],
        *,
        coord_type: str = "int",
        temp_type: str = "long long",
        temp: str = "t",
    ) -> Tuple[Stmt, ...]:
        """生成把扁平下标拆成坐标的语句序列（行主序，末维先拆）。

        形态与 ``_split_output_index`` 一致：

        ```
        int x  = (int)(flat % Sx);
        long long t = flat / Sx;
        int y  = (int)(t % Sy);
        t /= Sy;
        int n  = t / Sc;
        ```

        中间维先取模再整除；第二维只取模，最高维直接取商——所以最高维的
        长度不必给出。
        """
        n = len(self.dims)
        if len(names) != n:
            raise ValueError(f"坐标名有 {len(names)} 个，布局有 {n} 维")
        if n < 2:
            raise ValueError("解线性化至少要两维；一维下标本来就是坐标本身")
        source = _expr(flat)
        last = self._size(n - 1)
        steps = [
            Let(coord_type, names[-1], Cast(coord_type, Paren(Bin("%", source, last)))),
            Let(temp_type, temp, Bin("/", source, last)),
        ]
        for i in range(n - 2, 1, -1):
            steps.append(
                Let(
                    coord_type,
                    names[i],
                    Cast(coord_type, Paren(Bin("%", Var(temp), self._size(i)))),
                )
            )
            steps.append(AugAssign(Var(temp), "/", self._size(i)))
        steps.append(
            Let(
                coord_type,
                names[1],
                Cast(coord_type, Paren(Bin("%", Var(temp), self._size(1)))),
            )
        )
        steps.append(Let(temp_type, names[0], Bin("/", Var(temp), self._size(1))))
        return tuple(steps)
