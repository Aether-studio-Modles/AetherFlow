"""语义级索引层：坐标与扁平下标必须互为逆运算。

这些用例不打印内核，而是把生成的表达式**求值**——括号位置错一位，打印出来
仍然"看着对"，只有算一遍才知道 ``(a + b) * c`` 被写成了 ``a + b * c``。
"""

from __future__ import annotations

import random
import unittest

from aetherflow.lower.indexing import Layout
from aetherflow.lower.kernel import (
    AugAssign,
    Bin,
    Cast,
    Let,
    Neg,
    Num,
    Paren,
    Var,
)


def _eval(e, env):
    """把 L2 表达式当整数表达式求值。只覆盖索引层会用到的节点。"""
    if isinstance(e, Var):
        return env[e.name]
    if isinstance(e, Num):
        return int(e.text)
    if isinstance(e, (Paren, Cast)):
        return _eval(e.inner, env)
    if isinstance(e, Neg):
        return -_eval(e.inner, env)
    if isinstance(e, Bin):
        left = _eval(e.lhs, env)
        right = _eval(e.rhs, env)
        if e.op == "+":
            return left + right
        if e.op == "-":
            return left - right
        if e.op == "*":
            return left * right
        if e.op == "/":
            return left // right
        if e.op == "%":
            return left % right
    raise TypeError(f"测试求值器不认识 {type(e).__name__}")


def _show(e) -> str:
    """按 C 打印器的规则展开，用来断言括号的**位置**。"""
    if isinstance(e, Var):
        return e.name
    if isinstance(e, Num):
        return e.text
    if isinstance(e, Paren):
        return f"({_show(e.inner)})"
    if isinstance(e, Cast):
        return f"({e.type_name}){_show(e.inner)}"
    if isinstance(e, Bin):
        return f"{_show(e.lhs)} {e.op} {_show(e.rhs)}"
    if isinstance(e, Neg):
        return f"-{_show(e.inner)}"
    raise TypeError(f"测试打印器不认识 {type(e).__name__}")


def _reference_offset(sizes, coords) -> int:
    """行主序参考值：``sum(c[i] * prod(sizes[i+1:]))``。"""
    total = 0
    for i, coord in enumerate(coords):
        stride = 1
        for size in sizes[i + 1:]:
            stride *= size
        total += coord * stride
    return total


class TestLayoutValidation(unittest.TestCase):
    def test_empty_layout_is_rejected(self):
        with self.assertRaises(ValueError):
            Layout((), ())

    def test_dimension_count_must_match(self):
        with self.assertRaises(ValueError):
            Layout(("n", "c"), (None,))

    def test_inner_dimension_needs_a_length(self):
        with self.assertRaises(ValueError):
            Layout(("n", "c"), (None, None))

    def test_constant_length_must_be_positive(self):
        with self.assertRaises(ValueError):
            Layout(("n", "c"), (None, Num("0")))

    def test_too_many_coordinates(self):
        layout = Layout(("n", "c"), (None, "C"))
        with self.assertRaises(ValueError):
            layout.offset(("n", "c", "x"))

    def test_no_coordinates(self):
        with self.assertRaises(ValueError):
            Layout(("n", "c"), (None, "C")).offset(())

    def test_unravel_needs_at_least_two_dimensions(self):
        with self.assertRaises(ValueError):
            Layout(("n",), (None,)).unravel_steps(Var("idx"), ("n",))

    def test_unravel_name_count_must_match(self):
        layout = Layout(("n", "c"), (None, "C"))
        with self.assertRaises(ValueError):
            layout.unravel_steps(Var("idx"), ("n",))


class TestOffset(unittest.TestCase):
    def test_full_coordinates_match_row_major(self):
        layout = Layout(("n", "h", "w"), (None, Num("5"), Num("7")))
        for n in range(3):
            for h in range(5):
                for w in range(7):
                    expr = layout.offset((Num(str(n)), Num(str(h)), Num(str(w))))
                    self.assertEqual(_eval(expr, {}), (n * 5 + h) * 7 + w)

    def test_partial_coordinates_skip_trailing_dimensions(self):
        layout = Layout(("n", "c", "h", "w"), (None, Num("4"), Num("5"), Num("7")))
        expr = layout.offset((Num("2"), Num("3")))
        self.assertEqual(_eval(expr, {}), (2 * 4 + 3) * 5 * 7)

    def test_runtime_sizes_are_read_from_the_environment(self):
        layout = Layout(("n", "c", "h", "w"), (None, "C", "H", "W"))
        expr = layout.offset(("n", "c"))
        env = {"n": 1, "c": 2, "C": 3, "H": 4, "W": 5}
        self.assertEqual(_eval(expr, env), (1 * 3 + 2) * 4 * 5)

    def test_the_prefix_sum_is_parenthesised_before_the_product(self):
        layout = Layout(("n", "c", "h", "w"), (None, "C", "H", "W"))
        self.assertEqual(
            _show(layout.offset(("n", "c"))),
            "((long long)n * C + c) * (long long)H * W",
        )

    def test_two_dimensions_have_no_trailing_parentheses(self):
        layout = Layout(("h", "w"), (None, "W"))
        self.assertEqual(_show(layout.offset(("ih", "iw"))), "(long long)ih * W + iw")

    def test_cast_can_be_turned_off(self):
        layout = Layout(("h", "w"), (None, "W"))
        self.assertEqual(_show(layout.offset(("ih", "iw"), cast=None)), "ih * W + iw")

    def test_random_layouts_round_trip(self):
        rng = random.Random(20261108)
        for _ in range(200):
            ndim = rng.randint(2, 4)
            inner = [rng.randint(1, 9) for _ in range(ndim - 1)]
            dims = tuple(f"d{i}" for i in range(ndim))
            layout = Layout(dims, (None,) + tuple(Num(str(s)) for s in inner))
            coords = tuple(Num(str(rng.randrange(s))) for s in (inner + [inner[-1]]))
            sizes = [None] + inner
            self.assertEqual(
                _eval(layout.offset(coords), {}),
                _reference_offset(sizes, [int(c.text) for c in coords]),
            )


class TestUnravel(unittest.TestCase):
    def _run(self, steps, flat):
        env = {}
        for stmt in steps:
            if isinstance(stmt, Let):
                env[stmt.name] = _eval(stmt.value, env)
            elif isinstance(stmt, AugAssign):
                self.assertEqual(stmt.op, "/")
                env[stmt.target.name] = env[stmt.target.name] // _eval(stmt.value, env)
            else:  # pragma: no cover - 生成器只产出这两种语句
                raise TypeError(type(stmt).__name__)
        return env

    def test_statement_order_is_last_dimension_first(self):
        layout = Layout(("n", "c", "ho", "wo"), (None, "C", "HO", "WO"))
        steps = layout.unravel_steps(Var("idx"), ("n", "c", "ho", "wo"))
        self.assertEqual(
            [type(s).__name__ for s in steps],
            ["Let", "Let", "Let", "AugAssign", "Let", "Let"],
        )
        self.assertEqual(_show(steps[0].value), "(int)(idx % WO)")
        self.assertEqual(_show(steps[1].value), "idx / WO")
        self.assertEqual(_show(steps[2].value), "(int)(t % HO)")
        self.assertEqual(_show(steps[3].value), "HO")
        self.assertEqual(_show(steps[4].value), "(int)(t % C)")
        self.assertEqual(_show(steps[5].value), "t / C")

    def test_round_trip_with_offset(self):
        rng = random.Random(8721)
        for _ in range(50):
            sizes = [rng.randint(1, 6) for _ in range(3)]
            layout = Layout(
                ("n", "a", "b", "c"),
                (None, Num(str(sizes[0])), Num(str(sizes[1])), Num(str(sizes[2]))),
            )
            total = sizes[0] * sizes[1] * sizes[2]
            # 多走几格让最高维（n）也取到非零值：它的长度不在布局里，
            # 但 offset 只用得到它的步长乘积。
            for flat in range(total + 3):
                env = self._run(
                    layout.unravel_steps(Num(str(flat)), ("n", "a", "b", "c")),
                    flat,
                )
                coords = (env["n"], env["a"], env["b"], env["c"])
                self.assertEqual(coords[3], flat % sizes[2])
                self.assertEqual(coords[2], (flat // sizes[2]) % sizes[1])
                self.assertEqual(
                    coords[1], (flat // (sizes[1] * sizes[2])) % sizes[0]
                )
                self.assertEqual(coords[0], flat // total)
                self.assertEqual(
                    _eval(layout.offset(tuple(Num(str(c)) for c in coords)), {}),
                    flat,
                )


class TestProductionLayouts(unittest.TestCase):
    """5.209：生产内核的行主序线性化由 ``Layout`` 生成，且 token 与手写一致。

    这几条直接看渲染出来的源码：抽象层一旦把 ``cast`` 打开或括号加错，
    求值测试未必接得住，但源码会立刻变样。
    """

    def setUp(self):
        from aetherflow.backends.cuda.codegen import render
        from aetherflow.lower import kernels

        self.render = render
        self.kernels = kernels

    def test_sparse_linear_weights_index(self):
        src = self.render(self.kernels.sparse_linear(), "float32")
        self.assertIn("r * cols + n", src)
        self.assertNotIn("(long long)r * cols", src)

    def test_scan_linear_element_index(self):
        src = self.render(self.kernels.scan_linear(), "float32")
        self.assertIn("t * elems + i", src)
        self.assertNotIn("(long long)t * elems", src)

    def test_spike_linear_spike_index(self):
        src = self.render(self.kernels.spike_linear(), "float32")
        self.assertIn("s * rows + r", src)

    def test_production_layouts_are_layout_objects(self):
        for name in ("_ROW_COLS", "_T_ELEMS", "_C_ELEMS", "_S_ROWS", "_T_WIDTH"):
            self.assertIsInstance(getattr(self.kernels, name), Layout)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
