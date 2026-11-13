"""跨图控制流的判据：按数据选支、按数据迭代、嵌套、序列化与参数契约。

这些用例只关心三件事：

1. **只有被选中的分支执行**——未选中分支里的错误不该被触发，否则"控制流"
   就退化成了"两个分支都算一遍再挑一个"；
2. **循环状态封闭**——状态从哪进、从哪出、类型是否一致，全部在构图期钉住，
   运行期用尽 ``max_steps`` 是显式失败而不是静默截断；
3. **接口确定**——同名输入同类型、同名输出同类型同顺序，序列化往返后
   执行结果逐位一致。
"""

import json
import unittest

import numpy as np

from aetherflow import (
    Conditional,
    ControlFlowError,
    Graph,
    GraphRegion,
    ReferenceBackend,
    WhileLoop,
    f32,
    f64,
    from_spec,
    i32,
    to_spec,
)


def _scalar(name="x", dtype=f32):
    return dtype[()]


def _predicate_graph(graph_name="pred"):
    """``step(x)``：x > 0 时为 1.0，否则 0.0。"""
    g = Graph(graph_name)
    x = g.input("x", _scalar())
    g.output(g.step(x, threshold=0.0), "flag")
    return g


def _scale_graph(k, graph_name="scale"):
    g = Graph(graph_name)
    x = g.input("x", _scalar())
    g.output(g.mul(x, g.constant(float(k))), "y")
    return g


def _boom_graph(graph_name="boom"):
    """整数除零：一旦真的执行就会抛 ValueError。用来证明分支是惰性的。"""
    g = Graph(graph_name)
    x = g.input("x", _scalar())
    zero = g.constant(0, dtype=i32, shape=())
    quotient = g.cast(g.div(g.cast(x, dtype="i32"), zero), dtype="f32")
    g.output(quotient, "y")
    return g


def _counter_condition(graph_name="while_cond"):
    """``4.5 - x > 0``，即 x < 4.5。"""
    g = Graph(graph_name)
    x = g.input("x", _scalar())
    limit = g.constant(4.5)
    g.output(g.step(g.sub(limit, x), threshold=0.0), "flag")
    return g


def _counter_body(graph_name="while_body"):
    """x_next = x + step（step 缺省为 1）。"""
    g = Graph(graph_name)
    x = g.input("x", _scalar())
    step = g.input("step", _scalar())
    g.output(g.add(x, step), "x_next")
    return g


def _feed(value):
    return np.asarray(value, dtype=np.float32)


class TestConditional(unittest.TestCase):
    def setUp(self):
        self.cond = Conditional(
            _predicate_graph(),
            _scale_graph(2.0, "then"),
            _scale_graph(-3.0, "otherwise"),
        )

    def test_picks_branch_by_data(self):
        self.assertEqual(self.cond.run({"x": _feed(1.5)})["y"], np.float32(3.0))
        self.assertEqual(self.cond.run({"x": _feed(-1.5)})["y"], np.float32(4.5))

    def test_unselected_branch_is_not_executed(self):
        """else 分支是整数除零；x > 0 时它根本不该被跑。"""
        cond = Conditional(_predicate_graph(), _scale_graph(2.0, "then"), _boom_graph())
        self.assertEqual(cond.run({"x": _feed(1.0)})["y"], np.float32(2.0))
        with self.assertRaises(ValueError):
            cond.run({"x": _feed(-1.0)})

    def test_interface_is_the_then_branch(self):
        self.assertEqual([name for name, _ in self.cond.inputs], ["x"])
        self.assertEqual([name for name, _ in self.cond.outputs], ["y"])
        self.assertEqual(self.cond.outputs[0][1], f32[()])

    def test_branch_outputs_must_agree(self):
        with self.assertRaises(ValueError):
            Conditional(
                _predicate_graph(),
                _scale_graph(2.0, "then"),
                GraphRegion(_different_output_name()),
            )

    def test_branch_output_types_must_agree(self):
        with self.assertRaises(ValueError):
            Conditional(
                _predicate_graph(),
                _scale_graph(2.0, "then"),
                GraphRegion(_f64_graph()),
            )

    def test_predicate_must_be_a_single_scalar(self):
        with self.assertRaises(ValueError):
            Conditional(_two_outputs(), _scale_graph(1.0, "then"), _scale_graph(-1.0, "otherwise"))
        with self.assertRaises(ValueError):
            Conditional(_vector_predicate(), _scale_graph(1.0, "then"), _scale_graph(-1.0, "otherwise"))

    def test_input_types_must_match_across_regions(self):
        with self.assertRaises(ValueError):
            Conditional(
                _predicate_graph(),
                GraphRegion(_f64_input_graph()),
                _scale_graph(-1.0, "otherwise"),
            )

    def test_branches_must_be_graphs_or_regions(self):
        with self.assertRaises(TypeError):
            Conditional(_predicate_graph(), None, _scale_graph(-1.0, "otherwise"))
        with self.assertRaises(TypeError):
            GraphRegion("not a graph")

    def test_missing_input_surfaces_from_the_backend(self):
        with self.assertRaises(KeyError):
            self.cond.run({})

    def test_name_must_be_a_non_empty_string(self):
        with self.assertRaises(ValueError):
            Conditional(
                _predicate_graph(),
                _scale_graph(1.0, "then"),
                _scale_graph(-1.0, "otherwise"),
                name="",
            )

    def test_nested_conditional(self):
        inner = Conditional(
            _predicate_graph("inner_pred"),
            _scale_graph(10.0, "inner_then"),
            _scale_graph(20.0, "inner_else"),
        )
        outer = Conditional(_predicate_graph("outer_pred"), inner, _scale_graph(-1.0, "outer_else"))
        self.assertEqual(outer.run({"x": _feed(1.0)})["y"], np.float32(10.0))
        # 只有一只 x，内外两层谓词看到的是同一个值：内层不执行，走外层 else
        self.assertEqual(outer.run({"x": _feed(-1.0)})["y"], np.float32(1.0))


class TestWhileLoop(unittest.TestCase):
    def _loop(self, max_steps=16):
        return WhileLoop(
            _counter_condition(),
            _counter_body(),
            carried={"x": "x_next"},
            max_steps=max_steps,
        )

    def test_counts_until_the_condition_goes_false(self):
        loop = self._loop()
        result = loop.run({"x": _feed(0.0), "step": _feed(1.0)})
        self.assertEqual(result["x"], np.float32(5.0))

        expected = 0.0
        while expected < 4.5:
            expected += 0.5
        half = loop.run({"x": _feed(0.0), "step": _feed(0.5)})
        self.assertEqual(half["x"], np.float32(expected))

    def test_interface_hides_the_carried_state(self):
        loop = self._loop()
        self.assertEqual([name for name, _ in loop.inputs], ["step"])
        self.assertEqual([name for name, _ in loop.outputs], ["x"])

    def test_exhausting_max_steps_is_fatal(self):
        loop = self._loop(max_steps=2)
        with self.assertRaises(ControlFlowError) as ctx:
            loop.run({"x": _feed(0.0), "step": _feed(1.0)})
        self.assertIn("max_steps", str(ctx.exception))

    def test_missing_initial_state_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self._loop().run({"step": _feed(1.0)})
        self.assertIn("初始状态", str(ctx.exception))

    def test_carried_must_be_a_mapping_with_valid_ends(self):
        with self.assertRaises(TypeError):
            WhileLoop(_counter_condition(), _counter_body(), carried=("x",), max_steps=4)
        with self.assertRaises(ValueError):
            WhileLoop(_counter_condition(), _counter_body(), carried={}, max_steps=4)
        with self.assertRaises(ValueError):
            WhileLoop(
                _counter_condition(), _counter_body(), carried={"x": "x"}, max_steps=4
            )

    def test_carried_must_be_an_input_and_an_output_of_the_body(self):
        with self.assertRaises(ValueError):
            WhileLoop(
                _counter_condition(), _counter_body(), carried={"missing": "x_next"}, max_steps=4
            )
        with self.assertRaises(ValueError):
            WhileLoop(
                _counter_condition(), _counter_body(), carried={"x": "missing"}, max_steps=4
            )

    def test_carried_types_must_round_trip(self):
        body = Graph("f64_body")
        x = body.input("x", f32[()])
        body.output(body.cast(body.add(x, body.constant(1.0)), dtype="f64"), "x_next")
        with self.assertRaises(ValueError):
            WhileLoop(_counter_condition(), body, carried={"x": "x_next"}, max_steps=4)

    def test_max_steps_must_be_a_positive_int(self):
        for bad in (0, -1, True, 1.5, "4"):
            with self.subTest(max_steps=bad):
                with self.assertRaises(ValueError):
                    WhileLoop(
                        _counter_condition(), _counter_body(), carried={"x": "x_next"}, max_steps=bad
                    )

    def test_loop_over_a_conditional_body(self):
        """循环体本身可以是控制流：两个分支都输出同一个状态名。"""
        body = Conditional(
            _predicate_graph("body_pred"),
            _counter_body("body_then"),
            GraphRegion(_counter_body("body_else")),
        )
        loop = WhileLoop(
            _counter_condition(), body, carried={"x": "x_next"}, max_steps=8
        )
        self.assertEqual(loop.run({"x": _feed(0.0), "step": _feed(1.0)})["x"], np.float32(5.0))


class TestSpecRoundTrip(unittest.TestCase):
    def test_conditional_round_trip(self):
        cond = Conditional(_predicate_graph(), _scale_graph(2.0, "then"), _scale_graph(-3.0, "otherwise"))
        spec = json.loads(json.dumps(to_spec(cond), ensure_ascii=False))
        restored = from_spec(spec)
        for value in (1.5, -1.5):
            np.testing.assert_array_equal(
                restored.run({"x": _feed(value)})["y"], cond.run({"x": _feed(value)})["y"]
            )

    def test_nested_and_loop_round_trip(self):
        loop = WhileLoop(
            _counter_condition(),
            _counter_body(),
            carried={"x": "x_next"},
            max_steps=16,
        )
        spec = json.loads(json.dumps(to_spec(loop), ensure_ascii=False))
        restored = from_spec(spec)
        self.assertEqual(restored.max_steps, 16)
        self.assertEqual(restored.carried, {"x": "x_next"})
        np.testing.assert_array_equal(
            restored.run({"x": _feed(0.0), "step": _feed(1.0)})["x"],
            loop.run({"x": _feed(0.0), "step": _feed(1.0)})["x"],
        )

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            from_spec({"kind": "fork"})
        with self.assertRaises(TypeError):
            from_spec(["cond"])

    def test_matches_the_reference_backend(self):
        loop = WhileLoop(
            _counter_condition(),
            _counter_body(),
            carried={"x": "x_next"},
            max_steps=16,
        )
        got = loop.run({"x": _feed(0.0), "step": _feed(1.0)})
        # 退出条件是 x < 4.5，步长 1 → 停在 5；它必须等于参考执行器在 x=4 上
        # 再走一步的结果，而不是"多走一轮"。
        last = ReferenceBackend().execute(_counter_body(), {"x": _feed(4.0), "step": _feed(1.0)})
        self.assertEqual(got["x"], np.float32(5.0))
        self.assertEqual(got["x"], last["x_next"])


# ---- 非法构图的素材 -----------------------------------------------------


def _different_output_name():
    g = Graph("other_name")
    x = g.input("x", f32[()])
    g.output(g.mul(x, g.constant(1.0)), "z")
    return g


def _f64_graph(output_name="y"):
    g = Graph("f64")
    x = g.input("x", f64[()])
    g.output(g.mul(x, g.constant(1.0, dtype=f64)), output_name)
    return g


def _f64_input_graph():
    """输入是 f64，输出仍是 f32：用来单独检验"同名输入必须同类型"。"""
    g = Graph("f64_input")
    x = g.input("x", f64[()])
    g.output(g.cast(x, dtype="f32"), "y")
    return g


def _two_outputs():
    g = Graph("two_outputs")
    x = g.input("x", f32[()])
    g.output(g.step(x, threshold=0.0), "a")
    g.output(x, "b")
    return g


def _vector_predicate():
    g = Graph("vector_pred")
    x = g.input("x", f32[2])
    g.output(g.step(x, threshold=0.0), "flag")
    return g


if __name__ == "__main__":
    unittest.main()
