"""动态执行层的测试：图按形状构建、形状复用、trace-once 重放，以及
"两次 run 之间读值分支"。"""

import unittest

import numpy as np

from aetherflow import DynamicRunner, Graph, ReferenceBackend, f32


def _relu(shapes):
    g = Graph("relu")
    x = g.input("x", f32[shapes["x"]])
    g.output(g.relu(x), "y")
    return g


class TestDynamicRunner(unittest.TestCase):
    def test_same_shape_reuses_the_graph(self):
        runner = DynamicRunner(_relu)
        a = np.array([[-1.0, 2.0]], dtype=np.float32)
        first = runner.run({"x": a})["y"]
        second = runner.run({"x": -a})["y"]
        self.assertEqual(runner.builds, 1)
        np.testing.assert_array_equal(first, [[0.0, 2.0]])
        np.testing.assert_array_equal(second, [[1.0, 0.0]])

    def test_new_shape_builds_again(self):
        runner = DynamicRunner(_relu)
        small = runner.run({"x": np.ones((2, 3), np.float32)})["y"]
        large = runner.run({"x": np.ones((7, 3), np.float32)})["y"]
        self.assertEqual(small.shape, (2, 3))
        self.assertEqual(large.shape, (7, 3))
        self.assertEqual(runner.builds, 2)

    def test_shape_key_ignores_input_order(self):
        """两个输入的声明顺序不该产生两份缓存。"""
        def add(shapes):
            g = Graph("add")
            x = g.input("x", f32[shapes["x"]])
            y = g.input("y", f32[shapes["y"]])
            g.output(g.add(x, y), "z")
            return g

        runner = DynamicRunner(add)
        x = np.ones((2, 2), np.float32)
        y = np.ones((2, 2), np.float32)
        runner.run({"x": x, "y": y})
        runner.run({"y": y, "x": x})
        self.assertEqual(runner.builds, 1)

    def test_capacity_evicts_the_least_recently_used(self):
        runner = DynamicRunner(_relu, capacity=1)
        for shape in ((2, 3), (4, 3), (2, 3)):
            runner.run({"x": np.ones(shape, np.float32)})
        self.assertEqual(runner.builds, 3)

    def test_value_can_be_read_between_runs(self):
        """数据依赖的控制流发生在两次 run 之间：读回真实值，再决定下一步。"""
        runner = DynamicRunner(_relu)
        probe = runner.run({"x": np.array([[-2.0, -1.0]], np.float32)})["y"]
        # 全被 relu 掐掉时走另一支——这是静态图内部做不到的
        feed = np.array([[5.0]], np.float32) if float(probe.sum()) == 0.0 else np.zeros((1, 1), np.float32)
        result = runner.run({"x": feed})["y"]
        np.testing.assert_array_equal(result, [[5.0]])

    def test_matches_the_reference_backend(self):
        feed = {"x": np.array([[-3.0, 0.5, 4.0]], np.float32)}
        expected = ReferenceBackend().execute(_relu({"x": (1, 3)}), feed)["y"]
        got = DynamicRunner(_relu).run(feed)["y"]
        np.testing.assert_array_equal(got, expected)

    def test_build_must_return_a_graph(self):
        runner = DynamicRunner(lambda shapes: None)
        with self.assertRaises(TypeError):
            runner.run({"x": np.ones((1, 2), np.float32)})

    def test_build_must_be_callable(self):
        with self.assertRaises(TypeError):
            DynamicRunner(object())
        for bad in (0, -1, True, 1.5):
            with self.subTest(capacity=bad):
                with self.assertRaises(ValueError):
                    DynamicRunner(_relu, capacity=bad)

    def test_inputs_must_be_arrays(self):
        runner = DynamicRunner(_relu)
        for bad in ({}, {"x": 3}, None):
            with self.subTest(inputs=bad):
                with self.assertRaises((ValueError, TypeError)):
                    runner.run(bad)

    def test_shapes_property_is_ordered_by_recent_use(self):
        runner = DynamicRunner(_relu)
        runner.run({"x": np.ones((1, 2), np.float32)})
        runner.run({"x": np.ones((3, 2), np.float32)})
        self.assertEqual([key[0][1] for key in runner.shapes], [(1, 2), (3, 2)])


class TestTraceOnce(unittest.TestCase):
    """声明动态轴后，一次构图覆盖这一维上的全部取值。"""

    def test_declared_axis_reuses_one_graph(self):
        runner = DynamicRunner(_relu, dynamic_axes={"x": {0: "B"}})
        for batch in (1, 3, 7):
            feed = {"x": np.linspace(-1.0, 1.0, batch * 4, dtype=np.float32).reshape(batch, 4)}
            got = runner.run(feed)["y"]
            np.testing.assert_array_equal(got, np.maximum(feed["x"], 0.0))
        self.assertEqual(runner.builds, 1)
        (key,) = runner.shapes
        self.assertEqual(key[0][1], ("B", 4))

    def test_undeclared_axis_still_rebuilds(self):
        """只声明了批维；特征维变化仍是另一张图。"""
        runner = DynamicRunner(_relu, dynamic_axes={"x": {0: "B"}})
        runner.run({"x": np.ones((2, 3), np.float32)})
        runner.run({"x": np.ones((2, 4), np.float32)})
        self.assertEqual(runner.builds, 2)

    def test_negative_axis_is_normalized(self):
        runner = DynamicRunner(_relu, dynamic_axes={"x": {-2: "B"}})
        runner.run({"x": np.ones((2, 3), np.float32)})
        runner.run({"x": np.ones((5, 3), np.float32)})
        self.assertEqual(runner.builds, 1)
        self.assertEqual(runner.shapes[0][0][1], ("B", 3))

    def test_shared_symbol_means_equal_dims(self):
        def add(shapes):
            g = Graph("add")
            x = g.input("x", f32[shapes["x"]])
            y = g.input("y", f32[shapes["y"]])
            g.output(g.add(x, y), "z")
            return g

        runner = DynamicRunner(add, dynamic_axes={"x": {0: "B"}, "y": {0: "B"}})
        out = runner.run(
            {
                "x": np.ones((3, 2), np.float32),
                "y": np.full((3, 2), 2.0, np.float32),
            }
        )["z"]
        np.testing.assert_array_equal(out, np.full((3, 2), 3.0))
        self.assertEqual(runner.builds, 1)

        with self.assertRaises(ValueError) as ctx:
            runner.run(
                {
                    "x": np.ones((3, 2), np.float32),
                    "y": np.ones((4, 2), np.float32),
                }
            )
        self.assertIn("符号维 B 绑定冲突", str(ctx.exception))

    def test_result_matches_the_reference_backend(self):
        for batch in (1, 4):
            feed = {"x": np.array([[-3.0, 0.5, 4.0]] * batch, np.float32)}
            expected = ReferenceBackend().execute(_relu({"x": (batch, 3)}), feed)["y"]
            runner = DynamicRunner(_relu, dynamic_axes={"x": {0: "B"}})
            np.testing.assert_array_equal(runner.run(feed)["y"], expected)

    def test_trace_replays_a_linear_layer(self):
        """一点不像玩具的图：矩阵乘的参数形状与批维无关，批维可绑定。"""
        def linear(shapes):
            g = Graph("linear")
            x = g.input("x", f32[shapes["x"]])
            w = g.input("w", f32[shapes["w"]])
            g.output(g.matmul(x, w), "y")
            return g

        runner = DynamicRunner(linear, dynamic_axes={"x": {0: "B"}})
        w = np.arange(8, dtype=np.float32).reshape(4, 2)
        for batch in (1, 3, 7):
            x = np.arange(batch * 4, dtype=np.float32).reshape(batch, 4)
            np.testing.assert_allclose(runner.run({"x": x, "w": w})["y"], x @ w)
        self.assertEqual(runner.builds, 1)

    def test_build_must_use_the_template(self):
        """把模板上的符号写成整数，第一次能过、换形状才炸——必须当场拒绝。"""
        def baked(shapes):
            g = Graph("baked")
            x = g.input("x", f32[2, 3])
            g.output(g.relu(x), "y")
            return g

        runner = DynamicRunner(baked, dynamic_axes={"x": {0: "B"}})
        with self.assertRaises(ValueError) as ctx:
            runner.run({"x": np.ones((2, 3), np.float32)})
        message = str(ctx.exception)
        self.assertIn("必须使用传入的形状模板", message)
        self.assertIn("f32[2, 3]", message)

    def test_declared_input_must_be_fed(self):
        runner = DynamicRunner(_relu, dynamic_axes={"z": {0: "B"}})
        with self.assertRaises(ValueError) as ctx:
            runner.run({"x": np.ones((2, 3), np.float32)})
        self.assertIn("但本次 run 没有提供", str(ctx.exception))

    def test_build_must_keep_the_declared_input(self):
        def only_x(shapes):
            g = Graph("only_x")
            x = g.input("x", f32[shapes["x"]])
            g.output(x, "y")
            return g

        runner = DynamicRunner(
            only_x, dynamic_axes={"x": {0: "B"}, "z": {0: "B"}}
        )
        with self.assertRaises(ValueError) as ctx:
            runner.run(
                {
                    "x": np.ones((2, 3), np.float32),
                    "z": np.ones((2, 3), np.float32),
                }
            )
        self.assertIn("图里没有这个输入", str(ctx.exception))

    def test_axis_must_exist_in_the_actual_shape(self):
        runner = DynamicRunner(_relu, dynamic_axes={"x": {2: "B"}})
        with self.assertRaises(ValueError) as ctx:
            runner.run({"x": np.ones((2, 3), np.float32)})
        self.assertIn("的秩是 2", str(ctx.exception))

    def test_axes_cannot_collide_after_normalization(self):
        runner = DynamicRunner(_relu, dynamic_axes={"x": {0: "B", -2: "C"}})
        with self.assertRaises(ValueError) as ctx:
            runner.run({"x": np.ones((2, 3), np.float32)})
        self.assertIn("被重复声明", str(ctx.exception))


class TestDynamicAxesDeclaration(unittest.TestCase):
    """声明本身的校验：错在构造期，不要等到构图或运行。"""

    def test_none_keeps_the_static_behaviour(self):
        runner = DynamicRunner(_relu, dynamic_axes=None)
        runner.run({"x": np.ones((2, 3), np.float32)})
        runner.run({"x": np.ones((4, 3), np.float32)})
        self.assertEqual(runner.builds, 2)

    def test_bad_shapes_of_the_declaration(self):
        cases = [
            ([("x", 0)], TypeError),
            ({"x": 0}, ValueError),
            ({"x": {}}, ValueError),
            ({"x": {True: "B"}}, ValueError),
            ({"x": {0: 0}}, ValueError),
            ({"x": {0: "f32"}}, ValueError),
            ({"x": {0: "not an identifier"}}, ValueError),
        ]
        for dynamic_axes, error in cases:
            with self.subTest(dynamic_axes=dynamic_axes):
                with self.assertRaises(error):
                    DynamicRunner(_relu, dynamic_axes=dynamic_axes)


if __name__ == "__main__":
    unittest.main()
