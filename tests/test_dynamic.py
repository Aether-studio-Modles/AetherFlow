"""动态执行层的测试：图按形状构建、形状复用、以及"两次 run 之间读值分支"。"""

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


if __name__ == "__main__":
    unittest.main()
