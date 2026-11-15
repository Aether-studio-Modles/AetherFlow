"""静态图的执行快路径：计划期固化判断，运行期只做"取输入 → 调内核 → 写回"。

判据是"与通用路径逐位相同"：同一组参考内核，一条走 ``invoke_group``，
一条走 ``_fast_step`` 的直连记录，输出、dtype、形状报错都必须一致。
另有四条结构性判据：快路径确实启用、符号图不启用、融合组不启用、bf16 仍量化。
"""

import unittest

import numpy as np

from aetherflow import Graph, bf16, f32
from aetherflow.runtime.executor import ShapeMismatchError
from aetherflow.runtime.reference import ReferenceExecutor


class _Fast(ReferenceExecutor):
    """把每个节点的内核函数在计划期绑定好，让 ``_fast_step`` 有东西可用。

    参考执行器本身不预绑定（它没有"每节点一次的固定检查"可省），这里补上
    正是为了单独验证执行骨架的快路径。
    """

    def _prepare_plan(self, plan):
        for group in plan:
            group.kernels = tuple(self.invoke for _ in group.nodes)


class _WrongShape(_Fast):
    def invoke(self, node, args, attrs):
        return np.zeros((1, 1), np.float32)


class _WrongShapeNoPlan(_WrongShape):
    """同一组错误形状内核，但不预绑定——强制走通用路径做对照。"""

    def _prepare_plan(self, plan):
        return None


class _WrongDtype(_Fast):
    """数值正确、dtype 比声明更宽：快路径必须再 cast 回声明类型。"""

    def invoke(self, node, args, attrs):
        return np.asarray(node.op.reference(args, attrs), dtype=np.float64)


class _Grouped(_Fast):
    """把整张图当成一个融合组：组内有多节点，不该进快路径。"""

    def group_nodes(self, graph):
        return [tuple(graph.nodes)]


def _linear(batch=4, width=8):
    g = Graph("linear")
    x = g.input("x", f32[batch, width])
    w = g.input("w", f32[width, width])
    y = g.relu(g.matmul(x, w))
    g.output(g.reduce_mean(y), "loss")
    return g


def _feed(batch=4, width=8, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "x": rng.standard_normal((batch, width)).astype(np.float32),
        "w": rng.standard_normal((width, width)).astype(np.float32),
    }


class FastPathEquivalenceTest(unittest.TestCase):
    def test_outputs_match_generic_path(self):
        graph = _linear()
        feed = _feed()
        plain = ReferenceExecutor().run(graph, feed)
        fast = _Fast().run(graph, feed)
        np.testing.assert_array_equal(fast["loss"], plain["loss"])

    def test_environment_matches_generic_path(self):
        graph = _linear()
        feed = _feed()
        plain = ReferenceExecutor().run_device(graph, feed)
        fast = _Fast().run_device(graph, feed)
        self.assertEqual(set(plain), set(fast))
        for key in plain:
            np.testing.assert_array_equal(fast[key], plain[key])

    def test_fast_record_is_armed_after_first_run(self):
        graph = _linear()
        executor = _Fast()
        executor.run(graph, _feed())
        plan = executor._plan(graph)
        self.assertTrue(plan)
        for group in plan:
            self.assertTrue(group.fast, "静态单节点组应该收成直连记录")
            self.assertEqual(group.fast[0], executor.invoke)

    def test_casts_back_to_declared_dtype(self):
        graph = _linear()
        feed = _feed()
        plain = ReferenceExecutor().run(graph, feed)
        fast = _WrongDtype().run(graph, feed)
        self.assertEqual(fast["loss"].dtype, plain["loss"].dtype)
        np.testing.assert_array_equal(fast["loss"], plain["loss"])

    def test_shape_mismatch_message_matches_generic_path(self):
        graph = _linear()
        feed = _feed()
        with self.assertRaises(ShapeMismatchError) as generic:
            _WrongShapeNoPlan().run(graph, feed)
        with self.assertRaises(ShapeMismatchError) as direct:
            _WrongShape().run(graph, feed)
        self.assertEqual(str(direct.exception), str(generic.exception))


class FastPathBoundaryTest(unittest.TestCase):
    def test_symbolic_graph_stays_on_generic_path(self):
        g = Graph("dynamic")
        x = g.input("x", f32["n", 8])
        g.output(g.relu(x), "y")
        executor = _Fast()
        executor.run(g, {"x": np.zeros((4, 8), np.float32)})
        for group in executor._plan(g):
            self.assertEqual(group.fast, (), "含符号维的组不能直连")

    def test_multi_node_group_stays_on_generic_path(self):
        graph = _linear()
        executor = _Grouped()
        got = executor.run(graph, _feed())
        want = ReferenceExecutor().run(graph, _feed())
        np.testing.assert_array_equal(got["loss"], want["loss"])
        for group in executor._plan(graph):
            self.assertEqual(group.fast, (), "融合组不能走单节点直连")

    def test_bf16_output_is_still_quantized(self):
        g = Graph("bf16")
        x = g.input("x", bf16[4, 5])
        w = g.input("w", bf16[5, 2])
        g.output(g.matmul(x, w), "y")
        rng = np.random.default_rng(3)
        feed = {
            "x": rng.standard_normal((4, 5)).astype(np.float32),
            "w": rng.standard_normal((5, 2)).astype(np.float32),
        }
        plain = ReferenceExecutor().run(g, feed)
        executor = _Fast()
        fast = executor.run(g, feed)
        np.testing.assert_array_equal(fast["y"], plain["y"])
        group = executor._plan(g)[-1]
        self.assertTrue(group.fast[7], "bf16 输出必须走量化，直连也要保留这一步")


if __name__ == "__main__":
    unittest.main()
