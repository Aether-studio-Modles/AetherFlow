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

    def test_fast_record_is_armed_during_planning(self):
        graph = _linear()
        executor = _Fast()
        plan = executor._plan(graph)
        self.assertTrue(plan)
        for group in plan:
            self.assertTrue(group.fast, "静态单节点组应该收成直连记录")
            self.assertEqual(group.fast[0], executor.invoke)

    def test_dispatch_selection_runs_once_per_plan(self):
        class Counted(_Fast):
            def __init__(self):
                self.selections = 0

            def _fast_step(self, group):
                self.selections += 1
                return super()._fast_step(group)

        graph = _linear()
        executor = Counted()
        plan = executor._plan(graph)
        self.assertEqual(executor.selections, len(plan))
        for _ in range(2):
            executor.run_device(graph, _feed(), release=True)
        self.assertEqual(executor.selections, len(plan))
        executor._clear_plan_values()
        executor.run_device(graph, _feed(), release=True)
        self.assertEqual(executor.selections, 2 * len(plan))

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


class ReleasePlanTest(unittest.TestCase):
    def _executor(self, base):
        class Traced(base):
            def __init__(self):
                self.produced = {}
                self.released = []

            def invoke(self, node, args, attrs):
                out = super().invoke(node, args, attrs)
                self.produced[id(out)] = node.output.id
                return out

            def dispose(self, value):
                self.released.append(self.produced[id(value)])

        return Traced()

    def test_duplicate_edges_and_branch_release_in_order(self):
        for base in (ReferenceExecutor, _Fast):
            with self.subTest(base=base):
                g = Graph()
                x = g.input("x", f32[4])
                a = g.relu(x)
                b = g.add(a, a)
                c = g.silu(a)
                g.output(g.add(b, c), "y")
                executor = self._executor(base)
                feed = {"x": np.arange(4, dtype=np.float32)}
                for _ in range(2):
                    executor.released.clear()
                    env = executor.run_device(g, feed, release=True)
                    self.assertEqual(set(env), {x.id, g.outputs[0].id})
                    self.assertEqual(executor.released, [a.id, b.id, c.id])
                executor.released.clear()
                env = executor.run_device(g, feed, release=False)
                self.assertEqual(len(env), len(g.nodes) + 1)
                self.assertEqual(executor.released, [])

    def test_group_internal_values_are_not_disposed(self):
        g = _linear()
        executor = self._executor(_Grouped)
        env = executor.run_device(g, _feed(), release=True)
        self.assertEqual(set(env), {v.id for v in (*g.inputs, *g.outputs)})
        self.assertEqual(executor.released, [])

    def test_output_registration_rebuilds_release_plan(self):
        g = Graph()
        x = g.input("x", f32[4])
        a = g.relu(x)
        g.output(g.silu(a), "y")
        executor = self._executor(_Fast)
        feed = {"x": np.arange(4, dtype=np.float32)}
        executor.run_device(g, feed, release=True)
        old = executor._plan(g)
        g.output(a, "saved")
        executor.released.clear()
        env = executor.run_device(g, feed, release=True)
        self.assertIsNot(executor._plan(g), old)
        self.assertIn(a.id, env)
        self.assertEqual(executor.released, [])

    def test_cached_constants_are_kept_after_preparation(self):
        class Cached(_Fast):
            def _prepare_plan(self, plan):
                super()._prepare_plan(plan)
                for group in plan:
                    node = group.nodes[0]
                    if node.op.name == "constant":
                        group.constants[node.output.id] = self.invoke(node, [], node.attrs)

        g = Graph()
        x = g.input("x", f32[4])
        c = g.constant(2.0, shape=(4,))
        g.output(g.add(x, c), "y")
        executor = self._executor(Cached)
        feed = {"x": np.arange(4, dtype=np.float32)}
        plan = executor._plan(g)
        cached = next(group for group in plan if group.constants)
        self.assertEqual(cached.cached[0], c.id)
        self.assertIs(cached.cached[1], cached.constants[c.id])
        self.assertEqual(cached.fast, ())
        first = executor.run_device(g, feed, release=True)
        second = executor.run_device(g, feed, release=True)
        self.assertIs(first[c.id], second[c.id])
        self.assertEqual(executor.released, [])
        np.testing.assert_array_equal(second[g.outputs[0].id], feed["x"] + 2)

    def test_symbolic_shapes_reuse_release_plan(self):
        g = Graph()
        x = g.input("x", f32["n", 4])
        a = g.relu(x)
        g.output(g.silu(a), "y")
        executor = self._executor(_Fast)
        for n in (2, 5, 2):
            executor.released.clear()
            env = executor.run_device(
                g, {"x": np.zeros((n, 4), np.float32)}, release=True
            )
            self.assertEqual(env[g.outputs[0].id].shape, (n, 4))
            self.assertEqual(executor.released, [a.id])


if __name__ == "__main__":
    unittest.main()
