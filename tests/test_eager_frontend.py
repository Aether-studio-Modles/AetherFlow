"""eager 前端测试：语法糖不改变语义。

这里只验证两件事：

1. 用运算符写出来的图与手写 ``add_node`` 的图**完全相同**，所以 IR、
   verifier、序列化、自动微分都不需要为它改一行；
2. 依赖张量值的写法（``if`` / 下标 / 跨图运算）必须当场报错，而不是猜一个
   编译期真值——这类错误在数值对拍里看不出来，静默通过最危险。
"""

import unittest

import numpy as np

from aetherflow import Graph, ReferenceBackend, backward, dumps, loads, verify
from aetherflow.frontend import EagerGraph, Tensor
from aetherflow.ir.types import f32


class TestOperatorSugar(unittest.TestCase):
    def setUp(self):
        self.ref = ReferenceBackend()

    def _out(self, g, feed):
        return list(self.ref.execute(g, feed).values())[0]

    def test_arithmetic_matches_numpy(self):
        g = EagerGraph()
        x = g.input("x", f32[2, 3])
        g.output(x * 0.5 + 1.0, "y")
        data = np.arange(6, dtype=np.float32).reshape(2, 3)
        np.testing.assert_allclose(self._out(g, {"x": data}), data * 0.5 + 1.0, rtol=1e-6)

    def test_reflected_operators(self):
        g = EagerGraph()
        x = g.input("x", f32[2, 3])
        g.output(2.0 - x, "sub")
        g.output(12.0 / x, "div")
        data = np.array([[1.0, 2.0, 4.0], [3.0, 6.0, 12.0]], dtype=np.float32)
        got = self.ref.execute(g, {"x": data})
        np.testing.assert_allclose(got["sub"], 2.0 - data, rtol=1e-6)
        np.testing.assert_allclose(got["div"], 12.0 / data, rtol=1e-6)

    def test_neg_and_matmul(self):
        g = EagerGraph()
        x = g.input("x", f32[2, 3])
        w = g.input("w", f32[3, 4])
        g.output(-x, "n")
        g.output(x @ w, "y")
        xd = np.arange(6, dtype=np.float32).reshape(2, 3) * 0.1
        wd = np.arange(12, dtype=np.float32).reshape(3, 4) * 0.2
        got = self.ref.execute(g, {"x": xd, "w": wd})
        np.testing.assert_allclose(got["n"], -xd, rtol=1e-6)
        np.testing.assert_allclose(got["y"], xd @ wd, rtol=1e-5)

    def test_graph_is_identical_to_handwritten(self):
        """同一个算式的两种写法产出同一份 IR：这是"零改动"的直接证据。"""
        g = EagerGraph()
        x = g.input("x", f32[2, 3])
        g.output(x * 0.5 + 1.0, "y")

        h = Graph()
        hx = h.input("x", f32[2, 3])
        h.output(h.add(h.mul(hx, h.constant(0.5)), h.constant(1.0)), "y")

        self.assertEqual(g.to_ir(), h.to_ir())
        self.assertEqual([n.op.name for n in g.nodes], [n.op.name for n in h.nodes])

    def test_dynamic_op_method_returns_tensor(self):
        """``g.relu(x)`` 这类算子名方法也要走 Tensor，否则混用时会漏包装。"""
        g = EagerGraph()
        x = g.input("x", f32[2, 3])
        y = g.relu(x)
        self.assertIsInstance(y, Tensor)
        g.output(y, "y")
        data = np.array([[-1.0, 0.0, 2.0]] * 2, dtype=np.float32)
        np.testing.assert_allclose(self._out(g, {"x": data}), np.maximum(data, 0.0))

    def test_zero_dim_constant_broadcasts(self):
        """标量常量是 0 维的：最容易被后端当成"形状不匹配"而拒绝。"""
        g = EagerGraph()
        x = g.input("x", f32[3, 4])
        g.output(x / 2.0, "y")
        const = [n for n in g.nodes if n.op.name == "constant"][0]
        self.assertEqual(const.output.type.shape, ())
        data = np.ones((3, 4), dtype=np.float32)
        np.testing.assert_allclose(self._out(g, {"x": data}), data / 2.0, rtol=1e-6)

    def test_tensor_op_methods_match_graph_dsl(self):
        """``x.relu()`` 与 ``g.relu(x)`` 必须产出同一份 IR。"""
        g = EagerGraph()
        x = g.input("x", f32[2, 3])
        g.output(x.relu().reshape(shape=(3, 2)).transpose(perm=(1, 0)), "y")

        h = Graph()
        hx = h.input("x", f32[2, 3])
        h.output(
            h.transpose(h.reshape(h.relu(hx), shape=(3, 2)), perm=(1, 0)),
            "y",
        )
        self.assertEqual(g.to_ir(), h.to_ir())

    def test_tensor_op_method_values(self):
        g = EagerGraph()
        x = g.input("x", f32[2, 3])
        g.output(x.softmax(axis=-1), "p")
        g.output(x.gelu(), "a")
        data = np.array([[-1.0, 0.0, 2.0], [0.5, 0.25, -0.5]], dtype=np.float32)
        shifted = data - data.max(axis=-1, keepdims=True)
        expect = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
        got = self.ref.execute(g, {"x": data})
        np.testing.assert_allclose(got["p"], expect, rtol=1e-6)
        c = np.float32(0.7978845608028654)
        expect_gelu = 0.5 * data * (
            1.0 + np.tanh(c * (data + np.float32(0.044715) * data**3))
        )
        np.testing.assert_allclose(got["a"], expect_gelu, rtol=1e-5, atol=1e-6)

    def test_tensor_op_method_rejects_unknown_names(self):
        g = EagerGraph()
        x = g.input("x", f32[2, 3])
        with self.assertRaises(AttributeError):
            x.not_an_op
        with self.assertRaises(AttributeError):
            x._private


class TestExplicitRejections(unittest.TestCase):
    """构图期的错误必须当场抛出，不能靠数值对拍兜底。"""

    def setUp(self):
        self.g = EagerGraph()
        self.x = self.g.input("x", f32[2, 3])

    def test_bool_raises(self):
        with self.assertRaises(TypeError):
            bool(self.x)

    def test_control_flow_raises(self):
        with self.assertRaises(TypeError):
            if self.x:  # noqa: SIM223 - 这里就是要验证它会报错
                pass

    def test_getitem_raises(self):
        with self.assertRaises(TypeError):
            self.x[0]

    def test_cross_graph_arithmetic_raises(self):
        other = EagerGraph()
        y = other.input("y", f32[2, 3])
        with self.assertRaises(ValueError):
            self.x + y

    def test_cross_graph_output_raises(self):
        other = EagerGraph()
        y = other.input("y", f32[2, 3])
        with self.assertRaises(ValueError):
            self.g.output(y, "y")

    def test_foreign_array_operand_raises(self):
        """numpy 数组的形状不会进 IR，等于绕过类型推导与后端能力预检。"""
        with self.assertRaises(TypeError):
            self.x + np.ones(3, dtype=np.float32)


class TestEagerAutodiff(unittest.TestCase):
    def test_backward_matches_analytic(self):
        g = EagerGraph()
        x = g.input("x", f32[2, 3])
        w = g.input("w", f32[3, 4])
        b = g.input("b", f32[4])
        g.output(g.relu(x @ w + b), "y")

        rg = backward(g, wrt=[w.value, b.value])
        self.assertEqual(rg.output_names, ["dw", "db"])

        xd = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], dtype=np.float32)
        wd = np.arange(12, dtype=np.float32).reshape(3, 4) * 0.2 - 1.0
        bd = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
        dy = np.ones((2, 4), dtype=np.float32)

        got = ReferenceBackend().execute(rg, {"x": xd, "w": wd, "b": bd, "dy": dy})
        mask = (xd @ wd + bd) > 0
        np.testing.assert_allclose(got["dw"], xd.T @ mask.astype(np.float32), rtol=1e-5)
        np.testing.assert_allclose(got["db"], mask.sum(axis=0).astype(np.float32), rtol=1e-6)


class TestEagerIsPlainGraph(unittest.TestCase):
    """eager 图的产物必须能被其余所有环节直接消费。"""

    def _build(self):
        g = EagerGraph()
        x = g.input("x", f32[2, 3])
        g.output(x * 0.5 + 1.0, "y")
        return g

    def test_verify_passes(self):
        self.assertEqual(verify(self._build()), [])

    def test_json_roundtrip(self):
        g = self._build()
        restored = loads(dumps(g))
        data = np.arange(6, dtype=np.float32).reshape(2, 3)
        ref = ReferenceBackend()
        np.testing.assert_allclose(
            list(ref.execute(restored, {"x": data}).values())[0],
            data * 0.5 + 1.0,
            rtol=1e-6,
        )

    def test_no_tensor_objects_leak_into_ir(self):
        """IR 里只能有 Value：Tensor 是前端对象，后端不得看见它。"""
        g = self._build()
        for v in g.values:
            self.assertNotIsInstance(v, Tensor)


class TestImmediateBackward(unittest.TestCase):
    """写一行、立刻拿梯度：反向图建一次、执行一次，梯度落回 Tensor。"""

    def _build(self):
        g = EagerGraph("eager_bwd")
        x = g.input("x", f32[2, 3])
        w = g.input("w", f32[3, 1])
        w.requires_grad = True
        y = x @ w
        rows = g.add_node("reduce_mean", [y], {"axis": -1, "keepdims": False})
        loss = g.add_node("reduce_mean", [rows], {"axis": 0, "keepdims": False})
        g.output(loss, "loss")
        return g, x, w, loss

    def test_grad_matches_analytic(self):
        g, x, w, loss = self._build()
        xv = np.arange(6, dtype=np.float32).reshape(2, 3)
        wv = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)

        grads = loss.backward(feed={"x": xv, "w": wv})

        self.assertEqual(list(grads), ["w"])
        # mean 过 2x1 张量，每个元素的余切是 0.5；dW = X^T @ dY
        np.testing.assert_allclose(w.grad, xv.T @ np.full((2, 1), 0.5), rtol=1e-6)
        np.testing.assert_array_equal(grads["w"], w.grad)
        # 没标 requires_grad 的输入不算梯度，也不会被写回
        self.assertIsNone(x.grad)

    def test_repeated_backward_reuses_the_same_graph(self):
        g, x, w, loss = self._build()
        feed = {
            "x": np.ones((2, 3), dtype=np.float32),
            "w": np.ones((3, 1), dtype=np.float32),
        }
        first = loss.backward(feed=feed)
        cached = g._bwd_cache[1]
        second = loss.backward(feed=feed)
        self.assertIs(g._bwd_cache[1], cached)
        np.testing.assert_array_equal(first["w"], second["w"])

    def test_growing_the_graph_rebuilds_the_backward_graph(self):
        g, x, w, loss = self._build()
        feed = {
            "x": np.ones((2, 3), dtype=np.float32),
            "w": np.ones((3, 1), dtype=np.float32),
        }
        loss.backward(feed=feed)
        cached = g._bwd_cache[1]

        scale = g.input("scale", f32[2, 3])
        rows = g.add_node("reduce_mean", [scale], {"axis": -1, "keepdims": False})
        total = g.add_node("reduce_mean", [rows], {"axis": 0, "keepdims": False})
        scaled = loss * total
        g.output(scaled, "scaled")
        scaled.backward(feed={**feed, "scale": np.ones((2, 3), dtype=np.float32)})
        self.assertIsNot(g._bwd_cache[1], cached)

    def test_loss_must_be_a_graph_output(self):
        g = EagerGraph()
        x = g.input("x", f32[4])
        hidden = g.silu(x)
        g.output(hidden * hidden, "y")
        with self.assertRaises(ValueError) as ctx:
            hidden.backward(feed={"x": np.zeros(4, dtype=np.float32)})
        self.assertIn("不是图输出", str(ctx.exception))

    def test_non_scalar_loss_rejected(self):
        g = EagerGraph()
        x = g.input("x", f32[2, 4])
        rows = g.add_node("reduce_mean", [x], {"axis": 0, "keepdims": False})
        g.output(rows, "rows")
        with self.assertRaises(ValueError) as ctx:
            rows.backward(feed={"x": np.zeros((2, 4), dtype=np.float32)})
        self.assertIn("必须是标量", str(ctx.exception))

    def test_no_targets_raises(self):
        g, x, w, loss = self._build()
        w.requires_grad = False
        feed = {
            "x": np.ones((2, 3), dtype=np.float32),
            "w": np.ones((3, 1), dtype=np.float32),
        }
        with self.assertRaises(ValueError) as ctx:
            loss.backward(feed=feed)
        self.assertIn("requires_grad", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
