"""Module / Parameter：参数声明、一次构图、重放与 state_dict。

这里不测数值算子（那属于 IR 与后端测试），只锁定 ``Module`` 的契约：

1. 参数是图输入，``data`` 是执行期数组，两者靠名字绑定；
2. 构图只发生一次，第二次调用必须报错而不是静默复用；
3. ``arrays()`` 给的是引用（优化器原地更新才生效），``state_dict()`` 给的是
   副本（保存与对拍不该被后续训练改写）。
"""

import unittest

import numpy as np

from aetherflow.frontend import Module
from aetherflow.ir.types import f32


class Tiny(Module):
    def __init__(self, name: str = ""):
        super().__init__(name)
        self.w = self.param("w", (3, 2), init="normal", value=0.5)
        self.b = self.param("b", (2,), init="zeros")

    def forward(self, x):
        return x @ self.w + self.b


class TestModuleBuild(unittest.TestCase):
    def setUp(self):
        self.model = Tiny()
        self.x = self.model.graph.input("x", f32[2, 3])

    def test_build_records_output_and_replays(self):
        self.model(self.x)
        self.assertTrue(self.model.built)
        self.assertEqual(len(self.model.graph.output_names), 1)
        self.assertEqual(len(self.model.graph.outputs), 1)

        weight = np.arange(6, dtype=np.float32).reshape(3, 2)
        self.model.load_state_dict({"w": weight, "b": np.zeros(2, dtype=np.float32)})
        data = np.arange(6, dtype=np.float32).reshape(2, 3)
        got = self.model.run({"x": data})
        np.testing.assert_allclose(got, data @ weight, rtol=1e-6)

    def test_second_call_is_rejected(self):
        self.model(self.x)
        with self.assertRaises(RuntimeError):
            self.model(self.x)

    def test_run_before_build_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.model.run({"x": np.zeros((2, 3), dtype=np.float32)})

    def test_prefixed_names_are_used(self):
        model = Tiny("encoder")
        self.assertEqual(sorted(model.arrays()), ["encoder.b", "encoder.w"])
        node_name = model.param("extra", (1,)).name
        self.assertEqual(node_name, "encoder.extra")


class TestModuleParameters(unittest.TestCase):
    def test_arrays_are_references(self):
        """优化器原地改写 arrays 之后，重放要用到新值。"""
        model = Tiny()
        x = model.graph.input("x", f32[2, 3])
        model(x)
        arrays = model.arrays()
        arrays["w"][...] = 1.0
        data = np.ones((2, 3), dtype=np.float32)
        np.testing.assert_allclose(model.run({"x": data}), np.full((2, 2), 3.0), rtol=1e-6)

    def test_state_dict_is_a_copy(self):
        model = Tiny()
        state = model.state_dict()
        state["w"][...] = 7.0
        self.assertFalse(np.any(model.arrays()["w"] == 7.0))

    def test_state_dict_roundtrip_checks_keys_and_shapes(self):
        model = Tiny()
        state = model.state_dict()
        state["w"][...] = 2.0
        model.load_state_dict(state)
        np.testing.assert_allclose(model.arrays()["w"], 2.0)

        with self.assertRaises(KeyError):
            model.load_state_dict({"w": state["w"]})
        with self.assertRaises(ValueError):
            model.load_state_dict(
                {"w": np.zeros((2, 3), dtype=np.float32), "b": state["b"]}
            )

    def test_feed_rejects_input_parameter_clash(self):
        model = Tiny()
        with self.assertRaises(ValueError):
            model.feed({"w": np.zeros((3, 2), dtype=np.float32)})

    def test_symbolic_parameter_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            Tiny().param("bad", (3, "B"))


class TestMigrationPath(unittest.TestCase):
    """已有构图函数自己声明输入时，用 adopt 接管参数。"""

    def test_adopt_registers_existing_input(self):
        model = Module()
        w = model.graph.input("w", f32[3, 2])
        adopted = model.adopt("w", w.value, np.ones((3, 2), dtype=np.float32))
        self.assertIs(adopted.value, w.value)
        np.testing.assert_allclose(model.arrays()["w"], 1.0)

    def test_adopt_joins_backward_targets(self):
        """接管后要进 _input_tensors，否则 backward 的缺省目标会漏掉它。"""
        model = Module()
        w = model.graph.input("w", f32[3, 2])
        x = model.graph.input("x", f32[2, 3])
        model.adopt("w", w, np.ones((3, 2), dtype=np.float32))
        loss = model.op("reduce_sum", x @ w, axis=(0, 1), keepdims=False)
        model.graph.output(loss)
        grads = model.graph.backward(
            loss,
            feed={
                "w": np.ones((3, 2), dtype=np.float32),
                "x": np.ones((2, 3), dtype=np.float32),
            },
        )
        self.assertEqual(sorted(grads), ["w"])
        np.testing.assert_allclose(grads["w"], np.full((3, 2), 2.0), rtol=1e-6)

    def test_adopt_rejects_unknown_or_mismatched(self):
        model = Module()
        w = model.graph.input("w", f32[3, 2])
        other = model.graph.input("other", f32[3, 2])
        with self.assertRaises(ValueError):
            model.adopt("missing", w.value, np.ones((3, 2), dtype=np.float32))
        with self.assertRaises(ValueError):
            model.adopt("w", other.value, np.ones((3, 2), dtype=np.float32))
        with self.assertRaises(ValueError):
            model.adopt("w", w.value, np.ones((2, 3), dtype=np.float32))

    def test_adopt_rejects_foreign_tensor(self):
        left = Module()
        left.graph.input("w", f32[3, 2])
        right = Module()
        foreign = right.graph.input("w", f32[3, 2])
        with self.assertRaises(ValueError):
            left.adopt("w", foreign, np.ones((3, 2), dtype=np.float32))

    def test_adopt_copies_data(self):
        """接管外部数组时必须拷贝：torch 的 .numpy() 是共享内存的视图。"""
        model = Module()
        w = model.graph.input("w", f32[2, 2])
        source = np.ones((2, 2), dtype=np.float32)
        model.adopt("w", w, source)
        source[...] = 7.0
        np.testing.assert_allclose(model.arrays()["w"], 1.0)
        model.arrays()["w"][...] = 3.0
        np.testing.assert_allclose(source, 7.0)

    def test_p_returns_graph_value(self):
        model = Tiny()
        self.assertIs(model.p("w"), model.graph.find("w"))
        with self.assertRaises(KeyError):
            model.p("nope")


class TestInitialization(unittest.TestCase):
    def test_same_name_gives_same_values(self):
        left = Tiny().state_dict()
        right = Tiny().state_dict()
        for name in left:
            np.testing.assert_array_equal(left[name], right[name])

    def test_prefix_changes_initialization_stream(self):
        """初始化种子混入完整参数名：改前缀不该复用同一串随机数。"""
        base = Tiny().state_dict()["w"]
        prefixed = Tiny("encoder").state_dict()["encoder.w"]
        self.assertFalse(np.array_equal(base, prefixed))


if __name__ == "__main__":
    unittest.main()
