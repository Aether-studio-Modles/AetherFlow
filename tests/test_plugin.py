"""插件算子扩展面。

三条护栏各对应一种真实的失控方式：

- 没有命名空间 → 两个插件撞名，谁先被 import 谁生效，且极难复现；
- 允许覆盖内置 → 参考语义被悄悄换掉，conformance 依然 PASS；
- 不能卸载 → 测试之间互相污染，后跑的用例看见前面遗留的算子。

还要固定一条收益：注册之后**不需要**额外适配。参考后端按算子表动态取语义，
所以插件算子立刻可执行、可被 conformance 对拍——这正是"先进 IR 再谈后端"
这条约定的价值所在。
"""

import unittest

import numpy as np

from aetherflow import (
    BackendCapabilityError,
    Case,
    Graph,
    Op,
    Portability,
    ReferenceBackend,
    f32,
    plugin_ops,
    register_operator,
    run_conformance,
    unregister_operator,
)
from aetherflow.ir.ops import OPS

NAME = "demo.double"


def _double_op(name=NAME, portability=Portability.PORTABLE):
    """测试用算子：逐元素乘 2。语义简单，便于手工核对。"""
    return Op(
        name=name,
        infer=lambda inputs, attrs: inputs[0],
        reference=lambda args, attrs: args[0] * 2.0,
        portability=portability,
        doc="测试用插件算子：逐元素乘 2",
    )


def _graph(name=NAME, shape=(3, 4)):
    g = Graph(name)
    x = g.input("x", f32[shape])
    g.output(g.add_node(name, [x]), "y")
    return g


class TestRegistrationBoundaries(unittest.TestCase):
    def setUp(self):
        self.added = []
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for name in reversed(self.added):
            if name in OPS:
                unregister_operator(name)

    def _register(self, op):
        register_operator(op, namespace="demo")
        self.added.append(op.name)
        return op

    def test_missing_namespace_prefix_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            register_operator(_double_op(name="double"), namespace="demo")
        self.assertIn("demo.", str(ctx.exception))
        self.assertNotIn("double", OPS)

    def test_empty_namespace_rejected(self):
        with self.assertRaises(ValueError):
            register_operator(_double_op(), namespace="")

    def test_non_op_rejected(self):
        with self.assertRaises(TypeError):
            register_operator("demo.double", namespace="demo")

    def test_duplicate_rejected(self):
        self._register(_double_op())
        with self.assertRaises(ValueError):
            register_operator(_double_op(), namespace="demo")

    def test_builtin_cannot_be_overwritten_or_unloaded(self):
        """内置算子既不能被同名注册覆盖，也不能被卸载。

        前缀规则已经让"带命名空间的插件名"不可能等于内置名，这里的显式检查是
        第二道锁：规则将来若放宽（比如允许无点号的插件名），这一条仍然成立。
        """
        with self.assertRaises(ValueError):
            register_operator(
                Op(name="add", infer=lambda i, a: i[0], reference=lambda a, t: a[0]),
                namespace="add",
            )
        with self.assertRaises(ValueError):
            unregister_operator("add")
        self.assertIn("add", OPS)

    def test_unregister_restores_table(self):
        self._register(_double_op())
        before = len(OPS)
        unregister_operator(NAME)
        self.assertNotIn(NAME, OPS)
        self.assertEqual(len(OPS), before - 1)
        self.assertNotIn(NAME, plugin_ops())

    def test_unregister_unknown(self):
        with self.assertRaises(KeyError):
            unregister_operator("demo.nothing")

    def test_plugin_ops_only_lists_plugins(self):
        self._register(_double_op())
        self.assertEqual(plugin_ops(), (NAME,))


class TestPluginOperatorRuns(unittest.TestCase):
    def setUp(self):
        register_operator(_double_op(), namespace="demo")
        self.addCleanup(lambda: OPS.pop(NAME, None))

    def test_reference_backend_picks_it_up(self):
        """注册即生效：参考后端的能力与执行都按算子表动态派生。"""
        self.assertIn(NAME, ReferenceBackend().capabilities())
        out = ReferenceBackend().execute(_graph(), {"x": np.ones((3, 4), dtype=np.float32)})
        np.testing.assert_allclose(out["y"], np.full((3, 4), 2.0, dtype=np.float32))

    def test_conformance_accepts_a_plugin_case(self):
        case = Case(
            name="plugin_double",
            graph=_graph(shape=(2, 3)),
            inputs={"x": np.arange(6, dtype=np.float32).reshape(2, 3)},
            ops=frozenset({NAME}),
        )
        report = run_conformance(ReferenceBackend(), cases=[case])
        self.assertTrue(report.ok, report.text())
        self.assertEqual(report.covered_ops, (NAME,))

    def test_library_portability_is_not_served_by_reference(self):
        """LIBRARY/VENDOR 插件算子不会被参考后端承接——拒绝本身是提示。"""
        name = "demo.libop"
        register_operator(_double_op(name=name, portability=Portability.LIBRARY), namespace="demo")
        self.addCleanup(lambda: OPS.pop(name, None))
        self.assertNotIn(name, ReferenceBackend().capabilities())
        with self.assertRaises(BackendCapabilityError):
            ReferenceBackend().execute(_graph(name=name), {"x": np.ones((3, 4), dtype=np.float32)})


if __name__ == "__main__":
    unittest.main()
