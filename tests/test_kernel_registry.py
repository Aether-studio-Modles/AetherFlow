"""算子注册表：注册即声明，查表即调度。

这一组固定三条边界：

- 名字拼错在**注册时**失败，而不是等第一次推理；
- 重复注册直接报错，不静默覆盖（否则"哪个内核在跑"不可知）；
- 能力声明由注册表派生，不可能与实现分叉。

另外用 ``inspect`` 做一次不需要 GPU 的静态对账：CUDA 的类级声明
（``CUDA_OPS``）与 ``_op_*`` 方法必须一一对应。构造期的对账在 GPU 机器上跑，
这条静态检查在没有显卡的机器上也能拦住同一类错误。
"""

import unittest

import numpy as np

from aetherflow import BackendCapabilityError, KernelRegistry
from aetherflow.backends.cuda import CUDA_AVAILABLE, CUDA_OPS, CudaBackend
from aetherflow.ir import Graph, f32
from aetherflow.ir.ops import OPS


def _noop(node, args, attrs):
    return "ok"


class TestRegistration(unittest.TestCase):
    def test_register_then_dispatch(self):
        r = KernelRegistry("fake")
        r.register("add", _noop)
        self.assertEqual(r.capabilities(), frozenset({"add"}))
        self.assertEqual(r.dispatch("add", None, [], {}), "ok")
        self.assertIn("add", r)
        self.assertEqual(len(r), 1)

    def test_unknown_operator_rejected(self):
        """拼错的名字必须在注册时报错，不能成为一个永远查不到的死条目。"""
        r = KernelRegistry("fake")
        with self.assertRaises(KeyError):
            r.register("conv3d", _noop)

    def test_duplicate_rejected(self):
        r = KernelRegistry("fake")
        r.register("relu", _noop)
        with self.assertRaises(ValueError):
            r.register("relu", _noop)

    def test_non_callable_rejected(self):
        r = KernelRegistry("fake")
        with self.assertRaises(TypeError):
            r.register("relu", 42)

    def test_dispatch_unregistered_is_capability_error(self):
        r = KernelRegistry("fake")
        with self.assertRaises(BackendCapabilityError) as ctx:
            r.dispatch("softmax", None, [], {}, context="节点 %y")
        message = str(ctx.exception)
        self.assertIn("fake", message)
        self.assertIn("softmax", message)
        self.assertIn("节点 %y", message)

    def test_dispatch_accepts_a_node_as_context(self):
        """热路径上不拼字符串：传节点对象，节点名只在报错时才展开。"""
        g = Graph("demo")
        x = g.input("x", f32[2])
        g.add(x, x)
        node = g.nodes[-1]
        r = KernelRegistry("fake")
        with self.assertRaises(BackendCapabilityError) as ctx:
            r.dispatch("softmax", None, [], {}, context=node)
        self.assertIn(f"节点 %{node.output.name}", str(ctx.exception))

    def test_decorator_form(self):
        r = KernelRegistry("fake")

        @r.register("relu")
        def kernel(node, args, attrs):
            return 1

        self.assertIs(r.get("relu"), kernel)
        self.assertIsNone(r.get("add"))

    def test_register_all(self):
        r = KernelRegistry("fake")
        r.register_all({"add": _noop, "mul": _noop})
        self.assertEqual(r.capabilities(), frozenset({"add", "mul"}))


class TestMultiImplementation(unittest.TestCase):
    """同名多实现：按 cost 选择，而不是按注册顺序。

    这组测试要固定的是"选择规则可解释"：代价小的赢、不可用的候选不参与、
    混合有代价与无代价的候选直接拒绝——因为那种混合下胜出的必然是注册顺序。
    """

    def _register_pair(self, r, cost_a, cost_b):
        r.register("matmul", lambda n, a, t: "a", cost=cost_a, label="a")
        r.register("matmul", lambda n, a, t: "b", cost=cost_b, label="b")

    def test_picks_lowest_cost(self):
        r = KernelRegistry("fake")
        self._register_pair(r, lambda n, a, t: 10.0, lambda n, a, t: 1.0)
        self.assertEqual(r.dispatch("matmul", None, [], {}), "b")

    def test_cost_sees_the_actual_arguments(self):
        """cost 与内核同参，所以可以按形状决定谁更便宜。"""
        r = KernelRegistry("fake")
        small = lambda n, a, t: 1.0 if a[0].size < 64 else 100.0
        big = lambda n, a, t: 100.0 if a[0].size < 64 else 1.0
        r.register("matmul", lambda n, a, t: "small", cost=small, label="small")
        r.register("matmul", lambda n, a, t: "big", cost=big, label="big")

        tiny = [np.zeros((4, 4), dtype=np.float32)]
        large = [np.zeros((16, 16), dtype=np.float32)]
        self.assertEqual(r.dispatch("matmul", None, tiny, {}), "small")
        self.assertEqual(r.dispatch("matmul", None, large, {}), "big")

    def test_all_unavailable_is_capability_error(self):
        """全部候选都不适用时，报的是"当前形状下没有内核"，不是内部异常。"""
        r = KernelRegistry("fake")
        self._register_pair(r, lambda n, a, t: None, lambda n, a, t: float("inf"))
        with self.assertRaises(BackendCapabilityError) as ctx:
            r.dispatch("matmul", None, [], {}, context="节点 %y")
        message = str(ctx.exception)
        self.assertIn("fake", message)
        self.assertIn("matmul", message)
        self.assertIn("节点 %y", message)
        self.assertIn("a", message)

    def test_mixing_ranked_and_unranked_rejected(self):
        """先注册无 cost 的实现，再追加带 cost 的候选必须失败。"""
        r = KernelRegistry("fake")
        r.register("matmul", lambda n, a, t: "plain")
        with self.assertRaises(ValueError) as ctx:
            r.register("matmul", lambda n, a, t: "ranked", cost=lambda n, a, t: 1.0)
        self.assertIn("cost", str(ctx.exception))

    def test_duplicate_label_rejected(self):
        r = KernelRegistry("fake")
        self._register_pair(r, lambda n, a, t: 1.0, lambda n, a, t: 2.0)
        with self.assertRaises(ValueError):
            r.register("matmul", lambda n, a, t: "c", cost=lambda n, a, t: 3.0, label="a")

    def test_non_callable_cost_rejected(self):
        r = KernelRegistry("fake")
        with self.assertRaises(TypeError):
            r.register("matmul", lambda n, a, t: "x", cost=1.0)

    def test_introspection(self):
        """多实现时 get() 只作自省，variants() 才是真实候选集合。"""
        r = KernelRegistry("fake")
        self._register_pair(r, lambda n, a, t: 1.0, lambda n, a, t: 2.0)
        self.assertEqual(len(r), 1)  # 算子个数，不是候选个数
        self.assertEqual(r.capabilities(), frozenset({"matmul"}))
        self.assertEqual([v.label for v in r.variants("matmul")], ["a", "b"])
        self.assertEqual(r.get("matmul").__name__, "<lambda>")
        self.assertEqual(r.variants("mm"), ())


class TestCudaDeclarations(unittest.TestCase):
    """不需要 GPU：类级声明与内核方法必须一一对应。"""

    def test_declared_ops_match_kernel_methods(self):
        implemented = {
            name[4:] for name in dir(CudaBackend) if name.startswith("_op_")
        }
        self.assertEqual(set(CUDA_OPS), implemented)

    def test_declared_ops_all_exist_in_ir(self):
        self.assertLessEqual(set(CUDA_OPS), set(OPS))


@unittest.skipUnless(CUDA_AVAILABLE, "需要 cupy 与 CUDA 设备")
class TestCudaRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = CudaBackend()

    def test_capabilities_come_from_registry(self):
        self.assertEqual(
            self.backend.capabilities(), self.backend.kernels.capabilities()
        )
        self.assertEqual(self.backend.capabilities(), CUDA_OPS)

    def test_kernels_are_bound_methods(self):
        self.assertEqual(self.backend.kernels.get("add").__name__, "_op_add")


if __name__ == "__main__":
    unittest.main()
