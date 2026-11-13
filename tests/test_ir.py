"""基础 IR 能力测试：类型系统、广播推导、图构建、verifier、后端能力协商。"""

import unittest

import numpy as np

from aetherflow import (
    BackendCapabilityError,
    Graph,
    Portability,
    ReferenceBackend,
    f32,
    i32,
    verify,
)
from aetherflow.ir.ops import OPS, Op, register
from aetherflow.ir.verifier import ERROR, verify_portable_only


class TestTypes(unittest.TestCase):
    def test_symbolic_type(self):
        t = f32["B", "S", 64]
        self.assertEqual(t.rank, 3)
        self.assertEqual(t.symbols, {"B", "S"})
        self.assertFalse(t.is_static)
        self.assertEqual(str(t), "f32[B, S, 64]")
        with self.assertRaises(ValueError):
            t.numel()

    def test_static_type(self):
        t = f32[2, 3, 4]
        self.assertTrue(t.is_static)
        self.assertEqual(t.numel(), 24)
        self.assertEqual(t.resolve({}), t)

    def test_resolve_binding(self):
        t = f32["B", 64]
        self.assertEqual(t.resolve({"B": 8}), f32[8, 64])
        with self.assertRaises(KeyError):
            t.resolve({})

    def test_reject_bad_symbol(self):
        with self.assertRaises(ValueError):
            f32["B S"]  # 含空格，不是合法标识符

    def test_symbol_clashing_with_dtype_name_is_rejected(self):
        """形状符号叫 ``f32`` 时，``f32[f32, 4]`` 与属性里的 ``dtype="f32"``
        无法区分：属性解析会把 dtype 换成整数。必须在类型构造期就拒绝。"""
        for name in ("f32", "i8", "bool"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    f32[name, 4]


class TestInference(unittest.TestCase):
    def test_broadcast_between_symbols(self):
        g = Graph()
        x = g.input("x", f32["B", 1, 64])
        y = g.input("y", f32["B", "S", 64])
        z = g.add(x, y)
        self.assertEqual(z.type, f32["B", "S", 64])

    def test_broadcast_static_conflict(self):
        g = Graph()
        x = g.input("x", f32[2, 3])
        y = g.input("y", f32[2, 4])
        with self.assertRaises(ValueError):
            g.add(x, y)

    def test_symbol_against_static_non_one_is_rejected(self):
        """``(B, 2) + (3, 2)`` 在 B=1 时结果是 ``(3, 2)``，在 B=3 时是
        ``(3, 2)``，在别的 B 下又非法——结果维究竟是静态 3 还是符号 B，
        IR 表达不了。此前取符号 B，等于把"B 就是结果维"写进类型。"""
        g = Graph()
        x = g.input("x", f32["B", 2])
        y = g.input("y", f32[3, 2])
        with self.assertRaises(ValueError) as ctx:
            g.add(x, y)
        self.assertIn("无法在 IR 中表示", str(ctx.exception))

    def test_two_different_symbols_are_rejected(self):
        g = Graph()
        x = g.input("x", f32["B", 4])
        y = g.input("y", f32["S", 4])
        with self.assertRaises(ValueError):
            g.add(x, y)

    def test_symbol_against_static_one_still_broadcasts(self):
        """静态 1 与符号广播是确定的：结果就是那个符号，不需要表达选择。"""
        g = Graph()
        x = g.input("x", f32["B", 4])
        y = g.input("y", f32[1, 4])
        self.assertEqual(g.add(x, y).type, f32["B", 4])

    def test_binary_dtype_mismatch(self):
        g = Graph()
        x = g.input("x", f32[2, 3])
        y = g.input("y", i32[2, 3])
        with self.assertRaises(TypeError):
            g.add(x, y)

    def test_matmul_inner_dim_mismatch(self):
        g = Graph()
        a = g.input("a", f32["B", 8, 16])
        b = g.input("b", f32[32, 4])
        with self.assertRaises(ValueError):
            g.matmul(a, b)

    def test_reshape_requires_defined_symbols(self):
        g = Graph()
        x = g.input("x", f32["B", 64])
        with self.assertRaises(ValueError):
            g.reshape(x, shape=("B", "T"))

    def test_unknown_attr_rejected(self):
        g = Graph()
        x = g.input("x", f32[4, 4])
        with self.assertRaises(TypeError):
            g.softmax(x, dim=-1)


class TestGraphAndVerifier(unittest.TestCase):
    def test_ir_text_and_topology(self):
        g = Graph("f")
        x = g.input("x", f32["B", 64])
        w = g.input("w", f32[64, 64])
        y = g.matmul(x, w)
        g.output(y, "y")
        text = g.to_ir()
        self.assertIn("func @f", text)
        self.assertIn("%v2 = matmul(%x, %w)", text)
        self.assertIn("return %v2", text)
        self.assertEqual(verify(g), [])

    def test_verifier_catches_type_tampering(self):
        g = Graph()
        x = g.input("x", f32[4])
        y = g.add(x, x)
        y.type = f32[5]
        g.output(y)
        issues = verify(g)
        self.assertTrue(any(i.level == ERROR and "类型不一致" in i.message for i in issues))

    def test_verifier_requires_output(self):
        g = Graph()
        g.input("x", f32[4])
        issues = verify(g)
        self.assertTrue(any("没有声明任何输出" in i.message for i in issues))

    def test_cross_graph_reference_rejected(self):
        g1 = Graph()
        g2 = Graph()
        x = g1.input("x", f32[4])
        with self.assertRaises(ValueError):
            g2.add_node("add", [x, x])

    def test_foreign_value_with_identical_name_and_type_is_rejected(self):
        """同名同形的外图输入不能按"结构相等"混进来。

        ``Value`` 若用 dataclass 的默认相等，两张图各自的第一个输入 ``%x``
        会被判为相等；构图期放行、执行时却按本图的 id 去读，读到的是一条
        完全无关的边。
        """
        g1 = Graph()
        g2 = Graph()
        foreign = g1.input("x", f32[4])
        local = g2.input("x", f32[4])
        self.assertIsNot(foreign, local)
        self.assertNotEqual(foreign, local)
        with self.assertRaises(ValueError):
            g2.add_node("add", [foreign, local])
        with self.assertRaises(ValueError):
            g2.output(foreign, "y")

    def test_revision_tracks_structure_changes(self):
        g = Graph()
        self.assertEqual(g.revision, 0)
        x = g.input("x", f32[4])
        self.assertEqual(g.revision, 1)
        y = g.add(x, x)
        self.assertEqual(g.revision, 2)
        g.output(y)
        self.assertEqual(g.revision, 3)


class TestExecutorPlanCache(unittest.TestCase):
    """执行计划按图缓存，图结构变了必须失效（5.72）。"""

    def test_growing_a_graph_invalidates_the_plan(self):
        from aetherflow import ReferenceExecutor

        g = Graph()
        x = g.input("x", f32[4])
        y = g.add(x, x)
        g.output(y, "y")
        executor = ReferenceExecutor()
        feed = {"x": np.ones(4, dtype=np.float32)}

        first = executor.run(g, feed)
        self.assertTrue(np.allclose(first["y"], 2.0))

        # 往同一张图里追加节点并声明新输出：缓存的计划里没有它，必须重建
        z = g.mul(y, y)
        g.output(z, "z")
        second = executor.run(g, feed)
        self.assertTrue(np.allclose(second["y"], 2.0))
        self.assertTrue(np.allclose(second["z"], 4.0))

    def test_static_and_symbolic_graphs_both_repeat(self):
        from aetherflow import ReferenceExecutor

        executor = ReferenceExecutor()
        static = Graph()
        sx = static.input("x", f32[4])
        static.output(static.silu(sx), "y")

        symbolic = Graph()
        yx = symbolic.input("x", f32["B", 4])
        symbolic.output(symbolic.silu(yx), "y")

        for graph, shape in ((static, (4,)), (symbolic, (3, 4))):
            feed = {"x": np.full(shape, 0.5, dtype=np.float32)}
            a = executor.run(graph, feed)
            b = executor.run(graph, feed)
            # 第二次走的是缓存下来的计划，结果必须与第一次逐位相同
            self.assertTrue(np.array_equal(a["y"], b["y"]))

    def test_shape_switch_invalidates_the_resolved_attrs(self):
        """符号图的 attrs / 期望类型按绑定备忘，换形状必须重算而不是沿用。"""
        from aetherflow import ReferenceExecutor

        g = Graph()
        x = g.input("x", f32["B", "S", 4])
        g.output(g.reshape(x, shape=("B", "S", 4)), "y")
        executor = ReferenceExecutor()

        # 交替形状：中间那次会命中上一次的备忘，第三次再切回来必须重新解析
        for shape in ((2, 3, 4), (5, 1, 4), (2, 3, 4)):
            feed = {"x": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}
            out = executor.run(g, feed)["y"]
            self.assertEqual(out.shape, shape)
            self.assertTrue(np.array_equal(out, feed["x"]))

    def test_resolve_attrs_replaces_symbols_but_not_dtype_literals(self):
        from aetherflow.runtime.executor import resolve_attrs

        attrs = {"shape": ["B", 4], "dtype": "f32", "axis": (0,)}
        resolved = resolve_attrs(attrs, {"B": 3})
        self.assertEqual(resolved, {"shape": [3, 4], "dtype": "f32", "axis": (0,)})


class TestGraphSeal(unittest.TestCase):
    """AF-32：``seal()`` 把"构建后只读"从注释变成契约。

    执行计划与捕获计划都按 ``revision`` 复用；只要还能改图，"计划仍命中
    但图已经变了"就是一条静默错误通道。这些用例锁住的是"改不动"本身。
    """

    def _build(self):
        g = Graph("sealed")
        x = g.input("x", f32[2, 3])
        y = g.relu(x)
        c = g.constant(1.5, shape=(2, 3))
        g.output(g.add(y, c), "out")
        return g, x

    def test_seal_rejects_structure_changes_and_is_idempotent(self):
        g, x = self._build()
        revision = g.revision
        self.assertFalse(g.sealed)
        self.assertIs(g.seal(), g)
        self.assertTrue(g.sealed)
        self.assertEqual(g.revision, revision)

        with self.assertRaises(RuntimeError):
            g.input("y", f32[2, 3])
        with self.assertRaises(RuntimeError):
            g.relu(x)  # 语法糖最终也走 add_node
        with self.assertRaises(RuntimeError):
            g.output(x, "x2")

        # 幂等：重复 seal 既不抛错也不推进版本号
        self.assertIs(g.seal(), g)
        self.assertEqual(g.revision, revision)

    def test_seal_freezes_values_nodes_and_attrs(self):
        g, x = self._build()
        g.seal()
        constant = g.nodes[1]
        # 顶层 list 被换成 tuple：只包只读外壳的话 append 仍能绕过
        self.assertEqual(constant.attrs["shape"], (2, 3))

        with self.assertRaises(AttributeError):
            x.name = "q"
        with self.assertRaises(AttributeError):
            x.producer = constant
        with self.assertRaises(AttributeError):
            constant.output = x
        with self.assertRaises(TypeError):
            constant.attrs["value"] = 2.0
        with self.assertRaises(AttributeError):
            constant.attrs["shape"].append(4)

    def test_seal_freezes_constant_payloads(self):
        """审计 P1：``data`` 曾豁免冻结，seal 后仍能原地改常量内容。

        两条路都要堵：调用方手里留着原 list，以及 ``attrs["data"][0]``
        的嵌套 list。前者在 ``constant`` 写入时就转成 tuple，后者是同一份
        嵌套 tuple 的自然结果。
        """
        g = Graph("payload")
        holder = [[1.0, 2.0], [3.0, 4.0]]
        value = g.constant(dtype=f32, shape=(2, 2), data=holder)
        g.output(value, "c")
        g.seal()

        holder[0].append(5.0)
        with self.assertRaises(AttributeError):
            value.producer.attrs["data"][0].append(6.0)
        np.testing.assert_array_equal(
            ReferenceBackend().execute(g, {})["c"],
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )

    def test_seal_keeps_read_only_surface_execution_and_plan_cache(self):
        from aetherflow import ReferenceExecutor, dumps, loads, to_dict

        g, _ = self._build()
        g.seal()
        feed = {"x": np.array([[-1.0, 2.0, 0.0], [1.0, -1.0, 3.0]], dtype=np.float32)}
        np.testing.assert_allclose(
            ReferenceBackend().execute(g, feed)["out"],
            [[1.5, 3.5, 1.5], [2.5, 1.5, 4.5]],
        )

        # 只读访问面与序列化不受影响：seal 冻的是写入，不是读取
        self.assertEqual(g.output_names, ["out"])
        self.assertIs(g.find("x"), g.inputs[0])
        self.assertIn("relu", g.op_names())
        self.assertIn("%v1 = relu(%x)", g.to_ir())
        self.assertEqual(to_dict(loads(dumps(g))), to_dict(g))

        # 版本号不再变化，执行计划因此可以跨执行长期命中
        executor = ReferenceExecutor()
        plan = executor._plan(g)
        executor.run(g, feed)
        self.assertIs(executor._plan(g), plan)


class TestPortability(unittest.TestCase):
    def test_vendor_node_rejected_by_reference_backend(self):
        register(
            Op(
                name="_test_vendor_op",
                infer=lambda inputs, attrs: inputs[0],
                reference=lambda args, attrs: args[0],
                portability=Portability.VENDOR,
            )
        )
        try:
            g = Graph()
            x = g.input("x", f32[4])
            y = g.add_node("_test_vendor_op", [x])
            g.output(y)
            self.assertTrue(any("不可移植" in i.message for i in verify_portable_only(g)))
            with self.assertRaises(BackendCapabilityError):
                ReferenceBackend().execute(g, {"x": np.zeros(4, dtype=np.float32)})
        finally:
            del OPS["_test_vendor_op"]

    def test_reference_backend_lists_portable_ops_only(self):
        caps = ReferenceBackend().capabilities()
        self.assertIn("matmul", caps)
        self.assertIn("softmax", caps)
        self.assertIn("layernorm", caps)

    def test_capability_gap_detected(self):
        g = Graph()
        x = g.input("x", f32[4])
        g.output(g.silu(x))
        issues = verify(g, capabilities={"add"})
        self.assertTrue(any("不支持算子 silu" in i.message for i in issues))


class TestReferenceExecution(unittest.TestCase):
    def test_symbol_binding_conflict(self):
        from aetherflow.runtime import ShapeMismatchError

        g = Graph()
        x = g.input("x", f32["B", 2])
        y = g.input("y", f32["B", 3])
        g.output(x)
        with self.assertRaises(ShapeMismatchError):
            ReferenceBackend().execute(
                g,
                {
                    "x": np.zeros((4, 2), dtype=np.float32),
                    "y": np.zeros((5, 3), dtype=np.float32),
                },
            )

    def test_same_graph_runs_at_multiple_shapes(self):
        """符号维的核心价值：一次构图，多形状执行，无需重编译。"""
        g = Graph()
        x = g.input("x", f32["B", "S", 8])
        w = g.input("w", f32[8, 4])
        g.output(g.matmul(x, w), "y")

        backend = ReferenceBackend()
        for b, s in ((1, 3), (2, 5), (7, 1)):
            xv = np.random.rand(b, s, 8).astype(np.float32)
            wv = np.random.rand(8, 4).astype(np.float32)
            out = backend.execute(g, {"x": xv, "w": wv})["y"]
            np.testing.assert_allclose(out, xv @ wv, rtol=1e-5, atol=1e-5)
            self.assertEqual(out.shape, (b, s, 4))

    def test_runtime_shape_checked_against_ir(self):
        from aetherflow.runtime import ShapeMismatchError

        g = Graph()
        x = g.input("x", f32[2, 3])
        g.output(g.relu(x))
        with self.assertRaises(ShapeMismatchError):
            ReferenceBackend().execute(g, {"x": np.zeros((2, 4), dtype=np.float32)})

    def test_input_is_converted_to_the_declared_dtype_at_the_boundary(self):
        """AF-10：声明 f32 就必须按 f32 算，不能先用 f64 再在输出降回去。

        ``1 + 2**-30`` 在 f32 里就是 1.0，在 f64 里还留着那一点。若入口
        不转换，``x - 1`` 会得到 ``9.3e-10``；转换后应为 0。
        """
        g = Graph()
        x = g.input("x", f32[1])
        one = g.add_node(
            "constant", [], {"dtype": "f32", "shape": [1], "value": 1.0}
        )
        g.output(g.sub(x, one), "y")

        backend = ReferenceBackend()
        f64_feed = {"x": np.array([1.0 + 2.0**-30], dtype=np.float64)}
        out = backend.execute(g, f64_feed)["y"]
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(out, np.zeros(1, np.float32))

    def _ids_graph(self):
        g = Graph()
        ids = g.input("ids", i32[2])
        g.output(g.add(ids, ids), "y")
        return g

    def test_float_input_cannot_be_silently_truncated_into_an_index(self):
        """审计 P0：``0.9`` 交给 i32 声明会变成 0，token 在入口就被改掉。"""
        with self.assertRaisesRegex(TypeError, "浮点转整数"):
            ReferenceBackend().execute(
                self._ids_graph(), {"ids": np.array([0.9, 1.0], np.float32)}
            )

    def test_integer_narrowing_checks_the_range_before_casting(self):
        """``2**32`` 塞进 i32 会包成 0；越界必须在入口报错而不是静默取模。"""
        backend = ReferenceBackend()
        with self.assertRaisesRegex(ValueError, "超出范围"):
            backend.execute(
                self._ids_graph(), {"ids": np.array([2**32, 0], np.int64)}
            )
        with self.assertRaisesRegex(ValueError, "超出范围"):
            backend.execute(
                self._ids_graph(), {"ids": np.array([0, -(2**31) - 1], np.int64)}
            )
        # 在范围内则照常窄化
        out = backend.execute(
            self._ids_graph(), {"ids": np.array([3, 4], np.int64)}
        )["y"]
        np.testing.assert_array_equal(out, np.array([6, 8], np.int32))

    def test_boolean_is_not_an_index(self):
        with self.assertRaisesRegex(TypeError, "布尔"):
            ReferenceBackend().execute(
                self._ids_graph(), {"ids": np.array([True, False])}
            )

    def test_integer_to_float_is_still_lossless(self):
        g = Graph()
        x = g.input("x", f32[2])
        g.output(g.mul(x, x), "y")
        out = ReferenceBackend().execute(g, {"x": np.array([2, 3], np.int64)})["y"]
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(out, np.array([4.0, 9.0], np.float32))

    def test_embedding_id_bounds_are_the_same_rule_on_every_backend(self):
        """负下标在 numpy 里会回绕到倒数几行；参考语义必须与设备侧一样拒绝。"""
        g = Graph()
        table = g.input("table", f32[3, 2])
        ids = g.input("ids", i32[1])
        g.output(g.embedding(table, ids), "y")
        feed = {"table": np.zeros((3, 2), np.float32)}
        for bad in (-1, 3):
            with self.assertRaisesRegex(ValueError, r"下标必须落在 \[0, 3\)"):
                ReferenceBackend().execute(g, {**feed, "ids": np.array([bad], np.int32)})


class TestConv2D(unittest.TestCase):
    """CNN 的第一块砖：先把 NCHW 形状语义与参考语义定死，后端实现随后追。

    这里的期望值尽量手算（全 1 输入、全 1 权重），而不是再写一遍参考实现——
    否则测试与实现同错时会一起"通过"。
    """

    def _run(self, x_shape, w_shape, feed, attrs=None, bias_len=None):
        g = Graph("conv")
        x = g.input("x", f32[x_shape])
        w = g.input("w", f32[w_shape])
        args = [x, w]
        if bias_len is not None:
            b = g.input("b", f32[bias_len])
            args.append(b)
            feed["b"] = np.zeros((bias_len,), dtype=np.float32)
        g.output(g.add_node("conv2d", args, attrs or {}), "y")
        return ReferenceBackend().execute(g, feed)["y"]

    def test_3x3_valid_full_sum(self):
        """3x3、3 通道、无 padding：每个输出是 27 个 1 之和。"""
        x = np.ones((1, 3, 5, 5), dtype=np.float32)
        w = np.ones((2, 3, 3, 3), dtype=np.float32)
        out = self._run(
            (1, 3, 5, 5), (2, 3, 3, 3), {"x": x, "w": w},
            {"stride": (1, 1)}, bias_len=2,
        )
        self.assertEqual(out.shape, (1, 2, 3, 3))
        np.testing.assert_allclose(out, 27.0)

    def test_3x3_pad1_keeps_spatial_size_and_counts_edges(self):
        """same padding 保持 5x5；角落只覆盖 2x2 个位置（12），中心是 27。"""
        x = np.ones((1, 3, 5, 5), dtype=np.float32)
        w = np.ones((2, 3, 3, 3), dtype=np.float32)
        out = self._run(
            (1, 3, 5, 5), (2, 3, 3, 3), {"x": x, "w": w},
            {"stride": (1, 1), "padding": (1, 1)}, bias_len=2,
        )
        self.assertEqual(out.shape, (1, 2, 5, 5))
        np.testing.assert_allclose(out[:, :, 0, 0], 12.0)
        np.testing.assert_allclose(out[:, :, 0, 1], 18.0)
        np.testing.assert_allclose(out[:, :, 1, 1], 27.0)
        np.testing.assert_allclose(out[:, :, 4, 4], 12.0)

    def test_1x1_mixes_channels(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((1, 2, 3, 4)).astype(np.float32)
        w = rng.standard_normal((3, 2, 1, 1)).astype(np.float32)
        out = self._run((1, 2, 3, 4), (3, 2, 1, 1), {"x": x, "w": w})
        self.assertEqual(out.shape, (1, 3, 3, 4))
        expected = np.einsum("nchw,oc->nohw", x, w[:, :, 0, 0])
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_stride_batch_and_non_square(self):
        """stride 2 与非方形核同时生效：形状按 floor 公式，值仍是 12 个 1。"""
        x = np.ones((2, 2, 6, 4), dtype=np.float32)
        w = np.ones((5, 2, 3, 2), dtype=np.float32)
        out = self._run((2, 2, 6, 4), (5, 2, 3, 2), {"x": x, "w": w},
                        {"stride": (2, 1)})
        self.assertEqual(out.shape, (2, 5, 2, 3))
        np.testing.assert_allclose(out, 12.0)

    def test_dilation_expands_receptive_field(self):
        """dilation=2 的 2x2 核覆盖 3x3 区域，5x5 输入只剩 3x3 输出。"""
        x = np.ones((1, 1, 5, 5), dtype=np.float32)
        w = np.ones((1, 1, 2, 2), dtype=np.float32)
        out = self._run((1, 1, 5, 5), (1, 1, 2, 2), {"x": x, "w": w},
                        {"dilation": (2, 2)})
        self.assertEqual(out.shape, (1, 1, 3, 3))
        np.testing.assert_allclose(out, 4.0)

    def test_symbolic_batch_and_spatial_kept(self):
        """符号 batch 与 1x1 卷积下的符号空间维：一次构图，多形状执行。"""
        g = Graph()
        x = g.input("x", f32["N", 2, "H", "W"])
        w = g.input("w", f32[3, 2, 1, 1])
        g.output(g.conv2d(x, w), "y")
        self.assertEqual(g.outputs[0].type, f32["N", 3, "H", "W"])

        rng = np.random.default_rng(1)
        xv = rng.standard_normal((4, 2, 3, 5)).astype(np.float32)
        wv = rng.standard_normal((3, 2, 1, 1)).astype(np.float32)
        out = ReferenceBackend().execute(g, {"x": xv, "w": wv})["y"]
        self.assertEqual(out.shape, (4, 3, 3, 5))

    def test_symbolic_spatial_derives_expression(self):
        """空间维是符号时，卷积输出用表达式表达，而不是拒绝整张图。"""
        g = Graph()
        x = g.input("x", f32["N", 3, "H", "W"])
        w = g.input("w", f32[4, 3, 3, 3])

        # same padding + stride 1：结构上化简回 H / W 本身
        y = g.conv2d(x, w, padding=(1, 1))
        self.assertEqual(y.type, f32["N", 4, "H", "W"])

        # stride 2：输出长度是派生维
        z = g.conv2d(x, w, stride=(2, 2), padding=(1, 1))
        self.assertEqual(str(z.type), "f32[N, 4, (((H - 1) / 2) + 1), (((W - 1) / 2) + 1)]")
        self.assertEqual(z.type.resolve({"N": 2, "H": 8, "W": 10}), f32[2, 4, 4, 5])

    def test_symbolic_spatial_runs_multiple_resolutions(self):
        g = Graph()
        x = g.input("x", f32["N", 3, "H", "W"])
        w = g.input("w", f32[4, 3, 3, 3])
        g.output(g.conv2d(x, w, stride=(2, 2), padding=(1, 1)), "y")

        rng = np.random.default_rng(3)
        wv = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
        for h, w_ in ((8, 8), (12, 10)):
            xv = rng.standard_normal((2, 3, h, w_)).astype(np.float32)
            out = ReferenceBackend().execute(g, {"x": xv, "w": wv})["y"]
            self.assertEqual(out.shape, (2, 4, (h + 2 - 3) // 2 + 1, (w_ + 2 - 3) // 2 + 1))

    def test_invalid_attributes_and_shapes_rejected(self):
        cases = [
            ("groups", dict(groups=2), ValueError),
            ("padding", dict(padding=(-1, 0)), ValueError),
            ("stride", dict(stride=(0, 1)), ValueError),
        ]
        for name, attrs, exc in cases:
            with self.subTest(attr=name):
                g = Graph()
                x = g.input("x", f32[1, 3, 8, 8])
                w = g.input("w", f32[4, 3, 3, 3])
                with self.assertRaises(exc):
                    g.conv2d(x, w, **attrs)

        g = Graph()
        x = g.input("x", f32[1, 3, 8, 8])
        w = g.input("w", f32[4, 2, 3, 3])
        with self.assertRaises(ValueError):
            g.conv2d(x, w)

        g = Graph()
        x = g.input("x", f32[1, 3, 8, 8])
        w = g.input("w", f32[4, 3, 3, 3])
        b = g.input("b", f32[5])
        with self.assertRaises(ValueError):
            g.conv2d(x, w, b)


class TestMaxPool2D(unittest.TestCase):
    """池化只有形状与"取最大"两条语义，但边界决定了它会不会悄悄吃掉 padding。

    期望值用手算的递增矩阵，不用参考实现反算。
    """

    def _run(self, x, attrs):
        shape = x.shape
        g = Graph("pool")
        v = g.input("x", f32[shape])
        g.output(g.maxpool2d(v, **attrs), "y")
        return ReferenceBackend().execute(g, {"x": x})["y"]

    def test_default_stride_equals_kernel(self):
        """stride 缺省等于 kernel：4x4 切成不重叠的 2x2 块。"""
        x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        out = self._run(x, {"kernel": 2})
        self.assertEqual(out.shape, (1, 1, 2, 2))
        np.testing.assert_allclose(out[0, 0], [[5.0, 7.0], [13.0, 15.0]])

    def test_overlapping_windows(self):
        """kernel=2、stride=1：窗口互相重叠，3x3 只缩到 2x2。"""
        x = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
        out = self._run(x, {"kernel": 2, "stride": 1})
        self.assertEqual(out.shape, (1, 1, 2, 2))
        np.testing.assert_allclose(out[0, 0], [[4.0, 5.0], [7.0, 8.0]])

    def test_padding_cells_do_not_take_part_in_max(self):
        """padding 填 -inf 而不是 0：否则负输入会被边界"抬"成 0。"""
        x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        out = self._run(x, {"kernel": 2, "stride": 2, "padding": 1})
        self.assertEqual(out.shape, (1, 1, 3, 3))
        np.testing.assert_allclose(
            out[0, 0],
            [[0.0, 2.0, 3.0], [8.0, 10.0, 11.0], [12.0, 14.0, 15.0]],
        )

        neg = np.full((1, 1, 4, 4), -5.0, dtype=np.float32)
        np.testing.assert_allclose(self._run(neg, {"kernel": 2, "padding": 1}), -5.0)

    def test_non_square_kernel_stride_and_batch(self):
        x = np.ones((2, 3, 5, 6), dtype=np.float32)
        out = self._run(x, {"kernel": (3, 2), "stride": (2, 1)})
        self.assertEqual(out.shape, (2, 3, 2, 5))
        np.testing.assert_allclose(out, 1.0)

    def test_symbolic_batch_and_1x1_spatial_kept(self):
        g = Graph()
        x = g.input("x", f32["N", 3, "H", "W"])
        g.output(g.maxpool2d(x, kernel=1), "y")
        self.assertEqual(g.outputs[0].type, f32["N", 3, "H", "W"])

        rng = np.random.default_rng(2)
        xv = rng.standard_normal((4, 3, 5, 7)).astype(np.float32)
        out = ReferenceBackend().execute(g, {"x": xv})["y"]
        np.testing.assert_allclose(out, xv)

    def test_symbolic_spatial_derives_expression(self):
        """池化的空间维同样是符号时推导成表达式，并能直接跑。"""
        g = Graph()
        x = g.input("x", f32["N", 3, "H", "W"])
        y = g.maxpool2d(x, kernel=2)
        g.output(y, "y")
        self.assertEqual(str(y.type), "f32[N, 3, (((H - 2) / 2) + 1), (((W - 2) / 2) + 1)]")
        self.assertEqual(y.type.resolve({"N": 2, "H": 6, "W": 9}), f32[2, 3, 3, 4])

        rng = np.random.default_rng(4)
        xv = rng.standard_normal((2, 3, 6, 9)).astype(np.float32)
        out = ReferenceBackend().execute(g, {"x": xv})["y"]
        self.assertEqual(out.shape, (2, 3, 3, 4))
        # 右侧不足一个窗口的部分被丢弃，所以先裁到 6x8 再按窗口取最大
        windows = xv[:, :, :6, :8].reshape(2, 3, 3, 2, 4, 2).max(axis=(3, 5))
        np.testing.assert_allclose(out, windows)

    def test_invalid_attributes_rejected(self):
        for name, attrs, exc in [
            ("kernel0", dict(kernel=0), ValueError),
            ("stride0", dict(kernel=2, stride=0), ValueError),
            ("negative_padding", dict(kernel=2, padding=-1), ValueError),
            ("unknown_attr", dict(kernel=2, dilation=1), TypeError),
        ]:
            with self.subTest(attr=name):
                g = Graph()
                x = g.input("x", f32[1, 2, 6, 6])
                with self.assertRaises(exc):
                    g.maxpool2d(x, **attrs)

        g = Graph()
        x = g.input("x", f32[2, 6])
        with self.assertRaises(ValueError):
            g.maxpool2d(x, kernel=2)


class TestSymExpr(unittest.TestCase):
    """符号维表达式：常量折叠、恒等式化简、求值与序列化。"""

    def test_constant_folding(self):
        from aetherflow.ir.types import sym_add, sym_binary

        self.assertEqual(sym_add(2, 3), 5)
        self.assertEqual(sym_binary("floordiv", 7, 2), 3)
        self.assertEqual(sym_binary("mul", 4, 5), 20)

    def test_identity_simplification(self):
        from aetherflow.ir.types import sym_add, sym_binary

        self.assertEqual(sym_add("H", 0), "H")
        self.assertEqual(sym_binary("mul", "H", 1), "H")
        self.assertEqual(sym_binary("floordiv", "H", 1), "H")

    def test_constant_absorption_recovers_the_symbol(self):
        """``(H - 1) + 1`` 必须化简回 ``H``，否则形状比较会凭空分叉。"""
        from aetherflow.ir.types import sym_add

        self.assertEqual(sym_add(sym_add("H", -1), 1), "H")

    def test_evaluation_and_symbols(self):
        from aetherflow.ir.types import dim_resolve, sym_add, sym_binary

        expr = sym_add(sym_binary("floordiv", sym_add("H", -1), 2), 1)
        self.assertEqual(expr.symbols, frozenset({"H"}))
        self.assertEqual(dim_resolve(expr, {"H": 9}), 5)
        with self.assertRaises(KeyError):
            dim_resolve(expr, {})

    def test_printing_is_fully_parenthesized(self):
        from aetherflow.ir.types import sym_add, sym_binary

        expr = sym_add(sym_binary("floordiv", sym_add("H", -1), 2), 1)
        self.assertEqual(str(expr), "(((H - 1) / 2) + 1)")

    def test_serialization_roundtrip_keeps_expression(self):
        from aetherflow import dumps, loads, to_dict

        g = Graph("sym")
        x = g.input("x", f32["N", 3, "H", "W"])
        w = g.input("w", f32[4, 3, 3, 3])
        g.output(g.conv2d(x, w, stride=(2, 2), padding=(1, 1)), "y")

        restored = loads(dumps(g))
        self.assertEqual(to_dict(restored), to_dict(g))
        self.assertEqual(str(restored.outputs[0].type), str(g.outputs[0].type))


class TestSerialization(unittest.TestCase):
    """IR 是唯一契约，那它必须能离开进程、并且能被版本拒收。"""

    def _build(self):
        g = Graph("demo")
        x = g.input("x", f32["B", "S", 8])
        w = g.input("w", f32[8, 4])
        h = g.matmul(x, w)
        t = g.transpose(h, perm=(0, 2, 1))
        s = g.softmax(t, axis=-1)
        g.output(g.relu(s), "y")
        g.output(g.reduce_sum(s, axis=-1, keepdims=True), "aux")
        return g

    def test_roundtrip_preserves_graph(self):
        from aetherflow import dumps, loads, to_dict

        g = self._build()
        restored = loads(dumps(g))

        self.assertEqual(restored.name, g.name)
        self.assertEqual(restored.ir_version, g.ir_version)
        self.assertEqual(to_dict(restored), to_dict(g))
        self.assertEqual(restored.op_names(), g.op_names())
        self.assertEqual(restored.output_names, g.output_names)
        self.assertEqual([str(v.type) for v in restored.values],
                         [str(v.type) for v in g.values])

        # 符号维与属性（perm 是 tuple）也必须原样回来
        self.assertEqual(restored.nodes[1].attrs["perm"], (0, 2, 1))
        self.assertIn("B", str(restored.inputs[0].type))

        rng = np.random.default_rng(0)
        feed = {
            "x": rng.standard_normal((2, 3, 8)).astype(np.float32),
            "w": rng.standard_normal((8, 4)).astype(np.float32),
        }
        backend = ReferenceBackend()
        for name in g.output_names:
            np.testing.assert_array_equal(
                backend.execute(restored, feed)[name], backend.execute(g, feed)[name]
            )

    def test_major_version_mismatch_rejected(self):
        from aetherflow import IRVersionError, from_dict, to_dict

        data = to_dict(self._build())
        data["ir_version"] = [2, 0]
        with self.assertRaises(IRVersionError):
            from_dict(data)

    def test_newer_minor_version_rejected(self):
        """次版本更高说明用了本实现不认识的特性，不能按旧规则硬读。"""
        from aetherflow import IRVersionError, from_dict, to_dict

        data = to_dict(self._build())
        data["ir_version"] = [1, 99]
        with self.assertRaises(IRVersionError):
            from_dict(data)

    def test_older_minor_version_accepted(self):
        from aetherflow import from_dict, to_dict

        data = to_dict(self._build())
        data["ir_version"] = [1, 0]
        self.assertEqual(len(from_dict(data).nodes), 5)

    def test_missing_version_rejected(self):
        from aetherflow import IRVersionError, from_dict, to_dict

        data = to_dict(self._build())
        del data["ir_version"]
        with self.assertRaises(IRVersionError):
            from_dict(data)

    def test_declared_type_mismatch_rejected(self):
        """声明类型与实现推导不一致 = 图作者与算子语义已经分叉，必须失败。"""
        from aetherflow import from_dict, to_dict

        data = to_dict(self._build())
        data["nodes"][0]["output"]["shape"] = ["B", "S", 999]
        with self.assertRaises(ValueError):
            from_dict(data)

    def test_unknown_op_rejected(self):
        from aetherflow import from_dict, to_dict

        data = to_dict(self._build())
        # 用确定不存在的名字：以前这里写 "gelu"，它后来成了真算子，用例就失效了
        data["nodes"][0]["op"] = "definitely_not_an_op"
        with self.assertRaises(KeyError):
            from_dict(data)

    def test_verifier_rejects_incompatible_version(self):
        g = self._build()
        g.ir_version = (2, 0)
        issues = verify(g)
        self.assertTrue(any("主版本" in i.message for i in issues))

    def test_user_names_that_look_automatic_roundtrip(self):
        """用户把输入命名成 ``v1`` 时，自动 SSA 名不能覆盖它。

        同名覆盖会让 ``_by_name`` 指向另一条边；序列化按名字重放后，输出会
        从 ``[0, 0]`` 静默变成 ``[-2, -4]``。
        """
        from aetherflow import dumps, loads

        g = Graph("collide")
        x = g.input("v1", f32[2])
        a = g.neg(x)
        y = g.add(x, a)
        g.output(y, "y")
        self.assertNotEqual(a.name, "v1")
        self.assertIs(g.find("v1"), x)

        feed = {"v1": np.zeros(2, np.float32)}
        backend = ReferenceBackend()
        want = backend.execute(g, feed)["y"]
        np.testing.assert_array_equal(want, [0.0, 0.0])
        restored = loads(dumps(g))
        np.testing.assert_array_equal(backend.execute(restored, feed)["y"], want)

    def test_inputs_declared_between_nodes_roundtrip(self):
        """先建节点、后声明输入的图也要能存下来。

        eager 前端就是按需长大、随时新增输入的；序列化如果只依赖"当前值
        个数"重新编号，重放时会得到另一串名字，节点引用直接断掉（审计
        AF-09）。节点输出带显式名字之后，回读把输入全部前置只是让值 id
        重排，引用仍按名字解析。
        """
        from aetherflow import dumps, loads, to_dict

        g = Graph("interleaved")
        x = g.input("x", f32[2])
        a = g.neg(x)
        w = g.input("w", f32[2])
        g.output(g.add(a, w), "y")

        feed = {
            "x": np.array([1.0, 2.0], np.float32),
            "w": np.array([10.0, 20.0], np.float32),
        }
        backend = ReferenceBackend()
        want = backend.execute(g, feed)["y"]
        restored = loads(dumps(g))

        # 回读把输入全部前置：w 的 id 从 2 变成 1，结构契约是名字不是 id
        self.assertEqual([v.id for v in g.inputs], [0, 2])
        self.assertEqual([v.id for v in restored.inputs], [0, 1])
        self.assertEqual([v.name for v in restored.inputs], [v.name for v in g.inputs])
        self.assertEqual(to_dict(restored), to_dict(g))
        self.assertEqual(dumps(restored), dumps(g))
        self.assertEqual(verify(restored), [])
        self.assertEqual(restored.op_names(), g.op_names())
        np.testing.assert_array_equal(backend.execute(restored, feed)["y"], want)

    def test_node_declared_before_any_input_roundtrip(self):
        """极端交错：第一个值就是节点输出（常量），此时一个输入都还没声明。"""
        from aetherflow import dumps, loads, to_dict

        g = Graph("nodes_first")
        c = g.constant(3.0, dtype=f32, shape=(2,))
        x = g.input("x", f32[2])
        g.output(g.add(x, c), "y")
        self.assertEqual(g.values[0].name, "v0")

        feed = {"x": np.array([1.0, 2.0], np.float32)}
        backend = ReferenceBackend()
        want = backend.execute(g, feed)["y"]
        restored = loads(dumps(g))
        self.assertEqual(to_dict(restored), to_dict(g))
        self.assertEqual(verify(restored), [])
        np.testing.assert_array_equal(backend.execute(restored, feed)["y"], want)


if __name__ == "__main__":
    unittest.main()
