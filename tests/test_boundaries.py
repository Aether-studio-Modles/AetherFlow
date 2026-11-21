"""边界值测试：按"入口 × 边界"组织，而不是零散地试几个值。

两类边界分开断言，这是整份文件的结构：

1. **非法值在构造期显式报错**——不静默截断、不静默回退、不把 ``None`` 当成
   一个合法轴；
2. **极端值有确定的 IEEE 语义**——``log(0) = -inf``、``0/0 = nan``、
   f16 溢出成 ``inf``。这些不是错误，但必须是**被钉住**的行为，而且不能在
   宿主上打 numpy 的 RuntimeWarning。

最后两组是长上下文与序列化：前者的契约是"分块续接的误差不随块数增长"，
后者是"``dumps`` 不替你把非法图修好，``verify`` 也不"。
"""

import json
import unittest
import warnings

import numpy as np

from aetherflow import (
    Graph,
    ReferenceBackend,
    bf16,
    dumps,
    f16,
    f32,
    loads,
    verify,
)
from aetherflow.ir.types import DType
from aetherflow.models import LMSpec, build_language_model_stateful

BACKEND = ReferenceBackend()


def _run(graph, feed):
    return BACKEND.execute(graph, feed)


def _assert_quiet(test, fn):
    """执行 ``fn`` 并要求没有 RuntimeWarning 落到宿主上。"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn()
    noisy = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    test.assertEqual(
        [str(w.message) for w in noisy], [], "浮点边界不应在宿主上打 RuntimeWarning"
    )
    return result


class TestShapeBoundaries(unittest.TestCase):
    """形状：0 元素、秩 0、空轴、越界轴、重复的轴。"""

    def test_empty_tensor_flows_through_elementwise(self):
        g = Graph("empty")
        x = g.input("x", f32[0, 3])
        g.output(g.relu(g.add(x, x)), "y")
        out = _run(g, {"x": np.zeros((0, 3), np.float32)})["y"]
        self.assertEqual(out.shape, (0, 3))

    def test_rank_zero_is_a_first_class_value(self):
        g = Graph("scalar")
        x = g.input("x", f32[()])
        g.output(g.mul(x, x), "y")
        self.assertEqual(_run(g, {"x": np.float32(3.0)})["y"], np.float32(9.0))

    def test_reduce_axis_table(self):
        """一张表说清每个轴形态的结果：合法给形状，非法给异常。"""
        shape = (2, 3, 4)

        def build(axis, keepdims):
            g = Graph("reduce")
            x = g.input("x", f32[shape])
            g.output(g.reduce_sum(x, axis=axis, keepdims=keepdims), "y")
            return g

        legal = {
            (0, False): (3, 4),
            (-1, False): (2, 3),
            ((0, 2), False): (3,),
            ((0, 2), True): (1, 3, 1),
            ((), False): (2, 3, 4),  # 空轴是恒等，不是"全归约"
        }
        for (axis, keepdims), want in legal.items():
            got = _run(build(axis, keepdims), {"x": np.ones(shape, np.float32)})
            self.assertEqual(got["y"].shape, want, f"axis={axis} keepdims={keepdims}")

        for axis in (3, -4, (0, 0), None):
            with self.assertRaises(ValueError, msg=repr(axis)):
                build(axis, False)

    def test_axis_none_points_at_the_explicit_form(self):
        g = Graph("all")
        x = g.input("x", f32[2, 3])
        with self.assertRaises(ValueError) as ctx:
            g.reduce_sum(x, axis=None)
        self.assertIn("tuple(range(2))", str(ctx.exception))

    def test_reshape_must_conserve_the_element_count(self):
        g = Graph("reshape")
        x = g.input("x", f32[4])
        with self.assertRaises(ValueError) as ctx:
            g.reshape(x, shape=(3,))
        self.assertIn("元素个数", str(ctx.exception))
        self.assertEqual(g.reshape(x, shape=(2, 2)).type.shape, (2, 2))

    def test_duplicate_names_are_rejected(self):
        with self.assertRaises(ValueError):
            g = Graph("dup")
            g.input("x", f32[2])
            g.input("x", f32[2])

        g = Graph("dup_out")
        x = g.input("x", f32[2])
        g.output(x, "y")
        with self.assertRaises(ValueError):
            g.output(x, "y")


class TestDtypeBoundaries(unittest.TestCase):
    def test_unknown_dtype_is_rejected(self):
        for bad in ("u8", "float32", ""):
            with self.assertRaises(ValueError, msg=repr(bad)):
                DType(bad)

    def test_bf16_is_carried_by_float32_on_the_reference_backend(self):
        """契约：bf16 在 numpy 里没有原生类型，参考后端借用 f32 承载。"""
        self.assertEqual(bf16.np_dtype, "float32")
        g = Graph("bf16")
        x = g.input("x", f32[2])
        y = g.cast(x, dtype=bf16)
        g.output(y, "y")
        self.assertEqual(y.type.dtype, bf16)
        out = _run(g, {"x": np.array([1.5, -2.0], np.float32)})["y"]
        self.assertEqual(out.dtype, np.float32)

    def test_f16_overflow_is_inf_not_an_error(self):
        g = Graph("f16")
        x = g.input("x", f16[2])
        g.output(g.add(x, x), "y")
        out = _assert_quiet(
            self, lambda: _run(g, {"x": np.array([3.5e4, 1.0], np.float16)})["y"]
        )
        self.assertTrue(np.isinf(out[0]))
        self.assertEqual(out[1], np.float16(2.0))


class TestNumericBoundaries(unittest.TestCase):
    """极端值：IEEE 语义 + 宿主安静。"""

    def _unary(self, name, values):
        g = Graph("unary")
        x = g.input("x", f32[len(values)])
        g.output(getattr(g, name)(x), "y")
        return _run(g, {"x": np.asarray(values, np.float32)})["y"]

    def test_division_by_zero_is_ieee_and_quiet(self):
        g = Graph("div")
        a = g.input("a", f32[4])
        b = g.input("b", f32[4])
        g.output(g.div(a, b), "y")
        out = _assert_quiet(
            self,
            lambda: _run(
                g,
                {
                    "a": np.array([1.0, -1.0, 0.0, 2.0], np.float32),
                    "b": np.zeros(4, np.float32),
                },
            )["y"],
        )
        self.assertTrue(np.isposinf(out[0]))
        self.assertTrue(np.isneginf(out[1]))
        self.assertTrue(np.isnan(out[2]))

    def test_log_domain_edges_are_ieee_and_quiet(self):
        out = _assert_quiet(
            self, lambda: self._unary("log", [0.0, -1.0, 1.0, np.e])
        )
        self.assertTrue(np.isneginf(out[0]))
        self.assertTrue(np.isnan(out[1]))
        self.assertEqual(out[2], 0.0)
        np.testing.assert_allclose(out[3], 1.0, rtol=1e-6)

    def test_sqrt_and_exp_edges_are_ieee_and_quiet(self):
        sqrt = _assert_quiet(self, lambda: self._unary("sqrt", [-1.0, 4.0]))
        self.assertTrue(np.isnan(sqrt[0]))
        self.assertEqual(sqrt[1], 2.0)

        exp = _assert_quiet(self, lambda: self._unary("exp", [1000.0, 0.0]))
        self.assertTrue(np.isposinf(exp[0]))
        self.assertEqual(exp[1], 1.0)

    def test_softmax_of_large_logits_stays_finite_and_normalized(self):
        g = Graph("softmax")
        x = g.input("x", f32[4])
        g.output(g.softmax(x, axis=-1), "y")
        out = _assert_quiet(
            self,
            lambda: _run(
                g,
                {"x": np.array([1000.0, 0.0, -1000.0, 999.0], np.float32)},
            )["y"],
        )
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertAlmostEqual(float(out.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(out[0]), 0.7310586, places=5)

    def test_nan_propagates_through_elementwise(self):
        out = _assert_quiet(
            self,
            lambda: self._unary("relu", [np.nan, -np.inf, np.inf]),
        )
        self.assertTrue(np.isnan(out[0]))
        self.assertEqual(out[1], 0.0)
        self.assertTrue(np.isposinf(out[2]))

    def test_reduce_max_over_an_empty_axis_is_explicit(self):
        g = Graph("rmax")
        x = g.input("x", f32[0])
        g.output(g.reduce_max(x, axis=0), "y")
        with self.assertRaises(ValueError):
            _run(g, {"x": np.zeros(0, np.float32)})


class TestSerializationBoundaries(unittest.TestCase):
    def test_graph_without_outputs_is_an_error_issue(self):
        issues = verify(Graph("empty"))
        self.assertTrue(any(i.level == "error" for i in issues))
        self.assertTrue(any("没有声明任何输出" in i.message for i in issues))

    def test_truncated_json_is_rejected(self):
        with self.assertRaises(json.JSONDecodeError):
            loads('{"nodes": [')

    def test_dumps_does_not_repair_an_invalid_graph(self):
        """dumps 只做序列化，不顺手补输出；坏图往返之后仍然是坏的。"""
        payload = dumps(Graph("empty"))
        clone = loads(payload)
        self.assertEqual(len(clone.nodes), 0)
        self.assertTrue(any(i.level == "error" for i in verify(clone)))


class TestLongContextNumerics(unittest.TestCase):
    """长上下文的数值契约：分块续接的误差不随块数增长。

    判据不是"误差为零"（那是同一条代码路径的恒等，5.100 已测），而是
    "1024 步、切成 4/16/32 块后，末位 logits 与整段前向的差都在 f32 的舍入
    量级，并且块数翻倍不会让误差放大"。
    """

    steps = 1024
    vocab = 8

    def _spec(self, steps):
        return LMSpec(
            vocab_size=self.vocab,
            d_model=8,
            num_layers=1,
            d_state=4,
            d_conv=3,
            expand=2,
            steps=steps,
            batch=1,
            tie_weights=True,
        )

    def _zeros(self, spec):
        return {
            name: np.zeros(shape, np.float32)
            for name, shape in spec.state_shapes.items()
        }

    def _whole(self, spec, params, ids):
        g = Graph("long_whole")
        _, logits, _ = build_language_model_stateful(g, spec)
        g.output(logits, "logits")
        feed = dict(params)
        feed["ids"] = ids
        feed.update(self._zeros(spec))
        return _run(g, feed)["logits"]

    def _chunked(self, spec, params, ids, pieces):
        chunk = spec.with_steps(spec.steps // pieces)
        g = Graph(f"long_chunk_{pieces}")
        _, logits, state_out = build_language_model_stateful(g, chunk)
        g.output(logits, "logits")
        for name, value in state_out.items():
            g.output(value, name)
        state = self._zeros(chunk)
        out = None
        for index in range(pieces):
            feed = dict(params)
            feed["ids"] = ids[:, index * chunk.steps : (index + 1) * chunk.steps]
            feed.update(state)
            out = _run(g, feed)
            state = chunk.carry_state(out)
        return out["logits"][0, -1]

    def test_chunked_error_does_not_grow_with_the_number_of_chunks(self):
        spec = self._spec(self.steps)
        params = spec.init_params(seed=7)
        ids = np.random.default_rng(3).integers(
            0, self.vocab, (1, self.steps)
        ).astype(np.int32)

        whole = self._whole(spec, params, ids)
        self.assertTrue(np.all(np.isfinite(whole)))
        last = whole[0, -1]

        errors = {}
        for pieces in (4, 16, 32):
            got = self._chunked(spec, params, ids, pieces)
            errors[pieces] = float(
                np.max(np.abs(got - last)) / (np.max(np.abs(last)) + 1e-6)
            )
            self.assertLess(errors[pieces], 1e-5, f"{pieces} 块的相对误差过大")

        self.assertLessEqual(errors[32], 3.0 * errors[16] + 1e-6)

    def test_states_stay_finite_over_a_long_scan(self):
        spec = self._spec(self.steps)
        params = spec.init_params(seed=8)
        ids = np.random.default_rng(4).integers(
            0, self.vocab, (1, self.steps)
        ).astype(np.int32)
        g = Graph("long_state")
        _, logits, state_out = build_language_model_stateful(g, spec)
        g.output(logits, "logits")
        for name, value in state_out.items():
            g.output(value, name)
        feed = dict(params)
        feed["ids"] = ids
        feed.update(self._zeros(spec))
        out = _run(g, feed)
        for name, value in out.items():
            self.assertTrue(np.all(np.isfinite(value)), f"{name} 出现非有限值")


if __name__ == "__main__":
    unittest.main()
