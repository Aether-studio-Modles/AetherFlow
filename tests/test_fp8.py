"""fp8（e4m3 / e5m2）的参考语义与单字节存储。

四条判据：

- 编码：已知位模式表、RNE 中点；e4m3 没有 inf，超出 448 饱和；e5m2 按 IEEE
  溢出到 inf；
- 解码：全部 256 个位模式都能解释，非 NaN 位模式 ``pack ∘ unpack`` 往返恒等；
- 执行：fp8 的图输入与节点输出都量化到格式格点，参与算术时每个边界都量化；
- 拒绝：未知格式、非 uint8 位模式、控制流谓词不接受 f8；设备端的计划期
  只把 ``cast`` 的 fp8 输出预绑定进批量派发快路径，其它算子仍拒绝。
"""

import unittest

import numpy as np

from aetherflow import (
    Conditional,
    Graph,
    ReferenceBackend,
    dumps,
    f8e4m3,
    f8e5m2,
    f32,
    loads,
)
from aetherflow.backends.device_plan import _PlanMixin
from aetherflow.conformance import default_cases, run_conformance
from aetherflow.runtime.fp8 import (
    FP8_KINDS,
    STORAGE_BYTES,
    pack_fp8,
    round_fp8,
    unpack_fp8,
)

BACKEND = ReferenceBackend()


def _codes(values):
    return [int(x) for x in pack_fp8(np.array(values, np.float32))]


class TestFp8Codec(unittest.TestCase):
    def test_storage_is_one_byte(self):
        self.assertEqual(STORAGE_BYTES, 1)
        self.assertEqual(FP8_KINDS, ("e4m3", "e5m2"))

    def test_e4m3_codes_decode_to_known_values(self):
        codes = np.array([0x00, 0x01, 0x08, 0x38, 0x3C, 0x40, 0x7E], np.uint8)
        got = unpack_fp8(codes, "e4m3")
        want = np.array(
            [0.0, 2.0 ** -9, 2.0 ** -6, 1.0, 1.5, 2.0, 448.0], np.float32
        )
        np.testing.assert_array_equal(got, want)
        # 0x7F 是"指数全 1 + 尾数全 1"：e4m3 没有 inf，它是唯一的 NaN。
        self.assertTrue(np.isnan(unpack_fp8(np.array([0x7F], np.uint8), "e4m3"))[0])
        # 同一码字加符号位是 -448，不是 NaN。
        self.assertEqual(unpack_fp8(np.array([0xFE], np.uint8), "e4m3")[0], np.float32(-448.0))
        self.assertTrue(np.isnan(unpack_fp8(np.array([0xFF], np.uint8), "e4m3"))[0])

    def test_e5m2_codes_decode_to_known_values(self):
        codes = np.array([0x00, 0x01, 0x04, 0x3C, 0x7B, 0x7C], np.uint8)
        got = unpack_fp8(codes, "e5m2")
        want = np.array(
            [0.0, 2.0 ** -16, 2.0 ** -14, 1.0, 57344.0, np.inf], np.float32
        )
        np.testing.assert_array_equal(got, want)
        self.assertTrue(np.isnan(unpack_fp8(np.array([0x7D], np.uint8), "e5m2")[0]))
        self.assertTrue(np.isinf(unpack_fp8(np.array([0xFC], np.uint8), "e5m2")[0]))
        self.assertLess(unpack_fp8(np.array([0xFC], np.uint8), "e5m2")[0], 0)

    def test_pack_produces_known_codes(self):
        values = [0.0, -0.0, 1.0, 1.5, 2.0, -1.0, 448.0, 1000.0, np.nan, np.inf]
        self.assertEqual(_codes(values), [0x00, 0x80, 0x38, 0x3C, 0x40, 0xB8, 0x7E, 0x7E, 0x7F, 0x7E])
        values = [0.0, -0.0, 1.0, 57344.0, 100000.0, np.nan, np.inf]
        self.assertEqual(
            [int(x) for x in pack_fp8(np.array(values, np.float32), "e5m2")],
            [0x00, 0x80, 0x3C, 0x7B, 0x7C, 0x7F, 0x7C],
        )

    def test_rounding_is_nearest_even(self):
        # e4m3 在 [1, 2) 的 ulp 是 1/8：1.0625 恰好是 1.0 与 1.125 的中点，
        # 进位到尾数偶数（1.0）；1.1875 是 1.125 与 1.25 的中点，进位到 1.25。
        got = round_fp8(np.array([1.0625, 1.1875, 1.3], np.float32), "e4m3")
        np.testing.assert_array_equal(got, np.array([1.0, 1.25, 1.25], np.float32))
        # e5m2 在 [1, 2) 的 ulp 是 1/4：1.125 是中点，进位到 1.0。
        got = round_fp8(np.array([1.0625, 1.1875, 1.125], np.float32), "e5m2")
        np.testing.assert_array_equal(got, np.array([1.0, 1.25, 1.0], np.float32))

    def test_e4m3_saturates_where_e5m2_overflows(self):
        x = np.array([1e30, -1e30, np.inf, -np.inf], np.float32)
        np.testing.assert_array_equal(
            round_fp8(x, "e4m3"), np.array([448.0, -448.0, 448.0, -448.0], np.float32)
        )
        got = round_fp8(np.array([100000.0, -100000.0], np.float32), "e5m2")
        self.assertTrue(np.isinf(got[0]) and got[0] > 0)
        self.assertTrue(np.isinf(got[1]) and got[1] < 0)
        # 57344 与 inf 的中点是 61440：按 RNE 走到 inf（无穷的尾数为 0，算偶数）。
        self.assertTrue(np.isinf(round_fp8(np.array([61440.0], np.float32), "e5m2")[0]))
        self.assertEqual(round_fp8(np.array([60000.0], np.float32), "e5m2")[0], np.float32(57344.0))

    def test_subnormal_range(self):
        # 最小可表示值：e4m3 是 2^-9、e5m2 是 2^-16；再小的一半按 RNE 归零。
        got = round_fp8(np.array([2.0 ** -10, 1.5 * 2.0 ** -10], np.float32), "e4m3")
        np.testing.assert_array_equal(got, np.array([0.0, 2.0 ** -9], np.float32))
        self.assertEqual(round_fp8(np.array([2.0 ** -16], np.float32), "e5m2")[0], np.float32(2.0 ** -16))

    def test_signed_zero_and_nan_survive(self):
        got = round_fp8(np.array([0.0, -0.0, np.nan], np.float32), "e4m3")
        self.assertEqual(got[0], 0.0)
        self.assertTrue(np.signbit(got[1]))  # -0 的符号位保留
        self.assertTrue(np.isnan(got[2]))

    def test_non_nan_codes_roundtrip(self):
        for kind in FP8_KINDS:
            codes = np.arange(256, dtype=np.uint8)
            decoded = unpack_fp8(codes, kind)
            alive = ~np.isnan(decoded)
            again = pack_fp8(decoded[alive], kind)
            np.testing.assert_array_equal(again, codes[alive])

    def test_rejects_unknown_kind_and_non_byte_bits(self):
        with self.assertRaises(ValueError):
            pack_fp8(np.array([1.0], np.float32), "e3m4")
        with self.assertRaises(ValueError):
            unpack_fp8(np.array([0], np.int32), "e4m4")
        # 位模式必须是 uint8：给 int32 数组说明调用方把它当整数数值了。
        with self.assertRaises(TypeError):
            unpack_fp8(np.array([1, 2], np.int32), "e4m3")


class TestFp8Executor(unittest.TestCase):
    def test_cast_quantizes_the_node_output(self):
        g = Graph("cast")
        x = g.input("x", f32[(4,)])
        g.output(g.cast(x, dtype="f8e4m3"), "y")
        feed = {"x": np.array([0.5, 1.0625, 1000.0, -0.0], np.float32)}
        got = BACKEND.execute(g, feed)["y"]
        np.testing.assert_array_equal(
            got, np.array([0.5, 1.0, 448.0, -0.0], np.float32)
        )
        self.assertEqual(got.dtype, np.float32)  # 载体仍是 f32

    def test_fp8_graph_input_is_quantized_on_entry(self):
        g = Graph("cast")
        x = g.input("x", f8e5m2[(3,)])
        g.output(g.cast(x, dtype="f32"), "y")
        got = BACKEND.execute(g, {"x": np.array([0.1, 70000.0, -0.0], np.float32)})["y"]
        self.assertEqual(got[0], round_fp8(np.array([0.1], np.float32), "e5m2")[0])
        self.assertTrue(np.isinf(got[1]))
        self.assertTrue(np.signbit(got[2]))

    def test_each_arithmetic_boundary_is_quantized(self):
        # 1.0 + 0.0625 在 f32 里是 1.0625，但 f8e4m3 的出口必须按 RNE 回到 1.0。
        g = Graph("add")
        a = g.input("a", f8e4m3[()])
        b = g.input("b", f8e4m3[()])
        g.output(g.add(a, b), "z")
        got = BACKEND.execute(g, {"a": np.float32(1.0), "b": np.float32(0.0625)})["z"]
        self.assertEqual(got, np.float32(1.0))

    def test_graph_survives_json_roundtrip(self):
        g = Graph("cast")
        x = g.input("x", f32[(2, 2)])
        g.output(g.cast(x, dtype="f8e4m3"), "y")
        feed = {"x": np.array([[0.1, 1.3], [-1000.0, 2.5]], np.float32)}
        want = BACKEND.execute(g, feed)["y"]
        again = BACKEND.execute(loads(dumps(g)), feed)["y"]
        np.testing.assert_array_equal(want, again)

    def test_control_flow_predicate_rejects_fp8(self):
        pred = Graph("pred")
        x = pred.input("x", f8e4m3[()])
        pred.output(x, "flag")
        then = Graph("then")
        y = then.input("x", f32[()])
        then.output(then.mul(y, then.constant(1.0)), "y")
        otherwise = Graph("otherwise")
        z = otherwise.input("x", f32[()])
        otherwise.output(otherwise.mul(z, otherwise.constant(-1.0)), "y")
        # 谓词类型白名单不含 f8：控制流按数值选支，而 f8 目前只有参考语义，
        # 没有可比较的确定性存储。
        with self.assertRaises(TypeError):
            Conditional(pred, then, otherwise)


class _PlanStub(_PlanMixin):
    """只够跑 ``_planned_kernel`` 的替身：不建任何设备上下文。

    真实后端的 ``kernels`` 是注册表，这里只需要 ``get`` 返回一个非 ``None``
    的对象，代表"算子已注册"。
    """

    kernels = {"cast": object(), "add": object()}
    _f16_ops = frozenset()
    _bf16_ops = frozenset()


class TestFp8DevicePlan(unittest.TestCase):
    """计划期的 fp8 放行面：只有 ``cast``（DESIGN 5.325）。

    ``cast`` 的输入是 f32 载体、fp8 只出现在输出上，快路径末尾按
    ``out_dtype`` 调 ``quantize`` 落回格点，因此可以预绑定。其它算子
    （例如 ``add(f8, f8)``）仍必须留给 ``invoke`` 报 refused——预绑定会让
    批量派发快路径绕过唯一的 fp8 检查，把 f8 当成 f32 算掉。
    """

    def _cast_node(self, dtype):
        g = Graph("cast")
        x = g.input("x", f32[(2,)])
        g.output(g.cast(x, dtype=dtype), "y")
        return g.nodes[-1]

    def _add_node(self, dtype):
        g = Graph("add")
        a = g.input("a", dtype[(2,)])
        b = g.input("b", dtype[(2,)])
        g.output(g.add(a, b), "y")
        return g.nodes[-1]

    def test_fp8_cast_output_is_prebound(self):
        stub = _PlanStub()
        self.assertIsNotNone(stub._planned_kernel(self._cast_node("f8e4m3")))
        self.assertIsNotNone(stub._planned_kernel(self._cast_node("f8e5m2")))

    def test_non_cast_fp8_node_is_not_prebound(self):
        stub = _PlanStub()
        self.assertIsNone(stub._planned_kernel(self._add_node(f8e4m3)))
        self.assertIsNone(stub._planned_kernel(self._add_node(f8e5m2)))

    def test_plain_cast_is_still_prebound(self):
        # f32 的 cast 仍然走预绑定快路径。
        stub = _PlanStub()
        self.assertIsNotNone(stub._planned_kernel(self._cast_node("f32")))


class TestFp8Conformance(unittest.TestCase):
    def test_reference_backend_passes_fp8_cases(self):
        names = {"cast_f8e4m3", "cast_f8e5m2"}
        cases = [c for c in default_cases() if c.name in names]
        self.assertEqual({c.name for c in cases}, names)
        report = run_conformance(BACKEND, cases=cases)
        self.assertTrue(report.ok, report.text())


if __name__ == "__main__":
    unittest.main()
