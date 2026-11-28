"""safetensors 读取与装载的验收。

三条判据：

1. **字节级往返**：写进去什么 dtype / shape / 值，读回来必须逐位相同（bf16 是
   唯一的例外——numpy 没有这个类型，约定无损展宽成 f32）。
2. **坏文件在拿到内存之前就被拒**：头长、offsets、shape 与字节数对不上的组合
   逐条断言。权重文件是外部输入，解析器的默认假设必须是"它是坏的"。
3. **装填要么全成、要么全不动**：一个参数对不上时，别的参数不能被悄悄写进去。
   半新半旧的参数不报错、训练也能跑，但结果永远是错的。
"""

import json
import os
import struct
import tempfile
import unittest

import numpy as np

from aetherflow.models.quantize import (
    dequantize_weight,
    dequantize_weight_asymmetric,
    quantize_weight,
    quantize_weight_asymmetric,
)
from aetherflow.train.params import ParameterStore
from aetherflow.weights import (
    QuantizedWeight,
    SafeTensorsError,
    SafeTensorsFile,
    build_quantized_weight,
    load_into_store,
    load_quantized_linear,
    load_safetensors,
    load_safetensors_dir,
)

_NUMPY_TO_ST = {
    np.dtype(np.float32): "F32",
    np.dtype(np.float16): "F16",
    np.dtype(np.float64): "F64",
    np.dtype(np.int8): "I8",
    np.dtype(np.int16): "I16",
    np.dtype(np.int32): "I32",
    np.dtype(np.uint8): "U8",
    np.dtype(np.uint16): "U16",
    np.dtype(np.bool_): "BOOL",
}


def _raw_file(path, header, data=b""):
    """按给定头与数据区写文件，头由调用方负责（包括写错的那些）。"""
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(data)


def _file(path, tensors, metadata=None):
    """把 ``{名字: ndarray}`` 正常打包成一个 safetensors 文件。"""
    header = {}
    chunks = []
    offset = 0
    for name, arr in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        header[name] = {
            "dtype": _NUMPY_TO_ST[arr.dtype],
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        chunks.append(raw)
        offset += len(raw)
    if metadata is not None:
        header["__metadata__"] = dict(metadata)
    _raw_file(path, header, b"".join(chunks))


class _TempDir(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name

    def path(self, name):
        return os.path.join(self.dir, name)


class TestRead(_TempDir):
    def test_round_trip_supported_dtypes(self):
        cases = {
            "a_f32": np.arange(6, dtype=np.float32).reshape(2, 3),
            "b_f16": np.array([0.5, -1.5], dtype=np.float16),
            "c_f64": np.array([1.25], dtype=np.float64),
            "d_i8": np.array([-128, 0, 127], dtype=np.int8),
            "e_i32": np.array([[1, -2], [3, -4]], dtype=np.int32),
            "f_bool": np.array([True, False, True]),
            "g_scalar": np.array(3.5, dtype=np.float32),
        }
        target = self.path("weights.safetensors")
        _file(target, cases)
        got = load_safetensors(target)
        self.assertEqual(set(got), set(cases))
        for name, want in cases.items():
            self.assertEqual(got[name].dtype, want.dtype, name)
            self.assertEqual(got[name].shape, want.shape, name)
            np.testing.assert_array_equal(got[name], want, err_msg=name)

    def test_bf16_is_widened_to_float32(self):
        # 0x3F80 = 1.0、0xBF80 = -1.0、0x0000 = 0.0、0x7FC0 = NaN（bf16 位模式）
        bits = np.array([0x3F80, 0xBF80, 0x0000, 0x7FC0], dtype="<u2")
        target = self.path("bf16.safetensors")
        _raw_file(
            target,
            {
                "w": {
                    "dtype": "BF16",
                    "shape": [4],
                    "data_offsets": [0, bits.nbytes],
                }
            },
            bits.tobytes(),
        )
        out = load_safetensors(target)["w"]
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out[0], 1.0)
        self.assertEqual(out[1], -1.0)
        self.assertEqual(out[2], 0.0)
        self.assertTrue(np.isnan(out[3]))

    def test_metadata_and_names(self):
        target = self.path("meta.safetensors")
        _file(target, {"x": np.zeros((2,), np.float32)}, metadata={"format": "pt"})
        with SafeTensorsFile(target) as fh:
            self.assertEqual(fh.metadata, {"format": "pt"})
            self.assertEqual(fh.names, ("x",))
            self.assertIn("x", fh)
            self.assertEqual(len(fh), 1)
            info = fh.info("x")
            self.assertEqual((info.dtype, info.shape, info.nbytes), ("F32", (2,), 8))

    def test_tensor_is_writable_copy(self):
        target = self.path("copy.safetensors")
        _file(target, {"x": np.array([1.0, 2.0], np.float32)})
        with SafeTensorsFile(target) as fh:
            first = fh.tensor("x")
            first[0] = 99.0
            second = fh.tensor("x")
        self.assertEqual(second[0], 1.0)

    def test_missing_tensor_key(self):
        target = self.path("missing.safetensors")
        _file(target, {"x": np.zeros((1,), np.float32)})
        with SafeTensorsFile(target) as fh:
            with self.assertRaises(KeyError):
                fh.tensor("nope")

    def test_fp8_is_rejected_explicitly(self):
        target = self.path("fp8.safetensors")
        _raw_file(
            target,
            {"w": {"dtype": "F8_E4M3", "shape": [2], "data_offsets": [0, 2]}},
            b"\x00\x01",
        )
        with SafeTensorsFile(target) as fh:
            with self.assertRaises(SafeTensorsError) as ctx:
                fh.tensor("w")
        self.assertIn("fp8", str(ctx.exception))


class TestHeaderRejections(_TempDir):
    def _reject(self, header, data=b"", raw=None):
        target = self.path("bad.safetensors")
        if raw is None:
            _raw_file(target, header, data)
        else:
            with open(target, "wb") as fh:
                fh.write(raw)
        with self.assertRaises(SafeTensorsError):
            with SafeTensorsFile(target):
                pass

    def test_short_file(self):
        target = self.path("short.safetensors")
        with open(target, "wb") as fh:
            fh.write(b"\x01\x02\x03")
        with self.assertRaises(SafeTensorsError):
            with SafeTensorsFile(target):
                pass

    def test_header_length_over_limit(self):
        self._reject(None, raw=struct.pack("<Q", 1 << 62))

    def test_header_past_end_of_file(self):
        raw = struct.pack("<Q", 100) + b"{}"
        self._reject(None, raw=raw)

    def test_header_not_json(self):
        self._reject(None, raw=struct.pack("<Q", 3) + b"{{}")

    def test_header_not_object(self):
        self._reject(None, raw=struct.pack("<Q", 5) + b"[1,2]")

    def test_metadata_must_be_string_map(self):
        self._reject({"__metadata__": {"k": 1}})

    def test_unknown_dtype(self):
        self._reject({"w": {"dtype": "F4", "shape": [1], "data_offsets": [0, 1]}})

    def test_shape_must_be_list(self):
        self._reject({"w": {"dtype": "F32", "shape": 2, "data_offsets": [0, 8]}}, b"\x00" * 8)

    def test_negative_dim(self):
        self._reject({"w": {"dtype": "F32", "shape": [-1], "data_offsets": [0, 4]}}, b"\x00" * 4)

    def test_offsets_not_a_pair(self):
        self._reject({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0]}}, b"\x00" * 4)

    def test_offsets_reversed(self):
        self._reject({"w": {"dtype": "F32", "shape": [1], "data_offsets": [4, 0]}}, b"\x00" * 4)

    def test_offsets_past_data(self):
        self._reject({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, b"\x00" * 2)

    def test_bytes_do_not_match_shape(self):
        # 2 个 f32 应为 8 字节，头里只给了 4 字节
        self._reject({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}, b"\x00" * 8)

    def test_tensor_spec_must_be_object(self):
        self._reject({"w": [1, 2]})

    def test_empty_tensor_name(self):
        self._reject({"": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]}})


class TestDirectory(_TempDir):
    def test_shards_are_merged(self):
        _file(self.path("model-00001-of-00002.safetensors"), {"a": np.ones((2,), np.float32)})
        _file(self.path("model-00002-of-00002.safetensors"), {"b": np.zeros((3,), np.float16)})
        merged = load_safetensors_dir(self.dir)
        self.assertEqual(set(merged), {"a", "b"})
        np.testing.assert_array_equal(merged["a"], np.ones((2,), np.float32))

    def test_duplicate_tensor_across_shards(self):
        _file(self.path("a.safetensors"), {"w": np.ones((2,), np.float32)})
        _file(self.path("b.safetensors"), {"w": np.zeros((2,), np.float32)})
        with self.assertRaises(SafeTensorsError) as ctx:
            load_safetensors_dir(self.dir)
        message = str(ctx.exception)
        self.assertIn("a.safetensors", message)
        self.assertIn("b.safetensors", message)

    def test_empty_directory(self):
        with self.assertRaises(SafeTensorsError):
            load_safetensors_dir(self.dir)

    def test_not_a_directory(self):
        target = self.path("w.safetensors")
        _file(target, {"x": np.zeros((1,), np.float32)})
        with self.assertRaises(SafeTensorsError):
            load_safetensors_dir(target)


class TestLoadIntoStore(unittest.TestCase):
    def store(self):
        return ParameterStore(
            {
                "w": np.zeros((3, 4), np.float32),
                "b": np.ones((4,), np.float32),
            }
        )

    def test_same_name(self):
        store = self.store()
        tensors = {"w": np.arange(12, dtype=np.float32).reshape(3, 4), "b": np.full((4,), 2.0, np.float32)}
        unused = load_into_store(store, tensors)
        self.assertEqual(unused, ())
        np.testing.assert_array_equal(store["w"], tensors["w"])
        np.testing.assert_array_equal(store["b"], tensors["b"])

    def test_mapping_renames(self):
        store = self.store()
        tensors = {
            "model.w": np.ones((3, 4), np.float32),
            "model.b": np.zeros((4,), np.float32),
        }
        unused = load_into_store(store, tensors, mapping={"w": "model.w", "b": "model.b"})
        self.assertEqual(unused, ())
        np.testing.assert_array_equal(store["w"], np.ones((3, 4), np.float32))

    def test_transpose_torch_linear_layout(self):
        store = ParameterStore({"w": np.zeros((4, 3), np.float32)})
        torch_layout = np.arange(12, dtype=np.float32).reshape(3, 4)
        load_into_store(store, {"w": torch_layout}, transpose=["w"])
        np.testing.assert_array_equal(store["w"], torch_layout.T)

    def test_transpose_requires_2d(self):
        store = ParameterStore({"b": np.zeros((4,), np.float32)})
        with self.assertRaises(SafeTensorsError):
            load_into_store(store, {"b": np.zeros((4,), np.float32)}, transpose=["b"])

    def test_reports_every_problem_at_once(self):
        store = self.store()
        with self.assertRaises(SafeTensorsError) as ctx:
            load_into_store(
                store,
                {"w": np.zeros((3, 5), np.float32), "b": np.zeros((5,), np.float32)},
            )
        message = str(ctx.exception)
        self.assertIn("'w'", message)
        self.assertIn("'b'", message)

    def test_failure_leaves_store_untouched(self):
        store = self.store()
        before = store["b"].copy()
        with self.assertRaises(SafeTensorsError):
            load_into_store(
                store,
                {"w": np.ones((3, 4), np.float32), "b": np.zeros((9,), np.float32)},
            )
        # w 的形状是对的、b 是错的：整体失败时 w 也不能被写进去
        np.testing.assert_array_equal(store["w"], np.zeros((3, 4), np.float32))
        np.testing.assert_array_equal(store["b"], before)

    def test_write_is_in_place(self):
        store = self.store()
        identity = store.arrays()["w"]
        load_into_store(store, {"w": np.ones((3, 4), np.float32), "b": np.ones((4,), np.float32)})
        self.assertIs(store.arrays()["w"], identity)

    def test_f16_source_widens_without_flag(self):
        store = self.store()
        load_into_store(
            store,
            {
                "w": np.ones((3, 4), np.float16),
                "b": np.ones((4,), np.float16),
            },
        )
        self.assertEqual(store["w"].dtype, np.float32)

    def test_downcast_requires_explicit_flag(self):
        store = self.store()
        tensors = {"w": np.ones((3, 4), np.float64), "b": np.ones((4,), np.float64)}
        with self.assertRaises(SafeTensorsError) as ctx:
            load_into_store(store, tensors)
        self.assertIn("allow_downcast", str(ctx.exception))
        load_into_store(store, tensors, allow_downcast=True)
        np.testing.assert_array_equal(store["w"], np.ones((3, 4), np.float32))

    def test_strict_missing_tensor(self):
        with self.assertRaises(SafeTensorsError) as ctx:
            load_into_store(self.store(), {"w": np.zeros((3, 4), np.float32)})
        self.assertIn("'b'", str(ctx.exception))

    def test_unused_tensors_are_reported(self):
        unused = load_into_store(
            self.store(),
            {
                "w": np.zeros((3, 4), np.float32),
                "b": np.zeros((4,), np.float32),
                "lm_head.weight": np.zeros((8, 3), np.float32),
            },
        )
        self.assertEqual(unused, ("lm_head.weight",))

    def test_unknown_mapping_or_transpose_key(self):
        with self.assertRaises(SafeTensorsError):
            load_into_store(self.store(), {}, mapping={"nope": "x"})
        with self.assertRaises(SafeTensorsError):
            load_into_store(self.store(), {}, transpose=["nope"])

    def test_rejects_non_store(self):
        with self.assertRaises(TypeError):
            load_into_store({"w": np.zeros((3, 4), np.float32)}, {})


class TestQuantizedWeight(_TempDir):
    """量化权重的读取入口：张量校验 + metadata 自描述一致性。

    判据与模块说明一致：数值上必须与 ``models.quantize`` 的同一份语义逐位
    相等（``dequantize`` 走的是 IR 的 reference，不是另写一份）；打包错误
    （非 int8 权重、零点越界、形状对不上、metadata 与张量矛盾）必须在
    拿到 ``QuantizedWeight`` 之前报错，不能靠"看着像"去猜。
    """

    W = np.array(
        [[-1.5, 0.25, 2.0], [0.5, -0.75, 0.0], [1.25, 0.125, -2.5]],
        dtype=np.float32,
    )

    def _symmetric_file(self, metadata=None):
        q, scale = quantize_weight(self.W)
        target = self.path("q8.safetensors")
        _file(target, {"w_q": q, "w_scale": scale}, metadata=metadata)
        return target, q, scale

    def _asymmetric_file(self, metadata=None, zp_dtype=np.int32):
        q, scale, zp = quantize_weight_asymmetric(self.W)
        target = self.path("q8zp.safetensors")
        _file(
            target,
            {"w_q": q, "w_scale": scale, "w_zp": zp.astype(zp_dtype)},
            metadata=metadata,
        )
        return target, q, scale, zp

    def test_symmetric_round_trip(self):
        target, q, scale = self._symmetric_file({"quantization": "int8"})
        got = load_quantized_linear(target, "w_q", "w_scale")
        self.assertIsInstance(got, QuantizedWeight)
        self.assertTrue(got.symmetric)
        self.assertEqual(got.shape, (3, 3))
        self.assertEqual((got.in_features, got.out_features), (3, 3))
        np.testing.assert_array_equal(got.q, q)
        np.testing.assert_array_equal(got.scale, scale)
        np.testing.assert_array_equal(got.dequantize(), dequantize_weight(q, scale))

    def test_asymmetric_round_trip_widens_i32_zero_point(self):
        target, q, scale, zp = self._asymmetric_file({"quantization": "int8_zp"})
        got = load_quantized_linear(target, "w_q", "w_scale", "w_zp")
        self.assertFalse(got.symmetric)
        self.assertEqual(got.zero_point.dtype, np.int8)
        np.testing.assert_array_equal(got.zero_point, zp)
        np.testing.assert_array_equal(
            got.dequantize(), dequantize_weight_asymmetric(q, scale, zp)
        )

    def test_transpose_reads_torch_layout(self):
        w = np.arange(12, dtype=np.float32).reshape(3, 4) - 5.0
        q, scale = quantize_weight(w)  # q 是 (3, 4)，scale 是 (4,)
        target = self.path("torch.safetensors")
        _file(target, {"w": q.T, "scale": scale})
        got = load_quantized_linear(target, "w", "scale", transpose=True)
        self.assertEqual(got.shape, (3, 4))
        np.testing.assert_array_equal(got.q, q)

    def test_metadata_absent_is_accepted(self):
        target, q, scale = self._symmetric_file()
        got = load_quantized_linear(target, "w_q", "w_scale")
        np.testing.assert_array_equal(got.dequantize(), dequantize_weight(q, scale))

    def test_group_size_matching_shape_is_accepted(self):
        target, _, _ = self._symmetric_file(
            {"quantization": "int8", "quantization.group_size": "3"}
        )
        self.assertTrue(load_quantized_linear(target, "w_q", "w_scale").symmetric)

    def test_group_size_must_match_shape(self):
        # W 是 3x3，逐通道 scale 意味着每组 3 行；声明 1 与形状矛盾。
        target, _, _ = self._symmetric_file({"quantization.group_size": "1"})
        with self.assertRaises(SafeTensorsError) as ctx:
            load_quantized_linear(target, "w_q", "w_scale")
        self.assertIn("group_size", str(ctx.exception))

    def test_per_group_round_trip(self):
        q = np.array(
            [[1, -2], [3, -4], [5, -6], [7, -8]], dtype=np.int8
        )  # (K, N) = (4, 2)，每组 2 行
        scale = np.array([[0.5, 0.25], [1.5, 2.0]], dtype=np.float32)  # (2, 2)
        zp = np.array([[0, 1], [-2, 3]], dtype=np.int8)
        target = self.path("group.safetensors")
        _file(
            target,
            {"q": q, "s": scale, "z": zp},
            metadata={"quantization": "int8_zp", "quantization.group_size": "2"},
        )
        got = load_quantized_linear(target, "q", "s", "z")
        self.assertFalse(got.symmetric)
        self.assertEqual(got.group_size, 2)
        self.assertTrue(got.per_group)
        want = (q.astype(np.int32) - np.repeat(zp.astype(np.int32), 2, axis=0)).astype(
            np.float32
        ) * np.repeat(scale, 2, axis=0)
        np.testing.assert_array_equal(got.dequantize(), want)

    def test_metadata_kind_must_match_zero_point(self):
        target, _, _ = self._symmetric_file({"quantization": "int8_zp"})
        with self.assertRaises(SafeTensorsError):
            load_quantized_linear(target, "w_q", "w_scale")
        target, _, _, _ = self._asymmetric_file({"quantization": "int8"})
        with self.assertRaises(SafeTensorsError):
            load_quantized_linear(target, "w_q", "w_scale", "w_zp")

    def test_unknown_quantization_kind(self):
        target, _, _ = self._symmetric_file({"quantization": "fp8"})
        with self.assertRaises(SafeTensorsError):
            load_quantized_linear(target, "w_q", "w_scale")

    def test_group_size_must_be_positive_integer(self):
        target, _, _ = self._symmetric_file({"quantization.group_size": "batch"})
        with self.assertRaises(SafeTensorsError):
            load_quantized_linear(target, "w_q", "w_scale")
        target, _, _ = self._symmetric_file({"quantization.group_size": "0"})
        with self.assertRaises(SafeTensorsError):
            load_quantized_linear(target, "w_q", "w_scale")

    def test_check_metadata_false_only_skips_self_description(self):
        target, q, scale = self._symmetric_file({"quantization": "int8_zp"})
        got = load_quantized_linear(
            target, "w_q", "w_scale", check_metadata=False
        )
        np.testing.assert_array_equal(got.dequantize(), dequantize_weight(q, scale))

    def test_build_from_tensor_map(self):
        q, scale = quantize_weight(self.W)
        got = build_quantized_weight({"w_q": q, "w_scale": scale}, "w_q", "w_scale")
        self.assertTrue(got.symmetric)
        np.testing.assert_array_equal(got.dequantize(), dequantize_weight(q, scale))

    def test_rejects_non_int8_q(self):
        target = self.path("f32q.safetensors")
        _file(target, {"q": np.ones((2, 3), np.float32), "s": np.ones((3,), np.float32)})
        with self.assertRaises(SafeTensorsError) as ctx:
            load_quantized_linear(target, "q", "s")
        self.assertIn("int8", str(ctx.exception))

    def test_rejects_zero_point_out_of_range(self):
        target = self.path("zp.safetensors")
        _file(
            target,
            {
                "q": np.zeros((2, 3), np.int8),
                "s": np.ones((3,), np.float32),
                "z": np.array([0, 128, -129], np.int32),
            },
        )
        with self.assertRaises(SafeTensorsError):
            load_quantized_linear(target, "q", "s", "z")

    def test_rejects_float_zero_point(self):
        target = self.path("zpf.safetensors")
        _file(
            target,
            {
                "q": np.zeros((2, 3), np.int8),
                "s": np.ones((3,), np.float32),
                "z": np.zeros((3,), np.float32),
            },
        )
        with self.assertRaises(SafeTensorsError):
            load_quantized_linear(target, "q", "s", "z")

    def test_rejects_scale_length_mismatch(self):
        target = self.path("slen.safetensors")
        _file(target, {"q": np.zeros((2, 3), np.int8), "s": np.ones((2,), np.float32)})
        with self.assertRaises(SafeTensorsError):
            load_quantized_linear(target, "q", "s")

    def test_rejects_non_positive_scale(self):
        target = self.path("sneg.safetensors")
        _file(
            target,
            {
                "q": np.zeros((2, 3), np.int8),
                "s": np.array([0.5, -0.5, 1.0], np.float32),
            },
        )
        with self.assertRaises(SafeTensorsError):
            load_quantized_linear(target, "q", "s")

    def test_scale_f64_requires_allow_downcast(self):
        target = self.path("s64.safetensors")
        _file(target, {"q": np.zeros((2, 3), np.int8), "s": np.ones((3,), np.float64)})
        with self.assertRaises(SafeTensorsError):
            load_quantized_linear(target, "q", "s")
        got = load_quantized_linear(target, "q", "s", allow_downcast=True)
        self.assertEqual(got.scale.dtype, np.float32)

    def test_missing_tensor(self):
        target = self.path("miss.safetensors")
        _file(target, {"q": np.zeros((2, 3), np.int8)})
        with self.assertRaises(SafeTensorsError) as ctx:
            load_quantized_linear(target, "q", "s")
        self.assertIn("'s'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
