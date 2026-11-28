"""GGUF 容器与量化块解码的用例。

这里的文件不是拿 llama.cpp 生成的，而是按 GGUF 的字节布局手工拼出来的：
容器部分独立于实现写一遍，块部分用已知的 ``d`` / 缩放算出期望值。这样测的是
"解出来的每个数对不对"，而不是"读了同一个文件两遍结果一样"。
"""

from __future__ import annotations

import struct
import unittest

import numpy as np

from aetherflow.gguf import GGML_TYPES, GGUFError, read_gguf

MAGIC = 0x46554747


def _pack_str(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


_FMTS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<B",
    10: "<Q",
    11: "<q",
    12: "<d",
}


def _pack_value(vtype: int, value) -> bytes:
    if vtype == 8:
        return _pack_str(value)
    if vtype == 9:
        element_type, items = value
        out = struct.pack("<IQ", element_type, len(items))
        for item in items:
            out += _pack_value(element_type, item)
        return out
    return struct.pack(_FMTS[vtype], value)


def _kv(key: str, vtype: int, value) -> bytes:
    return _pack_str(key) + struct.pack("<I", vtype) + _pack_value(vtype, value)


def build_gguf(
    tensors,
    metadata=(),
    alignment: int = 32,
    version: int = 3,
    align_tensors: bool = True,
) -> bytes:
    """``tensors`` 是 ``(名字, dims, type_id, 数据字节)`` 的列表，按 ggml 维序。"""
    infos = b""
    chunks = []
    offset = 0
    for name, dims, type_id, data in tensors:
        infos += _pack_str(name) + struct.pack("<I", len(dims))
        for dim in dims:
            infos += struct.pack("<Q", dim)
        infos += struct.pack("<IQ", type_id, offset)
        chunk = data
        if align_tensors:
            chunk += b"\x00" * ((-len(chunk)) % alignment)
        chunks.append(chunk)
        offset += len(chunk)
    prefix = struct.pack("<IIQQ", MAGIC, version, len(tensors), len(metadata))
    prefix += b"".join(metadata) + infos
    prefix += b"\x00" * ((-len(prefix)) % alignment)
    return prefix + b"".join(chunks)


def _type_id(name: str) -> int:
    return GGML_TYPES[name][0]


def _tensor(name, dims, type_name, data):
    return (name, dims, _type_id(type_name), data)


def _read_one(tensors, metadata=(), **kwargs):
    return read_gguf(build_gguf(list(tensors), metadata, **kwargs))


class TestContainer(unittest.TestCase):
    def test_header_and_metadata(self):
        metadata = [
            _kv("general.architecture", 8, "llama"),
            _kv("llama.block_count", 4, 32),
            _kv("llama.rope.freq_base", 6, 10000.0),
            _kv("llama.use_parallel_residual", 7, False),
            _kv("llama.some_count", 11, -7),
            _kv("tokenizer.ggml.tokens", 9, (8, ["<unk>", "<s>", "he"])),
        ]
        data = np.arange(6, dtype="<f4").tobytes()
        with _read_one([_tensor("blk.0", (2, 3), "F32", data)], metadata) as f:
            self.assertEqual(f.version, 3)
            self.assertEqual(f.metadata["general.architecture"], "llama")
            self.assertEqual(f.metadata["llama.block_count"], 32)
            self.assertAlmostEqual(f.metadata["llama.rope.freq_base"], 10000.0)
            self.assertIs(f.metadata["llama.use_parallel_residual"], False)
            self.assertEqual(f.metadata["llama.some_count"], -7)
            self.assertEqual(f.metadata["tokenizer.ggml.tokens"], ["<unk>", "<s>", "he"])
            self.assertEqual(f.alignment, 32)
            self.assertEqual(len(f), 1)
            self.assertIn("blk.0", f)

    def test_dims_are_reversed_for_numpy(self):
        data = np.arange(6, dtype="<f4").tobytes()
        with _read_one([_tensor("w", (2, 3), "F32", data)]) as f:
            info = f.tensor_info("w")
            self.assertEqual(info.dims, (2, 3))
            self.assertEqual(info.shape, (3, 2))
            array = f.tensor("w")
            self.assertEqual(array.shape, (3, 2))
            np.testing.assert_array_equal(array, np.arange(6, dtype=np.float32).reshape(3, 2))

    def test_alignment_is_honoured(self):
        metadata = [_kv("general.alignment", 4, 64)]
        data = np.zeros(8, dtype="<f4").tobytes()
        with _read_one([_tensor("w", (8,), "F32", data)], metadata, alignment=64) as f:
            self.assertEqual(f.alignment, 64)
            self.assertEqual(f.data_offset % 64, 0)
            self.assertEqual(f.tensor("w").shape, (8,))

    def test_version_2_is_accepted(self):
        data = np.ones(2, dtype="<f4").tobytes()
        with _read_one([_tensor("w", (2,), "F32", data)], version=2) as f:
            self.assertEqual(f.version, 2)
            np.testing.assert_array_equal(f.tensor("w"), [1.0, 1.0])

    def test_path_round_trip_and_close(self):
        import tempfile
        from pathlib import Path

        data = np.array([1.5, -2.5], dtype="<f4").tobytes()
        blob = build_gguf([_tensor("w", (2,), "F32", data)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.gguf"
            path.write_bytes(blob)
            with read_gguf(path) as f:
                np.testing.assert_array_equal(f.tensor("w"), [1.5, -2.5])
                self.assertEqual(f.path, str(path))
            with self.assertRaises(GGUFError):
                f.tensor("w")

    def test_magic_mismatch(self):
        blob = bytearray(build_gguf([_tensor("w", (2,), "F32", b"\x00" * 8)]))
        blob[0:4] = b"GGML"
        with self.assertRaisesRegex(GGUFError, "不是 GGUF"):
            read_gguf(bytes(blob))

    def test_version_1_is_rejected(self):
        blob = build_gguf([_tensor("w", (2,), "F32", b"\x00" * 8)], version=1)
        with self.assertRaisesRegex(GGUFError, "版本 1"):
            read_gguf(blob)

    def test_truncated_header(self):
        with self.assertRaisesRegex(GGUFError, "越过文件末尾"):
            read_gguf(build_gguf([], [])[:16])

    def test_truncated_tensor_data(self):
        blob = build_gguf([_tensor("w", (8,), "F32", np.zeros(8, dtype="<f4").tobytes())])
        with self.assertRaisesRegex(GGUFError, "越过文件末尾"):
            read_gguf(blob[:-8])

    def test_tensor_count_lies(self):
        blob = bytearray(build_gguf([_tensor("w", (2,), "F32", b"\x00" * 8)]))
        blob[8:16] = struct.pack("<Q", 3)
        with self.assertRaisesRegex(GGUFError, "对不上"):
            read_gguf(bytes(blob))

    def test_metadata_count_lies(self):
        blob = bytearray(build_gguf([]))
        blob[16:24] = struct.pack("<Q", 9)
        with self.assertRaisesRegex(GGUFError, "对不上"):
            read_gguf(bytes(blob))

    def test_unsupported_metadata_type(self):
        metadata = [_kv("x", 4, 1)]
        raw = bytearray(build_gguf([], metadata))
        # 把值类型从 UINT32(4) 改成不存在的 13
        key_len = struct.unpack("<Q", raw[24:32])[0]
        pos = 32 + key_len
        raw[pos : pos + 4] = struct.pack("<I", 13)
        with self.assertRaisesRegex(GGUFError, "值类型 13"):
            read_gguf(bytes(raw))

    def test_nested_array_is_rejected(self):
        key = _pack_str("x")
        value = key + struct.pack("<I", 9) + struct.pack("<IQ", 9, 1)
        with self.assertRaisesRegex(GGUFError, "二维数组"):
            read_gguf(build_gguf([], [value]))

    def test_duplicate_metadata_key(self):
        metadata = [_kv("a", 4, 1), _kv("a", 4, 2)]
        with self.assertRaisesRegex(GGUFError, "重复"):
            read_gguf(build_gguf([], metadata))

    def test_invalid_utf8_key(self):
        raw = _pack_str("x") + b"\xff\xfe"
        key = struct.pack("<Q", len(raw)) + raw
        blob = struct.pack("<IIQQ", MAGIC, 3, 0, 1) + key + struct.pack("<I", 4) + b"\x01\x00\x00\x00"
        with self.assertRaisesRegex(GGUFError, "UTF-8"):
            read_gguf(blob)

    def test_duplicate_tensor_name(self):
        tensors = [
            _tensor("w", (2,), "F32", b"\x00" * 8),
            _tensor("w", (2,), "F32", b"\x00" * 8),
        ]
        with self.assertRaisesRegex(GGUFError, "重复"):
            read_gguf(build_gguf(tensors))

    def test_unsupported_quant_type_id(self):
        blob = build_gguf([("w", (256,), 16, b"\x00" * 66)])
        with self.assertRaisesRegex(GGUFError, "id=16"):
            read_gguf(blob)

    def test_elements_not_multiple_of_block(self):
        data = b"\x00" * 18
        with self.assertRaisesRegex(GGUFError, "不是 Q4_0 块大小"):
            read_gguf(build_gguf([("w", (48,), 2, data)]))

    def test_misaligned_tensor_offset(self):
        tensors = [
            _tensor("a", (2,), "F32", b"\x00" * 8),
            _tensor("b", (2,), "F32", b"\x00" * 8),
        ]
        with self.assertRaisesRegex(GGUFError, "对齐"):
            read_gguf(build_gguf(tensors, align_tensors=False))

    def test_bad_alignment_value(self):
        metadata = [_kv("general.alignment", 4, 24)]
        with self.assertRaisesRegex(GGUFError, "2 的幂"):
            read_gguf(build_gguf([], metadata))

    def test_zero_dim_is_rejected(self):
        blob = build_gguf([("w", (0, 2), 0, b"")])
        with self.assertRaisesRegex(GGUFError, "含 0"):
            read_gguf(blob)

    def test_missing_tensor_name(self):
        with _read_one([_tensor("w", (2,), "F32", b"\x00" * 8)]) as f:
            with self.assertRaisesRegex(GGUFError, "没有张量"):
                f.tensor("nope")


class TestPlainTypes(unittest.TestCase):
    def test_f32(self):
        data = np.array([1.0, -2.0], dtype="<f4").tobytes()
        with _read_one([_tensor("w", (2,), "F32", data)]) as f:
            self.assertEqual(f.tensor("w").dtype, np.float32)
            np.testing.assert_array_equal(f.tensor("w"), [1.0, -2.0])

    def test_f16(self):
        data = np.array([1.0, -2.0, 0.5], dtype="<f2").tobytes()
        with _read_one([_tensor("w", (3,), "F16", data)]) as f:
            np.testing.assert_array_equal(f.tensor("w"), [1.0, -2.0, 0.5])

    def test_bf16(self):
        data = struct.pack("<3H", 0x3F80, 0xC000, 0x0000)
        with _read_one([_tensor("w", (3,), "BF16", data)]) as f:
            np.testing.assert_array_equal(f.tensor("w"), [1.0, -2.0, 0.0])

    def test_integer_types(self):
        cases = [
            ("I8", np.array([-1, 2], dtype="i1")),
            ("I16", np.array([-300, 700], dtype="<i2")),
            ("I32", np.array([-70000, 70000], dtype="<i4")),
            ("I64", np.array([-10**10, 10**10], dtype="<i8")),
        ]
        for name, values in cases:
            with _read_one([_tensor("w", (2,), name, values.tobytes())]) as f:
                array = f.tensor("w")
                self.assertEqual(array.dtype, values.dtype)
                np.testing.assert_array_equal(array, values)

    def test_f64(self):
        data = np.array([1.25, -3.5], dtype="<f8").tobytes()
        with _read_one([_tensor("w", (2,), "F64", data)]) as f:
            array = f.tensor("w")
            self.assertEqual(array.dtype, np.float64)
            np.testing.assert_array_equal(array, [1.25, -3.5])

    def test_block_sizes_match_the_type_table(self):
        # 类型表里的块字节数必须与 32 / 256 个元素的实际编码长度一致。
        for name, (_type_id_, block_size, block_bytes) in GGML_TYPES.items():
            data = b"\x00" * block_bytes
            with _read_one([_tensor("w", (block_size,), name, data)]) as f:
                self.assertEqual(f.tensor("w").size, block_size)


class TestQuantBlocks(unittest.TestCase):
    def _dequant(self, type_name, dims, block: bytes):
        with _read_one([_tensor("w", dims, type_name, block)]) as f:
            return f.tensor("w")

    def test_q4_0(self):
        block = struct.pack("<e16s", 2.0, bytes([0x32]) * 16)
        values = self._dequant("Q4_0", (32,), block)
        np.testing.assert_array_equal(values[:16], [-12.0] * 16)
        np.testing.assert_array_equal(values[16:], [-10.0] * 16)

    def test_q4_1(self):
        block = struct.pack("<ee16s", 0.5, 1.0, bytes([0x21]) * 16)
        values = self._dequant("Q4_1", (32,), block)
        np.testing.assert_array_equal(values[:16], [1.5] * 16)
        np.testing.assert_array_equal(values[16:], [2.0] * 16)

    def test_q5_0_high_bits(self):
        block = struct.pack("<e4s16s", 1.0, bytes([0xFF, 0, 0, 0]), bytes(16))
        values = self._dequant("Q5_0", (32,), block)
        np.testing.assert_array_equal(values[:8], [16.0] * 8)
        np.testing.assert_array_equal(values[8:], [0.0] * 24)

    def test_q5_1(self):
        block = struct.pack("<ee4s16s", 1.0, 0.5, bytes([0xFF, 0, 0, 0]), bytes(16))
        values = self._dequant("Q5_1", (32,), block)
        np.testing.assert_array_equal(values[:8], [16.5] * 8)
        np.testing.assert_array_equal(values[8:], [0.5] * 24)

    def test_q8_0(self):
        codes = (np.arange(32, dtype=np.int16) - 16).astype(np.int8)
        block = struct.pack("<e32s", 0.5, codes.tobytes())
        values = self._dequant("Q8_0", (32,), block)
        np.testing.assert_array_equal(values, 0.5 * np.arange(-16, 16, dtype=np.float32))

    def test_q8_1(self):
        block = struct.pack("<ee32s", 0.25, 0.0, np.arange(1, 33, dtype=np.int8).tobytes())
        values = self._dequant("Q8_1", (32,), block)
        np.testing.assert_array_equal(values, 0.25 * np.arange(1, 33, dtype=np.float32))

    def test_q8_k(self):
        codes = np.arange(256, dtype=np.int16).astype(np.int8)
        block = struct.pack("<f256s32s", 0.5, codes.tobytes(), b"\x00" * 32)
        values = self._dequant("Q8_K", (256,), block)
        np.testing.assert_array_equal(values, 0.5 * codes.astype(np.float32))

    def test_q2_k(self):
        block = struct.pack("<16s64see", bytes([0x11]) * 16, bytes([0xE4]) * 64, 1.0, 0.0)
        values = self._dequant("Q2_K", (256,), block)
        expected = np.concatenate([[0.0] * 32, [1.0] * 32, [2.0] * 32, [3.0] * 32] * 2)
        np.testing.assert_array_equal(values, expected)

    def test_q2_k_min_is_subtracted(self):
        block = struct.pack("<16s64see", bytes([0x11]) * 16, bytes(64), 1.0, 1.0)
        values = self._dequant("Q2_K", (256,), block)
        np.testing.assert_array_equal(values, [-1.0] * 256)

    def test_q3_k_six_bit_scales(self):
        # 12 字节 scales 编码 16 个 6 bit 值 63（解码后减 32 → 31）：
        # 前 8 字节是每个字节"低 4 位 + 另一码点的低 4 位"合装，
        # 后 4 字节每字节装 4 个码点的高 2 位，全 1 即 0xFF。
        raw_scales = bytes([0xFF] * 12)
        block = struct.pack("<32s64s12se", bytes([0xFF]) * 32, bytes([0xE4]) * 64, raw_scales, 1.0)
        values = self._dequant("Q3_K", (256,), block)
        expected = np.concatenate(
            [[0.0] * 32, [31.0] * 32, [62.0] * 32, [93.0] * 32] * 2
        )
        np.testing.assert_array_equal(values, expected)

    def test_q3_k_hmask_subtracts_four(self):
        # hmask 全 0 时每个码点减 4；scales 全 0 → (0 - 32) = -32 倍。
        block = struct.pack("<32s64s12se", bytes(32), bytes([0xE4]) * 64, bytes(12), 1.0)
        values = self._dequant("Q3_K", (256,), block)
        expected = np.concatenate(
            [[128.0] * 32, [96.0] * 32, [64.0] * 32, [32.0] * 32] * 2
        )
        np.testing.assert_array_equal(values, expected)

    def test_q4_k(self):
        scales = bytes([1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8])
        block = struct.pack("<ee12s128s", 1.0, 0.0, scales, bytes([0x21]) * 128)
        values = self._dequant("Q4_K", (256,), block)
        expected = []
        for factor in (1, 4, 3, 8, 5, 12, 7, 16):
            expected.extend([float(factor)] * 32)
        np.testing.assert_array_equal(values, expected)

    def test_q4_k_min_and_scale(self):
        # 子块 0：缩放 2、最小值 3 → (2 * 码点) * d - 3 * dmin；子块 1 的缩放与
        # 最小值都是 0，所以高半字节那一批全是 0。
        scales = bytes([2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0])
        block = struct.pack("<ee12s128s", 0.5, 2.0, scales, bytes([0x10]) * 128)
        values = self._dequant("Q4_K", (256,), block)
        np.testing.assert_array_equal(values[:32], [-6.0] * 32)
        np.testing.assert_array_equal(values[32:], [0.0] * 224)

    def test_q5_k(self):
        scales = bytes([1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8])
        block = struct.pack("<ee12s32s128s", 1.0, 0.0, scales, bytes([0xFF]) * 32, bytes(128))
        values = self._dequant("Q5_K", (256,), block)
        expected = []
        for factor in (1, 2, 3, 4, 5, 6, 7, 8):
            expected.extend([float(16 * factor)] * 32)
        np.testing.assert_array_equal(values, expected)

    def test_q6_k(self):
        scales = np.arange(1, 17, dtype=np.int8).tobytes()
        block = struct.pack("<128s64s16se", bytes(128), bytes(64), scales, 0.5)
        values = self._dequant("Q6_K", (256,), block)
        expected = []
        for group in range(8):
            first = -16.0 * (2 * group + 1)
            second = -16.0 * (2 * group + 2)
            expected.extend([first] * 16 + [second] * 16)
        np.testing.assert_array_equal(values, expected)

    def test_multi_block_tensor(self):
        block = struct.pack("<e16s", 1.0, bytes([0x50]) * 16)  # 低 0 / 高 5
        # ggml 的 dims 是 (最内层, 外层)，两个 32 元素的块 → numpy 形状 (2, 32)。
        values = self._dequant("Q4_0", (32, 2), block * 2)
        self.assertEqual(values.shape, (2, 32))
        np.testing.assert_array_equal(values[:, :16], -8.0)
        np.testing.assert_array_equal(values[:, 16:], -3.0)


if __name__ == "__main__":
    unittest.main()
