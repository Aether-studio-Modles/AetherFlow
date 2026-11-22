"""Check the paged decode IR against an independently assembled dense attention."""

import unittest

import numpy as np

from aetherflow import Graph, NoGradientError, ReferenceBackend, backward, dumps, loads
from aetherflow.backends.base import unsupported_ops
from aetherflow.backends.cuda import CUDA_AVAILABLE, CudaBackend
from aetherflow.backends.device_backend import GPU_OPS
from aetherflow.backends.registry import KernelRegistry
from aetherflow.ir.types import f16, f32, f64, i32


def _case(dtype):
    rng = np.random.default_rng(327)
    shape = (8, 4, 2, 16)
    return {
        "q": rng.standard_normal((6, 8, 16)).astype(dtype),
        "k_pages": rng.standard_normal(shape).astype(dtype),
        "v_pages": rng.standard_normal(shape).astype(dtype),
        "block_table": np.array([[6, 0, 3], [2, -1, 1], [7, 5, 0],
                                 [4, 1, 2], [-1, 8, 99], [3, 0, 3]], dtype=np.int32),
        "lengths": np.array([10, 9, 0, -7, 10, 99], dtype=np.int32),
    }


def _graph(feed, **attrs):
    dtypes = {np.dtype("float16"): f16, np.dtype("float32"): f32, np.dtype("int32"): i32}
    g = Graph("paged_decode")
    inputs = [g.input(name, dtypes[x.dtype][x.shape]) for name, x in feed.items()]
    g.output(g.paged_decode(*inputs, **attrs), "y")
    return g


def _dense(feed, window=0, max_len=None, scale=None):
    q, keys, values, table, lengths = tuple(feed.values())[:5]
    pool, page_size, kvheads, dim = keys.shape
    out = np.zeros_like(q)
    capacity = table.shape[1] * page_size if max_len is None else max_len
    scale = 1.0 / np.sqrt(dim) if scale is None else scale
    groups = q.shape[1] // kvheads
    for row, length in enumerate(lengths):
        logical = list(range(min(max(0, int(length)), capacity)))
        addresses = [(table[row, p // page_size], p % page_size) for p in logical]
        if "current_k" in feed:
            addresses.append(None)
        if window:
            addresses = addresses[-window:]
        seq = [(keys[a[0], a[1]], values[a[0], a[1]]) if a is not None else
               (feed["current_k"][row], feed["current_v"][row])
               for a in addresses if a is None or 0 <= a[0] < pool]
        if not seq:
            continue
        k, v = (np.repeat(np.stack(items), groups, axis=1).astype(np.float64) for items in zip(*seq))
        scores = (q[row, :, None, :].astype(np.float64) @ k.transpose(1, 2, 0)) * scale
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        out[row] = ((weights / weights.sum(axis=-1, keepdims=True)) @ v.transpose(1, 0, 2))[:, 0]
    return out


def _long_case(dtype):
    rng = np.random.default_rng(753)
    pool, size, dim = 720, 16, 160
    pages = (pool, size, 2, dim)
    table = np.stack([rng.permutation(pool)[:289] for _ in range(4)]).astype(np.int32)
    table[0, 2:12] = -1
    table[1, 1] = pool
    table[2] = -1
    return {
        "q": rng.standard_normal((4, 6, dim)).astype(dtype),
        "k_pages": rng.standard_normal(pages).astype(dtype),
        "v_pages": rng.standard_normal(pages).astype(dtype),
        "block_table": table,
        "lengths": np.array([4097, 130, 17, -9], dtype=np.int32),
    }


def _with_current(feed):
    rng = np.random.default_rng(21)
    q, k = feed["q"], feed["k_pages"]
    shape = (q.shape[0], k.shape[2], q.shape[-1])
    return dict(feed, current_k=rng.normal(size=shape).astype(q.dtype),
                current_v=rng.normal(size=shape).astype(q.dtype))


class TestPagedDecode(unittest.TestCase):
    def test_current_token_counts_toward_window_and_works_without_history(self):
        backend = ReferenceBackend()
        feed = _with_current(_case(np.float32))
        for attrs in ({"window": 1}, {"window": 4}, {"max_len": 0}, {}):
            with self.subTest(attrs=attrs):
                got = backend.execute(_graph(feed, **attrs), feed)["y"]
                np.testing.assert_allclose(got, _dense(feed, **attrs), rtol=1e-4, atol=1e-6)

    def test_matches_dense_with_ragged_batches_and_shuffled_pages(self):
        backend = ReferenceBackend()
        for dtype in (np.float16, np.float32):
            feed = _case(dtype)
            for attrs in ({}, {"window": 5}, {"window": 17, "max_len": 9, "scale": 0.09},
                          {"window": 2, "max_len": 8, "scale": 0.0}, {"max_len": 0}):
                with self.subTest(dtype=dtype, attrs=attrs):
                    actual = backend.execute(_graph(feed, **attrs), feed)["y"]
                    self.assertEqual(actual.dtype, feed["q"].dtype)
                    self.assertEqual(actual.shape, feed["q"].shape)
                    np.testing.assert_allclose(actual, _dense(feed, **attrs), atol=2e-5, rtol=1e-3)
                    np.testing.assert_array_equal(actual[2:5], 0)
                    if attrs.get("window") == 2:
                        # This row's final two logical slots are both invalid. Do not
                        # substitute older valid slots from before the sliding window.
                        np.testing.assert_array_equal(actual[1], 0)

    def test_does_not_read_masked_or_unwritten_tokens(self):
        feed = _case(np.float16)
        feed["lengths"][:] = [10, 0, 0, 0, 0, 0]
        feed["k_pages"][3, 2:] = np.nan
        feed["v_pages"][3, 2:] = np.nan
        out = ReferenceBackend().execute(_graph(feed), feed)["y"]
        self.assertTrue(np.isfinite(out).all())
        np.testing.assert_allclose(out, _dense(feed), rtol=1e-3, atol=2e-5)

    def test_stable_softmax_for_large_f16_logits(self):
        feed = _case(np.float16)
        feed["q"] *= 512
        feed["k_pages"] *= 512
        out = ReferenceBackend().execute(_graph(feed), feed)["y"]
        self.assertTrue(np.isfinite(out).all())
        np.testing.assert_allclose(out, _dense(feed), rtol=1e-3, atol=2e-5)

    def test_graph_roundtrip_and_forward_only_capability(self):
        feed = _case(np.float32)
        graph = _graph(feed, window=5, max_len=10, scale=0.2)
        restored = loads(dumps(graph))
        backend = ReferenceBackend()
        np.testing.assert_array_equal(backend.execute(restored, feed)["y"],
                                      backend.execute(graph, feed)["y"])
        caps = backend.capabilities()
        self.assertTrue(caps.supports_node(restored.nodes[0]))
        self.assertFalse(caps.supports_node(restored.nodes[0], backward=True))
        self.assertEqual(unsupported_ops(restored, GPU_OPS), ["paged_decode"])
        with self.assertRaisesRegex(NoGradientError, "paged_decode"):
            backward(restored, wrt=[restored.inputs[0]])

    def test_cuda_registry_without_device(self):
        backend = CudaBackend.__new__(CudaBackend)
        backend.kernels = KernelRegistry("cuda")
        backend._register_kernels()
        caps = backend.capabilities()
        self.assertIn("paged_decode", caps)
        self.assertFalse(caps.describe("paged_decode").backward)

    def test_rejects_invalid_options(self):
        feed = _case(np.float32)
        for key, values in {"window": (-1, True, 1.5), "max_len": (-1, 13, True, 4.0),
                            "scale": (True, "1.0", float("inf"), float("nan"), 10**1000)}.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex((ValueError, TypeError), key):
                        _graph(feed, **{key: value})

    def test_rejects_bad_shapes_and_dtypes_before_execution(self):
        shapes = [f32[6, 8, 16], f32[8, 4, 2, 16], f32[8, 4, 2, 16], i32[6, 3], i32[6]]
        cases = (
            (0, f32[6, 8, 1, 16], ValueError, "expects"),
            (0, f32[6, 7, 16], ValueError, "divisible"),
            (0, f32["B", 8, 16], ValueError, "static"),
            (0, f32[6, 0, 16], ValueError, "positive"),
            (0, f64[6, 8, 16], TypeError, "matching f16 or f32"),
            (1, f16[8, 4, 2, 16], TypeError, "matching f16 or f32"),
            (2, f32[8, 4, 2, 8], ValueError, "shapes"),
            (3, f32[6, 3], TypeError, "i32"),
            (3, i32[5, 3], ValueError, "shapes"),
            (4, i32[5], ValueError, "shapes"),
        )
        for index, spec, error, message in cases:
            with self.subTest(index=index, spec=spec):
                g = Graph("bad_paged_decode")
                inputs = [g.input(f"x{i}", spec if i == index else s) for i, s in enumerate(shapes)]
                with self.assertRaisesRegex(error, message):
                    g.paged_decode(*inputs)


@unittest.skipUnless(CUDA_AVAILABLE, "requires CuPy and a CUDA device")
class TestCudaPagedDecode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = CudaBackend()

    def _check(self, feed, **attrs):
        resident = {name: self.backend.to_array(x) for name, x in feed.items()}
        out = self.backend.execute_device(_graph(feed, **attrs), resident)["y"]
        self.assertIsInstance(out, self.backend.cp.ndarray)
        np.testing.assert_allclose(self.backend.to_host(out), _dense(feed, **attrs),
                                   atol=5e-4, rtol=2e-3)

    def test_single_tile_masks_poisoned_tail(self):
        for dtype in (np.float16, np.float32):
            feed = _case(dtype)
            feed["lengths"][:] = [10, 0, 0, 0, 0, 0]
            feed["k_pages"][3, 2:] = np.nan
            feed["v_pages"][3, 2:] = np.nan
            with self.subTest(dtype=dtype):
                self._check(feed, window=5)
                self._check(feed, max_len=0)

    def test_split_merge_ragged_rows(self):
        for dtype in (np.float16, np.float32):
            feed = _long_case(dtype)
            for attrs in ({}, {"window": 131, "scale": 0.1}):
                with self.subTest(dtype=dtype, attrs=attrs):
                    self._check(feed, **attrs)

    def test_current_token_across_splits_and_sliding_window(self):
        for dtype in (np.float16, np.float32):
            feed = _with_current(_long_case(dtype))
            for attrs in ({}, {"window": 129}, {"window": 1}, {"max_len": 0}):
                with self.subTest(dtype=dtype, attrs=attrs):
                    self._check(feed, **attrs)

    def test_multiple_tiles_in_one_split(self):
        # Repeat physical pages to cross the split cap without a large pool.
        feed = _case(np.float32)
        feed["block_table"] = np.tile(feed["block_table"][:1], (1, 3000))
        feed["q"] = feed["q"][:1]
        feed["lengths"] = np.array([33333], dtype=np.int32)
        self._check(feed)

    def test_capture_reuses_buffers_with_updated_metadata(self):
        feed = _long_case(np.float16)
        graph = _graph(feed)
        buffers = {name: self.backend.to_array(x) for name, x in feed.items()}
        plan = self.backend.capture(graph, buffers, shared_buffers=buffers)
        for lengths in ([4097, 130, 17, 0], [0, 258, 0, 15]):
            feed["lengths"][:] = lengths
            buffers["lengths"].set(feed["lengths"])
            np.testing.assert_allclose(plan.run(buffers)["y"], _dense(feed),
                                       atol=5e-4, rtol=2e-3)

    def test_capture_and_replay_do_not_truncate_integer_metadata(self):
        feed = _case(np.float32)
        graph = _graph(feed)
        plan = self.backend.capture(graph, dict(feed, lengths=feed["lengths"].astype(np.int64)))
        self.assertEqual(plan.buffers["lengths"].dtype, np.dtype(np.int32))
        for values, error in ((np.full(6, 1.5), TypeError),
                              (np.full(6, 2**32, dtype=np.int64), ValueError)):
            with self.assertRaisesRegex(error, "%lengths"):
                plan.run(dict(feed, lengths=values))
        np.testing.assert_allclose(plan.run(feed)["y"], _dense(feed), atol=5e-4, rtol=2e-3)
