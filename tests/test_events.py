"""SNN 稀疏事件表示与事件内存池的测试。

判据只有三类：**往返无损**（压了再还原必须逐位相同）、**计数正确**（每步的
事件数与前缀偏移必须与手算一致）、**池真的在复用**（同样的形状第二次 pack
不能再分配）。稀疏表示的价值全在这三条上；它不参与求导，也不改变图语义。
"""

import unittest

import numpy as np

from aetherflow.backends.cuda import CUDA_AVAILABLE, CudaBackend
from aetherflow.runtime.events import (
    EventPool,
    dense_to_events,
    event_mean_rate,
    events_to_dense,
)


def _pulses(rng, shape, density=0.25, dtype=np.float32):
    return (rng.random(shape) < density).astype(dtype)


class TestSpikeEvents(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_roundtrip_2d(self):
        s = _pulses(self.rng, (6, 11), density=0.3)
        ev = dense_to_events(s)
        np.testing.assert_array_equal(events_to_dense(ev), s)

    def test_roundtrip_keeps_tail_shape(self):
        """(T, B, D) 的尾形状必须原样还原，不能压成 (T, B*D)。"""
        s = _pulses(self.rng, (5, 2, 3), density=0.4)
        ev = dense_to_events(s)
        self.assertEqual(ev.shape, (2, 3))
        dense = events_to_dense(ev)
        self.assertEqual(dense.shape, (5, 2, 3))
        np.testing.assert_array_equal(dense, s)

    def test_counts_and_offsets_match_manual(self):
        """每步事件数与前缀偏移就是 CSR 的全部索引信息，必须逐值对齐。"""
        s = _pulses(self.rng, (4, 9), density=0.3)
        ev = dense_to_events(s)
        counts = s.astype(np.int64).sum(axis=1)
        np.testing.assert_array_equal(np.asarray(ev.counts), counts)
        np.testing.assert_array_equal(np.asarray(ev.offsets), np.concatenate([[0], np.cumsum(counts)[:-1]]))
        self.assertEqual(ev.nnz, int(counts.sum()))

        # 每步切片里的下标必须与该步的稠密非零位置逐位相同
        for t in range(4):
            np.testing.assert_array_equal(
                np.asarray(ev.step_indices(t)), np.flatnonzero(s[t])
            )

    def test_nnz_and_density(self):
        s = np.zeros((2, 4), dtype=np.float32)
        s[0, 1] = s[1, 0] = s[1, 3] = 1.0
        ev = dense_to_events(s)
        self.assertEqual(ev.nnz, 3)
        self.assertAlmostEqual(ev.density, 3 / 8)
        self.assertGreater(ev.nbytes, 0)

    def test_all_zero(self):
        s = np.zeros((3, 5), dtype=np.float32)
        ev = dense_to_events(s)
        self.assertEqual(ev.nnz, 0)
        self.assertEqual(ev.density, 0.0)
        np.testing.assert_array_equal(events_to_dense(ev), s)

    def test_all_one(self):
        s = np.ones((3, 5), dtype=np.float32)
        ev = dense_to_events(s)
        self.assertEqual(ev.nnz, 15)
        np.testing.assert_array_equal(events_to_dense(ev), s)

    def test_empty_steps(self):
        for shape in ((0, 4), (3, 0)):
            s = np.zeros(shape, dtype=np.float32)
            ev = dense_to_events(s)
            self.assertEqual(ev.nnz, 0)
            self.assertEqual(events_to_dense(ev).shape, shape)
            self.assertEqual(event_mean_rate(ev).shape, shape[1:])

    def test_rejects_non_binary(self):
        """混着膜电位之类的小数必须报错：压缩丢弃的是零，不是"取整"。"""
        s = np.zeros((2, 3), dtype=np.float32)
        s[0, 0] = 0.5
        with self.assertRaises(ValueError):
            dense_to_events(s)

    def test_rejects_rank1(self):
        with self.assertRaises(ValueError):
            dense_to_events(np.ones(4, dtype=np.float32))

    def test_float16_pulses(self):
        s = _pulses(self.rng, (4, 7), density=0.3, dtype=np.float16)
        ev = dense_to_events(s)
        np.testing.assert_array_equal(events_to_dense(ev), s.astype(np.float32))

    def test_event_mean_rate_matches_dense(self):
        """只读事件算出的发放率必须与稠密均值一致——否则"稀疏"只是换了个存法。"""
        s = _pulses(self.rng, (8, 3, 5), density=0.2)
        ev = dense_to_events(s)
        np.testing.assert_allclose(
            np.asarray(event_mean_rate(ev)), s.mean(axis=0), rtol=0, atol=1e-6
        )


class TestEventPool(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(11)
        self.pool = EventPool()

    def test_first_pack_allocates(self):
        s = _pulses(self.rng, (4, 10))
        self.pool.pack(s)
        st = self.pool.stats()
        self.assertEqual(st.allocations, 1)
        self.assertEqual(st.growths, 1)
        self.assertGreaterEqual(st.capacity, st.nnz)

    def test_second_pack_reuses_buffer(self):
        """同样的规模第二次 pack 不能再分配；这是池存在的唯一理由。"""
        s = _pulses(self.rng, (4, 10))
        self.pool.pack(s)
        before = self.pool.stats()
        self.pool.pack(s)
        after = self.pool.stats()
        self.assertEqual(after.allocations, before.allocations)
        self.assertEqual(after.growths, before.growths)
        self.assertEqual(after.capacity, before.capacity)

    def test_growth_doubles_capacity(self):
        small = _pulses(self.rng, (2, 4), density=0.5)
        self.pool.pack(small)
        cap = self.pool.stats().capacity
        big = _pulses(self.rng, (2, 64), density=0.9)
        ev = self.pool.pack(big)
        st = self.pool.stats()
        self.assertGreater(st.capacity, cap)
        self.assertEqual(st.allocations, 2)
        self.assertEqual(st.growths, 2)
        np.testing.assert_array_equal(events_to_dense(ev), big)

    def test_pack_matches_plain_compression(self):
        s = _pulses(self.rng, (5, 6), density=0.4)
        pooled = self.pool.pack(s)
        plain = dense_to_events(s)
        np.testing.assert_array_equal(np.asarray(pooled.counts), np.asarray(plain.counts))
        np.testing.assert_array_equal(np.asarray(pooled.indices), np.asarray(plain.indices))

    def test_view_is_overwritten_by_next_pack(self):
        """返回的是池内视图：这条语义必须被测试钉住，否则调用方会踩悬空数据。"""
        first = self.pool.pack(_pulses(self.rng, (2, 6), density=0.5))
        snapshot = np.array(first.indices)
        second = self.pool.pack(_pulses(self.rng, (2, 6), density=0.5))
        if first.nnz and second.nnz:
            self.assertFalse(np.array_equal(np.asarray(first.indices), snapshot))

    def test_negative_capacity_rejected(self):
        with self.assertRaises(ValueError):
            EventPool(capacity=-1)

    def test_stats_bytes_tracks_capacity(self):
        self.pool.pack(_pulses(self.rng, (2, 32), density=0.5))
        st = self.pool.stats()
        self.assertEqual(st.bytes, st.capacity * np.dtype(np.int32).itemsize)


@unittest.skipUnless(CUDA_AVAILABLE, "需要 cupy 与 CUDA 设备")
class TestEventsOnDevice(unittest.TestCase):
    """同一套实现必须能在显存数组上跑：稀疏化发生在设备侧，不是先拷回主机。"""

    def setUp(self):
        self.cuda = CudaBackend()
        self.cp = self.cuda.cp
        self.rng = np.random.default_rng(3)

    def test_roundtrip_on_device(self):
        s = self.cp.asarray(_pulses(self.rng, (6, 4, 5), density=0.3))
        ev = dense_to_events(s)
        self.assertEqual(ev.shape, (4, 5))
        np.testing.assert_array_equal(
            self.cp.asnumpy(events_to_dense(ev)),
            self.cp.asnumpy(s),
        )

    def test_pool_reuses_device_buffer(self):
        pool = EventPool()
        s = _pulses(self.rng, (4, 8), density=0.4)
        pool.pack(self.cp.asarray(s))
        before = pool.stats()
        pool.pack(self.cp.asarray(s))
        after = pool.stats()
        self.assertEqual(after.allocations, before.allocations)

    def test_event_mean_rate_on_device(self):
        s = _pulses(self.rng, (7, 3, 4), density=0.25)
        ev = dense_to_events(self.cp.asarray(s))
        np.testing.assert_allclose(
            self.cp.asnumpy(event_mean_rate(ev)), s.mean(axis=0), rtol=0, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
