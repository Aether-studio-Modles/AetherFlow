"""分页 KV cache 的存储语义验收。

判据都是可执行的：

1. **追加 / 取回**：跨页追加后 ``gather`` 与"逐段拼起来"逐位一致。
2. **回收与复用**：页被释放后发给另一条序列，两条序列的数据不串。
3. **回滚**：``truncate`` 只释放长度之外的整页，缩短后再追加仍然正确。
4. **容量**：页不够时抛 ``PagedCacheFullError``，且失败前状态原样保留。
5. **拒绝面**：形状、层数、补齐长度、重复分配都必须显式报错。
"""

import unittest

import numpy as np

from aetherflow.runtime import (
    PagedCacheFullError,
    PagedCacheManager,
    PagedKVCache,
)


def _cache(num_blocks=4, block_size=2, layers=2, heads=2, dim=3):
    return PagedKVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=layers,
        num_heads=heads,
        head_dim=dim,
    )


def _kv(tokens, heads=2, dim=3, seed=0):
    rng = np.random.default_rng(seed)
    keys = rng.standard_normal((tokens, heads, dim)).astype(np.float32)
    values = rng.standard_normal((tokens, heads, dim)).astype(np.float32)
    return keys, values


def _append(cache, rid, tokens, seed=0):
    """按层追加同一批 token，返回 ``(keys, values)`` 两份按层列表。

    ``cache`` 可以是 ``PagedKVCache`` 或 ``PagedCacheManager``——后者的
    ``append`` 才会演练换出 / 换入策略。
    """
    store = getattr(cache, "cache", cache)
    keys = []
    values = []
    for layer in range(store.num_layers):
        k, v = _kv(tokens, store.num_heads, store.head_dim, seed=seed + layer)
        keys.append(k)
        values.append(v)
    cache.append(rid, keys, values)
    return keys, values


class TestPagedKVCache(unittest.TestCase):
    def test_allocate_and_basic_views(self):
        cache = _cache()
        cache.allocate(7)
        self.assertIn(7, cache)
        self.assertEqual(cache.length(7), 0)
        self.assertEqual(cache.block_table(7), ())
        self.assertEqual(cache.used_blocks, 0)
        self.assertEqual(cache.free_blocks, 4)
        self.assertEqual(cache.num_sequences, 1)
        self.assertEqual(cache.stats()["block_size"], 2)
        self.assertEqual(cache.capacity_tokens, 8)

    def test_append_across_pages_and_gather(self):
        cache = _cache(block_size=2)
        cache.allocate(0)
        pieces = []
        for step in range(5):
            keys, _ = _append(cache, 0, 1, seed=step)
            pieces.append(keys[0][0])
        self.assertEqual(cache.length(0), 5)
        # 5 个 token、页宽 2 → 3 页
        self.assertEqual(len(cache.block_table(0)), 3)
        self.assertEqual(cache.used_blocks, 3)
        got, _ = cache.gather(0, 0)
        np.testing.assert_array_equal(got, np.stack(pieces))

    def test_layers_share_the_block_table(self):
        cache = _cache()
        cache.allocate(1)
        keys, values = _append(cache, 1, 3, seed=5)
        np.testing.assert_array_equal(cache.gather(1, 0)[0], keys[0])
        np.testing.assert_array_equal(cache.gather(1, 1)[0], keys[1])
        np.testing.assert_array_equal(cache.gather(1, 1)[1], values[1])
        self.assertEqual(cache.length(1), 3)

    def test_free_and_reuse_keeps_sequences_isolated(self):
        cache = _cache(num_blocks=1, block_size=2)
        cache.allocate(0)
        cache.allocate(1)
        _append(cache, 0, 2, seed=1)
        self.assertEqual(cache.free_blocks, 0)
        with self.assertRaises(PagedCacheFullError):
            _append(cache, 1, 2, seed=3)
        # 释放第一条后，页可以发给第二条
        self.assertTrue(cache.free(0))
        second, _ = _append(cache, 1, 2, seed=2)
        np.testing.assert_array_equal(cache.gather(1, 0)[0], second[0])
        self.assertNotIn(0, cache)
        self.assertEqual(cache.num_sequences, 1)

    def test_free_is_idempotent(self):
        cache = _cache()
        cache.allocate(3)
        self.assertTrue(cache.free(3))
        self.assertFalse(cache.free(3))
        self.assertEqual(cache.used_blocks, 0)

    def test_truncate_releases_whole_pages_and_reappend_works(self):
        cache = _cache(num_blocks=4, block_size=2)
        cache.allocate(0)
        keys, _ = _append(cache, 0, 5, seed=7)
        self.assertEqual(len(cache.block_table(0)), 3)
        freed = cache.truncate(0, 2)
        self.assertEqual(freed, 2)
        self.assertEqual(cache.length(0), 2)
        self.assertEqual(cache.used_blocks, 1)
        np.testing.assert_array_equal(cache.gather(0, 0)[0], keys[0][:2])
        # 缩短后再追加，新数据落在从空闲链领来的页上
        extra, _ = _append(cache, 0, 2, seed=8)
        joined = np.concatenate([keys[0][:2], extra[0]], axis=0)
        np.testing.assert_array_equal(cache.gather(0, 0)[0], joined)

    def test_truncate_to_zero_then_reset(self):
        cache = _cache()
        cache.allocate(0)
        _append(cache, 0, 4, seed=9)
        cache.truncate(0, 0)
        self.assertEqual(cache.length(0), 0)
        self.assertEqual(cache.used_blocks, 0)
        _append(cache, 0, 3, seed=10)
        freed = cache.reset(0)
        self.assertEqual(freed, 2)
        self.assertEqual(cache.length(0), 0)
        self.assertIn(0, cache)

    def test_capacity_failure_leaves_state_untouched(self):
        cache = _cache(num_blocks=1, block_size=2)
        cache.allocate(0)
        _append(cache, 0, 2, seed=11)
        before = cache.stats()
        with self.assertRaisesRegex(PagedCacheFullError, "新页"):
            _append(cache, 0, 1, seed=12)
        self.assertEqual(cache.stats(), before)
        self.assertEqual(cache.length(0), 2)
        # 回滚一个 token 后，腾出的页内空间可以继续用（没有半写状态）
        cache.truncate(0, 1)
        _append(cache, 0, 1, seed=13)
        self.assertEqual(cache.length(0), 2)

    def test_pad_to_fills_with_zeros(self):
        cache = _cache(num_blocks=2, block_size=3)
        cache.allocate(0)
        keys, values = _append(cache, 0, 2, seed=14)
        padded_k, padded_v = cache.gather(0, 0, pad_to=5)
        self.assertEqual(padded_k.shape, (5, 2, 3))
        np.testing.assert_array_equal(padded_k[:2], keys[0])
        np.testing.assert_array_equal(padded_k[2:], np.zeros((3, 2, 3)))
        np.testing.assert_array_equal(padded_v[:2], values[0])
        np.testing.assert_array_equal(padded_v[2:], np.zeros((3, 2, 3)))

    def test_single_layer_accepts_flat_arrays(self):
        cache = _cache(layers=1)
        cache.allocate(0)
        keys, values = _kv(3, seed=15)
        cache.append(0, keys, values)
        np.testing.assert_array_equal(cache.gather(0, 0)[0], keys)

    def test_gather_returns_copies(self):
        cache = _cache()
        cache.allocate(0)
        keys, _ = _append(cache, 0, 2, seed=16)
        got, _ = cache.gather(0, 0)
        got[0] = 123.0
        again, _ = cache.gather(0, 0)
        np.testing.assert_array_equal(again[0], keys[0][0])

    def test_peak_blocks_tracks_history(self):
        cache = _cache(num_blocks=4, block_size=2)
        cache.allocate(0)
        cache.allocate(1)
        _append(cache, 0, 4, seed=17)
        _append(cache, 1, 2, seed=18)
        self.assertEqual(cache.peak_blocks, 3)
        cache.free(0)
        cache.free(1)
        self.assertEqual(cache.used_blocks, 0)
        self.assertEqual(cache.peak_blocks, 3)

    def test_rejections(self):
        cache = _cache()
        cache.allocate(0)
        keys, _ = _kv(1)
        with self.assertRaisesRegex(ValueError, "已经分配"):
            cache.allocate(0)
        with self.assertRaises(TypeError):
            cache.allocate(True)
        with self.assertRaises(KeyError):
            cache.length(99)
        with self.assertRaises(KeyError):
            cache.block_table(99)
        with self.assertRaises(KeyError):
            _append(cache, 99, 1)
        with self.assertRaisesRegex(ValueError, "每层一项"):
            cache.append(0, [keys], [keys])
        with self.assertRaisesRegex(ValueError, "各层追加长度必须一致"):
            cache.append(0, [keys, _kv(2)[0]], [keys, _kv(2)[1]])
        with self.assertRaisesRegex(ValueError, "形状"):
            cache.append(0, [_kv(1, heads=3)[0]] * 2, [_kv(1, heads=3)[1]] * 2)
        with self.assertRaisesRegex(ValueError, "一致"):
            cache.append(0, [_kv(1)[0]] * 2, [_kv(2)[1]] * 2)
        with self.assertRaisesRegex(ValueError, "至少"):
            cache.append(0, [np.zeros((0, 2, 3))] * 2, [np.zeros((0, 2, 3))] * 2)
        with self.assertRaises(TypeError):
            cache.append(0, 1, 1)
        cache.append(0, [keys] * 2, [_kv(1)[1]] * 2)
        with self.assertRaisesRegex(ValueError, "只能缩短"):
            cache.truncate(0, 3)
        with self.assertRaisesRegex(ValueError, "不能为负"):
            cache.truncate(0, -1)
        with self.assertRaises(ValueError):
            cache.gather(0, 2)
        with self.assertRaises(ValueError):
            cache.gather(0, 0, pad_to=-1)
        with self.assertRaises(ValueError):
            cache.gather(0, 0, pad_to=cache.capacity_tokens + 1)
        with self.assertRaises(TypeError):
            cache.gather(0, 0, pad_to=True)

    def test_configuration_rejections(self):
        with self.assertRaises(ValueError):
            PagedKVCache(
                num_blocks=0, block_size=2, num_layers=1, num_heads=1, head_dim=1
            )
        with self.assertRaises(TypeError):
            PagedKVCache(
                num_blocks=True, block_size=2, num_layers=1, num_heads=1, head_dim=1
            )
        with self.assertRaises(TypeError):
            PagedKVCache(
                num_blocks=1,
                block_size=2,
                num_layers=1,
                num_heads=1,
                head_dim=1,
                dtype=np.int32,
            )


class TestSwap(unittest.TestCase):
    """换出 / 换入：释放物理页、保留主机副本，换回后逐位一致。

    判据只有一条是数值的（换入后的 ``gather`` 必须与换出前相同），其余都是
    容量语义：换出的页能被别的序列拿走、换入页不够时保持换出态、换出期间
    读写被显式拒绝。
    """

    def test_swap_out_and_back_preserves_values(self):
        cache = _cache(num_blocks=6, block_size=2, layers=2)
        cache.allocate(1)
        _append(cache, 1, 5, seed=3)
        before = [cache.gather(1, layer) for layer in range(cache.num_layers)]
        self.assertEqual(cache.swap_out(1), 3)
        self.assertEqual(cache.used_blocks, 0)
        self.assertEqual(cache.free_blocks, 6)
        self.assertEqual(cache.length(1), 5)
        self.assertTrue(cache.is_swapped(1))
        self.assertEqual(cache.swapped_tokens, 5)
        self.assertEqual(cache.block_table(1), ())
        self.assertEqual(cache.stats()["swapped_sequences"], 1)
        self.assertEqual(cache.swap_in(1), 3)
        self.assertFalse(cache.is_swapped(1))
        self.assertEqual(cache.used_blocks, 3)
        self.assertEqual(cache.swapped_tokens, 0)
        for layer, (want_k, want_v) in enumerate(before):
            got_k, got_v = cache.gather(1, layer)
            np.testing.assert_array_equal(got_k, want_k)
            np.testing.assert_array_equal(got_v, want_v)

    def test_swapped_pages_are_reusable_and_swap_in_keeps_state(self):
        cache = _cache(num_blocks=2, block_size=2)
        cache.allocate(1)
        _append(cache, 1, 4, seed=1)
        want = cache.gather(1, 0)[0]
        cache.swap_out(1)

        cache.allocate(2)
        _append(cache, 2, 4, seed=9)
        self.assertEqual(cache.free_blocks, 0)
        with self.assertRaises(PagedCacheFullError):
            cache.swap_in(1)
        # 失败必须原样保留：仍然是换出态，长度与主机副本都在
        self.assertTrue(cache.is_swapped(1))
        self.assertEqual(cache.length(1), 4)
        self.assertEqual(cache.used_blocks, 2)

        cache.free(2)
        self.assertEqual(cache.swap_in(1), 2)
        np.testing.assert_array_equal(cache.gather(1, 0)[0], want)

    def test_swapped_sequence_rejects_reads_and_writes(self):
        cache = _cache()
        cache.allocate(1)
        _append(cache, 1, 3, seed=2)
        cache.swap_out(1)
        with self.assertRaisesRegex(ValueError, "已换出"):
            cache.gather(1, 0)
        with self.assertRaisesRegex(ValueError, "已换出"):
            _append(cache, 1, 1)
        self.assertEqual(cache.length(1), 3)

    def test_truncate_swapped_copy_then_swap_in(self):
        cache = _cache(num_blocks=4, block_size=2)
        cache.allocate(1)
        keys, values = _append(cache, 1, 5, seed=4)
        cache.swap_out(1)
        self.assertEqual(cache.truncate(1, 3), 0)
        self.assertEqual(cache.length(1), 3)
        self.assertEqual(cache.swapped_tokens, 3)
        self.assertEqual(cache.swap_in(1), 2)
        np.testing.assert_array_equal(cache.gather(1, 0)[0], keys[0][:3])
        np.testing.assert_array_equal(cache.gather(1, 1)[1], values[1][:3])
        # 换入后可以继续追加，长度接着走
        _append(cache, 1, 2, seed=7)
        self.assertEqual(cache.length(1), 5)

    def test_reset_and_free_drop_swapped_copy(self):
        cache = _cache()
        cache.allocate(1)
        _append(cache, 1, 3, seed=5)
        cache.swap_out(1)
        self.assertEqual(cache.reset(1), 0)
        self.assertFalse(cache.is_swapped(1))
        self.assertEqual(cache.length(1), 0)
        self.assertEqual(cache.swapped_tokens, 0)
        _append(cache, 1, 2, seed=6)
        self.assertEqual(cache.length(1), 2)

        cache.swap_out(1)
        self.assertTrue(cache.free(1))
        self.assertNotIn(1, cache)
        self.assertEqual(cache.swapped_tokens, 0)
        self.assertEqual(cache.free_blocks, 4)

    def test_swap_rejections(self):
        cache = _cache()
        with self.assertRaises(KeyError):
            cache.swap_out(9)
        cache.allocate(1)
        with self.assertRaisesRegex(ValueError, "没有换出"):
            cache.swap_in(1)
        _append(cache, 1, 1)
        cache.swap_out(1)
        with self.assertRaisesRegex(ValueError, "已经换出"):
            cache.swap_out(1)

    def test_single_layer_swap(self):
        cache = _cache(num_blocks=4, block_size=2, layers=1)
        cache.allocate(1)
        keys, values = _append(cache, 1, 3, seed=8)
        cache.swap_out(1)
        self.assertEqual(cache.swap_in(1), 2)
        got_k, got_v = cache.gather(1, 0)
        np.testing.assert_array_equal(got_k, keys[0])
        np.testing.assert_array_equal(got_v, values[0])


class TestPagedCacheManager(unittest.TestCase):
    """调度策略：容量不足按 LRU 换出、访问自动换入、pin 保护、失败不丢数据。

    判据是三条可执行的等式：

    1. 换出 / 换入后的 ``gather`` 与换出前逐位相同；
    2. 换出顺序严格是"最久未使用优先"，``touch`` 能改变受害者；
    3. 换不进去时抛 ``PagedCacheFullError``，所有序列的长度与数据保持不变。
    """

    def test_append_evicts_lru_then_reload_is_bitwise(self):
        cache = _cache(num_blocks=3, block_size=2, layers=2)
        mgr = PagedCacheManager(cache)
        mgr.allocate(1)
        _append(mgr, 1, 4, seed=1)
        before = [cache.gather(1, layer) for layer in range(cache.num_layers)]

        mgr.allocate(2)
        _append(mgr, 2, 4, seed=2)  # 空闲页为 0 → 换出最久未使用的序列 1
        self.assertTrue(cache.is_swapped(1))
        self.assertFalse(cache.is_swapped(2))
        self.assertEqual(cache.length(1), 4)
        self.assertEqual(mgr.stats()["swaps_out"], 1)
        self.assertEqual(mgr.stats()["swaps_in"], 0)

        # 访问仍在页上的序列不产生换入
        mgr.gather(2, 0)
        self.assertEqual(mgr.stats()["swaps_in"], 0)

        # 访问被换出的序列：先换出 2，再换入 1，数值逐位不变
        for layer, (want_k, want_v) in enumerate(before):
            got_k, got_v = mgr.gather(1, layer)
            np.testing.assert_array_equal(got_k, want_k)
            np.testing.assert_array_equal(got_v, want_v)
        self.assertEqual(mgr.stats()["swaps_in"], 1)
        self.assertTrue(cache.is_swapped(2))

    def test_touch_changes_the_victim(self):
        cache = _cache(num_blocks=3, block_size=2)
        mgr = PagedCacheManager(cache)
        for rid in (1, 2, 3):
            mgr.allocate(rid)
            _append(mgr, rid, 2, seed=rid)  # 各占 1 页，池满
        mgr.touch(1)  # 使用顺序变成 2, 3, 1
        mgr.allocate(4)
        _append(mgr, 4, 2, seed=4)
        self.assertTrue(cache.is_swapped(2))
        self.assertFalse(cache.is_swapped(3))
        self.assertFalse(cache.is_swapped(1))

    def test_pinned_sequences_are_skipped(self):
        cache = _cache(num_blocks=4, block_size=2)
        mgr = PagedCacheManager(cache)
        mgr.allocate(1)
        _append(mgr, 1, 4, seed=1)
        mgr.pin(1)
        mgr.allocate(2)
        _append(mgr, 2, 4, seed=2)
        self.assertFalse(cache.is_swapped(1))
        self.assertFalse(cache.is_swapped(2))

        mgr.allocate(3)
        _append(mgr, 3, 2, seed=3)  # 只能换出 2
        self.assertTrue(cache.is_swapped(2))
        self.assertFalse(cache.is_swapped(1))
        self.assertFalse(cache.is_swapped(3))
        self.assertTrue(mgr.is_pinned(1))
        self.assertTrue(mgr.unpin(1))
        self.assertFalse(mgr.unpin(1))
        self.assertFalse(mgr.unpin(2))  # 2 没被 pin

    def test_all_pinned_raises_and_keeps_every_state(self):
        cache = _cache(num_blocks=4, block_size=2)
        mgr = PagedCacheManager(cache)
        for rid in (1, 2):
            mgr.allocate(rid)
            _append(mgr, rid, 4, seed=rid)
            mgr.pin(rid)
        before = cache.gather(1, 0)[0]
        mgr.allocate(3)
        with self.assertRaises(PagedCacheFullError):
            _append(mgr, 3, 2, seed=3)
        self.assertEqual(cache.length(3), 0)
        self.assertEqual(cache.used_blocks, 4)
        self.assertFalse(cache.is_swapped(1))
        self.assertFalse(cache.is_swapped(2))
        np.testing.assert_array_equal(cache.gather(1, 0)[0], before)

    def test_single_sequence_cannot_evict_itself(self):
        cache = _cache(num_blocks=2, block_size=2)
        mgr = PagedCacheManager(cache)
        mgr.allocate(1)
        _append(mgr, 1, 4, seed=1)
        before = cache.gather(1, 0)[0]
        with self.assertRaises(PagedCacheFullError):
            _append(mgr, 1, 2, seed=2)
        self.assertEqual(cache.length(1), 4)
        np.testing.assert_array_equal(cache.gather(1, 0)[0], before)

    def test_swap_in_failure_keeps_swapped_state_then_retries(self):
        cache = _cache(num_blocks=2, block_size=2)
        mgr = PagedCacheManager(cache)
        mgr.allocate(1)
        _append(mgr, 1, 4, seed=1)
        want = cache.gather(1, 0)[0]
        mgr.evict(1)

        mgr.allocate(2)
        _append(mgr, 2, 4, seed=9)
        mgr.pin(2)
        with self.assertRaises(PagedCacheFullError):
            mgr.gather(1, 0)
        self.assertTrue(cache.is_swapped(1))
        self.assertEqual(cache.length(1), 4)

        mgr.free(2)
        np.testing.assert_array_equal(mgr.gather(1, 0)[0], want)

    def test_hooks_report_pages(self):
        cache = _cache(num_blocks=2, block_size=2)
        events = []
        mgr = PagedCacheManager(
            cache,
            on_swap_out=lambda rid, pages: events.append(("out", rid, pages)),
            on_swap_in=lambda rid, pages: events.append(("in", rid, pages)),
        )
        mgr.allocate(1)
        _append(mgr, 1, 4, seed=1)
        mgr.allocate(2)
        _append(mgr, 2, 4, seed=2)  # 换出 1
        mgr.gather(1, 0)  # 换出 2，再换入 1
        self.assertEqual(
            events, [("out", 1, 2), ("out", 2, 2), ("in", 1, 2)]
        )

    def test_explicit_evict_ignores_pin(self):
        cache = _cache(num_blocks=4, block_size=2)
        mgr = PagedCacheManager(cache)
        mgr.allocate(1)
        _append(mgr, 1, 4, seed=1)
        mgr.pin(1)
        self.assertEqual(mgr.evict(1), 2)
        self.assertTrue(cache.is_swapped(1))
        self.assertTrue(mgr.is_pinned(1))
        self.assertEqual(mgr.ensure_resident(1), 2)
        self.assertEqual(mgr.ensure_resident(1), 0)
        self.assertEqual(mgr.stats()["swaps_in"], 1)

    def test_truncate_and_reset_forwarding(self):
        cache = _cache(num_blocks=4, block_size=2)
        mgr = PagedCacheManager(cache)
        mgr.allocate(1)
        keys, _ = _append(mgr, 1, 5, seed=4)
        mgr.evict(1)
        self.assertEqual(mgr.truncate(1, 3), 0)
        self.assertEqual(mgr.length(1), 3)
        np.testing.assert_array_equal(mgr.gather(1, 0)[0], keys[0][:3])
        self.assertEqual(mgr.reset(1), 2)
        self.assertEqual(mgr.length(1), 0)
        self.assertFalse(cache.is_swapped(1))

    def test_rejections_and_idempotent_free(self):
        cache = _cache()
        with self.assertRaises(TypeError):
            PagedCacheManager("not a cache")
        with self.assertRaises(TypeError):
            PagedCacheManager(cache, on_swap_out=1)
        mgr = PagedCacheManager(cache)
        with self.assertRaises(KeyError):
            mgr.pin(9)
        with self.assertRaises(KeyError):
            mgr.unpin(9)
        with self.assertRaises(KeyError):
            mgr.evict(9)
        with self.assertRaises(KeyError):
            mgr.length(9)
        self.assertFalse(mgr.free(9))

        mgr.allocate(1)
        mgr.pin(1)
        self.assertTrue(mgr.unpin(1))
        self.assertFalse(mgr.unpin(1))
        _append(mgr, 1, 2, seed=1)
        mgr.evict(1)
        with self.assertRaisesRegex(ValueError, "已经换出"):
            mgr.evict(1)
        mgr.allocate(2)
        with self.assertRaisesRegex(ValueError, "没有换出"):
            cache.swap_in(2)


class TestSharedPrefix(unittest.TestCase):
    """``fork`` 的写时复制语义：共享页只算一次，写前先拆页。"""

    def _check_invariant(self, cache):
        """每条序列持有的页数之和 == 全部引用计数之和。"""
        pages = sum(len(table) for table in cache._tables.values())
        self.assertEqual(sum(cache._refs.values()), pages)
        self.assertEqual(len(cache._refs), cache.used_blocks)

    def test_fork_shares_pages_and_reads_are_identical(self):
        cache = _cache(num_blocks=4, block_size=2)
        cache.allocate(1)
        keys, _ = _append(cache, 1, 3)
        used = cache.used_blocks
        cache.fork(1, 2)
        self.assertEqual(cache.used_blocks, used)  # 分叉不多占页
        self.assertEqual(cache.length(2), 3)
        self.assertEqual(cache.shared_blocks, 2)
        self.assertEqual(cache.stats()["forks"], 1)
        self.assertEqual(cache.block_table(1), cache.block_table(2))
        for layer in range(cache.num_layers):
            k1, v1 = cache.gather(1, layer)
            k2, v2 = cache.gather(2, layer)
            np.testing.assert_array_equal(k1, k2)
            np.testing.assert_array_equal(v1, v2)
            np.testing.assert_array_equal(k1, keys[layer])
        self._check_invariant(cache)

    def test_child_append_cows_only_the_written_page(self):
        cache = _cache(num_blocks=4, block_size=2)
        cache.allocate(1)
        keys, _ = _append(cache, 1, 3)  # 2 页，第 2 页只用了 1 个 slot
        cache.fork(1, 2)
        shared_tail = cache.block_table(2)[1]
        extra, _ = _append(cache, 2, 1, seed=99)
        self.assertEqual(cache.length(1), 3)
        self.assertEqual(cache.length(2), 4)
        self.assertEqual(cache.used_blocks, 3)  # 拆尾页多占一页
        self.assertEqual(cache.shared_blocks, 1)  # 只剩首页仍共享
        self.assertNotEqual(cache.block_table(2)[1], shared_tail)
        for layer in range(cache.num_layers):
            k1, _ = cache.gather(1, layer)
            k2, _ = cache.gather(2, layer)
            np.testing.assert_array_equal(k1, keys[layer])
            np.testing.assert_array_equal(k2[:3], k1)
            np.testing.assert_array_equal(k2[3:], extra[layer])
        self._check_invariant(cache)

    def test_parent_append_does_not_reach_the_child(self):
        cache = _cache(num_blocks=4, block_size=2)
        cache.allocate(1)
        _append(cache, 1, 3)
        cache.fork(1, 2)
        child_keys, child_values = cache.gather(2, 0)
        extra, _ = _append(cache, 1, 1, seed=7)
        k1, v1 = cache.gather(1, 0)
        k2, v2 = cache.gather(2, 0)
        np.testing.assert_array_equal(k1[3:], extra[0])
        np.testing.assert_array_equal(k2, child_keys)
        np.testing.assert_array_equal(v2, child_values)
        self.assertEqual(cache.length(2), 3)
        self._check_invariant(cache)

    def test_free_one_side_keeps_pages_for_the_other(self):
        cache = _cache(num_blocks=4, block_size=2)
        cache.allocate(1)
        keys, _ = _append(cache, 1, 4)
        cache.fork(1, 2)
        self.assertTrue(cache.free(2))
        self.assertEqual(cache.used_blocks, 2)  # 一条放手，页还留着
        self.assertEqual(cache.shared_blocks, 0)
        k, _ = cache.gather(1, 0)
        np.testing.assert_array_equal(k, keys[0])
        self.assertTrue(cache.free(1))
        self.assertEqual(cache.used_blocks, 0)
        self.assertEqual(cache.free_blocks, cache.num_blocks)
        self._check_invariant(cache)

    def test_truncate_on_one_side_only_drops_references(self):
        cache = _cache(num_blocks=4, block_size=2)
        cache.allocate(1)
        keys, _ = _append(cache, 1, 4)
        cache.fork(1, 2)
        # 回滚掉的那页只减引用；首页仍被两条序列共享。
        self.assertEqual(cache.truncate(1, 1), 0)
        self.assertEqual(cache.used_blocks, 2)
        self.assertEqual(cache.shared_blocks, 1)
        k2, _ = cache.gather(2, 0)
        np.testing.assert_array_equal(k2, keys[0])
        self.assertTrue(cache.free(2))
        self.assertEqual(cache.used_blocks, 1)  # 父序列还占着首页
        self.assertTrue(cache.free(1))
        self.assertEqual(cache.used_blocks, 0)
        self._check_invariant(cache)

    def test_cow_capacity_failure_keeps_state(self):
        cache = _cache(num_blocks=3, block_size=2)
        cache.allocate(1)
        keys, _ = _append(cache, 1, 3)  # 2 页，最后 1 页只有 1 个 slot
        cache.fork(1, 2)
        self.assertEqual(cache.free_blocks, 1)
        table_before = cache.block_table(2)
        shared_before = cache.shared_blocks
        # 拆尾页 1 个 + 新增 1 个 = 2 页，空闲只有 1 个。
        with self.assertRaises(PagedCacheFullError):
            _append(cache, 2, 2)
        self.assertEqual(cache.length(2), 3)
        self.assertEqual(cache.block_table(2), table_before)
        self.assertEqual(cache.free_blocks, 1)
        self.assertEqual(cache.shared_blocks, shared_before)
        k, _ = cache.gather(1, 0)
        np.testing.assert_array_equal(k, keys[0])
        self._check_invariant(cache)

    def test_swap_out_of_shared_sequence_frees_nothing(self):
        cache = _cache(num_blocks=4, block_size=2)
        cache.allocate(1)
        keys, _ = _append(cache, 1, 4)
        cache.fork(1, 2)
        self.assertEqual(cache.swap_out(1), 0)  # 全是共享页，一页也没腾出
        self.assertEqual(cache.used_blocks, 2)
        self.assertTrue(cache.is_swapped(1))
        self.assertEqual(cache.exclusive_blocks(1), 0)
        k2, _ = cache.gather(2, 0)
        np.testing.assert_array_equal(k2, keys[0])
        self.assertEqual(cache.swap_in(1), 2)  # 换入领新页，不再共享
        self.assertEqual(cache.shared_blocks, 0)
        k1, _ = cache.gather(1, 0)
        np.testing.assert_array_equal(k1, keys[0])
        self._check_invariant(cache)

    def test_fork_chain_shares_a_single_set_of_pages(self):
        cache = _cache(num_blocks=4, block_size=2)
        cache.allocate(1)
        _append(cache, 1, 2)
        cache.fork(1, 2)
        cache.fork(2, 3)
        self.assertEqual(cache.used_blocks, 1)
        self.assertEqual(cache.shared_blocks, 1)
        self.assertEqual(cache.stats()["forks"], 2)
        _append(cache, 2, 2)  # 只有 2 追加到新页，不动共享页
        self.assertEqual(cache.used_blocks, 2)
        self.assertEqual(cache.length(1), 2)
        self.assertEqual(cache.length(3), 2)
        self._check_invariant(cache)

    def test_fork_rejections(self):
        cache = _cache()
        cache.allocate(1)
        with self.assertRaises(KeyError):
            cache.fork(9, 2)
        cache.allocate(3)
        with self.assertRaisesRegex(ValueError, "已经分配"):
            cache.fork(1, 3)
        with self.assertRaises(TypeError):
            cache.fork(1, True)
        _append(cache, 1, 2)
        cache.swap_out(1)
        with self.assertRaisesRegex(ValueError, "先 swap_in"):
            cache.fork(1, 2)


class TestManagerFork(unittest.TestCase):
    def test_manager_forwards_fork_and_evicts_for_cow(self):
        cache = _cache(num_blocks=4, block_size=2)
        mgr = PagedCacheManager(cache)
        mgr.allocate(1)
        _append(mgr, 1, 3)  # 父：2 页，尾页只有 1 个 slot
        mgr.fork(1, 2)
        mgr.allocate(3)
        _append(mgr, 3, 4)  # 占满剩下的 2 页
        self.assertEqual(cache.free_blocks, 0)
        self.assertEqual(mgr.stats()["forks"], 1)
        self.assertEqual(mgr.stats()["shared_blocks"], 2)
        # 子序列写共享尾页需要 1 个新页：LRU 里父序列全是共享页（换出腾不出），
        # 序列 3 独占两页，于是它被换出。
        extra, _ = _append(mgr, 2, 1, seed=5)
        self.assertEqual(mgr.length(2), 4)
        self.assertTrue(cache.is_swapped(3))
        self.assertEqual(mgr.stats()["swaps_out"], 1)
        self.assertEqual(mgr.length(1), 3)
        self.assertEqual(cache.shared_blocks, 1)
        k2, _ = cache.gather(2, 0)
        np.testing.assert_array_equal(k2[3:], extra[0])


if __name__ == "__main__":
    unittest.main()
