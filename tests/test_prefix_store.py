"""5.337 落盘 / 跨进程前缀缓存：存储层往返与生成侧接线。

判据分两层：存储层"存进去、换个进程读回来逐位一致"，生成层"盘命中只省
prefill、不改变 token"——后者的参照是不共享任何缓存的稠密图贪心。
"""

import math
import os
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

from aetherflow.models import PagedGenerator
from aetherflow.runtime import DiskPrefixCache
from tests.test_attention_cache import (
    BACKEND,
    _dense_greedy,
    _vocab_params,
    _vocab_spec,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestDiskPrefixCache(unittest.TestCase):
    """存储层：查询只认严格前缀，失配与损坏都退回"没有条目"。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = self.tmp.name

    def _cache(self, **overrides):
        options = dict(
            model_id="demo",
            block_size=2,
            num_layers=2,
            num_heads=2,
            head_dim=2,
        )
        options.update(overrides)
        return DiskPrefixCache(self.folder, **options)

    def _payload(self, tokens, seed=0):
        """每层一份 ``(T, 2, 2)``，层间数值不同，便于逐层比对。"""
        rng = np.random.default_rng(seed)
        shape = (len(tokens), 2, 2)
        return (
            [rng.standard_normal(shape).astype(np.float32) for _ in range(2)],
            [rng.standard_normal(shape).astype(np.float32) for _ in range(2)],
        )

    def _store(self, cache, tokens, seed=0):
        keys, values = self._payload(tokens, seed)
        cache.store(tokens, keys=keys, values=values)
        return keys, values

    def test_store_and_lookup_roundtrip(self):
        cache = self._cache()
        keys, values = self._store(cache, [1, 4, 7])
        length, payload = cache.lookup([1, 4, 7, 2])
        self.assertEqual(length, 3)
        self.assertEqual(payload["tokens"], [1, 4, 7])
        for layer in range(2):
            np.testing.assert_array_equal(payload["keys"][layer], keys[layer])
            np.testing.assert_array_equal(payload["values"][layer], values[layer])
        info = cache.stats()
        self.assertEqual(info["entries"], 1)
        self.assertEqual(info["stores"], 1)
        self.assertEqual(info["hits"], 1)
        self.assertEqual(info["misses"], 0)
        self.assertEqual(info["corrupt"], 0)

    def test_equal_length_is_not_a_hit(self):
        cache = self._cache()
        self._store(cache, [1, 4, 7])
        self.assertIsNone(cache.lookup([1, 4, 7]))
        info = cache.stats()
        self.assertEqual(info["hits"], 0)
        self.assertEqual(info["misses"], 1)

    def test_longest_strict_prefix_wins(self):
        cache = self._cache()
        self._store(cache, [1, 4], seed=1)
        self._store(cache, [1, 4, 7], seed=2)
        length, payload = cache.lookup([1, 4, 7, 9])
        self.assertEqual(length, 3)
        self.assertEqual(payload["tokens"], [1, 4, 7])

    def test_new_instance_scans_existing_entries(self):
        self._store(self._cache(), [1, 4, 7])
        fresh = self._cache()
        self.assertEqual(len(fresh), 1)
        length, payload = fresh.lookup([1, 4, 7, 2])
        self.assertEqual(length, 3)
        self.assertEqual(payload["tokens"], [1, 4, 7])

    def test_cross_process_readback(self):
        code = (
            "import numpy as np;"
            "from aetherflow.runtime import DiskPrefixCache;"
            "cache = DiskPrefixCache(%r, model_id='demo', block_size=2, "
            "num_layers=2, num_heads=2, head_dim=2);"
            "keys = [np.full((3, 2, 2), float(i + 1), dtype=np.float32) "
            "for i in range(2)];"
            "values = [np.full((3, 2, 2), -1.0, dtype=np.float32) "
            "for _ in range(2)];"
            "cache.store([1, 4, 7], keys=keys, values=values);"
            "print('stored')" % self.folder
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = _ROOT + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "stored")

        cache = self._cache()
        length, payload = cache.lookup([1, 4, 7, 2])
        self.assertEqual(length, 3)
        for layer in range(2):
            np.testing.assert_array_equal(
                payload["keys"][layer],
                np.full((3, 2, 2), float(layer + 1), dtype=np.float32),
            )
            np.testing.assert_array_equal(
                payload["values"][layer],
                np.full((3, 2, 2), -1.0, dtype=np.float32),
            )

    def test_config_mismatch_is_skipped_not_raised(self):
        self._store(self._cache(), [1, 4, 7])
        for label, options in (
            ("model_id", {"model_id": "other"}),
            ("num_heads", {"num_heads": 3}),
            ("head_dim", {"head_dim": 4}),
            ("num_layers", {"num_layers": 1}),
            ("block_size", {"block_size": 4}),
        ):
            with self.subTest(field=label):
                other = self._cache(**options)
                self.assertIsNone(other.lookup([1, 4, 7, 2]))
                self.assertEqual(other.stats()["entries"], 0)
                self.assertEqual(other.stats()["corrupt"], 1)

    def test_broken_file_is_rejected(self):
        cache = self._cache()
        self._store(cache, [1, 4, 7])
        path = os.path.join(self.folder, os.listdir(self.folder)[0])
        with open(path, "wb") as handle:
            handle.write(b"not a zip")

        fresh = self._cache()
        self.assertEqual(fresh.stats()["entries"], 0)
        self.assertGreaterEqual(fresh.stats()["corrupt"], 1)
        self.assertIsNone(fresh.lookup([1, 4, 7, 2]))

        # 同一实例的索引里还留着这条：读取时才发现坏了，也要剔除。
        self.assertIsNone(cache.lookup([1, 4, 7, 2]))
        self.assertEqual(cache.stats()["entries"], 0)
        self.assertGreaterEqual(cache.stats()["corrupt"], 1)

    def test_tampered_tokens_are_rejected(self):
        cache = self._cache()
        self._store(cache, [1, 4, 7])
        path = os.path.join(self.folder, os.listdir(self.folder)[0])
        with np.load(path, allow_pickle=False) as data:
            meta = data["meta"]
            keys = data["keys"]
            values = data["values"]
        np.savez(
            path,
            meta=meta,
            tokens=np.asarray([9, 9, 9], dtype=np.int64),
            keys=keys,
            values=values,
        )
        fresh = self._cache()
        self.assertIsNone(fresh.lookup([1, 4, 7, 2]))
        self.assertEqual(fresh.stats()["entries"], 0)
        self.assertGreaterEqual(fresh.stats()["corrupt"], 1)

    def test_max_entries_evicts_oldest(self):
        cache = self._cache(max_entries=2)
        for index, tokens in enumerate(([1], [1, 4], [1, 4, 7])):
            self._store(cache, tokens, seed=index)
            time.sleep(0.01)
        self.assertEqual(len(cache), 2)
        self.assertIsNone(cache.lookup([1]))
        self.assertIsNotNone(cache.lookup([1, 4, 7]))
        self.assertGreaterEqual(cache.stats()["evictions"], 1)

    def test_max_bytes_evicts_until_fit(self):
        cache = self._cache(max_bytes=1)
        self._store(cache, [1, 4, 7])
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.stats()["bytes"], 0)
        self.assertEqual(cache.stats()["evictions"], 1)

    def test_clear_removes_entries(self):
        cache = self._cache()
        self._store(cache, [1], seed=1)
        self._store(cache, [1, 4], seed=2)
        self.assertEqual(cache.clear(), 2)
        self.assertEqual(len(cache), 0)
        self.assertEqual(os.listdir(self.folder), [])

    def test_store_rejects_bad_payloads(self):
        cache = self._cache()
        good = [np.zeros((1, 2, 2), dtype=np.float32)]
        with self.assertRaises(ValueError):
            cache.store([], keys=good * 2, values=good * 2)
        with self.assertRaises(ValueError):
            cache.store([1], keys=good, values=good)
        with self.assertRaises(ValueError):
            cache.store(
                [1],
                keys=[np.zeros((2, 2, 2), dtype=np.float32)] * 2,
                values=[np.zeros((2, 2, 2), dtype=np.float32)] * 2,
            )
        with self.assertRaises(TypeError):
            cache.store(
                [1],
                keys=[np.zeros((1, 2, 2), dtype=np.int32)] * 2,
                values=[np.zeros((1, 2, 2), dtype=np.int32)] * 2,
            )

    def test_same_key_is_not_rewritten(self):
        cache = self._cache()
        self._store(cache, [1, 4, 7])
        time.sleep(0.01)
        self._store(cache, [1, 4, 7])
        self.assertEqual(cache.stats()["stores"], 1)
        self.assertEqual(cache.stats()["entries"], 1)

    def test_atomic_write_leaves_no_temp_files(self):
        cache = self._cache()
        self._store(cache, [1, 4, 7])
        names = os.listdir(self.folder)
        self.assertEqual(len(names), 1)
        self.assertTrue(names[0].endswith(".npz"))

    def test_payload_is_a_copy(self):
        cache = self._cache()
        keys, _ = self._store(cache, [1, 4, 7])
        _, payload = cache.lookup([1, 4, 7, 2])
        payload["keys"][0][:] = 0.0
        _, again = cache.lookup([1, 4, 7, 2])
        np.testing.assert_array_equal(again["keys"][0], keys[0])


class TestGeneratorDiskCache(unittest.TestCase):
    """生成侧：盘命中只省 prefill，token 序列必须与稠密图逐 token 相同。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = self.tmp.name
        self.spec = _vocab_spec()
        self.params = _vocab_params(self.spec, seed=30)

    def _disk(self, **overrides):
        options = dict(
            model_id="tiny",
            block_size=2,
            num_layers=self.spec.num_layers,
            num_heads=self.spec.num_heads,
            head_dim=self.spec.head_dim,
        )
        options.update(overrides)
        return DiskPrefixCache(self.folder, **options)

    def _generator(self, disk, **overrides):
        options = dict(
            num_blocks=max(8, math.ceil(self.spec.seq / 2)),
            block_size=2,
            max_steps=self.spec.seq,
            disk_cache=disk,
        )
        options.update(overrides)
        return PagedGenerator(BACKEND, self.params, self.spec, **options)

    def test_new_instance_hits_disk_prefix(self):
        prompt = [1, 4, 7]
        first = self._generator(self._disk())
        first.generate(prompt, max_new_tokens=0, rid=1)
        self.assertEqual(first.stats()["disk_stores"], 1)

        second = self._generator(self._disk())
        longer = prompt + [2, 5]
        got = second.generate(longer, max_new_tokens=3, rid=1)
        self.assertEqual(got, _dense_greedy(self.params, self.spec, longer, 3))
        self.assertEqual(second.tokens(1), longer + got[:-1])
        info = second.stats()
        self.assertEqual(info["disk_hits"], 1)
        self.assertEqual(info["disk_reused_tokens"], len(prompt))
        self.assertEqual(info["prefix_hits"], 0)
        self.assertEqual(info["disk_stores"], 1)

    def test_equal_length_entry_is_not_reused(self):
        first = self._generator(self._disk())
        first.generate([1, 4, 7], max_new_tokens=0, rid=1)

        second = self._generator(self._disk())
        got = second.generate([1, 4, 7], max_new_tokens=1, rid=1)
        self.assertEqual(got, _dense_greedy(self.params, self.spec, [1, 4, 7], 1))
        info = second.stats()
        self.assertEqual(info["disk_hits"], 0)
        self.assertEqual(info["disk_misses"], 1)

    def test_disabled_prefix_cache_skips_disk(self):
        disk = self._disk()
        first = self._generator(disk)
        first.generate([1, 4, 7], max_new_tokens=0, rid=1)
        self.assertEqual(len(disk), 1)

        off = self._generator(disk, prefix_cache=False)
        got = off.generate([1, 4, 7, 2], max_new_tokens=1, rid=1)
        self.assertEqual(got, _dense_greedy(self.params, self.spec, [1, 4, 7, 2], 1))
        info = off.stats()
        self.assertEqual(info["disk_hits"], 0)
        self.assertEqual(info["disk_misses"], 0)
        self.assertEqual(info["disk_stores"], 0)
        self.assertEqual(len(disk), 1)

    def test_corrupt_entry_falls_back_to_prefill(self):
        first = self._generator(self._disk())
        first.generate([1, 4, 7], max_new_tokens=0, rid=1)
        for name in os.listdir(self.folder):
            with open(os.path.join(self.folder, name), "wb") as handle:
                handle.write(b"broken")

        second = self._generator(self._disk())
        prompt = [1, 4, 7, 2]
        got = second.generate(prompt, max_new_tokens=2, rid=1)
        self.assertEqual(got, _dense_greedy(self.params, self.spec, prompt, 2))
        info = second.stats()
        self.assertEqual(info["disk_hits"], 0)
        self.assertEqual(info["disk_misses"], 1)
        self.assertEqual(info["prefix_misses"], 1)

    def test_finish_persists_manual_sequence(self):
        """手工 feed / prefill 的序列在 finish 释放前落盘，并可被新实例复用。"""
        disk = self._disk()
        gen = self._generator(disk)
        prompt = [1, 4, 7]
        gen.submit(1)
        gen.prefill(1, prompt)
        self.assertEqual(len(disk), 0)
        gen.finish(1)
        self.assertEqual(gen.stats()["disk_stores"], 1)
        self.assertEqual(len(disk), 1)
        self.assertNotIn(1, gen)

        second = self._generator(self._disk())
        longer = prompt + [2]
        got = second.generate(longer, max_new_tokens=2, rid=1)
        self.assertEqual(got, _dense_greedy(self.params, self.spec, longer, 2))
        info = second.stats()
        self.assertEqual(info["disk_hits"], 1)
        self.assertEqual(info["disk_reused_tokens"], len(prompt))

    def test_finish_after_generate_does_not_rewrite(self):
        """generate 已落过盘，紧接着 finish 只刷新 mtime，不重复计落盘次数。"""
        disk = self._disk()
        gen = self._generator(disk)
        gen.generate([1, 4, 7], max_new_tokens=1, rid=1)
        self.assertEqual(gen.stats()["disk_stores"], 1)
        gen.finish(1)
        self.assertEqual(gen.stats()["disk_stores"], 1)
        self.assertEqual(len(disk), 1)

    def test_finish_skips_persist_when_swapped(self):
        """换出态 KV 不在池里：finish 跳过落盘，但仍能正常释放。"""
        disk = self._disk()
        gen = self._generator(disk)
        gen.submit(1)
        gen.prefill(1, [1, 4, 7])
        gen.suspend(1, evict=True)
        self.assertTrue(gen.is_swapped(1))
        gen.finish(1)
        self.assertEqual(gen.stats()["disk_stores"], 0)
        self.assertEqual(len(disk), 0)
        self.assertNotIn(1, gen)

    def test_finish_without_tokens_stores_nothing(self):
        disk = self._disk()
        gen = self._generator(disk)
        gen.submit(1)
        gen.finish(1)
        self.assertEqual(gen.stats()["disk_stores"], 0)
        self.assertEqual(len(disk), 0)

    def test_finish_respects_disabled_prefix_cache(self):
        disk = self._disk()
        gen = self._generator(disk, prefix_cache=False)
        gen.submit(1)
        gen.prefill(1, [1, 4, 7])
        gen.finish(1)
        self.assertEqual(gen.stats()["disk_stores"], 0)
        self.assertEqual(len(disk), 0)

    def test_mismatched_disk_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            self._generator(self._disk(num_heads=self.spec.num_heads + 1))
        with self.assertRaises(TypeError):
            self._generator("not a cache")


if __name__ == "__main__":
    unittest.main()
