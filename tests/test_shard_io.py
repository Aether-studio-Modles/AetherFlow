"""分布式存档的合并与重分片：N 个分片 ↔ 一份完整状态。

四条判据：

1. **合并是对的**：合出来的槽位逐位等于各片原值，写出完整 checkpoint 后
   ``load_checkpoint`` 能读回、参数与步数都对；
2. **合并来得及续训**：合并后的单份状态再走一步，与多 rank 再走一步逐位相等；
3. **换 world_size 接得上**：2 份 → 4 份（以及 2 → 3）后装回新实例，参数拼得
   回来、槽位落在新归属上；
4. **不猜**：缺片、重复片、步数/超参/布局不一致、块布局不一致、拿错权重，全部
   显式报错。

进程内用 ``LoopbackWorld`` 让多个 rank 并发跑集合调用；合并不走通信，直接读
文件，所以"分片写盘"与"合并读盘"之间没有任何时序耦合。
"""

import copy
import json
import os
import tempfile
import threading
import unittest

import numpy as np

from aetherflow.train import (
    SGD,
    LoopbackWorld,
    ParameterStore,
    ShardedCheckpoint,
    Zero2Rank,
    Zero3Rank,
    ZeroRank,
    load_checkpoint,
    merge_shards,
    save_checkpoint,
    shard_params,
    split_checkpoint,
)

_SHAPES = {"w1": (4, 3), "b1": (3,), "w2": (2, 4), "b2": (2,)}
_META_KEY = "__meta__"


def _params(seed=3):
    rng = np.random.default_rng(seed)
    return {
        name: rng.standard_normal(shape).astype(np.float32)
        for name, shape in _SHAPES.items()
    }


def _grads(rank, step=0):
    rng = np.random.default_rng(100 + 10 * step + rank)
    return {
        name: rng.standard_normal(shape).astype(np.float32)
        for name, shape in _SHAPES.items()
    }


def _factory(params):
    return SGD(params, lr=0.1, momentum=0.9)


def _parallel(funcs):
    """让若干 rank 并发执行各自的闭包（集合调用会互相等待）。"""
    errors = []

    def target(index):
        try:
            funcs[index]()
        except BaseException as exc:  # noqa: BLE001
            errors.append((index, exc))

    threads = [
        threading.Thread(target=target, args=(index,)) for index in range(len(funcs))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0][1]


def _build(kind, world_size):
    """造一组 rank（``zero1`` / ``zero2`` / ``zero3``）与它们各自的参数表。"""
    world = LoopbackWorld(world_size)
    params_list = [_params() for _ in range(world_size)]
    cls = {"zero1": ZeroRank, "zero2": Zero2Rank, "zero3": Zero3Rank}[kind]
    ranks = [
        cls(
            params_list[rank],
            _factory,
            rank=rank,
            world_size=world_size,
            collective=world.collective(rank),
        )
        for rank in range(world_size)
    ]
    return ranks, params_list


def _train(ranks, step=0):
    _parallel(
        [
            lambda index=index: ranks[index].step(_grads(index, step))
            for index in range(len(ranks))
        ]
    )


def _materialize_all(ranks):
    seen = [None] * len(ranks)
    _parallel(
        [
            lambda index=index: seen.__setitem__(index, ranks[index].materialize())
            for index in range(len(ranks))
        ]
    )
    return seen


def _save_all(ranks, folder, prefix="rank"):
    return [
        ranks[rank].save_shard(os.path.join(folder, f"{prefix}{rank}.npz"))
        for rank in range(len(ranks))
    ]


def _mean_grads(world_size, step=0):
    total = _grads(0, step)
    for rank in range(1, world_size):
        for name in total:
            total[name] = total[name] + _grads(rank, step)[name]
    return {name: array / float(world_size) for name, array in total.items()}


class MergeNamesTest(unittest.TestCase):
    """按参数名分片（ZeRO-1 / ZeRO-2）的合并。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_zero1_merge_is_a_loadable_checkpoint(self):
        ranks, params_list = _build("zero1", 2)
        _train(ranks, 0)
        _train(ranks, 1)
        paths = _save_all(ranks, self._tmp.name)

        merged = merge_shards(paths, params=params_list[0])
        self.assertIsInstance(merged, ShardedCheckpoint)
        self.assertEqual(merged.step, 2)

        # 每个参数只在一个 rank 的优化器里：合并出来的槽位必须就是那一份。
        for name, owner in merged.assignments.items():
            saved = ranks[owner].optimizer.state_dict()["slots"]["buf"][name]
            np.testing.assert_array_equal(merged.slots["buf"][name], saved)

        target = os.path.join(self._tmp.name, "full.npz")
        merged.save(target)
        store = ParameterStore(
            {name: np.zeros_like(array) for name, array in params_list[0].items()}
        )
        optimizer = SGD(store.arrays(), lr=0.1, momentum=0.9)
        load_checkpoint(target, store, optimizer)
        for name in params_list[0]:
            np.testing.assert_array_equal(store[name], params_list[0][name])
        self.assertEqual(optimizer.step_count, 2)

    def test_zero2_round_trip_keeps_training_identical(self):
        ranks, params_list = _build("zero2", 2)
        _train(ranks, 0)
        paths = _save_all(ranks, self._tmp.name)
        merged = merge_shards(paths, params=params_list[0])

        # 原样写回两份，装进新实例；这一步同时验证"同 world_size 不必给参数"。
        again = merged.write_shards(os.path.join(self._tmp.name, "again{rank}.npz"))
        self.assertEqual(len(again), 2)
        # 新世界必须共用同一个 LoopbackWorld，否则集合调用对不上。
        world = LoopbackWorld(2)
        fresh = [
            Zero2Rank(
                copy.deepcopy(params_list[rank]),
                _factory,
                rank=rank,
                world_size=2,
                collective=world.collective(rank),
            )
            for rank in range(2)
        ]
        for rank in range(2):
            fresh[rank].load_shard(again[rank])
            np.testing.assert_array_equal(fresh[rank].params["w1"], ranks[rank].params["w1"])

        _train(ranks, 1)
        _train(fresh, 1)
        for rank in range(2):
            for name in _SHAPES:
                np.testing.assert_array_equal(
                    fresh[rank].params[name], ranks[rank].params[name]
                )

    def test_zero1_merge_without_params_cannot_be_saved(self):
        ranks, _ = _build("zero1", 2)
        _train(ranks, 0)
        merged = merge_shards(_save_all(ranks, self._tmp.name))
        self.assertEqual(dict(merged.params), {})
        with self.assertRaises(ValueError) as caught:
            merged.save(os.path.join(self._tmp.name, "full.npz"))
        self.assertIn("params=", str(caught.exception))


class MergeNamesValidationTest(unittest.TestCase):
    """合并的拒绝面：不齐、重复、不同轨迹、不同布局。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_missing_rank_is_refused(self):
        ranks, _ = _build("zero1", 2)
        paths = _save_all(ranks, self._tmp.name)
        with self.assertRaises(ValueError) as caught:
            merge_shards(paths[:1])
        self.assertIn("不齐", str(caught.exception))

    def test_duplicate_rank_is_refused(self):
        ranks, _ = _build("zero1", 2)
        paths = _save_all(ranks, self._tmp.name)
        with self.assertRaises(ValueError) as caught:
            merge_shards([paths[0], paths[0]])
        self.assertIn("重复", str(caught.exception))

    def test_mixed_world_size_is_refused(self):
        two, _ = _build("zero1", 2)
        three, _ = _build("zero1", 3)
        paths = _save_all(two, self._tmp.name, "a") + _save_all(three, self._tmp.name, "b")
        with self.assertRaises(ValueError) as caught:
            merge_shards(paths)
        self.assertIn("world_size", str(caught.exception))

    def test_mixed_step_is_refused(self):
        ranks, _ = _build("zero1", 2)
        _train(ranks, 0)
        paths = _save_all(ranks, self._tmp.name)
        _train(ranks, 1)
        paths[1] = ranks[1].save_shard(os.path.join(self._tmp.name, "rank1b.npz"))
        with self.assertRaises(ValueError) as caught:
            merge_shards(paths)
        self.assertIn("步数", str(caught.exception))

    def test_mixed_layout_is_refused(self):
        ranks, _ = _build("zero1", 2)
        paths = _save_all(ranks, self._tmp.name)

        # 把 rank1 的归属表换成块布局：其余字段保持一致，只有布局对不上。
        with np.load(paths[1], allow_pickle=False) as data:
            arrays = {key: np.array(data[key], copy=True) for key in data.files}
        meta = json.loads(str(arrays[_META_KEY]))
        meta.pop("assignments")
        meta["plan"] = {name: [list(shape), 1] for name, shape in _SHAPES.items()}
        arrays[_META_KEY] = np.array(json.dumps(meta, ensure_ascii=False))
        mixed = os.path.join(self._tmp.name, "mixed.npz")
        np.savez(mixed, **arrays)

        with self.assertRaises(ValueError) as caught:
            merge_shards([paths[0], mixed])
        self.assertIn("布局", str(caught.exception))

    def test_plain_checkpoint_is_not_a_shard(self):
        params = _params()
        store = ParameterStore(params)
        target = os.path.join(self._tmp.name, "plain.npz")
        save_checkpoint(target, store)
        with self.assertRaises(ValueError) as caught:
            merge_shards([target])
        self.assertIn("不是分片存档", str(caught.exception))


class ReshardNamesTest(unittest.TestCase):
    """按名布局换 world_size：归属重新按字节均衡划分。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_two_to_four_ranks_lands_on_the_new_assignments(self):
        ranks, params_list = _build("zero1", 2)
        _train(ranks, 0)
        merged = merge_shards(_save_all(ranks, self._tmp.name), params=params_list[0])
        paths = merged.write_shards(os.path.join(self._tmp.name, "w4_{rank}.npz"), 4)
        self.assertEqual(len(paths), 4)

        world = LoopbackWorld(4)
        fresh = [
            ZeroRank(
                copy.deepcopy(params_list[0]),
                _factory,
                rank=rank,
                world_size=4,
                collective=world.collective(rank),
            )
            for rank in range(4)
        ]
        for rank in range(4):
            fresh[rank].load_shard(paths[rank])

        buckets = shard_params(params_list[0], 4)
        for rank, bucket in enumerate(buckets):
            self.assertEqual(set(fresh[rank].my_names), set(bucket))
            for name in bucket:
                np.testing.assert_array_equal(
                    fresh[rank].optimizer.state_dict()["slots"]["buf"][name],
                    merged.slots["buf"][name],
                )

    def test_new_world_size_without_params_is_refused(self):
        ranks, _ = _build("zero1", 2)
        _train(ranks, 0)
        merged = merge_shards(_save_all(ranks, self._tmp.name))
        with self.assertRaises(ValueError) as caught:
            merged.split(4)
        self.assertIn("params=", str(caught.exception))

    def test_same_world_size_reuses_the_saved_assignments(self):
        ranks, params_list = _build("zero1", 2)
        _train(ranks, 0)
        merged = merge_shards(_save_all(ranks, self._tmp.name))
        paths = merged.write_shards(os.path.join(self._tmp.name, "same{rank}.npz"))
        world = LoopbackWorld(2)
        fresh = [
            ZeroRank(
                copy.deepcopy(params_list[rank]),
                _factory,
                rank=rank,
                world_size=2,
                collective=world.collective(rank),
            )
            for rank in range(2)
        ]
        for rank in range(2):
            fresh[rank].load_shard(paths[rank])
        self.assertEqual(set(fresh[0].my_names), set(ranks[0].my_names))
        for rank in range(2):
            np.testing.assert_array_equal(
                fresh[rank].optimizer.state_dict()["slots"]["buf"][
                    fresh[rank].my_names[0]
                ],
                ranks[rank].optimizer.state_dict()["slots"]["buf"][
                    ranks[rank].my_names[0]
                ],
            )


class MergeElementsTest(unittest.TestCase):
    """按元素分片（ZeRO-3）的合并与重切。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_merge_restores_the_materialized_params(self):
        ranks, _ = _build("zero3", 2)
        _train(ranks, 0)
        _train(ranks, 1)
        seen = _materialize_all(ranks)
        merged = merge_shards(_save_all(ranks, self._tmp.name))
        for name in _SHAPES:
            np.testing.assert_array_equal(merged.params[name], seen[0][name])
            np.testing.assert_array_equal(merged.params[name], seen[1][name])

    def test_merged_state_continues_bitwise_on_one_process(self):
        ranks, _ = _build("zero3", 2)
        _train(ranks, 0)
        merged = merge_shards(_save_all(ranks, self._tmp.name))
        target = os.path.join(self._tmp.name, "full.npz")
        merged.save(target)

        store = ParameterStore(
            {name: np.zeros_like(array) for name, array in merged.params.items()}
        )
        optimizer = SGD(store.arrays(), lr=0.1, momentum=0.9)
        load_checkpoint(target, store, optimizer)
        optimizer.step(_mean_grads(2, step=1))

        _train(ranks, 1)
        seen = _materialize_all(ranks)
        for name in _SHAPES:
            np.testing.assert_array_equal(store[name], seen[0][name])

    def test_two_to_three_ranks_keeps_params_and_updates(self):
        ranks, _ = _build("zero3", 2)
        _train(ranks, 0)
        merged = merge_shards(_save_all(ranks, self._tmp.name))
        paths = merged.write_shards(os.path.join(self._tmp.name, "w3_{rank}.npz"), 3)

        world = LoopbackWorld(3)
        fresh = [
            Zero3Rank(
                _params(seed=rank + 10),  # 构造用的完整参数随后被 load_shard 覆盖
                _factory,
                rank=rank,
                world_size=3,
                collective=world.collective(rank),
            )
            for rank in range(3)
        ]
        for rank in range(3):
            fresh[rank].load_shard(paths[rank])
        seen = _materialize_all(fresh)
        for name in _SHAPES:
            np.testing.assert_array_equal(seen[0][name], merged.params[name])

        _parallel(
            [
                lambda index=index: fresh[index].step(_grads(index, 1))
                for index in range(3)
            ]
        )
        after = merge_shards(_save_all(fresh, self._tmp.name, "after"))
        store = ParameterStore(
            {name: np.array(array, copy=True) for name, array in merged.params.items()}
        )
        optimizer = SGD(store.arrays(), lr=0.1, momentum=0.9)
        optimizer.load_state_dict(
            {"step": merged.step, "hyper": merged.hyper, "slots": merged.slots}
        )
        optimizer.step(_mean_grads(3, step=1))
        for name in _SHAPES:
            np.testing.assert_array_equal(after.params[name], store[name])


class MergeElementsValidationTest(unittest.TestCase):
    """元素布局的拒绝面。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _rewrite(self, path, mutate):
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: np.array(data[key], copy=True) for key in data.files}
        mutate(arrays)
        out = path.replace(".npz", "-bad.npz")
        np.savez(out, **arrays)
        return out

    def test_short_piece_is_refused(self):
        ranks, _ = _build("zero3", 2)
        paths = _save_all(ranks, self._tmp.name)

        def shorten(arrays):
            arrays["param::w1"] = arrays["param::w1"][:-1]

        bad = self._rewrite(paths[1], shorten)
        with self.assertRaises(ValueError) as caught:
            merge_shards([paths[0], bad])
        self.assertIn("块长", str(caught.exception))

    def test_plan_mismatch_is_refused(self):
        ranks, _ = _build("zero3", 2)
        paths = _save_all(ranks, self._tmp.name)

        def bump_chunk(arrays):
            meta = json.loads(str(arrays[_META_KEY]))
            meta["plan"]["w1"][1] += 1
            arrays[_META_KEY] = np.array(json.dumps(meta, ensure_ascii=False))

        bad = self._rewrite(paths[1], bump_chunk)
        with self.assertRaises(ValueError) as caught:
            merge_shards([paths[0], bad])
        self.assertIn("块布局", str(caught.exception))

    def test_given_params_must_match_the_shards(self):
        ranks, _ = _build("zero3", 2)
        _train(ranks, 0)
        wrong = _params(seed=99)
        with self.assertRaises(ValueError) as caught:
            merge_shards(_save_all(ranks, self._tmp.name), params=wrong)
        self.assertIn("另一份权重", str(caught.exception))


class SplitFullCheckpointTest(unittest.TestCase):
    """一份完整 checkpoint → 每个 rank 一个分片文件。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _full(self):
        params = _params()
        store = ParameterStore(params)
        optimizer = SGD(store.arrays(), lr=0.1, momentum=0.9)
        optimizer.step(_grads(0, 0))
        target = os.path.join(self._tmp.name, "full.npz")
        save_checkpoint(target, store, optimizer)
        return target, params, optimizer

    def test_split_into_elements_is_loadable_by_zero3(self):
        target, params, _ = self._full()
        paths = split_checkpoint(
            target, 3, os.path.join(self._tmp.name, "e{rank}.npz"), layout="elements"
        )
        world = LoopbackWorld(3)
        ranks = [
            Zero3Rank(
                copy.deepcopy(params),
                _factory,
                rank=rank,
                world_size=3,
                collective=world.collective(rank),
            )
            for rank in range(3)
        ]
        for rank in range(3):
            ranks[rank].load_shard(paths[rank])
        seen = _materialize_all(ranks)
        for name in _SHAPES:
            np.testing.assert_array_equal(seen[0][name], params[name])

    def test_split_into_names_is_loadable_by_zero1(self):
        target, params, optimizer = self._full()
        paths = split_checkpoint(
            target, 3, os.path.join(self._tmp.name, "n{rank}.npz"), layout="names"
        )
        world = LoopbackWorld(3)
        ranks = [
            ZeroRank(
                copy.deepcopy(params),
                _factory,
                rank=rank,
                world_size=3,
                collective=world.collective(rank),
            )
            for rank in range(3)
        ]
        for rank in range(3):
            ranks[rank].load_shard(paths[rank])
        buckets = shard_params(params, 3)
        whole = optimizer.state_dict()["slots"]["buf"]
        for rank, bucket in enumerate(buckets):
            self.assertEqual(set(ranks[rank].my_names), set(bucket))
            for name in bucket:
                np.testing.assert_array_equal(
                    ranks[rank].optimizer.state_dict()["slots"]["buf"][name],
                    whole[name],
                )

    def test_pattern_without_rank_is_refused(self):
        target, _, _ = self._full()
        with self.assertRaises(ValueError) as caught:
            split_checkpoint(target, 2, os.path.join(self._tmp.name, "same.npz"))
        self.assertIn("{rank}", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
