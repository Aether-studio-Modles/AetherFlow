"""ZeRO-3 接进训练循环：逐段拼回参数、算完即释放、只更本块。

判据与 5.233 的分片容器一致，只是换成了"真的在训"：同一张图、同一批数据、
同一组超参下，两个 rank 的 ZeRO-3 训练与单份参数的分段训练**逐位**对拍
（每个 rank 喂同一批数据，ReduceScatter 求和再除以 2 是精确运算）。

另外三条结构性判据：

1. 每段只借这一段的参数，不借整份；一步跑完缓存必须为空；
2. ``store`` 里的参数是借出的旧副本，``sync_to_host()`` 之后才等于分片；
3. 拒绝面：没有分段、设备态、master weights、GradScaler、优化器不是分片
   优化器、参数集合 / 形状 / dtype 不一致，全部在构造期报错。
"""

import os
import tempfile
import threading
import unittest

import numpy as np

from aetherflow import Graph, f32
from aetherflow.train import (
    SGD,
    GradScaler,
    LoopbackWorld,
    ParameterStore,
    Trainer,
    Zero3Rank,
)

_WIDTH = 8
_DEPTH = 4


def _chain(depth=_DEPTH, width=_WIDTH, batch=4):
    """一条链式 MLP：每层权重只被下一层消费，切段后参数天然分开。"""
    g = Graph("mlp")
    x = g.input("x", f32[batch, width])
    h = x
    for index in range(depth):
        w = g.input(f"w{index}", f32[width, width])
        h = g.relu(g.matmul(h, w))
    w = g.input("wout", f32[width, 1])
    g.output(g.reduce_mean(g.reduce_mean(g.matmul(h, w))), "loss")
    return g


def _params(seed=3, depth=_DEPTH, width=_WIDTH):
    rng = np.random.default_rng(seed)
    out = {
        f"w{index}": rng.standard_normal((width, width)).astype(np.float32)
        for index in range(depth)
    }
    out["wout"] = rng.standard_normal((width, 1)).astype(np.float32)
    return out


def _batch(rng, batch=4, width=_WIDTH):
    return {"x": rng.standard_normal((batch, width)).astype(np.float32)}


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


def _ranks(world_size=2, lr=0.05, momentum=0.9):
    """每个 rank 一套 Zero3Rank，初值相同（各自切自己的块）。"""
    params = _params()
    world = LoopbackWorld(world_size)
    return [
        Zero3Rank(
            {name: value.copy() for name, value in params.items()},
            lambda sub, lr=lr, momentum=momentum: SGD(
                sub, lr=lr, momentum=momentum
            ),
            rank=rank,
            world_size=world_size,
            collective=world.collective(rank),
        )
        for rank in range(world_size)
    ]


def _trainers(graph, ranks, **kwargs):
    """每个 rank 一个 Trainer：``store`` 只是借出的完整副本，优化器是分片那个。"""
    return [
        Trainer(
            graph,
            ParameterStore({name: value.copy() for name, value in _params().items()}),
            rank.optimizer,
            segments=2,
            zero3=rank,
            **kwargs,
        )
        for rank in ranks
    ]


def _plain(graph, lr=0.05, momentum=0.9):
    """整份参数的对照组：同样走两段，保证反向的分段顺序与 ZeRO-3 一致。"""
    store = ParameterStore({name: value.copy() for name, value in _params().items()})
    return Trainer(
        graph,
        store,
        SGD(store.arrays(), lr=lr, momentum=momentum),
        segments=2,
    ), store


def _step_all(trainers, batch):
    losses = [None] * len(trainers)
    _parallel(
        [
            lambda index=index: losses.__setitem__(index, trainers[index].step(batch))
            for index in range(len(trainers))
        ]
    )
    return losses


def _materialize_all(ranks):
    seen = [None] * len(ranks)
    _parallel(
        [
            lambda index=index: seen.__setitem__(index, ranks[index].materialize())
            for index in range(len(ranks))
        ]
    )
    return seen


class Zero3TrainerEquivalenceTest(unittest.TestCase):
    def test_two_ranks_match_plain_trainer(self):
        graph = _chain()
        ranks = _ranks()
        trainers = _trainers(graph, ranks)
        plain, plain_store = _plain(graph)
        rng = np.random.default_rng(11)
        for _ in range(2):
            batch = _batch(rng)
            want = plain.step(batch)
            for loss in _step_all(trainers, batch):
                self.assertEqual(loss, want)
        seen = _materialize_all(ranks)
        for name in plain_store.names:
            np.testing.assert_array_equal(seen[0][name], plain_store[name])
            np.testing.assert_array_equal(seen[1][name], plain_store[name])

    def test_each_step_only_borrows_the_segment_params(self):
        graph = _chain()
        ranks = _ranks()
        trainers = _trainers(graph, ranks)
        calls = []
        original = ranks[0].materialize

        def tracked(names=None):
            calls.append(None if names is None else tuple(names))
            return original(names)

        ranks[0].materialize = tracked
        _step_all(trainers, _batch(np.random.default_rng(41)))
        self.assertTrue(calls)
        self.assertNotIn(None, calls, "训练期间不该整份拼回参数")
        for names in calls:
            self.assertLess(len(names), len(ranks[0].param_names))
        self.assertEqual(len(ranks[0].resident), 0, "一步跑完缓存必须为空")

    def test_store_is_stale_until_sync_to_host(self):
        graph = _chain()
        initial = _params()
        ranks = _ranks()
        trainers = _trainers(graph, ranks)
        _step_all(trainers, _batch(np.random.default_rng(51)))
        for name in initial:
            np.testing.assert_array_equal(trainers[0].store[name], initial[name])
        self.assertFalse(trainers[0].training_state().host_in_sync)
        _parallel(
            [lambda index=index: trainers[index].sync_to_host() for index in range(2)]
        )
        seen = _materialize_all(ranks)
        for name in initial:
            np.testing.assert_array_equal(
                trainers[0].store[name], seen[0][name]
            )
        self.assertTrue(trainers[0].training_state().host_in_sync)

    def test_accumulation_and_grad_hook_still_apply(self):
        graph = _chain()
        ranks = _ranks()
        calls = []

        def hook(grads):
            calls.append(len(grads))
            return grads

        trainers = _trainers(
            graph, ranks, accumulate_steps=2, grad_hook=hook
        )
        batch = _batch(np.random.default_rng(31))
        _step_all(trainers, batch)
        self.assertEqual(trainers[0].optimizer.step_count, 0)
        _step_all(trainers, batch)
        self.assertEqual(trainers[0].optimizer.step_count, 1)
        self.assertEqual(len(calls), 2, "每个 rank 各调一次钩子")
        self.assertEqual(trainers[0].steps, 2)

    def test_resume_from_shards_matches_uninterrupted(self):
        graph = _chain()
        first = _batch(np.random.default_rng(21))
        second = _batch(np.random.default_rng(22))

        ranks = _ranks()
        trainers = _trainers(graph, ranks)
        for batch in (first, first, second, second):
            _step_all(trainers, batch)
        want = _materialize_all(ranks)

        part_ranks = _ranks()
        part_trainers = _trainers(graph, part_ranks)
        for batch in (first, first):
            _step_all(part_trainers, batch)
        with tempfile.TemporaryDirectory() as tmp:
            paths = [os.path.join(tmp, f"rank{rank}.npz") for rank in range(2)]
            _parallel(
                [
                    lambda index=index: part_ranks[index].save_shard(paths[index])
                    for index in range(2)
                ]
            )
            resumed_ranks = _ranks()
            resumed = _trainers(graph, resumed_ranks)
            _parallel(
                [
                    lambda index=index: resumed_ranks[index].load_shard(paths[index])
                    for index in range(2)
                ]
            )
            for batch in (second, second):
                _step_all(resumed, batch)
        got = _materialize_all(resumed_ranks)
        for name in want[0]:
            np.testing.assert_array_equal(got[0][name], want[0][name])
            np.testing.assert_array_equal(got[1][name], want[0][name])
        self.assertEqual(resumed_ranks[0].steps, 4, "分片存档要带回优化器步数")


class Zero3TrainerValidationTest(unittest.TestCase):
    def _make(self, exc, **kwargs):
        graph = _chain()
        rank = _ranks(world_size=1)[0]
        store = ParameterStore(
            {name: value.copy() for name, value in _params().items()}
        )
        options = {"segments": 2, "zero3": rank}
        options.update(kwargs)
        optimizer = options.pop("optimizer", rank.optimizer)
        with self.assertRaises(exc):
            Trainer(graph, store, optimizer, **options)

    def test_requires_segments(self):
        self._make(ValueError, segments=None)

    def test_rejects_device_state(self):
        self._make(ValueError, device_state=True)

    def test_rejects_master_weights(self):
        self._make(ValueError, master_weights=True, mixed_precision="f16")

    def test_rejects_grad_scaler(self):
        self._make(ValueError, scaler=GradScaler())

    def test_rejects_plain_optimizer(self):
        graph = _chain()
        rank = _ranks(world_size=1)[0]
        store = ParameterStore(
            {name: value.copy() for name, value in _params().items()}
        )
        with self.assertRaises(ValueError):
            Trainer(
                graph,
                store,
                SGD(store.arrays(), lr=0.1),
                segments=2,
                zero3=rank,
            )

    def test_rejects_mismatched_param_set(self):
        graph = _chain()
        rank = _ranks(world_size=1)[0]
        partial = _params()
        partial.pop("wout")
        store = ParameterStore(partial)
        with self.assertRaises(KeyError):
            Trainer(graph, store, rank.optimizer, segments=2, zero3=rank)

    def test_rejects_mismatched_shape(self):
        graph = _chain()
        rank = _ranks(world_size=1)[0]
        params = _params()
        params["w0"] = np.zeros((8, 3), dtype=np.float32)
        store = ParameterStore(params)
        with self.assertRaises(ValueError):
            Trainer(graph, store, rank.optimizer, segments=2, zero3=rank)

    def test_rejects_mismatched_dtype(self):
        graph = _chain()
        rank = _ranks(world_size=1)[0]
        params = _params()
        params["w0"] = params["w0"].astype(np.float64)
        store = ParameterStore(params)
        with self.assertRaises(TypeError):
            Trainer(graph, store, rank.optimizer, segments=2, zero3=rank)


if __name__ == "__main__":
    unittest.main()
