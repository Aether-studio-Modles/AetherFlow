"""跨进程 ZeRO-3：参数分块驻留、拼回即释放、只更本块。

判据与 ZeRO-1/2 一致：分片只换布局，不换运算顺序，因此与整份 SGD **逐位**
相等。进程内的 ``LoopbackWorld`` 让多个 ``Zero3Rank`` 并发跑集合调用；拒绝面
（未接通信、梯度不齐、dtype 混用、装错片）全部在入口处显式报错。
"""

import json
import os
import tempfile
import threading
import unittest

import numpy as np

from aetherflow.train import SGD, LoopbackWorld, Zero3Rank
from aetherflow.train.zero3 import plan_chunks

_SHAPES = {"w1": (4, 3), "b1": (4,), "w2": (2, 4), "b2": (2,)}
#: 元素数不能被 world_size 整除的一组，用来检查补零不影响数值。
_RAGGED = {"w1": (4, 3), "b1": (3,), "w2": (2, 4), "b2": (2,)}


def _params(seed=3, shapes=None):
    rng = np.random.default_rng(seed)
    return {
        name: rng.standard_normal(shape).astype(np.float32)
        for name, shape in (shapes or _SHAPES).items()
    }


def _grads(rank, step=0, shapes=None):
    rng = np.random.default_rng(100 + 10 * step + rank)
    return {
        name: rng.standard_normal(shape).astype(np.float32)
        for name, shape in (shapes or _SHAPES).items()
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


def _build(world_size, shapes=None, collective=None, params_list=None):
    world = collective or LoopbackWorld(world_size)
    params_list = params_list or [_params(shapes=shapes) for _ in range(world_size)]
    instances = [
        Zero3Rank(
            params_list[rank],
            _factory,
            rank=rank,
            world_size=world_size,
            collective=world.collective(rank),
        )
        for rank in range(world_size)
    ]
    return world, instances


def _materialize_all(instances):
    """并发拼回完整参数，返回每个 rank 看到的那一份。"""
    seen = [None] * len(instances)
    _parallel(
        [
            lambda index=index: seen.__setitem__(
                index, instances[index].materialize()
            )
            for index in range(len(instances))
        ]
    )
    return seen


def _step_all(instances, grads_by_rank):
    _parallel(
        [
            lambda index=index: instances[index].step(grads_by_rank[index])
            for index in range(len(instances))
        ]
    )


def _full_reference(shapes, steps, world_size):
    """主机侧的"整份参数 + 整份 SGD"参照，梯度先按 rank 求均值。"""
    params = _params(shapes=shapes)
    optimizer = SGD(params, lr=0.1, momentum=0.9)
    for step in range(steps):
        grads = [
            _grads(rank, step, shapes=shapes) for rank in range(world_size)
        ]
        total = {name: np.array(array, copy=True) for name, array in grads[0].items()}
        for other in grads[1:]:
            for name in total:
                total[name] += other[name]
        optimizer.step(
            {name: array / float(world_size) for name, array in total.items()}
        )
    return params


class PlanChunksTest(unittest.TestCase):
    """块长只由元素数与 world_size 决定，与 rank 无关。"""

    def test_chunk_is_ceiling_division(self):
        plan = plan_chunks(_params(), 3)
        self.assertEqual(plan["w1"], ((4, 3), 12, 4))
        self.assertEqual(plan["b1"][2], 2)
        self.assertEqual(plan["w2"][2], 3)
        self.assertEqual(plan["b2"][2], 1)

    def test_ragged_params_round_up(self):
        """元素数 3、W=2 → 块长 2，最后一块只有一个真实元素。"""
        plan = plan_chunks(_params(shapes=_RAGGED), 2)
        self.assertEqual(plan["b1"][1], 3)
        self.assertEqual(plan["b1"][2], 2)

    def test_bad_world_size_is_refused(self):
        with self.assertRaises(TypeError):
            plan_chunks(_params(), True)
        with self.assertRaises(TypeError):
            plan_chunks(_params(), 2.0)
        with self.assertRaises(ValueError):
            plan_chunks(_params(), 0)

    def test_empty_table_and_empty_tensor_are_refused(self):
        with self.assertRaises(ValueError):
            plan_chunks({}, 2)
        with self.assertRaises(ValueError):
            plan_chunks({"x": np.zeros((0,), np.float32)}, 2)


class Zero3SemanticsTest(unittest.TestCase):
    """拼回、更新、释放三件事的数值语义。"""

    def test_materialize_restores_full_params_bitwise(self):
        for world_size in (2, 3):
            params = _params()
            _, instances = _build(
                world_size, params_list=[_params() for _ in range(world_size)]
            )
            seen = _materialize_all(instances)
            for name in _SHAPES:
                for rank in range(world_size):
                    np.testing.assert_array_equal(seen[rank][name], params[name])

    def test_one_step_matches_full_param_sgd(self):
        world_size = 2
        _, instances = _build(world_size)
        _step_all(instances, {rank: _grads(rank, 0) for rank in range(world_size)})
        expected = _full_reference(_SHAPES, 1, world_size)
        seen = _materialize_all(instances)
        for name in _SHAPES:
            for rank in range(world_size):
                np.testing.assert_array_equal(seen[rank][name], expected[name])

    def test_two_steps_match_full_param_sgd(self):
        """两步：第一步之后拼回来的参数必须能继续参与训练。"""
        world_size = 2
        _, instances = _build(world_size)
        for step in range(2):
            _step_all(
                instances, {rank: _grads(rank, step) for rank in range(world_size)}
            )
        expected = _full_reference(_SHAPES, 2, world_size)
        seen = _materialize_all(instances)
        for name in _SHAPES:
            np.testing.assert_array_equal(seen[0][name], expected[name])

    def test_ragged_params_keep_padding_out_of_the_result(self):
        """补零位置既不参与数值，也不会被写回参数。"""
        world_size = 2
        _, instances = _build(world_size, shapes=_RAGGED)
        _step_all(
            instances,
            {rank: _grads(rank, 0, shapes=_RAGGED) for rank in range(world_size)},
        )
        expected = _full_reference(_RAGGED, 1, world_size)
        seen = _materialize_all(instances)
        for name in _RAGGED:
            np.testing.assert_array_equal(seen[0][name], expected[name])
        # 尾块的补零位置在优化器里始终是 0（动量没有被"假梯度"污染）。
        b1 = instances[1].shards["b1"]
        self.assertEqual(b1.shape, (2,))
        self.assertEqual(float(b1[1]), 0.0)

    def test_resident_params_are_one_slice_per_rank(self):
        world_size = 2
        _, instances = _build(world_size)
        for instance in instances:
            self.assertLess(instance.owned_bytes(), instance.full_bytes())
        total = sum(instance.owned_bytes() for instance in instances)
        # 合计略大于一份整参数——多出来的是每个参数末尾的补零。
        slack = world_size * len(_SHAPES) * np.dtype(np.float32).itemsize
        self.assertGreaterEqual(total, instances[0].full_bytes())
        self.assertLessEqual(total, instances[0].full_bytes() + slack)

    def test_materialize_reuses_cache_until_release(self):
        _, instances = _build(2)
        first = _materialize_all(instances)
        second = _materialize_all(instances)
        for rank in range(2):
            for name in _SHAPES:
                self.assertIs(first[rank][name], second[rank][name])
            instances[rank].release()
            self.assertEqual(dict(instances[rank].resident), {})

    def test_step_invalidates_the_cache(self):
        """参数变了，拼回来的副本必须立刻作废，否则就是两份真相。"""
        _, instances = _build(2)
        _materialize_all(instances)
        _step_all(instances, {rank: _grads(rank, 0) for rank in range(2)})
        for instance in instances:
            self.assertEqual(dict(instance.resident), {})

    def test_optimizer_only_holds_chunks(self):
        _, instances = _build(3)
        for instance in instances:
            for name, array in instance.optimizer.params.items():
                self.assertEqual(array.shape, (instance.chunk_size(name),))
            self.assertEqual(set(instance.optimizer.params), set(_SHAPES))


class Zero3ValidationTest(unittest.TestCase):
    """拒绝面：不静默降级，也不"能算多少算多少"。"""

    def test_materialize_without_collective_is_refused(self):
        instance = Zero3Rank(_params(), _factory, rank=0, world_size=2)
        with self.assertRaises(RuntimeError) as caught:
            instance.materialize()
        self.assertIn("集合通信", str(caught.exception))

    def test_step_without_collective_is_refused(self):
        instance = Zero3Rank(_params(), _factory, rank=0, world_size=2)
        with self.assertRaises(RuntimeError):
            instance.step(_grads(0))

    def test_bad_grads_are_refused(self):
        instance = Zero3Rank(
            _params(), _factory, rank=0, world_size=1,
            collective=LoopbackWorld(1).collective(0),
        )
        missing = _grads(0)
        missing.pop("b2")
        with self.assertRaises(KeyError):
            instance.step(missing)
        extra = dict(_grads(0), ghost=np.zeros(2, np.float32))
        with self.assertRaises(KeyError):
            instance.step(extra)
        wrong_shape = dict(_grads(0), b2=np.zeros(3, np.float32))
        with self.assertRaises(ValueError):
            instance.step(wrong_shape)
        wrong_dtype = dict(_grads(0), b2=np.zeros(2, np.float64))
        with self.assertRaises(TypeError):
            instance.step(wrong_dtype)

    def test_bad_reduction_and_names_are_refused(self):
        instance = Zero3Rank(
            _params(), _factory, rank=0, world_size=1,
            collective=LoopbackWorld(1).collective(0),
        )
        with self.assertRaises(ValueError):
            instance.step(_grads(0), reduction="max")
        with self.assertRaises(TypeError):
            instance.materialize("w1")
        with self.assertRaises(KeyError):
            instance.materialize(["ghost"])

    def test_attach_checks_methods_and_identity(self):
        instance = Zero3Rank(_params(), _factory, rank=0, world_size=2)
        with self.assertRaises(TypeError):
            instance.attach(object())
        with self.assertRaises(ValueError):
            instance.attach(LoopbackWorld(2).collective(1))
        with self.assertRaises(ValueError):
            instance.attach(LoopbackWorld(3).collective(0))

    def test_factory_copying_the_subtable_is_refused(self):
        def copying(params):
            return SGD(
                {name: np.array(array) for name, array in params.items()},
                lr=0.1,
            )

        with self.assertRaises(ValueError) as caught:
            Zero3Rank(_params(), copying, rank=0, world_size=1)
        self.assertIn("本块那个数组", str(caught.exception))

    def test_mixed_dtype_and_empty_table_are_refused(self):
        params = _params()
        params["b2"] = params["b2"].astype(np.float64)
        with self.assertRaises(TypeError):
            Zero3Rank(params, _factory, rank=0, world_size=1)
        with self.assertRaises(ValueError):
            Zero3Rank({}, _factory, rank=0, world_size=1)

    def test_bad_rank_and_world_size(self):
        with self.assertRaises(TypeError):
            Zero3Rank(_params(), _factory, rank=0, world_size=1.0)
        with self.assertRaises(ValueError):
            Zero3Rank(_params(), _factory, rank=2, world_size=2)
        with self.assertRaises(TypeError):
            Zero3Rank(_params(), _factory, rank=True, world_size=2)


class Zero3CheckpointTest(unittest.TestCase):
    """参数本身也分片了，存档必须同时带参数分片与槽位。"""

    def test_save_only_contains_owned_chunks(self):
        _, instances = _build(2)
        _step_all(instances, {rank: _grads(rank, 0) for rank in range(2)})
        with tempfile.TemporaryDirectory() as workdir:
            for rank, instance in enumerate(instances):
                path = instance.save_shard(os.path.join(workdir, f"rank{rank}"))
                with np.load(path, allow_pickle=False) as data:
                    keys = list(data.files)
                    meta = json.loads(str(data["__meta__"]))
                    params = [key for key in keys if key.startswith("param::")]
                    self.assertEqual(
                        {key.partition("::")[2] for key in params}, set(_SHAPES)
                    )
                    for key in params:
                        name = key.partition("::")[2]
                        self.assertEqual(
                            data[key].shape, (instance.chunk_size(name),)
                        )
                self.assertEqual(meta["rank"], rank)
                self.assertEqual(meta["world_size"], 2)

    def test_round_trip_restores_params_and_steps(self):
        _, instances = _build(2)
        for step in range(2):
            _step_all(
                instances, {rank: _grads(rank, step) for rank in range(2)}
            )
        expected = _materialize_all(instances)
        steps = [instance.steps for instance in instances]
        with tempfile.TemporaryDirectory() as workdir:
            paths = [
                instance.save_shard(os.path.join(workdir, f"rank{rank}"))
                for rank, instance in enumerate(instances)
            ]
            _, fresh = _build(2)
            for rank, path in enumerate(paths):
                fresh[rank].load_shard(path)
            self.assertEqual([instance.steps for instance in fresh], steps)
            seen = _materialize_all(fresh)
            for rank in range(2):
                for name in _SHAPES:
                    np.testing.assert_array_equal(
                        seen[rank][name], expected[rank][name]
                    )

    def test_loading_another_rank_or_plan_is_refused(self):
        _, instances = _build(2)
        with tempfile.TemporaryDirectory() as workdir:
            other = instances[1].save_shard(os.path.join(workdir, "rank1"))
            with self.assertRaises(ValueError) as caught:
                instances[0].load_shard(other)
            self.assertIn("rank 1", str(caught.exception))

            path = instances[0].save_shard(os.path.join(workdir, "rank0"))
            bigger = Zero3Rank(_params(), _factory, rank=0, world_size=3)
            with self.assertRaises(ValueError) as caught:
                bigger.load_shard(path)
            self.assertIn("world_size", str(caught.exception))

            raggedy = Zero3Rank(_params(shapes=_RAGGED), _factory, rank=0, world_size=2)
            with self.assertRaises(ValueError) as caught:
                raggedy.load_shard(path)
            self.assertIn("块长", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
