"""跨进程 ZeRO-2：等长分段、梯度只留本片、AllGather 拼回参数。

与 5.229 的 ZeRO-1 用例共用同一台"假 world"（``LoopbackWorld``）与同一组
参数形状，判据仍取**逐位**：Loopback 按 rank 序号累加，与主机侧的
``sum(grads) / world_size`` 是同一条加法链，浮点结果应当完全一致。

真机（2×V100 的 ReduceScatter + AllGather）本轮没有条件跑，因此这里不写
"已在真机验证"的断言；``NcclHostCollective`` 的两个新方法与 NCCL 的形参
一致性由 ``tests/test_nccl.py`` 的绑定检查覆盖。
"""

import os
import tempfile
import unittest

import numpy as np

from aetherflow.train import SGD, LoopbackWorld, Zero2Rank
from aetherflow.train.zero2 import plan_segments

_SHAPES = {"w1": (4, 3), "b1": (4,), "w2": (2, 4), "b2": (2,)}
_LR = 0.1
_MOMENTUM = 0.9


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
    return SGD(params, lr=_LR, momentum=_MOMENTUM)


def _build(world_size, params_list=None):
    """各 rank 用同一批初值的**独立副本**（真实场景就是各自加载同一份权重）。"""
    world = LoopbackWorld(world_size)
    if params_list is None:
        params_list = [_params() for _ in range(world_size)]
    else:
        params_list = [
            {name: np.array(value, copy=True) for name, value in table.items()}
            for table in params_list
        ]
    instances = [
        Zero2Rank(
            params_list[rank],
            _factory,
            rank=rank,
            world_size=world_size,
            collective=world.collective(rank),
        )
        for rank in range(world_size)
    ]
    return world, instances


def _run(instances, grads_by_rank):
    import threading

    errors = []

    def target(index):
        try:
            instances[index].step(grads_by_rank[index])
        except BaseException as exc:  # noqa: BLE001
            errors.append((index, exc))

    threads = [
        threading.Thread(target=target, args=(index,))
        for index in range(len(instances))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0][1]


def _expected(params, grads_by_rank, steps=1):
    """主机侧答案：完整参数 + 各 rank 梯度的均值，走同样步数的 SGD。"""
    full = {name: np.array(value, copy=True) for name, value in params.items()}
    optimizer = SGD(full, lr=_LR, momentum=_MOMENTUM)
    for step in range(steps):
        merged = {
            name: sum(grads[name] for grads in grads_by_rank) / len(grads_by_rank)
            for name in full
        }
        optimizer.step(merged)
    return full


class PlanSegmentsTest(unittest.TestCase):
    """分段是个纯函数：覆盖、均衡、补齐长度都可以单独断言。"""

    def test_each_param_belongs_to_exactly_one_segment(self):
        params = _params()
        segments, _ = plan_segments(params, 3)
        seen = [name for names, _ in segments for name in names]
        self.assertEqual(sorted(seen), sorted(_SHAPES))
        self.assertEqual(len(seen), len(set(seen)))

    def test_segment_order_follows_global_param_order(self):
        params = _params()
        segments, _ = plan_segments(params, 2)
        order = {name: index for index, name in enumerate(params)}
        for names, _ in segments:
            self.assertEqual(names, sorted(names, key=order.__getitem__))

    def test_segment_sizes_are_balanced(self):
        params = _params()  # 12 + 4 + 8 + 2 = 26 个元素
        segments, pad = plan_segments(params, 3)
        sizes = [size for _, size in segments]
        largest = max(int(np.prod(shape)) for shape in _SHAPES.values())
        self.assertEqual(sum(sizes), 26)
        self.assertLessEqual(max(sizes) - min(sizes), largest)
        self.assertEqual(pad, max(sizes))

    def test_world_size_type_and_range(self):
        params = _params()
        with self.assertRaises(TypeError):
            plan_segments(params, True)
        with self.assertRaises(TypeError):
            plan_segments(params, 2.0)
        with self.assertRaises(ValueError):
            plan_segments(params, 0)

    def test_empty_param_table_is_rejected(self):
        with self.assertRaises(ValueError):
            plan_segments({}, 2)


class Zero2SemanticsTest(unittest.TestCase):
    """两步更新与"完整参数 + 梯度均值"逐位相同，且各 rank 收尾一致。"""

    def test_two_steps_match_full_param_sgd(self):
        world_size = 3
        base = _params()
        _, instances = _build(world_size, [base for _ in range(world_size)])
        per_step = [
            [_grads(rank, step) for rank in range(world_size)]
            for step in (0, 1)
        ]
        for step in (0, 1):
            _run(instances, per_step[step])
        optimizer = SGD(
            {name: np.array(value, copy=True) for name, value in base.items()},
            lr=_LR,
            momentum=_MOMENTUM,
        )
        for step in (0, 1):
            merged = {
                name: sum(
                    per_step[step][rank][name] for rank in range(world_size)
                )
                / world_size
                for name in base
            }
            optimizer.step(merged)
        for index, instance in enumerate(instances):
            for name, value in optimizer.params.items():
                np.testing.assert_array_equal(
                    instance.params[name],
                    value,
                    err_msg=f"rank {index} 的 {name} 与主机期望不同",
                )

    def test_all_ranks_end_with_identical_params(self):
        world_size = 4
        base = _params()
        _, instances = _build(world_size, [base for _ in range(world_size)])
        _run(instances, [_grads(rank) for rank in range(world_size)])
        first = instances[0].params
        for instance in instances[1:]:
            for name in _SHAPES:
                np.testing.assert_array_equal(instance.params[name], first[name])

    def test_optimizer_sees_only_its_segment(self):
        world_size = 3
        base = _params()
        _, instances = _build(world_size, [base for _ in range(world_size)])
        for rank, instance in enumerate(instances):
            self.assertEqual(
                sorted(instance.optimizer.params), sorted(instance.my_names)
            )
            self.assertEqual(
                sorted(instance.assignments),
                sorted(_SHAPES),
                msg=f"rank {rank} 的归属表应覆盖全部参数",
            )

    def test_owned_slots_sum_to_full_slots(self):
        world_size = 4
        base = _params()
        _, instances = _build(world_size, [base for _ in range(world_size)])
        _run(instances, [_grads(rank) for rank in range(world_size)])
        total = sum(instance.owned_bytes() for instance in instances)
        full = sum(
            int(np.prod(shape)) * 4  # float32 的动量槽位一份
            for shape in _SHAPES.values()
        )
        self.assertEqual(total, full)
        for instance in instances:
            self.assertLess(instance.owned_bytes(), full)

    def test_empty_segments_are_allowed(self):
        world_size = 5
        base = {"w1": _params()["w1"], "b2": _params()["b2"]}
        shares = [{name: value.copy() for name, value in base.items()} for _ in range(world_size)]
        _, instances = _build(world_size, shares)
        empties = [instance for instance in instances if not instance.my_names]
        self.assertTrue(empties, "2 个参数切 5 段必然有空段")
        grads_by_rank = [
            {
                name: np.full_like(value, float(rank + 1))
                for name, value in base.items()
            }
            for rank in range(world_size)
        ]
        _run(instances, grads_by_rank)
        expected = _expected(base, grads_by_rank)
        for instance in instances:
            for name in base:
                np.testing.assert_allclose(
                    instance.params[name], expected[name], rtol=0, atol=1e-6
                )

    def test_communication_elements_are_two_halves(self):
        world_size = 4
        _, instances = _build(world_size)
        instance = instances[0]
        self.assertEqual(
            instance.communication_elements(), instance.pad * world_size
        )
        self.assertGreaterEqual(instance.communication_elements(), 26)

    def test_mean_and_sum_differ_by_world_size(self):
        world_size = 2
        base = _params()
        mean_world, mean_ranks = _build(world_size, [base for _ in range(world_size)])
        sum_shares = [{name: value.copy() for name, value in base.items()} for _ in range(world_size)]
        sum_world, sum_ranks = _build(world_size, sum_shares)
        grads_by_rank = [_grads(rank) for rank in range(world_size)]

        def one(instances, world, reduction):
            import threading

            errors = []

            def target(index):
                try:
                    instances[index].step(grads_by_rank[index], reduction=reduction)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [
                threading.Thread(target=target, args=(index,))
                for index in range(len(instances))
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            if errors:
                raise errors[0]

        one(mean_ranks, mean_world, "mean")
        one(sum_ranks, sum_world, "sum")
        for name in _SHAPES:
            mean_delta = mean_ranks[0].params[name] - base[name]
            sum_delta = sum_ranks[0].params[name] - base[name]
            np.testing.assert_allclose(
                sum_delta, mean_delta * world_size, rtol=0, atol=1e-6
            )

    def test_step_counter_advances(self):
        world_size = 2
        base = _params()
        _, instances = _build(world_size, [base for _ in range(world_size)])
        _run(instances, [_grads(rank) for rank in range(world_size)])
        _run(instances, [_grads(rank, 1) for rank in range(world_size)])
        for instance in instances:
            self.assertEqual(instance.steps, 2)


class Zero2ValidationTest(unittest.TestCase):
    """拒绝面：构造期能查出来的错误不留到集合调用里。"""

    def test_collective_needs_both_collectives(self):
        class OnlyAllReduce:
            rank = 0
            world_size = 2

            def all_reduce(self, array):
                return array

        base = _params()
        with self.assertRaises(TypeError):
            Zero2Rank(
                {name: value.copy() for name, value in base.items()},
                _factory,
                rank=0,
                world_size=2,
                collective=OnlyAllReduce(),
            )

    def test_collective_rank_mismatch(self):
        world = LoopbackWorld(2)
        base = _params()
        with self.assertRaises(ValueError):
            Zero2Rank(
                {name: value.copy() for name, value in base.items()},
                _factory,
                rank=0,
                world_size=2,
                collective=world.collective(1),
            )

    def test_collective_world_mismatch(self):
        world = LoopbackWorld(3)
        base = _params()
        with self.assertRaises(ValueError):
            Zero2Rank(
                {name: value.copy() for name, value in base.items()},
                _factory,
                rank=0,
                world_size=2,
                collective=world.collective(0),
            )

    def test_step_without_collective(self):
        base = _params()
        instance = Zero2Rank(base, _factory, rank=0, world_size=1)
        with self.assertRaises(RuntimeError):
            instance.step(_grads(0))

    def test_rank_and_world_validation(self):
        base = _params()
        with self.assertRaises(TypeError):
            Zero2Rank(base, _factory, rank=True, world_size=2)
        with self.assertRaises(ValueError):
            Zero2Rank(base, _factory, rank=2, world_size=2)
        with self.assertRaises(ValueError):
            Zero2Rank(base, _factory, rank=0, world_size=0)

    def test_missing_and_unknown_gradients(self):
        world_size = 2
        base = _params()
        _, instances = _build(world_size, [base for _ in range(world_size)])
        grads = [_grads(rank) for rank in range(world_size)]
        incomplete = dict(grads[0])
        incomplete.pop("w2")
        with self.assertRaises(KeyError):
            instances[0].step(incomplete)
        extra = dict(grads[0])
        extra["bogus"] = np.zeros(1, dtype=np.float32)
        with self.assertRaises(KeyError):
            instances[0].step(extra)

    def test_shape_and_dtype_mismatch(self):
        world_size = 2
        base = _params()
        _, instances = _build(world_size, [base for _ in range(world_size)])
        bad_shape = _grads(0)
        bad_shape["w1"] = np.zeros((3, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            instances[0].step(bad_shape)
        bad_dtype = _grads(0)
        bad_dtype["w1"] = bad_dtype["w1"].astype(np.float64)
        with self.assertRaises(TypeError):
            instances[0].step(bad_dtype)

    def test_mixed_param_dtypes_rejected(self):
        base = {name: value.copy() for name, value in _params().items()}
        base["b2"] = base["b2"].astype(np.float64)
        with self.assertRaises(TypeError):
            Zero2Rank(base, _factory, rank=0, world_size=1)

    def test_factory_copying_subtable_rejected(self):
        def copying(params):
            return SGD({name: value.copy() for name, value in params.items()}, lr=0.1)

        base = _params()
        with self.assertRaises(ValueError):
            Zero2Rank(base, copying, rank=0, world_size=1)

    def test_invalid_reduction(self):
        world_size = 1
        base = _params()
        world, instances = _build(world_size, [base])
        with self.assertRaises(ValueError):
            instances[0].step(_grads(0), reduction="max")
        del world


class Zero2CheckpointTest(unittest.TestCase):
    """分片存档：本 rank 只写本段，装错片要报错。"""

    def _trained(self, world_size=3):
        base = _params()
        _, instances = _build(world_size, [base for _ in range(world_size)])
        _run(instances, [_grads(rank) for rank in range(world_size)])
        return instances

    def test_save_contains_only_owned_names(self):
        instances = self._trained()
        with tempfile.TemporaryDirectory() as folder:
            for rank, instance in enumerate(instances):
                path = instance.save_shard(os.path.join(folder, f"shard{rank}"))
                with np.load(path, allow_pickle=False) as data:
                    keys = [key for key in data.files if key != "__meta__"]
                names = {key.split("::", 1)[1] for key in keys}
                self.assertEqual(names, set(instance.my_names))

    def test_load_wrong_rank_is_rejected(self):
        instances = self._trained()
        with tempfile.TemporaryDirectory() as folder:
            path = instances[0].save_shard(os.path.join(folder, "shard0"))
            with self.assertRaises(ValueError):
                instances[1].load_shard(path)

    def test_round_trip_keeps_slots(self):
        instances = self._trained()
        with tempfile.TemporaryDirectory() as folder:
            for rank, instance in enumerate(instances):
                instance.save_shard(os.path.join(folder, f"restore{rank}"))
        baseline = instances[0].params
        # 换个初值重建，再装回存档，参数本身不存盘，所以这里比较槽位与步数
        base = _params(seed=9)
        _, fresh = _build(3, [base for _ in range(3)])
        with tempfile.TemporaryDirectory() as folder:
            for rank, instance in enumerate(instances):
                path = instance.save_shard(os.path.join(folder, f"s{rank}"))
                fresh[rank].load_shard(path)
                self.assertEqual(fresh[rank].steps, instance.steps)
                self.assertEqual(
                    fresh[rank].optimizer.step_count, instance.optimizer.step_count
                )
        del baseline


class LoopbackCollectiveExtensionTest(unittest.TestCase):
    """新增的两个集合面：长度校验与语义手算。"""

    def test_reduce_scatter_matches_manual_sum(self):
        world = LoopbackWorld(2)
        ranks = [world.collective(index) for index in range(2)]
        sends = [
            np.arange(6, dtype=np.float64) + index * 10 for index in range(2)
        ]
        recvs = [np.empty(3) for _ in range(2)]
        results = [None, None]

        def target(index):
            results[index] = np.array(
                ranks[index].reduce_scatter(sends[index], recvs[index])
            )

        import threading

        threads = [threading.Thread(target=target, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        total = sends[0] + sends[1]
        np.testing.assert_array_equal(results[0], total[:3])
        np.testing.assert_array_equal(results[1], total[3:])
        np.testing.assert_array_equal(results[0], recvs[0])

    def test_all_gather_concatenates_in_rank_order(self):
        world = LoopbackWorld(2)
        ranks = [world.collective(index) for index in range(2)]
        sends = [np.array([index + 1.0, index + 2.0]) for index in range(2)]
        recvs = [np.empty(4) for _ in range(2)]
        results = [None, None]

        def target(index):
            results[index] = np.array(
                ranks[index].all_gather(sends[index], recvs[index])
            )

        import threading

        threads = [threading.Thread(target=target, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        expected = np.concatenate(sends)
        for result in results:
            np.testing.assert_array_equal(result, expected)

    def test_length_validation(self):
        world = LoopbackWorld(1)
        rank = world.collective(0)
        with self.assertRaises(ValueError):
            rank.reduce_scatter(np.zeros(5), np.zeros(3))
        with self.assertRaises(ValueError):
            rank.all_gather(np.zeros(3), np.zeros(5))
        with self.assertRaises(ValueError):
            rank.reduce_scatter(np.zeros(4), np.zeros(2), op="max")

    def test_zero2_rejects_wrong_length_from_collective(self):
        class ShortScatter:
            rank = 0
            world_size = 1

            def reduce_scatter(self, send, recv, op="sum"):
                return np.zeros(1, dtype=np.float32)

            def all_gather(self, send, recv):
                return recv

        base = _params()
        instance = Zero2Rank(base, _factory, rank=0, world_size=1)
        instance.attach(ShortScatter())
        with self.assertRaises(ValueError):
            instance.step(_grads(0))


if __name__ == "__main__":
    unittest.main()
