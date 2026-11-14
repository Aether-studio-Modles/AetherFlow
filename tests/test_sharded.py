"""ZeRO 分片的验收测试。

判据只有一条主轴：**分片后的每一步必须与整份优化器逐位相同**（不是"近似
相等"）。围绕它再补三类边界：分片本身的确定性与均衡、非法配置当场拒绝、
存档往返后能接着原来的轨迹跑。

刻意不做"每片的槽位数组确实只覆盖本片"的显式断言——那是实现细节；真正
决定正确性的是"整份看不到的那部分状态没有影响结果"，而逐位一致已经涵盖它。
"""

import unittest

import numpy as np

from aetherflow.train import SGD, Adam, ShardedOptimizer, shard_params


def _params(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.standard_normal((6, 5), dtype=np.float32),
        "b1": rng.standard_normal((5,), dtype=np.float32),
        "w2": rng.standard_normal((5, 4), dtype=np.float32),
        "b2": rng.standard_normal((4,), dtype=np.float32),
        "emb": rng.standard_normal((9, 3), dtype=np.float32),
    }


def _grads(params, seed):
    rng = np.random.default_rng(seed)
    return {
        name: rng.standard_normal(arr.shape, dtype=np.float32)
        for name, arr in params.items()
    }


class ShardParamsTest(unittest.TestCase):
    def test_every_name_appears_once_and_coverage_is_total(self):
        params = _params()
        shards = shard_params(params, 3)
        self.assertEqual(len(shards), 3)
        flat = [name for shard in shards for name in shard]
        self.assertEqual(sorted(flat), sorted(params))

    def test_partition_is_deterministic(self):
        params = _params()
        self.assertEqual(shard_params(params, 2), shard_params(params, 2))

    def test_balance_is_within_one_parameter(self):
        params = _params()
        sizes = {n: int(np.prod(a.shape)) for n, a in params.items()}
        total = sum(sizes.values())
        shards = shard_params(params, 3)
        loads = [sum(sizes[n] for n in shard) for shard in shards]
        self.assertLessEqual(max(loads) - min(loads), max(sizes.values()))
        self.assertLessEqual(max(loads), total)

    def test_more_shards_than_parameters_yields_empty_shards(self):
        shards = shard_params({"only": np.zeros((4,), np.float32)}, 3)
        self.assertEqual(len(shards), 3)
        self.assertEqual(sum(len(s) for s in shards), 1)
        self.assertEqual(sum(1 for s in shards if not s), 2)

    def test_bad_world_size_is_refused(self):
        for value in (0, -1):
            with self.assertRaises(ValueError):
                shard_params(_params(), value)
        for value in (True, "2", 2.0):
            with self.assertRaises(TypeError):
                shard_params(_params(), value)

    def test_empty_params_is_refused(self):
        with self.assertRaises(ValueError):
            shard_params({}, 2)


class ShardedOptimizerTest(unittest.TestCase):
    def _compare(self, make, world_size=3, steps=5):
        whole = _params()
        full_opt = make(whole)
        sharded = _params()
        opt = ShardedOptimizer(lambda p: make(p), sharded, world_size=world_size)
        for step in range(steps):
            grads = _grads(sharded, seed=100 + step)
            full_opt.step(dict(grads))
            opt.step(dict(grads))
        for name in whole:
            np.testing.assert_array_equal(
                whole[name], sharded[name], err_msg=f"参数 {name} 与整份更新不一致"
            )
        return full_opt, opt

    def test_adam_matches_whole_optimizer_bitwise(self):
        _, opt = self._compare(lambda p: Adam(p, lr=1e-2))
        self.assertEqual(opt.step_count, 5)
        for shard in opt.shards:
            self.assertEqual(shard.step_count, 5)

    def test_sgd_momentum_matches_whole_optimizer_bitwise(self):
        self._compare(lambda p: SGD(p, lr=0.05, momentum=0.9, nesterov=True))

    def test_empty_shards_still_advance(self):
        _, opt = self._compare(lambda p: Adam(p, lr=1e-2), world_size=8, steps=2)
        self.assertEqual(len(opt.shards), 8)
        self.assertEqual(sum(1 for s in opt.shards if not s.params), 3)
        for shard in opt.shards:
            self.assertEqual(shard.step_count, 2)

    def test_missing_grad_is_refused(self):
        params = _params()
        opt = ShardedOptimizer(lambda p: Adam(p, lr=1e-2), params, world_size=2)
        grads = _grads(params, seed=1)
        grads.pop("b1")
        with self.assertRaises(KeyError):
            opt.step(grads)

    def test_unknown_grad_is_refused(self):
        params = _params()
        opt = ShardedOptimizer(lambda p: Adam(p, lr=1e-2), params, world_size=2)
        grads = _grads(params, seed=1)
        grads["extra"] = np.zeros((2,), np.float32)
        with self.assertRaises(KeyError):
            opt.step(grads)

    def test_factory_must_return_optimizer(self):
        with self.assertRaises(TypeError):
            ShardedOptimizer(lambda p: object(), _params(), world_size=2)

    def test_mismatched_hyperparams_are_refused(self):
        def factory(sub):
            # 只在含 w2 的那片用不同学习率
            return Adam(sub, lr=1e-2 if "w2" in sub else 1e-3)

        with self.assertRaises(ValueError):
            ShardedOptimizer(factory, _params(), world_size=2)

    def test_lr_setter_reaches_every_shard(self):
        opt = ShardedOptimizer(lambda p: Adam(p, lr=1e-2), _params(), world_size=2)
        opt.lr = 5e-3
        self.assertEqual(opt.lr, 5e-3)
        for shard in opt.shards:
            self.assertEqual(shard.lr, 5e-3)

    def test_assignments_is_the_inverse_of_shards(self):
        opt = ShardedOptimizer(lambda p: Adam(p, lr=1e-2), _params(), world_size=3)
        assignments = opt.assignments
        for index, shard in enumerate(opt.shards):
            for name in shard.params:
                self.assertEqual(assignments[name], index)

    def test_state_round_trip_resumes_identically(self):
        make = lambda p: Adam(p, lr=1e-2)

        base = _params()
        reference = {n: v.copy() for n, v in base.items()}
        ref_opt = make(reference)
        for step in range(5):
            ref_opt.step(_grads(reference, seed=step))

        live = {n: v.copy() for n, v in base.items()}
        opt = ShardedOptimizer(lambda p: make(p), live, world_size=2)
        for step in range(3):
            opt.step(_grads(live, seed=step))
        saved = opt.state_dict()
        at_three = {n: v.copy() for n, v in live.items()}
        for step in range(3, 5):
            opt.step(_grads(live, seed=step))

        for name in live:
            live[name][...] = at_three[name]
        restored = ShardedOptimizer(lambda p: make(p), live, world_size=2)
        restored.load_state_dict(saved)
        self.assertEqual(restored.step_count, 3)
        for step in range(3, 5):
            restored.step(_grads(live, seed=step))

        for name in reference:
            np.testing.assert_array_equal(reference[name], live[name])

    def test_load_refuses_different_world_size(self):
        params = _params()
        opt = ShardedOptimizer(lambda p: Adam(p, lr=1e-2), params, world_size=2)
        opt.step(_grads(params, seed=0))
        saved = opt.state_dict()
        other = ShardedOptimizer(lambda p: Adam(p, lr=1e-2), _params(), world_size=3)
        with self.assertRaises(ValueError):
            other.load_state_dict(saved)

    def test_load_refuses_different_assignment(self):
        params = _params()
        opt = ShardedOptimizer(lambda p: Adam(p, lr=1e-2), params, world_size=2)
        opt.step(_grads(params, seed=0))
        saved = opt.state_dict()
        saved = {**saved, "assignments": dict(saved["assignments"])}
        saved["assignments"]["w1"] = 1 - saved["assignments"]["w1"]
        other = ShardedOptimizer(lambda p: Adam(p, lr=1e-2), _params(), world_size=2)
        with self.assertRaises(ValueError):
            other.load_state_dict(saved)


if __name__ == "__main__":
    unittest.main()
