"""跨进程 ZeRO-1：分片归属、只持本片槽位、通信对拍与存档。

两层：

- **离线**：用 ``LoopbackWorld``（线程屏障的假 world）让多个 ``ZeroRank``
  并发跑同一组集合调用，判据是与 ``ShardedOptimizer`` **逐位相同**——分片
  只换布局，不换运算顺序。拒绝面（factory 复制子表、混合 dtype、梯度不齐、
  attach 不匹配）全部在构造 / 入口处报错。
- **真机**：两个子进程各占一张 V100，走 NCCL 的 AllReduce + Broadcast 跑
  两步，与主机侧期望值在 1e-5 内一致。判据不能收紧到逐位：NCCL 的归约顺序
  由拓扑决定，浮点加法不满足结合律。
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

import aetherflow.train.distributed as distributed
from aetherflow.backends.cuda import CUDA_AVAILABLE
from aetherflow.train import (
    SGD,
    LoopbackWorld,
    NcclDeviceCollective,
    NcclHostCollective,  # noqa: F401  （导出面的一部分，真机 worker 用）
    ShardedOptimizer,
    ZeroRank,
    allreduce_grads,
    nccl,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SHAPES = {"w1": (4, 3), "b1": (4,), "w2": (2, 4), "b2": (2,)}


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


def _run(instances, grads_by_rank):
    """让多个 rank 并发 step；谁抛异常就记下来，最后重新抛出。"""
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


def _build(world_size, params_list=None, collective=None):
    world = collective or LoopbackWorld(world_size)
    params_list = params_list or [_params() for _ in range(world_size)]
    instances = [
        ZeroRank(
            params_list[rank],
            _factory,
            rank=rank,
            world_size=world_size,
            collective=world.collective(rank),
        )
        for rank in range(world_size)
    ]
    return world, instances


class ZeroRankSemanticsTest(unittest.TestCase):
    """分片语义：数值、归属与"只持本片"三件事。"""

    def test_two_steps_match_sharded_optimizer_bitwise(self):
        """两步都对拍：第一步验证广播后的参数能继续训练，而不只是首次一致。"""
        params_list = [_params(), _params()]
        _, instances = _build(2, params_list)
        _run(instances, {rank: _grads(rank, 0) for rank in range(2)})
        _run(instances, {rank: _grads(rank, 1) for rank in range(2)})

        whole = _params()
        optimizer = ShardedOptimizer(_factory, whole, world_size=2)
        for step in range(2):
            optimizer.step(
                allreduce_grads({rank: _grads(rank, step) for rank in range(2)})
            )

        for name in whole:
            for rank in range(2):
                np.testing.assert_array_equal(params_list[rank][name], whole[name])

    def test_slots_are_only_the_owned_shard(self):
        """槽位字节之和等于不分片的一份，且每片严格更小。"""
        _, instances = _build(2)
        _run(instances, {rank: _grads(rank) for rank in range(2)})
        whole = ShardedOptimizer(_factory, _params(), world_size=2)
        whole.step(allreduce_grads({rank: _grads(rank) for rank in range(2)}))
        owned = [instance.owned_bytes() for instance in instances]
        total = sum(owned)
        reference = sum(whole.slot_bytes_by_shard())
        self.assertEqual(total, reference)
        self.assertLess(max(owned), reference)

    def test_optimizer_holds_exactly_its_names(self):
        _, instances = _build(2)
        seen = []
        for instance in instances:
            self.assertEqual(
                list(instance.optimizer.params), list(instance.my_names)
            )
            seen.extend(instance.my_names)
        self.assertEqual(sorted(seen), sorted(_SHAPES))

    def test_assignments_match_shard_params(self):
        from aetherflow.train.sharded import shard_params

        _, instances = _build(2)
        expected = {
            name: index
            for index, names in enumerate(shard_params(_params(), 2))
            for name in names
        }
        self.assertEqual(dict(instances[0].assignments), expected)

    def test_more_ranks_than_params_leaves_empty_shards(self):
        """空片是合法状态：该 rank 不更新参数，但仍要参与广播。"""
        world_size = len(_SHAPES) + 1
        params_list = [_params() for _ in range(world_size)]
        _, instances = _build(world_size, params_list)
        empty = [instance for instance in instances if not instance.my_names]
        self.assertEqual(len(empty), 1)
        _run(
            instances,
            {rank: _grads(rank) for rank in range(world_size)},
        )
        for rank in range(1, world_size):
            for name in _SHAPES:
                np.testing.assert_array_equal(
                    params_list[rank][name], params_list[0][name]
                )


class ZeroRankValidationTest(unittest.TestCase):
    """拒绝面：构造期与入口都要显式报错，不静默降级。"""

    def test_factory_copying_the_subtable_is_refused(self):
        def copying(params):
            return SGD({name: np.array(array) for name, array in params.items()}, lr=0.1)

        with self.assertRaises(ValueError) as caught:
            ZeroRank(_params(), copying, rank=0, world_size=1)
        self.assertIn("同一个数组", str(caught.exception))

    def test_factory_must_return_optimizer(self):
        with self.assertRaises(TypeError):
            ZeroRank(_params(), lambda p: object(), rank=0, world_size=1)

    def test_empty_params_are_refused(self):
        with self.assertRaises(ValueError):
            ZeroRank({}, _factory, rank=0, world_size=1)

    def test_mixed_dtypes_are_refused(self):
        params = _params()
        params["b2"] = params["b2"].astype(np.float64)
        with self.assertRaises(TypeError) as caught:
            ZeroRank(params, _factory, rank=0, world_size=1)
        self.assertIn("同一 dtype", str(caught.exception))

    def test_bad_rank_and_world_size(self):
        with self.assertRaises(TypeError):
            ZeroRank(_params(), _factory, rank=0, world_size=1.0)
        with self.assertRaises(ValueError):
            ZeroRank(_params(), _factory, rank=2, world_size=2)
        with self.assertRaises(TypeError):
            ZeroRank(_params(), _factory, rank=True, world_size=2)

    def test_attach_requires_collective_methods(self):
        instance = ZeroRank(_params(), _factory, rank=0, world_size=2)
        with self.assertRaises(TypeError):
            instance.attach(object())

    def test_attach_checks_rank_and_world(self):
        instance = ZeroRank(_params(), _factory, rank=0, world_size=2)
        other = LoopbackWorld(2).collective(1)
        with self.assertRaises(ValueError):
            instance.attach(other)
        bigger = LoopbackWorld(3).collective(0)
        with self.assertRaises(ValueError):
            instance.attach(bigger)

    def test_step_without_collective_is_refused(self):
        instance = ZeroRank(_params(), _factory, rank=0, world_size=1)
        with self.assertRaises(RuntimeError) as caught:
            instance.step(_grads(0))
        self.assertIn("集合通信", str(caught.exception))

    def test_bad_grads_are_refused(self):
        instance = ZeroRank(_params(), _factory, rank=0, world_size=1)
        instance.attach(LoopbackWorld(1).collective(0))
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

    def test_bad_reduction_is_refused(self):
        instance = ZeroRank(_params(), _factory, rank=0, world_size=1)
        instance.attach(LoopbackWorld(1).collective(0))
        with self.assertRaises(ValueError):
            instance.step(_grads(0), reduction="max")


class ZeroRankCheckpointTest(unittest.TestCase):
    """每进程只落地本片：文件只含本片、装错片报错。"""

    def test_save_only_contains_owned_names(self):
        _, instances = _build(2)
        _run(instances, {rank: _grads(rank) for rank in range(2)})
        with tempfile.TemporaryDirectory() as workdir:
            for rank, instance in enumerate(instances):
                path = instance.save_shard(os.path.join(workdir, f"rank{rank}"))
                with np.load(path, allow_pickle=False) as data:
                    keys = [key for key in data.files if key != "__meta__"]
                    meta = json.loads(str(data["__meta__"]))
                names = {key.partition("::")[2] for key in keys}
                self.assertEqual(names, set(instance.my_names))
                self.assertEqual(meta["rank"], rank)
                self.assertEqual(
                    {name for name, owner in meta["assignments"].items()},
                    set(instance.my_names),
                )

    def test_loading_another_rank_is_refused(self):
        _, instances = _build(2)
        with tempfile.TemporaryDirectory() as workdir:
            other = instances[1].save_shard(os.path.join(workdir, "rank1"))
            with self.assertRaises(ValueError) as caught:
                instances[0].load_shard(other)
            self.assertIn("rank 1", str(caught.exception))

    def test_world_size_mismatch_is_refused(self):
        _, instances = _build(2)
        with tempfile.TemporaryDirectory() as workdir:
            path = instances[0].save_shard(os.path.join(workdir, "rank0"))
            smaller = ZeroRank(_params(), _factory, rank=0, world_size=3)
            with self.assertRaises(ValueError) as caught:
                smaller.load_shard(path)
            self.assertIn("world_size", str(caught.exception))

    def test_round_trip_keeps_slots(self):
        _, instances = _build(2)
        _run(instances, {rank: _grads(rank) for rank in range(2)})
        with tempfile.TemporaryDirectory() as workdir:
            before = instances[0].owned_bytes()
            path = instances[0].save_shard(os.path.join(workdir, "rank0"))
            instances[0].load_shard(path)
            self.assertEqual(instances[0].owned_bytes(), before)


def _device_count():
    if not CUDA_AVAILABLE:
        return 0
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:  # noqa: BLE001
        return 0


@unittest.skipUnless(
    CUDA_AVAILABLE and nccl.available() and _device_count() >= 2,
    "需要 NCCL 与至少两块 CUDA 设备",
)
class ZeroRankOnDeviceTest(unittest.TestCase):
    """真机：两个进程各占一张卡，两步 ZeRO 与主机期望值对拍。"""

    def test_two_steps_match_converged_host_reference(self):
        world = 2
        with tempfile.TemporaryDirectory() as workdir:
            uid_path = os.path.join(workdir, "uid.bin")
            outs = [os.path.join(workdir, f"rank{rank}.npz") for rank in range(world)]
            env = dict(os.environ)
            env["PYTHONPATH"] = _ROOT + os.pathsep + env.get("PYTHONPATH", "")
            procs = [
                subprocess.Popen(
                    [
                        sys.executable, "-m", "tests._zero_worker",
                        str(rank), str(world), uid_path, outs[rank],
                    ],
                    cwd=_ROOT, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                )
                for rank in range(world)
            ]
            logs = []
            for proc in procs:
                output, _ = proc.communicate(timeout=300)
                logs.append(output.decode("utf-8", "replace"))
                if proc.returncode != 0:
                    self.fail(
                        f"ZeRO worker 退出码 {proc.returncode}：\n" + "\n".join(logs)
                    )
            payloads = [dict(np.load(path, allow_pickle=False)) for path in outs]

        expected = _params()
        optimizer = ShardedOptimizer(_factory, expected, world_size=world)
        for step in range(2):
            optimizer.step(
                allreduce_grads(
                    {rank: _grads(rank, step) for rank in range(world)}
                )
            )

        for rank in range(world):
            for name in _SHAPES:
                np.testing.assert_allclose(
                    payloads[rank][f"p::{name}"], expected[name],
                    rtol=0, atol=1e-5,
                )
            if rank:
                for name in _SHAPES:
                    np.testing.assert_allclose(
                        payloads[rank][f"p::{name}"], payloads[0][f"p::{name}"],
                        rtol=0, atol=1e-5,
                    )

        metas = [json.loads(str(payload["__meta__"])) for payload in payloads]
        self.assertEqual(metas[0]["assignments"], metas[1]["assignments"])
        owned = [meta["owned_bytes"] for meta in metas]
        self.assertEqual(sum(owned), sum(optimizer.slot_bytes_by_shard()))


class NcclDeviceCollectiveFromEnvTest(unittest.TestCase):
    """5.331：DDP 默认自建设备侧 communicator，不再要求调用方先建 ``comm``。

    rank 0 生成 uid 并挂分发线程，其余 rank 从 ``AF_MASTER_ADDR`` /
    ``AF_MASTER_PORT`` 取回；组大小为 1 时连 NCCL 都不导入。
    """

    class _Comm:
        def __init__(self, rank, world_size):
            self.rank = rank
            self.world_size = world_size

    def test_single_rank_is_passthrough_without_nccl(self):
        collective = NcclDeviceCollective.from_env(rank=0, world_size=1)
        self.assertEqual((collective.rank, collective.world_size), (0, 1))
        value = np.array([1.0, 2.0], np.float32)
        # 组大小为 1：归约是恒等，也不需要导入 cupy
        self.assertIs(collective.all_reduce(value), value)

    def test_rank_zero_publishes_uid_then_initializes(self):
        from types import SimpleNamespace

        payload = b"u" * 128
        uid = SimpleNamespace(to_bytes=lambda: payload)
        server = mock.Mock()
        with mock.patch.object(nccl, "unique_id", return_value=uid), mock.patch.object(
            nccl, "init_rank", return_value=self._Comm(0, 2)
        ) as init, mock.patch.object(
            distributed, "UniqueIdServer", return_value=server
        ) as server_cls, mock.patch.object(
            distributed, "env_master_addr", return_value="127.0.0.1"
        ), mock.patch.object(distributed, "env_master_port", return_value=29500):
            collective = NcclDeviceCollective.from_env(rank=0, world_size=2, device=0)

        server_cls.assert_called_once_with(
            2, host="127.0.0.1", port=29500, timeout=60.0
        )
        server.serve.assert_called_once_with(payload)
        init.assert_called_once_with(uid, 0, 2, device=0)
        self.assertEqual(collective.device, 0)

    def test_worker_fetches_uid_over_master_endpoint(self):
        payload = b"v" * 128
        with mock.patch.object(
            distributed, "fetch_unique_id", return_value=payload
        ) as fetch, mock.patch.object(
            nccl, "init_rank", return_value=self._Comm(1, 2)
        ) as init, mock.patch.object(
            distributed, "env_master_addr", return_value="10.0.0.7"
        ), mock.patch.object(distributed, "env_master_port", return_value=1234):
            collective = NcclDeviceCollective.from_env(rank=1, world_size=2, device=1)

        fetch.assert_called_once_with("10.0.0.7", 1234, 1, timeout=60.0)
        self.assertEqual(init.call_args[0][0].to_bytes(), payload)
        self.assertEqual((collective.rank, collective.world_size), (1, 2))

    def test_rejects_inconsistent_rank_and_world(self):
        with self.assertRaises(ValueError):
            NcclDeviceCollective.from_env(rank=2, world_size=2)
        with self.assertRaises(ValueError):
            NcclDeviceCollective.from_env(rank=0, world_size=0)


if __name__ == "__main__":
    unittest.main()
