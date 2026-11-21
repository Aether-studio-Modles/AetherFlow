"""``DistributedDataParallel`` 与主机中转集合通信的验收测试。

判据分三层：

1. **数值等价**：W 个 rank 各跑本片两步，损失与参数必须与"单进程跑合并
   batch"两步一致。只测"接口能调用"发现不了梯度忘了平均、某片少喂一路
   输入这类缺陷。
2. **钩子看到的是全局梯度**：用户 ``grad_hook`` 在归约之后调用，它算出的
   梯度范数必须与单进程整批的一致——否则裁剪阈值在每个 rank 上都不一样。
3. **拒绝面当场报错**：随机流不匹配、batch 不整除、图的静态 batch 不等于
   分片大小、设备态 / GradScaler / ZeRO-3、collective 缺 ``all_reduce``、
   缺 ``AF_RELAY_PORT``，以及协议里 dtype / count 不一致。

CPU 侧用 ``LoopbackWorld`` 的线程集合通信做 2 rank 对拍；另有一条真多进程
用例走 ``launch``，验证"环境变量 + 主机中转 + 子进程"整条链路。
"""

import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

from aetherflow import Graph, ReferenceBackend, f32
from aetherflow.train import (
    SGD,
    GradScaler,
    DistributedDataParallel,
    HostCollective,
    HostRelay,
    LoopbackWorld,
    ParameterStore,
    Trainer,
    Zero3Rank,
    init_array,
    launch,
    mse,
)


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


def _linear_graph(batch=8):
    g = Graph("ddp")
    x = g.input("x", f32[batch, 3])
    w = g.input("w", f32[3, 1])
    b = g.input("b", f32[1])
    target = g.input("target", f32[batch, 1])
    g.output(mse(g, g.add(g.matmul(x, w), b), target), "loss")
    return g


def _linear_batch(total=8, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "x": rng.standard_normal((total, 3)).astype(np.float32),
        "target": rng.standard_normal((total, 1)).astype(np.float32),
    }


def _linear_store():
    return ParameterStore(
        {
            "w": init_array("w", (3, 1)),
            "b": np.zeros((1,), dtype=np.float32),
        }
    )


def _relay(world_size, timeout=10.0):
    relay = HostRelay(world_size, timeout=timeout)
    relay.serve()
    return relay


class TestHostRelay(unittest.TestCase):
    """真 TCP 上的中转：两 rank 求和、dtype / count 不一致、重复 serve。"""

    def _connect(self, relay, rank, world, timeout=10.0):
        return HostCollective(
            rank, world, host=relay.host, port=relay.port, timeout=timeout
        )

    def test_two_rank_sum_over_tcp(self):
        relay = _relay(2)
        clients = []
        try:
            clients = [self._connect(relay, rank, 2) for rank in range(2)]
            got = [None, None]

            def run(rank):
                got[rank] = clients[rank].all_reduce(
                    np.asarray([1.0, 2.0], dtype=np.float32) + rank * 2
                )

            _parallel([lambda rank=rank: run(rank) for rank in range(2)])
            for rank in range(2):
                np.testing.assert_array_equal(
                    got[rank], np.asarray([4.0, 6.0], dtype=np.float32)
                )
                self.assertEqual(got[rank].dtype, np.float32)
        finally:
            for client in clients:
                client.close()
            relay.close()

    def test_dtype_mismatch_is_reported(self):
        relay = _relay(2)
        errors = []
        clients = []
        try:
            clients = [self._connect(relay, rank, 2) for rank in range(2)]

            def target(rank, dtype):
                try:
                    clients[rank].all_reduce(np.ones((2,), dtype=dtype))
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [
                threading.Thread(target=target, args=(0, np.float32)),
                threading.Thread(target=target, args=(1, np.float64)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertTrue(errors, "dtype 不一致必须让两端都失败")
            self.assertIsNotNone(relay.error)
        finally:
            for client in clients:
                client.close()
            relay.close()

    def test_count_mismatch_is_reported(self):
        relay = _relay(2)
        errors = []
        clients = []
        try:
            clients = [self._connect(relay, rank, 2) for rank in range(2)]

            def target(rank, size):
                try:
                    clients[rank].all_reduce(np.ones((size,), dtype=np.float32))
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [
                threading.Thread(target=target, args=(0, 2)),
                threading.Thread(target=target, args=(1, 3)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertTrue(errors, "count 不一致必须让两端都失败")
            self.assertIsNotNone(relay.error)
        finally:
            for client in clients:
                client.close()
            relay.close()

    def test_duplicate_serve_is_rejected(self):
        relay = _relay(1)
        try:
            with self.assertRaises(RuntimeError):
                relay.serve()
        finally:
            relay.close()

    def test_all_reduce_rejects_non_sum_and_integer(self):
        relay = _relay(1)
        client = self._connect(relay, 0, 1)
        try:
            with self.assertRaises(ValueError):
                client.all_reduce(np.ones((2,), dtype=np.float32), op="max")
            with self.assertRaises(TypeError):
                client.all_reduce(np.ones((2,), dtype=np.int32))
        finally:
            client.close()
            relay.close()

    def test_from_env_requires_relay_port(self):
        with mock.patch.dict(
            os.environ, {"AF_MASTER_ADDR": "127.0.0.1"}, clear=False
        ):
            os.environ.pop("AF_RELAY_PORT", None)
            with self.assertRaises(RuntimeError):
                HostCollective.from_env(rank=0, world_size=1)

    def test_from_env_requires_master_addr(self):
        with mock.patch.dict(os.environ, {"AF_RELAY_PORT": "1"}, clear=False):
            os.environ.pop("AF_MASTER_ADDR", None)
            with self.assertRaises(RuntimeError):
                HostCollective.from_env(rank=0, world_size=1)


class TestDataInputShapes(unittest.TestCase):
    def test_excludes_params_and_rng_inputs(self):
        store = _linear_store()
        trainer = Trainer(_linear_graph(4), store, SGD(store.arrays(), lr=0.1))
        self.assertEqual(
            trainer.data_input_shapes(),
            {"x": (4, 3), "target": (4, 1)},
        )


class _CaptureBackend:
    """只满足"设备态训练"的入口检查；不执行任何图。"""

    def capture(self, graph, inputs):
        return lambda feed: {}

    def to_array(self, value):
        return np.asarray(value)

    def to_host(self, value):
        return np.asarray(value)

    def execute(self, graph, inputs):
        return ReferenceBackend().execute(graph, inputs)


class _NoReduce:
    rank = 0
    world_size = 1


def _build_ranks(world_size=2, batch_per_rank=4, hooks=None):
    """每个 rank 一份独立参数与 Trainer；初值相同（真实场景各自加载同一份权重）。"""
    world = LoopbackWorld(world_size)
    built = []
    for rank in range(world_size):
        store = _linear_store()
        options = {"rng_rank": rank, "rng_world_size": world_size}
        if hooks is not None:
            options["grad_hook"] = hooks[rank]
        trainer = Trainer(
            _linear_graph(batch_per_rank),
            store,
            SGD(store.arrays(), lr=0.1),
            **options,
        )
        collective = world.collective(rank)
        built.append((store, DistributedDataParallel(trainer, collective)))
    return built


class TestDistributedDataParallel(unittest.TestCase):
    """两 rank 两步，必须与单进程合并 batch 两步逐位接近。"""

    def _single(self, total=8, steps=2, hook=None):
        store = _linear_store()
        options = {"grad_hook": hook} if hook is not None else {}
        trainer = Trainer(
            _linear_graph(total), store, SGD(store.arrays(), lr=0.1), **options
        )
        batch = _linear_batch(total)
        losses = [trainer.step(batch) for _ in range(steps)]
        return store, losses

    def test_matches_single_process_merged_batch(self):
        single, want_losses = self._single()
        built = _build_ranks()
        batch = _linear_batch(8)
        got = [[], []]

        def run(rank):
            for _ in range(2):
                got[rank].append(built[rank][1].step(batch))

        _parallel([lambda rank=rank: run(rank) for rank in range(2)])
        for rank in range(2):
            store, ddp = built[rank]
            np.testing.assert_allclose(
                got[rank], want_losses, rtol=1e-6, atol=1e-7
            )
            for name in single.names:
                np.testing.assert_allclose(
                    store[name], single[name], rtol=1e-6, atol=1e-7,
                    err_msg=f"rank {rank} 的 {name}",
                )
            self.assertEqual(ddp.steps, 2)
            self.assertEqual(ddp.rank, rank)
            self.assertEqual(ddp.world_size, 2)
            self.assertIn("world_size=2", repr(ddp))

    def test_returned_loss_is_global_mean(self):
        built = _build_ranks()
        batch = _linear_batch(8)
        got = [None, None]

        def run(rank):
            got[rank] = built[rank][1].step(batch)

        _parallel([lambda rank=rank: run(rank) for rank in range(2)])
        self.assertAlmostEqual(got[0], got[1], places=7)
        _, want = self._single(steps=1)
        self.assertAlmostEqual(got[0], want[0], places=6)

    def test_user_hook_sees_reduced_grads(self):
        """钩子里的梯度范数必须等于整批梯度——即归约发生在钩子之前。"""
        single_norms = []

        def single_hook(grads):
            single_norms.append(_norm(grads))
            return grads

        self._single(hook=single_hook)

        hooks = []
        per_rank = []
        for _ in range(2):
            bucket = []
            per_rank.append(bucket)

            def hook(grads, bucket=bucket):
                bucket.append(_norm(grads))
                return grads

            hooks.append(hook)
        built = _build_ranks(hooks=hooks)
        batch = _linear_batch(8)

        def run(rank):
            built[rank][1].step(batch)

        _parallel([lambda rank=rank: run(rank) for rank in range(2)])
        for bucket in per_rank:
            np.testing.assert_allclose(bucket[0], single_norms[0], rtol=1e-6)

    # -- 拒绝面 -------------------------------------------------------

    def test_rejects_rng_mismatch(self):
        store = _linear_store()
        trainer = Trainer(
            _linear_graph(4), store, SGD(store.arrays(), lr=0.1),
            rng_rank=0, rng_world_size=1,
        )
        world = LoopbackWorld(2)
        with self.assertRaises(ValueError):
            DistributedDataParallel(trainer, world.collective(0))

    def test_rejects_indivisible_batch(self):
        built = _build_ranks()
        with self.assertRaises(ValueError):
            built[0][1].step(_linear_batch(7))

    def test_rejects_graph_not_per_rank_batch(self):
        world = LoopbackWorld(2)
        store = _linear_store()
        trainer = Trainer(
            _linear_graph(5), store, SGD(store.arrays(), lr=0.1),
            rng_rank=0, rng_world_size=2,
        )
        ddp = DistributedDataParallel(trainer, world.collective(0))
        with self.assertRaises(ValueError):
            ddp.step(_linear_batch(8))

    def test_rejects_bad_data_names(self):
        built = _build_ranks()
        ddp = built[0][1]
        batch = _linear_batch(8)
        batch.pop("target")
        with self.assertRaises(KeyError):
            ddp.step(batch)
        with self.assertRaises(KeyError):
            ddp.step({**_linear_batch(8), "y": np.zeros((8, 1), np.float32)})

    def test_rejects_device_state_and_scaler_and_zero3(self):
        world = LoopbackWorld(1)
        collective = world.collective(0)
        store = _linear_store()
        device = Trainer(
            _linear_graph(8), store, SGD(store.arrays(), lr=0.1),
            backend=_CaptureBackend(), device_state=True,
        )
        with self.assertRaises(ValueError):
            DistributedDataParallel(device, collective)

        store = _linear_store()
        scaled = Trainer(
            _linear_graph(8), store, SGD(store.arrays(), lr=0.1),
            mixed_precision="f16", scaler=GradScaler(),
        )
        with self.assertRaises(ValueError):
            DistributedDataParallel(scaled, collective)

        params = {name: _linear_store()[name].copy() for name in ("w", "b")}
        rank = Zero3Rank(
            params,
            lambda sub: SGD(sub, lr=0.1),
            rank=0,
            world_size=1,
            collective=LoopbackWorld(1).collective(0),
        )
        store = ParameterStore({name: value.copy() for name, value in params.items()})
        sharded = Trainer(
            _linear_graph(8), store, rank.optimizer, segments=2, zero3=rank
        )
        with self.assertRaises(ValueError):
            DistributedDataParallel(sharded, collective)

    def test_rejects_bad_collective(self):
        store = _linear_store()
        trainer = Trainer(
            _linear_graph(4), store, SGD(store.arrays(), lr=0.1),
            rng_rank=0, rng_world_size=2,
        )
        with self.assertRaises(TypeError):
            DistributedDataParallel(trainer, _NoReduce())

        class WrongWorld:
            rank = 0
            world_size = 1

            def all_reduce(self, array, *, op="sum"):
                return array

        with self.assertRaises(ValueError):
            DistributedDataParallel(trainer, WrongWorld())

        class BadRank:
            rank = "0"
            world_size = 2

            def all_reduce(self, array, *, op="sum"):
                return array

        with self.assertRaises(TypeError):
            DistributedDataParallel(trainer, BadRank())

    def test_rejects_bad_reduction_and_trainer(self):
        store = _linear_store()
        trainer = Trainer(
            _linear_graph(4), store, SGD(store.arrays(), lr=0.1),
            rng_rank=0, rng_world_size=2,
        )
        world = LoopbackWorld(2)
        with self.assertRaises(ValueError):
            DistributedDataParallel(
                trainer, world.collective(0), reduction="median"
            )
        with self.assertRaises(TypeError):
            DistributedDataParallel(object(), world.collective(0))


def _norm(grads):
    return float(
        np.sqrt(sum(float(np.sum(np.asarray(value) ** 2)) for value in grads.values()))
    )


class _CountingCollective:
    """包一层 collective：统计调用次数与同时进行的峰值，可选注入网络延迟。"""

    def __init__(self, inner, delay=0.0):
        self.inner = inner
        self.delay = delay
        self.rank = inner.rank
        self.world_size = inner.world_size
        self.calls = 0
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()

    def all_reduce(self, array, *, op="sum"):
        with self._lock:
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            return self.inner.all_reduce(array, op=op)
        finally:
            with self._lock:
                self.active -= 1


def _many_param_graph(count=6, batch=4):
    """``count`` 个同形参数依次相加：足够多的张量才看得出拼桶的效果。"""
    g = Graph("buckets")
    x = g.input("x", f32[batch, 4])
    target = g.input("target", f32[batch, 4])
    y = x
    for i in range(count):
        y = g.add(y, g.input(f"p{i}", f32[4, 4]))
    g.output(mse(g, y, target), "loss")
    return g


class TestGradientBuckets(unittest.TestCase):
    """拼桶：调用次数下降、数值逐位不变、桶之间并发提交。"""

    count = 6

    def _build(self, bucket_bytes, threads=1, delay=0.0):
        graph = _many_param_graph(self.count)
        store = ParameterStore(
            {
                f"p{i}": np.full((4, 4), 0.1 * (i + 1), np.float32)
                for i in range(self.count)
            }
        )
        trainer = Trainer(graph, store, SGD(store.arrays(), lr=0.1))
        collective = _CountingCollective(
            LoopbackWorld(1).collective(0), delay=delay
        )
        ddp = DistributedDataParallel(
            trainer, collective, bucket_bytes=bucket_bytes, threads=threads
        )
        rng = np.random.default_rng(7)
        batch = {
            "x": rng.standard_normal((4, 4)).astype(np.float32),
            "target": rng.standard_normal((4, 4)).astype(np.float32),
        }
        return store, collective, ddp, batch

    def test_bucketing_cuts_collective_calls(self):
        # 每个参数 4×4 f32 = 64 B，128 B 的桶正好装两个 → 6 个参数 3 次调用；
        # 再加损失跨 rank 取均值的那一次，共 4 次。
        store, collective, ddp, batch = self._build(bucket_bytes=128)
        ddp.step(batch)
        self.assertEqual(collective.calls, 4)

    def test_zero_bucket_keeps_one_call_per_tensor(self):
        store, collective, ddp, batch = self._build(bucket_bytes=0)
        ddp.step(batch)
        self.assertEqual(collective.calls, self.count + 1)

    def test_bucketed_result_is_bitwise_identical(self):
        _, _, one_by_one, batch = self._build(bucket_bytes=0)
        _, _, bucketed, _ = self._build(bucket_bytes=128)
        one_by_one.step(batch)
        bucketed.step(batch)
        for name in one_by_one.trainer.store.names:
            np.testing.assert_array_equal(
                bucketed.trainer.store[name], one_by_one.trainer.store[name]
            )

    def test_buckets_are_submitted_concurrently(self):
        """带 20 ms 延迟的桶：三个桶并发提交，峰值必须 > 1。"""
        _, collective, ddp, batch = self._build(
            bucket_bytes=128, threads=3, delay=0.02
        )
        ddp.step(batch)
        self.assertGreaterEqual(collective.peak, 2)


class TestLaunch(unittest.TestCase):
    """真多进程：``launch`` 起两个进程，主机中转归约，结果与单进程对拍。"""

    def test_two_processes_match_single_process(self):
        store = _linear_store()
        single = Trainer(_linear_graph(8), store, SGD(store.arrays(), lr=0.1))
        batch = _linear_batch(8)
        want_losses = [single.step(batch) for _ in range(2)]

        with tempfile.TemporaryDirectory() as folder:
            codes = launch(
                "tests._ddp_worker:train",
                2,
                folder,
                timeout=180.0,
            )
            self.assertEqual(codes, [0, 0])
            payloads = []
            for rank in range(2):
                with open(
                    os.path.join(folder, f"rank{rank}.json"), encoding="utf-8"
                ) as handle:
                    payloads.append(json.load(handle))
        for index, payload in enumerate(payloads):
            self.assertEqual(payload["rank"], index)
            self.assertEqual(payload["world"], 2)
            self.assertEqual(payload["steps"], 2)
            np.testing.assert_allclose(
                payload["losses"], want_losses, rtol=1e-5, atol=1e-6
            )
            for name in store.names:
                np.testing.assert_allclose(
                    np.asarray(payload["params"][name]),
                    store[name],
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=name,
                )
