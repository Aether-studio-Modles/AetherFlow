"""多卡闭环的验收测试：跨卡拷贝、主机侧归约、数据并行一步。

判据不是"接口能调用"，而是：

1. **数值等价**：数据并行 N 步的损失与参数，必须与"单卡跑合并 batch"一致。
   只测形状发现不了"梯度忘了平均""某块卡少喂一路输入"这类缺陷。
2. **非法参数当场拒绝**：越界 / 重复的设备索引、batch 不能被设备数整除、
   数据输入多传少传、梯度名字分叉，都不能留到执行期或静默丢弃。
3. **设备状态还原**：一次 step 结束后，后端仍停在调用前的设备上。

CPU 侧用 numpy 上的伪多设备后端（数组就是 numpy，"设备"只记录索引）；
CUDA 真机侧用 3 × Tesla M40 的前两块卡对拍。
"""

import unittest

import numpy as np

from aetherflow import Graph, ReferenceBackend, f32
from aetherflow.backends.contracts import DeviceInfo
from aetherflow.backends.cuda import CUDA_AVAILABLE, CudaBackend
from aetherflow.train import (
    SGD,
    DataParallelTrainer,
    DeviceCollectiveUnavailable,
    ParameterStore,
    Trainer,
    allreduce,
    allreduce_grads,
    collective_support,
    copy_across_devices,
    init_array,
    mse,
)


class _FakeDevices:
    """numpy 上的伪多设备后端：只记录"现在用哪块卡"。

    数组就是 numpy 数组，``to_array`` / ``to_host`` 是同一性转换。这样数据
    并行的切分 / 归约 / 写回逻辑可以在没有 GPU 的机器上逐位验证。
    """

    def __init__(self, count=2):
        self._count = int(count)
        self._current = 0
        self.switches = []
        self._ref = ReferenceBackend()

    # -- DeviceManagerContract --

    def enumerate_devices(self):
        return tuple(
            DeviceInfo(name=f"fake{index}", runtime=1, total_memory_bytes=1 << 20)
            for index in range(self._count)
        )

    def current_device(self):
        return self._current

    def set_device(self, index):
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"设备索引必须是 int，收到 {index!r}")
        if not 0 <= index < self._count:
            raise ValueError(f"设备索引 {index} 越界")
        self._current = index
        self.switches.append(index)

    # -- 数组与执行 --

    def to_array(self, value):
        return np.asarray(value)

    def to_host(self, value):
        return np.asarray(value)

    def execute(self, graph, inputs):
        return self._ref.execute(graph, inputs)


def _linear_graph(batch=8):
    """``mse(x @ w + b, target)``：静态形状、标量损失、两个参数。

    第 0 维是**每卡** batch；数据并行时总 batch 是它的 ``world_size`` 倍。
    """
    g = Graph("dp")
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


class _FakeComm:
    """假 communicator：只记录调用，不改数据。

    跨卡求和是 NCCL 的职责，不在 provider 的逻辑里；这里要验证的是
    provider 发起了几次 ``sum``、是否在设备侧补了 mean 的除法，以及设备
    切换是否归位。
    """

    def __init__(self):
        self.calls = []
        self.destroyed = False

    def all_reduce(self, array, *, op="sum", out=None, stream=None):
        self.calls.append((op, stream))
        return array

    def destroy(self):
        self.destroyed = True


class TestNcclDeviceCollective(unittest.TestCase):
    """内置 provider（``ncclCommInitAll``）：原地归约、mean 在设备侧补除。"""

    def _make(self, count=2):
        from unittest import mock

        from aetherflow.train import parallel

        backend = _FakeDevices(count)
        comms = [_FakeComm() for _ in range(count)]
        with mock.patch("aetherflow.train.nccl.init_all", return_value=comms):
            collective = parallel.nccl_device_collective(backend, list(range(count)))
        return backend, comms, collective

    def _grads(self):
        return {
            0: {
                "w": np.array([1.0, 2.0], np.float32),
                "b": np.array([0.5], np.float32),
            },
            1: {
                "w": np.array([3.0, 6.0], np.float32),
                "b": np.array([1.5], np.float32),
            },
        }

    def test_mean_is_applied_on_device(self):
        backend, comms, collective = self._make()
        grads = self._grads()
        reduced = collective(grads, 0, reduction="mean")
        np.testing.assert_allclose(reduced["w"], [0.5, 1.0])
        np.testing.assert_allclose(reduced["b"], [0.25])
        np.testing.assert_allclose(grads[1]["w"], [1.5, 3.0])
        self.assertEqual([c.calls for c in comms], [[("sum", 0)] * 2] * 2)
        # 归约结束后回到调用前的设备
        self.assertEqual(backend.current_device(), 0)

    def test_sum_keeps_scale(self):
        _, _, collective = self._make()
        reduced = collective(self._grads(), 0, reduction="sum")
        np.testing.assert_allclose(reduced["w"], [1.0, 2.0])
        np.testing.assert_allclose(reduced["b"], [0.5])

    def test_rejects_unknown_reduction(self):
        _, _, collective = self._make()
        with self.assertRaises(ValueError):
            collective(self._grads(), 0, reduction="avg")

    def test_close_destroys_every_comm(self):
        _, comms, collective = self._make()
        collective.close()
        self.assertTrue(all(comm.destroyed for comm in comms))
        self.assertEqual(collective._comms, [])

    def test_missing_library_is_explicit(self):
        from aetherflow.train import nccl, nccl_device_collective

        if nccl.available():
            self.skipTest("本机可加载 NCCL / RCCL，这条只验证缺库时的拒绝面")
        with self.assertRaises(nccl.NcclUnavailable):
            nccl.init_all([0])
        with self.assertRaises(nccl.NcclUnavailable):
            nccl_device_collective(_FakeDevices(1), [0])


class TestAllreduce(unittest.TestCase):
    """主机侧归约：数值、顺序、以及三种不一致都必须报错。"""

    def test_mean_and_sum(self):
        first = np.array([1.0, 2.0], dtype=np.float32)
        second = np.array([3.0, 6.0], dtype=np.float32)
        np.testing.assert_allclose(allreduce([first, second]), [2.0, 4.0])
        np.testing.assert_allclose(
            allreduce([first, second], reduction="sum"), [4.0, 8.0]
        )
        # 输入不被改写
        np.testing.assert_allclose(first, [1.0, 2.0])

    def test_single_array_is_identity(self):
        value = np.array([[1.5, -2.0]], dtype=np.float32)
        np.testing.assert_array_equal(allreduce([value]), value)

    def test_rejects_bad_reduction(self):
        with self.assertRaises(ValueError):
            allreduce([np.zeros(1)], reduction="max")

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            allreduce([])

    def test_rejects_shape_and_dtype_mismatch(self):
        with self.assertRaises(ValueError):
            allreduce([np.zeros(2, np.float32), np.zeros(3, np.float32)])
        with self.assertRaises(TypeError):
            allreduce([np.zeros(2, np.float32), np.zeros(2, np.float64)])

    def test_rejects_integer_arrays(self):
        with self.assertRaises(TypeError):
            allreduce([np.zeros(2, np.int32), np.zeros(2, np.int32)])

    def test_grads_require_same_names(self):
        grads = allreduce_grads(
            {
                0: {"w": np.ones(2, np.float32), "b": np.zeros(1, np.float32)},
                1: {"w": np.full(2, 3.0, np.float32), "b": np.full(1, 2.0, np.float32)},
            }
        )
        self.assertEqual(list(grads), ["w", "b"])
        np.testing.assert_allclose(grads["w"], [2.0, 2.0])
        np.testing.assert_allclose(grads["b"], [1.0])
        with self.assertRaises(KeyError):
            allreduce_grads({0: {"w": np.zeros(1, np.float32)}, 1: {"b": np.zeros(1, np.float32)}})
        with self.assertRaises(TypeError):
            allreduce_grads({0: {"w": np.zeros(1, np.float32)}, True: {"w": np.zeros(1, np.float32)}})
        with self.assertRaises(ValueError):
            allreduce_grads({})


class TestCopyAcrossDevices(unittest.TestCase):
    """跨卡拷贝：值一致、切卡成对、目标卡与源卡由调用方决定。"""

    def test_copies_value_and_restores_device(self):
        backend = _FakeDevices(3)
        backend.set_device(2)
        backend.switches.clear()
        value = np.arange(6, dtype=np.float32).reshape(2, 3)
        out = copy_across_devices(backend, value, 1)
        np.testing.assert_array_equal(out, value)
        self.assertEqual(backend.current_device(), 2)
        self.assertEqual(backend.switches, [1, 2])

    def test_same_device_is_noop(self):
        backend = _FakeDevices(2)
        value = np.arange(4, dtype=np.float32)
        self.assertIs(copy_across_devices(backend, value, 0), value)
        self.assertEqual(backend.switches, [])

    def test_rejects_bad_targets(self):
        backend = _FakeDevices(2)
        value = np.zeros(1, np.float32)
        for bad in (2, -1):
            with self.assertRaises(ValueError):
                copy_across_devices(backend, value, bad)
        for bad in (True, "1", 1.0):
            with self.assertRaises(TypeError):
                copy_across_devices(backend, value, bad)
        with self.assertRaises(TypeError):
            copy_across_devices(backend, value, 1, src_device="0")

    def test_requires_device_manager(self):
        with self.assertRaises(TypeError):
            copy_across_devices(ReferenceBackend(), np.zeros(1, np.float32), 0)


class TestDataParallelTrainer(unittest.TestCase):
    """两台"设备"上的两步，必须与单卡合并 batch 的两步逐位接近。"""

    def _compare_two_steps(self, reduction="mean"):
        total, world = 8, 2
        batch = _linear_batch(total)

        single = _linear_store()
        single_opt = SGD(single.arrays(), lr=0.1)
        single_trainer = Trainer(_linear_graph(total), single, single_opt)
        single_losses = [single_trainer.step(batch) for _ in range(2)]

        multi_backend = _FakeDevices(world)
        multi = _linear_store()
        multi_opt = SGD(multi.arrays(), lr=0.1)
        multi_trainer = DataParallelTrainer(
            _linear_graph(total // world),
            multi,
            multi_opt,
            backend=multi_backend,
            devices=list(range(world)),
            reduction=reduction,
        )
        multi_losses = [multi_trainer.step(batch) for _ in range(2)]
        return single, single_losses, multi, multi_losses, multi_backend, multi_trainer

    def test_matches_single_card_merged_batch(self):
        single, single_losses, multi, multi_losses, backend, trainer = (
            self._compare_two_steps()
        )
        np.testing.assert_allclose(multi_losses, single_losses, rtol=1e-6, atol=1e-7)
        for name in single.names:
            np.testing.assert_allclose(
                multi[name], single[name], rtol=1e-6, atol=1e-7, err_msg=name
            )
        self.assertEqual(trainer.steps, 2)
        # 每步两块卡都会被切到，且结束后回到调用前的设备
        self.assertEqual(backend.current_device(), 0)
        self.assertIn(1, backend.switches)
        self.assertEqual(trainer.backend, backend)

    def test_uses_every_device(self):
        backend = _FakeDevices(3)
        graph = _linear_graph(4)
        store = _linear_store()
        trainer = DataParallelTrainer(
            graph, store, SGD(store.arrays(), lr=0.1), backend=backend, devices=[0, 2]
        )
        batch = _linear_batch()
        trainer.step(batch)
        self.assertEqual(set(backend.switches), {0, 2})

    def test_rejects_indivisible_batch(self):
        graph = _linear_graph(8)
        store = _linear_store()
        trainer = DataParallelTrainer(
            graph, store, SGD(store.arrays(), lr=0.1),
            backend=_FakeDevices(3), devices=[0, 1, 2],
        )
        with self.assertRaises(ValueError):
            trainer.step(_linear_batch(8))

    def test_rejects_graph_not_per_rank_batch(self):
        """图声明的是每卡 batch；按总 batch 构图时当场拒绝并说明原因。"""
        graph = _linear_graph(8)
        store = _linear_store()
        trainer = DataParallelTrainer(
            graph, store, SGD(store.arrays(), lr=0.1),
            backend=_FakeDevices(2), devices=[0, 1],
        )
        with self.assertRaises(ValueError):
            trainer.step(_linear_batch(8))

    def test_rejects_bad_data_names(self):
        graph = _linear_graph()
        store = _linear_store()
        trainer = DataParallelTrainer(
            graph, store, SGD(store.arrays(), lr=0.1),
            backend=_FakeDevices(2), devices=[0, 1],
        )
        batch = _linear_batch()
        with self.assertRaises(KeyError):
            trainer.step({k: v for k, v in batch.items() if k != "target"})
        with self.assertRaises(KeyError):
            trainer.step({**batch, "extra": np.zeros(8, np.float32)})
        with self.assertRaises(KeyError):
            trainer.step({**batch, "w": np.zeros((8, 3), np.float32)})

    def test_rejects_inconsistent_first_dim(self):
        graph = _linear_graph()
        store = _linear_store()
        trainer = DataParallelTrainer(
            graph, store, SGD(store.arrays(), lr=0.1),
            backend=_FakeDevices(2), devices=[0, 1],
        )
        batch = _linear_batch()
        batch["target"] = batch["target"][:4]
        with self.assertRaises(ValueError):
            trainer.step(batch)

    def test_rejects_bad_device_list(self):
        graph = _linear_graph()
        store = _linear_store()
        for devices, error in (
            ([], ValueError),
            ([0, 0], ValueError),
            ([0, 2], ValueError),
            ([-1], ValueError),
            ([0, True], TypeError),
            ("01", TypeError),
        ):
            with self.assertRaises(error):
                DataParallelTrainer(
                    graph, store, SGD(store.arrays(), lr=0.1),
                    backend=_FakeDevices(2), devices=devices,
                )

    def test_rejects_unsupported_backend_and_reduction(self):
        graph = _linear_graph()
        store = _linear_store()
        with self.assertRaises(TypeError):
            DataParallelTrainer(
                graph, store, SGD(store.arrays(), lr=0.1),
                backend=ReferenceBackend(), devices=[0],
            )
        with self.assertRaises(ValueError):
            DataParallelTrainer(
                graph, store, SGD(store.arrays(), lr=0.1),
                backend=_FakeDevices(2), devices=[0, 1], reduction="max",
            )

    def test_rejects_non_scalar_target(self):
        g = Graph("vector")
        x = g.input("x", f32[4, 2])
        w = g.input("w", f32[2, 2])
        g.output(g.matmul(x, w), "y")
        store = ParameterStore({"w": init_array("w", (2, 2))})
        with self.assertRaises(ValueError):
            DataParallelTrainer(
                g, store, SGD(store.arrays(), lr=0.1),
                backend=_FakeDevices(2), devices=[0, 1],
            )


@unittest.skipUnless(CUDA_AVAILABLE, "需要 cupy 与 CUDA 设备")
class TestNcclProviderOnCuda(unittest.TestCase):
    """内置 provider 的真机面：库可加载时能建 communicator 并原地归约。

    world=1 只验证"能建、能调、能归位"（归约是恒等）；跨卡数值在
    ``TestCudaDataParallel`` 的两卡用例里（需要至少 2 块可见设备）。
    """

    def setUp(self):
        from aetherflow.train import nccl

        if not nccl.available():
            self.skipTest("没有可加载的 NCCL")
        import cupy as cp

        self.origin = int(cp.cuda.Device().id)

    def tearDown(self):
        import cupy as cp

        cp.cuda.Device(self.origin).use()

    def test_world_one_is_identity(self):
        from aetherflow.train import nccl_device_collective

        backend = CudaBackend(device=0)
        with nccl_device_collective(backend, [0]) as collective:
            host = np.array([1.0, 2.0], np.float32)
            for reduction in ("sum", "mean"):
                value = backend.to_array(host)
                out = collective({0: {"w": value}}, 0, reduction=reduction)
                np.testing.assert_allclose(backend.to_host(out["w"]), host)
        self.assertEqual(backend.current_device(), self.origin)


@unittest.skipUnless(CUDA_AVAILABLE, "需要 cupy 与 CUDA 设备")
class TestCudaDataParallel(unittest.TestCase):
    """真机 2 卡：跨卡拷贝逐位一致，数据并行两步与单卡合并 batch 对齐。"""

    @classmethod
    def setUpClass(cls):
        import cupy as cp

        cls.cp = cp
        if int(cp.cuda.runtime.getDeviceCount()) < 2:
            raise unittest.SkipTest("多卡用例需要至少 2 块可见设备")

    def setUp(self):
        self.origin = int(self.cp.cuda.Device().id)

    def tearDown(self):
        self.cp.cuda.Device(self.origin).use()

    def test_copies_array_bitwise_between_cards(self):
        backend = CudaBackend(device=0)
        host = np.arange(24, dtype=np.float32).reshape(4, 6) / 7.0
        source = backend.to_array(host)
        copied = copy_across_devices(backend, source, 1, src_device=0)
        self.assertEqual(int(copied.device.id), 1)
        np.testing.assert_array_equal(backend.to_host(copied), host)
        self.assertEqual(backend.current_device(), self.origin)

    def test_two_steps_match_single_card(self):
        total, world = 8, 2
        batch = _linear_batch(total, seed=3)

        single_store = _linear_store()
        single_opt = SGD(single_store.arrays(), lr=0.1)
        single = Trainer(
            _linear_graph(total), single_store, single_opt, backend=CudaBackend(device=0)
        )
        single_losses = [single.step(batch) for _ in range(2)]

        multi_store = _linear_store()
        multi_opt = SGD(multi_store.arrays(), lr=0.1)
        multi = DataParallelTrainer(
            _linear_graph(total // world),
            multi_store,
            multi_opt,
            backend=CudaBackend(device=0),
            devices=[0, 1],
        )
        multi_losses = [multi.step(batch) for _ in range(2)]

        np.testing.assert_allclose(multi_losses, single_losses, rtol=1e-4, atol=1e-6)
        for name in single_store.names:
            np.testing.assert_allclose(
                multi_store[name], single_store[name],
                rtol=1e-4, atol=1e-6, err_msg=name,
            )
        self.assertEqual(multi.steps, 2)

    def test_nccl_provider_reduces_across_two_cards(self):
        from aetherflow.train import nccl, nccl_device_collective

        if not nccl.available():
            self.skipTest("没有可加载的 NCCL")
        backend = CudaBackend(device=0)
        with nccl_device_collective(backend, [0, 1]) as collective:
            left = backend.to_array(np.array([1.0, 2.0], np.float32))
            backend.set_device(1)
            right = backend.to_array(np.array([3.0, 4.0], np.float32))
            backend.set_device(0)
            out = collective({0: {"w": left}, 1: {"w": right}}, 0, reduction="sum")
            np.testing.assert_allclose(backend.to_host(out["w"]), [4.0, 6.0])
            np.testing.assert_allclose(backend.to_host(right), [4.0, 6.0])

            left = backend.to_array(np.ones(2, np.float32))
            backend.set_device(1)
            right = backend.to_array(np.full(2, 3.0, np.float32))
            backend.set_device(0)
            out = collective({0: {"w": left}, 1: {"w": right}}, 0, reduction="mean")
            np.testing.assert_allclose(backend.to_host(out["w"]), [2.0, 2.0])
        self.assertEqual(backend.current_device(), self.origin)


def _mean_provider(calls):
    """一个"设备侧"归约的替身：按 allreduce 的累加顺序求均值。

    真实的 NCCL / RCCL 由调用方注入；这里只验证契约与接线——数值必须与主机
    路径逐位相同，否则说明两条路径的累加顺序变了。
    """

    def provider(per_device, target, *, reduction="mean"):
        calls.append((tuple(per_device), target, reduction))
        ranks = list(per_device)
        names = list(per_device[ranks[0]])
        out = {}
        for name in names:
            total = np.array(np.asarray(per_device[ranks[0]][name]), copy=True)
            for rank in ranks[1:]:
                total += np.asarray(per_device[rank][name])
            if reduction == "mean":
                total /= len(ranks)
            out[name] = total
        return out

    provider.name = "fake-nccl"
    return provider


class TestCollectiveSeam(unittest.TestCase):
    """AF-33 补：设备侧集合通信的能力声明与接线。

    判据是"声明不撒谎"：没有实现时查询得到 False 且带原因，请求设备路径
    当场抛能力错误；注入 provider 后数值与主机路径逐位一致，且只把结果
    拉回主机一次。
    """

    def test_support_declares_host_only_without_provider(self):
        """没有 provider 时：设备归约不可用、重叠不可用，原因点名 NCCL / RCCL。

        5.244 起理由按本机真实加载到的库命名：HIP 机器上是 RCCL，CUDA 机器
        上是 NCCL，两套都加载不到时是「NCCL / RCCL」。判据只要求点名，不
        要求点名哪一家——写死一家在另一种卡上必红。
        """
        support = collective_support(_FakeDevices(2))
        self.assertFalse(support.device_allreduce)
        self.assertFalse(support.overlap)
        self.assertIsNone(support.provider)
        self.assertRegex(support.reason, "NCCL|RCCL")
        self.assertIn("重叠", support.reason)

    def test_support_reflects_injected_provider(self):
        """注入 provider 后设备归约声明为可用，但重叠仍然如实为不可用。"""
        support = collective_support(_FakeDevices(2), provider=_mean_provider([]))
        self.assertTrue(support.device_allreduce)
        self.assertEqual(support.provider, "fake-nccl")
        self.assertFalse(support.overlap)
        self.assertIn("重叠", support.reason)
        with self.assertRaisesRegex(TypeError, "provider"):
            collective_support(_FakeDevices(2), provider=3)

    def test_device_mode_without_provider_is_explicit(self):
        """请求设备路径但没有实现：抛能力错误并带原因，不静默退回主机。"""
        with self.assertRaisesRegex(DeviceCollectiveUnavailable, "NCCL|RCCL"):
            DataParallelTrainer(
                _linear_graph(),
                _linear_store(),
                SGD(_linear_store().arrays(), lr=0.1),
                backend=_FakeDevices(2),
                devices=[0, 1],
                collective="device",
            )

    def test_inconsistent_collective_arguments_are_rejected(self):
        """非法 collective 取值、注入了 provider 却选 host、provider 不可调用。"""
        store = _linear_store()
        with self.assertRaisesRegex(ValueError, "collective"):
            DataParallelTrainer(
                _linear_graph(),
                store,
                SGD(store.arrays(), lr=0.1),
                backend=_FakeDevices(2),
                devices=[0, 1],
                collective="nccl",
            )
        store = _linear_store()
        with self.assertRaisesRegex(ValueError, "device_collective"):
            DataParallelTrainer(
                _linear_graph(),
                store,
                SGD(store.arrays(), lr=0.1),
                backend=_FakeDevices(2),
                devices=[0, 1],
                device_collective=_mean_provider([]),
            )
        store = _linear_store()
        with self.assertRaisesRegex(TypeError, "device_collective"):
            DataParallelTrainer(
                _linear_graph(),
                store,
                SGD(store.arrays(), lr=0.1),
                backend=_FakeDevices(2),
                devices=[0, 1],
                collective="device",
                device_collective=3,
            )

    def test_device_path_matches_host_path(self):
        """两条归约路径的损失与参数逐位相同，provider 每步只被调用一次。"""
        calls = []
        host_store = _linear_store()
        device_store = _linear_store()
        host = DataParallelTrainer(
            _linear_graph(batch=4),
            host_store,
            SGD(host_store.arrays(), lr=0.1),
            backend=_FakeDevices(2),
            devices=[0, 1],
        )
        device = DataParallelTrainer(
            _linear_graph(batch=4),
            device_store,
            SGD(device_store.arrays(), lr=0.1),
            backend=_FakeDevices(2),
            devices=[0, 1],
            collective="device",
            device_collective=_mean_provider(calls),
        )
        for step in range(2):
            batch = _linear_batch(seed=step)
            self.assertEqual(device.step(batch), host.step(batch))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1], 0)  # 归约目标是第一块卡
        self.assertEqual(calls[0][2], "mean")
        for name in host_store.names:
            np.testing.assert_array_equal(device_store[name], host_store[name])

    def test_provider_result_is_validated(self):
        """provider 返回值是外部输入：名字、形状、类型不对都当场报错。"""
        cases = (
            (lambda per_device, target, **kw: [1, 2], TypeError),
            (lambda per_device, target, **kw: {"w": np.zeros((3, 1), np.float32)}, KeyError),
            (
                lambda per_device, target, **kw: {
                    name: np.zeros((1, 1), np.float32) for name in ("w", "b")
                },
                ValueError,
            ),
        )
        for provider, error in cases:
            with self.subTest(error=error.__name__):
                store = _linear_store()
                trainer = DataParallelTrainer(
                    _linear_graph(batch=4),
                    store,
                    SGD(store.arrays(), lr=0.1),
                    backend=_FakeDevices(2),
                    devices=[0, 1],
                    collective="device",
                    device_collective=provider,
                )
                # 不只断言异常类型：优化器本身也会对缺名字 / 形状不符报错，
                # 那种"碰巧也是 KeyError"的通过说明不了归约结果被校验过。
                with self.assertRaises(error) as caught:
                    trainer.step(_linear_batch())
                self.assertIn("device_collective", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
