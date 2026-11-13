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
    ParameterStore,
    Trainer,
    allreduce,
    allreduce_grads,
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


if __name__ == "__main__":
    unittest.main()
