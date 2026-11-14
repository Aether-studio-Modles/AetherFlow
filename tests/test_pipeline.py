"""GPipe 式流水的验收测试。

主判据与数据并行一致：**流水 N 步的参数必须与单卡整图 N 步在 f32 容差内
相同**。这里刻意不要求逐位相等——micro-batch 分别算再相加，矩阵乘内部的
归约顺序与整批一次算不同。

另外要证明的是结构性的三件事：段间激活真的经过了跨卡拷贝、micro-batch
的梯度按 1/份数 加权、一次 step 结束后后端回到调用前的设备。
"""

import unittest

import numpy as np

from aetherflow import Graph, ReferenceBackend, backward, f32
from aetherflow.backends.contracts import DeviceInfo
from aetherflow.train import (
    Adam,
    ParameterStore,
    PipelineStage,
    PipelineTrainer,
    mse,
)


class _FakeDevices:
    """numpy 上的伪多设备后端（与 test_parallel 的同名实现同构）。"""

    def __init__(self, count=2):
        self._count = int(count)
        self._current = 0
        self.switches = []
        self._ref = ReferenceBackend()

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

    def to_array(self, value):
        return np.asarray(value)

    def to_host(self, value):
        return np.asarray(value)

    def execute(self, graph, inputs):
        return self._ref.execute(graph, inputs)


def _store(seed=0):
    rng = np.random.default_rng(seed)
    return ParameterStore(
        {
            "w1": (rng.standard_normal((3, 5)) * 0.2).astype(np.float32),
            "b1": np.zeros((5,), dtype=np.float32),
            "w2": (rng.standard_normal((5, 1)) * 0.2).astype(np.float32),
            "b2": np.zeros((1,), dtype=np.float32),
        }
    )


def _store3(seed=0):
    rng = np.random.default_rng(seed)
    return ParameterStore(
        {
            "w1": (rng.standard_normal((3, 5)) * 0.2).astype(np.float32),
            "b1": np.zeros((5,), dtype=np.float32),
            "w2": (rng.standard_normal((5, 5)) * 0.2).astype(np.float32),
            "b2": np.zeros((5,), dtype=np.float32),
            "w3": (rng.standard_normal((5, 1)) * 0.2).astype(np.float32),
            "b3": np.zeros((1,), dtype=np.float32),
        }
    )


def _batch(total=8, seed=1):
    rng = np.random.default_rng(seed)
    return {
        "x": rng.standard_normal((total, 3)).astype(np.float32),
        "target": rng.standard_normal((total, 1)).astype(np.float32),
    }


def _whole(batch=4):
    g = Graph("whole")
    x = g.input("x", f32[batch, 3])
    w1 = g.input("w1", f32[3, 5])
    b1 = g.input("b1", f32[5])
    w2 = g.input("w2", f32[5, 1])
    b2 = g.input("b2", f32[1])
    target = g.input("target", f32[batch, 1])
    h = g.relu(g.add(g.matmul(x, w1), b1))
    g.output(mse(g, g.add(g.matmul(h, w2), b2), target), "loss")
    return g


def _whole3(batch=4):
    g = Graph("whole3")
    x = g.input("x", f32[batch, 3])
    w1 = g.input("w1", f32[3, 5])
    b1 = g.input("b1", f32[5])
    w2 = g.input("w2", f32[5, 5])
    b2 = g.input("b2", f32[5])
    w3 = g.input("w3", f32[5, 1])
    b3 = g.input("b3", f32[1])
    target = g.input("target", f32[batch, 1])
    h1 = g.relu(g.add(g.matmul(x, w1), b1))
    h2 = g.relu(g.add(g.matmul(h1, w2), b2))
    g.output(mse(g, g.add(g.matmul(h2, w3), b3), target), "loss")
    return g


def _stage0(batch=4):
    g = Graph("stage0")
    x = g.input("x", f32[batch, 3])
    w1 = g.input("w1", f32[3, 5])
    b1 = g.input("b1", f32[5])
    g.output(g.relu(g.add(g.matmul(x, w1), b1)), "h1")
    return g


def _stage_mid(batch=4):
    g = Graph("stage_mid")
    h1 = g.input("h1", f32[batch, 5])
    w2 = g.input("w2", f32[5, 5])
    b2 = g.input("b2", f32[5])
    g.output(g.relu(g.add(g.matmul(h1, w2), b2)), "h2")
    return g


def _stage_last(batch=4, hidden=5, incoming="h", weight="w3", bias="b3"):
    g = Graph("stage_last")
    h = g.input(incoming, f32[batch, hidden])
    w = g.input(weight, f32[hidden, 1])
    b = g.input(bias, f32[1])
    target = g.input("target", f32[batch, 1])
    g.output(mse(g, g.add(g.matmul(h, w), b), target), "loss")
    return g


def _stages(batch=4):
    return [
        PipelineStage(_stage0(batch), params=("w1", "b1"), output="h1"),
        PipelineStage(
            _stage_last(batch, incoming="h1", weight="w2", bias="b2"),
            params=("w2", "b2"),
            inputs=("h1",),
            output="loss",
        ),
    ]


def _run_single_card(graph, store, data, batch, micro, steps):
    """单卡整图跑 ``steps`` 步，作为唯一真值。"""
    opt = Adam(store.arrays(), lr=1e-2)
    bwd = backward(graph, wrt=[graph.find(n) for n in store.names])
    backend = ReferenceBackend()
    per = batch * micro
    for step in range(steps):
        start = step * per
        feed = store.feed(
            {
                "x": data["x"][start : start + per],
                "target": data["target"][start : start + per],
            }
        )
        backend.execute(graph, feed)
        seed = np.asarray(1.0, dtype=np.float32)
        result = backend.execute(bwd, {**feed, "dloss": seed})
        opt.step({name: result["d" + name] for name in store.names})


def _run_pipeline(stages, store, data, batch, micro, steps, devices=2):
    trainer = PipelineTrainer(
        stages,
        store,
        Adam(store.arrays(), lr=1e-2),
        backend=_FakeDevices(devices),
        devices=list(range(devices)),
    )
    total = batch * micro
    for step in range(steps):
        start = step * total
        trainer.step(
            {
                "x": data["x"][start : start + total],
                "target": data["target"][start : start + total],
            },
            micro_batches=micro,
        )
    return trainer


class PipelineNumericsTest(unittest.TestCase):
    def test_two_stages_match_single_card(self):
        batch, micro, steps = 4, 2, 3
        data = _batch(batch * micro * steps)

        reference = _store()
        _run_single_card(_whole(batch * micro), reference, data, batch, micro, steps)

        store = _store()
        _run_pipeline(_stages(batch), store, data, batch, micro, steps)

        for name in reference.names:
            np.testing.assert_allclose(
                store[name], reference[name], rtol=1e-5, atol=1e-6,
                err_msg=f"参数 {name} 与单卡整图不一致",
            )

    def test_three_stages_match_single_card(self):
        batch, micro, steps = 4, 2, 2
        data = _batch(batch * micro * steps)

        reference = _store3()
        _run_single_card(_whole3(batch * micro), reference, data, batch, micro, steps)

        stages = [
            PipelineStage(_stage0(batch), params=("w1", "b1"), output="h1"),
            PipelineStage(_stage_mid(batch), params=("w2", "b2"), inputs=("h1",), output="h2"),
            PipelineStage(
                _stage_last(batch, incoming="h2"), params=("w3", "b3"), inputs=("h2",), output="loss"
            ),
        ]
        store = _store3()
        _run_pipeline(stages, store, data, batch, micro, steps, devices=3)

        for name in reference.names:
            np.testing.assert_allclose(
                store[name], reference[name], rtol=1e-5, atol=1e-6,
                err_msg=f"参数 {name} 与单卡整图不一致",
            )

    def test_single_stage_is_degenerate_but_valid(self):
        store = _store()
        trainer = PipelineTrainer(
            [
                PipelineStage(
                    _whole(4),
                    params=("w1", "b1", "w2", "b2"),
                    output="loss",
                )
            ],
            store,
            Adam(store.arrays(), lr=1e-2),
            backend=_FakeDevices(1),
            devices=[0],
        )
        before = {name: store[name].copy() for name in store.names}
        trainer.step(_batch(4))
        for name in store.names:
            self.assertFalse(np.array_equal(before[name], store[name]))

    def test_micro_batch_one_matches_single_card(self):
        batch, micro, steps = 4, 1, 2
        data = _batch(batch * steps)
        reference = _store()
        _run_single_card(_whole(batch * micro), reference, data, batch, micro, steps)
        store = _store()
        _run_pipeline(_stages(batch), store, data, batch, micro, steps)
        for name in reference.names:
            np.testing.assert_allclose(store[name], reference[name], rtol=1e-5, atol=1e-6)


class PipelineValidationTest(unittest.TestCase):
    def _trainer(self, stages, devices=(0, 1), store=None, **kwargs):
        store = _store() if store is None else store
        return PipelineTrainer(
            stages,
            store,
            Adam(store.arrays(), lr=1e-2),
            backend=_FakeDevices(len(devices)),
            devices=list(devices),
            **kwargs,
        )

    def test_device_count_must_match_stage_count(self):
        with self.assertRaises(ValueError):
            self._trainer(self._stages(), devices=(0,))

    def _stages(self):
        return [
            PipelineStage(_stage0(4), params=("w1", "b1"), output="h1"),
            PipelineStage(
                _stage_last(4, incoming="h1", weight="w2", bias="b2"), params=("w2", "b2"), inputs=("h1",), output="loss"
            ),
        ]

    def test_input_name_must_match_previous_output(self):
        stages = list(self._stages())
        stages[1] = PipelineStage(
            _stage_last(4, incoming="h1", weight="w2", bias="b2"), params=("w2", "b2"), inputs=("wrong",), output="loss"
        )
        with self.assertRaises(ValueError):
            self._trainer(stages)

    def test_first_stage_must_not_have_input(self):
        stages = [
            PipelineStage(_stage0(4), params=("w1", "b1"), inputs=("h1",), output="h1"),
            PipelineStage(
                _stage_last(4, incoming="h1", weight="w2", bias="b2"), params=("w2", "b2"), inputs=("h1",), output="loss"
            ),
        ]
        with self.assertRaises(ValueError):
            self._trainer(stages)

    def test_parameter_must_belong_to_exactly_one_stage(self):
        """同名参数出现在两段：第二段必须当场拒绝，而不是两份槽位各更各的。"""
        g = Graph("dup")
        h1 = g.input("h1", f32[4, 5])
        w1 = g.input("w1", f32[5, 1])
        b = g.input("b", f32[1])
        target = g.input("target", f32[4, 1])
        g.output(mse(g, g.add(g.matmul(h1, w1), b), target), "loss")
        stages = [
            PipelineStage(_stage0(4), params=("w1", "b1"), output="h1"),
            PipelineStage(g, params=("w1", "b"), inputs=("h1",), output="loss"),
        ]
        with self.assertRaises(ValueError):
            self._trainer(stages)

    def test_store_and_stages_must_cover_the_same_names(self):
        store = ParameterStore(
            {
                "w1": np.zeros((3, 5), np.float32),
                "b1": np.zeros((5,), np.float32),
            }
        )
        with self.assertRaises(ValueError):
            self._trainer(self._stages(), store=store)

    def test_loss_must_be_scalar(self):
        g = Graph("bad_loss")
        h = g.input("h", f32[4, 5])
        w3 = g.input("w3", f32[5, 1])
        b3 = g.input("b3", f32[1])
        g.output(g.add(g.matmul(h, w3), b3), "loss")
        stages = [
            PipelineStage(_stage0(4), params=("w1", "b1"), output="h1"),
            PipelineStage(g, params=("w3", "b3"), inputs=("h1",), output="loss"),
        ]
        with self.assertRaises(ValueError):
            self._trainer(stages)

    def test_stage_graph_cannot_have_extra_inputs(self):
        stages = [
            PipelineStage(_stage0(4), params=("w1", "b1"), output="h1"),
            PipelineStage(
                _whole(4), params=("w2", "b2"), inputs=("h1",), output="loss"
            ),
        ]
        # _whole 的输入是 x/w1/b1/w2/b2/target，与"参数 + 段间输入"不符
        with self.assertRaises(ValueError):
            self._trainer(stages)

    def test_micro_batches_must_divide_batch(self):
        trainer = self._trainer(self._stages())
        with self.assertRaises(ValueError):
            trainer.step(_batch(9), micro_batches=2)

    def test_bad_micro_batches_is_refused(self):
        trainer = self._trainer(self._stages())
        for value in (0, -1):
            with self.assertRaises(ValueError):
                trainer.step(_batch(8), micro_batches=value)
        for value in (True, 2.0):
            with self.assertRaises(TypeError):
                trainer.step(_batch(8), micro_batches=value)

    def test_batch_shape_must_match_static_declaration(self):
        store = _store()
        trainer = self._trainer(
            [
                PipelineStage(_stage0(2), params=("w1", "b1"), output="h1"),
                PipelineStage(
                    _stage_last(2, incoming="h1", weight="w2", bias="b2"),
                    params=("w2", "b2"),
                    inputs=("h1",),
                    output="loss",
                ),
            ],
            store=store,
        )
        with self.assertRaises(ValueError):
            trainer.step(_batch(8), micro_batches=2)

    def test_missing_data_input_is_refused(self):
        trainer = self._trainer(self._stages())
        batch = _batch(8)
        batch.pop("target")
        with self.assertRaises(KeyError):
            trainer.step(batch, micro_batches=2)

    def test_device_is_restored_after_step(self):
        backend = _FakeDevices(2)
        backend.set_device(1)
        store = _store()
        trainer = PipelineTrainer(
            self._stages(),
            store,
            Adam(store.arrays(), lr=1e-2),
            backend=backend,
            devices=[0, 1],
        )
        trainer.step(_batch(8), micro_batches=2)
        self.assertEqual(backend.current_device(), 1)

    def test_activation_crosses_devices(self):
        backend = _FakeDevices(2)
        store = _store()
        trainer = PipelineTrainer(
            self._stages(),
            store,
            Adam(store.arrays(), lr=1e-2),
            backend=backend,
            devices=[0, 1],
        )
        trainer.step(_batch(4))
        self.assertIn(0, backend.switches)
        self.assertIn(1, backend.switches)


class PipelineScheduleTest(unittest.TestCase):
    def _stages3(self, batch=4):
        return [
            PipelineStage(_stage0(batch), params=("w1", "b1"), output="h1"),
            PipelineStage(
                _stage_mid(batch), params=("w2", "b2"), inputs=("h1",), output="h2"
            ),
            PipelineStage(
                _stage_last(batch, incoming="h2"),
                params=("w3", "b3"),
                inputs=("h2",),
                output="loss",
            ),
        ]

    def _make(self, stages, store, devices, schedule):
        return PipelineTrainer(
            stages,
            store,
            Adam(store.arrays(), lr=1e-2),
            backend=_FakeDevices(devices),
            devices=list(range(devices)),
            schedule=schedule,
        )

    def test_1f1b_matches_gpipe(self):
        data = _batch(32)
        stores = {}
        for schedule in ("gpipe", "1f1b", "sequential"):
            store = _store3()
            trainer = self._make(self._stages3(), store, 3, schedule)
            for step in range(2):
                trainer.step(
                    {
                        "x": data["x"][step * 16 : step * 16 + 16],
                        "target": data["target"][step * 16 : step * 16 + 16],
                    },
                    micro_batches=4,
                )
            stores[schedule] = store
        for name in stores["gpipe"].names:
            for schedule in ("1f1b", "sequential"):
                np.testing.assert_allclose(
                    stores[schedule][name],
                    stores["gpipe"][name],
                    rtol=1e-6,
                    atol=1e-7,
                    err_msg=(
                        f"{schedule} 与 gpipe 的 {name} 不一致："
                        "换的是调度顺序，不该换数值"
                    ),
                )

    def test_sequential_peak_is_one(self):
        store = _store3()
        trainer = self._make(self._stages3(), store, 3, "sequential")
        trainer.step(_batch(16), micro_batches=4)
        self.assertEqual(trainer.last_resident_peak, 1)

    def test_gpipe_peak_equals_micro_batches(self):
        store = _store3()
        trainer = self._make(self._stages3(), store, 3, "gpipe")
        trainer.step(_batch(16), micro_batches=4)
        self.assertEqual(trainer.last_resident_peak, 4)

    def test_1f1b_peak_is_stages_minus_one(self):
        store = _store3()
        trainer = self._make(self._stages3(), store, 3, "1f1b")
        trainer.step(_batch(16), micro_batches=4)
        # 三段、四份：预热两份，稳态里始终只有两份"前向已完、反向未做"
        self.assertEqual(trainer.last_resident_peak, 2)

    def test_1f1b_peak_capped_by_micro_batches(self):
        store = _store3()
        trainer = self._make(self._stages3(), store, 3, "1f1b")
        trainer.step(_batch(4), micro_batches=1)
        # 只有一份时预热 0 份：前向完立刻反向，驻留峰值就是 1
        self.assertEqual(trainer.last_resident_peak, 1)

    def test_single_stage_1f1b(self):
        store = _store()
        trainer = PipelineTrainer(
            [
                PipelineStage(
                    _whole(4),
                    params=("w1", "b1", "w2", "b2"),
                    output="loss",
                )
            ],
            store,
            Adam(store.arrays(), lr=1e-2),
            backend=_FakeDevices(1),
            devices=[0],
            schedule="1f1b",
        )
        before = {name: store[name].copy() for name in store.names}
        trainer.step(_batch(8), micro_batches=2)
        self.assertEqual(trainer.last_resident_peak, 1)
        for name in store.names:
            self.assertFalse(np.array_equal(before[name], store[name]))

    def test_unknown_schedule_is_refused(self):
        store = _store()
        with self.assertRaises(ValueError):
            self._make(_stages(), store, 2, "interleaved")


if __name__ == "__main__":
    unittest.main()
