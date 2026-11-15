"""设备端循环的判据：捕获一次、状态留在显存、语义与宿主路径逐位一致。

离线替身 ``_NumpyDevice`` 把 numpy 数组当"设备数组"：它不碰显存，但把
``capture`` / ``to_array`` / ``to_host`` 的调用次数如实记账——本层的两个
主张（"循环只捕获两次"与"每轮只有谓词回主机"）都靠这份账来判。
"""

import unittest

import numpy as np

from aetherflow import (
    Conditional,
    ControlFlowError,
    Graph,
    GraphRegion,
    ReferenceBackend,
    WhileLoop,
    f32,
)
from aetherflow.runtime import DeviceLoopError, DeviceWhileLoop

try:  # 真机部分：没有 ROCm 或设备不可枚举时整组跳过
    from aetherflow.backends.hip import HIP_AVAILABLE, HipBackend
except Exception:  # noqa: BLE001 - 库缺失、驱动缺失都按"不可用"处理
    HIP_AVAILABLE, HipBackend = False, None


def _scalar(name="x"):
    return f32[()]


def _counter_condition(graph_name="while_cond"):
    """``4.5 - x > 0``，即 x < 4.5。"""
    g = Graph(graph_name)
    x = g.input("x", _scalar())
    limit = g.constant(4.5)
    g.output(g.step(g.sub(limit, x), threshold=0.0), "flag")
    return g


def _counter_body(graph_name="while_body"):
    """x_next = x + step；``step`` 是自由输入。"""
    g = Graph(graph_name)
    x = g.input("x", _scalar())
    step = g.input("step", _scalar())
    g.output(g.add(x, step), "x_next")
    return g


def _loop(max_steps=16):
    return WhileLoop(
        _counter_condition(),
        _counter_body(),
        carried={"x": "x_next"},
        max_steps=max_steps,
    )


def _feed(value):
    return np.asarray(value, dtype=np.float32)


class _Dev:
    """替身的"设备数组"：与 numpy 数组刻意区分，好让上传次数可判。

    真设备上 ``to_array`` 只该在"从主机来"时发生；如果替身直接用 numpy
    数组，``_assign`` 分不清"循环体刚产出的设备数组"与"调用方传进来的主机
    数组"，账就记错了。
    """

    def __init__(self, value):
        self.value = np.array(value, copy=True)

    def __array__(self, dtype=None):
        if dtype is None:
            return self.value
        return self.value.astype(dtype)

    @property
    def shape(self):
        return self.value.shape

    @property
    def dtype(self):
        return self.value.dtype


class _NumpyPlan:
    """替身计划：每次 replay 用参考后端现算，输出当作"设备数组"交回去。"""

    def __init__(self, backend, graph, shared):
        self._backend = backend
        self.graph = graph
        self._buffers = dict(shared)
        self._last = {}

    def _execute(self, inputs):
        feed = dict(self._buffers)
        feed.update(inputs)
        produced = self._backend.reference.execute(
            self.graph, {name: np.asarray(value) for name, value in feed.items()}
        )
        return {name: _Dev(produced[name]) for name in self.graph.output_names}

    def replay(self, inputs):
        self._last = self._execute(inputs)

    def device_outputs(self):
        return self._last

    def run_device(self, inputs):
        self._backend.plan_syncs += 1
        self.replay(inputs)
        return self.device_outputs()


class _NumpyDevice:
    """只满足设备端循环前置能力的替身后端。"""

    def __init__(self):
        self.reference = ReferenceBackend()
        self.captured = []
        self.streams = []
        self.capture_streams = []
        self.to_host_calls = 0
        self.to_array_calls = 0
        self.plan_syncs = 0

    def new_stream(self):
        stream = object()
        self.streams.append(stream)
        return stream

    def to_array(self, value):
        self.to_array_calls += 1
        return np.array(value, copy=True)

    def to_host(self, value):
        self.to_host_calls += 1
        return np.asarray(value)

    def capture(self, graph, sample_inputs, reuse_memory=True, shared_buffers=None,
                stream=None):
        shared = dict(shared_buffers if shared_buffers is not None else sample_inputs)
        for name in shared:
            if name not in sample_inputs and name not in shared:
                raise KeyError(name)
        self.captured.append(graph)
        self.capture_streams.append(stream)
        return _NumpyPlan(self, graph, shared)


class _NoCaptureBackend:
    """缺 capture 的后端：设备端循环必须在构造期拒绝，而不是跑到一半才发现。"""

    def to_array(self, value):
        return np.asarray(value)

    def to_host(self, value):
        return np.asarray(value)


class TestDeviceLoopNumerics(unittest.TestCase):
    def test_matches_host_loop_bit_for_bit(self):
        for step in (1.0, 0.5):
            loop = _loop()
            backend = _NumpyDevice()
            plan = DeviceWhileLoop(_loop(), {"x": _feed(0.0), "step": _feed(step)}, backend)
            # 宿主路径的初值来自同一个 loop 的 run；两步分别建计划，避免互相污染
            expected = loop.run({"x": _feed(0.0), "step": _feed(step)})
            got = plan.run({"x": _feed(0.0), "step": _feed(step)})
            self.assertEqual(got["x"], expected["x"])

    def test_run_device_returns_resident_state(self):
        plan = DeviceWhileLoop(_loop(), {"x": _feed(0.0), "step": _feed(1.0)}, _NumpyDevice())
        state = plan.run_device()
        self.assertEqual(float(np.asarray(state["x"])), 5.0)

    def test_repeated_runs_restart_from_the_initial_state(self):
        """第二次 run 必须重新从初值跑 5 轮，而不是接着上一次的 5 立刻退出。

        只比数值不够：从 5 开始跑会立刻退出，结果同样是 5。轮数才是判据。
        """
        plan = DeviceWhileLoop(_loop(), {"x": _feed(0.0), "step": _feed(1.0)}, _NumpyDevice())
        self.assertEqual(float(plan.run()["x"]), 5.0)
        self.assertEqual(plan.iterations, 5)
        self.assertEqual(float(plan.run()["x"]), 5.0)
        self.assertEqual(plan.iterations, 5)

    def test_iteration_count_is_reported(self):
        plan = DeviceWhileLoop(_loop(), {"x": _feed(0.0), "step": _feed(1.0)}, _NumpyDevice())
        plan.run()
        self.assertEqual(plan.iterations, 5)


class TestCaptureReuse(unittest.TestCase):
    def test_two_graphs_are_captured_once(self):
        backend = _NumpyDevice()
        plan = DeviceWhileLoop(_loop(), {"x": _feed(0.0), "step": _feed(1.0)}, backend)
        self.assertEqual(plan.captures, 2)
        self.assertEqual(len(backend.captured), 2)
        plan.run()
        plan.run()
        self.assertEqual(len(backend.captured), 2)

    def test_only_the_predicate_round_trips_per_iteration(self):
        """状态不掉头：5 轮 = 6 次谓词读（第 6 次是退出那次）+ 收尾 1 次状态。"""
        backend = _NumpyDevice()
        plan = DeviceWhileLoop(_loop(), {"x": _feed(0.0), "step": _feed(1.0)}, backend)
        backend.to_host_calls = 0
        plan.run()
        self.assertEqual(backend.to_host_calls, plan.iterations + 2)
        # 每轮只有条件那一次同步；体的 replay 是异步的，由下一轮条件兜住。
        self.assertEqual(plan.syncs, plan.iterations + 1)
        self.assertEqual(backend.plan_syncs, plan.iterations + 1)

    def test_both_graphs_are_captured_on_the_same_stream(self):
        backend = _NumpyDevice()
        DeviceWhileLoop(_loop(), {"x": _feed(0.0), "step": _feed(1.0)}, backend)
        self.assertEqual(len(backend.streams), 1)
        self.assertEqual(backend.capture_streams, [backend.streams[0]] * 2)

    def test_state_is_never_uploaded_again(self):
        backend = _NumpyDevice()
        plan = DeviceWhileLoop(_loop(), {"x": _feed(0.0), "step": _feed(1.0)}, backend)
        backend.to_array_calls = 0
        plan.run()
        plan.run()
        self.assertEqual(backend.to_array_calls, 0)

    def test_new_free_input_is_uploaded_once(self):
        backend = _NumpyDevice()
        plan = DeviceWhileLoop(_loop(), {"x": _feed(0.0), "step": _feed(1.0)}, backend)
        backend.to_array_calls = 0
        plan.run({"step": _feed(2.0)})
        self.assertEqual(backend.to_array_calls, 1)


class TestContract(unittest.TestCase):
    def test_nested_control_region_is_refused(self):
        nested = WhileLoop(
            Conditional(
                _counter_condition(),
                GraphRegion(_counter_body()),
                GraphRegion(_counter_body()),
            ),
            _counter_body(),
            carried={"x": "x_next"},
            max_steps=4,
        )
        with self.assertRaises(DeviceLoopError):
            DeviceWhileLoop(
                nested, {"x": _feed(0.0), "step": _feed(1.0)}, _NumpyDevice()
            )

    def test_backend_without_capture_is_refused_at_construction(self):
        with self.assertRaises(TypeError) as caught:
            DeviceWhileLoop(_loop(), {"x": _feed(0.0), "step": _feed(1.0)}, _NoCaptureBackend())
        self.assertIn("capture", str(caught.exception))

    def test_missing_initial_state_is_explicit(self):
        with self.assertRaises(ValueError) as caught:
            DeviceWhileLoop(_loop(), {"step": _feed(1.0)}, _NumpyDevice())
        self.assertIn("初始状态", str(caught.exception))

    def test_missing_free_input_is_explicit(self):
        with self.assertRaises(KeyError) as caught:
            DeviceWhileLoop(_loop(), {"x": _feed(0.0)}, _NumpyDevice())
        self.assertIn("step", str(caught.exception))

    def test_unknown_name_is_explicit(self):
        with self.assertRaises(KeyError) as caught:
            DeviceWhileLoop(
                _loop(), {"x": _feed(0.0), "step": _feed(1.0), "y": _feed(0.0)},
                _NumpyDevice(),
            )
        self.assertIn("y", str(caught.exception))

    def test_exhausting_max_steps_is_fatal(self):
        plan = DeviceWhileLoop(
            _loop(max_steps=2), {"x": _feed(0.0), "step": _feed(1.0)}, _NumpyDevice()
        )
        with self.assertRaises(ControlFlowError) as caught:
            plan.run()
        self.assertIn("max_steps", str(caught.exception))

    def test_shape_change_is_refused(self):
        """捕获图只对应一组固定形状：换形状必须显式报错，不能悄悄重捕获。"""
        plan = DeviceWhileLoop(_loop(), {"x": _feed(0.0), "step": _feed(1.0)}, _NumpyDevice())
        with self.assertRaises(ValueError):
            plan.run({"step": np.zeros(4, np.float32)})

    def test_wrong_dtype_is_refused(self):
        plan = DeviceWhileLoop(_loop(), {"x": _feed(0.0), "step": _feed(1.0)}, _NumpyDevice())
        with self.assertRaises(ValueError):
            plan.run({"step": np.asarray(1.0, np.float64)})

    def test_non_loop_is_refused(self):
        with self.assertRaises(TypeError):
            DeviceWhileLoop(GraphRegion(_counter_body()), {}, _NumpyDevice())


@unittest.skipUnless(HIP_AVAILABLE, "本机没有 HIP 设备")
class TestOnHipDevice(unittest.TestCase):
    """真机一条：HIP 上两次捕获、五轮 replay，结果与参考后端逐位相等。"""

    def test_matches_reference_on_hip(self):
        backend = HipBackend()
        plan = DeviceWhileLoop(
            _loop(), {"x": _feed(0.0), "step": _feed(1.0)}, backend
        )
        self.assertEqual(plan.captures, 2)
        got = plan.run()
        expected = _loop().run({"x": _feed(0.0), "step": _feed(1.0)})
        self.assertEqual(float(got["x"]), float(expected["x"]))
        self.assertEqual(plan.iterations, 5)
        # 条件 run_device 6 次（含退出那次）；体是异步 replay，不另计同步。
        self.assertEqual(plan.syncs, 6)


if __name__ == "__main__":
    unittest.main()
