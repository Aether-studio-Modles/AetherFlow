"""设备端控制流的判据：捕获一次、状态留在显存、语义与宿主路径逐位一致。

离线替身 ``_NumpyDevice`` 把 numpy 数组当"设备数组"：它不碰显存，但把
``capture`` / ``to_array`` / ``to_host`` 的调用次数如实记账——本层的主张
（"循环只捕获两次"、"每轮只有谓词回主机"、"分支只 replay 选中一支"）都靠
这份账来判。
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
from aetherflow.runtime import DeviceConditional, DeviceLoopError, DeviceWhileLoop

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
        self.replays = 0

    def _execute(self, inputs):
        feed = dict(self._buffers)
        feed.update(inputs)
        produced = self._backend.reference.execute(
            self.graph, {name: np.asarray(value) for name, value in feed.items()}
        )
        return {name: _Dev(produced[name]) for name in self.graph.output_names}

    def replay(self, inputs):
        self.replays += 1
        self._last = self._execute(inputs)

    def device_outputs(self):
        return self._last

    def run_device(self, inputs):
        self._backend.plan_syncs += 1
        self.replay(inputs)
        return self.device_outputs()


class _Stream:
    """替身的流：只需要可同步（真机上分支 replay 之后要等它完成）。"""

    def __init__(self):
        self.syncs = 0

    def synchronize(self):
        self.syncs += 1


class _NumpyDevice:
    """只满足设备端循环前置能力的替身后端。"""

    def __init__(self):
        self.reference = ReferenceBackend()
        self.captured = []
        self.plans = []
        self.streams = []
        self.capture_streams = []
        self.to_host_calls = 0
        self.to_array_calls = 0
        self.plan_syncs = 0

    def new_stream(self):
        stream = _Stream()
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
        plan = _NumpyPlan(self, graph, shared)
        self.plans.append(plan)
        return plan


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


def _pick_condition(graph_name="pick_cond"):
    """``1 - x > 0``，即 x < 1。"""
    g = Graph(graph_name)
    x = g.input("x", _scalar())
    g.output(g.step(g.sub(g.constant(1.0), x), threshold=0.0), "flag")
    return g


def _offset_branch(offset, graph_name):
    """y = x + offset。"""
    g = Graph(graph_name)
    x = g.input("x", _scalar())
    g.output(g.add(x, g.constant(offset)), "y")
    return g


def _conditional(name="pick"):
    return Conditional(
        GraphRegion(_pick_condition()),
        GraphRegion(_offset_branch(10.0, "then")),
        GraphRegion(_offset_branch(-10.0, "else")),
        name=name,
    )


def _conditional_device(backend=None):
    backend = backend if backend is not None else _NumpyDevice()
    return DeviceConditional(_conditional(), {"x": _feed(0.0)}, backend), backend


class TestDeviceConditionalNumerics(unittest.TestCase):
    def test_matches_host_conditional_bit_for_bit(self):
        plan, _ = _conditional_device()
        for value in (0.0, 0.999, 1.0, 3.0):
            got = plan.run({"x": _feed(value)})
            expected = _conditional().run({"x": _feed(value)})
            self.assertEqual(float(got["y"]), float(expected["y"]))

    def test_run_device_keeps_outputs_on_device(self):
        plan, backend = _conditional_device()
        backend.to_host_calls = 0
        backend.to_array_calls = 0
        state = plan.run_device()
        self.assertIsInstance(state["y"], _Dev)
        self.assertEqual(backend.to_host_calls, 1)  # 只读谓词
        self.assertEqual(backend.to_array_calls, 0)

    def test_run_returns_host_arrays_after_one_read(self):
        plan, backend = _conditional_device()
        backend.to_host_calls = 0
        got = plan.run({"x": _feed(0.0)})
        self.assertIsInstance(got["y"], np.ndarray)
        self.assertEqual(backend.to_host_calls, 2)  # 谓词 + 分支输出

    def test_selected_branch_flips_with_predicate(self):
        plan, backend = _conditional_device()
        self.assertEqual(float(plan.run({"x": _feed(0.0)})["y"]), 10.0)
        self.assertEqual(float(plan.run({"x": _feed(2.0)})["y"]), -8.0)
        self.assertEqual([p.replays for p in backend.plans], [2, 1, 1])


class TestDeviceConditionalCapture(unittest.TestCase):
    def test_three_graphs_captured_once_on_one_stream(self):
        plan, backend = _conditional_device()
        self.assertEqual(plan.captures, 3)
        self.assertEqual(len(backend.captured), 3)
        self.assertEqual(len(backend.streams), 1)
        for captured_stream in backend.capture_streams:
            self.assertIs(captured_stream, backend.streams[0])
        for _ in range(3):
            plan.run()
        self.assertEqual(len(backend.captured), 3)
        self.assertEqual(len(backend.streams), 1)

    def test_unselected_branch_never_replays(self):
        plan, backend = _conditional_device()
        plan.run({"x": _feed(0.0)})  # then
        self.assertEqual([p.replays for p in backend.plans], [1, 1, 0])
        plan.run({"x": _feed(5.0)})  # otherwise
        self.assertEqual([p.replays for p in backend.plans], [2, 1, 1])

    def test_one_sync_and_one_round_trip_per_run(self):
        plan, backend = _conditional_device()
        backend.plan_syncs = 0
        plan.run_device()
        self.assertEqual(plan.syncs, 1)
        self.assertEqual(backend.plan_syncs, 1)
        self.assertEqual(plan.branches, 1)

    def test_new_free_input_is_uploaded_once(self):
        plan, backend = _conditional_device()
        backend.to_array_calls = 0
        plan.run({"x": _feed(1.0)})
        self.assertEqual(backend.to_array_calls, 1)


class TestDeviceConditionalContract(unittest.TestCase):
    def test_nested_region_in_a_branch_is_refused(self):
        inner = Conditional(
            GraphRegion(_pick_condition("inner_cond")),
            GraphRegion(_offset_branch(1.0, "inner_then")),
            GraphRegion(_offset_branch(2.0, "inner_else")),
        )
        outer = Conditional(
            GraphRegion(_pick_condition("outer_cond")),
            inner,
            GraphRegion(_offset_branch(-10.0, "outer_else")),
        )
        with self.assertRaises(DeviceLoopError) as caught:
            DeviceConditional(outer, {"x": _feed(0.0)}, _NumpyDevice())
        self.assertIn("嵌套控制区域", str(caught.exception))

    def test_backend_without_capture_is_refused_at_construction(self):
        with self.assertRaises(TypeError) as caught:
            DeviceConditional(
                _conditional(), {"x": _feed(0.0)}, _NoCaptureBackend()
            )
        self.assertIn("capture", str(caught.exception))

    def test_missing_input_is_explicit(self):
        with self.assertRaises(KeyError) as caught:
            DeviceConditional(_conditional(), {}, _NumpyDevice())
        self.assertIn("x", str(caught.exception))

    def test_unknown_name_is_explicit(self):
        with self.assertRaises(KeyError) as caught:
            DeviceConditional(
                _conditional(),
                {"x": _feed(0.0), "y": _feed(0.0)},
                _NumpyDevice(),
            )
        self.assertIn("y", str(caught.exception))

    def test_shape_change_is_refused(self):
        plan, _ = _conditional_device()
        with self.assertRaises(ValueError):
            plan.run({"x": np.zeros(2, np.float32)})

    def test_wrong_dtype_is_refused(self):
        plan, _ = _conditional_device()
        with self.assertRaises(ValueError):
            plan.run({"x": np.asarray(0.0, np.float64)})

    def test_non_conditional_is_refused(self):
        with self.assertRaises(TypeError):
            DeviceConditional(
                GraphRegion(_pick_condition()), {"x": _feed(0.0)}, _NumpyDevice()
            )


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

    def test_conditional_matches_reference_on_hip(self):
        backend = HipBackend()
        plan = DeviceConditional(_conditional(), {"x": _feed(0.0)}, backend)
        self.assertEqual(plan.captures, 3)
        for value in (0.0, 2.0):
            got = plan.run({"x": _feed(value)})
            expected = _conditional().run({"x": _feed(value)})
            self.assertEqual(float(got["y"]), float(expected["y"]))
        self.assertEqual(plan.syncs, 1)


if __name__ == "__main__":
    unittest.main()
