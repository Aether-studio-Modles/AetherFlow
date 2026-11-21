"""HIP 后端的准入测试。

这里的东西等价于 DESIGN 里 M4 的判据：**conformance 跑通**才算承接，
而不是"能跑几个 demo"。没有 ROCm 的机器上整组跳过，但模块本身仍可导入——
``HipBackend()`` 的失败必须带明确原因，不能是 ImportError。
"""

import ast
import inspect
import textwrap
import unittest

import numpy as np

from aetherflow import Graph, ReferenceBackend, f16, f32, run_conformance
from aetherflow.backends.base import BackendCapabilityError
from aetherflow.backends.contracts import (
    AllocatorContract,
    implemented_contracts,
    supports,
)
from aetherflow.backends.cuda.backend import CudaBackend
from aetherflow.backends.device_backend import DeviceBackend
from aetherflow.backends.hip import HIP_AVAILABLE, HipBackend, runtime
from aetherflow.runtime.telemetry import TelemetryRecorder
from aetherflow.train import Adam, ParameterStore, SGD, Trainer, mse


class TestHipStreamContract(unittest.TestCase):
    """当前流必须能被 hipBLAS 与基准工具直接使用。

    ``benchmarks/run.py::_sync`` 只做一件事：拿当前流、``synchronize()``。
    HIP 侧少了这个方法不会在导入或构造时暴露，而是等到第一次计时才抛
    ``AttributeError``——正好是最不该出错的地方。用纯结构断言盯住它。
    """

    def test_current_stream_is_an_address(self):
        # hipblas.set_stream 要整数地址；0 是默认流
        self.assertEqual(runtime.current_stream(), 0)

    def test_stream_supports_benchmark_usage(self):
        stream = runtime._Stream()
        self.assertEqual(stream.ptr, 0)
        self.assertTrue(callable(stream.synchronize))


def _init_fields(cls) -> set:
    """``cls.__init__`` 里赋给 ``self.<name>`` 的字段名。

    只看 ``__init__`` 的函数体，不递归到被调用的辅助方法：这里要回答的是
    "构造完成时哪些属性该存在"，不是"整个类用了哪些属性"。
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls.__init__)))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for tgt in targets:
            if (
                isinstance(tgt, ast.Attribute)
                and isinstance(tgt.value, ast.Name)
                and tgt.value.id == "self"
            ):
                found.add(tgt.attr)
    return found


class TestHipAvailabilityProbe(unittest.TestCase):
    """审计 P1：``available()`` 曾只探测库能不能 ``CDLL``，把"设备起不来"当可用。

    WSL 里缺 ``HSA_ENABLE_DXG_DETECTION=1`` 时库照样能加载，``hipGetDeviceCount``
    却给 0；注册表会把 HIP 选走，然后第一次分配才失败。
    """

    class _Lib:
        def __init__(self, rc, count):
            self.rc = rc
            self.count = count

        def hipGetDeviceCount(self, pointer):
            pointer._obj.value = self.count
            return self.rc

    def _probe(self, lib):
        original = runtime._load_library
        runtime._load_library = lambda: lib
        try:
            return runtime.available()
        finally:
            runtime._load_library = original

    def test_library_without_devices_is_not_available(self):
        self.assertFalse(self._probe(self._Lib(0, 0)))

    def test_enumeration_failure_is_not_available(self):
        self.assertFalse(self._probe(self._Lib(100, 2)))

    def test_library_with_devices_is_available(self):
        self.assertTrue(self._probe(self._Lib(0, 2)))

    def test_missing_library_is_not_available(self):
        def boom():
            raise OSError("没有 HIP 运行时")

        original = runtime._load_library
        runtime._load_library = boom
        try:
            self.assertFalse(runtime.available())
        finally:
            runtime._load_library = original


class TestBackendSplit(unittest.TestCase):
    """HIP 与 CUDA 是兄弟关系，共享设备无关的执行计划层（AF-32 第四点）。

    以前 ``HipBackend`` 继承 ``CudaBackend``，HIP 的手抄 ``__init__`` 一旦漏字段
    就是一条 AttributeError 崩溃路径。现在两者都只继承 ``DeviceBackend``，
    这一组断言守的就是"别再挂回去、别再手抄"。
    """

    def test_hip_is_not_a_cuda_subclass(self):
        self.assertFalse(issubclass(HipBackend, CudaBackend))

    def test_both_share_the_device_backend(self):
        self.assertIs(HipBackend.__mro__[1], DeviceBackend)
        self.assertIs(CudaBackend.__mro__[1], DeviceBackend)

    def test_hip_init_only_sets_container_fields(self):
        # 构造字段全部由基类建；HIP 只负责把容器、方言与 SM 数交给它。
        self.assertLessEqual(_init_fields(HipBackend), {"cp", "codegen", "sm_count"})


@unittest.skipUnless(HIP_AVAILABLE, "需要 HIP 运行时（ROCm）")
class TestHipConformance(unittest.TestCase):
    """HIP 后端必须自证：声明支持的都对齐，边界必须显式拒绝。"""

    @classmethod
    def setUpClass(cls):
        cls.backend = HipBackend()
        cls.report = run_conformance(cls.backend)

    def test_passes(self):
        self.assertTrue(self.report.ok, self.report.text())

    def test_known_limits_are_explicit_refusals(self):
        refused = {c.name for c in self.report.refused}
        for name in (
            "matmul_partial_batch",
            "f64_input_relu",
        ):
            self.assertIn(name, refused, self.report.text())

    def test_all_declared_operators_exercised(self):
        self.assertEqual(set(self.report.covered_ops), set(self.report.capabilities))

    def test_numerics_align(self):
        for case in self.report.executed:
            self.assertLess(case.max_error, 1e-3, f"{case.name}: {case.max_error}")

    def test_device_contract(self):
        info = self.backend.device_info()
        self.assertTrue(info.name)
        self.assertGreater(info.total_memory_bytes, 0)
        # 契约名用短名上报（与 CUDA 侧一致），不是类名
        contracts = implemented_contracts(self.backend)
        self.assertIn("device", contracts)
        self.assertIn("device_manager", contracts)
        self.assertIn("codegen", contracts)
        self.assertIn("tuning", contracts)
        self.assertIn("memory", contracts)
        self.assertIn("stream", contracts)
        # 分配器是后端交出来的对象，与 CUDA 那份契约同形
        self.assertTrue(supports(self.backend.allocator(), AllocatorContract))

    def test_wrapping_keeps_every_contract(self):
        """遥测包装不许把契约"吃掉"（5.167）。

        基准与 ``agent_api`` 都从包装器的 ``implemented_contracts`` 判断能力：
        少一项就等于把多卡机器看成单卡、把有旋钮的后端看成没有。真机上用
        整表相等断言，避免只盯住刚补的那几项。
        """
        wrapped = TelemetryRecorder().wrap(self.backend, gpu_timing=False)
        self.assertEqual(
            implemented_contracts(wrapped), implemented_contracts(self.backend)
        )
        self.assertEqual(wrapped.current_device(), self.backend.current_device())

    def test_device_manager(self):
        """枚举按索引顺序；非法索引当场报错，且状态不变。"""
        self.backend.set_device(0)
        devices = self.backend.enumerate_devices()
        self.assertEqual(len(devices), runtime.device_count())
        self.assertGreaterEqual(len(devices), 1)
        self.assertEqual(self.backend.current_device(), 0)
        self.assertEqual(devices[0].name, self.backend.device_info().name)
        self.assertGreater(devices[0].total_memory_bytes, 0)
        with self.assertRaises(runtime.HipRuntimeError):
            self.backend.set_device(len(devices))
        with self.assertRaises(runtime.HipRuntimeError):
            self.backend.set_device(-1)
        with self.assertRaises(runtime.HipRuntimeError):
            self.backend.set_device("0")  # type: ignore[arg-type]
        self.assertEqual(self.backend.current_device(), 0)

    def test_capture_without_reuse_is_refused(self):
        """HIP 没有默认内存池：不复用内存的捕获必须明确拒绝。

        流捕获期间 ``hipMalloc`` 不被允许，没有 arena 就无处安放中间张量。
        拒绝要发生在碰 ``graph`` 之前，所以这里能传 ``None``。
        """
        with self.assertRaises(BackendCapabilityError):
            self.backend.capture(None, {}, reuse_memory=False)


class TestHipCaptureSurface(unittest.TestCase):
    """捕获相关的运行时接口必须在导入层面就存在。

    没有 ROCm 的机器跑不到真机路径，但漏一个 ``ctypes`` 签名或一个薄封装
    只会在真机上炸。这里只断言符号，不触发任何驱动调用。
    """

    def test_runtime_exposes_capture_api(self):
        for name in (
            "new_stream",
            "set_allocator",
            "get_allocator",
            "capture",
            "graph_exec",
        ):
            self.assertTrue(callable(getattr(runtime, name, None)), name)

    def test_runtime_exposes_device_api(self):
        """多设备的薄封装必须存在；没有 ROCm 的机器也能查符号。"""
        for name in (
            "device_count",
            "current_device",
            "set_device",
            "enumerate_devices",
            "device_info",
            "sm_count",
        ):
            self.assertTrue(callable(getattr(runtime, name, None)), name)

    def test_stream_is_a_context_manager(self):
        # 后端用 ``with stream:`` 切换当前流，缺少这两个钩子会退化成
        # "所有内核都落在默认流"，而捕获发生在另一条流上——图会是空的。
        self.assertTrue(hasattr(runtime._Stream, "__enter__"))
        self.assertTrue(hasattr(runtime._Stream, "__exit__"))

    def test_arena_ledger_is_offline(self):
        from aetherflow.backends.hip.memory import Arena

        # 只读账本：构造 arena 不碰驱动，因此这组断言不需要 ROCm
        arena = Arena()
        self.assertEqual(arena.total_bytes, 0)
        self.assertEqual(arena.peak_live_bytes, 0)

    def test_arena_recycle_reuses_warmup_blocks(self):
        """预热轮收回后，第二轮申请必须从旧块里切，不再向驱动要新块。

        这正是捕获轮的现实：hipMalloc 在流捕获期间被拒绝，一旦账本上
        还记着预热轮的输出，捕获第一步就失败。用一个只记账的 arena 把
        这条性质钉在离线测试里。
        """
        from aetherflow.backends.hip.memory import Arena

        class _FakeArena(Arena):
            """地址簿替身：不碰驱动，按 chunk 粒度为每次扩容发一个新地址。"""

            def __init__(self):
                super().__init__()
                self._next = 0x10000

            def _new_chunk(self, size):
                capacity = max(size, 1 << 20)
                addr = self._next
                self._next += capacity
                self._chunks.append((addr, capacity))
                self._bump.append(0)
                return addr

        arena = _FakeArena()
        arena.malloc(512 * 1024)
        arena.malloc(512 * 1024)
        total = arena.total_bytes

        arena.recycle()
        self.assertEqual(arena.live_bytes, 0)
        self.assertEqual(arena.peak_live_bytes, 0)

        arena.malloc(512 * 1024)
        arena.malloc(512 * 1024)
        self.assertEqual(arena.total_bytes, total, "回收后不应触发新的驱动分配")

    def test_arena_alias_holds_source_until_last_view_dies(self):
        """视图还按源地址读时，那块内存不能还给空闲表。

        执行器按"最后一次被读"回收的是**源数组**；slice / reshape / 恒等
        transpose 的视图这时还在环境里。源先死就还块，下一块同尺寸分配会
        顶掉它，replay 读到的是别人写进去的数据。离线账本钉住三条：视图
        计数未归零不还块、归零才还、还完之后能拿回同一段地址。
        """
        from aetherflow.backends.hip.memory import Arena

        class _FakeArena(Arena):
            """地址簿替身：不碰驱动，按 chunk 粒度为每次扩容发一个新地址。"""

            def __init__(self):
                super().__init__()
                self._next = 0x10000

            def _new_chunk(self, size):
                capacity = max(size, 1 << 20)
                addr = self._next
                self._next += capacity
                self._chunks.append((addr, capacity))
                self._bump.append(0)
                return addr

        arena = _FakeArena()
        src = arena.malloc(4096)
        arena.alias(src.ptr)
        arena.alias(src.ptr)  # 两个视图同时活着
        arena.free(src)  # 源数组先被回收
        self.assertEqual(arena.live_bytes, 4096, "视图还活着就不能还块")
        arena.unalias(src.ptr)
        self.assertEqual(arena.live_bytes, 4096, "还剩一个视图")
        arena.unalias(src.ptr)
        self.assertEqual(arena.live_bytes, 0)
        # 真正还块之后，同尺寸分配应当拿回同一段地址
        self.assertEqual(arena.malloc(4096).ptr, src.ptr)

        # 反方向：视图先死、源后死，应当直接还块
        other = arena.malloc(2048)
        arena.alias(other.ptr)
        arena.unalias(other.ptr)
        arena.free(other)
        self.assertEqual(arena.live_bytes, 4096)


@unittest.skipUnless(HIP_AVAILABLE, "需要 HIP 运行时（ROCm）")
class TestHipGraphCapture(unittest.TestCase):
    """hipGraph 的准入：捕获一次、replay 多次，结果与参考实现对齐。

    与 CUDA 侧同构。matmul 会走 hipBLAS，因此也顺带验证了"库调用能被流
    捕获记录"这条前提——若 hipBLAS 内部有同步或分配，捕获会直接失败。
    """

    @classmethod
    def setUpClass(cls):
        cls.backend = HipBackend()
        cls.ref = ReferenceBackend()

    def setUp(self):
        self.g = Graph()
        a = self.g.input("a", f32[4, 8])
        b = self.g.input("b", f32[8, 6])
        self.g.output(self.g.matmul(a, b), "y")
        self.feed = {"a": self.rand(4, 8, seed=7), "b": self.rand(8, 6, seed=9)}

    @staticmethod
    def rand(*shape, seed=7):
        return np.random.default_rng(seed).random(shape, dtype=np.float32)

    def _expected(self, feed):
        return self.ref.execute(self.g, feed)["y"]

    def test_matches_reference(self):
        plan = self.backend.capture(self.g, self.feed)
        np.testing.assert_allclose(
            plan.run(self.feed)["y"],
            self._expected(self.feed),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_replay_uses_new_inputs(self):
        plan = self.backend.capture(self.g, self.feed)
        plan.run(self.feed)
        second = {"a": self.rand(4, 8, seed=11), "b": self.rand(8, 6, seed=13)}
        got = plan.run(second)["y"]
        np.testing.assert_allclose(got, self._expected(second), rtol=1e-4, atol=1e-4)
        # 同时确认结果确实变了，排除"其实一直在用捕获时缓冲"的假通过
        self.assertGreater(
            float(np.abs(got - self._expected(self.feed)).max()), 1e-3
        )

    def test_replay_rejects_shape_change(self):
        plan = self.backend.capture(self.g, self.feed)
        with self.assertRaises(ValueError):
            plan.replay({"a": self.rand(5, 8), "b": self.rand(8, 6)})

    def test_repeated_replay_stable(self):
        """同一输入重复 replay 结果逐次一致（排除 arena 复用导致的踩踏）。"""
        plan = self.backend.capture(self.g, self.feed)
        first = plan.run(self.feed)["y"]
        for _ in range(5):
            np.testing.assert_array_equal(plan.run(self.feed)["y"], first)

    def test_slice_view_survives_source_release(self):
        """``slice`` 返回的是带偏移的视图：源数组先被回收，地址也不能复用。

        HIP 侧的 arena 此前没有 ``dispose``（中间值从不回收），这条别名
        假设因此从未被检验；一旦按 CUDA 的回收策略执行，扣账必须落在源
        地址上——视图自己的 ``ptr`` 带偏移，按它扣一条也匹配不上。
        """
        g = Graph()
        x = g.input("x", f32[4, 6])
        y = g.input("y", f32[4, 6])
        t = g.mul(x, x)  # 中间值：只被 slice 读，读完就该回收
        s = g.add_node("slice", [t], {"dim": 0, "start": 2, "stop": 4})
        k = g.mul(y, y)  # 同尺寸分配，会顶掉 t 的地址
        u = g.add_node("slice", [k], {"dim": 0, "start": 2, "stop": 4})
        g.output(g.add(s, u), "out")

        feed = {"x": self.rand(4, 6, seed=17), "y": self.rand(4, 6, seed=19)}
        expected = self.ref.execute(g, feed)["out"]
        plan = self.backend.capture(g, feed)
        np.testing.assert_allclose(plan.run(feed)["out"], expected, rtol=1e-4, atol=1e-4)

    def test_constant_broadcast_survives_capture(self):
        """捕获期 H2D 的主机源必须活到计划销毁。

        修复前的行为：``full`` 的主机 numpy 是函数内临时，捕获只记下它的
        地址；replay 时源早已释放，2048 个 1.0 读成别的内存内容（实测是
        整数 4 的位模式），events 图整张输出全错。
        """
        g = Graph()
        ones = g.broadcast_to(g.constant(1.0, dtype=f32), shape=(2048,))
        g.output(ones, "out")
        plan = self.backend.capture(g, {})
        np.testing.assert_array_equal(
            plan.run({})["out"], np.ones(2048, dtype=np.float32)
        )

    def test_snn_events_graph_captures(self):
        """events 图在捕获下与逐节点结果逐位一致。

        它同时踩到三条捕获期才暴露的路径：``broadcast_to`` 的常量物化、
        ``spike_offsets`` 的前缀和写入、``spike_csr`` 的整数输出，以及
        ``sparse_linear`` 读同一份 offsets。任何一条在 replay 时读错地址，
        这里的逐位比较都会失败。
        """
        from aetherflow.models import SNNSpec, build_snn_events

        spec = SNNSpec(input_dim=8, hidden=(16, 16), steps=4, batch=2)
        g = Graph("events")
        _, out = build_snn_events(g, spec)
        g.output(out, "out")

        rng = np.random.default_rng(23)
        feed = {
            "x": (rng.random((spec.steps, spec.batch, spec.input_dim)) > 0.5).astype(
                np.float32
            )
        }
        for name, shape in spec.param_shapes.items():
            feed[name] = (rng.random(shape) - 0.5).astype(np.float32)

        expected = self.backend.execute(g, feed)["out"]
        plan = self.backend.capture(g, feed)
        np.testing.assert_array_equal(plan.run(feed)["out"], expected)

    def test_default_stream_memset_precedes_private_stream_write(self):
        """默认流的同步 ``memset`` 返回时必须已经落地。

        WSL/DXG 上 ``hipMemset`` 只把命令排进默认流就返回；它之后的写入若
        发生在一条非阻塞私有流上，两者没有顺序关系，memset 会晚一步把刚
        写好的数据清掉（实测把 ``spike_offsets`` 的最后一个元素清零，
        5.135）。这里用 runtime 复现同一时序：默认流清零 → 私有流写 1，
        读回必须是 1。竞争窗口大小依驱动与环境而定，本机不一定每次都能
        命中原始时序；整条链的守门人是 ``test_snn_events_graph_captures``。
        """
        n = 256
        kernel = runtime.function(
            ("af_test_fill_last",),
            'extern "C" __global__ void af_fill_last(float* out, int n) {'
            " int i = blockIdx.x * blockDim.x + threadIdx.x;"
            " if (i == n - 1) out[i] = 1.0f; }",
            "af_fill_last",
            "af_fill_last",
        )
        # zeros 内部的默认流 memset 是最后一条默认流命令
        buf = runtime.zeros((n,), dtype=runtime.float32)
        stream = runtime.new_stream()
        with stream:
            kernel((1,), (n,), (buf, np.int32(n)))
        stream.synchronize()
        got = runtime.asnumpy(buf)
        self.assertEqual(float(got[-1]), 1.0, "memset 事后覆盖了内核算出的最后一个元素")
        np.testing.assert_array_equal(got[:-1], np.zeros(n - 1, dtype=np.float32))

    def test_memory_stats_reported(self):
        """``memory_stats`` 必须报出这张图的真实占用，而不是编一个数字。"""
        plan = self.backend.capture(self.g, self.feed)
        stats = self.backend.memory_stats()
        self.assertIsNotNone(stats)
        self.assertGreater(stats.total_bytes, 0)
        # 峰值不可能超过真正向驱动申请的容量
        self.assertGreaterEqual(stats.total_bytes, stats.peak_bytes)
        self.assertEqual(plan.peak_bytes, stats.peak_bytes)

    def test_capture_when_output_fills_first_chunk(self):
        """输出把首块占满时，捕获轮也必须只在已有块里安身。

        修复前的行为：预热轮的图输出不被 ``dispose``，仍记在 arena 账上，
        捕获轮第一个中间值就得 ``hipMalloc``，而流捕获期间该调用被拒绝——
        conv2d 的 512 KiB 输出正好把 1 MiB 首块占满，于是整条捕获失败。
        """
        g = Graph()
        x = g.input("x", f32[2, 32, 32, 32])
        w = g.input("w", f32[64, 32, 1, 1])
        b = g.input("b", f32[64])
        g.output(g.conv2d(x, w, b), "y")
        feed = {
            "x": self.rand(2, 32, 32, 32, seed=3),
            "w": self.rand(64, 32, 1, 1, seed=5),
            "b": self.rand(64, seed=7),
        }
        plan = self.backend.capture(g, feed)
        np.testing.assert_allclose(
            plan.run(feed)["y"],
            self.ref.execute(g, feed)["y"],
            rtol=1e-4,
            atol=1e-4,
        )


@unittest.skipUnless(HIP_AVAILABLE, "需要 HIP 运行时（ROCm）")
class TestHipAutoGraph(unittest.TestCase):
    """``graph_mode="auto"`` 在 HIP 上的自动捕获面。

    这一层完全在 ``DeviceBackend`` 里（按"图 + 版本 + 输入签名"缓存、
    LRU 上限、节点数闸门），厂商侧只提供 ``capture``。5.61 验证了
    ``HipGraphPlan`` 本身，5.79 的 auto 却只在 CUDA 上跑过——这组用例补上
    HIP 真机，顺带确认"计划键与签名"没有隐含依赖 CUDA 的东西。
    """

    @classmethod
    def setUpClass(cls):
        cls.ref = ReferenceBackend()

    @staticmethod
    def rand(*shape, seed=7):
        return np.random.default_rng(seed).random(shape, dtype=np.float32)

    def _two_node_graph(self):
        g = Graph()
        a = g.input("a", f32[4, 8])
        b = g.input("b", f32[8, 6])
        g.output(g.relu(g.matmul(a, b)), "y")
        return g

    def _feed(self, seed=7):
        return {"a": self.rand(4, 8, seed=seed), "b": self.rand(8, 6, seed=seed + 2)}

    def test_auto_matches_off_and_reuses_plan(self):
        g = self._two_node_graph()
        feed = self._feed()
        auto = HipBackend(graph_mode="auto")
        off = HipBackend()
        first = auto.execute(g, feed)["y"]
        second = auto.execute(g, feed)["y"]
        self.assertEqual(len(auto._graph_plans), 1)
        self.assertEqual(dict(auto._graph_failures), {})
        np.testing.assert_allclose(first, off.execute(g, feed)["y"], rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(second, first)

    def test_new_input_values_stay_on_same_plan(self):
        """同形状换值不该重新捕获，且 replay 必须读到新输入。"""
        g = self._two_node_graph()
        auto = HipBackend(graph_mode="auto")
        auto.execute(g, self._feed(7))
        other = self._feed(21)
        got = auto.execute(g, other)["y"]
        self.assertEqual(len(auto._graph_plans), 1)
        np.testing.assert_allclose(
            got, self.ref.execute(g, other)["y"], rtol=1e-4, atol=1e-4
        )
        self.assertGreater(
            float(np.abs(got - self.ref.execute(g, self._feed(7))["y"]).max()), 1e-3
        )

    def test_single_node_graph_is_not_captured(self):
        """低于 ``min_graph_nodes`` 的图不进捕获：省下的 launch 抵不过簿记。"""
        g = Graph()
        x = g.input("x", f32[4, 8])
        g.output(g.relu(x), "y")
        auto = HipBackend(graph_mode="auto")
        got = auto.execute(g, {"x": self.rand(4, 8)})
        self.assertEqual(len(auto._graph_plans), 0)
        self.assertEqual(tuple(got["y"].shape), (4, 8))

    def test_plan_lru_bound(self):
        """``max_graph_plans`` 是硬上限，且被挤掉的计划不影响正确性。"""
        auto = HipBackend(graph_mode="auto", max_graph_plans=1)
        shapes = ((4, 8), (5, 8))
        for rows in shapes:
            g = Graph()
            a = g.input("a", f32[rows[0], rows[1]])
            b = g.input("b", f32[rows[1], 6])
            g.output(g.relu(g.matmul(a, b)), "y")
            feed = {
                "a": self.rand(rows[0], rows[1], seed=rows[0]),
                "b": self.rand(rows[1], 6, seed=rows[0] + 1),
            }
            np.testing.assert_allclose(
                auto.execute(g, feed)["y"],
                self.ref.execute(g, feed)["y"],
                rtol=1e-4,
                atol=1e-4,
            )
            self.assertLessEqual(len(auto._graph_plans), 1)


@unittest.skipUnless(HIP_AVAILABLE, "需要 HIP 运行时（ROCm）")
class TestHipContainer(unittest.TestCase):
    """设备数组的视图与拷贝语义：后端代码依赖这几条，单独钉住。"""

    @classmethod
    def setUpClass(cls):
        from aetherflow.backends.hip import runtime

        cls.rt = runtime

    def test_roundtrip(self):
        host = np.arange(12, dtype=np.float32).reshape(3, 4)
        dev = self.rt.asarray(host)
        np.testing.assert_array_equal(self.rt.asnumpy(dev), host)

    def test_slice_and_contiguous(self):
        host = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        dev = self.rt.asarray(host)
        # 切中间一轴：视图不连续，物化后必须与 numpy 一致
        view = dev[:, 1:3, :]
        self.assertFalse(view.flags.c_contiguous)
        np.testing.assert_array_equal(
            self.rt.asnumpy(self.rt.ascontiguousarray(view)), host[:, 1:3, :]
        )

    def test_transpose_is_a_view(self):
        host = np.arange(6, dtype=np.float32).reshape(2, 3)
        dev = self.rt.asarray(host)
        turned = self.rt.transpose(dev, (1, 0))
        self.assertEqual(turned.shape, (3, 2))
        np.testing.assert_array_equal(self.rt.asnumpy(turned), host.T)

    def test_broadcast_view(self):
        host = np.arange(3, dtype=np.float32)
        dev = self.rt.asarray(host)
        wide = self.rt.broadcast_to(dev, (2, 3))
        self.assertEqual(wide.strides, (0, 1))
        np.testing.assert_array_equal(
            self.rt.asnumpy(self.rt.ascontiguousarray(wide)),
            np.broadcast_to(host, (2, 3)),
        )

    def test_assign_into_strided_view(self):
        """非连续目标写回：分组卷积按组拼输出依赖这一条（5.165）。"""
        host = np.zeros((2, 4, 3), dtype=np.float32)
        dev = self.rt.asarray(host)
        part = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
        dev[:, 1:3, :] = self.rt.asarray(part)
        host[:, 1:3, :] = part
        np.testing.assert_array_equal(self.rt.asnumpy(dev), host)
        # 非连续源也接受：先物化一份再写
        base = self.rt.asarray(
            np.concatenate([part, part], axis=1)
        )
        sliced = base[:, :2, :]
        self.assertFalse(sliced.flags.c_contiguous)
        dev[:, 1:3, :] = sliced
        np.testing.assert_array_equal(self.rt.asnumpy(dev), host)

    def test_f64_is_carried_then_refused(self):
        """f64 必须能被承载，才能让"不支持"发生在能力检查里。"""
        dev = self.rt.asarray(np.ones(3, dtype=np.float64))
        self.assertEqual(dev.dtype, np.dtype("float64"))


@unittest.skipUnless(HIP_AVAILABLE, "需要 HIP 运行时（ROCm）")
class TestHipAttentionMask(unittest.TestCase):
    """因果 mask 走共享路径：只保留一份 ``(S, T)``，结果与整张 ``(BH,S,T)`` 一致。

    这条用例同时守住两件事：共享路径的寻址正确（内核用 ``mask_shared`` 跳
    过 ``bh`` 偏移），以及"逐 batch/head 的 mask 仍会被完整复制"没有被共享
    判断误吞。
    """

    S = 64
    HEADS = (2, 3)
    D = 8

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(5)
        shape = cls.HEADS + (cls.S, cls.D)
        cls.q = rng.standard_normal(shape).astype(np.float32)
        cls.k = rng.standard_normal(shape).astype(np.float32)
        cls.v = rng.standard_normal(shape).astype(np.float32)
        causal = np.tril(np.ones((cls.S, cls.S), np.float32))
        cls.mask = np.where(causal > 0, 0.0, -1e9).astype(np.float32)[None, None]

    @classmethod
    def _graph(cls, mask_shape):
        g = Graph("attn")
        q = g.input("q", f32[cls.HEADS + (cls.S, cls.D)])
        k = g.input("k", f32[cls.HEADS + (cls.S, cls.D)])
        v = g.input("v", f32[cls.HEADS + (cls.S, cls.D)])
        m = g.input("mask", f32[mask_shape])
        g.output(g.attention(q, k, v, m, scale=0.125), "y")
        return g

    def _run(self, backend, mask):
        feed = {"q": self.q, "k": self.k, "v": self.v, "mask": mask}
        return backend.execute(self._graph(mask.shape), feed)["y"]

    def test_shared_and_expanded_mask_agree(self):
        backend = HipBackend()
        want = self._run(ReferenceBackend(), self.mask)
        shared = self._run(backend, self.mask)
        expanded = np.broadcast_to(self.mask, self.HEADS + (self.S, self.S)).copy()
        copied = self._run(backend, expanded)

        np.testing.assert_allclose(shared, want, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(copied, want, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(shared, copied, rtol=1e-6, atol=1e-6)


@unittest.skipUnless(HIP_AVAILABLE, "需要 HIP 运行时（ROCm）")
class TestHipAttentionFloat16(unittest.TestCase):
    """f16 融合 attention：存储半精度，点积与在线 softmax 在 float 里。

    HIP 的 f16 存取走 ``__builtin_bit_cast(_Float16, ...)``，这条用例让它第一次
    落到分块 attention 上：q/k/v/mask/out 是 ``unsigned short``，共享内存、
    行最大值、归一化因子与输出累加器仍是 float。

    参考实现逐位 f16 运算（每一步都舍入），内核只在写回时舍入一次，因此差异
    来自舍入位置而非实现错误；容差按约两位有效数字取 5e-2。
    """

    SHAPE = (2, 2, 6, 8)
    VSHAPE = (2, 2, 6, 5)
    MASK_SHAPE = (1, 1, 6, 6)

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(17)
        cls.q = (rng.standard_normal(cls.SHAPE) * 0.3).astype(np.float16)
        cls.k = (rng.standard_normal(cls.SHAPE) * 0.3).astype(np.float16)
        cls.v = rng.standard_normal(cls.VSHAPE).astype(np.float16)
        cls.mask = np.linspace(-0.5, 0.5, 36, dtype=np.float16).reshape(
            cls.MASK_SHAPE
        )

    def _graph(self, *, dropout=0.0):
        g = Graph("attn_f16")
        q = g.input("q", f16[self.SHAPE])
        k = g.input("k", f16[self.SHAPE])
        v = g.input("v", f16[self.VSHAPE])
        m = g.input("mask", f16[self.MASK_SHAPE])
        g.output(
            g.attention(q, k, v, m, scale=0.25, dropout=dropout, seed=11), "y"
        )
        return g

    def _check(self, *, dropout=0.0):
        g = self._graph(dropout=dropout)
        feed = {"q": self.q, "k": self.k, "v": self.v, "mask": self.mask}
        want = ReferenceBackend().execute(g, feed)["y"]
        got = HipBackend().execute(g, feed)["y"]
        self.assertEqual(got.dtype, np.float16)
        np.testing.assert_allclose(got, want, rtol=5e-2, atol=5e-2)

    def test_plain(self):
        self._check()

    def test_with_dropout(self):
        self._check(dropout=0.3)


@unittest.skipUnless(HIP_AVAILABLE, "需要 HIP 运行时（ROCm）")
class TestTieredAllocator(unittest.TestCase):
    """执行期分层分配器：小对象进池、大对象直调、统计可读。

    这组只碰分配器本身，不构图——它比"跑一张图"更能钉住分层的边界：
    同一档的块必须复用同一个地址，而超过 ``_LARGE_MIN`` 的块释放后必须
    立刻从 ``total_bytes`` 里消失（那说明它是直调、没有留在池里）。
    """

    def test_small_blocks_are_recycled(self):
        from aetherflow.backends.hip.memory import TieredAllocator

        alloc = TieredAllocator()
        try:
            first = alloc(1000)
            addr = first.ptr
            del first
            second = alloc(1000)
            self.assertEqual(second.ptr, addr)
            self.assertGreater(alloc.stats().requested_bytes, 0)
        finally:
            alloc.close()

    def test_large_blocks_bypass_the_pool(self):
        from aetherflow.backends.hip.memory import _LARGE_MIN, TieredAllocator

        alloc = TieredAllocator()
        try:
            block = alloc(_LARGE_MIN + 1)
            self.assertGreaterEqual(alloc.total_bytes, _LARGE_MIN)
            self.assertEqual(alloc.stats().live_bytes, _LARGE_MIN + 1)
            del block
            self.assertEqual(alloc.total_bytes, 0)
            self.assertEqual(alloc.stats().live_bytes, 0)
        finally:
            alloc.close()

    def test_peak_covers_live(self):
        from aetherflow.backends.hip.memory import TieredAllocator

        alloc = TieredAllocator()
        try:
            blocks = [alloc(4096) for _ in range(4)]
            stats = alloc.stats()
            self.assertGreaterEqual(stats.peak_bytes, stats.live_bytes)
            del blocks
            self.assertGreaterEqual(alloc.stats().peak_bytes, 4 * 4096)
        finally:
            alloc.close()


@unittest.skipUnless(HIP_AVAILABLE, "需要 HIP 运行时（ROCm）")
class TestHipTieredExecution(unittest.TestCase):
    """``memory_mode="tiered"`` 的执行路径：数值不变、统计有值、分配器还原。"""

    @staticmethod
    def _graph():
        g = Graph("tiered")
        a = g.input("a", f32[4, 8])
        b = g.input("b", f32[8])
        g.output(g.mul(g.relu(g.add(a, b)), a), "y")
        return g

    @staticmethod
    def _feed():
        return {
            "a": (np.arange(32, dtype=np.float32).reshape(4, 8) - 16.0) / 8.0,
            "b": np.linspace(-1.0, 1.0, 8, dtype=np.float32),
        }

    def test_tiered_matches_container(self):
        graph = self._graph()
        want = HipBackend(memory_mode="container").execute(graph, self._feed())["y"]

        backend = HipBackend(memory_mode="tiered")
        got = backend.execute(graph, self._feed())["y"]
        np.testing.assert_array_equal(got, want)

        stats = backend.memory_stats()
        self.assertIsNotNone(stats)
        self.assertGreater(stats.total_bytes, 0)
        self.assertGreaterEqual(stats.peak_bytes, stats.live_bytes)

    def test_allocator_is_restored_after_execute(self):
        backend = HipBackend(memory_mode="tiered")
        before = runtime.get_allocator()
        backend.execute(self._graph(), self._feed())
        self.assertIs(runtime.get_allocator(), before)

    def test_container_mode_does_not_install_allocator(self):
        backend = HipBackend(memory_mode="container")
        before = runtime.get_allocator()
        backend.execute(self._graph(), self._feed())
        self.assertIs(runtime.get_allocator(), before)
        # 容器模式不写执行期统计：捕获前仍应是 None
        self.assertIsNone(backend.memory_stats())


@unittest.skipUnless(HIP_AVAILABLE, "需要 HIP 运行时（ROCm）")
class TestHipCaptureDeferredFree(unittest.TestCase):
    """捕获窗口里的释放必须推迟，否则会让捕获失效（901）。

    真实触发条件是 GC 恰好落在捕获期间：跑全量时偶发过一次
    ``conv2d_tiled`` 报 901、单独复跑不复现。这里用确定性的析构把窗口内
    释放钉住——``hipFree`` / ``hipHostFree`` 在捕获期都是非法操作。
    """

    def test_device_and_pinned_free_are_deferred(self):
        runtime.synchronize()
        rt = runtime._Runtime.instance()
        dead = runtime.empty((256,), np.float32)
        host = runtime.pinned((64,), np.float32)
        with runtime.new_stream() as stream:
            with runtime.capture(stream) as session:
                # 捕获期内释放：释放必须被延迟到图执行之后，而不是立即归还。
                del dead, host
            self.assertNotEqual(session.graph, 0)
        self.assertEqual(rt._deferred_free, [])
        self.assertEqual(rt._deferred_pinned, [])


@unittest.skipUnless(HIP_AVAILABLE, "需要 HIP 运行时（ROCm）")
class TestHipPinnedArray(unittest.TestCase):
    """replay 的主机输入必须走 pinned 暂存（5.156）。

    判据不是"有没有调某个函数"，而是缓冲真的按 ``(名字, 形状, dtype)``
    复用了、dtype 转换发生在主机侧、并且失败时能明确报错——pinned 池是
    每步都走的热路径，任何一处静默出错都会变成"偶发的错误输入"。
    """

    @classmethod
    def setUpClass(cls):
        cls.backend = HipBackend()

    def test_buffer_is_reused_per_key_shape_and_dtype(self):
        first = np.arange(6, dtype=np.float32).reshape(2, 3)
        holder = self.backend.pinned_array("x", first, np.float32)
        np.testing.assert_array_equal(holder.view, first)

        second = first + 1.0
        again = self.backend.pinned_array("x", second, np.float32)
        self.assertIs(again, holder, "同 key/形状/dtype 必须复用同一块 pinned 内存")
        np.testing.assert_array_equal(holder.view, second)

        other = self.backend.pinned_array("x", first, np.float16)
        self.assertIsNot(other, holder)
        self.assertEqual(other.view.dtype, np.dtype(np.float16))

    def test_host_dtype_is_cast_inside_the_pinned_buffer(self):
        host = np.arange(4, dtype=np.float64) + 0.25
        holder = self.backend.pinned_array("y", host, np.float32)
        self.assertEqual(holder.view.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(holder.view, host.astype(np.float32))

    def test_replay_registers_pinned_inputs(self):
        g = Graph()
        a = g.input("a", f32[4, 5])
        g.output(g.relu(a), "y")
        feed = {"a": np.zeros((4, 5), dtype=np.float32)}
        plan = self.backend.capture(g, feed)
        plan.run({"a": np.ones((4, 5), dtype=np.float32)})
        keys = {key[0] for key in (self.backend._pinned or {})}
        self.assertIn("a", keys)

    def test_pinned_rejects_unsupported_dtype(self):
        # int16 既不在容器承载面，也不对应任何 IR dtype
        with self.assertRaises(runtime.HipRuntimeError):
            runtime.pinned((4,), np.int16)

    def test_pinned_accepts_complex(self):
        # 复数必须能被承载：HIP 没有 rocFFT，``fft`` 要在 invoke 里以
        # "未实现算子"被拒，而不是在拷贝阶段变成一个裸的运行时错误
        holder = runtime.pinned((4,), np.complex64)
        self.assertEqual(holder.view.dtype, np.dtype("complex64"))
        self.assertEqual(holder.nbytes, 32)


@unittest.skipUnless(HIP_AVAILABLE, "需要 HIP 运行时（ROCm）")
class TestHipElementwise(unittest.TestCase):
    """``DeviceArray`` 的受控运算符面（优化器与全局范数所需）。

    HIP 的容器以前只有切片与整块拷贝，``train/_arrays.array_module`` 只能
    回落到 numpy，设备态训练因此在第一次 step 就被拒绝（5.87）。这组用例
    逐项对拍 numpy，覆盖优化器实际用到的每一种组合。
    """

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        cls.a = rng.normal(size=(3, 4)).astype(np.float32)
        cls.b = rng.normal(size=(3, 4)).astype(np.float32)

    def _pair(self):
        return runtime.asarray(self.a), runtime.asarray(self.b)

    def test_binary_ops_match_numpy(self):
        a, b = self._pair()
        cases = {
            "add": (lambda: a + b, self.a + self.b),
            "sub": (lambda: a - b, self.a - self.b),
            "mul": (lambda: a * b, self.a * self.b),
            "div": (lambda: a / b, self.a / self.b),
        }
        for name, (run, expected) in cases.items():
            with self.subTest(op=name):
                np.testing.assert_allclose(
                    runtime.asnumpy(run()), expected, rtol=1e-6, atol=1e-7
                )

    def test_scalar_and_reverse_forms(self):
        a, _ = self._pair()
        np.testing.assert_allclose(runtime.asnumpy(a * 2.5), self.a * 2.5, rtol=1e-6)
        np.testing.assert_allclose(runtime.asnumpy(2.5 * a), self.a * 2.5, rtol=1e-6)
        np.testing.assert_allclose(runtime.asnumpy(a + 0.5), self.a + 0.5, rtol=1e-6)
        np.testing.assert_allclose(runtime.asnumpy(0.5 - a), 0.5 - self.a, rtol=1e-6)
        np.testing.assert_allclose(runtime.asnumpy(a / 4.0), self.a / 4.0, rtol=1e-6)
        np.testing.assert_allclose(runtime.asnumpy(4.0 / a), 4.0 / self.a, rtol=1e-6)

    def test_inplace_ops_and_scalar_fill(self):
        a, b = self._pair()
        expected = self.a.copy()
        a += b
        expected += self.b
        a *= 2.0
        expected *= 2.0
        a -= 0.25
        expected -= 0.25
        a /= 8.0
        expected /= 8.0
        np.testing.assert_allclose(runtime.asnumpy(a), expected, rtol=1e-6)

        a[...] = 0
        np.testing.assert_array_equal(runtime.asnumpy(a), np.zeros_like(self.a))

    def test_broadcast_and_unary(self):
        row = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        col = np.array([[2.0], [4.0]], dtype=np.float32)
        got = runtime.asarray(row) + runtime.asarray(col)
        np.testing.assert_allclose(runtime.asnumpy(got), row + col, rtol=1e-6)

        pos = np.abs(self.a).astype(np.float32)
        root = runtime.sqrt(runtime.asarray(pos))
        np.testing.assert_allclose(runtime.asnumpy(root), np.sqrt(pos), rtol=1e-6)

        squared = runtime.asarray(self.a) ** 2
        np.testing.assert_allclose(runtime.asnumpy(squared), self.a ** 2, rtol=1e-6)

        negated = -runtime.asarray(self.a)
        np.testing.assert_allclose(runtime.asnumpy(negated), -self.a, rtol=1e-6)

    def test_inplace_broadcast_shares_one_decomposition(self):
        """连续目标与广播视图必须共用同一套维度分解。

        连续 ``(2,16,4,4)`` 按自身步长会折成 ``[2,256,1,1]``，而
        ``(1,16,1,1)`` 的广播视图会折成 ``[2,16,16,1]``。内核只有一套 dims，
        若按目标折叠却乘视图的步长，channel 的步长就落到 h*w 维上，结果是
        所有通道都读成 channel 0（MIOpen 对照里 bias 就是这样错的）。
        """
        rng = np.random.default_rng(13)
        y_host = rng.normal(size=(2, 16, 4, 4)).astype(np.float32)
        bias = rng.normal(size=16).astype(np.float32)
        y = runtime.asarray(y_host)
        y += runtime.asarray(bias).reshape(1, -1, 1, 1)
        np.testing.assert_allclose(
            runtime.asnumpy(y),
            y_host + bias.reshape(1, -1, 1, 1),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_sum_copy_and_zeros_like(self):
        a, _ = self._pair()
        total = runtime.sum(a)
        self.assertEqual(total.shape, ())
        self.assertEqual(total.dtype, np.dtype(np.float64))
        self.assertAlmostEqual(float(runtime.asnumpy(total)), float(self.a.sum()), 5)

        copied = a.copy()
        copied[...] = 0
        np.testing.assert_array_equal(runtime.asnumpy(a), self.a, "copy 不能共享存储")
        self.assertEqual(runtime.zeros_like(a).shape, a.shape)

    def test_pow_rejects_other_exponents(self):
        a, _ = self._pair()
        with self.assertRaises(runtime.HipRuntimeError):
            a ** 3


def _linear(batch=8):
    g = Graph("hip_lin")
    x = g.input("x", f32[batch, 4])
    t = g.input("t", f32[batch, 1])
    w = g.input("w", f32[4, 1])
    b = g.input("b", f32[1])
    g.output(mse(g, g.add(g.matmul(x, w), b), t), "loss")
    return g


def _linear_data(seed=0, n=32):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    true_w = np.array([[1.0], [-2.0], [0.5], [3.0]], dtype=np.float32)
    true_b = np.array([0.7], dtype=np.float32)
    return x, (x @ true_w + true_b).astype(np.float32)


@unittest.skipUnless(HIP_AVAILABLE, "需要 HIP 运行时（ROCm）")
class TestHipDeviceStateTraining(unittest.TestCase):
    """参数常驻显存的训练轨迹必须与主机态一致（5.156）。

    这条链路同时依赖四块：replay 的 pinned H2D、捕获流与默认流的顺序、
    ``DeviceArray`` 的运算符面、以及 ``array_module`` 的 HIP 分派。任何
    一块缺失，损失或参数都会在几步内偏离主机态。
    """

    def test_sgd_trajectory_matches_host(self):
        x, t = _linear_data(n=8)
        g = _linear()

        host_store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=11)
        host = Trainer(
            g, host_store, SGD(host_store.arrays(), lr=0.05), backend=HipBackend()
        )
        host_losses = [host.step({"x": x, "t": t}) for _ in range(10)]

        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=11)
        init = {name: store[name].copy() for name in store.names}
        backend = HipBackend()
        trainer = Trainer(
            g,
            store,
            SGD(store.arrays(), lr=0.05),
            backend=backend,
            device_state=True,
        )
        losses = [trainer.step({"x": x, "t": t}) for _ in range(10)]

        np.testing.assert_allclose(losses, host_losses, rtol=1e-4, atol=1e-5)
        for name in store.names:
            self.assertIsInstance(trainer.optimizer.params[name], runtime.DeviceArray)
            np.testing.assert_array_equal(store[name], init[name])
        self.assertIs(trainer._fwd_plan._buffers["w"], trainer.optimizer.params["w"])
        self.assertIs(trainer._bwd_plan._buffers["w"], trainer.optimizer.params["w"])
        trainer.verify_device_state()

        trainer.sync_to_host()
        for name in store.names:
            np.testing.assert_allclose(
                store[name], host_store[name], rtol=1e-4, atol=1e-5
            )

    def test_adam_trajectory_matches_host(self):
        x, t = _linear_data(n=8)
        g = _linear()
        runs = {}
        for device_state in (False, True):
            store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=5)
            trainer = Trainer(
                g,
                store,
                Adam(store.arrays(), lr=0.02),
                backend=HipBackend(),
                device_state=device_state,
            )
            losses = [trainer.step({"x": x, "t": t}) for _ in range(5)]
            trainer.sync_to_host()
            runs[device_state] = (losses, {n: store[n].copy() for n in store.names})

        host_losses, host_params = runs[False]
        dev_losses, dev_params = runs[True]
        np.testing.assert_allclose(dev_losses, host_losses, rtol=1e-4, atol=1e-5)
        for name, expected in host_params.items():
            np.testing.assert_allclose(dev_params[name], expected, rtol=1e-4, atol=1e-5)

    def test_global_norm_clip_runs_on_device(self):
        """``clip_global_norm`` 的设备分支：平方和、开方、逐元素缩放都在显存。"""
        from aetherflow.train import clip_global_norm

        grads = {
            "w": runtime.asarray(np.array([[3.0], [4.0]], dtype=np.float32)),
            "b": runtime.asarray(np.array([0.0], dtype=np.float32)),
        }
        clipped, total = clip_global_norm(grads, max_norm=1.0)
        self.assertAlmostEqual(total, 5.0, places=4)
        for name, value in clipped.items():
            self.assertIsInstance(value, runtime.DeviceArray)
        # [3, 4] 的范数是 5，缩到 1 之后最大分量是 4/5
        self.assertAlmostEqual(
            float(np.abs(runtime.asnumpy(clipped["w"])).max()), 0.8, places=4
        )


@unittest.skipUnless(HIP_AVAILABLE, "需要 HIP 运行时（ROCm）")
class TestHipReduceTuning(unittest.TestCase):
    """归约类的 block 旋钮在 HIP 上也必须落到 launch（5.160）。

    内核语义来自 ``DeviceBackend``，所以这道闸门在两个硬件后端上都要验：
    覆盖优先、只影响被覆盖的算子、小于列数的块不改变数值结论。
    """

    @classmethod
    def setUpClass(cls):
        cls.backend = HipBackend()

    def test_reduce_ops_are_tunable(self):
        tunable = self.backend.tunable_ops()
        for op in ("softmax", "layernorm", "rmsnorm", "reduce_sum", "batch_norm"):
            self.assertIn(op, tunable)
            self.assertIn(None, self.backend.tuning_space(op).values("block"))

    def test_reduce_block_override_reaches_launch(self):
        self.assertIsNone(self.backend.block)
        self.assertEqual(self.backend._reduce_block("softmax", 100), 128)
        try:
            self.backend.set_tuning("softmax", {"block": 128})
            self.assertEqual(self.backend._reduce_block("softmax", 5000), 128)
            self.assertEqual(self.backend._reduce_block("layernorm", 5000), 1024)
        finally:
            self.backend.clear_tuning("softmax")

    def test_reduce_block_change_stays_close(self):
        """block 小于列数时每个线程跨步多走几步，数值只能近似一致。"""
        g = Graph("softmax_tune")
        x = g.input("x", f32[256, 2048])
        g.output(g.softmax(x), "y")
        feed = {
            "x": np.random.default_rng(17)
            .standard_normal((256, 2048))
            .astype(np.float32)
        }
        default = self.backend.execute(g, feed)["y"]
        try:
            for block in (128, 1024):
                self.backend.set_tuning("softmax", {"block": block})
                np.testing.assert_allclose(
                    self.backend.execute(g, feed)["y"],
                    default,
                    rtol=1e-5,
                    atol=1e-6,
                )
        finally:
            self.backend.clear_tuning("softmax")


@unittest.skipUnless(HIP_AVAILABLE, "需要 HIP 运行时（ROCm）")
class TestHipGpuTiming(unittest.TestCase):
    """5.161：HIP 后端能报逐节点 GPU 时间。

    事件对排在当前流上，整次执行只同步一次；读出来的 ``gpu_ms`` 必须非负，
    且整图累加不能是 0——512 阶 matmul 再快也要几十微秒。
    """

    def test_nodes_report_gpu_time(self):
        from aetherflow.runtime.telemetry import TelemetryRecorder

        g = Graph("gpu_timing")
        a = g.input("a", f32[512, 512])
        b = g.input("b", f32[512, 512])
        g.output(g.matmul(a, b), "y")
        rng = np.random.default_rng(5)
        feed = {
            "a": rng.random((512, 512), dtype=np.float32),
            "b": rng.random((512, 512), dtype=np.float32),
        }

        recorder = TelemetryRecorder()
        recorder.wrap(HipBackend()).execute(g, feed)
        record = recorder.last

        self.assertEqual(len(record.nodes), 1)
        item = record.nodes[0]
        self.assertIsNotNone(item.gpu_ms)
        self.assertGreaterEqual(item.gpu_ms, 0.0)
        self.assertGreater(item.gpu_ms, 0.0)
        self.assertTrue(record.to_dict()["node_gpu_totals"])
        self.assertEqual(
            record.node_gpu_totals()[0][0], "matmul", "聚合按算子名"
        )


if __name__ == "__main__":
    unittest.main()
