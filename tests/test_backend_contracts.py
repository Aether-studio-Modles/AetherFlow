"""可选契约（``DeviceContract`` / ``CodegenContract`` / ``TuningContract`` /
``MemoryContract`` / ``StreamContract``）的探测语义。

这一组要固定三件事：

1. **最小面没有变。** 参考后端不实现任何可选契约，仍然是合格后端——
   否则"拆契约"就变成了"加门槛"。
2. **探测只看有没有。** ``isinstance(obj, Protocol)`` 不查签名，
   所以结论只能是"声称具备"，不能是"实现正确"。测试把这个局限写成断言。
3. **报告把契约当信息项。** conformance 会列出实现了哪些契约，
   但缺契约不判失败、不影响报告结论。
"""

import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from aetherflow import (
    AllocatorContract,
    BackendCapabilities,
    CodegenContract,
    DeviceContract,
    DeviceInfo,
    DeviceManagerContract,
    Graph,
    MemoryContract,
    MemoryStats,
    OpCapability,
    ReferenceBackend,
    StreamContract,
    TuningContract,
    VerifierError,
    capability_gaps,
    default_cases,
    f16,
    f32,
    implemented_contracts,
    require,
    run_conformance,
    supports,
    verify,
)
from aetherflow.backends.base import Backend, BackendCapabilityError
from aetherflow.backends.cuda import CUDA_AVAILABLE, CudaBackend
from aetherflow.backends.hip import HipBackend
from aetherflow.ir.ops import OPS, Portability


class _Delegate:
    """承接全部算子，但不声明任何可选契约。"""

    name = "delegate"

    def __init__(self):
        self._ref = ReferenceBackend()

    def capabilities(self):
        return self._ref.capabilities()

    def execute(self, graph, inputs):
        return self._ref.execute(graph, inputs)


class _NameOnly(_Delegate):
    """把细化声明降级成纯算子名集合：模拟只承诺名字的旧式后端。"""

    name = "name_only"

    def capabilities(self):
        return frozenset(self._ref.capabilities())


class _WithDevice(_Delegate):
    name = "with_device"

    def device_info(self):
        return DeviceInfo(name="fake", runtime=1, total_memory_bytes=1 << 30)


class _WithBoth(_WithDevice):
    name = "with_both"

    def memory_stats(self):
        return MemoryStats(live_bytes=1, peak_bytes=2, total_bytes=3, requested_bytes=4)

    def allocator(self):
        return _FakeArena()


class _WithDeviceManager(_WithDevice):
    """多设备契约：三个方法齐了才算具备。"""

    name = "with_device_manager"

    def enumerate_devices(self):
        return (DeviceInfo(name="fake", runtime=1, total_memory_bytes=1 << 30),)

    def current_device(self):
        return 0

    def set_device(self, index):
        return None


class _WithStream(_WithBoth):
    name = "with_stream"

    def new_stream(self):
        return object()


class _WithCodegen(_WithDevice):
    """编译期契约：只要交出 ``kernel`` 就算声称具备。"""

    name = "with_codegen"

    def kernel(self, key, source, func):
        return (key, func)


class _WithTuning(_WithCodegen):
    """调优面：五个方法齐了才算具备。"""

    name = "with_tuning"

    def tunable_ops(self):
        return frozenset()

    def tuning_space(self, op):
        return None

    def set_tuning(self, op, config):
        return None

    def clear_tuning(self, op=None):
        return None

    def tuning_config(self):
        return {}


class _FakeArena:
    """``AllocatorContract`` 的最小实现：只记字节数，不碰显存。"""

    def __init__(self):
        self.live = 0
        self.total = 0

    def malloc(self, size):
        self.live += size
        self.total += size
        return self.live

    def free(self, ptr):
        return None

    def stats(self):
        return MemoryStats(
            live_bytes=self.live, peak_bytes=self.live,
            total_bytes=self.total, requested_bytes=self.total,
        )

    def reset_stats(self):
        return None

    def close(self):
        return None


class TestProbing(unittest.TestCase):
    def test_minimal_backend_still_valid(self):
        ref = ReferenceBackend()
        self.assertIsInstance(ref, Backend)
        self.assertEqual(implemented_contracts(ref), ())

    def test_device_only(self):
        backend = _WithDevice()
        self.assertTrue(supports(backend, DeviceContract))
        self.assertFalse(supports(backend, MemoryContract))
        self.assertEqual(implemented_contracts(backend), ("device",))

    def test_device_manager_is_separate(self):
        """多设备与"这块设备是什么"分开：三个方法齐了才多报一项。"""
        backend = _WithDeviceManager()
        self.assertTrue(supports(backend, DeviceManagerContract))
        self.assertFalse(supports(_WithDevice(), DeviceManagerContract))
        self.assertEqual(
            implemented_contracts(backend), ("device", "device_manager")
        )
        # 枚举设备不隐含内存统计，也不隐含开流
        self.assertFalse(supports(backend, MemoryContract))
        self.assertFalse(supports(backend, StreamContract))

    def test_device_manager_requires_all_three_methods(self):
        """少一个就不算——只做枚举、不能切卡的后端不是设备管理器。"""

        class OnlyEnumerate:
            def enumerate_devices(self):
                return ()

            def current_device(self):
                return 0

        self.assertFalse(supports(OnlyEnumerate(), DeviceManagerContract))

    def test_gpu_backends_declare_device_manager(self):
        """能力声明不建设备上下文也能读：两家都声称实现多设备契约。"""
        self.assertTrue(supports(CudaBackend, DeviceManagerContract))
        self.assertTrue(supports(HipBackend, DeviceManagerContract))
        self.assertFalse(supports(ReferenceBackend(), DeviceManagerContract))

    def test_both_in_declaration_order(self):
        self.assertEqual(implemented_contracts(_WithBoth()), ("device", "memory"))

    def test_stream_is_its_own_contract(self):
        """流管理与设备信息分开：有 ``new_stream`` 才多报一项，顺序按声明表。"""
        backend = _WithStream()
        self.assertTrue(supports(backend, StreamContract))
        self.assertEqual(
            implemented_contracts(backend), ("device", "memory", "stream")
        )
        # 只有 device_info 的后端不该被当成"能开流"
        self.assertFalse(supports(_WithDevice(), StreamContract))

    def test_allocator_is_a_value_contract(self):
        """``AllocatorContract`` 挂在后端交出的对象上，不挂在后端自己身上。"""
        arena = _WithBoth().allocator()
        self.assertTrue(supports(arena, AllocatorContract))
        self.assertEqual(arena.stats().live_bytes, 0)
        self.assertGreater(arena.malloc(64), 0)
        # 参考后端既没有内存统计，也就交不出分配器
        self.assertFalse(supports(ReferenceBackend(), MemoryContract))

    def test_codegen_and_tuning_are_independent(self):
        """编译期与调优面各自独立：有 ``kernel`` 不等于会调优，反之亦然。"""
        self.assertEqual(
            implemented_contracts(_WithCodegen()), ("device", "codegen")
        )
        self.assertEqual(
            implemented_contracts(_WithTuning()), ("device", "codegen", "tuning")
        )
        self.assertFalse(supports(_WithCodegen(), TuningContract))
        self.assertFalse(supports(_WithTuning(), MemoryContract))
        # 参考后端什么都不声称
        self.assertFalse(supports(ReferenceBackend(), CodegenContract))
        self.assertFalse(supports(ReferenceBackend(), TuningContract))

    def test_tuning_requires_all_five_methods(self):
        """少一个就不算——契约是整组能力，不是"有一个搜索方法"。"""

        class Partial:
            def tunable_ops(self):
                return frozenset()

            def tuning_space(self, op):
                return None

            def set_tuning(self, op, config):
                return None

            def clear_tuning(self, op=None):
                return None

            # 缺 tuning_config

        self.assertFalse(supports(Partial(), TuningContract))
        self.assertFalse(supports(_WithDevice(), CodegenContract))

    def test_probe_is_structural_only(self):
        """探测不查签名：返回元组的方法也会被认成 DeviceContract。

        这是 ``runtime_checkable`` 的既定行为，不是缺陷——契约的正确性靠
        实际调用与 conformance 证明，不靠 isinstance。
        """

        class WrongSignature:
            def device_info(self):
                return ("gpu", 1, 2)

        self.assertTrue(supports(WrongSignature(), DeviceContract))


class TestValueTypes(unittest.TestCase):
    def test_device_info_is_frozen(self):
        info = DeviceInfo(name="RTX 2080 Ti", runtime=12090, total_memory_bytes=11 * 1024**3)
        self.assertEqual(info.total_memory_gb, 11.0)
        with self.assertRaises(FrozenInstanceError):
            info.name = "别的卡"

    def test_memory_stats_is_frozen(self):
        stats = MemoryStats(live_bytes=1, peak_bytes=2, total_bytes=3, requested_bytes=4)
        with self.assertRaises(FrozenInstanceError):
            stats.peak_bytes = 0


class TestReportListsContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 单用例即可验证报告格式，不必跑全套 28 个
        cls.case = next(c for c in default_cases() if c.name == "relu")

    def test_minimal_backend_reports_none(self):
        report = run_conformance(ReferenceBackend(), cases=[self.case])
        self.assertEqual(report.contracts, ())
        self.assertIn("contracts", report.text())
        self.assertIn("仅最小面", report.text())
        self.assertTrue(report.ok, report.text())

    def test_contracts_are_informational(self):
        report = run_conformance(_WithBoth(), cases=[self.case])
        self.assertEqual(report.contracts, ("device", "memory"))
        self.assertIn("device, memory", report.text())
        self.assertTrue(report.ok, report.text())

    def test_missing_contract_is_not_a_failure(self):
        report = run_conformance(_Delegate(), cases=[self.case])
        self.assertTrue(report.ok, report.text())
        self.assertEqual(report.contracts, ())


@unittest.skipUnless(CUDA_AVAILABLE, "需要 cupy 与 CUDA 设备")
class TestCudaContracts(unittest.TestCase):
    """CUDA 插件是这些契约的真实实现，必须在真机上验证。"""

    @classmethod
    def setUpClass(cls):
        cls.backend = CudaBackend()

    def test_reports_device_and_memory(self):
        self.assertEqual(
            implemented_contracts(self.backend),
            ("device", "device_manager", "codegen", "tuning", "memory", "stream"),
        )

    def test_enumerates_devices(self):
        """枚举按索引顺序，且不改变当前设备；越界索引当场报错。"""
        devices = self.backend.enumerate_devices()
        self.assertGreaterEqual(len(devices), 1)
        self.assertEqual(self.backend.current_device(), 0)
        self.assertEqual(devices[0].name, self.backend.device_info().name)
        # 每项都是完整信息，不是占位
        self.assertGreater(devices[0].total_memory_bytes, 0)
        with self.assertRaises(BackendCapabilityError):
            self.backend.set_device(len(devices))
        with self.assertRaises(BackendCapabilityError):
            self.backend.set_device(-1)
        # 报错之后仍在原来的卡上
        self.assertEqual(self.backend.current_device(), 0)

    def test_kernel_entry_compiles_source(self):
        """``kernel`` 是 codegen 契约的公开入口，不是空壳转发。"""
        source = (
            'extern "C" __global__ void af_contract_probe(float* x) {\n'
            "  x[threadIdx.x] = 1.0f;\n"
            "}\n"
        )
        handle = self.backend.kernel(("contract-probe",), source, "af_contract_probe")
        # 同一个 key 再取一次必须命中缓存，拿到同一个句柄
        self.assertIs(
            self.backend.kernel(("contract-probe",), source, "af_contract_probe"),
            handle,
        )
        self.assertTrue(callable(handle))
        # 公开入口统一提供 launch：CUDA 的原生句柄只有 __call__，HIP 的只有
        # launch，包装器把两侧收口（5.168）
        self.assertTrue(callable(handle.launch))

    def test_allocator_satisfies_contract(self):
        """捕获用的分配器就是 ``allocator()`` 交出来的那个对象。"""
        arena = self.backend.allocator()
        self.assertTrue(supports(arena, AllocatorContract))
        self.assertEqual(arena.stats().total_bytes, 0)
        arena.close()

    def test_device_info_is_populated(self):
        info = self.backend.device_info()
        self.assertIsInstance(info, DeviceInfo)
        self.assertTrue(info.name)
        self.assertGreater(info.total_memory_bytes, 0)
        self.assertGreater(info.total_memory_gb, 0)

    def test_memory_stats_none_before_capture(self):
        """没捕获过就没有分层分配器可报告。不能拿"已分配字节"冒充峰值。"""
        self.assertIsNone(CudaBackend().memory_stats())

    def test_memory_stats_after_capture(self):
        g = Graph("add")
        a = g.input("a", f32[4, 8])
        b = g.input("b", f32[4, 8])
        g.output(g.add_node("add", [a, b]), "y")
        feed = {
            "a": np.ones((4, 8), dtype=np.float32),
            "b": np.ones((4, 8), dtype=np.float32),
        }
        self.backend.capture(g, feed, reuse_memory=True)

        stats = self.backend.memory_stats()
        self.assertIsInstance(stats, MemoryStats)
        # peak 含捕获前已存活的量，requested 只统计 reset 之后的新请求，两者不可比
        self.assertGreater(stats.peak_bytes, 0)
        self.assertGreater(stats.requested_bytes, 0)
        self.assertGreater(stats.live_bytes, 0)
        self.assertGreaterEqual(stats.total_bytes, stats.peak_bytes)


class TestCapabilityContract(unittest.TestCase):
    """能力声明从"算子的名字"扩展到"算子的形态"（审计 AF-32 第二点）。

    要钉住的是四件事：

    1. 集合语义不破——``BackendCapabilities`` 仍是 ``frozenset``，既有对账
       代码（``set()`` / ``len`` / ``==`` / 与静态算子表比较）不用改。
    2. 有明细就按明细判——dtype / axis 在构图期由 ``verify`` 指出，而不是等
       执行到一半才报。
    3. 用途承诺（反向 / 捕获 / 确定性）不来自图，只能显式查询。
    4. 只声明名字的后端行为完全不变，未细化不等于失败。
    """

    @staticmethod
    def _mixed_graph():
        g = Graph("mixed")
        x = g.input("x", f16[2, 4])
        r = g.add_node("reduce_sum", [x], {"axis": 0, "keepdims": False})
        g.output(g.add_node("add", [x, r]), "y")
        return g

    def test_reference_declares_more_than_names(self):
        caps = ReferenceBackend().capabilities()
        self.assertIsInstance(caps, frozenset)
        portable = {n for n, op in OPS.items() if op.portability is Portability.PORTABLE}
        self.assertEqual(set(caps), portable)
        self.assertEqual(caps.unrefined, frozenset(), "参考后端应当逐个给出形态明细")

        # backward 不能比实现更宽：没有 VJP 的算子必须声明不支持反向
        for name in caps:
            self.assertEqual(
                caps.describe(name).backward,
                OPS[name].vjp is not None,
                f"{name} 的反向声明与算子表不一致",
            )
        # 参考执行器没有图捕获路径；随机性全部来自显式 seed
        self.assertFalse(caps.describe("add").capture)
        self.assertTrue(caps.describe("add").deterministic)

    def test_set_semantics_survive_details(self):
        caps = BackendCapabilities(
            ["add", "relu"],
            details={"add": OpCapability(dtypes={"f32"})},
        )
        self.assertEqual(caps, frozenset(caps))
        self.assertEqual(hash(caps), hash(frozenset(caps)))
        self.assertEqual(sorted(caps), ["add", "relu"])
        self.assertIn("add", caps)
        self.assertEqual(len(caps), 2)
        self.assertEqual(caps.refined, frozenset({"add"}))
        self.assertEqual(caps.unrefined, frozenset({"relu"}))

    def test_details_must_name_declared_ops_and_be_op_capability(self):
        with self.assertRaises(ValueError):
            BackendCapabilities(["add"], details={"relu": OpCapability()})
        with self.assertRaises(TypeError):
            BackendCapabilities(["add"], details={"add": "f32"})

    def test_dtype_and_axis_are_rejected_at_verify_time(self):
        caps = BackendCapabilities(
            ["add", "reduce_sum"],
            details={
                "add": OpCapability(dtypes={"f32"}),
                "reduce_sum": OpCapability(axes={-1}),
            },
        )
        graph = self._mixed_graph()

        messages = [i.message for i in verify(graph, caps)]
        self.assertEqual(len(messages), 2, messages)
        joined = " ".join(messages)
        self.assertIn("dtype f16", joined)
        self.assertIn("axis [0]", joined)
        self.assertEqual(
            [name for name, _ in capability_gaps(graph, caps)],
            ["reduce_sum", "add"],
        )

        with self.assertRaises(VerifierError):
            require(graph, caps)

    def test_name_only_declaration_behaviour_is_unchanged(self):
        graph = self._mixed_graph()

        # 名字都在：dtype / axis 不参与判定
        self.assertEqual(verify(graph, {"add", "reduce_sum"}), [])

        # 名字缺一个：仍是原来的"不支持算子"错误
        issues = verify(graph, {"add"})
        self.assertEqual(len(issues), 1)
        self.assertIn("目标后端不支持算子 reduce_sum", issues[0].message)

    def test_use_requirements_are_queried_separately(self):
        caps = BackendCapabilities(
            ["add"],
            details={"add": OpCapability(backward=False, capture=False)},
        )
        g = Graph("use")
        a = g.input("a", f32[2, 2])
        b = g.input("b", f32[2, 2])
        g.output(g.add_node("add", [a, b]), "y")
        node = g.nodes[0]

        self.assertTrue(caps.supports_node(node))
        self.assertFalse(caps.supports_node(node, backward=True))
        self.assertFalse(caps.supports_node(node, capture=True))
        self.assertTrue(caps.supports_node(node, deterministic=True))

    def test_conformance_reports_refined_declaration(self):
        case = next(c for c in default_cases() if c.name == "relu")
        report = run_conformance(ReferenceBackend(), cases=[case])
        contract = next(c for c in report.checks if c.name == "contract")
        self.assertTrue(contract.ok, contract.detail)
        # 数量取自报告本身：加算子不需要回来改这行，测试只钉"带明细"这件事
        self.assertIn(f"{len(report.capabilities)} 个算子已声明", contract.detail)
        self.assertIn("带形态明细", contract.detail)
        self.assertTrue(report.ok, report.text())

    def test_conformance_keeps_unrefined_note(self):
        case = next(c for c in default_cases() if c.name == "relu")
        report = run_conformance(_NameOnly(), cases=[case])
        contract = next(c for c in report.checks if c.name == "contract")
        self.assertTrue(contract.ok, contract.detail)
        self.assertIn("仅算子名", contract.detail)


if __name__ == "__main__":
    unittest.main()
