"""NCCL / RCCL 绑定：登记表、参数校验、厂商判定与真机四集合对拍。

两层：

- **离线**：dtype / 归约登记表、``ncclUniqueId`` 的字节往返、communicator 的
  非法 rank / world_size、主机数组与不连续数组的拒绝、按库文件名判定厂商
  （``librccl`` → HIP）、以及能力声明的理由文本。这些不需要 GPU，也不需要
  加载库（参数校验发生在 ``_load`` 之前，或是纯函数）。
- **真机**：两个子进程各占一张卡，跑 AllReduce / AllGather / ReduceScatter /
  Broadcast，与主机侧期望值对拍。没有 NCCL / RCCL 或卡数不够时整组跳过。
  CUDA 用 cupy 数组（``_nccl_worker``），HIP 用项目自己的 ``DeviceArray``
  （``_rccl_worker``）；后者只有一块卡时按 world=1 跑，验证的是加载、版本、
  comm 初始化与四个调用面，而不是跨卡数值。
"""

import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from aetherflow.backends.cuda import CUDA_AVAILABLE
from aetherflow.train import nccl
from aetherflow.train.parallel import collective_support

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestRegistry(unittest.TestCase):
    """登记表就是覆盖声明：表内对上，表外显式拒绝。"""

    def test_float_dtypes_map_to_nccl_enums(self):
        self.assertEqual(nccl.data_type(np.float16), 6)
        self.assertEqual(nccl.data_type(np.float32), 7)
        self.assertEqual(nccl.data_type(np.float64), 8)
        self.assertEqual(nccl.data_type("float32"), 7)

    def test_bfloat16_is_refused_with_reason(self):
        """numpy 没有 bf16：主机侧算不出期望值，不能只凭 NCCL 支持就放开。"""
        with self.assertRaises(nccl.NcclUnsupported) as caught:
            nccl.data_type("bfloat16")
        self.assertIn("bf16", str(caught.exception))

    def test_integer_dtype_is_refused(self):
        with self.assertRaises(nccl.NcclUnsupported):
            nccl.data_type(np.int32)

    def test_reduction_names_map(self):
        self.assertEqual(nccl.red_op("sum"), 0)
        self.assertEqual(nccl.red_op("max"), 2)

    def test_average_is_refused(self):
        """ncclAvg 需要 2.10 以上，且等价于 sum 再除——不让开关藏在版本里。"""
        with self.assertRaises(nccl.NcclUnsupported) as caught:
            nccl.red_op("avg")
        self.assertIn("sum", str(caught.exception))


class TestUniqueId(unittest.TestCase):
    """``ncclUniqueId`` 是 128 字节不透明结构，跨进程只传字节。"""

    def test_round_trip(self):
        raw = bytes(range(128))
        value = nccl.NcclUniqueId.from_bytes(raw)
        self.assertEqual(value.to_bytes(), raw)

    def test_wrong_length_is_refused(self):
        with self.assertRaises(ValueError):
            nccl.NcclUniqueId.from_bytes(b"\x00" * 127)


class TestCommValidation(unittest.TestCase):
    """communicator 的入口校验：rank / world_size 不合法时不碰驱动。"""

    def test_world_size_must_be_positive(self):
        for value in (0, -1):
            with self.assertRaises(ValueError):
                nccl.NcclComm(0, value, b"\x00" * 128)

    def test_world_size_must_be_int(self):
        with self.assertRaises(TypeError):
            nccl.NcclComm(0, 2.0, b"\x00" * 128)

    def test_rank_must_be_in_range(self):
        for value in (-1, 2):
            with self.assertRaises(ValueError):
                nccl.NcclComm(value, 2, b"\x00" * 128)

    def test_rank_must_be_int(self):
        with self.assertRaises(TypeError):
            nccl.NcclComm(True, 2, b"\x00" * 128)

    @unittest.skipIf(nccl.available(), "本机有 NCCL：这条只验证库缺失时的错误")
    def test_missing_library_is_explicit(self):
        with self.assertRaises(nccl.NcclUnavailable) as caught:
            nccl.NcclComm(0, 1, b"\x00" * 128)
        self.assertIn("NCCL", str(caught.exception))


class TestInitRank(unittest.TestCase):
    """``init_rank`` 的价值只在"先切卡"：漏了这一步，第一次集合调用才炸。"""

    def _cupy(self, used):
        class _Device:
            def __init__(self, index):
                self.index = index

            def __enter__(self):
                used.append(self.index)

            def __exit__(self, *exc):
                return None

        return SimpleNamespace(cuda=SimpleNamespace(Device=_Device))

    def test_switches_to_requested_device_before_initializing(self):
        used = []
        with mock.patch.dict(sys.modules, {"cupy": self._cupy(used)}), \
                mock.patch.object(nccl, "NcclComm") as comm_cls:
            comm = nccl.init_rank(b"u" * 128, 1, 2, device=3)
        self.assertIs(comm, comm_cls.return_value)
        self.assertEqual(used, [3])
        comm_cls.assert_called_once_with(1, 2, b"u" * 128)

    def test_defaults_device_to_rank(self):
        used = []
        with mock.patch.dict(sys.modules, {"cupy": self._cupy(used)}), \
                mock.patch.object(nccl, "NcclComm"):
            nccl.init_rank(b"u" * 128, 1, 2)
        self.assertEqual(used, [1])

    def test_rejects_negative_device(self):
        with self.assertRaises(ValueError):
            nccl.init_rank(b"u" * 128, 0, 2, device=-1)


class _FakeBuffer:
    """只有 ``.data.ptr`` / ``.size`` / ``.dtype`` / ``.flags`` 的假设备数组。"""

    class _Data:
        ptr = 4096

    def __init__(self, size, dtype="float32", contiguous=True):
        self.size = size
        self.dtype = np.dtype(dtype)
        self.data = self._Data()
        self.flags = SimpleNamespace(c_contiguous=contiguous)


class TestBufferValidation(unittest.TestCase):
    """NCCL 只认连续设备缓冲：主机数组与不连续数组都要在调用前拒绝。"""

    def test_host_array_is_refused(self):
        with self.assertRaises(TypeError) as caught:
            nccl._pointer(np.zeros(4, np.float32), "输入")
        self.assertIn("data.ptr", str(caught.exception))

    def test_non_contiguous_is_refused(self):
        with self.assertRaises(TypeError):
            nccl._contiguous(_FakeBuffer(4, contiguous=False), "输入")

    def test_contiguous_device_buffer_passes(self):
        buffer = _FakeBuffer(4)
        self.assertIsNotNone(nccl._pointer(buffer, "输入"))
        nccl._contiguous(buffer, "输入")


class TestVendorDetection(unittest.TestCase):
    """厂商只由库文件名决定：不 import cupy、不碰设备。"""

    def test_rccl_names_are_hip(self):
        for path in (
            "librccl.so.2",
            "librccl.so.1",
            "/opt/rocm/lib/librccl.so.1",
            "C:\\Windows\\System32\\rccl.dll",
        ):
            self.assertEqual(nccl._vendor_from_path(path), "hip", path)

    def test_nccl_names_are_cuda(self):
        for path in (
            "libnccl.so.2",
            "nccl.dll",
            "/usr/lib/x86_64-linux-gnu/libnccl.so.2",
            "C:/Program Files/nccl/nccl.dll",
        ):
            self.assertEqual(nccl._vendor_from_path(path), "cuda", path)

    def test_candidates_cover_both_sonames(self):
        """NCCL 的 soname 是 .so.2，RCCL 的是 .so.1：两边都要在候选里。"""
        names = list(nccl._CANDIDATES)
        self.assertIn("libnccl.so.2", names)
        self.assertIn("librccl.so.1", names)
        self.assertIn("rccl.dll", names)


class TestVersionFormat(unittest.TestCase):
    def test_integer_version_becomes_dotted(self):
        self.assertEqual(nccl._format_version(22707), "2.27.7")
        self.assertEqual(nccl._format_version(21803), "2.18.3")
        self.assertEqual(nccl._format_version(10000), "1.0.0")

    @unittest.skipIf(nccl.available(), "本机有 NCCL / RCCL：这条只验证库缺失")
    def test_missing_library_makes_vendor_explicit(self):
        for call in (nccl.vendor, nccl.version_string):
            with self.assertRaises(nccl.NcclUnavailable) as caught:
                call()
            self.assertIn("NCCL", str(caught.exception))

    @unittest.skipUnless(nccl.available(), "本机没有 NCCL / RCCL")
    def test_loaded_vendor_and_version_are_consistent(self):
        self.assertIn(nccl.vendor(), ("cuda", "hip"))
        self.assertRegex(nccl.version_string(), r"^\d+\.\d+\.\d+$")


class _HipLikeBuffer:
    """``DeviceArray`` 的最小形态：只提供绑定层要的四个属性。"""

    class _Data:
        ptr = 8192

    def __init__(self, size=4):
        self.size = size
        self.dtype = np.dtype("float32")
        self.data = self._Data()
        self.flags = SimpleNamespace(c_contiguous=True)


class TestHipArrayShape(unittest.TestCase):
    """RCCL 的输入是 ``backends.hip`` 的 DeviceArray，不是 cupy 数组。"""

    def test_device_array_minimum_is_accepted(self):
        buffer = _HipLikeBuffer(8)
        ptr = nccl._pointer(buffer, "输入")
        self.assertEqual(ptr.value, 8192)
        nccl._contiguous(buffer, "输入")


class _FakeMultiDevice:
    """只满足"多设备后端"契约的替身：能力声明不看它，只看库。"""

    def enumerate_devices(self):
        return ()

    def current_device(self):
        return 0

    def set_device(self, index):
        return None

    def execute(self, graph, feed):
        raise NotImplementedError

    def to_array(self, value):
        return value

    def to_host(self, value):
        return value


class TestCollectiveCapability(unittest.TestCase):
    """声明跟着"库能不能加载"走：能 ⇒ 设备侧默认可用，不能 ⇒ 主机中转。"""

    def test_reason_names_the_loaded_library(self):
        with mock.patch.object(nccl, "vendor", return_value="hip"), mock.patch.object(
            nccl, "version_string", return_value="2.27.7"
        ):
            support = collective_support(_FakeMultiDevice())
        self.assertTrue(support.device_allreduce)
        self.assertEqual(support.provider, "nccl")
        self.assertIn("RCCL 2.27.7", support.reason)
        self.assertIn("NcclDeviceCollective.from_env", support.reason)

    def test_reason_says_when_library_is_absent(self):
        with mock.patch.object(
            nccl, "vendor", side_effect=nccl.NcclUnavailable("没有库")
        ):
            support = collective_support(_FakeMultiDevice())
        self.assertFalse(support.device_allreduce)
        self.assertIsNone(support.provider)
        self.assertIn("加载不到", support.reason)

    def test_cuda_vendor_keeps_nccl_name(self):
        with mock.patch.object(nccl, "vendor", return_value="cuda"), mock.patch.object(
            nccl, "version_string", return_value="2.18.3"
        ):
            support = collective_support(_FakeMultiDevice())
        self.assertIn("NCCL 2.18.3", support.reason)


def _device_count():
    if not CUDA_AVAILABLE:
        return 0
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:  # noqa: BLE001
        return 0


def _hip_device_count():
    """HIP 侧的设备数；没有 ROCm 时为 0，不抛错。"""
    try:
        from aetherflow.backends.hip import runtime

        return int(runtime.device_count())
    except Exception:  # noqa: BLE001
        return 0


def _vendor_or_none():
    try:
        return nccl.vendor()
    except nccl.NcclUnavailable:
        return None


class _CollectiveWorkerCheck:
    """两份真机 worker 共用的启动与对拍逻辑（world 泛化到 1）。"""

    WORKER = ""

    def test_four_collectives(self):
        world = self.WORLD
        with tempfile.TemporaryDirectory() as workdir:
            uid_path = os.path.join(workdir, "uid.bin")
            outs = [os.path.join(workdir, f"rank{rank}.npz") for rank in range(world)]
            env = dict(os.environ)
            env["PYTHONPATH"] = _ROOT + os.pathsep + env.get("PYTHONPATH", "")
            procs = [
                subprocess.Popen(
                    [
                        sys.executable, "-m", self.WORKER,
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
                        f"{self.WORKER} 退出码 {proc.returncode}：\n"
                        + "\n".join(logs)
                    )
            results = [
                dict(np.load(path, allow_pickle=False)) for path in outs
            ]

        self._check_allreduce(results, world)
        self._check_allgather(results, world)
        self._check_reduce_scatter(results, world)
        self._check_broadcast(results, world)

    def _check_allreduce(self, results, world):
        rng = np.arange(8, dtype=np.float32)
        expected = np.zeros(8, np.float32)
        for rank in range(world):
            expected += rng + rank
        for rank in range(world):
            np.testing.assert_allclose(
                results[rank]["allreduce"], expected, rtol=0, atol=1e-5
            )

    def _check_allgather(self, results, world):
        expected = np.concatenate(
            [np.arange(4, dtype=np.float32) + rank * 10 for rank in range(world)]
        )
        for rank in range(world):
            np.testing.assert_allclose(
                results[rank]["allgather"], expected, rtol=0, atol=1e-5
            )

    def _check_reduce_scatter(self, results, world):
        base = np.arange(4 * world, dtype=np.float32)
        for rank in range(world):
            segment = base[rank * 4 : (rank + 1) * 4]
            expected = segment * world + 100 * sum(range(world))
            np.testing.assert_allclose(
                results[rank]["reduce_scatter"], expected, rtol=0, atol=1e-5
            )

    def _check_broadcast(self, results, world):
        expected = np.arange(6, dtype=np.float32)
        for rank in range(world):
            np.testing.assert_allclose(
                results[rank]["broadcast"], expected, rtol=0, atol=1e-5
            )


@unittest.skipUnless(
    CUDA_AVAILABLE and _vendor_or_none() == "cuda" and _device_count() >= 2,
    "需要 NCCL 与至少两块 CUDA 设备",
)
class TestCollectivesOnDevice(_CollectiveWorkerCheck, unittest.TestCase):
    """真机：两个进程各占一张卡，四个集合操作与主机期望值对拍。"""

    WORKER = "tests._nccl_worker"
    WORLD = 2


@unittest.skipUnless(
    _vendor_or_none() == "hip" and _hip_device_count() >= 1,
    "需要 RCCL 与至少一块 HIP 设备",
)
class TestRcclOnDevice(_CollectiveWorkerCheck, unittest.TestCase):
    """真机：RCCL 单 rank 跑通加载 / 版本 / comm 初始化与四个调用面。

    跨卡数值只能靠 ``TestCollectivesOnDevice``；这里在一块卡上验的是
    "同一份 ctypes 绑定确实能被 RCCL 接住"，world=1 时期望值等于输入。
    """

    WORKER = "tests._rccl_worker"
    WORLD = 1


if __name__ == "__main__":
    unittest.main()
