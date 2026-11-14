"""NCCL 绑定：登记表、参数校验与真机四集合对拍。

两层：

- **离线**：dtype / 归约登记表、``ncclUniqueId`` 的字节往返、communicator 的
  非法 rank / world_size、主机数组与不连续数组的拒绝。这些不需要 GPU，也
  不需要加载库（参数校验发生在 ``_load`` 之前，或是纯函数）。
- **真机**：两个子进程各占一张卡，跑 AllReduce / AllGather / ReduceScatter /
  Broadcast，与主机侧期望值对拍。没有 NCCL 或不到两块卡时整组跳过。
"""

import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from aetherflow.backends.cuda import CUDA_AVAILABLE
from aetherflow.train import nccl

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
class TestCollectivesOnDevice(unittest.TestCase):
    """真机：两个进程各占一张卡，四个集合操作与主机期望值对拍。"""

    def test_four_collectives(self):
        world = 2
        with tempfile.TemporaryDirectory() as workdir:
            uid_path = os.path.join(workdir, "uid.bin")
            outs = [os.path.join(workdir, f"rank{rank}.npz") for rank in range(world)]
            env = dict(os.environ)
            env["PYTHONPATH"] = _ROOT + os.pathsep + env.get("PYTHONPATH", "")
            procs = [
                subprocess.Popen(
                    [
                        sys.executable, "-m", "tests._nccl_worker",
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
                        f"NCCL worker 退出码 {proc.returncode}：\n"
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


if __name__ == "__main__":
    unittest.main()
