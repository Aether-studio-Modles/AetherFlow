"""多 rank 启动：环境变量约定、unique id 分发、进程拉起。

三层判据：

- **环境变量**：``AF_*`` 优先、torchrun 的 ``RANK`` / ``WORLD_SIZE`` 兜底；
  两者都没有时要求调用方给默认值，否则报错——猜 0 会让 W 个进程抢同一个
  rank，症状是 NCCL 挂住。
- **unique id 分发**：128 字节原样往返；长度不对、rank 0 来拉、连不上、
  服务端重复启动都是显式错误。
- **拉起进程**：``spawn`` 起真进程，每个进程拿到自己的 rank 与同一份 uid；
  子进程非 0 退出时父进程抛错并带上它的输出。
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from aetherflow.train.distributed import (
    UID_SIZE,
    UniqueIdServer,
    env_local_rank,
    env_master_port,
    env_rank,
    env_world_size,
    fetch_unique_id,
    spawn,
)

_WORKER = "tests._spawn_worker:main"
_BOOM = "tests._spawn_worker:boom"


class EnvTest(unittest.TestCase):
    """环境变量读取：顺序、回退与"缺失就报错"。"""

    def setUp(self):
        keys = [
            "AF_RANK",
            "AF_WORLD_SIZE",
            "AF_LOCAL_RANK",
            "AF_MASTER_PORT",
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "MASTER_PORT",
        ]
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in keys:
            os.environ.pop(key, None)

    def test_prefers_af_prefix(self):
        os.environ["AF_RANK"] = "2"
        os.environ["RANK"] = "7"
        self.assertEqual(env_rank(), 2)

    def test_falls_back_to_torchrun_names(self):
        os.environ["RANK"] = "3"
        os.environ["WORLD_SIZE"] = "4"
        os.environ["LOCAL_RANK"] = "1"
        os.environ["MASTER_PORT"] = "29500"
        self.assertEqual(env_rank(), 3)
        self.assertEqual(env_world_size(), 4)
        self.assertEqual(env_local_rank(), 1)
        self.assertEqual(env_master_port(), 29500)

    def test_missing_without_default_raises(self):
        with self.assertRaises(RuntimeError):
            env_rank()
        with self.assertRaises(RuntimeError):
            env_world_size()

    def test_default_is_used_when_absent(self):
        self.assertEqual(env_rank(0), 0)
        self.assertEqual(env_world_size(1), 1)
        self.assertEqual(env_local_rank(0), 0)

    def test_non_integer_is_rejected(self):
        os.environ["AF_RANK"] = "abc"
        with self.assertRaises(ValueError):
            env_rank()


class UniqueIdTest(unittest.TestCase):
    """uid 分发：正常往返与四类拒绝面。"""

    def _serve(self, world_size, uid, timeout=10.0):
        server = UniqueIdServer(world_size, timeout=timeout)
        server.serve(uid)
        self.addCleanup(server.close)
        return server

    def test_round_trip(self):
        uid = bytes(range(UID_SIZE))
        server = self._serve(3, uid)
        payloads = [
            fetch_unique_id(server.host, server.port, rank, timeout=10.0)
            for rank in (1, 2)
        ]
        server.wait(timeout=10.0)
        for payload in payloads:
            self.assertEqual(payload, uid)
        self.assertEqual(server.delivered, 2)

    def test_wrong_uid_length_is_rejected(self):
        server = UniqueIdServer(2)
        self.addCleanup(server.close)
        with self.assertRaises(ValueError):
            server.serve(b"short")

    def test_double_serve_is_rejected(self):
        server = self._serve(2, b"x" * UID_SIZE)
        with self.assertRaises(RuntimeError):
            server.serve(b"y" * UID_SIZE)

    def test_rank_zero_cannot_fetch(self):
        with self.assertRaises(ValueError):
            fetch_unique_id("127.0.0.1", 1, 0, timeout=0.5)

    def test_rank_type_is_checked(self):
        with self.assertRaises(TypeError):
            fetch_unique_id("127.0.0.1", 1, True, timeout=0.5)

    def test_closed_server_is_a_runtime_error(self):
        """服务端已关：Linux 是连接重置、Windows 是等待超时，两者都报 RuntimeError。"""
        server = UniqueIdServer(2)
        port = server.port
        server.close()
        with self.assertRaises(RuntimeError):
            fetch_unique_id("127.0.0.1", port, 1, timeout=0.5)

    def test_world_size_validation(self):
        with self.assertRaises(TypeError):
            UniqueIdServer(True)
        with self.assertRaises(ValueError):
            UniqueIdServer(0)


class SpawnTest(unittest.TestCase):
    """真进程：rank 分发、同一份 uid、失败不静默。"""

    def test_spawn_delivers_rank_and_uid(self):
        with tempfile.TemporaryDirectory() as folder:
            codes = spawn(_WORKER, 3, folder, timeout=120.0)
            self.assertEqual(codes, [0, 0, 0])
            rows = []
            for rank in range(3):
                with open(
                    os.path.join(folder, f"rank{rank}.json"), encoding="utf-8"
                ) as handle:
                    rows.append(json.load(handle))
        self.assertEqual(sorted(row["rank"] for row in rows), [0, 1, 2])
        self.assertTrue(all(row["world"] == 3 for row in rows))
        uids = {row["uid"] for row in rows if row["rank"] != 0}
        self.assertEqual(len(uids), 1)
        self.assertEqual(len(next(iter(uids))) // 2, UID_SIZE)

    def test_child_failure_is_raised(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "unused")
            with self.assertRaises(RuntimeError) as caught:
                spawn(_BOOM, 2, path, timeout=60.0)
        self.assertIn("退出码 3", str(caught.exception))

    def test_bad_entrypoint_is_rejected(self):
        with self.assertRaises(ValueError):
            spawn("tests._spawn_worker", 1, "x")
        with self.assertRaises(ValueError):
            spawn("tests._spawn_worker:", 1, "x")

    def test_non_string_argument_is_rejected(self):
        with self.assertRaises(TypeError):
            spawn(_WORKER, 1, 123)

    def test_world_size_validation(self):
        with self.assertRaises(TypeError):
            spawn(_WORKER, True, "x")
        with self.assertRaises(ValueError):
            spawn(_WORKER, 0, "x")


if __name__ == "__main__":
    unittest.main()
