"""执行期调度策略（5.333）：配置库按形状命中并接到 launch 上。

这一组在没有 GPU 的机器上钉住三件事：

1. 库的命中判据与离线写库脚本用的是**同一个**签名函数（否则库里有记录、
   运行时却永远不命中）；
2. 优先级：显式 ``set_tuning`` → 构造参数 ``block`` → 配置库 → 自动；
3. 坏记录（非 2 的幂、非正整数）明确报错，不静默退回默认。
"""

import json
import unittest

from aetherflow.backends.device_backend import DeviceBackend
from aetherflow.tuning import ConfigStore, SchedulePolicy, TuningRecord
from aetherflow.tuning.signature import block_signature, conv2d_signature


class _FakeDevice:
    def __init__(self, name):
        self.name = name


class _FakeBackend:
    """只提供 ``from_backend`` 需要的两项：名字与设备信息。"""

    name = "cuda"

    def __init__(self, device="RTX 2080 Ti"):
        self._device = device

    def device_info(self):
        return _FakeDevice(self._device)


class _Wrapper:
    def __init__(self, inner):
        self.inner = inner


def _record(op, signature, config, *, backend="cuda", device="RTX 2080 Ti"):
    return TuningRecord(
        backend=backend,
        device=device,
        op=op,
        signature=signature,
        config=config,
        median_ms=1.0,
    )


def _bare_backend(store=None, *, block=None, name="cuda"):
    """绕过构造的 ``DeviceBackend``：只装 ``_launch_block`` 用到的字段。

    这里刻意不启动硬件——要验证的是"优先级对不对"，不是"这张卡快不快"。
    """
    backend = object.__new__(DeviceBackend)
    backend.name = name
    backend.block = block
    backend.sm_count = 4
    backend._blocks = {}
    backend._launch_cache = {}
    backend._conv_tile = None
    backend._conv_roc = None
    backend._conv_cb = None
    backend._config_epoch = 0
    backend._graph_plans = {}
    backend._graph_failures = {}
    backend._schedule_policy = None
    backend.device_info = lambda: _FakeDevice("RTX 2080 Ti")
    if store is not None:
        backend.set_schedule(store)
    return backend


class TestSignatures(unittest.TestCase):
    def test_block_signature_has_only_work_and_vec(self):
        self.assertEqual(block_signature("add", work=4096), {"work": 4096, "vec": 1})
        self.assertEqual(
            block_signature("add", work=4096, vec=4), {"work": 4096, "vec": 4}
        )

    def test_block_signature_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            block_signature("", work=1)
        with self.assertRaises(ValueError):
            block_signature("add", work=-1)
        with self.assertRaises(ValueError):
            block_signature("add", work=1, vec=0)

    def test_conv2d_signature_is_stable_across_tuple_and_list(self):
        left = conv2d_signature((2, 3, 8, 8), (4, 3, 3, 3), stride=(2, 2), dtype="f32")
        right = conv2d_signature([2, 3, 8, 8], [4, 3, 3, 3], stride=[2, 2], dtype="f32")
        self.assertEqual(left, right)
        # 配置库按规范 JSON 比较签名：往返一次不能变成另一种东西
        self.assertEqual(json.loads(json.dumps(left)), left)
        self.assertEqual(left["stride"], [2, 2])
        self.assertEqual(left["groups"], 1)

    def test_conv2d_signature_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):
            conv2d_signature((2, 3, 8), (4, 3, 3, 3))
        with self.assertRaises(ValueError):
            conv2d_signature((2, 3, 8, 8), (4, 3, 3, 3), stride=(2,))
        with self.assertRaises(ValueError):
            conv2d_signature((2, 3, 8, 8), (4, 3, 3, 3), groups=0)


class TestSchedulePolicy(unittest.TestCase):
    def _store(self, **config):
        store = ConfigStore()
        store.put(
            _record("add", block_signature("add", work=4096), config or {"block": 256})
        )
        return store

    def test_lookup_hits(self):
        policy = SchedulePolicy.from_backend(_FakeBackend(), self._store())
        self.assertEqual(policy.backend, "cuda")
        self.assertEqual(policy.device, "RTX 2080 Ti")
        self.assertEqual(policy.lookup("add", block_signature("add", work=4096)), {"block": 256})

    def test_lookup_miss(self):
        policy = SchedulePolicy.from_backend(_FakeBackend(), self._store())
        self.assertIsNone(policy.lookup("add", block_signature("add", work=1024)))
        self.assertEqual(policy.stats()["miss"], 1)

    def test_disabled_lookup_is_not_a_miss(self):
        store = self._store()
        policy = SchedulePolicy.from_backend(_FakeBackend(), store, enabled=False)
        self.assertIsNone(policy.lookup("add", block_signature("add", work=4096)))
        self.assertEqual(policy.stats()["lookup"], 0)
        # 关掉之后可以再打开，不需要重建
        policy.enabled = True
        self.assertEqual(policy.lookup("add", block_signature("add", work=4096)), {"block": 256})

    def test_repeated_lookup_is_cached(self):
        policy = SchedulePolicy.from_backend(_FakeBackend(), self._store())
        first = policy.lookup("add", block_signature("add", work=4096))
        second = policy.lookup("add", block_signature("add", work=4096))
        self.assertEqual(first, second)
        self.assertIsNot(first, second)  # 副本：调用方改写不能污染缓存
        stats = policy.stats()
        self.assertEqual(stats["lookup"], 2)
        self.assertEqual(stats["hit"], 1)
        self.assertEqual(stats["cached"], 1)

    def test_empty_signature_is_rejected(self):
        policy = SchedulePolicy.from_backend(_FakeBackend(), self._store())
        with self.assertRaises(ValueError):
            policy.lookup("add", {})

    def test_from_backend_strips_wrapper(self):
        policy = SchedulePolicy.from_backend(_Wrapper(_FakeBackend()), self._store())
        self.assertEqual(policy.backend, "cuda")

    def test_from_backend_device_override(self):
        policy = SchedulePolicy.from_backend(_FakeBackend(), self._store(), device="别的卡")
        self.assertEqual(policy.device, "别的卡")
        self.assertIsNone(policy.lookup("add", block_signature("add", work=4096)))

    def test_missing_device_info_becomes_unknown(self):
        class _NoInfo:
            name = "cuda"

        policy = SchedulePolicy.from_backend(_NoInfo(), self._store())
        self.assertEqual(policy.device, "unknown")

    def test_invalid_arguments(self):
        with self.assertRaises(TypeError):
            SchedulePolicy({}, backend="cuda", device="x")
        with self.assertRaises(ValueError):
            SchedulePolicy(ConfigStore(), backend="", device="x")
        with self.assertRaises(ValueError):
            SchedulePolicy(ConfigStore(), backend="cuda", device="")

    def test_overlay_overrides_known_keys(self):
        policy = SchedulePolicy.from_backend(_FakeBackend(), ConfigStore())
        merged = policy.overlay(
            {"tile_w": 32, "roc": 4}, {"tile_w": None, "roc": None, "cb": None}
        )
        self.assertEqual(merged, {"tile_w": 32, "roc": 4, "cb": None})

    def test_overlay_ignores_unknown_keys(self):
        policy = SchedulePolicy.from_backend(_FakeBackend(), ConfigStore())
        merged = policy.overlay({"block": 128, "grid": 8}, {"block": None})
        self.assertEqual(merged, {"block": 128})
        self.assertEqual(policy.stats()["unknown"], 1)

    def test_overlay_rejects_bad_values(self):
        """库里的记录等于预置的调优覆盖：写坏了要在应用时暴露。"""
        policy = SchedulePolicy.from_backend(_FakeBackend(), ConfigStore())
        for value in (0, -4, True, 1.5, "128"):
            with self.assertRaises(ValueError):
                policy.overlay({"block": value}, {"block": None})

    def test_none_in_overlay_means_auto(self):
        policy = SchedulePolicy.from_backend(_FakeBackend(), ConfigStore())
        self.assertEqual(policy.overlay({"block": None}, {"block": 128}), {"block": None})


class TestBackendWiring(unittest.TestCase):
    def test_launch_block_uses_store_when_auto(self):
        store = ConfigStore()
        store.put(_record("add", block_signature("add", work=4096), {"block": 128}))
        backend = _bare_backend(store)
        self.assertEqual(backend._launch_block("add", 4096), 128)

    def test_explicit_tuning_wins_over_store(self):
        store = ConfigStore()
        store.put(_record("add", block_signature("add", work=4096), {"block": 128}))
        backend = _bare_backend(store)
        backend._blocks["add"] = 512
        self.assertEqual(backend._launch_block("add", 4096), 512)

    def test_constructor_block_wins_over_store(self):
        store = ConfigStore()
        store.put(_record("add", block_signature("add", work=4096), {"block": 128}))
        backend = _bare_backend(store, block=1024)
        self.assertEqual(backend._launch_block("add", 4096), 1024)

    def test_miss_falls_back_to_auto(self):
        backend = _bare_backend(ConfigStore())
        plain = _bare_backend()
        self.assertEqual(
            backend._launch_block("add", 4096), plain._launch_block("add", 4096)
        )

    def test_reduce_block_uses_store(self):
        store = ConfigStore()
        store.put(_record("sum", block_signature("sum", work=1024), {"block": 64}))
        backend = _bare_backend(store)
        self.assertEqual(backend._reduce_block("sum", 1024), 64)

    def test_invalid_block_in_store_raises(self):
        """非 2 的幂的 block 会让内核出错，不能静默退回自动。"""
        store = ConfigStore()
        store.put(_record("add", block_signature("add", work=4096), {"block": 96}))
        backend = _bare_backend(store)
        with self.assertRaises(ValueError):
            backend._launch_block("add", 4096)

    def test_clear_schedule_restores_auto(self):
        store = ConfigStore()
        store.put(_record("add", block_signature("add", work=4096), {"block": 128}))
        backend = _bare_backend(store)
        self.assertEqual(backend._launch_block("add", 4096), 128)
        backend.clear_schedule()
        self.assertIsNone(backend.schedule_policy())
        self.assertEqual(
            backend._launch_block("add", 4096),
            _bare_backend()._launch_block("add", 4096),
        )

    def test_set_schedule_returns_policy_and_invalidates(self):
        backend = _bare_backend()
        backend._launch_cache[("add", 4096, 1)] = (256, 16)
        policy = backend.set_schedule(ConfigStore())
        self.assertIs(backend.schedule_policy(), policy)
        self.assertEqual(backend._config_epoch, 1)
        self.assertEqual(backend._launch_cache, {})

    def test_conv2d_record_only_used_when_all_knobs_auto(self):
        """conv2d 的三个旋钮只要有一项被手动设定，就不查库。"""
        store = ConfigStore()
        store.put(
            _record(
                "conv2d",
                conv2d_signature((1, 3, 8, 8), (4, 3, 3, 3), dtype="f32"),
                {"tile_w": 32, "roc": 2, "cb": 16},
            )
        )
        backend = _bare_backend(store)
        policy = backend.schedule_policy()
        signature = conv2d_signature((1, 3, 8, 8), (4, 3, 3, 3), dtype="f32")
        self.assertIsNotNone(policy.lookup("conv2d", signature))
        merged = policy.overlay(
            policy.lookup("conv2d", signature),
            {"tile_w": None, "roc": None, "cb": None},
        )
        self.assertEqual(merged, {"tile_w": 32, "roc": 2, "cb": 16})


if __name__ == "__main__":
    unittest.main()
