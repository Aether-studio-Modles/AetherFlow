"""自动调优的本地测试：参数空间、配置库、搜索器。

这一组要在**没有 GPU** 的机器上把"调优逻辑"钉住，因为真机只回答
"这个 block 快不快"，不回答"配置库会不会命中错设备"或"全不可行时会不会
假装成功"。CUDA 侧的应用与对拍在 ``test_cuda_backend`` 里另行验证。
"""

import json
import os
import tempfile
import time
import unittest

from aetherflow.tuning import (
    AllRejectedError,
    ConfigStore,
    TuningRecord,
    TuningSpace,
    canonical_signature,
    tune,
    tune_paired,
    tune_replicated,
    tune_staged,
)


class TestTuningSpace(unittest.TestCase):
    def test_empty_space_is_rejected(self):
        with self.assertRaises(ValueError):
            TuningSpace.build({})

    def test_empty_candidates_are_rejected(self):
        with self.assertRaises(ValueError):
            TuningSpace.build({"block": ()})

    def test_duplicate_candidates_are_rejected(self):
        with self.assertRaises(ValueError):
            TuningSpace.build({"block": (128, 128)})

    def test_unhashable_candidates_are_rejected(self):
        with self.assertRaises(TypeError):
            TuningSpace.build({"block": ([128],)})

    def test_expansion_order_is_deterministic(self):
        space = TuningSpace.build({"a": (1, 2), "b": ("x", "y")})
        configs = list(space.configs())
        self.assertEqual(
            configs,
            [
                {"a": 1, "b": "x"},
                {"a": 1, "b": "y"},
                {"a": 2, "b": "x"},
                {"a": 2, "b": "y"},
            ],
        )
        self.assertEqual(space.size(), 4)
        self.assertEqual(list(space.configs()), configs)

    def test_default_is_first_candidate(self):
        space = TuningSpace.build({"block": (256, 128)})
        self.assertEqual(space.default(), {"block": 256})
        self.assertEqual(space.names(), ("block",))
        self.assertIn("block", space)
        self.assertNotIn("grid", space)

    def test_unknown_parameter_raises(self):
        with self.assertRaises(KeyError):
            TuningSpace.build({"block": (128,)}).values("grid")

    def test_to_dict_roundtrip(self):
        space = TuningSpace.build({"block": (128, 256)})
        self.assertEqual(space.to_dict(), {"block": [128, 256]})


class TestCanonicalSignature(unittest.TestCase):
    def test_tuple_and_list_are_normalized(self):
        self.assertEqual(
            canonical_signature(inputs=((4, 8),), dtype="f32"),
            {"inputs": [[4, 8]], "dtype": "f32"},
        )
        self.assertEqual(
            canonical_signature(inputs=[(4, 8)], dtype="f32"),
            {"inputs": [[4, 8]], "dtype": "f32"},
        )

    def test_none_is_dropped(self):
        self.assertEqual(canonical_signature(dtype="f32", extra=None), {"dtype": "f32"})


class TestConfigStore(unittest.TestCase):
    def _record(self, **overrides):
        data = {
            "backend": "cuda",
            "device": "RTX 2080 Ti",
            "op": "add",
            "signature": {"dtype": "f32", "inputs": [[4, 8]]},
            "config": {"block": 512},
            "median_ms": 0.25,
            "best_ms": 0.22,
            "p95_ms": 0.31,
            "samples": 3,
        }
        data.update(overrides)
        return TuningRecord(**data)

    def test_put_and_get(self):
        store = ConfigStore()
        record = store.put(self._record())
        hit = store.get("cuda", "RTX 2080 Ti", "add", {"inputs": [[4, 8]], "dtype": "f32"})
        self.assertIs(hit, record)
        self.assertEqual(len(store), 1)

    def test_missing_device_does_not_hit(self):
        store = ConfigStore()
        store.put(self._record())
        self.assertIsNone(store.get("cuda", "别的卡", "add", {"dtype": "f32"}))

    def test_same_key_overwrites(self):
        store = ConfigStore()
        store.put(self._record(median_ms=0.4, best_ms=0.38, p95_ms=0.45))
        store.put(self._record(median_ms=0.2, best_ms=0.18, p95_ms=0.24))
        self.assertEqual(len(store), 1)
        self.assertEqual(store.records()[0].median_ms, 0.2)

    def test_remove(self):
        store = ConfigStore()
        store.put(self._record())
        self.assertTrue(store.remove("cuda", "RTX 2080 Ti", "add", {"dtype": "f32", "inputs": [[4, 8]]}))
        self.assertFalse(
            store.remove("cuda", "RTX 2080 Ti", "add", {"dtype": "f32", "inputs": [[4, 8]]})
        )

    def test_whitespace_in_device_is_normalized(self):
        store = ConfigStore()
        store.put(self._record(device="RTX  2080 Ti\n"))
        self.assertIsNotNone(store.get("cuda", "RTX 2080 Ti", "add", self._record().signature))

    def test_json_roundtrip(self):
        store = ConfigStore()
        store.put(self._record())
        restored = ConfigStore.loads(store.dumps())
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored.records()[0].config, {"block": 512})
        self.assertEqual(restored.records()[0].median_ms, 0.25)
        self.assertEqual(restored.records()[0].best_ms, 0.22)
        self.assertEqual(restored.records()[0].p95_ms, 0.31)

    def test_bad_json_is_explicit(self):
        with self.assertRaises(ValueError):
            ConfigStore.loads("{not json")

    def test_unknown_version_is_rejected(self):
        payload = {"store_version": 99, "records": []}
        with self.assertRaises(ValueError):
            ConfigStore.from_dict(payload)
        payload = {"records": []}
        with self.assertRaises(ValueError):
            ConfigStore.from_dict(payload)

    def test_missing_record_field_is_explicit(self):
        payload = {"store_version": 2, "records": [{"backend": "cuda"}]}
        with self.assertRaises(ValueError):
            ConfigStore.from_dict(payload)

    def test_invalid_record_is_rejected(self):
        with self.assertRaises(ValueError):
            self._record(samples=0)
        with self.assertRaises(ValueError):
            self._record(config={})
        with self.assertRaises(ValueError):
            self._record(op="")

    def test_best_cannot_exceed_median(self):
        """把"最小值"填进 median_ms 正是 v1 的问题，v2 直接拒绝。"""
        with self.assertRaises(ValueError) as ctx:
            self._record(best_ms=0.4)
        self.assertIn("best_ms", str(ctx.exception))

    def test_p95_cannot_undershoot_median(self):
        with self.assertRaises(ValueError) as ctx:
            self._record(p95_ms=0.1)
        self.assertIn("p95_ms", str(ctx.exception))

    def test_optional_percentiles_can_be_missing(self):
        record = self._record(best_ms=None, p95_ms=None)
        payload = record.to_dict()
        # "没采到"必须写出来，而不是悄悄省掉让读者猜
        self.assertIn("best_ms", payload)
        self.assertIsNone(payload["best_ms"])
        self.assertIsNone(payload["p95_ms"])
        restored = TuningRecord.from_dict(payload)
        self.assertIsNone(restored.best_ms)
        self.assertIsNone(restored.p95_ms)

    def test_v1_store_is_rejected(self):
        payload = {
            "store_version": 1,
            "records": [
                {
                    "backend": "cuda",
                    "device": "RTX 2080 Ti",
                    "op": "add",
                    "signature": {"dtype": "f32"},
                    "config": {"block": 512},
                    "median_ms": 0.25,
                }
            ],
        }
        with self.assertRaises(ValueError) as ctx:
            ConfigStore.from_dict(payload)
        self.assertIn("重新采集", str(ctx.exception))

    def test_save_is_atomic_and_leaves_no_temp_file(self):
        store = ConfigStore()
        store.put(self._record())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tuning.json")
            store.save(path)
            self.assertTrue(os.path.exists(path))
            self.assertFalse(os.path.exists(f"{path}.tmp"))
            loaded = ConfigStore.load(path)
            self.assertEqual(len(loaded), 1)
            with open(path, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["store_version"], 2)


class TestTune(unittest.TestCase):
    def test_picks_the_fastest(self):
        space = TuningSpace.build({"block": (128, 256, 512)})
        table = {128: 1.0, 256: 0.5, 512: 0.75}
        result = tune(space, lambda cfg: table[cfg["block"]])
        self.assertEqual(result.best.config, {"block": 256})
        self.assertEqual(result.best.median_ms, 0.5)
        self.assertEqual(result.best.best_ms, 0.5)
        self.assertEqual(result.best.spread, 0.0)
        self.assertEqual(len(result.candidates), 3)
        self.assertEqual(result.rejected, ())
        self.assertAlmostEqual(result.speedup, 2.0)
        self.assertAlmostEqual(result.best_speedup, 2.0)

    def test_tie_keeps_first_candidate(self):
        space = TuningSpace.build({"block": (128, 256)})
        result = tune(space, lambda cfg: 1.0)
        self.assertEqual(result.best.config, {"block": 128})

    def test_rounds_report_median_best_and_p95_separately(self):
        """多轮只报一个数时，"最佳可达"会被当成典型延迟。"""
        space = TuningSpace.build({"block": (128,)})
        values = iter([3.0, 1.0, 2.0])
        result = tune(space, lambda cfg: next(values), rounds=3)
        self.assertEqual(result.best.median_ms, 2.0)  # 典型延迟
        self.assertEqual(result.best.best_ms, 1.0)  # 最佳可达
        self.assertEqual(result.best.p95_ms, 3.0)
        self.assertEqual(result.best.spread, 0.5)
        self.assertEqual(result.best.samples, 3)

    def test_median_not_best_sample_decides_ranking(self):
        """一轮抖动不该让整体更慢的配置凭单次最优胜出。"""
        space = TuningSpace.build({"block": (128, 256)})
        samples = {
            128: iter([1.0, 10.0, 10.0]),
            256: iter([5.0, 5.0, 5.0]),
        }
        result = tune(space, lambda cfg: next(samples[cfg["block"]]), rounds=3)
        self.assertEqual(result.best.config, {"block": 256})
        unstable = result.candidates[0]
        self.assertEqual(unstable.best_ms, 1.0)  # 最佳可达确实更低
        self.assertEqual(unstable.median_ms, 10.0)  # 但典型延迟更差

    def test_insignificant_gain_keeps_default(self):
        """差距落在噪声内时保留默认，不制造"更快"的假确定性。"""
        space = TuningSpace.build({"block": (128, 256)})
        result = tune(space, lambda cfg: 1.0 if cfg["block"] == 128 else 0.99)
        self.assertEqual(result.best.config, {"block": 128})
        self.assertEqual(result.tie_epsilon, 0.02)

    def test_significant_gain_replaces_default(self):
        space = TuningSpace.build({"block": (128, 256)})
        result = tune(space, lambda cfg: 1.0 if cfg["block"] == 128 else 0.9)
        self.assertEqual(result.best.config, {"block": 256})

    def test_tie_epsilon_is_validated(self):
        space = TuningSpace.build({"block": (128,)})
        with self.assertRaises(ValueError):
            tune(space, lambda cfg: 1.0, tie_epsilon=1.0)
        with self.assertRaises(ValueError):
            tune(space, lambda cfg: 1.0, tie_epsilon=-0.1)
        with self.assertRaises(TypeError):
            tune(space, lambda cfg: 1.0, tie_epsilon=True)

    def test_failed_candidate_is_rejected_not_ranked(self):
        space = TuningSpace.build({"block": (128, 256, 512)})

        def measure(config):
            if config["block"] == 512:
                raise RuntimeError("shared memory 不够")
            return 2.0 if config["block"] == 128 else 1.0

        result = tune(space, measure)
        self.assertEqual(result.best.config, {"block": 256})
        self.assertEqual(len(result.rejected), 1)
        self.assertIn("RuntimeError", result.rejected[0].reason)
        self.assertIn("shared memory", result.rejected[0].reason)
        # 被拒的候选不报延迟：没有测量值，不能编一个 0 或无穷大
        payload = result.rejected[0].to_dict()
        self.assertNotIn("median_ms", payload)
        self.assertIn("rejected", payload)

    def test_invalid_measurement_is_rejected(self):
        space = TuningSpace.build({"block": (128, 256)})

        def measure(config):
            return None if config["block"] == 128 else float("nan")

        with self.assertRaises(AllRejectedError) as ctx:
            tune(space, measure, op="add")
        message = str(ctx.exception)
        self.assertIn("add", message)
        self.assertEqual(len(ctx.exception.candidates), 2)

    def test_all_rejected_is_an_error_not_a_slow_winner(self):
        space = TuningSpace.build({"block": (128, 256)})

        def measure(config):
            raise OSError("驱动拒绝")

        with self.assertRaises(AllRejectedError):
            tune(space, measure)

    def test_result_is_json_serializable(self):
        space = TuningSpace.build({"block": (128, 256)})
        result = tune(space, lambda cfg: 1.0 if cfg["block"] == 128 else 0.5)
        payload = json.loads(json.dumps(result.to_dict()))
        self.assertEqual(payload["best"]["config"], {"block": 256})
        self.assertEqual(payload["best"]["median_ms"], 0.5)
        self.assertEqual(payload["best"]["best_ms"], 0.5)
        self.assertEqual(payload["best"]["p95_ms"], 0.5)
        self.assertEqual(payload["tie_epsilon"], 0.02)


class TestTunePaired(unittest.TestCase):
    """交替 A/B 的纯逻辑：顺序、漂移、噪声地板与显式降级。

    用合成测量而不是真机：这些用例要钉住的是"配对差能不能把整机漂移与
    顺序效应摘掉"，不是某张卡的绝对毫秒。
    """

    def test_each_round_visits_all_candidates_and_reverses(self):
        space = TuningSpace.build({"block": (128, 256, 512)})
        seen = []

        def measure(config):
            seen.append(config["block"])
            return 1.0

        tune_paired(space, measure, rounds=4)
        self.assertEqual(
            seen,
            [128, 256, 512, 512, 256, 128, 128, 256, 512, 512, 256, 128],
        )

    def test_common_drift_is_removed_by_pairing(self):
        """所有候选一起被整机状态拖慢 50%，配对差仍然为 0。"""
        space = TuningSpace.build({"block": (128, 256)})
        drift = [1.0, 1.5, 0.8, 1.2, 1.4, 0.9, 1.1, 1.3]
        turn = {"value": -1}
        position = {"value": 0}

        def measure(config):
            if position["value"] % 2 == 0:
                turn["value"] += 1
            position["value"] += 1
            return 2.0 * drift[turn["value"]]

        result = tune_paired(space, measure, rounds=8)
        candidate = result.candidates[1]
        self.assertEqual(candidate.verdict, "noise")
        self.assertAlmostEqual(candidate.diff_median_pct, 0.0)
        self.assertEqual(result.best.config, {"block": 128})
        self.assertFalse(result.decided)

    def test_order_effect_cancels_when_rounds_reverse(self):
        """先测者固定获得 10% 优势时，逐轮反转把它平均掉。"""
        space = TuningSpace.build({"block": (128, 256)})
        calls = {"n": 0}

        def measure(config):
            first = calls["n"] % 2 == 0
            calls["n"] += 1
            return 1.1 if first else 1.0

        result = tune_paired(space, measure, rounds=12)
        candidate = result.candidates[1]
        # 固定顺序会得到恒定的 -9.09%（"256 更快"）；反转后中位差回到 0 附近
        self.assertLess(abs(candidate.diff_median_pct), 1.0)
        self.assertEqual(candidate.verdict, "noise")

    def test_opposite_halves_are_not_a_conclusion(self):
        """整体区间显著、但前后两半方向相反：这是整机状态换了方向，不是配置差异。"""
        space = TuningSpace.build({"block": (128, 256)})
        calls = {"n": 0}

        def measure(config):
            turn = calls["n"] // 2
            calls["n"] += 1
            if turn < 4:
                return 1.0 if config["block"] == 128 else 0.8
            return 1.0 if config["block"] == 128 else 1.02

        result = tune_paired(space, measure, rounds=8)
        candidate = result.candidates[1]
        self.assertLess(candidate.ci_high_pct, 0.0)  # 整体区间看起来"显著更快"
        self.assertLess(candidate.first_half_pct, 0.0)
        self.assertGreater(candidate.second_half_pct, 0.0)
        self.assertEqual(candidate.verdict, "unstable")
        self.assertFalse(candidate.stable)
        self.assertEqual(result.best.config, {"block": 128})
        self.assertFalse(result.decided)

    def test_real_gain_is_decided(self):
        space = TuningSpace.build({"block": (128, 256)})
        calls = {"n": 0}

        def measure(config):
            jitter = 1.0 + 0.002 * (-1 if calls["n"] % 2 else 1)
            calls["n"] += 1
            return (1.0 if config["block"] == 128 else 0.8) * jitter

        result = tune_paired(space, measure, rounds=8, noise_floor=0.01)
        candidate = result.candidates[1]
        self.assertEqual(candidate.verdict, "faster")
        self.assertLess(candidate.ci_high_pct, -1.0)
        self.assertEqual(result.best.config, {"block": 256})
        self.assertTrue(result.decided)
        self.assertAlmostEqual(result.speedup, 1.25, places=2)

    def test_gain_below_noise_floor_keeps_default(self):
        """0.5% 的稳定领先不值得替换默认：它省不下时间，却多一条不可复现的记录。"""
        space = TuningSpace.build({"block": (128, 256)})

        def measure(config):
            return 1.0 if config["block"] == 128 else 0.995

        result = tune_paired(space, measure, rounds=8, noise_floor=0.01)
        self.assertEqual(result.candidates[1].verdict, "noise")
        self.assertEqual(result.best.config, {"block": 128})
        self.assertFalse(result.decided)

    def test_rejected_candidate_does_not_disturb_pairing(self):
        space = TuningSpace.build({"block": (128, 256, 512)})

        def measure(config):
            if config["block"] == 512:
                raise RuntimeError("shared memory 不够")
            return 1.0 if config["block"] == 128 else 0.7

        result = tune_paired(space, measure, rounds=4)
        self.assertEqual(len(result.rejected), 1)
        self.assertIn("shared memory", result.rejected[0].reason)
        self.assertEqual(result.candidates[2].samples, 0)
        self.assertEqual(result.best.config, {"block": 256})
        self.assertTrue(result.decided)

    def test_unusable_baseline_degrades_explicitly(self):
        """默认候选跑不起来时，不能假装配对差存在。"""
        space = TuningSpace.build({"block": (128, 256, 512)})

        def measure(config):
            if config["block"] == 128:
                raise OSError("默认配置不可用")
            return 2.0 if config["block"] == 256 else 1.0

        result = tune_paired(space, measure, rounds=4)
        self.assertEqual(result.best.config, {"block": 512})
        self.assertFalse(result.decided)
        for item in result.candidates[1:]:
            self.assertEqual(item.verdict, "no-baseline")
            self.assertIsNone(item.diff_median_pct)

    def test_all_rejected_is_an_error(self):
        space = TuningSpace.build({"block": (128, 256)})

        def measure(config):
            raise OSError("驱动拒绝")

        with self.assertRaises(AllRejectedError):
            tune_paired(space, measure, rounds=2)

    def test_invalid_arguments(self):
        space = TuningSpace.build({"block": (128,)})
        with self.assertRaises(ValueError):
            tune_paired(space, lambda cfg: 1.0, rounds=1)
        with self.assertRaises(ValueError):
            tune_paired(space, lambda cfg: 1.0, noise_floor=1.0)
        with self.assertRaises(TypeError):
            tune_paired(space, lambda cfg: 1.0, noise_floor=True)
        with self.assertRaises(ValueError):
            tune_paired(space, lambda cfg: 1.0, bootstrap=10)
        with self.assertRaises(ValueError):
            tune_paired(space, lambda cfg: 1.0, alpha=1.0)

    def test_fixed_seed_is_reproducible(self):
        space = TuningSpace.build({"block": (128, 256)})
        values = [1.0, 1.002, 0.998, 1.001, 0.999, 1.0, 1.002, 0.998]

        def run():
            calls = {"n": 0}

            def measure(config):
                value = values[calls["n"] % len(values)]
                calls["n"] += 1
                return value * (1.0 if config["block"] == 128 else 0.95)

            return tune_paired(space, measure, rounds=8, seed=7)

        left, right = run(), run()
        self.assertEqual(
            (left.candidates[1].ci_low_pct, left.candidates[1].ci_high_pct),
            (right.candidates[1].ci_low_pct, right.candidates[1].ci_high_pct),
        )

    def test_shuffle_visits_every_candidate_once_per_round(self):
        space = TuningSpace.build({"block": (128, 256, 512)})
        seen = []

        def measure(config):
            seen.append(config["block"])
            return 1.0

        result = tune_paired(space, measure, rounds=4, order="shuffle")
        self.assertEqual(len(seen), 12)
        for turn in range(4):
            chunk = seen[turn * 3:(turn + 1) * 3]
            self.assertEqual(sorted(chunk), [128, 256, 512])
        self.assertEqual(len(result.order), 4)
        for perm in result.order:
            self.assertEqual(sorted(perm), [0, 1, 2])

    def test_shuffle_is_reproducible_with_same_seed(self):
        space = TuningSpace.build({"block": (128, 256, 512, 1024)})

        def run(seed):
            return tune_paired(space, lambda cfg: 1.0, rounds=6, order="shuffle", seed=seed)

        left, right = run(3), run(3)
        self.assertEqual(left.order, right.order)
        self.assertNotEqual(left.order, run(4).order)
        self.assertEqual(left.order_mode, "shuffle")

    def test_invalid_order_is_rejected(self):
        space = TuningSpace.build({"block": (128,)})
        with self.assertRaises(ValueError):
            tune_paired(space, lambda cfg: 1.0, rounds=2, order="random")

    def test_long_round_degrades_without_touching_baseline(self):
        """一轮太长时，"更快"不成立；但基线自身仍是基线，不是"跨度超限"。"""
        space = TuningSpace.build({"block": (128, 256)})

        def measure(config):
            time.sleep(0.01)
            return 1.0 if config["block"] == 128 else 0.8

        result = tune_paired(space, measure, rounds=2, max_round_seconds=0.005)
        self.assertEqual(len(result.round_seconds), 2)
        self.assertTrue(all(seconds > 0.005 for seconds in result.round_seconds))
        self.assertEqual(result.max_round_seconds, 0.005)
        self.assertEqual(result.candidates[0].verdict, "baseline")
        candidate = result.candidates[1]
        # 区间照算，但方向不作数
        self.assertLess(candidate.ci_high_pct, 0.0)
        self.assertEqual(candidate.verdict, "long-round")
        self.assertFalse(candidate.decided)
        self.assertFalse(candidate.stable)
        self.assertEqual(result.best.config, {"block": 128})
        self.assertFalse(result.decided)

    def test_threshold_above_actual_round_keeps_verdict(self):
        space = TuningSpace.build({"block": (128, 256)})
        result = tune_paired(
            space,
            lambda cfg: 1.0 if cfg["block"] == 128 else 0.8,
            rounds=4,
            max_round_seconds=60.0,
        )
        self.assertEqual(result.candidates[1].verdict, "faster")
        self.assertTrue(result.decided)

    def test_no_threshold_does_not_change_behavior(self):
        space = TuningSpace.build({"block": (128, 256)})
        result = tune_paired(space, lambda cfg: 1.0, rounds=4)
        self.assertEqual(len(result.round_seconds), 4)
        self.assertIsNone(result.max_round_seconds)
        self.assertEqual(result.candidates[1].verdict, "noise")

    def test_invalid_max_round_seconds(self):
        space = TuningSpace.build({"block": (128,)})
        for value in (0, -1.0):
            with self.assertRaises(ValueError):
                tune_paired(space, lambda cfg: 1.0, rounds=2, max_round_seconds=value)
        for value in (True, "1.0"):
            with self.assertRaises(TypeError):
                tune_paired(space, lambda cfg: 1.0, rounds=2, max_round_seconds=value)

    def test_relative_threshold_uses_warmup_as_ruler(self):
        """相对门槛 = factor × 预热轮耗时；超限只为非默认候选降级。"""
        space = TuningSpace.build({"block": (128, 256)})
        calls = {"n": 0}

        def measure(config):
            calls["n"] += 1
            # 预热 2 次 + 前两轮 4 次都正常；第 3 轮起变慢，整轮越过 2× 预热
            time.sleep(0.15 if calls["n"] > 6 else 0.02)
            return 1.0 if config["block"] == 128 else 0.8

        result = tune_paired(
            space, measure, rounds=4, warmup_rounds=1, max_round_factor=2.0
        )
        self.assertGreater(result.warmup_seconds, 0.0)
        self.assertAlmostEqual(
            result.round_budget_seconds, 2.0 * result.warmup_seconds, delta=1e-4
        )
        self.assertGreater(max(result.round_seconds), result.round_budget_seconds)
        self.assertEqual(result.candidates[0].verdict, "baseline")
        self.assertEqual(result.candidates[1].verdict, "long-round")
        self.assertEqual(result.best.config, {"block": 128})
        self.assertFalse(result.decided)

    def test_relative_threshold_requires_warmup(self):
        """没有预热轮就没有标尺，必须显式拒绝而不是退回绝对秒数。"""
        space = TuningSpace.build({"block": (128,)})
        with self.assertRaises(ValueError):
            tune_paired(space, lambda cfg: 1.0, rounds=2, max_round_factor=2.0)

    def test_relative_and_absolute_thresholds_are_exclusive(self):
        space = TuningSpace.build({"block": (128,)})
        with self.assertRaises(ValueError):
            tune_paired(
                space,
                lambda cfg: 1.0,
                rounds=2,
                warmup_rounds=1,
                max_round_seconds=1.0,
                max_round_factor=2.0,
            )

    def test_invalid_max_round_factor(self):
        space = TuningSpace.build({"block": (128,)})
        for value in (0, -1.0):
            with self.assertRaises(ValueError):
                tune_paired(
                    space, lambda cfg: 1.0, rounds=2,
                    warmup_rounds=1, max_round_factor=value,
                )
        for value in (True, "2.0"):
            with self.assertRaises(TypeError):
                tune_paired(
                    space, lambda cfg: 1.0, rounds=2,
                    warmup_rounds=1, max_round_factor=value,
                )

    def test_relative_budget_is_recorded(self):
        space = TuningSpace.build({"block": (128, 256)})
        result = tune_paired(
            space, lambda cfg: 1.0, rounds=2, warmup_rounds=1, max_round_factor=100.0
        )
        self.assertEqual(result.max_round_factor, 100.0)
        self.assertIsNotNone(result.round_budget_seconds)
        data = json.loads(json.dumps(result.to_dict()))
        self.assertEqual(data["max_round_factor"], 100.0)
        self.assertEqual(data["round_budget_seconds"], result.round_budget_seconds)

    def test_warmup_round_visits_all_candidates_and_is_discarded(self):
        """预热轮按正序连测一遍，但不进入 order / round_seconds / 样本。"""
        space = TuningSpace.build({"block": (128, 256, 512)})
        seen = []

        def measure(config):
            seen.append(config["block"])
            return 1.0

        result = tune_paired(space, measure, rounds=2, warmup_rounds=1)
        self.assertEqual(seen[:3], [128, 256, 512])  # 预热轮：固定正序
        self.assertEqual(len(seen), 9)  # 3 次预热 + 2 轮正式
        self.assertEqual(len(result.order), 2)
        self.assertEqual(len(result.round_seconds), 2)
        self.assertEqual(result.warmup_rounds, 1)
        self.assertGreaterEqual(result.warmup_seconds, 0.0)

    def test_warmup_does_not_change_shuffle_order(self):
        """预热轮不消耗排列随机流：预热只多跑一轮，不改正式轮的顺序。"""
        space = TuningSpace.build({"block": (128, 256, 512, 1024)})

        def run(warmup):
            return tune_paired(
                space, lambda cfg: 1.0, rounds=4, order="shuffle", seed=5,
                warmup_rounds=warmup,
            )

        self.assertEqual(run(0).order, run(1).order)

    def test_warmup_rejects_candidate_before_formal_rounds(self):
        space = TuningSpace.build({"block": (128, 256)})
        calls = {"n": 0}

        def measure(config):
            calls["n"] += 1
            if config["block"] == 256:
                raise OSError("预热期失败")
            return 1.0

        result = tune_paired(space, measure, rounds=4, warmup_rounds=1)
        self.assertEqual(len(result.rejected), 1)
        self.assertEqual(result.candidates[1].samples, 0)
        # 预热轮测了 2 个候选（第二个抛异常），正式 4 轮只测剩下的 128
        self.assertEqual(calls["n"], 6)

    def test_invalid_warmup_rounds(self):
        space = TuningSpace.build({"block": (128,)})
        with self.assertRaises(ValueError):
            tune_paired(space, lambda cfg: 1.0, rounds=2, warmup_rounds=-1)
        for value in (True, "1"):
            with self.assertRaises(TypeError):
                tune_paired(space, lambda cfg: 1.0, rounds=2, warmup_rounds=value)

    def test_result_is_json_serializable(self):
        space = TuningSpace.build({"block": (128, 256)})
        result = tune_paired(space, lambda cfg: 1.0 if cfg["block"] == 128 else 0.8, rounds=4)
        payload = json.loads(json.dumps(result.to_dict()))
        self.assertEqual(payload["method"], "paired")
        self.assertEqual(payload["order_mode"], "alternate")
        self.assertEqual(payload["best"]["config"], {"block": 256})
        self.assertEqual(payload["order"], ["forward", "reverse", "forward", "reverse"])
        self.assertEqual(len(payload["candidates"]), 2)
        self.assertEqual(len(payload["round_seconds"]), 4)
        self.assertEqual(payload["warmup_rounds"], 0)
        self.assertEqual(payload["warmup_seconds"], 0.0)
        self.assertIsNone(payload["max_round_seconds"])
        self.assertTrue(payload["decided"])


class TestSameSignFilter(unittest.TestCase):
    """5.160：最后一轮方向翻转时，"更快"不能当结论。

    构造：256 在前六轮都快 10%，第七轮慢 2%。均值区间仍然整体落在负侧、
    中位差与前两半也都判"更快"——一个幅度很小的翻转轮次被平均掉了，这正是
    同号计数要补的漏：它不重看幅度，只问"方向每一轮都成立吗"。
    """

    @staticmethod
    def _measure():
        seen = {"n": 0}

        def measure(config):
            if config["block"] == 128:
                return 1.0
            index = seen["n"]
            seen["n"] += 1
            return 0.9 if index < 6 else 1.02

        return measure

    def test_five_of_five_keeps_the_verdict(self):
        space = TuningSpace.build({"block": (128, 256)})
        result = tune_paired(
            space,
            lambda cfg: 1.0 if cfg["block"] == 128 else 0.9,
            rounds=5,
            min_same_sign=5,
        )
        candidate = result.candidates[1]
        self.assertEqual(candidate.verdict, "faster")
        self.assertEqual(candidate.same_sign, 5)
        self.assertEqual(result.best.config, {"block": 256})
        self.assertEqual(result.min_same_sign, 5)

    def test_one_flip_demotes_a_faster_verdict(self):
        space = TuningSpace.build({"block": (128, 256)})
        loose = tune_paired(space, self._measure(), rounds=7)
        self.assertEqual(loose.candidates[1].verdict, "faster")
        self.assertEqual(loose.candidates[1].same_sign, 0)  # 未检查

        strict = tune_paired(space, self._measure(), rounds=7, min_same_sign=7)
        self.assertEqual(strict.candidates[1].verdict, "unstable")
        self.assertEqual(strict.candidates[1].same_sign, 6)
        # 证据不足时默认胜出
        self.assertEqual(strict.best.config, {"block": 128})
        self.assertFalse(strict.decided)

    def test_six_of_seven_passes_a_lower_threshold(self):
        space = TuningSpace.build({"block": (128, 256)})
        result = tune_paired(space, self._measure(), rounds=7, min_same_sign=6)
        self.assertEqual(result.candidates[1].verdict, "faster")
        self.assertEqual(result.candidates[1].same_sign, 6)

    def test_invalid_thresholds_are_rejected(self):
        space = TuningSpace.build({"block": (128, 256)})
        for value in (-1, 1, 8):
            with self.assertRaises(ValueError):
                tune_paired(space, lambda cfg: 1.0, rounds=7, min_same_sign=value)
        for value in (True, 2.5, "5"):
            with self.assertRaises(TypeError):
                tune_paired(space, lambda cfg: 1.0, rounds=7, min_same_sign=value)

    def test_threshold_survives_staged_and_replicated(self):
        space = TuningSpace.build({"block": (128, 256)})
        staged = tune_staged(space, self._measure(), rounds=7, min_same_sign=7)
        self.assertEqual(staged.min_same_sign, 7)

        replicated = tune_replicated(
            space,
            lambda cfg: 1.0 if cfg["block"] == 128 else 0.9,
            rounds=7,
            repeats=2,
            min_same_sign=7,
        )
        self.assertEqual(replicated.min_same_sign, 7)
        self.assertEqual(replicated.best.config, {"block": 256})


class TestTuneStaged(unittest.TestCase):
    """两阶段搜索：每段单独判定，最终只和默认做一次两候选对照。"""

    def test_two_axes_are_searched_in_order(self):
        space = TuningSpace.build({"a": (None, 1), "b": (None, 1)})

        def measure(config):
            value = 1.0
            if config["b"] == 1:
                value *= 0.8      # b 单独就快 25%
            if config["a"] == 1:
                value *= 1.25     # a 在两个 b 下都更慢
            return value

        result = tune_staged(space, measure, rounds=8, first=("b",))
        self.assertEqual(result.best.config, {"a": None, "b": 1})
        self.assertAlmostEqual(result.speedup, 1.25, places=3)
        self.assertTrue(result.decided)
        self.assertEqual(len(result.stages), 2)
        self.assertEqual(result.stages[0]["keys"], ["b"])
        self.assertEqual(result.stages[0]["winner"], {"a": None, "b": 1})
        self.assertEqual(result.stages[1]["keys"], ["a"])
        # 阶段 2 在 b=1 下扫 a，默认仍是 a=None
        self.assertEqual(result.stages[1]["winner"], {"a": None, "b": 1})

    def test_single_key_space_degrades_to_paired(self):
        space = TuningSpace.build({"block": (128, 256)})
        result = tune_staged(
            space, lambda cfg: 1.0 if cfg["block"] == 128 else 0.8, rounds=4,
        )
        self.assertEqual(len(result.stages), 1)
        self.assertEqual(result.best.config, {"block": 256})
        self.assertTrue(result.decided)

    def test_first_defaults_to_the_smallest_axis(self):
        space = TuningSpace.build({"a": (0, 1, 2), "b": (0, 1)})
        result = tune_staged(space, lambda cfg: 1.0, rounds=4)
        self.assertEqual(result.stages[0]["keys"], ["b"])
        self.assertEqual(len(result.stages), 2)

    def test_invalid_first_is_rejected(self):
        space = TuningSpace.build({"a": (None, 1)})
        with self.assertRaises(ValueError):
            tune_staged(space, lambda cfg: 1.0, rounds=2, first=("nope",))
        with self.assertRaises(ValueError):
            tune_staged(space, lambda cfg: 1.0, rounds=2, first=())
        with self.assertRaises(TypeError):
            tune_staged(space, lambda cfg: 1.0, rounds=2, first="a")

    def test_staged_result_is_json_serializable(self):
        space = TuningSpace.build({"a": (None, 1), "b": (None, 1)})
        result = tune_staged(space, lambda cfg: 1.0, rounds=4, first=("b",))
        payload = json.loads(json.dumps(result.to_dict()))
        self.assertEqual(len(payload["stages"]), 2)
        self.assertEqual(payload["stages"][0]["keys"], ["b"])
        self.assertEqual(payload["best"]["config"], {"a": None, "b": None})
        self.assertFalse(payload["decided"])


class TestTuneReplicated(unittest.TestCase):
    """跨采集复现：只有每次都同向的候选才替换默认。"""

    def test_same_direction_twice_is_accepted(self):
        space = TuningSpace.build({"block": (128, 256)})
        result = tune_replicated(
            space, lambda cfg: 1.0 if cfg["block"] == 128 else 0.8, rounds=4,
        )
        self.assertEqual(result.repeats, 2)
        self.assertEqual(len(result.replicates), 2)
        self.assertEqual(result.best.config, {"block": 256})
        self.assertEqual(result.candidates[1].verdict, "faster")
        self.assertTrue(result.decided)

    def test_lost_direction_is_unreplicated(self):
        space = TuningSpace.build({"block": (128, 256)})
        calls = {"n": 0}

        def measure(config):
            calls["n"] += 1
            if config["block"] == 128:
                return 1.0
            # 第一次采集快 20%，第二次与默认持平
            return 0.8 if calls["n"] <= 8 else 1.0

        result = tune_replicated(space, measure, rounds=4)
        self.assertEqual(result.candidates[1].verdict, "unreplicated")
        self.assertFalse(result.candidates[1].stable)
        self.assertEqual(result.best.config, {"block": 128})
        self.assertFalse(result.decided)
        self.assertTrue(result.replicates[0]["decided"])
        self.assertFalse(result.replicates[1]["decided"])

    def test_infeasible_in_one_run_is_unreplicated(self):
        space = TuningSpace.build({"block": (128, 256)})
        calls = {"n": 0}

        def measure(config):
            calls["n"] += 1
            if config["block"] == 256 and calls["n"] > 8:
                raise OSError("第二次采集失败")
            return 1.0 if config["block"] == 128 else 0.8

        result = tune_replicated(space, measure, rounds=4)
        self.assertEqual(result.candidates[1].verdict, "unreplicated")
        self.assertFalse(result.candidates[1].feasible)
        self.assertEqual(result.best.config, {"block": 128})

    def test_invalid_repeats(self):
        space = TuningSpace.build({"block": (128,)})
        for value in (1, 0, -1):
            with self.assertRaises(ValueError):
                tune_replicated(space, lambda cfg: 1.0, rounds=2, repeats=value)
        for value in (True, "2"):
            with self.assertRaises(TypeError):
                tune_replicated(space, lambda cfg: 1.0, rounds=2, repeats=value)

    @staticmethod
    def _by_replica(rounds, candidates, plan):
        """按"第几次采集"给出耗时：``plan[i](config)`` 是第 i 次采集的测量。"""
        calls = {"n": 0}

        def measure(config):
            index = calls["n"] // (rounds * candidates)
            calls["n"] += 1
            return plan[index](config)

        return measure

    def test_vote_all_stays_strict(self):
        """全票口径不因新增多数票而放宽：2 次更快 + 1 次噪声仍算未复现。"""
        space = TuningSpace.build({"block": (128, 256)})
        fast = lambda cfg: 1.0 if cfg["block"] == 128 else 0.8  # noqa: E731
        flat = lambda cfg: 1.0  # noqa: E731
        result = tune_replicated(
            space, self._by_replica(4, 2, [fast, fast, flat]),
            rounds=4, repeats=3,
        )
        self.assertEqual(result.candidates[1].verdict, "unreplicated")
        self.assertEqual(result.vote, "all")
        self.assertFalse(result.decided)

    def test_majority_accepts_two_of_three(self):
        space = TuningSpace.build({"block": (128, 256)})
        fast = lambda cfg: 1.0 if cfg["block"] == 128 else 0.8  # noqa: E731
        flat = lambda cfg: 1.0  # noqa: E731
        result = tune_replicated(
            space, self._by_replica(4, 2, [fast, fast, flat]),
            rounds=4, repeats=3, vote="majority",
        )
        candidate = result.candidates[1]
        self.assertEqual(candidate.verdict, "faster")
        self.assertEqual(dict(candidate.votes), {"faster": 2, "noise": 1})
        self.assertEqual(result.best.config, {"block": 256})
        self.assertTrue(result.decided)
        self.assertEqual(result.vote, "majority")

    def test_majority_rejects_one_of_three(self):
        """1 次更快 + 2 次噪声：没有过半，按多数判定（噪声）保留默认。"""
        space = TuningSpace.build({"block": (128, 256)})
        fast = lambda cfg: 1.0 if cfg["block"] == 128 else 0.8  # noqa: E731
        flat = lambda cfg: 1.0  # noqa: E731
        result = tune_replicated(
            space, self._by_replica(4, 2, [fast, flat, flat]),
            rounds=4, repeats=3, vote="majority",
        )
        candidate = result.candidates[1]
        self.assertEqual(candidate.verdict, "noise")
        self.assertEqual(dict(candidate.votes), {"noise": 2, "faster": 1})
        self.assertEqual(result.best.config, {"block": 128})
        self.assertFalse(result.decided)

    def test_majority_conflicting_direction_is_unreplicated(self):
        """更快与更慢互现：方向都定不下来，多数的意义不大。"""
        space = TuningSpace.build({"block": (128, 256)})
        fast = lambda cfg: 1.0 if cfg["block"] == 128 else 0.8  # noqa: E731
        slow = lambda cfg: 1.0 if cfg["block"] == 128 else 1.25  # noqa: E731
        result = tune_replicated(
            space, self._by_replica(4, 2, [fast, slow, fast]),
            rounds=4, repeats=3, vote="majority",
        )
        candidate = result.candidates[1]
        self.assertEqual(candidate.verdict, "unreplicated")
        self.assertEqual(dict(candidate.votes), {"faster": 2, "slower": 1})
        self.assertFalse(result.decided)

    def test_majority_infeasible_once_is_unreplicated(self):
        """任一次不可行都不是投票能抵消的。"""
        space = TuningSpace.build({"block": (128, 256)})
        calls = {"n": 0}

        def measure(config):
            calls["n"] += 1
            if config["block"] == 256 and calls["n"] > 16:
                raise OSError("第三次采集失败")
            return 1.0 if config["block"] == 128 else 0.8

        result = tune_replicated(
            space, measure, rounds=4, repeats=3, vote="majority",
        )
        self.assertEqual(result.candidates[1].verdict, "unreplicated")
        self.assertEqual(result.best.config, {"block": 128})

    def test_invalid_vote(self):
        space = TuningSpace.build({"block": (128,)})
        with self.assertRaises(ValueError):
            tune_replicated(space, lambda cfg: 1.0, rounds=2, vote="best-of")
        with self.assertRaises(ValueError):
            tune_replicated(
                space, lambda cfg: 1.0, rounds=2, repeats=2, vote="majority",
            )

    def test_votes_are_serialized(self):
        space = TuningSpace.build({"block": (128, 256)})
        fast = lambda cfg: 1.0 if cfg["block"] == 128 else 0.8  # noqa: E731
        flat = lambda cfg: 1.0  # noqa: E731
        result = tune_replicated(
            space, self._by_replica(4, 2, [fast, fast, flat]),
            rounds=4, repeats=3, vote="majority",
        )
        payload = json.loads(json.dumps(result.to_dict()))
        self.assertEqual(payload["vote"], "majority")
        self.assertEqual(payload["candidates"][1]["votes"], {"faster": 2, "noise": 1})
        self.assertEqual(payload["candidates"][0]["votes"], {"baseline": 3})

    def test_replicate_diffs_follow_replicates(self):
        """逐次配对差：长度等于采集次数，顺序按采集排，与票数一一对应。"""
        space = TuningSpace.build({"block": (128, 256)})
        fast = lambda cfg: 1.0 if cfg["block"] == 128 else 0.8  # noqa: E731
        flat = lambda cfg: 1.0  # noqa: E731
        result = tune_replicated(
            space, self._by_replica(4, 2, [fast, fast, flat]),
            rounds=4, repeats=3, vote="majority",
        )
        candidate = result.candidates[1]
        # 更快那次每轮配对差 (0.8/1.0-1)*100 = -20%，持平那次是 0
        self.assertEqual(candidate.replicate_diffs, (-20.0, -20.0, 0.0))
        self.assertEqual(len(candidate.replicate_diffs), len(result.replicates))
        self.assertEqual(
            len(candidate.replicate_diffs), sum(dict(candidate.votes).values())
        )
        # 基线是配对差的参照物，每次都是 0
        self.assertEqual(result.candidates[0].replicate_diffs, (0.0, 0.0, 0.0))

    def test_replicate_diffs_mark_infeasible_run_none(self):
        """某次采集不可行时那一位是 None，而不是丢掉、也不是补 0。"""
        space = TuningSpace.build({"block": (128, 256)})
        calls = {"n": 0}

        def measure(config):
            calls["n"] += 1
            if config["block"] == 256 and calls["n"] > 16:
                raise OSError("第三次采集失败")
            return 1.0 if config["block"] == 128 else 0.8

        result = tune_replicated(
            space, measure, rounds=4, repeats=3, vote="majority",
        )
        self.assertEqual(
            result.candidates[1].replicate_diffs, (-20.0, -20.0, None)
        )
        self.assertEqual(len(result.candidates[1].replicate_diffs), 3)

    def test_replicate_diffs_are_serialized(self):
        space = TuningSpace.build({"block": (128, 256)})
        fast = lambda cfg: 1.0 if cfg["block"] == 128 else 0.8  # noqa: E731
        flat = lambda cfg: 1.0  # noqa: E731
        result = tune_replicated(
            space, self._by_replica(4, 2, [fast, fast, flat]),
            rounds=4, repeats=3, vote="majority",
        )
        payload = json.loads(json.dumps(result.to_dict()))
        self.assertEqual(
            payload["candidates"][1]["replicate_diffs"], [-20.0, -20.0, 0.0]
        )
        self.assertEqual(
            payload["candidates"][0]["replicate_diffs"], [0.0, 0.0, 0.0]
        )

    def test_single_collection_has_no_replicate_diffs(self):
        """单次采集没有"逐次"可言：字段保持空、且不落盘。"""
        space = TuningSpace.build({"block": (128, 256)})
        result = tune_paired(
            space, lambda cfg: 1.0 if cfg["block"] == 128 else 0.8, rounds=4,
        )
        self.assertEqual(result.candidates[0].replicate_diffs, ())
        self.assertNotIn("replicate_diffs", result.candidates[1].to_dict())

    def test_result_is_json_serializable(self):
        space = TuningSpace.build({"block": (128, 256)})
        result = tune_replicated(space, lambda cfg: 1.0, rounds=2)
        payload = json.loads(json.dumps(result.to_dict()))
        self.assertEqual(payload["repeats"], 2)
        self.assertEqual(len(payload["replicates"]), 2)
        self.assertEqual(payload["replicates"][0]["index"], 0)
        self.assertFalse(payload["decided"])


if __name__ == "__main__":
    unittest.main()
