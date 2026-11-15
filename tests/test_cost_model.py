"""成本模型与"按估算排序后只测前 N 个"的本地测试（5.242）。

这一组全部是纯 Python：成本模型只吃 ``lower/plan.py`` 的派生量，不碰设备，
因此本地无 GPU 就能把"为什么这么排"钉死。真机只回答"这个配置快不快"，
不回答"排在前面的组合是不是真的更省访存"。

判据分三类：

1. **形状与计划一致**：估算里的 ``threads`` / ``shared_bytes`` / ``oc_tiles``
   必须等于 ``plan_conv_tiled`` 的输出——两处各推一遍必然会漂。
2. **方向性单调**：``cb`` 变大 staging 下降、``roc`` 变大输入读数下降、
   ``tile_w`` 超过 ``wo`` 被截断（两个配置的估算必须逐项相等）。
3. **搜索语义不变**：裁剪只改变"测了哪些"，不改变排名与平局规则；
   默认配置永远排第一；全不可行时抛 ``AllRejectedError``。
"""

import math
import unittest

from aetherflow.lower.plan import plan_conv_tiled
from aetherflow.tuning import (
    M40_LIKE,
    AllRejectedError,
    CostBreakdown,
    MachineModel,
    TuningSpace,
    conv_tile_cost,
    cost_ranked_tune,
    elementwise_cost,
    measure_configs,
    rank_configs,
    tune,
)

CONV_SHAPE = dict(kh=3, kw=3, c=64, o=64, ho=32, wo=32, n=1)


def conv_space(**overrides):
    """与后端 ``tuning_space("conv2d")`` 同构的三旋钮空间（默认在前）。"""
    params = {
        "tile_w": (None, 16, 32, 64),
        "roc": (None, 1, 2, 4),
        "cb": (None, 1, 2, 4, 8),
    }
    params.update(overrides)
    return TuningSpace.build(params)


class TestMachineModel(unittest.TestCase):
    def test_defaults_are_positive(self):
        self.assertGreater(M40_LIKE.sm_count, 0)
        self.assertGreater(M40_LIKE.us(1000), 0)

    def test_negative_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            MachineModel(sm_count=0)
        with self.assertRaises(ValueError):
            MachineModel(bytes_per_cycle=0.0)
        with self.assertRaises(ValueError):
            MachineModel(sync_cycles=-1)

    def test_occupancy_is_bounded_by_threads_short(self):
        # 256 线程 → 每 SM 最多 8 个；shared 9216 B 在 48KB 下最多 5 个。
        self.assertEqual(M40_LIKE.occupancy(threads=256, shared_bytes=9216), 5)
        self.assertEqual(M40_LIKE.occupancy(threads=256, shared_bytes=0), 8)

    def test_occupancy_rejects_zero_threads(self):
        with self.assertRaises(ValueError):
            M40_LIKE.occupancy(threads=0, shared_bytes=0)


class TestConvTileCost(unittest.TestCase):
    def setUp(self):
        self.cost = conv_tile_cost(**CONV_SHAPE)

    def test_terms_are_fixed_six(self):
        item = self.cost({"tile_w": None, "roc": None, "cb": None})
        self.assertTrue(item.feasible)
        self.assertEqual(
            [name for name, _ in item.terms],
            ["arith", "read", "staging", "sync", "tail", "dead"],
        )
        self.assertGreater(item.total_us, 0)
        self.assertIn(item.dominant, {name for name, _ in item.terms})

    def test_detail_matches_plan_layer(self):
        item = self.cost({"tile_w": None, "roc": None, "cb": None})
        plan = plan_conv_tiled(
            kh=3, kw=3, c=64, o=64, ho=32, wo=32
        )
        detail = dict(item.detail)
        self.assertEqual(detail["threads"], plan.tile_w * plan.block_y)
        self.assertEqual(detail["shared_bytes"], plan.shared_bytes)
        self.assertEqual(detail["oc_tiles"], plan.oc_tiles)
        self.assertEqual(detail["blocks"], 1 * CONV_SHAPE["ho"] * 1 * plan.oc_tiles)

    def test_larger_cb_lowers_staging_and_sync(self):
        small = self.cost({"tile_w": None, "roc": None, "cb": 1})
        large = self.cost({"tile_w": None, "roc": None, "cb": 8})
        self.assertLess(large.term("staging"), small.term("staging"))
        self.assertLess(large.term("sync"), small.term("sync"))
        # 算术与输入读数与 cb 无关：cb 只改"搬多少权重、同步几次"。
        self.assertAlmostEqual(large.term("arith"), small.term("arith"), places=9)
        self.assertAlmostEqual(large.term("read"), small.term("read"), places=9)

    def test_larger_roc_lowers_input_reads(self):
        one = self.cost({"tile_w": None, "roc": 1, "cb": None})
        four = self.cost({"tile_w": None, "roc": 4, "cb": None})
        self.assertAlmostEqual(four.term("read") * 4, one.term("read"), places=6)
        self.assertAlmostEqual(four.term("arith"), one.term("arith"), places=9)

    def test_tile_w_is_clamped_to_wo(self):
        wide = self.cost({"tile_w": 64, "roc": None, "cb": None})
        exact = self.cost({"tile_w": 32, "roc": None, "cb": None})
        self.assertEqual(wide.terms, exact.terms)
        self.assertEqual(dict(wide.detail), dict(exact.detail))

    def test_dead_threads_shrink_with_narrow_tile(self):
        # wo=20：tile_w=8 时三个瓦片共 24 个槽，tile_w=16 时两个瓦片共 32 个槽。
        cost = conv_tile_cost(kh=3, kw=3, c=64, o=64, ho=8, wo=20, n=1)
        narrow = cost({"tile_w": 8, "roc": None, "cb": None})
        wide = cost({"tile_w": 16, "roc": None, "cb": None})
        self.assertGreater(wide.term("dead"), 0.0)
        self.assertLess(narrow.term("dead"), wide.term("dead"))
        # 瓦片更窄 → 块数更多（3 个 x 瓦片对 2 个）。
        self.assertGreater(
            dict(narrow.detail)["blocks"], dict(wide.detail)["blocks"]
        )

    def test_roc_must_divide_oc_tile(self):
        item = self.cost({"tile_w": None, "roc": 3, "cb": None})
        self.assertFalse(item.feasible)
        self.assertIn("整除", item.reason)
        self.assertEqual(item.total_us, math.inf)
        self.assertEqual(item.dominant, "")

    def test_oversized_block_is_infeasible(self):
        # tile_w 会被截到 wo，所以要用宽输出才能撞到 block 上限：
        # tile_w=1024 × block_y=8 = 8192 > 1024。
        cost = conv_tile_cost(kh=3, kw=3, c=64, o=64, ho=8, wo=2048, n=1)
        item = cost({"tile_w": 1024, "roc": None, "cb": None})
        self.assertFalse(item.feasible)
        self.assertIn("上限", item.reason)

    def test_cb_over_budget_is_infeasible(self):
        over_budget = self.cost({"tile_w": None, "roc": None, "cb": 64})
        self.assertFalse(over_budget.feasible)
        self.assertIn("预算", over_budget.reason)

    def test_unknown_knob_is_infeasible(self):
        item = self.cost({"tile_w": None, "roc": None, "cb": None, "block": 128})
        self.assertFalse(item.feasible)
        self.assertIn("不认识", item.reason)

    def test_bad_shape_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            conv_tile_cost(kh=0, kw=3, c=64, o=64, ho=8, wo=8)
        with self.assertRaises(TypeError):
            conv_tile_cost(**CONV_SHAPE, machine="m40")

    def test_explain_mentions_dominant(self):
        item = self.cost({"tile_w": None, "roc": None, "cb": None})
        text = item.explain()
        self.assertIn("主导", text)
        self.assertIn(item.dominant, text)

    def test_infeasible_explain_keeps_reason(self):
        item = self.cost({"tile_w": None, "roc": 3, "cb": None})
        self.assertIn("不可行", item.explain())
        self.assertEqual(item.to_dict()["rejected"], item.reason)

    def test_missing_knobs_mean_auto(self):
        auto = self.cost({})
        explicit = self.cost({"tile_w": None, "roc": None, "cb": None})
        self.assertEqual(auto.terms, explicit.terms)

    def test_breakdown_is_frozen(self):
        item = self.cost({"tile_w": None, "roc": None, "cb": None})
        self.assertIsInstance(item, CostBreakdown)
        with self.assertRaises(Exception):
            item.total_us = 0.0  # type: ignore[misc]  # frozen dataclass


class TestElementwiseCost(unittest.TestCase):
    def test_auto_block_matches_plan_layer(self):
        cost = elementwise_cost(4096)
        item = cost({"block": None})
        self.assertTrue(item.feasible)
        self.assertEqual(
            [name for name, _ in item.terms], ["read", "write", "tail", "dead"]
        )
        self.assertGreater(dict(item.detail)["threads"], 0)

    def test_vectorized_runs_fewer_blocks(self):
        scalar = elementwise_cost(4096, vec=1)({"block": 128})
        vector = elementwise_cost(4096, vec=4)({"block": 128})
        self.assertEqual(
            dict(scalar.detail)["blocks"], 4 * dict(vector.detail)["blocks"]
        )
        # 搬的字节数一样：向量化只减少块数与空线程，不减少流量。
        self.assertAlmostEqual(scalar.term("read"), vector.term("read"), places=9)
        self.assertAlmostEqual(scalar.term("write"), vector.term("write"), places=9)

    def test_dead_threads_only_on_ragged_tail(self):
        exact = elementwise_cost(8192, machine=M40_LIKE)({"block": 256})
        ragged = elementwise_cost(8000, machine=M40_LIKE)({"block": 256})
        self.assertEqual(exact.term("dead"), 0.0)
        self.assertGreater(ragged.term("dead"), 0.0)

    def test_non_power_of_two_block_is_infeasible(self):
        item = elementwise_cost(1024)({"block": 100})
        self.assertFalse(item.feasible)
        self.assertIn("2 的幂", item.reason)

    def test_unknown_knob_and_bad_vec_rejected(self):
        self.assertFalse(elementwise_cost(1024)({"tile_w": 32}).feasible)
        with self.assertRaises(ValueError):
            elementwise_cost(1024, vec=3)
        with self.assertRaises(ValueError):
            elementwise_cost(-1)


class TestRankConfigs(unittest.TestCase):
    def test_ranking_covers_every_config(self):
        space = conv_space()
        cost = conv_tile_cost(**CONV_SHAPE)
        ranking = rank_configs(space, cost)
        self.assertEqual(len(ranking), space.size())
        self.assertEqual({str(i.config) for i in ranking}, {str(c) for c in space.configs()})

    def test_feasible_before_infeasible_and_stable(self):
        space = conv_space(roc=(3, 1, 2))
        cost = conv_tile_cost(**CONV_SHAPE)
        first = rank_configs(space, cost)
        second = rank_configs(space, cost)
        self.assertEqual([i.config for i in first], [i.config for i in second])
        flags = [item.feasible for item in first]
        # 不可行项全部殿后。
        self.assertEqual(flags, sorted(flags, reverse=True))
        self.assertTrue(any(not flag for flag in flags))
        totals = [item.total_us for item in first if item.feasible]
        self.assertEqual(totals, sorted(totals))

    def test_all_infeasible_keeps_space_order(self):
        space = conv_space(tile_w=(2048, 4096), roc=(3,), cb=(None,))
        ranking = rank_configs(space, conv_tile_cost(**CONV_SHAPE))
        self.assertTrue(all(not item.feasible for item in ranking))
        self.assertEqual(
            [item.config for item in ranking],
            [
                {"tile_w": 2048, "roc": 3, "cb": None},
                {"tile_w": 4096, "roc": 3, "cb": None},
            ],
        )

    def test_type_errors(self):
        with self.assertRaises(TypeError):
            rank_configs({"a": 1}, lambda cfg: None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            rank_configs(conv_space(), "cost")  # type: ignore[arg-type]


class TestCostRankedTune(unittest.TestCase):
    def setUp(self):
        self.space = conv_space()
        self.cost = conv_tile_cost(**CONV_SHAPE)

    def test_budget_limits_measurements_and_keeps_default_first(self):
        seen = []

        def measure(config):
            seen.append(dict(config))
            return 10.0

        result = cost_ranked_tune(
            self.space, self.cost, measure, op="conv2d", max_evals=3
        )
        self.assertEqual(len(seen), 3)
        self.assertEqual(seen[0], self.space.default())
        self.assertEqual(result.evaluated, 3)
        self.assertEqual(
            result.skipped,
            len([i for i in result.ranking if i.feasible]) - 3,
        )
        self.assertEqual(len(result.result.candidates), 3)

    def test_order_of_measurements_follows_cost(self):
        seen = []

        def measure(config):
            seen.append(dict(config))
            return 10.0

        result = cost_ranked_tune(
            self.space, self.cost, measure, max_evals=5
        )
        feasible = [
            dict(item.config) for item in result.ranking if item.feasible
        ]
        defaults = dict(self.space.default())
        rest = [config for config in feasible if config != defaults]
        self.assertEqual(seen, [defaults] + rest[:4])

    def test_without_budget_agrees_with_tune_on_ties(self):
        def measure(config):
            return 10.0

        ranked = cost_ranked_tune(self.space, self.cost, measure, op="conv2d")
        plain = tune(self.space, measure, op="conv2d")
        self.assertEqual(dict(ranked.best.config), dict(plain.best.config))
        self.assertEqual(ranked.skipped, 0)
        self.assertEqual(ranked.evaluated, self.space.size())

    def test_measured_win_overrides_cost_estimate(self):
        # 估算把某个非默认配置排在后面，但只要预算覆盖到它、实测更快，它就赢。
        target = {"tile_w": 16, "roc": 4, "cb": 8}

        def measure(config):
            return 1.0 if dict(config) == target else 10.0

        result = cost_ranked_tune(self.space, self.cost, measure, tie_epsilon=0.02)
        self.assertEqual(dict(result.best.config), target)
        self.assertGreater(result.result.speedup, 1.0)

    def test_measurement_errors_become_rejections(self):
        def measure(config):
            if dict(config).get("tile_w") == 16:
                raise RuntimeError("设备忙")
            return 5.0

        result = cost_ranked_tune(
            self.space, self.cost, measure
        )
        rejected = [i for i in result.result.candidates if not i.feasible]
        self.assertTrue(any("设备忙" in i.reason for i in rejected))

    def test_all_infeasible_raises(self):
        space = conv_space(tile_w=(2048,), roc=(3,), cb=(None,))
        with self.assertRaises(AllRejectedError) as ctx:
            cost_ranked_tune(space, conv_tile_cost(**CONV_SHAPE), lambda c: 1.0)
        self.assertIn("不可行", str(ctx.exception))

    def test_bad_budget_is_rejected(self):
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                cost_ranked_tune(
                    self.space, self.cost, lambda c: 1.0, max_evals=value
                )

    def test_to_dict_records_estimates(self):
        result = cost_ranked_tune(
            self.space, self.cost, lambda c: 10.0, max_evals=2
        )
        payload = result.to_dict()
        self.assertEqual(payload["evaluated"], 2)
        self.assertEqual(len(payload["ranking"]), self.space.size())
        self.assertIn("result", payload)
        self.assertIn("total_us", payload["ranking"][0])

    def test_elementwise_backend_space_works(self):
        space = TuningSpace.build({"block": (None, 128, 256, 512)})
        cost = elementwise_cost(1 << 20)
        result = cost_ranked_tune(space, cost, lambda c: 1.0, max_evals=2)
        self.assertEqual(result.evaluated, 2)
        self.assertEqual(dict(result.best.config), space.default())


class TestMeasureConfigs(unittest.TestCase):
    def test_empty_configs_rejected(self):
        with self.assertRaises(ValueError):
            measure_configs((), lambda c: 1.0)

    def test_tie_keeps_first(self):
        configs = ({"block": 128}, {"block": 256})
        result = measure_configs(configs, lambda c: 10.0, tie_epsilon=0.02)
        self.assertEqual(dict(result.best.config), {"block": 128})

    def test_significant_win_replaces_first(self):
        configs = ({"block": 128}, {"block": 256})
        result = measure_configs(
            configs, lambda c: 5.0 if c["block"] == 256 else 10.0
        )
        self.assertEqual(dict(result.best.config), {"block": 256})
        self.assertAlmostEqual(result.speedup, 2.0)

    def test_invalid_measurements_are_rejected(self):
        configs = ({"block": 128}, {"block": 256})
        with self.assertRaises(AllRejectedError) as ctx:
            measure_configs(configs, lambda c: None)
        self.assertEqual(len(ctx.exception.candidates), 2)
        with self.assertRaises(AllRejectedError):
            measure_configs(configs, lambda c: float("nan"))


if __name__ == "__main__":
    unittest.main()
