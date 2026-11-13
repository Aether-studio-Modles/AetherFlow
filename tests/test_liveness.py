"""编译期静态显存预算的测试。

预算的定义是"从 IR 直接算"，所以这里不做任何执行：只用手算的几张图
把峰值、总字节、对齐与符号维绑定钉住。

与真机 ``Arena`` 的对照在 ``tests/test_cuda_backend.py`` 的
``TestCudaStaticBudget`` 里——那需要 GPU，本地只验证算法本身。
"""

import unittest

from aetherflow import Graph, backward, f32, plan_memory
from aetherflow.ir.liveness import ALIGNMENT


class TestStaticBudget(unittest.TestCase):
    def test_linear_chain(self):
        """x(4,4) -> relu -> silu：每步 64 B，峰值是 2 个活跃值。"""
        g = Graph()
        x = g.input("x", f32[4, 4])
        v1 = g.relu(x)
        g.output(g.silu(v1), "y")

        budget = plan_memory(g)
        self.assertEqual(budget.total_bytes, 3 * 64)
        self.assertEqual(budget.peak_live_bytes, 2 * 64)
        self.assertEqual(budget.intermediate_peak_bytes, 2 * 64)

    def test_branch_peak_includes_both_sides(self):
        g = Graph()
        x = g.input("x", f32[4, 4])
        a = g.relu(x)
        b = g.silu(x)
        g.output(g.add(a, b), "y")

        budget = plan_memory(g)
        self.assertEqual(budget.total_bytes, 4 * 64)
        # 两个分支的结果与合并结果同时活跃
        self.assertEqual(budget.peak_live_bytes, 3 * 64)
        self.assertEqual(budget.intermediate_peak_bytes, 3 * 64)

    def test_reuse_ratio(self):
        """一条长链：总量的绝大部分都可以被复用掉。"""
        g = Graph()
        x = g.input("x", f32[16, 16])
        v = x
        for _ in range(8):
            v = g.relu(v)
        g.output(v, "y")

        budget = plan_memory(g)
        self.assertEqual(budget.total_bytes, 9 * 1024)
        self.assertEqual(budget.peak_live_bytes, 2 * 1024)
        self.assertEqual(budget.reuse_ratio, 4.5)

    def test_footprint_lifetimes(self):
        g = Graph()
        x = g.input("x", f32[2])
        a = g.relu(x)
        g.output(g.silu(a), "y")

        budget = plan_memory(g)
        by_name = {f.name: f for f in budget.values}
        self.assertEqual(by_name["x"].defined_at, -1)
        self.assertEqual(by_name["x"].last_used_at, 0)
        self.assertEqual(by_name["v1"].defined_at, 0)
        self.assertEqual(by_name["v1"].last_used_at, 1)
        self.assertEqual(by_name["v2"].last_used_at, 2)  # 图输出活到最后

    def test_alignment_is_a_ceiling(self):
        g = Graph()
        x = g.input("x", f32[3])  # 12 B
        g.output(g.relu(x), "y")

        budget = plan_memory(g)
        self.assertGreater(budget.aligned_peak_bytes, budget.peak_live_bytes)
        self.assertEqual(budget.aligned_peak_bytes, 2 * ALIGNMENT)

    def test_symbolic_shape_requires_binding(self):
        g = Graph()
        x = g.input("x", f32["B", 8])
        g.output(g.relu(x), "y")

        with self.assertRaises(ValueError):
            plan_memory(g)
        budget = plan_memory(g, symbols={"B": 4})
        self.assertEqual(budget.peak_live_bytes, 2 * 4 * 8 * 4)

    def test_zero_element_tensor(self):
        g = Graph()
        x = g.input("x", f32[0, 4])
        g.output(g.relu(x), "y")
        budget = plan_memory(g)
        self.assertEqual(budget.peak_live_bytes, 0)

    def test_as_dict_keys(self):
        g = Graph()
        x = g.input("x", f32[2, 2])
        g.output(g.relu(x), "y")
        d = plan_memory(g).as_dict()
        self.assertEqual(d["peak_live_bytes"], 32)
        self.assertEqual(d["symbols"], {})
        self.assertIs(d["release"], True)

    def test_release_false_is_the_plain_execution(self):
        """普通执行不回收中间值：峰值就该等于总量，复用倍数为 1。

        ``Executor.run_device`` 的 ``release`` 默认是 ``False``，所以没有
        显式回收的路径实际占用是 ``total_bytes``，不是 liveness 峰值。这条
        把"预算口径必须跟着执行策略走"钉住。
        """
        g = Graph()
        x = g.input("x", f32[16, 16])
        v = x
        for _ in range(4):
            v = g.relu(v)
        g.output(v, "y")

        tight = plan_memory(g)
        plain = plan_memory(g, release=False)

        self.assertIs(tight.release, True)
        self.assertIs(plain.release, False)
        self.assertEqual(plain.intermediate_peak_bytes, plain.intermediate_total_bytes)
        self.assertEqual(plain.peak_live_bytes, plain.total_bytes)
        self.assertEqual(plain.reuse_ratio, 1.0)
        # 同一个"活到最后"的口径：释放策略省下来的正是峰值差
        self.assertEqual(plain.intermediate_peak_bytes, tight.intermediate_total_bytes)
        self.assertLess(tight.intermediate_peak_bytes, plain.intermediate_peak_bytes)

    def test_release_false_keeps_inputs_alive_to_the_end(self):
        """输入也留在环境里，不按"最后一次使用"截断——否则会低估占用。"""
        g = Graph()
        x = g.input("x", f32[4, 4])
        g.output(g.silu(g.relu(x)), "y")

        plain = plan_memory(g, release=False)
        by_name = {f.name: f for f in plain.values}
        self.assertEqual(by_name["x"].defined_at, -1)
        self.assertEqual(by_name["x"].last_used_at, len(g.nodes))
        self.assertEqual(plain.peak_live_bytes, plain.total_bytes)

    def test_event_scan_matches_brute_force(self):
        """事件扫描与逐时刻暴力扫描必须逐项相等。

        图刻意做出交错的生命周期：``a`` 定义得早、最后一次使用在末尾，
        ``b`` 被两个节点读，``x`` 被隔开使用。只在"相邻值首尾相接"的链上
        测，事件排序的同刻顺序不会被检验到。
        """
        g = Graph()
        x = g.input("x", f32[4, 4])
        a = g.relu(x)
        b = g.silu(a)
        c = g.add(x, b)
        d = g.relu(b)
        e = g.mul(c, d)
        g.output(g.add(a, e), "y")

        def brute(budget, intermediates_only, aligned):
            pool = [
                f for f in budget.values if not (intermediates_only and f.is_input)
            ]
            last = max(f.last_used_at for f in pool)
            best = 0
            for moment in range(-1, last + 1):
                live = sum(
                    (f.aligned_bytes if aligned else f.bytes)
                    for f in pool
                    if f.defined_at <= moment <= f.last_used_at
                )
                best = max(best, live)
            return best

        names = {
            (False, False): "peak_live_bytes",
            (False, True): "intermediate_peak_bytes",
            (True, False): "aligned_peak_bytes",
            (True, True): "aligned_intermediate_peak_bytes",
        }

        for release in (True, False):
            budget = plan_memory(g, release=release)
            for aligned in (False, True):
                for only_intermediates in (False, True):
                    want = brute(budget, only_intermediates, aligned)
                    got = getattr(budget, names[(aligned, only_intermediates)])
                    self.assertEqual(
                        got,
                        want,
                        f"release={release} aligned={aligned} "
                        f"intermediates={only_intermediates}",
                    )


class TestAttentionBackwardBudget(unittest.TestCase):
    """AF-30：反向重算 scores/probs，但二次缓冲必须能被回收复用。

    若某次改动让这些 ``(BH,S,T)`` 中间量都活到图结束，峰值会退化成"不复用"
    口径的数值；这里用比例钉住"复用确实发生了"，而不是只看绝对字节。
    """

    def test_quadratic_buffers_are_reused(self):
        b, h, s, d = 2, 2, 32, 8
        g = Graph("attn")
        q = g.input("q", f32[b, h, s, d])
        k = g.input("k", f32[b, h, s, d])
        v = g.input("v", f32[b, h, s, d])
        g.output(g.attention(q, k, v, scale=0.125), "y")

        bwd = backward(g, wrt=[q, k, v])
        kept = plan_memory(bwd, release=False)
        reused = plan_memory(bwd, release=True)

        self.assertGreater(kept.peak_live_bytes, reused.peak_live_bytes)
        # 本图实测约 4 倍；留出余量，钉住"至少省掉三分之二"这个量级
        self.assertLess(reused.peak_live_bytes * 3, kept.peak_live_bytes)


if __name__ == "__main__":
    unittest.main()
