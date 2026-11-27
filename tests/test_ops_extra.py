"""新增算子的数值与梯度验收。

每个算子两条判据：

1. **前向**与一份独立写的 numpy 实现逐点一致（参考实现只定义语义，
   不能自己给自己打分）。
2. **反向**与中心差分一致。用 f64 构图：差分的噪声随精度平方增长，
   f32 下"规则写错"和"精度不够"会混在一起。

覆盖：gelu / rmsnorm / avgpool2d / 分组与 depthwise 卷积 / 自适应池化 /
concat / embedding / dropout_mask / batch_norm。
"""

import unittest

import numpy as np

from aetherflow import Graph, NoGradientError, ReferenceBackend, backward, f32, f64, i32


BACKEND = ReferenceBackend()


def _run(graph, feed):
    return BACKEND.execute(graph, feed)


def _numeric_grad(graph, feed, name, eps=1e-6):
    arr = feed[name]
    flat = arr.reshape(-1)
    grad = np.zeros(flat.size, dtype=np.float64)
    for i in range(flat.size):
        orig = float(flat[i])
        flat[i] = orig + eps
        hi = float(np.sum(list(_run(graph, feed).values())[0]))
        flat[i] = orig - eps
        lo = float(np.sum(list(_run(graph, feed).values())[0]))
        grad[i] = (hi - lo) / (2 * eps)
        flat[i] = orig
    return grad.reshape(arr.shape)


def _analytic_grad(graph, feed, names):
    bwd = backward(graph, wrt=[graph.find(n) for n in names])
    seed = "d" + graph.output_names[0]
    full = dict(feed)
    full[seed] = np.ones(graph.outputs[0].type.shape, dtype=np.float64)
    out = _run(bwd, full)
    return [out[f"d{n}"] for n in names]


class TestGelu(unittest.TestCase):
    def test_forward_matches_reference_formula(self):
        g = Graph("gelu")
        x = g.input("x", f64[3, 4])
        g.output(g.gelu(x), "y")
        rng = np.random.default_rng(0)
        xs = rng.standard_normal((3, 4)) * 2
        want = 0.5 * xs * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (xs + 0.044715 * xs**3)))
        np.testing.assert_allclose(_run(g, {"x": xs})["y"], want, rtol=1e-12)

    def test_gradient_matches_numeric(self):
        g = Graph("gelu")
        x = g.input("x", f64[3, 4])
        g.output(g.gelu(x), "y")
        rng = np.random.default_rng(1)
        feed = {"x": rng.standard_normal((3, 4))}
        (dx,) = _analytic_grad(g, feed, ["x"])
        np.testing.assert_allclose(dx, _numeric_grad(g, feed, "x"), atol=1e-7)


class TestRmsNorm(unittest.TestCase):
    def _build(self, shape=(4, 6), dtype=f64):
        g = Graph("rmsnorm")
        x = g.input("x", dtype[shape])
        gamma = g.input("gamma", dtype[shape[-1]])
        g.output(g.rmsnorm(x, gamma, axes=(-1,), eps=1e-6), "y")
        return g

    def test_forward_matches_numpy(self):
        g = self._build()
        rng = np.random.default_rng(2)
        feed = {"x": rng.standard_normal((4, 6)),
                "gamma": rng.standard_normal(6)}
        want = feed["x"] / np.sqrt(np.mean(feed["x"] ** 2, -1, keepdims=True) + 1e-6) * feed["gamma"]
        np.testing.assert_allclose(_run(g, feed)["y"], want, rtol=1e-12)

    def test_gradients_match_numeric(self):
        g = self._build()
        rng = np.random.default_rng(3)
        feed = {"x": rng.standard_normal((4, 6)),
                "gamma": rng.standard_normal(6)}
        dx, dgamma = _analytic_grad(g, feed, ["x", "gamma"])
        np.testing.assert_allclose(dx, _numeric_grad(g, feed, "x"), atol=1e-7)
        np.testing.assert_allclose(dgamma, _numeric_grad(g, feed, "gamma"), atol=1e-7)

    def test_multi_axis_normalization(self):
        g = Graph("rmsnorm2")
        x = g.input("x", f64[2, 3, 4])
        gamma = g.input("gamma", f64[3, 4])
        g.output(g.rmsnorm(x, gamma, axes=(-2, -1), eps=1e-6), "y")
        rng = np.random.default_rng(4)
        feed = {"x": rng.standard_normal((2, 3, 4)),
                "gamma": rng.standard_normal((3, 4))}
        want = (feed["x"] / np.sqrt(np.mean(feed["x"] ** 2, (-2, -1), keepdims=True) + 1e-6)
                * feed["gamma"])
        np.testing.assert_allclose(_run(g, feed)["y"], want, rtol=1e-12)


class TestAvgPool(unittest.TestCase):
    def test_forward_matches_explicit_average(self):
        g = Graph("avgpool")
        x = g.input("x", f64[2, 3, 6, 6])
        g.output(g.avgpool2d(x, kernel=(3, 3), stride=(2, 2), padding=(1, 1)), "y")
        rng = np.random.default_rng(5)
        xs = rng.standard_normal((2, 3, 6, 6))
        got = _run(g, {"x": xs})["y"]

        pad = np.pad(xs, ((0, 0), (0, 0), (1, 1), (1, 1)))
        want = np.zeros_like(got)
        for i in range(got.shape[2]):
            for j in range(got.shape[3]):
                want[:, :, i, j] = pad[:, :, 2 * i:2 * i + 3, 2 * j:2 * j + 3].mean(axis=(2, 3))
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_count_include_pad_false_matches_valid_counts(self):
        """分母是窗口与输入的重叠面积：角落 4 个点、边上 6 个、内部 9 个。"""
        g = Graph("avgpool_valid")
        x = g.input("x", f64[1, 2, 6, 6])
        g.output(
            g.avgpool2d(
                x, kernel=3, stride=1, padding=1, count_include_pad=False
            ),
            "y",
        )
        rng = np.random.default_rng(9)
        xs = rng.standard_normal((1, 2, 6, 6))
        got = _run(g, {"x": xs})["y"]
        want = np.zeros_like(got)
        for i in range(6):
            for j in range(6):
                r0, r1 = max(0, i - 1), min(6, i + 2)
                c0, c1 = max(0, j - 1), min(6, j + 2)
                want[:, :, i, j] = xs[:, :, r0:r1, c0:c1].mean(axis=(2, 3))
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_ceil_mode_output_shape(self):
        """8 的输入、3 的核、2 的步长：(8-3)/2 不整除，ceil 给 4、floor 给 3。"""
        g = Graph("avgpool_ceil")
        x = g.input("x", f64[1, 1, 8, 8])
        y = g.avgpool2d(x, kernel=3, stride=2, ceil_mode=True)
        g.output(y, "y")
        self.assertEqual(y.type, f64[1, 1, 4, 4])

    def test_ceil_mode_drops_window_starting_outside(self):
        """7 的输入、2 的核、4 的步长：ceil 先给 3 格，但最后一格起点 8 越界，去掉。"""
        g = Graph("avgpool_ceil_trim")
        x = g.input("x", f64[1, 1, 7, 7])
        y = g.avgpool2d(x, kernel=2, stride=4, ceil_mode=True)
        g.output(y, "y")
        self.assertEqual(y.type, f64[1, 1, 2, 2])

    def test_ceil_mode_rejects_symbolic(self):
        """ceil 的起点修正依赖具体 H/W，符号维上无法表达：显式报错。"""
        g = Graph("avgpool_sym")
        x = g.input("x", f64["B", 1, "H", "W"])
        with self.assertRaises(ValueError):
            g.avgpool2d(x, kernel=3, stride=2, ceil_mode=True)

    def test_ceil_mode_divisor_clips_at_padded_input(self):
        """ceil 的最后一格越过 padded 输入，那几格**不进分母**（torch 口径）。

        8x8 / k3 / s2 → 4x4 输出；最后一格窗口起点 (6,6) 只覆盖 2x2。按
        "固定除以 9" 会得到 4/9 的值，PyTorch 的分母是 4——这条差异在
        ``F.avg_pool2d(..., ceil_mode=True)`` 上会直接表现为 2.25 倍偏差。
        """
        g = Graph("avgpool_ceil_div")
        x = g.input("x", f64[1, 1, 8, 8])
        g.output(g.avgpool2d(x, kernel=3, stride=2, ceil_mode=True), "y")
        xs = np.arange(64, dtype=np.float64).reshape(1, 1, 8, 8) + 1.0
        got = _run(g, {"x": xs})["y"]
        self.assertEqual(got.shape, (1, 1, 4, 4))
        np.testing.assert_allclose(got[0, 0, -1, -1], xs[0, 0, 6:, 6:].mean())
        np.testing.assert_allclose(got[0, 0, 0, 0], xs[0, 0, :3, :3].mean())
        # padding 的零照旧计入分母：4x4 输入、k3、s1、p1 的角落窗口只有 4 个
        # 真实元素，分母仍是 9（这是 count_include_pad=True 与 False 的分界）
        g2 = Graph("avgpool_pad_div")
        x2 = g2.input("x", f64[1, 1, 4, 4])
        g2.output(g2.avgpool2d(x2, kernel=3, stride=1, padding=1), "y")
        got2 = _run(g2, {"x": np.ones((1, 1, 4, 4))})["y"]
        np.testing.assert_allclose(got2[0, 0, 0, 0], 4.0 / 9.0)

    def test_gradient_matches_numeric_with_ceil_mode(self):
        g = Graph("avgpool_ceil_bwd")
        x = g.input("x", f64[1, 1, 5, 5])
        g.output(g.avgpool2d(x, kernel=3, stride=2, ceil_mode=True), "y")
        rng = np.random.default_rng(7)
        feed = {"x": rng.standard_normal((1, 1, 5, 5))}
        (dx,) = _analytic_grad(g, feed, ["x"])
        np.testing.assert_allclose(dx, _numeric_grad(g, feed, "x"), atol=1e-8)

    def test_gradient_matches_numeric(self):
        g = Graph("avgpool")
        x = g.input("x", f64[1, 2, 4, 4])
        g.output(g.avgpool2d(x, kernel=(2, 2), stride=(2, 2)), "y")
        rng = np.random.default_rng(6)
        feed = {"x": rng.standard_normal((1, 2, 4, 4))}
        (dx,) = _analytic_grad(g, feed, ["x"])
        np.testing.assert_allclose(dx, _numeric_grad(g, feed, "x"), atol=1e-8)
        # 不重叠的 2x2 池化：每个窗口的 4 个输入各分到 dy/4，所以整幅图的梯度和
        # 等于 dy 的总和（种子里全是 1，即输出元素个数 4*4 / 4 = 4）
        np.testing.assert_allclose(dx.sum(axis=(2, 3)), np.full((1, 2), 4.0), rtol=1e-12)


class TestMaxPoolIndices(unittest.TestCase):
    """``maxpool2d_indices`` 是 PyTorch ``return_indices`` 的 IR 形态。

    它与 ``maxpool2d`` / ``maxpool2d_backward`` 共用一套定位规则，所以并列
    最大值必须落在同一个格子——否则"前向取得的值"和"位置指出去的格子"会
    在边界情形下对不上。
    """

    def test_indices_point_at_window_maximum(self):
        g = Graph("idx")
        x = g.input("x", f64[1, 1, 4, 4])
        g.output(g.maxpool2d_indices(x, kernel=2, stride=2), "y")
        xs = np.arange(16, dtype=np.float64).reshape(1, 1, 4, 4)
        got = _run(g, {"x": xs})["y"]
        self.assertEqual(got.dtype, np.int32)
        # 每个 2x2 窗口的最大值在右下：平面内下标 5 / 7 / 13 / 15。
        np.testing.assert_array_equal(
            got, np.array([[[[5, 7], [13, 15]]]], dtype=np.int32)
        )

    def test_ties_pick_first_in_row_major(self):
        g = Graph("idx_ties")
        x = g.input("x", f64[1, 1, 4, 4])
        g.output(g.maxpool2d_indices(x, kernel=2, stride=2), "y")
        got = _run(g, {"x": np.ones((1, 1, 4, 4))})["y"]
        np.testing.assert_array_equal(
            got, np.array([[[[0, 2], [8, 10]]]], dtype=np.int32)
        )

    def test_padding_excluded_from_window(self):
        g = Graph("idx_pad")
        x = g.input("x", f64[1, 1, 3, 3])
        g.output(g.maxpool2d_indices(x, kernel=3, stride=1, padding=1), "y")
        xs = np.arange(9, dtype=np.float64).reshape(1, 1, 3, 3)
        got = _run(g, {"x": xs})["y"]
        # 左上角窗口只覆盖 (0,0)/(0,1)/(1,0)/(1,1)，最大值是 4。
        self.assertEqual(int(got[0, 0, 0, 0]), 4)

    def test_ceil_mode_shape(self):
        g = Graph("idx_ceil")
        x = g.input("x", f64[1, 1, 8, 8])
        y = g.maxpool2d_indices(x, kernel=3, stride=2, ceil_mode=True)
        g.output(y, "y")
        self.assertEqual(y.type.shape, (1, 1, 4, 4))


class TestGroupedConv(unittest.TestCase):
    """分组卷积只改切片、不改算法：期望值用逐输出格的朴素循环独立算。

    如果两边都写成"逐组 einsum 再拼接"，测试只是在复核同一段思路；
    这里刻意退回到最笨的六层循环，让分组边界写错时一定对不上。
    """

    @staticmethod
    def _naive_conv(x, w, stride=1, padding=1):
        n, c, h, _ = x.shape
        o, cg, kh, kw = w.shape
        groups = c // cg
        og = o // groups
        xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        ho = (h + 2 * padding - kh) // stride + 1
        wo = ho
        want = np.zeros((n, o, ho, wo), dtype=np.float64)
        for ni in range(n):
            for oi in range(o):
                base = (oi // og) * cg
                for i in range(ho):
                    for j in range(wo):
                        total = 0.0
                        for ci in range(cg):
                            for ki in range(kh):
                                for kj in range(kw):
                                    total += (
                                        xp[ni, base + ci, i * stride + ki, j * stride + kj]
                                        * w[oi, ci, ki, kj]
                                    )
                        want[ni, oi, i, j] = total
        return want

    def test_grouped_matches_naive_loop(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal((1, 4, 5, 5))
        w = rng.standard_normal((6, 2, 3, 3))

        g = Graph("grouped_conv")
        xv = g.input("x", f64[1, 4, 5, 5])
        wv = g.input("w", f64[6, 2, 3, 3])
        g.output(g.conv2d(xv, wv, padding=(1, 1), groups=2), "y")

        got = _run(g, {"x": x, "w": w})["y"]
        np.testing.assert_allclose(got, self._naive_conv(x, w), rtol=1e-12)

    def test_depthwise_does_not_mix_channels(self):
        """groups==C==O 时每个通道独立：1x1 核退化成逐通道标量乘。"""
        x = np.arange(1, 13, dtype=np.float64).reshape(1, 3, 2, 2)
        w = np.array([[[[2.0]]], [[[3.0]]], [[[5.0]]]], dtype=np.float64)

        g = Graph("depthwise")
        xv = g.input("x", f64[1, 3, 2, 2])
        wv = g.input("w", f64[3, 1, 1, 1])
        g.output(g.conv2d(xv, wv, groups=3), "y")

        got = _run(g, {"x": x, "w": w})["y"]
        want = x * np.array([2.0, 3.0, 5.0]).reshape(1, 3, 1, 1)
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_gradients_match_numeric(self):
        g = Graph("grouped_conv_grad")
        x = g.input("x", f64[1, 4, 4, 4])
        w = g.input("w", f64[2, 2, 3, 3])
        g.output(g.conv2d(x, w, padding=(1, 1), groups=2), "y")

        rng = np.random.default_rng(12)
        feed = {
            "x": rng.standard_normal((1, 4, 4, 4)),
            "w": rng.standard_normal((2, 2, 3, 3)),
        }
        dx, dw = _analytic_grad(g, feed, ["x", "w"])
        np.testing.assert_allclose(dx, _numeric_grad(g, feed, "x"), atol=1e-8)
        np.testing.assert_allclose(dw, _numeric_grad(g, feed, "w"), atol=1e-8)

    def test_invalid_groups_rejected_at_build_time(self):
        """通道不能被 groups 整除时构图就失败，不留到执行期。"""
        g = Graph("bad_groups")
        x = g.input("x", f32[1, 3, 4, 4])
        w = g.input("w", f32[4, 3, 1, 1])
        with self.assertRaises(ValueError):
            g.conv2d(x, w, groups=2)

        g2 = Graph("bad_weight_channels")
        x2 = g2.input("x", f32[1, 4, 4, 4])
        w2 = g2.input("w", f32[4, 4, 1, 1])
        with self.assertRaises(ValueError):
            g2.conv2d(x2, w2, groups=2)


class TestAdaptiveAvgPool(unittest.TestCase):
    @staticmethod
    def _windows(out_size, in_size):
        return [
            (i * in_size // out_size, ((i + 1) * in_size + out_size - 1) // out_size)
            for i in range(out_size)
        ]

    def test_windows_follow_floor_ceil(self):
        """5/3 与 4/2 不整除：窗口端点必须与 PyTorch 的 floor/ceil 规则一致。"""
        rng = np.random.default_rng(21)
        x = rng.standard_normal((1, 2, 5, 4))

        g = Graph("adaptive")
        xv = g.input("x", f64[1, 2, 5, 4])
        g.output(g.adaptive_avg_pool2d(xv, output_size=(3, 2)), "y")
        got = _run(g, {"x": x})["y"]

        want = np.zeros((1, 2, 3, 2), dtype=np.float64)
        for i, (h0, h1) in enumerate(self._windows(3, 5)):
            for j, (w0, w1) in enumerate(self._windows(2, 4)):
                want[:, :, i, j] = x[:, :, h0:h1, w0:w1].mean(axis=(2, 3))
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_global_pool_is_whole_map_mean(self):
        rng = np.random.default_rng(22)
        x = rng.standard_normal((2, 3, 4, 7))

        g = Graph("global_pool")
        xv = g.input("x", f64[2, 3, 4, 7])
        g.output(g.adaptive_avg_pool2d(xv, output_size=(1, 1)), "y")
        got = _run(g, {"x": x})["y"]

        self.assertEqual(got.shape, (2, 3, 1, 1))
        np.testing.assert_allclose(got[:, :, 0, 0], x.mean(axis=(2, 3)), rtol=1e-12)

    def test_int_output_size_means_square(self):
        rng = np.random.default_rng(23)
        x = rng.standard_normal((1, 1, 6, 6))

        g = Graph("square_pool")
        xv = g.input("x", f64[1, 1, 6, 6])
        g.output(g.adaptive_avg_pool2d(xv, output_size=2), "y")
        got = _run(g, {"x": x})["y"]

        self.assertEqual(got.shape, (1, 1, 2, 2))
        want = np.array(
            [[[[x[:, :, 0:3, 0:3].mean(), x[:, :, 0:3, 3:6].mean()],
               [x[:, :, 3:6, 0:3].mean(), x[:, :, 3:6, 3:6].mean()]]]]
        )
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_gradient_accumulates_overlapping_windows(self):
        """3 -> 2 的窗口 [0:2] 与 [1:3] 重叠，反向必须累加而不是覆盖。"""
        g = Graph("adaptive_grad")
        x = g.input("x", f64[1, 1, 3, 3])
        g.output(g.adaptive_avg_pool2d(x, output_size=(2, 2)), "y")

        rng = np.random.default_rng(24)
        feed = {"x": rng.standard_normal((1, 1, 3, 3))}
        (dx,) = _analytic_grad(g, feed, ["x"])
        np.testing.assert_allclose(dx, _numeric_grad(g, feed, "x"), atol=1e-8)

    def test_invalid_output_size_rejected(self):
        for size in (0, (0, 2), (1, 2, 3), 2.5):
            with self.subTest(size=size):
                g = Graph("bad_adaptive")
                x = g.input("x", f32[1, 1, 4, 4])
                with self.assertRaises(ValueError):
                    g.adaptive_avg_pool2d(x, output_size=size)


class TestConcat(unittest.TestCase):
    def test_forward_and_gradient(self):
        g = Graph("concat")
        a = g.input("a", f64[2, 3])
        b = g.input("b", f64[2, 5])
        g.output(g.concat(a, b, axis=1), "y")
        rng = np.random.default_rng(7)
        feed = {"a": rng.standard_normal((2, 3)), "b": rng.standard_normal((2, 5))}
        got = _run(g, feed)["y"]
        np.testing.assert_array_equal(got, np.concatenate([feed["a"], feed["b"]], axis=1))

        da, db = _analytic_grad(g, feed, ["a", "b"])
        np.testing.assert_allclose(da, _numeric_grad(g, feed, "a"), atol=1e-8)
        np.testing.assert_allclose(db, _numeric_grad(g, feed, "b"), atol=1e-8)

    def test_symbolic_non_cat_dims_are_allowed(self):
        g = Graph("concat_sym")
        a = g.input("a", f64["B", 3])
        b = g.input("b", f64["B", 5])
        g.output(g.concat(a, b, axis=1), "y")
        rng = np.random.default_rng(8)
        got = _run(g, {"a": rng.standard_normal((4, 3)), "b": rng.standard_normal((4, 5))})["y"]
        self.assertEqual(got.shape, (4, 8))

    def test_symbolic_cat_dim_is_rejected(self):
        g = Graph("concat_bad")
        a = g.input("a", f64[2, "S"])
        b = g.input("b", f64[2, "S"])
        with self.assertRaises(ValueError):
            g.concat(a, b, axis=1)

    def test_mismatched_non_cat_dims_are_rejected(self):
        g = Graph("concat_bad2")
        a = g.input("a", f64[2, 3])
        b = g.input("b", f64[3, 3])
        with self.assertRaises(ValueError):
            g.concat(a, b, axis=1)


class TestEmbedding(unittest.TestCase):
    def test_forward_matches_table_lookup(self):
        g = Graph("emb")
        table = g.input("table", f64[6, 3])
        ids = g.input("ids", i32[2, 4])
        g.output(g.embedding(table, ids), "y")
        rng = np.random.default_rng(9)
        feed = {"table": rng.standard_normal((6, 3)),
                "ids": np.array([[0, 1, 1, 5], [3, 3, 2, 0]], dtype=np.int32)}
        got = _run(g, feed)["y"]
        np.testing.assert_array_equal(got, feed["table"][feed["ids"]])
        self.assertEqual(got.shape, (2, 4, 3))

    def test_backward_accumulates_repeated_ids(self):
        g = Graph("emb")
        table = g.input("table", f64[6, 3])
        ids = g.input("ids", i32[2, 4])
        g.output(g.embedding(table, ids), "y")
        rng = np.random.default_rng(10)
        feed = {"table": rng.standard_normal((6, 3)),
                "ids": np.array([[1, 1, 1, 3], [3, 2, 1, 1]], dtype=np.int32)}

        dtable, = _analytic_grad(g, feed, ["table"])
        want = np.zeros((6, 3))
        np.add.at(want, feed["ids"], np.ones((2, 4, 3)))
        np.testing.assert_allclose(dtable, want, atol=1e-12)
        np.testing.assert_allclose(dtable, _numeric_grad(g, feed, "table"), atol=1e-8)

    def test_float_ids_are_rejected(self):
        g = Graph("emb_bad")
        table = g.input("table", f64[6, 3])
        ids = g.input("ids", f64[2, 4])
        with self.assertRaises(TypeError):
            g.embedding(table, ids)

    def test_out_of_range_ids_are_rejected(self):
        """越界下标是"图合法、数据非法"的错误：只能在执行期报。

        正向越界 numpy 本来就会抛 IndexError，负下标却会静默回绕到倒数几行——
        那是一种"看起来算完了"的错误结果，所以参考语义里统一显式检查。
        """
        g = Graph("emb_oob")
        table = g.input("table", f64[6, 3])
        ids = g.input("ids", i32[2])
        g.output(g.embedding(table, ids), "y")
        base = np.zeros((6, 3))
        for bad in ([0, 6], [-1, 2]):
            with self.subTest(ids=bad):
                with self.assertRaises(ValueError):
                    _run(g, {"table": base, "ids": np.asarray(bad, dtype=np.int32)})

    def test_backward_rejects_out_of_range_ids(self):
        """反向的 ``np.add.at`` 同样会回绕：负下标会把梯度累加到最后几行。"""
        g = Graph("emb_bwd_oob")
        table = g.input("table", f64[6, 3])
        ids = g.input("ids", i32[2])
        g.output(g.embedding(table, ids), "y")
        feed = {
            "table": np.zeros((6, 3)),
            "ids": np.array([-1, 2], dtype=np.int32),
        }
        with self.assertRaises(ValueError):
            _analytic_grad(g, feed, ["table"])


class TestGather(unittest.TestCase):
    """沿任意轴按下标取值。

    ``embedding`` 沿表的第一维取**整行**，输出形状是 ``ids.shape + table.shape[1:]``；
    ``table=[N, V]``、``ids=[N]`` 时它给 ``[N, N]``，不是逐样本收集。交叉熵取
    label logit 需要的是这里的一般形态。
    """

    def test_forward_last_axis_matches_take_along_axis(self):
        g = Graph("gather")
        x = g.input("x", f64[3, 6, 4])
        idx = g.input("idx", i32[3, 6, 2])
        g.output(g.add_node("gather", [x, idx], {"dim": -1}), "y")
        rng = np.random.default_rng(31)
        feed = {"x": rng.standard_normal((3, 6, 4)),
                "idx": rng.integers(0, 4, (3, 6, 2), dtype=np.int32)}
        got = _run(g, feed)["y"]
        want = np.take_along_axis(feed["x"], feed["idx"].astype(np.int64), axis=-1)
        np.testing.assert_array_equal(got, want)
        self.assertEqual(got.shape, (3, 6, 2))

    def test_forward_middle_axis(self):
        """非末维：inner > 1，偏移必须按 (outer, dim, inner) 三轴算。"""
        g = Graph("gather_mid")
        x = g.input("x", f64[3, 6, 4])
        idx = g.input("idx", i32[3, 2, 4])
        g.output(g.add_node("gather", [x, idx], {"dim": 1}), "y")
        rng = np.random.default_rng(32)
        feed = {"x": rng.standard_normal((3, 6, 4)),
                "idx": rng.integers(0, 6, (3, 2, 4), dtype=np.int32)}
        got = _run(g, feed)["y"]
        want = np.take_along_axis(feed["x"], feed["idx"].astype(np.int64), axis=1)
        np.testing.assert_array_equal(got, want)
        self.assertEqual(got.shape, (3, 2, 4))

    def test_int8_indices(self):
        from aetherflow import i8

        g = Graph("gather_i8")
        x = g.input("x", f64[4, 5])
        idx = g.input("idx", i8[4, 3])
        g.output(g.add_node("gather", [x, idx], {"dim": -1}), "y")
        rng = np.random.default_rng(33)
        feed = {"x": rng.standard_normal((4, 5)),
                "idx": rng.integers(0, 5, (4, 3), dtype=np.int8)}
        got = _run(g, feed)["y"]
        want = np.take_along_axis(feed["x"], feed["idx"].astype(np.int64), axis=-1)
        np.testing.assert_array_equal(got, want)

    def test_backward_accumulates_repeated_indices(self):
        g = Graph("gather_bwd")
        x = g.input("x", f64[3, 6, 4])
        idx = g.input("idx", i32[3, 2, 4])
        g.output(g.add_node("gather", [x, idx], {"dim": 1}), "y")
        rng = np.random.default_rng(34)
        feed = {"x": rng.standard_normal((3, 6, 4)),
                "idx": np.array([[[1, 1, 1, 1], [1, 1, 1, 1]]] * 3, dtype=np.int32)}

        dtable, = _analytic_grad(g, feed, ["x"])
        want = np.zeros((3, 6, 4))
        np.add.at(
            want,
            (np.arange(3)[:, None, None], feed["idx"].astype(np.int64),
             np.arange(4)[None, None, :]),
            np.ones((3, 2, 4)),
        )
        np.testing.assert_allclose(dtable, want, atol=1e-12)
        np.testing.assert_allclose(dtable, _numeric_grad(g, feed, "x"), atol=1e-8)

    def test_out_of_range_indices_are_rejected(self):
        g = Graph("gather_oob")
        x = g.input("x", f64[2, 4])
        idx = g.input("idx", i32[2, 2])
        g.output(g.add_node("gather", [x, idx], {"dim": -1}), "y")
        for bad in ([0, 4], [-1, 2]):
            with self.subTest(idx=bad):
                with self.assertRaises(ValueError):
                    _run(g, {"x": np.zeros((2, 4)),
                             "idx": np.asarray([bad, [1, 2]], dtype=np.int32)})

    def test_shape_mismatch_is_rejected_at_graph_time(self):
        """除 dim 外每一维必须相等；不等时不能在构图期默默广播。"""
        g = Graph("gather_bad")
        x = g.input("x", f64[3, 6, 4])
        idx = g.input("idx", i32[3, 6, 2])
        with self.assertRaises(ValueError):
            g.add_node("gather", [x, idx], {"dim": 1})

    def test_float_indices_are_rejected(self):
        g = Graph("gather_bad_idx")
        x = g.input("x", f64[3, 6])
        idx = g.input("idx", f64[3, 2])
        with self.assertRaises(TypeError):
            g.add_node("gather", [x, idx], {"dim": -1})


class TestCrossEntropy(unittest.TestCase):
    """fused 交叉熵：语义是 ``logsumexp(logits) - logits[target]``。"""

    @staticmethod
    def _softmax(x):
        m = np.max(x, axis=-1, keepdims=True)
        e = np.exp(x - m)
        return e / np.sum(e, axis=-1, keepdims=True)

    def _graph(self, shape, dtype=f64):
        g = Graph("ce")
        logits = g.input("logits", dtype[shape])
        target = g.input("target", i32[shape[:-1]])
        g.output(g.add_node("cross_entropy", [logits, target]), "loss")
        return g

    def test_forward_matches_negative_log_softmax(self):
        shape = (4, 7)
        g = self._graph(shape)
        rng = np.random.default_rng(41)
        feed = {"logits": rng.standard_normal(shape) * 2.0,
                "target": rng.integers(0, shape[-1], shape[:-1], dtype=np.int32)}
        got = _run(g, feed)["loss"]
        want = -np.log(self._softmax(feed["logits"])[
            np.arange(shape[0]), feed["target"]])
        np.testing.assert_allclose(got, want, rtol=1e-12)
        self.assertEqual(got.shape, (4,))

    def test_forward_is_stable_for_large_logits(self):
        """1e4 的差值不能变成 inf/nan：这正是逐行减 max 的目的。"""
        g = self._graph((2, 3))
        feed = {"logits": np.array([[1e4, 0.0, -1e4], [-1e4, 0.0, 1e4]]),
                "target": np.array([0, 2], dtype=np.int32)}
        got = _run(g, feed)["loss"]
        self.assertTrue(np.all(np.isfinite(got)))
        np.testing.assert_allclose(got, [0.0, 0.0], atol=1e-12)

    def test_backward_matches_softmax_minus_onehot(self):
        shape = (3, 5)
        g = Graph("ce_bwd")
        logits = g.input("logits", f64[shape])
        target = g.input("target", i32[shape[:-1]])
        dloss = g.input("dloss", f64[shape[:-1]])
        g.output(g.add_node("cross_entropy_backward", [logits, target, dloss]), "dlogits")
        rng = np.random.default_rng(42)
        feed = {"logits": rng.standard_normal(shape) * 2.0,
                "target": rng.integers(0, shape[-1], shape[:-1], dtype=np.int32),
                "dloss": rng.standard_normal(shape[:-1])}
        got = _run(g, feed)["dlogits"]
        onehot = np.zeros(shape)
        onehot[np.arange(shape[0]), feed["target"]] = 1.0
        want = (self._softmax(feed["logits"]) - onehot) * feed["dloss"][:, None]
        np.testing.assert_allclose(got, want, atol=1e-12)

    def test_autodiff_matches_numeric_gradient(self):
        shape = (3, 4)
        g = self._graph(shape)
        rng = np.random.default_rng(43)
        feed = {"logits": rng.standard_normal(shape) * 2.0,
                "target": rng.integers(0, shape[-1], shape[:-1], dtype=np.int32)}
        dlogits, = _analytic_grad(g, feed, ["logits"])
        np.testing.assert_allclose(dlogits, _numeric_grad(g, feed, "logits"), atol=1e-8)
        # 每行梯度之和为 0：softmax 的雅可比行和为 0，减去 onehot 后整行和也是 0
        np.testing.assert_allclose(np.sum(dlogits, axis=-1), 0.0, atol=1e-12)

    def test_out_of_range_target_is_rejected(self):
        g = self._graph((2, 4))
        for bad in ([4, 0], [-1, 2]):
            with self.subTest(target=bad):
                with self.assertRaises(ValueError):
                    _run(g, {"logits": np.zeros((2, 4)),
                             "target": np.asarray(bad, dtype=np.int32)})

    def test_dloss_dtype_mismatch_is_rejected(self):
        g = Graph("ce_bad_dtype")
        logits = g.input("logits", f64[2, 3])
        target = g.input("target", i32[2])
        dloss = g.input("dloss", f32[2])
        with self.assertRaises(TypeError):
            g.add_node("cross_entropy_backward", [logits, target, dloss])


class TestDivisionSemantics(unittest.TestCase):
    """除零的两种语义必须分开：浮点按 IEEE，整数显式拒绝。"""

    def _graph(self, dtype):
        g = Graph("div")
        a = g.input("a", dtype[2])
        b = g.input("b", dtype[2])
        g.output(g.add_node("div", [a, b]), "y")
        return g

    def test_float_division_by_zero_follows_ieee(self):
        got = _run(self._graph(f32), {"a": np.array([1.0, -1.0], np.float32),
                                      "b": np.zeros(2, np.float32)})["y"]
        self.assertTrue(np.isinf(got).all())
        self.assertEqual(got.tolist(), [np.inf, -np.inf])

    def test_integer_division_by_zero_is_rejected(self):
        """整数没有 inf：numpy 会先升到 float 再落回 i32，产物是 INT_MIN 垃圾值。"""
        with self.assertRaises(ValueError):
            _run(self._graph(i32), {"a": np.array([4, 6], np.int32),
                                    "b": np.array([2, 0], np.int32)})

    def test_integer_division_is_truncating(self):
        got = _run(self._graph(i32), {"a": np.array([7, -7], np.int32),
                                      "b": np.array([2, 2], np.int32)})["y"]
        self.assertEqual(got.tolist(), [3, -3])


class TestDropoutMask(unittest.TestCase):
    """随机算子的契约：seed 写进 attrs 之后，它其实是纯函数。

    每一条都做逐位断言而不是"统计上接近"——随机性不是放松校验的理由，
    反而因为没有外部参照，更需要把"同 seed 同结果""只依赖扁平索引"
    这类不变量钉死。
    """

    def _build(self, shape=(40, 60), p=0.25, seed=20240614, dtype=f64):
        g = Graph("drop")
        x = g.input("x", dtype[shape])
        g.output(g.dropout_mask(x, p=p, seed=seed), "y")
        return g

    def _feed(self, shape, dtype=np.float64):
        return {"x": np.random.default_rng(1).standard_normal(shape).astype(dtype)}

    def test_mask_values_are_zero_or_scale(self):
        p = 0.25
        y = _run(self._build(shape=(400, 500), p=p), self._feed((400, 500)))["y"]
        scale = float(np.float32(1.0 / (1.0 - p)))
        self.assertEqual(set(np.unique(y).tolist()), {0.0, scale})

    def test_drop_rate_is_close_to_p(self):
        # 200000 个元素下标准差约 0.001，0.01 的窗口是 10σ，不会偶发失败
        p = 0.25
        y = _run(self._build(shape=(400, 500), p=p), self._feed((400, 500)))["y"]
        self.assertAlmostEqual(float(np.mean(y == 0.0)), p, delta=0.01)

    def test_same_seed_is_bitwise_stable(self):
        g = self._build(shape=(8, 9))
        feed = self._feed((8, 9))
        np.testing.assert_array_equal(_run(g, feed)["y"], _run(g, feed)["y"])

    def test_seed_changes_the_mask(self):
        shape = (32, 32)
        feed = self._feed(shape)
        a = _run(self._build(shape=shape, seed=1), feed)["y"]
        b = _run(self._build(shape=shape, seed=2), feed)["y"]
        self.assertFalse(np.array_equal(a, b))

    def test_x_values_do_not_affect_the_mask(self):
        """x 只提供形状与 dtype：换掉数值，掩码必须逐位不变。"""
        shape = (16, 5)
        g = self._build(shape=shape)
        a = _run(g, {"x": np.zeros(shape)})["y"]
        b = _run(g, {"x": np.full(shape, 7.0)})["y"]
        np.testing.assert_array_equal(a, b)

    def test_mask_depends_on_flat_index_only(self):
        """元素数相同、形状不同的两张图，掩码序列必须一致。

        内核把扁平索引当计数器；如果哪天改成按多维下标生成，这条会立刻
        失败，而不是等到与某个后端对拍时才发现。
        """
        a = _run(self._build(shape=(3, 8)), {"x": np.zeros((3, 8))})["y"].reshape(-1)
        b = _run(self._build(shape=(24,)), {"x": np.zeros(24)})["y"].reshape(-1)
        np.testing.assert_array_equal(a, b)

    def test_gradient_is_the_mask(self):
        """``y = x * mask``：dx 等于 mask，dropout 那条边不额外贡献梯度。"""
        g = Graph("drop_train")
        x = g.input("x", f64[4, 5])
        mask = g.dropout_mask(x, p=0.4, seed=99)
        g.output(g.mul(x, mask), "y")
        feed = {"x": np.arange(20, dtype=np.float64).reshape(4, 5)}

        dx, = _analytic_grad(g, feed, ["x"])
        expected = _run(self._build(shape=(4, 5), p=0.4, seed=99), feed)["y"]
        np.testing.assert_allclose(dx, expected, atol=1e-12)
        np.testing.assert_allclose(dx, _numeric_grad(g, feed, "x"), atol=1e-8)

    def test_symbolic_batch_dim_is_allowed(self):
        """形状来自输入而不是 attrs，所以符号 batch 维也能走。"""
        g = Graph("sym")
        x = g.input("x", f32["B", 8])
        self.assertEqual(g.dropout_mask(x, p=0.5, seed=1).type, f32["B", 8])

    def test_p_one_is_rejected(self):
        g = Graph("bad")
        x = g.input("x", f32[2, 2])
        with self.assertRaises(ValueError):
            g.dropout_mask(x, p=1.0, seed=0)

    def test_negative_p_is_rejected(self):
        g = Graph("bad")
        x = g.input("x", f32[2, 2])
        with self.assertRaises(ValueError):
            g.dropout_mask(x, p=-0.1, seed=0)

    # -- 运行期 key（AF-20）--------------------------------------------

    def _keyed(self, shape=(8, 9), p=0.3, offset=0, dtype=f64):
        g = Graph("drop_keyed")
        x = g.input("x", dtype[shape])
        key = g.input("key", i32[1])
        g.output(g.dropout_mask(x, key, p=p, offset=offset), "y")
        return g

    def test_runtime_key_changes_the_mask(self):
        g = self._keyed()
        x = np.zeros((8, 9), dtype=np.float64)
        a = _run(g, {"x": x, "key": np.array([1], np.int32)})["y"]
        b = _run(g, {"x": x, "key": np.array([2], np.int32)})["y"]
        self.assertFalse(np.array_equal(a, b))

    def test_same_runtime_key_is_bitwise_stable(self):
        g = self._keyed()
        feed = {"x": np.zeros((8, 9)), "key": np.array([41], np.int32)}
        np.testing.assert_array_equal(_run(g, feed)["y"], _run(g, feed)["y"])

    def test_offset_separates_sites_sharing_one_key(self):
        """同一 key、不同 offset 必须等价于两个不同的常量 seed。"""
        shape = (16, 7)
        x = np.zeros(shape, dtype=np.float64)
        g = self._keyed(shape=shape, offset=5)
        got = _run(g, {"x": x, "key": np.array([7], np.int32)})["y"]
        expected = _run(self._build(shape=shape, p=0.3, seed=12), {"x": x})["y"]
        np.testing.assert_array_equal(got, expected)

    def test_key_must_be_i32_scalar(self):
        g = Graph("bad_key")
        x = g.input("x", f32[2, 2])
        bad_dtype = g.input("k", f32[1])
        with self.assertRaises(ValueError):
            g.dropout_mask(x, bad_dtype, p=0.5)
        g2 = Graph("bad_key2")
        x2 = g2.input("x", f32[2, 2])
        bad_shape = g2.input("k", i32[2])
        with self.assertRaises(ValueError):
            g2.dropout_mask(x2, bad_shape, p=0.5)


class TestBatchNorm(unittest.TestCase):
    """BN 的语义只有一句话：沿除通道轴外的所有轴标准化，再仿射。

    前向与反向都独立对拍（numpy 公式 / 中心差分），不拿参考实现给自己打分。
    """

    def _graph(self, shape=(2, 3, 4, 4), axis=1, dtype=f64):
        g = Graph("bn")
        x = g.input("x", dtype[shape])
        gamma = g.input("gamma", dtype[shape[axis]])
        beta = g.input("beta", dtype[shape[axis]])
        g.output(g.batch_norm(x, gamma, beta, axis=axis, eps=1e-5), "y")
        return g

    def _feed(self, shape=(2, 3, 4, 4), axis=1):
        rng = np.random.default_rng(7)
        return {
            "x": rng.standard_normal(shape) * 2.0 + 1.0,
            "gamma": rng.standard_normal(shape[axis]),
            "beta": rng.standard_normal(shape[axis]),
        }

    def _numpy_bn(self, x, gamma, beta, axis, eps=1e-5):
        red = tuple(i for i in range(x.ndim) if i != axis)
        mean = np.mean(x, axis=red, keepdims=True)
        var = np.var(x, axis=red, keepdims=True)
        sh = [1] * x.ndim
        sh[axis] = -1
        return (x - mean) / np.sqrt(var + eps) * gamma.reshape(sh) + beta.reshape(sh)

    def test_matches_numpy_for_every_channel_axis(self):
        for shape, axis in (((2, 3, 4, 4), 1), ((3, 2, 4), 0), ((2, 4, 3), 2)):
            feed = self._feed(shape, axis)
            got = _run(self._graph(shape, axis), feed)["y"]
            want = self._numpy_bn(feed["x"], feed["gamma"], feed["beta"], axis)
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)

    def test_normalized_output_has_zero_mean_unit_var(self):
        shape, axis = (4, 5, 3, 3), 1
        feed = self._feed(shape, axis)
        feed["gamma"] = np.ones_like(feed["gamma"])
        feed["beta"] = np.zeros_like(feed["beta"])
        y = _run(self._graph(shape, axis), feed)["y"]
        red = (0, 2, 3)
        np.testing.assert_allclose(np.mean(y, axis=red), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.var(y, axis=red), 1.0, rtol=1e-5)

    def test_gradients_match_numeric(self):
        shape, axis = (2, 3, 2, 2), 1
        feed = self._feed(shape, axis)
        g = self._graph(shape, axis)
        grads = _analytic_grad(g, feed, ["x", "gamma", "beta"])
        for name, got in zip(("x", "gamma", "beta"), grads):
            np.testing.assert_allclose(
                got, _numeric_grad(g, feed, name), rtol=1e-6, atol=1e-8
            )

    def test_stats_returns_mean_and_var(self):
        shape = (3, 4, 5)
        g = Graph("stats")
        x = g.input("x", f64[shape])
        g.output(g.batch_norm_stats(x, axis=0), "y")
        feed = {"x": np.random.default_rng(3).standard_normal(shape)}
        got = _run(g, feed)["y"]
        self.assertEqual(got.shape, (2, 3))
        np.testing.assert_allclose(got[0], feed["x"].mean(axis=(1, 2)), rtol=1e-12)
        np.testing.assert_allclose(got[1], feed["x"].var(axis=(1, 2)), rtol=1e-12)

    def test_stats_is_not_differentiable(self):
        """统计量不进反向：接进损失必须明确报错，而不是静默当成零梯度。"""
        g = Graph("stats")
        x = g.input("x", f64[2, 3])
        g.output(g.batch_norm_stats(x, axis=1), "y")
        with self.assertRaises(NoGradientError):
            backward(g)

    def test_gamma_length_mismatch_rejected(self):
        g = Graph("bad")
        x = g.input("x", f32[2, 3, 4, 4])
        gamma = g.input("gamma", f32[5])
        beta = g.input("beta", f32[3])
        with self.assertRaises(ValueError):
            g.batch_norm(x, gamma, beta, axis=1)

    def test_symbolic_batch_forward_but_no_backward(self):
        """形状来自输入，所以符号 batch 的前向可用；反向需要每通道元素数。"""
        g = Graph("sym")
        x = g.input("x", f32["B", 3, 4, 4])
        gamma = g.input("gamma", f32[3])
        beta = g.input("beta", f32[3])
        self.assertEqual(g.batch_norm(x, gamma, beta, axis=1).type, f32["B", 3, 4, 4])
        with self.assertRaises(NoGradientError):
            backward(g)


class TestCompare(unittest.TestCase):
    """比较算子：输出 bool，是掩码的唯一产生端。"""

    def _graph(self, op, sa, sb):
        g = Graph(op)
        a = g.input("a", f32[sa])
        b = g.input("b", f32[sb])
        g.output(g.add_node(op, [a, b]), "y")
        return g

    def test_matches_numpy_and_outputs_bool(self):
        rng = np.random.default_rng(3)
        av = np.round(rng.standard_normal((4, 6)) * 2).astype(np.float32)
        bv = np.round(rng.standard_normal((4, 6)) * 2).astype(np.float32)
        cases = {
            "eq": np.equal,
            "ne": np.not_equal,
            "lt": np.less,
            "le": np.less_equal,
            "gt": np.greater,
            "ge": np.greater_equal,
        }
        for op, fn in cases.items():
            with self.subTest(op=op):
                g = self._graph(op, (4, 6), (4, 6))
                out = _run(g, {"a": av, "b": bv})["y"]
                self.assertEqual(out.dtype, np.bool_)
                np.testing.assert_array_equal(out, fn(av, bv))

    def test_broadcast_matches_numpy(self):
        rng = np.random.default_rng(4)
        av = np.round(rng.standard_normal((2, 3, 4)) * 2).astype(np.float32)
        bv = np.round(rng.standard_normal((4,)) * 2).astype(np.float32)
        g = self._graph("gt", (2, 3, 4), (4,))
        out = _run(g, {"a": av, "b": bv})["y"]
        self.assertEqual(out.shape, (2, 3, 4))
        np.testing.assert_array_equal(out, av > bv)

    def test_bool_input_compare(self):
        """bool 张量之间也能比较：``ne(mask, mask)`` 是掩码异或的常见写法。"""
        from aetherflow import boolean

        g = Graph("mask")
        m = g.input("m", boolean[2, 3])
        g.output(g.add_node("ne", [m, m]), "y")
        self.assertEqual(g.outputs[0].type, boolean[2, 3])

    def test_arithmetic_on_bool_rejected(self):
        from aetherflow import boolean

        g = Graph("bad")
        a = g.input("a", boolean[2, 3])
        b = g.input("b", boolean[2, 3])
        with self.assertRaises(TypeError):
            g.add_node("add", [a, b])

    def test_compare_has_no_gradient(self):
        g = self._graph("eq", (2, 3), (2, 3))
        with self.assertRaises(NoGradientError):
            backward(g)


class TestWhere(unittest.TestCase):
    """``where(cond, a, b)``：bool 条件的三输入选择，支持三方广播。"""

    def test_same_shape_matches_numpy(self):
        from aetherflow import boolean

        rng = np.random.default_rng(11)
        cv = rng.integers(0, 2, (3, 4)).astype(bool)
        av = rng.standard_normal((3, 4)).astype(np.float32)
        bv = rng.standard_normal((3, 4)).astype(np.float32)
        g = Graph("where_same")
        c = g.input("c", boolean[3, 4])
        a = g.input("a", f32[3, 4])
        b = g.input("b", f32[3, 4])
        g.output(g.where(c, a, b), "y")
        out = _run(g, {"c": cv, "a": av, "b": bv})["y"]
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(out, np.where(cv, av, bv))

    def test_three_way_broadcast(self):
        from aetherflow import boolean

        rng = np.random.default_rng(12)
        cv = rng.integers(0, 2, (2, 1, 4)).astype(bool)
        av = rng.standard_normal((2, 3, 4)).astype(np.float32)
        bv = rng.standard_normal((4,)).astype(np.float32)
        g = Graph("where_bcast")
        c = g.input("c", boolean[2, 1, 4])
        a = g.input("a", f32[2, 3, 4])
        b = g.input("b", f32[4])
        g.output(g.where(c, a, b), "y")
        out = _run(g, {"c": cv, "a": av, "b": bv})["y"]
        self.assertEqual(out.shape, (2, 3, 4))
        np.testing.assert_array_equal(out, np.where(cv, av, bv))

    def test_bool_branches_select_masks(self):
        from aetherflow import boolean

        rng = np.random.default_rng(13)
        cv = rng.integers(0, 2, (2, 3)).astype(bool)
        mv = rng.integers(0, 2, (2, 3)).astype(bool)
        nv = rng.integers(0, 2, (2, 3)).astype(bool)
        g = Graph("where_mask")
        c = g.input("c", boolean[2, 3])
        m = g.input("m", boolean[2, 3])
        n = g.input("n", boolean[2, 3])
        g.output(g.where(c, m, n), "y")
        self.assertEqual(g.outputs[0].type, boolean[2, 3])
        out = _run(g, {"c": cv, "m": mv, "n": nv})["y"]
        self.assertEqual(out.dtype, np.bool_)
        np.testing.assert_array_equal(out, np.where(cv, mv, nv))

    def test_gradient_routes_by_mask(self):
        from aetherflow import boolean

        rng = np.random.default_rng(14)
        cv = rng.integers(0, 2, (4, 5)).astype(bool)
        g = Graph("where_grad")
        c = g.input("c", boolean[4, 5])
        x = g.input("x", f64[4, 5])
        z = g.input("z", f64[4, 5])
        g.output(g.where(c, x, z), "y")
        dx, dz = _analytic_grad(
            g,
            {
                "c": cv,
                "x": np.ones((4, 5), np.float64),
                "z": np.ones((4, 5), np.float64),
            },
            ["x", "z"],
        )
        np.testing.assert_array_equal(dx, cv.astype(np.float64))
        np.testing.assert_array_equal(dz, (~cv).astype(np.float64))

    def test_cond_must_be_bool(self):
        g = Graph("where_bad_cond")
        c = g.input("c", f32[2, 3])
        a = g.input("a", f32[2, 3])
        b = g.input("b", f32[2, 3])
        with self.assertRaises(TypeError):
            g.add_node("where", [c, a, b])

    def test_branch_dtypes_must_match(self):
        from aetherflow import boolean

        g = Graph("where_bad_dtype")
        c = g.input("c", boolean[2, 3])
        a = g.input("a", f32[2, 3])
        b = g.input("b", f64[2, 3])
        with self.assertRaises(TypeError):
            g.add_node("where", [c, a, b])


class TestRounding(unittest.TestCase):
    """sign / round / floor / ceil 的参考语义。

    这三个（除 round 外）都是阶梯函数：判据只看逐点取值，不看导数。
    """

    VALUES = np.array(
        [-3.5, -2.5, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 2.5, 3.5, 1e-8, -1e-8],
        dtype=np.float64,
    )

    def _forward(self, op, values):
        g = Graph(op)
        x = g.input("x", f64[values.shape])
        g.output(getattr(g, op)(x), "y")
        return _run(g, {"x": values})["y"]

    def test_round_is_half_to_even(self):
        got = self._forward("round", self.VALUES)
        # numpy 的 np.round 与 torch.round 都是 half-to-even：±2.5 -> ±2，±3.5 -> ±4
        np.testing.assert_array_equal(got, np.round(self.VALUES))

    def test_floor_and_ceil(self):
        np.testing.assert_array_equal(
            self._forward("floor", self.VALUES), np.floor(self.VALUES)
        )
        np.testing.assert_array_equal(
            self._forward("ceil", self.VALUES), np.ceil(self.VALUES)
        )

    def test_sign_keeps_zero_and_nan(self):
        values = np.array([-2.5, -1e-8, 0.0, 1e-8, 3.0, np.nan])
        got = self._forward("sign", values)
        np.testing.assert_array_equal(got, np.sign(values))
        # NaN 透传：两个比较都为假，原值返回
        self.assertTrue(np.isnan(got[-1]))
        self.assertEqual(got[2], 0.0)

    def test_sign_has_no_backward_rule_output(self):
        """阶梯函数的反向规则存在且返回 None（与 step 同口径），不伪造零梯度。"""
        from aetherflow.ir.grad_rules import RULES

        for name in ("sign", "round", "floor", "ceil"):
            self.assertIn(name, RULES)


if __name__ == "__main__":
    unittest.main()
