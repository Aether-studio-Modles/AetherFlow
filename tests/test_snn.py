"""SNN 与替代梯度的测试。

这一组要证明四件事：

1. `spike` 的前向与 `step` **逐位相同**——替代梯度只改反向，不改语义；
2. 反向确实使用三角窗，且峰值 / 边界的数值可手算验证；
3. LIF 的一步与多步展开结果与手算一致（不是"能跑"就算过）；
4. 端到端训练能让损失下降——否则"可训练"只是口号。

另外验证：反向图仍然过 verifier、能 JSON 往返、且全是 PORTABLE 节点。
"""

import unittest

import numpy as np

from aetherflow import (
    Graph,
    ReferenceBackend,
    backward,
    dumps,
    f32,
    f64,
    i32,
    loads,
    verify,
    verify_portable_only,
)
from aetherflow.models import (
    LIFSpec,
    SNNSpec,
    build_snn,
    build_snn_events,
    build_snn_network,
    lif_step,
    mean_rate,
)

try:  # 内核与全部测试必须在没有 torch 的机器上也能跑（ADR-005）
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

_REF = ReferenceBackend()


def _run(graph, feed):
    return _REF.execute(graph, feed)


class TestSpikeForward(unittest.TestCase):
    def test_forward_matches_step(self):
        """spike 与 step 的前向必须逐位相同（阈值语义一致）。"""
        rng = np.random.default_rng(0)
        xs = (rng.standard_normal((7, 5)) * 1.5 + 0.5).astype(np.float32)

        g_step = Graph("step")
        x = g_step.input("x", f32[7, 5])
        g_step.output(g_step.add_node("step", [x], {"threshold": 1.0}), "y")

        g_spike = Graph("spike")
        x2 = g_spike.input("x", f32[7, 5])
        g_spike.output(
            g_spike.add_node("spike", [x2], {"threshold": 1.0, "alpha": 0.3}), "y"
        )

        np.testing.assert_array_equal(
            _run(g_step, {"x": xs})["y"], _run(g_spike, {"x": xs})["y"]
        )

    def test_default_threshold_is_one(self):
        """不写 threshold 时按神经元语义取 1.0，而不是 step 的 0.0。"""
        g = Graph("spike")
        x = g.input("x", f32[4])
        g.output(g.add_node("spike", [x]), "y")
        out = _run(g, {"x": np.array([-0.5, 0.0, 0.5, 1.5], dtype=np.float32)})["y"]
        np.testing.assert_array_equal(out, np.array([0, 0, 0, 1], dtype=np.float32))


class TestSurrogateGradient(unittest.TestCase):
    """三角窗替代梯度：峰值 1/alpha 落在阈值上，窗外为 0。

    这不是"近似阶跃的导数"，而是**定义**——所以数值必须精确，用 assert_allclose
    到机器精度，而不是给个容差糊过去。
    """

    def _dx(self, xs, threshold, alpha):
        g = Graph("surrogate")
        x = g.input("x", f64[len(xs)])
        g.output(
            g.add_node("spike", [x], {"threshold": threshold, "alpha": alpha}), "s"
        )
        bwd = backward(g, wrt=[x], seeds={"s": "ds"})
        return _run(bwd, {"x": np.asarray(xs, dtype=np.float64),
                          "ds": np.ones(len(xs), dtype=np.float64)})["dx"]

    def test_alpha_one(self):
        xs = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0]
        got = self._dx(xs, threshold=1.0, alpha=1.0)
        np.testing.assert_allclose(got, [0.0, 0.0, 0.5, 1.0, 0.5, 0.0], atol=1e-12)

    def test_alpha_half_scales_peak(self):
        # 窗宽减半：窗变窄，峰值翻倍
        got = self._dx([0.75, 1.0, 1.25], threshold=1.0, alpha=0.5)
        np.testing.assert_allclose(got, [1.0, 2.0, 1.0], atol=1e-12)

    def test_gradient_finite_at_threshold(self):
        """x 恰好等于阈值时不能出现 NaN（用 sqrt(d*d) 表达 |d| 就会）。"""
        got = self._dx([1.0, 1.0, 1.0], threshold=1.0, alpha=1.0)
        self.assertTrue(np.isfinite(got).all())
        np.testing.assert_allclose(got, [1.0, 1.0, 1.0], atol=1e-12)

    def test_nonpositive_alpha_rejected(self):
        g = Graph("surrogate")
        x = g.input("x", f64[2])
        g.output(g.add_node("spike", [x], {"threshold": 1.0, "alpha": 0.0}), "s")
        with self.assertRaises(ValueError):
            backward(g, wrt=[x])


class TestLifStep(unittest.TestCase):
    def _step(self, x, v, spec):
        g = Graph("lif")
        xv = g.input("x", f32[1, 1])
        vv = g.input("v", f32[1, 1])
        s, vn = lif_step(g, xv, vv, spec)
        g.output(s, "s")
        g.output(vn, "v_next")
        return _run(g, {"x": np.array([[x]], dtype=np.float32),
                        "v": np.array([[v]], dtype=np.float32)})

    def test_spike_and_subtract_reset(self):
        # v_pre = 0.5*0.6 + 0.8 = 1.1 > 1 → 发放，重置后 0.1
        out = self._step(0.8, 0.6, LIFSpec(decay=0.5, threshold=1.0))
        self.assertEqual(float(out["s"][0, 0]), 1.0)
        self.assertAlmostEqual(float(out["v_next"][0, 0]), 0.1, places=6)

    def test_no_spike_keeps_membrane(self):
        # v_pre = 0.5*0.6 + 0.2 = 0.5 < 1 → 不发放，电位留到 0.5
        out = self._step(0.2, 0.6, LIFSpec(decay=0.5, threshold=1.0))
        self.assertEqual(float(out["s"][0, 0]), 0.0)
        self.assertAlmostEqual(float(out["v_next"][0, 0]), 0.5, places=6)

    def test_reset_none_accumulates(self):
        out = self._step(0.8, 0.6, LIFSpec(decay=0.5, threshold=1.0, reset="none"))
        self.assertEqual(float(out["s"][0, 0]), 1.0)
        self.assertAlmostEqual(float(out["v_next"][0, 0]), 1.1, places=6)

    def test_dtype_mismatch_rejected(self):
        g = Graph("lif")
        x = g.input("x", f32[1, 1])
        v = g.input("v", f64[1, 1])
        with self.assertRaises(TypeError):
            lif_step(g, x, v)


class TestTemporalUnroll(unittest.TestCase):
    def _graph(self, spec=None, steps=2):
        spec = spec or LIFSpec(decay=0.5, threshold=1.0)
        g = Graph("snn")
        x = g.input("x", f32[steps, 1, 1])
        spikes, v = build_snn(g, x, spec)
        g.output(mean_rate(g, spikes), "rate")
        g.output(v, "v")
        return g

    def test_two_steps_hand_computed(self):
        # t0: v = 0.8，不发放；t1: v_pre = 0.5*0.8 + 0.8 = 1.2，发放，重置 0.2
        g = self._graph()
        out = _run(g, {"x": np.array([[[0.8]], [[0.8]]], dtype=np.float32)})
        self.assertAlmostEqual(float(out["rate"][0, 0]), 0.5, places=6)
        self.assertAlmostEqual(float(out["v"][0, 0]), 0.2, places=6)

    def test_no_spikes(self):
        g = self._graph()
        out = _run(g, {"x": np.zeros((2, 1, 1), dtype=np.float32)})
        self.assertEqual(float(out["rate"][0, 0]), 0.0)

    def test_symbolic_time_rejected(self):
        g = Graph("snn_bad")
        x = g.input("x", f32["T", 2, 3])
        with self.assertRaises(ValueError):
            build_snn(g, x)

    def test_initial_membrane_honored(self):
        g = Graph("snn_init")
        x = g.input("x", f32[1, 1, 1])
        v0 = g.input("v0", f32[1, 1])
        spikes, v = build_snn(g, x, LIFSpec(decay=0.5, threshold=1.0), init=v0)
        g.output(v, "v")
        # v_pre = 0.5*1.0 + 0.0 = 0.5 → 不发放
        out = _run(g, {"x": np.zeros((1, 1, 1), dtype=np.float32),
                       "v0": np.ones((1, 1), dtype=np.float32)})
        self.assertAlmostEqual(float(out["v"][0, 0]), 0.5, places=6)


class TestSnnGraphContract(unittest.TestCase):
    def _graph(self):
        g = Graph("snn_contract")
        w = g.input("w", f32[3, 2])
        b = g.input("b", f32[2])
        x = g.input("x", f32[3, 4, 3])
        spikes, _ = build_snn(g, x, LIFSpec(decay=0.5, threshold=0.5))
        logits = g.matmul(mean_rate(g, spikes), w)
        g.output(g.softmax(g.add(logits, b), axis=-1), "y")
        return g

    def test_forward_is_portable_and_verifies(self):
        g = self._graph()
        verify(g)
        verify_portable_only(g)

    def test_backward_graph_is_portable_and_verifies(self):
        g = self._graph()
        bwd = backward(g, wrt=[v for v in g.inputs if v.name != "x"])
        verify(bwd)
        verify_portable_only(bwd)

    def test_json_round_trip(self):
        g = self._graph()
        g2 = loads(dumps(g))
        rng = np.random.default_rng(3)
        feed = {
            "w": (rng.standard_normal((3, 2)) * 0.3).astype(np.float32),
            "b": np.zeros(2, dtype=np.float32),
            "x": (rng.random((3, 4, 3)) * 0.8).astype(np.float32),
        }
        np.testing.assert_allclose(_run(g, feed)["y"], _run(g2, feed)["y"], rtol=1e-6)

    def test_backward_json_round_trip(self):
        g = self._graph()
        bwd = backward(g, wrt=[g.find("w")])
        bwd2 = loads(dumps(bwd))
        rng = np.random.default_rng(4)
        feed = {
            "w": (rng.standard_normal((3, 2)) * 0.3).astype(np.float32),
            "b": np.zeros(2, dtype=np.float32),
            "x": (rng.random((3, 4, 3)) * 0.8).astype(np.float32),
            "dy": rng.standard_normal((4, 2)).astype(np.float32),
        }
        np.testing.assert_allclose(
            _run(bwd, feed)["dw"], _run(bwd2, feed)["dw"], rtol=1e-6
        )


class TestSparseLinear(unittest.TestCase):
    """事件驱动的稀疏线性：CSR 事件必须与稠密 ``matmul(spikes, weights)`` 同义。

    SNN 里真正贵的是物化 ``T × K`` 的脉冲矩阵，所以每个用例都拿等价的稠密
    乘法当参照——对拍的不是"能跑"，而是"稀疏路径没有偷偷改变语义"。
    """

    ROWS, COLS, STEPS = 6, 5, 4
    # 第 1 段为空（offsets[1] == offsets[2]），中间两段含重复行
    OFFSETS = np.array([0, 2, 2, 4, 5], dtype=np.int32)
    INDICES = np.array([1, 3, 3, 0, 1], dtype=np.int32)

    def _feed(self, seed=0, indices=None, offsets=None):
        indices = self.INDICES if indices is None else indices
        offsets = self.OFFSETS if offsets is None else offsets
        rng = np.random.default_rng(seed)
        return {
            "weights": rng.random((self.ROWS, self.COLS), dtype=np.float32),
            "indices": np.asarray(indices, dtype=np.int32),
            "values": rng.random(len(indices), dtype=np.float32),
            "offsets": np.asarray(offsets, dtype=np.int32),
        }

    def _graph(self):
        g = Graph("sparse_linear")
        weights = g.input("weights", f32[self.ROWS, self.COLS])
        indices = g.input("indices", i32[self.INDICES.size])
        values = g.input("values", f32[self.INDICES.size])
        offsets = g.input("offsets", i32[self.STEPS + 1])
        g.output(
            g.add_node("sparse_linear", [weights, indices, values, offsets]), "y"
        )
        return g

    @staticmethod
    def _dense(feed):
        """把事件还原成稠密脉冲矩阵，再做普通矩阵乘。"""
        steps, rows = feed["offsets"].size - 1, feed["weights"].shape[0]
        dense = np.zeros((steps, rows), dtype=np.float32)
        for s in range(steps):
            lo, hi = int(feed["offsets"][s]), int(feed["offsets"][s + 1])
            np.add.at(dense[s], feed["indices"][lo:hi], feed["values"][lo:hi])
        return dense @ feed["weights"]

    def test_forward_matches_dense_matmul(self):
        feed = self._feed(seed=5)
        got = _run(self._graph(), feed)["y"]
        np.testing.assert_allclose(got, self._dense(feed), rtol=1e-6, atol=1e-6)

    def test_empty_step_stays_zero(self):
        """空步必须输出全零，而不是沿用上一步或跳过整行。"""
        feed = self._feed(seed=6)
        got = _run(self._graph(), feed)["y"]
        np.testing.assert_array_equal(got[1], np.zeros(self.COLS, dtype=np.float32))

    def test_repeated_row_accumulates_within_a_step(self):
        """同一步内同一神经元多次发放：贡献是累加，不是覆盖。"""
        offsets = np.array([0, 2, 2, 2, 2], dtype=np.int32)
        indices = np.array([3, 3, 2, 2, 2], dtype=np.int32)
        feed = self._feed(seed=7, indices=indices, offsets=offsets)
        got = _run(self._graph(), feed)["y"]
        expect = (feed["values"][0] + feed["values"][1]) * feed["weights"][3]
        np.testing.assert_allclose(got[0], expect, rtol=1e-6, atol=1e-6)

    def test_weight_gradient_matches_dense(self):
        feed = self._feed(seed=8)
        rng = np.random.default_rng(9)
        dout = rng.random((self.STEPS, self.COLS), dtype=np.float32)

        g = self._graph()
        bwd = backward(g, wrt=[g.find("weights")], seeds={"y": "dy"})
        got = _run(bwd, {**feed, "dy": dout})["dweights"]

        dense = np.zeros((self.STEPS, self.ROWS), dtype=np.float32)
        for s in range(self.STEPS):
            lo, hi = int(feed["offsets"][s]), int(feed["offsets"][s + 1])
            np.add.at(dense[s], feed["indices"][lo:hi], feed["values"][lo:hi])
        np.testing.assert_allclose(got, dense.T @ dout, rtol=1e-6, atol=1e-6)

    def test_event_count_can_be_symbolic(self):
        """事件数 E 是符号维：同一张图能接不同长度的 CSR 事件。

        这正是 SNN 的常态（每个 batch 的发放次数不同），也是"稠密脉冲矩阵
        从不物化"这句话在形状系统里的落点。
        """
        g = Graph("sparse_symbolic")
        weights = g.input("weights", f32[self.ROWS, self.COLS])
        indices = g.input("indices", i32["E"])
        values = g.input("values", f32["E"])
        offsets = g.input("offsets", i32[self.STEPS + 1])
        g.output(
            g.add_node("sparse_linear", [weights, indices, values, offsets]), "y"
        )

        for nnz in (0, 3, 5):
            indices = np.arange(nnz, dtype=np.int32) % self.ROWS
            offsets = np.zeros(self.STEPS + 1, dtype=np.int32)
            offsets[1:] = np.linspace(0, nnz, self.STEPS, dtype=np.int32)
            feed = self._feed(seed=nnz, indices=indices, offsets=offsets)
            got = _run(g, feed)["y"]
            np.testing.assert_allclose(got, self._dense(feed), rtol=1e-6, atol=1e-6)

    def test_out_of_range_index_is_rejected(self):
        feed = self._feed(seed=10, indices=np.array([1, 6, 3, 0, 1], dtype=np.int32))
        with self.assertRaises(ValueError):
            _run(self._graph(), feed)

    def test_offsets_must_be_non_decreasing(self):
        feed = self._feed(
            seed=11, offsets=np.array([0, 2, 1, 4, 5], dtype=np.int32)
        )
        with self.assertRaises(ValueError):
            _run(self._graph(), feed)

    def test_symbolic_offsets_length_is_refused_at_build(self):
        """offsets 的长度决定输出步数，符号维表达不了它，必须在构图期拒绝。"""
        g = Graph("bad_offsets")
        weights = g.input("weights", f32[self.ROWS, self.COLS])
        indices = g.input("indices", i32["E"])
        values = g.input("values", f32["E"])
        offsets = g.input("offsets", i32["T"])
        with self.assertRaises(ValueError):
            g.add_node("sparse_linear", [weights, indices, values, offsets])

    def test_graph_verifies_and_round_trips(self):
        g = self._graph()
        verify(g)
        feed = self._feed(seed=12)
        np.testing.assert_allclose(
            _run(g, feed)["y"], _run(loads(dumps(g)), feed)["y"], rtol=1e-6
        )

        bwd = backward(g, wrt=[g.find("weights")])
        np.testing.assert_allclose(
            _run(bwd, {**feed, "dy": np.ones((self.STEPS, self.COLS), np.float32)})[
                "dweights"
            ],
            _run(loads(dumps(bwd)), {
                **feed,
                "dy": np.ones((self.STEPS, self.COLS), np.float32),
            })["dweights"],
            rtol=1e-6,
        )


class TestSpikeLinear(unittest.TestCase):
    """融合事件线性：事件幅度直接取自脉冲，必须与稠密 ``matmul`` 同义。

    脉冲值刻意不取 1，并且留一个空步——前者能发现"把幅度当常量"的实现，
    后者能发现"空步沿用上一行"的实现。
    """

    STEPS, ROWS, COLS = 4, 6, 5
    OFFSETS = np.array([0, 2, 2, 4, 5], dtype=np.int32)
    INDICES = np.array([1, 3, 3, 4, 2], dtype=np.int32)
    VALUES = np.array([0.5, 1.5, 2.5, 0.25, 1.0], dtype=np.float32)

    def _spikes(self):
        spikes = np.zeros((self.STEPS, self.ROWS), dtype=np.float32)
        for s in range(self.STEPS):
            lo, hi = int(self.OFFSETS[s]), int(self.OFFSETS[s + 1])
            spikes[s, self.INDICES[lo:hi]] = self.VALUES[lo:hi]
        return spikes

    def _feed(self, seed=0):
        rng = np.random.default_rng(seed)
        return {
            "spikes": self._spikes(),
            "weights": rng.random((self.ROWS, self.COLS), dtype=np.float32),
            "indices": self.INDICES,
            "offsets": self.OFFSETS,
        }

    def _graph(self):
        g = Graph("spike_linear")
        spikes = g.input("spikes", f32[self.STEPS, self.ROWS])
        weights = g.input("weights", f32[self.ROWS, self.COLS])
        indices = g.input("indices", i32[self.INDICES.size])
        offsets = g.input("offsets", i32[self.STEPS + 1])
        g.output(g.add_node("spike_linear", [spikes, weights, indices, offsets]), "y")
        return g

    def test_forward_matches_dense(self):
        feed = self._feed(seed=1)
        got = _run(self._graph(), feed)["y"]
        np.testing.assert_allclose(
            got, feed["spikes"] @ feed["weights"], rtol=1e-6, atol=1e-6
        )

    def test_empty_step_stays_zero(self):
        got = _run(self._graph(), self._feed(seed=2))["y"]
        np.testing.assert_array_equal(got[1], np.zeros(self.COLS, dtype=np.float32))

    def test_spike_gradient_matches_dense_matmul(self):
        feed = self._feed(seed=3)
        rng = np.random.default_rng(4)
        feed["dy"] = rng.random((self.STEPS, self.COLS), dtype=np.float32)

        g = self._graph()
        bwd = backward(g, wrt=[g.find("spikes")], seeds={"y": "dy"})
        np.testing.assert_allclose(
            _run(bwd, feed)["dspikes"],
            feed["dy"] @ feed["weights"].T,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_unfired_positions_receive_gradient(self):
        """零脉冲位置的梯度必须非零，否则 surrogate 在窗内的项会被丢掉。"""
        feed = self._feed(seed=5)
        feed["dy"] = np.ones((self.STEPS, self.COLS), dtype=np.float32)
        g = self._graph()
        bwd = backward(g, wrt=[g.find("spikes")], seeds={"y": "dy"})
        got = _run(bwd, feed)["dspikes"]
        zero = feed["spikes"] == 0
        self.assertTrue(zero.any())
        self.assertGreater(float(np.abs(got[zero]).max()), 0.0)

    def test_weight_gradient_matches_dense(self):
        feed = self._feed(seed=6)
        rng = np.random.default_rng(7)
        feed["dy"] = rng.random((self.STEPS, self.COLS), dtype=np.float32)
        g = self._graph()
        bwd = backward(g, wrt=[g.find("weights")], seeds={"y": "dy"})
        np.testing.assert_allclose(
            _run(bwd, feed)["dweights"],
            feed["spikes"].T @ feed["dy"],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_graph_round_trips(self):
        g = self._graph()
        feed = self._feed(seed=8)
        np.testing.assert_allclose(
            _run(g, feed)["y"], _run(loads(dumps(g)), feed)["y"], rtol=1e-6
        )


class TestSnnEventsModel(unittest.TestCase):
    """事件驱动构件：前向必须与稠密多层 SNN 完全一致。

    ``build_snn_events`` 换掉的只是每层的矩阵乘；权重、偏置、膜电位与读出
    全部同构，所以这里可以对拍到逐位相等，而不是"大致接近"。
    """

    @staticmethod
    def _feed(spec, seed=0):
        rng = np.random.default_rng(seed)
        feed = {
            "x": rng.random(
                (spec.steps, spec.batch, spec.input_dim), dtype=np.float32
            )
        }
        for name, shape in spec.param_shapes.items():
            feed[name] = (rng.random(shape, dtype=np.float32) - 0.5) * 0.8
        return feed

    def test_matches_dense_network_for_every_depth(self):
        for hidden in ((8,), (8, 8), (6, 10, 7)):
            spec = SNNSpec(input_dim=6, hidden=hidden, steps=4, batch=3)
            dense = Graph("dense")
            _, dy = build_snn_network(dense, spec)
            dense.output(dy, "out")
            events = Graph("events")
            _, ey = build_snn_events(events, spec)
            events.output(ey, "out")
            verify(dense)
            verify(events)

            feed = self._feed(spec)
            want = _run(dense, feed)["out"]
            self.assertGreater(float(np.abs(want).sum()), 0.0, "对拍要有发放，不能全零")
            np.testing.assert_array_equal(_run(events, feed)["out"], want, str(hidden))

    def test_spike_offsets_and_csr_split_rows(self):
        """两算子分工：起点表给出每步区间，列号只在区间内按列递增写入。"""
        spikes = np.zeros((4, 5), dtype=np.float32)
        spikes[0, [1, 4]] = 1.0
        spikes[2, [0, 2, 3]] = 1.0
        spikes[3, 1] = 1.0
        g = Graph("csr")
        x = g.input("x", f32[4, 5])
        offsets = g.add_node("spike_offsets", [x])
        indices = g.add_node("spike_csr", [x, offsets], {"capacity": 12})
        g.output(offsets, "offsets")
        g.output(indices, "indices")
        out = _run(g, {"x": spikes})
        np.testing.assert_array_equal(out["offsets"], [0, 2, 2, 5, 6])
        np.testing.assert_array_equal(
            out["indices"], [1, 4, 0, 2, 3, 1, 0, 0, 0, 0, 0, 0]
        )

    def test_spike_csr_rejects_overflow(self):
        """容量是静态图的硬边界：溢出必须报错，不能静默丢事件。"""
        g = Graph("csr_small")
        x = g.input("x", f32[4, 5])
        offsets = g.add_node("spike_offsets", [x])
        g.output(g.add_node("spike_csr", [x, offsets], {"capacity": 2}), "indices")
        with self.assertRaises(ValueError):
            _run(g, {"x": np.ones((4, 5), dtype=np.float32)})

    def test_capacity_above_element_count_is_refused_at_build(self):
        g = Graph("csr_big")
        x = g.input("x", f32[4, 5])
        offsets = g.add_node("spike_offsets", [x])
        with self.assertRaises(ValueError):
            g.add_node("spike_csr", [x, offsets], {"capacity": 21})

    def test_events_graph_round_trips(self):
        spec = SNNSpec(input_dim=4, hidden=(5, 5), steps=3, batch=2)
        g = Graph("events_json")
        _, y = build_snn_events(g, spec)
        g.output(y, "out")
        feed = self._feed(spec, seed=3)
        np.testing.assert_array_equal(
            _run(g, feed)["out"], _run(loads(dumps(g)), feed)["out"]
        )

    def test_gradients_match_dense_network(self):
        """事件路径可训练：输入与全部参数的梯度都要与稠密路径一致。

        反向对脉冲走普通 ``matmul``（而不是只散回非零事件），所以未发放
        位置的 surrogate 梯度也在；否则初始层一次都不发放时梯度会恒为零。
        前向逐位相同，反向只差浮点累加顺序（权重梯度的原子加），因此这里是
        1e-6 量级的对拍。
        """
        spec = SNNSpec(input_dim=6, hidden=(8, 8), steps=4, batch=3)
        dense = Graph("dense")
        di, dy = build_snn_network(dense, spec)
        dense.output(dy, "out")
        events = Graph("events")
        ei, ey = build_snn_events(events, spec)
        events.output(ey, "out")

        feed = self._feed(spec, seed=11)
        rng = np.random.default_rng(12)
        feed["dout"] = rng.standard_normal(
            (spec.batch, spec.hidden[-1])
        ).astype(np.float32)

        names = ["x", *spec.param_shapes]
        dbwd = backward(dense, wrt=[di[n] for n in names], seeds={"out": "dout"})
        ebwd = backward(events, wrt=[ei[n] for n in names], seeds={"out": "dout"})
        got_d = _run(dbwd, feed)
        got_e = _run(ebwd, feed)
        for n in names:
            np.testing.assert_allclose(
                got_e["d" + n],
                got_d["d" + n],
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"事件路径与稠密路径的梯度不一致: {n}",
            )

    def test_gradients_match_dense_on_device(self):
        """设备后端也要与稠密路径一致：事件内核不能"只在参考实现里对"。

        没有 GPU 运行时的时候整条跳过；HIP 与 CUDA 复用同一份 L2 描述，
        跑的是同一套内核，因此这条测试对两者都成立。
        """
        backends = []
        try:
            from aetherflow.backends.hip import HIP_AVAILABLE, HipBackend

            if HIP_AVAILABLE:
                backends.append(("hip", HipBackend()))
        except Exception:
            pass
        try:
            from aetherflow.backends.cuda import CUDA_AVAILABLE, CudaBackend

            if CUDA_AVAILABLE:
                backends.append(("cuda", CudaBackend()))
        except Exception:
            pass
        if not backends:
            self.skipTest("没有可用的 GPU 后端")

        spec = SNNSpec(input_dim=6, hidden=(8, 8), steps=4, batch=3)
        dense = Graph("dense_dev")
        di, dy = build_snn_network(dense, spec)
        dense.output(dy, "out")
        events = Graph("events_dev")
        ei, ey = build_snn_events(events, spec)
        events.output(ey, "out")

        feed = self._feed(spec, seed=11)
        rng = np.random.default_rng(12)
        feed["dout"] = rng.standard_normal(
            (spec.batch, spec.hidden[-1])
        ).astype(np.float32)

        names = ["x", *spec.param_shapes]
        dbwd = backward(dense, wrt=[di[n] for n in names], seeds={"out": "dout"})
        ebwd = backward(events, wrt=[ei[n] for n in names], seeds={"out": "dout"})

        for name, backend in backends:
            with self.subTest(backend=name):
                got_d = backend.execute(dbwd, feed)
                got_e = backend.execute(ebwd, feed)
                for n in names:
                    np.testing.assert_allclose(
                        got_e["d" + n],
                        got_d["d" + n],
                        rtol=1e-5,
                        atol=1e-5,
                        err_msg=f"{name} 后端上事件路径的梯度不一致: {n}",
                    )


class TestSnnTraining(unittest.TestCase):
    def test_sgd_on_spiking_net_reduces_loss(self):
        """替代梯度必须真的能让参数学到东西，否则它只是个好看的公式。"""
        g = Graph("snn_cls")
        x = g.input("x", f32[4, 6, 3])  # T=4, B=6, D=3
        w = g.input("w", f32[3, 2])
        b = g.input("b", f32[2])
        spikes, _ = build_snn(g, x, LIFSpec(decay=0.5, threshold=0.5, alpha=0.5))
        logits = g.matmul(mean_rate(g, spikes), w)
        g.output(g.softmax(g.add(logits, b), axis=-1), "y")

        rng = np.random.default_rng(0)
        xs = (rng.random((4, 6, 3)) * 1.2).astype(np.float32)
        target = np.zeros((6, 2), dtype=np.float32)
        target[:, 0] = 1.0
        params = {
            "w": (rng.standard_normal((3, 2)) * 0.5).astype(np.float32),
            "b": np.zeros(2, dtype=np.float32),
        }

        def forward(p):
            return _run(g, {"x": xs, **p})["y"]

        def loss_of(p):
            return float(0.5 * np.sum((forward(p) - target) ** 2))

        bwd = backward(g, wrt=[w, b])
        before = loss_of(params)
        lr = 0.5
        for _ in range(40):
            seed = (forward(params) - target).astype(np.float32)
            res = _run(bwd, {"x": xs, "dy": seed, **params})
            params = {k: (params[k] - lr * res[f"d{k}"]).astype(np.float32) for k in params}

        after = loss_of(params)
        self.assertLess(after, before * 0.9, f"损失没有明显下降：{before} -> {after}")

    def test_gradient_flows_through_spikes(self):
        """至少有一个参数拿到非零梯度——全零说明替代梯度没接上。"""
        g = Graph("snn_grad")
        x = g.input("x", f32[2, 3, 2])
        w = g.input("w", f32[2, 2])
        spikes, _ = build_snn(g, x, LIFSpec(decay=0.5, threshold=0.3))
        g.output(g.matmul(mean_rate(g, spikes), w), "y")
        bwd = backward(g, wrt=[w, x])
        rng = np.random.default_rng(1)
        out = _run(bwd, {
            "x": (rng.random((2, 3, 2)) * 0.8).astype(np.float32),
            "w": np.eye(2, dtype=np.float32),
            "dy": np.ones((3, 2), dtype=np.float32),
        })
        self.assertTrue(np.isfinite(out["dw"]).all())
        self.assertGreater(float(np.abs(out["dw"]).sum()), 0.0)

    def test_sgd_on_event_path_reduces_loss(self):
        """事件路径不只是"梯度对得上"：同一套 SGD 真的能把损失压下去。

        两层（第 0 层稠密、第 1 层走 CSR 事件）覆盖压缩、散射与替代梯度三条
        链路。阈值取宽，因为这里验证的是"方向对"，不是收敛速度。
        """
        spec = SNNSpec(
            input_dim=3,
            hidden=(4, 4),
            steps=4,
            batch=6,
            num_classes=2,
            lif=LIFSpec(decay=0.5, threshold=0.5, alpha=0.5),
        )
        g = Graph("events_cls")
        inputs, logits = build_snn_events(g, spec)
        g.output(g.softmax(logits, axis=-1), "y")
        bwd = backward(
            g, wrt=[inputs[n] for n in spec.param_shapes], seeds={"y": "dy"}
        )

        rng = np.random.default_rng(0)
        xs = (rng.random((spec.steps, spec.batch, spec.input_dim)) * 1.2).astype(
            np.float32
        )
        target = np.zeros((spec.batch, spec.num_classes), dtype=np.float32)
        target[:, 0] = 1.0
        params = {
            n: (rng.standard_normal(shape) * 0.5).astype(np.float32)
            if len(shape) > 1
            else np.zeros(shape, dtype=np.float32)
            for n, shape in spec.param_shapes.items()
        }

        def forward(p):
            return _run(g, {"x": xs, **p})["y"]

        def loss_of(p):
            return float(0.5 * np.sum((forward(p) - target) ** 2))

        before = loss_of(params)
        lr = 0.5
        for _ in range(60):
            seed = (forward(params) - target).astype(np.float32)
            res = _run(bwd, {"x": xs, "dy": seed, **params})
            params = {
                k: (params[k] - lr * res[f"d{k}"]).astype(np.float32)
                for k in params
            }

        after = loss_of(params)
        self.assertLess(
            after,
            before * 0.5,
            f"事件路径的损失没有明显下降：{before} -> {after}",
        )


class TestTorchSurrogateCheck(unittest.TestCase):
    """用一份手写 PyTorch 的替代梯度做外部参照。

    torch 没有 LIF，所以这里按 spikingjelly 一类库的通行写法自定义
    ``autograd.Function``，把三角窗写进 ``backward``，再对比两条路径的参数梯度。
    它验证的不是"能跑"，而是**我们定义的替代梯度与主流实现是同一个东西**。
    """

    @unittest.skipUnless(TORCH_AVAILABLE, "需要 torch")
    def test_snn_gradients_match_torch(self):
        T, B, D, C = 3, 4, 5, 2
        decay, thr, alpha = 0.5, 0.5, 1.0
        rng = np.random.default_rng(7)
        xs = (rng.random((T, B, D)) * 1.2).astype(np.float32)
        ws = (rng.standard_normal((D, C)) * 0.5).astype(np.float32)
        bs = (rng.standard_normal(C) * 0.1).astype(np.float32)
        target = np.zeros((B, C), dtype=np.float32)
        target[:, 0] = 1.0

        # --- IR 路径 ---
        g = Graph("snn_torch")
        x = g.input("x", f32[T, B, D])
        w = g.input("w", f32[D, C])
        bias = g.input("b", f32[C])
        spikes, _ = build_snn(g, x, LIFSpec(decay=decay, threshold=thr, alpha=alpha))
        logits = g.add(g.matmul(mean_rate(g, spikes), w), bias)
        g.output(g.softmax(logits, axis=-1), "y")

        feed = {"x": xs, "w": ws, "b": bs}
        y0 = _run(g, feed)["y"]
        bwd = backward(g, wrt=[w, bias])
        got = _run(bwd, {**feed, "dy": (y0 - target).astype(np.float32)})

        # --- torch 路径：同样的一步 LIF 与三角窗替代梯度 ---
        class Spike(torch.autograd.Function):
            @staticmethod
            def forward(ctx, v):
                ctx.save_for_backward(v)
                return (v > thr).to(v.dtype)

            @staticmethod
            def backward(ctx, g_):
                (v,) = ctx.saved_tensors
                wgt = torch.relu(1.0 - (v - thr).abs() / alpha) / alpha
                return g_ * wgt

        tx = torch.tensor(xs)
        tw = torch.tensor(ws, requires_grad=True)
        tb = torch.tensor(bs, requires_grad=True)
        v = torch.zeros(B, D)
        acc = torch.zeros(B, D)
        for t in range(T):
            v = decay * v + tx[t]
            s = Spike.apply(v)
            v = v - thr * s
            acc = acc + s
        ty = torch.softmax((acc / T) @ tw + tb, dim=-1)
        loss = 0.5 * ((ty - torch.tensor(target)) ** 2).sum()
        loss.backward()

        np.testing.assert_allclose(
            got["dw"], tw.grad.numpy(), rtol=1e-4, atol=1e-5, err_msg="dw"
        )
        np.testing.assert_allclose(
            got["db"], tb.grad.numpy(), rtol=1e-4, atol=1e-5, err_msg="db"
        )


class TestLifSpecValidation(unittest.TestCase):
    def test_bad_decay(self):
        with self.assertRaises(ValueError):
            LIFSpec(decay=1.0)

    def test_bad_threshold(self):
        with self.assertRaises(ValueError):
            LIFSpec(threshold=0.0)

    def test_bad_reset(self):
        with self.assertRaises(ValueError):
            LIFSpec(reset="soft")


if __name__ == "__main__":
    unittest.main()
