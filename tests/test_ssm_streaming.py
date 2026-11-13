"""SSM 流式推理（prefill / decode / reset）的验收。

判据是三组恒等式，对应审计报告 AF-26 要求的最小流式接口：

1. **分块等价**：任意切点串起来（state 逐段传递）== 整段前向，输出与末状态都一致；
2. **逐 token 等价**：``steps=1`` 的图循环 T 次 == 一次 prefill；
3. **状态隔离**：带 ``reset`` 的一次前向 == 各段独立跑（零初值），且改未来 token
   不影响前缀。

``selective scan`` 层用 f64：递归把舍入误差按步数累乘，f32 下"规则写错"与
"精度不够"混在一起无法区分。Mamba block 的参数表写死 f32，所以模型级用例用
f32 并放宽容差；等价性仍要求重排后的运算顺序不改变数学。
"""

import os
import tempfile
import unittest

import numpy as np

from aetherflow import Graph, ReferenceBackend, backward, f32, f64, plan_memory
from aetherflow.backends.cuda import CUDA_AVAILABLE, CudaBackend
from aetherflow.models import (
    MambaModelSpec,
    MambaSpec,
    build_mamba_block,
    build_mamba_block_stateful,
    build_mamba_model,
    build_mamba_model_stateful,
    build_selective_scan,
    build_selective_scan_stateful,
    causal_depthwise_conv1d_stateful,
)
from aetherflow.train import ParameterStore, load_checkpoint, save_checkpoint

_REF = ReferenceBackend()


def _run(graph, feed):
    return _REF.execute(graph, feed)


# --------------------------------------------------------------------------
# selective scan：末状态与重置
# --------------------------------------------------------------------------


class TestScanStreaming(unittest.TestCase):
    dims, states = 3, 2

    def _graph(self, steps, with_reset=False):
        g = Graph("scan_stream")
        x = g.input("x", f64[steps, self.dims])
        dt = g.input("dt", f64[steps, self.dims])
        A = g.input("A", f64[self.dims, self.states])
        B = g.input("B", f64[steps, self.states])
        C = g.input("C", f64[steps, self.states])
        h0 = g.input("h0", f64[self.dims, self.states])
        reset = g.input("reset", f64[steps]) if with_reset else None
        y, h = build_selective_scan_stateful(
            g, x, dt, A, B, C, h0=h0, reset=reset
        )
        g.output(y, "y")
        g.output(h, "h")
        return g

    def _feed(self, steps, seed=0):
        rng = np.random.default_rng(seed)
        return {
            "x": rng.standard_normal((steps, self.dims)),
            "dt": rng.random((steps, self.dims)) * 0.3 + 0.05,
            "A": -(rng.random((self.dims, self.states)) + 0.5),
            "B": rng.standard_normal((steps, self.states)),
            "C": rng.standard_normal((steps, self.states)),
            "h0": rng.standard_normal((self.dims, self.states)),
        }

    @staticmethod
    def _slice(feed, start, stop):
        timed = ("x", "dt", "B", "C", "reset")
        return {k: (v[start:stop] if k in timed else v) for k, v in feed.items()}

    def test_chunked_matches_whole(self):
        """任意切点：状态传递不丢、不重，末状态也一致。"""
        steps = 6
        feed = self._feed(steps, seed=1)
        whole = _run(self._graph(steps), feed)

        # 第一段继承整段的非零初值，之后靠末状态传递
        state = np.array(feed["h0"], copy=True)
        pieces, start = [], 0
        for length in (2, 1, 3):
            part = self._slice(feed, start, start + length)
            part["h0"] = state
            res = _run(self._graph(length), part)
            pieces.append(res["y"])
            state = res["h"]
            start += length

        np.testing.assert_allclose(
            np.concatenate(pieces), whole["y"], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(state, whole["h"], rtol=1e-12, atol=1e-12)

    def test_single_step_repeats_the_whole_sequence(self):
        steps = 5
        feed = self._feed(steps, seed=2)
        whole = _run(self._graph(steps), feed)

        state = np.array(feed["h0"], copy=True)
        rows = []
        for t in range(steps):
            part = self._slice(feed, t, t + 1)
            part["h0"] = state
            res = _run(self._graph(1), part)
            rows.append(res["y"])
            state = res["h"]
        np.testing.assert_allclose(
            np.concatenate(rows), whole["y"], rtol=1e-12, atol=1e-12
        )

    def test_reset_matches_independent_runs(self):
        """reset 位置：状态与历史都归零，等价于重新开一段。"""
        steps, cut = 6, 3
        feed = self._feed(steps, seed=3)
        feed["h0"] = np.zeros((self.dims, self.states))
        reset = np.zeros(steps)
        reset[cut] = 1.0

        got = _run(self._graph(steps, with_reset=True), {**feed, "reset": reset})

        zeros = np.zeros((self.dims, self.states))
        first = _run(self._graph(cut), {**self._slice(feed, 0, cut), "h0": zeros})
        second = _run(self._graph(steps - cut), {**self._slice(feed, cut, steps), "h0": zeros})

        np.testing.assert_allclose(
            got["y"], np.concatenate([first["y"], second["y"]]), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(got["h"], second["h"], rtol=1e-12, atol=1e-12)

    def test_reset_isolates_prefix_from_future(self):
        """reset 之后只依赖 reset 之后的数据：改前缀不影响后缀。"""
        steps, cut = 5, 2
        feed = self._feed(steps, seed=4)
        feed["h0"] = np.zeros((self.dims, self.states))
        reset = np.zeros(steps)
        reset[cut] = 1.0

        other = dict(feed)
        other["x"] = feed["x"].copy()
        other["x"][:cut] += 10.0
        other["dt"] = feed["dt"].copy()
        other["dt"][:cut] += 1.0

        graph = self._graph(steps, with_reset=True)
        a = _run(graph, {**feed, "reset": reset})
        b = _run(graph, {**other, "reset": reset})
        np.testing.assert_allclose(a["y"][cut:], b["y"][cut:], rtol=1e-12, atol=1e-12)

    def test_initial_state_gradient(self):
        """非零 h0 的导数不能被当成常量：中心差分对照。"""
        steps = 3
        g = Graph("scan_stream_grad")
        x = g.input("x", f64[steps, self.dims])
        dt = g.input("dt", f64[steps, self.dims])
        A = g.input("A", f64[self.dims, self.states])
        B = g.input("B", f64[steps, self.states])
        C = g.input("C", f64[steps, self.states])
        h0 = g.input("h0", f64[self.dims, self.states])
        y, _ = build_selective_scan_stateful(g, x, dt, A, B, C, h0=h0)
        g.output(y, "y")

        feed = self._feed(steps, seed=5)
        weight = np.random.default_rng(6).standard_normal((steps, self.dims))
        bwd = backward(g, wrt=[g.find("h0")])
        got = _run(bwd, {**feed, "dy": weight})["dh0"]

        eps = 1e-6
        want = np.zeros_like(feed["h0"])
        for idx in np.ndindex(*feed["h0"].shape):
            orig = float(feed["h0"][idx])
            feed["h0"][idx] = orig + eps
            hi = float(np.sum(_run(g, feed)["y"] * weight))
            feed["h0"][idx] = orig - eps
            lo = float(np.sum(_run(g, feed)["y"] * weight))
            feed["h0"][idx] = orig
            want[idx] = (hi - lo) / (2 * eps)
        np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-9)

    def test_reset_shape_is_checked(self):
        g = Graph("scan_stream_bad_reset")
        x = g.input("x", f64[3, self.dims])
        dt = g.input("dt", f64[3, self.dims])
        A = g.input("A", f64[self.dims, self.states])
        B = g.input("B", f64[3, self.states])
        C = g.input("C", f64[3, self.states])
        reset = g.input("reset", f64[4])
        with self.assertRaises(ValueError):
            build_selective_scan_stateful(g, x, dt, A, B, C, reset=reset)


# --------------------------------------------------------------------------
# Mamba block：prefill / decode / reset
# --------------------------------------------------------------------------


def _spec(steps, batch=2, d_model=4, d_state=3, d_conv=3, expand=2):
    return MambaSpec(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        steps=steps,
        batch=batch,
    )


def _params(spec, seed=0):
    """量级小的随机参数：f32 下也要让 ``a = exp(dt * A)`` 落在 (0, 1)。"""
    rng = np.random.default_rng(seed)
    out = {}
    for name, shape in spec.param_shapes.items():
        if name == "A_raw":
            out[name] = rng.uniform(-2.0, 0.0, size=shape).astype(np.float32)
        elif name in ("norm_g", "norm2_g"):
            out[name] = (1.0 + 0.1 * rng.standard_normal(shape)).astype(np.float32)
        else:
            out[name] = (0.3 * rng.standard_normal(shape)).astype(np.float32)
    return out


def _zero_state(spec):
    return {
        "conv0": np.zeros(spec.conv_state_shape, dtype=np.float32),
        "h0": np.zeros(spec.ssm_state_shape, dtype=np.float32),
    }


def _whole_graph(spec):
    g = Graph("mamba_whole")
    _, y = build_mamba_block(g, spec)
    g.output(y, "y")
    return g


def _stateful_graph(spec, with_reset=False):
    g = Graph("mamba_stream")
    _, y, state = build_mamba_block_stateful(g, spec, with_reset=with_reset)
    g.output(y, "y")
    g.output(state["conv"], "conv")
    g.output(state["h"], "h")
    return g


class TestMambaStreaming(unittest.TestCase):
    def setUp(self):
        self.batch, self.d_model = 2, 4
        self.x = np.random.default_rng(10).standard_normal(
            (self.batch, 7, self.d_model)
        ).astype(np.float32)

    def test_chunked_prefill_matches_whole(self):
        """卷积窗口也要跨段接上：切点落在窗口内部最容易暴露边界错误。"""
        spec = _spec(7, batch=self.batch, d_model=self.d_model)
        params = _params(spec)
        whole = _run(_whole_graph(spec), {**params, "x": self.x})

        state = _zero_state(spec)
        pieces, start = [], 0
        for length in (2, 3, 2):
            part = _spec(length, batch=self.batch, d_model=self.d_model)
            feed = {**params, **state, "x": self.x[:, start:start + length, :]}
            res = _run(_stateful_graph(part), feed)
            pieces.append(res["y"])
            state = {"conv0": res["conv"], "h0": res["h"]}
            start += length

        np.testing.assert_allclose(
            np.concatenate(pieces, axis=1), whole["y"], rtol=1e-5, atol=1e-6
        )

    def test_token_by_token_matches_prefill(self):
        """decode：steps=1 的同一份构图循环 T 次，与一次 prefill 等价。"""
        spec = _spec(7, batch=self.batch, d_model=self.d_model)
        params = _params(spec)
        whole = _run(_whole_graph(spec), {**params, "x": self.x})

        state = _zero_state(spec)
        rows = []
        for t in range(self.x.shape[1]):
            part = _spec(1, batch=self.batch, d_model=self.d_model)
            feed = {**params, **state, "x": self.x[:, t:t + 1, :]}
            res = _run(_stateful_graph(part), feed)
            rows.append(res["y"])
            state = {"conv0": res["conv"], "h0": res["h"]}

        np.testing.assert_allclose(
            np.concatenate(rows, axis=1), whole["y"], rtol=1e-5, atol=1e-6
        )

    def test_zero_state_reproduces_the_stateless_block(self):
        """stateful 的零初值必须与 ``build_mamba_block`` 逐位一致。"""
        spec = _spec(5, batch=self.batch, d_model=self.d_model)
        feed = {**_params(spec), "x": self.x[:, :5, :]}
        whole = _run(_whole_graph(spec), feed)
        stream = _run(_stateful_graph(spec), {**feed, **_zero_state(spec)})
        np.testing.assert_array_equal(stream["y"], whole["y"])

    def test_reset_matches_separate_runs(self):
        spec = _spec(6, batch=self.batch, d_model=self.d_model)
        params = _params(spec)
        reset = np.zeros((self.batch, 6), dtype=np.float32)
        reset[:, 3] = 1.0
        x = self.x[:, :6, :]

        got = _run(
            _stateful_graph(spec, with_reset=True),
            {**params, **_zero_state(spec), "x": x, "reset": reset},
        )

        first_spec = _spec(3, batch=self.batch, d_model=self.d_model)
        second_spec = _spec(3, batch=self.batch, d_model=self.d_model)
        first = _run(
            _stateful_graph(first_spec),
            {**_params(first_spec), **_zero_state(first_spec), "x": x[:, :3, :]},
        )
        second = _run(
            _stateful_graph(second_spec),
            {**_params(second_spec), **_zero_state(second_spec), "x": x[:, 3:, :]},
        )

        np.testing.assert_allclose(
            got["y"], np.concatenate([first["y"], second["y"]], axis=1),
            rtol=1e-5, atol=1e-6,
        )
        np.testing.assert_allclose(got["conv"], second["conv"], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(got["h"], second["h"], rtol=1e-6, atol=1e-7)

    def test_causality(self):
        """改未来输入，前缀输出必须逐位不变。"""
        spec = _spec(6, batch=self.batch, d_model=self.d_model)
        feed = {**_params(spec), **_zero_state(spec), "x": self.x[:, :6, :].copy()}
        graph = _stateful_graph(spec)

        before = _run(graph, feed)["y"]
        feed["x"][:, 4:, :] += 5.0
        after = _run(graph, feed)["y"]
        np.testing.assert_array_equal(before[:, :4, :], after[:, :4, :])
        self.assertFalse(np.array_equal(before[:, 4:, :], after[:, 4:, :]))

    def test_state_shapes_are_checked(self):
        """状态形状是协议：错一个维度就要在构图期报出来。"""
        g = Graph("conv_bad_state")
        x = g.input("x", f32[2, 4, 8])
        w = g.input("w", f32[8, 3])
        b = g.input("b", f32[8])
        state = g.input("s", f32[2, 1, 8])  # 应为 (2, 2, 8)
        with self.assertRaises(ValueError):
            causal_depthwise_conv1d_stateful(g, x, w, b, state=state)

        g2 = Graph("scan_bad_h0")
        xs = g2.input("x", f32[3, 4])
        dt = g2.input("dt", f32[3, 4])
        A = g2.input("A", f32[4, 2])
        B = g2.input("B", f32[3, 2])
        C = g2.input("C", f32[3, 2])
        h0 = g2.input("h0", f32[4, 3])  # 应为 (4, 2)
        with self.assertRaises(ValueError):
            build_selective_scan_stateful(g2, xs, dt, A, B, C, h0=h0)

    def test_conv_state_shape_is_reported(self):
        spec = _spec(4, batch=self.batch, d_model=self.d_model, d_conv=3)
        self.assertEqual(spec.conv_state_shape, (self.batch, 2, spec.d_inner))
        self.assertEqual(spec.ssm_state_shape, (self.batch, spec.d_inner, spec.d_state))


# --------------------------------------------------------------------------
# 审计 P0：reset 落在块边界时，带出的卷积窗口必须清掉上一段的帧
# --------------------------------------------------------------------------


class TestResetAcrossChunks(unittest.TestCase):
    """``[10,20,30,1] + [2]`` 在 ``1`` 处 reset：分块必须与整段一样得到 3。

    修复前只有当段的抽头被遮住，带出去的状态仍是 ``[30, 1]``，下一块把 ``30``
    重新读进窗口得到 33。判据是"整段 == 分块"，且窗口状态里严格早于最后一次
    reset 的行全为 0。
    """

    def _run(self, tokens, reset, history):
        tokens = np.asarray(tokens, dtype=np.float32)
        reset = np.asarray(reset, dtype=np.float32)
        history = np.asarray(history, dtype=np.float32)
        if history.ndim == 2:
            history = history[:, :, None]  # (B, K-1) -> (B, K-1, 1)
        batch, steps = tokens.shape
        kernel = history.shape[1] + 1
        g = Graph(f"conv_reset_{steps}")
        y, state = causal_depthwise_conv1d_stateful(
            g,
            g.input("x", f32[batch, steps, 1]),
            g.input("w", f32[1, kernel]),
            g.input("b", f32[1]),
            g.input("history", f32[batch, kernel - 1, 1]),
            g.input("reset", f32[batch, steps]),
        )
        g.output(y, "y")
        g.output(state, "state")
        return _run(g, {
            "x": tokens.reshape(batch, steps, 1),
            "w": np.ones((1, kernel), np.float32),
            "b": np.zeros(1, np.float32),
            "history": history.reshape(batch, kernel - 1, 1),
            "reset": reset,
        })

    def test_reset_at_block_end_does_not_leak_into_next_chunk(self):
        zero = np.zeros((1, 2), np.float32)
        whole = self._run([[10, 20, 30, 1, 2]], [[0, 0, 0, 1, 0]], zero)
        head = self._run([[10, 20, 30, 1]], [[0, 0, 0, 1]], zero)
        tail = self._run([[2]], [[0]], head["state"])

        # 只保留 reset 位置那一帧；30 属于上一段，必须被清掉
        np.testing.assert_array_equal(head["state"][0, :, 0], [0.0, 1.0])
        self.assertAlmostEqual(float(whole["y"][0, -1, 0]), 3.0)
        self.assertAlmostEqual(float(tail["y"][0, -1, 0]), 3.0)

    def test_reset_in_first_k_minus_one_positions(self):
        """reset 落在块首：块内所有帧都有效，块前的历史无效。"""
        history = np.array([[7.0, 8.0, 9.0]], np.float32)  # K-1=3 帧
        got = self._run([[4, 5]], [[1, 0]], history)
        np.testing.assert_array_equal(got["state"][0, :, 0], [0.0, 4.0, 5.0])
        # 整段 [7,8,9,4,5] 在 4 处重置：位置 1 的窗口是 4+5
        self.assertAlmostEqual(float(got["y"][0, 1, 0]), 9.0)

    def test_chunk_shorter_than_window(self):
        """块短于 K-1：状态里同时有"块前"与"块内"的帧，只有后者能留下。"""
        history = np.array([[7.0, 8.0, 9.0]], np.float32)  # K=4
        got = self._run([[4]], [[1]], history)
        np.testing.assert_array_equal(got["state"][0, :, 0], [0.0, 0.0, 4.0])
        # 下一块的窗口只应读到 4 与新输入
        nxt = self._run([[3]], [[0]], got["state"])
        self.assertAlmostEqual(float(nxt["y"][0, 0, 0]), 7.0)

    def test_each_row_resets_independently(self):
        """批内第 0 行在块尾 reset、第 1 行不 reset：只有第 0 行的历史被清。"""
        got = self._run(
            [[10, 20, 30], [10, 20, 30]],
            [[0, 0, 1], [0, 0, 0]],
            np.zeros((2, 2), np.float32),
        )
        np.testing.assert_array_equal(got["state"][0, :, 0], [0.0, 30.0])
        np.testing.assert_array_equal(got["state"][1, :, 0], [20.0, 30.0])

    def test_block_reset_matches_single_run(self):
        """模型级：reset 落在第一块末尾，第二块开头仍要与整段一致。"""
        batch, d_model, steps = 2, 4, 6
        x = np.random.default_rng(41).standard_normal(
            (batch, steps, d_model)
        ).astype(np.float32)
        spec = _spec(steps, batch=batch, d_model=d_model)
        params = _params(spec)
        reset = np.zeros((batch, steps), np.float32)
        reset[:, 2] = 1.0
        whole = _run(
            _stateful_graph(spec, with_reset=True),
            {**params, **_zero_state(spec), "x": x, "reset": reset},
        )

        conv0 = np.zeros(spec.conv_state_shape, np.float32)
        h0 = np.zeros(spec.ssm_state_shape, np.float32)
        first = _run(
            _stateful_graph(_spec(3, batch=batch, d_model=d_model), with_reset=True),
            {**params, "conv0": conv0, "h0": h0, "x": x[:, :3], "reset": reset[:, :3]},
        )
        second = _run(
            _stateful_graph(_spec(3, batch=batch, d_model=d_model)),
            {**params, "conv0": first["conv"], "h0": first["h"], "x": x[:, 3:]},
        )
        np.testing.assert_allclose(
            second["y"], whole["y"][:, 3:], rtol=1e-5, atol=1e-6
        )


# --------------------------------------------------------------------------
# 多层 Mamba 栈：模型级 prefill / decode
# --------------------------------------------------------------------------


def _model_spec(steps, batch=2, d_model=4, d_state=3, d_conv=3, expand=2, num_layers=2):
    return MambaModelSpec(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        steps=steps,
        batch=batch,
        num_layers=num_layers,
    )


def _model_params(spec, seed=0):
    """每层用不同 seed：形状相同但数值不同，才能发现"层被串错"的错误。"""
    out = {}
    for i in range(spec.num_layers):
        for base, value in _params(spec.layer, seed=seed + i).items():
            out[f"l{i}_{base}"] = value
    return out


def _model_zero_state(spec):
    return {k: np.zeros(v, dtype=np.float32) for k, v in spec.state_shapes.items()}


def _model_graph(spec):
    g = Graph("mamba_model_whole")
    _, y = build_mamba_model(g, spec)
    g.output(y, "y")
    return g


def _model_stream_graph(spec, with_reset=False):
    g = Graph("mamba_model_stream")
    _, y, state = build_mamba_model_stateful(g, spec, with_reset=with_reset)
    g.output(y, "y")
    for name, value in state.items():
        g.output(value, name)
    return g


class TestMambaModelStreaming(unittest.TestCase):
    """整栈的验收：状态是"每层各一对"，层与层之间不能串。

    判据与单层相同，但参数故意逐层不同——如果状态在层间错位一格，输出会立刻
    偏离；如果只给第一层传状态，后面几层会退回零初值，分块与整段也会不等。
    """

    def setUp(self):
        self.batch, self.d_model = 2, 4
        self.x = np.random.default_rng(20).standard_normal(
            (self.batch, 7, self.d_model)
        ).astype(np.float32)

    def test_model_chunked_prefill_matches_whole(self):
        spec = _model_spec(7)
        params = _model_params(spec)
        whole = _run(_model_graph(spec), {**params, "x": self.x})

        state = _model_zero_state(spec)
        pieces, start = [], 0
        for length in (2, 3, 2):
            part = spec.with_steps(length)
            feed = {**params, **state, "x": self.x[:, start:start + length, :]}
            res = _run(_model_stream_graph(part), feed)
            pieces.append(res["y"])
            state = spec.carry_state(res)
            start += length

        np.testing.assert_allclose(
            np.concatenate(pieces, axis=1), whole["y"], rtol=1e-5, atol=1e-6
        )

    def test_model_token_by_token_matches_prefill(self):
        spec = _model_spec(7)
        params = _model_params(spec)
        whole = _run(_model_graph(spec), {**params, "x": self.x})

        decode = _model_stream_graph(spec.with_steps(1))
        state = _model_zero_state(spec)
        rows = []
        for t in range(self.x.shape[1]):
            feed = {**params, **state, "x": self.x[:, t:t + 1, :]}
            res = _run(decode, feed)
            rows.append(res["y"])
            state = spec.carry_state(res)

        np.testing.assert_allclose(
            np.concatenate(rows, axis=1), whole["y"], rtol=1e-5, atol=1e-6
        )

    def test_model_reset_matches_separate_runs(self):
        spec = _model_spec(6)
        params = _model_params(spec)
        reset = np.zeros((self.batch, 6), dtype=np.float32)
        reset[:, 3] = 1.0
        x = self.x[:, :6, :]

        got = _run(
            _model_stream_graph(spec, with_reset=True),
            {**params, **_model_zero_state(spec), "x": x, "reset": reset},
        )

        half = spec.with_steps(3)
        first = _run(
            _model_stream_graph(half),
            {**params, **_model_zero_state(half), "x": x[:, :3, :]},
        )
        second = _run(
            _model_stream_graph(half),
            {**params, **_model_zero_state(half), "x": x[:, 3:, :]},
        )

        np.testing.assert_allclose(
            got["y"], np.concatenate([first["y"], second["y"]], axis=1),
            rtol=1e-5, atol=1e-6,
        )
        carry_got = spec.carry_state(got)
        carry_second = spec.carry_state(second)
        for name in spec.state_shapes:
            np.testing.assert_allclose(
                carry_got[name], carry_second[name], rtol=1e-6, atol=1e-7
            )

    def test_model_zero_state_reproduces_stateless(self):
        """零初值的流式图必须与训练口径逐位一致——同一份层主体。"""
        spec = _model_spec(5)
        feed = {**_model_params(spec), "x": self.x[:, :5, :]}
        whole = _run(_model_graph(spec), feed)
        stream = _run(_model_stream_graph(spec), {**feed, **_model_zero_state(spec)})
        np.testing.assert_array_equal(stream["y"], whole["y"])

    def test_single_layer_matches_the_block(self):
        """``num_layers=1`` 与 ``build_mamba_block`` 是同一张图，只是参数名有前缀。"""
        spec = _model_spec(5, num_layers=1)
        params = _params(spec.layer, seed=3)
        x = self.x[:, :5, :]

        g1 = Graph("mamba_one_block")
        _, y1 = build_mamba_block(g1, spec.layer)
        g1.output(y1, "y")
        block = _run(g1, {**params, "x": x})["y"]

        g2 = Graph("mamba_one_model")
        _, y2 = build_mamba_model(g2, spec)
        g2.output(y2, "y")
        model = _run(g2, {**{f"l0_{k}": v for k, v in params.items()}, "x": x})["y"]

        np.testing.assert_array_equal(block, model)

    def test_model_state_shapes_and_with_steps(self):
        spec = _model_spec(3)
        self.assertEqual(
            set(spec.state_shapes),
            {"l0_conv0", "l0_h0", "l1_conv0", "l1_h0"},
        )
        self.assertEqual(spec.state_shapes["l1_conv0"], (self.batch, 2, spec.layer.d_inner))
        self.assertEqual(spec.state_shapes["l0_h0"], (self.batch, spec.layer.d_inner, 3))
        # with_steps 只改长度，其余字段原样保留
        single = spec.with_steps(1)
        self.assertEqual(single.steps, 1)
        self.assertEqual(single.num_layers, spec.num_layers)
        self.assertEqual(single.param_shapes, spec.param_shapes)
        self.assertEqual(single.state_shapes, spec.state_shapes)

    def test_left_padding_matches_per_sample_runs(self):
        """混合有效长度：left padding + 起点 reset 等价于各自单独跑。

        这是阶段 B 验收表里的 padding 项。left padding 把短样本的真实 token 靠右
        放，并在有效区间起点打 reset：末状态因此停在有效前缀末，而不是被 pad 段
        拖走。right padding 做不到——pad 会继续推进状态，末状态对不上。
        """
        steps, lengths = 7, (7, 4, 2)
        batch = len(lengths)
        spec = _model_spec(steps, batch=batch, d_model=self.d_model)
        params = _model_params(spec)
        tokens = np.random.default_rng(31).standard_normal(
            (batch, steps, self.d_model)
        ).astype(np.float32)

        padded = np.zeros_like(tokens)
        reset = np.zeros((batch, steps), dtype=np.float32)
        for i, n in enumerate(lengths):
            padded[i, steps - n:] = tokens[i, :n]
            if n < steps:
                reset[i, steps - n] = 1.0

        got = _run(
            _model_stream_graph(spec, with_reset=True),
            {**params, **_model_zero_state(spec), "x": padded, "reset": reset},
        )

        for i, n in enumerate(lengths):
            one = _model_spec(n, batch=1, d_model=self.d_model)
            single = _run(
                _model_stream_graph(one),
                {**params, **_model_zero_state(one), "x": tokens[i:i + 1, :n]},
            )
            np.testing.assert_allclose(
                got["y"][i, steps - n:], single["y"][0], rtol=1e-5, atol=1e-6,
                err_msg=f"样本 {i}（有效长度 {n}）的输出与单独跑不一致",
            )
            for name in spec.state_shapes:
                np.testing.assert_allclose(
                    got[name[:-1]][i], single[name[:-1]][0], rtol=1e-5, atol=1e-6,
                    err_msg=f"样本 {i} 的 {name} 与单独跑不一致",
                )

    def test_reset_positions_are_per_sample(self):
        """batch 内各样本在不同位置 reset：谁也不能影响谁（会话隔离）。"""
        steps, batch = 6, 2
        spec = _model_spec(steps, batch=batch, d_model=self.d_model)
        params = _model_params(spec)
        x = np.random.default_rng(32).standard_normal(
            (batch, steps, self.d_model)
        ).astype(np.float32)
        reset = np.zeros((batch, steps), dtype=np.float32)
        reset[0, 3] = 1.0  # 样本 0 在 t=3 重开一段；样本 1 全程连续

        got = _run(
            _model_stream_graph(spec, with_reset=True),
            {**params, **_model_zero_state(spec), "x": x, "reset": reset},
        )

        for i, cut in ((0, 3), (1, 0)):
            one = _model_spec(steps - cut, batch=1, d_model=self.d_model)
            single = _run(
                _model_stream_graph(one),
                {**params, **_model_zero_state(one), "x": x[i:i + 1, cut:]},
            )
            np.testing.assert_allclose(
                got["y"][i, cut:], single["y"][0], rtol=1e-5, atol=1e-6,
                err_msg=f"样本 {i}（切点 {cut}）与单独跑不一致",
            )
            for name in spec.state_shapes:
                np.testing.assert_allclose(
                    got[name[:-1]][i], single[name[:-1]][0], rtol=1e-5, atol=1e-6,
                    err_msg=f"样本 {i} 的 {name} 与单独跑不一致",
                )


class TestStreamCheckpoint(unittest.TestCase):
    """状态与权重同源存档：恢复后继续 decode 必须与从未中断逐位一致。

    这条判据比"能读回数组"强得多：存档要真的把**每一层**的卷积窗口与 SSM 状态
    都带上，且加载后写回的必须是同一份数值。任何一层状态丢了或错位，逐 token
    的输出在第一帧就会偏离，而不是在若干帧后慢慢漂。
    """

    def setUp(self):
        self.batch, self.d_model, self.steps = 2, 4, 7
        self.x = np.random.default_rng(30).standard_normal(
            (self.batch, self.steps, self.d_model)
        ).astype(np.float32)
        self.spec = _model_spec(self.steps, batch=self.batch, d_model=self.d_model)
        self.params = _model_params(self.spec)

    def _prefill(self, length):
        part = self.spec.with_steps(length)
        return _run(
            _model_stream_graph(part),
            {**self.params, **_model_zero_state(part), "x": self.x[:, :length, :]},
        )

    def _decode(self, weights, state, start):
        """从 ``start`` 起逐 token 解码，返回每一步的输出。"""
        graph = _model_stream_graph(self.spec.with_steps(1))
        rows = []
        for t in range(start, self.steps):
            res = _run(graph, {**weights, **state, "x": self.x[:, t:t + 1, :]})
            rows.append(res["y"])
            state = self.spec.carry_state(res)
        return rows

    def test_resume_reproduces_uninterrupted_decode(self):
        cut = 3
        prefill = self._prefill(cut)
        state = {
            name: np.array(value, copy=True)
            for name, value in self.spec.carry_state(prefill).items()
        }

        # 不中断：prefill 之后直接接着 decode
        straight = self._decode(self.params, state, cut)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "session.npz")
            save_checkpoint(
                path, ParameterStore(self.params), meta={"tokens": cut}, state=state
            )

            # 模拟新进程：权重全错、状态全零，只有存档能把它们救回来
            fresh = ParameterStore(
                {name: np.zeros_like(arr) for name, arr in self.params.items()}
            )
            template = {
                name: np.zeros(shape, np.float32)
                for name, shape in self.spec.state_shapes.items()
            }
            meta = load_checkpoint(path, fresh, state=template)
            resumed = self._decode(fresh.arrays(), template, cut)

        self.assertEqual(meta, {"tokens": cut})
        self.assertEqual(len(straight), self.steps - cut)
        for step, (a, b) in enumerate(zip(straight, resumed)):
            np.testing.assert_array_equal(
                a, b, err_msg=f"恢复后第 {step} 个 decode 步与不中断不一致"
            )

        # 同时也与一次整段前向对齐（逐 token 与 prefill 只允许浮点重排的差异）
        whole = _run(_model_graph(self.spec), {**self.params, "x": self.x})
        np.testing.assert_allclose(
            np.concatenate(resumed, axis=1), whole["y"][:, cut:, :],
            rtol=1e-5, atol=1e-6,
        )

    def test_checkpoint_without_state_only_loads_weights(self):
        """只存权重（旧格式）仍是最常见的用法：不传 state 就能读。"""
        prefill = self._prefill(self.steps)
        state = self.spec.carry_state(prefill)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "weights.npz")
            save_checkpoint(path, ParameterStore(self.params))
            fresh = ParameterStore(
                {name: np.zeros_like(arr) for name, arr in self.params.items()}
            )
            load_checkpoint(path, fresh)
            for name in self.params:
                np.testing.assert_array_equal(fresh[name], self.params[name])
            # 存档里没有状态，硬要续接就必须失败（否则会静默从零开始）
            with self.assertRaises(ValueError):
                load_checkpoint(
                    path, fresh,
                    state={k: np.array(v, copy=True) for k, v in state.items()},
                )


# --------------------------------------------------------------------------
# 审计 AF-27：T 增长时物化了什么、流式为什么有界
# --------------------------------------------------------------------------


class TestScanFootprint(unittest.TestCase):
    """节点数不随 T 变，物化量随 T 线性——两件事都必须用数说清。

    ``plan_memory`` 是编译期预算（不执行、不需要 GPU），因此这条用例能在任何
    机器上复现审计的量化口径；它给的是下界，真实后端的 arena 还会多出对齐与
    临时区（5.36 的实测差 8192 / 6656 B）。
    """

    dims, states, batch = 32, 8, 1
    steps = (64, 128, 256, 512)

    def _graph(self, steps):
        g = Graph("scan_budget")
        x = g.input("x", f32[steps, self.batch, self.dims])
        dt = g.input("dt", f32[steps, self.batch, self.dims])
        A = g.input("A", f32[self.dims, self.states])
        B = g.input("B", f32[steps, self.batch, self.states])
        C = g.input("C", f32[steps, self.batch, self.states])
        D = g.input("D", f32[self.dims])
        g.output(build_selective_scan(g, x, dt, A, B, C, D_skip=D), "y")
        return g

    def test_node_count_is_independent_of_steps(self):
        self.assertEqual({len(self._graph(t).nodes) for t in self.steps}, {14})

    def test_intermediate_bytes_grow_linearly(self):
        budgets = {t: plan_memory(self._graph(t)) for t in self.steps}
        for attr in ("intermediate_total_bytes", "intermediate_peak_bytes"):
            per_token = {
                (getattr(budgets[2 * t], attr) - getattr(budgets[t], attr)) / t
                for t in self.steps[:-1]
            }
            self.assertEqual(len(per_token), 1, f"{attr} 不是 T 的线性函数：{per_token}")
            self.assertGreater(per_token.pop(), 0)

    def test_streaming_state_is_independent_of_history(self):
        """续接只需一个常量大小的状态：与已经处理了多少 token 无关。

        每层 = 卷积窗口 ``B*(K-1)*D_inner`` + SSM 状态 ``B*D_inner*N``（f32）。
        这正是"按 chunk 跑 prefill"能把峰值压成常数的原因：图里只有当前 chunk
        的 T 维中间值，历史缩成了这两个张量。
        """
        spec = MambaModelSpec(
            d_model=self.dims, d_state=self.states, d_conv=4, expand=2,
            steps=1, batch=self.batch, num_layers=2,
        )

        def size(s):
            return sum(int(np.prod(shape)) * 4 for shape in s.state_shapes.values())

        per_layer = (
            self.batch * (4 - 1) * spec.layer.d_inner
            + self.batch * spec.layer.d_inner * self.states
        ) * 4
        self.assertEqual(size(spec), spec.num_layers * per_layer)
        # 同一个模型处理 4096 个 token 与处理 1 个 token，状态大小完全相同
        self.assertEqual(size(spec.with_steps(4096)), size(spec))


@unittest.skipUnless(CUDA_AVAILABLE, "需要 CUDA")
class TestMambaStreamingCuda(unittest.TestCase):
    """CUDA 上跑同一组恒等式：concat / step / scan_linear 的组合必须真的可用。

    参考后端能算不代表后端能算——``reset`` 依赖 ``step``、分段依赖 ``concat``，
    这两条路径此前没有在设备上被任何模型构件走过。
    """

    def test_token_by_token_matches_prefill(self):
        cuda = CudaBackend()
        batch, d_model, steps = 2, 4, 5
        x = np.random.default_rng(11).standard_normal(
            (batch, steps, d_model)
        ).astype(np.float32)

        spec = _spec(steps, batch=batch, d_model=d_model)
        params = _params(spec)
        whole = np.asarray(cuda.execute(_whole_graph(spec), {**params, "x": x})["y"])

        state = _zero_state(spec)
        rows = []
        for t in range(steps):
            part = _spec(1, batch=batch, d_model=d_model)
            res = cuda.execute(
                _stateful_graph(part),
                {**params, **state, "x": x[:, t:t + 1, :]},
            )
            rows.append(np.asarray(res["y"]))
            state = {"conv0": np.asarray(res["conv"]), "h0": np.asarray(res["h"])}

        np.testing.assert_allclose(
            np.concatenate(rows, axis=1), whole, rtol=1e-4, atol=1e-5
        )

    def test_reset_matches_separate_runs_on_device(self):
        cuda = CudaBackend()
        batch, d_model = 2, 4
        x = np.random.default_rng(12).standard_normal(
            (batch, 6, d_model)
        ).astype(np.float32)
        spec = _spec(6, batch=batch, d_model=d_model)
        params = _params(spec)
        reset = np.zeros((batch, 6), dtype=np.float32)
        reset[:, 3] = 1.0

        got = cuda.execute(
            _stateful_graph(spec, with_reset=True),
            {**params, **_zero_state(spec), "x": x, "reset": reset},
        )

        half = _spec(3, batch=batch, d_model=d_model)
        first = cuda.execute(
            _stateful_graph(half),
            {**params, **_zero_state(half), "x": x[:, :3, :]},
        )
        second = cuda.execute(
            _stateful_graph(half),
            {**params, **_zero_state(half), "x": x[:, 3:, :]},
        )
        np.testing.assert_allclose(
            np.asarray(got["y"]),
            np.concatenate([np.asarray(first["y"]), np.asarray(second["y"])], axis=1),
            rtol=1e-4, atol=1e-5,
        )

    def test_model_token_by_token_matches_prefill_on_device(self):
        """整栈（2 层）在设备上逐 token 与整段等价：每层状态都要真的接上。"""
        cuda = CudaBackend()
        batch, d_model, steps = 2, 4, 5
        x = np.random.default_rng(13).standard_normal(
            (batch, steps, d_model)
        ).astype(np.float32)

        spec = _model_spec(steps, batch=batch, d_model=d_model)
        params = _model_params(spec)
        whole = np.asarray(cuda.execute(_model_graph(spec), {**params, "x": x})["y"])

        decode = _model_stream_graph(spec.with_steps(1))
        state = _model_zero_state(spec)
        rows = []
        for t in range(steps):
            res = cuda.execute(
                decode, {**params, **state, "x": x[:, t:t + 1, :]}
            )
            rows.append(np.asarray(res["y"]))
            state = spec.carry_state({k: np.asarray(v) for k, v in res.items()})

        np.testing.assert_allclose(
            np.concatenate(rows, axis=1), whole, rtol=1e-4, atol=1e-5
        )

    def test_long_running_decode_keeps_memory_flat(self):
        """长时间解码：预热后活跃显存不随步数增长。

        审计要求的"固定 batch / 状态连续解码，预热后记录 VRAM"。判据是
        cupy 默认池的 ``used_bytes()``（当前被张量占住的字节），不是
        ``total_bytes()``（池从驱动要到的量，扩容后不会归还）。前者涨说明
        每一步都在留下新张量：要么历史被物化进图，要么临时量没被复用。
        """
        cuda = CudaBackend()
        batch, d_model = 2, 8
        spec = _model_spec(1, batch=batch, d_model=d_model)
        params = _model_params(spec)
        graph = _model_stream_graph(spec)
        token = np.random.default_rng(33).standard_normal(
            (batch, 1, d_model)
        ).astype(np.float32)
        state = _model_zero_state(spec)

        def step():
            nonlocal state
            res = cuda.execute(graph, {**params, **state, "x": token})
            state = spec.carry_state({k: np.asarray(v) for k, v in res.items()})

        for _ in range(10):  # 预热：编译、池扩容、形状稳定
            step()
        pool = cuda.cp.get_default_memory_pool()
        pool.free_all_blocks()
        before = pool.used_bytes()
        for _ in range(200):
            step()
        after = pool.used_bytes()

        self.assertLessEqual(
            after, before + (4 << 20),
            f"200 步 decode 后活跃显存 {after} B、起点 {before} B，增量超过 4 MiB",
        )


if __name__ == "__main__":
    unittest.main()
