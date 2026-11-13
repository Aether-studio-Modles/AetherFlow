"""SSM（selective scan）的测试。

四条判据，缺一不可：

1. ``scan_linear`` 的前向与"逐时间步朴素循环"一致——递归不能只是"能跑"；
2. ``scan_linear_backward`` 确实是前向的伴随，且是反向的唯一手段；
3. 对 a / b / h0 的梯度与中心差分一致——反向规则对不对不靠肉眼看；
4. 整张 selective scan 图（离散化 + 递归 + 读出 + 跳连）与 numpy 朴素实现、
   以及 torch 展开循环 + autograd 双向对拍。

另外钉住三件容易退化的事：符号时间维必须拒绝（递归步数要写进图里）、
长序列不能溢出、反向图仍然是可移植的普通 IR。

全程用 float64：递归把误差按步数累乘，f32 的噪声与"规则写错"很难区分。
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
    loads,
    verify,
    verify_portable_only,
)
from aetherflow.models import build_selective_scan

try:  # 内核与全部测试必须在没有 torch 的机器上也能跑（ADR-005）
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

_REF = ReferenceBackend()


def _run(graph, feed):
    return _REF.execute(graph, feed)


def _numeric_grad(graph, feed, name, weight, eps=1e-6):
    """中心差分：loss = sum(weight * y)，对 ``feed[name]`` 逐元素扰动。"""
    arr = feed[name]
    flat = arr.reshape(-1)
    grad = np.zeros(flat.size, dtype=np.float64)
    for i in range(flat.size):
        orig = float(flat[i])
        flat[i] = orig + eps
        hi = float(np.sum(_run(graph, feed)["y"] * weight))
        flat[i] = orig - eps
        lo = float(np.sum(_run(graph, feed)["y"] * weight))
        grad[i] = (hi - lo) / (2 * eps)
        flat[i] = orig
    return grad.reshape(arr.shape)


def _scan_graph(steps, tail, dtype=f64):
    g = Graph("scan")
    a = g.input("a", dtype[(steps, *tail)])
    b = g.input("b", dtype[(steps, *tail)])
    h0 = g.input("h0", dtype[tail])
    g.output(g.add_node("scan_linear", [a, b, h0]), "y")
    return g


def _naive_scan(a, b, h0):
    h = np.array(h0, copy=True)
    out = np.empty_like(a)
    for t in range(a.shape[0]):
        h = a[t] * h + b[t]
        out[t] = h
    return out


def _naive_adjoint(a, dy):
    """伴随递归：dh_t = dy_t + a_{t+1} * dh_{t+1}。"""
    out = np.empty_like(dy)
    dh = np.zeros_like(dy[0])
    for t in range(dy.shape[0] - 1, -1, -1):
        nxt = a[t + 1] * dh if t + 1 < dy.shape[0] else 0.0
        dh = dy[t] + nxt
        out[t] = dh
    return out


# --------------------------------------------------------------------------
# 算子：前向
# --------------------------------------------------------------------------


class TestScanLinearForward(unittest.TestCase):
    def test_matches_naive_loop(self):
        steps, tail = 4, (2, 3)
        rng = np.random.default_rng(0)
        a = (rng.standard_normal((steps, *tail)) * 0.5).astype(np.float64)
        b = rng.standard_normal((steps, *tail)).astype(np.float64)
        h0 = rng.standard_normal(tail).astype(np.float64)

        got = _run(_scan_graph(steps, tail), {"a": a, "b": b, "h0": h0})["y"]
        np.testing.assert_allclose(got, _naive_scan(a, b, h0), rtol=1e-12, atol=1e-12)

    def test_single_step_is_exact(self):
        """T=1 时 h = a0*h0 + b0：没有历史项，可以逐位核对。"""
        rng = np.random.default_rng(1)
        a = rng.standard_normal((1, 3)).astype(np.float64)
        b = rng.standard_normal((1, 3)).astype(np.float64)
        h0 = rng.standard_normal((3,)).astype(np.float64)

        got = _run(_scan_graph(1, (3,)), {"a": a, "b": b, "h0": h0})["y"]
        np.testing.assert_allclose(got[0], a[0] * h0 + b[0], rtol=1e-15, atol=0)
        np.testing.assert_allclose(got, _naive_scan(a, b, h0), rtol=1e-15, atol=0)

    def test_initial_state_enters_only_first_step(self):
        """h0 的影响按 prod(a) 衰减：T=3 时只通过 a1*a2 之类的乘积体现。"""
        a = np.full((3, 1), 0.5, dtype=np.float64)
        b = np.zeros((3, 1), dtype=np.float64)
        got = _run(_scan_graph(3, (1,)), {"a": a, "b": b, "h0": np.array([8.0])})["y"]
        np.testing.assert_allclose(got.ravel(), [4.0, 2.0, 1.0], rtol=1e-15)


# --------------------------------------------------------------------------
# 算子：伴随
# --------------------------------------------------------------------------


class TestScanLinearBackwardOp(unittest.TestCase):
    def _graph(self, steps, tail=(2,)):
        g = Graph("scan_adj")
        a = g.input("a", f64[(steps, *tail)])
        dy = g.input("dy", f64[(steps, *tail)])
        g.output(g.add_node("scan_linear_backward", [a, dy]), "y")
        return g

    def test_matches_naive_adjoint(self):
        steps, tail = 5, (2, 3)
        rng = np.random.default_rng(2)
        a = (rng.standard_normal((steps, *tail)) * 0.5).astype(np.float64)
        dy = rng.standard_normal((steps, *tail)).astype(np.float64)

        got = _run(self._graph(steps, tail), {"a": a, "dy": dy})["y"]
        np.testing.assert_allclose(got, _naive_adjoint(a, dy), rtol=1e-12, atol=1e-12)

    def test_single_step(self):
        a = np.array([[0.5]], dtype=np.float64)
        dy = np.array([[2.0]], dtype=np.float64)
        got = _run(self._graph(1, (1,)), {"a": a, "dy": dy})["y"]
        np.testing.assert_allclose(got, dy, rtol=1e-15)

    def test_shape_mismatch_rejected(self):
        g = Graph("scan_adj_bad")
        a = g.input("a", f64[3, 2])
        dy = g.input("dy", f64[4, 2])
        with self.assertRaises(ValueError):
            g.add_node("scan_linear_backward", [a, dy])


# --------------------------------------------------------------------------
# 算子：梯度（中心差分）
# --------------------------------------------------------------------------


class TestScanLinearGradients(unittest.TestCase):
    def _setup(self, steps=4, tail=(3,), seed=3):
        rng = np.random.default_rng(seed)
        feed = {
            "a": (rng.standard_normal((steps, *tail)) * 0.4).astype(np.float64),
            "b": rng.standard_normal((steps, *tail)).astype(np.float64),
            "h0": rng.standard_normal(tail).astype(np.float64),
        }
        g = _scan_graph(steps, tail)
        weight = rng.standard_normal((steps, *tail)).astype(np.float64)
        bwd = backward(g, wrt=[g.find("a"), g.find("b"), g.find("h0")])
        res = _run(bwd, {**feed, "dy": weight})
        return g, feed, weight, res

    def test_gradients_match_numeric(self):
        g, feed, weight, res = self._setup()
        for name in ("a", "b", "h0"):
            want = _numeric_grad(g, feed, name, weight)
            np.testing.assert_allclose(
                res[f"d{name}"], want, rtol=1e-6, atol=1e-8, err_msg=f"d{name}"
            )

    def test_dh0_is_first_adjoint(self):
        """h0 的梯度 = a_0 * 伴随递归的第一步：h_0 = a_0*h0 + b_0，别漏掉 a_0。"""
        g, feed, weight, res = self._setup()
        want = feed["a"][0] * _naive_adjoint(feed["a"], weight)[0]
        np.testing.assert_allclose(res["dh0"], want, rtol=1e-12, atol=1e-12)

    def test_da_uses_previous_state(self):
        """da_t = dh_t * h_{t-1}：T=1 时 h_{t-1} 就是 h0，可以直接手算。"""
        a = np.array([[0.7]], dtype=np.float64)
        b = np.array([[0.3]], dtype=np.float64)
        h0 = np.array([2.0], dtype=np.float64)
        dy = np.array([[1.0]], dtype=np.float64)

        g = _scan_graph(1, (1,))
        bwd = backward(g, wrt=[g.find("a")])
        got = _run(bwd, {"a": a, "b": b, "h0": h0, "dy": dy})["da"]
        self.assertAlmostEqual(float(got[0, 0]), float(dy[0, 0] * h0[0]), places=12)


# --------------------------------------------------------------------------
# 算子：契约
# --------------------------------------------------------------------------


class TestScanLinearContract(unittest.TestCase):
    def test_symbolic_time_rejected(self):
        g = Graph("scan_sym")
        a = g.input("a", f32["T", 2])
        b = g.input("b", f32["T", 2])
        h0 = g.input("h0", f32[2])
        with self.assertRaises(ValueError):
            g.add_node("scan_linear", [a, b, h0])

    def test_rank_below_two_rejected(self):
        g = Graph("scan_rank1")
        a = g.input("a", f32[3])
        b = g.input("b", f32[3])
        h0 = g.input("h0", f32[3])
        with self.assertRaises(ValueError):
            g.add_node("scan_linear", [a, b, h0])

    def test_dtype_mismatch_rejected(self):
        g = Graph("scan_dtype")
        a = g.input("a", f32[3, 2])
        b = g.input("b", f64[3, 2])
        h0 = g.input("h0", f32[2])
        with self.assertRaises(TypeError):
            g.add_node("scan_linear", [a, b, h0])

    def test_initial_state_shape_rejected(self):
        g = Graph("scan_h0")
        a = g.input("a", f32[3, 2])
        b = g.input("b", f32[3, 2])
        h0 = g.input("h0", f32[3, 2])
        with self.assertRaises(ValueError):
            g.add_node("scan_linear", [a, b, h0])

    def test_backward_graph_is_portable_and_round_trips(self):
        steps, tail = 3, (2,)
        g = _scan_graph(steps, tail)
        bwd = backward(g, wrt=[g.find("a"), g.find("b"), g.find("h0")])
        verify(bwd)
        verify_portable_only(bwd)

        bwd2 = loads(dumps(bwd))
        rng = np.random.default_rng(5)
        feed = {
            "a": (rng.standard_normal((steps, *tail)) * 0.4).astype(np.float64),
            "b": rng.standard_normal((steps, *tail)).astype(np.float64),
            "h0": rng.standard_normal(tail).astype(np.float64),
            "dy": rng.standard_normal((steps, *tail)).astype(np.float64),
        }
        for name in ("da", "db", "dh0"):
            np.testing.assert_allclose(
                _run(bwd, feed)[name], _run(bwd2, feed)[name], rtol=1e-12
            )


# --------------------------------------------------------------------------
# 模型：selective scan
# --------------------------------------------------------------------------


def _naive_selective_scan(x, dt, A, B, C, D_skip=None, h0=None):
    """逐时间步的朴素实现，作为"参考的参考"（不经过任何 IR）。"""
    steps, dims = x.shape[0], x.shape[-1]
    states = A.shape[1]
    tail = x.shape[1:-1]
    h = np.zeros((*tail, dims, states)) if h0 is None else np.array(h0, copy=True)
    out = np.empty(x.shape)
    for t in range(steps):
        a = np.exp(dt[t][..., None] * A)  # (..., D, N)
        b = (dt[t][..., None] * B[t][..., None, :]) * x[t][..., None]
        h = a * h + b
        y = (h * C[t][..., None, :]).sum(-1)
        if D_skip is not None:
            y = y + D_skip * x[t]
        out[t] = y
    return out


def _ssm_graph(shape, dims, states, skip=False, h0=False, dtype=f64):
    """shape = (T, ...)，末维由 ``dims`` 决定，前三者是时间/批维。"""
    lead = shape[:-1]
    g = Graph("ssm")
    x = g.input("x", dtype[shape])
    dt = g.input("dt", dtype[shape])
    A = g.input("A", dtype[dims, states])
    B_t = g.input("B", dtype[(*lead, states)])
    C_t = g.input("C", dtype[(*lead, states)])
    kwargs = {}
    if skip:
        kwargs["D_skip"] = g.input("Dskip", dtype[dims])
    if h0:
        kwargs["h0"] = g.input("h0", dtype[(*lead[1:], dims, states)])
    g.output(build_selective_scan(g, x, dt, A, B_t, C_t, **kwargs), "y")
    return g


def _ssm_feed(shape, dims, states, skip=False, h0=False, seed=6):
    rng = np.random.default_rng(seed)
    lead = shape[:-1]
    feed = {
        "x": rng.standard_normal(shape).astype(np.float64),
        "dt": (rng.random(shape) * 0.3 + 0.05).astype(np.float64),
        "A": -(rng.random((dims, states)) + 0.5).astype(np.float64),
        "B": rng.standard_normal((*lead, states)).astype(np.float64),
        "C": rng.standard_normal((*lead, states)).astype(np.float64),
    }
    if skip:
        feed["Dskip"] = rng.standard_normal(dims).astype(np.float64)
    if h0:
        feed["h0"] = rng.standard_normal((*lead[1:], dims, states)).astype(np.float64)
    return feed


class TestSelectiveScanForward(unittest.TestCase):
    def test_matches_numpy(self):
        shape, dims, states = (5, 3), 3, 2
        g = _ssm_graph(shape, dims, states)
        feed = _ssm_feed(shape, dims, states)

        got = _run(g, feed)["y"]
        want = _naive_selective_scan(
            feed["x"], feed["dt"], feed["A"], feed["B"], feed["C"]
        )
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)

    def test_batched_and_skip_and_initial_state(self):
        """(T, B, D) + 跳连 + 非零初始状态：4 维广播与两处可选输入的组合。"""
        shape, dims, states = (4, 2, 3), 3, 5
        g = _ssm_graph(shape, dims, states, skip=True, h0=True)
        feed = _ssm_feed(shape, dims, states, skip=True, h0=True)

        got = _run(g, feed)["y"]
        want = _naive_selective_scan(
            feed["x"], feed["dt"], feed["A"], feed["B"], feed["C"],
            D_skip=feed["Dskip"], h0=feed["h0"],
        )
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)

    def test_zero_initial_state_is_self_contained(self):
        """不给 h0 时图里必须有 constant 节点——反向图不能靠外部喂常量。"""
        shape, dims, states = (3, 2), 2, 2
        g = _ssm_graph(shape, dims, states)
        self.assertIn("constant", g.op_names())
        feed = _ssm_feed(shape, dims, states)
        self.assertEqual(set(_run(g, feed)), {"y"})

    def test_symbolic_sequence_rejected(self):
        g = Graph("ssm_sym")
        x = g.input("x", f32["T", 2])
        dt = g.input("dt", f32["T", 2])
        A = g.input("A", f32[2, 3])
        B_t = g.input("B", f32["T", 3])
        C_t = g.input("C", f32["T", 3])
        with self.assertRaises(ValueError):
            build_selective_scan(g, x, dt, A, B_t, C_t)

    def test_mismatched_dt_rejected(self):
        g = Graph("ssm_bad_dt")
        x = g.input("x", f32[4, 2])
        dt = g.input("dt", f32[5, 2])
        A = g.input("A", f32[2, 3])
        B_t = g.input("B", f32[4, 3])
        C_t = g.input("C", f32[4, 3])
        with self.assertRaises(ValueError):
            build_selective_scan(g, x, dt, A, B_t, C_t)

    def test_long_sequence_stays_finite(self):
        """T=64：a_t = exp(dt*A) 恒在 (0,1)，递归不放大；溢出说明公式写反了。"""
        shape, dims, states = (64, 3), 3, 4
        g = _ssm_graph(shape, dims, states)
        feed = _ssm_feed(shape, dims, states)
        got = _run(g, feed)["y"]
        self.assertTrue(np.isfinite(got).all())
        want = _naive_selective_scan(
            feed["x"], feed["dt"], feed["A"], feed["B"], feed["C"]
        )
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-10)


class TestSelectiveScanBackward(unittest.TestCase):
    def _names(self):
        return ("x", "dt", "A", "B", "C")

    def test_gradients_match_numeric(self):
        shape, dims, states = (4, 2), 2, 2
        g = _ssm_graph(shape, dims, states)
        feed = _ssm_feed(shape, dims, states)
        weight = np.random.default_rng(7).standard_normal(shape)

        bwd = backward(g, wrt=[g.find(n) for n in self._names()])
        res = _run(bwd, {**feed, "dy": weight})
        for name in self._names():
            want = _numeric_grad(g, feed, name, weight)
            np.testing.assert_allclose(
                res[f"d{name}"], want, rtol=1e-5, atol=1e-7, err_msg=f"d{name}"
            )

    def test_skip_and_initial_state_gradients(self):
        shape, dims, states = (3, 2, 3), 3, 2
        g = _ssm_graph(shape, dims, states, skip=True, h0=True)
        feed = _ssm_feed(shape, dims, states, skip=True, h0=True)
        weight = np.random.default_rng(8).standard_normal(shape)

        bwd = backward(g, wrt=[g.find(n) for n in (*self._names(), "Dskip", "h0")])
        res = _run(bwd, {**feed, "dy": weight})
        for name in ("Dskip", "h0"):
            want = _numeric_grad(g, feed, name, weight)
            np.testing.assert_allclose(
                res[f"d{name}"], want, rtol=1e-5, atol=1e-7, err_msg=f"d{name}"
            )

    def test_backward_graph_is_portable_and_round_trips(self):
        shape, dims, states = (4, 2), 2, 2
        g = _ssm_graph(shape, dims, states)
        bwd = backward(g, wrt=[g.find(n) for n in self._names()])
        verify(bwd)
        verify_portable_only(bwd)

        bwd2 = loads(dumps(bwd))
        feed = _ssm_feed(shape, dims, states)
        feed["dy"] = np.random.default_rng(9).standard_normal(shape)
        for name in ("dx", "ddt", "dA", "dB", "dC"):
            np.testing.assert_allclose(
                _run(bwd, feed)[name], _run(bwd2, feed)[name], rtol=1e-12
            )

    @unittest.skipUnless(TORCH_AVAILABLE, "需要 torch")
    def test_gradients_match_torch_unroll(self):
        """torch 显式展开时间循环 + autograd：同一个数学，两条独立实现。"""
        shape, dims, states = (5, 3), 3, 2
        g = _ssm_graph(shape, dims, states, skip=True)
        feed = _ssm_feed(shape, dims, states, skip=True, seed=10)
        weight = np.random.default_rng(11).standard_normal(shape)

        bwd = backward(g, wrt=[g.find(n) for n in (*self._names(), "Dskip")])
        got = _run(bwd, {**feed, "dy": weight})

        tx = torch.tensor(feed["x"], requires_grad=True)
        tdt = torch.tensor(feed["dt"], requires_grad=True)
        tA = torch.tensor(feed["A"], requires_grad=True)
        tB = torch.tensor(feed["B"], requires_grad=True)
        tC = torch.tensor(feed["C"], requires_grad=True)
        tD = torch.tensor(feed["Dskip"], requires_grad=True)

        steps = shape[0]
        h = torch.zeros(dims, states, dtype=torch.float64)
        ys = []
        for t in range(steps):
            a = torch.exp(tdt[t][:, None] * tA)
            b = (tdt[t][:, None] * tB[t][None, :]) * tx[t][:, None]
            h = a * h + b
            ys.append((h * tC[t][None, :]).sum(-1) + tD * tx[t])
        ty = torch.stack(ys)
        (ty * torch.tensor(weight)).sum().backward()

        pairs = {
            "x": (got["dx"], tx.grad),
            "dt": (got["ddt"], tdt.grad),
            "A": (got["dA"], tA.grad),
            "B": (got["dB"], tB.grad),
            "C": (got["dC"], tC.grad),
            "Dskip": (got["dDskip"], tD.grad),
        }
        for name, (mine, theirs) in pairs.items():
            self.assertIsNotNone(theirs, name)
            np.testing.assert_allclose(
                mine, theirs.detach().numpy(), rtol=1e-6, atol=1e-8, err_msg=name
            )


if __name__ == "__main__":
    unittest.main()
