"""模型构件的验收测试。

每个构件都要同时满足四件事，缺一不可：

1. **IR 合规**——``verify`` 与 ``verify_portable_only`` 都没有 error；
2. **语义正确**——参考后端的输出必须与一份独立的 numpy 实现数值一致。
   只用"形状对"或"跑得通"验收，会把"图接错了但形状恰好对"这类错误
   放过去（残差接错位置、掩码加在 softmax 之后、门控顺序搞反都属于此类）；
3. **可训练**——``backward`` 能展开反向图，梯度有限；小模型上再用中心差分
   抽查解析梯度；
4. **可序列化**——JSON 往返后执行结果逐位相同。

numpy 复算刻意写成与构件代码不同风格（显式循环、逐门计算），避免把同一份
错误抄两遍。模型图一律是 f32，所以对拍在 float64 里算完再比较，容差取
1e-4 量级；差分测试的噪声也随之放大，容差单独放宽。
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from aetherflow import (
    Graph,
    ReferenceBackend,
    backward,
    dumps,
    f32,
    loads,
    verify,
    verify_portable_only,
)
from aetherflow.models import (
    CNNSpec,
    MLPSpec,
    MambaModelSpec,
    MambaSpec,
    RNNSpec,
    SNNSpec,
    TransformerSpec,
    build_cnn,
    build_mamba_block,
    build_mamba_model,
    build_mlp,
    build_rnn,
    build_snn_network,
    build_transformer,
    causal_mask_bias,
    decay_diagnostics,
    effective_half_lives,
    init_mamba_params,
)
from aetherflow.models.mamba import _softplus

# 数据输入的名字：它们不是参数，训练时要从 ParameterStore 里排除。
_DATA_INPUTS = ("x", "ids", "pos", "mask")


# --------------------------------------------------------------------------
# 通用工具
# --------------------------------------------------------------------------


def _feed(inputs, seed: int = 0, overrides=None):
    """按输入类型造一组确定的喂入数据。"""
    rng = np.random.default_rng(seed)
    feed = {}
    for name, value in inputs.items():
        shape = tuple(int(d) for d in value.type.shape)
        dtype = value.type.dtype.value
        if dtype in ("i32", "i8"):
            feed[name] = rng.integers(0, 4, size=shape, dtype=np.int32)
        elif dtype == "bool":
            feed[name] = rng.integers(0, 2, size=shape).astype(bool)
        else:
            feed[name] = (rng.standard_normal(shape) * 0.5).astype(np.float32)
    feed.update(overrides or {})
    return feed


def _run(graph, feed, alias: str = "y"):
    return ReferenceBackend().execute(graph, feed)[alias]


def _assert_ir_ok(test: unittest.TestCase, graph) -> None:
    test.assertEqual(verify(graph), [])
    test.assertEqual(verify_portable_only(graph), [])


def _wrt_of(inputs):
    """可训练输入（除数据外的全部输入）。"""
    return [v for name, v in inputs.items() if name not in _DATA_INPUTS]


def _grads(graph, inputs, feed, seed_name: str = "dy"):
    """展开反向图并执行，返回 ``{参数名: 梯度}``。"""
    wrt = _wrt_of(inputs)
    bwd = backward(graph, wrt=wrt)
    out_shape = tuple(int(d) for d in graph.outputs[0].type.shape)
    bfeed = dict(feed)
    bfeed[seed_name] = np.ones(out_shape, dtype=np.float32)
    res = ReferenceBackend().execute(bwd, bfeed)
    names = {v.name for v in wrt}
    return {k[1:]: v for k, v in res.items() if k.startswith("d") and k[1:] in names}, bwd


def _numeric_grad(graph, feed, param: str, eps: float = 1e-3) -> np.ndarray:
    """对单个参数做中心差分（图输出元素求和作为标量目标）。"""
    arr = feed[param]
    flat = arr.reshape(-1)
    grad = np.zeros(flat.size, dtype=np.float64)
    for i in range(flat.size):
        orig = float(flat[i])
        flat[i] = orig + eps
        hi = float(np.sum(_run(graph, feed), dtype=np.float64))
        flat[i] = orig - eps
        lo = float(np.sum(_run(graph, feed), dtype=np.float64))
        flat[i] = orig
        grad[i] = (hi - lo) / (2 * eps)
    return grad.reshape(arr.shape)


def _assert_roundtrip(test: unittest.TestCase, graph, feed, alias: str = "y") -> None:
    restored = loads(dumps(graph))
    test.assertEqual(restored.op_names(), graph.op_names())
    test.assertEqual(restored.output_names, graph.output_names)
    np.testing.assert_array_equal(_run(restored, feed, alias), _run(graph, feed, alias))


def _relu(x):
    return np.maximum(x, 0.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x):
    return x * _sigmoid(x)


def _gelu(x):
    # 与 ops.gelu 相同的 tanh 近似
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _layernorm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def _rmsnorm(x, gamma, eps=1e-6):
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps) * gamma


# --------------------------------------------------------------------------
# MLP
# --------------------------------------------------------------------------


class TestMLP(unittest.TestCase):
    def _build(self, spec=None):
        spec = spec or MLPSpec(sizes=(3, 5, 2), activation="gelu", batch=4)
        g = Graph("mlp")
        inputs, out = build_mlp(g, spec)
        g.output(out, "y")
        return g, inputs, spec

    def test_ir_is_portable(self):
        g, _, _ = self._build()
        _assert_ir_ok(self, g)

    def test_matches_numpy(self):
        g, inputs, spec = self._build()
        feed = _feed(inputs, seed=1)
        h = feed["x"]
        last = len(spec.sizes) - 2
        for i in range(last + 1):
            h = h @ feed[f"w{i}"] + feed[f"b{i}"]
            if i < last:
                h = _gelu(h)
        np.testing.assert_allclose(_run(g, feed), h, rtol=1e-5, atol=1e-5)

    def test_activation_variants(self):
        for act, fn in (("relu", _relu), ("silu", _silu), ("tanh", np.tanh)):
            g, inputs, spec = self._build(
                MLPSpec(sizes=(3, 4, 2), activation=act, output_activation="tanh", batch=2)
            )
            _assert_ir_ok(self, g)
            feed = _feed(inputs, seed=2)
            h = fn(feed["x"] @ feed["w0"] + feed["b0"])
            h = np.tanh(h @ feed["w1"] + feed["b1"])
            np.testing.assert_allclose(_run(g, feed), h, rtol=1e-5, atol=1e-5)

    def test_gradient_matches_numeric(self):
        g, inputs, _ = self._build(MLPSpec(sizes=(2, 3, 1), batch=2))
        feed = _feed(inputs, seed=3)
        analytic, bwd = _grads(g, inputs, feed)
        _assert_ir_ok(self, bwd)
        for name in ("w0", "b0", "w1", "b1"):
            numeric = _numeric_grad(g, feed, name)
            np.testing.assert_allclose(
                analytic[name], numeric, rtol=1e-2, atol=2e-2,
                err_msg=f"参数 {name} 的解析梯度与中心差分不一致",
            )

    def test_serialization_roundtrip(self):
        g, inputs, _ = self._build()
        _assert_roundtrip(self, g, _feed(inputs, seed=4))

    def test_trains_xor_with_parameter_store(self):
        """构件必须能直接接上训练闭环：ParameterStore + backward + Adam。"""
        from aetherflow.train import Adam, ParameterStore

        spec = MLPSpec(sizes=(2, 8, 1), activation="tanh", batch=4)
        g = Graph("xor")
        inputs, out = build_mlp(g, spec)
        g.output(out, "y")

        # "x" 是数据输入，必须显式排除，否则会被当成可训练参数
        store = ParameterStore.from_graph(g, exclude=("x",), seed=0)
        self.assertNotIn("x", store)
        # 优化器只认 dict[str, ndarray]；arrays() 返回内部引用，原地更新才会生效
        opt = Adam(store.arrays(), lr=0.05)
        bwd = backward(g, wrt=[g.find(n) for n in store.names])

        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([[0], [1], [1], [0]], dtype=np.float32)
        backend = ReferenceBackend()
        loss = float("inf")
        for _ in range(1500):
            feed = store.feed({"x": x})
            pred = backend.execute(g, feed)["y"]
            diff = pred - y
            loss = float(np.mean(diff * diff))
            bfeed = dict(feed)
            bfeed["dy"] = (2.0 / y.size) * diff
            res = backend.execute(bwd, bfeed)
            opt.step({name: res["d" + name] for name in store.names})
        self.assertLess(loss, 0.05, f"XOR 未收敛，最终 loss={loss}")

    def test_invalid_spec(self):
        with self.assertRaises(ValueError):
            MLPSpec(sizes=(4,))
        with self.assertRaises(ValueError):
            MLPSpec(sizes=(4, 2), activation="swish")


# --------------------------------------------------------------------------
# CNN
# --------------------------------------------------------------------------


def _conv2d(x, w, b, stride=1, padding=1):
    n, c, h, wd = x.shape
    o, _, kh, kw = w.shape
    ho = (h + 2 * padding - kh) // stride + 1
    wo = (wd + 2 * padding - kw) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out = np.zeros((n, o, ho, wo), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            patch = xp[:, :, i : i + stride * (ho - 1) + 1 : stride,
                       j : j + stride * (wo - 1) + 1 : stride]
            out += np.einsum("nchw,oc->nohw", patch, w[:, :, i, j])
    return out + b.reshape(1, -1, 1, 1)


def _maxpool2(x):
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


def _batch_norm(x, gamma, beta, eps=1e-5):
    # 通道轴固定是 1（NCHW），与构件里的 axis=1 对应
    mean = x.mean(axis=(0, 2, 3), keepdims=True)
    var = x.var(axis=(0, 2, 3), keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma.reshape(1, -1, 1, 1) + beta.reshape(1, -1, 1, 1)


def _numpy_cnn(x, p, spec):
    def block(h, tag, w, b):
        h = _conv2d(h, w, b)
        if spec.batch_norm:
            h = _batch_norm(h, p[tag + "_bn_g"], p[tag + "_bn_b"], spec.eps)
        return _relu(h)

    h = block(x, "stem", p["stem_w"], p["stem_b"])
    for i in range(len(spec.widths) - 1):
        h = block(h, f"stage{i}", p[f"stage{i}_w"], p[f"stage{i}_b"])
        h = _maxpool2(h)
    h = h.mean(axis=(2, 3), keepdims=True)  # kernel=side、stride=1 的全局平均
    flat = h.reshape(h.shape[0], -1)
    return flat @ p["head_w"] + p["head_b"]


class TestCNN(unittest.TestCase):
    def _build(self, spec=None):
        spec = spec or CNNSpec(in_channels=1, image=8, num_classes=3, widths=(2, 4), batch=2)
        g = Graph("cnn")
        inputs, out = build_cnn(g, spec)
        g.output(out, "y")
        return g, inputs, spec

    def test_ir_is_portable(self):
        g, _, _ = self._build()
        _assert_ir_ok(self, g)

    def test_matches_numpy(self):
        g, inputs, spec = self._build()
        feed = _feed(inputs, seed=5, overrides={"x": np.random.default_rng(0).random(
            (2, 1, 8, 8)).astype(np.float32)})
        expected = _numpy_cnn(feed["x"].astype(np.float64), {k: v.astype(np.float64) for k, v in feed.items()}, spec)
        np.testing.assert_allclose(_run(g, feed), expected, rtol=1e-4, atol=1e-4)

    def test_residual_block_identity_shortcut(self):
        from aetherflow.models import residual_block
        from aetherflow.ir.types import f32

        g = Graph("res")
        x = g.input("x", f32[1, 2, 4, 4])
        w = g.input("w", f32[2, 2, 3, 3])
        b = g.input("b", f32[2])
        out = residual_block(g, x, w, b, w, b)
        g.output(out, "y")
        _assert_ir_ok(self, g)

        rng = np.random.default_rng(6)
        feed = {
            "x": rng.standard_normal((1, 2, 4, 4)).astype(np.float32),
            "w": np.zeros((2, 2, 3, 3), dtype=np.float32),
            "b": np.zeros(2, dtype=np.float32),
        }
        # 两个卷积都退化为零，残差块输出就是 relu(x)
        np.testing.assert_array_equal(_run(g, feed), _relu(feed["x"]))

    def test_residual_block_requires_projection(self):
        from aetherflow.models import residual_block
        from aetherflow.ir.types import f32

        g = Graph("bad")
        x = g.input("x", f32[1, 2, 4, 4])
        w = g.input("w", f32[4, 2, 3, 3])
        b = g.input("b", f32[4])
        with self.assertRaises(ValueError):
            residual_block(g, x, w, b, w, b)  # 通道 2 -> 4 且没有 shortcut

    def test_gradient_matches_numeric(self):
        g, inputs, _ = self._build(
            CNNSpec(in_channels=1, image=4, num_classes=2, widths=(1, 1), batch=1)
        )
        feed = _feed(inputs, seed=7, overrides={
            "x": np.zeros((1, 1, 4, 4), dtype=np.float32) + 0.3,
        })
        analytic, bwd = _grads(g, inputs, feed)
        _assert_ir_ok(self, bwd)
        numeric = _numeric_grad(g, feed, "stem_w")
        np.testing.assert_allclose(
            analytic["stem_w"], numeric, rtol=2e-2, atol=2e-2,
            err_msg="卷积权重的解析梯度与中心差分不一致",
        )

    def test_serialization_roundtrip(self):
        g, inputs, _ = self._build()
        _assert_roundtrip(self, g, _feed(inputs, seed=8))

    def test_batch_norm_matches_numpy(self):
        g, inputs, spec = self._build(
            CNNSpec(in_channels=1, image=8, num_classes=3, widths=(2, 4), batch=2, batch_norm=True)
        )
        _assert_ir_ok(self, g)
        feed = _feed(inputs, seed=21, overrides={
            "x": np.random.default_rng(1).random((2, 1, 8, 8)).astype(np.float32),
        })
        expected = _numpy_cnn(
            feed["x"].astype(np.float64),
            {k: v.astype(np.float64) for k, v in feed.items()},
            spec,
        )
        np.testing.assert_allclose(_run(g, feed), expected, rtol=1e-4, atol=1e-4)

    def test_batch_norm_adds_params(self):
        spec = CNNSpec(in_channels=1, image=8, num_classes=3, widths=(2, 4), batch=2, batch_norm=True)
        shapes = spec.param_shapes
        for name in ("stem_bn_g", "stem_bn_b", "stage0_bn_g", "stage0_bn_b"):
            self.assertIn(name, shapes)
        self.assertEqual(shapes["stage0_bn_g"], (4,))
        self.assertNotIn("stem_bn_g", CNNSpec(widths=(2, 4)).param_shapes)

    def test_batch_norm_gradients_are_finite(self):
        g, inputs, _ = self._build(
            CNNSpec(in_channels=1, image=4, num_classes=2, widths=(1, 2), batch=2, batch_norm=True)
        )
        grads, bwd = _grads(g, inputs, _feed(inputs, seed=22))
        _assert_ir_ok(self, bwd)
        self.assertIn("stem_bn_g", grads)
        for name, value in grads.items():
            self.assertTrue(np.all(np.isfinite(value)), f"{name} 梯度非有限")

    def test_dropout_is_deterministic(self):
        """掩码写在 attrs 里：同一张图同一份输入，两次执行逐位相同。"""
        g, inputs, _ = self._build(
            CNNSpec(in_channels=1, image=8, num_classes=3, widths=(2, 4), batch=2,
                    dropout=0.5, dropout_seed=99)
        )
        _assert_ir_ok(self, g)
        self.assertIn("dropout_mask", g.op_names())
        feed = _feed(inputs, seed=23)
        np.testing.assert_array_equal(_run(g, feed), _run(g, feed))

    def test_dropout_off_adds_no_node(self):
        g, _, _ = self._build()
        self.assertNotIn("dropout_mask", g.op_names())

    def test_bn_infer_matches_numpy(self):
        from aetherflow.models import bn_infer
        from aetherflow.ir.types import f32

        g = Graph("bn_infer")
        x = g.input("x", f32[2, 3, 4, 4])
        mean = g.input("mean", f32[3])
        var = g.input("var", f32[3])
        gamma = g.input("gamma", f32[3])
        beta = g.input("beta", f32[3])
        g.output(bn_infer(g, x, mean, var, gamma, beta, eps=1e-3), "y")
        _assert_ir_ok(self, g)

        rng = np.random.default_rng(24)
        feed = {
            "x": rng.standard_normal((2, 3, 4, 4)).astype(np.float32),
            "mean": rng.standard_normal(3).astype(np.float32),
            "var": (rng.random(3) + 0.5).astype(np.float32),
            "gamma": rng.standard_normal(3).astype(np.float32),
            "beta": rng.standard_normal(3).astype(np.float32),
        }
        gd = {k: v.astype(np.float64) for k, v in feed.items()}
        expected = (gd["x"] - gd["mean"].reshape(1, -1, 1, 1)) / np.sqrt(
            gd["var"].reshape(1, -1, 1, 1) + 1e-3
        ) * gd["gamma"].reshape(1, -1, 1, 1) + gd["beta"].reshape(1, -1, 1, 1)
        np.testing.assert_allclose(_run(g, feed), expected, rtol=1e-4, atol=1e-5)

    def test_invalid_dropout(self):
        with self.assertRaises(ValueError):
            CNNSpec(dropout=1.0)

    def test_invalid_spec(self):
        with self.assertRaises(ValueError):
            CNNSpec(image=7, widths=(2, 4))  # 7 不能被 2 整除


# --------------------------------------------------------------------------
# RNN / LSTM / GRU
# --------------------------------------------------------------------------


def _numpy_rnn(x, p, spec):
    seq = [x[:, t, :].astype(np.float64) for t in range(spec.steps)]
    for layer in range(spec.num_layers):
        prefix = f"l{layer}_"
        wx, wh, b = p[prefix + "wx"], p[prefix + "wh"], p[prefix + "b"]
        h = np.zeros((spec.batch, spec.hidden_size))
        c = np.zeros_like(h)
        outs = []
        for xt in seq:
            if spec.cell == "rnn":
                h = np.tanh(xt @ wx + h @ wh + b)
            elif spec.cell == "lstm":
                pre = xt @ wx + h @ wh + b
                i, f, gg, o = np.split(pre, 4, axis=1)
                c = f * c + i * np.tanh(gg)
                h = o * np.tanh(c)
            else:
                gx = xt @ wx + b
                gh = h @ wh
                gx_z, gx_r, gx_n = np.split(gx, 3, axis=1)
                gh_z, gh_r, gh_n = np.split(gh, 3, axis=1)
                z = _sigmoid(gx_z + gh_z)
                r = _sigmoid(gx_r + gh_r)
                n = np.tanh(gx_n + r * gh_n)
                h = (1.0 - z) * n + z * h
            outs.append(h)
        seq = outs
    return seq[-1]


class TestRNN(unittest.TestCase):
    def _build(self, cell="lstm", layers=2):
        spec = RNNSpec(
            input_size=3, hidden_size=4, num_layers=layers, steps=4, batch=2, cell=cell
        )
        g = Graph("rnn_" + cell)
        inputs, seq, state = build_rnn(g, spec)
        g.output(seq[-1], "y")
        return g, inputs, spec, state

    def test_ir_is_portable(self):
        for cell in ("rnn", "lstm", "gru"):
            g, _, _, _ = self._build(cell)
            _assert_ir_ok(self, g)

    def test_matches_numpy(self):
        for cell in ("rnn", "lstm", "gru"):
            g, inputs, spec, _ = self._build(cell)
            feed = _feed(inputs, seed=9)
            expected = _numpy_rnn(
                feed["x"], {k: v.astype(np.float64) for k, v in feed.items()}, spec
            )
            np.testing.assert_allclose(
                _run(g, feed), expected, rtol=1e-4, atol=1e-4, err_msg=f"cell={cell}"
            )

    def test_state_dict_keys(self):
        for cell, keys in (
            ("rnn", {"l0_h", "l1_h"}),
            ("lstm", {"l0_h", "l0_c", "l1_h", "l1_c"}),
            ("gru", {"l0_h", "l1_h"}),
        ):
            _, _, _, state = self._build(cell)
            self.assertEqual(set(state), keys)

    def test_gradients_are_finite(self):
        for cell in ("rnn", "lstm", "gru"):
            g, inputs, _, _ = self._build(cell, layers=1)
            feed = _feed(inputs, seed=10)
            grads, bwd = _grads(g, inputs, feed)
            _assert_ir_ok(self, bwd)
            self.assertEqual(set(grads), set(inputs) - set(_DATA_INPUTS))
            for name, value in grads.items():
                self.assertTrue(np.all(np.isfinite(value)), f"{cell}:{name} 梯度非有限")

    def test_serialization_roundtrip(self):
        g, inputs, _, _ = self._build("gru")
        _assert_roundtrip(self, g, _feed(inputs, seed=11))

    def test_invalid_spec(self):
        with self.assertRaises(ValueError):
            RNNSpec(cell="mgru")


# --------------------------------------------------------------------------
# Mamba
# --------------------------------------------------------------------------


def _numpy_mamba(x, p, spec):
    p = {k: v.astype(np.float64) for k, v in p.items()}
    x = x.astype(np.float64)
    batch, steps, d = x.shape
    di, n, k = spec.d_inner, spec.d_state, spec.d_conv

    h = _rmsnorm(x, p["norm_g"])
    xg = h @ p["w_in"]
    x_branch, z = xg[..., :di], xg[..., di:]

    padded = np.pad(x_branch, ((0, 0), (k - 1, 0), (0, 0)))
    conv = np.zeros_like(x_branch) + p["conv_b"]
    for j in range(k):
        conv += padded[:, j : j + steps, :] * p["conv_w"][:, j].reshape(1, 1, di)
    act = _silu(conv)

    dt = np.log1p(np.exp(act * p["dt_w"] + p["dt_b"]))
    A = -np.exp(p["A_raw"])

    xt = act.transpose(1, 0, 2)          # (T, B, D)
    dtt = dt.transpose(1, 0, 2)
    Bt = xt @ p["B_w"]
    Ct = xt @ p["C_w"]

    state = np.zeros((batch, di, n))
    ys = []
    for t in range(steps):
        state = (
            np.exp(dtt[t][:, :, None] * A) * state
            + (dtt[t][:, :, None] * Bt[t][:, None, :]) * xt[t][:, :, None]
        )
        ys.append((state * Ct[t][:, None, :]).sum(axis=-1))
    y = np.stack(ys, axis=0).transpose(1, 0, 2)
    # 与 build_selective_scan 的 D_skip 跳连一致（在 (T, B, D) 域上加回）
    y = y + xt.transpose(1, 0, 2) * p["D_skip"]

    y = y * _silu(z)
    y = _rmsnorm(y, p["norm2_g"]) @ p["w_out"]
    return x + y


class TestMamba(unittest.TestCase):
    def _build(self):
        spec = MambaSpec(d_model=4, d_state=2, d_conv=3, expand=2, steps=4, batch=2)
        g = Graph("mamba")
        inputs, out = build_mamba_block(g, spec)
        g.output(out, "y")
        return g, inputs, spec

    def test_ir_is_portable(self):
        g, _, _ = self._build()
        _assert_ir_ok(self, g)

    def test_matches_numpy(self):
        g, inputs, spec = self._build()
        feed = _feed(inputs, seed=12)
        np.testing.assert_allclose(
            _run(g, feed), _numpy_mamba(feed["x"], feed, spec), rtol=1e-4, atol=1e-4
        )

    def test_depthwise_conv_is_causal(self):
        from aetherflow.ir.types import f32
        from aetherflow.models import causal_depthwise_conv1d

        g = Graph("conv1d")
        x = g.input("x", f32[2, 5, 3])
        w = g.input("w", f32[3, 3])
        b = g.input("b", f32[3])
        g.output(causal_depthwise_conv1d(g, x, w, b), "y")
        _assert_ir_ok(self, g)

        rng = np.random.default_rng(13)
        feed = {
            "x": rng.standard_normal((2, 5, 3)).astype(np.float32),
            "w": rng.standard_normal((3, 3)).astype(np.float32),
            "b": rng.standard_normal(3).astype(np.float32),
        }
        # 改未来位置不应影响过去位置
        base = _run(g, feed)
        perturbed = dict(feed)
        perturbed["x"] = feed["x"].copy()
        perturbed["x"][:, 4, :] += 10.0
        np.testing.assert_array_equal(_run(g, perturbed)[:, :4], base[:, :4])

    def test_gradients_are_finite(self):
        g, inputs, _ = self._build()
        feed = _feed(inputs, seed=14)
        grads, bwd = _grads(g, inputs, feed)
        _assert_ir_ok(self, bwd)
        for name, value in grads.items():
            self.assertTrue(np.all(np.isfinite(value)), f"{name} 梯度非有限")

    def test_serialization_roundtrip(self):
        g, inputs, _ = self._build()
        _assert_roundtrip(self, g, _feed(inputs, seed=15))

    def test_special_init_matches_param_shapes(self):
        """模型专属初始化：键/形状/类型与图输入一致，且可复现。"""
        spec = MambaSpec(d_model=4, d_state=2, d_conv=3, expand=2, steps=4, batch=2)
        first = init_mamba_params(spec, seed=0)
        again = init_mamba_params(spec, seed=0)
        other = init_mamba_params(spec, seed=1)

        self.assertEqual(set(first), set(spec.param_shapes))
        for name, shape in spec.param_shapes.items():
            self.assertEqual(first[name].shape, shape)
            self.assertEqual(first[name].dtype, np.float32)
            np.testing.assert_array_equal(first[name], again[name])

        # seed 只动随机参数；时间尺度网格是确定性的（换 seed 做消融时不漂移）
        self.assertFalse(np.array_equal(first["w_in"], other["w_in"]))
        np.testing.assert_array_equal(first["A_raw"], other["A_raw"])
        np.testing.assert_array_equal(first["dt_b"], other["dt_b"])

    def test_special_init_covers_multi_timescales(self):
        """半衰期逐单元等于对数网格：短/中/长记忆都有通道覆盖。"""
        spec = MambaSpec(d_model=4, d_state=4, d_conv=3, expand=2, steps=2, batch=1)
        params = init_mamba_params(spec, seed=3)
        lives = effective_half_lives(spec, params)

        self.assertEqual(lives.shape, (spec.d_inner, spec.d_state))
        self.assertTrue(np.all(np.isfinite(lives)))
        self.assertTrue(np.all(lives > 0.0))

        ordered = np.sort(lives.ravel())
        self.assertAlmostEqual(float(ordered[0]), 1.0, places=4)
        self.assertAlmostEqual(float(ordered[-1]), 32768.0, delta=32768.0 * 1e-5)
        # 对数等距：相邻比值的差异在 f32 存储误差之内
        ratios = ordered[1:] / ordered[:-1]
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-4)

        # 与通用初始化对比：后者所有通道的半衰期都在 1 token 附近
        from aetherflow.train import init_array

        generic = {
            name: init_array(name, shape, f32, seed=3)
            for name, shape in spec.param_shapes.items()
        }
        generic_lives = effective_half_lives(spec, generic)
        self.assertGreater(
            lives.max() / lives.min(), 100.0 * generic_lives.max() / generic_lives.min()
        )

    def test_special_init_is_trainable(self):
        """用专属初始化喂图：前向与反向都必须是有限值。"""
        g, inputs, spec = self._build()
        feed = _feed(inputs, seed=16, overrides=init_mamba_params(spec, seed=1))
        out = _run(g, feed)
        self.assertTrue(np.all(np.isfinite(out)))

        grads, bwd = _grads(g, inputs, feed)
        _assert_ir_ok(self, bwd)
        for name, value in grads.items():
            self.assertTrue(np.all(np.isfinite(value)), f"{name} 梯度非有限")

    def test_model_init_covers_every_layer(self):
        """整栈初始化：键与 param_shapes 一致，层间随机参数不同。"""
        spec = MambaModelSpec(
            d_model=4, d_state=2, d_conv=3, expand=2, steps=3, batch=2, num_layers=2
        )
        params = spec.init_params(seed=5)
        self.assertEqual(set(params), set(spec.param_shapes))
        for name, shape in spec.param_shapes.items():
            self.assertEqual(params[name].shape, shape)

        self.assertFalse(np.array_equal(params["l0_w_in"], params["l1_w_in"]))
        np.testing.assert_array_equal(params["l0_A_raw"], params["l1_A_raw"])

        g = Graph("mamba_model_init")
        inputs, _ = build_mamba_model(g, spec)
        self.assertEqual(set(inputs) - {"x"}, set(params))

    def test_invalid_spec(self):
        with self.assertRaises(ValueError):
            MambaSpec(d_conv=0)

    def _decay_params(self, dt_b=-2.0):
        """最小参数集：|A|=1，B 全一，便于手算 a 与 M。"""
        spec = MambaSpec(d_model=1, d_state=2, d_conv=1, expand=1, steps=1, batch=1)
        params = {
            "A_raw": np.zeros((1, 2), np.float32),
            "dt_w": np.zeros((1,), np.float32),
            "dt_b": np.full((1,), dt_b, np.float32),
            "B_w": np.ones((1, 2), np.float32),
        }
        return spec, params

    def test_decay_diagnostics_reports_contractive_recursion(self):
        spec = MambaSpec(d_model=4, d_state=4, d_conv=3, expand=2, steps=2, batch=1)
        params = init_mamba_params(spec, seed=2)
        report = decay_diagnostics(spec, params)

        self.assertTrue(report["contractive"])
        self.assertLess(report["q"], 1.0)
        self.assertEqual(report["nonfinite"]["a"], 0)
        # 没给 activation 就没有 M，不能凭空给界
        self.assertIsNone(report["state_bound"])
        self.assertIn("activation", report["state_bound_reason"])
        self.assertLessEqual(report["delta"]["min"], report["delta"]["p50"])
        self.assertLessEqual(report["delta"]["p50"], report["delta"]["p95"])

    def test_decay_diagnostics_bound_needs_input_amplitude(self):
        spec = MambaSpec(d_model=4, d_state=2, d_conv=3, expand=2, steps=2, batch=1)
        params = init_mamba_params(spec, seed=2)
        x = np.random.default_rng(0).standard_normal((3, spec.d_inner)).astype(np.float32)
        report = decay_diagnostics(
            spec, params, activation=x, horizon=10, state_init_norm=1.0
        )

        self.assertIsNotNone(report["injection_bound"])
        self.assertIsNotNone(report["state_bound"])
        self.assertGreaterEqual(report["state_bound"], report["injection_bound"])

    def test_decay_near_one_is_reported_not_hidden(self):
        """Δ→0 时 a→1：q 仍然 <1，但报告必须显示"几乎不衰减"。"""
        spec, params = self._decay_params(dt_b=-20.0)
        report = decay_diagnostics(spec, params)

        self.assertTrue(report["contractive"])
        self.assertGreater(report["q"], 0.999)
        self.assertGreater(report["log_a"]["max"], -1e-3)

    def test_decay_diagnostics_flags_f16_rounding(self):
        """f16 里很小的衰减会舍入成 a=1：必须数出来，而不是当作稳定。"""
        spec, params = self._decay_params(dt_b=-20.0)
        report = decay_diagnostics(spec, params, dtype="f16")

        self.assertIsNotNone(report["f16_a_rounds_to_one"])
        self.assertGreater(report["f16_a_rounds_to_one"], 0)

    def test_decay_diagnostics_reports_nonfinite(self):
        spec, params = self._decay_params()
        params["dt_w"] = np.ones((1,), np.float32)
        x = np.zeros((1, 1), np.float32)
        x[0, 0] = np.nan
        report = decay_diagnostics(spec, params, activation=x)

        self.assertGreater(report["nonfinite"]["delta"], 0)
        self.assertGreater(report["nonfinite"]["a"], 0)
        self.assertTrue(report["nonfinite"]["first"])

    def test_injection_bound_sums_over_d_inner(self):
        """审计 P1：``B_t = x·B_w`` 要先沿 d_inner 求和再乘 Δ·x。

        逐单元乘 ``B_w`` 少了一层求和；全一输入下前向的注入是
        ``log(2) * d_inner``（这里 1.3863），旧写法只报 0.6931。
        """
        spec = MambaSpec(d_model=2, d_state=2, d_conv=1, expand=1, steps=1, batch=1)
        params = {
            "A_raw": np.zeros((2, 2), np.float32),
            "dt_w": np.zeros((2,), np.float32),
            "dt_b": np.zeros((2,), np.float32),  # Δ = softplus(0) = log 2
            "B_w": np.ones((2, 2), np.float32),
        }
        report = decay_diagnostics(
            spec, params, activation=np.ones((1, 2), np.float32)
        )
        self.assertAlmostEqual(
            report["injection_bound"], math.log(2.0) * 2.0, places=6
        )
        self.assertGreater(report["state_bound"], report["injection_bound"])

    def test_nonfinite_units_are_never_reported_contractive(self):
        """审计 P1：``finite_a`` 过滤掉 NaN 后 q 仍然 < 1，旧实现照样报收缩。"""
        spec, params = self._decay_params(dt_b=-800.0)  # Δ 下溢到 0
        params["A_raw"] = np.full((1, 2), np.inf, np.float32)  # 0 * ∞ -> NaN
        with np.errstate(over="ignore", invalid="ignore"):
            report = decay_diagnostics(spec, params)

        self.assertGreater(report["nonfinite"]["a"], 0)
        self.assertFalse(report["contractive"])
        self.assertIsNone(report["state_bound"])
        self.assertIn("非有限", report["state_bound_reason"])

    def test_decay_zero_delta_is_not_contractive(self):
        """Δ 下溢到 0 时 a 恒为 1：没有收缩常数，必须明说而不是给个界。"""
        spec, params = self._decay_params(dt_b=-1000.0)
        report = decay_diagnostics(spec, params, activation=np.ones((1, 1), np.float32))

        self.assertFalse(report["contractive"])
        self.assertEqual(report["q"], 1.0)
        self.assertIsNone(report["state_bound"])
        self.assertIn("q < 1", report["state_bound_reason"])

    def test_decay_f16_drops_the_longest_channels(self):
        """初始化里最长的半衰期在 f16 下会舍入成 a=1：这是已知退化，要报告。"""
        spec = MambaSpec(d_model=8, d_state=4, d_conv=3, expand=2, steps=1, batch=1)
        params = init_mamba_params(spec, seed=0)
        report = decay_diagnostics(spec, params, dtype="f16")

        self.assertTrue(report["contractive"])
        self.assertGreater(report["f16_a_rounds_to_one"], 0)

    def test_decay_diagnostics_validates_shapes(self):
        spec, params = self._decay_params()
        with self.assertRaises(ValueError):
            decay_diagnostics(spec, params, activation=np.zeros((1, 2), np.float32))
        bad = dict(params)
        bad["A_raw"] = np.zeros((2, 2), np.float32)
        with self.assertRaises(ValueError):
            decay_diagnostics(spec, bad)


class TestSoftplus(unittest.TestCase):
    """``_softplus`` 的稳定形式：两头都不该出 inf，反向也不该出 NaN。

    原来的 ``log(1 + exp(x))`` 在 x > 88 时指数溢出（f32），dt 一旦变大
    就会把 inf 带进整个 scan。
    """

    def _values(self, xs):
        g = Graph("softplus")
        x = g.input("x", f32[len(xs)])
        g.output(_softplus(g, x), "y")
        return ReferenceBackend().execute(g, {"x": np.asarray(xs, np.float32)})["y"]

    def test_extremes_stay_finite(self):
        out = self._values([-500.0, -100.0, -88.0, 0.0, 50.0, 100.0, 500.0])
        self.assertTrue(np.all(np.isfinite(out)), out)

    def test_matches_log1p_on_the_stable_range(self):
        xs = np.array([-8.0, -3.0, -0.5, 0.0, 0.7, 4.0, 30.0], np.float32)
        ref = np.log1p(np.exp(-np.abs(xs.astype(np.float64)))) + np.maximum(xs, 0.0)
        np.testing.assert_allclose(
            self._values(xs).astype(np.float64), ref, rtol=1e-5, atol=1e-6
        )

    def test_gradient_stays_finite_at_extremes(self):
        g = Graph("softplus")
        x = g.input("x", f32[4])
        g.output(_softplus(g, x), "y")
        bwd = backward(g, wrt=[x])
        xs = np.array([-500.0, -50.0, 50.0, 500.0], np.float32)
        grads = ReferenceBackend().execute(
            bwd, {"x": xs, "dy": np.ones(4, np.float32)}
        )
        self.assertTrue(np.all(np.isfinite(grads["dx"])), grads["dx"])
        # d/dx softplus = sigmoid(x)：两端分别是 0 与 1
        np.testing.assert_allclose(grads["dx"][1], 0.0, atol=1e-6)
        np.testing.assert_allclose(grads["dx"][2], 1.0, atol=1e-6)

    def test_gradient_at_zero_is_sigmoid_of_zero(self):
        """审计 P1：``relu`` 拼接在 0 处取零子梯度，导数曾错成 0（真值 0.5）。

        ``max(x,0) + log(1+exp(-|x|))`` 在 ``x=0`` 左右两侧都"对"，但两条
        分支的导数各自取 0；写成 ``(x+|x|)/2`` 后，``x`` 的 1 与 ``|x|`` 的
        0 相加得 0.5，与解析值一致。
        """
        g = Graph("softplus")
        x = g.input("x", f32[2])
        g.output(_softplus(g, x), "y")
        bwd = backward(g, wrt=[x])
        got = ReferenceBackend().execute(
            bwd, {"x": np.zeros(2, np.float32), "dy": np.ones(2, np.float32)}
        )
        np.testing.assert_allclose(got["dx"], np.full(2, 0.5), rtol=0.0, atol=1e-6)


# --------------------------------------------------------------------------
# Transformer（decoder-only）
# --------------------------------------------------------------------------


def _numpy_transformer(x, p, spec, mask):
    p = {k: v.astype(np.float64) for k, v in p.items()}
    x = x.astype(np.float64) + p["pos"]
    d, heads, dh, seq = spec.d_model, spec.num_heads, spec.head_dim, spec.seq
    batch = x.shape[0]
    scale = 1.0 / math.sqrt(dh)

    for i in range(spec.num_layers):
        prefix = f"l{i}_"
        h1 = _layernorm(x, p[prefix + "ln1_g"], p[prefix + "ln1_b"], spec.eps)
        q = (h1 @ p[prefix + "wq"]).reshape(batch, seq, heads, dh).transpose(0, 2, 1, 3)
        k = (h1 @ p[prefix + "wk"]).reshape(batch, seq, heads, dh).transpose(0, 2, 1, 3)
        v = (h1 @ p[prefix + "wv"]).reshape(batch, seq, heads, dh).transpose(0, 2, 1, 3)
        scores = q @ k.transpose(0, 1, 3, 2)
        attn = _softmax(scores * scale + mask)
        ctx = attn @ v
        merged = ctx.transpose(0, 2, 1, 3).reshape(batch, seq, d)
        x = x + merged @ p[prefix + "wo"]

        h2 = _layernorm(x, p[prefix + "ln2_g"], p[prefix + "ln2_b"], spec.eps)
        x = x + _gelu(h2 @ p[prefix + "w1"]) @ p[prefix + "w2"]

    return _layernorm(x, p["final_g"], p["final_b"], spec.eps)


class TestTransformer(unittest.TestCase):
    def _build(self, vocab=0, layers=1, dropout=0.0, seed=0):
        spec = TransformerSpec(
            d_model=4, num_heads=2, d_ff=4, num_layers=layers, seq=3, vocab=vocab, batch=1,
            dropout=dropout, dropout_seed=seed,
        )
        g = Graph("lm")
        inputs, out = build_transformer(g, spec)
        g.output(out, "y")
        return g, inputs, spec

    def test_causal_mask_bias(self):
        bias = causal_mask_bias(3)
        self.assertEqual(bias.shape, (1, 1, 3, 3))
        # 上三角（未来）被禁止，下三角与对角线允许
        self.assertTrue(np.all(bias[0, 0][np.triu_indices(3, k=1)] <= -1e8))
        self.assertTrue(np.all(bias[0, 0][np.tril_indices(3)] == 0.0))
        with self.assertRaises(ValueError):
            causal_mask_bias(0)

    def test_ir_is_portable(self):
        for vocab in (0, 5):
            g, _, _ = self._build(vocab)
            _assert_ir_ok(self, g)

    def test_matches_numpy(self):
        g, inputs, spec = self._build(vocab=0, layers=2)
        feed = _feed(inputs, seed=16, overrides={"mask": causal_mask_bias(spec.seq)})
        expected = _numpy_transformer(feed["x"], feed, spec, feed["mask"].astype(np.float64))
        np.testing.assert_allclose(_run(g, feed), expected, rtol=1e-4, atol=1e-4)

    def test_vocab_path_uses_embedding(self):
        g, inputs, spec = self._build(vocab=5, layers=1)
        feed = _feed(inputs, seed=17, overrides={"mask": causal_mask_bias(spec.seq)})
        # 词嵌入路径：显式把 ids 换成 one-hot 查表后的隐状态再对比，最后过 LM head
        expected = _numpy_transformer(
            feed["tok_emb"][feed["ids"]], feed, spec, feed["mask"].astype(np.float64)
        ) @ feed["w_head"].astype(np.float64)
        np.testing.assert_allclose(_run(g, feed), expected, rtol=1e-4, atol=1e-4)

    def test_causality(self):
        """改最后一个位置的 token，前面位置的 logits 必须逐位不变。"""
        g, inputs, spec = self._build(vocab=5, layers=2)
        feed = _feed(inputs, seed=18, overrides={"mask": causal_mask_bias(spec.seq)})
        base = _run(g, feed)
        perturbed = dict(feed)
        perturbed["ids"] = feed["ids"].copy()
        perturbed["ids"][:, -1] = (perturbed["ids"][:, -1] + 1) % spec.vocab
        np.testing.assert_array_equal(_run(g, perturbed)[:, :-1], base[:, :-1])

    def test_gradients_are_finite(self):
        g, inputs, spec = self._build(vocab=5, layers=1)
        feed = _feed(inputs, seed=19, overrides={"mask": causal_mask_bias(spec.seq)})
        grads, bwd = _grads(g, inputs, feed)
        _assert_ir_ok(self, bwd)
        self.assertIn("tok_emb", grads)
        self.assertIn("w_head", grads)
        for name, value in grads.items():
            self.assertTrue(np.all(np.isfinite(value)), f"{name} 梯度非有限")

    def test_serialization_roundtrip(self):
        g, inputs, spec = self._build(vocab=5)
        _assert_roundtrip(self, g, _feed(inputs, seed=20, overrides={
            "mask": causal_mask_bias(spec.seq)
        }))

    def test_dropout_inserts_two_masks_per_layer(self):
        g, _, _ = self._build(layers=2, dropout=0.1)
        _assert_ir_ok(self, g)
        self.assertEqual(g.op_names().count("dropout_mask"), 4)

    def test_dropout_off_adds_no_node(self):
        g, _, _ = self._build(layers=2)
        self.assertNotIn("dropout_mask", g.op_names())

    def test_dropout_is_deterministic_and_trainable(self):
        g, inputs, spec = self._build(vocab=5, layers=2, dropout=0.2, seed=7)
        feed = _feed(inputs, seed=25, overrides={"mask": causal_mask_bias(spec.seq)})
        np.testing.assert_array_equal(_run(g, feed), _run(g, feed))
        grads, bwd = _grads(g, inputs, feed)
        _assert_ir_ok(self, bwd)
        for name, value in grads.items():
            self.assertTrue(np.all(np.isfinite(value)), f"{name} 梯度非有限")

    def test_layers_use_distinct_seeds(self):
        """层与位置之间的 seed 必须互不相同，否则整栈会复用同一张掩码。"""
        g, _, _ = self._build(layers=3, dropout=0.5, seed=5)
        seeds = [n.attrs["seed"] for n in g.nodes if n.op.name == "dropout_mask"]
        self.assertEqual(len(seeds), 6)
        self.assertEqual(len(set(seeds)), 6)

    def test_invalid_dropout(self):
        with self.assertRaises(ValueError):
            TransformerSpec(dropout=-0.1)

    def test_invalid_spec(self):
        with self.assertRaises(ValueError):
            TransformerSpec(d_model=5, num_heads=2)


# --------------------------------------------------------------------------
# SNN
# --------------------------------------------------------------------------


def _numpy_snn(x, p, spec):
    """两层/多层 LIF 的朴素展开，与构件代码逐门对齐。"""
    p = {k: v.astype(np.float64) for k, v in p.items()}
    x = x.astype(np.float64)
    batch, steps = spec.batch, spec.steps
    seq = [x[t] for t in range(steps)]

    for i, width in enumerate(spec.hidden):
        v = np.zeros((batch, width))
        spikes = []
        for xt in seq:
            pre = xt @ p[f"l{i}_w"] + p[f"l{i}_b"]
            v_pre = spec.lif.decay * v + pre
            s = (v_pre > spec.lif.threshold).astype(np.float64)
            if spec.lif.reset == "subtract":
                v = v_pre - spec.lif.threshold * s
            else:
                v = v_pre
            spikes.append(s)
        seq = spikes

    rate = sum(seq) / len(seq)
    if spec.num_classes:
        return rate @ p["head_w"] + p["head_b"]
    return rate


class TestSNNNetwork(unittest.TestCase):
    def _build(self, num_classes=0):
        spec = SNNSpec(
            input_dim=3, hidden=(4, 4), steps=3, batch=2, num_classes=num_classes
        )
        g = Graph("snn")
        inputs, out = build_snn_network(g, spec)
        g.output(out, "y")
        return g, inputs, spec

    def test_ir_is_portable(self):
        for num_classes in (0, 3):
            g, _, _ = self._build(num_classes)
            _assert_ir_ok(self, g)

    def test_matches_numpy(self):
        for num_classes in (0, 3):
            g, inputs, spec = self._build(num_classes)
            feed = _feed(inputs, seed=21, overrides={
                "x": (np.random.default_rng(1).random((3, 2, 3)) * 1.5).astype(np.float32)
            })
            np.testing.assert_allclose(
                _run(g, feed), _numpy_snn(feed["x"], feed, spec), rtol=1e-5, atol=1e-5
            )

    def test_constant_current_encode(self):
        from aetherflow.ir.types import f32
        from aetherflow.models import constant_current_encode, mean_rate

        g = Graph("enc")
        x = g.input("x", f32[2, 3])
        steps = constant_current_encode(g, x, 4)
        self.assertEqual(len(steps), 4)
        self.assertTrue(all(s is x for s in steps))
        g.output(mean_rate(g, steps), "y")
        _assert_ir_ok(self, g)
        feed = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
        np.testing.assert_array_equal(_run(g, feed), feed["x"])

    def test_gradients_are_finite(self):
        g, inputs, _ = self._build(num_classes=3)
        feed = _feed(inputs, seed=22, overrides={
            "x": (np.random.default_rng(2).random((3, 2, 3)) * 1.5).astype(np.float32)
        })
        grads, bwd = _grads(g, inputs, feed)
        _assert_ir_ok(self, bwd)
        for name, value in grads.items():
            self.assertTrue(np.all(np.isfinite(value)), f"{name} 梯度非有限")

    def test_serialization_roundtrip(self):
        g, inputs, _ = self._build(num_classes=3)
        _assert_roundtrip(self, g, _feed(inputs, seed=23))

    def test_invalid_spec(self):
        with self.assertRaises(ValueError):
            SNNSpec(hidden=())
        with self.assertRaises(ValueError):
            SNNSpec(hidden=(2,), num_classes=-1)


if __name__ == "__main__":
    unittest.main()
