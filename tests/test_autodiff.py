"""自动微分的验收测试。

三条独立的判据，缺一不可：

1. **数值梯度**（中心差分）——不依赖任何外部框架，覆盖每个算子的 vjp 规则；
2. **torch.autograd 对拍**——只作为"另一个实现"来交叉验证，不是运行期依赖
   （ADR-008 的例外条款：torch 可以出现在测试对照里）；
3. **反向图本身合规**——它必须能过 verifier、能 JSON 往返、能在参考后端跑，
   这样才能支撑"反向图也是普通 IR"这条设计。

用 float64 做对拍：差分噪声随精度平方增长，f32 的噪声与某些规则的量级接近，
会把"规则写错"和"精度不够"混在一起。
"""

from __future__ import annotations

import unittest

import numpy as np

from aetherflow import (
    Graph,
    NoGradientError,
    ReferenceBackend,
    backward,
    dumps,
    f32,
    f64,
    hessian_vector_product,
    loads,
    verify,
)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # 内核与全部测试必须在没有 torch 的机器上也能跑（ADR-005）
    TORCH_AVAILABLE = False


# --------------------------------------------------------------------------
# 通用对拍工具
# --------------------------------------------------------------------------


def _execute(graph, feed):
    return ReferenceBackend().execute(graph, feed)


def _scalar_loss(graph, feed) -> float:
    return float(sum(float(np.sum(v)) for v in _execute(graph, feed).values()))


def _numeric_grad(graph, feed, name: str, eps: float = 1e-6) -> np.ndarray:
    """中心差分：对 ``feed[name]`` 的每个元素单独扰动。

    先复制成 ndarray 再放回 ``feed``：标量输入（``shape=()``）在某些写法下
    是 numpy 标量，``reshape(-1)`` 只是视图的临时桥接，原地扰动改不到
    ``feed``，数值梯度会静默变成 0——测试会误判实现有问题。
    """
    arr = np.array(feed[name], copy=True)
    feed[name] = arr
    flat = arr.reshape(-1)
    grad = np.zeros(flat.size, dtype=np.float64)
    for i in range(flat.size):
        orig = float(flat[i])
        flat[i] = orig + eps
        hi = _scalar_loss(graph, feed)
        flat[i] = orig - eps
        lo = _scalar_loss(graph, feed)
        grad[i] = (hi - lo) / (2 * eps)
        flat[i] = orig
    return grad.reshape(arr.shape)


def _analytic_grad(graph, feed, names):
    wrt = [graph.find(n) for n in names]
    bwd = backward(graph, wrt=wrt)
    seed = "d" + graph.output_names[0]
    out_type = graph.outputs[0].type
    seed_feed = dict(feed)
    seed_feed[seed] = np.ones(out_type.shape, dtype=np.float64)
    res = _execute(bwd, seed_feed)
    return [res[f"d{n}"] for n in names], bwd


class _GradTest(unittest.TestCase):
    def assertGradients(self, build, names=None, rtol=1e-5, atol=1e-7):
        graph, feed = build()
        names = names or [v.name for v in graph.inputs]
        analytic, bwd = _analytic_grad(graph, feed, names)
        for name, got in zip(names, analytic):
            want = _numeric_grad(graph, feed, name)
            np.testing.assert_allclose(
                got, want, rtol=rtol, atol=atol, err_msg=f"d{name} 与数值梯度不一致"
            )
        return graph, bwd, analytic


def _rand(shape, seed=0, scale=1.0, positive=False):
    rng = np.random.default_rng(seed)
    if positive:
        return np.abs(rng.standard_normal(shape)) * scale + 0.5
    return rng.standard_normal(shape) * scale


# --------------------------------------------------------------------------
# 逐元素
# --------------------------------------------------------------------------


class TestElementwiseGrad(_GradTest):
    def _unary(self, op, positive=False, scale=1.0):
        def build():
            g = Graph(op)
            x = g.input("x", f64[3, 4])
            g.output(g.add_node(op, [x]), "y")
            return g, {"x": _rand((3, 4), scale=scale, positive=positive)}

        return build

    def test_unary_rules(self):
        cases = {
            "relu": (False, 1.0),
            "silu": (False, 1.0),
            "exp": (False, 0.5),
            "log": (True, 1.0),
            "sqrt": (True, 1.0),
            "neg": (False, 1.0),
            "tanh": (False, 1.0),
            "sigmoid": (False, 1.0),
        }
        for op, (positive, scale) in cases.items():
            with self.subTest(op=op):
                self.assertGradients(self._unary(op, positive=positive, scale=scale))

    def _binary(self, op):
        def build():
            g = Graph(op)
            a = g.input("a", f64[3, 4])
            b = g.input("b", f64[3, 4])
            g.output(g.add_node(op, [a, b]), "y")
            feed = {"a": _rand((3, 4), seed=1)}
            feed["b"] = _rand((3, 4), seed=2) + (3.0 if op == "div" else 0.0)
            return g, feed

        return build

    def test_binary_rules(self):
        for op in ("add", "sub", "mul", "div"):
            with self.subTest(op=op):
                self.assertGradients(self._binary(op))

    def test_gated_silu_gate_gradient(self):
        """门控 SiLU 对 gate 的梯度必须带上游权重（F01）。

        必须用非均匀 seed 才能发现「漏乘上游 g」：全 1 seed 时两种写法结果相同，
        所以这里额外乘一个权重输入，让数值差分的判据真正生效。
        """

        def build():
            g = Graph("gated_silu")
            x = g.input("x", f64[2, 3])
            gate = g.input("gate", f64[2, 3])
            w = g.input("w", f64[2, 3])
            y = g.add_node("silu", [x, gate])
            g.output(g.add_node("mul", [y, w]), "loss")
            feed = {
                "x": _rand((2, 3), seed=3),
                "gate": _rand((2, 3), seed=4),
                "w": _rand((2, 3), seed=5),
            }
            return g, feed

        self.assertGradients(build, names=["x", "gate"])


# --------------------------------------------------------------------------
# 广播
# --------------------------------------------------------------------------


class TestBroadcastGrad(_GradTest):
    def test_trailing_broadcast(self):
        def build():
            g = Graph("add_bcast")
            a = g.input("a", f64[2, 3, 4])
            b = g.input("b", f64[4])
            g.output(g.add_node("add", [a, b]), "y")
            return g, {"a": _rand((2, 3, 4)), "b": _rand((4,), seed=3)}

        self.assertGradients(build)

    def test_middle_broadcast(self):
        def build():
            g = Graph("mul_bcast")
            a = g.input("a", f64[2, 1, 4])
            b = g.input("b", f64[1, 3, 1])
            g.output(g.add_node("mul", [a, b]), "y")
            return g, {"a": _rand((2, 1, 4)), "b": _rand((1, 3, 1), seed=4)}

        self.assertGradients(build)

    def test_five_dimensional_broadcast(self):
        def build():
            g = Graph("sub_bcast_5d")
            a = g.input("a", f64[2, 3, 1, 4, 5])
            b = g.input("b", f64[1, 1, 6, 1, 1])
            g.output(g.add_node("sub", [a, b]), "y")
            return g, {"a": _rand((2, 3, 1, 4, 5)), "b": _rand((1, 1, 6, 1, 1), seed=5)}

        self.assertGradients(build)


# --------------------------------------------------------------------------
# 矩阵 / 归一化 / 归约
# --------------------------------------------------------------------------


class TestLinearAlgebraGrad(_GradTest):
    def test_matmul_2d(self):
        def build():
            g = Graph("matmul")
            a = g.input("a", f64[3, 4])
            b = g.input("b", f64[4, 2])
            g.output(g.add_node("matmul", [a, b]), "y")
            return g, {"a": _rand((3, 4)), "b": _rand((4, 2), seed=6)}

        self.assertGradients(build)

    def test_matmul_batched_shared_weight(self):
        def build():
            g = Graph("matmul_batched")
            a = g.input("a", f64[3, 2, 4])
            w = g.input("w", f64[4, 5])
            g.output(g.add_node("matmul", [a, w]), "y")
            return g, {"a": _rand((3, 2, 4)), "w": _rand((4, 5), seed=7)}

        self.assertGradients(build)

    def test_softmax_last(self):
        def build():
            g = Graph("softmax")
            x = g.input("x", f64[4, 5])
            g.output(g.add_node("softmax", [x], {"axis": -1}), "y")
            return g, {"x": _rand((4, 5), scale=2.0)}

        self.assertGradients(build)

    def test_softmax_axis0(self):
        def build():
            g = Graph("softmax")
            x = g.input("x", f64[4, 5])
            g.output(g.add_node("softmax", [x], {"axis": 0}), "y")
            return g, {"x": _rand((4, 5), scale=2.0)}

        self.assertGradients(build)

    def test_layernorm(self):
        def build():
            g = Graph("layernorm")
            x = g.input("x", f64[5, 6])
            gamma = g.input("gamma", f64[6])
            beta = g.input("beta", f64[6])
            g.output(g.add_node("layernorm", [x, gamma, beta], {"axes": (-1,)}), "y")
            return g, {
                "x": _rand((5, 6)),
                "gamma": _rand((6,), seed=8),
                "beta": _rand((6,), seed=9),
            }

        self.assertGradients(build)

    def test_scalar_norm_parameters_keep_their_shape(self):
        """广播成标量的 gamma/beta，梯度形状必须回到参数形状（F09）。

        归一化轴是最后一维时，对"其余轴"求和得到的是 (3,)，而参数是 ()；
        不做最终归约的话，优化器会在形状校验处直接失败（或更糟：静默错用）。
        """

        def build():
            g = Graph("layernorm_scalar")
            x = g.input("x", f64[2, 3])
            gamma = g.input("gamma", f64[()])
            beta = g.input("beta", f64[()])
            g.output(g.add_node("layernorm", [x, gamma, beta], {"axes": (-1,)}), "y")
            return g, {
                "x": _rand((2, 3)),
                "gamma": _rand((), seed=1),
                "beta": _rand((), seed=2),
            }

        _, _, analytic = self.assertGradients(build)
        self.assertEqual(analytic[1].shape, ())
        self.assertEqual(analytic[2].shape, ())

    def test_scalar_rmsnorm_gamma_keeps_its_shape(self):
        def build():
            g = Graph("rmsnorm_scalar")
            x = g.input("x", f64[2, 3])
            gamma = g.input("gamma", f64[()])
            g.output(g.add_node("rmsnorm", [x, gamma], {"axes": (-1,)}), "y")
            return g, {"x": _rand((2, 3)), "gamma": _rand((), seed=1)}

        _, _, analytic = self.assertGradients(build)
        self.assertEqual(analytic[1].shape, ())

    def test_reduce_sum_keepdims(self):
        def build():
            g = Graph("reduce_sum_keepdims")
            x = g.input("x", f64[3, 4])
            g.output(g.add_node("reduce_sum", [x], {"axis": -1, "keepdims": True}), "y")
            return g, {"x": _rand((3, 4))}

        self.assertGradients(build)

    def test_reduce_sum_dropdims(self):
        def build():
            g = Graph("reduce_sum")
            x = g.input("x", f64[3, 4])
            g.output(g.add_node("reduce_sum", [x], {"axis": 0, "keepdims": False}), "y")
            return g, {"x": _rand((3, 4))}

        self.assertGradients(build)

    def test_attention_bias_broadcast_gets_a_gradient(self):
        """加性 bias 广播到 batch 上时，梯度要沿被广播的轴求和。

        旧规则用"形状完全相同才给梯度"，把 (1,3,3) 的偏置当成结构性 mask
        返回 None；可学习的相对位置偏置正是这一路的正常用法，梯度必须给出。
        """

        def build():
            g = Graph("attention_bias")
            q = g.input("q", f64[2, 3, 4])
            k = g.input("k", f64[2, 3, 4])
            v = g.input("v", f64[2, 3, 4])
            bias = g.input("bias", f64[1, 3, 3])
            g.output(g.add_node("attention", [q, k, v, bias], {"scale": 0.7}), "y")
            return g, {
                "q": _rand((2, 3, 4), seed=21),
                "k": _rand((2, 3, 4), seed=22),
                "v": _rand((2, 3, 4), seed=23),
                "bias": _rand((1, 3, 3), seed=24, scale=0.5),
            }

        self.assertGradients(build)


# --------------------------------------------------------------------------
# 形状 / 索引
# --------------------------------------------------------------------------


class TestShapeGrad(_GradTest):
    def test_reshape(self):
        def build():
            g = Graph("reshape")
            x = g.input("x", f64[2, 6])
            g.output(g.add_node("reshape", [x], {"shape": [3, 4]}), "y")
            return g, {"x": _rand((2, 6))}

        self.assertGradients(build)

    def test_transpose(self):
        def build():
            g = Graph("transpose")
            x = g.input("x", f64[2, 3, 4])
            g.output(g.add_node("transpose", [x], {"perm": [0, 2, 1]}), "y")
            return g, {"x": _rand((2, 3, 4))}

        self.assertGradients(build)

    def test_slice(self):
        def build():
            g = Graph("slice")
            x = g.input("x", f64[3, 6])
            g.output(g.add_node("slice", [x], {"dim": 1, "start": 1, "stop": 4}), "y")
            return g, {"x": _rand((3, 6))}

        self.assertGradients(build)

    def test_pad(self):
        def build():
            g = Graph("pad")
            x = g.input("x", f64[3, 4])
            g.output(g.add_node("pad", [x], {"dim": 1, "before": 2, "after": 1}), "y")
            return g, {"x": _rand((3, 4))}

        self.assertGradients(build)

    def test_broadcast_to(self):
        def build():
            g = Graph("broadcast_to")
            x = g.input("x", f64[1, 3])
            g.output(g.add_node("broadcast_to", [x], {"shape": [4, 3]}), "y")
            return g, {"x": _rand((1, 3))}

        self.assertGradients(build)

    def test_reduce_max(self):
        def build():
            g = Graph("reduce_max")
            x = g.input("x", f64[3, 5])
            g.output(
                g.add_node("reduce_max", [x], {"axis": -1, "keepdims": True}), "y"
            )
            # 最大值与次大值拉得足够开，1e-6 的扰动不会跨过 argmax 边界
            return g, {"x": _rand((3, 5), seed=7) * 10.0}

        self.assertGradients(build)

    def test_reduce_mean(self):
        def build():
            g = Graph("reduce_mean")
            x = g.input("x", f64[3, 5])
            g.output(
                g.add_node("reduce_mean", [x], {"axis": -1, "keepdims": False}), "y"
            )
            return g, {"x": _rand((3, 5), seed=11)}

        self.assertGradients(build)

    def test_reduce_max_ties_split_gradient(self):
        """并列最大值时梯度均分——与 PyTorch 的 max 反向一致。"""
        g = Graph("reduce_max")
        x = g.input("x", f64[1, 3])
        g.output(g.add_node("reduce_max", [x], {"axis": -1, "keepdims": True}), "y")
        bwd = backward(g)
        res = _execute(bwd, {"x": np.array([[1.0, 5.0, 5.0]]), "dy": np.ones((1, 1))})
        np.testing.assert_allclose(res["dx"], [[0.0, 0.5, 0.5]])

    def test_argmax_is_not_differentiable(self):
        """argmax 输出是下标，接进损失必须明确报错，而不是静默给零。"""
        g = Graph("argmax")
        x = g.input("x", f64[2, 4])
        g.output(g.add_node("argmax", [x], {"axis": -1}), "y")
        with self.assertRaises(NoGradientError):
            backward(g)

    def test_cast_returns_original_dtype(self):
        # 数值差分在这里没有意义：cast 到 f32 的舍入会淹没 1e-6 的扰动。
        # 能测的是规则本身——梯度必须被投回输入的类型，而不是留在 f32。
        def build():
            g = Graph("cast")
            x = g.input("x", f64[2, 3])
            y = g.add_node("cast", [x], {"dtype": "f32"})
            g.output(g.add_node("cast", [y], {"dtype": "f64"}), "y")
            return g, {"x": _rand((2, 3))}

        g, feed = build()
        bwd = backward(g)
        res = _execute(bwd, {**feed, "dy": np.ones((2, 3))})
        self.assertEqual(res["dx"].dtype, np.float64)
        np.testing.assert_allclose(res["dx"], np.ones((2, 3)))


# --------------------------------------------------------------------------
# 卷积 / 池化
# --------------------------------------------------------------------------


class TestConvPoolGrad(_GradTest):
    def _conv(self, x_shape, w_shape, **attrs):
        def build():
            g = Graph("conv2d")
            x = g.input("x", f64[x_shape])
            w = g.input("w", f64[w_shape])
            b = g.input("b", f64[w_shape[0]])
            g.output(g.add_node("conv2d", [x, w, b], attrs), "y")
            return g, {
                "x": _rand(x_shape, scale=0.5),
                "w": _rand(w_shape, seed=10, scale=0.25),
                "b": _rand((w_shape[0],), seed=11, scale=0.1),
            }

        return build

    def test_conv_padding(self):
        self.assertGradients(self._conv((1, 2, 5, 5), (3, 2, 3, 3), padding=(1, 1)))

    def test_conv_stride_dilation(self):
        self.assertGradients(
            self._conv((1, 2, 7, 7), (2, 2, 3, 3), stride=(2, 2), padding=(1, 1), dilation=(2, 2))
        )

    def test_conv_no_bias(self):
        def build():
            g = Graph("conv2d")
            x = g.input("x", f64[1, 1, 4, 4])
            w = g.input("w", f64[2, 1, 2, 2])
            g.output(g.add_node("conv2d", [x, w], {}), "y")
            return g, {"x": _rand((1, 1, 4, 4)), "w": _rand((2, 1, 2, 2), seed=12)}

        self.assertGradients(build)

    def test_maxpool(self):
        def build():
            g = Graph("maxpool2d")
            x = g.input("x", f64[1, 2, 4, 4])
            g.output(g.add_node("maxpool2d", [x], {"kernel": (2, 2)}), "y")
            # 避免并列最大值：并列时"取第一个"的约定会让数值梯度与解析梯度
            # 在扰动跨越边界时不连续，那测的是约定不是规则。
            return g, {"x": _rand((1, 2, 4, 4)) * 10.0}

        self.assertGradients(build)

    def test_maxpool_padding_overlap(self):
        def build():
            g = Graph("maxpool2d")
            x = g.input("x", f64[1, 1, 5, 5])
            g.output(
                g.add_node("maxpool2d", [x], {"kernel": (3, 3), "stride": (1, 1), "padding": (1, 1)}),
                "y",
            )
            return g, {"x": _rand((1, 1, 5, 5)) * 10.0}

        self.assertGradients(build)


# --------------------------------------------------------------------------
# 反向图本身的性质
# --------------------------------------------------------------------------


class TestBackwardGraph(unittest.TestCase):
    def _mlp(self):
        g = Graph("mlp")
        x = g.input("x", f64[4, 3])
        w1 = g.input("w1", f64[3, 5])
        b1 = g.input("b1", f64[5])
        w2 = g.input("w2", f64[5, 2])
        h = g.add_node("relu", [g.add_node("add", [g.add_node("matmul", [x, w1]), b1])])
        g.output(g.add_node("matmul", [h, w2]), "y")
        feed = {
            "x": _rand((4, 3)),
            "w1": _rand((3, 5), seed=13),
            "b1": _rand((5,), seed=14),
            "w2": _rand((5, 2), seed=15),
        }
        return g, feed

    def test_backward_graph_passes_verifier(self):
        g, _ = self._mlp()
        bwd = backward(g)
        caps = ReferenceBackend().capabilities()
        verify(bwd, caps)
        self.assertTrue(all(n.op.portability.value == "portable" for n in bwd.nodes))

    def test_backward_graph_serializes(self):
        g, feed = self._mlp()
        bwd = backward(g)
        restored = loads(dumps(bwd))
        want = _execute(bwd, {**feed, "dy": np.ones((4, 2))})
        got = _execute(restored, {**feed, "dy": np.ones((4, 2))})
        # 反序列化后的图必须能独立跑，且结果一致
        for name in want:
            np.testing.assert_allclose(got[name], want[name], rtol=1e-12)

    def test_parameter_gradient_named_after_input(self):
        g, _ = self._mlp()
        bwd = backward(g)
        self.assertEqual(
            list(bwd.output_names), [f"d{v.name}" for v in g.inputs]
        )

    def test_gradients_match_numeric(self):
        g, feed = self._mlp()
        names = ["x", "w1", "b1", "w2"]
        analytic, _ = _analytic_grad(g, feed, names)
        for name, got in zip(names, analytic):
            np.testing.assert_allclose(got, _numeric_grad(g, feed, name), rtol=1e-5, atol=1e-7)

    def test_custom_seed_and_wrt_subset(self):
        g, feed = self._mlp()
        w2 = g.find("w2")
        bwd = backward(g, wrt=[w2], seeds={"y": "my_seed"})
        self.assertEqual(bwd.output_names, ["dw2"])
        res = _execute(bwd, {**feed, "my_seed": np.ones((4, 2))})
        analytic = _analytic_grad(g, feed, ["w2"])[0][0]
        np.testing.assert_allclose(res["dw2"], analytic, rtol=1e-12)

    def test_intermediate_value_can_be_a_wrt_target(self):
        """``wrt`` 允许中间值：``h=x*x``、``L=sum(h)``，``dL/dh`` 应全为 1。

        逆拓扑循环里 h 的 cotangent 会被 ``pop`` 出去交给它的生产节点；
        不给 ``wrt`` 留一份引用，收集时就会误报"不在依赖路径上"。
        """
        g = Graph("mid")
        x = g.input("x", f64[3])
        h = g.mul(x, x)
        g.output(g.reduce_sum(h, axis=-1, keepdims=False), "l")

        aliaz = "d" + h.name
        bwd = backward(g, wrt=[h])
        self.assertEqual(bwd.output_names, [aliaz])
        res = ReferenceBackend().execute(
            bwd, {"x": np.array([2.0, -3.0, 0.5]), "dl": np.ones(())}
        )
        np.testing.assert_allclose(res[aliaz], np.ones(3))

        # 中间值与输入一起求：两份梯度都要在
        both = backward(g, wrt=[h, x])
        self.assertEqual(both.output_names, [aliaz, "dx"])
        res2 = ReferenceBackend().execute(
            both, {"x": np.array([2.0, -3.0, 0.5]), "dl": np.ones(())}
        )
        np.testing.assert_allclose(res2[aliaz], np.ones(3))
        np.testing.assert_allclose(res2["dx"], np.array([4.0, -6.0, 1.0]))

    def test_missing_gradient_raises(self):
        # y 与输出无关：不能凭空造一张零梯度敷衍过去
        g2 = Graph("dead")
        x = g2.input("x", f64[2, 2])
        y = g2.input("y", f64[2, 2])
        g2.output(g2.add_node("add", [x, x]), "z")
        with self.assertRaises(NoGradientError):
            backward(g2, wrt=[y])

    def test_no_rule_operator_raises(self):
        g = Graph("no_rule")
        dy_in = g.input("gin", f64[1, 2, 4, 4])
        w = g.input("w", f64[2, 2, 3, 3])
        y = g.add_node(
            "conv2d_input_grad", [dy_in, w], {"input_shape": [1, 2, 4, 4], "padding": (1, 1)}
        )
        g.output(y, "y")
        with self.assertRaises(NoGradientError):
            backward(g, wrt=[dy_in])

    def test_seed_name_conflict_raises(self):
        g = Graph("conflict")
        x = g.input("x", f64[2])
        g.output(g.add_node("relu", [x]), "y")
        with self.assertRaises(ValueError):
            backward(g, seeds={"y": "x"})

    def test_symbolic_broadcast_reduction_is_supported(self):
        """符号批维上的广播反向：内部广播维求和只依赖秩，可以在运行期完成。

        ``a[B,1] + b[B,4]`` 对 ``a`` 的梯度要把最后一维求和回长度 1。这条路径
        用 transpose + ``reduce_sum(keepdims=True)`` 实现，不需要在构图期拿到
        具体数字；``B`` 的值只在执行时绑定。
        """
        g = Graph("sym")
        a = g.input("a", f64["B", 1])
        b = g.input("b", f64["B", 4])
        g.output(g.add_node("add", [a, b]), "y")
        bwd = backward(g, wrt=[a])
        backend = ReferenceBackend()
        for batch in (1, 3):
            grads = backend.execute(
                bwd,
                {
                    "a": np.zeros((batch, 1)),
                    "b": np.zeros((batch, 4)),
                    "dy": np.ones((batch, 4)),
                },
            )
            np.testing.assert_array_equal(
                grads["da"], np.full((batch, 1), 4.0, dtype=np.float64)
            )

    def test_inactive_output_is_pruned_not_zero_seeded(self):
        """AF-04：未激活输出要从反向图里剪掉，而不是喂零种子。

        ``loss=sum(x*x)`` 配 ``diag=sum(log(x))``、x=0：零种子在 log 的 vjp
        上仍算出 ``0/0``，把对的梯度污染成 NaN。剪枝后 diag 分支不在反向图
        里，输入里也不会多出 ``ddiag``。
        """
        g = Graph("diag")
        x = g.input("x", f64[2])
        g.output(g.reduce_sum(g.mul(x, x), axis=-1, keepdims=False), "loss")
        g.output(
            g.reduce_sum(g.add_node("log", [x]), axis=-1, keepdims=False), "diag"
        )

        bwd = backward(g, wrt=[x], active=["loss"])
        self.assertEqual(bwd.output_names, ["dx"])
        self.assertEqual([v.name for v in bwd.inputs], ["x", "dloss"])
        with np.errstate(invalid="ignore", divide="ignore"):
            res = ReferenceBackend().execute(
                bwd, {"x": np.zeros(2), "dloss": np.ones(())}
            )
        self.assertTrue(np.array_equal(res["dx"], np.zeros(2)))

        # 对照：喂零种子（旧做法）得到 NaN——说明"剪枝"不是装饰。
        keep = backward(g, wrt=[x])
        with np.errstate(invalid="ignore", divide="ignore"):
            poisoned = ReferenceBackend().execute(
                keep,
                {"x": np.zeros(2), "dloss": np.ones(()), "ddiag": np.zeros(())},
            )
        self.assertFalse(np.all(np.isfinite(poisoned["dx"])))

        with self.assertRaises(KeyError):
            backward(g, wrt=[x], active=["nope"])
        with self.assertRaises(ValueError):
            backward(g, wrt=[x], active=[])


# --------------------------------------------------------------------------
# 反向图的最小化：只对求导目标的下游调用 vjp
# --------------------------------------------------------------------------


class TestBackwardMinimization(unittest.TestCase):
    """``minimize=True`` 的判据：节点更少、梯度逐位相同、非法参数显式拒绝。"""

    def _two_branch(self):
        g = Graph("two_branch")
        x = g.input("x", f64[4])
        y = g.input("y", f64[4])
        a = g.mul(x, x)
        b = g.tanh(y)  # 与 x 无关，但可微：默认路径会为它建反向节点
        g.output(g.add(a, g.mul(b, b)), "loss")
        feed = {
            "x": np.array([0.5, -1.5, 2.0, 0.25]),
            "y": np.array([0.3, -0.7, 1.1, 0.2]),
            "dloss": np.ones(4),  # loss 是向量
        }
        return g, feed

    def test_irrelevant_branch_loses_its_backward_nodes(self):
        g, feed = self._two_branch()
        x = g.find("x")
        full = backward(g, wrt=[x])
        small = backward(g, wrt=[x], minimize=True)
        self.assertLess(len(small.nodes), len(full.nodes))
        want = _execute(full, feed)
        got = _execute(small, feed)
        self.assertEqual(sorted(got), ["dx"])
        np.testing.assert_array_equal(got["dx"], want["dx"])
        np.testing.assert_array_equal(got["dx"], 2.0 * feed["x"])

    def test_intermediate_target_stops_upstream_propagation(self):
        g = Graph("mid")
        x = g.input("x", f64[3])
        h = g.mul(x, x)
        g.output(g.reduce_sum(h, axis=-1, keepdims=False), "l")
        full = backward(g, wrt=[h])
        small = backward(g, wrt=[h], minimize=True)
        self.assertLess(len(small.nodes), len(full.nodes))
        feed = {"x": np.array([2.0, -3.0, 0.5]), "dl": np.ones(())}
        got = _execute(small, feed)["d" + h.name]
        np.testing.assert_array_equal(got, np.ones(3))

    def test_all_inputs_as_targets_is_unchanged(self):
        g, feed = self._two_branch()
        full = backward(g)
        small = backward(g, minimize=True)
        self.assertEqual(len(small.nodes), len(full.nodes))
        want = _execute(full, feed)
        got = _execute(small, feed)
        for name in want:
            np.testing.assert_array_equal(got[name], want[name])

    def test_minimized_graph_is_valid_ir(self):
        g, _ = self._two_branch()
        small = backward(g, wrt=[g.find("x")], minimize=True)
        verify(small, ReferenceBackend().capabilities())

    def test_minimize_rejects_non_bool(self):
        g, _ = self._two_branch()
        for bad in (1, 0, "yes", None):
            with self.assertRaises(TypeError):
                backward(g, minimize=bad)


# --------------------------------------------------------------------------
# 端到端：训练一步
# --------------------------------------------------------------------------


class TestTrainingLoop(unittest.TestCase):
    def _net(self):
        g = Graph("net")
        x = g.input("x", f32[8, 4])
        w = g.input("w", f32[4, 3])
        b = g.input("b", f32[3])
        h = g.add_node("relu", [g.add_node("add", [g.add_node("matmul", [x, w]), b])])
        g.output(g.add_node("softmax", [h], {"axis": -1}), "y")
        return g, x, w, b

    def test_sgd_step_reduces_loss(self):
        g, x, w, b = self._net()
        rng = np.random.default_rng(0)
        xs = rng.standard_normal((8, 4)).astype(np.float32)
        target = rng.standard_normal((8, 3)).astype(np.float32)
        params = {
            "w": (rng.standard_normal((4, 3)) * 0.1).astype(np.float32),
            "b": np.zeros(3, dtype=np.float32),
        }

        def forward(p):
            return _execute(g, {"x": xs, **p})["y"]

        def loss_of(p):
            # 0.5 * sum((y - target)^2)：种子就是 y - target，不需要额外推导系数
            return float(0.5 * np.sum((forward(p) - target) ** 2))

        bwd = backward(g, wrt=[w, b])
        before = loss_of(params)
        lr = 0.05
        for _ in range(20):
            seed = forward(params) - target
            res = _execute(bwd, {"x": xs, "dy": seed.astype(np.float32), **params})
            params = {k: (params[k] - lr * res[f"d{k}"]).astype(np.float32) for k in params}
        self.assertLess(loss_of(params), before, "20 步 SGD 后损失没有下降")

    def test_loss_and_grad_match_finite_difference(self):
        g, _, w, b = self._net()
        rng = np.random.default_rng(1)
        xs = rng.standard_normal((8, 4)).astype(np.float32)
        feed = {
            "x": xs,
            "w": rng.standard_normal((4, 3)).astype(np.float32) * 0.2,
            "b": np.zeros(3, dtype=np.float32),
        }
        bwd = backward(g, wrt=[w, b], seeds={"y": "dy"})
        res = _execute(bwd, {"dy": np.ones((8, 3), dtype=np.float32), **feed})
        for name in ("w", "b"):
            arr = feed[name]
            flat = arr.reshape(-1)
            got = res[f"d{name}"].reshape(-1)
            eps = 1e-3
            for i in range(flat.size):
                orig = float(flat[i])
                flat[i] = orig + eps
                hi = _scalar_loss(g, feed)
                flat[i] = orig - eps
                lo = _scalar_loss(g, feed)
                flat[i] = orig
                want = (hi - lo) / (2 * eps)
                self.assertAlmostEqual(float(got[i]), want, delta=2e-2)


# --------------------------------------------------------------------------
# torch.autograd 交叉验证
# --------------------------------------------------------------------------


@unittest.skipUnless(TORCH_AVAILABLE, "需要 torch")
class TestTorchCrossCheck(unittest.TestCase):
    def test_mlp_gradients_match_torch(self):
        rng = np.random.default_rng(2)
        xs = rng.standard_normal((4, 3)).astype(np.float64)
        w1 = rng.standard_normal((3, 5)).astype(np.float64) * 0.3
        b1 = rng.standard_normal((5,)).astype(np.float64) * 0.1
        w2 = rng.standard_normal((5, 2)).astype(np.float64) * 0.3

        # IR 侧
        g = Graph("mlp")
        x = g.input("x", f64[4, 3])
        w1v = g.input("w1", f64[3, 5])
        b1v = g.input("b1", f64[5])
        w2v = g.input("w2", f64[5, 2])
        h = g.add_node("tanh", [g.add_node("add", [g.add_node("matmul", [x, w1v]), b1v])])
        g.output(g.add_node("matmul", [h, w2v]), "y")
        feed = {"x": xs, "w1": w1, "b1": b1, "w2": w2}
        names = ["w1", "b1", "w2"]
        ours, _ = _analytic_grad(g, feed, names)

        # torch 侧
        t = {k: torch.tensor(v, requires_grad=True) for k, v in feed.items()}
        out = (torch.tanh(t["x"] @ t["w1"] + t["b1"])) @ t["w2"]
        out.sum().backward()

        for name, got in zip(names, ours):
            want = t[name].grad.detach().numpy()
            np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)

    def test_conv_gradients_match_torch(self):
        import torch.nn.functional as F

        rng = np.random.default_rng(3)
        xs = rng.standard_normal((1, 2, 6, 6)).astype(np.float64)
        ws = rng.standard_normal((3, 2, 3, 3)).astype(np.float64) * 0.2
        bs = rng.standard_normal((3,)).astype(np.float64) * 0.1

        g = Graph("conv")
        x = g.input("x", f64[1, 2, 6, 6])
        w = g.input("w", f64[3, 2, 3, 3])
        b = g.input("b", f64[3])
        g.output(g.add_node("conv2d", [x, w, b], {"padding": (1, 1)}), "y")
        feed = {"x": xs, "w": ws, "b": bs}
        ours, _ = _analytic_grad(g, feed, ["x", "w", "b"])

        tx = torch.tensor(xs, requires_grad=True)
        tw = torch.tensor(ws, requires_grad=True)
        tb = torch.tensor(bs, requires_grad=True)
        out = F.conv2d(tx, tw, tb, padding=1)
        out.sum().backward()
        for name, tensor, got in (("x", tx, ours[0]), ("w", tw, ours[1]), ("b", tb, ours[2])):
            np.testing.assert_allclose(
                got, tensor.grad.detach().numpy(), rtol=1e-10, atol=1e-12, err_msg=name
            )

    def test_maxpool_gradients_match_torch(self):
        import torch.nn.functional as F

        rng = np.random.default_rng(4)
        xs = (rng.standard_normal((1, 2, 6, 6)) * 5.0).astype(np.float64)

        g = Graph("pool")
        x = g.input("x", f64[1, 2, 6, 6])
        g.output(g.add_node("maxpool2d", [x], {"kernel": (2, 2)}), "y")
        ours, _ = _analytic_grad(g, {"x": xs}, ["x"])

        tx = torch.tensor(xs, requires_grad=True)
        F.max_pool2d(tx, 2).sum().backward()
        np.testing.assert_allclose(ours[0], tx.grad.detach().numpy(), rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------
# 二阶导数
# --------------------------------------------------------------------------


class TestSecondOrder(unittest.TestCase):
    """二阶导数：Hessian-vector product 的判据是解析二阶导与中心差分两条。

    用 f64：差分噪声随精度平方增长，f32 上"规则写错"与"精度不够"会混在一起。
    """

    def _cube(self):
        """``f(x) = sum(x**3)``：一阶 ``3x**2``、二阶对角 ``6x``。"""
        g = Graph("cube")
        x = g.input("x", f64[3])
        g.output(
            g.reduce_sum(g.mul(g.mul(x, x), x), axis=-1, keepdims=False), "y"
        )
        return g, x

    def test_hessian_vector_product_matches_the_analytic_value(self):
        g, x = self._cube()
        hvp = hessian_vector_product(g, [x])
        # 输入 = 前向输入 + 一阶种子 + 二阶种子；输出沿用一阶的命名
        self.assertEqual([v.name for v in hvp.inputs], ["x", "dy", "ddx"])
        self.assertEqual(hvp.output_names, ["dx"])

        xv = np.array([0.5, -1.25, 2.0])
        v = np.array([1.0, -2.0, 0.5])
        res = _execute(hvp, {"x": xv, "dy": np.ones(()), "ddx": v})
        np.testing.assert_allclose(
            res["dx"], 6.0 * xv * v, rtol=1e-12, atol=1e-12
        )

    def test_hvp_matches_the_central_difference_of_the_gradient(self):
        """独立判据：不借用"二阶规则"，只比较一阶梯度在两个扰动点的差商。"""
        g, x = self._cube()
        bwd = backward(g, wrt=[x])
        hvp = hessian_vector_product(g, [x])

        xv = np.array([0.7, -0.3, 1.1])
        v = np.array([0.2, -0.4, 0.9])
        eps = 1e-5

        def grad_at(point):
            return _execute(bwd, {"x": point, "dy": np.ones(())})["dx"]

        numeric = (grad_at(xv + eps * v) - grad_at(xv - eps * v)) / (2 * eps)
        got = _execute(hvp, {"x": xv, "dy": np.ones(()), "ddx": v})["dx"]
        np.testing.assert_allclose(got, numeric, rtol=1e-6, atol=1e-8)

    def test_default_wrt_covers_every_input(self):
        """不给 wrt 时每个图输入各一项二阶输出，混合偏导也在这张图里。"""
        g = Graph("mixed")
        x = g.input("x", f64[2])
        w = g.input("w", f64[2])
        g.output(
            g.reduce_sum(
                g.mul(g.mul(x, x), g.mul(w, w)), axis=-1, keepdims=False
            ),
            "y",
        )
        hvp = hessian_vector_product(g)
        self.assertEqual(hvp.output_names, ["dx", "dw"])
        self.assertEqual(
            [v.name for v in hvp.inputs], ["x", "w", "dy", "ddx", "ddw"]
        )

        xv = np.array([1.5, -2.0])
        wv = np.array([0.5, 3.0])
        vx = np.array([1.0, -1.0])
        vw = np.array([0.25, 2.0])
        res = _execute(
            hvp,
            {"x": xv, "w": wv, "dy": np.ones(()), "ddx": vx, "ddw": vw},
        )
        # f = sum(x**2 * w**2) 的完整 Hessian 分块：
        #   d2/dx2 = 2*w**2   d2/dw2 = 2*x**2   d2/dxdw = 4*x*w
        # 二阶反向从每个一阶输出各给一个种子，两路的 cotangent 在公共祖先
        # 汇合，因此每个分量都带上了另一块的混合项——这是完整 Hessian 乘
        # 向量，不是逐块独立。
        np.testing.assert_allclose(res["dx"], 2.0 * wv**2 * vx + 4.0 * xv * wv * vw)
        np.testing.assert_allclose(res["dw"], 4.0 * xv * wv * vx + 2.0 * xv**2 * vw)

    def test_newton_step_is_exact_on_a_quadratic(self):
        """收敛判据：用 HVP 拼出 Hessian 再解 ``H p = g``，二次目标一步到位。"""
        g = Graph("quad")
        x = g.input("x", f64[2])
        target = g.input("t", f64[2])
        diff = g.sub(x, target)
        g.output(g.reduce_sum(g.mul(diff, diff), axis=-1, keepdims=False), "y")

        bwd = backward(g, wrt=[x])
        hvp = hessian_vector_product(g, [x])

        x0 = np.array([3.0, -1.0])
        tv = np.array([1.5, -2.0])
        grad = _execute(bwd, {"x": x0, "t": tv, "dy": np.ones(())})["dx"]
        hess = np.stack(
            [
                _execute(
                    hvp, {"x": x0, "t": tv, "dy": np.ones(()), "ddx": basis}
                )["dx"]
                for basis in np.eye(2)
            ],
            axis=1,
        )
        np.testing.assert_allclose(hess, 2 * np.eye(2), atol=1e-12)
        step = np.linalg.solve(hess, grad)
        np.testing.assert_allclose(x0 - step, tv, atol=1e-12)

    def test_zero_curvature_is_an_error_not_a_zero(self):
        """线性目标的二阶导不存在"零张量"：IR 里没有无来源的值，显式报错。"""
        g = Graph("linear")
        x = g.input("x", f64[3])
        w = g.input("w", f64[3])
        g.output(g.reduce_sum(g.mul(x, w), axis=-1, keepdims=False), "y")
        with self.assertRaises(NoGradientError):
            hessian_vector_product(g, [x])

    def test_ops_without_second_order_rules_are_named(self):
        g = Graph("conv")
        x = g.input("x", f64[1, 1, 4, 4])
        w = g.input("w", f64[1, 1, 3, 3])
        y = g.conv2d(x, w, padding=(1, 1))
        reduced = g.reduce_sum(y, axis=-1, keepdims=False)
        g.output(g.reduce_sum(reduced, axis=-1, keepdims=False), "y")
        with self.assertRaises(NoGradientError) as ctx:
            hessian_vector_product(g, [x])
        self.assertIn("conv2d", str(ctx.exception))

    def test_seed_name_collision_is_explicit(self):
        g = Graph("collide")
        x = g.input("x", f64[2])
        # 第二个输入故意占用默认的二阶种子名
        g.input("ddx", f64[2])
        g.output(g.reduce_sum(g.mul(g.mul(x, x), x), axis=-1, keepdims=False), "y")
        with self.assertRaises(ValueError) as ctx:
            hessian_vector_product(g, [x])
        self.assertIn("seeds", str(ctx.exception))
        hvp = hessian_vector_product(g, [x], seeds=["v"])
        self.assertIn("v", [value.name for value in hvp.inputs])

    def test_argument_validation(self):
        g, x = self._cube()
        with self.assertRaises(ValueError):
            hessian_vector_product(g, [])
        with self.assertRaises(ValueError):
            hessian_vector_product(g, [x], seeds=["a", "b"])
        with self.assertRaises(ValueError):
            hessian_vector_product(g, [x], seeds=["a", "a"])

    def test_hvp_graph_is_an_ordinary_graph(self):
        """二阶图仍然是普通 IR：能过 verifier、能 JSON 往返、执行逐位一致。"""
        g, x = self._cube()
        hvp = hessian_vector_product(g, [x])
        verify(hvp)
        clone = loads(dumps(hvp))
        feed = {
            "x": np.array([0.5, -1.25, 2.0]),
            "dy": np.ones(()),
            "ddx": np.array([1.0, -2.0, 0.5]),
        }
        np.testing.assert_array_equal(
            _execute(hvp, feed)["dx"], _execute(clone, feed)["dx"]
        )


if __name__ == "__main__":
    unittest.main()
