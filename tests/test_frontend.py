"""前端 adapter 测试：torch.export -> IR，并验证"降级没有改变语义"。

注意这里验证的**不是**跨后端一致性，而是：
把模型翻译成 IR 之后，在参考后端上算出来的结果，与 torch eager 是否一致。
它回答的是"翻译是否保真"，不回答"不同硬件是否一致"（那需要真硬件）。
"""

import unittest

import numpy as np

from aetherflow import Graph, ReferenceBackend, verify
from aetherflow.backends.cuda import CUDA_AVAILABLE

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "需要 torch")
class TestTorchExport(unittest.TestCase):
    def setUp(self):
        from aetherflow.frontend import import_exported

        self.import_exported = import_exported
        self.ref = ReferenceBackend()

    def _roundtrip(self, model, example, atol=1e-5):
        ep = torch.export.export(model, (example,))
        imported = self.import_exported(ep)
        feed = imported.feed(**{imported.user_inputs[0]: example.numpy()})
        with torch.no_grad():
            expected = model(example).numpy()
        got = list(self.ref.execute(imported.graph, feed).values())[0]
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=atol)
        return imported

    def test_linear_bias_only(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 4)

            def forward(self, x):
                return self.fc(x)

        imported = self._roundtrip(M().eval(), torch.randn(3, 5, 8))
        # linear 应降级为 matmul + add（add 是 (3,5,4)+(4,) 的广播）
        self.assertIn("add", [n.op.name for n in imported.graph.nodes])

    def test_layernorm_silu(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = nn.LayerNorm(16)

            def forward(self, x):
                return F.silu(self.ln(x))

        self._roundtrip(M().eval(), torch.randn(2, 5, 16))

    def test_small_block(self):
        class Block(nn.Module):
            def __init__(self, d=16):
                super().__init__()
                self.ln = nn.LayerNorm(d)
                self.proj = nn.Linear(d, d)
                self.ff = nn.Linear(d, 32)

            def forward(self, x):
                h = self.ln(x)
                return F.silu(self.ff(self.proj(h)))

        self._roundtrip(Block().eval(), torch.randn(2, 5, 16))

    def test_inplace_relu_and_add_are_folded(self):
        """``relu(inplace=True)`` / ``residual += x`` 在导出图里是 ``aten.relu_`` /
        ``aten.add_``。图是 SSA 的，原地写回对别的读者不可见，所以它们必须按函数式
        版本降级，而不是报"未支持的 aten 算子"。
        """

        class Net(nn.Module):
            def __init__(self, d=8):
                super().__init__()
                self.fc = nn.Linear(d, d)

            def forward(self, x):
                h = F.relu(self.fc(x), inplace=True)
                h += x
                return F.relu(h, inplace=True)

        self._roundtrip(Net().eval(), torch.randn(3, 8))

    def test_attention_shapes(self):
        class Attn(nn.Module):
            def __init__(self, d=16, h=4):
                super().__init__()
                self.h = h
                self.qkv = nn.Linear(d, 3 * d)
                self.out = nn.Linear(d, d)

            def forward(self, x):
                b, s, d = x.shape
                q, k, v = self.qkv(x).chunk(3, dim=-1)
                q = q.view(b, s, self.h, d // self.h).transpose(1, 2)
                k = k.view(b, s, self.h, d // self.h).transpose(1, 2)
                v = v.view(b, s, self.h, d // self.h).transpose(1, 2)
                att = torch.softmax(q @ k.transpose(-1, -2), dim=-1)
                y = (att @ v).transpose(1, 2).reshape(b, s, d)
                return self.out(y)

        self._roundtrip(Attn().eval(), torch.randn(2, 5, 16), atol=1e-5)

    def test_imported_graph_is_valid_ir(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 8)

            def forward(self, x):
                return torch.relu(self.fc(x))

        imported = self._roundtrip(M().eval(), torch.randn(2, 3, 8))
        self.assertEqual([str(i) for i in verify(imported.graph)], [])

    def test_unsupported_op_is_rejected(self):
        from aetherflow.frontend.torch_export import UnsupportedOperatorError

        class M(nn.Module):
            def forward(self, x):
                # trunc 不在映射表里（floor/ceil/round/sign 已接上），必须显式
                # 拒绝而不是猜一个近邻语义。
                return torch.trunc(x)

        ep = torch.export.export(M().eval(), (torch.randn(4),))
        with self.assertRaises(UnsupportedOperatorError):
            self.import_exported(ep)

    def test_lenet_cnn(self):
        """LeNet 形态：conv -> relu -> pool 两轮，再 flatten + linear 端到端对拍。"""

        class LeNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv2d(1, 6, 5)
                self.c2 = nn.Conv2d(6, 16, 5)
                self.fc = nn.Linear(16 * 5 * 5, 10)

            def forward(self, x):
                x = F.max_pool2d(F.relu(self.c1(x)), 2)
                x = F.max_pool2d(F.relu(self.c2(x)), 2)
                return self.fc(torch.flatten(x, 1))

        imported = self._roundtrip(LeNet().eval(), torch.randn(2, 1, 32, 32))
        ops = [n.op.name for n in imported.graph.nodes]
        self.assertEqual(ops.count("conv2d"), 2)
        self.assertEqual(ops.count("maxpool2d"), 2)

    def test_conv_with_stride_padding_dilation(self):
        """非默认参数各走一条路径：stride/dilation/output shape 都由 IR 推导。"""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.c = nn.Conv2d(
                    3, 4, (3, 2), stride=(2, 1), padding=(1, 2), dilation=(2, 1)
                )

            def forward(self, x):
                return self.c(x)

        self._roundtrip(M().eval(), torch.randn(1, 3, 12, 9))

    def test_grouped_and_depthwise_conv(self):
        """分组卷积（含 depthwise）已支持：groups 原样写进 IR，结果与 eager 一致。"""

        class Grouped(nn.Module):
            def __init__(self):
                super().__init__()
                self.c = nn.Conv2d(4, 6, 3, padding=1, groups=2)

            def forward(self, x):
                return self.c(x)

        class Depthwise(nn.Module):
            def __init__(self):
                super().__init__()
                self.c = nn.Conv2d(4, 4, 3, padding=1, groups=4)

            def forward(self, x):
                return self.c(x)

        for model in (Grouped().eval(), Depthwise().eval()):
            with self.subTest(model=type(model).__name__):
                imported = self._roundtrip(model, torch.randn(2, 4, 8, 8))
                conv = [n for n in imported.graph.nodes if n.op.name == "conv2d"][0]
                self.assertEqual(conv.attrs["groups"], 2 if isinstance(model, Grouped) else 4)

    def test_adaptive_avg_pool(self):
        """自适应池化（含全局池化）与 eager 一致，且只要一个节点。"""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                return self.pool(x)

        imported = self._roundtrip(M().eval(), torch.randn(2, 3, 7, 5))
        ops = [n.op.name for n in imported.graph.nodes]
        self.assertIn("adaptive_avg_pool2d", ops)

    def test_conv_transpose2d(self):
        """转置卷积落到 ``conv2d_input_grad``：不新增算子也要与 eager 一致。"""

        class M(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.c = nn.ConvTranspose2d(2, 3, 3, **kwargs)

            def forward(self, x):
                return self.c(x)

        cases = [
            ({}, torch.randn(1, 2, 5, 5)),
            ({"stride": 2, "padding": 1, "output_padding": 1}, torch.randn(1, 2, 4, 4)),
            ({"stride": (2, 1), "padding": (1, 0), "output_padding": (1, 0)},
             torch.randn(2, 2, 3, 5)),
        ]
        for kwargs, example in cases:
            with self.subTest(kwargs=kwargs):
                imported = self._roundtrip(M(**kwargs).eval(), example)
                ops = [n.op.name for n in imported.graph.nodes]
                self.assertIn("conv2d_input_grad", ops)
                self.assertNotIn("conv2d", ops)

    def test_conv_transpose2d_grouped(self):
        """分组 / depthwise 转置卷积：weight 布局原样交给 input_grad 的内核。"""

        class Depthwise(nn.Module):
            def __init__(self):
                super().__init__()
                self.c = nn.ConvTranspose2d(4, 4, 3, stride=2, padding=1,
                                            output_padding=1, groups=4)

            def forward(self, x):
                return self.c(x)

        imported = self._roundtrip(Depthwise().eval(), torch.randn(1, 4, 5, 5))
        node = [n for n in imported.graph.nodes if n.op.name == "conv2d_input_grad"][0]
        self.assertEqual(node.attrs["groups"], 4)
        self.assertEqual(tuple(node.attrs["input_shape"]), (1, 4, 10, 10))

    def test_unsupported_cnn_variants_are_rejected(self):
        from aetherflow.frontend.torch_export import UnsupportedOperatorError

        class DilatedPool(nn.Module):
            def forward(self, x):
                return F.max_pool2d(x, 2, dilation=2)

        cases = [
            (DilatedPool().eval(), (torch.randn(1, 1, 6, 6),)),
        ]
        for model, example in cases:
            with self.subTest(model=type(model).__name__):
                ep = torch.export.export(model, example)
                with self.assertRaises(UnsupportedOperatorError):
                    self.import_exported(ep)

    def test_conv_transpose_output_padding_not_less_than_stride(self):
        """output_padding >= stride 会破坏"按 dy 形状反推输入尺寸"，必须拒绝。"""
        from aetherflow import f32
        from aetherflow.frontend.torch_export import (
            UnsupportedOperatorError,
            _conv_transpose2d,
        )

        class _Node:
            name = "bad"

        g = Graph("bad")
        x = g.input("x", f32[1, 2, 3, 3])
        with self.assertRaises(UnsupportedOperatorError):
            _conv_transpose2d(
                g, _Node(), [None, None, None], lambda i: x,
                stride=[1, 1], padding=[0, 0], dilation=[1, 1],
                output_padding=(1, 0), groups=1,
            )

    def test_adapter_is_removable(self):
        """ADR-005 判据：内核不依赖 frontend 模块。"""
        import aetherflow

        self.assertFalse(hasattr(aetherflow, "__torch_required__"))
        # IR / 运行时 / 后端在未导入 frontend 的情况下完全可用
        g = Graph()
        x = g.input("x", aetherflow.f32[4])
        g.output(g.relu(x), "y")
        out = self.ref.execute(g, {"x": np.array([-1.0, 2.0, -3.0, 4.0], np.float32)})
        np.testing.assert_array_equal(out["y"], [0.0, 2.0, 0.0, 4.0])

    # -- 真实模型的算子覆盖（5.39）------------------------------------

    def test_batch_norm_inference(self):
        """推理态 BN 用 running stats，与 IR 的训练态 batch_norm 是两件事。"""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm2d(4)

            def forward(self, x):
                return self.bn(x)

        model = M().eval()
        with torch.no_grad():
            model.bn.running_mean.normal_()
            model.bn.running_var.uniform_(0.5, 1.5)
        self._roundtrip(model, torch.randn(2, 4, 8, 8), atol=1e-6)

    def test_batch_norm_training(self):
        """训练态 BN 用当前 batch 的统计量，与 IR 的 batch_norm 语义相同（5.208）。"""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm2d(4)

            def forward(self, x):
                return self.bn(x)

        model = M().train()
        imported = self._roundtrip(model, torch.randn(3, 4, 6, 6), atol=1e-6)
        self.assertEqual(imported.graph.op_names(), ["batch_norm"])
        # 只把 x 当用户输入：running stats 是参数表里的常量，num_batches_tracked
        # 的原地自增被跳过，不该冒出来要求喂一个 int64 标量。
        self.assertEqual(imported.user_inputs, ["x"])

    def test_batch_norm_training_after_decompositions(self):
        """``run_decompositions()`` 之后的训练态 BN 走同一条导入口。"""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm2d(4)

            def forward(self, x):
                return self.bn(x)

        model = M().train()
        example = torch.randn(3, 4, 6, 6)
        ep = torch.export.export(model, (example,)).run_decompositions()
        imported = self.import_exported(ep)
        self.assertEqual(imported.graph.op_names(), ["batch_norm"])
        self.assertEqual(imported.user_inputs, ["x"])
        with torch.no_grad():
            expected = model(example).numpy()
        got = list(self.ref.execute(imported.graph, imported.feed(x=example.numpy())).values())[0]
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_batch_norm_training_is_trainable(self):
        """训练态 BN 导入后要能求导：dx / dweight / dbias 与 torch 对拍。"""
        from aetherflow import backward

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm2d(4)

            def forward(self, x):
                return self.bn(x)

        torch.manual_seed(3)
        model = M().train()
        example = torch.randn(3, 4, 6, 6, requires_grad=True)
        ep = torch.export.export(model, (example,))
        imported = self.import_exported(ep)
        out_name = imported.graph.output_names[0]
        x_name = imported.user_inputs[0]
        # 训练态 BN 会把 running_mean / running_var 也提升成图输入（它们是
        # buffer，不是用户输入），但它们不在输出依赖路径上；默认 wrt=全部
        # 输入会按 5.19x 的契约报 NoGradientError，所以要显式挑目标。
        param_inputs = {v: k for k, v in ep.graph_signature.inputs_to_parameters.items()}
        w_name = next(v for k, v in param_inputs.items() if k.split(".")[-1] == "weight")
        b_name = next(v for k, v in param_inputs.items() if k.split(".")[-1] == "bias")
        wanted = {x_name, w_name, b_name}
        bwd = backward(
            imported.graph, wrt=[v for v in imported.graph.inputs if v.name in wanted]
        )

        rng = np.random.default_rng(5)
        seed = rng.standard_normal((3, 4, 6, 6)).astype(np.float32)
        out = model(example)
        (out * torch.from_numpy(seed)).sum().backward()

        feed = imported.feed(**{x_name: example.detach().numpy()})
        feed["d" + out_name] = seed
        got = self.ref.execute(bwd, feed)

        np.testing.assert_allclose(
            got["d" + x_name], example.grad.numpy(), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            got["d" + w_name], model.bn.weight.grad.numpy(), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            got["d" + b_name], model.bn.bias.grad.numpy(), rtol=1e-4, atol=1e-4
        )

    def test_avg_pool_and_cat(self):
        class M(nn.Module):
            def forward(self, x, y):
                pooled = F.avg_pool2d(x, 2)
                return torch.cat([pooled, y], dim=1)

        model = M().eval()
        x = torch.randn(1, 2, 8, 8)
        y = torch.randn(1, 3, 4, 4)
        ep = torch.export.export(model, (x, y))
        imported = self.import_exported(ep)
        feed = imported.feed(
            **{imported.user_inputs[0]: x.numpy(), imported.user_inputs[1]: y.numpy()}
        )
        expected = model(x, y).numpy()
        got = list(self.ref.execute(imported.graph, feed).values())[0]
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_hardtanh_and_mean(self):
        class M(nn.Module):
            def forward(self, x):
                return F.hardtanh(x.mean(dim=(2, 3)).relu(), 0.0, 6.0)

        self._roundtrip(M().eval(), torch.randn(2, 5, 4, 4) * 4, atol=1e-6)

    def test_exact_gelu(self):
        """PyTorch 默认 GELU 是 erf 精确式，不能拿 tanh 近似顶替。"""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.act = nn.GELU()

            def forward(self, x):
                return self.act(x)

        self._roundtrip(M().eval(), torch.randn(3, 7) * 2, atol=1e-6)

    def test_tanh_gelu(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.act = nn.GELU(approximate="tanh")

            def forward(self, x):
                return self.act(x)

        self._roundtrip(M().eval(), torch.randn(3, 7) * 2, atol=1e-6)

    def test_unary_math_family(self):
        """IR 已有的一元数学算子全部接上：每个都与 eager 逐点一致。"""
        base = torch.randn(2, 3, 4) + 1.0
        cases = {
            "tanh": (lambda x: torch.tanh(x), base, 1e-6),
            "exp": (lambda x: torch.exp(x - 2.0), base - 2.0, 1e-6),
            "log": (lambda x: torch.log(x.abs() + 1.0), base, 1e-6),
            "neg": (lambda x: -x, base, 1e-6),
            "sqrt": (lambda x: torch.sqrt(x.abs() + 1.0), base, 1e-6),
            "erf": (lambda x: torch.erf(x), base, 1e-6),
            "rsqrt": (lambda x: torch.rsqrt(x.abs() + 1.0), base, 1e-5),
            "reciprocal": (lambda x: torch.reciprocal(x + 1.0), base, 1e-6),
            "square": (lambda x: x.square(), base, 1e-6),
            "abs": (lambda x: x.abs(), base, 1e-6),
            "log1p": (lambda x: torch.log1p(x.abs()), base, 1e-5),
            "expm1": (lambda x: torch.expm1(x), base - 1.0, 1e-6),
        }

        for name, (fn, example, atol) in cases.items():
            with self.subTest(op=name):
                class M(nn.Module):
                    def forward(self, x):
                        return fn(x)

                imported = self._roundtrip(M().eval(), example, atol=atol)
                ops = [n.op.name for n in imported.graph.nodes]
                self.assertTrue(ops, "降级后不应是空图")

    def test_activation_expansions(self):
        """没有专门 IR 算子的激活：用已有原语精确展开后与 eager 对拍。"""
        cases = {
            "leaky_relu": (lambda x: F.leaky_relu(x, 0.1), 1e-6),
            "elu": (lambda x: F.elu(x, 1.3), 1e-6),
            "hardsigmoid": (lambda x: F.hardsigmoid(x), 1e-6),
            "hardswish": (lambda x: F.hardswish(x), 1e-6),
            "relu6": (lambda x: F.relu6(x), 1e-6),
            "softplus": (lambda x: F.softplus(x), 1e-6),
            "log_softmax": (lambda x: F.log_softmax(x, dim=-1), 1e-5),
        }
        example = torch.randn(3, 8) * 3
        for name, (fn, atol) in cases.items():
            with self.subTest(op=name):
                class M(nn.Module):
                    def forward(self, x):
                        return fn(x)

                self._roundtrip(M().eval(), example, atol=atol)

    def test_softplus_handles_large_values(self):
        """稳定恒等式在 ±50 上不溢出：torch 的 threshold 保护与精确解只差 e^-20。"""
        class M(nn.Module):
            def forward(self, x):
                return F.softplus(x)

        example = torch.tensor([[-50.0, -21.0, 0.0, 20.0, 50.0, 80.0]])
        self._roundtrip(M().eval(), example, atol=1e-6)

    def test_softplus_custom_threshold_is_rejected(self):
        from aetherflow.frontend import UnsupportedOperatorError

        class M(nn.Module):
            def forward(self, x):
                return F.softplus(x, threshold=5.0)

        with self.assertRaisesRegex(UnsupportedOperatorError, "threshold"):
            self.import_exported(torch.export.export(M().eval(), (torch.randn(3),)))

    def test_clamp_variants(self):
        cases = {
            "clamp": lambda x: torch.clamp(x, -1.0, 2.0),
            "clamp_min": lambda x: torch.clamp_min(x, 0.5),
            "clamp_max": lambda x: torch.clamp_max(x, 0.5),
        }
        for name, fn in cases.items():
            with self.subTest(op=name):
                class M(nn.Module):
                    def forward(self, x):
                        return fn(x)

                self._roundtrip(M().eval(), torch.randn(3, 5) * 2, atol=1e-6)

    def test_clamp_with_tensor_bounds_is_rejected(self):
        """IR 的 clamp 边界写在属性里；图内边界不能静默当成常量。"""
        from aetherflow.frontend import UnsupportedOperatorError

        class M(nn.Module):
            def forward(self, x, lo):
                return torch.clamp(x, min=lo)

        args = (torch.randn(3, 4), torch.randn(3, 4))
        with self.assertRaisesRegex(UnsupportedOperatorError, "clamp"):
            self.import_exported(torch.export.export(M().eval(), args))

    def test_scalar_binary_ops(self):
        cases = {
            "add": (lambda x: x + 1.5, lambda x: x + 1.5),
            "sub": (lambda x: x - 1.5, lambda x: x - 1.5),
            "mul": (lambda x: x * 2.5, lambda x: x * 2.5),
            "div": (lambda x: x / 2.5, lambda x: x / 2.5),
            "rsub": (lambda x: 1.5 - x, lambda x: 1.5 - x),
        }
        for name, (fn, _) in cases.items():
            with self.subTest(op=name):
                class M(nn.Module):
                    def forward(self, x):
                        return fn(x)

                self._roundtrip(M().eval(), torch.randn(2, 5), atol=1e-6)

    def test_pow_scalar_expansions(self):
        cases = {
            "square": (lambda x: x ** 2, 1e-6),
            "cube": (lambda x: x ** 3, 1e-6),
            "neg1": (lambda x: x ** -1, 1e-5),
            "sqrt": (lambda x: x ** 0.5, 1e-6),
            "zero": (lambda x: x ** 0, 0.0),
            "pow16": (lambda x: x ** 16, 1e-5),
        }
        example = torch.rand(2, 4) + 0.5
        for name, (fn, atol) in cases.items():
            with self.subTest(op=name):
                class M(nn.Module):
                    def forward(self, x):
                        return fn(x)

                self._roundtrip(M().eval(), example, atol=atol)

    def test_pow_fractional_exponent_is_rejected(self):
        """exp(y*log(x)) 在非正输入上不是原式，不能当成等价降级。"""
        from aetherflow.frontend import UnsupportedOperatorError

        class M(nn.Module):
            def forward(self, x):
                return x ** 1.5

        with self.assertRaisesRegex(UnsupportedOperatorError, "pow"):
            self.import_exported(torch.export.export(M().eval(), (torch.rand(3) + 1.0,)))

    def test_dtype_cast(self):
        class M(nn.Module):
            def forward(self, x):
                # f32 -> f64 -> f32 往返：cast 只改 dtype，不改值
                return x.to(torch.float64).to(torch.float32)

        self._roundtrip(M().eval(), torch.randn(2, 3), atol=0.0)

    def test_type_as_uses_the_other_tensor_dtype(self):
        class M(nn.Module):
            def forward(self, x, ref):
                return x.type_as(ref)

        x = torch.randn(3, 4)
        ref = torch.randn(3, 4).double()
        _, expected, got = self._import_and_run(M().eval(), (x, ref))
        self.assertEqual(got.dtype, np.float64)
        np.testing.assert_allclose(got, expected, rtol=0, atol=0)

    def test_cast_to_integer_matches_truncation(self):
        """f32 -> i32 是向零截断，与 torch 的 to(int64) 在 i32 范围内逐点一致。"""

        class M(nn.Module):
            def forward(self, x):
                return x.to(torch.int64).to(torch.float32)

        self._roundtrip(M().eval(), torch.randn(3) * 10, atol=0.0)

    def test_rounding_unary_ops(self):
        """sign / round / floor / ceil 与 torch 逐位一致（round 是 half-to-even）。"""
        cases = {
            "sign": lambda x: torch.sign(x),
            "round": lambda x: torch.round(x),
            "floor": lambda x: torch.floor(x),
            "ceil": lambda x: torch.ceil(x),
        }
        # 取 .5 附近的点专门检验舍入方向：2.5 -> 2、3.5 -> 4
        x = torch.tensor([-3.5, -2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 3.5])
        for name, fn in cases.items():
            class M(nn.Module):
                def forward(self, t):
                    return fn(t)

            self._roundtrip(M().eval(), x.clone(), atol=0.0)

    def test_trunc_is_rejected(self):
        from aetherflow.frontend import UnsupportedOperatorError

        class M(nn.Module):
            def forward(self, x):
                return torch.trunc(x)

        with self.assertRaises(UnsupportedOperatorError):
            self.import_exported(torch.export.export(M().eval(), (torch.randn(3),)))

    def test_reductions(self):
        cases = {
            "mean_all": lambda x: x.mean(),
            "sum_all": lambda x: x.sum(),
            "sum_dim": lambda x: x.sum(dim=1, keepdim=True),
            "sum_multi": lambda x: x.sum(dim=(0, 2)),
            "amax": lambda x: torch.amax(x, dim=1),
            "amax_multi": lambda x: torch.amax(x, dim=(1, 2), keepdim=True),
        }
        example = torch.randn(2, 3, 4)
        for name, fn in cases.items():
            with self.subTest(op=name):
                class M(nn.Module):
                    def forward(self, x):
                        return fn(x)

                self._roundtrip(M().eval(), example, atol=1e-5)

    def test_max_dim_values_and_indices(self):
        class Values(nn.Module):
            def forward(self, x):
                return torch.max(x, dim=1)[0]

        class Indices(nn.Module):
            def forward(self, x):
                return torch.max(x, dim=1)[1]

        example = torch.randn(2, 5, 3)
        for model, atol in ((Values().eval(), 1e-6), (Indices().eval(), 0.0)):
            with self.subTest(model=type(model).__name__):
                imported, expected, got = self._import_and_run(model, (example,))
                np.testing.assert_allclose(got, expected, rtol=0, atol=atol)
                self.assertIn("argmax", [n.op.name for n in imported.graph.nodes])

    def test_argmax_global(self):
        class M(nn.Module):
            def forward(self, x):
                return torch.argmax(x)

        example = torch.randn(3, 4)
        _, expected, got = self._import_and_run(M().eval(), (example,))
        self.assertEqual(int(got), int(expected))

    def test_tril_triu_cumsum(self):
        class M(nn.Module):
            def forward(self, x):
                return torch.tril(x, -1) + torch.triu(x, 1) + torch.cumsum(x, dim=1)

        self._roundtrip(M().eval(), torch.randn(3, 4), atol=1e-6)

    def test_triu_diagonal(self):
        cases = {
            "k0": lambda x: torch.triu(x),
            "k2": lambda x: torch.triu(x, 2),
            "kneg": lambda x: torch.triu(x, -1),
        }
        for name, fn in cases.items():
            with self.subTest(op=name):
                class M(nn.Module):
                    def forward(self, x):
                        return fn(x)

                self._roundtrip(M().eval(), torch.randn(4, 5), atol=0.0)

    def test_pad(self):
        class M(nn.Module):
            def forward(self, x):
                return F.pad(x, (1, 2, 0, 1))

        imported = self._roundtrip(M().eval(), torch.randn(2, 3, 4), atol=0.0)
        # 两对 pad 各自落到一个 IR 节点
        self.assertEqual([n.op.name for n in imported.graph.nodes].count("pad"), 2)

    def test_pad_reflect_is_rejected(self):
        from aetherflow.frontend import UnsupportedOperatorError

        class M(nn.Module):
            def forward(self, x):
                return F.pad(x, (1, 1, 1, 1), mode="reflect")

        with self.assertRaisesRegex(UnsupportedOperatorError, "mode"):
            self.import_exported(torch.export.export(M().eval(), (torch.randn(1, 1, 4, 4),)))

    def test_pad_nonzero_value_is_rejected(self):
        from aetherflow.frontend import UnsupportedOperatorError

        class M(nn.Module):
            def forward(self, x):
                return F.pad(x, (1, 1), value=1.0)

        with self.assertRaisesRegex(UnsupportedOperatorError, "value"):
            self.import_exported(torch.export.export(M().eval(), (torch.randn(1, 1, 3, 3),)))

    def test_expand_as_stack_and_split(self):
        class M(nn.Module):
            def forward(self, x):
                y = torch.stack([x, x + 1.0], dim=0)
                first, second = torch.split(y, [1, 1], dim=0)
                return first + second

        example = torch.randn(2, 3)
        _, expected, got = self._import_and_run(M().eval(), (example,))
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_unbind_and_expand_as(self):
        class M(nn.Module):
            def forward(self, x, ref):
                a, b = torch.unbind(x, dim=0)
                return (a + b).expand_as(ref)

        x = torch.randn(2, 3, 4)
        ref = torch.randn(3, 4)
        _, expected, got = self._import_and_run(M().eval(), (x, ref))
        np.testing.assert_allclose(got, expected, rtol=0, atol=0)

    def test_narrow_squeeze_flatten_and_t(self):
        class M(nn.Module):
            def forward(self, x):
                y = x.narrow(1, 1, 2).unsqueeze(0)  # (1,2,2,4)
                y = y.squeeze()                      # (2,2,4)
                return y.reshape(4, 4).t().flatten()  # (16,)

        self._roundtrip(M().eval(), torch.randn(2, 3, 4), atol=0.0)

    def test_constant_construction(self):
        cases = {
            "zeros": lambda x: x + torch.zeros(2, 3),
            "ones": lambda x: x + torch.ones(2, 3),
            "full": lambda x: x + torch.full((2, 3), 2.5),
            "arange": lambda x: x + torch.arange(3.0),
            "zeros_like": lambda x: x + torch.zeros_like(x),
            "ones_like": lambda x: x + torch.ones_like(x),
            "full_like": lambda x: x + torch.full_like(x, 2.5),
        }
        for name, fn in cases.items():
            with self.subTest(constructor=name):
                class M(nn.Module):
                    def forward(self, x):
                        return fn(x)

                imported = self._roundtrip(M().eval(), torch.randn(2, 3), atol=0.0)
                ops = [n.op.name for n in imported.graph.nodes]
                self.assertEqual(ops.count("constant"), 1, ops)

    def test_constant_over_limit_is_rejected(self):
        """塞进图属性的常量有元素上限；更大的值应改成参数或显式输入。"""
        from aetherflow.frontend import UnsupportedOperatorError

        class M(nn.Module):
            def forward(self, x):
                return x + torch.zeros(300, 300)

        with self.assertRaisesRegex(UnsupportedOperatorError, "65536"):
            self.import_exported(
                torch.export.export(M().eval(), (torch.randn(300, 300),))
            )

    def test_matmul_family(self):
        class MM(nn.Module):
            def forward(self, x, y):
                return torch.mm(x, y)

        class MV(nn.Module):
            def forward(self, x, v):
                return torch.mv(x, v)

        class Dot(nn.Module):
            def forward(self, x, y):
                return torch.dot(x, y)

        class AddMM(nn.Module):
            def forward(self, b, x, y):
                return torch.addmm(b, x, y, beta=0.5, alpha=2.0)

        cases = [
            (MM().eval(), (torch.randn(3, 4), torch.randn(4, 5))),
            (MV().eval(), (torch.randn(3, 4), torch.randn(4))),
            (Dot().eval(), (torch.randn(5), torch.randn(5))),
            (AddMM().eval(), (torch.randn(3, 5), torch.randn(3, 4), torch.randn(4, 5))),
        ]
        for model, example in cases:
            with self.subTest(model=type(model).__name__):
                _, expected, got = self._import_and_run(model, example)
                np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    def test_shape_ops(self):
        class M(nn.Module):
            def forward(self, x):
                y = x.unsqueeze(0).expand(3, -1, -1, -1, -1)
                y = y.squeeze(0).select(1, 0)
                return y[..., 1:4]

        self._roundtrip(M().eval(), torch.randn(2, 4, 6, 5), atol=1e-6)

    def test_scaled_dot_product_attention(self):
        class M(nn.Module):
            def forward(self, q, k, v):
                return F.scaled_dot_product_attention(q, k, v)

        model = M().eval()
        q, k, v = (torch.randn(2, 3, 5, 8) for _ in range(3))
        ep = torch.export.export(model, (q, k, v))
        imported = self.import_exported(ep)
        feed = imported.feed(
            **{
                name: tensor.numpy()
                for name, tensor in zip(imported.user_inputs, (q, k, v))
            }
        )
        expected = model(q, k, v).numpy()
        got = list(self.ref.execute(imported.graph, feed).values())[0]
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def _import_and_run(self, model, example):
        """多输入模型的端到端对拍：IR 结果必须等于 torch eager 的结果。"""
        ep = torch.export.export(model, tuple(example))
        imported = self.import_exported(ep)
        feed = imported.feed(
            **{
                name: tensor.numpy()
                for name, tensor in zip(imported.user_inputs, example)
            }
        )
        with torch.no_grad():
            expected = model(*example).numpy()
        got = list(self.ref.execute(imported.graph, feed).values())[0]
        return imported, expected, got

    def test_add_alpha_is_not_ignored(self):
        class M(nn.Module):
            def forward(self, x, y):
                return torch.add(x, y, alpha=3.0)

        example = (torch.randn(3, 4), torch.randn(3, 4))
        _, expected, got = self._import_and_run(M().eval(), example)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(got, (example[0] + 3.0 * example[1]).numpy())

    def test_sub_alpha_is_not_ignored(self):
        class M(nn.Module):
            def forward(self, x, y):
                return torch.sub(x, y, alpha=2.5)

        example = (torch.randn(3, 4), torch.randn(3, 4))
        _, expected, got = self._import_and_run(M().eval(), example)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(got, (example[0] - 2.5 * example[1]).numpy())

    def test_layer_norm_positional_eps_is_not_ignored(self):
        """eps 是位置参数：只读 kwargs 会让 0.1 静默退回 1e-5。"""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.randn(8))
                self.b = nn.Parameter(torch.randn(8))

            def forward(self, x):
                return F.layer_norm(x, (8,), self.w, self.b, 0.1)

        # 缩小输入方差，让 eps 的量级差异在数值上明显
        example = (torch.randn(4, 8) * 0.01,)
        model = M().eval()
        _, expected, got = self._import_and_run(model, example)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
        # 反向证明：用 1e-5 的结果必须明显偏离
        with torch.no_grad():
            wrong = F.layer_norm(example[0], (8,), model.w, model.b, 1e-5)
        self.assertGreater(float(np.abs(expected - wrong.numpy()).max()), 1e-2)

    def test_sdpa_scale_is_not_ignored(self):
        """scale 是 kw-only：按 args[6] 读永远读不到，会退回 1/sqrt(D)。"""

        class M(nn.Module):
            def forward(self, q, k, v):
                return F.scaled_dot_product_attention(q, k, v, scale=0.01)

        example = tuple(torch.randn(2, 3, 5, 8) for _ in range(3))
        _, expected, got = self._import_and_run(M().eval(), example)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_sdpa_enable_gqa_is_reported(self):
        """enable_gqa=True 是 kw-only：拒绝分支必须真的走到。"""
        from aetherflow.frontend import UnsupportedOperatorError

        class M(nn.Module):
            def forward(self, q, k, v):
                return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

        q = torch.randn(2, 4, 5, 8)
        k = torch.randn(2, 2, 5, 8)
        v = torch.randn(2, 2, 5, 8)
        with self.assertRaises(UnsupportedOperatorError):
            self.import_exported(torch.export.export(M().eval(), (q, k, v)))

    def test_maxpool_ceil_mode(self):
        class M(nn.Module):
            def forward(self, x):
                return F.max_pool2d(x, 3, 2, ceil_mode=True)

        imported = self._roundtrip(M().eval(), torch.randn(1, 2, 7, 9), atol=1e-6)
        # 5.154：ceil_mode 现在是 IR 属性，不再靠前端复制边界来凑尺寸
        node = imported.graph.nodes[-1]
        self.assertEqual(node.op.name, "maxpool2d")
        self.assertTrue(node.attrs.get("ceil_mode"))
        self.assertEqual([n.op.name for n in imported.graph.nodes].count("concat"), 0)

    def test_avgpool_ceil_mode(self):
        class M(nn.Module):
            def forward(self, x):
                return F.avg_pool2d(x, 3, 2, ceil_mode=True)

        imported = self._roundtrip(M().eval(), torch.randn(1, 2, 8, 8), atol=1e-6)
        node = imported.graph.nodes[-1]
        self.assertEqual(node.op.name, "avgpool2d")
        self.assertTrue(node.attrs.get("ceil_mode"))

    def test_avgpool_count_include_pad_false(self):
        class M(nn.Module):
            def forward(self, x):
                return F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)

        imported = self._roundtrip(M().eval(), torch.randn(1, 2, 5, 5), atol=1e-6)
        node = imported.graph.nodes[-1]
        self.assertIs(node.attrs.get("count_include_pad"), False)

    def test_avgpool_divisor_override_is_rejected(self):
        from aetherflow.frontend import UnsupportedOperatorError

        class M(nn.Module):
            def forward(self, x):
                return F.avg_pool2d(x, 2, divisor_override=3)

        with self.assertRaises(UnsupportedOperatorError):
            self.import_exported(torch.export.export(M().eval(), (torch.randn(1, 2, 4, 4),)))

    def test_training_dropout_is_rejected(self):
        from aetherflow.frontend import UnsupportedOperatorError

        class M(nn.Module):
            def forward(self, x):
                return F.dropout(x, 0.5, training=True)

        model = M().train()
        with self.assertRaises(UnsupportedOperatorError):
            self.import_exported(torch.export.export(model, (torch.randn(4),)))

    def test_causal_attention_is_rejected(self):
        from aetherflow.frontend import UnsupportedOperatorError

        class M(nn.Module):
            def forward(self, q, k, v):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        model = M().eval()
        args = tuple(torch.randn(1, 2, 4, 8) for _ in range(3))
        with self.assertRaises(UnsupportedOperatorError):
            self.import_exported(torch.export.export(model, args))

    def test_resnet18_matches_torch(self):
        try:
            import torchvision.models as tvm
        except ImportError:  # pragma: no cover
            self.skipTest("需要 torchvision")
        model = tvm.resnet18(weights=None).eval()
        self._roundtrip(model, torch.randn(1, 3, 64, 64), atol=1e-5)

    def test_squeezenet_matches_torch(self):
        try:
            import torchvision.models as tvm
        except ImportError:  # pragma: no cover
            self.skipTest("需要 torchvision")
        model = tvm.squeezenet1_0(weights=None).eval()
        self._roundtrip(model, torch.randn(1, 3, 64, 64), atol=1e-5)

    def test_embedding_with_int64_tokens(self):
        """AF-32 补：int64 token 输入按 i32 承载，``table[ids]`` 落成 embedding。"""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(10, 4)

            def forward(self, ids):
                return self.emb(ids)

        ids = torch.tensor([[1, 2], [3, 9]], dtype=torch.int64)
        imported = self._roundtrip(M().eval(), ids)
        self.assertEqual(imported.user_inputs, ["ids"])
        ids_input = next(v for v in imported.graph.inputs if v.name == "ids")
        self.assertEqual(
            ids_input.type.dtype.value, "i32",
            "int64 应按 IR 的 i32 声明",
        )
        self.assertIn("embedding", [n.op.name for n in imported.graph.nodes])

    def test_functional_embedding_without_padding(self):
        """``F.embedding``（无 padding_idx）同样落到 embedding。"""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.randn(7, 3))

            def forward(self, ids):
                return F.embedding(ids, self.w)

        ids = torch.tensor([0, 6, 2], dtype=torch.int64)
        imported = self._roundtrip(M().eval(), ids)
        self.assertEqual([n.op.name for n in imported.graph.nodes], ["embedding"])

    def test_embedding_padding_idx_is_rejected(self):
        """padding_idx 有"该行梯度恒为零"的语义，IR 没有 → 显式拒绝。"""
        from aetherflow.frontend import UnsupportedOperatorError

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.randn(10, 4))

            def forward(self, ids):
                return F.embedding(ids, self.w, padding_idx=0)

        ids = torch.tensor([0, 3], dtype=torch.int64)
        with self.assertRaisesRegex(UnsupportedOperatorError, "padding_idx"):
            self.import_exported(torch.export.export(M().eval(), (ids,)))

    def test_general_advanced_index_is_rejected(self):
        """多索引的高级索引不能退化成"只取第一个索引"。"""
        from aetherflow.frontend import UnsupportedOperatorError

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.randn(5, 5))

            def forward(self, a, b):
                return self.w[a, b]

        a = torch.tensor([0, 1], dtype=torch.int64)
        with self.assertRaisesRegex(UnsupportedOperatorError, "高级索引"):
            self.import_exported(torch.export.export(M().eval(), (a, a)))

    def test_lifted_constant_is_not_a_user_input(self):
        """字面量张量被提升为 placeholder，但它是常量而不是用户输入。"""

        class M(nn.Module):
            def forward(self, x):
                return x + torch.tensor([1.0, 2.0, 3.0])

        imported = self._roundtrip(M().eval(), torch.randn(2, 3))
        self.assertEqual(imported.user_inputs, ["x"])
        self.assertIn("constant", [n.op.name for n in imported.graph.nodes])
        self.assertEqual(imported.parameters, {}, "常量不能混进参数表")

    def test_get_attr_constant_becomes_constant_node(self):
        """``get_attr`` 常量同样落成 constant，而不是报未支持。"""
        from types import SimpleNamespace

        from torch.fx import Graph, GraphModule

        class Holder(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.tensor([1.0, 2.0])

        graph = Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.randn(2, 2)
        bias = graph.get_attr("bias")
        out = graph.call_function(torch.ops.aten.add.Tensor, (x, bias))
        graph.output((out,))
        gm = GraphModule(Holder(), graph)
        sig = SimpleNamespace(
            inputs_to_parameters={},
            inputs_to_buffers={},
            inputs_to_lifted_tensor_constants={},
            user_inputs=("x",),
            user_outputs=(),  # 空表示"输出槽全部保留"（与真实签名的非空分支等价）
        )
        ep = SimpleNamespace(graph_module=gm, graph_signature=sig, state_dict={})
        imported = self.import_exported(ep)
        self.assertEqual(imported.user_inputs, ["x"])
        self.assertIn("constant", [n.op.name for n in imported.graph.nodes])

        feed = imported.feed(x=np.ones((2, 2), "float32"))
        got = list(self.ref.execute(imported.graph, feed).values())[0]
        np.testing.assert_allclose(got, np.ones((2, 2)) + np.array([1.0, 2.0]))

    def test_oversized_lifted_constant_is_rejected(self):
        """常量落进 attrs 有体积上限，超限显式拒绝而不是塞进图里。"""
        from types import SimpleNamespace

        from aetherflow.frontend import UnsupportedOperatorError
        from aetherflow.frontend.torch_export import MAX_IMPORTED_CONSTANT_ELEMENTS
        from torch.fx import Graph, GraphModule

        size = MAX_IMPORTED_CONSTANT_ELEMENTS + 1

        class Holder(nn.Module):
            def __init__(self):
                super().__init__()
                self.big = torch.zeros(size)

        graph = Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.randn(1)
        big = graph.get_attr("big")
        out = graph.call_function(torch.ops.aten.add.Tensor, (x, big))
        graph.output((out,))
        gm = GraphModule(Holder(), graph)
        ep = SimpleNamespace(
            graph_module=gm,
            graph_signature=SimpleNamespace(
                inputs_to_parameters={},
                inputs_to_buffers={},
                inputs_to_lifted_tensor_constants={},
                user_inputs=("x",),
                user_outputs=(),
            ),
            state_dict={},
        )
        with self.assertRaisesRegex(UnsupportedOperatorError, "导入上限"):
            self.import_exported(ep)

    def test_out_of_range_integer_constant_is_rejected(self):
        """常量不进执行器入口，所以 i32 范围检查必须在导入时做。"""
        from types import SimpleNamespace

        from aetherflow.frontend import UnsupportedOperatorError
        from torch.fx import Graph, GraphModule

        class Holder(nn.Module):
            def __init__(self):
                super().__init__()
                self.big = torch.tensor([2 ** 40], dtype=torch.int64)

        graph = Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.zeros(1, dtype=torch.int32)
        big = graph.get_attr("big")
        out = graph.call_function(torch.ops.aten.add.Tensor, (x, big))
        graph.output((out,))
        gm = GraphModule(Holder(), graph)
        ep = SimpleNamespace(
            graph_module=gm,
            graph_signature=SimpleNamespace(
                inputs_to_parameters={},
                inputs_to_buffers={},
                inputs_to_lifted_tensor_constants={},
                user_inputs=("x",),
                user_outputs=(),
            ),
            state_dict={},
        )
        with self.assertRaisesRegex(UnsupportedOperatorError, "i32"):
            self.import_exported(ep)

    def test_int64_input_beyond_i32_is_rejected_at_entry(self):
        """i64 → i32 是收窄：越界值在入口报错，不静默包成别的下标。"""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(10, 4)

            def forward(self, ids):
                return self.emb(ids)

        ids = torch.tensor([1, 2], dtype=torch.int64)
        imported = self.import_exported(torch.export.export(M().eval(), (ids,)))
        feed = imported.feed(ids=np.array([[2 ** 31]], dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "i32|范围|越界"):
            self.ref.execute(imported.graph, feed)


    # ------------------------------------------------------------------
    # bool 掩码链路：比较 / where / masked_fill
    # ------------------------------------------------------------------

    def _roundtrip_args(self, model, args):
        """多输入版本：按位置喂 numpy，逐位与 eager 比对。"""
        ep = torch.export.export(model, args)
        imported = self.import_exported(ep)
        feed = imported.feed(
            **{name: arg.detach().numpy() for name, arg in zip(imported.user_inputs, args)}
        )
        with torch.no_grad():
            expected = model(*args).numpy()
        got = list(self.ref.execute(imported.graph, feed).values())[0]
        np.testing.assert_array_equal(got, expected)
        return imported

    def test_masked_fill_lowers_to_where(self):
        class M(nn.Module):
            def forward(self, x, mask):
                return torch.masked_fill(x, mask, 7.5)

        imported = self._roundtrip_args(
            M().eval(), (torch.randn(3, 4), torch.rand(3, 4) > 0.5)
        )
        names = [n.op.name for n in imported.graph.nodes]
        self.assertIn("where", names)
        self.assertNotIn("masked_fill", names)

    def test_compare_then_where(self):
        class M(nn.Module):
            def forward(self, x, y):
                return torch.where(x > y, x - y, y - x)

        imported = self._roundtrip_args(M().eval(), (torch.randn(3, 4), torch.randn(3, 4)))
        names = [n.op.name for n in imported.graph.nodes]
        self.assertIn("gt", names)
        self.assertIn("where", names)

    def test_bool_input_flows_through_where(self):
        class M(nn.Module):
            def forward(self, x, mask):
                return torch.where(mask, x, torch.zeros_like(x))

        imported = self._roundtrip_args(
            M().eval(), (torch.randn(2, 3), torch.rand(2, 3) > 0.5)
        )
        # 掩码是用户输入，导出的图里应保留 bool 声明的 placeholder
        mask = imported.graph.find("mask")
        self.assertEqual(mask.type.dtype.value, "bool")

    def test_where_requires_bool_condition(self):
        class M(nn.Module):
            def forward(self, c, x, y):
                return torch.where(c > 0, x, y)

        self._roundtrip_args(
            M().eval(),
            (torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)),
        )


@unittest.skipUnless(TORCH_AVAILABLE and CUDA_AVAILABLE, "需要 torch 与 GPU")
class TestTorchExportOnCuda(unittest.TestCase):
    def test_block_matches_on_gpu(self):
        from aetherflow.backends.cuda import CudaBackend
        from aetherflow.frontend import import_exported

        class Block(nn.Module):
            def __init__(self, d=32):
                super().__init__()
                self.ln = nn.LayerNorm(d)
                self.proj = nn.Linear(d, d)

            def forward(self, x):
                return F.silu(self.proj(self.ln(x)))

        model = Block().eval()
        x = torch.randn(2, 8, 32)
        imported = import_exported(torch.export.export(model, (x,)))
        feed = imported.feed(x=x.numpy())

        ref = ReferenceBackend().execute(imported.graph, feed)
        cuda = CudaBackend().execute(imported.graph, feed)
        for k in ref:
            np.testing.assert_allclose(cuda[k], ref[k], rtol=1e-4, atol=1e-4)

    def test_cnn_matches_on_gpu(self):
        """CNN 的真正验收：torch 导出 -> IR -> CUDA 与参考后端逐点一致。"""
        from aetherflow.backends.cuda import CudaBackend
        from aetherflow.frontend import import_exported

        class LeNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv2d(1, 6, 5)
                self.c2 = nn.Conv2d(6, 16, 5)
                self.fc = nn.Linear(16 * 5 * 5, 10)

            def forward(self, x):
                x = F.max_pool2d(F.relu(self.c1(x)), 2)
                x = F.max_pool2d(F.relu(self.c2(x)), 2)
                return self.fc(torch.flatten(x, 1))

        model = LeNet().eval()
        x = torch.randn(2, 1, 32, 32)
        imported = import_exported(torch.export.export(model, (x,)))
        feed = imported.feed(x=x.numpy())

        ref = ReferenceBackend().execute(imported.graph, feed)
        cuda = CudaBackend().execute(imported.graph, feed)
        for k in ref:
            np.testing.assert_allclose(cuda[k], ref[k], rtol=1e-4, atol=1e-4)

        with torch.no_grad():
            np.testing.assert_allclose(
                list(ref.values())[0], model(x).numpy(), rtol=1e-4, atol=1e-4
            )


class TestTorchCompatMatrix(unittest.TestCase):
    """AF-32 补：torch 兼容矩阵把"验证过的版本 / dtype"写成可查询的声明。

    矩阵的意义不是"多支持几个版本"，而是让**矩阵之外**的版本显式失败：
    ``torch.export`` 的 IR 细节会随版本变，没跑过就是没跑过。
    """

    def test_matrix_rows_are_well_formed(self):
        """区间有序、verified 版本落在区间内、dtype 都是 IR 认得的名字。"""
        from aetherflow.frontend import TORCH_MATRIX, parse_torch_version
        from aetherflow.frontend.torch_export import _TORCH_DTYPE

        self.assertTrue(TORCH_MATRIX)
        for row in TORCH_MATRIX:
            self.assertLessEqual(row.min_version, row.max_version)
            self.assertTrue(row.dtypes)
            for name in row.dtypes:
                self.assertIn(name, _TORCH_DTYPE, msg=f"{name} 没有落到 IR dtype")
            for text in row.verified:
                version = parse_torch_version(text)
                self.assertTrue(row.covers(version), msg=f"{text} 不在 {row.span} 内")

    def test_parse_version_accepts_build_and_prerelease_suffixes(self):
        """``2.13.0+cpu`` / ``2.9.0a0`` / ``2.13`` 都解析成三元组。"""
        from aetherflow.frontend import parse_torch_version

        self.assertEqual(parse_torch_version("2.13.0+cpu"), (2, 13, 0))
        self.assertEqual(parse_torch_version("2.9.0a0"), (2, 9, 0))
        self.assertEqual(parse_torch_version(" 2.13 "), (2, 13, 0))
        for bad in ("", "torch", "v2"):
            with self.subTest(text=bad):
                with self.assertRaisesRegex(ValueError, "版本号"):
                    parse_torch_version(bad)

    def test_out_of_matrix_version_is_rejected(self):
        """矩阵之外的版本抛 UnsupportedTorchVersion，并带上已验证区间。"""
        from aetherflow.frontend import UnsupportedTorchVersion, matrix_row

        with self.assertRaisesRegex(UnsupportedTorchVersion, "兼容矩阵"):
            matrix_row((9, 9, 0))
        with self.assertRaisesRegex(UnsupportedTorchVersion, r"2\.9"):
            matrix_row((1, 0, 0))

    def test_check_torch_validates_the_module_version(self):
        """check_torch 只认版本号，不认"装了就行"。"""
        from types import SimpleNamespace

        from aetherflow.frontend import UnsupportedTorchVersion, check_torch

        self.assertIsNotNone(check_torch(SimpleNamespace(__version__="2.13.0+cpu")))
        with self.assertRaises(UnsupportedTorchVersion):
            check_torch(SimpleNamespace(__version__="1.7.1"))
        with self.assertRaisesRegex(ValueError, "版本号"):
            check_torch(SimpleNamespace(__version__="nightly"))

    @unittest.skipUnless(TORCH_AVAILABLE, "需要 torch")
    def test_installed_torch_is_verified(self):
        """本机装的 torch 必须落在矩阵内，且版本被标记为 verified 之外也不静默。"""
        from aetherflow.frontend import check_torch, parse_torch_version

        row = check_torch(torch)
        version = parse_torch_version(torch.__version__)
        self.assertTrue(row.covers(version))

    @unittest.skipUnless(TORCH_AVAILABLE, "需要 torch")
    def test_dtype_coverage_matches_matrix(self):
        """矩阵内的 dtype（含 bool 掩码）能导出；矩阵外显式拒绝。"""
        from aetherflow.frontend import TORCH_MATRIX
        from aetherflow.frontend.torch_export import (
            UnsupportedOperatorError,
            _require_torch,
            _to_dtype,
        )
        from aetherflow.ir.types import bf16, boolean, i32

        row = _require_torch()
        self.assertIn(row.span, "、".join(r.span for r in TORCH_MATRIX))
        self.assertIs(_to_dtype(torch.bfloat16), bf16)
        # IR 只有 i32：int32 无损、int64 收窄（越界值在入口拒绝）
        self.assertIs(_to_dtype(torch.int32), i32)
        self.assertIs(_to_dtype(torch.int64), i32)
        # bool 只作掩码载体（比较 / where / masked_fill 的条件）
        self.assertIs(_to_dtype(torch.bool), boolean)
        with self.assertRaisesRegex(UnsupportedOperatorError, "dtype"):
            _to_dtype(torch.complex64)


if __name__ == "__main__":
    unittest.main()
