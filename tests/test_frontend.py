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
                return torch.tanh(x)

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


if __name__ == "__main__":
    unittest.main()
