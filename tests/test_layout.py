"""通道在后（NHWC）布局的测试：IR 语义、后端声明与传播 pass。

三条主线：

1. **语义逐位一致**：``layout="nhwc"`` 不是第二套数学，参考执行结果必须与
   同一张图的 NCHW 版本完全相同（用 ``assert_array_equal``，不是 allclose）。
2. **传播只搬不改**：翻转后的图与翻转前的图在同样输入下输出一致，形状仍是
   NCHW；转置只出现在区域边界，数得清。
3. **不静默降级**：非法 layout、没有反向规则、后端不承接 NHWC 都必须显式
   报错，而不是算出一份"看着有值"的结果。
"""

import unittest

import numpy as np

from aetherflow import Graph, ReferenceBackend, backward, f32
from aetherflow.backends.base import (
    BackendCapabilityError,
    DEFAULT_SUPPORTED_LAYOUTS,
    supported_layouts_of,
)
from aetherflow.backends.device_backend import DeviceBackend
from aetherflow.backends.hip import HIP_AVAILABLE, HipBackend
from aetherflow.ir.ops import NoGradientError
from aetherflow.passes import LayoutError, LayoutPropagation

_RNG = np.random.default_rng(246)


def _conv_graph(layout=None, activation=True):
    """x(2,3,8,8) -> conv(4 通道,3x3,s=1,p=1) [-> relu]，输出 NCHW。"""
    attrs = {"stride": 1, "padding": 1}
    if layout is not None:
        attrs["layout"] = layout
    g = Graph()
    x_shape = (2, 8, 8, 3) if layout == "nhwc" else (2, 3, 8, 8)
    x = g.input("x", f32[x_shape])
    w = g.input("w", f32[4, 3, 3, 3])
    y = g.conv2d(x, w, **attrs)
    if activation:
        y = g.relu(y)
    g.output(y, "y")
    return g, x, w


def _feeds(shape=(2, 3, 8, 8)):
    return {
        "x": _RNG.standard_normal(shape).astype(np.float32),
        "w": _RNG.standard_normal((4, 3, 3, 3)).astype(np.float32),
    }


class TestLayoutSemantics(unittest.TestCase):
    """IR 层：NHWC 与 NCHW 是同一份数学的两种排列。"""

    def test_conv2d_nhwc_matches_nchw_bitwise(self):
        feed = _feeds()
        nchw, _, _ = _conv_graph()
        nhwc, _, _ = _conv_graph(layout="nhwc")
        ref = ReferenceBackend()

        # NHWC 图的输入是同一份数据的转置视图；比较时把输出转回 NCHW。
        feed_nhwc = {"x": np.transpose(feed["x"], (0, 2, 3, 1)), "w": feed["w"]}
        got = ref.execute(nhwc, feed_nhwc)["y"]
        want = ref.execute(nchw, feed)["y"]
        self.assertEqual(got.shape, (2, 8, 8, 4))
        np.testing.assert_array_equal(np.transpose(got, (0, 3, 1, 2)), want)

    def test_maxpool2d_nhwc_matches_nchw_bitwise(self):
        g_nchw = Graph()
        x = g_nchw.input("x", f32[2, 3, 8, 8])
        g_nchw.output(g_nchw.maxpool2d(x, kernel=2, stride=2), "y")
        g_nhwc = Graph()
        xh = g_nhwc.input("x", f32[2, 8, 8, 3])
        g_nhwc.output(g_nhwc.maxpool2d(xh, kernel=2, stride=2, layout="nhwc"), "y")

        data = _RNG.standard_normal((2, 3, 8, 8)).astype(np.float32)
        ref = ReferenceBackend()
        want = ref.execute(g_nchw, {"x": data})["y"]
        got = ref.execute(g_nhwc, {"x": np.transpose(data, (0, 2, 3, 1))})["y"]
        self.assertEqual(got.shape, (2, 4, 4, 3))
        np.testing.assert_array_equal(np.transpose(got, (0, 3, 1, 2)), want)

    def test_avgpool2d_nhwc_matches_nchw_bitwise(self):
        g_nchw = Graph()
        x = g_nchw.input("x", f32[2, 3, 6, 6])
        g_nchw.output(g_nchw.avgpool2d(x, kernel=3, stride=1, padding=1), "y")
        g_nhwc = Graph()
        xh = g_nhwc.input("x", f32[2, 6, 6, 3])
        g_nhwc.output(
            g_nhwc.avgpool2d(xh, kernel=3, stride=1, padding=1, layout="nhwc"), "y"
        )

        data = _RNG.standard_normal((2, 3, 6, 6)).astype(np.float32)
        ref = ReferenceBackend()
        want = ref.execute(g_nchw, {"x": data})["y"]
        got = ref.execute(g_nhwc, {"x": np.transpose(data, (0, 2, 3, 1))})["y"]
        np.testing.assert_array_equal(np.transpose(got, (0, 3, 1, 2)), want)

    def test_nhwc_infer_keeps_weight_layout(self):
        """权重与 bias 不随布局翻转：只有激活的维度顺序变了。"""
        g = Graph()
        x = g.input("x", f32[2, 8, 8, 3])
        w = g.input("w", f32[5, 3, 3, 3])
        b = g.input("b", f32[5])
        y = g.conv2d(x, w, b, stride=2, padding=1, layout="nhwc")
        self.assertEqual(tuple(y.type.shape), (2, 4, 4, 5))

    def test_illegal_layout_is_rejected(self):
        g = Graph()
        x = g.input("x", f32[2, 3, 8, 8])
        w = g.input("w", f32[4, 3, 3, 3])
        with self.assertRaises(ValueError) as ctx:
            g.conv2d(x, w, layout="chw")
        self.assertIn("layout", str(ctx.exception))

    def test_nhwc_forward_has_no_backward_rule(self):
        """NHWC 是前向布局，反向规则定义在 NCHW 上，必须显式拒绝。"""
        g = Graph()
        x = g.input("x", f32[2, 8, 8, 3])
        w = g.input("w", f32[4, 3, 3, 3])
        g.output(g.conv2d(x, w, padding=1, layout="nhwc"), "y")
        with self.assertRaises(NoGradientError) as ctx:
            backward(g)
        self.assertIn("nhwc", str(ctx.exception))


class TestLayoutDeclaration(unittest.TestCase):
    """后端布局声明：参考后端两种都接，设备后端只接 NCHW。"""

    def test_reference_backend_declares_both(self):
        self.assertEqual(
            ReferenceBackend().supported_layouts, frozenset({"nchw", "nhwc"})
        )

    def test_device_backend_declares_nchw_only(self):
        self.assertEqual(DeviceBackend.supported_layouts, frozenset({"nchw"}))

    def test_undeclared_backend_falls_back_to_nchw(self):
        """没写布局声明的后端按最小承诺处理，而不是"默认什么都行"。"""

        class _Old:
            name = "old"

        self.assertEqual(supported_layouts_of(_Old()), DEFAULT_SUPPORTED_LAYOUTS)
        self.assertEqual(supported_layouts_of(_Old()), frozenset({"nchw"}))


class TestLayoutPropagation(unittest.TestCase):
    """传播 pass：区域边界插转置，区域内部改属性，数值不变。"""

    def _ops(self, graph):
        return [n.op.name for n in graph.nodes]

    def _transposes(self, graph):
        return [n for n in graph.nodes if n.op.name == "transpose"]

    def test_single_conv_is_flipped(self):
        g, _, _ = _conv_graph(activation=False)
        out = LayoutPropagation().run(g)
        self.assertEqual(self._ops(out), ["transpose", "conv2d", "transpose"])
        conv = out.nodes[1]
        self.assertEqual(conv.attrs["layout"], "nhwc")
        self.assertEqual(tuple(conv.output.type.shape), (2, 8, 8, 4))

    def test_chain_is_flipped_as_one_region(self):
        """conv -> relu -> pool 整段翻，只付两次转置。"""
        g = Graph()
        x = g.input("x", f32[2, 3, 8, 8])
        w = g.input("w", f32[4, 3, 3, 3])
        h = g.relu(g.conv2d(x, w, padding=1))
        g.output(g.maxpool2d(h, kernel=2, stride=2), "y")

        out = LayoutPropagation().run(g)
        self.assertEqual(
            self._ops(out),
            ["transpose", "conv2d", "relu", "maxpool2d", "transpose"],
        )
        self.assertEqual(len(self._transposes(out)), 2)

    def test_flipped_graph_matches_unflipped(self):
        g, _, _ = _conv_graph()
        ref = ReferenceBackend()
        feed = _feeds()
        want = ref.execute(g, feed)["y"]
        got = ref.execute(LayoutPropagation().run(g), feed)["y"]
        self.assertEqual(got.shape, want.shape)
        np.testing.assert_array_equal(got, want)

    def test_flipped_chain_matches_unflipped(self):
        g = Graph()
        x = g.input("x", f32[2, 3, 8, 8])
        w = g.input("w", f32[4, 3, 3, 3])
        h = g.gelu(g.conv2d(x, w, padding=1))
        g.output(g.maxpool2d(h, kernel=2, stride=2), "y")

        ref = ReferenceBackend()
        feed = _feeds()
        want = ref.execute(g, feed)["y"]
        out = LayoutPropagation().run(g)
        got = ref.execute(out, feed)["y"]
        np.testing.assert_array_equal(got, want)
        self.assertIn("gelu", self._ops(out))
        # gelu 与 pool 都被吸收进区域，节点数只多了两次转置。
        self.assertEqual(len(out.nodes), len(g.nodes) + 2)

    def test_graph_output_stays_nchw(self):
        g, _, _ = _conv_graph()
        out = LayoutPropagation().run(g)
        self.assertEqual(tuple(out.outputs[0].type.shape), (2, 4, 8, 8))

    def test_pure_elementwise_is_not_flipped(self):
        g = Graph()
        x = g.input("x", f32[2, 3, 8, 8])
        g.output(g.relu(g.gelu(x)), "y")
        out = LayoutPropagation().run(g)
        self.assertEqual(self._ops(out), ["gelu", "relu"])

    def test_reshape_cuts_the_region(self):
        """reshape 依赖维度顺序，不能跨过它翻；两侧各自成区域。"""
        g = Graph()
        x = g.input("x", f32[2, 3, 8, 8])
        w = g.input("w", f32[4, 3, 3, 3])
        h = g.conv2d(x, w, padding=1)
        r = g.reshape(h, shape=(2, 4, 8, 8))
        w2 = g.input("w2", f32[6, 4, 3, 3])
        y = g.conv2d(r, w2, padding=1)
        g.output(g.relu(y), "y")

        out = LayoutPropagation().run(g)
        self.assertEqual(
            self._ops(out),
            [
                "transpose",
                "conv2d",
                "transpose",
                "reshape",
                "transpose",
                "conv2d",
                "relu",
                "transpose",
            ],
        )

    def test_external_consumer_shares_one_exit_transpose(self):
        """区域输出同时被区域内与区域外消费时，只插一次出口转置。"""
        g = Graph()
        x = g.input("x", f32[2, 3, 8, 8])
        w = g.input("w", f32[4, 3, 3, 3])
        h = g.conv2d(x, w, padding=1)
        g.output(g.relu(h), "y")
        g.output(g.reduce_sum(h, axis=(1, 2, 3), keepdims=True), "s")

        out = LayoutPropagation().run(g)
        # 入口一次；conv 的输出有两个消费者（relu 在区域内、reduce_sum 在
        # 区域外），出口仍只转一次；relu 的输出是图输出，再转一次。
        self.assertEqual(len(self._transposes(out)), 3)
        ref = ReferenceBackend()
        feed = _feeds()
        want = ref.execute(g, feed)
        got = ref.execute(out, feed)
        for name in want:
            np.testing.assert_array_equal(got[name], want[name])

    def test_weight_from_flippable_node_is_not_propagated(self):
        """权重来自可翻转节点时，绝不能把它当成激活去转置。"""
        g = Graph()
        x = g.input("x", f32[2, 3, 8, 8])
        w_in = g.input("w", f32[4, 3, 3, 3])
        w = g.relu(w_in)  # 逐元素节点本身可翻，但它的值在这里是**参数**
        y = g.conv2d(x, w, padding=1)
        g.output(y, "y")

        out = LayoutPropagation().run(g)
        # 这个卷积被剔除候选（权重来自可翻转节点）；relu 没有 spatial 邻居，
        # 也不成区域。整张图保持原样，代价是不做优化而不是算错。
        self.assertEqual(self._ops(out), ["relu", "conv2d"])

    def test_disjoint_regions_each_pay_two_transposes(self):
        g = Graph()
        x1 = g.input("x1", f32[2, 3, 8, 8])
        x2 = g.input("x2", f32[2, 3, 8, 8])
        w1 = g.input("w1", f32[4, 3, 3, 3])
        w2 = g.input("w2", f32[4, 3, 3, 3])
        g.output(g.conv2d(x1, w1, padding=1), "y1")
        g.output(g.conv2d(x2, w2, padding=1), "y2")

        out = LayoutPropagation().run(g)
        self.assertEqual(len(self._transposes(out)), 4)

    def test_backend_without_nhwc_is_refused(self):
        g, _, _ = _conv_graph()
        with self.assertRaises(LayoutError) as ctx:
            LayoutPropagation(supported_layouts={"nchw"}).run(g)
        self.assertIn("nhwc", str(ctx.exception))

    def test_graph_without_spatial_ops_needs_no_declaration(self):
        """没有可翻区域时不要求后端声明 NHWC。"""
        g = Graph()
        x = g.input("x", f32[4])
        g.output(g.relu(x), "y")
        out = LayoutPropagation(supported_layouts={"nchw"}).run(g)
        self.assertEqual(self._ops(out), ["relu"])

    def test_pass_does_not_mutate_input_graph(self):
        g, _, _ = _conv_graph()
        before = self._ops(g)
        LayoutPropagation().run(g)
        self.assertEqual(self._ops(g), before)
        self.assertEqual(g.nodes[1].attrs.get("layout", "nchw"), "nchw")


@unittest.skipUnless(HIP_AVAILABLE, "需要 HIP 运行时（ROCm）")
class TestHipRejectsNHWC(unittest.TestCase):
    """设备端没有原生 NHWC 内核，必须显式拒绝而不是按 NCHW 读内存。"""

    def test_nhwc_conv_is_a_capability_error(self):
        g = Graph()
        x = g.input("x", f32[2, 8, 8, 3])
        w = g.input("w", f32[4, 3, 3, 3])
        g.output(g.conv2d(x, w, padding=1, layout="nhwc"), "y")
        backend = HipBackend()
        feed = {
            "x": _RNG.standard_normal((2, 8, 8, 3)).astype(np.float32),
            "w": _RNG.standard_normal((4, 3, 3, 3)).astype(np.float32),
        }
        with self.assertRaises(BackendCapabilityError) as ctx:
            backend.execute(g, feed)
        self.assertIn("NHWC", str(ctx.exception))
        self.assertIn("NCHW", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
