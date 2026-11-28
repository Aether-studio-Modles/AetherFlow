"""5.334 设备端原生 NHWC：内核描述、签名与入口拒绝。

设备上的数值正确性由 ``test_layout.py::TestHipNativeNHWC`` 在真机上比对；
这里覆盖无 GPU 也能验证的三层：

1. 内核描述换的是**寻址**而不是数学——同名内核只把通道偏移从
   ``* H * W`` 换成"相邻"；
2. 签名按布局分开，NCHW 的旧签名不带 ``layout`` 字段，库里既有记录不失效；
3. 不支持实现（im2col / 隐式 / Winograd / 分组 / f16）在入口显式失败。
"""

import unittest

import numpy as np

from aetherflow.backends.base import BackendCapabilityError
from aetherflow.backends.cuda import codegen
from aetherflow.backends.device_backend import DeviceBackend
from aetherflow.lower.kernels.conv import conv2d_tiled
from aetherflow.lower.kernels.pool import avgpool2d, maxpool2d
from aetherflow.tuning.signature import conv2d_signature


def _src(kernel):
    return codegen.render(kernel, np.float32)


class TestKernelDescriptions(unittest.TestCase):
    """内核名与寻址：NHWC 版不是第二套数学，只换步长。"""

    def test_names_carry_layout_suffix(self):
        self.assertEqual(conv2d_tiled().name, "conv2d_tiled")
        self.assertEqual(conv2d_tiled(layout="nhwc").name, "conv2d_tiled_nhwc")
        self.assertEqual(maxpool2d().name, "maxpool2d")
        self.assertEqual(maxpool2d(layout="nhwc").name, "maxpool2d_nhwc")
        self.assertEqual(avgpool2d().name, "avgpool2d")
        self.assertEqual(avgpool2d(layout="nhwc").name, "avgpool2d_nhwc")

    def test_unknown_layout_is_refused(self):
        for factory in (conv2d_tiled, maxpool2d, avgpool2d):
            with self.assertRaises(ValueError):
                factory(layout="chw")

    def test_conv_channel_offset_becomes_contiguous(self):
        """通道偏移：NCHW 每步跳 ``H*W``，NHWC 是相邻地址。"""
        nchw = _src(conv2d_tiled())
        nhwc = _src(conv2d_tiled(layout="nhwc"))
        self.assertIn("xn + (long long)(c0 + cc) * H * W", nchw)
        self.assertIn("xn + (long long)(c0 + cc);", nhwc)

    def test_conv_output_writeback_moves_channel_to_last(self):
        nchw = _src(conv2d_tiled())
        nhwc = _src(conv2d_tiled(layout="nhwc"))
        self.assertIn("((long long)n * O + oc) * HO * WO", nchw)
        self.assertIn("((long long)n * HO + ho) * WO * O + wo * O + oc", nhwc)

    def test_pool_base_and_tap_follow_nhwc(self):
        nchw = _src(maxpool2d())
        nhwc = _src(maxpool2d(layout="nhwc"))
        self.assertIn("x + ((long long)n * C + c) * (long long)H * W", nchw)
        self.assertIn("x + ((long long)n) * (long long)H * W * C + c", nhwc)
        self.assertIn("base[(long long)ih * W + iw]", nchw)
        self.assertIn("((long long)ih * W + iw) * (long long)C", nhwc)

    def test_avgpool_nhwc_keeps_window_semantics(self):
        """池化的窗口 / 分母逻辑与布局无关，只换下标。"""
        nhwc = _src(avgpool2d(layout="nhwc"))
        self.assertIn("avgpool2d_nhwc", nhwc)
        self.assertIn("count_include_pad", nhwc)


class TestSignature(unittest.TestCase):
    """布局进签名：同一形状的两种排列不能共用调优结果。"""

    def test_nchw_keeps_legacy_signature(self):
        default = conv2d_signature((1, 3, 8, 8), (4, 3, 3, 3))
        explicit = conv2d_signature((1, 3, 8, 8), (4, 3, 3, 3), layout="nchw")
        self.assertNotIn("layout", default)
        self.assertEqual(default, explicit)

    def test_nhwc_carries_layout(self):
        sig = conv2d_signature((1, 8, 8, 3), (4, 3, 3, 3), layout="nhwc")
        self.assertEqual(sig["layout"], "nhwc")

    def test_unknown_layout_is_refused(self):
        with self.assertRaises(ValueError):
            conv2d_signature((1, 8, 8, 3), (4, 3, 3, 3), layout="chw")


def _bare(**overrides):
    """只要入口检查能跑到的最小后端：不需要设备，也不需要内核缓存。"""
    backend = object.__new__(DeviceBackend)
    backend.cp = np
    backend.conv_impl = "auto"
    for key, value in overrides.items():
        setattr(backend, key, value)
    return backend


class TestEntryRefusals(unittest.TestCase):
    """覆盖面收窄要显式说明，不能静默换布局或换实现。"""

    def _call(self, backend, x, weight, **attrs):
        return backend._conv2d_nhwc(x, weight, None, attrs)

    def test_non_tiled_impl_is_refused(self):
        x = np.zeros((1, 8, 8, 3), np.float32)
        weight = np.zeros((4, 3, 3, 3), np.float32)
        with self.assertRaises(BackendCapabilityError) as ctx:
            self._call(_bare(conv_impl="im2col"), x, weight)
        self.assertIn("NHWC", str(ctx.exception))
        self.assertIn("nchw", str(ctx.exception))

    def test_groups_is_refused(self):
        x = np.zeros((1, 8, 8, 3), np.float32)
        weight = np.zeros((4, 3, 3, 3), np.float32)
        with self.assertRaises(BackendCapabilityError) as ctx:
            self._call(_bare(), x, weight, groups=2)
        self.assertIn("groups=1", str(ctx.exception))

    def test_float16_is_refused(self):
        x = np.zeros((1, 8, 8, 3), np.float16)
        weight = np.zeros((4, 3, 3, 3), np.float16)
        with self.assertRaises(BackendCapabilityError) as ctx:
            self._call(_bare(), x, weight)
        self.assertIn("float32", str(ctx.exception))

    def test_channel_mismatch_is_refused(self):
        x = np.zeros((1, 8, 8, 3), np.float32)
        weight = np.zeros((4, 5, 3, 3), np.float32)
        with self.assertRaises(BackendCapabilityError) as ctx:
            self._call(_bare(), x, weight, padding=1)
        self.assertIn("不一致", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
