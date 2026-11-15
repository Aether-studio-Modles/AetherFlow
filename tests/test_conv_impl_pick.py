"""auto 模式下 conv2d 的实现选择（不碰设备）。

5.238：HIP 上中等通道 3x3/5x5 的隐式 GEMM 比 im2col 快 1.26～1.63×，所以
把"K 超过分块阈值、核面积大于 1"的 auto 决策交给厂商开关
``CONV_AUTO_IMPLICIT``；CUDA 保持 5.151 的结论（隐式在 K >= 288 输给
im2col 1.37～2.32×）。判据本身是纯算术，这里把它钉在纯函数与类属性上。
"""

import unittest

from aetherflow.backends.cuda import CudaBackend
from aetherflow.backends.device_backend import DeviceBackend
from aetherflow.backends.hip.backend import HipBackend
from aetherflow.lower.plan import (
    CONV_ROC_MAX,
    CONV_SHARED_LIMIT,
    conv_implicit_fits,
    conv_implicit_oc_tile,
    plan_conv_implicit,
)


class TestConvImplicitFit(unittest.TestCase):
    """隐式 GEMM 的可行性判据：预算与 ``plan_conv_implicit`` 是同一套。"""

    def test_mid_channel_3x3_fits(self):
        """中等通道 3x3 全都能接住——这正是 5.238 要覆盖的形状。"""
        for c in (8, 16, 24, 32):
            self.assertTrue(
                conv_implicit_fits(
                    kh=3, kw=3, c=c, o=32, wo=64, sw=1, dw=1
                ),
                msg=f"c={c}",
            )

    def test_mid_channel_5x5_fits(self):
        self.assertTrue(
            conv_implicit_fits(kh=5, kw=5, c=16, o=32, wo=64, sw=1, dw=1)
        )

    def test_tile_matches_plan(self):
        """toc 的选择必须与 plan 完全一致，否则 fits 会为一次必然失败的 launch 放行。"""
        plan = plan_conv_implicit(kh=3, kw=3, c=16, o=32, wo=64, sw=1, dw=1)
        span = plan.tile_w + 2  # tile_w * 1 + (3 - 1) * 1
        self.assertEqual(conv_implicit_oc_tile(3, 3, 32, span), plan.toc)
        self.assertLessEqual(plan.shared_bytes, CONV_SHARED_LIMIT)

    def test_window_competes_with_weights(self):
        """窗口也算预算：大核上 toc 会被窗口挤小，只看权重会高估。

        kh=32、kw=48（每个 OC 1536 个 float）、tile_w=32 时窗口占 32*79 个
        float：只看权重能给到 toc=8，加上窗口只放得下 4。
        """
        toc = conv_implicit_oc_tile(32, 48, 32, 79)
        self.assertEqual(toc, 4)
        self.assertLessEqual(
            toc * 32 * 48 * 4 + 32 * 79 * 4, CONV_SHARED_LIMIT
        )
        self.assertGreater(
            8 * 32 * 48 * 4 + 32 * 79 * 4, CONV_SHARED_LIMIT
        )
        self.assertLessEqual(8 * 32 * 48 * 4, CONV_SHARED_LIMIT)

    def test_oversized_kernel_does_not_fit(self):
        """一个输出通道的权重加窗口就超过 48 KB：任何一种 toc 都放不下。"""
        self.assertFalse(
            conv_implicit_fits(kh=120, kw=120, c=4, o=8, wo=64, sw=1, dw=1)
        )

    def test_rejects_empty_and_negative_dimensions(self):
        bad = (
            dict(kh=0, kw=3, c=4, o=4, wo=4, sw=1, dw=1),
            dict(kh=3, kw=3, c=0, o=4, wo=4, sw=1, dw=1),
            dict(kh=3, kw=3, c=4, o=0, wo=4, sw=1, dw=1),
            dict(kh=3, kw=3, c=4, o=4, wo=0, sw=1, dw=1),
            dict(kh=3, kw=3, c=4, o=4, wo=4, sw=0, dw=1),
            dict(kh=3, kw=3, c=4, o=4, wo=4, sw=1, dw=-1),
            dict(kh=3, kw=3, c=4, o=4, wo=4, sw=1, dw=1, tile_w=0),
        )
        for kwargs in bad:
            self.assertFalse(conv_implicit_fits(**kwargs), msg=str(kwargs))

    def test_roc_default_stays_within_block_limit(self):
        """fits 也要守住 block 上限，否则 auto 会选出一个 launch 就失败的实现。"""
        self.assertTrue(CONV_ROC_MAX >= 1)
        plan = plan_conv_implicit(kh=3, kw=3, c=32, o=32, wo=64, sw=1, dw=1)
        self.assertLessEqual(plan.tile_w * plan.block_y, 1024)


class TestAutoConvImplPick(unittest.TestCase):
    """auto 的厂商差异：HIP 选隐式，CUDA 保持 im2col。

    用 ``__new__`` 绕过 ``__init__``：这里只验判据，不建设备上下文。
    """

    @staticmethod
    def _backend(cls, impl: str = "auto"):
        backend = cls.__new__(cls)
        backend.conv_impl = impl
        backend._conv_tile = None
        backend._conv_roc = None
        backend._conv_cb = None
        return backend

    def test_vendor_switch(self):
        self.assertFalse(DeviceBackend.CONV_AUTO_IMPLICIT)
        self.assertFalse(CudaBackend.CONV_AUTO_IMPLICIT)
        self.assertTrue(HipBackend.CONV_AUTO_IMPLICIT)

    def test_hip_auto_picks_implicit_for_mid_channel(self):
        """K > 32 的 3x3/5x5 在 HIP 上交给隐式 GEMM。"""
        backend = self._backend(HipBackend)
        cases = (
            dict(n=4, o=32, k=8 * 9, kh=3, kw=3, ho=64, wo=64, c=8),
            dict(n=4, o=32, k=16 * 9, kh=3, kw=3, ho=64, wo=64, c=16),
            dict(n=4, o=32, k=32 * 9, kh=3, kw=3, ho=64, wo=64, c=32),
            dict(n=4, o=32, k=16 * 25, kh=5, kw=5, ho=64, wo=64, c=16),
        )
        for kwargs in cases:
            self.assertEqual(
                backend._pick_conv_kernel(sw=1, dw=1, **kwargs),
                "implicit",
                msg=str(kwargs),
            )

    def test_hip_auto_keeps_im2col_for_1x1(self):
        """1x1 没有窗口重叠，HIP 实测隐式慢 6%，auto 不该改道。"""
        backend = self._backend(HipBackend)
        self.assertEqual(
            backend._pick_conv_kernel(
                n=2, o=256, k=256, kh=1, kw=1, ho=28, wo=28, c=256, sw=1, dw=1
            ),
            "im2col",
        )

    def test_hip_auto_keeps_tiled_for_small_k(self):
        """K <= 32 仍在分块直算的甜点区。"""
        backend = self._backend(HipBackend)
        self.assertEqual(
            backend._pick_conv_kernel(
                n=2, o=16, k=27, kh=3, kw=3, ho=64, wo=64, c=3, sw=1, dw=1
            ),
            "tiled",
        )

    def test_cuda_auto_keeps_im2col_for_mid_channel(self):
        """CUDA 的 auto 不变：同样的形状仍是 im2col（5.151 的结论）。"""
        backend = self._backend(CudaBackend)
        self.assertEqual(
            backend._pick_conv_kernel(
                n=4, o=32, k=16 * 9, kh=3, kw=3, ho=64, wo=64, c=16, sw=1, dw=1
            ),
            "im2col",
        )

    def test_explicit_impl_is_not_rewritten(self):
        """显式指定 im2col / tiled 时，厂商开关不能把它改掉。"""
        for impl, want in (("im2col", "im2col"), ("tiled", "tiled")):
            backend = self._backend(HipBackend, impl)
            self.assertEqual(
                backend._pick_conv_kernel(
                    n=4, o=32, k=16 * 9, kh=3, kw=3, ho=64, wo=64,
                    c=16, sw=1, dw=1,
                ),
                want,
            )


if __name__ == "__main__":
    unittest.main()
