"""执行计划层的离线测试：不建 CUDA 上下文、不需要 GPU。

这一层的错误有两种表现，都很差：预算算小了是驱动报
``invalid configuration argument``（看不出是哪一维），预算算大了是静默
多占 shared memory。所以边界不靠"跑起来试试"，而是逐条钉在纯函数上。
"""

import unittest

from aetherflow.lower.plan import (
    BLOCK_DEFAULT,
    BLOCK_MAX,
    BLOCK_MIN,
    CONV_SHARED_LIMIT,
    CONV_TILE_OPTIONS,
    CSR_SCAN_MAX,
    CSR_SCAN_SLOTS,
    REDUCE_SCRATCH,
    REDUCE_TILE_X,
    SCAN_CHUNK,
    SCAN_CHUNK_MAX_ELEMS,
    SCAN_LAST_MAX_ELEMS,
    SHARED_LIMIT,
    ConvTilePlan,
    Launch,
    attention_block,
    auto_block,
    can_vectorize,
    check_block,
    conv_oc_tile,
    conv_tiled_preferred,
    grid_for,
    plan_conv_implicit,
    plan_conv_tiled,
    plan_csr_scan,
    plan_csr_scan_enabled,
    plan_elementwise,
    plan_reduce,
    plan_reduce_tiled,
    pow2_block,
    scan_auto_chunk,
    scan_chunk,
    scan_last_auto_chunk,
)


class TestBlockShape(unittest.TestCase):
    def test_pow2_rounds_up(self):
        self.assertEqual(pow2_block(0), 1)
        self.assertEqual(pow2_block(1), 1)
        self.assertEqual(pow2_block(32), 32)
        self.assertEqual(pow2_block(33), 64)
        self.assertEqual(pow2_block(500), 512)

    def test_pow2_capped(self):
        self.assertEqual(pow2_block(5000), BLOCK_MAX)
        self.assertEqual(pow2_block(5000, cap=64), 64)

    def test_check_block_accepts_powers_of_two(self):
        check_block(BLOCK_MIN)
        check_block(BLOCK_MAX)

    def test_check_block_rejects_bad_values(self):
        for bad in (0, 31, 48, BLOCK_MAX * 2, True, "256"):
            with self.assertRaises(ValueError, msg=f"block={bad!r}"):
                check_block(bad)


class TestLaunchValidation(unittest.TestCase):
    def test_rejects_zero_grid(self):
        with self.assertRaises(ValueError):
            Launch(block=256, grid=0)

    def test_rejects_odd_vec(self):
        with self.assertRaises(ValueError):
            Launch(block=256, grid=1, vec=2)

    def test_rejects_shared_over_budget(self):
        with self.assertRaises(ValueError):
            Launch(block=256, grid=1, shared_bytes=SHARED_LIMIT + 1)


class TestVectorize(unittest.TestCase):
    def test_requires_f32_divisible_and_aligned(self):
        self.assertTrue(can_vectorize(64, dtype="f32", aligned=True))

    def test_rejects_f16(self):
        self.assertFalse(can_vectorize(64, dtype="f16", aligned=True))

    def test_rejects_unaligned(self):
        self.assertFalse(can_vectorize(64, dtype="f32", aligned=False))

    def test_rejects_remainder(self):
        self.assertFalse(can_vectorize(63, dtype="f32", aligned=True))

    def test_rejects_empty(self):
        self.assertFalse(can_vectorize(0, dtype="f32", aligned=True))


class TestGridFor(unittest.TestCase):
    def test_empty_still_launches_one_block(self):
        # 零元素时内核什么都不做，但 launch 一个空块比"提前 return"少一个分支
        self.assertEqual(grid_for(0, block=256), 1)

    def test_rounds_up(self):
        self.assertEqual(grid_for(1024, block=256), 4)
        self.assertEqual(grid_for(1025, block=256), 5)

    def test_vec_shrinks_grid(self):
        # 每个线程 4 个元素，块数应当少 4 倍
        self.assertEqual(grid_for(1024, block=256, vec=4), 1)

    def test_capped_by_max_blocks(self):
        self.assertEqual(grid_for(10**9, block=256, max_blocks=1024), 1024)

    def test_rejects_negative_and_bad_vec(self):
        with self.assertRaises(ValueError):
            grid_for(-1, block=256)
        with self.assertRaises(ValueError):
            grid_for(16, block=256, vec=3)


class TestAutoBlock(unittest.TestCase):
    """按规模推 block：上限是 preferred，两条规则都不含测量结论。"""

    def test_large_work_uses_the_default(self):
        self.assertEqual(auto_block(1 << 20), BLOCK_DEFAULT)
        self.assertEqual(auto_block(1 << 20, vec=4), BLOCK_DEFAULT)

    def test_small_work_does_not_spawn_idle_threads(self):
        self.assertEqual(auto_block(0), BLOCK_MIN)
        self.assertEqual(auto_block(1), BLOCK_MIN)
        self.assertEqual(auto_block(33), 64)
        self.assertEqual(auto_block(200), 256)

    def test_vec_divides_the_work(self):
        # 向量化之后每个线程 4 个元素，需要的线程数少 4 倍
        self.assertEqual(auto_block(128, vec=4), BLOCK_MIN)
        self.assertEqual(auto_block(256, vec=4), 64)

    def test_target_blocks_shrinks_to_fill_the_device(self):
        # 8192 个元素、目标 80 块：256 只够 32 块，缩到 64 才有 128 块
        self.assertEqual(auto_block(8192, target_blocks=80), 64)
        # 刚好够一屏时不动：20480 / 256 = 80
        self.assertEqual(auto_block(20480, target_blocks=80), 256)
        # 再小也不会低于一个 warp
        self.assertEqual(auto_block(100, target_blocks=80), BLOCK_MIN)

    def test_never_exceeds_preferred(self):
        self.assertEqual(auto_block(1 << 20, preferred=1024), 1024)
        self.assertEqual(auto_block(4096, preferred=128), 128)
        # preferred 是上限，不是起点：小工作量照样往下缩
        self.assertEqual(auto_block(64, preferred=1024), 64)

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            auto_block(-1)
        with self.assertRaises(ValueError):
            auto_block(16, vec=3)
        with self.assertRaises(ValueError):
            auto_block(16, preferred=48)
        for bad in (0, -4, 2.5, True):
            with self.subTest(target_blocks=bad):
                with self.assertRaises(ValueError):
                    auto_block(1024, target_blocks=bad)

    def test_result_is_always_a_legal_block(self):
        for work in (0, 1, 31, 32, 33, 255, 256, 4095, 4096, 10**6):
            for vec in (1, 4):
                with self.subTest(work=work, vec=vec):
                    block = auto_block(work, vec=vec, target_blocks=48)
                    check_block(block)
                    self.assertGreaterEqual(block, BLOCK_MIN)
                    self.assertLessEqual(block, BLOCK_DEFAULT)


class TestPlanElementwise(unittest.TestCase):
    def test_conservative_without_alignment(self):
        # 默认 aligned=False：没拿到对齐信息就只走标量通路
        launch = plan_elementwise(1024, block=256)
        self.assertEqual((launch.vec, launch.grid), (1, 4))

    def test_vectorized_when_allowed(self):
        launch = plan_elementwise(1024, block=256, aligned=True)
        self.assertEqual((launch.vec, launch.grid), (4, 1))

    def test_f16_never_vectorizes(self):
        launch = plan_elementwise(1024, block=256, dtype="f16", aligned=True)
        self.assertEqual(launch.vec, 1)

    def test_vectorize_can_be_disabled(self):
        launch = plan_elementwise(1024, block=256, aligned=True, vectorize=False)
        self.assertEqual(launch.vec, 1)

    def test_rejects_negative(self):
        with self.assertRaises(ValueError):
            plan_elementwise(-4)

    def test_block_defaults_to_auto(self):
        # 大工作量落到默认值，小工作量按需缩小
        self.assertEqual(plan_elementwise(1 << 20).block, BLOCK_DEFAULT)
        self.assertEqual(plan_elementwise(64).block, 64)

    def test_auto_reaches_the_device_fill_target(self):
        launch = plan_elementwise(8192, target_blocks=80)
        self.assertEqual(launch.block, 64)
        self.assertGreaterEqual(launch.grid, 80)

    def test_explicit_block_is_kept(self):
        # 显式给值就说"我要这么大"，自动不参与
        self.assertEqual(plan_elementwise(64, block=512).block, 512)


class TestPlanReduce(unittest.TestCase):
    def test_block_tracks_columns(self):
        self.assertEqual(plan_reduce(0), 1)
        self.assertEqual(plan_reduce(1), 1)
        self.assertEqual(plan_reduce(1000), 1024)

    def test_block_capped(self):
        self.assertEqual(plan_reduce(4096), BLOCK_MAX)

    def test_custom_cap(self):
        self.assertEqual(plan_reduce(4096, cap=256), 256)

    def test_rejects_negative(self):
        with self.assertRaises(ValueError):
            plan_reduce(-1)


class TestPlanReduceTiled(unittest.TestCase):
    """非末维归约的两维铺法：``tx`` 沿 inner、``ty`` 切 reduce 轴（5.174）。"""

    def test_tx_follows_inner_capped_by_one_warp(self):
        self.assertEqual(plan_reduce_tiled(1, 8), (1, 8))
        self.assertEqual(plan_reduce_tiled(3, 8), (4, 8))
        self.assertEqual(plan_reduce_tiled(32, 8), (32, 8))
        self.assertEqual(plan_reduce_tiled(1000, 8), (REDUCE_TILE_X, 8))

    def test_ty_follows_reduce_length(self):
        self.assertEqual(plan_reduce_tiled(8, 1), (8, 1))
        self.assertEqual(plan_reduce_tiled(8, 5), (8, 8))
        self.assertEqual(plan_reduce_tiled(8, 1000), (8, 128))

    def test_block_never_exceeds_the_hard_limit(self):
        for inner in (1, 3, 17, 32, 64, 4096):
            for reduce_len in (1, 7, 33, 1024, 65536):
                tx, ty = plan_reduce_tiled(inner, reduce_len)
                self.assertGreaterEqual(tx, 1)
                self.assertLessEqual(tx, REDUCE_TILE_X)
                self.assertLessEqual(tx * ty, BLOCK_MAX)
                # 两维都必须是 2 的幂：归约树按 >> 1 折半，否则会漏元素
                self.assertEqual(tx & (tx - 1), 0)
                self.assertEqual(ty & (ty - 1), 0)

    def test_shared_scratch_always_fits(self):
        """``tx * ty`` 就是 shared 暂存槽的使用量，不能超过 ``REDUCE_SCRATCH``。"""
        for inner in (1, 5, 32, 1000):
            for reduce_len in (1, 64, 4096):
                tx, ty = plan_reduce_tiled(inner, reduce_len)
                self.assertLessEqual(tx * ty, REDUCE_SCRATCH)

    def test_rejects_empty_dimensions(self):
        with self.assertRaises(ValueError):
            plan_reduce_tiled(0, 4)
        with self.assertRaises(ValueError):
            plan_reduce_tiled(4, 0)


class TestAttentionBlock(unittest.TestCase):
    def test_floor_is_one_warp(self):
        self.assertEqual(attention_block(0), 32)
        self.assertEqual(attention_block(16), 32)

    def test_rounds_up_to_power_of_two(self):
        self.assertEqual(attention_block(100), 128)

    def test_capped(self):
        self.assertEqual(attention_block(4096), 256)

    def test_rejects_negative(self):
        with self.assertRaises(ValueError):
            attention_block(-1)


class TestConvTiling(unittest.TestCase):
    def test_tile_capped_by_shared_budget(self):
        # 3x3 核时 32 行权重只要 32*9*4 = 1152 B，直接拉满
        self.assertEqual(conv_oc_tile(9, 64), 32)

    def test_tile_capped_by_output_channels(self):
        self.assertEqual(conv_oc_tile(9, 6), 4)

    def test_oversized_kernel_has_no_tile(self):
        self.assertIsNone(conv_oc_tile(CONV_SHARED_LIMIT // 4 + 1, 4))

    def test_plan_for_typical_shape(self):
        tiling = plan_conv_tiled(kh=3, kw=3, c=8, o=64, ho=32, wo=50)
        self.assertIsInstance(tiling, ConvTilePlan)
        self.assertEqual(tiling.tile_w, 32)
        self.assertEqual((tiling.toc, tiling.roc), (32, 4))
        self.assertEqual(tiling.block, (32, 8))
        self.assertEqual(tiling.oc_tiles, 2)
        # CB 受 c 与剩余预算双重约束：这里 c=8 先到顶
        self.assertEqual(tiling.cb, 8)
        self.assertEqual(tiling.shared_bytes, 32 * 8 * 9 * 4)

    def test_tile_width_follows_output_when_narrow(self):
        self.assertEqual(plan_conv_tiled(kh=3, kw=3, c=4, o=8, ho=8, wo=10).tile_w, 10)

    def test_cb_capped_by_budget(self):
        # 7x7 核时一个通道的权重 tile 是 32*49*4 = 6272 B，剩余预算只够 7 个
        # 通道，于是 CB 由预算（7）而不是上限（8）决定
        tiling = plan_conv_tiled(kh=7, kw=7, c=1024, o=32, ho=8, wo=8)
        self.assertEqual(tiling.cb, 7)
        self.assertEqual(tiling.shared_bytes, 32 * 49 * 4 * 7)
        self.assertLessEqual(tiling.shared_bytes, CONV_SHARED_LIMIT)

    def test_oversized_kernel_rejected(self):
        # 128x128 核的权重 tile 是 16384 个 float（64KB），超过 48KB 预算
        with self.assertRaises(ValueError):
            plan_conv_tiled(kh=128, kw=128, c=1, o=1, ho=4, wo=4)

    def test_rejects_empty_dimensions(self):
        with self.assertRaises(ValueError):
            plan_conv_tiled(kh=3, kw=3, c=0, o=4, ho=4, wo=4)

    def test_tile_width_override(self):
        """调优旋钮：显式 tile_w 只改 block 的 x 维，其余推导不动。"""
        default = plan_conv_tiled(kh=3, kw=3, c=8, o=64, ho=32, wo=50)
        narrow = plan_conv_tiled(
            kh=3, kw=3, c=8, o=64, ho=32, wo=50, tile_w=16
        )
        self.assertEqual(narrow.tile_w, 16)
        self.assertEqual(narrow.block, (16, default.block_y))
        self.assertEqual(
            (narrow.toc, narrow.roc, narrow.cb, narrow.shared_bytes),
            (default.toc, default.roc, default.cb, default.shared_bytes),
        )

    def test_tile_width_override_clamped_by_output_width(self):
        """W 比 tile 窄时截断：多出来的线程只会走边界慢路径。"""
        tiling = plan_conv_tiled(kh=3, kw=3, c=4, o=8, ho=8, wo=10, tile_w=64)
        self.assertEqual(tiling.tile_w, 10)

    def test_tile_width_override_rejects_oversized_block(self):
        # block_y = 8，tile_w=256 会让一个 block 达到 2048 个线程
        with self.assertRaises(ValueError):
            plan_conv_tiled(
                kh=3, kw=3, c=8, o=64, ho=8, wo=1024, tile_w=256
            )

    def test_tile_width_must_be_positive_int(self):
        for bad in (0, -8, 16.0, True):
            with self.assertRaises(ValueError):
                plan_conv_tiled(
                    kh=3, kw=3, c=4, o=8, ho=8, wo=8, tile_w=bad
                )

    def test_roc_override_redistributes_block_y(self):
        """调优旋钮：roc 决定 block 的 y 维，toc 与 shared 预算不动。"""
        default = plan_conv_tiled(kh=3, kw=3, c=8, o=64, ho=32, wo=50)
        widest = plan_conv_tiled(kh=3, kw=3, c=8, o=64, ho=32, wo=50, roc=1)
        self.assertEqual(widest.roc, 1)
        self.assertEqual(widest.block, (default.tile_w, default.toc))
        self.assertEqual(widest.shared_bytes, default.shared_bytes)

    def test_roc_must_be_a_divisor_of_the_output_tile(self):
        # toc 是 2 的幂，roc=3 会让 block 的 y 维不整
        with self.assertRaises(ValueError):
            plan_conv_tiled(kh=3, kw=3, c=8, o=64, ho=8, wo=8, roc=3)
        # o=3 时 toc 只有 2，roc=4 超过 tile
        with self.assertRaises(ValueError):
            plan_conv_tiled(kh=3, kw=3, c=4, o=3, ho=8, wo=8, roc=4)

    def test_cb_override_shrinks_the_staging_depth(self):
        tiling = plan_conv_tiled(kh=3, kw=3, c=8, o=64, ho=32, wo=50, cb=2)
        self.assertEqual(tiling.cb, 2)
        self.assertEqual(tiling.shared_bytes, 32 * 9 * 4 * 2)

    def test_cb_rejected_when_over_channels_or_budget(self):
        # c=4 时 staging 不可能覆盖 8 个通道
        with self.assertRaises(ValueError):
            plan_conv_tiled(kh=3, kw=3, c=4, o=64, ho=8, wo=8, cb=8)
        # 7x7 核一个通道占 6272 B，预算只够 7 个
        with self.assertRaises(ValueError):
            plan_conv_tiled(kh=7, kw=7, c=1024, o=32, ho=8, wo=8, cb=8)

    def test_roc_and_cb_must_be_positive_int(self):
        for bad in (0, -1, 2.0, True):
            with self.assertRaises(ValueError):
                plan_conv_tiled(kh=3, kw=3, c=4, o=8, ho=8, wo=8, roc=bad)
            with self.assertRaises(ValueError):
                plan_conv_tiled(kh=3, kw=3, c=4, o=8, ho=8, wo=8, cb=bad)

    def test_every_offered_option_plans_cleanly(self):
        """公开的候选值必须都能跑通，否则调优搜索会撞上一个不可用的候选。"""
        for tile_w in CONV_TILE_OPTIONS:
            tiling = plan_conv_tiled(
                kh=3, kw=3, c=8, o=64, ho=32, wo=256, tile_w=tile_w
            )
            self.assertLessEqual(
                tiling.tile_w * tiling.block_y, BLOCK_MAX
            )
        # roc / cb 的候选同理：toc=32、c=8，全部落在预算与整除条件内
        for roc in (1, 2, 4):
            tiling = plan_conv_tiled(
                kh=3, kw=3, c=8, o=64, ho=32, wo=256, roc=roc
            )
            self.assertEqual(tiling.roc, roc)
            self.assertLessEqual(tiling.tile_w * tiling.block_y, BLOCK_MAX)
        for cb in (1, 2, 4, 8):
            tiling = plan_conv_tiled(
                kh=3, kw=3, c=8, o=64, ho=32, wo=256, cb=cb
            )
            self.assertEqual(tiling.cb, cb)
            self.assertLessEqual(tiling.shared_bytes, CONV_SHARED_LIMIT)


class TestConvImplicit(unittest.TestCase):
    """隐式 GEMM 卷积的计划：权重与输入窗口一起占 shared 预算。

    与 ``plan_conv_tiled`` 的差别全在 shared：只按权重挑 ``toc`` 会在宽核上
    把预算撑爆，因此这里额外验证"窗口也算进去"这件事本身。
    """

    def test_span_follows_stride_and_dilation(self):
        tiling = plan_conv_implicit(
            kh=3, kw=3, c=16, o=32, wo=64, sw=2, dw=2
        )
        # span = tile_w * stride_w + (kw - 1) * dilation_w
        self.assertEqual(tiling.span, tiling.tile_w * 2 + 4)
        self.assertEqual(tiling.block[0], tiling.tile_w)

    def test_shared_covers_weights_and_window(self):
        tiling = plan_conv_implicit(kh=3, kw=3, c=8, o=64, wo=50, sw=1, dw=1)
        per_cb = tiling.toc * 9 * 4 + 3 * tiling.span * 4
        self.assertEqual(tiling.shared_bytes, per_cb * tiling.cb)
        self.assertLessEqual(tiling.shared_bytes, CONV_SHARED_LIMIT)

    def test_window_counts_against_the_budget(self):
        """每批预算是"权重 + 窗口"：只按权重算会低估，宽核上就会撑爆 shared。"""
        tiling = plan_conv_implicit(kh=7, kw=7, c=1024, o=32, wo=8, sw=1, dw=1)
        weights_only = tiling.toc * 49 * 4
        self.assertGreater(tiling.shared_bytes // tiling.cb, weights_only)
        self.assertLessEqual(tiling.shared_bytes, CONV_SHARED_LIMIT)

    def test_rejects_empty_dimensions(self):
        with self.assertRaises(ValueError):
            plan_conv_implicit(kh=3, kw=3, c=0, o=4, wo=4, sw=1, dw=1)
        with self.assertRaises(ValueError):
            plan_conv_implicit(kh=3, kw=3, c=4, o=4, wo=4, sw=0, dw=1)

    def test_rejects_bad_roc_and_cb(self):
        with self.assertRaises(ValueError):
            plan_conv_implicit(
                kh=3, kw=3, c=8, o=64, wo=32, sw=1, dw=1, roc=3
            )
        with self.assertRaises(ValueError):
            plan_conv_implicit(
                kh=3, kw=3, c=4, o=64, wo=32, sw=1, dw=1, cb=8
            )

    def test_tile_width_override_keeps_span_consistent(self):
        tiling = plan_conv_implicit(
            kh=3, kw=3, c=8, o=64, wo=50, sw=1, dw=1, tile_w=16
        )
        self.assertEqual(tiling.tile_w, 16)
        self.assertEqual(tiling.span, 18)
        self.assertEqual(tiling.tw, 1)

    def test_tw_widens_window_but_not_block(self):
        """5.254：``tw`` 只放大窗口与 grid，block 的 x 维仍是 ``tile_w``。"""
        single = plan_conv_implicit(kh=3, kw=3, c=8, o=64, wo=64, sw=1, dw=1)
        double = plan_conv_implicit(
            kh=3, kw=3, c=8, o=64, wo=64, sw=1, dw=1, tw=2
        )
        self.assertEqual(double.tw, 2)
        self.assertEqual(double.tile_w, single.tile_w)
        self.assertEqual(double.block, single.block)
        self.assertEqual(double.span, 2 * single.tile_w + 2)
        self.assertLessEqual(double.shared_bytes, CONV_SHARED_LIMIT)

    def test_oversized_kernel_rejected(self):
        # 128x128 核的权重加窗口远超 48KB，明确拒绝而不是静默改用 im2col
        with self.assertRaises(ValueError):
            plan_conv_implicit(
                kh=128, kw=128, c=1, o=1, wo=4, sw=1, dw=1
            )


class TestConvDecision(unittest.TestCase):
    def test_small_k_with_enough_work(self):
        self.assertTrue(conv_tiled_preferred(32, 4096))

    def test_large_k_prefers_gemm(self):
        self.assertFalse(conv_tiled_preferred(33, 1 << 20))

    def test_tiny_work_prefers_gemm(self):
        self.assertFalse(conv_tiled_preferred(32, 4095))


class TestScanChunk(unittest.TestCase):
    """分块扫描的自动判据（5.155）：只在"长 T + 小尾维"时打开。

    这两条边界都是从 M40 的扫描曲线里读出来的：``elems`` 一超过 2048，
    串行版已经跑在带宽上限，分块多出的两趟 scratch 流量只会更慢；``steps``
    不到 512 时块数太少，块前缀摊不开——T=256 实测三格持平或略慢、
    elems=1024 那格反而慢 29%。
    """

    def test_short_sequence_stays_serial(self):
        for steps in (1, 64, 128, 256, 511):
            self.assertEqual(scan_auto_chunk(steps, 8), 0, msg=f"steps={steps}")

    def test_long_small_tail_uses_chunk(self):
        for steps in (512, 1024, 8192, 16384):
            self.assertEqual(scan_auto_chunk(steps, 8), SCAN_CHUNK, msg=f"steps={steps}")

    def test_wide_tail_stays_serial(self):
        for elems in (SCAN_CHUNK_MAX_ELEMS + 1, 4096, 16384):
            self.assertEqual(scan_auto_chunk(1024, elems), 0, msg=f"elems={elems}")

    def test_boundaries(self):
        self.assertEqual(scan_auto_chunk(511, 8), 0)
        self.assertEqual(scan_auto_chunk(512, 8), SCAN_CHUNK)
        self.assertEqual(scan_auto_chunk(1024, SCAN_CHUNK_MAX_ELEMS), SCAN_CHUNK)
        self.assertEqual(scan_auto_chunk(1024, SCAN_CHUNK_MAX_ELEMS + 1), 0)

    def test_empty_shapes(self):
        self.assertEqual(scan_auto_chunk(0, 8), 0)
        self.assertEqual(scan_auto_chunk(1024, 0), 0)
        self.assertEqual(scan_last_auto_chunk(0, 0), 0)

    def test_last_state_has_wider_tail_budget(self):
        """``scan_last_state`` 不写整段输出，尾维上限放宽到 4096。"""
        self.assertEqual(scan_last_auto_chunk(1024, SCAN_LAST_MAX_ELEMS), SCAN_CHUNK)
        self.assertEqual(scan_auto_chunk(1024, SCAN_LAST_MAX_ELEMS), 0)
        self.assertEqual(scan_last_auto_chunk(1024, SCAN_LAST_MAX_ELEMS + 1), 0)
        self.assertEqual(scan_last_auto_chunk(128, 64), 0)

    def test_explicit_values_win_over_auto(self):
        self.assertEqual(scan_chunk(None, 1024, 8), SCAN_CHUNK)
        self.assertEqual(scan_chunk(None, 1024, 4096, last=True), SCAN_CHUNK)
        # 0 = 强制串行，哪怕形状落在自动档的甜点区
        self.assertEqual(scan_chunk(0, 8192, 8), 0)
        self.assertEqual(scan_chunk(128, 1024, 8), 128)

    def test_explicit_chunk_needs_at_least_two_chunks(self):
        """块数 < 2 时第一趟只是把整段搬一遍，白花流量，退回串行。"""
        self.assertEqual(scan_chunk(128, 128, 8), 0)
        self.assertEqual(scan_chunk(128, 127, 8), 0)
        self.assertEqual(scan_chunk(128, 129, 8), 128)


class TestPlanCsrScan(unittest.TestCase):
    """``spike_csr`` 的块大小只跟列数走（5.176）。"""

    def test_block_follows_width(self):
        self.assertEqual(plan_csr_scan(1), 1)
        self.assertEqual(plan_csr_scan(8), 8)
        self.assertEqual(plan_csr_scan(200), 256)
        self.assertEqual(plan_csr_scan(256), 256)

    def test_block_is_capped_at_256(self):
        """段内只做一次比较与前缀和，块再宽只会让同步的层数变多。"""
        self.assertEqual(plan_csr_scan(4096), 256)

    def test_block_never_exceeds_scan_slots(self):
        for width in (1, 3, 17, 255, 256, 1024, 100000):
            with self.subTest(width=width):
                self.assertLessEqual(plan_csr_scan(width), CSR_SCAN_MAX)
                # 扫描区 + "本段总数" + 预留：槽位必须够放
                self.assertGreaterEqual(CSR_SCAN_SLOTS, plan_csr_scan(width) + 2)

    def test_empty_width_is_rejected(self):
        for bad in (0, -1):
            with self.subTest(width=bad):
                with self.assertRaises(ValueError):
                    plan_csr_scan(bad)


class TestPlanCsrScanEnabled(unittest.TestCase):
    """并行铺法只在串行铺法填不满机器时才启用（5.176）。"""

    def test_threshold_is_32_steps_per_sm(self):
        self.assertTrue(plan_csr_scan_enabled(767, 24))
        self.assertFalse(plan_csr_scan_enabled(768, 24))
        self.assertTrue(plan_csr_scan_enabled(31, 1))
        self.assertFalse(plan_csr_scan_enabled(32, 1))

    def test_few_rows_always_scan(self):
        for steps in (1, 8, 64, 256):
            with self.subTest(steps=steps):
                self.assertTrue(plan_csr_scan_enabled(steps, 24))

    def test_many_rows_stay_serial(self):
        """M40 上 2048 行、宽 4096 时串行版已反超：这类形状不能再走并行。"""
        for steps in (2048, 4096, 8192):
            with self.subTest(steps=steps):
                self.assertFalse(plan_csr_scan_enabled(steps, 24))

    def test_empty_and_bad_values(self):
        self.assertFalse(plan_csr_scan_enabled(0, 24))
        with self.assertRaises(ValueError):
            plan_csr_scan_enabled(-1, 24)
        with self.assertRaises(ValueError):
            plan_csr_scan_enabled(8, 0)


if __name__ == "__main__":
    unittest.main()
