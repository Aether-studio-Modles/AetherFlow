"""CUDA 后端测试。

本地没有 GPU 时整组跳过；在远程 GPU 机器上跑全部用例。

这一组测试要证明三件事：
1. 每个算子的结果与参考语义对齐（不是与 cuBLAS 对齐）
2. Transformer Block 端到端对齐
3. 支持不了的地方**明确失败**，而不是静默换实现或降精度
"""

import unittest
from unittest import mock

import numpy as np

from aetherflow import Graph, ReferenceBackend, backward, bf16, f16, f32, f64, i8, i32
from aetherflow.backends.base import BackendCapabilityError
from aetherflow.backends.cuda import CUDA_AVAILABLE, CudaBackend
from aetherflow.backends.device_backend import _group_elementwise, _mask_shared_over_heads
from aetherflow.backends.device_shared import _mask_target_shape
from aetherflow.lower.plan import (
    BLOCK_OPTIONS,
    CONV_SHARED_LIMIT,
    auto_block,
    conv_oc_tile,
    plan_csr_scan_enabled,
)
from aetherflow.models import BlockSpec, build_transformer_block
from aetherflow.models import LIFSpec, build_snn, mean_rate
from aetherflow.runtime.executor import Executor


class TestConvTilePlan(unittest.TestCase):
    """分块卷积的 OC tile 规划：纯算术，不需要 GPU。

    shared memory 预算算错的表现是"构造成功、运行期 illeagal memory access"，
    很难从现象倒推；所以这层判断单独测，把它钉在纯函数上。
    注意入参是**核面积** KH*KW，不是 K——K 方向的通道维是流式分块的，
    不占 shared memory。
    """

    def test_tile_capped_by_shared_budget(self):
        # 3x3 核时 32 行权重只要 32*9*4 = 1152 B，直接拉满
        self.assertEqual(conv_oc_tile(9, 64), 32)

    def test_tile_capped_by_output_channels(self):
        # 只有 6 个输出通道时，没必要为 32 行权重申请 shared memory
        self.assertEqual(conv_oc_tile(9, 6), 4)

    def test_budget_boundary(self):
        k = CONV_SHARED_LIMIT // 4 // 4  # 恰好 4 行塞满预算
        self.assertEqual(conv_oc_tile(k, 64), 4)
        self.assertEqual(conv_oc_tile(k + 1, 64), 2)

    def test_oversized_kernel_rejected(self):
        # 核面积超过预算时不硬撑：连 1 行都放不下就必须换实现
        self.assertIsNone(conv_oc_tile(CONV_SHARED_LIMIT // 4 + 1, 4))


class TestAttentionMaskSharing(unittest.TestCase):
    """共享因果 mask 不能被复制成 (BH,S,T)：判断必须严格。"""

    def test_causal_mask_is_shared(self):
        self.assertTrue(_mask_shared_over_heads((1, 1, 8, 8), 8, 8))
        self.assertTrue(_mask_shared_over_heads((8, 8), 8, 8))
        self.assertTrue(_mask_shared_over_heads((1, 8, 8), 8, 8))

    def test_per_batch_or_head_mask_is_not_shared(self):
        self.assertFalse(_mask_shared_over_heads((2, 8, 8), 8, 8))
        self.assertFalse(_mask_shared_over_heads((2, 4, 8, 8), 8, 8))

    def test_broadcast_shapes_still_take_the_copy_path(self):
        # 可广播但最后两维不是 (S,T) 的 mask 仍走 broadcast_to
        self.assertFalse(_mask_shared_over_heads((1, 8, 1), 8, 8))
        self.assertFalse(_mask_shared_over_heads((1, 8, 9), 8, 8))

    def test_leading_singletons_are_stripped_by_shape(self):
        """降维只算目标形状、不改数组：天数 DeviceArray 拒绝整数索引。"""
        self.assertEqual(_mask_target_shape((1, 1, 8, 8), 3), (1, 8, 8))
        self.assertEqual(_mask_target_shape((1, 1, 8, 8), 2), (8, 8))
        self.assertEqual(_mask_target_shape((1, 1, 1, 8, 8), 3), (1, 8, 8))
        self.assertEqual(_mask_target_shape((1, 8, 8), 3), (1, 8, 8))
        self.assertEqual(_mask_target_shape((8, 8), 3), (8, 8))

    def test_non_singleton_leading_dim_is_rejected(self):
        self.assertIsNone(_mask_target_shape((2, 1, 8, 8), 3))
        self.assertIsNone(_mask_target_shape((2, 8, 8), 2))


class TestPlannedDispatch(unittest.TestCase):
    """批量派发的骨架语义：计划期绑定内核，运行期完全绕开 ``invoke``。

    这层与设备无关：它只保证"计划里绑了什么就调什么""没有绑定就退回原
    路径""同一张图的计划只准备一次"。真机上的数值一致性由
    ``TestBatchDispatch`` 对照。
    """

    class Stub(Executor):
        def __init__(self, bind=True):
            self.bind = bind
            self.calls = []
            self.prepared = 0
            self.invoked = 0

        def group_nodes(self, graph):
            return [(node,) for node in graph.nodes]

        def _prepare_plan(self, plan):
            self.prepared += 1
            if not self.bind:
                return
            for group in plan:
                group.kernels = (self._fn(),)

        @staticmethod
        def _fn():
            def call(node, args, attrs):
                return np.asarray(args[0]) + 1.0

            return call

        def _record(self, name):
            self.calls.append(name)

        def invoke(self, node, args, attrs):
            self.invoked += 1
            self._record(node.op.name)
            return np.asarray(args[0]) + 1.0

    @staticmethod
    def _graph():
        g = Graph()
        a = g.input("a", f32[4])
        mid = g.add(a, a)
        g.output(g.mul(mid, a), "y")
        return g

    def test_bound_plan_bypasses_invoke(self):
        stub = self.Stub()
        g = self._graph()
        feed = {"a": np.arange(4, dtype=np.float32)}
        out = stub.run(g, feed)["y"]
        # 两个节点各 +1，结果与逐节点路径一致
        np.testing.assert_allclose(out, feed["a"] + 2.0)
        self.assertEqual(stub.invoked, 0)
        self.assertEqual(stub.prepared, 1)

    def test_plan_is_prepared_once_per_graph(self):
        stub = self.Stub()
        g = self._graph()
        feed = {"a": np.ones(4, dtype=np.float32)}
        stub.run(g, feed)
        stub.run(g, feed)
        self.assertEqual(stub.prepared, 1)

    def test_unbound_plan_falls_back_to_invoke(self):
        stub = self.Stub(bind=False)
        g = self._graph()
        feed = {"a": np.ones(4, dtype=np.float32)}
        out = stub.run(g, feed)["y"]
        np.testing.assert_allclose(out, feed["a"] + 2.0)
        self.assertEqual(stub.invoked, 2)

    def test_binding_covers_every_node_in_group(self):
        stub = self.Stub()
        g = self._graph()
        stub.run(g, {"a": np.ones(4, dtype=np.float32)})
        plan = stub._plan(g)
        self.assertEqual([len(group.kernels) for group in plan], [1, 1])


class TestConstantPlanCache(unittest.TestCase):
    """常量随计划缓存（5.172）：只上传一次，且不会被 ``release`` 回收。

    5.161 的 mask 用例里，常量折叠的产物 256 KB、原计算很便宜，逐节点 GPU
    时间反而 +22.4%——根因就是每次 execute 都把常量重新 H2D。这条路径与
    设备无关，用替身执行器把"计划期上传一次、run 期直接复用"钉住。
    """

    class Stub(Executor):
        def __init__(self):
            self.prepared = 0
            self.uploads = 0
            self.disposed = []

        def _prepare_plan(self, plan):
            self.prepared += 1
            for group in plan:
                if len(group.nodes) != 1:
                    continue
                node = group.nodes[0]
                if node.op.name == "constant":
                    self.uploads += 1
                    group.constants[node.output.id] = self._constant(node)

        @staticmethod
        def _constant(node):
            return np.full(node.output.type.shape, 7.0, dtype=np.float32)

        def invoke(self, node, args, attrs):
            if node.op.name == "constant":
                # 走到这里说明计划缓存没命中，等于回到修复前的行为
                self.uploads += 1
                return self._constant(node)
            return np.asarray(args[0]) + np.asarray(args[1])

        def dispose(self, value):
            self.disposed.append(value)

    @staticmethod
    def _graph():
        g = Graph()
        c = g.constant(7.0, shape=(2, 2))
        x = g.input("x", f32[2, 2])
        g.output(g.add(c, x), "y")
        return g

    def test_constant_is_uploaded_once_per_graph(self):
        stub = self.Stub()
        g = self._graph()
        feed = {"x": np.ones((2, 2), dtype=np.float32)}
        first = stub.run(g, feed)["y"]
        second = stub.run(g, feed)["y"]
        np.testing.assert_allclose(first, 8.0)
        np.testing.assert_array_equal(second, first)
        self.assertEqual(stub.uploads, 1)
        self.assertEqual(stub.prepared, 1)

    def test_constant_cached_value_survives_release(self):
        """``run`` 走 ``release=True``，常量不能被当成"最后使用即回收"。"""
        stub = self.Stub()
        g = self._graph()
        stub.run(g, {"x": np.ones((2, 2), dtype=np.float32)})
        plan = stub._plan(g)
        const = plan[0].constants[next(iter(plan[0].constants))]
        self.assertFalse(any(d is const for d in stub.disposed))

    def test_clearing_plan_values_forces_reupload(self):
        """切卡 / 关闭 arena 之后，下一次执行必须重新上传到新地址。"""
        stub = self.Stub()
        g = self._graph()
        feed = {"x": np.ones((2, 2), dtype=np.float32)}
        stub.run(g, feed)
        stub._clear_plan_values()
        out = stub.run(g, feed)["y"]
        np.testing.assert_allclose(out, 8.0)
        self.assertEqual(stub.uploads, 2)
        self.assertEqual(stub.prepared, 2)


class TestBroadcastPlan(unittest.TestCase):
    """广播索引规划的正确性：与 numpy 广播逐元素对拍。

    不需要 GPU——这里复现的正是 CUDA kernel 内部的索引计算
    （对齐 stride、把广播维置 0、折叠相邻维），因此能在本地就发现错误。
    """

    CASES = [
        ((3, 5), (3, 1)),
        ((3, 5), (5,)),
        ((2, 3, 4), (4,)),
        ((2, 1, 4), (1, 3, 1)),
        ((2, 3, 1, 4, 5), (1, 1, 6, 1, 1)),
        ((2, 1), (1, 3)),
        ((4, 1, 1), (1, 3, 1)),
        ((5,), (1, 1)),  # 标量快速路径不能误用：广播结果是 (1,5)
    ]

    def test_plan_matches_numpy(self):
        from aetherflow.backends.cuda.backend import CudaBackend

        rng = np.random.default_rng(1)
        for shape_a, shape_b in self.CASES:
            out_shape = np.broadcast_shapes(shape_a, shape_b)
            a = rng.standard_normal(shape_a).astype(np.float32)
            b = rng.standard_normal(shape_b).astype(np.float32)
            groups = CudaBackend._collapse_dims(
                out_shape,
                CudaBackend._elem_strides(shape_a, out_shape),
                CudaBackend._elem_strides(shape_b, out_shape),
            )
            self.assertLessEqual(len(groups), 4)

            got = np.empty(out_shape, dtype=np.float32)
            for i in range(int(got.size)):
                rem, oa, ob = i, 0, 0
                for dim, step_a, step_b in reversed(groups):
                    c = rem % dim
                    rem //= dim
                    oa += c * step_a
                    ob += c * step_b
                got.flat[i] = a.flat[oa] + b.flat[ob]
            np.testing.assert_allclose(got, a + b)


class TestIndexMath(unittest.TestCase):
    """新增内核的索引循环：用 numpy 复刻，不需要 GPU 就能对拍。

    concat 的段偏移、embedding 的 (i, j) 拆分、反向散射的重叠累加，
    在 GPU 上的表现是"数值不对"而不是崩溃，本地先钉住能省一次远程往返。
    复刻的是 kernel 里的**同一套公式**，不是另写一份实现。
    """

    def test_concat_copy_offsets(self):
        rng = np.random.default_rng(3)
        parts = [
            rng.standard_normal(s).astype(np.float32)
            for s in ((2, 3, 4), (2, 5, 4))
        ]
        axis = 1
        inner = int(np.prod(parts[0].shape[axis + 1 :]))
        out_axis = sum(p.shape[axis] for p in parts)
        flat_out = np.empty(2 * out_axis * 4, dtype=np.float32)
        offset = 0
        for p in parts:
            flat = p.reshape(-1)
            seg = p.shape[axis]
            for idx in range(flat.size):
                j = idx % inner
                t = idx // inner
                s = t % seg
                o = t // seg
                flat_out[(o * out_axis + offset + s) * inner + j] = flat[idx]
            offset += seg
        np.testing.assert_allclose(flat_out.reshape(2, out_axis, 4), np.concatenate(parts, axis=axis))

    def test_embedding_gather(self):
        rng = np.random.default_rng(4)
        table = rng.standard_normal((10, 2, 3)).astype(np.float32)
        ids = rng.integers(0, 10, (4, 5), dtype=np.int32)
        flat_table = table.reshape(10, -1)
        row = flat_table.shape[1]
        flat_ids = ids.reshape(-1)
        flat_out = np.empty((flat_ids.size, row), dtype=np.float32)
        for idx in range(flat_ids.size * row):
            j = idx % row
            i = idx // row
            flat_out[i][j] = flat_table[flat_ids[i]][j]
        np.testing.assert_allclose(flat_out.reshape(4, 5, 2, 3), table[ids])

    def test_gather_three_axis_offsets(self):
        """gather 内核的 (outer, idx_dim, inner) 拆分，用非末维验证。"""
        rng = np.random.default_rng(6)
        x = rng.standard_normal((3, 6, 4)).astype(np.float32)
        idx = rng.integers(0, 6, (3, 2, 4), dtype=np.int32)
        dim = 1
        idx_dim = 2
        inner = 4
        x_dim = 6
        flat_x = x.reshape(-1)
        flat_idx = idx.reshape(-1)
        flat_out = np.empty(flat_idx.size, dtype=np.float32)
        for p in range(flat_idx.size):
            i = p % inner
            t = p // inner
            o = t // idx_dim
            v = int(flat_idx[p])
            flat_out[p] = flat_x[(o * x_dim + v) * inner + i]
        np.testing.assert_array_equal(
            flat_out.reshape(3, 2, 4), np.take_along_axis(x, idx.astype(np.int64), axis=dim)
        )

    def test_gather_backward_scatter_add(self):
        """重复下标必须累加：内核里是原子加，不是赋值。"""
        rng = np.random.default_rng(7)
        dy = rng.standard_normal((3, 2, 4)).astype(np.float32)
        idx = np.full((3, 2, 4), 1, dtype=np.int32)
        idx_dim = 2
        inner = 4
        x_dim = 6
        dx = np.zeros((3, 6, 4), dtype=np.float32)
        flat_dy = dy.reshape(-1)
        for p in range(flat_dy.size):
            i = p % inner
            t = p // inner
            o = t // idx_dim
            v = int(idx.reshape(-1)[p])
            dx.reshape(-1)[(o * x_dim + v) * inner + i] += flat_dy[p]
        want = np.zeros((3, 6, 4), dtype=np.float32)
        np.add.at(
            want,
            (np.arange(3)[:, None, None], idx.astype(np.int64),
             np.arange(4)[None, None, :]),
            dy,
        )
        np.testing.assert_allclose(dx, want, rtol=0, atol=0)

    def test_avgpool_backward_scatter(self):
        rng = np.random.default_rng(5)
        n, c, h, w = 1, 2, 6, 6
        kh = kw = 3
        sh = sw = 1
        ph = pw = 1
        ho = (h + 2 * ph - kh) // sh + 1
        wo = (w + 2 * pw - kw) // sw + 1
        dy = rng.standard_normal((n, c, ho, wo)).astype(np.float32)
        dx = np.zeros((n, c, h, w), dtype=np.float64)
        scale = 1.0 / (kh * kw)
        for idx in range(n * c * ho * wo):
            wo_i = idx % wo
            t = idx // wo
            ho_i = t % ho
            t //= ho
            ci = t % c
            ni = t // c
            g = float(dy.reshape(-1)[idx]) * scale
            for ki in range(kh):
                ih = ho_i * sh - ph + ki
                if ih < 0 or ih >= h:
                    continue
                for kj in range(kw):
                    iw = wo_i * sw - pw + kj
                    if iw < 0 or iw >= w:
                        continue
                    dx[ni, ci, ih, iw] += g
        graph = Graph()
        dy_in = graph.input("dy", f32[(n, c, ho, wo)])
        graph.output(
            graph.avgpool2d_backward(
                dy_in,
                input_shape=[n, c, h, w],
                kernel=(kh, kw),
                stride=(sh, sw),
                padding=(ph, pw),
            ),
            "y",
        )
        expected = ReferenceBackend().execute(graph, {"dy": dy})["y"]
        np.testing.assert_allclose(dx.astype(np.float32), expected, rtol=1e-5, atol=1e-5)


    def test_conv_input_grad_gather_index(self):
        """卷积输入梯度的反查公式：``(ih+PH-ki*DH) % SH == 0`` 才构成贡献。

        这个整除判断写错不会崩，只会让某几行/列梯度少算或多算，远程调一次
        要花一轮往返，所以在本地用 numpy 复刻同一套索引先对拍。
        """
        rng = np.random.default_rng(6)
        n, c, h, w = 1, 2, 5, 5
        o, kh, kw = 3, 3, 3
        sh = sw = 2
        ph = pw = 1
        dh = dw = 1
        ho = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        wo = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        dy = rng.standard_normal((n, o, ho, wo)).astype(np.float32)
        weight = rng.standard_normal((o, c, kh, kw)).astype(np.float32)

        dx = np.zeros((n, c, h, w), dtype=np.float64)
        for idx in range(n * c * h * w):
            iw = idx % w
            t = idx // w
            ih = t % h
            t //= h
            ci = t % c
            ni = t // c
            acc = 0.0
            for oi in range(o):
                for ki in range(kh):
                    num = ih + ph - ki * dh
                    if num < 0 or num % sh:
                        continue
                    hi = num // sh
                    if hi >= ho:
                        continue
                    for kj in range(kw):
                        numw = iw + pw - kj * dw
                        if numw < 0 or numw % sw:
                            continue
                        wi = numw // sw
                        if wi >= wo:
                            continue
                        acc += float(dy[ni, oi, hi, wi]) * float(weight[oi, ci, ki, kj])
            dx[ni, ci, ih, iw] = acc

        graph = Graph()
        dy_in = graph.input("dy", f32[(n, o, ho, wo)])
        w_in = graph.input("w", f32[(o, c, kh, kw)])
        graph.output(
            graph.conv2d_input_grad(
                dy_in,
                w_in,
                input_shape=[n, c, h, w],
                stride=(sh, sw),
                padding=(ph, pw),
                dilation=(dh, dw),
            ),
            "y",
        )
        expected = ReferenceBackend().execute(graph, {"dy": dy, "w": weight})["y"]
        np.testing.assert_allclose(dx.astype(np.float32), expected, rtol=1e-5, atol=1e-5)


class TestElementwiseFusionPlan(unittest.TestCase):
    """融合的**计划**：哪些相邻节点能合成一组。纯静态，不需要设备。

    这里每个用例都对应一条"不能融"的理由，防止为了少几次 launch 而悄悄
    改变可观测行为（物化中间值、广播、多消费者）。
    """

    def _groups(self, graph):
        return [tuple(n.op.name for n in group) for group in _group_elementwise(graph)]

    def test_adjacent_elementwise_chain_is_one_group(self):
        g = Graph()
        a = g.input("a", f32[4, 8])
        b = g.input("b", f32[4, 8])
        g.output(g.mul(g.relu(g.add(a, b)), b), "y")
        self.assertEqual(self._groups(g), [("add", "relu", "mul")])

    def test_scalar_side_is_fused(self):
        g = Graph()
        a = g.input("a", f32[4, 8])
        s = g.input("s", f32[1])
        g.output(g.relu(g.mul(a, s)), "y")
        self.assertEqual(self._groups(g), [("mul", "relu")])

    def test_clamp_joins_the_chain(self):
        """``clamp`` 的边界是属性里的常量，不占操作数，可以和邻居同组。

        SSM / 门控里的 ``clamp(x, min=-80) → mul → add`` 因此少一次物化。
        """
        g = Graph()
        a = g.input("a", f32[4, 8])
        s = g.input("s", f32[1])
        g.output(g.mul(g.clamp(g.relu(a), min=-1.0, max=1.0), s), "y")
        self.assertEqual(self._groups(g), [("relu", "clamp", "mul")])

    def test_step_joins_the_chain(self):
        """阈值是属性常量的阶跃同样不占操作数：SNN 的 step → mul 合成一个内核。"""
        g = Graph()
        a = g.input("a", f32[4, 8])
        g.output(g.mul(g.step(a, threshold=0.5), a), "y")
        self.assertEqual(self._groups(g), [("step", "mul")])

    def test_graph_output_intermediate_is_materialized(self):
        """链中间的边是图输出时必须物化；链尾本身是图输出不影响合并。"""
        g = Graph()
        a = g.input("a", f32[4, 8])
        b = g.input("b", f32[4, 8])
        mid = g.add(a, b)
        g.output(mid, "mid")
        g.output(g.relu(g.mul(mid, b)), "y")
        self.assertEqual(self._groups(g), [("add",), ("mul", "relu")])

    def test_value_used_twice_breaks_the_chain(self):
        g = Graph()
        a = g.input("a", f32[4, 8])
        mid = g.relu(a)
        g.output(g.mul(mid, mid), "y")
        self.assertEqual(self._groups(g), [("relu",), ("mul",)])

    def test_non_scalar_broadcast_is_not_fused(self):
        g = Graph()
        a = g.input("a", f32[3, 5])
        b = g.input("b", f32[5])
        g.output(g.relu(g.add(a, b)), "y")
        self.assertEqual(self._groups(g), [("add",), ("relu",)])

    def test_node_without_expression_table_breaks_the_chain(self):
        g = Graph()
        a = g.input("a", f32[2, 4])
        cast = g.cast(g.add(a, a), dtype="f16")
        g.output(g.relu(cast), "y")
        self.assertEqual(self._groups(g), [("add",), ("cast",), ("relu",)])

    def test_gated_silu_is_not_a_chain_step(self):
        g = Graph()
        a = g.input("a", f32[2, 4])
        b = g.input("b", f32[2, 4])
        g.output(g.relu(g.silu(a, b)), "y")
        self.assertEqual(self._groups(g), [("silu",), ("relu",)])


class TestConvEpiloguePlan(unittest.TestCase):
    """conv 与后继逐元素链合并的静态判据（5.270，不需要设备）。

    合并的前提是"conv 输出不再物化"：这些节点会被一个内核算完，中间值
    没有写回点。任何一条前提不成立都必须拆开，否则要么改变可观测行为
    （中间值是图输出 / 有第二个消费者），要么算出不同数值（f16 输出要先
    舍入、逐通道参数之外的大张量无法按通道索引）。
    """

    @staticmethod
    def _backend():
        # 只调 ``group_nodes`` 的静态判据，不建设备上下文。
        return CudaBackend.__new__(CudaBackend)

    def _groups(self, graph):
        return [
            tuple(node.op.name for node in group)
            for group in self._backend().group_nodes(graph)
        ]

    def _net(self, dtype=f32, chain=True, reuse=False):
        """conv → mul(1,C,1,1) → add(1,C,1,1) → relu。"""
        g = Graph()
        c = 4
        x = g.input("x", dtype[2, c, 5, 5])
        w = g.constant(dtype=dtype, shape=(c, c, 3, 3),
                       data=np.full(c * c * 9, 0.1, np.float32).tolist())
        b = g.constant(dtype=dtype, shape=(c,),
                       data=np.full(c, 0.5, np.float32).tolist())
        # BN 的参数要在 conv **之前**建节点：``_merge_conv_epilogue`` 只看
        # "conv 的下一组"，插在中间的常量会把链挡在外面。
        s = t = None
        if chain:
            s = g.constant(dtype=dtype, shape=(1, c, 1, 1),
                           data=np.full(c, 1.5, np.float32).tolist())
            t = g.constant(dtype=dtype, shape=(1, c, 1, 1),
                           data=np.full(c, 0.25, np.float32).tolist())
        y = g.conv2d(x, w, b, padding=1)
        if reuse:
            # conv 输出有第二个消费者：中间值必须物化。
            g.output(y, "mid")
        if not chain:
            g.output(y, "y")
            return g
        g.output(g.relu(g.add(g.mul(y, s), t)), "y")
        return g

    def test_channel_bn_and_relu_join_the_conv(self):
        self.assertIn(("conv2d", "mul", "add", "relu"), self._groups(self._net()))

    def test_conv_left_alone_without_a_chain(self):
        self.assertIn(("conv2d",), self._groups(self._net(chain=False)))

    def test_f16_conv_is_not_merged(self):
        """f16 / bf16 的 conv 输出要先舍入到存储类型，融合会改写数值。"""
        self.assertIn(("conv2d",), self._groups(self._net(dtype=f16)))

    def test_materialized_conv_output_is_not_merged(self):
        self.assertIn(("conv2d",), self._groups(self._net(reuse=True)))

    def test_second_full_size_operand_is_not_merged(self):
        """链里出现同形的大张量时没有可索引的通道变量，必须拆开。"""
        g = Graph()
        x = g.input("x", f32[2, 4, 5, 5])
        w = g.constant(dtype=f32, shape=(4, 4, 3, 3),
                       data=np.full(144, 0.1, np.float32).tolist())
        b = g.constant(dtype=f32, shape=(4,), data=np.full(4, 0.5,
                                                          np.float32).tolist())
        other = g.input("other", f32[2, 4, 5, 5])
        g.output(g.relu(g.mul(g.conv2d(x, w, b, padding=1), other)), "y")
        self.assertIn(("conv2d",), self._groups(g))


@unittest.skipUnless(CUDA_AVAILABLE, "需要 cupy 与 CUDA 设备")
class CudaTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cuda = CudaBackend()
        cls.ref = ReferenceBackend()
        cls.rng = np.random.default_rng(0)

    def rand(self, *shape):
        return self.rng.standard_normal(shape).astype(np.float32)

    def assert_backends_equal(self, graph, feed, rtol=1e-4, atol=1e-4, name="y"):
        expected = self.ref.execute(graph, feed)[name]
        got = self.cuda.execute(graph, feed)[name]
        np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol)
        return got


class TestCudaElementwise(CudaTestCase):
    def _binary(self, op):
        g = Graph()
        x = g.input("x", f32[3, 5])
        x2 = g.input("x2", f32[3, 5])
        g.output(g.add_node(op, [x, x2]), "y")
        feed = {"x": self.rand(3, 5), "x2": self.rand(3, 5) + 1.5}
        self.assert_backends_equal(g, feed)

    def test_add(self):
        self._binary("add")

    def test_sub(self):
        self._binary("sub")

    def test_mul(self):
        self._binary("mul")

    def test_div(self):
        self._binary("div")

    def test_relu(self):
        g = Graph()
        x = g.input("x", f32[7, 9])
        g.output(g.relu(x), "y")
        self.assert_backends_equal(g, {"x": self.rand(7, 9)})

    def test_silu(self):
        g = Graph()
        x = g.input("x", f32[7, 9])
        g.output(g.silu(x), "y")
        self.assert_backends_equal(g, {"x": self.rand(7, 9) * 3})

    def test_silu_gate(self):
        """SwiGLU 形态：silu(x) * gate 一个内核完成。"""
        g = Graph()
        x = g.input("x", f32[7, 9])
        gate = g.input("gate", f32[7, 9])
        g.output(g.silu(x, gate), "y")
        self.assert_backends_equal(
            g, {"x": self.rand(7, 9) * 3, "gate": self.rand(7, 9)}
        )

    def test_silu_gate_non_divisible_length(self):
        """长度不整除 block，尾部线程必须不写越界。"""
        g = Graph()
        x = g.input("x", f32[37])
        gate = g.input("gate", f32[37])
        g.output(g.silu(x, gate), "y")
        self.assert_backends_equal(g, {"x": self.rand(37), "gate": self.rand(37)})

    def test_broadcast_2d(self):
        """(3,5) + (3,1)：列向量广播，这是 bias 的典型形态。"""
        g = Graph()
        x = g.input("x", f32[3, 5])
        col = g.input("col", f32[3, 1])
        g.output(g.add(x, col), "y")
        self.assert_backends_equal(
            g, {"x": self.rand(3, 5), "col": self.rand(3, 1)}
        )

    def test_broadcast_trailing_dim(self):
        """(2,3,4) + (4,)：Transformer 里 linear 降级出的 bias 形态。"""
        g = Graph()
        x = g.input("x", f32[2, 3, 4])
        bias = g.input("bias", f32[4])
        g.output(g.add(x, bias), "y")
        self.assert_backends_equal(
            g, {"x": self.rand(2, 3, 4), "bias": self.rand(4)}
        )

    def test_broadcast_both_sides(self):
        """(2,1,4) * (1,3,1)：两侧都广播，中间维被展开。"""
        g = Graph()
        a = g.input("a", f32[2, 1, 4])
        b = g.input("b", f32[1, 3, 1])
        g.output(g.mul(a, b), "y")
        self.assert_backends_equal(
            g, {"a": self.rand(2, 1, 4), "b": self.rand(1, 3, 1)}
        )

    def test_broadcast_high_rank(self):
        """5 维广播：3 个不可折叠维 + 1 个展开维，验证折叠逻辑。"""
        g = Graph()
        a = g.input("a", f32[2, 3, 1, 4, 5])
        b = g.input("b", f32[1, 1, 6, 1, 1])
        g.output(g.sub(a, b), "y")
        self.assert_backends_equal(
            g, {"a": self.rand(2, 3, 1, 4, 5), "b": self.rand(1, 1, 6, 1, 1)}
        )

    def test_incompatible_shapes_rejected(self):
        """非法广播在构图期就被 IR 拒绝，不会留到后端运行期。"""
        g = Graph()
        x = g.input("x", f32[3, 5])
        other = g.input("other", f32[4])
        with self.assertRaises(ValueError):
            g.add(x, other)

    def test_scalar_broadcast(self):
        g = Graph()
        x = g.input("x", f32[3, 5])
        s = g.input("s", f32[1])
        g.output(g.mul(x, s), "y")
        feed = {"x": self.rand(3, 5), "s": np.array([2.5], dtype=np.float32)}
        self.assert_backends_equal(g, feed)

    def test_scalar_side_with_extra_rank(self):
        """(5,) 与 (1,1)：一侧 size 为 1，但广播结果是 (1,5)，不能走标量快速路径。"""
        g = Graph()
        a = g.input("a", f32[5])
        b = g.input("b", f32[1, 1])
        g.output(g.add(a, b), "y")
        feed = {"a": self.rand(5), "b": self.rand(1, 1)}
        self.assert_backends_equal(g, feed)


class TestCudaElementwiseChain(CudaTestCase):
    """相邻逐元素算子真的只发射一个内核，且数值与逐节点执行一致。"""

    def _graph(self, dtype=None):
        dtype = dtype or f32
        g = Graph()
        a = g.input("a", dtype[4, 8])
        b = g.input("b", dtype[4, 8])
        s = g.input("s", dtype[1])
        g.output(g.mul(g.relu(g.add(a, b)), s), "y")
        return g

    def _feed(self, dtype=np.float32):
        return {
            "a": self.rand(4, 8).astype(dtype),
            "b": self.rand(4, 8).astype(dtype),
            "s": np.array([1.5], dtype=dtype),
        }

    def test_chain_matches_elementwise_reference(self):
        g = self._graph()
        self.assert_backends_equal(g, self._feed(), rtol=1e-6, atol=1e-6)

    def test_chain_compiles_one_vectorized_kernel(self):
        g = self._graph()
        before = set(self.cuda._functions)
        self.cuda.execute(g, self._feed())
        new = [k for k in self.cuda._functions if k not in before]
        self.assertEqual(len(new), 1, new)
        key = new[0]
        self.assertEqual(key[0], "chain")
        self.assertEqual(key[3], 4, "32 个元素、16 字节对齐时应走 4 通道")
        self.assertIn(key, self.cuda._modules)

    def test_odd_length_falls_back_to_scalar_lanes(self):
        g = Graph()
        a = g.input("a", f32[15])
        b = g.input("b", f32[15])
        g.output(g.relu(g.add(a, b)), "y")
        feed = {"a": self.rand(15), "b": self.rand(15)}
        before = set(self.cuda._functions)
        self.assert_backends_equal(g, feed, rtol=1e-6, atol=1e-6)
        new = [k for k in self.cuda._functions if k not in before]
        self.assertEqual(len(new), 1, new)
        self.assertEqual(new[0][3], 1, "15 不能被 4 整除，只能是标量通道")

    def test_f16_chain_matches_elementwise_reference(self):
        g = self._graph(f16)
        self.assert_backends_equal(g, self._feed(np.float16), rtol=1e-3, atol=1e-3)

    def test_materialized_intermediate_still_runs(self):
        """中间值是图输出时不能融：两次执行必须给出同一组结果。"""
        g = Graph()
        a = g.input("a", f32[4, 8])
        b = g.input("b", f32[4, 8])
        mid = g.relu(g.add(a, b))
        g.output(mid, "mid")
        g.output(g.mul(mid, b), "y")
        feed = self._feed()
        expected = self.ref.execute(g, feed)
        got = self.cuda.execute(g, feed)
        for name in ("mid", "y"):
            np.testing.assert_allclose(got[name], expected[name], rtol=1e-6, atol=1e-6)


class TestCudaCacheKeys(CudaTestCase):
    """两个缓存键必须覆盖"决定结果的那部分输入"。

    这两条都属于"键少了一个维度、程序被复用但语义变了"的缺陷：数值仍然
    有限、形状也对，只有对拍能发现。
    """

    def test_chain_cache_distinguishes_operand_order(self):
        """``t=x+y; out=t-z`` 与 ``t=x+y; out=z-t`` 不能命中同一份源码。

        两条链的算子序列、参数个数、标量标记、通道数完全一致，旧键
        ``(op, 参数个数)`` 会碰撞；生成的源码却是 ``t - z`` 与 ``z - t``。
        """
        a = self.rand(4, 8)
        b = self.rand(4, 8)
        z = self.rand(4, 8)

        def build(flip):
            g = Graph("flip" if flip else "plain")
            x = g.input("x", f32[4, 8])
            y = g.input("y", f32[4, 8])
            zz = g.input("z", f32[4, 8])
            t = g.add(x, y)
            g.output(g.sub(zz, t) if flip else g.sub(t, zz), "out")
            return g

        feed = {"x": a, "y": b, "z": z}
        first = self.cuda.execute(build(False), feed)["out"]
        second = self.cuda.execute(build(True), feed)["out"]
        np.testing.assert_allclose(first, a + b - z, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(second, z - (a + b), rtol=1e-6, atol=1e-6)

    def test_graph_plan_key_tracks_mutable_inputs(self):
        """先驻留 w 用 ``{x}`` 捕获、再显式传 ``{x, w}``：必须换成新 w。

        旧键只看合并后输入的 (名字, 形状, dtype)，两次相同；但捕获时记为
        可变的只有 x，新 w 不会被拷进静态缓冲，输出仍是旧权重的结果。
        """
        g = Graph("resident_switch")
        x = g.input("x", f32[2, 3])
        w = g.input("w", f32[3, 4])
        g.output(g.relu(g.matmul(x, w)), "y")

        backend = CudaBackend(graph_mode="auto")
        x0 = self.rand(2, 3)
        w0 = self.rand(3, 4)
        w1 = self.rand(3, 4)
        backend.resident_parameters(g, {"w": w0})
        first = backend.execute(g, {"x": x0})["y"]
        second = backend.execute(g, {"x": x0, "w": w1})["y"]
        np.testing.assert_allclose(first, np.maximum(x0 @ w0, 0), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(second, np.maximum(x0 @ w1, 0), rtol=1e-5, atol=1e-5)


class TestCudaSymbolicSpatial(CudaTestCase):
    """符号空间维（派生维）在真机上必须能一图多分辨率。"""

    def test_conv2d_derived_spatial_at_multiple_resolutions(self):
        g = Graph()
        x = g.input("x", f32["N", 3, "H", "W"])
        w = g.input("w", f32[4, 3, 3, 3])
        g.output(g.conv2d(x, w, stride=(2, 2), padding=(1, 1)), "y")
        wv = self.rand(4, 3, 3, 3)
        for h, ww in ((8, 8), (12, 10)):
            self.assert_backends_equal(g, {"x": self.rand(2, 3, h, ww), "w": wv})

    def test_conv_pool_chain_at_multiple_resolutions(self):
        g = Graph()
        x = g.input("x", f32["N", 3, "H", "W"])
        w = g.input("w", f32[4, 3, 3, 3])
        y = g.relu(g.conv2d(x, w, stride=(2, 2), padding=(1, 1)))
        g.output(g.maxpool2d(y, kernel=2), "y")
        for h, ww in ((8, 8), (12, 10)):
            self.assert_backends_equal(
                g, {"x": self.rand(2, 3, h, ww), "w": self.rand(4, 3, 3, 3)}
            )

    def test_backward_keeps_derived_shapes(self):
        g = Graph()
        x = g.input("x", f32["N", 3, "H", "W"])
        w = g.input("w", f32[4, 3, 3, 3])
        y = g.relu(g.conv2d(x, w, stride=(2, 2), padding=(1, 1)))
        g.output(g.maxpool2d(y, kernel=2), "y")

        bwd = backward(g, wrt=[x, w])
        feed = {
            "x": self.rand(2, 3, 8, 8),
            "w": self.rand(4, 3, 3, 3),
            "dy": self.rand(2, 4, 2, 2),
        }
        expected = self.ref.execute(bwd, feed)
        got = self.cuda.execute(bwd, feed)
        for name in expected:
            np.testing.assert_allclose(got[name], expected[name], rtol=1e-4, atol=1e-4)


class TestTransposeMatmulPlan(unittest.TestCase):
    @staticmethod
    def graph(left=True, right=True, dtype=f32):
        graph = Graph()
        a = graph.input("a", dtype[7, 3] if left else dtype[3, 7])
        b = graph.input("b", dtype[5, 7] if right else dtype[7, 5])
        x = graph.transpose(a, perm=(1, 0)) if left else a
        y = graph.transpose(b, perm=(1, 0)) if right else b
        graph.output(graph.matmul(x, y), "y")
        return graph

    def test_adjacent_transposes_merge_on_either_side(self):
        backend = CudaBackend.__new__(CudaBackend)
        for left, right in ((True, False), (False, True), (True, True)):
            graph = self.graph(left, right)
            groups = backend.group_nodes(graph)
            self.assertEqual(len(groups), 1)
            self.assertEqual(len(groups[0]), 1 + left + right)

    def test_output_and_extra_consumer_prevent_merge(self):
        backend = CudaBackend.__new__(CudaBackend)
        for extra_consumer in (False, True):
            graph = self.graph(True, False)
            transposed = graph.nodes[0].output
            if extra_consumer:
                graph.output(graph.relu(transposed), "extra")
            else:
                graph.output(transposed, "extra")
            self.assertEqual(len(backend.group_nodes(graph)[0]), 1)

    def test_batched_and_empty_matrices_keep_original_groups(self):
        backend = CudaBackend.__new__(CudaBackend)
        for shape, perm, other in (((2, 7, 3), (0, 2, 1), (2, 7, 5)),
                                   ((7, 0), (1, 0), (7, 5))):
            graph = Graph()
            a = graph.input("a", f32[shape])
            b = graph.input("b", f32[other])
            graph.output(graph.matmul(graph.transpose(a, perm=perm), b), "y")
            self.assertTrue(all(len(group) == 1 for group in backend.group_nodes(graph)))

    def test_disabled_backend_keeps_original_groups(self):
        backend = CudaBackend.__new__(CudaBackend)
        backend.GEMM_TRANSPOSE = False
        self.assertEqual(len(backend.group_nodes(self.graph())), 3)

    def test_closure_uses_sources_and_rejects_noncontiguous_layout(self):
        from aetherflow.runtime.executor import _GroupPlan

        backend = CudaBackend.__new__(CudaBackend)
        backend.cp = np
        blas = mock.Mock()
        backend._cublas = lambda: blas
        backend.current_stream_ptr = lambda: 0
        graph = self.graph()
        group = _GroupPlan(backend.group_nodes(graph)[0], True)
        call = backend._planned_transpose_matmul(group)
        a, b = np.ones((7, 3), np.float32), np.ones((5, 7), np.float32)
        out = call([[a], [b], [None, None]])
        self.assertEqual(out.shape, (3, 5))
        args = blas.gemm_transposed.call_args.args
        self.assertIs(args[0], a)
        self.assertIs(args[1], b)
        self.assertEqual(args[3:], (True, True))
        self.assertIsNone(call([[a[::-1]], [b], [None, None]]))


class TestCudaMatmul(CudaTestCase):
    def test_transpose_fusion_dtypes_capture_and_release(self):
        for dtype in (f32, f16, bf16):
            for left, right in ((True, False), (False, True), (True, True)):
                graph = TestTransposeMatmulPlan.graph(left, right, dtype)
                feed = {v.name: self.rand(*v.type.shape) for v in graph.inputs}
                expected = self.ref.execute(graph, feed)["y"]
                for mode in ("off", "auto"):
                    for memory in ("container", "tiered"):
                        with self.subTest(dtype=dtype, left=left, right=right,
                                          mode=mode, memory=memory):
                            backend = CudaBackend(graph_mode=mode, memory_mode=memory)
                            for _ in range(3):
                                got = backend.execute(graph, feed)["y"]
                                np.testing.assert_allclose(
                                    got, expected, rtol=1e-4 if dtype is f32 else .03,
                                    atol=1e-4 if dtype is f32 else .04,
                                )
                            if mode == "auto":
                                self.assertGreater(backend.graph_captures, 0)
                            dev = {name: backend.to_array(value) for name, value in feed.items()}
                            for release in (False, True):
                                env = backend.run_device(graph, dev, release=release)
                                np.testing.assert_allclose(
                                    backend.to_host(env[graph.outputs[0].id]), expected,
                                    rtol=.03, atol=.04,
                                )

    def test_transpose_graph_output_invalidates_fused_plan(self):
        graph = TestTransposeMatmulPlan.graph(True, False)
        feed = {v.name: self.rand(*v.type.shape) for v in graph.inputs}
        self.cuda.execute(graph, feed)
        graph.output(graph.nodes[0].output, "transposed")
        got = self.cuda.execute(graph, feed)
        expected = self.ref.execute(graph, feed)
        for name in expected:
            np.testing.assert_allclose(got[name], expected[name], rtol=1e-4, atol=1e-4)
        self.assertEqual(len(self.cuda.group_nodes(graph)), 2)

    def test_2d(self):
        g = Graph()
        a = g.input("a", f32[4, 8])
        b = g.input("b", f32[8, 6])
        g.output(g.matmul(a, b), "y")
        self.assert_backends_equal(g, {"a": self.rand(4, 8), "b": self.rand(8, 6)})

    def test_batched(self):
        g = Graph()
        a = g.input("a", f32[2, 4, 8])
        b = g.input("b", f32[2, 8, 6])
        g.output(g.matmul(a, b), "y")
        self.assert_backends_equal(g, {"a": self.rand(2, 4, 8), "b": self.rand(2, 8, 6)})

    def test_batch_broadcast_single_weight(self):
        """(B,M,K) @ (K,N)：权重对所有 batch 共用，步长置 0 复用同一份数据。"""
        g = Graph()
        a = g.input("a", f32[3, 4, 8])
        b = g.input("b", f32[8, 6])
        g.output(g.matmul(a, b), "y")
        self.assert_backends_equal(g, {"a": self.rand(3, 4, 8), "b": self.rand(8, 6)})

    def test_partial_batch_broadcast_rejected(self):
        """(2,1,M,K) @ (2,3,K,N)：一个标量 stride 表达不了部分广播，必须明确失败。"""
        g = Graph()
        a = g.input("a", f32[2, 1, 4, 8])
        b = g.input("b", f32[2, 3, 8, 6])
        g.output(g.matmul(a, b), "y")
        with self.assertRaises(BackendCapabilityError):
            self.cuda.execute(g, {"a": self.rand(2, 1, 4, 8), "b": self.rand(2, 3, 8, 6)})


class TestCudaGatherCrossEntropy(CudaTestCase):
    """gather 与 fused cross_entropy 的设备实现对拍参考语义。

    越界下标在宿主侧是 ``ValueError``，设备端统一收成
    ``BackendCapabilityError``：内核只做 clamp 保证不非法访存，判断与报文
    留在宿主侧，普通执行与 Graph replay 共用一个出口。
    """

    def test_gather_last_and_middle_axis(self):
        for dim, idx_shape, extent in ((-1, (3, 6, 2), 4), (1, (3, 2, 4), 6)):
            with self.subTest(dim=dim):
                g = Graph("gather")
                x = g.input("x", f32[3, 6, 4])
                idx = g.input("idx", i32[idx_shape])
                g.output(g.add_node("gather", [x, idx], {"dim": dim}), "y")
                feed = {
                    "x": self.rand(3, 6, 4),
                    "idx": self.rng.integers(0, extent, idx_shape, dtype=np.int32),
                }
                self.assert_backends_equal(g, feed)

    def test_gather_int8_ids(self):
        g = Graph("gather_i8")
        x = g.input("x", f32[4, 5])
        idx = g.input("idx", i8[4, 3])
        g.output(g.add_node("gather", [x, idx], {"dim": -1}), "y")
        feed = {"x": self.rand(4, 5),
                "idx": self.rng.integers(0, 5, (4, 3), dtype=np.int8)}
        self.assert_backends_equal(g, feed)

    def test_gather_rejects_out_of_range_ids(self):
        g = Graph("gather_oob")
        x = g.input("x", f32[2, 4])
        idx = g.input("idx", i32[2, 2])
        g.output(g.add_node("gather", [x, idx], {"dim": -1}), "y")
        for bad in ([0, 4], [-1, 2]):
            with self.subTest(idx=bad):
                with self.assertRaises(BackendCapabilityError):
                    self.cuda.execute(g, {
                        "x": self.rand(2, 4),
                        # 形状必须与图声明一致：越界检查在绑定之后才走到
                        "idx": np.asarray([bad, [1, 2]], dtype=np.int32),
                    })

    def test_gather_backward_accumulates_repeated_ids(self):
        g = Graph("gather_bwd")
        x = g.input("x", f32[3, 6, 4])
        idx = g.input("idx", i32[3, 2, 4])
        g.output(g.add_node("gather", [x, idx], {"dim": 1}), "y")
        bwd = backward(g, wrt=[x])
        feed = {
            "x": self.rand(3, 6, 4),
            "idx": np.full((3, 2, 4), 1, dtype=np.int32),
            "dy": self.rand(3, 2, 4),
        }
        self.assert_backends_equal(bwd, feed, name="dx")

    def test_cross_entropy_f32_and_f16(self):
        for dtype, tol in ((f32, 1e-5), (f16, 1e-3)):
            with self.subTest(dtype=str(dtype)):
                g = Graph("ce")
                logits = g.input("logits", dtype[8, 11])
                target = g.input("target", i32[8])
                g.output(g.add_node("cross_entropy", [logits, target]), "loss")
                feed = {
                    "logits": self.rand(8, 11) * 2.0,
                    "target": self.rng.integers(0, 11, 8, dtype=np.int32),
                }
                self.assert_backends_equal(g, feed, rtol=tol, atol=tol, name="loss")

    def test_cross_entropy_backward(self):
        g = Graph("ce_bwd")
        logits = g.input("logits", f32[6, 13])
        target = g.input("target", i32[6])
        dloss = g.input("dloss", f32[6])
        g.output(
            g.add_node("cross_entropy_backward", [logits, target, dloss]),
            "dlogits",
        )
        feed = {
            "logits": self.rand(6, 13) * 2.0,
            "target": self.rng.integers(0, 13, 6, dtype=np.int32),
            "dloss": self.rand(6),
        }
        self.assert_backends_equal(g, feed, name="dlogits")

    def test_cross_entropy_rejects_out_of_range_target(self):
        g = Graph("ce_oob")
        logits = g.input("logits", f32[2, 4])
        target = g.input("target", i32[2])
        g.output(g.add_node("cross_entropy", [logits, target]), "loss")
        with self.assertRaises(BackendCapabilityError):
            self.cuda.execute(g, {"logits": self.rand(2, 4),
                                  "target": np.array([4, 0], dtype=np.int32)})


class TestCudaBf16(CudaTestCase):
    """bf16 的设备表示：``uint16`` 位模式 + 按位存取 + cuBLAS 的 CUDA_R_16BF。

    这一组盯住三件事：

    - **逐位一致**：逐元素链 / cast / constant / 归约 / softmax / layernorm
      必须与参考后端（f32 载体 + ``round_bf16``）逐位相同，不是"近似"；
    - **真 16 位存储**：计划里的期望 dtype 是 ``uint16``，而不是 f32 载体；
    - **支持面之外明确失败**：conv2d 的 bf16 版本还没验证过，必须抛
      ``BackendCapabilityError``，不能悄悄用 f32 算。
    """

    def test_device_container_is_two_bytes(self):
        from aetherflow.ir.types import DType

        self.assertEqual(self.cuda.device_dtype(DType.BF16), np.dtype(np.uint16))
        self.assertEqual(self.cuda.device_dtype(DType.F32), np.dtype(np.float32))

    def test_elementwise_chain_is_bit_exact(self):
        g = Graph("ew")
        x = g.input("x", bf16[64])
        w = g.input("w", bf16[64])
        b = g.input("b", bf16[64])
        g.output(g.relu(g.add(g.mul(x, w), b)), "y")
        feed = {"x": self.rand(64), "w": self.rand(64), "b": self.rand(64)}
        got = self.cuda.execute(g, feed)["y"]
        want = self.ref.execute(g, feed)["y"]
        np.testing.assert_array_equal(got, want)
        self.assertEqual(got.dtype, np.float32)

    def test_cast_roundtrip_is_exact(self):
        g = Graph("cast")
        x = g.input("x", f32[16])
        g.output(g.cast(g.cast(x, dtype=bf16), dtype=f32), "y")
        feed = {"x": self.rand(16)}
        got = self.cuda.execute(g, feed)["y"]
        np.testing.assert_array_equal(got, self.ref.execute(g, feed)["y"])

    def test_constant_is_quantized(self):
        g = Graph("const")
        x = g.input("x", bf16[8])
        g.output(g.add(x, g.constant(
            dtype=bf16, shape=(8,), data=[1.234567] * 8)), "y")
        feed = {"x": self.rand(8)}
        np.testing.assert_array_equal(
            self.cuda.execute(g, feed)["y"], self.ref.execute(g, feed)["y"]
        )

    def test_matmul_small_is_bit_exact(self):
        """小 K 上 cuBLAS 与 numpy 的 f32 累加顺序相同，可以逐位比对。"""
        g = Graph("mm")
        a = g.input("a", bf16[32, 64])
        b = g.input("b", bf16[64, 16])
        g.output(g.matmul(a, b), "y")
        feed = {"a": self.rand(32, 64), "b": self.rand(64, 16)}
        np.testing.assert_array_equal(
            self.cuda.execute(g, feed)["y"], self.ref.execute(g, feed)["y"]
        )

    def test_matmul_batched_tensor_core(self):
        """批量形状走 GemmStridedBatchedEx；容差是一个 bf16 ulp 的量级。"""
        g = Graph("mm")
        a = g.input("a", bf16[2, 64, 128])
        b = g.input("b", bf16[2, 128, 64])
        g.output(g.matmul(a, b), "y")
        feed = {"a": self.rand(2, 64, 128), "b": self.rand(2, 128, 64)}
        got = self.cuda.execute(g, feed)["y"]
        want = self.ref.execute(g, feed)["y"]
        np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2)

    def test_reduce_softmax_layernorm_are_bit_exact(self):
        g = Graph("sum")
        x = g.input("x", bf16[8, 32])
        g.output(g.reduce_sum(x, axis=-1), "y")
        feed = {"x": self.rand(8, 32)}
        np.testing.assert_array_equal(
            self.cuda.execute(g, feed)["y"], self.ref.execute(g, feed)["y"]
        )

        g = Graph("softmax")
        x = g.input("x", bf16[8, 32])
        g.output(g.softmax(x, axis=-1), "y")
        np.testing.assert_array_equal(
            self.cuda.execute(g, feed)["y"], self.ref.execute(g, feed)["y"]
        )

        g = Graph("ln")
        x = g.input("x", bf16[8, 32])
        gamma = g.input("g", bf16[32])
        beta = g.input("b", bf16[32])
        g.output(g.layernorm(x, gamma, beta), "y")
        ln_feed = {"x": self.rand(8, 32), "g": np.ones(32, np.float32),
                   "b": self.rand(32)}
        np.testing.assert_array_equal(
            self.cuda.execute(g, ln_feed)["y"], self.ref.execute(g, ln_feed)["y"]
        )

    def test_conv2d_outside_support_surface_is_rejected(self):
        g = Graph("conv")
        x = g.input("x", bf16[1, 4, 8, 8])
        w = g.input("w", bf16[4, 4, 3, 3])
        g.output(g.conv2d(x, w, padding=1), "y")
        with self.assertRaises(BackendCapabilityError):
            self.cuda.execute(g, {"x": self.rand(1, 4, 8, 8),
                                  "w": self.rand(4, 4, 3, 3)})


class TestCudaConv(CudaTestCase):
    """im2col + cuBLAS 的卷积实现：语义必须与 IR 的参考实现逐点对齐。

    覆盖 padding / stride / dilation / 非方形核 / 无 bias / 非对称参数，
    因为这些参数各自影响 im2col 的索引计算，错一个就会在边界处偏掉。
    """

    def _conv(self, x_shape, w_shape, attrs=None, bias=True):
        g = Graph()
        x = g.input("x", f32[x_shape])
        w = g.input("w", f32[w_shape])
        args = [x, w]
        feed = {"x": self.rand(*x_shape), "w": self.rand(*w_shape) * 0.3}
        if bias:
            b = g.input("b", f32[w_shape[0]])
            args.append(b)
            feed["b"] = self.rand(w_shape[0])
        g.output(g.add_node("conv2d", args, attrs or {}), "y")
        return g, feed

    def test_3x3_pad1(self):
        g, feed = self._conv((2, 3, 8, 8), (4, 3, 3, 3), {"padding": (1, 1)})
        self.assert_backends_equal(g, feed)

    def test_stride2(self):
        g, feed = self._conv((1, 2, 9, 9), (3, 2, 3, 3), {"stride": (2, 2)})
        self.assert_backends_equal(g, feed)

    def test_dilation(self):
        g, feed = self._conv(
            (1, 2, 9, 9), (3, 2, 3, 3), {"dilation": (2, 2), "padding": (2, 2)}
        )
        self.assert_backends_equal(g, feed)

    def test_1x1_without_bias(self):
        g, feed = self._conv((3, 4, 5, 6), (7, 4, 1, 1), bias=False)
        self.assert_backends_equal(g, feed)

    def test_non_square_kernel_and_asymmetric_padding(self):
        g, feed = self._conv((2, 2, 7, 5), (3, 2, 3, 2), {"padding": (1, 0)})
        self.assert_backends_equal(g, feed)

    def test_asymmetric_stride(self):
        g, feed = self._conv(
            (1, 1, 10, 7), (2, 1, 2, 2), {"stride": (1, 2), "padding": (0, 1)}
        )
        self.assert_backends_equal(g, feed)

    def test_grouped(self):
        """groups=2：权重第二维是 C/groups，输出通道也按组切分。"""
        g, feed = self._conv(
            (2, 4, 8, 8), (6, 2, 3, 3), {"groups": 2, "padding": (1, 1)}
        )
        self.assert_backends_equal(g, feed)

    def test_depthwise(self):
        """depthwise：groups==C==O，每个输入通道一个独立核。"""
        g, feed = self._conv(
            (2, 4, 7, 7), (4, 1, 3, 3), {"groups": 4, "padding": (1, 1)}
        )
        self.assert_backends_equal(g, feed)

    def test_depthwise_stride2(self):
        g, feed = self._conv(
            (1, 4, 9, 9),
            (4, 1, 3, 3),
            {"groups": 4, "stride": (2, 2), "padding": (1, 1)},
        )
        self.assert_backends_equal(g, feed)

    def test_grouped_dilation(self):
        """分组 + dilation：直算核的窗口步长必须乘上 DH/DW。"""
        g, feed = self._conv(
            (1, 4, 13, 13),
            (4, 2, 3, 3),
            {"groups": 2, "dilation": (2, 2), "padding": (2, 2)},
        )
        got = CudaBackend(conv_impl="grouped").execute(g, feed)["y"]
        expected = self.ref.execute(g, feed)["y"]
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)

    def test_grouped_matches_im2col(self):
        """显式 grouped 与强制 im2col 数值一致（累加顺序不同，不比逐位）。"""
        g, feed = self._conv(
            (2, 6, 9, 9),
            (9, 2, 3, 3),
            {"groups": 3, "stride": (2, 1), "padding": (1, 0)},
        )
        fast = CudaBackend(conv_impl="grouped").execute(g, feed)["y"]
        slow = CudaBackend(conv_impl="im2col").execute(g, feed)["y"]
        np.testing.assert_allclose(fast, slow, rtol=1e-5, atol=1e-5)

    def test_auto_uses_grouped_only_for_depthwise(self):
        """auto 的分组决策：depthwise 走直算，非 depthwise 不赌（5.171）。

        M40 上 cuBLAS 的 im2col 在 g2/g4 上反超直算核，HIP 上则相反；
        跨厂商没有统一结论，所以 auto 只把"必然更快"的 depthwise 收进来。
        """
        backend = CudaBackend()
        hits = []
        original = backend._conv2d_grouped

        def spy(*args, **kwargs):
            hits.append(kwargs["groups"])
            return original(*args, **kwargs)

        backend._conv2d_grouped = spy

        dw, dw_feed = self._conv(
            (1, 4, 7, 7), (4, 1, 3, 3), {"groups": 4, "padding": (1, 1)}
        )
        backend.execute(dw, dw_feed)
        self.assertEqual(hits, [4])

        hits.clear()
        g, feed = self._conv(
            (2, 4, 8, 8), (6, 2, 3, 3), {"groups": 2, "padding": (1, 1)}
        )
        backend.execute(g, feed)
        self.assertEqual(hits, [])

    def test_grouped_rejects_forced_tiled(self):
        """强制 tiled 时分组卷积必须明确失败，而不是静默换成 im2col。"""
        g, feed = self._conv((1, 4, 8, 8), (6, 2, 3, 3), {"groups": 2})
        with self.assertRaises(BackendCapabilityError):
            CudaBackend(conv_impl="tiled").execute(g, feed)

    def test_conv_epilogue_fusion_is_bitwise_identical(self):
        """conv → BN（逐通道）→ relu 与"conv 落盘 + 逐元素链"逐位一致。

        融合只跳过中间值的物化，不改变任何一个乘加的次序与结合方式，因此
        两条路径必须逐位相同；一旦出现差异就说明表达式拼接或通道索引错了。
        """
        g = Graph()
        # 规模要落在 auto 选 Winograd 的窗口里：epilogue 融合挂在输出变换的
        # 写回点上，im2col / tiled 没有这个位置（那就退化成逐节点执行）。
        c = 16
        x = g.input("x", f32[2, c, 16, 16])
        w = g.constant(dtype=f32, shape=(c, c, 3, 3),
                       data=(self.rand(c, c, 3, 3) * 0.2).ravel().tolist())
        b = g.constant(dtype=f32, shape=(c,), data=self.rand(c).tolist())
        s = g.constant(dtype=f32, shape=(1, c, 1, 1),
                       data=self.rand(c).tolist())
        t = g.constant(dtype=f32, shape=(1, c, 1, 1),
                       data=self.rand(c).tolist())
        y = g.conv2d(x, w, b, padding=1)
        g.output(g.relu(g.add(g.mul(y, s), t)), "y")
        feed = {"x": self.rand(2, c, 16, 16)}

        fused = CudaBackend()
        plain = CudaBackend()
        plain._merge_conv_epilogue = lambda graph, groups: groups

        plan = fused._plan(g)
        self.assertIn(4, [len(p.nodes) for p in plan])
        got = fused.execute(g, feed)["y"]
        # 融合闭包在计划第一次执行时才绑定（``_prepare_plan``），所以要在
        # 运行之后再取计划。
        self.assertTrue(any(p.fused is not None for p in fused._plan(g)))
        self.assertTrue(np.array_equal(got, plain.execute(g, feed)["y"]))
        expected = self.ref.execute(g, feed)["y"]
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)


class TestCudnnFallback(unittest.TestCase):
    """auto 的回落面不依赖 GPU：engine 构造失败只记录、不冒泡。

    真机上会撞到的形态是"库在、符号没了"：cuDNN 版本不匹配时，取
    ``cudnnGetConvolutionForwardAlgorithm_v7`` 会抛 ``AttributeError``。
    它和"库不在"是同一类事实（这台机器用不了这条快路径），auto 必须回落；
    漏掉它会让默认的 ``conv_impl="auto"`` 在构造期就整体失败。

    这里用 ``__new__`` 绕过 ``__init__``，只测判定逻辑，不碰 GPU。
    """

    @staticmethod
    def _backend(impl: str):
        backend = CudaBackend.__new__(CudaBackend)
        backend.conv_impl = impl
        backend._cudnn = None
        backend._cudnn_broken = None
        backend._capturing = False
        return backend

    def test_auto_falls_back_when_symbols_are_missing(self):
        backend = self._backend("auto")
        with mock.patch(
            "aetherflow.backends.cuda.backend.CudnnConv",
            side_effect=AttributeError("no cudnnGetConvolutionForwardAlgorithm_v7"),
        ):
            # flops 要越过启用门槛，否则 _pick_cudnn 在碰库之前就返回了
            self.assertFalse(backend._pick_cudnn(np.dtype("float32"), 1, 5.0))
        self.assertIn("AttributeError", backend._cudnn_broken)
        # 失败只试一次：第二次不该再构造 engine
        with mock.patch(
            "aetherflow.backends.cuda.backend.CudnnConv",
            side_effect=AssertionError("不该再尝试构造"),
        ):
            self.assertFalse(backend._pick_cudnn(np.dtype("float32"), 1, 5.0))

    def test_explicit_cudnn_reports_the_reason(self):
        """显式 cudnn 不做回落：构造期就把原因报出来，别让排障的人猜。"""
        backend = self._backend("cudnn")
        with mock.patch(
            "aetherflow.backends.cuda.backend.CudnnConv",
            side_effect=AttributeError("no v7"),
        ):
            with self.assertRaises(BackendCapabilityError) as ctx:
                backend._cudnn_engine()
        self.assertIn("cuDNN", str(ctx.exception))

    def test_f64_and_grouped_never_pick_cudnn(self):
        backend = self._backend("auto")
        self.assertFalse(backend._pick_cudnn(np.dtype("float64"), 1))
        self.assertFalse(backend._pick_cudnn(np.dtype("float32"), 2))

    def test_capture_keeps_the_path_fixed(self):
        backend = self._backend("auto")
        backend._capturing = True
        self.assertFalse(backend._pick_cudnn(np.dtype("float32"), 1, 5.0))

    def test_auto_requires_enough_flops(self):
        """1 GFLOP 以下 cuDNN 的固定开销比省下的卷积还贵（5.252）。"""
        backend = self._backend("auto")
        with mock.patch("aetherflow.backends.cuda.backend.CudnnConv"):
            self.assertFalse(backend._pick_cudnn(np.dtype("float32"), 1, 0.3))
            self.assertTrue(backend._pick_cudnn(np.dtype("float32"), 1, 2.4))
        # 显式指定不受门槛约束：使用者明确要 cuDNN 就给他
        explicit = self._backend("cudnn")
        self.assertTrue(explicit._pick_cudnn(np.dtype("float32"), 1, 0.0))


@unittest.skipUnless(CUDA_AVAILABLE, "需要 cupy 与 CUDA 设备")
class TestCudaConvCudnn(unittest.TestCase):
    """cuDNN 快路径（5.74）：只覆盖 groups=1 的 f32/f16，结果与 im2col 逐点一致。

    这组用例同时是"回落语义"的守门人：cuDNN 不可用时整组跳过，而 auto 模式
    仍必须算对——``TestCudaConv`` 那组不指定 conv_impl，走的正是 auto。
    """

    @classmethod
    def setUpClass(cls):
        from aetherflow.backends.cuda.cudnn import available

        if not available():
            raise unittest.SkipTest("系统里没有可用的 cuDNN")
        cls.fast = CudaBackend(conv_impl="cudnn")
        cls.im2col = CudaBackend(conv_impl="im2col")
        # 库能加载不等于这台机器跑得动：cuDNN 9.28 要求 Linux 驱动 >= 570.26，
        # 在驱动 550 上句柄、描述符、v7 算法选择全部返回 0，执行期才报 5003
        # （CUDNN_STATUS_EXECUTION_FAILED_CUDART）。这让"显式要求 cuDNN"的
        # 一组用例整组变红，而根因是机器能力，与"没有 cuDNN"同性质——
        # 所以先用一个最小卷积探一次，能力不足就整组跳过（5.230）。
        try:
            probe = Graph()
            px = probe.input("x", f32[(1, 1, 3, 3)])
            pw = probe.input("w", f32[(1, 1, 1, 1)])
            probe.output(probe.add_node("conv2d", [px, pw], {}), "y")
            cls.fast.execute(
                probe,
                {
                    "x": np.ones((1, 1, 3, 3), np.float32),
                    "w": np.ones((1, 1, 1, 1), np.float32),
                },
            )
        except RuntimeError as exc:
            text = str(exc)
            if "5003" in text or "EXECUTION_FAILED" in text:
                raise unittest.SkipTest(f"cuDNN 可加载但执行失败（驱动代际不足）: {exc}")
            raise

    def _check(self, x_shape, w_shape, attrs=None, bias=True, dtype=f32, tol=1e-3):
        g = Graph()
        x = g.input("x", dtype[x_shape])
        w = g.input("w", dtype[w_shape])
        args = [x, w]
        rng = np.random.default_rng(7)
        np_dtype = np.float16 if dtype is f16 else np.float32
        feed = {
            "x": rng.standard_normal(x_shape).astype(np_dtype),
            "w": (rng.standard_normal(w_shape) * 0.3).astype(np_dtype),
        }
        if bias:
            b = g.input("b", dtype[w_shape[0]])
            args.append(b)
            feed["b"] = rng.standard_normal(w_shape[0]).astype(np_dtype)
        g.output(g.add_node("conv2d", args, attrs or {}), "y")

        want = self.im2col.execute(g, feed)["y"]
        got = self.fast.execute(g, feed)["y"]
        np.testing.assert_allclose(got, want, rtol=tol, atol=tol)

    def test_3x3_pad1(self):
        self._check((2, 3, 8, 8), (4, 3, 3, 3), {"padding": (1, 1)})

    def test_stride_and_dilation(self):
        self._check(
            (1, 2, 11, 11),
            (3, 2, 3, 3),
            {"stride": (2, 2), "dilation": (2, 2), "padding": (2, 2)},
        )

    def test_1x1_without_bias(self):
        self._check((3, 4, 5, 6), (7, 4, 1, 1), bias=False)

    def test_float16(self):
        self._check((2, 8, 16, 16), (16, 8, 3, 3), {"padding": (1, 1)}, dtype=f16, tol=1e-2)

    def test_wide_channels_keep_f32_precision(self):
        """宽通道 f32 卷积曾被 cuDNN 悄悄降到 TF32（差 5e-3，5.251）。

        这条用例的容差比默认的 1e-3 更紧：锁死 FMA 数学之后，两条路径只差
        累加顺序（~1e-5）。
        """
        self._check((2, 64, 16, 16), (64, 64, 3, 3), {"padding": (1, 1)}, tol=1e-4)

    def test_grouped_never_uses_cudnn(self):
        """groups>1 不进 cuDNN；强制 cudnn 时应回落到 im2col 且结果一致。"""
        self._check((2, 4, 8, 8), (6, 2, 3, 3), {"groups": 2, "padding": (1, 1)})


@unittest.skipUnless(CUDA_AVAILABLE, "需要 cupy 与 CUDA 设备")
class TestCudaConvTiled(unittest.TestCase):
    """分块直算核：与 im2col 路径、参考实现三方逐点一致。

    默认形状都很小，``conv_impl="auto"`` 会选 im2col；这里强制 tiled，
    让权重 staging、H/W 边界判断和"O 不是 OC tile 整数倍"的收尾也进入回归。
    """

    # 与 TestCudaConv 同一组参数：两条实现路径面对同一批语义边界
    SHAPES = (
        ((2, 3, 8, 8), (4, 3, 3, 3), {"padding": (1, 1)}, True),
        ((1, 2, 9, 9), (3, 2, 3, 3), {"stride": (2, 2)}, True),
        ((1, 2, 9, 9), (3, 2, 3, 3), {"dilation": (2, 2), "padding": (2, 2)}, True),
        ((3, 4, 5, 6), (7, 4, 1, 1), {}, False),
        ((2, 2, 7, 5), (3, 2, 3, 2), {"padding": (1, 0)}, True),
        ((1, 1, 10, 7), (2, 1, 2, 2), {"stride": (1, 2), "padding": (0, 1)}, True),
        # O=70 不是 32 的整数倍：grid.z 的最后一个 tile 只有 6 个有效通道
        ((2, 3, 9, 9), (70, 3, 3, 3), {"padding": (1, 1)}, True),
    )

    @classmethod
    def setUpClass(cls):
        cls.tiled = CudaBackend(conv_impl="tiled")
        cls.im2col = CudaBackend(conv_impl="im2col")

    def _build(self, x_shape, w_shape, attrs, bias):
        g = Graph()
        x = g.input("x", f32[x_shape])
        w = g.input("w", f32[w_shape])
        args = [x, w]
        rng = np.random.default_rng(0)
        feed = {
            "x": rng.standard_normal(x_shape).astype(np.float32),
            "w": (rng.standard_normal(w_shape) * 0.3).astype(np.float32),
        }
        if bias:
            b = g.input("b", f32[w_shape[0]])
            args.append(b)
            feed["b"] = rng.standard_normal(w_shape[0]).astype(np.float32)
        g.output(g.add_node("conv2d", args, attrs), "y")
        return g, feed

    def test_matches_reference_and_im2col(self):
        ref = ReferenceBackend()
        for x_shape, w_shape, attrs, bias in self.SHAPES:
            with self.subTest(x=x_shape, w=w_shape, attrs=attrs, bias=bias):
                g, feed = self._build(x_shape, w_shape, attrs, bias)
                got = self.tiled.execute(g, feed)["y"]
                np.testing.assert_allclose(
                    got, ref.execute(g, feed)["y"], rtol=1e-4, atol=1e-4
                )
                np.testing.assert_allclose(
                    got, self.im2col.execute(g, feed)["y"], rtol=1e-4, atol=1e-4
                )

    def test_forced_tiled_ignores_auto_thresholds(self):
        """强制 tiled 时绕过 auto 的规模门限：门限只是选择策略，不是正确性前提。

        这个形状的输出只有 2*3*3*3 = 54 个元素，auto 会判给 im2col。
        """
        g, feed = self._build((2, 3, 4, 4), (3, 3, 3, 3), {"padding": (1, 1)}, True)
        np.testing.assert_allclose(
            self.tiled.execute(g, feed)["y"],
            self.im2col.execute(g, feed)["y"],
            rtol=1e-4,
            atol=1e-4,
        )


@unittest.skipUnless(CUDA_AVAILABLE, "需要 cupy 与 CUDA 设备")
class TestCudaConvImplicit(unittest.TestCase):
    """隐式 GEMM 核：输入窗口进 shared，与参考实现、im2col 三方逐点一致。

    这组用例同时覆盖两件容易错的事：窗口 staging 的跨列重叠（stride=1 时相邻
    线程读到的窗口有 ``kw-1`` 列重合），以及 padding 边界——边界不再在累加里
    逐点判断，而是在 staging 时填 0，所以"贴边输出"是必须显式测的形状。
    """

    SHAPES = (
        ((2, 3, 8, 8), (4, 3, 3, 3), {"padding": (1, 1)}, True),
        ((1, 2, 9, 9), (3, 2, 3, 3), {"stride": (2, 2)}, True),
        ((1, 2, 9, 9), (3, 2, 3, 3), {"dilation": (2, 2), "padding": (2, 2)}, True),
        ((3, 4, 5, 6), (7, 4, 1, 1), {}, False),
        ((2, 2, 7, 5), (3, 2, 3, 2), {"padding": (1, 0)}, True),
        ((1, 1, 10, 7), (2, 1, 2, 2), {"stride": (1, 2), "padding": (0, 1)}, True),
        # K = 64*9 = 576：正是分块直算判给 im2col 的那一段，隐式 GEMM 的目标形状
        ((1, 64, 16, 64), (32, 64, 3, 3), {"padding": (1, 1)}, True),
    )

    @classmethod
    def setUpClass(cls):
        cls.implicit = CudaBackend(conv_impl="implicit")
        cls.im2col = CudaBackend(conv_impl="im2col")

    def _build(self, x_shape, w_shape, attrs, bias):
        g = Graph()
        x = g.input("x", f32[x_shape])
        w = g.input("w", f32[w_shape])
        args = [x, w]
        rng = np.random.default_rng(0)
        feed = {
            "x": rng.standard_normal(x_shape).astype(np.float32),
            "w": (rng.standard_normal(w_shape) * 0.3).astype(np.float32),
        }
        if bias:
            b = g.input("b", f32[w_shape[0]])
            args.append(b)
            feed["b"] = rng.standard_normal(w_shape[0]).astype(np.float32)
        g.output(g.add_node("conv2d", args, attrs), "y")
        return g, feed

    def test_matches_reference_and_im2col(self):
        ref = ReferenceBackend()
        for x_shape, w_shape, attrs, bias in self.SHAPES:
            with self.subTest(x=x_shape, w=w_shape, attrs=attrs, bias=bias):
                g, feed = self._build(x_shape, w_shape, attrs, bias)
                got = self.implicit.execute(g, feed)["y"]
                np.testing.assert_allclose(
                    got, ref.execute(g, feed)["y"], rtol=1e-4, atol=1e-4
                )
                np.testing.assert_allclose(
                    got, self.im2col.execute(g, feed)["y"], rtol=1e-4, atol=1e-4
                )

    def test_forced_implicit_works_on_small_shapes(self):
        """强制 implicit 时不分规模：它是显式选择的对照路径，不是自动策略。

        5.151 的 A/B 里显式路径全部慢于 im2col，因此 auto 从不选它；但"强制
        选了它也要算对"仍然是这条路径存在的前提。
        """
        g, feed = self._build((2, 3, 4, 4), (3, 3, 3, 3), {"padding": (1, 1)}, True)
        np.testing.assert_allclose(
            self.implicit.execute(g, feed)["y"],
            self.im2col.execute(g, feed)["y"],
            rtol=1e-4,
            atol=1e-4,
        )

    def test_rejects_float16(self):
        g = Graph()
        x = g.input("x", f16[(1, 2, 6, 6)])
        w = g.input("w", f16[(3, 2, 3, 3)])
        g.output(g.add_node("conv2d", [x, w], {"padding": (1, 1)}), "y")
        with self.assertRaises(BackendCapabilityError):
            self.implicit.execute(
                g,
                {
                    "x": np.zeros((1, 2, 6, 6), dtype=np.float16),
                    "w": np.zeros((3, 2, 3, 3), dtype=np.float16),
                },
            )

    def test_rejects_groups(self):
        g = Graph()
        x = g.input("x", f32[(1, 4, 6, 6)])
        w = g.input("w", f32[(4, 2, 3, 3)])
        g.output(
            g.add_node("conv2d", [x, w], {"groups": 2, "padding": (1, 1)}), "y"
        )
        with self.assertRaises(BackendCapabilityError):
            self.implicit.execute(
                g,
                {
                    "x": np.zeros((1, 4, 6, 6), dtype=np.float32),
                    "w": np.zeros((4, 2, 3, 3), dtype=np.float32),
                },
            )


@unittest.skipUnless(CUDA_AVAILABLE, "需要 cupy 与 CUDA 设备")
class TestCudaAnyAxis(unittest.TestCase):
    """非末维归约与 softmax：末维走块内归约，其余轴走通用串行路径。

    这两条路径的分界点必须测到两侧：同一个算子、同样的形状，只把 ``axis``
    从 -1 换成 0/1，结果都得和参考后端一致，且形状（keepdims）不受路径影响。
    """

    @classmethod
    def setUpClass(cls):
        cls.backend = CudaBackend()
        cls.ref = ReferenceBackend()

    def _run(self, op, shape, axis, keepdims=False, dtype="f32"):
        g = Graph()
        x = g.input("x", f32[shape] if dtype == "f32" else f16[shape])
        op_fn = getattr(g, op)
        node = (
            op_fn(x, axis=axis, keepdims=keepdims)
            if op.startswith("reduce")
            else op_fn(x, axis=axis)
        )
        g.output(node, "y")
        rng = np.random.default_rng(0)
        feed = {"x": rng.standard_normal(shape).astype(np.float32)}
        if dtype == "f16":
            feed["x"] = feed["x"].astype(np.float16)
        return (
            self.backend.execute(g, feed)["y"],
            self.ref.execute(g, feed)["y"],
        )

    def test_reduce_non_last_axis(self):
        for op in ("reduce_sum", "reduce_max", "reduce_mean"):
            for axis in (0, 1, 2):
                with self.subTest(op=op, axis=axis):
                    got, want = self._run(op, (3, 4, 5), axis)
                    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

    def test_reduce_keepdims_shape(self):
        got, want = self._run("reduce_sum", (3, 4, 5), 1, keepdims=True)
        self.assertEqual(got.shape, want.shape)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

    def test_reduce_matches_last_dim_path(self):
        """同一组数据，axis=-1 走块内归约、axis=1 走非末维路径，结果必须一致。

        (2, 6, 4) 的 ``inner`` 是 4，非末维一侧走两维铺法（5.174）；两条路的
        归约顺序不同，总和必须一致到浮点容差以内。
        """
        last, _ = self._run("reduce_sum", (2, 6, 4), -1)
        mid, _ = self._run("reduce_sum", (2, 6, 4), 1)
        np.testing.assert_allclose(last.sum(), mid.sum(), rtol=1e-4)

    def test_reduce_float16_non_last_axis(self):
        got, want = self._run("reduce_sum", (2, 3, 4), 1, dtype="f16")
        self.assertEqual(got.dtype, np.float16)
        np.testing.assert_allclose(
            got.astype(np.float32), want.astype(np.float32), rtol=2e-3, atol=2e-3
        )

    def test_reduce_narrow_inner_uses_two_dim_tiling(self):
        """``inner`` 窄到一个 warp 都凑不满：走两维铺法（5.174）。

        这些形状是一维铺法最差的格子：输出平面只有几十个元素，归约轴却要
        串行走完；两维铺法把 ``blockDim.y`` 摊到 reduce 轴上。数值必须不变。
        """
        for op in ("reduce_sum", "reduce_max", "reduce_mean"):
            for shape, axis in (((16, 37, 5), 1), ((8, 5, 3), 1), ((6, 9), 0)):
                with self.subTest(op=op, shape=shape, axis=axis):
                    got, want = self._run(op, shape, axis)
                    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

    def test_reduce_narrow_inner_float16(self):
        """f16 的容差比 f32 宽：参考实现按 f16 累加，本内核按 float 累加。

        ``reduce_len=33`` 时参考实现要走 33 次 f16 加法，单次舍入就有半个
        ulp（量级 10 处约 0.005），而内核先读成 float、树归约后只在写回时
        舍入一次——两边都不"错"，差的只是累加顺序与中间精度。
        """
        got, want = self._run("reduce_sum", (8, 33, 3), 1, dtype="f16")
        self.assertEqual(got.dtype, np.float16)
        np.testing.assert_allclose(
            got.astype(np.float32), want.astype(np.float32), rtol=1e-2, atol=1e-2
        )

    def test_reduce_wide_plane_keeps_one_dim_path(self):
        """``inner >= 32`` 且输出平面够大：仍走一维铺法，结果不变。

        这条路径与两维铺法只差线程铺法，数值语义相同；用例放在这里是为了
        保证两条路都真的被跑到（形状按 ``sm_count * BLOCK_DEFAULT`` 选）。
        """
        got, want = self._run("reduce_sum", (1024, 64, 64), 1)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

    def test_softmax_non_last_axis(self):
        for axis in (0, 1):
            with self.subTest(axis=axis):
                got, want = self._run("softmax", (4, 6), axis)
                np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(
                    got.sum(axis=axis), np.ones_like(got.sum(axis=axis)),
                    rtol=1e-5, atol=1e-5,
                )

    def test_softmax_rank3_middle_axis(self):
        got, want = self._run("softmax", (2, 5, 3), 1)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

    def test_softmax_float16_non_last_axis(self):
        got, want = self._run("softmax", (3, 5), 0, dtype="f16")
        np.testing.assert_allclose(
            got.astype(np.float32), want.astype(np.float32), rtol=3e-3, atol=3e-3
        )

    def test_softmax_empty_axis_rejected(self):
        g = Graph()
        x = g.input("x", f32[(2, 0, 3)])
        g.output(g.softmax(x, axis=1), "y")
        with self.assertRaises(BackendCapabilityError):
            self.backend.execute(g, {"x": np.zeros((2, 0, 3), dtype=np.float32)})

    def _assert_norm(self, op, shape, axes, param_shape, residual=False, dtype="f32"):
        """归一化算子的"同一组数据、不同 axes"对照：后端只改布局，数值不变。"""
        t = f32 if dtype == "f32" else f16
        np_dtype = np.float32 if dtype == "f32" else np.float16
        g = Graph()
        x = g.input("x", t[shape])
        gamma = g.input("gamma", t[param_shape])
        args = [x, gamma]
        if op == "layernorm":
            args.append(g.input("beta", t[param_shape]))
        if residual:
            args.append(g.input("r", t[shape]))
        node = getattr(g, op)(*args, axes=axes)
        g.output(node, "y")
        rng = np.random.default_rng(2)
        feed = {
            "x": rng.standard_normal(shape).astype(np_dtype),
            "gamma": (rng.standard_normal(param_shape) * 0.5 + 1.0).astype(np_dtype),
        }
        if op == "layernorm":
            feed["beta"] = (rng.standard_normal(param_shape) * 0.2).astype(np_dtype)
        if residual:
            feed["r"] = rng.standard_normal(shape).astype(np_dtype)
        got = self.backend.execute(g, feed)["y"]
        want = self.ref.execute(g, feed)["y"]
        self.assertEqual(got.shape, want.shape)
        tol = 1e-5 if dtype == "f32" else 5e-3
        np.testing.assert_allclose(
            got.astype(np.float32), want.astype(np.float32), rtol=tol, atol=tol
        )

    def test_layernorm_multi_axis_suffix(self):
        """axes 是最后两维：keep 之后直接是 norm，无需置换。"""
        self._assert_norm("layernorm", (2, 3, 4), (1, 2), (3, 4))

    def test_layernorm_non_last_axis(self):
        """axes 落在首维：keep=(1,2)、norm=(0,)，要置换再换回。"""
        self._assert_norm("layernorm", (3, 4, 5), (0,), (3, 1, 1))

    def test_layernorm_non_contiguous_axes(self):
        """axes=(0,2)：归一化维在内存里被中间维隔开，置换后展平必须对齐。"""
        self._assert_norm("layernorm", (2, 3, 4), (0, 2), (2, 1, 4))

    def test_layernorm_residual_non_last_axis(self):
        self._assert_norm("layernorm", (3, 4, 5), (1,), (1, 4, 1), residual=True)

    def test_layernorm_float16_non_last_axis(self):
        self._assert_norm("layernorm", (2, 3, 4), (0,), (2, 1, 1), dtype="f16")

    def test_rmsnorm_multi_axis(self):
        """gamma 在归一化首轴上取 1：沿该轴广播，展平后必须复制出 8 个系数。"""
        self._assert_norm("rmsnorm", (2, 3, 4), (0, 2), (1, 1, 4))

    def test_norm_param_must_not_vary_per_row(self):
        """gamma 在非归一化维上不是 1：每行要用不同参数，内核装不下，显式拒绝。"""
        g = Graph()
        x = g.input("x", f32[2, 3, 4])
        gamma = g.input("gamma", f32[2, 4])
        beta = g.input("beta", f32[2, 4])
        g.output(g.layernorm(x, gamma, beta, axes=(1, 2)), "y")
        feed = {
            "x": np.zeros((2, 3, 4), dtype=np.float32),
            "gamma": np.ones((2, 4), dtype=np.float32),
            "beta": np.zeros((2, 4), dtype=np.float32),
        }
        with self.assertRaises(BackendCapabilityError):
            self.backend.execute(g, feed)


class TestCudaMaxPool(CudaTestCase):
    """纯索引算子：错误只会出现在边界索引上，所以用例集中在 padding/stride 组合。"""

    def _pool(self, x_shape, attrs):
        g = Graph()
        x = g.input("x", f32[x_shape])
        g.output(g.maxpool2d(x, **attrs), "y")
        return g, {"x": self.rand(*x_shape)}

    def test_2x2_stride2(self):
        g, feed = self._pool((2, 4, 8, 8), {"kernel": 2, "stride": 2})
        self.assert_backends_equal(g, feed)

    def test_default_stride_equals_kernel(self):
        g, feed = self._pool((1, 3, 7, 7), {"kernel": 3})
        self.assert_backends_equal(g, feed)

    def test_overlapping_with_padding(self):
        g, feed = self._pool((1, 2, 6, 5), {"kernel": 3, "stride": 1, "padding": 1})
        self.assert_backends_equal(g, feed)

    def test_non_square_kernel_and_stride(self):
        g, feed = self._pool((2, 3, 9, 7), {"kernel": (3, 2), "stride": (2, 1)})
        self.assert_backends_equal(g, feed)

    def test_negative_values_not_clamped_to_zero(self):
        """全局负值时输出必须仍是负数：padding 用 -inf 而不是 0。"""
        g, _ = self._pool((1, 2, 4, 4), {"kernel": 2, "padding": 1})
        feed = {"x": -np.abs(self.rand(1, 2, 4, 4)) - 1.0}
        out = self.assert_backends_equal(g, feed)
        self.assertTrue((out < 0).all())


class TestCudaAvgPool(CudaTestCase):
    """平均池化：与 maxpool 共用索引，但分母固定为 kernel 面积。"""

    def _pool(self, x_shape, attrs):
        g = Graph()
        x = g.input("x", f32[x_shape])
        g.output(g.avgpool2d(x, **attrs), "y")
        return g, {"x": self.rand(*x_shape)}

    def test_2x2_stride2(self):
        g, feed = self._pool((2, 4, 8, 8), {"kernel": 2, "stride": 2})
        self.assert_backends_equal(g, feed)

    def test_default_stride_equals_kernel(self):
        g, feed = self._pool((1, 3, 7, 7), {"kernel": 3})
        self.assert_backends_equal(g, feed)

    def test_overlapping_with_padding(self):
        g, feed = self._pool((1, 2, 6, 5), {"kernel": 3, "stride": 1, "padding": 1})
        self.assert_backends_equal(g, feed)

    def test_non_square_kernel_and_stride(self):
        g, feed = self._pool((2, 3, 9, 7), {"kernel": (3, 2), "stride": (2, 1)})
        self.assert_backends_equal(g, feed)

    def test_divisor_is_kernel_area_not_valid_count(self):
        """padding 的零也计入分母：角点只有一个有效 tap，值必须是 1/4 而不是 1。"""
        g, _ = self._pool((1, 1, 3, 3), {"kernel": 2, "stride": 2, "padding": 1})
        out = self.assert_backends_equal(g, {"x": np.ones((1, 1, 3, 3), dtype=np.float32)})
        self.assertEqual(tuple(out.shape), (1, 1, 2, 2))
        np.testing.assert_allclose(out[0, 0, 0, 0], 0.25)


class TestCudaAvgPoolBackward(CudaTestCase):
    """反向必须散射累加：重叠窗口会在同一个输入格上叠加。"""

    def _graph(self, dy_shape, attrs):
        g = Graph()
        dy = g.input("dy", f32[dy_shape])
        g.output(g.avgpool2d_backward(dy, **attrs), "y")
        return g, {"dy": self.rand(*dy_shape)}

    def test_stride2_no_overlap(self):
        g, feed = self._graph(
            (1, 4, 4, 4), {"input_shape": [1, 4, 8, 8], "kernel": (2, 2), "stride": (2, 2)}
        )
        self.assert_backends_equal(g, feed)

    def test_overlapping_windows_accumulate(self):
        g, feed = self._graph(
            (1, 2, 6, 6),
            {"input_shape": [1, 2, 6, 6], "kernel": (3, 3), "stride": (1, 1), "padding": (1, 1)},
        )
        got = self.assert_backends_equal(g, feed)
        # 中心格被 9 个窗口覆盖，必须大于只被 1 个窗口覆盖的角落
        self.assertGreater(abs(float(got[0, 0, 3, 3])), abs(float(got[0, 0, 0, 0])))

    def test_requires_matching_dy_shape(self):
        g, _ = self._graph(
            (1, 2, 5, 5), {"input_shape": [1, 2, 8, 8], "kernel": (2, 2), "stride": (2, 2)}
        )
        with self.assertRaises(BackendCapabilityError):
            self.cuda.execute(g, {"dy": self.rand(1, 2, 5, 5)})


class TestCudaAdaptiveAvgPool(CudaTestCase):
    """自适应池化：窗口端点随输出位置变化，非整除时窗口重叠。

    用例刻意选 H/OH、W/OW 都不能整除的形状（7/2、5/3），
    这样 floor/ceil 的窗口端点计算才会真正被覆盖。
    """

    def _pool(self, x_shape, output_size):
        g = Graph()
        x = g.input("x", f32[x_shape])
        g.output(g.adaptive_avg_pool2d(x, output_size=output_size), "y")
        return g, {"x": self.rand(*x_shape)}

    def test_non_divisible(self):
        g, feed = self._pool((1, 3, 7, 5), (2, 3))
        got = self.assert_backends_equal(g, feed)
        self.assertEqual(got.shape, (1, 3, 2, 3))

    def test_global_pool(self):
        """output_size=(1,1) 就是全局平均池化：应等于整图均值。"""
        g, feed = self._pool((2, 4, 6, 5), (1, 1))
        got = self.assert_backends_equal(g, feed)
        np.testing.assert_allclose(
            got.reshape(2, 4), feed["x"].mean(axis=(-2, -1)), rtol=1e-5, atol=1e-5
        )

    def test_downsample_by_two(self):
        g, feed = self._pool((1, 2, 8, 8), (4, 4))
        self.assert_backends_equal(g, feed)

    def test_upsample_larger_output(self):
        """输出比输入大时窗口仍非空（start<end），必须逐格正确。"""
        g, feed = self._pool((1, 2, 3, 3), (5, 4))
        self.assert_backends_equal(g, feed)

    def test_int_output_size(self):
        g = Graph()
        x = g.input("x", f32[1, 2, 6, 6])
        g.output(g.adaptive_avg_pool2d(x, output_size=3), "y")
        got = self.assert_backends_equal(g, {"x": self.rand(1, 2, 6, 6)})
        self.assertEqual(got.shape, (1, 2, 3, 3))


class TestCudaAdaptiveAvgPoolBackward(CudaTestCase):
    """反向必须散射累加：非整除时同一输入格会被多个窗口覆盖。"""

    def _graph(self, dy_shape, input_shape, output_size):
        g = Graph()
        dy = g.input("dy", f32[dy_shape])
        g.output(
            g.adaptive_avg_pool2d_backward(
                dy, input_shape=input_shape, output_size=output_size
            ),
            "y",
        )
        return g, {"dy": self.rand(*dy_shape)}

    def test_non_overlapping_windows(self):
        g, feed = self._graph((1, 2, 4, 4), [1, 2, 8, 8], (4, 4))
        self.assert_backends_equal(g, feed)

    def test_overlapping_windows_accumulate(self):
        g, feed = self._graph((1, 2, 2, 3), [1, 2, 7, 5], (2, 3))
        self.assert_backends_equal(g, feed)

    def test_shape_mismatch_rejected(self):
        g, feed = self._graph((1, 2, 3, 3), [1, 2, 7, 5], (2, 3))
        with self.assertRaises(BackendCapabilityError):
            self.cuda.execute(g, feed)


class TestCudaPaddingAndMaxPoolBackward(CudaTestCase):
    def test_pad_middle_dim(self):
        g = Graph()
        x = g.input("x", f32[2, 3, 4])
        g.output(g.pad(x, dim=1, before=2, after=1), "y")
        got = self.assert_backends_equal(g, {"x": self.rand(2, 3, 4)})
        self.assertEqual(got.shape, (2, 6, 4))

    def test_pad_first_dim(self):
        g = Graph()
        x = g.input("x", f32[3, 4])
        g.output(g.pad(x, dim=0, before=1, after=2), "y")
        self.assert_backends_equal(g, {"x": self.rand(3, 4)})

    def test_pad_last_dim(self):
        g = Graph()
        x = g.input("x", f32[2, 3, 5])
        g.output(g.pad(x, dim=-1, before=0, after=3), "y")
        self.assert_backends_equal(g, {"x": self.rand(2, 3, 5)})

    def test_maxpool2d_backward_stride2(self):
        g = Graph()
        x = g.input("x", f32[1, 2, 6, 6])
        dy = g.input("dy", f32[1, 2, 3, 3])
        g.output(g.maxpool2d_backward(x, dy, kernel=2, stride=2), "y")
        self.assert_backends_equal(
            g, {"x": self.rand(1, 2, 6, 6), "dy": self.rand(1, 2, 3, 3)}
        )

    def test_maxpool2d_backward_with_padding(self):
        g = Graph()
        x = g.input("x", f32[1, 2, 7, 7])
        dy = g.input("dy", f32[1, 2, 4, 4])
        g.output(g.maxpool2d_backward(x, dy, kernel=3, stride=2, padding=1), "y")
        self.assert_backends_equal(
            g, {"x": self.rand(1, 2, 7, 7), "dy": self.rand(1, 2, 4, 4)}
        )

    def test_maxpool2d_backward_ties_pick_first(self):
        """全相同的输入下每个窗口都并列最大：必须落到行优先的第一个格子。"""
        g = Graph()
        x = g.input("x", f32[1, 1, 4, 4])
        dy = g.input("dy", f32[1, 1, 2, 2])
        g.output(g.maxpool2d_backward(x, dy, kernel=2, stride=2), "y")
        got = self.assert_backends_equal(
            g,
            {
                "x": np.ones((1, 1, 4, 4), dtype=np.float32),
                "dy": np.ones((1, 1, 2, 2), dtype=np.float32),
            },
        )
        expected = np.zeros((1, 1, 4, 4), dtype=np.float32)
        expected[0, 0, 0, 0] = 1.0
        expected[0, 0, 0, 2] = 1.0
        expected[0, 0, 2, 0] = 1.0
        expected[0, 0, 2, 2] = 1.0
        np.testing.assert_allclose(got, expected)

    def test_maxpool2d_backward_requires_matching_dy(self):
        g = Graph()
        x = g.input("x", f32[1, 2, 6, 6])
        dy = g.input("dy", f32[1, 2, 2, 2])
        g.output(g.maxpool2d_backward(x, dy, kernel=2, stride=2), "y")
        with self.assertRaises(BackendCapabilityError):
            self.cuda.execute(
                g, {"x": self.rand(1, 2, 6, 6), "dy": self.rand(1, 2, 2, 2)}
            )

    def test_maxpool2d_ceil_mode(self):
        """ceil_mode 的最后一格越过输入末尾：窗口截断，padding 不参与取 max。"""
        g = Graph()
        x = g.input("x", f32[1, 2, 8, 8])
        g.output(g.maxpool2d(x, kernel=3, stride=2, ceil_mode=True), "y")
        got = self.assert_backends_equal(g, {"x": self.rand(1, 2, 8, 8)})
        self.assertEqual(got.shape, (1, 2, 4, 4))

    def test_maxpool2d_ceil_mode_padding(self):
        g = Graph()
        x = g.input("x", f32[1, 2, 6, 5])
        g.output(
            g.maxpool2d(x, kernel=3, stride=2, padding=1, ceil_mode=True), "y"
        )
        self.assert_backends_equal(g, {"x": self.rand(1, 2, 6, 5)})

    def test_avgpool2d_ceil_mode(self):
        g = Graph()
        x = g.input("x", f32[1, 2, 8, 8])
        g.output(g.avgpool2d(x, kernel=3, stride=2, ceil_mode=True), "y")
        got = self.assert_backends_equal(g, {"x": self.rand(1, 2, 8, 8)})
        self.assertEqual(got.shape, (1, 2, 4, 4))

    def test_avgpool2d_count_include_pad_false(self):
        """分母换成有效点数后，边界格与内部格不再是同一个常数缩放。"""
        g = Graph()
        x = g.input("x", f32[1, 2, 8, 8])
        g.output(
            g.avgpool2d(
                x, kernel=3, stride=1, padding=1, count_include_pad=False
            ),
            "y",
        )
        self.assert_backends_equal(g, {"x": self.rand(1, 2, 8, 8)})

    def test_avgpool2d_backward_count_include_pad_false(self):
        g = Graph()
        dy = g.input("dy", f32[1, 2, 8, 8])
        g.output(
            g.avgpool2d_backward(
                dy,
                input_shape=[1, 2, 8, 8],
                kernel=3,
                stride=1,
                padding=1,
                count_include_pad=False,
            ),
            "y",
        )
        self.assert_backends_equal(g, {"dy": self.rand(1, 2, 8, 8)})

    def test_avgpool2d_backward_ceil_mode(self):
        g = Graph()
        dy = g.input("dy", f32[1, 2, 4, 4])
        g.output(
            g.avgpool2d_backward(
                dy, input_shape=[1, 2, 8, 8], kernel=3, stride=2, ceil_mode=True
            ),
            "y",
        )
        got = self.assert_backends_equal(g, {"dy": self.rand(1, 2, 4, 4)})
        self.assertEqual(got.shape, (1, 2, 8, 8))

    def test_maxpool2d_indices_matches_reference(self):
        g = Graph()
        x = g.input("x", f32[1, 2, 6, 7])
        g.output(
            g.maxpool2d_indices(x, kernel=3, stride=2, padding=1), "y"
        )
        got = self.assert_backends_equal(g, {"x": self.rand(1, 2, 6, 7)})
        self.assertEqual(got.dtype, np.int32)
        self.assertEqual(got.shape, (1, 2, 3, 4))

    def test_maxpool2d_indices_ties_pick_first(self):
        """全相同输入下并列最大值取行优先第一个：与反向散射落点一致。"""
        g = Graph()
        x = g.input("x", f32[1, 1, 4, 4])
        g.output(g.maxpool2d_indices(x, kernel=2, stride=2), "y")
        got = self.assert_backends_equal(
            g, {"x": np.ones((1, 1, 4, 4), dtype=np.float32)}
        )
        # 每个 2x2 窗口的第一个格子：平面内下标 0/2/8/10。
        np.testing.assert_array_equal(
            got, np.array([[[[0, 2], [8, 10]]]], dtype=np.int32)
        )

    def test_maxpool2d_indices_float16(self):
        g = Graph()
        x = g.input("x", f16[1, 2, 6, 6])
        g.output(g.maxpool2d_indices(x, kernel=2, stride=2), "y")
        self.assert_backends_equal(
            g, {"x": self.rand(1, 2, 6, 6).astype(np.float16)}
        )


class TestCudaConvGrad(CudaTestCase):
    def test_input_grad_pad1(self):
        g = Graph()
        dy = g.input("dy", f32[2, 4, 6, 6])
        w = g.input("w", f32[4, 3, 3, 3])
        g.output(
            g.conv2d_input_grad(dy, w, input_shape=[2, 3, 6, 6], stride=1, padding=1),
            "y",
        )
        self.assert_backends_equal(g, {"dy": self.rand(2, 4, 6, 6), "w": self.rand(4, 3, 3, 3)})

    def test_input_grad_stride_and_dilation(self):
        g = Graph()
        dy = g.input("dy", f32[1, 2, 2, 2])
        w = g.input("w", f32[2, 3, 3, 3])
        g.output(
            g.conv2d_input_grad(
                dy, w, input_shape=[1, 3, 7, 7], stride=2, dilation=2
            ),
            "y",
        )
        self.assert_backends_equal(g, {"dy": self.rand(1, 2, 2, 2), "w": self.rand(2, 3, 3, 3)})

    def test_input_grad_grouped(self):
        """groups=2：每组只看自己那几个输入通道，dx 的通道切片必须精确对应。"""
        g = Graph()
        dy = g.input("dy", f32[1, 4, 6, 6])
        w = g.input("w", f32[4, 2, 3, 3])
        g.output(
            g.conv2d_input_grad(
                dy, w, input_shape=[1, 4, 6, 6], stride=1, padding=1, groups=2
            ),
            "y",
        )
        self.assert_backends_equal(
            g, {"dy": self.rand(1, 4, 6, 6), "w": self.rand(4, 2, 3, 3)}
        )

    def test_input_grad_rejects_mismatched_groups(self):
        """通道数与 groups 不匹配在构图期就被 IR 拒绝。"""
        g = Graph()
        dy = g.input("dy", f32[1, 2, 6, 6])
        w = g.input("w", f32[2, 3, 3, 3])
        with self.assertRaises(ValueError):
            g.conv2d_input_grad(dy, w, input_shape=[1, 3, 6, 6], groups=2)

    def test_weight_grad_pad1(self):
        g = Graph()
        x = g.input("x", f32[2, 3, 6, 6])
        dy = g.input("dy", f32[2, 4, 6, 6])
        g.output(
            g.conv2d_weight_grad(x, dy, weight_shape=[4, 3, 3, 3], stride=1, padding=1),
            "y",
        )
        got = self.assert_backends_equal(
            g, {"x": self.rand(2, 3, 6, 6), "dy": self.rand(2, 4, 6, 6)}
        )
        self.assertEqual(got.shape, (4, 3, 3, 3))

    def test_weight_grad_stride2(self):
        g = Graph()
        x = g.input("x", f32[1, 3, 8, 8])
        dy = g.input("dy", f32[1, 2, 4, 4])
        g.output(
            g.conv2d_weight_grad(x, dy, weight_shape=[2, 3, 2, 2], stride=2),
            "y",
        )
        self.assert_backends_equal(g, {"x": self.rand(1, 3, 8, 8), "dy": self.rand(1, 2, 4, 4)})

    def test_weight_grad_grouped(self):
        """groups=2：每组权重只从自己那对 (x, dy) 通道切片累加。"""
        g = Graph()
        x = g.input("x", f32[1, 4, 6, 6])
        dy = g.input("dy", f32[1, 4, 6, 6])
        g.output(
            g.conv2d_weight_grad(
                x, dy, weight_shape=[4, 2, 3, 3], stride=1, padding=1, groups=2
            ),
            "y",
        )
        got = self.assert_backends_equal(
            g, {"x": self.rand(1, 4, 6, 6), "dy": self.rand(1, 4, 6, 6)}
        )
        self.assertEqual(got.shape, (4, 2, 3, 3))

    def test_weight_grad_depthwise(self):
        g = Graph()
        x = g.input("x", f32[2, 4, 5, 5])
        dy = g.input("dy", f32[2, 4, 5, 5])
        g.output(
            g.conv2d_weight_grad(
                x, dy, weight_shape=[4, 1, 3, 3], stride=1, padding=1, groups=4
            ),
            "y",
        )
        self.assert_backends_equal(
            g, {"x": self.rand(2, 4, 5, 5), "dy": self.rand(2, 4, 5, 5)}
        )


class TestCudaActivations(CudaTestCase):
    def test_gelu(self):
        g = Graph()
        x = g.input("x", f32[7, 9])
        g.output(g.gelu(x), "y")
        self.assert_backends_equal(g, {"x": self.rand(7, 9) * 3})

    def test_gelu_saturation(self):
        """两端饱和区必须仍是 0 / x，不能出现 NaN（tanh 的输入很大）。"""
        g = Graph()
        x = g.input("x", f32[1, 5])
        g.output(g.gelu(x), "y")
        feed = {"x": np.array([[-30.0, -3.0, 0.0, 3.0, 30.0]], dtype=np.float32)}
        self.assert_backends_equal(g, feed)

    def test_rmsnorm(self):
        g = Graph()
        x = g.input("x", f32[5, 16])
        gamma = g.input("gamma", f32[16])
        g.output(g.rmsnorm(x, gamma, axes=(-1,), eps=1e-6), "y")
        feed = {"x": self.rand(5, 16), "gamma": self.rand(16) + 1.0}
        self.assert_backends_equal(g, feed)

    def test_rmsnorm_gamma_broadcast_shape(self):
        """gamma 是 (1, cols) 时展平后与 (cols,) 逐元素相同，内核直接用。"""
        g = Graph()
        x = g.input("x", f32[5, 16])
        gamma = g.input("gamma", f32[1, 16])
        g.output(g.rmsnorm(x, gamma, axes=(-1,)), "y")
        feed = {"x": self.rand(5, 16), "gamma": self.rand(1, 16) + 1.0}
        self.assert_backends_equal(g, feed)

    def test_rmsnorm_non_last_axis(self):
        g = Graph()
        x = g.input("x", f32[4, 6])
        gamma = g.input("gamma", f32[4, 1])
        g.output(g.rmsnorm(x, gamma, axes=(0,)), "y")
        feed = {"x": self.rand(4, 6), "gamma": self.rand(4, 1) + 1.0}
        self.assert_backends_equal(g, feed)


class TestCudaIndexed(CudaTestCase):
    """拼接与查表：索引类算子的正确性集中在"偏移'与'越界'两件事上。"""

    def test_concat_axis1(self):
        g = Graph()
        a = g.input("a", f32[3, 4])
        b = g.input("b", f32[3, 2])
        c = g.input("c", f32[3, 5])
        g.output(g.concat(a, b, c, axis=1), "y")
        feed = {"a": self.rand(3, 4), "b": self.rand(3, 2), "c": self.rand(3, 5)}
        self.assert_backends_equal(g, feed)

    def test_concat_axis0(self):
        g = Graph()
        a = g.input("a", f32[2, 3])
        b = g.input("b", f32[4, 3])
        g.output(g.concat(a, b, axis=0), "y")
        self.assert_backends_equal(g, {"a": self.rand(2, 3), "b": self.rand(4, 3)})

    def test_concat_middle_axis_rank3(self):
        """inner > 1 的中间维：最容易把偏移算错的位置。"""
        g = Graph()
        a = g.input("a", f32[2, 3, 4])
        b = g.input("b", f32[2, 5, 4])
        g.output(g.concat(a, b, axis=1), "y")
        self.assert_backends_equal(g, {"a": self.rand(2, 3, 4), "b": self.rand(2, 5, 4)})

    def test_concat_last_axis(self):
        g = Graph()
        a = g.input("a", f32[2, 3])
        b = g.input("b", f32[2, 4])
        g.output(g.concat(a, b, axis=-1), "y")
        self.assert_backends_equal(g, {"a": self.rand(2, 3), "b": self.rand(2, 4)})

    def _embedding(self, ids_dtype, ids_shape, table_shape):
        g = Graph()
        table = g.input("table", f32[table_shape])
        ids = g.input("ids", ids_dtype[ids_shape])
        g.output(g.embedding(table, ids), "y")
        feed = {
            "table": self.rand(*table_shape),
            "ids": self.rng.integers(0, table_shape[0], ids_shape).astype(ids_dtype.np_dtype),
        }
        return g, feed

    def test_embedding_duplicate_ids(self):
        g, feed = self._embedding(i32, (3, 5), (10, 4))
        got = self.assert_backends_equal(g, feed)
        self.assertEqual(got.shape, (3, 5, 4))

    def test_embedding_trailing_dims(self):
        g, feed = self._embedding(i32, (4, 5), (10, 2, 3))
        got = self.assert_backends_equal(g, feed)
        self.assertEqual(got.shape, (4, 5, 2, 3))

    def test_embedding_int8_ids(self):
        g, feed = self._embedding(i8, (3, 5), (100, 4))
        self.assert_backends_equal(g, feed)

    def test_embedding_i32_constant_ids(self):
        """i32 常量要能物化成设备张量：HelixLM 的分组尺度表就是常量索引。"""
        g = Graph()
        table = g.input("table", f32[10, 4])
        ids = g.constant(dtype=i32, shape=(5,), data=[0, 3, 2, 9, 1])
        g.output(g.embedding(table, ids), "y")
        got = self.assert_backends_equal(g, {"table": self.rand(10, 4)})
        self.assertEqual(got.shape, (5, 4))

    def test_embedding_out_of_range_rejected(self):
        """越界不能静默 clamp 成 0：必须显式报错。"""
        g = Graph()
        table = g.input("table", f32[10, 4])
        ids = g.input("ids", i32[5])
        g.output(g.embedding(table, ids), "y")
        feed = {"table": self.rand(10, 4), "ids": np.array([0, 1, 10, 2, 3], dtype=np.int32)}
        with self.assertRaises(BackendCapabilityError):
            self.cuda.execute(g, feed)

    def test_embedding_backward_repeated_ids(self):
        g = Graph()
        dy = g.input("dy", f32[3, 5, 4])
        ids = g.input("ids", i32[3, 5])
        g.output(g.embedding_backward(dy, ids, table_shape=[10, 4]), "y")
        feed = {
            "dy": self.rand(3, 5, 4),
            "ids": np.array([[0, 0, 1, 1, 1], [2, 2, 2, 3, 3], [0, 4, 4, 4, 9]], dtype=np.int32),
        }
        got = self.assert_backends_equal(g, feed)
        self.assertEqual(got.shape, (10, 4))

    def test_embedding_backward_accumulates_duplicates(self):
        """同一个下标出现两次时，结果必须是两行之和而不是其中一行。"""
        g = Graph()
        dy = g.input("dy", f32[2, 3])
        ids = g.input("ids", i32[2])
        g.output(g.embedding_backward(dy, ids, table_shape=[4, 3]), "y")
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        got = self.cuda.execute(g, {"dy": a, "ids": np.array([1, 1], dtype=np.int32)})["y"]
        np.testing.assert_allclose(got[1], a.sum(axis=0), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(got[0], 0.0, rtol=0, atol=0)


class TestCudaReductions(CudaTestCase):
    def test_softmax(self):
        g = Graph()
        x = g.input("x", f32[6, 11])
        g.output(g.softmax(x, axis=-1), "y")
        self.assert_backends_equal(g, {"x": self.rand(6, 11) * 2})

    def test_softmax_with_scale(self):
        """融合形态：softmax(x, scale) 与 softmax(x * scale) 必须等价。"""
        g = Graph()
        x = g.input("x", f32[6, 11])
        s = g.input("s", f32[1])
        g.output(g.softmax(x, s, axis=-1), "y")
        feed = {"x": self.rand(6, 11) * 2, "s": np.array([0.7], dtype=np.float32)}
        self.assert_backends_equal(g, feed)

    def test_softmax_non_last_axis(self):
        """非末维不再拒绝：同形状只换 axis，结果与参考一致。"""
        g = Graph()
        x = g.input("x", f32[6, 11])
        g.output(g.softmax(x, axis=0), "y")
        self.assert_backends_equal(g, {"x": self.rand(6, 11)})

    def test_layernorm(self):
        g = Graph()
        x = g.input("x", f32[5, 16])
        gamma = g.input("gamma", f32[16])
        beta = g.input("beta", f32[16])
        g.output(g.layernorm(x, gamma, beta, axes=(-1,), eps=1e-5), "y")
        feed = {
            "x": self.rand(5, 16),
            "gamma": self.rand(16),
            "beta": self.rand(16),
        }
        self.assert_backends_equal(g, feed)

    def test_reduce_sum_keepdims(self):
        g = Graph()
        x = g.input("x", f32[4, 7])
        g.output(g.reduce_sum(x, axis=-1, keepdims=True), "y")
        self.assert_backends_equal(g, {"x": self.rand(4, 7)})

    def test_reduce_sum_no_keepdims(self):
        g = Graph()
        x = g.input("x", f32[4, 7])
        g.output(g.reduce_sum(x, axis=-1, keepdims=False), "y")
        self.assert_backends_equal(g, {"x": self.rand(4, 7)})

    def test_reduce_max_keepdims(self):
        g = Graph()
        x = g.input("x", f32[4, 7])
        g.output(g.reduce_max(x, axis=-1, keepdims=True), "y")
        self.assert_backends_equal(g, {"x": self.rand(4, 7)})

    def test_reduce_max_no_keepdims(self):
        g = Graph()
        x = g.input("x", f32[4, 7])
        g.output(g.reduce_max(x, axis=-1, keepdims=False), "y")
        self.assert_backends_equal(g, {"x": self.rand(4, 7)})

    def test_reduce_mean_keepdims(self):
        g = Graph()
        x = g.input("x", f32[4, 7])
        g.output(g.reduce_mean(x, axis=-1, keepdims=True), "y")
        self.assert_backends_equal(g, {"x": self.rand(4, 7)})

    def test_reduce_mean_no_keepdims(self):
        g = Graph()
        x = g.input("x", f32[4, 7])
        g.output(g.reduce_mean(x, axis=-1, keepdims=False), "y")
        self.assert_backends_equal(g, {"x": self.rand(4, 7)})

    def test_argmax(self):
        g = Graph()
        x = g.input("x", f32[4, 7])
        g.output(g.argmax(x, axis=-1, keepdims=False), "y")
        feed = {"x": self.rand(4, 7)}
        expected = self.ref.execute(g, feed)["y"]
        got = self.cuda.execute(g, feed)["y"]
        self.assertEqual(got.dtype, np.int32)
        np.testing.assert_array_equal(got, expected)

    def test_argmax_keepdims(self):
        g = Graph()
        x = g.input("x", f32[4, 7])
        g.output(g.argmax(x, axis=-1, keepdims=True), "y")
        feed = {"x": self.rand(4, 7)}
        expected = self.ref.execute(g, feed)["y"]
        got = self.cuda.execute(g, feed)["y"]
        self.assertEqual(got.shape, (4, 1))
        np.testing.assert_array_equal(got, expected)

    def test_argmax_tie_returns_first_index(self):
        """并列最大值必须回到第一次出现的位置（numpy/PyTorch 约定）。"""
        x = np.zeros((2, 5), dtype=np.float32)
        x[0, 3] = 5.0
        x[1, 1] = 5.0
        x[1, 4] = 5.0
        g = Graph()
        v = g.input("x", f32[2, 5])
        g.output(g.argmax(v, axis=-1), "y")
        got = self.cuda.execute(g, {"x": x})["y"]
        np.testing.assert_array_equal(got, np.array([3, 1], dtype=np.int32))

    def test_argmax_non_last_axis(self):
        """任意单轴都支持：非末轴先置换到最后，下标语义仍是该轴上的位置。"""
        for shape, axis in (((4, 7), 0), ((3, 4, 5), 0), ((3, 4, 5), 1), ((2, 3, 4, 5), 2)):
            with self.subTest(shape=shape, axis=axis):
                g = Graph()
                x = g.input("x", f32[shape])
                g.output(g.argmax(x, axis=axis), "y")
                got = self.assert_backends_equal(g, {"x": self.rand(*shape)}, rtol=0, atol=0)
                self.assertEqual(got.dtype, np.int32)

    def test_argmax_non_last_axis_keepdims(self):
        g = Graph()
        x = g.input("x", f32[3, 4, 5])
        g.output(g.argmax(x, axis=1, keepdims=True), "y")
        got = self.assert_backends_equal(g, {"x": self.rand(3, 4, 5)}, rtol=0, atol=0)
        self.assertEqual(got.shape, (3, 1, 5))

    def test_reduce_multi_axis(self):
        """多轴归约：先置换再合并成一个轴，结果与 numpy 相同。"""
        for op in ("reduce_sum", "reduce_max", "reduce_mean"):
            for axes in ((0, 1), (1, 2), (0, 2), (-1, -2)):
                for keep in (False, True):
                    with self.subTest(op=op, axes=axes, keepdims=keep):
                        g = Graph()
                        x = g.input("x", f32[3, 4, 5])
                        g.output(
                            g.add_node(op, [x], {"axis": axes, "keepdims": keep}),
                            "y",
                        )
                        self.assert_backends_equal(g, {"x": self.rand(3, 4, 5)})

    def test_reduce_mean_multi_axis_float16(self):
        g = Graph()
        x = g.input("x", f16[4, 6, 5])
        g.output(
            g.add_node("reduce_mean", [x], {"axis": (0, 2), "keepdims": False}),
            "y",
        )
        got = self.assert_backends_equal(
            g, {"x": self.rand(4, 6, 5).astype(np.float16)}, rtol=2e-3, atol=2e-3
        )
        self.assertEqual(got.dtype, np.float16)


class TestCudaEvents(CudaTestCase):
    """稀疏事件的两趟内核：``spike_offsets`` 前缀和 + ``spike_csr`` 块内扫描。

    ``spike_csr`` 从 5.176 起一个 block 负责一行，行内按 ``blockDim.x`` 分段做
    块内前缀和。所以"列数不是 block 的整数倍"是最关键的边界：最后一段里有一半
    线程读到的是行外，必须短路成未命中，否则会多写事件。
    """

    def _graph(self, steps, width, capacity=None):
        g = Graph("events")
        x = g.input("x", f32[steps, width])
        offsets = g.add_node("spike_offsets", [x])
        cap = steps * width if capacity is None else capacity
        indices = g.add_node("spike_csr", [x, offsets], {"capacity": cap})
        g.output(offsets, "offsets")
        g.output(indices, "indices")
        return g

    def _spikes(self, steps, width, density=0.05):
        mask = self.rng.random((steps, width)) < density
        x = np.zeros((steps, width), dtype=np.float32)
        x[mask] = self.rng.standard_normal(int(mask.sum())).astype(np.float32)
        return x

    def _check(self, steps, width, density=0.05):
        graph = self._graph(steps, width)
        x = self._spikes(steps, width, density)
        expected = self.ref.execute(graph, {"x": x})
        got = self.cuda.execute(graph, {"x": x})
        np.testing.assert_array_equal(got["offsets"], expected["offsets"])
        np.testing.assert_array_equal(got["indices"], expected["indices"])
        return got

    def test_single_lane_row(self):
        """width=1：block 只有 1 个线程，扫描不进入折半循环。"""
        self._check(6, 1, density=0.5)

    def test_one_segment(self):
        """width=33、block=64：正好一段，一半线程落在行外。"""
        self._check(5, 33, density=0.3)

    def test_two_segments(self):
        """width=300、block=256：两段，第二段只有 44 个线程有活。"""
        got = self._check(8, 300, density=0.2)
        self.assertGreater(int(got["offsets"][-1]), 0, "对拍要有事件，不能全零")

    def test_sparse_long_rows(self):
        """width=1024、block=256：四段，段间靠 ``written`` 累加保序。"""
        self._check(3, 1024, density=0.02)

    def test_dense_row_keeps_column_order(self):
        """高发放率下每段都满命中：写入位置仍必须按列号递增。"""
        got = self._check(4, 512, density=0.5)
        per_row = got["offsets"]
        for t in range(per_row.size - 1):
            seg = got["indices"][per_row[t] : per_row[t + 1]]
            self.assertTrue(np.all(seg[1:] > seg[:-1]), f"第 {t} 行没有按列递增")

    def test_many_rows_stay_on_serial_path(self):
        """行数超过 SM×32 时留在串行铺法：这条分支也要有真机用例。

        阈值是 ``SM 数 × 32``，所以行数必须按当前卡算，不能写死 1024：
        M40（24 SM，阈值 768）上 1024 行已经在串行侧，V100（80 SM，阈值 2560）
        上同样的 1024 行仍是并行侧。
        """
        steps = max(1024, self.cuda.sm_count * 32 + 8)
        self.assertFalse(
            plan_csr_scan_enabled(steps, self.cuda.sm_count),
            f"用例前提是 {steps} 行判给串行铺法",
        )
        self._check(steps, 64, density=0.1)

    def test_capacity_overflow_is_reported(self):
        """容量不足：设备侧记 ``err``，主机侧抛错，绝不静默丢事件。"""
        graph = self._graph(4, 8, capacity=1)
        x = np.ones((4, 8), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.ref.execute(graph, {"x": x})
        with self.assertRaises(BackendCapabilityError):
            self.cuda.execute(graph, {"x": x})


class TestCudaScan(CudaTestCase):
    """线性递归（及其伴随）：元素并行、时间串行。

    形状上刻意覆盖四种情形：T=1（退化成 ``a0*h0+b0``）、三维尾（``(T,B,D)``，
    SSM 的真实形态）、尾维跨多个 block、以及非连续（Fortran 序）输入。
    递归是顺序的，因此"与参考逐点一致"就是唯一判据——参考实现也是同一个
    顺序循环，两边连累加顺序都相同。
    """

    def _forward(self, a_shape, h0_shape):
        g = Graph()
        a = g.input("a", f32[a_shape])
        b = g.input("b", f32[a_shape])
        h0 = g.input("h0", f32[h0_shape])
        g.output(g.add_node("scan_linear", [a, b, h0]), "y")
        feed = {
            "a": self.rand(*a_shape) * 0.3,
            "b": self.rand(*a_shape),
            "h0": self.rand(*h0_shape),
        }
        return g, feed

    def _backward(self, shape):
        g = Graph()
        a = g.input("a", f32[shape])
        dy = g.input("dy", f32[shape])
        g.output(g.add_node("scan_linear_backward", [a, dy]), "y")
        return g, {"a": self.rand(*shape) * 0.3, "dy": self.rand(*shape)}

    def test_2d(self):
        g, feed = self._forward((5, 7), (7,))
        self.assert_backends_equal(g, feed)

    def test_3d_tail(self):
        """(T, B, D)：尾维压平后与 (T, B*D) 等价，必须仍然正确。"""
        g, feed = self._forward((4, 2, 3), (2, 3))
        self.assert_backends_equal(g, feed)

    def test_single_step(self):
        """T=1 没有前驱累加，只走一次 h = a0*h0 + b0。"""
        g, feed = self._forward((1, 6), (6,))
        got = self.assert_backends_equal(g, feed)
        np.testing.assert_allclose(
            got[0], feed["a"][0] * feed["h0"] + feed["b"][0], rtol=1e-6, atol=1e-6
        )

    def test_long_sequence(self):
        """T=128：长序列下累加顺序不能漂。"""
        g, feed = self._forward((128, 5), (5,))
        self.assert_backends_equal(g, feed)

    def test_wide_tail_multiple_blocks(self):
        """尾维 2048 个元素：超过单 block 线程数，必须靠 grid-stride 覆盖。"""
        g, feed = self._forward((3, 2, 1024), (2, 1024))
        self.assert_backends_equal(g, feed)

    def test_non_contiguous_input(self):
        """Fortran 序输入：host 侧要先转连续，不能直接按线性下标读。"""
        g = Graph()
        a = g.input("a", f32[4, 3])
        b = g.input("b", f32[4, 3])
        h0 = g.input("h0", f32[3])
        g.output(g.add_node("scan_linear", [a, b, h0]), "y")
        feed = {
            "a": np.asfortranarray(self.rand(4, 3) * 0.3),
            "b": np.asfortranarray(self.rand(4, 3)),
            "h0": self.rand(3),
        }
        self.assert_backends_equal(g, feed)

    def test_backward(self):
        g, feed = self._backward((6, 4))
        self.assert_backends_equal(g, feed)

    def test_backward_3d_tail(self):
        g, feed = self._backward((5, 2, 3))
        self.assert_backends_equal(g, feed)

    def test_backward_single_step(self):
        """T=1 时没有后继，dh_0 就等于 dy_0。"""
        g, feed = self._backward((1, 7))
        got = self.assert_backends_equal(g, feed)
        np.testing.assert_allclose(got, feed["dy"], rtol=1e-6, atol=1e-6)

    def test_float16(self):
        """f16：同一套模板，累加在 float 里做，只有存储降精度。"""
        g = Graph()
        a = g.input("a", f16[4, 5])
        b = g.input("b", f16[4, 5])
        h0 = g.input("h0", f16[5])
        g.output(g.add_node("scan_linear", [a, b, h0]), "y")
        feed = {
            "a": (self.rand(4, 5) * 0.3).astype(np.float16),
            "b": self.rand(4, 5).astype(np.float16),
            "h0": self.rand(5).astype(np.float16),
        }
        expected = self.ref.execute(g, feed)["y"]
        got = self.cuda.execute(g, feed)["y"]
        self.assertEqual(got.dtype, np.float16)
        np.testing.assert_allclose(got, expected, rtol=2e-2, atol=2e-2)

        gb = Graph()
        a2 = gb.input("a", f16[4, 5])
        dy = gb.input("dy", f16[4, 5])
        gb.output(gb.add_node("scan_linear_backward", [a2, dy]), "y")
        feedb = {
            "a": (self.rand(4, 5) * 0.3).astype(np.float16),
            "dy": self.rand(4, 5).astype(np.float16),
        }
        np.testing.assert_allclose(
            self.cuda.execute(gb, feedb)["y"],
            self.ref.execute(gb, feedb)["y"],
            rtol=2e-2,
            atol=2e-2,
        )


@unittest.skipUnless(CUDA_AVAILABLE, "需要 cupy 与 CUDA 设备")
class TestCudaChunkedScan(CudaTestCase):
    """分块扫描：块内逐字相同、块间走复合，只承诺数值等价而非全程逐位。

    验收分三层：
    1. 显式 ``chunked_scan=0`` 必须仍然走串行内核（与"块长大于 T"的后端逐位相等）；
    2. 第一个块没有块前缀，它的输出必须与串行版**逐位**相同——这钉住"块内
       递推没有被我改坏"；
    3. 整段结果与串行版的相对误差在 1e-5 量级（f32），f16 放宽到 2e-2。
    4. 默认（``None``）按规模自适应：长 T + 小尾维等价于显式 64，短 T 或
       宽尾维逐位等于串行（5.155）。
    """

    CHUNK = 7

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.chunked = CudaBackend(chunked_scan=cls.CHUNK)
        cls.serial = CudaBackend(chunked_scan=0)
        # 块长大于任何用例的 T：分块分支不触发，等价于"关闭"
        cls.off = CudaBackend(chunked_scan=1_000_000)

    def test_rejects_invalid_options(self):
        for bad in (-1, True, 2.5):
            with self.assertRaises(ValueError, msg=repr(bad)):
                CudaBackend(chunked_scan=bad)

    def _forward(self, shape, h0_shape):
        g = Graph()
        a = g.input("a", f32[shape])
        b = g.input("b", f32[shape])
        h0 = g.input("h0", f32[h0_shape])
        g.output(g.add_node("scan_linear", [a, b, h0]), "y")
        feed = {
            "a": self.rand(*shape) * 0.3,
            "b": self.rand(*shape),
            "h0": self.rand(*h0_shape),
        }
        return g, feed

    def _backward(self, shape):
        g = Graph()
        a = g.input("a", f32[shape])
        dy = g.input("dy", f32[shape])
        g.output(g.add_node("scan_linear_backward", [a, dy]), "y")
        return g, {"a": self.rand(*shape) * 0.3, "dy": self.rand(*shape)}

    def test_default_path_is_serial(self):
        """显式 0（强制串行）与"块长 > T"的后端逐位相等：关闭档没有变化。"""
        g, feed = self._forward((64, 9), (9,))
        want = self.off.execute(g, feed)["y"]
        got = self.serial.execute(g, feed)["y"]
        np.testing.assert_array_equal(got, want)

    def test_auto_matches_explicit_chunk_on_long_small_tail(self):
        """T=1024 ≥ 256、尾维 8 ≤ 2048：自动档必须选中块长 64。

        与显式 64 **逐位**相等，才能证明自动档真的进了分块分支（第二趟
        内核一样、块长一样，舍入路径就一样）。数据取 ``a ≈ 0.9``（SSM 的
        真实取值）：``a ≈ 0.3`` 时初值十几步内就被衰减到 ulp 以下，串行与
        分块恰好逐位相同，那条断言就成了空转。
        """
        shape = (1024, 8)
        g = Graph()
        a = g.input("a", f32[shape])
        b = g.input("b", f32[shape])
        h0 = g.input("h0", f32[(8,)])
        g.output(g.add_node("scan_linear", [a, b, h0]), "y")
        feed = {
            "a": (0.9 + 0.05 * self.rand(*shape)).astype(np.float32),
            "b": self.rand(*shape).astype(np.float32),
            "h0": self.rand(8).astype(np.float32),
        }
        got = CudaBackend().execute(g, feed)["y"]
        explicit = CudaBackend(chunked_scan=64).execute(g, feed)["y"]
        np.testing.assert_array_equal(got, explicit)
        # 顺带确认它确实不是串行那条，否则上面的相等是空转
        self.assertFalse(
            np.array_equal(got, self.serial.execute(g, feed)["y"])
        )

    def test_auto_stays_serial_for_short_sequence(self):
        """T=256 低于下限 512：自动档逐位等于显式串行。

        T=256 是实测里分块开始变慢的那一档（elems=1024 时 +29%），钉住它
        等于钉住"下限不能再往回降到 256"。
        """
        g, feed = self._forward((256, 16), (16,))
        np.testing.assert_array_equal(
            CudaBackend().execute(g, feed)["y"],
            self.serial.execute(g, feed)["y"],
        )

    def test_auto_stays_serial_for_wide_tail(self):
        """尾维 4096 超过 2048：串行已被带宽喂饱，自动档不分块。"""
        g, feed = self._forward((512, 4096), (4096,))
        np.testing.assert_array_equal(
            CudaBackend().execute(g, feed)["y"],
            self.serial.execute(g, feed)["y"],
        )

    def test_auto_last_state_matches_explicit_chunk(self):
        """``scan_last_state`` 的甜点区更宽：尾维 4096 仍走块长 64。"""
        import cupy as cp

        shape, tail = (512, 4096), (4096,)
        g, feed = self._forward(shape, tail)
        want = CudaBackend(chunked_scan=64).scan_last_state(
            cp.asarray(feed["a"]), cp.asarray(feed["b"]), cp.asarray(feed["h0"])
        )
        got = CudaBackend().scan_last_state(
            cp.asarray(feed["a"]), cp.asarray(feed["b"]), cp.asarray(feed["h0"])
        )
        np.testing.assert_array_equal(cp.asnumpy(got), cp.asnumpy(want))

    def test_first_chunk_is_bitwise_equal(self):
        """第一块没有块前缀，块内递推必须与串行版逐位相同。"""
        steps = 64
        g, feed = self._forward((steps, 11), (11,))
        want = self.serial.execute(g, feed)["y"]
        got = self.chunked.execute(g, feed)["y"]
        np.testing.assert_array_equal(got[: self.CHUNK], want[: self.CHUNK])

    def test_forward_matches_serial(self):
        g, feed = self._forward((128, 16), (16,))
        want = self.serial.execute(g, feed)["y"]
        got = self.chunked.execute(g, feed)["y"]
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

    def test_forward_3d_tail(self):
        g, feed = self._forward((96, 2, 3), (2, 3))
        want = self.serial.execute(g, feed)["y"]
        got = self.chunked.execute(g, feed)["y"]
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

    def test_backward_matches_serial(self):
        g, feed = self._backward((128, 16))
        want = self.serial.execute(g, feed)["y"]
        got = self.chunked.execute(g, feed)["y"]
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

    def test_backward_3d_tail(self):
        g, feed = self._backward((96, 2, 3))
        want = self.serial.execute(g, feed)["y"]
        got = self.chunked.execute(g, feed)["y"]
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

    def test_float16(self):
        """f16 存储 + f32 scratch：分块与串行的差距仍在小量级。"""
        g = Graph()
        a = g.input("a", f16[64, 8])
        b = g.input("b", f16[64, 8])
        h0 = g.input("h0", f16[8])
        g.output(g.add_node("scan_linear", [a, b, h0]), "y")
        feed = {
            "a": (self.rand(64, 8) * 0.3).astype(np.float16),
            "b": self.rand(64, 8).astype(np.float16),
            "h0": self.rand(8).astype(np.float16),
        }
        got = self.chunked.execute(g, feed)["y"]
        self.assertEqual(got.dtype, np.float16)
        np.testing.assert_allclose(
            got, self.serial.execute(g, feed)["y"], rtol=2e-2, atol=2e-2
        )

    def test_last_state_matches_full_scan(self):
        """``scan_last_state`` 只写末状态，值必须等于整段扫描的最后一步。"""
        import cupy as cp

        shape, tail = (128, 3, 4), (3, 4)
        g, feed = self._forward(shape, tail)
        want = self.serial.execute(g, feed)["y"][-1]
        for backend in (self.serial, self.chunked):
            got = cp.asnumpy(
                backend.scan_last_state(
                    cp.asarray(feed["a"]),
                    cp.asarray(feed["b"]),
                    cp.asarray(feed["h0"]),
                )
            )
            self.assertEqual(got.shape, tail)
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


class TestCudaLayout(CudaTestCase):
    def test_reshape(self):
        g = Graph()
        x = g.input("x", f32[4, 6])
        g.output(g.reshape(x, shape=(2, 12)), "y")
        self.assert_backends_equal(g, {"x": self.rand(4, 6)})

    def test_transpose(self):
        g = Graph()
        x = g.input("x", f32[2, 3, 4])
        g.output(g.transpose(x, perm=(0, 2, 1)), "y")
        self.assert_backends_equal(g, {"x": self.rand(2, 3, 4)})

    def test_cast(self):
        g = Graph()
        x = g.input("x", f32[4, 5])
        g.output(g.cast(x, dtype="f32"), "y")
        self.assert_backends_equal(g, {"x": self.rand(4, 5)})


class TestCudaStrictness(CudaTestCase):
    def test_float64_is_rejected(self):
        g = Graph()
        x = g.input("x", f64[4, 5])
        g.output(g.relu(x), "y")
        with self.assertRaises(BackendCapabilityError):
            self.cuda.execute(g, {"x": self.rand(4, 5).astype(np.float64)})

    def test_float16_is_supported(self):
        """f16 已接入：同一条 relu 图必须能在 f16 上跑通并保持 f16 输出。"""
        g = Graph()
        x = g.input("x", f16[4, 5])
        g.output(g.relu(x), "y")
        feed = {"x": self.rand(4, 5).astype(np.float16)}
        got = self.cuda.execute(g, feed)["y"]
        self.assertEqual(got.dtype, np.float16)
        np.testing.assert_allclose(got, self.ref.execute(g, feed)["y"], rtol=2e-2, atol=2e-2)

    def test_zero_dim_input_keeps_rank_zero(self):
        """cupy 的 ascontiguousarray 会把 0 维变成 (1,)；标量种子输入必须保持 0 维。"""
        g = Graph()
        x = g.input("x", f32[4])
        d = g.input("dy", f32[()])
        g.output(g.mul(x, d), "y")
        got = self.cuda.execute(g, {"x": np.ones(4, np.float32), "dy": np.float32(2.0)})["y"]
        np.testing.assert_allclose(got, np.full(4, 2.0, np.float32))

    def test_unsupported_op_is_rejected(self):
        g = Graph()
        x = g.input("x", f32[4, 5])
        g.output(g.softmax(x, axis=-1), "y")
        caps = self.cuda.capabilities()
        self.assertIn("softmax", caps)
        self.assertNotIn("conv3d", caps)


class TestCudaFloat16(CudaTestCase):
    """f16 存储路径：语义与 f32 相同，差别只在存储宽度与舍入。

    内核一律把 f16 读成 float 再算、只在写回时降精度，所以这里比较的是
    "同一份语义 + 一次 f16 舍入"；参考后端也用 numpy 的 float16，容差按
    f16 的约三位十进制有效数字取 2e-2。
    """

    rtol = 2e-2
    atol = 2e-2

    def f16rand(self, *shape):
        return self.rand(*shape).astype(np.float16)

    def check(self, graph, feed, name="y", rtol=None, atol=None):
        expected = self.ref.execute(graph, feed)[name]
        got = self.cuda.execute(graph, feed)[name]
        self.assertEqual(got.dtype, np.float16, f"{name} 的输出必须是 f16")
        np.testing.assert_allclose(
            got,
            expected,
            rtol=self.rtol if rtol is None else rtol,
            atol=self.atol if atol is None else atol,
        )
        return got

    # -- 逐元素 ------------------------------------------------------

    def test_binary_and_broadcast(self):
        g = Graph()
        a = g.input("a", f16[3, 4])
        b = g.input("b", f16[4])
        g.output(g.mul(a, b), "y")
        self.check(g, {"a": self.f16rand(3, 4), "b": self.f16rand(4)})

    def test_scalar_binary(self):
        g = Graph()
        x = g.input("x", f16[4, 5])
        c = g.constant(2.0, dtype=f16)
        g.output(g.add(x, c), "y")
        self.check(g, {"x": self.f16rand(4, 5)})

    def test_unary_and_step(self):
        g = Graph()
        x = g.input("x", f16[6, 7])
        g.output(g.sigmoid(x), "y")
        self.check(g, {"x": self.f16rand(6, 7)})

        g2 = Graph()
        x2 = g2.input("x", f16[6, 7])
        g2.output(g2.add_node("step", [x2], {"threshold": 0.1}), "y")
        self.check(g2, {"x": self.f16rand(6, 7)})

    def test_constant_is_f16(self):
        g = Graph()
        c = g.constant(0.25, dtype=f16, shape=(2, 3))
        g.output(c, "y")
        got = self.cuda.execute(g, {})["y"]
        self.assertEqual(got.dtype, np.float16)
        np.testing.assert_allclose(got, np.full((2, 3), 0.25, dtype=np.float16))

    def test_cast_f16_to_f32(self):
        g = Graph()
        x = g.input("x", f16[4, 5])
        g.output(g.cast(x, dtype=f32), "y")
        got = self.cuda.execute(g, {"x": self.f16rand(4, 5)})["y"]
        self.assertEqual(got.dtype, np.float32)

    def test_dropout_mask_is_f16(self):
        g = Graph()
        x = g.input("x", f16[4, 9])
        g.output(g.dropout_mask(x, p=0.3, seed=99), "y")
        self.check(g, {"x": self.f16rand(4, 9)})

    def test_silu_gate_is_f16(self):
        g = Graph()
        x = g.input("x", f16[5, 6])
        gate = g.input("gate", f16[5, 6])
        g.output(g.silu(x, gate), "y")
        self.check(g, {"x": self.f16rand(5, 6), "gate": self.f16rand(5, 6)})

    def test_layernorm_residual_is_f16(self):
        g = Graph()
        a = g.input("a", f16[4, 6])
        r = g.input("r", f16[4, 6])
        gamma = g.input("gamma", f16[6])
        beta = g.input("beta", f16[6])
        g.output(g.layernorm(a, gamma, beta, r, axes=(-1,)), "y")
        self.check(
            g,
            {
                "a": self.f16rand(4, 6),
                "r": self.f16rand(4, 6),
                "gamma": self.f16rand(6),
                "beta": self.f16rand(6),
            },
        )

    # -- 矩阵 / 卷积 ------------------------------------------------

    def test_matmul(self):
        g = Graph()
        a = g.input("a", f16[4, 8])
        b = g.input("b", f16[8, 6])
        g.output(g.matmul(a, b), "y")
        self.check(g, {"a": self.f16rand(4, 8), "b": self.f16rand(8, 6)})

    def test_matmul_batched(self):
        g = Graph()
        a = g.input("a", f16[2, 4, 8])
        b = g.input("b", f16[2, 8, 6])
        g.output(g.matmul(a, b), "y")
        self.check(g, {"a": self.f16rand(2, 4, 8), "b": self.f16rand(2, 8, 6)})

    def test_conv2d(self):
        g = Graph()
        x = g.input("x", f16[2, 3, 8, 8])
        w = g.input("w", f16[4, 3, 3, 3])
        b = g.input("b", f16[4])
        g.output(g.conv2d(x, w, b, padding=(1, 1)), "y")
        self.check(
            g,
            {
                "x": self.f16rand(2, 3, 8, 8),
                "w": (self.f16rand(4, 3, 3, 3) * np.float16(0.3)),
                "b": self.f16rand(4),
            },
        )

    def test_conv2d_tiled_is_rejected_for_f16(self):
        """分块核是 f32 专用；强制 tiled 必须明确失败，而不是悄悄换实现。"""
        tiled = CudaBackend(conv_impl="tiled")
        g = Graph()
        x = g.input("x", f16[1, 3, 8, 8])
        w = g.input("w", f16[4, 3, 3, 3])
        g.output(g.conv2d(x, w, padding=(1, 1)), "y")
        feed = {"x": self.f16rand(1, 3, 8, 8), "w": self.f16rand(4, 3, 3, 3)}
        with self.assertRaises(BackendCapabilityError):
            tiled.execute(g, feed)

    def test_conv2d_grouped_and_depthwise(self):
        g = Graph()
        x = g.input("x", f16[2, 4, 8, 8])
        w = g.input("w", f16[6, 2, 3, 3])
        g.output(g.conv2d(x, w, padding=(1, 1), groups=2), "y")
        self.check(
            g,
            {
                "x": self.f16rand(2, 4, 8, 8),
                "w": self.f16rand(6, 2, 3, 3) * np.float16(0.3),
            },
        )

        g2 = Graph()
        x2 = g2.input("x", f16[1, 4, 7, 7])
        w2 = g2.input("w", f16[4, 1, 3, 3])
        g2.output(g2.conv2d(x2, w2, padding=(1, 1), groups=4), "y")
        self.check(
            g2, {"x": self.f16rand(1, 4, 7, 7), "w": self.f16rand(4, 1, 3, 3)}
        )

    def test_adaptive_avg_pool_forward_and_backward(self):
        g = Graph()
        x = g.input("x", f16[1, 3, 7, 5])
        g.output(g.adaptive_avg_pool2d(x, output_size=(2, 3)), "y")
        self.check(g, {"x": self.f16rand(1, 3, 7, 5)})

        g2 = Graph()
        dy = g2.input("dy", f16[1, 3, 2, 3])
        g2.output(
            g2.adaptive_avg_pool2d_backward(
                dy, input_shape=(1, 3, 7, 5), output_size=(2, 3)
            ),
            "y",
        )
        self.check(g2, {"dy": self.f16rand(1, 3, 2, 3)})

    def test_pool_forward_and_backward(self):
        g = Graph()
        x = g.input("x", f16[2, 3, 6, 6])
        g.output(g.maxpool2d(x, kernel=(2, 2), stride=(2, 2)), "y")
        self.check(g, {"x": self.f16rand(2, 3, 6, 6)})

        g2 = Graph()
        x2 = g2.input("x", f16[2, 3, 6, 6])
        g2.output(g2.avgpool2d(x2, kernel=(2, 2), stride=(2, 2)), "y")
        self.check(g2, {"x": self.f16rand(2, 3, 6, 6)})

        g3 = Graph()
        dy = g3.input("dy", f16[1, 2, 3, 3])
        g3.output(
            g3.avgpool2d_backward(dy, input_shape=(1, 2, 6, 6), kernel=(2, 2)),
            "y",
        )
        self.check(g3, {"dy": self.f16rand(1, 2, 3, 3)})

    # -- 归约 --------------------------------------------------------

    def test_softmax(self):
        g = Graph()
        x = g.input("x", f16[6, 11])
        g.output(g.softmax(x, axis=-1), "y")
        self.check(g, {"x": (self.f16rand(6, 11) * np.float16(2.0))})

    def test_layernorm(self):
        g = Graph()
        x = g.input("x", f16[5, 16])
        gamma = g.input("gamma", f16[16])
        beta = g.input("beta", f16[16])
        g.output(g.layernorm(x, gamma, beta, axes=(-1,)), "y")
        self.check(
            g,
            {"x": self.f16rand(5, 16), "gamma": self.f16rand(16), "beta": self.f16rand(16)},
        )

    def test_rmsnorm(self):
        g = Graph()
        x = g.input("x", f16[5, 16])
        gamma = g.input("gamma", f16[16])
        g.output(g.rmsnorm(x, gamma, axes=(-1,)), "y")
        self.check(g, {"x": self.f16rand(5, 16), "gamma": self.f16rand(16) + np.float16(1.0)})

    def test_batch_norm_and_stats(self):
        g = Graph()
        x = g.input("x", f16[3, 4, 5])
        gamma = g.input("gamma", f16[4])
        beta = g.input("beta", f16[4])
        g.output(g.batch_norm(x, gamma, beta, axis=1), "y")
        self.check(
            g,
            {"x": self.f16rand(3, 4, 5), "gamma": self.f16rand(4), "beta": self.f16rand(4)},
        )

        g2 = Graph()
        x2 = g2.input("x", f16[3, 4, 5])
        g2.output(g2.batch_norm_stats(x2, axis=1), "y")
        self.check(g2, {"x": self.f16rand(3, 4, 5)})

    def test_reduce_sum(self):
        g = Graph()
        x = g.input("x", f16[4, 7])
        g.output(g.reduce_sum(x, axis=-1, keepdims=True), "y")
        self.check(g, {"x": self.f16rand(4, 7)})

    def test_reduce_max_and_mean(self):
        g = Graph()
        x = g.input("x", f16[4, 7])
        g.output(g.reduce_max(x, axis=-1), "y")
        self.check(g, {"x": self.f16rand(4, 7)})

        g2 = Graph()
        x2 = g2.input("x", f16[4, 7])
        g2.output(g2.reduce_mean(x2, axis=-1, keepdims=True), "y")
        self.check(g2, {"x": self.f16rand(4, 7)})

    def test_argmax(self):
        g = Graph()
        x = g.input("x", f16[4, 7])
        g.output(g.argmax(x, axis=-1), "y")
        feed = {"x": self.f16rand(4, 7)}
        expected = self.ref.execute(g, feed)["y"]
        got = self.cuda.execute(g, feed)["y"]
        self.assertEqual(got.dtype, np.int32)
        np.testing.assert_array_equal(got, expected)

    # -- 布局 / 索引 -------------------------------------------------

    def test_layout_ops(self):
        g = Graph()
        x = g.input("x", f16[2, 3, 4])
        g.output(g.transpose(x, perm=(0, 2, 1)), "y")
        self.check(g, {"x": self.f16rand(2, 3, 4)})

        g2 = Graph()
        x2 = g2.input("x", f16[2, 12])
        g2.output(g2.reshape(x2, shape=(2, 3, 4)), "y")
        self.check(g2, {"x": self.f16rand(2, 12)})

        g3 = Graph()
        x3 = g3.input("x", f16[4, 6])
        g3.output(g3.slice(x3, dim=1, start=1, stop=5), "y")
        self.check(g3, {"x": self.f16rand(4, 6)})

    # -- 注意力 ------------------------------------------------------

    def test_attention(self):
        """融合 attention 的 f16 是混合精度：q/k/v/mask/out 存 f16，
        点积、在线 softmax 的行统计与输出累加器都在 float 里。

        参考实现是逐位 f16 运算（每一步都舍入），内核只在写回时舍入一次，
        所以容差比逐元素算子宽：差异来自"舍入位置不同"，不是实现错误。
        """
        g = Graph()
        q = g.input("q", f16[2, 3, 4, 6])
        k = g.input("k", f16[2, 3, 4, 6])
        v = g.input("v", f16[2, 3, 4, 5])
        mask = g.input("mask", f16[1, 1, 4, 4])
        g.output(g.attention(q, k, v, mask, scale=0.25), "y")
        feed = {
            "q": self.f16rand(2, 3, 4, 6) * np.float16(0.3),
            "k": self.f16rand(2, 3, 4, 6) * np.float16(0.3),
            "v": self.f16rand(2, 3, 4, 5),
            "mask": np.linspace(-0.5, 0.5, 16, dtype=np.float16).reshape(1, 1, 4, 4),
        }
        self.check(g, feed, rtol=5e-2, atol=5e-2)

    def test_attention_dropout_is_f16(self):
        g = Graph()
        q = g.input("q", f16[2, 3, 5])
        k = g.input("k", f16[2, 3, 5])
        v = g.input("v", f16[2, 3, 4])
        g.output(g.attention(q, k, v, scale=0.5, dropout=0.3, seed=7), "y")
        feed = {
            "q": self.f16rand(2, 3, 5) * np.float16(0.3),
            "k": self.f16rand(2, 3, 5) * np.float16(0.3),
            "v": self.f16rand(2, 3, 4),
        }
        self.check(g, feed, rtol=5e-2, atol=5e-2)

    def test_concat_and_pad(self):
        g = Graph()
        a = g.input("a", f16[2, 3])
        b = g.input("b", f16[2, 4])
        g.output(g.concat(a, b, axis=1), "y")
        self.check(g, {"a": self.f16rand(2, 3), "b": self.f16rand(2, 4)})

        g2 = Graph()
        x = g2.input("x", f16[2, 3])
        g2.output(g2.pad(x, dim=1, before=1, after=2), "y")
        self.check(g2, {"x": self.f16rand(2, 3)})

    def test_broadcast_to(self):
        g = Graph()
        x = g.input("x", f16[1, 3])
        g.output(g.broadcast_to(x, shape=[4, 3]), "y")
        self.check(g, {"x": self.f16rand(1, 3)})

    def test_embedding(self):
        g = Graph()
        table = g.input("table", f16[10, 4])
        ids = g.input("ids", i32[3, 5])
        g.output(g.embedding(table, ids), "y")
        self.check(
            g,
            {
                "table": self.f16rand(10, 4),
                "ids": self.rng.integers(0, 10, (3, 5)).astype(np.int32),
            },
        )

    def test_embedding_backward(self):
        g = Graph()
        dy = g.input("dy", f16[2, 3])
        ids = g.input("ids", i32[2])
        g.output(g.embedding_backward(dy, ids, table_shape=[4, 3]), "y")
        self.check(
            g,
            {"dy": self.f16rand(2, 3), "ids": np.array([1, 1], dtype=np.int32)},
        )


class TestCudaFusionForms(CudaTestCase):
    """融合产出的"扩展输入"形态必须在 CUDA 上真的能被承接。

    IR 层融合（passes/fusion.py）只保证语义合法；能不能执行由后端能力决定。
    这一组把三种形态都在真机上跑一遍，避免"融合完没人能执行"。
    """

    def test_layernorm_with_residual(self):
        g = Graph()
        a = g.input("a", f32[4, 6])
        r = g.input("r", f32[4, 6])
        gamma = g.input("gamma", f32[6])
        beta = g.input("beta", f32[6])
        g.output(g.layernorm(a, gamma, beta, r, axes=(-1,)), "y")
        feed = {
            "a": self.rand(4, 6),
            "r": self.rand(4, 6),
            "gamma": self.rand(6),
            "beta": self.rand(6),
        }
        self.assert_backends_equal(g, feed, rtol=1e-5, atol=1e-5)

    def test_fused_pipeline_graph_executes(self):
        """融合后的图（softmax 缩放、layernorm 残差、SwiGLU）端到端对齐。"""
        from aetherflow.passes import fused_pipeline

        g = Graph()
        a = g.input("a", f32[3, 8])
        r = g.input("r", f32[3, 8])
        gate = g.input("gate", f32[3, 8])
        gamma = g.input("gamma", f32[8])
        beta = g.input("beta", f32[8])
        s = g.layernorm(g.add(a, r), gamma, beta, axes=(-1,))
        g.output(g.mul(s, g.silu(gate)), "y")

        fused = fused_pipeline().run(g)
        self.assertEqual(fused.op_names().count("add"), 0)
        self.assertEqual(fused.op_names().count("mul"), 0)

        feed = {
            "a": self.rand(3, 8),
            "r": self.rand(3, 8),
            "gate": self.rand(3, 8),
            "gamma": self.rand(8),
            "beta": self.rand(8),
        }
        self.assert_backends_equal(g, feed, rtol=1e-5, atol=1e-5)
        self.assert_backends_equal(fused, feed, rtol=1e-5, atol=1e-5)

    def test_folded_constant_executes(self):
        """折叠产物是任意常量张量（``data`` 属性），CUDA 必须能直接物化。"""
        from aetherflow.passes import default_pipeline

        g = Graph()
        a = g.input("a", f32[2, 3])
        prod = g.mul(
            g.constant(data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], shape=(2, 3)),
            g.constant(3.0, shape=()),
        )
        g.output(g.add(a, prod), "y")

        folded = default_pipeline().run(g)
        constants = [n for n in folded.nodes if n.op.name == "constant"]
        self.assertEqual(len(constants), 1, "未使用的原始常量应由 DCE 清掉")
        self.assertIn("data", constants[0].attrs)
        self.assertNotIn("value", constants[0].attrs)

        feed = {"a": self.rand(2, 3)}
        self.assert_backends_equal(g, feed, rtol=1e-6, atol=1e-6)
        self.assert_backends_equal(folded, feed, rtol=1e-6, atol=1e-6)

    def test_layernorm_rejects_non_last_axis(self):
        g = Graph()
        x = g.input("x", f32[4, 6])
        gamma = g.input("gamma", f32[4])
        beta = g.input("beta", f32[4])
        g.output(g.layernorm(x, gamma, beta, axes=(0,)), "y")
        with self.assertRaises(BackendCapabilityError):
            self.cuda.execute(
                g,
                {
                    "x": self.rand(4, 6),
                    "gamma": self.rand(4),
                    "beta": self.rand(4),
                },
            )

    def test_layernorm_residual_shape_is_checked_at_build_time(self):
        """residual 与 x 不同形时在构图期就失败，而不是进内核再越界。"""
        g = Graph()
        a = g.input("a", f32[4, 6])
        r = g.input("r", f32[4, 5])
        gamma = g.input("gamma", f32[6])
        beta = g.input("beta", f32[6])
        with self.assertRaises(TypeError):
            g.layernorm(a, gamma, beta, r, axes=(-1,))


class TestCodegenDtypes(unittest.TestCase):
    """模板渲染的静态约束，不需要 GPU。

    NVRTC 只接受 ASCII 源码，所以哪怕 Python 注释是中文，raw string 里也不
    能出现非 ASCII 字符——这条只能靠测试守住。
    """

    TEMPLATES = (
        "STEP_SRC", "SOFTMAX_SRC", "ATTENTION_SRC", "LAYERNORM_SRC",
        "LAYERNORM_RESIDUAL_SRC", "SILU_GATE_SRC",
        "BATCH_NORM_SRC", "REDUCE_SUM_SRC", "IM2COL_SRC", "MAXPOOL2D_SRC",
        "CONV2D_TILED_SRC", "RMSNORM_SRC", "AVGPOOL2D_SRC",
        "AVGPOOL2D_BACKWARD_SRC", "CONCAT_COPY_SRC", "MAXPOOL2D_BACKWARD_SRC",
        "CONV2D_INPUT_GRAD_SRC", "CONV2D_WEIGHT_GRAD_SRC", "DROPOUT_MASK_SRC",
    )

    def test_rendered_sources_are_ascii(self):
        from aetherflow.backends.cuda import codegen

        for name in self.TEMPLATES:
            for dtype in ("float32", "float16"):
                src = codegen.render(getattr(codegen, name), dtype)
                self.assertTrue(src.isascii(), f"{name}/{dtype} 含非 ASCII 字符")

    def test_render_expands_type_placeholders(self):
        from aetherflow.backends.cuda import codegen

        for dtype, storage in (("float32", "float"), ("float16", "unsigned short")):
            src = codegen.render(codegen.STEP_SRC, dtype)
            self.assertIn(f"const {storage}* __restrict__ in", src)
            for leftover in ("%T%", "%LD%", "%ST%", "%ATOMIC%", "%PRELUDE%"):
                self.assertNotIn(leftover, src)


class TestCudaTransformerBlock(CudaTestCase):
    spec = BlockSpec(d_model=32, num_heads=4, d_ff=64)

    def _feed(self):
        feed = {
            "x": self.rng.standard_normal((2, 5, self.spec.d_model)).astype(np.float32)
        }
        for name, shape in self.spec.param_shapes.items():
            if name == "scale":
                feed[name] = np.array(
                    [1.0 / np.sqrt(self.spec.head_dim)], dtype=np.float32
                )
            else:
                feed[name] = (self.rng.standard_normal(shape) * 0.1).astype(np.float32)
        return feed

    def test_matches_reference(self):
        g = Graph("block")
        inputs, out = build_transformer_block(g, self.spec)
        g.output(out, "y")
        self.assert_backends_equal(g, self._feed(), rtol=1e-4, atol=1e-4)

    def test_fused_pipeline_matches_reference(self):
        """融合后的图（softmax 吸收缩放）在 GPU 上与参考语义一致。"""
        from aetherflow.passes import fused_pipeline

        g = Graph("block")
        _, out = build_transformer_block(g, self.spec)
        g.output(out, "y")
        fused = fused_pipeline().run(g)
        self.assertLess(len(fused.nodes), len(g.nodes))
        self.assert_backends_equal(fused, self._feed(), rtol=1e-4, atol=1e-4)


class TestCudaAttention(CudaTestCase):
    """融合 attention 前向：分块 online softmax 与参考语义对拍。

    重点不在"能跑"，而在三条最容易写错的边界：

    1. T 跨过 blockDim 时，后一块的 max 变大必须把先前累积的归一化因子
       与输出一起按 ``corr`` 拉回来；
    2. mask 的广播（``(T,)`` / ``(S, T)``）在 Python 侧完成，内核只认
       ``(BH, S, T)``，几种写法的结果必须一致；
    3. dropout 掩码按 ``(seed, 扁平索引)`` 生成，参考与 CUDA 必须落在
       同一个索引上。
    """

    def _build(
        self,
        q_shape,
        k_shape,
        v_shape,
        *,
        mask_shape=None,
        mask_value=0.0,
        dropout=0.0,
        seed=0,
        scale=None,
        dtype=f32,
        np_dtype=np.float32,
    ):
        g = Graph("attention")
        q = g.input("q", dtype[q_shape])
        k = g.input("k", dtype[k_shape])
        v = g.input("v", dtype[v_shape])
        args = [q, k, v]
        feed = {
            "q": self.rand(*q_shape).astype(np_dtype),
            "k": self.rand(*k_shape).astype(np_dtype),
            "v": self.rand(*v_shape).astype(np_dtype),
        }
        if mask_shape is not None:
            args.append(g.input("mask", dtype[mask_shape]))
            feed["mask"] = np.full(mask_shape, mask_value, dtype=np_dtype)
        attrs = {"dropout": dropout, "seed": seed}
        if scale is not None:
            attrs["scale"] = scale
        g.output(g.attention(*args, **attrs), "y")
        return g, feed

    def test_basic(self):
        g, feed = self._build(
            (2, 2, 3, 2), (2, 2, 3, 2), (2, 2, 3, 2), scale=0.5
        )
        self.assert_backends_equal(g, feed, rtol=1e-5, atol=1e-5)

    def test_cross_attention(self):
        # S != T 且 E != D：三套索引（q 行、k 行、v 列）各自独立
        g, feed = self._build(
            (2, 3, 2, 3), (2, 3, 4, 3), (2, 3, 4, 5), scale=0.5
        )
        self.assert_backends_equal(g, feed, rtol=1e-5, atol=1e-5)

    def test_chunked_keys(self):
        # T=300 跨过 256 的块边界：没有 corr 重缩放时这一条会明显偏
        g, feed = self._build(
            (2, 300, 4), (2, 300, 4), (2, 300, 4), scale=0.5
        )
        self.assert_backends_equal(g, feed, rtol=1e-5, atol=1e-5)

    def test_two_dimensional_input(self):
        # 无批维：BH=1，grid 的第二维退化成 1
        g, feed = self._build((3, 2), (4, 2), (4, 2), scale=0.5)
        self.assert_backends_equal(g, feed, rtol=1e-5, atol=1e-5)

    def test_float16_mixed_precision(self):
        """f16 只省存储：q/k/v 存半精度，行统计与累加仍在 float（5.152）。

        取 T=300 跨过块边界：真按 f16 累加误差会随 T 线性放大，混合精度则
        只在写回时舍入一次，与参考后端的差距停在 f16 的 ulp 量级（1e-2）。
        """
        g, feed = self._build(
            (2, 300, 4), (2, 300, 4), (2, 300, 4), scale=0.5,
            dtype=f16, np_dtype=np.float16,
        )
        self.assert_backends_equal(g, feed, rtol=2e-2, atol=2e-2)

    def test_mask_broadcast_forms_agree(self):
        # (1, T)、(S, T) 都应广播成 (BH, S, T)；结果与整张掩码逐点一致
        shapes = [(1, 4), (3, 4)]
        outs = []
        state = self.rng.bit_generator.state
        for mask_shape in shapes:
            # 两次构图必须用同一组 q/k/v，否则比的不是广播形态而是随机数
            self.rng.bit_generator.state = state
            g, feed = self._build(
                (2, 3, 2), (2, 4, 2), (2, 4, 2),
                mask_shape=mask_shape, mask_value=-0.75, scale=0.5,
            )
            outs.append(self.assert_backends_equal(g, feed, rtol=1e-5, atol=1e-5))
        np.testing.assert_allclose(outs[0], outs[1], rtol=1e-6, atol=1e-6)

    def test_causal_mask(self):
        from aetherflow.models import causal_mask_bias

        g = Graph("attention")
        q = g.input("q", f32[2, 4, 3])
        k = g.input("k", f32[2, 4, 3])
        v = g.input("v", f32[2, 4, 3])
        mask = g.input("mask", f32[1, 1, 4, 4])
        g.output(g.attention(q, k, v, mask, scale=0.5), "y")
        feed = {
            "q": self.rand(2, 4, 3),
            "k": self.rand(2, 4, 3),
            "v": self.rand(2, 4, 3),
            "mask": causal_mask_bias(4),
        }
        self.assert_backends_equal(g, feed, rtol=1e-5, atol=1e-5)

    def test_mask_with_multiple_leading_singletons(self):
        """``(1,1,1,S,T)`` 直接可用：降维不得依赖整数索引。

        天数 ``DeviceArray`` 只支持切片、拒绝降维索引，``mask[0]`` 会抛
        ``TypeError``；这里用两层前导 1 覆盖"连续去掉多个前导维"的路径。
        """
        from aetherflow.models import causal_mask_bias

        g = Graph("attention")
        q = g.input("q", f32[2, 4, 3])
        k = g.input("k", f32[2, 4, 3])
        v = g.input("v", f32[2, 4, 3])
        mask = g.input("mask", f32[1, 1, 1, 4, 4])
        g.output(g.attention(q, k, v, mask, scale=0.5), "y")
        feed = {
            "q": self.rand(2, 4, 3),
            "k": self.rand(2, 4, 3),
            "v": self.rand(2, 4, 3),
            "mask": causal_mask_bias(4).reshape(1, 1, 1, 4, 4),
        }
        self.assert_backends_equal(g, feed, rtol=1e-5, atol=1e-5)

    def test_dropout_matches_reference(self):
        g, feed = self._build(
            (2, 2, 3, 2), (2, 2, 3, 2), (2, 2, 3, 2),
            dropout=0.3, seed=7, scale=0.5,
        )
        # 掩码位级一致，差异只来自浮点累加顺序
        self.assert_backends_equal(g, feed, rtol=1e-5, atol=1e-5)

    def test_fused_transformer_block(self):
        """``fused_attention=True`` 的 Block 在 GPU 上与参考语义一致。

        融合必须是实现替换：整块（含残差、LayerNorm、FFN）一起对拍，
        才能证明 attention 的输出布局与反向依赖的形状没有漂移。
        """
        spec = BlockSpec(
            d_model=32, num_heads=4, d_ff=64, fused_attention=True, dropout=0.2,
            dropout_seed=11,
        )
        g = Graph("block_fused")
        _, out = build_transformer_block(g, spec)
        g.output(out, "y")

        rng = np.random.default_rng(5)
        feed = {"x": (rng.standard_normal((2, 5, spec.d_model)) * 0.2).astype(np.float32)}
        for name, shape in spec.param_shapes.items():
            feed[name] = (rng.standard_normal(shape) * 0.1).astype(np.float32)
        self.assert_backends_equal(g, feed, rtol=1e-4, atol=1e-4)


class TestDeviceResidency(CudaTestCase):
    """M3-1：参数常驻显存，H2D 只发生在激活值上。

    共享的 backend 实例在类级创建，这里每个用例新建一个，
    避免驻留表在用例之间互相污染。
    """

    def setUp(self):
        self.cuda = CudaBackend()
        self.ref = ReferenceBackend()

    def _graph(self):
        g = Graph()
        w = g.input("w", f32[4])
        x = g.input("x", f32[3, 4])
        g.output(g.add(x, w), "y")
        return g

    def test_feed_only_activations(self):
        g = self._graph()
        w, x = self.rand(4), self.rand(3, 4)
        self.cuda.resident_parameters(g, {"w": w})
        got = self.cuda.execute(g, {"x": x})["y"]
        np.testing.assert_allclose(
            got, self.ref.execute(g, {"x": x, "w": w})["y"], rtol=1e-5, atol=1e-5
        )

    def test_reused_across_calls(self):
        g = self._graph()
        w = self.rand(4)
        self.cuda.resident_parameters(g, {"w": w})
        for _ in range(3):
            x = self.rand(3, 4)
            got = self.cuda.execute(g, {"x": x})["y"]
            np.testing.assert_allclose(
                got, self.ref.execute(g, {"x": x, "w": w})["y"], rtol=1e-5, atol=1e-5
            )

    def test_explicit_value_overrides_resident(self):
        """显式传入优先，避免"以为换了权重其实没用上"的隐蔽问题。"""
        g = self._graph()
        self.cuda.resident_parameters(g, {"w": self.rand(4)})
        w_new, x = self.rand(4), self.rand(3, 4)
        got = self.cuda.execute(g, {"x": x, "w": w_new})["y"]
        np.testing.assert_allclose(
            got, self.ref.execute(g, {"x": x, "w": w_new})["y"], rtol=1e-5, atol=1e-5
        )

    def test_unknown_name_rejected(self):
        g = self._graph()
        with self.assertRaises(KeyError):
            self.cuda.resident_parameters(g, {"not_an_input": self.rand(4)})

    def test_wrong_shape_fails_at_residency_time(self):
        """错形状的权重应当在驻留时就失败，而不是等到第一次推理。"""
        from aetherflow.runtime.executor import ShapeMismatchError

        g = self._graph()
        with self.assertRaises(ShapeMismatchError):
            self.cuda.resident_parameters(g, {"w": self.rand(5)})

    def test_release_restores_normal_path(self):
        g = self._graph()
        self.cuda.resident_parameters(g, {"w": self.rand(4)})
        self.cuda.release_parameters()
        self.assertEqual(dict(self.cuda.resident), {})
        with self.assertRaises(KeyError):
            self.cuda.execute(g, {"x": self.rand(3, 4)})

    def test_block_with_all_params_resident(self):
        """Transformer Block 的全部参数驻留，只喂激活值。"""
        spec = BlockSpec(d_model=32, num_heads=4, d_ff=64)
        g = Graph("block")
        _, out = build_transformer_block(g, spec)
        g.output(out, "y")

        params = {}
        for name, shape in spec.param_shapes.items():
            if name == "scale":
                params[name] = np.array(
                    [1.0 / np.sqrt(spec.head_dim)], dtype=np.float32
                )
            else:
                params[name] = (self.rng.standard_normal(shape) * 0.1).astype(np.float32)
        x = self.rng.standard_normal((2, 5, spec.d_model)).astype(np.float32)

        self.cuda.resident_parameters(g, params)
        got = self.cuda.execute(g, {"x": x})["y"]
        expected = self.ref.execute(g, {"x": x, **params})["y"]
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)


class TestCudaGraph(CudaTestCase):
    """M3：运行期计划（CUDA Graph）。

    要证明三件事：整图（含 cuBLAS 的 matmul）能被捕获；replay 结果与参考
    语义一致；重复 replay 用的是**本次传入的新输入**，而不是捕获时的旧缓冲。
    """

    def setUp(self):
        self.cuda = CudaBackend()
        self.ref = ReferenceBackend()

    def _simple(self):
        g = Graph()
        a = g.input("a", f32[4, 8])
        b = g.input("b", f32[8, 6])
        g.output(g.matmul(a, b), "y")
        return g

    def test_simple_matmul(self):
        g = self._simple()
        feed = {"a": self.rand(4, 8), "b": self.rand(8, 6)}
        plan = self.cuda.capture(g, feed)
        got = plan.run(feed)["y"]
        np.testing.assert_allclose(
            got, self.ref.execute(g, feed)["y"], rtol=1e-4, atol=1e-4
        )

    def test_conv_capture(self):
        """卷积（im2col + 批量 cuBLAS）也必须能被整图捕获并 replay。

        cuBLAS 在捕获期有已知限制（模块文档里写了为什么自己绑 ctypes），
        卷积是第一个同时用到自研 kernel 与批量 GEMM 的算子，值得单独钉住。
        """
        g = Graph()
        x = g.input("x", f32[1, 3, 8, 8])
        w = g.input("w", f32[4, 3, 3, 3])
        b = g.input("b", f32[4])
        g.output(g.conv2d(x, w, b, padding=(1, 1)), "y")

        feed = {
            "x": self.rand(1, 3, 8, 8),
            "w": self.rand(4, 3, 3, 3) * 0.3,
            "b": self.rand(4),
        }
        plan = self.cuda.capture(g, feed)
        got = plan.run(feed)["y"]
        np.testing.assert_allclose(
            got, self.ref.execute(g, feed)["y"], rtol=1e-4, atol=1e-4
        )

    def test_capture_with_constant_uses_capture_round_copy(self):
        """捕获图里的常量必须指向捕获轮的副本（5.172）。

        HIP 的捕获轮在预热后会 ``Arena.recycle()``，把预热期地址整体收回；
        常量若复用预热副本，图里读到的是后来写进去的中间值（设备态训练实测
        loss 从 4.37 变成 10^2 量级）。这里把"常量 + 整图捕获 + 多次 replay"
        钉在真机上，CUDA 侧也走同一条"捕获轮重新上传"的路径。
        """
        g = Graph()
        c = g.constant(2.5, shape=(4, 4))
        x = g.input("x", f32[4, 4])
        g.output(g.mul(g.add(x, c), c), "y")

        feed = {"x": self.rand(4, 4)}
        plan = self.cuda.capture(g, feed)
        expected = self.ref.execute(g, feed)["y"]
        for _ in range(3):
            np.testing.assert_allclose(
                plan.run(feed)["y"], expected, rtol=1e-5, atol=1e-6
            )

    def test_capture_with_i32_constant(self):
        """i32 索引表常量也必须走计划期物化（捕获流禁止 H2D）。

        HelixLM 的"按通道分组选尺度"用 i32 常量喂 embedding。常量若等到
        捕获轮才第一次上传，驱动会拒掉捕获流上的 H2D
        （"H2D transfers are disallowed"），整图退回逐节点执行。
        """
        g = Graph()
        table = g.constant(
            dtype=f32, shape=(10, 2),
            data=(np.arange(20, dtype=np.float32).reshape(10, 2) * 0.1).tolist(),
        )
        ids = g.constant(dtype=i32, shape=(5,), data=[0, 3, 2, 9, 1])
        x = g.input("x", f32[5, 2])
        g.output(g.add(g.embedding(table, ids), x), "y")

        feed = {"x": self.rand(5, 2)}
        plan = self.cuda.capture(g, feed)
        expected = self.ref.execute(g, feed)["y"]
        for _ in range(3):
            np.testing.assert_allclose(plan.run(feed)["y"], expected, rtol=1e-5, atol=1e-6)

    def test_cnn_capture(self):
        """conv -> relu -> maxpool：一个最小 CNN 尖峰，整图捕获并 replay。"""
        g = Graph()
        x = g.input("x", f32[1, 3, 8, 8])
        w = g.input("w", f32[4, 3, 3, 3])
        b = g.input("b", f32[4])
        h = g.relu(g.conv2d(x, w, b, padding=(1, 1)))
        g.output(g.maxpool2d(h, kernel=2, stride=2), "y")

        feed = {
            "x": self.rand(1, 3, 8, 8),
            "w": self.rand(4, 3, 3, 3) * 0.3,
            "b": self.rand(4),
        }
        plan = self.cuda.capture(g, feed)
        np.testing.assert_allclose(
            plan.run(feed)["y"], self.ref.execute(g, feed)["y"], rtol=1e-4, atol=1e-4
        )

    def test_replay_uses_new_inputs(self):
        g = self._simple()
        first = {"a": self.rand(4, 8), "b": self.rand(8, 6)}
        plan = self.cuda.capture(g, first)
        plan.run(first)
        second = {"a": self.rand(4, 8), "b": self.rand(8, 6)}
        got = plan.run(second)["y"]
        np.testing.assert_allclose(
            got, self.ref.execute(g, second)["y"], rtol=1e-4, atol=1e-4
        )
        # 同时确认结果确实变了，排除"其实一直在用捕获时缓冲"的假通过
        old = self.ref.execute(g, first)["y"]
        self.assertGreater(float(np.abs(got - old).max()), 1e-3)

    def test_replay_rejects_shape_change(self):
        g = self._simple()
        plan = self.cuda.capture(g, {"a": self.rand(4, 8), "b": self.rand(8, 6)})
        with self.assertRaises(ValueError):
            plan.replay({"a": self.rand(5, 8), "b": self.rand(8, 6)})

    def test_replay_reports_out_of_range_ids(self):
        """捕获后的 replay 也必须报越界，而不是静默 clamp。

        捕获期不能 D2H，所以标志只能登记；replay 同步之后再复查。报错前
        清零标志，同一个计划的后续合法 replay 不能受牵连。
        """
        g = Graph()
        table = g.input("table", f32[10, 4])
        ids = g.input("ids", i32[5])
        g.output(g.embedding(table, ids), "y")

        good = {
            "table": self.rand(10, 4),
            "ids": np.array([0, 1, 2, 3, 4], dtype=np.int32),
        }
        plan = self.cuda.capture(g, good)

        bad = dict(good, ids=np.array([0, 1, 10, 2, 3], dtype=np.int32))
        with self.assertRaises(BackendCapabilityError):
            plan.run(bad)

        # 标志已清零：同一个计划继续可用，合法输入不再误报
        np.testing.assert_allclose(
            plan.run(good)["y"], self.ref.execute(g, good)["y"], rtol=1e-5, atol=1e-5
        )

    def test_replay_advances_dropout_key(self):
        """捕获之后 dropout 仍能重新采样：key 必须在设备内存里。

        常量 seed 是内核标量参数，CUDA Graph 会按值冻结它，replay 只能得到
        同一张掩码。运行期 key 走设备指针，每次 replay 前 H2D 更新。
        """
        g = Graph()
        x = g.input("x", f32[16, 8])
        key = g.input("key", i32[1])
        g.output(g.dropout_mask(x, key, p=0.5), "y")

        zeros = np.zeros((16, 8), dtype=np.float32)
        first = {"x": zeros, "key": np.array([1], np.int32)}
        plan = self.cuda.capture(g, first)
        a = plan.run(first)["y"]

        second = {"x": zeros, "key": np.array([2], np.int32)}
        b = plan.run(second)["y"]

        self.assertFalse(np.array_equal(a, b))
        np.testing.assert_array_equal(a, self.ref.execute(g, first)["y"])
        np.testing.assert_array_equal(b, self.ref.execute(g, second)["y"])
        np.testing.assert_array_equal(plan.run(first)["y"], a)

    def test_repeated_replay_stable(self):
        """同一输入重复 replay 结果逐次一致（排除显存池复用导致的踩踏）。"""
        g = self._simple()
        feed = {"a": self.rand(4, 8), "b": self.rand(8, 6)}
        plan = self.cuda.capture(g, feed)
        first = plan.run(feed)["y"]
        for _ in range(5):
            np.testing.assert_array_equal(plan.run(feed)["y"], first)

    def test_arena_matches_pool_path(self):
        """复用内存与 cupy 池两条捕获路径必须给出完全相同的结果。"""
        g = self._simple()
        feed = {"a": self.rand(4, 8), "b": self.rand(8, 6)}
        reused = self.cuda.capture(g, feed, reuse_memory=True)
        pooled = self.cuda.capture(g, feed, reuse_memory=False)
        np.testing.assert_array_equal(reused.run(feed)["y"], pooled.run(feed)["y"])

    def test_two_plans_coexist(self):
        """两张图的 arena 各自独立：交替 replay 不能互相踩内存。"""
        g = self._simple()
        f1 = {"a": self.rand(4, 8), "b": self.rand(8, 6)}
        f2 = {"a": self.rand(4, 8), "b": self.rand(8, 6)}
        p1 = self.cuda.capture(g, f1)
        p2 = self.cuda.capture(g, f2)
        for _ in range(3):
            np.testing.assert_allclose(
                p1.run(f1)["y"], self.ref.execute(g, f1)["y"], rtol=1e-4, atol=1e-4
            )
            np.testing.assert_allclose(
                p2.run(f2)["y"], self.ref.execute(g, f2)["y"], rtol=1e-4, atol=1e-4
            )

    def test_arena_reuses_freed_block(self):
        """释放的块必须被下一次同尺寸分配复用，而不是再向驱动要一块。"""
        from aetherflow.backends.cuda.memory import Arena

        arena = Arena(self.cuda.cp, chunk_bytes=8 << 20)
        a = arena.malloc(1 << 20)
        arena.malloc(1 << 20)  # 让 a 后面还有活块，排除"尾部回退"这种假复用
        before = arena.total_bytes
        arena.free(a)
        c = arena.malloc(1 << 20)
        self.assertEqual(arena.total_bytes, before)
        self.assertEqual(c.ptr, a.ptr)
        del a, c
        arena.close()

    def test_arena_merges_adjacent_blocks(self):
        """相邻块先后释放后必须合并，才能承接一个更大的请求。"""
        from aetherflow.backends.cuda.memory import Arena

        arena = Arena(self.cuda.cp, chunk_bytes=8 << 20)
        a = arena.malloc(1 << 20)
        b = arena.malloc(1 << 20)
        arena.free(a)
        arena.free(b)
        big = arena.malloc(2 << 20)
        self.assertEqual(big.ptr, a.ptr)
        self.assertEqual(arena.total_bytes, 8 << 20)
        del a, b, big
        arena.close()

    def test_arena_does_not_merge_across_chunks(self):
        """地址相邻的两个 chunk 不能合并，否则会切出跨分配的指针。

        两个 chunk 是两次独立的 ``cudaMalloc``；合并后 best-fit 发出的
        ``MemoryPointer`` 会越过第一个分配的末尾，D2D 拷贝按分配校验范围
        就会失败。这里只走账本（``free`` 不碰驱动），因此不依赖真实 malloc
        是否恰好给出相邻地址。
        """
        from types import SimpleNamespace

        from aetherflow.backends.cuda.memory import Arena

        arena = Arena(self.cuda.cp, chunk_bytes=1 << 20)
        arena._chunks = [
            SimpleNamespace(ptr=0x10000, size=1 << 20),
            SimpleNamespace(ptr=0x110000, size=1 << 20),
        ]
        for addr in (0x10000, 0x110000):
            arena._live[addr] = 1 << 20
            arena._live_bytes += 1 << 20
            arena.free(SimpleNamespace(ptr=addr))
        self.assertEqual(len(arena._free), 2, "跨 chunk 的相邻空闲块不能合并")
        for addr, size in arena._free:
            self.assertTrue(
                any(
                    chunk.ptr <= addr and addr + size <= chunk.ptr + chunk.size
                    for chunk in arena._chunks
                ),
                "空闲块必须完整落在某个 chunk 内",
            )

    def test_capture_reports_memory_budget(self):
        """捕获后必须能报出峰值与总占用：峰值是下界，总占用只多不少。"""
        spec = BlockSpec(d_model=32, num_heads=4, d_ff=64)
        g = Graph("block")
        _, out = build_transformer_block(g, spec)
        g.output(out, "y")

        params = {}
        for name, shape in spec.param_shapes.items():
            if name == "scale":
                params[name] = np.array(
                    [1.0 / np.sqrt(spec.head_dim)], dtype=np.float32
                )
            else:
                params[name] = (self.rng.standard_normal(shape) * 0.1).astype(np.float32)
        x = self.rng.standard_normal((2, 5, spec.d_model)).astype(np.float32)

        self.cuda.resident_parameters(g, params)
        plan = self.cuda.capture(g, {"x": x})

        # 复用确实发生：峰值严格小于若不回收所需的累计请求量
        self.assertLess(plan.peak_bytes, plan._allocator.requested_bytes)
        self.assertGreaterEqual(plan.memory_bytes, plan.peak_bytes)
        np.testing.assert_allclose(
            plan.run({"x": x})["y"],
            self.ref.execute(g, {"x": x, **params})["y"],
            rtol=1e-4,
            atol=1e-4,
        )

    def test_static_budget_matches_captured_peak(self):
        """编译期静态预算（``plan_memory``）与捕获期峰值必须同量级。

        只对照中间缓冲：Arena 不参与图输入与驻留参数的分配。两者不要求严格
        相等——arena 的 512B 对齐与后端小临时让实测偏大，连续张量上的
        reshape 走视图又让它偏小。超出 2 倍就是模型失真。
        """
        from aetherflow.ir.liveness import plan_memory

        spec = BlockSpec(d_model=32, num_heads=4, d_ff=64)
        g = Graph("block")
        _, out = build_transformer_block(g, spec)
        g.output(out, "y")

        budget = plan_memory(g, symbols={"B": 2, "S": 5})
        self.assertGreater(budget.intermediate_peak_bytes, 0)
        # 静态预算在图还没执行时就能报出来，这是它相对 Arena 的全部意义
        self.assertGreater(budget.reuse_ratio, 1.0)

        params = {}
        for name, shape in spec.param_shapes.items():
            if name == "scale":
                params[name] = np.array(
                    [1.0 / np.sqrt(spec.head_dim)], dtype=np.float32
                )
            else:
                params[name] = (self.rng.standard_normal(shape) * 0.1).astype(np.float32)
        x = self.rng.standard_normal((2, 5, spec.d_model)).astype(np.float32)

        self.cuda.resident_parameters(g, params)
        plan = self.cuda.capture(g, {"x": x})

        static = budget.aligned_intermediate_peak_bytes
        actual = plan.peak_bytes
        self.assertLessEqual(actual, static * 2, "实测峰值不应显著超过静态预算")
        self.assertGreaterEqual(actual * 2, static, "预算不应比实测高一个数量级")
        np.testing.assert_allclose(
            plan.run({"x": x})["y"],
            self.ref.execute(g, {"x": x, **params})["y"],
            rtol=1e-4,
            atol=1e-4,
        )

    def test_block_with_resident_params(self):
        spec = BlockSpec(d_model=32, num_heads=4, d_ff=64)
        g = Graph("block")
        _, out = build_transformer_block(g, spec)
        g.output(out, "y")

        params = {}
        for name, shape in spec.param_shapes.items():
            if name == "scale":
                params[name] = np.array(
                    [1.0 / np.sqrt(spec.head_dim)], dtype=np.float32
                )
            else:
                params[name] = (self.rng.standard_normal(shape) * 0.1).astype(np.float32)
        x = self.rng.standard_normal((2, 5, spec.d_model)).astype(np.float32)

        self.cuda.resident_parameters(g, params)
        plan = self.cuda.capture(g, {"x": x})
        self.assertEqual(plan.mutable_inputs, ("x",))

        got = plan.run({"x": x})["y"]
        np.testing.assert_allclose(
            got, self.ref.execute(g, {"x": x, **params})["y"], rtol=1e-4, atol=1e-4
        )

        x2 = self.rng.standard_normal((2, 5, spec.d_model)).astype(np.float32)
        got2 = plan.run({"x": x2})["y"]
        np.testing.assert_allclose(
            got2, self.ref.execute(g, {"x": x2, **params})["y"], rtol=1e-4, atol=1e-4
        )


@unittest.skipUnless(CUDA_AVAILABLE, "需要 cupy 与 CUDA 设备")
class TestAutoGraph(CudaTestCase):
    """5.79：``execute`` 的自动图执行路径。

    默认仍是逐节点；这里显式打开，钉住四件事：结果与逐节点逐位一致、
    每种输入签名各有一份计划、计划数封顶并按最近使用淘汰、捕获失败回落。
    """

    def _symbolic(self):
        g = Graph("sym")
        a = g.input("a", f32["B", 8])
        b = g.input("b", f32[8, 6])
        g.output(g.relu(g.matmul(a, b)), "y")
        return g

    def test_matches_node_by_node(self):
        g = Graph("auto")
        a = g.input("a", f32[4, 8])
        b = g.input("b", f32[8, 6])
        g.output(g.relu(g.matmul(a, b)), "y")

        backend = CudaBackend(graph_mode="auto")
        for _ in range(3):
            feed = {"a": self.rand(4, 8), "b": self.rand(8, 6)}
            np.testing.assert_array_equal(
                backend.execute(g, feed)["y"], self.cuda.execute(g, feed)["y"]
            )
        self.assertEqual(len(backend._graph_plans), 1)

    def test_execute_device_reuses_captured_plan(self):
        """设备侧口径也要吃捕获：否则量到的是逐节点 Python 开销（5.251）。

        捕获计划返回的是计划持有的输出缓冲，下一次 replay 会覆盖它——所以
        这里每次立刻拷回主机再比对，不能跨轮攒着看。
        """
        import cupy as cp

        g = Graph("auto_dev")
        a = g.input("a", f32[4, 8])
        b = g.input("b", f32[8, 6])
        g.output(g.relu(g.matmul(a, b)), "y")

        backend = CudaBackend(graph_mode="auto")
        feed = {"a": self.rand(4, 8), "b": self.rand(8, 6)}
        for _ in range(3):
            got = cp.asnumpy(backend.execute_device(g, feed)["y"])
            np.testing.assert_array_equal(got, backend.execute(g, feed)["y"])
        self.assertEqual(len(backend._graph_plans), 1)

    def test_shape_switch_captures_per_signature(self):
        g = self._symbolic()
        backend = CudaBackend(graph_mode="auto")
        w = self.rand(8, 6)
        for batch in (1, 3, 1):
            feed = {"a": self.rand(batch, 8), "b": w}
            np.testing.assert_array_equal(
                backend.execute(g, feed)["y"], self.cuda.execute(g, feed)["y"]
            )
        self.assertEqual(len(backend._graph_plans), 2)

    def test_plan_cache_is_bounded_and_lru(self):
        g = self._symbolic()
        backend = CudaBackend(graph_mode="auto", max_graph_plans=2)
        captured = []
        real = backend.capture

        def counting(graph, sample_inputs, reuse_memory=True):
            captured.append(tuple(sample_inputs["a"].shape))
            return real(graph, sample_inputs, reuse_memory=reuse_memory)

        backend.capture = counting
        w = self.rand(8, 6)
        for batch in (1, 2, 3):
            backend.execute(g, {"a": self.rand(batch, 8), "b": w})
        self.assertEqual(captured, [(1, 8), (2, 8), (3, 8)])
        self.assertEqual(len(backend._graph_plans), 2)

        # batch=1 是最久未用的那份，已被淘汰：再跑它必须重新捕获
        backend.execute(g, {"a": self.rand(1, 8), "b": w})
        self.assertEqual(captured[-1], (1, 8))
        self.assertEqual(len(backend._graph_plans), 2)

    def test_capture_failure_falls_back(self):
        g = self._symbolic()
        backend = CudaBackend(graph_mode="auto")

        def broken(*args, **kwargs):
            raise RuntimeError("捕获期不允许主机同步")

        backend.capture = broken
        feed = {"a": self.rand(2, 8), "b": self.rand(8, 6)}
        np.testing.assert_array_equal(
            backend.execute(g, feed)["y"], self.cuda.execute(g, feed)["y"]
        )
        # 失败原因必须可查询，而不是只留一个"已关闭"的布尔
        self.assertIn("主机同步", backend.graph_capture_failure(g) or "")
        # 回落之后不再反复尝试捕获同一张图
        backend.execute(g, feed)
        self.assertEqual(len(backend._graph_plans), 0)

    def test_capture_failure_is_isolated_per_graph(self):
        """一张图捕不了，不该把其他图的自动图执行一起关掉。"""
        bad = self._symbolic()
        good = self._symbolic()
        backend = CudaBackend(graph_mode="auto")
        real = backend.capture
        attempts = []

        def selective(graph, sample_inputs, *args, **kwargs):
            attempts.append(id(graph))
            if graph is bad:
                raise RuntimeError("图里有主机同步")
            return real(graph, sample_inputs, *args, **kwargs)

        backend.capture = selective
        feed = {"a": self.rand(2, 8), "b": self.rand(8, 6)}
        backend.execute(bad, feed)
        self.assertIn("主机同步", backend.graph_capture_failure(bad) or "")
        self.assertIsNone(backend.graph_capture_failure(good))

        backend.execute(good, feed)
        self.assertEqual(len(backend._graph_plans), 1)
        self.assertEqual(attempts.count(id(good)), 1)

        # 失败过的那张图仍不再重试；另一张图不受牵连
        backend.execute(bad, feed)
        self.assertEqual(attempts.count(id(bad)), 1)

    def test_tuning_change_retries_capture(self):
        """换配置是新前提：旧的捕获失败记录不能继续压住这张图。"""
        g = self._symbolic()
        backend = CudaBackend(graph_mode="auto")
        real = backend.capture
        attempts = []

        def flaky(graph, sample_inputs, *args, **kwargs):
            attempts.append(id(graph))
            if len(attempts) == 1:
                raise RuntimeError("第一次捕获失败")
            return real(graph, sample_inputs, *args, **kwargs)

        backend.capture = flaky
        feed = {"a": self.rand(2, 8), "b": self.rand(8, 6)}
        backend.execute(g, feed)
        self.assertIsNotNone(backend.graph_capture_failure(g))

        epoch = backend._config_epoch
        backend.set_tuning("add", {"block": 256})
        self.assertGreater(backend._config_epoch, epoch)
        self.assertIsNone(backend.graph_capture_failure(g))

        backend.execute(g, feed)
        self.assertEqual(len(backend._graph_plans), 1)

    def test_resident_weights_are_fixed_in_plan(self):
        g = Graph("resident")
        x = g.input("x", f32[2, 8])
        w = g.input("w", f32[8, 6])
        g.output(g.relu(g.matmul(x, w)), "y")

        backend = CudaBackend(graph_mode="auto")
        first = self.rand(8, 6)
        backend.resident_parameters(g, {"w": first})
        feed = {"x": self.rand(2, 8)}
        np.testing.assert_array_equal(
            backend.execute(g, feed)["y"],
            self.cuda.execute(g, {"w": first, **feed})["y"],
        )
        (_, plan), = backend._graph_plans.values()
        self.assertEqual(plan.mutable_inputs, ("x",))

        # 换了权重，旧计划写死的是旧缓冲：必须失效并重新捕获
        second = self.rand(8, 6)
        backend.resident_parameters(g, {"w": second})
        self.assertEqual(len(backend._graph_plans), 0)
        np.testing.assert_array_equal(
            backend.execute(g, feed)["y"],
            self.cuda.execute(g, {"w": second, **feed})["y"],
        )

    def test_single_node_graph_skips_capture(self):
        """单算子图不捕获：一次 launch 不值得 replay 的簿记（5.81）。

        5.79 实测里唯一"变慢"的几个用例都是单算子图（``matmul_batched`` +8%），
        多出来的就是捕获计划固定的 cuBLAS 算法与 replay 开销。
        """
        g = Graph("single")
        a = g.input("a", f32[2, 8])
        b = g.input("b", f32[8, 6])
        g.output(g.matmul(a, b), "y")

        backend = CudaBackend(graph_mode="auto")
        feed = {"a": self.rand(2, 8), "b": self.rand(8, 6)}
        np.testing.assert_array_equal(
            backend.execute(g, feed)["y"], self.cuda.execute(g, feed)["y"]
        )
        self.assertEqual(len(backend._graph_plans), 0)

        # 阈值可调：显式降到 1 时恢复"单算子也捕获"
        eager = CudaBackend(graph_mode="auto", min_graph_nodes=1)
        eager.execute(g, feed)
        self.assertEqual(len(eager._graph_plans), 1)

    def test_views_are_not_freed_early(self):
        """reshape / transpose 的输出是视图，捕获期不能当独立分配回收。

        反向图里到处都是这类视图（转置权重、按最后一维归约前的 reshape）。
        arena 只按地址认块，把视图还回去就等于把原数组的内存提前复用掉，
        replay 出来的数值会悄悄变错——不报错，只是算得不对。
        """
        g = Graph("views")
        a = g.input("a", f32[4, 6])
        b = g.input("b", f32[6, 3])
        h = g.matmul(a, b)
        h = g.reshape(h, shape=(3, 4))
        h = g.transpose(h, perm=(1, 0))
        g.output(g.reduce_sum(g.mul(h, h), axis=1, keepdims=False), "y")

        feed = {"a": self.rand(4, 6), "b": self.rand(6, 3)}
        expected = CudaBackend(graph_mode="off").execute(g, feed)["y"]

        auto = CudaBackend(graph_mode="auto")
        auto.execute(g, feed)  # 第一遍捕获
        got = auto.execute(g, feed)["y"]  # 第二遍 replay
        self.assertEqual(len(auto._graph_plans), 1)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_reshape_source_is_not_reused_before_view_dies(self):
        """视图的源数组先死、地址随后被下一块分配顶掉——这才是真正算错的情形。

        ``reshape`` 之后源数组的引用计数归零，执行器立刻把它还给 arena；紧接
        着同尺寸的算子就会拿到同一段地址。视图这时还没被读，读到的已经是别人
        写进去的数据。``Arena.pin`` 在创建视图时钉住源地址，这条链才安全。
        """
        g = Graph("reshape_alias")
        x = g.input("x", f32[4, 6])
        y = g.input("y", f32[6, 4])
        w = g.input("w", f32[4, 4])
        t = g.mul(x, x)  # 中间值：只有 reshape 读它，读完就该被回收
        h = g.reshape(t, shape=(6, 4))  # 视图，与 t 同地址
        k = g.matmul(y, w)  # 同尺寸分配，会顶掉 t 的地址
        g.output(g.add(h, k), "out")

        feed = {"x": self.rand(4, 6), "y": self.rand(6, 4), "w": self.rand(4, 4)}
        expected = CudaBackend(graph_mode="off").execute(g, feed)["out"]

        auto = CudaBackend(graph_mode="auto")
        auto.execute(g, feed)  # 第一遍捕获
        got = auto.execute(g, feed)["out"]  # 第二遍 replay
        self.assertEqual(len(auto._graph_plans), 1)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_slice_source_is_not_reused_before_view_dies(self):
        """``slice`` 同样是零拷贝视图，而且它的 ``ptr`` 可能带偏移。

        执行器回收的是**源数组**，视图这时还在环境里等着被读；不记账就会
        被下一块同尺寸分配顶掉。与 ``reshape`` 那条的区别在于 ``start>0``
        时视图地址 ≠ 源地址，扣账必须按源地址做，否则记账永远归不了零。
        """
        g = Graph("slice_alias")
        x = g.input("x", f32[4, 6])
        y = g.input("y", f32[4, 6])
        t = g.mul(x, x)  # 中间值：只被 slice 读，读完就该回收
        s = g.add_node("slice", [t], {"dim": 0, "start": 2, "stop": 4})
        k = g.mul(y, y)  # 同尺寸分配，会顶掉 t 的地址
        u = g.add_node("slice", [k], {"dim": 0, "start": 2, "stop": 4})
        g.output(g.add(s, u), "out")

        feed = {"x": self.rand(4, 6), "y": self.rand(4, 6)}
        expected = CudaBackend(graph_mode="off").execute(g, feed)["out"]

        auto = CudaBackend(graph_mode="auto")
        auto.execute(g, feed)  # 第一遍捕获
        got = auto.execute(g, feed)["out"]  # 第二遍 replay
        self.assertEqual(len(auto._graph_plans), 1)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_off_never_captures(self):
        g = self._symbolic()
        backend = CudaBackend()
        self.assertEqual(backend.graph_mode, "off")
        backend.execute(g, {"a": self.rand(2, 8), "b": self.rand(8, 6)})
        self.assertEqual(len(backend._graph_plans), 0)

    def test_rejects_invalid_options(self):
        with self.assertRaises(ValueError):
            CudaBackend(graph_mode="always")
        with self.assertRaises(ValueError):
            CudaBackend(max_graph_plans=0)
        with self.assertRaises(ValueError):
            CudaBackend(min_graph_nodes=0)


class TestBatchDispatch(CudaTestCase):
    """批量派发在真机上的等价性：计划期绑定内核，结果必须逐位不变。

    快路径省的是"每节点一次的注册表查找 + dtype 检查"，一旦预判与运行期
    事实不符（例如某个算子其实不接受 f16、某个下标声明不是整数），就必须
    老实地退回原路径并保持原来的报错，而不是把错误吞掉或提前抛出。
    """

    @staticmethod
    def _graph():
        g = Graph()
        a = g.input("a", f32[4, 8])
        b = g.input("b", f32[8])
        g.output(g.mul(g.relu(g.add(a, b)), a), "y")
        return g

    def test_plan_binds_every_group(self):
        backend = CudaBackend()
        g = self._graph()
        backend.execute(g, {"a": self.rand(4, 8), "b": self.rand(8)})
        plan = backend._plan(g)
        self.assertEqual(sum(len(group.nodes) for group in plan), 3)
        self.assertTrue(
            all(
                group.kernels is not None
                and len(group.kernels) == len(group.nodes)
                for group in plan
            )
        )

    def test_planned_path_is_bitwise_identical(self):
        feed = {"a": self.rand(4, 8), "b": self.rand(8)}
        g = self._graph()
        backend = CudaBackend()
        fast = backend.execute(g, feed)["y"]
        # 把计划退回原路径，再跑一遍：两份结果必须逐位相同
        for group in backend._plan(g):
            group.kernels = None
        slow = backend.execute(g, feed)["y"]
        np.testing.assert_array_equal(fast, slow)

    def test_unsupported_dtype_falls_back_and_still_raises(self):
        # f64 不在设备后端的支持面内：不能预绑定，报错时机与文案保持原样
        g = Graph()
        a = g.input("a", f64[4])
        g.output(g.relu(a), "y")
        backend = CudaBackend()
        with self.assertRaises(BackendCapabilityError):
            backend.execute(g, {"a": np.ones(4, dtype=np.float64)})
        self.assertIsNone(backend._plan(g)[0].kernels)

    def test_index_dtype_gate(self):
        # i32 下标可以预绑定；换成 f32 下标则必须留在原路径上（运行期报同样的错）
        g = Graph()
        ids = g.input("ids", i32[3])
        table = g.input("table", f32[6, 4])
        g.output(g.embedding(table, ids), "y")
        backend = CudaBackend()
        backend.execute(
            g, {"ids": np.array([0, 1, 2], np.int32), "table": self.rand(6, 4)}
        )
        self.assertIsNotNone(backend._plan(g)[0].kernels)


class TestCudaAutodiff(CudaTestCase):
    """反向图也是普通 IR：凡是前向能跑的图，反向图在 CUDA 上也必须与参考一致。

    这里刻意只覆盖"反向图用到的算子都已在 CUDA 实现"的模型（MLP / LayerNorm）。
    卷积与池化的反向算子还没有 CUDA 内核，它们必须**显式拒绝**而不是退回参考实现。
    """

    def test_new_elementwise_match_reference(self):
        for op, positive in (
            ("exp", False),
            ("log", True),
            ("sqrt", True),
            ("neg", False),
            ("tanh", False),
            ("sigmoid", False),
        ):
            with self.subTest(op=op):
                g = Graph(op)
                x = g.input("x", f32[5, 7])
                g.output(g.add_node(op, [x]), "y")
                data = self.rand(5, 7)
                if positive:
                    data = np.abs(data) + 0.5
                self.assert_backends_equal(g, {"x": data})

    def test_step_constant_broadcast(self):
        g = Graph("step_constant")
        x = g.input("x", f32[4, 6])
        c = g.constant(0.5)
        y = g.add_node("step", [g.add_node("mul", [x, c])], {"threshold": 0.1})
        g.output(y, "y")
        self.assert_backends_equal(g, {"x": self.rand(4, 6)})

        g2 = Graph("broadcast_to")
        v = g2.input("v", f32[1, 3])
        g2.output(g2.add_node("broadcast_to", [v], {"shape": [4, 3]}), "y")
        self.assert_backends_equal(g2, {"v": self.rand(1, 3)})

    def _mlp(self):
        g = Graph("mlp")
        x = g.input("x", f32[4, 3])
        w1 = g.input("w1", f32[3, 5])
        b1 = g.input("b1", f32[5])
        w2 = g.input("w2", f32[5, 2])
        h = g.add_node("relu", [g.add_node("add", [g.add_node("matmul", [x, w1]), b1])])
        g.output(g.add_node("matmul", [h, w2]), "y")
        feed = {
            "x": self.rand(4, 3),
            "w1": self.rand(3, 5) * 0.3,
            "b1": self.rand(5) * 0.1,
            "w2": self.rand(5, 2) * 0.3,
        }
        return g, feed

    def _assert_graphs_equal(self, bwd, feed, rtol=1e-4, atol=1e-4):
        expected = self.ref.execute(bwd, feed)
        got = self.cuda.execute(bwd, feed)
        for name in expected:
            np.testing.assert_allclose(
                got[name], expected[name], rtol=rtol, atol=atol, err_msg=name
            )

    def test_mlp_backward_matches_reference(self):
        g, feed = self._mlp()
        bwd = backward(g)
        self._assert_graphs_equal(bwd, {**feed, "dy": np.ones((4, 2), dtype=np.float32)})

    def test_eager_frontend_matches_reference(self):
        """eager 前端只是语法糖：它产出的图在 CUDA 上必须与参考一致。

        重点在 0 维标量常量——``x * 0.5 + 1.0`` 会把常量插成 ``f32[]``，
        这是最窄、也最容易被广播路径处理错的一种输入。
        """
        from aetherflow.frontend import EagerGraph

        g = EagerGraph("eager_mlp")
        x = g.input("x", f32[4, 3])
        w = g.input("w", f32[3, 5])
        b = g.input("b", f32[5])
        g.output(g.relu(x @ w + b) * 0.5 + 1.0, "y")

        feed = {
            "x": self.rand(4, 3),
            "w": self.rand(3, 5) * 0.3,
            "b": self.rand(5) * 0.1,
        }
        self.assert_backends_equal(g, feed)

        bwd = backward(g)
        self._assert_graphs_equal(bwd, {**feed, "dy": np.ones((4, 5), dtype=np.float32)})

    def test_spike_forward_and_backward(self):
        """spike 前向复用 step kernel；反向是 IR 层的替代梯度，后端无感知。"""
        g = Graph("spike")
        x = g.input("x", f32[5, 7])
        g.output(g.add_node("spike", [x], {"threshold": 0.5, "alpha": 1.0}), "y")
        feed = {"x": self.rand(5, 7) * 1.5}
        self.assert_backends_equal(g, feed)

        bwd = backward(g)
        self._assert_graphs_equal(bwd, {**feed, "dy": np.ones((5, 7), dtype=np.float32)})

    def test_snn_backward_matches_reference(self):
        """T 步展开后的 SNN：前向与反向都要与参考后端逐点一致。"""
        g = Graph("snn")
        x = g.input("x", f32[3, 4, 5])
        spikes, _ = build_snn(g, x, LIFSpec(decay=0.5, threshold=0.5, alpha=1.0))
        g.output(mean_rate(g, spikes), "y")
        feed = {"x": np.abs(self.rand(3, 4, 5)) * 0.8}
        self.assert_backends_equal(g, feed)

        bwd = backward(g)
        self._assert_graphs_equal(bwd, {**feed, "dy": np.ones((4, 5), dtype=np.float32)})

    def test_layernorm_backward_matches_reference(self):
        g = Graph("ln")
        x = g.input("x", f32[4, 6])
        gamma = g.input("gamma", f32[6])
        beta = g.input("beta", f32[6])
        g.output(g.add_node("layernorm", [x, gamma, beta], {"axes": (-1,)}), "y")
        feed = {"x": self.rand(4, 6), "gamma": self.rand(6), "beta": self.rand(6)}
        bwd = backward(g)
        self._assert_graphs_equal(bwd, {**feed, "dy": np.ones((4, 6), dtype=np.float32)})

    def test_softmax_backward_matches_reference(self):
        g = Graph("sm")
        x = g.input("x", f32[4, 6])
        g.output(g.add_node("softmax", [x], {"axis": -1}), "y")
        feed = {"x": self.rand(4, 6) * 2.0}
        bwd = backward(g)
        self._assert_graphs_equal(bwd, {**feed, "dy": self.rand(4, 6)})

    def test_conv_backward_matches_reference(self):
        """卷积反向图（conv2d_input_grad + conv2d_weight_grad）已有 CUDA 内核。"""
        g = Graph("conv")
        x = g.input("x", f32[1, 2, 5, 5])
        w = g.input("w", f32[3, 2, 3, 3])
        g.output(g.add_node("conv2d", [x, w], {"padding": (1, 1)}), "y")
        bwd = backward(g)
        self._assert_graphs_equal(
            bwd,
            {
                "x": self.rand(1, 2, 5, 5),
                "w": self.rand(3, 2, 3, 3),
                "dy": self.rand(1, 3, 5, 5),
            },
        )

    def test_dropout_mask_matches_reference_bitwise(self):
        """随机算子也要对拍，而且是逐位——两边共用 SplitMix64 与 24 位映射。

        容差 0 是有意的：如果哪天参考实现改用 numpy 的 Generator，这条会
        立刻失败。那不是"随机所以只能近似"，而是跨后端复现性被破坏了。
        """
        for p in (0.0, 0.25, 0.9):
            g = Graph("drop")
            x = g.input("x", f32[8, 16])
            g.output(g.dropout_mask(x, p=p, seed=12345), "y")
            self.assert_backends_equal(g, {"x": self.rand(8, 16)}, rtol=0.0, atol=0.0)

    def test_dropout_train_forward_and_backward(self):
        """``x * dropout_mask(x)``：前向逐位一致，反向只经乘法那条边。"""
        g = Graph("drop_train")
        x = g.input("x", f32[6, 10])
        mask = g.dropout_mask(x, p=0.3, seed=7)
        g.output(g.mul(x, mask), "y")
        feed = {"x": self.rand(6, 10)}
        self.assert_backends_equal(g, feed, rtol=0.0, atol=0.0)

        bwd = backward(g)
        self._assert_graphs_equal(bwd, {**feed, "dy": np.ones((6, 10), dtype=np.float32)})

    def test_batch_norm_forward_and_backward(self):
        """BN 的前向是单内核；反向由 transpose + 最后一维归约组合而成。"""
        for shape, axis in (((2, 3, 4, 4), 1), ((3, 2, 4), 0)):
            g = Graph("bn")
            x = g.input("x", f32[shape])
            gamma = g.input("gamma", f32[shape[axis]])
            beta = g.input("beta", f32[shape[axis]])
            g.output(g.batch_norm(x, gamma, beta, axis=axis), "y")
            feed = {
                "x": self.rand(*shape),
                "gamma": self.rand(shape[axis]),
                "beta": self.rand(shape[axis]),
            }
            self.assert_backends_equal(g, feed)
            self._assert_graphs_equal(backward(g), {**feed, "dy": self.rand(*shape)})

    def test_batch_norm_stats_matches_reference(self):
        g = Graph("stats")
        x = g.input("x", f32[3, 4, 5])
        g.output(g.batch_norm_stats(x, axis=0), "y")
        self.assert_backends_equal(g, {"x": self.rand(3, 4, 5)})


class TestCudaModelComponents(CudaTestCase):
    """构件层的开关必须能直接落到 CUDA。

    这两条覆盖的是"组合"而不是单个算子：BN 是自研内核，dropout 是逐位可复现的
    掩码，串进真实 CNN / Transformer 图之后，形状、广播与内存布局都可能出错。
    """

    def test_cnn_with_batch_norm_and_dropout(self):
        from aetherflow.models import CNNSpec, build_cnn

        spec = CNNSpec(
            in_channels=1, image=8, num_classes=3, widths=(2, 4), batch=2,
            batch_norm=True, dropout=0.3, dropout_seed=123,
        )
        g = Graph("cnn_bn_drop")
        inputs, out = build_cnn(g, spec)
        g.output(out, "y")

        rng = np.random.default_rng(31)
        feed = {}
        for name, value in inputs.items():
            shape = tuple(int(d) for d in value.type.shape)
            scale = 1.0 if name == "x" else 0.4
            feed[name] = (rng.standard_normal(shape) * scale).astype(np.float32)
        self.assert_backends_equal(g, feed)

    def test_transformer_with_dropout(self):
        from aetherflow.models import TransformerSpec, build_transformer, causal_mask_bias

        spec = TransformerSpec(
            d_model=8, num_heads=2, d_ff=16, num_layers=2, seq=4, vocab=0,
            batch=2, dropout=0.25, dropout_seed=3,
        )
        g = Graph("lm_drop")
        inputs, out = build_transformer(g, spec)
        g.output(out, "y")

        feed = {
            name: self.rand(*tuple(int(d) for d in value.type.shape))
            for name, value in inputs.items()
        }
        feed["mask"] = causal_mask_bias(spec.seq)
        self.assert_backends_equal(g, feed)


@unittest.skipUnless(CUDA_AVAILABLE, "需要 cupy 与 CUDA 设备")
class TestCudaMultiDevice(unittest.TestCase):
    """多卡真机：枚举、切卡后重建设备侧缓存、驻留参数随卡失效。

    这台机器上 0/1 号卡可能被别的服务占着显存，所以用例只跑很小的图，
    不假设"每块卡都空着"。
    """

    @classmethod
    def setUpClass(cls):
        import cupy as cp

        cls.cp = cp
        if int(cp.cuda.runtime.getDeviceCount()) < 2:
            raise unittest.SkipTest("多卡用例需要至少 2 块可见设备")
        cls.ref = ReferenceBackend()

    def setUp(self):
        self.origin = int(self.cp.cuda.Device().id)

    def tearDown(self):
        self.cp.cuda.Device(self.origin).use()

    def _graph(self):
        g = Graph("multi")
        x = g.input("x", f32[16, 16])
        w = g.input("w", f32[16, 16])
        g.output(g.add_node("matmul", [x, w]), "y")
        rng = np.random.default_rng(0)
        feed = {
            "x": rng.standard_normal((16, 16)).astype(np.float32),
            "w": rng.standard_normal((16, 16)).astype(np.float32),
        }
        return g, feed

    def test_enumerates_every_visible_device(self):
        backend = CudaBackend()
        before = backend.current_device()
        infos = backend.enumerate_devices()
        self.assertEqual(len(infos), int(self.cp.cuda.runtime.getDeviceCount()))
        for info in infos:
            self.assertTrue(info.name)
            self.assertGreater(info.total_memory_bytes, 0)
        # 枚举本身不能改变"现在用哪块卡"
        self.assertEqual(backend.current_device(), before)

    def test_runs_on_another_device_after_switch(self):
        """切卡必须丢掉旧 context 的编译产物与计划；否则这里会启动失败。"""
        backend = CudaBackend()
        g, feed = self._graph()
        expected = self.ref.execute(g, feed)["y"]
        np.testing.assert_allclose(
            backend.execute(g, feed)["y"], expected, rtol=1e-4, atol=1e-4
        )
        backend.set_device(1)
        self.assertEqual(backend.current_device(), 1)
        np.testing.assert_allclose(
            backend.execute(g, feed)["y"], expected, rtol=1e-4, atol=1e-4
        )

    def test_switch_drops_resident_parameters(self):
        backend = CudaBackend()
        g, feed = self._graph()
        backend.resident_parameters(g, {"w": feed["w"]})
        self.assertEqual(list(backend.resident), ["w"])
        backend.set_device(1)
        self.assertEqual(dict(backend.resident), {})

    def test_each_device_runs_its_own_graph(self):
        backends = [CudaBackend(device=0), CudaBackend(device=1)]
        g, feed = self._graph()
        expected = self.ref.execute(g, feed)["y"]
        for index, backend in enumerate(backends):
            with self.cp.cuda.Device(index):
                np.testing.assert_allclose(
                    backend.execute(g, feed)["y"], expected, rtol=1e-4, atol=1e-4
                )

    def test_rejects_bad_index(self):
        backend = CudaBackend()
        count = int(self.cp.cuda.runtime.getDeviceCount())
        for bad in (count, -1, "0", True):
            with self.assertRaises(BackendCapabilityError):
                backend.set_device(bad)
        self.assertEqual(backend.current_device(), int(self.cp.cuda.Device().id))


@unittest.skipUnless(CUDA_AVAILABLE, "需要 cupy 与 CUDA 设备")
class TestCudaTieredExecution(unittest.TestCase):
    """``memory_mode="tiered"`` 在 CUDA 上的执行路径。

    与 HIP 那份同形：数值与 container 一致、统计有值、分配器成对还原。
    额外钉一条 CUDA 特有的：同一张图重复执行不新要显存（分层真的在复用）。
    """

    @staticmethod
    def _graph():
        g = Graph("tiered")
        a = g.input("a", f32[4, 8])
        b = g.input("b", f32[8])
        g.output(g.mul(g.relu(g.add(a, b)), a), "y")
        return g

    @staticmethod
    def _feed():
        return {
            "a": (np.arange(32, dtype=np.float32).reshape(4, 8) - 16.0) / 8.0,
            "b": np.linspace(-1.0, 1.0, 8, dtype=np.float32),
        }

    def test_tiered_matches_container(self):
        graph = self._graph()
        want = CudaBackend(memory_mode="container").execute(graph, self._feed())["y"]
        backend = CudaBackend(memory_mode="tiered")
        got = backend.execute(graph, self._feed())["y"]
        np.testing.assert_array_equal(got, want)

        stats = backend.memory_stats()
        self.assertIsNotNone(stats)
        self.assertGreater(stats.total_bytes, 0)
        self.assertGreater(stats.peak_bytes, 0)
        self.assertGreaterEqual(stats.peak_bytes, stats.live_bytes)

    def test_allocator_is_restored_after_execute(self):
        import cupy as cp

        backend = CudaBackend(memory_mode="tiered")
        before = cp.cuda.get_allocator()
        backend.execute(self._graph(), self._feed())
        self.assertIs(cp.cuda.get_allocator(), before)

    def test_container_mode_does_not_install_allocator(self):
        import cupy as cp

        backend = CudaBackend(memory_mode="container")
        before = cp.cuda.get_allocator()
        backend.execute(self._graph(), self._feed())
        self.assertIs(cp.cuda.get_allocator(), before)
        self.assertIsNone(backend.memory_stats())

    def test_repeated_execute_reuses_arena(self):
        backend = CudaBackend(memory_mode="tiered")
        graph = self._graph()
        backend.execute(graph, self._feed())
        first = backend.memory_stats()
        backend.execute(graph, self._feed())
        second = backend.memory_stats()
        self.assertGreater(first.total_bytes, 0)
        # 同图同形状第二遍不该向驱动再要新块
        self.assertLessEqual(second.total_bytes, first.total_bytes)
        self.assertLessEqual(second.peak_bytes, first.peak_bytes)


@unittest.skipUnless(CUDA_AVAILABLE, "需要 cupy 与 CUDA 设备")
class TestCudaConstantCache(unittest.TestCase):
    """常量在真机上只 H2D 一次，且切卡 / 清计划后会重新上传（5.172）。"""

    @staticmethod
    def _graph():
        g = Graph("constant_cache")
        c = g.constant(3.0, shape=(8, 8))
        x = g.input("x", f32[8, 8])
        g.output(g.mul(x, c), "y")
        return g

    @staticmethod
    def _counting(backend):
        calls = []
        original = backend._op_constant

        def counting(node, args, attrs):
            calls.append(node.output.id)
            return original(node, args, attrs)

        backend._op_constant = counting
        return calls

    def test_constant_uploaded_once_across_executes(self):
        backend = CudaBackend()
        calls = self._counting(backend)
        graph = self._graph()
        feed = {"x": np.ones((8, 8), dtype=np.float32)}
        first = backend.execute(graph, feed)["y"]
        second = backend.execute(graph, feed)["y"]
        np.testing.assert_allclose(first, 3.0)
        np.testing.assert_array_equal(second, first)
        self.assertEqual(len(calls), 1)

    def test_constant_reuploaded_after_plan_reset(self):
        backend = CudaBackend()
        calls = self._counting(backend)
        graph = self._graph()
        feed = {"x": np.ones((8, 8), dtype=np.float32)}
        backend.execute(graph, feed)
        backend._clear_plan_values()
        out = backend.execute(graph, feed)["y"]
        np.testing.assert_allclose(out, 3.0)
        self.assertEqual(len(calls), 2)


@unittest.skipUnless(CUDA_AVAILABLE, "需要 cupy 与 CUDA 设备")
class TestCudaTuning(unittest.TestCase):
    """自动调优的真机闭环：空间 → 搜索 → 应用 → 结果逐位一致。

    本地只能测搜索器本身（``test_tuning``）；这里证明 CUDA 插件真的把
    block 接进了 launch，并且换 block 不改变数值。
    """

    @classmethod
    def setUpClass(cls):
        cls.backend = CudaBackend()

    def _relu_graph(self):
        g = Graph("relu")
        x = g.input("x", f32[1024, 1024])
        g.output(g.relu(x), "y")
        rng = np.random.default_rng(5)
        return g, {"x": rng.standard_normal((1024, 1024)).astype(np.float32)}

    def test_tunable_ops_are_registered_and_reductions_are_included(self):
        tunable = self.backend.tunable_ops()
        for op in tunable:
            self.assertIn(op, self.backend.capabilities())
        # 5.160：归约类的 shared 暂存区是固定 4 KB、内核沿 blockDim.x 跨步读，
        # block 因此可以拧（此前按"shared 与 blockDim 绑定"排除，那个前提不成立）
        for op in ("softmax", "layernorm", "rmsnorm", "reduce_sum"):
            self.assertIn(op, tunable)
            self.assertIn(None, self.backend.tuning_space(op).values("block"))

    def test_space_starts_with_the_current_default(self):
        space = self.backend.tuning_space("add")
        self.assertEqual(space.default(), {"block": self.backend.block})
        values = space.values("block")
        # 默认（None = 自动）也是候选：一个形状一个值，钉不成常数
        self.assertEqual(len(values), len(BLOCK_OPTIONS) + 1)
        self.assertIn(None, values)
        self.assertEqual(len(set(values)), len(values))

    def test_unknown_or_invalid_tuning_is_rejected(self):
        with self.assertRaises(ValueError):
            self.backend.tuning_space("no_such_op")
        with self.assertRaises(ValueError):
            self.backend.set_tuning("matmul", {"block": 128})
        for bad in (0, 48, 2048, 128.0, True):
            with self.assertRaises(ValueError):
                self.backend.set_tuning("add", {"block": bad})
        with self.assertRaises(ValueError):
            self.backend.set_tuning("add", {"grid": 128})
        with self.assertRaises(ValueError):
            self.backend.set_tuning("add", {})

    def test_block_change_keeps_results_bit_exact(self):
        g, feed = self._relu_graph()
        default = self.backend.execute(g, feed)["y"]
        try:
            for block in (128, 512, 1024):
                self.backend.set_tuning("relu", {"block": block})
                self.assertEqual(
                    self.backend.tuning_config(), {"relu": {"block": block}}
                )
                np.testing.assert_array_equal(self.backend.execute(g, feed)["y"], default)
        finally:
            self.backend.clear_tuning()
        self.assertEqual(self.backend.tuning_config(), {})

    def test_auto_block_follows_the_shape(self):
        """没人指定 block 时按规模推：小张量缩块，大张量落到默认值。"""
        self.assertIsNone(self.backend.block)
        small = self.backend._launch_block("relu", 4096)
        self.assertEqual(
            small, auto_block(4096, target_blocks=self.backend.sm_count)
        )
        if self.backend.sm_count > 16:  # 4096 个元素够不上 16 块时不会缩
            self.assertLess(small, 256)
        self.assertEqual(self.backend._launch_block("relu", 1 << 20), 256)
        # 调优覆盖优先于自动
        try:
            self.backend.set_tuning("relu", {"block": 512})
            self.assertEqual(self.backend._launch_block("relu", 4096), 512)
        finally:
            self.backend.clear_tuning("relu")
        # block=None 表示"回到自动"：撤掉覆盖，而不是存一个 None 进去
        try:
            self.backend.set_tuning("relu", {"block": None})
            self.assertEqual(self.backend.tuning_config(), {})
            self.assertEqual(self.backend._launch_block("relu", 4096), small)
        finally:
            self.backend.clear_tuning("relu")

    def test_small_shape_auto_matches_a_fixed_block(self):
        """块只决定线程怎么分组：自动与小 block/大 block 都必须逐位相同。"""
        g = Graph("tiny")
        x = g.input("x", f32[64])
        g.output(g.relu(x), "y")
        feed = {"x": np.linspace(-1.0, 1.0, 64, dtype=np.float32)}
        default = self.backend.execute(g, feed)["y"]
        try:
            for block in (32, 256, 1024):
                self.backend.set_tuning("relu", {"block": block})
                np.testing.assert_array_equal(
                    self.backend.execute(g, feed)["y"], default
                )
        finally:
            self.backend.clear_tuning("relu")

    def test_reduce_block_override_reaches_launch(self):
        """归约类的 block 覆盖必须真的落到 launch 参数上（5.160）。

        这里只读 ``_reduce_block``，不启动内核：默认按列数取 2 的幂、覆盖优先、
        撤掉覆盖后回到按列数推导。归约的 shared 是固定大小，覆盖值可以故意
        小于列数——内核沿 ``blockDim.x`` 跨步读。
        """
        self.assertIsNone(self.backend.block)
        self.assertEqual(self.backend._reduce_block("softmax", 100), 128)
        self.assertEqual(self.backend._reduce_block("softmax", 5000), 1024)
        try:
            self.backend.set_tuning("softmax", {"block": 128})
            self.assertEqual(self.backend._reduce_block("softmax", 5000), 128)
            # 只影响被覆盖的那个算子
            self.assertEqual(self.backend._reduce_block("layernorm", 5000), 1024)
        finally:
            self.backend.clear_tuning("softmax")

    def test_reduce_block_change_stays_close(self):
        """归约 block 变小后每个线程跨步多走几步，数值只能近似一致。

        归约顺序随块大小改变，所以这里不能用 ``assert_array_equal``——这是
        逐元素类与归约类在"换 block 不改变结果"上的唯一差别。
        """
        g = Graph("softmax_tune")
        x = g.input("x", f32[512, 4096])
        g.output(g.softmax(x), "y")
        feed = {
            "x": np.random.default_rng(11)
            .standard_normal((512, 4096))
            .astype(np.float32)
        }
        default = self.backend.execute(g, feed)["y"]
        try:
            for block in (128, 1024):
                self.backend.set_tuning("softmax", {"block": block})
                np.testing.assert_allclose(
                    self.backend.execute(g, feed)["y"],
                    default,
                    rtol=1e-5,
                    atol=1e-6,
                )
        finally:
            self.backend.clear_tuning("softmax")

    def test_conv_tile_override_keeps_results_bit_exact(self):
        """三个旋钮只改线程铺法，累加顺序不变，数值必须逐位相同。"""
        g = Graph("conv_tile")
        x = g.input("x", f32[1, 2, 16, 40])
        w = g.input("w", f32[8, 2, 3, 3])
        g.output(g.conv2d(x, w, padding=(1, 1)), "y")
        rng = np.random.default_rng(11)
        feed = {
            "x": rng.standard_normal((1, 2, 16, 40)).astype(np.float32),
            "w": rng.standard_normal((8, 2, 3, 3)).astype(np.float32) * 0.3,
        }
        # K = 2*3*3 = 18 <= 32 且 work = 8*16*40 = 5120 >= 4096：auto 会选分块核
        default = self.backend.execute(g, feed)["y"]
        overrides = (
            {"tile_w": 16},
            {"tile_w": 64},
            {"roc": 1},
            {"roc": 2},
            {"cb": 1},
            {"tile_w": 16, "roc": 2, "cb": 1},
        )
        try:
            for config in overrides:
                # 每条覆盖都从"无覆盖"开始：set_tuning 是增量合并，不清会串味
                self.backend.clear_tuning("conv2d")
                self.backend.set_tuning("conv2d", config)
                self.assertEqual(
                    self.backend.tuning_config(), {"conv2d": dict(config)}
                )
                np.testing.assert_array_equal(
                    self.backend.execute(g, feed)["y"], default
                )
        finally:
            self.backend.clear_tuning()
        self.assertEqual(self.backend.tuning_config(), {})

    def test_conv_tuning_space_and_validation(self):
        """conv2d 的旋钮是分块核的三个参数，不是 block；非法值明确拒绝。"""
        self.assertIn("conv2d", self.backend.tunable_ops())
        space = self.backend.tuning_space("conv2d")
        self.assertEqual(space.names(), ("tile_w", "roc", "cb"))
        # 无覆盖时首候选是"自动"：默认值随形状变，不能钉成某个常数
        self.assertEqual(
            space.default(), {"tile_w": None, "roc": None, "cb": None}
        )
        with self.assertRaises(ValueError):
            self.backend.set_tuning("conv2d", {"block": 256})
        with self.assertRaises(ValueError):
            self.backend.set_tuning("conv2d", {})
        for name in ("tile_w", "roc", "cb"):
            for bad in (0, -4, 8.0, True):
                with self.assertRaises(ValueError):
                    self.backend.set_tuning("conv2d", {name: bad})
        try:
            self.backend.set_tuning("conv2d", {"tile_w": 16})
            self.assertEqual(
                self.backend.tuning_config(), {"conv2d": {"tile_w": 16}}
            )
            self.assertEqual(
                self.backend.tuning_space("conv2d").default(),
                {"tile_w": 16, "roc": None, "cb": None},
            )
            self.backend.set_tuning("conv2d", {"roc": 2, "cb": 4})
            self.assertEqual(
                self.backend.tuning_config(),
                {"conv2d": {"tile_w": 16, "roc": 2, "cb": 4}},
            )
            # None 只撤掉那一项覆盖，其余项保留
            self.backend.set_tuning("conv2d", {"roc": None})
            self.assertEqual(
                self.backend.tuning_config(),
                {"conv2d": {"tile_w": 16, "cb": 4}},
            )
        finally:
            self.backend.clear_tuning("conv2d")
        self.assertEqual(
            self.backend.tuning_space("conv2d").default(),
            {"tile_w": None, "roc": None, "cb": None},
        )

    def test_dropout_mask_is_block_independent(self):
        """掩码由全局下标决定，换 block 必须逐位相同。"""
        g = Graph("dropout")
        x = g.input("x", f32[4096])
        g.output(g.dropout_mask(x, p=0.25, seed=99), "y")
        feed = {"x": np.ones((4096,), dtype=np.float32)}
        default = self.backend.execute(g, feed)["y"]
        try:
            self.backend.set_tuning("dropout_mask", {"block": 1024})
            np.testing.assert_array_equal(self.backend.execute(g, feed)["y"], default)
        finally:
            self.backend.clear_tuning("dropout_mask")

    def test_search_produces_a_reusable_store(self):
        from benchmarks.tune import TUNE_CASES, apply_and_verify, search

        case = next(c for c in TUNE_CASES if c.name == "relu_1m")
        # 交替 A/B 至少 2 轮（1 轮无法抵消顺序效应），默认走配对测量
        store = search(self.backend, cases=[case], rounds=2)
        self.assertEqual(len(store), 1)
        record = store.records()[0]
        self.assertEqual(record.op, "relu")
        # None 就是"自动"，它与四档固定 block 一样是合法赢家
        self.assertIn(record.config["block"], (None,) + BLOCK_OPTIONS)
        self.assertEqual(record.samples, 2)
        self.assertGreater(record.median_ms, 0.0)

        hits, problems = apply_and_verify(self.backend, store, cases=[case])
        self.assertEqual(hits, 1)
        self.assertEqual(problems, [])
        self.assertEqual(self.backend.tuning_config(), {})

    def test_telemetry_wraps_cuda_and_reports_device(self):
        from aetherflow import TelemetryRecorder, implemented_contracts

        g = Graph("add_relu")
        a = g.input("a", f32[64, 64])
        b = g.input("b", f32[64, 64])
        g.output(g.relu(g.add(a, b)), "y")
        rng = np.random.default_rng(7)
        feed = {
            "a": rng.standard_normal((64, 64)).astype(np.float32),
            "b": rng.standard_normal((64, 64)).astype(np.float32),
        }
        recorder = TelemetryRecorder()
        wrapped = recorder.wrap(self.backend)
        wrapped.execute(g, feed, units={"elements": 4096})

        record = recorder.last
        self.assertEqual(record.backend, "cuda")
        self.assertEqual(record.device, self.backend.device_info().name)
        self.assertEqual(len(record.nodes), len(g.nodes))
        self.assertGreater(record.throughput()["elements"], 0.0)
        # 包装器不能凭空声称多出来的契约
        self.assertEqual(
            set(implemented_contracts(wrapped)),
            set(implemented_contracts(self.backend)),
        )


class TestGpuClockElapsed(unittest.TestCase):
    """``_CudaGpuClock.elapsed`` 要同时兼容两代 cupy 的事件计时接口。

    cupy 13 移除了 ``cupy.cuda.Event.elapsed_time``，只剩
    ``runtime.eventElapsedTime(start.ptr, stop.ptr)``。旧实现直接调前者，抛出的
    ``AttributeError`` 会被调用方当成"没有计时"静默吞掉——整块逐节点 GPU
    归因能力形同虚设，而且不会报错。这里用替身把两条分支都钉住，不需要 GPU。
    """

    class _OldEvent:
        """旧版 cupy 的事件：自带 ``elapsed_time``。"""

        def __init__(self, ptr):
            self.ptr = ptr

        def elapsed_time(self, stop):
            return 1.5

    class _NewEvent:
        """cupy 13+ 的事件：只有 ``ptr``，计时走 runtime。"""

        def __init__(self, ptr):
            self.ptr = ptr

    class _Backend:
        def __init__(self, runtime):
            self.cp = mock.MagicMock()
            self.cp.cuda.runtime = runtime

    def test_new_cupy_uses_runtime_event_elapsed_time(self):
        from aetherflow.backends.cuda.backend import _CudaGpuClock

        calls = []

        class _Runtime:
            @staticmethod
            def eventElapsedTime(start, stop):
                calls.append((start, stop))
                return 2.25

        clock = _CudaGpuClock(TestGpuClockElapsed._Backend(_Runtime))
        start, stop = self._NewEvent(11), self._NewEvent(22)
        self.assertEqual(clock.elapsed(start, stop), 2.25)
        self.assertEqual(calls, [(11, 22)], "必须传 .ptr 而不是事件对象")

    def test_old_cupy_prefers_event_method(self):
        from aetherflow.backends.cuda.backend import _CudaGpuClock

        class _Runtime:
            @staticmethod
            def eventElapsedTime(start, stop):  # 不该被走到
                raise AssertionError("旧版 cupy 应优先用 elapsed_time")

        clock = _CudaGpuClock(TestGpuClockElapsed._Backend(_Runtime))
        self.assertEqual(clock.elapsed(self._OldEvent(1), self._OldEvent(2)), 1.5)


if __name__ == "__main__":
    unittest.main()
