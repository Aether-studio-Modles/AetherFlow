"""张量并行：列 / 行切分的数值等价、头切分与 TP 构建器。

判据是"切分后的结果等于不分片的同一张图"：单卡用 ``ReferenceBackend``，
多 rank 用 ``LoopbackWorld`` 的线程屏障（与 ZeRO / 流水用例同一个替身）。
Loopback 按 rank 顺序累加部分和，所以与不分片的矩阵乘只差 BLAS 的求和
顺序，断言取 ``1e-5`` 的相对 / 绝对容差，不写"逐位"。

本轮不接反向、不接优化器分片：用例只覆盖前向与切分接口的拒绝面。
"""

import threading
import unittest

import numpy as np

from aetherflow import Graph, f32, verify
from aetherflow.backends.reference_backend import ReferenceBackend
from aetherflow.models.transformer import BlockSpec, build_transformer_block
from aetherflow.train import LoopbackWorld
from aetherflow.train.tensor_parallel import (
    TensorParallelBlock,
    build_tensor_parallel_attention,
    column_parallel_linear,
    merge_axis,
    row_parallel_linear,
    shard_size,
    split_axis,
)


def _dense_params(spec, seed=0):
    rng = np.random.default_rng(seed)
    return {
        name: (rng.standard_normal(shape) * 0.2).astype(np.float32)
        for name, shape in spec.param_shapes.items()
    }


def _dense_block(spec, params, x):
    g = Graph("dense")
    _, out = build_transformer_block(g, spec)
    g.output(out, "y")
    feed = {"x": x}
    feed.update(params)
    return ReferenceBackend().execute(g, feed)["y"]


def _run_ranks(blocks, x, per_rank, world):
    """多 rank 并发跑前向；任一 rank 抛错就原样抛出。"""
    results = {}
    errors = []

    def work(block):
        try:
            results[block.rank] = block.run(world.collective(block.rank), x, per_rank[block.rank])
        except BaseException as exc:  # noqa: BLE001 - 测试要把首个异常带回主线程
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(b,)) for b in blocks]
    for item in threads:
        item.start()
    for item in threads:
        item.join(timeout=30.0)
        if item.is_alive():
            raise AssertionError("张量并行的 rank 线程没有在 30 秒内结束")
    if errors:
        raise errors[0]
    return results


class TestShardUtilities(unittest.TestCase):
    def test_shard_size_divides(self):
        self.assertEqual(shard_size(12, 3), 4)
        self.assertEqual(shard_size(7, 1), 7)
        self.assertEqual(shard_size(8, 4, what="num_heads"), 2)

    def test_shard_size_rejects_indivisible(self):
        with self.assertRaises(ValueError) as ctx:
            shard_size(10, 3, what="d_ff")
        self.assertIn("不能被 world_size=3 整除", str(ctx.exception))

    def test_shard_size_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            shard_size(4, 0)
        with self.assertRaises(TypeError):
            shard_size(True, 1)
        with self.assertRaises(TypeError):
            shard_size(4, True)

    def test_split_axis_is_contiguous_and_copied(self):
        array = np.arange(24, dtype=np.float32).reshape(4, 6)
        first = split_axis(array, 1, 0, 3)
        second = split_axis(array, 1, 1, 3)
        third = split_axis(array, 1, 2, 3)
        self.assertEqual(first.shape, (4, 2))
        np.testing.assert_array_equal(first, array[:, :2])
        np.testing.assert_array_equal(second, array[:, 2:4])
        np.testing.assert_array_equal(third, array[:, 4:])
        # 切片必须是副本：优化器原地更新不能跨 rank 互相污染
        first[...] = -1.0
        np.testing.assert_array_equal(array[:, :2], np.array([[0, 1], [6, 7], [12, 13], [18, 19]], np.float32))

    def test_split_axis_rejects_out_of_range(self):
        array = np.zeros((4, 6), dtype=np.float32)
        with self.assertRaises(ValueError):
            split_axis(array, 1, 3, 3)
        with self.assertRaises(ValueError):
            split_axis(array, 2, 0, 2)
        with self.assertRaises(TypeError):
            split_axis(array, 1, True, 2)

    def test_merge_axis_roundtrip(self):
        rng = np.random.default_rng(4)
        array = rng.standard_normal((3, 8, 5)).astype(np.float32)
        for axis in (1, 2, 0):
            for world in (1, 2, 4):
                if array.shape[axis] % world:
                    continue
                parts = [split_axis(array, axis, r, world) for r in range(world)]
                merged = merge_axis(parts, axis)
                np.testing.assert_array_equal(merged, array)

    def test_merge_axis_rejects_mismatch(self):
        with self.assertRaises(ValueError):
            merge_axis([], 0)
        with self.assertRaises(ValueError):
            merge_axis([np.zeros((2, 3)), np.zeros((2, 4))], 0)
        with self.assertRaises(TypeError):
            merge_axis("not a sequence", 0)


class TestParallelLinearGraphs(unittest.TestCase):
    def _run_columns(self, x, weight, bias=None, world=2):
        parts = []
        for rank in range(world):
            g = Graph(f"col{rank}")
            xv = g.input("x", f32[2, 3])
            wv = g.input("w", f32[3, weight.shape[1] // world])
            feed = {"x": x, "w": split_axis(weight, 1, rank, world)}
            biasv = None
            if bias is not None:
                biasv = g.input("b", f32[bias.shape[0] // world])
                feed["b"] = split_axis(bias, 0, rank, world)
            g.output(column_parallel_linear(g, xv, wv, biasv), "y")
            self.assertEqual([str(i) for i in verify(g)], [])
            parts.append(ReferenceBackend().execute(g, feed)["y"])
        return parts

    def test_column_parallel_concat_equals_dense(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((2, 3)).astype(np.float32)
        weight = rng.standard_normal((3, 8)).astype(np.float32)
        parts = self._run_columns(x, weight)
        np.testing.assert_allclose(
            merge_axis(parts, 1), x @ weight, rtol=1e-6, atol=1e-6
        )

    def test_column_parallel_bias_is_sharded(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((2, 3)).astype(np.float32)
        weight = rng.standard_normal((3, 6)).astype(np.float32)
        bias = rng.standard_normal(6).astype(np.float32)
        parts = self._run_columns(x, weight, bias, world=3)
        np.testing.assert_allclose(
            merge_axis(parts, 1), x @ weight + bias, rtol=1e-6, atol=1e-6
        )

    def test_row_parallel_partials_sum_to_dense(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((2, 8)).astype(np.float32)
        weight = rng.standard_normal((8, 5)).astype(np.float32)
        world = 4
        parts = []
        for rank in range(world):
            g = Graph(f"row{rank}")
            xv = g.input("x", f32[2, 8 // world])
            wv = g.input("w", f32[8 // world, 5])
            g.output(row_parallel_linear(g, xv, wv), "y")
            parts.append(
                ReferenceBackend().execute(
                    g,
                    {"x": split_axis(x, 1, rank, world), "w": split_axis(weight, 0, rank, world)},
                )["y"]
            )
        total = np.zeros((2, 5), dtype=np.float32)
        for part in parts:
            total += part
        np.testing.assert_allclose(total, x @ weight, rtol=1e-6, atol=1e-6)

    def test_row_parallel_rejects_bias(self):
        g = Graph("row_bias")
        xv = g.input("x", f32[2, 4])
        wv = g.input("w", f32[4, 3])
        bv = g.input("b", f32[3])
        with self.assertRaises(ValueError) as ctx:
            row_parallel_linear(g, xv, wv, bv)
        self.assertIn("不接受 bias", str(ctx.exception))

    def test_linear_rejects_bad_shapes(self):
        g = Graph("bad")
        xv = g.input("x", f32[2, 4])
        wv = g.input("w", f32[3, 3])
        with self.assertRaises(ValueError) as ctx:
            column_parallel_linear(g, xv, wv)
        self.assertIn("内维不匹配", str(ctx.exception))
        w3 = g.input("w3", f32[2, 2, 2])
        with self.assertRaises(ValueError) as ctx:
            column_parallel_linear(g, xv, w3)
        self.assertIn("必须是 2 维", str(ctx.exception))


class TestAttentionHeadSplit(unittest.TestCase):
    def _build(self, spec, rank, world, fused=False):
        g = Graph("tp_attn")
        h = g.input("h", f32[spec.batch, spec.seq, spec.d_model])
        width = (spec.num_heads // world) * spec.head_dim
        wq = g.input("wq", f32[spec.d_model, width])
        wk = g.input("wk", f32[spec.d_model, width])
        wv = g.input("wv", f32[spec.d_model, width])
        wo = g.input("wo", f32[width, spec.d_model])
        scale = None if fused else g.input("scale", f32[1])
        out = build_tensor_parallel_attention(
            g,
            h,
            wq,
            wk,
            wv,
            wo,
            num_heads=spec.num_heads,
            d_model=spec.d_model,
            batch=spec.batch,
            seq=spec.seq,
            rank=rank,
            world_size=world,
            scale=scale,
            fused_attention=fused,
        )
        g.output(out, "attn")
        return g

    def test_heads_split_matches_dense_attention(self):
        spec = BlockSpec(d_model=16, num_heads=4, d_ff=32)
        rng = np.random.default_rng(7)
        h = rng.standard_normal((2, 3, spec.d_model)).astype(np.float32)
        weights = {
            name: (rng.standard_normal(shape) * 0.2).astype(np.float32)
            for name, shape in {
                "wq": (16, 16),
                "wk": (16, 16),
                "wv": (16, 16),
                "wo": (16, 16),
            }.items()
        }
        scale = np.array([1.0 / np.sqrt(spec.head_dim)], dtype=np.float32)
        world = 2
        partials = []
        for rank in range(world):
            g = self._build(spec, rank, world)
            feed = {
                "h": h,
                "wq": split_axis(weights["wq"], 1, rank, world),
                "wk": split_axis(weights["wk"], 1, rank, world),
                "wv": split_axis(weights["wv"], 1, rank, world),
                "wo": split_axis(weights["wo"], 0, rank, world),
                "scale": scale,
            }
            partials.append(ReferenceBackend().execute(g, feed)["attn"])
        got = sum(partials[1:], partials[0])

        # 不分片的同一段数学（自己按完整头数算一遍）：先做 q/k/v 投影再分头
        q = (h @ weights["wq"]).reshape(2, 3, 4, 4).transpose(0, 2, 1, 3)
        k = (h @ weights["wk"]).reshape(2, 3, 4, 4).transpose(0, 2, 1, 3)
        v = (h @ weights["wv"]).reshape(2, 3, 4, 4).transpose(0, 2, 1, 3)
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        prob = np.exp(scores - scores.max(-1, keepdims=True))
        prob /= prob.sum(-1, keepdims=True)
        ctx = np.matmul(prob, v).transpose(0, 2, 1, 3).reshape(2, 3, 16)
        want = ctx @ weights["wo"]
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

    def test_head_count_must_divide(self):
        spec = BlockSpec(d_model=24, num_heads=6, d_ff=16)
        with self.assertRaises(ValueError) as ctx:
            self._build(spec, 0, 4)
        self.assertIn("num_heads=6 不能被 world_size=4 整除", str(ctx.exception))

    def test_missing_scale_rejected(self):
        g = Graph("no_scale")
        h = g.input("h", f32[2, 3, 16])
        w = g.input("w", f32[16, 8])
        wo = g.input("wo", f32[8, 16])
        with self.assertRaises(ValueError) as ctx:
            build_tensor_parallel_attention(
                g,
                h,
                w,
                w,
                w,
                wo,
                num_heads=4,
                d_model=16,
                batch=2,
                seq=3,
                rank=0,
                world_size=2,
                scale=None,
                fused_attention=False,
            )
        self.assertIn("需要 scale", str(ctx.exception))


class TestTensorParallelBlock(unittest.TestCase):
    spec = BlockSpec(d_model=32, num_heads=4, d_ff=64)

    def _weights(self):
        return _dense_params(self.spec, seed=11)

    def _check_world(self, world, fused=False):
        spec = self.spec
        params = self._weights()
        rng = np.random.default_rng(5)
        x = rng.standard_normal((2, 3, spec.d_model)).astype(np.float32)
        blocks = [TensorParallelBlock(spec, rank, world) for rank in range(world)]
        per_rank = {b.rank: b.split_dense(params) for b in blocks}
        world_obj = LoopbackWorld(world)
        got = _run_ranks(blocks, x, per_rank, world_obj)[0]
        want = _dense_block(spec, params, x)
        self.assertEqual(got.shape, want.shape)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
        return blocks, per_rank, params

    def test_world_one_matches_dense(self):
        self._check_world(1)

    def test_world_two_matches_dense(self):
        self._check_world(2)

    def test_world_four_matches_dense(self):
        self._check_world(4)

    def test_fused_attention_matches_dense(self):
        spec = BlockSpec(d_model=32, num_heads=4, d_ff=64, fused_attention=True)
        params = _dense_params(spec, seed=13)
        rng = np.random.default_rng(6)
        x = rng.standard_normal((2, 4, spec.d_model)).astype(np.float32)
        blocks = [TensorParallelBlock(spec, r, 2) for r in range(2)]
        per_rank = {b.rank: b.split_dense(params) for b in blocks}
        got = _run_ranks(blocks, x, per_rank, LoopbackWorld(2))[0]
        np.testing.assert_allclose(
            got, _dense_block(spec, params, x), rtol=1e-5, atol=1e-5
        )

    def test_local_shapes_shrink_by_world(self):
        blocks, per_rank, params = self._check_world(4)
        block = blocks[0]
        self.assertEqual(block.param_shapes["wq"], (32, 8))
        self.assertEqual(block.param_shapes["wo"], (8, 32))
        self.assertEqual(block.param_shapes["w1"], (32, 16))
        self.assertEqual(block.param_shapes["w2"], (16, 32))
        # 复制类参数不缩水
        self.assertEqual(block.param_shapes["ln1_g"], (32,))
        self.assertEqual(block.param_shapes["scale"], (1,))
        local_bytes = sum(int(np.asarray(v).nbytes) for v in per_rank[0].values())
        full_bytes = sum(int(np.asarray(v).nbytes) for v in params.values())
        self.assertLess(local_bytes, full_bytes)

    def test_graphs_verify_and_are_portable(self):
        block = TensorParallelBlock(self.spec, 0, 2)
        for graph in block.build():
            self.assertEqual([str(i) for i in verify(graph)], [])

    def test_split_dense_rejects_bad_params(self):
        block = TensorParallelBlock(self.spec, 0, 2)
        params = self._weights()
        missing = dict(params)
        missing.pop("wq")
        with self.assertRaises(ValueError) as ctx:
            block.split_dense(missing)
        self.assertIn("缺少参数", str(ctx.exception))
        extra = dict(params)
        extra["bogus"] = np.zeros(2, dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            block.split_dense(extra)
        self.assertIn("多出参数", str(ctx.exception))
        wrong = dict(params)
        wrong["wq"] = np.zeros((32, 31), dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            block.split_dense(wrong)
        self.assertIn("不一致", str(ctx.exception))

    def test_run_rejects_bad_input(self):
        block = TensorParallelBlock(self.spec, 0, 2)
        params = block.split_dense(self._weights())
        world = LoopbackWorld(2)
        collective = world.collective(0)
        with self.assertRaises(ValueError) as ctx:
            block.run(collective, np.zeros((2, 32), np.float32), params)
        self.assertIn("必须是 3 维", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            block.run(collective, np.zeros((2, 3, 8), np.float32), params)
        self.assertIn("d_model", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            block.run(collective, np.zeros((2, 3, 32), np.float32), {"wq": params["wq"]})
        self.assertIn("缺少本 rank 的参数", str(ctx.exception))

    def test_run_without_collective(self):
        block = TensorParallelBlock(self.spec, 0, 2)
        params = block.split_dense(self._weights())
        with self.assertRaises(TypeError) as ctx:
            block.run(None, np.zeros((2, 3, 32), np.float32), params)
        self.assertIn("all_reduce", str(ctx.exception))

    def test_block_rejects_bad_configuration(self):
        with self.assertRaises(ValueError) as ctx:
            TensorParallelBlock(BlockSpec(d_model=32, num_heads=4, d_ff=50), 0, 4)
        self.assertIn("d_ff=50", str(ctx.exception))
        with self.assertRaises(ValueError):
            TensorParallelBlock(self.spec, 2, 2)
        with self.assertRaises(ValueError):
            TensorParallelBlock(self.spec, 0, 0)
        with self.assertRaises(ValueError) as ctx:
            TensorParallelBlock(BlockSpec(dropout=0.1), 0, 2)
        self.assertIn("不支持 dropout", str(ctx.exception))
        with self.assertRaises(TypeError):
            TensorParallelBlock(self.spec, True, 2)


if __name__ == "__main__":
    unittest.main()
