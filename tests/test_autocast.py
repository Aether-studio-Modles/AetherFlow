"""自动混合精度 pass 的测试：不需要 GPU。

三层判据：
- 图层面：哪些算子被降到 f16、哪些被升回 f32、边界插了几次 cast；
- 数值：与全 f32 图在 f16 容差内一致，且对外输出的 dtype 不变；
- 闭环：反向图的参数梯度回到 f32（这正是"主权重 f32 + 计算 f16"的意思）。
"""

import unittest

import numpy as np

from aetherflow import Graph, ReferenceBackend, backward, f16, f32, f64, i32
from aetherflow.backends.cuda import CUDA_AVAILABLE, CudaBackend
from aetherflow.backends.cuda.codegen import cast_fn
from aetherflow.ir.verifier import verify
from aetherflow.passes import Autocast, PassManager
from aetherflow.train import ParameterStore, SGD, Trainer, mse

BACKEND = ReferenceBackend()


def _mlp():
    """``x @ w + b`` 再 softmax：最小的"该降 + 该留"链。"""
    g = Graph()
    x = g.input("x", f32[4, 3])
    w = g.input("w", f32[3, 5])
    b = g.input("b", f32[1, 5])
    h = g.add_node("matmul", [x, w])
    z = g.add(h, b)
    y = g.add_node("softmax", [z], {"axis": -1})
    g.output(y, "y")
    return g


def _feed(shape_x=(4, 3), shape_w=(3, 5)):
    rng = np.random.default_rng(0)
    return {
        "x": rng.normal(size=shape_x).astype(np.float32),
        "w": rng.normal(size=shape_w).astype(np.float32),
        "b": rng.normal(size=(1, shape_w[1])).astype(np.float32),
    }


def _by_op(graph, name):
    return [n for n in graph.nodes if n.op.name == name]


class TestAutocastRewrites(unittest.TestCase):
    def setUp(self):
        self.g = _mlp()
        self.out = Autocast().run(self.g)

    def test_matmul_inputs_become_f16(self):
        (mm,) = _by_op(self.out, "matmul")
        self.assertEqual(mm.output.type.dtype, f16)
        for arg in mm.inputs:
            self.assertEqual(arg.type.dtype, f16)
            self.assertEqual(arg.producer.op.name, "cast")

    def test_softmax_stays_f32(self):
        (sm,) = _by_op(self.out, "softmax")
        self.assertEqual(sm.output.type.dtype, f32)
        # 加法已经是 f32（它把 matmul 的 f16 结果提升回 f32），所以 softmax 不必再插 cast
        self.assertEqual(sm.inputs[0].type.dtype, f32)
        self.assertEqual(sm.inputs[0].producer.op.name, "add")

    def test_cast_inserted_where_f16_meets_f32_only_op(self):
        """matmul 直接喂 softmax：边界必须插一个 cast 把精度升回来。"""
        g = Graph()
        x = g.input("x", f32[4, 3])
        w = g.input("w", f32[3, 5])
        h = g.add_node("matmul", [x, w])
        g.output(g.add_node("softmax", [h], {"axis": -1}), "y")

        out = Autocast().run(g)
        (sm,) = _by_op(out, "softmax")
        self.assertEqual(sm.output.type.dtype, f32)
        self.assertEqual(sm.inputs[0].type.dtype, f32)
        self.assertEqual(sm.inputs[0].producer.op.name, "cast")
        self.assertEqual(sm.inputs[0].producer.inputs[0].type.dtype, f16)

    def test_add_promotes_to_widest_input(self):
        """``x @ w + b``：加法有 f32 输入 → 加法在 f32 里算，matmul 输出被提升。"""
        (add,) = [n for n in self.out.nodes if n.op.name == "add"]
        self.assertEqual(add.output.type.dtype, f32)
        produced = [a for a in add.inputs if a.producer is not None]
        self.assertEqual(len(produced), 1)
        lifted = produced[0]
        self.assertEqual(lifted.type.dtype, f32)
        self.assertEqual(lifted.producer.op.name, "cast")
        self.assertEqual(lifted.producer.inputs[0].type.dtype, f16)

    def test_graph_interface_is_unchanged(self):
        self.assertEqual([v.name for v in self.out.inputs], ["x", "w", "b"])
        for old, new in zip(self.g.inputs, self.out.inputs):
            self.assertEqual(old.type, new.type)
        self.assertEqual(self.out.outputs[0].type.dtype, f32)

    def test_verifier_accepts_result(self):
        self.assertEqual([i for i in verify(self.out) if i.level == "error"], [])

    def test_is_idempotent(self):
        again = Autocast().run(self.out)
        self.assertEqual([n.op.name for n in again.nodes], [n.op.name for n in self.out.nodes])

    def test_f64_is_left_alone(self):
        g = Graph()
        x = g.input("x", f64[4, 3])
        w = g.input("w", f64[3, 2])
        g.output(g.add_node("matmul", [x, w]), "y")
        out = Autocast().run(g)
        self.assertEqual([n.op.name for n in out.nodes], ["matmul"])
        self.assertEqual(out.outputs[0].type.dtype, f64)

    def test_integer_graph_is_left_alone(self):
        g = Graph()
        x = g.input("x", f32[4, 6])
        g.output(g.add_node("argmax", [x], {"axis": 1}), "y")
        out = Autocast().run(g)
        # argmax 不在两张表里：跟随输入，仍是 f32；输出仍是指标类型
        self.assertEqual(out.outputs[0].type, g.outputs[0].type)

    def test_integer_operands_are_not_cast(self):
        """embedding 的 ids 是下标，不是"被降精度的浮点操作数"。

        ``_work_dtype`` 回答的是"这个节点的**浮点**工作精度"；把它套到全部
        操作数上，i32 的 ids 会被 cast 成 f32，embedding 在构图期就拒绝。
        """
        g = Graph("emb")
        table = g.input("table", f32[6, 4])
        ids = g.input("ids", i32[2, 3])
        g.output(g.embedding(table, ids), "y")

        out = Autocast().run(g)
        (emb,) = _by_op(out, "embedding")
        self.assertEqual(emb.inputs[1].name, "ids")
        self.assertEqual(emb.inputs[1].type.dtype, i32)
        # embedding 的浮点工作在 f32，ids 不需要任何 cast
        self.assertEqual([n.op.name for n in out.nodes], ["embedding"])

        rng = np.random.default_rng(0)
        feed = {
            "table": rng.normal(size=(6, 4)).astype(np.float32),
            "ids": np.array([[0, 5, 2], [4, 1, 3]], dtype=np.int32),
        }
        np.testing.assert_allclose(
            BACKEND.execute(out, feed)["y"], BACKEND.execute(g, feed)["y"]
        )

    def test_empty_sets_are_not_defaults(self):
        """``low_precision=()`` 是"一个都不降"，``force_f32=()`` 是"一个都不强制"。

        两者若用 ``or`` 回落默认集，"显式放行全部算子"这个意思就表达不出来。
        """
        g = Graph("mm_softmax")
        x = g.input("x", f32[4, 8])
        w = g.input("w", f32[8, 8])
        h = g.add_node("matmul", [x, w], {})
        g.output(g.add_node("softmax", [h], {"axis": -1}), "y")

        default = Autocast(dtype="f16").run(g)
        (mm,) = _by_op(default, "matmul")
        (sm,) = _by_op(default, "softmax")
        self.assertEqual(mm.output.type.dtype, f16)
        self.assertEqual(sm.output.type.dtype, f32)

        no_force = Autocast(dtype="f16", force_f32=()).run(g)
        (mm2,) = _by_op(no_force, "matmul")
        (sm2,) = _by_op(no_force, "softmax")
        self.assertEqual(mm2.output.type.dtype, f16)
        self.assertEqual(sm2.output.type.dtype, f16)

        untouched = Autocast(dtype="f16", low_precision=()).run(g)
        self.assertEqual([n for n in untouched.nodes if n.op.name == "cast"], [])
        self.assertEqual(untouched.outputs[0].type.dtype, f32)


class TestAutocastNumerics(unittest.TestCase):
    def test_matches_f32_within_f16_tolerance(self):
        feed = _feed()
        ref = BACKEND.execute(_mlp(), feed)["y"]
        got = BACKEND.execute(Autocast().run(_mlp()), feed)["y"]
        self.assertEqual(got.dtype, np.float32)
        np.testing.assert_allclose(got, ref, rtol=1e-2, atol=1e-3)

    def test_parameter_gradient_returns_to_f32(self):
        """前向在 f16 里算，参数梯度必须回到 f32——否则主权重会被逐步降精度。"""
        g16 = Autocast().run(_mlp())
        bwd = backward(g16, wrt=[g16.find("w")])
        feed = _feed()
        feed["dy"] = np.ones((4, 5), np.float32)
        grads = BACKEND.execute(bwd, feed)
        self.assertEqual(grads["dw"].dtype, np.float32)

        # 与全 f32 图的梯度在 f16 容差内一致
        g32 = _mlp()
        bwd32 = backward(g32, wrt=[g32.find("w")])
        feed32 = _feed()
        feed32["dy"] = np.ones((4, 5), np.float32)
        ref = BACKEND.execute(bwd32, feed32)["dw"]
        np.testing.assert_allclose(grads["dw"], ref, rtol=5e-2, atol=1e-3)

    def test_pipeline_keeps_graph_valid(self):
        manager = PassManager([Autocast(), Autocast()])
        out = manager.run(_mlp())
        self.assertEqual(out.outputs[0].type.dtype, f32)


class TestCastKernelSource(unittest.TestCase):
    """``cast`` 的源码：读取跟着源 dtype，写入跟着目标 dtype。

    这条内核是唯一"两端存储宽度不同"的，cupy 的 ``astype`` 在 f16 上会链接不到
    ``__half`` 的构造函数，所以生产路径必须自己生成源码。断言落在源码文本上，
    这样不需要 GPU 也能盯住两端的 helper 有没有用反。
    """

    def test_f32_to_f16_rounds_on_store(self):
        src = cast_fn(np.dtype("float32"), np.dtype("float16"))
        self.assertIn("unsigned short* __restrict__ out", src)
        self.assertIn("float x = af_ld_f(in[i]);", src)
        self.assertIn("out[i] = af_st_h(x);", src)

    def test_f16_to_f32_reads_through_ptx(self):
        src = cast_fn(np.dtype("float16"), np.dtype("float32"))
        self.assertIn("float* __restrict__ out", src)
        self.assertIn("float x = af_ld_h(in[i]);", src)
        self.assertIn("out[i] = x;", src)

    def test_same_width_is_rejected(self):
        with self.assertRaises(ValueError):
            cast_fn(np.dtype("float32"), np.dtype("float32"))

    def test_hip_cast_uses_clang_helpers(self):
        """HIP 侧的 cast 复用同一份 L2 描述，但 f16 helper 必须是 clang 版。"""
        from aetherflow.backends.hip.codegen import cast_fn as hip_cast_fn

        src = hip_cast_fn(np.dtype("float32"), np.dtype("float16"))
        self.assertIn("_Float16", src)
        self.assertNotIn("cvt.f32.f16", src)
        self.assertIn("out[i] = af_st_h(x);", src)


class TestAutocastInTrainer(unittest.TestCase):
    """``Trainer(mixed_precision="f16")``：混精训练在参考后端上真的能学。"""

    def _setup(self, mixed=None, seed=0):
        g = Graph("lin")
        x = g.input("x", f32[8, 4])
        t = g.input("t", f32[8, 1])
        w = g.input("w", f32[4, 1])
        b = g.input("b", f32[1])
        g.output(mse(g, g.add(g.matmul(x, w), b), t), "loss")
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=seed)
        trainer = Trainer(
            g, store, SGD(store.arrays(), lr=0.1), mixed_precision=mixed
        )
        return trainer, store

    @staticmethod
    def _data(n=32, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, 4)).astype(np.float32)
        true_w = np.array([[1.0], [-2.0], [0.5], [3.0]], np.float32)
        return x, (x @ true_w + 0.7).astype(np.float32)

    def test_training_step_reduces_loss(self):
        trainer, _ = self._setup("f16")
        x, t = self._data()
        batch = {"x": x[:8], "t": t[:8]}
        first = trainer.step(batch)
        for _ in range(19):
            last = trainer.step(batch)
        self.assertLess(last, first)

    def test_parameters_and_gradients_stay_f32(self):
        trainer, store = self._setup("f16")
        x, t = self._data()
        trainer.step({"x": x[:8], "t": t[:8]})
        for name in store.names:
            self.assertEqual(store[name].dtype, np.float32)
        self.assertEqual(trainer.optimizer.params["w"].dtype, np.float32)

    def test_graph_really_uses_f16_for_matmul(self):
        trainer, _ = self._setup("f16")
        (mm,) = [n for n in trainer.graph.nodes if n.op.name == "matmul"]
        self.assertEqual(mm.inputs[0].type.dtype, f16)
        self.assertEqual(trainer.graph.outputs[0].type.dtype, f32)

    def test_none_keeps_the_original_graph(self):
        g = Graph("lin")
        x = g.input("x", f32[2, 2])
        w = g.input("w", f32[2, 1])
        g.output(mse(g, g.matmul(x, w), g.input("t", f32[2, 1])), "loss")
        store = ParameterStore({"w": np.ones((2, 1), np.float32)})
        trainer = Trainer(g, store, SGD(store.arrays(), lr=0.1))
        self.assertEqual([n.op.name for n in trainer.graph.nodes],
                         [n.op.name for n in g.nodes])

    def test_rejects_f64_target(self):
        with self.assertRaises(ValueError):
            self._setup("f64")


class TestAutocastLowPrecisionInputs(unittest.TestCase):
    """``low_precision_inputs``：参数以低精度驻留，图里不再每步 cast。"""

    def test_named_inputs_become_f16(self):
        out = Autocast(low_precision_inputs=("w", "b")).run(_mlp())
        dtypes = {v.name: v.type.dtype for v in out.inputs}
        self.assertEqual(dtypes["w"], f16)
        self.assertEqual(dtypes["b"], f16)
        self.assertEqual(dtypes["x"], f32)

    def test_unknown_input_is_rejected(self):
        with self.assertRaises(KeyError):
            Autocast(low_precision_inputs=("nope",)).run(_mlp())

    def test_f64_input_is_left_alone(self):
        g = Graph()
        x = g.input("x", f64[4, 3])
        w = g.input("w", f64[3, 2])
        g.output(g.add_node("matmul", [x, w]), "y")
        out = Autocast(low_precision_inputs=("w",)).run(g)
        self.assertEqual({v.name: v.type.dtype for v in out.inputs}["w"], f64)

    def test_is_idempotent(self):
        once = Autocast(low_precision_inputs=("w",)).run(_mlp())
        twice = Autocast(low_precision_inputs=("w",)).run(once)
        self.assertEqual([n.op.name for n in once.nodes], [n.op.name for n in twice.nodes])
        self.assertEqual([v.type for v in once.inputs], [v.type for v in twice.inputs])


class TestAutocastRejects(unittest.TestCase):
    def test_non_float_target(self):
        with self.assertRaises(ValueError):
            Autocast(dtype="int32")

    def test_f64_target(self):
        with self.assertRaises(ValueError):
            Autocast(dtype="f64")


@unittest.skipUnless(CUDA_AVAILABLE, "需要 CUDA 运行时")
class TestAutocastOnCuda(unittest.TestCase):
    """图重写之后必须真的能在设备上跑：f16 的 cast / matmul / softmax 都有人接。"""

    @classmethod
    def setUpClass(cls):
        cls.backend = CudaBackend()

    def test_forward_matches_f32(self):
        feed = _feed()
        ref = self.backend.execute(_mlp(), feed)["y"]
        got = self.backend.execute(Autocast().run(_mlp()), feed)["y"]
        self.assertEqual(got.dtype, np.float32)
        np.testing.assert_allclose(got, ref, rtol=1e-2, atol=1e-3)

    def test_backward_returns_f32_gradients(self):
        g16 = Autocast().run(_mlp())
        bwd = backward(g16, wrt=[g16.find("w")])
        feed = _feed()
        feed["dy"] = np.ones((4, 5), np.float32)
        grads = self.backend.execute(bwd, feed)
        self.assertEqual(grads["dw"].dtype, np.float32)
        self.assertTrue(np.isfinite(grads["dw"]).all())

    def test_trainer_steps_with_mixed_precision(self):
        """``Trainer(mixed_precision="f16")`` 在设备上跑得起来，步时才有意义。"""
        g = Graph("lin")
        x = g.input("x", f32[8, 4])
        t = g.input("t", f32[8, 1])
        w = g.input("w", f32[4, 1])
        b = g.input("b", f32[1])
        g.output(mse(g, g.add(g.matmul(x, w), b), t), "loss")
        store = ParameterStore.from_graph(g, exclude=("x", "t"), seed=0)
        trainer = Trainer(
            g, store, SGD(store.arrays(), lr=0.1),
            backend=self.backend, mixed_precision="f16",
        )
        rng = np.random.default_rng(0)
        xv = rng.normal(size=(32, 4)).astype(np.float32)
        tv = (xv @ np.array([[1.0], [-2.0], [0.5], [3.0]], np.float32) + 0.7).astype(np.float32)
        batch = {"x": xv[:8], "t": tv[:8]}
        first = trainer.step(batch)
        losses = [trainer.step(batch) for _ in range(19)]
        self.assertLess(losses[-1], first)
        self.assertEqual(store["w"].dtype, np.float32)


class TestCastGradient(unittest.TestCase):
    """``cast`` 的 VJP：已是输入类型时不建恒等节点，跨宽度时仍要转回去。

    反向图里会把路径上的前向 ``cast`` 一并复制进来，所以"新增了几个"要看
    有没有第二个节点，而不是数总数。
    """

    @staticmethod
    def _backward(target):
        g = Graph("c")
        x = g.input("x", f32[2, 3])
        g.output(g.add_node("cast", [x], {"dtype": target}), "y")
        bwd = backward(g, wrt=[x])
        return [n for n in bwd.nodes if n.op.name == "cast"]

    def test_same_dtype_adds_nothing(self):
        casts = self._backward("f32")
        self.assertEqual(len(casts), 1)
        self.assertEqual(casts[0].inputs[0].type.dtype, f32)

    def test_cross_dtype_adds_one_conversion(self):
        casts = self._backward("f16")
        extra = [n for n in casts if n.inputs[0].type.dtype == f16]
        self.assertEqual(len(extra), 1)
        self.assertEqual(extra[0].attrs["dtype"], "f32")
