"""循环级 IR 的判据：不变计算外提、数值等价、序列化与能力边界。

这些用例只关心三件事：

1. **外提不改变数值**——preheader 只跑一次，循环体每轮跑，结果必须与
   原 ``WhileLoop`` 逐位一致；
2. **只搬纯函数**——依赖 carried 的节点、带 RNG / STATE / IO 的节点必须
   留在体内；
3. **边界显式**——嵌套控制区域进不了循环级 IR 时抛 ``LoopIRError``，
   不静默降级成"每轮重算"或"只算一次"。
"""

import json
import unittest

import numpy as np

from aetherflow import (
    Conditional,
    Graph,
    GraphRegion,
    LoopIR,
    LoopIRError,
    LoopSplit,
    ReferenceBackend,
    WhileLoop,
    f32,
    loop_from_spec,
    loop_to_spec,
    optimize_loop,
)


def _scalar(dtype=f32):
    return dtype[()]


def _flag_graph(name="pred"):
    g = Graph(name)
    x = g.input("x", _scalar())
    g.output(g.step(x, threshold=0.0), "flag")
    return g


def _cycle_graphs():
    """条件 ``s < 3``；体里 ``bias`` 只依赖自由输入 ``x``，每轮重算但不变。"""
    cond = Graph("loop_cond")
    s = cond.input("s", _scalar())
    cond.output(cond.step(cond.sub(cond.constant(3.0), s), threshold=0.0), "flag")

    body = Graph("loop_body")
    s_in = body.input("s", _scalar())
    acc = body.input("acc", _scalar())
    step = body.input("step", _scalar())
    x = body.input("x", _scalar())
    scale = body.mul(x, body.constant(2.0))
    bias = body.add(scale, body.constant(1.0))
    nxt = body.add(s_in, step)
    y = body.add(nxt, bias)
    acc_next = body.add(acc, y)
    body.output(nxt, "s_next")
    body.output(acc_next, "acc_next")
    body.output(y, "y")
    return cond, body


def _cycle_loop(max_steps=8):
    cond, body = _cycle_graphs()
    return WhileLoop(
        cond,
        body,
        carried={"s": "s_next", "acc": "acc_next"},
        max_steps=max_steps,
        name="cycle",
    )


def _feeds():
    return {
        "s": np.float32(0.0),
        "acc": np.float32(0.0),
        "step": np.float32(1.0),
        "x": np.float32(2.0),
    }


def _nested_loop():
    """合法的 WhileLoop，但体是 Conditional：循环级 IR 应显式拒绝。"""

    def branch(name, offset):
        g = Graph(name)
        x = g.input("x", _scalar())
        g.output(g.add(x, g.constant(offset)), "y")
        return g

    return WhileLoop(
        _flag_graph("cond"),
        Conditional(_flag_graph("cond"), branch("then", 1.0), branch("else", 2.0)),
        carried={"x": "y"},
        max_steps=2,
    )


class TestLoopIR(unittest.TestCase):
    def test_extracts_the_single_graph_form(self):
        loop = LoopIR.from_control(_cycle_loop())
        self.assertEqual(loop.name, "cycle")
        self.assertEqual(loop.carried, {"s": "s_next", "acc": "acc_next"})
        self.assertEqual(loop.max_steps, 8)
        self.assertEqual([name for name, _ in loop.outputs], ["s", "acc"])

    def test_round_trips_to_while_loop(self):
        loop = LoopIR.from_control(_cycle_loop())
        control = loop.to_control()
        self.assertIsInstance(control, WhileLoop)
        np.testing.assert_array_equal(
            control.run(_feeds())["s"], np.float32(3.0)
        )

    def test_nested_region_is_refused(self):
        with self.assertRaises(LoopIRError):
            LoopIR.from_control(_nested_loop())

    def test_condition_must_be_a_graph(self):
        _, body = _cycle_graphs()
        with self.assertRaises(TypeError):
            LoopIR(
                GraphRegion(_flag_graph()),
                body,
                carried={"s": "s_next"},
                max_steps=2,
            )

    def test_body_must_be_a_graph(self):
        cond, _ = _cycle_graphs()
        with self.assertRaises(TypeError):
            LoopIR(
                cond,
                GraphRegion(_flag_graph()),
                carried={"s": "s_next"},
                max_steps=2,
            )

    def test_contracts_are_reused_from_while_loop(self):
        cond, body = _cycle_graphs()
        with self.assertRaises(ValueError):
            LoopIR(cond, body, carried={}, max_steps=2)
        with self.assertRaises(ValueError):
            LoopIR(cond, body, carried={"s": "s_next"}, max_steps=0)

    def test_invariant_nodes_are_the_state_independent_pure_nodes(self):
        loop = LoopIR.from_control(_cycle_loop())
        names = [node.output.name for node in loop.invariant_nodes()]
        self.assertEqual(names, ["v5", "v7"])
        self.assertEqual(len(loop.body.nodes), 7)

    def test_state_dependent_nodes_are_not_invariant(self):
        loop = LoopIR.from_control(_cycle_loop())
        names = {node.output.name for node in loop.invariant_nodes()}
        self.assertEqual(names & {"v8", "v9", "v10"}, set())

    def test_impure_nodes_are_not_invariant(self):
        cond = Graph("cond")
        s = cond.input("s", _scalar())
        cond.output(cond.step(s, threshold=0.5), "flag")

        body = Graph("body")
        s_in = body.input("s", _scalar())
        x = body.input("x", _scalar())
        mask = body.dropout_mask(x, p=0.5, seed=1, offset=0)
        scaled = body.mul(mask, x)
        nxt = body.add(s_in, body.constant(1.0))
        body.output(nxt, "s_next")
        body.output(scaled, "masked")
        loop = LoopIR(cond, body, carried={"s": "s_next"}, max_steps=4)
        names = [node.output.name for node in loop.invariant_nodes()]
        self.assertEqual(names, [])

    def test_constant_only_chain_can_be_hoisted(self):
        cond = Graph("cond")
        s = cond.input("s", _scalar())
        cond.output(cond.step(s, threshold=0.5), "flag")

        body = Graph("body")
        s_in = body.input("s", _scalar())
        offset = body.add(body.constant(1.0), body.constant(2.0))
        nxt = body.add(s_in, offset)
        body.output(nxt, "s_next")
        loop = LoopIR(cond, body, carried={"s": "s_next"}, max_steps=4)
        self.assertEqual(
            [node.output.name for node in loop.invariant_nodes()], ["v3"]
        )


class TestSplitInvariants(unittest.TestCase):
    def test_moves_invariant_nodes_into_the_preheader(self):
        split = LoopIR.from_control(_cycle_loop()).split_invariants()
        self.assertIsInstance(split, LoopSplit)
        self.assertEqual(split.hoisted_nodes, 4)
        self.assertEqual(len(split.hoisted), 2)
        self.assertEqual(len(split.loop.body.nodes), 3)

    def test_new_body_takes_hoisted_values_as_inputs(self):
        split = LoopIR.from_control(_cycle_loop()).split_invariants()
        inputs = {value.name for value in split.loop.body.inputs}
        self.assertTrue(set(split.hoisted) <= inputs)
        self.assertEqual({name for name, _ in split.loop.inputs}, {
            "step", "x", *split.hoisted
        })

    def test_condition_graph_is_shared_not_rewritten(self):
        loop = LoopIR.from_control(_cycle_loop())
        split = loop.split_invariants()
        self.assertIs(split.loop.condition, loop.condition)

    def test_carried_mapping_is_unchanged(self):
        loop = LoopIR.from_control(_cycle_loop())
        split = loop.split_invariants()
        self.assertEqual(split.loop.carried, loop.carried)

    def test_numerics_match_the_unoptimized_loop(self):
        loop = LoopIR.from_control(_cycle_loop())
        split = loop.split_invariants()
        before = loop.to_control().run(_feeds())
        after = split.run(_feeds())
        np.testing.assert_array_equal(before["s"], after["s"])
        np.testing.assert_array_equal(before["acc"], after["acc"])
        self.assertEqual(after["acc"], np.float32(21.0))

    def test_split_run_is_repeatable(self):
        split = LoopIR.from_control(_cycle_loop()).split_invariants()
        first = split.run(_feeds())
        second = split.run(_feeds())
        np.testing.assert_array_equal(first["acc"], second["acc"])

    def test_extra_body_outputs_still_see_the_hoisted_value(self):
        """``y`` 不是 carried，但它必须拿到与未外提时相同的 bias。"""
        loop = LoopIR.from_control(_cycle_loop())
        split = loop.split_invariants()
        backend = ReferenceBackend()
        scope = dict(_feeds())
        produced = backend.execute(
            split.preheader,
            {value.name: scope[value.name] for value in split.preheader.inputs},
        )
        scope.update(produced)
        got = backend.execute(split.loop.body, scope)
        want = backend.execute(loop.body, _feeds())
        np.testing.assert_array_equal(got["s_next"], want["s_next"])
        np.testing.assert_array_equal(got["y"], want["y"])

    def test_returns_self_when_nothing_can_be_hoisted(self):
        cond = Graph("cond")
        s = cond.input("s", _scalar())
        cond.output(cond.step(s, threshold=0.5), "flag")

        body = Graph("body")
        s_in = body.input("s", _scalar())
        nxt = body.add(s_in, body.constant(1.0))
        body.output(nxt, "s_next")
        loop = LoopIR(cond, body, carried={"s": "s_next"}, max_steps=4)
        split = loop.split_invariants()
        self.assertIsNone(split.preheader)
        self.assertEqual(split.hoisted, ())
        self.assertIs(split.loop, loop)
        self.assertEqual(split.hoisted_nodes, 0)

    def test_rewritten_graphs_pass_the_verifier(self):
        from aetherflow import verify

        split = LoopIR.from_control(_cycle_loop()).split_invariants()
        for graph in (split.preheader, split.loop.body):
            errors = [issue for issue in verify(graph) if issue.level == "error"]
            self.assertEqual(errors, [])

    def test_optimize_loop_is_the_one_step_entry(self):
        split = optimize_loop(_cycle_loop())
        self.assertIsInstance(split, LoopSplit)
        np.testing.assert_array_equal(split.run(_feeds())["acc"], np.float32(21.0))

    def test_optimize_loop_refuses_nested_regions(self):
        with self.assertRaises(LoopIRError):
            optimize_loop(_nested_loop())

    def test_missing_preheader_input_is_an_explicit_key_error(self):
        split = LoopIR.from_control(_cycle_loop()).split_invariants()
        with self.assertRaises(KeyError):
            split.run({"s": np.float32(0.0), "acc": np.float32(0.0),
                       "step": np.float32(1.0)})

    def test_missing_carried_initial_value_is_an_explicit_value_error(self):
        split = LoopIR.from_control(_cycle_loop()).split_invariants()
        with self.assertRaises(ValueError):
            split.run({"step": np.float32(1.0), "x": np.float32(2.0)})


class TestLoopSerialization(unittest.TestCase):
    def test_spec_is_pure_json(self):
        loop = LoopIR.from_control(_cycle_loop())
        spec = loop_to_spec(loop)
        json.dumps(spec)
        self.assertEqual(spec["kind"], "loop")
        self.assertEqual(spec["carried"], {"s": "s_next", "acc": "acc_next"})

    def test_round_trip_executes_identically(self):
        loop = LoopIR.from_control(_cycle_loop())
        restored = loop_from_spec(loop_to_spec(loop))
        np.testing.assert_array_equal(
            restored.to_control().run(_feeds())["acc"],
            loop.to_control().run(_feeds())["acc"],
        )

    def test_round_trip_keeps_hoisting_decisions(self):
        split = LoopIR.from_control(_cycle_loop()).split_invariants()
        restored = loop_from_spec(loop_to_spec(split.loop))
        self.assertEqual(
            [node.output.name for node in restored.invariant_nodes()], []
        )

    def test_wrong_kind_is_rejected(self):
        loop = LoopIR.from_control(_cycle_loop())
        spec = loop_to_spec(loop)
        spec["kind"] = "while"
        with self.assertRaises(ValueError):
            loop_from_spec(spec)

    def test_missing_fields_are_rejected(self):
        loop = LoopIR.from_control(_cycle_loop())
        spec = loop_to_spec(loop)
        del spec["body"]
        with self.assertRaises(KeyError):
            loop_from_spec(spec)

    def test_non_mapping_is_rejected(self):
        with self.assertRaises(TypeError):
            loop_from_spec([1, 2, 3])

    def test_non_loop_input_is_rejected(self):
        with self.assertRaises(TypeError):
            loop_to_spec(_cycle_loop())


if __name__ == "__main__":
    unittest.main()
