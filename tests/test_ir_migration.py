"""IR 跨版本迁移：注册表、路径、结构改写与反序列化集成。

要钉住四件事：

1. 路径唯一且只升不降——每个源版本最多一条出边，缺哪一段就报哪一段。
2. 迁移作用于**序列化结构**，不就地改调用方的数据。
3. 返回类型不对时显式失败，不让"没改"和"改错"混在一起。
4. ``from_dict`` 真的会走迁移，并记下来源版本（``Graph.migrated_from``）。
"""

import unittest
from unittest.mock import patch

from aetherflow import (
    Graph,
    IR_VERSION,
    IRVersionError,
    f32,
    from_dict,
    migrate,
    migration_path,
    register_migration,
    registered_migrations,
    to_dict,
    unregister_migration,
)
from aetherflow.ir import version as version_module


def _add_eps(data):
    """1.0 → 1.1：给 layernorm 补上在新版本里才成为必填的 eps。"""
    for node in data["nodes"]:
        if node["op"] == "layernorm":
            node.setdefault("attrs", {})["eps"] = 1e-5
    return data


def _bump_eps(data):
    """1.1 → 1.2：改掉默认值，用来验证两步按顺序执行。"""
    for node in data["nodes"]:
        if node["op"] == "layernorm":
            node["attrs"]["eps"] = 1e-6
    return data


def _layernorm_graph():
    g = Graph("ln")
    x = g.input("x", f32[2, 8])
    gamma = g.input("gamma", f32[8])
    beta = g.input("beta", f32[8])
    g.output(g.add_node("layernorm", [x, gamma, beta], {"axes": [-1]}), "y")
    return g


class _Registered:
    """记录本测试注册过的源版本，tearDown 时反向撤销。"""

    def __init__(self):
        self.sources = []

    def add(self, source, target, fn):
        register_migration(source, target, fn)
        self.sources.append(tuple(source))

    def cleanup(self):
        for source in reversed(self.sources):
            if source in dict(registered_migrations()):
                unregister_migration(source)
        self.sources.clear()


class TestMigrationPath(unittest.TestCase):
    def setUp(self):
        self.registered = _Registered()

    def tearDown(self):
        self.registered.cleanup()

    def test_same_version_is_empty(self):
        self.assertEqual(migration_path(IR_VERSION), ())
        self.assertEqual(migration_path((1, 0), (1, 0)), ())

    def test_missing_edge_names_the_gap(self):
        with self.assertRaises(IRVersionError) as ctx:
            migration_path((1, 0), (1, 3))
        message = str(ctx.exception)
        self.assertIn("缺 1.0 的出边", message)
        self.assertIn("无", message)

    def test_downgrade_is_rejected(self):
        with self.assertRaises(IRVersionError):
            migration_path((1, 2), (1, 1))

    def test_chain_is_walked_in_order(self):
        self.registered.add((1, 0), (1, 1), _add_eps)
        self.registered.add((1, 1), (1, 2), _bump_eps)
        self.assertEqual(
            migration_path((1, 0), (1, 2)), ((1, 0), (1, 1))
        )

    def test_path_cannot_overshoot_target(self):
        self.registered.add((1, 0), (1, 2), _add_eps)
        with self.assertRaises(IRVersionError) as ctx:
            migration_path((1, 0), (1, 1))
        self.assertIn("越过了请求的 1.1", str(ctx.exception))


class TestRegisterValidation(unittest.TestCase):
    def tearDown(self):
        for source in [s for s in dict(registered_migrations())]:
            unregister_migration(source)

    def test_target_must_be_higher(self):
        with self.assertRaises(ValueError):
            register_migration((1, 1), (1, 1), _add_eps)
        with self.assertRaises(ValueError):
            register_migration((1, 2), (1, 1), _add_eps)

    def test_migration_must_be_callable(self):
        with self.assertRaises(TypeError):
            register_migration((1, 0), (1, 1), "not-callable")

    def test_duplicate_source_is_rejected(self):
        register_migration((1, 0), (1, 1), _add_eps)
        with self.assertRaises(ValueError) as ctx:
            register_migration((1, 0), (1, 2), _bump_eps)
        self.assertIn("每个源版本只允许一条出边", str(ctx.exception))

    def test_bad_version_text(self):
        with self.assertRaises(IRVersionError):
            register_migration((1,), (1, 1), _add_eps)


class TestMigrateStructure(unittest.TestCase):
    def setUp(self):
        self.registered = _Registered()
        self.registered.add((1, 0), (1, 1), _add_eps)
        self.registered.add((1, 1), (1, 2), _bump_eps)

    def tearDown(self):
        self.registered.cleanup()

    def test_applies_whole_chain(self):
        data = to_dict(_layernorm_graph())
        out = migrate(data, target=(1, 2))
        self.assertEqual(out["ir_version"], [1, 2])
        self.assertEqual(out["nodes"][0]["attrs"]["eps"], 1e-6)

    def test_input_is_not_modified(self):
        data = to_dict(_layernorm_graph())
        before = repr(data)
        migrate(data, target=(1, 2))
        self.assertEqual(repr(data), before)

    def test_same_version_returns_input_unchanged(self):
        data = to_dict(_layernorm_graph())
        self.assertIs(migrate(data), data)

    def test_bad_return_type_is_rejected(self):
        self.registered.add((2, 0), (2, 1), lambda _: None)
        data = {"ir_version": [2, 0], "name": "x", "inputs": [], "nodes": [], "outputs": []}
        with patch.object(version_module, "IR_VERSION", (2, 1)):
            with self.assertRaises(TypeError):
                migrate(data)

    def test_non_dict_input_is_rejected(self):
        with self.assertRaises(TypeError):
            migrate("{}")


class TestFromDictIntegration(unittest.TestCase):
    def setUp(self):
        self.registered = _Registered()

    def tearDown(self):
        self.registered.cleanup()

    def test_older_graph_is_migrated_and_recorded(self):
        self.registered.add((1, 0), (1, 1), _add_eps)
        source = to_dict(_layernorm_graph())

        with patch.object(version_module, "IR_VERSION", (1, 1)):
            graph = from_dict(source)

        self.assertEqual(graph.ir_version, (1, 1))
        self.assertEqual(graph.migrated_from, (1, 0))
        self.assertEqual(graph.nodes[0].attrs["eps"], 1e-5)
        # 迁移产物仍是可重建的图：输入、输出与节点都在原位。
        self.assertEqual([v.name for v in graph.inputs], ["x", "gamma", "beta"])
        self.assertEqual(graph.output_names, ["y"])

    def test_current_graph_has_no_migration_marker(self):
        graph = from_dict(to_dict(_layernorm_graph()))
        self.assertEqual(graph.ir_version, IR_VERSION)
        self.assertIsNone(graph.migrated_from)

    def test_missing_migration_blocks_load(self):
        source = to_dict(_layernorm_graph())
        with patch.object(version_module, "IR_VERSION", (1, 1)):
            with self.assertRaises(IRVersionError) as ctx:
                from_dict(source)
        self.assertIn("缺 1.0 的出边", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
