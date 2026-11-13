"""趋势采集的本地用例：用例筛选、元素数口径、预热不入账。

重点不是"数字准不准"（参考后端的绝对耗时没有意义），而是三条结构约定：

1. 预热执行不产生记录，否则趋势的第一点会被编译/首次分配抬高；
2. 记录带上用例名，同名图（不同规模的 matmul）不会混在一起；
3. 符号维算不出元素数时不报吞吐，而不是报一个错的数。
"""

import unittest

import numpy as np

from aetherflow import Graph, ReferenceExecutor, f32
from benchmarks.cases import Case
from benchmarks.trend import _select, _units, collect


def _build(shape=(2, 3)):
    def build(rng):
        graph = Graph("tiny")
        x = graph.input("x", f32[shape])
        graph.output(graph.relu(x), "y")
        return graph, {"x": np.ones(shape, dtype=np.float32)}

    return build


class TestSelect(unittest.TestCase):
    def setUp(self):
        self.cases = [
            Case("tiny", _build(), 0, 1, ""),
            Case("other", _build((1, 2)), 0, 1, ""),
        ]

    def test_empty_selection_keeps_everything(self):
        self.assertEqual([case.name for case in _select(self.cases, [])], ["tiny", "other"])

    def test_filter_by_name_preserves_declaration_order(self):
        self.assertEqual([case.name for case in _select(self.cases, ["other"])], ["other"])

    def test_unknown_name_is_fatal(self):
        with self.assertRaises(SystemExit):
            _select(self.cases, ["missing"])


class TestUnits(unittest.TestCase):
    def test_static_output_counts_elements(self):
        graph, _ = _build((2, 3))(None)
        self.assertEqual(_units(graph), {"elements": 6})

    def test_symbolic_output_has_no_throughput(self):
        graph = Graph("dyn")
        x = graph.input("x", f32["B", 3])
        graph.output(graph.relu(x), "y")
        self.assertIsNone(_units(graph))


class TestCollect(unittest.TestCase):
    def test_preheat_is_not_recorded(self):
        case = Case("tiny", _build(), 0, 1, "")
        recorder = collect(ReferenceExecutor(), [case], rounds=3)
        self.assertEqual(len(recorder.records), 3)
        self.assertEqual(recorder.last.label, "tiny")
        self.assertEqual(recorder.last.graph, "tiny")
        self.assertEqual(recorder.last.units, {"elements": 6})
        trend = recorder.trend()
        self.assertEqual(trend["runs"], 3)
        self.assertEqual(trend["graphs"]["tiny"]["count"], 3)


if __name__ == "__main__":
    unittest.main()
