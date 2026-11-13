"""部署包的验收测试。

判据：

1. **往返**：图、权重、meta 读回来与写出前逐位 / 逐字段相等；
2. **自校验**：内容被改过就必须在加载期失败，而不是跑起来才算错；
3. **跨进程**：另起一个解释器只靠包本身跑一次前向，输出与进程内逐位一致。
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np

from aetherflow import (
    BundleError,
    Graph,
    ReferenceBackend,
    f32,
    load_bundle,
    save_bundle,
    to_dict,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = ReferenceBackend()


def _graph():
    g = Graph("deploy_mlp")
    x = g.input("x", f32[4, 8])
    w1 = g.input("w1", f32[8, 16])
    w2 = g.input("w2", f32[16, 2])
    g.output(g.matmul(g.relu(g.matmul(x, w1)), w2), "y")
    return g


def _weights():
    rng = np.random.default_rng(21)
    return {
        "w1": rng.standard_normal((8, 16)).astype(np.float32),
        "w2": rng.standard_normal((16, 2)).astype(np.float32),
    }


def _members(path):
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _rewrite(path, members):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)


class TestDeployBundle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.graph = _graph()
        self.weights = _weights()
        self.path = os.path.join(self.tmp.name, "model.afpkg")

    def test_roundtrip_preserves_graph_weights_and_meta(self):
        save_bundle(self.path, self.graph, self.weights, meta={"accuracy": 0.97})
        loaded = load_bundle(self.path)
        self.assertEqual(to_dict(loaded.graph), to_dict(self.graph))
        self.assertEqual(loaded.meta, {"accuracy": 0.97})
        for name, array in self.weights.items():
            np.testing.assert_array_equal(loaded.weights[name], array)

    def test_run_matches_the_in_process_execution(self):
        save_bundle(self.path, self.graph, self.weights)
        x = np.random.default_rng(3).standard_normal((4, 8)).astype(np.float32)
        expected = BACKEND.execute(self.graph, {**self.weights, "x": x})["y"]
        actual = load_bundle(self.path).run({"x": x})
        np.testing.assert_array_equal(actual["y"], expected)

    def test_tampered_graph_is_rejected(self):
        save_bundle(self.path, self.graph, self.weights)
        members = _members(self.path)
        members["graph.json"] = members["graph.json"].replace(
            b"deploy_mlp", b"tampered"
        )
        _rewrite(self.path, members)
        with self.assertRaises(BundleError) as caught:
            load_bundle(self.path)
        self.assertIn("校验和", str(caught.exception))

    def test_tampered_weight_is_rejected(self):
        save_bundle(self.path, self.graph, self.weights)
        members = _members(self.path)
        manifest = json.loads(members["manifest.json"].decode("utf-8"))
        # 只改清单里声明的摘要：数组没变，摘要对不上
        manifest["weights"]["w1"]["sha256"] = "0" * 64
        members["manifest.json"] = json.dumps(manifest).encode("utf-8")
        _rewrite(self.path, members)
        with self.assertRaises(BundleError) as caught:
            load_bundle(self.path)
        self.assertIn("w1", str(caught.exception))

    def test_extra_member_is_rejected(self):
        save_bundle(self.path, self.graph, self.weights)
        members = _members(self.path)
        members["sitecustomize.py"] = b"raise SystemExit(1)\n"
        _rewrite(self.path, members)
        with self.assertRaises(BundleError):
            load_bundle(self.path)

    def test_unknown_format_version_is_rejected(self):
        save_bundle(self.path, self.graph, self.weights)
        members = _members(self.path)
        manifest = json.loads(members["manifest.json"].decode("utf-8"))
        manifest["version"] = 99
        members["manifest.json"] = json.dumps(manifest).encode("utf-8")
        _rewrite(self.path, members)
        with self.assertRaises(BundleError):
            load_bundle(self.path)

    def test_weights_must_match_graph_inputs(self):
        with self.assertRaises(BundleError):
            save_bundle(self.path, self.graph, {"nope": np.zeros(3, dtype=np.float32)})
        with self.assertRaises(BundleError):
            save_bundle(
                self.path,
                self.graph,
                {**self.weights, "w1": np.zeros((16, 8), dtype=np.float32)},
            )

    def test_suffix_is_enforced(self):
        with self.assertRaises(ValueError):
            save_bundle(os.path.join(self.tmp.name, "model.zip"), self.graph, self.weights)

    def test_new_process_produces_the_same_bits(self):
        save_bundle(self.path, self.graph, self.weights)
        x = np.random.default_rng(11).standard_normal((4, 8)).astype(np.float32)
        feed = os.path.join(self.tmp.name, "x.npy")
        np.save(feed, x)
        expected = BACKEND.execute(self.graph, {**self.weights, "x": x})["y"]
        want = hashlib.sha256(np.ascontiguousarray(expected).tobytes()).hexdigest()

        code = (
            "import hashlib,sys;"
            "import numpy as np;"
            "from aetherflow import load_bundle;"
            "b=load_bundle(sys.argv[1]);"
            "x=np.load(sys.argv[2]);"
            "y=b.run({'x': x})['y'];"
            "print(hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest())"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = _ROOT + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-c", code, self.path, feed],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), want)


if __name__ == "__main__":
    unittest.main()
