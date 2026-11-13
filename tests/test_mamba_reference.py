"""AF-25：官方 Mamba-1 扫描参考实现的对拍与参数映射规则。

对拍的意义不是"证明我们等于官方"——三处结构差异（逐通道 dt、读出后
RMSNorm、门控位置）仍然存在；它只证明扫描公式本身一致，包括
``b = Δ·B·x`` 这一形式。参数映射的测试则相反：能一一对应的只有 A 与 D，
整块迁移必须被拒绝。
"""

import unittest

import numpy as np

from aetherflow import Graph, ReferenceBackend, f32
from aetherflow.models.mamba import MambaSpec
from aetherflow.models.mamba_reference import (
    from_official_checkpoint,
    map_official_params,
    official_selective_scan,
)
from aetherflow.models.ssm import build_selective_scan_stateful

T, BATCH, DIMS, STATES = 4, 2, 3, 2


def _feed(with_h0=False, with_reset=False):
    rng = np.random.default_rng(7)
    feed = {
        "x": rng.standard_normal((T, BATCH, DIMS)).astype(np.float32),
        "dt": (np.abs(rng.standard_normal((T, BATCH, DIMS))) + 0.1).astype(np.float32),
        "A": (-np.abs(rng.standard_normal((DIMS, STATES))) - 0.1).astype(np.float32),
        "Bt": rng.standard_normal((T, BATCH, STATES)).astype(np.float32),
        "Ct": rng.standard_normal((T, BATCH, STATES)).astype(np.float32),
        "D": rng.standard_normal((DIMS,)).astype(np.float32),
    }
    if with_h0:
        feed["h0"] = rng.standard_normal((BATCH, DIMS, STATES)).astype(np.float32)
    if with_reset:
        mask = np.zeros((T, BATCH), np.float32)
        mask[2, 0] = 1.0
        feed["reset"] = mask
    return feed


def _build(with_h0=False, with_reset=False):
    g = Graph("scan_ref")
    x = g.input("x", f32[(T, BATCH, DIMS)])
    dt = g.input("dt", f32[(T, BATCH, DIMS)])
    A = g.input("A", f32[(DIMS, STATES)])
    Bt = g.input("Bt", f32[(T, BATCH, STATES)])
    Ct = g.input("Ct", f32[(T, BATCH, STATES)])
    Dskip = g.input("D", f32[(DIMS,)])
    h0 = g.input("h0", f32[(BATCH, DIMS, STATES)]) if with_h0 else None
    reset = g.input("reset", f32[(T, BATCH)]) if with_reset else None
    y, h = build_selective_scan_stateful(
        g, x, dt, A, Bt, Ct, Dskip, h0=h0, reset=reset
    )
    g.output(y, "y")
    g.output(h, "h")
    return g


def _compare(feed, with_h0, with_reset):
    got = ReferenceBackend().execute(_build(with_h0, with_reset), feed)
    ref_y, ref_h = official_selective_scan(
        feed["x"].transpose(1, 2, 0),
        feed["dt"].transpose(1, 2, 0),
        feed["A"],
        feed["Bt"].transpose(1, 0, 2),
        feed["Ct"].transpose(1, 0, 2),
        D=feed["D"],
        h0=feed.get("h0"),
        reset=None if not with_reset else feed["reset"].T.astype(bool),
    )
    np.testing.assert_allclose(
        got["y"], ref_y.transpose(2, 0, 1), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(got["h"], ref_h, rtol=1e-5, atol=1e-6)


class TestSelectiveScanReference(unittest.TestCase):
    def test_matches_framework_without_state(self):
        _compare(_feed(), with_h0=False, with_reset=False)

    def test_matches_framework_with_state_and_reset(self):
        _compare(_feed(with_h0=True, with_reset=True), with_h0=True, with_reset=True)

    def test_gate_and_skip_are_applied_once(self):
        rng = np.random.default_rng(3)
        u = rng.standard_normal((BATCH, DIMS, T))
        delta = np.abs(rng.standard_normal((BATCH, DIMS, T))) + 0.1
        A = -np.abs(rng.standard_normal((DIMS, STATES))) - 0.1
        B = rng.standard_normal((BATCH, T, STATES))
        C = rng.standard_normal((BATCH, T, STATES))
        D = rng.standard_normal((DIMS,))
        z = rng.standard_normal((BATCH, DIMS, T))

        plain, _ = official_selective_scan(u, delta, A, B, C, D=D)
        gated, _ = official_selective_scan(u, delta, A, B, C, D=D, z=z)
        expected = plain * (0.5 * z * (1.0 + np.tanh(0.5 * z)))
        np.testing.assert_allclose(gated, expected, rtol=1e-12, atol=1e-12)

    def test_rejects_wrong_shapes(self):
        u = np.zeros((BATCH, DIMS, T))
        delta = np.zeros((BATCH, DIMS, T))
        A = np.zeros((DIMS, STATES))
        B = np.zeros((BATCH, T, STATES))
        C = np.zeros((BATCH, T, STATES))
        with self.assertRaises(ValueError):
            official_selective_scan(u, delta, A, B, C, D=np.zeros((DIMS + 1,)))
        with self.assertRaises(ValueError):
            official_selective_scan(u, delta, A, B[:, :, :1], C)
        with self.assertRaises(ValueError):
            official_selective_scan(u, delta, A, B, C, z=np.zeros((1, 1, 1)))


class TestParamMapping(unittest.TestCase):
    def _spec(self):
        return MambaSpec(d_model=4, d_state=2, d_conv=3, expand=2, steps=1, batch=1)

    def test_only_A_and_D_are_equivalent(self):
        spec = self._spec()
        official = {
            "A_log": np.zeros((spec.d_inner, spec.d_state), np.float32),
            "D": np.ones((spec.d_inner,), np.float32),
            "dt_proj.weight": np.zeros((spec.d_inner, 4), np.float32),
            "in_proj.weight": np.zeros((2 * spec.d_inner, spec.d_model), np.float32),
        }
        params, unmapped = map_official_params(official, spec)

        self.assertEqual(sorted(params), ["A_raw", "D_skip"])
        self.assertEqual(unmapped, ["dt_proj.weight", "in_proj.weight"])

    def test_full_migration_rejects_unmapped(self):
        spec = self._spec()
        official = {
            "A_log": np.zeros((spec.d_inner, spec.d_state), np.float32),
            "dt_proj.weight": np.zeros((spec.d_inner, 4), np.float32),
        }
        with self.assertRaises(ValueError) as ctx:
            from_official_checkpoint(official, spec)
        message = str(ctx.exception)
        self.assertIn("dt_proj.weight", message)
        self.assertIn("RMSNorm", message)

    def test_shapes_are_checked(self):
        spec = self._spec()
        with self.assertRaises(ValueError):
            map_official_params({"A_log": np.zeros((1, 1), np.float32)}, spec)
        with self.assertRaises(ValueError):
            map_official_params(
                {
                    "A_log": np.zeros((spec.d_inner, spec.d_state), np.float32),
                    "D": np.zeros((2,), np.float32),
                },
                spec,
            )


if __name__ == "__main__":
    unittest.main()
