"""Winograd 融合内核的对照：权重驻留到底省下了什么。

用法（需要 GPU 与 cuDNN 9）：

```bash
python -m benchmarks.winograd_compare
python -m benchmarks.winograd_compare --rounds 5 --single
```

四路耗时，同一形状、同一份数据：

1. ``输入权重``：``conv_impl="winograd_fused"``，权重是**图输入**。基准
   脚本的常见形态，调用方可以随时换权重，因此每次执行都要重算
   ``U = G g G^T``；
2. ``常量权重``：同一张图，权重由 ``constant`` 节点产生。执行计划持有
   权重变换的结果，``U`` 只在第一次执行（或计划重建后）算一次；
3. ``cuDNN``：直调 cuDNN 的 legacy API，口径见 ``cudnn_compare``。

最后一列 ``U/20`` 是预热之后连续执行 20 次里权重变换内核的调用次数，取自
``DeviceBackend.conv_weight_transforms``——它是"驻留有没有真的生效"的
直接证据，比计时更不容易被噪声骗。常量权重的一列应当是 0，输入权重的一列
应当是 20。

**只比设备侧**：本项目侧走 ``execute_device``，输入先转成设备数组；cuDNN
侧同样不出卡。``NVIDIA_TF32_OVERRIDE=0`` 必须设，否则 cuDNN 会走 TF32，
数值对拍会被 1e-3 的差值拦下（本项目没有 TF32 路径）。

**注意这两列都是显式融合版**（``conv_impl="winograd_fused"``），而 auto 的
接管面上限是 C=24（``plan.conv_winograd_fused_preferred``）：c32/c64 的
"常量/cuDNN"是这条对照路径的比值，不是 ``auto`` 的表现。auto 在 C ≥ 32 走
四段版，实测为 cuDNN 的 1.02×/1.36×（DESIGN 5.268），要看那张表请用
``benchmarks.conv_compare``。
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Optional, Sequence

import numpy as np

from aetherflow import Graph, f32
from aetherflow.backends.cuda import CudaBackend
from aetherflow.ir.types import DType
from benchmarks.conv_compare import SHAPES
from benchmarks.cudnn_compare import measure, run_cudnn

# 3x3 形状才进 Winograd；覆盖"自动接管窗口"（c8～c24）与两侧的对照点。
# C 上界取 64：融合版把整块 ``V`` 放进共享内存（``16*C*TT*4`` 字节），
# 48 KB 预算只到 C=96，c128 会被派发处明确拒绝。
STAGES = tuple(s for s in SHAPES if s[2][2] == 3 and s[2][3] == 3 and s[1][1] <= 64)


def build(shape, constant_weight: bool):
    """同一份权重数据，两种来源：图输入 / 常量节点。"""
    name, x_shape, w_shape, pad = shape
    # 生成顺序必须与 ``cudnn_compare.build`` 完全一致（x → w → b），
    # 否则三列测的不是同一份数据，数值对拍会以"算法错误"的形式失败。
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(x_shape).astype(np.float32)
    w_np = (rng.standard_normal(w_shape) * 0.1).astype(np.float32)
    b_np = rng.standard_normal(w_shape[0]).astype(np.float32)

    g = Graph(name)
    x = g.input("x", f32[x_shape])
    b = g.input("b", f32[w_shape[0]])
    if constant_weight:
        w = g.constant(dtype=DType.F32, shape=w_shape, data=w_np.tolist())
        feed = {"x": x_np, "b": b_np}
    else:
        w = g.input("w", f32[w_shape])
        feed = {"x": x_np, "w": w_np, "b": b_np}
    g.output(g.add_node("conv2d", [x, w, b], {"padding": pad}), "y")
    return g, feed


def _measure(backend, stages, constant_weight: bool, rounds: int) -> Dict[str, Dict[str, float]]:
    import cupy as cp

    sync = cp.cuda.get_current_stream().synchronize
    out: Dict[str, Dict[str, float]] = {}
    for shape in stages:
        g, feed = build(shape, constant_weight)
        feed_dev = {k: cp.asarray(v) for k, v in feed.items()}
        best: Optional[Dict[str, float]] = None
        for _ in range(rounds):
            stats = measure(lambda: backend.execute_device(g, feed_dev), sync, 3, 10)
            if best is None or stats["burst_ms"] < best["burst_ms"]:
                best = stats
        out[shape[0]] = best
    return out


def transform_growth(backend, shape, constant_weight: bool, times: int = 20) -> int:
    """连续执行 ``times`` 次，返回权重变换内核的调用次数。"""
    import cupy as cp

    g, feed = build(shape, constant_weight)
    feed_dev = {k: cp.asarray(v) for k, v in feed.items()}
    backend.execute_device(g, feed_dev)  # 预热，把"首次必算"的那一次排除掉
    cp.cuda.get_current_stream().synchronize()
    before = backend.conv_weight_transforms
    for _ in range(times):
        backend.execute_device(g, feed_dev)
    cp.cuda.get_current_stream().synchronize()
    return backend.conv_weight_transforms - before


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Winograd 融合内核：权重驻留对照")
    parser.add_argument("--rounds", type=int, default=3, help="每形状跑几轮取最好（默认 3）")
    parser.add_argument(
        "--single", action="store_true",
        help="报单次延迟而不是稳态（单次含 0.08 ms 级固定开销，见 5.253）",
    )
    args = parser.parse_args(argv)
    key = "median_ms" if args.single else "burst_ms"

    try:
        cudnn_ms, cudnn_out, conv = run_cudnn(STAGES, args.rounds)
    except OSError as exc:
        print(f"跳过：找不到 cuDNN 9（{exc}）")
        return 0

    backend = CudaBackend(conv_impl="winograd_fused")
    in_ms = _measure(backend, STAGES, constant_weight=False, rounds=args.rounds)
    const_ms = _measure(backend, STAGES, constant_weight=True, rounds=args.rounds)
    tiled_ms = _measure(
        CudaBackend(conv_impl="tiled"), STAGES, constant_weight=False,
        rounds=args.rounds,
    )

    # 数值：两条路径必须逐位一致；对 cuDNN 按 1e-4 对齐（Winograd 的变换
    # 会放大舍入误差，容差与 cudnn_compare 一致）。
    for shape in STAGES:
        name = shape[0]
        g_in, feed_in = build(shape, False)
        g_c, feed_c = build(shape, True)
        a = backend.execute(g_in, feed_in)["y"]
        b = backend.execute(g_c, feed_c)["y"]
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(b, cudnn_out[name], rtol=1e-4, atol=1e-4)

    print(f"cuDNN {conv.version}（{conv.name}），"
          f"{'单次延迟' if args.single else '稳态（连续 20 次平摊）'}，ms")
    print(
        f"{'shape':10s} {'tiled':>8s} {'输入权重':>9s} {'常量权重':>9s} {'cuDNN':>9s}"
        f" {'常量/cuDNN':>11s} {'U/20(输入)':>11s} {'U/20(常量)':>11s}"
    )
    for shape in STAGES:
        name = shape[0]
        t = tiled_ms[name][key]
        i = in_ms[name][key]
        c = const_ms[name][key]
        d = cudnn_ms[name][key]
        gi = transform_growth(backend, shape, constant_weight=False)
        gc = transform_growth(backend, shape, constant_weight=True)
        print(
            f"{name:10s} {t:8.4f} {i:9.4f} {c:9.4f} {d:9.4f}"
            f" {c / d:10.2f}x {gi:11d} {gc:11d}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
