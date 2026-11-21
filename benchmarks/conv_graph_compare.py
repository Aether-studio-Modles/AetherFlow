"""层 / 网络级对照：conv-bn-relu 多节点图在捕获前后的设备侧耗时。

用法（需要 GPU）：

```bash
python -m benchmarks.conv_graph_compare
python -m benchmarks.conv_graph_compare --blocks 6 --rounds 5
python -m benchmarks.conv_graph_compare --cases scalar
```

为什么要单独一个脚本：``winograd_compare`` 与 ``cudnn_compare`` 都只测**单个
卷积**，回答不了两个问题——

1. 逐元素融合把 ``mul`` / ``add`` / ``relu`` 合成一个内核之后，一次前向到底
   要派发几次？这由 ``group_nodes`` 直接数出来，不靠计时推断；
2. ``graph_mode="auto"``（CUDA Graph 捕获）在一张**由这些组构成**的图上还能
   省多少？单节点图没有收益，节点多的图才有（见 DESIGN 5.251）。

两个用例的区别只在 BN 的表达：

- ``scalar``：``conv → mul(标量) → add(标量) → relu``。标量一侧恰好一个元素，
  满足 ``_chainable`` 的条件，三个逐元素算子被合成一组；
- ``channel4d``：``conv → mul(1,C,1,1) → add(1,C,1,1) → relu``。逐通道广播
  不在融合链的判据里（``_chainable`` 只认整块同形或单元素），所以 ``mul`` /
  ``add`` 各自成组——这一列就是"融合判据没覆盖到的部分"的代价；
- ``channel``：同上，但参数以 ``(C,)`` 给出、经 ``reshape`` 变成
  ``(1,C,1,1)``，与前端导入的形态一致。它与 ``channel4d`` 的差就是那两个
  ``reshape`` 节点的代价。

BN 本身在这份 IR 里是训练形态（沿除通道轴外的轴求统计量），推理形态等价于
"逐通道仿射 + relu"，所以这里直接用 ``mul`` / ``add`` 表达，卷积权重与 BN
参数都走 ``constant`` 节点：既让权重驻留生效（5.257），也不给图增加输入。
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

from aetherflow import Graph, f32
from aetherflow.backends.cuda import CudaBackend
from aetherflow.ir.types import DType
from benchmarks.cudnn_compare import measure

# 一层 block 的通道数与空间尺寸。c16 落在融合版 Winograd 的窗口内
# （8 ≤ C ≤ 24），所以 conv 那一段本身是最快的那条路。
CHANNELS = 16
SPATIAL = 56
BATCH = 4


def _const(g: Graph, shape, values: np.ndarray):
    return g.constant(dtype=DType.F32, shape=shape, data=values.tolist())


def build(blocks: int, mode: str):
    """``blocks`` 个 conv-bn-relu，返回 ``(graph, feed, 期望的组数)``。"""
    rng = np.random.default_rng(0)
    shape = (BATCH, CHANNELS, SPATIAL, SPATIAL)
    g = Graph(f"conv_bn_relu_x{blocks}_{mode}")
    x = g.input("x", f32[shape])
    h = x
    for i in range(blocks):
        w = _const(
            g,
            (CHANNELS, CHANNELS, 3, 3),
            (rng.standard_normal((CHANNELS, CHANNELS, 3, 3)) * 0.08).astype(np.float32),
        )
        b = _const(g, (CHANNELS,), rng.standard_normal(CHANNELS).astype(np.float32))
        c = g.add_node("conv2d", [h, w, b], {"padding": 1})
        if mode == "scalar":
            s = _const(g, (1,), np.array([0.8], np.float32))
            t = _const(g, (1,), np.array([0.1], np.float32))
        elif mode == "channel4d":
            # 导出器直接给出 (1, C, 1, 1) 的形态：没有 reshape，但广播仍在。
            s = _const(
                g, (1, CHANNELS, 1, 1),
                (1.0 + 0.1 * rng.standard_normal(CHANNELS)).astype(np.float32)[None, :, None, None],
            )
            t = _const(
                g, (1, CHANNELS, 1, 1),
                (0.1 * rng.standard_normal(CHANNELS)).astype(np.float32)[None, :, None, None],
            )
        else:
            # IR 的广播是右对齐的，(C,) 对不上 NCHW 的最后一维，所以逐通道
            # 参数先 reshape 成 (1, C, 1, 1)。reshape 是视图，但它在 IR 里仍
            # 是一个节点、也不会进融合链——这正是这一列要量的代价。
            s = g.add_node(
                "reshape",
                [_const(
                    g, (CHANNELS,),
                    (1.0 + 0.1 * rng.standard_normal(CHANNELS)).astype(np.float32),
                )],
                {"shape": (1, CHANNELS, 1, 1)},
            )
            t = g.add_node(
                "reshape",
                [_const(
                    g, (CHANNELS,),
                    (0.1 * rng.standard_normal(CHANNELS)).astype(np.float32),
                )],
                {"shape": (1, CHANNELS, 1, 1)},
            )
        m = g.add_node("mul", [c, s], {})
        a = g.add_node("add", [m, t], {})
        h = g.add_node("relu", [a], {})
    g.output(h, "y")
    x_np = rng.standard_normal(shape).astype(np.float32)
    return g, {"x": x_np}


def dispatch_lines(backend: CudaBackend, graph: Graph) -> List[str]:
    """每组一行：组内的节点名。组内多于一个节点就是融合生效的地方。"""
    out = []
    for group in backend.group_nodes(graph):
        out.append("+".join(node.op.name for node in group))
    return out


def run_case(backend: CudaBackend, graph: Graph, feed, blocks: int, rounds: int):
    import cupy as cp

    sync = cp.cuda.get_current_stream().synchronize
    feed_dev = {k: cp.asarray(v) for k, v in feed.items()}
    best: Optional[Dict[str, float]] = None
    for _ in range(rounds):
        stats = measure(lambda: backend.execute_device(graph, feed_dev), sync, 3, 10)
        if best is None or stats["burst_ms"] < best["burst_ms"]:
            best = stats
    return best


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="conv-bn-relu 多节点图的设备侧对照")
    parser.add_argument("--blocks", type=int, default=4, help="conv-bn-relu 的层数")
    parser.add_argument("--rounds", type=int, default=3, help="每例跑几轮取最好")
    parser.add_argument("--cases", default="scalar,channel,channel4d",
                        help="用例，逗号分隔")
    args = parser.parse_args(argv)

    modes = [s for s in args.cases.split(",") if s]
    backends = {
        "off": CudaBackend(graph_mode="off"),
        "auto": CudaBackend(graph_mode="auto"),
    }
    rows = []
    for mode in modes:
        graph, feed = build(args.blocks, mode)
        # 分组只看图与后端配置，与 graph_mode 无关；用 off 的那个数。
        lines = dispatch_lines(backends["off"], graph)
        fused = sum(1 for line in lines if "+" in line)
        print(f"\n[{mode}] {len(graph.nodes)} 个节点 → {len(lines)} 组，其中 {fused} 组是融合链")
        for line in lines:
            print(f"  {line}")
        best = {}
        for label, backend in backends.items():
            best[label] = run_case(backend, graph, feed, args.blocks, args.rounds)
        rows.append((mode, len(graph.nodes), len(lines), best["off"], best["auto"]))

    print(
        f"\n{'case':10s} {'nodes':>6s} {'groups':>7s} {'off ms':>9s}"
        f" {'capture ms':>11s} {'speedup':>8s}"
    )
    for mode, nodes, groups, off, auto in rows:
        print(
            f"{mode:10s} {nodes:6d} {groups:7d} {off['burst_ms']:9.4f}"
            f" {auto['burst_ms']:11.4f} {off['burst_ms'] / auto['burst_ms']:7.2f}x"
        )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
