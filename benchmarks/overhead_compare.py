"""运行时固定开销分解：一次 ``execute_device`` 的主机时间花在哪。

背景（DESIGN 5.251 / 5.253 / 5.259）：c16 3x3 的内核已经反超 cuDNN
（0.78～0.81×），但单次 ``execute_device`` 仍有约 50 µs 的主机提交成本，
与 cuDNN 端到端 65 µs 同量级。要压这一段，先得知道它由哪几块构成，而不是
继续凭"感觉"改派发。

这个脚本把一次执行拆成可以独立测量的几层：

1. ``burst``：``measure`` 的稳态口径（20 次背靠背提交 + 一次同步，摊薄），
   是"框架 + 内核"的端到端时间；
2. ``host``：同样 20 次提交但不带同步，只量主机侧走完
   ``execute_device → run_device → 逐组派发`` 的时间。GPU 慢于主机时会
   被反压，因此它同时是"主机成本"与"设备下限"的较大者；与 ``burst``
   对照即可看出是否被反压；
3. ``kernel``：跳过执行骨架，直接调用计划里已绑定的内核函数 20 次。
   它包含内核函数内部的输出分配与 launch 参数打包，但不含图环境维护、
   输入校验、按组回收与计划查找。``host - kernel`` 就是骨架的净成本；
4. ``empty``：同形状 ``cp.empty`` 20 次——输出分配的最小成本；
5. ``pack``：一组 20 个 ``np.int32`` 的构造成本——卷积 launch 参数的
   最小成本（按 tiled 核最长的签名数）；
6. ``capture``：``graph_mode="auto"`` 的 ``burst``。捕获成功后主机侧只剩
   一次 replay，所以它逼近纯设备时间；``burst - capture`` 是"Python 递交
   开销"的上界（只在多节点图上可测）。

另外 ``--profile`` 用 ``cProfile`` 按 ``tottime`` 打出热点函数，直接回答
"主机时间具体落在哪个 Python 行"，不必逐段插桩。

用法（需要 GPU）：

```bash
python -m benchmarks.overhead_compare
python -m benchmarks.overhead_compare --cases single,block,wide --profile
```

口径与 ``cudnn_compare`` 一致：输入先 ``cp.asarray`` 常驻，权重走 IR 的
``constant`` 节点（让 5.257 的权重驻留生效），只测计算。所有数字都是
**本机相对值**，跨机只比比例。
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from aetherflow import Graph, f32
from aetherflow.backends.cuda import CudaBackend
from aetherflow.ir.types import DType
from benchmarks.conv_graph_compare import build as build_blocks
from benchmarks.cudnn_compare import measure

BURST = 20


def _const(g: Graph, shape, values: np.ndarray):
    return g.constant(dtype=DType.F32, shape=shape, data=values.tolist())


def build_single(channels: int, spatial: int, batch: int) -> Tuple[Graph, dict]:
    """单层 3x3 卷积：权重与偏置都是常量，走融合 Winograd + 权重驻留。"""
    rng = np.random.default_rng(0)
    shape = (batch, channels, spatial, spatial)
    g = Graph(f"conv_single_c{channels}")
    x = g.input("x", f32[shape])
    w = _const(
        g, (channels, channels, 3, 3),
        (rng.standard_normal((channels, channels, 3, 3)) * 0.08).astype(np.float32),
    )
    b = _const(g, (channels,), rng.standard_normal(channels).astype(np.float32))
    y = g.add_node("conv2d", [x, w, b], {"padding": 1})
    g.output(y, "y")
    return g, {"x": rng.standard_normal(shape).astype(np.float32)}


def _host_time(fn: Callable[[], None], sync, warmup: int = 3,
               repeat: int = BURST, rounds: int = 5) -> float:
    """连续 ``repeat`` 次提交、中途不同步，平摊后的主机时间（ms）。"""
    for _ in range(warmup):
        fn()
    sync()
    best: Optional[float] = None
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(repeat):
            fn()
        dt = (time.perf_counter() - t0) * 1e3 / repeat
        best = dt if best is None or dt < best else best
    sync()
    return best


def _pack_cost(repeat: int = BURST) -> float:
    """卷积 launch 的 20 个 ``np.int32`` 参数构造成本（ms/次）。"""
    c = h = w = o = 16
    kh = kw = ho = wo = 56
    sh = sw = ph = pw = dh = dw = 1
    toc = roc = 3
    cb = 16
    oc_tiles = 1
    span = 0
    has_bias = 1

    def pack():
        return (
            np.int32(c), np.int32(h), np.int32(w),
            np.int32(o), np.int32(kh), np.int32(kw),
            np.int32(ho), np.int32(wo),
            np.int32(sh), np.int32(sw), np.int32(ph), np.int32(pw),
            np.int32(dh), np.int32(dw),
            np.int32(toc), np.int32(roc), np.int32(cb),
            np.int32(oc_tiles), np.int32(has_bias), np.int32(span),
        )

    return _host_time(pack, lambda: None, warmup=100, repeat=repeat * 20)


def _kernel_probe(backend: CudaBackend, graph: Graph, feed_dev: dict,
                  sync) -> Tuple[float, str, int]:
    """直接调一个已绑定的内核函数，返回 ``(ms/次, 节点名, 组数)``。

    先用 ``run_device`` 拿一份完整环境（常量已在计划里驻留，设备副本复用），
    再挑第一个非常量组、且只有一个节点的组做探针。多节点融合组要重新构造
    中间值，测到的是融合路径而不是单内核，这里统一跳过。
    """
    env = backend.run_device(graph, feed_dev, release=False)
    plan = backend._plan(graph)
    for group in plan:
        if len(group.nodes) != 1 or group.kernels is None:
            continue
        node = group.nodes[0]
        if node.op.name == "constant":
            continue
        kernel = group.kernels[0]
        args = [env[a.id] for a in node.inputs]
        ms = _host_time(lambda: kernel(node, args, node.attrs), sync, warmup=5)
        return ms, node.op.name, len(plan)
    return float("nan"), "-", len(plan)


def _case(backend: CudaBackend, graph: Graph, feed: dict, label: str,
          profile: bool, steps: int = 40):
    import cupy as cp

    sync = cp.cuda.get_current_stream().synchronize
    feed_dev = {k: cp.asarray(v) for k, v in feed.items()}

    # 预热：让计划构建、常量上传与权重驻留都发生在计时窗口之外。
    backend.execute_device(graph, feed_dev)
    sync()

    burst = measure(lambda: backend.execute_device(graph, feed_dev), sync, 3, 10)
    host = _host_time(lambda: backend.execute_device(graph, feed_dev), sync)

    kernel_ms, op, groups = _kernel_probe(backend, graph, feed_dev, sync)
    out_shape = graph.outputs[0].type.shape
    empty_ms = _host_time(lambda: cp.empty(out_shape, dtype=cp.float32), sync,
                          warmup=100, repeat=BURST * 20)
    pack_ms = _pack_cost()

    capture_ms = float("nan")
    if backend.graph_mode == "auto":
        capture_ms = measure(
            lambda: backend.execute_device(graph, feed_dev), sync, 3, 10
        )["burst_ms"]

    if profile:
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(steps):
            backend.execute_device(graph, feed_dev)
        pr.disable()
        sync()
        print(f"\n--- cProfile [{label}] {steps} 步，tottime 前 12 ---")
        pstats.Stats(pr).sort_stats("tottime").print_stats(12)

    return {
        "label": label,
        "nodes": len(graph.nodes),
        "groups": groups,
        "burst": burst["burst_ms"],
        "host": host,
        "kernel": kernel_ms,
        "skeleton": host - kernel_ms,
        "empty": empty_ms,
        "pack": pack_ms,
        "capture": capture_ms,
        "op": op,
    }


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="execute_device 固定开销分解")
    parser.add_argument("--cases", default="single,block,wide",
                        help="逗号分隔：single / block / wide")
    parser.add_argument("--blocks", type=int, default=8, help="block 用例的层数")
    parser.add_argument("--profile", action="store_true", help="附 cProfile 热点")
    args = parser.parse_args(argv)

    cases = [s for s in args.cases.split(",") if s]
    off = CudaBackend(graph_mode="off")
    auto = CudaBackend(graph_mode="auto")

    rows: List[Dict[str, float]] = []
    for name in cases:
        if name == "single":
            graph, feed = build_single(16, 56, 4)
            label = "single c16"
        elif name == "wide":
            graph, feed = build_single(64, 56, 4)
            label = "single c64"
        elif name == "block":
            graph, feed = build_blocks(args.blocks, "scalar")
            label = f"block x{args.blocks}"
        else:
            raise SystemExit(f"未知用例: {name}")
        rows.append(_case(off, graph, feed, label, args.profile))
        if name != "single":
            rows[-1]["capture"] = _case(auto, graph, feed, label, False)["burst"]

    print(
        f"\n{'case':12s} {'nodes':>5s} {'grp':>4s} {'burst':>8s} {'host':>8s}"
        f" {'kernel':>8s} {'skel':>7s} {'empty':>7s} {'pack':>7s} {'cap':>8s}"
    )
    for r in rows:
        cap = f"{r['capture']:8.4f}" if np.isfinite(r["capture"]) else f"{'-':>8s}"
        print(
            f"{r['label']:12s} {r['nodes']:5d} {r['groups']:4d}"
            f" {r['burst']:8.4f} {r['host']:8.4f} {r['kernel']:8.4f}"
            f" {r['skeleton']:7.4f} {r['empty']:7.4f} {r['pack']:7.4f} {cap}"
        )
    print(
        "\n单位 ms：burst = 20 次提交 + 同步的稳态；host = 20 次提交不带同步；\n"
        "kernel = 直接调已绑定内核函数；skel = host - kernel；\n"
        "empty = 同形状 cp.empty；pack = 20 个 np.int32 的构造；"
        "cap = 捕获后 replay。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
