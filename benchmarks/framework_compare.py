"""torch eager / torch.compile 与 AetherFlow 的同机对照。

与 ``cudnn_compare``（库对库）不同，这一份是**框架对框架**：同机、同形状、
同精度、同一份输入，先核对数值，再比耗时。数字不跨机外推。

两种口径：

* **纯算**：输入（含权重）常驻显存，只计内核；两边都是。
* **端到端**：每轮把全部输入上传、把输出取回主机；AetherFlow 走 ``execute``，
  torch 侧每轮 ``.to(device)`` / ``.cpu()``。

torch 侧**不经过 AetherFlow 前端**：torch 2.5.1 不在兼容矩阵内，走
``torch.export`` 会被入口按设计拒绝；对照只需要同一算式，不需要同一张图。

用法::

    python -m benchmarks.framework_compare --device cuda
    python -m benchmarks.framework_compare --cases matmul_1024,conv2d_3x3_pad1 \\
        --compile --rounds 3

无 CUDA 时退回 CPU（只用于验证脚本与数值对拍，耗时没有比较意义）。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from benchmarks.cases import CASES

DEFAULT_CASES = ("matmul_1024", "conv2d_3x3_pad1", "maxpool2x2")


def _measure(fn: Callable[[], object], sync: Callable[[], None], warmup: int,
             burst: int, rounds: int) -> Dict[str, float]:
    """稳态口径：连续 ``burst`` 次平摊，取多轮最好的一轮（与 cudnn_compare 一致）。"""
    for _ in range(warmup):
        fn()
    sync()
    samples: List[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(burst):
            fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1e3 / burst)
    return {"best_ms": round(min(samples), 4),
            "median_ms": round(statistics.median(samples), 4)}


def _torch_call(name: str, tensors: Mapping[str, object], F):
    """把 case 名字映射成一条 torch 算式；形状必须与 ``benchmarks.cases`` 一致。"""
    if name.startswith("matmul"):
        return tensors["a"] @ tensors["b"]
    if name.startswith("conv2d"):
        return F.conv2d(tensors["x"], tensors["w"], tensors["b"], stride=1, padding=1)
    if name.startswith("maxpool"):
        return F.max_pool2d(tensors["x"], 2, 2)
    raise SystemExit(f"未登记的对照用例：{name}（在 _torch_call 里补一条等价算式）")


def _synthetic_matmul(n: int):
    """合成 ``n x n x n`` 的 GEMM：基准图集里最大只到 1024²，
    那个尺寸下单次派发就能淹没内核差异，看框架对比要用更大的。"""
    from aetherflow.ir.graph import Graph
    from aetherflow.ir.types import f32

    g = Graph(f"matmul_{n}")
    a = g.input("a", f32[n, n])
    b = g.input("b", f32[n, n])
    g.output(g.matmul(a, b), "y")
    rng = np.random.default_rng(20261121)
    return g, {"a": rng.standard_normal((n, n)).astype(np.float32),
               "b": (rng.standard_normal((n, n)) * 0.1).astype(np.float32)}


def _build(name: str):
    table = {case.name: case for case in CASES}
    if name in table:
        return table[name].build(np.random.default_rng(20261121))
    if name.startswith("matmul_") and name[len("matmul_"):].isdigit():
        return _synthetic_matmul(int(name[len("matmul_"):]))
    raise SystemExit(f"未知用例：{name}；可选 {', '.join(table)} 或 matmul_<N>")


def _backend(device: str):
    if device.startswith("cuda"):
        from aetherflow.backends.cuda import CudaBackend

        return CudaBackend(), "cuda"
    from aetherflow import ReferenceBackend

    return ReferenceBackend(), "ref"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="torch eager / compile 与 AetherFlow 的同机对照")
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES), help="逗号分隔的用例名")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto（默认）")
    parser.add_argument("--rounds", type=int, default=3, help="每例跑几轮取最好（默认 3）")
    parser.add_argument("--burst", type=int, default=10, help="每轮连续调用次数（默认 10）")
    parser.add_argument("--warmup", type=int, default=3, help="预热次数（默认 3）")
    parser.add_argument("--compile", action="store_true", help="额外测 torch.compile")
    parser.add_argument("--compile-mode", default="default",
                        help="torch.compile 的模式（default / max-autotune）")
    args = parser.parse_args(argv)

    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("跳过：没有安装 torch")
        return 0

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    backend, backend_kind = _backend(device)
    if device.startswith("cuda"):
        import cupy as cp

        sync_af = cp.cuda.get_current_stream().synchronize
        sync_torch = torch.cuda.synchronize
    else:
        cp = None  # type: ignore[assignment]
        sync_af = lambda: None
        sync_torch = lambda: None

    # f32 数学类型锁死：TF32 会让对拍差到 1e-3，与实现无关（见 DESIGN 5.251）
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    names = [n for n in args.cases.split(",") if n]
    print(f"设备 {device}，torch {torch.__version__}，"
          f"aetherflow 后端 {backend_kind}，"
          f"{'稳态（连续 %d 次平摊）' % args.burst if args.burst > 1 else '单次'}")

    rows: List[Tuple[str, Dict[str, float], Dict[str, float], Optional[Dict[str, float]],
                     Optional[float]]] = []
    for name in names:
        graph, feed = _build(name)
        # 常驻输入：两边都先把同一份数据搬到设备
        if cp is not None:
            feed_dev = {k: cp.asarray(v) for k, v in feed.items()}
            tensors = {k: torch.as_tensor(v, device=device) for k, v in feed.items()}
        else:
            feed_dev = feed
            tensors = {k: torch.as_tensor(v) for k, v in feed.items()}
        fn_torch = lambda t=tensors, n=name: _torch_call(n, t, F)

        # 1) 数值对拍：先对拍，再比耗时；不一致就不出数字
        af_out = backend.execute(graph, feed)[graph.output_names[0]]
        torch_out = fn_torch().detach().cpu().numpy()
        np.testing.assert_allclose(af_out, torch_out, rtol=1e-4, atol=1e-4,
                                   err_msg=f"{name} 的数值与 torch 不一致")

        # 2) 纯算：输入常驻，输出不回传
        dev_stats = _measure(
            (lambda: backend.execute_device(graph, feed_dev))
            if hasattr(backend, "execute_device")
            else (lambda: backend.execute(graph, feed)),
            sync_af, args.warmup, args.burst, args.rounds)
        eager_stats = _measure(fn_torch, sync_torch, args.warmup, args.burst, args.rounds)

        compile_stats: Optional[Dict[str, float]] = None
        compile_ms: Optional[float] = None
        if args.compile:
            compiled = torch.compile(fn_torch, mode=args.compile_mode)
            t0 = time.perf_counter()
            compiled()
            sync_torch()
            compile_ms = round((time.perf_counter() - t0) * 1e3, 1)
            compile_stats = _measure(compiled, sync_torch, 0, args.burst, args.rounds)

        rows.append((name, dev_stats, eager_stats, compile_stats, compile_ms))
        if compile_stats:
            print(f"  {name}: 纯算 af {dev_stats['best_ms']:.4f} / "
                  f"eager {eager_stats['best_ms']:.4f} / "
                  f"compile {compile_stats['best_ms']:.4f} ms")
        else:
            print(f"  {name}: 纯算 af {dev_stats['best_ms']:.4f} / "
                  f"eager {eager_stats['best_ms']:.4f} ms")

    print("\n纯算（输入常驻、输出不回传），ms；比值 = torch / aetherflow，<1 表示 torch 更快")
    header = f"{'case':<22}{'af':>9}{'eager':>9}{'compile':>9}{'eager/af':>10}{'comp/af':>9}{'编译':>9}"
    print(header)
    for name, dev_stats, eager_stats, compile_stats, compile_ms in rows:
        eager_ratio = eager_stats["best_ms"] / dev_stats["best_ms"]
        if compile_stats:
            comp_ratio = f"{compile_stats['best_ms'] / dev_stats['best_ms']:.2f}x"
            comp_ms = f"{compile_stats['best_ms']:.4f}"
            compile_time = f"{compile_ms:.0f}ms"
        else:
            comp_ratio = comp_ms = compile_time = "-"
        print(f"{name:<22}{dev_stats['best_ms']:>9.4f}{eager_stats['best_ms']:>9.4f}"
              f"{comp_ms:>9}{eager_ratio:>9.2f}x{comp_ratio:>9}{compile_time:>9}")

    if device.startswith("cuda"):
        print(f"\n峰值显存（torch 实测）：{torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")
        print("AetherFlow 侧显存用 plan_memory 的静态估计，与本行不是同一口径，未并列。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
