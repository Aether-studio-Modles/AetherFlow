"""torch eager / torch.compile 与 AetherFlow 的同机对照。

与 ``cudnn_compare``（库对库）不同，这一份是**框架对框架**：同机、同形状、
同精度、同一份输入，先核对数值，再比耗时。数字不跨机外推。

两种口径：

* **设备驻留调用**：输入（含权重）已在设备，输出不回主机；计入 Python
  派发、接口内部复制与同步，以及设备计算，并非只计内核。
* **训练一步**：设备驻留输入下的前向、反向与可选 SGD 更新。

纯算用例里既有单算子（``matmul`` / ``conv2d`` / ``maxpool``），也有整图
（``transformer_block`` / ``cnn_lenet`` 及其权重常驻版本）。整图用例同样是
"同一份权重、先对拍数值、再计时"，区别只在于它检验的是整图调度与多种 kernel
混合，而不是单个内核。torch 侧按 ``benchmarks.cases`` 的结构逐项重写算式，
不经过 AetherFlow 前端。

本脚本当前不测每轮 H2D / D2H 的主机端到端路径。

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
import os
import statistics
import sys
import time
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from aetherflow import backward
from benchmarks.cases import CASES

DEFAULT_CASES = (
    "matmul_1024",
    "conv2d_3x3_pad1",
    "transformer_block",
    "cnn_lenet",
)


def _stats(samples: List[float]) -> Dict[str, object]:
    return {"best_ms": round(min(samples), 4),
            "median_ms": round(statistics.median(samples), 4),
            "samples_ms": [round(value, 4) for value in samples]}


def _sample(fn: Callable[[], object], sync: Callable[[], None], burst: int) -> float:
    t0 = time.perf_counter()
    for _ in range(burst):
        fn()
    sync()
    return (time.perf_counter() - t0) * 1e3 / burst


def _measure(fn: Callable[[], object], sync: Callable[[], None], warmup: int,
             burst: int, rounds: int) -> Dict[str, float]:
    """稳态口径：连续 ``burst`` 次平摊，取多轮最好的一轮（与 cudnn_compare 一致）。"""
    for _ in range(warmup):
        fn()
    sync()
    return _stats([_sample(fn, sync, burst) for _ in range(rounds)])


def _measure_interleaved(fn_a: Callable[[], object], sync_a: Callable[[], None],
                         fn_b: Callable[[], object], sync_b: Callable[[], None],
                         warmup: int, burst: int, rounds: int):
    """两边交替测，每轮各测一次。

    共享机上的负载会漂移：先连续测完 A 再连续测 B，A 与 B 可能落在不同的
    时钟 / 占用状态上。逐轮轮换先后次序只能减轻顺序偏差；不同执行结构
    仍可能受到不同程度的干扰，共享卡上的比值不保证可靠。
    """
    for _ in range(warmup):
        fn_a()
        fn_b()
    sync_a()
    sync_b()
    samples_a: List[float] = []
    samples_b: List[float] = []
    for index in range(rounds):
        if index % 2 == 0:
            samples_a.append(_sample(fn_a, sync_a, burst))
            samples_b.append(_sample(fn_b, sync_b, burst))
        else:
            samples_b.append(_sample(fn_b, sync_b, burst))
            samples_a.append(_sample(fn_a, sync_a, burst))
    return _stats(samples_a), _stats(samples_b)


def _torch_transformer_block(tensors: Mapping[str, object], F):
    """与 ``benchmarks.cases._transformer_block`` 的 pre-norm 块逐项对齐。

    ``heads`` / ``eps`` 与那里的 ``BlockSpec(d_model=64, num_heads=4,
    d_ff=128)`` 绑定；该用例**没有因果掩码**（双向注意力），这里不能偷偷
    补一个，否则只是"看起来对"。
    """
    x = tensors["x"]
    dim = x.shape[-1]
    heads = 4
    head_dim = dim // heads
    eps = 1e-5
    h1 = F.layer_norm(x, (dim,), tensors["ln1_g"], tensors["ln1_b"], eps=eps)
    q = h1 @ tensors["wq"]
    k = h1 @ tensors["wk"]
    v = h1 @ tensors["wv"]
    batch, seq = x.shape[0], x.shape[1]
    qh = q.reshape(batch, seq, heads, head_dim).transpose(1, 2)
    kh = k.reshape(batch, seq, heads, head_dim).transpose(1, 2)
    vh = v.reshape(batch, seq, heads, head_dim).transpose(1, 2)
    attn = F.softmax((qh @ kh.transpose(-1, -2)) * tensors["scale"], dim=-1)
    merged = (attn @ vh).transpose(1, 2).reshape(batch, seq, dim)
    x1 = x + merged @ tensors["wo"]
    h2 = F.layer_norm(x1, (dim,), tensors["ln2_g"], tensors["ln2_b"], eps=eps)
    return x1 + F.silu(h2 @ tensors["w1"]) @ tensors["w2"]


def _torch_cnn_lenet(tensors: Mapping[str, object], F):
    """与 ``benchmarks.cases._cnn_lenet`` 同构的两个卷积块 + 全连接。"""
    h = F.max_pool2d(
        F.relu(F.conv2d(tensors["x"], tensors["w1"], tensors["b1"])), 2, 2
    )
    h = F.max_pool2d(
        F.relu(F.conv2d(h, tensors["w2"], tensors["b2"])), 2, 2
    )
    h = h.reshape(h.shape[0], -1)
    return F.relu(h @ tensors["fc"]) + tensors["fcb"]


def _torch_call(name: str, tensors: Mapping[str, object], F):
    """把 case 名字映射成一条 torch 算式；形状必须与 ``benchmarks.cases`` 一致。"""
    if name.startswith("matmul"):
        return tensors["a"] @ tensors["b"]
    if name.startswith("conv2d"):
        return F.conv2d(tensors["x"], tensors["w"], tensors["b"], stride=1, padding=1)
    if name.startswith("maxpool"):
        return F.max_pool2d(tensors["x"], 2, 2)
    if name.startswith("transformer_block"):
        return _torch_transformer_block(tensors, F)
    if name.startswith("cnn_lenet"):
        return _torch_cnn_lenet(tensors, F)
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


def _backend(device: str, graph_mode: str = "off"):
    if device.startswith("cuda"):
        from aetherflow.backends.cuda import CudaBackend

        return CudaBackend(graph_mode=graph_mode), "cuda"
    from aetherflow import ReferenceBackend

    return ReferenceBackend(), "ref"


def _other_gpu_processes() -> List[str]:
    """卡上除本进程外的计算进程（拿不到就不报）。

    查询覆盖所有可见 GPU，并未按目标卡过滤。存在其他进程时，绝对耗时和
    比值都可能受干扰（见 DESIGN 5.287），这里只警告、不阻止。
    """
    import subprocess

    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    mine = str(os.getpid())
    return [row.strip() for row in done.stdout.splitlines()
            if row.strip() and row.split(",")[0].strip() != mine]


def _synthetic_train(batch: int = 128, d: int = 1024):
    """两条 ``d x d`` 的 MLP + 交叉熵：前向、反向、参数更新都能对上 torch。"""
    from aetherflow.ir.graph import Graph
    from aetherflow.ir.types import f32, i32

    g = Graph("train_mlp")
    x = g.input("x", f32[batch, d])
    w1 = g.input("w1", f32[d, d])
    b1 = g.input("b1", f32[d])
    w2 = g.input("w2", f32[d, d])
    b2 = g.input("b2", f32[d])
    labels = g.input("labels", i32[batch])
    h = g.relu(g.add(g.matmul(x, w1), b1))
    logits = g.add(g.matmul(h, w2), b2)
    loss = g.reduce_mean(g.add_node("cross_entropy", [logits, labels], {}))
    g.output(loss, "loss")

    rng = np.random.default_rng(20261121)
    feed = {
        "x": rng.standard_normal((batch, d)).astype(np.float32),
        "w1": (rng.standard_normal((d, d)) * 0.02).astype(np.float32),
        "b1": np.zeros(d, dtype=np.float32),
        "w2": (rng.standard_normal((d, d)) * 0.02).astype(np.float32),
        "b2": np.zeros(d, dtype=np.float32),
        "labels": rng.integers(0, d, size=batch).astype(np.int32),
    }
    return g, feed, (w1, b1, w2, b2)


def _run_training(device, backend, cp, sync_af, sync_torch, torch, F, args) -> int:
    """训练一步的对照：前向 + 交叉熵 + 反向 + SGD。

    反向图只构一次（``Trainer`` 也是这么做的），循环里只执行；torch 侧
    ``loss.backward()`` 每步重建 autograd 图，这是两边的真实行为差异。
    """
    batch, dim = 128, 1024
    graph, feed, params = _synthetic_train(batch, dim)
    bwd = backward(graph, wrt=params)
    raw_nodes = len(bwd.nodes)
    if not args.raw_backward:
        from aetherflow.passes import backward_pipeline

        bwd = backward_pipeline().run(bwd)
    print(f"反向执行图：{raw_nodes} → {len(bwd.nodes)} 节点"
          f"（{'原始图' if args.raw_backward else '框架标准反向优化管线'}）")
    lr = 1e-6

    def to_dev(v):
        return cp.asarray(v) if cp is not None else v

    # CPU 上两框架也必须独立持有参数，不能通过 numpy 共享存储互相更新。
    feed_dev = {k: to_dev(v.copy()) for k, v in feed.items()}
    params_dev = [feed_dev[p.name] for p in params]
    seed_dev = to_dev(np.float32(1.0))
    grad_names = bwd.output_names
    af_exec = (backend.execute_device if hasattr(backend, "execute_device")
               else backend.execute)

    def af_step():
        out = af_exec(bwd, {**feed_dev, "dloss": seed_dev})
        if args.no_update:
            return
        for p, name in zip(params_dev, grad_names):
            grad = out[name]
            grad *= lr   # 原地缩放：避免每步 8 次临时分配，分配器状态会污染计时
            p -= grad

    tensors = {k: torch.tensor(v, device=device) for k, v in feed.items()}
    tensors["labels"] = tensors["labels"].long()
    ps = [tensors["w1"], tensors["b1"], tensors["w2"], tensors["b2"]]
    for p in ps:
        p.requires_grad_(True)

    def torch_step():
        logits = F.relu(tensors["x"] @ ps[0] + tensors["b1"]) @ ps[2] + tensors["b2"]
        loss = F.cross_entropy(logits, tensors["labels"])
        loss.backward()
        if args.no_update:
            for p in ps:
                p.grad = None
            return
        with torch.no_grad():
            for p in ps:
                p.grad.mul_(lr)
                p.sub_(p.grad)
                p.grad = None

    # 对拍完整损失、全部参数梯度与实际执行路径上的一步更新。
    af_loss = backend.execute(graph, feed)["loss"]
    af_grads = backend.execute(bwd, {**feed, "dloss": np.float32(1.0)})
    af_device_grads = af_exec(bwd, {**feed_dev, "dloss": seed_dev})
    sync_af()
    logits = F.relu(tensors["x"] @ ps[0] + tensors["b1"]) @ ps[2] + tensors["b2"]
    torch_loss = F.cross_entropy(logits, tensors["labels"])
    torch_loss.backward()
    np.testing.assert_allclose(af_loss, torch_loss.detach().cpu().numpy(),
                               rtol=1e-4, atol=1e-5, err_msg="损失与 torch 不一致")
    for param, name in zip(ps, grad_names):
        expected = param.grad.detach().cpu().numpy()
        np.testing.assert_allclose(af_grads[name], expected, rtol=1e-3, atol=1e-5,
                                   err_msg=f"完整梯度 {name} 与 torch 不一致")
        actual = (cp.asnumpy(af_device_grads[name]) if cp is not None
                  else af_device_grads[name])
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-5,
                                   err_msg=f"设备路径完整梯度 {name} 与 torch 不一致")
        param.grad = None
    af_step()
    torch_step()
    sync_af()
    sync_torch()
    for af_param, param in zip(params_dev, ps):
        actual = cp.asnumpy(af_param) if cp is not None else af_param
        np.testing.assert_allclose(actual, param.detach().cpu().numpy(),
                                   rtol=1e-4, atol=1e-6,
                                   err_msg="一步更新后的参数与 torch 不一致")
    for param, value in zip(params_dev, params):
        param[...] = to_dev(feed[value.name])
    with torch.no_grad():
        for p, key in zip(ps, ("w1", "b1", "w2", "b2")):
            p.copy_(torch.as_tensor(feed[key], device=device))
            p.grad = None

    af_stats, torch_stats = _measure_interleaved(
        af_step, sync_af, torch_step, sync_torch, args.warmup, args.burst, args.rounds)

    print(f"\n训练一步（batch={batch}, d={dim}, f32, lr={lr}；反向图复用）")
    ratio = torch_stats["median_ms"] / af_stats["median_ms"]
    print("设备驻留调用，中位数 ms；包含接口内部复制、主机派发与同步")
    print(f"{'case':<14}{'af':>10}{'torch':>10}{'torch/af':>10}")
    print(f"{'mlp_train':<14}{af_stats['median_ms']:>10.3f}"
          f"{torch_stats['median_ms']:>10.3f}{ratio:>9.2f}x")
    print(f"逐轮 AF：{af_stats['samples_ms']}；torch：{torch_stats['samples_ms']}")
    print(f"最好值 AF {af_stats['best_ms']:.4f} / "
          f"torch {torch_stats['best_ms']:.4f} ms（仅作补充）")

    if device.startswith("cuda"):
        print(f"图捕获 {getattr(backend, 'graph_captures', 0)} 次 / "
              f"命中 {getattr(backend, 'graph_replays', 0)} 次；"
              f"失败原因：{backend.graph_capture_failure(bwd) or '无'}")
        stats = backend.graph_profile_stats(bwd)
        if stats and stats["steps"]:
            n = stats["steps"]
            print("重放分段（每步 ms）："
                  f"upload {stats['upload'] / n * 1e3:.3f} / "
                  f"launch {stats['launch'] / n * 1e3:.3f} / "
                  f"sync {stats['sync'] / n * 1e3:.3f} / "
                  f"guards {stats['guards'] / n * 1e3:.3f}")
        print("显存未并列：torch allocated 与 CuPy 池步后增量不是同一口径，"
              "步后采样也不能代表峰值。")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="torch eager / compile 与 AetherFlow 的同机对照")
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES), help="逗号分隔的用例名")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto（默认）")
    parser.add_argument("--rounds", type=int, default=3, help="每例测量轮数（默认 3）")
    parser.add_argument("--burst", type=int, default=10, help="每轮连续调用次数（默认 10）")
    parser.add_argument("--warmup", type=int, default=3, help="预热次数（默认 3）")
    parser.add_argument("--compile", action="store_true", help="额外测 torch.compile")
    parser.add_argument("--compile-mode", default="default",
                        help="torch.compile 的模式（default / max-autotune）")
    parser.add_argument("--training", action="store_true",
                        help="改测训练一步（前向 + 交叉熵 + 反向 + SGD）")
    parser.add_argument("--graph-mode", default="off",
                        help="AetherFlow 的图重放模式（off / auto / capture）")
    parser.add_argument("--raw-backward", action="store_true",
                        help="训练时执行原始反向图，用于与标准优化管线作消融对照")
    parser.add_argument("--no-update", action="store_true",
                        help="只算梯度、不更新参数（分离参数更新成本）")
    parser.add_argument("--profile", action="store_true",
                        help="打开图重放的分段计时（upload / launch / sync / guards）")
    args = parser.parse_args(argv)
    if args.training and args.compile:
        parser.error("训练模式尚未实现 torch.compile 对照，不能把 eager 结果标作 compile")
    if args.rounds < 1 or args.burst < 1 or args.warmup < 0:
        parser.error("rounds、burst 必须为正数，warmup 不能为负数")

    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("跳过：没有安装 torch")
        return 0

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.profile:
        os.environ["AETHERFLOW_GRAPH_PROFILE"] = "1"

    backend, backend_kind = _backend(device, args.graph_mode)
    if device.startswith("cuda"):
        import cupy as cp

        others = _other_gpu_processes()
        if others:
            print("警告：可见 GPU 上存在其他计算进程（未按目标卡过滤）：")
            for row in others:
                print(f"  {row}")
            print("  若与目标卡共享，耗时及比值都可能受干扰，交替测量不能保证消除。")

        # 设备级同步：replay 跑在捕获流（非阻塞流）上，只同步「当前流」
        # 会漏掉它，测出来的是提交时间而不是完成时间；torch 侧用的也是
        # 全设备同步，两边必须同口径。
        sync_af = cp.cuda.Device().synchronize
        sync_torch = torch.cuda.synchronize
    else:
        cp = None  # type: ignore[assignment]
        sync_af = lambda: None
        sync_torch = lambda: None

    # f32 数学类型锁死：TF32 会让对拍差到 1e-3，与实现无关（见 DESIGN 5.251）
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    if args.training:
        return _run_training(device, backend, cp, sync_af, sync_torch, torch, F, args)

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

        # 2) 纯算：输入常驻，输出不回传；两边交替测，抵消负载漂移
        af_fn = ((lambda: backend.execute_device(graph, feed_dev))
                 if hasattr(backend, "execute_device")
                 else (lambda: backend.execute(graph, feed)))
        dev_stats, eager_stats = _measure_interleaved(
            af_fn, sync_af, fn_torch, sync_torch,
            args.warmup, args.burst, args.rounds)

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
