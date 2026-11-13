"""分块 selective scan 的精度 / 性能 / 显存基准。在远程 GPU 上跑。

输出一个 JSON，包含：
- 前向误差（chunked vs serial、vs numpy 参考）
- VJP 误差（scan 反向、SSM 端到端梯度）
- 训练曲线（loss / grad norm 逐步）
- 峰值设备内存（执行增量）
- tokens/s、TTFT（prefill / decode）、逐 token 延迟
"""

import argparse
import json
import time

import numpy as np

import cupy as cp

from aetherflow import Graph, ReferenceBackend, backward, f32
from aetherflow.backends.cuda import CudaBackend
from aetherflow.models import build_selective_scan

# ``chunked_scan`` 的三态（5.155）：None = 按 (T, 尾维) 自动，0 = 强制串行，
# 正数 = 显式块长。基准要分别报"自动"与"串行"两条基线。
SCAN_VARIANTS = (("auto", None), ("serial", 0), ("64", 64), ("128", 128), ("256", 256))


def sync():
    cp.cuda.Stream.null.synchronize()


def timed(fn, warmup=3, repeat=10):
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    den = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / (den if den else 1.0))


# ----------------------------------------------------------------------
# 1. scan 算子：性能 + 显存
# ----------------------------------------------------------------------
def scan_inputs(T, B, D, seed=0):
    rng = np.random.default_rng(seed)
    a = (0.9 + 0.05 * rng.random((T, B, D))).astype(np.float32)
    b = (0.1 * rng.standard_normal((T, B, D))).astype(np.float32)
    h0 = np.zeros((B, D), np.float32)
    return cp.asarray(a), cp.asarray(b), cp.asarray(h0)


def bench_scan():
    out = []
    for T, B, D in ((1024, 16, 64), (4096, 16, 64), (16384, 16, 64)):
        a, b, h0 = scan_inputs(T, B, D)
        tokens = T * B
        row = {"T": T, "B": B, "D": D, "tokens": tokens}
        for label, chunk in SCAN_VARIANTS:
            be = CudaBackend(chunked_scan=chunk)
            be._scan(a, b, h0, backward=False)  # 预热并触发编译
            ms = timed(lambda: be._scan(a, b, h0, backward=False))
            row[f"chunk{label}_ms"] = round(ms * 1e3, 3)
            row[f"chunk{label}_tok_s"] = round(tokens / ms)
            row[f"chunk{label}_us_per_tok"] = round(ms * 1e6 / tokens, 3)
        out.append(row)
    return out


def bench_scan_backward():
    out = []
    for T, B, D in ((1024, 16, 64), (4096, 16, 64)):
        a, b, _ = scan_inputs(T, B, D)
        tokens = T * B
        row = {"T": T, "tokens": tokens}
        for label, chunk in (("auto", None), ("serial", 0), ("128", 128)):
            be = CudaBackend(chunked_scan=chunk)
            be._scan(a, b, None, backward=True)
            ms = timed(lambda: be._scan(a, b, None, backward=True))
            row[f"chunk{label}_ms"] = round(ms * 1e3, 3)
            row[f"chunk{label}_tok_s"] = round(tokens / ms)
        out.append(row)
    return out


def bench_memory():
    """free_all_blocks 后一次执行的设备内存增量（输入已在池中，不计入）。"""
    T, B, D = 4096, 16, 64
    a, b, h0 = scan_inputs(T, B, D)
    pool = cp.get_default_memory_pool()
    rows = []
    for name, fn in (
        ("serial", lambda be: be._scan(a, b, h0, backward=False)),
        ("chunk128", lambda be: be._scan(a, b, h0, backward=False)),
        ("last_serial", lambda be: be.scan_last_state(a, b, h0)),
        ("last_chunk128", lambda be: be.scan_last_state(a, b, h0)),
    ):
        chunk = 128 if "chunk" in name else 0
        be = CudaBackend(chunked_scan=chunk)
        fn(be)
        sync()
        pool.free_all_blocks()
        sync()
        base = pool.used_bytes()
        keep = fn(be)
        sync()
        rows.append({"case": name, "extra_bytes": pool.used_bytes() - base})
        del keep
        pool.free_all_blocks()
    return rows


# ----------------------------------------------------------------------
# 2. 延迟：prefill（TTFT）与 decode
# ----------------------------------------------------------------------
def bench_latency():
    rows = {}
    # prefill：T=512 一次前向 = 首 token 之前要算完的整段
    for T in (512, 1024):
        a, b, h0 = scan_inputs(T, 16, 64)
        tokens = T * 16
        for label, chunk in SCAN_VARIANTS:
            be = CudaBackend(chunked_scan=chunk)
            be._scan(a, b, h0, backward=False)
            ms = timed(lambda: be._scan(a, b, h0, backward=False), repeat=15)
            rows[f"prefill_T{T}_chunk{label}"] = {
                "ms": round(ms * 1e3, 3),
                "us_per_tok": round(ms * 1e6 / tokens, 3),
            }
    # decode：T=1，单步延迟；scan_last_state 只写末状态
    a1, b1, h0 = scan_inputs(1, 16, 64)
    for name, fn in (
        ("full_scan", lambda be: be._scan(a1, b1, h0, backward=False)),
        ("last_state", lambda be: be.scan_last_state(a1, b1, h0)),
    ):
        be = CudaBackend(chunked_scan=128)
        fn(be)
        ms = timed(lambda: fn(be), repeat=30)
        rows[f"decode_{name}"] = {"us": round(ms * 1e6, 2)}
    return rows


# ----------------------------------------------------------------------
# 3. SSM：前向 / VJP 误差
# ----------------------------------------------------------------------
SSM = {"steps": 64, "B": 2, "D": 16, "N": 8}


def ssm_graph():
    s = SSM
    g = Graph("ssm")
    x = g.input("x", f32[(s["steps"], s["B"], s["D"])])
    dt = g.input("dt", f32[(s["steps"], s["B"], s["D"])])
    A = g.input("A", f32[(s["D"], s["N"])])
    Bt = g.input("Bt", f32[(s["steps"], s["B"], s["N"])])
    Ct = g.input("Ct", f32[(s["steps"], s["B"], s["N"])])
    y = build_selective_scan(g, x, dt, A, Bt, Ct)
    g.output(y, "y")
    return g


def ssm_feed(seed=1):
    s = SSM
    rng = np.random.default_rng(seed)
    return {
        "x": rng.standard_normal((s["steps"], s["B"], s["D"])).astype(np.float32),
        "dt": (0.01 + 0.02 * rng.random((s["steps"], s["B"], s["D"]))).astype(np.float32),
        "A": (-np.exp(rng.standard_normal((s["D"], s["N"])) * 0.1)).astype(np.float32),
        "Bt": rng.standard_normal((s["steps"], s["B"], s["N"])).astype(np.float32),
        "Ct": rng.standard_normal((s["steps"], s["B"], s["N"])).astype(np.float32),
    }


def bench_accuracy():
    g = ssm_graph()
    feed = ssm_feed()
    ref = ReferenceBackend()
    # 逐位基线：显式关掉分块（0），不用默认的自动档
    serial = CudaBackend(chunked_scan=0)
    chunked = CudaBackend(chunked_scan=16)
    y_ref = ref.execute(g, feed)["y"]
    y_s = serial.execute(g, feed)["y"]
    y_c = chunked.execute(g, feed)["y"]

    # VJP：dy 随机，比较 chunked 与 serial 的梯度
    names = ["x", "dt", "A", "Bt", "Ct"]
    bwd = backward(g, wrt=[g.find(n) for n in names])
    rng = np.random.default_rng(2)
    dy = rng.standard_normal(y_ref.shape).astype(np.float32)
    gs = serial.execute(bwd, {**feed, "dy": dy})
    gc = chunked.execute(bwd, {**feed, "dy": dy})
    vjp = {f"d{n}": {"chunk_vs_serial": rel(gc[f"d{n}"], gs[f"d{n}"])} for n in names}
    # da0 单独报"串行 vs 参考"的绝对差距，说明分块的误差没有超过基线
    vjp["forward"] = {
        "serial_vs_ref": rel(y_s, y_ref),
        "chunk_vs_serial": rel(y_c, y_s),
        "chunk_vs_ref": rel(y_c, y_ref),
        "max_abs_diff_chunk_serial": float(
            np.max(np.abs(np.asarray(y_c, np.float64) - np.asarray(y_s, np.float64)))
        ),
    }
    return vjp


# ----------------------------------------------------------------------
# 4. 训练曲线：loss / grad norm（serial vs chunked）
# ----------------------------------------------------------------------
def train_curve(chunk, iters=8, lr=0.05):
    g = ssm_graph()
    names = ["x", "dt", "A", "Bt", "Ct"]
    bwd = backward(g, wrt=[g.find(n) for n in names])
    be = CudaBackend(chunked_scan=chunk)
    feed = {k: v.copy() for k, v in ssm_feed().items()}
    hist = []
    for _ in range(iters):
        y = be.execute(g, feed)["y"]
        loss = float(np.mean(np.asarray(y, np.float64) ** 2))
        dy = (2.0 * np.asarray(y) / y.size).astype(np.float32)
        grads = be.execute(bwd, {**feed, "dy": dy})
        gn = float(
            np.sqrt(sum(np.sum(np.asarray(grads[f"d{n}"], np.float64) ** 2) for n in names))
        )
        hist.append({"loss": round(loss, 8), "grad_norm": round(gn, 8)})
        for n in names:
            feed[n] = feed[n] - lr * np.asarray(grads[f"d{n}"], np.float32)
    return hist


def main(argv=None):
    parser = argparse.ArgumentParser(description="分块 selective scan 的精度 / 性能 / 显存基准")
    parser.add_argument("--out", default=None, help="把 JSON 结果写到这个路径")
    args = parser.parse_args(argv)
    result = {
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "forward_perf": bench_scan(),
        "backward_perf": bench_scan_backward(),
        "memory": bench_memory(),
        "latency": bench_latency(),
        "ssm_accuracy": bench_accuracy(),
        "train_serial": train_curve(0),
        "train_chunk16": train_curve(16),
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
