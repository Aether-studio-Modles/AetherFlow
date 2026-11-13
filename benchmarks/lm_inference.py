"""语言模型推理路径基准：固定块预填充 / 末位置投影 / 权重常驻。

审计 §8 的验收口径是"相同权重对拍整段 / 分块 / 逐 token，并覆盖 1K、8K、
32K 的预填充与长 decode"，记录四类数字：

- **首 token 延迟（TTFT）**：分块预填充 + 一次采样（含 ``max_new_tokens=1``
  时 prefill 的最后一次 execute 与一次 D2H 取 logits 行）；
- **稳态 tok/s**：``(total - ttft) / (N - 1)``，即首 token 之后的逐步 decode；
- **显存峰值**：``--memory-mode tiered`` 时读 Arena 的 ``peak_bytes``；
  container 模式没有峰值接口，如实写 ``None``，不拿"当前占用"冒充；
- **跨步传输字节**：权重常驻、状态留在设备上时，每步只剩 ids/reset 的 H2D
  与一行 logits 的 D2H；状态本身不经过主机。

运行（远程 GPU）：

```bash
python -m benchmarks.lm_inference --backend cuda --lengths 1024,8192,32768
```

1K 也可以在参考后端上粗跑，但 32K 的逐节点解释执行不具参考价值，默认只给
设备后端。
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from aetherflow.models import LMSpec, generate


def make_backend(name: str, memory_mode: str, graph_mode: str):
    if name == "cuda":
        from aetherflow.backends.cuda import CudaBackend

        return CudaBackend(memory_mode=memory_mode, graph_mode=graph_mode)
    if name == "hip":
        from aetherflow.backends.hip import HipBackend

        return HipBackend(memory_mode=memory_mode, graph_mode=graph_mode)
    raise ValueError(f"未知后端 {name!r}；可选 cuda / hip")


def sync(backend) -> None:
    """等设备算完。HIP 是模块级 ``synchronize``，CUDA 走流的空流同步。"""
    runtime = backend.cp
    module_sync = getattr(runtime, "synchronize", None)
    if callable(module_sync):
        module_sync()
    else:
        runtime.cuda.Stream.null.synchronize()


def timed(fn, backend, *, warmup: int = 1, repeat: int = 3) -> float:
    for _ in range(warmup):
        fn()
    sync(backend)
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        sync(backend)
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def memory_snapshot(backend):
    """最近一次执行的显存快照；没有统计接口时明确返回 ``None``。"""
    stats = backend.memory_stats() if hasattr(backend, "memory_stats") else None
    if stats is not None:
        return {
            "live_bytes": int(stats.live_bytes),
            "peak_bytes": int(stats.peak_bytes),
            "total_bytes": int(stats.total_bytes),
            "requested_bytes": int(stats.requested_bytes),
        }
    get_pool = getattr(backend.cp, "get_default_memory_pool", None)
    if get_pool is None:
        return {"peak_bytes": None, "note": "该后端没有显存统计接口"}
    pool = get_pool()
    return {
        "used_bytes": int(pool.used_bytes()),
        "reserved_bytes": int(pool.total_bytes()),
        "peak_bytes": None,
        "note": "container 模式没有峰值接口；用 --memory-mode tiered 取 peak_bytes",
    }


def pick_prompt(backend, params, spec, length, new_tokens, seed):
    """挑一个不会在 ``new_tokens`` 内撞上 EOS 的 prompt，让稳态样本可比。"""
    for attempt in range(4):
        rng = np.random.default_rng(seed + attempt)
        prompt = rng.integers(2, spec.vocab_size, length).astype(np.int32)
        got = generate(
            backend, params, prompt, spec,
            max_new_tokens=new_tokens, chunk_size=spec.steps, resident=True,
        )
        if len(got) == new_tokens:
            return prompt
    raise RuntimeError(
        f"四次尝试都在 {new_tokens} 步内遇到 EOS；请换 seed 或调大词表"
    )


def bench_one(backend, spec, params, length, new_tokens, seed):
    prompt = pick_prompt(backend, params, spec, length, new_tokens, seed)

    def run(steps):
        return generate(
            backend, params, prompt, spec,
            max_new_tokens=steps, chunk_size=spec.steps, resident=True,
        )

    # 预热：让图捕获、编译与驻留路径都走一遍，计入 TTFT 之前先排除掉
    run(new_tokens)
    ttft = timed(lambda: run(1), backend)
    ttft_mem = memory_snapshot(backend)
    total = timed(lambda: run(new_tokens), backend)
    total_mem = memory_snapshot(backend)

    decode_steps = max(new_tokens - 1, 1)
    decode_seconds = max(total - ttft, 0.0)
    state_bytes = sum(
        int(np.prod(shape)) * 4 for shape in spec.state_shapes.values()
    )
    params_bytes = sum(int(value.nbytes) for value in params.values())
    logits_row_bytes = spec.vocab_size * 4
    return {
        "prompt_tokens": length,
        "chunk_size": spec.steps,
        "chunks": -(-length // spec.steps),
        "new_tokens": new_tokens,
        "ttft_ms": round(ttft * 1e3, 3),
        "prefill_tok_s": round(length / ttft, 1) if ttft > 0 else None,
        "decode_ms_per_token": round(decode_seconds / decode_steps * 1e3, 3),
        "decode_tok_s": round(decode_steps / decode_seconds, 1)
        if decode_seconds > 0
        else None,
        "memory_prefill": ttft_mem,
        "memory_end": total_mem,
        "transfer": {
            "weights_bytes_per_decode_step": 0,
            "state_bytes_per_decode_step": 0,
            "host_to_device_bytes_per_decode_step": 4 + 4,
            "device_to_host_bytes_per_decode_step": logits_row_bytes,
            "host_to_device_bytes_per_prefill_chunk": spec.steps * 4 * 2,
            "weights_bytes_if_not_resident": params_bytes,
            "state_bytes_if_transferred": state_bytes,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="LM 推理路径基准")
    parser.add_argument("--backend", default="cuda", help="cuda / hip")
    parser.add_argument("--lengths", default="1024,8192,32768")
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--memory-mode", default="tiered")
    parser.add_argument("--graph-mode", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="把 JSON 结果写到这个路径")
    args = parser.parse_args(argv)

    lengths = [int(part) for part in args.lengths.split(",") if part.strip()]
    spec = LMSpec(
        vocab_size=1024,
        d_model=64,
        num_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
        steps=args.chunk,
        batch=1,
        tie_weights=True,
        pad_id=0,
        eos_id=1,
    )
    backend = make_backend(args.backend, args.memory_mode, args.graph_mode)
    params = spec.init_params(seed=0)
    info = backend.device_info() if hasattr(backend, "device_info") else None

    result = {
        "backend": args.backend,
        "device": getattr(info, "name", None),
        "graph_mode": args.graph_mode,
        "memory_mode": args.memory_mode,
        "spec": {
            "vocab_size": spec.vocab_size,
            "d_model": spec.d_model,
            "num_layers": spec.num_layers,
            "d_state": spec.d_state,
            "chunk": spec.steps,
        },
        "cases": [],
    }
    for length in lengths:
        if length < 1:
            raise ValueError(f"长度必须为正，得到 {length}")
        result["cases"].append(
            bench_one(backend, spec, params, length, args.new_tokens, args.seed)
        )

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
