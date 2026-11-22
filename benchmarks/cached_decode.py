"""Wall time for a complete cached Transformer hidden-state forward on synthetic data.

Includes graph construction, data transfers, cache initialization, projections,
attention, FFN and return of all hidden states. No tokenizer, LM head or sampling.
Both paths use the same model, inputs, backend and page size, without auto capture.
"""

import argparse
import json
import statistics
import time

import numpy as np

from aetherflow.backends.cuda import CudaBackend
from aetherflow.models import TransformerSpec, cached_forward


def compare(backend, steps, block_size, rounds):
    spec = TransformerSpec(d_model=128, num_heads=4, d_ff=256, num_layers=2,
                           batch=1, seq=steps, vocab=0, dropout=0.0, fused_attention=True)
    rng = np.random.default_rng(482)
    params = {}
    for name, shape in spec.param_shapes.items():
        if name.endswith("_g"):
            params[name] = np.ones(shape, dtype=np.float32)
        elif name.endswith("_b") or name == "pos":
            params[name] = np.zeros(shape, dtype=np.float32)
        else:
            params[name] = rng.standard_normal(shape, dtype=np.float32) / np.float32(shape[0] ** 0.5)
    xs = rng.standard_normal((steps, spec.d_model), dtype=np.float32)

    def run(direct):
        return cached_forward(backend, params, spec, xs, block_size=block_size, direct_paged=direct)

    gathered, direct = run(False), run(True)
    np.testing.assert_allclose(direct, gathered, atol=5e-5, rtol=3e-4)
    max_error = float(np.max(np.abs(direct - gathered)))
    samples = {"gather": [], "direct": []}
    variants = (("gather", False), ("direct", True))
    for r in range(rounds):
        for label, use_direct in (variants if r % 2 == 0 else variants[::-1]):
            t0 = time.perf_counter()
            run(use_direct)
            samples[label].append((time.perf_counter() - t0) * 1000)
    gather_ms, direct_ms = (statistics.median(samples[label]) for label in ("gather", "direct"))
    info = backend.device_info()
    return dict(device=info.name, runtime=info.runtime, dtype="f32", steps=steps, batch=1,
                d_model=spec.d_model, layers=spec.num_layers, block_size=block_size,
                gather_ms=gather_ms, direct_ms=direct_ms, speedup=gather_ms / direct_ms,
                max_abs_error=max_error, samples_ms=samples)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--length", type=int, nargs="+", default=[128, 512, 1024])
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--rounds", type=int, default=5)
    a = p.parse_args()
    if min(*a.length, a.block_size, a.rounds) < 1:
        p.error("length, block-size, and rounds must be positive")
    backend = CudaBackend(device=a.device, graph_mode="off")
    for steps in a.length:
        print(json.dumps(compare(backend, steps, a.block_size, a.rounds)), flush=True)


if __name__ == "__main__":
    main()
