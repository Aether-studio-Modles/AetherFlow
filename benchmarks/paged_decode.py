"""Same-device A/B: shuffled physical pages vs. GPU gather into dense page order.

Both paths call the same attention kernel with device-resident inputs. The
control copies K and V in one vectorized kernel into preallocated buffers and
then attends with an identity page table. Both paths are captured as CUDA
Graphs to reduce launch noise. Each batch has equal nonempty lengths. This
measures neither host transfers nor end-to-end decoding.
"""

import argparse
import json
import statistics

import numpy as np

from aetherflow import Graph, ReferenceBackend
from aetherflow.backends.cuda import CudaBackend
from aetherflow.ir.types import f16, f32, i32


_PACK = r"""
extern "C" __global__ void pack_kv(const uint4* k, const uint4* v, const int* table,
                                    uint4* packed_k, uint4* packed_v, int vectors) {
    const int dst = blockIdx.x;
    const int src = table[dst];
    for (int i = threadIdx.x; i < vectors; i += blockDim.x) {
        packed_k[dst * vectors + i] = k[src * vectors + i];
        packed_v[dst * vectors + i] = v[src * vectors + i];
    }
}
"""


def _case(batch, length, dtype):
    page_size, heads, kvheads, dim = 16, 16, 4, 64
    width = (length + page_size - 1) // page_size
    pages = batch * width
    rng = np.random.default_rng(125)
    pool = (pages, page_size, kvheads, dim)
    host = {
        "q": rng.standard_normal((batch, heads, dim), dtype=np.float32).astype(dtype.np_dtype),
        "k": rng.standard_normal(pool, dtype=np.float32).astype(dtype.np_dtype),
        "v": rng.standard_normal(pool, dtype=np.float32).astype(dtype.np_dtype),
        "table": rng.permutation(pages).astype(np.int32).reshape(batch, width),
        "lengths": np.full(batch, length, dtype=np.int32),
    }
    g = Graph("paged_decode_bench")
    values = [g.input(name, (i32 if x.dtype.kind == "i" else dtype)[x.shape])
              for name, x in host.items()]
    g.output(g.paged_decode(*values, max_len=length), "y")
    return g, host


def _time(cp, stream, graph, repeats):
    begin, end = cp.cuda.Event(), cp.cuda.Event()
    with stream:
        begin.record()
        for _ in range(repeats):
            graph.launch(stream)
        end.record()
    end.synchronize()
    return cp.cuda.get_elapsed_time(begin, end) * 1000 / repeats


def _capture(cp, stream, call):
    pool = cp.cuda.MemoryPool()
    with stream, cp.cuda.using_allocator(pool.malloc):
        # Warm this stream's allocation pool and compile kernels before capture.
        for _ in range(3):
            call()
        stream.synchronize()
        stream.begin_capture()
        try:
            output = call()
        finally:
            graph = stream.end_capture()
    # The captured addresses outlive the Python arrays used during capture.
    return graph, output, pool


def compare(backend, batch, length, dtype, rounds, repeats):
    cp = backend.cp
    graph, host = _case(batch, length, dtype)
    feed = {name: backend.to_array(x) for name, x in host.items()}
    pages = feed["table"].size
    sequential = cp.arange(pages, dtype=cp.int32).reshape(feed["table"].shape)
    packed = dict(feed, k=cp.empty_like(feed["k"]), v=cp.empty_like(feed["v"]), table=sequential)
    pack = backend._kernel("paged_decode_bench_pack", _PACK, "pack_kv")
    vectors = np.int32(feed["k"].nbytes // pages // 16)
    cp.cuda.get_current_stream().synchronize()

    def direct():
        return backend.execute_device(graph, feed)["y"]

    def gather():
        pack((pages,), (128,), (feed["k"], feed["v"], feed["table"], packed["k"], packed["v"], vectors))
        return backend.execute_device(graph, packed)["y"]

    oracle = ReferenceBackend().execute(graph, host)["y"]
    stream = backend.new_stream()
    direct_graph, direct_y, direct_pool = _capture(cp, stream, direct)
    gather_graph, gather_y, gather_pool = _capture(cp, stream, gather)
    variants = (("direct", direct_graph), ("gather", gather_graph))
    for captured, output in ((direct_graph, direct_y), (gather_graph, gather_y)):
        captured.launch(stream)
        stream.synchronize()
        np.testing.assert_allclose(backend.to_host(output), oracle, atol=5e-4, rtol=2e-3)

    samples = {"direct": [], "gather": []}
    for r in range(rounds):
        for label, captured in (variants if r % 2 == 0 else variants[::-1]):
            samples[label].append(_time(cp, stream, captured, repeats))
    direct_us = statistics.median(samples["direct"])
    gather_us = statistics.median(samples["gather"])
    info = backend.device_info()
    return dict(device=info.name, runtime=info.runtime, mode="cuda_graph", batch=batch, length=length, dtype=dtype.value,
                direct_us=direct_us, gather_us=gather_us, speedup=gather_us / direct_us,
                gathered_bytes=host["k"].nbytes + host["v"].nbytes, samples_us=samples,
                direct_pool_bytes=direct_pool.total_bytes(), gather_pool_bytes=gather_pool.total_bytes())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    p.add_argument("--length", type=int, nargs="+", default=[128, 2048, 8192])
    p.add_argument("--dtype", choices=("f16", "f32"), default="f16")
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--repeats", type=int, default=100)
    args = p.parse_args()
    if min(*args.batch, *args.length, args.rounds, args.repeats) <= 0:
        p.error("batch, length, rounds, and repeats must be positive")
    backend = CudaBackend(device=args.device)
    for batch in args.batch:
        for length in args.length:
            print(json.dumps(compare(backend, batch, length, f16 if args.dtype == "f16" else f32,
                                     args.rounds, args.repeats)), flush=True)


if __name__ == "__main__":
    main()
