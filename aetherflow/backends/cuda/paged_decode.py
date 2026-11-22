"""CUDA single-query attention over a device-resident KV page pool."""

import numpy as np

from ...ir.ops import paged_decode_options
from ..base import BackendCapabilityError
from . import codegen


_THREADS = 128
_MAX_SPLITS = 256

# Every block owns one query head and a contiguous slice of logical positions.
# Only the per-tile softmax state reaches global memory, never gathered K/V.
_SOURCE = r"""
%PRELUDE%

extern "C" __global__ void paged_decode_tiles(
    const %T%* q, const %T%* k, const %T%* v, const int* table, const int* lengths,
    const %T%* current_k, const %T%* current_v,
    %T%* out, float* workspace, int heads, int groups, int kvheads, int dim,
    int page_size, int width, int pool, int limit, int window, float scale) {
    constexpr int TILE = @TILE@;
    constexpr bool CURRENT = @CURRENT@;
    const int row = blockIdx.x, batch = row / heads, kh = (row % heads) / groups;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int split = blockIdx.y, splits = gridDim.y;
    const int history = max(0, min(lengths[batch], limit));
    const int end = history + CURRENT;
    const int start = max(0, end - window);
    const int span = end - start;
    // Partition by logical tiles, allowing more than one tile per split on very long rows.
    const int tiles = span / TILE + (span % TILE != 0);
    const int first_tile = (int)((long long)tiles * split / splits);
    const int last_tile = (int)((long long)tiles * (split + 1) / splits);
    float* state = splits == 1 ? nullptr : workspace + ((long long)row * splits + split) * (dim + 2);
    const long long base = (long long)row * dim;

    // The current token lives in a separate tensor, not a physical page.
    const long long current_offset = ((long long)batch * kvheads + kh) * dim;
    __shared__ long long addr[128];
    __shared__ float weights[128], scratch[4], stats[4];
    float part_max = 0.0f, part_sum = 0.0f;
    float numer[2] = {0.0f, 0.0f};
    for (int tile = first_tile; tile < last_tile; ++tile) {
        const int first = start + tile * TILE;
        const int count = min(TILE, end - first);
        long long offset = -1;
        if (tid < count) {
            const int pos = first + tid;
            if (CURRENT && pos == history) offset = -2;
            else {
                const int p = table[(long long)batch * width + pos / page_size];
                if (p >= 0 && p < pool)
                    offset = (((long long)p * page_size + pos % page_size) * kvheads + kh) * dim;
            }
        }
        addr[tid] = offset;
        weights[tid] = __int_as_float(0xff800000);
        __syncthreads();

        for (int i = warp; i < count; i += 4) {
            const long long j = addr[i];
            float dot = 0.0f;
            if (j != -1) {
                const %T%* key = CURRENT && j == -2 ? current_k + current_offset : k + j;
                for (int d = lane; d < dim; d += 32)
                    dot = fmaf(%LD%(q[base + d]), %LD%(key[d]), dot);
                for (int step = 16; step; step /= 2)
                    dot += __shfl_xor_sync(0xffffffffu, dot, step);
            }
            if (lane == 0 && j != -1) weights[i] = dot * scale;
        }
        __syncthreads();
        float top = weights[tid];
        for (int step = 16; step; step /= 2)
            top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, step));
        if (lane == 0) scratch[warp] = top;
        __syncthreads();
        top = fmaxf(fmaxf(scratch[0], scratch[1]), fmaxf(scratch[2], scratch[3]));
        float w = offset != -1 ? expf(weights[tid] - top) : 0.0f;
        weights[tid] = w;
        for (int step = 16; step; step /= 2)
            w += __shfl_xor_sync(0xffffffffu, w, step);
        if (lane == 0) stats[warp] = w;
        __syncthreads();
        const float total = (stats[0] + stats[1]) + (stats[2] + stats[3]);
        const float merged = part_sum == 0.0f ? top : fmaxf(part_max, top);
        const float old_scale = part_sum == 0.0f ? 0.0f : expf(part_max - merged);
        const float new_scale = total == 0.0f ? 0.0f : expf(top - merged);
        if (total != 0.0f) {
            for (int d = tid, j = 0; d < dim; d += 128, ++j) {
                float acc = 0.0f;
                for (int i = 0; i < count; ++i) {
                    const long long j = addr[i];
                    if (j != -1) {
                        const %T%* value = CURRENT && j == -2 ? current_v + current_offset : v + j;
                        acc = fmaf(weights[i], %LD%(value[d]), acc);
                    }
                }
                numer[j] = numer[j] * old_scale + acc * new_scale;
            }
            part_sum = part_sum * old_scale + total * new_scale;
            part_max = merged;
        }
        __syncthreads();
    }
    for (int d = tid, j = 0; d < dim; d += 128, ++j) {
        if (state) state[d] = numer[j];
        else out[base + d] = %ST%(part_sum == 0.0f ? 0.0f : numer[j] / part_sum);
    }
    if (state && tid == 0) {
        state[dim] = part_max;
        state[dim + 1] = part_sum;
    }
}

extern "C" __global__ void paged_decode_merge(const float* workspace, %T%* out, int dim, int splits) {
    const long long row = blockIdx.x;
    const float* src = workspace + row * splits * (dim + 2);
    __shared__ float scales[256];
    __shared__ float sum;
    if (threadIdx.x == 0) {
        float top = __int_as_float(0xff800000);
        for (int i = 0; i < splits; ++i) {
            const float* state = src + i * (dim + 2);
            if (state[dim + 1] != 0.0f) top = fmaxf(top, state[dim]);
        }
        sum = 0.0f;
        for (int i = 0; i < splits; ++i) {
            const float* state = src + i * (dim + 2);
            scales[i] = state[dim + 1] == 0.0f ? 0.0f : expf(state[dim] - top);
            sum = fmaf(scales[i], state[dim + 1], sum);
        }
    }
    __syncthreads();
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < splits; ++i) acc = fmaf(scales[i], src[i * (dim + 2) + d], acc);
        out[row * dim + d] = %ST%(sum == 0.0f ? 0.0f : acc / sum);
    }
}
"""


def _tile_size(rows, span):
    # Smaller tiles increase parallelism when few query heads can occupy the device.
    if span <= 32 or (rows <= 16 and span <= 128):
        return 32
    if span <= 128 or (rows <= 16 and span <= 2048):
        return 64
    return 128


def launch_paged_decode(backend, args, attrs):
    cp = backend.cp
    q, k, v, table, lengths = args[:5]
    current = len(args) == 7
    batch, heads, dim = map(int, q.shape)
    pool, page_size, kvheads, _ = map(int, k.shape)
    width = int(table.shape[1])
    window, limit, scale = paged_decode_options(attrs, dim, width * page_size)
    rows = batch * heads
    int_limit = np.iinfo(np.int32).max
    if dim > 256 or max(rows, pool, page_size, width, limit + current) > int_limit:
        raise BackendCapabilityError("CUDA paged_decode requires head_dim <= 256 and int32 launch dimensions")
    if table.dtype != np.dtype(np.int32) or lengths.dtype != np.dtype(np.int32):
        raise BackendCapabilityError("CUDA paged_decode needs int32 page tables and lengths")
    if limit == 0 and not current:
        return cp.zeros(q.shape, dtype=q.dtype)
    if not all(a.flags.c_contiguous for a in args):
        raise BackendCapabilityError("CUDA paged_decode needs contiguous query, KV pages and metadata")

    span = min(window, limit + current) if window else limit + current
    tile = _tile_size(rows, span)
    splits = min(_MAX_SPLITS, (span + tile - 1) // tile)
    out = cp.empty(q.shape, dtype=q.dtype)
    scratch = cp.empty((rows, splits, dim + 2), dtype=cp.float32) if splits > 1 else out
    kind = backend._kind(q)
    source = lambda: codegen.render(
        _SOURCE.replace("@TILE@", str(tile)).replace("@CURRENT@", str(int(current))), q.dtype
    )
    tiles = backend._kernel(("paged_decode_tiles", kind, tile, current), source, "paged_decode_tiles")
    dims = tuple(map(np.int32, (heads, heads // kvheads, kvheads, dim, page_size, width,
                                pool, limit, span)))
    current_args = tuple(args[5:]) if current else (q, q)
    tiles((rows, splits), (_THREADS,), (q, k, v, table, lengths, *current_args,
                                       out, scratch, *dims, np.float32(scale)))
    if splits > 1:
        merge = backend._kernel(("paged_decode_merge", kind), source, "paged_decode_merge")
        merge((rows,), (_THREADS,), (scratch, out, np.int32(dim), np.int32(splits)))
        backend.dispose(scratch)
    return out
