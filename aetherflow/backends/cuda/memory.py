"""捕获期的显存规划：一个可复用的 arena。

为什么需要它：CUDA Graph 把每个 kernel 用到的缓冲地址写死，所以中间张量的
复用必须发生在**捕获期间**。

注意收益归属：实测（medium，24 节点 Block）不回收时 cupy 池占 70 MiB，
执行器按生命周期回收后降到 34 MiB——**那 2 倍是 liveness 的功劳**；arena
再省到 32 MiB，只多 6%，峰值 20 MiB。所以 arena 的价值不是"更省"，而是
**确定性布局 + 可报告的峰值**（``peak_live_bytes``），这两样池给不出。

arena 只做三件事：从大块里切、释放时按地址合并相邻块、需要时再申请大块。
分配是确定性的（best-fit），所以同一张图的捕获结果可复现。

回收点由执行器显式给出（``Executor.dispose`` -> ``CudaBackend.dispose``）：
cupy 14 的 ``MemoryPointer`` 析构时不会回调自定义分配器，靠引用计数回收
在这条路径上不成立。因此被复用的块一定已经被"最后一次使用"过了。
"""

from __future__ import annotations

import bisect
from typing import List, Tuple

from ..contracts import MemoryStats

_ALIGN = 512  # cudaMalloc 至少 256 字节对齐，取 512 兼顾 float4 向量化访问
_CHUNK_START = 1 << 20  # 小块粒度 1 MiB
_CHUNK_MAX = 8 << 20  # 单块上限；更大的请求按精确大小单独成块


def _align_up(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


class Arena:
    """给捕获期用的确定性分配器。

    与 ``cupy.cuda.MemoryPool`` 的区别只有两点，但都直接影响峰值：

    - 释放时立即合并相邻空闲块（避免碎裂）
    - 不做按 size 分桶，统一 best-fit（小块不会占住大块）
    """

    def __init__(self, cp, chunk_bytes: int = _CHUNK_START) -> None:
        self._cp = cp
        self._chunk_bytes = chunk_bytes
        self._chunks: List[object] = []
        self._raw: List[int] = []  # 驱动返回的裸指针，用于最终释放
        self._bump: List[int] = []  # 每个 chunk 下一个可切的位置
        self._free: List[Tuple[int, int]] = []  # (绝对地址, 字节数)，按地址升序
        self._live: dict = {}  # 绝对地址 -> 字节数
        self._aliases: dict = {}  # 绝对地址 -> 还活着的视图数
        self._pending: set = set()  # 源数组已请求回收、还在等视图死掉的地址
        self._live_bytes = 0  # 当前存活字节数
        self._peak = 0  # 历史峰值存活字节数，即这张图的显存下界
        self._requested = 0  # 累计请求量（未去重），用于对照复用效果

    # ------------------------------------------------------------------

    @property
    def total_bytes(self) -> int:
        """向驱动实际申请的总字节（这是真实的显存占用）。"""
        return sum(c.size for c in self._chunks)

    @property
    def live_bytes(self) -> int:
        """当前仍在使用中的字节数。"""
        return self._live_bytes

    @property
    def peak_live_bytes(self) -> int:
        """历史峰值存活字节：任何分配器都绕不过去的下界。"""
        return self._peak

    @property
    def requested_bytes(self) -> int:
        """累计请求量（不做复用的话需要这么多）。"""
        return self._requested

    def reset_stats(self) -> None:
        """把峰值与请求量清零。预热轮与捕获轮跑的是同一张图，
        只统计后者才不会被算成两倍。"""
        self._requested = 0
        self._peak = self._live_bytes

    def stats(self) -> MemoryStats:
        """当前状态快照，供 ``MemoryContract`` 上报。"""
        return MemoryStats(
            live_bytes=self._live_bytes,
            peak_bytes=self._peak,
            total_bytes=self.total_bytes,
            requested_bytes=self._requested,
        )

    # ------------------------------------------------------------------

    def malloc(self, size: int):
        size = _align_up(max(int(size), 1))
        self._requested += size

        for i, (addr, avail) in enumerate(self._free):
            if avail >= size:
                self._free.pop(i)
                if avail > size:
                    bisect.insort(self._free, (addr + size, avail - size))
                return self._take(addr, size)

        # 没有合适的空闲块：从最后一块的尾部切，避免每次都新申请
        if self._chunks:
            idx = len(self._chunks) - 1
            chunk = self._chunks[idx]
            off = self._bump[idx]
            if off + size <= chunk.size:
                self._bump[idx] = off + size
                return self._take(chunk.ptr + off, size, idx)

        chunk = self._new_chunk(size)
        idx = len(self._chunks) - 1
        self._bump[idx] = size
        return self._take(chunk.ptr, size, idx)

    def alias(self, ptr) -> None:
        """登记一次"视图引用了这段地址"。

        ``reshape``、恒等 ``transpose`` 返回的是零拷贝视图，和源数组共用同一段
        内存。执行器按"最后一次被读"回收的是**源数组**，而那个视图这时还在
        环境里等着被读——直接还块，replay 就会读到别人写进去的数据。

        所以源地址的回收要等最后一个视图死掉：这里只记账，真正的回收点仍然
        由执行器给出（``dispose`` 里调 ``free`` / ``unalias``）。
        """
        addr = int(ptr)
        self._aliases[addr] = self._aliases.get(addr, 0) + 1

    def unalias(self, ptr) -> None:
        """视图死亡：扣掉一次引用；归零且源数组已经请求过回收就真正还块。"""
        addr = int(ptr)
        count = self._aliases.get(addr, 0)
        if count <= 0:
            return
        if count > 1:
            self._aliases[addr] = count - 1
            return
        self._aliases.pop(addr, None)
        if addr in self._pending:
            self._pending.remove(addr)
            self._release(addr)

    def free(self, ptr, size=None) -> None:
        addr = ptr.ptr
        if self._aliases.get(addr):
            self._pending.add(addr)  # 视图还按这段地址读，等最后一个死掉
            return
        self._release(addr)

    def pending(self, ptr) -> bool:
        """这段地址是否"已请求回收、但还等着视图死掉"。

        整段切片与同形状 ``reshape`` 返回的**就是源数组本身**：执行器却按
        两条 Value 各 dispose 一次。第一次走 ``free`` 记成待回收，第二次
        要能认出来"这是同一段地址的另一份引用"，扣掉视图记账而不是再 free
        一次——否则这份记账永远归不了零，地址挂到捕获结束。
        """
        return int(ptr) in self._pending

    def _release(self, addr: int) -> None:
        got = self._live.pop(addr, None)
        if got is None:  # 不是本 arena 发出的指针，忽略（例如 slice 出的视图）
            return
        self._live_bytes -= got
        bisect.insort(self._free, (addr, got))
        self._coalesce()

    def _coalesce(self) -> None:
        """把**同一个 chunk 内**地址相邻的空闲块合成一块。

        不能跨 chunk 合并：两个 chunk 是两次独立的 ``cudaMalloc``，地址
        相邻只是虚拟地址的巧合。合并后 best-fit 会切出跨越分配边界的
        ``MemoryPointer``，D2D 拷贝按分配校验范围就会失败（HIP 侧实测为
        code=1，见 ``hip/memory.py`` 的同名方法）。
        """
        merged: List[Tuple[int, int]] = []
        for a, n in self._free:
            if (
                merged
                and merged[-1][0] + merged[-1][1] == a
                and self._chunk_of(merged[-1][0]) == self._chunk_of(a)
            ):
                merged[-1] = (merged[-1][0], merged[-1][1] + n)
            else:
                merged.append((a, n))
        self._free = merged

    # ------------------------------------------------------------------

    def _new_chunk(self, size: int):
        """向驱动直接申请一块，绕开当前 allocator（否则会递归回自己）。"""
        # 块大小只跟本次请求与小块粒度有关，**不随历史翻倍**：翻倍会让
        # 小块图（small）也背上 8 MiB 的块，实测多占 6.5 MiB。
        capacity = max(size, min(self._chunk_bytes, _CHUNK_MAX))
        ptr = self._cp.cuda.runtime.malloc(capacity)
        self._raw.append(ptr)
        # 存 BaseMemory（UnownedMemory），MemoryPointer 只能以它为 base
        self._chunks.append(self._cp.cuda.UnownedMemory(ptr, capacity, self))
        self._bump.append(0)  # 已用字节数，由调用方按本次切走的量推进
        return self._chunks[-1]

    def _take(self, addr: int, size: int, chunk_index: int | None = None):
        """切出 [addr, addr+size)，返回 cupy 的 MemoryPointer。"""
        if chunk_index is None:
            chunk_index = self._chunk_of(addr)
        chunk = self._chunks[chunk_index]
        self._live[addr] = size
        self._live_bytes += size
        if self._live_bytes > self._peak:
            self._peak = self._live_bytes
        # cupy 的 MemoryPointer 只接受 (mem, offset)，size 由 mem 推出；
        # 我们的 free 用 _live 查表，不依赖它传回的 size。
        return self._cp.cuda.MemoryPointer(chunk, addr - chunk.ptr)

    def _chunk_of(self, addr: int) -> int:
        for i, chunk in enumerate(self._chunks):
            if chunk.ptr <= addr < chunk.ptr + chunk.size:
                return i
        raise RuntimeError(f"地址 0x{addr:x} 不属于本 arena 的任何一块")

    def close(self) -> None:
        cp = self._cp
        for ptr in self._raw:
            cp.cuda.runtime.free(ptr)
        self._raw.clear()
        self._chunks.clear()
        self._bump.clear()
        self._free.clear()
        self._live.clear()
        self._aliases.clear()
        self._pending.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # 解释器退出时 CUDA 上下文可能已经没了
            pass
