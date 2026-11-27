"""分页 KV cache：把每条序列的键 / 值按**固定大小的页**存放，而不是按最大长度预留。

这一层回答的问题是"多条长度不同的序列，怎么共享一块固定的 K/V 显存"，而不是
"怎么算 attention"。它不新增算子、不碰设备：

- 物理存储是每层一份 ``(num_blocks, block_size, num_heads, head_dim)`` 的数组；
- 每条序列只持有一张**块表**（逻辑页号 → 物理页号），页从一张空闲链上领；
- 对外只有"追加 / 取回 / 回滚 / 回收 / 换出 / 换入 / 分叉"七个动作，取回时按
  块表拼成连续张量。

五条结构性事实（都是"不满足就必须显式报错"的那类）：

1. **追加是全有或全无的。** 一次追加需要的新页在写入前一次性领齐；空闲页不够
   就抛 :class:`PagedCacheFullError`，**不留下半条序列**。留下半段比直接失败更
   难排查：长度对不上，后续每一步都算错，却不会报错。
2. **页号只在块表里有意义。** 页被回收后可以立刻发给另一条序列，因此调用方
   不能缓存 ``gather`` 之外的东西；``gather`` 返回的是**副本**，不是视图。
3. **回滚按页对齐回收。** ``truncate`` 只释放长度之外的整页，页内残留数据不
   清零——它永远在 ``length`` 之外，``gather`` 不会返回。清零是纯浪费。
4. **换出只搬数据，不改长度。** ``swap_out`` 把整条序列的 K/V 拷一份到主机、
   释放物理页；``swap_in`` 重新领页并写回，页号可能与换出前不同，数值不变。
   换出期间块表为空，所以只允许查长度，读写一律显式要求先换入。
5. **共享前缀是写时复制的。** ``fork`` 让子序列指向父序列的同一批物理页，每页
   的引用计数加一；任何一方之后写到这些页时先**拆页**（领新页 + 整页拷贝），
   所以追加 / 回滚不会穿透到另一方。页只在最后一个持有者放手时回到空闲链，
   ``used_blocks`` 也只算一次。

策略面由 :class:`PagedCacheManager` 提供：容量不足时按最近最少使用（LRU）自动
换出**未被 pin 的**序列，被换出的序列再次被访问时自动换入。它不改变上面这些
存储语义，只决定"谁被换出、什么时候换入"。

**这一层不负责数值算子的实现。** 它只提供"按块表取回一段连续 K/V"的存储语义，
真正的 attention 由 :mod:`aetherflow.models.cached_attention` 用 IR 表达。当前
语言模型主干是 Mamba（固定大小的递归状态，没有随上下文增长的 KV），所以这里
的分页缓存**没有**接进 ``generate``；接进去的是一个显式的增量注意力解码器。

**投机解码的回滚不靠这里。** 递归状态无法按 token 截断，投机解码用的是状态
快照（见 :mod:`aetherflow.models.speculative`）；``truncate`` 面向前缀缓存与
注意力路径。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

__all__ = [
    "PagedCacheFullError",
    "PagedCacheManager",
    "PagedKVCache",
]


class PagedCacheFullError(RuntimeError):
    """空闲页不足以完成一次追加。

    这是**容量**条件而不是参数错误，所以刻意不继承 :class:`ValueError`：
    调用方该做的是缩短序列、回收别的会话，或者调大 ``num_blocks``。
    """


def _positive_int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{what} 必须是 int，得到 {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{what} 必须 >= 1，得到 {value}")
    return value


def _pages_for(tokens: int, block_size: int) -> int:
    """``tokens`` 个 token 需要几页（``tokens=0`` 时为 0）。"""
    return (tokens + block_size - 1) // block_size


class PagedKVCache:
    """按页分配的多序列 K/V 存储。

    每一层各自一份物理数组，页号在各层之间**共用同一张块表**（同一序列的第 i
    页在每一层都落在同一个物理页号上），这样上层只需要维护一张表。

    ``dtype`` 只影响存储，不参与任何计算；写入时按它转换，读取时原样返回。
    """

    def __init__(
        self,
        *,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype=np.float32,
    ) -> None:
        self.num_blocks = _positive_int(num_blocks, "num_blocks")
        self.block_size = _positive_int(block_size, "block_size")
        self.num_layers = _positive_int(num_layers, "num_layers")
        self.num_heads = _positive_int(num_heads, "num_heads")
        self.head_dim = _positive_int(head_dim, "head_dim")
        self.dtype = np.dtype(dtype)
        if self.dtype.kind != "f":
            raise TypeError(f"dtype 必须是浮点类型，得到 {self.dtype}")

        self._free: List[int] = list(range(self.num_blocks - 1, -1, -1))
        self._tables: Dict[int, List[int]] = {}
        self._lengths: Dict[int, int] = {}
        # 物理页 → 持有它的序列数。只在 fork 后大于 1；归零才回空闲链。
        self._refs: Dict[int, int] = {}
        self._forks = 0
        # 每层懒分配：只用了 2 层就只占 2 层的显存。
        self._keys: Dict[int, np.ndarray] = {}
        self._values: Dict[int, np.ndarray] = {}
        # 换出到主机的 K/V：rid → (层号表, 每层 keys, 每层 values, 长度)。
        # 只记录**已经分配过**的层，未用过的层换出后仍然不占空间。
        self._swapped: Dict[
            int, Tuple[List[int], List[np.ndarray], List[np.ndarray], int]
        ] = {}
        self._peak_blocks = 0

    # -- 只读视图 -----------------------------------------------------

    @property
    def capacity_tokens(self) -> int:
        """所有页都发出去时能装下的 token 数。"""
        return self.num_blocks * self.block_size

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - len(self._free)

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def peak_blocks(self) -> int:
        """历史最高占用页数（回收不会把它减回去）。"""
        return self._peak_blocks

    @property
    def num_sequences(self) -> int:
        return len(self._tables)

    @property
    def shared_blocks(self) -> int:
        """引用计数大于 1 的物理页数，即被多条序列共享的页。"""
        return sum(1 for count in self._refs.values() if count > 1)

    def exclusive_blocks(self, rid: int) -> int:
        """只被 ``rid`` 引用的物理页数，即换出它最多能腾出的页数。

        换出态返回 0（没有物理页）；未分配抛 ``KeyError``。
        """
        self.length(rid)
        if rid in self._swapped:
            return 0
        return sum(1 for page in self._tables[rid] if self._refs[page] == 1)

    def __contains__(self, rid: object) -> bool:
        return rid in self._tables

    def length(self, rid: int) -> int:
        """序列当前的 token 数；未分配时抛 ``KeyError``。"""
        try:
            return self._lengths[rid]
        except KeyError:
            raise KeyError(f"序列 {rid!r} 没有分配页；先 allocate(rid)") from None

    def block_table(self, rid: int) -> Tuple[int, ...]:
        """块表的副本；改它不影响缓存内部状态。"""
        try:
            return tuple(self._tables[rid])
        except KeyError:
            raise KeyError(f"序列 {rid!r} 没有分配页；先 allocate(rid)") from None

    def is_swapped(self, rid: int) -> bool:
        """``rid`` 是否处于换出状态（没有物理页，只有主机副本）。

        换出态仍然是"已分配"（``rid in cache`` 为真、长度可查），但
        ``append`` / ``gather`` 会显式要求先 :meth:`swap_in`。
        """
        return rid in self._swapped

    @property
    def swapped_tokens(self) -> int:
        """换出到主机的 token 总数（按序列长度累加，不是页数）。"""
        return sum(item[3] for item in self._swapped.values())

    def stats(self) -> Dict[str, int]:
        """一眼看清占用：页数、空闲页、页大小、序列数、历史峰值。"""
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "used_blocks": self.used_blocks,
            "free_blocks": self.free_blocks,
            "peak_blocks": self._peak_blocks,
            "sequences": self.num_sequences,
            "shared_blocks": self.shared_blocks,
            "forks": self._forks,
            "swapped_sequences": len(self._swapped),
            "swapped_tokens": self.swapped_tokens,
        }

    # -- 页的引用计数 -------------------------------------------------

    def _take_page(self) -> int:
        """从空闲链领一页，引用计数置 1。调用方必须先确认空闲页够。"""
        page = self._free.pop()
        self._refs[page] = 1
        return page

    def _give_back(self, page: int) -> bool:
        """放手一页：引用计数减 1，降到 0 才真正回到空闲链。

        返回 ``True`` 表示这页确实腾出来了（可以再分给别人）；返回 ``False``
        表示还有别的序列引用它，什么都没腾出。
        """
        count = self._refs[page] - 1
        if count:
            self._refs[page] = count
            return False
        del self._refs[page]
        self._free.append(page)
        return True

    # -- 生命周期 -----------------------------------------------------

    def allocate(self, rid: int) -> None:
        """为 ``rid`` 建一张空块表；重复分配是错误（会覆盖已有序列）。"""
        if isinstance(rid, bool) or not isinstance(rid, int):
            raise TypeError(f"rid 必须是 int，得到 {type(rid).__name__}")
        if rid in self._tables:
            raise ValueError(f"序列 {rid} 已经分配；先 free(rid) 再重用这个 id")
        self._tables[rid] = []
        self._lengths[rid] = 0

    def free(self, rid: int) -> bool:
        """回收整条序列的页；返回 ``True`` 表示这次确实回收了。

        未分配的 id 返回 ``False``——回收是幂等的，重复回收不是错误。
        """
        table = self._tables.pop(rid, None)
        if table is None:
            return False
        self._lengths.pop(rid, None)
        self._swapped.pop(rid, None)  # 换出态没有物理页，只丢弃主机副本
        for block in table:
            self._give_back(block)
        return True

    def reset(self, rid: int) -> int:
        """把序列长度清零并放手全部页，但保留分配；返回**实际腾出**的页数。

        共享页只减引用，还有别的序列引用它时不算腾出。
        """
        self.length(rid)  # 未分配时显式抛 KeyError
        if self._swapped.pop(rid, None) is not None:
            # 换出态：页已在换出时归还，这里只丢弃主机副本。
            self._lengths[rid] = 0
            return 0
        table = self._tables[rid]
        released = 0
        for block in table:
            if self._give_back(block):
                released += 1
        self._tables[rid] = []
        self._lengths[rid] = 0
        return released

    def truncate(self, rid: int, length: int) -> int:
        """把序列缩短到 ``length`` 个 token，返回**实际腾出**的页数。

        ``length`` 不能超过当前长度（伸长不是回滚，应该走 ``append``）。页内的
        残留数据不清零：它一定落在新长度之外，``gather`` 不会返回。被别的序列
        共享的页只减引用，不算腾出。
        """
        current = self.length(rid)
        if isinstance(length, bool) or not isinstance(length, int):
            raise TypeError(f"length 必须是 int，得到 {type(length).__name__}")
        if length < 0:
            raise ValueError(f"length 不能为负，得到 {length}")
        if length > current:
            raise ValueError(
                f"truncate 只能缩短：当前 {current}，要求 {length}；"
                "要追加请用 append"
            )
        if rid in self._swapped:
            # 换出态没有物理页；直接裁剪主机副本，不涉及页回收。
            if length < current:
                layers, keys, values, _ = self._swapped[rid]
                self._swapped[rid] = (
                    layers,
                    [item[:length].copy() for item in keys],
                    [item[:length].copy() for item in values],
                    length,
                )
                self._lengths[rid] = length
            return 0
        table = self._tables[rid]
        keep = _pages_for(length, self.block_size)
        freed = table[keep:]
        del table[keep:]
        released = 0
        for block in freed:
            if self._give_back(block):
                released += 1
        self._lengths[rid] = length
        return released

    def fork(self, parent: int, child: int) -> None:
        """把 ``parent`` 当前的块表整份共享给 ``child``，走写时复制。

        子序列与父序列指向同一批物理页，页的引用计数加一，``used_blocks`` 与
        ``shared_blocks`` 都不增加页数（后者反而变大）。任何一方之后写这些页
        时先拆页（新页 + 整页拷贝），所以追加 / 回滚不会穿透到另一方。

        典型用法是"同一段前缀 + 多条后继"：多采样、束搜索、同一个系统提示下
        的多条请求。只接受驻留序列——换出态没有物理页，先 ``swap_in``；
        ``child`` 必须是一个没分配过的 id。
        """
        self.length(parent)  # 未分配时显式抛 KeyError
        if parent in self._swapped:
            raise ValueError(
                f"序列 {parent} 已换出到主机；先 swap_in 再 fork"
            )
        if isinstance(child, bool) or not isinstance(child, int):
            raise TypeError(f"child 必须是 int，得到 {type(child).__name__}")
        if child in self._tables:
            raise ValueError(f"序列 {child} 已经分配；fork 需要一个新的 id")
        table = list(self._tables[parent])
        for page in table:
            self._refs[page] += 1
        self._tables[child] = table
        self._lengths[child] = self._lengths[parent]
        self._forks += 1

    # -- 换出与换入 ---------------------------------------------------

    def swap_out(self, rid: int) -> int:
        """把整条序列的 K/V 拷到主机暂存并放手物理页，返回**实际腾出**的页数。

        换出后长度不变、仍然是"已分配"，但块表为空：``append`` / ``gather``
        会显式要求先 :meth:`swap_in`，只有 ``length`` / ``is_swapped`` 可查。
        暂存的是副本，且只覆盖**已经分配过**的层——从没用过的层换出后仍然
        不占空间。被别的序列共享的页只减引用：一条全共享的序列换出后一页也
        腾不出（返回 0），但它的主机副本照样独立存在。

        这一层不做真正的 D2H：缓存本身就在主机内存，"搬到主机"是把页归还
        空闲链、把副本挂到序列上。真机上的显存换出由上层按同一语义执行。
        """
        current = self.length(rid)
        if rid in self._swapped:
            raise ValueError(f"序列 {rid} 已经换出；重复换出会覆盖主机副本")
        layers = [layer for layer in range(self.num_layers) if layer in self._keys]
        keys: List[np.ndarray] = []
        values: List[np.ndarray] = []
        for layer in layers:
            key, value = self.gather(rid, layer)
            keys.append(key)
            values.append(value)
        table = self._tables[rid]
        released = 0
        for block in table:
            if self._give_back(block):
                released += 1
        self._tables[rid] = []
        self._swapped[rid] = (layers, keys, values, current)
        return released

    def swap_in(self, rid: int) -> int:
        """把换出的序列换回物理页，返回本次领取的页数。

        页不够时抛 :class:`PagedCacheFullError`，序列保持换出态（主机副本
        原样保留，腾出页后可以再试）。换入的页号与换出前无关：块表按当前
        空闲链重新分配。
        """
        self.length(rid)  # 未分配时显式抛 KeyError
        item = self._swapped.get(rid)
        if item is None:
            raise ValueError(f"序列 {rid} 没有换出；swap_in 只接受换出态")
        layers, keys, values, length = item
        need = _pages_for(length, self.block_size)
        if need > len(self._free):
            raise PagedCacheFullError(
                f"换入序列 {rid} 需要 {need} 个页，空闲页只有 "
                f"{len(self._free)}（block_size={self.block_size}）；"
                "序列保持换出态，腾出页后可重试"
            )
        table = [self._take_page() for _ in range(need)]
        for layer, key, value in zip(layers, keys, values):
            store_keys, store_values = self._layer_arrays(layer)
            for position in range(length):
                page, slot = divmod(position, self.block_size)
                physical = table[page]
                store_keys[physical, slot] = key[position]
                store_values[physical, slot] = value[position]
        self._tables[rid] = table
        del self._swapped[rid]
        self._peak_blocks = max(self._peak_blocks, self.used_blocks)
        return need

    # -- 写入与读取 ---------------------------------------------------

    def _layer_arrays(self, layer: int) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise TypeError(f"layer 必须是 int，得到 {type(layer).__name__}")
        if not 0 <= layer < self.num_layers:
            raise ValueError(f"layer={layer} 越界：[0, {self.num_layers})")
        if layer not in self._keys:
            shape = (self.num_blocks, self.block_size, self.num_heads, self.head_dim)
            self._keys[layer] = np.zeros(shape, dtype=self.dtype)
            self._values[layer] = np.zeros(shape, dtype=self.dtype)
        return self._keys[layer], self._values[layer]

    def _check_pair(self, keys: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        keys = np.asarray(keys)
        values = np.asarray(values)
        want = (self.num_heads, self.head_dim)
        for name, arr in (("keys", keys), ("values", values)):
            if arr.ndim != 3 or arr.shape[1:] != want:
                raise ValueError(
                    f"{name} 形状必须是 (T, {want[0]}, {want[1]})，得到 {arr.shape}"
                )
        if keys.shape != values.shape:
            raise ValueError(
                f"keys 与 values 形状必须一致，得到 {keys.shape} 与 {values.shape}"
            )
        if keys.shape[0] < 1:
            raise ValueError("append 至少要写入一个 token；空追加请直接返回")
        return keys, values

    def append(self, rid: int, keys, values) -> int:
        """为**每一层**追加同样的 ``T`` 个位置，返回新的长度。

        ``keys`` / ``values`` 是长度为 ``num_layers`` 的序列，每项形状
        ``(T, num_heads, head_dim)``——即**单个序列**的一段 K/V。所有层必须
        等长：它们描述的是同一批 token。页不够时抛 :class:`PagedCacheFullError`，
        且追加之前的状态与块表原样保留。

        ``num_layers=1`` 时也接受单个 ``(T, H, D)`` 数组，省掉一层列表。
        """
        current = self.length(rid)
        if rid in self._swapped:
            raise ValueError(
                f"序列 {rid} 已换出到主机；先 swap_in 再追加"
            )
        if self.num_layers == 1 and isinstance(keys, np.ndarray):
            key_list: List[object] = [keys]
            value_list: List[object] = [values]
        else:
            try:
                key_list = list(keys)
                value_list = list(values)
            except TypeError:
                raise TypeError(
                    "keys / values 必须是按层给的序列（每项 (T, H, D)）"
                ) from None
        if len(key_list) != self.num_layers or len(value_list) != self.num_layers:
            raise ValueError(
                f"keys / values 必须各有 {self.num_layers} 项（每层一项），"
                f"得到 {len(key_list)} / {len(value_list)}"
            )
        pairs = [
            self._check_pair(key_list[layer], value_list[layer])
            for layer in range(self.num_layers)
        ]
        tokens = pairs[0][0].shape[0]
        for layer, (k, _) in enumerate(pairs):
            if k.shape[0] != tokens:
                raise ValueError(
                    f"各层追加长度必须一致：第 0 层 {tokens}，第 {layer} 层 "
                    f"{k.shape[0]}"
                )
        need = _pages_for(current + tokens, self.block_size)
        table = self._tables[rid]
        have = len(table)
        missing = need - have
        # 写时复制：这次会写到的**已有**页若仍被别的序列共享，必须先拆开。
        first = current // self.block_size
        last = (current + tokens - 1) // self.block_size
        cow = [
            page
            for page in range(first, min(last, have - 1) + 1)
            if self._refs[table[page]] > 1
        ]
        if missing + len(cow) > len(self._free):
            extra = f"，另有 {len(cow)} 个共享页需要拆分" if cow else ""
            raise PagedCacheFullError(
                f"追加 {tokens} 个 token 需要 {missing} 个新页{extra}，空闲页只有 "
                f"{len(self._free)}（block_size={self.block_size}）；"
                "缩短序列、回收其它会话，或调大 num_blocks"
            )
        for _ in range(missing):
            table.append(self._take_page())
        for page in cow:
            old = table[page]
            fresh = self._take_page()
            for stores in (self._keys, self._values):
                for store in stores.values():
                    store[fresh] = store[old]
            table[page] = fresh
            self._give_back(old)
        for layer, (keys_layer, values_layer) in enumerate(pairs):
            store_keys, store_values = self._layer_arrays(layer)
            for offset in range(tokens):
                position = current + offset
                page, slot = divmod(position, self.block_size)
                physical = table[page]
                store_keys[physical, slot] = keys_layer[offset]
                store_values[physical, slot] = values_layer[offset]
        new_length = current + tokens
        self._lengths[rid] = new_length
        used = self.used_blocks
        if used > self._peak_blocks:
            self._peak_blocks = used
        return new_length

    def gather(
        self, rid: int, layer: int, *, pad_to: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """按块表取回第 ``layer`` 层的 ``(K, V)``，形状 ``(L, H, D)``。

        ``pad_to`` 给出时补零到该长度（必须 >= ``length`` 且不超过缓存容量），
        供固定形状的增量图使用；补出来的位置一定在 mask 之外，不参与注意力。
        返回的数组是**副本**：页随时可能被别的序列拿走。
        """
        length = self.length(rid)
        if rid in self._swapped:
            raise ValueError(
                f"序列 {rid} 已换出到主机；先 swap_in 再取回"
            )
        if pad_to is None:
            width = length
        else:
            if isinstance(pad_to, bool) or not isinstance(pad_to, int):
                raise TypeError(f"pad_to 必须是 int 或 None，得到 {type(pad_to).__name__}")
            if pad_to < length:
                raise ValueError(
                    f"pad_to={pad_to} 小于当前长度 {length}；补齐不能丢数据"
                )
            if pad_to > self.capacity_tokens:
                raise ValueError(
                    f"pad_to={pad_to} 超过缓存容量 {self.capacity_tokens}"
                )
            width = pad_to
        store_keys, store_values = self._layer_arrays(layer)
        key_out = np.zeros((width, self.num_heads, self.head_dim), dtype=self.dtype)
        value_out = np.zeros_like(key_out)
        table = self._tables[rid]
        for position in range(length):
            page, slot = divmod(position, self.block_size)
            physical = table[page]
            key_out[position] = store_keys[physical, slot]
            value_out[position] = store_values[physical, slot]
        return key_out, value_out

    def __repr__(self) -> str:
        return (
            f"PagedKVCache(blocks={self.num_blocks}, block_size={self.block_size}, "
            f"used={self.used_blocks}, shared={self.shared_blocks}, "
            f"sequences={self.num_sequences})"
        )


class PagedCacheManager:
    """在 :class:`PagedKVCache` 之上决定"谁被换出、什么时候换入"。

    这一层回答的是服务控制面的问题：多条序列共享固定容量的 KV 池，**容量不足时
    怎么办**。它只有一条策略——按最近最少使用（LRU）换出，访问时自动换入：

    - 每次 ``append`` / ``gather`` 都把该序列标成"最近使用"；
    - ``append`` / ``ensure_resident`` 遇到 :class:`PagedCacheFullError`，就从
      **最久未使用**的一条开始换出（跳过自己、已换出的、pin 住的，以及所有页都
      被别的序列共享的——换出它也腾不出页），再重试；
    - 被换出的序列在下一次 ``append`` / ``gather`` 前自动换入，页不够时同样
      先换出别人再重试。

    **共享前缀也按"两条序列"管理。** :meth:`fork` 转发给缓存层做块级共享，两条
    序列各自进出 LRU 顺序；共享页在任一方换出时只减引用，所以"一条全共享的父
    序列先被换出"不会腾出任何容量——这正是 `_evict_one` 要跳过它的原因。

    **pin 的语义是"自动换出时可以跳过"。** 正在解码的序列一旦被换出，下一步就得
    再换入，白白搬一遍数据；服务层应在请求进入解码槽位时 ``pin``、请求结束或
    被挂起时 ``unpin``。显式 :meth:`evict` 不受 pin 约束——那是调用方明确要求。

    **换出是"有损但可恢复"的。** 换出后主机副本仍在缓存里，失败（例如能换的都
    换完仍不够）不会丢数据，只会把 :class:`PagedCacheFullError` 原样抛出；那些
    已被换出的序列保持换出态，等页腾出来可以再换入。这条与 ``append`` 的"全有
    或全无"一致：宁可失败，也不留下半条序列。

    **这一层同样不做真正的 D2H。** 真机上的显存搬运由 ``on_swap_out`` /
    ``on_swap_in`` 两个钩子接上：它们在页已归还 / 已重新领到时被调用，参数是
    ``(rid, pages)``，适合用来释放或重建设备侧数组。钩子抛出的异常会向上传播。
    """

    def __init__(
        self,
        cache: PagedKVCache,
        *,
        on_swap_out: Optional[Callable[[int, int], None]] = None,
        on_swap_in: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        if not isinstance(cache, PagedKVCache):
            raise TypeError(
                f"cache 必须是 PagedKVCache，得到 {type(cache).__name__}"
            )
        for name, hook in (("on_swap_out", on_swap_out), ("on_swap_in", on_swap_in)):
            if hook is not None and not callable(hook):
                raise TypeError(f"{name} 必须是可调用对象或 None，得到 {hook!r}")
        self.cache = cache
        self._on_swap_out = on_swap_out
        self._on_swap_in = on_swap_in
        # 迭代顺序就是最近使用顺序：最早使用在最前，touch 把它移到末尾。
        self._order: "OrderedDict[int, None]" = OrderedDict()
        self._pinned: Set[int] = set()
        self._swaps_out = 0
        self._swaps_in = 0

    # -- 只读视图 -----------------------------------------------------

    def __contains__(self, rid: object) -> bool:
        return rid in self.cache

    def length(self, rid: int) -> int:
        return self.cache.length(rid)

    def is_swapped(self, rid: int) -> bool:
        return self.cache.is_swapped(rid)

    @property
    def pinned(self) -> int:
        return len(self._pinned)

    def is_pinned(self, rid: int) -> bool:
        self._require(rid)
        return rid in self._pinned

    def stats(self) -> Dict[str, int]:
        """存储面的统计（同 :meth:`PagedKVCache.stats`）加管理面计数。"""
        info = self.cache.stats()
        info.update(
            {
                "swaps_out": self._swaps_out,
                "swaps_in": self._swaps_in,
                "pinned_sequences": len(self._pinned),
            }
        )
        return info

    # -- 生命周期 -----------------------------------------------------

    def _require(self, rid: int) -> None:
        if rid not in self._order:
            raise KeyError(f"序列 {rid!r} 没有分配页；先 allocate(rid)")

    def allocate(self, rid: int) -> None:
        self.cache.allocate(rid)
        self._order[rid] = None

    def free(self, rid: int) -> bool:
        """回收序列；未分配的 id 返回 ``False``（幂等，与 ``PagedKVCache`` 一致）。"""
        freed = self.cache.free(rid)
        self._order.pop(rid, None)
        self._pinned.discard(rid)
        return freed

    def reset(self, rid: int) -> int:
        self._require(rid)
        return self.cache.reset(rid)

    def truncate(self, rid: int, length: int) -> int:
        self._require(rid)
        self._order.move_to_end(rid)
        return self.cache.truncate(rid, length)

    def fork(self, parent: int, child: int) -> None:
        """把 ``parent`` 的块表共享给新的 ``child``（写时复制）。

        两条序列都进入 LRU 顺序，父序列顺手标成最近使用。共享页在换出时
        只减引用：只要还有一条序列持有，页就不会被别的会话拿走。
        """
        self._require(parent)
        self.cache.fork(parent, child)
        self._order[child] = None
        self._order.move_to_end(parent)

    # -- pin 与显式换出 -----------------------------------------------

    def pin(self, rid: int) -> None:
        """标记 ``rid`` 不被**自动**换出；重复 pin 不是错误。"""
        self._require(rid)
        self._pinned.add(rid)

    def unpin(self, rid: int) -> bool:
        """解除 pin；返回 ``True`` 表示这次确实解除了（重复调用返回 ``False``）。"""
        self._require(rid)
        if rid in self._pinned:
            self._pinned.discard(rid)
            return True
        return False

    def touch(self, rid: int) -> None:
        """把 ``rid`` 标成最近使用；``append`` / ``gather`` 会自动做。"""
        self._require(rid)
        self._order.move_to_end(rid)

    def evict(self, rid: int) -> int:
        """显式换出 ``rid``，返回释放的页数；不受 pin 约束。

        已换出的序列由 :class:`PagedKVCache` 抛 ``ValueError``（重复换出会覆盖
        主机副本），未分配的 id 抛 ``KeyError``。
        """
        self._require(rid)
        pages = self.cache.swap_out(rid)
        self._swaps_out += 1
        if self._on_swap_out is not None:
            self._on_swap_out(rid, pages)
        return pages

    def ensure_resident(self, rid: int) -> int:
        """确保 ``rid`` 有物理页；必要时先换出别人。返回本次换入的页数。

        已经在物理页上的序列返回 0（不产生任何拷贝）；换不进去时抛
        :class:`PagedCacheFullError`，``rid`` 保持换出态。
        """
        self._require(rid)
        self._order.move_to_end(rid)
        if not self.cache.is_swapped(rid):
            return 0
        while True:
            try:
                pages = self.cache.swap_in(rid)
            except PagedCacheFullError:
                if not self._evict_one(exclude=rid):
                    raise
                continue
            self._swaps_in += 1
            if self._on_swap_in is not None:
                self._on_swap_in(rid, pages)
            return pages

    # -- 读写：换出后自动换入，容量不足自动换出 -----------------------

    def append(self, rid: int, keys, values) -> int:
        """追加 token；序列被换出时先换入，页不够时按 LRU 换出别的序列再重试。"""
        self._require(rid)
        self._order.move_to_end(rid)
        if self.cache.is_swapped(rid):
            self.ensure_resident(rid)
        while True:
            try:
                return self.cache.append(rid, keys, values)
            except PagedCacheFullError:
                if not self._evict_one(exclude=rid):
                    raise

    def gather(
        self, rid: int, layer: int, *, pad_to: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """取回第 ``layer`` 层的 K/V；序列被换出时先换入。"""
        self._require(rid)
        self._order.move_to_end(rid)
        if self.cache.is_swapped(rid):
            self.ensure_resident(rid)
        return self.cache.gather(rid, layer, pad_to=pad_to)

    # -- 换出选择 -----------------------------------------------------

    def _evict_one(self, *, exclude: int) -> bool:
        """按 LRU 换出**一条**序列；没有可换的返回 ``False``。

        跳过 ``exclude``（正在等页的那条）、被 pin 的、已经换出的、长度为 0 的
        （换出它也腾不出页），以及**所有页都被别的序列共享**的（换出是白搬，
        页一页也腾不出）。每调用一次至少推进一条，所以调用方的重试循环最多
        迭代"序列数 + 1"次。
        """
        for rid in list(self._order):
            if rid == exclude or rid in self._pinned:
                continue
            if self.cache.is_swapped(rid) or self.cache.length(rid) == 0:
                continue
            if self.cache.exclusive_blocks(rid) == 0:
                continue
            pages = self.cache.swap_out(rid)
            self._swaps_out += 1
            if self._on_swap_out is not None:
                self._on_swap_out(rid, pages)
            return True
        return False

    def __repr__(self) -> str:
        return (
            f"PagedCacheManager({self.cache!r}, pinned={len(self._pinned)}, "
            f"swaps_out={self._swaps_out}, swaps_in={self._swaps_in})"
        )
