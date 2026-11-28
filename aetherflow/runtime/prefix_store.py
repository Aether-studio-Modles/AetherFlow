"""落盘前缀缓存：把 token 序列与每层 K/V 存到磁盘，供新进程首次请求复用。

这一层回答的问题是"进程重启之后，上一次解过的相同前缀能不能直接拿来用"，
而不是"怎么算 attention"。它不新增算子、不碰设备：

- 一个条目就是一个 ``.npz`` 文件，含 ``meta``（UTF-8 JSON 字节）、``tokens``
  （int64）、``keys`` / ``values``（``(num_layers, T, H, D)``，按缓存 dtype 存）；
- 文件名是 ``sha256(model_id + tokens)``，``meta`` 里记同一份摘要，两者对不上
  就当作损坏剔除；zip 成员自带的 CRC 负责数据段的完整性；
- 写入先落同目录临时文件再 ``os.replace``，任何时刻可见的条目都是完整的一份；
- 查询只接受"条目 token 是请求 token 的**严格**前缀"里最长的一条，与
  :class:`aetherflow.models.PagedGenerator` 的内存前缀缓存同一判据。

四条结构性事实（都是"不满足就必须显式拒绝"的那类）：

1. **配置失配不是错误，是不命中。** 模型指纹、形状、dtype、页宽任何一项不同，
   条目都不能用；扫描时跳过并计入 ``corrupt``，而不是抛异常——同一个目录里可能
   混着别的配置留下的文件，那不该让当前进程启动失败。
2. **落盘内容按缓存 dtype 存，不做量化也不做对齐。** 它只是页内容的副本，
   导入方自己决定要不要转换（当前的导入方只接受同一 dtype）。
3. **淘汰按文件 mtime。** 跨进程没有可靠的访问计数，命中时顺手 ``utime`` 就是
   最便宜的"最近使用"。
4. **这一层不做并发控制。** 不同条目是不同文件名，原子替换保证了"要么旧要么新"；
   同一 key 并发写只会互相覆盖成同一份内容。

**它不负责分页。** 导入方拿到的是每层连续的一段 ``(T, H, D)``，怎么装进分页
池由 :class:`aetherflow.runtime.PagedKVCache` 决定。条目长度也不必是页宽的整数倍。
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["DiskPrefixCache"]

# 条目格式版本：字段含义变化时递增，旧条目会被当作配置失配跳过。
_VERSION = 1
_SUFFIX = ".npz"

# 只捕获"这个文件读不出来"的那几类：路径没了、不是 zip、zip 内容坏了、
# 少了成员、meta 不是合法 UTF-8 / JSON（后两者都是 ValueError 的子类）。
_READ_ERRORS = (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile)


def _positive_int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{what} 必须是 int，得到 {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{what} 必须 >= 1，得到 {value}")
    return value


def _tokens_digest(model_id: str, tokens) -> str:
    """``model_id`` 与 token 序列的联合摘要；同 key 的条目名唯一。"""
    hasher = hashlib.sha256()
    hasher.update(model_id.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(np.asarray(tokens, dtype=np.int64).tobytes())
    return hasher.hexdigest()


def _token_list(tokens) -> List[int]:
    if isinstance(tokens, (str, bytes)):
        raise TypeError(f"tokens 必须是整数序列，不能是 {type(tokens).__name__}")
    out = []
    for index, token in enumerate(tokens):
        if isinstance(token, bool) or not isinstance(token, int):
            raise TypeError(
                f"tokens[{index}] 必须是 int，得到 {type(token).__name__}"
            )
        if token < 0:
            raise ValueError(f"tokens[{index}] 不能为负，得到 {token}")
        out.append(token)
    return out


class DiskPrefixCache:
    """一个目录里的前缀条目集合：查询最长严格前缀 / 原子落盘 / 按容量淘汰。

    ``path`` 目录不存在时会创建。``model_id`` 是调用方给的模型指纹（通常是权重
    摘要），同一个目录必须只放同一份配置——不同配置的条目会被扫描跳过并计入
    ``corrupt``，``clear()`` 可以一次性清掉。

    ``max_entries`` 是条目数上限；``max_bytes`` 是磁盘占用上限（``None`` 表示
    不限）。两者同时生效时先按条数淘汰，再按字节数淘汰，都是"最久未使用先走"。
    """

    def __init__(
        self,
        path,
        *,
        model_id: str,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype=np.float32,
        max_entries: int = 64,
        max_bytes: Optional[int] = None,
    ) -> None:
        folder = os.fspath(path)
        if isinstance(folder, bytes):
            folder = os.fsdecode(folder)
        if not folder:
            raise ValueError("path 不能是空字符串")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(f"model_id 必须是非空 str，得到 {model_id!r}")
        self.path = folder
        self.model_id = model_id
        self.block_size = _positive_int(block_size, "block_size")
        self.num_layers = _positive_int(num_layers, "num_layers")
        self.num_heads = _positive_int(num_heads, "num_heads")
        self.head_dim = _positive_int(head_dim, "head_dim")
        self.dtype = np.dtype(dtype)
        if self.dtype.kind != "f":
            raise TypeError(f"dtype 必须是浮点类型，得到 {self.dtype}")
        self.max_entries = _positive_int(max_entries, "max_entries")
        if max_bytes is not None:
            self.max_bytes = _positive_int(max_bytes, "max_bytes")
        else:
            self.max_bytes = None

        os.makedirs(self.path, exist_ok=True)
        # 文件名 → {"tokens": [...], "length": int, "digest": str}。
        # 扫描只读 meta，不碰 K/V 数据段：几十个条目也只是几十次小解压。
        self._entries: Dict[str, Dict[str, object]] = {}
        self._hits = 0
        self._misses = 0
        self._corrupt = 0
        self._stores = 0
        self._evictions = 0
        self._scan()

    # -- 只读视图 -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, tokens: object) -> bool:
        """这个 token 序列是否**恰好**已有一条可用的落盘条目。"""
        if isinstance(tokens, (str, bytes)):
            return False
        try:
            seq = _token_list(tokens)
        except (TypeError, ValueError):
            return False
        if not seq:
            return False
        return os.path.exists(self._entry_path(_tokens_digest(self.model_id, seq)))

    def entries(self) -> List[Dict[str, object]]:
        """当前可用的条目，最近的在前（按文件 mtime，命中会刷新）。"""
        rows = []
        for name, meta in self._entries.items():
            full = os.path.join(self.path, name)
            try:
                info = os.stat(full)
            except OSError:
                size, stamp = 0, 0.0
            else:
                size, stamp = info.st_size, info.st_mtime
            rows.append(
                {
                    "name": name,
                    "tokens": list(meta["tokens"]),
                    "length": meta["length"],
                    "digest": meta["digest"],
                    "size": size,
                    "mtime": stamp,
                }
            )
        rows.sort(key=lambda row: row["mtime"], reverse=True)
        return rows

    def stats(self) -> Dict[str, int]:
        """条目数、字节数、命中/未命中/损坏/落盘/淘汰计数。

        ``max_bytes`` 不限时以 0 表示（与其余字段一样保持纯整数）。
        """
        return {
            "entries": len(self._entries),
            "bytes": self._total_bytes(),
            "hits": self._hits,
            "misses": self._misses,
            "corrupt": self._corrupt,
            "stores": self._stores,
            "evictions": self._evictions,
            "max_entries": self.max_entries,
            "max_bytes": self.max_bytes or 0,
        }

    # -- 扫描与解析 ---------------------------------------------------

    def _entry_path(self, digest: str) -> str:
        return os.path.join(self.path, digest + _SUFFIX)

    def _scan(self) -> None:
        """只读每个条目的 meta 建索引；坏文件与配置失配都跳过。"""
        try:
            names = sorted(os.listdir(self.path))
        except OSError:
            return
        for name in names:
            if not name.endswith(_SUFFIX):
                continue
            meta = self._read_meta(name)
            if meta is None:
                self._corrupt += 1
                continue
            self._entries[name] = meta

    def _same_int(self, raw: Dict[str, object], key: str, want: int) -> bool:
        value = raw.get(key)
        return (
            not isinstance(value, bool)
            and isinstance(value, int)
            and value == want
        )

    def _canonical_meta(
        self, name: str, raw: object
    ) -> Optional[Dict[str, object]]:
        """校验一份 meta；不合法或配置失配返回 ``None``。"""
        if not isinstance(raw, dict):
            return None
        if not self._same_int(raw, "version", _VERSION):
            return None
        if raw.get("model_id") != self.model_id:
            return None
        if not self._same_int(raw, "block_size", self.block_size):
            return None
        if not self._same_int(raw, "num_layers", self.num_layers):
            return None
        if not self._same_int(raw, "num_heads", self.num_heads):
            return None
        if not self._same_int(raw, "head_dim", self.head_dim):
            return None
        if raw.get("dtype") != self.dtype.name:
            return None
        tokens = raw.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            return None
        try:
            seq = _token_list(tokens)
        except (TypeError, ValueError):
            return None
        length = raw.get("length")
        if isinstance(length, bool) or not isinstance(length, int):
            return None
        if length != len(seq):
            return None
        digest = raw.get("digest")
        if not isinstance(digest, str) or digest != _tokens_digest(self.model_id, seq):
            return None
        if name != digest + _SUFFIX:
            return None
        return {"tokens": seq, "length": length, "digest": digest}

    def _read_meta(self, name: str) -> Optional[Dict[str, object]]:
        full = os.path.join(self.path, name)
        try:
            with np.load(full, allow_pickle=False) as data:
                blob = np.asarray(data["meta"]).reshape(-1).astype(np.uint8)
                raw = json.loads(bytes(blob).decode("utf-8"))
        except _READ_ERRORS:
            return None
        return self._canonical_meta(name, raw)

    def _drop(self, name: str) -> None:
        """把条目从索引里剔除并计入损坏；文件本身留给 ``clear()`` 清理。"""
        if self._entries.pop(name, None) is not None:
            self._corrupt += 1

    # -- 查询 ---------------------------------------------------------

    def lookup(self, tokens) -> Optional[Tuple[int, Dict[str, object]]]:
        """最长严格前缀；没有可用条目时返回 ``None``。

        返回 ``(length, payload)``，``payload`` 含 ``tokens`` / ``keys`` /
        ``values``（每层一份 ``(length, H, D)`` 副本）。等长条目不命中——生成
        需要一个尚未解码的位置才能续写。候选按长度从长到短尝试，最长的那条
        读不出来（文件被删 / 截断 / 摘要不符）就继续试次长的。
        """
        seq = _token_list(tokens)
        if not seq:
            return None
        candidates = []
        for name, meta in self._entries.items():
            size = meta["length"]
            if size >= len(seq):
                continue
            if seq[:size] != meta["tokens"]:
                continue
            candidates.append((size, name))
        candidates.sort(reverse=True)
        for size, name in candidates:
            payload = self._read_payload(name)
            if payload is None:
                continue
            self._hits += 1
            self._touch(name)
            return size, payload
        self._misses += 1
        return None

    def _read_payload(self, name: str) -> Optional[Dict[str, object]]:
        meta = self._entries[name]
        length = meta["length"]
        full = os.path.join(self.path, name)
        try:
            with np.load(full, allow_pickle=False) as data:
                tokens = np.asarray(data["tokens"])
                keys = np.asarray(data["keys"])
                values = np.asarray(data["values"])
        except _READ_ERRORS:
            self._drop(name)
            return None
        want = (self.num_layers, length, self.num_heads, self.head_dim)
        ok = (
            tokens.shape == (length,)
            and tokens.dtype == np.int64
            and list(tokens.tolist()) == meta["tokens"]
            and keys.shape == want
            and values.shape == want
            and keys.dtype == self.dtype
            and values.dtype == self.dtype
        )
        if not ok:
            self._drop(name)
            return None
        return {
            "tokens": list(meta["tokens"]),
            "keys": [keys[layer].copy() for layer in range(self.num_layers)],
            "values": [values[layer].copy() for layer in range(self.num_layers)],
        }

    # -- 落盘 ---------------------------------------------------------

    def store(self, tokens, *, keys, values) -> str:
        """把一段前缀与每层 K/V 落盘，返回条目文件名。

        ``keys`` / ``values`` 是按层给的序列，每项形状 ``(T, H, D)``，``T`` 必须
        等于 ``len(tokens)``。同 key 的条目已存在且长度一致时只刷新 mtime，不重写
        ——确定性推理下内容必然相同。写入成功后按容量上限淘汰。
        """
        seq = _token_list(tokens)
        length = len(seq)
        if length < 1:
            raise ValueError("store 至少要有一个 token；空前缀没有复用价值")
        want = (length, self.num_heads, self.head_dim)
        prepared = []
        for label, items in (("keys", keys), ("values", values)):
            try:
                layers = list(items)
            except TypeError:
                raise TypeError(
                    f"{label} 必须是按层给的序列（每项 (T, H, D)）"
                ) from None
            if len(layers) != self.num_layers:
                raise ValueError(
                    f"{label} 必须各有 {self.num_layers} 项（每层一项），"
                    f"得到 {len(layers)}"
                )
            out = []
            for layer, item in enumerate(layers):
                arr = np.ascontiguousarray(item)
                if arr.shape != want:
                    raise ValueError(
                        f"{label}[{layer}] 形状必须是 {want}，得到 {arr.shape}"
                    )
                if arr.dtype.kind != "f":
                    raise TypeError(
                        f"{label}[{layer}] 必须是浮点数组，得到 {arr.dtype}"
                    )
                out.append(arr.astype(self.dtype, copy=False))
            prepared.append(out)

        digest = _tokens_digest(self.model_id, seq)
        name = digest + _SUFFIX
        full = self._entry_path(digest)
        existing = self._entries.get(name)
        if existing is not None and existing["length"] == length:
            self._touch(name)
            return name

        meta = {
            "version": _VERSION,
            "model_id": self.model_id,
            "block_size": self.block_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "dtype": self.dtype.name,
            "length": length,
            "tokens": seq,
            "digest": digest,
        }
        blob = json.dumps(meta, ensure_ascii=False, sort_keys=True).encode("utf-8")
        tmp = f"{full}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            with open(tmp, "wb") as handle:
                np.savez(
                    handle,
                    meta=np.frombuffer(blob, dtype=np.uint8),
                    tokens=np.asarray(seq, dtype=np.int64),
                    keys=np.stack(prepared[0], axis=0),
                    values=np.stack(prepared[1], axis=0),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, full)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

        self._entries[name] = {"tokens": seq, "length": length, "digest": digest}
        self._stores += 1
        self._evict()
        return name

    # -- 淘汰与清理 ---------------------------------------------------

    def _total_bytes(self) -> int:
        total = 0
        for name in self._entries:
            try:
                total += os.stat(os.path.join(self.path, name)).st_size
            except OSError:
                continue
        return total

    def _remove_oldest(self) -> bool:
        """删掉 mtime 最旧的一条；一条都没有时返回 ``False``。"""
        oldest = None
        for name in self._entries:
            try:
                stamp = os.stat(os.path.join(self.path, name)).st_mtime_ns
            except OSError:
                stamp = -1
            if oldest is None or stamp < oldest[0]:
                oldest = (stamp, name)
        if oldest is None:
            return False
        name = oldest[1]
        try:
            os.remove(os.path.join(self.path, name))
        except OSError:
            pass
        self._entries.pop(name, None)
        return True

    def _evict(self) -> int:
        removed = 0
        while len(self._entries) > self.max_entries:
            if not self._remove_oldest():
                break
            removed += 1
        if self.max_bytes is not None:
            while self._entries and self._total_bytes() > self.max_bytes:
                if not self._remove_oldest():
                    break
                removed += 1
        self._evictions += removed
        return removed

    def _touch(self, name: str) -> None:
        try:
            os.utime(os.path.join(self.path, name), None)
        except OSError:
            pass

    def clear(self) -> int:
        """删掉目录下全部条目，返回删除的文件数。"""
        removed = 0
        try:
            names = sorted(os.listdir(self.path))
        except OSError:
            names = []
        for name in names:
            if not name.endswith(_SUFFIX):
                continue
            try:
                os.remove(os.path.join(self.path, name))
            except OSError:
                continue
            removed += 1
        self._entries.clear()
        return removed

    def __repr__(self) -> str:
        return (
            f"DiskPrefixCache(path={self.path!r}, model_id={self.model_id!r}, "
            f"entries={len(self._entries)}, bytes={self._total_bytes()})"
        )
