"""跨进程主机中转集合通信：把各 rank 的数组汇到父进程、求和后再散回去。

5.232 只解决了"怎么把 W 个进程拉起来、让它们拿到同一份 unique id"，
``NcclHostCollective`` 只包了真机 NCCL。中间一直缺一条**离线可验证**的路：
没有 NCCL 的机器（本地 CPU、CI、WSL 未挂 GPU 时）也要能把多进程数据并行
完整跑一遍，否则"多进程对不对"只能靠在有卡机器上观察损失曲线。

这里的做法是最朴素的一种：父进程开一个 TCP 服务，W 个子进程各连一条连接，
同一 ``seq`` 的消息收齐后求和、按原路回发。

**为什么是主机中转，而不是设备侧。** 它不追求性能。5.331 起
``collective_support`` 在 NCCL / RCCL 可加载时默认走设备侧，这里保留为
"没有显卡 / 没有 NCCL 的机器（本地 CPU、CI、WSL 未挂 GPU）"以及显式
``device=False`` / ``collective="host"`` 时的路径，把同一条语义在跨进程里
实现一遍，让 :class:`DistributedDataParallel` 的数值判据（与单进程全 batch
逐步对拍）在没有显卡的机器上也能成立。

**协议。** 客户端先发 1 字节 rank 握手。之后每次集合调用发一帧
``struct("!QQ8s")`` = (seq, count, dtype 字符串左对齐 8 字节) + ``count`` 个
元素的裸字节。服务端按 ``seq`` 归组，收齐 ``world_size`` 份后校验 count 与
dtype 一致，按 rank 顺序求和，把同样的帧回发给每个等待的客户端。

**形状不进协议。** 只传元素个数与 dtype：每个 rank 本来就知道自己的形状，
服务端也不需要 reshape。这让归约退化成"一段字节相加"，校验面反而是明确的
——count 与 dtype 不一致就当场报错，不做隐式类型提升。

**为什么不复用 ``UniqueIdServer``。** 那个是"发一次 128 字节"的一问一答；
这里需要长期连接与多轮 barrier。硬塞进去只会让它变成一个协议很别扭的通用
服务器，两边的错误信息也会互相干扰。
"""

from __future__ import annotations

import os
import socket
import struct
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .distributed import env_rank, env_world_size, spawn

__all__ = ["HostRelay", "HostCollective", "launch"]

#: 帧头：seq、元素个数、dtype 字符串（``np.dtype.str`` 最长 8 字节）。
_HEADER = struct.Struct("!QQ8s")
_ADDR_KEY = "AF_MASTER_ADDR"
_PORT_KEY = "AF_RELAY_PORT"


def _recv_exact(conn: socket.socket, size: int) -> Optional[bytes]:
    """收满 ``size`` 字节；对端先关闭时返回 ``None``（当作正常收尾）。"""
    chunks = b""
    while len(chunks) < size:
        chunk = conn.recv(size - len(chunks))
        if not chunk:
            return None if not chunks else _short(size, len(chunks))
        chunks += chunk
    return chunks


def _short(size: int, got: int) -> bytes:
    """读到一半连接就断了：这是错误，不是收尾。"""
    raise RuntimeError(f"连接在收满 {size} 字节前关闭（只收到 {got} 字节）")


def _dtype_code(raw: bytes) -> str:
    """把帧头里的 dtype 字符串还原；不认识的 dtype 当场拒绝。"""
    code = raw.rstrip(b"\x00").decode("ascii")
    try:
        dtype = np.dtype(code)
    except TypeError:
        raise ValueError(f"帧头里的 dtype {code!r} 不是合法 numpy dtype") from None
    if not np.issubdtype(dtype, np.floating):
        raise TypeError(
            f"集合通信只接受浮点数组，收到 {dtype}（整数求平均会被静默截断）"
        )
    return code


class HostRelay:
    """父进程侧的中转服务：W 条连接、按 seq 对齐、求和后回发。

    ``serve()`` 起后台线程就返回，所以"开始监听"与"拉起子进程"可以分开：
    先 ``serve`` 再 ``spawn``，子进程连上来时服务端一定已经在听。

    错误不吞：后台线程里的异常存进 ``_error``，``wait()`` 时重新抛出；正在
    等其余 rank 的线程会被唤醒并按同一条错误退出，而不是各自等满超时。
    """

    def __init__(
        self,
        world_size: int,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        timeout: float = 60.0,
    ) -> None:
        if isinstance(world_size, bool) or not isinstance(world_size, int):
            raise TypeError(f"world_size 必须是 int，收到 {world_size!r}")
        if world_size < 1:
            raise ValueError(f"world_size 必须 >= 1，收到 {world_size}")
        if timeout <= 0:
            raise ValueError(f"timeout 必须为正，得到 {timeout}")
        self.world_size = world_size
        self.timeout = float(timeout)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(max(1, world_size))
        self.host, self.port = self._sock.getsockname()[:2]
        self._cond = threading.Condition()
        self._pending: Dict[int, Dict[int, Tuple[int, str, bytes]]] = {}
        self._results: Dict[int, Tuple[int, str, bytes]] = {}
        self._conns: Dict[int, socket.socket] = {}
        self._error: Optional[BaseException] = None
        self._closed = False
        self._thread: Optional[threading.Thread] = None

    # -- 生命周期 -----------------------------------------------------

    @property
    def error(self) -> Optional[BaseException]:
        """后台线程里发生过的第一个错误；正常时为 ``None``。"""
        return self._error

    def serve(self) -> None:
        """开始接受连接；返回时后台线程已在听，不阻塞。"""
        if self._thread is not None:
            raise RuntimeError("这个 HostRelay 已经在服务，不能再 serve 一次")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def wait(self, timeout: Optional[float] = None) -> None:
        """等后台线程收尾；线程里发生过的异常在这里重新抛出。

        ``close()`` 主动关掉监听后，``accept`` 的 ``OSError`` 属于正常收尾，
        不会被当成错误。
        """
        if self._thread is not None:
            self._thread.join(timeout if timeout is not None else self.timeout)
        if self._error is not None:
            raise RuntimeError(f"主机中转失败：{self._error}") from self._error

    def close(self) -> None:
        """关掉监听与全部连接；正在阻塞的读写会尽快失败。"""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
            conns = list(self._conns.values())
        _close_quietly(self._sock)
        for conn in conns:
            _shutdown(conn)

    # -- 服务端 -------------------------------------------------------

    def _run(self) -> None:
        try:
            conns: Dict[int, socket.socket] = {}
            for _ in range(self.world_size):
                conn, _ = self._sock.accept()
                conn.settimeout(self.timeout)
                rank = _recv_exact(conn, 1)
                if rank is None:
                    raise RuntimeError("连接在 rank 握手前就关闭了")
                index = rank[0]
                if not 0 <= index < self.world_size:
                    raise RuntimeError(
                        f"客户端声称的 rank {index} 越界："
                        f"world_size={self.world_size}"
                    )
                if index in conns:
                    raise RuntimeError(f"rank {index} 重复连接了两次")
                conns[index] = conn
            with self._cond:
                self._conns = conns
            threads = [
                threading.Thread(
                    target=self._serve_rank, args=(rank, conn), daemon=True
                )
                for rank, conn in sorted(conns.items())
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        except OSError as exc:
            if not self._closed:
                self._fail(exc)
        except BaseException as exc:
            self._fail(exc)
        finally:
            _close_quietly(self._sock)
            for conn in list(self._conns.values()):
                _shutdown(conn)

    def _serve_rank(self, rank: int, conn: socket.socket) -> None:
        try:
            while True:
                header = _recv_exact(conn, _HEADER.size)
                if header is None:
                    return
                seq, count, raw = _HEADER.unpack(header)
                dtype = _dtype_code(raw)
                body = _recv_exact(conn, count * np.dtype(dtype).itemsize)
                if body is None:
                    raise RuntimeError(
                        f"rank {rank} 在 seq {seq} 上发完帧头就关闭了连接"
                    )
                result = self._submit(seq, rank, count, dtype, body)
                conn.sendall(_encode(seq, result))
        except (OSError, RuntimeError) as exc:
            if not self._closed:
                self._fail(exc)
        except BaseException as exc:
            self._fail(exc)

    def _submit(
        self, seq: int, rank: int, count: int, dtype: str, body: bytes
    ) -> Tuple[int, str, bytes]:
        """把一份数据放进 ``seq`` 的槽位，等齐后求和；每个 rank 只等一次。"""
        deadline = time.monotonic() + self.timeout
        with self._cond:
            batch = self._pending.setdefault(seq, {})
            if rank in batch:
                raise RuntimeError(f"rank {rank} 在 seq {seq} 上发了两次")
            batch[rank] = (count, dtype, body)
            while seq not in self._results:
                if self._error is not None:
                    raise RuntimeError(f"中继已失败：{self._error}") from self._error
                if self._closed:
                    raise RuntimeError("中继已关闭")
                if len(self._pending[seq]) == self.world_size:
                    self._results[seq] = self._reduce(seq, self._pending.pop(seq))
                    self._cond.notify_all()
                    break
                left = deadline - time.monotonic()
                if left <= 0 or not self._cond.wait(left):
                    raise RuntimeError(
                        f"等待 seq {seq} 的其余 rank 超过 {self.timeout}s"
                    )
            return self._results[seq]

    def _reduce(
        self, seq: int, batch: Dict[int, Tuple[int, str, bytes]]
    ) -> Tuple[int, str, bytes]:
        """校验整组一致后按 rank 顺序相加；形状由各客户端自己还原。"""
        reference: Optional[Tuple[int, str]] = None
        parts: List[np.ndarray] = []
        for rank in range(self.world_size):
            if rank not in batch:
                raise RuntimeError(
                    f"seq {seq} 收齐了 {len(batch)} 份却缺 rank {rank}"
                )
            count, dtype, body = batch[rank]
            if reference is None:
                reference = (count, dtype)
            elif (count, dtype) != reference:
                raise RuntimeError(
                    f"seq {seq} 上 rank {rank} 的帧是 count={count}/dtype={dtype}，"
                    f"与 rank 0 的 count={reference[0]}/dtype={reference[1]} 不一致"
                )
            parts.append(np.frombuffer(body, dtype=np.dtype(dtype)))
        total = np.array(parts[0], copy=True)
        for part in parts[1:]:
            total += part
        return (reference[0], reference[1], total.tobytes())

    def _fail(self, exc: BaseException) -> None:
        with self._cond:
            if self._error is None:
                self._error = exc
            self._cond.notify_all()
        for conn in list(self._conns.values()):
            _shutdown(conn)


class HostCollective:
    """子进程侧的集合通信：连上父进程的中继，按 seq 收发。

    ``rank`` 与 ``world_size`` 必须与 ``Trainer`` 的随机流、优化器分片一致；
    :class:`DistributedDataParallel` 在构造时会核对，这里只保证接口面：
    ``all_reduce(array, op="sum")`` 返回一个同形状、同 dtype 的新数组。
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        timeout: float = 60.0,
    ) -> None:
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError(f"rank 必须是 int，收到 {rank!r}")
        if isinstance(world_size, bool) or not isinstance(world_size, int):
            raise TypeError(f"world_size 必须是 int，收到 {world_size!r}")
        if world_size < 1:
            raise ValueError(f"world_size 必须 >= 1，收到 {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} 越界：world_size={world_size}")
        if timeout <= 0:
            raise ValueError(f"timeout 必须为正，得到 {timeout}")
        self.rank = rank
        self.world_size = world_size
        self.timeout = float(timeout)
        self._lock = threading.Lock()
        self._seq = 0
        self._sock = self._connect(host, int(port))
        try:
            self._sock.sendall(bytes([rank]))
        except OSError as exc:
            self._close_quietly()
            raise RuntimeError(f"连上 {host}:{port} 后发 rank 握手失败：{exc}") from exc

    @classmethod
    def from_env(
        cls,
        *,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        timeout: float = 60.0,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> "HostCollective":
        """按 ``AF_*`` 环境变量建一个实例。

        ``rank`` / ``world_size`` 缺省读 ``AF_RANK`` / ``AF_WORLD_SIZE``；
        地址与端口读 ``AF_MASTER_ADDR`` / ``AF_RELAY_PORT``，**缺了直接报错**
        而不是回落到 127.0.0.1:0——端口 0 会连到一个随机端口上，症状是
        "永远等不到其余 rank"，而不是一条能看懂的配置错误。
        """
        resolved_rank = env_rank() if rank is None else rank
        resolved_world = env_world_size() if world_size is None else world_size
        address = host if host is not None else os.environ.get(_ADDR_KEY)
        if not address:
            raise RuntimeError(
                f"环境变量 {_ADDR_KEY} 未设置；由 launch() 拉起时会自动注入，"
                "手工启动请显式给出 host"
            )
        if port is None:
            raw = os.environ.get(_PORT_KEY)
            if raw is None or raw == "":
                raise RuntimeError(
                    f"环境变量 {_PORT_KEY} 未设置；由 launch() 拉起时会自动注入，"
                    "手工启动请显式给出 port"
                )
            try:
                port = int(raw)
            except ValueError:
                raise ValueError(f"环境变量 {_PORT_KEY}={raw!r} 不是整数") from None
        return cls(
            resolved_rank,
            resolved_world,
            host=address,
            port=port,
            timeout=timeout,
        )

    # -- 集合通信 -----------------------------------------------------

    def all_reduce(self, array: Any, *, op: str = "sum") -> np.ndarray:
        """对同一形状的数组求和，各 rank 拿回同一份结果。"""
        if op != "sum":
            raise ValueError(
                f"HostCollective 只实现 op='sum'，收到 {op!r}；"
                "平均在调用方做（除 world_size），服务端不做第二条归约语义"
            )
        value = np.asarray(array)
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(
                f"集合通信只接受浮点数组，收到 {value.dtype}"
                "（整数求平均会被静默截断）"
            )
        code = value.dtype.str
        if len(code) > 8:
            raise ValueError(f"dtype 字符串 {code!r} 超过帧头的 8 字节")
        body = np.ascontiguousarray(value).tobytes()
        with self._lock:
            seq = self._seq
            self._seq += 1
            self._sock.sendall(
                _HEADER.pack(seq, value.size, code.encode("ascii").ljust(8, b"\x00"))
                + body
            )
            header = _recv_exact(self._sock, _HEADER.size)
            if header is None:
                raise RuntimeError(f"中继在 seq {seq} 上关闭了连接")
            got_seq, count, raw = _HEADER.unpack(header)
            if got_seq != seq:
                raise RuntimeError(f"期望 seq {seq} 的回包，收到 {got_seq}")
            if count != value.size:
                raise RuntimeError(
                    f"seq {seq} 的回包元素数是 {count}，本 rank 发出的是 {value.size}"
                )
            if _dtype_code(raw) != code:
                raise RuntimeError(
                    f"seq {seq} 的回包 dtype 是 {_dtype_code(raw)}，"
                    f"本 rank 发出的是 {code}"
                )
            payload = _recv_exact(self._sock, count * value.dtype.itemsize)
            if payload is None:
                raise RuntimeError(f"seq {seq} 的回包在数据段前被截断")
        return np.frombuffer(payload, dtype=value.dtype).reshape(value.shape).copy()

    def close(self) -> None:
        """断开与中继的连接；重复调用是空操作。"""
        self._close_quietly()

    def __repr__(self) -> str:
        return (
            f"HostCollective(rank={self.rank}, world_size={self.world_size})"
        )

    # -- 内部 ---------------------------------------------------------

    def _connect(self, host: str, port: int) -> socket.socket:
        """连中继；重试到 ``timeout``，与服务端谁先起来无关。"""
        deadline = time.time() + self.timeout
        while True:
            try:
                sock = socket.create_connection((host, port), timeout=2.0)
            except OSError as exc:
                if time.time() >= deadline:
                    raise RuntimeError(
                        f"{self.timeout}s 内连不上主机中转 {host}:{port}：{exc}"
                    ) from exc
                time.sleep(0.05)
                continue
            sock.settimeout(self.timeout)
            return sock

    def _close_quietly(self) -> None:
        _close_quietly(self._sock)


def launch(
    entrypoint: str,
    world_size: int,
    *args: Any,
    master_addr: str = "127.0.0.1",
    timeout: float = 600.0,
    check: bool = True,
    python: Optional[str] = None,
) -> List[int]:
    """一条命令拉起 W 个进程，并替它们开好主机中转。

    ``entrypoint`` 与 ``spawn`` 相同（``"包.模块:函数"``）；额外的环境变量
    只有 ``AF_RELAY_PORT``——``AF_MASTER_ADDR`` 已由 ``spawn`` 按 uid 服务
    写入。子进程里用 :meth:`HostCollective.from_env` 就能拿到连接参数。

    中继在 ``spawn`` 期间保持监听，退出后立刻关闭。子进程若在集合调用上
    失败会以非 0 退出，``spawn`` 先抛错并带出它的输出；这里再补查一次中继
    自身的错误，避免"服务端早就挂了、子进程只是被连累"的情况被误导。
    """
    relay = HostRelay(world_size, host=master_addr, timeout=timeout)
    relay.serve()
    try:
        codes = spawn(
            entrypoint,
            world_size,
            *args,
            master_addr=master_addr,
            timeout=timeout,
            check=check,
            python=python,
            extra_env={_PORT_KEY: str(relay.port)},
        )
    finally:
        relay.close()
    if relay.error is not None:
        raise RuntimeError(f"主机中转失败：{relay.error}") from relay.error
    return codes


def _encode(seq: int, frame: Tuple[int, str, bytes]) -> bytes:
    count, dtype, body = frame
    return _HEADER.pack(seq, count, dtype.encode("ascii").ljust(8, b"\x00")) + body


def _close_quietly(sock: socket.socket) -> None:
    try:
        sock.close()
    except OSError:
        pass


def _shutdown(sock: socket.socket) -> None:
    """先 ``shutdown`` 再 ``close``：另一个线程里阻塞的 ``recv`` 才会醒。"""
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    _close_quietly(sock)
