"""多 rank 启动：环境变量约定、unique id 分发与进程拉起。

跨进程 ZeRO 的**语义**在 5.229 / 5.231 里已经验证过（分片归属、数值对拍、
存档往返）。缺的一直是运行入口：怎么把 W 个进程同时拉起来，并让它们拿到
同一份 ``ncclUniqueId``——这是工程问题，不是算法问题，但没有它，ZeRO 只能
在每个进程里手工 `python -m ...` 敲一遍。

本模块只做三件事：

1. **环境变量**：``rank`` / ``world_size`` / ``local_rank`` / master 地址与端口。
   优先读 ``AF_*``，其次读 torchrun 的 ``RANK`` / ``WORLD_SIZE`` /
   ``LOCAL_RANK``。两者都没有时用调用方给的默认值；**没有默认值就报错**，
   不猜 0——猜错会让 W 个进程抢同一个 rank，症状是 NCCL 初始化挂住而不是报错。
2. **unique id 分发**：``ncclUniqueId`` 是 128 字节不透明数据，只有 rank 0 能
   生成。这里用一条 TCP 连接把它发给其余 rank：不用共享文件（跨机时文件系统
   不共享），也不用广播（rank 之间通常没有公网可达的端口）。
3. **拉起进程**：``spawn`` 用 ``subprocess`` 而不是 ``multiprocessing``。
   spawn 方式下 multiprocessing 会在子进程里重新 import ``__main__``，
   在 ``python -m unittest`` 下那等于把整个测试套件再跑一遍；``python -c``
   没有这个问题，也便于逐进程注入环境变量。

**不依赖 torch。** 因此 unique id 的载体是普通 bytes，真机上由调用方传
``nccl.NcclUniqueId.to_bytes()`` 的结果，离线测试传任意 128 字节。
"""

from __future__ import annotations

import os
import secrets
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Mapping, Optional

__all__ = [
    "UID_SIZE",
    "env_rank",
    "env_world_size",
    "env_local_rank",
    "env_master_addr",
    "env_master_port",
    "UniqueIdServer",
    "fetch_unique_id",
    "spawn",
]

#: ``ncclUniqueId`` 的字节数。分发层不认识它的内容，只保证原样送达。
UID_SIZE = 128

_RANK_KEY = "AF_RANK"
_WORLD_KEY = "AF_WORLD_SIZE"
_LOCAL_RANK_KEY = "AF_LOCAL_RANK"
_MASTER_ADDR_KEY = "AF_MASTER_ADDR"
_MASTER_PORT_KEY = "AF_MASTER_PORT"


def _int_env(primary: str, legacy: str, default: Optional[int], name: str) -> int:
    """按 ``AF_*`` → 兼容键 → 默认值 的顺序取一个整数；都没有就报错。"""
    for key in (primary, legacy):
        raw = os.environ.get(key)
        if raw is None or raw == "":
            continue
        try:
            return int(raw)
        except ValueError:
            raise ValueError(f"环境变量 {key}={raw!r} 不是整数") from None
    if default is None:
        raise RuntimeError(
            f"环境变量 {primary}（或 {legacy}）未设置；"
            "单进程运行时请显式传默认值，不要靠猜"
        )
    return int(default)


def env_rank(default: Optional[int] = None) -> int:
    """当前进程的全局 rank。"""
    return _int_env(_RANK_KEY, "RANK", default, "rank")


def env_world_size(default: Optional[int] = None) -> int:
    """进程总数。"""
    return _int_env(_WORLD_KEY, "WORLD_SIZE", default, "world_size")


def env_local_rank(default: Optional[int] = None) -> int:
    """本机内的卡号（单机启动时等于全局 rank）。"""
    return _int_env(_LOCAL_RANK_KEY, "LOCAL_RANK", default, "local_rank")


def env_master_addr(default: Optional[str] = "127.0.0.1") -> str:
    """rank 0 的可达地址；没有设置时用默认值。"""
    for key in (_MASTER_ADDR_KEY, "MASTER_ADDR"):
        value = os.environ.get(key)
        if value:
            return value
    if default is None:
        raise RuntimeError(f"环境变量 {_MASTER_ADDR_KEY}（或 MASTER_ADDR）未设置")
    return default


def env_master_port(default: Optional[int] = None) -> int:
    """rank 0 的分发端口。"""
    return _int_env(_MASTER_PORT_KEY, "MASTER_PORT", default, "master_port")


class UniqueIdServer:
    """rank 0 侧：把 128 字节的 unique id 分发给 world_size - 1 个 rank。

    ``serve`` 起一个后台线程阻塞在 ``accept`` 上，所以"何时开始分发"与
    "何时拉起子进程"可以分开：先 ``serve``，再 ``spawn``，子进程连上来时
    服务端一定已经在听。

    协议只有两步：客户端先发 1 字节 rank，服务端回 128 字节 uid。不做鉴权——
    它假定 master 地址只在本机或可信内网可达，这与 NCCL 自身的假定一致。
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
        self._sock.listen(max(1, world_size - 1))
        self.host, self.port = self._sock.getsockname()[:2]
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None
        self._delivered = 0

    @property
    def delivered(self) -> int:
        """已经收到 uid 的 rank 数（不含 rank 0 自己）。"""
        return self._delivered

    def serve(self, uid: bytes) -> None:
        """开始分发 ``uid``；返回时后台线程已在听，不阻塞。"""
        if len(uid) != UID_SIZE:
            raise ValueError(
                f"unique id 必须是 {UID_SIZE} 字节，收到 {len(uid)} 字节"
            )
        if self._thread is not None:
            raise RuntimeError("这个 UniqueIdServer 已经在分发，不能再 serve 一次")
        payload = bytes(uid)
        self._thread = threading.Thread(
            target=self._run, args=(payload,), daemon=True
        )
        self._thread.start()

    def _run(self, payload: bytes) -> None:
        try:
            for _ in range(self.world_size - 1):
                conn, _ = self._sock.accept()
                with conn:
                    conn.settimeout(self.timeout)
                    rank = _recv_exact(conn, 1)
                    if not 0 <= rank[0] < self.world_size:
                        raise RuntimeError(
                            f"客户端声称的 rank {rank[0]} 越界："
                            f"world_size={self.world_size}"
                        )
                    conn.sendall(payload)
                    self._delivered += 1
        except BaseException as exc:  # 线程里抛出的错误要带回主线程
            self._error = exc
        finally:
            try:
                self._sock.close()
            except OSError:
                pass

    def wait(self, timeout: Optional[float] = None) -> None:
        """等分发结束；线程里发生过的异常在这里重新抛出。"""
        if self._thread is not None:
            self._thread.join(timeout if timeout is not None else self.timeout)
        if self._error is not None:
            raise RuntimeError(f"unique id 分发失败：{self._error}") from self._error

    def close(self) -> None:
        """关掉监听 socket；未消费的等待会立刻失败。"""
        try:
            self._sock.close()
        except OSError:
            pass


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    """``recv`` 不保证一次收满，短读要接着读。"""
    chunks = b""
    while len(chunks) < size:
        chunk = conn.recv(size - len(chunks))
        if not chunk:
            raise RuntimeError(
                f"连接在收满 {size} 字节前关闭（只收到 {len(chunks)} 字节）"
            )
        chunks += chunk
    return chunks


def fetch_unique_id(
    host: str, port: int, rank: int, *, timeout: float = 60.0
) -> bytes:
    """从 rank 0 拉取 unique id；rank 必须 >= 1（rank 0 自己持有它）。

    连接会重试到 ``timeout``：服务端可能比客户端晚起来几百毫秒，让客户端
    直接失败会逼调用方写重试循环，那是每个用框架的人都要重复一遍的错。
    """
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError(f"rank 必须是 int，收到 {rank!r}")
    if rank < 1:
        raise ValueError(
            f"rank 0 持有 unique id，不需要拉取；收到 rank={rank}"
        )
    if timeout <= 0:
        raise ValueError(f"timeout 必须为正，得到 {timeout}")
    deadline = time.time() + timeout
    sock: Optional[socket.socket] = None
    while sock is None:
        try:
            sock = socket.create_connection((host, int(port)), timeout=2.0)
        except OSError as exc:
            if time.time() >= deadline:
                raise RuntimeError(
                    f"{timeout}s 内连不上 {host}:{port} 的 unique id 服务：{exc}"
                ) from exc
            time.sleep(0.05)
    with sock:
        sock.settimeout(max(1.0, timeout))
        try:
            sock.sendall(bytes([rank]))
            return _recv_exact(sock, UID_SIZE)
        except OSError as exc:
            # 连上之后连接被对端重置，说明服务端已经关了或中途退出。两个平台
            # 的表现不一样：Linux 当场抛 ConnectionResetError，Windows 则让
            # recv 一直等到超时。统一成同一条可读错误，调用方不必按平台分支。
            raise RuntimeError(
                f"已连上 {host}:{port}，但读取 unique id 失败：{exc}"
            ) from exc


def _entrypoint(spec: str) -> str:
    """把 ``"包.模块:函数"`` 变成一段 ``-c`` 代码；格式不对就在本地报错。"""
    if not isinstance(spec, str):
        raise TypeError(f"entrypoint 必须是字符串，收到 {type(spec).__name__}")
    module, sep, func = spec.partition(":")
    if not sep or not module.strip() or not func.strip():
        raise ValueError(
            f"entrypoint 必须形如 '包.模块:函数'，收到 {spec!r}"
        )
    return (
        "import importlib, sys;"
        f"m = importlib.import_module({module!r});"
        f"getattr(m, {func!r})(*sys.argv[1:])"
    )


def spawn(
    entrypoint: str,
    world_size: int,
    *args: Any,
    master_addr: str = "127.0.0.1",
    uid: Optional[bytes] = None,
    timeout: float = 600.0,
    check: bool = True,
    python: Optional[str] = None,
    extra_env: Optional[Mapping[str, str]] = None,
) -> List[int]:
    """拉起 ``world_size`` 个进程，每个都拿到 rank 环境变量与同一份 uid。

    ``entrypoint`` 是 ``"包.模块:函数"``；``args`` 会作为命令行参数原样传给
    那个函数（因此只能是字符串——跨进程没有 pickle，这是刻意的：隐式序列化
    会把"模型能不能 pickle"变成启动期才暴露的问题）。

    没给 ``uid`` 时用 ``secrets.token_bytes`` 生成 128 字节占位符：离线测试
    用它验证分发链路，真机上必须传 ``ncclUniqueId`` 的字节。

    ``extra_env`` 是追加给子进程的环境变量（例如主机中转的端口）。**不允许
    覆盖本函数自己写入的五个键**：``AF_RANK`` / ``AF_WORLD_SIZE`` /
    ``AF_LOCAL_RANK`` / ``AF_MASTER_ADDR`` / ``AF_MASTER_PORT``——手滑覆盖
    会让 W 个进程抢同一个 rank，症状是集合通信挂住而不是一条配置错误。
    """
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise TypeError(f"world_size 必须是 int，收到 {world_size!r}")
    if world_size < 1:
        raise ValueError(f"world_size 必须 >= 1，收到 {world_size}")
    if timeout <= 0:
        raise ValueError(f"timeout 必须为正，得到 {timeout}")
    code = _entrypoint(entrypoint)
    payload = secrets.token_bytes(UID_SIZE) if uid is None else bytes(uid)
    if len(payload) != UID_SIZE:
        raise ValueError(
            f"unique id 必须是 {UID_SIZE} 字节，收到 {len(payload)} 字节"
        )
    for value in args:
        if not isinstance(value, str):
            raise TypeError(
                f"spawn 的参数必须是字符串（跨进程不 pickle），收到 "
                f"{type(value).__name__}: {value!r}"
            )
    reserved = {
        _RANK_KEY,
        _WORLD_KEY,
        _LOCAL_RANK_KEY,
        _MASTER_ADDR_KEY,
        _MASTER_PORT_KEY,
    }
    extra: Dict[str, str] = {}
    if extra_env is not None:
        if not isinstance(extra_env, Mapping):
            raise TypeError(
                f"extra_env 必须是名字到值的映射，得到 {type(extra_env).__name__}"
            )
        for key, value in extra_env.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError(
                    f"extra_env 的键与值都必须是 str，收到 {key!r}: {value!r}"
                )
            if key in reserved:
                raise ValueError(
                    f"extra_env 不能覆盖 {key}：它由 spawn 按 rank 写入"
                )
            extra[key] = value

    server = UniqueIdServer(world_size, host=master_addr, timeout=timeout)
    server.serve(payload)
    interpreter = python or sys.executable
    processes: List[subprocess.Popen] = []
    try:
        for rank in range(world_size):
            env = os.environ.copy()
            env.update(
                {
                    _RANK_KEY: str(rank),
                    _WORLD_KEY: str(world_size),
                    _LOCAL_RANK_KEY: str(rank),
                    _MASTER_ADDR_KEY: str(server.host),
                    _MASTER_PORT_KEY: str(server.port),
                }
            )
            env.update(extra)
            processes.append(
                subprocess.Popen(
                    [interpreter, "-c", code, *args],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        deadline = time.time() + timeout
        codes: List[int] = []
        for rank, process in enumerate(processes):
            remaining = deadline - time.time()
            if remaining <= 0:
                for other in processes:
                    other.kill()
                raise RuntimeError(
                    f"rank {rank} 在 {timeout}s 内没有退出；已终止全部进程"
                )
            code_rank = process.wait(timeout=remaining)
            codes.append(code_rank)
            if code_rank != 0:
                output = process.stdout.read() if process.stdout else ""
                for other in processes:
                    if other.poll() is None:
                        other.kill()
                raise RuntimeError(
                    f"rank {rank} 退出码 {code_rank}；输出：\n{output}"
                )
        # 正常收尾：确认 world_size - 1 个 rank 都拿到了 uid。放在这里而不是
        # finally 里，是为了不让"分发线程的失败"盖掉"子进程退出码"这个更
        # 直接的诊断信息。
        server.wait(timeout=min(30.0, timeout))
        return codes
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            if process.stdout is not None:
                process.stdout.close()
        server.close()
