"""多 rank 启动器的测试进程：拉 unique id 并追加一行 JSON。

单独成文件而不是写进测试模块，是因为 ``spawn`` 用 ``python -c`` 起进程，
被 import 的模块不能有副作用；这里只有一个 ``main`` 与一个故意失败的
``boom``，供父进程断言退出码。
"""

import json

from aetherflow.train.distributed import (
    env_master_addr,
    env_master_port,
    env_rank,
    env_world_size,
    fetch_unique_id,
)


def main(path):
    """每个 rank 写自己的文件：多进程往同一个文件追加在 Windows 上不保证原子。"""
    import os

    rank = env_rank()
    world = env_world_size()
    uid = b""
    if rank != 0:  # rank 0 自己持有 uid，不需要向服务端拉取
        uid = fetch_unique_id(
            env_master_addr(), env_master_port(), rank, timeout=30.0
        )
    with open(
        os.path.join(path, f"rank{rank}.json"), "w", encoding="utf-8"
    ) as handle:
        handle.write(
            json.dumps(
                {"rank": rank, "world": world, "uid": bytes(uid).hex()}
            )
        )


def boom(path):
    """让子进程以非 0 退出码结束，验证父进程不会静默吞掉失败。"""
    raise SystemExit(3)
