"""NCCL 集合通信的真机 worker（由 ``tests/test_nccl.py`` 以子进程启动）。

用法：``python -m tests._nccl_worker <rank> <world_size> <uid_path> <out_path>``。
rank 0 负责生成 ``ncclUniqueId`` 并写到 ``uid_path``，其余 rank 轮询读取；
之后所有 rank 依次跑四个集合操作，把结果写进 ``out_path``（``.npz``）。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from aetherflow.train import nccl


def _wait_for_uid(path: str) -> bytes:
    deadline = time.time() + 120.0
    while time.time() < deadline:
        if os.path.exists(path):
            with open(path, "rb") as handle:
                data = handle.read()
            if len(data) == 128:
                return data
        time.sleep(0.05)
    raise SystemExit(f"等待 unique id 超时：{path}")


def main(argv: list) -> int:
    if len(argv) != 5:
        raise SystemExit("用法：python -m tests._nccl_worker rank world uid_path out_path")
    rank, world = int(argv[1]), int(argv[2])
    uid_path, out_path = argv[3], argv[4]

    if rank == 0:
        value = nccl.unique_id()
        temporary = uid_path + ".tmp"
        with open(temporary, "wb") as handle:
            handle.write(value.to_bytes())
        os.replace(temporary, uid_path)
    data = _wait_for_uid(uid_path)

    import cupy as cp

    with cp.cuda.Device(rank):
        results = {}
        with nccl.NcclComm(rank, world, data) as comm:
            # AllReduce：每个 rank 贡献 arange(8)+rank
            reduced = cp.arange(8, dtype=cp.float32) + rank
            comm.all_reduce(reduced, op="sum")
            comm.synchronize()
            results["allreduce"] = cp.asnumpy(reduced)

            # AllGather：每个 rank 贡献 arange(4)+rank*10
            send = cp.arange(4, dtype=cp.float32) + rank * 10
            recv = cp.empty(4 * world, dtype=cp.float32)
            comm.all_gather(send, recv)
            comm.synchronize()
            results["allgather"] = cp.asnumpy(recv)

            # ReduceScatter：发送缓冲按 rank 切段，本 rank 收第 rank 段
            payload = cp.arange(4 * world, dtype=cp.float32) + rank * 100
            scattered = cp.empty(4, dtype=cp.float32)
            comm.reduce_scatter(payload, scattered)
            comm.synchronize()
            results["reduce_scatter"] = cp.asnumpy(scattered)

            # Broadcast：root 0 的 arange(6) 覆盖全员
            buffer = (
                cp.arange(6, dtype=cp.float32)
                if rank == 0
                else cp.full((6,), -1, dtype=cp.float32)
            )
            comm.broadcast(buffer, root=0)
            comm.synchronize()
            results["broadcast"] = cp.asnumpy(buffer)
    np.savez(out_path, **results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
