"""RCCL 集合通信的真机 worker（由 ``tests/test_nccl.py`` 以子进程启动）。

用法与 ``_nccl_worker`` 相同：``python -m tests._rccl_worker <rank> <world> <uid_path> <out_path>``。

与 CUDA 版的唯一区别是数组来自项目自己的 HIP 运行时（``DeviceArray``），
不是 cupy：ROCm 机器上通常没有 cupy，而 ``DeviceArray`` 同样提供
``data.ptr`` / ``size`` / ``dtype`` / ``flags``，正好是绑定层要求的四件事。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from aetherflow.backends.hip import runtime
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
        raise SystemExit("用法：python -m tests._rccl_worker rank world uid_path out_path")
    rank, world = int(argv[1]), int(argv[2])
    uid_path, out_path = argv[3], argv[4]

    if rank == 0:
        value = nccl.unique_id()
        temporary = uid_path + ".tmp"
        with open(temporary, "wb") as handle:
            handle.write(value.to_bytes())
        os.replace(temporary, uid_path)
    data = _wait_for_uid(uid_path)

    runtime.set_device(rank)
    results = {}
    with nccl.NcclComm(rank, world, data) as comm:
        reduced = runtime.array(np.arange(8, dtype=np.float32) + rank)
        comm.all_reduce(reduced, op="sum")
        comm.synchronize()
        results["allreduce"] = runtime.asnumpy(reduced)

        send = runtime.array(np.arange(4, dtype=np.float32) + rank * 10)
        recv = runtime.empty(4 * world, np.float32)
        comm.all_gather(send, recv)
        comm.synchronize()
        results["allgather"] = runtime.asnumpy(recv)

        payload = runtime.array(np.arange(4 * world, dtype=np.float32) + rank * 100)
        scattered = runtime.empty(4, np.float32)
        comm.reduce_scatter(payload, scattered)
        comm.synchronize()
        results["reduce_scatter"] = runtime.asnumpy(scattered)

        buffer = runtime.array(
            np.arange(6, dtype=np.float32)
            if rank == 0
            else np.full((6,), -1, dtype=np.float32)
        )
        comm.broadcast(buffer, root=0)
        comm.synchronize()
        results["broadcast"] = runtime.asnumpy(buffer)
    np.savez(out_path, **results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
