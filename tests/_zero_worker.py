"""跨进程 ZeRO 的真机 worker（由 ``tests/test_zero.py`` 以子进程启动）。

用法：``python -m tests._zero_worker <rank> <world_size> <uid_path> <out_path>``。
每个 rank 用同一份初始参数，用**不同**的梯度（模拟数据并行的不同数据分片）
跑两步 ZeRO-1，把最终参数、本片槽位字节与归属表写进 ``out_path``（``.npz``）。
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from aetherflow.train import SGD, NcclHostCollective, ZeroRank, nccl

SHAPES = {"w1": (4, 3), "b1": (4,), "w2": (2, 4), "b2": (2,)}


def make_params() -> dict:
    rng = np.random.default_rng(3)
    return {
        name: rng.standard_normal(shape).astype(np.float32)
        for name, shape in SHAPES.items()
    }


def grads_for(rank: int, step: int) -> dict:
    rng = np.random.default_rng(100 + 10 * step + rank)
    return {
        name: rng.standard_normal(shape).astype(np.float32)
        for name, shape in SHAPES.items()
    }


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
        raise SystemExit("用法：python -m tests._zero_worker rank world uid_path out_path")
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
        params = make_params()
        comm = nccl.NcclComm(rank, world, data)
        try:
            zero = ZeroRank(
                params,
                lambda p: SGD(p, lr=0.1, momentum=0.9),
                rank=rank,
                world_size=world,
                collective=NcclHostCollective(comm, device=rank),
            )
            for step in range(2):
                zero.step(grads_for(rank, step))
        finally:
            comm.destroy()

    payload = {f"p::{name}": params[name] for name in params}
    payload["__meta__"] = np.array(
        json.dumps(
            {
                "rank": rank,
                "world_size": world,
                "owned_bytes": zero.owned_bytes(),
                "my_names": list(zero.my_names),
                "assignments": dict(zero.assignments),
            },
            ensure_ascii=False,
        )
    )
    np.savez(out_path, **payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
