"""``DistributedDataParallel`` 的跨进程测试进程：训练两步并写一份 JSON。

单独成文件而不是写进测试模块：``launch`` 用 ``python -c`` 起进程，被 import
的模块不能有副作用。每个 rank 写自己的文件——多进程往同一个文件追加在
Windows 上不保证原子。

判据由父进程做：每个 rank 写回的损失与参数，必须与"单进程跑合并 batch"
两步之后一致。
"""

import json
import os

import numpy as np

from aetherflow import Graph, f32
from aetherflow.train import (
    SGD,
    DistributedDataParallel,
    HostCollective,
    ParameterStore,
    Trainer,
    env_rank,
    env_world_size,
    init_array,
    mse,
)


def _graph(batch):
    """与 ``test_ddp`` 里的线性回归同构：``mse(x @ w + b, target)``。"""
    g = Graph("ddp")
    x = g.input("x", f32[batch, 3])
    w = g.input("w", f32[3, 1])
    b = g.input("b", f32[1])
    target = g.input("target", f32[batch, 1])
    g.output(mse(g, g.add(g.matmul(x, w), b), target), "loss")
    return g


def _batch(total, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "x": rng.standard_normal((total, 3)).astype(np.float32),
        "target": rng.standard_normal((total, 1)).astype(np.float32),
    }


def train(out_dir):
    """环境变量里读 rank / world，跑两步，把损失与参数写进 out_dir。"""
    rank = env_rank()
    world = env_world_size()
    collective = HostCollective.from_env(
        rank=rank, world_size=world, timeout=60.0
    )
    try:
        store = ParameterStore(
            {
                "w": init_array("w", (3, 1)),
                "b": np.zeros((1,), dtype=np.float32),
            }
        )
        trainer = Trainer(
            _graph(8 // world),
            store,
            SGD(store.arrays(), lr=0.1),
            rng_rank=rank,
            rng_world_size=world,
        )
        ddp = DistributedDataParallel(trainer, collective)
        data = _batch(8)
        losses = [ddp.step(data) for _ in range(2)]
        payload = {
            "rank": rank,
            "world": world,
            "steps": ddp.steps,
            "losses": losses,
            "params": {name: store[name].tolist() for name in store.names},
        }
        with open(
            os.path.join(out_dir, f"rank{rank}.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(payload, handle)
    finally:
        collective.close()
