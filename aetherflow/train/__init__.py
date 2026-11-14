"""训练闭环：参数容器 + 优化器 + 图内损失 + 训练循环。

这一层回答的是"这套语义 IR 能不能真的训一个模型"，而不是"再加一个框架"：

    trainer = Trainer(g, store, opt)       # 反向图只建一次
    losses  = trainer.run(loader, epochs=10)

``Trainer`` 内部就是下面四步，没有新的执行机制：

    feed  = store.feed({"x": batch})       # 参数 + 数据一起喂
    loss  = forward_loss(g, ...)           # 图内的标量
    bwd   = backward(g, wrt=params)        # 反向图也是普通 IR
    grads = ReferenceBackend().execute(bwd, {**feed, "d_loss": 1.0})
    opt.step(grads)                        # 原地更新同一批数组

训练图必须是静态形状：反向图的归约 vjp 需要把梯度广播回输入形状，符号维
下拿不到（``ir/grad_rules.py`` 明确拒绝）。
"""

from .amp import GradScaler
from .checkpoint import TrainingSnapshot, load_checkpoint, save_checkpoint
from .data import DataLoader
from .ema import EMA
from .grads import GradientAccumulator, clip_global_norm, global_norm
from .loss import l2_penalty, mean, mse, softmax_cross_entropy
from .lm import lm_batch
from .loop import Trainer, TrainingState

from .optim import SGD, Adam, Optimizer
from .parallel import (
    DataParallelTrainer,
    allreduce,
    allreduce_grads,
    copy_across_devices,
)
from .params import ParameterStore, default_init, init_array
from .pipeline import PipelineStage, PipelineTrainer
from .sharded import ShardedOptimizer, shard_params
from .schedule import CosineDecay, ExponentialDecay, StepDecay, Warmup, apply_lr

__all__ = [
    "ParameterStore",
    "default_init",
    "init_array",
    "Optimizer",
    "SGD",
    "Adam",
    "ShardedOptimizer",
    "shard_params",
    "Trainer",
    "TrainingState",
    "DataParallelTrainer",
    "PipelineStage",
    "PipelineTrainer",
    "allreduce",
    "allreduce_grads",
    "copy_across_devices",
    "TrainingSnapshot",
    "mean",
    "mse",
    "softmax_cross_entropy",
    "l2_penalty",
    "save_checkpoint",
    "load_checkpoint",
    "DataLoader",
    "GradScaler",
    "CosineDecay",
    "StepDecay",
    "ExponentialDecay",
    "Warmup",
    "apply_lr",
    "global_norm",
    "clip_global_norm",
    "GradientAccumulator",
    "EMA",
    "lm_batch",
]
