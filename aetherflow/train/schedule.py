"""学习率调度：只回答"第几步该用多大学习率"，不碰参数、不持有优化器。

为什么做成无状态的纯函数 ``lr_at(step)``，而不是 PyTorch 那种
``scheduler.step()`` 自增计数：

1. **续训不走样。** 唯一需要存档的步数是 ``optimizer.step_count``（5.27 已经存了）。
   调度器自己再维护一份计数，就多了一份可能与优化器分叉的状态；分叉时学习率
   会悄悄偏移，而损失曲线看起来只是"没那么好"。
2. **梯度累积时定义清楚。** 累积窗口内走了 N 次反向但只更新一次参数，
   "第几步"必须指优化器真实更新的次数，不能是微批次数。纯函数让调用方
   明确地传 ``optimizer.step_count``。
3. **可组合。** warmup 是"前 K 步换一条曲线"，用组合器表达比给每个调度器
   都加一个 ``warmup_steps`` 参数更不容易写错。

所有调度器都保证 ``lr_at`` 单调不增（warmup 除外，它显式声明是升的），
且返回值不低于 0。越界步数不报错：``step >= total_steps`` 时停在下界，
因为"训练跑长了"不该让训练崩掉。
"""

from __future__ import annotations

import math

from .optim import Optimizer

__all__ = [
    "CosineDecay",
    "StepDecay",
    "ExponentialDecay",
    "Warmup",
    "apply_lr",
]


class CosineDecay:
    """余弦退火：从 ``base_lr`` 平滑降到 ``min_lr``，在 ``total_steps`` 处到底。

    余弦而不是线性：末端斜率趋近于零，最后一段学习率变化很慢，相当于在
    最优点附近多停留了一会儿；线性衰减的末端仍是恒定斜率，会直接冲过谷底。
    """

    def __init__(self, base_lr: float, total_steps: int, min_lr: float = 0.0) -> None:
        if base_lr <= 0.0:
            raise ValueError(f"base_lr 必须为正，得到 {base_lr}")
        if int(total_steps) < 1:
            raise ValueError(f"total_steps 必须 >= 1，得到 {total_steps}")
        if not 0.0 <= min_lr <= base_lr:
            raise ValueError(f"min_lr 必须落在 [0, base_lr]，得到 {min_lr}")
        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)

    def lr_at(self, step: int) -> float:
        if step <= 0:
            return self.base_lr
        if step >= self.total_steps:
            return self.min_lr
        progress = step / self.total_steps
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

    def __repr__(self) -> str:
        return (
            f"CosineDecay(base_lr={self.base_lr:g}, total_steps={self.total_steps}, "
            f"min_lr={self.min_lr:g})"
        )


class StepDecay:
    """阶梯衰减：每 ``step_size`` 步乘一次 ``gamma``，不低于 ``min_lr``。"""

    def __init__(
        self,
        base_lr: float,
        step_size: int,
        gamma: float = 0.1,
        min_lr: float = 0.0,
    ) -> None:
        if base_lr <= 0.0:
            raise ValueError(f"base_lr 必须为正，得到 {base_lr}")
        if int(step_size) < 1:
            raise ValueError(f"step_size 必须 >= 1，得到 {step_size}")
        if not 0.0 < gamma < 1.0:
            raise ValueError(f"gamma 必须落在 (0, 1)，得到 {gamma}")
        if not 0.0 <= min_lr <= base_lr:
            raise ValueError(f"min_lr 必须落在 [0, base_lr]，得到 {min_lr}")
        self.base_lr = float(base_lr)
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        self.min_lr = float(min_lr)

    def lr_at(self, step: int) -> float:
        if step <= 0:
            return self.base_lr
        return max(self.base_lr * self.gamma ** (step // self.step_size), self.min_lr)

    def __repr__(self) -> str:
        return (
            f"StepDecay(base_lr={self.base_lr:g}, step_size={self.step_size}, "
            f"gamma={self.gamma:g}, min_lr={self.min_lr:g})"
        )


class ExponentialDecay:
    """每步乘 ``gamma`` 的指数衰减，不低于 ``min_lr``。"""

    def __init__(self, base_lr: float, gamma: float, min_lr: float = 0.0) -> None:
        if base_lr <= 0.0:
            raise ValueError(f"base_lr 必须为正，得到 {base_lr}")
        if not 0.0 < gamma < 1.0:
            raise ValueError(f"gamma 必须落在 (0, 1)，得到 {gamma}")
        if not 0.0 <= min_lr <= base_lr:
            raise ValueError(f"min_lr 必须落在 [0, base_lr]，得到 {min_lr}")
        self.base_lr = float(base_lr)
        self.gamma = float(gamma)
        self.min_lr = float(min_lr)

    def lr_at(self, step: int) -> float:
        if step <= 0:
            return self.base_lr
        return max(self.base_lr * self.gamma ** step, self.min_lr)

    def __repr__(self) -> str:
        return (
            f"ExponentialDecay(base_lr={self.base_lr:g}, gamma={self.gamma:g}, "
            f"min_lr={self.min_lr:g})"
        )


class Warmup:
    """在前 ``steps`` 步内从 ``start_lr`` 线性升到内层调度器的起点。

    ``start_lr`` 故意没有默认值：取 0 会让第一步参数完全不动，取多少是训练策略，
    不该由框架猜。常用做法是 ``base_lr / 100`` 或 ``base_lr / 1000``。

    warmup 之后完全委托给内层，且内层按**绝对步数**计算——所以
    ``Warmup(CosineDecay(lr, 1000), 100)`` 的余弦会在第 100 步从余弦曲线上
    第 100 步的位置继续，而不是从头开始。这让"总步数"只需在一处定义。
    """

    def __init__(self, schedule, steps: int, start_lr: float) -> None:
        if int(steps) < 1:
            raise ValueError(f"warmup 步数必须 >= 1，得到 {steps}")
        if start_lr < 0.0:
            raise ValueError(f"start_lr 不能为负，得到 {start_lr}")
        self.schedule = schedule
        self.steps = int(steps)
        self.start_lr = float(start_lr)
        self.target_lr = float(schedule.lr_at(self.steps))
        if self.target_lr < self.start_lr:
            raise ValueError(
                f"warmup 的终点 {self.target_lr:g} 低于起点 {self.start_lr:g}；"
                "warmup 必须是升的，请检查内层调度器与 start_lr"
            )

    def lr_at(self, step: int) -> float:
        if step >= self.steps:
            return self.schedule.lr_at(step)
        if step <= 0:
            return self.start_lr
        frac = step / self.steps
        return self.start_lr + (self.target_lr - self.start_lr) * frac

    def __repr__(self) -> str:
        return f"Warmup({self.schedule!r}, steps={self.steps}, start_lr={self.start_lr:g})"


def apply_lr(optimizer: Optimizer, schedule) -> float:
    """把调度器在"当前步"的值写进优化器，返回写入的学习率。

    步数取 ``optimizer.step_count``，即真实更新次数；梯度累积窗口内的多次反向
    不计入，因此不会让 warmup 提前结束。
    """
    lr = float(schedule.lr_at(optimizer.step_count))
    optimizer.lr = lr
    return lr
