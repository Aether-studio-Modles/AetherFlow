"""f16 训练的梯度缩放（loss scaling）。

为什么 f16 需要缩放：f16 的最小正规数约 ``6.1e-5``，再小就进入次正规区
（精度骤降），小于约 ``6e-8`` 直接变成 0。深度网络里靠后的层梯度经过多次
连乘后很容易落进这个区间——它们会静默变成零，参数一动不动，而损失曲线
只是"学得慢一点"，看不出任何异常。

为什么缩放能解决：反向图对种子是线性的。把种子从 1 改成 ``scale``，
所有梯度同比放大，先落进 f16 的可表示范围，取回后再除以 ``scale``，
真实梯度就被完整保留了。

为什么用**改种子**而不是在 loss 上乘常量：静态图里 loss 的常量在构图期
就固定了，换 scale 就得重构图。种子是每次执行都可以换的输入，图完全不用动；
代价只有一处——缩放的乘除发生在**图外**，不产生任何 IR 节点。

两条刻意的约束：

1. **反缩放先升 f32 再除。** 在 f16 上直接除以 16384 会把刚刚被缩放救回来
   的小梯度再推回下溢区，等于白缩放。升精度后除，结果才是真实梯度。
2. **溢出就跳过这一步。** 梯度里出现 inf/nan 时把 scale 减半并**不更新参数**；
   把被污染的梯度交给优化器，一次就能把参数打成 nan，训练不可恢复。
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional

import numpy as np

from ._arrays import array_module, to_host
from .optim import Optimizer

__all__ = ["GradScaler"]


class GradScaler:
    """动态梯度缩放。

    典型用法（图里声明 f16 输入，参数在 numpy 侧保留 f32 master）：

        scaler = GradScaler()
        feed = {n: arr.astype(np.float16) for n, arr in store.arrays().items()}
        feed.update(batch)
        grads = backend.execute(bwd, {**feed, "dloss": scaler.backward_seed(np.float16)})
        scaler.step(opt, grads)      # 溢出时返回 False，参数与步数都不动

    ``step`` 只接受**还被 scale 乘着**的梯度；先自己调过 ``unscale`` 再
    交给 ``step`` 会二次相除，结果缩小 ``scale`` 倍。
    """

    def __init__(
        self,
        init_scale: float = 2.0 ** 14,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        max_scale: float = 2.0 ** 15,
    ) -> None:
        if init_scale <= 0.0:
            raise ValueError(f"init_scale 必须为正，得到 {init_scale}")
        if growth_factor <= 1.0:
            raise ValueError(f"growth_factor 必须大于 1，得到 {growth_factor}")
        if not 0.0 < backoff_factor < 1.0:
            raise ValueError(f"backoff_factor 必须落在 (0, 1)，得到 {backoff_factor}")
        if int(growth_interval) <= 0:
            raise ValueError(f"growth_interval 必须为正，得到 {growth_interval}")
        if max_scale < init_scale:
            raise ValueError(f"max_scale 不能小于 init_scale，得到 {max_scale} < {init_scale}")

        # 默认 2^14 起步、2^15 封顶：两个值都要能塞进 f16 种子（上限 65504）。
        # 不设上限的话，连续若干轮不溢出后 scale 会涨过 f16 能表示的范围，
        # 种子变成 inf，整张图的梯度全废——这不是理论风险，是默认参数的必然结果。
        self._scale = float(init_scale)
        self.max_scale = float(max_scale)
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self.growth_interval = int(growth_interval)
        self._growth = 0
        self.found_nonfinite = False
        self.skipped = 0

    # -- 状态 ---------------------------------------------------------

    @property
    def scale(self) -> float:
        return self._scale

    def state_dict(self) -> dict:
        """缩放状态（三个标量）。写进存档后，续训不必从 ``init_scale`` 重新摸索。

        只存"现在是多少"，不存构造参数：``init_scale`` 是起点，恢复的是终点。
        混在一起会让"换一组超参重跑"看起来像是恢复了旧状态。
        """
        return {"scale": self._scale, "growth": self._growth, "skipped": self.skipped}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """恢复缩放状态；先全部校验再一次性写回。

        ``scale`` 允许大于构造时的 ``max_scale``——存档可能来自另一组
        ``growth_interval``，把上限写死会让恢复变成"必须改超参才能加载"。
        但必须为正且有限：``inf`` 会让下一步的种子直接溢出。
        """
        missing = [key for key in ("scale", "growth", "skipped") if key not in state]
        if missing:
            raise KeyError(f"缩放状态缺少 {missing}")
        scale = float(state["scale"])
        growth = int(state["growth"])
        skipped = int(state["skipped"])
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"scale 必须是有限正数，得到 {scale!r}")
        if growth < 0:
            raise ValueError(f"growth 不能为负，得到 {growth}")
        if skipped < 0:
            raise ValueError(f"skipped 不能为负，得到 {skipped}")
        self._scale = scale
        self._growth = growth
        self.skipped = skipped

    def backward_seed(self, dtype=np.float16) -> np.ndarray:
        """反向图的种子：0 维数组，值等于当前 scale。

        种子的 dtype 必须与图里 loss 的 dtype 一致。若 scale 超出该 dtype
        的表示范围，这里直接报错——否则种子会变成 inf，整张图的梯度全废。
        """
        np_dtype = np.dtype(dtype)
        limit = float(np.finfo(np_dtype).max)
        if self._scale > limit:
            raise ValueError(
                f"scale {self._scale:g} 超出 {np_dtype} 的表示范围（上限 {limit:g}）；"
                "请把 init_scale 降到范围内，或改用 float32 种子"
            )
        return np.array(self._scale, dtype=np_dtype)

    # -- 反缩放与更新 -------------------------------------------------

    def unscale(self, grads: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """还原成 f32 梯度（副本）。同时记录本轮是否出现非有限值。

        设备态训练里梯度在显存：反缩放也在显存做，只在最后把"是否非有限"
        这一个布尔搬回主机（逐张量同步会让缩放变成每一步的串行点）。
        """
        scale = self._scale
        out: Dict[str, np.ndarray] = {}
        finite = None
        for name, grad in grads.items():
            xp = array_module(grad)
            arr = grad if isinstance(grad, xp.ndarray) else xp.asarray(grad)
            # 先升到 f32 再除：在 f16 上除会把刚被缩放救回来的小梯度再推下溢
            arr = arr.astype(np.float32, copy=False) / scale
            out[name] = arr
            ok = xp.isfinite(arr).all()
            finite = ok if finite is None else finite & ok
        self.found_nonfinite = finite is not None and not bool(to_host(finite))
        return out

    def step(self, optimizer: Optimizer, grads: Mapping[str, np.ndarray]) -> bool:
        """反缩放后走一步；梯度非有限时跳过更新并减半 scale。

        返回 ``True`` 表示这一步真的更新了参数；``False`` 表示被跳过
        （``optimizer.step_count`` 也不增加，Adam 的偏差修正因此不受影响）。
        """
        unscaled = self.unscale(grads)
        if self.found_nonfinite:
            self.backoff()
            return False
        return self._commit(optimizer, unscaled)

    def update(self, optimizer: Optimizer, grads: Mapping[str, np.ndarray]) -> bool:
        """对**已经反缩放**的梯度走一步。

        梯度累积的路径要用它：反缩放必须发生在每个微批上（否则窗口里混着
        不同 scale 的梯度），所以更新时不能再调一次 ``step``——那会二次相除，
        把这一步的更新量缩小 ``scale`` 倍。

        这里仍然检查一次有限性：钩子（裁剪、加噪）可能在反缩放之后才把梯度
        改坏，而一个 inf 交给 Adam 就能把参数永久打成 nan。
        """
        if self._nonfinite(grads):
            self.backoff()
            return False
        return self._commit(optimizer, grads)

    def backoff(self) -> None:
        """记一次溢出：scale 减半、增长计数清零、不更新参数。

        ``step`` / ``update`` 在检测到非有限梯度时调它；累积窗口被整窗丢弃时
        也直接调它——两种情形都是"这一批算术已经不可信"。
        """
        self._scale *= self.backoff_factor
        self._growth = 0
        self.skipped += 1

    def _commit(self, optimizer: Optimizer, grads: Mapping[str, np.ndarray]) -> bool:
        optimizer.step(grads)
        self._growth += 1
        if self._growth >= self.growth_interval:
            self._scale = min(self._scale * self.growth_factor, self.max_scale)
            self._growth = 0
        return True

    def _nonfinite(self, grads: Mapping[str, np.ndarray]) -> bool:
        """梯度里是否有非有限值；顺带把结论写进 ``found_nonfinite``。"""
        finite: Optional[object] = None
        for grad in grads.values():
            xp = array_module(grad)
            arr = grad if isinstance(grad, xp.ndarray) else xp.asarray(grad)
            ok = xp.isfinite(arr).all()
            finite = ok if finite is None else finite & ok
        if finite is None:
            return False
        self.found_nonfinite = not bool(to_host(finite))
        return self.found_nonfinite

    def __repr__(self) -> str:
        return (
            f"GradScaler(scale={self._scale:g}, growth={self._growth}/"
            f"{self.growth_interval}, skipped={self.skipped})"
        )
