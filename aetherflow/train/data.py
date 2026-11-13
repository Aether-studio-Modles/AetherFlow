"""批数据迭代：把 ``dict[str, ndarray]`` 按第 0 维切成小批。

为什么不用 ``tf.data`` / ``torch.utils.data`` 那一套：

1. **这一层只需要"切一批、洗一次牌"。** Dataset / Sampler / collate_fn
   三层抽象是为了异构样本（图片解码、变长序列）准备的；这套 IR 的输入
   本来就是一组已经对齐的定长数组，多一层只会多一个可能对不齐的地方。
2. **洗牌必须可复现。** 每轮用 ``seed + 轮次`` 生成独立的 ``Generator``，
   而不是共享一个全局随机状态：全局 rng 的下一串数取决于之前谁用过它，
   跑两次顺序就不同，实验无法复现。
3. **每轮顺序不同，但同一轮的每一次迭代相同。** 这是能"续训时倒回复现"
   的前提：只要把轮次记进 meta，中断后从同一轮重跑就是同一条轨迹。
"""

from __future__ import annotations

from typing import Dict, Iterator, Mapping

import numpy as np

__all__ = ["DataLoader"]


class DataLoader:
    """按第 0 维切批。

    ``arrays`` 里每个数组的第 0 维都是样本维，长度必须一致（形状其余的
    维度是每个样本自己的形状，可以是任意阶）。迭代产出
    ``dict[str, ndarray]``，键与 ``arrays`` 完全相同，可以直接喂给
    ``ParameterStore.feed(...)``。
    """

    def __init__(
        self,
        arrays: Mapping[str, np.ndarray],
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if not arrays:
            raise ValueError("DataLoader 至少需要一个数组")
        size = int(batch_size)
        if size <= 0:
            raise ValueError(f"batch_size 必须为正，得到 {batch_size}")

        normalized: Dict[str, np.ndarray] = {}
        first: int = -1
        for name, arr in arrays.items():
            arr = np.asarray(arr)
            if arr.ndim == 0:
                raise ValueError(f"数组 {name!r} 是标量，没有样本维")
            if first < 0:
                first = int(arr.shape[0])
            elif arr.shape[0] != first:
                raise ValueError(
                    f"数组 {name!r} 的样本数 {arr.shape[0]} 与其它数组的 {first} 不一致"
                )
            normalized[name] = arr

        self._arrays = normalized
        self.batch_size = size
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0

    # -- 属性 ---------------------------------------------------------

    @property
    def size(self) -> int:
        """样本总数（不是批数）。"""
        return int(next(iter(self._arrays.values())).shape[0])

    @property
    def epoch(self) -> int:
        """已开始的轮次。``reset()`` 会把它归零。"""
        return self._epoch

    def __len__(self) -> int:
        total = self.size
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size

    def reset(self) -> None:
        """把轮次归零：下一次迭代重放第一轮的顺序。"""
        self._epoch = 0

    # -- 迭代 ---------------------------------------------------------

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        # 取迭代器时就推进轮次：即使调用方没遍历完，这一轮的随机数也已经定下来了
        order = None
        if self.shuffle:
            order = np.random.default_rng(self.seed + self._epoch).permutation(self.size)
        self._epoch += 1
        return self._batches(order)

    def _batches(self, order) -> Iterator[Dict[str, np.ndarray]]:
        total = self.size
        for start in range(0, total, self.batch_size):
            stop = min(start + self.batch_size, total)
            if order is None:
                index = slice(start, stop)
                count = stop - start
            else:
                index = order[start:stop]
                count = int(index.size)
            if self.drop_last and count < self.batch_size:
                return
            # ascontiguousarray 只在不连续时才拷贝：花式索引本来就会产生新数组，
            # 顺序切片则保持视图，避免每批都白拷一份数据
            yield {
                name: np.ascontiguousarray(arr[index])
                for name, arr in self._arrays.items()
            }

    def __repr__(self) -> str:
        return (
            f"DataLoader(size={self.size}, batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, drop_last={self.drop_last}, epoch={self._epoch})"
        )
