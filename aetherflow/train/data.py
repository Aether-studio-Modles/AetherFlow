"""批数据迭代：把 ``dict[str, ndarray]`` 按第 0 维切成小批。

为什么不用 ``tf.data`` / ``torch.utils.data`` 那一套：

1. **这一层只需要"切一批、洗一次牌"。** Dataset / Sampler / collate_fn
   三层抽象是为了异构样本（图片解码、变长序列）准备的；这套 IR 的输入
   本来就是一组已经对齐的定长数组，多一层只会多一个可能对不齐的地方。
2. **洗牌必须可复现。** 每轮用 ``seed + 轮次`` 生成独立的 ``Generator``，
   而不是共享一个全局随机状态：全局 rng 的下一串数取决于之前谁用过它，
   跑两次顺序就不同，实验无法复现。
3. **每轮顺序不同，但同一轮的每一次迭代相同。** 这是能"续训时倒回复现"
   的前提：只要把轮次与批位置记进 meta，中断后从同一批继续就是同一条轨迹
   （见 :meth:`DataLoader.state_dict`）。
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
        # 当前轮已经产出的批数。0 表示"这一轮还没开始（或已经走完）"，
        # 非 0 表示迭代器还在这一轮的中间；续训从下一批接上。
        self._cursor = 0

    # -- 属性 ---------------------------------------------------------

    @property
    def size(self) -> int:
        """样本总数（不是批数）。"""
        return int(next(iter(self._arrays.values())).shape[0])

    @property
    def epoch(self) -> int:
        """已开始的轮次。``reset()`` 会把它归零。"""
        return self._epoch

    @property
    def cursor(self) -> int:
        """当前轮已经产出的批数；0 表示停在轮次边界上。"""
        return self._cursor

    def __len__(self) -> int:
        total = self.size
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size

    def reset(self) -> None:
        """把轮次与批位置归零：下一次迭代重放第一轮的顺序。"""
        self._epoch = 0
        self._cursor = 0

    # -- 续训位置 -----------------------------------------------------

    def state_dict(self) -> Dict[str, int]:
        """当前迭代位置，可直接放进训练存档的 meta。

        只存"走了多远"不够：洗牌顺序由 ``seed + 轮次`` 决定，配置换了以后
        同一个位置对应的是另一组样本。所以配置一起存，由
        :meth:`load_state_dict` 逐项校验。

        存的是**已经产出**的批数：在第 k 批之后存档，恢复后先产出第 k+1 批。
        """
        return {
            "epoch": self._epoch,
            "batch": self._cursor,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "size": self.size,
            "shuffle": int(self.shuffle),
            "drop_last": int(self.drop_last),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """恢复迭代位置；配置不一致时在**改动任何状态之前**报错。

        拒绝而不是"尽量对上"：配置不同意味着同一个游标指向另一批样本，
        静默恢复会把数据顺序悄悄换掉，而这种偏差在损失曲线上看不出来。
        """
        epoch, batch = self._parse_state(state)
        self._epoch = epoch
        self._cursor = batch

    def validate_state_dict(self, state: Mapping[str, object]) -> None:
        """只校验"这个位置能不能接上"，不改动任何状态。

        训练器需要在提交权重**之前**确认数据位置与配置一致；那时若调用
        :meth:`load_state_dict` 就可能出现"数据位置已改、权重没恢复"的半现场。
        """
        self._parse_state(state)

    def _parse_state(self, state: Mapping[str, object]):
        required = ("epoch", "batch", "seed", "batch_size", "size", "shuffle", "drop_last")
        missing = [key for key in required if key not in state]
        if missing:
            raise KeyError(f"数据迭代状态缺少 {missing}；无法确定从哪一批继续")
        epoch = int(state["epoch"])
        batch = int(state["batch"])
        if epoch < 0:
            raise ValueError(f"数据迭代状态的 epoch 不能为负，得到 {epoch}")
        if batch > 0 and epoch < 1:
            raise ValueError(
                f"数据迭代状态停在 epoch={epoch} 的第 {batch} 批；"
                "进行中的轮次编号至少是 1（epoch 表示已开始的轮次）"
            )
        if not 0 <= batch <= len(self):
            raise ValueError(
                f"数据迭代状态的 batch={batch} 不在 [0, {len(self)}]；"
                "存档的批位置超出了这一轮的批数"
            )
        current = self.state_dict()
        bad = [
            f"{key}: 存档 {state[key]!r} / 当前 {current[key]!r}"
            for key in ("seed", "batch_size", "size", "shuffle", "drop_last")
            if int(state[key]) != current[key]
        ]
        if bad:
            raise ValueError(
                "存档的数据迭代位置与当前 DataLoader 的配置不一致：" + "；".join(bad)
            )
        return epoch, batch

    # -- 迭代 ---------------------------------------------------------

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        # 游标非 0 表示上次是在一轮中间停下的：沿用**同一个轮次编号**重建
        # 洗牌顺序，否则恢复后拿到的是另一次置换。
        resume = self._cursor
        if resume == 0:
            epoch = self._epoch
            self._epoch += 1
        else:
            epoch = self._epoch - 1
        order = None
        if self.shuffle:
            order = np.random.default_rng(self.seed + epoch).permutation(self.size)
        return self._batches(order, resume)

    def _batches(self, order, start_batch: int) -> Iterator[Dict[str, np.ndarray]]:
        total = self.size
        for index, start in enumerate(range(0, total, self.batch_size)):
            if index < start_batch:
                continue
            stop = min(start + self.batch_size, total)
            if order is None:
                index_slice = slice(start, stop)
                count = stop - start
            else:
                index_slice = order[start:stop]
                count = int(index_slice.size)
            if self.drop_last and count < self.batch_size:
                self._cursor = 0
                return
            # 先推进游标再产出：调用方拿到这一批之后随时可以存档，
            # 那时记录的必须已经是"这一批吃过了"。
            self._cursor = index + 1
            # ascontiguousarray 只在不连续时才拷贝：花式索引本来就会产生新数组，
            # 顺序切片则保持视图，避免每批都白拷一份数据
            yield {
                name: np.ascontiguousarray(arr[index_slice])
                for name, arr in self._arrays.items()
            }
        # 只有完整走完一轮才回到轮次边界；迭代器被提前 close 时走到这里的是
        # GeneratorExit，游标保持原值，正好支撑"中途存档"。
        self._cursor = 0

    def __repr__(self) -> str:
        return (
            f"DataLoader(size={self.size}, batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, drop_last={self.drop_last}, "
            f"epoch={self._epoch}, cursor={self._cursor})"
        )
