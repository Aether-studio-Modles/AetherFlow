"""语言模型的数据侧：变长 token 序列 → 训练图要的 ``(ids, targets, mask)``。

审计 AF-34 的产品层里点名了 padding / label mask。这一层不引入算子，只把
"哪一位有下一个 token""哪一位是 padding"算清楚，交给
:func:`aetherflow.models.lm.masked_lm_loss`（one-hot）或
:func:`aetherflow.models.lm.masked_lm_loss_indexed`（整数下标）使用。

**为什么是 left padding**：Mamba 的状态沿时间推进，右侧补 pad 会让 pad 继续
推进状态，末状态与"只跑真实 token"不一致（5.104 已用 reset 掩码论证过同一
件事）；左侧补 pad 时真实 token 仍然连续落在末尾，decoder 的末状态与单独跑
该序列一致。代价是 batch 内每行的起点不同，所以 mask 必须逐行算。

**两种标签编码**：one-hot 与 logits 同形，``B*T*V`` 个 float32；整数下标只有
``B*T`` 个 int32，少 ``V`` 倍。V=32K、B=8、T=1K 时前者 1 GiB、后者 32 KB——
one-hot 在这个规模下根本传不动。整数编码是**扁平下标**
``(b*T + t)*V + y``，不是裸 token id：图里的 ``embedding`` 查表语义只认
第一维下标，而 CUDA 的逐元素核只承接 f16/f32，图内算不了整数偏移。
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

__all__ = ["lm_batch"]


def lm_batch(
    sequences: Sequence[Sequence[int]],
    vocab_size: int,
    *,
    steps: Optional[int] = None,
    pad_id: int = 0,
    labels: str = "onehot",
) -> Dict[str, np.ndarray]:
    """把一批变长 token 序列拼成静态形状的四个数组。

    返回：

    - ``ids``：``(B, steps)`` int32，左侧补 ``pad_id``
    - ``targets``：``t`` 位是 ``ids[t+1]`` 的标签；最后一位与 padding 位无标签
    - ``mask``：``(B, steps)`` f32，只有"有下一个真实 token"的位置是 1
    - ``reset``：``(B, steps)`` f32，每行**首个真实 token**位置为 1

    ``labels="onehot"``（默认）时 ``targets`` 是 ``(B, steps, vocab_size)``
    f32；与 logits 同形，因此不需要对 logits 切片——最后一位的 mask 是 0，
    它对损失的贡献恒为 0。这样少一个 ``slice`` 节点，也少一条"切片边界"的
    隐式约定。代价是每个位置都占满一份词表，V 大时传输与存储都不可接受。

    ``labels="index"`` 时 ``targets`` 是 ``(B, steps)`` int32 的**扁平下标**
    ``(b*steps + t)*vocab_size + y``，无标签位填 0（被 mask 遮住，且 0 一定是
    合法下标）。配合 :func:`aetherflow.models.lm.masked_lm_loss_indexed` 使用；
    图里用 ``embedding`` 的查表语义取回目标位的 logit，不物化 one-hot。

    ``reset`` 必须作为图输入交给卷积与 SSM：只遮 loss 不够，pad token 仍会
    推进状态、进入后续真实 token 的卷积窗口（``models/lm.py`` 的
    ``with_reset=True``）。它同时覆盖了块尾 reset 的跨块泄漏——那是同一张
    掩码在分块续接时的另一半语义。若要把这一批当作已有会话的续段，请把
    第 0 列的 reset 清掉，否则会丢掉调用方带进来的状态。
    """
    vocab = int(vocab_size)
    if vocab < 2:
        raise ValueError(f"vocab_size 至少为 2，得到 {vocab}")
    if not 0 <= int(pad_id) < vocab:
        raise ValueError(f"pad_id 必须落在 [0, {vocab})，得到 {pad_id}")
    mode = str(labels)
    if mode not in ("onehot", "index"):
        raise ValueError(f"labels 只能是 'onehot' / 'index'，得到 {labels!r}")

    rows = [[int(token) for token in seq] for seq in sequences]
    if not rows:
        raise ValueError("sequences 不能为空")
    if any(not row for row in rows):
        raise ValueError("sequences 里有空序列；至少给一个 BOS 或真实 token")
    for index, row in enumerate(rows):
        if any(token == int(pad_id) for token in row):
            raise ValueError(
                f"第 {index} 条序列里出现了 pad_id={pad_id}；"
                "padding 位与真实 token 必须可区分"
            )
        if any(not 0 <= token < vocab for token in row):
            raise ValueError(f"第 {index} 条序列里有超出 [0, {vocab}) 的 token")

    longest = max(len(row) for row in rows)
    total = longest if steps is None else int(steps)
    if total < longest:
        raise ValueError(f"steps={total} 小于最长序列 {longest}，请加长或先截断")

    batch = len(rows)
    if mode == "index":
        span = batch * total * vocab
        if span > np.iinfo(np.int32).max:
            # 下标本身溢出会静默回绕到别的 (b,t)，比 one-hot 的"占内存"更危险。
            raise ValueError(
                f"扁平下标会溢出 int32（B*T*V={span} > {np.iinfo(np.int32).max}）；"
                "请减小 batch / steps / vocab_size，或改用 labels='onehot'"
            )
        targets = np.zeros((batch, total), dtype=np.int32)
    else:
        targets = np.zeros((batch, total, vocab), dtype=np.float32)
    ids = np.full((batch, total), int(pad_id), dtype=np.int32)
    mask = np.zeros((batch, total), dtype=np.float32)
    reset = np.zeros((batch, total), dtype=np.float32)
    for index, row in enumerate(rows):
        start = total - len(row)
        ids[index, start:] = row
        reset[index, start] = 1.0
        # 位置 t 的目标是 ids[t+1]：只有 t+1 仍在真实区间时才有标签。
        for position in range(start, total - 1):
            token = int(ids[index, position + 1])
            if mode == "index":
                targets[index, position] = (index * total + position) * vocab + token
            else:
                targets[index, position, token] = 1.0
            mask[index, position] = 1.0
    return {"ids": ids, "targets": targets, "mask": mask, "reset": reset}
