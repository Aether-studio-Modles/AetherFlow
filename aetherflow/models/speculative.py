"""投机解码：用一份便宜的草稿先猜几个 token，再由目标模型一次验收。

这一层回答的问题是"一次前向只出一个 token，能不能让它确认多个"，而不是
"怎么把草稿模型训得更准"。它不新增算子、不改 IR：验收用的就是同一张
``head="all"`` 的流式图，一次喂 ``k`` 个草稿 token，拿到每个位置的目标分布。

算法只有一条，且只有**贪心**版本：

1. 目标模型在 ``prefix`` 末尾给出下一个 token 的分布，草稿模型给出 ``k`` 个
   候选 ``d0..d(k-1)``；
2. 把 ``d0..d(k-1)`` 一次送进目标图，得到每步之后的分布；于是
   ``argmax(pred_i)``（``pred_0`` 来自前缀末位置）就是目标对 ``d_i`` 的判定；
3. 取最长的、与草稿逐 token 相同的前缀（长度 ``m``），再从第一个分歧处的
   目标分布里取**一个奖励 token**；
4. 状态回滚到 ``prefix + 已接受前缀``，再把"已接受前缀 + 奖励 token"重跑一遍，
   得到可用于下一轮的精确状态与末位置分布。

第 4 步是必要的，不是保守：本项目的语言模型主干是 Mamba，递归状态**不能**
像注意力 KV 那样按 token 截断，所以拒绝之后只能从快照重算被接受的那一小段。
代价是"验收一次 + 提交一次"，收益是"多轮里一次前向产出多个 token"；这里
**不声称**吞吐提升，只保证输出与 ``generate(temperature=0)`` 逐位相同。

四条显式边界：

- **只支持贪心。** 带温度的投机采样需要按 ``min(1, p/q)`` 的接受率做拒绝
  采样，那是另一套正确性证明；这里直接不提供 ``temperature``。
- **重复惩罚 / top-k / top-p / 文法约束都不提供。** 它们改的是采样分布，验收
  必须与目标分布逐位一致，混进来就会让"验收"变成"另一个模型"。
- **只支持 batch=1。** 每行的接受长度不同，状态回滚会退化成"最短行决定整批"。
- **空草稿是弃权，不是错误。** 草稿返回空序列时退化成目标模型自己的一个
  token（典型场景是草稿算出下一步就是 EOS）；它不改变结果，也不虚报收益。

同一个循环有两个入口：:class:`SpeculativeDecoder` 自建图缓存，
:meth:`aetherflow.models.lm.LMRunner.generate_speculative` 复用会话的图缓存、
驻留权重与 ``prime`` 前缀。两者共用 ``_speculative_loop``，因此数值路径不会
分叉。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .lm import (
    LMSpec,
    _check_request,
    _host_array,
    _host_row,
    _prefill,
    _require_resident,
    _stream_graph,
    generate,
)

__all__ = [
    "DraftModel",
    "SpeculativeDecoder",
    "greedy_draft",
]

# 草稿模型的接口：给定历史与想要几个 token，返回候选 token。
DraftModel = Callable[[Sequence[int], int], Sequence[int]]


def _check_draft_tokens(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"draft_tokens 必须是 int，得到 {type(value).__name__}")
    if value < 1:
        raise ValueError(f"draft_tokens 必须 >= 1，得到 {value}")
    return value


# -- 状态快照与单次验收（两个入口共用） ---------------------------------


def _snapshot_state(backend, state) -> Dict[str, np.ndarray]:
    """把状态拷到主机：递归状态回滚只能靠拷贝，不能靠截断。"""
    return {
        name: np.array(_host_array(backend, value), copy=True)
        for name, value in state.items()
    }


def _feed_state(spec: LMSpec, inputs, ids: np.ndarray, state, *,
                resident: bool, params: Mapping[str, np.ndarray]):
    feed: Dict[str, np.ndarray] = {} if resident else dict(params)
    feed[inputs["ids"].name] = ids
    feed[inputs["reset"].name] = np.zeros(ids.shape, dtype=np.float32)
    for name in spec.state_shapes:
        feed[inputs[name].name] = state[name]
    return feed


def _verify_tokens(
    backend,
    spec: LMSpec,
    graph_for,
    tokens: Sequence[int],
    state,
    *,
    resident: bool,
    params: Mapping[str, np.ndarray],
):
    """把 ``tokens`` 一次送进目标图，返回 ``(末状态, 每步之后的分布)``。"""
    position = 0
    current = state
    pieces: List[np.ndarray] = []
    while position < len(tokens):
        length = min(spec.steps, len(tokens) - position)
        graph, inputs, _ = graph_for(length, "all")
        ids = np.asarray(
            [list(tokens[position : position + length])], dtype=np.int32
        )
        out = backend.execute(
            graph,
            _feed_state(spec, inputs, ids, current,
                        resident=resident, params=params),
        )
        current = spec.carry_state(out)
        pieces.append(_host_array(backend, out["logits"])[0])
        position += length
    return current, np.concatenate(pieces, axis=0)


def _speculative_loop(
    backend,
    params: Mapping[str, np.ndarray],
    spec: LMSpec,
    prompt: Sequence[int],
    draft: DraftModel,
    *,
    chunk: int,
    limit: int,
    stop: int,
    draft_tokens: int,
    resident: bool,
    graph_for,
    state=None,
    logits=None,
    stats: Optional[Dict[str, int]] = None,
) -> List[int]:
    """草稿 → 验收 → 提交的主循环。

    ``state`` / ``logits`` 都非 ``None`` 时直接作为起点（``LMRunner`` 已经把
    会话推进到 ``prompt`` 末尾，可能复用了 ``prime`` 的前缀），否则从
    ``prompt`` 全量预填充。``stats`` 由调用方持有，循环只累加。
    """
    counts = stats if stats is not None else {}
    counts.setdefault("rounds", 0)
    counts.setdefault("proposed", 0)
    counts.setdefault("accepted", 0)
    if state is None or logits is None:
        state, logits = _prefill(
            backend,
            params,
            [prompt],
            spec,
            chunk=chunk,
            resident=resident,
            graph_for=graph_for,
        )
    # 提交段的块长：至少装得下一轮"已接受 + 奖励"，但不越过单块上限。
    commit_chunk = min(spec.steps, max(chunk, draft_tokens + 1))
    history: List[int] = list(prompt)
    generated: List[int] = []

    while len(generated) < limit:
        remaining = limit - len(generated)
        wanted = min(draft_tokens, remaining)
        prefix_last = _host_row(backend, logits[0, -1, :])
        proposals = list(draft(tuple(history), wanted))
        if not proposals:
            # 草稿弃权（例如它自己算出下一步就是 EOS）：退化成目标自己的
            # 一个 token，不影响结果，也不算错误。
            proposals = [int(np.argmax(prefix_last))]
        if len(proposals) > wanted:
            raise ValueError(
                f"draft 返回了 {len(proposals)} 个 token，超过要求的 {wanted}"
            )
        for token in proposals:
            if isinstance(token, bool) or not isinstance(token, int):
                raise TypeError(
                    f"draft 返回的 token 必须是 int，得到 {token!r}"
                )
            if not 0 <= token < spec.vocab_size:
                raise ValueError(
                    f"draft 返回的 token {token} 越界：词表大小 "
                    f"{spec.vocab_size}"
                )

        snapshot = _snapshot_state(backend, state)
        verified_state, after = _verify_tokens(
            backend, spec, graph_for, proposals, state,
            resident=resident, params=params,
        )

        # 目标对 d_i 的判定：i=0 用前缀末位置，i>0 用前一个草稿位置之后
        accepted = 0
        for index, token in enumerate(proposals):
            if token == stop:
                # 草稿猜到了停止符：目标若也选它，序列就在这里结束；
                # 后面即使有 token 也不能接受（与 generate 的口径一致）。
                break
            predicted = (
                prefix_last if index == 0 else after[index - 1]
            )
            if int(np.argmax(predicted)) != token:
                break
            accepted += 1
        counts["rounds"] = counts.get("rounds", 0) + 1
        counts["proposed"] = counts.get("proposed", 0) + len(proposals)
        counts["accepted"] = counts.get("accepted", 0) + accepted

        if accepted < len(proposals):
            bonus_pred = (
                prefix_last if accepted == 0 else after[accepted - 1]
            )
        else:
            bonus_pred = after[-1]
        bonus = int(np.argmax(bonus_pred))

        for token in proposals[:accepted]:
            if len(generated) >= limit:
                break
            generated.append(token)
            history.append(token)
        if len(generated) >= limit:
            break
        if bonus == stop:
            break
        generated.append(bonus)
        history.append(bonus)
        if len(generated) >= limit:
            break

        if accepted == len(proposals):
            # 全接受：验收后的状态正好是 prefix+k，只需再喂奖励 token
            base_state = verified_state
            block = [bonus]
        else:
            # 被拒绝：递归状态无法截断，回滚快照后重跑"已接受 + 奖励"
            base_state = snapshot
            block = list(proposals[:accepted]) + [bonus]
        state, logits = _prefill(
            backend,
            params,
            [history],
            spec,
            chunk=commit_chunk,
            resident=resident,
            graph_for=graph_for,
            consumed=[len(history) - len(block)],
            state=base_state,
        )
    return generated


class SpeculativeDecoder:
    """持有一份图缓存与（可选）驻留权重的投机解码会话。

    与 :class:`aetherflow.models.lm.LMRunner` 同一套生命周期：图按
    ``(长度, head)`` 缓存、``resident=True`` 时权重只登记一次。数值路径与
    :func:`aetherflow.models.lm.generate` 在 ``temperature=0`` 下逐位一致——
    验收只使用目标模型的 argmax，草稿只影响"一次前向确认几个"，不影响结果。

    需要同时复用图缓存、驻留权重与 ``prime`` 前缀时，用
    ``LMRunner.generate_speculative``：它走同一个循环，但会话是服务的那个。
    """

    def __init__(
        self,
        backend,
        params: Mapping[str, np.ndarray],
        spec: Optional[LMSpec] = None,
        *,
        resident: bool = False,
        max_graphs: int = 16,
    ) -> None:
        spec = spec or LMSpec()
        if spec.batch != 1:
            raise ValueError(
                f"投机解码需要 batch=1 的 spec，得到 {spec.batch}；"
                "每行的接受长度不同，状态回滚无法整批共享"
            )
        if isinstance(max_graphs, bool) or not isinstance(max_graphs, int):
            raise TypeError(f"max_graphs 必须是 int，得到 {max_graphs!r}")
        if max_graphs < 1:
            raise ValueError(f"max_graphs 必须 >= 1，得到 {max_graphs}")
        if resident:
            _require_resident(backend, True, "SpeculativeDecoder(resident=True)")
        self.backend = backend
        self.params = dict(params)
        self.spec = spec
        self.resident = bool(resident)
        self.max_graphs = max_graphs
        self._graphs: "OrderedDict[Tuple[int, str], Any]" = OrderedDict()
        self._resident_ready = False
        self._closed = False
        # 最近一次 generate 的验收统计（只用于观测，不参与任何决策）
        self.last_stats: Dict[str, int] = {
            "rounds": 0,
            "proposed": 0,
            "accepted": 0,
        }

    # -- 图与驻留 -----------------------------------------------------

    def _graph_for(self, length: int, head: str):
        if self._closed:
            raise ValueError("SpeculativeDecoder 已关闭，请新建一个会话")
        key = (length, head)
        item = self._graphs.get(key)
        if item is None:
            item = _stream_graph(length, self.spec, head)
            self._graphs[key] = item
            while len(self._graphs) > self.max_graphs:
                self._graphs.popitem(last=False)
        else:
            self._graphs.move_to_end(key)
        if self.resident and not self._resident_ready:
            self.backend.resident_parameters(item[0], self.params)
            self._resident_ready = True
        return item

    @property
    def graph_count(self) -> int:
        return len(self._graphs)

    def close(self) -> None:
        """释放驻留权重并拒绝后续调用；重复关闭是幂等的。"""
        if self._closed:
            return
        self._closed = True
        if self._resident_ready:
            self.backend.release_parameters()
            self._resident_ready = False
        self._graphs.clear()

    def __enter__(self) -> "SpeculativeDecoder":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:
        return (
            f"SpeculativeDecoder(graphs={len(self._graphs)}, "
            f"resident={self.resident}, closed={self._closed})"
        )

    # -- 生成 ---------------------------------------------------------

    def generate(
        self,
        prompt: Sequence[int],
        draft: DraftModel,
        *,
        max_new_tokens: int = 16,
        draft_tokens: int = 4,
        eos_id: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> List[int]:
        """与 ``generate(..., temperature=0)`` 同参（去掉采样项）同返回值。

        ``draft(history, k)`` 必须返回**至多** ``k`` 个候选 token（至少一个），
        ``history`` 是 prompt 加已生成的部分。候选与目标判定不符时会被拒绝，
        因此草稿再差也只是少赚，不会改变结果。
        """
        if self._closed:
            raise ValueError("SpeculativeDecoder 已关闭，请新建一个会话")
        if not callable(draft):
            raise TypeError(f"draft 必须是可调用对象，得到 {draft!r}")
        wanted = _check_draft_tokens(draft_tokens)
        prompt, chunk, limit, stop = _check_request(
            self.spec, prompt, chunk_size, max_new_tokens, eos_id, None
        )
        self.last_stats = {"rounds": 0, "proposed": 0, "accepted": 0}
        if limit == 0:
            return []
        return _speculative_loop(
            self.backend,
            self.params,
            self.spec,
            prompt,
            draft,
            chunk=chunk,
            limit=limit,
            stop=stop,
            draft_tokens=wanted,
            resident=self.resident,
            graph_for=self._graph_for,
            stats=self.last_stats,
        )


def greedy_draft(
    backend,
    params: Mapping[str, np.ndarray],
    spec: Optional[LMSpec] = None,
    *,
    chunk_size: Optional[int] = None,
) -> DraftModel:
    """用**同一个**模型做草稿：每次从 ``history`` 重新贪心生成 ``k`` 个 token。

    这是一个**正确性参照**，不是性能方案：它每次都要重新预填充，比目标模型
    自身的逐 token 解码还贵。它的价值是让"全接受"这条路径在没有第二个模型的
    情况下也能被逐位验证——草稿与目标同构时，验收应当一路通过。
    """

    def propose(history: Sequence[int], k: int) -> List[int]:
        return list(
            generate(
                backend,
                params,
                list(history),
                spec,
                max_new_tokens=k,
                temperature=0.0,
                chunk_size=chunk_size,
            )
        )

    return propose
