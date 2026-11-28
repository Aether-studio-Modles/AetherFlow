"""带 KV cache 的增量注意力解码：把分页缓存里的 K/V 送进同一套 IR。

这一层回答的问题是"已经有 ``t`` 个 token 的 K/V，怎么只算当前这一个 token"，
而不是"怎么把 attention 写得更快"。它不新增算子：

- 图输入是当前 token 的隐状态 ``(B, 1, d)``、每层**已经取回**的 K/V
  ``(B, cache_steps, H, D)`` 与一个加性掩码；
- 图内把当前 token 的 k/v 拼到缓存末尾（``concat``），做一次单查询注意力，
  再走残差、FFN 与末端归一化；
- 图输出是当前 token 的隐状态与**本步新算出的 k/v**，由调用方写回缓存。

数值判据是"与 :func:`aetherflow.models.transformer.build_transformer` 在同样
权重、同样输入下逐位置一致"，不是"看起来像增量解码"。

三条边界：

1. **只做推理。** ``dropout`` 必须为 0：增量图里没有"重采样掩码"的位置，
   训练图与推理图是两张各自干净的图。
2. **只吃隐状态。** 位置编码由调用方在进图前加好；词表投影不在这里做。
   ``TransformerSpec.vocab`` 必须为 0，否则同名的 ``w_head`` 会被静默忽略。
3. **缓存布局由 :class:`aetherflow.runtime.paged_cache.PagedKVCache` 决定。**
   这里拿到的永远是"按块表拼好的连续副本"，它不知道页号，也不该知道。
"""

from __future__ import annotations

import math
import operator
from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np

from ..ir.graph import Graph, Value
from ..ir.types import f32
from ..runtime.paged_cache import PagedCacheFullError, PagedCacheManager, PagedKVCache
from ..runtime.prefix_store import DiskPrefixCache
from .transformer import TransformerSpec

__all__ = [
    "PagedDecodeScheduler",
    "PagedGenerator",
    "build_cached_decoder_step",
    "cached_forward",
]


def _layer_weight_names() -> Tuple[str, ...]:
    """每层参与增量计算的权重名（与 ``TransformerSpec.param_shapes`` 对齐）。"""
    return (
        "ln1_g",
        "ln1_b",
        "wq",
        "wk",
        "wv",
        "wo",
        "ln2_g",
        "ln2_b",
        "w1",
        "w2",
    )


def _all_param_names(spec: TransformerSpec) -> Tuple[str, ...]:
    """增量图需要的全部权重名（逐层十项 + 末端两个）。"""
    return tuple(
        f"l{layer}_{base}"
        for layer in range(spec.num_layers)
        for base in _layer_weight_names()
    ) + ("final_g", "final_b")


def _check_spec(spec: TransformerSpec) -> None:
    if spec.vocab != 0:
        raise ValueError(
            f"增量解码只输出隐状态，要求 vocab=0，得到 vocab={spec.vocab}；"
            "词表投影请在拿到隐状态之后自己接"
        )
    if spec.dropout != 0.0:
        raise ValueError(
            f"增量解码只支持 dropout=0（推理图），得到 {spec.dropout}；"
            "训练请用 build_transformer"
        )
    if spec.rng_input is not None:
        raise ValueError(
            "增量解码不接受 rng_input：它只在 dropout>0 时有意义，而这里只做推理"
        )


def build_cached_decoder_step(
    g: Graph, spec: TransformerSpec, *, cache_steps: int
) -> Tuple[Dict[str, Value], Value, Dict[str, Value]]:
    """构建单步增量解码图，返回 ``(inputs, y, new_kv)``。

    ``cache_steps`` 是缓存里**已保留的固定宽度**（含补零位），当前 token 的
    k/v 会拼在它末尾，因此注意力长度是 ``cache_steps + 1``。

    输入表：

    ``x``
        当前 token 的隐状态 ``(B, 1, d)``（位置编码已由调用方加好）。
    ``mask``
        加性偏置 ``(1, 1, 1, cache_steps + 1)``，0 表示可见、``-1e9`` 表示屏蔽。
        补零位与未来位都靠它屏蔽。
    ``l{i}_k`` / ``l{i}_v``
        第 ``i`` 层已缓存（或补零）的 K/V ``(B, cache_steps, H, D)``。
    ``l{i}_*``
        与 :class:`TransformerSpec` 同名的层权重。

    输出表：``y`` 与 ``l{i}_k_out`` / ``l{i}_v_out``（本步新算出的
    ``(B, 1, H, D)``，调用方负责写回缓存；名字带 ``_out`` 是因为图里输入与
    输出不能同名）。
    """
    if isinstance(cache_steps, bool) or not isinstance(cache_steps, int):
        raise TypeError(f"cache_steps 必须是 int，得到 {type(cache_steps).__name__}")
    if cache_steps < 1:
        raise ValueError(f"cache_steps 必须 >= 1，得到 {cache_steps}")
    _check_spec(spec)

    d, h, dh = spec.d_model, spec.num_heads, spec.head_dim
    batch = spec.batch
    scale_value = 1.0 / math.sqrt(dh)

    inputs: Dict[str, Value] = {
        "x": g.input("x", f32[batch, 1, d]),
        "mask": g.input("mask", f32[1, 1, 1, cache_steps + 1]),
    }
    for layer in range(spec.num_layers):
        for base in _layer_weight_names():
            inputs[f"l{layer}_{base}"] = g.input(
                f"l{layer}_{base}", f32[spec.param_shapes[f"l{layer}_{base}"]]
            )
        inputs[f"l{layer}_k"] = g.input(
            f"l{layer}_k", f32[batch, cache_steps, h, dh]
        )
        inputs[f"l{layer}_v"] = g.input(
            f"l{layer}_v", f32[batch, cache_steps, h, dh]
        )
    for base in ("final_g", "final_b"):
        inputs[base] = g.input(base, f32[spec.param_shapes[base]])

    scale = None
    if not spec.fused_attention:
        scale = g.constant(scale_value, dtype=f32, shape=(1,))

    x = inputs["x"]
    new_kv: Dict[str, Value] = {}
    for layer in range(spec.num_layers):
        prefix = f"l{layer}_"
        h1 = g.layernorm(
            x, inputs[prefix + "ln1_g"], inputs[prefix + "ln1_b"], axes=(-1,), eps=spec.eps
        )
        q = g.matmul(h1, inputs[prefix + "wq"])
        k = g.matmul(h1, inputs[prefix + "wk"])
        v = g.matmul(h1, inputs[prefix + "wv"])

        qh = g.transpose(
            g.reshape(q, shape=(batch, 1, h, dh)), perm=(0, 2, 1, 3)
        )
        kh = g.transpose(
            g.reshape(k, shape=(batch, 1, h, dh)), perm=(0, 2, 1, 3)
        )
        vh = g.transpose(
            g.reshape(v, shape=(batch, 1, h, dh)), perm=(0, 2, 1, 3)
        )
        # 缓存里的 (B, T, H, D) 换成 (B, H, T, D) 再与当前这一个位置拼接
        past_k = g.transpose(inputs[prefix + "k"], perm=(0, 2, 1, 3))
        past_v = g.transpose(inputs[prefix + "v"], perm=(0, 2, 1, 3))
        k_all = g.concat(past_k, kh, axis=2)
        v_all = g.concat(past_v, vh, axis=2)

        if spec.fused_attention:
            ctx = g.attention(
                qh,
                k_all,
                v_all,
                inputs["mask"],
                scale=scale_value,
                dropout=0.0,
                seed=0,
            )
        else:
            scores = g.matmul(qh, g.transpose(k_all, perm=(0, 1, 3, 2)))
            scores = g.add(g.mul(scores, scale), inputs["mask"])
            attn = g.softmax(scores, axis=-1)
            ctx = g.matmul(attn, v_all)

        merged = g.reshape(g.transpose(ctx, perm=(0, 2, 1, 3)), shape=(batch, 1, d))
        x = g.add(x, g.matmul(merged, inputs[prefix + "wo"]))

        h2 = g.layernorm(
            x, inputs[prefix + "ln2_g"], inputs[prefix + "ln2_b"], axes=(-1,), eps=spec.eps
        )
        act = g.gelu(g.matmul(h2, inputs[prefix + "w1"]))
        x = g.add(x, g.matmul(act, inputs[prefix + "w2"]))

        # 输出名必须与输入名（缓存里的 ``l{i}_k``）区分开：图里同名只能是
        # 同一个 Value，输入与输出不能共用一个名字。
        new_kv[prefix + "k_out"] = g.transpose(kh, perm=(0, 2, 1, 3))
        new_kv[prefix + "v_out"] = g.transpose(vh, perm=(0, 2, 1, 3))

    y = g.layernorm(x, inputs["final_g"], inputs["final_b"], axes=(-1,), eps=spec.eps)
    return inputs, y, new_kv


def _causal_step_mask(cache_steps: int, valid: int) -> np.ndarray:
    """第 ``valid`` 步的加性掩码：前 ``valid`` 个缓存位与末尾的当前位可见。"""
    mask = np.full((1, 1, 1, cache_steps + 1), -1e9, dtype=np.float32)
    mask[:, :, :, :valid] = 0.0
    mask[:, :, :, cache_steps] = 0.0
    return mask


def _host(backend, value) -> np.ndarray:
    """把（可能驻留在设备上的）数组取回主机；与 ``lm._host_array`` 同义。"""
    if isinstance(value, np.ndarray):
        return value
    convert = getattr(backend, "to_host", None)
    return np.asarray(convert(value) if convert is not None else value)


def cached_forward(
    backend,
    params: Dict[str, np.ndarray],
    spec: TransformerSpec,
    xs,
    *,
    block_size: int = 4,
) -> np.ndarray:
    """逐 token 走增量解码，返回 ``(1, T, d)`` 的隐状态序列。

    ``xs`` 是 ``(T, d)`` 的输入隐状态（位置编码请自己加好）。每一步都从
    :class:`PagedKVCache` 按块表取回该层的 K/V，执行一张固定形状的增量图，
    再把新算出的 k/v 写回缓存——与"整段跑一遍稠密图"逐位置一致。

    页宽 ``block_size`` 只影响分配粒度，不影响数值；``T`` 个 token 至少需要
    ``ceil(T / block_size)`` 页，这里按需分配，用完不回收（调用方持有返回值，
    缓存生命周期就是这次调用）。
    """
    _check_spec(spec)
    if isinstance(spec.batch, bool) or not isinstance(spec.batch, int):
        raise TypeError(
            f"cached_forward 需要静态的整数 batch，得到 {spec.batch!r}；"
            "符号 batch 无法在这里分配缓存"
        )
    if spec.batch != 1:
        raise ValueError(
            f"cached_forward 目前只支持 batch=1，得到 {spec.batch}；"
            "多序列请每行各自一份缓存"
        )
    values = np.asarray(xs, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != spec.d_model:
        raise ValueError(
            f"xs 形状必须是 (T, {spec.d_model})，得到 {values.shape}"
        )
    steps = values.shape[0]
    if steps < 1:
        raise ValueError("xs 至少要有一个位置")
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise TypeError(f"block_size 必须是 int，得到 {type(block_size).__name__}")
    if block_size < 1:
        raise ValueError(f"block_size 必须 >= 1，得到 {block_size}")

    cache = PagedKVCache(
        num_blocks=(steps + block_size - 1) // block_size,
        block_size=block_size,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        head_dim=spec.head_dim,
    )
    rid = 0
    cache.allocate(rid)

    g = Graph("cached_decoder_step")
    inputs, y, new_kv = build_cached_decoder_step(g, spec, cache_steps=steps)
    g.output(y, "y")
    for name, value in new_kv.items():
        g.output(value, name)

    layer_names = _all_param_names(spec)

    outs = []
    for step in range(steps):
        feed: Dict[str, np.ndarray] = {}
        for name in layer_names:
            if name not in params:
                raise KeyError(f"params 缺少权重 {name!r}")
            feed[inputs[name].name] = np.asarray(params[name], dtype=np.float32)
        feed[inputs["x"].name] = values[step][None, None, :]
        feed[inputs["mask"].name] = _causal_step_mask(steps, step)
        for layer in range(spec.num_layers):
            keys, vals = cache.gather(rid, layer, pad_to=steps)
            feed[inputs[f"l{layer}_k"].name] = keys[None]
            feed[inputs[f"l{layer}_v"].name] = vals[None]
        out = backend.execute(g, feed)
        outs.append(_host(backend, out["y"]).reshape(1, spec.d_model))
        keys_out = [
            _host(backend, out[f"l{layer}_k_out"])[0]
            for layer in range(spec.num_layers)
        ]
        values_out = [
            _host(backend, out[f"l{layer}_v_out"])[0]
            for layer in range(spec.num_layers)
        ]
        cache.append(rid, keys_out, values_out)
    return np.stack(outs, axis=1)


class PagedDecodeScheduler:
    """多序列共享一个分页 KV 池的交错增量解码。

    :func:`cached_forward` 是"一条序列、一次跑完"；服务里是"多条序列、每步
    轮转"，而池子装不下所有序列。这一层把两者接起来：

    - 构造时建一张固定形状的单步图（缓存宽度 ``max_steps``），权重只喂一次；
    - :meth:`submit` 分配序列并 **pin**——正在解码的请求不被自动换出；
    - :meth:`step` 只算一个 token：缓存里的 K/V 由 :class:`PagedCacheManager`
      自动换入，新 K/V 自动写回；页不够时它先换出**已挂起**的序列（LRU），
      所以所有已提交序列的总长度可以超过池容量；
    - :meth:`suspend` 解除 pin，让这条序列成为换出候选；:meth:`resume` 先换入
      再 pin 回来。挂起期间数据留在主机副本里，恢复后逐位继续；
    - :meth:`fork` 让一条新序列共享父序列的前缀（块级写时复制），同提示的多路
      采样 / 束搜索不必各存一份 K/V；
    - :meth:`finish` 回收序列与页。

    **失败面**：所有序列都 active（pinned）且池满时，``step`` / ``resume`` 抛
    ``PagedCacheFullError``。此时本步的 y 已算出但没有写回缓存，序列的长度与
    数据不变；调用方可以 ``suspend`` 一条低优先级请求后重试同一步，stub 不会
    重复计数。

    数值判据与 :func:`cached_forward` 相同：交错执行的每步隐状态与
    ``build_transformer`` 整段前向的对应位置逐位一致——池的大小与换出次数都
    不该改数值。
    """

    def __init__(
        self,
        backend,
        params: Dict[str, np.ndarray],
        spec: TransformerSpec,
        *,
        num_blocks: int,
        max_steps: int,
        block_size: int = 4,
        on_swap_out=None,
        on_swap_in=None,
    ) -> None:
        _check_spec(spec)
        if isinstance(spec.batch, bool) or not isinstance(spec.batch, int):
            raise TypeError(
                f"分页解码需要静态的整数 batch，得到 {spec.batch!r}"
            )
        if spec.batch != 1:
            raise ValueError(
                f"分页解码目前只支持 batch=1，得到 {spec.batch}；多序列请各用各的 rid"
            )
        for name, value in (
            ("num_blocks", num_blocks),
            ("block_size", block_size),
            ("max_steps", max_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} 必须是 int，得到 {type(value).__name__}")
            if value < 1:
                raise ValueError(f"{name} 必须 >= 1，得到 {value}")

        self.backend = backend
        self.params = params
        self.spec = spec
        self.max_steps = max_steps
        self.cache = PagedCacheManager(
            PagedKVCache(
                num_blocks=num_blocks,
                block_size=block_size,
                num_layers=spec.num_layers,
                num_heads=spec.num_heads,
                head_dim=spec.head_dim,
            ),
            on_swap_out=on_swap_out,
            on_swap_in=on_swap_in,
        )
        self._state: Dict[int, str] = {}

        g = Graph("paged_decode_step")
        self._inputs, self._y, self._new_kv = build_cached_decoder_step(
            g, spec, cache_steps=max_steps
        )
        g.output(self._y, "y")
        for name, value in self._new_kv.items():
            g.output(value, name)
        self._graph = g

        self._weight_feed: Dict[str, np.ndarray] = {}
        for name in _all_param_names(spec):
            if name not in params:
                raise KeyError(f"params 缺少权重 {name!r}")
            self._weight_feed[self._inputs[name].name] = np.asarray(
                params[name], dtype=np.float32
            )

    # -- 只读视图 -----------------------------------------------------

    def __contains__(self, rid: object) -> bool:
        return rid in self._state

    def _require(self, rid: int) -> str:
        if rid not in self._state:
            raise KeyError(f"序列 {rid!r} 没有提交；先 submit(rid)")
        return self._state[rid]

    def length(self, rid: int) -> int:
        self._require(rid)
        return self.cache.length(rid)

    def is_swapped(self, rid: int) -> bool:
        self._require(rid)
        return self.cache.is_swapped(rid)

    def stats(self) -> Dict[str, int]:
        info = self.cache.stats()
        info["active_sequences"] = sum(
            1 for state in self._state.values() if state == "active"
        )
        info["suspended_sequences"] = sum(
            1 for state in self._state.values() if state == "suspended"
        )
        return info

    # -- 生命周期 -----------------------------------------------------

    def submit(self, rid: int) -> None:
        """接入一条新序列：分配空缓存并 pin。"""
        if rid in self._state:
            raise ValueError(f"序列 {rid!r} 已经提交；先 finish(rid) 再复用这个 id")
        self.cache.allocate(rid)
        self.cache.pin(rid)
        self._state[rid] = "active"

    def fork(self, parent: int, child: int) -> None:
        """把 ``parent`` 已解码的前缀共享给新的 ``child``，之后各自解码。

        两条序列指向同一批物理页（块级写时复制），不额外占页；任一方之后
        ``step`` 都会先拆开被写的尾页，所以各自的 K/V 不会互相覆盖。子序列
        同样是 active（会被 pin），也要自己 ``suspend`` / ``finish``。

        ``parent`` 必须已提交且驻留：换出态请先 :meth:`resume`。``child`` 必须
        是没提交过的 id。典型用法是同一提示下的多路采样与束搜索。
        """
        self._require(parent)
        if child in self._state:
            raise ValueError(f"序列 {child!r} 已经提交；fork 需要一个新的 id")
        if self.cache.is_swapped(parent):
            raise ValueError(
                f"序列 {parent!r} 已换出；先 resume({parent}) 再 fork"
            )
        self.cache.fork(parent, child)
        self.cache.pin(child)
        self._state[child] = "active"

    def step(self, rid: int, x) -> np.ndarray:
        """解一个 token，返回长度为 ``d_model`` 的隐状态。

        ``x`` 取 ``(d,)`` 或 ``(1, 1, d)``；位置编码由调用方加好。返回的数组
        是主机态，设备后端会自动回传这一个向量。
        """
        state = self._require(rid)
        if state != "active":
            raise ValueError(f"序列 {rid!r} 处于挂起状态；先 resume(rid) 再解码")
        width = self.spec.d_model
        values = np.asarray(x, dtype=np.float32)
        if values.shape == (width,):
            values = values.reshape(1, 1, width)
        elif values.shape != (1, 1, width):
            raise ValueError(
                f"x 形状必须是 ({width},) 或 (1, 1, {width})，得到 {values.shape}"
            )
        valid = self.cache.length(rid)
        if valid >= self.max_steps:
            raise ValueError(
                f"序列 {rid!r} 已达到 max_steps={self.max_steps}，不能再解码"
            )

        feed = dict(self._weight_feed)
        feed[self._inputs["x"].name] = values
        feed[self._inputs["mask"].name] = _causal_step_mask(self.max_steps, valid)
        for layer in range(self.spec.num_layers):
            keys, vals = self.cache.gather(rid, layer, pad_to=self.max_steps)
            feed[self._inputs[f"l{layer}_k"].name] = keys[None]
            feed[self._inputs[f"l{layer}_v"].name] = vals[None]

        out = self.backend.execute(self._graph, feed)
        y = _host(self.backend, out["y"]).reshape(width)
        keys_out = [
            _host(self.backend, out[f"l{layer}_k_out"])[0]
            for layer in range(self.spec.num_layers)
        ]
        values_out = [
            _host(self.backend, out[f"l{layer}_v_out"])[0]
            for layer in range(self.spec.num_layers)
        ]
        # 页不够且都 pin 住时这里抛 PagedCacheFullError；append 是全有或全无，
        # 所以序列长度不变，重试同一步不会丢 token。
        self.cache.append(rid, keys_out, values_out)
        return y

    def suspend(self, rid: int, *, evict: bool = False) -> bool:
        """解除 pin，使 ``rid`` 成为换出候选；重复挂起返回 ``False``。

        ``evict=True`` 时立即换出（页马上归还），适合"这条请求要等很久"的
        调度决策；默认只解除保护，等别人需要页时再按 LRU 换出。
        """
        state = self._require(rid)
        if state == "suspended":
            return False
        self.cache.unpin(rid)
        self._state[rid] = "suspended"
        if evict and not self.cache.is_swapped(rid):
            self.cache.evict(rid)
        return True

    def resume(self, rid: int) -> int:
        """换入并按回 active；返回本次换入的页数（已在页上则为 0）。

        页不够且无可换出的序列时抛 ``PagedCacheFullError``，``rid`` 保持挂起态。
        """
        state = self._require(rid)
        if state == "active":
            return self.cache.ensure_resident(rid)
        pages = self.cache.ensure_resident(rid)
        self.cache.pin(rid)
        self._state[rid] = "active"
        return pages

    def finish(self, rid: int) -> None:
        """结束请求：解 pin、回收页，之后这个 id 可以重新 ``submit``。"""
        self._require(rid)
        self.cache.free(rid)
        del self._state[rid]

    def __repr__(self) -> str:
        return (
            f"PagedDecodeScheduler({self.cache!r}, max_steps={self.max_steps}, "
            f"active={sum(1 for s in self._state.values() if s == 'active')}, "
            f"suspended={sum(1 for s in self._state.values() if s == 'suspended')})"
        )


def _token_id(token, vocab: int, label: str) -> int:
    """把一个 token 规范成 ``int`` 并校验范围；bool 与浮点都显式拒绝。

    用 ``operator.index`` 而不是 ``isinstance(int)``：numpy 整数、枚举与其他
    实现 ``__index__`` 的整数类型都可以，浮点与字符串不行。
    """
    if isinstance(token, bool):
        raise TypeError(f"{label} 必须是整数 token id，得到 bool")
    try:
        value = operator.index(token)
    except TypeError:
        raise TypeError(
            f"{label} 必须是整数 token id，得到 {type(token).__name__}"
        ) from None
    if not 0 <= value < vocab:
        raise ValueError(f"{label}={value} 越界：词表大小是 {vocab}")
    return value


def _token_ids(tokens, vocab: int, label: str) -> List[int]:
    if isinstance(tokens, (str, bytes)):
        raise TypeError(f"{label} 必须是 token id 序列，不能是 {type(tokens).__name__}")
    return [_token_id(token, vocab, f"{label}[{i}]") for i, token in enumerate(tokens)]


class PagedGenerator:
    """分页 KV 池上的贪心生成闭环：``prompt`` 进、token 出。

    :class:`PagedDecodeScheduler` 解的是隐状态；真实生成还缺三件东西：把
    token 变成隐状态的嵌入与位置编码、把隐状态变成下一个 token 的词表头、
    以及逐 token 的停止条件。本类只补这三件，分页 / 换出换入 / 共享前缀
    全部复用调度器：

    - :meth:`submit` / :meth:`fork` / :meth:`suspend` / :meth:`resume` /
      :meth:`finish` 与调度器同名同义，另外维护每条序列的 token 历史
      （``fork`` 需要它来校验前缀真的相同）；
    - :meth:`feed` 喂一个 token，返回下一个 token 的贪心 id；
    - :meth:`prefill` 连续喂一段，返回下一 token；
    - :meth:`generate` 是 "prefill + 贪心续写 + 停止符" 的闭环。

    嵌入与位置编码在主机侧相加（``tok_emb[token] + pos[t]``）：增量图的输入
    本来就是隐状态，位置信息不必在设备上再算一遍。词表头是一张
    ``(1, d) → (1, V)`` 的小图，语义等同 :func:`build_transformer` 的末层
    ``x @ w_head``。

    **边界**：缓存宽度与位置表都按 ``max_steps`` 固定，``prompt`` 长度加
    ``max_new_tokens`` 超过它就显式报错——位置编码是有限表，越界没有定义。
    池满且所有序列都 active 时 ``step`` 照旧抛 ``PagedCacheFullError``，此时
    序列长度与 token 历史都不变，``suspend`` 别的请求后可以重试。

    **前缀缓存**（5.335，``prefix_cache=True`` 时默认开启）：``generate`` 不传
    ``fork_from`` 时，自动在所有驻留序列里找"token 序列是 prompt 严格前缀"
    且最长的一条，``fork`` 它并只 prefill 多出来的部分。父子共享页，命中不占
    新页；命中数与被复用的 token 数记在 :meth:`stats` 的 ``prefix_hits`` /
    ``prefix_reused_tokens`` / ``prefix_misses`` 里。换出态序列不参与匹配
    （``fork`` 要求驻留），等长序列也不行——生成需要一个尚未解码的位置。

    **落盘前缀缓存**（5.337，``disk_cache`` 非 ``None`` 时开启）：内存里没
    命中时去 :class:`aetherflow.runtime.DiskPrefixCache` 查同一判据的最长严格
    前缀，命中就把条目 KV 直接 ``append`` 进池，省掉整个 prefill；``generate``
    结束与 ``finish`` 释放前再把这条序列已解码的全长落盘（手工 ``feed`` /
    ``prefill`` 驱动的序列以后者为准），新进程首次请求就能复用。命中数、
    被复用的 token 数与落盘次数记在 :meth:`stats` 的 ``disk_hits`` /
    ``disk_reused_tokens`` / ``disk_misses`` / ``disk_stores`` 里。条目装不进
    池（``PagedCacheFullError``）时退回普通 prefill；``prefix_cache=False``
    同时关闭盘缓存查询——那是对"自动前缀复用"的显式拒绝。
    """

    def __init__(
        self,
        backend,
        params: Dict[str, np.ndarray],
        spec: TransformerSpec,
        *,
        num_blocks: int,
        max_steps: int,
        block_size: int = 4,
        on_swap_out=None,
        on_swap_in=None,
        prefix_cache: bool = True,
        disk_cache=None,
    ) -> None:
        if isinstance(spec.vocab, bool) or not isinstance(spec.vocab, int):
            raise TypeError(
                f"PagedGenerator 需要整数 vocab，得到 {spec.vocab!r}"
            )
        if spec.vocab < 1:
            raise ValueError(
                f"PagedGenerator 需要 vocab>=1（词表头是生成闭环的一部分），"
                f"得到 vocab={spec.vocab}；只解隐状态请用 PagedDecodeScheduler"
            )
        for name in ("tok_emb", "pos", "w_head"):
            if name not in params:
                raise KeyError(f"params 缺少权重 {name!r}")
        pos = np.asarray(params["pos"], dtype=np.float32)
        if pos.ndim != 3 or pos.shape[0] != 1 or pos.shape[2] != spec.d_model:
            raise ValueError(
                f"params['pos'] 形状必须是 (1, seq, {spec.d_model})，得到 {pos.shape}"
            )
        embed = np.asarray(params["tok_emb"], dtype=np.float32)
        if embed.shape != (spec.vocab, spec.d_model):
            raise ValueError(
                f"params['tok_emb'] 形状必须是 {(spec.vocab, spec.d_model)}，"
                f"得到 {embed.shape}"
            )
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise TypeError(f"max_steps 必须是 int，得到 {type(max_steps).__name__}")
        if max_steps < 1:
            raise ValueError(f"max_steps 必须 >= 1，得到 {max_steps}")
        if max_steps > pos.shape[1]:
            raise ValueError(
                f"max_steps={max_steps} 超过位置编码表长度 {pos.shape[1]}；"
                "生成图与位置表必须等宽"
            )

        self.backend = backend
        self.params = params
        self.spec = spec
        self.max_steps = max_steps
        self.vocab = spec.vocab
        self._embed = embed
        self._pos = pos[0]  # (seq, d)
        # 分页解码只解隐状态，词表头在下面的小图里；传一份 vocab=0 的 spec
        # 给调度器，避免它去要 tok_emb / w_head 这两个不属于它的权重。
        self._scheduler = PagedDecodeScheduler(
            backend,
            params,
            replace(spec, vocab=0),
            num_blocks=num_blocks,
            max_steps=max_steps,
            block_size=block_size,
            on_swap_out=on_swap_out,
            on_swap_in=on_swap_in,
        )

        g = Graph("paged_head")
        y = g.input("y", f32[1, spec.d_model])
        w_head = g.input("w_head", f32[spec.param_shapes["w_head"]])
        g.output(g.matmul(y, w_head), "logits")
        self._head_graph = g
        self._head_feed = {w_head.name: np.asarray(params["w_head"], dtype=np.float32)}

        self._tokens: Dict[int, List[int]] = {}
        self._prefix_cache = bool(prefix_cache)
        self._prefix_stats = {
            "prefix_hits": 0,
            "prefix_reused_tokens": 0,
            "prefix_misses": 0,
        }
        self._disk_cache = self._check_disk_cache(disk_cache)
        self._disk_stats = {
            "disk_hits": 0,
            "disk_reused_tokens": 0,
            "disk_misses": 0,
            "disk_stores": 0,
        }

    def _check_disk_cache(self, disk_cache):
        """盘缓存的页宽 / 形状 / dtype 必须与这份池一致：不一致就早失败。

        不做"临时转换再导入"——那会让同一个目录在两种配置下都能用，等于让
        配置失配从显式错误变成静默的慢路径。换配置请换目录。
        """
        if disk_cache is None:
            return None
        if not isinstance(disk_cache, DiskPrefixCache):
            raise TypeError(
                f"disk_cache 必须是 DiskPrefixCache 或 None，"
                f"得到 {type(disk_cache).__name__}"
            )
        store = self._scheduler.cache.cache
        problems = []
        for label, got, want in (
            ("block_size", disk_cache.block_size, store.block_size),
            ("num_layers", disk_cache.num_layers, store.num_layers),
            ("num_heads", disk_cache.num_heads, store.num_heads),
            ("head_dim", disk_cache.head_dim, store.head_dim),
            ("dtype", disk_cache.dtype, store.dtype),
        ):
            if got != want:
                problems.append(f"{label}={got!r}（池里是 {want!r}）")
        if problems:
            raise ValueError("disk_cache 与分页池不一致：" + "；".join(problems))
        return disk_cache

    # -- 只读视图 -----------------------------------------------------

    def __contains__(self, rid: object) -> bool:
        return rid in self._tokens

    def length(self, rid: int) -> int:
        return self._scheduler.length(rid)

    def is_swapped(self, rid: int) -> bool:
        return self._scheduler.is_swapped(rid)

    def stats(self) -> Dict[str, int]:
        info = self._scheduler.stats()
        info.update(self._prefix_stats)
        info.update(self._disk_stats)
        return info

    def tokens(self, rid: int) -> List[int]:
        """这条序列已经解码过的 token（``fork`` 前缀校验用的就是它）。"""
        if rid not in self._tokens:
            raise KeyError(f"序列 {rid!r} 没有提交；先 submit(rid)")
        return list(self._tokens[rid])

    # -- 生命周期 -----------------------------------------------------

    def submit(self, rid: int) -> None:
        if rid in self._tokens:
            raise ValueError(f"序列 {rid!r} 已经提交；先 finish(rid) 再复用这个 id")
        self._scheduler.submit(rid)
        self._tokens[rid] = []

    def fork(self, parent: int, child: int) -> None:
        if parent not in self._tokens:
            raise KeyError(f"序列 {parent!r} 没有提交；先 submit(rid)")
        if child in self._tokens:
            raise ValueError(f"序列 {child!r} 已经提交；fork 需要一个新的 id")
        self._scheduler.fork(parent, child)
        self._tokens[child] = list(self._tokens[parent])

    def _match_prefix(self, ids: List[int], *, exclude: int):
        """最长的"token 序列是 ``ids`` 严格前缀"的驻留序列；没有返回 ``None``。

        扫全部驻留序列而不是维护哈希索引：序列数在单进程里是个位数到几十，
        而索引要在 ``finish`` / ``fork`` / ``suspend`` 三处同步失效，维护成本
        高于收益。三个排除条件都是正确性要求：自身（还没 submit，本就不会
        命中，保留是为了防御调用顺序变化）、换出态（``fork`` 要求驻留）、
        等长（生成需要一个尚未解码的位置）。
        """
        best_rid, best_len = None, 0
        for rid, seen in self._tokens.items():
            size = len(seen)
            if rid == exclude or size <= best_len or size >= len(ids):
                continue
            if self._scheduler.is_swapped(rid):
                continue
            if ids[:size] != seen:
                continue
            best_rid, best_len = rid, size
        return best_rid

    def _load_from_disk(self, rid: int, ids: List[int]) -> int:
        """查盘并导入最长严格前缀；返回复用的 token 数（未命中为 0）。

        命中时在这里直接提交 ``rid`` 并把条目 KV 追加进池；装不下
        （``PagedCacheFullError``）就结束这条刚提交的序列并退回普通 prefill——
        "盘上有但池里放不下"是容量条件，不该让请求失败。
        """
        found = self._disk_cache.lookup(ids)
        if found is None:
            self._disk_stats["disk_misses"] += 1
            return 0
        length, payload = found
        self.submit(rid)
        try:
            self._scheduler.cache.append(rid, payload["keys"], payload["values"])
        except PagedCacheFullError:
            self.finish(rid)
            self._disk_stats["disk_misses"] += 1
            return 0
        self._tokens[rid] = list(payload["tokens"])
        self._disk_stats["disk_hits"] += 1
        self._disk_stats["disk_reused_tokens"] += length
        return length

    def _persist(self, rid: int) -> None:
        """把这条序列已解码的全长落盘；没有盘缓存或关闭自动复用时什么都不做。

        ``generate`` 结束与 ``finish`` 释放前都走这里——后者让手工 ``feed`` /
        ``prefill`` 驱动的序列也有落盘机会。调用方保证序列当前驻留：换出态的
        KV 不在池里，``gather`` 取不到。
        """
        if self._disk_cache is None or not self._prefix_cache:
            return
        seen = self._tokens.get(rid)
        if not seen:
            return
        keys, values = [], []
        for layer in range(self.spec.num_layers):
            key, value = self._scheduler.cache.gather(rid, layer)
            keys.append(key)
            values.append(value)
        before = self._disk_cache.stores
        self._disk_cache.store(seen, keys=keys, values=values)
        if self._disk_cache.stores > before:
            self._disk_stats["disk_stores"] += 1

    def suspend(self, rid: int, *, evict: bool = False) -> bool:
        return self._scheduler.suspend(rid, evict=evict)

    def resume(self, rid: int) -> int:
        return self._scheduler.resume(rid)

    def finish(self, rid: int) -> None:
        """结束请求：释放页，并在此之前把驻留序列落盘。

        ``feed`` / ``prefill`` 手工驱动的序列没有 ``generate`` 的结束点，
        ``finish`` 就是它们的结束信号。换出态跳过落盘：KV 已不在池里，
        ``gather`` 取不到；为一次释放先换入再落盘等于多付一次换入。
        """
        if rid not in self._tokens:
            raise KeyError(f"序列 {rid!r} 没有提交；先 submit(rid)")
        if not self._scheduler.is_swapped(rid):
            self._persist(rid)
        self._scheduler.finish(rid)
        del self._tokens[rid]

    # -- 解码 ---------------------------------------------------------

    def _logits(self, y: np.ndarray) -> np.ndarray:
        feed = dict(self._head_feed)
        feed["y"] = np.asarray(y, dtype=np.float32).reshape(1, self.spec.d_model)
        out = self.backend.execute(self._head_graph, feed)
        return _host(self.backend, out["logits"]).reshape(-1)

    def feed(self, rid: int, token) -> int:
        """喂一个 token，返回下一个位置上的贪心 token id。

        位置取序列当前长度（即这个 token 的位置）；位置编码与嵌入在主机侧
        相加后交给调度器。``PagedCacheFullError`` 时序列长度不变，可以挂起
        别的序列后重试。
        """
        if rid not in self._tokens:
            raise KeyError(f"序列 {rid!r} 没有提交；先 submit(rid)")
        token = _token_id(token, self.vocab, "token")
        position = self._scheduler.length(rid)
        if position >= self.max_steps:
            raise ValueError(
                f"序列 {rid!r} 已达到 max_steps={self.max_steps}，不能再解码"
            )
        x = self._embed[token] + self._pos[position]
        y = self._scheduler.step(rid, x)
        self._tokens[rid].append(token)
        return int(np.argmax(self._logits(y)))

    def prefill(self, rid: int, tokens) -> int:
        """连续喂一段 token，返回下一位置上的贪心 token id。"""
        seq = _token_ids(tokens, self.vocab, "tokens")
        if not seq:
            raise ValueError("prefill 至少要有一个 token")
        if self._scheduler.length(rid) + len(seq) > self.max_steps:
            raise ValueError(
                f"prefill 后长度 {self._scheduler.length(rid) + len(seq)} "
                f"超过 max_steps={self.max_steps}"
            )
        next_id = 0
        for token in seq:
            next_id = self.feed(rid, token)
        return next_id

    def generate(
        self,
        prompt,
        *,
        max_new_tokens: int = 16,
        eos_id=None,
        rid: int = 0,
        fork_from=None,
    ) -> List[int]:
        """贪心生成：返回新产生的 token（不含 ``prompt``）。

        ``fork_from`` 给出时先共享父序列的前缀：``prompt`` 必须以父序列已解码
        的全部 token 开头，只对多出来的部分做 prefill。生成结束后序列仍然
        驻留（可以继续 :meth:`feed` 或 :meth:`stats`），回收请显式
        :meth:`finish`。

        ``fork_from=None`` 且构造时 ``prefix_cache=True``（默认）时，按
        "最长严格前缀"自动选父序列——调用方不需要知道父序列的 id，命中只
        多出 prefill 剩余 token 的成本。内存里没命中且给了 ``disk_cache``
        时再查盘，命中就整段导入 KV、跳过 prefill。显式 ``fork_from`` 优先，
        且不在这两个分支记命中/未命中（那是调用方的显式决策，不是缓存判断）。
        """
        ids = _token_ids(prompt, self.vocab, "prompt")
        if not ids:
            raise ValueError("prompt 至少要有一个 token：没有输入就算不出下一个")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError(
                f"max_new_tokens 必须是 int，得到 {type(max_new_tokens).__name__}"
            )
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens 必须 >= 0，得到 {max_new_tokens}")
        if eos_id is not None:
            eos_id = _token_id(eos_id, self.vocab, "eos_id")
        if len(ids) + max_new_tokens > self.max_steps:
            raise ValueError(
                f"prompt({len(ids)}) + max_new_tokens({max_new_tokens}) = "
                f"{len(ids) + max_new_tokens} 超过 max_steps={self.max_steps}"
            )

        if fork_from is None:
            parent = self._match_prefix(ids, exclude=rid) if self._prefix_cache else None
            if parent is not None:
                consumed = len(self._tokens[parent])
                self._prefix_stats["prefix_hits"] += 1
                self._prefix_stats["prefix_reused_tokens"] += consumed
                self.fork(parent, rid)
            else:
                consumed = 0
                if self._prefix_cache and self._disk_cache is not None:
                    consumed = self._load_from_disk(rid, ids)
                if not consumed:
                    if self._prefix_cache:
                        self._prefix_stats["prefix_misses"] += 1
                    self.submit(rid)
        else:
            if fork_from not in self._tokens:
                raise KeyError(f"序列 {fork_from!r} 没有提交；先 submit(rid)")
            shared = self._tokens[fork_from]
            if len(ids) < len(shared) or ids[: len(shared)] != shared:
                raise ValueError(
                    f"fork_from={fork_from!r} 的前缀不一致：父序列已解码 "
                    f"{len(shared)} 个 token，prompt 只有 {len(ids)} 个或开头不同"
                )
            if len(ids) == len(shared):
                # 生成需要一个"尚未解码的位置"来取下一个 token。父序列最后一步
                # 的 logits 当时没有保存，这里不回滚重算，直接要求多喂一个。
                raise ValueError(
                    "fork 之后 prompt 必须比父序列多至少一个 token："
                    "父序列最后一步的 logits 没有保存，无法从同一位置续写"
                )
            consumed = len(shared)
            self.fork(fork_from, rid)

        next_id = self.prefill(rid, ids[consumed:])

        out: List[int] = []
        for _ in range(max_new_tokens):
            out.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break
            if len(out) == max_new_tokens:
                break
            next_id = self.feed(rid, next_id)
        self._persist(rid)
        return out

    def __repr__(self) -> str:
        return (
            f"PagedGenerator({self._scheduler.cache!r}, vocab={self.vocab}, "
            f"max_steps={self.max_steps}, sequences={len(self._tokens)}, "
            f"disk={'on' if self._disk_cache is not None else 'off'})"
        )
