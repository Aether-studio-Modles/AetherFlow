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
from typing import Dict, Tuple

import numpy as np

from ..ir.graph import Graph, Value
from ..ir.types import f32
from ..runtime.paged_cache import PagedCacheManager, PagedKVCache
from .transformer import TransformerSpec

__all__ = [
    "PagedDecodeScheduler",
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
