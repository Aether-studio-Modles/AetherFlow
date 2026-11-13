"""AF-25：官方 Mamba-1 选择性扫描的参考实现与参数映射。

``models/mamba.py`` 是**受 Mamba 启发**的变体，不是 mamba-ssm 的复刻。三处
结构差异决定了它不能承诺官方权重兼容或性能继承：

1. dt 是逐通道的 ``dt_w·x + dt_b``；官方先把输入低秩投影到 ``dt_rank``，
   再投影回 ``d_inner``；
2. 读出前多一次 RMSNorm（官方 ``out_proj`` 前没有）；
3. 门控与归一化的位置只在"输入 → 卷积 → scan → 门控 → 投影"这条链上近似，
   残差与层间幅度因此不同。

这里做两件事，都是审计要求的前置：

``official_selective_scan``
    按 mamba-ssm ``selective_scan_ref`` 的语义逐步递推，用来对拍
    ``ssm.build_selective_scan_stateful``。它验证的是扫描公式本身——包括
    ``b = Δ·B·x`` 这一形式，官方参考实现也这么写，所以这一项**不是**偏离。

``map_official_params`` / ``from_official_checkpoint``
    把官方命名映射到本项目参数表。只有 ``A_log → A_raw`` 与 ``D → D_skip``
    是纯改名；其余项没有等价物，整块迁移会被显式拒绝——"改个参数名就算
    迁移"正是审计点名的误判。
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from .mamba import MambaSpec

__all__ = [
    "official_selective_scan",
    "map_official_params",
    "from_official_checkpoint",
]


def _silu(values: np.ndarray) -> np.ndarray:
    """``z·sigmoid(z)`` 的稳定写法：tanh 形式在两端都不溢出。"""
    return 0.5 * values * (1.0 + np.tanh(0.5 * values))


def official_selective_scan(
    u: np.ndarray,
    delta: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    *,
    D: Optional[np.ndarray] = None,
    z: Optional[np.ndarray] = None,
    h0: Optional[np.ndarray] = None,
    reset: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """官方参考语义的选择性扫描，形状沿用 mamba-ssm 的 ``(B, D, T)``。

    参数
    ----
    u, delta
        ``(batch, dim, steps)``；``delta`` 必须已过 softplus（与图里一致）。
    A
        ``(dim, states)``，连续时间系数，通常为负。
    B, C
        ``(batch, steps, states)``，随步变化——selective 的来源。
    D
        可选跳连 ``(dim,)``。
    z
        可选门控 ``(batch, dim, steps)``，按 ``y * silu(z)`` 作用。
    h0
        可选初始状态 ``(batch, dim, states)``。
    reset
        可选重置掩码 ``(batch, steps)``，真值位置先清空状态再累加。

    返回 ``(y, h_last)``，``y`` 为 ``(batch, dim, steps)``。内部用 f64 累加，
    这是参考实现，不承担与设备端 f32 逐位一致的义务。
    """
    u = np.asarray(u, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    if u.shape != delta.shape or u.ndim != 3:
        raise ValueError(f"u / delta 形状应为 (batch, dim, steps)，得到 {u.shape} / {delta.shape}")
    batch, dims, steps = u.shape
    if A.ndim != 2 or A.shape[0] != dims:
        raise ValueError(f"A 的形状应为 ({dims}, states)，得到 {A.shape}")
    states = A.shape[1]
    if B.shape != (batch, steps, states) or C.shape != (batch, steps, states):
        raise ValueError(
            f"B / C 形状应为 {(batch, steps, states)}，得到 {B.shape} / {C.shape}"
        )

    h = np.zeros((batch, dims, states), dtype=np.float64) if h0 is None else np.array(
        h0, dtype=np.float64, copy=True
    )
    if h.shape != (batch, dims, states):
        raise ValueError(f"h0 形状应为 {(batch, dims, states)}，得到 {h.shape}")

    y = np.empty_like(u)
    for t in range(steps):
        if reset is not None:
            mask = np.asarray(reset, dtype=bool)[:, t]
            h = np.where(mask[:, None, None], 0.0, h)
        a_t = np.exp(delta[:, :, t, None] * A[None, :, :])
        b_t = delta[:, :, t, None] * u[:, :, t, None] * B[:, t, None, :]
        h = a_t * h + b_t
        y[:, :, t] = np.einsum("bdn,bn->bd", h, C[:, t, :])

    if D is not None:
        skip = np.asarray(D, dtype=np.float64)
        if skip.shape != (dims,):
            raise ValueError(f"D 形状应为 {(dims,)}，得到 {skip.shape}")
        y = y + skip[None, :, None] * u
    if z is not None:
        gate = np.asarray(z, dtype=np.float64)
        if gate.shape != u.shape:
            raise ValueError(f"z 形状应与 u 一致，得到 {gate.shape} 与 {u.shape}")
        y = y * _silu(gate)
    return y, h


#: 官方参数里没有本项目等价物的那些项。
OFFICIAL_ONLY = (
    "dt_proj.weight",
    "dt_proj.bias",
    "x_proj.weight",
    "in_proj.weight",
    "conv1d.weight",
    "conv1d.bias",
    "out_proj.weight",
)


def map_official_params(
    official: Mapping[str, np.ndarray], spec: MambaSpec
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """把官方命名映射到本项目参数表，返回 ``(params, unmapped)``。

    ``params`` 里只有能一一对应的两项：``A_log → A_raw``（两者的 A 都是
    ``-exp(raw)``）与 ``D → D_skip``。``unmapped`` 列出官方还有、本项目没有
    等价物的参数——调用方必须检查它，见 :func:`from_official_checkpoint`。
    """
    if "A_log" not in official:
        raise ValueError("官方参数里缺少 A_log")

    a_log = np.asarray(official["A_log"])
    if a_log.shape != (spec.d_inner, spec.d_state):
        raise ValueError(
            f"A_log 形状应为 {(spec.d_inner, spec.d_state)}，得到 {a_log.shape}"
        )

    params: Dict[str, np.ndarray] = {"A_raw": np.array(a_log, copy=True)}
    if "D" in official:
        skip = np.asarray(official["D"])
        if skip.shape != (spec.d_inner,):
            raise ValueError(
                f"D 形状应为 {(spec.d_inner,)}，得到 {skip.shape}"
            )
        params["D_skip"] = np.array(skip, copy=True)

    unmapped = [name for name in OFFICIAL_ONLY if name in official]
    return params, unmapped


def from_official_checkpoint(
    official: Mapping[str, np.ndarray], spec: MambaSpec
) -> Dict[str, np.ndarray]:
    """整块迁移：任何一项没有等价物就拒绝，而不是只改参数名。

    错误信息给出三处结构差异，因为这三处才是"不能承诺兼容"的原因；仅仅
    形状对不上并不能解释为什么迁移后行为会变。
    """
    params, unmapped = map_official_params(official, spec)
    if unmapped:
        raise ValueError(
            "这些官方参数没有等价物，不能只改参数名就迁移："
            f"{', '.join(unmapped)}。本项目的 dt 是逐通道 dt_w·x + dt_b"
            "（官方是低秩 dt 投影），读出前还有一次 RMSNorm（官方没有），"
            "门控与归一化的位置也不同"
        )
    return params
