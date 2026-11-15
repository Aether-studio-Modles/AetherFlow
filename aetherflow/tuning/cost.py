"""成本模型：不测也能排出候选的先后（5.242）。

调优面走到 5.241 时有一个具体问题：conv2d 的三个旋钮（``tile_w`` / ``roc`` /
``cb``）合起来是 4 × 3 × 4 = 48 个组合，其中还有一批**在形状上就不可行**
（``roc`` 不能整除 OC tile、``cb`` 撑爆 shared 预算）。把它们全部交给实测，
等于让驱动去逐个报错；而在真机上每测一个候选都要付一次同步与计时开销，
48 个组合的网格在"每形状一遍"的代价下很快就不可接受。

所以这一层做两件事，都不碰设备：

1. **可行性判定**：直接复用 ``lower/plan.py`` 的 ``plan_conv_tiled`` /
   ``plan_elementwise``。放不下的组合在这里就变成一条带原因的"不可行"，
   不进入测量队列——这也是唯一能保证"排序与后台真实铺法一致"的做法：
   估算用的派生量（block 线程数、shared 字节、输出瓦片数）全部取自同一份
   计划，而不是在这里第二次推导。
2. **成本排序**：按 roofline 的量级把配置折算成微秒，拆成
   ``arith`` / ``read`` / ``staging`` / ``sync`` / ``tail`` / ``dead`` 六项。
   排序与"为什么选它"都来自这六项，因此结论可核对：
   ``dominant`` 指出哪一项最大，``explain()`` 把它写成人话。

**这不是性能预测器。** 机器常数（每周期 FMA 数、每周期字节数、同步延时）
是量级假设而不是测量结论；它们只用来回答"在两个配置之间，哪一个的访存量
少一半、哪一个的空线程多一倍"这种**方向性问题**。真的要在某块卡上分胜负，
仍然必须实测（``tune`` / ``tune_paired``），成本模型只负责把候选队列排短、
把不可行的先剔掉。

**默认值永远排在第一位。** ``cost_ranked_tune`` 的测量顺序是"默认配置 +
按估算排序的其余候选"，而不是纯粹的估算顺序：估算只在 ``tie_epsilon``
之外才允许改变结论，默认配置在噪声范围内继续胜出。这条与 ``search.py``
的平局规则是同一条，不另立一套。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from ..lower.plan import (
    BLOCK_MAX,
    ConvTilePlan,
    auto_block,
    cdiv,
    plan_conv_tiled,
)
from .search import AllRejectedError, CandidateResult, TuningResult, measure_configs
from .space import TuningSpace

__all__ = [
    "MachineModel",
    "M40_LIKE",
    "V100_LIKE",
    "CostBreakdown",
    "conv_tile_cost",
    "elementwise_cost",
    "rank_configs",
    "RankedTuningResult",
    "cost_ranked_tune",
]


@dataclass(frozen=True)
class MachineModel:
    """成本模型需要的机器常数（**量级假设**，不是测量结论）。

    只有两个量真的影响排序：**每周期能算多少 FMA** 与**每周期能搬多少字节**；
    其余三个（同步延时、每 SM 线程上限、每 SM shared 上限）决定占用率与
    波次量化。``clock_hz`` 只把周期换算成微秒，取整不影响排序。
    """

    name: str = "generic"
    sm_count: int = 24
    # 全设备每时钟周期的 FMA 数（一次 FMA 记 1，不是 2 flops）。
    fma_per_cycle: int = 1536
    # 全设备每时钟周期能搬的字节数（显存带宽 / 频率）。
    bytes_per_cycle: float = 300.0
    clock_hz: float = 1.0e9
    # 一次 __syncthreads 的近似代价（周期）：包含栅栏与最慢线程的等待。
    sync_cycles: int = 500
    # 占用率的两条硬上限：每 SM 的驻留线程数与 block 数。
    max_threads_per_sm: int = 2048
    max_blocks_per_sm: int = 32
    # 每 SM 可用的 shared memory（M40 是 48KB，V100 是 96KB）。
    shared_per_sm: int = 48 * 1024

    def __post_init__(self) -> None:
        for field in (
            "sm_count",
            "fma_per_cycle",
            "sync_cycles",
            "max_threads_per_sm",
            "max_blocks_per_sm",
            "shared_per_sm",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"MachineModel.{field} 必须是正整数，收到 {value!r}")
        for field in ("bytes_per_cycle", "clock_hz"):
            value = getattr(self, field)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"MachineModel.{field} 必须为正数，收到 {value!r}")
        if self.sm_count * self.max_blocks_per_sm < 1:  # pragma: no cover - 已被上面拦下
            raise ValueError("MachineModel 的 SM 数与每 SM block 上限不能同时为 0")

    def us(self, cycles: float) -> float:
        """周期 → 微秒。"""
        return cycles / self.clock_hz * 1e6

    def occupancy(self, *, threads: int, shared_bytes: int) -> int:
        """每个 SM 能驻留多少个这样的 block（至少 1：放不下时驱动会拒绝，不是这里的事）。"""
        if threads < 1:
            raise ValueError(f"线程数至少为 1，收到 {threads}")
        by_threads = self.max_threads_per_sm // threads
        by_shared = (
            self.shared_per_sm // shared_bytes if shared_bytes > 0 else self.max_blocks_per_sm
        )
        return max(1, min(by_threads, by_shared, self.max_blocks_per_sm))


# 两个预设：M40（手上有实测结论的那张卡）与 V100（远端那台已释放的机器）。
# 数值取量级：M40 3072 个 FP32 核 / 288 GB/s @ ~1.0 GHz / 48KB shared；
# V100 5120 个 / 900 GB/s @ ~1.5 GHz / 96KB shared。
M40_LIKE = MachineModel(
    name="m40",
    sm_count=24,
    fma_per_cycle=1536,
    bytes_per_cycle=300.0,
    clock_hz=1.0e9,
)
V100_LIKE = MachineModel(
    name="v100",
    sm_count=80,
    fma_per_cycle=2560,
    bytes_per_cycle=600.0,
    clock_hz=1.5e9,
    shared_per_sm=96 * 1024,
)


@dataclass(frozen=True)
class CostBreakdown:
    """一个候选的估算结果：六项耗时（微秒）+ 派生量 + 结论。

    ``feasible=False`` 时 ``terms`` 为空、``total_us`` 为 ``inf``：把不可行
    折算成无穷大是为了让"按总耗时排序"永远不会把不可行项排到前面，而
    ``reason`` 保留原始报错（"roc 必须整除 OC tile"之类），不做二次概括。
    """

    config: Mapping[str, Any]
    terms: Tuple[Tuple[str, float], ...] = ()
    detail: Tuple[Tuple[str, int], ...] = ()
    feasible: bool = True
    reason: str = ""

    @property
    def total_us(self) -> float:
        if not self.feasible:
            return math.inf
        return round(sum(value for _, value in self.terms), 6)

    @property
    def dominant(self) -> str:
        """耗时最大的那一项；不可行或全为零时返回空串。

        平局时返回**先声明**的项（``terms`` 的顺序固定），因此同一个配置
        两次评估给出同一个答案。
        """
        if not self.feasible or not self.terms:
            return ""
        name, value = self.terms[0]
        for candidate, cost in self.terms[1:]:
            if cost > value:
                name, value = candidate, cost
        return name if value > 0 else ""

    def term(self, name: str) -> float:
        for candidate, value in self.terms:
            if candidate == name:
                return value
        raise KeyError(f"这个估算里没有项 {name!r}，可选：{[n for n, _ in self.terms]}")

    def explain(self) -> str:
        """一句话结论：可行时给"主导项 + 总估算"，不可行时给原因。"""
        if not self.feasible:
            return f"{dict(self.config)} 不可行：{self.reason}"
        parts = ", ".join(f"{name}={value:.3f}us" for name, value in self.terms)
        return f"{dict(self.config)} 估算 {self.total_us:.3f}us（主导 {self.dominant or '无'}；{parts}）"

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"config": dict(self.config), "feasible": self.feasible}
        if self.feasible:
            out["total_us"] = self.total_us
            out["dominant"] = self.dominant
            out["terms"] = {name: round(value, 6) for name, value in self.terms}
            out["detail"] = dict(self.detail)
        else:
            out["rejected"] = self.reason
        return out


CostFn = Callable[[Mapping[str, Any]], CostBreakdown]


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} 必须是正整数，收到 {value!r}")
    return value


def _breakdown(
    config: Mapping[str, Any],
    *,
    terms: Tuple[Tuple[str, float], ...],
    detail: Tuple[Tuple[str, int], ...],
) -> CostBreakdown:
    return CostBreakdown(config=dict(config), terms=terms, detail=detail)


def _infeasible(config: Mapping[str, Any], exc: Exception) -> CostBreakdown:
    return CostBreakdown(config=dict(config), feasible=False, reason=str(exc) or type(exc).__name__)


def conv_tile_cost(
    *,
    kh: int,
    kw: int,
    c: int,
    o: int,
    ho: int,
    wo: int,
    n: int = 1,
    machine: MachineModel = M40_LIKE,
    dtype_bytes: int = 4,
) -> CostFn:
    """分块直算 conv2d 的成本函数：``config -> CostBreakdown``。

    六个成本的来历（全部由计划层的派生量算出）：

    - ``arith``：``n * ho * wo * o * kh * kw * c`` 次 FMA，除以每周期 FMA 数；
    - ``read``：输入读数。每个线程算 ``roc`` 个输出、共享同一次读入，
      因此是 ``arith 的输入量 / roc``——``roc`` 越大，输入流量越小；
    - ``staging``：权重从全局搬进 shared 的总量。每个 block 搬
      ``toc * kh * kw`` 个元素、搬 ``ceil(c / cb)`` 次，块数是
      ``ceil(wo/tile_w) * ho * n * ceil(o/toc)``。``cb`` 越大搬得越少，
      但受 shared 预算约束（预算越界在计划层就拒绝）；
    - ``sync``：每次 staging 一前一后两次 ``__syncthreads``。栅栏在 **block 内**
      串行、同一波内的 block 并行，所以按"波次 × staging 次数"折算——按块数
      折算会把并发执行的同步串成一条链，在块多的形状上盖过算术与读；
    - ``tail``：波次量化。块数不是"每 SM 驻留数 × SM 数"的整数倍时，
      最后一波只有一部分 SM 有活干，浪费的比例按基础耗时的比例计入；
    - ``dead``：最后一个 ``tile_w`` 里超出 ``wo`` 的线程。它们不做事，
      但占着驻留槽，按"空槽比例 × 基础耗时"计入。

    配置里缺的键按计划层的默认推导处理（``None`` 等价于不写），因此这个
    函数既能给完整三维网格打分，也能给"只改一个旋钮"的候选打分。
    """
    for name, value in (("kh", kh), ("kw", kw), ("c", c), ("o", o), ("ho", ho), ("wo", wo)):
        _positive_int(value, name)
    _positive_int(n, "n")
    _positive_int(dtype_bytes, "dtype_bytes")
    if not isinstance(machine, MachineModel):
        raise TypeError(f"machine 必须是 MachineModel，得到 {type(machine).__name__}")

    khkw = kh * kw
    total_out = n * ho * wo * o
    macs = total_out * khkw * c

    def score(config: Mapping[str, Any]) -> CostBreakdown:
        knobs = dict(config)
        unknown = sorted(set(knobs) - {"tile_w", "roc", "cb"})
        if unknown:
            return CostBreakdown(
                config=knobs,
                feasible=False,
                reason=f"分块卷积不认识调优参数 {unknown}",
            )
        try:
            plan = plan_conv_tiled(
                kh=kh,
                kw=kw,
                c=c,
                o=o,
                ho=ho,
                wo=wo,
                tile_w=knobs.get("tile_w"),
                roc=knobs.get("roc"),
                cb=knobs.get("cb"),
            )
        except (ValueError, TypeError) as exc:
            return _infeasible(knobs, exc)

        blocks = _conv_blocks(plan, ho=ho, wo=wo, n=n)
        staging_chunks = cdiv(c, plan.cb)
        weight_elems = blocks * staging_chunks * plan.toc * khkw
        input_elems = macs / plan.roc

        arith_us = machine.us(macs / machine.fma_per_cycle)
        read_us = machine.us(input_elems * dtype_bytes / machine.bytes_per_cycle)
        staging_us = machine.us(weight_elems * dtype_bytes / machine.bytes_per_cycle)

        threads = plan.tile_w * plan.block_y
        per_sm = machine.occupancy(threads=threads, shared_bytes=plan.shared_bytes)
        capacity = per_sm * machine.sm_count
        waves = cdiv(blocks, capacity)
        # 同步是 **block 内** 的栅栏：同一波里的 block 并行，因此按"波次 ×
        # 每块的 staging 次数"计，而不是按块数计——按块数计会把并发执行
        # 的同步串成一条链，在块多的形状上直接盖过算术与读。
        sync_us = machine.us(waves * staging_chunks * 2 * machine.sync_cycles)

        base_us = arith_us + read_us + staging_us + sync_us
        # 最后一波空出来的部分按基础耗时的比例折算；块数正好整除时为 0。
        tail_us = (waves * capacity - blocks) / max(1, blocks) * base_us
        # 最后一个 x 瓦片里的空线程：它们不做事，但占着驻留槽。
        slots = cdiv(wo, plan.tile_w) * plan.tile_w
        dead_us = (slots - wo) / max(1, slots) * (arith_us + read_us)

        return _breakdown(
            knobs,
            terms=(
                ("arith", arith_us),
                ("read", read_us),
                ("staging", staging_us),
                ("sync", sync_us),
                ("tail", tail_us),
                ("dead", dead_us),
            ),
            detail=(
                ("blocks", blocks),
                ("threads", threads),
                ("blocks_per_sm", per_sm),
                ("waves", waves),
                ("shared_bytes", plan.shared_bytes),
                ("oc_tiles", plan.oc_tiles),
            ),
        )

    return score


def _conv_blocks(plan: ConvTilePlan, *, ho: int, wo: int, n: int) -> int:
    """输出瓦片总数：x 方向按 ``tile_w`` 切、y 方向一行一个块、OC 方向按 tile 切。"""
    return cdiv(wo, plan.tile_w) * ho * n * plan.oc_tiles


def elementwise_cost(
    n: int,
    *,
    machine: MachineModel = M40_LIKE,
    dtype_bytes: int = 4,
    vec: int = 1,
) -> CostFn:
    """逐元素内核的成本函数：``{"block": ...} -> CostBreakdown``。

    逐元素核只有三项：``read``、``write``（各 ``n * dtype_bytes``）、``tail``
    （波次量化）。算术量按每元素一次运算计，与访存相比小两个量级，因此并进
    ``read`` 里不再单列——把它单列只会让"主导项"在两个几乎相等的项之间
    反复摇摆。

    ``block=None`` 表示按规模自动选（与后端的默认行为一致），这里调
    ``auto_block`` 换出具体值；``vec`` 由调用方说明（真实后端还要判对齐）。
    """
    _positive_int(dtype_bytes, "dtype_bytes")
    if vec not in (1, 4):
        raise ValueError(f"vec 只支持 1 或 4，收到 {vec}")
    if n < 0:
        raise ValueError(f"元素数不能为负，收到 {n}")
    if not isinstance(machine, MachineModel):
        raise TypeError(f"machine 必须是 MachineModel，得到 {type(machine).__name__}")

    def score(config: Mapping[str, Any]) -> CostBreakdown:
        knobs = dict(config)
        unknown = sorted(set(knobs) - {"block"})
        if unknown:
            return CostBreakdown(
                config=knobs,
                feasible=False,
                reason=f"逐元素内核不认识调优参数 {unknown}",
            )
        block = knobs.get("block")
        if block is None:
            block = auto_block(n, vec=vec)
        try:
            threads = _positive_int(block, "block")
        except ValueError as exc:
            return _infeasible(knobs, exc)
        if threads > BLOCK_MAX or threads & (threads - 1):
            return CostBreakdown(
                config=knobs,
                feasible=False,
                reason=f"block 必须是 2 的幂且不超过 {BLOCK_MAX}，收到 {threads}",
            )

        per_thread = threads * vec
        blocks = max(1, cdiv(n, per_thread))
        moved = n * dtype_bytes
        read_us = machine.us(moved / machine.bytes_per_cycle)
        write_us = machine.us(moved / machine.bytes_per_cycle)
        per_sm = machine.occupancy(threads=threads, shared_bytes=0)
        capacity = per_sm * machine.sm_count
        waves = cdiv(blocks, capacity)
        base_us = read_us + write_us
        tail_us = (waves * capacity - blocks) / max(1, blocks) * base_us
        # 最后一个 block 里没被填满的线程（n 不整除 per_thread 时），按全部
        # 线程槽的比例计入：空线程不做事，但占着驻留槽。
        total_slots = blocks * per_thread
        dead_frac = 0.0 if n >= total_slots else (total_slots - n) / total_slots
        dead_us = dead_frac * base_us

        return _breakdown(
            knobs,
            terms=(
                ("read", read_us),
                ("write", write_us),
                ("tail", tail_us),
                ("dead", dead_us),
            ),
            detail=(
                ("blocks", blocks),
                ("threads", threads),
                ("blocks_per_sm", per_sm),
                ("waves", waves),
            ),
        )

    return score


def rank_configs(space: TuningSpace, cost_fn: CostFn) -> Tuple[CostBreakdown, ...]:
    """按成本估算给整个空间的配置排序：可行项按总耗时升序，不可行项殿后。

    **排序稳定**：总耗时相同时保持 ``space.configs()`` 的原顺序，不可行项
    之间同样保持原顺序。因此"第几个候选"是可复现的，不依赖浮点比较的
    偶然抖动或字典顺序。

    不可行项**不丢弃**：它们带着原始原因进入结果，这是"为什么没测这个
    组合"的唯一记录。把它们静默过滤掉，事后无法区分"空间里没有"和
    "有但放不下"。
    """
    if not isinstance(space, TuningSpace):
        raise TypeError(f"space 必须是 TuningSpace，得到 {type(space).__name__}")
    if not callable(cost_fn):
        raise TypeError(f"cost_fn 必须可调用，得到 {type(cost_fn).__name__}")
    scored = [(cost_fn(config), index) for index, config in enumerate(space.configs())]
    scored.sort(key=lambda item: (not item[0].feasible, item[0].total_us, item[1]))
    return tuple(item[0] for item in scored)


@dataclass(frozen=True)
class RankedTuningResult:
    """``cost_ranked_tune`` 的结果：实测结论 + 估算排序 + 省掉了多少个候选。

    ``ranking`` 是全空间的估算排序（含不可行项），``evaluated`` 是真正测过
    的个数，``skipped`` 是"估算判为可行但没轮到"的个数。三者分开，是为了
    让"估算把空间剪掉了多少"可以被核对，而不是只看到一个赢家。
    """

    result: TuningResult
    ranking: Tuple[CostBreakdown, ...]
    evaluated: int
    skipped: int

    @property
    def best(self) -> CandidateResult:
        return self.result.best

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result": self.result.to_dict(),
            "ranking": [item.to_dict() for item in self.ranking],
            "evaluated": self.evaluated,
            "skipped": self.skipped,
        }


def cost_ranked_tune(
    space: TuningSpace,
    cost_fn: CostFn,
    measure: Callable[[Mapping[str, Any]], Optional[float]],
    *,
    op: str = "",
    rounds: int = 1,
    tie_epsilon: float = 0.02,
    max_evals: Optional[int] = None,
) -> RankedTuningResult:
    """先估算排序、再只测前 ``max_evals`` 个可行候选。

    测量顺序是**默认配置在前、其余按估算升序**：默认配置必须留在第一位，
    否则 ``tie_epsilon`` 以内的"平局"会判给一个只是估算更低的候选，
    而 ``TuningResult.speedup`` 的基准也随之变成另一个配置。

    ``max_evals=None`` 表示不裁剪（可行项全测），此时结论与 ``tune`` 完全
    一致——这一点有测试钉住：估算只在裁剪时才改变"测了哪些"，不改变
    测量与平局的语义。

    空间里**没有一个可行组合**时抛 ``AllRejectedError``，与 ``tune`` 在
    全部候选都测失败时的行为一致：那不是"没找到更快的"，而是"这个形状
    在计划层就跑不起来"。
    """
    if max_evals is not None:
        if isinstance(max_evals, bool) or not isinstance(max_evals, int) or max_evals < 1:
            raise ValueError(f"max_evals 必须是正整数或 None，收到 {max_evals!r}")

    ranking = rank_configs(space, cost_fn)
    feasible = [item for item in ranking if item.feasible]
    if not feasible:
        raise AllRejectedError(
            op or "<unknown>",
            tuple(
                CandidateResult(config=dict(item.config), reason=item.reason)
                for item in ranking
            ),
        )

    defaults = space.default()
    ordered = [item.config for item in feasible]
    # 默认配置若可行，必须排第一；它本来就在候选里，这里只把它提到最前。
    for index, config in enumerate(ordered):
        if dict(config) == defaults:
            ordered.insert(0, ordered.pop(index))
            break
    chosen = ordered if max_evals is None else ordered[:max_evals]

    result = measure_configs(
        chosen,
        measure,
        op=op,
        rounds=rounds,
        tie_epsilon=tie_epsilon,
    )
    return RankedTuningResult(
        result=result,
        ranking=ranking,
        evaluated=len(chosen),
        skipped=len(feasible) - len(chosen),
    )
