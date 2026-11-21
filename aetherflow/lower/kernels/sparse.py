"""事件驱动稀疏与脉冲内核：CSR 起点表、稀疏线性及其梯度。"""

from __future__ import annotations

from typing import Tuple
from ..kernel import (
    Assign,
    AtomicAdd,
    Axis,
    Bin,
    Call,
    Cast,
    Cond,
    Expr,
    For,
    If,
    Index,
    Kernel,
    Let,
    Load,
    Num,
    Paren,
    Scalar,
    Shared,
    Stmt,
    Store,
    Sync,
    Tensor,
    Var,
)
from ..indexing import Layout
from ..plan import CSR_SCAN_SLOTS
from .common import grid_stride


def _clamp_index(var: str, limit: str) -> Expr:
    """把 ``var`` 夹到 ``[0, limit - 1]``。

    只在越界已经记进 ``err`` 之后使用：夹取是为了让这次执行不非法访存，
    真正让主机侧抛异常的是那个标志，不是这里的结果。
    """
    return Cond(
        Bin("<", Var(var), Num("0")),
        Num("0"),
        Cond(
            Bin(">=", Var(var), Var(limit)),
            Bin("-", Var(limit), Num("1")),
            Var(var),
        ),
    )


def _event_segment() -> Tuple[Stmt, ...]:
    """切出第 ``s`` 步的事件区间 ``[lo, hi)``，越界时记 ``err`` 并夹回。

    捕获期不能把标量读回主机（读回会打断捕获），所以错误统一记在设备端的
    ``err[0]`` 上，由后端 ``_check_ids`` 在需要时同步并报错——与 embedding
    的越界检查走同一条通路。
    """
    lo, hi = Var("lo"), Var("hi")
    return (
        Let("long long", "lo", Index("offsets", Var("s"))),
        Let("long long", "hi", Index("offsets", Bin("+", Var("s"), Num("1")))),
        If(
            Bin(
                "||",
                Bin("||", Bin("<", lo, Num("0")), Bin(">", hi, Var("nnz"))),
                Bin("<", hi, lo),
            ),
            (Store("err", Num("0"), Num("1"), raw=True),),
        ),
        If(Bin("<", lo, Num("0")), (Assign(lo, Num("0")),)),
        If(Bin(">", hi, Var("nnz")), (Assign(hi, Var("nnz")),)),
        If(Bin("<", hi, lo), (Assign(hi, lo),)),
    )


def _event_row_check() -> Tuple[Stmt, ...]:
    """检查事件下标 ``r`` 落在 ``[0, rows)``，越界记 ``err`` 并夹回。"""
    row = Var("r")
    return (
        Let("long long", "r", Index("indices", Var("e"))),
        If(
            Bin("||", Bin("<", row, Num("0")), Bin(">=", row, Var("rows"))),
            (
                Store("err", Num("0"), Num("1"), raw=True),
                Assign(row, _clamp_index("r", "rows")),
            ),
        ),
    )


def spike_offsets() -> Kernel:
    """统计每行非零数并写 CSR 起点表（长度 ``T+1``）。

    两阶段放在**同一个 block** 里：先让每个线程跨步数若干行的非零数、写进
    ``offsets[t+1]``，``__syncthreads()`` 之后由 0 号线程串行做前缀和。前缀和
    因此只有 ``T`` 步，而不是旧实现里"单线程扫完 ``T×K`` 再累加"——大 ``T`` 下
    差的正是 ``K`` 倍的串行读。

    语义没有变：起点表的定义、越界行为与之前完全一致，``offsets`` 仍是唯一
    输出、仍然单输出，因此不需要新的 L2 能力。
    """
    hit = Bin(
        "!=",
        Load("spikes", _T_WIDTH.offset(("t", "k"), cast=None)),
        Num("0.f"),
    )
    return Kernel(
        name="spike_offsets",
        params=(
            Tensor("spikes"),
            Tensor("offsets", read_only=False, type_name="int"),
            Scalar("long long", "width"),
            Scalar("long long", "steps"),
        ),
        body=(
            # 阶段 1：并行数每行的非零数，先放在 offsets[t + 1] 上
            For(
                "t",
                Axis("tid"),
                Var("steps"),
                Axis("bdim"),
                (
                    Let("long long", "c", Num("0")),
                    For(
                        "k",
                        Num("0"),
                        Var("width"),
                        Num("1"),
                        (
                            If(
                                hit,
                                (
                                    Assign(
                                        Var("c"),
                                        Bin("+", Var("c"), Num("1")),
                                    ),
                                ),
                            ),
                        ),
                        incr="++",
                    ),
                    Store("offsets", Bin("+", Var("t"), Num("1")), Var("c"), raw=True),
                ),
            ),
            Sync(),
            # 阶段 2：0 号线程把 T 个计数就地变成前缀和
            If(
                Bin("==", Axis("tid"), Num("0")),
                (
                    Let("long long", "acc", Num("0")),
                    For(
                        "t",
                        Num("0"),
                        Var("steps"),
                        Num("1"),
                        (
                            Let(
                                "long long",
                                "cur",
                                Index("offsets", Bin("+", Var("t"), Num("1"))),
                            ),
                            Store("offsets", Var("t"), Var("acc"), raw=True),
                            Assign(Var("acc"), Bin("+", Var("acc"), Var("cur"))),
                        ),
                        incr="++",
                    ),
                    Store("offsets", Var("steps"), Var("acc"), raw=True),
                ),
            ),
        ),
    )


def spike_csr_serial() -> Kernel:
    """``spike_csr`` 的旧铺法：一个线程负责一行，沿列号串行。

    保留只为对照（5.176 的 ``benchmarks/spike_csr_scan.py`` 在实例上盖掉
    ``_use_scan_csr`` 后走这里），生产路径恒为 :func:`spike_csr`。数值语义
    两者相同——都是按列号递增写同一个 ``indices``。
    """
    hit = Bin(
        "!=",
        Load("spikes", _T_WIDTH.offset(("t", "k"), cast=None)),
        Num("0.f"),
    )
    lo, hi = Var("lo"), Var("hi")
    return Kernel(
        name="spike_csr_serial",
        params=(
            Tensor("spikes"),
            Tensor("offsets", type_name="int"),
            Tensor("indices", read_only=False, type_name="int"),
            Scalar("long long", "capacity"),
            Scalar("long long", "width"),
            Scalar("long long", "steps"),
            Tensor("err", read_only=False, type_name="int"),
        ),
        body=grid_stride(
            Var("steps"),
            *(
                Let("const long long", "t", Var("i")),
                Let("long long", "lo", Index("offsets", Var("t"))),
                Let(
                    "long long",
                    "hi",
                    Index("offsets", Bin("+", Var("t"), Num("1"))),
                ),
                If(
                    Bin(
                        "||",
                        Bin("||", Bin("<", lo, Num("0")), Bin(">", hi, Var("capacity"))),
                        Bin("<", hi, lo),
                    ),
                    (Store("err", Num("0"), Num("1"), raw=True),),
                ),
                If(Bin("<", lo, Num("0")), (Assign(lo, Num("0")),)),
                If(
                    Bin(">", hi, Var("capacity")),
                    (Assign(hi, Var("capacity")),),
                ),
                If(Bin("<", hi, lo), (Assign(hi, lo),)),
                Let("long long", "at", Var("lo")),
                For(
                    "k",
                    Num("0"),
                    Var("width"),
                    Num("1"),
                    (
                        If(
                            hit,
                            (
                                If(
                                    Bin("<", Var("at"), Var("hi")),
                                    (
                                        Store("indices", Var("at"), Var("k"), raw=True),
                                        Assign(
                                            Var("at"),
                                            Bin("+", Var("at"), Num("1")),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                    incr="++",
                ),
            ),
        ),
    )


def spike_csr() -> Kernel:
    """按起点表把每行的非零列号写进 ``indices[capacity]``。

    一个 **block** 负责一行（5.176）：行内必须按列号递增写，才能与稠密矩阵
    乘的累加顺序一致（``sparse_linear`` 逐事件累加），所以并行化不能改成
    "一个线程一个列号"。做法是分段扫——每段 ``blockDim.x`` 个列号，段内做一次
    块内排他前缀和（``block_scan_sum``），第 ``tid`` 个线程拿到"本段排在自己
    前面的命中数"，写位置是 ``lo + 已写 + 前缀``。段间串行、段内并行，递增
    顺序因此天然保持；段数 ``ceil(width / blockDim.x)`` 对全块一致，不会出现
    一半线程跳过 ``__syncthreads`` 的情况。

    起点表越界（倒序或超过容量）记 ``err`` 并夹回：夹取只保证不非法访存，
    主机侧仍会抛错——与 ``sparse_linear`` 的越界检查同一条通路。
    """
    t = Var("t")
    tid = Axis("tid")
    bdim = Axis("bdim")
    lo, hi = Var("lo"), Var("hi")
    k = Var("k")
    segment = Var("seg")
    prefix = Var("pre")
    written = Var("written")

    # ``? :`` 的分支在 C 里是短路的：``k`` 越界时不读 ``spikes``，所以这里
    # 不必先夹取索引。命中记 1、未命中记 0，恰好就是前缀和的被扫描值。
    counted = Cond(
        Bin("<", k, Var("width")),
        Cond(
            Bin(
                "!=",
                Load("spikes", Bin("+", Bin("*", t, Var("width")), k)),
                Num("0.f"),
            ),
            Num("1"),
            Num("0"),
        ),
        Num("0"),
    )
    position = Bin("+", Bin("+", lo, written), prefix)
    total_slot = Index("smem", bdim)
    taken = Var("taken")
    written_hit = Bin("==", taken, Num("1"))

    return Kernel(
        name="spike_csr",
        params=(
            Tensor("spikes"),
            Tensor("offsets", type_name="int"),
            Tensor("indices", read_only=False, type_name="int"),
            Scalar("long long", "capacity"),
            Scalar("long long", "width"),
            Scalar("long long", "steps"),
            Tensor("err", read_only=False, type_name="int"),
        ),
        body=(
            Let("const long long", "t", Cast("long long", Axis("bid"))),
            Shared("int", "smem", CSR_SCAN_SLOTS),
            Let("long long", "lo", Index("offsets", t)),
            Let("long long", "hi", Index("offsets", Bin("+", t, Num("1")))),
            If(
                Bin(
                    "||",
                    Bin(
                        "||",
                        Bin("<", lo, Num("0")),
                        Bin(">", hi, Var("capacity")),
                    ),
                    Bin("<", hi, lo),
                ),
                (Store("err", Num("0"), Num("1"), raw=True),),
            ),
            If(Bin("<", lo, Num("0")), (Assign(lo, Num("0")),)),
            If(
                Bin(">", hi, Var("capacity")),
                (Assign(hi, Var("capacity")),),
            ),
            If(Bin("<", hi, lo), (Assign(hi, lo),)),
            Let(
                "long long",
                "segments",
                # 打印器不推断优先级：分子的加法必须自带括号，否则会变成
                # ``width + blockDim.x - 1 / blockDim.x``。
                Bin(
                    "/",
                    Paren(Bin("-", Bin("+", Var("width"), bdim), Num("1"))),
                    bdim,
                ),
            ),
            Let("long long", "written", Num("0")),
            For(
                "seg",
                Num("0"),
                Var("segments"),
                Num("1"),
                (
                    Let(
                        "long long",
                        "k",
                        Bin("+", Bin("*", segment, bdim), Cast("long long", tid)),
                    ),
                    Let("int", "taken", counted),
                    Let(
                        "int",
                        "pre",
                        Call("block_scan_sum", (taken, Var("smem"))),
                    ),
                    If(
                        Bin("&&", written_hit, Bin("<", position, hi)),
                        (Store("indices", position, Cast("int", k), raw=True),),
                    ),
                    If(
                        Bin("&&", written_hit, Bin(">=", position, hi)),
                        (Store("err", Num("0"), Num("1"), raw=True),),
                    ),
                    Assign(written, Bin("+", written, total_slot)),
                ),
                incr="++",
                type_name="long long",
            ),
            # 参考实现要求"本行命中数 == hi - lo"；多了在段内已经记过，少了
            # 只能等全部段扫完才知道。两侧都对上才算 offsets 与脉冲一致。
            If(
                Bin("!=", Bin("+", lo, written), hi),
                (Store("err", Num("0"), Num("1"), raw=True),),
            ),
        ),
    )


def sparse_linear() -> Kernel:
    """事件驱动稀疏线性：``out[s, n] = Σ_e values[e] * W[indices[e], n]``。

    ``e`` 扫的是第 ``s`` 步**真正发放的事件**（CSR 的一段），而不是全部 K 个
    神经元——这就是"事件驱动"的全部含义：稠密的 ``(S, K)`` 脉冲矩阵从不
    物化，读的事件数只与发放率成正比。

    输出本身是稠密的 ``(S, N)``，所以一个线程负责一个输出元素；选"按输出
    收集"而不是"按事件散射"，是为了每个输出只写一次、不需要原子加。
    """
    weight_at = _ROW_COLS.offset(("r", "n"), cast=None)
    return Kernel(
        name="sparse_linear",
        params=(
            Tensor("weights"),
            Tensor("indices", type_name="int"),
            Tensor("values"),
            Tensor("offsets", type_name="int"),
            Tensor("out", read_only=False),
            Scalar("long long", "cols"),
            Scalar("long long", "rows"),
            Scalar("long long", "nnz"),
            Scalar("long long", "total"),
            Tensor("err", read_only=False, type_name="int"),
        ),
        body=grid_stride(
            Var("total"),
            *(
                Let("const long long", "s", Bin("/", Var("i"), Var("cols"))),
                Let("const long long", "n", Bin("%", Var("i"), Var("cols"))),
                *_event_segment(),
                Let("float", "acc", Num("0.f")),
                For(
                    "e",
                    Var("lo"),
                    Var("hi"),
                    Num("1"),
                    (
                        *_event_row_check(),
                        Assign(
                            Var("acc"),
                            Bin(
                                "+",
                                Var("acc"),
                                Bin(
                                    "*",
                                    Load("values", Var("e")),
                                    Load("weights", weight_at),
                                ),
                            ),
                        ),
                    ),
                    incr="++",
                ),
                Store("out", Var("i"), Var("acc")),
            ),
        ),
    )


def sparse_linear_weight_grad() -> Kernel:
    """``sparse_linear`` 对权重的梯度：把 ``values[e] * dout[s]`` 散射回 ``idx[e]`` 行。

    同一行会被多个事件命中（同一神经元在 batch 内多次发放、或同一 batch 里
    重复的 index），写回必须原子加；输出缓冲由后端先清零。
    """
    dst = _ROW_COLS.offset(("r", "n"), cast=None)
    return Kernel(
        name="sparse_linear_weight_grad",
        params=(
            Tensor("dout"),
            Tensor("indices", type_name="int"),
            Tensor("values"),
            Tensor("offsets", type_name="int"),
            Tensor("dweights", read_only=False),
            Scalar("long long", "cols"),
            Scalar("long long", "rows"),
            Scalar("long long", "nnz"),
            Scalar("long long", "total"),
            Tensor("err", read_only=False, type_name="int"),
        ),
        body=grid_stride(
            Var("total"),
            *(
                Let("const long long", "s", Bin("/", Var("i"), Var("cols"))),
                Let("const long long", "n", Bin("%", Var("i"), Var("cols"))),
                *_event_segment(),
                Let("const float", "g", Load("dout", Var("i"))),
                For(
                    "e",
                    Var("lo"),
                    Var("hi"),
                    Num("1"),
                    (
                        *_event_row_check(),
                        AtomicAdd(
                            "dweights",
                            dst,
                            Bin("*", Load("values", Var("e")), Var("g")),
                        ),
                    ),
                    incr="++",
                ),
            ),
        ),
    )


def sparse_linear_values_grad() -> Kernel:
    """``sparse_linear`` 对事件幅度的梯度：``dvalues[e] = dout[s] · W[r]``。

    与权重梯度"按输出散射、重复下标要原子加"不同，这里**按步并行**：第
    ``i`` 个线程独占第 ``i`` 步的全部事件，逐个算一个长度 ``cols`` 的点积，
    直接写 ``dvalues[e]``。每个事件只被一个线程写一次，因此不需要原子加。

    并行度取步数而不是事件数，是为了避免"事件 → 步"的映射：那需要在设备上
    对 ``offsets`` 做二分查找（L2 没有 while，只能展开成定长 for），而这里
    ``offsets`` 只需顺序读一遍。代价是同一 batch 内步数少时并行度偏低、且
    空步的线程直接空转——两点都比多一层二分简单。
    """
    dout_at = Bin("+", Bin("*", Var("i"), Var("cols")), Var("n"))
    weight_at = _ROW_COLS.offset(("r", "n"), cast=None)
    return Kernel(
        name="sparse_linear_values_grad",
        params=(
            Tensor("dout"),
            Tensor("weights"),
            Tensor("indices", type_name="int"),
            Tensor("offsets", type_name="int"),
            Tensor("dvalues", read_only=False),
            Scalar("long long", "cols"),
            Scalar("long long", "rows"),
            Scalar("long long", "nnz"),
            Scalar("long long", "steps"),
            Tensor("err", read_only=False, type_name="int"),
        ),
        body=grid_stride(
            Var("steps"),
            *(
                Let("const long long", "s", Var("i")),
                *_event_segment(),
                For(
                    "e",
                    Var("lo"),
                    Var("hi"),
                    Num("1"),
                    (
                        *_event_row_check(),
                        Let("float", "acc", Num("0.f")),
                        For(
                            "n",
                            Num("0"),
                            Var("cols"),
                            Num("1"),
                            (
                                Assign(
                                    Var("acc"),
                                    Bin(
                                        "+",
                                        Var("acc"),
                                        Bin(
                                            "*",
                                            Load("dout", dout_at),
                                            Load("weights", weight_at),
                                        ),
                                    ),
                                ),
                            ),
                            incr="++",
                        ),
                        Store("dvalues", Var("e"), Var("acc")),
                    ),
                    incr="++",
                ),
            ),
        ),
    )


def spike_linear() -> Kernel:
    """事件驱动的脉冲线性：``out[s, n] = Σ_e spikes[s, indices[e]] * W[indices[e], n]``。

    与 ``sparse_linear`` 的唯一差别是事件幅度直接读 ``spikes[s, r]``，而不是
    读单独的 ``values`` 缓冲。这一步省掉的正是"把脉冲值再拷一份到 capacity
    长度"的缓冲，也让反向可以把脉冲当成普通稠密矩阵来求导。

    ``e`` 仍只扫第 ``s`` 步真正发放的区间：稠密的 ``(S, K)`` 乘积从不物化。
    """
    weight_at = _ROW_COLS.offset(("r", "n"), cast=None)
    spike_at = _S_ROWS.offset(("s", "r"), cast=None)
    return Kernel(
        name="spike_linear",
        params=(
            Tensor("spikes"),
            Tensor("weights"),
            Tensor("indices", type_name="int"),
            Tensor("offsets", type_name="int"),
            Tensor("out", read_only=False),
            Scalar("long long", "cols"),
            Scalar("long long", "rows"),
            Scalar("long long", "nnz"),
            Scalar("long long", "total"),
            Tensor("err", read_only=False, type_name="int"),
        ),
        body=grid_stride(
            Var("total"),
            *(
                Let("const long long", "s", Bin("/", Var("i"), Var("cols"))),
                Let("const long long", "n", Bin("%", Var("i"), Var("cols"))),
                *_event_segment(),
                Let("float", "acc", Num("0.f")),
                For(
                    "e",
                    Var("lo"),
                    Var("hi"),
                    Num("1"),
                    (
                        *_event_row_check(),
                        Assign(
                            Var("acc"),
                            Bin(
                                "+",
                                Var("acc"),
                                Bin(
                                    "*",
                                    Load("spikes", spike_at),
                                    Load("weights", weight_at),
                                ),
                            ),
                        ),
                    ),
                    incr="++",
                ),
                Store("out", Var("i"), Var("acc")),
            ),
        ),
    )


def spike_linear_weight_grad() -> Kernel:
    """``spike_linear`` 对权重的梯度：把 ``spikes[s, r] * dout[s]`` 散射回 ``r`` 行。

    值由脉冲矩阵现场读取，因此不需要 values 缓冲；同一行被多个事件命中时
    必须原子累加，输出缓冲由后端先清零。
    """
    dst = _ROW_COLS.offset(("r", "n"), cast=None)
    spike_at = _S_ROWS.offset(("s", "r"), cast=None)
    return Kernel(
        name="spike_linear_weight_grad",
        params=(
            Tensor("dout"),
            Tensor("spikes"),
            Tensor("indices", type_name="int"),
            Tensor("offsets", type_name="int"),
            Tensor("dweights", read_only=False),
            Scalar("long long", "cols"),
            Scalar("long long", "rows"),
            Scalar("long long", "nnz"),
            Scalar("long long", "total"),
            Tensor("err", read_only=False, type_name="int"),
        ),
        body=grid_stride(
            Var("total"),
            *(
                Let("const long long", "s", Bin("/", Var("i"), Var("cols"))),
                Let("const long long", "n", Bin("%", Var("i"), Var("cols"))),
                *_event_segment(),
                Let("const float", "g", Load("dout", Var("i"))),
                For(
                    "e",
                    Var("lo"),
                    Var("hi"),
                    Num("1"),
                    (
                        *_event_row_check(),
                        AtomicAdd(
                            "dweights",
                            dst,
                            Bin("*", Load("spikes", spike_at), Var("g")),
                        ),
                    ),
                    incr="++",
                ),
            ),
        ),
    )


# 布局常量：生产内核里反复出现的两维行主序线性化，统一由 ``Layout`` 生成。
# ``cast=None`` 关掉首因子到 64 位的提升，生成的表达式与原先手写的
# ``a * stride + b`` 逐 token 相同（5.209 的判据是对渲染源码取哈希比对）。
_ROW_COLS = Layout(("r", "n"), (None, "cols"))


_S_ROWS = Layout(("s", "r"), (None, "rows"))


_T_WIDTH = Layout(("t", "k"), (None, "width"))
