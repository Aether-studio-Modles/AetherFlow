"""IR 的差分模糊测试。

conformance 用固定图集回答"后端有没有承接这套 IR"；模糊测试回答的是另一个
问题：**把图随机拼起来，IR 自己会不会坏？** 固定用例只能覆盖作者想到的组合，
而算子之间的边界（reshape 的元素数守恒、concat 的非拼接维、transpose 的置换、
slice 的区间）恰恰是手写用例最容易漏的地方。

每个种子做五件事，任何一步失败都记成一条可复现的缺陷：

1. **构图**：随机挑算子、随机挑形状，图必须能构造出来；
2. **校验**：``verify`` 不能报 error；
3. **序列化**：``loads(dumps(g))`` 读回来的图与原图逐字段相等；
4. **执行**：参考执行成功，同一输入跑两次逐位相同；
5. **优化**：默认管线（Simplify / 常量折叠 / CSE / DCE）之后的图仍能校验、
   能序列化，且输出与优化前**逐位一致**。

另外每轮还会做一次**变异**：复制随机节点的属性，注入一条非法属性或一个越界
取值，要求构图期明确抛 ``ValueError`` / ``TypeError`` / ``KeyError``。被接受
的非法输入是更严重的问题——它意味着 verifier 放行了一张语义未定义的图。

还有一组**运行期语义错误**是构图期永远看不出来的：下标与分母都是数据，只有
喂进执行器才知道。它们不进随机构图，而是固定构造、每轮跑一次（见
``_runtime_cases``）：``embedding`` 的正向 / 负向越界下标、事件类算子的越界
``indices`` 与溢出 ``offsets``、整数 ``div`` 的分母为 0 都必须**显式抛
``ValueError``**；浮点 ``div`` 的除零则必须**不**抛错并给出 IEEE 的 inf——
把"除零"的两种语义分开钉住，任何一边被静默改写都算失败。

传入 ``backends=("cuda", "hip")`` 时，同一张图还会再交给硬件后端执行一次，与
参考语义按 ``rtol=atol=1e-4`` 对拍（整型输出逐位相等）。后端按契约拒绝
（``BackendCapabilityError``）记为"拒绝"而不是失败——随机图本来就会拼出后端
没承接的形态，这正是它该说"不"的地方；除此之外的任何异常或数值偏差都算失败。
默认不接后端，所以没有 GPU 的机器上行为与以前完全一致。

**覆盖范围是诚实标注的**：随机生成用"形状与数值"这批算子（逐元素、广播二元、
matmul、softmax、reduce_sum、reshape、transpose、cast、slice、broadcast_to、
pad、concat、step/spike、dropout_mask、constant）打乱图的主体，再用一块
固定形状的素材（``img`` / ``tri`` / ``tab`` / ``ids``）接上几何类算子：
conv2d（含分组 / dilation）、maxpool2d、avgpool2d、attention（含 mask 与
dropout）、scan_linear、embedding。素材的数值由**独立种子**生成，不额外
消耗主 rng 的随机序列；但模板池本身变大了，所以种子到图的映射与 5.38 时
不同——种子承诺的是"同一版本内可复现"，不是跨版本稳定。

仍然不进随机生成器的是依赖符号维的组合（conv/attention 的符号维已有专项
用例），随机图只用静态形状。

用法::

    python -m aetherflow.fuzz --iterations 200
    python -m aetherflow.fuzz --start 37 --iterations 1   # 复现某个种子
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .ir.graph import Graph, Node, Value
from .ir.serialize import dumps, loads, to_dict
from .ir.types import DType, f32, i32
from .ir.verifier import ERROR, verify
from .passes import default_pipeline
from .runtime.reference import ReferenceExecutor

__all__ = [
    "FuzzCase",
    "FuzzFailure",
    "FuzzReport",
    "random_case",
    "run_fuzz",
]

_REJECTED = (ValueError, TypeError, KeyError)


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FuzzCase:
    """一个种子生成出的图与输入。"""

    seed: int
    graph: Graph
    feed: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class FuzzFailure:
    seed: int
    stage: str
    detail: str

    def __str__(self) -> str:
        return f"seed={self.seed} [{self.stage}] {self.detail}"


@dataclass
class FuzzReport:
    checked: int
    mutated: int
    skipped: int
    failures: List[FuzzFailure]
    ops: FrozenSet[str]
    # 额外接入的硬件后端：名字、成功对拍的例数、按契约拒绝的例数
    backends: Tuple[str, ...] = ()
    compared: Dict[str, int] = field(default_factory=dict)
    refused: Dict[str, int] = field(default_factory=dict)
    # 语义非法节点的尝试数与其中被明确拒绝的例数
    semantic: int = 0
    semantic_skipped: int = 0
    # 运行期语义错误：给合法图喂非法数据，要求显式报错的例数
    runtime: int = 0

    @property
    def ok(self) -> bool:
        return not self.failures

    def text(self) -> str:
        head = (
            f"fuzz: {self.checked} 例，{self.mutated} 次非法输入被拒绝，"
            f"{self.skipped} 次无可用变异；覆盖 {len(self.ops)} 个算子"
        )
        for name in self.backends:
            refused = self.refused.get(name, 0)
            extra = f"，拒绝 {refused} 例" if refused else ""
            head += f"；{name} 对拍 {self.compared.get(name, 0)} 例{extra}"
        if self.semantic:
            head += f"；语义非法节点拒绝 {self.semantic} 例"
        if self.runtime:
            head += f"；运行期语义错误拒绝 {self.runtime} 例"
        if self.ok:
            return head + "；全部通过"
        lines = [head + f"；{len(self.failures)} 条失败"]
        lines.extend(f"  {item}" for item in self.failures)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# 随机构图
# --------------------------------------------------------------------------

_UNARY = ("relu", "tanh", "sigmoid", "exp", "neg", "gelu", "silu")
_BINARY = ("add", "sub", "mul", "div")
_MAX_ELEMENTS = 1024


def _values(rng: np.random.Generator, shape: Tuple[int, ...]) -> np.ndarray:
    """输入取正值：让 log / sqrt 也能落进随机图而不必额外插保护节点。"""
    return (np.abs(rng.standard_normal(shape)) + 0.5).astype(np.float32)


def _random_shape(rng: np.random.Generator) -> Tuple[int, ...]:
    rank = int(rng.integers(1, 4))
    dims = [int(rng.integers(1, 5)) for _ in range(rank)]
    while int(np.prod(dims)) > 64:
        dims[int(rng.integers(0, rank))] = 1
    return tuple(dims)


def _pick(pool: Sequence[Value], rng: np.random.Generator) -> Optional[Value]:
    if not pool:
        return None
    return pool[int(rng.integers(0, len(pool)))]


def _fits(value: Value) -> bool:
    return value.type.numel() <= _MAX_ELEMENTS


def _add_unary(g: Graph, pool: List[Value], rng: np.random.Generator) -> Optional[Value]:
    x = _pick(pool, rng)
    if x is None or not _fits(x):
        return None
    name = _UNARY[int(rng.integers(0, len(_UNARY)))]
    return g.add_node(name, [x])


def _add_binary(g: Graph, pool: List[Value], rng: np.random.Generator) -> Optional[Value]:
    a = _pick(pool, rng)
    if a is None or not _fits(a):
        return None
    # 同形状一定可广播；三成概率换成全 1 的常量，专门走广播路径
    if rng.random() < 0.3 and a.type.rank >= 1:
        b = g.constant(1.0, dtype=a.type.dtype, shape=(1,) * a.type.rank)
    else:
        same = [v for v in pool if v.type.dtype == a.type.dtype and v.type.shape == a.type.shape]
        b = same[int(rng.integers(0, len(same)))]
    name = _BINARY[int(rng.integers(0, len(_BINARY)))]
    if name == "div" and a.type.dtype in (DType.I8, DType.I32):
        # 整数除法不做隐式舍入，池子里的整数可能是 0；随机抽中它当除数会让
        # "合法图 + 合法输入"直接除零，掩盖真正想找的语义问题。
        b = g.constant(2, dtype=a.type.dtype, shape=a.type.shape)
    return g.add_node(name, [a, b])


def _add_matmul(g: Graph, pool: List[Value], rng: np.random.Generator) -> Optional[Value]:
    pairs = [
        (a, b)
        for a in pool
        for b in pool
        if a.type.rank >= 2
        and b.type.rank >= 2
        and a.type.dtype == b.type.dtype
        and a.type.shape[-1] == b.type.shape[-2]
        and a.type.numel() <= _MAX_ELEMENTS
        and b.type.numel() <= _MAX_ELEMENTS
    ]
    if not pairs:
        return None
    a, b = pairs[int(rng.integers(0, len(pairs)))]
    return g.add_node("matmul", [a, b])


def _add_softmax(g: Graph, pool: List[Value], rng: np.random.Generator) -> Optional[Value]:
    x = _pick(pool, rng)
    if x is None or x.type.rank < 1:
        return None
    axis = -1 if rng.random() < 0.5 else int(rng.integers(0, x.type.rank))
    return g.add_node("softmax", [x], {"axis": axis})


def _add_reduce(g: Graph, pool: List[Value], rng: np.random.Generator) -> Optional[Value]:
    x = _pick(pool, rng)
    if x is None or x.type.rank < 1:
        return None
    rank = x.type.rank
    if rank >= 2 and rng.random() < 0.3:
        axes = tuple(sorted(int(i) for i in rng.choice(rank, size=int(rng.integers(1, rank)), replace=False)))
    else:
        axes = int(rng.integers(0, rank))
    keepdims = bool(rng.integers(0, 2))
    return g.add_node("reduce_sum", [x], {"axis": axes, "keepdims": keepdims})


def _add_reshape(g: Graph, pool: List[Value], rng: np.random.Generator) -> Optional[Value]:
    x = _pick(pool, rng)
    if x is None:
        return None
    n = x.type.numel()
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    d = divisors[int(rng.integers(0, len(divisors)))]
    target = (1, d, n // d) if rng.random() < 0.3 else (d, n // d)
    if target == tuple(x.type.shape):
        return None
    return g.add_node("reshape", [x], {"shape": target})


def _add_transpose(g: Graph, pool: List[Value], rng: np.random.Generator) -> Optional[Value]:
    x = _pick(pool, rng)
    if x is None or x.type.rank < 2:
        return None
    perm = tuple(int(i) for i in rng.permutation(x.type.rank))
    return g.add_node("transpose", [x], {"perm": perm})


def _add_cast(g: Graph, pool: List[Value], rng: np.random.Generator) -> Optional[Value]:
    x = _pick(pool, rng)
    if x is None:
        return None
    # 只在这两种类型之间来回切：它们不改变形状，也不引入精度比较问题
    dtype = "f32" if x.type.dtype.value == "i32" else "i32"
    return g.add_node("cast", [x], {"dtype": dtype})


def _add_slice(g: Graph, pool: List[Value], rng: np.random.Generator) -> Optional[Value]:
    x = _pick(pool, rng)
    if x is None or x.type.rank < 1:
        return None
    dim = int(rng.integers(0, x.type.rank))
    size = x.type.shape[dim]
    start = int(rng.integers(0, size))
    stop = int(rng.integers(start + 1, size + 1))
    return g.add_node("slice", [x], {"dim": dim, "start": start, "stop": stop})


def _add_broadcast(g: Graph, pool: List[Value], rng: np.random.Generator) -> Optional[Value]:
    x = _pick(pool, rng)
    if x is None:
        return None
    extra = tuple(int(rng.integers(1, 3)) for _ in range(int(rng.integers(1, 3))))
    target = extra + tuple(x.type.shape)
    if int(np.prod(target)) > _MAX_ELEMENTS:
        return None
    return g.add_node("broadcast_to", [x], {"shape": target})


def _add_pad(g: Graph, pool: List[Value], rng: np.random.Generator) -> Optional[Value]:
    x = _pick(pool, rng)
    if x is None or x.type.rank < 1:
        return None
    dim = int(rng.integers(0, x.type.rank))
    attrs = {
        "dim": dim,
        "before": int(rng.integers(0, 3)),
        "after": int(rng.integers(0, 3)),
    }
    return g.add_node("pad", [x], attrs)


def _add_concat(g: Graph, pool: List[Value], rng: np.random.Generator) -> Optional[Value]:
    same = [
        (a, b)
        for a in pool
        for b in pool
        if a is not b and a.type.dtype == b.type.dtype and a.type.shape == b.type.shape
    ]
    if not same:
        return None
    a, b = same[int(rng.integers(0, len(same)))]
    if a.type.rank < 1:
        return None
    axis = int(rng.integers(0, a.type.rank))
    return g.add_node("concat", [a, b], {"axis": axis})


def _add_activation(g: Graph, pool: List[Value], rng: np.random.Generator) -> Optional[Value]:
    x = _pick(pool, rng)
    if x is None:
        return None
    if rng.random() < 0.5:
        return g.add_node("step", [x], {"threshold": float(rng.normal())})
    return g.add_node(
        "spike", [x], {"threshold": float(rng.normal()), "alpha": float(rng.uniform(0.1, 1.0))}
    )


def _add_dropout(g: Graph, pool: List[Value], rng: np.random.Generator) -> Optional[Value]:
    x = _pick(pool, rng)
    if x is None:
        return None
    p = float(rng.choice([0.0, 0.2, 0.5, 0.75]))
    return g.add_node("dropout_mask", [x], {"p": p, "seed": int(rng.integers(0, 1 << 30))})


def _add_constant(g: Graph, pool: List[Value], rng: np.random.Generator) -> Optional[Value]:
    shape = _random_shape(rng)
    return g.constant(float(rng.normal()), shape=shape)


# --- 几何类算子 -----------------------------------------------------------
#
# 它们的属性之间必须自洽（卷积核与 stride/padding 决定输出尺寸，attention 的
# q/k/v 内维必须对齐……）。随机组合很容易落进"非法几何"，那样失败原因就混在
# 一起了。所以这里先按素材形状算出合法区间，再在区间里随机——变的是参数，
# 不是合法性。


def _const(g: Graph, rng: np.random.Generator, shape: Tuple[int, ...], scale: float = 1.0) -> Value:
    """随机常量张量；data 用嵌套 list，序列化往返后仍逐字段相等。"""
    data = (rng.standard_normal(shape) * scale).astype(np.float32).tolist()
    return g.constant(dtype=f32, shape=shape, data=data)


def _aux(g: Graph, name: str) -> Optional[Value]:
    try:
        return g.find(name)
    except KeyError:
        return None


def _add_conv(g: Graph, pool: List[Value], rng: np.random.Generator) -> Optional[Value]:
    x = _aux(g, "img")
    if x is None:
        return None
    _, c, h, w = x.type.shape
    groups = c if rng.random() < 0.25 else 1  # 25% 走 depthwise
    o = c if groups > 1 else int(rng.integers(1, 4))
    kh, kw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    sh, sw = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    ph, pw = int(rng.integers(0, 2)), int(rng.integers(0, 2))
    dh, dw = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    ho = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wo = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    if ho <= 0 or wo <= 0:
        return None
    args = [x, _const(g, rng, (o, c // groups, kh, kw))]
    if rng.random() < 0.5:
        args.append(_const(g, rng, (o,)))
    attrs = {
        "stride": (sh, sw),
        "padding": (ph, pw),
        "dilation": (dh, dw),
        "groups": groups,
    }
    return g.add_node("conv2d", args, attrs)


def _add_pool(g: Graph, pool: List[Value], rng: np.random.Generator) -> Optional[Value]:
    x = _aux(g, "img")
    if x is None:
        return None
    _, _, h, w = x.type.shape
    kh, kw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    sh, sw = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    ph, pw = int(rng.integers(0, 2)), int(rng.integers(0, 2))
    if (h + 2 * ph - kh) // sh + 1 <= 0 or (w + 2 * pw - kw) // sw + 1 <= 0:
        return None
    name = "maxpool2d" if rng.random() < 0.5 else "avgpool2d"
    attrs = {"kernel": (kh, kw), "stride": (sh, sw), "padding": (ph, pw)}
    return g.add_node(name, [x], attrs)


def _add_attention(g: Graph, pool: List[Value], rng: np.random.Generator) -> Optional[Value]:
    q = _aux(g, "tri")
    if q is None:
        return None
    batch, steps, dim = q.type.shape
    t = int(rng.integers(1, 4))
    e = int(rng.integers(1, 4))
    args = [q, _const(g, rng, (batch, t, dim)), _const(g, rng, (batch, t, e))]
    attrs = {"scale": float(rng.choice([0.5, 1.0]))}
    if rng.random() < 0.3:
        attrs["dropout"] = 0.25
        attrs["seed"] = int(rng.integers(0, 1 << 30))
    if rng.random() < 0.3:
        # 加性 mask：用有限负值，整行全 -inf 会让 softmax 变成 nan
        args.append(_const(g, rng, (batch, steps, t), scale=-1.0))
    return g.add_node("attention", args, attrs)


def _add_scan(g: Graph, pool: List[Value], rng: np.random.Generator) -> Optional[Value]:
    same = [
        (a, b)
        for a in pool
        for b in pool
        if a is not b
        and a.type.dtype == b.type.dtype
        and a.type.shape == b.type.shape
        and a.type.rank >= 2
        and a.type.numel() <= _MAX_ELEMENTS
    ]
    if not same:
        return None
    a, b = same[int(rng.integers(0, len(same)))]
    h0 = _const(g, rng, tuple(a.type.shape[1:]))
    return g.add_node("scan_linear", [a, b, h0])


def _add_embedding(g: Graph, pool: List[Value], rng: np.random.Generator) -> Optional[Value]:
    table, ids = _aux(g, "tab"), _aux(g, "ids")
    if table is None or ids is None:
        return None
    if rng.random() < 0.5:
        # 切片后的下标仍是 i32，顺带覆盖 embedding 接在 slice 之后
        ids = g.add_node("slice", [ids], {"dim": 0, "start": 0, "stop": 2})
    return g.add_node("embedding", [table, ids])


_TEMPLATES: Tuple[Callable[[Graph, List[Value], np.random.Generator], Optional[Value]], ...] = (
    _add_unary,
    _add_binary,
    _add_matmul,
    _add_softmax,
    _add_reduce,
    _add_reshape,
    _add_transpose,
    _add_cast,
    _add_slice,
    _add_broadcast,
    _add_pad,
    _add_concat,
    _add_activation,
    _add_dropout,
    _add_constant,
    _add_conv,
    _add_pool,
    _add_attention,
    _add_scan,
    _add_embedding,
)


def random_case(seed: int) -> FuzzCase:
    """按种子生成一张随机图。同一 seed 在任意机器上得到同一张图。"""
    rng = np.random.default_rng(seed)
    graph = Graph(f"fuzz_{seed}")
    pool: List[Value] = []
    feed: Dict[str, np.ndarray] = {}

    # 两个 4x4 输入保证 matmul 一定有素材；其余输入形状随机
    for name in ("a", "b"):
        shape = (4, 4)
        value = graph.input(name, f32[shape])
        pool.append(value)
        feed[name] = _values(rng, shape)
    for i in range(int(rng.integers(0, 2))):
        shape = _random_shape(rng)
        name = f"x{i}"
        value = graph.input(name, f32[shape])
        pool.append(value)
        feed[name] = _values(rng, shape)

    # 几何类算子的固定素材：形状固定，数值由独立种子生成，主 rng 消耗不变。
    # 它们不进 pool——常规模板不会把图像/查表素材当成任意中间值来用。
    aux = np.random.default_rng(seed ^ 0xA5A5)
    for name, shape in (("img", (1, 2, 6, 6)), ("tri", (2, 3, 4)), ("tab", (8, 4))):
        graph.input(name, f32[shape])
        feed[name] = _values(aux, shape)
    graph.input("ids", i32[(3, 2)])
    feed["ids"] = aux.integers(0, 8, size=(3, 2), dtype=np.int32)

    for _ in range(int(rng.integers(3, 11))):
        template = _TEMPLATES[int(rng.integers(0, len(_TEMPLATES)))]
        out = template(graph, pool, rng)
        if out is not None:
            pool.append(out)

    count = int(rng.integers(1, 3))
    indices = sorted(int(i) for i in rng.choice(len(pool), size=count, replace=False))
    for i, index in enumerate(indices):
        graph.output(pool[index], f"out{i}")
    return FuzzCase(seed=seed, graph=graph, feed=feed)


# --------------------------------------------------------------------------
# 不变量检查
# --------------------------------------------------------------------------


def _same(left, right) -> bool:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return bool(np.array_equal(left, right, equal_nan=True))


def _check_case(
    case: FuzzCase, executor: ReferenceExecutor
) -> Tuple[Optional[Tuple[str, str]], Optional[Dict[str, np.ndarray]]]:
    """返回 ``(阶段, 说明)`` 与参考输出；前者为 ``None`` 表示全部通过。"""
    graph = case.graph
    baseline: Optional[Dict[str, np.ndarray]] = None

    errors = [item for item in verify(graph) if item.level == ERROR]
    if errors:
        return ("verify", "; ".join(str(item) for item in errors)), None

    try:
        if to_dict(graph) != to_dict(loads(dumps(graph))):
            return ("serialize", "JSON 往返后的图与原图不一致"), None
    except Exception as exc:  # noqa: BLE001 - 任何异常都是序列化的缺陷
        return ("serialize", f"{type(exc).__name__}: {exc}"), None

    try:
        with np.errstate(all="ignore"):
            first = executor.run(graph, case.feed)
            second = executor.run(graph, case.feed)
    except Exception as exc:  # noqa: BLE001
        return ("execute", f"{type(exc).__name__}: {exc}"), None
    baseline = first
    for name in first:
        if not _same(first[name], second[name]):
            return ("determinism", f"输出 {name} 两次执行不一致"), baseline

    try:
        optimized = default_pipeline().run(graph)
        with np.errstate(all="ignore"):
            result = executor.run(optimized, case.feed)
    except Exception as exc:  # noqa: BLE001
        return ("optimize", f"{type(exc).__name__}: {exc}"), baseline
    if set(result) != set(first):
        return ("optimize", f"输出名集合发生变化：{sorted(first)} -> {sorted(result)}"), baseline
    for name in first:
        if not _same(first[name], result[name]):
            return ("optimize", f"优化改变了输出 {name} 的数值"), baseline

    errors = [item for item in verify(optimized) if item.level == ERROR]
    if errors:
        return ("optimize", "; ".join(str(item) for item in errors)), baseline
    try:
        if to_dict(optimized) != to_dict(loads(dumps(optimized))):
            return ("optimize", "优化后的图 JSON 往返不一致"), baseline
    except Exception as exc:  # noqa: BLE001
        return ("optimize", f"{type(exc).__name__}: {exc}"), baseline
    return None, baseline


# --------------------------------------------------------------------------
# 硬件后端的数值对拍
# --------------------------------------------------------------------------

_RTOL = 1e-4
_ATOL = 1e-4


def _compare_backend(backend, case: FuzzCase, expected: Mapping[str, np.ndarray]):
    """返回 ``("ok" | "refused" | "failed", 说明)``。

    后端按契约拒绝（``BackendCapabilityError``）不算失败：随机图会拼出它没
    承接的形态，这正是"不支持要显式说"的地方。其它异常与数值偏差才是缺陷。
    """
    from .backends import BackendCapabilityError

    try:
        got = backend.execute(case.graph, case.feed)
    except BackendCapabilityError:
        return "refused", ""
    except Exception as exc:  # noqa: BLE001
        return "failed", f"{type(exc).__name__}: {exc}"

    if set(got) != set(expected):
        return "failed", f"输出名不一致：{sorted(got)} vs {sorted(expected)}"
    for name, exp in expected.items():
        arr = np.asarray(got[name])
        ref = np.asarray(exp)
        if arr.shape != ref.shape or arr.dtype != ref.dtype:
            return "failed", (
                f"输出 {name} 的形状 / dtype 不一致："
                f"{arr.shape}/{arr.dtype} vs {ref.shape}/{ref.dtype}"
            )
        if arr.dtype.kind in "fc":
            if not np.allclose(arr, ref, rtol=_RTOL, atol=_ATOL, equal_nan=True):
                diff = np.abs(arr.astype(np.float64) - ref.astype(np.float64))
                peak = float(np.nanmax(diff)) if diff.size else 0.0
                return "failed", (
                    f"输出 {name} 与参考语义不一致（最大绝对误差 {peak:.3e}，"
                    f"允许 rtol={_RTOL:g} atol={_ATOL:g}）"
                )
        elif not np.array_equal(arr, ref):
            return "failed", f"输出 {name} 与参考语义不一致（整型要求逐位相等）"
    return "ok", ""


# --------------------------------------------------------------------------
# 变异：非法输入必须被明确拒绝
# --------------------------------------------------------------------------


def _mut_unknown(attrs: dict, node: Node, rng: np.random.Generator) -> Optional[dict]:
    attrs["__fuzz_unknown__"] = 1
    return attrs


def _mut_reshape(attrs: dict, node: Node, rng: np.random.Generator) -> Optional[dict]:
    attrs["shape"] = (node.output.type.numel() + 1,)
    return attrs


def _mut_transpose(attrs: dict, node: Node, rng: np.random.Generator) -> Optional[dict]:
    rank = node.inputs[0].type.rank
    if rank < 2:
        return None
    attrs["perm"] = (0,) * rank
    return attrs


def _mut_slice(attrs: dict, node: Node, rng: np.random.Generator) -> Optional[dict]:
    attrs["start"] = 0
    attrs["stop"] = 0
    return attrs


def _mut_pad(attrs: dict, node: Node, rng: np.random.Generator) -> Optional[dict]:
    attrs["before"] = -1
    return attrs


def _mut_broadcast(attrs: dict, node: Node, rng: np.random.Generator) -> Optional[dict]:
    shape = node.inputs[0].type.shape
    if all(dim == 1 for dim in shape):
        return None
    attrs["shape"] = tuple(dim + 1 for dim in shape)
    return attrs


def _mut_reduce(attrs: dict, node: Node, rng: np.random.Generator) -> Optional[dict]:
    attrs["axis"] = node.inputs[0].type.rank + 5
    return attrs


def _mut_cast(attrs: dict, node: Node, rng: np.random.Generator) -> Optional[dict]:
    attrs["dtype"] = "q7"
    return attrs


def _mut_dropout(attrs: dict, node: Node, rng: np.random.Generator) -> Optional[dict]:
    attrs["p"] = 1.0
    return attrs


def _mut_concat(attrs: dict, node: Node, rng: np.random.Generator) -> Optional[dict]:
    attrs["axis"] = node.inputs[0].type.rank + 3
    return attrs


def _mut_constant(attrs: dict, node: Node, rng: np.random.Generator) -> Optional[dict]:
    attrs["dtype"] = "q7"
    return attrs


def _mut_conv(attrs: dict, node: Node, rng: np.random.Generator) -> Optional[dict]:
    # groups 不能被输入通道整除，必须在构图期拒绝
    attrs["groups"] = int(node.inputs[0].type.shape[1]) + 1
    return attrs


def _mut_pool(attrs: dict, node: Node, rng: np.random.Generator) -> Optional[dict]:
    attrs["kernel"] = (0, 1)
    return attrs


def _mut_attention(attrs: dict, node: Node, rng: np.random.Generator) -> Optional[dict]:
    attrs["dropout"] = 1.0
    return attrs


_MUTATORS: Dict[str, Callable[[dict, Node, np.random.Generator], Optional[dict]]] = {
    "reshape": _mut_reshape,
    "transpose": _mut_transpose,
    "slice": _mut_slice,
    "pad": _mut_pad,
    "broadcast_to": _mut_broadcast,
    "reduce_sum": _mut_reduce,
    "cast": _mut_cast,
    "dropout_mask": _mut_dropout,
    "concat": _mut_concat,
    "constant": _mut_constant,
    "conv2d": _mut_conv,
    "maxpool2d": _mut_pool,
    "avgpool2d": _mut_pool,
    "attention": _mut_attention,
}


def _mutate_rejected(case: FuzzCase) -> Optional[bool]:
    """返回 True=被拒绝，False=被接受（缺陷），None=没有可用的变异。"""
    clone = loads(dumps(case.graph))
    nodes = list(clone.nodes)
    if not nodes:
        return None
    rng = np.random.default_rng(case.seed ^ 0x5EED)
    node = nodes[int(rng.integers(0, len(nodes)))]
    attrs = dict(node.attrs)
    if rng.random() < 0.5 or node.op.name not in _MUTATORS:
        attrs = _mut_unknown(attrs, node, rng)
    else:
        attrs = _MUTATORS[node.op.name](attrs, node, rng)
    if attrs is None:
        return None
    try:
        clone.add_node(node.op.name, list(node.inputs), attrs)
    except _REJECTED:
        return True
    return False


def _semantic_rejected(case: FuzzCase) -> Tuple[int, List[str], int]:
    """用合法属性名 + 非法值构造节点，要求构图期明确拒绝。

    与 ``_mutate_rejected`` 互补：那里问"未知属性 / 越界取值会不会被放行"，
    这里问"已知属性配上语义非法的值会不会被放行"。返回
    ``(尝试数, 失败说明, 跳过数)``。
    """
    attempts = 0
    failures: List[str] = []
    skipped = 0

    def expect_reject(label: str, build: Callable[[Graph, List[Value]], Any]) -> None:
        nonlocal attempts
        attempts += 1
        clone = loads(dumps(case.graph))
        # 必须用克隆图自己的 Value：把原图的 Value 塞进克隆图，add_node 会先
        # 报"引用了不属于本图的 Value"，那样每次拒绝都来自跨图引用，语义
        # 根本没被检查到（这里踩过一次）。
        local = list(clone.values)
        try:
            build(clone, local)
        except _REJECTED:
            return
        except Exception as exc:  # noqa: BLE001 - 意外异常同样是缺陷
            failures.append(f"{label} 抛出 {type(exc).__name__}: {exc}")
            return
        failures.append(f"{label} 被接受")

    values = list(case.graph.values)
    if not values:
        return 0, [], 3
    dims = list(values[0].type.shape)
    if all(isinstance(d, int) for d in dims):
        count = int(np.prod(dims)) if dims else 1
        expect_reject(
            "reshape 元素数不等",
            lambda g, vs: g.add_node("reshape", [vs[0]], {"shape": (count + 1,)}),
        )
    else:
        skipped += 1

    index_pairs = [
        (i, j)
        for i, left in enumerate(values)
        for j, right in enumerate(values)
        if len(left.type.shape) == 2
        and len(right.type.shape) == 2
        and left.type.shape[1] != right.type.shape[0]
    ]
    if index_pairs:
        i, j = index_pairs[0]
        expect_reject(
            "matmul 内维不匹配",
            lambda g, vs: g.add_node("matmul", [vs[i], vs[j]], {}),
        )
    else:
        skipped += 1

    expect_reject(
        "cast 到未知 dtype",
        lambda g, vs: g.add_node("cast", [vs[0]], {"dtype": "f8"}),
    )
    return attempts, failures, skipped


# --------------------------------------------------------------------------
# 运行期语义错误：构图期看不出来，喂进执行器才知道
# --------------------------------------------------------------------------


def _runtime_cases() -> Tuple[Tuple[str, Callable[[], Any], bool], ...]:
    """固定构造的运行期语义错误用例：``(名字, 执行函数, 是否应当抛错)``。

    下标与分母都是**数据**，verifier 看不到它们，只能在执行器里判。``ValueError``
    是这些错误的统一出口。第三项为 ``False`` 的那条是浮点除零——它有定义过的
    IEEE 语义（±inf / nan），必须不抛错；把它改成拒绝执行同样是退化。

    这些用例不进随机构图：随机生成要求"图本身合法"，而这里要的恰恰是"图合法、
    数据非法"。
    """

    def run_embedding(ids):
        def run():
            g = Graph("fuzz_runtime_embedding")
            table = g.input("table", f32[(4, 3)])
            index = g.input("ids", i32[(2,)])
            g.output(g.add_node("embedding", [table, index]), "y")
            feed = {
                "table": np.arange(12, dtype=np.float32).reshape(4, 3),
                "ids": np.asarray(ids, dtype=np.int32),
            }
            return ReferenceExecutor().run(g, feed)["y"]

        return run

    def run_embedding_backward(ids):
        def run():
            g = Graph("fuzz_runtime_embedding_backward")
            dy = g.input("dy", f32[(2, 3)])
            index = g.input("ids", i32[(2,)])
            node = g.add_node(
                "embedding_backward", [dy, index], {"table_shape": (4, 3)}
            )
            g.output(node, "y")
            feed = {
                "dy": np.ones((2, 3), dtype=np.float32),
                "ids": np.asarray(ids, dtype=np.int32),
            }
            return ReferenceExecutor().run(g, feed)["y"]

        return run

    def run_sparse(indices):
        def run():
            g = Graph("fuzz_runtime_sparse_linear")
            w = g.input("w", f32[(4, 2)])
            idx = g.input("indices", i32[(2,)])
            val = g.input("values", f32[(2,)])
            off = g.input("offsets", i32[(2,)])
            g.output(g.add_node("sparse_linear", [w, idx, val, off]), "y")
            feed = {
                "w": np.zeros((4, 2), dtype=np.float32),
                "indices": np.asarray(indices, dtype=np.int32),
                "values": np.ones(2, dtype=np.float32),
                "offsets": np.array([0, 2], dtype=np.int32),
            }
            return ReferenceExecutor().run(g, feed)["y"]

        return run

    def run_spike_linear(indices):
        def run():
            g = Graph("fuzz_runtime_spike_linear")
            sp = g.input("spikes", f32[(1, 4)])
            w = g.input("w", f32[(4, 2)])
            idx = g.input("indices", i32[(2,)])
            off = g.input("offsets", i32[(2,)])
            g.output(g.add_node("spike_linear", [sp, w, idx, off]), "y")
            feed = {
                "spikes": np.ones((1, 4), dtype=np.float32),
                "w": np.zeros((4, 2), dtype=np.float32),
                "indices": np.asarray(indices, dtype=np.int32),
                "offsets": np.array([0, 2], dtype=np.int32),
            }
            return ReferenceExecutor().run(g, feed)["y"]

        return run

    def run_spike_csr():
        g = Graph("fuzz_runtime_spike_csr")
        sp = g.input("spikes", f32[(1, 3)])
        off = g.input("offsets", i32[(2,)])
        g.output(g.add_node("spike_csr", [sp, off], {"capacity": 3}), "y")
        feed = {
            "spikes": np.ones((1, 3), dtype=np.float32),
            "offsets": np.array([0, 5], dtype=np.int32),
        }
        return ReferenceExecutor().run(g, feed)["y"]

    def run_div(dtype, a, b):
        def run():
            g = Graph("fuzz_runtime_div")
            left = g.input("a", dtype[(2,)])
            right = g.input("b", dtype[(2,)])
            g.output(g.add_node("div", [left, right]), "y")
            feed = {
                "a": np.asarray(a, dtype=dtype.np_dtype),
                "b": np.asarray(b, dtype=dtype.np_dtype),
            }
            with np.errstate(all="ignore"):
                return ReferenceExecutor().run(g, feed)["y"]

        return run

    return (
        ("embedding 正越界下标", run_embedding([0, 4]), True),
        ("embedding 负下标", run_embedding([-1, 2]), True),
        ("embedding_backward 负下标", run_embedding_backward([-1, 2]), True),
        ("sparse_linear 越界 indices", run_sparse([0, 4]), True),
        ("spike_linear 越界 indices", run_spike_linear([4, 1]), True),
        ("spike_csr 溢出 offsets", run_spike_csr, True),
        ("div 整数分母为 0", run_div(i32, [4, 6], [2, 0]), True),
        ("div 浮点分母为 0", run_div(f32, [1.0, -1.0], [0.0, 0.0]), False),
    )


def _check_runtime(failures: List[FuzzFailure], seed: int) -> int:
    """跑一遍 ``_runtime_cases``，把不符合预期的记成失败，返回通过的例数。"""
    passed = 0
    for label, run, should_raise in _runtime_cases():
        try:
            out = run()
        except ValueError:
            if should_raise:
                passed += 1
            else:
                failures.append(
                    FuzzFailure(seed, "runtime", f"{label} 不该报错，却抛了 ValueError")
                )
            continue
        except Exception as exc:  # noqa: BLE001 - 意外异常同样是缺陷
            failures.append(
                FuzzFailure(seed, "runtime", f"{label} 抛出 {type(exc).__name__}: {exc}")
            )
            continue
        if should_raise:
            failures.append(
                FuzzFailure(seed, "runtime", f"{label} 被静默接受（预期 ValueError）")
            )
        elif np.any(~np.isfinite(np.asarray(out))):
            passed += 1
        else:
            failures.append(
                FuzzFailure(seed, "runtime", f"{label} 没有给出 IEEE 的 inf / nan")
            )
    return passed


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def run_fuzz(
    iterations: int = 200, start: int = 0, backends: Sequence[str] = ()
) -> FuzzReport:
    """跑 ``iterations`` 个连续种子，返回结构化报告。

    ``backends`` 里列出的后端会与参考语义逐例对拍；名字按
    ``aetherflow.backends.get_backend`` 解析，拿不到就记一条失败（显式要求
    了却悄悄跳过，等于把"没测"报成"测过"）。
    """
    if iterations < 1:
        raise ValueError(f"iterations 至少为 1，得到 {iterations!r}")
    if start < 0:
        raise ValueError(f"start 不能为负，得到 {start!r}")

    from .backends import BackendUnavailableError, get_backend

    executor = ReferenceExecutor()
    failures: List[FuzzFailure] = []
    ops: set = set()
    checked = 0
    mutated = 0
    skipped = 0
    semantic = 0
    semantic_skipped = 0

    names = list(dict.fromkeys(backends))
    resolved = []
    for name in names:
        try:
            resolved.append((name, get_backend(name)))
        except (BackendUnavailableError, LookupError) as exc:
            failures.append(FuzzFailure(start, "backend", f"{name} 不可用：{exc}"))
    compared: Dict[str, int] = {name: 0 for name, _ in resolved}
    refused: Dict[str, int] = {name: 0 for name, _ in resolved}

    # 运行期语义错误与种子无关：固定构造、每轮跑一次。放在种子循环之前，
    # 这样"图合法、数据非法"的缺陷不会被随机图的失败遮住。
    runtime = _check_runtime(failures, start)

    for seed in range(start, start + iterations):
        try:
            case = random_case(seed)
        except Exception as exc:  # noqa: BLE001 - 生成器自己坏了也要记下来
            failures.append(FuzzFailure(seed, "generate", f"{type(exc).__name__}: {exc}"))
            continue
        checked += 1
        ops.update(node.op.name for node in case.graph.nodes)

        problem, baseline = _check_case(case, executor)
        if problem is not None:
            failures.append(FuzzFailure(seed, problem[0], problem[1]))
        elif baseline is not None:
            for name, backend in resolved:
                verdict, detail = _compare_backend(backend, case, baseline)
                if verdict == "refused":
                    refused[name] += 1
                elif verdict == "failed":
                    failures.append(FuzzFailure(seed, "backend", f"{name}: {detail}"))
                else:
                    compared[name] += 1

        try:
            verdict = _mutate_rejected(case)
        except Exception as exc:  # noqa: BLE001
            failures.append(FuzzFailure(seed, "mutate", f"{type(exc).__name__}: {exc}"))
            continue
        if verdict is True:
            mutated += 1
        elif verdict is False:
            failures.append(FuzzFailure(seed, "mutate", "非法属性被接受：构图期没有拒绝"))
        else:
            skipped += 1

        attempts, semantic_failures, semantic_missed = _semantic_rejected(case)
        semantic += attempts - len(semantic_failures)
        semantic_skipped += semantic_missed
        for detail in semantic_failures:
            failures.append(FuzzFailure(seed, "semantic", detail))

    return FuzzReport(
        checked=checked,
        mutated=mutated,
        skipped=skipped,
        failures=failures,
        ops=frozenset(ops),
        backends=tuple(names),
        compared=compared,
        refused=refused,
        semantic=semantic,
        semantic_skipped=semantic_skipped,
        runtime=runtime,
    )


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AetherFlow IR 差分模糊测试")
    parser.add_argument("--iterations", type=int, default=200, help="跑多少个种子（默认 200）")
    parser.add_argument("--start", type=int, default=0, help="起始种子（复现单例时用）")
    parser.add_argument(
        "--backend",
        action="append",
        default=[],
        metavar="NAME",
        help="额外用该后端对拍随机图（可重复，也可逗号分隔：cuda,hip）",
    )
    parser.add_argument("--quiet", action="store_true", help="只打印结论，不打印算子覆盖")
    args = parser.parse_args(argv)

    names = [item.strip() for arg in args.backend for item in arg.split(",") if item.strip()]
    report = run_fuzz(args.iterations, args.start, names)
    print(report.text())
    if not args.quiet:
        print("覆盖算子：" + "、".join(sorted(report.ops)))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
