"""后端一致性（conformance）套件。

这个套件回答一个问题：**一个后端凭什么可以声称自己"承接了这套 IR"？**

判据不是"能跑通几个 demo"，而是五条可执行的检查：

1. 契约完整：``name`` / ``capabilities()`` / ``execute()`` 齐备，
   声明的算子名必须真实存在（不能凭空声明）。
2. 能力声明诚实：声明支持 ⇒ 必须跑通且与参考语义一致；
   未声明 ⇒ 必须抛 ``BackendCapabilityError``，不能静默执行、也不能换个实现硬算。
3. 数值一致：逐 case 与参考后端比对（rtol / atol）。
4. 可移植性：图集只含 ``PORTABLE`` 节点，且 JSON 往返后语义不变。
5. 确定性：同一输入执行两次必须逐位相同——否则它不能充当一致性基准。

**为什么对"部分支持"采用条件判定**：CUDA 后端声明了 softmax，但只支持最后一维。
把它判成失败是错的（后端已经如实报错），判成通过也不完整（能力声明里看不出边界）。
所以套件的规则是：声明支持但算不了的，只要**显式拒绝**（BackendCapabilityError）
就算合规，并单独记为 ``refused``；真正的失败只有两种——算错了，或者
用"能力协商之外的方式"（别的异常、静默降级、未声明却执行）绕过契约。

仅有这条规则会让"全部拒绝"也能通过，因此再加一条覆盖度检查：
每个被声明、且套件里有对应用例的算子，至少要有一次**成功执行**。
这样"声称支持但一个用例都跑不了"会被抓住。

与 5.8 的序列化配合：图集由 IR 构造，可导出为 JSON 交给第三方实现，
对方只要实现 ``Backend`` 接口就能用同一套图自证。

报告里另有一行 ``contracts``：后端声称实现了哪些**可选契约**
（``DeviceContract`` / ``MemoryContract``，见 5.10）。它是信息项，不参与判定——
最小面（三个方法）就足够当一个合格后端，缺可选契约不是缺陷。放在报告里是因为
"这个后端能被问什么"本身就是接入方要判断的事。
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .backends.base import BackendCapabilityError
from .backends.capabilities import (
    BackendCapabilities,
    capability_gaps,
    to_capabilities,
)
from .backends.contracts import implemented_contracts
from .backends.reference_backend import ReferenceBackend
from .ir.graph import Graph
from .ir.ops import OPS
from .ir.serialize import dumps, loads, to_dict
from .ir.types import boolean, c64, f32, f64, i8, i32
from .ir.verifier import verify_portable_only
from .models.transformer import BlockSpec, build_transformer_block

__all__ = [
    "Case",
    "CaseResult",
    "CheckResult",
    "Report",
    "EXECUTED",
    "REFUSED",
    "FAILED",
    "default_cases",
    "run_conformance",
]

EXECUTED = "executed"
REFUSED = "refused"
FAILED = "failed"


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """一个固定的检验用例：一张 IR 图 + 一组输入。

    ``ops`` 由图自身推导，不手写——手写会与图脱节，而判断"该跑通还是该拒绝"
    完全依赖它。
    """

    name: str
    graph: Graph
    inputs: Mapping[str, np.ndarray]
    ops: FrozenSet[str]
    rtol: float = 1e-4
    atol: float = 1e-4
    note: str = ""
    # 位级确定性。scatter-add 类算子（重复下标累加同一地址）的结果依赖
    # 硬件原子加的执行顺序，浮点加法不满足结合律，两次运行可能差 1 ulp。
    # 这类用例仍要参与数值对拍，但不参与"两次结果必须完全相同"的检查——
    # PyTorch 的 index_add / embedding 反向同样是非确定的，隐藏这一点才是撒谎。
    deterministic: bool = True


@dataclass
class CaseResult:
    name: str
    ops: Tuple[str, ...]
    status: str
    detail: str = ""
    max_error: float = 0.0


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Report:
    backend: str
    capabilities: FrozenSet[str]
    checks: List[CheckResult]
    cases: List[CaseResult]
    contracts: Tuple[str, ...] = ()
    notes: List[str] = field(default_factory=list)

    @property
    def failed_cases(self) -> List[CaseResult]:
        return [c for c in self.cases if c.status == FAILED]

    @property
    def executed(self) -> List[CaseResult]:
        return [c for c in self.cases if c.status == EXECUTED]

    @property
    def refused(self) -> List[CaseResult]:
        return [c for c in self.cases if c.status == REFUSED]

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks) and not self.failed_cases

    @property
    def covered_ops(self) -> Tuple[str, ...]:
        ops = {op for c in self.executed for op in c.ops}
        return tuple(sorted(ops))

    def text(self) -> str:
        optionals = ", ".join(self.contracts) if self.contracts else "仅最小面（未实现可选契约）"
        lines = [
            "AetherFlow conformance report",
            f"backend : {self.backend}",
            f"caps    : {len(self.capabilities)} 个算子",
            f"contracts: {optionals}",
            "checks  :",
        ]
        for check in self.checks:
            mark = "PASS" if check.ok else "FAIL"
            lines.append(f"  [{mark}] {check.name:<18}{check.detail}")

        lines.append(
            f"cases   : {len(self.executed)} executed / "
            f"{len(self.refused)} refused / {len(self.failed_cases)} failed"
        )
        for case in self.cases:
            ops = ",".join(case.ops)
            head = f"  [{case.status:<8}] {case.name:<26} ops={ops:<28}"
            if case.status == EXECUTED:
                lines.append(f"{head} err={case.max_error:.1e}")
            else:
                detail = case.detail if len(case.detail) <= 72 else case.detail[:69] + "..."
                lines.append(f"{head} {detail}")

        if self.notes:
            lines.append("notes   :")
            lines.extend(f"  - {n}" for n in self.notes)

        lines.append(f"result  : {'PASS' if self.ok else 'FAIL'}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.text()


# --------------------------------------------------------------------------
# 用例构造
# --------------------------------------------------------------------------


def _case(
    name: str,
    build: Callable[[np.random.Generator], Tuple[Graph, Dict[str, np.ndarray]]],
    rtol: float = 1e-4,
    atol: float = 1e-4,
    note: str = "",
    deterministic: bool = True,
) -> Case:
    # 用名字的 CRC 作种子：输入每次运行都完全一致，报告才能拿来 diff
    seed = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
    graph, feed = build(np.random.default_rng(seed))
    return Case(
        name=name,
        graph=graph,
        inputs=feed,
        ops=frozenset(n.op.name for n in graph.nodes),
        rtol=rtol,
        atol=atol,
        note=note,
        deterministic=deterministic,
    )


def _f32(rng: np.random.Generator, shape: tuple, scale: float = 1.0, offset: float = 0.0):
    return (rng.standard_normal(shape) * scale + offset).astype(np.float32)


def _binary(op: str, shape_a: tuple, shape_b: tuple, offset_b: float = 0.0):
    def build(rng):
        g = Graph(op)
        a = g.input("a", f32[shape_a])
        b = g.input("b", f32[shape_b])
        g.output(g.add_node(op, [a, b]), "y")
        return g, {"a": _f32(rng, shape_a), "b": _f32(rng, shape_b, offset=offset_b)}

    return build


def _cmp_case(op: str, shape_a: tuple, shape_b: tuple, offset_b: float = 0.0):
    """比较用例：输入先取整。

    连续随机数之间几乎不可能相等，``eq`` / ``ne`` 会退化成"恒 false / 恒
    true"，掩盖真正的错误。取整之后两种结果都能覆盖到，广播用例还同时
    检验坐标还原。
    """

    def build(rng):
        g = Graph(op)
        a = g.input("a", f32[shape_a])
        b = g.input("b", f32[shape_b])
        g.output(g.add_node(op, [a, b]), "y")
        av = np.round(_f32(rng, shape_a) * 2.0).astype(np.float32)
        bv = np.round(_f32(rng, shape_b) * 2.0 + offset_b).astype(np.float32)
        return g, {"a": av, "b": bv}

    return build


def _where_case(shape_c: tuple, shape_a: tuple, shape_b: tuple):
    """``where`` 用例：条件是真 bool 输入，两个分支支持三方广播。"""

    def build(rng):
        g = Graph("where")
        c = g.input("c", boolean[shape_c])
        a = g.input("a", f32[shape_a])
        b = g.input("b", f32[shape_b])
        g.output(g.where(c, a, b), "y")
        cv = rng.integers(0, 2, size=shape_c).astype(bool)
        return g, {"c": cv, "a": _f32(rng, shape_a), "b": _f32(rng, shape_b)}

    return build


def _unary(op: str, shape: tuple, scale: float = 1.0):
    def build(rng):
        g = Graph(op)
        x = g.input("x", f32[shape])
        g.output(g.add_node(op, [x]), "y")
        return g, {"x": _f32(rng, shape, scale=scale)}

    return build


def _matmul(shape_a: tuple, shape_b: tuple):
    def build(rng):
        g = Graph("matmul")
        a = g.input("a", f32[shape_a])
        b = g.input("b", f32[shape_b])
        g.output(g.matmul(a, b), "y")
        return g, {"a": _f32(rng, shape_a), "b": _f32(rng, shape_b)}

    return build


def _quantize_i8(shape: tuple = (8, 6)):
    def build(rng):
        g = Graph("quantize_i8")
        w = g.input("w", f32[shape])
        s = g.input("s", f32[(shape[-1],)])
        g.output(g.add_node("quantize_i8", [w, s]), "q")
        # scale 必须严格为正；用一个与权重同量级的正数，避免整片饱和到端点
        scale = (np.abs(_f32(rng, (shape[-1],), scale=0.2)) + 0.05).astype(np.float32)
        return g, {"w": _f32(rng, shape, scale=0.5), "s": scale}

    return build


def _dequantize_i8(shape: tuple = (8, 6)):
    def build(rng):
        g = Graph("dequantize_i8")
        q = g.input("q", i8[shape])
        s = g.input("s", f32[(shape[-1],)])
        g.output(g.add_node("dequantize_i8", [q, s]), "w")
        scale = (np.abs(_f32(rng, (shape[-1],), scale=0.2)) + 0.05).astype(np.float32)
        return g, {"q": rng.integers(-127, 128, size=shape, dtype=np.int8), "s": scale}

    return build


def _fake_quantize_i8(shape: tuple = (8, 6)):
    def build(rng):
        g = Graph("fake_quantize_i8")
        x = g.input("x", f32[shape])
        s = g.input("s", f32[(shape[-1],)])
        g.output(g.add_node("fake_quantize_i8", [x, s]), "y")
        scale = (np.abs(_f32(rng, (shape[-1],), scale=0.2)) + 0.05).astype(np.float32)
        return g, {"x": _f32(rng, shape, scale=0.5), "s": scale}

    return build


def _fake_quantize_i8_lsq(shape: tuple = (8, 6)):
    """LSQ 的前向与 STE 版逐位相同，因此这里只校验"同一个出口"。

    反向规则不参与 conformance（套件只跑前向）；把形态放进来是为了让
    设备端"声明承接"与"真的能算"对得上——LSQ 在设备上跑训练时，前向走的
    仍是同一对量化 / 反量化内核。
    """

    def build(rng):
        g = Graph("fake_quantize_i8_lsq")
        x = g.input("x", f32[shape])
        s = g.input("s", f32[(shape[-1],)])
        g.output(g.add_node("fake_quantize_i8_lsq", [x, s]), "y")
        scale = (np.abs(_f32(rng, (shape[-1],), scale=0.2)) + 0.05).astype(np.float32)
        return g, {"x": _f32(rng, shape, scale=0.5), "s": scale}

    return build


def _linear_i8(x_shape: tuple = (2, 4, 8), w_shape: tuple = (8, 6)):
    def build(rng):
        g = Graph("linear_i8")
        x = g.input("x", f32[x_shape])
        q = g.input("q", i8[w_shape])
        s = g.input("s", f32[(w_shape[-1],)])
        b = g.input("b", f32[(w_shape[-1],)])
        g.output(g.add_node("linear_i8", [x, q, s, b]), "y")
        scale = (np.abs(_f32(rng, (w_shape[-1],), scale=0.2)) + 0.05).astype(np.float32)
        return g, {
            "x": _f32(rng, x_shape, scale=0.5),
            "q": rng.integers(-127, 128, size=w_shape, dtype=np.int8),
            "s": scale,
            "b": _f32(rng, (w_shape[-1],), scale=0.1),
        }

    return build


def _linear_i8a8(x_shape: tuple = (2, 3, 8), w_shape: tuple = (8, 6)):
    def build(rng):
        g = Graph("linear_i8a8")
        q = g.input("x_q", i8[x_shape])
        sx = g.input("x_scale", f32[x_shape[:-1] + (1,)])
        wq = g.input("w_q", i8[w_shape])
        sw = g.input("w_scale", f32[(w_shape[-1],)])
        b = g.input("b", f32[(w_shape[-1],)])
        g.output(g.add_node("linear_i8a8", [q, sx, wq, sw, b]), "y")
        return g, {
            "x_q": rng.integers(-127, 128, size=x_shape, dtype=np.int8),
            "x_scale": (
                np.abs(_f32(rng, x_shape[:-1] + (1,), scale=0.1)) + 0.02
            ).astype(np.float32),
            "w_q": rng.integers(-127, 128, size=w_shape, dtype=np.int8),
            "w_scale": (
                np.abs(_f32(rng, (w_shape[-1],), scale=0.2)) + 0.05
            ).astype(np.float32),
            "b": _f32(rng, (w_shape[-1],), scale=0.1),
        }

    return build


def _quantize_i8_zp(shape: tuple = (8, 6)):
    """非对称量化：值域用满 ``[-128, 127]``，零点按通道不同。

    零点覆盖三个区域：``-128``（把 0 映射到负端点）、``0``（退化为对称但
    负端点仍是 -128）、正零点是"0 落在格子内部"。参考实现先舍入再平移，
    这三类零点足以把"先加零点再舍入"的错误写法区分出来。
    """

    def build(rng):
        g = Graph("quantize_i8_zp")
        w = g.input("w", f32[shape])
        s = g.input("s", f32[(shape[-1],)])
        zp = g.input("zp", i8[(shape[-1],)])
        g.output(g.add_node("quantize_i8_zp", [w, s, zp]), "q")
        scale = (np.abs(_f32(rng, (shape[-1],), scale=0.2)) + 0.05).astype(np.float32)
        zps = np.array([-128, 0, 1, -127, 127, -64], dtype=np.int8)
        return g, {"w": _f32(rng, shape, scale=0.5), "s": scale, "zp": zps}

    return build


def _dequantize_i8_zp(shape: tuple = (8, 6)):
    """非对称反量化：``q - zp`` 在 int32 里做，``(-128) - 127`` 不能回绕成 1。"""

    def build(rng):
        g = Graph("dequantize_i8_zp")
        q = g.input("q", i8[shape])
        s = g.input("s", f32[(shape[-1],)])
        zp = g.input("zp", i8[(shape[-1],)])
        g.output(g.add_node("dequantize_i8_zp", [q, s, zp]), "w")
        scale = (np.abs(_f32(rng, (shape[-1],), scale=0.2)) + 0.05).astype(np.float32)
        zps = np.array([-128, 0, 1, -127, 127, -64], dtype=np.int8)
        return g, {
            "q": rng.integers(-128, 128, size=shape, dtype=np.int8),
            "s": scale,
            "zp": zps,
        }

    return build


def _linear_i8_zp(x_shape: tuple = (2, 4, 8), w_shape: tuple = (8, 6)):
    def build(rng):
        g = Graph("linear_i8_zp")
        x = g.input("x", f32[x_shape])
        q = g.input("q", i8[w_shape])
        s = g.input("s", f32[(w_shape[-1],)])
        zp = g.input("zp", i8[(w_shape[-1],)])
        b = g.input("b", f32[(w_shape[-1],)])
        g.output(g.add_node("linear_i8_zp", [x, q, s, zp, b]), "y")
        scale = (np.abs(_f32(rng, (w_shape[-1],), scale=0.2)) + 0.05).astype(np.float32)
        zps = np.array([-128, 0, 1, -127, 127, -64], dtype=np.int8)
        return g, {
            "x": _f32(rng, x_shape, scale=0.5),
            "q": rng.integers(-128, 128, size=w_shape, dtype=np.int8),
            "s": scale,
            "zp": zps,
            "b": _f32(rng, (w_shape[-1],), scale=0.1),
        }

    return build


def _linear_i8a8_zp(x_shape: tuple = (2, 3, 8), w_shape: tuple = (8, 6)):
    def build(rng):
        g = Graph("linear_i8a8_zp")
        q = g.input("x_q", i8[x_shape])
        sx = g.input("x_scale", f32[x_shape[:-1] + (1,)])
        xzp = g.input("x_zp", i8[x_shape[:-1] + (1,)])
        wq = g.input("w_q", i8[w_shape])
        sw = g.input("w_scale", f32[(w_shape[-1],)])
        wzp = g.input("w_zp", i8[(w_shape[-1],)])
        b = g.input("b", f32[(w_shape[-1],)])
        g.output(
            g.add_node(
                "linear_i8a8_zp", [q, sx, xzp, wq, sw, wzp, b]
            ),
            "y",
        )
        return g, {
            "x_q": rng.integers(-128, 128, size=x_shape, dtype=np.int8),
            "x_scale": (
                np.abs(_f32(rng, x_shape[:-1] + (1,), scale=0.1)) + 0.02
            ).astype(np.float32),
            # 激活零点全为 -128（ReLU 之后的典型形态：值域全正），
            # 权重零点逐通道不同，覆盖"零点确实非零"的路径。
            "x_zp": np.full(x_shape[:-1] + (1,), -128, dtype=np.int8),
            "w_q": rng.integers(-128, 128, size=w_shape, dtype=np.int8),
            "w_scale": (
                np.abs(_f32(rng, (w_shape[-1],), scale=0.2)) + 0.05
            ).astype(np.float32),
            "w_zp": np.array([-128, 0, 1, -127, 127, -64], dtype=np.int8),
            "b": _f32(rng, (w_shape[-1],), scale=0.1),
        }

    return build


def _softmax(axis: int = -1, scaled: bool = False, shape: tuple = (6, 11)):
    def build(rng):
        g = Graph("softmax")
        x = g.input("x", f32[shape])
        feed = {"x": _f32(rng, shape, scale=2.0)}
        args = [x]
        if scaled:
            s = g.input("s", f32[1])
            args.append(s)
            feed["s"] = np.array([0.7], dtype=np.float32)
        g.output(g.add_node("softmax", args, {"axis": axis}), "y")
        return g, feed

    return build


def _attention(with_mask: bool = False, shape: tuple = (2, 2, 3, 2)):
    def build(rng):
        g = Graph("attention")
        q = g.input("q", f32[shape])
        k = g.input("k", f32[shape])
        v = g.input("v", f32[shape])
        feed = {
            "q": _f32(rng, shape),
            "k": _f32(rng, shape),
            "v": _f32(rng, shape),
        }
        args = [q, k, v]
        if with_mask:
            mask_shape = (1, 1, shape[-2], shape[-2])
            args.append(g.input("mask", f32[mask_shape]))
            feed["mask"] = np.zeros(mask_shape, dtype=np.float32)
        g.output(g.attention(*args, scale=1.0 / np.sqrt(shape[-1])), "y")
        return g, feed

    return build


def _layernorm(axes: tuple = (-1,), shape: tuple = (5, 16), norm: tuple = (16,)):
    def build(rng):
        g = Graph("layernorm")
        x = g.input("x", f32[shape])
        gamma = g.input("gamma", f32[norm])
        beta = g.input("beta", f32[norm])
        g.output(g.layernorm(x, gamma, beta, axes=axes, eps=1e-5), "y")
        return g, {
            "x": _f32(rng, shape),
            "gamma": _f32(rng, norm),
            "beta": _f32(rng, norm),
        }

    return build


def _batch_norm():
    def build(rng):
        g = Graph("batch_norm")
        x = g.input("x", f32[2, 3, 4, 4])
        gamma = g.input("gamma", f32[3])
        beta = g.input("beta", f32[3])
        g.output(g.batch_norm(x, gamma, beta, axis=1, eps=1e-5), "y")
        return g, {
            "x": _f32(rng, (2, 3, 4, 4)),
            "gamma": _f32(rng, (3,)),
            "beta": _f32(rng, (3,)),
        }

    return build


def _batch_norm_stats(shape: tuple = (2, 3, 4, 4)):
    def build(rng):
        g = Graph("batch_norm_stats")
        x = g.input("x", f32[shape])
        g.output(g.batch_norm_stats(x, axis=1), "y")
        return g, {"x": _f32(rng, shape)}

    return build


def _reduce_sum(axis: int = -1, keepdims: bool = False, shape: tuple = (4, 7)):
    def build(rng):
        g = Graph("reduce_sum")
        x = g.input("x", f32[shape])
        g.output(g.reduce_sum(x, axis=axis, keepdims=keepdims), "y")
        return g, {"x": _f32(rng, shape)}

    return build


def _reduce_other(op: str, axis: int = -1, keepdims: bool = False,
                  shape: tuple = (4, 7)):
    def build(rng):
        g = Graph(op)
        x = g.input("x", f32[shape])
        g.output(g.add_node(op, [x], {"axis": axis, "keepdims": keepdims}), "y")
        return g, {"x": _f32(rng, shape)}

    return build


def _argmax(axis: int = -1, keepdims: bool = False, shape: tuple = (4, 7)):
    def build(rng):
        g = Graph("argmax")
        x = g.input("x", f32[shape])
        g.output(g.add_node("argmax", [x], {"axis": axis, "keepdims": keepdims}), "y")
        return g, {"x": _f32(rng, shape)}

    return build


def _topk_mask(k: int = 2, axis: int = -1, shape: tuple = (3, 5)):
    def build(rng):
        g = Graph("topk_mask")
        x = g.input("x", f32[shape])
        g.output(g.add_node("topk_mask", [x], {"k": k, "axis": axis}), "y")
        return g, {"x": _f32(rng, shape)}

    return build


def _topk_values(k: int = 2, axis: int = -1, shape: tuple = (3, 5)):
    def build(rng):
        g = Graph("topk_values")
        x = g.input("x", f32[shape])
        g.output(g.add_node("topk_values", [x], {"k": k, "axis": axis}), "y")
        return g, {"x": _f32(rng, shape)}

    return build


def _topk_indices(k: int = 2, axis: int = -1, shape: tuple = (3, 5)):
    def build(rng):
        g = Graph("topk_indices")
        x = g.input("x", f32[shape])
        g.output(g.add_node("topk_indices", [x], {"k": k, "axis": axis}), "y")
        return g, {"x": _f32(rng, shape)}

    return build


def _reshape():
    def build(rng):
        g = Graph("reshape")
        x = g.input("x", f32[4, 6])
        g.output(g.reshape(x, shape=(2, 12)), "y")
        return g, {"x": _f32(rng, (4, 6))}

    return build


def _transpose():
    def build(rng):
        g = Graph("transpose")
        x = g.input("x", f32[2, 3, 4])
        g.output(g.transpose(x, perm=(0, 2, 1)), "y")
        return g, {"x": _f32(rng, (2, 3, 4))}

    return build


def _cast(dtype: str = "f64"):
    def build(rng):
        g = Graph("cast")
        x = g.input("x", f32[4, 5])
        g.output(g.cast(x, dtype=dtype), "y")
        return g, {"x": _f32(rng, (4, 5))}

    return build


def _complex(shape: tuple, rng: np.random.Generator) -> np.ndarray:
    return (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ).astype(np.complex64)


def _fft(norm: str = "ortho", inverse: bool = False, shape: tuple = (3, 8)):
    def build(rng):
        g = Graph("fft")
        x = g.input("x", c64[shape])
        g.output(g.add_node("fft", [x], {"norm": norm, "inverse": inverse}), "y")
        return g, {"x": _complex(shape, rng)}

    return build


def _rfft(norm: str = "backward", shape: tuple = (4, 10)):
    def build(rng):
        g = Graph("rfft")
        x = g.input("x", f32[shape])
        g.output(g.add_node("rfft", [x], {"norm": norm}), "y")
        return g, {"x": _f32(rng, shape)}

    return build


def _slice():
    def build(rng):
        g = Graph("slice")
        x = g.input("x", f32[3, 8])
        g.output(g.slice(x, dim=1, start=2, stop=6), "y")
        return g, {"x": _f32(rng, (3, 8))}

    return build


def _float64_input():
    def build(rng):
        g = Graph("f64_input")
        x = g.input("x", f64[4, 5])
        g.output(g.relu(x), "y")
        return g, {"x": _f32(rng, (4, 5)).astype(np.float64)}

    return build


def _conv2d(
    x_shape: tuple = (1, 3, 8, 8),
    w_shape: tuple = (4, 3, 3, 3),
    stride: tuple = (1, 1),
    padding: tuple = (1, 1),
    groups: int = 1,
):
    def build(rng):
        g = Graph("conv2d")
        x = g.input("x", f32[x_shape])
        w = g.input("w", f32[w_shape])
        b = g.input("b", f32[w_shape[0]])
        attrs = {"stride": stride, "padding": padding}
        if groups != 1:
            attrs["groups"] = groups
        y = g.add_node("conv2d", [x, w, b], attrs)
        g.output(y, "y")
        return g, {
            "x": _f32(rng, x_shape),
            "w": _f32(rng, w_shape, scale=0.25),
            "b": _f32(rng, (w_shape[0],), scale=0.1),
        }

    return build


def _maxpool2d(
    x_shape: tuple = (1, 4, 8, 8),
    kernel: tuple = (2, 2),
    stride: tuple = (2, 2),
    padding: tuple = (0, 0),
    ceil_mode: bool = False,
):
    def build(rng):
        g = Graph("maxpool2d")
        x = g.input("x", f32[x_shape])
        attrs = {"kernel": kernel, "stride": stride, "padding": padding}
        if ceil_mode:
            attrs["ceil_mode"] = True
        y = g.add_node("maxpool2d", [x], attrs)
        g.output(y, "y")
        return g, {"x": _f32(rng, x_shape)}

    return build


def _maxpool2d_indices(
    x_shape: tuple = (1, 4, 8, 8),
    kernel: tuple = (2, 2),
    stride: tuple = (2, 2),
):
    """PyTorch ``max_pool2d(return_indices=True)`` 的第二个输出（平面内下标）。"""

    def build(rng):
        g = Graph("maxpool2d_indices")
        x = g.input("x", f32[x_shape])
        y = g.add_node(
            "maxpool2d_indices", [x], {"kernel": kernel, "stride": stride}
        )
        g.output(y, "y")
        return g, {"x": _f32(rng, x_shape)}

    return build


def _positive_unary(op: str, shape: tuple):
    """log / sqrt 的输入必须为正，否则参考实现自己就会出 NaN，对照没意义。"""

    def build(rng):
        g = Graph(op)
        x = g.input("x", f32[shape])
        g.output(g.add_node(op, [x]), "y")
        return g, {"x": np.abs(_f32(rng, shape, scale=2.0)) + 0.5}

    return build


def _step():
    def build(rng):
        g = Graph("step")
        x = g.input("x", f32[5, 7])
        g.output(g.add_node("step", [x], {"threshold": 0.25}), "y")
        return g, {"x": _f32(rng, (5, 7))}

    return build


def _spike():
    def build(rng):
        g = Graph("spike")
        x = g.input("x", f32[5, 7])
        g.output(
            g.add_node("spike", [x], {"threshold": 1.0, "alpha": 0.5}), "y"
        )
        # 跨过阈值：既覆盖发放（1）也覆盖不发放（0）
        return g, {"x": _f32(rng, (5, 7), scale=1.5, offset=0.5)}

    return build


def _scan_linear(steps: int = 5, tail: tuple = (2, 3)):
    def build(rng):
        g = Graph("scan_linear")
        a = g.input("a", f32[(steps, *tail)])
        b = g.input("b", f32[(steps, *tail)])
        h0 = g.input("h0", f32[tail])
        g.output(g.add_node("scan_linear", [a, b, h0]), "y")
        # |a| < 1：递归不放大，误差不会被指数级吹大，对照才有意义
        return g, {
            "a": _f32(rng, (steps, *tail), scale=0.3),
            "b": _f32(rng, (steps, *tail)),
            "h0": _f32(rng, tail),
        }

    return build


def _scan_linear_backward(steps: int = 5, tail: tuple = (2, 3)):
    def build(rng):
        g = Graph("scan_linear_backward")
        a = g.input("a", f32[(steps, *tail)])
        dy = g.input("dy", f32[(steps, *tail)])
        g.output(g.add_node("scan_linear_backward", [a, dy]), "y")
        return g, {
            "a": _f32(rng, (steps, *tail), scale=0.3),
            "dy": _f32(rng, (steps, *tail)),
        }

    return build


def _cumsum(shape: tuple = (2, 4, 3), axis: int = 1):
    def build(rng):
        g = Graph("cumsum")
        x = g.input("x", f32[shape])
        g.output(g.add_node("cumsum", [x], {"axis": axis}), "y")
        # axis=1 而不是末轴：中间轴才能同时压到 outer 与 inner 两侧的折维
        return g, {"x": _f32(rng, shape)}

    return build


def _tril(shape: tuple = (2, 5, 5), diagonal: int = -1):
    def build(rng):
        g = Graph("tril")
        x = g.input("x", f32[shape])
        g.output(g.add_node("tril", [x], {"diagonal": diagonal}), "y")
        # 秩 3 + 负偏移：覆盖"前导维逐片独立"与"对角线以下"两种边界
        return g, {"x": _f32(rng, shape)}

    return build


def _clamp(lo=None, hi=None, shape: tuple = (5, 7)):
    def build(rng):
        g = Graph("clamp")
        x = g.input("x", f32[shape])
        attrs = {}
        if lo is not None:
            attrs["min"] = float(lo)
        if hi is not None:
            attrs["max"] = float(hi)
        g.output(g.add_node("clamp", [x], attrs), "y")
        # 跨度 4 倍区间：既覆盖两侧截断，也覆盖区间内的原样透传
        return g, {"x": _f32(rng, shape, scale=4.0)}

    return build


def _mod_int(shape: tuple = (4, 6)):
    def build(rng):
        g = Graph("mod")
        a = g.input("a", i32[shape])
        b = g.input("b", i32[shape])
        g.output(g.add_node("mod", [a, b]), "y")
        # 被除数必须有负值，模数也要正负混合：只测正数就分不出
        # "C 的截断取余"与 Python / torch 的 floor 语义
        divisor = rng.integers(-7, 8, size=shape, dtype=np.int32)
        divisor[divisor == 0] = 3
        return g, {
            "a": rng.integers(-50, 50, size=shape, dtype=np.int32),
            "b": divisor,
        }

    return build


def _constant():
    def build(rng):
        g = Graph("constant")
        x = g.input("x", f32[2, 3, 4])
        c = g.constant(1.5, shape=(4,))
        g.output(g.add_node("mul", [x, c]), "y")
        return g, {"x": _f32(rng, (2, 3, 4))}

    return build


def _broadcast_to():
    def build(rng):
        g = Graph("broadcast_to")
        x = g.input("x", f32[1, 4])
        g.output(g.add_node("broadcast_to", [x], {"shape": [3, 4]}), "y")
        return g, {"x": _f32(rng, (1, 4))}

    return build


def _pad():
    def build(rng):
        g = Graph("pad")
        x = g.input("x", f32[3, 5])
        g.output(g.add_node("pad", [x], {"dim": 1, "before": 1, "after": 2}), "y")
        return g, {"x": _f32(rng, (3, 5))}

    return build


def _conv2d_input_grad():
    def build(rng):
        g = Graph("conv2d_input_grad")
        dy = g.input("dy", f32[1, 4, 8, 8])
        w = g.input("w", f32[4, 3, 3, 3])
        y = g.add_node(
            "conv2d_input_grad",
            [dy, w],
            {"input_shape": [1, 3, 8, 8], "padding": (1, 1)},
        )
        g.output(y, "y")
        return g, {"dy": _f32(rng, (1, 4, 8, 8)), "w": _f32(rng, (4, 3, 3, 3), scale=0.25)}

    return build


def _conv_transpose2d_form():
    """转置卷积的 IR 形态：``conv2d_input_grad`` + ``output_padding`` 撑大的输出。

    stride=2 / padding=1 / output_padding=1 时输出是输入的 2 倍再减 1，
    多出的最后一行/列没有输入贡献（恒为 0）。这条用例盯的是"后端按 dy
    形状反推的输入尺寸仍与 input_shape 一致"——否则转置卷积会整体错位。
    """

    def build(rng):
        g = Graph("conv_transpose2d_form")
        x = g.input("x", f32[1, 4, 5, 5])
        w = g.input("w", f32[4, 3, 3, 3])
        y = g.add_node(
            "conv2d_input_grad",
            [x, w],
            {
                "input_shape": [1, 3, 10, 10],
                "stride": (2, 2),
                "padding": (1, 1),
            },
        )
        g.output(y, "y")
        return g, {"x": _f32(rng, (1, 4, 5, 5)), "w": _f32(rng, (4, 3, 3, 3), scale=0.25)}

    return build


def _conv2d_weight_grad():
    def build(rng):
        g = Graph("conv2d_weight_grad")
        x = g.input("x", f32[1, 3, 8, 8])
        dy = g.input("dy", f32[1, 4, 8, 8])
        y = g.add_node(
            "conv2d_weight_grad",
            [x, dy],
            {"weight_shape": [4, 3, 3, 3], "padding": (1, 1)},
        )
        g.output(y, "y")
        return g, {
            "x": _f32(rng, (1, 3, 8, 8)),
            "dy": _f32(rng, (1, 4, 8, 8)),
        }

    return build


def _maxpool2d_backward():
    def build(rng):
        g = Graph("maxpool2d_backward")
        x = g.input("x", f32[1, 4, 8, 8])
        dy = g.input("dy", f32[1, 4, 4, 4])
        y = g.add_node(
            "maxpool2d_backward",
            [x, dy],
            {"kernel": (2, 2), "stride": (2, 2)},
        )
        g.output(y, "y")
        return g, {"x": _f32(rng, (1, 4, 8, 8)), "dy": _f32(rng, (1, 4, 4, 4))}

    return build


def _rmsnorm(shape: tuple = (5, 16), norm: tuple = (16,)):
    def build(rng):
        g = Graph("rmsnorm")
        x = g.input("x", f32[shape])
        gamma = g.input("gamma", f32[norm])
        g.output(g.rmsnorm(x, gamma, axes=(-1,), eps=1e-6), "y")
        return g, {"x": _f32(rng, shape), "gamma": _f32(rng, norm, scale=0.5, offset=1.0)}

    return build


def _avgpool2d(
    x_shape: tuple = (1, 4, 8, 8),
    kernel: tuple = (2, 2),
    stride: tuple = (2, 2),
    padding: tuple = (0, 0),
    ceil_mode: bool = False,
    count_include_pad: bool = True,
):
    def build(rng):
        g = Graph("avgpool2d")
        x = g.input("x", f32[x_shape])
        attrs = {"kernel": kernel, "stride": stride, "padding": padding}
        if ceil_mode:
            attrs["ceil_mode"] = True
        if not count_include_pad:
            attrs["count_include_pad"] = False
        g.output(g.add_node("avgpool2d", [x], attrs), "y")
        return g, {"x": _f32(rng, x_shape)}

    return build


def _avgpool2d_backward():
    def build(rng):
        g = Graph("avgpool2d_backward")
        dy = g.input("dy", f32[1, 4, 4, 4])
        y = g.add_node(
            "avgpool2d_backward",
            [dy],
            {"input_shape": [1, 4, 8, 8], "kernel": (2, 2), "stride": (2, 2)},
        )
        g.output(y, "y")
        return g, {"dy": _f32(rng, (1, 4, 4, 4))}

    return build


def _adaptive_avg_pool2d(x_shape: tuple = (1, 3, 7, 5), output_size: tuple = (2, 3)):
    def build(rng):
        g = Graph("adaptive_avg_pool2d")
        x = g.input("x", f32[x_shape])
        y = g.add_node(
            "adaptive_avg_pool2d", [x], {"output_size": list(output_size)}
        )
        g.output(y, "y")
        return g, {"x": _f32(rng, x_shape)}

    return build


def _adaptive_avg_pool2d_backward(
    input_shape: tuple = (1, 3, 7, 5), output_size: tuple = (2, 3)
):
    def build(rng):
        n, c, _, _ = input_shape
        oh, ow = output_size
        g = Graph("adaptive_avg_pool2d_backward")
        dy = g.input("dy", f32[(n, c, oh, ow)])
        y = g.add_node(
            "adaptive_avg_pool2d_backward",
            [dy],
            {"input_shape": list(input_shape), "output_size": list(output_size)},
        )
        g.output(y, "y")
        return g, {"dy": _f32(rng, (n, c, oh, ow))}

    return build


def _concat():
    def build(rng):
        g = Graph("concat")
        a = g.input("a", f32[3, 4])
        b = g.input("b", f32[3, 2])
        c = g.input("c", f32[3, 5])
        g.output(g.add_node("concat", [a, b, c], {"axis": 1}), "y")
        return g, {"a": _f32(rng, (3, 4)), "b": _f32(rng, (3, 2)), "c": _f32(rng, (3, 5))}

    return build


def _embedding():
    def build(rng):
        g = Graph("embedding")
        table = g.input("table", f32[10, 4])
        ids = g.input("ids", i32[3, 5])
        g.output(g.add_node("embedding", [table, ids]), "y")
        return g, {
            "table": _f32(rng, (10, 4)),
            # 有意带重复下标：查表前向允许，反向必须累加
            "ids": rng.integers(0, 10, (3, 5), dtype=np.int32),
        }

    return build


def _embedding_backward():
    def build(rng):
        g = Graph("embedding_backward")
        dy = g.input("dy", f32[3, 5, 4])
        ids = g.input("ids", i32[3, 5])
        g.output(
            g.add_node("embedding_backward", [dy, ids], {"table_shape": [10, 4]}), "y"
        )
        return g, {
            "dy": _f32(rng, (3, 5, 4)),
            "ids": rng.integers(0, 10, (3, 5), dtype=np.int32),
        }

    return build


def _gather(dim: int = -1):
    """沿 dim 取下标：idx 的该维长度小于 x，其余维必须相同。"""

    def build(rng):
        g = Graph("gather")
        shape = [3, 6, 4]
        d = dim % len(shape)
        idx_shape = list(shape)
        idx_shape[d] = 2
        x = g.input("x", f32[tuple(shape)])
        idx = g.input("idx", i32[tuple(idx_shape)])
        g.output(g.add_node("gather", [x, idx], {"dim": dim}), "y")
        return g, {
            "x": _f32(rng, tuple(shape)),
            # 有意重复：同一位置被选多次是常态（反向必须累加）
            "idx": rng.integers(0, shape[d], tuple(idx_shape), dtype=np.int32),
        }

    return build


def _gather_backward(dim: int = -1):
    def build(rng):
        g = Graph("gather_backward")
        shape = [3, 6, 4]
        d = dim % len(shape)
        idx_shape = list(shape)
        idx_shape[d] = 2
        dy = g.input("dy", f32[tuple(idx_shape)])
        idx = g.input("idx", i32[tuple(idx_shape)])
        g.output(
            g.add_node(
                "gather_backward", [dy, idx], {"x_shape": shape, "dim": dim}
            ),
            "y",
        )
        return g, {
            "dy": _f32(rng, tuple(idx_shape)),
            "idx": rng.integers(0, shape[d], tuple(idx_shape), dtype=np.int32),
        }

    return build


def _cross_entropy(shape: tuple = (3, 5, 7)):
    def build(rng):
        g = Graph("cross_entropy")
        logits = g.input("logits", f32[shape])
        target = g.input("target", i32[shape[:-1]])
        g.output(g.add_node("cross_entropy", [logits, target]), "y")
        return g, {
            # 放大 logits：让 softmax 真正有区分度，而不是接近均匀分布
            "logits": _f32(rng, shape, scale=3.0),
            "target": rng.integers(0, shape[-1], shape[:-1], dtype=np.int32),
        }

    return build


def _cross_entropy_backward(shape: tuple = (3, 5, 7)):
    def build(rng):
        g = Graph("cross_entropy_backward")
        logits = g.input("logits", f32[shape])
        target = g.input("target", i32[shape[:-1]])
        dloss = g.input("dloss", f32[shape[:-1]])
        g.output(
            g.add_node("cross_entropy_backward", [logits, target, dloss]), "y"
        )
        return g, {
            "logits": _f32(rng, shape, scale=2.0),
            "target": rng.integers(0, shape[-1], shape[:-1], dtype=np.int32),
            "dloss": _f32(rng, shape[:-1]),
        }

    return build


def _spike_offsets(steps: int = 4, width: int = 5):
    """稠密脉冲 → 每步起点表（CSR 的前一半）。"""

    def build(rng):
        g = Graph("spike_offsets")
        x = g.input("x", f32[steps, width])
        g.output(g.add_node("spike_offsets", [x]), "y")
        spikes = np.zeros((steps, width), dtype=np.float32)
        # 交替发放：第 t 步从 t%2 开始每隔一个位置置 1
        for t in range(steps):
            spikes[t, t % 2 :: 2] = 1.0
        return g, {"x": spikes}

    return build


def _spike_csr(steps: int = 4, width: int = 5, capacity: int = 12):
    """起点表 + 稠密脉冲 → CSR 列号（CSR 的后一半）。"""

    def build(rng):
        g = Graph("spike_csr")
        x = g.input("x", f32[steps, width])
        offsets = g.input("offsets", i32[steps + 1])
        g.output(g.add_node("spike_csr", [x, offsets], {"capacity": capacity}), "y")
        spikes = np.zeros((steps, width), dtype=np.float32)
        for t in range(steps):
            spikes[t, t % 2 :: 2] = 1.0
        # 起点表必须与脉冲的实际非零数一致，参考实现会逐段核对
        table = np.zeros(steps + 1, dtype=np.int32)
        table[1:] = np.cumsum(np.count_nonzero(spikes, axis=1), dtype=np.int64)
        return g, {"x": spikes, "offsets": table}

    return build


def _sparse_linear(
    steps: int = 4, rows: int = 6, cols: int = 5, counts=(2, 0, 2, 1)
):
    """SNN 的事件驱动线性：CSR 事件 × 稠密权重。

    ``counts`` 里刻意留一个 0——空步必须输出全零而不是沿用上一步；下标里
    刻意带重复行——同一步内同一神经元多次发放，梯度必须累加。
    """

    def build(rng):
        offsets = np.zeros(steps + 1, dtype=np.int32)
        for i, c in enumerate(counts):
            offsets[i + 1] = offsets[i] + c
        nnz = int(offsets[-1])
        g = Graph("sparse_linear")
        weights = g.input("weights", f32[rows, cols])
        indices = g.input("indices", i32[nnz])
        values = g.input("values", f32[nnz])
        slots = g.input("offsets", i32[steps + 1])
        g.output(
            g.add_node("sparse_linear", [weights, indices, values, slots]), "y"
        )
        return g, {
            "weights": _f32(rng, (rows, cols)),
            "indices": np.array([1, 3, 3, 0, 1], dtype=np.int32),
            "values": _f32(rng, (nnz,)),
            "offsets": offsets,
        }

    return build


def _sparse_linear_weight_grad(
    steps: int = 4, rows: int = 6, cols: int = 5, counts=(2, 0, 2, 1)
):
    def build(rng):
        offsets = np.zeros(steps + 1, dtype=np.int32)
        for i, c in enumerate(counts):
            offsets[i + 1] = offsets[i] + c
        nnz = int(offsets[-1])
        g = Graph("sparse_linear_weight_grad")
        dout = g.input("dout", f32[steps, cols])
        indices = g.input("indices", i32[nnz])
        values = g.input("values", f32[nnz])
        slots = g.input("offsets", i32[steps + 1])
        g.output(
            g.add_node(
                "sparse_linear_weight_grad",
                [dout, indices, values, slots],
                {"rows": rows},
            ),
            "y",
        )
        return g, {
            "dout": _f32(rng, (steps, cols)),
            "indices": np.array([1, 3, 3, 0, 1], dtype=np.int32),
            "values": _f32(rng, (nnz,)),
            "offsets": offsets,
        }

    return build


def _sparse_linear_values_grad(
    steps: int = 4, rows: int = 6, cols: int = 5, counts=(2, 0, 2, 1)
):
    """``sparse_linear`` 对事件幅度的梯度：逐事件与所属步的 ``dout`` 做点积。

    事件表刻意含一个空步（第 1 步）与重复行（第 0、2 步都命中行 3）：前者
    检验空段的线程不写脏值，后者检验"梯度按事件独立、与重复行无关"。
    """

    def build(rng):
        offsets = np.zeros(steps + 1, dtype=np.int32)
        for i, c in enumerate(counts):
            offsets[i + 1] = offsets[i] + c
        nnz = int(offsets[-1])
        g = Graph("sparse_linear_values_grad")
        dout = g.input("dout", f32[steps, cols])
        weights = g.input("weights", f32[rows, cols])
        indices = g.input("indices", i32[nnz])
        slots = g.input("offsets", i32[steps + 1])
        g.output(
            g.add_node("sparse_linear_values_grad", [dout, weights, indices, slots]),
            "y",
        )
        return g, {
            "dout": _f32(rng, (steps, cols)),
            "weights": _f32(rng, (rows, cols)),
            "indices": np.array([1, 3, 3, 0, 1], dtype=np.int32),
            "offsets": offsets,
        }

    return build


def _spike_linear(
    steps: int = 4, rows: int = 6, cols: int = 5, counts=(2, 0, 2, 1)
):
    """融合事件线性：事件幅度直接取自脉冲矩阵，必须等于稠密矩阵乘。

    脉冲值刻意不取 1：若内核错误地拿常量当幅度，结果就对不上。
    """

    def build(rng):
        offsets = np.zeros(steps + 1, dtype=np.int32)
        for i, c in enumerate(counts):
            offsets[i + 1] = offsets[i] + c
        nnz = int(offsets[-1])
        indices = np.array([1, 3, 3, 4, 2], dtype=np.int32)[:nnz]
        spikes = np.zeros((steps, rows), dtype=np.float32)
        values = np.array([0.5, 1.5, 2.5, 0.25, 1.0], dtype=np.float32)[:nnz]
        for s in range(steps):
            lo, hi = int(offsets[s]), int(offsets[s + 1])
            spikes[s, indices[lo:hi]] = values[lo:hi]
        g = Graph("spike_linear")
        sp = g.input("spikes", f32[steps, rows])
        weights = g.input("weights", f32[rows, cols])
        idx = g.input("indices", i32[nnz])
        slots = g.input("offsets", i32[steps + 1])
        g.output(g.add_node("spike_linear", [sp, weights, idx, slots]), "y")
        return g, {
            "spikes": spikes,
            "weights": _f32(rng, (rows, cols)),
            "indices": indices,
            "offsets": offsets,
        }

    return build


def _spike_linear_weight_grad(
    steps: int = 4, rows: int = 6, cols: int = 5, counts=(2, 0, 2, 1)
):
    def build(rng):
        offsets = np.zeros(steps + 1, dtype=np.int32)
        for i, c in enumerate(counts):
            offsets[i + 1] = offsets[i] + c
        nnz = int(offsets[-1])
        indices = np.array([1, 3, 3, 4, 2], dtype=np.int32)[:nnz]
        spikes = np.zeros((steps, rows), dtype=np.float32)
        values = np.array([0.5, 1.5, 2.5, 0.25, 1.0], dtype=np.float32)[:nnz]
        for s in range(steps):
            lo, hi = int(offsets[s]), int(offsets[s + 1])
            spikes[s, indices[lo:hi]] = values[lo:hi]
        g = Graph("spike_linear_weight_grad")
        dout = g.input("dout", f32[steps, cols])
        sp = g.input("spikes", f32[steps, rows])
        idx = g.input("indices", i32[nnz])
        slots = g.input("offsets", i32[steps + 1])
        g.output(
            g.add_node("spike_linear_weight_grad", [dout, sp, idx, slots]), "y"
        )
        return g, {
            "dout": _f32(rng, (steps, cols)),
            "spikes": spikes,
            "indices": indices,
            "offsets": offsets,
        }

    return build


def _dropout_mask(p: float = 0.25):
    def build(rng):
        g = Graph("dropout_mask")
        x = g.input("x", f32[4, 6])
        g.output(g.add_node("dropout_mask", [x], {"p": p, "seed": 20240614}), "y")
        # x 的数值不参与计算，但图里仍然是一条真实的输入边
        return g, {"x": _f32(rng, (4, 6))}

    return build


def _dropout_train(p: float = 0.3):
    """训练态的 dropout：``x * dropout_mask(x)``。

    mask 与 mul 共用同一个 x，反向时 dropout 那条边必须是 None（掩码与数值
    无关），梯度只经乘法回传——这个用例同时钉住这两条语义。
    """

    def build(rng):
        g = Graph("dropout_train")
        x = g.input("x", f32[5, 7])
        mask = g.add_node("dropout_mask", [x], {"p": p, "seed": 7})
        g.output(g.mul(x, mask), "y")
        return g, {"x": _f32(rng, (5, 7))}

    return build


def _transformer_block():
    def build(rng):
        spec = BlockSpec(d_model=32, num_heads=4, d_ff=64)
        g = Graph("block")
        _, out = build_transformer_block(g, spec)
        g.output(out, "y")

        feed = {"x": _f32(rng, (2, 4, spec.d_model), scale=0.5)}
        for name, shape in spec.param_shapes.items():
            if name == "scale":
                feed[name] = np.array([1.0 / math.sqrt(spec.head_dim)], dtype=np.float32)
            else:
                feed[name] = _f32(rng, shape, scale=0.1)
        return g, feed

    return build


def default_cases() -> List[Case]:
    """固定图集。每个算子至少一个用例，另含"后端边界"用例。

    ``note`` 里写的是这个用例存在的原因，不是它做了什么——
    前者才是报告被读时的价值。
    """
    return [
        # 逐元素：基本形态
        _case("add_basic", _binary("add", (4, 6), (4, 6))),
        _case("sub_basic", _binary("sub", (4, 6), (4, 6))),
        _case("mul_basic", _binary("mul", (4, 6), (4, 6))),
        _case("div_basic", _binary("div", (4, 6), (4, 6), offset_b=3.0),
              note="除数偏移 3，避免靠近 0"),
        # 比较：输出 bool，是掩码的唯一产生端
        _case("eq_same", _cmp_case("eq", (4, 6), (4, 6))),
        _case("ne_same", _cmp_case("ne", (4, 6), (4, 6))),
        _case("lt_same", _cmp_case("lt", (4, 6), (4, 6))),
        _case("le_same", _cmp_case("le", (4, 6), (4, 6))),
        _case("gt_same", _cmp_case("gt", (4, 6), (4, 6))),
        _case("ge_same", _cmp_case("ge", (4, 6), (4, 6))),
        _case("gt_broadcast_bias", _cmp_case("gt", (2, 3, 4), (4,)),
              note="比较复用广播骨架，检验 strided 坐标还原"),
        _case("lt_broadcast_scalar", _cmp_case("lt", (3, 5), (1,))),
        # 掩码消费端：条件与两个分支三方广播
        _case("where_same", _where_case((4, 6), (4, 6), (4, 6))),
        _case("where_broadcast", _where_case((2, 1, 4), (2, 3, 4), (4,)),
              note="条件 / 分支各自广播，检验三组 stride 的坐标还原"),
        # 逐元素：广播（前端把 linear 降级成 matmul + add 后的真实形态）
        _case("add_broadcast_bias", _binary("add", (2, 3, 4), (4,)),
              note="尾部维广播，bias 的典型形态"),
        _case("mul_broadcast_scalar", _binary("mul", (3, 5), (1,))),
        _case("add_broadcast_both", _binary("add", (2, 1, 4), (1, 3, 1))),
        _case("sub_broadcast_5d", _binary("sub", (2, 3, 1, 4, 5), (1, 1, 6, 1, 1)),
              note="高秩广播，检验维折叠"),
        # 一元
        _case("relu", _unary("relu", (7, 9))),
        _case("silu", _unary("silu", (7, 9), scale=3.0)),
        _case("gelu", _unary("gelu", (7, 9), scale=3.0),
              note="tanh 近似：与 PyTorch approximate='tanh' 逐点一致"),
        # 自动微分需要的初等函数与常量/广播原语
        _case("exp", _unary("exp", (6, 8))),
        _case("log", _positive_unary("log", (6, 8)),
              note="输入必须为正，否则参考实现自己就出 NaN"),
        _case("sqrt", _positive_unary("sqrt", (6, 8))),
        _case("neg", _unary("neg", (6, 8))),
        _case("tanh", _unary("tanh", (6, 8))),
        _case("sigmoid", _unary("sigmoid", (6, 8))),
        _case("erf", _unary("erf", (6, 8), scale=2.0),
              note="精确 GELU 的组成部分；PyTorch approximate='none' 用的就是它"),
        _case("sign", _unary("sign", (6, 8), scale=2.0),
              note="三路选择：0 保持 0，NaN 两个比较都为假而原样透传"),
        _case("round", _unary("round", (6, 8), scale=2.0),
              note="half-to-even，必须与 torch.round 一致（roundf 是 half-away）"),
        _case("floor", _unary("floor", (6, 8), scale=2.0)),
        _case("ceil", _unary("ceil", (6, 8), scale=2.0)),
        _case("step", _step(),
              note="Heaviside 阶跃，SNN 的脉冲发射与 ReLU 的导数都用它"),
        _case("spike", _spike(),
              note="前向与 step 相同；差异只在反向的替代梯度（SNN 可训练的前提）"),
        _case("constant", _constant(),
              note="零输入节点：图自包含的前提，反向图不能靠外部喂常量"),
        _case("fft_ortho", _fft(),
              note="复数 dtype 的参照语义；CUDA 侧由 cuFFT 承接"),
        _case("fft_inverse", _fft(norm="backward", inverse=True),
              note="逆变换：cuFFT 不归一化，缩放必须由实现显式补上"),
        _case("rfft_backward", _rfft(),
              note="实数半谱：形状 (..., n) -> (..., n//2+1)，dtype f32 -> c64"),
        _case("broadcast_to", _broadcast_to(),
              note="归约/softmax 反向把梯度广播回原形状的唯一手段"),
        _case("pad", _pad(), note="slice 的反向；补零必须与切片严格对称"),
        # 序列扫描（SSM 的线性递归）
        _case("scan_linear", _scan_linear(),
              note="SSM 的全部递归都收在这一个算子；时间维必须静态，否则后端无从展开"),
        _case("scan_linear_backward", _scan_linear_backward(),
              note="伴随递归：从后往前累积，是 scan_linear 反向的唯一手段"),
        _case("cumsum", _cumsum(),
              note="非末轴前缀和：折成 (outer, steps, inner) 后并行链数来自 outer*inner"),
        _case("tril", _tril(),
              note="前导维逐片独立；negative diagonal 覆盖对角线以下才保留的情况"),
        # 数值稳定（SSM）与整数哈希（Engram）
        _case("clamp_range", _clamp(-1.0, 1.0),
              note="双侧截断：反向是闭区间 0/1 门，边界点必须保留梯度"),
        _case("clamp_min_only", _clamp(lo=-2.0),
              note="缺省上界：has_hi 开关必须真的关掉那一侧的比较"),
        _case("clamp_max_only", _clamp(hi=2.0),
              note="缺省下界：SSM 的 dt.clamp(max=2) 就是这一形态"),
        _case("mod_int_floor", _mod_int(),
              note="负被除数 / 负模数都取 floor 语义；C 的截断取余会算错"),
        # SNN 事件驱动（稀疏）
        _case("spike_offsets", _spike_offsets(),
              note="统计每步非零数并给出起点表；与 spike_csr 拆成两个单输出节点"),
        _case("spike_offsets_strided", _spike_offsets(steps=300, width=8),
              note="行数超过一个 block：内核靠跨步循环覆盖全部行，前缀和仍要有序"),
        _case("spike_csr", _spike_csr(),
              note="按起点表写 CSR 列号：同一段内必须按列号递增，累加顺序才能与稠密一致"),
        _case("sparse_linear", _sparse_linear(),
              note="CSR 事件 × 稠密权重：稠密脉冲矩阵不物化；含一个空步，必须输出全零"),
        _case("sparse_linear_weight_grad", _sparse_linear_weight_grad(),
              note="散射累加回权重行：同一行被多次命中时必须是累加而不是覆盖"),
        _case("sparse_linear_values_grad", _sparse_linear_values_grad(),
              note="逐事件回收一行点积：空步不写脏值，重复行不改变事件梯度"),
        _case("spike_linear", _spike_linear(),
              note="事件幅度直接取自脉冲矩阵：必须等于稠密矩阵乘，含一个空步"),
        _case("spike_linear_weight_grad", _spike_linear_weight_grad(),
              note="融合事件线性的权重梯度：值从脉冲读，重复行必须累加"),
        # 卷积 / 池化的反向（反向图用的语义算子，不是某个后端的内核）
        _case("conv2d_input_grad", _conv2d_input_grad(),
              note="散射关系写成独立语义算子，而不是引入第二套卷积语义"),
        _case("conv_transpose2d_form", _conv_transpose2d_form(),
              note="转置卷积就是 conv2d_input_grad：output_padding 只体现在 input_shape 上"),
        _case("conv2d_weight_grad", _conv2d_weight_grad()),
        _case("maxpool2d_backward", _maxpool2d_backward(),
              note="并列最大值取第一个，结果确定才能对拍"),
        _case("avgpool2d_2x2", _avgpool2d(),
              note="默认分母是 kernel 面积（padding 的零也计入），反向因此是常数缩放"),
        _case("avgpool2d_pad1", _avgpool2d(kernel=(3, 3), stride=(1, 1), padding=(1, 1)),
              note="重叠窗口 + padding：默认除数仍是常数，不是有效元素个数"),
        _case("avgpool2d_no_count_pad",
              _avgpool2d(kernel=(3, 3), stride=(1, 1), padding=(1, 1),
                         count_include_pad=False),
              note="count_include_pad=False：分母换成窗口与输入的重叠面积，逐格不同"),
        _case("avgpool2d_ceil", _avgpool2d(kernel=(3, 3), stride=(2, 2), ceil_mode=True),
              note="ceil_mode 与 maxpool2d 同规则：先向上取整，再去掉起点越界的窗口"),
        _case("avgpool2d_backward", _avgpool2d_backward(),
              note="均摊回原位置；窗口重叠时梯度必须叠加而不是覆盖"),
        # 矩阵
        _case("matmul_2d", _matmul((4, 8), (8, 6))),
        _case("matmul_batched", _matmul((2, 4, 8), (2, 8, 6))),
        _case("matmul_weight_shared", _matmul((3, 4, 8), (8, 6)),
              note="权重对所有 batch 共用"),
        _case("matmul_partial_batch", _matmul((2, 1, 4, 8), (2, 3, 8, 6)),
              note="批量的部分广播：可用标量 stride 的后端会拒绝"),
        # weight-only int8
        _case("quantize_i8_channel", _quantize_i8(),
              note="RNE 舍入 + [-127,127] 饱和；i8 输出必须与参考逐位相同，"
                   "不是近似对齐"),
        _case("dequantize_i8_channel", _dequantize_i8(),
              note="放大回 f32：权重驻留宽度从 4 字节降到 1 字节的唯一读出口"),
        _case("fake_quantize_i8_channel", _fake_quantize_i8(),
              note="QAT 的假量化：前向必须逐位等于 dequantize_i8(quantize_i8(x, s), s)，"
                   "设备端由已有两个内核组合承接"),
        _case("fake_quantize_i8_lsq_channel", _fake_quantize_i8_lsq(),
              note="LSQ 版前向与上一形态逐位相同；差异只在反向（dx 带非饱和掩码、"
                   "scale 接收梯度），设备端仍复用同一对内核"),
        _case("linear_i8_bias", _linear_i8(),
              note="反量化 GEMM 的完整形态（含 bias）：数值必须等于先 dequantize "
                   "再 matmul"),
        # W8A8 int8
        _case("linear_i8a8_bias", _linear_i8a8(),
              note="int8 激活 × int8 权重先在 int32 里累加，再乘两个 scale；"
                   "尚未承接 int8 矩阵乘的后端必须显式拒绝，而不是退回 f32 累加"),
        # 非对称（带零点）int8
        _case("quantize_i8_zp_channel", _quantize_i8_zp(),
              note="非对称量化用满 [-128,127]，零点覆盖 -128 / 0 / 正数；"
                   "按通道给定的零点必须是先舍入再平移，不是先平移再舍入"),
        _case("dequantize_i8_zp_channel", _dequantize_i8_zp(),
              note="q - zp 在 int32 里算：(-128) - 127 = -255 不能回绕成 1"),
        _case("linear_i8_zp_bias", _linear_i8_zp(),
              note="非对称 weight-only：数值必须等于 dequantize_i8_zp 之后再 matmul"),
        _case("linear_i8a8_zp_bias", _linear_i8a8_zp(),
              note="非对称 W8A8：先各自减零点再 int32 累加；"
                   "激活零点取 -128（全正激活的典型形态），权重零点逐通道不同"),
        # 卷积
        _case("conv2d_3x3_pad1", _conv2d(),
              note="CNN 的最小真实形态：3x3 + same padding + bias；未声明 conv2d 的后端必须显式拒绝"),
        _case("conv2d_grouped",
              _conv2d(x_shape=(1, 4, 8, 8), w_shape=(6, 2, 3, 3), groups=2),
              note="分组卷积：每组只看 C/groups 个输入通道，输出通道也按组切分"),
        _case("conv2d_depthwise",
              _conv2d(x_shape=(1, 4, 8, 8), w_shape=(4, 1, 3, 3), groups=4),
              note="depthwise：groups==C==O，每个输入通道一个独立核"),
        _case("adaptive_avg_pool2d_2x3", _adaptive_avg_pool2d(),
              note="窗口随输出位置变化且可重叠；除数不是常数，这是它不能复用 avgpool2d 的原因"),
        _case("adaptive_avg_pool2d_global", _adaptive_avg_pool2d(output_size=(1, 1)),
              note="全局平均池化：不新增算子，就是 output_size=(1,1) 的特例"),
        _case("adaptive_avg_pool2d_backward", _adaptive_avg_pool2d_backward(),
              note="窗口重叠时梯度必须累加，与 avgpool2d_backward 的常数缩放不同"),
        _case("maxpool2d_2x2", _maxpool2d(),
              note="池化不带 bias、无权重，是「纯索引」算子的代表"),
        _case("maxpool2d_3x3_pad1", _maxpool2d(kernel=(3, 3), stride=(1, 1), padding=(1, 1)),
              note="重叠窗口 + padding：边界不能把填充值混进 max"),
        _case("maxpool2d_ceil", _maxpool2d(kernel=(3, 3), stride=(2, 2), ceil_mode=True),
              note="ceil_mode：输出向上取整，并去掉窗口起点落在输入之外的多余一格"),
        _case("maxpool2d_indices", _maxpool2d_indices(),
              note="PyTorch 的 return_indices：i32 平面内下标，并列最大值取第一个"),
        # 归一化
        _case("softmax_last", _softmax()),
        _case("softmax_scaled", _softmax(scaled=True),
              note="融合形态 softmax(x, scale)，等价于 softmax(x * scale)"),
        _case("softmax_axis0", _softmax(axis=0),
              note="非最后维：末维走块内归约，其余轴走一线程一行的串行扫描"),
        _case("attention_basic", _attention(),
              note="融合注意力：一个节点代替缩放+softmax+第二个 matmul，中间分数矩阵不进图"),
        _case("attention_masked", _attention(with_mask=True),
              note="加性 mask 走第 4 个输入；分块内核必须与物化实现逐位一致"),
        _case("layernorm_last", _layernorm()),
        _case("layernorm_axis0", _layernorm(axes=(0,), norm=(5, 1)),
              note="非最后维归一化：先置换到 [keep..., norm...] 再走同一个内核"),
        _case("rmsnorm_last", _rmsnorm(),
              note="LLaMA/Mamba 的默认归一化：不做均值中心化，反向多一项 1/r 的耦合"),
        _case("batch_norm", _batch_norm(),
              note="沿除通道轴外的所有轴求统计量；gamma/beta 形状 (C,)"),
        _case("batch_norm_stats", _batch_norm_stats(),
              note="(2, C) 的 mean/var：训练循环据此更新 running stats，不进反向"),
        # 归约
        _case("reduce_sum_keepdims", _reduce_sum(keepdims=True)),
        _case("reduce_sum_last", _reduce_sum(keepdims=False)),
        _case("reduce_sum_axis0", _reduce_sum(axis=0),
              note="非最后维归约：reshape 成 (outer, reduce, inner) 后按两维铺法归约"),
        _case("reduce_max_keepdims", _reduce_other("reduce_max", keepdims=True)),
        _case("reduce_max_last", _reduce_other("reduce_max")),
        _case("reduce_max_axis0", _reduce_other("reduce_max", axis=0),
              note="非最后维归约的另一半：最大值走 block_reduce_max_y"),
        _case("reduce_mean_keepdims", _reduce_other("reduce_mean", keepdims=True)),
        _case("reduce_mean_last", _reduce_other("reduce_mean")),
        _case("reduce_mean_axis0", _reduce_other("reduce_mean", axis=0),
              note="非最后维均值：求和后除以归约长度，除法在 float 里做"),
        _case("argmax_last", _argmax(), rtol=0.0, atol=0.0,
              note="输出 i32 下标：必须精确相等，且并列时取第一次出现"),
        _case("argmax_keepdims", _argmax(keepdims=True), rtol=0.0, atol=0.0),
        _case("argmax_axis0", _argmax(axis=0, shape=(4, 7)), rtol=0.0, atol=0.0,
              note="任意单轴：非末轴先置换到最后再复用同一个内核，下标语义不变"),
        _case("argmax_axis1_keepdims", _argmax(axis=1, keepdims=True, shape=(3, 4, 5)),
              rtol=0.0, atol=0.0),
        _case("topk_mask_last", _topk_mask(), rtol=0.0, atol=0.0,
              note="路由掩码：输出 0/1，必须精确相等；同值按下标小者优先（稳定排序）"),
        _case("topk_mask_axis0", _topk_mask(k=1, axis=0, shape=(4, 3)),
              rtol=0.0, atol=0.0),
        _case("topk_values_last", _topk_values(),
              note="前 k 大的值：降序，并列时小下标优先（稳定排序）"),
        _case("topk_values_axis0", _topk_values(k=3, axis=0, shape=(4, 3))),
        _case("topk_indices_last", _topk_indices(), rtol=0.0, atol=0.0,
              note="输出 i32 下标：必须精确相等，顺序与 topk_values 一致"),
        _case("topk_indices_axis0", _topk_indices(k=1, axis=0, shape=(4, 3)),
              rtol=0.0, atol=0.0),
        _case("reduce_sum_multi_axis", _reduce_sum(axis=(0, 2), shape=(3, 4, 5)),
              note="多轴归约：置换到 [keep..., reduce...] 再合并成一个轴"),
        _case("reduce_mean_multi_axis_keepdims",
              _reduce_other("reduce_mean", axis=(1, 2), keepdims=True,
                            shape=(2, 3, 4))),
        # 形状 / 类型
        _case("reshape", _reshape()),
        _case("transpose", _transpose()),
        _case("cast_to_f64", _cast()),
        _case("slice", _slice()),
        _case("concat_axis1", _concat(),
              note="多输入拼接；反向按各段长度切片回去，不需要新的 split 算子"),
        _case("embedding_lookup", _embedding(),
              note="ids 为 i32；非整数下标的图必须在编译期就被拒绝"),
        _case("embedding_backward", _embedding_backward(),
              note="重复下标必须累加：用一个 batch 里同一 token 出现两次来覆盖",
              deterministic=False),
        _case("gather_last", _gather(),
              note="沿末维取值：输出形状等于 idx，除 dim 外每一维必须与 x 相等"),
        _case("gather_dim1", _gather(dim=1),
              note="非末维：inner > 1，必须按 (outer, dim, inner) 三轴寻址"),
        _case("gather_backward", _gather_backward(),
              note="重复下标必须累加，设备端走原子加；顺序不固定",
              deterministic=False),
        _case("cross_entropy", _cross_entropy(),
              note="融合 logsumexp：不物化 (rows, vocab) 的 softmax；越界 target 显式拒绝"),
        _case("cross_entropy_backward", _cross_entropy_backward(),
              note="(softmax - onehot) * dloss：重算行统计，不缓存前向概率"),
        # 随机
        _case("dropout_mask", _dropout_mask(),
              note="随机算子也要可对拍：seed 写在 attrs 里，参考后端与 CUDA 用同一套 SplitMix64"),
        _case("dropout_train", _dropout_train(),
              note="mask 与 mul 共用同一个 x：反向时 dropout 那条边不给梯度，梯度只走乘法"),
        # 精度边界
        _case("f64_input_relu", _float64_input(),
              note="只支持 f32 的后端必须显式拒绝，不能静默降精度"),
        # 真实组合图
        _case("transformer_block", _transformer_block(), rtol=1e-4, atol=1e-4,
              note="14 个算子的真实组合：reshape/transpose 的 4D 组合、fanout、批量 matmul"),
    ]


# --------------------------------------------------------------------------
# 检查实现
# --------------------------------------------------------------------------


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__


def _check_contract(backend, checks: List[CheckResult]) -> FrozenSet[str]:
    problems: List[str] = []
    caps: FrozenSet[str] = frozenset()
    refined = False

    for attr in ("name", "capabilities", "execute"):
        if not hasattr(backend, attr):
            problems.append(f"缺少 {attr}")

    if not problems:
        try:
            raw = backend.capabilities()
        except Exception as exc:  # 能力声明本身就崩了，没什么可继续的
            problems.append(f"capabilities() 失败: {type(exc).__name__}: {_first_line(exc)}")
        else:
            if not isinstance(raw, (set, frozenset)):
                problems.append(
                    f"capabilities() 返回 {type(raw).__name__}，契约要求集合"
                )
            else:
                # 保留形态明细（若有）：后面判定 refused 原因、以及"哪些算子
                # 只有名字"都要用到；它仍是 frozenset 子类，集合用法不变。
                caps = to_capabilities(raw)
                refined = isinstance(raw, BackendCapabilities)
                unknown = sorted(set(caps) - set(OPS))
                if unknown:
                    problems.append(f"声明了不存在的算子: {unknown}")

    if problems:
        checks.append(CheckResult("contract", False, "；".join(problems)))
    else:
        detail = f"{len(caps)} 个算子已声明"
        if refined:
            detail += f"；{len(caps.refined)} 个带形态明细"
            if caps.unrefined:
                detail += f"，{len(caps.unrefined)} 个仅算子名"
        else:
            detail += "（仅算子名，未细化 dtype/axis/反向/捕获）"
        checks.append(CheckResult("contract", True, detail))
    return caps


def _check_ir(cases: Sequence[Case], checks: List[CheckResult]) -> None:
    """与后端无关的两项：可移植性标注、以及图能否原样往返 JSON。"""
    portable_bad: List[str] = []
    roundtrip_bad: List[str] = []

    for case in cases:
        issues = verify_portable_only(case.graph)
        if issues:
            portable_bad.append(f"{case.name}: {issues[0]}")
        if to_dict(loads(dumps(case.graph))) != to_dict(case.graph):
            roundtrip_bad.append(case.name)

    checks.append(
        CheckResult(
            "portability",
            not portable_bad,
            "；".join(portable_bad) or f"{len(cases)} 张图全部只含 PORTABLE 节点",
        )
    )
    checks.append(
        CheckResult(
            "serialization",
            not roundtrip_bad,
            "；".join(roundtrip_bad) or f"{len(cases)} 张图 JSON 往返全等",
        )
    )


def _compare(got: Mapping[str, np.ndarray], expected: Mapping[str, np.ndarray], case: Case):
    """返回 (是否一致, 说明, 最大绝对误差)。"""
    max_error = 0.0
    for name, exp in expected.items():
        if name not in got:
            return False, f"缺少输出 {name}", max_error
        got_arr = np.asarray(got[name])
        exp_arr = np.asarray(exp)
        try:
            np.testing.assert_allclose(got_arr, exp_arr, rtol=case.rtol, atol=case.atol)
        except AssertionError:
            # 先在公共 dtype 上求差再取模：复数不能先转 float64，否则虚部被
            # 静默丢弃，误差会算成 0。bool 不能做减法，用不等元素个数。
            if got_arr.dtype == np.bool_ or exp_arr.dtype == np.bool_:
                peak = float(np.count_nonzero(got_arr != exp_arr))
            else:
                diff = np.abs(got_arr - exp_arr)
                peak = float(np.nanmax(diff)) if diff.size else 0.0
            return (
                False,
                f"输出 {name} 与参考语义不一致（最大绝对误差 {peak:.3e}，"
                f"允许 rtol={case.rtol:g} atol={case.atol:g}）",
                peak,
            )
        if got_arr.size:
            if got_arr.dtype == np.bool_ or exp_arr.dtype == np.bool_:
                max_error = max(max_error, float(np.count_nonzero(got_arr != exp_arr)))
            else:
                diff = np.abs(got_arr - exp_arr)
                max_error = max(max_error, float(np.nanmax(diff)))
    return True, "", max_error


def _run_case(
    backend,
    reference: ReferenceBackend,
    case: Case,
    caps: FrozenSet[str],
    check_determinism: bool,
    strictness: List[str],
) -> CaseResult:
    ops = tuple(sorted(case.ops))
    missing = tuple(sorted(case.ops - caps))
    # 名字都在声明里，但 dtype / axis / layout 可能被细化声明排除；
    # 这类拒绝不是"漏声明算子"，报错原因要说清是哪一维。
    gaps = capability_gaps(case.graph, caps) if not missing else []

    try:
        got = backend.execute(case.graph, case.inputs)
    except BackendCapabilityError as exc:
        if missing:
            return CaseResult(case.name, ops, REFUSED, f"未声明算子 {list(missing)}，按契约拒绝")
        if gaps:
            return CaseResult(case.name, ops, REFUSED, f"声明不支持该形态：{gaps[0][1]}")
        return CaseResult(case.name, ops, REFUSED, _first_line(exc))
    except Exception as exc:
        # 契约要求"不支持"必须走 BackendCapabilityError；其他异常让调用方无从判断
        detail = f"抛了 {type(exc).__name__}: {_first_line(exc)}，而不是 BackendCapabilityError"
        strictness.append(f"{case.name}: {detail}")
        return CaseResult(case.name, ops, FAILED, detail)

    if missing:
        # 能力声明里没有，却执行了。前端据此做拆分/降级决策，声明必须可信。
        detail = f"能力声明缺少 {list(missing)}，却执行成功"
        strictness.append(f"{case.name}: {detail}")
        return CaseResult(case.name, ops, FAILED, detail)

    try:
        expected = reference.execute(case.graph, case.inputs)
    except Exception as exc:
        return CaseResult(
            case.name, ops, FAILED,
            f"参考语义执行失败（套件缺陷）: {type(exc).__name__}: {_first_line(exc)}",
        )

    ok, why, max_error = _compare(got, expected, case)
    if not ok:
        return CaseResult(case.name, ops, FAILED, why, max_error)

    if check_determinism and case.deterministic:
        try:
            again = backend.execute(case.graph, case.inputs)
        except Exception as exc:
            detail = f"第二次执行失败: {type(exc).__name__}: {_first_line(exc)}"
            strictness.append(f"{case.name}: {detail}")
            return CaseResult(case.name, ops, FAILED, detail, max_error)
        for name in expected:
            if not np.array_equal(np.asarray(got[name]), np.asarray(again[name])):
                detail = f"同一输入两次执行结果不同（输出 {name}），不能充当一致性基准"
                return CaseResult(case.name, ops, FAILED, detail, max_error)

    return CaseResult(case.name, ops, EXECUTED, "", max_error)


def _check_coverage(
    cases: Sequence[Case],
    caps: FrozenSet[str],
    results: Sequence[CaseResult],
    checks: List[CheckResult],
    notes: List[str],
) -> None:
    relevant = set()
    for case in cases:
        if case.ops <= caps:
            relevant |= case.ops

    proven = set()
    for result in results:
        if result.status == EXECUTED:
            proven |= set(result.ops)

    unproven = sorted(relevant - proven)
    if unproven:
        checks.append(
            CheckResult(
                "coverage",
                False,
                f"声明支持但没有任何用例成功执行: {unproven}",
            )
        )
    else:
        checks.append(
            CheckResult(
                "coverage",
                True,
                f"{len(relevant)}/{len(relevant)} 个被声明的算子至少成功执行一次",
            )
        )

    untested = sorted(caps - relevant)
    if untested:
        notes.append(f"套件未覆盖这些声明算子，其能力无法被本套件证实: {untested}")


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def run_conformance(
    backend,
    cases: Optional[Sequence[Case]] = None,
    check_determinism: bool = True,
) -> Report:
    """对任意后端跑完整套件，返回结构化报告。

    ``cases`` 可替换成调用方自己的图集（例如从 5.8 的 JSON 文件加载），
    但默认图集保证覆盖全部已声明算子。
    """
    selected = list(cases) if cases is not None else default_cases()
    checks: List[CheckResult] = []
    notes: List[str] = []
    strictness: List[str] = []

    caps = _check_contract(backend, checks)
    _check_ir(selected, checks)

    reference = ReferenceBackend()
    results = [
        _run_case(backend, reference, case, caps, check_determinism, strictness)
        for case in selected
    ]

    checks.append(
        CheckResult(
            "explicit_failure",
            not strictness,
            "；".join(strictness)
            or f"{sum(1 for r in results if r.status == REFUSED)} 处拒绝全部为 BackendCapabilityError",
        )
    )
    _check_coverage(selected, caps, results, checks, notes)

    if check_determinism:
        exempt = [c.name for c in selected if not c.deterministic]
        if exempt:
            notes.append(
                "以下用例按设计不参与位级确定性检查（scatter-add 的累加顺序不可交换）："
                f"{exempt}"
            )

    name = getattr(backend, "name", type(backend).__name__)
    return Report(
        backend=str(name),
        capabilities=caps,
        checks=checks,
        cases=results,
        contracts=implemented_contracts(backend),
        notes=notes,
    )
