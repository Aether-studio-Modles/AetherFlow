"""C 风格方言：把 L2 内核打印成 CUDA C / HIP C。

两种方言的语法完全相同，差异只有两处，都由后端在构造方言时注入：

- **f16 的存取与原子加**：CUDA 用 PTX 的 ``cvt`` 约束，AMD 的 clang 不认，
  必须换成 ``_Float16`` + ``__builtin_bit_cast``；
- **prelude**：归约辅助函数两边共用，dtype 辅助函数各自随上面的实现走。

除此之外——参数声明、循环、索引、同步、共享内存——逐字相同。这正是 5.40 里
"内核模板逐字复用"可以被结构化的原因：差别不在模板，在方言。
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Optional, Sequence

from ..ir.types import DType
from .kernel import (
    ArrayDecl,
    Assign,
    AtomicAdd,
    AtomicMin,
    AugAssign,
    Axis,
    Bin,
    Call,
    Cast,
    Cond,
    Continue,
    Expr,
    For,
    If,
    Index,
    Kernel,
    Let,
    LetMulti,
    Load,
    Neg,
    Not,
    Narrow,
    Member,
    Num,
    Paren,
    Ptr,
    Return,
    Scalar,
    Shared,
    Store,
    Stmt,
    Sync,
    Tensor,
    Var,
    VecLoad,
    VecStore,
    Widen,
)

__all__ = ["CStyleDialect"]

_INDENT = "    "


class CStyleDialect:
    """一套 C 语法方言的**数据**部分：类型映射、辅助函数、prelude。

    打印逻辑在 :class:`_CPrinter` 里，与方言数据分离——加一种新硬件只需要
    构造一个 ``CStyleDialect``（必要时才覆盖打印器），不用改 L2 的节点。
    """

    def __init__(
        self,
        name: str,
        *,
        storage: Mapping[DType, str],
        load: Mapping[DType, str],
        store: Mapping[DType, str],
        helpers: Mapping[DType, str],
        atomic: Optional[Mapping[DType, str]] = None,
        reduce_helpers: str = "",
    ) -> None:
        self.name = name
        self.storage = dict(storage)
        self.load = dict(load)
        self.store = dict(store)
        self.helpers = dict(helpers)
        # 原子加是"可选能力"：只打印逐元素内核的方言不必填它，
        # 真要用到 AtomicAdd 时才要求（并给出明确错误）。
        self.atomic = dict(atomic or {})
        self.reduce_helpers = reduce_helpers

    def storage_of(self, dtype: DType) -> str:
        try:
            return self.storage[dtype]
        except KeyError:
            raise ValueError(
                f"{self.name} 方言不支持 dtype {dtype.value}；"
                f"已支持: {sorted(d.value for d in self.storage)}"
            ) from None

    def prelude(self, dtype: DType) -> str:
        try:
            helpers = self.helpers[dtype]
        except KeyError:
            raise ValueError(
                f"{self.name} 方言没有 {dtype.value} 的辅助函数"
            ) from None
        return (self.reduce_helpers + helpers).rstrip() + "\n\n"

    def atomic_of(self, dtype: DType) -> str:
        try:
            return self.atomic[dtype]
        except KeyError:
            raise ValueError(
                f"{self.name} 方言没有 {dtype.value} 的原子加实现；"
                f"已提供: {sorted(d.value for d in self.atomic)}"
            ) from None

    def printer(self) -> Callable[[Kernel, DType], str]:
        return _CPrinter(self).print_kernel


class _CPrinter:
    def __init__(self, dialect: CStyleDialect) -> None:
        self.d = dialect

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------

    def print_kernel(self, kernel: Kernel, dtype: DType) -> str:
        # 方言数据按 dtype 定一次，表达式与语句打印时直接取。
        self._dtype = dtype
        self._type = self.d.storage_of(dtype)
        try:
            self._load_name = self.d.load[dtype]
            self._store_name = self.d.store[dtype]
        except KeyError:
            raise ValueError(f"{self.d.name} 方言缺少 {dtype.value} 的存取函数") from None

        head = f'extern "C" __global__ void {kernel.name}('
        pad = " " * len(head)
        params = self._params(kernel.params, dtype)
        signature = (",\n" + pad).join(params)

        lines = [self.d.prelude(dtype) + kernel.helpers + head + signature + ") {"]
        lines.extend(self._stmts(kernel.body, 1))
        lines.append("}")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # 声明
    # ------------------------------------------------------------------

    def _params(self, params: Sequence[object], dtype: DType) -> list:
        t = self.d.storage_of(dtype)
        out = []
        for p in params:
            if isinstance(p, Tensor):
                const = "const " if p.read_only else ""
                pt = p.type_name or t
                out.append(f"{const}{pt}* __restrict__ {p.name}")
            elif isinstance(p, Scalar):
                out.append(f"{p.type_name} {p.name}")
            else:
                raise TypeError(f"未知参数类型: {type(p).__name__}")
        return out

    # ------------------------------------------------------------------
    # 语句
    # ------------------------------------------------------------------

    def _stmts(self, stmts: Iterable[Stmt], depth: int) -> list:
        lines = []
        for s in stmts:
            lines.extend(self._stmt(s, depth))
        return lines

    def _stmt(self, s: Stmt, depth: int) -> list:
        pad = _INDENT * depth
        if isinstance(s, Let):
            type_name = s.type_name
            if type_name is None:
                type_name = self._type
            else:
                # ``%T%`` 是存储类型的占位符：混合精度内核要声明"指向存储类型
                # 的指针"（f32 是 float*，f16 是 unsigned short*），而类型名
                # 本身由 dtype 决定，不能在这里写死。
                type_name = type_name.replace("%T%", self._type)
            return [f"{pad}{type_name} {s.name} = {self._expr(s.value)};"]
        if isinstance(s, LetMulti):
            decls = ", ".join(
                f"{name} = {self._expr(value)}" for name, value in s.items
            )
            return [f"{pad}{s.type_name} {decls};"]
        if isinstance(s, ArrayDecl):
            if s.values is None:
                return [f"{pad}{s.type_name} {s.name}[{s.count}];"]
            values = ", ".join(self._expr(v) for v in s.values)
            return [
                f"{pad}{s.type_name} {s.name}[{s.count}] = {{{values}}};"
            ]
        if isinstance(s, Ptr):
            const = "const " if s.const else ""
            return [
                f"{pad}{const}{self._type}* {s.name} = "
                f"{s.base} + {self._expr(s.offset)};"
            ]
        if isinstance(s, Shared):
            if s.extern:
                return [f"{pad}extern __shared__ {s.type_name} {s.name}[];"]
            return [f"{pad}__shared__ {s.type_name} {s.name}[{s.count}];"]
        if isinstance(s, For):
            incr = (
                f"{s.incr}{s.var}"
                if s.incr
                else f"{s.var} += {self._expr(s.step)}"
            )
            if s.start is None:
                head = (
                    f"{pad}for (; {s.var} {s.cmp} {self._expr(s.stop)}; "
                    f"{incr}) {{"
                )
            else:
                head = (
                    f"{pad}for ({s.type_name} {s.var} = {self._expr(s.start)}; "
                    f"{s.var} {s.cmp} {self._expr(s.stop)}; "
                    f"{incr}) {{"
                )
            return [head] + self._stmts(s.body, depth + 1) + [f"{pad}}}"]
        if isinstance(s, If):
            lines = [f"{pad}if ({self._expr(s.cond)}) {{"]
            lines.extend(self._stmts(s.body, depth + 1))
            if s.orelse:
                lines.append(f"{pad}}} else {{")
                lines.extend(self._stmts(s.orelse, depth + 1))
            lines.append(f"{pad}}}")
            return lines
        if isinstance(s, Assign):
            return [f"{pad}{self._expr(s.target)} = {self._expr(s.value)};"]
        if isinstance(s, AugAssign):
            return [
                f"{pad}{self._expr(s.target)} {s.op}= {self._expr(s.value)};"
            ]
        if isinstance(s, Store):
            value = self._expr(s.value)
            if not s.raw:
                value = f"{self._store_fn()}({value})"
            return [f"{pad}{s.buf}[{self._expr(s.index)}] = {value};"]
        if isinstance(s, VecStore):
            values = ", ".join(self._expr(v) for v in s.values)
            return [
                f"{pad}{self._vec_store_fn(s.width)}("
                f"{s.buf}, {self._expr(s.index)}, {values});"
            ]
        if isinstance(s, AtomicAdd):
            return [
                f"{pad}{self._atomic_fn()}(&{s.buf}[{self._expr(s.index)}], "
                f"{self._expr(s.value)});"
            ]
        if isinstance(s, AtomicMin):
            return [
                f"{pad}atomicMin(&{s.buf}[{self._expr(s.index)}], "
                f"{self._expr(s.value)});"
            ]
        if isinstance(s, Continue):
            return [f"{pad}continue;"]
        if isinstance(s, Sync):
            return [f"{pad}__syncthreads();"]
        if isinstance(s, Return):
            return [f"{pad}return;"]
        raise TypeError(f"未知语句类型: {type(s).__name__}")

    # ------------------------------------------------------------------
    # 表达式
    # ------------------------------------------------------------------

    def _load_fn(self) -> str:
        # dtype 在 print_kernel 里确定，缓存到实例上供表达式打印用。
        return self._load_name

    def _store_fn(self) -> str:
        return self._store_name

    def _atomic_fn(self) -> str:
        # 只有真的打印 AtomicAdd 时才检查方言是否提供了原子加。
        return self.d.atomic_of(self._dtype)

    def _vec_load_fn(self, width: int) -> str:
        """向量读取的辅助函数名。只对 f32 + 4 通道有实现。

        不把名字放进方言数据，因为"哪种 dtype 能向量化"目前是**实现边界**而
        不是方言差异：f16 的 16 位标量存取要靠 PTX/内建位转换，攒成 8 字节
        向量需要另一套写法，还没做。用明确的错误说明边界，好过让打印器生成
        一个不存在的函数名，把问题推迟到 NVRTC 编译期。
        """
        if width != 4 or self._dtype is not DType.F32:
            raise ValueError(
                f"L2 的向量存取只实现了 f32 的 4 通道，收到 "
                f"{self._dtype.value} x{width}"
            )
        return "af_ld4_f"

    def _vec_store_fn(self, width: int) -> str:
        self._vec_load_fn(width)  # 同样的边界检查
        return "af_st4_f"

    def _expr(self, e: Expr) -> str:
        if isinstance(e, Var):
            return e.name
        if isinstance(e, Num):
            return e.text
        if isinstance(e, Bin):
            return f"{self._expr(e.lhs)} {e.op} {self._expr(e.rhs)}"
        if isinstance(e, Neg):
            return f"-{self._expr(e.inner)}"
        if isinstance(e, Not):
            return f"!{self._expr(e.inner)}"
        if isinstance(e, Paren):
            return f"({self._expr(e.inner)})"
        if isinstance(e, Cond):
            return (
                f"{self._expr(e.cond)} ? {self._expr(e.then)} "
                f": {self._expr(e.other)}"
            )
        if isinstance(e, Cast):
            return f"({e.type_name}){self._expr(e.inner)}"
        if isinstance(e, Call):
            args = ", ".join(self._expr(a) for a in e.args)
            return f"{e.fn}({args})"
        if isinstance(e, Index):
            return f"{e.buf}[{self._expr(e.index)}]"
        if isinstance(e, Axis):
            return _AXIS[e.kind].format(axis=e.axis)
        if isinstance(e, Load):
            return f"{self._load_fn()}({e.buf}[{self._expr(e.index)}])"
        if isinstance(e, Widen):
            return f"{self._load_fn()}({self._expr(e.inner)})"
        if isinstance(e, Narrow):
            return f"{self._store_fn()}({self._expr(e.inner)})"
        if isinstance(e, Member):
            return f"{self._expr(e.inner)}.{e.field}"
        if isinstance(e, VecLoad):
            return (
                f"{self._vec_load_fn(e.width)}({e.buf}, {self._expr(e.index)})"
            )
        raise TypeError(f"未知表达式类型: {type(e).__name__}")


_AXIS = {
    "tid": "threadIdx.{axis}",
    "bid": "blockIdx.{axis}",
    "bdim": "blockDim.{axis}",
    "gdim": "gridDim.{axis}",
}
