"""L2 内核 IR：把"算子怎么算"从"用谁的方言写"里拆出来。

L1（``ir`` 包的 Graph/Node）描述**算什么**；本模块描述**怎么算**：线程索引、
循环、共享内存、标量表达式、读写。它仍然不认识任何厂商——``threadIdx`` 是并行
执行模型的通用概念，打印成什么名字由方言决定。

这一层刻意停在**语法级**：一条语句对应目标语言的一条语句。它带来的是可检视、
可双路打印、可做局部改写；不带来自动分块 / 向量化——那需要索引分析与布局决策，
属于语义级 L2，是下一阶段的事（见 DESIGN 5.42 的"没做的"）。

同一份 L2 描述配合不同 dtype 与不同方言，应该产生语义等价的两份源码：

    kernel = ew_binary("add")
    emit(kernel, CUDA_DIALECT, DType.F32)   # -> CUDA C
    emit(kernel, HIP_DIALECT, DType.F16)    # -> HIP C（f16 存取换成 _Float16）

括号在 L2 里是显式的（``Paren`` 节点）：源码的括号结构属于语义的一部分，
打印器不替作者推断优先级，这样"生成的源码"与"手写模板"可以逐 token 对齐。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from ..ir.types import DType

__all__ = [
    "Expr",
    "Var",
    "Num",
    "Bin",
    "Neg",
    "Not",
    "Paren",
    "Cond",
    "Cast",
    "Call",
    "Index",
    "Axis",
    "Load",
    "Widen",
    "Narrow",
    "Member",
    "VecLoad",
    "VecStore",
    "Stmt",
    "Let",
    "LetMulti",
    "ArrayDecl",
    "Ptr",
    "Shared",
    "For",
    "If",
    "Assign",
    "AugAssign",
    "Store",
    "AtomicAdd",
    "AtomicMin",
    "Continue",
    "Sync",
    "Return",
    "Tensor",
    "Scalar",
    "Kernel",
]


# ---------------------------------------------------------------------------
# 表达式
# ---------------------------------------------------------------------------


class Expr:
    """表达式基类（仅用于类型标注）。"""


@dataclass(frozen=True)
class Var(Expr):
    """局部变量 / 内核参数名。"""

    name: str


@dataclass(frozen=True)
class Num(Expr):
    """字面量，原样打印（``0.f`` / ``1024``）。"""

    text: str


@dataclass(frozen=True)
class Bin(Expr):
    """二元运算。打印为 ``lhs op rhs``，不加括号——需要括号时用 ``Paren``。"""

    op: str
    lhs: Expr
    rhs: Expr


@dataclass(frozen=True)
class Neg(Expr):
    """一元负号：``-x``。"""

    inner: Expr


@dataclass(frozen=True)
class Not(Expr):
    """逻辑非：``!x``。

    和 ``Neg`` 分开是必要的：``!has_mask`` 与 ``-x`` 在源码里是两个 token，
    而 L2 的判据是逐 token 相等，不能靠打印器猜类型。
    """

    inner: Expr


@dataclass(frozen=True)
class Paren(Expr):
    inner: Expr


@dataclass(frozen=True)
class Cond(Expr):
    """三元条件：``cond ? then : other``。"""

    cond: Expr
    then: Expr
    other: Expr


@dataclass(frozen=True)
class Cast(Expr):
    """C 风格显式转换，例如 ``(long long)blockIdx.x``。"""

    type_name: str
    inner: Expr


@dataclass(frozen=True)
class Call(Expr):
    fn: str
    args: Tuple[Expr, ...] = ()


@dataclass(frozen=True)
class Index(Expr):
    """``buf[index]``。buf 可以是数组、指针或参数名。"""

    buf: str
    index: Expr


@dataclass(frozen=True)
class Axis(Expr):
    """并行执行模型的内置轴。

    ``kind`` 取值：``tid`` / ``bid`` / ``bdim`` / ``gdim``；``axis`` 是 ``x``/``y``/``z``。
    方言负责把它打印成 ``threadIdx.x`` 之类的名字。
    """

    kind: str
    axis: str = "x"


@dataclass(frozen=True)
class Load(Expr):
    """按存储类型读成 float：``af_ld_f(x[i])``。dtype 由 ``emit`` 统一决定。"""

    buf: str
    index: Expr


@dataclass(frozen=True)
class Widen(Expr):
    """把存储类型的标量读成计算宽度：``af_ld_h(xv)``。

    ``Load`` 是它在"带索引的缓冲"上的特化；当标量已经是一个局部变量
    （例如融合残差里的 ``xv``）时用这个节点。
    """

    inner: Expr


@dataclass(frozen=True)
class Narrow(Expr):
    """把计算宽度舍入回存储类型：``af_st_h(sum)``。

    这一步不是可选的精度修饰，而是语义的一部分：f16 下 ``x + res`` 必须
    先舍入回 f16，再参与后续计算，才与参考实现（numpy 用两个 f16 相加）一致。
    """

    inner: Expr


@dataclass(frozen=True)
class Member(Expr):
    """向量局部量的成员：``v.x``。

    这是向量化唯一的"新语法"：``float4`` 的四个分量没有下标形式，只能按
    名字取。把它单独做成一类节点而不是让打印器猜 ``Index`` 的含义，是因为
    ``float4[0]`` 在 C 里根本不合法——语法级的判据就要求节点与 token 对齐。
    """

    inner: Expr
    field: str


@dataclass(frozen=True)
class VecLoad(Expr):
    """一次 128 位读取：``af_ld4_f(a, i)``，得到 ``float4``。

    ``index`` 以**元素**为单位（不是字节），与 ``Load`` 一致；地址对齐由
    决定 ``width`` 的那一层负责（转成向量指针需要 16 字节对齐）。
    """

    buf: str
    index: Expr
    width: int = 4


# ---------------------------------------------------------------------------
# 语句
# ---------------------------------------------------------------------------


class Stmt:
    """语句基类（仅用于类型标注）。"""


@dataclass(frozen=True)
class Let(Stmt):
    """``type name = value;``

    ``type_name is None`` 表示用方言的存储类型（模板里的 ``%T%``）——融合
    残差的 LayerNorm 需要把 ``x + res`` 先落到存储类型上。

    ``type_name`` 里可以写 ``%T%`` 占位符，打印时替换成存储类型：混合精度
    内核要声明"指向存储类型的指针"（``const %T%* mrow``），f32 下是
    ``const float*``、f16 下是 ``const unsigned short*``。
    """

    type_name: Optional[str]
    name: str
    value: Expr


@dataclass(frozen=True)
class LetMulti(Stmt):
    """一条语句声明多个变量：``type a = x, b = 0;``"""

    type_name: str
    items: Tuple[Tuple[str, Expr], ...]


@dataclass(frozen=True)
class ArrayDecl(Stmt):
    """``type name[count] = {v0, v1, ...};``——定长数组与初始化列表。

    ``values is None`` 时只声明不初始化（``float acc[8];``）：累加器数组
    要在循环里按下标清零，初始化列表反而写不出来。
    """

    type_name: str
    name: str
    count: int
    values: Optional[Tuple[Expr, ...]] = ()


@dataclass(frozen=True)
class Ptr(Stmt):
    """``const T* name = base + offset;``——指向某块缓冲的视图。"""

    name: str
    base: str
    offset: Expr
    const: bool = True


@dataclass(frozen=True)
class Shared(Stmt):
    """共享内存声明。

    ``count`` 有值时是静态分配（``__shared__ float smem[1024];``）；
    ``extern=True`` 且 ``count is None`` 时是启动时按字节数传入的动态分配
    （``extern __shared__ float sw[];``）——两者是同一个地址空间的两种来源，
    共用节点可以少一类语句。
    """

    type_name: str
    name: str
    count: Optional[int] = None
    extern: bool = False


@dataclass(frozen=True)
class For(Stmt):
    """``for`` 循环。

    ``start is None`` 时打印 ``for (; stop; var += step)``——变量已在前面声明，
    这是"网格步进"的写法；否则打印 ``for (type_name var = start; ...)``。

    ``cmp`` 是循环条件，``incr`` 非空时步进打印成 ``++var`` / ``--var``
    （倒序循环用），否则打印 ``var += step``。
    """

    var: str
    start: Optional[Expr]
    stop: Expr
    step: Expr
    body: Tuple[Stmt, ...]
    cmp: str = "<"
    incr: str = ""
    type_name: str = "long long"


@dataclass(frozen=True)
class If(Stmt):
    cond: Expr
    body: Tuple[Stmt, ...]
    orelse: Tuple[Stmt, ...] = ()


@dataclass(frozen=True)
class Assign(Stmt):
    target: Expr
    value: Expr


@dataclass(frozen=True)
class AugAssign(Stmt):
    target: Expr
    op: str
    value: Expr


@dataclass(frozen=True)
class Store(Stmt):
    """``buf[index] = af_st_f(value);``

    ``raw`` 为真时打印成裸赋值 ``buf[index] = value;``：纯搬运（拼接、补零）
    不是计算，值本来就已是存储类型，套一层 ``af_st`` 只会多一次无意义的转换。
    """

    buf: str
    index: Expr
    value: Expr
    raw: bool = False


@dataclass(frozen=True)
class VecStore(Stmt):
    """一次 128 位写入：``af_st4_f(buf, i, x, y, z, w)``。

    ``values`` 是四条通道各自的值，而不是一个向量表达式：向量化后的算术仍
    逐通道写成标量表达式（L2 没有 float4 的四则运算节点），所以写回时拿到的
    就是四个独立的标量。把它们交给一个辅助函数，而不是拼一个 ``make_float4``，
    是为了让同一条 L2 描述在 CUDA 与 HIP 上都能打印。
    """

    buf: str
    index: Expr
    values: Tuple[Expr, ...]
    width: int = 4


@dataclass(frozen=True)
class AtomicAdd(Stmt):
    """``af_add_f(&buf[index], value);``——把 float 累加进存储类型。

    重叠窗口会命中同一个输入单元，普通写入只会留下最后一个窗口的贡献，
    所以池化的反向必须用它。
    """

    buf: str
    index: Expr
    value: Expr


@dataclass(frozen=True)
class AtomicMin(Stmt):
    """``atomicMin(&buf[index], value);``——多个线程竞争同一个最小下标。

    与 ``AtomicAdd`` 分开：前者把浮点累加进存储类型（dtype 相关），后者是
    整数索引的最小值竞争（dtype 无关）。打印器不能靠值的类型猜，所以两个节点。
    """

    buf: str
    index: Expr
    value: Expr


@dataclass(frozen=True)
class Continue(Stmt):
    """``continue;``——跳过当前迭代的剩余部分（越界采样点的写法）。"""


@dataclass(frozen=True)
class Sync(Stmt):
    """块内同步（``__syncthreads()``）。"""


@dataclass(frozen=True)
class Return(Stmt):
    """裸 ``return;``（用于提前退出某个 block）。"""


# ---------------------------------------------------------------------------
# 内核
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tensor:
    """张量参数。``read_only`` 决定是否加 ``const``。

    ``type_name`` 用于"固定用 float 存储"的辅助张量（例如 softmax 的
    scale），此时不跟随 dtype 的存储类型。
    """

    name: str
    read_only: bool = True
    type_name: Optional[str] = None


@dataclass(frozen=True)
class Scalar:
    """标量参数（``long long n``）。"""

    type_name: str
    name: str


@dataclass(frozen=True)
class Kernel:
    name: str
    params: Tuple[Union[Tensor, Scalar], ...]
    body: Tuple[Stmt, ...]
    # 该内核私有的设备函数（例如 dropout 的 splitmix64）。放在方言 prelude
    # 之后、内核签名之前，只随用它的内核出现，不污染其它算子的源码。
    helpers: str = ""


def emit(kernel: Kernel, dialect, dtype: DType) -> str:
    """把 L2 内核打印成目标方言的源码（含方言自己的 prelude）。"""
    printer = dialect.printer()
    return printer(kernel, dtype)
