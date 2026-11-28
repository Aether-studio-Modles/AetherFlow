"""IR 算子表。

每个算子由三部分组成，缺一不可：

1. ``infer``：类型推导。纯符号运算，必须能在符号维存在时工作。
2. ``reference``：参考语义。这是整个框架的"唯一真值"。
   任何后端（CUDA、其他硬件）的输出都必须与此语义等价。
   参考实现刻意只用 numpy 的标量/逐元素语义，不绑定任何加速库，
   也不约束累加顺序——位级一致性由后端自己负责，语义一致性以此为准。
3. ``portability``：可移植性标注。这是防止 IR 退化成"某一个后端调度器"
   的强制手段，verifier 会据此检查图是否可被目标后端承接。
4. ``effects``：副作用声明。空集表示纯函数，CSE / DCE / 常量折叠 / 激活重算
   都可以按"同样输入必定同样输出、删掉不留痕"来推理；一旦声明了 RNG /
   STATE / IO，这些优化必须让路（见 ``Effect``）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from .types import (
    DType,
    Dim,
    TensorType,
    _as_shape,
    dim_static,
    dim_symbols,
    sym_add,
    sym_binary,
)


class NoGradientError(ValueError):
    """这张图在该位置没有定义梯度。

    刻意不是 ``NotImplementedError``：它描述的是"数学上这里没定义/没给规则"，
    而不是"这个后端还没实现"，换一个后端也不会变好。反向规则缺失与
    算子缺失是两类问题，混用异常类型会把排查引向错误方向。
    """


class Portability(str, Enum):
    """节点可移植性。

    PORTABLE  通用数学语义，任何后端都必须支持或能拆解实现。
    LIBRARY   绑定某个外部库的能力与约束（如 cuDNN 卷积），仅该后端承接。
    VENDOR    厂商专有硬件能力，不可移植，禁止进入共享优化路径。
    """

    PORTABLE = "portable"
    LIBRARY = "library"
    VENDOR = "vendor"


class Effect(str, Enum):
    """算子的可观察效果。

    三者对优化的约束不同，不能压成一个"有副作用/没副作用"的布尔量：

    - ``RNG``：采样。输出由 seed / key 完全决定，但语义是"每次重新采样"。
      因此它**可以**在激活重算里重放（同一 key 复现同一位模式），但**不能**
      被常量折叠（噪声被固化）、也**不能**被 CSE 合并（两次采样被并成一次）。
    - ``STATE``：读写图外可变状态（内存池、缓存、文件句柄）。重放会推进或
      改写状态，所以不能进激活重算；删除未使用的节点也不是"清理"，而是改变
      行为。
    - ``IO``：对外可见的输入输出（打印、日志、主机传输）。同样不能被删或重放。

    空集表示纯函数——这正是 CSE / DCE / 折叠 / 重算默认成立的前提。
    """

    RNG = "rng"
    STATE = "state"
    IO = "io"


#: 不能在重算段里重复执行的效果。``RNG`` 不在其中：显式 key 下重放可复现。
UNREPLAYABLE_EFFECTS: FrozenSet[Effect] = frozenset({Effect.STATE, Effect.IO})


InferFn = Callable[[Tuple[TensorType, ...], dict], TensorType]
RefFn = Callable[[List[np.ndarray], dict], np.ndarray]
# 反向规则。签名 ``(ctx) -> 每个输入的梯度``，``ctx`` 见 ``ir/grad.py`` 的 GradCtx。
# 返回 None 的元素表示"该输入不接收梯度"（例如 softmax 的 scale、步进函数）。
VjpFn = Callable[[Any], Sequence[Optional[Any]]]


@dataclass(frozen=True)
class Op:
    name: str
    infer: InferFn
    reference: RefFn
    portability: Portability = Portability.PORTABLE
    attrs_schema: Tuple[str, ...] = field(default_factory=tuple)
    doc: str = ""
    vjp: Optional[VjpFn] = None
    """反向规则；None 表示这张图的这个算子不支持求导。

    "None" 与"规则里返回 None"是两种语义：前者在 ``backward`` 时**报错**，
    后者表示该输入确实不接收梯度。前者防止把"忘了写规则"静默当成"梯度为零"。
    """
    nullary: bool = False
    """是否允许零输入节点。目前只有 ``constant`` 需要。"""

    effects: FrozenSet[Effect] = field(default_factory=frozenset)
    """副作用集合。空集 = 纯函数，可被 CSE / DCE / 常量折叠 / 激活重算处理。"""

    def __post_init__(self) -> None:
        # 插件可能传 set / list / 裸字符串；统一成冻结集合，并拒绝拼错的名字。
        if not isinstance(self.effects, frozenset):
            object.__setattr__(self, "effects", frozenset(self.effects))
        bad = [e for e in self.effects if not isinstance(e, Effect)]
        if bad:
            raise TypeError(f"算子的 effects 必须是 Effect，得到 {bad}")

    @property
    def pure(self) -> bool:
        """没有任何可观察效果。优化 pass 的默认推理前提。"""
        return not self.effects

    def check_attr_names(self, attrs: dict) -> None:
        unknown = set(attrs) - set(self.attrs_schema)
        if unknown:
            raise TypeError(f"算子 {self.name} 不接受属性: {sorted(unknown)}")


OPS: Dict[str, Op] = {}


def register(op: Op) -> Op:
    if op.name in OPS:
        raise ValueError(f"算子重复注册: {op.name}")
    OPS[op.name] = op
    return op


def get_op(name: str) -> Op:
    op = OPS.get(name)
    if op is None:
        raise KeyError(f"未知算子: {name}")
    return op


# --------------------------------------------------------------------------
# 形状工具
# --------------------------------------------------------------------------


def _broadcast_shape(a: Sequence[Dim], b: Sequence[Dim]) -> Tuple[Dim, ...]:
    """numpy 风格广播，但对符号维做保守处理。

    - 相同 → 保留
    - 一方为 1 → 取另一方
    - 两边都是静态且不等 → 报错
    - 其余（符号 vs 静态非 1、两个不同符号）→ 构图期拒绝

    最后一条是刻意的：符号维只知道自己"可能是 1"，不知道具体值，于是结果的
    那一维可能是静态值、也可能是符号，IR 里没有能表达这种选择的广播维表达式。
    此前取符号一方，等于把一个未必成立的假设写进类型——``(B, 2) + (3, 2)`` 在
    B=1 时本应是 ``(3, 2)``，运行期就会撞上"实际形状与 IR 推导不一致"。与其
    等到执行才报形状不符，不如在构图期就说清楚"这一维无法表示"。
    """
    out: List[Dim] = []
    for x, y in zip(reversed(a), reversed(b)):
        if x == y:
            out.append(x)
        elif x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        elif isinstance(x, int) and isinstance(y, int):
            raise ValueError(f"无法广播的维度: {x} 与 {y}")
        else:
            raise ValueError(
                f"符号维 {x} 与 {y} 的广播结果无法在 IR 中表示："
                f"其中任一方都可能取 1。请把该维改成静态形状，"
                f"或先显式 reshape 到已知形状再相加"
            )
    longer = a if len(a) >= len(b) else b
    out.extend(reversed(longer[: abs(len(a) - len(b))]))
    return tuple(reversed(out))


def _norm_axes(axis, rank: int) -> Tuple[int, ...]:
    if axis is None:
        # 显式拒绝，而不是让 tuple(None) 抛一句 "'NoneType' object is not
        # iterable"：全归约在 IR 里没有独立的语义位，写清楚该写什么。
        raise ValueError(
            "axis=None（全归约）不受支持；"
            f"请显式列出所有轴，例如 axis=tuple(range({rank}))"
        )
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    norm = tuple(a + rank if a < 0 else a for a in axes)
    for a in norm:
        if a < 0 or a >= rank:
            raise ValueError(f"轴越界: {a}，秩为 {rank}")
    if len(set(norm)) != len(norm):
        raise ValueError(f"重复的轴: {norm}")
    return norm


def _reject_bool(dtype: DType, where: str) -> None:
    """布尔张量只参与判定：比较、逻辑、``where`` 与显式 ``cast``，不做算术。

    设备端的 bool 容器是 1 字节，``add`` 之类的内核会把它当数值算，结果虽然
    能跑但语义上无意义（``真 + 真 = 2`` 存回 bool 又变成 1）。这里显式拒绝，
    让用户在图上插一个 ``cast`` 把意图写清楚。
    """
    if dtype is DType.BOOL:
        raise TypeError(
            f"{where}不接受 bool 输入：布尔张量只能参与比较与 where，"
            "要参与算术请先显式 cast 成数值类型"
        )


def _elementwise_binary_infer(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    a, b = inputs
    if a.dtype != b.dtype:
        raise TypeError(f"二元算子要求两侧 dtype 一致，得到 {a.dtype.value} 与 {b.dtype.value}；请显式插入 cast")
    _reject_bool(a.dtype, "二元算术算子")
    return TensorType(a.dtype, _broadcast_shape(a.shape, b.shape))


def _elementwise_unary_infer(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    _reject_bool(inputs[0].dtype, "一元算术算子")
    return inputs[0]


def _compare_infer(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    """比较：两侧同 dtype（布尔也可以互比），输出 bool，形状按广播。

    输出类型钉死 BOOL 而不是跟随输入：后端的 bool 存储只有 1 字节，与 f32
    不同宽，跟随输入会让"比较结果"这块缓冲被当成浮点写。
    """
    a, b = inputs
    if a.dtype != b.dtype:
        raise TypeError(
            f"比较算子要求两侧 dtype 一致，得到 {a.dtype.value} 与 {b.dtype.value}；"
            "请显式插入 cast"
        )
    return TensorType(DType.BOOL, _broadcast_shape(a.shape, b.shape))


def _compare_ref(np_fn):
    """比较的参考实现：结果为 numpy bool，与 BOOL 的承载类型一致。"""

    def ref(args: List[np.ndarray], attrs: dict) -> np.ndarray:
        with np.errstate(all="ignore"):
            return np.asarray(np_fn(args[0], args[1]), dtype=np.bool_)

    return ref


def _where_infer(inputs, attrs):
    """``where(cond, a, b)``：cond 必须是 bool，a / b 同 dtype，三者广播。

    bool 的 ``a`` / ``b`` 也允许（掩码之间选择），但浮点条件必须由上游
    显式比较产生——这样可以保证"0.5 算真吗"这类歧义不会进入 IR。
    """
    cond, a, b = inputs
    if cond.dtype is not DType.BOOL:
        raise TypeError(
            f"where 的 cond 必须是 bool，得到 {cond.dtype.value}；"
            "浮点条件请先插入比较算子"
        )
    if a.dtype != b.dtype:
        raise TypeError(
            f"where 的两侧 dtype 必须一致，得到 {a.dtype.value} 与 {b.dtype.value}；"
            "请显式插入 cast"
        )
    return TensorType(
        a.dtype, _broadcast_shape(_broadcast_shape(cond.shape, a.shape), b.shape)
    )


def _where_ref(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    cond, a, b = args
    with np.errstate(all="ignore"):
        return np.where(np.asarray(cond, dtype=np.bool_), a, b)


def _binary_ref(np_fn):
    """二元逐元素：溢出 / 无效值按 IEEE 返回，且不在宿主上打警告。

    f16 的 ``add`` 溢出成 ``inf`` 是位宽决定的语义，不是错误；numpy 默认会
    往 stderr 打一条 RuntimeWarning。这条口径与 ``div``、``exp/log/sqrt``
    一致——同一张图在参考后端与设备后端上不该有"日志差异"。
    """

    def ref(args: List[np.ndarray], attrs: dict) -> np.ndarray:
        with np.errstate(all="ignore"):
            return np_fn(args[0], args[1])

    return ref


# --------------------------------------------------------------------------
# 逐元素算子
# --------------------------------------------------------------------------


for _name, _np_fn in (
    ("add", np.add),
    ("sub", np.subtract),
    ("mul", np.multiply),
):
    register(
        Op(
            name=_name,
            infer=_elementwise_binary_infer,
            reference=_binary_ref(_np_fn),
            doc=f"逐元素 {_name}，支持 numpy 广播",
        )
    )


# 比较算子：唯一产出 BOOL 的一族。没有 vjp——比较在数学上不可导，注册一条
# 规则会让"对 mask 求梯度"这件事看起来合法。
for _name, _np_fn in (
    ("eq", np.equal),
    ("ne", np.not_equal),
    ("lt", np.less),
    ("le", np.less_equal),
    ("gt", np.greater),
    ("ge", np.greater_equal),
):
    register(
        Op(
            name=_name,
            infer=_compare_infer,
            reference=_compare_ref(_np_fn),
            doc=f"逐元素比较 {_name}，输出 bool，支持 numpy 广播",
        )
    )


register(
    Op(
        name="where",
        infer=_where_infer,
        reference=_where_ref,
        doc="按 bool 条件逐元素选择，支持三方 numpy 广播",
    )
)


def _logical_binary_infer(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    """逻辑与 / 或：两侧都必须是 bool，输出 bool，形状按广播。

    与 ``eq`` 等比较算子不同，这里不接受浮点输入——"0.5 算真吗"的口径一旦
    放开就会在两端产生分歧。浮点掩码需要先显式比较。
    """
    a, b = inputs
    for t in (a, b):
        if t.dtype is not DType.BOOL:
            raise TypeError(
                f"逻辑算子只接受 bool 输入，得到 {t.dtype.value}；"
                "浮点请先用比较算子（如 ne(x, 0)）落成 bool"
            )
    return TensorType(DType.BOOL, _broadcast_shape(a.shape, b.shape))


def _logical_binary_ref(np_fn):
    def ref(args: List[np.ndarray], attrs: dict) -> np.ndarray:
        with np.errstate(all="ignore"):
            return np.asarray(
                np_fn(
                    np.asarray(args[0], dtype=np.bool_),
                    np.asarray(args[1], dtype=np.bool_),
                ),
                dtype=np.bool_,
            )

    return ref


# 逻辑算子：没有 vjp，理由与比较算子相同——布尔代数不可导。
for _name, _np_fn in (
    ("logical_and", np.logical_and),
    ("logical_or", np.logical_or),
):
    register(
        Op(
            name=_name,
            infer=_logical_binary_infer,
            reference=_logical_binary_ref(_np_fn),
            doc=f"逐元素逻辑 {_name}，两侧 bool、输出 bool，支持 numpy 广播",
        )
    )


def _infer_logical_not(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    t = inputs[0]
    if t.dtype is not DType.BOOL:
        raise TypeError(
            f"logical_not 只接受 bool 输入，得到 {t.dtype.value}；"
            "浮点请先用比较算子落成 bool"
        )
    return t


def _ref_logical_not(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    return np.asarray(np.logical_not(np.asarray(args[0], dtype=np.bool_)), dtype=np.bool_)


register(
    Op(
        name="logical_not",
        infer=_infer_logical_not,
        reference=_ref_logical_not,
        doc="逐元素逻辑非，输入输出都是 bool",
    )
)


def _div_ref(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """``div``：浮点按 IEEE，整数分母为 0 必须显式报错。

    浮点除零是**定义过的**（±inf / nan），这里不拦；整数没有这个语义——numpy
    会先升到 float64 再落回整数，``inf`` 转 i32 得到 INT_MIN，产物是垃圾值而
    警告在默认配置下被忽略。整数除法本身按截断定义（与参考执行器的 final cast
    一致），只有分母为 0 这件事无解，所以当场拒绝。
    """
    a, b = args
    if a.dtype.kind not in "fc" and np.any(b == 0):
        raise ValueError(
            f"div 的整数分母不能为 0（收到 dtype={a.dtype}）；"
            f"需要 IEEE 的 inf/nan 语义请先 cast 到浮点"
        )
    # 浮点除零是正常语义，不该顺带在宿主上打 numpy 警告
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(a, b)


register(
    Op(
        name="div",
        infer=_elementwise_binary_infer,
        reference=_div_ref,
        doc="逐元素 div，支持 numpy 广播；浮点除零按 IEEE，整数分母为 0 显式拒绝",
    )
)


def _infer_clamp(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    """三参截断：``clamp(x, min=..., max=...)``，至少给出其中一侧。

    上下界刻意放在**属性**里而不是输入里：它们在前端是 Python 常量
    （``logA.clamp(-4, 0)``、``cum_in.clamp(min=-80)``），做成输入会白白多
    两条图边与一次设备驻留。真要动态上下界，用 ``maximum`` / ``minimum``
    语义自己拼（IR 没有这两个算子，用 ``step`` 的门也可以）。
    """
    if len(inputs) != 1:
        raise ValueError(f"clamp 只接受 1 个输入，得到 {len(inputs)} 个；上下界由属性给出")
    if not inputs[0].dtype.is_float:
        raise TypeError(f"clamp 只支持浮点 dtype，得到 {inputs[0].dtype.value}")
    lo, hi = attrs.get("min"), attrs.get("max")
    if lo is None and hi is None:
        raise ValueError("clamp 至少要给出 min 或 max 之一")
    if lo is not None and hi is not None and float(lo) > float(hi):
        raise ValueError(f"clamp 的 min 不能大于 max，得到 {lo} > {hi}")
    return inputs[0]


def _clamp_ref(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x = args[0]
    # 比较宽度统一到 f32：f16 输入先升宽比较、再舍回存储类型，与设备端
    # "读成 float、比较、写回"逐位一致。直接在 f16 里比较会让"边界常量恰好
    # 落在两个 f16 之间"的情形与设备端分叉。
    cmp = np.float32 if x.dtype == np.float16 else x.dtype
    out = np.clip(x.astype(cmp, copy=False), attrs.get("min"), attrs.get("max"))
    return out.astype(x.dtype, copy=False)


register(
    Op(
        name="clamp",
        infer=_infer_clamp,
        reference=_clamp_ref,
        attrs_schema=("min", "max"),
        doc="逐元素截断到 [min, max]：至少给出 min / max 之一。NaN 原样透传"
        "（比较为假），边界点属于保留区间；反向是区间内的 0/1 门",
    )
)


def _infer_mod(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    """整数取模，语义与 Python / ``torch.remainder`` 一致（向下取整）。

    为什么单独做算子而不是复用 ``div``：浮点 ``div`` 的语义是 IEEE 除法，
    拿不到整数余数；而 C 的 ``%`` 是**截断取余**，与 Python 在负被除数上
    差一个模数。Engram 的哈希（``(slot*131 + g) % n_slots``）依赖后者，
    这条语义必须显式写出来，不能靠后端各自解释。
    """
    a, b = inputs
    if a.dtype != b.dtype:
        raise TypeError(
            f"mod 要求两侧 dtype 一致，得到 {a.dtype.value} 与 {b.dtype.value}；"
            "请显式插入 cast"
        )
    if a.dtype is not DType.I32:
        raise TypeError(
            f"mod 只承接 i32（整数哈希的常态），得到 {a.dtype.value}；"
            "i8 请先 cast 到 i32，否则中间乘积会静默溢出"
        )
    return TensorType(a.dtype, _broadcast_shape(a.shape, b.shape))


def _mod_ref(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    a, b = args
    if np.any(b == 0):
        raise ValueError("mod 的模数不能为 0")
    # numpy 的 ``mod`` 就是 floor 语义：结果与除数同号，与 Python 的 ``%``
    # 和 torch.remainder 一致。C 的 ``%`` 在设备端另有修正，见 ew_mod_int。
    return np.mod(a, b)


register(
    Op(
        name="mod",
        infer=_infer_mod,
        reference=_mod_ref,
        attrs_schema=(),
        doc="整数取模（i32）：结果与除数同号，即 floor 语义，与 Python % / "
        "torch.remainder 一致；模数为 0 显式拒绝",
    )
)


_FLOAT_DTYPES = (DType.F32, DType.F16, DType.BF16, DType.F64)


def _infer_atan2(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    """``atan2(a, b)``：四象限反正切，``a`` 是 y、``b`` 是 x。

    与 ``torch.atan2(input, other)`` / ``np.arctan2`` 的参数顺序一致。刻意
    不接受整数：``atan2`` 的输出是浮点，而 IR 的二元算子要求两侧同 dtype，
    允许整数输入就必须在构图期插入隐式提升，那会让"这一维到底是 i32 还是
    f32"变得不可见。整数请显式 ``cast``。
    """
    a, b = inputs
    if a.dtype != b.dtype:
        raise TypeError(
            f"atan2 要求两侧 dtype 一致，得到 {a.dtype.value} 与 {b.dtype.value}；"
            "请显式插入 cast"
        )
    if a.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"atan2 只承接浮点，得到 {a.dtype.value}；整数请先 cast")
    return TensorType(a.dtype, _broadcast_shape(a.shape, b.shape))


def _atan2_ref(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.arctan2(args[0], args[1])


register(
    Op(
        name="atan2",
        infer=_infer_atan2,
        reference=_atan2_ref,
        doc="四象限反正切 atan2(a, b)，a 为 y、b 为 x，支持 numpy 广播",
    )
)


def _infer_remainder(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    """浮点取余，floor 语义（结果与除数同号）。

    与 ``mod`` 的关系：``mod`` 是 i32 的整数取余（哈希用），这里是浮点域，
    语义都是"向负无穷取整后的余数"，也就是 ``torch.remainder`` / Python
    ``%``。两者不能合并——``mod`` 的内核走整数 ``%`` 加修正，浮点走
    ``fmod`` 加修正，合并会把一条很短的语义绑到两套设备实现上。
    """
    a, b = inputs
    if a.dtype != b.dtype:
        raise TypeError(
            f"remainder 要求两侧 dtype 一致，得到 {a.dtype.value} 与 {b.dtype.value}；"
            "请显式插入 cast"
        )
    if a.dtype not in _FLOAT_DTYPES:
        raise TypeError(
            f"remainder 只承接浮点，得到 {a.dtype.value}；整数取余请用 mod"
        )
    return TensorType(a.dtype, _broadcast_shape(a.shape, b.shape))


def _remainder_ref(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    # numpy 的 ``mod`` 就是 floor 语义；除数为 0 时按 IEEE 给 nan，
    # 与 torch.remainder 一致，不在宿主上打警告。
    with np.errstate(all="ignore"):
        return np.mod(args[0], args[1])


register(
    Op(
        name="remainder",
        infer=_infer_remainder,
        reference=_remainder_ref,
        doc="浮点取余：floor 语义（结果与除数同号），与 torch.remainder / Python % 一致",
    )
)


def _infer_repeat_interleave(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    """沿单轴逐元素重复 ``repeats`` 次，``repeats`` 是构图期常量。

    只支持"每个元素重复同样次数"这一种形态：``torch.repeat_interleave``
    还接受长度等于该轴的张量作为逐元素次数，那需要按输入张量做前缀和，
    与这里"静态常量"的内核完全不同，等有真实模型需要时再单独加。
    """
    (t,) = inputs
    repeats = attrs.get("repeats")
    if isinstance(repeats, bool) or not isinstance(repeats, int):
        raise TypeError(
            f"repeat_interleave 的 repeats 必须是正整数，得到 {repeats!r}"
        )
    if repeats <= 0:
        raise ValueError(f"repeat_interleave 的 repeats 必须大于 0，得到 {repeats}")
    axis = _norm_axes(attrs.get("axis", -1), t.rank)[0]
    dim = t.shape[axis]
    if not isinstance(dim, int):
        raise ValueError(
            f"repeat_interleave 要求 axis={axis} 是静态维，得到符号维 {dim}；"
            "请先把该维固定成具体长度"
        )
    shape = list(t.shape)
    shape[axis] = dim * repeats
    return TensorType(t.dtype, tuple(shape))


def _repeat_interleave_ref(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    return np.repeat(args[0], attrs["repeats"], axis=attrs.get("axis", -1))


register(
    Op(
        name="repeat_interleave",
        infer=_infer_repeat_interleave,
        reference=_repeat_interleave_ref,
        attrs_schema=("repeats", "axis"),
        doc="沿单轴把每个元素重复 repeats 次（对应 torch.repeat_interleave 的常量形态）",
    )
)


def _infer_silu(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) not in (1, 2):
        raise ValueError(f"silu 需要 (x) 或 (x, gate) 输入，得到 {len(inputs)}")
    if len(inputs) == 2 and inputs[1] != inputs[0]:
        raise TypeError(f"silu 的 gate 必须与 x 同 dtype、同形状，得到 {inputs[1]} 与 {inputs[0]}")
    return inputs[0]


def _silu_ref(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x = args[0]
    # clip 只是为了避免 exp 溢出警告，不改变数学语义
    out = x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))
    if len(args) > 1:
        # 可选门控：silu(x) * gate。SwiGLU 的逐元素乘法因此不需要单独物化。
        out = out * args[1]
    return out


register(
    Op(
        name="silu",
        infer=_infer_silu,
        reference=_silu_ref,
        doc="SiLU / swish: x * sigmoid(x)；可选第 2 个输入 gate 表示 silu(x) * gate"
        "（SwiGLU 的门控乘法，ADR-010 的输入扩展）",
    )
)


def _relu_ref(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    return np.maximum(args[0], 0.0)


register(Op(name="relu", infer=_elementwise_unary_infer, reference=_relu_ref, doc="ReLU"))


_GELU_C = 0.7978845608028654  # sqrt(2/pi)
_GELU_K = 0.044715


def _gelu_ref(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x = args[0]
    return 0.5 * x * (1.0 + np.tanh(_GELU_C * (x + _GELU_K * x * x * x)))


register(
    Op(
        name="gelu",
        infer=_elementwise_unary_infer,
        reference=_gelu_ref,
        doc="GELU 的 tanh 近似：0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))。"
        "不用 erf 精确式，因为 erf 不是已有算子；本式与 PyTorch 的 "
        "approximate='tanh' 逐点一致，是 GPT/BERT 系列实际使用的那个定义",
    )
)


# 自动微分需要的初等一元函数。写成一元算子而不是"反向专用算子"，
# 因为反向规则本身应当由**已有语义**组合出来：dx 只依赖 x、y 与基本算术，
# 这样反向图天然是 PORTABLE 的，任何能跑前向的后端都能跑反向。
# 只补 AD 与真实模型真正需要的那几个，不做成通用数学库。
# ``erf`` 是精确 GELU（PyTorch ``approximate='none'``）的必要项：
# 它无法由已有初等函数精确表出，所以只能作为一元算子存在。
def _sigmoid_ref(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x = np.clip(args[0], -60.0, 60.0)  # 只为避免 exp 溢出告警，不改语义
    return 1.0 / (1.0 + np.exp(-x))


# numpy 本身没有 erf（那是 scipy 的能力）。参考实现只负责定义语义，
# 用 math.erf 逐元素求值即可；慢一点没关系，精度是 float64。
_ERF_VEC = np.vectorize(math.erf, otypes=[np.float64])


def _erf_ref(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x = args[0]
    return _ERF_VEC(x).astype(x.dtype, copy=False)


def _ieee_math(fn):
    """定义域外按 IEEE 返回，且不在宿主上打 numpy RuntimeWarning。

    ``log(0) = -inf``、``log(-1) = nan``、``sqrt(-1) = nan``、
    ``exp(x)`` 溢出成 ``inf`` 都是浮点语义的一部分（设备内核同样如此），
    不是错误；numpy 默认还会往 stderr 打一条 RuntimeWarning，同一张图在
    参考后端上跑就会污染日志。``div`` 早已按这条口径处理，这里补齐数学族。
    """

    def reference(args: List[np.ndarray], attrs: dict) -> np.ndarray:
        with np.errstate(all="ignore"):
            return fn(args[0])

    return reference


for _name, _ref, _doc in (
    ("exp", _ieee_math(np.exp), "自然指数 e^x"),
    ("log", _ieee_math(np.log), "自然对数 ln(x)"),
    ("sqrt", _ieee_math(np.sqrt), "平方根"),
    ("neg", lambda a, at: -a[0], "取负"),
    ("tanh", lambda a, at: np.tanh(a[0]), "双曲正切"),
    ("sigmoid", _sigmoid_ref, "logistic 函数 1/(1+e^-x)"),
    ("erf", _erf_ref, "误差函数 erf(x)"),
    # 取整与符号。``round`` 用 numpy 的 half-to-even，与 ``torch.round`` 一致；
    # 设备端对应 ``nearbyintf``（默认舍入模式是 round-half-to-even），不能用
    # ``roundf``（那是 half-away-from-zero）。
    ("sign", lambda a, at: np.sign(a[0]), "符号函数：正 1 / 负 -1 / 零 0，NaN 透传"),
    ("round", lambda a, at: np.round(a[0]), "四舍六入五取偶（与 torch.round 一致）"),
    ("floor", lambda a, at: np.floor(a[0]), "向下取整"),
    ("ceil", lambda a, at: np.ceil(a[0]), "向上取整"),
):
    register(Op(name=_name, infer=_elementwise_unary_infer, reference=_ref, doc=_doc))


def _ref_step(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    threshold = float(attrs.get("threshold", 0.0))
    return (args[0] > threshold).astype(args[0].dtype)


register(
    Op(
        name="step",
        infer=_elementwise_unary_infer,
        reference=_ref_step,
        attrs_schema=("threshold",),
        doc="Heaviside 阶跃：x > threshold → 1，否则 0（0 点取 0，与 ReLU 导数一致）",
    )
)


def _ref_spike(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    # 前向与 step 完全相同：只判断"是否越过阈值"，与膜电位的历史无关。
    # 阈值默认取 1.0（神经元语义），而 step 默认 0.0（通用阶跃）。
    threshold = float(attrs.get("threshold", 1.0))
    return (args[0] > threshold).astype(args[0].dtype)


register(
    Op(
        name="spike",
        infer=_elementwise_unary_infer,
        reference=_ref_spike,
        attrs_schema=("threshold", "alpha"),
        doc="脉冲发放：x > threshold → 1，否则 0。前向同 step，反向用替代梯度"
        "（窗宽 alpha 的三角窗），使 SNN 的发放过程可训练",
    )
)


# --------------------------------------------------------------------------
# SNN 事件驱动计算
# --------------------------------------------------------------------------


def _check_event_inputs(
    indices: TensorType,
    values: TensorType,
    offsets: TensorType,
    *,
    op: str,
) -> int:
    """校验事件三件套，返回 ``offsets`` 的**静态**长度（步数 + 1）。

    ``indices`` / ``values`` 的长度是事件数 ``E``：它随输入脉冲变化，在静态
    图里只能写成符号维（同一次执行里两者必须等长）。``offsets`` 的长度决定
    输出步数，必须静态——否则 ``infer`` 给不出输出形状。
    """
    if indices.rank != 1 or values.rank != 1 or offsets.rank != 1:
        raise ValueError(
            f"{op} 的 indices / values / offsets 都必须是一维，"
            f"得到 {indices.shape} / {values.shape} / {offsets.shape}"
        )
    if indices.dtype not in (DType.I32, DType.I8):
        raise TypeError(f"{op} 的 indices 必须是 i32/i8，得到 {indices.dtype}")
    if offsets.dtype != DType.I32:
        raise TypeError(f"{op} 的 offsets 必须是 i32，得到 {offsets.dtype}")
    if indices.shape[0] != values.shape[0]:
        raise ValueError(
            f"{op} 的 indices 与 values 必须等长，得到 "
            f"{indices.shape[0]} 与 {values.shape[0]}"
        )
    slots = dim_static(offsets.shape[0])
    if slots is None:
        raise ValueError(
            f"{op} 的 offsets 长度必须静态：它决定输出步数，"
            f"符号维 {offsets.shape[0]} 无法表达"
        )
    if slots < 1:
        raise ValueError(f"{op} 的 offsets 至少要有一个元素，得到 {slots}")
    return slots


def _check_event_offsets(offsets: np.ndarray, nnz: int, op: str) -> None:
    """``offsets`` 必须是 ``[0, nnz]`` 上的非降序列。

    内核按它切段；倒序或越界会读到别人的事件（甚至越界访存），因此这是
    **系统边界**上的校验，参考实现与后端都不能省略。
    """
    prev = int(offsets[0]) if offsets.size else 0
    if prev < 0 or prev > nnz:
        raise ValueError(f"{op} 的 offsets[0] 必须是 [0, {nnz}]，得到 {prev}")
    for s in range(1, int(offsets.shape[0])):
        cur = int(offsets[s])
        if cur < prev or cur > nnz:
            raise ValueError(
                f"{op} 的 offsets 必须是 [0, {nnz}] 上非降序列，"
                f"第 {s} 项是 {cur}（前一项 {prev}）"
            )
        prev = cur


def _infer_sparse_linear(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    weights, indices, values, offsets = inputs
    if weights.rank != 2:
        raise ValueError(
            f"sparse_linear 的 weights 必须是 (K, N) 二维，得到 {weights.shape}"
        )
    slots = _check_event_inputs(indices, values, offsets, op="sparse_linear")
    if values.dtype != weights.dtype:
        raise TypeError(
            f"sparse_linear 的 values 与 weights 必须同 dtype，"
            f"得到 {values.dtype} 与 {weights.dtype}"
        )
    return TensorType(values.dtype, (slots - 1, weights.shape[1]))


def _ref_sparse_linear(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    weights, indices, values, offsets = args
    steps = int(offsets.shape[0]) - 1
    rows, cols = int(weights.shape[0]), int(weights.shape[1])
    nnz = int(indices.shape[0])
    _check_event_offsets(np.asarray(offsets), nnz, "sparse_linear")
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= rows):
        raise ValueError(
            f"sparse_linear 的 indices 必须落在 [0, {rows})，"
            f"得到 [{int(idx.min())}, {int(idx.max())}]"
        )
    out = np.zeros((steps, cols), dtype=values.dtype)
    # f16 的累加在内核里是 float；参考实现用同一宽度，对拍的才不是舍入差。
    acc = np.float32 if values.dtype == np.float16 else values.dtype
    w = weights.astype(acc, copy=False)
    v = values.astype(acc, copy=False)
    for s in range(steps):
        lo, hi = int(offsets[s]), int(offsets[s + 1])
        if hi > lo:
            out[s] = (v[lo:hi] @ w[idx[lo:hi]]).astype(values.dtype)
    return out


register(
    Op(
        name="sparse_linear",
        infer=_infer_sparse_linear,
        reference=_ref_sparse_linear,
        doc="事件驱动的稀疏线性：按 CSR 事件 (indices, values, offsets) 把"
        "weights[K, N] 的行加权求和成 values.dtype[S, N]。稠密的 S×K 脉冲"
        "矩阵从不物化，每个输出元素只读该步真正发放的事件；indices/offsets 是"
        "i32，事件数 E 可以是符号维",
    )
)


def _infer_sparse_linear_weight_grad(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    dout, indices, values, offsets = inputs
    if dout.rank != 2:
        raise ValueError(
            f"sparse_linear_weight_grad 的 dout 必须是 (S, N) 二维，得到 {dout.shape}"
        )
    slots = _check_event_inputs(
        indices, values, offsets, op="sparse_linear_weight_grad"
    )
    if dout.shape[0] != slots - 1:
        raise ValueError(
            f"sparse_linear_weight_grad 的 dout 步数应为 {slots - 1}，"
            f"得到 {dout.shape[0]}"
        )
    if values.dtype != dout.dtype:
        raise TypeError(
            f"sparse_linear_weight_grad 的 values 与 dout 必须同 dtype，"
            f"得到 {values.dtype} 与 {dout.dtype}"
        )
    rows = attrs.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise TypeError(
            f"sparse_linear_weight_grad 的 rows 必须是正整数，得到 {rows!r}"
        )
    return TensorType(dout.dtype, (rows, dout.shape[1]))


def _ref_sparse_linear_weight_grad(
    args: List[np.ndarray], attrs: dict
) -> np.ndarray:
    dout, indices, values, offsets = args
    rows = int(attrs["rows"])
    steps, cols = int(dout.shape[0]), int(dout.shape[1])
    nnz = int(indices.shape[0])
    _check_event_offsets(np.asarray(offsets), nnz, "sparse_linear_weight_grad")
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= rows):
        raise ValueError(
            f"sparse_linear_weight_grad 的 indices 必须落在 [0, {rows})，"
            f"得到 [{int(idx.min())}, {int(idx.max())}]"
        )
    acc = np.float32 if dout.dtype == np.float16 else dout.dtype
    dw = np.zeros((rows, cols), dtype=acc)
    v = values.astype(acc, copy=False)
    for s in range(steps):
        lo, hi = int(offsets[s]), int(offsets[s + 1])
        if hi > lo:
            # add.at 是"重复下标累加"：同一个神经元在一个 batch 里发放多次
            # 是常态，普通下标赋值只会留下最后一次贡献。
            contrib = v[lo:hi][:, None] * dout[s].astype(acc, copy=False)[None, :]
            np.add.at(dw, idx[lo:hi], contrib)
    return dw.astype(dout.dtype)


register(
    Op(
        name="sparse_linear_weight_grad",
        infer=_infer_sparse_linear_weight_grad,
        reference=_ref_sparse_linear_weight_grad,
        attrs_schema=("rows",),
        doc="sparse_linear 对 weights 的梯度：把 values[e] * dout[s] 散射累加"
        "回 indices[e] 那一行。rows 是权重行数 K——反向图里不再有 weights，"
        "只能由属性带过来",
    )
)


def _infer_sparse_linear_values_grad(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    """``(dout, weights, indices, offsets) -> (E,)``。

    输出形状直接取 ``indices`` 的长度：它是事件数，可以和前向一样是符号维，
    且每个事件恰好产出一个标量。这里没有 ``values`` 这个输入，于是把
    ``indices`` 自己传给 ``_check_event_inputs`` 顶替"等长"这一条——事件数
    本来就是由 ``indices`` 定义的，与 ``values`` 等长在前向构图时已经查过。
    """
    dout, weights, indices, offsets = inputs
    if dout.rank != 2:
        raise ValueError(
            f"sparse_linear_values_grad 的 dout 必须是 (S, N) 二维，"
            f"得到 {dout.shape}"
        )
    if weights.rank != 2:
        raise ValueError(
            f"sparse_linear_values_grad 的 weights 必须是 (K, N) 二维，"
            f"得到 {weights.shape}"
        )
    slots = _check_event_inputs(indices, indices, offsets, op="sparse_linear_values_grad")
    if dout.shape[1] != weights.shape[1]:
        raise ValueError(
            f"sparse_linear_values_grad 的 dout 与 weights 最后一维必须一致，"
            f"得到 {dout.shape[1]} 与 {weights.shape[1]}"
        )
    if dout.shape[0] != slots - 1:
        raise ValueError(
            f"sparse_linear_values_grad 的 dout 步数应为 {slots - 1}，"
            f"得到 {dout.shape[0]}"
        )
    if dout.dtype != weights.dtype:
        raise TypeError(
            f"sparse_linear_values_grad 的 dout 与 weights 必须同 dtype，"
            f"得到 {dout.dtype} 与 {weights.dtype}"
        )
    return TensorType(dout.dtype, indices.shape)


def _ref_sparse_linear_values_grad(
    args: List[np.ndarray], attrs: dict
) -> np.ndarray:
    """``dvalues[e] = dout[s(e), :] · weights[indices[e], :]``。

    ``s(e)`` 是事件 ``e`` 所属的步，由 ``offsets`` 唯一决定。与权重梯度不同，
    这里**没有重复下标问题**：每个事件只写一次，不需要 ``add.at``。
    """
    dout, weights, indices, offsets = args
    steps = int(dout.shape[0])
    rows = int(weights.shape[0])
    nnz = int(indices.shape[0])
    _check_event_offsets(np.asarray(offsets), nnz, "sparse_linear_values_grad")
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= rows):
        raise ValueError(
            f"sparse_linear_values_grad 的 indices 必须落在 [0, {rows})，"
            f"得到 [{int(idx.min())}, {int(idx.max())}]"
        )
    acc = np.float32 if dout.dtype == np.float16 else dout.dtype
    d = dout.astype(acc, copy=False)
    w = weights.astype(acc, copy=False)
    dv = np.zeros(nnz, dtype=acc)
    for s in range(steps):
        lo, hi = int(offsets[s]), int(offsets[s + 1])
        if hi > lo:
            dv[lo:hi] = (w[idx[lo:hi]] * d[s][None, :]).sum(axis=1)
    return dv.astype(dout.dtype)


register(
    Op(
        name="sparse_linear_values_grad",
        infer=_infer_sparse_linear_values_grad,
        reference=_ref_sparse_linear_values_grad,
        doc="sparse_linear 对事件幅度 values 的梯度：dvalues[e] = dout[s(e)] · "
        "weights[indices[e]]。s(e) 由 offsets 决定，每个事件只写一次，"
        "不需要原子加。输出长度等于事件数 E，可以是符号维",
    )
)


def _check_spike_matrix(spikes: TensorType, op: str) -> Tuple[int, int]:
    if spikes.rank != 2:
        raise ValueError(
            f"{op} 的输入必须是 (T, K) 二维脉冲矩阵，得到 {spikes.shape}"
        )
    if spikes.dtype not in (DType.F32, DType.F16):
        raise TypeError(f"{op} 的输入必须是 f32/f16，得到 {spikes.dtype}")
    rows = dim_static(spikes.shape[0])
    cols = dim_static(spikes.shape[1])
    if rows is None or cols is None:
        raise ValueError(
            f"{op} 的 T 与 K 必须静态：容量与输出长度都写在属性/形状里，"
            f"得到 {spikes.shape}"
        )
    return rows, cols


def _infer_spike_offsets(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    (spikes,) = inputs
    rows, _ = _check_spike_matrix(spikes, "spike_offsets")
    # 每行起点：长度为 T+1，最后一项是总事件数
    return TensorType(DType.I32, (rows + 1,))


def _ref_spike_offsets(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    (spikes,) = args
    rows = int(spikes.shape[0])
    offsets = np.zeros(rows + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(
        np.count_nonzero(spikes.astype(np.float32), axis=1), dtype=np.int64
    )
    return offsets


register(
    Op(
        name="spike_offsets",
        infer=_infer_spike_offsets,
        reference=_ref_spike_offsets,
        doc="统计 (T, K) 脉冲矩阵每行的非零数，给出长度为 T+1 的 CSR 起点表；"
        "末项就是总事件数。与 spike_csr 合起来构成层间压缩的两半",
    )
)


def _check_spike_offsets(
    spikes: TensorType, offsets: TensorType, op: str
) -> Tuple[int, int]:
    """校验 ``(T, K)`` 脉冲矩阵与长度 ``T+1`` 的 i32 起点表，返回 ``(T, K)``。"""
    rows, cols = _check_spike_matrix(spikes, op)
    if offsets.rank != 1 or offsets.dtype != DType.I32:
        raise TypeError(
            f"{op} 的 offsets 必须是一维 i32，得到 {offsets.shape}/{offsets.dtype}"
        )
    slots = dim_static(offsets.shape[0])
    if slots != rows + 1:
        raise ValueError(f"{op} 的 offsets 长度必须是 T+1={rows + 1}，得到 {slots}")
    return rows, cols


def _check_event_capacity(attrs: dict, rows: int, cols: int, op: str) -> int:
    """校验事件容量：必须是正整数，且不超过 ``T×K``。"""
    capacity = attrs.get("capacity")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise TypeError(f"{op} 的 capacity 必须是正整数，得到 {capacity!r}")
    if capacity > rows * cols:
        raise ValueError(
            f"{op} 的 capacity={capacity} 超过 T×K={rows * cols}："
            "非零元素不可能比元素总数还多"
        )
    return capacity


def _infer_spike_csr(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    spikes, offsets = inputs
    rows, cols = _check_spike_offsets(spikes, offsets, "spike_csr")
    capacity = _check_event_capacity(attrs, rows, cols, "spike_csr")
    return TensorType(DType.I32, (capacity,))


def _ref_spike_csr(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    spikes, offsets = args
    rows = int(spikes.shape[0])
    capacity = int(attrs["capacity"])
    indices = np.zeros(capacity, dtype=np.int32)
    for t in range(rows):
        lo, hi = int(offsets[t]), int(offsets[t + 1])
        if hi < lo or hi > capacity:
            raise ValueError(
                f"spike_csr 的 offsets 必须在 [0, {capacity}] 上非降，"
                f"第 {t} 段是 [{lo}, {hi})"
            )
        cols_hit = np.nonzero(spikes[t].astype(np.float32))[0]
        if cols_hit.size != hi - lo:
            raise ValueError(
                f"spike_csr 的 offsets 与脉冲不一致：第 {t} 步有 "
                f"{cols_hit.size} 个非零，offsets 给出 {hi - lo}"
            )
        indices[lo:hi] = cols_hit
    return indices


register(
    Op(
        name="spike_csr",
        infer=_infer_spike_csr,
        reference=_ref_spike_csr,
        attrs_schema=("capacity",),
        doc="按 spike_offsets 给出的起点表，把 (T, K) 脉冲矩阵的非零列号写进"
        "长度固定的 indices[capacity]。容量固定是静态图的前提，溢出必须报错",
    )
)


def _check_event_indices(indices: TensorType, op: str) -> None:
    """校验 CSR 列号是一维 i32/i8。"""
    if indices.rank != 1 or indices.dtype not in (DType.I32, DType.I8):
        raise TypeError(
            f"{op} 的 indices 必须是 i32/i8 一维，"
            f"得到 {indices.shape}/{indices.dtype}"
        )


def _check_spike_linear_inputs(
    spikes: TensorType,
    weights: TensorType,
    indices: TensorType,
    offsets: TensorType,
    op: str,
) -> Tuple[int, int]:
    """校验融合事件线性的四个输入，返回 ``(步数, 权重列数)``。"""
    rows, cols = _check_spike_offsets(spikes, offsets, op)
    if weights.rank != 2:
        raise ValueError(f"{op} 的 weights 必须是 (K, N) 二维，得到 {weights.shape}")
    if weights.shape[0] != cols:
        raise ValueError(
            f"{op} 的 weights 行数应等于脉冲宽度 K={cols}，得到 {weights.shape[0]}"
        )
    if weights.dtype != spikes.dtype:
        raise TypeError(
            f"{op} 的 spikes 与 weights 必须同 dtype，"
            f"得到 {spikes.dtype} 与 {weights.dtype}"
        )
    _check_event_indices(indices, op)
    # 不把 N 转成 int：与 sparse_linear 一致，允许最后一维是符号维。
    return rows, weights.shape[1]


def _infer_spike_linear(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    spikes, weights, indices, offsets = inputs
    rows, out_cols = _check_spike_linear_inputs(
        spikes, weights, indices, offsets, "spike_linear"
    )
    return TensorType(spikes.dtype, (rows, out_cols))


def _ref_spike_linear(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    spikes, weights, indices, offsets = args
    rows, cols = int(spikes.shape[0]), int(weights.shape[1])
    width = int(weights.shape[0])
    nnz = int(indices.shape[0])
    _check_event_offsets(np.asarray(offsets), nnz, "spike_linear")
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= width):
        raise ValueError(
            f"spike_linear 的 indices 必须落在 [0, {width})，"
            f"得到 [{int(idx.min())}, {int(idx.max())}]"
        )
    out = np.zeros((rows, cols), dtype=spikes.dtype)
    acc = np.float32 if spikes.dtype == np.float16 else spikes.dtype
    sp = spikes.astype(acc, copy=False)
    w = weights.astype(acc, copy=False)
    for s in range(rows):
        lo, hi = int(offsets[s]), int(offsets[s + 1])
        if hi > lo:
            hit = idx[lo:hi]
            out[s] = (sp[s, hit] @ w[hit]).astype(spikes.dtype)
    return out


register(
    Op(
        name="spike_linear",
        infer=_infer_spike_linear,
        reference=_ref_spike_linear,
        doc="事件驱动的脉冲线性：out[s, n] = Σ_e spikes[s, indices[e]] * W[indices[e], n]，"
        "e 只扫第 s 步实际发放的事件。值直接从脉冲矩阵读，不物化 values 缓冲，"
        "因此反向对 spikes 就是普通的 matmul，未发放位置的 surrogate 梯度不会丢",
    )
)


def _infer_spike_linear_weight_grad(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    dout, spikes, indices, offsets = inputs
    if dout.rank != 2:
        raise ValueError(
            f"spike_linear_weight_grad 的 dout 必须是 (S, N) 二维，得到 {dout.shape}"
        )
    steps, width = _check_spike_offsets(spikes, offsets, "spike_linear_weight_grad")
    if dout.shape[0] != steps:
        raise ValueError(
            f"spike_linear_weight_grad 的 dout 步数应为 {steps}，得到 {dout.shape[0]}"
        )
    if dout.dtype != spikes.dtype:
        raise TypeError(
            f"spike_linear_weight_grad 的 spikes 与 dout 必须同 dtype，"
            f"得到 {spikes.dtype} 与 {dout.dtype}"
        )
    _check_event_indices(indices, "spike_linear_weight_grad")
    return TensorType(dout.dtype, (width, dout.shape[1]))


def _ref_spike_linear_weight_grad(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    dout, spikes, indices, offsets = args
    steps, width = int(dout.shape[0]), int(spikes.shape[1])
    cols = int(dout.shape[1])
    nnz = int(indices.shape[0])
    _check_event_offsets(np.asarray(offsets), nnz, "spike_linear_weight_grad")
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= width):
        raise ValueError(
            f"spike_linear_weight_grad 的 indices 必须落在 [0, {width})，"
            f"得到 [{int(idx.min())}, {int(idx.max())}]"
        )
    acc = np.float32 if dout.dtype == np.float16 else dout.dtype
    dw = np.zeros((width, cols), dtype=acc)
    sp = spikes.astype(acc, copy=False)
    do = dout.astype(acc, copy=False)
    for s in range(steps):
        lo, hi = int(offsets[s]), int(offsets[s + 1])
        if hi > lo:
            hit = idx[lo:hi]
            # add.at 是"重复下标累加"：同一神经元在同一步多次发放是常态。
            np.add.at(dw, hit, sp[s, hit][:, None] * do[s][None, :])
    return dw.astype(dout.dtype)


register(
    Op(
        name="spike_linear_weight_grad",
        infer=_infer_spike_linear_weight_grad,
        reference=_ref_spike_linear_weight_grad,
        doc="spike_linear 对权重的梯度：把 spikes[s, idx[e]] * dout[s] 散射累加回 "
        "idx[e] 那一行。值由脉冲矩阵现场读取，不需要单独的 values 缓冲",
    )
)


def _infer_constant(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    return TensorType(DType(attrs["dtype"]), _as_shape(attrs["shape"]))


def _ref_constant(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    shape = _as_shape(attrs["shape"])
    dtype = DType(attrs["dtype"]).np_dtype
    data = attrs.get("data")
    if data is None:
        return np.full(shape, float(attrs.get("value", 0.0)), dtype=dtype)
    # data 是常量折叠产出的任意张量；JSON 里是嵌套 list，读回来可能是 tuple。
    return np.array(np.asarray(data, dtype=dtype).reshape(shape), dtype=dtype)


register(
    Op(
        name="constant",
        infer=_infer_constant,
        reference=_ref_constant,
        attrs_schema=("dtype", "shape", "value", "data"),
        doc="常量张量。零输入节点；值写在属性里，因此图仍然可序列化、可独立执行。"
        "``value`` 表示标量填充，``data`` 表示任意常量张量（常量折叠的产物），"
        "两者互斥",
        nullary=True,
    )
)


# --------------------------------------------------------------------------
# 矩阵算子
# --------------------------------------------------------------------------


def _infer_matmul(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    a, b = inputs
    if a.rank < 2 or b.rank < 2:
        raise ValueError(f"matmul 需要秩 >= 2，得到 {a.rank} 与 {b.rank}")

    ka, kb = a.shape[-1], b.shape[-2]
    static_ka, static_kb = dim_static(ka), dim_static(kb)
    if static_ka is not None and static_kb is not None:
        if static_ka != static_kb:
            raise ValueError(f"matmul 内维不匹配: {ka} vs {kb}")
    elif static_ka is None and static_kb is None and ka != kb:
        raise ValueError(f"matmul 内维符号不一致: {ka} vs {kb}")

    batch = _broadcast_shape(a.shape[:-2], b.shape[:-2])
    return TensorType(a.dtype, batch + (a.shape[-2], b.shape[-1]))


def _ref_matmul(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    return np.matmul(args[0], args[1])


register(
    Op(
        name="matmul",
        infer=_infer_matmul,
        reference=_ref_matmul,
        doc="批量矩阵乘：(..., M, K) x (..., K, N) -> (..., M, N)",
    )
)


# --------------------------------------------------------------------------
# 量化（weight-only int8 与 W8A8）
# --------------------------------------------------------------------------
#
# 语义边界写在这里，因为它是"能不能对外说支持量化"的分界线：
#
# - 对称量化，无零点：``q = clip(rint(w / scale), -127, 127)``。刻意不用
#   -128：反量化后两端量程不对称，同一份权重在不同实现里会被舍到不同端点，
#   逐位对拍就失去意义。
# - ``rint`` 是 round-half-to-even，与 numpy / PyTorch 的 ``round`` 一致；
#   四舍五入（half-away-from-zero）会在 .5 处产生系统性偏差。
# - 缩放系数必须严格为正。``scale <= 0`` 不是"退化成另一种量化"，而是
#   语义未定义（除零 / 符号翻转），因此参考实现与设备实现都显式拒绝。
# - **weight-only 的权重驻留是 i8，计算仍是 f32**：``linear_i8`` 先反量化再走
#   普通 GEMM。它省的是权重占用与读带宽，不是算力。
# - **W8A8 的累加在 int32 里**：``linear_i8a8`` 先做整数矩阵乘，再乘激活 /
#   权重两个 scale。它才是 int8 tensor-core GEMM 的累加语义，因此另有两条
#   约束：scale 必须沿压缩轴恒定（否则提不出求和号），且
#   ``K * 127 * 127`` 不得超过 int32 上界（否则回绕）。设备内核见 5.309。
# - **QAT 的假量化（``fake_quantize_i8``）不产出 int8**：它把 f32 量化到
#   int8 的格点上再原样放大回 f32，前向逐位等于
#   ``dequantize_i8(quantize_i8(x, scale), scale)``；区别只在反向——整数
#   中转没有可用的梯度，假量化用直通估计器（STE）把梯度送到连续权重上。
# - **LSQ 版本（``fake_quantize_i8_lsq``）前向与上一行完全相同**，只有反向
#   不同：梯度不再无条件穿过量化，而是按"是否被夹到端点"分两段，并把
#   ``scale`` 当作可学参数给出梯度（Esser et al. 2020）。拆成两个算子而不是
#   给一个算子加开关：STE 与 LSQ 对 ``scale`` 的语义不同（一是不接收梯度，
#   一是必须接收），写成属性会让"这个 scale 到底更不更新"藏在 attrs 里。

_QUANT_MIN = -127
_QUANT_MAX = 127


def _check_broadcastable(
    target: TensorType, param: TensorType, what: str, label: str
) -> None:
    """校验量化参数（scale / zero_point）能广播到 target。

    与 ``_broadcast_shape`` 同一套保守规则：符号维只在"等于 1 或两边相同"
    时才放行，其余情形构图期拒绝——IR 没有能表达"这一维可能是 1"的广播维。
    """
    if param.rank > target.rank:
        raise ValueError(
            f"{what} 的 {label} 秩 {param.rank} 大于被量化张量的秩 {target.rank}"
        )
    for s, d in zip(reversed(param.shape), reversed(target.shape)):
        if s == 1 or s == d:
            continue
        if isinstance(s, int) and isinstance(d, int):
            raise ValueError(
                f"{what} 的 {label} 形状 {param.shape} 不能广播到 {target.shape}"
            )
        raise ValueError(
            f"{what} 的 {label} 符号维 {s} 与目标维 {d} 无法在 IR 里表达广播："
            "请给出静态的 1 或相同的符号"
        )


def _check_scale_broadcast(target: TensorType, scale: TensorType, what: str) -> None:
    _check_broadcastable(target, scale, what, "scale")


def _infer_quantize_i8(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 2:
        raise ValueError(f"quantize_i8 需要 (weight, scale) 两个输入，得到 {len(inputs)}")
    weight, scale = inputs
    if weight.dtype != DType.F32:
        raise TypeError(
            f"quantize_i8 只接受 f32 权重，得到 {weight.dtype.value}；"
            "bf16/f16 权重要先 cast 到 f32，量化语义只在 f32 上定义"
        )
    if scale.dtype != DType.F32:
        raise TypeError(f"quantize_i8 的 scale 必须是 f32，得到 {scale.dtype.value}")
    if weight.rank < 1:
        raise ValueError("quantize_i8 至少需要 1 维权重")
    _check_scale_broadcast(weight, scale, "quantize_i8")
    return TensorType(DType.I8, weight.shape)


def _ref_quantize_i8(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    weight, scale = args
    if float(np.min(scale)) <= 0.0:
        raise ValueError(
            f"quantize_i8 的 scale 必须严格为正，得到最小值 {float(np.min(scale))}"
        )
    q = np.rint(weight / scale)
    return np.clip(q, _QUANT_MIN, _QUANT_MAX).astype(np.int8)


register(
    Op(
        name="quantize_i8",
        infer=_infer_quantize_i8,
        reference=_ref_quantize_i8,
        doc="对称 int8 量化：q = clip(rint(w / scale), -127, 127)，"
        "scale 广播到权重（每输出通道给一个标量就是逐通道量化）；"
        "输出 i8 与权重同形状",
    )
)


def _infer_dequantize_i8(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 2:
        raise ValueError(f"dequantize_i8 需要 (q, scale) 两个输入，得到 {len(inputs)}")
    q, scale = inputs
    if q.dtype != DType.I8:
        raise TypeError(f"dequantize_i8 的第一个输入必须是 i8，得到 {q.dtype.value}")
    if scale.dtype != DType.F32:
        raise TypeError(f"dequantize_i8 的 scale 必须是 f32，得到 {scale.dtype.value}")
    _check_scale_broadcast(q, scale, "dequantize_i8")
    return TensorType(DType.F32, q.shape)


def _ref_dequantize_i8(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    q, scale = args
    return q.astype(np.float32) * scale


register(
    Op(
        name="dequantize_i8",
        infer=_infer_dequantize_i8,
        reference=_ref_dequantize_i8,
        doc="反量化：f32(q) * scale（scale 广播到 q），输出与 q 同形状的 f32",
    )
)


def _infer_fake_quantize_i8(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 2:
        raise ValueError(
            f"fake_quantize_i8 需要 (x, scale) 两个输入，得到 {len(inputs)}"
        )
    x, scale = inputs
    if x.dtype != DType.F32:
        raise TypeError(
            f"fake_quantize_i8 只接受 f32 输入，得到 {x.dtype.value}；"
            "假量化与 quantize_i8 同源，语义只在 f32 上定义"
        )
    if scale.dtype != DType.F32:
        raise TypeError(f"fake_quantize_i8 的 scale 必须是 f32，得到 {scale.dtype.value}")
    if x.rank < 1:
        raise ValueError("fake_quantize_i8 至少需要 1 维输入")
    _check_scale_broadcast(x, scale, "fake_quantize_i8")
    return TensorType(DType.F32, x.shape)


def _ref_fake_quantize_i8(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x, scale = args
    if float(np.min(scale)) <= 0.0:
        raise ValueError(
            f"fake_quantize_i8 的 scale 必须严格为正，得到最小值 {float(np.min(scale))}"
        )
    # 先真的落成 int8 再放大：这样前向与"量化算子 + 反量化算子"逐位同源，
    # 而不是另一份"round 之后乘回去"的近似实现。
    q = np.clip(np.rint(x / scale), _QUANT_MIN, _QUANT_MAX).astype(np.int8)
    return q.astype(np.float32) * scale


register(
    Op(
        name="fake_quantize_i8",
        infer=_infer_fake_quantize_i8,
        reference=_ref_fake_quantize_i8,
        doc="量化感知训练（QAT）的假量化：f32 量化到 int8 格点再放大回 f32，"
        "前向逐位等于 dequantize_i8(quantize_i8(x, scale), scale)；"
        "反向是直通估计器（STE），scale 不接收梯度",
    )
)


def _infer_fake_quantize_i8_lsq(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    if len(inputs) != 2:
        raise ValueError(
            f"fake_quantize_i8_lsq 需要 (x, scale) 两个输入，得到 {len(inputs)}"
        )
    x, scale = inputs
    if x.dtype != DType.F32:
        raise TypeError(
            f"fake_quantize_i8_lsq 只接受 f32 输入，得到 {x.dtype.value}；"
            "它的前向与 fake_quantize_i8 逐位相同，语义只在 f32 上定义"
        )
    if scale.dtype != DType.F32:
        raise TypeError(
            f"fake_quantize_i8_lsq 的 scale 必须是 f32，得到 {scale.dtype.value}"
        )
    if x.rank < 1:
        raise ValueError("fake_quantize_i8_lsq 至少需要 1 维输入")
    _check_scale_broadcast(x, scale, "fake_quantize_i8_lsq")
    return TensorType(DType.F32, x.shape)


register(
    Op(
        name="fake_quantize_i8_lsq",
        infer=_infer_fake_quantize_i8_lsq,
        # 前向与 fake_quantize_i8 共用同一份参考实现：LSQ 只是反向不同，
        # 若另写一份舍入就迟早会在端点 / 半偶舍入上分叉。
        reference=_ref_fake_quantize_i8,
        doc="LSQ（Esser et al. 2020）的假量化：前向与 fake_quantize_i8 逐位"
        "相同，反向按非饱和区直通、并把量化误差 v_hat - v 的归约和送给"
        "scale，使量化步长本身可学（scale 必须是图输入，不能是 constant）",
    )
)


#: int32 累加器的安全上界：``K * 127 * 127`` 不得超过 ``2**31 - 1``。
#:
#: 超过这个 K 时，即使每个元素都取 int8 端点，int32 累加也会回绕；而回绕
#: 依赖溢出行为，不是能跨后端对拍的语义。构图期（K 静态时）与执行期都
#: 显式拒绝，而不是"让它绕一圈"。
_I32_ACC_LIMIT = (2**31 - 1) // (_QUANT_MAX * _QUANT_MAX)


def _infer_linear_i8(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) not in (3, 4):
        raise ValueError(
            f"linear_i8 需要 (x, qweight, scale[, bias]) 输入，得到 {len(inputs)}"
        )
    x, qweight, scale = inputs[0], inputs[1], inputs[2]
    if x.dtype != DType.F32:
        raise TypeError(
            f"linear_i8 的 x 必须是 f32，得到 {x.dtype.value}；"
            "反量化之后的 GEMM 用 f32 累加，混精度入口不在这一版"
        )
    if x.rank < 2:
        raise ValueError(f"linear_i8 的 x 至少是 2 维，得到秩 {x.rank}")
    if qweight.dtype != DType.I8:
        raise TypeError(f"linear_i8 的权重必须是 i8，得到 {qweight.dtype.value}")
    if qweight.rank != 2:
        raise ValueError(f"linear_i8 的量化权重必须是 2 维 (K, N)，得到 {qweight.shape}")
    if scale.dtype != DType.F32:
        raise TypeError(f"linear_i8 的 scale 必须是 f32，得到 {scale.dtype.value}")
    _check_scale_broadcast(qweight, scale, "linear_i8")

    k, n = qweight.shape
    kx = x.shape[-1]
    if isinstance(kx, int) and isinstance(k, int):
        if kx != k:
            raise ValueError(
                f"linear_i8 的内维不匹配：x 的最后一维 {kx} 与权重的 K={k}"
            )
    elif kx != k:
        raise ValueError(f"linear_i8 的内维符号不一致：x={kx} 与权重 K={k}")

    if len(inputs) == 4:
        bias = inputs[3]
        if bias.dtype != DType.F32:
            raise TypeError(f"linear_i8 的 bias 必须是 f32，得到 {bias.dtype.value}")
        if bias.rank != 1:
            raise ValueError(f"linear_i8 的 bias 必须是秩 1，得到秩 {bias.rank}")
        if isinstance(bias.shape[0], int) and isinstance(n, int) and bias.shape[0] != n:
            raise ValueError(f"linear_i8 的 bias 长度 {bias.shape[0]} 与 N={n} 不一致")
    return TensorType(DType.F32, x.shape[:-1] + (n,))


def _ref_linear_i8(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x, qweight, scale = args[0], args[1], args[2]
    if float(np.min(scale)) <= 0.0:
        raise ValueError(
            f"linear_i8 的 scale 必须严格为正，得到最小值 {float(np.min(scale))}"
        )
    weight = qweight.astype(np.float32) * scale
    out = np.matmul(x, weight)
    if len(args) > 3:
        out = out + args[3]
    return out


register(
    Op(
        name="linear_i8",
        infer=_infer_linear_i8,
        reference=_ref_linear_i8,
        doc="weight-only int8 线性层：(..., K) x i8(K, N) -> (..., N)。"
        "权重按 scale 反量化后在 f32 上做 GEMM，因此数值等于"
        "``matmul(x, dequantize_i8(qweight, scale)) [+ bias]``；"
        "省的是权重占用与读带宽，不是算力",
    )
)


def _infer_linear_i8_group(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    if len(inputs) not in (3, 4):
        raise ValueError(
            f"linear_i8_group 需要 (x, qweight, scale[, bias]) 输入，得到 {len(inputs)}"
        )
    x, qweight, scale = inputs[0], inputs[1], inputs[2]
    if x.dtype != DType.F32:
        raise TypeError(
            f"linear_i8_group 的 x 必须是 f32，得到 {x.dtype.value}；"
            "反量化之后的 GEMM 用 f32 累加，混精度入口不在这一版"
        )
    if x.rank < 2:
        raise ValueError(f"linear_i8_group 的 x 至少是 2 维，得到秩 {x.rank}")
    if qweight.dtype != DType.I8:
        raise TypeError(
            f"linear_i8_group 的权重必须是 i8，得到 {qweight.dtype.value}"
        )
    if qweight.rank != 2:
        raise ValueError(
            f"linear_i8_group 的量化权重必须是 2 维 (K, N)，得到 {qweight.shape}"
        )
    if scale.dtype != DType.F32:
        raise TypeError(f"linear_i8_group 的 scale 必须是 f32，得到 {scale.dtype.value}")
    _check_group_params(qweight, scale, "linear_i8_group", "scale")

    k, n = qweight.shape
    kx = x.shape[-1]
    if isinstance(kx, int) and isinstance(k, int):
        if kx != k:
            raise ValueError(
                f"linear_i8_group 的内维不匹配：x 的最后一维 {kx} 与权重的 K={k}"
            )
    elif kx != k:
        raise ValueError(f"linear_i8_group 的内维符号不一致：x={kx} 与权重 K={k}")

    if len(inputs) == 4:
        bias = inputs[3]
        if bias.dtype != DType.F32:
            raise TypeError(f"linear_i8_group 的 bias 必须是 f32，得到 {bias.dtype.value}")
        if bias.rank != 1:
            raise ValueError(f"linear_i8_group 的 bias 必须是秩 1，得到秩 {bias.rank}")
        if isinstance(bias.shape[0], int) and isinstance(n, int) and bias.shape[0] != n:
            raise ValueError(
                f"linear_i8_group 的 bias 长度 {bias.shape[0]} 与 N={n} 不一致"
            )
    return TensorType(DType.F32, x.shape[:-1] + (n,))


def _ref_linear_i8_group(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x, qweight, scale = args[0], args[1], args[2]
    if float(np.min(scale)) <= 0.0:
        raise ValueError(
            f"linear_i8_group 的 scale 必须严格为正，得到最小值 {float(np.min(scale))}"
        )
    repeated = np.repeat(scale, qweight.shape[0] // scale.shape[0], axis=0)
    out = np.matmul(x, qweight.astype(np.float32) * repeated)
    if len(args) > 3:
        out = out + args[3]
    return out


register(
    Op(
        name="linear_i8_group",
        infer=_infer_linear_i8_group,
        reference=_ref_linear_i8_group,
        doc="weight-only int8 线性层，权重按**组**共用 scale：(..., K) x i8(K, N) "
        "-> (..., N)，scale 是 (K/G, N)，组大小由形状推出。数值等于 "
        "``matmul(x, dequantize_i8_group(qweight, scale)) [+ bias]``；"
        "与 ``linear_i8`` 的差别只有权重侧的量化粒度（GPTQ / AWQ 的 group_size 形态）",
    )
)


# --------------------------------------------------------------------------
# fp8 线性层
# --------------------------------------------------------------------------
#
# 与 int8 的两个版本一一对应，差别只在语义来源：fp8 的载荷是 f32 载体
# （``DType.F8E4M3`` / ``F8E5M2`` 的 ``np_dtype`` 都是 float32），值已经在
# 写入侧落到格式格点上，所以这里反量化就是一次乘法，不需要解码位模式。
#
# 这一版**不承诺算力**：两条路径都是"反量化成 f32 再 GEMM"，与
# ``linear_i8`` 的定位相同——省的是权重占用与读带宽（真正 1 字节存储要到
# 5.328，届时带宽才会下降），不是 tensor core 吞吐。fp8 tensor core 需要
# 另一套缩放与累加语义，不在本版声称范围内。

_FP8_TYPES: FrozenSet[DType] = frozenset({DType.F8E4M3, DType.F8E5M2})


def _check_same_fp8(x: TensorType, w: TensorType, what: str) -> None:
    for item, role in ((x, "激活"), (w, "权重")):
        if item.dtype not in _FP8_TYPES:
            raise TypeError(
                f"{what} 的{role}必须是 fp8（f8e4m3 / f8e5m2），"
                f"得到 {item.dtype.value}"
            )
    if x.dtype != w.dtype:
        raise TypeError(
            f"{what} 的激活与权重必须是同一种 fp8 格式，得到 "
            f"{x.dtype.value} 与 {w.dtype.value}；混合格式需要显式 cast"
        )


def _check_bias(
    inputs: Tuple[TensorType, ...], count_without_bias: int, n: Dim, what: str
) -> None:
    if len(inputs) != count_without_bias + 1:
        return
    bias = inputs[-1]
    if bias.dtype != DType.F32:
        raise TypeError(f"{what} 的 bias 必须是 f32，得到 {bias.dtype.value}")
    if bias.rank != 1:
        raise ValueError(f"{what} 的 bias 必须是秩 1，得到秩 {bias.rank}")
    if isinstance(bias.shape[0], int) and isinstance(n, int) and bias.shape[0] != n:
        raise ValueError(f"{what} 的 bias 长度 {bias.shape[0]} 与 N={n} 不一致")


def _infer_linear_fp8(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) not in (3, 4):
        raise ValueError(
            "linear_fp8 需要 (x, qweight, weight_scale[, bias]) 输入，"
            f"得到 {len(inputs)}"
        )
    x, qweight, scale = inputs[0], inputs[1], inputs[2]
    if x.dtype != DType.F32:
        raise TypeError(
            f"linear_fp8 的 x 必须是 f32，得到 {x.dtype.value}；"
            "反量化之后的 GEMM 用 f32 累加，混精度入口不在这一版"
        )
    if x.rank < 2:
        raise ValueError(f"linear_fp8 的 x 至少是 2 维，得到秩 {x.rank}")
    if qweight.dtype not in _FP8_TYPES:
        raise TypeError(
            f"linear_fp8 的权重必须是 fp8（f8e4m3 / f8e5m2），"
            f"得到 {qweight.dtype.value}"
        )
    if qweight.rank != 2:
        raise ValueError(f"linear_fp8 的量化权重必须是 2 维 (K, N)，得到 {qweight.shape}")
    if scale.dtype != DType.F32:
        raise TypeError(f"linear_fp8 的 weight_scale 必须是 f32，得到 {scale.dtype.value}")
    _check_scale_broadcast(qweight, scale, "linear_fp8 的 weight_scale")

    k, n = qweight.shape
    kx = x.shape[-1]
    if isinstance(kx, int) and isinstance(k, int):
        if kx != k:
            raise ValueError(
                f"linear_fp8 的内维不匹配：x 的最后一维 {kx} 与权重的 K={k}"
            )
    elif kx != k:
        raise ValueError(f"linear_fp8 的内维符号不一致：x={kx} 与权重 K={k}")

    _check_bias(inputs, 3, n, "linear_fp8")
    return TensorType(DType.F32, x.shape[:-1] + (n,))


def _ref_linear_fp8(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x, qweight, scale = args[0], args[1], args[2]
    if float(np.min(scale)) <= 0.0:
        raise ValueError(
            f"linear_fp8 的 weight_scale 必须严格为正，得到最小值 {float(np.min(scale))}"
        )
    weight = qweight.astype(np.float32) * scale
    out = np.matmul(x, weight)
    if len(args) > 3:
        out = out + args[3]
    return out


register(
    Op(
        name="linear_fp8",
        infer=_infer_linear_fp8,
        reference=_ref_linear_fp8,
        doc="weight-only fp8 线性层：(..., K) x f8(K, N) -> (..., N)。"
        "权重按 weight_scale 反量化后在 f32 上做 GEMM，数值等于"
        "``matmul(x, qweight * weight_scale) [+ bias]``；"
        "省的是权重占用与读带宽，不是算力（1 字节存储见 5.328）",
    )
)


def _infer_linear_fp8a8(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) not in (4, 5):
        raise ValueError(
            "linear_fp8a8 需要 (x_q, x_scale, w_q, w_scale[, bias]) 输入，"
            f"得到 {len(inputs)}"
        )
    xq, xs, wq, ws = inputs[0], inputs[1], inputs[2], inputs[3]
    _check_same_fp8(xq, wq, "linear_fp8a8")
    if xq.rank < 2:
        raise ValueError(f"linear_fp8a8 的激活至少是 2 维，得到秩 {xq.rank}")
    if wq.rank != 2:
        raise ValueError(f"linear_fp8a8 的量化权重必须是 2 维 (K, N)，得到 {wq.shape}")
    if xs.dtype != DType.F32:
        raise TypeError(f"linear_fp8a8 的激活 scale 必须是 f32，得到 {xs.dtype.value}")
    if ws.dtype != DType.F32:
        raise TypeError(f"linear_fp8a8 的权重 scale 必须是 f32，得到 {ws.dtype.value}")

    k, n = wq.shape
    kx = xq.shape[-1]
    if isinstance(kx, int) and isinstance(k, int):
        if kx != k:
            raise ValueError(
                f"linear_fp8a8 的内维不匹配：激活的最后一维 {kx} 与权重的 K={k}"
            )
    elif kx != k:
        raise ValueError(f"linear_fp8a8 的内维符号不一致：激活 {kx} 与权重 K={k}")

    _check_scale_broadcast(xq, xs, "linear_fp8a8 的激活 scale")
    # 反量化后再做 f32 GEMM，所以不要求 scale 沿压缩轴恒定；但仍要求
    # "每个 token 一个激活 scale、每个输出通道一个权重 scale"——这是写入侧
    # （quantize_activation_fp8 / quantize_weight_fp8）产出的形态，允许更宽的
    # 逐元素 scale 只会让设备端无法按行 / 按列取，属于未定义的模糊地带。
    if xs.rank > 0 and xs.shape[-1] != 1:
        raise ValueError(
            f"linear_fp8a8 的激活 scale 形状 {xs.shape} 沿 K 维不是 1："
            "需要每个 token 一个 scale（如 (B, T, 1) 或标量）"
        )
    if ws.rank > 1:
        raise ValueError(
            f"linear_fp8a8 的权重 scale 必须是逐输出通道 (N,) 或逐张量标量，"
            f"得到秩 {ws.rank} 形状 {ws.shape}"
        )
    if ws.rank == 1 and ws.shape[0] != 1 and ws.shape[0] != n:
        raise ValueError(f"linear_fp8a8 的权重 scale 长度 {ws.shape[0]} 与 N={n} 不一致")

    _check_bias(inputs, 4, n, "linear_fp8a8")
    return TensorType(DType.F32, xq.shape[:-1] + (n,))


def _ref_linear_fp8a8(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    xq, xs, wq, ws = args[0], args[1], args[2], args[3]
    if float(np.min(xs)) <= 0.0 or float(np.min(ws)) <= 0.0:
        raise ValueError(
            "linear_fp8a8 的激活 / 权重 scale 必须严格为正，得到 "
            f"{float(np.min(xs))} 与 {float(np.min(ws))}"
        )
    x = xq.astype(np.float32) * xs
    weight = wq.astype(np.float32) * ws
    out = np.matmul(x, weight)
    if len(args) > 4:
        out = out + args[4]
    return out


register(
    Op(
        name="linear_fp8a8",
        infer=_infer_linear_fp8a8,
        reference=_ref_linear_fp8a8,
        doc="W8A8 fp8 线性层：(..., K) f8、双 scale -> (..., N)。"
        "激活与权重各自按 scale 反量化后做 f32 GEMM，数值等于"
        "``matmul(x_q * x_scale, w_q * w_scale) [+ bias]``；"
        "不承诺 fp8 tensor core，累加仍是 f32",
    )
)


# ---------------------------------------------------------------------------
# fp8 的 1 字节存储（5.328）
#
# ``linear_fp8`` 只把数值落到 fp8 格点，IR 里的载体仍是 f32：省的是写入侧
# 的精度的描述，不省驻留与带宽。要真的降到 1/4，需要把 fp8 存成 8 位字节：
#
# - ``pack_fp8``：f8 载体 -> i8 位模式（每元素 1 字节）；
# - ``unpack_fp8``：位模式 -> f8 载体（精确，无信息损失）；
# - ``linear_fp8_packed``：直接吃 i8 位模式的 weight-only 线性层，设备端在
#   寄存器里解码，不物化 f32 权重。
#
# 位模式用 ``i8`` 承载：IR 没有 u8，而 ``signed char`` 正好 8 位。它不代表
# 整数，只代表 fp8 的字节；参考语义按位转成 uint8 再解码，因此 int8 的负数
# 不会参与任何数值运算。
# ---------------------------------------------------------------------------

_FP8_KIND_DTYPE: Dict[str, DType] = {"e4m3": DType.F8E4M3, "e5m2": DType.F8E5M2}


def _fp8_kind_attr(attrs: dict, what: str) -> str:
    kind = attrs.get("kind")
    if kind not in _FP8_KIND_DTYPE:
        raise ValueError(f"{what} 需要 attr kind='e4m3' 或 'e5m2'，得到 {kind!r}")
    return kind


def _infer_pack_fp8(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 1:
        raise ValueError(f"pack_fp8 需要 (x,) 输入，得到 {len(inputs)}")
    kind = _fp8_kind_attr(attrs, "pack_fp8")
    x = inputs[0]
    expected = _FP8_KIND_DTYPE[kind]
    if x.dtype != expected:
        raise TypeError(
            f"pack_fp8 的输入 dtype 与 kind={kind} 不一致："
            f"期望 {expected.value}，得到 {x.dtype.value}"
        )
    return TensorType(DType.I8, x.shape)


def _ref_pack_fp8(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    from ..runtime.fp8 import pack_fp8

    kind = _fp8_kind_attr(attrs, "pack_fp8")
    return pack_fp8(args[0], kind).view(np.int8)


register(
    Op(
        name="pack_fp8",
        infer=_infer_pack_fp8,
        reference=_ref_pack_fp8,
        attrs_schema=("kind",),
        doc="f8 载体 -> i8 位模式：每元素 1 字节，数值与 ``cast`` 相同。"
        "``kind`` 必须与输入 dtype 一致；往返由 ``unpack_fp8`` 闭合",
    )
)


def _infer_unpack_fp8(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 1:
        raise ValueError(f"unpack_fp8 需要 (q,) 输入，得到 {len(inputs)}")
    kind = _fp8_kind_attr(attrs, "unpack_fp8")
    q = inputs[0]
    if q.dtype != DType.I8:
        raise TypeError(
            f"unpack_fp8 的输入必须是 i8 位模式（用 pack_fp8 生成），"
            f"得到 {q.dtype.value}"
        )
    return TensorType(_FP8_KIND_DTYPE[kind], q.shape)


def _ref_unpack_fp8(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    from ..runtime.fp8 import unpack_fp8

    kind = _fp8_kind_attr(attrs, "unpack_fp8")
    codes = np.ascontiguousarray(args[0], dtype=np.int8).view(np.uint8)
    return unpack_fp8(codes, kind)


register(
    Op(
        name="unpack_fp8",
        infer=_infer_unpack_fp8,
        reference=_ref_unpack_fp8,
        attrs_schema=("kind",),
        doc="i8 位模式 -> f8 载体：逐位精确（含 ±inf / NaN 与次正规）。"
        "主机侧等价 ``pack_fp8`` 的逆运算，设备端有对应内核",
    )
)


def _infer_linear_fp8_packed(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    if len(inputs) not in (3, 4):
        raise ValueError(
            "linear_fp8_packed 需要 (x, qweight, weight_scale[, bias]) 输入，"
            f"得到 {len(inputs)}"
        )
    _fp8_kind_attr(attrs, "linear_fp8_packed")
    x, qweight, scale = inputs[0], inputs[1], inputs[2]
    if x.dtype != DType.F32:
        raise TypeError(
            f"linear_fp8_packed 的 x 必须是 f32，得到 {x.dtype.value}；"
            "反量化之后的 GEMM 用 f32 累加，混精度入口不在这一版"
        )
    if x.rank < 2:
        raise ValueError(f"linear_fp8_packed 的 x 至少是 2 维，得到秩 {x.rank}")
    if qweight.dtype != DType.I8:
        raise TypeError(
            f"linear_fp8_packed 的权重必须是 i8 位模式（用 pack_fp8 生成），"
            f"得到 {qweight.dtype.value}；已解码的 f8 载体请用 linear_fp8"
        )
    if qweight.rank != 2:
        raise ValueError(
            f"linear_fp8_packed 的量化权重必须是 2 维 (K, N)，得到 {qweight.shape}"
        )
    if scale.dtype != DType.F32:
        raise TypeError(
            f"linear_fp8_packed 的 weight_scale 必须是 f32，得到 {scale.dtype.value}"
        )

    k, n = qweight.shape
    kx = x.shape[-1]
    if isinstance(kx, int) and isinstance(k, int):
        if kx != k:
            raise ValueError(
                f"linear_fp8_packed 的内维不匹配：x 的最后一维 {kx} 与权重的 K={k}"
            )
    elif kx != k:
        raise ValueError(f"linear_fp8_packed 的内维符号不一致：x={kx} 与权重 K={k}")

    # 设备内核按列取 scale（``ws_stride`` 只有 0 / 1 两种）。逐组 scale
    # 需要把 K 方向折回也带进内核，属于下一阶段；这里显式拒绝而不是让
    # 参考后端悄悄支持、设备端再报错（不静默降级）。
    if scale.rank > 1:
        raise ValueError(
            f"linear_fp8_packed 的 weight_scale 只能是逐输出通道 (N,) 或逐张量标量，"
            f"得到秩 {scale.rank} 形状 {scale.shape}"
        )
    if scale.rank == 1 and scale.shape[0] != 1 and scale.shape[0] != n:
        raise ValueError(
            f"linear_fp8_packed 的 weight_scale 长度 {scale.shape[0]} 与 N={n} 不一致"
        )

    _check_bias(inputs, 3, n, "linear_fp8_packed")
    return TensorType(DType.F32, x.shape[:-1] + (n,))


def _ref_linear_fp8_packed(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    from ..runtime.fp8 import unpack_fp8

    kind = _fp8_kind_attr(attrs, "linear_fp8_packed")
    x, qweight, scale = args[0], args[1], args[2]
    if float(np.min(scale)) <= 0.0:
        raise ValueError(
            "linear_fp8_packed 的 weight_scale 必须严格为正，"
            f"得到最小值 {float(np.min(scale))}"
        )
    codes = np.ascontiguousarray(qweight, dtype=np.int8).view(np.uint8)
    weight = unpack_fp8(codes, kind) * scale
    out = np.matmul(x, weight)
    if len(args) > 3:
        out = out + args[3]
    return out


register(
    Op(
        name="linear_fp8_packed",
        infer=_infer_linear_fp8_packed,
        reference=_ref_linear_fp8_packed,
        attrs_schema=("kind",),
        doc="weight-only fp8 线性层，权重以 i8 位模式驻留：(..., K) x i8(K, N)"
        " -> (..., N)。数值等于 ``matmul(x, unpack_fp8(q) * weight_scale) [+ bias]``；"
        "省的是权重驻留与读带宽（1/4），不是算力（没有 fp8 tensor core）",
    )
)


# --------------------------------------------------------------------------
# fp8 逐组 scale 与 W8A8 packed 激活（5.342）
# --------------------------------------------------------------------------
#
# 5.328 把 fp8 的权重压到 1 字节，但 scale 只接逐张量 / 逐输出通道；逐组
# （``(K/G, N)``，GPTQ / AWQ 的 group_size 形态）在构图期显式拒绝，留给了本
# 阶段。这里的三个算子补齐缺口：
#
# - ``dequantize_fp8_group``：组内共用一份 scale 的 fp8 反量化，与
#   ``dequantize_i8_group`` 同形同口径；
# - ``linear_fp8_group``：逐组 weight-only 线性层，权重仍是 i8 位模式；
# - ``linear_fp8a8_packed``：**两端都打包**的 W8A8——激活与权重各以 1 字节
#   码驻留，设备端在寄存器里解码后做 f32 累加。它才是 5.329 遗留的"packed
#   激活"：此前的 ``linear_fp8a8`` 激活仍是 f32 载体。
#
# 三者都不承诺 fp8 tensor core：累加仍是 f32，收益只在驻留与读带宽。


def _infer_dequantize_fp8_group(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    if len(inputs) != 2:
        raise ValueError(
            f"dequantize_fp8_group 需要 (q, scale) 两个输入，得到 {len(inputs)}"
        )
    _fp8_kind_attr(attrs, "dequantize_fp8_group")
    q, scale = inputs
    if q.dtype != DType.I8:
        raise TypeError(
            f"dequantize_fp8_group 的第一个输入必须是 i8 位模式（用 pack_fp8 生成），"
            f"得到 {q.dtype.value}"
        )
    if q.rank != 2:
        raise ValueError(
            f"dequantize_fp8_group 的 q 必须是二维 (K, N)，得到秩 {q.rank}"
        )
    if scale.dtype != DType.F32:
        raise TypeError(
            f"dequantize_fp8_group 的 scale 必须是 f32，得到 {scale.dtype.value}"
        )
    _check_group_params(q, scale, "dequantize_fp8_group", "scale")
    # 反量化的结果是真实数值，不是格式格点上的载体：输出 f32。若沿用 f8
    # 载体，执行时会在节点边界把 ``unpack(q) * scale`` 二次舍入回 fp8 格点
    # （scale 不是 2 的幂时可见），丢掉刚乘上去的精度。
    return TensorType(DType.F32, q.shape)


def _ref_dequantize_fp8_group(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    from ..runtime.fp8 import unpack_fp8

    kind = _fp8_kind_attr(attrs, "dequantize_fp8_group")
    q, scale = args
    if float(np.min(scale)) <= 0.0:
        raise ValueError(
            "dequantize_fp8_group 的 scale 必须严格为正，"
            f"得到最小值 {float(np.min(scale))}"
        )
    codes = np.ascontiguousarray(q, dtype=np.int8).view(np.uint8)
    repeated = np.repeat(scale, q.shape[0] // scale.shape[0], axis=0)
    return unpack_fp8(codes, kind) * repeated


register(
    Op(
        name="dequantize_fp8_group",
        infer=_infer_dequantize_fp8_group,
        reference=_ref_dequantize_fp8_group,
        attrs_schema=("kind",),
        doc="逐组 fp8 反量化：i8 位模式 (K, N) 沿输入维分成 K/G 组，scale "
        "(K/G, N) 组内共用（GPTQ / AWQ 的 group_size 形态）；组数由形状推出，"
        "不另设 attr。数值是 ``unpack_fp8(q) * repeat(scale, G)``，输出 f32",
    )
)


def _infer_linear_fp8_group(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    if len(inputs) not in (3, 4):
        raise ValueError(
            "linear_fp8_group 需要 (x, qweight, weight_scale[, bias]) 输入，"
            f"得到 {len(inputs)}"
        )
    _fp8_kind_attr(attrs, "linear_fp8_group")
    x, qweight, scale = inputs[0], inputs[1], inputs[2]
    if x.dtype != DType.F32:
        raise TypeError(
            f"linear_fp8_group 的 x 必须是 f32，得到 {x.dtype.value}；"
            "反量化之后的 GEMM 用 f32 累加，混精度入口不在这一版"
        )
    if x.rank < 2:
        raise ValueError(f"linear_fp8_group 的 x 至少是 2 维，得到秩 {x.rank}")
    if qweight.dtype != DType.I8:
        raise TypeError(
            f"linear_fp8_group 的权重必须是 i8 位模式（用 pack_fp8 生成），"
            f"得到 {qweight.dtype.value}；已解码的 f8 载体请用 linear_fp8"
        )
    if qweight.rank != 2:
        raise ValueError(
            f"linear_fp8_group 的量化权重必须是 2 维 (K, N)，得到 {qweight.shape}"
        )
    if scale.dtype != DType.F32:
        raise TypeError(
            f"linear_fp8_group 的 weight_scale 必须是 f32，得到 {scale.dtype.value}"
        )

    k, n = qweight.shape
    kx = x.shape[-1]
    if isinstance(kx, int) and isinstance(k, int):
        if kx != k:
            raise ValueError(
                f"linear_fp8_group 的内维不匹配：x 的最后一维 {kx} 与权重的 K={k}"
            )
    elif kx != k:
        raise ValueError(f"linear_fp8_group 的内维符号不一致：x={kx} 与权重 K={k}")

    _check_group_params(qweight, scale, "linear_fp8_group", "weight_scale")
    _check_bias(inputs, 3, n, "linear_fp8_group")
    return TensorType(DType.F32, x.shape[:-1] + (n,))


def _ref_linear_fp8_group(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    from ..runtime.fp8 import unpack_fp8

    kind = _fp8_kind_attr(attrs, "linear_fp8_group")
    x, qweight, scale = args[0], args[1], args[2]
    if float(np.min(scale)) <= 0.0:
        raise ValueError(
            "linear_fp8_group 的 weight_scale 必须严格为正，"
            f"得到最小值 {float(np.min(scale))}"
        )
    codes = np.ascontiguousarray(qweight, dtype=np.int8).view(np.uint8)
    repeated = np.repeat(scale, qweight.shape[0] // scale.shape[0], axis=0)
    weight = unpack_fp8(codes, kind) * repeated
    out = np.matmul(x, weight)
    if len(args) > 3:
        out = out + args[3]
    return out


register(
    Op(
        name="linear_fp8_group",
        infer=_infer_linear_fp8_group,
        reference=_ref_linear_fp8_group,
        attrs_schema=("kind",),
        doc="逐组 weight-only fp8 线性层：(..., K) x i8(K, N) -> (..., N)，"
        "权重沿输入维每 K/G 行共用一份 scale（(K/G, N)）。数值等于 "
        "``matmul(x, unpack_fp8(q) * repeat(scale, G)) [+ bias]``；"
        "省的是权重驻留与读带宽，不是算力",
    )
)


def _infer_linear_fp8a8_packed(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    if len(inputs) not in (4, 5):
        raise ValueError(
            "linear_fp8a8_packed 需要 (x_q, x_scale, w_q, w_scale[, bias]) 输入，"
            f"得到 {len(inputs)}"
        )
    _fp8_kind_attr(attrs, "linear_fp8a8_packed")
    xq, xs, wq, ws = inputs[0], inputs[1], inputs[2], inputs[3]
    if xq.dtype != DType.I8:
        raise TypeError(
            f"linear_fp8a8_packed 的激活必须是 i8 位模式（用 pack_fp8 生成），"
            f"得到 {xq.dtype.value}；f8 载体的激活请用 linear_fp8a8"
        )
    if wq.dtype != DType.I8:
        raise TypeError(
            f"linear_fp8a8_packed 的权重必须是 i8 位模式，得到 {wq.dtype.value}"
        )
    if xs.dtype != DType.F32:
        raise TypeError(
            f"linear_fp8a8_packed 的激活 scale 必须是 f32，得到 {xs.dtype.value}"
        )
    if ws.dtype != DType.F32:
        raise TypeError(
            f"linear_fp8a8_packed 的权重 scale 必须是 f32，得到 {ws.dtype.value}"
        )
    if xq.rank < 2:
        raise ValueError(f"linear_fp8a8_packed 的激活至少是 2 维，得到秩 {xq.rank}")
    if wq.rank != 2:
        raise ValueError(
            f"linear_fp8a8_packed 的量化权重必须是 2 维 (K, N)，得到 {wq.shape}"
        )

    k, n = wq.shape
    kx = xq.shape[-1]
    if isinstance(kx, int) and isinstance(k, int):
        if kx != k:
            raise ValueError(
                f"linear_fp8a8_packed 的内维不匹配：激活的最后一维 {kx} 与权重的 K={k}"
            )
    elif kx != k:
        raise ValueError(f"linear_fp8a8_packed 的内维符号不一致：激活 {kx} 与权重 K={k}")

    _check_scale_broadcast(xq, xs, "linear_fp8a8_packed 的激活 scale")
    # f32 累加不要求 scale 能提到求和号之外，但设备内核按行取 scale（每行
    # 一个），沿 K 维变化的形态没有对应的读法；与 linear_fp8a8 同一口径。
    if xs.rank > 0 and xs.shape[-1] != 1:
        raise ValueError(
            f"linear_fp8a8_packed 的激活 scale 必须沿 K 维恒定"
            f"（形状 {xq.shape[:-1]}+(1,) 或标量），得到 {xs.shape}"
        )
    if ws.rank > 1:
        raise ValueError(
            f"linear_fp8a8_packed 的权重 scale 必须是逐输出通道 (N,) 或逐张量标量，"
            f"得到秩 {ws.rank} 形状 {ws.shape}"
        )
    if ws.rank == 1 and ws.shape[0] != 1 and ws.shape[0] != n:
        raise ValueError(
            f"linear_fp8a8_packed 的权重 scale 长度 {ws.shape[0]} 与 N={n} 不一致"
        )

    _check_bias(inputs, 4, n, "linear_fp8a8_packed")
    return TensorType(DType.F32, xq.shape[:-1] + (n,))


def _ref_linear_fp8a8_packed(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    from ..runtime.fp8 import unpack_fp8

    kind = _fp8_kind_attr(attrs, "linear_fp8a8_packed")
    xq, xs, wq, ws = args[0], args[1], args[2], args[3]
    if float(np.min(xs)) <= 0.0:
        raise ValueError(
            f"linear_fp8a8_packed 的激活 scale 必须严格为正，得到最小值 {float(np.min(xs))}"
        )
    if float(np.min(ws)) <= 0.0:
        raise ValueError(
            f"linear_fp8a8_packed 的权重 scale 必须严格为正，得到最小值 {float(np.min(ws))}"
        )
    xcodes = np.ascontiguousarray(xq, dtype=np.int8).view(np.uint8)
    wcodes = np.ascontiguousarray(wq, dtype=np.int8).view(np.uint8)
    xd = unpack_fp8(xcodes, kind) * np.broadcast_to(xs, xq.shape)
    wd = unpack_fp8(wcodes, kind) * ws
    out = np.matmul(xd, wd)
    if len(args) > 4:
        out = out + args[4]
    return out


register(
    Op(
        name="linear_fp8a8_packed",
        infer=_infer_linear_fp8a8_packed,
        reference=_ref_linear_fp8a8_packed,
        attrs_schema=("kind",),
        doc="两端都打包的 fp8 W8A8 线性层：i8 位模式激活 (..., K) 与 i8 位模式"
        "权重 (K, N)，各配一份 scale，数值等于 "
        "``matmul(unpack(x_q) * x_scale, unpack(w_q) * w_scale) [+ bias]``。"
        "激活与权重的驻留都是 1 字节；累加仍是 f32，不承诺 fp8 tensor core",
    )
)


# --------------------------------------------------------------------------
# int4 逐组量化（5.343）
# --------------------------------------------------------------------------
#
# 4 bit 的承载方式与 fp8 一致：**值**用 i8 张量装（值域 0..15），**打包后的
# 字节**也用 i8 装位模式。不新增 u4 dtype 的理由：IR 的 dtype 决定存储宽度与
# 内核签名，加一个 0.5 字节类型会让所有既有内核都要多一条分支，而"4 bit
# 打包"本质是存储形态（与 5.328 的 fp8 pack / unpack 同构），放在算子层更
# 清楚。
#
# 语义是非对称逐组：``w ≈ (q - zp) * scale``，``q ∈ [0, 15]``。对称是
# ``zp = 8`` 的特例，不做成两个算子：AWQ / GPTQ 的 4 bit 导出普遍逐组带
# 零点，而对称在这里没有可复用的数值结构（缩放不是 2 的幂）。
#
# 这一版同样**不承诺算力**：``linear_u4_group`` 是"反量化成 f32 再 GEMM"，
# 省的是权重占用（打包后 0.5 字节 / 权重）与读带宽，不是 int4 tensor core。

_U4_MAX = 15


def _check_u4_values(q: np.ndarray, what: str) -> None:
    """4 bit 值域检查。越界值多半意味着把 i8 权重误喂给了 u4 算子。"""
    lo = float(np.min(q))
    hi = float(np.max(q))
    if lo < 0.0 or hi > float(_U4_MAX):
        raise ValueError(
            f"{what} 的 4 bit 值必须在 [0, 15] 内，得到 [{lo}, {hi}]；"
            "越界值通常意味着误用了 i8 量化权重"
        )


def _infer_pack_int4(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 1:
        raise ValueError(f"pack_int4 只接受 1 个输入，得到 {len(inputs)}")
    q = inputs[0]
    if q.dtype != DType.I8:
        raise TypeError(
            f"pack_int4 的输入必须是 i8（承载 0..15 的 4 bit 值），"
            f"得到 {q.dtype.value}"
        )
    if q.rank < 1:
        raise ValueError(f"pack_int4 至少需要 1 维，得到秩 {q.rank}")
    n = q.shape[-1]
    if not isinstance(n, int):
        raise ValueError(
            f"pack_int4 的最后一维必须是静态的（打包要求偶数长度），得到 {q.shape}"
        )
    if n % 2 != 0:
        raise ValueError(
            f"pack_int4 要求最后一维为偶数，得到 {n}：每个字节要装两个 4 bit 值"
        )
    return TensorType(DType.I8, q.shape[:-1] + (n // 2,))


def _ref_pack_int4(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    # 位序约定：偶数下标进低 4 位，奇数下标进高 4 位。与 unpack_int4 互为逆。
    q = np.asarray(args[0], dtype=np.int8)
    _check_u4_values(q, "pack_int4")
    u = q.astype(np.uint8)
    packed = np.ascontiguousarray(
        (u[..., 0::2] | ((u[..., 1::2] << 4) & 0xF0)).astype(np.uint8)
    )
    return packed.view(np.int8)


register(
    Op(
        name="pack_int4",
        infer=_infer_pack_int4,
        reference=_ref_pack_int4,
        doc="把 i8 承载的 4 bit 值 (..., N) 打包成 (..., N/2) 个字节的 i8 位模式："
        "偶数下标存低 4 位、奇数下标存高 4 位。N 必须为偶数；输入值域 0..15",
    )
)


def _infer_unpack_int4(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 1:
        raise ValueError(f"unpack_int4 只接受 1 个输入，得到 {len(inputs)}")
    packed = inputs[0]
    if packed.dtype != DType.I8:
        raise TypeError(
            f"unpack_int4 的输入必须是 i8 位模式，得到 {packed.dtype.value}"
        )
    if packed.rank < 1:
        raise ValueError(f"unpack_int4 至少需要 1 维，得到秩 {packed.rank}")
    n = packed.shape[-1]
    if not isinstance(n, int):
        raise ValueError(
            f"unpack_int4 的最后一维必须是静态的，得到 {packed.shape}"
        )
    return TensorType(DType.I8, packed.shape[:-1] + (2 * n,))


def _ref_unpack_int4(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    p = np.asarray(args[0], dtype=np.int8).astype(np.uint8)
    out = np.empty(p.shape[:-1] + (2 * p.shape[-1],), dtype=np.uint8)
    out[..., 0::2] = p & 0x0F
    out[..., 1::2] = (p >> 4) & 0x0F
    return out.view(np.int8)


register(
    Op(
        name="unpack_int4",
        infer=_infer_unpack_int4,
        reference=_ref_unpack_int4,
        doc="把 i8 位模式 (..., N/2) 解成 i8 承载的 4 bit 值 (..., N)："
        "低 4 位回偶数下标、高 4 位回奇数下标，输出值域 0..15",
    )
)


def _infer_dequantize_u4_group(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    if len(inputs) != 3:
        raise ValueError(
            "dequantize_u4_group 需要 (q, scale, zero_point) 三个输入，"
            f"得到 {len(inputs)}"
        )
    q, scale, zp = inputs
    if q.dtype != DType.I8:
        raise TypeError(
            f"dequantize_u4_group 的 q 必须是 i8（承载 0..15），得到 {q.dtype.value}"
        )
    if q.rank != 2:
        raise ValueError(
            f"dequantize_u4_group 的 q 必须是二维 (K, N)，得到秩 {q.rank}"
        )
    if scale.dtype != DType.F32:
        raise TypeError(
            f"dequantize_u4_group 的 scale 必须是 f32，得到 {scale.dtype.value}"
        )
    if zp.dtype != DType.F32:
        raise TypeError(
            f"dequantize_u4_group 的 zero_point 必须是 f32（4 bit 零点 0..15），"
            f"得到 {zp.dtype.value}"
        )
    _check_group_params(q, scale, "dequantize_u4_group", "scale")
    if tuple(zp.shape) != tuple(scale.shape):
        raise ValueError(
            f"dequantize_u4_group 的 zero_point 形状 {zp.shape} "
            f"必须与 scale {scale.shape} 相同（同一组共用一对参数）"
        )
    return TensorType(DType.F32, q.shape)


def _ref_dequantize_u4_group(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    q, scale, zp = args
    _check_u4_values(q, "dequantize_u4_group")
    if float(np.min(scale)) <= 0.0:
        raise ValueError(
            f"dequantize_u4_group 的 scale 必须严格为正，"
            f"得到最小值 {float(np.min(scale))}"
        )
    if float(np.min(zp)) < 0.0 or float(np.max(zp)) > float(_U4_MAX):
        raise ValueError(
            f"dequantize_u4_group 的 zero_point 必须在 [0, 15] 内，"
            f"得到 [{float(np.min(zp))}, {float(np.max(zp))}]"
        )
    g = q.shape[0] // scale.shape[0]
    # 减法留在 f32 里做：q 只有 0..15，与 zp 都是精确可表示的整数，
    # 因此与设备内核 ``(float)q - zp`` 的顺序逐位一致。
    diff = q.astype(np.float32) - np.repeat(zp, g, axis=0)
    return diff * np.repeat(scale, g, axis=0)


register(
    Op(
        name="dequantize_u4_group",
        infer=_infer_dequantize_u4_group,
        reference=_ref_dequantize_u4_group,
        doc="逐组非对称 4 bit 反量化：(q - zero_point) * scale，q (K, N) 值域 "
        "0..15，scale / zero_point 形状都是 (K/G, N)，组内共用。"
        "对称 4 bit 是 zero_point=8 的特例；组大小由形状推出，不另设 attr",
    )
)


def _infer_linear_u4_group(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    if len(inputs) not in (4, 5):
        raise ValueError(
            "linear_u4_group 需要 (x, qweight, scale, zero_point[, bias]) 输入，"
            f"得到 {len(inputs)}"
        )
    x, qweight, scale, zp = inputs[0], inputs[1], inputs[2], inputs[3]
    if x.dtype != DType.F32:
        raise TypeError(
            f"linear_u4_group 的 x 必须是 f32，得到 {x.dtype.value}；"
            "反量化之后的 GEMM 用 f32 累加，混精度入口不在这一版"
        )
    if x.rank < 2:
        raise ValueError(f"linear_u4_group 的 x 至少是 2 维，得到秩 {x.rank}")
    if qweight.dtype != DType.I8:
        raise TypeError(
            f"linear_u4_group 的权重必须是 i8（承载 0..15），"
            f"得到 {qweight.dtype.value}"
        )
    if qweight.rank != 2:
        raise ValueError(
            f"linear_u4_group 的量化权重必须是 2 维 (K, N)，得到 {qweight.shape}"
        )
    if scale.dtype != DType.F32:
        raise TypeError(f"linear_u4_group 的 scale 必须是 f32，得到 {scale.dtype.value}")
    if zp.dtype != DType.F32:
        raise TypeError(
            f"linear_u4_group 的 zero_point 必须是 f32，得到 {zp.dtype.value}"
        )
    _check_group_params(qweight, scale, "linear_u4_group", "scale")
    if tuple(zp.shape) != tuple(scale.shape):
        raise ValueError(
            f"linear_u4_group 的 zero_point 形状 {zp.shape} "
            f"必须与 scale {scale.shape} 相同（同一组共用一对参数）"
        )

    k, n = qweight.shape
    kx = x.shape[-1]
    if isinstance(kx, int) and isinstance(k, int):
        if kx != k:
            raise ValueError(
                f"linear_u4_group 的内维不匹配：x 的最后一维 {kx} 与权重的 K={k}"
            )
    elif kx != k:
        raise ValueError(f"linear_u4_group 的内维符号不一致：x={kx} 与权重 K={k}")

    _check_bias(inputs, 4, n, "linear_u4_group")
    return TensorType(DType.F32, x.shape[:-1] + (n,))


def _ref_linear_u4_group(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x, qweight, scale, zp = args[0], args[1], args[2], args[3]
    _check_u4_values(qweight, "linear_u4_group")
    if float(np.min(scale)) <= 0.0:
        raise ValueError(
            f"linear_u4_group 的 scale 必须严格为正，得到最小值 {float(np.min(scale))}"
        )
    if float(np.min(zp)) < 0.0 or float(np.max(zp)) > float(_U4_MAX):
        raise ValueError(
            f"linear_u4_group 的 zero_point 必须在 [0, 15] 内，"
            f"得到 [{float(np.min(zp))}, {float(np.max(zp))}]"
        )
    g = qweight.shape[0] // scale.shape[0]
    w = (qweight.astype(np.float32) - np.repeat(zp, g, axis=0)) * np.repeat(
        scale, g, axis=0
    )
    out = np.matmul(x, w)
    if len(args) > 4:
        out = out + args[4]
    return out


register(
    Op(
        name="linear_u4_group",
        infer=_infer_linear_u4_group,
        reference=_ref_linear_u4_group,
        doc="weight-only 4 bit 线性层，权重按组共用 (scale, zero_point)："
        "(..., K) x i8(K, N) -> (..., N)，语义等于 "
        "``matmul(x, (qweight - zp) * scale) [+ bias]``，参数形状 (K/G, N)。"
        "与 ``linear_i8_group`` 的差别只有权重位宽与是否需要零点；"
        "反量化后仍是 f32 GEMM，不承诺 int4 tensor core",
    )
)


def _infer_linear_i8a8(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) not in (4, 5):
        raise ValueError(
            "linear_i8a8 需要 (x_q, x_scale, w_q, w_scale[, bias]) 输入，"
            f"得到 {len(inputs)}"
        )
    xq, xs, wq, ws = inputs[0], inputs[1], inputs[2], inputs[3]
    if xq.dtype != DType.I8:
        raise TypeError(
            f"linear_i8a8 的激活必须是 i8，得到 {xq.dtype.value}；"
            "浮点激活请先走 quantize_i8，或改用 linear_i8"
        )
    if wq.dtype != DType.I8:
        raise TypeError(f"linear_i8a8 的权重必须是 i8，得到 {wq.dtype.value}")
    if xs.dtype != DType.F32:
        raise TypeError(f"linear_i8a8 的激活 scale 必须是 f32，得到 {xs.dtype.value}")
    if ws.dtype != DType.F32:
        raise TypeError(f"linear_i8a8 的权重 scale 必须是 f32，得到 {ws.dtype.value}")
    if xq.rank < 2:
        raise ValueError(f"linear_i8a8 的激活至少是 2 维，得到秩 {xq.rank}")
    if wq.rank != 2:
        raise ValueError(f"linear_i8a8 的量化权重必须是 2 维 (K, N)，得到 {wq.shape}")

    k, n = wq.shape
    kx = xq.shape[-1]
    if isinstance(kx, int) and isinstance(k, int):
        if kx != k:
            raise ValueError(
                f"linear_i8a8 的内维不匹配：激活的最后一维 {kx} 与权重的 K={k}"
            )
    elif kx != k:
        raise ValueError(f"linear_i8a8 的内维符号不一致：激活 {kx} 与权重 K={k}")

    _check_scale_broadcast(xq, xs, "linear_i8a8 的激活 scale")
    # int32 累加的前提是 scale 能提到求和之外：激活 scale 必须沿 K 维恒定
    # （每个 token 一个），权重 scale 必须沿 K 维恒定（每个输出通道一个）。
    # 沿压缩轴逐元素变化的 scale 让累加不再成立，直接拒绝而不是静默近似。
    if xs.rank > 0 and xs.shape[-1] != 1:
        raise ValueError(
            f"linear_i8a8 的激活 scale 形状 {xs.shape} 沿 K 维不是 1："
            "int32 累加要求每个 token 一个 scale（如 (B, T, 1) 或标量）"
        )
    if ws.rank > 1:
        raise ValueError(
            f"linear_i8a8 的权重 scale 必须是逐输出通道 (N,) 或逐张量标量，"
            f"得到秩 {ws.rank} 形状 {ws.shape}；沿 K 维逐元素变化的 scale "
            "无法用 int32 累加表达"
        )
    if ws.rank == 1 and ws.shape[0] != 1 and ws.shape[0] != n:
        raise ValueError(f"linear_i8a8 的权重 scale 长度 {ws.shape[0]} 与 N={n} 不一致")
    if isinstance(k, int) and k > _I32_ACC_LIMIT:
        raise ValueError(
            f"linear_i8a8 的 K={k} 超过 int32 累加上限 {_I32_ACC_LIMIT}："
            "K * 127 * 127 会溢出 int32，累加结果不再确定；请分段累加"
        )

    if len(inputs) == 5:
        bias = inputs[4]
        if bias.dtype != DType.F32:
            raise TypeError(f"linear_i8a8 的 bias 必须是 f32，得到 {bias.dtype.value}")
        if bias.rank != 1:
            raise ValueError(f"linear_i8a8 的 bias 必须是秩 1，得到秩 {bias.rank}")
        if isinstance(bias.shape[0], int) and isinstance(n, int) and bias.shape[0] != n:
            raise ValueError(f"linear_i8a8 的 bias 长度 {bias.shape[0]} 与 N={n} 不一致")
    return TensorType(DType.F32, xq.shape[:-1] + (n,))


def _ref_linear_i8a8(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    xq, xs, wq, ws = args[0], args[1], args[2], args[3]
    if float(np.min(xs)) <= 0.0:
        raise ValueError(
            f"linear_i8a8 的激活 scale 必须严格为正，得到最小值 {float(np.min(xs))}"
        )
    if float(np.min(ws)) <= 0.0:
        raise ValueError(
            f"linear_i8a8 的权重 scale 必须严格为正，得到最小值 {float(np.min(ws))}"
        )
    k = int(xq.shape[-1])
    if k > _I32_ACC_LIMIT:
        raise ValueError(
            f"linear_i8a8 的 K={k} 超过 int32 累加上限 {_I32_ACC_LIMIT}："
            "K * 127 * 127 会溢出 int32，累加结果不再确定；请分段累加"
        )
    # int32 累加是这一版与 weight-only 的唯一数值差别：先整数求和，再乘两个
    # scale 的积，而不是先反量化再在 f32 上逐项累加。
    acc = np.matmul(xq.astype(np.int32), wq.astype(np.int32))
    row = np.broadcast_to(xs, xq.shape)[..., :1].astype(np.float32)
    out = acc.astype(np.float32) * row * ws
    if len(args) > 4:
        out = out + args[4]
    return out


register(
    Op(
        name="linear_i8a8",
        infer=_infer_linear_i8a8,
        reference=_ref_linear_i8a8,
        doc="W8A8 线性层：(..., K) i8 激活与 (K, N) i8 权重先在 int32 里累加，"
        "再乘激活 scale（每个 token 一个）与权重 scale（每个输出通道一个），"
        "最后加 bias。语义是 ``(int32(x_q) @ int32(w_q)).f32() * xs * ws [+ b]``；"
        "K * 127 * 127 超过 int32 上界时显式拒绝，不做回绕",
    )
)


# --------------------------------------------------------------------------
# 非对称（带零点）int8 量化
# --------------------------------------------------------------------------
#
# 与对称版本共用舍入与饱和口径（RNE、scale 严格为正），差别只有零点：
#
# - 对称：``q = clip(rint(x / scale), -127, 127)``，0 恒映射到 0。
# - 非对称：``q = clip(rint(x / scale) + zp, -128, 127)``，反量化是
#   ``x = (q - zp) * scale``。值域用满整段 256 级，因此负端点是 -128：
#   两级都用来表示实际取值范围，而不是留一级维持对称性。
# - 零点以 **i8 张量**进图：取值天然落在 ``[-128, 127]``，不需要一个
#   "越界即拒绝"的哨兵，也不会出现"按 i32 传进来的零点超出 int8 桶"。
# - **非对称的收益在激活**：权重分布通常近对称，零点接近 0；ReLU 之后的
#   激活全为正，对称量化会把一半格子分给永远不会出现的负数。因此真正值得
#   走这条路的入口是 W8A8（``linear_i8a8_zp``）；weight-only 版本
#   （``linear_i8_zp``）主要给"导出模型里权重零点确实非零"的场景直接跑。
# - W8A8 的 int32 累加上界比对称版更紧：``xq - zp`` 与 ``wq - zp`` 各自
#   可取到 255，边界因此是 ``K * 255 * 255``，而不是 ``K * 127 * 127``。

#: 非对称量化的负端点。两级都用满，所以是 -128，不是对称版的 -127。
_QUANT_MIN_ZP = -128

#: 非对称 W8A8 的 int32 累加上界：``K * 255 * 255 <= 2**31 - 1``。
_I32_ACC_LIMIT_ZP = (2**31 - 1) // (255 * 255)


def _infer_quantize_i8_zp(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 3:
        raise ValueError(
            "quantize_i8_zp 需要 (x, scale, zero_point) 三个输入，"
            f"得到 {len(inputs)}"
        )
    x, scale, zp = inputs
    if x.dtype != DType.F32:
        raise TypeError(
            f"quantize_i8_zp 只接受 f32 输入，得到 {x.dtype.value}；"
            "bf16/f16 要先用 cast 转到 f32，量化语义只在 f32 上定义"
        )
    if scale.dtype != DType.F32:
        raise TypeError(f"quantize_i8_zp 的 scale 必须是 f32，得到 {scale.dtype.value}")
    if zp.dtype != DType.I8:
        raise TypeError(
            f"quantize_i8_zp 的 zero_point 必须是 i8，得到 {zp.dtype.value}；"
            "整数零点作为 i8 缓冲进图，取值天然落在 [-128, 127]"
        )
    if x.rank < 1:
        raise ValueError("quantize_i8_zp 至少需要 1 维输入")
    _check_broadcastable(x, scale, "quantize_i8_zp", "scale")
    _check_broadcastable(x, zp, "quantize_i8_zp", "zero_point")
    return TensorType(DType.I8, x.shape)


def _ref_quantize_i8_zp(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x, scale, zp = args
    if float(np.min(scale)) <= 0.0:
        raise ValueError(
            f"quantize_i8_zp 的 scale 必须严格为正，得到最小值 {float(np.min(scale))}"
        )
    # 先舍入到格点再平移：零点不是"在值域里加的偏移"，而是整数格点的平移，
    # 两种写法在 x/scale 不是整数时结果不同（差一个舍入）。
    q = np.rint(x / scale) + zp.astype(np.float32)
    return np.clip(q, _QUANT_MIN_ZP, _QUANT_MAX).astype(np.int8)


register(
    Op(
        name="quantize_i8_zp",
        infer=_infer_quantize_i8_zp,
        reference=_ref_quantize_i8_zp,
        doc="非对称 int8 量化：q = clip(rint(x / scale) + zp, -128, 127)，"
        "scale 与零点是 i8/f32 张量，都广播到 x；输出 i8 与 x 同形状。"
        "负端点用满 -128（对称版刻意止步于 -127）",
    )
)


def _infer_dequantize_i8_zp(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    if len(inputs) != 3:
        raise ValueError(
            "dequantize_i8_zp 需要 (q, scale, zero_point) 三个输入，"
            f"得到 {len(inputs)}"
        )
    q, scale, zp = inputs
    if q.dtype != DType.I8:
        raise TypeError(f"dequantize_i8_zp 的第一个输入必须是 i8，得到 {q.dtype.value}")
    if scale.dtype != DType.F32:
        raise TypeError(f"dequantize_i8_zp 的 scale 必须是 f32，得到 {scale.dtype.value}")
    if zp.dtype != DType.I8:
        raise TypeError(
            f"dequantize_i8_zp 的 zero_point 必须是 i8，得到 {zp.dtype.value}"
        )
    _check_broadcastable(q, scale, "dequantize_i8_zp", "scale")
    _check_broadcastable(q, zp, "dequantize_i8_zp", "zero_point")
    return TensorType(DType.F32, q.shape)


def _ref_dequantize_i8_zp(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    q, scale, zp = args
    # ``q - zp`` 先在 int32 里算：q=-128、zp=127 时差值是 -255，落在 i8 之外。
    diff = q.astype(np.int32) - zp.astype(np.int32)
    return diff.astype(np.float32) * scale


register(
    Op(
        name="dequantize_i8_zp",
        infer=_infer_dequantize_i8_zp,
        reference=_ref_dequantize_i8_zp,
        doc="非对称反量化：f32(q - zp) * scale，减法在 int32 里做；"
        "输出与 q 同形状的 f32",
    )
)


def _check_group_params(
    q: TensorType, param: TensorType, what: str, label: str
) -> None:
    """逐组量化参数：二维 ``(K/G, N)``，列数与 q 对齐，``K/G`` 必须整除 K。

    组大小不单独入图：``G = K / K_g`` 由两张量的形状唯一确定。多留一个 attr
    就等于多一份真值，两者对不上时没人知道该信谁。
    """
    if param.rank != 2:
        raise ValueError(
            f"{what} 的 {label} 必须是二维 (K/G, N)，得到秩 {param.rank}"
        )
    k, n = q.shape
    kg, pn = param.shape
    if pn != n:
        raise ValueError(
            f"{what} 的 {label} 形状 {param.shape} 的列数与 q 的输出维 {n} 不一致"
        )
    if not isinstance(k, int) or not isinstance(kg, int):
        raise ValueError(
            f"{what} 的分组要求静态的 K 与 K/G，得到 q {q.shape}、{label} {param.shape}"
        )
    if k <= 0 or kg <= 0:
        raise ValueError(f"{what} 的形状非法：q {q.shape}，{label} {param.shape}")
    if k % kg != 0:
        raise ValueError(
            f"{what} 的 {label} 有 {kg} 组，但 q 的输入维 K={k} 不能被它整除"
        )


def _infer_dequantize_i8_group(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    if len(inputs) != 2:
        raise ValueError(
            f"dequantize_i8_group 需要 (q, scale) 两个输入，得到 {len(inputs)}"
        )
    q, scale = inputs
    if q.dtype != DType.I8:
        raise TypeError(
            f"dequantize_i8_group 的第一个输入必须是 i8，得到 {q.dtype.value}"
        )
    if q.rank != 2:
        raise ValueError(
            f"dequantize_i8_group 的 q 必须是二维 (K, N)，得到秩 {q.rank}"
        )
    if scale.dtype != DType.F32:
        raise TypeError(
            f"dequantize_i8_group 的 scale 必须是 f32，得到 {scale.dtype.value}"
        )
    _check_group_params(q, scale, "dequantize_i8_group", "scale")
    return TensorType(DType.F32, q.shape)


def _ref_dequantize_i8_group(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    q, scale = args
    # 组内共用一份 scale：把 (K/G, N) 沿 K 重复 G 次再乘。
    repeated = np.repeat(scale, q.shape[0] // scale.shape[0], axis=0)
    return q.astype(np.float32) * repeated


register(
    Op(
        name="dequantize_i8_group",
        infer=_infer_dequantize_i8_group,
        reference=_ref_dequantize_i8_group,
        doc="逐组对称反量化：q (K, N) 沿输入维分成 K/G 组，scale (K/G, N) "
        "组内共用（GPTQ / AWQ 的 group_size 形态）；scale 的组数为 1 时等于逐通道，"
        "为 K 时等于逐行。组大小由形状推出，不另设 attr",
    )
)


def _infer_dequantize_i8_group_zp(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    if len(inputs) != 3:
        raise ValueError(
            "dequantize_i8_group_zp 需要 (q, scale, zero_point) 三个输入，"
            f"得到 {len(inputs)}"
        )
    q, scale, zp = inputs
    if q.dtype != DType.I8:
        raise TypeError(
            f"dequantize_i8_group_zp 的第一个输入必须是 i8，得到 {q.dtype.value}"
        )
    if q.rank != 2:
        raise ValueError(
            f"dequantize_i8_group_zp 的 q 必须是二维 (K, N)，得到秩 {q.rank}"
        )
    if scale.dtype != DType.F32:
        raise TypeError(
            f"dequantize_i8_group_zp 的 scale 必须是 f32，得到 {scale.dtype.value}"
        )
    if zp.dtype != DType.I8:
        raise TypeError(
            f"dequantize_i8_group_zp 的 zero_point 必须是 i8，得到 {zp.dtype.value}"
        )
    _check_group_params(q, scale, "dequantize_i8_group_zp", "scale")
    if tuple(zp.shape) != tuple(scale.shape):
        raise ValueError(
            f"dequantize_i8_group_zp 的 zero_point 形状 {zp.shape} "
            f"必须与 scale {scale.shape} 相同（同一组共用一对参数）"
        )
    return TensorType(DType.F32, q.shape)


def _ref_dequantize_i8_group_zp(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    q, scale, zp = args
    # 与 dequantize_i8_zp 同一顺序：先在 int32 里减，再放大。
    diff = q.astype(np.int32) - np.repeat(zp.astype(np.int32), q.shape[0] // zp.shape[0], axis=0)
    repeated = np.repeat(scale, q.shape[0] // scale.shape[0], axis=0)
    return diff.astype(np.float32) * repeated


register(
    Op(
        name="dequantize_i8_group_zp",
        infer=_infer_dequantize_i8_group_zp,
        reference=_ref_dequantize_i8_group_zp,
        doc="逐组非对称反量化：f32(q - zp) * scale，减法在 int32 里做；"
        "scale / zero_point 形状都是 (K/G, N)，组内共用（AWQ / GPTQ 的 group_size 形态）",
    )
)


def _infer_linear_i8_zp(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) not in (4, 5):
        raise ValueError(
            "linear_i8_zp 需要 (x, qweight, scale, zero_point[, bias]) 输入，"
            f"得到 {len(inputs)}"
        )
    x, qweight, scale, zp = inputs[0], inputs[1], inputs[2], inputs[3]
    if x.dtype != DType.F32:
        raise TypeError(
            f"linear_i8_zp 的 x 必须是 f32，得到 {x.dtype.value}；"
            "反量化之后的 GEMM 用 f32 累加"
        )
    if x.rank < 2:
        raise ValueError(f"linear_i8_zp 的 x 至少是 2 维，得到秩 {x.rank}")
    if qweight.dtype != DType.I8:
        raise TypeError(f"linear_i8_zp 的权重必须是 i8，得到 {qweight.dtype.value}")
    if qweight.rank != 2:
        raise ValueError(
            f"linear_i8_zp 的量化权重必须是 2 维 (K, N)，得到 {qweight.shape}"
        )
    if scale.dtype != DType.F32:
        raise TypeError(f"linear_i8_zp 的 scale 必须是 f32，得到 {scale.dtype.value}")
    if zp.dtype != DType.I8:
        raise TypeError(f"linear_i8_zp 的 zero_point 必须是 i8，得到 {zp.dtype.value}")
    _check_broadcastable(qweight, scale, "linear_i8_zp", "scale")
    _check_broadcastable(qweight, zp, "linear_i8_zp", "zero_point")

    k, n = qweight.shape
    kx = x.shape[-1]
    if isinstance(kx, int) and isinstance(k, int):
        if kx != k:
            raise ValueError(
                f"linear_i8_zp 的内维不匹配：x 的最后一维 {kx} 与权重的 K={k}"
            )
    elif kx != k:
        raise ValueError(f"linear_i8_zp 的内维符号不一致：x={kx} 与权重 K={k}")

    if len(inputs) == 5:
        bias = inputs[4]
        if bias.dtype != DType.F32:
            raise TypeError(f"linear_i8_zp 的 bias 必须是 f32，得到 {bias.dtype.value}")
        if bias.rank != 1:
            raise ValueError(f"linear_i8_zp 的 bias 必须是秩 1，得到秩 {bias.rank}")
        if isinstance(bias.shape[0], int) and isinstance(n, int) and bias.shape[0] != n:
            raise ValueError(f"linear_i8_zp 的 bias 长度 {bias.shape[0]} 与 N={n} 不一致")
    return TensorType(DType.F32, x.shape[:-1] + (n,))


def _ref_linear_i8_zp(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x, qweight, scale, zp = args[0], args[1], args[2], args[3]
    if float(np.min(scale)) <= 0.0:
        raise ValueError(
            f"linear_i8_zp 的 scale 必须严格为正，得到最小值 {float(np.min(scale))}"
        )
    weight = (qweight.astype(np.int32) - zp.astype(np.int32)).astype(np.float32) * scale
    out = np.matmul(x, weight)
    if len(args) > 4:
        out = out + args[4]
    return out


register(
    Op(
        name="linear_i8_zp",
        infer=_infer_linear_i8_zp,
        reference=_ref_linear_i8_zp,
        doc="非对称 weight-only int8 线性层：(..., K) x i8(K, N) -> (..., N)。"
        "数值等于 ``matmul(x, dequantize_i8_zp(qweight, scale, zero_point)) "
        "[+ bias]``；零点让导出模型里非零中心的那条权重轴不必再补一个偏置",
    )
)


def _infer_linear_i8a8_zp(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    if len(inputs) not in (6, 7):
        raise ValueError(
            "linear_i8a8_zp 需要 (x_q, x_scale, x_zero_point, w_q, w_scale, "
            f"w_zero_point[, bias]) 输入，得到 {len(inputs)}"
        )
    xq, xs, xzp, wq, ws, wzp = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]
    if xq.dtype != DType.I8:
        raise TypeError(
            f"linear_i8a8_zp 的激活必须是 i8，得到 {xq.dtype.value}；"
            "浮点激活请先走 quantize_i8_zp，或改用 linear_i8_zp"
        )
    if wq.dtype != DType.I8:
        raise TypeError(f"linear_i8a8_zp 的权重必须是 i8，得到 {wq.dtype.value}")
    if xs.dtype != DType.F32:
        raise TypeError(
            f"linear_i8a8_zp 的激活 scale 必须是 f32，得到 {xs.dtype.value}"
        )
    if ws.dtype != DType.F32:
        raise TypeError(f"linear_i8a8_zp 的权重 scale 必须是 f32，得到 {ws.dtype.value}")
    if xzp.dtype != DType.I8:
        raise TypeError(
            f"linear_i8a8_zp 的激活 zero_point 必须是 i8，得到 {xzp.dtype.value}"
        )
    if wzp.dtype != DType.I8:
        raise TypeError(
            f"linear_i8a8_zp 的权重 zero_point 必须是 i8，得到 {wzp.dtype.value}"
        )
    if xq.rank < 2:
        raise ValueError(f"linear_i8a8_zp 的激活至少是 2 维，得到秩 {xq.rank}")
    if wq.rank != 2:
        raise ValueError(
            f"linear_i8a8_zp 的量化权重必须是 2 维 (K, N)，得到 {wq.shape}"
        )

    k, n = wq.shape
    kx = xq.shape[-1]
    if isinstance(kx, int) and isinstance(k, int):
        if kx != k:
            raise ValueError(
                f"linear_i8a8_zp 的内维不匹配：激活的最后一维 {kx} 与权重的 K={k}"
            )
    elif kx != k:
        raise ValueError(f"linear_i8a8_zp 的内维符号不一致：激活 {kx} 与权重 K={k}")

    _check_broadcastable(xq, xs, "linear_i8a8_zp", "激活 scale")
    _check_broadcastable(xq, xzp, "linear_i8a8_zp", "激活 zero_point")
    # int32 累加要求量化参数能提到求和号之外：激活侧必须沿 K 维恒定
    # （每个 token 一组），权重侧必须沿 K 维恒定（每个输出通道一组）。
    for label, param in (("激活 scale", xs), ("激活 zero_point", xzp)):
        if param.rank > 0 and param.shape[-1] != 1:
            raise ValueError(
                f"linear_i8a8_zp 的{label}形状 {param.shape} 沿 K 维不是 1："
                "int32 累加要求每个 token 一组量化参数（如 (B, T, 1) 或标量）"
            )
    for label, param in (("权重 scale", ws), ("权重 zero_point", wzp)):
        if param.rank > 1:
            raise ValueError(
                f"linear_i8a8_zp 的{label}必须是逐输出通道 (N,) 或逐张量标量，"
                f"得到秩 {param.rank} 形状 {param.shape}；沿 K 维逐元素变化的"
                "量化参数无法用 int32 累加表达"
            )
        if param.rank == 1 and param.shape[0] != 1 and param.shape[0] != n:
            raise ValueError(
                f"linear_i8a8_zp 的{label}长度 {param.shape[0]} 与 N={n} 不一致"
            )
    if isinstance(k, int) and k > _I32_ACC_LIMIT_ZP:
        raise ValueError(
            f"linear_i8a8_zp 的 K={k} 超过 int32 累加上限 {_I32_ACC_LIMIT_ZP}："
            "(xq - zp) 与 (wq - zp) 各可取 255，K * 255 * 255 会溢出 int32；"
            "请分段累加"
        )

    if len(inputs) == 7:
        bias = inputs[6]
        if bias.dtype != DType.F32:
            raise TypeError(
                f"linear_i8a8_zp 的 bias 必须是 f32，得到 {bias.dtype.value}"
            )
        if bias.rank != 1:
            raise ValueError(f"linear_i8a8_zp 的 bias 必须是秩 1，得到秩 {bias.rank}")
        if isinstance(bias.shape[0], int) and isinstance(n, int) and bias.shape[0] != n:
            raise ValueError(
                f"linear_i8a8_zp 的 bias 长度 {bias.shape[0]} 与 N={n} 不一致"
            )
    return TensorType(DType.F32, xq.shape[:-1] + (n,))


def _ref_linear_i8a8_zp(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    xq, xs, xzp, wq, ws, wzp = args[0], args[1], args[2], args[3], args[4], args[5]
    if float(np.min(xs)) <= 0.0:
        raise ValueError(
            f"linear_i8a8_zp 的激活 scale 必须严格为正，得到最小值 {float(np.min(xs))}"
        )
    if float(np.min(ws)) <= 0.0:
        raise ValueError(
            f"linear_i8a8_zp 的权重 scale 必须严格为正，得到最小值 {float(np.min(ws))}"
        )
    k = int(xq.shape[-1])
    if k > _I32_ACC_LIMIT_ZP:
        raise ValueError(
            f"linear_i8a8_zp 的 K={k} 超过 int32 累加上限 {_I32_ACC_LIMIT_ZP}："
            "(xq - zp) 与 (wq - zp) 各可取 255，K * 255 * 255 会溢出 int32；"
            "请分段累加"
        )
    # 与对称版唯一的数值差别是每个乘子先减自己的零点。先减再乘、在 int32 里
    # 累加，展开后的偏置项（-zx*colsum - zw*rowsum + K*zx*zw）才与整数累加
    # 逐位一致；"先算 acc 再在 f32 里补偏置"需要额外的归约，数值上也不等价。
    xq32 = xq.astype(np.int32) - xzp.astype(np.int32)
    wq32 = wq.astype(np.int32) - wzp.astype(np.int32)
    acc = np.matmul(xq32, wq32)
    row = np.broadcast_to(xs, xq.shape)[..., :1].astype(np.float32)
    out = acc.astype(np.float32) * row * ws
    if len(args) > 6:
        out = out + args[6]
    return out


register(
    Op(
        name="linear_i8a8_zp",
        infer=_infer_linear_i8a8_zp,
        reference=_ref_linear_i8a8_zp,
        doc="非对称 W8A8 线性层：激活与权重各带一个 i8 零点，先做 "
        "(x_q - x_zp) 与 (w_q - w_zp) 的 int32 累加，再乘两个 scale 的积并加 "
        "bias。语义是 ``((x_q - x_zp) @ (w_q - w_zp)).f32() * xs * ws [+ b]``；"
        "K * 255 * 255 超过 int32 上界时显式拒绝",
    )
)


# --------------------------------------------------------------------------
# 卷积
# --------------------------------------------------------------------------


def _pair(value, name: str) -> Tuple[int, int]:
    """把 int 或二元组规范成 ``(h, w)``。

    JSON 往返后 tuple 会变成 list，所以两种容器都必须接受。
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} 不能是 bool")
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise TypeError(f"{name} 必须是 int 或二元组，得到 {value!r}")


#: 卷积 / 池化接受的张量布局。
#:
#: ``nhwc`` 不是第二套数学：参考语义把输入转成 NCHW 用**同一份实现**算，
#: 再把输出转回 NHWC。这样两条路径逐位相同，位级差异不可能从布局里冒出来。
#: 它的意义在 IR 表达力与传播 pass（``passes/layout.py``）上：通道在后时
#: 卷积输出的通道维与下游逐元素算子共享同一根轴，传播因此不需要额外转置。
_LAYOUTS: Tuple[str, ...] = ("nchw", "nhwc")


def _layout_of(attrs: dict, op_name: str) -> str:
    """读取并校验 ``layout`` 属性；缺省 NCHW。非法值当场拒绝。"""
    layout = attrs.get("layout", "nchw")
    if layout not in _LAYOUTS:
        raise ValueError(f"{op_name} 的 layout 只能是 {_LAYOUTS}，得到 {layout!r}")
    return layout


def _nchw_type(t: TensorType) -> TensorType:
    """(N,H,W,C) -> (N,C,H,W)，dtype 不变。"""
    n, h, w, c = t.shape
    return TensorType(t.dtype, (n, c, h, w))


def _nhwc_type(t: TensorType) -> TensorType:
    """(N,C,H,W) -> (N,H,W,C)，dtype 不变。"""
    n, c, h, w = t.shape
    return TensorType(t.dtype, (n, h, w, c))


def _conv_out_dim(size: Dim, kernel: int, stride: int, padding: int, dilation: int) -> Dim:
    """NCHW 单个空间维的输出长度。

    静态维按定义式 ``floor((size + 2p - span) / stride) + 1`` 计算。

    符号维只在**结果恰好等于输入**时沿用原符号（kernel=1、stride=1、
    padding=0、dilation=1）；其余符号情形返回一个派生表达式，运行期绑定后
    求值。表达式只引用输入里已有的符号，不编造新名字。
    """
    span = dilation * (kernel - 1) + 1
    if not isinstance(size, int):
        if span == 1 and padding == 0 and stride == 1:
            return size
        # 其余符号情形用表达式表达，运行期绑定后再求值。符号维必须来自
        # 已出现的输入符号（与 reshape 的约定一致），表达式不引入新符号。
        out = sym_add(sym_binary("floordiv", sym_add(size, 2 * padding - span), stride), 1)
        return out
    out = (int(size) + 2 * padding - span) // stride + 1
    if out <= 0:
        raise ValueError(
            f"conv2d 输出尺寸非法: 输入 {size}、有效核 {span}、padding {padding}、stride {stride}"
        )
    return out


def _infer_conv2d(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    """按 ``layout`` 适配维度顺序，数学部分复用 NCHW 推导。

    NHWC 下权重仍是 ``(O, C/groups, kh, kw)``——布局说的是**激活**的排列，
    权重布局不随之前后颠倒，这与 cuDNN / TensorFlow 的惯例一致。
    """
    if _layout_of(attrs, "conv2d") == "nchw":
        return _infer_conv2d_nchw(inputs, attrs)
    x = inputs[0]
    if x.rank != 4:
        raise ValueError(f"conv2d 的 layout=nhwc 要求 x 是秩 4，得到秩 {x.rank}")
    return _nhwc_type(_infer_conv2d_nchw((_nchw_type(x),) + inputs[1:], attrs))


def _infer_conv2d_nchw(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) not in (2, 3):
        raise ValueError(f"conv2d 需要 2 或 3 个输入（x, weight[, bias]），得到 {len(inputs)}")
    x, weight = inputs[0], inputs[1]
    if x.rank != 4 or weight.rank != 4:
        raise ValueError(f"conv2d 要求 x/weight 都是 NCHW 秩 4，得到 {x.rank} 与 {weight.rank}")
    if x.dtype != weight.dtype:
        raise TypeError(
            f"conv2d 要求 x 与 weight 的 dtype 一致，得到 {x.dtype.value} 与 {weight.dtype.value}"
        )

    groups = int(attrs.get("groups", 1))
    if groups < 1:
        raise ValueError(f"conv2d 的 groups 必须 >= 1，得到 {groups}")

    sh, sw = _pair(attrs.get("stride", (1, 1)), "stride")
    ph, pw = _pair(attrs.get("padding", (0, 0)), "padding")
    dh, dw = _pair(attrs.get("dilation", (1, 1)), "dilation")
    if sh < 1 or sw < 1 or dh < 1 or dw < 1:
        raise ValueError(f"conv2d 的 stride/dilation 必须 >= 1，得到 {(sh, sw)} 与 {(dh, dw)}")
    if ph < 0 or pw < 0:
        raise ValueError(f"conv2d 的 padding 必须 >= 0，得到 {(ph, pw)}")

    n, c, h, w = x.shape
    o, cg, kh, kw = weight.shape
    if not isinstance(kh, int) or not isinstance(kw, int):
        raise ValueError(f"conv2d 的卷积核尺寸必须静态已知，得到 {kh!r}x{kw!r}")
    if isinstance(o, int) and o % groups != 0:
        raise ValueError(f"conv2d 输出通道 {o} 不能被 groups={groups} 整除")
    if isinstance(c, int) and isinstance(cg, int):
        if c % groups != 0:
            raise ValueError(f"conv2d 输入通道 {c} 不能被 groups={groups} 整除")
        if cg != c // groups:
            raise ValueError(
                f"conv2d 权重通道 {cg} 应等于 输入通道/groups = {c // groups}"
            )
    elif not isinstance(c, int) and not isinstance(cg, int) and groups == 1 and c != cg:
        raise ValueError(f"conv2d 输入通道符号 {c} 与权重通道符号 {cg} 不一致")

    ho = _conv_out_dim(h, kh, sh, ph, dh)
    wo = _conv_out_dim(w, kw, sw, pw, dw)

    if len(inputs) == 3:
        bias = inputs[2]
        if bias.rank != 1:
            raise ValueError(f"conv2d 的 bias 必须是秩 1，得到秩 {bias.rank}")
        if bias.dtype != x.dtype:
            raise TypeError(
                f"conv2d 要求 bias 与 x 的 dtype 一致，得到 {bias.dtype.value} 与 {x.dtype.value}"
            )
        if isinstance(bias.shape[0], int) and isinstance(o, int) and bias.shape[0] != o:
            raise ValueError(f"conv2d bias 长度 {bias.shape[0]} 与输出通道 {o} 不一致")

    return TensorType(x.dtype, (n, o, ho, wo))


def _conv2d_cross(x, weight, sh, sw, ph, pw, dh, dw) -> np.ndarray:
    """单组 NCHW 交叉相关；调用方保证 ``x`` 通道数 == ``weight`` 第二维。"""
    n, c, h, w = x.shape
    o, _, kh, kw = weight.shape
    ho = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wo = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    padded = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw))) if (ph or pw) else x
    out = np.zeros((n, o, ho, wo), dtype=np.result_type(x.dtype, weight.dtype))
    for i in range(kh):
        for j in range(kw):
            patch = padded[
                :,
                :,
                i * dh : i * dh + sh * (ho - 1) + 1 : sh,
                j * dw : j * dw + sw * (wo - 1) + 1 : sw,
            ]
            out += np.einsum("nchw,oc->nohw", patch, weight[:, :, i, j])
    return out


def _ref_conv2d(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """按 ``layout`` 适配维度顺序；数学部分复用 NCHW 参考实现。"""
    if _layout_of(attrs, "conv2d") == "nchw":
        return _ref_conv2d_nchw(args, attrs)
    out = _ref_conv2d_nchw([np.transpose(args[0], (0, 3, 1, 2)), *args[1:]], attrs)
    return np.transpose(out, (0, 2, 3, 1))


def _ref_conv2d_nchw(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """NCHW 交叉相关（与 PyTorch ``conv2d`` 一致，不翻转核）。

    用 ``einsum`` 逐 (kh, kw) 位置累加，刻意不做 im2col：
    参考实现的职责是定义语义，不是定义性能。累加顺序固定，因而确定。
    分组时按输入/输出通道切片逐组计算，再沿输出通道拼回——与
    PyTorch 的 ``groups`` 语义一致：第 g 组只看 ``C/groups`` 个输入通道。
    """
    x, weight = args[0], args[1]
    sh, sw = _pair(attrs.get("stride", (1, 1)), "stride")
    ph, pw = _pair(attrs.get("padding", (0, 0)), "padding")
    dh, dw = _pair(attrs.get("dilation", (1, 1)), "dilation")
    groups = int(attrs.get("groups", 1))

    if groups == 1:
        out = _conv2d_cross(x, weight, sh, sw, ph, pw, dh, dw)
    else:
        c = x.shape[1]
        o = weight.shape[0]
        cg, og = c // groups, o // groups
        parts = [
            _conv2d_cross(
                x[:, gi * cg : (gi + 1) * cg],
                weight[gi * og : (gi + 1) * og],
                sh, sw, ph, pw, dh, dw,
            )
            for gi in range(groups)
        ]
        out = np.concatenate(parts, axis=1)
    if len(args) > 2:
        out += args[2].reshape(1, -1, 1, 1)
    return out


register(
    Op(
        name="conv2d",
        infer=_infer_conv2d,
        reference=_ref_conv2d,
        attrs_schema=("stride", "padding", "dilation", "groups", "layout"),
        doc="二维交叉相关：layout='nchw'（默认）时 x(N,C,H,W)，layout='nhwc' 时"
        " x(N,H,W,C)；weight 恒为 (O,C/groups,kh,kw)，输出布局与 x 一致。"
        "attrs 另有 stride/padding/dilation 二元组与 groups（groups>1 即分组卷积）",
    )
)


def _pool_out_dim(
    size: Dim, kernel: int, stride: int, padding: int, ceil_mode: bool = False
) -> Dim:
    """池化单个空间维的输出长度；符号维规则与 ``conv2d`` 一致。

    ``ceil_mode`` 用 PyTorch 的规则：先向上取整，再把"窗口起点落在输入（含左侧
    padding）之外"的那个多余窗口去掉（``(out - 1) * stride >= size + padding``）。
    第二条修正不能省——它保证每个窗口至少覆盖一个真实元素，也让 ceil 与 floor
    在整除时给出同一个结果。符号维上无法表达这个条件分支，因此显式报错。
    """
    if not isinstance(size, int):
        if ceil_mode:
            raise ValueError("ceil_mode 需要静态空间维：输出长度的上限修正依赖具体的 H / W")
        if kernel == 1 and padding == 0 and stride == 1:
            return size
        # 与卷积同构：符号维上保留派生表达式，运行期绑定后求值。
        return sym_add(
            sym_binary("floordiv", sym_add(size, 2 * padding - kernel), stride), 1
        )
    if ceil_mode:
        out = -((int(size) + 2 * padding - kernel) // -stride) + 1
        if (out - 1) * stride >= int(size) + padding:
            out -= 1
    else:
        out = (int(size) + 2 * padding - kernel) // stride + 1
    if out <= 0:
        raise ValueError(
            f"池化输出尺寸非法: 输入 {size}、核 {kernel}、padding {padding}、"
            f"stride {stride}、ceil_mode={ceil_mode}"
        )
    return out


def _pool_flags(attrs: dict) -> Tuple[bool, bool]:
    """池化的布尔属性 ``(ceil_mode, count_include_pad)``，缺省都是 False / True。

    ``count_include_pad`` 只对平均池化有意义，但放在这里统一解析：语义上它就是
    "分母是否含 padding 的零"，与 ``ceil_mode`` 一起决定窗口与除数。
    """
    ceil_mode = attrs.get("ceil_mode", False)
    if not isinstance(ceil_mode, bool):
        raise ValueError(f"ceil_mode 必须是 bool，得到 {ceil_mode!r}")
    count_include_pad = attrs.get("count_include_pad", True)
    if not isinstance(count_include_pad, bool):
        raise ValueError(f"count_include_pad 必须是 bool，得到 {count_include_pad!r}")
    return ceil_mode, count_include_pad


def _infer_maxpool2d(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if _layout_of(attrs, "maxpool2d") == "nchw":
        return _infer_maxpool2d_nchw(inputs, attrs)
    x = inputs[0]
    if x.rank != 4:
        raise ValueError(f"maxpool2d 的 layout=nhwc 要求 x 是秩 4，得到秩 {x.rank}")
    return _nhwc_type(_infer_maxpool2d_nchw((_nchw_type(x),), attrs))


def _infer_maxpool2d_nchw(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    x = inputs[0]
    if x.rank != 4:
        raise ValueError(f"maxpool2d 要求 NCHW 秩 4 输入，得到秩 {x.rank}")

    kh, kw = _pair(attrs["kernel"], "kernel")
    raw_stride = attrs.get("stride")
    sh, sw = _pair(attrs["kernel"] if raw_stride is None else raw_stride, "stride")
    ph, pw = _pair(attrs.get("padding", (0, 0)), "padding")
    if kh < 1 or kw < 1 or sh < 1 or sw < 1:
        raise ValueError(f"maxpool2d 的 kernel/stride 必须 >= 1，得到 {(kh, kw)} 与 {(sh, sw)}")
    if ph < 0 or pw < 0:
        raise ValueError(f"maxpool2d 的 padding 必须 >= 0，得到 {(ph, pw)}")
    ceil_mode, _ = _pool_flags(attrs)

    n, c, h, w = x.shape
    ho = _pool_out_dim(h, kh, sh, ph, ceil_mode)
    wo = _pool_out_dim(w, kw, sw, pw, ceil_mode)
    return TensorType(x.dtype, (n, c, ho, wo))


def _ref_maxpool2d(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    if _layout_of(attrs, "maxpool2d") == "nchw":
        return _ref_maxpool2d_nchw(args, attrs)
    out = _ref_maxpool2d_nchw([np.transpose(args[0], (0, 3, 1, 2))], attrs)
    return np.transpose(out, (0, 2, 3, 1))


def _ref_maxpool2d_nchw(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """NCHW 最大池化；padding 区域不参与取 max（等价于填 -inf）。"""
    x = args[0]
    (kh, kw), (sh, sw), (ph, pw) = _pool_geometry(attrs)
    ceil_mode, _ = _pool_flags(attrs)

    n, c, h, w = x.shape
    ho = _pool_out_dim(h, kh, sh, ph, ceil_mode)
    wo = _pool_out_dim(w, kw, sw, pw, ceil_mode)

    # ceil_mode 的最后一个窗口会越过 H + 2p：补到窗口实际够用的长度，切片才
    # 不会短于输出长度（短了就无法与累积结果逐元素比较）。
    need_h = max(h + 2 * ph, (ho - 1) * sh + kh)
    need_w = max(w + 2 * pw, (wo - 1) * sw + kw)
    padded = np.pad(
        x,
        ((0, 0), (0, 0), (ph, need_h - h - ph), (pw, need_w - w - pw)),
        constant_values=-np.inf,
    )
    out = np.full((n, c, ho, wo), -np.inf, dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            patch = padded[
                :, :, i : i + sh * (ho - 1) + 1 : sh, j : j + sw * (wo - 1) + 1 : sw
            ]
            out = np.maximum(out, patch)
    return out


register(
    Op(
        name="maxpool2d",
        infer=_infer_maxpool2d,
        reference=_ref_maxpool2d,
        attrs_schema=("kernel", "stride", "padding", "ceil_mode", "layout"),
        doc="二维最大池化：kernel 必填，stride 默认等于 kernel，padding 默认 0；"
        "ceil_mode=True 时输出向上取整并去掉起点越界的多余窗口；"
        "layout 默认 nchw，nhwc 时输入/输出为 (N,H,W,C)",
    )
)


def _pool_geometry(attrs: dict) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """池化的 (kernel, stride, padding)；stride 缺省等于 kernel。"""
    kh, kw = _pair(attrs["kernel"], "kernel")
    raw_stride = attrs.get("stride")
    sh, sw = _pair(attrs["kernel"] if raw_stride is None else raw_stride, "stride")
    ph, pw = _pair(attrs.get("padding", (0, 0)), "padding")
    if kh < 1 or kw < 1 or sh < 1 or sw < 1:
        raise ValueError(f"池化的 kernel/stride 必须 >= 1，得到 {(kh, kw)} 与 {(sh, sw)}")
    if ph < 0 or pw < 0:
        raise ValueError(f"池化的 padding 必须 >= 0，得到 {(ph, pw)}")
    return (kh, kw), (sh, sw), (ph, pw)


def _infer_avgpool2d(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if _layout_of(attrs, "avgpool2d") == "nchw":
        return _infer_avgpool2d_nchw(inputs, attrs)
    x = inputs[0]
    if x.rank != 4:
        raise ValueError(f"avgpool2d 的 layout=nhwc 要求 x 是秩 4，得到秩 {x.rank}")
    return _nhwc_type(_infer_avgpool2d_nchw((_nchw_type(x),), attrs))


def _infer_avgpool2d_nchw(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    x = inputs[0]
    if x.rank != 4:
        raise ValueError(f"avgpool2d 要求 NCHW 秩 4 输入，得到秩 {x.rank}")
    if x.dtype not in (DType.F16, DType.BF16, DType.F32, DType.F64):
        raise TypeError(f"avgpool2d 需要浮点输入，得到 {x.dtype.value}")
    (kh, kw), (sh, sw), (ph, pw) = _pool_geometry(attrs)
    ceil_mode, _ = _pool_flags(attrs)
    n, c, h, w = x.shape
    return TensorType(
        x.dtype,
        (
            n,
            c,
            _pool_out_dim(h, kh, sh, ph, ceil_mode),
            _pool_out_dim(w, kw, sw, pw, ceil_mode),
        ),
    )


def _pool_count_window(attrs: dict, h: int, w: int, ho: int, wo: int,
                       need_h: int, need_w: int) -> np.ndarray:
    """每个输出格的分母：窗口落在 padded 输入里的格数。

    ``count_include_pad=True`` 时 padding 计入（值按 0），区间是
    ``[0, H + 2p)`` / ``[0, W + 2p)``；``False`` 时只数真实输入，区间是
    ``[p, p + H)``。``ceil_mode`` 让最后一格越过 padded 输入时，越界部分两边
    都不计入——PyTorch 的 ``avg_pool2d`` 正是这么定义的，只按 ``KH * KW``
    除会在这种角落上偏小。分母兜底到 1，防零窗口除法。
    """
    (kh, kw), (sh, sw), (ph, pw) = _pool_geometry(attrs)
    _, count_include_pad = _pool_flags(attrs)
    if count_include_pad:
        lo_h, hi_h = 0, h + 2 * ph
        lo_w, hi_w = 0, w + 2 * pw
    else:
        lo_h, hi_h = ph, ph + h
        lo_w, hi_w = pw, pw + w
    keep = np.zeros((1, 1, need_h, need_w), dtype=np.float64)
    keep[:, :, lo_h:hi_h, lo_w:hi_w] = 1.0
    count = np.zeros((1, 1, ho, wo), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            count += keep[
                :, :, i : i + sh * (ho - 1) + 1 : sh, j : j + sw * (wo - 1) + 1 : sw
            ]
    np.maximum(count, 1.0, out=count)
    return count


def _ref_avgpool2d(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    if _layout_of(attrs, "avgpool2d") == "nchw":
        return _ref_avgpool2d_nchw(args, attrs)
    out = _ref_avgpool2d_nchw([np.transpose(args[0], (0, 3, 1, 2))], attrs)
    return np.transpose(out, (0, 2, 3, 1))


def _ref_avgpool2d_nchw(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """NCHW 平均池化。

    分母是窗口落在 padded 输入里的格数：``count_include_pad=True`` 时 padding
    的零也计入，``False`` 时只数真实输入；``ceil_mode`` 的越界部分两边都不
    计入。窗口外（``ceil_mode`` 的最后一格）按 0 参与求和。
    """
    x = args[0]
    (kh, kw), (sh, sw), (ph, pw) = _pool_geometry(attrs)
    ceil_mode, _ = _pool_flags(attrs)
    n, c, h, w = x.shape
    ho = _pool_out_dim(h, kh, sh, ph, ceil_mode)
    wo = _pool_out_dim(w, kw, sw, pw, ceil_mode)

    # 与 maxpool2d 一样补到窗口实际够用的长度：ceil 的最后一格会越过 H + 2p。
    need_h = max(h + 2 * ph, (ho - 1) * sh + kh)
    need_w = max(w + 2 * pw, (wo - 1) * sw + kw)
    padded = np.pad(
        x, ((0, 0), (0, 0), (ph, need_h - h - ph), (pw, need_w - w - pw))
    )
    count = _pool_count_window(attrs, h, w, ho, wo, need_h, need_w)
    acc = np.zeros((n, c, ho, wo), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            window = (
                slice(None),
                slice(None),
                slice(i, i + sh * (ho - 1) + 1, sh),
                slice(j, j + sw * (wo - 1) + 1, sw),
            )
            acc += padded[window]
    return (acc / count).astype(x.dtype)


register(
    Op(
        name="avgpool2d",
        infer=_infer_avgpool2d,
        reference=_ref_avgpool2d,
        attrs_schema=("kernel", "stride", "padding", "ceil_mode", "count_include_pad", "layout"),
        doc="平均池化：kernel 必填，stride 默认等于 kernel，padding 默认 0；"
        "count_include_pad=True（默认）时分母固定为 kernel 面积，False 时只数窗口"
        "与输入的重叠面积；ceil_mode 与 maxpool2d 同规则；"
        "layout 默认 nchw，nhwc 时输入/输出为 (N,H,W,C)",
    )
)


def _infer_avgpool2d_backward(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    dy = inputs[0]
    shape = _as_shape(attrs["input_shape"])
    if len(shape) != 4:
        raise ValueError(f"input_shape 必须是 NCHW 四元组，得到 {shape}")
    return TensorType(dy.dtype, shape)


def _ref_avgpool2d_backward(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """平均池化对输入的梯度：每个窗口把它那格的 dy/分母 均摊回去。

    分母与 ``avgpool2d`` 前向逐格对应（``_pool_count_window``）：padding 计不
    计入、``ceil_mode`` 的越界格都不进分母，所以不同输出格的分母可能不同，
    梯度不再是"乘一个常数"。
    """
    dy = args[0]
    n, c, h, w = (int(d) for d in attrs["input_shape"])
    (kh, kw), (sh, sw), (ph, pw) = _pool_geometry(attrs)
    ceil_mode, _ = _pool_flags(attrs)
    ho = _pool_out_dim(h, kh, sh, ph, ceil_mode)
    wo = _pool_out_dim(w, kw, sw, pw, ceil_mode)
    if dy.shape != (n, c, ho, wo):
        raise ValueError(
            f"avgpool2d_backward 的 dy 形状应为 {(n, c, ho, wo)}，得到 {dy.shape}"
        )

    need_h = max(h + 2 * ph, (ho - 1) * sh + kh)
    need_w = max(w + 2 * pw, (wo - 1) * sw + kw)
    padded = np.zeros((n, c, need_h, need_w), dtype=np.float64)
    share = dy.astype(np.float64) / _pool_count_window(
        attrs, h, w, ho, wo, need_h, need_w
    )
    for i in range(kh):
        for j in range(kw):
            padded[
                :, :, i : i + sh * (ho - 1) + 1 : sh, j : j + sw * (wo - 1) + 1 : sw
            ] += share
    return padded[:, :, ph : ph + h, pw : pw + w].astype(dy.dtype)


register(
    Op(
        name="avgpool2d_backward",
        infer=_infer_avgpool2d_backward,
        reference=_ref_avgpool2d_backward,
        attrs_schema=(
            "input_shape",
            "kernel",
            "stride",
            "padding",
            "ceil_mode",
            "count_include_pad",
        ),
        doc="平均池化对输入的梯度：dy(N,C,HO,WO) -> dx(N,C,H,W)，"
        "input_shape 为原输入形状；除数与前向的 ceil_mode / count_include_pad 一致，"
        "与 maxpool2d_backward 一样只定义一阶",
    )
)


def _adaptive_window(index: int, out_size: int, in_size: int) -> Tuple[int, int]:
    """PyTorch 自适应池化的窗口 ``[start, end)``。

    定义是 ``start = floor(i*H/OH)``、``end = ceil((i+1)*H/OH)``。这里用
    非负整数的整除形式表达，避免浮点误差：``start = i*H // OH``，
    ``end = ((i+1)*H + OH - 1) // OH``。当 H 不是 OH 的整数倍时窗口可以
    重叠，反向因此必须累加而不是覆盖。
    """
    start = (index * in_size) // out_size
    end = ((index + 1) * in_size + out_size - 1) // out_size
    return start, end


def _adaptive_output_size(attrs: dict) -> Tuple[int, int]:
    raw = attrs["output_size"]
    if isinstance(raw, int) and not isinstance(raw, bool):
        oh = ow = int(raw)
    elif isinstance(raw, (tuple, list)) and len(raw) in (1, 2):
        oh = int(raw[0])
        ow = int(raw[-1])
    else:
        raise ValueError(f"output_size 必须是 int 或长度 1/2 的序列，得到 {raw!r}")
    if oh < 1 or ow < 1:
        raise ValueError(f"adaptive_avg_pool2d 的 output_size 必须 >= 1，得到 {(oh, ow)}")
    return oh, ow


def _infer_adaptive_avg_pool2d(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    x = inputs[0]
    if x.rank != 4:
        raise ValueError(f"adaptive_avg_pool2d 要求 NCHW 秩 4 输入，得到秩 {x.rank}")
    if x.dtype not in (DType.F16, DType.BF16, DType.F32, DType.F64):
        raise TypeError(f"adaptive_avg_pool2d 需要浮点输入，得到 {x.dtype.value}")
    oh, ow = _adaptive_output_size(attrs)
    n, c, _, _ = x.shape
    return TensorType(x.dtype, (n, c, oh, ow))


def _ref_adaptive_avg_pool2d(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """自适应平均池化：每个输出格是它那个窗口的均值。

    与 PyTorch ``adaptive_avg_pool2d`` 一致（``output_size=(1, 1)`` 就是
    全局平均池化）。窗口大小随输出位置变化，因此没有固定分母——这正是
    它与 ``avgpool2d`` 需要分成两个算子的原因。
    """
    x = args[0]
    n, c, h, w = x.shape
    oh, ow = _adaptive_output_size(attrs)
    out = np.empty((n, c, oh, ow), dtype=np.result_type(x.dtype, np.float32))
    for i in range(oh):
        hs, he = _adaptive_window(i, oh, h)
        for j in range(ow):
            ws, we = _adaptive_window(j, ow, w)
            out[:, :, i, j] = x[:, :, hs:he, ws:we].mean(axis=(-2, -1))
    return out.astype(x.dtype)


register(
    Op(
        name="adaptive_avg_pool2d",
        infer=_infer_adaptive_avg_pool2d,
        reference=_ref_adaptive_avg_pool2d,
        attrs_schema=("output_size",),
        doc="NCHW 自适应平均池化：output_size 为 (OH, OW)，窗口按 "
        "start=floor(i*H/OH)、end=ceil((i+1)*H/OH) 取；output_size=(1,1) 即全局池化",
    )
)


def _infer_adaptive_avg_pool2d_backward(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    dy = inputs[0]
    shape = _as_shape(attrs["input_shape"])
    if len(shape) != 4:
        raise ValueError(f"input_shape 必须是 NCHW 四元组，得到 {shape}")
    if dy.rank != 4:
        raise ValueError(f"adaptive_avg_pool2d_backward 要求 dy 是秩 4，得到秩 {dy.rank}")
    return TensorType(dy.dtype, shape)


def _ref_adaptive_avg_pool2d_backward(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """自适应平均池化对输入的梯度：每个窗口把 dy/窗口面积 均摊回去。"""
    dy = args[0]
    n, c, h, w = (int(d) for d in attrs["input_shape"])
    oh, ow = _adaptive_output_size(attrs)
    if dy.shape != (n, c, oh, ow):
        raise ValueError(
            f"adaptive_avg_pool2d_backward 的 dy 形状应为 {(n, c, oh, ow)}，得到 {dy.shape}"
        )

    dx = np.zeros((n, c, h, w), dtype=np.float64)
    for i in range(oh):
        hs, he = _adaptive_window(i, oh, h)
        for j in range(ow):
            ws, we = _adaptive_window(j, ow, w)
            share = dy[:, :, i, j].astype(np.float64) / float((he - hs) * (we - ws))
            # 窗口可以重叠（H 不是 OH 的整数倍时），必须累加；
            # 这里用 += 而不是赋值，与 CUDA 侧的 atomicAdd 对应。
            dx[:, :, hs:he, ws:we] += share[:, :, None, None]
    return dx.astype(dy.dtype)


register(
    Op(
        name="adaptive_avg_pool2d_backward",
        infer=_infer_adaptive_avg_pool2d_backward,
        reference=_ref_adaptive_avg_pool2d_backward,
        attrs_schema=("input_shape", "output_size"),
        doc="自适应平均池化对输入的梯度：dy(N,C,OH,OW) -> dx(N,C,H,W)，"
        "input_shape 为原输入形状；窗口重叠处累加",
    )
)


# --------------------------------------------------------------------------
# 空间插值
# --------------------------------------------------------------------------


def _interp_size(attrs: dict) -> Tuple[int, int]:
    """``size`` 属性：目标空间尺寸 ``(OH, OW)``。"""
    raw = attrs["size"]
    if isinstance(raw, int) and not isinstance(raw, bool):
        oh = ow = int(raw)
    elif isinstance(raw, (tuple, list)) and len(raw) in (1, 2):
        oh = int(raw[0])
        ow = int(raw[-1])
    else:
        raise ValueError(
            f"interpolate 的 size 必须是 int 或长度 1/2 的序列，得到 {raw!r}"
        )
    if oh < 1 or ow < 1:
        raise ValueError(f"interpolate 的 size 必须 >= 1，得到 {(oh, ow)}")
    return oh, ow


def _interp_mode(attrs: dict) -> str:
    mode = attrs.get("mode", "nearest")
    if mode not in ("nearest", "bilinear"):
        raise ValueError(
            f"interpolate 只支持 mode='nearest' / 'bilinear'，得到 {mode!r}；"
            "bicubic、三线性与 antialias 尚未实现"
        )
    return mode


def _interp_align(attrs: dict, mode: str) -> bool:
    align = attrs.get("align_corners", False)
    if not isinstance(align, bool):
        raise ValueError(f"interpolate 的 align_corners 必须是 bool，得到 {align!r}")
    if mode == "nearest" and align:
        raise ValueError(
            "interpolate 的 nearest 模式不接受 align_corners=True："
            "最近邻没有角对齐语义（PyTorch 同样拒绝这一组合）"
        )
    return align


def _interp_axis(
    in_size: int, out_size: int, align_corners: bool, nearest: bool
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """单轴的源坐标与权重，``(i0, i1, lam)``（nearest 的 lam 为 None）。

    坐标用整数分子/分母表示，避免"参考实现与设备内核用浮点算 scale、在边界
    差一格"：

    - nearest：``i = (dst * in) // out``。
    - bilinear / align_corners=True：``src = dst * (in-1) / (out-1)``；
      ``out == 1`` 时整轴取 0。
    - bilinear / align_corners=False：``src = ((2*dst+1)*in - out) / (2*out)``，
      再截断到 ``[0, in-1]``（下界对应 torch 的 ``src<0 -> 0``，上界是双线性
      插值的标准定义，in==1 时整轴取 0）。

    ``i1`` 截断到 ``in-1``：当 ``src`` 落在最后一格时 ``lam`` 可能恰好为 1，
    与 ``i1 == i0`` 等价，不会读出界。
    """
    dst = np.arange(out_size, dtype=np.int64)
    if nearest:
        idx = (dst * int(in_size)) // int(out_size)
        return idx, idx, None
    if align_corners:
        if out_size > 1:
            num = dst * (int(in_size) - 1)
            den = int(out_size) - 1
        else:
            num = np.zeros(1, dtype=np.int64)
            den = 1
    else:
        num = (2 * dst + 1) * int(in_size) - int(out_size)
        den = 2 * int(out_size)
        np.maximum(num, 0, out=num)
        if in_size > 1:
            np.minimum(num, (int(in_size) - 1) * den, out=num)
        else:
            num[:] = 0
    i0 = num // den
    rem = num - i0 * den
    lam = rem.astype(np.float64) / float(den)
    i1 = np.minimum(i0 + 1, int(in_size) - 1)
    return i0, i1, lam


def _infer_interpolate(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    x = inputs[0]
    if x.rank != 4:
        raise ValueError(f"interpolate 要求 NCHW 秩 4 输入，得到秩 {x.rank}")
    if not x.dtype.is_float:
        raise TypeError(f"interpolate 需要浮点输入，得到 {x.dtype.value}")
    mode = _interp_mode(attrs)
    _interp_align(attrs, mode)
    oh, ow = _interp_size(attrs)
    n, c, _, _ = x.shape
    return TensorType(x.dtype, (n, c, oh, ow))


def _ref_interpolate(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x = args[0]
    mode = _interp_mode(attrs)
    align = _interp_align(attrs, mode)
    oh, ow = _interp_size(attrs)
    ih = _interp_axis(x.shape[2], oh, align, mode == "nearest")
    iw = _interp_axis(x.shape[3], ow, align, mode == "nearest")
    if mode == "nearest":
        return x[:, :, ih[0][:, None], iw[0][None, :]]

    h0, h1, lh = ih
    w0, w1, lw = iw
    xf = x.astype(np.float64)
    h0, h1, lh = h0[:, None], h1[:, None], lh[:, None]
    w0, w1, lw = w0[None, :], w1[None, :], lw[None, :]
    out = (
        (1.0 - lh) * (1.0 - lw) * xf[:, :, h0, w0]
        + (1.0 - lh) * lw * xf[:, :, h0, w1]
        + lh * (1.0 - lw) * xf[:, :, h1, w0]
        + lh * lw * xf[:, :, h1, w1]
    )
    return out.astype(x.dtype)


register(
    Op(
        name="interpolate",
        infer=_infer_interpolate,
        reference=_ref_interpolate,
        attrs_schema=("size", "mode", "align_corners"),
        doc="NCHW 二维空间插值（只缩空间维，不改 N/C）：size=(OH,OW)，"
        "mode 默认 nearest、也接受 bilinear；bilinear 的 align_corners=False "
        "用半像素对齐（(dst+0.5)*in/out-0.5），True 用角对齐（dst*(in-1)/(out-1)）；"
        "bicubic / 三线性 / antialias / scale_factor 未实现",
    )
)


def _infer_interpolate_backward(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    dy = inputs[0]
    shape = _as_shape(attrs["input_shape"])
    if len(shape) != 4:
        raise ValueError(f"input_shape 必须是 NCHW 四元组，得到 {shape}")
    if dy.rank != 4:
        raise ValueError(f"interpolate_backward 要求 dy 是秩 4，得到秩 {dy.rank}")
    mode = _interp_mode(attrs)
    _interp_align(attrs, mode)
    _interp_size(attrs)
    return TensorType(dy.dtype, shape)


def _ref_interpolate_backward(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """插值对输入的梯度：逐个输出格把 dy 散射回它的源位置。

    nearest 一个源点、bilinear 四个角；多个输出格可能命中同一输入格
    （最近邻的重复采样、bilinear 在 in/out 不成比例时的重叠），所以必须
    ``+=`` / 设备侧 atomicAdd，不能覆盖。
    """
    dy = args[0]
    n, c, h, w = (int(d) for d in attrs["input_shape"])
    mode = _interp_mode(attrs)
    align = _interp_align(attrs, mode)
    oh, ow = _interp_size(attrs)
    if dy.shape != (n, c, oh, ow):
        raise ValueError(
            f"interpolate_backward 的 dy 形状应为 {(n, c, oh, ow)}，得到 {dy.shape}"
        )
    ih = _interp_axis(h, oh, align, mode == "nearest")
    iw = _interp_axis(w, ow, align, mode == "nearest")
    dx = np.zeros((n, c, h, w), dtype=np.float64)
    dyf = dy.astype(np.float64)
    if mode == "nearest":
        hmap, wmap = ih[0], iw[0]
        for i in range(oh):
            for j in range(ow):
                dx[:, :, hmap[i], wmap[j]] += dyf[:, :, i, j]
        return dx.astype(dy.dtype)

    h0, h1, lh = ih
    w0, w1, lw = iw
    for i in range(oh):
        for j in range(ow):
            part = dyf[:, :, i, j]
            dx[:, :, h0[i], w0[j]] += part * ((1.0 - lh[i]) * (1.0 - lw[j]))
            dx[:, :, h0[i], w1[j]] += part * ((1.0 - lh[i]) * lw[j])
            dx[:, :, h1[i], w0[j]] += part * (lh[i] * (1.0 - lw[j]))
            dx[:, :, h1[i], w1[j]] += part * (lh[i] * lw[j])
    return dx.astype(dy.dtype)


register(
    Op(
        name="interpolate_backward",
        infer=_infer_interpolate_backward,
        reference=_ref_interpolate_backward,
        attrs_schema=("input_shape", "size", "mode", "align_corners"),
        doc="插值对输入的梯度：dy(N,C,OH,OW) -> dx(N,C,H,W)，input_shape 为原输入形状；"
        "重复命中的输入位置累加（nearest 的重复采样、bilinear 的四角），只定义一阶",
    )
)


# --------------------------------------------------------------------------
# 归一化与归约
# --------------------------------------------------------------------------


def _infer_softmax(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    x = inputs[0]
    if len(inputs) > 1:
        scale = inputs[1]
        if not (scale.is_static and scale.numel() == 1):
            raise ValueError(f"softmax 的 scale 必须是标量，得到 {scale}")
    return x


def _ref_softmax(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x = args[0]
    if len(args) > 1:
        x = x * args[1]
    axis = attrs.get("axis", -1)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


register(
    Op(
        name="softmax",
        infer=_infer_softmax,
        reference=_ref_softmax,
        attrs_schema=("axis",),
        doc="沿 axis 的数值稳定 softmax；可选第二输入为标量缩放，等价于 softmax(x * scale)",
    )
)


def _infer_logsumexp(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 1:
        raise ValueError(f"logsumexp 需要 1 个输入，得到 {len(inputs)}")
    t = inputs[0]
    if not t.dtype.is_float:
        raise TypeError(
            f"logsumexp 只接受浮点输入，得到 {t.dtype.value}："
            "整数上的 exp / log 会先在 dtype 上截断，结果不是这个算子承诺的值"
        )
    axes = _norm_axes(attrs.get("axis", -1), t.rank)
    if len(axes) != 1:
        raise ValueError(f"logsumexp 只接受一个轴，得到 {axes}")
    if attrs.get("keepdims", False):
        shape = tuple(1 if i in axes else d for i, d in enumerate(t.shape))
    else:
        shape = tuple(d for i, d in enumerate(t.shape) if i not in axes)
    return TensorType(t.dtype, shape)


def _ref_logsumexp(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """log(sum(exp(x))) 的数值稳定实现：先减最大值再算。

    直接 ``np.log(np.sum(np.exp(x)))`` 在 x 含 90 以上的数时先溢出成 inf，
    在 x 全为 -100 时先下溢成 log(0)= -inf；减最大值后两种情形都回到正常区间。
    f16 在 float32 里算，理由同 ``cumsum``。
    """
    x = args[0]
    axis = _norm_axes(attrs.get("axis", -1), x.ndim)[0]
    work = x.astype(np.float32) if x.dtype == np.float16 else x
    keep = attrs.get("keepdims", False)
    m = np.max(work, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(work - m), axis=axis, keepdims=True))
    if not keep:
        out = np.squeeze(out, axis=axis)
    return out.astype(x.dtype, copy=False)


register(
    Op(
        name="logsumexp",
        infer=_infer_logsumexp,
        reference=_ref_logsumexp,
        attrs_schema=("axis", "keepdims"),
        doc="沿单个轴的 log(sum(exp(x)))，先减最大值保证不溢出 / 不下溢；"
        "反向是 g 乘同轴 softmax（不需要单独的反向内核）。与 softmax 不重复："
        "后者归一化成概率，这里保留绝对量级（配分函数的对数）",
    )
)


def _infer_attention(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) not in (3, 4):
        raise ValueError(f"attention 需要 3 或 4 个输入（q, k, v[, mask]），得到 {len(inputs)}")
    q, k, v = inputs[0], inputs[1], inputs[2]
    for label, t in (("q", q), ("k", k), ("v", v)):
        if t.rank < 2:
            raise ValueError(f"attention 的 {label} 至少是 2 维，得到 {t}")
        if t.dtype != q.dtype:
            raise TypeError(
                f"attention 的 q/k/v dtype 必须一致，得到 {q.dtype.value} 与 {t.dtype.value}"
            )
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(
            f"attention 的 q/k 内维必须相等，得到 {q.shape[-1]} 与 {k.shape[-1]}"
        )
    if k.shape[-2] != v.shape[-2]:
        raise ValueError(
            f"attention 的 k/v 序列长度必须相等，得到 {k.shape[-2]} 与 {v.shape[-2]}"
        )
    if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
        raise ValueError(
            f"attention 要求 q/k/v 的批维完全一致，得到 "
            f"{q.shape[:-2]} / {k.shape[:-2]} / {v.shape[:-2]}；"
            "批维广播会让反向必须做运行期归约（与 conv 的符号维拒绝同一条理由）"
        )
    if len(inputs) == 4:
        mask = inputs[3]
        if mask.dtype != q.dtype:
            raise TypeError(
                f"attention 的 mask dtype 必须与 q 一致，得到 {mask.dtype.value} 与 {q.dtype.value}"
            )
        if mask.rank < 2:
            raise ValueError(f"attention 的 mask 至少是 2 维，得到 {mask}")
        if isinstance(mask.shape[-1], int) and isinstance(k.shape[-2], int) and mask.shape[-1] != k.shape[-2]:
            raise ValueError(
                f"attention 的 mask 最后一维 {mask.shape[-1]} 必须等于 k 的序列长度 {k.shape[-2]}"
            )
    float(attrs.get("scale", 1.0))
    p = float(attrs.get("dropout", 0.0))
    if not 0.0 <= p < 1.0:
        raise ValueError(f"attention 的 dropout 必须在 [0, 1) 内，得到 {p}")
    return TensorType(q.dtype, q.shape[:-2] + (q.shape[-2], v.shape[-1]))


def _ref_attention(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """``softmax(q @ k^T * scale [+ mask]) @ v``，可选对概率做 dropout。

    这里刻意物化 ``(..., S, T)`` 的分数矩阵：参考后端的职责是**定义语义**，
    不是省内存。分块、不物化、按 causal 跳过上三角块都是后端实现的事，
    用它们替换这里的实现不会改变任何数值定义。
    """
    q, k, v = args[0], args[1], args[2]
    scale = float(attrs.get("scale", 1.0))
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    if len(args) > 3:
        mask = args[3]
        # mask 可以比 scores 多前导维（例如 (1,1,S,T)）：先去掉长度 1 的
        # 前导维，否则 numpy 会把 (B,H,S,T) + (1,1,S,T) 广播成 (1,B,H,S,T)，
        # 输出凭空多一维——IR 推导与运行期形状校验都会拒绝这种结果。
        while mask.ndim > scores.ndim:
            if mask.shape[0] != 1:
                raise ValueError(
                    f"attention 的 mask 前导维必须是 1，得到 {mask.shape}"
                )
            mask = mask[0]
        scores = scores + mask
    m = np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores - m)
    probs = e / np.sum(e, axis=-1, keepdims=True)
    dropout = float(attrs.get("dropout", 0.0))
    if dropout > 0.0:
        probs = probs * _ref_dropout_mask(
            [probs], {"p": dropout, "seed": attrs.get("seed", 0)}
        )
    return np.matmul(probs, v)


register(
    Op(
        name="attention",
        infer=_infer_attention,
        reference=_ref_attention,
        attrs_schema=("scale", "dropout", "seed"),
        doc="融合注意力：softmax(q @ k^T * scale [+ mask]) @ v，对 (..., S, D) x "
        "(..., T, D) x (..., T, E) 求值，输出 (..., S, E)。mask 是可选的加性偏置"
        "（禁止位取大负数，见 causal_mask_bias），dropout 作用在 softmax 概率上。"
        "IR 里只留一个节点：中间分数矩阵不进入图，反向图也因此只有一份；"
        "是否分块、是否跳过上三角由后端决定，不改变这里的语义",
    )
)


def _infer_layernorm(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) not in (3, 4):
        raise ValueError(
            f"layernorm 需要 (x, gamma, beta) 或 (x, gamma, beta, residual) 输入，得到 {len(inputs)}"
        )
    x = inputs[0]
    if len(inputs) == 4:
        res = inputs[3]
        if res != x:
            raise TypeError(
                f"layernorm 的 residual 必须与 x 同 dtype、同形状，得到 {res} 与 {x}"
            )
    return x


def _ref_layernorm(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x, gamma, beta = args[0], args[1], args[2]
    if len(args) > 3:
        # 残差在归一化之前相加：整段融合的关键是 add 的输出从不物化。
        # 用 numpy 的原 dtype 相加（f16 相加会按 f16 舍入），与内核的
        # "先舍入回存储类型、再累加"保持一致。
        x = x + args[3]
    axes = _norm_axes(attrs.get("axes", (-1,)), x.ndim)
    eps = float(attrs.get("eps", 1e-5))
    mean = np.mean(x, axis=axes, keepdims=True)
    var = np.var(x, axis=axes, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


register(
    Op(
        name="layernorm",
        infer=_infer_layernorm,
        reference=_ref_layernorm,
        attrs_schema=("axes", "eps"),
        doc="LayerNorm(x, gamma, beta)，gamma/beta 广播到最后若干维；"
        "第 4 个可选输入 residual 使 LayerNorm(x + residual) 成为一个节点"
        "（ADR-010 的输入扩展，残差因此不需要先物化）",
    )
)


def _infer_rmsnorm(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 2:
        raise ValueError(f"rmsnorm 需要 (x, gamma) 两个输入，得到 {len(inputs)}")
    x, gamma = inputs
    if x.dtype != gamma.dtype:
        raise TypeError(
            f"rmsnorm 要求 x 与 gamma 同 dtype，得到 {x.dtype.value} 与 {gamma.dtype.value}"
        )
    if gamma.rank > x.rank:
        raise ValueError(f"gamma 的秩 {gamma.rank} 大于 x 的秩 {x.rank}")
    if x.is_static and gamma.is_static:
        try:
            np.broadcast_shapes(x.shape, gamma.shape)
        except ValueError as exc:
            raise ValueError(f"gamma 形状 {gamma.shape} 无法广播到 {x.shape}: {exc}") from exc
    return TensorType(x.dtype, x.shape)


def _ref_rmsnorm(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x, gamma = args
    axes = _norm_axes(attrs.get("axes", (-1,)), x.ndim)
    eps = float(attrs.get("eps", 1e-6))
    ms = np.mean(x * x, axis=axes, keepdims=True)
    return x / np.sqrt(ms + eps) * gamma


register(
    Op(
        name="rmsnorm",
        infer=_infer_rmsnorm,
        reference=_ref_rmsnorm,
        attrs_schema=("axes", "eps"),
        doc="RMSNorm(x, gamma) = x / sqrt(mean(x^2) + eps) * gamma；"
        "没有均值中心化与 beta，比 LayerNorm 少一次归约（LLaMA/Mamba 的默认归一化）",
    )
)


def _norm_channel_axis(x: TensorType, attrs: dict, name: str) -> int:
    """通道轴规范化。BatchNorm 需要一个明确的通道维，不像 LayerNorm 可以给一组轴。"""
    if x.rank < 2:
        raise ValueError(f"{name} 要求秩 >= 2（batch 维 + 通道维），得到 {x.rank}")
    axis = int(attrs.get("axis", 1))
    if axis < 0:
        axis += x.rank
    if not 0 <= axis < x.rank:
        raise ValueError(f"{name} 的 axis 越界: {axis}，秩为 {x.rank}")
    return axis


def _infer_batch_norm(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 3:
        raise ValueError(f"batch_norm 需要 (x, gamma, beta) 三个输入，得到 {len(inputs)}")
    x, gamma, beta = inputs
    if x.dtype != gamma.dtype or x.dtype != beta.dtype:
        raise TypeError(
            f"batch_norm 要求三者同 dtype，得到 {x.dtype.value} / "
            f"{gamma.dtype.value} / {beta.dtype.value}"
        )
    axis = _norm_channel_axis(x, attrs, "batch_norm")
    for name, t in (("gamma", gamma), ("beta", beta)):
        if t.rank != 1:
            raise ValueError(f"batch_norm 的 {name} 必须是一维 (C,)，得到秩 {t.rank}")
        ch = x.shape[axis]
        if isinstance(ch, int) and isinstance(t.shape[0], int) and t.shape[0] != ch:
            raise ValueError(f"batch_norm 的 {name} 长度 {t.shape[0]} 与通道数 {ch} 不符")
    return TensorType(x.dtype, x.shape)


def _ref_batch_norm(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x, gamma, beta = args
    axis = int(attrs.get("axis", 1)) % x.ndim
    eps = float(attrs.get("eps", 1e-5))
    red = tuple(i for i in range(x.ndim) if i != axis)
    mean = np.mean(x, axis=red, keepdims=True)
    var = np.var(x, axis=red, keepdims=True)
    bshape = [1] * x.ndim
    bshape[axis] = -1
    return (x - mean) / np.sqrt(var + eps) * gamma.reshape(bshape) + beta.reshape(bshape)


register(
    Op(
        name="batch_norm",
        infer=_infer_batch_norm,
        reference=_ref_batch_norm,
        attrs_schema=("axis", "eps"),
        doc="BatchNorm（训练态）：沿除 axis 外的所有轴求均值与方差，再按 "
        "(x-mean)/sqrt(var+eps)*gamma+beta 归一化，gamma/beta 形状 (C,)。"
        "running stats 不在图内：训练循环用 batch_norm_stats 取 mean/var 后自己维护，"
        "因为“更新动量”是训练策略，不是张量语义",
    )
)


def _infer_batch_norm_stats(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 1:
        raise ValueError(f"batch_norm_stats 需要 1 个输入，得到 {len(inputs)}")
    x = inputs[0]
    axis = _norm_channel_axis(x, attrs, "batch_norm_stats")
    return TensorType(x.dtype, (2, x.shape[axis]))


def _ref_batch_norm_stats(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x = args[0]
    axis = int(attrs.get("axis", 1)) % x.ndim
    red = tuple(i for i in range(x.ndim) if i != axis)
    return np.stack([np.mean(x, axis=red), np.var(x, axis=red)]).astype(x.dtype)


register(
    Op(
        name="batch_norm_stats",
        infer=_infer_batch_norm_stats,
        reference=_ref_batch_norm_stats,
        attrs_schema=("axis",),
        doc="BatchNorm 的批统计量：返回 (2, C)，第 0 行是均值、第 1 行是方差。"
        "单独成算子是为了让训练循环能更新 running stats；它不参与反向，"
        "所以没有 vjp 规则——把它接进损失会在 backward 时明确报错，而不是静默给零",
    )
)


def _infer_reduce(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    t = inputs[0]
    if t.dtype is DType.BOOL:
        raise TypeError(
            "reduce_sum / reduce_max / reduce_mean 不接受 bool 输入："
            "bool 求和会退化成计数、最大/最小会退化成 OR/AND，两者都不是"
            "这个算子名字承诺的语义。判定掩码请用 reduce_all / reduce_any，"
            "计数请先 cast 到 f32"
        )
    axes = _norm_axes(attrs.get("axis", -1), t.rank)
    if attrs.get("keepdims", False):
        shape = tuple(1 if i in axes else d for i, d in enumerate(t.shape))
    else:
        shape = tuple(d for i, d in enumerate(t.shape) if i not in axes)
    return TensorType(t.dtype, shape)


def _reduce_axis(attrs: dict):
    """把 axis 归一化成 numpy 认识的形式。

    JSON 往返与反向规则都会把 axis 存成 list；numpy 只认 tuple/int。
    """
    axis = attrs.get("axis", -1)
    return tuple(axis) if isinstance(axis, list) else axis


def _ref_reduce_sum(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    return np.sum(
        args[0], axis=_reduce_axis(attrs), keepdims=attrs.get("keepdims", False)
    )


register(
    Op(
        name="reduce_sum",
        infer=_infer_reduce,
        reference=_ref_reduce_sum,
        attrs_schema=("axis", "keepdims"),
        doc="求和归约",
    )
)


def _ref_reduce_max(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    return np.max(
        args[0], axis=_reduce_axis(attrs), keepdims=attrs.get("keepdims", False)
    )


register(
    Op(
        name="reduce_max",
        infer=_infer_reduce,
        reference=_ref_reduce_max,
        attrs_schema=("axis", "keepdims"),
        doc="最大值归约。反向把梯度均分给所有等于最大值的元素（与 PyTorch 一致）",
    )
)


def _ref_reduce_mean(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    return np.mean(
        args[0], axis=_reduce_axis(attrs), keepdims=attrs.get("keepdims", False)
    )


register(
    Op(
        name="reduce_mean",
        infer=_infer_reduce,
        reference=_ref_reduce_mean,
        attrs_schema=("axis", "keepdims"),
        doc="均值归约（求和不带缩放，缩放放在反向）",
    )
)


def _infer_reduce_bool(what: str):
    """bool 归约：输入必须是 bool，输出仍是 bool。

    与 ``reduce_sum`` 分开成算子，是为了让"这一维是不是全真 / 有没有真"
    这件事在 IR 里就没有歧义：``reduce_sum(bool)`` 的结果是计数还是 OR
    取决于实现，而 ``reduce_all`` / ``reduce_any`` 只有一种读法。
    """

    def infer(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
        t = inputs[0]
        if t.dtype is not DType.BOOL:
            raise TypeError(
                f"{what} 只接受 bool 输入，得到 {t.dtype.value}；"
                "浮点掩码请先用比较算子落成 bool"
            )
        axes = _norm_axes(attrs.get("axis", -1), t.rank)
        if attrs.get("keepdims", False):
            shape = tuple(1 if i in axes else d for i, d in enumerate(t.shape))
        else:
            shape = tuple(d for i, d in enumerate(t.shape) if i not in axes)
        return TensorType(DType.BOOL, shape)

    return infer


def _reduce_bool_ref(np_fn):
    def ref(args: List[np.ndarray], attrs: dict) -> np.ndarray:
        with np.errstate(all="ignore"):
            return np.asarray(
                np_fn(
                    np.asarray(args[0], dtype=np.bool_),
                    axis=_reduce_axis(attrs),
                    keepdims=attrs.get("keepdims", False),
                ),
                dtype=np.bool_,
            )

    return ref


for _name, _np_fn, _what in (
    ("reduce_all", np.all, "reduce_all"),
    ("reduce_any", np.any, "reduce_any"),
):
    register(
        Op(
            name=_name,
            infer=_infer_reduce_bool(_what),
            reference=_reduce_bool_ref(_np_fn),
            attrs_schema=("axis", "keepdims"),
            doc=f"bool 归约 {_name}：沿 axis 判定全真 / 有真，输出 bool；不可导",
        )
    )


def _infer_argmax(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    t = inputs[0]
    # numpy 的 argmax 一次只归约一个轴；这里同样只接受单轴，避免
    # "多轴时索引该按什么顺序编号"这种必须自己定义语义的模糊地带。
    axes = _norm_axes(attrs.get("axis", -1), t.rank)
    if len(axes) != 1:
        raise ValueError(f"argmax 只接受一个轴，得到 {axes}")
    if attrs.get("keepdims", False):
        shape = tuple(1 if i in axes else d for i, d in enumerate(t.shape))
    else:
        shape = tuple(d for i, d in enumerate(t.shape) if i not in axes)
    return TensorType(DType.I32, shape)


def _ref_argmax(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    axis = _norm_axes(attrs.get("axis", -1), args[0].ndim)[0]
    out = np.argmax(args[0], axis=axis)
    if attrs.get("keepdims", False):
        out = np.expand_dims(out, axis)
    return out.astype(np.int32)


register(
    Op(
        name="argmax",
        infer=_infer_argmax,
        reference=_ref_argmax,
        attrs_schema=("axis", "keepdims"),
        doc="沿单个轴取最大值的下标，输出 i32。它不可导：接进损失会在 backward 时"
        "明确报错，而不是静默给零",
    )
)


def _infer_argmin(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 1:
        raise ValueError(f"argmin 需要 1 个输入，得到 {len(inputs)}")
    t = inputs[0]
    axes = _norm_axes(attrs.get("axis", -1), t.rank)
    if len(axes) != 1:
        raise ValueError(f"argmin 只接受一个轴，得到 {axes}")
    if attrs.get("keepdims", False):
        shape = tuple(1 if i in axes else d for i, d in enumerate(t.shape))
    else:
        shape = tuple(d for i, d in enumerate(t.shape) if i not in axes)
    return TensorType(DType.I32, shape)


def _ref_argmin(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """沿单个轴取最小值的下标；并列取**第一个**（与 numpy / PyTorch 一致）。"""
    axis = _norm_axes(attrs.get("axis", -1), args[0].ndim)[0]
    out = np.argmin(args[0], axis=axis)
    if attrs.get("keepdims", False):
        out = np.expand_dims(out, axis)
    return out.astype(np.int32)


register(
    Op(
        name="argmin",
        infer=_infer_argmin,
        reference=_ref_argmin,
        attrs_schema=("axis", "keepdims"),
        doc="沿单个轴取最小值的下标，输出 i32（并列取第一个）。与 argmax 同构、"
        "不可导：索引不是连续量，接进损失会在 backward 时明确报错",
    )
)


def _infer_sort(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 1:
        raise ValueError(f"sort 需要 1 个输入，得到 {len(inputs)}")
    t = inputs[0]
    axes = _norm_axes(attrs.get("axis", -1), t.rank)
    if len(axes) != 1:
        raise ValueError(f"sort 只接受一个轴，得到 {axes}")
    # descending 拼错成字符串会让参考实现静默按 False 走，所以钉住类型。
    desc = attrs.get("descending", False)
    if not isinstance(desc, bool):
        raise TypeError(f"sort 的 descending 必须是 bool，得到 {desc!r}")
    return t


def _ref_sort(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """沿单个轴升序（或降序）排序，**不返回置换下标**。

    只给值不给下标，是因为下标是整数、接进损失没有意义；需要下标请用
    ``topk_indices`` / ``argmax``。NaN 的位置随 numpy 的 ``sort``：升序时在末尾，
    降序时（整体反转）在最前面。
    """
    axis = _norm_axes(attrs.get("axis", -1), args[0].ndim)[0]
    out = np.sort(args[0], axis=axis)
    if attrs.get("descending", False):
        out = np.flip(out, axis=axis)
    return out


register(
    Op(
        name="sort",
        infer=_infer_sort,
        reference=_ref_sort,
        attrs_schema=("axis", "descending"),
        doc="沿单个轴排序（升序 / 降序），只输出值。不可导：没有置换下标就无从把"
        "梯度搬回原位置，接进损失会在 backward 时明确报错",
    )
)


# --------------------------------------------------------------------------
# 前缀和 / 三角掩码
# --------------------------------------------------------------------------


def _infer_cumsum(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 1:
        raise ValueError(f"cumsum 需要 1 个输入，得到 {len(inputs)}")
    axes = _norm_axes(attrs.get("axis", -1), inputs[0].rank)
    if len(axes) != 1:
        raise ValueError(f"cumsum 只接受一个轴，得到 {axes}")
    return inputs[0]


def _ref_cumsum(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """沿单个轴的**含端点**前缀和。

    f16 在 float32 里累加：前缀和的每一步都是最终输出，逐项 f16 累加在长序列
    上会先溢出再失真，而这里没有"最后再 cast 一次"能补救的机会。这与 PyTorch
    归约类算子的 acc_type 口径一致（累加用 float，结果落回原 dtype）。
    """
    x = args[0]
    axis = _norm_axes(attrs.get("axis", -1), x.ndim)[0]
    work = x.astype(np.float32) if x.dtype == np.float16 else x
    return np.cumsum(work, axis=axis).astype(x.dtype, copy=False)


register(
    Op(
        name="cumsum",
        infer=_infer_cumsum,
        reference=_ref_cumsum,
        attrs_schema=("axis",),
        doc="沿单个轴的含端点前缀和；SSM 分块扫描的块内 log-space 前缀和"
        "（数值稳定性的命脉）与因果累积均值都靠它。与 scan_linear 不重复："
        "后者是 a_t 可变的递归，这里是 a_t == 1 的纯累加，后端可以按纯前缀和"
        "优化（分块 / 并行），不必套进递归的串行骨架",
    )
)


def _infer_cumprod(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 1:
        raise ValueError(f"cumprod 需要 1 个输入，得到 {len(inputs)}")
    axes = _norm_axes(attrs.get("axis", -1), inputs[0].rank)
    if len(axes) != 1:
        raise ValueError(f"cumprod 只接受一个轴，得到 {axes}")
    return inputs[0]


def _ref_cumprod(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """沿单个轴的**含端点**前缀积。

    f16 在 float32 里累乘，理由与 ``cumsum`` 相同：每一步都是最终输出，
    逐项 f16 累乘会先溢出再失真。
    """
    x = args[0]
    axis = _norm_axes(attrs.get("axis", -1), x.ndim)[0]
    work = x.astype(np.float32) if x.dtype == np.float16 else x
    return np.cumprod(work, axis=axis).astype(x.dtype, copy=False)


register(
    Op(
        name="cumprod",
        infer=_infer_cumprod,
        reference=_ref_cumprod,
        attrs_schema=("axis",),
        doc="沿单个轴的含端点前缀积；衰减因子的连乘轨迹（如逐帧的 α^t 权重）"
        "用它比显式幂次更稳。反向由伴随扫描给出：dx = exclusive_cumprod(x) * "
        "scan_linear_backward(x, g)，不在乘法链上做除法，所以 x 含零也成立",
    )
)


# --------------------------------------------------------------------------
# 重排：flip / roll（逐元素搬运，不改变形状与 dtype）
# --------------------------------------------------------------------------


def _infer_flip(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 1:
        raise ValueError(f"flip 需要 1 个输入，得到 {len(inputs)}")
    t = inputs[0]
    axes = _norm_axes(attrs.get("axis", -1), t.rank)
    if len(axes) != 1:
        raise ValueError(f"flip 只接受一个轴，得到 {axes}")
    return t


def _ref_flip(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    axis = _norm_axes(attrs.get("axis", -1), args[0].ndim)[0]
    return np.flip(args[0], axis=axis).copy()


register(
    Op(
        name="flip",
        infer=_infer_flip,
        reference=_ref_flip,
        attrs_schema=("axis",),
        doc="沿单个轴反转；形状与 dtype 不变。反向是同一个算子（反转自伴）；"
        "cumprod 的伴随扫描与自回归的倒序展开都靠它",
    )
)


def _infer_roll(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 1:
        raise ValueError(f"roll 需要 1 个输入，得到 {len(inputs)}")
    t = inputs[0]
    axes = _norm_axes(attrs.get("axis", -1), t.rank)
    if len(axes) != 1:
        raise ValueError(f"roll 只接受一个轴，得到 {axes}")
    shift = attrs.get("shift", 0)
    if isinstance(shift, bool) or not isinstance(shift, int):
        raise TypeError(f"roll 的 shift 必须是整数，得到 {shift!r}")
    return t


def _ref_roll(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """沿单个轴循环移位；``shift`` 为负时向反方向移位（numpy 语义）。"""
    axis = _norm_axes(attrs.get("axis", -1), args[0].ndim)[0]
    return np.roll(args[0], int(attrs.get("shift", 0)), axis=axis).copy()


register(
    Op(
        name="roll",
        infer=_infer_roll,
        reference=_ref_roll,
        attrs_schema=("shift", "axis"),
        doc="沿单个轴循环移位，shift 可正可负（numpy 语义）。反向是 shift 取负的"
        "同一个算子；滑动窗口的相位对齐与周期填充都靠它",
    )
)


def _infer_tril(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    t = inputs[0]
    if t.rank < 2:
        raise ValueError(f"tril 至少需要秩 2，得到 {t.rank}")
    # diagonal 只影响取值，不影响形状；但拼错成字符串会让参考实现静默用 0，
    # 所以在构图期就把类型钉住。
    k = attrs.get("diagonal", 0)
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"tril 的 diagonal 必须是整数，得到 {k!r}")
    return t


def _ref_tril(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """最后两维的下三角保留（``k`` 为对角线偏移），其余置零。

    numpy 的 ``tril`` 对 ndim > 2 的数组正好作用在最后两维，与本算子的定义
    一致：前导维（batch / head）逐片独立，不做跨片操作。
    """
    return np.tril(args[0], k=int(attrs.get("diagonal", 0)))


register(
    Op(
        name="tril",
        infer=_infer_tril,
        reference=_ref_tril,
        attrs_schema=("diagonal",),
        doc="最后两维保留下三角（含对角线偏移 k），其余置零。"
        "分块 SSM 的块内因果权重与窗口注意力的因果掩码都靠它；"
        "反向是同一个算子（下三角是逐元素 0/1 掩码的乘法，掩码自伴）",
    )
)


# --------------------------------------------------------------------------
# 稀疏选择
# --------------------------------------------------------------------------


def _topk_axis_k_shape(shape: Tuple, attrs: dict, name: str) -> Tuple[int, int]:
    """``topk`` 族共用的属性校验：单个轴 + 正整数 k + k 不超过轴长。

    三个算子（mask / values / indices）必须对同一份 attrs 给出同样的解释，
    所以校验只写一遍；分开写迟早会出现"mask 拒绝 k=0 而 values 接受"这种
    只有边界输入才暴露的分叉。参考实现手里只有 ndarray，没有 ``TensorType``，
    所以按 shape 校验、``_topk_axis_k`` 只是包一层。
    """
    axes = _norm_axes(attrs.get("axis", -1), len(shape))
    if len(axes) != 1:
        raise ValueError(f"{name} 只接受一个轴，得到 {axes}")
    k = attrs.get("k")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"{name} 的 k 必须是整数，得到 {k!r}")
    if k <= 0:
        raise ValueError(f"{name} 的 k 必须为正，得到 {k}")
    dim = shape[axes[0]]
    if isinstance(dim, int) and k > dim:
        raise ValueError(f"{name} 的 k={k} 超过轴 {axes[0]} 的长度 {dim}")
    return axes[0], k


def _topk_axis_k(t: TensorType, attrs: dict, name: str) -> Tuple[int, int]:
    return _topk_axis_k_shape(t.shape, attrs, name)


def _topk_shape(t: TensorType, axis: int, k: int) -> Tuple:
    return tuple(k if i == axis else d for i, d in enumerate(t.shape))


def _infer_topk_mask(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    t = inputs[0]
    _topk_axis_k(t, attrs, "topk_mask")
    return t


def _topk_order(x: np.ndarray, axis: int, k: int) -> np.ndarray:
    """前 k 大的**下标**，按值降序、并列时小下标优先。

    ``kind="stable"`` 是"并列时小下标优先"的实现：排序键相同的元素保持原
    顺序，也就是下标升序。NaN 先替换成 +inf 再取负，于是被当成最大值排到
    最前——这是 ``torch.topk`` 的口径，numpy 原生的 argsort 会把 NaN 丢到
    末尾（升序时的未定义区），两边的 topk 结果会完全不同。
    """
    if x.dtype.kind == "f":
        key = np.where(np.isnan(x), np.inf, x)
    else:
        key = x
    order = np.argsort(-key, axis=axis, kind="stable")
    return np.take(order, np.arange(k), axis=axis)


def _ref_topk_mask(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """沿单个轴取前 k 大的位置置 1，其余置 0；同值按下标小者优先。

    只输出"选谁"的掩码，不输出被选值本身：MoE 的门控权重仍由 softmax
    概率乘掩码得到，梯度因此只经概率那条边回传，选择本身不可导（与
    ``argmax`` 同一条约定）。稳定排序保证同一输入在下一次执行里选出同一组
    下标，DP 的多份参考执行不会因为并列值而产生不同的路由。
    """
    x = args[0]
    axis = _norm_axes(attrs.get("axis", -1), x.ndim)[0]
    k = int(attrs["k"])
    picked = _topk_order(x, axis, k)
    mask = np.zeros_like(x)
    np.put_along_axis(mask, picked, 1, axis=axis)
    return mask


register(
    Op(
        name="topk_mask",
        infer=_infer_topk_mask,
        reference=_ref_topk_mask,
        attrs_schema=("k", "axis"),
        doc="沿单个轴取前 k 大的位置生成同形状 0/1 掩码，dtype 与输入一致。"
        "它不可导（选择是离散的），接进损失会在 backward 时明确报错，"
        "而不是静默给零；MoE 路由把掩码只用在门控权重的乘法里，"
        "梯度经 softmax 概率那条边回传",
    )
)


def _infer_topk_values(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    t = inputs[0]
    if t.dtype is DType.BOOL:
        raise TypeError("topk_values 不接受 bool 张量；掩码请用 topk_mask")
    axis, k = _topk_axis_k(t, attrs, "topk_values")
    return TensorType(t.dtype, _topk_shape(t, axis, k))


def _ref_topk_values(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x = args[0]
    axis, k = _topk_axis_k_shape(tuple(x.shape), attrs, "topk_values")
    return np.take_along_axis(x, _topk_order(x, axis, k), axis=axis)


register(
    Op(
        name="topk_values",
        infer=_infer_topk_values,
        reference=_ref_topk_values,
        attrs_schema=("k", "axis"),
        doc="沿单个轴取前 k 大的值本身，dtype 与输入一致，按值降序、并列时小下标优先。"
        "与 topk_indices 是两个算子而不是一个双输出算子：IR 节点是单输出，"
        "拆开之后两者都能独立参与 DCE。它不可导（下标由离散比较决定）；"
        "需要梯度请用 topk_mask 乘原值，让梯度走乘法那条边",
    )
)


def _infer_topk_indices(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    t = inputs[0]
    if t.dtype is DType.BOOL:
        raise TypeError("topk_indices 不接受 bool 张量；掩码请用 topk_mask")
    axis, k = _topk_axis_k(t, attrs, "topk_indices")
    return TensorType(DType.I32, _topk_shape(t, axis, k))


def _ref_topk_indices(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x = args[0]
    axis, k = _topk_axis_k_shape(tuple(x.shape), attrs, "topk_indices")
    return _topk_order(x, axis, k).astype(np.int32)


register(
    Op(
        name="topk_indices",
        infer=_infer_topk_indices,
        reference=_ref_topk_indices,
        attrs_schema=("k", "axis"),
        doc="沿单个轴取前 k 大的下标，输出 i32，顺序与 topk_values 逐位一致"
        "（值降序、并列时小下标优先）。它是离散结果，不可导；"
        "torch.topk 的两个槽位由前端分别映射到 topk_values / topk_indices",
    )
)


# --------------------------------------------------------------------------
# 形状与类型算子
# --------------------------------------------------------------------------


def _infer_reshape(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    t = inputs[0]
    target = _as_shape(attrs["shape"])

    # reshape 不允许引入新的未知量：符号维必须在源类型中出现过。目标里可以
    # 出现表达式（``(B,4) -> (4*B,)``），它的取值到执行期按符号绑定求值；
    # 元素数守恒无法在结构上判定时交给 reshape 本身——形状真的对不上，执行期
    # 的 numpy/torch 会报错，而不是像"静态检查漏过"那样算出错误结果。
    for d in target:
        unknown = dim_symbols(d) - t.symbols
        if unknown:
            raise ValueError(f"reshape 引入了未定义的符号维: {sorted(unknown)}")

    if t.is_static and all(isinstance(d, int) for d in target):
        n_src, n_dst = t.numel(), 1
        for d in target:
            n_dst *= d
        if n_src != n_dst:
            raise ValueError(f"reshape 元素个数不守恒: {n_src} -> {n_dst}")

    return TensorType(t.dtype, target)


def _ref_reshape(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    return np.reshape(args[0], attrs["shape"])


register(
    Op(
        name="reshape",
        infer=_infer_reshape,
        reference=_ref_reshape,
        attrs_schema=("shape",),
        doc="改变形状，元素顺序不变；不接受 -1，请显式给全维",
    )
)


def _infer_transpose(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    t = inputs[0]
    perm = tuple(attrs["perm"])
    if sorted(perm) != list(range(t.rank)):
        raise ValueError(f"非法置换 {perm}，秩为 {t.rank}")
    return TensorType(t.dtype, tuple(t.shape[i] for i in perm))


def _ref_transpose(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    return np.transpose(args[0], attrs["perm"])


register(
    Op(
        name="transpose",
        infer=_infer_transpose,
        reference=_ref_transpose,
        attrs_schema=("perm",),
        doc="维度置换",
    )
)


def _infer_cast(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    return TensorType(DType(attrs["dtype"]), inputs[0].shape)


def _ref_cast(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    return args[0].astype(DType(attrs["dtype"]).np_dtype)


register(
    Op(
        name="cast",
        infer=_infer_cast,
        reference=_ref_cast,
        attrs_schema=("dtype",),
        doc="元素类型转换",
    )
)


def _infer_slice(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    t = inputs[0]
    dim = int(attrs["dim"]) % t.rank
    start = int(attrs.get("start", 0))
    stop = attrs.get("stop", t.shape[dim])
    stop = t.shape[dim] if stop is None else int(stop)
    if stop <= start:
        raise ValueError(f"slice 区间非法: start={start} stop={stop}")
    shape = list(t.shape)
    shape[dim] = stop - start
    return TensorType(t.dtype, tuple(shape))


def _ref_slice(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x = args[0]
    dim = int(attrs["dim"]) % x.ndim
    start = int(attrs.get("start", 0))
    stop = attrs.get("stop", x.shape[dim])
    stop = x.shape[dim] if stop is None else int(stop)
    index = [slice(None)] * x.ndim
    index[dim] = slice(start, stop)
    return x[tuple(index)]


register(
    Op(
        name="slice",
        infer=_infer_slice,
        reference=_ref_slice,
        attrs_schema=("dim", "start", "stop"),
        doc="沿单个维度切片 [start, stop)，step 固定为 1",
    )
)


def _infer_broadcast_to(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    t = inputs[0]
    target = _as_shape(attrs["shape"])
    if t.is_static and all(isinstance(d, int) for d in target):
        try:
            np.broadcast_shapes(t.shape, target)
        except ValueError as exc:
            raise ValueError(f"broadcast_to 无法把 {t.shape} 广播到 {target}: {exc}") from exc
    return TensorType(t.dtype, target)


def _ref_broadcast_to(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    return np.array(np.broadcast_to(args[0], _as_shape(attrs["shape"])))


register(
    Op(
        name="broadcast_to",
        infer=_infer_broadcast_to,
        reference=_ref_broadcast_to,
        attrs_schema=("shape",),
        doc="显式广播到目标形状（物料化，不返回视图）",
    )
)


def _infer_pad(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    t = inputs[0]
    dim = int(attrs["dim"]) % t.rank
    before = int(attrs.get("before", 0))
    after = int(attrs.get("after", 0))
    if before < 0 or after < 0:
        raise ValueError(f"pad 的 before/after 必须 >= 0，得到 {before} 与 {after}")
    shape = list(t.shape)
    if isinstance(shape[dim], int):
        shape[dim] = int(shape[dim]) + before + after
    else:
        raise ValueError(
            f"pad 的第 {dim} 维是符号维 {shape[dim]!r}，加上常数后需要派生符号；"
            "请给出静态维"
        )
    return TensorType(t.dtype, tuple(shape))


def _ref_pad(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x = args[0]
    dim = int(attrs["dim"]) % x.ndim
    before = int(attrs.get("before", 0))
    after = int(attrs.get("after", 0))
    widths = [(0, 0)] * x.ndim
    widths[dim] = (before, after)
    return np.pad(x, widths)


register(
    Op(
        name="pad",
        infer=_infer_pad,
        reference=_ref_pad,
        attrs_schema=("dim", "before", "after"),
        doc="沿单个维度两侧补零；是 slice 的反向，语义与切片严格对称",
    )
)


def _infer_concat(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if not inputs:
        raise ValueError("concat 至少需要一个输入")
    first = inputs[0]
    ax = _norm_axes(attrs.get("axis", 0), first.rank)[0]
    shape = list(first.shape)
    total = 0
    for t in inputs:
        if t.dtype != first.dtype:
            raise TypeError(
                f"concat 要求所有输入 dtype 一致，得到 {first.dtype.value} 与 {t.dtype.value}"
            )
        if t.rank != first.rank:
            raise ValueError(f"concat 要求所有输入同秩，得到 {first.rank} 与 {t.rank}")
        for i in range(first.rank):
            if i == ax:
                continue
            if t.shape[i] != first.shape[i]:
                raise ValueError(
                    f"concat 的非拼接维必须一致：第 {i} 维为 {first.shape[i]} 与 {t.shape[i]}"
                )
        dim = t.shape[ax]
        if not isinstance(dim, int):
            raise ValueError(
                f"concat 的拼接维必须是静态量，得到符号维 {dim!r}；"
                "输出形状需要它求和，而符号维相加不是已定义的符号运算"
            )
        total += dim
    shape[ax] = total
    return TensorType(first.dtype, tuple(shape))


def _ref_concat(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    return np.concatenate(args, axis=_norm_axes(attrs.get("axis", 0), args[0].ndim)[0])


register(
    Op(
        name="concat",
        infer=_infer_concat,
        reference=_ref_concat,
        attrs_schema=("axis",),
        doc="沿 axis 拼接任意个同秩、非拼接维一致的张量；拼接维必须静态（输出形状要相加）",
    )
)


def _infer_embedding(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 2:
        raise ValueError(f"embedding 需要 (table, ids) 两个输入，得到 {len(inputs)}")
    table, ids = inputs
    if table.rank < 1:
        raise ValueError("embedding 的 table 至少是秩 1（V, ...）")
    if ids.dtype not in (DType.I32, DType.I8):
        raise TypeError(f"embedding 的 ids 必须是整数下标，得到 {ids.dtype.value}")
    if not isinstance(table.shape[0], int):
        raise ValueError(f"embedding 的词表大小必须静态，得到 {table.shape[0]!r}")
    return TensorType(table.dtype, tuple(ids.shape) + tuple(table.shape[1:]))


def _check_embedding_ids(ids: np.ndarray, rows: int, op: str) -> np.ndarray:
    """下标必须落在 ``[0, rows)``，越界显式报错。

    正向越界 numpy 自己会抛 IndexError，但**负下标**会静默回绕到倒数几行
    （``table[-1]`` 就是最后一行），反向的 ``np.add.at`` 同样会回绕——那是一种
    "看起来算完了"的错误结果。所以在参考语义里统一检查一次。
    """
    idx = np.asarray(ids)
    if idx.size:
        low, high = int(idx.min()), int(idx.max())
        if low < 0 or high >= rows:
            raise ValueError(
                f"{op} 的下标必须落在 [0, {rows})，得到 [{low}, {high}]"
            )
    return idx


def _ref_embedding(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """按第一维查表：``out[i] = table[ids[i]]``，越界显式报错。"""
    table, ids = args
    _check_embedding_ids(ids, int(table.shape[0]), "embedding")
    return table[ids]


register(
    Op(
        name="embedding",
        infer=_infer_embedding,
        reference=_ref_embedding,
        doc="查表：table(V, ...) 与整数 ids(...) -> (ids.shape..., table.shape[1:])；"
        "词表大小必须静态（越界检查与反向积累都依赖它）",
    )
)


def _infer_embedding_backward(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    dy = inputs[0]
    shape = _as_shape(attrs["table_shape"])
    if len(shape) < 1:
        raise ValueError(f"table_shape 至少需要一维，得到 {shape}")
    if isinstance(shape[0], str):
        raise ValueError(f"词表大小必须静态，得到 {shape[0]!r}")
    return TensorType(dy.dtype, shape)


def _ref_embedding_backward(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """查表的反向：把每一条 dy 散射累加回对应行。

    重复下标必须**累加**，不能用赋值——同一个 token 在一个 batch 里出现多次
    是常态。``np.add.at`` 正是为此存在的无缓冲散射。
    """
    dy, ids = args
    shape = tuple(int(d) for d in attrs["table_shape"])
    expected = tuple(ids.shape) + shape[1:]
    if dy.shape != expected:
        raise ValueError(f"embedding_backward 的 dy 形状应为 {expected}，得到 {dy.shape}")
    _check_embedding_ids(ids, shape[0], "embedding_backward")
    out = np.zeros(shape, dtype=dy.dtype)
    np.add.at(out, ids, dy)
    return out


register(
    Op(
        name="embedding_backward",
        infer=_infer_embedding_backward,
        reference=_ref_embedding_backward,
        attrs_schema=("table_shape",),
        doc="embedding 对表参数的梯度：dy(..., D) 与 ids(...) -> dtable(V, D)；重复下标累加",
    )
)


# --------------------------------------------------------------------------
# 任意轴下标取值与交叉熵
# --------------------------------------------------------------------------
# ``embedding`` 只沿表的第一维取整行，形状语义固定为
# ``table.shape[1:]`` 追加在 ids 后面。``gather`` 是它的一般形态：沿任意
# 一维按下标取值，输出形状等于 idx 的形状（与 ``torch.gather`` 一致）。
# 交叉熵必须用它取 label 那一格的 logit；拿 embedding 去凑会在
# ``table=[N, V]``、``ids=[N]`` 时得到 ``[N, N]`` 而不是逐样本收集。


def _norm_axis(axis: Any, rank: int, op: str) -> int:
    """把负轴归一化，并显式拒绝越界。

    不用 ``%``：``dim=-rank-1`` 会被绕回一个合法轴，静默选错方向。
    """
    a = int(axis)
    if a < 0:
        a += rank
    if not 0 <= a < rank:
        raise ValueError(f"{op} 的 dim 越界：得到 {axis}，秩为 {rank}")
    return a


def _check_gather_shapes(
    x_shape: Sequence[Dim], idx_shape: Sequence[Dim], dim: int, op: str
) -> None:
    """gather 系列的形状约束：秩相同，除 dim 外每一维相等。

    这是 ``torch.gather`` 在"idx 各维不超过 x"之上的收紧形态。索引表比
    目标小的时候，输出里没有对应坐标，语义仍然成立；但参考实现与设备内核
    都要额外处理"输出与 x 的 outer 不一致"，收益为零。收紧后 ``x`` 与
    ``idx`` 只差 dim 那一维，内核按 ``(outer, dim, inner)`` 三轴寻址即可。
    """
    if len(x_shape) != len(idx_shape):
        raise ValueError(
            f"{op} 的 idx 秩必须与 x 相同（{len(x_shape)}），得到 {len(idx_shape)}"
        )
    for d, (a, b) in enumerate(zip(x_shape, idx_shape)):
        if d == dim:
            continue
        if a != b:
            raise ValueError(
                f"{op} 要求除 dim={dim} 外每一维相等：第 {d} 维是 {a} 与 {b}"
            )


def _infer_gather(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 2:
        raise ValueError(f"gather 需要 (x, idx) 两个输入，得到 {len(inputs)}")
    x, idx = inputs
    if x.rank < 1:
        raise ValueError(f"gather 的 x 至少是 1 维，得到 {x}")
    if idx.dtype not in (DType.I32, DType.I8):
        raise TypeError(f"gather 的 idx 必须是整数下标，得到 {idx.dtype.value}")
    dim = _norm_axis(attrs.get("dim", -1), x.rank, "gather")
    _check_gather_shapes(x.shape, idx.shape, dim, "gather")
    return TensorType(x.dtype, tuple(idx.shape))


def _check_gather_ids(idx: np.ndarray, extent: int, op: str) -> np.ndarray:
    """下标必须落在 ``[0, extent)``。

    与 ``_check_embedding_ids`` 同一条理由：numpy 对负下标会静默回绕，
    设备端若只做 clamp 就会给出"看起来算完了"的错误结果。这里统一在
    参考语义里检查，设备端用 err 标量走同一条报错通路。
    """
    idx = np.asarray(idx)
    if idx.size:
        low, high = int(idx.min()), int(idx.max())
        if low < 0 or high >= extent:
            raise ValueError(f"{op} 的下标必须落在 [0, {extent})，得到 [{low}, {high}]")
    return idx


def _ref_gather(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x, idx = args
    dim = _norm_axis(attrs.get("dim", -1), x.ndim, "gather")
    _check_gather_shapes(x.shape, idx.shape, dim, "gather")
    _check_gather_ids(idx, int(x.shape[dim]), "gather")
    return np.take_along_axis(x, idx.astype(np.int64), axis=dim)


register(
    Op(
        name="gather",
        infer=_infer_gather,
        reference=_ref_gather,
        attrs_schema=("dim",),
        doc="沿 dim 按下标取值：out[i...] = x[..., idx[i...], ...]；"
        "输出形状等于 idx，除 dim 外每一维必须与 x 相等（torch.gather 的收紧形态）",
    )
)


def _infer_gather_backward(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 2:
        raise ValueError(f"gather_backward 需要 (dy, idx) 两个输入，得到 {len(inputs)}")
    dy, idx = inputs
    shape = _as_shape(attrs["x_shape"])
    if len(shape) < 1:
        raise ValueError(f"gather_backward 的 x_shape 至少 1 维，得到 {shape}")
    if idx.dtype not in (DType.I32, DType.I8):
        raise TypeError(f"gather_backward 的 idx 必须是整数下标，得到 {idx.dtype.value}")
    dim = _norm_axis(attrs.get("dim", -1), len(shape), "gather_backward")
    if tuple(dy.shape) != tuple(idx.shape):
        raise ValueError(
            f"gather_backward 的 dy 形状必须等于 idx 形状 {idx.shape}，得到 {dy.shape}"
        )
    _check_gather_shapes(shape, dy.shape, dim, "gather_backward")
    return TensorType(dy.dtype, shape)


def _ref_gather_backward(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """``gather`` 对 x 的梯度：把 dy 散射累加回被选中的位置。

    重复下标必须累加：上游的同一个坐标可以被多次选中（例如词表里同一行
    在多处出现），赋值会丢掉除最后一次以外的全部贡献。
    """
    dy, idx = args
    shape = tuple(int(d) for d in attrs["x_shape"])
    dim = _norm_axis(attrs.get("dim", -1), len(shape), "gather_backward")
    if tuple(dy.shape) != tuple(idx.shape):
        raise ValueError(
            f"gather_backward 的 dy 形状必须等于 idx 形状 {idx.shape}，得到 {dy.shape}"
        )
    _check_gather_shapes(shape, dy.shape, dim, "gather_backward")
    _check_gather_ids(idx, shape[dim], "gather_backward")

    outer = int(np.prod(shape[:dim])) if dim else 1
    inner = int(np.prod(shape[dim + 1 :])) if dim + 1 < len(shape) else 1
    idx_dim = int(dy.shape[dim])
    flat_dy = dy.reshape(outer, idx_dim, inner)
    flat_idx = idx.astype(np.int64).reshape(outer, idx_dim, inner)
    dx = np.zeros(shape, dtype=dy.dtype)
    flat_dx = dx.reshape(outer, shape[dim], inner)
    o = np.arange(outer)[:, None, None]
    i = np.arange(inner)[None, None, :]
    np.add.at(flat_dx, (o, flat_idx, i), flat_dy)
    return dx


register(
    Op(
        name="gather_backward",
        infer=_infer_gather_backward,
        reference=_ref_gather_backward,
        attrs_schema=("x_shape", "dim"),
        doc="gather 对 x 的梯度：dy 与 idx -> dx(x_shape)；重复下标用散射累加",
    )
)


def _infer_cross_entropy(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 2:
        raise ValueError(f"cross_entropy 需要 (logits, target) 两个输入，得到 {len(inputs)}")
    logits, target = inputs
    if logits.rank < 1:
        raise ValueError(f"cross_entropy 的 logits 至少是 1 维，得到 {logits}")
    if not logits.dtype.is_float:
        raise TypeError(f"cross_entropy 的 logits 必须是浮点，得到 {logits.dtype.value}")
    if target.dtype not in (DType.I32, DType.I8):
        raise TypeError(f"cross_entropy 的 target 必须是整数下标，得到 {target.dtype.value}")
    if tuple(target.shape) != tuple(logits.shape[:-1]):
        raise ValueError(
            f"cross_entropy 的 target 形状应为 {logits.shape[:-1]}（去掉词表维），"
            f"得到 {target.shape}"
        )
    vocab = logits.shape[-1]
    if isinstance(vocab, int) and vocab < 1:
        raise ValueError(f"cross_entropy 的词表维必须 >= 1，得到 {vocab}")
    return TensorType(logits.dtype, tuple(logits.shape[:-1]))


def _ref_cross_entropy(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """逐行 ``logsumexp(logits) - logits[target]``。

    刻意不物化 softmax：参考实现直接算对数域的和，既与 fused 内核同义，
    也不会在 ``(B, T, V)`` 上多出一份与 logits 同大的临时量。
    """
    logits, target = args
    vocab = int(logits.shape[-1])
    idx = _check_gather_ids(target, vocab, "cross_entropy")
    acc = np.float64 if logits.dtype == np.float64 else np.float32
    x = logits.astype(acc, copy=False)
    m = np.max(x, axis=-1, keepdims=True)
    lse = m[..., 0] + np.log(np.sum(np.exp(x - m), axis=-1))
    picked = np.take_along_axis(x, idx[..., None].astype(np.int64), axis=-1)[..., 0]
    return (lse - picked).astype(logits.dtype, copy=False)


register(
    Op(
        name="cross_entropy",
        infer=_infer_cross_entropy,
        reference=_ref_cross_entropy,
        doc="逐样本交叉熵：logits(..., V) 与 target(...) -> (...)；"
        "语义是 logsumexp - 目标 logit，不物化 softmax",
    )
)


def _infer_cross_entropy_backward(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    if len(inputs) != 3:
        raise ValueError(
            f"cross_entropy_backward 需要 (logits, target, dloss) 三个输入，得到 {len(inputs)}"
        )
    logits, target, dloss = inputs
    if logits.rank < 1:
        raise ValueError(f"cross_entropy_backward 的 logits 至少是 1 维，得到 {logits}")
    if target.dtype not in (DType.I32, DType.I8):
        raise TypeError(
            f"cross_entropy_backward 的 target 必须是整数下标，得到 {target.dtype.value}"
        )
    head = tuple(logits.shape[:-1])
    if tuple(target.shape) != head:
        raise ValueError(
            f"cross_entropy_backward 的 target 形状应为 {head}，得到 {target.shape}"
        )
    if tuple(dloss.shape) != head:
        raise ValueError(
            f"cross_entropy_backward 的 dloss 形状应为 {head}，得到 {dloss.shape}"
        )
    if dloss.dtype != logits.dtype:
        raise TypeError(
            f"cross_entropy_backward 的 dloss 与 logits dtype 必须一致，得到 "
            f"{dloss.dtype.value} 与 {logits.dtype.value}"
        )
    return logits


def _ref_cross_entropy_backward(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """``(softmax(logits) - onehot(target)) * dloss``，按样本缩放。"""
    logits, target, dloss = args
    vocab = int(logits.shape[-1])
    idx = _check_gather_ids(target, vocab, "cross_entropy_backward")
    head = tuple(logits.shape[:-1])
    if tuple(dloss.shape) != head:
        raise ValueError(
            f"cross_entropy_backward 的 dloss 形状应为 {head}，得到 {dloss.shape}"
        )
    acc = np.float64 if logits.dtype == np.float64 else np.float32
    x = logits.astype(acc, copy=False)
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    p = e / np.sum(e, axis=-1, keepdims=True)
    onehot = idx[..., None].astype(np.int64)
    np.put_along_axis(
        p, onehot, np.take_along_axis(p, onehot, axis=-1) - 1.0, axis=-1
    )
    p *= np.asarray(dloss, dtype=acc)[..., None]
    return p.astype(logits.dtype, copy=False)


register(
    Op(
        name="cross_entropy_backward",
        infer=_infer_cross_entropy_backward,
        reference=_ref_cross_entropy_backward,
        doc="cross_entropy 对 logits 的梯度：(softmax - onehot) * dloss；"
        "dloss 的形状是每样本损失形状（与 target 相同）",
    )
)


# --------------------------------------------------------------------------
# 序列扫描（SSM / 线性递归）
# --------------------------------------------------------------------------
# 递归 h_t = a_t * h_{t-1} + b_t 无法用逐元素算子展开成固定图（T 会变成节点数），
# 也没有闭式解可用：连乘 a 会在 T 长时下溢。所以它必须是一个语义算子。
# 注意这里只定义**最朴素的线性递归**，Mamba 的离散化（exp(dt*A)、dt*B*x）
# 全部由已有算子在前端组合出来——算子越简单，反向与跨后端实现越可控。


def _infer_scan_linear(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    a, b, h0 = inputs
    if a.dtype != b.dtype or a.dtype != h0.dtype:
        raise TypeError(
            f"scan_linear 要求三个输入同 dtype，得到 "
            f"{a.dtype.value} / {b.dtype.value} / {h0.dtype.value}"
        )
    if a.shape != b.shape:
        raise ValueError(f"scan_linear 的 a 与 b 形状必须一致，得到 {a.shape} 与 {b.shape}")
    if a.rank < 2:
        raise ValueError(f"scan_linear 至少需要 (T, D) 两个维度，得到 {a.shape}")
    if h0.shape != a.shape[1:]:
        raise ValueError(
            f"初始状态形状应为 {a.shape[1:]}（去掉时间维），得到 {h0.shape}"
        )
    if not isinstance(a.shape[0], int):
        raise ValueError(
            f"scan_linear 的时间维必须是静态整数，得到符号维 {a.shape[0]!r}；"
            "递归的边界与反向的切片位置都依赖它"
        )
    return a


def _ref_scan_linear(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    a, b, h0 = args
    steps = a.shape[0]
    out = np.empty_like(a)
    h = np.array(h0, copy=True)
    for t in range(steps):
        h = a[t] * h + b[t]
        out[t] = h
    return out


register(
    Op(
        name="scan_linear",
        infer=_infer_scan_linear,
        reference=_ref_scan_linear,
        attrs_schema=(),
        doc="线性递归 h_t = a_t * h_{t-1} + b_t（h_{-1} = h0）；SSM 的 selective scan 核心",
    )
)


def _infer_scan_linear_backward(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    a, dy = inputs
    if a.dtype != dy.dtype:
        raise TypeError(
            f"scan_linear_backward 要求 a 与 dy 同 dtype，得到 "
            f"{a.dtype.value} 与 {dy.dtype.value}"
        )
    if a.shape != dy.shape:
        raise ValueError(
            f"scan_linear_backward 的 a 与 dy 形状必须一致，得到 {a.shape} 与 {dy.shape}"
        )
    if a.rank < 2:
        raise ValueError(f"scan_linear_backward 至少需要 (T, D) 两个维度，得到 {a.shape}")
    if not isinstance(a.shape[0], int):
        raise ValueError(
            f"scan_linear_backward 的时间维必须是静态整数，得到符号维 {a.shape[0]!r}"
        )
    return dy


def _ref_scan_linear_backward(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """反向递归 dh_t = dy_t + a_{t+1} * dh_{t+1}，边界 dh_T = 0。

    这是线性递归的伴随：前向从 0 走到 T-1，反向从 T-1 走回 0。它同样不能
    用逐元素算子展开，所以与前向并列成为一个语义算子——不是"某个后端的内核"。
    """
    a, dy = args
    steps = a.shape[0]
    out = np.empty_like(dy)
    dh = np.zeros_like(dy[0])
    for t in range(steps - 1, -1, -1):
        nxt = a[t + 1] * dh if t + 1 < steps else 0.0
        dh = dy[t] + nxt
        out[t] = dh
    return out


register(
    Op(
        name="scan_linear_backward",
        infer=_infer_scan_linear_backward,
        reference=_ref_scan_linear_backward,
        attrs_schema=(),
        doc="线性递归的伴随：dh_t = dy_t + a_{t+1} * dh_{t+1}；scan_linear 反向图的组成部分",
    )
)


# --------------------------------------------------------------------------
# 卷积 / 池化的反向
# --------------------------------------------------------------------------


def _conv_geometry(attrs: dict):
    stride = _pair(attrs.get("stride", (1, 1)), "stride")
    padding = _pair(attrs.get("padding", (0, 0)), "padding")
    dilation = _pair(attrs.get("dilation", (1, 1)), "dilation")
    return stride, padding, dilation


def _infer_conv2d_input_grad(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    dy, weight = inputs
    if dy.rank != 4 or weight.rank != 4:
        raise ValueError(f"conv2d_input_grad 要求 dy/weight 都是秩 4，得到 {dy.rank} 与 {weight.rank}")
    if dy.dtype != weight.dtype:
        raise TypeError(
            f"conv2d_input_grad 要求 dy 与 weight dtype 一致，得到 "
            f"{dy.dtype.value} 与 {weight.dtype.value}"
        )
    shape = _as_shape(attrs["input_shape"])
    if len(shape) != 4:
        raise ValueError(f"input_shape 必须是 4 维，得到 {shape}")
    groups = int(attrs.get("groups", 1))
    if groups < 1:
        raise ValueError(f"conv2d_input_grad 的 groups 必须 >= 1，得到 {groups}")
    c = shape[1]
    cw = weight.shape[1]
    if isinstance(c, int) and isinstance(cw, int) and cw * groups != c:
        raise ValueError(
            f"conv2d_input_grad 的权重通道 {cw} * groups={groups} 与 input_shape 通道 {c} 不一致"
        )
    return TensorType(dy.dtype, shape)


def _conv2d_input_grad_single(dy, weight, n, c, h, w, sh, sw, ph, pw, dh, dw):
    """单组卷积对输入的梯度；``dy``/``weight`` 都已按组切好。"""
    _, _, kh, kw = weight.shape
    _, _, ho, wo = dy.shape

    padded = np.zeros((n, c, h + 2 * ph, w + 2 * pw), dtype=dy.dtype)
    for i in range(kh):
        for j in range(kw):
            contrib = np.einsum("nohw,oc->nchw", dy, weight[:, :, i, j])
            padded[
                :,
                :,
                i * dh : i * dh + sh * (ho - 1) + 1 : sh,
                j * dw : j * dw + sw * (wo - 1) + 1 : sw,
            ] += contrib
    return padded[:, :, ph : ph + h, pw : pw + w]


def _ref_conv2d_input_grad(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """卷积 w.r.t. 输入的梯度。

    ``dx[n,c,ih,iw] = sum_{o,ki,kj} dy[n,o,ho,wo] * w[o,c,ki,kj]``，其中
    ``ih = ho*sh - ph + ki*dh``。这就是"转置卷积"的那条散射关系，但这里
    不引入转置卷积的概念，只定义输入梯度——IR 里不该出现两套卷积语义。
    分组时每组独立做一次，再沿输入通道拼回。
    """
    dy, weight = args[0], args[1]
    (sh, sw), (ph, pw), (dh, dw) = _conv_geometry(attrs)
    n, c, h, w = (int(v) for v in _as_shape(attrs["input_shape"]))
    groups = int(attrs.get("groups", 1))
    cw = weight.shape[1]
    if cw * groups != c:
        raise ValueError(
            f"conv2d_input_grad 的权重通道 {cw} * groups={groups} 与 "
            f"input_shape 通道 {c} 不一致"
        )
    if groups == 1:
        return _conv2d_input_grad_single(dy, weight, n, c, h, w, sh, sw, ph, pw, dh, dw)

    og = weight.shape[0] // groups
    parts = [
        _conv2d_input_grad_single(
            dy[:, gi * og : (gi + 1) * og],
            weight[gi * og : (gi + 1) * og],
            n, cw, h, w, sh, sw, ph, pw, dh, dw,
        )
        for gi in range(groups)
    ]
    return np.concatenate(parts, axis=1)


register(
    Op(
        name="conv2d_input_grad",
        infer=_infer_conv2d_input_grad,
        reference=_ref_conv2d_input_grad,
        attrs_schema=("input_shape", "stride", "padding", "dilation", "groups"),
        doc="conv2d 对 x 的梯度：dy(N,O,HO,WO) 与 w(O,C,kh,kw) -> dx(N,C,H,W)，"
        "几何参数与 conv2d 同名同义",
    )
)


def _infer_conv2d_weight_grad(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    x, dy = inputs
    if x.rank != 4 or dy.rank != 4:
        raise ValueError(
            f"conv2d_weight_grad 要求 x/dy 都是秩 4，得到 {x.rank} 与 {dy.rank}"
        )
    shape = _as_shape(attrs["weight_shape"])
    if len(shape) != 4:
        raise ValueError(f"weight_shape 必须是 4 维，得到 {shape}")
    groups = int(attrs.get("groups", 1))
    if groups < 1:
        raise ValueError(f"conv2d_weight_grad 的 groups 必须 >= 1，得到 {groups}")
    c = x.shape[1]
    cw = shape[1]
    if isinstance(c, int) and isinstance(cw, int) and cw * groups != c:
        raise ValueError(
            f"conv2d_weight_grad 的 x 通道 {c} 与 weight_shape 通道 {cw} * groups={groups} 不一致"
        )
    return TensorType(x.dtype, shape)


def _conv2d_weight_grad_single(x, dy, o, kh, kw, sh, sw, ph, pw, dh, dw):
    """单组卷积对权重的梯度；``x``/``dy`` 都已按组切好。"""
    n, c, h, w = x.shape
    _, _, ho, wo = dy.shape

    padded = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw))) if (ph or pw) else x
    dw_out = np.zeros((o, c, kh, kw), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            patch = padded[
                :,
                :,
                i * dh : i * dh + sh * (ho - 1) + 1 : sh,
                j * dw : j * dw + sw * (wo - 1) + 1 : sw,
            ]
            dw_out[:, :, i, j] = np.einsum("nohw,nchw->oc", dy, patch)
    return dw_out


def _ref_conv2d_weight_grad(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """卷积 w.r.t. 权重的梯度：``dw[o,c,ki,kj] = sum_{n,ho,wo} dy*x``。"""
    x, dy = args[0], args[1]
    (sh, sw), (ph, pw), (dh, dw) = _conv_geometry(attrs)
    n, c, h, w = x.shape
    o, cw, kh, kw = _as_shape(attrs["weight_shape"])
    groups = int(attrs.get("groups", 1))
    if cw * groups != c:
        raise ValueError(
            f"conv2d_weight_grad 的 x 通道 {c} 与 weight_shape 通道 {cw} * groups={groups} 不一致"
        )
    _, _, ho, wo = dy.shape

    if groups == 1:
        return _conv2d_weight_grad_single(x, dy, o, kh, kw, sh, sw, ph, pw, dh, dw)

    og = o // groups
    parts = [
        _conv2d_weight_grad_single(
            x[:, gi * cw : (gi + 1) * cw],
            dy[:, gi * og : (gi + 1) * og],
            og, kh, kw, sh, sw, ph, pw, dh, dw,
        )
        for gi in range(groups)
    ]
    return np.concatenate(parts, axis=0)


register(
    Op(
        name="conv2d_weight_grad",
        infer=_infer_conv2d_weight_grad,
        reference=_ref_conv2d_weight_grad,
        attrs_schema=("weight_shape", "stride", "padding", "dilation", "groups"),
        doc="conv2d 对 weight 的梯度：x(N,C,H,W) 与 dy(N,O,HO,WO) -> dw(O,C,kh,kw)",
    )
)


def _infer_maxpool2d_backward(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    return inputs[0]


def _ref_maxpool2d_backward(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """最大池化的输入梯度：把 dy 散射到每个窗口的最大值位置。

    并列最大值取**第一个**（行优先），因此结果确定、可对拍。
    padding 区域不参与竞争，与 ``maxpool2d`` 的前向语义严格对应。
    """
    x, dy = args[0], args[1]
    kh, kw = _pair(attrs["kernel"], "kernel")
    raw_stride = attrs.get("stride")
    sh, sw = _pair(attrs["kernel"] if raw_stride is None else raw_stride, "stride")
    ph, pw = _pair(attrs.get("padding", (0, 0)), "padding")

    n, c, h, w = x.shape
    _, _, ho, wo = dy.shape
    dx = np.zeros_like(x)
    for ni in range(n):
        for ci in range(c):
            for hi in range(ho):
                for wi in range(wo):
                    best = -np.inf
                    best_ij = None
                    for i in range(kh):
                        ih = hi * sh - ph + i
                        if ih < 0 or ih >= h:
                            continue
                        for j in range(kw):
                            iw = wi * sw - pw + j
                            if iw < 0 or iw >= w:
                                continue
                            v = x[ni, ci, ih, iw]
                            if best_ij is None or v > best:
                                best = v
                                best_ij = (ih, iw)
                    if best_ij is not None:
                        dx[ni, ci, best_ij[0], best_ij[1]] += dy[ni, ci, hi, wi]
    return dx


register(
    Op(
        name="maxpool2d_backward",
        infer=_infer_maxpool2d_backward,
        reference=_ref_maxpool2d_backward,
        attrs_schema=("kernel", "stride", "padding", "ceil_mode"),
        doc="最大池化对输入的梯度：x(N,C,H,W) 与 dy(N,C,HO,WO) -> dx(N,C,H,W)；"
        "并列最大值取第一个；ceil_mode 只改变前向的输出长度，梯度沿同一组窗口散射",
    )
)


def _infer_maxpool2d_indices(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    x = inputs[0]
    if x.rank != 4:
        raise ValueError(f"maxpool2d_indices 要求 NCHW 秩 4 输入，得到秩 {x.rank}")
    (kh, kw), (sh, sw), (ph, pw) = _pool_geometry(attrs)
    ceil_mode, _ = _pool_flags(attrs)
    n, c, h, w = x.shape
    return TensorType(
        DType.I32,
        (
            n,
            c,
            _pool_out_dim(h, kh, sh, ph, ceil_mode),
            _pool_out_dim(w, kw, sw, pw, ceil_mode),
        ),
    )


def _ref_maxpool2d_indices(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """最大池化的取值位置：每个窗口里最大元素的 ``ih * W + iw``（平面内展平下标）。

    这与 PyTorch ``max_pool2d(..., return_indices=True)`` 的第二个返回值同一个
    定义，也是 ``maxpool2d_backward`` 内部使用的定位规则——并列取第一个，因此
    三个算子（前向、位置、反向）在并列时选中同一个元素。
    """
    x = args[0]
    (kh, kw), (sh, sw), (ph, pw) = _pool_geometry(attrs)
    ceil_mode, _ = _pool_flags(attrs)
    n, c, h, w = x.shape
    ho = _pool_out_dim(h, kh, sh, ph, ceil_mode)
    wo = _pool_out_dim(w, kw, sw, pw, ceil_mode)
    out = np.full((n, c, ho, wo), -1, dtype=np.int32)
    for ni in range(n):
        for ci in range(c):
            for hi in range(ho):
                for wi in range(wo):
                    best = -np.inf
                    bi = bj = -1
                    for i in range(kh):
                        ih = hi * sh - ph + i
                        if ih < 0 or ih >= h:
                            continue
                        for j in range(kw):
                            iw = wi * sw - pw + j
                            if iw < 0 or iw >= w:
                                continue
                            v = x[ni, ci, ih, iw]
                            if bi < 0 or v > best:
                                best = v
                                bi, bj = ih, iw
                    if bi >= 0:
                        out[ni, ci, hi, wi] = bi * w + bj
    return out


register(
    Op(
        name="maxpool2d_indices",
        infer=_infer_maxpool2d_indices,
        reference=_ref_maxpool2d_indices,
        attrs_schema=("kernel", "stride", "padding", "ceil_mode"),
        doc="最大池化的取值位置（i32，平面内展平下标 ih*W+iw）："
        "对应 PyTorch max_pool2d(return_indices=True) 的第二个输出；"
        "并列最大值取第一个，与 maxpool2d_backward 的散射位置一致",
    )
)


# --------------------------------------------------------------------------
# 随机
# --------------------------------------------------------------------------

# SplitMix64 的三个常数。选它不是为了"够随机"，而是因为它只由整数乘法与
# 异或构成：numpy 参考实现与 CUDA 内核可以写出**逐位相同**的序列，
# 于是随机算子也能像确定性算子一样跨后端对拍。
# 用 numpy 内置的 Generator 会绑死 PCG64 的位级实现，而 CUDA 侧没有对应物。
_SM64_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_SM64_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_SM64_MUL2 = np.uint64(0x94D049BB133111EB)
_F32_2P24 = np.float32(16777216.0)  # 2^24，掩码只用 24 位尾数


def _splitmix64(z: np.ndarray) -> np.ndarray:
    """逐元素的 SplitMix64 混合函数（uint64 数组）。"""
    z = (z ^ (z >> np.uint64(30))) * _SM64_MUL1
    z = (z ^ (z >> np.uint64(27))) * _SM64_MUL2
    return z ^ (z >> np.uint64(31))


def _infer_dropout_mask(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) not in (1, 2):
        raise ValueError(
            "dropout_mask 需要 1 或 2 个输入（x 只取形状与 dtype；"
            f"可选的第二个是运行时 key），得到 {len(inputs)}"
        )
    p = float(attrs.get("p", 0.5))
    if not 0.0 <= p < 1.0:
        # p = 1 时缩放因子 1/(1-p) 无定义；与其产生 inf，不如在构图期拒绝。
        raise ValueError(f"dropout 的 p 必须在 [0, 1) 内，得到 {p}")
    if len(inputs) == 2:
        key = inputs[1]
        if key.dtype is not DType.I32:
            raise ValueError(
                f"dropout_mask 的 key 必须是 i32 输入，得到 {key.dtype}"
            )
        if tuple(key.shape) not in ((), (1,)):
            raise ValueError(
                f"dropout_mask 的 key 必须是标量或长度 1 的向量，得到形状 {key.shape}"
            )
    return inputs[0]


def _dropout_seed(args: List[np.ndarray], attrs: dict) -> np.uint64:
    """取本次调用的 seed：运行时 key 输入优先，其次是 attrs 里的常量。

    ``offset`` 让同一张图里的多个 dropout 站点共用**一个** key 输入，却
    拿到互不相同的掩码；否则同形状的两层会得到逐位相同的掩码。
    """
    if len(args) > 1:
        seed = int(np.asarray(args[1]).reshape(-1)[0])
    else:
        seed = int(attrs.get("seed", 0))
    offset = int(attrs.get("offset", 0))
    return np.uint64((seed + offset) & 0xFFFFFFFFFFFFFFFF)


def _ref_dropout_mask(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """按扁平索引生成缩放后的掩码：保留 → 1/(1-p)，丢弃 → 0。

    输入 x **只用于确定形状与 dtype，数值完全不参与**——因此这个算子对 x
    没有梯度（见 ``grad_rules``），梯度全部经由 ``mul(x, mask)`` 那条边回传。
    """
    x = args[0]
    p = float(attrs.get("p", 0.5))
    seed = _dropout_seed(args, attrs)

    n = int(x.size)
    idx = np.arange(n, dtype=np.uint64)
    z = _splitmix64(seed + idx * _SM64_GOLDEN)
    # 取高 24 位：恰好是 float32 的尾数宽度，numpy 与 CUDA 得到同一个 float
    u = ((z >> np.uint64(40)) & np.uint64(0xFFFFFF)).astype(np.float32) / _F32_2P24
    keep = (u >= np.float32(p)).astype(np.float32)
    scale = np.float32(1.0 / (1.0 - p))
    return (keep * scale).reshape(x.shape).astype(x.dtype)


register(
    Op(
        name="dropout_mask",
        infer=_infer_dropout_mask,
        reference=_ref_dropout_mask,
        attrs_schema=("p", "seed", "offset"),
        doc="dropout 掩码：按 (seed + offset, 扁平索引) 用 SplitMix64 生成，"
        "保留的元素取 1/(1-p)、丢弃的元素取 0。输入 x 只提供形状与 dtype，"
        "数值不参与计算，也不接收梯度。seed 可以写进 attrs（图自包含、"
        "可复现），也可以由第二个 i32 输入在运行期给出——训练循环靠它"
        "逐步推进随机状态，同一 optimizer step 的前向与重算用同一个值；"
        "offset 让同图多个站点共用一个 key 却拿到不同掩码。"
        "参考后端与 CUDA 用同一套位运算，因此掩码可以逐位对拍",
        # 显式 seed/key 让结果确定，但语义是"每次重新采样"：不能折叠成常量，
        # 也不能被 CSE 合并成一次采样。重算段里可以重放（同一 key 同一结果）。
        effects=frozenset({Effect.RNG}),
    )
)


# --------------------------------------------------------------------------
# 傅里叶变换
# --------------------------------------------------------------------------

#: numpy 与 cuFFT 共有的三档归一化。``backward`` 只在正向除以 N（numpy 默认），
#: ``ortho`` 两侧各除 √N，``forward`` 只在反向除以 N。
_FFT_NORMS = ("backward", "ortho", "forward")


def _fft_norm(attrs: dict) -> str:
    norm = attrs.get("norm", "backward")
    if norm not in _FFT_NORMS:
        raise ValueError(f"fft 的 norm 只能是 {_FFT_NORMS} 之一，得到 {norm!r}")
    return norm


def _infer_fft(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    """复数到复数的 DFT（沿最后一维）。形状与 dtype 都不变。

    只接受复数输入：实数进复数出是 ``rfft`` 的语义，混在同一个算子里会让
    输出形状依赖 dtype，类型推导就得多一条分支；更重要的是**默不作声地
    把实数升成复数**属于隐式降级，宁可显式要求 cast。
    """
    x = inputs[0]
    if not x.dtype.is_complex:
        raise TypeError(
            f"fft 的输入必须是复数 dtype（c64/c128），得到 {x.dtype.value}；"
            "实数序列请用 rfft，或先用 cast 显式转成 c64"
        )
    if x.rank == 0:
        raise ValueError("fft 至少需要 1 维（变换沿最后一维）")
    _fft_norm(attrs)
    inverse = attrs.get("inverse", False)
    if not isinstance(inverse, bool):
        raise TypeError(f"fft 的 inverse 必须是 bool，得到 {type(inverse).__name__}")
    return x


def _ref_fft(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    norm = _fft_norm(attrs)
    fn = np.fft.ifft if attrs.get("inverse", False) else np.fft.fft
    return fn(args[0], axis=-1, norm=norm)


register(
    Op(
        name="fft",
        infer=_infer_fft,
        reference=_ref_fft,
        attrs_schema=("norm", "inverse"),
        doc="沿最后一维的离散傅里叶变换（复数 → 复数）。``inverse=True`` 走逆变换；"
        "``norm`` 取 backward / ortho / forward，与 numpy.fft 同义。"
        "复数 dtype 只有 c64 / c128 两种，形状与 dtype 都保持不变。"
        "反向图暂不提供 VJP（复数线性算子的伴随需要共轭转置，"
        "IR 里还没有 conj 算子），求导会显式报 NoGradientError。",
    )
)


def _infer_rfft(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    """实数到复数的半谱 DFT：``f32 -> c64``、``f64 -> c128``。

    f16 / bf16 显式拒绝：半精度 FFT 的舍入误差会盖过数值本身，且
    numpy 没有对应的参考实现——没有参考语义就不该进入 IR。
    """
    x = inputs[0]
    if x.dtype == DType.F32:
        out = DType.C64
    elif x.dtype == DType.F64:
        out = DType.C128
    else:
        raise TypeError(
            f"rfft 只接受 f32 / f64 实数输入（分别得到 c64 / c128），"
            f"得到 {x.dtype.value}；f16 请先 cast 到 f32"
        )
    if x.rank == 0:
        raise ValueError("rfft 至少需要 1 维（变换沿最后一维）")
    _fft_norm(attrs)
    last = x.shape[-1]
    if isinstance(last, int):
        last = last // 2 + 1
    else:
        last = sym_binary("add", sym_binary("floordiv", last, 2), 1)
    return TensorType(out, x.shape[:-1] + (last,))


def _ref_rfft(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    return np.fft.rfft(args[0], axis=-1, norm=_fft_norm(attrs))


register(
    Op(
        name="rfft",
        infer=_infer_rfft,
        reference=_ref_rfft,
        attrs_schema=("norm",),
        doc="沿最后一维的实数半谱 DFT：形状 ``(..., n) -> (..., n//2+1)``，"
        "dtype 由 f32/f64 变成 c64/c128。只保留非负频率那一半，"
        "因为实序列的谱是共轭对称的。反向同样不提供 VJP。",
    )
)
