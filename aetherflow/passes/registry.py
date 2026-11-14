"""Pass 注册表与按名执行入口。

``default_pipeline`` / ``fused_pipeline`` 过去各自写死一串具体类：想只跑一个
pass、或按配置挑一串，只能 import 具体类自己拼。注册表把"名字 → 工厂"固定
成唯一来源，并给出三条显式约定：

1. **名字唯一**：重名默认报错；只有显式 ``replace=True`` 才覆盖。
2. **工厂每次给新实例**：pass 可以带状态，单例共享会让两次执行互相污染。
3. **未知名字立刻报错**并列出可用项，不做"跳过未知 pass"的静默降级。

带参数的 pass（``Autocast`` 的 dtype / 算子集）**不进注册表**：名字无法表达
参数，用一个默认值偷偷构造出来只会让调用方以为配置生效了。这类 pass 直接
实例化，再和名字一起交给 :func:`build_pipeline`（管线元素允许是 ``Pass``
实例）。

三条管线：默认、融合、反向图（``backward_pipeline``）。反向图那条不含任何
融合 pass——融合会改变算子输入形态、要求目标后端承接，而反向图的设计前提是
"任何承接前向的后端都能直接跑它"。

每条执行入口都走 :class:`~aetherflow.passes.base.PassManager`，因此每个
pass 之后都会过一次 verifier——单独跑一个 pass 也不能绕开这条约定。
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Union

from ..ir.graph import Graph
from .base import Pass, PassManager
from .cse import CommonSubexpressionElimination, CommutativeCSE
from .dce import DeadCodeElimination
from .fold import ConstantFolding
from .fusion import FuseLayerNormResidual, FuseSiLU, FuseSoftmaxScale
from .simplify import SimplifyShapeOps

__all__ = [
    "PassFactory",
    "register_pass",
    "unregister_pass",
    "available_passes",
    "get_pass",
    "build_pipeline",
    "run_pass",
    "run_passes",
    "DEFAULT_PASS_NAMES",
    "FUSED_PASS_NAMES",
    "BACKWARD_PASS_NAMES",
    "default_pipeline",
    "fused_pipeline",
    "backward_pipeline",
]

PassFactory = Callable[[], Pass]
PassSpec = Union[str, Pass]

_REGISTRY: Dict[str, PassFactory] = {}


def register_pass(
    name: str, factory: PassFactory = None, *, replace: bool = False
) -> Union[PassFactory, Callable[[PassFactory], PassFactory]]:
    """注册一个 pass 工厂。

    可作装饰器使用：``@register_pass("my_pass")``；也可以直接
    ``register_pass("my_pass", MyPass)``。工厂必须是无参可调用的（类本身
    就可以），返回一个全新的 :class:`Pass` 实例。

    注册名必须与 pass 自己的 ``name`` 属性一致：名字是唯一标识，若两者
    可以不同，日志里打印的名字和按名查到的 pass 就会对不上。这条在注册期
    用一个丢弃的实例校验，拼错当场失败。
    """

    def install(candidate: PassFactory) -> PassFactory:
        if not isinstance(name, str) or not name:
            raise ValueError(f"pass 名字必须是非空字符串，得到 {name!r}")
        if not callable(candidate):
            raise TypeError(f"pass 工厂必须可调用，得到 {candidate!r}")
        if name in _REGISTRY and not replace:
            raise ValueError(
                f"pass 名字 {name!r} 已注册（{_REGISTRY[name]!r}）；"
                "要覆盖请显式传 replace=True"
            )
        probe = candidate()
        if not isinstance(probe, Pass):
            raise TypeError(
                f"pass 工厂 {name!r} 返回了 {type(probe).__name__}，不是 Pass 子类"
            )
        if probe.name != name:
            raise ValueError(
                f"注册名 {name!r} 与 pass 自身的 name {probe.name!r} 不一致；"
                "注册名必须等于 pass.name"
            )
        _REGISTRY[name] = candidate
        return candidate

    if factory is None:
        return install
    return install(factory)


def unregister_pass(name: str) -> None:
    """注销一个 pass；名字不存在时显式报错（拼错不该被当成成功）。"""
    if name not in _REGISTRY:
        raise KeyError(f"没有注册名为 {name!r} 的 pass；可用 {available_passes()}")
    del _REGISTRY[name]


def available_passes() -> List[str]:
    """已注册的 pass 名字，按字典序排列（便于打印与断言）。"""
    return sorted(_REGISTRY)


def get_pass(name: str) -> Pass:
    """按名字构造一个**新**的 pass 实例。"""
    if not isinstance(name, str):
        raise TypeError(f"pass 名字必须是字符串，得到 {name!r}")
    factory = _REGISTRY.get(name)
    if factory is None:
        raise ValueError(
            f"没有注册名为 {name!r} 的 pass；可用 {available_passes()}"
        )
    instance = factory()
    if not isinstance(instance, Pass):
        raise TypeError(
            f"pass 工厂 {name!r} 返回了 {type(instance).__name__}，不是 Pass 子类"
        )
    return instance


def _resolve(spec: PassSpec) -> Pass:
    if isinstance(spec, Pass):
        return spec
    if isinstance(spec, str):
        return get_pass(spec)
    raise TypeError(f"管线元素必须是 pass 名字或 Pass 实例，得到 {spec!r}")


def build_pipeline(specs: Iterable[PassSpec]) -> PassManager:
    """按名字 / 实例混合构造管线；空序列是恒等管线。"""
    if isinstance(specs, (str, Pass)):
        raise TypeError(
            f"build_pipeline 需要名字或实例的序列，得到单个 {specs!r}；"
            "只有一个元素时请写成列表"
        )
    return PassManager([_resolve(spec) for spec in specs])


def run_pass(graph: Graph, spec: PassSpec, *, check: bool = True) -> Graph:
    """单独执行一个 pass，并（默认）过 verifier。"""
    return PassManager([_resolve(spec)], check=check).run(graph)


def run_passes(
    graph: Graph, specs: Iterable[PassSpec], *, check: bool = True
) -> Graph:
    """按顺序执行一串 pass，等价于 ``build_pipeline(specs).run(graph)``。"""
    return PassManager([_resolve(spec) for spec in specs], check=check).run(graph)


# 内置注册表：只有无参可构造的 pass。顺序在这里不重要，管线顺序由两个
# ``*_PASS_NAMES`` 明确写出。
register_pass("simplify-shape-ops", SimplifyShapeOps)
register_pass("constant-folding", ConstantFolding)
register_pass("cse", CommonSubexpressionElimination)
register_pass("cse-commutative", CommutativeCSE)
register_pass("dce", DeadCodeElimination)
register_pass("fuse_softmax_scale", FuseSoftmaxScale)
register_pass("fuse_layernorm_residual", FuseLayerNormResidual)
register_pass("fuse_silu_gate", FuseSiLU)


# 无硬件假设的默认管线。（``ConstantFolding`` 在 ``SimplifyShapeOps`` 之后，
# 因为化简会制造新的常量输入。）
DEFAULT_PASS_NAMES = ("simplify-shape-ops", "constant-folding", "cse", "dce")

# 融合会改变算子的输入形态，要求目标后端承接，因此单独成一条管线。
FUSED_PASS_NAMES = (
    "constant-folding",
    "simplify-shape-ops",
    "cse",
    "fuse_softmax_scale",
    "fuse_layernorm_residual",
    "fuse_silu_gate",
    "dce",
)


def default_pipeline() -> PassManager:
    """无硬件假设的默认优化管线。"""
    return build_pipeline(DEFAULT_PASS_NAMES)


def fused_pipeline() -> PassManager:
    """带融合的优化管线（要求后端承接融合后的输入形态）。"""
    return build_pipeline(FUSED_PASS_NAMES)


# 反向图专用管线。它与默认管线的差别只有一处：CSE 换成交换律规范化版本。
# 反向图里 cotangent 的累加顺序由逆拓扑遍历决定，同一个值的多份贡献会以
# ``add(a, b)`` / ``add(b, a)`` 两种形态出现，普通 CSE 认不出它们同源。
# 不跑融合：反向图要能在任何承接前向的后端上执行，融合会引入新的输入形态，
# 那要求后端额外承诺。
BACKWARD_PASS_NAMES = (
    "simplify-shape-ops",
    "constant-folding",
    "cse-commutative",
    "dce",
)


def backward_pipeline() -> PassManager:
    """反向图专用优化管线（无融合、无后端额外假设）。"""
    return build_pipeline(BACKWARD_PASS_NAMES)
