"""插件扩展面：把第三方算子接进 IR。

``ir.ops.register`` 已经能往算子表里加东西，但直接把这个动作暴露给插件，会缺三条
护栏——``OPS`` 是**进程级全局状态**，没有进程边界可依靠，一次错误的注册会污染
之后的所有构图：

1. **命名空间**。插件算子必须以 ``<插件名>.`` 开头。内置算子表会继续增长，没有
   前缀的名字早晚撞上；而撞上的时刻通常是"另一个插件先被 import 了"，
   这类失败极难复现。
2. **内置不可覆盖**。重名本身已由 ``ir.ops.register`` 拒绝，这里再明确一次：
   内置算子不是插件能改的东西；``unregister_operator`` 同样拒绝卸载内置算子。
3. **可卸载**。测试与热插拔需要把表恢复原状，只对自己注册的名字生效。

注册之后，**不用再教参考后端认识这个算子**：``ReferenceBackend`` 的语义按
``OPS`` 表动态取，所以任何 ``PORTABLE`` 插件算子立刻能被参考执行器跑、被
conformance 对拍。硬件后端仍需各自登记内核（``KernelRegistry.register``），
这一点与内置算子没有区别。

**一个必须说清的限制**：含插件算子的图只能被"同样加载了该插件"的进程读回。
算子表是全局状态，这是它的自然结果，不是缺陷——图的可移植性以两边算子表一致
为前提。因此把插件算子写进长期存档前，应当先确认加载路径里包含该插件。
"""

from __future__ import annotations

from typing import FrozenSet, Tuple

from .ir.ops import OPS, Op, register

__all__ = ["BUILTIN_OPS", "register_operator", "unregister_operator", "plugin_ops"]

# 本模块导入时算子表里的全部名字，此后新增的都算插件算子。
# 依赖导入顺序：``aetherflow/__init__.py`` 先加载 ``ir``，再加载本模块。
BUILTIN_OPS: FrozenSet[str] = frozenset(OPS)


def register_operator(op: Op, *, namespace: str) -> Op:
    """把一个插件算子登记进 IR 算子表。

    ``namespace`` 是插件自己的名字，算子名必须以 ``namespace + "."`` 开头；
    两者分开传，而不是从前缀里反推，是为了让"这个算子属于谁"在调用点就写明，
    而不是靠字符串切分猜。
    """
    if not isinstance(op, Op):
        raise TypeError(f"register_operator 需要 Op 实例，得到 {type(op).__name__}")
    if not namespace:
        raise ValueError("插件必须给出非空的命名空间")
    if op.name in BUILTIN_OPS:
        raise ValueError(f"算子 {op.name!r} 是内置算子，插件不能注册同名算子")
    prefix = f"{namespace}."
    if not op.name.startswith(prefix):
        raise ValueError(
            f"插件算子名 {op.name!r} 必须以 {prefix!r} 开头；"
            "没有前缀的名字会与内置算子或别的插件相撞"
        )
    return register(op)


def unregister_operator(name: str) -> Op:
    """移除一个插件算子，返回它。

    只对插件算子生效。卸载后，含该算子的图仍然存在，但再也读不回来——
    调用方应当先确保没有存档还引用它。
    """
    if name in BUILTIN_OPS:
        raise ValueError(f"算子 {name!r} 是内置算子，插件不能卸载")
    op = OPS.pop(name, None)
    if op is None:
        raise KeyError(f"算子 {name!r} 不在算子表里，无法卸载")
    return op


def plugin_ops() -> Tuple[str, ...]:
    """当前已注册的插件算子名（按字典序），用于自省与清理。"""
    return tuple(sorted(set(OPS) - BUILTIN_OPS))
