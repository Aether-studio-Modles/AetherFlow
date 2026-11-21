"""后端装配：把"用哪块硬件"从代码里挪到配置里。

在此之前，换后端要同时改 import 与类名：

    from aetherflow.backends.hip import HipBackend
    backend = HipBackend()

同一份前端代码想跑在 CUDA / HIP / 参考实现上，只能自己写 if/else。装配层只做
两件事：**记住有哪些后端**（内置三个 + pip 包通过 entry point 注册），
**按名字或自动探测挑一个**。它不认识任何厂商细节——挑选之后拿到的仍然是
``Backend`` 协议，与直接构造出来的对象没有区别。

三种挑法，优先级从高到低：

1. 显式传名：``get_backend("hip")``；
2. 环境变量：``AETHERFLOW_BACKEND=hip``；
3. 自动探测：``get_backend()`` 依次试 CUDA、HIP，第一个真能初始化的胜出。

"真能初始化"是实际构造一次，而不是看某个布尔标志：``CUDA_AVAILABLE`` 只表示
cupy 装没装，没装驱动时构造仍会失败。探测只发生一次失败尝试，代价远小于
"以为在用 GPU、其实在跑 CPU"这种静默降级。

自动探测在两边都失败时会退回参考后端（纯 CPU 语义执行），并**发一条
RuntimeWarning**——退回是允许的，但不能悄悄发生。需要硬性保证时传
``fallback=False``，让它直接抛错。

第三方硬件包用 pip entry point 接入，不需要改这里：::

    [project.entry-points."aetherflow.backends"]
    mychip = "mychip.backend:MyChipBackend"

装包后 ``get_backend("mychip")`` 即可用。内置名字不会被第三方覆盖：同名时
第三方条目被忽略并记录原因，避免"装了个包，CUDA 后端被顶掉"。
"""

from __future__ import annotations

import importlib.metadata
import os
import warnings
from typing import Callable, Dict, Optional, Tuple

__all__ = [
    "BackendUnavailableError",
    "register_backend",
    "unregister_backend",
    "registered_backends",
    "get_backend",
    "BUILTIN_BACKENDS",
]

_ENV_VAR = "AETHERFLOW_BACKEND"
_ENTRY_GROUP = "aetherflow.backends"
# 自动探测顺序：先 CUDA，再 HIP；两者都没有才退回参考后端。
_AUTO_ORDER = ("cuda", "hip")
# 内置三个不允许被注销：它们是退回路径的兜底，删掉会让 fallback 无处可去。
# 硬件包也不能占用这些名字（见 hardware.install_hardware）。
BUILTIN_BACKENDS = frozenset({"reference", "cuda", "hip"})


class BackendUnavailableError(RuntimeError):
    """名字存在，但这台机器上装不起来（没有驱动 / 库缺失 / 初始化失败）。

    与 ``BackendCapabilityError`` 的区别：后者是"能跑，但这个图算不了"，
    属于执行期能力边界；这里连设备都没拿到，是装配期失败。
    """


_FACTORIES: Dict[str, Callable[..., object]] = {}
_ENTRY_ERRORS: Dict[str, str] = {}
_discovered = False


# --------------------------------------------------------------------------
# 内置三个：工厂都是惰性的
# --------------------------------------------------------------------------
# 不在模块导入时就把插件拉进来：`import aetherflow` 不该因为没装 cupy 变慢或失败，
# 也不该无条件加载 100 KB 的 CUDA 插件。只有真被选中时才 import。

def _reference(**kwargs):
    from .reference_backend import ReferenceBackend

    return ReferenceBackend(**kwargs)


def _cuda(**kwargs):
    from .cuda import CudaBackend

    return CudaBackend(**kwargs)


def _hip(**kwargs):
    from .hip import HipBackend

    return HipBackend(**kwargs)


_FACTORIES.update(reference=_reference, cuda=_cuda, hip=_hip)


# --------------------------------------------------------------------------
# 注册
# --------------------------------------------------------------------------


def register_backend(
    name: str, factory: Callable[..., object], *, replace: bool = False
) -> None:
    """登记一个后端工厂。

    工厂的签名是 ``( **kwargs ) -> Backend``，参数原样来自 ``get_backend``，
    因此 ``get_backend("cuda", block=512)`` 等价于 ``CudaBackend(block=512)``。

    重复登记默认报错而不是覆盖：静默覆盖会让"我明明注册了"与"实际跑的是谁"分叉。
    插件作者要换实现，应该用新名字；``replace=True`` 留给测试与热修复。
    """
    key = str(name).strip().lower()
    if not key:
        raise ValueError("后端名不能为空")
    if not callable(factory):
        raise TypeError(
            f"后端 {key!r} 的工厂必须可调用，收到 {type(factory).__name__}"
        )
    if key in _FACTORIES and not replace:
        raise ValueError(f"后端 {key!r} 已注册；确需覆盖请传 replace=True")
    _FACTORIES[key] = factory


def unregister_backend(name: str, *, missing_ok: bool = True) -> bool:
    """注销一个非内置后端；返回是否真的删掉了。

    内置三个拒绝注销：退回路径依赖它们，删掉会让 ``fallback=True`` 无处可去。
    包卸载后调用它，是为了让"已经装好的进程"也不再挑到不存在的硬件。
    """
    key = str(name).strip().lower()
    if key in BUILTIN_BACKENDS:
        raise ValueError(f"内置后端 {key!r} 不能注销")
    if key not in _FACTORIES:
        if not missing_ok:
            raise LookupError(f"未注册的后端 {key!r}")
        return False
    del _FACTORIES[key]
    return True


def registered_backends() -> Tuple[str, ...]:
    """当前可用的后端名（内置 + entry point + 硬件包），按字母序。"""
    _discover()
    return tuple(sorted(_FACTORIES))


def _discover() -> None:
    """扫描 pip entry point，只扫一次。

    加载失败不打断流程：一个坏插件不该让其它后端也用不了。原因记在
    ``_ENTRY_ERRORS`` 里，排查时可以看。
    """
    global _discovered
    if _discovered:
        return
    _discovered = True
    try:
        found = importlib.metadata.entry_points()
        group = (
            found.select(group=_ENTRY_GROUP)
            if hasattr(found, "select")
            else found.get(_ENTRY_GROUP, ())
        )
    except Exception as exc:  # 元数据目录坏了也不该影响内置后端
        _ENTRY_ERRORS["<entry_points>"] = f"{type(exc).__name__}: {exc}"
        group = ()

    for entry in group:
        key = str(entry.name).strip().lower()
        if key in _FACTORIES:
            _ENTRY_ERRORS[key] = "与内置后端同名，已忽略"
            continue
        try:
            factory = entry.load()
        except Exception as exc:
            _ENTRY_ERRORS[key] = f"加载失败: {type(exc).__name__}: {exc}"
            continue
        if not callable(factory):
            _ENTRY_ERRORS[key] = "entry point 指向的不是可调用工厂"
            continue
        _FACTORIES[key] = factory

    # 硬件包（zip 安装到目录，见 hardware.py）排在 entry point 之后：两者的
    # 名字空间是同一个，先登记者优先，内置三个永远不被顶掉。
    try:
        from .hardware import build_factory, iter_installed
    except Exception as exc:
        _ENTRY_ERRORS["<hardware>"] = f"{type(exc).__name__}: {exc}"
        return
    for record in iter_installed():
        if record.name in _FACTORIES:
            _ENTRY_ERRORS[record.name] = (
                f"硬件包 {record.root} 与已注册后端同名，已忽略"
            )
            continue
        _FACTORIES[record.name] = build_factory(record)


# --------------------------------------------------------------------------
# 挑选
# --------------------------------------------------------------------------


def get_backend(name: Optional[str] = None, *, fallback: bool = True, **kwargs):
    """按名字 / 环境变量 / 自动探测挑一个后端并构造。

    ``kwargs`` 原样交给工厂，所以 ``get_backend("cuda", block=512)`` 与
    ``CudaBackend(block=512)`` 完全等价。
    """
    _discover()
    target = (name or os.environ.get(_ENV_VAR) or "auto").strip().lower()
    if target == "auto":
        return _autodetect(fallback=fallback, **kwargs)

    factory = _FACTORIES.get(target)
    if factory is None:
        known = "、".join(sorted(_FACTORIES)) or "（无）"
        raise LookupError(
            f"未注册的后端 {target!r}；已注册: {known}{_hardware_skip_hint(target)}"
        )
    return _build(target, factory, **kwargs)


def _hardware_skip_hint(name: str) -> str:
    """同名硬件包在目录里但扫描时被跳过，就把原因补进报错。

    典型场景：换机器时直接把源码覆盖进已安装的包目录，清单里的 sha256 与
    文件对不上。包被静默跳过，调用方只看到"未注册的后端"，很难联想到是
    校验失败。这里只补诊断，不改变"坏包跳过、不影响其它后端"的行为。
    """
    try:
        from .hardware import scan_errors
    except Exception:
        return ""
    try:
        reason = scan_errors().get(name)
    except Exception:
        return ""
    if not reason:
        return ""
    return (
        f"。注意：硬件包目录 {name!r} 存在但校验未通过，已被跳过：{reason}。"
        "不要直接覆盖已安装的文件，请重新打包后用 "
        "install_hardware(<zip 路径>, replace=True) 重装"
    )


def _build(name: str, factory: Callable[..., object], **kwargs):
    try:
        return factory(**kwargs)
    except Exception as exc:
        raise BackendUnavailableError(
            f"后端 {name!r} 已注册，但这台机器上初始化失败: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _autodetect(*, fallback: bool, **kwargs):
    reasons = []
    for name in _AUTO_ORDER:
        factory = _FACTORIES.get(name)
        if factory is None:
            continue
        try:
            return factory(**kwargs)
        except Exception as exc:
            reasons.append(f"{name}: {type(exc).__name__}: {exc}")

    detail = "；".join(reasons) or "未注册任何硬件后端"
    if not fallback:
        raise BackendUnavailableError(f"没有可用的硬件后端（{detail}）")

    # 退回 CPU 语义执行是允许的，但必须让调用方看得见：backend.name 会是
    # "reference"，同时这里发警告。两者都不足以被忽略。
    warnings.warn(
        "没有可用的硬件后端，退回参考后端（backend.name == 'reference'，"
        f"纯 CPU 语义执行，无硬件加速）：{detail}",
        RuntimeWarning,
        stacklevel=3,
    )
    return _build("reference", _FACTORIES["reference"], **kwargs)
