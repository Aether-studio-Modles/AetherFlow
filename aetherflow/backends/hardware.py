"""硬件包：把一个后端装成 zip，不改框架代码。

动机：pip entry point 要求"先打包、再 pip install"，对插件作者与现场部署都是
一次完整的环境操作。而硬件包里真正需要变的东西往往只有方言与少量运行时胶水，
把它做成"放进目录就识别"的 zip，换硬件不必重建环境。

**格式**（zip，根目录必须有 ``manifest.json``）::

    manifest.json
    mychip/__init__.py
    mychip/backend.py

    {
      "name": "mychip",                        # ^[a-z][a-z0-9_]*$，不能与内置同名
      "version": "0.1.0",
      "aetherflow_api": "1",                   # 可选，主版本必须与框架一致
      "factory": "mychip.backend:MyChipBackend",
      "vendor": "可选",
      "description": "可选",
      "files": {                               # 可选，声明了就强校验
        "mychip/__init__.py": "<sha256>",
        "mychip/backend.py": "<sha256>"
      },
      "requires": {                            # 可选
        "ops": ["matmul", "relu"],             # 框架算子表里必须都有
        "python": ["numpy"]                    # 当前环境必须能导入
      },
      "signature": {                           # 可选，见下
        "algorithm": "ed25519",
        "public_key": "<64 位十六进制>",
        "value": "<128 位十六进制>"
      }
    }

**内容摘要不是签名。** ``files`` 只能证明"包在打包后没被改过"（或者说，改动
与清单里的摘要不符）。它挡不住一个从头伪造的包——伪造者把清单也一起算一遍
就通过。

**可选的 Ed25519 签名。** 签名覆盖"除 ``signature`` 以外的整份清单"（规范化
JSON），而不只是文件摘要——否则改 ``version`` / ``factory`` 就绕过去了。信任
锚不在包里：``install_hardware`` 接受 ``trusted_keys``，包内声明的公钥必须在
安装方给出的列表里才继续验签。包带签名却不给 ``trusted_keys``、或者给了
``trusted_keys`` 但包没签名，都直接拒绝——"有签名但没人验"与没签名没有区别，
静默跳过是安全反模式。签名只证明"这份清单出自某把私钥"，渠道与私钥保管仍在
发布方。

**清单要么全有、要么没有。** 不声明 ``files`` 的包照旧加载（手工解压进目录
的场景没有清单）；一旦声明，就要求与包内文件**完全一致**，多一个未声明文件
同样拒绝——只校验声明过的子集，等于给"偷偷多放一个 sitecustomize.py"留门。

**识别位置**（先命中者优先，同名只认第一个）：

1. ``$AETHERFLOW_HARDWARE_DIR``：存在时**只**扫这一个，供测试与隔离部署；
2. ``~/.aetherflow/hardware``：``install_hardware`` 的安装目标；
3. ``aetherflow/hardware``：随框架分发，只读。

**边界（刻意不做的）**

- **zip 不原地 import**。必须解压：``zipimport`` 只能加载纯 Python，而硬件包的
  编译 / 加载发生在运行期，路径必须真实存在（NVRTC、``ctypes.CDLL`` 都按路径找库）。
- **这层不做 IR 转译**。转译的归属是 ``lower/``（L2 内核 IR）与
  ``backends/*/dialect.py``（方言数据）。硬件包只是把"方言 + 运行时胶水"**装进来**，
  它不是第四个转译层——把转译逻辑塞进装配层会让"一套 IR 落到不同硬件"重新变成
  各家自己解释 IR。
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import stat
import sys
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, Iterable, Iterator, Optional

__all__ = [
    "HardwarePackageError",
    "HardwareRecord",
    "search_roots",
    "install_root",
    "install_hardware",
    "uninstall_hardware",
    "installed_hardware",
    "iter_installed",
    "build_factory",
]

_ENV_DIR = "AETHERFLOW_HARDWARE_DIR"
_BUNDLED_DIR = Path(__file__).resolve().parent.parent / "hardware"

MANIFEST = "manifest.json"
# 上限用于挡住 zip bomb 与误打包的巨型资源：硬件包是代码，不该有几百 MB 权重。
MAX_ENTRIES = 2000
MAX_BYTES = 64 * 1024 * 1024

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
# module:attr，两段都允许点号（包路径 / 嵌套属性）
_FACTORY_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
# 运行期产物不该被当成"未声明文件"：导入过一次，包里就会多出这些。
_CACHE_PARTS = frozenset({"__pycache__"})
_REQUIRES_KEYS = frozenset({"ops", "python"})
_SIGNATURE_KEYS = frozenset({"algorithm", "public_key", "value"})
_SIGNATURE_ALGORITHMS = frozenset({"ed25519"})


class HardwarePackageError(RuntimeError):
    """硬件包不合法，或安装 / 加载失败。

    与 ``BackendUnavailableError`` 的分工：那个是"包装好了，但这台机器装不起来"；
    这个连包本身都不成立（缺 manifest、路径穿越、API 不匹配）。
    """


@dataclass(frozen=True)
class HardwareRecord:
    """一个已安装硬件包的元信息。``root`` 是解压后的目录。"""

    name: str
    version: str
    api: str
    factory: str
    root: Path
    vendor: str = ""
    description: str = ""
    signed: bool = False


# --------------------------------------------------------------------------
# 位置
# --------------------------------------------------------------------------


def search_roots() -> tuple:
    """按优先级返回硬件包的搜索根目录。"""
    env = os.environ.get(_ENV_DIR)
    if env:
        return (Path(env).expanduser(),)
    return (Path.home() / ".aetherflow" / "hardware", _BUNDLED_DIR)


def install_root() -> Path:
    """``install_hardware`` 的默认安装目录。"""
    return search_roots()[0]


# 信任记录放在**安装根目录**里，不能放进包目录：包内的每个文件都会参与摘要
# 校验，多一个记录文件会变成"未声明文件"而被判为坏包。
_TRUST_FILE = ".trust.json"


def _trust_path(root: Path) -> Path:
    return Path(root) / _TRUST_FILE


def _load_trust(root: Path) -> Dict[str, Dict[str, str]]:
    """读回安装时留下的来源记录；文件缺失或损坏时当作"没有记录"。

    损坏时选择空表而不是抛错：扫描目录的契约是"坏包跳过、不打断别人"，
    一个写坏的小 json 不该让所有已装硬件都消失。代价是带签名的包会因为
    找不到信任锚而在扫描时被跳过——这与"记录丢了"的事实一致。
    """
    path = _trust_path(root)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for name, entry in raw.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            continue
        algorithm = entry.get("algorithm")
        public_key = entry.get("public_key")
        if isinstance(algorithm, str) and isinstance(public_key, str):
            out[name] = {"algorithm": algorithm, "public_key": public_key}
    return out


def _save_trust(root: Path, entries: Dict[str, Dict[str, str]]) -> None:
    """原子替换信任记录；表空时直接删掉文件。"""
    path = _trust_path(root)
    if not entries:
        try:
            path.unlink()
        except OSError:
            pass
        return
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(entries, sort_keys=True, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# 校验
# --------------------------------------------------------------------------


def _framework_api() -> str:
    import aetherflow

    return str(getattr(aetherflow, "__api_version__", "1"))


def _check_api(declared: str, where: str) -> str:
    """只比主版本：次版本只增不减，不该阻止旧包加载。"""
    want = _framework_api().split(".")[0]
    got = str(declared).split(".")[0]
    if got != want:
        raise HardwarePackageError(
            f"{where}: 硬件包声明 aetherflow_api={declared!r}，"
            f"当前框架是 {_framework_api()!r}。不兼容的包不做静默尝试。"
        )
    return str(declared)


def _read_manifest(root: Path) -> dict:
    path = root / MANIFEST
    if not path.is_file():
        raise HardwarePackageError(f"{root}: 缺少 {MANIFEST}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HardwarePackageError(f"{path}: 读不出合法 JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise HardwarePackageError(f"{path}: 顶层必须是对象")
    return data


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_files(root: Path) -> Dict[str, Path]:
    """包里所有参与摘要的文件，键是相对 POSIX 路径。

    ``manifest.json`` 不在其中：清单就写在它里面，它不能有自己的摘要。
    ``__pycache__`` 与 ``*.pyc`` 是运行期产物，导入一次就会生成——把它们算进
    摘要，会让"装好、用一次、再扫描"变成摘要不符。
    """
    found: Dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == MANIFEST or relative.endswith(".pyc"):
            continue
        if _CACHE_PARTS.intersection(PurePosixPath(relative).parts):
            continue
        found[relative] = path
    return found


def _check_files(data: dict, root: Path) -> None:
    """校验内容摘要：声明了就要求与包内文件完全一致。"""
    declared = data.get("files")
    if declared is None:
        return
    if not isinstance(declared, dict):
        raise HardwarePackageError(f"{root}: files 必须是 {{路径: sha256}} 对象")

    actual = _package_files(root)
    for name in declared:
        if not isinstance(name, str) or not name:
            raise HardwarePackageError(f"{root}: files 的键必须是非空路径字符串")
        _safe_member(name)
        want = declared[name]
        if not isinstance(want, str) or not _SHA256_RE.match(want):
            raise HardwarePackageError(
                f"{root}: files[{name!r}] 必须是 64 位十六进制 sha256，收到 {want!r}"
            )
        path = actual.get(name)
        if path is None:
            raise HardwarePackageError(
                f"{root}: files 声明了 {name!r}，包里没有这个文件"
            )
        got = _digest(path)
        if got != want.lower():
            raise HardwarePackageError(
                f"{root}: {name!r} 摘要不符（声明 {want.lower()}，实际 {got}）；"
                "包在打包后被改动过"
            )

    extra = sorted(set(actual) - set(declared))
    if extra:
        raise HardwarePackageError(
            f"{root}: 这些文件没有写进 files: {extra}；摘要覆盖必须完整"
        )


def _check_requires(data: dict, root: Path) -> None:
    """校验依赖声明：包用到的算子与第三方模块，当前框架 / 环境得真的有。"""
    requires = data.get("requires")
    if requires is None:
        return
    if not isinstance(requires, dict):
        raise HardwarePackageError(f"{root}: requires 必须是对象")

    unknown = sorted(set(requires) - _REQUIRES_KEYS)
    if unknown:
        # 拼错的键（"op"、"python3"）会让整条声明静默失效，所以不放过。
        raise HardwarePackageError(
            f"{root}: requires 不认识的键 {unknown}；支持 {sorted(_REQUIRES_KEYS)}"
        )

    ops = requires.get("ops", ())
    if not isinstance(ops, (list, tuple)):
        raise HardwarePackageError(f"{root}: requires.ops 必须是字符串数组")
    from ..ir.ops import OPS

    missing = [name for name in ops if not isinstance(name, str) or name not in OPS]
    if missing:
        raise HardwarePackageError(
            f"{root}: 依赖的算子不在当前框架的算子表里: {missing}；包与框架版本不匹配"
        )

    modules = requires.get("python", ())
    if not isinstance(modules, (list, tuple)):
        raise HardwarePackageError(f"{root}: requires.python 必须是字符串数组")
    absent = []
    for name in modules:
        if not isinstance(name, str) or not name:
            raise HardwarePackageError(
                f"{root}: requires.python 里必须是非空字符串，收到 {name!r}"
            )
        # 只查顶层模块，且用 find_spec 而不是 import：安装期不该执行包作者的代码。
        try:
            found = importlib.util.find_spec(name.split(".")[0])
        except (ImportError, AttributeError, ValueError):
            found = None
        if found is None:
            absent.append(name)
    if absent:
        raise HardwarePackageError(
            f"{root}: 缺少依赖模块 {absent}；先装好再安装硬件包"
        )


def _public_key_bytes(value: object) -> bytes:
    """把公钥统一成 32 字节原始形态；只接受 bytes 或十六进制字符串。"""
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        try:
            raw = bytes.fromhex(value.strip())
        except ValueError:
            raise HardwarePackageError(
                f"公钥必须是十六进制字符串，收到 {value!r}"
            ) from None
    else:
        raise HardwarePackageError(
            f"公钥必须是 bytes 或十六进制字符串，收到 {type(value).__name__}"
        )
    if len(raw) != 32:
        raise HardwarePackageError(f"Ed25519 公钥必须是 32 字节，收到 {len(raw)} 字节")
    return raw


def _signature_payload(data: dict) -> bytes:
    """签名覆盖"除 signature 以外的整份清单"，规范化后取 UTF-8。

    不把 ``files`` 摘要单独拎出来签：那样改 ``version`` 或 ``factory`` 仍然
    能过验签，等于给"换个工厂指向恶意模块"留门。
    """
    body = {key: value for key, value in data.items() if key != "signature"}
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _trust_entry(manifest: dict) -> Optional[Dict[str, str]]:
    """从已校验的清单里取出要持久化的信任锚；未签名时返回 ``None``。"""
    signature = manifest.get("signature")
    if not isinstance(signature, dict):
        return None
    return {
        "algorithm": str(signature.get("algorithm")),
        "public_key": _public_key_bytes(signature.get("public_key")).hex(),
    }


def _check_signature(
    data: dict, root: Path, trusted_keys: Optional[Iterable[object]]
) -> bool:
    """校验可选签名，返回"这个包是否带签名"。

    ``trusted_keys is None`` 表示调用方不要求来源校验；此时带签名的包会被
    拒绝，而不是"验不了就当没看见"。
    """
    declared = data.get("signature")
    if declared is None:
        if trusted_keys is not None:
            raise HardwarePackageError(
                f"{root}: 指定了 trusted_keys，但这个包没有签名；拒绝未签名来源"
            )
        return False

    if not isinstance(declared, dict):
        raise HardwarePackageError(f"{root}: signature 必须是对象")
    unknown = sorted(set(declared) - _SIGNATURE_KEYS)
    if unknown:
        raise HardwarePackageError(
            f"{root}: signature 不认识的键 {unknown}；支持 {sorted(_SIGNATURE_KEYS)}"
        )

    algorithm = declared.get("algorithm")
    if algorithm not in _SIGNATURE_ALGORITHMS:
        raise HardwarePackageError(
            f"{root}: 不支持的签名算法 {algorithm!r}；支持 {sorted(_SIGNATURE_ALGORITHMS)}"
        )
    if trusted_keys is None:
        raise HardwarePackageError(
            f"{root}: 包带签名，但没有提供 trusted_keys，无法确认来源"
        )

    public_key = _public_key_bytes(declared.get("public_key"))
    trusted = {_public_key_bytes(key) for key in trusted_keys}
    if public_key not in trusted:
        raise HardwarePackageError(
            f"{root}: 包公钥不在 trusted_keys 里；这份签名不是受信任的发布方签的"
        )

    value = declared.get("value")
    if not isinstance(value, str):
        raise HardwarePackageError(f"{root}: signature.value 必须是十六进制字符串")
    try:
        raw = bytes.fromhex(value.strip())
    except ValueError:
        raise HardwarePackageError(f"{root}: signature.value 不是合法十六进制") from None
    if len(raw) != 64:
        raise HardwarePackageError(
            f"{root}: Ed25519 签名必须是 64 字节，收到 {len(raw)} 字节"
        )

    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:  # 不给可选依赖制造硬依赖
        raise HardwarePackageError(
            "校验硬件包签名需要 cryptography：pip install cryptography"
        ) from exc

    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            raw, _signature_payload(data)
        )
    except InvalidSignature:
        raise HardwarePackageError(
            f"{root}: 签名与清单不符；包在签名后被改过，或不是这把私钥签的"
        ) from None
    return True


def _validate(
    data: dict, root: Path, trusted_keys: Optional[Iterable[object]] = None
) -> HardwareRecord:
    name = data.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise HardwarePackageError(
            f"{root}: name 必须是 ^[a-z][a-z0-9_]*$ 的小写标识符，收到 {name!r}"
        )

    version = data.get("version")
    if not isinstance(version, str) or not version.strip():
        raise HardwarePackageError(f"{root}: version 必须是非空字符串")

    api = _check_api(data.get("aetherflow_api", "1"), root)

    factory = data.get("factory")
    if not isinstance(factory, str) or not _FACTORY_RE.match(factory):
        raise HardwarePackageError(
            f"{root}: factory 必须是 'module:attr' 形式，收到 {factory!r}"
        )

    for key in ("vendor", "description"):
        value = data.get(key, "")
        if not isinstance(value, str):
            raise HardwarePackageError(f"{root}: {key} 必须是字符串")

    # 放到最后：字段都合法了再算哈希与查依赖，错误顺序才符合直觉。
    _check_files(data, root)
    _check_requires(data, root)
    signed = _check_signature(data, root, trusted_keys)

    return HardwareRecord(
        name=name,
        version=version.strip(),
        api=api,
        factory=factory,
        root=root,
        vendor=data.get("vendor", ""),
        description=data.get("description", ""),
        signed=signed,
    )


def _record_from_dir(
    root: Path, trusted_keys: Optional[Iterable[object]] = None
) -> HardwareRecord:
    return _validate(_read_manifest(root), root, trusted_keys)


# --------------------------------------------------------------------------
# 扫描
# --------------------------------------------------------------------------


def iter_installed() -> Iterator[HardwareRecord]:
    """遍历所有可识别的硬件包。

    坏包跳过，不打断别人：manifest 不合法、文件摘要不符、依赖不满足，
    都属于"这个包在这台机器上用不了"，不是"整个硬件目录坏了"。

    安装时验过的来源会被持久化到安装根目录的 ``.trust.json``，扫描时按
    记录把公钥重新作为信任锚传进去。否则带签名的包装好之后，下次启动
    会因为没有 ``trusted_keys`` 而被判为"无法确认来源"，等于签名包只能用
    到进程退出——这比不签名还糟。
    """
    seen = set()
    for root in search_roots():
        if not root.is_dir():
            continue
        trust = _load_trust(root)
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if not (child / MANIFEST).is_file():
                continue
            entry = trust.get(child.name)
            trusted = [entry["public_key"]] if entry is not None else None
            try:
                record = _record_from_dir(child, trusted)
            except HardwarePackageError:
                continue
            if record.name in seen:
                continue
            seen.add(record.name)
            yield record


def installed_hardware() -> Dict[str, HardwareRecord]:
    """当前机器上可识别的硬件包，按名字索引。"""
    return {record.name: record for record in iter_installed()}


# --------------------------------------------------------------------------
# 加载
# --------------------------------------------------------------------------


def build_factory(record: HardwareRecord) -> Callable[..., object]:
    """把 manifest 里的 ``module:attr`` 变成一个惰性工厂。

    惰性很重要：扫描目录不该 import 任何硬件代码——没装驱动时 import 可能直接
    报错，那会让 ``registered_backends()`` 也一起坏掉。
    """

    def factory(**kwargs):
        return _resolve_target(record)(**kwargs)

    factory.__name__ = f"hardware_{record.name}"
    factory.__doc__ = f"硬件包 {record.name} {record.version}（{record.root}）"
    return factory


def _resolve_target(record: HardwareRecord) -> Callable[..., object]:
    root = str(record.root)
    # 追加到末尾而不是开头：硬件包不该遮蔽已安装的同名模块。
    if root not in sys.path:
        sys.path.append(root)
    importlib.invalidate_caches()

    module_name, _, attr = record.factory.partition(":")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise HardwarePackageError(
            f"硬件包 {record.name!r} 加载 {module_name!r} 失败: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    origin = os.path.abspath(getattr(module, "__file__", "") or "")
    if origin and not origin.startswith(os.path.abspath(root) + os.sep):
        raise HardwarePackageError(
            f"硬件包 {record.name!r} 的模块 {module_name!r} 已被 {origin} 占用；"
            "请给包内模块加前缀，避免与其它包同名"
        )

    target = module
    for part in attr.split("."):
        try:
            target = getattr(target, part)
        except AttributeError as exc:
            raise HardwarePackageError(
                f"硬件包 {record.name!r}: {module_name}.{attr} 不存在"
            ) from exc
    if not callable(target):
        raise HardwarePackageError(
            f"硬件包 {record.name!r}: {record.factory} 指向的不是可调用工厂"
        )
    return target


# --------------------------------------------------------------------------
# 安装 / 卸载
# --------------------------------------------------------------------------


def _safe_member(name: str) -> None:
    if "\\" in name:
        raise HardwarePackageError(f"包内路径不合法（反斜杠）: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or re.match(r"^[A-Za-z]:", name):
        raise HardwarePackageError(f"包内路径不合法（绝对路径）: {name!r}")
    if any(part == ".." for part in path.parts):
        raise HardwarePackageError(f"包内路径不合法（路径穿越）: {name!r}")


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def _extract(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(src) as zf:
            infos = zf.infolist()
            if len(infos) > MAX_ENTRIES:
                raise HardwarePackageError(
                    f"{src}: 条目数 {len(infos)} 超过上限 {MAX_ENTRIES}"
                )
            total = 0
            for info in infos:
                _safe_member(info.filename)
                if _is_symlink(info):
                    raise HardwarePackageError(
                        f"{src}: 包含符号链接 {info.filename!r}，拒绝解压"
                    )
                total += info.file_size
                if total > MAX_BYTES:
                    raise HardwarePackageError(
                        f"{src}: 解压后超过 {MAX_BYTES // (1024 * 1024)} MB 上限"
                    )
            zf.extractall(dest)
    except zipfile.BadZipFile as exc:
        raise HardwarePackageError(f"{src}: 不是合法的 zip: {exc}") from exc


def install_hardware(
    source, *, replace: bool = False, root: Optional[Path] = None,
    trusted_keys: Optional[Iterable[object]] = None,
) -> HardwareRecord:
    """安装一个 zip 硬件包并**立即登记**，返回它的元信息。

    解压先落到同一目录下的临时目录，校验通过后再原子改名：中途失败不会留下
    半个包，也不会破坏已安装的同名版本。

    ``trusted_keys`` 是来源校验的信任锚：给了它，包必须**带签名且公钥在列表
    里**；不给，带签名的包会被拒绝（"验不了就当没看见"等于没签）。扫描已装
    目录时不再验签——安装那一刻确认过来源，之后每次启动都要求密钥并不现实。
    """
    src = Path(source)
    if not src.is_file():
        raise HardwarePackageError(f"{src}: 文件不存在")

    target_root = Path(root) if root is not None else install_root()
    target_root.mkdir(parents=True, exist_ok=True)
    tmp = target_root / f".install-{uuid.uuid4().hex[:8]}"
    try:
        _extract(src, tmp)
        manifest = _read_manifest(tmp)
        record = _validate(manifest, tmp, trusted_keys)
        # 内置名不允许被硬件包占用：装一个叫 cuda 的包去顶掉真 CUDA 后端，
        # 是"看起来装好了、其实跑的是别人的实现"这类事故的标准模板。
        from .discovery import BUILTIN_BACKENDS

        if record.name in BUILTIN_BACKENDS:
            raise HardwarePackageError(
                f"硬件包不能用内置后端名 {record.name!r}；请换一个名字"
            )
        dest = target_root / record.name
        if dest.exists():
            if not replace:
                raise HardwarePackageError(
                    f"硬件包 {record.name!r} 已安装在 {dest}；要覆盖请传 replace=True"
                )
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        os.replace(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    # 把这次验过的来源记到安装根目录：下次扫描要用它重建信任锚，否则带签名
    # 的包会在重启后被判为"无法确认来源"。无签名的包必须清掉同名旧记录，
    # 不然扫描会拿旧公钥去要求一个本来就没有签名的包。
    trust = _load_trust(target_root)
    entry = _trust_entry(manifest)
    if entry is None:
        trust.pop(record.name, None)
    else:
        trust[record.name] = entry
    _save_trust(target_root, trust)

    record = HardwareRecord(
        name=record.name,
        version=record.version,
        api=record.api,
        factory=record.factory,
        root=dest,
        vendor=record.vendor,
        description=record.description,
        signed=record.signed,
    )
    # 本进程立即可用：_discover() 只扫一次，不重扫的话刚装的包要重启才生效。
    from .discovery import register_backend

    register_backend(record.name, build_factory(record), replace=True)
    return record


def uninstall_hardware(name: str) -> bool:
    """删除已安装的包装目录并注销登记。

    只删 ``install_root()`` 下的目录：随框架分发的只读包不归这里管。
    返回 False 表示本来就没装。
    """
    key = str(name).strip().lower()
    if not _NAME_RE.match(key):
        raise HardwarePackageError(f"硬件包名不合法: {name!r}")
    dest = install_root() / key
    if not dest.is_dir():
        return False
    shutil.rmtree(dest)
    trust = _load_trust(install_root())
    if key in trust:
        trust.pop(key)
        _save_trust(install_root(), trust)

    from .discovery import unregister_backend

    unregister_backend(key, missing_ok=True)
    return True
