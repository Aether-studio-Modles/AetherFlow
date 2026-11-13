"""部署包：把"图 + 权重 + 清单"打成一个自校验的文件。

动机：训练产物离开进程的方式目前是两条互不相干的路径——图走
``ir.serialize`` 的 JSON，权重走 ``train.checkpoint`` 的 ``.npz``。现场部署时
这两样必须**一起**出现，且必须能回答"这份权重属于这张图吗、文件有没有被改
过"。把它们装进一个文件、由清单把两者的摘要绑在一起，第三个问题才有答案。

**格式**（zip，固定三个成员，多一个少一个都拒绝）::

    manifest.json
    graph.json          # ir.serialize.dumps 的原文
    weights.npz         # 名字 -> 数组，只读、不允许 pickle

    {
      "format": "aetherflow-bundle",
      "version": 1,
      "ir_version": [1, 0],
      "graph": {"name": "mlp", "sha256": "<graph.json 的摘要>"},
      "weights": {
        "w1": {"shape": [16, 16], "dtype": "float32", "sha256": "<数组字节的摘要>"}
      },
      "meta": {"accuracy": 0.97}
    }

三条刻意的决定

1. **摘要按数组字节算，不按 npz 文件算。** ``np.savez`` 写的是 zip，条目里
   带时间戳，同一个数组两次打包的**文件**字节不同；按文件摘要校验会让"内容
   没变"被判成篡改。按 ``tobytes()`` 算则只取决于数值本身。
2. **清单是格式的一部分，不是注释。** 加载时依次校验格式名 / 版本 / IR 版本 /
   图摘要 / 每个权重的形状、dtype 与摘要；任何一处不符直接失败。这不是签名，
   挡不住"从头伪造一个包"，但那属于发布方信任链的问题，不在这里。
3. **权重必须对得上图的输入。** 名字不在图输入里、形状或 dtype 与声明不符，
   都在加载期拒绝——现场拿到一个"能加载但一跑就炸"的包是最坏的情况。
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .backends.base import Backend
from .backends.reference_backend import ReferenceBackend
from .ir import serialize
from .ir.graph import Graph

__all__ = ["BundleError", "Deployment", "save_bundle", "load_bundle"]

_FORMAT = "aetherflow-bundle"
_VERSION = 1
_SUFFIX = ".afpkg"

_MANIFEST = "manifest.json"
_GRAPH = "graph.json"
_WEIGHTS = "weights.npz"
_MEMBERS = (_MANIFEST, _GRAPH, _WEIGHTS)

# 解压上限：部署包可以带权重，但不该是几个 GB 的"压缩炸弹"。
MAX_BYTES = 8 * 1024 * 1024 * 1024


class BundleError(RuntimeError):
    """部署包不成立：格式不对、摘要不符、权重与图不匹配。"""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_digest(array: np.ndarray) -> str:
    return _digest(np.ascontiguousarray(array).tobytes())


def _check_weights(graph: Graph, weights: Mapping[str, np.ndarray]) -> None:
    inputs = {value.name: value for value in graph.inputs}
    for name, array in weights.items():
        if not isinstance(array, np.ndarray):
            raise TypeError(
                f"权重 {name!r} 必须是 numpy 数组，得到 {type(array).__name__}"
            )
        if name not in inputs:
            raise BundleError(f"权重 {name!r} 不是这张图的输入: {sorted(inputs)}")
        type_ = inputs[name].type
        if not type_.is_static:
            raise BundleError(
                f"权重 {name!r} 的形状 {type_} 含符号维；部署包的权重必须是静态形状"
            )
        if tuple(int(d) for d in type_.shape) != tuple(array.shape):
            raise BundleError(
                f"权重 {name!r} 的形状 {array.shape} 与图输入的 {tuple(type_.shape)} 不一致"
            )
        if np.dtype(type_.dtype.np_dtype) != array.dtype:
            raise BundleError(
                f"权重 {name!r} 的类型 {array.dtype} 与图输入的 "
                f"{np.dtype(type_.dtype.np_dtype)} 不一致"
            )


def save_bundle(
    path,
    graph: Graph,
    weights: Mapping[str, np.ndarray],
    meta: Optional[Mapping[str, Any]] = None,
) -> str:
    """把图与权重写成一个 ``.afpkg``，返回文件路径。"""
    target = str(path)
    if not target.endswith(_SUFFIX):
        raise ValueError(f"部署包后缀必须是 {_SUFFIX}，得到 {target!r}")
    if not isinstance(graph, Graph):
        raise TypeError(f"graph 必须是 Graph，得到 {type(graph).__name__}")
    weights = dict(weights)
    _check_weights(graph, weights)

    graph_text = serialize.dumps(graph)
    graph_bytes = graph_text.encode("utf-8")

    buffer = io.BytesIO()
    np.savez(buffer, **weights)

    meta = dict(meta or {})
    manifest = {
        "format": _FORMAT,
        "version": _VERSION,
        "ir_version": list(graph.ir_version),
        "graph": {"name": graph.name, "sha256": _digest(graph_bytes)},
        "weights": {
            name: {
                "shape": list(array.shape),
                "dtype": array.dtype.name,
                "sha256": _array_digest(array),
            }
            for name, array in weights.items()
        },
        "meta": meta,
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(_MANIFEST, manifest_bytes)
        archive.writestr(_GRAPH, graph_bytes)
        archive.writestr(_WEIGHTS, buffer.getvalue())
    return target


def _read(path: str) -> Dict[str, bytes]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            unknown = sorted(set(names) - set(_MEMBERS))
            if unknown:
                raise BundleError(f"部署包里有多余的成员: {unknown}")
            missing = [name for name in _MEMBERS if name not in names]
            if missing:
                raise BundleError(f"部署包缺少成员: {missing}")
            total = sum(archive.getinfo(name).file_size for name in _MEMBERS)
            if total > MAX_BYTES:
                raise BundleError(f"部署包解压后 {total} 字节，超过上限 {MAX_BYTES}")
            return {name: archive.read(name) for name in _MEMBERS}
    except (OSError, zipfile.BadZipFile) as exc:
        raise BundleError(f"读不出部署包 {path}：{exc}") from exc


@dataclass(frozen=True)
class Deployment:
    """加载后的部署包。``run`` 把权重与这一批数据合起来执行一次。"""

    graph: Graph
    weights: Dict[str, np.ndarray]
    meta: Dict[str, Any]

    def run(
        self, inputs: Mapping[str, Any], backend: Optional[Backend] = None
    ) -> Dict[str, Any]:
        """执行一次前向，返回图输出（名字与构图时一致）。"""
        engine = backend if backend is not None else ReferenceBackend()
        feed = dict(self.weights)
        for name, value in inputs.items():
            if name in feed:
                raise KeyError(f"输入 {name!r} 与权重同名")
            feed[name] = value
        return engine.execute(self.graph, feed)


def load_bundle(path) -> Deployment:
    """读入并校验部署包；任何一处不符都抛 ``BundleError``。"""
    source = str(path)
    members = _read(source)
    try:
        manifest = json.loads(members[_MANIFEST].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"manifest.json 不是合法的 JSON：{exc}") from exc

    if manifest.get("format") != _FORMAT:
        raise BundleError(
            f"不是部署包：format 为 {manifest.get('format')!r}，期望 {_FORMAT!r}"
        )
    if manifest.get("version") != _VERSION:
        raise BundleError(
            f"部署包版本 {manifest.get('version')!r} 不受支持（本实现是 {_VERSION}）"
        )

    graph_bytes = members[_GRAPH]
    declared = manifest.get("graph", {})
    actual = _digest(graph_bytes)
    if declared.get("sha256") != actual:
        raise BundleError(
            "graph.json 的校验和不符：清单声明 "
            f"{declared.get('sha256')}，实际 {actual}"
        )
    try:
        graph = serialize.loads(graph_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
        raise BundleError(f"graph.json 读不回来：{exc}") from exc
    if list(graph.ir_version) != list(manifest.get("ir_version", [])):
        raise BundleError(
            f"图的 IR 版本 {list(graph.ir_version)} 与清单声明的 "
            f"{manifest.get('ir_version')} 不一致"
        )

    try:
        with np.load(io.BytesIO(members[_WEIGHTS]), allow_pickle=False) as data:
            weights = {name: np.array(data[name], copy=True) for name in data.files}
    except (OSError, ValueError) as exc:
        raise BundleError(f"weights.npz 读不出来：{exc}") from exc

    declared_weights = manifest.get("weights", {})
    if set(declared_weights) != set(weights):
        raise BundleError(
            "权重集合与清单不一致：缺少 "
            f"{sorted(set(declared_weights) - set(weights))}，多出 "
            f"{sorted(set(weights) - set(declared_weights))}"
        )
    for name, array in weights.items():
        spec = declared_weights[name]
        if list(array.shape) != list(spec.get("shape", [])):
            raise BundleError(
                f"权重 {name!r} 的形状 {list(array.shape)} 与清单的 "
                f"{spec.get('shape')} 不一致"
            )
        if array.dtype.name != spec.get("dtype"):
            raise BundleError(
                f"权重 {name!r} 的类型 {array.dtype.name} 与清单的 "
                f"{spec.get('dtype')} 不一致"
            )
        digest = _array_digest(array)
        if digest != spec.get("sha256"):
            raise BundleError(f"权重 {name!r} 的校验和不符：实际 {digest}")
    _check_weights(graph, weights)

    meta = manifest.get("meta", {})
    if not isinstance(meta, dict):
        raise BundleError(f"manifest.meta 必须是对象，得到 {type(meta).__name__}")
    return Deployment(graph=graph, weights=weights, meta=meta)
