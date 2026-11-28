"""safetensors 权重读取与装载（生态入口的第一块）。

为什么要自己读 ``.safetensors`` 而不是引入 ``safetensors`` 包：

1. 本包的运行期依赖只有 numpy（``pyproject.toml`` 只声明 ``numpy>=1.24``）。
   为了读一个容器格式就多装一个二进制依赖，会让"参考后端在任意平台可用"
   这条约束退化。
2. 格式本身很小：8 字节小端头长 + 一段 JSON 头 + 连续数据区。用标准库解析
   比维护一条二进制依赖更容易审。
3. 解析器必须**先校验后读取**。坏文件（头长离谱、offsets 越界、形状与字节数
   对不上）要在拿到第一块内存之前报错——权重文件是外部输入，这里的默认假设
   应当是"它是坏的"，而不是"它是好的"。

**它不是 checkpoint 格式**。``train/checkpoint.py`` 存的是训练状态（参数 +
优化器槽位 + 流式状态 + meta），可续训；safetensors 只存张量，是"把别人的
权重搬进来"的入口。两者不重叠：前者自洽，后者要配一张名字映射表。

三条使用约束：

- **名字不会自动猜。** 外部权重的命名（``model.layers.0.self_attn.q_proj.weight``）
  与本仓库的图输入名（``w_q``）没有可推导关系，``load_into_store`` 默认同名
  对应，不同名的必须显式 ``mapping``。
- **布局不会自动猜。** PyTorch 的 ``nn.Linear`` 权重是 ``(out, in)``，本仓库的
  ``matmul`` 是 ``x @ W``（``(in, out)``），差一次转置。转置必须写在
  ``transpose`` 里——猜错不会报错，只会训出一个静默错误的模型。
- **装填是原地写。** 优化器持有 ``store.arrays()`` 里那批数组对象；替换引用
  会让优化器继续更新没人用的旧数组。所以这里先全量校验，再 ``dst[...] = ...``，
  中途失败不会留下"一半新一半旧"的参数。
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

__all__ = [
    "SafeTensorsError",
    "TensorInfo",
    "SafeTensorsFile",
    "load_safetensors",
    "load_safetensors_dir",
    "load_into_store",
]

# safetensors 官方实现对头的上限就是 100 MB。这里照着卡住：头是一个 JSON
# 字典，正常模型只有几十 KB；一个声称"头有 400 GB"的文件只可能是在骗读取方
# 先分配内存。
_MAX_HEADER_BYTES = 100 << 20

# dtype 名 → 每元素字节数。F8 两种格式登记在案但**不支持**（见 ``_read_tensor``）：
# 本仓库还没有 fp8 数值类型，假装读成 u8 只会把问题推到链路下游。
_ITEMSIZE: Dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

# 小端显式写出来：safetensors 规定小端，而 numpy 的默认字节序跟机器走。
# 在大端机上用本机序读会得到安静翻转的数值，这是最难查的一类错。
_NUMPY: Dict[str, str] = {
    "BOOL": "u1",
    "U8": "u1",
    "I8": "i1",
    "I16": "<i2",
    "U16": "<u2",
    "F16": "<f2",
    "I32": "<i4",
    "U32": "<u4",
    "F32": "<f4",
    "I64": "<i8",
    "U64": "<u8",
    "F64": "<f8",
}

_FP8 = ("F8_E4M3", "F8_E5M2")


class SafeTensorsError(ValueError):
    """权重文件不可读、不合规，或与目标参数对不上。

    继承 ``ValueError`` 是为了让调用方既能按"权重装载失败"统一接住，也不会
    在一个只捕获 ``ValueError`` 的外层里漏掉。错误信息里必须带上**具体是哪
    一项**（张量名 / 偏移 / 期望形状），否则现场只能靠猜。
    """


@dataclass(frozen=True)
class TensorInfo:
    """头里描述的一个张量；``offsets`` 是相对数据区的 ``(start, end)``。"""

    name: str
    dtype: str
    shape: Tuple[int, ...]
    offsets: Tuple[int, int]

    @property
    def nbytes(self) -> int:
        return self.offsets[1] - self.offsets[0]

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= int(dim)
        return total

    def __repr__(self) -> str:
        dims = ", ".join(str(d) for d in self.shape)
        return f"TensorInfo({self.name!r}, {self.dtype}, ({dims}), {self.nbytes} B)"


def _is_int(value: object) -> bool:
    # bool 是 int 的子类；JSON 里的 true/false 不该被当成偏移量收下。
    return isinstance(value, int) and not isinstance(value, bool)


def _shape_of(spec: Mapping, name: str) -> Tuple[int, ...]:
    raw = spec.get("shape")
    if not isinstance(raw, list):
        raise SafeTensorsError(f"张量 {name!r} 的 shape 必须是列表，得到 {type(raw).__name__}")
    dims: List[int] = []
    for dim in raw:
        if not _is_int(dim) or dim < 0:
            raise SafeTensorsError(f"张量 {name!r} 的 shape 含非法维度 {dim!r}（必须是 >= 0 的整数）")
        dims.append(int(dim))
    return tuple(dims)


def _offsets_of(spec: Mapping, name: str, data_bytes: int, itemsize: int, numel: int) -> Tuple[int, int]:
    raw = spec.get("data_offsets")
    if not isinstance(raw, list) or len(raw) != 2 or not all(_is_int(v) for v in raw):
        raise SafeTensorsError(
            f"张量 {name!r} 的 data_offsets 必须是两个整数，得到 {raw!r}"
        )
    start, end = int(raw[0]), int(raw[1])
    if start < 0 or end < start:
        raise SafeTensorsError(f"张量 {name!r} 的 data_offsets 非法：[{start}, {end})")
    if end > data_bytes:
        raise SafeTensorsError(
            f"张量 {name!r} 的数据区间 [{start}, {end}) 超出数据区（只有 {data_bytes} 字节）"
        )
    want = numel * itemsize
    if end - start != want:
        raise SafeTensorsError(
            f"张量 {name!r} 的字节数与 shape/dtype 不符：区间 {end - start} B，"
            f"按 {numel} 个 {itemsize} 字节算应为 {want} B"
        )
    return start, end


class SafeTensorsFile:
    """一个 ``.safetensors`` 文件的只读视图，按需读取单个张量。

    用 ``with`` 或显式 ``close()`` 释放句柄；``tensor()`` 每次从文件重读，
    不会在内存里留一整份权重（大模型上这一条比"省一次 syscall"重要得多）。
    """

    def __init__(self, path) -> None:
        self.path = os.fspath(path)
        self._fh = open(self.path, "rb")
        try:
            self._parse()
        except BaseException:
            self._fh.close()
            raise

    # -- 解析 ---------------------------------------------------------

    def _parse(self) -> None:
        total = os.fstat(self._fh.fileno()).st_size
        if total < 8:
            raise SafeTensorsError(f"{self.path}：文件只有 {total} 字节，连 8 字节的头长都不够")

        prefix = self._fh.read(8)
        if len(prefix) != 8:
            raise SafeTensorsError(f"{self.path}：读取头长时文件被截断")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len > _MAX_HEADER_BYTES:
            raise SafeTensorsError(
                f"{self.path}：头部声称 {header_len} 字节，超过上限 {_MAX_HEADER_BYTES}，拒绝读取"
            )
        if 8 + header_len > total:
            raise SafeTensorsError(
                f"{self.path}：头长 {header_len} 超出文件大小（共 {total} 字节）"
            )

        raw = self._fh.read(header_len)
        if len(raw) != header_len:
            raise SafeTensorsError(f"{self.path}：读取头部时文件被截断")
        try:
            header = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SafeTensorsError(f"{self.path}：头部不是合法的 UTF-8 JSON（{exc}）") from exc
        if not isinstance(header, dict):
            raise SafeTensorsError(f"{self.path}：头部必须是 JSON 对象，得到 {type(header).__name__}")

        header = dict(header)
        meta = header.pop("__metadata__", {})
        if not isinstance(meta, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in meta.items()
        ):
            raise SafeTensorsError(f"{self.path}：__metadata__ 必须是字符串到字符串的映射")
        self.metadata: Dict[str, str] = dict(meta)

        self._data_start = 8 + header_len
        data_bytes = total - self._data_start
        infos: Dict[str, TensorInfo] = {}
        for name, spec in header.items():
            if not isinstance(name, str) or not name:
                raise SafeTensorsError(f"{self.path}：张量名必须是非空字符串，得到 {name!r}")
            if not isinstance(spec, dict):
                raise SafeTensorsError(f"{self.path}：张量 {name!r} 的描述必须是对象")
            dtype = spec.get("dtype")
            if not isinstance(dtype, str) or dtype not in _ITEMSIZE:
                raise SafeTensorsError(
                    f"{self.path}：张量 {name!r} 的 dtype {dtype!r} 未知"
                    f"（支持 {', '.join(sorted(_ITEMSIZE))}）"
                )
            shape = _shape_of(spec, name)
            numel = 1
            for dim in shape:
                numel *= dim
            offsets = _offsets_of(spec, name, data_bytes, _ITEMSIZE[dtype], numel)
            infos[name] = TensorInfo(name, dtype, shape, offsets)
        self._infos = infos

    # -- 访问 ---------------------------------------------------------

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(self._infos)

    def info(self, name: str) -> TensorInfo:
        try:
            return self._infos[name]
        except KeyError:
            raise KeyError(f"{self.path} 里没有张量 {name!r}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._infos

    def __len__(self) -> int:
        return len(self._infos)

    def __iter__(self):
        return iter(self._infos)

    def tensor(self, name: str) -> np.ndarray:
        """读出张量，返回**可写**的 numpy 数组。

        返回值总是拷贝：``np.frombuffer(bytes)`` 得到的是只读视图，而参数要被
        优化器原地更新；让它带着只读标志流进训练，错误会在第一步更新时才炸。
        bf16 在这里无损展宽成 f32（左移 16 位是精确的），因为 numpy 没有
        bf16 原生类型；fp8 直接拒绝（见模块说明）。
        """
        info = self.info(name)
        if info.dtype in _FP8:
            raise SafeTensorsError(
                f"张量 {name!r} 的 dtype 是 {info.dtype}：本仓库还没有 fp8 数值类型"
                "（见 DESIGN 5.301 的量化缺口），不能把它当成权重装进来"
            )
        size = info.nbytes
        self._fh.seek(self._data_start + info.offsets[0])
        raw = self._fh.read(size)
        if len(raw) != size:
            raise SafeTensorsError(f"{self.path}：读取张量 {name!r} 时文件被截断")

        if info.dtype == "BF16":
            bits = np.frombuffer(raw, dtype="<u2")
            widened = (bits.astype(np.uint32) << 16).view(np.float32)
            return widened.reshape(info.shape).copy()
        if info.dtype == "BOOL":
            # BOOL 与 U8 的字节宽度相同，但语义不同：直接按 u1 读出去会让调用方
            # 拿到"值为 0/1 的整数"，与 numpy 的布尔数组在比较 / 索引上行为不同。
            return np.frombuffer(raw, dtype=np.uint8).astype(np.bool_).reshape(info.shape)
        arr = np.frombuffer(raw, dtype=_NUMPY[info.dtype]).reshape(info.shape)
        return arr.copy()

    # -- 生命周期 -----------------------------------------------------

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "SafeTensorsFile":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"SafeTensorsFile({self.path!r}, {len(self._infos)} 个张量)"


def load_safetensors(path) -> Dict[str, np.ndarray]:
    """把一个文件里的所有张量读进字典（键是文件里的名字）。

    大模型请用 ``SafeTensorsFile`` 按需取：返回值是全量拷贝，权重几 GB 就是
    几 GB 内存。
    """
    with SafeTensorsFile(path) as fh:
        return {name: fh.tensor(name) for name in fh.names}


def load_safetensors_dir(folder, pattern: str = "*.safetensors") -> Dict[str, np.ndarray]:
    """合并一个目录下的分片文件（HF 的 ``model-0000x-of-0000y.safetensors``）。

    ``*.index.json`` 只是"哪个张量在哪个分片"的清单，不参与解析：按文件名
    排序全读一遍，结果相同。真正要防的是**同名的张量出现在两个分片里**——
    那是打包错了，静默取后者会让模型加载出一半的旧权重。
    """
    import glob

    root = os.fspath(folder)
    if not os.path.isdir(root):
        raise SafeTensorsError(f"{root} 不是目录")
    files = sorted(glob.glob(os.path.join(root, pattern)))
    if not files:
        raise SafeTensorsError(f"{root} 下没有匹配 {pattern} 的文件")

    merged: Dict[str, np.ndarray] = {}
    origin: Dict[str, str] = {}
    for path in files:
        with SafeTensorsFile(path) as fh:
            for name in fh.names:
                if name in merged:
                    raise SafeTensorsError(
                        f"张量 {name!r} 同时出现在 {origin[name]} 与 {path}：分片重复"
                    )
                merged[name] = fh.tensor(name)
                origin[name] = path
    return merged


def load_into_store(
    store,
    tensors: Mapping[str, np.ndarray],
    mapping: Optional[Mapping[str, str]] = None,
    transpose: Iterable[str] = (),
    strict: bool = True,
    allow_downcast: bool = False,
) -> Tuple[str, ...]:
    """把外部张量装进 ``ParameterStore``，返回**没有被用到**的张量名。

    参数名默认与张量名相同；``mapping`` 只写"名字对不上"的那些
    （``{"w_q": "model.layers.0.self_attn.q_proj.weight"}``）。``transpose``
    列出需要按 PyTorch ``nn.Linear`` 布局转一次的参数名（见模块说明）。

    顺序是刻意的：**先把所有问题收集齐再报**，且全部校验通过后才写。装填中途
    失败会让一半参数是新权重、一半是旧初始化——不报错、训练也能跑，但结果
    永远是错的；这正是本项目拒绝的静默降级。

    ``strict=True`` 时要求 store 的每个参数都被喂到；``allow_downcast=False``
    时拒绝有损的 dtype 转换（f32 参数收 f16 权重是安全的，反过来要显式同意）。
    """
    from .train.params import ParameterStore

    if not isinstance(store, ParameterStore):
        raise TypeError(f"store 必须是 ParameterStore，得到 {type(store).__name__}")

    shapes = store.shapes()
    names = store.names
    mapping = dict(mapping or {})
    transposed = set(transpose)

    unknown = sorted(set(mapping) - set(names))
    if unknown:
        raise SafeTensorsError(f"mapping 里有 store 不认识的参数名: {unknown}")
    if bad := sorted(transposed - set(names)):
        raise SafeTensorsError(f"transpose 里有 store 不认识的参数名: {bad}")
    for key, value in mapping.items():
        if not isinstance(value, str) or not value:
            raise SafeTensorsError(f"mapping[{key!r}] 必须是非空张量名，得到 {value!r}")

    problems: List[str] = []
    plan: Dict[str, Tuple[str, np.ndarray]] = {}
    for param in names:
        source = mapping.get(param, param)
        if source not in tensors:
            problems.append(f"参数 {param!r} 需要张量 {source!r}，但文件里没有")
            continue
        arr = tensors[source]
        if param in transposed:
            if arr.ndim != 2:
                problems.append(
                    f"参数 {param!r} 被标了 transpose，但张量 {source!r} 是 {arr.ndim} 维"
                )
                continue
            arr = arr.T
        want = shapes[param]
        if tuple(arr.shape) != want:
            note = "（已按 transpose 转置）" if param in transposed else ""
            problems.append(
                f"参数 {param!r} 期望形状 {want}，张量 {source!r} 给的是 {tuple(arr.shape)}{note}"
            )
            continue
        target = store[param].dtype
        if not np.can_cast(arr.dtype, target, casting="safe") and not allow_downcast:
            problems.append(
                f"参数 {param!r} 是 {target}，张量 {source!r} 是 {arr.dtype}："
                "这是有损转换，需要 allow_downcast=True"
            )
            continue
        plan[param] = (source, arr)

    if problems:
        raise SafeTensorsError("权重装载失败：\n  " + "\n  ".join(problems))

    used = set()
    for param, (source, arr) in plan.items():
        # 原地写：优化器持有的是同一个数组对象（train/checkpoint.py 的同一约定）。
        store.arrays()[param][...] = arr.astype(store[param].dtype, copy=False)
        used.add(source)
    return tuple(sorted(set(tensors) - used))
