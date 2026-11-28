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

from .runtime.fp8 import FP8_KINDS, round_fp8, unpack_fp8

__all__ = [
    "SafeTensorsError",
    "TensorInfo",
    "SafeTensorsFile",
    "QuantizedWeight",
    "FP8Weight",
    "load_safetensors",
    "load_safetensors_dir",
    "build_quantized_weight",
    "load_quantized_linear",
    "build_fp8_weight",
    "load_fp8_linear",
    "load_into_store",
]

# safetensors 官方实现对头的上限就是 100 MB。这里照着卡住：头是一个 JSON
# 字典，正常模型只有几十 KB；一个声称"头有 400 GB"的文件只可能是在骗读取方
# 先分配内存。
_MAX_HEADER_BYTES = 100 << 20

# dtype 名 → 每元素字节数。F8 两种格式是 1 字节位模式，读取时按 ``_FP8``
# 解码成 f32 载体。
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

# safetensors 的 fp8 dtype 名 → ``runtime/fp8`` 的格式名。numpy 没有 fp8 标量
# 类型：读取时解码成 f32 载体（fp8 -> f32 是精确的），与 bf16 的展宽同一策略。
_FP8: Dict[str, str] = {"F8_E4M3": "e4m3", "F8_E5M2": "e5m2"}


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
        bf16 原生类型；fp8 同理解码成 f32 载体（``unpack_fp8``，fp8 -> f32
        是精确的）。
        """
        info = self.info(name)
        size = info.nbytes
        self._fh.seek(self._data_start + info.offsets[0])
        raw = self._fh.read(size)
        if len(raw) != size:
            raise SafeTensorsError(f"{self.path}：读取张量 {name!r} 时文件被截断")

        if info.dtype in _FP8:
            bits = np.frombuffer(raw, dtype=np.uint8)
            return unpack_fp8(bits, _FP8[info.dtype]).reshape(info.shape).copy()
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


# -- 量化权重 -------------------------------------------------------------
#
# 量化权重比普通权重多两样东西（scale / zero_point），而"safetensors 只存张量"
# 这一条让它们没有任何强制约定：打包错了（非对称的包漏了零点、对称的包多带了
# 零点、逐组量化的结果按逐通道读）都能被读成合法数组，然后在推理里安静地给出
# 错误的数。这里补两层：``QuantizedWeight`` 在拿到数组时就把形状 / dtype /
# 值域钉死；metadata 里的自描述（``quantization`` / ``quantization.group_size``）
# 一旦存在就必须与张量一致，矛盾直接报错、不猜。
#
# 约定（写入方按此打包，读取方按此校验）：
#
# - ``q``：``(K, N)`` 的 int8，``K`` 是输入维、``N`` 是输出维，与 ``linear_i8``
#   / ``linear_i8_zp`` 的 ``w_i8`` 同布局。PyTorch 的 ``nn.Linear`` 是
#   ``(N, K)``，要显式 ``transpose=True``。
# - ``scale``：f32，``(1,)`` / ``(N,)``（逐输出通道）或 ``(K/G, N)``（逐组，
#   GPTQ / AWQ 的 group_size 形态），与 ``quantize_weight`` /
#   ``quantize_weight_asymmetric`` 的输出一致。
# - ``zero_point``：只在非对称时存在，int8，形状与 ``scale`` 相同。
# - metadata ``quantization``：``"int8"``（对称）或 ``"int8_zp"``（非对称）。
#   缺省时不校验，兼容别处导出的文件。
# - metadata ``quantization.group_size``：可缺省（形状已经自描述）；一旦写了，
#   必须等于由形状推出的组大小。

_QUANTIZATION_KINDS = ("int8", "int8_zp")


@dataclass(frozen=True)
class QuantizedWeight:
    """一份已量化的线性层权重。

    ``q`` 是 ``(K, N)`` 的 int8，``scale`` 与 ``zero_point`` 是 ``(1,)`` /
    ``(N,)``（逐输出通道）或 ``(K/G, N)``（逐组）。``zero_point is None``
    表示对称量化。

    这三个数组可能直接引用调用方传进来的数组（大权重不重复拷贝），
    当成只读用。
    """

    q: np.ndarray
    scale: np.ndarray
    zero_point: Optional[np.ndarray] = None

    @property
    def symmetric(self) -> bool:
        return self.zero_point is None

    @property
    def shape(self) -> Tuple[int, int]:
        return (int(self.q.shape[0]), int(self.q.shape[1]))

    @property
    def in_features(self) -> int:
        return self.shape[0]

    @property
    def out_features(self) -> int:
        return self.shape[1]

    @property
    def group_size(self) -> int:
        """沿输入维每组包含多少行。

        一维 scale 表示整列共用一份参数，等价于 ``group_size = K``；
        二维 scale 的组大小由 ``K / K_g`` 推出。
        """
        if self.scale.ndim == 2:
            return self.in_features // int(self.scale.shape[0])
        return self.in_features

    @property
    def per_group(self) -> bool:
        return self.group_size < self.in_features

    def dequantize(self) -> np.ndarray:
        """按 IR 的同一份语义还原成 f32（``dequantize_i8*`` 系列）。"""
        from .ir.ops import get_op  # 延迟导入：本模块的运行期依赖保持 numpy + 标准库

        if self.scale.ndim == 2:
            if self.zero_point is None:
                return get_op("dequantize_i8_group").reference([self.q, self.scale], {})
            return get_op("dequantize_i8_group_zp").reference(
                [self.q, self.scale, self.zero_point], {}
            )
        if self.zero_point is None:
            return get_op("dequantize_i8").reference([self.q, self.scale], {})
        return get_op("dequantize_i8_zp").reference(
            [self.q, self.scale, self.zero_point], {}
        )


def _check_quantized_arrays(
    q: np.ndarray,
    scale: np.ndarray,
    zero_point: Optional[np.ndarray],
    *,
    transpose: bool,
    allow_downcast: bool,
    label: str,
) -> QuantizedWeight:
    """把三个数组校验并规整成 :class:`QuantizedWeight`（唯一的构造路径）。"""
    q = np.asarray(q)
    if q.dtype != np.int8:
        raise SafeTensorsError(
            f"{label}的 q 必须是 int8，得到 {q.dtype}："
            "非 int8 的量化结果说明打包时漏了或错了一次转换，"
            "按值强转会静默回绕（例如 300 → 44）"
        )
    if q.ndim != 2:
        raise SafeTensorsError(f"{label}的 q 必须是二维 (K, N)，得到形状 {q.shape}")
    if transpose:
        # PyTorch nn.Linear 的 (out, in) → 本仓库的 (in, out)。
        q = np.ascontiguousarray(q.T)
    k, n = q.shape
    if k <= 0 or n <= 0:
        raise SafeTensorsError(f"{label}的 q 形状非法：{tuple(q.shape)}（两维都必须为正）")

    scale = np.asarray(scale)
    if scale.dtype.kind != "f":
        raise SafeTensorsError(f"{label}的 scale 必须是浮点，得到 {scale.dtype}")
    if scale.ndim > 2:
        raise SafeTensorsError(
            f"{label}的 scale 必须是一维 (1,) / (N,) 或二维 (K/G, N)，"
            f"得到形状 {scale.shape}"
        )
    if scale.ndim == 1:
        if scale.size not in (1, n):
            raise SafeTensorsError(
                f"{label}的 scale 有 {scale.size} 个元素，但 q 的输出维是 {n}："
                "逐通道量化只接受 1 个（逐张量）或 N 个（逐通道）"
            )
    else:
        groups, cols = scale.shape
        if cols != n:
            raise SafeTensorsError(
                f"{label}的 scale 形状是 {scale.shape}，列数与 q 的输出维 {n} 不一致"
            )
        if groups <= 0 or k % groups != 0:
            raise SafeTensorsError(
                f"{label}的 scale 有 {groups} 组，但 q 的输入维 K={k} 不能被它整除"
            )
    if not allow_downcast and not np.can_cast(scale.dtype, np.float32, casting="safe"):
        raise SafeTensorsError(
            f"{label}的 scale 是 {scale.dtype}，转成 f32 是有损转换，需要 allow_downcast=True"
        )
    scale = scale.astype(np.float32)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise SafeTensorsError(
            f"{label}的 scale 必须是有限且严格为正的数，得到 {scale.tolist()}"
        )

    zp: Optional[np.ndarray] = None
    if zero_point is not None:
        zp = np.asarray(zero_point)
        if zp.dtype.kind not in "iu":
            raise SafeTensorsError(
                f"{label}的 zero_point 必须是整数（int8 / int32 / int64），"
                f"得到 {zp.dtype}：浮点零点说明它是中间量，不是量化零点"
            )
        if zp.ndim > 2:
            raise SafeTensorsError(
                f"{label}的 zero_point 必须是一维 (1,) / (N,) 或二维 (K/G, N)，"
                f"得到形状 {zp.shape}"
            )
        if scale.ndim == 2:
            if tuple(zp.shape) != tuple(scale.shape):
                raise SafeTensorsError(
                    f"{label}的 zero_point 形状 {zp.shape} 必须与 scale "
                    f"{scale.shape} 相同：同一组共用一对参数"
                )
        elif zp.ndim != 1 or zp.size not in (1, n):
            raise SafeTensorsError(
                f"{label}的 zero_point 形状 {zp.shape} 与逐通道 scale 不匹配"
                "（一维时必须是 (1,) 或 (N,)）"
            )
        lo, hi = int(zp.min()), int(zp.max())
        if lo < -128 or hi > 127:
            raise SafeTensorsError(
                f"{label}的 zero_point 越界：必须在 [-128, 127]，得到 [{lo}, {hi}]"
            )
        zp = zp.astype(np.int8)

    return QuantizedWeight(q=q, scale=scale, zero_point=zp)


def build_quantized_weight(
    tensors: Mapping[str, np.ndarray],
    weight: str,
    scale: str,
    zero_point: Optional[str] = None,
    *,
    transpose: bool = False,
    allow_downcast: bool = False,
) -> QuantizedWeight:
    """从"名字 → 数组"的表里组装量化权重（显式名字，不猜）。

    ``weight`` / ``scale`` / ``zero_point`` 是张量名。``zero_point`` 给了名字
    就按非对称校验，没给就是对称——不存在"看着像有零点就用"的推断。
    """
    needed = [weight, scale] + ([zero_point] if zero_point is not None else [])
    missing = [name for name in needed if name not in tensors]
    if missing:
        raise SafeTensorsError(f"量化权重缺少张量: {missing}")
    return _check_quantized_arrays(
        tensors[weight],
        tensors[scale],
        None if zero_point is None else tensors[zero_point],
        transpose=transpose,
        allow_downcast=allow_downcast,
        label=f"张量 {weight!r} ",
    )


def _check_group_size_metadata(
    metadata: Mapping[str, str], group_size: int, where: str
) -> None:
    """metadata 的 ``quantization.group_size`` 若存在，必须与形状推出的组大小一致。"""
    group = metadata.get("quantization.group_size")
    if group is None:
        return
    text = group.strip()
    try:
        size = int(text)
    except ValueError:
        raise SafeTensorsError(
            f"{where}：metadata 的 quantization.group_size={group!r} 不是正整数"
        ) from None
    if size <= 0:
        raise SafeTensorsError(
            f"{where}：metadata 的 quantization.group_size={group!r} 必须为正"
        )
    if size != group_size:
        raise SafeTensorsError(
            f"{where}：metadata 的 quantization.group_size={group!r} 与张量矛盾"
            f"（q 的 K 与 scale 推出的每组行数是 {group_size}）"
        )


def _check_quantization_metadata(
    metadata: Mapping[str, str], weight: QuantizedWeight, where: str
) -> None:
    """校验 metadata 的量化自描述与张量是否自洽（缺失即不校验）。"""
    kind = metadata.get("quantization")
    if kind is not None:
        if kind not in _QUANTIZATION_KINDS:
            raise SafeTensorsError(
                f"{where}：metadata 里的 quantization={kind!r} 未知"
                f"（支持 {', '.join(_QUANTIZATION_KINDS)}）"
            )
        if kind == "int8_zp" and weight.symmetric:
            raise SafeTensorsError(
                f"{where}：metadata 声称 quantization=int8_zp，但没有给 zero_point 张量"
            )
        if kind == "int8" and not weight.symmetric:
            raise SafeTensorsError(
                f"{where}：metadata 声称 quantization=int8，却给了 zero_point 张量"
            )

    _check_group_size_metadata(metadata, weight.group_size, where)


def load_quantized_linear(
    path,
    weight: str,
    scale: str,
    zero_point: Optional[str] = None,
    *,
    transpose: bool = False,
    allow_downcast: bool = False,
    check_metadata: bool = True,
) -> QuantizedWeight:
    """从 ``.safetensors`` 读一份量化线性层权重。

    与 :func:`load_into_store` 同一条原则：名字与布局都显式给，先校验再返回。
    ``check_metadata=False`` 只跳过 metadata 自洽性检查（有些导出器写错
    ``quantization`` 但张量本身是对的），张量层面的校验无法跳过。
    """
    root = os.fspath(path)
    with SafeTensorsFile(root) as fh:
        needed = [weight, scale] + ([zero_point] if zero_point is not None else [])
        missing = [name for name in needed if name not in fh]
        if missing:
            raise SafeTensorsError(f"{root} 里缺少量化权重需要的张量: {missing}")
        arrays = {name: fh.tensor(name) for name in needed}
        metadata = dict(fh.metadata)
    got = _check_quantized_arrays(
        arrays[weight],
        arrays[scale],
        None if zero_point is None else arrays[zero_point],
        transpose=transpose,
        allow_downcast=allow_downcast,
        label=f"{root} 的量化权重",
    )
    if check_metadata:
        _check_quantization_metadata(metadata, got, root)
    return got


# -- fp8 量化权重 ---------------------------------------------------------
#
# fp8 的容器与 int8 的差别只有两处：载荷是 f32 载体（`DType.F8E4M3` 在
# numpy 里没有标量类型，5.319 读文件时就解码成 f32），量化参数只有 scale
# （fp8 的 0 精确可表示，非对称没有收益）。因此这里额外守一条 int8 不需要
# 的规则：**载荷的每个值必须落在该格式的格点上**。
#
# 为什么这条必须查：`w` 的 dtype 是 f32，一份未经量化的 f32 权重与一份
# 量化过的 f8 载体在字节层面完全同型。不查，就会把"忘了量化"读成"量化
# 精度很高"，推理照跑、数值只差一点点——这类错误没有任何症状。
#
# 约定（与 HF 的 fp8 检查点对齐）：
#
# - 张量名：权重 `weight`、权重 scale `weight_scale`、可选的静态激活 scale
#   `input_scale`；名字由调用方显式给，不猜。
# - `w`：`(K, N)` 的 f32 载体，值落在 `kind` 的格点上，量级在缩放后的
#   `±max` 内。PyTorch / safetensors 的 `(N, K)` 布局要显式 `transpose=True`。
# - `weight_scale`：f32，`(1,)`（整张量一个，HF 最常见）/ `(N,)`（逐输出
#   通道）或 `(K/G, N)`（逐组）。
# - `input_scale`：可选，f32 标量。静态激活 scale 用它；动态 per-token
#   量化没有这个张量，缺省即不校验。
# - metadata `quantization`：缺省时不校验；写了就必须是 `"fp8"`（格式由
#   张量 dtype 或显式 kind 决定）、`"fp8_e4m3"` 或 `"fp8_e5m2"`，且与 kind
#   不矛盾。

#: metadata 里的 fp8 自描述 → 格式；``None`` 表示只声明"是 fp8"，格式另定。
_FP8_QUANTIZATION: Dict[str, Optional[str]] = {
    "fp8": None,
    "fp8_e4m3": "e4m3",
    "fp8_e5m2": "e5m2",
}


@dataclass(frozen=True)
class FP8Weight:
    """一份 fp8 量化线性层权重。

    ``w`` 是 ``(K, N)`` 的 f32 载体（值全部落在 ``kind`` 的格点、缩放后的
    量级），``weight_scale`` 是 ``(1,)`` / ``(N,)`` / ``(K/G, N)``。
    ``input_scale`` 是可选的静态激活 scale（标量），动态 per-token 场景为
    ``None``。三个数组可能直接引用调用方传入的数组，当成只读用。
    """

    w: np.ndarray
    weight_scale: np.ndarray
    kind: str = "e4m3"
    input_scale: Optional[np.ndarray] = None

    @property
    def shape(self) -> Tuple[int, int]:
        return (int(self.w.shape[0]), int(self.w.shape[1]))

    @property
    def in_features(self) -> int:
        return self.shape[0]

    @property
    def out_features(self) -> int:
        return self.shape[1]

    @property
    def group_size(self) -> int:
        """沿输入维每组包含多少行；一维 scale 等价于 ``group_size = K``。"""
        if self.weight_scale.ndim == 2:
            return self.in_features // int(self.weight_scale.shape[0])
        return self.in_features

    @property
    def per_group(self) -> bool:
        return self.group_size < self.in_features

    def dequantize(self) -> np.ndarray:
        """还原成 f32 的真实数值：``w * weight_scale``（逐组时沿 K 重复）。"""
        scale = self.weight_scale
        if scale.ndim == 2:
            scale = np.repeat(scale, self.in_features // int(scale.shape[0]), axis=0)
        return (self.w * scale.astype(np.float32)).astype(np.float32)


def _check_fp8_scale(
    scale: np.ndarray,
    q: np.ndarray,
    *,
    allow_downcast: bool,
    label: str,
    what: str,
) -> np.ndarray:
    """校验 weight scale：``(1,)`` / ``(N,)`` / ``(K/G, N)``，有限且严格为正。"""
    s = np.asarray(scale)
    if s.dtype.kind != "f":
        raise SafeTensorsError(f"{label}的 {what} 必须是浮点，得到 {s.dtype}")
    if s.ndim > 2:
        raise SafeTensorsError(
            f"{label}的 {what} 必须是一维 (1,) / (N,) 或二维 (K/G, N)，"
            f"得到形状 {s.shape}"
        )
    k, n = q.shape
    if s.ndim == 1:
        if s.size not in (1, n):
            raise SafeTensorsError(
                f"{label}的 {what} 有 {s.size} 个元素，但 q 的输出维是 {n}："
                "逐通道量化只接受 1 个（逐张量）或 N 个（逐通道）"
            )
    else:
        groups, cols = s.shape
        if cols != n:
            raise SafeTensorsError(
                f"{label}的 {what} 形状是 {s.shape}，列数与 q 的输出维 {n} 不一致"
            )
        if groups <= 0 or k % groups != 0:
            raise SafeTensorsError(
                f"{label}的 {what} 有 {groups} 组，但 q 的输入维 K={k} 不能被它整除"
            )
    if not allow_downcast and not np.can_cast(s.dtype, np.float32, casting="safe"):
        raise SafeTensorsError(
            f"{label}的 {what} 是 {s.dtype}，转成 f32 是有损转换，需要 allow_downcast=True"
        )
    s = s.astype(np.float32)
    if not np.all(np.isfinite(s)) or np.any(s <= 0.0):
        raise SafeTensorsError(
            f"{label}的 {what} 必须是有限且严格为正的数，得到 {s.tolist()}"
        )
    return s


def _check_fp8_arrays(
    w: np.ndarray,
    weight_scale: np.ndarray,
    kind: str,
    input_scale: Optional[np.ndarray],
    *,
    transpose: bool,
    allow_downcast: bool,
    label: str,
) -> FP8Weight:
    """校验并规整成 :class:`FP8Weight`（唯一的构造路径）。"""
    if kind not in FP8_KINDS:
        raise SafeTensorsError(f"未知的 fp8 格式 {kind!r}；支持 {FP8_KINDS}")

    w = np.asarray(w)
    if w.dtype.kind != "f":
        raise SafeTensorsError(
            f"{label}的 w 必须是浮点载体（fp8 位模式请先用 unpack_fp8 解码），"
            f"得到 {w.dtype}"
        )
    if not allow_downcast and not np.can_cast(w.dtype, np.float32, casting="safe"):
        raise SafeTensorsError(
            f"{label}的 w 是 {w.dtype}，转成 f32 是有损转换，需要 allow_downcast=True"
        )
    w = np.ascontiguousarray(w.astype(np.float32))
    if w.ndim != 2:
        raise SafeTensorsError(f"{label}的 w 必须是二维 (K, N)，得到形状 {w.shape}")
    if transpose:
        # PyTorch nn.Linear 的 (out, in) → 本仓库的 (in, out)。
        w = np.ascontiguousarray(w.T)
    k, n = w.shape
    if k <= 0 or n <= 0:
        raise SafeTensorsError(f"{label}的 w 形状非法：{tuple(w.shape)}（两维都必须为正）")
    if not np.all(np.isfinite(w)):
        raise SafeTensorsError(f"{label}的 w 含 NaN / Inf：量化权重必须全部有限")
    if not np.array_equal(round_fp8(w, kind), w, equal_nan=True):
        raise SafeTensorsError(
            f"{label}的 w 有值不在 {kind} 的格点上：这份数组不是 fp8 量化结果，"
            "而是原始 f32 权重（或用了另一种格式）。请先用 quantize_weight_fp8 量化，"
            "或确认 kind 与实际格式一致"
        )

    scale = _check_fp8_scale(
        weight_scale, w, allow_downcast=allow_downcast, label=label, what="weight_scale"
    )

    iscale: Optional[np.ndarray] = None
    if input_scale is not None:
        raw = np.asarray(input_scale)
        if raw.dtype.kind != "f":
            raise SafeTensorsError(f"{label}的 input_scale 必须是浮点，得到 {raw.dtype}")
        if raw.ndim > 1 or raw.size != 1:
            raise SafeTensorsError(
                f"{label}的 input_scale 必须是标量，得到形状 {raw.shape}："
                "动态 per-token 量化没有这个张量，静态场景也只有一个全局 scale"
            )
        if not allow_downcast and not np.can_cast(
            raw.dtype, np.float32, casting="safe"
        ):
            raise SafeTensorsError(
                f"{label}的 input_scale 是 {raw.dtype}，转成 f32 是有损转换，"
                "需要 allow_downcast=True"
            )
        iscale = raw.astype(np.float32).reshape(1)
        if not np.all(np.isfinite(iscale)) or np.any(iscale <= 0.0):
            raise SafeTensorsError(
                f"{label}的 input_scale 必须是有限且严格为正的数，得到 {iscale.tolist()}"
            )

    return FP8Weight(w=w, weight_scale=scale, kind=kind, input_scale=iscale)


def _check_fp8_metadata(
    metadata: Mapping[str, str], weight: FP8Weight, where: str
) -> None:
    """校验 metadata 的 fp8 自描述（缺失即不校验），矛盾直接报错。"""
    declared = metadata.get("quantization")
    if declared is not None:
        if declared not in _FP8_QUANTIZATION:
            raise SafeTensorsError(
                f"{where}：metadata 里的 quantization={declared!r} 不是 fp8 的写法"
                f"（支持 {', '.join(_FP8_QUANTIZATION)}）"
            )
        expected = _FP8_QUANTIZATION[declared]
        if expected is not None and expected != weight.kind:
            raise SafeTensorsError(
                f"{where}：metadata 声称 quantization={declared}，"
                f"但载荷按 {weight.kind} 校验"
            )
    _check_group_size_metadata(metadata, weight.group_size, where)


def build_fp8_weight(
    tensors: Mapping[str, np.ndarray],
    weight: str,
    weight_scale: str,
    input_scale: Optional[str] = None,
    *,
    kind: str = "e4m3",
    transpose: bool = False,
    allow_downcast: bool = False,
) -> FP8Weight:
    """从"名字 → 数组"的表里组装 fp8 量化权重（显式名字，不猜格式）。

    ``kind`` 必须显式给：数组是 f32 载体，格式信息不在 dtype 里。
    """
    needed = [weight, weight_scale] + ([input_scale] if input_scale is not None else [])
    missing = [name for name in needed if name not in tensors]
    if missing:
        raise SafeTensorsError(f"fp8 量化权重缺少张量: {missing}")
    return _check_fp8_arrays(
        tensors[weight],
        tensors[weight_scale],
        kind,
        None if input_scale is None else tensors[input_scale],
        transpose=transpose,
        allow_downcast=allow_downcast,
        label=f"张量 {weight!r} ",
    )


def load_fp8_linear(
    path,
    weight: str,
    weight_scale: str,
    input_scale: Optional[str] = None,
    *,
    kind: Optional[str] = None,
    transpose: bool = False,
    allow_downcast: bool = False,
    check_metadata: bool = True,
) -> FP8Weight:
    """从 ``.safetensors`` 读一份 fp8 量化线性层权重。

    格式的优先级是**文件说了算**：张量 dtype 是 ``F8_E4M3`` / ``F8_E5M2``
    时 ``kind`` 可缺省，显式给了就必须一致；权重存成 ``F32``（已解码的载体）
    时必须显式给 ``kind``。这与"名字与布局都显式给、不猜"是同一条原则——
    格式能由文件唯一确定时就不该再让调用方重复一遍。
    """
    root = os.fspath(path)
    with SafeTensorsFile(root) as fh:
        needed = [weight, weight_scale] + (
            [input_scale] if input_scale is not None else []
        )
        missing = [name for name in needed if name not in fh]
        if missing:
            raise SafeTensorsError(f"{root} 里缺少 fp8 量化权重需要的张量: {missing}")
        file_kind = _FP8.get(fh.info(weight).dtype)
        if kind is None:
            if file_kind is None:
                raise SafeTensorsError(
                    f"{root}：张量 {weight!r} 的 dtype 是 {fh.info(weight).dtype}，"
                    "推不出 fp8 格式；请显式给 kind"
                )
            resolved = file_kind
        else:
            if kind not in FP8_KINDS:
                raise SafeTensorsError(f"未知的 fp8 格式 {kind!r}；支持 {FP8_KINDS}")
            if file_kind is not None and file_kind != kind:
                raise SafeTensorsError(
                    f"{root}：张量 {weight!r} 的 dtype 是 {fh.info(weight).dtype}"
                    f"（{file_kind}），与 kind={kind} 矛盾"
                )
            resolved = kind
        arrays = {name: fh.tensor(name) for name in needed}
        metadata = dict(fh.metadata)
    got = _check_fp8_arrays(
        arrays[weight],
        arrays[weight_scale],
        resolved,
        None if input_scale is None else arrays[input_scale],
        transpose=transpose,
        allow_downcast=allow_downcast,
        label=f"{root} 的 fp8 权重",
    )
    if check_metadata:
        _check_fp8_metadata(metadata, got, root)
    return got


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
