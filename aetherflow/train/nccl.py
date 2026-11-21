"""NCCL 集合通信：跨进程的 AllReduce / ReduceScatter / AllGather / Broadcast。

``train/parallel.py`` 的数据并行是**单进程内**依次切卡：它能把一批数据摊到
多块卡上，但归约仍是"每卡把梯度拉回主机、主机求和、再传回去"。5.221 把
这件事写成能力声明（``device_allreduce=False``，原因点名 NCCL 未接入），
本模块补上设备侧那一半。

**为什么必须多进程。** NCCL 的一个 communicator 对应"一个 rank 一张卡"，
``ncclCommInitRank`` 会阻塞到全部 rank 都进来为止。因此同一次训练里"每块卡
一个进程"是它的使用前提，本模块不提供"单进程内多 comm"的包装：那种做法要
靠线程同时进入初始化，还会把 cupy 的"当前设备"变成线程局部状态，失败时
很难判断是哪一层没设对。单进程多卡的路径仍走 5.190 的主机中转。

**同一份绑定也用于 RCCL。** AMD 的 RCCL 实现了同一套 ``nccl*`` C 接口
（函数名、枚举值、``ncclUniqueId`` 的 128 字节布局都一致），所以这里不写第二份
ctypes 声明，只在加载时按库文件名判定厂商：命中 ``librccl`` 就是 HIP，其余是
CUDA。两者的差异只在两处——库的 soname（NCCL 是 ``libnccl.so.2``，RCCL 是
``librccl.so.1``）与"怎么同步当前流"（CUDA 用 cupy，HIP 用项目自己的
``backends.hip.runtime``）。**判定按文件名而不是按"能不能 import cupy"**：
设备上装了什么库与 Python 侧装了哪个包是两件独立的事。

**数组是外部输入，先在主机侧校验完再碰驱动。** NCCL 只认"连续设备缓冲 +
元素个数 + dtype 枚举"。dtype 不在登记表、数组不是 C 连续、收发缓冲长度
与 rank 数对不上，都在调用之前抛错——设备端越界读的代价是上下文中毒，
而这条路径上很难从结果里分辨"算错了"和"读错了"。

**数值不是逐位保证。** 归约顺序由 NCCL 的拓扑与算法决定（ring / tree），
浮点加法不满足结合律，所以这里只能承诺"与主机求和在容差内一致"。用例里
的判据因此是 ``1e-5``（f32），而不是 ``assert_array_equal``。

**覆盖范围。** dtype 只登记 f16 / f32 / f64（IR 支持、numpy 也有对应类型）；
bf16 在 NCCL 里有自己的枚举，但 numpy 没有 bf16，跨进程的"主机侧期望值"
算不出来，不放开。归约只登记 sum / max / min / prod——``ncclAvg`` 需要
2.10 以上，且它只是 sum 再除，调用方自己除更可控。
"""

from __future__ import annotations

import ctypes
import ctypes.util
from typing import Any, Optional, Tuple

import numpy as np

__all__ = [
    "NcclUnavailable",
    "NcclUnsupported",
    "available",
    "vendor",
    "version",
    "version_string",
    "unique_id",
    "NcclComm",
]

# ncclDataType_t
_DATA_TYPES = {
    np.dtype("float16"): 6,
    np.dtype("float32"): 7,
    np.dtype("float64"): 8,
}
# ncclRedOp_t
_RED_OPS = {"sum": 0, "prod": 1, "max": 2, "min": 3}
_NCCL_SUCCESS = 0

_CANDIDATES = (
    "libnccl.so.2", "libnccl.so", "nccl.dll",
    "librccl.so.2", "librccl.so.1", "librccl.so", "rccl.dll",
)

_loaded: Optional[Tuple[Any, str, str]] = None
_load_error: Optional[str] = None


class NcclUnavailable(RuntimeError):
    """系统里没有可用的 NCCL（或版本太旧，缺 ReduceScatter 等接口）。"""


class NcclUnsupported(ValueError):
    """NCCL 在，但这个 dtype / 归约 / 缓冲长度不在本模块的覆盖范围内。"""


class NcclUniqueId(ctypes.Structure):
    """``ncclUniqueId`` 是 128 字节的不透明结构，按值在进程间传递。"""

    _fields_ = [("internal", ctypes.c_char * 128)]

    def to_bytes(self) -> bytes:
        # 不能用 bytes(self.internal)：c_char 数组会按第一个 0 字节截断，
        # 而 unique id 里出现 0 是常态。
        return ctypes.string_at(ctypes.addressof(self), 128)

    @classmethod
    def from_bytes(cls, data: bytes) -> "NcclUniqueId":
        if len(data) != 128:
            raise ValueError(f"ncclUniqueId 必须是 128 字节，收到 {len(data)}")
        value = cls()
        ctypes.memmove(ctypes.addressof(value), data, 128)
        return value


def data_type(dtype: Any) -> int:
    """numpy / 容器 dtype → ``ncclDataType_t``；未登记的类型显式拒绝。"""
    try:
        key = np.dtype(dtype)
    except TypeError:
        raise NcclUnsupported(
            f"NCCL 路径只登记了 {sorted(str(k) for k in _DATA_TYPES)}，"
            f"收到 {dtype!r}；bf16 需要单独的期望值算法，当前不放开"
        ) from None
    try:
        return _DATA_TYPES[key]
    except KeyError:
        raise NcclUnsupported(
            f"NCCL 路径只登记了 {sorted(str(k) for k in _DATA_TYPES)}，"
            f"收到 {key}；bf16 需要单独的期望值算法，当前不放开"
        ) from None


def red_op(name: Any) -> int:
    """归约名 → ``ncclRedOp_t``。"""
    try:
        return _RED_OPS[name]
    except (KeyError, TypeError):
        raise NcclUnsupported(
            f"归约只登记了 {sorted(_RED_OPS)}，收到 {name!r}；"
            "ncclAvg 需要 NCCL 2.10 以上，且等价于 sum 再除，调用方自己除更可控"
        ) from None


def _vendor_from_path(path: str) -> str:
    """按库文件名判定厂商：``librccl`` / ``rccl.dll`` 是 HIP，其余是 CUDA。

    只看 basename 且转小写：``ctypes.util.find_library`` 可能返回绝对路径，
    Windows 上还会带盘符。这条路必须能在没有设备的机器上回答，所以是纯函数。
    """
    name = str(path).replace("\\", "/").rsplit("/", 1)[-1].lower()
    return "hip" if "rccl" in name else "cuda"


def _load() -> Tuple[Any, str, str]:
    """加载 NCCL / RCCL；失败抛 ``NcclUnavailable``（结果缓存，只试一次）。"""
    global _loaded, _load_error
    if _loaded is not None:
        return _loaded
    if _load_error is not None:
        raise NcclUnavailable(_load_error)

    names = list(_CANDIDATES)
    # ``find_library`` 的结果放最前，但保持"先 NCCL 后 RCCL"的优先级：
    # 一台机器同时装了两套库时，按候选顺序的第一份当作事实来源。
    found = [
        name
        for name in (
            ctypes.util.find_library("nccl"),
            ctypes.util.find_library("rccl"),
        )
        if name
    ]
    names[0:0] = found
    try:  # pip install nvidia-nccl-cu12 的包内 so 不在 ldconfig 路径里
        import nvidia.nccl as _pkg

        for root in list(getattr(_pkg, "__path__", []) or []):
            names.append(root + "/lib/libnccl.so.2")
    except ImportError:
        pass

    tried = []
    for name in names:
        try:
            lib = ctypes.CDLL(name)
        except OSError as exc:
            tried.append(f"{name}: {exc}")
            continue
        _loaded = (lib, name, _vendor_from_path(name))
        _bind(lib)
        return _loaded
    _load_error = (
        "找不到 NCCL / RCCL 动态库，尝试过：\n  " + "\n  ".join(tried)
    )
    raise NcclUnavailable(_load_error)


def _bind(lib: Any) -> None:
    """声明用到的接口签名。缺接口时由 ``NcclComm`` 在构造期显式报错。"""
    vp = ctypes.c_void_p
    lib.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.ncclGetVersion.restype = ctypes.c_int
    lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(NcclUniqueId)]
    lib.ncclGetUniqueId.restype = ctypes.c_int
    lib.ncclGetErrorString.argtypes = [ctypes.c_int]
    lib.ncclGetErrorString.restype = ctypes.c_char_p
    lib.ncclCommInitRank.argtypes = [
        ctypes.POINTER(vp), ctypes.c_int, NcclUniqueId, ctypes.c_int,
    ]
    lib.ncclCommInitRank.restype = ctypes.c_int
    if hasattr(lib, "ncclCommInitAll"):
        # 单进程多卡：一次把 world 个 communicator 建好，不需要 unique id 分发。
        lib.ncclCommInitAll.argtypes = [
            ctypes.POINTER(vp), ctypes.c_int, ctypes.POINTER(ctypes.c_int),
        ]
        lib.ncclCommInitAll.restype = ctypes.c_int
    lib.ncclCommDestroy.argtypes = [vp]
    lib.ncclCommDestroy.restype = ctypes.c_int
    if hasattr(lib, "ncclCommCount"):
        lib.ncclCommCount.argtypes = [vp, ctypes.POINTER(ctypes.c_int)]
        lib.ncclCommCount.restype = ctypes.c_int
    if hasattr(lib, "ncclCommUserRank"):
        lib.ncclCommUserRank.argtypes = [vp, ctypes.POINTER(ctypes.c_int)]
        lib.ncclCommUserRank.restype = ctypes.c_int
    if hasattr(lib, "ncclCommGetAsyncError"):
        lib.ncclCommGetAsyncError.argtypes = [
            vp, ctypes.POINTER(ctypes.c_int),
        ]
        lib.ncclCommGetAsyncError.restype = ctypes.c_int
    # AllReduce / ReduceScatter 带归约枚举；AllGather 没有归约参数。
    # 把两者绑成同一个形参表会在调用时直接 TypeError——真机上就是这样暴露的。
    for name in ("ncclAllReduce", "ncclReduceScatter"):
        getattr(lib, name).argtypes = [
            vp, vp, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, vp, vp,
        ]
        getattr(lib, name).restype = ctypes.c_int
    lib.ncclAllGather.argtypes = [
        vp, vp, ctypes.c_size_t, ctypes.c_int, vp, vp,
    ]
    lib.ncclAllGather.restype = ctypes.c_int
    lib.ncclBroadcast.argtypes = [
        vp, vp, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, vp, vp,
    ]
    lib.ncclBroadcast.restype = ctypes.c_int
    lib.ncclGroupStart.argtypes = []
    lib.ncclGroupStart.restype = ctypes.c_int
    lib.ncclGroupEnd.argtypes = []
    lib.ncclGroupEnd.restype = ctypes.c_int


def available() -> bool:
    """只回答"能不能加载"，不建 communicator，也不碰 GPU。"""
    try:
        _load()
    except NcclUnavailable:
        return False
    return True


def vendor() -> str:
    """``"cuda"``（NCCL）或 ``"hip"``（RCCL）；库缺失时抛 ``NcclUnavailable``。"""
    _, _, kind = _load()
    return kind


def _format_version(number: int) -> str:
    """``ncclGetVersion`` 的整数 → ``"2.27.7"``（major×10⁴ + minor×10² + patch）。"""
    return f"{number // 10000}.{number // 100 % 100}.{number % 100}"


def version() -> int:
    """NCCL / RCCL 版本号（如 22707 表示 2.27.7）；库缺失时抛 ``NcclUnavailable``。"""
    lib, _, _ = _load()
    number = ctypes.c_int()
    _check(lib.ncclGetVersion(ctypes.byref(number)), "ncclGetVersion")
    return int(number.value)


def version_string() -> str:
    """``version()`` 的可读形式；库缺失时抛 ``NcclUnavailable``。"""
    return _format_version(version())


def unique_id() -> NcclUniqueId:
    """生成一次性的 communicator id；由 rank 0 生成并分发给其余 rank。"""
    lib, _, _ = _load()
    value = NcclUniqueId()
    _check(lib.ncclGetUniqueId(ctypes.byref(value)), "ncclGetUniqueId")
    return value


def init_all(devices: Any) -> list:
    """单进程多卡的 communicator：``ncclCommInitAll``，返回值与 ``devices`` 同序。

    与 :func:`unique_id` + :class:`NcclComm` 的区别是**不需要跨进程分发 id**，
    适合 ``DataParallelTrainer`` 这种"一个进程切多张卡"的形态。每个 rank 一个
    进程的 DDP 仍用 ``ncclCommInitRank``。调用方负责在发起集合操作前把当前
    设备切到对应卡（``DataParallelTrainer`` 的 provider 已经这么做）。
    """
    lib, name, vendor = _load()
    if not hasattr(lib, "ncclCommInitAll"):
        raise NcclUnavailable(f"{name} 缺少 ncclCommInitAll")
    wanted = [int(device) for device in devices]
    handles = (ctypes.c_void_p * len(wanted))()
    devlist = (ctypes.c_int * len(wanted))(*wanted)
    _check(lib.ncclCommInitAll(handles, len(wanted), devlist), "ncclCommInitAll")
    return [
        NcclComm._from_handle(lib, name, vendor, handles[i], i, len(wanted))
        for i in range(len(wanted))
    ]


def _label() -> str:
    """错误信息里的库名：加载失败时两套都提，加载成功时按厂商只提一个。"""
    try:
        _, _, kind = _load()
    except NcclUnavailable:
        return "NCCL / RCCL"
    return "RCCL" if kind == "hip" else "NCCL"


def _message(status: int) -> str:
    try:
        lib, _, _ = _load()
        text = lib.ncclGetErrorString(status)
        if text:
            return text.decode("utf-8", "replace")
    except NcclUnavailable:
        pass
    return "未知 ncclResult_t"


def _check(status: int, what: str) -> None:
    if status != _NCCL_SUCCESS:
        raise RuntimeError(f"{_label()} {what} 失败（{status}）：{_message(status)}")


def _pointer(array: Any, label: str) -> ctypes.c_void_p:
    """取设备缓冲地址；不是设备数组就当场拒绝。"""
    data = getattr(array, "data", None)
    ptr = getattr(data, "ptr", None)
    if ptr is None:
        raise TypeError(
            f"{label} 必须是设备数组（需要有 .data.ptr），"
            f"得到 {type(array).__name__}；主机数组请先 to_array"
        )
    return ctypes.c_void_p(int(ptr))


def _contiguous(array: Any, label: str) -> None:
    """NCCL 按线性缓冲读，非连续数组会读错。"""
    flags = getattr(array, "flags", None)
    if flags is not None and not getattr(flags, "c_contiguous", True):
        raise TypeError(f"{label} 必须是 C 连续数组；先做 ascontiguousarray")


class NcclComm:
    """一个 rank 上的 communicator：持有 handle，提供四个集合操作。

    构造即调用 ``ncclCommInitRank``，**会阻塞到全部 rank 都进来**。调用前
    必须把"当前设备"切到本 rank 对应的卡（cupy 用 ``cp.cuda.Device(rank)``，
    HIP 用 ``hip.runtime`` 的同名接口）：NCCL 的 rank→device 绑定发生在这里，
    设错卡会在第一次集合调用时才以超时或非法地址的形式暴露。

    ``stream`` 为 ``None`` 表示 CUDA 默认流（0）。所有集合调用都是**异步**的：
    结果要在流同步之后才可读，``synchronize()`` 给出的就是这条语义（依赖
    cupy；没有 cupy 时请调用方自己同步）。
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        uid: Any,
        *,
        stream: Optional[int] = None,
    ) -> None:
        if isinstance(world_size, bool) or not isinstance(world_size, int):
            raise TypeError(f"world_size 必须是 int，收到 {world_size!r}")
        if world_size < 1:
            raise ValueError(f"world_size 必须 >= 1，收到 {world_size}")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError(f"rank 必须是 int，收到 {rank!r}")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} 越界：world_size={world_size}")

        lib, self.name, self.vendor = _load()
        required = (
            "ncclCommInitRank",
            "ncclAllReduce",
            "ncclAllGather",
            "ncclBroadcast",
            "ncclGroupStart",
            "ncclGroupEnd",
        )
        missing = [name for name in required if not hasattr(lib, name)]
        if missing:
            raise NcclUnavailable(f"{self.name} 缺少接口 {missing}")

        if isinstance(uid, NcclUniqueId):
            value = uid
        else:
            value = NcclUniqueId.from_bytes(bytes(uid))

        self.lib = lib
        self.rank = rank
        self.world_size = world_size
        self._stream = None if stream is None else int(stream)
        self.handle = ctypes.c_void_p()
        self._destroyed = False
        _check(
            lib.ncclCommInitRank(
                ctypes.byref(self.handle), world_size, value, rank
            ),
            "ncclCommInitRank",
        )

    @classmethod
    def _from_handle(
        cls, lib: Any, name: str, vendor: str, handle: Any,
        rank: int, world_size: int,
    ) -> "NcclComm":
        """``ncclCommInitAll`` 已经建好的 handle：跳过 init，只补齐字段。"""
        comm = cls.__new__(cls)
        comm.lib = lib
        comm.name = name
        comm.vendor = vendor
        comm.rank = rank
        comm.world_size = world_size
        comm._stream = None
        comm.handle = handle
        comm._destroyed = False
        return comm

    # -- 生命周期 -----------------------------------------------------

    def destroy(self) -> None:
        """显式销毁；重复调用是幂等的。"""
        if self._destroyed:
            return
        _check(self.lib.ncclCommDestroy(self.handle), "ncclCommDestroy")
        self._destroyed = True
        self.handle = ctypes.c_void_p()

    def __enter__(self) -> "NcclComm":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.destroy()

    def __repr__(self) -> str:
        return (
            f"NcclComm(rank={self.rank}, world_size={self.world_size}, "
            f"lib={self.name!r})"
        )

    # -- 语义检查 -----------------------------------------------------

    def _stream_ptr(self, stream: Optional[int]) -> ctypes.c_void_p:
        value = self._stream if stream is None else int(stream)
        return ctypes.c_void_p(0 if value is None else value)

    def _check_comm(self) -> None:
        if self._destroyed:
            raise RuntimeError("communicator 已销毁，不能再发起集合调用")

    def _sizes(self, array: Any, label: str) -> Tuple[ctypes.c_void_p, int, int]:
        _contiguous(array, label)
        ptr = _pointer(array, label)
        count = int(array.size)
        dtype = data_type(array.dtype)
        return ptr, count, dtype

    # -- 集合操作 -----------------------------------------------------

    def all_reduce(
        self,
        array: Any,
        *,
        op: str = "sum",
        out: Any = None,
        stream: Optional[int] = None,
    ) -> Any:
        """原地（或写入 ``out``）归约：每个 rank 结束后都拿到同一份结果。"""
        self._check_comm()
        target = array if out is None else out
        send, count, dtype = self._sizes(array, "all_reduce 的输入")
        recv, out_count, out_dtype = self._sizes(target, "all_reduce 的输出")
        if out_count != count:
            raise ValueError(
                f"all_reduce 的输入有 {count} 个元素，输出有 {out_count} 个"
            )
        if out_dtype != dtype:
            raise ValueError("all_reduce 的输入与输出 dtype 不一致")
        _check(
            self.lib.ncclAllReduce(
                send, recv, count, dtype, red_op(op), self.handle,
                self._stream_ptr(stream),
            ),
            "ncclAllReduce",
        )
        return target

    def all_gather(
        self,
        send: Any,
        recv: Any,
        *,
        stream: Optional[int] = None,
    ) -> Any:
        """每 rank 贡献 ``send``，``recv`` 按 rank 顺序拼接全部贡献。"""
        self._check_comm()
        send_ptr, count, dtype = self._sizes(send, "all_gather 的发送缓冲")
        recv_ptr, total, recv_dtype = self._sizes(recv, "all_gather 的接收缓冲")
        expected = count * self.world_size
        if total != expected:
            raise ValueError(
                f"all_gather 的接收缓冲需要 {expected} 个元素"
                f"（{count} × world_size {self.world_size}），得到 {total}"
            )
        if recv_dtype != dtype:
            raise ValueError("all_gather 的发送与接收 dtype 不一致")
        _check(
            self.lib.ncclAllGather(
                send_ptr, recv_ptr, count, dtype, self.handle,
                self._stream_ptr(stream),
            ),
            "ncclAllGather",
        )
        return recv

    def reduce_scatter(
        self,
        send: Any,
        recv: Any,
        *,
        op: str = "sum",
        stream: Optional[int] = None,
    ) -> Any:
        """把 ``send`` 切成 world_size 段归约，本 rank 只收第 ``rank`` 段。

        这里刻意不实现"变长分片"：NCCL 要求各段等长，按参数名分片会让每段
        大小不同。跨进程 ZeRO 的梯度同步因此用 AllReduce + 本地取片（ZeRO-1
        的通信量本来与 DDP 相同，省的是槽位内存）。
        """
        self._check_comm()
        send_ptr, total, dtype = self._sizes(send, "reduce_scatter 的发送缓冲")
        recv_ptr, count, recv_dtype = self._sizes(recv, "reduce_scatter 的接收缓冲")
        if total != count * self.world_size:
            raise ValueError(
                f"reduce_scatter 的发送缓冲需要 {count * self.world_size} 个元素"
                f"（每段 {count} × world_size {self.world_size}），得到 {total}"
            )
        if recv_dtype != dtype:
            raise ValueError("reduce_scatter 的发送与接收 dtype 不一致")
        _check(
            self.lib.ncclReduceScatter(
                send_ptr, recv_ptr, count, dtype, red_op(op), self.handle,
                self._stream_ptr(stream),
            ),
            "ncclReduceScatter",
        )
        return recv

    def broadcast(
        self,
        array: Any,
        *,
        root: int,
        stream: Optional[int] = None,
    ) -> Any:
        """把 ``root`` 的数组原地广播给所有 rank。"""
        self._check_comm()
        if isinstance(root, bool) or not isinstance(root, int):
            raise TypeError(f"root 必须是 int，收到 {root!r}")
        if not 0 <= root < self.world_size:
            raise ValueError(
                f"root {root} 越界：world_size={self.world_size}"
            )
        ptr, count, dtype = self._sizes(array, "broadcast 的缓冲")
        _check(
            self.lib.ncclBroadcast(
                ptr, ptr, count, dtype, root, self.handle,
                self._stream_ptr(stream),
            ),
            "ncclBroadcast",
        )
        return array

    def group(self) -> "NcclGroup":
        """``with comm.group():`` 把多个集合调用合成一次提交。"""
        self._check_comm()
        return NcclGroup(self)

    def synchronize(self) -> None:
        """等当前流上的集合调用完成。

        厂商由加载到的库决定：RCCL 走项目自己的 HIP 运行时（设备级同步），
        NCCL 走 cupy 的当前流。**不能两边都试**——HIP 机器上通常没有可用的
        cupy，CUDA 机器上 import 项目的 HIP 运行时也只会失败，靠 try/except
        猜厂商会把"同步失败"变成"同步被静默跳过"。
        """
        if self.vendor == "hip":
            from ..backends.hip import runtime

            runtime.synchronize()
            return
        import cupy as cp

        cp.cuda.get_current_stream().synchronize()


class NcclGroup:
    """``ncclGroupStart`` / ``ncclGroupEnd`` 的上下文管理器。

    放在组里的调用都是"先登记、组尾一次性提交"：小张量逐个广播时，组调用
    能把 n 次握手压成一次。组内不要做主机侧阻塞操作（同步会让组内其余 rank
    等不到对端）。
    """

    def __init__(self, comm: NcclComm) -> None:
        self._comm = comm

    def __enter__(self) -> "NcclGroup":
        _check(self._comm.lib.ncclGroupStart(), "ncclGroupStart")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # 组内已经出错时仍要 End，否则 communicator 停在"半开"状态，后续
        # 调用会以超时的形式炸在别处；End 的状态不覆盖原异常。
        status = self._comm.lib.ncclGroupEnd()
        if exc_type is None:
            _check(status, "ncclGroupEnd")
