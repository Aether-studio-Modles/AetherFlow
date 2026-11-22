"""训练循环：把"前向 → 反向 → 更新"的样板固化一次。

它**不是新的执行机制**：前向是一次 ``Backend.execute``，反向是另一张普通图
的一次执行，更新是优化器对数组的原地改写。封装只是去掉两处最容易写错的
细节——反向图的种子名与形状、每个参数的梯度名。

**为什么要求静态 batch**：反向图在构造期就要定形状。``reduce_sum`` 的 vjp
需要把梯度广播回输入形状，符号维下拿不到（``ir/grad_rules.py`` 明确拒绝），
而训练目标必然经过一次归约。所以这里在构造时就把"含符号维"拦下来，给一条
能看懂的错误，而不是等到 ``backward`` 里冒出一句 reduce_sum 的抱怨。
变长数据请补齐或丢弃（``DataLoader(drop_last=True)``）。
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from ..backends.base import Backend
from ..backends.reference_backend import ReferenceBackend
from ..ir.grad import backward
from ..ir.graph import Graph
from ..ir.serialize import dumps as _serialize_dumps
from ..ir.shape_contract import symbolic_inputs
from ..passes.autocast import Autocast
from ..runtime.bf16 import pack_bf16, unpack_bf16
from ..runtime.executor import _HostCopy
from ._arrays import array_module
from .amp import GradScaler
from .bn import RunningStats, bn_sites, build_stats_graph
from .checkpoint import TrainingSnapshot, inspect_checkpoint, load_checkpoint
from .data import DataLoader
from .grads import GradientAccumulator
from .optim import Optimizer
from .params import ParameterStore
from .recompute import CheckpointPlan
from .schedule import apply_lr
from .zero3 import Zero3Rank

__all__ = ["Trainer", "TrainingState"]

# key 是 i32 输入（dropout_mask 的第二个操作数），所以派生结果只取 31 位：
# 正数在 int32 / uint64 之间不会因为符号扩展产生差异。
_KEY_MASK = 0xFFFFFFFFFFFFFFFF
_KEY_GOLDEN = 0x9E3779B97F4A7C15
_KEY_MIX_A = 0xBF58476D1CE4E5B9
_KEY_MIX_B = 0x94D049BB133111EB


def _plain_config(value: Any) -> Any:
    """把配置值收成能进 meta 的标量：npz 的 meta 只收 0 维标量。"""
    if isinstance(value, Mapping):
        return {str(key): _plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_config(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f"配置凭证里的 {type(value).__name__} 无法序列化；"
        "hyperparams() 只能返回标量、标量列表或它们的嵌套字典"
    )


def _config_json(value: Any) -> str:
    """配置凭证的稳定字符串：键排序、无多余空白，可直接判等。"""
    return json.dumps(
        _plain_config(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _derive_rng_key(seed: int, step: int, rank: int) -> int:
    """把 ``(seed, step, rank)`` 混成一个 i32 key。

    为什么不是 ``step + rank * 大步长``：加法在相邻 rank 之间只差一个常数，
    SplitMix64 的输入因此落在同一条等差线上，低位相关性会传进掩码。这里
    用 SplitMix64 的 finalizer 做雪崩，再取 31 位正数。

    同一个 ``(seed, step, rank)`` 逐位可复现，与进程调度无关；不同 rank 在
    同一 step 拿到不同掩码，多卡的梯度噪声才是独立的。
    """
    z = (
        int(seed) * _KEY_GOLDEN
        + int(step) * _KEY_MIX_B
        + int(rank) * _KEY_MIX_A
    ) & _KEY_MASK
    z = ((z ^ (z >> 30)) * _KEY_MIX_A) & _KEY_MASK
    z = ((z ^ (z >> 27)) * _KEY_MIX_B) & _KEY_MASK
    return int((z ^ (z >> 31)) >> 33) & 0x7FFFFFFF


class TrainingState(NamedTuple):
    """训练状态的只读快照（不含数组，只描述"走到哪了"）。

    ``device`` 表示设备态**已经建立**（第一次 step 之后），不是构造参数；
    ``host_in_sync`` 表示 ``store`` 里的数值是否等于驻留参数——设备态训练
    期间它是 ``False``，直到调用 ``sync_to_host()``。
    """

    steps: int
    optimizer_step: int
    device: bool
    host_in_sync: bool
    params: Sequence[str]


class Trainer:
    """一张图 + 一组参数 + 一个优化器的训练循环。

    ``output`` 缺省取图的第一个输出，必须是标量损失。反向图在构造时建一次，
    之后每步只换数据——与 ``test_train.py`` 里手写循环的做法一致。

    ``segments`` 打开激活重算（梯度检查点）：前向按节点顺序切成这么多段，
    只保留段与段之间的边界，反向从最后一段往前跑。梯度数值与整图反向一致
    （边界被多个段使用时末位可能有差），峰值显存由"最宽的一段 + 全部边界"
    决定，而不是"整张反向图的所有中间值"。见 ``train/recompute.py``。

    ``device_state`` 打开后参数与梯度常驻显存：第一次 ``step`` 时上传参数，
    为前向与反向各捕获一张 CUDA Graph，两张图共享同一组参数缓冲。之后每步
    只把本批数据上传一次、把损失标量搬回主机，权重更新直接发生在优化器手里
    的显存数组上——既没有 H2D/D2H 往返，也不会因为"权重变了"让捕获作废。
    代价是 ``store`` 里的主机数组在训练期间是旧值，要主机侧数值时调
    ``sync_to_host()``；图与形状必须固定。

    **默认值 ``None`` 表示自动选择**：设备后端（有 ``capture``）上开设备态，
    参考后端或下列冲突场景留在主机态。这不是便利开关——主机态在设备后端上
    每步都要把全部参数重新 H2D，HelixLM 实测 0.28M 参数只有 torch 的 0.34×、
    4.2M 参数 0.27×；同一张图换设备态是 1.42× / 0.86×。以下情况自动留在
    主机态：``mixed_precision="bf16"``（设备后端只支持 f32/f16 缓冲）、
    ``zero3``（与常驻参数互斥）、含 BatchNorm 的图（running stats 未接）。
    需要每步读 ``store`` 或调试捕获时，显式传 ``False``。

    两个开关可以叠加：``segments`` 打开时，每一段各捕获一张前向图与一张反向
    图，段与段之间靠显存里的边界值衔接（边界值一组缓冲、cotangent 另一组），
    因此"省激活"和"省往返"同时成立。注意每段的捕获计划各持自己的 arena，
    显存总量约等于各段峰值之和——要峰值真的下降，用主机态 + ``segments``。

    ``grad_hook`` 插在"反向算完、优化器更新之前"这一段（裁剪、梯度累积、
    梯度噪声都在这里）：收到梯度字典，返回处理后的字典；返回 ``None`` 表示
    这一步不更新参数（例如累积窗口还没满）。两种模式共用同一个钩子，
    设备态下钩子里的辅助工具（``clip_global_norm`` / ``EMA`` 等）拿到的也是
    显存数组，不会把梯度拷回主机。

    ``mixed_precision="f16"`` 在构造期对图跑一遍 ``Autocast``：矩阵乘与卷积在
    f16 里算，归一化 / softmax / 归约留在 f32，其余算子跟随最宽输入。参数仍是
    f32（图输入与输出都保持原 dtype），所以优化器与存档代码不用改；反向图从
    转换后的前向图展开，梯度自然落回 f32。图必须是 f32 构图，f64 会被原样保留。

    ``mixed_precision="bf16"`` 走同一条路，差别只在数值与配套约束：
    bf16 的指数范围与 f32 相同（不会像 f16 那样溢出到 inf），尾数只有 7 位，
    所以**不需要** ``GradScaler``（传了会直接报错），相对误差也比 f16 大一个
    量级。参考执行器把每个 bf16 值量化到 bf16 精度；设备后端目前不支持 bf16，
    因此 ``device_state=True`` 与 bf16 的组合被拒绝。

    ``master_weights=True`` 让参数**以低精度驻留、在 f32 主副本上更新**，需要
    同时打开 ``mixed_precision="f16"`` 或 ``"bf16"``。它改三处：图输入声明为
    低精度（参数缓冲与读取带宽都减少）、``store`` 仍是 f32 主副本（优化器更新
    它、存档存它）、Trainer 内部多一份驻留缓冲，每次参数真的更新后从主副本
    覆盖写回。f16 驻留缓冲是 ``float16``；bf16 在 numpy 里没有原生类型，驻留
    缓冲用 2 字节的 ``uint16`` 存位模式，读图前解开成 f32 载体——这确实省下了
    一半的权重存储，而不是"借 f32 承载"的空标签。
    代价是参数多出一份 f32 主副本（设备态下总占用从 1 份 f32 变成 1.5 份），
    只被 f32-only 算子消费的参数也会在入口多一次 up-cast；净收益取决于模型
    里 f16 算子与 f32 算子的比例（见 DESIGN 5.126）。
    设备态下主副本与驻留缓冲都在显存：捕获图读 f16 那块，优化器写 f32 那块，
    每步末尾做一次 D2D 的 cast 写回。**外部改过 ``store``（例如
    ``load_checkpoint`` 之后）要调 ``refresh_resident()``**，否则训练读的仍是
    旧的低精度副本。

    ``rng_inputs`` 是带随机算子的图（``dropout_mask`` 的运行时 key）要声明的
    随机状态输入名。每一步先把当前 key 写进这些输入，再跑前向与反向，所以
    "同一 step 的前向、重算、反向共用同一个 key，下一步换新 key"。key 是图
    输入而不是隐藏的全局状态：同一份图 + 同一个 key 仍然逐位可复现，捕获成
    CUDA Graph 之后也能推进（标量 attrs 会被按值冻结）。

    ``rng_seed`` / ``rng_rank`` / ``rng_world_size`` 决定 key 的派生：
    ``key = mix(seed, step, rank)``（SplitMix64 的 finalizer，取 31 位）。
    只写 step 号时，同一 step 的所有 rank 会拿到同一张掩码——数据不同、
    噪声相同，多卡的随机性就退化了。``rank`` 必须落在 ``[0, world_size)``；
    同一个 ``(seed, step, rank)`` 的掩码逐位可复现，与进程调度无关。

    ``schedule`` 是学习率调度器（``train/schedule.py`` 里的任一对象）。每次
    **参数更新之前**按 ``optimizer.step_count`` 写一次 ``optimizer.lr``，因此
    步数是"真实更新次数"而不是微批次数：累积窗口内的几步不会让 warmup 提前
    用完。

    ``accumulate_steps > 1`` 打开梯度累积：连续这么多步只累加梯度、不更新
    参数，第 N 步用窗口平均值更新一次。累加发生在 ``grad_hook`` **之前**，
    所以钩子看到的是"整个窗口的梯度"——先累积再裁剪，那个上限才是有效 batch
    的上限；反过来每个微批各裁一次，窗口越长约束越松（见 ``train/grads.py``）。
    窗口内的梯度固定用 f32 缓冲。

    ``scaler`` 是 ``GradScaler``：反向种子从 1 换成当前 scale，取回梯度后先
    反缩放再交给优化器；溢出时跳过更新并减半 scale。与累积叠加时，反缩放
    发生在**每个微批**上（否则一个窗口里混着不同 scale 的梯度），任意微批
    出现非有限值就整窗作废——被污染的累加和没法只减掉出错的那一批。

    ``bn_momentum`` 维护训练态 BN 的 running stats（默认 0.1，与 torch 一致）。
    IR 的 ``batch_norm`` 只用当前 batch 的统计量，eval 需要的 running mean/var
    不在图里；不维护它，"训练输出对齐"就只是训练态对齐。开启后每步前向之后
    额外执行一张统计子图（只算到 BN 输入），按
    ``running = (1-m) * running + m * batch`` 更新，方差用无偏估计。
    ``None`` 表示显式关闭（例如图里没有 BN，或调用方自己维护）。**设备态
    训练暂不支持**：捕获计划要求每步地址固定，而 running stats 是主机侧
    EMA；含 BN 的图请用主机态，或用 ``bn_momentum=None`` 明确关掉。

    ``zero3`` 把参数本身也交给分片（``Zero3Rank``）：每段前向 / 反向之前
    只把**这一段要读的参数**拼回来（一次 AllGather），执行完立刻释放，因此
    任一时刻额外常驻的只有一段的参数，而不是整份。更新也只发生在本片
    （``step`` 里 ReduceScatter 出本块梯度）。三条硬约束：

    - **必须同时给 ``segments``**：整图前向要一次拼回全部参数，ZeRO-3 的
      "只留一段"就不成立了，这里直接拒绝而不是给一个看起来在分片的假象。
    - ``optimizer`` 必须是 ``zero3.optimizer``（分片优化器）：更新走
      ``zero3.step``，它需要完整梯度做 ReduceScatter，与整份优化器的
      ``step`` 不是同一条路。
    - 与 ``device_state`` / ``master_weights`` / ``scaler`` 互斥：前两个
      各自又要一份完整参数或显存缓冲，第三个的 unscale / 增长逻辑没有分片
      版本。``grad_hook`` / 累积 / ``schedule`` 仍然可用——它们只看完整梯度。

    ``store`` 里的参数在 ZeRO-3 下是**借出的完整副本**：训练期间它是旧值，
    ``sync_to_host()`` 把分片拼回来覆盖它（``snapshot`` 会先做这一步，避免
    存档静默写旧参数）。参数的真身始终是 ``zero3.shards``。
    """

    def __init__(
        self,
        graph: Graph,
        store: ParameterStore,
        optimizer: Optimizer,
        *,
        backend: Optional[Backend] = None,
        output: Optional[str] = None,
        segments: Optional[int] = None,
        device_state: Optional[bool] = None,
        mixed_precision: Optional[str] = None,
        master_weights: bool = False,
        grad_hook: Optional[Callable[[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]]]] = None,
        rng_inputs: Sequence[str] = (),
        rng_seed: int = 0,
        rng_rank: int = 0,
        rng_world_size: int = 1,
        schedule: Optional[Any] = None,
        accumulate_steps: int = 1,
        scaler: Optional[GradScaler] = None,
        bn_momentum: Optional[float] = 0.1,
        zero3: Optional[Zero3Rank] = None,
    ) -> None:
        if not isinstance(graph, Graph):
            raise TypeError(f"graph 必须是 Graph，得到 {type(graph).__name__}")
        if not isinstance(store, ParameterStore):
            raise TypeError(f"store 必须是 ParameterStore，得到 {type(store).__name__}")
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"optimizer 必须是 Optimizer，得到 {type(optimizer).__name__}")
        if isinstance(accumulate_steps, bool) or int(accumulate_steps) < 1:
            raise ValueError(f"accumulate_steps 必须是 >= 1 的整数，得到 {accumulate_steps!r}")
        if scaler is not None and not isinstance(scaler, GradScaler):
            raise TypeError(f"scaler 必须是 GradScaler，得到 {type(scaler).__name__}")
        if bn_momentum is not None:
            if isinstance(bn_momentum, bool) or not 0.0 <= float(bn_momentum) <= 1.0:
                raise ValueError(
                    f"bn_momentum 必须落在 [0, 1] 内（或 None 表示关闭），"
                    f"得到 {bn_momentum!r}"
                )
            bn_momentum = float(bn_momentum)
        if schedule is not None and not callable(getattr(schedule, "lr_at", None)):
            raise TypeError(
                f"schedule 必须有 lr_at(step)，得到 {type(schedule).__name__}"
            )
        if isinstance(rng_seed, bool) or not 0 <= int(rng_seed) < 2 ** 31:
            raise ValueError(f"rng_seed 必须落在 [0, 2^31)，得到 {rng_seed!r}")
        if isinstance(rng_world_size, bool) or int(rng_world_size) < 1:
            raise ValueError(
                f"rng_world_size 必须是 >= 1 的整数，得到 {rng_world_size!r}"
            )
        if isinstance(rng_rank, bool) or not 0 <= int(rng_rank) < int(rng_world_size):
            raise ValueError(
                f"rng_rank 必须落在 [0, {rng_world_size})，得到 {rng_rank!r}"
            )

        # 设备后端的默认训练态：参数常驻显存，前反向各一张捕获图，每步只
        # 传本批数据与损失标量。主机态在设备后端上每步都要把全部参数重新
        # H2D，实测 HelixLM 0.28M 参数 0.34×、4.2M 参数 0.27×（torch=1×）；
        # 同样的图换成设备态分别是 1.42× / 0.86×。以下情况必须留在主机态：
        # bf16（设备后端只支持 f32/f16 缓冲）、ZeRO-3（与常驻参数互斥）、
        # 含 BatchNorm 的图（设备态的 running stats 未接）。显式传 False
        # 可以回主机态，用于调试或需要每步读 store 的场景。
        sites = bn_sites(graph)
        if device_state is None:
            device_state = bool(
                backend is not None
                and hasattr(backend, "capture")
                and mixed_precision != "bf16"
                and zero3 is None
                and not sites
            )

        if master_weights and mixed_precision not in ("f16", "bf16"):
            raise ValueError(
                "master_weights=True 需要 mixed_precision=\"f16\" 或 \"bf16\"，"
                f"得到 {mixed_precision!r}：低精度驻留与低精度计算必须用同一种 dtype"
            )
        if scaler is not None and mixed_precision == "bf16":
            raise ValueError(
                "bf16 与 f32 的指数范围相同，梯度下溢不是它的问题，不需要 GradScaler；"
                "传 scaler 只会把梯度白乘一遍再除回来"
            )
        if device_state and mixed_precision == "bf16":
            raise ValueError(
                "设备态训练需要后端原生的 bf16 缓冲；当前设备后端只支持 f32 / f16。"
                "请在主机态使用 bf16，或改用 mixed_precision=\"f16\""
            )
        if master_weights:
            for name in store.names:
                if np.dtype(store[name].dtype) != np.dtype(np.float32):
                    raise TypeError(
                        f"参数 {name} 的 dtype 是 {store[name].dtype}；"
                        "master weights 需要 f32 主副本（f16 上做更新会被量化吞掉）"
                    )

        if zero3 is not None:
            if not isinstance(zero3, Zero3Rank):
                raise TypeError(
                    f"zero3 必须是 Zero3Rank，得到 {type(zero3).__name__}"
                )
            if device_state:
                raise ValueError(
                    "ZeRO-3 与设备态训练互斥：显存里仍要放一份完整参数作为图输入"
                    "缓冲，分片省下的常驻内存会被这份缓冲抵消；请用主机态"
                )
            if master_weights:
                raise ValueError(
                    "ZeRO-3 与 master_weights 互斥：低精度驻留缓冲是一份完整参数，"
                    "与「参数只留一块」相矛盾"
                )
            if scaler is not None:
                raise ValueError(
                    "ZeRO-3 的更新走分片优化器（zero3.step），没有与 GradScaler "
                    "的 unscale / 增长逻辑对接；请关闭 scaler"
                )
            if segments is None:
                raise ValueError(
                    "ZeRO-3 需要同时给 segments：整图前向要一次拼回全部参数，"
                    "分片就白做了；请给出段数（例如 segments=2）"
                )
            if zero3.optimizer is not optimizer:
                raise ValueError(
                    "ZeRO-3 下 optimizer 必须是 zero3.optimizer（分片优化器）："
                    "整份优化器会去改完整参数，而参数的真身是分片"
                )
            missing = sorted(set(zero3.param_names) - set(store.names))
            extra = sorted(set(store.names) - set(zero3.param_names))
            if missing or extra:
                raise KeyError(
                    f"分片与参数表的参数集合不一致：分片多 {missing}，"
                    f"参数表多 {extra}"
                )
            shapes = zero3.shapes()
            for name in store.names:
                if tuple(store[name].shape) != tuple(shapes[name]):
                    raise ValueError(
                        f"参数 {name} 的形状 {store[name].shape} 与分片计划 "
                        f"{shapes[name]} 不一致"
                    )
                if np.dtype(store[name].dtype) != np.dtype(zero3.shards[name].dtype):
                    raise TypeError(
                        f"参数 {name} 的 dtype {store[name].dtype} 与分片 "
                        f"{zero3.shards[name].dtype} 不一致"
                    )

        if mixed_precision is not None:
            # 先改图，再建反向图与捕获计划：下游看到的只有这一张前向图，
            # 不存在"前向 f16、反向 f32"的半套状态。
            graph = Autocast(
                dtype=mixed_precision,
                # 参数以 f16 驻留时，图输入也声明成 f16：参数缓冲减半，
                # 前向 / 反向读权重时带宽也减半。只被 f32-only 算子消费的
                # 参数会在入口多一次 cast 回到 f32——这是低精度驻留的代价，
                # 不是每个模型都划算（见 DESIGN 5.126 的结构数字）。
                low_precision_inputs=store.names if master_weights else (),
            ).run(graph)

        if device_state:
            # 主机态对符号批维是开放的（见 shape_contract 的 trainer 条目）：
            # 每一步按实际形状绑定，反向的归约计数也在运行期算。设备态不是
            # 同一个问题——捕获计划把每步的维度与地址写死在图里，换形状就得
            # 重新捕获，所以这里仍然要求静态。
            symbolic = symbolic_inputs(graph)
            if symbolic:
                listed = "、".join(f"%{name}" for name in symbolic)
                raise ValueError(
                    "设备态训练要求静态形状：前向/反向的捕获计划把每步的维度与"
                    f"地址写死在图里，这些输入含符号维: {listed}。"
                    "请改用主机态（device_state=False），或按固定 batch 构图"
                )

        inputs = {value.name for value in graph.inputs}
        missing = [name for name in store.names if name not in inputs]
        if missing:
            raise ValueError(f"这些参数不是图的输入: {missing}")

        outputs = list(graph.output_names)
        if not outputs:
            raise ValueError("图没有输出，无法定义训练目标")
        chosen = outputs[0] if output is None else output
        if chosen not in outputs:
            raise ValueError(f"输出 {chosen!r} 不在图输出里: {outputs}")
        target = graph.outputs[outputs.index(chosen)]
        if tuple(target.type.shape) != ():
            raise ValueError(
                f"训练目标必须是标量，输出 {chosen!r} 的形状是 {target.type.shape}"
            )

        self.graph = graph
        self.store = store
        self.optimizer = optimizer
        # 参数分片（ZeRO-3）：有它时 ``store`` 只是借出的完整副本，参数真身在
        # ``self._zero3.shards``；更新与同步都要多走一步。
        self._zero3 = zero3
        # 训练态 BN 的 running stats：IR 的 batch_norm 只要当前 batch 的统计量，
        # eval 需要的那对累积值由这里维护（见 ``train/bn.py``）。
        self.bn_momentum = bn_momentum
        self._bn_sites = sites
        self._bn_running: Optional[RunningStats] = None
        self._bn_stats_graph: Optional[Graph] = None
        if self._bn_sites and bn_momentum is not None:
            if device_state:
                raise ValueError(
                    "设备态训练不支持训练态 BN 的 running stats 维护：捕获计划要求"
                    "每步地址固定，而 running stats 是主机侧 EMA。请使用主机态，"
                    "或显式传 bn_momentum=None 关掉（此后 eval 状态需自行维护）"
                )
            self._bn_stats_graph = build_stats_graph(graph, self._bn_sites)
            self._bn_running = RunningStats(self._bn_sites)
        self.master_weights = bool(master_weights)
        self.mixed_precision = mixed_precision
        # 低精度驻留缓冲：只有 master weights 模式才有。``store`` 是 f32 主副本，
        # 这里是从它 cast 出来的一份低精度副本，图每步读这一份。f16 用
        # ``float16`` 容器；bf16 没有 numpy 类型，用 2 字节的 ``uint16``
        # 存位模式（``runtime/bf16.py``），读取时再还原成 f32 载体。
        self._resident: Optional[Dict[str, np.ndarray]] = (
            {name: self._to_resident(store[name]) for name in store.names}
            if self.master_weights
            else None
        )
        self.backend = backend if backend is not None else ReferenceBackend()
        self.output = chosen
        self.steps = 0
        self.device_state = bool(device_state)
        self.grad_hook = grad_hook
        self.schedule = schedule
        self.accumulate_steps = int(accumulate_steps)
        self.scaler = scaler
        # ``accumulate_steps == 1`` 时**不建累加器**：单步路径要保持逐位不变
        # （累加器会把梯度转成 f32 副本，f64 图的梯度精度会被悄悄改掉）。
        self._accumulator: Optional[GradientAccumulator] = (
            GradientAccumulator(self.accumulate_steps)
            if self.accumulate_steps > 1
            else None
        )

        # 设备态的运行期状态：参数/数据/种子的显存缓冲 + 两张捕获计划。
        # 主机态下这些一直是空的，``sync_to_host`` 因此是个空操作。
        self._param_names = frozenset(store.names)
        self._input_names = frozenset(value.name for value in graph.inputs)
        # 随机状态输入（dropout 的运行时 key）：每步写入派生 key，同一 step
        # 的前向、重算与反向都用同一个值。不是参数，也不该由用户逐批提供。
        self._rng_names = tuple(rng_inputs)
        self.rng_seed = int(rng_seed)
        self.rng_rank = int(rng_rank)
        self.rng_world_size = int(rng_world_size)
        unknown = [n for n in self._rng_names if n not in self._input_names]
        if unknown:
            raise ValueError(f"rng_inputs 里有不是图输入的名字: {unknown}")
        clash = [n for n in self._rng_names if n in self._param_names]
        if clash:
            raise ValueError(f"rng_inputs 不能是参数: {clash}")
        self._device_ready = False
        # 设备态下 ``store`` 何时变成旧值：只要走过一步参数更新就脏了，
        # ``sync_to_host()`` 之后重新一致。主机态永远是 False。
        # ZeRO-3 例外：训练期间 ``store`` 一直是借出的旧副本（真身在分片里），
        # 所以一开始就是脏的，``sync_to_host()`` 才把它拼回来。
        self._host_stale = self._zero3 is not None
        self._fwd_plan: Optional[Any] = None
        self._bwd_plan: Optional[Any] = None
        self._dev_params: Dict[str, Any] = {}
        # master weights 模式下的 f32 主副本（显存）；非该模式时为 None，
        # 主副本就是驻留参数本身。
        self._dev_master: Optional[Dict[str, Any]] = None
        self._dev_data: Dict[str, Any] = {}
        self._dev_seeds: Dict[str, Any] = {}

        # 分段设备态的对应状态：每段一张前向图与一张反向图，边界值与
        # cotangent 各占一组缓冲（前者给段反向重放，后者累加后当种子）。
        self._seg_fwd: List[Any] = []
        self._seg_bwd: List[Any] = []
        self._dev_acts: Dict[str, Any] = {}
        self._dev_cots: Dict[str, Any] = {}
        self._dev_grads: Dict[str, Any] = {}
        self._seg_stream: Optional[Any] = None

        self._outputs = list(zip(graph.outputs, outputs))
        self._seed_name = "d" + chosen
        self._loss_value = target
        self._params = [graph.find(name) for name in store.names]
        self.segments = segments
        if segments is None:
            # 只有 chosen 参与反向：其余输出在构图期就被剪掉，而不是喂零种子。
            # 零种子在导数链上碰上 0/x 这类项仍会算出 NaN，把对的梯度污染掉。
            self._bwd: Optional[Graph] = backward(
                graph, wrt=self._params, active=[chosen]
            )
            self._plan: Optional[CheckpointPlan] = None
        else:
            self._bwd = None
            self._plan = CheckpointPlan(
                graph, self._params, segments=segments, output=chosen
            )
        if self.device_state:
            if not all(
                callable(getattr(self.backend, name, None))
                for name in ("capture", "to_array", "to_host")
            ):
                raise TypeError(
                    "设备态训练需要支持整图捕获的后端（capture/to_array/to_host），"
                    f"{type(self.backend).__name__} 没有提供"
                )

    # -- 单步 ---------------------------------------------------------

    def data_input_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """每批需要调用方提供的输入，及它们的静态形状。

        排除参数（在 ``store`` 里）与随机状态输入（``rng_inputs``，由本类
        每步按 ``(seed, step, rank)`` 派生）。数据并行要按这张表核对"分成
        W 份后第 0 维是否等于图的静态 batch"——用"所有输入减参数"去猜会把
        dropout 的 key 也算成数据，报出一条对不上号的错误。

        含符号维的输入在这里显式拒绝：数据并行按每卡固定 batch 切分，符号
        批维没有"该切多少"的答案。主机态单机训练不走这张表，不受影响。
        """
        shapes: Dict[str, Tuple[int, ...]] = {}
        for value in self.graph.inputs:
            if value.name in self._param_names or value.name in self._rng_names:
                continue
            shape = tuple(value.type.shape)
            if not all(isinstance(d, int) for d in shape):
                raise ValueError(
                    f"数据输入 {value.name!r} 的形状 {shape} 含符号维；"
                    "数据并行按每卡固定 batch 切分，需要静态形状"
                )
            shapes[value.name] = shape
        return shapes

    def step(
        self,
        batch: Optional[Mapping[str, np.ndarray]] = None,
        *,
        weight: Optional[float] = None,
    ) -> float:
        """跑一步（前向、反向、更新）并返回这一步的损失。

        ``weight`` 只影响梯度累积：它表示这一微批在窗口平均里的份量。
        按 token 归一的损失（``masked_lm_loss``）应传这一批的有效 token 数
        （``float(mask.sum())``），这样窗口平均等于"按总有效 token 求均值"；
        变长样本与不同 padding 比例不会让某个微批被过度加权。缺省按 1 计，
        即逐个微批等权，与没有累积时的语义一致。
        """
        if self.device_state:
            return float(self._step_device(batch, weight=weight))
        feed = self._feed(batch)
        borrow = (
            (lambda segment: self._borrowed_params(feed, segment))
            if self._zero3 is not None
            else None
        )
        if self._plan is None:
            out = np.asarray(self.backend.execute(self.graph, feed)[self.output])
        else:
            acts = self._plan.forward(self.backend, feed, on_segment=borrow)
            out = np.asarray(acts[self._loss_value.name])
        if out.ndim != 0:
            raise ValueError(
                f"训练目标必须是标量，输出 {self.output!r} 的形状是 {out.shape}"
            )
        # BN 统计量属于"观察到的数据"，与这一步是否真的更新参数无关：
        # 累积窗口未满或 AMP 溢出跳过更新时，batch 统计量照样要进 EMA。
        self._observe_bn(feed)

        if self._plan is None:
            result = self.backend.execute(
                self._bwd,
                {**feed, self._seed_name: self._seed(out.dtype)},
            )
            grads = {name: result["d" + name] for name in self.store.names}
        else:
            # 分段反向也只喂损失那一个种子：CheckpointPlan 已按输出做了
            # 可达性剪枝，未参与损失的图输出不进任何段的反向图。
            seeds = {self._loss_value.name: self._seed(out.dtype)}
            grads = self._plan.backward(
                self.backend, feed, acts, seeds, on_segment=borrow
            )
        if self._update(grads, weight=weight):
            self._refresh_resident()
        self.steps += 1
        if self._zero3 is not None:
            # 更新只落在分片里，``store`` 里那份借出的完整副本已经旧了。
            self._host_stale = True
        return float(out)

    def step_async(
        self,
        batch: Optional[Mapping[str, np.ndarray]] = None,
        *,
        weight: Optional[float] = None,
    ) -> _HostCopy:
        """发起一步，立刻返回损失句柄；``handle.value()`` 才同步。

        与 :meth:`step` 的区别只在**损失的 D2H 什么时候等**。设备态下这一步
        的损失拷贝排在前向与反向之间：反向不读损失值（种子是常量 1 或当前
        scale），所以拷贝一排队，反向的 replay 就可以跟着排，主机不必停在
        那里等一次完整的传输。连续调用 ``step_async`` 时，每步那一次同步都
        被推到调用方真正需要损失的时刻。

        主机态没有设备拷贝，返回的句柄是"已经完成"的，语义与 ``step`` 相同。
        """
        if not self.device_state:
            return _HostCopy(self.step(batch, weight=weight))
        return self._step_device(batch, weight=weight)

    def _host_async(self, value: Any, stream: Any = None) -> _HostCopy:
        """损失的 D2H 句柄；后端没有异步面时退回同步拷贝。

        ``to_host_async`` 不是后端契约的一部分（5.271）：第三方后端只要给出
        ``to_host`` 就能训练，异步只是本家 CUDA 后端的优化，不实现也不会错。
        """
        fn = getattr(self.backend, "to_host_async", None)
        if fn is None:
            return _HostCopy(self.backend.to_host(value))
        return fn(value, stream)

    def _to_resident(self, value: np.ndarray) -> np.ndarray:
        """f32 主副本 -> 低精度驻留缓冲（f16 容器或 bf16 位模式）。"""
        if self.mixed_precision == "bf16":
            return pack_bf16(value)
        return np.asarray(value).astype(np.float16)

    def _resident_feed(self, name: str) -> np.ndarray:
        """低精度驻留缓冲 -> 图输入（bf16 解开成 f32 载体）。"""
        if self._resident is None:
            raise RuntimeError("没有低精度驻留缓冲")
        arr = self._resident[name]
        if self.mixed_precision == "bf16":
            return unpack_bf16(arr)
        return arr

    def _feed(self, batch: Optional[Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """主机态的完整输入：参数 + 本批数据 + 随机 key。

        参数用低精度驻留缓冲（master weights 模式），而不是 ``store`` 里的
        f32 主副本——图的输入已经声明成 f16 / bf16，交 f32 过去会让执行器
        直接报 dtype 不符；即便后端容忍隐式转换，那也等于每步都做一次没人
        记账的 cast。bf16 的驻留缓冲是打包的位模式，这里解开成 f32 载体
        （bf16 -> f32 是无损的，量化误差早已在打包时确定）。
        """
        feed = self.store.feed(dict(batch) if batch is not None else None)
        if self._zero3 is not None:
            # 参数不在 ``store`` 里取：真身在分片里，由每段执行前的借用回调
            # 临时拼进来（见 :meth:`_borrowed_params`）。喂旧副本会让训练
            # "看起来在跑"，权重却停在 materialize 的那一刻。
            for name in self.store.names:
                feed.pop(name, None)
        if self._resident is not None:
            for name in self.store.names:
                feed[name] = self._resident_feed(name)
        feed.update(self._rng_feed())
        return feed

    @contextmanager
    def _borrowed_params(self, feed: Dict[str, Any], segment: Any):
        """一段执行期间"借出"它要读的参数，退出时还回去（ZeRO-3）。

        借出 = 一次 AllGather 把这几个参数拼成完整形状，写进 ``feed``；
        还回 = 从 ``feed`` 删掉引用并 ``release`` 掉副本。两者必须在同一处
        成对出现：只删 feed 不做 release，缓存会一直留着；反之只 release
        不删 feed，``feed`` 里的引用会让那块内存根本回收不掉。
        """
        names = [orig for orig, _ in segment.inputs if orig in self._param_names]
        if not names:
            yield
            return
        feed.update(self._zero3.materialize(names))
        try:
            yield
        finally:
            for name in names:
                feed.pop(name, None)
            self._zero3.release(names)

    def _observe_bn(self, feed: Mapping[str, np.ndarray]) -> None:
        """执行统计子图，按 momentum 推进每个站点的 running stats。

        统计子图只覆盖 BN 输入之前的依赖闭包，因此代价远小于整张前向；
        没有开启 ``bn_momentum`` 或图里没有 BN 时是空操作。
        """
        if self._bn_running is None or self._bn_stats_graph is None:
            return
        local: Dict[str, Any] = dict(feed)
        borrowed: List[str] = []
        if self._zero3 is not None:
            # 统计子图只覆盖 BN 输入之前的那一小段，拼回这几个参数就够，
            # 不必为一次 EMA 观察把整份参数借出来。
            borrowed = [
                value.name
                for value in self._bn_stats_graph.inputs
                if value.name in self._param_names
            ]
            if borrowed:
                local.update(self._zero3.materialize(borrowed))
        try:
            stats = self.backend.execute(
                self._bn_stats_graph,
                {value.name: local[value.name] for value in self._bn_stats_graph.inputs},
            )
        finally:
            if borrowed:
                self._zero3.release(borrowed)
        for site in self._bn_sites:
            self._bn_running.update(site.name, stats[site.output], self.bn_momentum)

    def bn_running_stats(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """各站点的 ``(running_mean, running_var)`` 副本，可直接喂给 ``bn_infer``。

        训练态与推理态是两张图：把它们接起来的唯一状态就是这对统计量。
        没有开启维护时抛错，而不是返回初始值（那会让 eval 永远算错）。
        """
        if self._bn_running is None:
            raise ValueError(
                "训练器没有维护 BN running stats：图里没有 batch_norm 节点，"
                "或构造时传了 bn_momentum=None"
            )
        return self._bn_running.arrays()

    def _refresh_resident(self) -> None:
        """把 f32 主副本覆盖写回低精度驻留缓冲（主机态）。"""
        if self._resident is None:
            return
        for name in self.store.names:
            self._resident[name][...] = self._to_resident(self.store[name])

    def refresh_resident(self) -> None:
        """外部改过 ``store`` 之后，重建低精度驻留缓冲。

        ``load_checkpoint(store, ...)`` 直接写 ``store``；训练读的是内部那份
        f16 副本，不刷新的话权重看起来"加载成功"，实际这一步仍在用旧值。
        没开 ``master_weights`` 时调用它是配置错误，直接报错而不是静默跳过。
        """
        if self._resident is None:
            raise ValueError("训练器没有开启 master_weights，不存在低精度驻留缓冲")
        self._refresh_resident()

    def _rng_key(self) -> int:
        """第 ``steps`` 步、本 rank 的随机 key（见 :func:`_derive_rng_key`）。"""
        return _derive_rng_key(self.rng_seed, self.steps, self.rng_rank)

    def _rng_feed(self) -> Dict[str, np.ndarray]:
        """第 ``steps`` 步的随机 key（主机态）。

        同一个 step 里前向、重算与反向共用这一份；下一步自增，掩码才会
        重新采样。``steps`` 是"跑了几步"，与"更新了几次"无关。
        """
        value = np.asarray([self._rng_key()], dtype=np.int32)
        return {name: value for name in self._rng_names}

    def _advance_rng(self) -> None:
        """把第 ``steps`` 步的 key 写进设备缓冲（设备态）。"""
        if not self._rng_names:
            return
        value = np.asarray([self._rng_key()], dtype=np.int32)
        for name in self._rng_names:
            self._dev_data[name][...] = value

    def _advance_seed(self) -> None:
        """把当前 scale 写进反向种子的显存缓冲（设备态）。

        种子是捕获时写死的**地址**，不是写死的**值**：改这块缓冲的内容在
        replay 里立刻生效，scale 变化因此不需要重新捕获。
        """
        if not self._dev_seeds:
            return
        value = self._seed_scalar(self._loss_value.type.dtype.np_dtype)
        for buf in self._dev_seeds.values():
            buf[...] = value

    def _seed(self, dtype) -> np.ndarray:
        """反向种子的 0 维数组：有 scaler 时是当前 scale，否则是 1。

        缩放走"改种子"而不是"改损失常量"：种子是每次执行都能换的输入，
        图不用重建；代价是乘除发生在图外（见 ``train/amp.py``）。
        """
        if self.scaler is None:
            return np.asarray(1.0, dtype=dtype)
        return self.scaler.backward_seed(dtype)

    def _seed_scalar(self, np_dtype) -> float:
        """种子值本身（设备态给 0 维缓冲赋值用）。"""
        if self.scaler is None:
            return 1.0
        return float(self.scaler.backward_seed(np_dtype))

    def _update(
        self, grads: Mapping[str, np.ndarray], *, weight: Optional[float] = None
    ) -> bool:
        """反向之后、优化器之前：累积 → 钩子 → 调度 → 更新。

        顺序是刻意的：累积在钩子之前（钩子看到整个窗口的梯度，裁剪的上限
        才对得上有效 batch）；反缩放又在累积之前（窗口里不能混着不同 scale
        的梯度）。

        返回 ``True`` 表示参数真的被更新了（master weights 模式据此决定是否
        写回低精度驻留缓冲）。
        """
        if self._accumulator is not None:
            grads = self._accumulate(grads, weight)
            if grads is None:
                return False
        return self._apply(grads, already_unscaled=self._accumulator is not None)

    def _accumulate(
        self, grads: Mapping[str, np.ndarray], weight: Optional[float] = None
    ) -> Optional[Dict[str, np.ndarray]]:
        """把一个微批并进窗口；返回窗口加权平均梯度，窗口未满时返回 ``None``。

        AMP 与累积叠加时在这里反缩放，因为每个微批的 scale 可能不同。任何
        一个微批出现非有限值就整窗作废：累加和里没法只减掉出错的那一批。
        """
        if self.scaler is not None:
            grads = self.scaler.unscale(grads)
            if self.scaler.found_nonfinite:
                self._accumulator.discard()
                self.scaler.backoff()
                return None
        if not self._accumulator.add(
            grads, 1.0 if weight is None else float(weight)
        ):
            return None
        return self._accumulator.flush()

    def _apply(self, grads: Mapping[str, np.ndarray], *, already_unscaled: bool) -> bool:
        """反缩放 → 有限性 → 钩子 → 调度 → 优化器。返回值表示这一步是否更新了参数。

        顺序是刻意的：先反缩放再交给钩子，裁剪上限才对得上真实梯度量级；
        溢出时立刻回退并跳过，钩子与调度器都不会看到一批已经不可信的算术。
        钩子可能把梯度改坏（裁剪之外的加噪等），所以 ``update`` 会再查一次。
        """
        if self.scaler is not None and not already_unscaled:
            grads = self.scaler.unscale(grads)
            if self.scaler.found_nonfinite:
                self.scaler.backoff()
                return False
        if self.grad_hook is not None:
            grads = self.grad_hook(grads)
            if grads is None:
                return False
        if self.schedule is not None:
            # 用 ``optimizer.step_count``（真实更新次数）：累积窗口内的几次
            # 反向不会让 warmup 提前用完。
            apply_lr(self.optimizer, self.schedule)
        if self._zero3 is not None:
            # 完整梯度 → ReduceScatter 出本块 → 只更本块。``zero3.step`` 自带
            # ``/world_size``，与数据并行"梯度先求和再平均"的语义一致。
            self._zero3.step(grads)
            return True
        if self.scaler is None:
            self.optimizer.step(grads)
            return True
        # 到这里梯度必然已经反缩放过（累积路径在 ``_accumulate`` 里做）。
        return self.scaler.update(self.optimizer, grads)

    # -- 设备态 -------------------------------------------------------

    def _step_device(
        self,
        batch: Optional[Mapping[str, np.ndarray]],
        weight: Optional[float] = None,
    ) -> _HostCopy:
        """参数留在显存的一步。整图与分段两种形态走同一条入口。"""
        feed = self.store.feed(dict(batch) if batch is not None else None)
        data = {
            name: arr
            for name, arr in feed.items()
            if name not in self._param_names and name in self._input_names
        }
        if self._device_ready:
            # 每步先推进 key，再把缓冲本身递进 feed：replay 只换内容，不换地址
            self._advance_rng()
            self._advance_seed()
            data.update({name: self._dev_data[name] for name in self._rng_names})
        else:
            data.update(self._rng_feed())
        if self._plan is None:
            return self._step_device_whole(data, weight=weight)
        return self._step_device_segments(data, weight=weight)

    def _step_device_whole(
        self, data: Mapping[str, np.ndarray], weight: Optional[float] = None
    ) -> _HostCopy:
        """整图版：数据上拷一次、前向与反向各 replay 一次。

        全过程只有损失标量一次 D2H；梯度是显存里的图输出，直接进优化器。
        这次 D2H 用异步句柄：反向 replay 不读损失，排完拷贝就可以继续排，
        主机等它是在 ``step`` 取 ``float`` 的那一刻。
        """
        self._prepare_device(data)

        # 本批数据在这里（且只在这里）从主机拷进静态缓冲；host -> device 的
        # 拷贝由 replay 负责，所以直接传 host 数组即可。
        outs = self._fwd_plan.run_device({**data, **self._dev_params})
        loss = self._host_async(outs[self.output])

        # 反向图读的是前向刚写好的那批数据缓冲，不必再拷一遍。
        seeds = {**self._dev_data, **self._dev_params, **self._dev_seeds}
        result = self._bwd_plan.run_device(seeds)
        if self._update(
            {name: result["d" + name] for name in self.store.names}, weight=weight
        ):
            self._copy_master_to_resident()
        self.steps += 1
        self._host_stale = True
        return loss

    def _prepare_device(self, data: Mapping[str, np.ndarray]) -> None:
        """首步建立设备态：上传参数、分配数据缓冲、捕获前向与反向两张图。

        两张图拿到的是**同一组**参数缓冲（``shared_buffers``），所以优化器
        原地更新改的就是图里写死的那段地址；权重换代也不需要重捕获。
        """
        if self._device_ready:
            return
        backend = self.backend
        self._check_data_inputs(data)
        dev_params = self._upload_params()
        master = self._upload_master(dev_params)
        dev_data = {name: backend.to_array(arr) for name, arr in data.items()}
        # 反向图只有损失那一个种子输入（其余输出在构图期已剪掉）。
        loss_type = self._loss_value.type
        np_dtype = loss_type.dtype.np_dtype
        dev_seeds: Dict[str, Any] = {
            self._seed_name: backend.to_array(
                np.asarray(
                    self._seed_scalar(np_dtype),
                    dtype=np_dtype,
                )
            )
        }

        shared = {**dev_params, **dev_data}
        fwd = backend.capture(self.graph, shared, shared_buffers=shared)
        bwd_inputs = {**shared, **dev_seeds}
        bwd = backend.capture(self._bwd, bwd_inputs, shared_buffers=bwd_inputs)

        self.optimizer.rebind(master)
        self._dev_params = dev_params
        self._dev_master = None if master is dev_params else master
        self._dev_data = dev_data
        self._dev_seeds = dev_seeds
        self._fwd_plan = fwd
        self._bwd_plan = bwd
        self._device_ready = True

    # -- 设备态 + 激活重算 -------------------------------------------

    def _step_device_segments(
        self, data: Mapping[str, np.ndarray], weight: Optional[float] = None
    ) -> _HostCopy:
        """分段版：每段各有一张前向图与一张反向图，全部共享同一条流。

        显存省法不变（只留边界值），但边界与 cotangent 都在显存里，段间不再
        经过主机；每步仍然只有损失标量一次 D2H。
        """
        self._prepare_segments(data)
        segments = self._plan.segments

        # 数据只在"第一个用到它的段"里以主机数组传入，由 replay 走 pinned
        # 异步拷贝；后面再读同一个数据输入的段直接拿已就位的显存缓冲。
        pending = dict(data)
        for segment, plan in zip(segments, self._seg_fwd):
            feed = self._segment_feed(segment)
            for orig, local in segment.inputs:
                if orig in pending:
                    feed[local] = pending.pop(orig)
            outs = plan.run_device(feed)
            for orig, local in segment.outputs:
                self._dev_acts[orig][...] = outs[local]
        loss = self._host_async(
            self._dev_acts[self._loss_value.name], self._seg_stream
        )

        # cotangent 与参数梯度都是"累加"出来的，每步先清零：不清零会读到
        # 上一轮的残留。损失输出的 cotangent 是 dL/dL = 1。
        for grad in self._dev_grads.values():
            grad[...] = 0
        for cot in self._dev_cots.values():
            cot[...] = 0
        # 损失输出的 cotangent 是 dL/dL = 1；有 scaler 时是当前 scale
        # （d(scale*L)/dL），反缩放在 ``_update`` 里做。
        self._dev_cots[self._loss_value.name][...] = self._seed_scalar(
            self._loss_value.type.dtype.np_dtype
        )

        for index in range(len(segments) - 1, -1, -1):
            segment = segments[index]
            feed = self._segment_feed(segment)
            for orig, local in segment.outputs:
                # 种子就是 cotangent 缓冲本身：后段的贡献已经累加在里面。
                feed["d" + local] = self._dev_cots[orig]
            result = self._seg_bwd[index].run_device(feed)
            for orig, local in segment.inputs:
                if orig in self._dev_grads:
                    self._dev_grads[orig] += result["d" + local]
                elif orig in self._dev_cots:
                    self._dev_cots[orig] += result["d" + local]
                # 数据输入的 cotangent 没人要，直接丢掉。

        if self._update(dict(self._dev_grads), weight=weight):
            self._copy_master_to_resident()
        self.steps += 1
        self._host_stale = True
        return loss

    def _prepare_segments(self, data: Mapping[str, np.ndarray]) -> None:
        """分段版设备态：按段捕获，全部复用同一条流。

        与前向/反向两张图的版本相比，这里多了"边界缓冲"（``dev_acts``）：
        段的产出先落在自己的图输出里，再 D2D 拷进边界缓冲；反向时边界缓冲
        直接当种子输入，省掉一次拷贝。共用同一条流让 k 段 replay 天然有序，
        不必建 k-1 条跨流事件依赖。
        """
        if self._device_ready:
            return
        backend = self.backend
        self._check_data_inputs(data)
        dev_params = self._upload_params()
        master = self._upload_master(dev_params)
        dev_data = {name: backend.to_array(arr) for name, arr in data.items()}

        segments = self._plan.segments
        # 边界值与 cotangent 是两组缓冲：段反向要靠边界值重放前向，
        # 所以 cotangent 不能覆盖它（这与主机态里 acts 与 seeds 分开同源）。
        dev_acts: Dict[str, Any] = {}
        dev_cots: Dict[str, Any] = {}
        for segment in segments:
            for orig, _ in segment.outputs:
                type_ = segment.types[orig]
                shape = tuple(int(d) for d in type_.shape)
                np_dtype = type_.dtype.np_dtype
                dev_acts[orig] = backend.to_array(np.zeros(shape, dtype=np_dtype))
                dev_cots[orig] = backend.to_array(np.zeros(shape, dtype=np_dtype))

        stream = backend.new_stream()
        buffers = {**dev_params, **dev_data, **dev_acts}
        seg_fwd = []
        seg_bwd = []
        for segment in segments:
            fwd_shared = {local: buffers[orig] for orig, local in segment.inputs}
            seg_fwd.append(
                backend.capture(
                    segment.graph, fwd_shared, shared_buffers=fwd_shared, stream=stream
                )
            )
            bwd_shared = dict(fwd_shared)
            for orig, local in segment.outputs:
                bwd_shared["d" + local] = dev_cots[orig]
            seg_bwd.append(
                backend.capture(
                    segment.backward, bwd_shared, shared_buffers=bwd_shared, stream=stream
                )
            )

        dev_grads = {
            name: backend.to_array(np.zeros(self.store[name].shape, self.store[name].dtype))
            for name in self.store.names
        }
        self.optimizer.rebind(master)
        self._dev_params = dev_params
        self._dev_master = None if master is dev_params else master
        self._dev_data = dev_data
        self._dev_acts = dev_acts
        self._dev_cots = dev_cots
        self._dev_grads = dev_grads
        self._seg_fwd = seg_fwd
        self._seg_bwd = seg_bwd
        self._seg_stream = stream
        self._device_ready = True

    def _segment_feed(self, segment) -> Dict[str, Any]:
        """段的输入：参数/数据优先，其次前面某段的产出（与 ``CheckpointPlan`` 一致）。"""
        feed: Dict[str, Any] = {}
        for orig, local in segment.inputs:
            if orig in self._dev_data:
                feed[local] = self._dev_data[orig]
            elif orig in self._dev_params:
                feed[local] = self._dev_params[orig]
            else:
                feed[local] = self._dev_acts[orig]
        return feed

    # -- 设备态的公共部分 ---------------------------------------------

    def _check_data_inputs(self, data: Mapping[str, np.ndarray]) -> None:
        missing = [
            value.name
            for value in self.graph.inputs
            if value.name not in self._param_names and value.name not in data
        ]
        if missing:
            raise KeyError(
                f"设备态训练第一次 step 就要给出这些数据输入（形状此后固定）: {missing}"
            )

    def _upload_params(self) -> Dict[str, Any]:
        """上传图输入用的参数缓冲（master weights 模式下是低精度驻留副本）。"""
        backend = self.backend
        if self._resident is not None:
            source = {
                name: self._resident_feed(name) for name in self.store.names
            }
        else:
            source = self.store.arrays()
        dev_params = {name: backend.to_array(source[name]) for name in self.store.names}
        # 优化器按数组模块分派（numpy / cupy）。后端若给出第三种容器，
        # 更新会走到 numpy 分支并在一句隐式转换里失败——在这里拦下来，
        # 给一句能看懂的话，而不是让用户去猜 AttributeError。
        self._check_optimizable(dev_params[self.store.names[0]])
        return dev_params

    def _upload_master(self, resident: Dict[str, Any]) -> Dict[str, Any]:
        """上传 f32 主副本；没开 master weights 时主副本就是驻留参数本身。"""
        if self._resident is None:
            return resident
        master = {
            name: self.backend.to_array(self.store[name]) for name in self.store.names
        }
        self._check_optimizable(master[self.store.names[0]])
        return master

    @staticmethod
    def _check_optimizable(probe: Any) -> None:
        if array_module(probe) is np:
            raise TypeError(
                "设备态训练需要优化器能原地更新的设备数组"
                "（numpy / cupy / HIP 运行时），"
                f"当前后端返回 {type(probe).__name__}；"
                "自定义数组容器请先接入 train/_arrays.array_module"
            )

    def _copy_master_to_resident(self) -> None:
        """设备态：f32 主副本 cast 回低精度驻留缓冲（D2D，不经过主机）。"""
        if self._dev_master is None:
            return
        for name in self.store.names:
            resident = self._dev_params[name]
            resident[...] = self._dev_master[name].astype(resident.dtype)

    def sync_to_host(self) -> None:
        """把显存里的参数**原地**写回 ``store``。

        设备态训练期间 ``store`` 里是上传时的旧值：存档、打印权重或做数值
        检查之前调一次。原地写而不是换数组，外部持有的 ``store[name]``
        引用仍然指向同一块内存。主机态下是空操作。

        master weights 模式下写回的是 f32 主副本（不是 f16 驻留缓冲），
        并顺手刷新主机侧的低精度副本。

        ZeRO-3 下这是唯一把参数从分片搬回 ``store`` 的路径：拼回一次、
        覆盖写、再释放，``store`` 因此重新是"完整且最新"的一份。
        """
        if self._zero3 is not None:
            full = self._zero3.materialize()
            try:
                for name in self.store.names:
                    self.store[name][...] = full[name]
            finally:
                self._zero3.release()
            self._host_stale = False
            return
        if not self._device_ready:
            return
        source = self._dev_master if self._dev_master is not None else self._dev_params
        for name in self.store.names:
            self.store[name][...] = self.backend.to_host(source[name])
        self._refresh_resident()
        self._host_stale = False

    # -- 训练状态与存档 -----------------------------------------------

    def training_state(self) -> TrainingState:
        """当前训练状态（步数、是否设备态、主机副本是否过期）。"""
        return TrainingState(
            steps=self.steps,
            optimizer_step=self.optimizer.step_count,
            device=self._device_ready,
            host_in_sync=not self._host_stale,
            params=self.store.names,
        )

    def verify_device_state(self) -> Dict[str, object]:
        """校验"权重、驻留参数、捕获缓冲、优化器引用"属于同一版本。

        设备态训练的收益来自"三处指向同一块显存"：优化器原地更新的数组、
        捕获计划里写死的输入缓冲、以及 ``_dev_params`` 记录的那一份。任意
        一处被换成新对象（换优化器、重新 ``to_array``、手工 ``rebind``），
        训练仍然会跑，但更新的是一份没人读的副本——损失曲线看起来正常，
        参数却停在某一步。这个方法把那条隐式契约变成可调用的一次检查。

        检查全程只比对象身份与数组模块，不做 D2H，因此可以放在每 N 步的
        自检里。返回一份摘要；任何一处不一致都抛 ``RuntimeError`` 并列出
        全部问题（而不是遇到第一个就停）。
        """
        if not self._device_ready:
            raise RuntimeError(
                "设备态还没有建立（第一次 step 之前无法校验）；"
                "主机态训练不存在「驻留参数与捕获缓冲」这层契约"
            )
        plans: List[Any] = []
        if self._fwd_plan is not None:
            plans.append(("fwd", self._fwd_plan))
        if self._bwd_plan is not None:
            plans.append(("bwd", self._bwd_plan))
        plans.extend((f"seg{i}.fwd", p) for i, p in enumerate(self._seg_fwd))
        plans.extend((f"seg{i}.bwd", p) for i, p in enumerate(self._seg_bwd))

        problems: List[str] = []
        slots = self.optimizer._slots()
        for name in self.store.names:
            resident = self._dev_params[name]
            # 优化器更新的是主副本：非 master weights 模式下它就是驻留参数，
            # master weights 模式下是另一块 f32 缓冲（每步 cast 写回前者）。
            updated = (
                self._dev_master[name] if self._dev_master is not None else resident
            )
            if self.optimizer.params.get(name) is not updated:
                problems.append(f"优化器手里的 {name} 不是上传的那份参数")
            for label, plan in plans:
                buf = plan.buffers.get(name)
                if buf is not None and buf is not resident:
                    problems.append(f"{label} 捕获时用的 {name} 与驻留参数不是同一块")
            for slot, arrays in slots.items():
                arr = arrays.get(name)
                if arr is not None and array_module(arr) is not array_module(updated):
                    problems.append(f"槽位 {slot}/{name} 不在参数所在的设备上")
        if problems:
            raise RuntimeError(
                "设备态训练的内部引用已经分叉（训练仍会跑，但更新的可能是一份"
                "没人读的副本）：\n  " + "\n  ".join(problems)
            )
        return {
            "steps": self.steps,
            "optimizer_step": self.optimizer.step_count,
            "plans": [label for label, _ in plans],
            "params": len(self.store.names),
            "host_in_sync": not self._host_stale,
        }

    def graph_fingerprint(self) -> str:
        """前向图的稳定指纹（16 位十六进制）。

        参数集合、形状与 dtype 已经由 ``load_checkpoint`` 校验，但**计算图
        本身**换了（激活、残差连接、算子实现）时这些全都对得上，续训却走上
        另一条轨迹。这里对序列化后的 IR 取哈希，把"同一组参数"收紧成"同一
        张图"。autocast 会改图里的 dtype，因此指纹也隐含了精度信息。
        """
        text = _serialize_dumps(self.graph)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _schedule_config(self) -> str:
        """调度器配置凭证；自定义调度器没有 ``hyperparams`` 时只比类型。"""
        if self.schedule is None:
            return ""
        hyper = getattr(self.schedule, "hyperparams", None)
        return _config_json(hyper()) if callable(hyper) else ""

    def snapshot(self, meta: Optional[Mapping[str, object]] = None) -> TrainingSnapshot:
        """在**当前 step 边界**取一份一致的主机快照。

        设备态下参数与优化器槽位都在显存里，这里各做一次 D2H；主机态下是
        一次拷贝。两条路都保证返回值与后续训练不共享内存，因此可以交给
        后台线程写盘。

        带 ``scaler`` 时把它的三个标量以 ``amp_scale`` / ``amp_growth`` /
        ``amp_skipped`` 塞进 meta（同名键会被覆盖）：缩放值是训练状态的一
        部分，不存的话续训会从 ``init_scale`` 重新摸索，前面几十步的"溢出了
        几次、该用多大 scale"全部丢掉。恢复用 :meth:`load_amp_state`。

        累积窗口里**尚未提交**的梯度走 ``aux``（``grad/<参数名>``）：只存
        参数与优化器状态的话，窗口里已经算过的几个微批会静默消失，续训的
        等效 batch 变小。窗口长度、已累积数与 RNG 约定（``seed``/``rank``）
        一并进 meta，供 :meth:`load_training_state` 校验"这是同一条轨迹"。

        前向图指纹与调度器类型 / 超参也一起进 meta：参数名、形状与 dtype
        全都相同，但计算图换了、或者衰减曲线换了，续训同样不是同一条轨迹。

        ZeRO-3 下参数真身只在分片里，这里会先 ``sync_to_host()`` 拼回一次
        （一次 AllGather，主线程上完成），避免 checkpoint 静默写进旧副本。
        """
        if self._zero3 is not None and self._host_stale:
            self.sync_to_host()
        merged: Dict[str, object] = dict(meta or {})
        if self.scaler is not None:
            for key, value in self.scaler.state_dict().items():
                merged[f"amp_{key}"] = value
        aux: Dict[str, np.ndarray] = {}
        merged.setdefault("accum_steps", self.accumulate_steps)
        merged.setdefault("accum_pending", 0)
        if self._accumulator is not None:
            merged["accum_pending"] = self._accumulator.pending
            merged["accum_weight"] = self._accumulator.weight
            for name, arr in self._accumulator.state_dict()["sum"].items():
                aux[f"grad/{name}"] = arr
        merged.setdefault("rng_seed", self.rng_seed)
        merged.setdefault("rng_rank", self.rng_rank)
        # 计算精度也是"同一条轨迹"的一部分：bf16 与 f16 的量化误差不同，
        # 把 bf16 的存档接进 f16 训练器，损失曲线会先跳一下再继续。
        merged.setdefault("mixed_precision", self.mixed_precision or "f32")
        # 配置凭证：参数集合与形状由 load_checkpoint 校验，这里补上"图本身"
        # 与"调度器怎么衰减"——两者都改了、看起来却仍是同一个模型时，
        # 续训的轨迹会悄悄换一条。
        merged.setdefault("graph_fingerprint", self.graph_fingerprint())
        merged.setdefault(
            "schedule_kind",
            "" if self.schedule is None else type(self.schedule).__name__,
        )
        merged.setdefault("schedule_config", self._schedule_config())
        if self._bn_sites:
            # momentum 是"这条轨迹"的一部分：0.1 与 0.01 的 EMA 在同样数据上
            # 收敛到不同的 eval 统计量。站点集合不必单记，图指纹已经覆盖。
            merged.setdefault(
                "bn_momentum", "" if self.bn_momentum is None else self.bn_momentum
            )
            if self._bn_running is not None:
                aux.update(self._bn_running.state_dict())
        source: Mapping[str, Any]
        if self._device_ready:
            # master weights 模式下存档存的必须是 f32 主副本：f16 驻留缓冲
            # 只是主副本的量化视图，存它会把权重的精度永久降一级。
            source = (
                self._dev_master if self._dev_master is not None else self._dev_params
            )
        else:
            source = self.store.arrays()
        return TrainingSnapshot(
            {name: source[name] for name in self.store.names},
            self.optimizer.state_dict(),
            steps=self.steps,
            meta=merged,
            aux=aux,
        )

    def load_amp_state(self, meta: Mapping[str, object]) -> None:
        """从 ``load_checkpoint`` 返回的 meta 恢复缩放状态。

        缺键直接报错而不是"保持当前 scale"：静默保持会让续训看起来接上了，
        实际却是从另一条缩放轨迹开始。没有 ``scaler`` 时也报错——调用它的
        意图就是"这份存档带 AMP"，配置对不上必须让人知道。
        """
        if self.scaler is None:
            raise ValueError(
                "训练器没有 GradScaler，无法恢复缩放状态；"
                "请用同一个 scaler 重建 Trainer 后再加载"
            )
        keys = ("scale", "growth", "skipped")
        missing = [f"amp_{key}" for key in keys if f"amp_{key}" not in meta]
        if missing:
            raise KeyError(
                f"存档 meta 里缺少 {missing}；这份存档不是带 AMP 的训练存档"
            )
        self.scaler.load_state_dict({key: meta[f"amp_{key}"] for key in keys})

    def load_training_state(
        self,
        path,
        *,
        state: Optional[Mapping[str, np.ndarray]] = None,
        loader: Optional[DataLoader] = None,
    ) -> Dict[str, object]:
        """恢复一整份训练状态，包括**尚未提交**的梯度累积窗口。

        与模块级 ``load_checkpoint`` 的差别在于它知道训练配置：累积窗口长度、
        RNG 约定、有没有 scaler、前向图指纹、调度器类型与超参。任何一项对不上
        都会在加载前报错，而不是等到续训几十步之后才从损失曲线上看出"这不是
        同一条轨迹"。

        参数与槽位仍然原地写入 ``store`` / 优化器；设备态**已经建立**时拒绝
        （那份训练参数在显存里，写主机副本不会生效）：请在第一次 ``step``
        之前加载，或者重建 Trainer。

        返回 meta，调用方自己的键（epoch、数据位置、验证指标等）原样带回。
        """
        if self._device_ready:
            raise RuntimeError(
                "设备态已经建立，参数与优化器槽位都在显存里；此时加载只会写主机副本。"
                "请在第一次 step 之前加载，或重建 Trainer 后再加载"
            )
        like = {name: self.store[name] for name in self.store.names}
        # BN running stats 的模板（形状对、值占位）；实际是否进 ``extra``
        # 取决于存档里的累积窗口，见下面的 pending 分支。
        bn_template = (
            self._bn_running.template() if self._bn_running is not None else {}
        )

        # 第一步只读 meta 与附加数组键名，把所有"是不是同一条轨迹"的检查做完。
        # ``load_checkpoint`` 一旦通过就已经覆盖了参数与优化器状态，任何排在
        # 它之后的校验失败都会留下半加载现场（权重是新的、累积窗口是旧的）。
        inspected = inspect_checkpoint(path)
        saved_meta: Mapping[str, object] = inspected["meta"]  # type: ignore[assignment]
        saved_aux: list = inspected["aux"]  # type: ignore[assignment]

        # 数据迭代位置与权重是同一份状态的两半：存档记了位置就必须恢复，
        # 没记位置就不能假装"从头开始"是续训。两边的存在性必须一致。
        # 存档 meta 只收标量，所以位置序列化成单个 JSON 字符串键
        # ``loader_state``——用一个专属键而不是 ``data_*`` 前缀，避免占用
        # 调用方自己的 meta 命名空间（``data_dir`` 之类不应被当成游标）。
        saved_state = saved_meta.get("loader_state")
        saved_data = None
        if saved_state is not None:
            try:
                parsed = json.loads(str(saved_state))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"存档里的 loader_state 不是合法 JSON（{exc}）；存档已损坏"
                ) from exc
            if not isinstance(parsed, Mapping):
                raise ValueError("存档里的 loader_state 不是映射；存档已损坏")
            saved_data = parsed
        if loader is not None and not isinstance(loader, DataLoader):
            raise TypeError(f"loader 必须是 DataLoader，得到 {type(loader).__name__}")
        if loader is not None:
            if not saved_data:
                raise KeyError(
                    "存档里没有数据迭代位置（loader_state）；它来自没有记录 "
                    "DataLoader 状态的训练，不能按当前 loader 续接"
                )
            saved_kind = str(saved_meta.get("loader_kind", ""))
            if saved_kind != type(loader).__name__:
                raise ValueError(
                    f"存档的数据迭代位置来自 {saved_kind or '未知类型'}，当前是 "
                    f"{type(loader).__name__}；游标不能接到另一种数据流上"
                )
            # 只校验、不写：真正的恢复放到所有提交完成之后。
            loader.validate_state_dict(saved_data)
        elif saved_data:
            raise KeyError(
                "存档带数据迭代位置，但加载时没有传 loader；"
                "不恢复位置会让续训从轮首重新开始，静默改掉数据顺序"
            )

        saved_acc = int(saved_meta.get("accum_steps", self.accumulate_steps))
        if saved_acc != self.accumulate_steps:
            raise ValueError(
                f"存档的累积窗口长度是 {saved_acc}，当前训练器是 "
                f"{self.accumulate_steps}；两者的等效 batch 不同，不能接着续训"
            )
        for key, current in (("rng_seed", self.rng_seed), ("rng_rank", self.rng_rank)):
            if key in saved_meta and int(saved_meta[key]) != int(current):
                raise ValueError(
                    f"存档的 {key}={saved_meta[key]} 与当前 {current} 不一致；"
                    "随机流不同，dropout 与重算的掩码都对不上"
                )
        if "mixed_precision" in saved_meta:
            saved_mp = str(saved_meta["mixed_precision"])
            current_mp = self.mixed_precision or "f32"
            if saved_mp != current_mp:
                raise ValueError(
                    f"存档的 mixed_precision={saved_mp!r} 与当前 {current_mp!r} 不一致；"
                    "量化误差不同，这不是同一条训练轨迹"
                )

        if "graph_fingerprint" in saved_meta:
            saved_fp = str(saved_meta["graph_fingerprint"])
            current_fp = self.graph_fingerprint()
            if saved_fp != current_fp:
                raise ValueError(
                    f"存档的前向图指纹 {saved_fp} 与当前 {current_fp} 不一致；"
                    "参数名与形状可能相同，但计算图不同，续训不是同一条轨迹"
                )
        if "schedule_kind" in saved_meta:
            saved_kind = str(saved_meta["schedule_kind"])
            current_kind = (
                "" if self.schedule is None else type(self.schedule).__name__
            )
            if saved_kind != current_kind:
                raise ValueError(
                    f"存档的调度器是 {saved_kind or '无'}，当前是 "
                    f"{current_kind or '无'}；学习率轨迹不同，不能接着续训"
                )
            if "schedule_config" in saved_meta:
                saved_cfg = str(saved_meta["schedule_config"])
                current_cfg = self._schedule_config()
                if saved_cfg != current_cfg:
                    raise ValueError(
                        f"存档的调度器配置 {saved_cfg} 与当前 {current_cfg} 不一致；"
                        "类型相同但超参不同，学习率轨迹不是同一条"
                    )

        saved_bn_keys = {str(key) for key in saved_aux if str(key).startswith("bn/")}
        if self._bn_sites:
            saved_m = saved_meta.get("bn_momentum")
            if saved_m is None:
                raise KeyError(
                    "存档 meta 里没有 bn_momentum：无法判断这份存档的 BN running "
                    "stats 是不是按当前策略维护的"
                )
            saved_str = "" if str(saved_m) == "" else float(saved_m)
            current_str = "" if self.bn_momentum is None else self.bn_momentum
            if saved_str != current_str:
                raise ValueError(
                    f"存档的 bn_momentum={saved_str!r} 与当前 {current_str!r} 不一致；"
                    "同一批数据上 EMA 会收敛到不同的 eval 统计量"
                )
            want_bn = set(bn_template)
            if saved_bn_keys != want_bn:
                raise KeyError(
                    "存档里的 BN running stats 与当前图不一致：缺少 "
                    f"{sorted(want_bn - saved_bn_keys)}，多出 "
                    f"{sorted(saved_bn_keys - want_bn)}"
                )
        elif saved_bn_keys:
            raise ValueError(
                f"存档带 {len(saved_bn_keys)} 个 BN running stats 条目，"
                "但当前图里没有 batch_norm 节点；请用同一张图重建训练器"
            )

        pending = int(saved_meta.get("accum_pending", 0))
        if not 0 <= pending <= self.accumulate_steps:
            raise ValueError(
                f"存档里的已累积微批数 {pending} 不在 [0, {self.accumulate_steps}]"
            )
        accum_weight = 0.0
        if pending:
            if self._accumulator is None:
                raise ValueError(
                    f"存档里有 {pending} 个未提交的微批，但当前训练器 "
                    "accumulate_steps=1，没有累积窗口可以接"
                )
            if "accum_weight" not in saved_meta:
                raise KeyError(
                    "存档里有未提交的累积窗口，却没有 accum_weight；"
                    "窗口平均的分母无法恢复，续训会得到被整体缩放的梯度"
                )
            accum_weight = float(saved_meta["accum_weight"])
            if not np.isfinite(accum_weight) or accum_weight <= 0.0:
                raise ValueError(
                    f"存档里的累积权重 {saved_meta['accum_weight']!r} 不是有限正数"
                )
            want_aux = {f"grad/{name}" for name in self.store.names}
            saved_grad = {str(key) for key in saved_aux} - saved_bn_keys
            if saved_grad != want_aux:
                raise KeyError(
                    "存档里的梯度累加和与当前参数集合不一致：缺少 "
                    f"{sorted(want_aux - saved_grad)}，多出 "
                    f"{sorted(saved_grad - want_aux)}"
                )
        else:
            saved_grad = {str(key) for key in saved_aux} - saved_bn_keys
            if saved_grad:
                raise ValueError(
                    f"存档里已累积微批数为 0，却带了 {len(saved_grad)} 个梯度累加和；"
                    "两份计数互相矛盾，无法判断该不该接着旧窗口"
                )

        # ``extra`` 必须与存档里的 ``aux/`` 键集合完全一致，所以它只能包含
        # "这份存档真的会有的条目"：没提交的累积窗口不写梯度模板。
        template: Dict[str, np.ndarray] = {}
        if pending:
            for name in self.store.names:
                template[f"grad/{name}"] = np.zeros(
                    tuple(like[name].shape), dtype=np.float32
                )
        template.update(bn_template)

        amp_keys = [key for key in saved_meta if str(key).startswith("amp_")]
        if self.scaler is not None:
            missing = [
                f"amp_{key}"
                for key in ("scale", "growth", "skipped")
                if f"amp_{key}" not in saved_meta
            ]
            if missing:
                raise KeyError(
                    f"存档 meta 里缺少 {missing}；这份存档不是带 AMP 的训练存档"
                )
            amp_scale = float(saved_meta["amp_scale"])
            if not np.isfinite(amp_scale) or amp_scale <= 0.0:
                raise ValueError(f"存档里的 AMP scale {saved_meta['amp_scale']!r} 不是有限正数")
            if float(saved_meta["amp_growth"]) < 0.0:
                raise ValueError(f"存档里的 AMP growth {saved_meta['amp_growth']!r} 不能为负")
            if int(saved_meta["amp_skipped"]) < 0:
                raise ValueError(f"存档里的 AMP skipped {saved_meta['amp_skipped']!r} 不能为负")
        elif amp_keys:
            raise ValueError(
                "存档带 AMP 缩放状态，但当前训练器没有 GradScaler；"
                "请用同一个 scaler 重建 Trainer 后再加载"
            )

        # 全部检查通过，下面才开始提交。
        meta = load_checkpoint(
            path,
            self.store,
            self.optimizer,
            state=state,
            extra=template,
            # 有调度器时 lr 是"训练走到哪"的一部分，必须从存档恢复；
            # 没有调度器时 lr 是静态配置，换了就不算同一条轨迹。
            restore_lr=self.schedule is not None,
        )
        if "trainer_steps" in meta:
            self.steps = int(meta["trainer_steps"])

        if pending:
            self._accumulator.load_state_dict(
                {
                    "steps": self.accumulate_steps,
                    "count": pending,
                    "weight": accum_weight,
                    "sum": {
                        name: template[f"grad/{name}"] for name in self.store.names
                    },
                },
                like=like,
            )
        elif self._accumulator is not None:
            self._accumulator.discard()

        if self._bn_running is not None:
            # ``load_checkpoint`` 已把存档值原地写进 template，这里再拷进
            # running stats（与梯度窗口同一条"先校验、后提交"的顺序）。
            self._bn_running.load_state_dict(
                {key: template[key] for key in bn_template}
            )

        if self.scaler is not None:
            self.load_amp_state(meta)

        self._refresh_resident()
        self._host_stale = False
        if loader is not None:
            # 所有提交（权重、累积窗口、BN、AMP、驻留副本）都完成后才推进游标：
            # 排在中间的话，后面的步骤一旦失败就会留下"权重已恢复、数据位置
            # 已前进、其余状态是旧的"的半现场。
            loader.load_state_dict(saved_data)  # type: ignore[arg-type]
        return meta

    def save_checkpoint(
        self,
        path,
        *,
        meta: Optional[Mapping[str, object]] = None,
        loader: Optional[DataLoader] = None,
        blocking: bool = True,
    ):
        """取快照并落盘。``blocking=False`` 时写盘在后台进行，返回 ``Future``。

        设备态下不要直接调模块级的 ``save_checkpoint(store, ...)``：那要求
        先 ``sync_to_host()``，而它会覆盖"训练期间 store 是旧值"的契约。这里
        的入口只读驻留参数，不动 ``store``。

        传 ``loader`` 时把它的迭代位置（epoch / 批号 / 配置）一并写进 meta 的
        ``data``：权重与"下一批是哪批"是同一次训练检查点的两半。加载时交给
        :meth:`load_training_state` 的 ``loader`` 参数即可接上。
        """
        merged: Dict[str, object] = dict(meta or {})
        if loader is not None:
            if not isinstance(loader, DataLoader):
                raise TypeError(f"loader 必须是 DataLoader，得到 {type(loader).__name__}")
            occupied = [
                key for key in merged if key in ("loader_state", "loader_kind")
            ]
            if occupied:
                raise ValueError(
                    f"meta 里已经有 {occupied}；不要再同时传 loader，"
                    "否则两份数据位置会互相覆盖"
                )
            merged["loader_kind"] = type(loader).__name__
            # 存档 meta 只收标量：位置整体序列化成一个 JSON 字符串。
            merged["loader_state"] = json.dumps(
                {key: int(value) for key, value in loader.state_dict().items()},
                sort_keys=True,
            )
        snapshot = self.snapshot(meta=merged or None)
        if blocking:
            snapshot.save(path)
            return None
        return snapshot.save_async(path)

    def run_loader(
        self,
        loader: DataLoader,
        epochs: int = 1,
        on_step: Optional[Callable[[int, float], None]] = None,
    ) -> List[float]:
        """按 ``DataLoader`` 自己的迭代位置跑 ``epochs`` 轮，位置随之推进。

        与 :meth:`run` 的差别只在数据来源：游标由 loader 维护，所以"跑到一半
        存档、恢复后从同一批继续"不用调用方手工记位置——把 loader 一起交给
        :meth:`save_checkpoint` / :meth:`load_training_state` 即可。
        """
        if not isinstance(loader, DataLoader):
            raise TypeError(f"loader 必须是 DataLoader，得到 {type(loader).__name__}")
        return self.run(loader, epochs=epochs, on_step=on_step)

    def run(
        self,
        batches: Iterable[Mapping[str, np.ndarray]],
        epochs: int = 1,
        on_step: Optional[Callable[[int, float], None]] = None,
    ) -> List[float]:
        """跑 ``epochs`` 轮；``on_step(index, loss)`` 用全局步号回调。"""
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError(f"epochs 必须是正整数，得到 {epochs!r}")
        losses: List[float] = []
        for _ in range(epochs):
            for batch in batches:
                loss = self.step(batch)
                losses.append(loss)
                if on_step is not None:
                    on_step(len(losses) - 1, loss)
        return losses
