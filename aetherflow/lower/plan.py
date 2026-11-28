"""执行计划：把"多少线程、多少个块、每个线程几个元素、花多少 shared"从后端搬出来。

这些数字同时决定两件事：**内核源码的形状**（循环上界、向量宽度、shared 声明
大小）与 **launch 配置**。过去它们分散在 ``backends/cuda/backend.py`` 的常量与
``lower/kernels.py`` 的硬编码里，约束只写在注释中——block 必须是 2 的幂、
shared 不能超 48KB、``vec=4`` 要求长度整除且指针 16 字节对齐。约束没人执行时，
违反的代价是驱动报一个难懂的错误（``invalid configuration argument``）或
静默算错，而这些都是可以提前判定的纯算术。

所以这一层只做一件事：**把布局决策变成纯函数**。它们不 import 任何后端、
不碰设备，因此本地（没有 GPU）就能单测边界。后端拿到计划后只负责 launch。

术语：
- ``block``：一个线程块里的线程数（2 的幂）。
- ``grid``：线程块数，受 ``max_blocks`` 上限约束。
- ``vec``：每个线程一条通道处理的元素数（1 或 4）。
- ``shared_bytes``：该内核一次 launch 需要的动态 shared memory。

这一层仍然不认识 Graph 与算子语义：它只回答"给定规模，该怎么铺线程"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = [
    "BLOCK_MIN",
    "BLOCK_MAX",
    "BLOCK_OPTIONS",
    "BLOCK_DEFAULT",
    "SHARED_LIMIT",
    "REDUCE_SCRATCH",
    "REDUCE_BLOCK",
    "ATTENTION_BLOCK",
    "CONV_TILE_W",
    "CONV_TILE_OPTIONS",
    "CONV_SHARED_LIMIT",
    "CONV_TILED_MAX_K",
    "CONV_TILED_MIN_WORK",
    "CONV_WINOGRAD_TILE",
    "CONV_WINOGRAD_FUSED_MIN_C",
    "CONV_WINOGRAD_FUSED_MAX_C",
    "CONV_WINOGRAD_FUSED_RESIDENT_MIN_C",
    "CONV_IMPLICIT_MID_K_MAX",
    "CONV_IMPLICIT_MID_WORK",
    "CONV_CUDNN_MIN_GFLOPS",
    "CONV_ROC_MAX",
    "CONV_CB_MAX",
    "SCAN_CHUNK",
    "SCAN_CHUNK_MIN_STEPS",
    "SCAN_CHUNK_MAX_ELEMS",
    "SCAN_LAST_MAX_ELEMS",
    "VEC_WIDTHS",
    "VEC4_ALIGN",
    "Launch",
    "ConvTilePlan",
    "cdiv",
    "pow2_block",
    "check_block",
    "auto_block",
    "can_vectorize",
    "grid_for",
    "plan_elementwise",
    "plan_reduce",
    "REDUCE_TILE_X",
    "plan_reduce_tiled",
    "CSR_SCAN_MAX",
    "CSR_SCAN_SLOTS",
    "CSR_SCAN_STEPS_PER_SM",
    "plan_csr_scan",
    "plan_csr_scan_enabled",
    "attention_block",
    "conv_oc_tile",
    "conv_implicit_oc_tile",
    "conv_implicit_fits",
    "plan_conv_tiled",
    "plan_conv_implicit",
    "conv_tiled_preferred",
    "conv_implicit_preferred",
    "conv_winograd_supported",
    "conv_winograd_preferred",
    "conv_winograd_fused_preferred",
    "scan_auto_chunk",
    "scan_last_auto_chunk",
    "scan_chunk",
]


# ---- 常量 ----------------------------------------------------------------
# 候选取 2 的幂：warp 是 32 线程，非 2 的幂会让最后一个 warp 有一半空转。
# 上限 1024 是单 block 的硬限制，下限 32 是一个完整 warp。
BLOCK_MIN = 32
BLOCK_MAX = 1024
BLOCK_OPTIONS = (128, 256, 512, 1024)

# 自动选 block 的上限（5.65）。这么大已经是"一个大 block"的收益边界：
# 再往上只会让单块占用更多驻留槽，而 5.63 的实测也没能在四档之间分出胜负。
BLOCK_DEFAULT = 256

# 每 block 可用的 shared memory 预算。单 block 上限是 64KB，但超过 48KB
# 需要 opt-in 动态声明，NVRTC 的默认配置拿不到，所以按 48KB 卡。
SHARED_LIMIT = 48 * 1024

# 归约 / 扫描类内核的 shared 暂存槽数（float）。与 block 上限一致：
# 块内归约树的每一级占一个槽，block 不可能超过 1024，所以 1024 个 float
# 恰好够用，也不会把 shared 预算吃掉（4KB）。
REDUCE_SCRATCH = 1024

# "一行一 block" 的归约（sum / max / mean / softmax / layernorm / rmsnorm）的
# block 上限。取 256 而不是 1024：这类内核一行只有 cols 个元素（通常 512~4096），
# 用 1024 线程时每 SM 只能驻 2 行、且归约树有 10 级 barrier，行级并行度全被
# 屏障吃掉了。天数 MR-V100 实测（3355 万元素、f16/f32 各五档列数）：
# cols=1024 的 layernorm 1.62 → 0.62 ms、softmax 1.45 → 0.63 ms；
# cols=4096 的 layernorm 0.68 → 0.44 ms（463 GB/s，已到 elementwise 峰值）。
# 每 SM 2048 线程时 256 能驻 8 行，且每线程 4~16 个元素的串行链还能提供 ILP。
REDUCE_BLOCK = 256

# 非末维归约在 inner 方向的线程上限。一个 warp 32 个线程正好覆盖一整段连续
# inner：再宽不会让访存更合并，只会把 reduce 轴（blockDim.y）的并行度挤掉。
REDUCE_TILE_X = 32

# ``spike_csr`` 的块内扫描槽位（int）：``blockDim.x`` 个扫描格，再加两格——
# 一格放"本段命中总数"，一格放"这一行已经写了多少个"。上限与 block 上限一致，
# 一个 block 不可能比 1024 更宽；1024 + 2 个 int 是 4.1 KB，与归约暂存同量级。
CSR_SCAN_MAX = BLOCK_MAX
CSR_SCAN_SLOTS = CSR_SCAN_MAX + 2

# 并行铺法只在"行数少到串行铺法填不满机器"时才用。串行铺法一个线程一行、
# 沿列号串行读，每线程有 ``width`` 次迭代可以靠 ILP 藏住访存延迟——所以
# 行数不需要很多就能吃满带宽（M40 上 2048 行、宽 4096 时它已经反超并行版）。
# 每 SM 32 个长循环线程是实测的拐点：M40（24 SM）上阈值 768 行，256 行及以下
# 并行版快 1.5×～2.2×，2048 行以上串行版快 5%～22%。
CSR_SCAN_STEPS_PER_SM = 32

# attention 每块处理的 key 数上限。256 与默认 block 一致：再大只是让每个线程
# 在累加循环里多走几行，不会提高并行度。
ATTENTION_BLOCK = 256

# conv2d 分块核在 W 方向的线程数。32 让一个 warp 覆盖一行连续输出列，
# 相邻线程读相邻输入，合并访存；也是权重在 shared memory 里的复用倍数。
CONV_TILE_W = 32

# 调优面暴露的 W 方向候选：半行 / 一行 / 两行。grid 的 x 维随之变化，
# 但内核读的是 ``blockDim.x``，所以同一份源码适配全部候选。
CONV_TILE_OPTIONS = (16, 32, 64)

CONV_SHARED_LIMIT = SHARED_LIMIT

# 分块核每个线程要做 K 次全局读、K*ROC 次乘加，K 一大就被 cuBLAS 的分块
# GEMM 甩开（实测见 DESIGN 5.16：K=72 慢 16%，K=576 慢 54%）。
# 实测只有 K <= 32 时稳定占优（+14%），因此阈值就卡在 32。
CONV_TILED_MAX_K = 32

# 输出总量太小的时候，一个 block 的权重 staging 抵不过计算本身。
CONV_TILED_MIN_WORK = 4096

# 融合版 Winograd 的块内 tile 数（5.256）。8 是 A800 上 c3～c24 四档的
# 共同最好值：TT=2/4 时每块线程数不足，一块只有 1～2 个 warp，拉不满 SM
# （c8 的 device 时间 0.0364/0.0240 vs TT=8 的 0.0199）。共享内存按
# ``16*C*TT*4`` 算，C=24 时也只有 12 KB。
CONV_WINOGRAD_TILE = 8

# 融合版覆盖的输入通道区间（5.256）。下限 8：c3 上融合版端到端 0.0631 ms，
# 反而慢于分块直算的 0.0579——第一阶段只有 ``C*TT = 24`` 个线程干活，其余
# 一百多个空转，省下的 device 时间抵不过多出的一次权重变换 launch。
# 上限 24：c32 起 ``U`` 的重复全局读成为瓶颈——c32 的稳态中位数 0.1217
# 已经输给四段版的 0.1046，c64 退化到 0.3016（四段 0.1117）；把 ``U`` 分块
# 搬进共享内存也没能救回来（5.258）。
CONV_WINOGRAD_FUSED_MIN_C = 8
CONV_WINOGRAD_FUSED_MAX_C = 24

# 权重已驻留时的下限（5.257）。常量权重的 ``U`` 只在第一次执行时算，
# 那次 launch 不再计入每轮成本，c3 因此也划算：A800-40GB 上 0.0502 ms，
# 分块直算 0.0580、cuDNN 0.0450（1.12×）。图输入权重时 c3 仍是 0.0633，
# 所以下限的分档必须跟着"权重是不是常量"走，不能一概放宽。
CONV_WINOGRAD_FUSED_RESIDENT_MIN_C = 3

# 隐式 GEMM 的甜点区（A800 实测，5.252）。分界不是像素数，而是 im2col 要
# 物化的元素数 n*ho*wo*K：网格扫描里 <= 2.95M 的 15 个点在隐式上快 7～13%，
# >= 3.61M 的 10 个点全部是 im2col 快 5～42%，与 n / K / ho*wo 各自的取值
# 无关。3.2M 元素 ≈ 12.8 MB 的 f32 中间缓冲，正好卡在两组之间。
# K 的上界另算：K > 400 时隐式内核的内层循环太长，小图上也会输（K=1152 输
# 2.1×），所以甜点区封顶在 400。
CONV_IMPLICIT_MID_K_MAX = 400
CONV_IMPLICIT_MID_WORK = 3_200_000

# cuDNN 快路径的启用门槛（A800 实测，5.252）：卷积本身 0.03～0.09 ms，但走
# 框架要付描述符 / 算法查询 / 独立 bias 核的 host 开销。综合下来 1 GFLOP 以下
# 自研核更快（0.3 GFLOP 的 3x3_c64 上 cuDNN 慢 1.4×），2.4 GFLOP 起快 2.5×。
CONV_CUDNN_MIN_GFLOPS = 1.0

# 每个线程负责的输出通道数上限。摞到 8 会让 block 线程数减半、
# 占用率下降，反而比 4 慢（见 DESIGN 5.16）。
CONV_ROC_MAX = 4

# 每次 staging 覆盖的输入通道数上限：把同步次数从 C 次摊薄到 C/CB 次，
# 同时 smem 仍受预算约束。
CONV_CB_MAX = 8

# 分块扫描（5.155）的自动判据。串行扫描的并行度只有 ``elems``，每个线程沿
# T 一步步推进；分块扫描把它放大成 ``elems * chunks``，代价是两趟通信与一份
# ``2 * chunks * elems`` 的 float scratch。M40 实测（T × elems 全扫）：
#
# - ``elems <= 1024`` 且 ``steps >= 512`` 时稳定占优，最快 25×（T=16384,
#   elems=16 的 3803 us → 190 us），T=512 / elems=1024 也有 1.60×；
# - ``elems >= 4096`` 时一律变慢（T=8192 时 3.0 ms → 3.1 ms，T=16384 /
#   elems=4096 时 6.5 ms → 6.6 ms）：串行那头已经被带宽喂饱，分块多出的
#   两趟读写是纯开销；
# - ``steps <= 256`` 时块数太少、块前缀摊不开：T=256 的四个格子里三格
#   持平或略慢、elems=1024 那格慢 29%（138 us → 179 us），收益落在噪声里
#   甚至为负，因此下限取 512 而不是 256。
#
# 块长 64 在 18 个"分块占优"的格子里赢 11 格（128 赢 5 格、256 赢 2 格），
# 相邻档位差距多在 15% 以内，取 64。
SCAN_CHUNK = 64
SCAN_CHUNK_MIN_STEPS = 512
SCAN_CHUNK_MAX_ELEMS = 2048
# ``scan_last_state`` 不写整段输出，串行那条只有依赖链没有大流量，因此分块的
# 甜点区更宽：elems=4096 仍有 2.2～2.4×（T=8192: 3243 us → 1365 us）。再往上
# （16384）两者打平而分块要多占 16 MB scratch，于是卡在 4096。
SCAN_LAST_MAX_ELEMS = 4096

VEC_WIDTHS = (1, 4)

# ``float4`` 是 16 字节，所以元素索引必须是 4 的倍数、每个指针 16 字节对齐。
# 不满足时向量化是**非法访存**而不是慢路径，因此这个判断必须在 launch 前做。
VEC4_ALIGN = 16


# ---- 数据结构 ------------------------------------------------------------


@dataclass(frozen=True)
class Launch:
    """一次 launch 的完整配置。``vec`` 与 ``shared_bytes`` 也在这里，"""

    block: int
    grid: int
    vec: int = 1
    shared_bytes: int = 0

    def __post_init__(self) -> None:
        check_block(self.block)
        if self.grid < 1:
            raise ValueError(f"grid 至少为 1，收到 {self.grid}")
        if self.vec not in VEC_WIDTHS:
            raise ValueError(f"vec 只支持 {VEC_WIDTHS}，收到 {self.vec}")
        if self.shared_bytes < 0:
            raise ValueError(f"shared_bytes 不能为负，收到 {self.shared_bytes}")
        if self.shared_bytes > SHARED_LIMIT:
            raise ValueError(
                f"shared_bytes {self.shared_bytes} 超过单 block 预算 {SHARED_LIMIT}；"
                "需要另选分块或换实现"
            )


@dataclass(frozen=True)
class ConvTilePlan:
    """分块卷积的线程铺法（W 方向 tile × OC tile，输入通道流式 staging）。"""

    tile_w: int
    block_y: int
    toc: int
    roc: int
    cb: int
    oc_tiles: int
    shared_bytes: int
    # 输入窗口宽度（隐式 GEMM 用）：``tile_w * stride_w + (kw - 1) * dilation_w``。
    # 分块直算不暂存输入，因此是 0——它在这里只是让两条路径共用同一个返回类型。
    span: int = 0

    @property
    def block(self) -> Tuple[int, int]:
        return (self.tile_w, self.block_y)


# ---- 纯函数 --------------------------------------------------------------


def cdiv(x: int, y: int) -> int:
    """向上取整的整数除法。"""
    if y <= 0:
        raise ValueError(f"除数必须为正，收到 {y}")
    return -(-x // y)


def pow2_block(n: int, cap: int = BLOCK_MAX) -> int:
    """不小于 ``n`` 的最小 2 的幂，上限是 ``cap``。"""
    block = 1
    while block < n and block < cap:
        block <<= 1
    return block


def check_block(block: int) -> None:
    """block 必须是 2 的幂且落在硬限制内。非法值在这里就失败。"""
    if isinstance(block, bool) or not isinstance(block, int):
        raise ValueError(f"block 必须是整数，收到 {block!r}")
    if block < BLOCK_MIN or block > BLOCK_MAX or block & (block - 1):
        raise ValueError(
            f"block 必须是 2 的幂且落在 [{BLOCK_MIN}, {BLOCK_MAX}] 内，收到 {block}"
        )


def auto_block(
    work: int,
    *,
    vec: int = 1,
    preferred: int = BLOCK_DEFAULT,
    target_blocks: int = 1,
) -> int:
    """按规模推一个 block（5.65）：``preferred`` 是上限，不是起点。

    两条规则都不含测量结论：

    - **不派空线程**：线程数不超过把活干完所需的 2 的幂。n=100 的工作量
      用 256 个线程，其中 156 个只会走进"越界退出"；
    - **网格铺得开**：线程数不到 ``target_blocks * preferred`` 时，把 block
      缩小到 ``ceil(线程数 / target_blocks)`` 以下的 2 的幂，让网格至少有
      ``target_blocks`` 个块。块数少于一屏 SM 时，其余 SM 整轮空闲——这是
      **占空比**问题（可以差出整数倍），与 5.63 里被整机漂移淹没的那几个
      百分点不是一回事。

    ``target_blocks`` 由调用方按设备给（SM 数）：这一层不查设备，所以它
    接受这个数而不是去猜。默认 1 表示只要"不派空线程"这一条。

    返回的一定是合法的 block（2 的幂，落在 ``[BLOCK_MIN, preferred]``）。
    """
    if work < 0:
        raise ValueError(f"元素数不能为负，收到 {work}")
    if vec not in VEC_WIDTHS:
        raise ValueError(f"vec 只支持 {VEC_WIDTHS}，收到 {vec}")
    check_block(preferred)
    if isinstance(target_blocks, bool) or not isinstance(target_blocks, int):
        raise ValueError(f"target_blocks 必须是整数，收到 {target_blocks!r}")
    if target_blocks < 1:
        raise ValueError(f"target_blocks 至少为 1，收到 {target_blocks}")

    threads = cdiv(work, vec)
    # 不派空线程：所需的线程数向上取整到 2 的幂（0 个元素也留 1）。
    cap = pow2_block(max(1, threads))
    if target_blocks > 1:
        # 想让 grid >= target_blocks，每块线程数的 2 的幂上界就是这个。
        per_block = cdiv(threads, target_blocks)
        floor_pow2 = 1 << (per_block.bit_length() - 1) if per_block else 1
        cap = min(cap, floor_pow2)
    return max(BLOCK_MIN, min(preferred, cap))


def can_vectorize(n: int, *, dtype: str, aligned: bool) -> bool:
    """能否走 4 通道（``float4``）路径。

    三个条件缺一不可：f32（f16 的向量存取是另一套写法）、长度能被 4 整除
    （否则尾部会越界）、每个指针 16 字节对齐（否则是一次非法访存）。
    """
    return dtype == "f32" and n > 0 and n % 4 == 0 and aligned


def grid_for(n: int, *, block: int, vec: int = 1, max_blocks: int = 1024) -> int:
    """一维网格的块数：``n`` 个元素、每块 ``block`` 线程、每线程 ``vec`` 个元素。

    ``max_blocks`` 不是性能参数而是**正确性**参数：内核都写成网格步进，
    块数再多也只是让同一批线程多绕几圈；而块数超过上限会被驱动直接拒绝。
    """
    if n < 0:
        raise ValueError(f"元素数不能为负，收到 {n}")
    if vec not in VEC_WIDTHS:
        raise ValueError(f"vec 只支持 {VEC_WIDTHS}，收到 {vec}")
    if n == 0:
        return 1
    return max(1, min(max_blocks, cdiv(n, block * vec)))


def plan_csr_scan(width: int) -> int:
    """``spike_csr`` 每行的 block 大小：一整个 block 只负责一行。

    行内按 ``blockDim.x`` 分段扫列号，段内做块内前缀和，所以线程数不需要超过
    列数——``width=8`` 时 8 个线程就够（再宽只是空转），``width>=256`` 就按
    默认 block 走。上限 256 而不是 1024：每个线程只做一次比较与前缀和，块再
    宽只会让块内同步的层数变多。
    """
    if width < 1:
        raise ValueError(f"列数至少为 1，收到 {width}")
    return pow2_block(width, 256)


def plan_csr_scan_enabled(steps: int, sm_count: int) -> bool:
    """``spike_csr`` 该不该走块内扫描（5.176）。

    串行铺法一个线程一行：``steps`` 就是并行度，但每个线程有 ``width`` 次迭代，
    靠 ILP 就能藏住访存延迟。只有行数少到每 SM 摊不到 32 个长循环线程时，
    把一行摊给一个 block 才划算；行数再多，"并行"只是换成同步开销。

    M40（24 SM）上的实测拐点在 768 行：256 行及以下并行版快 1.5×～2.2×，
    2048 行及以上串行版快 5%～22%（见 ``benchmarks/spike_csr_scan.py``）。
    """
    if steps < 0:
        raise ValueError(f"步数不能为负，收到 {steps}")
    if steps <= 0:
        return False
    if sm_count < 1:
        raise ValueError(f"SM 数至少为 1，收到 {sm_count}")
    return steps < sm_count * CSR_SCAN_STEPS_PER_SM


def plan_elementwise(
    n: int,
    *,
    block: Optional[int] = None,
    max_blocks: int = 1024,
    dtype: str = "f32",
    aligned: bool = False,
    vectorize: bool = True,
    target_blocks: int = 1,
) -> Launch:
    """逐元素内核的铺法。

    ``block=None`` 表示**按规模自动选**（5.65，见 ``auto_block``）；显式给值
    时就是那个值，与从前一致——调用方明确要一个固定 block，这里不改写它。
    ``target_blocks`` 只在自动模式下生效，由后端按设备给（SM 数）。

    ``aligned=False`` 是刻意的默认值：对齐是运行期才知道的信息，调用方没说
    就等于没说，宁可少用一条向量路径，也不要在内核里制造非法访存。

    注意 grid 的分母是 ``block * vec``：向量化之后每个线程处理 4 个元素，
    再按元素数派线程会浪费 3/4 的线程（它们只会走进"越界退出"）。内核里仍
    保留网格步进，因此尾部不满一个 block 时也不会漏元素。
    """
    if n < 0:
        raise ValueError(f"元素数不能为负，收到 {n}")
    vec = 4 if (vectorize and can_vectorize(n, dtype=dtype, aligned=aligned)) else 1
    if block is None:
        block = auto_block(n, vec=vec, target_blocks=target_blocks)
    grid = grid_for(n, block=block, vec=vec, max_blocks=max_blocks)
    return Launch(block=block, grid=grid, vec=vec)


def plan_reduce(cols: int, *, cap: int = REDUCE_BLOCK) -> int:
    """一行一 block 的归约（sum / max / mean / softmax / layernorm）的 block。

    取 ``cols`` 向上取整到 2 的幂，最多 ``cap``：列数少时少派线程，
    列数多时每个线程串行多走几步。shared 只需要一块 ``REDUCE_SCRATCH``
    大小的暂存区，与列数无关——归约树深度由 block 决定。

    ``cap`` 默认 ``REDUCE_BLOCK``（256）而不是 ``BLOCK_MAX``：一行只有 cols
    个元素，block 铺满 1024 会同时丢掉行级并行度（每 SM 只能驻 2 块）并拉长
    归约树（10 级 barrier）。依据见 ``REDUCE_BLOCK`` 的注释。
    """
    if cols < 0:
        raise ValueError(f"列数不能为负，收到 {cols}")
    return pow2_block(cols, cap)


def plan_reduce_tiled(inner: int, reduce_len: int) -> Tuple[int, int]:
    """非末维归约的 ``(tx, ty)``：``tx`` 沿 inner、``ty`` 切 reduce 轴。

    ``tx`` 取 ``inner`` 向上取整到 2 的幂，上限 ``REDUCE_TILE_X``：一个 warp
    正好覆盖一整段连续 ``inner``（更宽不会让访存更合并，只会少掉 y 方向的
    并行度）。``ty`` 取 ``reduce_len`` 向上取整到 2 的幂，并保证
    ``tx * ty <= BLOCK_MAX``；两者都是 2 的幂，``block_reduce_*_y`` 里按
    ``>> 1`` 折半的归约树才每一级都有效。

    ``ty`` 以 ``reduce_len`` 为上限而不是一律拉满：归约轴很短时（例如长度
    8），多出来的线程只带着单位元参与树归约，纯属多一层同步。
    """
    if inner < 1:
        raise ValueError(f"inner 至少为 1，收到 {inner}")
    if reduce_len < 1:
        raise ValueError(f"归约长度至少为 1，收到 {reduce_len}")
    tx = pow2_block(inner, REDUCE_TILE_X)
    ty = pow2_block(reduce_len, BLOCK_MAX // tx)
    return tx, ty


def attention_block(t_len: int) -> int:
    """融合注意力的 block：至少一个 warp，最多 ``ATTENTION_BLOCK``。

    上界不随序列长度增长——T 方向只影响块内循环次数，不影响并行度。
    """
    if t_len < 0:
        raise ValueError(f"序列长度不能为负，收到 {t_len}")
    return pow2_block(max(32, min(t_len, ATTENTION_BLOCK)), ATTENTION_BLOCK)


def conv_oc_tile(khkw: int, o: int) -> Optional[int]:
    """在 shared 预算内能放下的最大 OC tile（取 2 的幂），放不下返回 None。

    shared memory 只存 **一个输入通道** 的权重（TOC*KH*KW 个 float），
    因此预算随核面积增长，而与输入通道数无关——这是 K 方向按通道分块
    换来的：深通道卷积不会把 shared memory 撑爆。
    """
    for toc in (32, 16, 8, 4, 2, 1):
        if toc <= o and toc * khkw * 4 <= CONV_SHARED_LIMIT:
            return toc
    return None


def conv_tiled_preferred(k: int, work: int) -> bool:
    """auto 模式下是否该走分块核：K 足够小，且输出总量撑得起一次 staging。"""
    return k <= CONV_TILED_MAX_K and work >= CONV_TILED_MIN_WORK


def conv_winograd_supported(
    *, kh: int, kw: int, sh: int, sw: int, dh: int, dw: int
) -> bool:
    """Winograd F(2x2,3x3) 能不能承接这个卷积（纯算术，不碰设备）。

    三个变换的系数矩阵是按 3x3 / stride=1 / dilation=1 推出来的定值，
    没有参数化的版本；分组卷积的每个组看的是不同输入通道，也套不进
    "16 个位置各做一次全通道 GEMM"的形状。这些都由调用方显式拒绝，
    不静默改写用户的 ``conv_impl``。
    """
    return kh == 3 and kw == 3 and sh == 1 and sw == 1 and dh == 1 and dw == 1


def conv_winograd_preferred(*, k: int, kh: int, kw: int, sw: int, dw: int,
                            work: int) -> bool:
    """auto 模式下 3x3 是否该交给 Winograd（A800 实测，5.255）。

    门槛与分块核共用：``k <= CONV_TILED_MAX_K`` 的分块直算在 3x3_c3 上
    仍然更快（0.077 ms 对 0.133 ms），所以不抢；``work`` 太小的一次
    staging 同样不值得付出三次变换的固定开销。

    实测（A800 稳态，f32，NCHW）：3x3_c8/c16/c24/c32/c64/c128 上 Winograd
    0.133～0.136 ms，im2col 0.16～0.31 ms；3x3_c3（K=27）分块 0.077 ms
    仍然最快。K 越大优势越大，因为乘法从 ``9*K`` 降到 ``16*K/4 = 4*K``。
    """
    return (
        kh == 3
        and kw == 3
        and sw == 1
        and dw == 1
        and k > CONV_TILED_MAX_K
        and work >= CONV_TILED_MIN_WORK
    )


def conv_winograd_fused_preferred(*, c: int, kh: int, kw: int, sw: int,
                                  dw: int, o: int, work: int,
                                  min_c: int = CONV_WINOGRAD_FUSED_MIN_C) -> bool:
    """auto 模式下是否该走**融合版** Winograd（A800-40GB 实测，5.256）。

    四段版（输入变换 + 16 个 cuBLAS GEMM + 输出变换）在 A800 上被固定开销
    支配：c3～c24 的稳态几乎都是 0.103 ms，与通道数无关。融合版把三段压进
    一次 launch、``V`` 只留共享内存，端到端（``execute_device``，输入驻留、
    burst 稳态）落在这几档：

    - c8 0.0644、c16 0.0651、c24 0.0661 ms，四段版都是 0.1022～0.1034，
      与 cuDNN 9 的 0.0639 / 0.0635 / 0.0641 基本持平（1.01～1.03 倍）；
    - c3 例外：融合版 0.0631 反而慢于分块直算的 0.0579。第一阶段只有
      ``C*TT = 24`` 个线程干活、其余空转，省下的 device 时间抵不过多出的
      一次权重变换 launch，所以下限设在 8；
    - c32 起融合版的乘加退化：``U`` 沿 ``C`` 的读次数是 ``16*C`` 每线程。
      c32 上三次交替 A/B 的稳态中位数是融合 0.1217、四段 0.1046，c64 是
      0.3016 对 0.1117——所以上限停在 24。把 ``U`` 搬进共享内存试过，
      没有改善（5.258）。

    因此门槛只看 ``C`` 的区间，不看 ``K``。其余几何条件与
    ``conv_winograd_supported`` 相同；``TT*O`` 不能超过单块线程上限。

    ``min_c`` 是下限的分档：权重是**常量节点**时调用方传
    ``CONV_WINOGRAD_FUSED_RESIDENT_MIN_C``——那一次权重变换 launch 被省掉，
    c3 也快于分块直算（5.257）。
    """
    return (
        kh == 3
        and kw == 3
        and sw == 1
        and dw == 1
        and min_c <= c <= CONV_WINOGRAD_FUSED_MAX_C
        and o * CONV_WINOGRAD_TILE <= BLOCK_MAX
        and work >= CONV_TILED_MIN_WORK
    )


def conv_implicit_preferred(*, n: int, k: int, kh: int, kw: int,
                            ho: int, wo: int) -> bool:
    """auto 模式下是否该走隐式 GEMM（只覆盖中等 K 的带窗卷积）。

    判据来自 A800 上的二维网格与 n 扫描（DESIGN 5.252）：隐式除以 im2col
    的耗时比几乎是 ``n*ho*wo*K`` 的单调函数——2.36M～2.95M 元素时 0.87～
    0.93，3.61M 起跳到 1.05 以上，9.44M 时 1.42。所以门槛直接落在这个物化
    规模上，而不是单独给 K 或特征图设档：

    - ``n*ho*wo*K <= 3.2M``（≈12.8 MB f32 中间缓冲）：隐式快 7～13%；
    - 超过就交回 im2col：cuBLAS 的大矩阵调度赢过隐式的一次性 staging。

    两个独立的排除项：

    - ``K > 400``：隐式内核的内层循环太长，小图上也会输（K=1152 输 2.1×、
      K=576+16x16 输 1.10×），所以甜点区封顶；
    - ``kh*kw == 1``：没有窗口重叠，隐式只多一次 staging（慢 6%）。

    ``K <= 32`` 仍归分块直算（``conv_tiled_preferred``），这里不碰。
    """
    if k <= CONV_TILED_MAX_K or kh * kw <= 1 or k > CONV_IMPLICIT_MID_K_MAX:
        return False
    return n * ho * wo * k <= CONV_IMPLICIT_MID_WORK


def conv_implicit_oc_tile(kh: int, kw: int, o: int, span: int) -> Optional[int]:
    """隐式 GEMM 在 shared 预算内能放下的最大 OC tile，放不下返回 ``None``。

    预算是"权重 + 输入窗口"两块之和，``span`` 由调用方按 tile_w 算好。
    与 ``conv_oc_tile`` 分开而不是加参数：分块直算不暂存窗口，两者对同一个
    ``o`` 会给出不同的答案，合在一起会让"谁该用哪个预算"变成调用方的责任。
    """
    for candidate in (32, 16, 8, 4, 2, 1):
        if (
            candidate <= o
            and candidate * kh * kw * 4 + kh * span * 4 <= CONV_SHARED_LIMIT
        ):
            return candidate
    return None


def conv_implicit_fits(
    *,
    kh: int,
    kw: int,
    c: int,
    o: int,
    wo: int,
    sw: int,
    dw: int,
    tile_w: Optional[int] = None,
) -> bool:
    """这个形状能不能被隐式 GEMM 接住（shared 预算与 block 上限一起判）。

    只有 auto 用它：显式 ``conv_impl="implicit"`` 放不下时仍由
    ``plan_conv_implicit`` 报错，不在这里静默改写意图。判据与
    ``plan_conv_implicit`` 的前半段一致，改预算公式时两处要一起看。
    """
    if min(kh, kw, c, o, wo, sw, dw) <= 0:
        return False
    if tile_w is None:
        tile_w = CONV_TILE_W
    if tile_w <= 0:
        return False
    tile_w = min(tile_w, wo)
    span = tile_w * sw + (kw - 1) * dw
    toc = conv_implicit_oc_tile(kh, kw, o, span)
    if toc is None:
        return False
    roc = min(CONV_ROC_MAX, toc)
    return tile_w * (toc // roc) <= BLOCK_MAX


def plan_conv_tiled(
    *,
    kh: int,
    kw: int,
    c: int,
    o: int,
    ho: int,
    wo: int,
    tile_w: Optional[int] = None,
    roc: Optional[int] = None,
    cb: Optional[int] = None,
) -> ConvTilePlan:
    """分块卷积的线程铺法。

    ``toc`` 先按 shared 预算挑一个粗粒度 tile，再按 ``roc`` 规整成
    ``block_y * roc``（block 的 y 维是线程数，必须整除 toc）。规整后的 toc
    可能大于原 tile，但 shared 预算仍按规整后的值校验——这一点在 5.16 里
    是靠注释约定的，现在由这里保证。

    三个可覆盖的旋钮，``None`` 表示用默认推导：

    - ``tile_w``：``W`` 方向线程数，按 ``wo`` 截断（多出来的线程只走边界慢路径）；
    - ``roc``：每个线程算几个输出通道，必须整除 ``toc``，否则 block 的 y 维不整；
    - ``cb``：每次 staging 覆盖几个输入通道，受 shared 预算与 ``c`` 双重约束。

    block 的总线程数是 ``tile_w * block_y``，超过 1024 会被驱动拒绝，因此在这里
    就判掉——否则调优搜索会在某个候选上撞见一个看不出原因的
    ``invalid configuration``。

    放不下时抛 ``ValueError``：调用方要么换实现，要么明确失败。
    """
    if min(kh, kw, c, o, wo) <= 0 or ho <= 0:
        raise ValueError(
            f"分块卷积的维度必须为正，收到 kh={kh} kw={kw} c={c} o={o} "
            f"ho={ho} wo={wo}"
        )
    toc = conv_oc_tile(kh * kw, o)
    if toc is None:
        raise ValueError(
            f"分块卷积的 shared memory 预算放不下一个输出通道"
            f"（核 {kh}x{kw} 需要 {kh * kw * 4} B，预算 {CONV_SHARED_LIMIT} B）"
        )
    if roc is None:
        roc = min(CONV_ROC_MAX, toc)
    else:
        roc = _positive(roc, "roc")
        if roc > toc:
            raise ValueError(f"roc({roc}) 不能超过输出通道 tile({toc})")
        if toc % roc:
            raise ValueError(
                f"roc({roc}) 必须整除输出通道 tile({toc})，否则 block 的 y 维不整"
            )
    # toc 是 2 的幂（conv_oc_tile 的候选集），roc 是它的约数，所以整除；
    # block 的 y 维因此恰好等于 toc / roc，不需要向上规整。
    block_y = toc // roc
    if tile_w is None:
        tile_w = CONV_TILE_W
    else:
        tile_w = _positive(tile_w, "tile_w")
    tile_w = min(tile_w, wo)
    if tile_w * block_y > BLOCK_MAX:
        raise ValueError(
            f"tile_w({tile_w}) * block_y({block_y}) = {tile_w * block_y} "
            f"超过单 block 上限 {BLOCK_MAX}；请调小 tile_w 或调大 roc"
        )
    weight_bytes = toc * kh * kw * 4
    # 每次 staging 覆盖 CB 个输入通道，剩余预算全给输入通道。
    limit_cb = max(1, CONV_SHARED_LIMIT // weight_bytes)
    if cb is None:
        cb = max(1, min(CONV_CB_MAX, c, limit_cb))
    else:
        cb = _positive(cb, "cb")
        if cb > c:
            raise ValueError(f"cb({cb}) 不能超过输入通道数 {c}")
        if cb > limit_cb:
            raise ValueError(
                f"cb({cb}) 超过 shared 预算允许的 {limit_cb} "
                f"（每次 staging 每通道 {weight_bytes} B，预算 {CONV_SHARED_LIMIT} B）"
            )
    return ConvTilePlan(
        tile_w=tile_w,
        block_y=block_y,
        toc=toc,
        roc=roc,
        cb=cb,
        oc_tiles=cdiv(o, toc),
        shared_bytes=weight_bytes * cb,
    )


def plan_conv_implicit(
    *,
    kh: int,
    kw: int,
    c: int,
    o: int,
    wo: int,
    sw: int,
    dw: int,
    tile_w: Optional[int] = None,
    roc: Optional[int] = None,
    cb: Optional[int] = None,
) -> ConvTilePlan:
    """隐式 GEMM 卷积的线程铺法：权重与输入窗口都进 shared。

    与 :func:`plan_conv_tiled` 的区别只有 shared 预算的算法：这里每个输入通道
    还要多放 ``kh * span`` 个 float，其中
    ``span = tile_w * stride_w + (kw - 1) * dilation_w`` 是一个 block 覆盖的
    输入列数。因此选 ``toc`` 时必须把两块一起算——只按权重挑会在宽核上把
    shared 撑爆。

    ``span`` 随 ``tile_w`` 变化，而 ``tile_w`` 可以在调优时被覆盖，所以这里
    返回的计划把它一起算好，内核不再自己推导。
    """
    if min(kh, kw, c, o, wo, sw, dw) <= 0:
        raise ValueError(
            f"隐式 GEMM 卷积的维度必须为正，收到 kh={kh} kw={kw} c={c} o={o} "
            f"wo={wo} sw={sw} dw={dw}"
        )
    if tile_w is None:
        tile_w = CONV_TILE_W
    else:
        tile_w = _positive(tile_w, "tile_w")
    tile_w = min(tile_w, wo)
    span = tile_w * sw + (kw - 1) * dw
    khkw = kh * kw
    input_bytes = kh * span * 4

    toc: Optional[int] = conv_implicit_oc_tile(kh, kw, o, span)
    if toc is None:
        raise ValueError(
            f"隐式 GEMM 卷积的 shared memory 放不下一个输出通道的权重加窗口"
            f"（核 {kh}x{kw}、窗口 {span} 列需要 {khkw * 4 + input_bytes} B，"
            f"预算 {CONV_SHARED_LIMIT} B）"
        )

    if roc is None:
        roc = min(CONV_ROC_MAX, toc)
    else:
        roc = _positive(roc, "roc")
        if roc > toc:
            raise ValueError(f"roc({roc}) 不能超过输出通道 tile({toc})")
        if toc % roc:
            raise ValueError(
                f"roc({roc}) 必须整除输出通道 tile({toc})，否则 block 的 y 维不整"
            )
    block_y = toc // roc
    if tile_w * block_y > BLOCK_MAX:
        raise ValueError(
            f"tile_w({tile_w}) * block_y({block_y}) = {tile_w * block_y} "
            f"超过单 block 上限 {BLOCK_MAX}；请调小 tile_w 或调大 roc"
        )

    per_cb = khkw * toc * 4 + input_bytes
    limit_cb = max(1, CONV_SHARED_LIMIT // per_cb)
    if cb is None:
        cb = max(1, min(CONV_CB_MAX, c, limit_cb))
    else:
        cb = _positive(cb, "cb")
        if cb > c:
            raise ValueError(f"cb({cb}) 不能超过输入通道数 {c}")
        if cb > limit_cb:
            raise ValueError(
                f"cb({cb}) 超过 shared 预算允许的 {limit_cb}"
                f"（每批权重加窗口 {per_cb} B，预算 {CONV_SHARED_LIMIT} B）"
            )
    return ConvTilePlan(
        tile_w=tile_w,
        block_y=block_y,
        toc=toc,
        roc=roc,
        cb=cb,
        oc_tiles=cdiv(o, toc),
        shared_bytes=per_cb * cb,
        span=span,
    )


def scan_auto_chunk(steps: int, elems: int) -> int:
    """完整扫描（前向 / 反向）的自动块长；0 表示保持串行。

    判据只用了两个量：时间步数 ``steps`` 与尾维元素数 ``elems``。串行扫描的
    并行度恰好是 ``elems``——小到填不满设备时，逐时间步的依赖链就是纯延迟；
    大到已经喂饱带宽时，再多并行度也换不来吞吐，只会多付两趟 scratch 流量。
    阈值与实测见 ``SCAN_CHUNK_*`` 的注释。
    """
    if steps <= 0 or elems <= 0:
        return 0
    if steps < SCAN_CHUNK_MIN_STEPS or elems > SCAN_CHUNK_MAX_ELEMS:
        return 0
    return SCAN_CHUNK


def scan_last_auto_chunk(steps: int, elems: int) -> int:
    """``scan_last_state`` 的自动块长；0 表示保持串行。

    与 :func:`scan_auto_chunk` 的唯一区别是 ``elems`` 上限更宽：这条路径不写
    整段输出，串行版没有大流量可压，分块在 ``elems=4096`` 仍有 2.4×。
    """
    if steps <= 0 or elems <= 0:
        return 0
    if steps < SCAN_CHUNK_MIN_STEPS or elems > SCAN_LAST_MAX_ELEMS:
        return 0
    return SCAN_CHUNK


def scan_chunk(
    chunked_scan: Optional[int], steps: int, elems: int, *, last: bool = False
) -> int:
    """把 ``chunked_scan`` 的三种取值解析成**这一次**的块长（0 = 串行）。

    - ``None``：按规模自动（默认），交给 ``scan_auto_chunk`` / ``scan_last_auto_chunk``；
    - ``0``：强制串行，块长 0——给"要逐位可复现"的调用方留的开关；
    - 正数：显式块长，且只有 ``steps`` 超过块长才真的分块（块数 < 2 时
      第一趟只是把整段搬一遍，白花流量）。

    负数与非法类型由构造期拒绝，这里只做算术。
    """
    if chunked_scan is None:
        return (
            scan_last_auto_chunk(steps, elems)
            if last
            else scan_auto_chunk(steps, elems)
        )
    chunk = int(chunked_scan)
    if chunk > 0 and int(steps) > chunk:
        return chunk
    return 0


def _positive(value: int, name: str) -> int:
    """``plan_conv_tiled`` 的旋钮校验：整数且为正。"""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} 必须是正整数，收到 {value!r}")
    return value
