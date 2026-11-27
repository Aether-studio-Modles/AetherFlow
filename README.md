# AetherFlow

统一深度学习基础设施：**一套语义 IR + 一份参考语义 + 可插拔硬件后端**。

同一张图、同一份权重，换一个后端插件就从 NVIDIA 跑到 AMD（以及将来的其它硬件），
数值对齐由一套跨后端一致性套件判定；支持不了的地方**明确失败**，不静默降级。

第一代版本 `v0.1.0`（API 主版本 `1`）：语义 IR、参考执行器、自动微分、训练与推理闭环、
CUDA / HIP 两个真机后端、插件契约与一致性套件均已交付。规范只有一份，
全部历史与实验结论在 [DESIGN.md](DESIGN.md)；发布说明、环境要求与已知限制在 [REMIND.md](REMIND.md)。

---

## 1. 设计约束

1. **前端与后端之间只有一份契约——IR。** 后端不得读取前端对象；新增硬件只写插件，
   不改 IR / verifier / 前端 / 优化。
2. **每个算子都有一份不依赖加速库的参考语义。** 它是跨后端对齐的唯一真值。
3. **每个节点带可移植性标注。** 跨后端共享的优化路径只允许 `PORTABLE` 节点。
4. **不静默降级。** 能力声明没覆盖到的形状 / 精度 / 算子，构图或执行期显式报错。

## 2. 分层

```
前端            torch.export 子集 / EagerGraph 构图糖 / 动态执行（按形状建图、声明动态轴后一次构图多次重放）
                ↓
高层语义 IR     TensorType（含符号维）/ verifier / JSON 序列化 + 版本治理 + 迁移
                ↓
中层优化 IR     DCE / CSE / 交换律 CSE / simplify / 常量折叠 / 3 条融合 / 逐元素链
                ↓
下层内核        L2 语法级 IR（表达式 + 语句 + C 打印器 + 方言）→ NVRTC / hipRTC
                ↓
后端插件        参考（numpy 语义，唯一真值）/ CUDA / HIP
                库复用：cuBLAS、cuSPARSE、cuFFT、hipBLAS（cuDNN / MIOpen 作对照物）
```

插件契约：必需面 `capabilities` + `execute`；可选契约 `device` / `device_manager` /
`codegen` / `tuning` / `memory` / `stream` / `allocator`。第三方包可用 pip entry point
（组 `aetherflow.backends`）注册，装包即被发现。

## 3. 快速开始

```bash
pip install -e .            # 只依赖 numpy；CUDA 需要 cupy，前端需要 torch
python demo_transformer_block.py
```

```python
import numpy as np
from aetherflow import Graph, ReferenceBackend, f32, verify

g = Graph("add")
x = g.input("x", f32[2, 3])
y = g.input("y", f32[2, 3])
g.output(g.add(x, y), "sum")

assert not verify(g)                       # verifier 无 issue
out = ReferenceBackend().execute(g, {
    "x": np.ones((2, 3), dtype=np.float32),
    "y": np.full((2, 3), 2.0, dtype=np.float32),
})["sum"]
```

换后端只改装配：`get_backend()` 自动挑可用后端，或 `AETHERFLOW_BACKEND=hip` 显式指定；
图与权重不用改一行。

### 训练与推理

```python
from aetherflow.train import Trainer, DistributedDataParallel, PipelineTrainer
from aetherflow.models import BlockSpec, build_transformer_block, build_mamba_model, build_snn
from aetherflow.models.lm import LMRunner, ServingEngine
from aetherflow.serve import OpenAIService, serve
```

推理服务：`serve(OpenAIService(ServingEngine(backend, params, spec, capacity=4)))`
即得到 OpenAI 兼容的 `/v1/completions` / `/v1/chat/completions`（含 SSE 流式）；
分词与模板默认取 `ByteTokenizer` / `ChatTemplate`，也可注入。
`api_key="sk-..."` 打开 Bearer 认证（`/healthz` 免认证），
`rate_limit=60, rate_window=60.0` 打开滑动窗口限流（超限 429 + `Retry-After`）。
`stop` 参与调度：命中后取消该请求、腾出槽位，流式会压住可能是 stop 开头的尾部。

支持的可训练构件：MLP、CNN（conv / 转置卷积 / 分组 / depthwise / 池化 / BN）、
RNN（RNN / LSTM / GRU）、多层 Transformer、Mamba block（`scan_linear`）、多层 SNN
（LIF + 替代梯度 + 稀疏事件）。

### 自检

```bash
python -m unittest discover -s tests -t .   # 本地必须用 Python 3.14（3.10 的 torch 会被兼容矩阵拦下）
python verify_cuda.py                       # 有 NVIDIA 卡时验证 CUDA 后端
python -m benchmarks.run --backend ref      # 性能基准（另有 hip / cuda）
python -m benchmarks.framework_compare --device cuda   # 与 torch eager / compile 的同机对照
```

当前基线：**3026 个用例**，本地 460 跳过（Python 3.14 + torch 2.13）全绿；
WSL HIP（RX 7650 GRE / ROCm 7.2）本轮只复跑了后端与控制流相关套件（见第 6 节）。

## 4. 第一代版本的能力面

| 面 | 已交付 |
|---|---|
| 前端 | `torch.export` 导入：卷积 / 池化 / linear / LayerNorm / softmax / SDPA / 归一化 / 索引与切片之外，P7 补上了逐元素数学、激活展开、clamp 与标量运算、dtype 转换、归约（mean / sum / amax / max.dim / argmax）、tril / triu / cumsum / pad、栈与切分、mm / mv / dot / addmm、常量构造（zeros / ones / full / arange 等）；不能精确表达的算子显式拒绝（见第 6 节） |
| 算子 | IR 里 72 个算子；CUDA / HIP 各自承接 69 个（68 个自研内核 + 1 个由反量化与 GEMM 组合的 `linear_i8`，其中 W8A8 的 `linear_i8a8` 是 int8 读入、int32 累加的专用内核，5.309），CUDA 另有 cuFFT 的 fft/rfft；参考后端承接全部 |
| 精度 | f32 全覆盖；f16 覆盖 64 个（68 个设备端算子去掉 3 个 int8 语义算子 `quantize_i8` / `dequantize_i8` / `linear_i8` 与整数 `mod`，它们的数据类型由语义决定；attention 为混合精度：存储半精度、累加 float）；**bf16** 覆盖逐元素 / 搬运 / 归约 / softmax / layernorm / matmul（32 个，`backends/cuda/backend.py` 的 `BF16_OPS`），设备侧是**真 16 位存储**——`uint16` 位模式，存取是纯位操作（不依赖 sm_80 的 `cvt`），GEMM 走 cuBLAS `CUDA_R_16BF`（A800 2048³ 实测 90.5 TFLOPS，f32 的 6.6×）；卷积 / 池化 / scan / sparse 的 bf16 尚未验证，明确拒绝；f64 / 复数部分承接；weight-only int8 权重（计算仍在 f32，见下） |
| 自动微分 | 49 条 VJP 规则，反向图是可序列化、可执行的普通图；二阶导、激活重算、反向图裁剪 |
| 控制流 | `Conditional` / `WhileLoop` 表达为子图关系（每次判定回主机）；`DeviceWhileLoop` / `DeviceConditional` 把条件、分支各捕获一次、状态与输出常驻显存，但每次判定仍要读一次谓词（无 device-side 真分支，且实测不比宿主路径快，见 DESIGN 5.245 / 5.277）；推理解码循环接不进这套形态——判定在主机采样之后（5.278），但 decode 的**状态**可以常驻（5.279） |
| 多卡 | 设备枚举 / 切换、跨卡拷贝、NCCL / RCCL 设备侧集合通信绑定（AllReduce / AllGather / ReduceScatter / Broadcast）、数据并行（默认主机中转归约；单进程多卡可显式接设备侧归约，`nccl_device_collective(backend, devices)`，见下）、ZeRO-1/2/3 分片、GPipe / 1F1B 流水、张量并行（列 / 行并行 + attention 头切分，仅前向） |
| 推理 | 固定块预填充、跨请求图缓存与权重驻留、前缀缓存、批量解码、**设备态解码**（`LMRunner(device_state=True)` / `ServingEngine(device_state=True)`：decode 状态常驻显存、每步只取回 logits，DESIGN 5.279～5.281；空闲 A800 24 层 81.2 → 8.2 ms/token，连续批处理 cap=8 时 66.1 → 491.6 tok/s）、束搜索、连续批处理、采样面（top-p / 重复惩罚）、分页 KV cache + 增量注意力解码（含换出 / 换入、LRU 调度与共享前缀 `fork` / 写时复制：`PagedCacheManager` / `PagedDecodeScheduler`）、分页 KV 上的贪心生成闭环（`PagedGenerator`：token 进 token 出，5.310）、贪心投机解码、OpenAI 兼容 HTTP 接口（`aetherflow.serve`：`/v1/completions`、`/v1/chat/completions`、SSE 流式） |
| 调优 | 与后端无关的参数空间 / 搜索 / 配置库；逐元素 block、归约 block、conv2d 的 tile/roc/cb；roofline 式成本模型（六项分解 + 可行性剪枝）先算再测，按估算排序只测前 N 个 |
| 工程 | conformance 套件、模糊测试、边界值用例、趋势聚合、Chrome Trace 导出（`python -m aetherflow.runtime.trace telemetry.json`）、NHWC 布局传播（`LayoutPropagation`，仅参考后端承接）、CI 四作业（含 wheel 构建与安装冒烟）、部署包、zip 硬件包 |

## 5. 性能现状（同机口径，不跨机外推）

性能有三种口径，混用会把 PCIe 带宽当成算力（见 [DESIGN 5.251](DESIGN.md)）：

- **端到端**：`execute()`，含输入 H2D 与输出 D2H；
- **设备侧**：`execute_device()`，输出不回传（基准加 `--device-outputs`）；
- **纯算**：输入常驻设备，只计内核。

训练同机对照：`python -m benchmarks.framework_compare --training --graph-mode auto`。
默认使用标准反向优化管线，`--raw-backward` 保留原始图作消融；计时前核对完整
损失、全部参数梯度和一步更新。输出是设备驻留调用的逐轮耗时与中位数，包含
Python 派发、接口内部复制及同步；训练暂不支持 `--compile`。共享卡上的耗时
和比值都可能受干扰。2026-11-21 A800 空闲窗口两批复测：本例优化图普通路径
1.073～1.098 ms、自动图 0.509～0.536 ms，同轮 torch eager 约 0.583～0.597 ms。
这只覆盖两层 MLP，不代表其他模型或 torch.compile（完整口径见 DESIGN 5.288）。

A800-SXM4-40GB（CUDA 12.9 / cupy 14.2 / `graph_mode="off"`，单位 ms）：

| 用例 | 端到端 | 设备侧 | 裸 cupy 常驻 |
| --- | --- | --- | --- |
| matmul_1024 | 2.13 | 1.16 | 0.19 |
| maxpool2x2 | 3.50 | 2.49 | — |
| transformer_block | 1.76 | 1.66 | — |
| cnn_lenet | 0.85 | 0.80 | — |
| relu（4M 元素） | 1.98 | — | 0.05 |

单次 `execute` 有约 0.1～1 ms 的固定成本，最大一项是 4 MB 输入的 H2D（实测
8 MB / 0.9 ms ≈ 9 GB/s PCIe）；设备侧口径下 `matmul_1024` 从 2.13 降到 1.16 ms。
纯算内核与裸 cupy 同量级（GEMM 1.16 ms 中 0.19 ms 是 cuBLAS，其余是输入上传）。

卷积的差距要按"稳态 + 驻留输入"口径读（DESIGN 5.253）：5.251 的 3～15× 混进了
每轮 H2D 与 0.08 ms 的单次延迟。修正后低 GFLOP 形状（3x3_c3、1x1_c32、
1x1_c256）与 cuDNN 同量级甚至更快（1x1_c32 是 0.72×），真实差距只在宽通道
3x3/5x5：im2col 稳态为 cuDNN 的 1.4～3.7×（3x3_c64 是 3.71 对 11.92 TFLOPS），
属于内核结构差距而非调度开销。`conv_impl="auto"` 按形状选实现（DESIGN 5.252）：
K 在 (32, 400] 且 im2col 物化缓冲 `n*ho*wo*K` 不超过 3.2M 元素时走隐式 GEMM
（快 7～13%），1 GFLOP 以上交给 cuDNN；A800 上 auto 与同形状最优实现之比为
1.00～1.05（小/中）与 0.99～1.03（大）。隐式核的空间寄存器分块（每线程 2 个
输出位置）在 A800 上 10 个形状无一获益、最差慢 29%，已按实测回退
（DESIGN 5.254）——当前瓶颈不是 shared 带宽。

`conv_impl="winograd"` 是 5.255 的第三条 3x3 路径（F(2x2,3x3)：三次变换 +
16 路批量 GEMM，乘法量降到 4*K），5.256 在其上加了**融合版**：`V` 留共享
内存、`M` 留寄存器，输入变换 + 乘加 + 输出变换压进一次 launch。5.257 再把
权重变换挪出每轮执行：权重来自 IR 的 `constant` 节点时 `U = G g G^T` 随执行
计划驻留；权重是图输入时仍每轮重算（不按指针猜"值没变"）。新机 A800-40GB
上 3x3_c8/c16/c24 的稳态 0.029～0.048 ms，为 cuDNN 9 的 0.45～0.75×
（5.256 时 0.064～0.066、1.00～1.02×）。5.260 又把静态 conv 的解析、线程
铺法与 launch 参数挪到计划期，这一列从 0.050～0.051 再降到 0.029～0.048，
其中 c8/c16 的 20 µs 级框架税被去掉；预热后连跑 20 次，常量权重的权重
变换 0 次、图输入 20 次（`DeviceBackend.conv_weight_transforms`）。数值与
四段版逐位相等，对 cuDNN 归一化误差 ≤ 5.3e-7。auto 的接管面按 `C` 与权重
来源分档（`plan.conv_winograd_fused_preferred`）：8 ≤ C ≤ 24 走融合；权重是
常量时下限放到 C=3（0.028 vs tiled 0.059），否则 C ≤ 7 仍分块直算；C ≥ 32
改走四段版。5.268 在空闲机上按同进程交替重测，`auto` 在 c32/c64 上分别是
cuDNN 9 的 **1.02×/1.36×**（0.0655/0.0887 对 0.0639/0.0654 ms），且是全实现
里最快的；旧表里的 1.80×/4.27× 是显式 `conv_impl="winograd_fused"` 的对照
列（融合版只接管到 C=24），不代表 auto。5.264 把四段版的
launch 也在计划期展开（三段 `grid`/`block`、静态参数、内核句柄一次算好），
c64 单层端到端 0.113 → 0.094 ms（空闲时；共享机上有负载时 `burst` 会到
0.21，稳定可复现的是 `host` / `gpu` / `kernel` 0.062 / 0.088 / 0.042）、
`kernel` 口径 0.085 → 0.042 ms；`U` 也随计划驻留，运行期只做输入变换 +
GEMM + 输出变换
（`benchmarks/overhead_compare.py`）；主机成本已不再与内核同量级，c64 剩下
的 1.36× 在 GEMM 本身（减少乘法量或 TF32 才能再压，见 DESIGN 5.268）。把 `U` 分块进共享内存
试过（5.258），c24 反而慢 1.9 倍、c32/c64 持平——`U` 的重复读本来就被
L1/L2 吸收。
复现：`python -m benchmarks.winograd_compare --rounds 3`（前两列是显式融合版，
auto 的实际比值看 `benchmarks.conv_compare`）。

- 层 / 网络级（5.259～5.266）：`conv → mul → add → relu` 叠 8 层，
  `mul+add+relu` 被合成一个内核（64 节点 → 48 组）。5.260 把静态 conv 的
  计划期开销去掉、5.261 把融合链的推导也挪到计划期，8 层 `scalar` 从
  0.919 ms 降到 **0.298 ms**（每层 37 µs：19 µs 卷积 + 18 µs 提交），
  4 层从 0.460 降到 0.157 ms。5.266 修掉了捕获失败：`capture` 在捕获流开始
  之前重建计划、把常量 H2D 留在默认流，`graph_mode="auto"` 现在真正捕获成功
  （与 `off` 逐位一致），8 层 `scalar` 0.338 → **0.201 ms**（1.68×）。5.262 把逐通道
  BN 的 `(1,C,1,1)` / `(C,1,1)` 广播也接进融合链（向量路径按 lane 单独取
  索引），8 层 `channel4d` 从 0.905 降到 **0.303 ms**；5.263 再把常量上的
  `reshape`（BN 参数由 `(C,)` 变 `(1,C,1,1)`）在计划期折成视图，`channel`
  从 1.022 降到 **0.302 ms**。三种形态现在齐平（差 ≤ 2%）；`channel` 形态的
  捕获收益待共享机空闲后与 cuDNN 基准列一起复测。5.267 又把 `clamp` 的上下界
  与 `step` / `spike` 的阈值（都是属性里的编译期常量）内联进链，SSM 的
  `clamp → mul → add` 与 SNN 的 `step → mul` 不再各写一次内存（NaN 仍透传）。
  5.270 再把整条链内联进 conv 输出变换的**写回点**：`conv → mul → add → relu`
  变成一次内核、`Y` 不再落盘（48 → 40 组），8 层 c16 从 0.306 降到
  **0.185 ms**（1.65×）、c32 从 0.607 降到 **0.473 ms**（1.28×）；融合路径与
  "关掉合并"逐位一致。范围是 f32 + Winograd（f16/bf16 需先舍入、im2col/tiled
  没有集中写回点）。复现：`python -m benchmarks.conv_graph_compare --blocks 8 --rounds 3`。
- 图重放：小算子的多节点图收益显著（HIP 上 `transformer_block` 5.90 → 0.83 ms；
  A800 设备侧 1.66 → 0.40 ms，LeNet 常驻 0.59 → 0.14 ms），单节点图无收益；
  默认 `graph_mode="off"`，语义差异需使用者知情。
- cuDNN 快路径的 f32 数学类型锁死为 FMA（关掉 TF32）：此前宽通道卷积会被
  悄悄降到 TF32、与参考语义差 5e-3（DESIGN 5.251）。
- 设备态解码（5.279～5.281）：24 层 Mamba（`d_model=1024`、`vocab=32000`）
  逐 token 解码，空闲 A800 上宿主 81.2 → 设备态 **8.2 ms/token**（9.89×）；
  RX 7650 GRE 上 8 / 12 / 24 层分别为 7.25× / 7.18× / 7.88×，逐 token 输出一致。
  连续批处理按容量扫描（`capacity=1/2/4/8`，32-token prompt、每条 32 token，
  每步状态 3.56～28.50 MiB）：宿主 77.3 / 95.7 / 102.0 / 117.3 ms/步，设备态
  5.5 / 8.2 / 9.5 / 12.1 ms/步，吞吐 12.5 / 20.2 / 38.0 / 66.1 →
  114.3 / 169.0 / 300.3 / **491.6 tok/s**（7.4～9.2×）。宿主每步成本几乎与行数
  无关（被整块状态往返拖住），设备态的边际行成本 0.93 ms，所以解码加速比随容量
  收窄、吞吐加速比稳定。设备路径吃 SM 时钟与同卡其它进程：同一段代码在 91%～
  100% 外部占用下测到 22.8～25.7 ms/token，宿主侧只差 6%，**设备态数字必须在
  空闲卡上取**。省下的是状态往返，留下的是每步 `capacity × 128 KB` logits；
  采样、停止判定与 `ids` 写回仍在主机，所以这不是"设备端全流程"的数字。
  束搜索的**状态重排**也搬到了显存，并进一步并进 decode 图（5.282 / 5.283）：
  状态输入是 `embedding(上一状态, parents)`，每步一次 replay，父行号只有 k 个
  int32 上传。**同机同口径**对照 A800 `num_beams=1/2/4`：14.818 / 21.947 /
  33.164 → **14.010 / 21.032 / 31.662 ms/步**（−5.5% / −4.2% / −4.5%），
  同机宿主为 92.9 / 117.4 / 128.6 ms/步（6.6× / 5.6× / 4.1×）。合并计划的收益
  小于预期，因为成本主体是每个 `embedding` 的越界守卫（每次 replay 后逐个
  `item()`），不是 replay 边界；下一步是汇总守卫。注意 5.282 文档里的宿主列
  （12.98 等）不在这一口径内，跨条目不可横比。HIP 上 5.282 的加速为
  1.09 / 1.08 / 0.98×，`num_beams=4` 当时不该开设备态。
- 调优：多轮采集显示"最优 block"在采集间不可复现，配置库只在同机同卡命中；
  成本模型（5.242）只给排序依据，不是性能预测器。

## 6. 与主流框架的边界

这不是通用训练框架，能力面按"语义 IR + 多后端 + 一致性"这个定位裁剪。相对
PyTorch / JAX / ONNX Runtime / TVM / vLLM 的主要差距：

- **集合通信**：有 NCCL / RCCL 的 ctypes 绑定（含 RCCL 单 rank 真机验证），但**没有接进数据并行路径**：
  跨卡归约仍经主机中转、单进程串行切卡，无 P2P；梯度已按 `bucket_bytes` 拼桶、桶间线程并发提交
  （200 个 2 KB 参数：65.4 → 28.5 ms/步，2.30×），但**没有**"梯度一就绪就归约"的反向重叠；
- **并行策略**：有张量并行的切分与对齐（列 / 行并行 + attention 头切分，仅前向、归约走主机缓冲），
  但无列并行 / 行并行的反向通信，也没有"一份参数按 FSDP 式跨进程切分 + 按需拼回"的完整闭环；
- **编译**：无 loop 级 IR，schedule 搜索只有离线的成本模型与按估算排序的串行搜索（未接进编译流水线），
  融合是规则表而非通用策略；
- **布局**：有 NHWC 传播 pass（conv / pool 的 `layout` 属性 + 区域边界转置）与后端布局声明，
  但只有参考后端承接 NHWC，设备后端显式拒绝（无原生 NHWC 内核），也没有 GEMM epilogue；
- **量化推理**：weight-only int8（`quantize_i8` / `dequantize_i8` / `linear_i8`，
  权重驻留 1 字节、GEMM 仍走 f32 反量化）与 W8A8（`linear_i8a8`：int8 激活 ×
  int8 权重在 int32 里累加后再乘两个 scale，CUDA 的内核走 `__dp4a`、HIP 走标量
  乘加，两者累加逐位相同；`quantize_activation` 按 token 量化激活）；此外无 fp8、
  无 QAT、无非对称零点，也没有 int8 tensor core 级的矩阵乘；
- **动态性**：主机态的执行与训练支持符号批维（一次构图跑不同 batch，归约反向的计数在
  运行期按实际形状算）；设备态的捕获计划、分段重算与需要合并维度的广播反向仍要求静态形状；
  控制流每次判定回主机；`DeviceWhileLoop` / `DeviceConditional` 只是把条件、分支
  各捕获一次、状态与输出留在显存，每次判定仍要读一次谓词，没有 device-side 真分支
  或 Graph 条件节点（实测不比宿主路径快，见 DESIGN 5.245 / 5.277）；推理解码的**分支**
  仍接不进这套形态（判定输入是主机采样出的 token，每步还要把它写回 `ids`，
  5.278），但 decode 的**状态**已能常驻：`LMRunner(device_state=True)` /
  `ServingEngine(device_state=True)` 捕获一次 `steps=1` 的 decode 图，每步只取回
  logits（5.279 / 5.280）。它要求 `resident=True` 与静态形状（换 batch 重建计划）；
  空闲 A800 24 层实测 81.2 → 8.2 ms/token（9.89×）、连续批处理 `capacity=8` 时
  66.1 → 491.6 tok/s（7.4×）、RX 7650 GRE 8～24 层 7.2～7.9×；设备态数字受同卡
  其它进程影响（5.281）；束搜索的状态重排已并进 decode 图（5.282 / 5.283，
  A800 同机同口径下 `num_beams=1/2/4` 各快 5.5% / 4.2% / 4.5%；5.282 的 HIP
  `num_beams=4` 打平），投机解码仍走宿主；
- **服务化**：分页 KV cache 与分页生成闭环（`PagedGenerator`）**未接进 `LMRunner`**
  ——主干是 Mamba，没有随上下文增长的 KV；投机解码已接进 `LMRunner`
  （`generate_speculative`，5.303，只做贪心验收与 batch=1）。分页缓存已有换出 /
  换入与 LRU 调度（`PagedCacheManager` 在容量不足时换出最久未使用且未 pin 的序列、
  访问时自动换回；`PagedDecodeScheduler` 交错解码多序列，`suspend` / `resume` 对应
  解除保护与换入；`fork` 让同前缀的多条序列共享物理页、写入时按页拆分，块级引用
  计数），`PagedGenerator`（5.310）在它之上补了 token→token 的贪心生成闭环，但仍
  是主机态组件、无真机 D2H 与拆页实测，`fork` 也不做前缀内容匹配；已有 OpenAI
  兼容子集（`/v1/completions`、`/v1/chat/completions`、SSE；`logprobs` / `n>1` /
  工具调用显式 400），静态单密钥 Bearer 认证与进程内滑动窗口限流（5.285），
  `stop` 命中后取消请求并腾出槽位；没有多副本与按密钥计量的集群配额；
- **生态**：只吃 `torch.export` 的 aten 子集，无 define-by-run，无采样 profiler（只有把已有遥测导成 Chrome Trace 的离线路径）；
  前端覆盖算子 `torch.export` 的常用面（逐元素数学 / 激活 / 归约 / 形状 / 常量构造 /
  比较 / `where` / 取整 / `topk`），
  但 `masked_fill`、
  `copy_`、图内 `clamp` 边界、非零或 reflect 的 `pad`、非整数 `pow` 指数都**显式拒绝**，
  不会退化成近似实现；
- **硬件**：NVIDIA sm_75 与 AMD gfx1100 / gfx1102 各一张卡，无 Intel / Apple / 移动端。
  WSL 上的 HIP 需要 `HSA_ENABLE_DXG_DETECTION=1`，否则 `hsa_init` 失败、设备数报 0。

反过来，主流框架没有的差异化能力：**SNN 的替代梯度与事件稀疏**、**SSM 的 scan**、
**跨厂商同一套 IR 的逐项一致性验证**、**IR 的版本治理与跨版本迁移**、**插件契约分层**。

## 7. 仓库结构

```
aetherflow/           包本体
  ir/                 类型、Graph、verifier、序列化与迁移、自动微分、控制流
  passes/             优化 pass 与管线
  lower/              执行计划纯函数、L2 内核 IR、索引 / 布局、方言转译
  backends/           参考 / CUDA / HIP、契约、注册表、硬件包
  runtime/            执行器、图计划、arena、事件池、遥测
  train/  infer/      训练闭环 / 推理会话与服务
  models/             MLP / CNN / RNN / Transformer / Mamba / SNN 构件
  frontend/           torch.export 适配
benchmarks/           固定用例、基线、趋势
tests/                3026 个用例
DESIGN.md             唯一规范与全部实验结论（含 250 余节实验记录）
REMIND.md             发布说明、环境要求与已知限制
```

## 8. 许可

[Apache License 2.0](LICENSE)。允许商用与闭源分发；修改、分发需保留版权与许可声明，
并附带 NOTICE（若存在）与许可证全文。许可证含专利授权与商标条款。
