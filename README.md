# AetherFlow

统一深度学习基础设施：**一套语义 IR + 一份参考语义 + 可插拔硬件后端**。

同一张图、同一份权重，换一个后端插件就从 NVIDIA 跑到 AMD（以及将来的其它硬件），
数值对齐由一套跨后端一致性套件判定；支持不了的地方**明确失败**，不静默降级。

第一代版本 `v0.1.0`（API 主版本 `1`）：语义 IR、参考执行器、自动微分、训练与推理闭环、
CUDA / HIP 两个真机后端、插件契约与一致性套件均已交付。规范只有一份，
全部历史与实验结论在 [DESIGN.md](DESIGN.md)；上手须知与踩坑在 [REMIND.md](REMIND.md)。

---

## 1. 设计约束

1. **前端与后端之间只有一份契约——IR。** 后端不得读取前端对象；新增硬件只写插件，
   不改 IR / verifier / 前端 / 优化。
2. **每个算子都有一份不依赖加速库的参考语义。** 它是跨后端对齐的唯一真值。
3. **每个节点带可移植性标注。** 跨后端共享的优化路径只允许 `PORTABLE` 节点。
4. **不静默降级。** 能力声明没覆盖到的形状 / 精度 / 算子，构图或执行期显式报错。

## 2. 分层

```
前端            torch.export 子集 / EagerGraph 构图糖 / 动态执行（按形状建图并复用）
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
```

支持的可训练构件：MLP、CNN（conv / 转置卷积 / 分组 / depthwise / 池化 / BN）、
RNN（RNN / LSTM / GRU）、多层 Transformer、Mamba block（`scan_linear`）、多层 SNN
（LIF + 替代梯度 + 稀疏事件）。

### 自检

```bash
python -m unittest discover -s tests -t .   # 本地必须用 Python 3.14（3.10 的 torch 会被兼容矩阵拦下）
python verify_cuda.py                       # 有 NVIDIA 卡时验证 CUDA 后端
python -m benchmarks.run --backend ref      # 性能基准（另有 hip / cuda）
```

当前基线：**2602 个用例**，本地 412 跳过、WSL HIP（RX 7650 GRE / ROCm 7.2）371 跳过，全绿。

## 4. 第一代版本的能力面

| 面 | 已交付 |
|---|---|
| 算子 | IR 里 63 个算子；CUDA / HIP 各自承接 60 个（59 个自研内核 + 1 个由反量化与 GEMM 组合的 `linear_i8`），CUDA 另有 cuFFT 的 fft/rfft；参考后端承接全部 |
| 精度 | f32 全覆盖；f16 覆盖 57 个（attention 为混合精度：存储半精度、累加 float）；f64 / 复数部分承接；weight-only int8 权重（计算仍在 f32，见下） |
| 自动微分 | 41 条 VJP 规则，反向图是可序列化、可执行的普通图；二阶导、激活重算、反向图裁剪 |
| 控制流 | `Conditional` / `WhileLoop` 表达为子图关系（每次判定回主机） |
| 多卡 | 设备枚举 / 切换、跨卡拷贝、数据并行（主机中转归约）、ZeRO-1/2/3 分片、GPipe / 1F1B 流水、张量并行（列 / 行并行 + attention 头切分，仅前向） |
| 推理 | 固定块预填充、跨请求图缓存与权重驻留、前缀缓存、批量解码、束搜索、连续批处理、采样面（top-p / 重复惩罚）、分页 KV cache + 增量注意力解码、贪心投机解码 |
| 调优 | 与后端无关的参数空间 / 搜索 / 配置库；逐元素 block、归约 block、conv2d 的 tile/roc/cb；roofline 式成本模型（六项分解 + 可行性剪枝）先算再测，按估算排序只测前 N 个 |
| 工程 | conformance 套件、模糊测试、边界值用例、趋势聚合、CI 三作业配置、部署包、zip 硬件包 |

## 5. 性能现状（同机口径，不跨机外推）

- 整图：`transformer_block` 在 HIP 上与 CUDA 参考实现差约 **8%**，主体是逐节点提交开销；
  图重放已交付（默认 `graph_mode="off"`，语义差异需使用者知情）。
- 卷积：自研分块核 / im2col 比 cuDNN 慢 **4～12×**（M40），比 MIOpen 慢 **4.52～37×**（RX 7650 GRE）。
- HIP 上中等通道 3x3/5x5 走隐式 GEMM，比 im2col 快 **1.33～1.63×**；CUDA 上结论相反，按厂商开关隔离。
- 调优：多轮采集显示"最优 block"在采集间不可复现，配置库只在同机同卡命中；尚未形成可复现的自动调优结论。
  成本模型（5.242）只在 conv tile 与逐元素 block 上给出**排序依据**，不是性能预测器：
  它能回答"哪些组合不必测"，单条估算与实测仍可能差数倍。

## 6. 与主流框架的边界

这不是通用训练框架，能力面按"语义 IR + 多后端 + 一致性"这个定位裁剪。相对
PyTorch / JAX / ONNX Runtime / TVM / vLLM 的主要差距：

- **集合通信**：无 NCCL / RCCL 绑定，数据并行经主机中转，单进程串行切卡，无 P2P；
- **并行策略**：有张量并行的切分与对齐（列 / 行并行 + attention 头切分，仅前向、归约走主机缓冲），
  但无列并行 / 行并行的反向通信，也没有"一份参数按 FSDP 式跨进程切分 + 按需拼回"的完整闭环；
- **编译**：无 loop 级 IR，schedule 搜索只有离线的成本模型与按估算排序的串行搜索（未接进编译流水线），
  融合是规则表而非通用策略；
- **布局**：全 NCHW，无布局传播，无 GEMM epilogue；
- **量化推理**：只有 weight-only int8（`quantize_i8` / `dequantize_i8` / `linear_i8`，
  权重驻留 1 字节、GEMM 仍走 f32 反量化），无 int8 张量核累加、无 fp8、无 QAT、无非对称零点；
- **动态性**：形状必须静态（符号维只做多分辨率），控制流每次判定回主机；
- **服务化**：分页 KV cache 与投机解码只有组件级交付（主干是 Mamba，没有增长 KV，
  二者未接进 `generate` / `LMRunner`；投机解码只做贪心验收与 batch=1），无 OpenAI 兼容 API；
- **生态**：只吃 `torch.export` 的 aten 子集，无 define-by-run，无视觉化 profiler；
- **硬件**：NVIDIA sm_75 与 AMD gfx1100 各一张卡，无 Intel / Apple / 移动端。

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
tests/                2602 个用例
DESIGN.md             唯一规范与全部实验结论（含 250 余节实验记录）
REMIND.md             再进来时先要知道的事
```

## 8. 许可

仓库暂未附许可证文件；在明确授权方式之前，默认保留全部权利。
