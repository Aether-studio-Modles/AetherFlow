# AetherFlow 发布说明

按版本记录**对外可见**的变化：能力面、兼容性、已知限制与升级注意事项。
设计规范、逐项验收与实现历史在 [DESIGN.md](DESIGN.md)，能力速览在 [README.md](README.md)。

当前版本：`v0.1.0`（API 主版本 `1`）

---

## v0.1.0

首个可用版本。一条闭环打通：静态图（或 torch.export 子集）→ 语义 IR → 中层优化 →
可插拔后端（参考 / CUDA / HIP）→ 训练（反向、优化器、分布式）→ 推理（会话、前缀缓存、
连续批处理）。核心命题"一套 IR 落到不同硬件"已在 NVIDIA `sm_70` 与 AMD `gfx1100`
上逐项对齐。

### 亮点

- **一份语义 IR，多个硬件后端**：67 个 IR 算子，CUDA / HIP 各自承接 64 个；
  新增硬件只需写插件，不改 IR、verifier、前端与优化。
- **一份参考语义**：每个算子都有不依赖加速库的 numpy 定义，是跨后端对齐的唯一真值；
  `conformance` 套件按算子逐项比对，支持不了的地方显式失败，不静默降级。
- **自动微分**：47 条 VJP 规则，反向图是普通可序列化图；支持二阶导、激活重算、反向图裁剪。
- **训练**：MLP / CNN / RNN / Transformer / Mamba / SNN 构件；数据并行（主机中转）、
  ZeRO-1/2/3、GPipe 与 1F1B 流水、张量并行（仅前向）。
- **推理**：跨请求图缓存与权重驻留、前缀缓存、批量解码、束搜索、连续批处理；
  分页 KV cache 与贪心投机解码（组件级，见"已知限制"）。
- **调优与可观测**：与后端无关的参数空间与搜索、conv / 逐元素 / 归约的调优旋钮、
  roofline 式成本模型（用于排序，不做性能预测）、Chrome Trace 导出。
- **量化与布局**：weight-only int8（权重驻留 1 字节，计算仍在 f32）；
  NHWC 布局传播 pass 与后端布局声明（仅参考后端承接）。

### 安装

```bash
pip install -e .              # 运行时只依赖 numpy
pip install -e ".[frontend]"  # 额外安装 torch，用于 torch.export 前端
```

CUDA 后端需要 `cupy-cuda12x`；HIP 后端需要 ROCm 与自带的 `hip.runtime`。
参考后端只需 numpy，任何平台可用。

### 环境要求

| 项 | 要求 |
|---|---|
| Python | ≥ 3.10（本版测试在 3.14 上跑） |
| numpy | ≥ 1.24 |
| torch（可选） | 前端适配器声明区间 2.9–2.13，本版仅在 2.13.0 上验证；区间内未验证版本不承诺回归覆盖，越界显式抛 `UnsupportedTorchVersion` |
| CUDA 后端 | CUDA 12.x + 驱动 550+（2×V100 `sm_70` 上验证）；cuFFT 提供 `fft` / `rfft` |
| HIP 后端 | ROCm 7.2（RX 7650 GRE `gfx1100`，WSL 上验证） |

### 快速上手

```python
import numpy as np
from aetherflow import Graph, ReferenceBackend, f32, get_backend, verify

g = Graph("add")
x = g.input("x", f32[2, 3])
y = g.input("y", f32[2, 3])
g.output(g.add(x, y), "sum")

assert not verify(g)
out = ReferenceBackend().execute(g, {
    "x": np.ones((2, 3), dtype=np.float32),
    "y": np.full((2, 3), 2.0, dtype=np.float32),
})["sum"]
```

换后端只改装配：`get_backend()` 自动挑可用后端，或用 `AETHERFLOW_BACKEND=hip`
显式指定；图与权重不用改。

```bash
python demo_transformer_block.py
python verify_cuda.py                        # 有 NVIDIA 卡时验证 CUDA 后端
python -m benchmarks.run --backend ref        # 性能基准（另有 hip / cuda）
```

### 已验证的配置

| 后端 | 环境 | 用例 |
|---|---|---|
| 参考 | Windows / Python 3.14 | 2675 个用例，416 跳过，全绿 |
| HIP | WSL + ROCm 7.2 / RX 7650 GRE（`gfx1100`） | 2675 个用例，373 跳过，全绿 |
| CUDA | Ubuntu 22.04 / 2× Tesla V100（`sm_70`）/ CUDA 12.4 / 驱动 550 | 全绿（cuDNN 因驱动版本不匹配按设计跳过） |

```bash
# 本地全量
python -m unittest discover -s tests -t .

# WSL + HIP 全量（WSL 内已装 ROCm，且用 ROCm 环境的 Python）
HSA_ENABLE_DXG_DETECTION=1 python -m unittest discover -s tests -t .
```

本地建议用带 torch 2.13 的解释器；3.10 环境的 torch 2.4.1 在兼容矩阵之外，
前端用例会被按设计拦下（不是代码缺陷）。

### 已知限制

- **分布式**：NCCL / RCCL 已到设备侧绑定（AllReduce / AllGather / ReduceScatter /
  Broadcast），但**未接进数据并行路径**，跨卡归约仍走主机中转；`overlap` 恒为
  `False`。RCCL 只在单卡 `world=1` 上验证。ZeRO-1/2/3 与流水部分参见 DESIGN §7。
- **张量并行**：只有前向（列 / 行并行 + attention 头切分）。行并行返回**部分和**，
  需调用方归约；反向通信与优化器分片未接。
- **量化**：weight-only int8。权重驻留 1 字节，GEMM 仍走 f32 反量化；
  无 int8 累加、无 fp8、无 QAT、无非对称零点——**不要期待"int8 算得更快"**。
- **布局**：NHWC 传播 pass 已交付，但只有参考后端承接；设备后端对 `layout="nhwc"`
  显式报错，原生设备端 NHWC 内核尚未实现。训练图保持 NCHW（NHWC 无反向规则）。
- **设备端循环**：`DeviceWhileLoop` 把条件与体各捕获一次、状态常驻显存，但每轮仍
  要读一次谓词；实测**不比宿主路径快**，交付的是执行形态而非加速。
- **控制流**：没有设备端真分支；`Conditional` / `WhileLoop` 每次判定回主机。
- **推理**：分页 KV cache 与投机解码是组件级交付，**未接进 `LMRunner`**——主干是
  Mamba，没有随上下文增长的 KV。投机只支持贪心验收与 `batch=1`。
- **前端**：torch.export 子集。整数只有 i32（int64 收窄、越界拒绝）；
  无 `bool` 常量、无 `arange`、`padding_idx` / `sparse` / `scale_grad_by_freq`
  显式拒绝。
- **数值与内核**：`fft` / `rfft` 仅 CUDA 的 cuFFT 路径，且无 VJP；`topk_mask` 只有
  参考实现；`spike_linear` 的 `dspikes` 必须稠密；卷积自研内核比 cuDNN / MIOpen
  慢 4～37×（详见 README §5）。
- **包发布**：CI 只构建 wheel 并上传 artifact，尚未接 PyPI / GitHub Release。
- **性能**：HIP 上整图调度被 launch 开销主导，`graph_mode="auto"` 能把
  Transformer block 从 5.9 ms 压到 0.83 ms；`maxpool2d` 在大形状上只有约
  2.6 GB/s（20 MB 访存耗 7.7 ms），`matmul` 1024³ 约 264 GFLOPS，两者都
  尚未优化（见 DESIGN 5.249，数据来自 RX 7650 GRE + ROCm 7.2）。

### 兼容性策略

- API 主版本为 `1`，`v0.1.x` 内不破坏现有公开接口。
- 能力缺失一律显式报错（`BackendCapabilityError` / `UnsupportedOperatorError` /
  `UnsupportedTorchVersion` / `LayoutError` 等），不做静默降级或回退。
- 后端能力以声明为准：`capabilities` 里没写的形状、精度、布局，执行期直接拒绝。

### 维护者备注

- 临时脚本（`_tmp_*.py` 等）用完即删；结论写进 DESIGN 对应小节，凭证不落盘。
- 提交后用 `git status` / `git diff --stat` 核对。曾出现工作区文件被旧版覆盖
  （表现为 `git diff` 里只有"删除已提交内容"）：用 `git checkout HEAD -- <file>`
  逐个恢复，再在其上继续编辑，不要重写整份文件。
- 代码与文档（DESIGN / 本文件 / README）分开提交。
- PowerShell 5.1 不支持 `&&` 与 heredoc：用 `;` 串联，`git commit` 用多个 `-m`。
- **整数内核渲染**：`render(kernel, dtype)` 的 `_dtype_key` 按 `dtype.name` 取
  `f32` / `f16`，所以必须传 `np.dtype` 实例。传 `np.float32`（类）或 `DType`
  枚举都会被拒（分别报"收到 `<class 'numpy.float32'>`"与"收到 `F32`"）；整数
  内核（如 `mod`）的类型由 `Tensor(type_name="int")` 钉死，dtype 只为拿同一份
  prelude。这个错误在参考后端的路径上不会触发，只有 HIP 真机跑 conformance
  才暴露，与 5.247 的 `grid_stride` 属同一类"渲染期才炸"的问题。
- **`mod` 必须做 floor 修正**：C 的 `%` 与 Python `%` / `torch.remainder` 在被除数
  为负时结果不同（C 是截断语义）。设备端在 `r = x % y` 之后必须补
  `r != 0 && (r < 0) != (y < 0) ⇒ r += y`，否则负槽位哈希会与参考实现分叉。

---

## 版本记录

| 版本 | 日期 | 摘要 |
|---|---|---|
| v0.1.0 | 2026-11-20 | 首个版本：IR / 参考执行器 / 自动微分 / 训练与推理闭环 / CUDA 与 HIP 后端 / 一致性套件 |
