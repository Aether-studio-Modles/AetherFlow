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

- **一份语义 IR，多个硬件后端**：111 个 IR 算子，CUDA 承接 111 个（含 cuFFT 的
  `fft` / `rfft`）、HIP 承接 109 个；
  新增硬件只需写插件，不改 IR、verifier、前端与优化。
- **一份参考语义**：每个算子都有不依赖加速库的 numpy 定义，是跨后端对齐的唯一真值；
  `conformance` 套件按算子逐项比对，支持不了的地方显式失败，不静默降级。
- **循环级 IR**（5.332）：`LoopIR` 把 `WhileLoop` 收敛成"单图条件 + 单图体 +
  carried"的规范形态，`split_invariants` 把每轮重算但不变的纯计算搬进
  preheader（`LoopSplit`），数值与原循环逐位相同；带 RNG / STATE / IO 的节点
  与依赖 carried 的节点留在体内。支持 JSON 往返，嵌套控制区域显式拒绝。
- **调优接进执行路径**（5.333）：`SchedulePolicy` 按 `backend + device + op +
  形状签名` 查配置库，命中后覆盖逐元素 / 归约的 block 与 conv2d 的
  tile/roc/cb。优先级是"显式 `set_tuning` → 构造参数 → 库 → 自动"；未命中
  统计可查，库里的坏值（如非 2 的幂的 block）当场报错而不是静默回退。
- **自动微分**：49 条 VJP 规则，反向图是普通可序列化图；支持二阶导、激活重算、反向图裁剪。
- **训练**：MLP / CNN / RNN / Transformer / Mamba / SNN 构件；数据并行
  （5.331 起 NCCL / RCCL 可加载时**默认设备侧归约**：单进程多卡自动
  `ncclCommInitAll`、每进程一卡自动 `ncclCommInitRank`；库缺失或显式
  `collective="host"` 才经主机中转）、
  ZeRO-1/2/3、GPipe 与 1F1B 流水、张量并行（仅前向）。
- **推理**：跨请求图缓存与权重驻留、前缀缓存、批量解码、束搜索、连续批处理；
  **分块预填充**（`ServingEngine(chunked_prefill=True)`：长 prompt 拆到多个解码步
  推进，已就绪的请求照常出 token；构图块与每步推进量都是 `chunk_size`，5.344；
  与 `device_state=True` 不组合，显式拒绝）；
  **设备态解码**（`LMRunner(device_state=True)` / `ServingEngine(device_state=True)`）：
  decode 图捕获一次、状态常驻显存、每步只把 logits 取回主机（DESIGN 5.279～5.281；
  空闲 A800 24 层 81.2 → 8.2 ms/token，连续批处理 `capacity=8` 时 66.1 → 491.6 tok/s）；
  分页 KV cache（含换出 / 换入、LRU 调度与共享前缀 `fork` / 写时复制，以及
  token→token 的贪心生成闭环 `PagedGenerator` 与自动前缀缓存，5.310 / 5.335；
  跨进程 / 落盘前缀缓存 `DiskPrefixCache`，5.337/5.339）
  与贪心投机
  解码（组件级，见"已知限制"）；OpenAI 兼容 HTTP 子集（`aetherflow.serve`：
  `/v1/completions`、`/v1/chat/completions`、SSE 流式）。
- **调优与可观测**：与后端无关的参数空间与搜索、conv / 逐元素 / 归约的调优旋钮、
  roofline 式成本模型（用于排序，不做性能预测）、Chrome Trace 导出。
- **量化与布局**：weight-only int8（权重驻留 1 字节，计算仍在 f32）与 W8A8
  （`linear_i8a8`：int32 累加后乘两个 scale，CUDA 走 `__dp4a`、HIP 标量乘加，
  V100 真机与参考逐位相等）；QAT 的假量化算子 `fake_quantize_i8`（f32 进出，
  前向等同量化-反量化，反向是直通估计器，5.311）与可学步长的 LSQ 版本
  `fake_quantize_i8_lsq`（前向逐位相同，反向按非饱和区直通并给 scale 梯度，
  5.312）；非对称零点的主机侧语义（`quantize_i8_zp` / `dequantize_i8_zp` /
  `linear_i8_zp` / `linear_i8a8_zp`，值域扩到含 0、W8A8 的 int32 上界收到
  `K*255*255`，5.315；量化 / 反量化两个设备内核见 5.322，`linear_i8_zp` /
  `linear_i8a8_zp` 两个融合线性层的设备内核见 5.323）与逐组 weight-only 反量化
  （`dequantize_i8_group` / `dequantize_i8_group_zp`：组大小由 `K / K_g` 推出，
  不另设 attr，5.317；设备内核 5.321，同一份源码供 NVRTC / hipRTC，只做反量化、
  不融合 GEMM，V100 上 conformance 逐位相同）；fp8（e4m3 / e5m2）的参考语义与单字节位模式
  编解码（`cast(dtype="f8e4m3")` / `pack_fp8` / `unpack_fp8`，e4m3 无 inf 并
  饱和到 ±448，e5m2 溢出到 inf，5.318），设备端 `cast` 已有量化内核（5.325，
  载体仍是 f32、不省带宽），写侧有带 scale 的量化（5.326：
  `quantize_weight_fp8` 逐输出通道 / `per_tensor=True` 对齐 HF 的
  `weight_scale`，`quantize_activation_fp8` 逐 token），线性层有
  `linear_fp8`（weight-only）与 `linear_fp8a8`（W8A8）（5.327：反量化后走
  既有 f32 GEMM、累加在浮点里，不承诺 fp8 tensor core），以及把 fp8 真正
  存成 1 字节的 `pack_fp8` / `unpack_fp8`（IR 用 i8 位模式承载）与融合内核
  `linear_fp8_packed`（5.328：权重 1 字节驻留、设备端解码后乘 scale 累加，
  读带宽降到 f32 载体的 1/4，但不省算力；主机侧入口
  `FP8Weight.pack()` / `load_fp8_linear(packed=True)`），逐组 scale 的
  `dequantize_fp8_group` / `linear_fp8_group`（`(K/G, N)` 组 scale，组数由
  形状推出；反量化输出 f32，不走 f8 载体，避免节点边界把乘过 scale 的值
  二次舍入回格点）与两端都打包的 W8A8 `linear_fp8a8_packed`（激活与权重
  各 1 字节、专用内核寄存器解码、f32 累加，5.342），主机侧入口
  `quantize_weight_fp8_group`；其它算子遇到 fp8 仍显式拒绝；没有 fp8
  tensor core，收益是读带宽不是算力；
  NHWC 布局传播 pass 与后端布局声明（设备端自 5.334 起原生承接 conv2d 分块直算
  与 maxpool2d / avgpool2d 前向）。
- **权重入口**：自带 `safetensors` 解析（不引第三方依赖，运行期仍只有 numpy），
  支持按需读取单张量与分片目录合并、bf16 无损展宽为 f32、fp8（e4m3 / e5m2）
  按位模式解码成 f32 载体（5.318 / 5.319）；
  `load_into_store` 把外部张量装进参数容器——名字映射、PyTorch `nn.Linear` 布局
  转置、有损 dtype 转换都要显式声明，先全量校验再原地写，失败不留半装状态
  （5.313）；`load_quantized_linear` / `QuantizedWeight` 读量化权重——int8 强制、
  scale 与零点的形状 / 值域 / 正值校验、逐通道与逐组、metadata 自描述
  （`quantization` / `quantization.group_size`）与张量一致（5.317）；
  `load_fp8_linear` / `FP8Weight` 读 fp8 量化权重——载荷每个值必须真的落在
  fp8 格点上（未量化的 f32 与 f8 载体同型，只能按值判定）、格式由文件 dtype
  推出、显式 kind 不许与文件矛盾、支持 `input_scale` 标量与 `(K/G, N)` 逐组
  scale（5.326）；`packed=True` 读 1 字节位模式（F8 / I8 / U8，I8 / U8 需显式
  kind，字节走 `SafeTensorsFile.raw` 不经解码再编码以免 NaN 多码归一化），
  `FP8Weight.pack()` 返回幂等的 1 字节副本且 `dequantize()` 数值不变（5.328）。
- **文本入口**：`BpeTokenizer` 读 HF `tokenizer.json` 的 ByteLevel BPE 谱系
  （GPT-2 / Qwen2 / Qwen3），不引 `tokenizers`；支持 merges 两种写法、added /
  special token 整体匹配、NFC 系 normalizer、`use_regex` / `add_prefix_space`；
  解码先把连续字节拼回再解 UTF-8。WordPiece / Unigram / SentencePiece /
  Llama-3 的 `Split` 预分词显式拒绝——近似实现产出的 id 每个都合法，但不是训练
  时那串（5.314）。

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
| CUDA 实测机 | A800-SXM4-40GB、驱动 550.127、CUDA 12.4、`cupy-cuda12x` 14.2（Ubuntu 22.04 / Python 3.10）。系统没装 CUDA toolkit，cuBLAS / cuDNN 由 miniconda 的 `nvidia/{cublas,cudnn}/lib` 提供，跑卷积必须把这两条加进 `LD_LIBRARY_PATH`；对照 cuDNN 再加 `NVIDIA_TF32_OVERRIDE=0` |
| HIP 后端 | ROCm 7.2（RX 7650 GRE `gfx1100` / `gfx1102`，WSL + DXG 直通上验证）。WSL 内必须 `export HSA_ENABLE_DXG_DETECTION=1`，否则 `hsa_init` 失败、`hipGetDeviceCount` 返回 0 |

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
python -m benchmarks.framework_compare --device cuda  # 与 torch eager / compile 同机对照（整图，5.316）
```

### 已验证的配置

| 后端 | 环境 | 用例 |
|---|---|---|
| 参考 | Windows / Python 3.14 | 3428 个用例，464 跳过，全绿 |
| HIP | WSL + ROCm 7.2 / RX 7650 GRE（`gfx1102`） | 本轮复跑 `test_hip_backend` 72 条、`test_device_control` + `test_lm_device_state` 44 条，全绿 |
| CUDA | Ubuntu 22.04 / A800-SXM4-40GB（`sm_80`）/ CUDA 12.4 / 驱动 550 | 2857 个用例、80 跳过；123 条前端 error 是该机 torch 2.5.1 不在兼容矩阵内（按设计显式拒绝），CUDA 组含新增用例全过 |
| CUDA | Tesla V100-SXM2-32GB（`sm_70`）/ 驱动 550.127.05 / Ubuntu 22.04 / python3.10.12 + cupy 14.2.0 | `python3 verify_cuda.py` → `139 executed / 1 refused / 0 failed`（5.325 后；`cast_f8e4m3` / `cast_f8e5m2` 首次执行、err=0.0，唯一的 refused 是 `f64_input_relu`；`matmul_partial_batch` err=4.8e-07 来自 f32 GEMM 累加顺序，`linear_i8a8_zp` 逐位相同、`linear_i8_zp` err=1.5e-05）；`tests.test_cuda_backend` 382 条 5.1 秒全过（2 跳过）；未跑全量 unittest |

```bash
# 本地全量
python -m unittest discover -s tests -t .

# WSL + HIP 全量（WSL 内已装 ROCm，且用 ROCm 环境的 Python）
HSA_ENABLE_DXG_DETECTION=1 python -m unittest discover -s tests -t .
```

本地建议用带 torch 2.13 的解释器；3.10 环境的 torch 2.4.1 在兼容矩阵之外，
前端用例会被按设计拦下（不是代码缺陷）。

### 已知限制

- **分布式**：NCCL / RCCL 绑定已接进数据并行默认路径（5.331）。单进程多卡
  `DataParallelTrainer(collective="auto")` 在库可加载时自动
  `nccl_device_collective(backend, devices)`（`ncclCommInitAll` + 每卡原地
  AllReduce）；每进程一卡的 `DistributedDataParallel.from_env` 在设备态默认
  `NcclDeviceCollective.from_env()`（rank 0 经 `AF_MASTER_ADDR` /
  `AF_MASTER_PORT` 分发 uid，其余 rank `ncclCommInitRank`）。库缺失时显式抛
  `DeviceCollectiveUnavailable`（不静默降级），显式 `device=False` 或
  `collective="host"` 才走主机中转。`overlap` 声明的是设备态重叠，仍恒为 `False`
  （设备态强制 `bucket_bytes=0` / `threads=1`）；主机态的桶间并发是
  `DistributedDataParallel(overlap=True)` 的独立开关（5.302），5.338 已把这两件事
  在文档与 reason 文本里分开。真机只到 `world=1`（RCCL 与 CUDA 各一条），
  多卡数值与 uid 分发用例已写但当前没有多卡机器、未执行。ZeRO-1/2/3 与流水
  部分参见 DESIGN §7。
- **张量并行**：只有前向（列 / 行并行 + attention 头切分）。行并行返回**部分和**，
  需调用方归约；反向通信与优化器分片未接。
- **量化**：weight-only int8（权重驻留 1 字节，GEMM 仍走 f32 反量化）与 W8A8
  （`linear_i8a8` + `quantize_activation`：int8 激活 × int8 权重先在 int32 里
  累加，再乘激活 / 权重两个 scale）。CUDA / HIP **已有设备内核**（CUDA 走
  `__dp4a`，HIP 走标量乘加，V100 真机与参考逐位相等，5.309）；但它不是 int8
  tensor core（没有 IMMA / wmma 分块），**不要期待"int8 算得更快"的结论**；
  QAT 到假量化算子（`fake_quantize_i8`，5.311）与 LSQ 的可学步长版本
  （`fake_quantize_i8_lsq`，5.312）——后者要求 scale 作为图输入才能收到梯度，
  没有 W8A8 的 QAT 构建器，也没把假量化自动接进训练器；fp8 有参考语义、
  写入 / 读取入口与设备端 `cast`（e4m3 / e5m2，5.318 / 5.319 / 5.325 /
  5.326：带 scale 的逐通道权重与逐 token 激活量化、`FP8Weight` 容器要求载荷
  值真的在格点上；设备上 `cast` / `linear_fp8` / `linear_fp8a8` 的载体仍是
  f32、只把值落到格式格点、不省带宽），
  `linear_fp8` / `linear_fp8a8` 已有 IR 与设备实现（5.327：反量化走
  `mul`、GEMM 复用 sgemm，不承诺 tensor core）；要真正省驻留 / 读带宽就走
  5.328 的 `pack_fp8` / `unpack_fp8` + `linear_fp8_packed`（1 字节权重、
  设备端解码，但算力与 f32 GEMM 持平，且只接逐张量 / 逐通道 scale、没有
  反向），其它算子遇到 fp8 仍显式拒绝，也没有 fp8 tensor core 与逐块 scale；
  W8A8 也尚无反向（bias 除外）。非对称零点的量化 / 反量化与融合线性层都已有
  设备内核（5.322 / 5.323，V100 上 conformance 通过），但 `linear_i8a8_zp`
  走标量乘加、没用 dp4a / IMMA，也**不融合 epilogue**（int32 累加后仍物化一份
  f32 输出）；逐组反量化已有设备内核（5.321，只做反量化、不融合 GEMM，
  融合成 `linear_i8_group` 仍未做），但没有主机侧的逐组量化 helper
  （写侧由导出器负责）。
- **布局**：NHWC 传播 pass 已交付；设备端自 5.334 起有原生 NHWC 内核——conv2d
  只走分块直算（f32、groups=1），maxpool2d / avgpool2d 只覆盖前向。im2col /
  隐式 / Winograd / 分组 / cuDNN 遇到 `layout="nhwc"` 在入口显式报错；
  池化反向与取值位置（`maxpool2d_indices`）没有 layout 属性，训练图仍是 NCHW
  （NHWC 无反向规则）。配置库的 conv2d 签名在 NHWC 下多一个 `layout` 字段，
  NCHW 保持旧签名。
- **国产卡**：以 zip 硬件包接入（与 novac 同层），不改框架核心。天数（Iluvatar
  CoreX）与沐曦（MetaX C500）两个包已在各自仓库的真机跑通并实测，但它们不在
  本仓发布、也不在本仓 CI 里；主仓只验收契约（能装、能识别、无厂商库时
  `BackendUnavailableError`、清单摘要失配即拒绝），主仓自身没有国产卡
  conformance 结果（DESIGN 5.336）。
- **循环级 IR**（5.332）：`LoopIR` / `LoopSplit` 只接受单图体，嵌套控制区域
  显式抛 `LoopIRError`；只做循环不变量外提，不做循环融合 / 分块 / 软件流水，
  也未接进 schedule 搜索或编译流水线。条件图不参与外提（每轮必须求值）；
  外提判据只有"是否依赖状态 + 是否纯"，没有成本模型，极小的不变计算可能
  不如留在体内。
- **配置库的执行期命中面很窄**（5.333）：只接了逐元素 / 归约 block 与
  conv2d 分块核三个旋钮；im2col / implicit GEMM / Winograd / matmul 的选型
  仍走启发式。没测过的形状照旧走自动，不做在线探索；库被改写要重新
  `set_schedule`，不做热更新。block 签名不含 dtype（线程铺法与元素类型无关），
  离线写库必须用同一个 `block_signature`。
- **设备端控制流**：`DeviceWhileLoop` / `DeviceConditional` 把条件、分支各捕获一次、
  状态与输出常驻显存，但每次判定仍要读一次谓词；实测**不比宿主路径快**，交付的是
  执行形态而非加速。没有 device-side 真分支，`Conditional` / `WhileLoop` 的判定
  本身仍回主机（DESIGN 5.245 / 5.277）。**解码的"分支"接不进去**：判定输入是主机
  采样出的 token、每步还要写回 `ids`，设备端循环的"条件→体"之间没有主机回调位
  （DESIGN 5.278，负结果）；**但 decode 的状态已经常驻**：
  `LMRunner(device_state=True)` / `ServingEngine(device_state=True)` 捕获一次
  `steps=1` 的 decode 图，状态每步 D2D、只回传 logits 块（5.279～5.281；
  空闲 A800 24 层 81.2 → 8.2 ms/token，RX 7650 GRE 8～24 层 7.2～7.9×）。它要求
  `resident=True` 与静态形状（换 batch 重建计划），prefill 与首步仍各回传/上传
  一次状态；束搜索的状态重排已并进 decode 图（5.282 / 5.283：状态输入是
  `embedding(上一状态, parents)`，每步一次 replay；A800 同机同口径
  `num_beams=1/2/4` 各快 5.5% / 4.2% / 4.5%），投机解码仍走宿主路径。
- **动态形状**：主机态的执行与训练支持符号批维——一次构图跑不同 batch，归约反向的
  计数在运行期按实际形状算；`DynamicRunner(dynamic_axes={"x": {0: "B"}})` 把这条能力
  包成"按形状模板缓存"，声明过的轴取任意值都复用同一张图（DESIGN 5.276）。设备态
  捕获计划、分段重算、需要把若干维合并成二维的广播反向（`sum_axes`）以及 torch
  导入的 SymInt 仍要求静态形状，入口显式拒绝（契约表见 DESIGN 5.217 / 5.275）。
- **推理**：`LMRunner` 主干是 Mamba，没有随上下文增长的 KV，所以分页 KV 不在它的
  主路径上。分页缓存已带换出 / 换入与 LRU 调度：`PagedCacheManager`
  在容量不足时换出最久未使用且未 pin 的序列、被访问时自动换回（`pin` 保护正在
  解码的请求）；`PagedDecodeScheduler` 把多序列交错解码接在同一张单步图上，
  `suspend` / `resume` 对应解除保护与换入，`fork` 让新序列共享父序列的前缀。
  `fork` 在缓存层是多条序列共享同一批物理页、写入时按页拆分（块级引用计数），
  适合同前缀多路采样 / 束搜索 / 同系统提示的多请求。`PagedGenerator`（5.310）
  在调度器之上补了 token→token 的贪心生成闭环（主机侧嵌入 + 位置编码、一层
  词表头、EOS 停止）；5.335 起 `generate` 默认做**自动前缀缓存**——不传
  `fork_from` 时扫描驻留序列挑最长严格前缀（跳过换出态与等长序列），fork 后
  只 prefill 剩余 token，命中统计在 `stats()` 的 `prefix_hits` /
  `prefix_reused_tokens` / `prefix_misses`。5.337 把同一判据接到盘上：
  `DiskPrefixCache` 一条序列一个 `.npz`（token 序列 + 每层 K/V，文件名与 meta
  都带 `model_id + tokens` 摘要），内存未命中时查盘、命中就整段导入 KV，
  `generate` 结束与 `finish` 释放前把全长落盘（5.339：换出态跳过、空序列
  不留条目、`disk_stores` 只在真写入时自增）；指纹 / 形状 / dtype / 页宽失配与盘上损坏都只算
  "没有条目"并退回 prefill。这些组件都是主机态，没有真机
  D2H 与拆页开销的实测，也要求 `batch=1` 与静态 `max_steps`。投机解码走
  `LMRunner.generate_speculative`（5.303），只支持贪心验收与 `batch=1`。
- **推理服务**：OpenAI 兼容只做子集——`logprobs` / `n>1` / 工具调用 /
  `response_format` / 非 0 penalty 显式 400。认证是**静态单密钥** Bearer
  （`api_key=...`，`/healthz` 免认证，`hmac.compare_digest` 比较），限流是
  **进程内全局**滑动窗口（`rate_limit` / `rate_window`，超限 429 + `Retry-After`，
  认证失败不占配额）；多副本与按密钥计量的集群配额没有。`stop` 已参与调度：
  每个解码步之前检查在跑请求的最长可解码前缀，命中就 `cancel` 并腾槽，
  `finish_reason` 为 `"stop"`；流式用 hold-back 压住可能是 stop 开头的尾部
  （否则前缀已发出、撤回不了）。检测粒度是 token，跨 token 的 stop 会多生成到
  完整命中为止。流式仍由每个请求轮询 `ServingEngine.progress()` 快照，并发很大
  时应改成调度器推送增量。
- **权重入口**：只读 `safetensors` 容器（不读 GGUF / PyTorch `.bin` / `.pt` / TF
  检查点）；fp4 与稀疏张量不支持，fp8 权重读入时解码成 f32 载体（5.319）、
  不保留单字节；量化权重
  支持本仓库自己的 `quantization` / `quantization.group_size` 约定（5.317），
  **不声称兼容 GPTQ / AWQ 原始检查点**（它们的 int32 交织打包与 `qzeros` 的
  +1 偏移是另一套约定）；没有内置任何具体模型的转换表——名字映射与
  `nn.Linear` 转置由调用方声明；大模型按需读取的接口已就位，但没有 GB 级权重
  的实机加载耗时与内存峰值数字（5.313）。
- **文本入口**：只支持 ByteLevel BPE，且不解析 `tokenizer_config.json` /
  `special_tokens_map.json`——`eos` / `pad` 靠 added token 的候选名推断或调用方
  显式传入，chat 模板仍用 `ChatTemplate` 手写。不支持 WordPiece（BERT 系）、
  Unigram（T5 系）、SentencePiece（Llama-2 系）、Llama-3 的 `Split` 预分词、
  BPE dropout 与 `lstrip` / `rstrip`；没有 offset mapping；用自造小词表逐 id
  对拍，未与 HF 官方实现在真实大词表上做全量对拍（5.314）。
- **前端**：torch.export 子集。整数只有 i32（int64 收窄、越界拒绝）；
  `bool` 作为掩码载体：比较 / `logical_and` / `logical_or` / `logical_not`
  （含 `& | ~` 写法）/ `where` / `masked_fill` / `cast` 双向已接（5.340），
  浮点 `logical_*` 与整数位运算显式拒绝；
  `sign` / `round` / `floor` / `ceil`、`baddbmm`、原地 `copy_` 无对应语义；
  非整数 `pow` 指数、图内 `clamp` 边界、非零或 reflect 的 `pad`、非默认
  `softplus.threshold` 一律拒绝；常量构造（zeros / ones / full / arange 等）
  落成图属性，单个常量上限 65536 元素；`padding_idx` / `sparse` /
  `scale_grad_by_freq` 显式拒绝。
- **数值与内核**：`fft` / `rfft` 仅 CUDA 的 cuFFT 路径，且无 VJP；`topk_values` /
  `topk_indices` / `topk_mask` 的设备内核是**朴素 k 轮块内选择**（f32 / f16，
  bf16 与 `argmax` 一样不支持，k 受 48 KiB 动态共享内存约束）；`spike_linear` 的
  `dspikes` 必须稠密；卷积的差距按稳态口径读
  （DESIGN 5.253）：低 GFLOP 形状与 cuDNN 同量级（1x1_c32 是 0.72×），真实差距
  只在宽通道 3x3/5x5 的内核结构——im2col 稳态为 cuDNN 的 1.4～3.7×，tiled 与
  implicit 落后 11～18×。隐式核的空间寄存器分块试过并回退（5.254）：A800 上
  10 个形状无一获益，最差慢 29%。`conv_impl="auto"` 已按形状选择（5.252）：
  K 在 (32, 400] 且 im2col 物化缓冲 `n*ho*wo*K` ≤ 3.2M 元素时走隐式 GEMM，
  1 GFLOP 以上交给 cuDNN；A800 上 auto 与同形状最优实现差 ≤5%。
  `conv_impl="winograd"`（5.255，F(2x2,3x3)，只做 f32/stride1/dilation1/groups1）
  的融合版与权重驻留（5.256/5.257）把 A800-40GB 上 3x3_c8/c16/c24 的稳态
  压到 0.029～0.048 ms，为 cuDNN 9 的 0.45～0.75×，对 cuDNN 相对误差约 2e-7；
  CUDA 的 auto 对 3x3 且 K > 32 选它，HIP 未开。5.260 再把静态 conv 的解析、
  线程铺法与 launch 参数挪到计划期，单层 c16 的端到端从 0.054 降到 0.027 ms
  （`benchmarks/overhead_compare.py`）。c32/c64 仍走四段版，5.264 把它的
  launch 也展开到计划期、`U` 随计划驻留，c64 单层端到端 0.113 → 0.094 ms；
  设备侧的三段内核 + batched GEMM 仍是主体，与 cuDNN 的比值待机器空闲后按
  同进程交替复测（此前记录的 1.80×/4.27× 未复测）。**绝对值随机器变，
  跨机只比比值**。
- **包发布**：CI 在 `v*` tag 上通过 PyPI trusted publishing 发布（先校验 tag 与
  `__version__` 一致）；GitHub Release 仍需手工创建。版本号只有
  `aetherflow/__init__.py` 一处事实来源，`pyproject.toml` 通过 `dynamic` 读取。
- **性能口径**：`execute()` 的契约是返回主机数组，每轮都付一次 D2H（A800 上
  4 MB 约 1.4 ms，比多数内核本身还贵）。测吞吐用 `execute_device()`，基准用
  `--device-outputs`；两种口径的数字不能互相比较（见 DESIGN 5.251）。
  `graph_mode="auto"` 收益取决于逐节点开销：HIP 上 Transformer block
  5.9 → 0.83 ms，A800 设备侧多节点图 3.4～6.0×（1.66 → 0.40 ms），
  单节点图无收益。注意 5.261 的发现：conv-bn-relu 这类图在捕获期仍会做
  H2D，`_graph_failure` 报 `RuntimeError: ... H2D transfers are disallowed`，
  实际回落到逐节点路径；这类图现在靠计划期展开提速（8 层 0.919 → 0.298 ms，
  逐通道 BN 的 `channel4d` 由 5.262 降到 0.303 ms、带常量 reshape 的 `channel`
  由 5.263 降到 0.302 ms，三种形态齐平）。
  显式 `conv_impl="cudnn"` 时整图无法捕获（algo 查询是 host API），`auto`
  会在捕获期自动改走 tiled。捕获为静态输入预分配的缓冲，其首次填充与预热
  流之间用事件建立依赖（5.265）——此前是默认流异步 H2D、预热在非阻塞流上
  抢跑，输入会读到未初始化内存。

### 兼容性策略

- API 主版本为 `1`，`v0.1.x` 内不破坏现有公开接口。
- 能力缺失一律显式报错（`BackendCapabilityError` / `UnsupportedOperatorError` /
  `UnsupportedTorchVersion` / `LayoutError` 等），不做静默降级或回退。
- 后端能力以声明为准：`capabilities` 里没写的形状、精度、布局，执行期直接拒绝。

### 维护者备注

- 临时脚本（`_tmp_*.py` 等）用完即删；结论写进 DESIGN 对应小节，凭证不落盘。
- **文档里的算子总数长期滞后**：5.323 后实测 `len(OPS) == 93`、`len(GPU_OPS) == 91`
  （HIP 的 `HIP_OPS` 直接继承，`CUDA_OPS` 同为 91），而正文明面上的 80 / 77 是
  5.312 起的增量口径（5.315、5.317、5.318、5.321、5.322、5.323 各加若干），与实际值
  仍差 13 / 14。未承接的只剩 `fft` / `rfft`（CUDA 由 cuFFT 承接）。下次统一校准一次，以
  `py -3.14 -c "from aetherflow.ir.ops import OPS; print(len(OPS))"` 与
  `device_shared.GPU_OPS` 为准，并把"70 个自研内核"这类分项一起重数，不要只改总数。
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
- **L2 的 `Bin` 打印不加括号**：`(a + b) * c` 必须显式写 `Paren(...)`，否则
  渲染成 `a + b * c`。`winograd_fused` 第一版就是共享内存索引
  `(p*C + c)*TT` 少了一层括号，读写落在不同格上，数值相对误差 1.0 量级但
  内核不报错（DESIGN 5.256）。凡是"先加后乘"的索引都过一遍这一条。
- **多维写回要逐边带行偏移**：同一内核里写 2x2 输出时，四条边各自要有
  `oy*WO`；只给其中两条加，会变成"每行都往第 0 行写"，表现为一半像素对、
  一半全错。数值测试要覆盖 `HO`/`WO` 为奇数与非零 `th` 的 tile（首个 tile
  之外的分块），只测左上角看不出这类错。
- **缓存"值没变"只能看 IR 结构，不能看指针**：5.257 的权重驻留只对
  "权重由 `constant` 节点产生"的 conv 生效；`_prepare_plan` 里的盒子必须由
  `group.kernels` 持有（随计划作废），不能放到 backend 的常驻 dict 里。
  图输入权重每次可能在同一个设备缓冲里换值，按地址或对象身份缓存会静默用
  旧权重；这类错误没有任何报错或数值异常，只有"换了权重输出不变"。
  新加缓存时先问一遍：**失效条件的证据来自哪里**，再写代码。
- **基准对拍要先对齐数据生成顺序**：`winograd_compare.build` 与
  `cudnn_compare.build` 都用 `default_rng(0)`，但前者若先抽权重再抽输入，
  两条路径算的就不是同一份数据，表现为"100% 元素不匹配、最大绝对差 3.35"，
  很容易被误判成算法错误。
- **计划期展开的快路径必须留同一条回退路径**：5.260 的
  `_planned_winograd_fused` 只处理"常量权重 + 静态形状 + 连续输入"，形状、
  dtype 或连续性任一条不符就回落 `_op_conv2d`。只写快路径不写回退，图输入
  换个形状就会按旧步长读内存；这些前提在计划构建时看不见（取决于运行期
  数组），所以检查只能放在闭包入口，且必须是原路径而不是另一份近似逻辑。
- **HIP 真机才暴露的两类问题**（5.279）：容器 dtype 表缺常量（`cp.uint16`）
  会让**任何图**首次 `invoke` 就 `AttributeError`，而 HIP 用例常年跳过所以
  一直没被发现——设备用例跳过时，"跳过"本身要当成未验证而不是通过；异步
  `replay` 之后必须同步捕获流，HIP 的默认流不会等它（CUDA legacy 默认流隐式
  同步），否则收尾 D2H 读到上一次 replay 的输出。WSL 里跑 HIP 必须带
  `HSA_ENABLE_DXG_DETECTION=1`，否则设备数报 0。
- **捕获期的零拷贝视图要按"独占内存"记账**（5.280）：arena 按"最后一个读者"
  回收，视图的源可能比视图先被释放。`_shares_storage` 用 `base is source` 判定，
  在**源本身是视图**时（状态连续 slice / reshape）必然失败，地址随即被复用——
  HIP 上表现为跨 chunk 合并后 D2D `code=1`，CUDA 上表现为捕获图与逐节点
  `execute` 首步分叉。判据：先让输出独占缓冲，再把记账统一到底层持有者；
  验收时捕获图必须与 `execute` 数值一致（后者不走回收，是安全基准）。
- **训练基准审计**（5.288）：默认接 `backward_pipeline()`，本例从 30 降到
  22 节点；用 `--raw-backward` 对照旧路径，不能混用旧成绩。计时前校验完整
  loss、全部梯度与一步更新，CPU 参数必须复制防止共享存储；训练 `--compile`
  尚未支持。共享 GPU 可能同时扭曲绝对耗时和比值，`wait_event` 不是主机同步。
  自动图本例每步仍复制约 8.51 MiB，贡献比例尚未取得干净消融。A800 空闲
  窗口两批复测：优化/off 1.073～1.098 ms、auto 0.509～0.536 ms；eager 约
  0.583～0.597 ms。后来新训练占卡，污染的剖析/消融耗时不作结论。等待与
  探针已完成，结果入 DESIGN；不能只凭等待的 PID 退出就断定 GPU 独占。
- **转置并入GEMM**（5.289，代码 `0120563`）：仅CUDA、静态非空二维、相邻且单
  消费者并非图输出时融合；f32/f16/bf16覆盖，物理源行宽决定leading dimension。
  MLP执行组21→18，消除3个转置副本；同进程15轮×40步交替，中位数off
  1.0481→1.0013 ms（下降4.5%），auto 0.4817→0.4645 ms（下降3.6%）。
  A800完整CUDA回归372项、跳过1项；本地42项通过。批量、符号维、空矩阵、
  多消费者、独立转置输出及HIP保持原路径；图变更必须使融合计划失效。
  原始轮次入DESIGN，临时消融脚本完成后删除；普通路径仍明显慢于torch eager。
- **生命周期回收计划**（5.290，代码 `7d69133`）：逐步全图计数和逐边扣减已移到
  计划期，保留原回收顺序、输入/输出/缓存常量和revision失效语义；新增5项边界
  测试，本地全量2880项、跳过446项，其余通过。24/96节点主机微基准同进程
  21轮交替，配对耗时下降13.75%/14.68%，不能当作GPU端到端收益。GPU有其他
  任务占用26690 MiB、利用率75%～100%，当时设备验证延期；后续已在5.291补齐
  CUDA回归与训练消融。临时脚本结果入DESIGN后删除，不干预其他任务。
- **执行组入口固化与GPU复测**（5.291，代码 `c2896f1`）：常量快捷入口和fast
  选择移到计划期；本地2881项、跳过446项，其余通过，测试机29项及CUDA372项
  （跳过1项）通过。同进程三版本15轮×40步交替：off 1.005797→0.960074→
  0.948206ms，回收/入口两项配对变化-4.328%/-0.957%，合计-5.726%；auto无
  明显变化。独立torch eager对照：off AF0.9613/torch0.5452，auto AF0.4746/
  torch0.5679ms；保留off异常轮和auto下降趋势，不把跨进程差当优化收益。
  本轮HIP未测。输入形状相等检查因符号图略退化已撤回。下一步分离图外参数更新
  与输入复制成本。GitHub主分支推送被服务端hook拒绝，不绕过或强推。
- **设备态性能数字只能在空闲卡上取**（5.281）：同一段代码、同一台 A800，外部占用
  91%～100%（显存被占 6.3～21.8 GiB）时设备态 22.8～25.7 ms/token，空闲时 8.2；
  宿主侧只差 6%（87.7～90.2 对 79.9～81.2）。宿主路径被整块状态往返拖住，对 SM
  争用不敏感；设备路径吃时钟与同卡其它进程。测这类数字前后各记一次 `nvidia-smi`
  （利用率 / 显存 / SM 时钟），跨时段的单点数字不能直接比。
- **跨条目的宿主每步成本不可横比**（5.282 / 5.283）：同一段 decode 代码在 A800 上
  量到 8.7 ms/步（`resident=True`）与 106.8 ms/步（`resident=False`），5.281 记的
  宿主是 77.3 且标注驻留；5.283 同机同口径量到的束搜索宿主是 92.9 / 117.4 /
  128.6 ms/步，而 5.282 文档记的是 12.98 / 22.07 / 36.01——同机检出 5.282 的提交
  重跑，宿主列是 91.8 / 111.6 / 122.2，说明 5.282 的宿主列不在这一口径内。新数字
  只在同一口径内比较，不拿旧条目做加速比。
- **捕获计划的越界守卫是逐算子读回的**（5.283）：`CudaGraphPlan._check_guards`
  对每个 `embedding` 的 err 标量做一次 `int(err.item())`，48 个状态重排节点就是
  48 次同步读回，成本约 2.4 ms/步。把重排并进 decode 图只省掉一次 replay 的边界
  （0.8～1.5 ms），守卫总数不变；要再省必须把同一张图的守卫汇总成一个标量。
- **设备张量不要用整数索引降维**（5.292）：`tensor[0]` 在 numpy / cupy 上
  得到降维视图，天数 `DeviceArray` 只支持切片、会抛 `TypeError`。设备后端要
  去掉长度 1 的前导维时按形状推导后 `reshape`；`attention` 的 `(1,1,S,T)`
  mask 已改为这条路，调用方不再需要自己 squeeze。
- **坏包跳过不能连原因一起吞**（5.293）：`iter_installed` 静默跳过校验失败的
  硬件包，会让"包在、但被改过"与"根本没装"都报成"未注册的后端"。跳过行为可以
  保留，但原因要能被装配层取到；`scan_errors()` 只在错误路径上调用，不要放进
  热路径。
- **HIP 上同进程第二个设备态会话会坏**（5.282）：24 层 / `d_model=1024` 时，先后
  建两个 `LMRunner`（即使换后端实例）会给出 NaN logits 或"捕获期 arena 不够
  4 MB"；纯 `generate`（不涉及束搜索）同样复现，一个进程一个配置就正常。基准
  脚本按"每个配置一个子进程"写。**本轮在本地 WSL 的 RX 7650 GRE（ROCm 7.2）
  上系统复测未复现**：host→device 两种顺序、同后端实例 / 换实例、两个
  `device_state=True` 会话同存活、host beam → device beam、`steps=32/tokens=32`、
  `_stream_graph` 的 `steps=1/8 × batch=1/2` 全部正常。可能已随 5.282 的
  `_decode_plan_for` 形状修复消失，也可能依赖当时的机器状态；在没有新证据前
  不改代码，只把它从"已知缺陷"降级为"历史记录·当前未复现"。
- **HIP `d_model=512 / 12 层 / vocab=8192 / batch=2` 的 `generate_batch` 出 NaN**
  （5.282 顺带发现）：同配置 batch=1 正常，`d_model=1024` 的 batch=2 正常。
  **本轮同机复测未复现**：`chunk=2/4/8/32` × `plain/device` 全部与参考后端一致；
  保留这条是为了提醒"形状相关的内核问题"曾经存在，而不是断言现在仍有。
- **`Module.param` 登记 `Parameter` 必须替换前端记录**（5.294）：`graph.input()`
  返回的是 `Tensor`，只把 `.value` 交给 `Parameter` 而不调 `_replace_tensor`，
  `backward()` 的缺省求导目标仍是旧 `Tensor`——梯度能算出来，但不会写回参数，
  现象是"训练一步不动"，且前向数值完全正常。新增参数类入口时照此检查。
- **`Module.adopt` 必须拷贝外部数组**（5.294）：torch 的 `state_dict()` 直接
  `.numpy()` 得到的是共享内存视图。不拷贝就是两个框架写同一份权重，双方优化器
  各走一步、参数只前进一次；对拍征兆是第 1 步参数差 2×lr 且方向相反。
  回归用例：`test_adopt_copies_data`。
- **长程 Adam 对拍只卡 loss**（5.294）：Adam 更新近似 `sign(grad)`，两侧 1 ulp
  的浮点差能翻转单个参数的更新方向（±lr），100 步后逐参数差到 1e-4 量级属正常。
  判据分层：前 20 步逐参数 < 2e-5，全程 100 步看 MSE 相对偏差 < 1e-3；拿 100 步
  逐参数差当硬门禁，会在实现正确时随 BLAS 版本随机变红。
- **`Tensor` 的算子方法是算子表的转发**（5.294）：`__getattr__` 只接受 `OPS`
  里的名字，签名与 `Graph` DSL 一致（`x.reshape(shape=...)` 而不是
  `x.reshape(3, 2)`）；不改 `Graph` 的调用约定，也不为 PyTorch 风格另开一套。
- **autocast 重写未命名输出不能沿用旧值名**（5.295）：`output_names` 对未命名
  输出返回的是值名 `v<id>`；重写插入 cast 后值会重新编号，把旧名当别名写进新图
  就会与自动名撞车（`名字重复: v245`）。任何 `Module(...)` 构图 + 混合精度都会
  撞上。判断"有没有显式别名"要拿原值的 SSA 名，不能用重写后的新值名。回归：
  `test_rewrite_does_not_collide_and_matches_f32`。
- **设备态续训要迁两处状态，不只是参数**（5.295）：`load_training_state` 在显存
  建立之前执行，累积窗口与优化器槽位都会恢复成主机数组。续训第一步会连撞两次：
  ① 窗口里 cupy 梯度撞上 numpy 累加和；② Adam 的 `m/v` 还是 numpy，与显存参数
  同算时报 `CScalar` 类型错。修法分别是 `GradientAccumulator.add` 按梯度设备迁移、
  `Optimizer.rebind` 让槽位跟随参数；不要用隐式主机往返"修好"它。回归：
  `test_device_state_resume_restores_pending_accumulation`。
- **别用臆造形状判断 GEMM 差距**（5.298）：先抓真实清单再下结论。大档 HelixLM
  （4.28M 参数 / batch=2 / seq=8）一步是 **196 次 batched GEMM，前导维 (2,8)，
  不是合并后的 16**；torch 侧 `aten::linear` 展平成 `mm([16,256],…)`，用的是
  `ampere_sgemm_32x32_sliced1x4_tn` / `gemmSN_*` 小 tile。真实形状 A/B：
  AF 的 batched 在五个主形状上比 torch 快 12%~53%（`lm_head` 快 2.1×），只有
  K=16 的权重梯度与 batch 内 16x16 小 bmm 慢 1.2×。反过来把 `(2,8)` 合并成
  `16` 反而慢 12%~21%——**不要为了对齐 torch 去合并前导维**。
  另一方面，大 M 窄 K 形状（如 `(131072,8)x(8,768)`）是写带宽受限（271 μs
  ≈ 1.50 TB/s，A800 HBM 饱和），那里 sgemm / cublasLt / torch.mm 三者同速，
  换 tile 不可能有收益。5.297 的「57.3 μs/次」均值把不同形状混在一起，不可
  用于归因。
- **布局物化的账要按 kernel 数算，不是按字节算**（5.299）：一步 1174 次
  `ascontiguousarray` 物化（90 MB）在 A800 上值 **2.4 ms（9%）**，而同样字节的
  纯流量按峰值只有 0.12 ms——成本几乎全是每次约 1.7 μs 的固定项。标定单次成本
  要捕成小图按 replay 摊（`us ≈ 1.7 + 字节 / 0.2 TB/s`，转置的有效带宽只有约
  0.2 TB/s）；用 `event` 包住 Python 调用会把分配开销算进去、高估 2~3×。
  计数还有两个坑：`transpose` 本身零拷贝，别把它和 `ascontiguousarray` 记成
  两次；设备态捕获会**预热一轮再捕获一轮**，Python 侧计数必须除以 2。
  同模型 torch 41.68 ms vs AF 26.07 ms（AF 快 1.60×），这一项不是差距主因，
  不值得为它改 strided 寻址。
- **单内核标定不能用 `plan.run()`**（5.300）：`CudaGraphPlan.run()` 会对每个
  输出做 D2H `to_host` 并逐个同步，8192 元素的逐元素一次被报成 **30 μs**，
  高估约 20×，一版数据全部作废。要用 `run_device()`（只 replay + 同步），
  且 replay 本身还有固定的 Python/下发开销，必须用**两点法**：同形状捕 R 次
  与 R0=4 次两张小图，`(t_hi - t_lo) / (R - R0)` 才是单次成本。
- **大形状广播慢在寻址，不在带宽**（5.300）：同样是 131072 元素的 add，
  同形连续访存 **2.28 μs ≈ 691 GB/s**，广播 `(8,2,512,16)×(8,2,512,1)`
  **7.55 μs ≈ 208 GB/s**（3.3~3.5×）。`ew_broadcast` 每个元素用最多 4 维的
  `%` / `/` 还原坐标，而 `shape` 是运行期参数，编译器不知道它们都是 2 的幂，
  只能发 64 位整数除法。这一步里 112 次 131072 元素广播合计 **0.87 ms**
  （8.1% step），是 elementwise 链（1.84 ms / 17.1%）里唯一集中的可挖项；
  8192 及以下已贴着 launch 下限。下一步：主机侧传 `log2(shape)` 与标志，
  用移位/掩码替代 div/mod。
  另注：5.300 的基准是 Mamba LM（3.39M 参数、fwd 608 / bwd 2722、
  9.92~10.78 ms/步），与 5.296~5.299 的 Transformer（4.28M、998 / 2464、
  26.07 ms）不是同一模型，绝对 ms 不可跨节比较。
- **远端执行与文档同步的两个坑**（5.300）：`_remote.py exec` 对带引号的
  命令不可靠——`git commit -m "docs: ..."` 只收到 `docs:`，`python3 -c "..."`
  被切成 `import` 一个词；复杂命令要写成脚本文件上传后执行，提交信息用
  `-F 文件`。本地上传的文本是 CRLF，直接 `git add` 会让整篇 diff 变成重写
  （5.300 首次远端提交 19434 行增/19303 行删），必须先
  `bytes.replace(b"\r\n", b"\n")` 再提交。另外远端只是运行副本，DESIGN /
  REMIND 不必同步进它的 git。
- **`round` 只能是 `nearbyintf`，`sign` 不能写成差值**（5.301）：`roundf` 是
  half-away-from-zero，`torch.round` / `np.round` 是 half-to-even，差在 x.5 上，
  且只在正好 .5 时出现——数值用例要故意喂 ±0.5 / ±1.5 / ±2.5，否则测不出来。
  `sign` 写 `(x>0)-(x<0)` 会把 NaN 变成 0（两个比较都假），必须用嵌套三目把
  原值透传。
- **bool 张量是 1 字节存储，写回不能套 store helper**（5.301）：`_STORAGE`
  多一个 `"bool" → unsigned char`、`BOOL_DIALECT` 提供 `af_ld_q` / `af_st_q`
  做 0/1 与 float 的转换；但 `Store` 对 bool 输出必须 `raw=True`，否则 helper
  会把 float 语义带进来。先例是 `ew_cast` / `ew_mod_int`，新增 bool 产出算子
  照抄。
- **三输入广播的 `_collapse_dims` 要传三组 stride**（5.301）：签名已泛化成
  `_collapse_dims(shape, *strides)`，二元调用方解包 `(dim, step_a, step_b)`
  不受影响；`where` 传 `(cond, a, b)`。合并判据必须对三组都成立，否则会把
  cond 的 stride 当成 a/b 的、合并出越界下标。
- **替换以"后续定义行"结尾的代码块会删掉那个定义**（5.301，本轮踩了两次）：
  用 `def broadcast_template(...)` / `class TestTorchExportOnCuda` 当 old_string
  的结尾、new_string 忘了原样保留，结果 `NameError: broadcast_template` 与
  测试类凭空消失。锚点要么选"上一段的最后一行"，要么把锚点行原样抄进
  new_string 末尾。改完立刻跑一次相关测试，别等到全量。
- **临时脚本放在仓库根会被同名包遮蔽**（5.301）：`g:\Cloumi` 下就有 `aetherflow`
  目录，脚本 `import aetherflow` 命中的是仓库根（命名空间包），报
  `is a namespace package, use of unknown location`。要么把 cwd 设成
  `g:\Cloumi\aetherflow`，要么用
  `python -c "import runpy; runpy.run_path(r'脚本', run_name='__main__')"`。
- **接入前端算子前先 grep 测试里的"不支持"反例**（5.301）：`sign` / `floor`
  原本被 `test_unsupported_op_is_rejected` 当反例，接入后两条测试直接翻红
  （`UnsupportedOperatorError not raised`）。反例要换成真的不支持的算子
  （现在是 `torch.trunc`），并补正向用例。
- **`topk` 不能做成单算子**（5.301）：`Op.infer` 返回单个 `TensorType`、
  `reference` 返回单个数组，values + indices 必须拆成两个算子，或在前端用
  `argmax + eq/where` 组合。设备端还需要"k 轮块内选择 + 跳过已选下标"的专用
  内核（`Shared("int","sel",k)`，cache key 含 k），不能像 `where` 那样只加
  一路 stride。**已按拆两个算子落地**：`topk_values` / `topk_indices`，前端
  `getitem` 的两个槽位各映射一个节点，DCE 之后只留用到的那一个；`topk_values`
  的反向重算下标后用 `gather_backward` 散射回原位置（`x_shape` 要求静态形状）。
  设备内核已在 5.305 补齐（k 轮块内选择、动态共享内存放已选下标），三个算子
  在 CUDA/HIP 上真实执行，不再抛 `BackendCapabilityError`。
- **`np.argsort` 的 NaN 位置与 torch 相反**（5.301）：numpy 升序把 NaN 放到
  末尾，取负做"降序"时 NaN 仍留在末尾，等于被 topk 排除；`torch.topk` 把 NaN
  当最大值排最前。跨框架对齐排序语义时必须显式
  `np.where(np.isnan(x), np.inf, x)` 再排序，否则含 NaN 的输入两边选出的下标
  完全不同——而且普通随机用例永远测不出来。
- **pass 的 `run()` 返回新图，返回值必须接住**（5.301）：`DeadCodeElimination()
  .run(g)` 不原地改 `g`，`GraphRewriter.finish()` 给出的是新图；把返回值丢掉
  再检查 `g.nodes`，会看到"该删的节点还在"，很容易误判成 pass 失效。测试里
  断言 pass 效果一律写成 `pruned = Pass().run(g)`，之后只读 `pruned`。
- **GPU 上的次正规数比不出"逐位一致"**（5.301 上机验证）：V100 + CUDA 12.4 +
  cupy 14.2 上，`cp.sign(1e-38)` / `cp.greater(1e-38, 0)` 全给 0，即 FTZ
  生效；numpy 参考不冲刷。把 `1e-38` 写进一致性用例必然报一个与代码无关的
  失败（本次 `sign_special` 就是这么红的）。判据：设备用例的输入取正常数域
  （`1e-30` 即可），subnormal 的差异当环境限制记录，不要试图在参考后端里
  模拟 FTZ 去"凑一致"。
- **远端验证环境的三件坑**（5.301）：① 共享机（`/root/afv` 独立目录，
  勿动共用机的 `/root/AetherFlow`）只装了 python3.10 + 无 pip/numpy，apt 源
  索引过期时要先 `apt-get update`，cupy 用
  `pip install cupy-cuda12x -i 清华源`；② paramiko `exec_command` 的命令里
  **不能用双引号**（本地 PowerShell 5 会吞掉，反而让远端把 `grep -E a|b`
  当管道拆开）；③ SSH 连接关闭不会杀掉远端进程，`apt-get` 之类的长任务会在
  后台继续持锁，重试前先 `ps` 看有没有自己的残留。
- **内核里的局部数组 + 动态下标会落到 local memory，比 64 位 div 还贵**
  （5.300 实测）：广播坐标还原原本想用"掩码 + 右移"替掉 `%` / `/`，为此
  多传一组 `int sh[4]`（各维 log2）。V100 上 131072 元素广播反而**慢 14~15%**
  （20.60 vs 17.70 μs；where 27.12 vs 23.62 μs，两次独立测量一致）。手写
  三内核对照定位到真凶是 `sh[d]`（`d` 是循环变量）：局部数组被动态下标寻址
  时，编译器只能发 local memory 访问，抵消了 `&` / `>>` 的收益。改用
  `31 - __clz((int)shape)` 去掉数组后与 div **打平**（1.00×），说明这条优化
  本身没有净收益，pow2 分支已整体删除。以后写内核，小数组要么用编译期
  常量下标、要么完全展开，否则默认按 local memory 估性能。

- **分桶反向不是免费的**（5.302）：把反向按参数桶拆开，能让"桶归约"与
  "后续桶反向"重叠，但每个桶都要从损失重算一遍共同链式路径，桶数就是反向
  计算的倍数。因此 `overlap=True` 强制 `bucket_bytes > 0`，禁止逐参数一桶；
  桶应该按层大块切（少而大），否则计算量爆炸。
- **重叠归约只能用单线程 FIFO**（5.302）：集合调用要求各 rank 同序；桶的
  完成顺序由反向图执行顺序决定，各 rank 图相同所以顺序一致，但线程池的
  提交顺序取决于调度，一侧先发 A、另一侧先发 B 就会互相等死。`overlap=True`
  强制 `threads=1`，后台线程按提交顺序发送。
- **`backward(..., active=...)` 要传图的输出别名**（5.302）：传 loss 节点名
  （`v19` 之类）会报"active 里有图输出中不存在的别名"；构造 Trainer 时把
  用户给的输出名记住，分桶反向直接用它。
- **设备数组不要用 `isinstance(np.ndarray)` 判断**（5.302）：cupy 数组走
  `__cuda_array_interface__`；用 `np.asarray` 会把整份梯度拉回主机，正好
  抵消设备侧归约的收益。设备态 DDP 只读 `.shape` / `.dtype` 元数据。

- **投机循环要与会话生命周期解耦**（5.303）：把草稿 → 验收 → 提交抽成模块级
  `_speculative_loop`，让 `SpeculativeDecoder` 与
  `LMRunner.generate_speculative` 共用一份实现。分开写一定会分叉（停止符、
  奖励 token、提交块长的口径），而"投机 = 贪心"的等式只在两份口径完全相同时
  成立。
- **测试驻留时，草稿的 feed 不算数**（5.303）：`greedy_draft` 如果也走被记录
  的 backend，`fed` 里就会出现含权重的 feed，误判"驻留没生效"。草稿用裸
  backend，recorder 只记录会话自己的 execute。
- **`prime` 前缀可以做投机起点，但生成后必须清空**（5.303）：验收与提交都
  推进会话状态，`_advance` 返回的 state 已越过前缀；再保留 `_prefix` 会让
  下一次 `prime` 的错误不报或续接错位。
- **`lm` 与 `speculative` 的循环依赖用函数内导入打破**（5.303）：
  `speculative` 顶层依赖 `lm` 的 `_check_request` / `_prefill` 等，而
  `LMRunner.generate_speculative` 只需要 `_check_draft_tokens` /
  `_speculative_loop`；把后者放在方法体内导入，模块图保持无环。

- **换出必须同时释放页并清空块表**（5.304）：只拷副本不清块表，换出期间
  那些页会被别的序列领走，换入前的任何 `gather` 都会读到别人的数据。
  换入必须**重新领页**，不能沿用旧块表——页号只在块表里有意义。
- **换入失败要保持换出态**（5.304）：先领齐全部页再写回；页不够就抛错并
  原样保留主机副本，腾出页后可重试。半换入会同时占页与丢副本，是两种故障
  里最难排查的一种。
- **topk 的最大值初值要用 -inf 的位模式**（5.305）：`-3.4e38f` 作哨兵时，整行
  全是 `-inf` 的输入会同时不满足 `v > best` 与 `v == best`，一个元素都选不出来。
  用 `__int_as_float(0xff800000)`，配合 `v == best && j < best_i` 与
  `best_i = 0x7fffffff` 初值，`-inf` 行也能按小下标次序正常选出。
- **topk 的"已选"标记放动态共享内存，不要 `used[cols]` 字节表**（5.305）：
  一个 bool 数组把行宽钉死在 48 KiB 内；改成 `extern __shared__ int sel[k]`
  存 k 个下标后，行宽不再受限，只受 `4k + 4 KiB ≤ 48 KiB` 约束。
- **三个 topk 入口统一用 `void* outp`**（5.305）：values / mask 输出 `%T%*`、
  indices 输出 `int*`，签名无法一致；在 `extern "C"` 入口内各自转一次类型，
  device 函数保持一份，避免三份几乎相同的 k 轮选择逻辑各自漂移。
- **换出候选要跳过"已换出"和"长度 0"**（5.306）：LRU 循环里不跳过这两类，
  `_evict_one` 会返回"换出了一条"却没腾出任何页，外层重试循环看起来在推进、
  实际原地打转。管理器保证每轮至少换出一条真正占页的序列，重试次数才有
  "序列数 + 1"的上界。
- **pin 只约束自动换出，不约束显式 `evict`**（5.306）：服务层的约定是
  active 序列 pin、挂起序列 unpin；调用方明确要求换出时（`suspend(evict=True)`）
  越过 pin 是预期行为。把两者混成一个"不可换出"标志会让挂起与回收互相挡住。
- **解码步要"先算后写"**（5.306）：`step` 先执行单步图、再 `append` 写回，
  失败时序列长度与 K/V 都不变，重试同一步不会重复计数。写成"先写占位再算"
  会在页不足时留下半条序列。
- **`free` 要同时清 `_order` 与 `_pinned`**（5.306）：id 复用是常态，残留的
  使用顺序会让新序列"一出生就是最久未使用"，残留 pin 则让它永远不被换出。
- **"放手"和"腾出"必须分开算**（5.307）：页被多条序列共享时，`truncate` /
  `reset` / `swap_out` 返回"块表里放手了几页"会骗人。统一改成返回**实际回到
  空闲链**的页数（共享页只减引用），一条全共享的序列换出后返回 0。想让换出
  腾出容量就直接读这个返回值，别拿块表长度当容量。
- **拆页只拆被写到的页**（5.307）：追加落到全新页时不需要拆；落回共享的尾页
  才拆。整条 `fork` 时全拆等于没共享，而"写前不查引用"会让两条序列互相覆盖。
- **拆页的页数要进容量检查**（5.307）：`append` 必须把"新增页 + 要拆的共享页"
  一起和空闲页比，拆到一半失败会留下引用计数与块表不一致的缓存，比直接抛
  `PagedCacheFullError` 难查得多。
- **自动换出要跳过全共享序列**（5.307）：LRU 先搬走一条 `exclusive_blocks == 0`
  的父序列，腾出零页、再搬下一条，白多一次换出还抬高 `swaps_out`。用
  `exclusive_blocks` 预筛，保证每次自动换出都真的腾出页。
- **上层错误信息要指向本层的动作**（5.307）：缓存层说"先 swap_in"，但
  `PagedDecodeScheduler` 的调用方只认识 `resume`。调度器层先判一次换出态，抛
  "先 resume(rid) 再 fork"，比把底层术语透出去更好排查。
- **能广播 ≠ 能提出求和号**（5.308）：W8A8 的 `(int32(x) @ int32(w)) * sx * sw`
  要求激活 scale 沿 K 维恒定、权重 scale 沿 K 维恒定。`sx` 形状 `(B, K)` 虽然
  能广播到 `(B, K)`，但每列不同，乘到累加结果上就不再等于原来的点积——必须
  在 infer 处显式拒绝"沿压缩轴不是 1"的 scale，而不是静默当 per-token。
- **整数累加的溢出要当语义问题**（5.308）：`K * 127 * 127 > 2**31 - 1` 时
  int32 会回绕，而回绕结果依赖溢出行为，不能跨后端对拍。构图期（K 静态）与
  执行期各查一次，宁可拒绝也不"让它绕一圈"。
- **只有参考实现的新算子也要有 conformance 形态**（5.308）：`covered_ops`
  必须覆盖全部 IR 算子；设备端没有内核时，形态会被记成 refused，这是**显式
  拒绝**的正常路径，不要为了凑 executed 数量而在设备上退回 f32 累加。
- **同名相近的两套累加语义要分开立形态**（5.309）：`linear_i8` 是"读 i8 权重、
  反量化后走 f32 GEMM"，`linear_i8a8` 是"i8 输入、int32 累加"。K=512 的用例已
  证明两者数值不同，conformance 不能用一个覆盖另一个。
- **共享签名、分叉实现是跨厂商 int8 的正确姿势**（5.309）：CUDA 直接用
  `__dp4a`（sm_61+，V100 实测编译通过），HIP 的 hipRTC 不带 `hip_runtime.h`，
  用标量乘加。两条路都是精确整数运算，累加逐位相同——分叉的只是指令，不是语义。
- **设备端的形状校验不能省**（5.309）：IR 已保证 `x_scale` 是"前导维 + 1"，
  但设备执行也可能来自手工构图或旧序列化图；把 dtype / 形状 / 长度再查一遍，
  让错误报成语义错误而不是越界读。
- **整次 `execute` 的墙钟不能当内核吞吐**（5.309）：512³ 的中位 1.14 ms 包含
  图查找、传输与启动；文档里必须写清口径，否则就成了没有依据的"加速"。
- **"未接主干"有时是分类错误，不是欠债**（5.310）：`LMRunner` 的主干是 Mamba，
  没有 K/V，"把分页 KV 接进 `LMRunner`"本身就不成立。接主路径之前先确认主干
  有没有这个对象；没有就补在真正的主路径上（Transformer 增量解码），而不是硬塞。
- **新增一层不要改被复用函数的返回值面**（5.310）：`build_cached_decoder_step`
  的 `(inputs, y, new_kv)` 被多处解包，为词表头改元数会牵动所有调用方。给生成器
  单独一张 `(1,d)→(1,V)` 的小图、多一次 `execute`，换来零侵入——正确性优先时
  这笔交易划算。
- **生成闭环要自己持有 token 历史**（5.310）：分页调度器只记长度，`fork` 的
  前缀一致性校验需要 token 本身；由此暴露"fork 后 prompt 必须比父长一个 token"
  这类边界，写进 API 与报错文案，不要靠调用方猜。
- **"串联两个算子能算出来"不等于"能训练"**（5.311）：
  `dequantize_i8(quantize_i8(x))` 的前向就是假量化，但中间类型是 i8、
  `quantize_i8` 没有也不该有反向规则，训练图到这里就断了。前向等价时可以
  接受组合路径，**反向不可导时必须补独立算子**；设备端仍可复用那两个内核。
- **STE 不是数学导数，语义要钉在文档里**（5.311）：直通估计器让 `rint` /
  `clip` 几乎处处为零的解析梯度不挡训练，这是训练时的选择而不是求导结果。
  写清"梯度直通、饱和区不额外截断、scale 不接收梯度、学 scale 是另一条
  规则（LSQ）"，否则读代码的人会以为反向写漏了。
- **可学参数必须是图输入，不能是 `constant`**（5.312）：vjp 返回非 `None` 的
  梯度会写成对应输入的 cotangent；若那个输入是 `constant` 节点，cotangent
  到常量处按"零输入"被静默丢弃——不报错，训练就是不更新。给"参数"发梯度
  前先确认它是 placeholder，并把这条约束写进算子 doc 与测试。
- **前向相同、反向不同时，拆成两个算子而不是加属性**（5.312）：`fake_quantize_i8`
  与 `fake_quantize_i8_lsq` 的前向逐位相同，只差 vjp。用 `learnable_scale`
  之类的 attrs 也能实现，但"这个 scale 到底会不会被训练"就藏进了属性；
  两个算子名让 `backward(wrt=[s])` 从报错变返回梯度的差异直接写在图上。
- **外部权重格式自己解析，不为一个容器加依赖**（5.313）：运行期只依赖 numpy 是
  "参考后端在任意平台可用"的前提。safetensors 的头就是 8 字节小端长度 + 一段
  JSON，标准库足够；而且校验口径（头长上限、offsets 边界、字节数与 shape 相符）
  要按本项目的"显式失败"自己定，通用读取器的默认宽容度不一样。
- **装载不能猜名字、布局与精度**（5.313）：HF 的 `...q_proj.weight` 与本地参数名
  没有可推导关系；PyTorch `nn.Linear` 是 `(out, in)` 而本仓库 `x @ W` 是
  `(in, out)`。猜错的后果不是崩溃，是静默训出一个错模型——所以映射、转置、
  有损 dtype 都必须显式声明。另：先全量校验再 `dst[...] = ...` 原地写，
  避免失败时留下"一半新权重、一半旧初始化"。
- **分词器的分歧不会报错，只会让模型变笨**（5.314）：预分词边界偏一位、BPE
  合并顺序换了、特殊 token 被拆成字节，产出的 id 都"合法"，只是不是训练时那串。
  所以判据必须是 id 级对拍（往返、rank 顺序、`"  hello"` 的空白切法、`it's`
  的后缀），而不是"encode 出来看着像"。规则里 `\p{L}` 别用 Python `re` 近似成
  `[A-Za-z]`——那会让中文整段错位；用 `unicodedata.category` 的 `L*` / `N*`。
- **窄支持面优于近似实现**（5.314）：Llama-3 的 `Split` 预分词带一个会改切分
  边界的正则（数字按 1～3 位分组），忽略它就是拿别的分词器当自己的用。宁可
  显式报"不支持 Split"，也不静默产出"看起来能用"的 id。
- **远端 V100 的凭证与环境**（5.304 / 5.309 / 5.320）：两条线路同一个端口与
  密码，移动 `ssh root@223.109.239.36 -p 24432`、电信
  `ssh root@180.127.11.177 -p 24432`，账户 `root` / `vipuser`，密码 `hogheiW9`。
  单卡 Tesla V100-SXM2-32GB（32768 MiB，compute_cap 7.0 / `sm_70`，驱动
  550.127.05），Ubuntu 22.04（内核 6.2.0-37）、python3.10.12 + numpy 2.2.6 +
  cupy 14.2.0，系统未装 nvcc。同步方式：本地
  `tar -czf`（排除 `.git` / `__pycache__` / `.pytest_cache`，含 `aetherflow
  tests benchmarks examples pyproject.toml verify_cuda.py`，约 1.37 MB）→
  paramiko SFTP 上传 `/root/af_latest.tgz` → `/root/afv3` 解压。上一台
  `223.109.239.32:20824` / `180.127.11.166:20824`（密码 `co7eeNie`、
  `/root/afv2`）已释放，别再用。
  SSH 偶发 `banner` 被重置，连接要带 3–5 次重试，别把一次失败当环境坏了。
- **两条执行路径要逐条对齐检查**（5.320）：设备后端同时有逐节点 `invoke` 与
  计划期预绑定后的批量派发快路径。5.318 只在 `invoke` 里判 fp8，`_planned_kernel`
  只扫输入 dtype，于是 `cast`（输入 f32、输出 fp8）被预绑定、绕开检查，真机上
  表现为 "算错" 而不是 "拒绝"（`verify_cuda.py` 从 2 refused 变 2 failed）。
  预绑定条件必须与 `invoke` 的检查一一对应，**输入与输出都要判**；判不了就返回
  `None` 退回原路径，报错时机与文案不变。
- **NVRTC / hipRTC 源码只能是 ASCII**（5.325）：`device_shared.py` 的 `_SRC`
  里写中文注释，本地看不出来，真机上抛
  `UnicodeEncodeError: 'ascii' codec can't encode characters`，并被 conformance
  记成 `failed`（不是编译错误）。内核注释一律用英文，Python 侧的注释不受影响。
- **设备端 fp8 是 f32 载体**（5.325）：`cast` 到 f8 只把值舍入到格式格点，
  不省带宽也不省显存；`invoke` 与 `_planned_kernel` 只对 `cast` 放行，其它
  算子仍 refused。`_fast_step` 的量化标记要包含 `F8E4M3` / `F8E5M2`，否则
  图输出会跳过 `quantize`、真机表现为 failed。
- **fp8 的"格式"在设备端不可见**（5.327）：载荷是 f32 载体，内核从数组
  dtype 分不出它是 e4m3、e5m2 还是未量化的 f32。所以设备端**不能**靠 dtype
  兜底，只能信 IR 声明 + 写入侧的格点校验（`load_fp8_linear`）；`linear_fp8`
  的内核注释写死了这条分工。放行名单用共享常量 `FP8_OPS`（cast + 两个
  线性层），`invoke` 与 `_planned_kernel` 同源判定，别再各自维护一份。
- **1 字节 fp8 用 i8 位模式承载**（5.328）：`pack_fp8` 的输出是 `int8` 的
  **位模式**，不是整数——`-128` 就是码 `0x80`。设备端读 `unsigned char`
  再解码，主机侧比对必须先 `view(np.uint8)`；直接拿 int8 值去比码会在
  负码上差 256。`linear_fp8_packed` 不在 `FP8_OPS` 里（输出是 f32，图里没有
  fp8 dtype），但它的权重输入是 1 字节，必须进 `_INDEX_ARGS`
  （`{1}`），否则字节数推断会按 f32 算成 4 倍。
- **packed 路径只降存储，不降算力**（5.328）：V100 是 sm_70，没有 fp8
  tensor core，也不存在吃 1 字节权重的 GEMM；`af_linear_fp8_packed` 是
  每线程一个输出元素的朴素内核（逐元素解码 + f32 累加），速度与 f32 GEMM
  持平甚至略慢。要速度得换硬件，不是换内核。逐组 scale 与 W8A8 的打包
  激活都未接，构图期显式拒绝，别静默降级成逐张量。
- **真机脚本不要硬编码 SSH 密码**：本机执行策略会以"物化具体凭证"为由
  拒绝运行（哪怕凭证已在本文件里）。需要真机回归时先向用户明确说明风险并
  取得授权，或用用户自己的终端执行同步与验证命令。
- **W8A8 的连续量只有两路 scale**（5.329）：`linear_i8a8` 的 vjp 对 int8 码
  返回 `None`，`backward` 对码求导会抛 `NoGradientError`；梯度只送
  `x_scale` / `w_scale`（`sum_to` 归约到声明形状），bias 沿前导维求和。
  想训就显式用 `build_int8a8_linear_qat`，别在部署路径里偷偷给码加 STE。
- **逐组 scale 只能沿 K 分组**（5.329）：`linear_i8_group` 的 scale 形状
  `(K/G, N)`，`K` 与 `K/G` 必须静态且整除（`_check_group_params`）。沿 K
  逐元素变化的 scale 提不到 int32 累加外面——这是 `linear_i8a8` 拒绝逐元素
  scale 的同一条理由，不是没实现。设备端是 `dequantize_i8_group` + f32
  GEMM 组合，**没有专用内核**，别以为它比 `linear_i8` 快。
- **QAT 与部署只共享"格点"**（5.329）：`fake_quantize_i8` 与
  `quantize_weight` / `quantize_activation` 共用 RNE + 饱和，所以导出得到的
  int8 码逐位相同；但训练走 f32 matmul、部署走 int32 累加，末位可能差
  1 ulp。写文档时别说成"逐位相同"。
- **融合 epilogue 必须显式 RN**（5.329）：`linear_i8a8_zp` 的 bias 进内核后，
  `(float)acc * xs * ws + bias` 若让编译器发 FFMA，与"两颗内核各自 RN 舍入"
  会差 1 ulp。用 `__fmul_rn` / `__fadd_rn` 钉死；带 bias 与无 bias 两个入口
  共用 `af_i8a8_zp_acc`，累加体只能有一份。dp4a 仍延后：零点相减后不再是
  int8，要复用整数点积只能走 `P - zx*Sw - zw*Sx + K*zx*zw` 的展开，等有
  真机能对拍再动。
- **动态 per-token scale 的零 token**（5.329）：`amax == 0` 时 `where` 必须
  给 1.0；给 0 会被 `quantize_i8` 以"非正 scale"拒绝。整条链只用既有算子
  （`ge` / `neg` / `where` / `reduce_max` / `gt` / `div` / `quantize_i8`），
  没有新增内核，量化格点与 `quantize_activation` 逐位相同。
- **5.329 的真机债**：`linear_i8_group` 的 conformance、
  `af_linear_i8a8_zp_bias` 的编译与逐位、动态 scale 链在 V100 上均未验证；
  新增内核注释为 ASCII（沿用 5.325 的约束）。
- **bool 求和不叫求和，叫计数**（5.330）：`reduce_sum(mask)` 在 numpy 里会
  把 true 当 1 累加，读代码的人却以为在求和。5.330 起 `_infer_reduce` 在入口
  拒绝 bool（`reduce_max` / `reduce_mean` 同样拒绝，它们会退化成 OR / 比例），
  错误信息给出两条替代：判定用 `reduce_all` / `reduce_any`，计数先 `cast`
  到 f32。别再给 bool 开"参考端能跑就行"的口子——两端不一致比不支持更糟。
- **逻辑算子不猜"非零即真"**（5.330）：`logical_and` / `logical_or` /
  `logical_not` 只接受 bool，浮点输入直接拒绝并提示 `ne(x, 0)`。隐式转换在
  NaN 与负数上会与显式比较分叉，等调用方踩到再解释就晚了。设备端复用比较
  模板（`&&` / `||`）与一元模板（`lnot`），**没有新内核**。
- **bool 结果必须 raw store，方言按 dtype 自动选**（5.330）：`logical_*` 的
  输出是 1 字节 bool，走 store helper 会被按 float 语义写宽。CUDA / HIP 的
  `render` 现在自己判断 `_dtype_key(dtype) == "bool"` 并选 `BOOL_DIALECT`，
  调用点不必各自记住传方言；新增 bool 产出算子时只要走 `render` 就安全。
- **cast 跨 bool 没有梯度**（5.330）：设备端 f32↔bool 走 `src != 0` / 升 f32，
  f16 / bf16 必须经 f32 中转（直连 `__half` 会 Unresolved extern，bf16 的
  uint16 按数值读会变成位模式）。反向对"源或目标是 bool"显式抛
  `ValueError`——掩码不该进损失；`reduce_sum(bool)` 的反向同理（原先会静默
  把 bool 梯度传回去）。
- **bool 归约里的"和"只是计数器**（5.341）：`reduce_any` / `reduce_all` 复用
  `block_reduce_sum` 的骨架，判定必须在块内归约**之后**——逐元素阶段只能
  确定"全假"，"全真"要等所有元素加完。`all` 的等式"和 == 列数"要求求和
  精确，因此轴长超过 2^24 显式拒绝（float 的整数精确区间），`any` 与它
  同口径。
- **5.341 的真机债**：`reduce_any` / `reduce_all` 已进 `GPU_OPS` /
  `_BOOL_OPS`，但 CUDA / HIP 上的执行、逻辑组合与 bool cast 的设备路径仍只有
  源码级检查，V100 / gfx1100 未回归。
- **能力声明按"库能不能加载"，不按"路径有没有接"**（5.331）：
  `collective_support` 先前恒报 `device_allreduce=False`，理由是"路径没接"，
  但 5.228 / 5.244 之后库已能加载。声明与实际不符会让调用方永远不敢用设备侧
  路径。现在的判据只有一个：`nccl.vendor()` / `version_string()` 成功即 True。
- **默认值可以改变行为，但必须写进 DESIGN**（5.331）：
  `DataParallelTrainer` 的 `collective` 从 `"host"` 改成 `"auto"`，在有 NCCL
  的机器上会把既有调用的通信从主机搬到设备。这是有意的（设备态梯度本就在
  显存里），但"默认值变更"属于对外行为，必须在小节里点名，不能只藏在代码。
- **不静默降级是硬约束**（5.331）：设备侧不可用时抛
  `DeviceCollectiveUnavailable` 并给出 `device=False` / `collective="host"`
  的出口，而不是悄悄走主机中转——否则调用方会以为通信留在设备上，性能结论
  整体失真。
- **uid 分发先挂线程再初始化自己**（5.331）：`ncclCommInitRank` 阻塞到全员
  到齐，rank 0 若先初始化再分发，其余 rank 会在 `fetch_unique_id` 上白等到
  超时。顺序反过来才对；`world_size == 1` 直接返回 `_LocalComm`，单卡调试
  不该被"必须先装 NCCL"挡住。
- **切卡与 `ncclCommInitRank` 必须合并**（5.331）：`nccl.init_rank` 先
  `cp.cuda.Device(device)` 再建 comm。分开写的话，"comm 绑错卡"要到第一次
  集合调用才炸，错误现场离配置错误太远；没有 cupy 时直接抛 `NcclUnavailable`，
  不猜卡号。
- **5.331 的真机债**：本机没有 NCCL / RCCL，所有"有库"分支靠 mock 覆盖；
  `ncclCommInitAll` 多卡归约、`NcclDeviceCollective.from_env` 的 uid 分发与
  两卡数值用例均待授权真机回归。`overlap` 仍为 False（它声明的是设备态重叠），
  设备态强制 `bucket_bytes=0` / `threads=1` / `overlap=False`；主机态桶间并发
  已在 5.302，两者不是同一件事（5.338）。
- **可外提的前提是"纯"与"不读状态"**（5.332）：只按 `op.pure` 与 carried
  依赖判断；RNG 每轮重新采样、STATE / IO 有副作用，都不能外提，它们的
  消费者也不能——否则新体会引用一个留在体内的值。
- **常量不参与外提，但要跟着复制**（5.332）：`constant` 从 `invariant_nodes`
  里排除（外提它只会多一次每轮输入），但外提节点依赖的常量必须复制进
  preheader，否则少一条边；体内直接引用的常量就地复制，不变成循环输入。
- **外提值用原名接回**（5.332）：新体把外提值作为同名输入，carried 映射与
  输出别名因此都不用改，调用方也从 preheader 输出名直接对上。
- **重写后必须过 verifier**（5.332）：外提最容易出的事故是"新图少一条边"，
  而它看起来仍是一张合法 DAG；`split_invariants` 对 preheader 与新体各跑
  一次 verify，ERROR 直接抛。
- **写库与查库必须是同一个签名函数**（5.333）：执行期用
  `tuning.signature.block_signature` / `conv2d_signature`，离线脚本也必须用
  它们。手工拼 `{"work": 4096}` 这类字典很容易多一个键或把 tuple 留成
  tuple，结果是"库里有记录、运行时永远 miss"，且不会报错。
- **库里的坏值要报错而不是回退**（5.333）：配置记录等于"预置的
  `set_tuning`"。`block=96` 会被内核按 3 的倍数派线程，静默退回自动只会
  表现为"调优没生效"；`overlay` 对非正整数抛错，几何越界交给
  `plan_conv_tiled` 在执行前拒绝。
- **conv2d 只在三个旋钮全为 None 时查库**（5.333）：手动设过任意一项说明
  正在做 A/B；此时叠库里的值会分不清"这次改的到底起没起作用"。命中后用
  局部变量，不回写 `_conv_*` 字段。
- **反量化的输出不是 f8 载体**（5.342）：`dequantize_fp8_group` 的输出
  dtype 是 f32。用 f8 载体的话，执行器会在节点边界把
  `unpack(q) * scale` 二次舍入回 fp8 格点——`1.875 * 0.75 = 1.40625`
  被舍成 `1.375`，误差一整个间距（scale 不是 2 的幂时必然发生）。`unpack_fp8`
  本身保持 f8 载体，因为它只做位模式到格点值、结果仍在格点上。这条也决定
  了它不进 `FP8_OPS`：图里没有 fp8 类型，白名单只管带 fp8 载体的节点。
- **fp8 逐组与两端打包的收益在读带宽**（5.342）：`linear_fp8a8_packed` 的
  激活与权重各 1 字节，`linear_fp8_group` 的权重常驻 1 字节；解码都在寄存器
  里做，没有 fp8 tensor core，算力与 f32 GEMM 持平或更慢。`linear_fp8_group`
  在设备端还会物化一份 f32 权重副本（反量化 → sgemm），运行期显存峰值不降。
  三个新算子都没有 VJP 规则（量化权重与量化激活按冻结常量处理），设备端
  只做源码级验证，真机 conformance 待授权。
- **4 bit 的值与打包字节都装在 i8 里**（5.343）：不新增 `u4` dtype。值域
  0..15 的值是一个 i8，打包字节是 i8 位模式，与 fp8 的 pack / unpack 同构。
  加 0.5 字节类型会让所有既有内核多一条存储宽度分支，收益只是省掉显式的
  `pack_int4`。
- **unpack 必须先转无符号**（5.343）：`0xA5` 作为 i8 是 -91，直接算术右移
  会得到 -6；先 `(unsigned char)` 再取半字节才是 10。参考实现先
  `astype(uint8)` 再取位，设备内核用 `(unsigned char)` 读。
- **零点用 f32 承载而不是 i8**（5.343）：4 bit 的 zp 只有 0..15，两种承载都
  放得下；选 f32 是为了与 scale 同形同类型，设备内核直接按浮点读、形状校验
  只写一次。
- **4 bit 逐组只改量化粒度，不改算力**（5.343）：`linear_u4_group` 读的 `q`
  仍是每权重 1 字节（i8 承载 4 bit 值），反量化后走 sgemm，没有 int4 tensor
  core；存储减半要靠 `pack_int4` 离线打包，本版没有打包线性层，设备端还会
  物化 f32 权重副本。没有 VJP（量化权重按冻结常量处理），设备端只做源码级
  验证，真机 conformance 待授权。

---

## 版本记录

| 版本 | 日期 | 摘要 |
|---|---|---|
| v0.1.0 | 2026-11-20 | 首个版本：IR / 参考执行器 / 自动微分 / 训练与推理闭环 / CUDA 与 HIP 后端 / 一致性套件 |
