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
  **设备态解码**（`LMRunner(device_state=True)` / `ServingEngine(device_state=True)`）：
  decode 图捕获一次、状态常驻显存、每步只把 logits 取回主机（DESIGN 5.279～5.281；
  空闲 A800 24 层 81.2 → 8.2 ms/token，连续批处理 `capacity=8` 时 66.1 → 491.6 tok/s）；
  分页 KV cache 与贪心投机解码（组件级，
  见"已知限制"）；OpenAI 兼容 HTTP 子集（`aetherflow.serve`：`/v1/completions`、
  `/v1/chat/completions`、SSE 流式）。
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
```

### 已验证的配置

| 后端 | 环境 | 用例 |
|---|---|---|
| 参考 | Windows / Python 3.14 | 2812 个用例，429 跳过，全绿 |
| HIP | WSL + ROCm 7.2 / RX 7650 GRE（`gfx1102`） | 本轮复跑 `test_hip_backend` 72 条、`test_device_control` + `test_lm_device_state` 44 条，全绿 |
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
  Broadcast）。单进程多卡可显式接设备侧归约：
  `DataParallelTrainer(..., collective="device",
  device_collective=nccl_device_collective(backend, devices))`（`ncclCommInitAll`
  + 每卡原地 AllReduce，5.284），不注入时仍报 `DeviceCollectiveUnavailable`；
  多进程 `DistributedDataParallel` **仍走主机中转**，`overlap` 恒为 `False`。
  真机只到 `world=1`（RCCL 与 CUDA 各一条），两卡数值用例已写但当前没有多卡
  机器、未执行。ZeRO-1/2/3 与流水部分参见 DESIGN §7。
- **张量并行**：只有前向（列 / 行并行 + attention 头切分）。行并行返回**部分和**，
  需调用方归约；反向通信与优化器分片未接。
- **量化**：weight-only int8。权重驻留 1 字节，GEMM 仍走 f32 反量化；
  无 int8 累加、无 fp8、无 QAT、无非对称零点——**不要期待"int8 算得更快"**。
- **布局**：NHWC 传播 pass 已交付，但只有参考后端承接；设备后端对 `layout="nhwc"`
  显式报错，原生设备端 NHWC 内核尚未实现。训练图保持 NCHW（NHWC 无反向规则）。
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
- **推理**：分页 KV cache 与投机解码是组件级交付，**未接进 `LMRunner`**——主干是
  Mamba，没有随上下文增长的 KV。投机只支持贪心验收与 `batch=1`。
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
- **前端**：torch.export 子集。整数只有 i32（int64 收窄、越界拒绝）；
  无 `bool` 张量，因此 `where` / `masked_fill` / 比较算子 / `topk` 显式拒绝；
  `sign` / `round` / `floor` / `ceil`、`baddbmm`、原地 `copy_` 无对应语义；
  非整数 `pow` 指数、图内 `clamp` 边界、非零或 reflect 的 `pad`、非默认
  `softplus.threshold` 一律拒绝；常量构造（zeros / ones / full / arange 等）
  落成图属性，单个常量上限 65536 元素；`padding_idx` / `sparse` /
  `scale_grad_by_freq` 显式拒绝。
- **数值与内核**：`fft` / `rfft` 仅 CUDA 的 cuFFT 路径，且无 VJP；`topk_mask` 只有
  参考实现；`spike_linear` 的 `dspikes` 必须稠密；卷积的差距按稳态口径读
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
- **HIP 上同进程第二个设备态会话会坏**（5.282）：24 层 / `d_model=1024` 时，先后
  建两个 `LMRunner`（即使换后端实例）会给出 NaN logits 或"捕获期 arena 不够
  4 MB"；纯 `generate`（不涉及束搜索）同样复现，一个进程一个配置就正常。基准
  脚本按"每个配置一个子进程"写。
- **HIP `d_model=512 / 12 层 / vocab=8192 / batch=2` 的 `generate_batch` 出 NaN**
  （5.282 顺带发现）：同配置 batch=1 正常，`d_model=1024` 的 batch=2 正常，属
  形状相关的内核问题，待查。

---

## 版本记录

| 版本 | 日期 | 摘要 |
|---|---|---|
| v0.1.0 | 2026-11-20 | 首个版本：IR / 参考执行器 / 自动微分 / 训练与推理闭环 / CUDA 与 HIP 后端 / 一致性套件 |
