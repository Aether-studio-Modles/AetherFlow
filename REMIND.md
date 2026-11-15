# REMIND

规范与历史全部在 [DESIGN.md](DESIGN.md)；这里只记"再进来时先要知道的事"。

## 第一代版本（v0.1.0）

对外一页速览在 [README.md](README.md)；逐条现状在 DESIGN §1.2，已知边界在 §7。

这一代交付的是**一条闭环**：静态图（或 torch.export 子集）→ 语义 IR → 中层优化 →
可插拔后端（参考 / CUDA / HIP）→ 训练（反向、优化器、分布式最小闭环）→
推理（会话、前缀缓存、连续批处理）。核心命题"一套 IR 落到不同硬件"已在
NVIDIA sm_75 与 AMD gfx1100 上逐项对齐。

读代码的顺序：`ir/graph.py` → `ir/ops.py` → `runtime/executor.py` →
`backends/device_backend.py` → `lower/plan.py` → `lower/kernels.py` →
`backends/cuda/backend.py`（HIP 是同构的第二份厂商钩子）。

这一代**不**包括（下一代清单）：把 NCCL / RCCL 接进数据并行路径（5.244 只做到
加载、厂商判定与四个集合调用面，`DeviceDataParallel` 仍走主机中转）、int8 / fp8
**计算**、NHWC 布局传播、真正的设备端分支（5.245 的 `DeviceWhileLoop` 只是把
条件与体各捕获一次、状态留在显存，每轮仍要读一次谓词）、把 schedule 搜索接进编译流水线（5.242
只有离线成本模型与按估算排序的串行搜索）、开放式的包发布（CI 只在构建 wheel 后
上传 artifact，没有 PyPI / GitHub Release 发布流程）。
不要把这些当作"已经能用"去写示例。

分页 KV cache 与投机解码已有组件级交付（5.241），但**没有接进语言模型那条路**：
主干是 Mamba，没有随上下文增长的 KV；两条能力各自有独立入口与逐位判据，
写示例时别声称"`generate` 已经用上了分页缓存 / 投机加速"。

张量并行只有**前向**的切分与对齐（5.240）：列 / 行并行、attention 头切分、
段间归约。反向通信、优化器分片与"按需拼回参数"（FSDP 式）都还没接，写训练
示例时不要把 TP 与数据并行混着讲。

量化已有 weight-only int8（5.239）：`quantize_i8` / `dequantize_i8` / `linear_i8`。
权重驻留 1 字节，**GEMM 仍走 f32 反量化**；没有 int8 累加、没有 fp8、没有 QAT、
没有非对称零点。写示例时不要暗示"int8 算得更快"。

## 验证命令

```powershell
# 本地全量
py -3.14 -m unittest discover -s tests -t .

# WSL HIP 全量
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/g/Cloumi/aetherflow && HSA_ENABLE_DXG_DETECTION=1 /opt/rocm-env/bin/python -m unittest discover -s tests -t ."
```

当前基线：本地 2675 个用例 / 416 跳过，WSL HIP 2675 个用例 / 373 跳过。
（两者各有一条与功能无关的失败——`test_conformance.TestReferenceConformance.test_covers_all_operators`，
是工作区里另一份未提交的 `cumsum` / `tril` 工作导致的，别去"修"它。）
本地必须用 `py -3.14`：3.10 环境里的 torch 是 2.4.1，会被兼容矩阵拦下 51 条
前端用例（不是代码问题）。

## CUDA 真机

本机显卡已拆、NVIDIA 驱动未加载（`nvidia-smi` 无法通信、`lsmod` 里没有 nvidia 模块）。
现在改用远程多卡机（用户提供，见下），CUDA 侧结论在那里复跑。

- 主机：`223.109.239.36`（移动）/ `180.127.11.177`（电信），端口 `24636`，用户 `root`
  （另有 `vipuser` 同密码；密码见对话记录，不写进版本库）。
- 系统：Ubuntu 22.04.3，16 核 / 31 GB / 196 GB，2× Tesla V100-PCIE-16GB（sm_70），
  驱动 550.127.05，CUDA 12.4 toolkit（含 libnvrtc / libcudart）；NCCL 2.32.6 见下。
- 环境：系统 python3.10 + numpy 2.2.6（apt 的 python3-numpy 只有 1.21，不满足
  `numpy>=1.24`，要用 pip 覆盖安装）；`cupy-cuda12x 14.2.0` 由官方 PyPI 装好
  （日志 `/root/pip_cupy2.log`，`getDeviceCount()==2`、sm 7.0）；无 Docker、无 cuDNN。
- NCCL 2.32.6：`pip install nvidia-nccl-cu12`（阿里镜像）装到
  `/usr/local/lib/python3.10/dist-packages/nvidia/nccl`，`ncclGetVersion` 返回 23206；
  `aetherflow/train/nccl.py` 先查 `find_library`，再按这个 pip 路径找 `libnccl.so.2`。
- 代码同步：`tar czf - --exclude=__pycache__ --exclude=.git . | ssh ... 'tar xzf - -C /root/aetherflow'`，
  远端目录 `/root/aetherflow`。
- 已验证：远端全量 `Ran 2331 tests / OK (skipped=120)` 全绿；CUDA 组
  `Ran 332 tests / OK (skipped=1)`，唯一跳过来自系统没有 cuDNN，另有 1 条"库缺失"的
  离线用例在有 NCCL 的机器上按设计跳过；NCCL / ZeRO 专项
  `python3 -m unittest tests.test_nccl tests.test_zero -v` → `Ran 36 tests / OK (skipped=1)`。
  收集到的用例比本地少 5 条（与 torch 版本相关）。
- 真机踩过的坑（已修，结论见 DESIGN 5.227）：`cusparseIndexType_t` 的
  `CUSPARSE_INDEX_32I` 是 **2** 不是 1——写成 1 会被当成 uint16 读 int32 的
  `rowOffsets`，表现为内核 `cudaErrorIllegalAddress` 而不是参数错误；一旦发生，
  上下文中毒会连带后续用例报 E。`cusparseSpMM` 是 11 个形参，别多传 C 描述符。
- 机器相关用例要按卡现算：CSR 铺法的阈值是 `SM 数 × 32`（M40 24 SM / V100 80 SM），
  行数写死会在另一张卡上走错分支。
- NCCL 的坑（结论见 DESIGN 5.228 / 5.229）：`ncclAllGather` 是 **6 个形参**（没有
  redOp），按 AllReduce 的 7 形参绑会在真机首个调用抛 `TypeError`；`ncclUniqueId`
  是 128 字节不透明结构，`bytes(c_char*128)` 会按第一个 NUL 截断成 `b''`，必须用
  `ctypes.string_at` / `ctypes.memmove`；优化器槽位是懒分配的，未 `step` 前
  `owned_bytes()==0`，比较分片大小之前要先真的走一步。


## weight-only int8（5.239）

- 三个算子：`quantize_i8`（f32 + scale → i8）、`dequantize_i8`（i8 + scale → f32）、
  `linear_i8`（x + i8 权重 + scale [+ bias] → f32）。语义只有一份，在
  `ir/ops.py` 的 reference 里；`models/quantize.py` 直接调用它，不另写舍入。
- 权重布局跟项目一致：`(K, N)`，不是 PyTorch 的 `(N, K)`。scale `(N,)` 就是
  逐输出通道；`(1, 1)` 是逐张量。
- 端点 `[-127, 127]`（不用 -128），舍入是 `rint` 的 **RNE**。要改这两条，
  必须同时改 `_QUANT_MIN` / `_QUANT_MAX`、`ops.py` 的 reference 和
  `device_backend.py` 里那两段 C 源码——三处不一致就会在 conformance 上炸。
- HIP 侧的 `cp` 是自家迷你运行时，**没有** `rint` / `clip` / `min`。所以量化
  不是 cupy 原语，而是 `_QUANTIZE_I8_SRC` / `_DEQUANTIZE_I8_SRC` 两段源码
  同时喂 NVRTC 与 hipRTC（`self._kernel(..., source, "af_...")`）。
- i8 是**数据输入**，不是下标：`_INDEX_ARGS` 必须声明
  `dequantize_i8: {0}`、`linear_i8: {1}`，否则 invoke 的"非浮点输入"检查会在
  派发前就以"不支持 int8"拦下（第一次跑 HIP conformance 就是这么红的）。
- `_quant_scale_arg` 的 `per_row` 快路径按 `s[i % cols]` 取 scale，避免把逐通道
  scale 物化成与权重同大的 f32（那会把"省 3/4"变成"多 1 倍临时量"）。
- 非正 scale 的检查走 `to_host`（scale 只有 N 个元素）；量化是离线步骤，
  别把它放进每步推理的热路径。

## 张量并行（5.240）

- 入口：`train/tensor_parallel.py` 的 `column_parallel_linear` / `row_parallel_linear` /
  `build_tensor_parallel_attention` / `TensorParallelBlock`。
- **行并行返回的是部分和**，必须由调用方 `all_reduce`（求和）之后再用。当成完整输出
  不会报错，只会得到 `1/W` 的结果。
- **行并行不接受 bias**：每个 rank 各加一次，归约后放大 `W` 倍。构图期直接拒绝。
- 头切分 = 列切分（q/k/v 的列按 `head * head_dim + i` 排列，`wo` 的行同理），
  前提是 `num_heads % world_size == 0`；`d_ff % world_size == 0` 同理。
- block 的 `dropout` 必须为 0：各 rank 头宽不同，同一 seed 采不到一致掩码。
- block 拆成 `attn` / `ff` 两张图，段间归约在主机侧，图外只剩两次残差加法。
  IR 里**没有** collective 节点，不要试图在 Graph 里找 `all_reduce`。
- `split_axis` 返回副本；`split_dense` 的复制类参数（`ln*_g` / `ln*_b` / `scale`）
  也必须拷贝，否则优化器原地更新会跨 rank 污染。
- 只做前向：反向通信与优化器分片没接，训练侧仍是数据并行 / 流水 / ZeRO。

## 分页 KV cache 与投机解码（5.241）

- 入口：`runtime/paged_cache.py` 的 `PagedKVCache`、`models/cached_attention.py` 的
  `build_cached_decoder_step` / `cached_forward`、`models/speculative.py` 的
  `SpeculativeDecoder` / `greedy_draft`。
- **`append` 是"全部层 + 全有或全无"**：`keys` / `values` 都是长度 `num_layers` 的序列，
  各层追加同样多的位置；页不够就整体抛 `PagedCacheFullError`，不改任何状态。
  不要试图"先写一部分再说"——长度与内容会不一致，而显存检查看不出来。
- 所有层共享**一张**块表（各层页号逐项相等）；`gather` 返回的是**副本**，
  写回要靠 `append`，改返回值不会改到缓存。
- `truncate` 只能缩短，按页释放、页内不清零；`length(rid)` 才是唯一真相。
- 增量图里输入与输出**不能同名**，所以缓存输入是 `l{i}_k`、本步新算的是 `l{i}_k_out`；
  `final_g` / `final_b` 也必须显式声明成输入，否则构图时 `KeyError`。
- 掩码是加性 `-1e9`，补零位与未来位都靠它；别把 `pad_to` 的补零当成"合法历史"。
- 投机解码被拒时**必须回到验证前的状态快照**：Mamba 的状态是递归的，不能按 token
  截断。`accepted < k` 时重喂"接受的前缀 + bonus"，不要相信"被拒的那步没写进状态"。
- 草稿可以返回空（下一步就是 EOS），这是合法弃权，会回退到目标的贪心 token；
  不要把它当错误抛出。
- 只做贪心验收与 batch=1；`temperature>0` 的采样验收没做。
  `greedy_draft` 是正确性参照，不是性能路径。
- 没有接进 `generate` / `LMRunner`：那条路径是 Mamba，没有增长 KV 可管。

## 成本模型与组合搜索（5.242）

- 入口：`tuning/cost.py` 的 `MachineModel` / `conv_tile_cost` / `elementwise_cost` /
  `rank_configs` / `cost_ranked_tune`，以及从 `tuning/search.py` 抽出的
  `measure_configs`（收"已展开的配置序列"，与 `tune` 共用排名与拒绝语义）。
- **它是排序依据，不是性能预测器**。单条估算与实测差数倍是正常的；它只负责
  回答"哪些组合不必测"和"为什么按这个顺序测"。不要拿 `total_us` 写性能结论。
- 派生量（`blocks` / `shared_bytes` / `threads` / `stages`）必须取自
  `lower/plan.py`，不要在这里重推一遍：两处推导迟早漂移，而漂移只会在真机上
  以"计划层拒绝、成本模型却认为可行"的形式暴露。
- 六项成本固定顺序 `arith / read / staging / sync / tail / dead`。`tail` 与 `dead`
  是**占空比**（按比例折算），不是绝对延迟；`sync` 按**波次**折算而不是按块数——
  按块数算会把小图算成同步主导（1×64×56×56 上曾算出 896us，改正后 16us）。
- `CostBreakdown.total_us` 对不可行项返回 `inf`，排序键里**必须**保留
  `not feasible`：只靠 `inf` 时，一旦有人把返回值改成 0 就会静默把不可行项排到最前。
- `cost_ranked_tune` 会把**默认配置提到有序列表最前**再裁剪：`TuningResult.speedup`
  以 `candidates[0]` 为基准，默认配置被裁掉时加速比会整体漂移。
- 裁剪后的候选集不是笛卡尔积，塞不回 `TuningSpace`——这是 `measure_configs`
  存在的唯一理由。平局语义一致：`configs[0]` 是默认，`tie_epsilon` 以内不替换。
- 只覆盖 conv2d 的 `tile_w/roc/cb` 与逐元素 `block`；归约类（softmax / reduce_sum /
  layernorm）还没有成本函数，`cost_ranked_tune` 对它们无从排序。

## Chrome Trace 导出（5.243）

- 入口：`runtime/trace.py` 的 `trace_events` / `build_trace` / `write_trace`，
  以及 `python -m aetherflow.runtime.trace telemetry.json [-o trace.json]`。
  输入是 `TelemetryRecorder.save` 落的那份 JSON（走 `load_records`）。
- 它不是采样 profiler：没有任何额外开销、也不新增观测，只是把已记录的东西
  摆到时间轴上。因此看到的时长与 `runtime/telemetry.py` 里的数完全一致，
  看不到的东西（内存、SM 占用、调用栈）也不会凭空出现。
- `start_ms` 的零点是**本次 `execute` 的起点**，与 `wall_ms` 同一个 `perf_counter`。
  两者对不齐时时间轴会出现负间隔或凭空长出的空白。旧记录没有这个字段，
  导出时按"接上一个节点的结束点"补齐，并在 `otherData.synthesized_nodes` 计数。
- 空白是**真实的**：CPU 轨道上的缝隙就是那段时间没在派发（等设备、等 Python）。
  不要为了"好看"把间隔抹平。
- GPU 轨道是**累计占位**，不是真实时间轴：逐节点的 `gpu_ms` 来自事件对，
  只有耗时没有绝对时刻。线程名写死"GPU 内核（累计占位）"、事件带
  `args.approximate: true`，就是为了防止有人从重叠关系里读出并行度。
- `ts` / `dur` 的单位是**微秒**。Chrome Trace 只认微秒；写毫秒会得到一条
  被压缩 1000 倍的图，且不会报错。
- 一份 `RunRecord` 一个 `pid`，多条记录按 `wall_ms` 依次错开；这是"连跑多次"
  能在同一张图里看的前提。
- 打包与 CI：CI 的 `build` 作业跑 `python -m build`，再把 wheel 装到临时目录
  做一次 `import aetherflow` 冒烟，最后 `upload-artifact`。本地自查用
  `python -m pip wheel . --no-deps`；`pyproject.toml` 的 `packages.find`
  带了 `include = ["aetherflow*"]`，新增子包不需要改配置。

## NCCL / RCCL 绑定（5.244）

- **只有一份 ctypes 声明**。RCCL 实现的函数名、枚举值、`ncclUniqueId` 的 128
  字节布局与 NCCL 相同，所以 `train/nccl.py` 直接加载 `librccl` 也能用；不要
  为 HIP 再抄一份签名表，两份迟早漂移。
- **soname 不同**：NCCL 是 `libnccl.so.2`，RCCL 是 `librccl.so.1`
  （WSL 实测 `/opt/rocm/lib/librccl.so.1`，`ncclGetVersion()` 返回 22707 =
  2.27.7）。候选表两种都列，`find_library("nccl")` 与 `find_library("rccl")`
  的结果放最前且保持 NCCL 优先。
- **厂商按库文件名判定**（`_vendor_from_path`），不看能不能 import cupy：
  设备上装了什么库与 Python 侧装了哪个包是两件独立的事。`vendor()` 在库缺失时
  抛 `NcclUnavailable`，不返回 `"unknown"`。
- **WSL 前置**：`HSA_ENABLE_DXG_DETECTION=1`。没有它时 `ncclGetUniqueId`
  不报错码，而是直接打印 `HIP failure: 'no ROCm-capable device is detected'`
  并 FATAL；这正是"看起来是 RCCL 的问题、其实是环境没开"的那类现象。
- **HIP 的数组不是 cupy**：`backends.hip.runtime.DeviceArray` 提供
  `data.ptr` / `size` / `dtype` / `flags`，刚好是绑定层要求的四项，
  `tests/_rccl_worker.py` 用它跑四个集合操作。
- **`synchronize()` 必须按厂商分叉**：HIP 走 `backends.hip.runtime.synchronize()`
  （设备级），CUDA 走 `cp.cuda.get_current_stream().synchronize()`。两边都试会
  把"同步失败"变成"同步被静默跳过"——负向对照里把 HIP 分支改成 cupy 分支，
  worker 立刻以 `ModuleNotFoundError` 退出。
- **真机只验到 world=1**（WSL 单卡 RX 7650 GRE）：加载、版本、unique id、
  comm 初始化与四个调用面都跑通了，但跨卡数值仍只有 CUDA 的 2×V100 结果。
  写结论时不要说"RCCL 多卡已验证"。
- 能力声明（`collective_support`）仍是 `device_allreduce=False`：数据并行路径
  没接线，跨卡归约照旧走主机中转；理由文本会区分"本机加载不到库"与"库在、
  只是没接线"。
- **判据里不要写死厂商名**：`tests/test_parallel.py` 里两条能力断言原本写
  `assertIn("NCCL", reason)`，在 HIP 机器上必红（理由是 `RCCL 2.27.7 …`）。
  改成了 `assertRegex(reason, "NCCL|RCCL")`。规律：凡是"理由文本由运行环境
  决定"的断言，都要接受本机能出现的全部形态，否则本地绿、真机红。

## 设备端循环（5.245）

- `runtime/device_loop.py` 的 `DeviceWhileLoop` **不是加速开关**。实测 64 轮：
  4 KB 状态 host 8.6 ms / device 37 ms，4 MB 状态 host 26 ms / device 44～46 ms
  （WSL HIP）。原因是在这张卡上捕获图 replay 的流同步比逐节点派发 + 默认流同步更贵。
  交付的是"状态不再往返"的执行形态；写进性能结论前先自己测一遍。
- **条件与体必须用同一条流捕获**（`backend.new_stream()` 一次，两次
  `capture(..., stream=self._stream)`）。体的 replay 是异步的，下一轮条件的
  replay 只有排在同一条流上，"体已完成"才有前提。负向对照里把体换成另一条流，
  `test_both_graphs_are_captured_on_the_same_stream` 立刻红。
- **每轮只有一次同步、一次 D2H**：读谓词标量。这是数据相关分支绕不过去的，
  藏起来只会让"零同步"变成假话。判据写的是 `syncs == iterations + 1`、
  `to_host_calls == iterations + 2`；体一旦改回 `run_device`（带同步）就多 n 次。
  替身要注意：`_NumpyPlan.run_device` 必须给 `plan_syncs` 记账，否则"体也同步了"
  这件事在第一版判据里根本看不出来。
- **状态复位是身份复位，不是重新上传**：`_reset` 把 `state` 指回建计划时的初值
  缓冲（`self._initial`），`to_array` 不调用；只有调用方显式给自由输入才上传并写回
  常驻缓冲。HIP 的 `DeviceArray.__setitem__` 不接受 numpy 主机数组，只有设备数组
  或标量——所以不能"把初值再写进状态缓冲"。
- **重复 run 的判据要比数值更狠**：只比终值抓不到"状态没复位"——从终值再跑会
  立刻退出、结果仍是 5。必须同时断言两次 `plan.iterations`，否则负向对照不红。
- 替身要区分"假设备数组"与"主机 numpy 数组"（测试里的 `_Dev` 包装类带 `__array__`）。
  两者混用会让 `to_array` 计数失真，"零重上传"那条判据就失去判别力。
- 边界：只支持条件与体都是单张 `Graph` 的形态，嵌套区域显式抛 `DeviceLoopError`
  （提示继续用 `WhileLoop.run`）；`max_steps` 用尽的 `ControlFlowError` 与宿主路径
  同形状；一张捕获图只对应一组固定形状，换形状换 dtype 直接 `ValueError`；
  未接进训练 / 推理主路径，CUDA 真机也未验证。

## 提交约定

- 仓库在 `g:\Cloumi\aetherflow`（工作区外，Shell 命令需要 `dangerouslyDisableSandbox: true`）。
- PowerShell 5.1 不支持 `&&` 与 heredoc：用 `;` 串联、`git commit` 用多个 `-m`。
- 提交身份：`git -c user.name=aetherflow -c user.email=dev@aetherflow.local commit`。
- 代码与 DESIGN 分两次提交。

## 已知边界（详见 DESIGN §7）

- 分片只省主机内存；流水只有"顺序"层面的重叠，单进程串行切卡没有真并发。
- `fft` 没有 VJP（IR 缺 `conj`）；HIP 没有 rocFFT，复数只在容器层面可承载。
- `spike_linear` 的 `dspikes` 必须稠密：替代梯度在未发放位置同样非零。
- `topk_mask`（5.218 MoE 路由用）只有参考实现，CUDA / HIP 没有对应内核。
- MoE 采用稠密参考语义：所有 token 过全部专家再按 top-k 门控加权，只支持单设备。
- 注意力反向按 query 行分块（5.219，块长 128）；`dropout > 0` 与符号序列维退回整块。
- `ServingEngine` 的超时只在 `step` 边界生效（协作式），单步长内核打断不了；超时记录走
  `take_timed_out()`，绝不混进完成列表。
- 集合通信（5.221）只做了能力声明与设备侧 provider 缝：没有接入真 NCCL / RCCL，
  `overlap` 恒为 `False`（计算/通信重叠未实现），未注入 provider 时显式报
  `DeviceCollectiveUnavailable`。5.228 起接了真 NCCL（`train/nccl.py`：AllReduce /
  AllGather / ReduceScatter / Broadcast，f16/f32/f64 + sum/prod/max/min），但只在
  单机多进程下验证；bf16 与 `avg` 归约显式拒绝，`overlap` 仍恒为 `False`。
- 跨进程 ZeRO（5.222）只把优化器槽位按 rank 切分落地（npz + `__meta__`），不省显存、
  不做 all-gather；`load_shard` 靠片号 / 分片数 / 归属表三道校验兜底（优化器自身不查）。
  5.229 起有跨进程 ZeRO-1（`train/zero.py`）：每进程仍持完整参数、只分片优化器槽位，
  通信量与 DDP 相同（整份 AllReduce）；没有 ZeRO-2/3 的参数分片与 all-gather。
- torch 兼容矩阵（5.223）是声明式区间（2.9–2.13，六个 dtype：四个浮点 + 两个整数）；
  区间内非 verified 版本
  不保证回归覆盖，越界或 dtype 缺席显式抛 `UnsupportedTorchVersion` /
  `UnsupportedOperatorError`。
- torch 入口（5.224）的整数只有 i32：int64 收窄、越界值在执行器入口拒绝；`padding_idx`
  / `sparse` / `scale_grad_by_freq` 仍未实现（显式拒绝）；提升常量与 `get_attr` 常量落成
  `constant`，单常量上限 65536 个元素，i32 常量在导入时做值域检查（常量不经过执行器
  入口），`_to_dtype` 用兼容矩阵的行做 dtype 校验；`bool` 常量仍无 IR 语义；
  `torch.arange` 之类仍是未支持算子。
- 续训的数据迭代位置（5.225）用专属标量键 `loader_state`（JSON）+ `loader_kind`，
  冲突检查只看这两个键；位置只到"第几轮第几批"，batch 内偏移不在协议内；流式 loader 的
  `size` 是构造时快照。游标在所有提交完成后才推进（5.226）。
- cuDNN（5.230）：远端 pip 装的是 `nvidia-cudnn-cu12` 9.28.1.4（`cudnnGetVersion`
  报 92801）。它要求 Linux 驱动 ≥570.26，这台机器是 550.127.05，于是句柄 / 描述符 /
  v7 算法选择全返回 0，**执行期**才报 `cudnnStatus_t=5003`
  （`CUDNN_STATUS_EXECUTION_FAILED_CUDART`）。结论：`available()` 只代表"库能加载"，
  不代表"这台机器跑得动"；`conv_impl="auto"` 会回落自研核（与 im2col 最大差 4.8e-7），
  显式 `conv_impl="cudnn"` 抛出带状态码的错误。`TestCudaConvCudnn` 因此在 `setUpClass`
  里加了一次最小卷积探测，命中 5003 就整组跳过，避免同一个环境问题把一组用例打成红。
- cuDNN 9.1.0.70（仍支持 sm_70、只要求驱动 ≥525）的 wheel 试过四种源：官方
  `files.pythonhosted.org` 单线程约 0.3 MB/s，8 段并发会被 429 限流；阿里 / 清华 /
  USTC 镜像要么 429、要么 TLS 握手超时。最终只下到 121/159 段（每段 4 MB）就停了，
  **卷积真机对拍与 `benchmarks/cudnn_compare.py` 的逐形状数据仍是空白**。下次别再用
  8 并发抓同一个 URL（429 会越退越久），也别把 `pip download` 与分段下载同时跑。
- **远端那台 2×V100 已由用户释放**（2026-11-15）。下次做 CUDA 真机验证要重新开机器、
  重装 NCCL 与 cuDNN，`DESIGN` §0 里的真机数字在此之前只能沿用上一轮（2335 / 120）。
- ZeRO-2（5.231，`train/zero2.py`）：梯度也只留本片。关键是 NCCL 的
  `ncclReduceScatter` 要求各段**等长**，而按参数名分片时每片大小不一 —— 因此按参数
  边界分桶后把每段补零到 `pad = max(段长)`，通信缓冲固定 `pad × world_size`。
  每步两步：`reduce_scatter` 拿回本片梯度 → 本地优化器更新 → `all_gather` 拼回完整
  参数并原地写回。省的是常驻内存（梯度峰值 `N` → `N/W + pad`），通信量与 ZeRO-1 同
  量级（每步收发各 `pad × W`）。数值不逐位保证（NCCL 归约顺序由拓扑决定）。
  `LoopbackWorld` 与 `NcclHostCollective` 各补了 `reduce_scatter` / `all_gather` 两个面。
- 存档：ZeRO-1 与 ZeRO-2 共用 `train/zero.py` 的模块级 `_save_shard` / `_load_shard`；
  `Zero2Rank.steps` 是 `optimizer.step_count` 的只读视图，**不要再单独记一份计数**
  （两份真相会让 `load_shard` 之后步数对不上）。
- 多 rank 启动器（5.232，`train/distributed.py`）：环境变量 `AF_RANK` /
  `AF_WORLD_SIZE` / `AF_LOCAL_RANK` / `AF_MASTER_ADDR` / `AF_MASTER_PORT` 优先，
  torchrun 的 `RANK` / `WORLD_SIZE` / `LOCAL_RANK` 兜底，都没有且无默认值就报错
  （不猜 0）。`ncclUniqueId` 的 128 字节只有 rank0 能生成，跨进程用 TCP 分发：
  `UniqueIdServer` 每个 follower 连上来先发 1 字节 rank，服务端回 128 字节；
  `fetch_unique_id` 从 rank ≥ 1 调用。
- `spawn` 用 **`subprocess` + `python -c`** 而不是 `multiprocessing`：spawn 会重新
  import `__main__`，在 `python -m unittest` 下等于递归重跑整份测试。入口写成
  `"包.模块:函数"`，子进程里 `importlib.import_module` 后把 `sys.argv[1:]` 原样传进去。
  注意把 `server.wait()` 放在**正常路径**里：放进 `finally` 会用它自己的异常盖掉
  "子进程退出码非 0"那条更有诊断价值的错误。
- 跨平台差异（HIP 全量第一次跑才暴露）：服务端 `close()` 之后，客户端 `connect`
  在 Linux 上**仍会成功**（backlog 里还排着 SYN），紧接着 `recv` 抛
  `ConnectionResetError`；Windows 上同一场景是静默等到超时。因此
  `fetch_unique_id` 把"连上之后的收发"包进 `except OSError` 转成 `RuntimeError`。
  教训：只在一个平台上跑的用例不算验证，涉及 socket 的用例必须过 WSL。
- ZeRO-3（5.233，`train/zero3.py`）：参数本身也只常驻一块。每个参数按**元素**均分
  成 `W` 块，`chunk = ceil(n/W)`，rank `r` 只留第 `r` 块（一维、尾部补零）；
  `materialize(names)` 每参数一次 AllGather 拼回完整形状进缓存，`release(names)`
  丢副本，`step(grads)` 每参数一次 ReduceScatter 取本块梯度、只更本块，step 之后
  必须 `_cache.clear()`（否则缓存与分片两份真相）。
  **为什么不拍平成一条大缓冲**：参数边界与块边界错位会把参数劈成两半、逐参数超参
  失效；代价是消息数随参数个数线性增长（与 ZeRO-1 的广播同量级）。
  前提是各 rank 的模型定义顺序一致（按 `param_names` 的块号对齐）；存档必须同时带
  参数分片与优化器槽位，`world_size` 变了要重新分片；`materialize` 返回的是**副本**，
  改它不会回头改分片。数值上拼回的完整参数与整份逐位相同，只有 `step` 的求和顺序
  由拓扑决定。
- ZeRO-3 接进训练循环（5.234，`train/loop.py` + `train/recompute.py`）：`CheckpointPlan`
  的 `forward` / `backward` 各多一个 `on_segment` 回调，每一段执行前**先**借出该段
  `segment.inputs` 里出现的参数（`zero3.materialize(names)` 塞进 feed），执行完
  `feed.pop` + `zero3.release(names)`。借用必须成对：只删引用不 release，缓存会一直
  钉住这块显存；只 release 不删引用，下一段的 `_feed_of` 会捡到上一段的旧副本。
  三条要点：①`store` 从"参数真身"降级为**借出的完整副本**，训练期间是旧值，
  `sync_to_host()` 才拼回、`snapshot()` 发现 `_host_stale` 会先自动同步；②更新走
  `zero3.step(grads)`，所以 `optimizer` 必须就是 `zero3.optimizer`；③零散的
  `device_state` / `master_weights` / `GradScaler` 全部构造期拒绝——它们各自还要一份
  完整参数或完整梯度，与"只常驻一块"直接冲突。强制要求 `segments`：整图前向本来就
  要把所有参数拼回来，不切段就只是"看起来在分片"。`Trainer.steps` 是进程内计数，
  `load_shard` 只带回 `Zero3Rank.steps`，续训用例断言别写错对象。
- 分布式存档的合并与重分片（5.235，`train/shard_io.py`）：`save_shard` 只写本片，
  合并/换 `world_size` 都是**纯文件操作**，不走通信——换卡数靠"先拼回完整状态、
  再按新布局切"，而不是让两个不同规模的 world 对话。布局由 meta 判定：有 `plan`
  是 ZeRO-3（带参数分片），有 `assignments` 是 ZeRO-1/2（**只带槽位、参数在模型
  里**）。因此 names 布局合并出权重必须 `params=`，save() 缺参数直接报错，
  不拿零填充。`elements` 合并后槽位要 reshape 回参数形状，否则
  `Optimizer.load_state_dict` 按"形状不符"拒绝；切分时再 ravel，等价往返。
  换 W：names 用 `shard_params`（需要形状），elements 用 `plan_chunks`；同 W 且
  没给参数时可以沿用存档里的归属表。合并 `params=` 在 elements 布局下是 oracle：
  逐位不等就报"分片来自另一份权重"。
- 主机中转的多进程数据并行（5.236，`train/host_collective.py` + `train/parallel.py`）：
  `HostRelay`（父进程 TCP 服务）+ `HostCollective`（子进程客户端）+ `launch`
  一条命令拉起 W 个进程；`DistributedDataParallel` 挂在 `trainer.grad_hook` 上
  做梯度 AllReduce，`step` 按 rank 切第 0 维并返回全局平均损失。帧格式
  `struct("!QQ8s")` = `(seq, count, dtype)` + 裸字节，按 seq 归组、校验
  count/dtype、按 rank 顺序求和；**只做 sum**，平均由调用方做。
  踩过的坑：①`code.ljust(8, b"\x00")` 里 `code` 是 str，会抛 TypeError，
  必须 `code.encode("ascii").ljust(8, b"\x00")`——这是真多进程用例失败的根因，
  且表面上只表现为"两个 relay 不一致用例断言 `relay.error` 为 None 却失败"；
  ②relay 测试的客户端必须在 `finally` 里 `close()`，否则 ResourceWarning；
  ③`payloads.index(payload)` 对 dict 是相等查找（会命中第一个），要
  `enumerate` 拿下标。切片的形状靠新增的 `Trainer.data_input_shapes()`（非参数、
  非 `rng_inputs`），每卡分片第 0 维必须等于图声明值。构造期拒绝 rng 不一致、
  `device_state`、`scaler`、`zero3`——都是"看起来能跑其实语义不对"。
- 静态执行快路径（5.237，`runtime/executor.py`）：`_GroupPlan.fast` 记录静态单节点组
  的 `(内核, 节点, 输入 id, 属性, 期望形状, 输出 id, dtype, 是否量化)`，`Executor._fast_step`
  在第一次执行时填。**三态语义**：`None` = 还没试过、`()`（`_NO_FAST`）= 已确认不能直连、
  非空元组 = 直连记录——用空元组而不是再加一个布尔字段。踩过的坑：①测骨架必须让 no-op
  执行器绑定 `kernels`（补一个 `_prepare_plan` 把 `group.kernels` 设成 `self.invoke`），
  否则 `_fast_step` 一律返回 `None`，读数看起来"快路径完全没效果"（69.3 vs 71.4）；
  ②`group.fast[0]` 不能用 `is` 和 `executor.invoke` 比较——绑定方法每次取都是新对象，
  要 `assertEqual`；③负向对照要分别改坏 `if out.shape != shape` 与 `if out.dtype != dtype`
  两处，各只变红 1 条，说明两道路径都在被执行。
- 卷积实现选择（5.238，`lower/plan.py` + `backends/device_backend.py` +
  `backends/hip/backend.py`）：**同一问题跨厂商结论可以相反**——M40 上 5.151
  的 A/B 是隐式 GEMM 在 K ≥ 288 输给 im2col 1.37～2.32×，HIP（RX 7650 GRE /
  ROCm 7.2）上中等通道 3x3/5x5 反过来赢 1.33～1.63×。所以加**厂商开关**
  `CONV_AUTO_IMPLICIT`（基类 False、`HipBackend` True），而不是改 `auto` 的
  全局默认，CUDA 行为逐字不变。两条限制都来自负向实测：`kh*kw > 1`（1x1 C=256
  上隐式慢 3%）、`k > CONV_TILED_MAX_K`（C=3 的 3x3 持平，tiled 仍优）。
  `conv_implicit_fits` 必须与 `plan_conv_implicit` **共用**同一份 shared 算式
  （权重 tile + 输入窗口都算进 48 KB），改一处必须同步另一处，否则会出现
  "选中了但规划抛 ValueError"。踩过的坑：①测量不能顺序单轮做，auto 与
  im2col 的差距会被 10～17% 的顺序噪声盖住，要 5 次预热 + 3 轮交错、每轮
  10 次中位再取最小；②负向对照的测试形状要能**区分**改坏的那一项，`kh=kw=40`
  只看权重也给同一个 tile，改掉窗口项后不变红，换成
  `conv_implicit_oc_tile(32, 48, 32, 79)`（只看权重 8、计入窗口 4）才行。
  临时脚本 `_tmp_conv.py` 用完已删。
- **工作区被旧缓冲覆盖（反复出现）**：提交后用 `git status` / `git diff --stat` 核对，
  出现过 `train/__init__.py` 退回旧版（丢 `Zero2Rank` / 启动器导出）、`DESIGN.md` 丢
  已提交小节的情况。发现 `git diff` 只是"删掉已提交内容"时，用
  `git checkout HEAD -- <file>` 逐个恢复，再在其上继续编辑，不要靠重写整份文件。

- **NHWC 传播的两个易错点（5.246）**：第一，逐元素算子**不接受** `layout` 属性
  （`check_attr_names` 会拒绝），传播时只给 conv / pool 设；第二，卷积的
  weight / bias 不是激活，只有第 0 个输入进 NHWC 域——把权重也 `to_nhwc` 会让
  6 条判据同时变红，而节点序列看起来**仍然是对的**，只有形状与数值对拍才抓得到。
- **设备端拒绝布局要拦两处**：`DeviceBackend.invoke` 与 `_planned_kernel`
  （图捕获 / 批量派发快路径）。只拦前者时快路径会拿 NCHW 内核去读 NHWC 缓冲区，
  报出来的是"输入通道 8 与权重通道 3 * groups=1 不一致"这类误导性错误；
  WSL HIP 上的负向对照已确认。
- **布局传播不进默认管线**：`LayoutPropagation` 依赖目标后端的声明（带参数），
  按约定不进 pass 注册表。默认 `supported_layouts` 是参考后端口径
  （`{"nchw","nhwc"}`），设备后端必须显式传 `DeviceBackend.supported_layouts`
  （只有 `nchw`，此时有可翻区域就抛 `LayoutError`，不静默跳过）。
- **NHWC 没有反向规则**：三个算子的 VJP 开头显式抛 `NoGradientError`，不要指望
  反向算子自己发现——它们会按 NCHW 的轴归约，产出"看着有值但全错"的梯度。
  布局传播按推理期优化设计，训练图保持 NCHW。

- **L2 描述的"本地全绿"不等于渲染正确（5.247）**：`tril()` 的 `grid_stride(limit, *body)`
  是**可变参数**，原写法把三条语句包成一个 tuple 传进去，渲染器报"未知语句类型:
  tuple"。本地无 GPU 时这条路径从不执行，单测与 conformance（参考后端）都全绿，
  只有 WSL HIP 真机 conformance 才暴露。写新的 `Kernel` 描述后，至少要在有设备的
  一侧跑一次 conformance；只看本地结果会漏掉整类"渲染期才炸"的错误。
- **设备端新算子的三处必须同时改**：`GPU_OPS` 声明、`_register_kernels` 登记、
  handler 实现。顺序无关，但漏一处在**构造期**就失败（`_OPS` 与已注册集合对账），
  比第一次推理才崩好。`HIP_OPS = GPU_OPS`，HIP 不用改代码。
- **conformance 的覆盖度是硬要求**：`GPU_OPS` 里加了算子但 `default_cases()` 没有
  用例，`test_all_declared_operators_exercised` 与套件的 coverage 检查都会报
  `untested`；用例要选能覆盖折维/掩码边界的形状（`cumsum` 用非末轴、`tril` 用负
  `diagonal`），否则等于没测。

## 临时脚本

调试用的临时脚本（`_tmp_*.py` 等）用完即删，关键结论、问题与解法写进
DESIGN 的对应小节，凭证不落盘。
