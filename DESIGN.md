# AetherFlow 设计与现状

> 状态：M0（语义 IR + 参考执行 + 优化框架）、M1（torch.export 导入）、M2 第一版（CUDA 后端）、
> M3（设备驻留 / 首条融合 / 运行期计划 / 内存规划）、M4（IR 版本化、后端一致性套件、
> 第二硬件后端 HIP/ROCm）均已交付并验证。本地机器无 GPU；CUDA 部分在远程
> RTX 2080 Ti 与 Tesla M40 24GB（sm_52，5.127 的 f16 原子加问题在此暴露并修复）
> 上验证，HIP 部分在 WSL2 + RX 7650 GRE（gfx1100 / ROCm 7.2）上验证；文档中的
> CUDA / HIP 数字来自这三台机器，逐条注明出处。
> 本文档同时是原始规划、决策记录、实现现状与路线图的唯一入口。

---

## 0. 一页速览

| 项目 | 内容 |
|---|---|
| 定位 | 统一深度学习基础设施：一套前端 IR，可插拔硬件后端 |
| 独立的含义 | **实现独立**（前端 / IR / 运行时 / 插件契约自研），**入口兼容**（能读 PyTorch 导出的图），**生态复用**（CUDA 库照常调用） |
| 当前代码 | 60 个算子（IR 里新增 `fft` / `rfft` 两个复数算子与 MoE 路由用的 `topk_mask`；CUDA 承接 57 个自研内核 + cuFFT 的 fft/rfft，全部支持 f16——attention 走混合精度模板、存储 f16 而累加在 float；HIP 沿用同一批模板，承接面仍是 57 个，没有 rocFFT 绑定；`topk_mask` 目前只有参考实现）、9 层模块、2255 个测试（本轮实测：本地 2255 个用例 / 410 跳过、WSL HIP（RX 7650 GRE / ROCm 7.2）2255 个用例 / 369 跳过；CUDA 真机（3×Tesla M40 24GB）本轮**未复跑**——SSH 可达，但 NVIDIA 驱动未加载（`nvidia-smi` 无法与驱动通信、`lsmod` 里没有 nvidia 模块，只剩陈旧的 `/dev/nvidiactl`），上一次为 1991 个用例 / 105 跳过（该机收集到的用例数比前两处少 12，历来如此，与 torch 版本相关）；上一轮新增的 46 条全部是纯 Python 用例（5.204 的 11 条：事件梯度的解析式、有限差分、符号事件数与越界拒绝；5.205 的 35 条：ZeRO 分片与整份逐位对拍、流水两段/三段的单卡对拍与接口拒绝面），本轮新增的 4 条同样是纯 Python（5.206 的逐行约束：按行生效、长度与接口拒绝、未完成按行报错、会话复用与无状态入口一致），5.207 的 12 条也是纯 Python（反向图按目标剪枝 5 条、交换律 CSE 与反向管线 7 条），5.208 的 3 条同样是纯 Python（训练态 BN 与 eager 逐位、分解后同一条路、导入后梯度与 torch autograd 对拍），5.209 的 4 条同样不碰设备（生产内核的线性化由 `Layout` 生成、源码形态与 `cast` 关闭、五个布局常量），5.210 的 23 条同样是纯 Python / 离线（复数 dtype 与 `fft` / `rfft` 的参考语义、拒绝面、序列化，以及 cuFFT 的归一化表与 dtype 登记表——本机没有可用 NVIDIA 显卡，`cufftExec*` 未跑），5.211 的 13 条同样是纯 Python（`sequential`/`gpipe`/`1f1b` 三种流水调度的数值等价与驻留峰值计数、Reduce-Scatter 与 All-Reduce 的逐位一致与接收量、拒绝面），上一轮新增的 20 条同样是纯 Python（5.212 的审计修复：门控 SiLU 反向、AMP 顺序与调度、累积权重与续训事务、分片先校验后提交、硬件包信任记录、归一化标量参数形状、整数入口范围、fuzz 整数除法）；上一轮新增的 23 条同样是纯 Python（5.213 的 bf16 闭环：RNE 量化与两字节打包、NaN 位模式的截断与 quiet 补位、执行器在 bf16 类型边界的量化、autocast 接受 bf16、bf16 训练驻留与拒绝面）；上一轮新增的 14 条同样是纯 Python（5.214 的续训闭环：DataLoader 的轮次/批游标与配置校验、前向图指纹、调度器类型与超参凭证）；上一轮新增的 9 条同样是纯 Python（5.215 的 BN running stats 闭环：EMA 与手写公式/无偏方差对拍、参数不更新时统计量照进、训练统计量接推理图、存档往返与 momentum 拒绝面、设备态与符号通道拒绝）；上一轮新增的 5 条同样是纯 Python（5.216 的组合矩阵：五开关全开的手算对照、窗口内续训、分段重算对整图、f16 主副本、设备驻留与主机态逐位一致），上一轮新增的 13 条同样是纯 Python（5.217 的动态形状契约：契约表与查询、未知环节拒绝、训练循环与分段重算的入口守卫、执行器多形状绑定、部署只要求权重静态、reshape 表达式目标被拒绝、torch 动态维在导入期显式拒绝），本轮新增的 13 条同样是纯 Python（5.218 的 MoE：top-k 路由与门控归一化、负载均衡损失手算、中心差分梯度、训练损失下降、专家并行拒绝面、序列化往返，以及 `topk_mask` 的掩码语义与两条 conformance 形态），设备路径由 conformance 的跨后端用例覆盖；另有 7 个多卡用例（设备切换 5 + 数据并行 2）在 3 卡全可见时单独跑通。跳过数的差异来自各机器缺哪一组硬件） |
| 已闭环的能力 | torch.export → IR → 校验 → 参考执行 → 中层优化 → 融合 → CUDA 后端对齐 → 运行期计划（CUDA Graph）→ 捕获期内存复用 → IR 序列化与版本治理 → 后端一致性（conformance）套件 → 插件契约分层（准入面 + 可选契约）→ 算子注册与调度 → 卷积与池化端到端（IR 语义 + 分块直算 / im2col/cuBLAS + 自研 maxpool kernel）→ CNN 前端导入（LeNet 端到端三方对齐）→ 性能基线与回归判定 + CI 配置 → 自动微分（语义 IR 层反向展开，反向图可序列化、可执行）→ 训练闭环（参数容器 / SGD+Adam / 图内损失，收敛性有测试）→ 六类模型构件（MLP / CNN / RNN(LSTM/GRU) / 多层 Transformer / Mamba block / 多层 SNN）→ 随机算子（seed 化 dropout_mask，跨后端逐位可复现）→ 批归一化（训练态 BN + 独立 stats 算子，反向全部复用已有语义）→ 构件层接入 BN 与 dropout（开关在构图期，同一份代码构建训练图与推理图，5.26）→ 权重存档与优化器状态恢复（续训轨迹与未中断逐位一致，5.27）→ 可复现的 DataLoader → f16 图与梯度缩放（把种子当缩放因子，图不用改，5.28）→ bf16（RNE 量化、2 字节驻留、autocast 与训练闭环，5.213）→ 融合注意力（一个 IR 节点，反向由已有算子组合，5.29）→ 插件扩展面（命名空间护栏的自定义算子注册 + 多实现按代价选择 + 开发指南，5.30）→ 训练辅助（warmup / 余弦与阶梯调度、全局范数裁剪、梯度累积、EMA，5.31）→ CUDA 融合注意力前向（分块 online softmax，分数矩阵不物化，5.32）→ CUDA f16（一套模板 + dtype 占位符展开、hgemm / hgemm_batched、卷积走 im2col+hgemm，5.33）→ 分组 / depthwise 卷积（按组切片复用 im2col，分块核明确拒绝 groups>1）与自适应 / 全局池化（PyTorch 的 floor/ceil 窗口规则、重叠窗口原子累加，5.34）→ CUDA 承接 SSM 线性递归（前向与伴随，一个尾元素一个线程、沿时间串行；5.155 起长 T 自动走分块，短 T 仍串行）→ SNN 稀疏事件表示与事件内存池（CSR 变体、只读事件的发放率、跨前向复用下标缓冲，5.35）→ 常量折叠（`constant` 承载任意张量 + PORTABLE 子图求值，5.36）→ 深层融合（LayerNorm+残差、SwiGLU 都表达为已有算子的输入扩展，5.36）→ 编译期静态显存预算（liveness 直接从 IR 算出峰值与复用倍数，5.36）→ 自动调优（参数空间 + 网格搜索 + 按设备/形状命中的配置库，5.37）→ 运行时数据上报（逐节点派发耗时、整图 wall、显存快照，5.37）→ 只读智能体查询面（后端画像 / 性能摘要 / 配置推荐，不执行不改图，5.37）→ IR 差分模糊测试（随机图 + 五条不变量 + 非法属性变异，5.38）→ CI 的模糊测试作业与完整 GPU 作业（5.38）→ 跨运行吞吐 / 显存趋势（5.38）→ 真实模型导入覆盖（7 个 torchvision 模型端到端对拍，推理态 BN / 精确 GELU 由前端分解承接、ceil_mode 池化改直通 IR（5.154），5.39）→ HIP/ROCm 第二硬件后端（复用同一批内核模板，hipRTC 运行时编译 + hipBLAS + ctypes 容器，conformance 62 executed / 5 refused / 0 failed，与 CUDA 逐项一致，5.40）→ 后端装配层（内置三工厂 + pip entry point，按名 / 环境变量 / 自动探测挑选，退回参考后端必须可见，5.41）→ L2 内核 IR（语法级：25 个表达式 / 语句节点 + C 语法打印器 + 方言数据；逐元素二元 / 一元 / step / 广播 / 归约 / softmax / LayerNorm（含融合残差）/ RMSNorm 九个内核族已接管生产路径，与手写模板逐 token 相等，CUDA 真机 conformance 62 / 5 / 0，5.42）→ zip 硬件包（manifest 校验 + 安全解压 + 目录扫描 + 装完即用，24 个用例，5.43）→ 示例硬件包 novac（一份方言 + 三方法后端类，装上 zip 即可打印内核并跑图，6 个用例，5.44）→ 归一化内核迁到 L2（Widen / Narrow / Let(type_name=None) 三个节点由模板倒推，LayerNorm / 融合残差 / RMSNorm 共 7 个新用例，5.45）→ 门控 SiLU 与批归一化迁到 L2（零新节点，批归一化模板拆成两个内核，6 个新用例，5.46）→ 池化家族迁到 L2（`AtomicAdd` / `Continue` 两个语句节点、方言新增可选的 `atomic` 能力；maxpool2d / avgpool2d / 自适应池化及三者的反向共六个内核，3 个新用例，5.47）→ 拼接 / 随机 / 扫描 / 卷积梯度迁到 L2（`Store(raw=True)` 裸赋值与内核私有 `helpers` 两项能力；concat_copy / dropout_mask / scan_linear 及其反向 / 卷积输入与权重梯度共六个内核，7 个新用例，5.48）→ im2col / 分块卷积 / 融合注意力迁到 L2（`Not`、`Shared(extern=True)` 动态共享内存、`ArrayDecl(values=None)` 三项能力；9 份字符串模板至此全部迁完，L2 成为唯一生产路径，6 个新用例，5.49）→ 归约族补到 max / mean / argmax（`AtomicMin` 语句节点，三者共用一份骨架，argmax 输出 i32 且并列取最小下标，7 个新用例，5.50）→ 转置卷积导入（不新增算子：`ConvTranspose2d` 直接落到 `conv2d_input_grad`，`output_padding` 由 `input_shape` 表达，3 个前端用例 + 1 条 conformance 形态，5.51）→ 语义级 L2（相邻逐元素算子融合成一条链，中间值不物化；f32 链在长度能被 4 整除且指针 16 字节对齐时走 `float4` 四通道，10 + 7 + 5 个新用例，5.52）→ 前端承接新版 `aten.batch_norm` 的 eval 形态（5.52）→ 符号形状表达式（`Dim = int | str | SymExpr`，卷积 / 池化的派生空间维留在 IR 里，一次构图多分辨率执行，7 个 IR + 3 个 GPU 新用例，5.53）→ 执行计划层（`lower/plan.py`：block / grid / vec / shared 的决策与校验变成纯函数，后端只剩 launch，41 个离线新用例，5.54）→ 共享网格步进骨架（`grid_stride` 一处定义，逐元素单算子与融合链共用；归约暂存区改由 `REDUCE_SCRATCH` 供给，源码逐 token 不变，5.55）→ 分块卷积接入调优面（conv2d 的 `tile_w` 成为旋钮：计划层构造期校验 block 上限，调优空间首候选是"自动"，调优图集补上 conv2d 用例，7 个新用例，5.56）→ 多维索引核实（逐字比对后只合并了 `conv2d_input_grad` 的一处真实重复，其余差异是语义差异，5.57）→ HIP 调优面修复（`HipBackend.__init__` 漏掉 conv2d 的 `tile_w` 覆盖字段，使 HIP 上分块卷积在 launch 前抛 `AttributeError`；补一行并用 AST 对账两份 `__init__` 的实例字段，1 个新用例，5.58）→ 基准 CLI 的后端选择改为查注册表（`--backend` 从写死的 `ref/cuda` 变成 `ref` + `registered_backends()`，三个脚本共用一个工厂；明确指定却装不起来时一行拒绝，绝不退回 CPU，4 个新用例，5.59）→ WSL ROCm 实跑（DXG 直通下 `HSA_ENABLE_DXG_DETECTION=1` 是前置；补上 `_Stream.synchronize`——缺它时 HIP 上调优的 9 个用例全部 `AllRejectedError`；conformance 13/13、全量 760 通过，`tune` 搜出 9 条配置并逐位对拍一致，2 个新用例，5.60）→ HIP 运行期图捕获（hipGraph：模块级流栈 + `_Stream` 上下文、捕获期自动切异步拷贝、分配器钩子与 `hip/memory.py` 的 arena、`HipGraphPlan`；`reuse_memory=False` 明确拒绝，含 hipBLAS 的图在 WSL 真机捕获并 replay 通过，8 个新用例，5.61）→ HIP 性能对照（`benchmarks/capture_compare.py` 在同一套 8 个用例上跑"捕获 replay vs 逐节点下发"两条口径，`--backend` 走注册表、先对拍数值再计时；顺带修掉捕获期扩容导致 4 个用例捕获失败的真缺陷——`Arena.recycle()` 收回预热轮的地址，2 个新用例，5.62）→ 跨后端调优对照（`benchmarks/tune_compare.py` 两段式：采集复用 `tune.TUNE_CASES` 与 `tune.search`，报告记录赢家、默认候选与机内提升；对照只比配置是否相同与各自"相对默认"的提升，不比跨机毫秒。9 个单算子用例在 CUDA 与 HIP 上各采两次，赢家两次之间只有 2/9 与 4/9 一致、跨后端 2/9 与 0/9 一致；重复测量与交替 A/B 证明 1M 逐元素算子的 `block` 真实差异 ≤3%，淹没在整机 30% 的漂移里，"最优 block"不可判定；9 个新用例，5.63）→ 硬件包的内容摘要与依赖声明（manifest 新增两个可选字段：`files` 是逐文件 sha256，声明了就要求与包内文件**完全一致**（多一个未声明文件也拒绝，`__pycache__` / `*.pyc` 是运行期产物不计入）；`requires` 声明用到的算子（必须在框架算子表里）与 Python 模块（用 `find_spec` 查，不 import）。安装期与扫描期都校验，坏包按既有策略跳过；摘要只证明"包在打包后没被改过"，不证明"谁发的"，来源签名仍没做；示例 `build.py` 改为现算清单；16 个新用例，5.64）→ 自动分块（`lower/plan.py` 的 `auto_block` 把 block 的取值变成纯函数：不派空线程 + 网格至少铺满一屏 SM 两条规则，`block=None` 成为"自动"这一可调优、可回退的状态，22 处 launch 收敛到一处解析，12 个新用例，5.65）→ 训练循环（`train/loop.py` 的 `Trainer`：反向图建一次，种子名/形状与多输出图的零种子由它负责，符号形状在构造期拦下并解释根因；9 个新用例，5.66）→ 硬件包来源签名（manifest 可选 `signature`：Ed25519 覆盖"除 signature 外的整份清单"，信任锚由安装方的 `trusted_keys` 给出，包内公钥不能自证；带签名不给信任键、给了信任键包未签名、公钥不受信三种情况都拒绝，未签名且未要求来源仍可装，9 个新用例，5.67）→ 动态执行第一版（`DynamicRunner` 按输入形状建图并按 LRU 复用，`build` 只能依赖形状，数据相关分支留在两次 `run` 之间；一次 `run` 内部仍是静态图，10 个新用例，5.68）→ cuDNN 对照（`benchmarks/cudnn_compare.py` 用 ctypes 直调 cuDNN 9 的 legacy API，同形状同 dtype 同时刻对照：自研分块核慢 4～12 倍、im2col+cuBLAS 慢 3.2～8.8 倍，中等通道的 3x3/5x5 差距最大；只做测量，不接生产路径，5.69）→ CI 作业真机逐条实跑（正确性三步 + 调优往返 + 趋势采集全绿；性能门禁为红，基线不重采，5.70）→ 内核源码惰性生成（`_kernel` 只在编译缓存未命中时才渲染源码，31 处 `codegen.render` 改惰性，分块卷积的缓存键补上 dtype；conv2d 单步 1.19 → 0.45 ms、cnn_lenet −61%，门禁差距由 3.4～4.5 倍收到 1.3～1.6 倍，5.71）→ 执行计划缓存（融合分组 / 静态 attrs / 期望类型 / 铺法只在建图后算一次，`_contig` 跳过冗余的连续性转换；host 侧 transformer 2.09 → 1.89 ms、cnn_lenet 端到端 −14%；以 `matmul_256` 为整机漂移标尺归一化后，卷积与 CNN 已追平、只有 transformer_block 还剩 23%，5.72）→ conv2d bias 融合（分块核写回时就地加 bias，省掉每步一次 `(1, O, 1, 1)` 的广播核；三处同改：L2 参数与写回表达式、手写对照模板、launch 传参）与符号图解析备忘（`_GroupPlan` 按本次绑定记住 attrs 与期望类型，形状切换即失效；卷积 −11%～−13%、transformer 1.495 → 1.362 ms）；门禁 8 个用例全绿、基线仍未重采，按 `matmul_256` 折算后只剩 transformer_block 的 8%，5.73）→ conv2d 的两条快路径（cuDNN 接成 `conv_impl="auto"` 的可回落实现，分组与 f64 仍走自研核；分块核的 `roc` / `cb` 也进调优面，5.74）→ 激活重算（`train/recompute.py` 分段前向只留边界、反向逐段重放，链式模型梯度与整图逐位一致且静态峰值下降，5.75）→ 部署包（`deploy.py` 把图 JSON + 权重 + 清单摘要装进一个 `.afpkg`，加载即校验、跨进程逐位一致，5.76）→ eager 前端的即时求导（`loss.backward()` 建一次反向图并按图版本缓存，仍是静态图不是 tape，5.77）→ 热路径上的三个 Python 常数（错误信息惰性化、dtype 在计划期解析、去掉多余小分配；门禁另加 0.5 s 热身以消除空闲降频误报，折算后 3.6%，5.78）→ 常规执行路径接进 CUDA Graph（`graph_mode="auto"` 按"图 + 结构版本 + 输入签名"缓存捕获计划、LRU 4 份、捕获失败回落逐节点；`transformer_block` −56%、`cnn_lenet` −55%，单算子图无稳定收益，默认仍 off，7 个新用例，5.79）→ 基准补"权重常驻"口径（`Case.split` + `prepare_case`，参考后端仍用完整输入；`transformer_block_resident` 1.0206 → 0.2179 ms、`cnn_lenet_resident` 0.5758 → 0.1264 ms，叠加 auto 后分别比基线 −81.6% / −81.2%，2 个新用例，5.80）→ 自动图按节点数开闸（`min_graph_nodes=2`，单算子图不进捕获、多节点图照旧；反向顺序复测证明 5.79 记的 `matmul_batched` +8% 是整机漂移，1 个新用例，5.81）→ CUDA 训练闭环有了门禁（`Trainer + CudaBackend`：线性回归收敛、CNN 一步与参考逐位一致、30 步损失下降），同时修掉换机才暴露的 cuDNN 描述符指针 bug（3 个新用例，5.82）→ 训练 step 进图（前向、反向各一张捕获计划，10 步的损失与全部参数和逐节点执行一致）与视图内存的生命周期（arena 按视图引用计数延迟回收，捕获峰值不涨；2 个新用例，5.83）→ 参数常驻显存的训练步（`Trainer(device_state=True)`：`capture` 新增 `shared_buffers`，前向与反向共用同一组权重缓冲，优化器在显存上原地更新，一步只有损失标量回主机；SGD 10 步与 Adam 5 步的轨迹都与主机态一致，2 个真机用例 + 1 个换表用例，5.84）→ 训练辅助工具跟着参数留在显存（`train/_arrays.py` 的 `array_module` / `to_host` / `host_scalar`，`clip_global_norm`、`GradientAccumulator`、`EMA`、`checkpoint`、`GradScaler.unscale` 全部按数组所在模块分派，设备态下只在取标量时同步一次；`grad_hook` 因此可以直接放裁剪与累积，2 个新用例，5.85）→ 数据侧 pinned 暂存（`pinned_array` 按 `(名字, 形状, dtype)` 复用固定内存，`replay` 里 numpy 输入走异步 H2D，共享缓冲直接跳过；配合"进流前先等一次当前流"的事件依赖，每步最后一次 H2D 的等待消失，5.86）→ HIP 对齐同一套运行期语义（`shared_buffers` / `stream` / `run_device` / 自拷贝跳过；不接 pinned、不支持设备态训练都在第一次 step 明确报错，5.87）→ 设备态叠加激活重算（逐段捕获、逐段 replay，全部共用一条流；边界值与 cotangent 分两组缓冲，数据输入只在首个使用它的段里走 pinned 上传；本地用 numpy 替身后端验收调度，1 个真机用例，5.88）→ 设备管理器与分层分配器的契约上提（`StreamContract` 独立成协议、`AllocatorContract` 只收两个 Arena 共有的五个方法、`MemoryContract.allocator()` 交出**新**分配器；`InstrumentedBackend` 按存在性转发新契约——这条是真机全量第一次跑才暴露的；同时给基准报告加机器指纹，换宿主后整体慢 1.9～2.7 倍因此只提示、不覆盖基线，3 个新用例，5.89）→ 自动混合精度（`passes/autocast.py` 的图重写 pass：matmul / matmul_batched / conv2d 降 f16，softmax / 归一化 / 归约强制回 f32，其余取最宽输入，图输出恢复原 dtype 因而幂等；反向不用特判，梯度天然落回 f32。真机上顺带修掉一条从未被走到的缺陷——`CudaBackend.cast` 到 f16 走 cupy 的 `astype` 会链接不到 `__half` 的构造函数，改为自研 PTX 内核 `ew_cast` + 只给这条内核用的 `CAST_DIALECT`（HIP 侧同一份描述换 `_Float16` 方言）；另把 torch 2.13 导出图里的 `aten.relu_` / `aten.add_` 降到函数式同名版，20 + 1 个新用例，5.90）→ 混精接进训练循环并量化（`Trainer(mixed_precision="f16")` 在构造期改图，反向图与两张捕获计划自动跟随；参数仍是 f32，优化器与存档不用改。真机 7 组口径实测：单算子 f16 GEMM 确实更快，但训练步在主机态慢 6～10%、设备态基本持平——重写把前向节点 11 → 17、反向 39 → 51（其中 12 个是 cast，因为 cast 的 VJP 也是 cast），主机态每多一个 cast 节点约 0.08～0.12 ms，与 GEMM 省下的时间同量级。开关默认关闭，6 个新用例，5.91）→ 换宿主后的基线重采（8 个用例齐平慢 52%～176%，形状确认是宿主差异而非回归；新基线带 machine 指纹、覆盖 10 个用例，把 5.80 的两个 resident 口径一并纳入；GPU 作业五步在真机复跑全绿，门禁由红转绿；托管平台仍未实跑，因为这份工程没有 git 仓库与 runner 令牌，5.92）→ 剥离派发后的 GEMM 对照（单算子方阵图、输入全常驻、`graph_mode="auto"` 只 replay：f16 相对 f32 为 1.19×（batch 128）/ 1.33×（batch 1024）；把 5.91"消掉 cast 就能拿回收益"的说法收窄成"必要条件而非充分条件"，混精在这个规模上的价值是省显存；顺带让 `cast` 的 VJP 在上游梯度已是输入 dtype 时直接透传，2 个新用例，5.93）→ 外部审计的 P0：softplus 溢出（`log(1+exp(x))` 在 f32 上 x>88 就 inf，inf 经 `a = exp(dt*A)` 污染整条 scan、反向变 NaN；改成 `max(x,0) + log(1+exp(-|x|))`，指数入参被压到非正半轴、两头都不溢出，绝对值用 `relu(x) + relu(-x)` 拼、不新增算子；3 个新用例，5.94）→ 外部语义审计的逐条核对与修复（报告基于 `853 通过 / 291 跳过` 的旧快照，先按当前代码逐条复核：AF-02 已在 5.94 修掉、AF-29 由 5.84 的设备态部分解决，其余仍成立的条目里本轮修 13 条可复现的语义红线：交叉熵改稳定 log-softmax、`Value` 走对象身份、自动 SSA 名避让用户输入、序列化显式写节点输出名、中间值可作 `wrt`、分段重算种子与下游 cotangent 相加且只对浮点输入求导、融合链与 CUDA Graph 的缓存键补全、优化器两阶段提交与槽位完整性、checkpoint 先校验后原地写、Adam 拒绝 f16 参数；14 个新用例，5.95）→ 阶段 A 收尾（attention 的加性偏置在广播时按被广播轴求和返回梯度——旧规则只在形状完全相同时给、把可学习的位置偏置当成常量；前端按 ATen schema 归一化参数读取：`add`/`sub` 的 kw-only `alpha`、`layer_norm` 的**位置** `eps`、SDPA 的 kw-only `scale` 与 `enable_gqa` 此前都读错了位置、会被静默换成默认值，另加"不认识的参数就拒绝"；6 个新用例，5.96）→ 阶段 A 续（未激活输出按可达性剪枝，不再靠喂零种子——零种子在 `0/x` 上仍是 NaN；`backward` 新增 `active=`，训练器与分段重算都只给损失那一个种子；符号维与静态非 1 或另一个符号的广播改为构图期拒绝，IR 里没有能表达"这一维可能是 1"的广播表达式；形状符号禁止与 dtype 名重名，否则 `f32[f32, 4]` 与属性里的 `dtype="f32"` 无法区分；输入在执行入口转换到 IR 声明的 dtype，不再"按 f64 算、输出才降 f32"；10 个新用例，5.97）→ 阶段 A 收官（AF-16：捕获期把越界标志登记进计划，replay 同步后复查、报错前清零，普通执行与图重放共用同一条错误消息，HIP 对称实现；AF-20：`dropout_mask` 增加运行期 i32 key 输入与 `offset`，随机源因此放在设备内存里、捕获后 replay 仍能重新采样，`Trainer(rng_inputs=...)` 每步推进，同一 step 的前向 / 分段重算 / 反向共用同一个 key；11 个新用例，5.98）→ 阶段 B 的第一块：SSM 流式推理（`build_selective_scan_stateful` 导出末状态、`reset` 掩码把递归在重置点切断；`build_mamba_block_stateful` 让卷积窗口与 SSM 状态都成为图输入/输出，`steps=1` 的同一份构图就是 decode；三条恒等式——任意分块 == 整段、逐 token == prefill、带 reset == 各段独立跑——在参考后端与 CUDA 上都成立，13 个新用例，5.99）→ 阶段 B 续：多层 Mamba 栈（`_mamba_layer` 抽出单层主体，`build_mamba_model` / `build_mamba_model_stateful` 把 N 层放进**同一张**图，键带 `l{i}_` 前缀；状态出参只差输入名末尾那个 0，`spec.carry_state` 负责翻译，分块续接就是 `feed.update(...)`；`with_steps(1)` 的同一份构图即 decode；单层退化与 `build_mamba_block` 逐位一致，整栈的分块 / 逐 token / reset 三条恒等式在参考后端与 CUDA 上都成立，7 个新用例，5.100）→ 阶段 B 续：Mamba 专属初始化（通用初始化让 `A_raw≈0`、`dt_b=0`，于是 `a≈0.5`、所有通道半衰期都约一个 token，长依赖没有梯度压力；`init_mamba_params` 改成直接指定目标半衰期网格——`(d_inner, d_state)` 每个单元在 1～32768 上对数等距分到一个 `H`，`dt_b` 逐通道在 `[1e-3, 1e-1]` 上对数等距给出名义 Δ，再由 `|A| = log2/(H·Δ)` 反解 `A_raw`，`dt_w` 初始化为零以免激活幅度盖过分层；`effective_half_lives` 用同一把尺子报告训练前后的分布，`MambaModelSpec.init_params` 逐层生成、靠名字前缀区分随机流；4 个新用例，5.101）→ 阶段 B 续：流式状态进存档与 AF-27/AF-28 量化（`save_checkpoint` / `load_checkpoint` 新增可选 `state`：键直接用图输入名 `l{i}_conv0` / `l{i}_h0`，加载后就是 decode 的 feed，模板原地写入并校验形状 / dtype；"传了目标却没存档"报错、"没传目标"忽略，旧格式存档照常可读；判据是 prefill → 存档 → 新进程式重建 → 逐 token decode 与从未中断**逐位**一致。AF-27 用 `plan_memory` 固定下来：节点数恒为 14，T 翻倍时中间值与峰值按每 token 6848 / 3232 B 线性增长；`state_shapes` 的字节数与已处理 token 数无关，"每 chunk 恒定状态"的边界因此可量化。AF-28 在 RTX 2080 Ti 上实测：14 节点整图在 T ≤ 512 时被 14 次 launch 的 ~1.6 ms 固定开销主导，隔离出的单节点 `scan_linear` 在 4096 → 16384 严格 4 倍、65536 时每 token 涨 35%，batch 从 1 到 16 耗时按总元素数增长、不摊薄——只记录口径，不改 kernel、不宣称加速；8 个新用例，5.102～5.103）→ 阶段 B 收官：混合长度与恒定显存（left padding + 有效起点 reset 让 batch 内不同长度的样本等价于各自单独跑，right padding 做不到——pad 会继续推进状态、末状态对不上；reset 是逐样本的，同一 batch 里一个样本重开一段不影响其他样本；真机上预热后连续 200 步 decode，cupy 默认池的 `used_bytes` 恒为 512 B、增量为 0，`total_bytes` 停在 56,320 B——池扩到这么大就不再要，历史既没被物化、临时量也被复用；3 个新用例，5.104）→ 审计 AF-31：预算口径跟着执行策略走（`plan_memory` 新增 `release` 参数，与 `Executor.run_device` 的同名开关一一对应：`False` 时所有值活到图结束、峰值等于总量、`reuse_ratio` 恒为 1；`MemoryBudget.release` 与 `as_dict()` 让引用数字必须带口径。`Executor.run` 固定走 `release=True`——它只读输出、不返回环境，普通执行因此真的回收中间值，而 `run_device` 保留"交出完整环境"的契约。峰值算法从逐时刻扫描改成事件扫描（O(V log V)），与测试里的暴力实现逐项对照。文档把"严格下界"改写为"指定存储假设下的 IR 估计"，并列出三条假设与方向性偏差：视图高估；内核临时区 / 库 workspace / 优化器槽位 / 捕获缓存 / 主机副本低估。真机 6 段 `relu(matmul)` 链：`run` 池总量 20.0 MiB、`run_device` 56.0 MiB，静态估计 8 / 48 MiB，差出来的约 12 MB 正是声明不算的 workspace；3 个新用例，5.105）→ 审计 AF-32 的图 seal（`Graph.seal()` 把"构建后只读"变成代码：`input` / `add_node` / `output` 统一拒绝、`Value` / `Node` 字段不可写、`Node.attrs` 变只读映射且形状类 list 转 tuple；`serialize` 的嵌套属性判断从 `dict` 放宽到 `Mapping`，seal 后的图照常导出；`revision` 因此是真正的不可变凭证，执行计划与捕获计划不会在"图已变、键未变"时静默命中；3 个新用例，5.106）→ 审计 AF-32 的后端能力契约（`BackendCapabilities` 是 `frozenset` 子类，集合语义与既有对账代码完全兼容；为算子附 `OpCapability` 明细：dtype / axis / layout 是排除项、`None` 表示未声明限制，backward / capture / deterministic 是用途承诺；`refined` / `unrefined` 让"只有名字"可以被单独看见。参考后端逐个声明 49 个 PORTABLE 算子的形态：backward 与算子表 VJP 一致、capture=False、deterministic=True，dtype / axis / layout 不设限；`verify` 与 `unsupported_ops` 在名字都命中时继续按输出 dtype、axis/axes、layout 判定，把"算不了这一形态"提前到构图期；conformance 报告区分"带形态明细"与"仅算子名"，只声明名字的旧式后端行为不变、不判失败；8 个新用例，5.107）→ 审计 AF-32 的算子效果描述（`ir/ops.py` 新增 `Effect` 枚举 RNG / STATE / IO 与 `Op.effects` / `Op.pure`，空集即纯函数，STATE / IO 汇总成 `UNREPLAYABLE_EFFECTS`；CSE 不合并非纯节点、DCE 把非纯节点当可达性根（`saved()` 只算纯节点）、常量折叠删掉 `_RANDOM_OPS` 白名单改判 `op.pure`、`CheckpointPlan` 在构造期拒绝会被重放两次的 STATE / IO；内置算子表里只有 `dropout_mask` 声明 `RNG`——它不能折叠也不能合并，但显式 key 下可以在激活重算里重放；11 个新用例，5.108）→ 审计 AF-32 的 HIP / CUDA 大类拆分（`backends/device_backend.py` 新增 `DeviceBackend(Executor)`：110 个成员、49 个算子面、执行计划 / 融合分组 / 调优 / 捕获缓存全部中立，厂商只留 9 个钩子；`CudaBackend` 与 `HipBackend` 变成兄弟类，继承树从 `HIP → CUDA → Executor` 改成两条 `→ DeviceBackend`。顺带拆掉 `hip/runtime.py` 里"假装是 cupy"的 `_CudaNamespace`，`current_stream()` 成为公开函数；拆分过程中修掉两个真 bug——`memory_stats` 被错放在 CUDA 一侧（HIP 上 `AttributeError`）、`transpose` 读视图的 `.base`（HIP 容器没有该属性，conformance 两个用例一直 FAIL）；`TestHipInitParity` 换成 `TestBackendSplit`，断言 MRO 与构造字段集；本地 1298 / WSL HIP 1298 / CUDA 真机 1298 三处全绿，5.109）→ 审计 AF-09 的交错输入序列化（根因已由 5.95 的"节点输出显式写名字"解决，本轮把证据补齐：审计原例现在 `to_dict(loads(dumps(g))) == to_dict(g)`、二次 `dumps` 逐字节相同、`verify` 无 issue、执行输出逐位一致，回读只把值 id 由 `[0,2]` 重排成 `[0,1]`。因此不采纳"input() 在节点出现后拒绝"这条短期方案——它假定格式表达不了交错，而该前提已不成立，加限制反而堵死 eager 的按需声明；需要守的是"名字到值的双射"，两个方向已在 `input()` / `add_node()` 上堵住。同时把"值 id 不跨序列化稳定、契约是名字与类型"写进 `serialize` 模块说明与两处 docstring，并修掉 `models/transformer.py` 里"输入必须在节点之前是硬约束"的过期注释；1 个新用例，5.110）→ 审计 AF-15 的收尾（5.83 的视图引用计数只覆盖 `reshape` 与恒等 `transpose`：`slice` 与退化成恒等的 `broadcast_to` 同样返回零拷贝视图，而 slice 的 `ptr` 带偏移，按视图地址扣账一条也匹配不上；HIP 侧则根本没有 `dispose`，捕获期中间值从不回收、复用收益为零。抽出 `_shared_view` 让四个返视图的算子统一登记，`out is src` 也登记——执行器按两条 Value 各 dispose 一次，第二次靠新增的 `Arena.pending` 认出同一段地址、扣账而不是重复 free；HIP 的 `Arena` 补齐 `alias` / `unalias` / `pending`，`DeviceArray` 加 `_view` 标记，`dispose` / `_alias_view` / `_shares_storage` 三个钩子接上。负向对照：短路登记后 CUDA 新旧两条别名用例、HIP 的 slice 用例全部失败（100% 元素错），恢复后三处全绿；3 个新用例，5.111）→ 审计 AF-19 的收尾（`save_checkpoint` 不再直接 `np.savez(target)`，改成同目录临时文件 + `fsync` + `os.replace`：写到一半被打断时旧存档原样保留、目录里不留半成品；`load_checkpoint` 的提交顺序改成 读取 → 参数校验 → 状态模板校验 → 预转换 → 优化器提交 → 原地写入，设备态的上传也挪到提交之前，坏状态模板不会再留下"优化器已恢复、参数还是旧的"反向半步。负向对照：退回直接 `np.savez` 时旧存档被写成半截，把优化器提交挪回状态校验之前时 `step_count` 变成 1；3 个新用例，5.112）→ 审计 AF-33 的统计口径与失效版本（`tune` 的 `median_ms` 现在真的是中位数，`best_ms` / `p95_ms` / `spread` 分开报；新增 `tie_epsilon=0.02`，差距不显著时保留默认候选。`STORE_VERSION` 与调优对照报告都升到 2，`best ≤ median ≤ p95` 在加载期校验、v1 显式拒绝并说明两个口径的差别。捕获失败从"整实例关掉"改成按图隔离：`_graph_failures` 记原因、`graph_capture_failure(graph)` 可查，同图同版本不再重试，其他图照常捕获。调优覆盖 / conv2d 旋钮 / 驻留权重统一走 `_invalidate_plans`：`_config_epoch` 递增并清铺法、捕获计划与失败记录三层，epoch 也进计划键。负向对照：退回按最小值排名、去掉阈值、让失败原因恒为空，对应用例全部失败；HIP 上用 `HipBackend` 实测三层缓存一起失效。HIP 上实跑调优往返：9 条 v2 记录、`--apply` 命中 9 条且逐位一致，3 个用例在阈值内保留默认；过程中修掉 `benchmarks.run._sync` 只认 `cp.cuda` 的缺陷（5.109 拆掉假命名空间后 HIP 上第一次 `measure` 就 `AttributeError`），补上模块级 `hip.runtime.synchronize()`。本地 1318 / WSL HIP 1318 全绿，CUDA 真机因研究机停机未复跑；13 个新用例，5.113）→ 审计 AF-23 的稳定性诊断（`decay_diagnostics` 把"Δ>0、A<0 ⇒ 稳定"换成可检查条件：`q=max(a)` 必须严格小于 1、`M=max|Δ·B·x|` 必须由输入给出，两者齐了才报 `q^t·|h0|+M/(1-q)`，否则在 `state_bound_reason` 里说明缺哪一条；同时报 Δ / log a 的分位、非有限值与索引、以及 f16 下舍入成 `a=1` 的单元数——初始化里最长半衰期的通道在 f16 下必然退化，报告而不掩盖。负向对照：放宽成 `q<=1`、无输入时把 `M` 填 0，对应用例失败；8 个新用例，5.114）→ 审计 AF-25 的 Mamba 口径与扫描对拍（模块说明明确"受 Mamba 启发的变体"、三处结构差异、不承诺官方权重兼容与性能继承；新增 `mamba_reference.official_selective_scan` 按 mamba-ssm `selective_scan_ref` 的语义逐步递推，与 `build_selective_scan_stateful` 在 `h0` / `reset` / `D` / 门控下逐位对拍；`map_official_params` 只映射 `A_log→A_raw` 与 `D→D_skip`，其余项列入 `unmapped`，`from_official_checkpoint` 因此拒绝整块迁移并解释三处差异。负向对照：翻转参考实现的 A 符号、跳过 `unmapped` 检查，对应用例失败；7 个新用例，5.115）→ 审计 AF-30 的前向 mask（`attention_fwd` 新增 `mask_shared`：`(1,1,S,T)` 的因果 mask 不再被复制成 `(BH,S,T)`，内核按 `(mask_shared ? 0 : bh)*S+s` 寻址，主机只在"最后两维正好 `(S,T)`、其余维全 1"时共用一份，`(1,S,1)` 这类可广播但不是 `(S,T)` 的形态仍走广播复制；HIP 实测同形状共享路径分配 5.0 MB、全量 mask 36.0 MB，差 31 MB 正是 `BH-1` 份。第一次改动落在 `cuda/codegen.py` 的非生产对照模板上，HIP 对拍 82% 元素不一致才定位到生产内核在 `lower/kernels.py`；3 个判断用例 + 1 个 HIP 对拍，5.116）→ 审计 AF-30 的反向量化（先用 `plan_memory` 量化再决定要不要写内核：B=8 / H=4 / S=T=512 / D=64 时一份 `(BH,S,T)` 缓冲 32.0 MB，反向图 26 节点，`release=False` 总量与峰值都是 439.1 MB（≈13.7 份），按"最后一次使用"回收后峰值 108.1 MB（≈3.4 份）。真实路径固定 `release=True`——`Executor.run` 与 CUDA/HIP 的 `capture` 都传它，所以 439 MB 只是"直接调 `run_device` 又不传 release"的上限。朴素下界"输入 + 输出"是 7 份 224 MB，反而高于复用后的 3.4 份，说明组合展开 + liveness 复用已经比手工分块更省；专用反向内核的收益只在计算（省一次 scores/probs 重算），代价是新增仅 CUDA/HIP 可用的算子与两套测试，因此不写。补 `TestAttentionBackwardBudget` 钉住"复用至少省掉三分之二"；1 个新用例，5.117）→ 审计 AF-29 的训练状态凭证与异步快照（前三条"设备端参数 / 槽位 / 捕获缓冲"已由 5.84～5.88 落地，本轮补审计最后那句"验证"：`Trainer.training_state()` 分开报 `steps` 与 `optimizer_step`（`grad_hook` 返回 `None` 的步只前进不更新），`host_in_sync` 说明 `store` 里是不是旧值；`Trainer.verify_device_state()` 只比对象身份与数组模块、不做 D2H，检查"优化器 params → 驻留参数 / 每张捕获计划里的缓冲 → 驻留参数 / 槽位与参数同一数组模块"三条链，任何一处分叉都抛 `RuntimeError` 并列出全部问题，为此 `CudaGraphPlan` 与 `HipGraphPlan` 新增 `buffers` 只读属性；`Trainer.snapshot()` 在 step 边界把参数与槽位拷成主机副本，`save_async` 交给模块级单线程池并返回 `Future`——**异步的是写盘、不是拷贝**，拷贝必须在调用方选定的时刻完成，否则会拿到跨若干步拼出来的权重；`Trainer.save_checkpoint()` 让设备态不必先 `sync_to_host()`（那一版会覆盖 `store`），快照另记 `trainer_steps`。`device_state` 默认仍为 `False`：设备态需要 `capture` / `to_array` / `to_host`，参考后端没有，默认开会把构图期说明变成运行期异常。负向对照三组：关掉优化器引用检查、关掉捕获缓冲检查、去掉快照拷贝，对应用例分别变红；12 个新用例，5.118）→ 审计 AF-34 的第一块，SSM 语言模型图内闭环（`models/lm.py`：`embedding` → N 层 Mamba（新增 `build_mamba_layers` 从已有 Value 叠层）→ `rmsnorm` → 词表头（`tie_weights` 时 `embed^T`）→ `masked_lm_loss`（只除以有效 token 数，全 padding 的一批是 0 而非 NaN）→ 流式 decode（`with_steps(1)` + `carry_state`）→ 主机侧 `sample_next_token`（greedy / temperature / top-k）→ `generate`（EOS 或 `max_new_tokens` 停止，只支持 batch=1）。两条流式恒等式（分块续接 == 整段、逐 token == prefill）与损失/采样/停止各有可执行判据；负向对照两组（去掉 mask 加权、EOS 先入列）分别变红。16 个新用例全部在参考后端上跑，5.119）→ 审计 AF-34 的第二块，词表双射 / chat 模板 / batch mask / 配置绑定（`models/lm_text.py` 的 `ByteTokenizer`：256 字节 + `<pad>=0` / `<eos>=1`，无 BPE，`decode(encode(s)) == s` 对任意 UTF-8 成立，越界 id 与截断 UTF-8 都拒绝；`ChatTemplate` 纯文本拼接、**不新增特殊 token**（新增会改词表让旧嵌入表整体错位），`add_generation_prompt` 补 assistant 前缀即 prefill 输入；`train/lm.py` 的 `lm_batch` 做 **left padding**（右侧 pad 会继续推进 Mamba 状态、末状态对不上），`targets` 与 logits 同形因而省掉一次切片；`LMSpec.meta()` / `check_meta()` 把形状与词表约定写进存档凭证、缺项即"未绑定"拒绝，布尔先于整数判断以免 `False` 被当成 `0`。负向对照两组：改成右对齐、eos 不停止解码，对应用例分别变红；25 个新用例，5.120）→ 审计 AF-30 的第三块，f16 融合注意力（前向内核改成**混合精度模板**：`q/k/v/mask/out` 按 dtype 声明指针（f16 下是 `unsigned short*`），读取走 `Load` 的 `af_ld_h`、写回走 `Store` 的 `af_st_h`，而共享内存、行最大值、归一化因子与输出累加器仍是 float——f16 只省存储，softmax 不在半精度里做。为了让"指向存储类型的指针"能被描述，`Let.type_name` 支持 `%T%` 占位符（`const %T%* mrow`），打印器按 dtype 展开；`Ptr` 接管 qs / vbase / orow 三处 base+offset 指针，`Index → Load`、`Assign → Store` 让存取跟着 dtype 走。生产侧 `_op_attention` 的输出 dtype 改为跟随 `q`、内核缓存键加上 dtype、渲染时传 `q.dtype`，并在运行期再校验一次 mask 与 q 同 dtype（IR 已有契约，内核按同一宽度读两者，混用会静默读错位宽）；`_F16_OPS` 的默认集从"`_OPS` 减 attention"变成 `_OPS`，49 个算子全部支持 f16。本机没有支持 `enable_gqa` 的 torch 时 `test_frontend` 的那条用例会直接 TypeError——它依赖 torch ≥ 2.5，与本次改动无关。负向对照两组：输出缓冲退回 f32（f16 内核按半字写，结果成垃圾）、mask 指针退回 `const float*`（HIP 编译期报类型不匹配），各自变红后还原；3 个新用例（1 个 emit 级 + 2 个 HIP 真机对拍），5.121）→ 审计 AF-34 的第三块，工具调用的受约束输出与执行回路（`models/lm_tools.py`：`JsonGrammar` 是**字节级** JSON 文法——`ByteTokenizer` 让"允许哪些 token"与"允许哪些字节"成为同一个集合，因此不需要前缀树、不需要 BPE 合并表，也不需要在每步把 token 展开成字节；状态由"标量阶段 + 容器栈 + 逐项 phase"三部分组成，对象区分 `key` / `key_more`（逗号之后不允许直接 `}`）以免放过尾逗号，数组区分 `first` / `value` 同理；数字的结束是隐式的，`complete` 必须按阶段判定（`12` 完整、`12.` 与 `1e` 不完整），否则写完整数不会收尾。`generate(constraint=...)` 每步用 `allowed()` 把其余 logits 置 `-inf`，`sample_next_token` 据此把"拒绝 NaN / +inf"与"允许 -inf（合法的屏蔽）"分开，并显式拒绝"整行被屏蔽"；文法没走完时抛错而不是返回半截 JSON——半截比报错危险得多。执行回路 `Tool` / `ToolRegistry` / `parse_tool_call` 只认 `{"name": ..., "arguments": {...}}` 这一个形状，必填 / 未知 / 类型不符全部拒绝，`bool` 不算 `int`、`float` 接受整数。`BYTE_OFFSET` 由 `lm_text.py` 公开导出，字节到 id 的映射不再各复制一份。负向对照两组：去掉 logits 屏蔽（约束只是摆设，`feed` 立刻拒绝越界 token）、`_after_value` 不区分 key 与值（多键对象的第二个 key 之后期望逗号而非冒号，合法 JSON 被拒），各自变红后还原；29 个新用例（正向闭包 21 条合法 JSON 逐字节 / 反向非法状态 / 随机游走 12 组必须 `json.loads` / 约束下的生成与未完成报错 / 执行回路），5.122）→ 阶段 C 的第一块，把散装的训练辅助串进 `Trainer`（`schedule=` / `accumulate_steps=` / `scaler=` 三个参数，默认值让老代码逐位不变；顺序固定为 unscale（每微批）→ accumulate → grad_hook → apply_lr → optimizer.step：反缩放必须在累积之前、否则窗口里混着不同 scale 的梯度，累积必须在钩子之前、否则裁剪上限对不上有效 batch，调度必须在真正更新之前、否则 warmup 会被微批次数提前用完。`accumulate_steps==1` 不建累加器，因为 `GradientAccumulator` 固定 f32 缓冲、会让 f64 图的梯度被悄悄降精度。scaler 与累积叠加时新增 `GradScaler.update` 接受已反缩放梯度（再调 `step` 会二次相除），溢出时 `GradientAccumulator.discard()` 整窗作废而不是只丢出错那一批（累加和里减不掉单批贡献）。缩放状态三标量以 `amp_*` 进快照 meta、`load_amp_state` 恢复，缺键或 `scale` 非有限正数都拒绝。设备态的种子改写缓冲内容即可推进（捕获写死的是地址不是值），分段路径的 cotangent 读同一个值。负向对照两组：注释掉 `apply_lr`、去掉 `discard()` 各自变红后还原——第一版整窗用例先喂坏批，窗口本来是空的，丢不丢都一样，改成先喂好批才有判别力；13 个新用例，5.123）→ 审计遗留的 f16 反向量化（`Autocast._align` 只对齐浮点操作数：此前把 i32 的 `ids` 也 cast 成 f32，带 embedding 的真实语言模型在构图期就 `TypeError`，5.90 的 20 个用例全是 matmul / softmax 因而没走到；`low_precision` / `force_f32` 的 `or` 改成 `is None`，显式空集不再回落默认集。同一份参数与数据对拍四组口径：混合精度（matmul 降 f16，scan / 归一化 / 归约留 f32）2 层 / 16 步最差梯度相对误差 2.3e-3、余弦 ≥0.999997、无 NaN，扩到 4 层 / 64 步为 4.0e-3，**不随深度与长度放大**，24 步训练轨迹的 loss 偏差 2.9e-4、参数偏差 4.9e-3；白名单扩到全部算子时 `l0_A_raw` 的梯度相对误差 **2.63e-1**、余弦 0.9696，它的梯度范数 6.5e-6 比 `embed` 小六个数量级——A 只通过状态衰减影响 loss，梯度是多时间步**相消**后的残差，各项在 f16 里的量化噪声在相消后占主导，`GradScaler` 同比放大量化噪声、救不回已丢的低位，把 `scan_linear` 单独留 f32 也无改善（26.3%）；纯 f16 的 scan 图（x / dt / A / B / C / D 全 f16）在 T = 4～64 上 A 的梯度误差 1.10e-3～1.67e-3、不随长度放大，scale 从 2^0 到 2^12 误差不变、2^14 时 `D_skip`（梯度范数 24）溢出成 7/16 个非有限值——scaler 的窗口由梯度的动态范围决定，必须保留溢出检测与回退。结论：`mixed_precision="f16"` 可用于训练，把逐元素链也降 f16 不可用（scan 本身是安全的）；3 个新用例，5.124）→ 阶段 C 的 RNG key 派生（`Trainer` 原来把 `steps` 直接写进 `rng_inputs`：单机可复现，但同一 step 的所有 rank 拿到同一个 key、同一张 dropout 掩码——数据分片不同、噪声相同，多卡的梯度噪声完全相关。新增 `_derive_rng_key(seed, step, rank)`，用 SplitMix64 的 finalizer 把三元组混成 i32（取 31 位正数；不用 `step + rank*K`，加法会让相邻 rank 落在等差线上、低位相关性传进掩码），`Trainer` 新增 `rng_seed=0` / `rng_rank=0` / `rng_world_size=1` 并校验 `0 <= rank < world_size`、`0 <= seed < 2^31`；`_rng_key()` 同时供主机的 `_rng_feed` 与设备态的 `_advance_rng`，两条路径的 key 一定一致，key 仍是图输入、同一 `(seed, step, rank)` 逐位可复现、捕获成 CUDA Graph 之后照常推进。负向对照把派生退化成 `return int(step)`，3 条用例变红后还原；4 个新用例，5.125）→ 阶段 C 的 f32 master weights（`Autocast` 新增 `low_precision_inputs`，把指定图输入直接声明成 f16；`Trainer(master_weights=True)` 要求 `mixed_precision="f16"` 且参数全 f32，图输入因此是 f16 驻留缓冲、`store` 仍是 f32 主副本，每次真的更新后把主副本覆盖写回驻留缓冲，设备态下上传 f16 / f32 两组缓冲、每步末尾做一次 D2D cast，`sync_to_host` 与快照都取 f32 主副本（取 f16 会把精度永久降一级）；`_update` 改为返回"参数是否真的动了"，`grad_hook` 返回 `None` / scaler 溢出 / 窗口未满都不写回；`refresh_resident()` 用于 `load_checkpoint` 之后重建副本，未开启时调用报错。结构代价实测（2 层 LM、26 个参数）：cast 30 → 45（down 18 → 9、up 12 → 36），全参数 f16 驻留并没有减少 cast，省的是参数缓冲与读取带宽；设备态参数总占用从 1 份 f32 变成 1.5 份。200 步线性回归与全 f32 参考的权重相对误差 6.07e-5 / 3.27e-4，与"参数 f32、图里 cast"的 5.91 轨迹同量级。负向对照三组（去掉主机态写回、`low_precision_inputs=()`、去掉设备态写回）各自变红后还原；13 个新用例，5.126）→ sm_52 的 f16 原子加（Tesla M40 真机上 26 个 f16 用例在 NVRTC 编译期全部失败：Maxwell 没有 `atomicCAS(unsigned short*)`；`af_cas_h` 在 `__CUDA_ARCH__ < 600` 时把 CAS 展宽到对齐的 32 位字并按半字掩码比较，比较值必须由调用方传入的 `assumed` 构造——第一版用 `*base` 现读整个字，成功时返回的旧值可能不等于 `assumed`，外层重试会二次累加，`test_embedding_backward` 因此得到 32.1 / −513 / 256.5；修正后 CUDA 后端 250 个用例全过（1 跳过）、训练用例 7 个全过、真机全量 1453 个用例跳过 64，5.127）→ 阶段 C 收官：分块 selective scan（两趟——块内复合 `(prod, sum)` 与块前缀/读出，前向与反向各两个内核；末状态专用内核只写 `(tail...)`，`out` 不物化；块内递推逐字不变、块间复合改变舍入，因此是显式开关 `chunked_scan=`，默认路径逐位不变。M40 上 `T=16384, B=16, D=64` 前向 5.48 → 1.76 ms（3.11×，47.9M → 148.9M tokens/s）、反向 3.56×，f32 前向相对误差 1.16e-7、VJP 1.2～1.9e-7、8 步训练轨迹 loss 逐位一致；末状态显存 16 MB → 4 KB、decode 单步 35.56 → 29.76 µs，12 个新用例，5.128）→ HIP 上的自动图实测（`graph_mode="auto"` 这一层 5.79 只在 CUDA 上跑过，本轮在 WSL ROCm 上补齐：同图同输入只捕一个计划、换值不重捕、单节点图不进捕获、`max_graph_plans=1` 是硬上限；`execute` 口径 `transformer_block` 5.510 → 2.935 ms（1.88×）、`cnn_lenet` 3.920 → 1.847 ms（2.12×），4 个新用例，5.129）→ 批量派发（计划期把"算子是否注册 / 输入 dtype 是否受支持"判定一次并绑定内核函数，运行期成批执行不再逐节点校验，未通过判定的组原地回退、报错时机与文案不变；`transformer_block` −6.3%、`cnn_lenet` −5.8%，单算子图无收益；第 21 行的"执行器架构改造"至此清空，8 个新用例，5.130）→ 执行期分层内存（`hip/memory.py` 的 `TieredAllocator`：`<=64KiB` 按 512B 对齐分档自由表、`<=2MiB` 走 Arena best-fit、`>2MiB` 直调 `hipMalloc`；`runtime.set_allocator` 是进程级钩子，`HipBackend._memory_scope` 严格成对安装/还原并把统计写进 `_last_arena`，`memory_stats()` 因此不用新增 API；`DeviceBackend` 新增 `memory_mode="container"｜"tiered"` 参数与 `_memory_scope` 钩子，CUDA 侧显式拒绝并说明未接；小对象块用 `_PooledBlock` 持地址、不能复用 `_Buffer`（后者的 `__del__` 直接 hipFree），6 个新用例，5.131）→ 事件驱动的 SNN 稀疏算子（`sparse_linear`：`weights[K,N]` × CSR 事件 `(indices, values, offsets)` 逐段加权求和成 `[S,N]`，稠密的 `S×K` 脉冲矩阵从不物化；事件数 `E` 是符号维，`offsets` 长度必须静态（它决定输出步数），越界在设备端记 `err` 后由 `_check_ids` 报错、捕获期也能用；反向只对 `weights` 求导，`sparse_linear_weight_grad` 用原子加把 `values[e]·dout[s]` 散射回行，`indices` / `offsets` / `values` 返回 `None`——事件幅度不训练。CUDA 与 HIP 共用同一份 L2 描述，conformance 新增两条用例，9 个新用例，5.132）→ 事件驱动的 SNN 构件（`build_snn_events`：第 0 层仍是稠密 `matmul`，之后每层用 `spike_offsets` + `spike_csr` + `sparse_linear` 按事件累加，与 `build_snn_network` 的输入/参数/输出完全一致；最初把列号与起点表打包进一个 i32 张量再 `slice`，参考后端逐位正确但 HIP 拒绝 `slice` 的 int32，于是拆成两个单输出算子——不扩大后端能力面；三组深度与稠密网络逐位相等，只支持前向，训练仍走稠密路径；HIP 上量化是负结果：小规模下事件路径 0.72×～0.96×，开权重常驻后仍只有 0.96×～1.00×，瓶颈是每层多出的两趟压缩与稀疏乘的访存地板；放大到 2048×1024×1024、常驻权重时 2% 稀疏才 1.54×（收益门槛约 10%），5 个新用例，5.133）→ CSR 压缩从单线程扫描改成块内并行（`spike_offsets` 原先一个 block 一个线程扫完 `T×K`，改为单 block 两阶段：每线程跨步数若干行的非零数、结果暂存在 `offsets[t+1]`，`__syncthreads` 后由 0 号线程做 `T` 步前缀和；输出缓冲当 scratch、仍是单输出单内核，不需要新 L2 能力，也解释了为什么不铺多 block——块间没有 `__syncthreads`。HIP 两趟合计 8192×512 从 205.6 → 10.6 ms（19.4×）、8192×64 26.2 → 5.1 ms，但 32×64 只快 6%，所以 5.133 的端到端负结论不变；conformance 新增 `steps=300` 的跨步用例，参考后端 79 executed / HIP 74 executed / 5 refused / 0 failed，5.134）→ 本轮补齐的九个能力面（隐式 GEMM 卷积的 A/B 否掉了自动选择器，5.151；f16 attention 的用例与文档，5.152；任意轴归约与归一化，5.153；池化的 ceil_mode / count_include_pad / maxpool2d_indices 直通 IR，5.154；scan 按 T 与尾维自适应分块，5.155；HIP 的 pinned 暂存与设备态训练，顺带修掉广播原地加的两套折叠混用，5.156；MIOpen 对照入口，legacy 前向在 ROCm 7.2 已无 invoker 因而改走 solution API，5.157；模糊测试扩到硬件后端，随机图逐例与参考语义对拍、按契约拒绝的不算失败，5.158；捕获窗口内的 `hipFree` / `hipHostFree` 推迟到 EndCapture 之后，修掉偶发 901，5.159；归约类的 block 开放给调优面，5 轮同号门槛把 ±5% 的方向翻转拦在默认之外，5.160；逐节点 GPU 计时把"某个 kernel 变慢"从整图 wall 里拆出来，CUDA/HIP 事件对只在收尾同步一次，5.161；常量折叠改为只物化存活边界，三张图分别 9→3 / 7→1 / 5→1，同时量化出"折叠产物大且原计算便宜时会倒退"这一边界，5.161；IR 跨版本迁移在序列化结构上完成、按单出边路径补齐缺字段并记录来源版本，5.162；编译期与调优面从 CUDA / HIP 两个实现提炼成 `CodegenContract` / `TuningContract`，5.163；多设备枚举与切换（HIP 侧补上 `hipGetDeviceCount` / `hipGetDevice` / `hipSetDevice`，属性与 SM 数按设备缓存；CUDA 侧去掉写死的 0 号卡；两边提炼 `DeviceManagerContract`），5.164；分组 / depthwise 卷积的直算核（一个线程一个输出元素，不物化列矩阵、不按组分次 launch，dilation 与 bias 一并支持，f32 的 auto 路径切换过去、im2col 保留为强制对照；HIP 容器的整块赋值同时放开"连续源 → 带步长目标"，分组路径的按组写回才真正可用；dilation 由内核自己的窗口循环处理，没有复用池化的无膨胀窗口），5.165；分组卷积的性能对照（`benchmarks/grouped_conv_compare.py`，HIP 上 9 个形状直算核快 2.51×～54.91×、列矩阵 1.72～27.56 MB 全部消失），5.166；遥测包装器按存在性转发全部六项可选契约（此前漏掉 5.163 的 `kernel` 与调优五方法、5.164 的多设备三方法，包一层后 `implemented_contracts` 会静默少三项），5.167；`kernel()` 的公开面收口（CUDA 的 cupy 句柄只有 `__call__`、HIP 的只有 `launch`，真机全量第一次跑就撞上；公开入口统一返回带 `launch` 的包装句柄，同一 key 仍是同一对象），5.168；多卡真机验证（3×Tesla M40：切卡时清空 NVRTC 模块、cuBLAS handle、驻留参数与捕获计划——它们分别绑在旧 context / 旧设备 / 旧卡显存上；5 个多卡用例在 3 卡全可见时通过），5.169；CUDA 侧的执行期分层内存（复用捕获期的 `Arena` 作 cupy 分配器，`Executor.run` 显式归还设备侧输入副本与输出，`memory_stats()` 与捕获期同口径，4 个用例），5.170；CUDA 上的分组卷积对照推翻 5.166 的跨厂商结论（depthwise 4.30×～21.50×，非 depthwise 在 cuBLAS 上出现 0.44× / 0.80× 反例；`auto` 收窄为只对 depthwise 走直算核，`conv_impl="grouped"` 变成显式选项，HIP 复测 2.07×～45.77×），5.171；常量节点的设备副本随执行计划缓存（计划期上传一次、运行期直接复用，`release` 不回收、切卡与捕获轮重建；mask 用例 3.384 → 0.339 ms，修掉捕获期复用预热副本导致设备态训练 loss 从 4.37 变成 10^2 量级的缺陷），5.172；`sparse_linear` 与 cuSPARSE SpMM 的内核级对照（把 H2D 从计时里拿掉后，比 cuSPARSE 快 1.73×～2.38×、比物化稠密乘快 2.18×～6.56×），5.173；非末维归约的两维铺法（`blockDim.x` 沿 inner 合并访存、`blockDim.y` 切归约轴，`inner=1` 时 8.6×～9.6×、小输出平面 3.3×，分派判据 `_use_tiled_reduce`），5.174；运行期语义错误进模糊测试（8 条固定用例覆盖 embedding 正/负越界下标、事件类越界 `indices` 与溢出 `offsets`、整数除零必须报错、浮点除零必须给 IEEE inf；顺带修掉两处静默——负下标回绕、整数除零截断成 INT_MIN），5.175）→ 语义级索引层（`lower/indexing.py` 的 `Layout` / `offset` / `unravel_steps`，把三处手写线性化收成纯函数、生产源码逐 token 不变；只覆盖线性化不覆盖布局决策，5.177）→ 语言模型与训练状态的边界（跨块 reset 清掉带出窗口、左填充隔离、训练状态存档绑定未提交梯度 / 超参 / scaler / RNG、入口 dtype 规则、f32 损失与初始化、softplus 零点导数与注入上界、seal 后常量载荷冻结、按 token 加权累积、受约束解码根形态与生成边界、HIP 探测区分"库可加载"与"设备可枚举"，5.178～5.186）→ LM 推理路径（固定块预填充：中间块 `head="none"` 只带状态、末块 `head="last"` 只投影末位置，尾块按实际长度构图避免 pad 污染；权重常驻与 `to_host` 取行让设备上每步只剩 8 B H2D / 4 KB D2H、状态跨步 0 B；`benchmarks/lm_inference.py` 在 M40 上 1K/8K/32K 实测 TTFT 289/446/1091 ms、边际预填充 3.5K/18.4K/30.0K tok/s，5.187）→ 语言模型的整数标签损失（`lm_batch(labels="index")` 返回 `(B,T)` int32 的扁平下标、`masked_lm_loss_indexed` 把 logits 拉直后借 `embedding` 的查表语义取目标位，不新增算子；标签内存与 H2D 少 V 倍，两条路径前向逐位相等，5.188）→ 工具的 JSON Schema 约束与按注册表合成的受约束文法（`Tool` 接受标准 JSON Schema 子集并输出 `{"name","description","parameters"}`，`ToolCallGrammar` 把工具名、参数键名、参数类型与必填顺序都在解码期限死，嵌套对象参数在构造期显式拒绝并要求改用 `JsonGrammar`；20 个新用例，5.189）→ 多卡最小闭环（`train/parallel.py`：`copy_across_devices` 经主机中转、默认不做 peer-to-peer；主机侧 `allreduce` 只接受同形同 dtype 的浮点；`DataParallelTrainer` 按每卡 batch 切分、逐卡前向反向、主机侧平均梯度后更新、参数整份写回；22 个新用例），5.190）→ 语言模型的跨请求推理会话（`models/lm.py` 的 `LMRunner`：图按 `(块长, head)` 缓存并用 `OrderedDict` 封顶、`resident=True` 权重只上传一次、`close()` 只释放自己登记的那一份；解码循环与 `generate` 共用同一份 `_decode`，因此首次请求逐位一致；M40 上 1K prompt / chunk=256 / 8 token 的重复请求 35.93 ms，无状态 `generate` 387.34 ms，10.78×；7 个新用例），5.191）→ 语言模型的批量解码（`generate_batch` / `LMRunner.generate_batch`：不同长度的 prompt 按左侧补 `pad_id` 对齐、每行首个真实 token 处置 `reset=1`，行与行互不影响；每行独立采样（第 `i` 行用 `seed + i`）与独立停止，已结束的行喂 pad 并丢弃输出；批量会话的图缓存与单条共用同一套 `(块长, head)` 键；`constraint` 在批量里显式拒绝；8 个新用例），5.192 → 语言模型的会话前缀缓存（`LMRunner.prime` 预先消费前缀并把 `(state, logits)` 留在会话里，后续 `generate` 只把新增 token 送进图；新 prompt 必须是已缓存前缀的真正前缀，否则显式报错并要求 `reset()`；生成结束后前缀自动失效，`reset()` 只清前缀、保留图缓存；只支持 batch=1，批量路径不做前缀续接；8 个新用例），5.193 → 语言模型的采样面补齐（`sample_next_token` 增加 top-p 核采样与 CTRL 式重复惩罚，处理顺序固定为"重复惩罚 → 贪心判定 → top-k → 温度 → 归一化 → top-p"；`generate` / `generate_batch` / `LMRunner` 三个入口按行维护 `prompt + 已生成 token` 的历史；新增 `generate_beam` / `LMRunner.generate_beam`：k 条假设放进图的 batch 维、按父假设重排状态、EOS 进完成池不占活动槽位、`length_penalty` 归一化分数，`num_beams=1` 与贪心逐位一致，宽度 >= 词表且两步时与穷举搜索分数相等；18 个新用例），5.194 → 后端生态的算子承接面补齐（`backends/device_backend.py`：任意单轴 `argmax` 与非末维归约一样先置换到末维再复用同一个内核，`keepdims` 按原轴序恢复；多轴 `reduce_sum` / `reduce_max` / `reduce_mean` 先置换到 `[keep..., reduce...]` 再合并成一个轴，空归约轴按 numpy 语义给单位元；conformance 新增 4 条用例，参考 91 executed / CUDA 89 / HIP 89；6 个新用例），5.195 → 推理服务层的请求队列与连续批处理（`ServingEngine`：提交即入队、`step()` 以**解码步**为调度单位——先给有 logits 的槽位采样、再把空槽按 FIFO 填入等待请求并只对本次准入的做一次批量预填充、最后对需要喂 token 的槽位跑一次 batch=capacity 的 decode；槽位一空即准入，慢请求不阻塞新请求的首 token；容量 1/2/4 与逐个 `generate` 逐位一致，约束、重复惩罚、上采样面、停止都按请求独立；7 个新用例），5.196 → 中层优化的独立 pass 层（`passes/registry.py`：pass 注册表要求名字唯一且等于 `pass.name`、工厂每次给新实例；`run_pass` / `run_passes` / `build_pipeline` 三个按名入口都走 `PassManager`，单独跑也过 verifier；带参数的 pass 按实例进管线；默认与融合两条管线的名字序列收敛到 `DEFAULT_PASS_NAMES` / `FUSED_PASS_NAMES` 一处；13 个新用例），5.197 → 二阶导数（`ir/second_order.py` 的 `hessian_vector_product`：把"对反向图再求导"的名字映射固定下来，二阶图输入 = 前向输入 + 一阶种子 `d<输出别名>` + 二阶种子 `dd<wrt 名>`，输出 = 每个 `wrt` 的 `d<wrt 名>`；判据是解析二阶导 `6x·v`、一阶梯度的中心差分、跨输入块的完整 Hessian 乘向量、以及用 HVP 拼 Hessian 解 `Hp=g` 在二次目标上一步收敛；零曲率与缺少二阶规则的算子（`conv2d_*_grad`）都显式报错而不是给零；9 个新用例），5.198 → 趋势的自动采集调度（`.github/workflows/ci.yml` 加每日 cron、GPU 作业接受 `schedule` 事件；趋势历史用 `actions/cache/restore` + `cache/save` 跨 run 接续（cache 不可变，key 带 run_id、restore-keys 取最近一次），每轮追加后 `--prune 200` 按**图名**分别裁剪——全局取最近 N 条会让冷门用例的曲线直接断掉；`benchmarks.trend.prune` 的边界是 `keep` 必须是 ≥1 的整数、`--prune` 必须与 `--save` 同用且参数校验在跑基准之前；4 个新用例），5.199 → 系统化边界值测试与长上下文数值契约（`tests/test_boundaries.py` 五组 20 条：形状 / dtype / 数值 / 序列化 / 1024 步长上下文；顺带修掉两处真实缺陷——exp / log / sqrt 与全部二元逐元素算子在定义域外会往宿主打 RuntimeWarning，现统一走 `np.errstate(all="ignore")`（结果不变、日志干净）；`axis=None` 从 `TypeError: 'NoneType' object is not iterable` 改成显式 `ValueError` 并给出 `axis=tuple(range(rank))`。判据包括 f16 溢出成 inf、softmax 大值仍归一、NaN 传播、bf16 借 f32 承载、`dumps` 不修坏图，以及"分块续接的误差不随块数增长"（4/16/32 块实测 0 / 1.4e-7 / 9.3e-8）），5.200 → 跨图控制流（`ir/control.py`：`Conditional` 按数据选支、`WhileLoop` 按数据迭代，子图仍是普通 `Graph`，因此照样过 verifier 与序列化；未选中的分支不执行、循环状态必须封闭、`max_steps` 显式给出、控制区域可互相嵌套；24 个新用例），5.201 → 代码生成的第三条路（`lower/translate.py` 的跨方言源码转译：只改登记过的标识符与 `#include` 头文件，注释 / 字符串 / 未登记标识符原样透传，同方言调用是恒等；PTX 内联汇编、带掩码的 warp 内建、半精度包装类型、Clang 位转换内建一律拒收并带行号——`backends/hip/codegen.py` 的 f16 辅助函数不能靠转译得到，用例直接拿两份真实 prelude 做判据；`compile_translated` 走 `CodegenContract` 的同一个编译缓存，键含原文 sha256 与两个方言名，目标方言与后端标签不符时当场拒绝；23 个新用例），5.202 → cuSPARSE 接成 `sparse_linear` 的可选实现（`backends/cuda/cusparse.py`：ctypes 绑定 CSR×稠密 SpMM，workspace 按 `(步数, 行, 列, nnz, dtype)` 缓存、描述符每次重建；`sparse_impl` 取 `custom` / `auto` / `cusparse`，前两者在基类、第三条由 CUDA 扩展、HIP 构造期拒绝；默认仍是带越界检查的自研核，`auto` 在库缺失 / 捕获期 / f16·f64 / i64 索引时回落，显式 `cusparse` 把这些情况变成明确错误；真机因驱动掉线未验证，覆盖只放 f32 + i32、默认不切换；26 个新用例），5.203 → 事件稀疏线性的幅度梯度（`sparse_linear` 的 VJP 此前让 `values` 返回 `None`、事件幅度永远不训练；新增 `sparse_linear_values_grad`：`dvalues[e] = dout[s(e)] · W[indices[e]]`，由 `offsets` 决定事件所属步，每个事件只写一次、**不需要原子加**——与权重梯度"散射回行、重复下标要累加"正好相反；内核按**步**并行（一个线程包干该步的全部事件，逐事件做长度 `cols` 的点积），省掉"事件号 → 步号"的设备端二分；输出长度就是事件数 `E`，可以是符号维，不把稠密 `(S, K)` 脉冲矩阵造回来；`indices` / `offsets` 仍无梯度，`dout` 与 `weights` 必须同 dtype（f16 用 f32 累加）；判据是解析式、f64 中心差分、同一步里两个事件命中同一行时的互不干扰、f16 与符号维；CUDA / HIP 共用同一份 L2 描述，conformance 新增一条用例，11 个新用例），5.204 → 训练侧的两块并行能力（`ShardedOptimizer` 把 Adam / SGD 的槽位按参数名分片、每片一个独立实例，分片更新与整份优化器**逐位一致**——分片只改存储布局、不改每个参数内部的运算序列，空片合法且步数同步；`PipelineTrainer` 把整图按用户给的切点拆段、每段独占一块卡，段间激活经主机中转、micro-batch 顺序流过全部段再逆序反向，参数梯度在主机侧按 1/份数 加权累加；判据是两段 / 三段流水 N 步的参数与单卡整图在 `rtol=1e-5` 内一致，35 个新用例），5.205，5.206 → 批量解码的逐行约束（`constraint` 在 `generate_batch` / `LMRunner.generate_batch` 里也按行接受：长度等于 `spec.batch` 的序列，每行一份文法实例或 `None`，`batch=1` 时可传单个实例；共享一份、长度不符、元素缺 `allowed`/`feed`/`done` 三面显式拒绝，未完成的文法按行号报错，4 个新用例），5.207 → 反向图专用优化（`backward(minimize=True)` 沿消费者方向限定 vjp 的传播范围，无关分支不再产生反向节点且梯度逐位不变；`cse-commutative` 把 `add` / `mul` 的签名按输入 id 排序后比较，认得出 cotangent 顺序不同的同源累加；`backward_pipeline()` 是第三条管线，不含任何融合 pass——反向图要能被任何承接前向的后端直接执行；12 个新用例），5.208 → 训练态 BatchNorm 的 torch.export 导入（`aten.batch_norm(training=True)` 直接映射为 IR 的 `batch_norm`：训练态输出只依赖 batch 统计量 / weight / bias / eps，与 IR 同义；`momentum` 与 running stats 不进图，`run_decompositions()` 后的 `_native_batch_norm_legit_functional` 走同一条翻译、只留第 0 槽；入口按 `graph_signature.user_outputs` 把 buffer 更新与 save 中间量连同其生产者闭包排除在构图之外，int64 的 `num_batches_tracked` 不会漏成用户输入；3 个新用例），5.209 → 索引/布局抽象铺到生产内核（`lower/indexing.py` 的 `Layout` 从 3 处扩到 22 处：五个模块级布局常量收掉 19 处二维行主序手写线性化，`cast=None` 关掉首因子提升，57 个内核工厂的 CUDA 源码 SHA-256 与改动前**逐字节相同**；带括号前缀乘积、64 位提升与跨步不连续的形态仍手写；4 个新用例），5.210 → 复数 dtype 与 FFT（IR 新增 `c64` / `c128` 两个 dtype 与 `fft`（含 `inverse`）/ `rfft` 两个算子：变换只沿最后一维，`norm` 三档与 numpy 同义，`rfft` 只收 f32/f64 并给出 c64/c128，f16/bf16 显式拒绝；参考语义由 `numpy.fft` 给出，反向不提供 VJP（复数伴随需要共轭转置，IR 里还没有 `conj`）；`backends/cuda/cufft.py` 用 `cufftPlanMany` 做离线绑定——dtype/方向登记表、`plan_key` 与归一化缩放都是纯函数，cuFFT 两个方向都不归一化，逆变换必须显式补 1/N（或 1/√N）；CUDA 的 `capabilities()` 只在 cuFFT 可加载时声明这两个算子，HIP 没有 rocFFT、容器改为能承载复数以便 `invoke` 给出"未实现算子"而不是裸运行时错误；23 个新用例），5.211 → 流水调度与 ZeRO 通信（`PipelineTrainer(schedule=...)` 新增 `sequential` / `gpipe` / `1f1b` 三种调度：前两者是既有顺序与经典批式，1F1B 预热"段数-1"份后交替"反向一份、前向下一份"，同时驻留的段间激活从份数降到 `min(段数-1, 份数)`，用 `last_resident_peak` 记账，三者参数在 `rtol=1e-6` 内一致；`train/sharded.py` 新增 `reduce_scatter_grads`：把整份 All-Reduce 换成 Reduce-Scatter，每 rank 只接收本片（接收总量从 W·N 降到 N），累加顺序与 `allreduce_grads` 相同因而**逐位一致**，`transport` 只看到本片的名字；给空片不调用 transport，世界大小大于参数个数仍是合法状态；13 个新用例），5.212 → 审计缺陷修复（门控 SiLU 的 gate 梯度补乘上游、AMP 先反缩放再判有限性且溢出步不推进调度、累积窗口权重随存档恢复、训练状态与分片状态的"先校验后提交"、调度器 lr 与静态超参的分工、签名硬件包的信任记录落盘、归一化标量参数的梯度归约、整数入口按可表示范围校验、fuzz 整数除数非零；20 个新用例）→ 完整续训的两块凭证（`DataLoader.state_dict` / `load_state_dict` 记录轮次编号、批游标与配置，中途停止后从同一批继续且下一轮顺序与未中断逐位一致，配置不同即拒绝；`Trainer` 的前向图指纹与调度器类型 / 超参进 meta 并在提交前校验，四个内置调度器各自实现 `hyperparams()`、`Warmup` 递归到内层，14 个新用例），5.214 → 训练态 BN 的 running stats 维护（`train/bn.py`：扫描 `batch_norm` 节点、按依赖闭包重建只到 BN 输入的统计子图、每步执行一次并按 EMA 更新；`Trainer(bn_momentum=0.1)` 默认开启，方差按 `n/(n-1)` 无偏修正因而与 torch 的 eval 语义一致，`bn_running_stats()` 直接接 `bn_infer`；存档写 `bn/mean|var/<站点>` 与 `bn_momentum` 凭证并在提交前校验，`None` 是显式关闭而非静默降级；设备态与含符号通道的 BN 显式拒绝；9 个新用例），5.215 → 训练开关的组合矩阵（`tests/test_train_combinations.py`：五开关全开时第一次更新等于手算的 `initial - lr_at(0)·clip(窗口加权梯度)`；窗口一半存档续训把权重、scale、调度步数一起接回去；分段重算与整图逐位一致；f16 驻留 + f32 主副本在量化误差内；设备驻留与主机态逐位一致，并钉住"设备态 store 在 `sync_to_host()` 之前是旧值"这条既有契约；5 个新用例），5.216 → 动态形状契约（`ir/shape_contract.py`：构图 / 执行 / 自动微分 / 训练循环 / 分段重算 / 部署 / torch 导入七个环节各自声明 `full`、`per-op`、`data-only` 或 `rejected`，`support_for` 对未知环节直接报错；`require_static_shapes` 是 rejected 环节的统一守卫，训练循环与分段重算在构造期报错并带环节名、具体输入与建议；契约声明的支持面各有用例：同一张符号图被执行器按多个 batch 跑、部署包只要求权重静态而数据输入可绑定、`reshape` 的 `(B,4) → (4*B,)` 在构图期拒绝、torch 动态维（SymInt）在导入期显式拒绝而不是 `int()` 具体化；13 个新用例），5.217 → MoE 构件（`models/moe.py`：`topk_mask` 选出前 k 个专家、概率乘掩码后按行重新归一化成门控权重，所有 token 流经全部专家再按门控加权组合；负载均衡损失 `E·Σ f_e·P_e` 可微并接进训练闭环；专家并行在 `require_expert_parallel` 处显式拒绝；13 个新用例），5.218 |
| 未开始 | **逐算子即时求值的** define-by-run 执行（5.68 的 `DynamicRunner` 只解决"形状随数据变化"，图内控制流已由 5.201 的 `Conditional` / `WhileLoop` 补齐；仍缺"边算边建图、张量带值回读"的 tape 形态；构图手感由 5.25 / 5.77 的 eager 前端补齐（含 `loss.backward()`））、CI 在托管平台实跑（`.github/workflows/ci.yml` 的 gpu 作业已在真机逐条执行、五步全绿，性能基线在 5.92 重采后门禁由红转绿；仍缺一个 git 仓库与 self-hosted runner 的注册令牌）。完整对照见 1.2，差距结构与顺序见 1.3 |
| 最大未验证假设 | ~~"一套 IR 能落到不同厂商的硬件"~~ —— 已由 CUDA（NVIDIA）与 HIP（AMD）两个后端跑同一套 conformance 初步证实（5.40）；运行期计划、性能与调优对照也已补齐（5.61～5.63）。多型号已从一种变成两种（sm_75 的 RTX 2080 Ti 与 sm_52 的 Tesla M40，后者还真暴露并修掉了 f16 原子加的架构差异，见 5.127）；剩下的是跨编译器版本与更多 AMD 型号（见 1.2 与 7 的风险表） |

---

## 1. 原始规划与现状对照

### 1.1 目标大纲（五块）

> 这一节按项目大纲原文整理，是"应该做成什么"的唯一记录；1.2 逐条对照现状，
> 1.3 归纳差距的结构与推进顺序。

**一、愿景与定位**

统一深度学习基础设施：一套统一前端、可插拔后端插件；兼容市面主流硬件、成熟主流模型
架构、完整 CUDA 生态。上接 CUDA 生态存量代码与模型，下接各类算力硬件，
**一套工作流只切换后端插件就能平滑迁移**。

**二、系统架构**

1. **统一前端**：静态图与动态图统一导入；形状推导；可扩展自动微分引擎，除常规反传
   外原生支持 SNN 这类非标准反传；兼容 CUDA 生态导出的图与模型，存量模型无需大改；
   前端逻辑对所有硬件后端保持一致。
2. **多层 IR**：高层语义 IR / 中层图优化 IR / 底层硬件相关 IR 三层，规范层间转换；
   内置常量折叠、算子融合、死代码消除。**IR 是前端与后端插件之间的标准协议**：
   CUDA 算子与图结构先转统一 IR，再由插件承接、转译为硬件指令。
3. **插件化后端抽象层**：标准化插件接口，包含**设备管理器、分层内存分配器、
   算子注册与调度接口**；CUDA 为官方一等插件；新增硬件不改前端与 IR 优化逻辑，
   只需独立开发插件；硬件独有的内存层级与指令集由插件自己管，向上暴露统一接口。

**三、核心模块**

1. **编译器与代码生成**（在插件内实现）：复用成熟编译栈；各插件实现专属 codegen，
   完成内存分块与指令映射；支持"复用 CUDA 核 / 转译 / 原生生成"三种模式；
   前端不感知底层硬件差异。
2. **算子库**：制定标准算子与自定义算子开发规范；算子语义定义在统一前端 IR 层，
   高性能内核由各插件分别实现；兼容并复用 cuDNN / cuBLAS 等 CUDA 库；
   支持用底层语言开发自研内核；搭建**跨后端统一自动调优**，搜索线程与分块配置。
3. **智能体辅助开发**：代码生成智能体、性能调优智能体接口；对接统一 IR 层，
   运行时采集各插件上报的内核延迟与显存开销；借助强化学习自动调优并更新算子配置库；
   对 CUDA 与其他硬件统一做性能评估。

**四、模型适配**（全部放在前端与中层 IR，一次优化、所有后端共享）

1. **密集计算**：原生支持 CNN、Transformer；实现分块计算、注意力矩阵内存优化；
   存量 CUDA 版模型可直接迁移、复用权重，换插件即可换硬件。
2. **稀疏与事件驱动**：SNN 的离散时间步调度、稀疏事件内存池；稀疏算子在 IR 层统一
   描述，各插件提供硬件专属实现。
3. **状态空间模型**：selective scan 的底层 IR；推理时隐藏状态固定大小更新的内存策略；
   兼容主流 SSM。

**五、测试与工程**

1. **自动化测试**：边界值分析 + 模糊测试做算子精度校验；**跨后端插件**数值一致性基准，
   校验同一模型、同一输入在不同插件上的输出对齐。
2. **CI/CD**：提交后自动编译、功能测试 + 性能回归；新增插件或改动前端 / IR 时自动跑
   全量后端测试；持续监控各插件吞吐与显存，保证不破坏 CUDA 生态兼容性与其余插件的
   正确性与性能。

### 1.2 对照当前实现

图例：✅ 已交付　🟡 部分　⬜ 未开始。

**一、愿景**

| 目标 | 现状 | 说明 |
|---|---|---|
| 一套 IR、可插拔后端 | ✅ | 三个后端：参考（CPU 语义）+ CUDA（NVIDIA 真机）+ HIP（AMD 真机）。插件只需实现 `capabilities` + `execute`，另有五项可选契约（device / codegen / tuning / memory / stream）与一个分配器契约（5.89、5.163）；HIP 是照文档接入的第一个真实实例（5.40） |
| 完整 CUDA 生态 | 🟡 | 生产路径复用 cuBLAS（含卷积的 GEMM）与 cuSPARSE（5.203 起是 `sparse_linear` 的可选实现，默认不启用）；cuDNN 已在 5.69 接入为**对照物**（ctypes 直调 legacy API，只测量不执行生产图）；cuFFT 自 5.210 起接成 `fft` / `rfft` 的 CUDA 实现（`cufftPlanMany` + 显式归一化），但只有离线验证，真机未跑；cuBLASLt / cuRAND / NCCL 仍未接 |
| 一套工作流换后端 | ✅ | 同一份 conformance 在 CUDA 与 HIP 上都是 5 refused / 0 failed（95 个用例、90 executed；参考后端 95 executed / 0 refused / 0 failed；HIP 的拒绝项是既有的 f64 与分组 depthwise 形态，加上没有 rocFFT 的 `fft` / `rfft` 三条）；装配层让迁移连 import 与类名都不用改：`get_backend()` 自动挑，或 `AETHERFLOW_BACKEND=hip` 指定（5.40、5.41）。HIP 的运行期面也已与 CUDA 对齐：pinned 暂存、跨流事件依赖与 `Trainer(device_state=True)` 的 SGD / Adam 都在真机跑通（5.156） |
| 主流模型架构 | 🟡 | 六类构件都有可训练表达（5.22）：MLP ✅；Transformer ✅（单块符号化 + 多层 decoder-only LM）；CNN 🟡（conv（含分组 / depthwise）/ 转置卷积（5.51）/ max-avg-自适应-全局池化 / BN / dropout / 残差块 / 分类网络；cuDNN 对照已做（5.69），自研分块核比它慢 4～12 倍——5.34 首次量到的 9.4～18 倍里有 host 侧重生成 kernel 源码的开销，5.71 修掉后才是 4～12 倍）；RNN ✅（RNN / LSTM / GRU 时间展开）；SNN 🟡（多层 LIF + 替代梯度；稀疏事件表示与事件驱动的构件已交付（5.35、5.132、5.133），层间压缩自 5.136 起可微（`spike_linear`：脉冲梯度走普通 `matmul`，只有权重梯度走事件散射），5.204 又把 `sparse_linear` 的事件幅度梯度补成 `sparse_linear_values_grad`；剩余边界是 `dspikes` 仍是稠密 GEMM，且 HIP 上事件路径尚未达到加速门槛）；SSM 🟡（完整 Mamba block 第一版，递归收敛为 `scan_linear`；语言模型侧的跨块 reset 与左填充隔离见 5.178 / 5.179，固定块预填充 / 末位置投影 / 权重常驻的推理路径见 5.187，整数标签损失（one-hot 的 `B*T*V` → `B*T` 下标）见 5.188，跨请求复用的推理会话（图缓存 + 权重驻留）见 5.191，批量解码（左填充对齐 + 逐行独立停止）见 5.192，会话前缀缓存（`prime` 预先消费 system prompt、后续只跑增量，batch=1）见 5.193，采样面（top-p / 重复惩罚）与束搜索 `generate_beam`（k 条假设进 batch 维）见 5.194，请求级并发（请求队列 + 连续批处理，以解码步为调度单位、容量固定）见 5.196） |
| 主流硬件 | 🟡 | NVIDIA（sm_75，RTX 2080 Ti）与 AMD（gfx1100，RX 7650 GRE / ROCm 7.2）各一台，同一套 conformance 全绿（5.40）；两家还各有一次库级对照——cuDNN（5.69）与 MIOpen（5.157，自研分块核慢 4.52×～37.03×）；Intel / Apple / 移动端未接，同厂商多型号未铺开 |

**二、架构**

| 目标 | 现状 | 说明 |
|---|---|---|
| 静态图导入 | ✅ | torch.export → IR，覆盖 Transformer 与 CNN 的 aten 子集；未知算子明确报错 |
| 动态图导入 | 🟡 | 不做 tape 式 define-by-run 执行：图必须静态，才能做构图期后端能力预检、让反向图自包含可序列化、并满足 CUDA Graph 的固定路径要求。补的是**构图手感**——`EagerGraph` / `Tensor` 用运算符直接写图，产物仍是同一份静态 IR（5.25）；数据相关的分支与循环由 `Conditional` / `WhileLoop` 显式表达，控制流是子图之间的关系、子图仍是普通静态图（5.201）；逐算子即时求值仍未做 |
| 形状推导（含符号维） | ✅ | 编译期推导 + 运行期回校；空间维可以是 `floor((S + 2p - k) / s) + 1` 形式的派生表达式，一次构图多分辨率执行（5.53） |
| 自动微分（含 SNN 非标准反传） | 🟡 | 语义 IR 层反向展开：40 条 VJP 规则，反向图仍是可序列化、可执行的普通 PORTABLE 图（5.17）；eager 前端可直接 `loss.backward()`，但它仍是这张图而不是 tape（5.77）；激活检查点由 `train/recompute.py` 的分段重算补上（5.75）；二阶导数由 `hessian_vector_product` 补上（对反向图再求导，判据是解析值 / 中心差分 / 二次目标一步 Newton，5.198），但只对具备 VJP 的算子成立；反向图专用优化自 5.207 起有两条：`backward(minimize=True)` 按求导目标限定 vjp 范围（无关分支不再建反向节点，梯度逐位不变），`backward_pipeline()` 用交换律 CSE + DCE 且不含融合；SNN 替代梯度只需替换算子 `vjp`，路已铺好 |
| 前端逻辑对所有后端一致 | ✅ | 执行骨架统一符号绑定与形状校验 |
| 高层语义 IR | ✅ | 类型系统、verifier、可移植性标注 |
| 中层优化 IR | ✅ | 独立一层：`passes/` 有八个可组合 pass（DCE / CSE / 交换律 CSE / simplify / 常量折叠 / 三个融合）、`PassManager` 在每个 pass 之后过 verifier、`registry.py` 提供注册表与按名入口（`run_pass` / `run_passes` / `build_pipeline`），三条管线（默认 / 融合 / 反向图）的名字序列只在 `DEFAULT_PASS_NAMES` / `FUSED_PASS_NAMES` / `BACKWARD_PASS_NAMES` 定义一处（5.36、5.197、5.207） |
| 底层硬件 IR | 🟡 | `aetherflow/lower` 的**语法级 L2**（29 个表达式 / 语句节点 + C 语法打印器 + 方言数据）已接管**全部**内核模板的生产路径：9 份字符串模板逐份迁完，CUDA 与 HIP 共用同一份 `Kernel` 描述，与手写模板逐 token 相等（5.42–5.49）；归约族补到 max / mean / argmax 后仍是同一份骨架（5.50）。**语义级 L2** 已有第一版（5.52）：同形相邻逐元素算子融成一条链、中间值不物化，f32 在长度与对齐条件满足时走 `float4` 四通道；索引 / 布局抽象自 5.177 起有 `Layout`，5.209 把它铺到生产内核的 19 处二维行主序线性化（源码逐字节不变），但带括号前缀乘积、64 位提升与跨步不连续的形态仍手写；block 的取值已改为按规模推导的纯函数（5.65，归约类与分块卷积不在其内） |
| 常量折叠 | ✅ | `ConstantFolding` 求值"输入全是常量"的子图，结果写成 `constant` 的 `data` 属性；符号维、带种子的随机算子与超大常量不折（5.36） |
| 算子融合 | 🟡 | 3 条规则：softmax 吸收缩放、LayerNorm 吸收残差、SwiGLU 门控——全部是 ADR-010 的"已有算子输入扩展"，不新增融合算子（5.36）。pre-norm Block 的残差 add fanout=2，因此不命中。另有后端侧的逐元素链融合（5.52）：相邻同形算子合成一个内核，中间值不物化 |
| 死代码消除 | ✅ | `passes/dce.py` |
| IR 作为标准协议 | ✅ | 类型 + verifier + JSON 序列化 + 版本治理 + 跨版本迁移（结构化改写、单出边路径、`Graph.migrated_from`，5.162）+ conformance 准入 |
| 设备管理器接口 | ✅ | `device_info()` 为可选契约 `DeviceContract`（5.10）；stream 管理已提为独立契约 `StreamContract`（5.89，CUDA 与 HIP 都实现）；多设备枚举与切换在 5.164 补齐：HIP 的 `runtime` 绑上 `hipGetDeviceCount` / `hipGetDevice` / `hipSetDevice`，属性与 SM 数按设备索引缓存，CUDA 侧去掉写死的 0 号卡，两边共同实现 `DeviceManagerContract`（`enumerate_devices` / `current_device` / `set_device`，非法索引当场拒绝）；跨卡拷贝与数据并行最小闭环在 5.190 补齐（`copy_across_devices` 经主机中转、`allreduce` 主机侧归约、`DataParallelTrainer` 逐卡前向反向）；5.205 补了优化器状态分片（ZeRO-1，`ShardedOptimizer` 与整份优化器逐位一致）与 GPipe 式流水（`PipelineTrainer`：段间激活跨卡、micro-batch 顺序流转、参数梯度按份数加权），仍无 peer-to-peer、也没有前向与反向的重叠执行 |
| 分层内存分配器接口 | 🟡 | `Arena` 的统计为可选契约 `MemoryContract`（5.10），分配器本身按 `AllocatorContract` 提炼并由 `MemoryContract.allocator()` 交出（5.89）；`plan_memory` 提供不执行、不需要 GPU 的编译期静态预算，其 `release` 参数与 `Executor.run_device` 的执行策略一一对应（5.36 / 5.105）。执行期分层通路已在两家都接通：HIP 是 `TieredAllocator`（<=64KiB 分档自由表 / <=2MiB Arena / >2MiB 直调，5.131），CUDA 复用 `cuda/memory.py::Arena` 并在 `execute` 期间成对安装 / 还原 cupy 分配器（5.170），两边都由 `memory_mode="tiered"` 显式打开、默认仍是 `container` |
| 算子注册与调度接口 | ✅ | `KernelRegistry`：注册即声明、查表即调度（5.11）；同名多实现按 `cost` 选择（5.30）；插件算子扩展面 `register_operator` / `unregister_operator`（5.30） |
| CUDA 一等插件 | ✅ | 57 个算子语义对齐（性能见 5.4；卷积见 5.12，池化见 5.13；反向图见 5.17；SNN 见 5.18；scan 见 5.19 / 5.35；新增五算子、两个反向算子、`pad` 与三个卷积/池化反向内核见 5.20；随机算子见 5.23；批归一化见 5.24；融合注意力见 5.32；f16 支持面见 5.33；分组卷积与自适应池化见 5.34；线性递归前向与伴随见 5.35；LayerNorm 残差与 SwiGLU 门控两个内核见 5.36；`erf` 见 5.39；任意轴归约 / softmax / 归一化见 5.153（非末维归约的两维铺法见 5.174）；池化三属性与 maxpool 下标见 5.154；scan 自适应分块见 5.155；隐式卷积仅作显式对照路径见 5.151；`spike_csr` 的块内扫描见 5.176；任意单轴 argmax 与多轴归约见 5.195；事件稀疏线性的两侧梯度见 5.132 / 5.204） |
| 新硬件只写插件 | ✅ | 契约分层 + 算子注册表 + 插件开发指南（5.30）齐备；HIP 后端是第一个真实实例——全部新增代码在 `backends/hip/` 一个包里，IR / verifier / 前端 / 优化 / 契约零改动，内核模板逐字复用（5.40）。第三方硬件包还能用 pip entry point（组 `aetherflow.backends`）注册，装包即被 `get_backend` 发现（5.41）；zip 包还能声明内容摘要与依赖，装包期就拒绝被改动或依赖不满足的包（5.64） |

**三、核心模块**

| 目标 | 现状 | 说明 |
|---|---|---|
| 插件内 codegen | ✅ | NVRTC 运行时编译（CUDA 侧）与 hipRTC 运行时编译（HIP 侧），目标机都不需要厂商 toolkit；公开入口 `DeviceBackend.kernel()` 已提炼为可选契约 `CodegenContract`（5.163） |
| 内存分块 / 指令映射 | 🟡 | conv2d 有真正的分块（权重复用 + 寄存器复用 + 边界快速路径，见 5.16）；其余算子仍只有广播维折叠与 block size 选择。分块与布局的**决策**已在 5.54 收进 `lower/plan.py`（block / grid / vec / shared / OC tile 全部是纯函数并带约束校验），一维网格步进骨架已在 5.55 统一（`kernels.grid_stride`），多维索引在 5.57 逐字比对后确认只值得合并一处，其余按语义差异保持手写（5.177 起 `_split_output_index` / `_image_slice` / `_tap` 三处线性化收进 `lower/indexing.py` 的 `Layout`，生产源码逐 token 不变；布局决策仍留在各内核）；block 的取值在 5.65 变成"按规模推导"的纯函数（只有上限与下限两条规则，不含性能结论） |
| 复用 / 转译 / 原生生成三模式 | ✅ | 三条都通：**原生生成**是 57 个算子的生产路径（L1 → L2 → 方言 → NVRTC / hipRTC），**复用**是 cuBLAS / cuDNN / cuSPARSE / hipBLAS / MIOpen 的库调用（cuSPARSE 自 5.203 起接管 `sparse_linear` 的可选路径），**转译**是 `lower/translate.py` 把别人写好的源码换成另一家方言的源码（5.202）。转译只做逐 token 可证的改写：注释 / 字符串 / 未登记标识符原样、命中拒收项（PTX 内联汇编、带掩码的 warp 内建、半精度包装类型、Clang 位转换内建）当场报错并带行号；`compile_translated` 与原生生成共用同一个编译缓存，键含原文 sha256 |
| 标准算子 / 自定义算子规范 | ✅ | `Op` 数据结构 + attrs schema + `register_operator` 命名空间护栏（5.30）；插件算子进 `OPS` 后立刻能被参考后端执行与 conformance 对拍 |
| 语义在 IR 层、内核在各插件 | ✅ | 参考后端承接全部 59 个算子的语义（`fft` / `rfft` 走 `numpy.fft`）；CUDA 承接 57 个自研内核 + cuFFT 的 `fft` / `rfft`（conv2d = 分块直算 / im2col + cuBLAS 两条生产路径，隐式 GEMM 只作显式对照、不进自动选择，5.151；分组 / depthwise 自 5.165 起有专用直算核，5.171 按真机数据把 `auto` 收窄为"只有 depthwise 走直算核"——CUDA 上非 depthwise 的 `g4` / `g2` 被 im2col + cuBLAS 反超（0.80× / 0.44×），HIP 上仍是正收益（2.07×），不能跨厂商外推；`conv_impl="grouped"` 可显式强制，5.34 的"按组切片复用 im2col"保留在 `conv_impl="im2col"` 下；maxpool2d = 自研索引 kernel，adaptive_avg_pool2d = 自研窗口 kernel，embedding 与池化/卷积反向的重叠写用原子加，attention = 不物化分数矩阵的分块 online softmax，scan_linear / scan_linear_backward = 一个尾元素一个线程、沿时间串行，长 T（>=512）且尾维 <=2048 时自动分块，见 5.35 / 5.155；归约与 softmax 的任意轴见 5.153；池化的 ceil_mode / count_include_pad / maxpool 下标见 5.154；融合形态也在这里：layernorm 的第 4 个 residual 输入、silu 的第 2 个 gate 输入，见 5.36）。dtype 上 f32 全覆盖，f16 也全覆盖 57 个（attention 自 5.121 起是混合精度：q/k/v/mask/out 存半精度，softmax 的行统计与输出累加仍在 float——f16 只省存储与带宽，不参与累加，5.152 补了跨块边界的用例）。HIP 承接同一批 57 个算子的方式不是复制：内核源码逐字复用上面这些模板，只换 f16 的存取与原子加（5.40） |
| 复用 cuDNN / cuBLAS / cuSPARSE | ✅ | cuBLAS 是 matmul 的生产路径；cuDNN 在 5.74 接成 `conv_impl="auto"` 的可回落快路径（`groups=1` 的 f32/f16，缺库或形状不支持就静默回到自研核），显式 `conv_impl="cudnn"` 可强制并在缺库时报能力错误。本轮真机没有 libcudnn，cuDNN 数值对拍按设计跳过，5.69 的对照数字仍然有效。cuSPARSE 自 5.203 起是 `sparse_linear` 的可选实现（`sparse_impl="custom"｜"auto"｜"cusparse"`，默认 custom；覆盖 f32 值 + i32 索引），本轮因真机驱动掉线没有数值对拍，库路径没有设为默认 |
| 自研底层内核 | ✅ | 逐元素（含 gelu、SwiGLU 门控）、归约（含任意单轴；非末维的两维铺法见 5.174）与 softmax（含任意单轴）、layernorm（含残差形态）/ rmsnorm、batch_norm、im2col、conv2d_tiled、分组 / depthwise 直算、隐式卷积（仅显式对照）、max/avgpool（含 ceil_mode 与 padded-边界分母）、maxpool 下标、adaptive_avg_pool2d、concat / pad copy、embedding gather / scatter、maxpool 与卷积的输入/权重梯度、SplitMix64 掩码、分块 attention（online softmax）、线性递归扫描（前向 + 伴随 + 自适应分块）、脉冲压缩（`spike_offsets` 前缀和 + `spike_csr` 块内扫描，见 5.176）、任意单轴 argmax 与多轴归约（5.195）、事件稀疏线性的两侧梯度（5.132 / 5.204） |
| 跨后端统一自动调优 | 🟡 | `aetherflow/tuning/` 提供与后端无关的参数空间、搜索与配置库（5.37）；CUDA 侧开放 17 个逐元素算子的 block 与 conv2d 分块核的 `tile_w`（5.56）；归约类在 5.160 也开放了（此前按"shared 与 block 绑定"排除，核对内核后该前提不成立——暂存区是固定 4 KB、内核沿 `blockDim.x` 跨步读），`tunable_ops()` 里除 conv2d 外全部是 block 旋钮。HIP 侧继承同一套调优面（5.58 补齐了漏写的 `_conv_tile`，并加了构造字段对账测试）；遥测层 `InstrumentedBackend` 本就与后端无关，5.59 又把三个基准 CLI 的 `--backend` 从写死的 `ref/cuda` 改成查注册表，5.60 已在 WSL 的 ROCm（RX 7650 GRE / DXG 直通）上实跑：conformance 13/13、调优搜出 9 条配置并逐位对拍一致；分块参数虽已在 5.54 集中到 `lower/plan.py`，`tile_w`（5.56）与 `roc` / `cb`（5.74）三个旋钮都已开放；block 的"自动"值本身也是一个候选（`{"block": None}` 表示撤掉覆盖、回到按形状推导，5.65）。5.140 把测量改成每轮连测、逐轮反转的交替 A/B，输出配对差的 bootstrap 区间与"两半同向"检查：HIP 三次采集里逐元素 block 全部保留默认，conv2d 却给出三个不同赢家（1.20× / 1.00× / 1.09×）——区间能给出，赢家仍不可复现。5.141 又加了 `--order shuffle`（每轮固定种子随机排列）并做 alternate/shuffle 的 A/B 对照：7 次采集里 conv2d 4 次判收益、3 次判噪声，4 次赢家互不相同；"批次不稳"候选数随整机状态在 12～41/45 间波动，与排列无关。根因是轮内跨度（conv2d 一轮约 2.3 秒）；5.142 已把跨度接成拒绝下结论的门槛（`--max-round-seconds`），一次 HIP 采集里 conv2d 44/45 个候选被降级为 `long-round`，同时暴露"首轮显著慢于后续轮次"的冷启动偏置。5.143 又加了 `--warmup-rounds`（默认 0）把预热轮移出样本与跨度：预热耗时与 5.142 的首轮同量级，正式轮全部回到稳态；dropout 不再被跨度误伤，但 5.142 的两个"收益"（`block=512` / 256）随之消失，说明它们更可能来自首轮污染。5.144 再做两阶段搜索（`--staged`）：conv2d 从 45 个候选拆成 4 + 11 + 2，单轮跨度降到 0.21 / 0.65 秒，`long-round` 归零；但阶段 2 的 10/11 个候选两半不稳，赢家仍保留默认——剩下的瓶颈是整机漂移下的 8 轮样本量，不是跨度。5.145 把轮数加到 16：不稳数减半（10/11 → 5/11），但仍无可接受赢家，8 轮的唯一收益（1.08×）消失，且该候选两半差达 +13.18% / -12.07%（采集内整机漂移约 25%），故不采用加轮数；下一步是跨采集复现或控制环境。5.146 已实现跨采集复现（`--repeats`）：HIP 上两次独立采集各判出不同赢家（add_broadcast 的 `block=512` 1.0756× 与 `1024` 1.1316×、relu 的默认与 `128` 1.0346×），逐候选合并后全部判 `unreplicated` 并保留默认，conv2d 两次一致地 44 个 `long-round`。它拦下了"两次都显著但赢家不同"的假阳性，代价是采集时间翻倍；控制环境仍是根因方向。5.147 在 CUDA 真机（Tesla M40）补齐同一实验：9 个用例全部保留默认，两次各一次 `decided=True`（sigmoid 的 `512` 1.0144×、silu_gate 的 `128` 1.0139×）合并后都判 `unreplicated`；同一套 1.0 秒跨度门槛在 M40 上不触发（conv2d 最长轮 0.874s），而 HIP 上 44/45 个候选触发（2.68s），说明绝对秒数门槛跨机不可移植。5.148 把门槛改成相对量（`--max-round-factor`，以预热轮平均耗时为标尺）：HIP 上 conv2d 的 `long-round` 归零；单次采集判出的 3 个收益（add_same_1m 5.50%、add_broadcast 4.13%、conv2d 6.66%）叠加 `--repeats 2` 后全部 `unreplicated`。5.149 又在跨采集复现上加多数票（`--vote majority`，需 3 次以上）：HIP 三次采集里 8 个逐元素用例仍全部保留默认（多数候选的 2/3 票型是 `unstable`），conv2d 的 `cb=2 / tile_w=64` 以 2/3 票（2 faster + 1 noise、无 slower）转正，配对差 -11.75%（CI -14.20～-2.33，1.037×），而同一批采集按全票合并时 9 个用例全部保留默认（conv2d 42/45 个候选 `unreplicated`）——多数票把"全票过严"换成"少数候选可通过"，但通过者的区间跨度 11.9 个百分点，只能当弱证据。5.150 给每个候选补了逐次配对差（`replicate_diffs`，与 `replicates` 对齐、不可行记 `None`），并把 `--repeats` 加到 5：HIP 上 9 个用例仍全部保留默认，45 次"用例 × 采集"里 8 次单次判出赢家但**两两不同**（conv2d 三次三个赢家：`64/4/1` 1.076×、`64/2/2` 1.098×、`16/auto/1` 1.130×）；5.149 的 2/3 转正候选在 5 次里拿到 0 张 `faster`，逐次差 `+2.95 / +0.27 / +5.34 / -2.20 / -8.05`（中位 +0.27%），方向本身在采集间翻转——那次转正只是"赢家每次都在换"的一个切片；M40（5 次多数票，驱动 535.309.01 / CUDA 12.2 / cupy-cuda12x 14.2.0）同样 9 个用例全部保留默认、425 次判定只有 4 张 `faster`，同一转正候选逐次差 `+0.49 / +0.04 / +1.71 / +0.42 / +0.92`（5 次全为正、0 张更快），conv2d 另有 21 个候选 5/5 全票更慢（`64/2/2` 逐次 +13.2%～+19.2%）——两台机器一致：瓶颈是整机漂移与 1% 地板。5.160 再补两道：归约类 block 开放，以及 `--min-same-sign N`（逐轮核对方向，同号不足把 `faster` / `slower` 降为 `unstable`）。HIP 两次采集里 softmax / reduce_sum / layernorm 三个用例全部保留默认：单次最优差在 −4.5%～+5.3% 之间，但同号只有 3～4/5，或 5/5 更慢；两次之间方向还会翻转（reduce_sum 的 `128` 一次 −0.9%、一次 +5.3%），与 5.63 的"±3% 真实差异 + 整机漂移"一致。同号门槛把"4/5 同号 + 4.5% 收益"这类弱证据直接拦下 |
| 智能体接口 | ✅ | `agent_api` 的只读查询面：后端画像、算子变体、调优空间、性能摘要、配置推荐，全部可 JSON 序列化（5.37）。明确不执行图、不接受代码、不写配置库 |
| 运行时数据上报 + RL 更新配置库 | 🟡 | `TelemetryRecorder` 产出结构化 `RunRecord`（整图 wall、逐节点派发耗时、设备、显存快照、吞吐），可 JSON 落盘（5.37）；配置库的更新是"实测更快的配置覆盖旧记录"，**不含 RL 训练**——没有策略网络、奖励信号或在线探索。5.167 补齐包装器的契约转发：`InstrumentedBackend` 按内层是否存在逐项绑定 `device` / `device_manager` / `codegen` / `tuning` / `memory` / `stream` 六项，包装后 `implemented_contracts` 与内层完全一致（此前漏掉 5.163 的 `kernel` 与调优五方法、5.164 的多设备三方法） |

**四、模型适配**

| 目标 | 现状 | 说明 |
|---|---|---|
| Transformer | ✅ | 24 节点 Block 端到端，参考后端与 CUDA 双向验证 |
| CNN | 🟡 | conv（含分组 / depthwise）/ max-avg-自适应池化 / flatten 全部可从 torch.export 导入，LeNet 端到端 torch / 参考 / CUDA 三方对齐（5.14）；分块卷积已交付（5.16），分组与自适应池化见 5.34，cuDNN 对照见 5.69 |
| 分块计算 / 注意力内存优化 | ✅ | 注意力收成一个 IR 节点（5.29）；CUDA 侧是分块 online softmax，分数矩阵从不落盘（5.32）。参考实现仍物化，因为它只负责定义语义 |
| 存量 CUDA 模型零改写迁移 | 🟡 | 推理侧已实证：resnet18 / mobilenet_v2 / efficientnet_b0 / squeezenet1_0 / shufflenet_v2 / densenet121 / vit_b_16 **7 个 torchvision 模型**导出后在参考后端与 torch eager 逐点一致，其中 resnet18 与 squeezenet1_0 还在真机 CUDA 上对拍（5.39）；训练态 BN 的导入口已开（5.208，含 `run_decompositions()` 后的形态），训练图的覆盖范围从"构图层自建"延伸到"torch.export 直出"；仍明确拒绝的是 `get_attr` 常量、带未消费参数的算子变体与三输出 `_native_batch_norm_legit` 原语 |
| SNN 离散时间步 + 事件内存池 | 🟡 | 离散 LIF + 时间步展开已跑通，替代梯度可训练（5.18）；脉冲的稀疏事件表示（CSR 变体）、事件内存池与只读事件的发放率已在数据侧交付（5.35）；压缩的第二趟 `spike_csr` 自 5.176 起是"一块一行 + 段内块内前缀和"（`steps=1` 2.2×、64 行 1.9×，行数 ≥ `sm_count × 32` 时判据 `plan_csr_scan_enabled` 仍回串行，因为那时一路线程沿行内串行反而更快），旧铺法保留为 `spike_csr_serial` 只作基准对照；事件驱动的稀疏内核自 5.132 起是 `sparse_linear`（权重梯度 + 5.204 的事件幅度梯度，两者都在事件形态、长度 `E`），但 `spike_linear` 的 `dspikes` 仍是稠密 GEMM，"事件驱动"在训练里省的是权重与幅度梯度而不是 `dspikes`；输入形状要求静态 |
| SSM selective scan | ✅ | Mamba 的 S6 第一版：只把线性递归做成 `scan_linear`（5.19）；离散化 / 读出 / 跳连都由已有算子组合；CUDA 已承接前向与伴随（5.35），输入仍要求静态形状 |
| 一次优化所有后端共享 | ✅ | pass 只在 IR 层，与后端无关（融合受能力协商约束，5.36）；静态显存预算同样只读 IR，不执行、不需要 GPU |

**五、测试与工程**

| 目标 | 现状 | 说明 |
|---|---|---|
| 边界值测试 | 🟡 | 有严格性用例（f64 拒绝、conv 的 stride/pad/dilation 边界、池化的 ceil_mode × count_include_pad 组合、scan 的三态取值等；非最后维不再拒绝，5.153），5.200 把边界值收成系统化用例集（`tests/test_boundaries.py` 五组 20 条：形状 / dtype / 数值 / 序列化 / 长上下文，并把 `log(0)` 等宿主告警与 `axis=None` 的 `TypeError` 两处真实缺陷一并修掉）；5.181 把图输入 dtype 的三条静默转换（浮点→整数截断、窄化越界回绕、布尔当整数）变成显式报错，5.182 补了 f16 全 padding 批的 0/0 NaN |
| 模糊测试 | ✅ | `aetherflow/fuzz.py`：按种子随机构图，检查校验 / 序列化往返 / 两次执行一致 / 默认优化管线前后逐位一致；每轮再注入一条非法属性，要求构图期明确拒绝（5.38）。5.139 把随机生成从 25 个算子扩到 31 个：卷积 / 池化 / attention / scan / embedding 也进随机图（属性先算到自洽，再在合法区间内取值）；但随机图仍只跑参考后端，符号维不随机。5.158 起可传 `backends=("cuda", "hip")` 让每张随机图再与硬件后端的输出按 1e-4 对拍：HIP 上 100 个种子 81 例对拍、19 例按契约拒绝（全部是 i32 未承接），符号维仍不随机。5.175 补上运行期这一层：8 条固定构造的"图合法、数据非法"用例（embedding 正/负越界下标、事件类越界 `indices` 与溢出 `offsets`、整数除零必须报错、浮点除零必须给 IEEE inf），失败与构图期共用同一通道、报告分开计数 |
| 跨后端数值一致性基准 | ✅ | `run_conformance`：92 用例 + 5 条检查（5.9，用例集随 5.17 / 5.18 / 5.19 / 5.29 / 5.34 / 5.39 / 5.50 / 5.51 / 5.132～5.134 / 5.153～5.155 / 5.174 / 5.195 / 5.204 扩增）；本轮实测参考后端 92 executed / 0 refused / 0 failed、HIP 90 / 2 / 0（拒绝项不变：f64 与分组 depthwise 的既有形态）；CUDA 真机本轮未复跑（见 §7 的 5.203 条） |
| 跨**硬件**一致性 | 🟡 | NVIDIA sm_75 与 AMD gfx1100 两个厂商各一台，同一套 conformance 逐项对齐（5.40）；仍只有各一张卡，同厂商多型号 / 多驱动版本未验证 |
| CI 流水线 | 🟡 | `.github/workflows/ci.yml`：`cpu` 跑正确性（Python 3.10 / 3.11），`fuzz` 每次 push 跑 200 个随机种子，`gpu` 手动触发或**每日定时**、串起 verify_cuda → 全量测试 → conformance → 回归 → 调优往返 → 趋势采集（历史用 `actions/cache` 跨 run 接续、`--prune 200` 按图名裁剪）并上传产物（5.38、5.199）；**仍未在托管平台实跑** |
| 性能回归门禁 | 🟡 | 10 个固定用例（每例 3 轮取最好）+ 同机基线 + 25% 阈值（5.15）；基线在 5.92 重采，带宿主指纹，真机上门禁已绿；`ci.yml` 的 GPU job 仍未在托管平台的 self-hosted runner 上跑过 |
| 吞吐 / 显存持续监控 | ✅ | `TelemetryRecorder.trend` 按图分组聚合 wall / 显存峰值 / 吞吐的首末、极值、中位数、变化率与逐次斜率；`benchmarks/trend.py` 支持 `--save` / `--compare` 跨运行累积（5.38），并新增 `--prune` 按图名裁剪（5.199）；采集调度由 CI 的每日 cron 与 `actions/cache` 跨 run 接续负责（5.199）。看板与告警仍不做——输出是 JSON / 表格，展示层不在这里 |

### 1.3 差距的结构：三堵承重墙

上表有 20 余项缺口，但按"是否卡住其它所有工作"排，其实集中在三处：

1. **插件还不是插件。** 大纲要求接口含设备管理器、分层内存分配器、算子注册与调度，
   现在只有两个方法。真接一个新硬件，第一件事就是发现设备、流、内存层级没有位置，
   只能改接口；改完 conformance 全部重跑。这一项**不依赖新硬件**，却决定了后面
   每一项的成本，越晚拆越贵。
2. **核心命题已被初步实证。** "一套 IR 落到不同硬件"现在有 NVIDIA（CUDA）与 AMD
   （HIP/ROCm）两家跑过同一套 conformance，数字逐项一致（5.40）。这一步暴露的
   不是 IR 漏洞，而是厂商编译器差异（f16 存取、无符号数原子加）——说明承重墙在
   内核方言层，不在 IR 层。剩下的成本在于：把同样的工作复制到第三家（HIP 侧的
   运行期计划与性能对照已由 5.61、5.62 补齐）。
3. **模型架构只覆盖一半。** Transformer 完整；CNN 的 cuDNN 对照已补（5.69，自研分块核慢 4～12 倍）；SNN / SSM 各有第一版。

三处表述需要顺手修正，避免对外夸大：

- **"底层硬件 IR"目前不存在独立一层**，只有插件内部拼出来的源码字符串；HIP 复用的
  正是这同一份字符串模板，因此"两家厂商共用一套 CUDA C"这个事实本身就是把它
  提升为独立一层的理由（5.40）。
- **"跨后端自动调优"与"智能体"已有第一版**（5.37）：参数空间、网格搜索、配置库与
  只读查询面都有了；但只有 CUDA 一个后端可实测，归约 / 卷积类 kernel 尚未开放调优，
  也没有强化学习式的在线更新。不宣称"跨后端"已成立。
- ~~**"分层内存分配器"只在 CUDA 捕获期有效**~~ —— 分配器本身已提为 `AllocatorContract`
  并由 `MemoryContract.allocator()` 交出，CUDA 与 HIP 两个 Arena 都满足（5.89）。
  **执行期的分层通路也已接通**：HIP 的 `TieredAllocator`（5.131）与 CUDA 复用
  `Arena` 的执行期安装（5.170）都由 `memory_mode="tiered"` 打开，`execute` 结束
  即还块并把统计写进 `memory_stats()`。仍保留的边界是默认值：两家的默认仍是
  `container`，分层要显式选择（CUDA 的 4 条真机用例也只验证了数值与成对还原）。

推进顺序（理由见上）：

| 顺序 | 事项 | 为什么排这里 |
|---|---|---|
| 1 | 拆插件契约 + 插件开发指南 | 承重墙，不依赖硬件；越晚做，后面每项都要为旧接口付利息 |
| 2 | CNN（conv / pool / 分块卷积 / cuDNN） | 补齐"主流模型架构"的另一半，也是底层 IR 与真 kernel 工程的入口 |
| 3 | CI + 性能回归基线 | 大纲明确要求，成本最低；之后每加一个后端都靠它兜底 |
| 4 | 自动微分 | 训练与 SNN 的前提，工作量最大，会反向影响 IR 设计 |
| 5 | 自动调优 + 智能体、SSM、SNN | 需要两个以上后端与真实性能压力才有可调对象 |

**进度**：第 1 项已完成——契约分层（5.10）、算子注册表（5.11）、多实现按代价选择与
插件开发指南（5.30）都已就位。HIP 后端（5.40）照这份清单接入了第一个非 NVIDIA 硬件，
全程没有改 IR / verifier / 前端 / 优化 / 契约一行代码；第 2 堵墙也因此拆掉。第 2 项进行中：conv2d（5.12，分块直算 / im2col + cuBLAS）、maxpool2d（5.13，自研索引
kernel）、前端 conv / pool / flatten 导入（5.14，LeNet 三方对齐）与分块卷积 codegen
（5.16）都已跑通，cuDNN 对照也已在 5.69 补上（自研分块核慢 4～12 倍）。第 3 项已落地第一版：10 个固定用例 + 两份同机基线 +
25% 判定 + CI 配置（5.15、M3.5）；GPU job 尚未接上真实 self-hosted runner（作业脚本已在真机逐条执行验证，5.70；性能门禁因换宿主变红，5.92 重采基线后已转绿）。
跨运行吞吐 / 显存趋势已可作为 CLI 显式采集（5.38）。
第 4 项已完成第一版：`backward()` 在语义 IR 层展开，37 条 VJP 规则，反向图可序列化、
可执行、可被任意后端承接（5.17）；CUDA 侧 MLP / layernorm / softmax 反向图已真机对拍，
卷积与池化的反向在 CUDA 上明确拒绝。二阶导数只对具备 VJP 的算子成立（5.198）；激活检查点已由 5.75 的分段重算补上
（不改 IR / 执行器，链式模型梯度与整图逐位一致）。eager 前端的 `loss.backward()` 也已在
5.77 补上，但它仍是"建图 + 执行"，不是 define-by-run 的 tape。
第 5 项先做了不依赖第二个后端的两项：SNN 把离散 LIF 的时间步展开进普通图，`spike` 算子
用替代梯度让发放可训练（5.18）；SSM 把 selective scan 的线性递归收敛成 `scan_linear`，
离散化 / 读出 / 跳连全部由已有算子组合（5.19）。稀疏事件表示与事件内存池已在数据侧交付
（5.35）；编译层的三项（常量折叠、深层融合、静态显存预算）也已补齐（5.36）。
第 5 项剩下的自动调优与智能体已落地第一版（5.37）：参数空间 + 网格搜索 + 按设备/形状
命中的配置库，运行时结构化上报，以及只读的智能体查询面。第二个实现（HIP，5.40）已经
出现，但**跨后端调优**尚未开始：HIP 侧的调优面靠继承就已具备（5.58 修掉了漏写的一个
字段，并把两份构造的字段对账做成了测试），遥测层也本就与后端无关、基准 CLI 的后端选项
已改为查注册表（5.59），并且已在 WSL 的 ROCm 上实跑通过（5.60：conformance 13/13，
调优搜出 9 条配置并逐位对拍一致）；CUDA 侧则只开放逐元素算子的 block 与 conv2d 的
tile_w。两侧都能出数据了，"同一个旋钮在两个厂商上各自最优值是否相同"才成为可回答的问题。
测试与工程这一项补上了模糊测试与趋势聚合（5.38）：`fuzz.py` 用随机图验证"校验通过 /
序列化往返 / 两次执行一致 / 优化前后逐位一致 / 非法属性被拒"五条不变量，`ci.yml` 增加
每次 push 的 fuzz 作业并补全 GPU 作业的步骤（verify_cuda、调优往返、趋势产物）；
`TelemetryRecorder.trend` 与 `benchmarks/trend.py` 提供跨运行的吞吐 / 显存聚合。
GPU 作业仍未接上真实 runner。

---

## 2. 架构决策记录（ADR）

按影响面排序，每条都写清"为什么"和"代价"。

### ADR-001　IR 是前后端之间唯一的契约

- **决策**：后端插件只能看到 IR，不得读取任何前端对象；前端不得感知任何后端。
- **理由**：这是"一套工作流切换后端"能够成立的唯一方式；也让后端可以独立开发、独立发版。
- **代价**：任何跨层便捷通道都被禁止，前端信息必须在 IR 中显式表达（例如 layout、精度模式）。

### ADR-002　每个算子必须有"参考语义"，它是唯一真值

- **决策**：算子定义必须同时提供 `infer`（类型推导）与 `reference`（不依赖加速库的数学语义）。
- **理由**：没有这一层，跨后端对不齐时无法判断问题出在 IR、kernel 还是库；最后会退化成"拿 cuDNN 当标准"。
- **代价**：算子开发量翻倍；参考实现要刻意保持"笨"，不能为了性能做特殊处理。

### ADR-003　节点强制标注可移植性

- **决策**：每个节点标记 `PORTABLE` / `LIBRARY` / `VENDOR`；跨后端共享的优化路径只允许 `PORTABLE`。
- **理由**：CUDA 库的约束（layout、workspace、算法选择）会不知不觉渗进 IR，等第二个后端接入才暴露，代价极高。标注 + verifier 是成本最低的闸门。
- **代价**：接入新库时需要人工判断归类，且 `LIBRARY` 节点无法参与共享优化。

### ADR-004　CUDA 生态照常复用，但必须走统一的 opaque 通道

- **决策**：允许调用 cuBLAS / cuDNN / cuSPARSE；每个外部调用都必须表达为一个可识别的 IR 节点，声明其约束（layout、精度、确定性、workspace）。
- **理由**：Transformer/CNN 的绝大部分算力集中在少数稠密算子上，重写它们不产生差异化价值。
- **代价**：性能上限部分由外部库决定；"自主编译器"的收益必须从库覆盖不到的地方（融合、新算子、内存规划、图级调度）去找。

### ADR-005　独立的是实现，入口保持兼容；适配器必须可移除

- **决策**：内核自研；支持读取外部模型格式（torch.export / ONNX）。所有外部框架依赖只允许出现在 `frontend/` 适配器内。
- **判据**：**删掉适配器目录、卸载外部框架后，内核与全部测试仍必须通过。**
- **理由**：大纲要求"存量模型零改写接入"，那入口就必须兼容；但内核独立不能被入口兼容拖垮。
- **代价**：适配器永远是"二等公民"，需要单独维护，且不能反向影响 IR 设计。

### ADR-006　符号维是一等公民

- **决策**：形状中的 `str` 表示运行期绑定的符号维，全链路（推导、校验、低层 IR）都必须支持。
- **理由**：真实推理场景 batch / 序列长度可变；若 IR 只支持静态形状，后期无法补。
- **代价**：类型推导复杂，广播规则在符号维上只能做保守判断。
- **补充（5.53）**：形状维从 `int | str` 扩到 `int | str | SymExpr`，符号可以参与整数算术；表达式仍是"只前向求值、不回解"，所以它不改变这条决策的边界，只是让派生维不必退化成静态数。

### ADR-007　Pass 函数式 + 每个 pass 后验 + 差分测试

- **决策**：Pass 输入图输出新图，不原地修改；每个 pass 结束后强制过 verifier；每个 pass 必须有"优化前后数值等价"的测试。
- **理由**：优化把图改坏是这类框架最常见的事故。只断言"节点变少了"的测试是不安全的。
- **代价**：重写图有内存与实现开销，但换来可组合性。

### ADR-008　不以 PyTorch 充当验证后端

- **决策**：不把 torch 作为第二个执行后端用于"跨后端一致性"验证。
- **理由**：它什么图都能跑，检出能力接近零，且会反向污染 IR 设计（让 IR 迁就 eager 语义）。
- **例外**：torch 可作为**导出器**用在适配器与测试对照中，不属于运行期依赖。
- **代价**：在真硬件到位之前，"跨后端"只能靠约束（类型系统、verifier、参考语义）来保证，无法实证。

### ADR-009　算子融合推迟到后端能力明确之后

- **决策**：先不实现融合 pass。
- **理由**：融合会产生新的融合算子，若目标后端无法承接，优化结果不可执行。必须先有"后端声明自己能接什么"的机制。
- **代价**：短期拿不到融合收益——而这恰恰是图级优化最主要的收益来源，所以它排在 CUDA 后端之后、第二轮优化之前。

### ADR-010　融合表达为「已有算子的输入扩展」，不发明融合算子

- **决策**：融合不创建新的融合算子，而是在语义完整的既有算子上扩展可选输入。
  第一条规则：`softmax(mul(x, s))` → `softmax(x, s)`（`s` 为标量）。
- **理由**：ADR-009 担心的"融合完没人执行"来自**新算子**。若融合结果本身就是
  一个 `PORTABLE` 算子的合法形态，参考执行器能执行、任何承接该算子的后端也能执行，
  于是不需要拆解 pass，也不存在能力协商失败。
- **代价**：表达力受限，只能吸收"算子本身可能带有的额外语义"（softmax 的缩放、
  matmul 的缩放/残差）。需要新计算的融合（如 GEMM epilogue、LayerNorm+残差整段）
  仍要另行设计。
- **约束**：仅在中间值 fanout == 1 且不是图输出时融合，保证语义不变；
  符号维（无法判定是否为 1 元素）不参与融合判定。

### ADR-011　IR 用一种扁平的 JSON 结构序列化，版本号必须被校验

- **决策**：图可以导出为纯 JSON（只用 JSON 原生类型），结构的第一个字段是
  `ir_version = [主, 次]`。读取时：主版本不同 → 拒绝；次版本高于本实现 → 拒绝；
  次版本更低 → 接受。
- **理由**：没有序列化，图永远由当前代码在进程内构造，"版本号"只是死代码，
  第三方插件也无法在不依赖本项目类的情况下解析图。ADR-001 说 IR 是唯一契约，
  那这份契约就必须能离开进程、并且能被拒收。
- **代价**：格式一旦对外就是兼容义务，新增字段必须保持可选；属性的 tuple/list
  差异需要统一（读入时一律还原成 tuple），否则"导出再导入"不相等。

### ADR-012　一致性按"能力声明"判定，允许显式拒绝

- **决策**：conformance 不要求后端承接全部算子，也不要求它承接某个算子的所有形态。
  判定规则是：**声明了就必须算对；算不了就必须抛 `BackendCapabilityError`**。
  另加覆盖度检查，防止"全部拒绝"也通过。
- **理由**：硬件能力参差是常态（CUDA 第一版就只支持最后维归约、只支持 f32）。
  若套件要求"全通过"，结果只会是后端虚报能力或套件偷偷放水——两种都会让能力协商
  失效，而能力协商正是插件化的前提。
- **代价**：报告通过不等于"该后端支持全部 IR"，只等于"它的声明可信"。所以报告必须
  同时给出 refused 清单与已覆盖算子，不能只给一个 PASS；调用方要按需读明细。

---

## 3. 代码结构

```
aetherflow/
├── aetherflow/
│   ├── frontend/           前端适配器（torch_export 是全项目唯一的 torch 依赖点）
│   │   ├── torch_export.py torch.export -> IR；未知算子明确报错；真实模型缺口用
│   │   │                   算子映射 / 分解承接（推理态 BN、精确 GELU，5.39）；
│   │   │                   ceil_mode / count_include_pad 直通 IR 属性（5.154）
│   │   └── eager.py        EagerGraph / Tensor：运算符直接构图，产物仍是同一份静态 IR
│   ├── ir/                 高层语义 IR
│   │   ├── types.py        DType / TensorType / 符号维与形状表达式（5.53）
│   │   ├── ops.py          算子语义表：infer + reference + portability + vjp
│   │   ├── grad.py         反向展开：backward() / GradCtx / NoGradientError
│   │   ├── grad_rules.py   37 条 VJP 规则与广播归约工具（导入即挂到 OPS）
│   │   ├── second_order.py 对反向图再求导：hessian_vector_product 的输入/输出命名映射（5.198）
│   │   ├── control.py     跨图控制流：ControlRegion / Conditional / WhileLoop + to_spec（5.201）
│   │   ├── graph.py        Graph / Value / Node，构图 DSL，文本 IR 打印
│   │   ├── version.py      IR 版本号与兼容判定（主版本不同即拒）
│   │   ├── liveness.py     编译期静态显存预算：liveness 峰值、复用倍数、符号维绑定（5.36）
│   │   ├── serialize.py    JSON 序列化：to_dict / from_dict / dumps / loads
│   │   └── verifier.py     版本校验 + 结构校验 + 可移植性校验 + 后端能力校验
│   ├── runtime/
│   │   ├── executor.py     后端无关执行骨架（符号绑定 + 形状校验 + 钩子）
│   │   ├── reference.py    参考语义执行器（唯一真值）
│   │   ├── telemetry.py    运行时上报：InstrumentedBackend + RunRecord（派发耗时 / wall / 显存，5.37）；
│   │   │                   逐节点 GPU 计时 `gpu_ms` 与 `node_gpu_totals`（5.161）；
│   │   │                   包装器按存在性转发六项可选契约，`implemented_contracts` 与内层一致（5.167）；
│   │   │                   跨运行趋势聚合与 JSON 往返（trend / load_records，5.38）
│   │   └── events.py       SNN 稀疏事件：CSR 变体表示 + 事件内存池 + 只读事件的发放率（5.35）
│   ├── tuning/             自动调优（与后端无关）
│   │   ├── space.py        TuningSpace：参数候选与确定性笛卡尔积（5.37）
│   │   ├── search.py       串行网格搜索：不可行候选显式拒绝，全不可行抛 AllRejectedError（5.37；5.140 起默认改用 paired.py）
│   │   ├── paired.py       交替 A/B：轮内配对差 + bootstrap 区间 + 两半同向检查；`--order shuffle` 可逐轮随机排列；轮内跨度超限时判 `long-round` 拒绝下结论；`warmup_rounds` 预热轮样本丢弃；`tune_staged` 两阶段坐标搜索；`tune_replicated` 跨采集复现（逐候选判定必须每次相同，5.146；`vote=majority` 允许少数次不同、需 3 次以上，5.149）；相对跨度门槛以预热轮平均耗时为标尺（5.148）；`min_same_sign` 逐轮核对方向、同号不足把 `faster`/`slower` 降为 `unstable`（5.160）（5.140～5.160）
│   │   └── store.py        ConfigStore：backend+device+op+signature 命中的 JSON 配置库（5.37）
│   ├── agent_api.py        只读智能体查询面：后端画像 / 算子变体 / 性能摘要 / 配置推荐（5.37）
│   ├── backends/
│   │   ├── base.py         准入面：name / capabilities() / execute()
│   │   ├── contracts.py    可选契约：device / device_manager / memory / stream + 分配器契约 + 结构探测（5.89、5.164）
│   │   ├── registry.py     算子注册表：注册即声明，查表即调度；同名多实现按 cost 选
│   │   ├── discovery.py    后端装配：内置三工厂 + pip entry point + 硬件包；按名/环境变量/自动探测挑（5.41）
│   │   ├── hardware.py     zip 硬件包：manifest 校验（含内容摘要与依赖声明）、安全解压、目录扫描、安装/卸载与登记（5.43 / 5.64）
│   │   ├── reference_backend.py
│   │   ├── device_backend.py CUDA / HIP 共享的设备执行器（注册、布局、内核派发）；任意单轴 argmax 与多轴 reduce 的多轴先置换合并（5.195）
│   │   ├── cuda/           CUDA 插件（codegen.py 生成 CUDA C 交 NVRTC 编译；cublas.py 是
│   │   │                   ctypes 薄绑定；cusparse.py 是 CSR×稠密 SpMM 的可选实现（5.203）；
│   │   │                   memory.py 是捕获期的 arena；backend.py 含参数常驻、
│   │   │                   CUDA Graph 运行期计划与中间缓冲回收）
│   │   └── hip/            HIP 插件（复用 cuda/codegen.py 的模板；rtc.py 是 hipRTC；
│   │                       runtime.py 是 ctypes 容器；blas.py 是 hipBLAS；backend.py 与
│   │                       CudaBackend 同为 DeviceBackend 的子类，只覆盖容器/编译/设备信息，5.40、5.109）
│   ├── passes/             中层图优化
│   │   ├── base.py         Pass / PassManager / GraphRewriter
│   │   ├── dce.py          死代码消除
│   │   ├── cse.py          公共子表达式消除
│   │   ├── simplify.py     形状与类型噪声消除
│   │   ├── fold.py         常量折叠（PORTABLE 子图求值，结果写成 constant.data；只物化存活边界，5.36、5.161）
│   │   ├── fusion.py       IR 层融合：softmax 吸收缩放、LayerNorm 吸收残差、SwiGLU 门控（5.36）
│   │   ├── registry.py     pass 注册表与按名入口：名字唯一、工厂每次给新实例、单独跑一个 pass（5.197）
│   │   └── autocast.py     自动混合精度：按表把计算降到 f16、在边界升回 f32（5.90）
│   ├── lower/              内核级下降（L2）：厂商无关的节点 + C 语法打印器（5.42）
│   │   ├── kernel.py       L2 节点：表达式 / 语句 / 参数 / Kernel
│   │   ├── cstyle.py       C 语法方言：打印器 + CStyleDialect（类型映射、prelude）
│   │   ├── plan.py         执行计划：block / grid / vec / shared / OC tile 的纯函数决策（5.54）；block 默认按规模自动推导（5.65）；scan 按 steps / elems 决定块长（5.155）
│   │   ├── elementwise.py  语义级 L2：逐元素表达式链与 4 通道向量化（5.52）
│   │   ├── indexing.py     语义级索引：`Layout` 的线性化与解线性化（5.177）
│   │   ├── translate.py    跨方言源码转译：登记过的标识符 / 头文件逐 token 改写，拒收项显式报错（5.202）
│   │   └── kernels.py      各类内核的 L2 描述（逐元素 / 归约 / 卷积 / 池化 / 注意力…）
│   ├── models/
│   │   ├── mlp.py          多层感知机（前馈最小闭环）
│   │   ├── cnn.py          卷积块 / 残差块 / 小型分类网络
│   │   ├── rnn.py          RNN / LSTM / GRU 的时间展开
│   │   ├── transformer.py  Transformer Block（符号化）+ 多层 decoder-only LM
│   │   ├── mamba.py        因果深度卷积 + selective scan + 门控的 Mamba block；带出窗口按最后一次 reset 清零（5.178）、稳定 softplus 与注入上界诊断（5.183）
│   │   ├── lm.py           SSM 语言模型：embedding → Mamba 栈 → rmsnorm → 词表头；f32 损失、reset 输入、生成边界与固定块推理（5.119、5.179、5.182、5.186、5.187、5.188）；跨请求会话 `LMRunner`（5.191）、批量解码 `generate_batch`（5.192）、会话前缀缓存 `prime`（5.193）与采样面（top-p / 重复惩罚 / 束搜索 `generate_beam`，5.194）、请求队列与连续批处理 `ServingEngine`（5.196）
│   │   ├── lm_text.py      ByteTokenizer（256 字节 + pad/eos）与 ChatTemplate（5.120）
│   │   ├── lm_tools.py     字节级 JSON 文法、受约束生成与工具执行回路；Tool 支持 JSON Schema 子集，ToolCallGrammar 按注册表合成（5.122、5.186、5.189）
│   │   ├── snn.py          LIF 一步 / T 步展开 / 平均发放率 / 多层网络
│   │   └── ssm.py          selective scan：离散化 / 递归 / 读出 / 跳连
│   ├── train/              训练闭环（唯一依赖 numpy 的训练层）
│   │   ├── params.py       参数初始化与 ParameterStore（按名字/秩选策略）
│   │   ├── optim.py        SGD（动量/Nesterov/weight decay）与 Adam（含偏差修正）+ 槽位状态
│   │   ├── loss.py         图内损失：mse / softmax_cross_entropy / l2_penalty
│   │   ├── checkpoint.py   权重 + 优化器状态 + meta 的 .npz 存档（原地覆盖加载）
│   │   ├── amp.py          f16 梯度缩放：种子当缩放因子，溢出跳过并回退
│   │   ├── data.py         DataLoader：按第 0 维切批，逐轮可复现洗牌
│   │   ├── schedule.py     学习率调度：无状态 lr_at(step) + warmup 组合器
│   │   ├── grads.py        全局范数裁剪 + 梯度累积（f32 缓冲、按权重和平均、按 token 加权）；窗口状态可存档（5.185、5.180）
│   │   ├── ema.py          参数滑动平均：初始化即复制，原地写回
│   │   ├── lm.py           语言模型数据侧：left padding + mask + reset；标签可编码成 one-hot 或 `(B,T)` 整数扁平下标（5.120、5.179、5.188）
│   │   ├── loop.py         训练循环：反向图建一次，step/run 负责种子与梯度名（5.66）；设备态常驻见 5.84；训练状态存档与续训见 5.180
│   │   └── parallel.py     多卡最小闭环：跨卡拷贝（主机中转）+ 主机侧 allreduce + 数据并行 Trainer（5.190）
│   ├── conformance.py      后端一致性套件：固定图集 + 五条契约检查
│   ├── fuzz.py             IR 差分模糊测试：随机图 + 五条不变量 + 非法属性变异 + 运行期语义错误（8 条固定用例），可选与 CUDA / HIP 逐例对拍（5.38、5.139、5.158、5.175）
│   └── plugin.py           插件算子扩展面：命名空间护栏 + 可卸载 + 内置保护
├── benchmarks/             性能基准：10 个固定用例 + 基线对比判回归
│   ├── cases.py            用例集（只依赖 IR，不依赖 torch）
│   ├── run.py              测量、存基线、判回归（CLI）
│   ├── tune.py             单算子调优：搜索 block、写配置库、应用后逐位对拍（5.37；5.140 起默认交替 A/B，5.141 加 `--order`，5.142 加 `--max-round-seconds`，5.143 加 `--warmup-rounds`，5.144 加 `--staged`，5.146 加 `--repeats`，5.148 加 `--max-round-factor`，5.149 加 `--vote`，5.150 打印逐次配对差，5.160 加 `--min-same-sign` 与三个归约用例）
│   ├── tune_compare.py     跨后端调优对照：同批旋钮两边选值是否相同（5.63；v3 报告带配对区间，5.140；5.141 记录 order_mode；5.142 记录轮内跨度；5.143 记录预热成本；5.144 记录两阶段分解；5.146 记录逐次复现摘要；5.148 记录相对门槛与实算秒数；5.149 记录投票方式与逐候选票数；5.150 记录逐次配对差）
│   ├── conv_compare.py     conv2d 两条实现路径的对照（不属于门禁）
│   ├── grouped_conv_compare.py  分组 / depthwise 卷积：直算核 vs 逐组 im2col（5.166；5.171 改显式 impl）
│   ├── sparse_compare.py   `sparse_linear` vs cuSPARSE SpMM vs 物化稠密乘（5.173）
│   ├── reduce_any_axis.py  非末维归约：一维铺法 vs 两维铺法（5.174）
│   ├── spike_csr_scan.py   `spike_csr`：行内串行 vs 块内扫描（5.176）
│   ├── cudnn_compare.py    CUDA 侧库级对照：ctypes 直调 cuDNN legacy API（5.69）
│   ├── miopen_compare.py   HIP 侧库级对照：ctypes 直调 MIOpen solution API（5.157）
│   ├── trend.py            跨运行吞吐 / 显存趋势：采集、按用例聚合、跨文件对比、按图名裁剪（5.38、5.199）
│   ├── baseline_ref.json   本地参考后端基线
│   └── baseline_cuda.json  RTX 2080 Ti 上的 CUDA 基线
├── examples/hardware/      示例硬件包 novac：一份方言 + 三方法后端类（5.44）
│   ├── novac/              manifest + novac/backend.py（NOVAC_DIALECT / NovaCBackend）
│   ├── build.py            按固定清单打 zip
│   └── novac-0.1.0.zip     交付物（2.9 KB / 3 个条目，可直接 install_hardware）
├── .github/workflows/ci.yml 云端正确性 + 模糊测试 + 手动/每日定时的 GPU 正确性/性能/趋势
├── tests/                  2003 个用例（无 GPU 时跳过 CUDA / HIP 组，无 torch 时跳过前端组）
├── demo_transformer_block.py
└── pyproject.toml          仅依赖 numpy
```

### 现有算子（46）

| 类别 | 算子 |
|---|---|
| 逐元素二元 | `add` `sub` `mul` `div`（numpy 广播规则，符号维保守处理） |
| 逐元素一元 | `relu` `silu` `gelu` `exp` `log` `sqrt` `neg` `tanh` `sigmoid` `erf`（后七个是 AD 需要的初等函数；`gelu` 是 tanh 近似，5.20；`erf` 为精确 GELU 引入，5.39） |
| 阶跃 / 脉冲 | `step(threshold)`（Heaviside，无梯度）；`spike(threshold, alpha)`（前向同 `step`，反向走三角窗替代梯度，5.18） |
| 常量 / 广播 / 填充 | `constant(dtype, shape, value)`（零输入节点）；`broadcast_to(shape)`；`pad(dim, before, after)` |
| 矩阵 | `matmul`（批量） |
| 卷积 | `conv2d(stride, padding, dilation, groups)`（NCHW 交叉相关）；反向语义 `conv2d_input_grad` `conv2d_weight_grad` |
| 池化 | `maxpool2d(kernel, stride, padding)`；`avgpool2d(kernel, stride, padding)`（分母固定，5.20）；`adaptive_avg_pool2d(output_size)`（窗口按 floor/ceil 划分、可重叠，全局池化 = output_size=(1,1)，5.34）；反向语义 `maxpool2d_backward` `avgpool2d_backward` `adaptive_avg_pool2d_backward` |
| 归一化 | `softmax(axis)` `layernorm(axes, eps)` `rmsnorm(axes, eps)` `batch_norm(axis, eps)`（训练态，5.24） |
| 注意力 | `attention(scale, dropout, seed)`（`softmax(q k^T * scale [+ mask]) @ v`，一个节点，5.29） |
| 随机 / 统计 | `dropout_mask(p, seed)`（SplitMix64，跨后端逐位一致，5.23）；`batch_norm_stats(axis)`（返回 `(2, C)` 的 mean/var，无 vjp，5.24） |
| 归约 | `reduce_sum(axis, keepdims)` |
| 形状/类型 | `reshape(shape)` `transpose(perm)` `cast(dtype)` `slice(dim,start,stop)` `concat(axis)` |
| 查表 | `embedding(table, ids)`；反向语义 `embedding_backward(table_shape)`（重复下标累加，5.20） |
| 序列扫描 | `scan_linear`（时间维静态的线性递归）；`scan_linear_backward`（其伴随递归，5.19） |

全部为 `PORTABLE`。当前没有任何 `LIBRARY` / `VENDOR` 节点——它们是给 CUDA 后端预留的位置。

---

## 4. 五条硬约定

写在代码注释里，这里汇总，作为 review 的检查项：

1. 后端不得在类型里附加私有字段（保证 IR 硬件中立）。
2. 每个算子必须有参考语义，且参考实现要"笨"。
3. 跨后端共享路径只允许 `PORTABLE` 节点。
4. Pass 不原地改图，且必须通过 verifier 与差分测试。
5. 外部框架依赖只允许出现在 `frontend/`，删除后内核必须仍然可用。

---

## 5. 已做的实验与结论

### 5.1 Transformer Block 端到端

24 个节点、12 个输入，全部 `PORTABLE`；与独立手写的 numpy 实现数值一致（rtol 1e-4）；同一份图在 `(1,1)`、`(3,7)` 等形状下复用，无需重建。

**结论**：当前的语义 IR 足以自然表达多头注意力、残差、FFN，没有出现需要"打补丁"的结构。reshape + transpose 的组合是目前最考验形状系统的地方。

### 5.2 中层优化在真实模型图上的收益

```
可合并的重复子表达式: 0
节点数: 24 -> 24
优化前后最大偏差: 0.000e+00
```

**结论（重要）**：真实模型图上几乎没有冗余子计算。DCE/CSE 的价值在于清理前端降级产生的噪声（重复 reshape、恒等 transpose、无用分支），**不是**主要性能来源。真正的图级收益必须来自：

- 算子融合（matmul+激活、LayerNorm 拆解合并）
- 内存规划（峰值显存、缓冲复用）

这条结论直接决定了路线图的排序：融合要排在 CUDA 后端之后，因为它需要后端先声明能力。

### 5.3　CUDA 后端验证（RTX 2080 Ti / sm_75 / CUDA 12.9）

CUDA 组全绿：14 个算子逐一与参考语义对齐，Transformer Block 端到端对齐（rtol 1e-4）。

实现路线（与 ADR-004 一致）：

- 逐元素与归约算子：**自己生成 CUDA C 源码，交给 NVRTC 运行时编译**。
  目标机器不需要安装 CUDA toolkit，只要有驱动即可，这让插件更轻。
- `matmul`：cuBLAS，通过自研 ctypes 薄绑定下发（为什么不用 `cupy.matmul`，见 5.4 M3-3）。
- 严格性验证通过：float64 输入被拒、非最后轴 softmax 被拒、非法广播在构图期被拒。
  失败一律显式，没有静默替换实现或精度。

### 5.4　性能实测（关键发现）

同一份 Transformer Block 图，参考后端（numpy）与 CUDA 后端对比：

| 配置 | d_model / d_ff / B×S | 参考 ms | CUDA ms | 加速比 |
|---|---|---|---|---|
| tiny | 128 / 512 / 2×64 | 1.46 | 1.69 | **0.9x** |
| small | 256 / 1024 / 4×128 | 14.44 | 2.09 | 6.9x |
| medium | 512 / 2048 / 4×256 | 54.15 | 3.99 | **13.6x** |

**结论**：小规模下 GPU 反而更慢，说明瓶颈不是算力，而是固定开销：

1. 每次 `execute` 都把全部输入（含所有参数）从主机拷到设备——参数本该常驻显存
2. 24 个节点独立 launch，没有融合，launch 开销占比高
3. 每个中间结果都完整物化到显存，没有复用

这三条正好对应 M3 的三件事（设备驻留内存、算子融合、内存规划）。
也就是说，**性能数据自己给出了下一步的优先级**，不需要靠猜。

#### M3-1 设备驻留实测

同一张图、同一个 backend 实例，只改变"参数是否驻留显存"：

| 配置 | d_model / d_ff | 每次 H2D ms | 参数驻留 ms | 提升 |
|---|---|---|---|---|
| tiny | 128 / 512 | 1.571 | 1.247 | 1.26x |
| small | 256 / 1024 | 1.980 | 1.498 | 1.32x |
| medium | 512 / 2048 | 3.849 | 2.475 | **1.56x** |

收益随参数规模增长，符合预期——省掉的正是参数字节数的 H2D。
但最大的配置也只提升 56%，说明**剩余开销主要不在拷贝**，而在
24 次独立 launch 与中间结果物化。这把 M3 的下一项精确地指向算子融合。

（本表与上表的配置不完全一致，只对比 cold/hot 的比值，不建议跨表比较绝对值。）

#### M3-2 融合实测（softmax 吸收 attention 缩放）

`fused_pipeline()` 把 `softmax(mul(scores, scale))` 融合为 `softmax(scores, scale)`：

| 配置 | 节点数 | 融合前 ms | 融合后 ms | 提升 |
|---|---|---|---|---|
| tiny | 24 → 23 | 1.168 | 1.124 | 1.04x |
| small | 24 → 23 | 1.357 | 1.297 | 1.05x |
| medium | 24 → 23 | 1.996 | 1.945 | 1.03x |

只有 3–5%，与理论一致：medium 下省掉的是 8MB 中间张量的一次写 + 一次读
（≈16MB 访存），相对 2ms 总时间本就只占百分之几。首次测量时曾出现 0.89x 的
反常值，提高迭代次数并取多轮最小值后稳定在 1.03x 以上——**单次测量不足以判断
几个百分点的优化**，这条经验已写进测试方法论。

**结论**：单条融合规则的收益不在吞吐，而在省掉一个峰值中间张量与一次 launch。
要拿到数量级收益，必须减少 kernel 数与中间张量（跨算子真融合）；其中 GEMM epilogue
这条捷径已被实验否定（见 M3-3），剩下的是自研 kernel 的整段融合。5.36 交付了其中
两条（LayerNorm+残差、SwiGLU 门控），CUDA 上各有一个内核；收益没有再单独做 A/B——
它们省掉的也是"一个中间张量的物化 + 一次 launch"，量级与这条规则相同（几个百分点）。

#### M3-3 运行期计划（CUDA Graph）实测

**动机**：tiny 配置下 GPU 实际计算只有约 5 μs，而 `execute` 要 0.9 ms——约 0.9 ms
全是 CPU 侧的固定开销（24 个节点 × 约 32 μs Python 胶水）。这部分与算力无关，
可以用"整图一次下发"直接抹掉。

原本计划的方向是 GEMM epilogue 融合（`matmul + add` 合并成一个 kernel），
实验后放弃，原因有两条：

1. cupy 没有暴露 cublasLt 的 epilogue 参数，cuBLAS 的输出必须物化；
2. 没有 epilogue 就等于没做融合——`matmul + add` 仍是两个 kernel、一次中间张量读写。

结论：**GEMM epilogue 在本项目里没有可落地的实现路径，收益为 0**，除非自研 GEMM。
这条因此从路线图移除，不做"看起来高级但没有实现手段"的规划。

改做的两件事：

- `cublas.py`：ctypes 直接绑定 `cublasSgemm_v2` 与 `cublasSgemmStridedBatched`，
  行主序用 `C^T = B^T A^T` 互换 A/B。**cupy 的 matmul 在 stream capture 期间会直接抛
  `NotImplementedError`（cuBLAS API during capture is unsupported），自研绑定才能被捕获。**
- `backend.capture(graph, sample_inputs) -> CudaGraphPlan`：先预热（NVRTC 编译、
  cuBLAS workspace 分配都不能发生在捕获期间），再用专属 `MemoryPool` 与 non-blocking
  stream 捕获整图；`plan.run(inputs)` 只做"拷输入 → 一次 graph launch → 同步 → 回读"。

两个踩过才知道的坑，都已固化进测试：

1. **输入拷贝与结果回读必须绑定到捕获流**。第一版用默认流拷贝，现象是"第一次结果
   正确、第二次拿到的是上一轮的结果"——不是计算错，而是读写跨流没有依赖。
   现在拷贝用 `ndarray.set(..., stream=...)`，回读前显式 `stream.synchronize()`。
2. **计划必须持有捕获期那个 MemoryPool**。图里写死的是捕获时那些显存地址，
   池被回收后 replay 就会写到别人的内存上。

实测（RTX 2080 Ti，24 节点 Block，参数驻留）：

| 配置 | exec ms | 纯设备 ms | replay ms | 计划 run ms | 端到端加速 | CPU 派发 |
|---|---|---|---|---|---|---|
| tiny 128/512, 2×64 | 0.922 | 0.771 | 0.222 | 0.278 | **3.3x** | 771 μs → 7.7 μs |
| small 256/1024, 4×128 | 1.081 | 0.776 | 0.370 | 0.539 | **2.0x** | 780 μs → 7.6 μs |
| medium 512/2048, 4×256 | 1.765 | 1.109 | 1.285 | 1.757 | 1.0x | 845 μs → 8.0 μs |

（"纯设备"= 仍逐节点走 Python 但不回读主机；"计划 run"= 含 H2D 与同步回读。）

**结论**：

- CPU 侧派发从约 800 μs 降到约 8 μs（**约 100x**），"每节点 32 μs 胶水"不再是瓶颈。
- 小规模端到端 3.3x；中规模被 GPU 计算（约 1.1 ms）与激活值 H2D 主导，
  图捕获不可能改善这部分——它的作用域只到"消除 CPU 开销"为止。
- 所以下一步的收益点已经不在"再省 CPU"，而在**真融合**（减少 kernel 数与访存）
  与**内存规划**（复用中间缓冲）。数据再次自己给出了优先级。

#### M3-4 捕获期内存复用实测

**动机**：CUDA Graph 把每个 kernel 用到的缓冲地址写死，所以中间张量的复用只能
发生在捕获期间——这是"库做不到、编译器该做"的地方，也是 M3 列出的最后一项。

第一版的做法是写一个确定性 arena（best-fit + 相邻块合并 + 按需向驱动申请），
替换 cupy 池。结果**反而更差**：medium 从池的 70 MiB 涨到 160 MiB。根因不在分配器，
而在执行器：`run_device` 把每个中间值都留在环境里直到整图结束，于是所有中间张量
**同时存活**，arena 没有任何可回收的块，只能一路向上要新块。

三处修复，每处都由实测逼出来：

1. **执行器按"最后一次被使用"回收**（`run_device(..., release=True)`）。节点执行完，
   把引用计数归零、又不是图输入的输入从环境里摘掉。预热轮必须同样开启，否则预热
   结束时 arena 里全是活块，捕获阶段拿不到可复用地址。
2. **回收点由执行器显式交给 arena**。cupy 14 的 `set_allocator` 只接受一个参数，
   释放走 `MemoryPointer.mem` 的 C 层虚函数；Python 子类即便实现了 `free()` 也接不到
   （实测 `BaseMemory` 可以子类化，但析构时不派发）。因此新增 `Executor.dispose()`
   钩子，CUDA 侧把地址还给 arena。
3. **arena 的块大小不随历史翻倍**。最初每开一块就把粒度翻倍（1→2→4→8 MiB），
   小块图（small）因此背上 8 MiB 的块，比池多占 6.5 MiB。改成"小块固定 1 MiB 粒度、
   大请求按精确大小成块"后打平。

实测（24 节点 Block，参数驻留，冻结统计窗口到捕获轮）：

| 配置 | 池 MiB | arena MiB | 峰值 MiB | 请求 MiB | 复用倍数 | replay 池 / arena |
|---|---|---|---|---|---|---|
| tiny 128/512, 2×64 | 1.1 | 1.0 | 0.6 | 2.2 | 3.5x | 0.1574 / 0.1564 ms |
| small 256/1024, 4×128 | 8.5 | 8.0 | 6.0 | 17.5 | 2.9x | 0.3582 / 0.3589 ms |
| medium 512/2048, 4×256 | 34.0 | 32.0 | 20.0 | 70.0 | 3.5x | 1.2788 / 1.2825 ms |

**结论**：

- **显存收益的主体来自 liveness，不是分配器**：medium 池从 70 MiB（不回收）降到
  34 MiB，是"用完即回收"的功劳；arena 再省到 32 MiB，只多 6%。
- **吞吐没有差异**：交替测量后 replay 完全打平。初次顺序测量曾得到 arena 慢 8–29%
  的结论，交替复测后消失——这再次印证 5.2 的那条经验，**单次顺序测量不足以判断
  几个百分点的差异**，探针必须交替。
- **真正可交付的是"可预算"**：`plan.peak_bytes`（峰值下界）与 `plan.memory_bytes`
  （实际占用）在捕获后直接可读，池给不出这两个数。编译期静态预算（不看运行、
  直接从 IR 算出峰值）已由 5.36 的 `plan_memory` 补齐，与捕获期峰值同量级。
- 因此默认仍是 `reuse_memory=True`，但它的价值是"报告与确定性"，不是"更快更省"。
  这条与 5.2 的结论同构：**图级优化的收益常在清理噪声，而不是主要性能来源**。

### 5.5　环境实测

- 本地：`py -3.10` + `numpy 2.2.6` 可用；`torch 2.4.1+cpu` 可用；无 NVIDIA GPU
- 远程验证机：Ubuntu 22 / RTX 2080 Ti 11GB / 驱动 550 / Python 3.10；
  已装 `numpy 2.2.6`、`cupy-cuda12x 14.2.0`、`triton 3.8.0`（Triton 暂未使用）
- 结论修正：Triton 对非 torch 张量的支持不可靠，最终选了
  "自研 CUDA C + NVRTC" 这条更干净、也更贴合本项目的路线

### 5.6　前端接入验证（M1）

`torch.export` 导出的图 → IR → 参考后端，与 torch eager 逐值对齐（rtol 1e-5），
覆盖 Linear、LayerNorm、SiLU、多头注意力（含 qkv 拆分、转置、批量 matmul）。

过程中暴露了两件真实的事，都按"补 IR 而不是绕开"处理：

1. **`linear` 带 bias 必然产生广播**。adapter 把 `aten.linear` 降级为
   `matmul + add`，其中 add 是 `(B,S,D) + (D,)`。这直接证明了"多维广播"
   不是可选项，已在 5.7 补进 CUDA 后端。
2. **qkv 拆分需要 `slice` 算子**。`chunk` / `split` 会落到 `aten.split.Tensor`，
   而 IR 里没有切片能力。没有选择"改测试模型以绕开"，而是新增了
   `slice(dim, start, stop)` 算子（14 个算子中的最后一个）。

另外 `clone` / `alias` / `detach` 在 SSA、值不可变的 IR 里是恒等，直接映射为同一值。

适配器支持范围：`linear`、`layer_norm`、`silu`、`relu`、
`add`/`sub`/`mul`/`div`、`matmul`/`bmm`、`softmax`、`view`/`reshape`、
`flatten`、`permute`、`transpose`、`split`/`chunk`、`conv2d`/`convolution`、
`max_pool2d`（含 `_with_indices` 变体）、`getitem`、`clone`。

未知算子抛 `UnsupportedOperatorError` 并指出补规则的位置，不静默 fallback。

**版本差异实测**：torch 2.4 的 export 会把 `chunk` 降解成 `aten.split.Tensor`，
torch 2.14 则原样保留 `aten.chunk.default`。两条规则都已实现，并分别在
本地 torch 2.4.1 与远程 torch 2.14.1 上验证。这是"入口兼容"的固有成本：
适配器必须跟着上游 schema 演进，而不是假设一套算子名永远不变。

### 5.7　CUDA 多维广播（M2 补全）

前端验证暴露的第一个真实缺口：`aten.linear` 降级成 `matmul + add` 后，
bias 的 `(B,S,D) + (D,)` 被 CUDA 后端拒绝。选择补后端，而不是绕开模型。

做法：

- 按 numpy 规则把两侧 stride 右对齐到输出形状，**广播维 stride 置 0**，
  同一个输入元素被多个输出位置复用
- 相邻且两侧 stride 都连续的维会折叠，`(B,S,D) + (D,)` 折叠后只剩 2 维，
  索引计算量不随秩增长
- 折叠后仍超过 4 维则抛 `BackendCapabilityError`，不静默换实现
- 标量（`size == 1`）保留专用快速 kernel，不进广播路径
- 非法广播在 **IR 构图期**就被 `_broadcast_shape` 拒绝，不会留到运行期

验证分两层，避免只有 GPU 才能发现问题：

1. 本地 `TestBroadcastPlan`：用 numpy 逐元素复现 kernel 的索引计算，
   与 numpy 广播结果对拍，不需要 GPU
2. 远程 GPU：`(3,5)+(3,1)`、`(2,3,4)+(4,)`、两侧同时广播、5 维广播，
   均与参考语义对齐

这条补充印证了一个模式：**前端接入是后端的压力测试**。真实模型图会自动
暴露"以为不重要"的缺口（广播、切片），比自设计的用例更可信。

### 5.8　IR 序列化与版本治理

**动机**：M4 里有两件事不依赖新硬件（见路线图），先做 IR 版本化——它是
"第三方插件敢押注"的前提，也是 conformance 套件能拿固定样例跑起来的基础。

**先要看清一个陷阱**：直接加一个 `IR_VERSION` 常量是没有意义的。图永远由当前
代码在进程内构造，版本号永远不会被读到，是死代码。所以第一步必须是**让图能离开
进程**——JSON 序列化。没有序列化就不该谈版本治理。

格式（见 ADR-011）是扁平 JSON，只用原生类型，第一个字段就是版本：

```json
{
  "ir_version": [1, 0],
  "inputs": [{"name": "x", "type": {"dtype": "f32", "shape": [2, "B", 32]}}],
  "nodes": [{"op": "matmul", "inputs": ["x", "w"], "attrs": {},
             "output": {"dtype": "f32", "shape": [2, "B", 32]}}],
  "outputs": [{"value": "v3", "name": "y"}]
}
```

三条关键选择：

1. **每个节点带声明输出类型**，读入时与实现推导的结果对照。不一致就失败——
   这说明图作者与算子语义已经分叉，继续跑只会得到错的结果。
2. **属性统一还原成 tuple**。JSON 没有 tuple，而构图时 `perm=(0, 2, 1)` 是 tuple；
   不统一，"导出再导入"就不相等，往返测试会立刻发现。
3. **版本判定放进 verifier**，而不是只在 `from_dict` 入口查一次。这样任何拿到
   图的后端（含第三方插件）执行前都会被拦住。

**验证**：8 个格式用例——往返全等、主版本不同拒绝、次版本更高拒绝、更低接受、
缺版本拒绝、声明类型不一致拒绝、未知算子拒绝、verifier 拒绝；另加 1 个真实模型图
的往返用例（Transformer Block，覆盖 14 个算子的实际组合）。往返用例除了比较
`to_dict` 全等，还比较后端在新旧图上的执行结果逐值相等：只断言"字段没丢"
不足以证明语义没变。

**结论**：这一步没有性能收益，它买的是**契约的可执行性**。没有它，"IR 是唯一
契约"只是一句口号；有了它，第三方可以先照着 JSON 写解析器，再用 conformance
套件证明自己对——顺序不能反。

顺带修正了一处过期描述：顶层 `aetherflow/__init__.py` 的 docstring 还写着
"尚未实现 torch.export / CUDA 后端"，与现状不符。

### 5.9　后端一致性（conformance）套件

**动机**：5.8 让图能离开进程，这一节回答"离开进程之后，凭什么相信某个后端
真的承接了这套 IR"。没有它，"插件化后端"只能靠 demo 证明，而 demo 是可以
挑着跑的——恰好是最不可信的证明方式。

`run_conformance(backend) -> Report` 用固定用例（IR 构图，含真实
Transformer Block、一个 3x3 卷积与两种池化）跑五条检查。下面是交付时的 31 个用例；
自动微分（5.17）、SNN（5.18）、SSM（5.19）、新增算子（5.20）、随机算子（5.23）、
批归一化（5.24）、融合注意力（5.32）与自适应池化（5.34）之后已扩到 66 个。

| 检查 | 不通过意味着什么 |
|---|---|
| `contract` | 接口不完整，或声明了不存在的算子 |
| `portability` | 图集里出现了非 PORTABLE 节点（IR 已泄漏硬件假设） |
| `serialization` | 图不能原样往返 JSON，IR 还不是可交付契约 |
| `explicit_failure` | 用契约之外的方式失败：抛别的异常、或执行了未声明的算子 |
| `coverage` | 声明支持，却没有任何用例真正跑起来 |

**关键设计：对"部分支持"采用条件判定。** CUDA 声明了 softmax，但只支持最后一维。
判它失败是错的（后端已如实报错），判它通过也不完整（能力声明里看不出边界）。
规则因此是：声明支持但算不了的，只要抛 `BackendCapabilityError` 就记为
`refused` 并通过；真正的失败只有两种——**算错了**，或者**用能力协商之外的
方式绕过契约**。

这条规则单独存在会让"全部拒绝"也能通过，所以补一条覆盖度检查：每个被声明、
且套件里有对应用例的算子，至少要成功执行一次。"声称支持但一个用例都跑不了"
因此会被抓住。

**用坏后端验证套件本身。** 只断言"参考后端通过"没有意义——一个永远返回 PASS
的套件也能通过那条断言。测试里造了四种失控后端，四种都被正确判负：

| 假后端 | 被哪条检查抓住 |
|---|---|
| 只声明 add/relu | 不算失败；其余用例全部 `refused`，整体 PASS（能力协商被尊重） |
| 声明全部但输出全零 | case 数值不一致 |
| 声明全部但抛 `RuntimeError` | `explicit_failure`（异常类型不符） |
| 每次结果差 1e-6 | 数值检查放过（< atol），确定性检查抓住 |

**结果（RTX 2080 Ti）**：

```
backend : cuda                              caps: 16
[PASS] contract          16 个算子已声明
[PASS] portability       31 张图全部只含 PORTABLE 节点
[PASS] serialization     31 张图 JSON 往返全等
[PASS] explicit_failure  5 处拒绝全部为 BackendCapabilityError
[PASS] coverage          16/16 个被声明的算子至少成功执行一次
cases   : 26 executed / 5 refused / 0 failed
result  : PASS
```

5 处拒绝精确命中已知边界，且每一处都点名原因：matmul 批量部分广播、
softmax / layernorm / reduce_sum 的非最后维、f64 输入。已执行用例的最大绝对
误差是 9.5e-07（matmul，cuBLAS 与 numpy 的累加顺序差异），卷积为 4.8e-07，
池化为 0（纯索引运算，结果逐位相同），其余逐元素算子为 0。

**结论**：这一步买的是**准入判据**。M4 的第二个硬件后端不必再靠"跑通几个 demo"
自证——`run_conformance(新后端)` 通过就是它的准入条件，且判据可复现、报告可 diff。
同时它把"跨后端"从一句声明变成了五条可执行约束；真正缺的仍然是一台非 CUDA 机器。

一个刻意没做的事：套件不提供"自动修复"或"放宽容差后重试"。一致性只有通过和
不通过两种结局，容差是图集里写死的数字，调用方改它就是在改考题。

### 5.10　插件契约的三层：准入面 / 可选契约 / 暂不定义

**问题**：1.3 把"插件还不是插件"列为第一承重墙。`Backend` 只有
`name` / `capabilities()` / `execute()`，大纲要求的设备管理器、分层内存分配器、
算子注册与调度在接口里没有位置。真接第二个硬件时，第一件事就是改接口，
改完 conformance 全部重跑。

**做法**：不往 `Backend` 上加方法，而是拆成**准入面 + 可选契约**。
准入面保持三个成员不变（少一个就不算后端），可选契约放进
`backends/contracts.py`，用 `runtime_checkable` 做结构探测：

| 层 | 内容 | 谁实现 |
|---|---|---|
| 准入面 `Backend` | `name` / `capabilities()` / `execute()` | 全部后端 |
| 可选 `DeviceContract` | `device_info() -> DeviceInfo` | CUDA |
| 可选 `MemoryContract` | `memory_stats() -> MemoryStats \| None` | CUDA（捕获期 Arena） |

> 上表是 5.10 当时的记录。契约后续扩到三项（新增 `StreamContract`），
> 并补了 `MemoryContract.allocator()` 与它交出的 `AllocatorContract`——见 5.89。

`implemented_contracts(backend)` 按声明顺序返回声称实现的契约名，
`supports(backend, Protocol)` 单独判断一项。conformance 报告因此多一行
`contracts`，**只作信息展示，不参与判定**——只实现最小面的后端依然合格，
缺可选契约不是缺陷。

**三个刻意的取舍**：

1. **不定义 `CodegenContract` / `TuningContract`。** 现在只有一种 codegen 实现
   （插件内部拼 NVRTC 源码字符串），按它猜出来的接口，等第二个后端出现几乎
   必然要改。契约应当从两个实现共同的部分里提炼，不能从一个实现里发明。
2. **探测不等于验证。** `isinstance` 只看方法存在，不看签名，更不看返回值。
   这条局限写进了模块注释，并有一条测试把它钉住（返回元组的方法也会被认成
   `DeviceContract`）。契约是否正确只能靠实际调用与 conformance 证明。
3. **没有统计就返回 `None`，不返回 0。** 0 是观测值，会让"没有统计"与
   "用了 0 字节"混为一谈。`memory_stats()` 只在 `reuse_memory=True` 的捕获后
   有值；普通执行走 cupy 池，池不提供峰值，就不编一个数字。

**收益与代价**：第二个后端不实现可选契约也能通过 conformance；实现了就多出
可被查询的能力，不必为了"接口完整"写一堆返回空值的占位方法。代价是能力发现
靠运行时探测而非静态类型——静态检查看不出某个后端是否实现了 `memory_stats`，
这一栏由 conformance 报告补上。

**验证（RTX 2080 Ti）**：报告首部如实列出 CUDA 插件实现了两项契约，判定不变：

```
backend : cuda
caps    : 16 个算子
contracts: device, memory
...
result  : PASS
```

`memory_stats()` 在捕获前返回 `None`、`reuse_memory=True` 捕获后返回
`MemoryStats`（峰值、总占用、累计请求、当前存活四个数），两个分支都有测试钉住。
写测试时发现一个容易误判的细节：`peak_bytes` 含捕获前就已存活的量，
而 `requested_bytes` 只统计 `reset_stats()` 之后的新请求，二者**不可比**——
最初写的 `requested >= peak` 断言在 1 KiB 的小图上直接失败。

**当时没做的**：算子注册与调度接口（已在 5.11 补上）、插件开发指南。
指南要等接口不再变的时候写，否则文档写的是还会变的接口。

### 5.11　算子注册与调度

**问题**：后端关于"我承接哪些算子"的信息有两份——`capabilities()` 返回的集合，
和方法名反射 `getattr(self, f"_op_{name}")`。两份信息分叉时，失败都是安静的：

- 声明里有、实现里没有 → 第一次推理才炸，而且报的是 `AttributeError`，
  不是契约要求的 `BackendCapabilityError`；
- 实现里有、声明里没有 → 前端明明可以用它，却绕开去做拆分或降级，白丢性能，
  且没有任何提示说明这件事发生过。

**做法**：`backends/registry.py` 里的 `KernelRegistry`，**注册即声明，查表即调度**。
`capabilities()` 直接由注册表派生，`invoke` 走 `dispatch`，两份信息合并成一份。
注册时三项校验，失败方式刻意不同：

| 情况 | 异常 | 为什么 |
|---|---|---|
| 算子名不在 `OPS` 里 | `KeyError` | 拼错的名字注册了也永远不会被图用到，是死条目 |
| 重复注册同一算子 | `ValueError` | 静默覆盖会让"哪个内核在跑"不可知 |
| 传了不可调用的对象 | `TypeError` | 早失败，别拖到调度那一刻 |

未注册的算子被调度时抛 `BackendCapabilityError`（带节点名），与 5.9 的
"不支持必须显式拒绝"是同一条规则。

**`CUDA_OPS` 为什么还留着**：它是**不建 CUDA 上下文就能读到**的能力清单。
构图工具、预检脚本不该为了问一句"接不接得住"而初始化一块显卡。为了不让它
变成第二份真相，做两道对账——构造时比较注册表与常量，不一致直接 `RuntimeError`
（GPU 机器上执行）；另有一条不依赖 GPU 的静态测试，用 `inspect` 比较
`CUDA_OPS` 与 `CudaBackend._op_*` 方法集合。前者抓"注册函数里漏写一行"，
后者抓"改了常量没改方法"。

**为什么这次改造是安全的**：对外行为完全不变——能力集合、异常类型、错误文案、
执行路径语义都没动，只是把"查哪张表"从反射换成了注册表。既有的 144 个测试
就是这次重构的回归证明；改造后 155 个测试本地与 RTX 2080 Ti 上均全绿，
conformance 报告不变（当时 14 算子、23 executed / 5 refused / PASS）。

**当时刻意不做的两件**：不做内核优先级与按代价选择（每种算子在每个后端目前只有
一个内核，多实现是自动调优阶段的事，等有可比较的候选再引入）；不让注册表
接受 `OPS` 之外的算子名（自定义算子该先进 IR 算子表，否则图里也表达不出来）。
前一件在 5.30 补上了机制——但真实候选还没出现（attention 在 5.32 落地的是唯一
实现，不是两条可比较的路）；后一件没有放宽，而是给了"进算子表"一条带护栏的路
（同样在 5.30）。

### 5.12　卷积语义（CNN 的第一刀）

**为什么先做语义、不先做内核。** 1.2 里 CNN 整行是空的，原因是缺 conv。补它的
自然顺序是从 IR 开始：IR 定义"卷积应该算什么"，后端再各自实现。反过来先写
CUDA kernel，语义就会被那一个 kernel 的边界（方形核、固定 stride、某种布局）
反向定义，IR 又一次变成某个后端的影子——这正是 ADR-002 要防的事。

**语义**：NCHW 交叉相关，与 PyTorch `conv2d` 一致（不翻转核）。属性为
`stride` / `padding` / `dilation`（int 或二元组，JSON 往返后是 list，两种都接受）
与 `groups`（暂只支持 1）；`bias(O,)` 是可选的第三输入。输出空间维按
`floor((H + 2p - d*(k-1) - 1) / s) + 1`，`<= 0` 直接报错。

**符号维的边界**：batch 与通道可以是符号维；空间维只在 **1x1 / stride=1 / pad=0**
（输出恰好等于输入）时保留原符号，其余情形显式拒绝。理由与 `reshape` 同一条
约定：符号维必须已在输入中出现，运行期才有东西可绑定；派生符号需要一套运行期
shape 表达式机制，那是另一件事。拒绝一个说清理由的编译期错误，好过编造一个
永远绑不上的符号名、等执行时才 `KeyError`。（5.53 补上了这套表达式：空间维现在
写成 `floor((S + 2p - k) / s) + 1` 的表达式留在 IR 里，执行期按已绑定符号求值核对；
kernel / stride 仍要求静态。）

**参考实现**：numpy 逐 `(kh, kw)` 位置用 `einsum` 累加，**不做 im2col**。
参考实现的职责是定义语义，不是定义性能；循环顺序固定，所以结果确定——
conformance 的确定性检查依赖这一点。CUDA 侧对齐的是这份语义，不是这份代码。

**CUDA 实现：im2col + cuBLAS。** 列矩阵按 batch 分段成 `(N, K, HO*WO)`
（`K = C*kh*kw`），权重 `(O, K)` 在所有 batch 间共用（strided batched GEMM 的
`strideA = 0`）。这样 GEMM 的输出天然就是 `(N, O, HO*WO)`，直接 reshape 成
NCHW，**不需要 col2im，也不需要转置拷贝**。im2col 本身是一个自研 NVRTC kernel
（一输出元素一线程、grid-stride），bias 复用已有的广播 `add` kernel 而不写
第二个自定义 kernel。后来又补了一条不物化列矩阵的分块直算路径，按形状自动
选择走哪条（5.16）。

一个踩到的坑：**NVRTC 的源码必须是 ASCII**。第一版在 CUDA 源码里写了中文注释，
本地（无 GPU）完全看不出来，远程一跑就是 `UnicodeEncodeError: 'ascii' codec ...`。
现在 codegen 的源码里留了这条约束的注释——它属于"只有真机才能暴露"的那类问题，
和 5.4 里那些只有跑起来才看得见的结论是一类。

**当时刻意不做的**（都属于 CNN 的后续切片）：分组卷积、`same` / `valid` 自动
padding、与 cuDNN 的性能对照。顺序仍是"IR 语义 → 参考实现 → 后端内核"，
与 Transformer 走过的路一致。im2col 的显存放大后来由分块直算解决，见 5.16；
分组卷积与 depthwise 在 5.34 补上（按组切片复用 im2col），当时的两条缺口只剩
自动 padding 与 cuDNN 对照（对照已在 5.69 补上）。

**验证**：本地 7 个边界用例（valid、same padding 的角落计数、stride、dilation、
batch>1、符号维执行、非法参数拒绝）；GPU 侧 6 个参数组合逐一与参考语义对拍
（padding / stride / dilation / 1x1 无 bias / 非方形核 / 非对称参数）。conformance
图集新增 `conv2d_3x3_pad1`：参考后端 31/31 执行，CUDA 侧当时 26 executed / 5 refused、
卷积最大误差 4.8e-07，报告 PASS（五个新算子在 5.20 补上原生内核后，最新一轮 CUDA
报告为 55 executed / 0 refused）。这条路径还说明了新算子进入 IR 的两种合规状态：
**先以 `refused` 出现（插件未声明），实现内核后转为 `executed`，两端的报告都 PASS**。

### 5.13　池化：一个根本不需要 GEMM 的 CNN 算子

卷积之后自然要接池化。它和卷积最大的不同在语义形状上：**没有权重、没有 bias、
没有浮点累加**，输出只取决于索引。这带来两个后果——参考实现可以完全逐元素地
定义，CUDA 侧也就不该去借 cuBLAS。

**语义**：NCHW 最大池化。`kernel` 必填（int 或二元组），`stride` **默认等于
kernel**（这是 PyTorch 的默认，也是"不重叠切块"的直觉），`padding` 默认 0。
输出空间维 `floor((H + 2p - k) / s) + 1`，`<= 0` 直接报错。符号维规则与卷积一致：
只有 1x1 / stride=1 / pad=0 时保留原符号，其余显式拒绝（5.53 起改为派生表达式）。

**一个容易写错的语义细节：padding 区域不参与取 max。** 参考实现把边界垫成
`-inf` 而不是 0；否则全负输入经过 padding 后，边界值会被"抬"到 0，模型不会报错，
只会安静地算错。这条在 IR 测试与 CUDA 测试里各钉了一次（`-5` 的输入必须得到 `-5`）。

**参考实现**同样不追求性能：按 `(ki, kj)` 偏移对 padded 输入做切片，
`np.maximum` 逐次归并，循环顺序固定所以结果确定。

**CUDA 实现**是一个自研 NVRTC kernel：一个输出元素一个线程、grid-stride，
在 kernel 内按 `(n, c, ho, wo)` 反解索引，窗口内跳过落在 padding 的格子。
空窗口的初值用 `__int_as_float(0xff800000)`（负无穷）——这里踩了一个只有真机
才暴露的坑：**NVRTC 默认不带 `<math.h>`，`INFINITY` 宏未定义**，第一版编译
直接失败（`identifier "INFINITY" is undefined`）。改用 IEEE 位模式构造后正常，
源码注释里记下了这条。

**当时刻意不做的**：平均池化 / 全局池化 / 自适应池化（语义与内核都是另一刀）、
`ceil_mode`、`return_indices`（反向传播需要，属于自动微分那一阶段）。
其中平均池化在 5.20 补上，自适应池化与全局池化在 5.34 补上——全局池化不单独设
算子，`output_size=(1, 1)` 就是它。仍未做的是 `ceil_mode` 与 `return_indices`。
CUDA 侧也没有做窗口内向量化加载——现在每个输出线程读 `k*k` 个元素，
对 2x2/3x3 这种小窗口来说访存已经接近最优。

**验证**：IR 侧 7 个用例（默认 stride、重叠窗口、padding 边界的手算期望、
非方形核 + batch、符号维、非法参数）；GPU 侧 5 个参数组合对拍，另加一条
`conv → relu → maxpool` 的整图 CUDA Graph 捕获用例，证明新增的索引 kernel
同样可被捕获与 replay。conformance 图集新增 `maxpool2d_2x2` 与
`maxpool2d_3x3_pad1`（重叠窗口 + padding），CUDA 侧两个用例误差均为 0。

### 5.14　CNN 的前端导入：算子有了不等于模型能用

5.12 与 5.13 让 `conv2d` / `maxpool2d` 在两个后端上都能算，但真实模型要经过
torch.export 才进得来。这一步补的是**入口**，验收标准也因此从"算子对拍"换成
"整个模型三方对齐"：torch eager、IR 的参考后端、CUDA 后端，同一个输入必须一致。

**新增的三类降级**

| aten 目标 | 降级 | 需要处理的差异 |
|---|---|---|
| `aten.conv2d.default` / `aten.convolution.default` | `conv2d` | 两者参数表不同：`convolution` 在 dilation 之后多出 `transposed` / `output_padding`，groups 在最后 |
| `aten.max_pool2d.default` / `_with_indices.default` | `maxpool2d` | 后者返回 `(values, indices)`，IR 没有 indices，槽位放 `None` |
| `aten.flatten.using_ints` | `reshape` | IR 的 reshape 要求静态形状，符号维落在合并区间里就明确拒绝 |

**两个刻意的选择**

1. **indices 不是"顺手丢掉"，而是留一个 `None` 槽位。** 如果下游真的去取 indices
   （比如要写反向传播），`a(i)` 会抛 `UnsupportedOperatorError` 并说明来源是未支持的
   多输出槽位。直接返回 values、假装 indices 不存在会让错误延后到一个更难解释的位置。
2. **不支持的 CNN 变体在前端就拒绝，不留给 IR。** 转置卷积、`ceil_mode`、
   dilated pooling 都抛 `UnsupportedOperatorError`。让 IR 抛 `ValueError` 也能跑，
   但前端异常类型是"这个模型不在支持范围"的统一信号，混用会让人以为是 IR 有 bug。
   （分组卷积当时也在这条拒绝清单里，5.34 之后改为直接透传 `groups`。）

**一个版本差异导致的测试失败**：GPU 上的端到端断言最初写死 `ref["out0"]`，本地通过、
远程 `KeyError`。原因是输出别名来自 `graph_signature.user_outputs`，不同 torch 版本
给出的名字不同（本地 2.4.1 / 远程 2.14.1）。测试改成取第一个输出——**把断言绑在
版本相关的别名上，本身就是个错误**。这与 5.6 里 `chunk` 被降级成 `split` 是同一类
成本：入口兼容必须跟着上游演进，不能假设 schema 稳定。

**验证**：LeNet 形态（2 层 conv + relu + pool，flatten 后接 linear）在参考后端上
与 torch eager 逐点一致；在 GPU 上再与 CUDA 后端三方对拍。另有一条非默认参数用例
（stride / 非方形核 / padding / dilation 同时生效）与三条拒绝用例（分组卷积、
转置卷积、dilated pooling；分组卷积在 5.34 转为支持）。本地 204 个测试
（130 通过 / 74 跳过），远程 204 全绿。

**当时仍然缺的**：cuDNN 对照（已在 5.69 补上：自研分块核慢 4～12 倍，im2col + cuBLAS 慢 5.5～12 倍）。`avg_pool2d` 与自适应 / 全局池化已分别由 5.20 与 5.34
补上，CNN 这一行因此还是 🟡，但缺口已经从"不能用"变成了"能跑、可选分块"。

### 5.15　性能基准与回归门禁

算子在、模型进得来，接着要回答的是另一个问题：**下一次改动怎么知道没变慢**。
这一刀交的是 `benchmarks/`（8 个固定用例 + 计时 + 判定 + CLI）、两份基线 JSON
与 `.github/workflows/ci.yml`。核心结论：性能门禁能不能用，取决于**在哪台机器上比**，
而不取决于阈值调得多细。

**8 个用例，分两层**

| 用例 | 形状 | 想钉住的东西 |
|---|---|---|
| `matmul_256` | `256³` | 小 GEMM：调度与 launch 开销占大头 |
| `matmul_1024` | `1024³` | 大 GEMM：算力为主，退化最敏感 |
| `matmul_batched` | `(8,128,256)×(8,256,128)` | strided batched 路径 |
| `conv2d_3x3_pad1` | `(2,3,64,64)*(16,3,3,3)` | 浅通道 3x3：K=27，auto 走分块直算 |
| `conv2d_1x1` | `(2,32,32,32)*(64,32,1,1)` | 无空间聚合：K=32，auto 走分块直算 |
| `maxpool2x2` | `(2,32,256,256)` | 纯索引 kernel，访存主导 |
| `transformer_block` | `x=(4,32,64)`，d=64/4 heads | 图级：多 kernel 混合 + 派发 |
| `cnn_lenet` | 手工 IR 的 LeNet | 图级：CNN 端到端 |

选取原则是"**两端 + 两层**"：GEMM 取大小两端，算子取计算密集与访存密集各一，
再用两个图级用例覆盖派发与内存分配。刻意不放"跑一次要几秒"的用例——回归门禁
一旦变慢到没人愿意等，就等于不存在。另一半约束来自实测：**测不准的用例不该进门禁**。
`maxpool2x2` 最初是 `(2,16,64,64)`，单次 0.1 ms，同一台机器上重复测量得到的中位数
在 0.097 ~ 0.186 ms 之间跳（一次 CPU 抢占就能翻倍），第一次跑门禁就报了 +62% 的假回归。
放大到 8 ms 量级后，同一组重复测量的波动降到 4% 以内。

**为什么用中位数，而不是均值**

GPU 计时受时钟、调度、邻进程影响，单次尖峰会明显拉偏均值。`measure` 取
`warmup` 次后再 `repeat` 次，报 `median_ms`，同时存 `p90_ms` / `min_ms` 只作诊断。
每次计时都包住 `execute` **加上 stream 同步**——少了同步，测到的是 CPU 派发耗时，
不是 GPU 实际完成时间，这会让"变小变快"和"变大变慢"同时失真。

**为什么还要"多轮取最好"**：中位数只压得住单次尖峰，压不住**整轮漂移**。
在远程那张 2080 Ti 上连续跑两次全量：`matmul_256` 0.229 → 0.299 ms（+30%）、
`matmul_1024` 1.683 → 2.147 ms（+28%），两次都没改一行代码，直接触发了假回归。
这种漂移来自整机状态（邻居负载、时钟），不是某个 kernel 变慢。所以每个用例跑
3 轮（`--rounds`），取三轮里最小的中位数：判定对准的是"这台机器**能做到**多快"，
而不是"这一分钟有多吵"。真实退化是系统性的，三轮都慢，仍然会被抓住。

**为什么阈值是 25%，以及为什么必须同机同后端**

两条都不是调出来的，是约束推出来的：

1. **同机**。基线与当前记录 `backend` 与 `device`，不一致直接判失败
   （`backend_mismatch`）。跨机器比中位数没有意义——一张 2080 Ti 和一台共享 runner
   的差距远大于任何回归阈值。
2. **25%**。固定机器上的噪声一般在 10% 以内（`p90/median` 可从基线里读出来），
   共享 runner 的抖动经常超过 25%。取 25% 是"容忍噪声、拦住真实退化"的交点：
   换掉一个 kernel、去掉一次融合、多一次来回拷贝，量级都在 25% 以上。

**判定规则**（`check_regression`，三类问题）

- `backend_mismatch`：基线是 `cuda`、当前跑的是 `reference`（或反之）——不可比就是失败
- `missing`：基线里有、当前结果里没有——用例被删或改名同样算退化，不能靠删用例变绿
- `slower`：`median > baseline_median × (1 + tol)`
- **新用例不算回归**：新增功能天然没有基线，报 `missing` 的方向反过来不成立

**超阈值不直接判负：重测一次**

`check_regression` 是纯函数，只看数字；但数字背后还有一层现实：整机级的瞬时干扰
（杀毒扫描、Windows 更新、邻居容器）能把**整轮**测量拖慢一倍。实测里同一分钟内两次
本地检查，第一次 `matmul_256` 0.2447 → 0.4493（+84%）、`transformer_block`
0.8469 → 1.4791（+75%），第二次全部回到正常。三轮取最好救不了这种情况——三轮都在
被污染的那几秒里。

所以 CLI 的规则是：**首轮超阈值不算数，对超阈值的用例再测一遍，取两次中更快者；
还超才算回归**（`recheck_slow`）。真实退化是可复现的，两次都慢；瞬时干扰不会。
代价是"只出现一次的偶发退化"会被放过，这是刻意接受的——门禁频繁误报的话，
最后一定会被关掉，那比漏掉一次更糟。

**基线数字**（median ms，同机采集）

| 用例 | 参考后端（numpy） | CUDA（RTX 2080 Ti） |
|---|---|---|
| `matmul_256` | 0.2447 | 0.2350 |
| `matmul_1024` | 4.8400 | 1.7596 |
| `matmul_batched` | 1.1868 | 0.4508 |
| `conv2d_3x3_pad1` | 6.0085 | 0.3393 |
| `conv2d_1x1` | 2.2064 | 0.3575 |
| `maxpool2x2` | 7.4866 | 2.9697 |
| `transformer_block` | 0.8469 | 1.1825 |
| `cnn_lenet` | 2.6658 | 0.6735 |

两列**不可互相比较**，各自只跟自己比：参考后端是 numpy，存在的意义是语义真值，
不是速度。`matmul_256` 两列几乎相等，是这个尺寸下 launch 与派发开销掩盖算力的典型；
`transformer_block` 也只差 1.4 倍——图小、kernel 多，固定开销占比高。这两行恰好说明
"哪个用例该快、快多少"本身是有信息量的：如果某天 `conv2d_3x3_pad1` 的 CUDA 数字
掉到 1 ms 以上，那不是噪声，是卷积路径出了问题。

**为什么性能门禁不进云端共享 runner**

这一条写进了 `ci.yml` 的注释里：把 25% 的门禁放到抖动本身就可能超过 25% 的机器上，
结果只能是"时绿时红"，几周之后没人再看它，门禁等于被关掉。所以拆成两个 job：

- `cpu`：ubuntu-latest，Python 3.10 / 3.11，跑全部单元测试 + 参考后端 conformance。
  正确性对机器不敏感，云端免费且可靠。
- `gpu`：`[self-hosted, gpu]`，**手动触发**（`workflow_dispatch`）——避免每次 push
  都占用自建机器，也避免因为没有 runner 让流水线长期红着。跑测试 + CUDA conformance
  + `benchmarks.run --check benchmarks/baseline_cuda.json --max-slowdown 0.25`。

**门禁自己也要被验证**：`tests/test_benchmark.py` 有 11 条用例，既测"变慢会被抓"
（10 ms → 13 ms 判回归），也测"阈值内不误报""新用例不算回归""后端不一致被拦"，
还有两条用假后端验证：第一轮很吵、第二轮正常时取后者；首轮超阈值、重测正常时不判
回归，而持续变慢仍判回归。没有这组测试，`check_regression` 完全可能是一个永远返回
"通过"的函数。

**仍然缺的**：GPU 基线只有一张卡（换卡就要重采）；只测延迟，没有吞吐与显存趋势；
没有"定位是哪次提交变慢"的自动二分；GPU job 尚未在真实 self-hosted runner 上跑过。
另外，重测只能挡住**瞬时**干扰，挡不住**持续**降频（负载下整机慢一倍时，两次都慢，
一定会被判回归）。所以阻断式门禁必须放在状态可控的机器上；本地那份基线只适合
看趋势，不适合当红线。

### 5.16　分块卷积：把 im2col 的临时张量去掉

5.12 的 im2col 路径有一个结构性代价：`(N, K, HO*WO)` 的列矩阵，大小是输入的重叠
倍数——3x3 卷积就是 9 倍。`benchmarks/conv_compare.py` 量过：`(4,64,56,56)*(64,64,3,3)`
这一层要 27.6 MB 临时缓冲，只是为了把它立刻读一遍。这一节交的是第二条实现路径：
**分块直算，不物化列矩阵**，并按形状自动选择。

**kernel 的形状**

一个 block 负责「一个 batch、一个输出行、W 方向 32 列、32 个输出通道」，
线程布局 `(32, 8)`：

- 每个线程算 **4 个输出通道 × 1 个空间位置**（ROC=4）。一次全局读喂 4 个累加器，
  输入读次数直接降到 1/4。
- 权重按 **CB 个输入通道** 分批 staging 到 shared memory（`TOC*CB*KH*KW` 个 float）。
  shared 只随核面积增长、不随输入通道数增长，所以深通道卷积不会把预算撑爆；
  分批还顺手把 `__syncthreads()` 的次数从 C 次摊薄到 C/CB 次。
- 窗口完全落在图内时走 **无边界判断的快速路径**，只有贴边线程走慢路径。
  stride=1 的 3x3 里 95% 以上的线程走快速路径。
- 越界线程不 return，而是继续参与 staging 与 `__syncthreads()`——提前退出会让
  块内同步死锁。这是分块 kernel 最容易写错的一处，注释里写明了。

**选择启发式**

`conv_impl` 三档：`auto`（默认）、`im2col`、`tiled`。`auto` 有两条门限：
K > 32、或输出元素少于 4096 时走 im2col。第二条是固定开销问题（staging 与同步
抵不过计算本身）；第一条来自下面的实测。

**实测：分块核只在 K 很小时占优**

RTX 2080 Ti，median ms，`benchmarks/conv_compare.py --rounds 5`：

| 形状 | K | tiled | im2col | 比值 | im2col 列矩阵 |
|---|---|---|---|---|---|
| `3x3_c3` (2,3,64,64)→(16,3,3,3) | 27 | 0.3266 | 0.3722 | **1.14x** | 0.84 MB |
| `1x1_c32` (2,32,32,32)→(64,32,1,1) | 32 | 0.3413 | 0.3936 | **1.15x** | 0.25 MB |
| `3x3_c8` (4,8,64,64)→(32,8,3,3) | 72 | 0.7388 | 0.6230 | 0.84x | 4.50 MB |
| `3x3_c16` (4,16,64,64)→(32,16,3,3) | 144 | 0.9821 | 0.9270 | 0.94x | 9.00 MB |
| `3x3_c32` (4,32,64,64)→(32,32,3,3) | 288 | 1.2959 | 0.9137 | 0.71x | 18.00 MB |
| `5x5_c16` (4,16,64,64)→(32,16,5,5) | 400 | 1.5216 | 0.8539 | 0.56x | 25.00 MB |
| `1x1_c256` (2,256,28,28)→(256,256,1,1) | 256 | 0.8950 | 0.6787 | 0.76x | 1.53 MB |
| `3x3_c64` (4,64,56,56)→(64,64,3,3) | 576 | 2.8311 | 1.3030 | 0.46x | 27.56 MB |
| `3x3_c128` (2,128,28,28)→(128,128,3,3) | 1152 | 1.6611 | 0.6147 | 0.37x | 6.89 MB |

结论写清楚，不做美化：**只有 K <= 32 时分块核更快（+14%）；K 一到 72 就落后，
K 大时差 2~3 倍。** 原因是分块核没有做空间方向的寄存器分块——每个线程仍然对每个
输出位置发 K 次全局读，指令吞吐与访存延迟压不过 cuBLAS 那种把整块输入搬进
shared memory、一次 FMA 服务多个输出的 GEMM。想赢，下一步得把「一个线程算多个
空间位置、窗口重叠部分复用」做进去；那是另一刀，没有它，K 大时不可能追上。

所以 `auto` 的阈值卡在 32：常见的 3x3 深层卷积继续走 im2col，只有第一层与浅通道
1x1 吃到分块核的收益。**这条路径的真正价值在显存，不在速度**——需要的时候可以
`CudaBackend(conv_impl="tiled")` 强制走它，省掉那张 `(N,K,HO*WO)` 的临时表；
速度代价就是上表的比值。这里不做"两个都好"的粉饰：选型有明确代价，才配叫选型。

**这次调参的两次失败也记下来**：第一版把整个 K 的权重一次性 stage 到 shared
（`TOC*K` floats），K=576 时只放得下 16 行、每 SM 挤进一个 block，`3x3_c64`
实测 3.86 ms——比 im2col 慢 3 倍，根因不是算法而是占用率；第二次把 ROC 从 4 提到 8，
`3x3_c64` 又回到 3.16 ms，因为 block 线程数同样减半。两条都留在注释里，免得
后来者再走一遍。

**正确性**：`tests/test_cuda_backend.py` 的 `TestCudaConvTiled` 用与 im2col 路径
**同一组**边界参数（padding / stride / dilation / 非方形核 / 无 bias / 非对称参数 /
O 不是 tile 整数倍）强制走 tiled，与参考后端和 im2col 三方逐点对拍；另有
`TestConvTilePlan` 在本地（无 GPU）单测 OC tile 的预算边界。分块核里"越界线程
必须继续参与 `__syncthreads()`"这条约束，就是靠 `O=70`（非 32 整数倍）与边界
padding 两个用例守住的。

### 5.17　自动微分：反向就是再构一张图

第 7 章的两条记录都指向同一个问题：自动微分放在语义 IR 层，还是新增一层。
这一节交的决定是**语义 IR 层**，理由、代价与第一版的边界一并写下。

**决定一：不新增 IR 层**

`backward(graph)` 返回的不是某种"反向 IR"，而是一张**同构的普通图**：前向节点
原样重放，反向节点由每个算子的 VJP 规则组合出来。所以它照样过 `verify`、照样能
`dumps` / `loads`、照样能交给任意后端执行——"一套 IR"这条主线不为 AD 开特例。

代价是反向图**重新计算前向**（中间值不靠外部喂）。两层 MLP + ReLU 的前向是 5 个
节点，反向图 23 个（含重放的 5 个）。换来的是自包含：一张图能单独导出、单独执行、
单独与 torch 对拍，不需要"调用方先把激活存好再喂进来"。检查点与重算最小化是后续
优化，不是第一版的前提——先正确，再省。

**决定二：规则挂在算子上，不在引擎里写分支**

`Op` 增两个字段：`vjp`（VJP 函数，`None` 表示无规则）与 `nullary`（零输入算子）。
引擎 [grad.py](file:///g:/Cloumi/aetherflow/aetherflow/ir/grad.py) 只做三件事：
逆拓扑序、cotangent 累加、缺规则时报错。"这个算子怎么求导"的知识属于算子自己
（[grad_rules.py](file:///g:/Cloumi/aetherflow/aetherflow/ir/grad_rules.py)，26 条规则）。
因此 SNN 的替代梯度（surrogate gradient）只是**替换某个算子的 `vjp`**，不是给引擎
塞 if/else，也不是为 SNN 单开一条 AD 路径——这是 1.3 里把 AD 排在 SNN 之前的全部理由。

两种"没有梯度"被严格区分，不能混成一种：

- `Op.vjp is None`：该算子不可微，`backward` 抛 `NoGradientError`，不静默跳过；
- 规则返回 `None`：该输入不接收梯度（如 `step` 的输入、`softmax` 的 `scale`），合法。

**决定三：反向只用已有语义，两个例外**

绝大多数反向是前向算子的组合：`mul` / `reduce_sum` / `transpose` / `reshape` /
`sqrt` / `broadcast_to` / `sigmoid` / `step`。没有"反向专用内核"这一层。

两个例外是卷积与最大池化：`dx` 要散射、`dw` 要对 batch 归约，用逐元素算子组合
表达不出来。它们被定义为**语义完整的 PORTABLE 算子**——`conv2d_input_grad`、
`conv2d_weight_grad`、`maxpool2d_backward`，各有独立参考实现（einsum 按偏移逐份
散射；四重循环的"笨"实现，并列最大值取第一个）。它们不是"某个后端的反向内核"，
参考语义依然是唯一真值。

**为了可移植而做的两处妥协**

1. **广播反向一律 `transpose + reduce_sum(-1)`**。CUDA 后端的 `reduce_sum` 只实现
   了最后一维；规则里若直接写 `reduce_sum(axis=0)`，参考后端能跑、CUDA 跑不了。
   所以多轴归约统一先置换到最后、归约、再置换回来（`sum_axes` / `sum_to`）。
   规则只有一份，两个后端都成立。
2. **需要归约的广播反向要求静态形状**。合并/拆开若干维要靠 `reshape`，而合并符号维
   需要 `reshape` 接受形状表达式，IR 现在不接受（与 conv 拒绝符号空间维同源的问题，
   5.53 只解决了卷积/池化的派生维）。含符号维时显式抛 `NoGradientError`，
   而不是留到执行期报形状不匹配。

**新增 `constant`：零输入节点**

`1/D`、`eps`、`1.0` 这些常数原本无处可放。方案不是"再多一个图输入"——那会让反向图
依赖外部投喂，自包含性就没了；而是新增零输入算子 `constant`（`Op.nullary=True`，
attrs: `dtype` / `shape` / `value`）。`add_node` 原先禁止零输入，这一处放开，但只对
`nullary` 算子放开。

**一个被序列化倒逼出来的约束**

`from_dict` 先恢复**全部** inputs，再重放 nodes。因此种子梯度输入必须**先于所有
节点**声明，否则 SSA 名字的顺序与写出去时不同，读回来引用就错位（表现为"节点引用
了未定义的值"）。这条约束由 `test_backward_graph_serializes` 钉住——第一版就是踩了
这个坑才发现的，不是预先设计出来的。

**验证**

| 判据 | 覆盖 |
|---|---|
| 中心差分数值梯度（f64） | 逐元素、广播、matmul、softmax、layernorm、reshape / transpose / slice / pad、conv2d、maxpool2d |
| `torch.autograd` 交叉验证 | MLP、conv2d、maxpool2d 三组（装有 torch 时执行，否则跳过） |
| 反向图合规性 | `verify` 通过、JSON 往返后语义不变、输出按 `d<输入名>` 命名 |
| 训练闭环 | 手写 SGD 迭代，loss 下降到阈值；再与中心差分逐参数对齐 |

共 37 个 AD 测试。全量：本地 247 个（含 80 个 CUDA 跳过），远程 GPU 247 个全绿。
conformance 从 31 个用例扩到 44 个（新增 9 个前向算子用例 + 4 个反向用例），
参考后端 44 executed / 0 refused / 0 failed，29 个被声明算子全部覆盖一次成功执行。

**CUDA 侧：反向能跑，卷积反向当时明确拒绝**

CUDA 后端承接 25 个算子，反向图用到的逐元素、归约、形状算子都在其中，所以
MLP / layernorm / softmax 的反向图**在真机上跑过，并与参考后端逐点对拍**。
当时未承接的 4 个是 `pad` 与三个反向算子（`conv2d_input_grad` / `conv2d_weight_grad` /
`maxpool2d_backward`），在 conformance 里落到 `refused`——这是能力协商的正确表现，
不是失败。要补的是内核（散射与并行归约），不是语义。
**补记：这 4 个内核已在 5.20 之后的批次补齐，见 5.20 末段；上面这段保留为当时的记录。**

**这一块明确没有做的**

- 二阶导数：对反向图再求导，`conv2d_*_grad` / `maxpool2d_backward` 会得到
  `NoGradientError`，而不是静默的二阶零。
  **补记：5.198 已提供 `hessian_vector_product` 入口与解析 / 差分 / Newton
  三条判据；上面的算子限制仍然成立（它们本来就没有 VJP）。**
- 检查点 / 重算最小化：反向图现在必然重算整个前向。
  **补记：5.75 已补上分段重算（`train/recompute.py`），此条已失效。**
- 反向图的专用优化：它走的是和普通图同一条 pass 管线，没有为"重放的前向"做
  DCE 之外的裁剪，也没有为反向做内存规划（捕获期的 arena 仍按整图算）。
- `pad` 的 CUDA 内核（前向也没有），因此含 `pad` 反向的图在 CUDA 上被拒绝。
  **补记：`pad` 与三个反向内核已在 5.20 之后的批次补齐，此条已失效。**

### 5.18　SNN：把时间展开进图，把不可导留给一条 VJP

1.3 的第 5 项里有 SNN。它的难点不在"多发几个脉冲"，而在两件反直觉的事：
**发放不可导**，以及**有状态**。这一节交的第一版没有为它们新增任何 IR 机制。

**时间不是新原语，而是展开**

静态 IR 里没有循环，所以 T 个时间步就直接**展开成 T 份子图**，膜电位作为显式
数据依赖从上一步传到下一步。展开后的图与普通图完全一样：verifier、JSON 往返、
CUDA Graph 捕获、任意后端都照常适用。`build_snn` 只是构图期的循环，
不是 IR 的新节点类型。

代价是图会随 T 线性变大（T=4、每步 10 个节点量级）：一个 T=4 的小 SNN 前向
42 个节点。这与 5.17 反向图"重放前向"是同一笔账——用图规模换表达简单。

**发放不可导：`spike` 与 `step` 是两个算子**

`step` 的反向是 `(None,)`：它出现在 `relu` 的反向里，本身不该被继续求导。
SNN 的发放若也用 `step`，梯度会在第一个脉冲处断掉，网络一点也学不到。

所以新增 `spike`：**前向与 `step` 逐位相同**（`x > threshold → 1`），只有反向不同
——它挂一条替代梯度（surrogate gradient）。选三角窗：

```
dL/dx = dL/ds * relu(1 - |x - thr| / alpha) / alpha
```

峰值 `1/alpha` 落在阈值上，两侧线性衰减，`|x-thr| >= alpha` 时为 0。它是**定义**，
不是"对阶跃的近似"——和直通估计器（STE）定义量化算子的梯度是同一类东西。

一个具体的坑：`|d|` 用 `sqrt(d*d)` 表达最自然，但反向是 `g / (2*sqrt(d*d))`，
当 `x` 恰好等于阈值时算出 NaN——而那恰恰是 SNN 最常停留的位置。所以改用
`relu(d) + relu(-d)`：在 0 处梯度取 0，不会污染整张反向图。
`test_gradient_finite_at_threshold` 就是钉这一点的。

**LIF 的方程**

```
v_pre  = decay * v + x
s      = spike(v_pre, threshold, alpha)
v_next = v_pre - threshold * s        # reset="subtract"；reset="none" 时 v_next = v_pre
```

`decay` 限制在 `[0, 1)`：等于 1 表示完全不泄漏，电位会无界累积，不是"更强的
记忆"，而是数值上不可用。`LIFSpec` 在构造时就拒绝这种配置。

**读出头：不用 `reduce_sum` 沿时间轴**

最常用的读出是平均发放率 `sum(spikes)/T`。这里逐元素相加，而不是 `reduce_sum(axis=0)`
——CUDA 的 `reduce_sum` 只实现最后一维，而时间维在最前面。与 5.17 广播反向用
`transpose + reduce_sum(-1)` 是同一条可移植性约束，只是这次用更便宜的加法绕开。

**验证**

| 判据 | 覆盖 |
|---|---|
| 前向等价 | `spike` 与 `step` 在同一组输入上逐位相同 |
| 梯度数值 | 三角窗的峰值、窗内线性、窗外为零（`atol=1e-12`，不是"大概对"） |
| LIF 手算 | 发放/重置、未发放、`reset="none"` 三种情形与手算逐值对齐 |
| 时间展开 | 两步手算、零输入、初始膜电位、符号时间维拒绝 |
| 图契约 | 前向与反向都过 `verify` 与 `verify_portable_only`；两者都能 JSON 往返 |
| 可训练性 | 40 步 SGD 让损失降到原来的 90% 以下；梯度非零且有限 |
| 外部参照 | 手写 PyTorch `autograd.Function`（与 spikingjelly 同一种写法）实现同一套 LIF 与三角窗，参数梯度与 IR 路径一致 |
| 跨后端 | CUDA 上 spike 前向/反向与 T 步 SNN 前向/反向都与参考后端逐点一致 |

新增 24 个 SNN 测试；全量 273 个（本地 191 通过 / 82 跳过，远程 GPU 全绿）。
conformance 从 44 扩到 45 个用例、30 个算子，参考后端 45 executed / 0 refused。
CUDA 当时承接 26 个算子（`spike` 复用 `step` 的 kernel，反向由 IR 规则生成，后端无感知）；
5.20 补齐五个新算子与两个反向算子后为 33 个，再补 `pad` 与三个卷积/池化反向
内核后为 37 个（只剩两个 scan 算子仍明确拒绝；5.35 已补齐为 45 个）。

**这一块明确没有做的**

- **事件驱动的稀疏 kernel**：稀疏事件表示与事件内存池已由 5.35 补上（数据侧，
  CSR 变体 + 可复用的下标缓冲 + 只读事件的发放率）；仍未实现"只对发放的神经元
  做计算"的稀疏 matmul / 卷积，那需要后端 kernel，不属于表示层。
- **符号时间维**：整个输入形状必须静态。原因不是时间维本身，而是初始膜电位是
  一个 `constant` 节点，它的形状写在属性里，无法随符号维在运行期变化。要支持
  变长序列，得先让常量形状可符号化，或改用图输入承载初值。
- **编码器**：rate / latency / delta 编码都没有封装成模型构件，输入直接当电流用。
- **自适应阈值、多房室、突触延迟**：都只在 `LIFSpec` 之外，需要新的状态量。
- **反向图规模**：T 步展开 + 5.17 的重放前向叠加，反向图比前向大一个数量级
  （T=4 时 42 → 162 个节点）。检查点在这里比在普通 MLP 上更有必要。

### 5.19　SSM：只把递归做成算子

1.3 第 5 项的另一半。Mamba 的 selective scan（S6）看起来是一整套新机制，
实际上只有一处无法用已有语义表达：**线性递归本身**。

**什么进算子，什么不进**

```
a_t = exp(dt_t * A)                     exp / mul（已有）
b_t = dt_t * B_t * x_t                  mul（已有，靠广播）
h_t = a_t * h_{t-1} + b_t               scan_linear   ← 唯一的新算子
y_t = sum_n C_t[n] * h_t[..., n]        mul + reduce_sum(-1)（已有）
y_t += D * x_t                          mul / add（已有）
```

递归必须成为算子，有两个独立原因。展开成 T 个节点会让图随**输入长度**线性膨胀
（5.18 为 SNN 付过同样的账，但那里 T 是模型超参，这里 T 是数据）；而连乘 a 的
闭式解在 T 长时下溢——`prod a_t = exp(sum dt_t * A)` 在 1000 步后是 exp(-1000)
量级，f32 早就是 0。

所以 `scan_linear` 的语义就是最短的那一行：`h_t = a_t * h_{t-1} + b_t`，
`a / b` 形状 `(T, ...)`、`h0` 形状 `(...)`、输出 `(T, ...)`，时间维**必须静态**。
`build_selective_scan` 只是把上面五行拼出来，不引入任何 SSM 专用节点。

**伴随也是一次递归**

反向同样是线性递归，同样不能展开：

```
dh_t = dy_t + a_{t+1} * dh_{t+1}
```

它作为第二个语义算子 `scan_linear_backward` 存在。这是 5.17 那条原则的延续：
反向规则尽量由**已有语义**组合出来；反过来，如果一条反向规则需要新的递归结构，
就把它定义成一个普通 PORTABLE 算子，而不是给 AD 开后门。

三个输入的梯度：

```
db   = dh
da_t = dh_t * h_{t-1}        # h_{-1} = h0
dh0  = a_0 * dh_0            # h_0 = a_0 * h0 + b_0
```

`h_{t-1}` 用 `pad + slice + add` 平移得到（`[h0, h_0, ..., h_{T-2}]`），不引入 concat。
`dh0` 漏掉 `a_0` 是这条规则唯一容易写错的地方，第一次实现就踩了：中心差分当场
抓住（现在由 `test_dh0_is_first_adjoint` 钉住）。这类错误不会被"能跑通"发现。

**CUDA 实现（5.35 补齐）**

第一版明确拒绝：在拿到可用的 GPU 机器之前，交付一个未测的扫描 kernel 比拒绝更糟。
5.35 补上两个内核——一个尾元素一个线程、沿 T 串行，并行度来自尾维而不是时间维，
前向与伴随共用同一套扁平索引布局（详见 5.35）。

**验证**

| 判据 | 覆盖 |
|---|---|
| 前向 | 与逐时间步的 numpy 循环一致（含 T=1 闭式、初始状态只进第一步） |
| 伴随 | `scan_linear_backward` 与手写倒序循环逐位一致 |
| 梯度数值 | 对 a / b / h0 的中心差分，f64、rtol=1e-6 |
| 模型前向 | selective scan 与 numpy 朴素实现一致（含 (T,B,D)、跳连、非零 h0） |
| 模型反向 | 与 torch 显式展开循环 + autograd 的 6 个梯度全部一致 |
| 图契约 | 前向与反向都过 `verify` 与 `verify_portable_only`，都能 JSON 往返 |
| 数值稳定 | T=64 时输出有限且与朴素实现一致 |
| 边界 | 符号时间维、x/dt 形状不匹配、秩 < 2、dtype 不一致、h0 形状错误全部显式拒绝 |

新增 24 个 SSM 测试；全量 297 个（本地 215 通过 / 82 跳过）。conformance 从 45
扩到 47 个用例、32 个算子，参考后端 47 executed / 0 refused；CUDA 当时仍是 26 个算子，
两个 scan 用例在它上面记为 refused（5.20 之后为 37 个，当时 scan 仍明确拒绝；
5.35 补齐为 45 个）——能力声明诚实，而不是静默降级。

**这一块明确没有做的**

- **符号时间维**：与 5.18 同因——常量形状与切片位置都写在图里。
- **并行扫描（Blelloch / chunked scan）**：参考实现与 CUDA 内核都是最朴素的顺序
  循环，语义正确但 O(T) 串行；5.35 的 CUDA 内核把并行度放在尾维上，T 很长时延迟
  仍随 T 线性增长。分块扫描需要两遍通信，留到有实测需求时再做。（**5.128 已补**：
  两趟分块扫描 + 末状态专用内核，显式开关、默认关闭。）
- **门控 / 卷积前置 / 归一化**：5.19 时只是单层 selective scan；门控、因果深度
  卷积与 RMSNorm 已在 5.22 的 `build_mamba_block` 里补齐，剩下的是**多层堆叠**
  与论文里的低秩 dt 投影。

---

### 5.20　补齐算子：GELU / RMSNorm / 平均池化 / 拼接 / 词嵌入

**动机**：模型构件的缺口先暴露成算子缺口。在写 5.22 的六个构件时，有五个结构
反复需要"某个不存在的东西"——如果每个都在模型层用 `exp/log/tanh` 现拼，语义会
散落在调用方，反向也要各写一遍。这一节把它们收敛成算子。

**五个前向算子 + 两个反向语义算子**

| 算子 | 语义 | 为什么必须精确成这样 |
|---|---|---|
| `gelu` | `0.5x(1+tanh(√(2/π)(x+0.044715x³)))` | 本式与 PyTorch `approximate='tanh'` 逐点一致，是 GPT/BERT 实际使用的定义。当时不引入 `erf`；5.39 为承接 PyTorch 默认的精确 GELU 补了 `erf`，`gelu` 本身仍是 tanh 近似（两种近似差 ~1e-3，由前端按 `approximate` 分派） |
| `rmsnorm(x, gamma)` | `x/√(mean(x²)+eps) * gamma` | 比 LayerNorm 少一次均值归约与两个参数，是 LLaMA / Mamba 的默认归一化；`gamma` 必须能广播到 `x` 的尾维 |
| `avgpool2d` | NCHW，分母固定为 `kh×kw` | 等价 `count_include_pad=True`。**不提供"按有效元素数平均"的开关**：那会让 padding 边界的语义与 maxpool 不对称，也使反向的除数依赖位置 |
| `concat` | 沿 `axis` 拼接 | 拼接维**必须是静态整数**：输出形状在构图期要能写出来。非拼接维可以带符号 |
| `embedding(table, ids)` | `ids` 逐元素查表 | `ids` 必须是 i32/i8（浮点下标是错的，直接拒绝）；词表大小（`table.shape[0]`）静态 |
| `avgpool2d_backward` | `dy/(kh·kw)` 散射累加 | 重叠窗口自然叠加；`input_shape` 是必填属性——反向图必须自包含 |
| `embedding_backward` | `np.add.at` 累加 | 同一个下标出现多次时梯度要**累加**，不能覆盖；这是手写反向最容易错的一处 |

**验证**

每个算子都有 infer + reference + vjp + conformance 用例，另有
`tests/test_ops_extra.py` 的 14 个用例做数值对拍：

| 判据 | 覆盖 |
|---|---|
| 前向 | 与独立 numpy 实现一致（`avgpool2d` 用手写循环，不用 reshape 技巧） |
| 梯度 | f64 中心差分，`gelu` / `rmsnorm` / `avgpool2d` / `concat` 全部对拍 |
| 守恒 | `avgpool2d` 反向的梯度总和 = 输入窗口数（4 个窗口各分 1/4，总和为 4） |
| 边界 | 浮点 `ids`、符号拼接维、非拼接维不一致、多轴 rmsnorm 全部显式拒绝 |
| 覆盖 | `test_covers_all_operators` 仍要求 `set(report.covered_ops) == set(OPS)` |
| CUDA 内核 | 七个算子全部原生实现：`gelu` 复用一元模板；`rmsnorm` 与 `layernorm` 同为"一行一个 block"，但只做一次 `mean(x²)` 归约；`avgpool2d` 与 `maxpool2d` 共用索引、分母固定为 kernel 面积；`concat` 按输入分段 launch strided copy（`(outer, axis, inner)` 视角，无需下标表）；`embedding` 是 gather，`avgpool2d_backward` / `embedding_backward` 用 atomicAdd 散射累加 |

CUDA 侧新增 23 个用例（本地无卡整组跳过），另有 3 个纯索引数学用例
（`TestIndexMath`）不需要 GPU：它们用 numpy 复刻内核里的同一套索引公式，
把"段偏移 / (i, j) 拆分 / 重叠散射"这三类错值错误钉在本地。

真机验证（RTX 2080 Ti / CUDA 12.4 / cupy 14.2）：`verify_cuda.py` 报告
**44 executed / 11 refused / 0 failed**（当时 33 个算子，见本小节末段的后续批次），
九个额外边界用例全部通过；七个新算子
的最大误差都在 5e-07 以内（`gelu` 4.8e-07、`rmsnorm` 4.8e-07、`avgpool2d` 6.0e-08、
`avgpool2d_backward` 0、`concat` 0、`embedding` 0、`embedding_backward` 0）。

**再补一批：`pad` 与三个反向内核**

上一批之后 CUDA 只剩"能做前向、不能做反向"的 CNN 断点。补齐四个内核后，
CNN 训练在 GPU 上不再断链：

| 算子 | 写法 |
|---|---|
| `pad` | 复用 `concat_copy` 的 strided copy（同一段源码、同一缓存 key），输出先清零，无需新模板 |
| `maxpool2d_backward` | 每线程一个输出格，在窗口内找 argmax（并列取行优先第一个，与参考语义一致），再 `atomicAdd` 散射 |
| `conv2d_input_grad` | **gather** 形式：每线程独占一个 `dx` 元素，反查 `(ih+PH-ki*DH) % SH == 0` 的贡献，无重叠写 |
| `conv2d_weight_grad` | **gather** 形式：每线程独占一个 `dw` 元素，遍历 `N/HO/WO` 累加 |

卷积反向刻意不用 atomicAdd：输出元素互不重叠，gather 既确定性又省掉重复读
`dy` 的代价；池化反向天然重叠（窗口共享输入），才用散射。

补完后真机 `verify_cuda.py`：**48 executed / 7 refused / 0 failed**，
`tests.test_cuda_backend` 114 个用例全过，九个边界用例仍全部通过。
剩余 7 个 refused 全是能力声明内的形状/精度边界：两个 scan、`matmul_partial_batch`、
`softmax_axis0`、`layernorm_axis0`、`reduce_sum_axis0`、`f64_input_relu`。

**明确没做的**

- **CUDA 上的越界检查与 CUDA Graph 不共存**：`embedding` / `embedding_backward` 的
  越界标志要读回主机才能转成 `BackendCapabilityError`，而捕获期不允许 D2H 同步。
  因此捕获的图里这条检查降级为 clamp，只有预热轮与普通执行会真正报错。
  这是 CUDA Graph"执行路径固定"的直接后果，已在代码里写明，不是隐藏的静默降级。
- **`avgpool2d` 的 `ceil_mode` / `count_include_pad=False`**：语义开关一旦进来，
  反向规则就要跟着分叉；当前先固定一种。
- **`embedding` 的 padding_idx 与梯度缩放**：属于模型侧策略，不进语义层。
- **`rmsnorm` 的可广播 gamma**：内核只接 `(cols,)`；更宽的形状请在前端插一次
  `broadcast_to`，不为它给内核加 stride 参数。
- **`pad` 只承接 IR 的语义**：沿单个指定维两侧补零、补的值恒为 0。通用的
  "多维 + 任意常量填充"不在语义层，内核也就没有。

---

### 5.21　训练闭环：参数、优化器、损失

**动机**：5.17 交付了 `backward`，但"能求导"不等于"能训练"。在 5.20/5.22 之前，
仓库里没有任何一处**真正把参数更新到收敛**的代码——所有测试都在验证单步梯度。
这是一个验收缺口，不是功能缺口：没有它，"模型构件可训练"这句话没有证据。

**四步循环，四段代码**

```
参数初始化 → 前向 → 反向 → 优化器更新
```

| 模块 | 职责 | 关键约束 |
|---|---|---|
| `train/params.py` | `ParameterStore` + 初始化策略 | 参数是**图输入**但不是数据；`from_graph(exclude=...)` 必须显式排除 `x`/`ids`/`mask`，不自动猜。默认策略按名字与秩选：秩≥2 → Xavier；以 `b`/`bias` 结尾 → 零；其余秩 1 → 一 |
| `train/optim.py` | `SGD`（动量 / Nesterov / weight decay）、`Adam` | 只接收 `dict[str, ndarray]`，**原地更新**；参数缺梯度直接 `KeyError`，不补零。Adam 含偏差修正——不修正时第一步的实际步长会小一个量级 |
| `train/loss.py` | `mse` / `softmax_cross_entropy` / `l2_penalty` | 损失是**图内节点**，梯度由 `backward` 统一给出，不另写一套手推公式 |
| `train/loop.py` | `Trainer`（5.66 补） | 反向图建一次；种子名与形状、多输出图的零种子由它负责。要求静态形状——反向图的归约 vjp 需要运行期形状表达式，符号维下构不出来。`device_state=True` 时参数与梯度留在显存（5.84），`store` 需 `sync_to_host()` 才更新 |

**两个实现细节**

1. **初始化的 `seed` 与参数名 CRC 混合。** 全图共用一个 rng 时，模型结构改一处，
   后面所有参数的初始值都会漂移，实验就无法复现；按名字派生让"改一层不影响
   另一层"。
2. **`mean` 的除数。** 缺省用 `type.numel()`，含符号维时直接报错——因为
   `reduce_sum` 只归约已知轴，元素总数是常量，必须静态。这与 RNN/SNN 要求静态
   形状是同一个根因。

**验证：不看形状，看收敛**

`tests/test_train.py` 的 19 个用例里，四个是真正的收敛判据：

| 目标 | 配置 | 判据 |
|---|---|---|
| 线性回归 | Adam，lr=0.1，400 步 | loss < 1e-6 |
| XOR | 两层 tanh MLP，SGD + momentum=0.9，lr=0.1，2000 步 | loss < 0.01 |
| 二次目标 | 梯度下降 | loss 单调下降 |
| 反向图契约 | 只构一次反向图 | JSON 往返后梯度逐位相同 |

XOR 最初用 lr=0.5 + momentum=0.9 发散成 NaN。先用解析梯度与中心差分对齐
（w2：数值 2.5968 vs 解析 2.5971），确认不是梯度错，而是步长过大——这类
"训练不收敛"必须能区分"规则写错"和"超参不对"，否则会去改对的代码。

**明确没做的**

- 学习率调度、梯度裁剪、梯度累积、EMA（已在 5.31 补齐）。
- 混合精度、参数检查点、数据加载器（已在 5.27 / 5.28 补齐；混合精度的"逐算子精度
  选择"由 5.90 的 `Autocast` 补上，不再是整图 f16）。
- 分布式 / 多设备。

---

### 5.22　模型构件：六类架构，一个新算子都没加

**动机**：5.19 结束时只有一个 Transformer Block 与单层 SSM。用户能问的问题很具体：
"这套 IR 到底能表达多少种模型？" 这一节用六个构件回答，**全部只用 5.20 之前的
语义加上那五个新算子**，没有为任何一个架构新增 IR 机制。

| 构件 | 文件 | 参数布局 / 结构 | 静态形状要求 |
|---|---|---|---|
| MLP | `models/mlp.py` | `w{i}(in,out)` + `b{i}`；激活可选 relu/silu/gelu/tanh | 无（batch 可符号化） |
| CNN | `models/cnn.py` | `conv_block`（3×3 same + relu）、`residual_block`（stride≠1 或通道变化必须有 1×1 投影）、`CNNSpec`（stem → conv+pool → 全局平均池化 → 线性头） | 图像空间维必须是整数（池化输出尺寸要写进类型） |
| RNN | `models/rnn.py` | RNN / LSTM / GRU 三种 cell，门序 `i,f,g,o`（LSTM）与 `z,r,n`（GRU），多层堆叠 | `T` 与 `batch` 静态（初始隐状态是零常量节点） |
| Mamba | `models/mamba.py` | rmsnorm → 输入投影切 x/z → **因果深度卷积**（pad+slice+mul+add 展开）→ silu → softplus(dt) → `scan_linear` → 门控 → rmsnorm → 输出投影 + 残差 | `T`、`batch` 静态；`d_conv` 小（3~4）时展开代价可忽略。流式形态见 5.99～5.104：`*_stateful` 把卷积窗口与 SSM 状态变成图输入/输出，`steps=1` 即 decode，状态可进存档、逐样本 reset、显存不随历史增长 |
| Transformer | `models/transformer.py` | 单块（序列维符号化）+ `TransformerSpec`（词嵌入 + 可学习位置编码 + 因果掩码 + N 层 + final LN + LM head） | `seq` 静态（位置编码与掩码内容随长度变化） |
| SNN | `models/snn.py` | 单层 `build_snn` + 多层 `build_snn_network`（linear + LIF），平均发放率读出，可选分类头 | `T`、`batch` 静态 |

**四个值得记录的设计决定**

1. **权重布局统一为 `(in, out)`。** 与 `x @ W` 一致，不是 PyTorch 的 `(out, in)`。
   混用会让形状检查"语义对、数值错"——CNN 的卷积权重仍是 PyTorch 的
   `(out, in, kh, kw)`，因为 `conv2d` 的语义就是这么定义的；这一点在
   `default_init` 的扇入/扇出里显式区分。
2. **Mamba 的因果卷积不新增算子。** 用 `pad + slice + mul + add` 展开 `K` 个抽头，
   `K=4` 时节点数远小于"新增 conv1d + input_grad + weight_grad + 后端实现"。
   代价是 `T` 必须静态，与 RNN/SNN 的理由相同。
3. **因果掩码用 `-1e9` 而不是 `-inf`。** softmax 对两者前向等价（概率 < 1e-300），
   但 `-inf` 参与 `0 * inf` 的反向会产生 NaN。有限的大负数在有界图里更安全。
4. **图输入必须全部声明在节点之前。** 这不是新约定，而是序列化格式的隐含前提：
   JSON 只记录节点顺序，`from_dict` 会先重建全部输入，靠"输入数 + 节点顺序"
   还原 SSA 名。`build_transformer` 最初先声明 `ids/tok_emb`、构完 `embedding`
   再声明 `pos/mask`，JSON 往返立刻报"节点引用了未定义的值 v2"。修法是调整声明
   顺序并把这条约束写进构件 docstring——它值得写进文档，因为**构图顺序是 IR
   JSON 契约的一部分**。

**验证：每个构件四条判据**

`tests/test_models_extra.py`（40 个用例）对六个构件逐一检查：

| 判据 | 做法 |
|---|---|
| IR 合规 | `verify` 与 `verify_portable_only` 均无 error |
| 语义正确 | 与独立 numpy 实现数值一致（Transformer 用 2 层；Mamba 含 `D_skip` 跳连；SNN 用多层 LIF 朴素循环） |
| 可训练 | `backward` 展开反向图并过 verifier；MLP 用中心差分抽查，XOR 用 `ParameterStore + Adam` 训练到 loss < 0.05 |
| 可序列化 | `dumps` → `loads` 后算子序列、输出别名、执行结果逐位相同 |

另外两条**结构性**检查比数值更值钱：

- **Transformer 的因果性**：改最后一个位置的 token，前面位置的 logits 必须
  **逐位不变**。掩码加错位置（例如加在 softmax 之后）会被这条抓住。
- **Mamba 卷积的因果性**：改未来位置不影响过去位置。

**明确没做的**

- **CNN**：`ceil_mode`、`flatten` 之外的形状推理；与 cuDNN 的差距已量化（5.69：
  分块核慢 4～12 倍，im2col + cuBLAS 慢 5.5～12 倍）但尚未缩小
  （分组卷积与自适应 / 全局池化已在 5.34 补齐，转置卷积已在 5.51 补齐）。
- **RNN**：变长序列、双向、`pack_padded_sequence`。
- **Mamba**：多层堆叠、低秩 dt 投影（现在用逐通道 `dt_w*x+dt_b`）、并行扫描。
- **Transformer**：权重共享、KV cache、RoPE（位置编码是可学习参数）。
- **SNN**：泊松/延迟编码（`constant_current_encode` 只是复制输入）、稀疏事件表示。
- **训练侧**：所有构件的损失与优化器都需调用方自己接；没有 `Model.fit` 这类门面
  （存档与 DataLoader 已在 5.27 补齐，但循环仍要自己串）。
- **dropout / BN 已接入构件**（5.26）：CNN 用 `batch_norm` / `dropout` 开关，
  Transformer 逐层插 dropout。仍未接入的是 RNN / MLP 的 dropout，以及
  running stats 的持久化（属于训练循环，不在图里）。

---

### 5.23　dropout：随机也可以是纯函数

**动机**：5.22 的模型构件里没有 dropout，因为"随机"看起来与"图自包含、可复现"
冲突。真正的冲突只在随机源放错位置——把 RNG 挂在后端上，同一张图两次执行
就不同，conformance 的确定性检查直接失效。

**做法**：`dropout_mask(x, p, seed)` 把 seed 写进 attrs，掩码是
`(seed, 扁平索引)` 的纯函数。

1. **x 只提供形状与 dtype**，数值不参与计算，也不接收梯度（vjp 返回 None）。
   形状来自输入而不是 attrs，符号 batch 维因此也能走。
2. **SplitMix64，而不是 numpy 的 Generator**：目标是让参考实现与 CUDA 内核
   生成逐位相同的序列。PCG64 的位级实现绑死 numpy，CUDA 侧没有对应物；
   SplitMix64 只有整数乘法与异或，两边照抄同一个式子。
3. **取高 24 位映射到 [0,1)**：24 是 float32 的尾数宽度，两边不需要额外的
   舍入约定就能落到同一个 float。
4. **缩放写在算子里**：保留 → `1/(1-p)`，丢弃 → 0。训练图只需
   `mul(x, dropout_mask(x))`，不存在"别忘了再乘 1/(1-p)"这种图外的隐含约定。

**验证**

| 判据 | 覆盖 |
|---|---|
| 复现性 | 同 seed 两次执行逐位相同；不同 seed 掩码不同 |
| 只依赖扁平索引 | `(3, 8)` 与 `(24,)` 两张图的掩码序列完全一致 |
| 与数值无关 | 把 x 换成全 0 / 全 7，掩码逐位不变 |
| 边界 | `p=0` 全 1；`p=1` 与负数在构图期拒绝；符号 batch 维允许 |
| 梯度 | `y = x * mask` 的 `dx` 等于 mask，且与中心差分一致 |

CUDA 侧用 `assert_array_equal`（容差 0）对拍，不是 allclose——随机不是放松
校验的理由，反而因为没有外部参照，更需要把不变量钉死。

真机：conformance 59 用例 **52 executed / 7 refused / 0 failed**；
`dropout_p{0.0, 0.25, 0.9}` 与 `x*mask` 四个边界用例全部逐位通过。

**明确没做的**

- **推理态不接 dropout**：推理图里根本不出现这个算子，而不是靠 `p` 开关。
- **不输出布尔掩码**：直接给缩放后的 f32，省掉一次 cast 与一次 mul；
  需要原始 0/1 掩码的场合自行比较非零。
- **只有 f32**：与后端"只支持 f32"一致；混合精度的那一批再一起做。

---

### 5.24　BatchNorm：统计量、反向与 running stats 的边界

**拆成两个算子**

| 算子 | 语义 |
|---|---|
| `batch_norm(x, gamma, beta, axis=1, eps=1e-5)` | 训练态：沿除通道轴外的所有轴算均值/方差，再仿射 |
| `batch_norm_stats(x, axis=1)` | 返回 `(2, C)` 的 mean/var |

第二个算子存在的理由不是"拆得细"，而是 **running stats 的更新不属于张量语义**：
`running = (1-momentum) * running + momentum * batch` 是训练策略，写进 IR 会把
图和训练循环耦死，也让同一张图在不同 momentum 下不可复现。图只负责"算出来"，
更新留给调用方。

同理，**推理态 BN 不需要新算子**：用 running stats 的 BN 就是一次仿射，
现有 `sub / sqrt / mul / add` 足够。再加一个 `batch_norm_infer` 只是多一份要维护的语义。

**CUDA：一个 block 一个通道**

x 被看成 `(outer, channels, inner)`，三元组由 `axis` 折算。同一个内核因此覆盖
NCHW（axis=1，inner=H*W）、通道在首维（axis=0）、通道在末维等情况，不必为
每种布局写模板。

用**两遍归约**而不是 `sumsq - mean^2`：参考实现是 `numpy.var`（两遍法），公式
一致才能把对拍容差压下来；`sumsq - mean^2` 在"大均值、小方差"时会有灾难性
抵消，那时"精度不够"和"公式写错"就分不开了。

**反向不新增算子**

`dx / dgamma / dbeta` 全部由已有算子组合，归约走 `sum_axes`（transpose +
二维化 + 最后一维求和），所以 CUDA 的 `reduce_sum` 只支持最后一维也能承接整张
反向图。代价与卷积反向一致：**要求静态形状**——"每通道元素数"要作为常量参与
计算。符号 batch 的前向可用、反向抛 `NoGradientError`，这是明确边界而非静默降级。

验证：前向与手写 numpy 公式在 axis=0/1/2 三个位置逐点一致；反向的
`dx / dgamma / dbeta` 与中心差分一致；归一化后（gamma=1, beta=0）逐通道
均值 0、方差 1；`batch_norm_stats` 与 `np.mean/np.var` 一致；`stats` 接进损失
在 `backward` 时明确报错。

真机：`batch_norm` err=2.4e-07、`batch_norm_stats` err=6.0e-08；
`batch_norm_axis0`、`batch_norm_single_sample`（N=1 且 1x1，方差为 0 只剩 eps）、
`batch_norm_stats_axis0` 三个边界用例全部通过。

**明确没做的**

- **不更新 running stats**：见上，属于训练循环。
- **没有 `momentum` / `track_running_stats` 开关**：图里没有状态，就没有开关。
- **不做 SyncBN**：单卡单流，跨设备统计量不在这一版。
- **`batch_norm_stats` 没有 vjp**：把它接进损失会在 `backward` 时报错。

---

### 5.25　eager 前端：写起来像动态图，跑起来还是静态图

**动机**：一个反复出现的质疑是"动态图不是更好吗"。这个质疑要拆成两半：

- **作为执行模型**（define-by-run，边算边建图），本项目不能要。三件事只有静态图能给：
  构图期就能回答"目标后端接不接得住"（`BackendCapabilityError` 在构图时抛，而不是跑到一半）、
  反向图必须自包含可序列化（`backward()` 就是再构一张普通图）、CUDA Graph 捕获要求
  执行路径与形状固定（派发 800μs → 8μs，见 5.4）。
- **作为编程模型**（运算符直接写、标量随手加），本项目确实缺，而且值得补。

所以缺的是**前端手感**，不是执行模型。补前端，不动 IR——这也是 PyTorch 2.x 的路线：
用户写 eager，`torch.compile` 之后内部仍是静态图。

**做法**：`frontend/eager.py` 只有两个类。

- `Tensor`：构图期的张量代理，只持有 `Value` 与所属图，**没有数据，也没有设备**。
  重载 `+ - * / @` 与取负，全部转成 `add_node`；与 Python 标量运算时把标量插成
  0 维 `constant`（dtype 取另一侧）。
- `EagerGraph(Graph)`：`input` / `add_node` / `constant` / `output` 收发 `Tensor`。
  其余一切（算子名方法 `g.relu(x)`、`to_ir`、序列化、verifier）全部继承，一行未改。

```python
g = EagerGraph()
x = g.input("x", f32[4, 3])
w = g.input("w", f32[3, 5])
g.output(g.relu(x @ w + 1.0) * 0.5, "y")
```

**三条刻意的拒绝**

1. **只接受同图 Tensor 与 Python 标量**。numpy / cupy 数组不收：它们的形状不会进 IR，
   等于绕过类型推导与后端能力预检（符号维也失去参与的机会）。
2. **`bool(tensor)` 直接报错**。构图期没有张量值，`if x:` 无论返回什么都是猜的；
   猜错的分支在数值对拍里看不出来。要分支就写成两次执行，或在图外用 Python 值决定。
3. **`tensor[...]` 直接报错**。IR 的切分需要显式维与范围：`g.slice(x, dim=..., start=..., stop=...)`
   没有歧义，`x[...]` 有。

**验证**：`x * 0.5 + 1.0` 与手写 `add(mul(x, constant(0.5)), constant(1.0))` 产出的 IR
文本**逐字相同**——这是"零改动"的直接证据；JSON 往返后仍可执行；反向图与解析梯度对拍；
`verify()` 无 issue。CUDA 侧新增一条与参考后端对拍的前向 + 反向用例，重点覆盖
`f32[]`（0 维标量常量）在广播路径上的行为。

真机（RTX 2080 Ti / sm_75 / CUDA 12.9）：远程全量 461 个测试全绿（本地跳过的
124 个 GPU 用例在真机上执行，远端仅剩 17 个平台相关跳过），`verify_cuda.py`
conformance PASS、16 个边界用例全过。0 维常量的广播与反向未做任何后端特判。

**明确没做的**

- **不是 define-by-run**：`for i in range(n)`、`if flag:`（flag 是 Python 值）可以正常展开，
  但张量值不参与任何 Python 控制流。
- **不做 in-place**：IR 是 SSA，`x = x + 1` 生成新 Value，语义等价，但不是原地写。
- **不重载比较运算符**：`==` / `<` 仍是 Python 的恒等比较。元素比较应当作为显式算子引入，
  而不是把 `bool` 与张量混在一起。
- **不接管求导的方式没变，但形式上有了 `loss.backward()`**（5.77）：它仍然调用
  `backward(graph, wrt=[...])` 造一张自包含的静态反向图再执行一次，梯度写回
  `Tensor.grad`；不是 autograd 的 tape，也没有逐算子即时求值。

---

### 5.26　把 BN 与 dropout 接进构件：开关在构图期，状态在调用方

**动机**：5.23 / 5.24 只做到算子层，构件里一处都没用。缺口不在算子，而在
"训练图与推理图怎么分开"——这是设计问题，不是实现问题。把"模式"做进图里，
图就不再自包含；把"模式"做成运行期开关，同一次编译的图会有两条执行路径，
CUDA Graph 也就捕获不了。

**做法：开关是构造参数**

- `CNNSpec` 增加 `batch_norm: bool = False`、`dropout: float = 0.0`、`dropout_seed`；
  `BlockSpec` / `TransformerSpec` 增加 `dropout` 与 `dropout_seed`。
- `dropout=0` **不插节点**：训练图与推理图是同一份代码构建出的两张图，
  各自干净，图里没有"模式标志"。
- `batch_norm=True` 时每个卷积块变成 `conv → BN → relu`，参数表相应多出
  `*_bn_g` / `*_bn_b`（BN 放在激活之前，是 ResNet 的经典顺序）。
- Transformer 每层两处 dropout：注意力概率（softmax 之后）与 FFN 激活。
  seed 取 `dropout_seed + layer * 2 (+1)`，否则整栈会复用同一张掩码，
  把"逐层独立"悄悄退化成"逐层相同"。

**推理态 BN 不需要新算子**

`bn_infer(g, x, mean, var, gamma, beta, eps)` 用 `sub / div / sqrt / mul / add`
拼出一次仿射。running stats 是**图输入**而不是 attrs：换一组统计量不必重构图，
"什么时候更新"留给训练循环（与 5.24 的边界一致）。

**一个必须说清楚的取舍**

dropout 的 seed 写在图里，所以"同一张图同一份输入"永远得到同一结果。
训练时想要每步不同的掩码，就要用不同 seed **重构图**，并与参数更新交替进行。
这是静态图的代价，也正是它能被 CUDA Graph 捕获、能跨后端逐位复现的原因。

**验证**：CNN 开 BN 后与独立的 numpy 复算（显式循环卷积 + 两遍法 BN）数值一致；
BN 参数梯度有限；`dropout=0` 时图里没有 `dropout_mask`，开启时同图两次执行
逐位相同；Transformer 3 层恰好 6 个掩码且 seed 互不相同；`bn_infer` 与手写公式
一致。真机：461 个测试全绿，CNN（BN+dropout）与 2 层 Transformer（dropout）
两张构件级图在 CUDA 上与参考后端一致。

**明确没做的**

- **RNN / MLP 没有 dropout 开关**：等有真实用例再接，不提前铺参数。
- **不做 running stats 更新**：与 5.24 一致，属于训练循环。
- **不做 `train()` / `eval()` 模式**：图里没有状态，就没有模式切换。
- **dropout 概率是全局一个**：一个 spec 一个 p；要分层配置就拆开调用构件。

### 5.27　存档与批数据：续训要和没中断过一样

**动机**：5.21 的训练闭环能收敛，但跑完就没了。缺的两块都不是算法，而是
"一次实验能不能重来"：权重与优化器状态要能落盘再读回，数据要能按批喂且每轮
顺序可复现。这两件事做错时的表现都很安静——损失曲线正常，只是结果对不上。

**存档：参数属于模型，槽位属于训练过程**

- `save_checkpoint(path, store, optimizer=None, meta=None)` 写一个 `.npz`。
  键布局：`param/<名字>`、`slot/<槽位>/<名字>`、`optim/step`、`meta/<键>`。
- 参数与优化器状态同文件不同前缀。只存权重是合法场景（推理用），但这种存档
  再想续训会被明确拒绝——没有 `optim/step`，Adam 的偏差修正就无法恢复。
- 槽位不是可有可无的缓存：少一个动量数组，续训的轨迹就与没存过不同，而这种
  偏差在损失曲线上几乎看不出来。
- 加载是**原地覆盖**（`arr[...] = loaded`）。优化器持有的是 `store.arrays()`
  返回的同一批数组；替换对象会让它继续更新一份没人用的旧数组，训练看起来在跑、
  参数却永远停在存档值。
- 参数集合、形状、dtype 任何一处不一致都直接报错，不做"能恢复多少恢复多少"。

**DataLoader：洗牌必须可复现**

- `DataLoader(arrays, batch_size, shuffle=False, seed=0, drop_last=False)`
  按第 0 维切批，产出 `dict[str, ndarray]`，键与传入完全一致，可以直接交给
  `ParameterStore.feed(...)`。
- 每轮用 `default_rng(seed + 轮次)` 独立生成排列，不共享全局随机状态：全局 rng
  的下一串数取决于之前谁用过它，跑两次顺序就不同。轮次在取迭代器时推进，
  `reset()` 归零。
- 顺序切片保持视图，花式索引本来就会产生新数组；只有非连续时才
  `ascontiguousarray`，不白拷数据。

**一条必须说清的限制**

`mse` / `mean` 需要静态元素个数来算平均，所以批大小必须与图的静态批维一致。
DataLoader 不会替你改图——它只负责切，切多大由构图时决定（符号批请自行用
`sum` 或显式 `count`）。

**验证**：新增 13 条用例。"前 3 步存档 + 续训 3 步"与"连续 6 步"的参数**逐位
相同**（参数与 Adam 槽位都恢复）；参数集合 / 形状 / 后缀 / meta 非标量 / 权重
存档误当续训存档，五类错误全部明确失败；DataLoader 的批覆盖、逐轮不同、
reset 重放、drop_last、长度不一致均有用例；一个 40 轮的完整训练循环用
DataLoader 喂数据并把损失降到 5% 以下。本地 474 个测试（350 通过 / 124 跳过）；
真机 474 个测试 skipped=17，`verify_cuda.py` PASS。

**明确没做的**

- **存档不含 BN 的 running stats**：它是数组，既不是可训练参数，也不是优化器
  槽位，`meta` 只收标量。当前只能由调用方另行保存；要进 checkpoint 得先决定
  "统计量算不算模型的一部分"。
- **不做分片 / 并行预取 / 变长序列的 padding 与 collate**：这一层只处理已经对齐
  的定长数组。
- **不做压缩存档**：`np.savez` 而非 `savez_compressed`，用体积换读写速度。
- **不做轮转与自动保留最新 N 个**：交给调用方，这一层不猜策略。

### 5.28　f16 与梯度缩放：把种子当缩放因子

**动机**：f16 的吸引力是显存与带宽减半，但它的最小正规数约 `6.1e-5`、
次正规下限约 `6e-8`。梯度一旦落进这个区间就静默变成 0——损失曲线只是
"学得慢一点"，看不出任何异常。要能训，先得解决下溢；要能复现，缩放不能改动图。

**先确认现状**：IR 早就有 `f16` / `bf16`（`DType` 枚举与 numpy 承载表），
参考后端本来就能跑 f16 的前向与反向，缺的是训练层的处理。所以这一批没有加算子、
没有改 IR、没有碰执行骨架。

**做法：缩放乘在反向种子上**

反向图对种子是线性的：种子从 1 改成 `scale`，所有梯度同比放大，图完全不变。
换 scale 不用重构图——这正是静态图该有的用法。对比之下，"在 loss 上乘 scale"
需要构图期常量，换 scale 就得重新构图；"把 scale 做成图输入"又多一个每步要绑的参数。

`GradScaler` 的三个动作：

- `backward_seed(dtype)`：返回 0 维种子，值等于当前 scale；超出该 dtype 的范围
  直接报错（否则种子变 inf，整张图的梯度全废）。
- `unscale(grads)`：**先升 f32 再除**。在 f16 上直接除以 16384，会把刚被缩放
  救回来的小梯度再推回下溢区，等于白缩放。
- `step(optimizer, grads)`：出现非有限值就减半 scale 并跳过更新
  （`step_count` 不增，Adam 的偏差修正因此不受影响）；连续 `growth_interval`
  步正常则翻倍。

**上界不是可选项**：默认 `init_scale = 2^14`、`max_scale = 2^15`。不封顶的话，
连续若干轮不溢出就会把 scale 涨过 65504，种子变成 inf——这不是理论风险，
是默认参数下的必然结果。

**参数留在 f32**：f16 的更新量会被量化吞掉（`lr=1e-3` 时更新量已接近 f16 的
最小间隔），所以 master 权重保持 f32，只在喂图时 cast 成 f16。这一条目前靠
调用方遵守：`ParameterStore` 仍然只接受 f32/f64。

**验证**：9 条用例。f16 图的前向与反向在量化误差内与 f32 一致；`1e-8` 的梯度
不缩放时为 0、缩放后可表示并在反缩放后还原（而在 f16 上直接相除会二次下溢）；
inf / -inf / nan 三种取值都跳过且不污染参数；scale 按间隔增长且被封顶；
非法超参报错；一个 300 步的 f16 训练循环（f32 master + 缩放）把损失降到初值的
5% 以下；CUDA 侧新增一条 f16 明确拒绝的用例。本地 484 个测试（359 通过 / 125 跳过）；
真机 484 个测试 skipped=17，`verify_cuda.py` PASS。

**明确没做的**

- **CUDA 没有 f16 内核**：逐元素与归约要模板化，matmul 要接 hgemm，卷积的
  im2col 还要决定 f16 存储与 f32 累加。这是独立一批工作，现在明确拒绝。
  **补记：已在 5.33 补齐**（除 attention 与 conv 分块核外全部支持 f16）。
- ~~**没有 autocast 策略**~~：5.90 补上了图重写版的 `Autocast`（`passes/autocast.py`），
  按白名单把 matmul / conv 降到 f16、把归一化与归约拉回 f32；本节记的是 5.33 时的限制。
- **不处理前向激活下溢**：缩放只作用于梯度；激活已经下溢的图，缩放救不回来。
- **不做 bf16 的专项验证**：`bf16` 在参考后端借用 f32 承载（只验数学语义，
  不代表位宽），不用于数值精度测试。

### 5.29　融合注意力：一个节点，反向由已有算子组合

**动机**：Transformer 里的注意力原本是五个节点（`matmul` / `mul` / `add` / `softmax` /
`matmul`），中间那个 `(B, H, S, S)` 分数矩阵因此是一等公民——它必须被分配、被写回、
被下一个节点再读。真实的注意力实现（FlashAttention 一类）从不物化它，而"不物化"
这件事没法由五个节点各自的语义表达出来。要把它变成一个后端可自主决定内存策略的
单元，只能在 IR 里收成一个节点。

**语义（一次性定死）**

```
attention(q, k, v [, mask]) = softmax(q @ k^T * scale [+ mask]) @ v
```

- `q` `(..., S, D)`、`k` `(..., T, D)`、`v` `(..., T, E)` → 输出 `(..., S, E)`；
  `S != T` 即交叉注意力，`D != E` 允许。
- `mask` 是可选的**加性偏置**（禁止位取大负数，见 `causal_mask_bias`），
  形状只需能广播到 `(..., S, T)`。
- `dropout` 作用在 softmax 概率上，`(p, seed)` 写在 attrs 里，与 `dropout_mask`
  共用同一套 SplitMix64。
- `scale` 是 attrs 而不是输入：它是构图期就确定的常数（`1/sqrt(d_head)`），
  做成输入只会多一个每步要绑的参数。

**批维不许广播**。这是刻意的：`(2,1,…)` 对 `(2,3,…)` 的批维广播会让反向必须做
运行期归约，而 `reshape` 现在不接受形状表达式（与 conv 拒绝符号空间维同源的问题；
5.53 的表达式只用在卷积/池化的派生维上）。要广播就先在前端显式 `broadcast_to`。

**反向不新增算子**：`dv = p^T dy`、`dp = dy v^T`、`ds = p * (dp - sum(dp*p))`、
`dq = (ds*scale) k`、`dk = (ds*scale)^T q`——全是 PORTABLE 算子
（`matmul` / `softmax` / `mul` / `sub` / `reduce_sum` / `dropout_mask`）。
于是反向图自动落在任何能跑前向的后端上：CUDA 只要有一个前向内核，整条训练链就通了，
不需要"反向专用内核"这一层。dropout 的掩码按同一 `(p, seed)` 重建；
`dropout_mask` 对输入没有梯度，被乘掉的那一份由乘法那条边负责。

**mask 的梯度**：形状与分数矩阵相同时（可学习的相对位置偏置）返回 `ds`；
广播形态（`(1,1,S,S)` → `(B,H,S,S)`）下返回 `None`。理由不是"懒得写"：
归约回广播形状需要静态形状，而这里恰好是符号的批 / 头维。`mask` 在惯例上也是
结构性常量、不进优化器（`build_transformer` 的文档要求把它放进 `exclude`）。

**构件层开关**：`BlockSpec.fused_attention` / `TransformerSpec.fused_attention`
默认关闭，打开后注意力变成单个 `attention` 节点，dropout 进入算子 attrs。Block 的
`scale` 原来是**参数**（`(1,)`），fused 路径下它变成构图期固定值，因此参数表里
也不再有它——否则会出现一个永远没有梯度的死输入。默认关闭是为了让"融合前后
逐位一致"这条对照随时可跑；等真机内核补齐，默认值再翻过来。

**验证**：28 条用例。前向与独立写的 numpy 实现逐点一致（批维、交叉注意力、
`v` 的通道数与 `q/k` 不同、零 mask、因果 mask、因果性）；反向与中心差分一致
（无 mask / 同形状 mask / 广播 mask / dropout / 交叉注意力）；dropout 同图可复现；
符号批可构图、反向也可构图；构图期拒绝批维不一致、内维不匹配、k/v 长度不等、
dtype 不一致、mask 末维不匹配、dropout 越界、输入个数不对；两层 decoder 在
`fused_attention` 开关前后的输出与全部参数梯度一致；fused 图不含 `scale` 参数
且 `verify_portable_only` 通过；CUDA 侧当时用一条"明确拒绝"的用例钉住边界
（5.32 落地内核后改为正面对拍）。
一致性套件补了 `attention_basic` / `attention_masked` 两个固定用例。
本地 513 个测试（387 通过 / 126 跳过）；真机 513 个测试 skipped=17，`verify_cuda.py` PASS。

**明确没做的**

- **CUDA 没有内核**：这一条已在 5.32 补齐（分块 online softmax，分数矩阵不物化）。
  仍未做的是按 causal 跳过上三角块——IR 没有 causal 标记，要跳过就得证明上三角
  确实被 mask 掉，而那个证明需要把 mask 读回主机，与捕获期冲突。
- **不接受布尔 mask**：只接受加性偏置，布尔 mask 由前端转换。
- **不支持 GQA / MQA**：现在要求 `k`/`v` 的批维（含头维）与 `q` 完全一致。
  这是有意收窄，等有真实需求再加，并同时定义它的反向。
- **dropout 掩码按下标生成**：分块内核按 `(bh*S + s)*T + t` 的扁平下标重建，
  与 `dropout_mask` 的约定一致（5.32）。

### 5.30　插件扩展面：算子怎么进表、多实现怎么选

**问题**：1.2 里两行一直标着 🟡——"自定义算子仍无法进 `OPS`"、"无多实现按代价
选择"。前者意味着第三方想加一个算子，只能改框架源码；后者意味着同一个算子若有两个
实现，"用哪个"由注册顺序决定，而这恰恰是最难解释的一种优先级。

#### 算子进表的护栏

`plugin.py` 提供 `register_operator(op, namespace=...)`。它不是 `ir.ops.register`
的别名，而是在它前面加了三道检查——因为 `OPS` 是**进程级全局状态**，没有进程边界
可以依靠，一次错误的注册会污染此后所有构图：

| 检查 | 失败方式 | 挡的是什么 |
|---|---|---|
| 名字必须以 `<namespace>.` 开头 | `ValueError` | 两个插件撞名，谁先被 import 谁生效，极难复现 |
| 名字不得是内置算子 | `ValueError` | 参考语义被悄悄换掉，conformance 仍然 PASS |
| `Op` 实例、可重复注册检查 | `TypeError` / `ValueError` | 传错对象；静默覆盖 |

`unregister_operator(name)` 只对插件算子生效，内置算子拒绝卸载；`plugin_ops()`
返回"当前表里有哪些是插件加的"。两者是为测试隔离与热插拔准备的——否则后跑的用例
会看见前面遗留的算子，失败位置和原因完全对不上。

**最关键的一点是"注册即生效"**：`ReferenceBackend` 的语义与能力都按 `OPS` 动态
派生，所以插件算子不需要再教参考后端认识它，立刻能执行、能进 conformance 对拍。
这正是"语义放 IR、内核放插件"这条约定第一次被第三方完整走通：

```python
from aetherflow import Op, Portability, register_operator

register_operator(
    Op(
        name="acme.double",
        infer=lambda inputs, attrs: inputs[0],
        reference=lambda args, attrs: args[0] * 2.0,
        portability=Portability.PORTABLE,
        doc="逐元素乘 2",
    ),
    namespace="acme",
)
```

**一个必须说清的限制**：含插件算子的图只能被"同样加载了该插件"的进程读回。
算子表是全局状态，这是它的自然结果；图的可移植性以两边算子表一致为前提，
写进长期存档前要先确认加载路径里有这个插件。

#### 多实现按代价选择

`KernelRegistry.register(op, kernel, cost=..., label=...)` 允许一个算子登记多个
候选，调度时取代价最小者。规则只有一条，但它是整个机制的支点：

- 单实现不写 `cost`，调度零开销，行为与 5.11 完全一致；
- 一旦出现第二个候选，**全部**候选都必须带 `cost`。混着"有代价"与"没代价"的
  候选，排名里就有一个无法比较的项，最后胜出的必然是注册顺序——那是隐式优先级，
  不是按代价选择。所以这种注册直接 `ValueError`，而不是"尽力比较"。

`cost(node, args, attrs)` 与内核同参，因此能按真实形状决定谁更便宜。返回 `None`
或非有限值表示"这个候选在当前形状下不适用"，全部候选都不适用时抛
`BackendCapabilityError`——对调用方而言，"这个形状没有内核"与"没有这个算子"
是同一件事，不该是两种异常。

**这次只做"选择"，不做"自动调优"**：注册表不测量、不缓存、不搜索候选，
代价函数由插件自己写。测量、上报与配置库是 M4 的事；现在把接口立住的理由是，
CUDA 的 attention 内核（5.29 明确没做的那一块）一落地就会出现两个真实候选
（分块内核 / 组合算子），那时需要一个已经定好的落点。

#### 插件开发指南

按顺序做，每一步都能独立验证：

1. **定义语义**。写 `Op(name, infer, reference, attrs_schema, portability)`。
   `infer` 必须是纯符号运算，能在符号维下工作；`reference` 只用 numpy 标量语义，
   不绑定加速库。名字带自己的命名空间，用 `register_operator` 进表。
2. **用参考后端跑通**。`ReferenceBackend().execute(graph, feed)` 即可验证；
   如果 `portability` 是 `LIBRARY` / `VENDOR`，参考后端会拒绝——这不是 bug，
   它提醒你这个算子已经不可能移植到别的硬件上了。
3. **补一条 conformance 用例**。构造 `Case(name, graph, inputs, ops=...)`，
   用 `run_conformance(backend, cases=[case])` 对拍。自定义图集会替换默认图集，
   所以插件应当自己维护一份覆盖清单。
4. **实现后端准入面**。`name` / `capabilities()` / `execute()` 三个成员，缺一个
   就不算后端。`capabilities()` 必须如实列出能承接的算子；算不了的抛
   `BackendCapabilityError`，不要用别的异常，也不要静默降级。
5. **按需实现可选契约**。`device_info()` 与 `memory_stats()` 都是可选的；
   没有统计数据时返回 `None`，不要返回 0——0 是观测值，会让"没有统计"和
   "用了 0 字节"混为一谈。
6. **登记内核**。用 `KernelRegistry.register` 显式登记，不要用
   `getattr(self, f"_op_{name}")` 反射：反射把"声明了哪些"和"实现了哪些"
   隔着一层字符串拼接，拼错要到第一次推理才暴露。有两个以上实现时给每个候选
   写 `cost` 与唯一 `label`。
7. **自证**。`run_conformance(你的后端)` 必须 PASS。报告里的 `refused` 是允许的
   （能力协商允许显式拒绝），`failed` 不允许。把报告贴进接入说明，比任何宣传
   都有说服力。

**验证**：新增 18 条测试（插件扩展面 11 条、多实现 7 条）。插件算子侧覆盖：命名
空间缺失、空命名空间、非 `Op` 对象、重复注册、内置算子不可覆盖 / 不可卸载、
卸载后表恢复、`plugin_ops()` 只列插件；注册后参考后端立刻承接、conformance 用
自定义 `Case` 判定 PASS、`LIBRARY` 插件算子被参考后端明确拒绝。多实现侧覆盖：
取代价最小者、代价随真实形状变化、全部不适用时抛 `BackendCapabilityError` 且
消息含算子名与位置、混合有 / 无代价的候选被拒、`label` 重复被拒、`cost`
不可调用被拒、`variants()` / `get()` 的自省语义。本地 531 个测试
（405 通过 / 126 跳过）。

**明确没做的**：不做候选的测量与缓存（M4 自动调优）；不做算子版本协商
（第二个后端还没有，协议版本从两个实现里提炼才有意义，见 5.10 的同一条理由）；
不让插件改内置算子——不是限制，而是"参考语义是唯一真值"这条约定的前提。

### 5.31　训练辅助：调度、裁剪、累积、EMA

**问题**：5.21 结尾列的"明确没做的"里有学习率调度与梯度裁剪。它们不影响
"这套 IR 能不能训一个模型"这个结论，但决定"能不能正经训一个模型"：
固定学习率在小模型上够用，在需要 warmup 的场景一开场就发散；没有梯度裁剪，
一次异常大的梯度就能把参数打成 nan，而训练日志只会显示"loss 变成 nan"。

四项都放在 `train/`，且都不碰 IR——它们是训练循环的胶水，不是新的执行机制。

#### 学习率调度：无状态的 `lr_at(step)`

`CosineDecay` / `StepDecay` / `ExponentialDecay` 只实现一个方法
`lr_at(step) -> float`；`Warmup` 是组合器，`apply_lr(optimizer, schedule)` 负责
把当前值写进优化器。

为什么不做成 PyTorch 那种自带计数的 `scheduler.step()`：

1. **续训只有一份真相。** 需要存档的步数是 `optimizer.step_count`（5.27 已经存了）。
   调度器再维护一份计数，两者分叉时学习率会悄悄偏移，而损失曲线看起来只是
   "没那么好"。
2. **梯度累积的语义是明确的。** 窗口内走了 N 次反向但只更新一次参数，"第几步"
   必须指真实更新次数；传 `optimizer.step_count` 就没有歧义。
3. **可组合。** warmup 是"前 K 步换一条曲线"，组合器比给每个调度器都加一个
   `warmup_steps` 参数更不容易写错。

`Warmup` 的 `start_lr` **故意没有默认值**：取 0 会让第一步参数完全不动，
取多少是训练策略，不该由框架猜。warmup 结束后内层按**绝对步数**继续，
所以 `Warmup(CosineDecay(lr, 1000), 100)` 的余弦从第 100 步接着走，
"总步数"只需在一处定义。

#### 梯度裁剪：全局范数，不是逐张量

`clip_global_norm(grads, max_norm)` 返回 `(裁剪后的副本, 裁剪前的范数)`。
逐张量裁剪会改变梯度**方向**（各分量被不同程度的缩放），而全局范数只缩整体
尺度——裁剪的目的是防止单步走太远，不是改变下降方向。

返回裁剪前的范数是故意的：范数突然飙升是"即将发散"最早的可见信号，比等
损失变 nan 早得多。范数含 inf/nan 时直接 `ValueError`，不替换成 `max_norm`——
把坏梯度裁成"看起来正常"的值只会推迟故障暴露。f16 训练的溢出由 `GradScaler`
跳过，与这里是两条互不替代的路径。

#### 梯度累积：除以实际累积数

`GradientAccumulator(steps)` 内部用 float32 缓冲（f16 累加几次就丢精度，
而累积的目的正是等效更大的 batch）。三个细节都写进了测试：

- `flush()` 除以**实际累积数**，不是窗口大小。最后一个窗口不足 `steps` 时
  除以 `steps` 会把梯度整体缩小，参数少走一步，损失曲线上完全看不出来。
- 窗口满后再 `add` 直接报错，不自动清空——自动清空会悄悄丢掉已累积的梯度。
- 顺序是"先累积、后裁剪"。反过来每个微批各自被裁到 `max_norm`，窗口总范数
  最多是 `N * max_norm`，等于放大 batch 时悄悄放松了正则。

#### EMA：初始化即复制

`EMA(params, decay)` 在构造时直接复制一份影子，而不是从零开始平均。零初始化
在前 `1/(1-decay)` 步里严重偏向 0（decay=0.999 时是一千步），必须配偏差修正；
复制则从第一刻起无偏，也不需要额外状态。影子用 float32 存：f16 上
`(1-decay)*param` 会被舍入归零，平均结果退化成几乎不动。

`copy_to` 是原地写（`arr[...] = shadow`），不替换数组：优化器持有的是数组引用，
替换成新数组会让它继续更新一批没人再读的旧内存。

**验证**：新增 26 条测试。调度侧覆盖端点与单调性、超界不报错、warmup 的连续性与
交接点、下降型 warmup 被拒、`apply_lr` 取 `step_count`；裁剪侧覆盖范数手算、
未超阈值不扰动、裁剪后方向不变且范数等于阈值、返回副本、inf 被拒；累积侧覆盖
微批平均、不足窗口除实际数、满了再 add 报错、名字 / 形状不一致报错、f16 输入
在 f32 缓冲里累加；EMA 侧覆盖初始化即复制、更新公式、原地写回、存档往返、
形状不匹配被拒。本地 557 个测试（431 通过 / 126 跳过）。

**明确没做的**：不做 `ReduceLROnPlateau` 这类按指标调度的策略（它需要把评估
结果喂回调度器，等于引入第二份训练状态）；不做 AdamW 式的解耦权重衰减
（那是优化器的语义变更，与本批无关）；不做梯度累积与 `GradScaler` 的自动串联
（顺序由调用方写，只有两行，隐式串联反而更难排查）。

### 5.32　CUDA 融合注意力：分块 online softmax

**问题**：5.29 把注意力收成一个 IR 节点，但 CUDA 侧一直明确拒绝。如果继续让它
退回"matmul + softmax + matmul"，同一张 IR 在两个后端上的内存特征就完全不同：
`(B, H, S, S)` 分数矩阵照样被分配、写回、再读，收成一个节点也就失去了意义。

**布局与分工**：一个 block 处理一个 `(batch*head, query 行)`，grid = `(S, BH)`，
block 是 `min(256, 2^k)`。key 方向流式分块，每块只保留三样东西跨块存活：
running max `m`、running normalizer `l`、输出累加器 `s_acc[E]`。核心循环：

1. 线程映射到 `t`，算 `dot(q[s], k[t]) * scale (+ mask)`；
2. `block_reduce_max` 得到块内最大，`m_new = max(m, blk_max)`；
3. `corr = exp(m - m_new)`，把 `l` 与 `s_acc` 一起拉回新基准；
4. `p = exp(sc - m_new)`，写进 shared 的 `s_w[]`，`block_reduce_sum(p)` 累加 `l`；
5. 线程切换角色、映射到 `e`，走完本块的 `s_w[j] * v[t0+j][e]`；
6. 块尾 `__syncthreads()`——`s_w` 下一块要被覆盖，所有人必须先读完。

第 5 步的访存是连续的：相邻线程读同一个 `t` 上相邻的 `e`。shared 三段分别是
`s_w[blockDim] + s_red[blockDim] + s_acc[E]`，超过 48KB 明确拒绝。

**dropout**：按 `(bh*S + s)*T + t` 的扁平下标重建 SplitMix64（取高 24 位映射到
`[0,1)`），缩放因子在主机侧算好。分母 `l` **只累加未缩放的 `exp`**，掩码只乘在
输出权重上——这与参考实现"dropout 乘在已归一化的概率上"严格同式；若把掩码也
乘进分母，期望值会偏 `1/(1-p)` 倍。

**mask**：Python 侧用 `cp.broadcast_to` 广播到 `(BH, S, T)` 再 `ascontiguousarray`，
内核只认这一种布局。`(T,)`、`(S, T)`、`(1, 1, S, T)` 的差异全部交给 numpy 的
广播规则，不在内核里复刻一份（内核里复刻的广播是上一版最容易出错的地方）。

**明确没做的**

- **不跳过上三角**：IR 没有 causal 标记。要跳过就得先证明上三角确实被 mask 掉，
  而那个证明需要把 mask 读回主机（D2H 同步），与捕获期"不许同步"直接冲突。
  等 causal 变成算子属性时再谈。
- **不做反向内核**：5.29 的反向规则全是 PORTABLE 算子，前向内核一落地，
  整条训练链就通了；单独写反向只会多一份需要独立对拍的实现。
- **不做 GQA / MQA**：与 5.29 同一条收窄。
- **不做 T 方向的二次分块**：分数矩阵本来就不在 shared 里，`T` 只受 launch 规模
  限制，没有分块的必要。

**验证**：新增 8 条 CUDA 用例（本地无 GPU 时自动跳过）：基本 4D、交叉注意力
（`S != T` 且 `D != E`）、`T=300` 跨过 256 的块边界、二维无批维输入、两种 mask
广播形态互相一致、因果 mask、dropout 与参考对拍、`fused_attention=True` 的
Block 端到端（含残差 / LayerNorm / FFN）。本地 565 个测试（431 通过 / 134 跳过）。
批次 5 的真机复跑顺带修好了两条此前从未真正执行过的 attention 用例
（mask 形状与随机数未重置），4 维 mask 的广播方向也是在那时定下来的，见 5.34。

### 5.33　CUDA f16：一套模板 + 四个占位符

**动机**：5.28 让参考后端能跑 f16 训练，但 CUDA 侧一直整体拒绝。这不是"再写一遍
kernel"的问题：如果 f32 / f16 各维护一套源码，任何语义修正都要改两处，两边迟早
漂移，而漂移只会在混合精度训练里以数值误差的形式暴露。

**做法**：每个 kernel 只写一份，用四个占位符描述"存储与算术的分界"：

| 占位符 | 含义 | f32 | f16 |
|---|---|---|---|
| `%T%` | 存储类型 | `float` | `unsigned short` |
| `%LD%` | 按存储类型读成 float | 恒等 | `cvt.f32.f16` |
| `%ST%` | 把 float 写回存储类型 | 恒等 | `cvt.rn.f16.f32` |
| `%ATOMIC%` | 把 float 累加进存储类型 | `atomicAdd` | `atomicCAS` 循环 |

`%PRELUDE%` 统一展开为"归约辅助 + 该 dtype 的四个辅助函数"，这样模板主体里
不会出现任何 dtype 分支——算术一律在 `float` 中完成（累加、归约、指数都不降精度），
只有存取时降回 16 位。这与 5.28 的"种子当缩放因子"是同一原则：精度损失必须是
**显式的实现边界**，不能是悄悄发生的量化。

**不 include `cuda_fp16.h`**：目标机可能只装了驱动而没有 toolkit，NVRTC 的 include
路径不可靠。`cvt.f32.f16` / `cvt.rn.f16.f32` 是 PTX 指令，所有 NVRTC 版本都认识，
`unsigned short` 的位布局与 `__half` 相同，主机侧用 numpy 的 `float16` 直接对拍。
16 位没有原生 `atomicAdd`，用 `atomicCAS` 循环实现；竞争只发生在重叠窗口与重复
下标上，代价可接受。

**GEMM**：新增 `hgemm` / `hgemm_batched`，走 `cublasGemmEx` 与
`cublasGemmExStridedBatched`（`CUDA_R_16F` 输入输出、`CUBLAS_COMPUTE_32F` 累加、
alpha/beta 仍是 `c_float`）。行主序转置技巧与 `sgemm` 完全一致（`n, m, k` 参数、
`lda=n` / `ldb=k` / `ldc=n`），这样两条路径共享同一份布局推理。

**卷积**：f16 一律走 im2col + `hgemm_batched`。分块直算核把权重与累加器写死在
`float` 上，改造成本高而收益不确定（它本来只在 `K <= 32` 时占优），因此强制
`conv_impl == "tiled"` 时直接抛 `BackendCapabilityError`，而不是悄悄换实现。

**attention 仍只有 f32**：5.32 的内核把 shared 布局写死为 `float`（`s_w` / `s_red` /
`s_acc` 三段），且 online softmax 的数值稳定性依赖 f32 的指数范围。f16 在这里
报错，而不是降级。

**dtype 纪律**：输出 dtype 一律跟随输入（`empty_like` 或显式 `dtype=x.dtype`），
kernel 缓存 key 里加上 dtype；`matmul` 的两个输入 dtype 必须一致。不做隐式升/降级，
错误信息直接提示"请在前端显式插入 cast"。`_F16_OPS = CUDA_OPS - {attention}`，
其余算子在 `invoke` 入口就拒绝 f16。

**验证**：新增 21 条 f16 CUDA 用例（无 GPU 时自动跳过）覆盖 binary / 广播 / 标量、
unary + step、constant、cast f16→f32、dropout_mask、matmul（2D 与 batched）、conv2d、
分块卷积对 f16 拒绝、max/avgpool 及其反向、softmax、layernorm、rmsnorm、batch_norm
与 stats、reduce_sum、transpose / reshape / slice、concat / pad、broadcast_to、
embedding 及其反向；容差 `2e-2`，并校验输出 dtype 确实是 `float16`。另有 2 条
**不需要 GPU** 的 codegen 用例：所有模板在 f32 / f16 下渲染都必须 ASCII（NVRTC
只接受 ASCII），且四个占位符全部展开。本地 587 个测试（433 通过 / 154 跳过）。真机复跑时发现批量 f16 GEMM 的符号名
写错（`cublasGemmExStridedBatched` 并不存在，正确的是 `cublasGemmStridedBatchedEx`）
——本地无 GPU 时这条路径整个跳过，只有真机能暴露，见 5.34。

**明确没做的**

- **不做 bf16**：PTX 的 bf16 转换指令需要 sm_80+，而验证机是 sm_75。硬做只能
  走"移位截断"，那不是同一个语义。
- **不做 f16 累加**：`hgemm` 的累加固定 `CUBLAS_COMPUTE_32F`。用 16 位累加会让
  长 K 的误差不可控，也不符合"参考语义是唯一真值"。
- ~~**不做 autocast**~~：5.90 之后由 `Autocast` pass 按表决定（matmul / conv 降 f16，
  softmax / 归一化 / 归约留 f32，其余跟随最宽输入）；这里记的是 5.33 当时的状态。

### 5.34　分组卷积与自适应池化：不新增内核，只改切片方式

**动机**：5.12 与 5.13 之后 CNN 还缺两块——分组卷积（含 depthwise）与自适应池化。
移动端与轻量网络大量使用 depthwise 可分离卷积，而 `AdaptiveAvgPool2d((1, 1))` 几乎是
每个分类网络的收尾层。前者被 5.14 的前端拒绝挡在门外，后者连 IR 里都没有名字。

**分组卷积的语义**：权重形状从 `(O, C, kh, kw)` 变成 `(O, C/groups, kh, kw)`。
第 `gi` 组只消费输入通道 `[gi*C/groups, (gi+1)*C/groups)`、只生产输出通道
`[gi*O/groups, (gi+1)*O/groups)`，组间没有任何数值交流。`groups == C == O` 时每组
只有一个通道，就是 depthwise。所以分组不是新算法，只是**切片规则**——这决定了实现
方式：不写新内核，而是把"切输入 / 切权重 / 写回输出"的循环放到调用方。

**IR 层**：`_infer_conv2d` 删除 `groups != 1` 的拒绝，改为三条静态校验——`groups >= 1`、
输入通道能被 `groups` 整除、权重通道等于 `C/groups`、输出通道能被 `groups` 整除。
参考实现按组切片，各组独立做交叉相关后沿通道轴 `concatenate`，最后统一加 bias；
反向的 `conv2d_input_grad` / `conv2d_weight_grad` 同样逐组计算再拼回。参考实现依旧
不用 im2col——语义由它定义，性能不是它的职责。

**CUDA 实现**：`_op_conv2d` 在 `groups == 1` 时保持原路径（K 小走分块直算，否则
im2col）；`groups > 1` 时逐组把 `x[:, gi*cg:(gi+1)*cg]` 的连续副本交给现有的
`_conv2d_im2col`，结果写进 `out[:, gi*og:(gi+1)*og]`。**不新增任何内核**，因此
f16 分组卷积自动可用（f16 本来就强制 im2col）。分块直算核把通道数写死在
`K = C*kh*kw` 上，改造成本高而收益不确定，所以 `groups > 1` 且强制
`conv_impl="tiled"` 时直接抛 `BackendCapabilityError`——与 5.33 对 f16 分块核的处理
是同一原则：**换实现必须显式，不能静默**。反向同理，逐组调用同一 kernel 后把
各组 `dx` 写回对应通道区间；权重梯度按输出通道切片拼回，切片是连续的。

**自适应池化**：窗口端点用 PyTorch 的整数规则
`start = floor(i*H/OH)`、`end = ceil((i+1)*H/OH)`，窗口可重叠、可大于步长，
`H` 不被 `OH` 整除时也不会漏格。这里最容易出的错是实现"看起来对"但端点差一，
所以 IR 参考实现与 CUDA kernel 用的是**逐字相同**的整数表达式，而不是各自推导。
全局池化不设新算子：`output_size=(1, 1)` 就是整图均值。CUDA 侧是一个输出元素
一个线程的自研 kernel；反向把 `dy[i,j] / 窗口面积` 用 `atomicAdd` 散射回输入，
重叠窗口自然累加（与 `avgpool2d_backward` 的做法一致，f16 走 `%ATOMIC%` 的
`atomicCAS` 循环）。

**前端**：`_conv2d` 不再拒绝 `groups != 1`，直接写进 attrs；
`aten.adaptive_avg_pool2d.default` / `aten._adaptive_avg_pool2d.default` 降级为
`adaptive_avg_pool2d`，`output_size` 统一成 `[OH, OW]`（int 或长度 1/2 的序列都接受）。

**验证**：本地新增 28 条用例——IR 侧分组卷积与朴素六层循环逐点对拍、depthwise
的通道独立性、分组前向 / 两个反向的中心差分、非法 `groups` 的构图期报错；自适应
池化的 floor/ceil 窗口、全局池化、方形 `output_size`、重叠窗口梯度、非法
`output_size` 拒绝；前端侧分组 / depthwise Conv2d 与 `AdaptiveAvgPool2d` 的
torch.export 往返对拍；CUDA 侧分组、depthwise、stride2、强制 tiled 拒绝、两个反向，
自适应池化的不整除 / 全局 / 下采样 / 上采样 / int 参数 / 重叠窗口累加 / 形状不符拒绝，
以及 f16 的分组卷积与自适应池化前后向。conformance 新增 `conv2d_grouped`、
`conv2d_depthwise`、`adaptive_avg_pool2d_2x3`、`adaptive_avg_pool2d_global` 与
`adaptive_avg_pool2d_backward`，用例数 61 → 66，覆盖全部 45 个算子。
本地 615 个测试（444 通过 / 171 跳过）；真机（RTX 2080 Ti）**615 全绿**（19 跳过），
`verify_cuda.py` 的 conformance 为 59 executed / 7 refused / 0 failed，18 条边界用例
（矩形核 + 非对称参数的分组卷积、depthwise stride2、两个分组反向、7x5 → 3x2 的
自适应池化、重叠窗口反向等）全部通过。

**真机复跑暴露的两个本地盲区**

1. **cuBLAS 批量 f16 GEMM 的符号名写错了。** 代码里写的是
   `cublasGemmExStridedBatched`，cuBLAS 实际导出的是 `cublasGemmStridedBatchedEx`。
   本地无 GPU 时所有 matmul / conv 用例都跳过，`ctypes` 只在真机上才会去取这个符号，
   于是整条路径以 `AttributeError` 失败。这类错误的特征是"本地全绿、真机全红"，
   与 5.12 的 NVRTC ASCII 限制同类——**没有真机就不能宣称内核可用**。
2. **4 维 mask 的广播方向。** 参考实现直接 `scores + mask`，`(B,H,S,T)` 与
   `(1,1,S,T)` 相加会得到 `(1,B,H,S,T)`，输出凭空多一维；CUDA 侧的
   `cp.broadcast_to` 又因为目标秩更小而直接报错。两侧现在都先去掉长度 1 的
   前导维再广播，语义与 PyTorch 的 mask 约定一致。

**明确没做的**

- **depthwise 专用内核**：现在每组单独一次 GEMM，`groups == C` 时是一次
  `kh*kw` 宽的极窄矩阵乘，访存远未打满。真正的 depthwise 核应让一个线程沿
  空间滑窗累加，不需要列矩阵。这是性能项，不是语义项。
- **分块直算核支持分组**：`K` 的分块参数与通道切片耦合，收益又不确定，保持拒绝。
- **`ceil_mode` / `count_include_pad` / `return_indices`**：池化的参数开关仍未做。
- **转置卷积**：前端仍明确拒绝，它与分组卷积是两件事。

---

### 5.35　CUDA 扫描与 SNN 稀疏事件：一个内核 + 一个表示

**动机**：1.2 里剩下的两个 🟡 是同一批缺口的两半。SSM 的 `scan_linear` 在 CUDA 上
一直被明确拒绝——不是"还没写"，而是没有机器验证；SNN 的脉冲只能按稠密张量处理，
稀疏表示与事件内存池都还只是缺口表上的一行字。两者都不需要新 IR：递归已经是
语义算子（5.19），稀疏化是无损的布局变换。

**一、CUDA 扫描：元素并行、时间串行**

前向内核的全部算术就是下面这段（`elems` 是尾维压平后的元素数）：

```
for each i (grid-stride):
    h = h0[i]
    for t in 0..T-1:
        h = a[t*elems + i] * h + b[t*elems + i]
        out[t*elems + i] = h
```

四个要点：

1. **尾维可以压平。** `a` / `b` 的布局是 `(T, 尾...)`，第 t 步的偏移恒为
   `t*elems + i`，与尾维的阶数无关。所以 `(T, D)`、`(T, B, D)`、四维尾共用
   同一段源码，SSM 的真实形态不需要特判。
2. **并行度来自尾维，不是时间。** 递推没有结合律之外的并行空间；尾维
   （batch × 通道 × state）通常远大于 T。chunked / Blelloch 扫描需要两遍通信，
   属于性能工作，不是语义。
3. **伴随的 carry 是 `a[t+1] * dh`。** 从后往前，最后一步没有后继，carry 取 0。
   少乘或多乘一步 `a` 在 T=1 上看不出来，所以真机边界用例刻意取 `|a| ~ 0.6`、
   T=9，让错误按指数放大。
4. **f16 走同一模板。** 读成 float、在 float 里累加、只在存取时降回 f16——与
   5.33 的占位符约定完全一致，因此扫描不需要单独的 dtype 分支。

**二、SNN 稀疏事件：数据侧，不动 IR**

`runtime/events.py` 四个构件：

| 构件 | 内容 |
|---|---|
| `SpikeEvents` | CSR 变体：`counts[t]`（该步事件数）+ `offsets[t]`（该步第一个事件在 `indices` 里的位置）+ `indices`（尾维下标，升序） |
| `dense_to_events` / `events_to_dense` | 压缩与还原 |
| `EventPool` | 复用 `indices` 缓冲：容量不足按 2 倍扩容，给出分配 / 扩容 / 峰值统计 |
| `event_mean_rate` | 只读事件算平均发放率，一次都不物化 (T, N) |

三个设计选择：

1. **`offsets` 长度是 T，不是 T+1。** 末尾总量由 `nnz` 表达，不存 `offsets[T]`
   哨兵：少一次 `concatenate`，也就少一份对库函数的依赖（numpy 与 cupy 的
   拼接 API 不是同一个名字）。
2. **只接受 0/1。** 事件表示丢弃的是零，不是"取整"；把 0.5 当事件是静默改变
   语义。这是系统边界，宁可多一次 pass 也要拒绝。
3. **池返回的是视图，单次 pack 仍有一次临时数组。** 库不提供"把全量非零写进
   现有缓冲"的接口，所以 `nonzero` 的结果必然是新分配的；池消除的是跨时间步、
   跨前向的反复分配。这两条语义都写进了 docstring，并各有一条测试钉住——
   把"零分配"写进注释很容易，做到却需要自己写 scatter kernel。

**为什么不进 IR。** 压缩与还原对值是**无损的布局变换**，在自动微分里对应恒等
映射；做成算子只会让反向图多一条恒等边，不会让任何后端变快。真正的收益是
"用稀疏表示做计算"（稀疏 matmul / 事件驱动卷积），那是后端 kernel 的事。

**验证**

| 判据 | 覆盖 |
|---|---|
| 前向 / 伴随 | conformance 的两个 scan 用例从 refused 变 executed；与参考实现逐位对齐（两边是同一个顺序循环） |
| T=1 | 前向退化成 `a0*h0+b0`；伴随 `dh0 = dy0` |
| 三维尾 | `(T,B,D)` 与 `(T,B*D)` 等价 |
| 长序列 / 宽尾 | T=64 × 尾 2048（跨多个 block，走 grid-stride）；T=128 |
| 非连续输入 | Fortran 序先转连续，不能按线性下标直接读 |
| f16 | 前向与伴随均在 2e-2 内，输出保持 f16 |
| 稀疏往返 | 随机 0/1、全零、全一、空步、零宽尾，逐位相同 |
| 索引正确 | `counts` / `offsets` 与手算一致；每步下标与该步 `flatnonzero` 一致 |
| 边界拒绝 | 非 0/1 取值、秩 < 2 |
| 发放率 | 只读事件的结果与稠密 `mean(axis=0)` 一致 |
| 池 | 首包 1 次分配；同规模不扩容；超容量 2 倍扩容；视图被下一次 pack 覆盖 |
| 设备侧 | numpy 与 cupy 共用同一份实现（device 上 pack / unpack / rate） |

新增 31 条本地用例（scan 10 条 + 事件 21 条，其中设备侧 3 条在真机上执行）。
全量 646 个测试：本地 462 通过 / 184 跳过；真机（RTX 2080 Ti）**646 全绿**
（19 跳过）。conformance 66 用例 **61 executed / 5 refused / 0 failed**、覆盖全部
45 个算子（剩余 5 条拒绝都是"能力声明之外"的既定边界：partial-batch matmul、
非最后维 softmax / layernorm / reduce_sum、f64 输入）。`verify_cuda.py` 的边界用例
从 18 条扩到 22 条（三维尾、T=1、T=64×尾 2048、T=9 的伴随），全部通过。

**真机暴露的本地盲区**：cupy 14 起禁止 `np.asarray(cupy_array)` 的隐式拷贝，
`SpikeEvents.nnz` 里那一行在本地只走 numpy 分支，永远碰不到这个限制，真机上
第一条事件用例就抛 `TypeError`。改成显式 `.get()`。与 5.34 的两条同类——
**本地跳过的路径不能算已验证**。

**明确没做的**

- **并行扫描**：见上。T 很长时延迟随 T 线性增长。
- **事件驱动的稀疏 kernel**：稀疏 matmul / 卷积没有实现，事件表示目前的价值是
  "少存一份稠密张量 + 少一次归属统计"，不是"少算一次乘加"。
- **池的多桶与分配器打通**：只保留一块缓冲，形状或 dtype 来回切换会反复重建；
  与 CUDA 的 `Arena`（捕获期复用）也还没有共用分配路径。
- **符号时间维 / 变长序列 / 编码器**：与 5.18 的边界相同，不是这一批的目标。

### 5.36　编译层：常量折叠、深层融合与静态显存预算

**动机**：1.2 里编译层剩下的三项互相咬合。常量折叠的前提是 `constant` 能承载
"任意张量"而不只是同值填充；深层融合要求"融合"不新增算子（ADR-010）；静态预算
则是把 `Arena` 在捕获期才敢说的话，提前到看图的时候说。三项都只在 IR 层，不依赖
任何后端。

**一、常量折叠：先让 `constant` 装得下张量**

`constant` 原来只有 `value`（标量填充）。若只折叠同值填充，真实图里几乎没有收益——
两个不同常量相加的结果本来就是任意张量。所以先加 `data`（JSON 嵌套 list），与
`value` 互斥：`value` 是填充，`data` 是任意常量。参考语义按 `np.asarray(data)` 还原，
CUDA 后端在主机侧构好数组再整块上传（不在 GPU 上逐元素赋值）。

`passes/fold.py` 的规则只有三条，都是"折了会改变语义或得不偿失"的边界：

| 不折的情形 | 理由 |
|---|---|
| 输出形状含符号维 | 没有运行期绑定就算不出元素数，硬折会把符号图退化成只能跑一种形状 |
| 带种子的随机算子（`dropout_mask`） | 结果确定，但语义是"每次重新采样"；固化掩码会让训练图永远复现同一份噪声 |
| 元素数超过 `max_elements`（默认 65536） | 折叠结果会写进节点属性并随图序列化，太大就不再"可读可传" |

求值统一走 `OPS[name].reference`，因此折叠前后的图在参考后端上逐位一致。折叠后
原常量节点仍会留在图里，由同一条管线里的 DCE 清掉——`fused_pipeline` 的固定顺序
是 `ConstantFolding → SimplifyShapeOps → CSE → FuseSoftmaxScale →
FuseLayerNormResidual → FuseSiLU → DCE`。

**二进制陷阱（顺手修的既有 bug）**：`SimplifyShapeOps._is_noop` 对零输入节点取
`node.inputs[0]`，纯常量图（折叠后常见的形态）会 `IndexError`。之前模型图恰好没有
零输入节点走到那里，所以一直没暴露。

**二、深层融合：两条真实会命中的规则**

两条都按 ADR-010 表达成"已有算子的可选输入扩展"，参考后端与 CUDA 后端都承接：

| 规则 | 形态 | 判据 |
|---|---|---|
| `FuseLayerNormResidual` | `layernorm(add(a, b), g, β)` → `layernorm(a, g, β, b)` | add 的 fanout == 1、不是图输出、a/b 与 layernorm 第 0 输入同类型（不吸收广播） |
| `FuseSiLU` | `mul(a, silu(b))` → `silu(b, a)` | silu 只有 1 个输入、fanout == 1、不是图输出、gate 与 silu 输出同类型 |

`silu(x, gate)` 的语义是 `silu(x) * gate`，正是 SwiGLU 的门控乘；在
`models/mamba.py` 的 `mul(y, silu(z))` 上真实命中。`layernorm` 的第 4 个输入在
归一化之前相加，于是 add 的输出从不物化——这是"整段融合"在语义层的准确表达。

**诚实的否定结论**：pre-norm Transformer Block 里 `x1 = add(x, ...)` 同时被
`layernorm(x1)` 和 `out = add(x1, ff)` 使用，fanout == 2，因此
`FuseLayerNormResidual` **不命中**。这不是实现缺陷，而是残差在 pre-norm 结构里
天然被共享；强行融合需要跨节点维护"残差流"，那已经是另一个 IR。测试
`TestFusedPipelineOnBlockExtended` 把这条否定结论钉住，避免后人误以为 Block
被融合了。

CUDA 侧新增两个内核：`LAYERNORM_RESIDUAL_SRC`（先按存储类型舍入做加法、再读成
float 归约）与 `SILU_GATE_SRC`（跨步循环），f16 走同一模板。反向规则同步扩展：
`layernorm` 的 vjp 在 4 输入时返回 `(dx, dgamma, dbeta, dx)`，`silu` 的 vjp 在
2 输入时返回 `(mul(dx, gate), silu(x))`。

**三、静态显存预算：不执行也能回答"至少要多少显存"**

`ir/liveness.py` 的核心只有一条：**值的生命周期就是"被定义"到"最后一次被使用"**。

- 节点输出的定义时刻 = 它在拓扑序里的下标；
- 图输入在时刻 -1 就已存在（由调用方 / 参数驻留持有）；
- 最后一次使用 = 把它当输入的最大节点下标；图输出则一直活到最后。

峰值取所有时刻活跃字节数之和的最大值。`MemoryBudget` 同时给出：

| 字段 | 含义 |
|---|---|
| `peak_live_bytes` | 含图输入的峰值；默认 `release=True` 口径，见 5.105 |
| `intermediate_peak_bytes` | 只算中间缓冲，与 `Arena.peak_live_bytes` 同口径（同为 `release=True`） |
| `total_bytes` / `intermediate_total_bytes` | 不复用时的累计需求 |
| `aligned_peak_bytes` / `aligned_intermediate_peak_bytes` | 每个值按 512B 对齐后的峰值 |
| `release` | 预算对应的执行策略：`True` = 用完即摘，`False` = 留到图结束（`run_device` 的默认） |
| `reuse_ratio` | `total_bytes / peak_live_bytes`；`release=False` 时恒为 1 |
| `symbols` | 符号维绑定；未绑定时明确抛 `ValueError`，不猜 |

**与 `Arena` 的对照**：RTX 2080 Ti 上的 Transformer Block（B=2, S=5, d_model=32，
d_ff=64）静态预算 6656 B，捕获期实测 8192 B。两者不要求相等——arena 对每个请求
512B 对齐、后端的 softmax scale 等小临时不体现在 IR 里，这些让实测偏大；反过来
连续张量上的 reshape 走视图时实测会偏小。相差 1.23 倍，远小于一个数量级，所以
静态预算可以直接用于容量评估，而不必先跑一遍。这个对照同时写进了
`verify_cuda.py`（超过 2 倍即失败）与 `TestCudaGraph`。

**验证**

| 判据 | 覆盖 |
|---|---|
| 折叠正确性 | 标量填充 / 任意 `data` / 链式折叠 / 与 DCE 协同后节点数 / 序列化往返 |
| 折叠边界 | 符号维不折、`dropout_mask` 不折、超大常量不折（`max_elements` 小上限） |
| 融合语义 | 融合前后参考后端逐位一致；`layernorm` 4 输入与 `silu` 2 输入的梯度对照 |
| 融合判据 | fanout == 2 不命中（pre-norm Block）、add 生产者不是 add 不命中、类型不同不命中 |
| 融合后端承接 | f32 / f16 的 residual 形态、SwiGLU 门控；长度不整除与列数非 2 的幂的边界 |
| 折叠后端承接 | 折叠出的 `data` 常量在 CUDA 上直接物化并参与计算 |
| 静态预算 | 线性链 / 分支 / 复用比 / 生命周期 / 512B 对齐 / 符号维绑定 / 零元素 / `as_dict` / 两种 `release` 口径 / 事件扫描对暴力扫描（5.105） |
| 预算对照 | 与捕获期峰值同量级（真机 8192 / 6656 B） |

新增 43 条用例：IR 层 33 条（常量折叠 11、`FuseLayerNormResidual` 7、`FuseSiLU` 5、
融合管线 2、liveness 8），CUDA 层 10 条（两个融合内核的 f32 / f16 形态、融合管线
端到端、折叠常量物化、静态预算对照等，在真机执行）。全量 689 个测试：
本地 495 通过 / 194 跳过；真机（RTX 2080 Ti）**689 全绿**（19 跳过）。
`verify_cuda.py` 的额外检查从 26 条扩到 31 条（22 条边界 + 4 条 dropout 之外，
新增 SwiGLU 不整除长度、layernorm 残差非 2 的幂列数、两个 f16 形态与静态预算
对照），全部通过。

**明确没做的**

- **跨节点的残差流融合**：pre-norm Block 的 fanout == 2 是结构性的，不做。
- **GEMM epilogue 融合**：cuBLAS 必须物化输出，已被实验否定，不再列为待办。
- **折叠结果的内存驻留**：折叠出的常量仍走普通 `constant` 路径，没有和
  `resident_parameters` 合并；小常量重复物化的成本没有测。
- **预算驱动的分块决策**：`plan_memory` 现在只是"能报出来"，还没有接进
  任何"选哪个分块 / 要不要换图"的决策。

### 5.37　自动调优、运行时上报与只读智能体接口

1.2 表里"跨后端统一自动调优 / 智能体接口 / 运行时数据上报 + RL 更新配置库"
三行从前都是 ⬜。这一刀把三件事各自做成**有明确失败、有诚实边界**的最小闭环：

- `aetherflow/tuning/`：参数空间 + 网格搜索 + 配置库（与后端无关）；
- `aetherflow/runtime/telemetry.py`：一次执行留一条结构化记录；
- `aetherflow/agent_api.py`：只读查询面，不执行、不改图。

**自动调优：插件声明旋钮，tuning 只负责展开与比较**

后端要做的只有三件事：`tuning_space(op)` 给出候选、`set_tuning(op, config)`
应用、`clear_tuning()` 回默认。CUDA 侧第一批只开放**逐元素类**算子的 block
（add/sub/mul/div/relu/silu/exp/log/sqrt/neg/tanh/sigmoid/gelu/step/spike/
dropout_mask），空间是 `{128, 256, 512, 1024}`。5.56 又开放了 conv2d 分块核的
`tile_w`（旋钮不同、空间形状也不同，见那一节）。

不开放归约类（softmax / layernorm / rmsnorm / reduce_sum / batch_norm）是刻意的：
它们的 shared memory 大小直接由 `blockDim` 决定，改 block 会连带改变 shared
预算与归约树深度。把两类混在一起搜索，失败会表现为"某个候选非法访存"，
而不是"这个候选更慢"——那是另一个问题，要单独验证。

> **5.160 更正**：上面这条前提不成立。归约族内核的暂存区是 `REDUCE_SCRATCH`
> 常量（1024 个 float = 4 KB，正好等于 block 上限），不随 `blockDim` 变化；
> 内核沿 `blockDim.x` 跨步读输入，块内归约树对任意 2 的幂都成立，合法的
> 32～1024 因此都不会越界。于是这批算子连同 `argmax` / `batch_norm_stats`
> 一起进入 `_REDUCE_TUNABLE_OPS`，launch 侧收口到 `_reduce_block(op, cols)`
> （覆盖 → 构造参数 → `plan_reduce`）。它们与逐元素类唯一的差别是"换 block
> 会改变归约顺序"，数值只能近似一致，测试因此用浮点容差而不是逐位相等。

**空间的首候选必须是当前生效值。** 搜索器报 `speedup` 的基准、配置库未命中时
智能体给出的建议，都取首候选。如果固定把 128 放第一位，那么一个默认
`block=256` 的后端会被推荐 128，凭空造出一个"默认值"；实际没调优，报告里
却像调过。CUDA 的 `tuning_space` 因此是"当前值 + 其余候选"，未调优时首候选
就是 `self.block`。

**搜索是最笨的网格，失败必须显式。** 候选空间只有四个点，贝叶斯搜索省不下
时间，只会引入"这次为什么选它"的不可解释性。`tune()` 对每个候选测
`rounds` 次取最小；候选抛异常、返回 `None` / NaN / 负数都记为**不可行**并继续。
只有全部候选都不可行时抛 `AllRejectedError`——"没有一个能跑"与"没有更快的"
必须分开：前者回落到默认配置会在更远的地方炸。

**配置库按 `backend + device + op + shape/dtype` 命中。** 配置的答案依赖机器
与形状：同一份 block 在 2080 Ti 上最优，换一张卡可能更慢。跨设备命中不是
"略差"，而是无法解释。device 取不到时写 `"unknown"`——那是一个具体的"不知道"。
坏 JSON、版本不认识、条目缺字段全部抛 `ValueError`，不跳过坏条目：配置库
悄悄丢条目，表现是"调优没生效"，排查方向会完全跑偏。保存用临时文件 +
`os.replace`，避免写一半留下截断 JSON。

**运行时上报：记录派发耗时，不冒充 GPU 时间**

`TelemetryRecorder.wrap(backend)` 返回一个 `InstrumentedBackend`：
`invoke` 前后计时，其余（容器钩子、能力、契约）全部委托。一次 `execute` 产出
一条 `RunRecord`：整图 `wall_ms`、逐节点 `dispatch_ms`、设备名、显存统计
（后端有 `MemoryContract` 才有）、以及调用方显式给的单位数（吞吐的分子）。

三个刻意的限制：

1. **`dispatch_ms` 不是 kernel 时间。** CUDA 是异步下发，`invoke` 返回时 kernel
   往往还没算完；这个数只回答"Python 派发是不是瓶颈"（DESIGN 5.4 里小规模
   CUDA 反而更慢就是这一类）。要看 GPU 时间得逐节点同步，那会把测量对象改成
   同步开销本身。`wall_ms` 含结尾 `to_host` 同步，是真实的完成时间。
2. **没有观测就是 `None`，不是 0。** 参考执行器没有设备、没有分层分配器，
   记录里就是 `device=None` / 没有 `memory` 字段；0 是一个具体观测值，
   两者混用会让"没有统计"看起来像"用了 0 字节"。
3. **包装器不凭空声称契约。** `device_info` / `memory_stats` 按内层**是否存在**
   逐项绑定，而不是无条件定义——否则参考执行器包一层之后会突然被
   `implemented_contracts` 认成实现了 `MemoryContract`，实际只能返回 `None`。
   只接受 `Executor` 子类（逐节点计时依赖 `invoke`），`ReferenceBackend` 这类
   只有 `execute` 的外观对象明确报错，建议改用 `ReferenceExecutor`。

**智能体接口：只读，且把"不做什么"写成测试**

`agent_api` 提供五个查询：`describe_backend` / `describe_operator` /
`performance_summary` / `recommend_config` / `snapshot`，返回全部可 JSON 序列化。
边界是硬约束：**没有执行入口、不接受代码或表达式、不写配置库**。
`describe_backend` 会剥掉上报包装层（问的是后端本身）；`recommend_config` 的
`source` 只有 `"store"`（实测过的最优）与 `"default"`（空间首候选）两个取值，
没有第三个——"猜一个"不是来源。测试里直接断言 `agent_api` 没有
`execute` / `eval` / `compile` 这些属性。

**验证**

| 层 | 覆盖 |
|---|---|
| 参数空间 | 空空间 / 空候选 / 重复候选 / 不可哈希候选被拒；展开顺序确定；默认=首候选 |
| 搜索器 | 取最小；平局取先出现者；异常与无效测量记为不可行；全不可行抛 `AllRejectedError`；`rounds` 取最小 |
| 配置库 | 往返、同键覆盖、设备不一致不命中、空白规范化、坏 JSON / 未知版本 / 缺字段显式失败、原子写无残留临时文件 |
| 上报 | 包不包上报输出逐位一致；节点数一致；无设备/显存时为 `None`；容量截断；摘要与落盘；不改图 |
| 智能体 | 能力与后端一致；未知算子/空名/重复后端名明确报错；配置库命中与默认回落；快照可 JSON 序列化；只读边界 |
| 真机 | 换 block（128/512/1024）后 relu 与 dropout_mask 逐位一致；搜索写库后可重新应用；上报能报出设备名 |

新增 59 条本地用例（调优 28、上报 14、智能体 17），另有 7 条 CUDA 真机用例。
全量 755 个测试：本地 554 通过 / 201 跳过；真机（RTX 2080 Ti）**755 全绿**
（19 跳过）。`verify_cuda.py` 的额外检查从 31 条扩到 32 条，新增
`tuning_roundtrip`（relu 候选 `[256, 128, 512, 1024]`，最优 `block=512`，
应用后与默认逐位一致）。

**明确没做的**

- **不做强化学习。** 配置库的更新规则只有一条：实测更快的配置覆盖旧记录。
  没有策略网络、奖励信号或在线探索——那要先有可信的测量与第二个后端。
- **不做逐节点 GPU 计时。** 见上，它会把测量对象改成同步开销。
- **不开放归约类与卷积类 kernel 的调优。** 前者 shared 预算与 block 绑定，
  后者的分块参数（`CONV_TILE_W` / OC tile）要连带改 codegen，是独立的一刀。
- **没有"配置库自动应用"**：`benchmarks/tune.py` 负责搜索与对拍，执行路径
  不会偷偷读库。要让某次运行用上配置，必须显式 `set_tuning`，否则
  "为什么这次快了"会变成隐式状态。
- **智能体接口不执行、不规划。** 它只回答"现在是什么"，不回答"下一步做什么"。

### 5.38　模糊测试、CI 作业与跨运行趋势

1.2 表里"模糊测试 / CI 流水线 / 吞吐与显存持续监控"三项此前分别是 ⬜ / 🟡 / ⬜。
这一节把它们推进到最后能推进的位置：前两项补齐，第三项从"没有"变成"有聚合、
无调度"。

**模糊测试：差分，而不是随机断言**

`aetherflow/fuzz.py` 不写"随机输入应该得到某个数"——那需要第二份语义实现，
而第二份实现同样可能错。它只验证与参考实现有关的不变量：

1. 随机拼出来的图能通过 `verify`；
2. `loads(dumps(g))` 与原图逐字段相等；
3. 同一输入执行两次逐位相同（`dropout_mask` 靠 attrs 里的 seed 也满足）；
4. 默认优化管线（Simplify → 常量折叠 → CSE → DCE）之后的图仍能校验、能序列化，
   且输出与优化前**逐位一致**；
5. 每轮再复制一个节点、注入一条非法属性或越界取值，要求构图期抛
   `ValueError` / `TypeError` / `KeyError`。

第 5 条比前四条更重要：**算错会被对拍抓住，而"非法输入被接受"不会**——
它让一张语义未定义的图安静地流进后端，直到某个 kernel 读到越界下标。所以
"变异被接受"会直接记成失败，不是提示。

生成器是模板式的：每个模板从当前值池里挑输入，形状推导全交给 `op.infer`，
不手写形状规则（手写一份就会与实现分叉，而这正是要测的东西）。输入取
`|N(0,1)| + 0.5`，让 `log` / `sqrt` 也能进入随机图而不用额外插保护节点；
除零产生的 inf 与 0/0 的 NaN 不判失败——参考实现与优化后的图都要给出同样的
NaN 位置，`array_equal(equal_nan=True)` 正好覆盖这一点。

**覆盖范围是克制的**：随机生成只碰 25 个"形状与数值"类算子（逐元素、广播二元、
matmul、softmax、reduce_sum、reshape、transpose、cast、slice、broadcast_to、
pad、concat、step/spike、dropout_mask、constant）。卷积、池化、attention、scan、
embedding 依赖复杂的几何属性，随机拼出来的失败原因会搅在一起；它们已有各自的
专项用例。随机图同样只用静态形状——符号维的绑定与校验另有专门用例。

**CI：三个作业，两种节奏**

- `cpu`：每次 push / PR，Python 3.10 与 3.11 全量测试 + 参考后端 conformance。
- `fuzz`：每次 push / PR，200 个固定种子。种子确定、无随机状态，失败时按日志里的
  `seed=N` 用 `--start N --iterations 1` 复现。
- `gpu`：手动触发，串起 `verify_cuda.py` → 全量测试 → CUDA conformance →
  25% 回归判定 → 调优往返（搜索 + 应用后逐位对拍）→ 趋势采集，并把
  配置库与趋势 JSON 作为 artifact 上传。

GPU 作业仍不放进 `push` 触发的路径：共享 runner 的抖动比 25% 的阈值还大，
放进去只会得到"时绿时红"的流水线。**它仍未接上真实的 self-hosted runner**，
所以 1.2 表里这一项保持 🟡——配置文件写完不等于跑过。

**趋势：先有可比的记录，再谈曲线**

`TelemetryRecorder.trend()` 按图分组，给出 wall、显存峰值、每个单位吞吐的
首末 / 极值 / 中位数 / 变化率 / 逐次斜率（最小二乘）。三个设计点：

- **分组键是 `label or graph`。** `RunRecord.graph` 是构图时写下的名字，
  `matmul_256` 与 `matmul_1024` 的图都叫 `matmul`；不引入 `label`，两条曲线
  会叠成一条没有意义的折线。`label` 只用于分组，不参与任何计算。
- **预热不入账。** `benchmarks/trend.py` 先用内层后端跑一次，再开始记录：
  第一次执行包含捕获 / 编译 / 首次分配，把它算进趋势会让第一点永远偏高。
- **斜率与首末一起给。** 只看首末会被一次抖动带偏；只看斜率会掩盖"某一次跳变"。
  两者并列，读者自己判断是漂移还是台阶。

`--save` / `--compare` 让 CI 每次只产出一小段、汇总时接起来：
`load_records(path)` 读回记录，`TelemetryRecorder.extend` 按时间顺序拼接。
`benchmarks/trend.py` 不负责画图——输出是 JSON 与表格，看板与告警属于展示层。

真机上的 2 次采样（`matmul_256` 0.766 ms → 0.435 ms，-43%）恰好说明趋势不能
当门禁：两次之间的整机状态差异远大于任何 kernel 变化。所以回归判定仍由
`benchmarks.run` 的多轮取最好 + 25% 阈值负责，趋势只回答"形状是否可疑"。

**验证**

| 项目 | 结果 |
|---|---|
| 本地全量 | `Ran 780 tests`，`OK (skipped=201)`（579 通过） |
| 200 个种子的模糊测试 | 200 例、200 次非法输入被拒绝、覆盖 25 个算子，全部通过（5.139 已扩到 31 个） |
| 1000 个种子的模糊测试 | 同上，全部通过（本地约 2 秒） |
| 远端全量 | `Ran 780 tests`，`OK (skipped=19)` |
| `verify_cuda.py` | conformance PASS；32 项额外检查 + 32 条边界用例全过 |
| CUDA 趋势 | `matmul_256` / `conv2d_1x1` 各 2 次，JSON 含 wall / 吞吐等全部实际观测字段；普通执行没有 arena，`peak_bytes` 按设计不出现 |

新增 25 条本地用例：模糊测试 9 条（可复现、图合法、序列化、变异、报告结构）、
趋势 6 条（用例筛选、元素数口径、预热不入账）、上报趋势 10 条（回落、窗口、
label 分组、斜率、落盘往返、坏文件拒绝）。

**明确没做的**

- **不把卷积 / 池化 / attention / scan / embedding 塞进随机生成器。** 它们的
  attrs 组合空间大且互相约束，随机生成大半会被 `infer` 拒绝，覆盖效率极低；
  专项用例更划算。（5.139 换了做法：先按固定素材的形状把属性算到自洽，只在
  合法区间里随机，于是这五类算子都进了随机图。）
- **不做符号维的随机化。** 符号绑定是执行器里唯一"运行期才知道"的逻辑，
  已有专门用例；混进随机图会让失败归因变难。
- **不做看板、告警、自动二分。** 趋势只产出 JSON 与表格；"哪次提交变慢"
  需要跨提交的采集与二分，那是另一套设施。
- **不用趋势自动判回归。** 见上，单机噪声足以制造假阳性；门禁仍看
  `benchmarks.run` 的同机基线与阈值。
- **CI 没有在托管平台实跑。** `fuzz` 与 `gpu` 作业的步骤都经过本地/远程手工等价
  验证（同样的命令在 RTX 2080 Ti 上通过），但"GitHub 真的调起了 runner"这件事
  没有发生，因此不宣称 CI 已闭环。

---

### 5.39　真实模型导入覆盖：把缺口留在前端，而不是 IR

**动机**：到 5.38 为止，"能读 PyTorch 导出的图"的证据是 LeNet 与手写的 Transformer
Block。这两张图都是照着 IR 现有算子设计的，等于**用 IR 的能力定义测试范围**。真实
模型不会迁就：`nn.BatchNorm2d` 导出的是 `_native_batch_norm_legit_no_training`，
squeezenet 的池化带 `ceil_mode=True`，ViT 的 GELU 默认是 erf 精确式。这一节把
"能不能跑别人的模型"变成一个可证伪的问题：**拿 7 个 torchvision 模型直接导出，
逐个对拍**。

**先盘点，再动手**

对 7 个模型（resnet18 / mobilenet_v2 / efficientnet_b0 / squeezenet1_0 /
shufflenet_v2 / densenet121 / vit_b_16）导出后统计未被识别的 aten 算子，去重后
只有 15 个。这个数字本身就是结论：**缺口几乎全在前端映射，不在 IR**。IR 里已有
`avgpool2d` / `concat` / `attention` / `batch_norm` / `reduce_sum` / `sigmoid` /
`gelu` / `broadcast_to` / `slice` / `reshape` / `constant`，足够承接一个真实 CNN。

**三类缺口，三种处理**

| 缺口 | 处理 | 为什么这么处理 |
|---|---|---|
| 语义等价、只是名字不同（`mean.dim` / `cat` / `sigmoid` / `expand` / `unsqueeze` / `squeeze` / `select` / `slice.Tensor`） | 直接映射 | `mean` 没有独立语义，`reduce_sum` + 除以静态元素数即可；不新增算子 |
| 语义不同、必须改表达（推理态 BN、`ceil_mode` 池化、`hardtanh`、精确 GELU） | 前端分解成已有算子 | 见下面三条 |
| IR 真的没有（`erf`） | 新增一元算子 | 精确 GELU 必须用 erf，tanh 近似差 ~1e-3，不能静默替换 |

**推理态 BN ≠ IR 的 `batch_norm`。** IR 的 `batch_norm` 用的是**当前 batch 的
统计量**（训练态，5.24 的定义）。推理图必须用 `running_mean` / `running_var`，
它是一个固定的仿射变换：

```
inv   = 1 / sqrt(running_var + eps)
scale = weight * inv
shift = bias - running_mean * scale
y     = x * scale + shift
```

关键在于 `scale` / `shift` 的形状是 `(C,)`，而 `x` 是 `(N,C,H,W)`——numpy 的广播
会把 `(C,)` 对齐到最后一维 `W`，**不会**报错，只会安静地算错。所以前端先
`reshape` 成 `(1,C,1,1)` 再乘，不依赖广播对齐。

**`ceil_mode` 的等价变换：复制边界，不是补零。** PyTorch 的 `ceil_mode=True`
在末尾多出一个窗口，越界元素被忽略。`pad` 补零会让一个全负窗口的最大值从"最大的
负元素"变成 0——语义就错了。正确做法是把边界值**复制（replicate）**到目标长度：

```
out    = ceil((size + 2p - k) / s) + 1
needed = (out - 1) * s + k - 2p
```

尾部 `needed - size` 个位置由 `slice` 取最后一列 + `concat` 拼出来。对 max 运算这是
精确等价的：复制最大值不改变最大值。

**精确 GELU。** PyTorch 的 `nn.GELU()` 默认 `approximate='none'`，用的是
`0.5x(1+erf(x/√2))`。IR 的 `gelu` 是 tanh 近似（5.20 刻意选的，与
`approximate='tanh'` 逐点一致），两者差约 1e-3——比所有其它新算子的 1e-6 量级
误差大三个数量级，不能"差不多"。所以：`approximate='tanh'` 走 `gelu`，
`'none'` 走 `0.5x(1+erf(x/√2))`（新算子 `erf` 由五个已有算子拼出来），其它
取值明确拒绝。`erf` 的 VJP 是 `2/√π · exp(-x²)`，CUDA 侧直接映射到 `erff`。

**torch.export 的一个坑：未被引用的 placeholder。** BN 的 `num_batches_tracked`
是 int64 标量 buffer，会被提升成图输入，但推理图里没人引用它。IR 的 dtype 体系里
没有 int64，直接按 placeholder 处理会在第一个模型上就失败。解法是先扫一遍
`all_input_nodes` 求出真正被引用的值，跳过其余——**不能靠"它是不是参数"来判断**，
因为 `running_mean` / `running_var` 同样不是参数，却必须喂进来（它们在
`inputs_to_buffers` 里）。

**验证**

| 模型 | IR 节点数 | 与 torch eager 的最大绝对误差 |
|---|---|---|
| resnet18 | 311 | 1.371e-06 |
| mobilenet_v2 | 1023 | 3.164e-15 |
| efficientnet_b0 | 829 | 4.955e-20 |
| squeezenet1_0 | 69 | 7.153e-07 |
| shufflenet_v2 | 918 | 1.819e-12 |
| densenet121 | 1881 | 8.345e-07 |
| vit_b_16 | 601 | 0.000e+00 |

7/7 通过（参考后端）。真机 CUDA 上另做了两条端到端：resnet18（311 节点，含
conv / BN / relu / add / maxpool / adaptive pool / linear）最大误差 1.79e-06，
squeezenet1_0（含 `ceil_mode` 分解）1.07e-06。两条都是"torch 导出 → IR JSON →
真机 CUDA 执行"，图与权重通过 JSON + npz 跨进程传输，未重新导出。

conformance 由 66 扩到 67 个用例（`erf`），CUDA 侧 46/46 算子覆盖、62 executed /
5 refused / 0 failed。本地全量 792 个测试（591 通过 / 201 跳过），真机 792 个
（skipped=31，缺 torch 那组）。

**明确没做的**

- **不做 fallback。** 遇到不认识的算子仍然直接报错，不静默跳过、不把整个子图
  交给 torch eager 执行。理由写在 ADR-005 与 `frontend/__init__.py` 的模块注释里：
  静默 fallback 会让"翻译是否保真"这个问题消失，也会把 torch 变成运行期依赖。
- **不覆盖训练态导入。** 训练态的 BN、`dropout(training=True)` 仍明确拒绝；能够
  导入的是**推理图**。训练图的来源是自己用 `aetherflow.models` 构图，不是导入。
- **不覆盖转置卷积、dilated 池化、`is_causal` 注意力、非零 dropout。** 前两个
  IR 没有对应语义，后两个需要额外的 mask / 种子语义；都显式报错（有测试钉住）。
- **没有跑更多模型。** 7 个是"覆盖到主流 CNN 骨架 + 一个 ViT"的最小充分集；
  检测 / 分割头（含 `interpolate` / `grid_sample` / NMS）没有覆盖。

### 5.40　HIP 后端：同一批内核落到第二家厂商

**动机**：M4 的硬件项是"真正的第二个硬件后端（不是 CPU 模拟）"，准入判据只有一条——
conformance 跑通。选 HIP/ROCm 的理由是它与 CUDA 的编程模型同源：`__global__`、
`__shared__`、`atomicAdd`、块/线程索引都能直接用，因此"跨厂商"这件事可以被压缩到
真正有差异的几处，而不是把 53 KB 的内核模板重写一遍。

**不复制后端**：为新硬件复制一份 `backend.py` + `codegen.py` 是最省事也最贵的做法——
两份代码从第一天起就在漂移（CUDA 补了边界检查，HIP 忘了）。实际的分工是：

| 层 | 做法 |
|---|---|
| 算子语义、形状检查、广播折叠、分块策略、能力边界 | 全部继承 `CudaBackend`，不复制一行 |
| 内核源码（46 个算子 / 57 个 f32 模板 + f16 变体） | 逐字复用 `cuda/codegen.py`；HIP 只注入 `%PRELUDE%` 处的方言差异 |
| 编译 | hipRTC（libhiprtc）→ AMDGPU ELF code object → `hipModuleLoadData` / `hipModuleGetFunction` / `hipModuleLaunchKernel` |
| 容器 | `hip/runtime.py`：ctypes 直连 `libamdhip64` 的轻量设备数组（不引入 cupy / torch） |
| 线性代数 | `hip/blas.py`：hipBLAS 的 sgemm / hgemm（GemmEx）与 strided-batched 变体 |
| 报错称呼 | 类属性 `_LABEL`：继承来的消息说"HIP 后端"，不再把使用者指向 CUDA |

**四处真实差异**（都在编译器 / 运行时层，不在 IR 层）：

1. **f16 存取**。CUDA 模板用 PTX 内联汇编 `asm("cvt.f32.f16")`，AMD clang 直接拒绝
   （`invalid output constraint '=f' in asm`）。换成 `_Float16` + `__builtin_bit_cast`；
   同时不能 include `hip/hip_fp16.h`——hipRTC 默认没有 include 搜索路径。
2. **无符号数原子加**。AMD 没有 `atomicAdd(unsigned short*)` 的重载，f16 的 scatter-add
   改成 `atomicCAS` 循环；`atomicAdd(float*)` 两家都有。
3. **库枚举值不同**。`HIPBLAS_OP_N=111`、`HIP_R_16F=2`、`HIPBLAS_GEMM_DEFAULT=160`
   与 cuBLAS 不是同一套数字，照抄会得到"参数合法但算错"的结果。
4. **内核参数打包**。`hipModuleLaunchKernel` 的 `kernelParams` 是指向参数值内存的数组，
   裸 Python int 会被截成 32 位（第一次探针就是这样 segfault 的）。`_pack` 只接受带
   ctypes 类型的持有者（另加 `np.uint64` 供 dropout seed），遇到裸 int 明确 TypeError。

以下三条是复用时踩到的硬约束，记下来省得下一家再踩：同一份源码重复编译会
`duplicate symbol`（每次编译必须新建 program）；hipRTC 的编译不需要 GPU，所以"模板能否
编译"可以在无显卡机器上先验证；内核的标量参数必须与内核签名逐一对齐，多一个少一个都是
静默的未定义行为。

**实测**（WSL2 + DXG 直通 / Ubuntu 22.04 / ROCm 7.2.53211 / RX 7650 GRE，gfx1100）：

```
result  : PASS
executed: 62   refused: 5   failed: 0
```

- 与 CUDA 真机（RTX 2080 Ti）逐项一致：同为 62 executed / 5 refused / 0 failed，
  拒绝项也相同（`matmul_partial_batch`、`softmax_axis0`、`layernorm_axis0`、
  `reduce_sum_axis0`、`f64_input_relu`），全部是 `BackendCapabilityError`。
- 连续多轮重复运行结果稳定；`tests/test_hip_backend.py` 把这条判据固化成准入测试
  （无 ROCm 的机器整组跳过，但模块仍可导入，构造失败会带明确原因而不是 ImportError）。
- 本地全量 803 个测试（591 通过 / 212 跳过：HIP 组 11 个用例因无 ROCm 跳过）；同一组
  在 WSL 里 11/11 通过。

**确定性的一条诚实记录**：`embedding_backward` 是 scatter-add（同一 token 在一个 batch
里出现多次是常态），两家都靠原子加完成，而浮点加法不满足结合律——两次运行的位级结果
可能差 1 ulp，HIP 上复现了。处理不是放宽数值比较，而是给用例加 `deterministic=False`：
它照常参与数值对拍（rtol/atol 1e-4），只是不参与"两次结果必须完全相同"，报告会把豁免
项列出来。PyTorch 的 `index_add` / embedding 反向同样非确定；把它藏起来才是虚假的确定性。

**明确没做的**

- **hipGraph 捕获**。CUDA 的 `capture` 依赖固定地址与专用 arena，HIP 侧这两件事都没做，
  `capture` 直接抛 `BackendCapabilityError`，不返回"能捕获但每次重算"的假计划。
- **捕获期 arena 与显存峰值统计**。`_last_arena` 恒为 None，`MemoryContract` 的方法在，
  但 `memory_stats()` 返回 None——不拿"当前分配量"冒充峰值。
- **调优与遥测**。`_blocks` 为空，等价于全部使用默认 block；`benchmarks/tune.py` 与
  `benchmarks/trend.py` 只接 CUDA。
- **性能数字**。本节结论全部是正确性，没有任何 HIP 与 CUDA 的吞吐对比。跨厂商性能比较
  要等两边在同一套基准（5.15 的 8 个固定用例）上跑过；在此之前说谁快没有依据。
- **多卡 / 多型号**。只有 gfx1100 一张卡；gfx9 的波前 64、`__shfl` 语义差异未触及。

### 5.41　后端装配：换硬件不改代码

5.40 证明了"新硬件只写插件"，但还留着一个装配点：迁移时得手写

```python
from aetherflow.backends.hip import HipBackend
backend = HipBackend()          # 原本是 CudaBackend()
```

前端逻辑一行没动，可这一行仍然是厂商名。装配层把这一行也拿掉。

**三个函数**

| API | 作用 |
|---|---|
| `get_backend(name=None, *, fallback=True, **kwargs)` | 挑一个后端并构造；`kwargs` 原样交给工厂，所以与直接构造完全等价 |
| `register_backend(name, factory, *, replace=False)` | 登记工厂；重名默认报错，覆盖必须显式 |
| `registered_backends()` | 当前可用名字（内置 + 已加载的第三方），排序后返回 |

**挑选顺序**（优先级从高到低）：显式传名 → 环境变量 `AETHERFLOW_BACKEND` →
自动探测。自动探测依次试 CUDA、HIP，第一个**真能构造出来**的胜出。

"真能构造"是实际初始化一次，而不是看布尔标志：`CUDA_AVAILABLE` 只表示 cupy 装没装，
没装驱动时构造仍会失败。探测只付一次失败尝试的代价，换来的是不会出现"以为在用 GPU、
其实在跑 CPU"。

**退回必须可见**：两边都失败时，默认退回参考后端，同时发一条 `RuntimeWarning`
（`backend.name == "reference"` 是第二个证据）。要硬性保证就传 `fallback=False`，
直接抛 `BackendUnavailableError`。三个异常各管一段：

- `LookupError`：名字没注册（消息里列出已注册的名字）；
- `BackendUnavailableError`：名字在，但这台机器初始化失败（`__cause__` 保留原始异常）；
- `BackendCapabilityError`：设备拿到了，但这张图算不了——仍是执行期的事。

**第三方硬件包**通过 pip entry point 接入，不改本仓库一行代码：

```toml
[project.entry-points."aetherflow.backends"]
mychip = "mychip.backend:MyChipBackend"
```

装包后 `get_backend("mychip")` 即可用。同名时第三方被忽略并记录原因，内置后端不会被
"装了个包"顶掉；一个坏插件加载失败也只影响它自己。

**惰性工厂**：内置三个后端的工厂都是 `from .cuda import CudaBackend` 这种延迟导入，
`import aetherflow` 不会因此加载 100 KB 的 CUDA 插件，也不会因为没装 cupy 而失败。

**实测**

```text
本地 Windows（无 GPU）：registered = ('cuda','hip','reference') → auto 选中 reference 并警告
WSL2 + RX 7650 GRE   ：auto 选中 hip，AETHERFLOW_BACKEND=hip 同样选中 hip
```

`tests/test_backend_discovery.py` 16 个用例钉住注册、覆盖、错误分类、kwargs 透传、
环境变量优先级、大小写与空格归一化，以及"退回必发警告 / 可关闭退回"。
本地全量 819 个测试（591 通过 / 212 跳过），WSL 里该组 16 个通过（2 个按硬件条件跳过）。

**明确没做的**

- **不做多后端拼图**：不会把一张图拆给几个后端分别执行。能力不够仍走
  `BackendCapabilityError`（ADR-005 的决定）。
- **不做设备选择**：只管厂商，不管"用第几张卡"。多卡与 stream 仍在插件内部。
- **不缓存实例**：每次 `get_backend` 都新建。后端持有常驻参数与设备缓冲，共享一个
  实例会把"谁的权重"搅在一起；要不要单例由调用方决定。

### 5.42　L2 内核 IR：把方言从模板里拆出来

**动机**：46 个算子的内核是手写 C 字符串，差异靠五个占位符表达（`%PRELUDE%` /
`%T%` / `%LD%` / `%ST%` / `%ATOMIC%`）。HIP 复用这套模板时只注入了 f16 的存取实现
（5.40），这恰好证明：**模板本身是硬件无关的，厂商差异只在方言**。既然如此，模板
字符串就不该是唯一的表达方式——它不可检视、不可逐条改写，也没法让第三方硬件
"只描述方言就接进来"。

**分层归属**

| 层 | 内容 | 归属 |
|---|---|---|
| L1 语义 IR | 算子 + 类型 + 符号维 | `ir/`（已有） |
| L2 内核 IR | 线程索引、循环、共享内存、表达式、读写 | `lower/`（本节产物，硬件无关） |
| 方言 | 类型映射、f16 存取、prelude | `backends/*/dialect.py`（是**数据**，不是逻辑） |
| 源码 | CUDA C / HIP C | `backends/*/codegen.py`（渲染入口；十八个内核族已由 L2 生成） |

**实现**

- `lower/kernel.py`：27 个表达式 / 语句节点。表达式 `Var` / `Num` / `Bin` / `Neg` /
  `Paren` / `Cond` / `Cast` / `Call` / `Index` / `Axis` / `Load` / `Widen` / `Narrow`，
  语句 `Let` / `LetMulti` / `ArrayDecl` / `Ptr` / `Shared` / `For` / `If` / `Assign` /
  `AugAssign` / `Store` / `AtomicAdd` / `Continue` / `Sync` / `Return`，容器 `Kernel` /
  `Tensor` / `Scalar`。
  `Neg` / `Cond` / `LetMulti` / `ArrayDecl` 是扩到一元、step、广播、softmax 时才加的；
  `For` 增加了 `cmp` / `incr` / `type_name`（倒序循环与 `int` 计数），`Tensor` 增加了
  `type_name`（softmax 的 `scale` 固定为 `const float*`）。5.45 又补了 `Widen` /
  `Narrow` 与 `Let(type_name=None)`——融合残差的 LayerNorm 需要在标量上做
  "存储类型 ↔ 计算宽度"的往返，`Load` 只覆盖了带索引的缓冲。只补到"现有模板能被
  逐 token 表达"为止，没有为通用性提前设计。
- `lower/cstyle.py`：C 语法打印器 + `CStyleDialect`（类型映射、存取函数名、prelude）。
  `CStyleDialect` 是数据，所以新增一家硬件只需要在 `backends/<name>/dialect.py` 填表，
  打印器不动。
- `lower/kernels.py`：十二个参考内核的 L2 描述（`ew_binary`、`ew_unary`、`ew_step`、
  `ew_broadcast`、`reduce_sum_last`、`softmax_last`、`layernorm_last`、
  `layernorm_residual_last`、`rmsnorm_last`、`silu_gate`、`batch_norm`、
  `batch_norm_stats`）。
- 括号是显式节点（`Paren`）：源码的括号结构属于语义，打印器不替作者推断优先级。

**验收**

1. **与手写模板逐 token 相等**：去掉空白与注释后，`emit` 的输出与
   `codegen.render(手写模板, dtype)` 完全一致。覆盖 add/sub/mul/div、十种一元算子、
   step、广播、`reduce_sum_last`、`softmax_last`，每种都跑 f32 与 f16。
2. **能被真机编译**：四份源码（2 内核 × 2 dtype）在 WSL 的 hipRTC 下全部编译成功，
   分别产出 5072 / 5384 / 6480 / 5384 字节的 AMDGPU code object。
3. **接管生产路径后验收不变**：`binary_source` / `reduce_sum_source` 改成返回
   `Kernel`，由 L2 打印出源码，其余 44 个算子的路径不动。CUDA 真机（RTX 2080 Ti，
   driver 550.127.05）conformance 仍是 62 executed / 5 refused / 0 failed，全量
   834 个测试 790 通过 / 44 跳过——与切换前一致。
4. **扩到六个内核族后验收不变**：`unary_source` / `step_source` /
   `broadcast_binary_source` / `softmax_source` 也改成返回 `Kernel`。新增内核同样
   满足"f32 两家一字不差、f16 只差 prelude"；本地全量 876 个测试 664 通过 / 212 跳过。
   CUDA 真机（RTX 2080 Ti，driver 550.127.05）全量 876 个测试 832 通过 / 44 跳过，
   conformance 62 executed / 5 refused / 0 failed，`verify_cuda.py` 的边界用例全部通过。

方言边界的第二个判据：f32 完全不碰 f16 方言，CUDA 与 HIP 的输出必须**一字不差**；
f16 只允许 prelude 不同（PTX 的 `cvt` vs `_Float16` + `__builtin_bit_cast`），
内核主体必须相同。这两条都由测试钉住。

**覆盖范围与边界**

- 覆盖十八个内核族：逐元素二元 / 一元 / step / 广播 / 门控 SiLU（网格步进）、单 block
  归约、softmax、LayerNorm（含融合残差）、RMSNorm、批归一化（含统计量）、
  maxpool2d（含反向）、avgpool2d（含反向）、自适应平均池化（含反向）。attention、
  卷积、索引类仍是手写字符串。
- matmul 没有内核：它走 cuBLAS / hipBLAS。线性代数的可移植性由库契约保证，不是
  L2 的职责——所以"三个算子"里 matmul 这一项从内核 IR 的角度不成立。
- **生产路径已切换，但只切了十八个入口**。`binary_source`（add/sub/mul/div）、
  `unary_source`、`step_source`、`broadcast_binary_source`、`reduce_sum_source`、
  `softmax_source`、`layernorm_source`、`layernorm_residual_source`、
  `rmsnorm_source`、`silu_gate_source`、`batch_norm_source`、
  `batch_norm_stats_source`、`maxpool2d_source`、`maxpool2d_backward_source`、
  `avgpool2d_source`、`avgpool2d_backward_source`、`adaptive_avg_pool2d_source`、
  `adaptive_avg_pool2d_backward_source` 现在返回 L2 的 `Kernel`，由 `render()` 交给
  方言打印。旧的字符串实现保留为 `*_template()`，测试仍拿它当逐 token 对照物——它是
  "方言打印器没写错"的证据，不是回退路径。

**明确没做的**

- **语义级 L2**：循环/索引的抽象表示、布局决策、自动分块、向量化。现在这层是
  语法级的（一条语句对应一条语句），能检视、能双路打印、能局部改写，但不带来
  任何优化。自动调优要的是语义级那一层，得先有索引分析。
- **L2 覆盖全部算子**：现在十八个内核族，attention / 卷积 / 索引类等仍是手写模板，
  第三方包仍要自带这些算子的胶水。5.44 的 novac 示例证明的是"外围链路（安装 →
  识别 → 打印 → 执行）已经通了"，不是"所有算子的内核都能由方言生成"。

### 5.43　zip 硬件包：放进目录就识别

**动机**：pip entry point 要求"先打包、再 `pip install`"（5.41）。对第三方硬件
厂商，这是一次完整的环境操作；对现场部署，"装个驱动就要重建虚拟环境"也说不通。
而硬件包里真正需要变的东西往往只有方言与少量运行时胶水——把它做成一个 zip，
解压到固定目录即可被识别，换硬件不必动环境。

**格式**（zip，根目录必须有 `manifest.json`）

```json
{
  "name": "mychip",
  "version": "0.1.0",
  "aetherflow_api": "1",
  "factory": "mychip.backend:MyChipBackend",
  "vendor": "可选",
  "description": "可选"
}
```

`factory` 是 `module:attr`，模块路径相对解压目录；`aetherflow_api` 只比**主版本**
（次版本只增不减，不该阻止旧包加载）。

**识别位置**（先命中者优先，同名只认第一个）

| 顺序 | 位置 | 用途 |
|---|---|---|
| 1 | `$AETHERFLOW_HARDWARE_DIR` | 存在时**只**扫这一个：测试隔离与只读部署 |
| 2 | `~/.aetherflow/hardware` | `install_hardware()` 的安装目标 |
| 3 | `aetherflow/hardware` | 随框架分发，只读 |

**安装流程**：解压到同目录的临时目录 → 校验 manifest → 检查不可占用内置名 →
原子改名到 `<root>/<name>` → 立即 `register_backend`（`_discover` 只扫一次，
不主动登记的话"装完要重启"）。中途失败整个临时目录删掉，不会留下半个包，也不会
破坏已安装的同名版本。

**安全边界**（都在安装期，不留到运行期）

- 路径穿越 / 绝对路径 / 反斜杠分隔符一律拒绝（`../evil.py`、`/tmp/evil.py`）；
- 符号链接条目拒绝（zip 里的 symlink 可以指向包外）；
- 条目数上限 2000、解压后总量上限 64 MB（挡 zip bomb 与误打进包里的权重）；
- `name` 必须是 `^[a-z][a-z0-9_]*$`；`cuda` / `hip` / `reference` 三个内置名不允许
  被占用——装一个叫 `cuda` 的包去顶掉真 CUDA 后端，是"看起来装好了、其实跑的是
  别人实现"的标准事故；
- 加载时把包目录**追加**到 `sys.path` 末尾，并检查目标模块的 `__file__` 是否真在
  包内：同名模块已被占用时明确报错，而不是悄悄用别人的模块。

**与 IR 的分工（对"IR 层根据硬件做三层转译"这个想法的明确回答）**

转译不住装配层。一条 IR 落到某家硬件上，路径固定为：

```
L1 语义 IR（ir/） → [可选] 中层 pass → L2 内核 IR（lower/） → 方言（backends/*/dialect.py）→ 源码
```

硬件包能替换的是**最后两段**（方言数据 + 运行时胶水），它不解释 IR。如果把转译
逻辑搬进装配层，每家硬件就得自己解释一遍 IR，"一套 IR 落到不同硬件"立刻退化成
N 套私有 IR——这正是 5.40 复用同一批模板、5.42 抽 L2 想避免的事。

**验收**

- 24 个用例（`tests/test_hardware_package.py`；5.64 又加了 16 条摘要与依赖用例，
  现共 40 条）：装完即可用（含 kwargs 透传）、
  manifest 六类非法输入、路径穿越 / 绝对路径 / symlink / 条目数上限 / 非 zip、
  坏包在扫描时跳过且不拖累别人、卸载后名字立刻从可选中消失、手动放进目录的包
  仅靠扫描被识别、工厂目标缺失时错误带 `__cause__`、内置名不可注销。
- 全量 876 个测试：本地 664 通过 / 212 跳过，无回归。
- 真实 zip 的端到端示例见 5.44（novac，6 个用例）。
- 真机复跑发现一个测试自身的环境缺陷：`test_not_a_zip_is_refused` 用
  `Path.write_text` 写中文，在 Linux 的 POSIX locale 下按 ASCII 编码而失败。已显式
  指定 `encoding="utf-8"`——测试不该依赖运行环境的默认编码。

**明确没做的**

- **依赖的自动安装**：包可以声明 `requires`（5.64），但缺的第三方库仍要用户自己装。
  加一层解析与安装就变成包管理器了，不是这一层的目标。
- **版本共存**：同名包只能有一个版本，`replace=True` 覆盖。
- **来源签名**：5.64 补了内容摘要（能证明"包在打包后没被改过"），但不证明"这个包
  是谁发的"——伪造者可以连清单一起重算。可信分发需要签名链，得先有发布渠道。
- **包内模块名冲突的自动隔离**：靠 manifest 作者给模块加前缀，冲突时明确报错。

### 5.44　示例硬件包 novac：只填方言就能接入

**目的**：5.42 证明内核可以由 L2 打印，5.43 证明包可以装进来。合起来要回答的是一个
更硬的问题——第三方**只提供方言数据**，能不能接进来？`examples/hardware/novac` 是这个
问题的最小答案：全部接入代码是一份 `CStyleDialect`（存储类型 + 存取函数名 + prelude）
加一个只有 `name` / `capabilities` / `execute` 的后端类，框架侧零改动。

**产物**

| 路径 | 内容 |
|---|---|
| `examples/hardware/novac/manifest.json` | 包元信息，`factory` 指向 `novac.backend:NovaCBackend` |
| `examples/hardware/novac/novac/backend.py` | 方言数据 `NOVAC_DIALECT` + `NovaCBackend` + `nova_source()` |
| `examples/hardware/build.py` | 按固定清单打 zip（先列文件再写，避免把输出自己打进去） |
| `examples/hardware/novac-0.1.0.zip` | 交付物，2.9 KB / 3 个条目，可直接 `install_hardware` |

Nova C 是虚构的加速器（与 CUDA 同一套 C 语法），存在的意义是让示例在没有对应硬件的
机器上也能跑完"安装 → 识别 → 出源码 → 执行"整条链路。真实硬件把 `source_for` /
`execute` 换成自己的编译与启动即可，方言数据不用动。

**验收（`tests/test_example_hardware.py`，6 个用例）**

- 装完即可用，`get_backend("novac", block=256)` 的 kwargs 透传到工厂；
- **新进程**只靠目录扫描识别（不写注册表、不改环境）；
- 同一份 `ew_binary("add")` 用 novac 方言打印，把 `nv_*` 换回 `af_*` 后与 CUDA
  逐 token 相等——证明差异只在方言，不在内核结构；
- f16 只换 prelude（`__half2float` / `nv_ld_h`），主体不变；
- 图能跑出结果（示例用参考执行器代替驱动）。

**边界**

- 归约辅助函数名 `block_reduce_sum` / `block_reduce_max` 仍写死在 L2
  （`lower/kernels.py`），方言只能换实现、不能换名字。这是已知的词汇表泄漏，等第二个
  真实硬件提出改名需求再参数化，现在不为假想需求加一层。
- 示例不绑定任何厂商 SDK：`execute` 不做真实编译与启动。它是"接入点在哪"的说明书，
  不是某个驱动的封装。

### 5.45　归一化内核迁到 L2

**动机**：LayerNorm / RMSNorm 是 Transformer 里除 matmul、softmax 之外最常驻的内核，
模板早就有了，但一直是字符串——第三方硬件即便只差一份方言，也用不上它们。这一批把
三个归一化内核（普通、融合残差、RMSNorm）迁到 L2，同时补上模板需要、而此前没有任何
内核用到的节点。

**新增节点（由模板倒推，不是提前设计）**

| 节点 | 打印 | 模板里为什么需要 |
|---|---|---|
| `Widen(expr)` | `af_ld_*(expr)` | `%LD%(xv)`：`xv` 是标量变量，不是带索引的缓冲，`Load` 表达不了 |
| `Narrow(expr)` | `af_st_*(expr)` | `%ST%(%LD%(xr[i]) + %LD%(rr[i]))`：f16 下 `x + res` 必须先舍入回存储类型 |
| `Let(type_name=None)` | `%T% xv = ...` | 局部变量是存储类型而不是 `float` |

`Narrow` 不是精度修饰，是语义的一部分：numpy 里两个 f16 相加得到 f16，融合残差如果
一路用 float 累加到最后才舍入，结果会和参考实现对不上。`Load` 则保留原样——它是
`Widen(Index(...))` 的特化，但几乎所有内核都在用，没有改写成组合形式的理由。

**产物**

- `lower/kernels.py`：`layernorm_last()` / `layernorm_residual_last()` /
  `rmsnorm_last()`，参数顺序与扫描结构照手写模板逐 token 对齐。
- `cuda/codegen.py`：三个 `*_source()` 返回 `Kernel`，同名 `*_template()` 留作对照物。
- `backend.py`：`_op_layernorm` 的两条分支（有 / 无残差）与 `_op_rmsnorm` 切到新入口。

**验收**

- `tests/test_lower_kernel.py` 新增 7 个用例：三个内核 × f32/f16 与手写模板逐 token
  相等，外加一条"f16 下存储舍入仍然存在"的断言；九个内核族统一满足"f32 两家一字
  不差、f16 只差 prelude"。
- 本地全量 883 个测试 671 通过 / 212 跳过。
- CUDA 真机（RTX 2080 Ti，driver 550.127.05）：`verify_cuda.py` 全部 PASS（其中
  `layernorm_residual_odd_cols`、`layernorm_residual_f16` 正好覆盖这次改动的内核），
  全量 883 个测试 839 通过 / 44 跳过。

**边界**

- `block_reduce_sum` / `block_reduce_max` 的名字仍写死在 L2，与 5.44 是同一条已知
  泄漏：方言只能换实现、不能换名字。
- 三个内核都只支持最后一维；融合残差要求 `res` 与 `x` 同形状、同 dtype，由后端拒绝
  而不是内核内补齐。
- `Widen` / `Narrow` 只表达"当前 dtype 的存储宽度"，不承担任意类型转换——`(float)cols`
  这类显式转换仍然走 `Cast`。

### 5.46　门控 SiLU 与批归一化迁到 L2

**动机**：5.45 之后，模板里最常用的两个还没迁的是 `silu_gate`（SwiGLU 的融合逐元素）
与 `batch_norm` / `batch_norm_stats`。它们各自代表一种此前没覆盖过的写法：前者是
"一个内核读两块输入、写一块输出"的逐元素形态，后者是"外层循环 × 内层线程扫描 ×
通道切片"的三层嵌套。

**没有新增节点**。两个内核只用现有节点就能逐 token 表达，需要的只是别把结构写复杂：

- `for (long long o = 0; o < outer; ++o)` 用 `For(..., incr="++")`；
- `const %T%* base = x + o * stride + c * inner;` 用 `Ptr`，且**允许在同一个内核里
  重复声明同名 `base`**——它们分处不同循环块作用域，C 语义合法；
- `(float)(outer * inner)` 用 `Cast("float", Paren(...))`，括号仍是显式节点；
- 两个内核共用的前缀（通道定位 / 共享内存 / stride / 元素数）与两次归约抽成了
  `_batch_norm_prefix()` / `_batch_norm_reductions()`，这是 Python 侧的复用，
  不是新的 IR 抽象。

**产物**

- `lower/kernels.py`：`silu_gate()`、`batch_norm()`、`batch_norm_stats()`。
- `cuda/codegen.py`：三个 `*_source()` 返回 `Kernel`，`*_template()` 留作对照物。
  批归一化的手写模板一份字符串里含两个内核（原先一起交给 NVRTC），拆开后各自编译，
  对照时按内核名截取。
- `backend.py`：`_op_silu_gate`、`_op_batch_norm`、`_op_batch_norm_stats` 切到新入口。

**验收**

- `tests/test_lower_kernel.py` 新增 6 个用例：`silu_gate` f32/f16 整份对照，
  `batch_norm` / `batch_norm_stats` f32/f16 按名字截取对照；十二个内核族统一满足
  "f32 两家一字不差、f16 只差 prelude"。
- 本地全量 889 个测试 677 通过 / 212 跳过。
- CUDA 真机（RTX 2080 Ti，driver 550.127.05）：conformance 62 executed / 5 refused /
  0 failed（`batch_norm` 误差 2.4e-07、`batch_norm_stats` 6.0e-08），`verify_cuda.py`
  全部 PASS（含 `batch_norm_axis0`、`batch_norm_single_sample`、
  `batch_norm_stats_axis0`、`silu_gate_odd_length`、`silu_gate_f16`），全量 889 个测试
  845 通过 / 44 跳过。

**边界**

- 批归一化的方差是**两遍法**（先求均值，再扫一遍求偏差平方和），与 `numpy.var` 的
  参考实现一致；没有换成 Welford 单遍算法——那会改变逐元素结果的最后几位，
  而现在的验收判据是"与参考实现逐位对齐"。
- 内核只认 `(outer, channels, inner)` 三元组，`axis` → 三元组的折算在调用方
  `_batch_norm_plan`。多轴归一化仍然拒绝。

### 5.47　池化家族迁到 L2

**动机**：池化是最后一批"结构复杂"的手写模板——它们同时用了三种此前没有内核表达的
写法：把展平的输出下标拆成 `(n, c, ho, wo)`、对窗口做双层扫描并在越界时跳过、
反向把梯度**累加**回重叠的输入单元。六个内核（maxpool2d / avgpool2d / 自适应平均池化
及三者的反向）迁到 L2 后，`codegen.py` 里剩下的字符串模板只有 attention、卷积与
索引类。

**新增节点（由模板倒推）**

| 节点 | 打印 | 模板里为什么需要 |
|---|---|---|
| `AtomicAdd(buf, index, value)` | `%ATOMIC%(&buf[index], value);` | `%ATOMIC%(&base[...], g)`：重叠窗口会命中同一个输入单元，普通写入只留最后一个贡献 |
| `Continue()` | `continue;` | `if (ih < 0 \|\| ih >= H) { continue; }`：padding 区域不参与比较，也不能被散射 |

方言因此多了**可选能力** `atomic`（`DType → 函数名`）。它是可选的：只打印逐元素内核的
方言不必填表，真要用到 `AtomicAdd` 时才由 `atomic_of()` 报出"缺哪个 dtype"的明确错误
——与 `storage` / `load` / `store` 的懒检查是同一条设计。CUDA 与 HIP 都填
`af_add_f` / `af_add_h`；f16 的原子加是 CAS 循环（与 5.40 的 `af_add_h` 同一份实现）。

**产物**

- `lower/kernels.py`：`maxpool2d()` / `maxpool2d_backward()` / `avgpool2d()` /
  `avgpool2d_backward()` / `adaptive_avg_pool2d()` / `adaptive_avg_pool2d_backward()`。
  输出下标拆分、图像切片、窗口扫描、自适应窗口边界抽成了 Python 侧 helper
  （`_split_output_index` / `_image_slice` / `_window_scan` / `_adaptive_bounds`），
  这是构造内核时的复用，不是新的 IR 抽象——每个 helper 返回的都是既有节点的元组。
- `cuda/codegen.py`：六个 `*_source()` 返回 `Kernel`，同名 `*_template()` 留作对照物。
- `backend.py`：六个池化入口（`_op_maxpool2d`、`_op_maxpool2d_backward`、
  `_op_avgpool2d`、`_op_avgpool2d_backward`、`_op_adaptive_avg_pool2d`、
  `_op_adaptive_avg_pool2d_backward`）切到新入口，HIP 由 `__getattr__` 透传自动跟上。

**验收**

- `tests/test_lower_kernel.py` 新增 3 个用例：六个内核 × f32/f16 与手写模板逐 token
  相等（循环内带名字断言）、三个反向内核确实打印 `af_add_f(&`、f16 反向打印
  `af_add_h(&`；十八个内核族统一满足"f32 两家一字不差、f16 只差 prelude"。
- `verify_cuda.py` 新增两个真机边界用例：`maxpool2d_pad_stride`（窗口伸出边界，
  越界采样被 `continue` 跳过）与 `maxpool2d_backward_ties_overlap`（步长 1 + padding 1
  的重叠窗口，输入量化到整数格点以制造并列最大值，钉住"取第一个最大值"的平局规则）。
- 本地全量 892 个测试 680 通过 / 212 跳过。
- CUDA 真机（RTX 2080 Ti，driver 550.127.05）：conformance 62 executed / 5 refused /
  0 failed（`maxpool2d_backward` 0.0e+00、`maxpool2d_2x2` / `maxpool2d_3x3_pad1`
  0.0e+00、`avgpool2d` 6.0e-08、`adaptive_avg_pool2d_backward` 1.5e-08），
  `verify_cuda.py` 全部 PASS（含上面两个新用例），全量 892 个测试 848 通过 / 44 跳过。

**边界**

- 反向内核假设调用方已把 `dx` 清零（后端用 `zeros` 分配），与手写模板一致；内核不做
  初始化，否则每个线程都要先写一遍整块缓冲。
- `avgpool2d` 的除数是**完整核面积**（`count_include_pad=True` 语义），不是有效采样点
  数；`adaptive_avg_pool2d` 的除数才是窗口自身面积。两者的 `scale` / 面积计算留在
  调用方，内核只做乘除。
- 自适应窗口的 floor/ceil 用整数除法表达（`(o*H)/OH` 与 `((o+1)*H + OH - 1)/OH`），
  避免浮点边界抖动；这与 `ir/ops.py` 的参考实现逐边界一致。
- 索引类算子（gather / scatter / slice 等）仍在模板里：它们的循环体是"拷贝一段连续
  区域"，与池化的窗口扫描不是同一种结构，留到下一批再判断是否需要新节点。

### 5.48　拼接 / 随机 / 扫描 / 卷积梯度迁到 L2

**动机**：剩余 9 份字符串模板里，这 6 份属于"结构简单但写法各异"的一组：拼接是纯搬运、
dropout 需要一段设备函数、扫描是嵌套的串行递推、卷积两个反向是三重 gather 累加。
它们需要的表达能力各不相同，正好检验 L2 是否只是"给常见逐元素内核准备的"。

**新增能力（由模板倒推）**

| 能力 | 打印 | 模板里为什么需要 |
|---|---|---|
| `Store(..., raw=True)` | `buf[index] = value;` | concat 是纯搬运，值本来就是存储类型；套 `af_st` 会多一次"读成 float 再舍入"，f16 下改变语义 |
| `Kernel(helpers=...)` | 方言 prelude 之后的设备函数 | dropout 的 `splitmix64` 只服务于它自己；放进全局 prelude 会让其余算子的源码都多一段无用函数 |

`For` 早已有 `incr` / `cmp` / `type_name` 三个字段，这一批才第一次用到 `++t` / `--t`
的递增写法与 `int` 循环变量——扫描的倒序递推与卷积的窗口遍历都靠它，没有新增节点。
`_flat_loop` 增加了 `index` / `count` / `stride` 三个参数：模板里的循环变量叫 `i`、
边界叫 `elems`、步长叫 `stride`，参数化之后仍然逐字对齐，这只是 Python 侧复用。

**产物**

- `lower/kernels.py`：`concat_copy()` / `dropout_mask()` / `scan_linear()` /
  `scan_linear_backward()` / `conv2d_input_grad()` / `conv2d_weight_grad()`；
  `_conv_tap` 拼 `((long long)(b * ch_dim + ch) * y_dim + y) * x_dim + x`，
  四个 gather 点共用同一份索引构造。
- `cuda/codegen.py`：六个 `*_source()` 返回 `Kernel`，同名 `*_template()` 留作对照物。
- `backend.py`：`_op_concat`（拼接与 pad 共用）、`_op_dropout_mask`、
  `_op_scan_linear` / `_op_scan_linear_backward`、`_op_conv2d_input_grad` /
  `_op_conv2d_weight_grad` 切到新入口；HIP 由 `__getattr__` 透传自动跟上。

**验收**

- `tests/test_lower_kernel.py` 新增 7 个用例：四个测试类覆盖六个内核 × f32/f16 与手写
  模板逐 token 相等；cat 的 f16 源码里没有 `af_ld_` / `af_st_`（钉住逐位搬运）、
  `splitmix64` 只随 dropout 出现、两个卷积反向不打印 `af_add_`（gather 不需要原子加）。
- L2 的对照用例从 43 增至 50。
- 本地全量 899 个测试 687 通过 / 212 跳过。

**边界**

- concat 对 f16 是**逐位复制**：这是拼接/补零的语义，不是精度优化。若改成读成 float
  再写回，f16 的 payload 会在往返中变化。
- dropout 的掩码只取决于 `(seed, 展平下标)`，与块的几何无关，因此换 block size 或换
  设备仍然逐位一致；SplitMix64 只做整数乘法与异或，参考实现可以照着写。
- 扫描的并行度来自尾维（batch / channel / state），时间轴本身串行；分块（Blelloch）
  扫描需要第二遍，本轮不做。（**5.128 已补**：`chunked_scan=`，块内递推逐字不变。）
- 卷积两个反向都是 gather：每个 `dx` / `dw` 元素只有一个线程写，因此不用 float 原子加，
  结果是确定的。代价是每个线程要跳过不整除的窗口，属于"语义正确优先"的写法。
- 索引类之外还剩三份模板：im2col、分块卷积、注意力。它们需要动态共享内存与多阶段
  循环（已在 5.49 处理）。

---

### 5.49　im2col / 分块卷积 / 融合注意力迁到 L2

**动机**：最后三份字符串模板。前两批迁的都是"静态共享内存 + 单层网格步进"的形状；
这三份第一次需要**动态共享内存**（启动时按字节数传入）、**指针别名**（把一段共享内存
切成三段视图）、以及**内核私有的随机函数**。把它们的写法逼进 L2，等于把"模板里还能
出现什么样的语句"这件事走完一遍。

**新增能力（由模板倒推）**

| 能力 | 打印 | 模板里为什么需要 |
|---|---|---|
| `Not(expr)` | `!x` | 分块卷积与注意力都有 `if (!active)`；`Neg` 只能打印 `-x`，而判据是逐 token 相等，打印器不能靠类型猜 |
| `Shared(..., extern=True)` | `extern __shared__ float sw[];` | 动态共享内存没有编译期尺寸；静态那份继续用 `count`，两种来源共用一个节点 |
| `ArrayDecl(values=None)` | `float acc[8];` | 分块卷积的累加器数组在 `for` 里按下标清零，写成初始化列表反而写不出来 |

`Let` 的显式类型这次承担了"指针别名"：`Let("const float*", "xn", ...)` 打印
`const float* xn = x + ...`。没有新增 `Ptr` 变体，是因为这三个内核的参数**固定是
float**（不跟 dtype 走），而 `Ptr` 会按方言的存储类型打印——用 `Let` 反而更准确。
`For.cmp` 也被复用到极限：注意力内层的 `j < blockDim.x && t0 + j < T` 是"第一个比较
由 `cmp` 提供、其余由 `stop` 表达式给出"。

**产物**

- `lower/kernels.py`：`im2col()` / `conv2d_tiled()` / `attention_fwd()`；分块卷积的
  权重落位与累加内层抽成 `_tiled_weight()` / `_tiled_accumulate()`，两处复用；
  注意力的 `attention_rand` 走 `Kernel(helpers=...)`，只随它自己出现。
- `lower/kernel.py`：新增 `Not`、`Shared(extern=)`、`ArrayDecl(values=None)`，节点总数
  28（14 表达式 + 14 语句）。
- `cuda/codegen.py`：三对 `*_source()` / `*_template()`。
- `backend.py`：`_conv2d_tiled`、`_conv2d_im2col`、`_op_attention` 三处切换；HIP 经
  `__getattr__` 透传自动跟上，不需要改一行。

**验收**

- `tests/test_lower_kernel.py` 新增 6 个用例，L2 对照用例从 50 增至 56：
  im2col 在 f32/f16 下与模板逐 token 相等；分块卷积**两种 dtype 打印出同一份源码**
  （钉住"这个内核没有按 dtype 参数化"，打印器里一旦泄漏存储类型就失败）；
  越界线程走 `continue` 而不是提前 `return`（屏障可达性）；注意力三段共享内存的偏移
  全部来自 `blockDim.x`，且 `attention_rand` 不出现在其它内核里。
- 本地全量 905 个测试 693 通过 / 212 跳过。
- CUDA 真机（RTX 2080 Ti）：`verify_cuda.py` 全部 PASS，全量 905 个测试 861 通过 /
  44 跳过——三份新内核（im2col、分块卷积、注意力）在真机上与参考实现对齐，
  切换 L2 没有改变任何数值。

**边界**

- 分块卷积与注意力固定 float：f16 走 im2col + hgemm 或 f32 attention。L2 描述不假装
  支持别的存储类型，`Tensor(type_name="float")` 把这件事写在声明里。
- 动态共享内存的**字节数**仍由后端计算（`shared_mem=`），L2 只描述"声明与切分"。
  预算检查（48KB / 上限拒绝）留在 backend，因为它是硬件约束而不是内核语义。
- 至此 9 份字符串模板全部迁完，L2 是唯一生产路径；剩下的 L2 工作是语义级（索引 /
  布局 / 分块），不是"再搬几份模板"。

---

### 5.50　归约族补齐：max / mean / argmax

**动机**：L2 迁移完成后，缺口已经不在内核表达力，而在"常用算子不够"。PyTorch 的
基本任务里归约族的出镜率很高——`max`（数值稳定与裁剪）、`mean`（损失与监控统计）、
`argmax`（分类精度）——而框架此前只有 `reduce_sum`。

**语义**

- `reduce_max` / `reduce_mean` 与 `reduce_sum` 同形：属性都是 `(axis, keepdims)`，
  类型推导共用 `_infer_reduce`，参考实现分别是 `np.max` / `np.mean`。
- `argmax` 只接受**单个轴**（numpy 的 argmax 也是），输出 `i32`。并列最大值取
  **第一次出现**的下标。它不可导：`Op.vjp=None`，接进损失会在 backward 时抛
  `NoGradientError`，而不是静默给零。

**反向（只用已有语义构图）**

- `mean`：`dx = broadcast(g / N)`，N 是归约元素数，由静态形状直接算出。
- `max`：`mask = 1 - step(m - x)`（`step` 是严格大于，差为 0 的位置才留下 1），
  `dx = g * mask / reduce_sum(mask)`。并列时**均分**，与 PyTorch 的约定一致。
  整条规则只用到 `sub` / `step` / `mul` / `div` / `reduce_sum` / `reshape` /
  `broadcast_to`，所以反向图在参考后端与 CUDA 上都能直接跑。

**内核**

- `_reduce_last(name, reduce_fn, mean=, init=, combine=)`：把 sum / max / mean 收成
  一份骨架。`combine="add"` 打印 `s += x`；`combine="max"` 打印
  `float v = ...; if (v > s) { s = v; }`；`mean` 在块归约后除以 `(float)cols`。
- `argmax_last()`：两段式——先 `block_reduce_max` 得到全局最大值，再让
  `best == m` 的线程用 `atomicMin` 竞争最小下标。为此新增 L2 节点 `AtomicMin`
  （`atomicMin(&buf[i], v);`），与 `AtomicAdd` 分开：前者是整数下标的最小值竞争
  （dtype 无关），后者是浮点累加（dtype 相关）。
- `out` 参数用 `Tensor(type_name="int")`：下标不是浮点激活值，不随输入 dtype 走。

**产物**

- `ir/ops.py`：三个算子；`_infer_reduce_sum` 改名为 `_infer_reduce` 供三者共用。
- `ir/grad_rules.py`：`reduce_mean` / `reduce_max` 的 VJP。
- `lower/kernel.py`：`AtomicMin`（节点总数 29：14 表达式 + 15 语句）。
- `lower/kernels.py`：`_reduce_last` 骨架 + `reduce_max_last` / `reduce_mean_last` /
  `argmax_last`。
- `cuda/codegen.py`：`REDUCE_MAX_SRC` / `REDUCE_MEAN_SRC` / `ARGMAX_SRC` 三份手写
  模板与三个入口；HIP 经 `__getattr__` 透传，不需要改一行。
- `backend.py`：`_reduce_last_dim` 收敛 sum / max / mean 的公共路径（多轴拒绝、
  keepdims 恢复、空输入）；`_op_argmax` 单独处理 i32 输出与空轴拒绝。

**验收**

- L2 对照新增 7 个用例（max / mean / argmax × f32 / f16，以及"`int*` 参数不随
  dtype 变化"），总数 56 → 63。
- 自动微分新增 4 个用例：max / mean 与中心差分一致；并列最大值梯度均分；argmax
  求导抛 `NoGradientError`。
- 本地全量 927 个测试 704 通过 / 223 跳过。
- CUDA 真机（RTX 2080 Ti / CUDA 12.9）：`verify_cuda.py` 全 PASS（conformance
  73 用例 68 executed / 5 refused / 0 failed，新增的 max / mean / argmax 六个用例
  err ≤ 4.5e-08），全量 927 个测试 883 通过 / 44 跳过。
- HIP 真机（gfx1100）：本轮环境不可用——WSL2 升级到 6.18 内核后 `/dev/dxg`
  的 `dxgkio_query_adapter_info` 返回 -22，`/dev/kfd` 未出现，GPU 直通在宿主侧
  就失败了（不是内核模板的问题）。HIP 侧的承接面仍由 5.40 的 conformance 给出，
  这三个新算子与 HIP 上已验证的其它归约共用同一份 `Kernel` 与 prelude。

**边界**

- CUDA 后端仍然只支持**最后一维**归约（与 `reduce_sum` 同一条边界），其它轴明确
  拒绝。参考后端支持多轴——这条差异由能力声明表达，不靠"悄悄只算一个轴"。
- `argmax` 的并列语义依赖"最小下标"，用 `atomicMin` 表达；初值 `-3.4e38f` 与
  maxpool 一致。空轴明确拒绝（numpy 也抛错）。
- `argmax` 没有反向：它输出下标，没有可定义的导数。

---

### 5.51　转置卷积：导入即支持，不新增算子

**动机**：分割、生成与上采样模型里 `nn.ConvTranspose2d` 很常见，而此前
`torch_export` 一遇到 `transposed=True` 就整图拒绝。

**结论**：IR 里**不需要**第二个卷积算子。转置卷积与"conv2d 对输入的梯度"是
同一条散射关系——

```
y[n, co, oh, ow] += x[n, ci, ih, iw] * w[ci, co, kh, kw]
oh = ih * sh - ph + kh * dh
```

`conv2d_input_grad` 已经精确表达它（5.34 连分组也做进去了），CUDA 与 HIP 的
内核都在生产路径上。再定义一个 `transposed_conv2d` 只会让"卷积到底以哪套
语义为准"重新变得含糊，所以这次只补前端。

**落点**

- `frontend/torch_export.py` 新增 `_conv_transpose2d`，接住
  `aten.conv_transpose2d.input` 与 `aten.convolution.default`（transposed=True）。
- weight 布局 `(C_in, C_out/groups, kH, kW)` 与 `conv2d_input_grad` 期望的
  `(O, C/groups, kH, kW)` 完全相同——转置卷积的"输出梯度通道"就是它的输入
  通道，不做转置，避免多一个 `transpose` 节点。
- `output_padding` 只撑大输出尺寸，多出的行/列没有输入贡献（恒为 0），
  因此它体现在 `input_shape` 上，不新增属性。后端按 `dy` 形状反推输入尺寸，
  这要求 `output_padding < stride`；更宽的情形显式拒绝，不做近似。
- bias 用 `(1, C_out, 1, 1)` 的 reshape 后广播 `add`（与推理态 BN 同一手法）。

**验收**

- conformance 新增 `conv_transpose2d_form`（74 用例）：stride=2 / padding=1 /
  output_padding=1（输出 10×10，输入 5×5），专盯"反推尺寸与 `input_shape`
  一致"这条不变量。CUDA 真机 err=2.4e-07。
- 前端新增 3 个用例：默认参数、stride+padding+output_padding（含非方形）、
  depthwise 分组；三者都与 torch eager 一致，非方形用例同时验证 10×10 的
  `input_shape` 推导。
- 本地全量 930 个测试 707 通过 / 223 跳过；CUDA 真机 930 个测试 883 通过 /
  47 跳过（多出的 3 个跳过就是这三条前端用例——真机没装 torch），
  `verify_cuda.py` 全 PASS（74 张图 69 executed / 5 refused / 0 failed）。

**边界**

- 只承接二维、静态 `H_in/W_in/kH/kW`（与前向 `conv2d` 同一条边界）。
- `output_padding >= stride` 会被拒绝：PyTorch 允许 `output_padding < max(stride,
  dilation)`，但 `conv2d_input_grad` 的内核按 `dy` 形状反推输入尺寸，跨过这条
  线就会整体错位，宁可在编译期说不。
- 反向：`conv2d_input_grad` 在 `ir/grad_rules.py` 里本来就没有二阶规则（与
  `conv2d_weight_grad` 一致），训练图对转置卷积求导只支持一阶。

### 5.52　语义级 L2：逐元素链融合与 4 通道向量化

**动机**：5.42–5.51 把全部内核迁到了 L2，但那一层仍是**语法级**——一个算子
一个内核。真实模型里 `x + b` → `relu` → `* scale` 这种相邻逐元素算子非常
密集，每个都物化一次中间张量，每次访存也只是 32 位。语法级 L2 对此无能为
力，因为它描述的是"一条语句"，不是"一次数据流"。

**结论**：L2 往上走一步，新增 `lower/elementwise.py`。`ElemInput` 描述链首
的每个张量（是否退化成标量），`ElemStep` 描述链里的一步值依赖，
`build_chain(steps, inputs, vec=1|4)` 把整条链编成**一个**内核：中间值不声明、
不物化，只有链首输入与链尾输出真正读写内存。算术完全复用 `kernels.py` 的
`BINARY_OP` / `UNARY_EXPR` 两张表——新加一个逐元素算子只需要写一次表达式，
融合与向量化自动获得。

**什么能融（静态计划，`_group_elementwise`）**。全部是图上的判据，不碰设备：

- 算子必须在一元/二元表达式表里，且参数个数与表一致（`silu` 在门控形态下
  是二元的，语义是 `silu(x) * gate`，不当成一元链的一步）；
- 相邻两节点之间那条边只被链内下一个节点使用，**且不是图输出**。链中间的值
  一旦是图输出就必须物化，融合会改变可观测行为。链尾是图输出没有这个问题：
  组结束后写回环境的就是它；
- 形状与 dtype 在 IR 上相同（符号维按名字比较）；大小为 1 的一侧可以当标量
  复用，其余广播形态不融；
- 长度 1 的组不合并：单算子仍走原来的专用内核，源码与数值行为逐字不变。

**运行期回退（`_try_chain`）**。静态计划是乐观的，运行期再查一次，任何一条
不满足就退回逐节点执行，结果仍然正确：

- dtype 只能是 f32 / f16，且所有外部输入同 dtype、C 连续；
- 形状必须能广播到一个共同输出形状，非标量输入必须与输出同形；
- 元素数为 0 不融；`vec=4` 还要求长度为 4 的倍数、输出与所有输入指针
  16 字节对齐。

**向量化**。`vec=4` 时每个线程负责 4 个**连续**元素：链首用 `af_ld4_f` 一次
读 128 位得到 `float4`，四条通道各算一遍逐字相同的标量表达式（只有 `.x/.y/
.z/.w` 不同），链尾用 `af_st4_f(p, i, x, y, z, w)` 一次写回。循环上界因此
是 `n / 4`、索引是 `i * 4`——这两点必须成对出现，少任何一个都会让
`float4` 落在非 16 字节边界上（真机上表现为 `cudaErrorMisalignedAddress`，
且是 sticky 错误，会污染整个进程后续所有 CUDA 调用）。向量化不改变算术：
每个元素的浮点运算顺序与标量路径完全一致，所以两条路径可以互相作对照。

这两个 helper 放在 `Kernel.helpers` 而不是方言 prelude：只有向量化的链才会
用到，塞进 prelude 会让每个逐元素内核（包括第三方方言的对照源码）都多出
两段没人调用的函数，直接破坏 5.44 的"只差函数名"断言。

**执行器的改动**。`runtime/executor.py` 新增两个可选钩子：

- `group_nodes(graph)`：把拓扑序切成执行组，默认一个节点一组；
- `invoke_group(group, args, attrs)`：执行一组、返回最后一个节点的输出。
  默认实现逐个 `invoke`，组内前序节点的输出直接用刚算出的结果补上——语义
  与"物化后重读"完全一致。

`run_device` 只把**组尾**的输出写回环境并做形状校验与 `cast`；组内中间值因此
从未进过环境。`args` 里组内前向依赖的位置是 `None` 占位，流式内核看到它
说明"这是链内值"，逐个执行时则由 `produced` 表补上。`release=True` 的回收
按组递减引用计数，中间值本来就不在环境里，减到 0 时 `pop` 到 `None` 也无害。

`InstrumentedBackend` 故意不覆盖 `group_nodes`：遥测要的是"逐节点派发耗时"，
融合会把这部分信息抹掉。融合只在纯 CUDA 后端生效，两者各取所需。

**f16 的语义**。不融合的图里每个算子都是一次真实的 f16 往返（写回再读进
`float`）。融合若一路在 `float` 里算，结果会与参考实现不同——差得不是一点。
所以 f16 链在**每个中间步**都插 `Narrow`（`af_st_h`）再 `Widen`（`af_ld_h`），
把舍入点原样保留；f32 的往返是恒等映射，不插这一层。`vec=4` 只对 f32 开放：
f16 的 16 位存取走 PTX 转换，攒成向量是另一套写法，需要时再单独做。

**踩过的坑**（都已固化进测试）：

- 向量路径最初让 `i` 直接当元素索引，长度 24、32 的图立刻 misaligned；
- `_try_chain` 的 `entries` 存的是 `(value_id, array)`，拼 `ElemInput` 时
  却写成了 `value.id`，第一次真机跑就炸在 `'int' object has no attribute 'id'`；
- 测试里同一张图前后调用两次 `feed()` 生成两份随机输入，融合链被测成"错"的
  ——问题在测试，不在实现。

**验收**：

- `tests/test_lower_kernel.py` 新增 10 个用例：标量路径每步物化、向量路径
  四条通道且中间值不落内存、`af_ld4_f` 只属于链内核、f16 每步往返、`vec=4`
  对 f16 抛错、未知算子 / 空链 / 未定义值 / 非法宽度抛错、f32 链在 CUDA 与
  HIP 上源码逐字相同。
- `tests/test_cuda_backend.py` 新增 7 个静态计划用例（成形、标量侧、图输出
  中间值、多消费者、非标量广播、cast 打断、门控 silu）+ 5 个 GPU 用例
  （数值对拍、只编译一个链内核且 `vec=4`、长度 15 退回标量通道、f16 对拍、
  中间值是图输出时仍与参考一致）。
- 本地 952 个测试 724 通过 / 228 跳过；CUDA 真机 952 个测试 937 通过 /
  15 跳过；`verify_cuda.py` 全 PASS（74 张图 69 executed / 5 refused /
  0 failed）。

**顺带修的前端问题**：新版 torch（2.14）对 eval 态的 `BatchNorm2d` 不再导出
`aten._native_batch_norm_legit_no_training`，而是保留 `aten.batch_norm.default`。
补了一条降级：`training=False` 时走同一条仿射展开（eps 在第 7 个参数），
`training=True` 明确拒绝——训练态要更新 running stats 且返回多值，IR 的
`batch_norm` 不承接这部分。

**边界**：

- 只融**同形**逐元素链，且只覆盖 `BINARY_OP` / `UNARY_EXPR` 里的算子；不涉及
  归约、广播（size-1 之外）、索引或布局变换。自动分块与索引抽象留给后续。
- `vec=4` 只在 f32、且运行期对齐/长度条件全满足时启用；条件不满足时静默
  退回 `vec=1`，再不行退回逐节点执行。三条路径的数值都逐元素一致。
- HIP 侧同样拿到这条链（同一份 `Kernel`，只换方言）；但 HIP 真机本轮未复跑。

### 5.53　符号形状表达式：让派生维留在 IR 里

**动机**：符号维此前只能"原样传递"。卷积/池化只要 stride≠1 或 kernel≠1，
空间维是符号就整张图拒绝——用户会问"为什么同一张图换个分辨率就不行"。
实测 CUDA 侧的内核键本来就不含具体形状（`("step", kind)`、`"conv2d_tiled"`
都是常量键，形状走运行期参数），所以"一图多形状"早已不需要新机制；真正
缺的是**形状的表达力**：`floor((H + 2p - k) / s) + 1` 写不出来。

**结论**：`Dim` 从 `int | str` 扩展为 `int | str | SymExpr`。`SymExpr` 是纯整数
表达式树（`add` / `sub` / `mul` / `floordiv` / `max` / `min`），构造时做常量折叠
与少量恒等式化简，卷积与池化的输出维直接推导成表达式。

**落点**：

- `ir/types.py`：`SymExpr` + `dim_static` / `dim_symbols` / `dim_resolve`；
  `sym_binary` 负责折叠（`+0`、`*1`、`/1`、常量吸收、`(x ± c1) ± c2` 合并）。
  化简不是装饰：same-padding 的卷积必须结构上化简回 `H`，否则后续算子之间
  的形状比较会凭空多出差异；
- `ir/ops.py`：`_broadcast_shape` 与 matmul 内维改成"静态对静态才严格比较"，
  `_conv_out_dim` / `_pool_out_dim` 在符号维上返回表达式；
- `runtime/executor.py`：`_bind` 对裸符号做绑定/冲突检查，对派生维用已绑定
  的符号求值再核对，不能反解时明确报"依赖未绑定的符号"；
- `ir/serialize.py`：表达式编码成 `{"expr": "floordiv", "args": [...]}`，
  读回时重建同一棵树（结构相等，round-trip 逐字段一致）；
- `reshape` 的目标形状明确拒绝表达式：参考执行器还没有"从目标形状反解符号"
  的能力，与其埋一个运行期崩溃，不如在构图期说清楚。

**边界**：

- 只支持整数算术，没有谓词与条件表达式；也不支持从"派生维 + 具体值"反解
  符号，因此派生维依赖的符号必须在更早的维或更早的输入里出现；
- 卷积/池化的 kernel、stride、padding、dilation 仍要求静态整数（它们决定
  内核结构与 shared memory 预算）；
- 没有 guard 与重编译策略：形状变了就是运行期参数变了，不需要重新编译。

**验收**：

- `tests/test_ir.py` 新增 7 个用例：常量折叠、恒等式、常量吸收、求值与符号
  集合、完全括号化的打印、带表达式的图 JSON round-trip、卷积/池化在符号
  空间维上的推导与多分辨率执行；
- `tests/test_cuda_backend.py` 新增 3 个 GPU 用例：符号空间维卷积多分辨率、
  卷积 + 池化串联、以及反向图（`dx`/`dw` 的形状同样是派生维）；
- 本地 962 个测试 731 通过 / 231 跳过；CUDA 真机 962 个测试 947 通过 /
  15 跳过；`verify_cuda.py` 全 PASS（69 executed / 5 refused / 0 failed）。

### 5.54　执行计划层：把"怎么铺线程"变成纯函数

**动机**：布局决策同时决定两件事——**内核源码的形状**（循环上界、向量宽度、
shared 声明大小）与 **launch 配置**（block、grid、动态 shared）。这两类的
数字此前分散在三个地方：`backends/cuda/backend.py` 的五个常量
（`CONV_TILE_W` / `_CONV_SHARED_LIMIT` / `_ATTENTION_BLOCK` /
`_CONV_TILED_MAX_K` / `_BLOCK_OPTIONS`）、两个纯算术函数
（`_pow2_block` / `_conv_oc_tile`）、以及 `lower/kernels.py` 里
`Shared("float", "smem", 1024)` 这样的硬编码。

问题不在重复，而在**约束没有执行者**：block 必须是 2 的幂、shared 不能超
48KB、`vec=4` 要求长度整除且指针 16 字节对齐——这些只写在注释里。违反的
代价是驱动报 `invalid configuration argument`（看不出是哪一维错了），或者
是一次非法访存。它们本来都是可以在主机侧判定的纯算术，却因为长在后端里
而只能"跑起来试试"，HIP 继承 CUDA 时也各自复制了一份。

**结论**：新增 `lower/plan.py`，把"给定规模，该怎么铺线程"定为纯函数，
不 import 任何后端、不碰设备。后端退化成两件事：拿计划、launch。

**落点**：

- 数据结构：`Launch(block, grid, vec, shared_bytes)` 在构造时校验全部约束
  （2 的幂、grid ≥ 1、vec ∈ {1,4}、shared ≤ 预算）；`ConvTilePlan` 描述
  分块卷积的 `tile_w / block_y / toc / roc / cb / oc_tiles / shared_bytes`；
- 纯函数：`pow2_block` / `check_block` / `can_vectorize` / `grid_for` /
  `plan_elementwise` / `plan_reduce` / `attention_block` / `conv_oc_tile` /
  `plan_conv_tiled` / `conv_tiled_preferred`；
- `backend.py`：删掉 `_pow2_block` 与 `_conv_oc_tile`、五个常量与本地
  `_cdiv`；20 处 `max(1, min(self.max_blocks, cdiv(...)))` 收敛成
  `grid_for(...)`；`_pick_conv_impl`（返回"toc 或 0"）改成
  `_use_tiled_conv`（返回 bool，把"0 表示另一条路径"这个隐式约定去掉）；
  `_conv2d_tiled` 只负责开缓冲与 launch，线程铺法全部来自计划；
- `lower/__init__.py` 暴露 `plan` 子模块；`tests/test_cuda_backend.py` 的
  `TestConvTilePlan` 改从计划层取函数（原来测的是后端私有函数）。

**一个行为变化**：向量链的 grid 从 `cdiv(n, block)` 改成
`cdiv(n, block * vec)`。此前每个线程算 4 个元素却按元素数派线程，后 3/4 的
线程只会走进"越界退出"；现在块数少 4 倍。内核本身没改，网格步进保证了
尾部不满一个 block 时也不会漏元素——真机 1003 个测试全绿，conformance
的数字与 5.53 完全一致（69 / 5 / 0），说明这是一次纯配置收敛。

**边界**：

- 这一刀只搬"决策在哪、怎么校验"，没有新内核、没有新的分块形状：
  `kernels.py` 里 `Shared(..., 1024)` 仍是硬编码（归约暂存区大小与 block
  的联动还没交给计划），`CONV_TILE_W` 也还不能调优；
- 中层 pass 层（DCE / CSE / fusion / 折叠）仍以函数式 pass 的形式存在，
  没有成为"独立一层"；本轮解决的是它下面那层（L2 的布局决策）；
- 计划层不认识算子语义与图：它不知道自己在给谁铺线程，因此也不做
  "这个算子该不该融合"这类判断。

**验收**：

- 新增 `tests/test_plan.py`（41 个用例，全部离线）：block 上取整与上限、
  `Launch` 的四类非法参数、向量化的三个前提、grid 的上限/取整/vec 收缩、
  `plan_elementwise` 的保守默认、归约与注意力的 block 边界、卷积 tile 的
  shared 预算边界（含 CB 由预算而非上限决定的那一档）、`conv_tiled_preferred`
  的 K 与工作量阈值；
- 本地 1003 个测试 772 通过 / 231 跳过；CUDA 真机 1003 个测试 988 通过 /
  15 跳过；`verify_cuda.py` 全 PASS（69 executed / 5 refused / 0 failed）。

---

### 5.55　共享网格步进骨架：一份循环形状，两个使用方

**动机**：5.54 把"怎么铺线程"的决策收进了 `lower/plan.py`，但"块内第一行
怎么写"仍是复制粘贴——`kernels._grid_stride_body` 与 `elementwise._grid_stride`
是同一段代码的两份副本（差别只在参数顺序），`ew_binary` 里还手写了第三份；
`Shared("float", "smem", 1024)` 在六个内核里重复，而 `plan.REDUCE_SCRATCH`
早就把这个数字定义好了，却没有人用。

复制本身不致命，问题是**约束没有落点**：stride 必须取自 `gridDim*blockDim`
（块数被 `max_blocks` 截断时，多出来的元素只能靠步进补），索引必须 64 位。
任何一份副本写错，都只在超大张量上表现为"少算了尾部一段"，而测试形状通常
看不出这种错。

**结论**：把一维网格步进提成一个公开骨架 `kernels.grid_stride(limit, *body)`，
两份实现合一、`ew_binary` 改用它；归约暂存区改由 `plan.REDUCE_SCRATCH` 供给。

**落点**：

- `lower/kernels.py`：`_grid_stride_body` 改成 `grid_stride`，第一个参数是
  循环上界（标量路径 `Var("n")`，向量路径 `n / 4`）；docstring 写清三个量
  各自为什么不能省；
- `lower/elementwise.py`：删掉自己的 `_grid_stride`，改从 `kernels` 取——这个
  依赖方向本来就存在（它已经在 import `BINARY_OP` / `UNARY_EXPR`）；
- `ew_binary` 的三行手写改用骨架；六处 `Shared("float", "smem", 1024)` 换成
  `REDUCE_SCRATCH`；
- `lower/__init__.py` 导出 `grid_stride`。

**零行为变化**：骨架生成的是同一棵 AST，所有内核的源码逐 token 不变，因此
cstyle 的对照测试与既有 1003 个测试同时是这次重构的验收。

**边界**：

- 只统一了**一维**网格步进。卷积 / 池化 / 注意力的多维块内索引（`oy*wo+ox`
  这类）仍逐内核手写：它们随算子变（窗口滑动、通道分组、tile 偏移），不是
  同一个形状；
- `_flat_loop` 没有并进来：它打印成 `for (long long idx = ...)`（init 内联、
  stride 带 `const`），与本骨架的 `long long i = ...; for (; ...)` 是两种源码
  形状，合并就会改变输出；
- `CONV_TILE_W` 等分块参数仍未接入调优面。

**验收**：本地 1003 个测试 772 通过 / 231 跳过；CUDA 真机 1003 个测试 988
通过 / 15 跳过；`verify_cuda.py` 全 PASS。

---

### 5.56　分块卷积接入调优面：tile_w 作为旋钮

**动机**：5.54 把 conv2d 的分块参数收进 `plan_conv_tiled`，5.55 统一了循环
骨架，但 `tile_w` 仍是常量 32——`tuning_space` 只认识逐元素的 block，卷积的
分块参数一个都不接受调优。5.16 的"K ≤ 32 时分块占优"这个结论本身是在
`tile_w=32` 下测出来的，换一个 W 或 OC 排布，32 未必仍是最好。

动手前先确认了一件容易被误判的事：**缓存键不需要改**。`conv2d_tiled` 的
`toc` / `roc` / `cb` 与 tile 相关的量都是**运行期标量参数**（`np.int32` 传进去），
内核源码读的是 `blockDim.x` 与这些参数，不是编译期常量。所以同一份源码适配
所有 tile 组合，`_kernel("conv2d_tiled", ...)` 的键不必纳入 plan——若把
toc/roc/cb 写成源码常量，这里就必须改键，否则不同 tile 会撞上同一份旧源码。

**结论**：把 `tile_w` 接进调优面，作为 conv2d 唯一的旋钮。

**落点**：

- `plan.plan_conv_tiled(..., tile_w=None)`：``None`` 用 `CONV_TILE_W`；
  显式值按 `wo` 截断（多出来的线程只会走边界慢路径）；`tile_w * block_y`
  超过 1024 在构造期拒绝——否则调优搜索会在某个候选上撞见驱动的
  `invalid configuration argument`，而那本来是可以提前判定的纯算术；
- `plan.CONV_TILE_OPTIONS = (16, 32, 64)`：半行 / 一行 / 两行的合并访存跨度，
  每个候选在典型形状下都能规划成功（有测试守着）；
- `CudaBackend`：`_conv_tile` 与 `_blocks` 分开存——一个是 block 的 x 维，
  一个是线程总数，校验规则不同，混在一张表里会让 `set_tuning` 需要按算子名分支；
- `tuning_space("conv2d")`：单参数空间，首候选是**当前生效值**，未调优时是
  `None`（"自动"）。默认的 `roc` / `cb` / `tile_w` 都随形状变，而调优空间
  不知道形状，钉死一个常数就会凭空造出一个"默认值"；
- `set_tuning("conv2d", {"tile_w": ...})`：只校验值本身（正整数或 `None`），
  形状相关的上限交给计划层在 launch 前判——空间不该假装自己是编译器；
- `tunable_ops()` 的语义从"接受 block 覆盖的算子"变成"有旋钮的算子"，
  `describe_operator("conv2d")` 因此能返回 `{"tile_w": [null, 16, 32, 64]}`；
- `tuning_config()` 从 `{op: block}` 改成 `{op: {参数: 值}}`：两类旋钮的参数名
  不同，同一种形状才不需要调用方按算子名分支。

**实测**（RTX 2080 Ti，输入 1×2×64×512、32 个 3×3 输出通道、K=18、取 40 次
中位数）：

| tile_w | 中位耗时 |
|---|---|
| None（默认 32） | 1.264 ms |
| 16 | 1.243 ms |
| 32 | 1.239 ms |
| 64 | 1.257 ms |

四个候选落在 2% 内：这个形状下默认值已经不差，旋钮的价值在别的形状（W 窄、
OC tile 大）而不是"32 永远错"。这恰好是把它做成旋钮、而不是改默认值的理由。

`benchmarks/tune.py` 的固定图集同时加了 `conv2d_tiled_1m`（1×2×64×512 输入、
32 个 3×3 输出通道），于是自动调优第一次真正覆盖到卷积：真机跑一轮
（`rounds=1`）选中 `tile_w=64`、1.289 ms，应用后与默认逐位一致、搜索结束
配置已清空。单轮选中 64 而 40 次中位数显示 32 略快，正好说明一次测量不足以
定论——这也是 CLI 默认 `rounds=2` 的原因。

**边界**：

- 只开放 `tile_w`。`roc` / `cb` 的候选来自 5.16 的实测上限，且与 shared
  预算耦合（改 toc 会连带改 cb），留给"分块参数整体搜索"；
- 调优空间**不知道形状**：形状感知靠配置库的签名命中（5.37 的键结构），
  `conv2d_tiled_1m` 按 (输入形状, 权重形状) 记录；
- `tile_w` 对 im2col 路径无效：覆盖只在分块核上生效，`conv_impl="im2col"`
  时安静地不起作用。

**验收**：

- 新增 7 个用例：`test_plan.py` 5 个（覆盖生效、按 W 截断、超 block 上限被拒、
  非正整数被拒、每个公开候选都能规划成功）+ `test_cuda_backend.py` 2 个
  （三个 tile_w 下逐位等于默认结果、空间形态与非法值被拒）；
- `benchmarks/tune.py` 的 `TUNE_CASES` 增加 `conv2d_tiled_1m`，真机上完成
  "搜索 → 应用 → 逐位对拍"闭环（1 个命中、0 个问题）；
- 本地 1010 个测试 777 通过 / 233 跳过；CUDA 真机 1010 个测试 995 通过 /
  15 跳过；`verify_cuda.py` 全 PASS。

---

### 5.57　多维块内索引：合并一处，其余保持手写

**动机**：5.55 统一了一维网格步进，剩下的一层是"展平的 ``idx`` →
``(n, c, y, x)``"与"窗口边界 clamp"。池化族六个内核加卷积梯度四个内核看起来
都在做同一件事，值得先确认重复的**真实范围**，再决定动不动手。

**调研结论**（逐字比对，不是印象）：

- 真正逐字相同、可以直接替换的只有一处：``conv2d_input_grad`` 开头 18 行的
  四维拆分，与 ``_split_output_index("iw", "ih", "W", "H")`` 连 ``int`` /
  ``long long`` 的类型和括号位置都一致；
- **名字相同但语义不同**：``conv2d_weight_grad`` 的末轴是"输出通道 ``oi``"，
  用纯除法、``int`` 类型；骨架的末两行是 ``% C`` 加 ``long long`` 除法。要合并
  就得给骨架加 ``last_cast`` / ``last_type`` / ``outer_mod`` 三个开关——为了
  消掉 20 行重复而引入三个分支，比重复更难读；
- **语句顺序不同**：自适应池化是 ``split → bounds → shared → image_slice``，
  非自适应是 ``split → image_slice → bounds``；
- **基址 cast 位置不同**：``_image_slice`` 是 ``(long long)H * W``，im2col 是
  ``H * W``（``H`` 未 cast），``conv2d_tiled`` 干脆用 block 轴分解而不是扁平
  ``idx``。

**结论**：只替换这一处，其余明确保持手写，并在 ``_split_output_index`` 的
docstring 里把"变量名是约定"与"权重梯度为什么不能用它"写清楚——避免下一个人
把同一段调研再走一遍。

**落点**：``conv2d_input_grad`` 的 body 从 ``_flat_loop((18 行索引, 其余…))``
变成 ``_flat_loop(_split_output_index(...) + (其余…))``。

**零行为变化**：生成的是同一棵 AST，源码逐 token 不变；本地 1010 个测试里包含
L2 与手写模板的对照，真机复跑结果一致。

**边界**：

- 没有新增"布局对象"或"多维索引节点"。L2 的索引已经是结构化表达式树，
  ``Index`` / ``Bin`` 够用；再加一层布局抽象，收益要等出现"同一算子多种布局"
  （如 NHWC）才成立，现在没有这个需求；
- 窗口边界（floor/ceil、clamp、``continue`` 跳过）明确**不**统一：三种规则
  语义不同（重叠窗口原子累加、越界跳过、整除门），合并只能靠模式开关；
- 这也标出了 5.54～5.57 这一串的收口位置：布局决策已上提（plan）、单维循环
  已统一（grid_stride）、多维索引的逐内核写法是**核实后的选择**，不是遗漏。

**验收**：本地 1010 个测试 777 通过 / 233 跳过；CUDA 真机 1010 个测试 995
通过 / 15 跳过；`verify_cuda.py` 全 PASS。

---

### 5.58　HIP 调优面：补一行字段，加一道静态对账

**动机**：文档里一直写着"HIP 侧还没有接入调优与遥测"。核对后发现这句话只对
一半：`tuning_space` / `set_tuning` / `clear_tuning` / `tuning_config` 全部
继承自 `CudaBackend`，HIP 不需要另写一份。真正缺的是一行**实例字段**。

**发现的缺陷**：`HipBackend.__init__` 是手抄的——它不能调 `super().__init__()`，
因为 CUDA 那份要 `import cupy` 并 `cp.ones(1)` 建常量，而 HIP 的容器是 ctypes
直连 hip 的 `DeviceArray`。手抄的代价是漂移：5.56 给 conv2d 加了
`self._conv_tile`，HIP 侧漏了。

后果比"调优用不了"更重：`_conv2d_tiled` **无条件**读
`self._conv_tile` 传给 `plan_conv_tiled`，所以 HIP 上凡是 auto 选中分块核的
卷积（K ≤ 32 且输出量 ≥ 4096，典型 3x3 卷积）都会在 launch 前抛
`AttributeError`，与调优无关。

**为什么一直没暴露**：`TestHipConformance` 整组 `skipUnless(HIP_AVAILABLE)`，
本地、CI、CUDA 真机三处都走不到；而分块核的选型是运行期启发式，静态读过
`_conv2d_tiled` 也未必会注意到继承来的字段。

**落点**：

- `hip/backend.py` 的 `__init__` 补 `self._conv_tile: Optional[int] = None`，
  与 CUDA 那份逐字对齐；
- 新增 `tests/test_hip_backend.py::TestHipInitParity`（唯一一个不需要 ROCm 的
  HIP 用例）：用 `ast` 解析两个 `__init__` 的函数体，收集所有
  `self.<name>` 赋值目标，`CudaBackend` 有而 `HipBackend` 没有的必须为空集。

**为什么用静态对账，而不是"上机跑一遍"**：没有 ROCm 硬件。静态检查在本地与
CI 都能跑，且失败信息直接是字段名（`_conv_tile`），不是一条看不出因果的
`AttributeError`。这也把 5.40 那句"模板只有一份，不存在 CUDA 修了、HIP 忘了"
从主张变成了可执行的断言。

**为什么不动构造签名**：把容器作为参数注入 `CudaBackend.__init__` 就能让 HIP
调 `super()`，但那是把"后端自持容器"的边界打薄，还要改三家工厂的装配代码；
补一行 + 一条对账测试是同等可靠、改动面小一个数量级的做法。

**边界**：

- 遥测（`runtime/telemetry`）在 HIP 侧当时没有接上；5.59 核实后发现代码层
  本来就与后端无关，真正缺的是基准 CLI 的后端选项，见那一节；
- `benchmarks/tune.py` 的 `--backend` 当时仍然只有 `cuda`。当时的判断是"放开
  到 HIP 只是把 choices 加一项，但本机与远端都没有 AMD 设备，写了也无法验证，
  因此不做"；5.59 改了主意——理由不是"现在能验证了"，而是"写死的名单本身
  就是缺陷"，见那一节；
- `roc` / `cb` 仍然不开放为旋钮：5.16 的实测已把上限（4 / 8）定为最优，把它们
  放进搜索空间只会让候选数乘以 32。

**验收**：本地 1011 个测试 778 通过 / 233 跳过；CUDA 真机 1011 个测试 996
通过 / 15 跳过；`verify_cuda.py` 全 PASS。

---

### 5.59　基准 CLI 的后端选择：从写死两个名字到查注册表

**动机**：5.58 修完 HIP 的调优面后，回头核对"HIP 侧遥测未接入"这句还剩多少。
核实结果是**遥测本身不缺**：``InstrumentedBackend`` 对 ``device_info`` /
``memory_stats`` 都按 ``hasattr`` 探测，``HipBackend`` 继承同一套
``execute`` / ``invoke``；连 ``_sync`` 读的 ``cp.cuda.get_current_stream()``
HIP 侧也有同形的 ``_CudaNamespace``（5.40 就做了）。缺口在工具层：三个基准
CLI 的 ``--backend`` 都写死成 ``("ref", "cuda")``，在 AMD 机器上没有任何一条
命令能产出遥测记录。

**落点**：

- ``benchmarks/run.py`` 新增 ``backend_names()``：``ref`` 加上
  ``registered_backends()``（去掉 ``reference``，避免同一后端出现两次）；
- 新增 ``open_backend(name, **kwargs)``：走 ``get_backend(name,
  fallback=False)``，装配失败转成 ``SystemExit``，命令行只看到一行原因；
- ``trend.py`` / ``tune.py`` 的 ``--backend`` 改用 ``backend_names()``，构造
  统一走 ``open_backend``，删掉各自的 ``_cuda()`` 与硬编码 ``CudaBackend``；
- ``conv_impl`` 只对 ``cuda`` / ``hip`` 透传——它是继承来的实现选择，硬件包
  没有这个概念。

**为什么不直接给 choices 加一个 ``"hip"``**：那仍然是一份写死的名单。
5.43/5.44 的硬件包装进来之后同样不会出现在选项里，于是每来一个后端就要改
一次 CLI。查注册表把"有哪些后端"交回给唯一的真相源。

**为什么要显式写 ``fallback=False``**：``get_backend`` 在**明确给出名字**时
本来就不走回退路径（回退只发生在 ``auto`` 分支），所以这一项今天是等价的。
写出来是为了把"基准工具绝不允许悄悄换后端"变成调用点的约束，而不是依赖
另一个模块当前的分支结构。

**为什么 ``ref`` 不交给工厂**：``run`` 用 ``ReferenceBackend``（带 ``name``
的包装器）、``trend`` 用 ``ReferenceExecutor``（执行器本身）。统一成
``get_backend("reference")`` 会改变 ``trend`` 拿到的对象，进而改变遥测里
``backend`` 字段的取值。既有报告格式不该因为一次重构而变。

**验收**：本地 1015 个测试 782 通过 / 233 跳过；CUDA 真机 1015 个测试 1000
通过 / 15 跳过；`verify_cuda.py` 全 PASS。命令行侧：

- 本地 ``run --backend ref --rounds 1`` 8 个用例全跑完；
  ``trend --backend ref --case matmul_256`` 输出趋势表；
  ``run --help`` 的选项是 ``{ref,cuda,hip}``；
  ``tune --backend hip`` 输出一行 ``--backend hip 在这台机器上不可用：…``；
- 真机 ``trend --backend cuda --case matmul_256 --rounds 2`` 产出两条记录与
  吞吐，``tune --backend hip`` 同样是干净的一行拒绝。

**边界**：这仍然只是"工具链不再排除 HIP"，不是"HIP 上跑出了记录"——没有
AMD 机器，5.58 与 5.59 合起来只把这条路修到**可执行**。5.60 在本机 WSL 的
ROCm 上把最后一步走完了（顺带揪出一个 `_Stream.synchronize` 缺失），
这里保留原文以记录当时的判断边界。

---

### 5.60　在 WSL 的 ROCm 上实跑：一个环境前提与一个真缺陷

**动机**：5.58 / 5.59 的边界都写成"缺一台 AMD 机器"。本机的 WSL2 里装了
ROCm 7.2 并用 DXG 直通，`rocminfo` 报 `AMD Radeon RX 7650 GRE`
（`gfx1100` / 实际 `gfx1102`）。这条路能走，于是把两份边界一次闭环。

**环境前提（不是代码问题，但必须写下来）**：不设
`HSA_ENABLE_DXG_DETECTION=1` 时，`hipInit(0)` 返回 **100**
（`hipErrorNoDevice`），而 `rocminfo` 明明已经枚举到 GPU。同一个进程里
`hipGetDeviceCount` 的 native 程序在设置该变量后立刻正常。也就是说 WSL 的
DXG 路径要求 HSA 层显式开启 dxg 探测；缺了它，设备枚举与 hip 初始化会给出
互相矛盾的答案。跑 HIP 的完整前置是：

```bash
export HSA_ENABLE_DXG_DETECTION=1
export LD_LIBRARY_PATH=/opt/rocm/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

**找到的真缺陷**：`_Stream` 只承载 `ptr`，没有 `synchronize()`。而
`benchmarks/run.py::_sync` 正是靠 `cp.cuda.get_current_stream().synchronize()`
把异步执行切断。后果是 HIP 上 `tune --backend hip` 的 **9 个用例全部**
`AllRejectedError`，每个候选的失败原因都是
`'_Stream' object has no attribute 'synchronize'`——搜索空间里没有任何一个
配置"可行"，所以连一条配置都搜不出来。

**为什么 `trend` 没暴露它**：`trend` 走 `TelemetryRecorder`，只包一层
`perf_counter`，不调 `_sync`；而 HIP 的 `execute` 在返回 host 数据前已经
同步过，wall 时间仍然可信。两条测量路径对同步的依赖程度不同，所以同一张
卡上"趋势能跑、调优全灭"。

**落点**：

- `_Runtime.stream_synchronize(ptr)`：薄封装 `hipStreamSynchronize`，
  `ptr == 0` 是默认流；
- `_Stream.synchronize()` 转调它，与 `cupy.cuda.Stream.synchronize` 同名同义；
- `TestHipStreamContract`：不建上下文、不碰设备，只断言替身满足
  `_sync` 的两项用法（`ptr == 0`、`synchronize` 可调用）。这类"少一个方法"
  的缺陷不会在导入或构造时暴露，只在第一次计时时炸，结构断言比上机便宜得多。

**实跑结果**（RX 7650 GRE / ROCm 7.2 / WSL2 DXG，Python 3.10）：

- `tests.test_hip_backend`：13 个用例全过（含 conformance 组）；
- 全量：**1016 个测试 760 通过 / 256 跳过**；
- `trend --backend hip --case matmul_256 --case conv2d_3x3_pad1 --rounds 2`：
  matmul 中位数 0.88 ms、conv2d 3x3 中位数 1.93 ms，吞吐 74.4M / 68.5M 元素每秒；
- `tune --backend hip --rounds 1 --save` 写出 9 条；再 `--apply` 命中 9 条，
  **全部与默认结果逐位一致**；
- conv2d 的 `tile_w` 在两次独立搜索里分别选中 64 与 32（1.05x / 1.06x），
  与 CUDA 上"一次测量不足以定论、默认 rounds=2"的结论一致；
- `add_broadcast` 在 apply 阶段反而慢了（0.80x）。`rounds=1` 下这是正常的
  噪声，不当成缺陷修——真要在小张量上定论，得提高轮次。

**边界**：WSL 的 DXG 直通不等同于原生 ROCm——没有 `/dev/kfd`，
`rocm-smi` 报"驱动未初始化"，这里的数字只代表这条路径，不能直接与裸机
ROCm 的公开数据对比；`hipGraph` 捕获仍未实现（HIP 后端明确拒绝，5.61 补齐）；
`benchmarks/conv_compare.py` 仍硬编码 CUDA 并自带一份
`get_current_stream().synchronize()`，在 HIP 上会撞同一个坑，但它本就是
CUDA 侧的对照实验，不改。

---

### 5.61　HIP 的 hipGraph 捕获：补上最后一块功能差距

**动机**：5.60 的边界里留着"`hipGraph` 捕获仍未实现"。在 CUDA 上这是运行期
的核心能力——`capture` 一次、`run` 多次，CPU 侧只剩一次下发；HIP 缺这块，
两个后端在"能做什么"上就仍然不对等。

**为什么不能照抄**：三处硬冲突，每一处都会在真机上直接报错，而不是慢一点。

1. `cuda/memory.py` 的 `Arena` 返回 cupy 的 `MemoryPointer`，HIP 的容器是
   ctypes 直连的 `DeviceArray`，没有 cupy 可依附；
2. HIP 侧没有"分配器回调"这层抽象：`runtime.empty` 直接调 `hipMalloc`，
   要让 arena 接管，必须自己开一个钩子；
3. 流捕获期间驱动禁止 `hipMalloc`、禁止同步 `hipMemcpy` / `hipMemset`、
   禁止任何 D2H 回读。"预热一轮再捕获"的写法在这里还需要异步版本，
   否则捕获会在第一个中间张量上失败。

**落点**：

- **流**：当前流从 `_Runtime` 移到模块级栈（`_STREAM_STACK`）。`_Stream`
  补上 `__enter__` / `__exit__`，内核 launch、`memcpy`、`memset` 一律从栈
  取流——否则内核落在默认流、捕获发生在非默认流，图会是空的。放在模块级
  还有一个副作用：`cuda.get_current_stream()` 不再需要初始化运行时，5.60
  那条离线结构测试继续成立。
- **拷贝**：`memcpy` / `memset` 在捕获期自动切到 `hipMemcpyAsync` /
  `hipMemsetAsync`，把拷贝记录成图节点；非捕获期保持同步语义不变。
- **分配**：`_Runtime` 增加 `set_allocator` / `alloc`，`empty` 从"直接
  `hipMalloc`"改成"问当前分配器要持有者"。默认分配器仍直连驱动，行为与
  从前一致；捕获期换成 arena。捕获期间若有遗漏的 `malloc`，直接抛出
  "预热轮的 arena 容量不足"——比驱动返回的
  `hipErrorStreamCaptureUnsupported` 更能指向原因。
- **arena**：新增 `hip/memory.py`。算法与 CUDA 版逐条相同（best-fit、释放
  即合并相邻块、块大小不随历史翻倍），只把返回类型换成不带 `__del__` 的
  `_Slice`：这个地址属于整块 chunk，单独回收会破坏账本，回收点只由
  `Executor.dispose` 显式给出。
- **计划**：`HipBackend.capture` 与 `HipGraphPlan`。捕获前先分配静态输入
  缓冲（走默认分配器，不属于 arena），预热一轮后 `reset_stats`，再捕获、
  `hipGraphInstantiate`；`replay` 更新可变输入后 `hipGraphLaunch` 一次。

**拒绝的边界**：`reuse_memory=False` 在 HIP 上明确拒绝。HIP 没有默认内存
池，而捕获期不允许 `hipMalloc`，这条路径没有实现可谈——不假装成功，也不
悄悄退回 arena。

**实跑结果**（RX 7650 GRE / ROCm 7.2 / WSL2 DXG）：

- `tests.test_hip_backend`：21 个用例全过（新增 5 个捕获用例、3 个离线结构
  断言）；
- 捕获路径含 hipBLAS：matmul 的 `hipblasSgemm` 被流捕获完整记录，
  `hipGraphInstantiate` 与多次 replay 均无错误。这条先验地不确定
  （hipBLAS 内部若有 workspace 分配就会打断捕获），实测没有；
- replay 结果与参考实现 `rtol=1e-4` 对齐；新输入生效、形状变化被拒、
  5 次重复 replay 逐位一致；
- 全量：**1024 个测试 768 通过 / 256 跳过**。

**边界**：捕获期需要的内存全部来自预热轮申请的那几块 chunk；图更大或
形状不同就要重新捕获，这与 CUDA 侧的约束相同。`_Slice` 不参与
`DeviceArray` 的生命周期管理，所以"忘了 dispose"只会浪费 arena 内的空间，
不会踩到别人的内存。

---

### 5.62　捕获与逐节点两条口径的对照：顺带修掉 HIP 捕获期扩容的真缺陷

**动机**：5.61 把 hipGraph 补齐后，两个后端在"能做什么"上已经对等，但还没有
任何入口能回答"捕获到底快多少"。`benchmarks/run.py` 只有逐节点一条口径，而
一张基线 JSON 只能对应一条执行路径——把捕获塞进去，"同一个用例"就会有两种
耗时，回归判定失去含义。

**落点**：新增 `benchmarks/capture_compare.py`，在 `cases.py` 的同一套 8 个用例
上跑两条路径：

- 每个用例先对拍数值（捕获 replay 与逐节点执行，`rtol/atol=1e-4`），不一致就
  不计时——没有数值一致，倍数没有意义；
- 计时区间都包含同步，测的是"算完"而不是"下发完"；
- 输出每例的耗时倍数，以及捕获计划的 `peak_bytes` / `total_bytes`；
- `--backend` 复用 `benchmarks.run` 的注册表工厂（明确指定却装不起来时一行
  拒绝，不退回 CPU）；捕获撞上能力 / 形状边界时只记一行原因，不中断整张对照表。

**顺带发现的真缺陷**：HIP 侧 8 个用例里有 4 个（`matmul_1024`、两个 conv2d、
`maxpool2x2`）捕获直接失败，报"捕获期不能申请新显存"。根因不在内核，在账本：

- 图输出属于执行器的 `keep`，**不会**被 `dispose`，所以预热轮结束后它们仍记在
  arena 的 `_live` 上；
- 捕获轮跑的是同一张图、同样的申请顺序，但空闲表是空的，只能向驱动要新块，
  而流捕获期间 `hipMalloc` 被拒绝；
- CUDA 侧不受影响——`cudaMalloc` 在捕获期可用，同一份用例 8/8 通过。这正是
  两个驱动的事实差异，不是算法差异。

**修复**：`hip/memory.py::Arena.recycle()`。捕获前把预热轮占着的地址整体收回
（清 `_live`、每个 chunk 全量记入空闲表、bump 视为切满），`HipBackend.capture`
在 `reset_stats()` 之前调用。收回后 best-fit 会给出与预热轮完全相同的布局，
捕获轮不再触发任何驱动分配。配套两条用例：离线账本断言（回收后第二轮申请
不再扩容）与真机用例（512 KiB 输出占满首块仍能捕获并数值对齐）。

**实跑数据**（每例 2 轮取最好；两机数字不可横向比较，只看同一后端内的倍数）：

| 用例 | CUDA 逐节点 | CUDA 捕获 | 倍数 | HIP 逐节点 | HIP 捕获 | 倍数 |
|---|---|---|---|---|---|---|
| `matmul_256` | 0.3010 | 0.2586 | 1.16x | 0.7782 | 0.9597 | 0.81x |
| `matmul_1024` | 1.7728 | 1.7941 | 0.99x | 8.1086 | 7.3156 | 1.11x |
| `matmul_batched` | 0.4848 | 0.4835 | 1.00x | 1.0599 | 1.3282 | 0.80x |
| `conv2d_3x3_pad1` | 1.1404 | 0.3408 | 3.35x | 1.9983 | 1.4139 | 1.41x |
| `conv2d_1x1` | 1.0943 | 0.3532 | 3.10x | 1.6379 | 1.3628 | 1.20x |
| `maxpool2x2` | 2.9903 | 2.9410 | 1.02x | 7.0006 | 6.2256 | 1.12x |
| `transformer_block` | 2.5851 | 0.6675 | 3.87x | 5.5207 | 4.0402 | 1.37x |
| `cnn_lenet` | 2.6779 | 0.3664 | 7.31x | 4.4703 | 2.1128 | 2.12x |

**读法**：

- 收益 ≈ 省下的 CPU 派发 ÷ 整图 GPU 时间。节点多、单节点轻的图最划算
  （`cnn_lenet` 7.31x、`transformer_block` 3.87x），计算主导的大 GEMM 基本持平
  （0.99x / 1.11x）——捕获不减少算力，只减少下发，这是它应有的样子。
- `maxpool2x2` 在 CUDA 上只有 1.02x：单内核、纯访存，派发本来就不是瓶颈。
- HIP 上 `matmul_256` 与 `matmul_batched` 反而更慢（0.81x / 0.80x）：在 WSL2
  DXG 这条路径上，`hipGraphLaunch` 的固定开销高于省下的几次 Python 派发。
  这是"这条路径"的结论，不能推广到裸机 ROCm。
- 显存列与 CUDA 侧统计口径一致（conv 两例 1 MB / 1 MB）；HIP 的 `total_bytes`
  不再翻倍，正是 `recycle()` 的效果——预热轮用过的块被捕获轮原样复用。

**边界**：

- 只有 f32 用例；f16 的捕获路径未纳入对照。
- HIP 数字来自 WSL2 DXG 直通（无 `/dev/kfd`，`rocm-smi` 报驱动未初始化），
  与裸机 ROCm 的公开数据不可直接对比。
- `benchmarks/conv_compare.py` 仍硬编码 CUDA 并自带一份同步调用，维持原样：
  它是卷积实现的 A/B，不是跨后端工具。
- 捕获只对应一组固定形状；换形状要重新捕获，倍数随之变化。

**验证**：本地 **1026 个测试 787 通过 / 239 跳过**；CUDA 真机 **1005 通过 /
21 跳过**，`verify_cuda.py` 全 PASS；WSL HIP 真机 **770 通过 / 256 跳过**，
8 个用例的捕获与逐节点两条路径全部跑通并数值对齐。

### 5.63　跨后端调优对照：同一个旋钮，两家厂商选出的值一样吗

**动机**：5.37 交付了参数空间、网格搜索与配置库，5.56 把分块卷积的 `tile_w`
接进同一套旋钮，5.58～5.60 在 HIP 上补平了调优面。但一直没有入口回答"同一个
`block` 在 NVIDIA 与 AMD 上是不是同一个最优值"——这决定了配置库能否跨厂商复用，
也决定调优这件事本身值不值得继续投入。

**落点**：新增 `benchmarks/tune_compare.py`，两段式（两张卡通常不在同一台机器上）：

- 采集 `--backend cuda --save cuda_tune.json`：直接复用 `tune.TUNE_CASES`
  （9 个单算子小图）与 `tune.search`，形状与随机种子都与 `tune.py` 一致；
  报告自带 `format` 版本与设备名，逐例记录赢家、**默认候选**、提升与被拒候选。
  默认候选是调优空间的第一个（后端此刻真正在跑的值），提升因此是"相对现状"。
- 对照 `--compare LEFT RIGHT`：按用例名对齐，只比较（a）配置是否相同，
  （b）各自**机内**相对默认的提升。跨机毫秒不可比，所以不打印、也不计算任何
  跨机耗时比值。

**两次独立采集（每例 2 轮取最好）**：

| 用例 | CUDA #1 | CUDA #2 | HIP #1 | HIP #2 |
|---|---|---|---|---|
| `add_same_1m` | 1024 | 256 | 128 | 512 |
| `add_broadcast` | 128 | 1024 | 1024 | 256 |
| `relu_1m` | 128 | 256 | 1024 | 1024 |
| `gelu_1m` | 1024 | 256 | 256 | 512 |
| `sigmoid_1m` | 512 | 256 | 1024 | 1024 |
| `silu_gate` | 1024 | 1024 | 256 | 512 |
| `step_1m` | 256 | 256 | 128 | 128 |
| `dropout_mask_1m` | 512 | 128 | 512 | 512 |
| `conv2d_tiled_1m` | auto | 64 | auto | 32 |

**第一件看见的事**：赢家对不上。同一台机器、同一份代码、同一个种子，CUDA 两次
只有 `silu_gate`(1024) 与 `step`(256) 一致，HIP 两次只有 `relu`、`sigmoid`、
`step`、`dropout_mask` 一致——9 例里换了 5～7 个。

**第二件事**：跨后端一致率取决于挑哪一次。CUDA #1 vs HIP #1 是 2/9
（`dropout_mask` 512、`conv2d` auto），CUDA #2 vs HIP #2 是 0/9。用这样的数据
回答"两家厂商最优值是否相同"，答案只会是采样噪声。

**于是先量噪声，不急着解释差异**：

1. 重复测量：同一配置连续测 4 次（每次 warmup 5 + 20 样本取中位数）。CUDA 上
   四档 `block` 的中位数差异 ≤3%（`relu` 1.1266～1.1521，`gelu` 1.0957～1.0994）；
   HIP 上 `relu` 为 5.2163～5.3175（2%），`add` 为 7.2575～7.7564（7%）。
2. 交替 A/B：只取两档，每轮反转顺序，各测 6 轮。

| 平台 | 用例 | 同轮内两档之差 | 跨轮台阶 |
|---|---|---|---|
| CUDA | `gelu` 256/512 | ≤1.5% | 无（1.178～1.199） |
| CUDA | `relu` 256/1024 | ≤1.4% | 无（1.182～1.205） |
| HIP | `gelu` 256/512 | ≤2%，第 4 轮例外：256 已落到 5.11，512 仍 6.70 | 6.9 → 5.1（1.34x） |
| HIP | `relu` 256/1024 | ≤2%，末轮例外：1024 先测 6.74，256 后测 5.09 | 6.8 → 5.1（1.33x） |

HIP 的 `gelu` 单看某一轮像"512 比 256 快 1.28x"，A/B 显示那是整机状态的台阶
（两档同时从 6.8ms 落到 5.2ms），不是配置差异。`relu` 末轮更直接：交换顺序后
先测的 1024 拿到旧状态、后测的 256 拿到新状态——"最优值"跟着测量顺序走。

**结论**：

1. 在 1M 逐元素张量上，四档 `block` 都不是有效旋钮：真实差异 ≤3%，整机漂移可达
   30%（WSL2 + DXG 路径更明显）。所谓"最优 `block`"只是噪声的一个样本。
2. 5.37 的"多轮取最小"能抵御整机漂移，但 `--rounds 2` 不足以在 3% 的差异上做
   判定；配置库按设备名命中，因此不会把某一次的噪声带到别的卡上——这是正确的行为。
3. "同一旋钮在两家厂商的最优值是否相同"在现有测量精度下**不可回答**。能回答的是
   "同一台机器、同一时间窗口内交替测量，快慢是否可复现"，而这个答案同样是否。
4. 5.60 在 HIP 上"搜出 9 条配置并逐位对拍一致"因此不能当成收益证据：它证明配置
   被正确应用，不证明配置更快。

**下一步的前提**（只记边界，不立项）：要让自动调优有结论，先把测量做成"交替 A/B +
重复到置信区间收窄"，再谈搜索算法；否则换搜索器只是在噪声上换一种采样方式。

> 5.140 已实现这半句（交替 A/B、配对差、bootstrap 区间、两半同向检查），
> 但结论仍然是否：逐元素 block 全部保留默认，conv2d 三次采集给出三个赢家。
> 区间能收窄，收窄后的中心却仍随采集漂移——"重复到区间收窄"本身不足以跨采集
> 定论，还要先消除轮内位置效应。
>
> 5.142 把"轮内跨度"接成拒绝下结论的门槛（`--max-round-seconds`）：跨度超限时
> 一律判 `long-round`、保留默认。它让 conv2d 的 44/45 个候选不再被当成结论，
> 也顺带暴露更强的偏置——首轮总是显著慢于后续轮次（0.87 vs 0.06、1.60 vs 0.40、
> 3.33 vs 2.4），更像冷启动而不是随机位置效应。5.143 加了预热轮
> （`--warmup-rounds`）：预热耗时与上面这些首轮同量级，正式轮全部回到稳态，
> dropout 的跨度误伤消失；但 5.142 判出的两个收益（`block=512` / 256）也随之
> 消失——首轮污染不只是让区间变宽，还会把配对差整体推向一侧。"按轮剔除超限
> 样本"仍未做。

**边界**：

- CUDA 侧是云上 RTX 2080 Ti（共享机器，时钟受邻居影响）；HIP 侧是 WSL2 DXG
  直通的 RX 7650 GRE（无 `/dev/kfd`）。两组数字都只在这条路径内成立。
- 只覆盖 f32 与 9 个形状；`block` 四档、`tile_w` 四档（含"自动"）。
  `tile_w` 唯一一次非默认赢家（CUDA #2 的 64，1.04x）同样在噪声内。
- 仓库里的 `benchmarks/cuda_tune.json` / `hip_tune.json` 是**一次采集**；重跑会得到
  不同赢家——这正是结论的一部分，不是文件过期。

**验证**：新增 `tests/test_tune_compare.py` 9 条用例，钉住报告对齐、键序无关、
缺一边不计入一致率、坏报告显式失败与采集报告的结构。本地 **1035 个测试
796 通过 / 239 跳过**；CUDA 真机 **1035 / 1014 通过 / 21 跳过**，`verify_cuda.py`
全 PASS；WSL HIP 真机 **1035 / 779 通过 / 256 跳过**。

### 5.64　硬件包：内容摘要与依赖声明

**动机**：5.43 的边界里留着两条——"依赖声明"与"签名与来源校验"都没做。于是
只要 zip 能解压、manifest 字段合法，包就会被识别并登记。两个后果都很隐蔽：包在
传输或落盘时坏了一个字节，框架照常加载，错误留到运行期以某条莫名其妙的 kernel
失败出现；包声明要用 `matmul`，而目标框架的算子表里没有它，也要等到第一次执行
才暴露。

**落点**：manifest 两个**可选**字段。

```json
{
  "files": {
    "mychip/__init__.py": "<sha256>",
    "mychip/backend.py": "<sha256>"
  },
  "requires": {"ops": ["matmul", "relu"], "python": ["numpy"]}
}
```

- `files`：相对路径 → sha256。`manifest.json` 自己不在清单里——它装着清单，
  不能自引用。
- `requires.ops`：包用到的算子名，必须都在框架的 `ir.ops.OPS` 里。
- `requires.python`：包导入的第三方模块，用 `importlib.util.find_spec` 查顶层名。

**三条判断（为什么这样做）**

1. **摘要不是签名。** `files` 只回答"包与打包时一致吗"，不回答"谁打的包"。
   伪造者把清单跟着重算一遍就能过。签名链要先有发布渠道与密钥管理，这一轮不做，
   也不把摘要包装成签名——把两者混为一谈比不做更危险。
2. **要么全有、要么没有。** 不声明 `files` 的包照旧加载：手工解压进目录（5.43
   的一等用法）本来就没有清单。但一旦声明，就要求**完全一致**——声明的文件必须
   存在、摘要必须相符、**包内不能有未声明的文件**。只校验声明过的子集，等于给
   "多放一个 `sitecustomize.py`"留门。运行期产物（`__pycache__`、`*.pyc`）不计入，
   否则"装好、用一次、再扫描"就会变成摘要不符。
3. **依赖在安装期查，只查"是否存在"。** 用 `find_spec` 而不是 import：安装期
   不执行包作者的代码（5.43 的"扫描不 import"原则不变）。`requires.python` 只取
   顶层模块名——"这台机器有没有装这个发行包"就是它能回答的全部；版本区间不做，
   那会滑向依赖解析（5.43 已明确这是包管理器的职责）。

**校验发生在两处**：`install_hardware` 解压到临时目录之后（失败则整个临时目录
删掉），以及目录扫描（`iter_installed`）。第二处不是重复劳动：手动放进目录是
5.43 支持的一等用法，只在安装路径校验的话，那条路径等于没有防护。坏包按既有
策略跳过，不拖累别的包。

**示例包**：`examples/hardware/build.py` 不再把源 manifest 原样塞进 zip，而是先
算 `files` 再写——写死的清单会随源码改动立刻过期，而"摘要过期"会被安装器当成
篡改拒绝。

**边界**

- 只覆盖文件内容；不覆盖时间戳、压缩率、条目顺序（它们不影响加载）。
- 不自动安装依赖，也不校验第三方库的版本；`requires.python` 只回答"能不能导入"。
- 不校验 `factory` 指向的模块是否真的导出那个类——那仍然要等到惰性加载
  （5.43 的取舍：扫描不 import 任何硬件代码）。

**验证**：`tests/test_hardware_package.py` 从 24 条增到 **40 条**，新增 16 条钉住：
正确清单可装；篡改 / 缺文件 / 多文件 / 坏摘要分别拒绝；无清单仍可用；运行期
字节码不计入；扫描期拒绝被改动过的已装包；不存在的算子被拒；缺失的模块被拒；
`requires` 的未知键与坏形状被拒。本地 **1051 个测试 812 通过 / 239 跳过**；
CUDA 真机 **1051 / 1030 通过 / 21 跳过**，`verify_cuda.py` 全 PASS；WSL HIP
真机 **1051 / 795 通过 / 256 跳过**。带清单的示例包在临时目录里端到端装过一次
（`novac 0.1.0`，49 个能力）。

### 5.65　自动分块：把 block 的取值从常量换成纯函数

**动机**：5.54 把 block 的**校验**提到了 `lower/plan.py`，但**取值**仍然是各处
写死的 `256`。两个后果方向相反，却都不报错：

- 工作量小的时候派空线程。`relu` 作用在 100 个元素上，一个 256 线程的块里有
  156 个线程什么都没做；整台机器只起来 1 个块，其余 67 个 SM 空着。
- 工作量中等的时候铺不满。8192 个元素按 256 分块只有 32 个块，而 2080 Ti 有
  68 个 SM，一半硬件没被派活。

**为什么不靠测**：5.63 已经在 9 个单算子用例上量过，1M 逐元素算子的 `block`
真实差异 ≤3%，淹没在整机 30% 的漂移里——"最优 block"在测量上不可判定。所以这里
**不做 autotune**，做的是把一个本来就该由形状决定的值交给一个纯函数，让约束有
执行者、决策只有一个来源。

**两条规则**（都不含测量结论，只回答"该不该派空线程"）

1. **不派空线程**：`block ≤ pow2_ceil(work / vec)`（向上取到 2 的幂），下限一个 warp。
2. **网格铺得开**：若 `work / block < target_blocks`，缩块到 `pow2_floor(work / (vec * target_blocks))`，
   使网格至少能盖住一屏 SM。

结果恒在 `[BLOCK_MIN, preferred]` 内，`preferred` 默认 `BLOCK_DEFAULT = 256`——
它是上限而不是起点：小工作量照样往下缩。

**语义**：`block=None`（默认）= 按规模自动；显式 `int` = "我要这么大"，自动不参与；
算子级调优覆盖 `_blocks[op]` 优先于两者。三者优先级固定，不会出现"自动把用户
指定的值改掉"。

**落点**

- `lower/plan.py`：新增 `auto_block(work, *, vec, preferred, target_blocks)` 与
  `BLOCK_DEFAULT`；`plan_elementwise` 的 `block` 默认 `None`，新增 `target_blocks`。
- `backends/cuda/backend.py`：构造函数 `block: Optional[int] = None`，`sm_count` 取自
  `getDeviceProperties(0)["multiProcessorCount"]`；新增 `_grid(op, work, vec)` 一处解析
  `覆盖 → 构造参数 → auto_block`，22 处 launch 全部改走它，不再各自算 block。
- 调优面：`tuning_space` 的首候选可以是 `None`（= 当前生效的自动值），
  `set_tuning({"block": None})` 等于撤掉覆盖，回到自动。
- `backends/hip/`：绑定 `hipDeviceGetAttribute` 取 `hipDevAttrMultiProcessorCount`。

**一个实测边界**：WSL2 + DXG 直通下 `hipDeviceGetAttribute` 基本不填值——枚举
0..44 号属性时大量返回 0，含 16（SM 数）本身。因此 HIP 侧 `sm_count` 退到 1，只剩
第一条规则（不派空线程）；在原生 Linux 的 ROCm 上应能取到真实值。没有改用
`hipDeviceProp_t` 结构体偏移解析：那要求对驱动结构体布局做假设，违背本项目
"不按布局读未知结构"的一贯做法，宁可少一条规则。

**实测（捕获口径，各形状 200 轮取中位数）**

| n | CUDA auto block | fix/auto | HIP auto block | fix/auto |
|---|---|---|---|---|
| 2048 | 32 | 1.00 | 256 | 1.00 |
| 4096 | 32 | 1.00 | 256 | 0.95 |
| 8192 | 64 | 1.01 | 256 | 1.00 |
| 16384 | 128 | 0.99 | 256 | 0.90 |
| 32768 | 256 | 0.96 | 256 | 1.00 |
| 65536 | 256 | 1.00 | 256 | 0.90 |

**结论**：没有可测的性能收益，也不宣称有。CUDA 上 `sm_count=68`，自动块确实随
形状变化；但整图 replay 的固定开销约 78 µs，把 kernel 本身的差异淹没了。HIP 侧
因为拿不到 SM 数、auto 恒为 256，退化成与固定值同一条路径（0.90 那两档是噪声）。
写这一节的意义是：block 的取值从此有单一定义处、有约束、可被调优面表达为
"自动"，而不是散落在 22 个 launch 点上的同一个魔数。

**边界**

- 只作用于"扁平内核"（逐元素 / 池化 / copy / gather 这类线性下标的）。归约类的
  block 与 shared 暂存区尺寸绑定，分块卷积有自己的 2D block，都不走 `auto_block`。
- 只用 SM 数，不用占用率模型：寄存器压力与共享内存占用在计划层看不到，套一个
  假的占用率公式只会把"不知道"包装成"知道"。
- 捕获路径的 block 在捕获时按当时形状固定；换形状要重新捕获，这与 5.61 的既有
  约定一致。

**验证**：`tests/test_plan.py` 新增 `TestAutoBlock`（7 条）与 `TestPlanElementwise`
的 3 条（默认自动、按 `target_blocks` 铺开、显式值保留）；`tests/test_cuda_backend.py`
新增 2 条（自动值随形状变化且覆盖优先、小张量上自动与 32/256/1024 三档逐位一致），
并把调优空间用例改成接受 `None`。总用例 **1051 → 1063**。本地 **1063 个测试
822 通过 / 241 跳过**；CUDA 真机 **1063 / 1042 通过 / 21 跳过**，`verify_cuda.py`
全 PASS（`relu` 候选 `[None, 128, 256, 512, 1024]`，搜索往返后的最优为
`{"block": None}`）；WSL HIP 真机 **1063 / 805 通过 / 258 跳过**。

### 5.66　训练循环：把四步样板收进一个类

**动机**：5.21 交付了"前向 → 反向 → 更新"四步，但样板留在每个调用方手里——
`test_train.py` 的 `_train` 与 `_train_steps` 各写一遍、示例脚本再写一遍。重复
的不是那四行调用，而是三处容易错且报错难看的地方：

- 种子名固定是 `"d" + 输出名`，写成 `"dy"` 或忘了加 `d` 时，报的是"缺少图输入"；
- 种子形状必须取自**前向输出**的类型（反向图的输入顺序是"前向输入 + 种子"，
  种子不固定在第一位）；
- 多输出图的每个输出都有一个种子输入，只喂目标那个会直接 `KeyError`
  （实测：`缺少图输入: dsquared`）。

**落点**：`train/loop.py` 的 `Trainer`。没有新的执行机制——`step` 就是一次前向
`execute`、一次反向图 `execute`、一次优化器原地更新；`run` 只是按 `epochs`
重复并把全局步号传给 `on_step`。反向图在构造期建一次，与手写循环的做法一致。

**两个判断**

1. **多输出图喂零，而不是让调用方补种子。** 非目标的输出视为"不参与损失"，
   种子填零。要求调用方自己给每个输出准备种子，等于把"反向图有几种输出"这件
   内部事实漏给用户。
2. **符号形状在构造期拦下，并解释原因。** 反向图的归约需要把梯度广播回输入
   形状（`ir/grad_rules.py::_static`），符号维下拿不到。错误信息直接指出是哪几个
   输入，并给出 `DataLoader(drop_last=True)` 或补齐的替代路径，而不是让用户
   看到一句 reduce_sum 的内部抱怨。

**试过又收回的方案（值得记下来）**：最初想给 `mse` / `softmax_cross_entropy`
加 `reduction="sum"`，再让循环在运行时把梯度与上报值同除 `1/N`，从而绕开
`mean` 对静态 count 的依赖。实现后发现阻塞点根本不在损失函数：只要图里有符号
维，构建反向图时 `reduce_sum` 的 vjp 就会失败——它需要静态形状才能 `reshape`
回原形状。`reduction` 参数因此没有独立价值，已撤回；真正的解法是运行期形状
表达式，属于 5.53 的后续，不是训练循环能绕过的。

**边界**

- 只做"静态形状、单设备、手动 `epochs`"；没有 `fit` 式的回调体系、验证指标与
  早停。
- 不接管调度、裁剪、累积、EMA：它们仍是可组合的独立件（`apply_lr`、
  `clip_global_norm`、`GradientAccumulator`、`EMA`）。
- `on_step` 只拿 `(index, loss)`，不传模型或优化器状态。

**验证**：`tests/test_train_loop.py` 9 条——200 步线性回归收敛（loss < 1e-4，
参数回到构造用的真值）、`step` 返回图内损失且参数被原地更新、非标量输出被拒、
多输出图可指定目标、参数必须是图输入、符号输入在构造期被拒、`on_step` 收到
全局步号、`epochs` 校验。本地 **1073 个测试 831 通过 / 242 跳过**；CUDA 真机
（RTX 2080 Ti，cupy 14.2）**1073 / 1020 通过 / 53 跳过**。

真机上的端到端核对用 `CNNSpec(in_channels=1, image=8, widths=(4, 8))` 建分类
网络：conv2d(padding=1) → relu → maxpool2d → 全局 avgpool2d → reshape →
matmul → 交叉熵，前向与反向都走后端实现。60 步内 loss 从 1.218371 降到
0.000027，参数最大变化量 5.098，首步损失与参考后端逐位相同。

**这条路径立刻暴露了一个真 bug**：`CudaBackend.to_array` 用
`cp.ascontiguousarray` 统一物化，而 cupy 对它作用于 0 维数组时返回 shape
`(1,)`（numpy 不会）。反向图的种子输入恰恰是 0 维，于是训练在该后端上直接报
「输入 %dloss 秩不匹配：期望 0，实际 1」——参考后端永远发现不了，因为 numpy
保留 0 维。修复是 0 维直接返回（标量天然连续，不需要复制），并加
`test_zero_dim_input_keeps_rank_zero` 钉住。结论很直接：**只在 CPU 上跑训练
测试，不足以证明训练能落到硬件上**。

---

### 5.67　硬件包来源签名：信任锚放在安装方，不放在包里

**动机**：5.64 的 `files` 摘要只回答"包在打包后有没有被改过"。它挡不住伪造——
攻击者拿到一个合法包，改几行代码、把 `files` 里的 sha256 全部重算，摘要依然
"完全吻合"。摘要与签名解决的是两个不同的问题：前者是完整性，后者是来源。

**信任模型**：包内声明的公钥不能自证。`install_hardware(..., trusted_keys=[...])`
由**安装方**给出受信任的公钥集合，包里的公钥必须命中这份列表才继续验签。这样
攻击者即使自带一把新私钥、把签名与公钥一起换掉，也会在"公钥不在列表里"被拦下。

**签名覆盖整份清单，而不只是文件清单**。签名载荷是"删掉 `signature` 字段后的
整份 manifest 的规范化 JSON"（`sort_keys=True`，紧凑分隔符，UTF-8）。只签
`files` 的话，改 `version` 或 `factory` 就能把包指向别的工厂函数而签名仍然有效。

**三种情况都拒绝，只有一种放行**

| 包带签名 | 调用方给了 `trusted_keys` | 结果 |
|---|---|---|
| 是 | 是 | 验签通过才装，`HardwareRecord.signed = True` |
| 是 | 否 | 拒绝——"有签名但没人验"与没签名等价，静默跳过是安全反模式 |
| 否 | 是 | 拒绝——要求了来源就不能装来路不明的包 |
| 否 | 否 | 放行（向后兼容：不关心来源的本地开发流程照旧） |

**落点**：`backends/hardware.py` 的 `_check_signature` / `_signature_payload` /
`_public_key_bytes`，`_validate` 与 `_record_from_dir` 透传 `trusted_keys`，
`install_hardware` 新增同名关键字参数并把 `signed` 记进 `HardwareRecord`。算法
白名单只有 `ed25519`，未知算法直接拒绝而不是忽略；签名必须是 64 字节。

**边界**

- 不解决"公钥怎么送到安装方"：命令行参数、环境变量、配置文件的渠道都在包外，
  属于发布方与部署方的约定。私钥保管同理。
- `cryptography` 是**惰性导入**的可选依赖：只有装带签名的包时才需要，缺库时报
  "pip install cryptography"而不是让整个导入链失败。
- 不做吊销、过期与密钥轮换。要换钥就换 `trusted_keys` 里的条目，旧的签名会自然
  失效。

**验证**：`tests/test_hardware_package.py` 的 `TestSignature` 9 条（无
`cryptography` 时整组跳过）——受信任密钥成功且 `signed` 为真、公钥可用十六进制
字符串给出、公钥不受信被拒、带签名却不给 `trusted_keys` 被拒、给了
`trusted_keys` 但包未签名被拒、未签名且未要求来源仍可装、改清单字段后签名失效、
改文件内容被 `files` 摘要先一步拦下、未知算法被拒、错误签名字节被拒。该文件
合计 **50 个用例通过**。

---

### 5.68　动态执行第一版：按形状建图，不放弃静态执行

**动机**：5.25 的 eager 前端补齐了"写起来像动态图"的手感，但没有解决真正的
variadic 需求——输入形状在运行时才定（变长序列、最后一档不满的 batch、
图像尺寸不固定）。静态图要求构图期就给出形状，这类需求此前无解。

**为什么不直接做真正的 define-by-run**：5.25 已写明静态图换来的三件东西——
构图期后端能力预检、反向图可序列化独立分发、CUDA Graph 捕获要求固定路径。
逐算子即时求值会把三条一起放弃，而那三条正是当前后端体系的地基。所以这一版
**不动静态路径**，另加一层：图按当前输入形状构建，同形状复用。

**机制**

```python
runner = DynamicRunner(build, capacity=8)   # build(shapes) -> Graph
y = runner.run({"x": batch})["y"]           # 首次按形状建图，之后命中缓存
```

缓存键是 `((输入名, 形状, dtype), ...)` 排序后的元组（与输入字典顺序无关），
超出 `capacity` 按最久未用淘汰。`builds` 记录建图次数，`shapes` 按最近使用
顺序暴露缓存键（调试用）。

**一条硬约束：`build` 只能依赖形状。** 缓存键里没有数据，如果 `build` 按数组
内容分支，形状相同而内容不同时就会命中错误的图，而且不会报错。数据相关的
控制流因此放在两次 `run` 之间——在 Python 里读回真实输出，再决定下一步调用哪张
图。这正是"动态执行手感"真正需要的东西，也把显式分步留在了用户手里，而不是在
构图函数里藏状态。

**它不是完整的 define-by-run**，一次 `run` 内部仍是静态图：没有逐算子即时求值，
也没有"中途读中间值再决定下一个算子"。这一版的定位是"形状随数据变化"，不是
"控制流进图"。要真做后者，前置是运行期形状表达式与可重入的图构建器，属于 5.53
之后的问题。

**边界**

- 后端默认参考后端；传 CUDA/HIP 后端时每步仍把结果拷回主机（与既有后端行为
  一致），`run` 的返回值是主机侧输出。
- 缓存只按形状区分，不感知 dtype 之外的输入属性（例如布局）；输入必须是带
  `.shape` 的数组，标量输入报错。
- 不做并发保护：`DynamicRunner` 不是线程安全的，多线程各自持有实例。

**验证**：`tests/test_dynamic.py` 10 条——同形状只建一次图、新形状重建、缓存键
与输入声明顺序无关、`capacity=1` 时按 LRU 驱逐（三次不同形状 → `builds == 3`）、
两次 `run` 之间按输出值分支、结果与参考后端逐位一致、`build` 返回非 `Graph` 被拒、
`build` 非可调用与 `capacity` 非法被拒、输入非数组或为空被拒、`shapes` 按最近
使用排序。

---

### 5.69　cuDNN 对照：自研卷积离工业实现差 4～12 倍

**动机**：1.2 里 CNN 一行挂着"缺 cuDNN 对照"很久了。没有这个数字，"分块直算
够用"就只是一句自我评价；有了它，才知道该不该继续投入 kernel。

**为什么直调 cuDNN 而不引 PyTorch**：要测的是"同一条卷积在硬件库上的耗时"，
不是框架差距。PyTorch 的 conv 外面套着 autograd、参数解析与 dispatcher，把这些
算进去会把差距夸大成"框架差距"。`benchmarks/cudnn_compare.py` 用 ctypes 调
cuDNN 的 legacy API（v9 仍保留），只依赖系统里已有的 `libcudnn.so.9`，不新增
Python 依赖；找不到库时打印一行原因并退出 0，与 `conv_compare.py` 一样是可选
测量，不进 CI 门禁。

**两条对齐细节**

1. **算法**：`cudnnGetConvolutionForwardAlgorithm_v7` 取第一个返回项（cuDNN 自己
   的启发式）。不遍历全部算法取最快——那是"cuDNN 调优后的上界"，不是"默认能用
   的水平"。
2. **bias**：本项目把 bias 融在卷积核里，cuDNN 侧在计时窗口内用一次 cupy 加法
   对齐，而不是 `cudnnAddTensor`（后者会额外引入一次全量读写）。

**结果（RTX 2080 Ti，cuDNN 9.27.0，f32 / NCHW，每步取 10 次中位数；数字在 5.71
修复之后重新采集）**

| 形状 | tiled | im2col | cuDNN | tiled/cuDNN | im2col/cuDNN |
|---|---|---|---|---|---|
| 3x3_c3 | 0.4765 | 0.4267 | 0.1165 | 4.1x | 3.7x |
| 1x1_c32 | 0.5034 | 0.4545 | 0.0706 | 7.1x | 6.4x |
| 3x3_c8 | 1.0477 | 0.6887 | 0.1091 | 9.6x | 6.3x |
| 3x3_c16 | 1.2748 | 0.7416 | 0.1190 | 10.7x | 6.2x |
| 3x3_c24 | 1.3921 | 1.1846 | 0.1344 | 10.4x | 8.8x |
| 3x3_c32 | 1.7851 | 0.8684 | 0.1555 | 11.5x | 5.6x |
| 5x5_c16 | 1.7670 | 0.8184 | 0.1451 | 12.2x | 5.6x |
| 1x1_c256 | 1.0498 | 1.0062 | 0.2050 | 5.1x | 4.9x |
| 3x3_c64 | 3.1155 | 1.2665 | 0.2892 | 10.8x | 4.4x |
| 3x3_c128 | 1.8671 | 0.6475 | 0.2005 | 9.3x | 3.2x |

5.71 之前同一张表是 9.4～18 倍——那组数字里有一半是"每步重新生成 kernel 源码"
的 host 开销，不是卷积本身。修掉之后倍数落到 4～12 倍，才是可以拿来讨论
"要不要继续写 kernel"的量级。

**三个读法**

1. **分块直算随通道数变差**（4.1x → 12x），且最差点不在最大通道，而在
   3x3_c32 / 5x5_c16 一带。它的每个输出元素独立做 `in_channels × k × k` 次乘加，
   权重只在 L1 里碰运气；cuDNN 走隐式 GEMM + 寄存器分块，复用随通道数增长。
2. **im2col + cuBLAS 在大通道上追得很近**（3x3_c128 只差 3.2x、3x3_c64 4.4x），
   小通道反而更差（3x3_c3 3.7x 与 tiled 同量级）——im2col 的列矩阵在 K 小时
   放大倍数低，收益抵不过那次显式拷贝。这与 `conv_compare.py` 的结论一致，只是
   现在有了外部锚点。
3. **1x1 卷积不再是最大短板**（5.1～7.1x）。它在数学上就是 GEMM，cuDNN 直接
   交给高效 GEMM；我们仍按空间卷积的分块模板处理，所以仍然吃亏，但差距已经
   小于中等通道的 3x3。

**必须说明的口径**：这组数字是**同一时刻**测的（cuDNN 与自研核在同一进程、
同一时钟状态），所以**比值**与环境无关；但**绝对毫秒**不能拿去和
`baseline_cuda.json` 比——那次的采集环境与现在不同（见 5.70）。

**边界**：只覆盖 f32 / NCHW / 方形核 / stride=1；不做 cuDNN 的算法搜索、winograd
与 f16。也不把 cuDNN 接进生产路径——1.2 里"cuDNN 未接"仍然成立，这里只是测量。

---

### 5.70　CI 作业在真机逐条实跑：正确性全绿，性能门禁为红

**动机**：`.github/workflows/ci.yml` 早就写好了，但一直没在托管平台上跑过
（缺 self-hosted GPU runner 的注册令牌）。能做的替代实验是：在 RTX 2080 Ti
上把 gpu 作业的六个步骤逐条执行，验证**命令本身**可跑通，并把真实结果记下来。

**结果**

| 步骤 | 结果 |
|---|---|
| 环境自检（`cupy` 版本 + `verify_cuda.py`） | 通过：cupy 14.2.0 / runtime 12090，全 PASS |
| 全量单元测试 | 通过：1094 个测试，53 跳过（即 1041 通过） |
| CUDA 一致性报告 | 通过：69 executed / 5 refused / 0 failed |
| 性能回归 `benchmarks.run --check` | **失败**：5 个用例超 25% 阈值（5.71 修复后仍红，见下） |
| 单算子调优往返（搜索 + 应用后逐位对拍） | 通过：9 条配置，全部逐位一致 |
| 吞吐 / 显存趋势 | 通过：写入 24 条记录 |

**性能门禁的失败细节（5.71 修复前，两轮独立采集）**

| 用例 | 基线 | 第一轮 | 第二轮 | 倍数 |
|---|---|---|---|---|
| matmul_256 | 0.2350 | 0.3144 | 0.3315 | 1.34～1.41 |
| matmul_1024 | 1.7596 | 2.0952 | 2.2182 | 1.19～1.26 |
| matmul_batched | 0.4508 | 0.5229 | 0.5361 | 1.16～1.19 |
| conv2d_3x3_pad1 | 0.3393 | 1.1922 | 1.2610 | 3.51～3.72 |
| conv2d_1x1 | 0.3575 | 1.2147 | 1.2251 | 3.40～3.43 |
| maxpool2x2 | 2.9697 | 3.4141 | 3.6196 | 1.15～1.22 |
| transformer_block | 1.1825 | 2.9066 | 2.9081 | 2.46 |
| cnn_lenet | 0.6735 | 2.7283 | 3.0255 | 4.05～4.49 |

**怎么读**：两轮之间的比值几乎不动，说明**不是抖动**；但整机层面已经整体慢了
15～40%（`maxpool2x2` 访存主导、`matmul_256` 算力主导，两者都在这条带上），
卷积与多节点图再额外慢 2～4.5 倍。负载下采样到的 SM 时钟是 1350 MHz（上限
2100），功耗 60 W（上限 250 W），温度 34 ℃——机器没有过热降频，但确实不在
基线采集时的状态。这个"额外慢 2～4.5 倍"被 5.71 追到了根因。

**5.71 修复后重测（同一台机器、同一份基线）**

| 用例 | 基线 | 修复后 | 倍数（前 → 后） |
|---|---|---|---|
| matmul_256 | 0.2350 | 0.3000 | 1.34 → 1.28 |
| matmul_1024 | 1.7596 | 2.0235 | 1.19 → 1.15 |
| matmul_batched | 0.4508 | 0.5175 | 1.16 → 1.15 |
| conv2d_3x3_pad1 | 0.3393 | 0.4478 | 3.51 → **1.32** |
| conv2d_1x1 | 0.3575 | 0.4718 | 3.40 → **1.32** |
| maxpool2x2 | 2.9697 | 3.3420 | 1.15 → 1.13 |
| transformer_block | 1.1825 | 1.8395 | 2.46 → 1.56 |
| cnn_lenet | 0.6735 | 1.0566 | 4.05 → **1.57** |

**仍然红，但性质变了**：超阈值的 5 个用例统一落在 1.28～1.57 倍，其中
`matmul_*` / `maxpool2x2` 是纯常数开销主导、对代码路径不敏感的用例，它们已经占到
1.13～1.28——也就是说**整机漂移本身就在阈值边缘**。卷积与多节点图剩下的
1.3～1.6 倍里，还能再拆成"整机漂移"与"每步 Python 派发常数"两层，后者属于执行
器架构（逐节点走 `invoke` / `dispatch` / `_bind`），不是一次能修掉的小改动。

**为什么不更新基线**：那台机器上没有版本历史，无法回退到基线采集时点做单变量
对照；把基线重采一遍只会把红灯改成假绿，后面真出回归也看不出来。所以这里的
处理是**记录**而不是修：即使 5.71 把差距砍掉了一半，门禁仍然红，证据与下一步
（执行计划缓存，让每步不再逐节点走 Python）都写在上面。

**顺带证明的一件事**：5.69 的 cuDNN 对照与这次门禁失败用的是同一台机器、
同一个 cupy，而 cuDNN 侧 3x3_c3 每步只要 0.1165 ms，同形状的 tiled 是 0.4765 ms。
也就是说剩下的差距不是后端连不上硬件，而是自研卷积与工业实现的真实距离；
环境漂移只是叠在上面的一层，且只值 1.15～1.28 倍。

**边界**：这次是"人肉按作业脚本执行"，不是 GitHub Actions 的调度、缓存、artifact
上传与失败传播。托管实跑还差 runner 注册与一次 `workflow_dispatch`。

---

### 5.71　每步重生成 kernel 源码：一次 61% 的 host 开销

**怎么发现的**：5.70 的性能门禁里，`conv2d_*` 与 `cnn_lenet` 慢 3.4～4.5 倍，
而同样是单节点的 `matmul` / `maxpool2x2` 只慢 1.15～1.25 倍。第一反应是 GPU
掉频，于是做了三件事把"host 还是 device"分开：

1. 单步同步计时（原口径）；
2. 只计 host 派发、不 `synchronize()`（cupy 的 launch 是异步的，纯 host 时间
   包含 Python 准备但不含等待）；
3. 连发 200 次再一次同步，除以 200 得到流水吞吐。

结果**三者几乎相等**（conv2d 1.194 / 1.170 / 1.240 ms），也就是 GPU 根本没在
等——时间全花在 Python 侧。再上 `cProfile`，热点非常集中：50 次 `execute` 里
`codegen.render → lower.emit → cstyle.print_kernel` 占 55%，其中
`_conv2d_tiled` 一次就要重新拼一遍 C 源码再打印成字符串。

**根因**：`_kernel(key, source, func)` 按 key 缓存的是**编译好的函数**，但
`source` 在传参时就已求值——每步、每个节点都白算一遍源码，然后丢掉。卷积的
源码与形状无关（形状走 kernel 参数），所以这 0.4 ms/步是纯浪费；逐元素链上
节点多，同样的浪费叠 20 多次，于是 `transformer_block` / `cnn_lenet` 一起变慢。

**修法**（两处，都很小）

```python
kernel = self._kernel(
    ("conv2d_tiled", self._kind(x)),          # 键里补上 dtype（原来缺）
    lambda: self.codegen.render(self.codegen.conv2d_tiled_source(), x.dtype),
    "conv2d_tiled",
)
```

`_kernel` 改为接受"返回源码的函数"，**只在编译缓存未命中时才调用它**：

```python
fn = self._functions.get(key)
if fn is None:
    text = source() if callable(source) else source
    module = self.cp.RawModule(code=text, options=("-std=c++11",))
    ...
```

31 处 `self.codegen.render(...)` 全部改成惰性；HIP 后端的 `_kernel` 同步支持
callable（它的缓存键带源码摘要，仍需拿到字符串，行为不变）。

**顺带修掉的一个隐患**：分块卷积的缓存键原本只有 `"conv2d_tiled"`，不含 dtype，
同一后端实例先跑 f32 再跑 f16 会复用错误的 kernel。这次改成
`("conv2d_tiled", kind)`。

**效果（RTX 2080 Ti，同机同口径）**

| 用例 | 修复前 | 修复后 | 变化 |
|---|---|---|---|
| conv2d_3x3_pad1 | 1.194 | 0.448 | −62% |
| conv2d_1x1 | 1.215 | 0.472 | −61% |
| transformer_block | 2.893 | 1.840 | −36% |
| cnn_lenet | 2.728 | 1.057 | −61% |

**为什么不是"缓存源码字符串"就完事**：源码里含形状与 attrs（`_binary` 的
broadcast 下标、融合链的步数），字符串缓存要自己维护键；把"生成时机"交给
已有的编译缓存，键的语义不变，也就不需要第二套键。

**边界**：源码生成只是一次性成本，真实训练里几百步以后占比会归零；这条修复
救的是"小张量 + 多节点"的基准与推理延迟。HIP 侧仍每步生成源码（缓存键要算
摘要），未改。

---

### 5.72　执行计划缓存：把"每步重算的静态计划"收回一次

5.71 之后 host 侧仍有一笔与形状无关的固定开销，`cProfile` 里分得很散，
没有单一热点。按"是不是每步都在重算同一个答案"筛了一遍，收掉三处：

| 每步重复的工作 | 位置 | 为什么可以缓存 |
|---|---|---|
| 融合分组 `_group_elementwise` | 执行器每步调用 `group_nodes` | 判据全是静态的（算子表、被使用次数、形状相同） |
| `resolve_attrs` 与 `TensorType.resolve` | 每个节点、每一步 | 图静态时与绑定无关；动态图里 attrs 通常也不含符号维 |
| `auto_block` + `grid_for` | 每个逐元素节点、每一步 | 纯函数，键是 `(op, work, vec)` |

执行器新增一个 `_GroupPlan`（每个执行组一份：输入在环境里的位置、组内前序
输出占位标记、静态 attrs、期望类型），按图缓存在 `WeakKeyDictionary` 里，
以 `Graph` 新增的结构版本号 `_revision` 失效——构建入口（`input` /
`add_node` / `output`）自增，节点 attrs 构建后只读（5.106 起由 `Graph.seal()`
强制，此前只是注释约定）。动态图（含符号维）的
`attrs` / `expected` 留 `None`，运行时仍按本次绑定解析，语义不变。

后端侧另加两处：`_contig()` 只在真的不连续时才调 `cp.ascontiguousarray`
（cupy 对连续数组是零拷贝，但每次调用仍要穿过一层 Python 包装，约 5 µs，
而它出现在几乎每个算子的参数入口上）；`_grid()` 把铺法记在
`(op, work, vec)` 上，`set_tuning` 改动 block 覆盖时整表清空。

**效果（RTX 2080 Ti，host 侧连发 200 次不 sync）**

| 用例 | 5.71 后 | 本轮后 | 变化 |
|---|---|---|---|
| conv2d_3x3_pad1 | 0.460 ms | 0.448 ms | −2.6% |
| transformer_block | 2.089 ms | 1.892 ms | −9.4% |
| cnn_lenet（门禁口径） | 1.057 ms | 0.913 ms | −14% |

transformer_block 的端到端数字没动（1.84 → 1.88，在噪声内）：它的 host 时间
本来就被 12 次 H2D（0.42 ms）与 8 次 cuBLAS 封装调用（0.19 ms）压住，省下的
0.20 ms 只是把已经排在队列后面的空隙填平。

**门禁为什么还红：用纯库用例当漂移标尺**

本轮重跑门禁，`matmul_256`（单节点、走 cuBLAS、本轮代码路径未改）也从
0.235 变成 0.310 ms——**1.32 倍**。它没有任何可优化的 Python 派发，只能说明
当前整机比采基线时慢三成。拿它当归一化标尺：

| 用例 | 当前 / 基线 | 相对标尺（1.32） |
|---|---|---|
| matmul_256（标尺） | 1.32 | 1.00 |
| conv2d_3x3_pad1 | 1.26 | 0.95 |
| conv2d_1x1 | 1.29 | 0.98 |
| cnn_lenet | 1.37 | 1.04 |
| transformer_block | 1.62 | 1.23 |

卷积与 CNN 已经追平甚至优于标尺，只有 24 节点的 Transformer Block 还剩
约 23% 的真实差距，来源是逐节点 `invoke` / `dispatch` / 逐节点分配输出。
两条路摆在这里：门禁引入"漂移标尺"（判定语义要改，且要防止把真实退化
一起归一化掉），或继续压派发常数（bias 融合、执行计划里预取 kernel）。
基线仍**不重采**。

### 5.73　压派发常数：卷积的 bias 融合与符号图的解析备忘

选定第二条路：不动判定语义，只把每步确实在重复付的钱收掉。`transformer_block`
不含卷积，所以两笔账要分开收——卷积走"少发一个核"，Transformer 走"少算
一遍同样的答案"。

**bias 融合（卷积）**：`conv2d` 此前在分块核算完之后固定再发一个
`(1, O, 1, 1)` 的广播加法核。现在 bias 作为内核参数，写回时就地融合成
`acc[j] + (has_bias ? b[oc + j] : 0.f)`：一次 launch、一次读写，而不是两次
launch 加一次全量往返。可空指针沿用注意力内核的先例——没有 bias 时传一个
已有的有效指针、把 `has_bias` 置 0，三元表达式只在有 bias 那一支解引用。
三处必须同时改：L2 的 `conv2d_tiled()`（参数表 + 写回表达式）、手写对照
模板 `conv2d_tiled_template()`（它与 L2 逐 token 相等，是这个内核的测试
判据）、`backend._conv2d_tiled` 的传参。f16、im2col 兜底与 `groups > 1`
仍走末尾的广播加法：它们的瓶颈不在这颗核上，不值得各写一条融合路径。

**解析备忘（符号图）**：`transformer_block` 的输入是 `f32[B, S, 64]`，整张
图是符号图，5.72 的静态计划对它无效——每步仍要做 24 次 `resolve_attrs`
（递归替换 attrs 里的符号）和 24 次 `TensorType.resolve`（逐维校验并重建
类型）。可这两样只取决于本次绑定值，形状不变时就是同一个答案。
`_GroupPlan.resolve()` 因此加了一格备忘：键是本次完整绑定，命中就直接复用
上次的 attrs 与期望类型。只留最后一格——形状切换就重算，内存不随形状种类
膨胀，语义也不依赖"形状稳定"这个假设。静态图仍然走构建期算好的那份，
一次字典比较都不用付。

**门禁（RTX 2080 Ti，每例 3 轮取最好一轮）**

| 用例 | 基线 | bias 融合后 | 加解析备忘后 | 归一化 |
|---|---|---|---|---|
| matmul_256（漂移标尺） | 0.235 | 0.250 | 0.250 | 1.00 |
| matmul_1024 | 1.7596 | 1.690 | 1.717 | 0.92 |
| matmul_batched | 0.4508 | 0.444 | 0.429 | 0.90 |
| conv2d_3x3_pad1 | 0.3393 | 0.303 | 0.304 | 0.84 |
| conv2d_1x1 | 0.3575 | 0.313 | 0.312 | 0.82 |
| maxpool2x2 | 2.9697 | 2.857 | 3.050 | 0.97 |
| transformer_block | 1.1825 | 1.495 | 1.362 | 1.08 |
| cnn_lenet | 0.6735 | 0.694 | 0.692 | 0.97 |

"归一化"一列按 `matmul_256` 的当前/基线比值（1.063）折算——它本轮代码
路径未改，是整机漂移的标尺。折算后除 `transformer_block` 还剩 8% 外，
全部达到或优于基线。bias 融合让两个卷积用例一次性越过基线（−11% / −13%）；
解析备忘让 transformer 从 1.495 降到 1.362 ms（−8.9%），跨过 25% 的门禁线。

**门禁转绿，基线仍未重采**：`benchmarks.run --check baseline_cuda.json
--max-slowdown 0.25` 输出"回归检查通过"。这一轮既没改判定语义（不做漂移
归一化），也没有重采基线；`matmul_256` 的 +6.3% 记在漂移账上。剩下的 8% 来自逐节点的 `invoke` / `dispatch` 与逐节点分配输出，属执行器架构改造，
不在"常数级"范围里。

---

### 5.74　conv2d 的两条快路径：cuDNN 接进生产路径，调优面放开到三个旋钮

**动机**：5.69 只把 cuDNN 当对照物量了一次（自研分块核慢 4～12 倍），生产路径
没变。同时 5.56 只把 `tile_w` 开放成旋钮，`roc`（输出通道分块）与 `cb`
（staging 深度）虽然早在 `lower/plan.py` 里就是纯函数参数，却只能靠改代码试。
两件事落在同一个位置：`conv2d` 用哪个实现、分块怎么铺。

**cuDNN 是可选实现，不是唯一实现**（`backends/cuda/cudnn.py`）

- 后端默认 `conv_impl="auto"`：能加载 libcudnn 就走 cuDNN，加载不到或形状不支持
  就静默回落自研核。回落只换实现、不换数值语义，测试用同形状对拍盯住。
- 覆盖范围刻意收窄：NCHW、`groups=1`、f32 / f16，任意 stride / pad / dilation。
  分组卷积与 f64 仍走 im2col——把 group 语义塞进来会多出一条只在部分机器上验证过
  的分支，收益抵不上风险。
- **加载失败只试一次**。缺 cuDNN 的机器上这个异常是必然发生的，每步 `try/except`
  会把 5.72 / 5.73 刚压下去的派发常数抬回去；失败原因存进 `_cudnn_broken`，之后
  直接判"不走 cuDNN"。显式 `conv_impl="cudnn"` 时仍抛 `BackendCapabilityError`，
  并带上当初的失败原因。
- **捕获期不切实现**。cuDNN 的 algo 查询是 host API，在 CUDA Graph 捕获里调用不
  安全；而且"平时 cuDNN、捕获时自研核"会让两条路径的数值与显存布局不一致。捕获
  要的是路径固定，这里就保持固定。
- 算法与 workspace 按 `(x 形状, w 形状, dtype, pad, stride, dilation)` 缓存；bias
  用 `(1, K, 1, 1)` 的 NCHW 描述符加一次 `cudnnAddTensor`，不改前向核。
- 这仍属 ADR-004：IR 里的 `conv2d` 是语义完整的 PORTABLE 算子，"用 cuDNN 实现它"
  是后端内部的实现选择，不产生 LIBRARY 节点。

**三个旋钮**：`_CONV_KNOBS` 把 `tile_w` / `roc` / `cb` 收进一张表（参数名 →
实例字段 → 候选值），`tuning_space("conv2d")`、`set_tuning`、`clear_tuning`、
`tuning_config` 全部按这张表遍历，不再各写一份。

- 候选值是 2 的幂，到计划层上限 `CONV_ROC_MAX` / `CONV_CB_MAX` 为止（5.16 已把
  上限本身定成最优值，候选没有理由越过它）；当前值排第一位，即空间默认值。
- `None` 表示"按形状自动推导"，它随形状变化、钉不成常数，所以永远在候选里。
- `set_tuning` **先整份校验再落库**：中间某个值非法时不能留下半应用的配置。
- 撤项就是 `clear_tuning` 把三个字段一起置 `None`；`tuning_config` 只输出非 `None`
  的项，所以"没覆盖"不会被误读成"覆盖成默认值"。
- 值本身合法（正整数或 `None`）只在这一层判；`tile_w × block_y` 是否超 block 上限、
  `roc` 能否整除 OC tile、`cb` 能否放进 shared 预算，都取决于具体形状，仍由
  `plan.plan_conv_tiled` 在 launch 前判定。
- HIP 的 `__init__` 同步补上 `_conv_roc` / `_conv_cb`（AST 对账测试盯着两份实例字段）。

**验证**：计划层 6 个新用例——`roc` 覆盖重分配 block_y、`roc` 必须整除 OC tile、
`cb` 覆盖收缩 staging 深度、`cb` 超通道或超预算被拒、`roc` / `cb` 必须为正整数，
并把"每个候选值都能顺利规划"扩到三参数。后端侧覆盖 `tuning_space` 的三参数空间、
逐参数非法值、增量合并与 `None` 撤项、`tuning_config` 只报非空项，以及
`tile_w` / `roc` / `cb` 任意子集覆盖后结果与默认**逐位相同**。

**真机的一条硬限制**：本轮 RTX 2080 Ti 上没有 libcudnn（`find / -name "libcudnn.so*"`
为空），cuDNN 组的 3 个用例按设计在 `setUpClass` 跳过；回落路径由不指定
`conv_impl` 的既有卷积用例覆盖（它们走的正是 auto → 无库 → 自研核）。所以这条
路径的**代码已接、结构已测、真机数值未跑**，文档不把它记成已验证。

**门禁**：8 个用例全绿，基线仍未重采。`conv_impl` 默认 `auto` 在无 cuDNN 的机器
上等价于从前的自研路径，数字与 5.73 同一量级。

---

### 5.75　激活重算：把"整张反向图"的峰值换成"边界 + 一段"

**动机**：`ir.grad` 造出的反向图是自包含的——它会把前向**整张重放**一遍，因此
不需要常驻激活。但一次 `execute` 期间执行器不释放任何节点输出，所以反向峰值等于
"整张反向图里所有值之和"，图一深就由这个和决定，而不是由最宽的那一刻决定。

**做法**（`train/recompute.py` 的 `CheckpointPlan`）：把前向按节点顺序切成 k 段，
前向只保留每段产出的边界值；反向从最后一段往前跑，每段自带一张反向图（同样会
重放这一段的前向）。任一时刻的存活集合 = 全部边界 + 一段的中间值。

三条刻意的约束：

1. **不改 IR、不改执行器**。段就是一张普通的 `Graph`：外部值变成带 `ck_` 前缀的
   图输入，段内产出的边界值变成图输出。因此它走同一套 `Backend.execute`，换后端
   不需要新代码。前缀还有一层作用：确保外部值名不会与子图自动生成的 `v{N}` 撞名。
2. **切分是纯拓扑的**。按节点顺序等分，不猜"哪个算子便宜"；切点落在哪都不影响
   正确性——段内用到的、段外定义的值一律提成输入。没有产出的段（既不能被后续读取，
   也无法给反向图提供种子）并进邻段。
3. **梯度按段倒序累加**。段内由 `ir.grad` 按逆拓扑累加；跨段的同一份 cotangent 按
   "后段先加"并入，与单张反向图的处理顺序一致。参数梯度在最后按名字分桶累加，
   数据输入的梯度算出来但不返回。

**代价要说清楚**

- 边界值被多个段使用时，浮点加法的**结合顺序**可能与整图反向不同（末位级别）。
  链式模型（边界只有一个消费者段）可以做到逐位一致，测试盯的正是这一档。
- 某个边界值不参与损失时，段反向仍需要它的种子，这里喂零。零梯度在结果里只加进
  精确的 `0`，但若那个分支的前向里出现 `inf` / `nan`，`0 * inf` 会污染结果——这与
  整图反向"跳过该分支"不同。

**集成**：`Trainer(..., segments=k)` 走分段路径，`segments=None` 保持原来的整图反向。
分段时 `step` 先逐段前向拿 `acts`（原名 → 容器值，含边界与图输出），再按 `seeds`
（loss 喂 1.0，其余边界零）逐段反向，最后交给优化器。两条路径的接口完全一致。

**验证**：6 个用例——`segments=1` / `3` 的梯度与整图反向**逐位相等**、4 步训练后
参数逐位一致、段数超过节点数被拒、段数必须为正整数（`bool` 不算）、每个节点在段内
恰好出现一次且每个边界只由一个段产出、静态峰值用 `plan_memory` 对比（
`held_bytes + max(段 total_bytes − 边界输入字节)` 小于整图反向的 `total_bytes`）。

---

### 5.76　部署包：图、权重、清单装进一个文件

**动机**：训练产物离开进程目前是两条互不相干的路径——图走 `ir.serialize` 的 JSON，
权重走 `train.checkpoint` 的 `.npz`。现场部署时这两样必须**一起**出现，且必须能回答
"这份权重属于这张图吗、文件有没有被改过"。装进一个文件、由清单把两者的摘要绑在
一起，第三个问题才有答案。

**格式**（`deploy.py`，zip，固定三个成员，多一个少一个都拒绝）

```
manifest.json
graph.json      # ir.serialize.dumps 的原文
weights.npz     # 名字 -> 数组，只读，不允许 pickle
```

清单里有格式名、版本、IR 版本、`graph.json` 的 sha256、每个权重的 shape / dtype /
sha256，以及可选的 `meta`（任意 JSON）。

三条刻意的决定：

1. **摘要按数组字节算，不按 npz 文件算**。`np.savez` 写的是 zip，条目带时间戳，
   同一个数组两次打包的文件字节不同；按文件摘要校验会把"内容没变"判成篡改。
   按 `ascontiguousarray(...).tobytes()` 算则只取决于数值本身。测试里跨进程重打包
   再比对，盯的就是这一条。
2. **清单是格式的一部分，不是注释**。加载时依次校验格式名 / 版本 / IR 版本 / 图摘要 /
   每个权重的形状、dtype 与摘要；任何一处不符直接失败。它不是签名，挡不住"从头伪造
   一个包"——那属于发布方信任链（硬件包的 Ed25519 是同一个问题的另一处答案，5.67）。
3. **权重必须对得上图的输入**。名字不在图输入里、形状或 dtype 与声明不符，都在加载期
   拒绝——现场拿到一个"能加载但一跑就炸"的包是最坏的情况。另外设了 8 GiB 解压上限，
   部署包可以带权重，但不该是压缩炸弹。

`load_bundle` 返回一个 `Deployment`：`graph`、`weights`、`meta`，`run(inputs)` 把
权重与输入合并执行，后端缺省用参考后端。

**验证**：9 个用例——图 / 权重 / meta 往返、`run` 与进程内执行一致、篡改 `graph.json`
被拒、篡改权重摘要被拒、多余成员被拒、未知版本被拒、权重名 / 形状 / dtype 与图不符被拒、
后缀不是 `.afpkg` 被拒、**新进程读同一个包得到逐位相同的输出**（`subprocess` 起独立
解释器，只靠包本身）。

---

### 5.77　eager 前端的即时求导：`loss.backward()` 仍是反向图

**动机**：5.25 当时明确拒绝"接管求导"——只有 `backward(graph, wrt=[...])`。缺的
其实不是 autograd 的 tape，而是"写一行、立刻拿梯度"的手感：算完 loss 还要回头找出
`Value`、拼 `wrt` 列表、再去跑一张看不见的图。

**做法**（`frontend/eager.py`）

- `Tensor.backward(feed, wrt, backend)` 委托 `EagerGraph.backward`：以本张量（必须是
  图输出）为损失，求导并**写回**每个 `Tensor.grad`，同时返回 `{输入名: ndarray}`。
- `wrt` 缺省取所有 `requires_grad=True` 的图输入；一个都没有时直接报错——静默返回
  空字典会让"忘了标参数"变成"训练不动"的谜题。
- 反向图按 `(graph.revision, 求导目标)` 缓存：形状不变、图不变时只建一次；图长出新
  节点（revision 变了）即失效重建。
- 两条明确的拒绝：`loss` 不是图输出、`loss` 不是标量，都在求导前报错，不进执行器。

**为什么它不是 tape**：反向仍然是 `ir.grad.backward` 造一张自包含的静态图，再加一次
`execute`。没有逐算子即时求值、没有随用随释放的激活带、Python 控制流也不参与张量值。
换句话说，5.25 的结论只改一半：**有 `loss.backward()` 了，但它还是"建图 + 执行"，
不是 define-by-run 的 autograd**。5.25 里"不接管求导"那条已同步改写。

**验证**：6 个用例——MLP 梯度与解析式一致（`dW = Xᵀ dY`）、重复调用复用同一张反向图、
图增长后重建、非图输出被拒、非标量被拒、没有 `requires_grad` 目标时报错。

---

### 5.78　热路径上的三个 Python 常数，以及一条被误报的门禁

**动机**：5.72 / 5.73 把分组、attrs 解析与铺法缓存之后，`transformer_block` 折算后
还剩 8%。剩下的不是算法，是"每个节点都要做一遍、而且与数据无关"的准备工作。

**三处**（都在每步每节点的必经路径上）

1. **两个 f-string**。`CudaBackend.invoke` 用 `f"节点 %{node.output.name}"` 造
   context，`KernelRegistry.dispatch` 又用 `f"（{context}）"` 包一层——两条字符串
   只服务于"报错时好看"。改成 `context` 可以传节点对象，只在失败路径展开；字符串
   仍然接受，原有报错格式不变。
2. **一次 dtype 字符串解析**。`expected.dtype.np_dtype` 是 `"float32"` 这样的字符串，
   `np.asarray(x, "float32")` 每次都要解析一遍。形状变了 dtype 也不会变，所以在
   `_GroupPlan` 构建期解析成 `np.dtype` 存下来。
3. **一次多余的小分配**。`tuple(out.shape)`——cupy 与 numpy 的 `.shape` 本来就是
   tuple，外层再包一次纯属每节点一次的垃圾。

量级：13 节点 ×（2 个字符串 + 1 次解析 + 1 个 tuple）≈ 1.5～2.5 μs/节点 ≈ 20～30 μs，
占 1.2 ms 的 2% 上下。

**实测**（RTX 2080 Ti，同一号机器，热身 0.5 s 后）

| 用例 | 基线 | 本次 | 同机折算 |
|---|---|---|---|
| `matmul_256`（标尺） | 0.2350 ms | 0.2503 ms | 整机漂移 +6.5% |
| `transformer_block` | 1.1825 ms | 1.3053 ms | +3.6%（5.73 时 +8%） |

不把 4.4 个百分点全归给这三处：单次测量的 p90 是 1.38 ms，噪声比改动本身大，
只能说方向与量级吻合。

**一条被误报的门禁，比这 4 个百分点更重要**

第一次跑门禁直接失败：`transformer_block` 1.1825 → 1.6250 ms（+37.4%）。查下来
GPU 空闲在 **300 MHz（满频 2160）**——机器闲置一段时间后 SM 会深度降频，而每个
用例自带的 `warmup` 只跑几次小图，升不了频。同一份代码先压半秒，只慢 10%。

所以 `benchmarks.run` 增加了 `--warmup`（默认 0.5 s）：正式测量前用全部用例循环压
设备到稳态。它比"取三轮最好"更靠前——整轮都是冷的时候，取多少轮都没用。参考后端
没有 `cp`，热身直接跳过，不影响本地测试。

记下来的理由：不写这一段，下次换一台空闲的机器，第一个人就会去改一段正确的代码。

### 5.79　常规执行路径接进 CUDA Graph

**动机**：5.78 之后 `transformer_block` 折算后只剩 3.6%，剩下的几乎全是逐节点
kernel launch。`capture` 与 `CudaGraphPlan` 从 5.11 起就在，但只有显式调用才能用；
常规 `execute` 仍然一个节点一次下发。

**做法**：`CudaBackend(graph_mode="off" | "auto")`，默认仍 off。

- auto 时 `execute` 按 `(图 id, 结构版本, 输入签名)` 查计划，命中只 replay。
- 未命中调 `capture`：只把本次**显式传入**的输入记为可变，驻留权重留在图里当常量。
  签名包含合并后的全部输入，权重换形状时不会误用旧计划。
- 计划上限 `max_graph_plans=4`，LRU 淘汰；每份计划持有一个 arena，都是常驻显存。
- 捕获失败（图里有主机同步、cuDNN algo 查询）就置 `_graph_disabled` 回落逐节点，
  不反复重试——回落是语义正确的路径，重试只会把每一步都拖慢一倍。（5.113 已改为按图
  隔离：失败只记在这张图 + 这个结构版本名下，`graph_capture_failure(graph)` 可查原因。）
- `resident_parameters` / `release_parameters` 清空计划：旧权重写死在静态缓冲里，
  不清就会"看起来正常"地用旧参数算结果。
- HIP 继承同一份 `execute`；`HipBackend.__init__` 是手抄的，新字段由 AST 对账盯住。

**实测**（RTX 2080 Ti，热身 3 s，同一条命令内交替跑 off / auto）

| 用例 | off | auto | 变化 |
|---|---|---|---|
| `transformer_block` | 1.5914 ms | 0.7016 ms | −55.9% |
| `cnn_lenet` | 0.8452 ms | 0.3806 ms | −55.0% |
| `matmul_256` | 0.2681 ms | 0.2623 ms | −2.2% |
| `matmul_1024` | 1.6930 ms | 1.8237 ms | +7.7% |
| `matmul_batched` | 0.4729 ms | 0.5113 ms | +8.1% |
| `conv2d_3x3_pad1` | 0.3298 ms | 0.3430 ms | +4.0% |

节点多的图收益巨大：13 次 launch 变成 1 次 replay。单算子图（matmul / conv）没有
稳定收益，`matmul_batched` 甚至略慢——replay 的簿记与 cuBLAS 在捕获期固定的算法，
抵不过本来就只有一次 launch 的节省。所以默认保持 off，让使用者按图选择。

**门禁**：0.5 s 热身下 `transformer_block` 1.1825 → 1.5898（+34.4%），与 5.78 记录的
误报是同一个现象（空闲 300 MHz）；把热身加到 3 s 后同一份代码 1.1899（+0.6%），
8 个用例全绿。顺带修掉 `recheck_slow` 重测时不热身的问题：冷启动的重测只会比首测
更慢，"取更快者"就永远取不到真实值。

**边界**：auto 下同一张图只对应一组形状，形状切换会重新捕获；每份计划的静态缓冲
（mutable 输入 + 中间值）会常驻，`memory_stats` 报的仍是最近一次捕获的 arena。
训练循环里如果每步都新建 `Graph`，计划缓存命中不了——这条路径留给显式
`capture` 或图复用。

**测试**：7 个新用例（与逐节点逐位一致、按签名分开捕获、上限与 LRU、捕获失败回落、
驻留权重固定在计划里且换权重即失效、off 不捕获、非法参数）。本地 1137
（883 通过 / 254 跳过），真机 1132（1110 通过 / 22 跳过），conformance 全绿。

### 5.80　基准补一条"权重常驻"的口径

**动机**：5.79 之后 auto 图把 `transformer_block` 从 1.59 ms 压到 0.70 ms，但基准
用例每步都把全部权重从主机拷一遍。真实推理会把权重放在显存里，只传激活——这条
`resident_parameters` + 图的路径此前没有任何用例覆盖。

**做法**：`Case` 增加可选 `split`；`prepare_case` 在支持 `resident_parameters` 的
后端上先驻留权重，只把激活交给 `execute`。新增 `transformer_block_resident` 与
`cnn_lenet_resident`；参考后端不区分设备内存，仍拿完整输入。

热身在完整输入上跑（只为升频）：驻留表是后端级的一份，逐个用例驻留会把前一个
用例的输入抽走。split 用例的旧计划会在正式测量前清空，并在**不计时**的
`warmup` 次数里按新口径重新捕获。

**实测**（RTX 2080 Ti，热身 3 s，同一会话）

| 用例 | off | auto | 变化 |
|---|---|---|---|
| `transformer_block` | 1.3078 ms | 0.6272 ms | −52.0% |
| `transformer_block_resident` | 1.0206 ms | 0.2179 ms | −78.6% |
| `cnn_lenet` | 0.6869 ms | 0.3594 ms | −47.7% |
| `cnn_lenet_resident` | 0.5758 ms | 0.1264 ms | −78.0% |

只做权重常驻、仍是逐节点下发，`transformer_block` 就省了 22%（1.3078 → 1.0206）：
那 ~0.29 ms 是每步都在付的 H2D。两项叠加后，从基线 1.1825 ms 到 0.2179 ms
（−81.6%）；`cnn_lenet` 从 0.6735 ms 到 0.1264 ms（−81.2%）。

单算子图这轮在 auto 下没有退化：`matmul_256` / `matmul_1024` / `matmul_batched`
是 0.2631 / 1.7486 / 0.4793，off 是 0.2560 / 1.7605 / 0.4886，落在同一噪声带；
`conv2d_3x3` 反而快 9%。5.79 记的 +8% 更像整机漂移，而不是 replay 的固定成本。

**门禁**：同一台机器的 `transformer_block` 在 1.30～1.55 ms 之间漂（`nvidia-smi`
的 SM 时钟在 1350 MHz 与满频 2160 之间跳，没有 throttle），3 s 与 10 s 热身都
不足以稳定复现基线 1.1825，本轮门禁因此为红。off 路径没有代码变化（5.79 只加了
一次布尔判断），同一会话内的 A/B 数字一致；基线不重采。

**测试**：2 个新基准用例 + 2 个新用例（拆分正确、假后端上驻留被调用、参考后端
保持完整输入）。本地 1139（885 通过 / 254 跳过），真机 1134（1112 通过 /
22 跳过），conformance 全绿。

---

### 5.81　自动图按节点数开闸

**动机**：5.79 的 `graph_mode="auto"` 对任何图都尝试捕获，包括只有一个节点的
matmul / conv。单算子图本来就只有一次 launch，捕获省不下派发，反而要多付
replay 的簿记，并把 cuBLAS / cuDNN 的算法固定成捕获时选中的那个——5.79 实测里
唯一"变慢"的几个用例正是这一类（`matmul_batched` +8%）。

**做法**：`CudaBackend(min_graph_nodes=2)`。`execute` 在 auto 分支先看
`len(graph.nodes)`，低于阈值直接走逐节点，不查计划也不捕获。阈值可调，
`min_graph_nodes=1` 恢复 5.79 的"单算子也捕获"；HIP 侧同步构造参数与字段，
由 AST 对账盯住。

**实测**（RTX 2080 Ti，热身 3 s，每档 2 轮取最好；off 先跑）

| 用例 | off | auto | 变化 |
|---|---|---|---|
| `matmul_256` | 0.2909 ms | 0.2890 ms | −0.7% |
| `matmul_1024` | 1.7686 ms | 1.9238 ms | +8.8% |
| `matmul_batched` | 0.4665 ms | 0.4657 ms | −0.2% |
| `conv2d_3x3_pad1` | 0.3346 ms | 0.3418 ms | +2.2% |
| `conv2d_1x1` | 0.3530 ms | 0.3615 ms | +2.4% |
| `maxpool2x2` | 3.0524 ms | 3.0106 ms | −1.4% |
| `transformer_block` | 1.4543 ms | 0.6812 ms | −53.2% |
| `transformer_block_resident` | 1.1343 ms | 0.2054 ms | −81.9% |
| `cnn_lenet` | 0.7800 ms | 0.3843 ms | −50.7% |
| `cnn_lenet_resident` | 0.6259 ms | 0.1277 ms | −79.6% |

`matmul_1024` 那行 +8.8% 不是捕获造成的：单节点图在 auto 下根本不进
`_graph_plan`，执行路径与 off 完全相同。把顺序反过来跑（auto 先、off 后）：
`matmul_1024` 变成 1.7045 / 1.7261（auto 反而快 1.3%），`matmul_batched`
0.4885 / 0.4746，`transformer_block` 0.7386 / 1.2991。同一份代码换了个先后
就翻转，说明这是整机漂移而不是 replay 成本——5.79 记的 +8% 同理。

多节点图的收益没有变化；单算子图现在与 off 走同一条路径，差异只来自测量噪声。

**测试**：1 个新用例（单节点图不捕获、`min_graph_nodes=1` 可恢复），另有
`min_graph_nodes=0` 的构造期拒绝。本地 1140（885 通过 / 255 跳过），真机
1135（1113 通过 / 22 跳过）。

---

### 5.82　CUDA 训练闭环的门禁，与换机才暴露的 cuDNN 指针 bug

换到一台装了 pip 版 `nvidia-cudnn-cu12` 的机器（RTX 2080 Ti、cupy 14.2.0、
Python 3.10.12）后，第一次跑全量就红了 4 个用例，全部在
`TestCudaConvCudnn`：

```
File "aetherflow/backends/cuda/cudnn.py", line 273, in convolution
    xd, wd, cd, yd, bd = (vp(d) for d in descs)
TypeError: cannot be converted to pointer
```

**根因**：`_plan()` 返回的描述符本来就是 `ctypes.c_void_p`，`vp(d)` 又套了
一层 `c_void_p()`；后者在 Python 3.9+ 只接受 int 或 None，直接抛
`TypeError`。这段代码此前从未在真机上执行过——旧机器找不到 libcudnn，整组
cuDNN 用例按设计跳过，于是"入库即带病"也没人发现。修复是一行：

```python
xd, wd, cd, yd, bd = descs
```

**教训**：被 skip 掩盖的路径等于没测。缺库时的跳过策略本身是对的（不能因为
没有 cuDNN 就让整个套件红），但"加载得到"这条路必须至少在一台真机上跑过。

**训练闭环**：`tests/test_cuda_train.py` 补了 3 个用例，回答"能不能在 GPU 上
训练模型"，而不只是"每个反向算子算得对"：

| 用例 | 判据 |
|---|---|
| `test_linear_regression_converges` | `Trainer + CudaBackend` 200 步，loss < 1e-4，w/b 回到真值 |
| `test_cnn_step_matches_reference` | 同一份初值走一步，loss 与**全部参数**和参考后端一致（conv 输入/权重反向、池化反向、reshape 反向都走了一遍） |
| `test_cnn_loss_decreases_over_a_few_steps` | 30 步内损失单调下降，没有 NaN |

此前训练收敛测试全部跑在 `ReferenceBackend` 上，CUDA 只有逐算子对拍；
现在「CUDA 能训练」是一条真机门禁，不再只是推断。真机 3 个用例 3.37 s。

**当前口径**：本地 1143（885 通过 / 258 跳过）；CUDA 真机 1143（1090 通过 /
53 跳过——新机未装 torch 34 个、ROCm 17 个、有硬件时测不到的回退路径 2 个）。

### 5.83　训练 step 进图，与视图内存的生命周期

`graph_mode="auto"` 到 5.82 只覆盖推理：`Trainer.step` 每步调两次
`backend.execute`（前向一张图、反向一张图），本该各吃一张捕获计划。补一条
门禁用例就发现两边对不上——前向一致，反向的梯度完全不同，训练在 auto 下
先降后爆：

```
off : 10.06 → 8.24 → 6.82 → 5.70 → 4.81 → …
auto: 10.06 → 7.75 → 8.80 → 11.20 → 15.40 → … → 679.09
```

**定位**：三条对照把范围从"图捕获"收窄到"捕获期的显存回收"。

| 对照 | 结果 |
|---|---|
| `capture(..., reuse_memory=False)`（用 cupy 池，不按地址回收） | 梯度正确 |
| `run_device(..., release=False)`（中间值不回收） | 梯度正确 |
| `run_device(..., release=True)` | 梯度错误，与 auto 训练一致 |

**根因**：`reshape`、恒等 `transpose` 返回的是共享底层缓冲的视图，
`ndarray.base` 指向源数组、`data.ptr` 与源数组相同。执行器按"最后一次被读"
回收的是**源数组**，于是把地址还给了 arena；视图这时还在环境里等着被读，
下一块同尺寸分配就拿到同一段地址，replay 读到的是别人写进去的数据。
反向图里每次归约都要 `transpose + reshape`（`grad_rules.sum_axes`），所以
前几步还看不出来，越往后越多视图同时挂着，误差迅速放大。

第一版修法是在 `dispose` 里跳过视图——不够，被提前释放的是**源数组**，不是
视图（真机确认：`value.base` 是 `ndarray`，`data.ptr` 与源数组逐一相等）。
第二版把源地址钉到捕获结束：数值对了，但 `transformer_block` 的捕获峰值
6656 → 20480 B，直接顶穿静态预算门禁（≤ 2 倍）——attention 里 reshape 太密，
钉住等于放弃这些小块的复用。

**最终做法**：arena 里给视图记账，回收点仍然只由执行器给出。

- 创建视图时（`_op_reshape`、恒等 `_op_transpose`）`Arena.alias(源地址)` +1
- 视图被回收时 `Arena.unalias` -1
- 源数组被回收时若计数 > 0，只记成"待回收"；计数归零才真正还块

复用窗口于是只缩短到"最后一个视图死掉"为止，而不是整段捕获。实测
`transformer_block` 捕获峰值仍是 8192 B（静态预算 6656 B），与 5.80 的口径
完全一致：正确性没有拿显存来换。

**回归用例**：

| 用例 | 判据 |
|---|---|
| `test_reshape_source_is_not_reused_before_view_dies` | reshape 之后用同尺寸分配顶掉源地址；去掉记账时 maxdiff 6.9（算成 `2k`），加上后 0 |
| `test_views_are_not_freed_early` | reshape + transpose 视图链进图后 auto 与 off 数值一致 |
| `test_auto_graph_training_matches_eager` | 同 seed 训练 10 步，off/auto 的每步损失与全部参数一致，且前向、反向各一条捕获计划 |

**当前口径**：本地 1146（883 通过 / 261 跳过；另有 2 个 torch 导出用例在本机
torch 2.13 上因 `aten.relu_` 未降级而报错，与本轮改动无关）；CUDA 真机 1146
（1093 通过 / 53 跳过）。

### 5.84　参数常驻显存：训练步不再搬权重

5.83 让训练 step 吃上了 CUDA Graph，但省下来的只是"每步两次 replay"：`Trainer`
仍然把 `store` 里的 numpy 权重喂进去（每次 run 都要 H2D），梯度又要 D2H 回主机
交给优化器，优化器改完再等下一次 H2D。权重、梯度、优化器状态这三类数据的往返
一次都没省——而它们才是训练每步真正反复搬运的大头。

**做法**：给 `capture` 加 `shared_buffers`——直接采用调用方给的显存数组当静态
输入缓冲，而不是另申请一块再拷一遍。`Trainer(device_state=True)` 在第一次
`step` 时做四件事：

1. 把参数上传成一组 cupy 数组，`optimizer.rebind` 换成这张设备表；
2. 用**同一组**缓冲捕获前向图，再捕获反向图（两张计划共享权重地址）；
3. 反向图的种子（损失收 1、其余输出收零）也一并固化在显存里；
4. 数据缓冲按第一次 step 的形状分配，此后形状固定。

之后每步只剩：本批数据一次 H2D → 前向 replay → 损失标量一次 D2H → 反向 replay
→ 优化器在显存上原地更新。权重与梯度一步都不回主机；捕获计划也不会因为"权重
变了"作废（`resident_parameters()` 那条"清计划"的路径完全不参与）。

三个前提必须同时成立，缺一个就会算错，所以三条都在用例里直接断言：

| 前提 | 由谁保证 |
|---|---|
| 优化器改的就是图里写死的那块地址 | `plan._buffers["w"] is optimizer.params["w"]`（用例断言）；计划不持有缓冲所有权，`Trainer` 负责保活 |
| replay 不会把共享缓冲再拷一遍 | `replay` 里 `arr is buf` 时跳过 |
| 优化器的更新对下一次 replay 可见 | 捕获流是 non-blocking，与默认流没有隐式同步：`replay` 先用事件等一次当前流 |

**优化器设备化**：`step` 原来一句 `np.asarray(grads[name], dtype=...)`，对 cupy
数组直接抛 `TypeError`（cupy 拒绝隐式回主机）。改成按参数所属模块分派
（`_array_module`）：梯度原样使用、槽位 `zeros_like`、`sqrt` 都跟着参数走。
存档格式不变——`state_dict` 仍然导出 numpy，`load_state_dict` 按参数所在设备
还原。

**为什么不做"每步重传 + 重捕获"**：那条路也能算对，但每步重捕获一张图是毫秒级
开销，等于把 CUDA Graph 的收益全部吐回去。

**回归用例**（真机 RTX 2080 Ti）：

| 用例 | 判据 |
|---|---|
| `test_device_state_matches_host` | 同 seed 10 步，SGD 的每步损失与全部参数与主机态一致；`optimizer.params` 是 cupy 数组、`store` 在 `sync_to_host` 之前保持初值；前向与反向的 `_buffers["w"]` 与优化器手里是同一个对象 |
| `test_device_state_adam_matches_host` | Adam 的 m/v 同样留在显存，5 步轨迹与主机态一致 |
| `test_rebind_switches_target_arrays`（本地） | 换表后更新只落在新表；名字或形状不一致直接拒绝 |

**当前口径**：5.91 之后本地 1187（919 通过 / 268 跳过）；CUDA 真机
1187（1133 通过 / 54 跳过，RTX 2080 Ti + Python 3.10 + cupy 14.2，`test_cuda_train`
7 个用例全绿，含下面两条设备态用例与新增的 `test_device_state_with_segments_matches_host`）。

### 5.85　训练辅助工具跟着参数留在显存

5.84 让权重、梯度、优化器状态都不回主机，但"梯度算完到更新之前"这一段还站着
一批按 numpy 写死的工具：`clip_global_norm` 里 `np.linalg.norm`、`GradientAccumulator`
里 `np.zeros_like`、`EMA` 里 `np.asarray`、`checkpoint.save` 里 `np.array`。设备态
下梯度是 cupy 数组，这些调用不是"慢"，而是**直接抛 TypeError**（cupy 拒绝隐式
回主机）。用户唯一的绕法是"先把梯度同步回来再裁剪"，等于把 5.84 省下的往返
原样还回去，还多了一次 D2H。

**做法**：新增 `train/_arrays.py`，把"这份数组归谁管"变成一个可回答的问题——

| 函数 | 语义 |
|---|---|
| `array_module(arr)` | 按 `type(arr).__module__` 的顶层包判断，cupy 数组返回 cupy，其余返回 numpy |
| `to_host(arr)` | 设备数组走 `cupy.asnumpy`，主机数组原样返回 |
| `host_scalar(value)` | 取标量的唯一出口，设备态下这一次同步无法避免 |

四处的改法都是"运算留在数组所在的设备上，只把结论搬回主机"：

- `global_norm`：平方和用 `xp.add.reduce` 在设备上累加，最后 `host_scalar` 一次
  拿范数；`clip_global_norm` 用 `array_module(grad).array(grad, copy=True)` 复制再缩放，
  不再假设 `np.ndarray`。
- `GradientAccumulator`：第一份微批决定 `_sum` 的数组模块（同一轮训练里梯度
  不会换设备），之后 `_sum += grads` 全是设备内加法。窗口满时 `flush` 出来的
  就是设备数组，可以直接交给 `optimizer.step`。
- `EMA`：影子按参数所在的模块分配，`update` 用 `xp.multiply`/`add` 原地算；
  `copy_to` 同设备直接 `astype`，跨设备才走 `to_host`；`state_dict` 仍然导出 numpy
  （存档里放主机数组），`load_state_dict` 按影子数组的模块还原。
- `checkpoint`：`save` 一律 `to_host`——显存里的参数先 `sync_to_host()` 再存档；
  `load` 按目标数组的模块 `asarray`，于是恢复进设备表不需要额外分支。
- `GradScaler.unscale`：同样按数组模块分派；`finite` 用 `&` 合并成一个布尔数组，
  最后 `bool(to_host(finite))`，一步只同步一次。

**回归用例**：`test_train_utils` 的 26 个辅助工具用例（本地）；
`test_grad_hook_can_skip_and_clip` 与 `test_grad_hook_accumulates_micro_batches`
把"钩子 = 裁剪 / 累积"这条典型组合钉住（两个 4 样本微批 == 一个 8 样本批次）。

### 5.86　数据侧的 pinned 暂存与常驻输入

5.84 之后设备态每步只剩一次 H2D（本批数据）与一次 D2H（损失标量）。但
`buf.set(pageable_array)` 对可分页内存会**退化成同步拷贝**：主机必须等这次 H2D
真正落地才能继续，捕获省下来的下发时间又被这一次等待吃掉。

**做法**（`CudaGraphPlan.replay` + `CudaBackend.pinned_array`）：

1. `pinned_array(key, arr, dtype)` 按 `(key, 形状, dtype)` 复用一块
   `cupy.cuda.alloc_pinned_memory`，用 `np.frombuffer` 包成 numpy 视图，
   `np.copyto(view, arr, casting="unsafe")` 拷进去；
2. 再用 `buf.set(view, stream=self._stream)` 做真正的异步 H2D；
3. `value is buf`（共享缓冲）时**完全跳过**拷贝——设备态训练里参数、边界值、
   种子都是这样传的；
4. 捕获流是 non-blocking，与默认流之间没有隐式同步，所以进流前先
   `previous = get_current_stream()`，再在 `with self._stream:` 里
   `wait_event(previous.record())`。

第 4 条有个容易写错的地方：`previous` 必须在 `with` **之前**取。写在里面拿到的是
捕获流本身，事件等的是它自己，等于没等——第一版就是这么错的，真机上表现为
"优化器更新完了，replay 还读到上一轮的权重"。

**为什么 pinned 缓冲要按名字复用**：训练里的输入种类固定（少量参数 + 少量数据
输入），按 `(名字, 形状, dtype)` 缓存就不会每次 `step` 都申请/释放一块固定内存。
安全前提是"上一次 H2D 已经同步过"——`run` / `run_device` 都会 `stream.synchronize()`，
所以缓冲区不会在传输途中被覆写。

### 5.87　HIP 对齐同一套运行期语义

HIP 后端此前只支持"捕获 + 从主机数组跑 replay"。设备态需要的三件事它都缺：
`shared_buffers`、`stream`、`run_device`。补齐的方式是**与 CUDA 逐条同义**，不是
另写一套：

| 能力 | HIP 侧实现 |
|---|---|
| `shared_buffers` | 直接采用调用方给的 `DeviceArray` 当静态输入缓冲；形状/dtype 不符即拒绝 |
| `stream` | `runtime.new_stream()`（非默认流）；段与段共用同一条流时 replay 天然有序 |
| `run_device` | replay + `stream.synchronize()`，返回 `DeviceArray` 输出，不做 D2H |
| 自拷贝跳过 | `value is buf` 时 `continue`；否则 `to_array` + dtype cast + D2D 赋值 |

**两处明确的不做**：

- **不接 pinned**：`hip/runtime.py` 没有 host 侧固定内存的接口，硬凑一条"看起来
  异步"的路只会把同步点藏起来。`HipBackend.__init__` 里仍然建出 `_pinned` 字段，
  那是为了让"CUDA 有的实例字段 HIP 也有"这条 AST 对账继续成立（否则
  `pinned_array` 一旦被继承调用就是 `AttributeError`）。
- **不支持设备态训练**：HIP 的容器是 ctypes 直连 hip 的 `DeviceArray`，不属于
  numpy 也不属于 cupy，优化器的原地更新没有可用的分派分支。`Trainer._upload_params`
  用数组模块探测把这种情况拦在第一次 `step`，报一句"请用 `device_state=False`"，
  而不是让它在一句 `TypeError: unsupported operand` 里失败。

> 已由 5.156 补齐：两条都拆掉了。pinned 走 `hipHostMalloc` / `hipHostFree`，
> `HipBackend.pinned_array` 按 `(名字, 形状, dtype)` 复用固定内存，`replay` 用
> 禁用计时的事件把捕获流挂到当前流上；`DeviceArray` 补齐优化器需要的运算符
> 面，`array_module` 认出 `aetherflow.backends.hip.*` 并交给 `hip.runtime`，
> `Trainer(device_state=True)` 的 SGD / Adam 轨迹与主机态一致。

### 5.88　设备态叠加激活重算

5.84 的设备态是"整图捕获"：前向一张、反向一张。5.75 的激活重算则是"逐段执行"。
两者叠加要回答的问题是：**分段之后，边界值还要不要回主机？** 不回，就意味着
每段各捕获一张前向图与一张反向图，段与段之间靠显存里的边界值衔接。

**做法**（`Trainer._prepare_segments` / `_step_device_segments`）：

- 每段的前向图与反向图都用 `capture(stream=...)` 捕获，**共用同一条非阻塞流**：
  k 段 replay 天然有序，不必建 k-1 条跨流事件依赖；
- 段图输入统一带 `ck_` 前缀，映射到同一组缓冲：参数 → `dev_params`、数据 →
  `dev_data`、边界 → `dev_acts`；段反向图另外把 `d<local>` 映射到 `dev_cots`；
- 每步顺序：逐段前向（段产出 D2D 拷进 `dev_acts`）→ 损失一次 D2H →
  清零 `dev_grads` 与 `dev_cots`，损失的 cotangent 置 1 → 倒序逐段反向，
  `d<param>` 累加进 `dev_grads`、`d<边界>` 累加进 `dev_cots`、`d<数据>` 丢掉 →
  `grad_hook` → 优化器原地更新。

**一个必须说清楚的坑：边界值与 cotangent 不能共用缓冲。** 段反向要把这一段的前向
重放一遍，重放读的正是边界值；而 cotangent 是"后段先加"出来的。任何"把梯度写回
边界缓冲"的省法都会让重放读到梯度——注意清零时机也在重放之后，救不回来。所以
这里老老实实开两组缓冲，多花一份"边界"的显存，换掉一类只在真机上才暴露的错误。

数据输入只在**第一个用到它的段**里以主机数组传入（走 5.86 的 pinned 异步拷贝），
后面再读同一个输入的段直接拿已经就位的缓冲。

**口径要诚实**：每段的捕获计划各持自己的 arena（`Arena` 按计划建，地址写死在图里，
跨计划共享会让两张图的中间值互相踩）。所以"设备态 + 分段"的显存总量约等于各段
峰值之和，它买到的是"单张捕获图更小、更容易塞进驱动与碎片限制"，以及段间传递
完全不经过主机。想要峰值真的下降，仍然用主机态 + `segments`（5.75）——那是靠
"反向只重放一段、中间值随段释放"换来的。

**回归用例**：

| 用例 | 判据 |
|---|---|
| `test_matches_host_segments`（本地，numpy 替身后端） | 2 段 3 步，损失与全部参数与主机态分段训练一致（rtol 1e-6）；每段各有前后向两张计划；边界与 cotangent 是两组不同对象 |
| `test_loss_is_read_from_the_boundary_buffer`（本地） | 损失只从边界缓冲 D2H 一次，读到的就是返回值 |
| `test_device_state_with_segments_matches_host`（CUDA 真机 ✅） | CNN、2 段、4 步，轨迹与主机态分段训练一致 |

真机上这一条同时把 5.86 的两处也压住了：每步的数据上传走 pinned 异步拷贝，
段间的 cotangent 累加发生在默认流、而下一段 replay 在捕获流上，只有事件依赖
真的生效，4 步后的参数才会与主机态对齐。

本地那两个用例用的是 `_FakeDeviceBackend` / `_FakePlan`：把 `capture` 换成"图读固定的
那组 numpy 缓冲"，因此调度逻辑（清零、累加、段序、边界与 cotangent 的分工）在没有
GPU 的机器上也能验收——这是本轮唯一能在本地验证设备态的手段。

### 5.89　设备管理器与分层分配器的契约上提

大纲里的"设备管理器接口"和"分层内存分配器接口"挂了很久，但契约层之前只覆盖了
**问得出来**那一半：`DeviceContract` 只有 `device_info()`，`MemoryContract` 只有
`memory_stats()`。后端内部明明有流、有分配器，调用方却只能绕过契约直接摸插件字段。
这一轮把两个真实实现（CUDA / HIP）都已经在用、且语义一致的那部分提上来。

**上提的三处**（`backends/contracts.py`）：

| 上提物 | 形状 | 为什么是它 |
|---|---|---|
| `StreamContract.new_stream()` | 交出一条非阻塞流 | 捕获必须发生在非阻塞流上（默认流上捕获会被驱动拒绝）；CUDA 与 HIP 的 `new_stream` 在 5.84 / 5.87 就已存在，语义一致 |
| `MemoryContract.allocator()` | 交出一个**新**分配器 | 调用方要的从来不是"一串统计数字"，而是"一个能记账、能分配的对象"；每次捕获必须换新的——地址写死在图里 |
| `AllocatorContract` | `malloc` / `free` / `stats` / `reset_stats` / `close` | 正好是 `cuda/memory.Arena` 与 `hip/memory.Arena` 共同的那五个方法 |

**`new_stream` 不塞进 `DeviceContract`。** 试过，会坏事：`implemented_contracts()`
是按协议**整体**判断的，`DeviceContract` 一旦多出 `new_stream`，那些只报了设备信息的
后端会悄无声息地从报告里掉出去——它没有报错，只是不再被算作实现了设备契约。
测试里的 `_WithDevice` 替身就是这个反例。拆开之后语义反而更准：设备信息是启动时
问一次，流是每次捕获、每步训练都要开的东西，生命周期本来就不同——这正是模块头
"生命周期不同就拆协议"那条规则的应用。

**`alias` / `unalias` / `recycle` 不进契约。** 三者在两个实现里不同形：`alias` 是
cupy 零拷贝视图特有的记账，HIP 的 `DeviceArray` 没有这一层；`recycle` 只有 HIP 需要
（CUDA 侧预热轮的块由池自己回收），写进契约等于要求 CUDA 实现一个永远不用的方法。
契约只收"两个实现都有、且调用方真的会调"的部分。

**`AllocatorContract` 不进 `CONTRACTS` 表。** 那张表回答的是"这个后端能做什么"，
而分配器是后端**交出来**的对象，不是后端自己。所以
`implemented_contracts()` 仍然是三项 `('device', 'memory', 'stream')`，
分配器单独用 `supports(backend.allocator(), AllocatorContract)` 查。

**调用点同改**：`CudaBackend.capture` 与 `HipBackend.capture` 都改成 `self.allocator()`，
"捕获期要一个干净的分配器"这件事在两个后端里收敛成同一个表达式。

**包装器必须跟着转发**（`runtime/telemetry.py`）。`InstrumentedBackend` 是"透传 + 计时"
的一层，之前只绑定 `device_info` / `memory_stats`；契约一变宽，被包过的后端就在
`implemented_contracts()` 里少两项。这个缺陷是 5.89 的第一次真机全量跑出来的
（`test_telemetry_wraps_cuda_and_reports_device`：`Items in the second set but not the
first: 'stream', 'memory'`），本地跑不到——本地没有可包装的 CUDA 后端。修法是
**按存在性转发**而不是无条件转发：包装器不该替内层承诺它没有的能力。

**回归用例**：

| 用例 | 判据 |
|---|---|
| `test_stream_is_its_own_contract`（本地） | 只有 `new_stream` 的后端实现 `StreamContract`、不实现 `DeviceContract`；反之亦然 |
| `test_allocator_is_a_value_contract`（本地） | `_FakeArena` 满足 `AllocatorContract` 而不满足 `MemoryContract`；且 `AllocatorContract` 不在 `CONTRACTS` 的取值里 |
| `test_device_contract`（HIP 真机） | 五项契约都在（5.163 起含 codegen / tuning），且 `supports(backend.allocator(), AllocatorContract)` |
| `test_telemetry_wraps_cuda_and_reports_device`（CUDA 真机 ✅） | 包装后契约集合不缩水 |

CUDA 真机实测（5.163 之前的口径；该版之后 `codegen` / `tuning` 也命中，结果为五项）：
`implemented_contracts(CudaBackend()) == ('device', 'memory', 'stream')`，
`device_info()` 与旧基线逐字一致（RTX 2080 Ti / runtime 12090 / 11539054592 B）——
上提契约没有动到任何设备侧行为。

**性能基线加机器指纹**（同属"设备侧事实必须可核对"，一并落在这一节）：

重采时立刻撞上一件怪事：8 个用例**整体**慢 1.9～2.7 倍，而且纯 launch 开销的
`matmul_256` 最明显（-50.3%），`matmul_1024` -59.5%、`transformer_block` -63.3%。
`nvidia-smi` 显示同一张 2080 Ti、SM 空载 300 MHz、没有别的进程；宿主是
Xeon E5-2678 v3（2014 年，2.5 GHz），不是上次采基线的那台。也就是说这不是代码回归，
是换了一台宿主更弱的实例——而"整体齐平变慢"与"某个 kernel 回归"在数字上完全同形。

**决定不覆盖基线。** 把这 2 倍写进 `baseline_cuda.json`，等于把门禁一次性放宽一倍，
之后真有 1.5 倍的回归也发现不了。基线只在"同一台机器、同一份代码"下才有意义。

取而代之的是给报告加**机器指纹**（`benchmarks/run.py` 的 `_machine_label`）：`machine`
字段记 host / cpu / cores / python；`--check` 发现指纹不同时先打印一句
"基线与当前不是同一台机器（…），比值只作参考"。**只提示、不判失败**：CI runner 改名、
换一批同型号实例都不该让门禁变红，真正要防的是"跨机器比较却当成回归"。基线里没有
`machine` 字段（5.89 之前采的）时 `machine_note` 返回空串——**无法判断就不要编一个结论**。

3 个新用例把这条钉住：同机静默、跨宿主 CPU 时报出差异、无指纹时静默。

### 5.90　自动混合精度：把"哪一层留 f32"写成表

5.28 的 f16 是**整图 f16**，5.33 当时明确记过"不做 autocast，哪些层留 f32 由调用方
在图里显式 `cast`"。这条限制的真实代价是：`matmul` 想吃 f16 的吞吐，`softmax` /
归约 / 归一化必须留 f32，手写 cast 要精确到每一条边——层数一多就没人维护得住，
而且漏一处只会表现为"训练收敛慢一点"，查不出来。

**做法**（`passes/autocast.py` 的 `Autocast`）：一个图重写 pass，两条表加一条兜底。

| 类别 | 算子 | 工作精度 |
|---|---|---|
| `LOW_PRECISION` | `matmul` / `matmul_batched` / `conv2d` | 降到 `f16` |
| `FORCE_F32` | `softmax` / `layernorm` / `rmsnorm` / `batch_norm` / `batch_norm_stats` / `reduce_sum` / `reduce_mean` / `reduce_max` | 拉回 `f32` |
| 其余 | 逐元素、reshape、池化…… | **最宽输入**：有 f32 输入就 f32，全是 f16 才 f16 |

第三条不是省略，正是残差连接要的行为：`x + matmul(x, w)` 里 matmul 在 f16 里算，
加法因为有 f32 的 `x` 而回到 f32。少了它，f16 的 matmul 结果会直接撞上一个 f32
输入，图连"同一节点的输入 dtype 一致"都不满足。

**四个刻意的取舍**：

1. **只降 f32，不碰 f64 与整数。** f64 是调用方明确要求的高精度；整数（argmax 的
   下标、dropout 掩码）根本不该参与浮点提升。两者原样搬运。
2. **`cast` 与 `constant` 不参与对齐。** 否则 `cast` 的输入会被按"工作精度"再包
   一层 cast，一路套下去；`constant` 没有输入，也无从对齐。
3. **图输出一律恢复成原图 dtype。** 打开 autocast 不改变对外接口：交进去、交出来
   的都是同一份 f32 张量，只有中间值变窄。这同时带来**幂等**——对已转换过的图再跑
   一次，一个 cast 都不会新增，有用例逐节点钉住。
4. **不进默认管线。** 它改变数值语义，和融合一样必须显式开启；`default_pipeline()`
   仍然是"无硬件假设"的四步。

**反向不需要特殊处理**：反向图是从已转换的前向图展开的，`cast` 的 VJP 早就在
`grad_rules` 里（反着 cast 回去），所以参数梯度天然落回 f32——这正是"主权重 f32 +
计算 f16"的意思，不像整图 f16 那样要调用方每步手工 `astype`。

**回归用例**（`tests/test_autocast.py`，20 个）：

| 用例 | 判据 |
|---|---|
| `test_matmul_inputs_become_f16` | matmul 的两个输入都挂着 cast，输出 f16 |
| `test_softmax_stays_f32` | 加法已把精度升回 f32，softmax 不必再插 cast |
| `test_cast_inserted_where_f16_meets_f32_only_op` | f16 直接撞上强制 f32 的算子时，边界插一个升精度 cast |
| `test_add_promotes_to_widest_input` | 残差加法留在 f32，matmul 的 f16 输出被提升 |
| `test_graph_interface_is_unchanged` | 输入名、输入类型、输出 dtype 都与原图一致 |
| `test_verifier_accepts_result` / `test_pipeline_keeps_graph_valid` | 重写后不违反 verifier；连续两次过 `PassManager` 也成立 |
| `test_is_idempotent` | 二次转换的节点序列完全不变 |
| `test_f64_is_left_alone` / `test_integer_graph_is_left_alone` | f64 与整数图原样搬运 |
| `test_matches_f32_within_f16_tolerance` | 参考后端上 rtol 1e-2 / atol 1e-3，输出 dtype 仍是 f32 |
| `test_parameter_gradient_returns_to_f32` | `dw` 是 f32，且与全 f32 图的梯度在 5e-2 内一致 |
| `test_forward_matches_f32` / `test_backward_returns_f32_gradients`（CUDA 真机 ✅） | 设备上能跑通，前向数值与梯度 dtype 都对 |
| `test_f32_to_f16_rounds_on_store` / `test_f16_to_f32_reads_through_ptx` / `test_same_width_is_rejected` | cast 源码里两端 helper 没接反：读跟源 dtype、写跟目标 dtype（无需 GPU） |
| `test_hip_cast_uses_clang_helpers` | HIP 侧复用同一份 L2 描述，但 f16 helper 是 `_Float16` 版，不含 PTX 的 `cvt` |
| `test_non_float_target` / `test_f64_target` | 目标 dtype 非法时在构造期拒绝 |

**真机才暴露的缺陷：`cast` 到 f16 从来没被走过。** 上面两个 CUDA 用例第一次上
RTX 2080 Ti 就红了，报 `ptxas fatal: Unresolved extern function '_ZN6__halfC1Ef'`。
根因不在 autocast：5.33 的 f16 用例全都**直接声明 f16 输入**，整条路径上一次都
没产生过 `cast` 节点，于是 `CudaBackend.cast` 一直走的是 cupy 的 `astype`——它的
f16 内核要 `__half` 的构造函数，而 NVRTC 只拿到声明、链接不到定义。

修法是把 `cast` 也收回自研内核：`lower/kernels.py` 的 `ew_cast(out_type)` 描述
"读按当前 dtype、写按目标 dtype"，`codegen.cast_fn` 按**源** dtype 渲染（决定用
`af_ld_f` 还是 `af_ld_h`），写回目标为 f16 时套一层 `af_st_h`（PTX `cvt.rn.f16.f32`），
目标为 f32 时裸赋值——再套 `af_st_f` 只会把刚从 f16 升上来的值舍回去。方言另起
一份 `CAST_DIALECT`（两套 helper 的并集），只给这一条内核用，不去污染其它内核的
源码：5.42 起有"生产源码与手写模板逐 token 相等"的对照用例盯着每一个字。

**HIP 走的是同一条路。** `HipBackend` 继承 `CudaBackend`，不改的话 HIP 上会拿到
PTX 版的 f16 helper，clang 直接报 `invalid output constraint`。因此 `hip/codegen.py`
也有一个 `cast_fn`：L2 描述与判定逻辑逐字共用，只把方言换成 `hip/dialect.py` 的
`CAST_DIALECT`（f16 那套是 `_Float16` + `__builtin_bit_cast`）。HIP 真机本轮没有
复跑，这一条只有源码级断言（`test_hip_cast_uses_clang_helpers`）。

**5.90 实测口径**：本地 1181（914 通过 / 267 跳过，`test_autocast` 18 过 2 跳）；
CUDA 真机 1181（1127 通过 / 54 跳过，RTX 2080 Ti + Python 3.10 + cupy 14.2），
`tests.test_autocast` 20 个全绿。顺带修掉本地 torch 2.13 上的两个既有失败：新版本
`torch.export` 会把 `relu(inplace=True)` / `residual += x` 原样保留成 `aten.relu_` /
`aten.add_`，`torch_export.py` 早先只认函数式名字。导出图是 SSA 的，原地写回对别的
读者不可见，所以这两条按函数式版本降级即可（新增 1 个用例钉住）。

### 5.91　混精接进训练循环，并量出它现在不划算

5.90 只交付了一个 pass：用户得自己 `Autocast().run(graph)`，还得自己保证反向图、
捕获计划、优化器看到的是同一张图。这条缝补起来的代价很小——`Trainer` 在构造期
最前面改一次图，之后所有下游（`backward`、`CheckpointPlan`、两张捕获计划）看到的
就是同一张前向图，不存在"前向 f16、反向 f32"的半套状态。

**API**：`Trainer(..., mixed_precision="f16")`。参数仍是 f32（图输入与输出都保持
原 dtype），所以 `SGD` / `Adam`、`ParameterStore`、`checkpoint`、部署包全都不用改；
非法值（`"f64"` / `"int32"`）在构造期由 `Autocast` 拒绝。开关默认仍是 `None`：
它改变数值语义，与融合同一个规矩。

**用例**（本地 5 个 + CUDA 真机 1 个）：20 步损失下降；参数与优化器手里的数组仍是
f32；`trainer.graph` 上的 matmul 输入确实是 f16 而图输出回到 f32；`mixed_precision=None`
时图一个节点都不多；`"f64"` 在构造期抛 `ValueError`；设备上 20 步训练跑得通。

**实测**（RTX 2080 Ti / Python 3.10 / cupy 14.2；两层 1024×1024 MLP + relu + MSE，
SGD，20 步取中位数）：

| 口径 | f32 | `mixed_precision="f16"` |
|---|---|---|
| 单算子 matmul 1024³（`execute` 含每次 H2D） | 3.06 ms | 1.77 ms |
| 前向，batch 128 | 2.55 ms | 3.30 ms |
| 前向，batch 1024 | 6.83 ms | 6.71 ms |
| 训练步，batch 128，主机态 | 13.3 ms | 14.2 ms |
| 训练步，batch 1024，主机态 | 18.0 ms | 19.8 ms |
| 训练步，batch 128，设备态 | 1.88 ms | 1.95 ms |
| 训练步，batch 1024，设备态 | 5.87 ms | 5.88 ms |

**为什么没有收益。** 重写后前向 11 → 17 个节点、反向 39 → 51 个（其中 12 个是
`cast`，因为 `cast` 的 VJP 本身也是一次 `cast`）。前向实测多花 0.75 ms、训练步多花
约 0.9 ms，折算每个 cast 节点 0.08～0.12 ms（一次 launch 加一趟显存往返）。另一头
5.93 把 H2D 与派发都剥掉量了纯 GEMM：f16 只比 f32 快 19%（batch 128）/ 33%
（batch 1024），折算到这两层 MLP 上约 0.1～0.3 ms。两边同量级，于是主机态净亏、
设备态持平。单算子那行的 1.7 倍不能当加速比看：它包含每次 `execute` 的 H2D，
两个数都虚高，只能说明 f16 GEMM 本身确实更快。

**结论**：开关留着、默认关闭。5.93 的纯 GEMM 对照把它的收益上限钉在两成到三成：
消掉 `cast` 节点是必要条件，但即使全消掉，在这张卡与这个规模上也只有
0.1～0.3 ms/步。混精在这里的价值是省显存，不是提速。

**5.91 实测口径**：本地 1187（919 通过 / 268 跳过）；CUDA 真机 1187（1133 通过 /
54 跳过），`tests.test_autocast` 26 个（本地 23 过 3 跳，设备上全绿）。5.93 之后
这两个数各加 2，数字见 5.93 末尾。

### 5.92　换宿主后的基线重采：门禁从红转绿

5.70 记下的是"正确性全绿、性能门禁为红"，5.89 又查明原因是换了一台宿主 CPU 更弱
的实例（纯 launch 开销的 `matmul_256` 最明显），当时只给基线加了机器指纹来**提示**，
没有重采——于是门禁一直红着。红着的门禁等于没有门禁：它既拦不住回归，也没人再看。

**重采前先量一次差距**（同一份 `baseline_cuda.json`，8 个用例）：

| 用例 | 旧基线 | 本轮实测 | 变化 |
|---|---|---|---|
| `matmul_256` | 0.2350 | 0.4453 | +89.5% |
| `matmul_1024` | 1.7596 | 2.8799 | +63.7% |
| `matmul_batched` | 0.4508 | 0.6895 | +52.9% |
| `conv2d_3x3_pad1` | 0.3393 | 0.7591 | +123.7% |
| `conv2d_1x1` | 0.3575 | 0.6875 | +92.3% |
| `maxpool2x2` | 2.9697 | 5.3883 | +81.4% |
| `transformer_block` | 1.1825 | 3.2589 | +175.6% |
| `cnn_lenet` | 0.6735 | 1.6104 | +139.1% |

八个用例**齐平**变慢、且最慢的是节点最多的两条整图用例，与"某个 kernel 退化"的
形状不同——确认是宿主差异，不是回归。

**新基线**（`benchmarks/baseline_cuda.json`）：带 `machine` 指纹
（`ubuntu22` / Xeon E5-2678 v3 / 8 核 / Python 3.10.12），覆盖 10 个用例——
5.80 加的两个"权重常驻"口径（`transformer_block_resident` 2.3609 ms、
`cnn_lenet_resident` 1.1109 ms）以前不在基线里，这次一并纳入，门禁从此也看着它们。
`graph_mode` 记为 `off`，与捕获路径无关。

重采后立刻用新基线复检一次：`回归检查通过（阈值 25%）`；再跑一轮同样通过。
两轮之间的抖动在 −10% 上下（`transformer_block` 3.28 → 2.96），远在阈值内。

**GPU 作业五步在真机复跑**：`verify_cuda.py` 全部通过（含 `tuning_roundtrip`）、
conformance `result: PASS`、性能门禁通过、`benchmarks.tune` 搜出 9 条配置且
`--apply` 后逐位一致、`benchmarks.trend --rounds 3` 写入 30 条记录。也就是说
5.70 记的那条"门禁为红"已经消掉，作业剩下的唯一缺口是托管平台本身。

**还没做的**：在 GitHub Actions 上实跑。本地这份工程没有 git 仓库、也没有 runner
注册令牌，这两样都不在代码里——接 runner 是仓库侧操作（`settings/actions/runners`
生成令牌 → 目标机 `config.sh` 注册标签 `[self-hosted, gpu]`），不做这一步就只能
继续手工逐条跑。基线已就绪，注册完第一件事应该是手动触发一次 `workflow_dispatch`，
确认云端路径与这五次真机执行一致。

### 5.93　剥离派发后的 GEMM 对照：f16 的真实优势是两成到三成

5.91 的结论里留了一句"消掉 cast 就能把 GEMM 省下的时间原样落到步时上"，但那句
话没有验证过。要验证它，得先把 H2D 与逐节点派发都剥掉——否则量到的永远是
"GEMM + 包装"的和。

**做法**：两张单算子图（`x @ w`，方阵 1024×1024，batch 分别 128 / 1024），
输入全部用 `resident_parameters` 常驻，`graph_mode="auto"` 让每次 `execute` 只做
一次 replay，重复 50 次取平均。这样剩下的几乎只有 GEMM 本身。

| batch | f32 | f16 | 加速比 |
|---|---|---|---|
| 128 | 0.406 ms | 0.342 ms | 1.19× |
| 1024 | 1.307 ms | 0.979 ms | 1.33× |

**怎么读这两个数**：每次 `execute` 仍要做一次输出 `to_host`（1024×1024 f32 是
4 MB），所以 GEMM 的真实差距只会比表里的更大；但也不会大到"换 f16 就翻倍"。
2080 Ti 上的 f16 GEMM 走 tensor core，而这批形状的 K 只有 1024、M 最小才 128，
本来就没喂饱。

**结论**：5.91 里"消掉 cast 就能拿回收益"的说法要收窄——`cast` 节点是必要条件，
不是充分条件。即便把 12 个 cast 全部消掉，两层 MLP 的步时也只该降 0.1～0.3 ms，
而当前混精净差是 −0.07～−1.7 ms，方向能对上。混精在这个规模上的价值是省显存，
提速要等 GEMM 占步时的比例更高（更大 batch、更深更宽的网络）。

**顺带**：`cast` 的 VJP 原来无条件建一个 `cast` 节点，即使上游梯度已经是输入的
dtype（恒等转换）。5.91 那张混精图里没有触发（12 个 cast 每一个都是真的跨宽度），
所以数节点数看不出差别；但它是可省的一次 launch，直接改成透传，并补了两个用例
（同 dtype 不加节点、跨宽度仍转回 f32）。

**5.93 实测口径**：本地 1189（921 通过 / 268 跳过）；CUDA 真机 1189（1135 通过 /
54 跳过），`tests.test_autocast` 28 个（本地 25 过 3 跳）。

### 5.94　外部审计的 P0：softplus 溢出

外部审计（2026-11-01）针对一份较早的快照（883 通过 / 2 失败）给出 10 条缺口。
先核对哪些还成立：

- **`aten.relu_` 前端缺口**：审计快照里的 2 个失败，5.90 已修，当前 1192 用例全过。
- **"Trainer 参数与优化器仍在 CPU"**：按**默认**路径判断是对的；5.84 的
  `device_state=True` 已经把参数与优化器放到显存上原地更新，只是默认值仍是
  `False`，所以审计看到的是真实的主路径行为。
- **其余 8 条成立**：scan 物化完整状态、时间维串行、没有 decode 状态接口、
  Mamba 是简化原型、softplus 会溢出、初始化没有时间尺度设计、没有 32K 测试、
  没有语义记忆与 Agent 层。

本轮只修其中最局部、也最没有争议的一条：softplus。

**问题**：`log(1 + exp(x))` 在 f32 下 x > 88 就溢出。`dt = softplus(dt_w * x + dt_b)`
一旦变大，inf 会顺着 `a = exp(dt * A)` 污染整条 scan；反向是 `sigmoid(x)`，
在 inf 处得到 NaN。这不是"极端输入才会碰到"的问题——`dt_b` 是可训练参数，
一次较大的更新就能把预激活推过 88。

**改法**：`max(x, 0) + log(1 + exp(-|x|))`。指数项的入参恒落在非正半轴，
`exp` 的取值范围是 (0, 1]，两个方向都不会溢出；x 很大时结果就是 x，x 很小时
是 `exp(x)` 的一阶近似，与原来的公式在正常区间内一致。绝对值用
`relu(x) + relu(-x)` 拼出来，不为它新增算子——Mamba block 里 dt 只有
`(B, T, d_inner)`，多出来的几个逐元素节点会被语义级 L2 融进一条链。

**验证**：3 个新用例——±500 输出有限、正常区间与 `log1p` 版本一致、极值处
梯度分别是 0 与 1；Mamba block 的数值对拍与梯度用例不变通过（正常参数下两种
写法等价）。本地 1192（924 通过 / 268 跳过），CUDA 真机 1192（1138 通过 /
54 跳过）。

**没有动的**：审计的其余 P0——chunked selective scan、时间维并行前缀扫描、
streaming decode 与卷积历史缓存、Mamba 的 A/dt 初始化——都不是"改一处"能收尾的。
前两项要同时改 `models/ssm.py`、`ir/ops.py` 的参考语义、`lower/kernels.py` 的内核
与 CUDA/HIP 两侧的 launch；decode 还要新增状态容器与缓存契约。这些需要单独立项，
并按审计建议的验收顺序（数值 → 分块 → decode → 语言模型 → 记忆）推进。

### 5.95　外部语义审计：35 条逐条核对，修掉 13 条语义红线

第二份审计报告（2026-11-01）给出 AF-01～AF-35 共 35 条缺陷，并按"冻结扩张修语义
红线 → SSM 流式 baseline → 设备驻留训练 → 语言模型可用性 → 记忆"五个阶段排序。
报告基于 `853 通过 / 291 跳过` 的快照，当前是 1206 个用例，所以第一步不是照单全改，
而是逐条回到源码确认"现在还成立吗"。

**已不成立的**：AF-02（softplus 溢出）在 5.94 修掉。AF-29（训练更新在主机）只是
部分成立——5.84 的 `device_state=True` 已把参数与优化器放到显存上原地更新，但默认
值仍是 `False`，报告描述的是主路径的实际行为，开关默认值没有改。5.118 补上了审计
最后那句"验证同一版本"与异步快照；默认值仍为 `False`，理由（设备态需要
`capture` / `to_array` / `to_host`，参考后端没有）写在该节。

**本轮修掉的 13 条**，按"错在哪 → 怎么改"记：

- **AF-01 交叉熵改变目标函数**：旧写法先算 `softmax` 再 `-log(p + eps)`，等于把
  目标函数乘上 `p_k / (p_k + eps)`，越需要纠正的样本权重越小；f16 下 `1e-12` 直接
  下溢成 0，`log(0)` 给 inf。改为稳定 log-softmax（减最大值、`logsumexp`），`eps`
  参数随之删除——它不再有存在意义。
- **AF-03 中间值不能作 `wrt`**：逆拓扑里 `cot.pop()` 之后中间值的梯度就丢了，
  `backward(g, wrt=[h])` 只能拿到全 0。给 `wrt` 里的值留一份引用（`saved`），
  弹栈后先存再查。输出名沿用 SSA 名（`"d" + h.name`），不是用户别名。
- **AF-05 分段重算把种子与 cotangent 二选一**：当同一段边界既是外部种子（该值
  同时是图输出）又是下游 cotangent 的落点时，旧实现只取其中一个。改为两组相加，
  两组都没有时按该值类型补零。
- **AF-06 重算对整数输入求导**：反向图对整个子图求导，`embedding` 的 `ids` 也被
  当作可微输入，vjp 返回 None 后断开。构造段反向时只取 `dtype.kind == "f"` 的输入，
  反向传播循环里跳过本段不存在的 cotangent 键。
- **AF-07 跨图 Value 按结构相等放行**：`Value` / `Node` 是 dataclass，默认 `eq=True`；
  两张图各自第一个输入同名同形，`a not in self._values` 直接放行，执行时按本图 id
  读到一条无关的边。改 `@dataclass(eq=False)`，比较与哈希都走对象身份。
- **AF-08 自动 SSA 名覆盖用户输入名**：自动名是 `v{len(_values)}`，用户把输入命名为
  `v1` 时后建的节点会顶掉它，`find("v1")` 指到别的值、序列化往返后结果变化。自动名
  改为避让（`v1_1`、`v1_2`…），用户显式给名重复则明确报错。
- **AF-09 序列化假设输入都在节点之前**：eager 前端是边构图边声明输入，重放时
  "当前值个数"与建图时不同，自动名会整体漂一位。取舍过"禁止交错声明"，它会打断
  eager 前端（8 个用例直接失败），因此采用报告的首选方案：`to_dict` 显式写每个节点
  的输出名，`add_node` 新增可选 `name` 参数，重放按名还原。
- **AF-14 Graph 计划键漏掉显式可变输入**：键里只有 `(id(graph), revision, 签名)`，
  驻留 `w` 时按 `{x}` 捕获、之后显式传 `{x, w}` 会命中旧计划。键补上
  `tuple(sorted(inputs))`。
- **AF-17 优化器逐参数更新留半步**：第 2 个参数的梯度形状非法时，第 1 个已经更新
  过了。改为两阶段：先全量 `_coerce_grad` 校验并构造 `prepared`，全部通过才逐个
  `_update`，最后才 `step_count += 1`。
- **AF-18 恢复时不校验槽位完整性**：训练过的存档缺 `v` 也能加载，续训时静默换成
  零动量。`load_state_dict` 先查未知槽位，`slots` 非空或 `step > 0` 时要求
  `set(slots) == set(known)` 且每个槽位覆盖全部参数；构造完再一次 `clear()+update()`。
- **AF-19 checkpoint 失败后参数已被改写**：旧实现边读边写回，后面某个形状不符时
  前面的参数已经变了。改为先把全部参数读进内存并校验，再提交优化器状态，最后才
  原地写入参数。
- **AF-22 Adam 直接更新 f16 参数会出 NaN**：`eps=1e-8` 在 f16 下下溢成 0，零梯度得到
  `0 / (sqrt(0) + 0)`。入口直接拒绝半精度参数，提示用 f32 master + GradScaler。
- **AF-35 融合链缓存键遗漏数据流**：键只有"叶子名 + 标量 + 载体"，`t = x + y; out =
  t - z` 与 `out = z - t` 会命中同一份源码。键改为包含每步完整 `(out, op, args)` 与
  `(leaf name, scalar)`。

**验证**：14 个新用例——交叉熵在饱和 logits 下不溢出且梯度为 `[[1,-1]]`、跨 1e4 量级
与 float64 独立实现对拍；非法梯度不留半步、Adam 拒绝 f16、缺槽位被拒、失败加载不改
参数；分段重算的种子相加与整数输入不求导；同名同形外图 Value 被拒、用户输入名不
被顶掉、交错声明可往返；中间值作 `wrt` 的梯度；融合链操作数顺序区分、Graph 计划键
跟踪可变输入（后两条只在 CUDA 真机执行）。本地 1206（936 通过 / 270 跳过），
CUDA 真机 1206（1152 通过 / 54 跳过）。

**没有动的 21 条**（35 − AF-02 − 本轮 13）：AF-04（未选输出的反向剪枝）、
AF-10～AF-13（输入 dtype 策略、符号广播约束、`resolve_attrs` 把 dtype 字符串当形状
符号、前端 ATen 的 `alpha`/`eps`/`scale`/`enable_gqa` 未归一化）、AF-15（5.83 的
arena 已按视图引用计数回收，但报告点名的"同一存储多视图共用一段区间"仍待真机
确认）、AF-16（capture 后越界索引的报错语义）、AF-20（dropout seed 不推进）、
AF-21（attention 广播 bias 无梯度）、AF-29（设备态默认值）、AF-23～AF-28（SSM 的
稳定性参数化、时间尺度初始化、Mamba 兼容性、持久状态接口、完整状态物化、时间维
串行）、AF-30～AF-34（训练显存、预算语义、可变 IR 的契约闭合、调优统计、语言模型
闭环）。其中 AF-04、AF-10～AF-13、AF-16、AF-20、AF-21 都是"改一处 + 补用例"的
规模，但没有本轮 P0 的紧迫性（AF-13 与 AF-21 已在 5.96 关闭，AF-04、AF-10～AF-12
已在 5.97 关闭）；AF-23～AF-28 与
AF-30～AF-34 属于阶段 B～E，要按审计的验收顺序单独立项。

**后续进展**（补记）：AF-16 与 AF-20 已在 5.98 关闭；AF-24、AF-26、AF-27、AF-28
与阶段 B 的验收表已在 5.99～5.104 关闭；AF-31 已在 5.105 关闭；AF-32 的图 seal
已在 5.106、能力契约已在 5.107、算子效果描述已在 5.108、HIP 与 CUDA 大类拆分已在
5.109 关闭；AF-09 在 5.110、AF-15 在 5.111、AF-19 在 5.112、AF-33 在 5.113、
AF-23 在 5.114、AF-25 在 5.115 关闭（对拍与参数映射已落地，消融实验仍待训练预算）、
AF-30 的前向 mask 在 5.116 关闭、反向的量化评估在 5.117 完成（结论是复用已优于手工
分块，不写专用内核）、f16 的混合精度模板在 5.121 关闭（49 个算子至此全部支持 f16）、
AF-29 在 5.118 关闭（设备驻留训练 5.84～5.88 已在，本轮补上
"同一版本"的可调用校验与 step 边界上的异步快照；`device_state` 默认关是有意为之，
理由写在 5.118）、AF-34 的图内闭环在 5.119、产品层第一块（tokenizer / chat template /
left padding batch / spec 绑定）在 5.120、工具调用的受约束输出与执行回路在 5.122
落地、阶段 C 的第一块（调度器 / 梯度累积 / loss scaling 接进 `Trainer`）在 5.123
落地、f16 反向的数值量化在 5.124 完成（混合精度 2 层 / 16 步最差 2.3e-3、扩到
4 层 / 64 步为 4.0e-3 且不随深度放大；把降精度白名单扩到全部算子会让 `A_raw`
这类相消后的小梯度失真到 26%，纯 f16 的 scan 则仍是 1e-3 量级，顺带修掉
`Autocast` 对整数操作数的误 cast 与"空集回落默认"）、阶段 C 的 RNG key 派生在
5.125 完成（key = mix(seed, step, rank)，同一 step 的不同 rank 不再共用掩码；
4 个新用例）、f32 master weights 在 5.126 完成（参数 f16 驻留 + f32 主副本、
每步覆盖写回；与全 f32 参考的权重相对误差 6.07e-5 / 3.27e-4，13 个新用例。
结构代价也量了：2 层 LM 上 cast 30 → 45，全参数降精度并不减少 cast，省的是
参数缓冲与读取带宽）。sm_52 的 f16 原子加在 5.127 修掉：Tesla M40 上有 26 个
f16 用例在 NVRTC 编译期失败（Maxwell 没有 `atomicCAS(unsigned short*)`），
`af_cas_h` 在 `__CUDA_ARCH__ < 600` 时展宽到对齐的 32 位字，修完后 CUDA 后端
250 个用例全过、真机全量 1453 个用例跳过 64，同一轮也就地验证了 5.126 的设备态
master weights（200 步 loss 7.1172 → 1.89e-5，`verify_device_state()` 无问题）。
**仍开着的**是 AF-34 的最后一小块（真实语料上的训练与
验证集，需要数据与训练预算；审计没有要求解码策略，不自己加需求），以及阶段 D～E
本身。阶段 C 的最后一块在 5.128 关闭：fused selective scan 的两趟实现（块内复合
`(prod, sum)` + 块前缀/读出）、分块前后向、`scan_last_state` 末状态专用内核；块内
递推逐字不变、默认路径逐位不变，真机全量 1465 个用例跳过 64。

### 5.96　阶段 A 收尾：attention 偏置的梯度，与 ATen 参数的读取位置

5.95 的未动清单里，AF-21 与 AF-13 是"改一处 + 补用例"的规模，又都在阶段 A
（语义红线）的范围内，本轮一并关掉。

**AF-21：广播偏置拿不到梯度。** `attention` 的 VJP 用"mask 与 `ds` 形状完全相同
才给梯度"处理第四个输入，其余一律返回 None。这等于把"mask 通常是结构性常量"
这个**调用者的角色判断**写进了数学算子：可学习的相对位置偏置就是 `(1, S, S)` 或
`(1, 1, S, S)` 广播到 `(B, H, S, S)` 的正常用法，它应当有梯度。改法是新增
`_sum_to_mask`：逐个对齐维度，`mask` 为 1 而 `ds` 不为 1 的轴就是被广播的轴，
沿这些轴求和并保留维度（形状正好回到 `mask` 的类型）。归约复用 `_sum_axis_keep`
——它把轴转置到最后再用 `reduce_sum`，所以 CUDA 后端"只实现最后一维归约"的
约束不会让这条规则变成"只有参考后端能跑"。

仍然返回 None 的两种情形都写进了注释：`ds` 含符号维（静态形状不可判定），或
`mask` 的秩比 `ds` 小（归约之后要 reshape 回去，同样需要运行期形状表达式）。
原来断言"广播 mask 无梯度"的用例改为断言数值梯度与 `(1,1,3,3)` 的形状，并补了
一个 `(1,3,3)` 的可学习偏置用例。

**AF-13：ATen 参数读错了位置。** 四个降级分支把"参数在哪"搞反了，而导出图不会
报错——它只是算另一个函数：

- `aten.add.Tensor` / `aten.sub.Tensor`：`alpha` 在 schema 里是 **kw-only**
  （`*` 之后），只出现在 `kwargs`。旧代码直接 `g.add(a(0), a(1))`，`x + 3y`
  算成 `x + y`。
- `aten.layer_norm.default`：`eps` 是**位置**参数第 5 个，旧代码只读
  `kwargs["eps"]`，`layer_norm(..., 0.1)` 退回 `1e-5`。
- `aten.scaled_dot_product_attention.default`：`scale` 与 `enable_gqa` 也是
  kw-only，旧代码按 `args[6]` / `args[7]` 读，永远读到 None——显式 `scale=0.01`
  被换成 `1/sqrt(D)`，`enable_gqa=True` 的拒绝分支根本走不到。

改法三件：`_aten_arg` 统一"先关键字、再位置、最后默认"的取值顺序；新增 `t()`
让张量参数也能从 `kwargs` 取（合成 FX 图会把 `attn_mask` 写成关键字）；
`_reject_unknown_kwargs` 把降级规则不认识的参数直接拒绝——静默忽略一个关键字
与"静默跳过未支持算子"是同一种错误，策略应当一致。`alpha != 1` 折成一次标量
乘法（`x ± alpha*y`，语义精确、不新增算子），`scale` 不是 Python 标量时明确报错。

**验证**：6 个新用例（1 个 attention 偏置的数值梯度 + 5 个端到端 torch 导出对拍：
`add alpha`、`sub alpha`、`layer_norm` 位置 `eps`、SDPA 的 `scale`、`enable_gqa`
必须报错）。前端的 5 个只在有 torch 的机器上执行——远端 CUDA 机器没有 torch
（ADR-005），所以真机跳过数从 54 变成 59，属于预期。本地 1212（942 通过 /
270 跳过），CUDA 真机 1212（1153 通过 / 59 跳过）。

### 5.97　阶段 A 续：剪掉未激活的输出，收紧符号与属性的歧义

继续关 5.95 的未动清单。这一轮四条都在阶段 A（语义红线）范围内。

**AF-04：给未选输出喂零种子不等于剪枝。** `Trainer` 此前对每个图输出都开一个
种子输入，只有损失收 1、其余收 0。零种子只在导数链上每一环都有限时才等价于
"不传播这条分支"：`loss=sum(x*x)` 配一个诊断输出 `sum(log(x))`，在 `x=0` 处
零种子仍要算 `0/x`，得到 NaN，把对的梯度一起污染。改变发生在构图期，而不是
执行期：

- `backward(..., active=[...])` 只给激活输出建种子，并沿它们做一次逆拓扑可达性
  扫描（节点按拓扑序排列，一次逆序遍历即可），未激活分支的前向节点**根本不重放**
  进反向图。`active=[]` 与拼错的别名都明确报错。
- `Trainer` 在构造期就只传 `active=[chosen]`，主机态不再补零种子；设备态整图版
  也只剩损失那一个种子缓冲，捕获计划的输入集合随之变小。
- `CheckpointPlan(output=...)` 先算出"沿损失可达"的值集合，切段时把到不了它的
  值排除在导出之外；段反向图因此不为它们建种子。副产物：某一段可能整段落在
  死分支上而没有任何导出，合并逻辑补了"尾段为空也要并回前一段"——原来只处理
  了前一段为空与首段为空。

**AF-11：符号广播只挑一方。** `_broadcast_shape` 原来的规则是"任一方是符号就取
那一方"。`(B, 2) + (3, 2)` 在 `B=1` 时结果应当是 `(3, 2)`，在别的 B 下直接
非法；结果维究竟是静态 3 还是符号 B，IR 里没有能表达这种选择的广播维表达式。
现在只有三种确定情形放行：相同、一方是静态 1、两边都静态且相等；其余在构图期
拒绝，并提示"改成静态形状或先显式 reshape"。与符号相关的运行期形状不符因此
提前成了构图期错误。

**AF-12：形状符号与 dtype 名共用字符串。** `resolve_attrs` 把属性里的每个字符串
都当符号查表，注释里"字面量不会被误改"只在符号不叫 `f32` 时成立。名字本身是
歧义的根源（`f32[f32, 4]` 对人和机器都分不清哪处是 dtype），所以修在类型入口：
`TensorType` 构造期拒绝与 `f16/bf16/f32/f64/i8/i32/bool` 重名的形状符号。
`resolve_attrs` 的宽泛替换保留——它还要处理 `constant` 的 `shape` 这类真正
承载维的属性——但前提现在成立，并在注释里指向这条校验。

**AF-10：输入 dtype 与 IR 声明。** 执行入口此前只校验形状，声明 f32 传 f64 时
整条链按 f64 算、直到输出才降回 f32：类型是协议，精度却不是。现在入口就把输入
转换到声明 dtype（`np.dtype(type.dtype.np_dtype)`，bf16 的承载本来就是 f32，
不做位宽转换）。判据用一个能区分两者的值：`1 + 2**-30` 在 f32 里就是 1.0，
若入口不转换，`x - 1` 会得到 `9.3e-10`。

**验证**：10 个新用例（符号广播拒绝 2、静态 1 仍放行 1、dtype 重名符号拒绝 1、
`resolve_attrs` 不动 dtype 字面量 1、`active` 剪枝与对照 NaN 1、Trainer 端到端 1、
分段剪枝与尾段合并 2、入口 dtype 转换 1）。本地 1222（952 通过 / 270 跳过），
CUDA 真机 1222（1163 通过 / 59 跳过），两边全绿。

尚未动：AF-16（CUDA 捕获期越界索引只在普通执行报错）与 AF-20（dropout 的 seed
是 attrs 常量、`Trainer` 不推进）。前者需要改捕获后读回设备错误标志，后者要把
RNG counter/key 变成显式状态，都属于"改一处不够"的规模，留待下一轮。

### 5.98　阶段 A 收官：捕获后的越界复查，dropout 的随机源变成显式状态

5.97 留的两条都不在单点能改完的形状里：一条要把"只在普通执行里做的检查"搬进
图重放的路径，另一条要把随机源从 attrs 挪到设备内存。本轮把两条都关掉。

**AF-16：捕获期的越界下标只在普通执行报错。** `_check_ids` 在 `self._capturing`
时直接 `return`，理由是捕获期不能做 D2H 同步读。代价是越界从"报错"退化成"内核
clamp 之后照常跑完"：普通执行拒绝的输入，捕获后 replay 悄悄接受，同一份程序两种
语义。修法把检查拆成两半——捕获期只登记，replay 后再读回：

- 捕获期把 `(err, op, allowed)` 追加进局部 `guards`，`capture` 结束时随计划一起
  交给 `CudaGraphPlan`；非捕获期 `self._capture_guards` 恒为 `None`，不增加常驻
  状态。
- `run()` / `run_device()` 在 `synchronize()` 之后调 `_check_guards()`：标志非零
  就**先在捕获流上清零**，再抛 `BackendCapabilityError`。清零这一步不是洁癖——
  图里写死的是同一个标量地址，不清零的话下一次合法 replay 会读到上一次的脏标志。
- 错误消息抽成 `_raise_bad_ids(op, allowed)`，普通执行与图重放共用同一条文案，
  避免两种路径的报错措辞漂移。
- HIP 侧对称实现（`HipGraphPlan` 存 `_guards`、同步后复查）；只有清零的写法不同：
  HIP 的 `DeviceArray.__setitem__` 只接受整块同形同 dtype 赋值，所以是
  `err[...] = runtime.zeros(err.shape, dtype=err.dtype)` 而不是 `err[...] = 0`。
  `_capture_guards` 在 `HipBackend.__init__` 里同名声明，5.58 那条 AST 对账测试
  因此仍然通过。

判据不是"报错了"而是"报错之后这张计划还能用"：捕获一张 embedding 图，先用越界
下标 replay（抛错），再用合法下标 replay 并与参考后端逐位一致——标志的清零如果
写错，第二条会误报。

**AF-20：dropout 的 seed 是常量，训练器不推进随机状态。** `dropout_mask` 的随机
源写在算子 attrs 里，语义上"同图同结果"很干净，但它和 CUDA Graph 的组合是错的：
捕获会把内核的标量参数按值冻结，`seed` 作为标量参数被烤进图里，replay 多少次都是
同一张掩码。`Trainer` 只换数据不换 seed，于是"训练"退化成"每步用同一份 dropout
掩码"。种子留在 attrs 里无论怎么改都绕不开冻结，所以随机源必须进设备内存：

- `dropout_mask(x[, key], p=, seed=, offset=)`：`key` 是可选的 i32 `()` / `(1,)`
  图输入。CUDA 内核改为从 `const int* __restrict__ key` 读种子（常量 seed 仍是
  普通标量参数），replay 前把新值 H2D 到同一地址即可推进——图捕获的正是这个地址。
- `offset` 让同一张图里的多个站点共用一个 key 却拿到不同掩码（`(layer*2)` 与
  `(layer*2+1)` 这种编号），否则同形状的两个 dropout 会撞成同一张掩码。参考实现
  抽出 `_dropout_seed(args, attrs)`，常量路径与 key 路径统一成
  `(seed + offset) & 0xFFFFFFFFFFFFFFFF`。
- L2 描述表达不了"整数指针参数"（`Tensor` 只有浮点存储、`Load` 只读成 float），
  所以带 key 的变体是一份手写模板 `DROPOUT_MASK_KEYED_SRC`，常量 seed 的主路径
  仍走 L2，不为了这一个参数把整个内核拉回手写。HIP 的 `codegen` 通过 `__getattr__`
  转发到 CUDA 模块并注入 f16 存取，因此同一份模板在 HIP 上自动可用。
- `Trainer(..., rng_inputs=("rng",))`：每步在执行前把 `np.asarray([steps], int32)`
  写进这些输入，所以同一 step 的前向、分段重算、反向共用同一个 key（重算复现的
  必须是同一次采样），下一步换新值。设备态用 `_dev_data[name][...]` 原地推进——
  replay 只换内容不换地址。`rng_inputs` 里出现参数名或非输入名都在构造期拒绝。
- 边界：`DType` 只有 `I32` 没有 `I64`，key 就限定 i32；`fused_attention` 的 dropout
  随机源是 attention 自己的 attrs（常量 seed），`rng_input` 与它同时打开、且
  dropout > 0 时在构图期拒绝，而不是留到执行期才发现不生效。

**真机首跑才暴露的一条。** `invoke` 的通用 dtype 检查拿每个输入去比激活 dtype，
key 的 i32 在进入内核之前就被"只支持浮点"挡了下来。`_INDEX_ARGS` 是给这类**整数
输入**开口子的表（`embedding` 的下标早已在其中），补一条 `dropout_mask: {1}` 即可。
本地全量跳过了所有 CUDA 用例，所以这条只有真机能发现——和 5.89 的
`InstrumentedBackend` 是同一类教训。

**验证**：11 个新用例（AF-16：捕获后越界 replay 报错且计划仍可用 1；AF-20：keyed
语义 4 条本地用例——换 key 换掩码、同 key 逐位稳定、offset 区分同 key
的两个站点、key 必须 i32 标量；Trainer 随机状态 5 条——逐步推进、同 step 可复现、
分段重算与整图梯度一致、非输入名拒绝、参数名拒绝；真机 1 条——捕获后 replay 换 key
得到不同掩码且各自与参考逐位一致）。本地 1233（961 通过 / 272 跳过），CUDA 真机
1233（1174 通过 / 59 跳过），两边全绿。

至此阶段 A（语义红线）在可复现的条目上全部关闭。

### 5.99　阶段 B 的第一块：SSM 的流式推理——末状态出参与重置

AF-26 指出的问题不是数值错误，而是**口径缺口**：`build_mamba_block` 只认"整段
一次喂完"。这是训练的口径，推理的口径是另一种——prompt 先整段 prefill，之后逐
token decode，每步只喂一个位置、把递归状态从上一段带过来。缺了状态出参，这个
用法表达不出来，只能绕开模型构件自己拼图。本轮把缺的补齐，且**没有新增算子**。

**末状态：`slice + reshape`。** `build_selective_scan_stateful` 与旧函数共用
`_selective_scan` 主体，只是多取一步：

    h_last = reshape(h[T-1], h0.shape)

`h` 本来就是 `scan_linear` 的输出，末状态是它的最后一个切片；缺 `h0` 时用零常量
（图仍然自包含），给了 `h0` 就校验形状为 `(..., D, N)`。返回的 `h_last` 可以直接
当作下一段的 `h0`——分块续接因此是纯图上的连接，不依赖任何运行期对象。

**重置：把 `a` 在重置位置乘 0。** 批量推理要在一个 batch 里塞多段互不相关的序列，
不切断的话后一段会继承前一段的状态。语义取 `h_t = keep_t * (a_t * h_{t-1} + b_t)`、
`keep = 1 - reset`：重置位置只用输入项重建状态。关键在于乘法发生在**递归内部**
（`a` 被改写后再进 `scan_linear`），所以前缀段的状态连反向图都不经过——不是"结果
看起来被冲掉"，而是数学上不存在这条路径。

**卷积窗口：`concat` 出历史，`cumsum` 判跨越。** 因果深度卷积要跨调用保住最后
`K-1` 帧：

- `source = concat(state, x, axis=1)`，第 `j` 个抽头就是 `source[t+j]`；新状态
  是 `source[T : T+K-1]`，边界既不丢也不重。`state` 缺省为零常量，等价于原先的
  左侧补零；`d_conv=1` 时中间维为 0，直接返回空状态，不特判拒绝。
- 卷积的 reset 比 scan 麻烦：窗口读的是**更早的输入**，要判定"位置 `p` 与 `t`
  之间有没有重置"。用累积和就够——`cs[b,t] = sum_{s<=t} reset[b,s]`，`cs_t != cs_p`
  等价于区间 `(p, t]` 内有重置。累积和用 `scan_linear(a=1, h0=0)` 表达，跨 reset
  的抽头乘 `valid = 1 - (step(gap) + step(-gap))` 归零（`step` 是既有的 `x>0`）。
  历史部分（`state`）的累积和按 0 计：窗口状态被视为"属于当前段之前"，于是段首
  一步的重置会把历史整体切断。

**共用主体与两个入口。** `_mamba_body` 收 `(conv0, h0, reset)` 三个可空值与一个
命名函数，`build_mamba_block` 传 `None` 保持原签名与原返回（既有 87 个用例不改一
行）；`build_mamba_block_stateful(prefix=, with_reset=)` 把两个状态变成图输入、把
它们与 `y` 一起返回。`MambaSpec` 新增 `conv_state_shape`（`(B, K-1, D_inner)`）与
`ssm_state_shape`（`(B, D_inner, N)`），形状只在这一处定义。

**`steps=1` 的同一份函数就是 decode。** 图不同（`T` 写死为 1），语义完全一致，
所以不需要第二套代码路径。单步时 `_reset_prefix` 省掉递归节点（累积和就是它自己），
这是 decode 的热路径。

**为什么不给卷积新算子。** 核长 3~4，展开成 `K` 次移位相加的节点数远少于"新增
算子 + 反向 + 两个后端实现"的成本——和 5.53 的理由相同。代价是序列长度必须静态，
这与 RNN/SNN 的时间展开是同一笔账。

**验证：三条恒等式。** 新增 `tests/test_ssm_streaming.py` 共 15 个用例（13 本地 +
2 CUDA），全部落在"同一段序列用不同喂法必须一致"上：

| 恒等式 | 判据 |
| --- | --- |
| 任意分块 == 整段 | 切点 2/1/3，首段继承非零 `h0`，末状态逐位相同 |
| 逐 token == prefill | 每步 `T=1`，状态手工串接，输出逐位相同 |
| 带 reset == 各段独立跑 | 一次带掩码的前向 == 把每段单独搭图跑一遍再拼起来 |

外加四条边界：零状态复现旧 `build_mamba_block`（`assert_array_equal`，`concat` 与
`pad` 的图不同但数值逐位一致）、因果性（改动 `t` 之后的输入不影响 `t` 的输出）、
`h0` 的中心差分梯度（非零初值确实参与反向）、形状校验（错误 conv state / `h0` /
reset 都在构图期拒绝）。CUDA 侧两条恒等式（逐 token、reset）在真机上用
`np.asarray(cuda.execute(...))` 取值比对。`d_conv=1` 的退化形状另有内联脚本验证
构图与执行都正常。

本地 1248（974 通过 / 274 跳过），CUDA 真机 1248（1189 通过 / 59 跳过），真机单跑
`tests.test_ssm_streaming` 15 个全绿（含 2 个 CUDA 用例）。

### 5.100　阶段 B 续：多层栈——状态是每层一对，名字不能和输入重名

5.99 证明的是"一层能续"。模型是 N 层，续接的最小单位因此变成"每层一对状态"。
本轮把 `_mamba_layer` 从 `_mamba_body` 里抽出来（参数改从 `params[prefix + 名字]`
取，`_mamba_body` 只剩"声明输入并调一次"），`build_mamba_block` 一个字符没改，
87 个既有用例因此没有任何一处需要调整。

**整栈是一张图，不是 N 张图手串。** `build_mamba_model` 与
`build_mamba_model_stateful` 逐层复用同一个 `_mamba_layer`，参数名与状态名都带
`l{i}_` 前缀：

    x -> l0_* -> l1_* -> ... -> l{N-1}_* -> y

这样 `ParameterStore.from_graph` 能一次拿到全部参数，`Trainer` 也能一次跑完整个
栈——"多层"不该意味着调用方要自己写循环并把中间结果在图外搬来搬去。

**`MambaModelSpec` 只管形状。** `num_layers` 与单层形状放在一起；`layer` 属性把
自己投影成一个 `MambaSpec`（所以每层的卷积窗口 / SSM 状态形状只有一处定义），
`with_steps(n)` 用 `dataclasses.replace` 换序列长度——prefill 用整段、decode 用
`with_steps(1)`，两张图共享同一套参数名与状态名。

**实测撞到的一条：图里不能拿同一个名字同时做输入与输出。** 第一版把 `state_out`
的键写成 `l{i}_conv0` / `l{i}_h0`（与输入同名，看起来很省事），`g.output()` 直接
抛"名字重复"。改成输出 `l{i}_conv` / `l{i}_h`（与单个 block 的 `conv` / `h` 同构，
输入才是 `conv0` / `h0`），并由 `MambaModelSpec.carry_state(state_out)` 做翻译：

    feed.update(spec.carry_state(state_out))

映射规则就是"去掉末尾那个 0"，但把它放进 spec 而不是散落在调用方，是因为只有
spec 同时知道输入名与输出名。这条约束在单层时不存在（block 的字典键与图名可以
分开），堆叠之后才暴露，属于"写测试比读代码更快发现"的那类。

**reset 全层共用一张掩码。** `with_reset=True` 只多一个 `reset (B, T)`，而不是每层
一个——"这个位置开始新序列"是数据层面的事实，逐层各说一遍只会让调用方有机会传错。

**验证**：7 个新用例（6 本地 + 1 CUDA）。判据仍是三条恒等式，但参数**逐层不同**
（每层用不同 seed 生成）：若状态在层间错位一格、或只把第一层状态接回去，分块与
整段立刻不等。另有三条结构性判据：

- `num_layers=1` 与 `build_mamba_block` 逐位一致（同一份层主体，只是参数名带前缀）；
- 零初值的流式图与训练口径 `build_mamba_model` 逐位一致（图不同，数值相同）；
- `with_steps` 只改 `steps`，参数名与状态形状原样保留。

CUDA 侧跑整栈（2 层）的逐 token vs prefill。

本地 1255（980 通过 / 275 跳过），CUDA 真机 1255（1196 通过 / 59 跳过），两边全绿。

### 5.101　阶段 B 续：多时间尺度初始化——半衰期是数出来的，不是猜的

状态能跨调用续上了，下一个问题落在"要不要续长"上。审计 AF-24 的切口很窄：
`train/params.py` 的通用规则给秩 ≥2 的参数用 Xavier、秩 1 且不以 `b` 结尾的用一，
所以 `A_raw ≈ 0`、`dt_b = 0`。代入模型自己的公式：

    |A| = exp(A_raw) ≈ 1，Δ = softplus(0) = log 2 ≈ 0.693
    a = exp(Δ·A) ≈ exp(-0.693) ≈ 0.5

`0.5^H = 1/2` 得 **H ≈ 1 token**。也就是说所有通道都只有一步记忆：这个初始化
下，"复制上一个 token"几乎能解释全部损失，长依赖的梯度信号弱到没有动力学压力。
这不是稳定性问题（`a<1` 仍然成立），而是**归纳偏置**问题——先验把模型按在了短记忆上。

审计同时指出不能直接照搬官方：官方把 Δ 走低秩投影再投回（`dt_proj`），本构件是
逐通道 `dt_w·x + dt_b`（AF-25 已声明这是 Mamba 风格变体），照抄 `dt_proj` 的
初始化没有落点。官方真正可迁移的是两件独立的事：`dt_b` 逐通道 log-uniform、
`A` 用 S4D-Real 的倍频序列。这里把二者合并成**一个可读的网格**。

**直接指定目标半衰期，反解 A。** 遍历 `(d_inner, d_state)` 的每个单元给一个
目标半衰期 `H`，在 `[H_min, H_max]` 上对数等距；`dt_b` 逐通道在 `[Δt_min, Δt_max]`
上对数等距给出名义步长；再由 `H = log 2 / (Δ|A|)` 解出

    |A| = log 2 / (H·Δ)，A_raw = log|A|

好处是**初始化的语义可读且可验收**：`a = exp(-log2/H) = 2^(-1/H)`，于是
`0.5^H` 的衰减是**逐单元精确成立**的，不是"大约在 1 token 附近"。默认取
`H ∈ [1, 32768]`、`Δ ∈ [1e-3, 1e-1]`，`d_model=32 / d_state=16` 的实测分布为

    分位数  0%    1%     25%    50%    75%    99%     100%
    H      1.00  1.11   13.5   181    2436   29532   32768

短（个位数 token）、中（百级）、长（万级）都有通道覆盖——这正好是审计要求报告的
"初始化后的有效半衰期分布"。`A_raw` 落 `[-8.46, 6.54]`，`|A| ∈ [2.1e-4, 693]`，
都在 f32 的正常区间，离 `exp` 溢出很远。

**`dt_w` 初始化为零，而不是"默认为一"。** `Δ = softplus(dt_w·x_act + dt_b)` 里
`x_act` 经过 RMSNorm + 投影 + 卷积后典型幅度在 0.4 量级；`dt_w = 1` 意味着输入项
约 0.4，比 `Δt_max = 0.1` 还大一个数量级，`dt_b` 的分层会被激活直接盖掉——"多时间
尺度"只在 `x_act = 0` 的纸面上成立。零起点让分层完全由 `dt_b` 决定，输入调制交给
训练。这也和"偏置从零开始"的默认价值观一致，只是把"零"放对了位置。

**接口：初始化在模型层，不在训练层。** `init_mamba_params(spec, seed, ...)` 返回
与 `MambaSpec.param_shapes` 一一对应的字典（`init_array` 负责随机部分，A/Δ 由网格
覆盖）；`effective_half_lives(spec, params)` 用同一个 `log2/(Δ|A|)` 读数，参数换成
训练后的当前值就能得到"训练后的分布"，两个时刻用同一把尺子。`MambaModelSpec.init_params`
逐层调用并加 `l{i}_` 前缀。三个细节值得记：

- **`init_array` 的随机流混的是名字的 CRC**，所以整栈初始化必须把层前缀传进去
  （`prefix=f"l{i}_"`），否则每层拿到完全相同的权重。第一版漏了这一步，用例
  `l0_w_in != l1_w_in` 立刻变红——这类"复制粘贴出来的对称"只能靠断言抓。
- **A/Δ 的网格刻意与 `seed` 无关**：换 seed 做消融时，时间尺度分配不该跟着漂。
  随机性只体现在投影与卷积上。
- **延迟导入 `train.params`**：构图层不在模块层依赖训练层（`models/*` 只用
  numpy 与 IR），`init_array` 的 Xavier / 常数策略在调用时取。

**没有做的部分，说明理由。** 审计的"迭代"还提到"部分参数不做 weight decay"。
`SGD` / `Adam` 目前的 `weight_decay` 是全局一个数，要做参数组得改优化器的公开面
（`step` 的签名或构造方式），而"哪些参数该衰减"本身应该由消融决定，现在把
`A_raw` / `dt_b` / 归一化 gamma 免掉的依据只有先验。所以本轮只落地 initializer，
参数组留到有对照实验时再动——这比先加一个没人验证的开关更诚实。

**验证**：4 个新用例。

- 键 / 形状 / f32 与 `param_shapes` 逐项一致，同 seed 逐位复现；换 seed 只改随机
  参数，`A_raw` / `dt_b` 不变；
- 半衰期逐单元落在对数网格上：排序后最小值 = 1、最大值 = 32768、相邻比值在 f32
  存储误差内相等；且相对通用初始化的跨度大过 100 倍（实测 8838 倍，通用初始化
  的所有单元都落在 0.53～1.97 之间——正是"只能记一个 token"的定量版本）；
- 专属初始化喂进图后前向有限、全参数梯度有限（反向图 `verify` 通过）；
- 整栈 `init_params` 的键集合等于 `param_shapes`、层间随机权重不同、A/Δ 网格
  逐层一致，并与 `build_mamba_model` 的图输入（除 `x`）完全对齐。

本地 1259（984 通过 / 275 跳过），CUDA 真机 1259（1200 通过 / 59 跳过），两边全绿。

### 5.102　阶段 B 续：流式状态进存档——重启之后接着 decode，与没断过一样

5.99 / 5.100 把状态带出了图，5.101 把时间尺度铺开。但状态只在进程里活着：参数能从
`.npz` 读回，状态没有落点。这恰好是流式推理最需要的能力——长会话要能搬家、现场要能
重启、评测要能从同一个前缀接着跑。

**为什么和权重存同一个文件。** 状态只在"这份权重"下有定义：`h` 的
`(B, d_inner, N)` 由模型形状决定，`conv` 的窗口长度由 `d_conv` 决定。分成两个文件，
现场就有机会把 A 模型的状态喂给 B 模型的权重，而形状检查要到 decode 第一步才炸。
装进同一个 `.npz`，参数与状态天然同源；`load_checkpoint` 的形状 / dtype 校验也能
一次覆盖两者。

**接口：键就是图输入名。** `save_checkpoint(path, store, optimizer=None, meta=None, state=None)`，
`state` 是"名字 → 数组"，名字取 `MambaModelSpec.state_shapes` 的键
（`l0_conv0` / `l0_h0` …）。`build_mamba_model_stateful` 的输入表用的正是这些名字，
加载之后不需要任何转换：

    state = {k: np.zeros(shape, np.float32) for k, shape in spec.state_shapes.items()}
    load_checkpoint(path, store, state=state)
    feed.update(state)          # 下一次 decode 直接用

**加载用"模板"而不是返回新字典。** 参数那条路是原地覆盖，因为优化器手里握着同一批
数组；状态没有这个约束，但模板同时解决了两件事：形状 / dtype 的对照物从哪来，以及
加载完的东西放哪。写入同样是原地，调用方手里的字典就是 decode 的 feed，不多一次拷贝
也不多一层改名。

**四类条目的加载规则是对称的。** 传了目标却没存档 → 报错（优化器那条早就如此；
状态同理，否则会静默从零开始、把历史白白丢掉）；没传目标 → 忽略存档里的那一类
（"只加载权重"永远合法）。于是旧格式存档照样可读，只是不能用来续接。

**先校验再提交。** 键集合、形状、dtype 任一不符，都在写任何数组之前失败——和 5.95
给参数加的那条同源。`test_stream_state_mismatch_leaves_template_intact` 断言模板
一个字节都没被改。

**验收判据是"续接后逐位一致"，不是"数组能读回来"。** prefill 3 个 token → 存档 →
用**全零**的权重和**全零**的状态模板重建 → 加载 → 逐 token decode 剩下 4 个，每一步
输出与"prefill 之后一直没中断"的逐 token 结果 `assert_array_equal`。任何一层状态丢了、
错位了、或者只接了第一层，第一帧就会不等。

**验证**：5 个新用例。3 个在 `test_train.py`（往返 + 原地写、旧格式缺状态报错、
键 / 形状不符且模板不动），2 个在 `test_ssm_streaming.py`（整栈恢复到逐位一致、
只存权重的存档仍能正常加载）。

### 5.103　审计 AF-27 / AF-28 的量化：物化随 T 线性，scan 的时间要在大 T 上才看得见

AF-27 与 AF-28 是同一条性能事实的两面：前者说"图不随 T 变，但张量随 T 变"，后者说
"所以别把复杂度当吞吐"。本轮把这两条从散文变成可复现的数。

**AF-27：静态预算，不执行、不需要 GPU。** `plan_memory` 按"定义到最后一次使用"算
liveness。图是 `build_selective_scan`（`T × B=1 × D=32 × N=8`，f32，含 `D_skip`）：

| T | 节点数 | 中间值总字节 | 中间活跃峰值 |
|---:|---:|---:|---:|
| 64 | 14 | 439,296 | 207,872 |
| 128 | 14 | 877,568 | 414,720 |
| 256 | 14 | 1,754,112 | 828,416 |
| 512 | 14 | 3,507,200 | 1,655,808 |

T 翻倍时总量与峰值都翻倍，差分给出的每 token 成本是 **6,848 B（总量）/ 3,232 B
（峰值）**，常数；节点数恒为 14。与审计表的数字逐行差 16,384 / 1,024 B，原因也能量
出来：审计的图是 12 节点、不带 `D_skip`，少的正好是 `mul(D_skip, x)` 与 `add` 两个
8,192 B 的 T 维值，以及 `A` 这个 1,024 B 的输入——同一把尺子，只是构图不同。

**"每 chunk 恒定状态"的边界。** 流式能救的只是"历史"，不是"这一段"。
`MambaModelSpec.state_shapes` 的字节数 `2 × (B·(K-1)·D_inner + B·D_inner·N) × 4`
与 `steps` 完全无关：`d_model=32 / d_state=8 / d_conv=4 / expand=2 / B=1 / 2 层` 时是
5,632 B，处理 1 个 token 与处理 4,096 个 token 一样大。但每次调用仍然要物化当前
chunk 的 `(T_chunk, B, D, N)` 中间值——所以分块把峰值变成 `O(T_chunk)`，而不是
`O(1)`。这条写进测试（`test_streaming_state_is_independent_of_history`），免得后来者
把"状态有界"误读成"显存有界"。

**没做**：审计建议的"fused selective scan lowering"（分块完成离散化、扫描与读出，
不把 DN 维中间值写回全局内存）需要一条新的 lowering 路径和等价性证明，本轮不碰；
先把量化口径固定下来，等它成为实际瓶颈再说。

**AF-28：只记录，不改 kernel。** 口径写死为：真机 RTX 2080 Ti、默认 `CudaBackend`
（`graph_mode` 默认 off）、`benchmarks.run.measure`（热身 20 次、计时 30 次、同步在
计时区间内、取中位数）。先测完整构造（14 节点，B=1）：

| T | 64 | 512 | 2,048 | 8,192 | 32,768 |
|---|---|---|---|---|---|
| median (ms) | 1.63 | 1.64 | 1.88 | 3.24 | 8.16 |
| µs / token | 25.4 | 3.20 | 0.92 | 0.40 | 0.25 |

T 从 64 涨到 512，总耗时**没动**：14 次 kernel launch 的固定成本约 1.6 ms，把 scan
的时间完全盖住了。这段数字唯一能说明的是"小 T 下整图耗时由派发主导"，不能拿来推断
并行度——µs/token 看起来随 T 骤降，完全是分母在变。

把 `scan_linear` 单独拎出来（1 节点）才看得到沿 T 串行的那一项：

| T | 1,024 | 4,096 | 16,384 | 65,536 |
|---|---|---|---|---|
| median (ms) | 1.12 | 2.90 | 11.30 | 61.21 |

4,096 → 16,384 恰好 4 倍 T、4 倍时间（约 0.63 µs/token，扣掉约 1 ms 的固定开销）；
65,536 时每 token 涨到 0.92 µs（+35%），是长步长的访存 / 缓存效应。再固定
T=16,384 扫 batch：

| B | 1 | 4 | 16 |
|---|---|---|---|
| median (ms) | 11.31 | 57.14 | 226.71 |

batch 翻 4 倍，耗时涨 4～5 倍：**没有摊薄**。这正面回答审计的疑问——"batch × 通道 ×
状态"的并行只能同时处理更多尾元素，不能缩短每个尾元素的 T 循环；工作量是
`B·T·D·N`，时间就跟着总量走。要真正缩短关键路径，得做块内并行（chunked scan /
Blelloch），那是另一件事。

**边界与结论**：T ≤ 2,048 时这条串行路径不是热点（launch 开销占大头），所以本轮
不动它；一旦 prefill 到 8K 以上，线性项开始可见。没有"优化前 / 优化后"的对比，
也没有外推任何加速比。

**验证**：3 个新用例（`test_ssm_streaming.py` 的 `TestScanFootprint`）：节点数与 T
无关、中间值与峰值按每 token 常数线性增长、状态字节数与已处理长度无关。AF-28 的
数字来自一次性实测脚本（跑完即删），不进 CI——计时在共享机器上不是可靠的门禁。

### 5.104　阶段 B 收官：混合长度与恒定显存

审计的阶段 B 验收表还剩两行：padding/reset（"混合有效长度与重置位置"）与长时间
运行（"固定 batch / 状态连续解码，预热后记录 VRAM"）。两行都补上了。

**混合长度：left padding + 有效起点 reset。** 一条 batch 里长度不同，只有 left
padding 能靠 reset 救回来：真实 token 靠右放，有效区间的第一个位置打 reset，递归
就在那里从零重建，末状态停在有效前缀末，而不是被前面的 pad 段拖走。用例取
`steps=7 / lengths=(7, 4, 2)` 三条样本，逐条与 `batch=1` 单独跑对照——输出比的是
有效段 `y[i, steps-n:]`，状态比的是全部 `state_shapes`，容差 f32 的 1e-5 / 1e-6。

right padding 做不到这件事：pad 接在有效 token 后面，会继续推进卷积窗口与 SSM 状态，
末状态落在 pad 上，对不上单独跑的结果。这不是实现缺陷，是"状态只能向前接线"的直接
后果——要右 padding 就得在末尾再打一次 reset 并把状态取在 reset 之前，接口不支持。

**reset 是逐样本的。** `reset` 是 `(B, T)` 掩码而不是标量，所以同一条 batch 里
一个样本在 t=3 重开一段、另一个样本全程连续时，两者互不影响。用例把样本 0 的切点
放在 3、样本 1 放在 0，分别与各自单独跑对齐（输出与全部状态）。这条是会话隔离的
最小验收：多用户共用一个 batch 时，谁的 session 清了不能波及别人。

**局限要写明**：pad 段的输出仍然是**算出来的**，不是零。reset 让它们不进入有效段的
递归，但图对每个位置都求值；调用方要自己切 `y[i, steps-n:]`。另外 pad 位置的输入
不能是 NaN——NaN 会经过卷积与 scan 的输出，虽然被 reset 挡住的是状态，不是当步的
激活。

**长时间运行：预热后连续 200 步 decode。** 判据用 cupy 默认池的 `used_bytes()`
（当前借出给张量的字节），不用 `total_bytes()`（池向驱动要到的总量，扩容后只增不减）。
口径：`d_model=8 / d_state=3 / d_conv=3 / 2 层 / B=2`，`steps=1` 的整栈流式图，
先跑 10 步热身（编译、池扩容、形状稳定），`free_all_blocks()` 清池，记起点，再跑
200 步，记终点。RTX 2080 Ti 上：

| 指标 | 值 |
|---|---|
| `used_bytes` 起点 | 512 B |
| `used_bytes` 200 步后 | 512 B |
| 增量 | 0 |
| `total_bytes` | 56,320 B |

起点那 512 B 是最后一步的输出/状态张量；跑 200 步后一模一样，说明每一步的中间量
都被复用、没有任何张量按历史长度累积。`total_bytes` 停在 56,320 B 是池的保留策略，
不是泄漏——它在这一套形状上第一次扩容到位后就不再向驱动要内存。

**这个用例能证明什么、不能证明什么。** 它证明"连续 decode 不随历史增长"，也就是
AF-27 那张静态预算表的动态版：状态是常量大小（5,632 B 那一条），图里只有当前一个
token 的中间值。它**不**证明逐字节不涨——阈值给了 4 MiB 余量，只能抓住"每步漏
≥20 KB"这一量级的泄漏；实测增量是 0，比阈值严格，但轮询池计数本身不是分配器审计。
真正的分配器复用另有 5.83 的视图生命周期用例。

**验证**：3 个新用例，2 个在参考后端（left padding 与逐样本 reset），1 个在真机
（显存；本地无 GPU 时跳过）。

本地 1270（994 通过 / 276 跳过），CUDA 真机 1270（1211 通过 / 59 跳过），两边全绿。

### 5.105　审计 AF-31：预算的口径必须跟着执行策略走

`plan_memory` 一直算的是"中间值用完就摘"的峰值，而 `Executor.run_device` 的
`release` 默认是 `False`——全部中间值活到图结束。于是同一张图，预算报 8 MiB，
普通执行实际要 48 MiB。报告点得对：**预算没有等同于真实执行策略**；"任何实现都
绕不过去的下界"这个说法也过强，因为每个 Value 都按独占存储计，视图会被高估，
内核临时区又完全不在里面。

**改法一：把执行策略变成参数。** `plan_memory(graph, symbols=None, release=True)`
与 `run_device` 的同名开关一一对应：`release=False` 时所有值（含图输入）都活到
图结束，`peak_live_bytes == total_bytes`、`reuse_ratio == 1`。`MemoryBudget` 多一个
`release` 字段，`as_dict()` 也输出它——引用数字时必须带上口径，否则"省了 6 倍"
和"没省"说的是两张不同的图。默认仍是 `True`，既有调用方与 `Arena` 的对照数字
因此不变。

**改法二：默认路径真的回收。** `Executor.run` 固定传 `release=True`：它只读图
输出、环境本身不返回，摘掉已经没人再读的中间值不改变任何可观察结果，却能让
分配器立刻复用那些缓冲。`run_device` 的默认值保持 `False`——它的契约就是"交出
完整环境"，摘键会破坏调用方的预期（捕获、逐值调试都靠它）。

**改法三：峰值算法从逐时刻扫描改成事件扫描。** 旧实现对 `-1 … count` 每个时刻
重扫一遍全部值，是 O(节点数 × 值数)；现在每个值只在 `defined_at` 加一次、
`last_used_at + 1` 减一次（半开区间等价于闭区间），排序后扫一遍，O(V log V)。
同一时刻先处理减、后处理加——排序键里负的 delta 在前，正好给出这个顺序：相邻
值之间"前者最后一步就是后者的定义步"时，前者在该时刻已经不再活跃。正确性由
测试里一份逐时刻暴力实现逐项对照（两种 `release` × 对齐与否 × 只算中间值）。

**文档口径同步改写**：模块开头的"下界"换成"指定存储假设下的 IR 估计"，并列出
三条假设与各自的方向性偏差——每个 Value 独占存储（视图高估）、生命周期由 IR
决定（`release` 参数）、只算 IR 里看得见的张量（内核临时区 / 库 workspace /
优化器槽位 / 捕获缓存 / 主机副本都低估）。

**真机对照**（RTX 2080 Ti，6 段 `relu(matmul(v, w))`，1024×1024 f32）：

| 口径 | 静态峰值 | 静态总量 | 池 `total_bytes` | 池 `used_bytes` |
|---|---:|---:|---:|---:|
| `run`（`release=True`） | 8,388,608 | 50,331,648 | 20,972,032 | 512 |
| `run_device`（`release=False`） | 50,331,648 | 50,331,648 | 58,721,280 | 58,721,280 |

静态的 8 / 48 MiB 是 6 倍，真机的 20.0 / 56.0 MiB 是 2.8 倍：差出来的这部分
正好是预算声明"不算"的内核临时区与 cuBLAS workspace（约 12 MB 的固定量）。
`run_device` 的 `used_bytes` 与 `total_bytes` 相等，说明跑完那一瞬间环境里还
压着全部中间值——与 `release=False` 的定义完全吻合。`run` 的 `used_bytes` 是
512 B，那是跑完归还后的当前占用，峰值要看池的 `total_bytes`（池不回退）。

**明确没做**：把 liveness 接进 Arena 的分层通路——执行期仍是 cupy 默认池，
Arena 只在捕获期使用（5.83 / 5.89 的契约把这条路留下了，但没有接）。也没有把
workspace / cache / optimizer 拆成独立字段：它们在预算之外，是"不声称"而不是
"算错"，分别测量要等容量决策真的依赖它们的时候。

**验证**：3 个新用例（`test_liveness.py`）——`release=False` 等于普通执行、
输入也活到最后、事件扫描对暴力扫描。本地 1273（997 通过 / 276 跳过），CUDA 真机
1273（1214 通过 / 59 跳过），两边全绿。

---

### 5.106　审计 AF-32 的图 seal：把"构建后只读"变成契约

`Graph._revision` 的既有约定是：每次 `input` / `add_node` / `output` 自增，节点
属性构建后只读。前半句有代码，后半句只有注释——`Node.attrs` 是普通 dict，
`Value.name` / `Value.type` / `Node.output` 也都能直接赋值。而这张图恰好是两处
缓存的失效依据：`Executor._plan` 按 `(图, revision)` 缓存分组与静态 attrs，
`CudaBackend._graph_plan` 把 `graph.revision` 写进捕获计划的键。只要图还能改，
"计划仍命中但图已经变了"就是一条静默错误通道，表现为某个后端结果对不上。

**改法一：结构入口统一拒绝。** 新增 `Graph.seal()`，置 `_sealed`，并把
`input` / `add_node` / `output` 的首行接到同一个 `_check_mutable()`：
seal 之后一律 `RuntimeError`，消息点明原因是执行计划与捕获缓存依赖结构不变。
`g.matmul(...)` 这类语法糖最终走 `add_node`，因此自动被覆盖。`seal()` 幂等、
返回自身（`g = build().seal()`），`revision` 此后不再变化——这正是缓存可以
长期复用的依据，而不只是"没再改动"。

**改法二：记录与属性也冻结。** `Value` / `Node` 共用一个 `_FrozenRecord` 基类，
`_frozen` 置位后任何字段赋值抛 `AttributeError`。`Node.attrs` 在 seal 时换成
`MappingProxyType`（顶层零复制），并把形状类属性里的 list 就地转成 tuple：
只包只读外壳的话，`node.attrs["shape"].append(...)` 仍能绕过，而形状一变、
执行计划却按旧签名命中，正是 seal 要堵的那条路。`constant` 的 `data` 例外——
常量载荷可能很大，逐层复制不划算，且它只由 `constant` 写入，没有后续改写路径。

**改法三：导出的类型判断放宽。** `serialize._to_json` 原来按 `isinstance(v, dict)`
处理嵌套属性，而 `MappingProxyType` 不是 `dict` 的子类，seal 后的图会导出成
不可序列化的对象。改成按 `collections.abc.Mapping` 判断。

**明确没做**：没有在 `run` / `capture` 里自动 seal。eager 前端是"边算边长"的，
自动冻结会让它在下一次追加节点时失败；要不要冻结图是调用方的决定（训练与部署
路径在构图结束、进捕获之前调一次即可）。也没有把 verifier 的完整性检查塞进
`seal()`——它只负责"改不动"，不负责"图对不对"。

**验证**：3 个新用例（`test_ir.py` 的 `TestGraphSeal`）——seal 后四类结构改写
均被拒且幂等；值与属性冻结（含 list 已转 tuple）；只读访问、执行、序列化往返
与执行计划复用都不受影响（两次 `_plan` 是同一个对象）。本地 1276（1000 通过 /
276 跳过），CUDA 真机 1276（1217 通过 / 59 跳过），两边全绿。

---

### 5.107　审计 AF-32 的后端能力契约：从算子的名字到算子的形态

5.9 的准入面只要求 `capabilities()` 交出算子名集合。它能回答"有没有这个算子"，
回答不了"f16 行不行""是不是只支持最后一维""这个算子能不能进反向图或图捕获"。
只声明名字时，某个用例跑通一种形态就会被读成全面支持——AF-32 第二点要的是让
这份声明能表达这些维度，而不是只把 conformance 的成功次数记下来。

**改法一：扩展而不替换。** 新增 `backends/capabilities.py`：`BackendCapabilities`
仍是 `frozenset` 的子类，元素仍是算子名，`set()` / `len` / `in` / 与普通
`frozenset` 的相等比较全部照旧——CUDA 后端把 `capabilities()` 与静态算子表直接
`!=` 对账的那段代码不用改，`verify(graph, {"add"})` 这类旧调用也不用改。明细不
参与相等与哈希：同一个算子集合可以有不同细粒度，`==` 仍按集合比较；要看细粒度
用 `describe()`。

**改法二：两类语义分开。** `OpCapability` 里 dtype / axis / layout 是**排除项**：
`None` 表示没有声明限制，给了集合就只接收集合里的取值；backward / capture /
deterministic 是**用途承诺**，默认 True，不来自图而来自用途（训练看反向、捕获
看 capture、一致性基准看 deterministic）。把两者压成一个"支持/不支持"布尔量，
"未声明"就会被误读成"已支持"——`refined` / `unrefined` 把这件事显式化。

**改法三：判定接进现有闸门。** `verify` 在算子名都命中之后继续调 `explain_node`：
按节点输出的 dtype、`axis` / `axes` 属性（按输入秩规范化，声明方写 -1 也可以）、
`layout` 属性逐项比对，不满足就出 ERROR。`unsupported_ops` 同步改走
`capability_gaps`，参考后端的拒绝信息因此能说清是"没有这个算子"还是"这一形态
超出声明"。verifier 只做鸭子类型（`getattr(caps, "explain_node")`），不 import
backends，避免 ir 反过来依赖后端包。

**参考后端如实声明。** 49 个 PORTABLE 算子逐个给 `OpCapability`：`backward` 直接
取算子表里的 `vjp is not None`（没有反向规则的不能声明支持反向）、`capture=False`
（参考执行器只有逐节点 numpy 求值，没有捕获路径）、`deterministic=True`（随机性
全部来自显式 seed）。dtype / axis / layout 留 `None`：参考语义只做数学对齐，BF16
借用 f32 承载，本来就不能宣称位级正确。声明按算子表长度缓存（`register` 只允许
新增，长度变化即失效）。

**conformance 区分粒度。** 报告仍统计"N 个算子已声明"，后面补"（仅算子名，未
细化…）"或"M 个带形态明细"；细化声明的后端被拒绝时，refused 原因写成"声明不支持
该形态：dtype f16 不在支持范围 ['f32']"，而不是笼统的算子缺失。只声明名字的后端
行为不变，也不因为没有明细判失败——未细化是"没承诺"，不是"不合格"。

**明确没做**：CUDA / HIP 仍返回普通 `frozenset`，即"只承诺算子名"。它们的 f16、
最后一维归约、分块 attention 等限制散落在内核的运行时检查里，逐条搬到声明上要对账
49 个算子的全部形态，本轮不做；conformance 会把它们标成"仅算子名"——这正是"一个
成功形态不能代表完整支持"的可见处，而不是假装已经细化。AF-32 另外两点（custom op
的 effect 描述、HIP 与 CUDA 大类的拆分）也没有跟着本轮动。

**验证**：8 个新用例（`test_backend_contracts.py` 的 `TestCapabilityContract`）——
参考后端逐个声明且 backward 与 VJP 表一致；带明细后集合语义（相等 / 哈希 / 排序 /
长度）不变；明细里的算子必须在集合里、值必须是 `OpCapability`；dtype 与 axis 在
`verify` 期被拒且 `require` 抛 `VerifierError`；只声明名字时判定不变；反向 / 捕获 /
确定性只能显式查询；conformance 分别报告"带形态明细"与"仅算子名"。本地 1284
（1008 通过 / 276 跳过），CUDA 真机 1284（1225 通过 / 59 跳过），两边全绿。

AF-32 只剩最后一点没动：HIP 继承 CUDA 大类会把假设与缺陷一起扩散。

---

### 5.108　审计 AF-32 的算子效果描述：让"纯函数"成为声明而不是假设

CSE 合并相同签名的节点、DCE 删除没有使用者的节点、常量折叠把全常量子图算掉、
激活重算把每段前向再跑一遍——这四个 pass 都默认算子**没有可观察效果**。这个假设
以前只写在文档里：算子表分不出 `add` 和 `dropout_mask`，`passes/fold.py` 里甚至
维护过一份 `_RANDOM_OPS` 白名单来单独跳开采样。白名单是每次新增随机算子都要记得
改的地方，而"记得改"不是约束。

**改法：把效果写进算子表。** `ir/ops.py` 新增 `Effect` 枚举，三项按约束强度分开：

- `RNG`：输出由 seed / key 完全决定，但语义是"每次重新采样"。**可以**在重算里重放
  （同一 key 复现同一位模式），**不能**折叠（噪声被固化进图）、**不能** CSE 合并
  （两次采样被并成一次）。
- `STATE`：读写图外可变状态。重放会推进或改写它，删除未使用的节点也改变行为。
- `IO`：与 STATE 同属不可重放。

`Op.effects` 是 `FrozenSet[Effect]`，空集即纯函数；`Op.pure` 是这个判断的唯一入口。
`__post_init__` 把插件传进来的 set / list 规范成 frozenset，并拒绝非 `Effect` 的值
（裸字符串 `"rng"` 会被拒——拼错的名字不该静默变成"有效果"）。`UNREPLAYABLE_EFFECTS
= {STATE, IO}` 单独导出，供重算判定用。

**四个消费方各走声明，不走白名单：**

- `passes/cse.py`：非纯节点不参与签名合并（`run` 与 `duplicates` 两处都跳过）。
- `passes/dce.py`：非纯节点是可达性根——它的效果不在数据流上，没有使用者也不能删，
  上游输入跟着保留；`saved()` 只统计纯节点，不把"省下来的"算在效果节点头上。
- `passes/fold.py`：删掉 `_RANDOM_OPS`，改判 `node.op.pure`。算子表里的声明成了
  唯一依据，新增随机算子不需要再改折叠。
- `train/recompute.py`：`CheckpointPlan` 构造期扫一遍节点，命中 `UNREPLAYABLE_EFFECTS`
  直接抛 `ValueError`（带节点序号与算子名）。重放一次 STATE / IO 就是副作用翻倍，
  必须在训练开始前拒绝，而不是跑到中途才发现。`RNG` 放行：dropout 的随机性来自
  显式 key，重放可复现（5.98 的运行期 key 正是为此）。

**只标记了 `dropout_mask`。** 它是内置算子表里唯一真正带效果的：`constant` 是在构图期
写死的数据，不是采样；`step` / `spike` 是确定的阈值函数。参考后端的 `deterministic=True`
（5.107）与此不矛盾——确定性说的是"同输入同输出"，`dropout_mask` 的 seed 不变时确实
如此；RNG 描述的是"语义上要求每次重新采样"，两者是两个维度。

**当时没做的**：HIP 与 CUDA 大类的拆分（AF-32 第四点）。它不能靠 IR 层改动完成，
随后由 5.109 单独处理。

**验证**：11 个新用例。`test_passes.py` 的 `TestEffects`（8 个）——只有 `dropout_mask`
声明效果、`pure` 默认成立、effects 被规范化、裸字符串被拒、CSE 不合并同 seed 的两次
采样、DCE 保留未使用的采样节点及其上游且 `saved()` 为 0、折叠跳过全常量输入的采样、
默认管线跑完采样节点仍在；`test_recompute.py` 的 `TestCheckpointEffects`（3 个）——
显式 seed 的采样在同一份前向上重放逐位一致、STATE / IO 在构造期被拒、`RNG` 不在
不可重放集合里。本地 1295（1019 通过 / 276 跳过）全绿；CUDA 真机 1295（1236 通过
/ 59 跳过）全绿，`tests.test_cuda_backend` + `tests.test_cuda_train` 单独复跑 254 个
用例确认全部实际执行、没有 skip。

---

### 5.109　审计 AF-32 的第四点：把 HIP 从 CUDA 里拆出来

审计原文：「HIP 继承 CUDA 后端大类，并替换成模拟相同容器接口；共有逻辑能少写代码，
也会把 CUDA 假设与 bug 扩散。应逐渐拆出设备无关执行计划、容器/流/分配器协议和供应商
实现，避免一次性重写。」

**问题不是"重复代码"，是错误的方向。** `HipBackend(CudaBackend)` 让 HIP 白拿近三千行
执行计划，代价是基类里每一处"顺手用一下 cupy"都变成了 HIP 的隐性契约。拆分前已经
暴露过三处：

- `dispose()` 用 `isinstance(value, self.cp.ndarray)` 判类型，HIP 的 `DeviceArray`
  不是 cupy 数组，捕获期内存回收对 HIP 静默失效；
- `transpose` 判"输出是不是视图"读 `out.base`，`DeviceArray` 没有这个属性，
  HIP 上一跑就 `AttributeError`；
- `matmul` 直接 `self.cp.cuda.get_current_stream().ptr` 给 cuBLAS 绑流，HIP 上靠
  `hip/runtime.py` 里一个假命名空间 `_CudaNamespace` 顶着。

第三处最说明问题：为了让继承成立，HIP 侧**伪造了 cupy 的 `cuda` 命名空间**。
这不是"少写代码"，是让 HIP 假装自己是 cupy。另外两处则是"bug 扩散"的实例——
其中 `transpose` 那条是基线里就存在的真缺陷，HIP 的 conformance 一直有两个 FAIL。

**改法：提一个设备无关基类，CUDA 与 HIP 变成兄弟。**

`aetherflow/backends/device_backend.py` 新增 `DeviceBackend(Executor)`（3086 行、
110 个成员），放的全部是"与厂商无关的执行计划"：

- **执行计划**：`execute` / `_graph_plan` / `_signature` / `group_nodes` /
  `invoke_group` / `_try_chain`，即"这张图怎么跑、哪些节点能合成一次 launch"；
- **49 个 `_op_*` 派发**与它们依赖的几何工具（广播折叠、卷积 / 池化几何、
  归约的 `_reduce_last_dim`、扫描的 `_scan`、索引的 `_check_ids`）；
- **调优**：`tunable_ops` / `tuning_space` / `set_tuning` / `tuning_config`；
- **容器协议的上半截**：`to_array` / `cast` / `to_host` / `dispose` / `_contig`、
  `resident_parameters`、内核注册与类级能力声明对账（`_register_kernels`）。

厂商只留九个钩子，缺一个就在构造或首次使用时明确报错，而不是猜：

| 钩子 | 作用 | CUDA | HIP |
|---|---|---|---|
| `cp` | 容器模块 | `cupy` | `hip.runtime` |
| `codegen` | 内核方言 | `cuda/codegen.py` | `hip/codegen.py` |
| `sm_count` | 自动 block 的目标块数 | 查设备属性 | 查设备属性 |
| `_kernel` | 编译并缓存一个内核 | NVRTC | hipRTC |
| `_cublas` | BLAS 句柄 | cuBLAS | hipBLAS |
| `current_stream_ptr` | 当前流的地址 | `cp.cuda.get_current_stream().ptr` | `runtime.current_stream()` |
| `new_stream` / `allocator` / `capture` | 流、捕获期分配器、整图计划 | CUDA Stream / `Arena` / `CudaGraphPlan` | hip 流 / `Arena` / `HipGraphPlan` |
| `_pick_cudnn` | 这一步卷积走不走厂商库 | 按 dtype / groups / 捕获态判定 | 默认 `False` |

拆分后三份文件的形状：`device_backend.py` 3086 行 / 110 个成员 / 49 个算子；
`cuda/backend.py` 487 行 / 13 个成员（cuBLAS、cuDNN、pinned 缓冲、内存池统计、
`CudaGraphPlan`）；`hip/backend.py` 322 行 / 8 个成员。CUDA 那份已经没有一个
`_op_*`，不会再有"改执行计划顺手把 HIP 带坏"的路径。

**能力面跟着中立化。** 算子表原本叫 `CUDA_OPS`，HIP 为了复用只能
`from ..cuda.backend import CUDA_OPS`——而 import 那个模块会加载 cupy，HIP 机器上
直接失败。现在冻结集合提到基类叫 `GPU_OPS`（49 个算子），`CUDA_OPS = GPU_OPS`、
`HIP_OPS = GPU_OPS` 各自声明在自己的模块里：能力面属于厂商声明，将来 HIP 要收窄
（比如接 MIOpen 之前先砍掉某个形态）改自己那份即可。构造期对账仍用
`self._OPS`，`_F16_OPS` / `CONV_IMPLS` 也变成可被子类覆盖的类属性，
`conv_impl="cudnn"` 在 HIP 上依旧构造期报错。

**顺手拆掉的假接口。** `hip/runtime.py` 的 `_CudaNamespace` 与模块级 `cuda = ...`
整个删除，改成公开函数 `current_stream() -> int`（没有设备时返回默认流 0）。
`HipBackend.current_stream_ptr()` 就返回它。这一段是拆分最直接的收益：
HIP 不再需要"看起来像 cupy"，因为基类已经不按 cupy 编程。

**拆分过程中修掉的两个真 bug**（都属于"把 CUDA 假设扩散出去"的账）：

1. `memory_stats()` 原来写在 CUDA 一侧。它只依赖 `self._last_arena`，是纯设备无关
   逻辑；挪进基类后 HIP 才有这个契约，之前 WSL 上是
   `AttributeError: 'HipBackend' object has no attribute 'memory_stats'`。
2. `transpose` 用 `out.base is not None` 判视图。改为基类的
   `_shares_storage(result, source)`：先比对象身份，再看 `base`（允许一层间接）。
   HIP 的 `DeviceArray` 没有 `.base` 但共享 `_Buffer`，这个实现两边都对。
   修的是基线遗留缺陷——`tests.test_hip_backend` 的两条 conformance 用例
   （`test_passes`、`test_all_declared_operators_exercised`）在 HIP 上原本一直接不住。

**测试跟着改。** `TestHipInitParity` 原本用 AST 对账"两份 `__init__` 的实例字段
是否一致"——那个测试存在的理由就是继承导致的字段漂移（5.58 的漏写 `tile_w`）。
现在 `__init__` 只有一份，对账没有意义，换成 `TestBackendSplit`：断言
`not issubclass(HipBackend, CudaBackend)`、两边 `__mro__[1] is DeviceBackend`、
HIP 构造前只设置 `{"cp", "codegen", "sm_count"}` 三个字段（其余由基类统一建）。
`test_hip_backend` 的流契约也改了：`current_stream()` 返回地址而不是流对象，
另有一条用例确认 `_Stream` 仍支持基准脚本用到的 `ptr` / `synchronize`。

**没做的**：`DeviceBackend` 里仍留着 `_pick_cudnn` 这个默认 `False` 的钩子和
`cupy` 的措辞（注释与错误信息层面），它们不影响 HIP 的正确性，但"名字里带 cuDNN"
在中立基类里并不干净；真正的做法是把它抽象成 `_pick_vendor_conv()`，等接入
MIOpen 时一起改。另外 HIP 仍不实现 `pinned_array`（设备态训练第一次 step 明确报错）
与 cuDNN 路径，这是实现边界，不是拆分遗留。（前一条已由 5.156 补齐：pinned 暂存与
设备态训练都在 HIP 真机跑通；厂商卷积库仍只作对照，MIOpen 侧见 5.157。）

**验证**：本地 1298 个用例（1022 通过 / 276 跳过）全绿；WSL2 + ROCm 7.2（RX 7650
GRE / gfx1100、DXG 直通、`HSA_ENABLE_DXG_DETECTION=1`）1298 个（1025 通过 /
273 跳过）全绿，`tests.test_hip_backend` 26 个全绿——其中两条是本次修好的；
CUDA 真机（RTX 2080 Ti）1298 个（1239 通过 / 59 跳过）全绿。三处的用例总数相同、
失败数都是 0，拆分没有让任何一侧少跑。

---

### 5.110　审计 AF-09：交错声明输入与序列化的顺序假设

审计原文：「公开构图允许输入与节点交错，序列化却假设输入都在前。先声明 x，再生成
`neg(x)`，之后声明 w、构造加法，原图能运行；`loads(dumps(g))` 报未定义的 v1。
修复：优先实现显式输出 id 的序列化；短期让 `input()` 在节点出现后明确拒绝，避免生成
能运行但不能保存的图。」

**这条在 5.95 修 AF-08 时已经一并解决，只是没有留下证据。** AF-08 之前，节点输出名
靠"当前值个数"在回读时重新生成：反序列化先重建全部输入，w 的 id 从 2 变成 1，于是
`neg` 的输出拿到 `v2`，后面那个引用 `v1` 的 `add` 直接断掉——审计复现的就是这一步。
5.95 之后 `to_dict` 给每个节点写出 `"name"`（[serialize.py](file:///g:/Cloumi/aetherflow/aetherflow/ir/serialize.py#L102-L122)），
回读时输入先重建只是让值 id 的分配顺序变了，节点之间的引用全部按名字解析，与
"输入是否在节点之前"无关。审计给的例子现在逐项成立：

- `to_dict(loads(dumps(g))) == to_dict(g)`，二次 `dumps` 逐字节相同；
- `verify(restored)` 无 issue（原报告是"该节点之前未被定义"）；
- 两个输入的 id 由 `[0, 2]` 重排成 `[0, 1]`，名字与类型不变；
- 参考后端在 `x=[1,2]`、`w=[10,20]` 上的输出 `y` 与回读前逐位相等。

**不采用"让 `input()` 在节点出现后拒绝"。** 那条短期方案的前提是"格式表达不了交错"，
现在这个前提不成立了；加上限制反而要把一条本来能工作的构图路径堵死（eager 前端就是
"用到时再声明"长起来的，`EagerGraph.input` 只是转发）。真正需要保证的是**名字到值
的映射是双射**，这一条由 `input()` 查 `_by_name`、`add_node()` 的自动名避让与显式名
查重共同守住，两个方向都堵上，不需要限制声明时机。

**代价写清楚：值 id 不跨序列化稳定。** 回读后输入占低位、节点顺延，id 可能与构图时
不同。id 本来就不进存档，对外契约是名字与类型，`Graph.find` / `output_names` /
执行入口都按名字工作；把这条从"隐含"改成明说，写进了 `serialize.py` 的模块说明与
`Graph.input` / `Graph.add_node` 的 docstring。

**顺带修掉的陈旧注释**：`models/transformer.py` 里"序列化只记录节点顺序、不记录 SSA
名，因此输入必须全部在节点之前是硬约束"——5.95 之后这句话已经不成立，留着会让人
按错误前提规避。

**用例**：`test_inputs_declared_between_nodes_roundtrip` 补上 id 重排、`to_dict` /
`dumps` 相等与 `verify` 干净三项断言；新增 `test_node_declared_before_any_input_roundtrip`
覆盖更极端的交错——第一个值就是常量节点，此时一个输入都还没声明。

**验证**：本地 1299 个用例（1023 通过 / 276 跳过）全绿。本轮只改注释、docstring 与
用例，执行路径一行未动，因此没有重跑 HIP / CUDA 真机。

### 5.111　审计 AF-15 的收尾：slice 的偏移地址与 HIP 侧的空 dispose

审计原文：「按 Value 释放内存，没有处理存储别名……reshape 可返回共享同一存储的视图，
生产者 Value 最后使用发生在 reshape 节点，原执行器因此立即 dispose 生产者，但视图还要
被后续节点使用……短期让 reshape / slice 等在 arena 路径显式复制可降低风险，但会损失
性能……HIP 有自己的容器和 arena，也需独立检查。」

5.83 给 reshape 与恒等 transpose 建了视图引用计数，峰值没有拿显存来换，但当时留了两处
没覆盖：

1. `slice` 与退化成恒等的 `broadcast_to` 同样返回零拷贝视图，`_op_slice` 没有登记。
   更麻烦的是 slice 的 `ptr` 带偏移（`start > 0` 时是"源地址 + 偏移"），而 `dispose`
   当时按视图自己的地址扣账——即使补上登记，也一条都对不上，源地址会永久挂住。
2. HIP 的 `Arena` 没有 `alias` / `unalias`，`HipBackend` 也没有 `dispose`。捕获期的
   中间值因此从不回收，`reuse_memory` 只是"每个中间值各占一块"。这不会算错，但 5.83
   的复用收益整个没有，捕获大图时会先撞上"流捕获期间 `hipMalloc` 被拒绝"。

**做法**（CUDA 与 HIP 逐条同义，不各写一套）：

| 位置 | 改动 |
|---|---|
| `device_backend._shared_view` | 返视图的算子统一走这里：`reshape` / 恒等 `transpose` / `slice` / 恒等 `broadcast_to`，含 `out is src` |
| `cuda/backend.dispose` | 视图按**最底层 base** 扣账；`out is src` 的第二条 Value 走 `Arena.pending` 判定后扣账 |
| `hip/runtime.DeviceArray` | 新增 `_view` 标记（切片 / reshape / transpose / broadcast 构造时置位） |
| `hip/memory.Arena` | 与 CUDA 同形的 `alias` / `unalias` / `pending`，`recycle` 一并清账 |
| `hip/backend` | `dispose` / `_alias_view` / `_shares_storage` 三个钩子接上 |

**`out is src` 为什么也要记账。** 整段切片（`start=0, stop=n`）与同形状 reshape 返回的
就是源数组本身，但执行器看到的是两条 Value：源那条在算子执行后 remaining 归零、立刻
dispose，而"输出"那条还要被后面的节点读。不记账，地址当场回到空闲表。记账后两次
dispose 各扣一次、归零才还块；第二次靠 `Arena.pending` 认出"这是同一段地址的另一份
引用"，不会把 `free` 再走一遍（那样记账永远归不了零）。

**验证**：

- 离线（不碰驱动）：HIP `Arena` 的记账——两个视图未死时不还块、最后一个死后还、
  还完之后同尺寸分配拿回同一段地址；反方向（视图先死、源后死）直接还块。
- CUDA 真机：新增 `test_slice_source_is_not_reused_before_view_dies`（`t = x*x` →
  `s = slice(t, 2:4)` → `k = y*y`），`start > 0` 让视图地址与源地址不同；auto 与 off
  的 replay 结果一致。
- HIP 真机：同构用例，外加离线记账用例与全量。
- 负向对照：把 `_shared_view` 的登记短路后重跑该用例，确认它失败——证明用例真的踩到
  了别名，而不是恰好没触发。CUDA 上新旧两条用例同时失败（slice 24/24 元素错），
  HIP 上 slice 用例 12/12 元素错；恢复登记后全绿。

**当时没做的**：`slice` 仍然物化（`ascontiguousarray`），没有走"视图 + strided kernel"
的省拷贝路径；审计建议里"短期显式复制"这条对 slice 已经成立，记账只是让物化之前那
一小段视图窗口安全。

**验证数字**：本地 1302 个用例（1024 通过 / 278 跳过）、CUDA 真机 1302（1242 通过 /
60 跳过）、WSL HIP 1302（1028 通过 / 274 跳过），三处全绿；新增 3 个用例（HIP 离线
记账、CUDA 与 HIP 各一条 slice 别名回归）。

### 5.112　审计 AF-19 的收尾：存档的原子替换与加载的提交顺序

审计原文：「完整读取与校验 → 构建待提交状态 → 一次提交。保存用临时文件加原子替换，
降低中断后产生不完整文件的风险。」

5.95 已经把参数写入挪到优化器检查之后，但审计点的两处都还差一步：

1. `save_checkpoint` 结尾是 `np.savez(target, **payload)`——直接写目标。写到一半被 kill
   或断电，留下的是**同名**的半个 `.npz`，上一份好存档已经被截断。下次加载才发现权重
   没了，而错误现场（文件坏了）与根因（保存被打断）隔了一次运行。
2. `load_checkpoint` 里优化器的提交排在流式状态校验之前。优化器自己虽然也是"先校验后
   提交"，但状态模板校验失败时它已经恢复，现场变成"步数与动量已更新、参数与状态还是
   旧的"——续训第一发的更新方向与中断前对不上，损失曲线上几乎看不出来。

**做法**：

| 位置 | 改动 |
|---|---|
| `train/checkpoint._atomic_savez` | 同目录 `mkstemp` → `np.savez` 写文件对象 → `flush` + `fsync` → `os.replace`；任何异常先删临时文件再原样抛出 |
| `train/checkpoint.load_checkpoint` | 顺序改为 读取 → 参数校验 → 状态模板校验 → 预转换 → 优化器提交 → 原地写入 |

临时文件必须落在**目标同目录**：`os.replace` 只在同一文件系统内原子，跨盘会退化成
"先删后拷"，等于没有保护。`fsync` 放在换名前：否则断电后换上的可能是一个还没落盘的
空文件。

加载侧只调顺序不够——`array_module(target).asarray(loaded, dtype=...)` 在设备态下要
上传到显存，这一步同样会失败。所以待写入的数组在提交前全部预转换好，提交之后只剩
不会失败的原地赋值。

**验证**：

- `test_interrupted_save_keeps_previous_file`：mock `np.savez` "先写半截再抛 `OSError`"。
  旧实现把目标截成 `PK\x03\x04partial`；新实现下旧文件逐字节不变，目录里也没有
  `.ckpt-*` 残留。
- `test_save_replaces_atomically_and_leaves_no_temp_file`：正常保存与覆盖已有存档后，
  目录里只有 `ck.npz`。
- `test_failed_stream_state_check_keeps_optimizer_untouched`：坏的流式状态模板 →
  `step_count` 仍为 0、`_m` 仍为空、参数与模板都未被写。
- 负向对照：把 `_atomic_savez` 换回 `np.savez`，第一条失败（旧存档被写成 `partial`）；
  把优化器提交挪回状态校验之前，第三条失败（`step_count` 变成 1）。

**当时没做的**：`os.replace` 之后没有再 `fsync` 父目录。POSIX 上严格保证还需要它——
断电后可能出现"数据在、文件名没落地"；Windows 上拿不到目录句柄，这份代码两边都要跑，
所以只做数据 fsync + 原子换名，并在 docstring 里写明这条边界。

**验证数字**：本地 1305 个用例（1005 通过 / 300 跳过）、CUDA 真机 1305（1245 通过 /
60 跳过）、WSL HIP 1305（1031 通过 / 274 跳过），三处全绿；新增 3 个用例。

---

### 5.113　审计 AF-33：调优的统计口径、捕获失败的作用域与派生的失效版本

审计原文：「`tuning/search.py` 多轮取最小值，但结果字段名为 `median_ms`；……可用于某种
"最佳可达"指标，但不等于典型延迟，容易受择优偏差与机器漂移影响。`_graph_plan` 捕获异常
后将整个后端实例 `_graph_disabled=True`，没有按图隔离失败或保留可查询原因。调优配置、
图计划与已捕获 launch 之间也应有明确的失效版本，而不能只清某一层缓存。」

三处缺口：

1. `tune(rounds>1)` 对每个候选测多次、取**最小**，字段却叫 `median_ms`。名字承诺"典型
   延迟"，值是"最好的那一轮"；两者混用后，`speedup` 会被一次偶然的空闲窗口抬高。
2. 捕获失败把 `self._graph_disabled` 置真，作用域是**整个后端实例**：一张含主机同步的图
   失败，之后所有图都不再尝试捕获；原因只剩一个布尔，诊断时只能靠猜。
3. 三层缓存各清各的：`set_tuning` 只清 `_launch_cache`，`_set_conv_tuning` 一层都不清，
   驻留权重只清 `_graph_plans`。谁忘掉一处，那层就带着旧前提继续命中——block 换了、
   replay 的还是旧 launch 序列。

**做法**：

| 位置 | 改动 |
|---|---|
| `tuning/search.CandidateResult` | `median_ms`（中位数，排名依据）/ `best_ms`（最小值，仅报告）/ `p95_ms` / `spread` 分开；分位用最近秩法，不插值造出没测到的数 |
| `tuning/search.tune` | 排名改看中位数；新增 `tie_epsilon=0.02`，差距不足 2% 不替换先出现的候选，噪声内的领先不再被写成"更快" |
| `tuning/store` | `STORE_VERSION` 升到 2；`TuningRecord` 增可选 `best_ms` / `p95_ms`，并校验 `best ≤ median ≤ p95`——把最小值填进中位数槽会当场拒绝；v1 文件显式拒绝并说明两个口径的差别 |
| `benchmarks/tune_compare` | `REPORT_FORMAT` 升到 2；报告同时写 `median_ms` / `best_ms` / `p95_ms` / `spread` / `best_speedup`，v1 报告拒绝读取 |
| `agent_api` | 配置推荐与快照透出 `best_ms` / `p95_ms`；默认候选时三者都是 `None`（"没测过"不是 0） |
| `DeviceBackend._graph_failures` | 取代 `_graph_disabled`：按"图对象 + revision"记原因，新增只读查询 `graph_capture_failure(graph)`；同图同版本不再重试，其他图照常捕获 |
| `DeviceBackend._invalidate_plans` | 调优覆盖、conv2d 旋钮、驻留权重、`clear_tuning` 全走同一处：`_config_epoch += 1`，同时清 `_launch_cache` / `_graph_plans` / `_graph_failures` |

排名仍要一个"谁更快"的答案，所以不能只把 `best_ms` 改名了事：min 是极端值，候选越多
越容易有一个"侥幸最快"的样本。中位数对单轮抖动不敏感，正是"典型延迟"要的口径；
`best_ms` 留在报告里，回答的是另一个问题——"最好能做到多快"。两者都在，才不需要
用一个数字兼两份职责。

`tie_epsilon` 与 5.63 的结论是同一条：1M 逐元素算子的 `block` 真实差异 ≤3%，淹没在整机
漂移里。既然不可判定，就让默认配置在证据不足时胜出，而不是把"这次谁赢了"写成"谁更好"。

`_config_epoch` 除了触发清表，还被写进捕获计划的键。清表是"事后的"——某条路径新增时
容易漏；键是"事前的"——只要 epoch 变了，旧计划在结构上就不可能命中。失败记录同样随
epoch 作废：换 block 可能改变共享内存预算，"这张图捕不了"的结论未必还成立。

失败的作用域带 `revision`：图被改写后（5.106 的 seal 之前）旧结论不该继续压着新图。
弱引用用来识别 `id` 复用——图被回收后，同一块内存地址上可能是另一张图。

**验证**：

- `test_rounds_report_median_best_and_p95_separately`：样本 `[3, 1, 2]` → 中位数 2、
  最小值 1、p95 3、spread 0.5。旧实现只报"1"。
- `test_median_not_best_sample_decides_ranking`：128 的样本 `[1, 10, 10]`、256 恒定 5 →
  选 256；同时断言 128 的 `best_ms` 确实是 1（"最佳可达更低，但典型延迟更差"）。
- `test_insignificant_gain_keeps_default` / `test_significant_gain_replaces_default`：
  0.99 对 1.0 保留默认，0.9 对 1.0 替换。
- `test_tie_epsilon_is_validated`：越界与布尔值都拒绝。
- `test_best_cannot_exceed_median` / `test_p95_cannot_undershoot_median` / `test_v1_store_is_rejected`：
  口径混填与旧版本都在加载期失败，不猜语义。
- `test_capture_failure_falls_back`（CUDA 真机）：失败后 `graph_capture_failure(g)` 能查到
  原因，且同一张图不再反复尝试。
- `test_capture_failure_is_isolated_per_graph`（CUDA 真机）：坏图失败后，好图仍被捕获一次。
- `test_tuning_change_retries_capture`（CUDA 真机）：换配置后失败记录作废、重新捕获成功。
- 负向对照：把排名改回"按 `best_ms` 取最小"，前两条用例失败；去掉 `tie_epsilon`（改回严格
  小于），不可显著区分那条失败；让 `_graph_failure` 恒返回 `None`，HIP 上用 `HipBackend`
  跑同一段脚本，`bad reason` 变成 `None`、`bad attempts` 从 1 变成 2（该组用例在 HIP 上因
  `CUDA_AVAILABLE` 为假被跳过，`HipBackend` 与它共用同一套 `DeviceBackend` 逻辑）。
- 三层缓存一起失效的 HIP 实测：`execute` 后 `plans=1 launch=1 epoch=0`；`set_tuning` 后
  `0 / 0 / 1`；再执行 `1 / 1`；`set_tuning("conv2d", ...)` 后 `0 / 0 / 2`；
  `resident_parameters` 后 `0 / 0 / 3`。conv2d 那一行在旧实现里完全不清。
- 端到端调优往返（HIP 真机）：`benchmarks.tune --rounds 2 --save` 写出 9 条记录，
  `store_version=2`，逐条满足 `best ≤ median ≤ p95`；再用 `--apply` 读回，命中 9 条，
  输出"对拍通过：所有命中配置与默认结果逐位一致"。`relu_1m` / `gelu_1m` / `silu_gate`
  三个用例的赢家仍是默认候选（应用后 0.99x / 1.04x / 0.94x，都在 2% 噪声内），
  正是 `tie_epsilon` 要的效果——旧口径会从这三个里挑一个"最短样本"写成更快。
- 顺带修掉一条被 HIP 实跑暴露的缺陷：`benchmarks.run._sync` 只认 `cp.cuda`，而 5.109
  拆掉的 `_CudaNamespace` 之后 HIP 的替身模块没有这个属性，所以 HIP 上第一次
  `measure` 就 `AttributeError`（调优 9 个用例全部 `AllRejectedError`）。补上模块级
  `hip.runtime.synchronize()`，`_sync` 按"有 `cp.cuda` 走流同步、否则走设备同步"分派；
  `test_cupy_style_backend_uses_current_stream` / `test_hip_style_backend_falls_back_to_module_sync` /
  `test_backend_without_cp_is_noop` 钉住三条分支。

**当时没做的**：

- 候选之间没有做交替 A/B 与固定热身（每轮测量内部仍有 warmup/repeat，但候选是串行测的），
  p50/p95 也没有接进性能门禁——门禁用的仍是基准报告自己的统计。
- `tie_epsilon` 只回答"这个候选是否值得替换当前赢家"，没有做多重比较校正；候选的先后
  顺序因此仍是先验（默认候选排在最前），它降低的是"把噪声写成提速"的概率，不是
  "选到全局最优"的保证。
- `TuningRecord` 只记 `backend + device + op + signature`，没存设备/驱动/编译器版本。
  审计方向里的"环境指纹"属于性能证据那条线，这里不扩字段。
- 失败记录以图对象为作用域，图被回收后靠弱引用识别；没有容量上限或定期清扫。失败很少
  发生，每次写入时顺带筛掉已回收的条目。
- CUDA 真机因研究机停机未复跑（用户已确认后续再开），本次只在本地与 WSL HIP 上验证。

**验证数字**：本地 1318 个用例（1038 通过 / 280 跳过）、WSL HIP 1318（1042 通过 / 276
跳过），两处全绿；CUDA 真机未复跑，上次 5.112 为 1305（1245 通过 / 60 跳过）。新增 13 个
用例，其中 11 个在本地执行、2 个是 CUDA 真机专用。

### 5.114　审计 AF-23：把"负 A 就稳定"换成可检查的收缩条件

**审计原文**（`AetherFlow_Semantic_Audit_2026-11-01.md` 5.2 节）：有限、合法实数下
`Δ>0, A<0` 只约束了 `0<a<1`。指数实现可先溢出、输入注入与读出仍能放大、`Δ→0` 时
`a→1` 没有统一收缩常数、深堆叠的梯度不由 `a` 决定、f16 里小衰减会舍入成 `a=1`。
只有在 `|a_t|≤q<1` 且 `|b_t|≤M` 时才能给出 `|h_t|≤q^t|h0|+M/(1-q)`。

**缺口**：这条论证之前散在 5.94 / 5.101 的注释里，没有一处能在训练前后跑一遍，把
`q`、`Δ`、`log a`、非有限值与 f16 舍入一起报出来；"初始化覆盖多时间尺度"也只证明了
初始化，不保证训练后仍满足 `q<1`。

**做法**：`models/mamba.py` 新增
`decay_diagnostics(spec, params, *, activation=None, horizon=0, state_init_norm=0.0, dtype=None)`：

| 字段 | 含义 |
|---|---|
| `delta` / `log_a` | `min / p50 / p95 / max`（最近秩，不插值） |
| `q` / `contractive` | f64 下的 `max(a)`；只有严格 `< 1` 才算有收缩常数 |
| `injection_bound` | 给 `activation` 才算的 `max|Δ·B·x|`；否则 `None` |
| `state_bound` / `state_bound_reason` | 只有 `q<1` 且 `M` 有限才给 `q^t·|h0|+M/(1-q)` |
| `f16_a_rounds_to_one` | `dtype="f16"` 时舍入成 `a=1` 的单元数 |
| `nonfinite` | `Δ` / `a` 的非有限计数与前 5 个索引 |

**为什么只覆盖递归。** 审计自己列出了 `a` 管不到的部分：输入注入、读出、门控、残差、
投影。把 `state_bound` 说成"模型稳定"会是另一种过度承诺；报告只声明"SSM 递归在给定
输入上收缩"，并在 `state_bound_reason` 里说明缺哪个条件。

**为什么 M 必须由输入给出。** `b=Δ·B·x`，没有 `x` 就没有幅度上界。给一个默认 0 会让
界看起来永远成立——负向对照正是把 `m` 的初值从 `None` 改成 `0.0`，`state_bound` 立刻
被填上，用例失败。

**已知退化（报告而不是修掉）。** f16 在 1 附近的最小间隔是 `2^-11≈4.9e-4`，初始化里
最长半衰期通道的 `Δ|A|` 只有 `1e-7` 量级，必然舍入成 `a=1`。
`test_decay_f16_drops_the_longest_channels` 把这个事实钉住：诊断要能显示"f32 下收缩、
f16 下这几条通道不衰减"。把状态强制成 f32 是下一步，但它会引入 cast 节点并改变捕获
图的结构，需要真机测数值与性能，本轮不做。

**验证**：8 个新用例——正常初始化严格收缩且不给界、给了输入才算 M、`Δ→0` 时 `q>0.999`
被如实报告、f16 舍入计数、NaN 输入的非有限计数与索引、`Δ` 下溢到 0 时 `contractive`
为假、f16 丢长通道、形状校验。负向对照两组：把 `q < 1` 放宽成 `q <= 1`，
`test_decay_zero_delta_is_not_contractive` 失败；无输入时把 `M` 填 0，
`test_decay_diagnostics_reports_contractive_recursion` 失败。

**当时没做的**：状态强制 f32（需要真机验证数值与性能）；把诊断接进 `Trainer` 的每步
日志（会引入主机同步，改变训练计时口径）；多层堆叠的联合谱半径（这里只算单层递归，
残差与层间耦合不在界内）；`q` 是 f64 下的值，图里 `a` 的实际 dtype 仍跟随输入。

**验证数字**：本地 1326 个用例（1046 通过 / 280 跳过）、WSL HIP 1326（1050 通过 / 276
跳过），两处全绿；CUDA 真机未复跑。

### 5.115　审计 AF-25：Mamba 风格变体的口径，与官方扫描公式的对拍

**审计原文**（`AetherFlow_Semantic_Audit_2026-11-01.md` 5.4 节）：当前 dt 是逐通道
`dt_w·x+dt_b`，官方 Mamba-1 先把输入低秩投影到 `dt_rank` 再投影回 `d_inner`；构件
还在读出后加了 RMSNorm。作为自定义设计可以成立，但不是"复刻"，不能承诺原版权重
兼容或性能继承。同时要避免两个反向误判：`b=Δ·B·x` 不等于完整 ZOH 输入矩阵，但
官方参考实现也采用这一形式；不能仅凭这一项判定"不是真 SSM"。

**缺口**：模块 docstring 只写了"证明 IR 能表达它"，没有把三处偏离与"不承诺兼容"
写成可检查的口径；也没有一处能把"我们的扫描公式与官方参考一致"这件事跑出来，
于是两种误判（以为等价、或以为公式错）都没有证据可依。

**做法**：

| 位置 | 内容 |
|---|---|
| `models/mamba.py` 模块说明 | 明确"受 Mamba 启发的变体"、三处结构差异、不承诺权重与性能；指出公式一致不等于整块等价 |
| `mamba_reference.official_selective_scan` | 按 mamba-ssm `selective_scan_ref` 的语义逐步递推，形状沿用 `(B, D, T)`，支持 `h0` / `reset` / `D` / 门控 |
| `mamba_reference.map_official_params` | 只映射 `A_log → A_raw`（两者都是 `-exp(raw)`）与 `D → D_skip`，其余列为 `unmapped` |
| `mamba_reference.from_official_checkpoint` | `unmapped` 非空就拒绝整块迁移，错误信息给出三处结构差异 |

**为什么对拍不算"证明等价"。** 对拍覆盖的是 `a=exp(Δ·A)`、`b=Δ·B·x`、`y=C·h+D·u`
与门控的递推语义；dt 的低秩投影、读出前的 RMSNorm、层间残差都不在这条公式里。
所以测试命名为"扫描公式一致"，而不是"块等价"。

**验证**：7 个新用例——无状态对拍、带 `h0` 与 `reset` 的对拍、门控只在最后作用一次、
参考实现的形状校验、只映射 A/D、整块迁移被拒绝、映射的形状校验。负向对照两组：
把参考实现的 `a_t` 翻成 `exp(-Δ·A)`，2 个对拍用例失败；跳过 `unmapped` 检查，
整块迁移用例失败。

**当时没做的**：审计要求的消融（相同数据、参数量、训练 token、优化器与算力下，
比较低秩 dt、RMSNorm、门控各自的贡献）——它需要多组训练预算，本轮只把可对拍的
baseline 与口径落地；官方权重的实际迁移数值验证（本机没有 mamba-ssm 导出的权重，
且映射规则本身已拒绝不等价项）；性能继承的对比（受真机停机限制）。

**验证数字**：本地 1333 个用例（1053 通过 / 280 跳过）、WSL HIP 1333（1057 通过 / 276
跳过），两处全绿；CUDA 真机未复跑。

### 5.116　审计 AF-30（前向 mask）：因果 mask 不再被复制成 (BH,S,T)

**审计原文**（`AetherFlow_Semantic_Audit_2026-11-01.md` 第 8 章）：mask 被广播到
`(BH,S,T)` 后连续物化；共享的因果 mask 可能因此重新引入大块二次内存；反向仍以
`O(BHST)` 中间量重建 `qk^T` 与概率；f16 融合注意力不支持。

**缺口**：`_op_attention` 对所有 mask 都执行 `broadcast_to(...)` +
`ascontiguousarray(...)`。`(1,1,S,T)` 的因果 mask 因此被复制成 `BH` 份——B=8、H=4、
S=T=512 时多出 31 MB，完全由广播产生。

**做法**：

| 位置 | 内容 |
|---|---|
| `lower/kernels.py` 的 `attention_fwd` | 新增 `mask_shared` 参数；`mrow` 的偏移改成 `(mask_shared ? 0 : bh) * S + s` |
| `device_backend._mask_shared_over_heads` | 最后两维正好是 `(S,T)`、其余维全为 1 才共享；否则仍走广播复制 |
| `device_backend._op_attention` | 共享时只保留一份连续 `(S,T)`，`mask_shared=1`；无 mask 时 `mask_shared=0` |
| `cuda/codegen.py` 的 `ATTENTION_SRC` | 同步参数与偏移。它是"不参与生产的手写对照物"，只被 ASCII 检查读取 |

**为什么不做成"总是共享"。** 随 batch 或 head 变化的 mask 必须各自寻址；`(1,S,1)`
这类可广播但不是 `(S,T)` 的形状也不能按共享处理，否则内核会读错行。判断严格，
两条路径都由用例钉住。

**验证**：
- 本地 3 个判断用例：因果 mask 共享、按 `(b,h)` 变化的 mask 不共享、可广播但形状不符
  仍走复制
- HIP 对拍：`(1,1,64,64)` 共享路径与 `(2,3,64,64)` 复制路径，二者都与参考一致且互等
- HIP 分配量：同一注意力（B=8、H=4、S=T=512、D=E=16）共享路径分配 5.0 MB，全量
  mask 分配 36.0 MB，差 31 MB = `(32-1)×512×512×4`，与 `BH-1` 份一致
- 负向对照：把条件写反成 `mask_shared ? bh : 0`，HIP 对拍 82% 元素不一致

**一次改错位置的记录**：第一次改的是 `cuda/codegen.py` 里的 `ATTENTION_SRC`，它不参与
生产——生产内核由 `lower/kernels.py` 的 AST 构建。HIP 对拍立刻失败，才定位到两份源码
分叉。现在对照模板已同步；是否直接删掉它另议，删改会影响 `TestCodegenDtypes` 的
ASCII 检查清单。

**当时没做的**：反向的 `O(BHST)` 中间量（`ir/grad_rules.py` 重算 `scores`/`probs`，
要专用反向内核并真机测显存）；f16 融合注意力；mask 与 dropout 组合的显存测试。

**验证数字**：本地 1337 个用例（1056 通过 / 281 跳过）、WSL HIP 1337（1061 通过 / 276
跳过），两处全绿；CUDA 真机未复跑。

### 5.117　审计 AF-30（反向）：先量化，结论是不写专用内核

**审计原文**：反向仍以 `O(BHST)` 中间量重建 `qk^T` 与概率；若改用专用反向内核或
分块策略，必须重新量化峰值，不能假定等价于线性注意力。

**量化**（`B=8, H=4, S=T=512, D=64`，一个 `(BH,S,T)` 缓冲 = 32.0 MB，反向图 26 个节点）：

| 口径 | 总量 | 峰值 | 折合 `(BH,S,T)` 份数 |
|---|---|---|---|
| `release=False`（不复用） | 439.1 MB | 439.1 MB | ≈13.7 |
| `release=True`（按最后一次使用回收） | 439.1 MB | 108.1 MB | ≈3.4 |

**真实路径用的是后一种口径**：`Executor.run` 固定 `release=True`
（`runtime/executor.py:153`），`capture` 也固定 `release=True`
（`cuda/backend.py:350/362`、`hip/backend.py:236/249`）；设备态训练的 replay 用的是
捕获时分配好的那批缓冲，峰值由捕获口径决定。所以 439 MB 是"直接调 `run_device` 且
不传 `release`"的上限，不是训练峰值。

**为什么仍不写专用内核**：

1. 朴素下界是"输入 + 输出"7 份（`q/k/v/dy/dq/dk/dv`，224 MB），而复用后只有 3.4 份
   （108 MB）——输入与输出在生命周期上错开，可以共用同一段内存。手工分块很难做得更好。
2. 分块反向的收益主要在计算（省一次 `scores`/`probs` 重算），代价是新增一个只有
   CUDA/HIP 可用的算子、参考实现、VJP 注册和两套测试。组合展开让"任何能跑
   `matmul`/`softmax` 的后端"都能反向，这是 `_attention` 的既定选择。
3. 审计要求的是"必须量化"，不是"必须写内核"；量化结果不支持写。

**补的回归**：`tests/test_liveness.py::TestAttentionBackwardBudget` 钉住"复用至少省掉
三分之二"，防止以后把二次缓冲改成活到图结束——那等于悄悄退回 439 MB。

**未做**：f16 融合注意力仍不支持。若以后做，必须按真机重新量化，不能沿用本节的
108 MB——分块内核的临时块与这里的 liveness 复用不是一回事。

**验证数字**：本地 1338 个用例（1057 通过 / 281 跳过）、WSL HIP 1338（1062 通过 / 276
跳过），两处全绿；CUDA 真机未复跑。

### 5.118　审计 AF-29：训练状态的版本凭证，与 step 边界上的异步快照

**审计原文**：`Trainer.step` 用 ParameterStore 的 NumPy 参数组装 feed，前向 execute
返回主机输出；反向 execute 再重放前向、返回主机梯度；优化器在 NumPy 上更新。方向是
训练态 DeviceParameterStore、设备端 optimizer/slots、execute_device、显式传输与同步
边界；CPU checkpoint 为异步快照。验证更新后的权重、驻留参数、捕获缓冲与优化器引用
属于同一版本。

**前三条已在 5.84～5.88 落地**（参数与槽位常驻显存、前向与反向各一张捕获计划共用同
一组权重缓冲、优化器原地更新、一步只有损失标量一次 D2H；分段设备态同样）。本轮补的
是审计最后那句"验证"与"异步快照"——收益依赖一条隐式契约，而契约此前只写在注释里：

1. `Trainer.training_state()`（`TrainingState`）报 `steps` / `optimizer_step` /
   `device` / `host_in_sync` / `params`。两个步数必须分开：`grad_hook` 返回 `None`
   的步只前进、不更新参数，用一个数字表示"训练到哪了"会掩盖这件事。
2. `Trainer.verify_device_state()` 把契约变成可调用的一次检查。只比对象身份与数组
   模块，不做 D2H，因此可以每 N 步自检。三条链：优化器 `params` 里的数组是不是驻留
   参数本身；前向、反向（以及分段模式下每一段的前后向）计划里该名字的缓冲是不是同
   一块；优化器槽位与参数是否在同一个数组模块。任何一处分叉都抛 `RuntimeError` 并
   **列出全部问题**，因为"换掉一份数组"通常不止一处。为此 `CudaGraphPlan` /
   `HipGraphPlan` 新增 `buffers` 属性（字典副本、数组引用——是给身份校验用的，不是
   拿来换缓冲的）。
3. `Trainer.snapshot()` 返回 `TrainingSnapshot`：在 step 边界把参数与优化器槽位一次
   性 D2H 并拷贝。`save` 同步写、`save_async` 交给模块级单线程池并返回 `Future`。
   **异步的是写盘，不是拷贝**——拷贝必须发生在调用方选定的时刻，否则拿到的是一份跨
   若干步拼出来的权重。写失败落在 `Future` 上，不打断主线程的下一步。
4. `Trainer.save_checkpoint(path, blocking=...)` 是便利入口。设备态下不必先
   `sync_to_host()`：那一版会覆盖 `store`，破坏"训练期间 `store` 是旧值"的契约。
   快照还会在 meta 里补 `trainer_steps`（与 `optim/step` 分开记录）。

**`device_state` 默认仍是 `False`**：设备态需要 `capture` / `to_array` / `to_host`
的后端，参考后端没有这些方法；改成默认开会把"换后端才能开"从一句构图期说明变成运行
期异常。审计要的是能力，不是默认策略——`training_state().device` 让调用方一眼看出当前
走的是哪条路。

**负向对照**（三组，每组都确认对应用例变红后还原）：关掉"优化器引用"检查 →
`test_verify_device_state_catches_a_detached_optimizer` 失败；关掉"捕获缓冲"检查 →
`test_verify_device_state_catches_a_replaced_capture_buffer` 失败；把 `_host_copy_of`
的拷贝去掉 → `test_snapshot_does_not_alias_live_parameters` 失败。

**验证数字**：本地 1350 个用例（1069 通过 / 281 跳过）、WSL HIP 1350（1074 通过 / 276
跳过），两处全绿；CUDA 真机未复跑。12 个新用例。

### 5.119　审计 AF-34（第一块）：SSM 语言模型的图内闭环

**审计原文**：能表达一个 block 不等于具备语言模型闭环。`build_mamba_model` 只是把 N 层
SSM 串起来，尚无词表嵌入、输出头、位置/无位置编码口径、EOS/停止符、采样策略、KV/SSM
状态缓存的 API、tokenizer、chat template、padding/label mask、工具调用的受约束格式、
真实语料上的训练验证与 checkpoint 的配置绑定。

本轮把闭环拆成**图里能验收的部分**，落成 `models/lm.py`：

    嵌入       embedding(table, ids)                    复用已有算子
    主干       build_mamba_layers → N 层 Mamba          复用 models.mamba 的层实现
    末端       rmsnorm
    词表头     matmul(flat, W)；tie_weights 时 W = embed^T（参数少一组，错配更少）
    损失       masked_lm_loss：只除以被选中的 token 数
    解码       prefill 整段 → with_steps(1) 逐 token，状态经 carry_state 带出
    采样       greedy / temperature / top-k / top-p / 重复惩罚，主机侧 numpy
    束搜索     num_beams 条假设进图的 batch 维，状态按父假设重排（5.194）
    停止       EOS 或 max_new_tokens

**`build_mamba_layers` 是这轮唯一改到既有模块的地方**：原来的 `build_mamba_model`
自己创建 `x` 图输入，而语言模型要把 embedding 的输出（一个算子的结果）接进层栈。
新函数只做"从已有 Value 出发叠层"，键保持原名（`_run_layers` 按 `l0_conv0` 查找），
图输入名才带 `prefix`——这也是 `prefix` 用例存在的原因：两套名字混用会让两张图
互相覆盖输入。

**几处显式取舍**（都写进了 docstring 与用例）：

- ``masked_lm_loss`` 的分母是有效 token 数，不是 ``B*T*V``。全 padding 的一批
  分母加 ``1e-12``（f32 下对任何 ``>=1`` 的计数不可见），损失是 0 而不是 NaN：
  "这一批没有可学的东西"不该让整个训练步变成非有限值。
- ``sample_next_token`` 只在主机上跑。采样依赖随机数与词表分布，放进图里会把
  "可复现"绑到后端实现上；``generate`` 里每次只把最后一行的 logits 拷回主机。
- ``generate`` 只接受 ``batch == 1``。批内多序列停止点不同，逐 token 解码会退化成
  "最短序列决定整批"——那是要显式设计的另一个问题，不该在这里悄悄发生。
- 不在本轮：tokenizer、chat template、工具调用的受约束格式、真实语料训练、
  checkpoint 与 spec 的绑定。它们是数据/协议层与产品层，不该借"多一个算子"来
  假装已经具备。

**负向对照**（两组）：把 ``weighted = mul(nll, mask)`` 改成 ``weighted = nll`` →
`test_padding_rows_do_not_contribute` 失败（1.69 vs 2.06）；把 EOS 检查挪到 append
之后 → `test_eos_after_some_tokens` 得到 ``[5, 1]``（期望 ``[5]``）。

**验证数字**：本地 1366 个用例（1085 通过 / 281 跳过）、WSL HIP 1366（1090 通过 / 276
跳过），两处全绿；CUDA 真机未复跑。16 个新用例，全部在参考后端上跑（不需要 GPU）。

### 5.120　审计 AF-34（第二块）：词表双射、chat 模板、batch mask 与配置绑定

5.119 把图内闭环补齐后，产品层还缺四件"错了不会报错、只会安静地错位"的东西。
本轮逐件给出可执行的不变量：

1. **`ByteTokenizer`（`models/lm_text.py`）**：词表 = 256 个字节 + `<pad>=0` /
   `<eos>=1`，字节 `b` 的 id 是 `2 + b`。没有 BPE 合并表，所以
   `decode(encode(s)) == s` 对任意 UTF-8 文本成立；代价是一个汉字 3 个 token。
   `decode` 跳过 pad、在第一个 eos 处停止，越界 id 与"截断的 UTF-8"都拒绝——
   后者给乱码比给一句"编码不完整"更有害。
2. **`ChatTemplate`**：纯文本拼接，**不新增特殊 token**。新增特殊 token 会改词表，
   让旧 checkpoint 的嵌入表整体错位；模板只负责把 `[(role, content)]` 排成一段
   文本，`add_generation_prompt=True` 时补 assistant 前缀，那正是 prefill 的输入。
   不校验轮次交替（那是具体产品的会话规则），只拒绝空内容与未知 role。
3. **`lm_batch`（`train/lm.py`）**：变长序列 → `(ids, targets, mask)`，**left
   padding**。理由与 5.104 同源：Mamba 的状态沿时间推进，右侧补 pad 会让 pad 继续
   推进状态、末状态与单独跑该序列不一致；左侧补 pad 时真实 token 仍连续落在末尾。
   `targets` 与 logits 同形（最后一位与 padding 位全 0、mask=0），因此不需要对
   logits 切 `slice`——少一个节点，也少一条"切片边界"的隐式约定。`labels="index"`
   改为返回 `(B,T)` int32 的扁平下标（`(b*T+t)*V+y`），配合
   `masked_lm_loss_indexed` 使用，内存少 `V` 倍（5.188）。
4. **`LMSpec.meta()` / `check_meta()`**：存档里的配置凭证（形状 + 词表约定 + eps，
   键带 `lm_` 前缀）。显式绑定：存档格式不偷偷塞配置，读的一方必须拿 spec 核对；
   缺项按"未绑定"直接拒绝。布尔先于整数判断——`bool` 是 `int` 的子类，顺序写反
   会把 `False` 当成 `0` 放过去。

**负向对照**（两组）：把 `start = total - len(row)` 改成 `start = 0`（即右对齐）→
3 个 batch 用例变红；把 `decode` 里的 eos `break` 改成 `continue` →
`test_decode_skips_padding_and_stops_at_eos` 得到 `"ok!"`（期望 `"ok"`）。

**仍未做**：工具调用的受约束格式（需要语法层，不是模板能解决的）、真实语料上的
训练验证（需要训练预算与数据）、采样时的重复惩罚/长度归一化等解码策略。

**验证数字**：本地 1391 个用例（1110 通过 / 281 跳过）、WSL HIP 1391（1115 通过 /
276 跳过），两处全绿；CUDA 真机未复跑。25 个新用例，全部不依赖 GPU。

---

### 5.121　审计 AF-30（第三块）：f16 的融合注意力

5.116 解决了"mask 被复制成 `(BH,S,T)`"，5.117 量化了反向的显存；剩下的最后一条是
f16：`_F16_OPS` 的默认集一直写作 `_OPS - {"attention"}`，理由是分块内核把共享内存
布局与指针类型都写死成 `float`。这一条本轮补上。

**改的是模板，不是算法。** 现在的 `attention_fwd` 是混合精度：

- `q / k / v / mask / out` 五个指针按 dtype 声明（f16 下是 `unsigned short*`），
  参数上的 `type_name` 因此去掉——它们是"存储"，不是"计算"；
- 读一律走 `Load`（f16 展开成 `af_ld_h(qs[d])`），写回一律走 `Store`
  （f16 展开成 `orow[e] = af_st_h(...)`）；
- 共享内存 `smem`、行最大值 `m`、归一化因子 `l`、分块累加器 `s_acc`、点积
  `dot` 与 `acc` **全部仍是 float**。f16 只省存储与带宽；softmax 的分母和
  输出累加如果落在半精度里，行长度一过 2048 就开始丢有效位。

**新增一项 L2 能力**：`Let.type_name` 支持 `%T%` 占位符，打印时按方言的存储类型
展开。`const float* mrow` 这种"指向存储类型的指针"没法在 f16 下照抄——mask 与 q
同 dtype 是 IR 的契约（`ir/ops.py` 的 infer 已经强制），指针必须跟着变宽窄。
`qs / vbase / orow` 三处 `base + offset` 改用已有的 `Ptr` 语句，f32 打印逐 token
不变；`mrow` 是条件表达式，走 `%T%`。

**生产侧**（`backends/device_backend.py`）：

- `out = cp.empty(..., dtype=q.dtype)`，不再是 `cp.float32`；
- 内核缓存键从 `"attention"` 改成 `("attention", self._kind(q))`；
- 渲染时传 `q.dtype`，f16 走 `_F16_HELPERS`（`af_ld_h` / `af_st_h`）；
- 运行期再校验一次 mask 与 q 同 dtype。IR 层已有契约，但直接调后端可以绕过；
  内核按同一宽度读两者，混用不会报错，只会静默读错位宽——这类错误必须在边界拒绝。
- `_F16_OPS` 默认值改为 `_OPS`：**49 个算子全部支持 f16**。

**参考实现与内核的差异**：参考后端是逐位 f16 运算（`matmul` → `softmax` → `matmul`
每一步都舍入），内核只在写回时舍入一次。所以对拍容差取 5e-2（约两位有效数字），
这不是实现错误而是舍入位置不同——同一个理由 5.33 给逐元素算子取 1e-2 时已经写过。

**负向对照**（两组）：

1. 输出缓冲退回 `cp.float32`：f16 内核按半字写进 f32 缓冲，`test_plain` 的元素
   几乎全错（不是"精度差"，是位模式对不上）；
2. `mrow` 退回 `const float*`：HIP 编译期报 `unsigned short*` → `float*` 类型不匹配，
   两个用例都在 `HipBackend().execute` 里失败。

两组各自变红后还原。

**验证数字**：本地 1396 个用例（1111 通过 / 285 跳过）、WSL HIP 1396（1118 通过 /
278 跳过），两处全绿；CUDA 真机未复跑（研究机停机）。5 个新用例：1 个 emit 级
（断言 f16 源码里指针 / 读 / 写都是半精度，而 `smem` 与 `acc` 仍是 float）、
2 个 HIP 真机对拍（普通与 dropout）、2 个 CUDA 真机对拍（同形状，走同一份 L2 描述
与 `_F16_HELPERS`，但本机没有 CUDA 设备所以只收集不执行）。另外本机 torch 2.4.1
不支持 `enable_gqa`，`test_frontend` 的那条用例会直接 TypeError——它依赖 torch ≥ 2.5，
与本次改动无关（用 Python 3.14 的 torch 2.13 跑同一份代码是绿的）。

**仍未做**：f16 的反向图数值（`cast` 的 VJP 已能透传，但半精度梯度在长链上的
累积误差没有量化）、mask 的结构化描述（现在是 `(S,T)` 加性偏置，稀疏的因果 mask
仍按稠密处理）。

---

### 5.122　审计 AF-34（第三块）：工具调用的受约束输出与执行回路

审计的原话是"工具调用需要受约束格式输出与执行回路，不能靠多一个算子自动获得"。
5.119 给了 LM 的图内闭环，5.120 给了词表与模板；剩下的这条是**解码侧**：模型
输出必须是能被解析的结构，而不是"提示词里写了'请输出 JSON'就指望它听话"。

**为什么这件事在字节级特别便宜。** ``ByteTokenizer`` 的词表就是 256 个字节
（5.120），于是"这一步允许哪些 token"与"这一步允许哪些字节"是**同一个集合**。
换成 BPE 词表，就必须为每个 token 保存它的字节串、在每一步判断"这个 token 的
字节能不能接在当前前缀后面"（前缀树或逐 token 比对），复杂度差一个量级。这不是
说字节级词表更好，而是说这一层的成本取决于词表的形状——选字节级时它几乎是免费
的，选 BPE 时它是一套独立的基础设施。

**文法**：`JsonGrammar` 是一个增量状态机，只回答 `allowed() -> frozenset[int]`，
`feed(token)` 推进。状态由三部分组成：

- **标量阶段**：`string` / `escape` / `unicode` / `number` / `literal`；
- **容器栈**：每项是 ``("{", phase)`` 或 ``("[", phase)``；
- **每项的 phase**：对象是 `key` → `colon` → `value` → `comma`，数组是
  `first` → `value` → `comma`。

`key` 与 `key_more` 必须分开：刚打开的对象允许 `}`（空对象），逗号之后的对象
不允许（否则 `{"a":1,}` 这种尾逗号会被放行）；数组的 `first` / `value` 同理。
**数字的结束是隐式的**——JSON 没有终止符，所以 `complete` 不能只看"容器栈空"，
还要看数字处在哪个阶段：`12` 已经完整、`12.` 与 `1e` 不完整。这一条在第一次
跑测试时直接暴露成"写完整数没有收尾"。

只生成**紧凑** JSON（不允许空白）：空白对模型没有信息量，却会让合法字节集合多
出一整类分支。字符串内容限定为可打印 ASCII，因为执行回路最终要把字节流解码成
`str`，非 UTF-8 字节会在那一步失败——与其让文法假装支持，不如在这里划清边界。

**接进 generate**：`generate(constraint=...)` 每步把 `allowed()` 之外的 logits
置 `-inf`，采样后再 `feed`。这要求 `sample_next_token` 的输入检查改一下：它原来
拒绝一切非有限值，而 `-inf` 正是"合法的屏蔽"。现在分开处理——拒绝 NaN 与 `+inf`
（前向已坏），允许 `-inf`，并显式拒绝"整行都被屏蔽"（否则 softmax 会得到 NaN）。
两个额外的拒绝：`eos_id` 与文法用的 EOS 不一致时报错（文法完成时只允许 EOS，
改成别的停止符会死循环）；`max_new_tokens` 用完而文法没走完时报错——**半截 JSON
比报错危险得多**，调用方会把它当成功结果去 `json.loads`。

**执行回路**：`Tool` 声明名字、实现与参数表（必填 / 可选，每项带期望类型），
`ToolRegistry.call(text)` 做四步——`parse_tool_call` 解析（只认
`{"name": str, "arguments": {...}}` 这一个形状，多余字段、类型不符都拒绝）→
查表 → `Tool.check` 校验参数 → 调用。`bool` 在 Python 里是 `int` 的子类，所以
声明 `int` 时必须显式排除 `True`；声明 `float` 时接受整数，因为 JSON 里没有
`1` 与 `1.0` 的类型区别。

**测试**分三层，共 29 个用例：

1. 正向闭包：21 条合法 JSON（数字的各种形态、转义与 `\u`、空容器、嵌套、真实的
   工具调用形状）逐字节喂进去，每一步的字节都必须在 `allowed()` 里，且最终
   `complete`；
2. 反向：根状态拒绝 `}` / `]` / `:` / `,` / 空白，对象拒绝非字符串 key 与缺失的
   冒号，字符串拒绝未转义的控制字符与非法转义，数字拒绝前导零与 `1.`，字面量
   必须逐字符走完；pad 永远非法，未写完的 JSON 遇到 EOS 直接报错；
3. 随机游走：从 `allowed()` 里随机取（12 个种子），走到完成后的文本必须能被
   `json.loads` 解析——生成器不知道 JSON 是什么，只按集合选字节。

**负向对照**（两组）：

1. 去掉 `_pick` 里的 logits 屏蔽：约束变成摆设，`feed` 立刻拒绝越界 token，
   两个生成用例失败；
2. `_after_value` 不区分 key 与值：多键对象的第二个 key 之后被期望逗号而非冒号，
   `{"name":"add","arguments":{...}}` 这种真实形状的第 25 个字节被拒。

两组各自变红后还原。

**验证数字**：本地 1425 个用例（1140 通过 / 285 跳过）、WSL HIP 1425（1147 通过 /
278 跳过），两处全绿；CUDA 真机未复跑（研究机停机）。

**仍未做**：文法只覆盖 JSON——工具参数的**语义**校验在 `Tool.check` 里，文法
管不了 `{"a": "x"}` 这种"JSON 合法但调用非法"；真实语料上的训练与验证集（审计
AF-34 的最后一条）需要数据与训练预算，本轮没碰；解码策略（重复惩罚、长度归一化）
审计没有要求，不自己加需求。

**5.189 起前半句已收窄**：`Tool` 接受完整 JSON Schema 子集并输出标准
`{"name","description","parameters"}`，`ToolCallGrammar` 按注册表合成——工具名、
参数键名、参数类型、必填 / 可选顺序都在解码期限死，`{"name":"sub"}`、
`{"a":1,"c":2}`、`integer` 参数写 `1.5` 都写不出来；剩下的语义校验只有 `enum`
取值（文法只认类型，不认取值），以及"嵌套对象参数"这条明确不支持、构造期拒绝并要求
改用 `JsonGrammar`。

---

### 5.123　阶段 C（第一块）：调度器、梯度累积与 loss scaling 接进训练循环

阶段 C 的第一句是"落实 f32 master/slots、设备端 optimizer、梯度累积与裁剪、
AMP scaler、RNG key、调度器、原子 checkpoint"。先做了一次盘点，结论是这些设施
**早就都在**，但它们是一组**散装的纯函数**：`schedule.py` 只回答"第几步该用多大
学习率"、`grads.py` 只回答"怎么累积与裁剪"、`amp.py` 只回答"怎么缩放"，谁来按
什么顺序调用它们，一直留给使用者自己写。这一层的缺陷不会报错——漏调 `apply_lr`
只是学习率恒为初值，漏调 `scaler` 只是梯度下溢，曲线看起来都只是"学得慢"。

所以本轮不新造设施，而是把三条路径固化进 `Trainer`：`schedule=`、
`accumulate_steps=`、`scaler=`。默认值分别是 `None` / `1` / `None`，老代码的
行为逐位不变。

**顺序是这一节的全部内容。** 一次更新里依次是：

    unscale（每个微批）→ accumulate → grad_hook → apply_lr → optimizer.step

- **反缩放必须在累积之前**：每个微批的 scale 可能不同（溢出一次就减半），
  混进同一个累加和就是两种尺度的梯度相加。
- **累积必须在钩子之前**：钩子里的裁剪上限要对得上**有效 batch**。反过来的话
  每个微批各裁一次到 `max_norm`，窗口越长总范数的实际上限越松（`grads.py`
  开头就写了这条，只是以前要靠使用者自己记住）。
- **调度必须在真正更新之前**：`apply_lr` 读的是 `optimizer.step_count`，也就是
  更新次数；累积窗口内的几次反向不会让 warmup 提前用完。

**`accumulate_steps == 1` 时不建累加器。** `GradientAccumulator` 固定用 f32
缓冲（f16 累加会丢精度），一旦让所有路径都过它，f64 图的梯度会被悄悄降到 f32。
单步路径保持原来的 `hook → step`，只有 `accumulate_steps > 1` 才走新链路。

**scaler 与累积叠加时的两个坑。**

1. 更新时不能再调 `GradScaler.step` —— 它在里面又除一次 scale，这一步的更新量
   会缩小 `scale` 倍。新增 `GradScaler.update(optimizer, grads)` 接受**已反缩放**
   的梯度，只做"查非有限 → 更新 → 增长/回退"；`step` 改成 `unscale` + `update`
   （不重复查一次非有限）。`backoff()` 从溢出分支提出来，两条路径共用。
2. 溢出时不能只丢出错的那一批。累加和已经包含前几批的贡献，单批无法从中减掉，
   稀释之后梯度依旧是坏的。`GradientAccumulator.discard()` 把整窗作废并返回被
   丢掉的微批数，随后 `backoff()`。

**缩放状态进存档。** `GradScaler.state_dict()` / `load_state_dict()` 只有三个
标量（`scale` / `growth` / `skipped`），`Trainer.snapshot()` 以 `amp_scale` /
`amp_growth` / `amp_skipped` 写进 meta（同名键会被覆盖），`load_amp_state(meta)`
恢复。不存的话续训仍能跑，只是从 `init_scale` 重新摸索——前面"溢出过几次、该用
多大 scale"全部丢掉。加载时的校验是两段式：缺键报错（不能"保持当前值"假装接上），
`scale` 必须有限且为正（`inf` 会让下一步的种子直接溢出）；`scale` 允许高于构造时
的 `max_scale`，因为上限是这组超参的起点、不是存档的属性。

**设备态的种子。** 捕获图里写死的是种子的**地址**而不是**值**，所以 scale 变化
不需要重捕获：`_advance_seed()` 每步把当前 scale 写进那块 0 维缓冲。分段路径的
损失 cotangent 也读同一个值（`_seed_scalar`），否则分段 + 缩放会少乘一次 scale。

**测试** 13 个新用例：

- 调度：阶梯衰减下 4 步的 `optimizer.lr` 序列是 `[1.0, 1.0, 0.5, 0.5]`；
  累积窗口 2 + 每步衰减时是 `[0.1, 1.0, 1.0, 0.5]`（窗口内不推进）；
- 累积：4 个 `batch=2` 的微批与 1 个 `batch=8` 的大 batch **参数逐位一致**
  （这是"等效 batch"的判据，不是步数对）；窗口未满时参数一动不动；钩子每个
  窗口只调一次；
- 缩放：`init_scale=4` 与不缩放的 3 步参数轨迹一致（scale 乘上去又原样除掉）；
  注入 inf 梯度时跳过更新、scale 减半、参数不变，去掉坏梯度后同一步恢复更新；
  窗口内溢出时整窗丢弃（`pending` 归零）且参数不动；三个标量往返存档；
  `load_amp_state` 在"没有 scaler"与"存档没有 amp_*"两种情况下都拒绝。

**负向对照**（两组）：注释掉 `_apply` 里的 `apply_lr`，两个调度用例失败；去掉
`_accumulate` 里的 `discard()`，整窗丢弃用例的 `pending` 变成 1 而不是 0。各自
变红后还原。（第一版的整窗丢弃用例是"先喂坏批"——窗口本来是空的，丢不丢都一样，
所以第一次对照是绿的；改成"先喂好批再喂坏批"才真的有判别力。）

**验证数字**：本地 1438 个用例（1153 通过 / 285 跳过）、WSL HIP 1438（1160 通过 /
278 跳过），两处全绿；CUDA 真机未复跑（研究机停机）。

**仍未做**（阶段 C 的其余部分）：

- ~~f32 master weights~~：已在 5.126 完成（参数 f16 驻留 + f32 主副本 + 每步
  覆盖写回；结构代价与显存账见该节）。
- ~~RNG key 的派生~~：已在 5.125 完成（`key = mix(seed, step, rank)`，同一
  step 的不同 rank 不再共用掩码）。
- **fused selective scan / 分块前后向 / last-state-only**：同批验收要求同时报
  前向误差、VJP 误差、loss 曲线、gradient norm、峰值 VRAM、tokens/s、TTFT 与
  逐 token latency，需要训练预算与真机；
- **CUDA 真机的设备态 + AMP 组合**：本机没有 GPU，这条路径只有主机态的等价
  用例在跑。

---

### 5.124　审计遗留：f16 反向的数值量化（混合精度、全算子降精度与纯 f16 scan）

5.121 末尾把"f16 反向的数值量化"列为未做：5.28 与 5.91 验证过的只是
"f16 图在量化误差内与 f32 一致"，图是小 matmul + add，**乘法链只有两三层**。
审计要的是半精度梯度在真实长链上的累积误差，所以这一轮不改语义层，只把
数字量出来，并顺手修掉量化过程中撞见的一个构图缺陷。

**四组口径**，同一份参数、同一批数据、同一组 fixed seed：

1. f32 图反向（基准）；
2. `Autocast(dtype="f16")` 混合精度：30 个 `cast`、9 个 `matmul` 降 f16，
   `scan_linear` / 归一化 / 归约 / softmax 全部留在 f32；
3. `Autocast(low_precision=全部算子)`：把逐元素 / 归一化 / 卷积也交给 f16
   （`force_f32` 保持默认），用于观察白名单扩大后的退化；
4. 手工把 selective scan 图的**全部输入**声明为 f16（`x` / `dt` / `A` /
   `B` / `C` / `D` 都是 f16），单独量化 scan 这条链，序列长度 4 → 64。

2、3 的反向种子统一乘 `2^14` 再除回来；4 另做 scale 扫描。

**混合精度：误差停在量化底线。** 2 层 / 16 步的 LM，loss 相对误差 `1.2e-5`；
每个参数的梯度相对误差 `4.7e-4 ~ 2.3e-3`，方向余弦 `≥ 0.999997`，没有非有限值。
把扫描扩到 4 层 / 64 步，最差参数误差 `4.0e-3`、余弦 `≥ 0.999993`——**没有随
深度或长度放大**。24 步训练轨迹（f32 对照）的损失最大相对偏差 `2.9e-4`、
参数最大相对偏差 `4.9e-3`，scaler 没有跳过、也没有涨到上限。

**白名单扩到全部算子：`A_raw` 坏掉。** 1 层 / 8 步下多数参数仍在
`1e-3 ~ 5e-3`（`l0_dt_w` 是 `1.9e-3`），但 `l0_A_raw` 的梯度相对误差是
**`2.63e-1`**、方向余弦 `0.9696`。它的梯度范数 `6.5e-6`，比 `embed` 的
`2.0` 小六个数量级：`A` 只通过状态衰减影响 loss，梯度是多个时间步贡献
**相消**后的残差。各项在 f16 里各带 `~1e-3` 的量化噪声，相消后只剩两位
有效数字——相对误差因此被放大到 26%。

**缩放救不回相消。** `GradScaler` 乘在反向种子上，把整个反向链同比放大，
各项的量化噪声也同比放大；反缩放之后，相消丢掉的低位不会回来。混合精度把
逐元素链留在 f32，只在 matmul 边界量化一次，误差因此回到 `2e-3`。所以"把
白名单扩到全部算子"不可用，不是 scaler 参数没调好，而是精度划分缺了一层。

**定位实验**：把 `scan_linear` 单独留在 f32，`A_raw` 的误差 `26.3% → 26.3%`，
没有改善——相消发生在逐元素 / 归一化链上，不是 scan 节点。

**scan 本身并不坏。** 第 4 组（纯 f16 的 scan 图）在 T = 4 / 8 / 16 / 32 / 64 上：
`A` 的梯度相对误差 `1.10e-3 ~ 1.67e-3`、`dt` `3.9e-4 ~ 8.8e-4`、`D_skip`
`4.9e-4 ~ 1.1e-3`，方向余弦都 `≥ 0.999999`，**不随长度放大**。同一张图做缩放
扫描（T = 64）：`2^0 ~ 2^12` 的误差完全不变（`A` `1.59e-3`、`D_skip`
`1.09e-3`），`2^14` 时 `D_skip` 的梯度（范数 `24`）溢出——16 个元素里 7 个变成
非有限值。这正是 5.28 让 `GradScaler` 带溢出检测与回退的原因：可用窗口的上限
由**最大**梯度决定、下限由**最小**梯度决定，同一个 scale 不可能对所有参数都合适。

这条对判据的写法有影响：`A_raw` 这类小梯度参数要看**相对误差与方向余弦**，
"没有 NaN / inf"不够；`D_skip` 这类大梯度要看**非有限值**——两者的失败模式
正好相反。

**顺带修掉的构图缺陷。** `Autocast._align` 原来对**所有**操作数做浮点对齐：
`_work_dtype` 回答的是"这个节点的浮点工作精度"，把它套到全部参数上，i32 的
`ids` 会被 cast 成 f32，带 `embedding` 的图在构图期直接
`TypeError: embedding 的 ids 必须是整数下标，得到 f32`。5.90 的 20 个用例全是
matmul / softmax，没有整数操作数，所以这条一直没被走到——**真实语言模型是第一个
带整数输入的图**。改法是 `_align` 只对齐浮点操作数，与模块开头"只降 f32，
不碰 f64 与整数"那句承诺终于一致。

**测试** 3 个新用例：

- `test_autocast.py::test_integer_operands_are_not_cast`：embedding 图，autocast
  后 `ids` 仍是 i32、图里一个 `cast` 都没有、前向与 f32 图逐位一致；
- `test_autocast.py::test_empty_sets_are_not_defaults`：`low_precision=()` /
  `force_f32=()` 表示空集而不是"回落到默认集"；默认下 `softmax` 被强制回 f32，
  空 `force_f32` 时留在 f16，空 `low_precision` 时图里一个 `cast` 都没有；
- `test_amp.py::TestF16BackwardQuantization`：1 层 / 8 步的 LM 图，混合精度反向
  的每个参数相对误差 `< 1e-2`、方向余弦 `> 0.999`，前向 loss 相对误差 `< 1e-3`。

**负向对照**：撤掉 `_align` 的整数判断，前两条新用例都因上面那句 `TypeError`
变红；把 `is None` 改回 `or`，第三条用例的"空集"断言变红。还原后全绿。
（`A_raw` 的全算子降精度数字只做记录，不写成断言——它依赖随机初始化，
合适的位置是文档而不是门禁。）

**验证数字**：本地 1441 个用例（1156 通过 / 285 跳过）、WSL HIP 1441
（1163 通过 / 278 跳过），两处全绿；CUDA 真机未复跑（研究机停机）。

**结论与边界**：

- `Trainer(mixed_precision="f16")` 的反向可以接受：误差在 `1e-3` 量级，
  且不随深度 / 长度放大；
- **不能把降精度白名单扩到全部算子**：`A_raw` 这类相消后的小梯度会失真到 26%，
  scaler 救不回；scan 单独跑 f16 是安全的（`1e-3`，不随长度放大）；
- scaler 的窗口由梯度的动态范围决定（本例 `2^12` 内稳定、`2^14` 时大梯度的
  `D_skip` 溢出），所以必须保留溢出检测与回退，不能写死一个大 scale；
- 仍未量化：超过 4 层 / 64 步的趋势、真实语料上的 loss 曲线（需要数据与预算）、
  bf16（尾数比 f16 少、指数范围大，下溢不是问题但相对精度更差）。

---


### 5.125　阶段 C：RNG key 的派生（多 rank 的随机性不再退化）

`Trainer` 原来把 `steps` 直接写进 `rng_inputs`。单机可复现，但同一个 step
里所有 rank 会拿到**同一个 key**——数据分片不同、dropout 掩码相同，多卡的
梯度噪声完全相关，等于没有独立采样。

**改法**：`_derive_rng_key(seed, step, rank)` 用 SplitMix64 的 finalizer 把
三元组混成一个 i32（取 31 位正数，避免 int32 / uint64 的符号扩展差异）。
不用 `step + rank * K`：加法让相邻 rank 落在同一条等差线上，低位相关性会
传进掩码。

`Trainer` 新增 `rng_seed=0`、`rng_rank=0`、`rng_world_size=1`，并校验
`0 <= rank < world_size`、`0 <= seed < 2^31`。`_rng_key()` 同时供主机的
`_rng_feed` 与设备态的 `_advance_rng` 使用，两条执行路径的 key 一定一致；
key 仍是图输入，同一 `(seed, step, rank)` 逐位可复现，捕获成 CUDA Graph
之后也照常推进（写缓冲，不重捕获）。

**测试** 4 个新用例：

- 64 个 `(step, rank)` 组合给出 64 个不同的 i32 正数；
- 同一个 `(seed, step, rank)` 相等，换 seed 或换 rank 都变值；
- 两个 rank 在同参数、同数据、同 step 下损失不同（端到端：掩码确实不同）；
- 5 种非法参数（rank 越界、负 rank、负 seed、`seed >= 2^31`、`world_size=0`）拒绝。

**负向对照**：把派生退化成 `return int(step)`，前三条用例变红
（`8 != 64`、`0 == 0`、`12.0 == 12.0`），还原后全绿。

**验证数字**：本地 1445 个用例（1160 通过 / 285 跳过）、WSL HIP 1445
（1167 通过 / 278 跳过），两处全绿。

**边界**：这里仍是"单进程模拟多个 rank"；真正的多进程要由启动器把
`rank` / `world_size` 传进来，并与数据分片（`DataLoader` 的 rank）共用同一个
seed。数据侧的 rank 分片审计没有要求，没有提前做。

---

### 5.126　阶段 C：f32 master weights（参数 f16 驻留，主副本更新）

5.123 末尾列的第二块。5.91 的 `mixed_precision="f16"` 只改图：参数仍是 f32
图输入，图里给每个参数插一次 down-cast。要"参数真的以 f16 驻留"，需要图输入
本身声明成 f16，同时把更新留在 f32 上——否则 f16 上的原地更新会被量化吞掉。

**改法**（三处）：

1. `Autocast` 新增 `low_precision_inputs`：把指定的图输入直接声明成目标
   dtype（只降 f32，f64 原样），名字不是图输入时抛 `KeyError`——拼错会让
   "参数降精度"静默失效。
2. `Trainer(master_weights=True)`：要求 `mixed_precision="f16"` 且参数全是
   f32。构造时把参数名交给上面那个选项，并在内部建一份 f16 驻留缓冲；
   `store` 保持 f32，继续当主副本（优化器更新它、存档存它）。
3. 更新路径：主机态每次 `_update` 返回 `True` 后用主副本覆盖驻留缓冲；设备态
   上传两组缓冲（f16 给捕获图、f32 给优化器），每步末尾做一次 D2D 的
   `astype` 写回。`sync_to_host` / `snapshot` 取的是 f32 主副本，若取 f16 会把
   权重的精度永久降一级。

`grad_hook` 返回 `None`、scaler 溢出、累积窗口未满时都不写回——`_update` 现在
返回"参数是否真的动了"，写回跟着它走。

**校验**：`master_weights=True` 搭配"无 `mixed_precision`"或 bf16 都拒绝；
f64 参数拒绝；`refresh_resident()` 在未开启时调用拒绝（不是静默空操作）。

**测试** 13 个新用例（6 个主机态 + 4 个 Autocast 结构 + 3 个设备态，设备态用
把 numpy 当显存的替身后端）：

- 图输入 w/b 是 f16、数据 x/t 仍是 f32；驻留 f16、`store` 与优化器 f32；
- matmul 的 w 输入 producer 是 `None`（参数直接当输入）；不开 master 的对照
  里它是 `cast`；
- 240 步与全 f32 参考的权重相对误差 `< 1e-3`，w 收敛到真值（atol 5e-2）；
- 外部改 `store` 后 `refresh_resident` 生效；未开启时调用报错；
- 快照里是 f32 主副本（与 `store` 逐位相等）；
- 三种非法组合；
- 设备态：捕获缓冲 f16、优化器持有 f32、每步更新后驻留等于主副本的 cast、
  `verify_device_state` 无问题、`sync_to_host` 写回主副本；
- Autocast：指定输入降 f16、数据输入不动、未知名字拒绝、f64 不动、幂等。

**负向对照** 三组：

1. 去掉主机态写回 → `test_training_converges_and_matches_f32_within_quantization`
   变红（权重相对误差 3.37）；
2. `low_precision_inputs=()` → dtype 与结构两条用例变红；
3. 去掉设备态写回 → 两条用例变红（损失停在同一值、驻留与主副本不等）。

还原后三处全绿。

**验证数字**：本地 1458 个用例（1173 通过 / 285 跳过）、WSL HIP 1458
（1180 通过 / 278 跳过）。

**结构代价（实测，不宣称加速）**：2 层 LM（vocab 258、d_model 16、steps 8、
26 个参数）上：

| 口径 | 节点 | cast | down(f32→f16) | up(f16→f32) | cast 直接吃参数 |
|---|---|---|---|---|---|
| 参数 f32（5.91） | 194 | 30 | 18 | 12 | 8 |
| 参数 f16 驻留 | 209 | 45 | 9 | 36 | 11 |

参数全量 f16 驻留**没有减少** cast：matmul / conv 权重省掉的 down-cast 被
归一化 / scan 参数（只被 f32-only 算子消费）的 up-cast 反超。因此这个开关的
收益是"参数缓冲与读取带宽减半"，不是"减少 cast"；要减少 cast，应该只把
matmul / conv 的权重列进 `low_precision_inputs`，而不是全量降精度。

**数值**：200 步线性回归（lr 0.05、Adam、scale 2^8），f16 驻留与全 f32 参考
的权重相对误差 6.07e-5（w）/ 3.27e-4（b）；对照"参数 f32、图里 cast"的 5.91
轨迹是 6.16e-5 / 2.13e-4——同一量级，说明这里换掉的是结构开销，不是用精度
换来的（量化误差由 f16 计算本身决定）。两者的损失都从 7.1172 降到约 2e-5。

**显存账**：设备态下非 master 模式是 1 份 f32 参数；master 模式是 f16 驻留
+ f32 主副本 = 1.5 份，另加优化器槽位（两者都是 f32）。所以它不是"省显存"
的开关，省的是每步读写参数时的带宽；主机态没有带宽收益，只有一份额外的
f16 副本与每步写回的开销。

**边界**：

- 驻留缓冲不存档（主副本才是权威）；`load_checkpoint` 之后要
  `refresh_resident()`，否则训练静默读旧副本。
- bf16 不支持：参考后端用 f32 承载 bf16，驻留省不了存储。
- 设备态的 D2D 写回走数组模块的 `astype`；HIP 的 `DeviceArray` 还没接入设备态
  优化器，与 5.84 的边界相同。
- 没有做"自动选择哪些参数该降精度"：取决于算子的消费方，目前交给用户显式
  构图（`Autocast(low_precision_inputs=...)`）。

---

### 5.127　CUDA 真机（Tesla M40 / sm_52）：f16 原子加的编译降级

研究机恢复了一台可用的 CUDA 机器：3 × Tesla M40 24GB（sm_52，Maxwell），
驱动 535.309.01，`cupy-cuda12x 14.2.0` + `cupy-cuda12x[ctk]`（CUDA 12.9 的
头文件与 NVRTC 来自 wheel，目标机不需要装 toolkit）。这是 5.84 之后第一次
在 NVIDIA 真机上跑全量。

**第一次跑：250 个 CUDA 后端用例里 26 个 NVRTC 编译失败**，全部落在
`TestCudaFloat16` / `TestCudaScan` 的 f16 用例与 `TestCudaStrictness` 的
"f16 可用"上。错误不是数值，而是编译：

```
error: no instance of overloaded function "atomicCAS" matches the argument list
            argument types are: (unsigned short *, unsigned short, unsigned short)
```

**根因**：`atomicCAS(unsigned short*)` 从 sm_60（Pascal）才有；codegen 的 f16
PRELUDE 无条件带着 `af_add_h`，于是 **任何一个 f16 内核**（哪怕不含 scatter）
在 Maxwell 上都编译不过。f32 内核不受影响，所以 224 个用例照常通过。

**修法**：`af_cas_h` 在 `__CUDA_ARCH__ < 600` 时把 CAS 展宽到对齐的 32 位字：

```cuda
unsigned long long addr = (unsigned long long)p;
unsigned int* base = (unsigned int*)(addr & ~2ULL);
unsigned int shift = (addr & 2ULL) ? 16u : 0u;
unsigned int mask = (addr & 2ULL) ? 0xFFFF0000u : 0x0000FFFFu;
```

第一次实现用"重新读整个字"的值当 CAS 比较值，f16 的 embedding 反向立刻暴露
出重复累加：`ids=[1,1]` 的目标行从 `[0.23, -0.67, 1.00]` 变成
`[32.1, -513, 256.5]`（分别被加了 139 / 768 / 256 次）。原因是外层重试循环在
"CAS 成功但返回的旧值 ≠ assumed"时会再算一次 `assumed + v`，而那次 CAS 其实
已经写入了。**比较值必须由调用方传进来的 `assumed` 构造**，成功时返回
`assumed`；只有目标半被邻居改了才把新值交回外层重算。修正后：

- 并发压力探针：1024 个线程对同一 32 位字里的两个 half 各加 1，结果
  `[512, 512, 0, 0]`（错误版本只能加到 2）；
- `TestCudaFloat16.test_embedding_backward` 通过；
- CUDA 后端 250 个用例全过（1 个跳过）、`test_cuda_train` 7 个全过。

**5.126 的真机验证**：`Trainer(device_state=True, mixed_precision="f16",
master_weights=True)` 在 M40 上跑 200 步线性回归，loss 7.1172 → 1.89e-5，
`w = [0.9976, -1.9996, 0.5025, 2.9956]`，显存里驻留 `float16` / 主副本
`float32`，`verify_device_state()` 无问题——D2D 的 `astype` 写回在 cupy 上成立。

**全量**：CUDA 真机 1453 个用例（跳过 64，其中 221 个"没有 GPU"的用例在真机
上真的执行了；与本地 1458 的差额推测来自 Python 3.14 与 3.10 的 `load_tests`
展开差异，未逐条核对）。本地 1458（1173 通过 / 285 跳过）、WSL HIP 1458
（1180 通过 / 278 跳过）不变。

**边界**：

- sm<60 的 f16 原子加是 32 位 CAS 模拟：同一字内两个 half 的并发更新是安全的
  （另一半变了就用最新值重试），代价是冲突时每半字多一次 CAS 重试。
- Maxwell 没有 fp16 算术；codegen 本来就用 f32 算、f16 存，所以这条修复只影响
  存储宽度，不改变数值路径。
- `cupy-cuda12x` 本身不带 CUDA 头文件；真机上要装 `cupy-cuda12x[ctk]`，否则
  第一个需要 NVRTC 的 kernel 就报 `Failed to find CUDA headers`。这是环境要求，
  不是框架依赖。
- 该机 0/1 号卡被常驻推理服务占用，测试用 `CUDA_VISIBLE_DEVICES=2` 跑在剩余
  14GB 的卡上；框架本身不感知这一点。

---

### 5.128　阶段 C 收官：分块 selective scan（两趟）与末状态专用内核

串行扫描的并行度只有尾元素数：`B·d_inner` 个线程，每个沿 `T` 走一遍。
T 一大，SM 就空着——5.103 在 RTX 2080 Ti 上量过，`T=4096 → 16384` 耗时严格
4 倍，说明这条路径完全没有随长度扩展的并行度。本轮补上分块两趟扫描。

**不能直接改现有内核**。`scan_linear` / `scan_linear_backward` 与手写模板之间
有"逐 token 相等"的用例盯着（5.48），任何改变递推顺序的实现都会破坏它。
分块版的块内递推**逐字不变**（还是 `h = a*h + b`，还是同一个
`a[idx]*h + b[idx]` 表达式），变的只有**块入口**：它由前若干块的复合
`h_in <- prod*h_in + sum` 给出，而逐步递推的舍入路径与"先乘再加"的复合
不同。因此分块必须是显式开关，默认路径一个字节都不许动：

```python
CudaBackend()                    # 串行，逐位与 5.35 一致
CudaBackend(chunked_scan=128)    # 分块，块长 128；0 = 关闭
```

**两趟的形状**：

1. `scan_linear_chunk_sum`：`chunks * elems` 个线程，每块独立算
   `(prod, sum)`——块内从 `p=1, s=0` 起递推 `s <- a*s + b`、`p <- a*p`，
   与块入口无关，所以完全可并行；
2. `scan_linear_chunk`：同样是 `chunks * elems` 个线程，每个线程先把前 `k`
   块的 `(prod, sum)` 复合成自己的 `h_in`，再在块内顺序写出 `out`。

第二趟没有拆成"块前缀一趟 + 读出再一趟"：前一趟的并行度只有 `elems`，
与串行版一样低，却要多一次 launch 与一份 `h_in` scratch。让每个线程各自
重算 `k` 步前缀，代价是 `O(chunks²·elems)` 次乘加，而 `chunks` 通常比 `T`
小两个数量级；实测这笔重算在最差的 T=16384 / chunk=64（256 块）下也没有
把收益吃掉（见下表）。

反向对称：`scan_linear_chunk_sum_backward` 把倒序第 `r` 步映射到
`t = steps-1-r`，把 `dh_t = dy_t + a_{t+1}*dh_{t+1}` 写成
`A = a_{t+1}`（`t = steps-1` 时为 0）、`B = dy_t` 的正向递推，块也按 `r` 切；
`scan_linear_chunk_backward` 同样先复合前缀再从后往前写。

末状态另有两个内核：`scan_linear_last`（串行、走完 T 步只写一次）与
`scan_linear_chunk_last`（读块和一次性复合，`O(chunks)` 步写一个数）。
流式推理每步只要 `(tail...)` 的末状态，没必要先把 `(T, tail)` 物化再切片；
`CudaBackend.scan_last_state(a, b, h0)` 是这条路线的入口。

**精度**（Tesla M40，SSM 端到端 `T=64, B=2, D=16, N=8`，`chunk=16`，f32）：

| 量 | 相对误差 |
|---|---|
| 前向 serial vs numpy 参考 | 2.83e-7 |
| 前向 chunk vs serial | 1.16e-7 |
| 前向 chunk vs 参考 | 2.76e-7（绝对最大差 2.68e-7） |
| VJP `dx` / `ddt` / `dA` / `dBt` / `dCt` | 1.81e-7 / 1.85e-7 / 1.70e-7 / 1.87e-7 / 1.18e-7 |

分块带来的误差没有超过"串行 vs 参考"本身的基线，VJP 也停在同一量级——
块间复合只多了一次 `prod*h + sum`，没有把误差量级抬高。

**性能**（M40 sm_52，`B=16, D=64`，尾维 1024 个元素，f32，中位数 10 次）：

| T | 串行 | chunk64 | chunk128 | chunk256 | 加速 |
|---|---|---|---|---|---|
| 1024 | 0.389 ms | 0.162 ms | 0.161 ms | 0.178 ms | 2.42× |
| 4096 | 1.402 ms | 0.466 ms | 0.465 ms | 0.452 ms | 3.02× |
| 16384 | 5.478 ms | 1.957 ms | 1.761 ms | 1.763 ms | 3.11× |

`T=16384` 上 tokens/s 从 47.9M 到 148.9M，逐 token 延迟 0.021 → 0.007 µs。
反向同形状：`T=1024` 0.434 → 0.164 ms（2.65×）、`T=4096` 1.631 → 0.458 ms
（3.56×），37.7M / 40.2M → 99.9M / 143.0M tokens/s。加速比随 T 上升，与"串行版
墙钟随 T 线性、分块版随 T/chunk 线性"的预期一致；chunk 从 64 到 256 之间只有
10% 左右的差别（T=1024 时 chunk=256 已经开始变差，块数太少摊不满 SM）。
M40 不锁频，同一台机器重复跑会有 10~25% 的时钟漂移；表中是最后一份 JSON
（`benchmarks/scan_chunked.py` 的输出），比值比绝对值可靠。

**显存**（`T=4096, B=16, D=64`，cupy 池，`free_all_blocks` 后一次执行的增量）：

| 口径 | 增量 |
|---|---|
| 串行整段 | 16,777,216 B |
| chunk128 整段 | 16,777,216 B |
| 串行末状态 | 4,096 B |
| chunk128 末状态 | 4,096 B |

分块的 `prod` / `sum` 各 `32 × 1024 × 4 = 131 KB`，落在 cupy 同一个 16 MB
池块里，所以整段口径看不出增量差异——**分块省的是时间，不是显存**；
真正省显存的是末状态内核（16 MB → 4 KB，4096 倍），它不分配 `out`。

**延迟**（同一台机器，`B=16, D=64`）：

| 口径 | 串行 | chunk64 | chunk128 |
|---|---|---|---|
| prefill `T=512` | 0.196 ms | 0.101 ms | 0.110 ms |
| prefill `T=1024` | 0.356 ms | 0.149 ms | 0.149 ms |
| decode `T=1`（整段 / 末状态） | 35.56 µs | — | 29.76 µs |

`T=1` 的 decode 不触发分块（`steps <= chunk`，走的还是串行内核），它的收益
全部来自末状态内核少写一次 16 MB 输出——`35.56 → 29.76 µs`（−16%）里省下的
是 `out` 的分配与写回，不是扫描本身。

**训练曲线**（8 步 SGD，`lr=0.05`，同一份参数与数据）：

| step | 串行 loss | 分块 loss | 串行 grad_norm | 分块 grad_norm |
|---|---|---|---|---|
| 1 | 0.05869843 | 0.05869843 | 0.36747796 | 0.36747796 |
| 4 | 0.04323522 | 0.04323522 | 0.26810684 | 0.26810684 |
| 8 | 0.03198751 | 0.03198751 | 0.20289719 | 0.20289719 |

8 步内只有第 2 步的 `grad_norm` 末位差 `1e-8`，loss 全部逐位相同——分块的
误差没有在优化里累积。

上面所有数字由 `benchmarks/scan_chunked.py` 产出（`--out` 落 JSON）：前向 / 反向
吞吐、显存增量、prefill / decode 延迟、SSM 前向与 VJP 误差、两条训练曲线都在
同一份脚本里，输入常驻显存，计时取中位数。

**测试**（`TestCudaChunkedScan` 8 个 GPU 用例 + `TestChunkedScan` 4 个离线 emit 用例，
真机上 8 个 GPU 用例全部执行）：

- `test_default_path_is_serial`：`chunked_scan=0` 与"块长 > T"的后端逐位相等，
  钉住默认路径没有变化；
- `test_first_chunk_is_bitwise_equal`：第一个块没有块前缀，它的输出必须与
  串行版**逐位**相同——块内递推没有被改坏；
- 前向 / 反向 / 3D 尾形状与串行的 `rtol=1e-5`；
- f16 存储 + f32 scratch 的 `rtol=2e-2`；
- `scan_last_state` 在两种配置下都等于整段扫描的最后一步。

真机全量 1465 个用例（跳过 64），本地 1470（跳过 293），WSL HIP 1470
（跳过 286，1184 通过）全绿——离线那 4 个 emit 用例钉住结构：f16 下块和仍是
`float*`、末状态内核只写 `out[i]`、第二趟的前缀循环上界是 `k`。分块扫描在 HIP
上单独验证：`HipBackend(chunked_scan=16)`
的前向相对误差 5.86e-8、反向 1.06e-7、末状态 1.51e-7，第一个块与串行版逐位相同
——HIP 走 `__getattr__` 透传同一份内核描述，两条硬件路径没有分叉。
新增内核没有手写对照模板：它们不是 `scan_linear` 的逐字等价物，
"逐 token 相等"的判据在这里不成立，验收靠数值对拍与默认路径的逐位不变。

**边界**：

- 块内递推逐字不变，块间复合改变舍入 → 分块**不逐位**等于串行，必须显式
  开启；`chunk >= T` 时两个实现逐位一致（测试里有这条）；
- `prod` / `sum` 固定 float32，不随 dtype 降成 f16：块内递推本来就在 float
  里做，把聚合值降精度会多两次舍入，误差比串行路径大一个数量级；
- 第二趟每线程重算 `O(chunks)` 步前缀，是"省一趟 launch"换来的；块数很大
  （`T/chunk > ~1024`）时重算会成为主要开销，此时应加大 chunk；
- decode 的单步（`T=1`）无法分块，收益来自末状态内核；分块只对 prefill /
  训练的长序列有意义；
- 显存增量在池粒度下看不出来，不能宣称分块省显存。

---

### 5.129　HIP 上的自动图（`graph_mode="auto"`）实测

5.61 验证的是 `HipGraphPlan` 本身（手工 `capture` / `replay`），5.62 用同一套
手工捕获对照了逐节点口径；而 `graph_mode="auto"` 这一层——按"图 + 版本 +
输入签名"缓存计划、LRU 上限、节点数闸门、失败隔离——5.79 只在 CUDA 上跑过。
这一层完全在 `DeviceBackend` 里，厂商侧只提供 `capture`，所以 HIP 与 CUDA 的
差异只可能来自 `HipGraphPlan` 的 replay 语义或捕获限制。

本轮在 WSL ROCm（gfx1100）上把这条路径补齐：

| 用例 | 结论 |
|---|---|
| 两次 `execute` 同一图同输入 | 只捕获 1 个计划，第二次 replay 结果与第一次**逐位**相同 |
| auto vs `graph_mode="off"` | 数值一致（`rtol=1e-4`），`_graph_failures` 为空 |
| 同形状换输入值 | 仍是同一个计划，且 replay 读到的是新输入（与旧输入的结果差 > 1e-3） |
| 单节点图 | `min_graph_nodes=2` 生效，`_graph_plans` 为空 |
| `max_graph_plans=1` 下换形状 | 计划数不超过 1，被挤掉的计划不影响正确性 |

收益（`execute` 口径，含 H2D/D2H，与 5.79 的 CUDA 数字同口径，取 5 次最好）：

| 用例 | 逐节点 | auto | 加速 |
|---|---|---|---|
| `transformer_block` | 5.510 ms | 2.935 ms | 1.88× |
| `cnn_lenet` | 3.920 ms | 1.847 ms | 2.12× |

两份都是 1 个计划。CUDA 上 5.79 的同一对用例是 −56% / −55%（约 2.3× / 2.2×），
HIP 低一些但同量级——收益来源相同：把一次 `execute` 里几十个 kernel launch
收敛成一次 replay，剩下的固定成本是输出 D2H 与簿记。

**边界**：auto 只对"同一张图反复执行"有效；形状变化会新增计划直到 LRU 上限，
超出后按最近最少使用淘汰并重新捕获（不是错误，只是多一次捕获成本）。小图
（节点数低于闸门）仍走逐节点，因为省下的 launch 抵不过计划查找与签名比对的
开销（5.81）。`execute` 口径的绝对值受 H2D/D2H 影响，比值才可比。

**真机复核**：同一份代码同步到 192.168.1.18 的 2 号卡（Tesla M40，Python 3.10）跑全量，
`Ran 1469 tests ... OK (skipped=68)`。与本地 `Ran 1474 / skipped 297` 的 5 个差不是缺用例：
先按 `TestLoader` 逐个收集 id，两边都是 1474 个、集合完全一致；跑起来少的那 5 个全来自
`test_cuda_backend.TestCudaConvCudnn`——它的 `setUpClass` 在无 cuDNN 时抛 `SkipTest`，
3.10 的 `_handleClassSetUp` 把整类折叠成一条 skip（`-v` 里只剩一行孤立的
`skipped '系统里没有可用的 cuDNN'`，`testsRun` 不加 5），3.14 则逐个测试计数。
第 19 行的"总数与本地差 5 尚未定位"就此关闭。

至此第 21 行"执行器架构改造"的余下部分只剩**批量派发**。

### 5.130　执行器架构改造收官：批量派发（计划期绑定内核，运行期不再逐节点校验）

5.79 把常规路径接进 CUDA Graph、5.83 把训练的前向与反向也接上、5.129 在 HIP
上补齐 auto 的实测。第 21 行剩下的最后一项是"批量派发"。动代码之前先把这块
还剩多少空间量清楚——否则很容易把"重写执行骨架"当成优化。

**第一步：host 与设备各占多少**（Tesla M40 / 2 号卡，`graph_mode="off"`，
每轮先同步；`run_device` 用 CUDA event 夹住整图，`execute` 与基准同口径）：

| 用例 | 节点 | host 提交 | 设备窗口 | execute | 其中 H2D | 纯派发 | 每节点 |
|---|---|---|---|---|---|---|---|
| `transformer_block` | 24 | 2.123 ms | 2.127 ms | 2.139 ms | 0.524 ms | 1.373 ms | 49.6 μs |
| `cnn_lenet` | 10 | 1.121 ms | 1.126 ms | 1.146 ms | 0.337 ms | 0.655 ms | 47.3 μs |
| `matmul_1024` | 1 | 0.738 ms | 1.819 ms | 2.536 ms | 0.658 ms | — | — |
| `maxpool2x2` | 1 | 1.199 ms | 3.021 ms | 3.722 ms | 1.119 ms | — | — |

单节点空壳图（两个 8 元素输入、一个 `add`）的 `run_device` 常数是 182 μs，
所以"每节点 48～50 μs"是扣掉固定常数后的净派发成本。两条结论：

- **多节点图是 host 主导**：host 提交与设备窗口几乎相等（2.123 vs 2.127），
  说明 GPU 一直在等 Python 把下一个 kernel 递进来，不是在算；
- **单算子图是设备主导**：`matmul_1024` 的设备窗口 1.819 ms 远大于 0.738 ms 的
  提交时间，且 H2D 就占掉 execute 的 26%。

同一批用例的 auto 是 0.588 / 0.378 ms（3.6× / 3.0×）——**数量级收益已经在
CUDA Graph 里拿到了**。逐节点路径剩下的可压缩空间就是那 1.37 / 0.65 ms。

**第二步：这 1.37 ms 花在哪**（30 次 `run_device` 的 cProfile，`tottime` 排序）：
算子内核本体（`_op_matmul` / cuBLAS 绑定 / `cp.empty` / `ascontiguousarray`）
占约 74%，`run_device` 的骨架簿记（组循环、args 构造、release 记账）约 11%，
`invoke` 的"查算子 + 逐参数 dtype 检查 + 从候选里挑"约 6%～14%。第三项有一个
特点：**结论只取决于 IR，与这一次的数值无关**（图输入在入口已 cast 成声明
dtype，组输出在骨架里也按声明 cast），所以它不需要每步重算。

**第三步：因此批量派发做成"计划期一次绑定、运行期成批执行"**，不重写骨架：

1. `_GroupPlan` 新增 `kernels` 字段（默认 `None`）。它承载"这一组每个节点的
   内核函数"，由后端在计划构建期填。
2. `Executor._plan` 构建完计划后调用一次 `_prepare_plan(plan)`；基类默认空操作
   （参考后端没有可省的固定检查）。
3. `DeviceBackend._prepare_plan` 逐组逐节点调用 `_planned_kernel(node)`：按 IR
   声明复用与 `invoke` 完全相同的判定——算子已注册；`_INDEX_ARGS` 指定的位置
   声明为 i32 / i8；其余输入声明为 f32，或 f16 且该算子在 `_f16_ops` 支持面内。
   全部通过就返回内核函数，否则整组留 `None`。
4. `invoke_group` 新增可选 `plan` 参数：`plan.kernels` 有值时走 `invoke_planned`
   （按计划里的函数逐个调用，组内前序输出照旧补位）；单节点组直接调
   `kernels[0]`，不进补位循环。
5. `run_device` 对单节点组直接按 `node.inputs` 取输入，不再套一层嵌套 listcomp。

**语义没有放松，只是把时机前移**：任一条判定不满足就整组退回原路径，所以
"未实现算子 / 不支持的 dtype"仍在第一次执行时按原来的文案抛出（`TestBatchDispatch`
的 f64 用例钉住这一点）。融合组优先走 `_try_chain`，融合失败才进快路径。

**收益**（同一后端上开 / 关 `_prepare_plan`，ABBA 交替 12 轮、各取中位数，
排除整机单向漂移）：

| 用例 | 逐节点 | 批量派发 | 变化 |
|---|---|---|---|
| `transformer_block` | 2.284 ms | 2.140 ms | **−6.3%** |
| `cnn_lenet` | 1.241 ms | 1.169 ms | **−5.8%** |
| `matmul_1024`（单节点） | 2.757 ms | 2.710 ms | −1.7% |
| `conv2d_3x3_pad1`（单节点） | 0.573 ms | 0.550 ms | −4.1% |

单节点图的 1.7%～4.1% 落在噪声里（同一份代码重复测量时正负都有），只说明
"没有变慢"。开 / 关的绝对值都比上面的探针高约 7%，那是整机漂移，比值才可比。

**边界与为什么到此为止**：剩下 11% 的骨架簿记（args 构造、`resolve`、形状比对、
release 记账）确实还能再计划化，但那是执行器的生命周期管理，与"派发"无关；
估下来再省 3～5%，代价是把符号形状校验与引用计数搬进预编译动作里，风险与
收益不成比例。**再往上就是重写每个 `_op_*` 的形状推导与库调用，等于换一个
后端**；而那个量级的收益已经由 auto 图拿走（3.6× / 3.0×）。这笔账写在这里，
是为了避免后来者再去"优化派发"时重复测量。

**测试**：4 个骨架用例（本地，纯 numpy 假执行器）——绑定后完全绕开 `invoke`、
同一张图的计划只准备一次、没绑定时退回原路径、绑定覆盖组内每个节点；4 个真机
用例——多节点图每组都绑上、快路径与强制回退的结果**逐位相同**、f64 组不绑定
且照原样抛 `BackendCapabilityError`、i32 下标的 embedding 能绑定。

**三处回归**：本地 `Ran 1482 ... OK (skipped=301)`；WSL ROCm `Ran 1482 ...
OK (skipped=290)`；CUDA 真机（M40 / Python 3.10）`Ran 1477 ... OK (skipped=68)`。
1477 与 1482 的 5 个差仍是 5.129 已定论的 `setUpClass` 计数差异，不是缺用例。

至此第 21 行的**执行器架构改造**清空：多节点图的 launch 收敛由 auto 图完成，
逐节点路径的每节点固定校验由本节前移到计划期。

---

### 5.131　执行期分层内存：HIP 上的契约通路（CUDA 待接）

1.2 里记的缺口是"执行期还没有'按需分层'的通路，插件的执行路径也只拿到
视图，不经过契约"。现状核对下来是两句话：`Arena` 只在 `capture()` 的窗口里
生效；普通执行时 `_active_arena` 恒为 `None`，`dispose` 是纯空操作，中间值
只能等容器的引用计数把它们还给 cupy / HIP 池。而 HIP 的 `runtime` 其实早就
有 `set_allocator(allocator)` 这个钩子（捕获期就在用），缺的只是一个分层
实现和把它接进 `execute` 的时机。

**分层怎么分**（`hip/memory.py::TieredAllocator`）：

| 层 | 范围 | 做法 | 理由 |
|---|---|---|---|
| 小对象 | `≤ 64 KiB` | 按 2 的幂分档的 free list | 逐元素内核一次几 KB，复用率最高，分档比 best-fit 查找便宜 |
| 中对象 | `64 KiB ~ 2 MiB` | 内部 `Arena` best-fit | 卷积 im2col / 中等激活的大小混合，需要合并相邻空闲块 |
| 大对象 | `> 2 MiB` | 直接 `hipMalloc`，释放即还 | 注意力与大 matmul 的临时块常驻会把峰值抬上去 |

持有者是新增的 `_PooledBlock`，**不能复用 `_Buffer`**：后者的 `__del__` 直接
调 `hipFree`，池化块一析构就还给驱动，池永远复用不到。生命周期因此完全靠
引用计数：`run_device(release=True)` 里 `env.pop` 让中间值的引用归零，
`_PooledBlock.__del__` 把它还给对应层；图输出由调用方持有，不会被回收——
这正是普通执行与捕获期最本质的差别（捕获期有执行器给的显式回收点）。

**怎么接进执行路径**：`DeviceBackend` 新增 `memory_mode`（`container` 默认 /
`tiered`），`execute` 拆成两层——外层 `_memory_scope()` 管分配器的安装与
还原，内层 `_execute()` 是原来的逻辑。`HipBackend._memory_scope` 在 `tiered`
模式下 `runtime.set_allocator(self._tiered)`，`finally` 里严格还原（这个钩子是
进程级的，泄漏出去会让后续所有分配走错池），并把分配器写进 `_last_arena`。
**`memory_stats()` 因此不需要新增 API**：它读到的永远是最近一次捕获或执行的
统计。

**CUDA 侧显式不接**：cupy 的分配器要求返回它自己的 `MemoryPointer`，自定义
持有者要碰 `cupy.cuda.memory.BaseMemory` 的内部协议，不同版本差异较大；没有
真机验证过就接，很容易变成"看起来支持、峰值却算错"。所以
`CudaBackend(memory_mode="tiered")` 直接抛 `BackendCapabilityError` 并指向
HIP——比静默退回容器池诚实。

**测试**（6 个，全部在 HIP 真机）：3 个只碰分配器——同档块释放后必须复用同一个
地址、`> 2 MiB` 的块释放后 `total_bytes` 必须归零、峰值不低于 live；3 个走
执行——`tiered` 与 `container` 的结果**逐位相同**、`execute` 结束后分配器必须
还原成进入前的那个、`container` 模式不装分配器且 `memory_stats()` 仍为 `None`。

**回归**：本地 `Ran 1488 ... OK (skipped=307)`；WSL ROCm（gfx1100，DXG 直通）
`Ran 1488 ... OK (skipped=290)`。CUDA 侧只多了构造参数与一条明确的拒绝，
真机数字见下一次同步。

**没做的**：host 暂存层（H2D 仍走 pageable，CUDA 的 `pinned_array` 只在图
replay 里用）；CUDA 的分层实现；跨多次执行的峰值归零策略（现在 `reset_stats`
只在分配器第一次创建时调用，多次执行共用一条峰值曲线——这是刻意选的，单次
执行的峰值对判断"这一步多大"更有用）。默认仍是 `container`：短小图装/卸
分配器的开销与收益未必划算，分层是给"想要可观测峰值、或显存吃紧"的场景用的
显式开关。

---

### 5.132　SNN 的事件驱动稀疏算子：CSR 事件 × 稠密权重

5.35 交付的是**数据侧**的稀疏：`SpikeEvents` 用 CSR 变体存发放位置、`EventPool`
复用下标缓冲，省的是"再存一份稠密脉冲张量"。但计算侧没有对应物——多层 SNN
仍然把事件摊回稠密矩阵再 `matmul`，稀疏性只落在存储上，FLOPs 一条没少。这一节
补的是计算侧：一个真正按事件数花时间的内核。

**形态**：新增两个 IR 算子。

| 算子 | 输入 | 输出 | 语义 |
|---|---|---|---|
| `sparse_linear` | `weights[K,N]`、`indices[E]`(i32)、`values[E]`、`offsets[S+1]`(i32) | `values.dtype[S,N]` | `out[s,n] = Σ_{e∈[off[s],off[s+1])} values[e]·W[indices[e], n]` |
| `sparse_linear_weight_grad` | `dout[S,N]`、`indices`、`values`、`offsets`，attr `rows=K` | `dout.dtype[K,N]` | 把 `values[e]·dout[s]` 原子散射回 `indices[e]` 行 |

`E` 是事件数，随输入脉冲变化；`S` 是步数，由 `offsets` 长度减一给出。
`indices` / `offsets` 固定 i32（IR 只有 i8 / i32 两种整数），`values` 与
`weights` 必须同 dtype。

**为什么按输出收集，而不是按事件散射**：输出 `(S,N)` 本身是稠密的（每一步每个
神经元都有膜电位），躲不掉。所以一个线程负责一个输出元素，扫自己那一段 CSR；
每个输出只写一次、不需要原子加。省下来的是**读**：第 `s` 步只读
`off[s+1]-off[s]` 个事件，而不是 `K` 个神经元。散射方案要先清零整块再对每个
事件做 `N` 次原子加，原子冲突随发放率上升，反而更慢。

**符号维与静态步数**：`indices` / `values` 的长度写成符号维 `E` 是这套设计的关键
（同一张图要接不同长度的 CSR），`infer` 只校验两者等长；`offsets` 的长度则
**必须静态**——`infer` 要给输出形状，符号维表达不了 `offsets.shape[0] - 1`，
构图期直接拒绝并说明原因。

**越界检查在设备端**：`offsets` 倒序/越界与 `indices` 越界都由内核记
`err[0] = 1` 并夹到合法范围，主机侧走与 `embedding` 同一条 `_check_ids`。
夹取只为避免非法访存，不掩盖错误：普通执行同步后抛 `BackendCapabilityError`，
捕获期把标志登记进计划、replay 时复查（5.98 的通路）。参考实现同样拒绝这两种
输入（`ValueError`），所以对拍的是"错得一样"。

**反向只对 `weights`**：`indices` / `offsets` 是整数索引，没有梯度；`values` 是
事件幅度（SNN 里是 0/1 的发放强度），按 `dropout_mask` 的约定返回 `None` 表示
"该输入确实不接收梯度"，而不是造一个零张量。`rows` 只能由属性带过去——反向图
里没有 `weights` 这个输入，散射内核却必须知道行数才能查越界。

**实现**：内核在 `lower/kernels.py` 用 L2 描述（`sparse_linear` /
`sparse_linear_weight_grad`），CUDA 与 HIP 共用同一份模板与方言，没有第二份
手写 C。`_INDEX_ARGS` 把 `indices` / `offsets` 标成非浮点输入，`GPU_OPS` 与
`_F16_OPS` 自动覆盖——f16 时存储是半字、累加仍是 float。

**测试与回归**：`tests/test_snn.py` 新增 `TestSparseLinear`（11 个用例）——
前向与等价的稠密 `matmul(spikes, weights)` 逐位对拍、空步必须全零、同一步内
重复行必须累加、权重梯度与 `dense.T @ dout` 对拍、事件数 `E` 用符号维接三种
长度（0 / 3 / 5）、越界下标与倒序 `offsets` 被拒绝、符号长度的 `offsets` 在
构图期被拒绝、图校验与 JSON 往返（含反向图）。conformance 新增两条用例
（`sparse_linear` 含空步与重复行、`sparse_linear_weight_grad`），CUDA 与 HIP
的"声明即覆盖"检查因此仍然成立。

数字：本地 `Ran 1497 ... OK (skipped=307)`；WSL ROCm（gfx1100，DXG 直通）
`Ran 1497 ... OK (skipped=290)`，conformance 两个新用例的最大误差
`2.4e-07` / `0`；CUDA 真机（Tesla M40，`CUDA_VISIBLE_DEVICES=2`）
`Ran 1492 ... OK (skipped=74)`。

**没做的**：`values` 的梯度（事件幅度的反向需要"事件 → 步"的反查，SNN 里不用）；
时间展开仍由前端构图决定，图随 `T` 线性增长这条边界没有变（5.18）；没有把
`SpikeEvents` 与这两个算子直接串起来的构图助手——数据侧与计算侧的接口是
"CSR 三件套"，前端已经能产出它，再加一层包装只会多一处可以漂移的映射。
5.133 的 `build_snn_events` 是第一处使用方，它直接消费 CSR 三件套，没有新增映射层。

---

### 5.133　事件驱动的 SNN 构件：把 5.132 的内核接进模型层

5.132 给出了 `sparse_linear`，但没有任何构件用它——`build_snn_network` 每层
仍然 `matmul` 一个稠密脉冲矩阵，稀疏性只落在数据表示上。这一节补模型层：
`build_snn_events` 与 `build_snn_network` 的输入 / 参数 / 输出完全一致，只把
每层的矩阵乘换成"压成 CSR → 按事件累加"。

**构图**：第 0 层的输入是连续电流，还没发放过，压不出稀疏，仍走稠密 `matmul`
（`_lif_layer`）；从第 1 层起走 `_event_layer`——`concat` 出 `(T×B, K)` 脉冲，
`spike_offsets` 给出每行起点，`spike_csr` 写出非零列号，`sparse_linear` 逐事件
累加出 `(T×B, N)`，再 reshape 回 `(T, B, N)` 逐时间步过 LIF。权重、偏置、膜电位
与读出复用**同一个** `lif_step`，因此两条路径可以逐位对拍。

**两算子而不是一个**：最初把列号与起点表打包进一个 i32 张量、下游用 `slice`
各取一段。参考后端上完全正确（与稠密路径 max diff 0.0），但 HIP 真机第一次跑
就拒绝：`HIP 后端的 slice 不支持 int32。不做隐式降级`。这不是缺陷而是既有边界
——整数索引从来不需要切片，给 `slice` 开 int32 只会扩大后端能力面。于是拆成
两个单输出算子：

| 算子 | 输入 | 输出 | 语义 |
|---|---|---|---|
| `spike_offsets` | `spikes[T,K]`（f32/f16） | `i32[T+1]` | `offsets[t]` 是第 t 行之前的事件累计数，末项为总事件数 |
| `spike_csr` | `spikes[T,K]`、`offsets[T+1]`（i32）、attr `capacity` | `i32[capacity]` | 第 t 段写入第 t 行的非零列号，按列号递增 |

`T` 与 `K` 必须静态：容量与输出长度都写在形状 / 属性里。`spike_offsets` 单线程
串行扫 `T×K`，前缀和天然有序、一次遍历就够，不引入 block 内扫描；`spike_csr`
一个线程一行，行内必须按列号递增写——`sparse_linear` 逐事件累加，顺序一变，
浮点求和顺序就变，逐位相等便不再成立。`offsets` 倒序 / 越界与容量溢出都由内核
记 `err` 并夹回合法范围，主机侧走与 `embedding` 同一条 `_check_ids`；参考实现
同样拒绝（`ValueError`），对拍的是"错得一样"。

**容量与静态图**：`capacity = T×B×K`，是非零数的上界，构图期一次分配；
`spike_csr` 的 `infer` 直接拒绝 `capacity > T×K`（元素总数都不够）。实际发放
远少于上界时只写前 `nnz` 个位置，多余容量不占计算。

**可微化在 5.136**：两个新算子输出整数，本身没有可微定义。层间压缩的梯度由
`spike_linear` 承担——脉冲梯度走普通 `matmul`，只有权重梯度保持事件散射。5.136
推翻了"事件值梯度 + 散回非零位置"这条最初路线：未发放位置的 surrogate 梯度会
因此被丢掉（初始层一次都不发放时梯度全零）。

**测试**：`tests/test_snn.py` 新增 `TestSnnEventsModel`（5 个用例）——三组深度
（1 / 2 / 3 层）与稠密网络**逐位相等**、两算子的分工与列号顺序、容量溢出报错、
`capacity > T×K` 在构图期拒绝、构件图 JSON 往返。conformance 把 `spike_csr`
用例改成"脉冲 + 起点表"，并新增 `spike_offsets`，CUDA 与 HIP 的"声明即覆盖"
检查因此仍然成立；`_INDEX_ARGS` 给 `spike_csr` 的第 2 个输入开口子，否则整数
起点表会在进入内核之前被"只支持浮点"挡下（HIP 首跑就是这么暴露的）。

**量化（负结果，且已排除"没开权重常驻"这一解释）**：WSL ROCm（gfx1100，DXG
直通），`steps=8 / batch=4 / in=32 / hidden=(64,64,64)`，固定权重与输入、只改
LIF 阈值改变发放率：

| 阈值 | 层0 发放率 | 层1 | 层2 | 稠密 ms | 事件 ms | 比 |
|---|---|---|---|---|---|---|
| 0.25 | 0.451 | 0.336 | 0.337 | 42.26 | 57.70 | 0.73× |
| 0.50 | 0.364 | 0.239 | 0.183 | 46.85 | 64.70 | 0.72× |
| 1.00 | 0.218 | 0.088 | 0.026 | 43.03 | 47.64 | 0.90× |
| 2.00 | 0.083 | 0.002 | 0.000 | 50.33 | 52.31 | 0.96× |

第一轮没有开 `resident_parameters`，所以先怀疑差距只是权重 H2D。开常驻后
（`execute` 只传激活）两列变为 37.63 / 39.04（0.96×）与 38.81 / 38.78（1.00×）
——差距收窄但收益仍是零，说明原因不在上传。

拆开量指向阶段化开销：同一规模下单层 CSR 两趟 1.32 ms、稠密 `matmul` 0.64 ms；
`sparse_linear` 在 32 行上约 1.6 ms 且**与 nnz 无关**（100% / 30% / 10% / 2% /
0% 分别是 1.60 / 1.59 / 1.88 / 1.63 / 1.40 ms）。也就是说，每层多出的两趟压缩
加一次稀疏乘，成本约等于它替换掉的那次小 `matmul`：省下的 FLOPs 被内核启动与
阶段化吃掉了。

放大到 `2048×1024×1024` 并常驻权重，收益才显形（稠密 `matmul` 7.19 ms）：

| 稀疏度 | nnz | 耗时 ms |
|---|---|---|
| 100% | 2097152 | 30.25 |
| 10% | 209715 | 6.49 |
| 2% | 41943 | 4.66 |
| 0.2% | 4194 | 4.17 |

**结论**：`sparse_linear` 的收益门槛约在 10% 稀疏，2% 约 1.54×、0.2% 约 1.72×。
但 0.2% 仍有 4.17 ms 地板：每个事件要读 `W` 的整行（4 KB），行号随机、几乎不
吃 L2，4194×4 KB ≈ 16.8 MB 只对应约 4 GB/s 的有效带宽——瓶颈是访存而不是启动
（最后一句是推断，未用 profiler 验证）。因此事件驱动构件要在真实 SNN 上跑赢
稠密，需要 `T×K` 足够大、发放率低于 10%；压缩两趟里 `spike_offsets` 已由 5.134
改成块内并行（大 `T` 下 19.4×），`spike_csr` 仍是"一行一线程、行内串行"——行内
必须按列号递增写才能与稠密矩阵乘的累加顺序一致，不能并行。目前构件的价值是打通
通路与逐位等价，不是性能；另外稠密基线自身在 42～50 ms 之间波动，第一张表的
差异有一部分落在噪声里，不宜据此调参。

数字：本地 `Ran 1502 ... OK (skipped=307)`；WSL HIP `Ran 1502 ... OK
(skipped=290)`，端到端对拍（`hidden=(10,8)`、`steps=5`、`batch=4`）`max diff
0.0`；CUDA 真机本轮未复跑（研究机不可达）。

---

### 5.134　CSR 压缩从单线程扫描改成块内并行

5.133 的量化把 `spike_offsets` 列成了瓶颈之一：它当时是**一个 block、一个线程**
从头扫完 `T×K` 再累加，串行步数随 `T×K` 线性增长。小 `T` 时这被 launch 开销盖住
（32×64 只要 1.06 ms），但真实 SNN 的时间维一长就露馅：8192×512 的两趟压缩合计
需要 205.6 ms。

改成**单 block 两阶段**：

1. 阶段 1：每线程按 `blockDim` 跨步认领若干行，只数这一行的非零个数，写进
   `offsets[t+1]`——把输出缓冲当临时区用，不新增一块 i32 scratch；
2. `__syncthreads()`；
3. 阶段 2：0 号线程把 `offsets[1..T]` 就地变成前缀和（`offsets[t] = acc`，
   `acc += 原 offsets[t+1]`）。读的是 `t+1`、写的是 `t`，从前向后不会覆盖还没读
   的值。

前缀和的串行步数因此从 `T×K` 降到 `T`，阶段 1 的读则铺满一个 block。仍然是
**单输出、单内核**，不需要新增 L2 能力（`Sync` 与 `Axis("bdim")` 归约族早就在用），
也不需要把 `steps` 做成编译期常量——步长与行数都可以是运行期标量。

**为什么不铺多个 block**：块间没有 `__syncthreads`。跨 block 的前缀和要么再开一个
内核（节点数 2 → 3，小规模下每多一个节点就是一次 launch 的固定成本），要么上
原子加 + 二次遍历，两者都比"单 block 多跑几百步"更贵。这条边界也解释了为什么
`block = min(256, steps)` 写死：它是"一个 block 能覆盖多少行"与"每线程扫多少列"
的折中，不是可调优旋钮。

**实测（HIP / gfx1100，两趟合计，`spike_csr` 未改所以加速比被它摊薄）**：

| steps×width | 串行 ms | 并行 ms | 加速 |
|---|---|---|---|
| 32×64 | 1.06 | 1.00 | 1.06× |
| 256×64 | 1.71 | 1.17 | 1.46× |
| 2048×64 | 6.53 | 1.38 | 4.7× |
| 8192×64 | 26.18 | 5.06 | 5.2× |
| 2048×512 | 49.60 | 5.33 | 9.3× |
| 8192×512 | 205.60 | 10.62 | 19.4× |

小规模只快 6%——这正是 5.133 的端到端结论不变的原因：构件规模下两趟压缩本来就
被 launch 主导，优化的是"大 `T` 会爆炸"这条尾巴，不是当前用例。

**测试**：conformance 新增 `spike_offsets_strided`（`steps=300 / width=8`，行数
超过一个 block），钉住跨步循环与前缀和顺序；参考后端 79 executed，HIP 74
executed / 5 refused / 0 failed，本地与 HIP 全量各 1502 个用例全绿。真机上还
核对过 `steps < block`（3×512）、跨步（257×64，f16）与 1000×37 三组随机脉冲，
`offsets` / `indices` 与参考后端逐位一致。

---

### 5.135　hipGraph 捕获下的两处主机侧语义

5.134 收尾后按计划量"捕获口径下 dense vs events"，一上手 events 图就在
`capture()` 的预热轮抛 `spike_csr 收到越界下标 [0, 2048]`。逐节点执行正常、
CSR 两个算子的单图捕获也正常——问题只出现在"整张 events 图 + 捕获"这个组合里。
追下来是两处**主机侧**语义在 HIP / WSL 上不成立，与 GPU 算得对不对无关。

#### 一、同步 `hipMemset` 在 DXG 上不保证已经落地

证据链：

1. 预热轮里 `spike_offsets` 返回后立刻 `hipDeviceSynchronize`，`offsets[32]`
   仍是 0（正确值 447），而 `offsets[0..31]` 全部正确；
2. 同一个内核放到默认流上跑，或 launch 前做一次设备级同步，结果正确；
3. 最小复现：`zeros`（默认流上的同步 `hipMemset`）→ 私有流 launch，最后一个
   元素丢失；`zeros` 之后补一次 `hipStreamSynchronize(0)` 即恢复正常。

结论：`hipMemset` 返回时命令可能仍留在默认流上；而捕获预热轮的内核跑在
**非阻塞私有流**上，两者没有任何顺序关系。memset 在 kernel 之后落地，把
`spike_offsets` 刚写好的最后一个元素清零，`spike_csr` 于是看到 `hi < lo`。

修法在 `hip/runtime.py`：非默认流上的 `memset` / `memcpy` 一律排进当前流并等它
完成；默认流的同步 `hipMemset` 之后补一次 `hipStreamSynchronize(0)`。`memcpy`
也走同一分支——既保证与内核的顺序，也保证"函数返回后 H2D 的源可以安全释放"。

#### 二、捕获期 H2D 的主机源活不过 `capture()`

修好第一处之后捕获能过，但 replay 输出全 0、常数变成 `1e-45` 量级。分层定位到
`broadcast_to(constant(1.0), (2048,))`：普通执行是 `[1.0] * 2048`，捕获后 replay
全是位模式 4（整数 4 的 f32 解释）。

根因：`full` / `array` 的源是函数内的临时 numpy，捕获期 `hipMemcpyAsync` 只把
**主机地址**记进图；函数返回后源就释放了，replay 拷的是已释放的内存。

修法：`_CaptureSession` 增加 `host_refs`，捕获期的 H2D 先把源拷进一块 `ctypes`
缓冲并登记；捕获结束把这份列表交给 `HipGraphPlan` 持有到销毁。CUDA 侧不需要这
一层（cupy 的捕获路径自己会把主机数据落进 pinned 缓冲，见 5.86）。

#### 三、回退一处为排障加的补丁

排查过程中曾把 `_check_ids` 改成"预热轮只登记、`capture()` 在
`stream.synchronize()` 之后统一复查"（`_warmup_guards`）。根因修好后不再需要：
`err.item()` 的 D2H 现在会等在私有流上排队的 kernel，于是恢复"立即回读"，
HIP 与 CUDA 两侧同时删掉这段分支，`_check_ids` 回到单一语义。

#### 测试

`test_constant_broadcast_survives_capture` 钉住第二处——常量 replay 之后必须仍是
`1.0`；`test_snn_events_graph_captures` 钉住整条链——events 图的逐节点结果与捕获
replay 逐位一致（同时覆盖常量物化、`spike_offsets` 的前缀和、`spike_csr` 的整数
输出与 `sparse_linear` 读同一份 offsets）。

`test_default_stream_memset_precedes_private_stream_write` 钉住第一处：默认流清零
之后，私有流上的内核只写最后一个元素，读回必须是 1。把 `memset` 后的流同步临时
去掉，这条在 WSL 上实测变红（读回 0），恢复后通过；竞争窗口依驱动与环境而定，
本机不一定每次都能命中原始时序，整条链仍由 events 那条守门。

三条口径：本地 `Ran 1505 ... OK (skipped=310)`（1195 通过）；WSL HIP
`Ran 1505 ... OK (skipped=290)`（1215 通过）；CUDA 真机（Tesla M40 / sm_52 /
Python 3.10 / cupy 14.2）`Ran 1500 ... OK (skipped=77)`，0 失败（新增的流顺序
用例属 HIP 专属类，在 CUDA 上整类跳过）；3.10 与 3.14 的计数差 5 个是未装 cuDNN
的用例（`setUpClass` 抛 `SkipTest` 在 3.10 不计入 `testsRun`），不是缺测。

#### 捕获口径下的 dense vs events

修好之后把原定那组对比量了：`hidden=(64, 64, 64)`、`steps=8`、`batch=4`，
两条路径同阈值（1.0），HIP 上取 20 次中位数，单位 ms。

| 路径 | 逐节点 | 捕获 replay |
| --- | --- | --- |
| dense | 37.965 | 2.746 |
| events | 46.212 | 3.171 |

两边捕获后都与逐节点结果逐位一致，events 与 dense 的最大差也是 0。捕获把
launch 开销摊掉后（约 14× / 15×），剩下的就是内核本身：这个宽度下稀疏度
不足以抵消 `spike_offsets` / `spike_csr` 的压缩与 `sparse_linear` 的散射，
events 捕获比 dense 慢约 15%（cap ratio 0.87×）。结论只到"本规模不占优"：
events 的收益要等宽度与稀疏度上去才会显现，这里不拿一组小图外推。

**结论**：这两处都不在"GPU 算得对不对"的层面，而是"捕获期的主机侧动作有没有
顺序与生命周期"。hipGraph 记录的是地址与命令流，任何"函数返回即失效"的主机缓冲
都会在 replay 时变成随机值；而同步 API 的名字在 WSL / DXG 上不能当作"已完成"的
保证。后续任何要进捕获的路径（调优、训练 step、设备态参数）都依赖这两条修复。

---

### 5.136　事件路径的可微化：脉冲梯度走稠密，权重梯度仍走稀疏

5.133 把 `build_snn_events` 定位成"只支持前向"，理由是 CSR 的两个整数输出没有
可微定义。这一节把它变成可训练，并在过程中推翻了一个看似自然的设计。

#### 一、为什么"事件值梯度 + 散回非零位置"是错的

最初的设计是三步：`spike_csr_values` 把脉冲矩阵的非零值拷进 `values` 缓冲、
`sparse_linear_values_grad` 算 `dvalues`、`spike_csr_values_grad` 按 offsets 把
`dvalues` 散回稠密。它在非零位置与稠密路径一致，但**默认未发放位置的梯度为 0**。

这个默认不成立。LIF 的 `v_next = v_pre - threshold·s` 让 `s` 同时参与重置，
`s` 的梯度经 `spike` 的替代梯度回到 `v_pre`；替代梯度只看膜电位是否落在
`(threshold-alpha, threshold+alpha)` 窗内，与这一拍有没有发放无关。实测
（`input_dim=6, hidden=(8,8), steps=4, batch=3`，第一层第一步）：

| 量 | 值 |
|---|---|
| 发放率 | 0.000（全零） |
| 未发放位置 `max|∂L/∂s|` | 2.1966 |
| 若只散回非零事件，梯度 | 全零 |

也就是说，初始层一次都不发放时稠密路径仍有梯度，而"只散回非零事件"会让整条
事件路径的梯度恒为零——不是精度问题，是学不动。所以反向必须让脉冲拿到完整的
`(S×K)` 梯度，稀疏只能落在权重梯度上。

#### 二、两个融合算子

把"取事件值"与"稀疏乘"合成一个算子，前向仍逐事件，反向对脉冲直接复用
`matmul` 的规则：

| 算子 | 输入 | 输出 | 语义 |
|---|---|---|---|
| `spike_linear` | `spikes[S,K]`、`weights[K,N]`、`indices[E]`（i32/i8）、`offsets[S+1]`（i32） | `[S,N]` | `out[s,n] = Σ_e spikes[s, indices[e]]·W[indices[e], n]`，`e` 只扫第 `s` 段 |
| `spike_linear_weight_grad` | `dout[S,N]`、`spikes[S,K]`、`indices`、`offsets` | `[K,N]` | 把 `spikes[s,r]·dout[s]` 原子累加回 `r` 行 |

`spike_linear` 的 VJP 只做三件事：

- `dspikes = matmul(dout, Wᵀ)`，与稠密 `matmul` 的反向**同构**，因此逐位一致；
- `dweights = spike_linear_weight_grad(dout, spikes, indices, offsets)`，只扫事件；
- `dindices` / `doffsets` 为 `None`：CSR 结构视为常量，这是"脉冲位置由前向阶跃
  决定、梯度只沿脉冲值流动"的既定语义（与 `spike` 的替代梯度同源）。

融合的收益不只是少一个算子：前向不必再物化 `capacity` 长度的 `values` 缓冲
（原来那条路径是 `broadcast_to(constant(1.0))`，或一份脉冲拷贝），少一趟读写；
反向也不必再构造"散射回稠密"的第三个内核。前向求和顺序仍是列号递增，跳过的
都是 0 项，因此与稠密 `matmul` **逐位相等**（5.133 的端到端对拍继续成立）。

#### 三、接线

- `ir/ops.py`：两个算子的 `infer` + 参考实现；`spike_csr` 的 offsets / capacity
  校验抽成 `_check_spike_offsets` / `_check_event_capacity` 复用，行为不变。
- `ir/grad_rules.py`：`@vjp("spike_linear")`。
- `lower/kernels.py`：两个 L2 描述并导出；`spike_linear` 与 `sparse_linear` 共用
  `_event_segment` / `_event_row_check`，只是把 `values[e]` 换成 `spikes[s, r]`。
- `cuda/codegen.py`：import 与 `spike_linear_source` /
  `spike_linear_weight_grad_source`；HIP 逐字复用同一份 L2。
- `device_backend.py`：`GPU_OPS`、`_INDEX_ARGS`（indices 与 offsets）、
  `_register_kernels()` 注册、两个 `_op_*`（输出先清零再原子累加）。
- `models/snn.py::_event_layer`：`sparse_linear + ones` 换成 `spike_linear`，
  `build_snn_events` 的"只支持前向"声明删除。
- `conformance.py`：两个用例，脉冲值刻意不取 1（否则"把幅度当常量"的实现也能
  对），并保留一个空步（否则"空步沿用上一行"的实现也能对）。

#### 四、测试与口径

`tests/test_snn.py` 新增 `TestSpikeLinear`（6 个）：前向对稠密 `matmul`、空步为 0、
脉冲梯度等于 `dout·Wᵀ`、**零脉冲位置梯度非零**、权重梯度等于 `spikesᵀ·dout`、
JSON 往返。`TestSnnEventsModel` 与 `TestSnnTraining` 新增三条端到端：

- `test_gradients_match_dense_network`：`x` 与全部参数对拍稠密路径，`rtol=atol=1e-6`
  （权重梯度是原子累加，差异只在求和顺序）；
- `test_gradients_match_dense_on_device`：同一组对拍在 HIP / CUDA 上跑，容差 1e-5；
  没有 GPU 运行时时整条跳过；
- `test_sgd_on_event_path_reduces_loss`：两层（第 0 层稠密、第 1 层 CSR 事件）上用
  同一套 SGD 跑 60 步，损失压到初值的 0.2% 以下（判据取 `after < before/2`）。
  梯度对拍证明"导数算对了"，这条证明" optimizer 真能沿它走"。

口径：本地 `Ran 1514 ... OK (skipped=311)`；WSL HIP `Ran 1514 ... OK
(skipped=290)`；CUDA 真机（Tesla M40 / sm_52 / Python 3.10 / cupy 14.2）
`Ran 1509 ... OK (skipped=77)`，0 失败。三条口径里设备上的事件梯度对拍都通过
（参考 1e-6、设备 1e-5）。真机的 `testsRun` 比本地少 5 个是未装 cuDNN 的用例
（`setUpClass` 抛 `SkipTest` 在 3.10 不计入），不是缺测。

### 5.137　事件路径的训练步成本：两条路径持平

5.136 让 `build_snn_events` 可训练之后，"事件驱动到底省不省"才第一次能按
**训练步**量，而不是只看前向。结论先说：**整步持平，收益门槛没有因为可微化而降低**。

#### 一、口径

同一份 `SNNSpec(input_dim=64, hidden=(64,64,64), steps=8, batch=4)` 分别构
`build_snn_network` 与 `build_snn_events` 的前向图和反向图；同一组输入与
`dout`，参数个数相同。每张图预热 3 次、取 12 次的最小值与中位数。逐节点
`execute`（非捕获）——反向图与参数更新不进 CUDA Graph，所以捕获口径只适用于
前向，不能代表训练步。

| 后端 | 路径 | 前向 min | 反向 min | 整步 events/dense |
|---|---|---|---|---|
| HIP RX 7650 GRE（WSL2） | dense | 36.66 ms | 175.10 ms | — |
| | events | 43.11 ms | 170.19 ms | **1.01×**（前向 1.18×，反向 0.97×） |
| CUDA Tesla M40 | dense | 13.36 ms | 52.99 ms | — |
| | events | 13.83 ms | 52.62 ms | **1.00×**（前向 1.04×，反向 0.99×） |

#### 二、为什么持平

前向多出的部分是每层两趟压缩（`spike_offsets` + `spike_csr` 各读一遍
`S×K` 脉冲）加一次按事件扫的稀疏乘。HIP 上这部分让前向慢 18%；M40 上只慢
4%，说明压缩是访存型、在更宽的显存上被摊薄，而不是算力瓶颈。

反向则由两笔相反的账抵掉：

- **省**：权重梯度从稠密 GEMM（`spikesᵀ·dout`）变成事件散射，只碰实际发放的行；
- **费**：反向图为了拿到 `indices` / `offsets`，会把压缩重算一遍；`dspikes` 仍是
  与稠密路径同价的稠密 GEMM（5.136 已解释：只散回非零事件会丢掉未发放位置的
  surrogate 梯度）。

两笔相抵，反向在两条后端上分别快 3% 与 1%，落在噪声量级。整步因此持平。

#### 三、结论与边界

- 事件路径的价值是**语义完整**：训练不再被迫换回 `build_snn_network`，而不是
  性能。要看到收益，仍得满足 5.133 的量级条件（约 10% 稀疏、`K`/`N` 足够大）。
- 想让训练步真正变快，下一步不是继续改 `spike_linear`，而是压缩本身。
  `spike_csr` 的行内并行已经试过（5.138），是负结果：按列段切分破坏了合并访存，
  2048×4096 上反而慢 87%。要动它得先给 L2 加 warp 级扫描原语，暂不做。
- 未测：捕获口径的训练步（反向与优化器更新不进图）、f16、以及跨 batch/T 的
  规模扫描。这张表只回答"同配置下两条路径谁快"，不回答"最优稀疏度在哪"。


### 5.138　`spike_csr` 的行内并行：访存布局比并行度更重要（负结果）

5.137 把训练步的成本定位到压缩后，"把 `spike_csr` 的行内串行并行掉"看起来是
唯一还剩的方向：旧实现一行一个线程、行内扫 `width` 步，2048×4096、2% 稀疏下
要 9.1 ms，而并行度只有 2048。试了，结论是**负收益，已回退**。

#### 一、做法

一行一个 block（128 线程）。线程 `tid` 负责连续列段
`[tid*chunk, min(width, (tid+1)*chunk))`：第一趟数本段命中数写进共享数组，
`__syncthreads()` 后 0 号线程串行累加 128 个计数（而不是 `width` 步），再同步
一次，各线程从自己的起点按列递增写。段按 `tid` 递增切分，所以整行仍是升序，
`sparse_linear` 与稠密路径的逐位对拍不受影响——测试确实全过。

#### 二、实测（HIP RX 7650 GRE，min of 8，同口径）

| 规模（steps×width） | 回退前 | 行内并行 | 回退后复测 |
|---|---|---|---|
| 32×64 | 0.57 ms | 0.61 ms | 0.65 ms |
| 512×256 | 0.77 ms | 0.99 ms | 0.88 ms |
| 2048×1024 | 2.73 ms | 5.10 ms | 3.01 ms |
| 2048×4096 | 9.09 ms | **16.97 ms** | 10.53 ms |

并行版在所有规模上都更慢，大 `width` 上慢 87%。

#### 三、原因

为了让段内列号升序，每个线程读的是**连续列段**：warp 内 32 个线程的地址相隔
`chunk` 个元素（4096 宽时是 32，即 128 字节）。这等于把旧实现"每线程一条独立
顺序流"的访存，换成了跨步访存——旧实现虽然行间也不合并，但每个线程有完整的
顺序流，硬件预取与内存级并行都能利用。再叠加两趟读 `spikes` 和每块 128 步的
串行前缀和，并行度带来的收益被访存损失吃掉了。

#### 四、可行的做法（未做）

块内先按 `tid` 交错、合并地把脉冲读进 `extern __shared__`，再做段内顺序扫描；
前缀和用 warp shuffle 而不是单线程串行。这需要给 L2 加扫描原语与动态共享内存
（`width` 目前是运行期标量），已经不是"改一个内核"的量级。在收益只出现在
大 `width`、而当前 SNN 配置 `width=64` 的前提下，暂不做，保留"一行一线程"。

结论：这条路径的瓶颈是**访存布局与行内串行**，不是并行度。5.137 的"下一步
并行化 `spike_csr`"到此有了实测答复——不做。


### 5.139　模糊测试的覆盖面：把几何类算子接进随机图

第 7 节缺口表里"模糊测试只覆盖形状 / 数值类算子"一直是中优先级。5.38 明确
写过不把卷积 / 池化 / attention / scan / embedding 塞进生成器，理由是 attrs
组合大半会被 `infer` 拒绝、覆盖效率低。这条判断对"随机取属性"成立，但可以
绕过去：**随机化只发生在合法区间内**——先按素材形状把属性算到自洽，再在
合法区间里取随机值，失败原因就不会和"随机取到非法几何"搅在一起。

#### 一、做法

四个固定素材输入（形状固定，数值由独立 rng `seed ^ 0xA5A5` 生成，不占用
主 rng 的随机序列，也不进 pool——常规模板不会把图像 / 查表素材当成任意
中间值来用）：

| 素材 | 形状 / dtype | 用途 |
|---|---|---|
| `img` | (1, 2, 6, 6) f32 | conv2d、maxpool2d、avgpool2d |
| `tri` | (2, 3, 4) f32 | attention 的 q（批维 2、序列 3、内维 4） |
| `tab` | (8, 4) f32 | embedding 的词表 |
| `ids` | (3, 2) i32 | embedding 的下标，值域 [0, 8) |

五个新模板先算合法区间、再随机：

- `conv2d`：`groups ∈ {1, C}`（25% 走 depthwise）、`o ∈ [1, 3]`、核 1..3、
  stride 1..2、padding 0..1、dilation 1..2；先算输出高宽，非正就跳过。
  权重与 bias 用随机常量（`data` 存嵌套 list，序列化往返仍逐字段相等）。
- `maxpool2d` / `avgpool2d`：同一套几何检查，只放开核 / stride / padding。
- `attention`：q 固定为 `tri`，k / v 用随机常量造（`T ∈ [1, 3]`、`E ∈ [1, 3]`），
  mask 用有限负值（整行 `-inf` 会让 softmax 出 nan）；30% 带 mask、30% 带
  `dropout=0.25`。
- `scan_linear`：从 pool 里挑一对同形状（秩 ≥ 2）的值当 a / b，h0 用常量。
- `embedding`：`tab × ids`，一半机会让 ids 先过一层 slice（输出仍是 i32）。

变异器补了三个：conv 的 `groups = C + 1`（不被输入通道整除）、池化的
`kernel = (0, 1)`、attention 的 `dropout = 1.0`。

#### 二、结果

| 口径 | 结果 |
|---|---|
| 200 个种子 | 200 例、199 次非法输入被拒绝、1 次无可用变异、**覆盖 31 个算子**，全部通过 |
| 1000 个种子 | 同上，全部通过 |
| 覆盖速度 | 40 个种子即覆盖全部 31 个算子（`run_fuzz(40)`） |
| 本地用例 | `tests/test_fuzz.py` 10 条通过，新增 `test_coverage_reaches_the_geometry_family` |

算子覆盖从 25 增到 31：新增 conv2d、maxpool2d、avgpool2d、attention、
scan_linear、embedding；`slice` 也因为 embedding 的下标切片多了一条 i32 路径。

#### 三、边界

- **种子到图的映射变了。** 模板池从 15 增到 20，同一个种子在 5.38 与 5.139
  下生成的图不同。种子承诺的是"同一版本内可复现"，不是跨版本稳定。
- conv / 池化只接 `img`，attention 的 q 只接 `tri`：中间结果里 rank 4 或
  批维对齐的素材很少，接上去大半会跳过。要把它们接进图的主体，需要生成器
  能反向"按目标形状造素材"，那是另一层设计。
- **符号维仍不随机化**，随机图只用静态形状（理由同 5.38）。
- **仍只跑参考执行器。** 随机图不上 CUDA / HIP；跨后端一致性仍靠 conformance
  的固定图集。这一条对应第 7 节风险行的新表述。
- 变异仍只看"非法属性是否被拒"，不构造语义错误（例如越界下标、除零）。


### 5.140　调优测量改成交替 A/B：区间有了，赢家仍然不可复现

5.63 的结论是"同一旋钮在两家的最优值是否相同"在当时的测量精度下不可回答，
并留下一个前提：先把测量改成交替 A/B + 置信区间，再谈搜索算法；第 7 节
风险行里记的也是这条。这一节交付那个前提，也照实记录它没能解决的问题。

#### 一、做法

新增 `aetherflow/tuning/paired.py`（纯逻辑，不碰后端）：

- **轮内连测**：每一轮把所有候选连着测一遍，共享同一段整机状态；
- **逐轮反转**：偶数轮正序、奇数轮反序；
- **配对差**：同一轮内 `candidate / baseline - 1`，整机漂移在比值的分子分母
  里同时出现，被约掉；
- **区间**：对配对差做 2000 次 bootstrap，给均值差的 95% 百分位区间
  （固定 `seed=0`，同一批样本永远得到同一个区间；不引入 scipy）；
- **判定**：`faster` / `slower` 要求区间整体越过 0 **且**中位差超过噪声地板
  （默认 1%），否则记 `noise`、保留默认；
- **显式降级**：某个候选抛异常就立即停测并隔离；基线不可行时所有候选记
  `no-baseline`，只按绝对中位数挑一个"能跑的"，`decided=False`——不拿别的
  候选当基线编造差异；
- **批次稳定性**：把轮次按时间切成前后两半，两半的配对差中位数必须同号，
  异号记 `unstable` 并按噪声处理（加它的原因见结果里的反例）。

接线：

- `benchmarks/tune.py` 默认走配对测量（8 轮），`--serial` 保留旧测法，
  `--noise-floor` 可调；逐候选打印 Δ中位、均值区间与判定；
- `benchmarks/tune_compare.py` 报告格式升到 v3，逐候选记录
  `verdict / diff_* / ci_* / first_half_pct / second_half_pct`；顶层与条目都带
  `method`，串行与配对不混读（v1/v2 报告拒读）。

#### 二、结果

HIP（WSL2 + DXG）上 `--rounds 8`、噪声地板 1% 采了三次：

| 采集 | 逐元素 8 用例 | conv2d 赢家 | 配对差中位 | 95% CI | 判定 |
|---|---|---|---|---|---|
| 1（只有 CI，输出被 `tail` 截断） | 可读的 3 个用例均保留默认 | `tile_w=64 / roc=4 / cb=None` | -15.84% | [-21.98, -1.89] | 更快 1.20× |
| 2（只有 CI） | 7 个保留默认；add_broadcast 判 `block=512` 更快 | 默认 | 该候选 +6.49% | [+2.39, +19.43] | 保留默认 |
| 3（CI + 稳定性） | 8 个全保留默认；13 个候选批次不稳 | `tile_w=None / roc=2 / cb=2` | -8.45% | [-11.28, -1.61] | 更快 1.09× |

三条关键观察：

1. **同一配置两次采集方向相反，而且两次都"显著"。** `tile_w=64 / roc=4`
   在采集 1 是 -15.84% 判更快，采集 2 是 +6.49% 判更慢。原因是轮内位置效应：
   conv2d 空间有 19 个候选，一轮约 1 秒，整机状态在轮内并不单调（升频后回落），
   正序时该配置偏后、反序时偏前，两头都容易落在"慢"的位置。**反转顺序只能
   抵消单调趋势，抵消不了 U 形的位置效应**，而 bootstrap 把 8 个自相关轮次当
   独立样本，区间因此过窄。这是本次实测最重要的发现，也是加"两半同向"检查的
   直接原因。
2. **批次稳定性检查有效，但不充分。** 采集 3 里 13 个逐元素候选被拦下
   （包括采集 2 判过"更快"的 add_broadcast `block=512`），8 个用例全部回到
   默认——这正是它该起的作用；但 conv2d 的赢家两半同号，仍被判 `faster`，
   且与采集 1、2 的赢家都不同。
3. **三次采集三个赢家。** 1.20× / 1.00× / 1.09×。conv2d 旋钮的收益量级确实
   在 5%～15%，但"哪个配置"仍不可复现。5.63 的"不可回答"因此变成了
   "能给出区间、能拦下大部分伪显著，但仍选不出一个跨采集稳定的赢家"——
   不是"已解决"。

#### 三、边界

- 只消除**轮内共同漂移**；轮内非单调的位置效应仍在（观察 1），这是当前
  最主要的残余偏差来源。
- bootstrap 假设轮次独立同分布，实际自相关；两半同向只是启发式，不提供
  统计保证，也不覆盖"位置"这类系统性偏差。
- 只在 HIP（WSL2 + DXG）实测；CUDA 侧待重采（旧 v2 报告已拒读）。
- 配置库仍只记赢家，CI 与 verdict 只在 `tune_compare` 的报告里，
  `TuningRecord` 未扩展——因为"赢家不可复现"这件事本身还没解决。
- 8 轮 × 19 候选的测量成本明显高于旧的 `--rounds 2`；没有做"成本 vs 证据"
  的自适应。4 轮（`tune_compare --rounds 4` 的端到端验证）又给出三个不同的
  逐元素赢家（add_broadcast 128、sigmoid 512、dropout_mask 256，均"已判定"），
  说明"两半各 2 轮"时同向检查形同虚设；8 轮是当前能用的下限，不是充分条件。

#### 四、下一步

- 每轮用固定种子**随机排列**候选（而不是正/反两向），把轮内位置效应打散成
  噪声；或每轮开始插入固定预热与固定间隔，让轮内状态稳定。两条都要先做
  A/B 对照，再判断是否值得。
- 在解决位置效应之前，配置库仍然只能当"记录"，不能当收益证据；
  `--serial` 只用于复现旧口径。

#### 五、验证

- 本地 `py -3.14 -m unittest discover -s tests`：**1529 tests OK
  (skipped=311)**，新增 14 条（12 个 `TestTunePaired` + 1 个串行报告用例
  + 1 个"未判定赢家加星标"的对照打印用例），覆盖顺序反转、共同漂移、
  顺序效应抵消、真实收益、噪声地板、批次不稳、不可行候选隔离、基线不可行、
  全拒、参数校验、种子可复现、JSON 可序列化。
- WSL HIP：`tests.test_tuning` + `tests.test_tune_compare` 共 59 条通过。
- CUDA 真机（Tesla M40 / Python 3.10）：全量 **1523 tests OK (skipped=77)**；
  其中 `test_cuda_backend` 的"搜索产出可复用配置库"用例从 `rounds=1` 改为
  `rounds=2`（配对测量不接受 1 轮）；`ci.yml` 的调优往返步骤同样从 1 轮改为
  2 轮，仍覆盖默认的配对路径。
- HIP 实采三次，报告写在临时目录，未入库。


### 5.141　轮内随机排列：位置偏置被摊平了，赢家仍不可复现

5.140 把残余偏差定位到"轮内非单调的位置效应"，下一步写的是每轮随机排列。
这一节实现它，并用 A/B 采集回答"换了排列，赢家会不会稳定"。

#### 一、做法

- `tune_paired(order="shuffle")`：每轮用 `default_rng(seed ^ 0x5EED)` 生成排列。
  排列与 bootstrap 用两个独立 rng，避免同一个 `seed` 让"顺序"和"重采样"产生
  人为相关（异或常量沿用 `fuzz.py` 的做法）；
- `PairedResult` 增加 `order_mode`，`order` 记录每轮实际排列，报告可复现；
- `tune.py` / `tune_compare.py` 增加 `--order {alternate,shuffle}`；默认仍是
  `alternate`，换默认要先有 A/B 证据；
- 新增 3 条测试：每轮恰好覆盖全部候选、同种子排列可复现、不同种子排列不同。

#### 二、结果

HIP（RX 7650 GRE / DXG），`--rounds 8`，每次采 4 个用例
（conv2d / add_broadcast / relu_1m / dropout_mask_1m），按
alternate → shuffle → alternate → shuffle 顺序各采一次：

| 次 | order | conv2d 赢家 | 配对差中位 | 95% CI | 判定 | 批次不稳 |
|---|---|---|---|---|---|---|
| 1 | alternate | `tile_w=64 / roc=4 / cb=2` | -7.68% | [-13.63, -1.87] | 更快 1.07× | 26/45 |
| 2 | shuffle | `tile_w=auto / roc=4 / cb=1` | -6.81% | [-12.69, -0.90] | 更快 1.09× | 29/45 |
| 3 | alternate | 默认 | 0.00% | [0.00, 0.00] | 噪声内 | 41/45 |
| 4 | shuffle | 默认 | 0.00% | [0.00, 0.00] | 噪声内 | 12/45 |

逐元素用例：第 1 次 `add_broadcast` 判 `block=128` 更快 1.07×；第 4 次
`dropout_mask` 判 `block=512` 更快 1.02×；其余全部保留默认。

三条结论：

1. **随机排列没有让赢家稳定。** 两种顺序都是"一次判收益、一次判噪声"；
   把 5.140 的三次也算上，7 次 8 轮采集里 conv2d 有 4 次判收益、3 次判噪声，
   而 4 次判收益的赢家各不相同（`64/4/auto`、`auto/2/2`、`64/4/2`、`auto/4/1`），
   没有一次重复。
2. **主因不是排列，而是整机状态。** "批次不稳"的候选数在 4 次采集里是
   26 / 29 / 41 / 12（共 45 个可行候选），波动与 order 无关。随机排列只是把
   同一个整机漂移映射成了另一种形状的偏差：第 3 次 41/45 个候选两半异号，
   说明那 18 秒里整机换过方向，这与某个候选快慢无关。
3. **轮内跨度是更接近根因的变量。** 逐元素的轮内跨度约 0.26 秒（4 个候选），
   7 次采集里只有 3 次判过一次收益，且都没有在下一次重复；conv2d 的轮内跨度约
   2.3 秒（45 个候选 × 13 次执行 × ~4ms），7 次里 4 次判收益、3 次判噪声。
   候选越多、一轮越长，结论越依赖"这一轮恰好赶上了什么"。

#### 三、边界

- `shuffle` 不消除轮内漂移，只改变漂移与候选的对应关系；8 轮不足以让它
  在均值上抵消。它把偏置转成了方差，表现为区间更宽、`unstable` 更多。
- 只在一个 HIP 环境实测；CUDA 侧未采。
- 没有做"每轮固定预热"，也没有缩短轮内跨度——这两件都还没验证。

#### 四、下一步

- **缩短轮内跨度**，而不是继续换排列：conv2d 的 45 个候选可以拆成两阶段
  （先粗扫 `tile_w` 4 档，再在选中的 `tile_w` 下扫 `roc × cb`），单轮降到
  约 10 个候选、轮内跨度约 0.5 秒；
- 或者把"轮内跨度"本身作为测量参数：跨度超过阈值时拒绝下结论，而不是给
  一个看起来很窄的区间。

#### 五、验证

- 本地 `py -3.14 -m unittest discover -s tests`：**1532 tests OK
  (skipped=311)**（5.140 的 1529 + 3 条 shuffle 用例）。
- WSL HIP：`tests.test_tuning` + `tests.test_tune_compare` 共 62 条通过。
- HIP A/B 采集 4 次，报告在临时目录，未入库。


### 5.142　轮内跨度作为拒绝下结论的门槛：拦住了 conv2d，也暴露了首轮冷启动

5.141 的下一步留了两条路：缩短轮内跨度，或把"跨度"本身作为测量参数。这一节
走第二条——不改变测量方式，只给"这一轮是否还共享同一段整机状态"加一个可检查
的上限。

#### 一、做法

- `tune_paired(..., max_round_seconds=None)`：默认不检查，保持 5.140/5.141 的
  口径不变；非 `None` 时必须为正数，`True` / 字符串显式拒绝（不做隐式转换）；
- 每轮用 `time.perf_counter()` 包住"该轮全部候选的 measure"，记录
  `PairedResult.round_seconds`；任一轮超过阈值，就把**所有非默认候选**的
  `verdict` 从 `faster` / `slower` / `noise` 覆盖为 `long-round`，基线仍是
  `baseline`。区间、中位差、两半差照常算——读者能看到"差多少"，只是这个差
  不再作为结论；
- `decided` 因此为假，赢家保持默认；`PairedCandidate.stable` 把 `long-round`
  与 `unstable` 同等看待；
- `tune.py` / `tune_compare.py` 增加 `--max-round-seconds`（默认不检查；
  `--serial` 下显式报错而不是静默忽略）；报告落盘 `round_seconds` 与
  `max_round_seconds`，打印层把 `long-round` 翻成"跨度超限"；
- 新增 6 条测试：跨度超限时基线判定不受影响、阈值高于实际跨度时判定不变、
  不设阈值时行为不变、非法参数（0 / 负数 / bool / 字符串）被拒，以及
  `tune_compare` 报告记录跨度并在超限时拒绝结论。

#### 二、结果

HIP（RX 7650 GRE / DXG），`--rounds 8 --max-round-seconds 1.0`，一次采 4 个用例：

| 用例 | 每轮跨度（秒，按轮次） | 最长 | 判定 |
|---|---|---|---|
| add_broadcast | 0.873 / 0.061 / 0.064 / 0.063 / 0.059 / 0.058 / 0.058 / 0.059 | 0.87 | 未超限；`block=512` 1.03×，Δ -5.35% [-8.40, -1.18]，已判定 |
| relu_1m | 0.860 / 0.388 / 0.427 / 0.418 / 0.405 / 0.431 / 0.423 / 0.413 | 0.86 | 未超限；`block=256` 1.05×，Δ -5.68% [-6.51, -1.22]，已判定 |
| dropout_mask_1m | 1.601 / 0.381 / 0.348 / 0.393 / 0.421 / 0.406 / 0.415 / 0.447 | 1.60 | 超限；4/5 个可行候选 `long-round`，保留默认 |
| conv2d_tiled_1m | 3.334 / 2.621 / 2.240 / 2.435 / 2.211 / 2.386 / 2.356 / 2.453 | 3.33 | 超限；44/45 个可行候选 `long-round`，保留默认 |

四条结论：

1. **门槛按预期拦住了 conv2d。** 8 轮全部落在 2.2～3.3 秒，44 个非默认候选
   全部降级，赢家保持默认。这是第一次让"这一轮太长"直接成为拒绝理由，而不是
   靠两半异号间接拦下（5.141 第 3 次是 41/45 两半异号）。
2. **门槛也误伤了 dropout。** 只有第 1 轮 1.60 秒超限，后 7 轮是 0.35～0.45 秒；
   "任一轮超限即整次降级"把 7 轮有效数据一起作废。对这类用例，1.0 秒阈值实际
   变成了"首轮冷启动检测器"。
3. **首轮系统性偏慢，不是随机噪声。** 四个用例的第一轮分别是后续轮次中位数的
   约 14×、2.1×、4.0×、1.3×（0.87 vs 0.06、0.86 vs 0.41、1.60 vs 0.40、
   3.33 vs 2.4）。这比"位置效应"更像冷启动（首次 kernel 加载 / 升频 / 页表
   预热），也补上了 5.140 "轮内非单调"的另一半解释：第一轮总在曲线上端。
4. **赢家仍不可复现。** add_broadcast 这次判 `block=512`（5.141 第 1 次是 128），
   relu 判 256（5.141 全部保留默认）。跨度门槛只回答"这次采集该不该下结论"，
   不回答"哪个配置真的更快"。

#### 三、边界

- 只在 HIP 采一次，CUDA 侧未采；1.0 秒是为这四个用例选的，不是通用默认。
- 门槛粒度是**整次采集**：任一轮超限 → 所有非默认候选降级，既不区分哪一轮，
  也不剔除该轮样本。
- 没有预热轮：首轮冷启动同时污染样本与跨度，是当前最大的混杂因素。
- `round_seconds` 只记录"该轮实际执行了多少候选的 measure"，跳过不可行候选，
  不含轮间空档；因此含不可行候选的轮跨度会偏短（如实反映"这一轮测了什么"）。

#### 四、下一步

- **加预热轮**（`warmup_rounds`）：正式轮之前先连测至少一轮并丢弃，既去掉首轮
  样本，也让跨度门槛只衡量稳态轮；这是本次实测暴露出的、比调阈值更接近根因的
  改动；
- 或把门槛细化到轮次：只丢弃超限轮的样本，剩余轮次 ≥2 时照常配对（需要把
  "可用轮"从 `samples` 的键里显式筛出来）；
- 两阶段搜索仍有效，且与预热轮不冲突：缩短跨度是治本，预热轮是去掉首轮的固定
  偏置。

#### 五、验证

- 本地 `py -3.14 -m unittest discover -s tests`：**1538 tests OK
  (skipped=311)**（5.141 的 1532 + 4 条跨度门槛用例 + 2 条 `tune_compare`）。
- WSL HIP：`tests.test_tuning` + `tests.test_tune_compare` 共 68 条通过。
- HIP 实测 1 次（报告在临时目录，未入库）：conv2d 44/45、dropout 4/5 判
  `long-round`，add_broadcast 与 relu 不受影响。


### 5.143　预热轮：把首轮冷启动挪出样本与跨度

5.142 的实测把根因从"轮内位置效应"收窄到"首轮冷启动"：四个用例的第一轮分别
是后续轮次中位数的约 14× / 2.1× / 4.0× / 1.3×。这一节不调阈值，而是在正式轮
之前加预热轮：先连测一遍、样本丢弃，让冷启动不进入配对差，也不抬高跨度门槛。

#### 一、做法

- `tune_paired(..., warmup_rounds=0)`：默认 0，保持 5.140～5.142 的口径；
  必须是非负整数，`bool` / 字符串显式拒绝；
- 预热轮**固定正序**、不写 `samples`、不记 `order`、不计 `round_seconds`，
  也**不消耗** `shuffle_rng`——因此同一 `seed` 下预热 0 轮与 1 轮的正式轮排列
  完全相同，A/B 只差"冷启动有没有跑掉"；
- 预热耗时单独记在 `PairedResult.warmup_seconds`，报告与打印都能看到成本；
- 预热轮里抛异常的候选同样判 `rejected`，正式轮不再尝试它（与正式轮的可行性
  口径一致，避免"预热跑挂了还继续测"）；
- `tune.py` / `tune_compare.py` 增加 `--warmup-rounds`（默认 0；负值、
  `--serial` 组合都显式报错）；报告记录 `warmup_rounds` / `warmup_seconds`；
- 新增 5 条测试：预热轮不进入样本与跨度、不改变 shuffle 排列、预热期异常直接
  拒绝候选、非法参数被拒、报告记录预热成本。

#### 二、结果

HIP（RX 7650 GRE / DXG），`--rounds 8 --warmup-rounds 1 --max-round-seconds 1.0`，
用例与 5.142 相同（其余条件一致，可逐项对照）：

| 用例 | 预热耗时 | 正式轮跨度（秒） | 5.142 的首轮 / 后续 | 判定 |
|---|---|---|---|---|
| add_broadcast | 0.667 | 0.052～0.057 | 0.873 / 0.058～0.064 | 全部默认（3 个 `unstable`） |
| relu_1m | 0.856 | 0.335～0.393 | 0.860 / 0.388～0.431 | 全部默认（1 个 `unstable`） |
| dropout_mask_1m | 0.944 | 0.340～0.385 | 1.601 / 0.348～0.447 | 全部默认（3 个 `unstable`；5.142 是 4/5 `long-round`） |
| conv2d_tiled_1m | 2.789 | 2.227～2.531 | 3.334 / 2.211～2.453 | 44/45 `long-round`，保留默认 |

四条结论：

1. **预热轮确实吃掉了冷启动。** 预热耗时与 5.142 的首轮同量级
   （0.67 vs 0.87、0.86 vs 0.86、0.94 vs 1.60、2.79 vs 3.33），正式轮全部落在
   5.142 的"稳态"区间；conv2d 的最慢轮从 3.33 降到 2.53，说明降下来的那一段
   就是首轮。
2. **跨度误伤消失。** dropout 从"4/5 个候选 `long-round`"变成"0 个超限、
   3 个 `unstable`"：门槛不再因为首轮 1.6 秒而作废后 7 轮有效数据。conv2d 仍
   44/45 超限，但这次是它的稳态本身就 2.2～2.5 秒，门槛按设计拦截。
3. **5.142 的两个"收益"消失。** add_broadcast 的 `block=512`（1.03×）与 relu
   的 `block=256`（1.05×）都退回默认。首轮冷启动不只是让区间变宽，还会把配对差
   整体推向某一侧（基线与候选的样本落在冷启动曲线的不同位置）；预热后收益消失，
   更可能是污染而不是真实收益。
4. **赢家仍不可复现，但失败方式变干净了。** 从"一次判收益、一次判噪声"变成
   "全部保留默认 + 1～3 个候选两半不稳"。这是"证据不足"，不是"假阳性"。

#### 三、边界

- 只采一次，CUDA 侧未采；不能证明预热让结论可复现，只能证明它去掉了首轮的系统性
  偏差。
- 预热轮不进入 `order` / `round_seconds`，因此报告里"8 轮"始终指正式轮；成本看
  `warmup_seconds`（本次约等于 1 轮正式轮）。
- 预热轮抛异常的候选直接判 `rejected`：若异常是偶发的（例如驱动瞬时 OOM），
  这个候选会被永久跳过。这是既有"一次异常即不可行"语义的延伸，没有加重试。
- 默认仍为 0。要不要把 1 轮设为默认，需要跨机器、多次采集的证据；本次只证明
  它有效，没证明它总是必要。

#### 四、下一步

- 预热后 conv2d 仍 44/45 超限、逐元素仍全部保留默认（1～3 个 `unstable`），
  说明 45 个候选的空间本身太长。两阶段搜索（先扫 `tile_w`，再扫 `roc × cb`）
  仍是治本，且与预热轮不冲突（5.144 已实现）；
- CUDA 侧各采一次 `--warmup-rounds 1`，确认冷启动量级与 HIP 一致（M40 的首次
  加载路径不同，不能直接沿用 1.0 秒阈值）；
- 预热轮成本是约 1 轮正式轮（conv2d 约 2.8 秒），若后续缩短了单轮跨度，成本会
  同步下降。

#### 五、验证

- 本地 `py -3.14 -m unittest discover -s tests`：**1543 tests OK
  (skipped=311)**（5.142 的 1538 + 5 条预热轮用例）。
- WSL HIP：`tests.test_tuning` + `tests.test_tune_compare` 共 73 条通过。
- CUDA 真机（Tesla M40）：1537 tests OK (skipped=77)，与本地差 6 条（未装
  cuDNN）。
- HIP 实测 1 次（报告在临时目录，未入库）：预热 1 轮后 dropout 不再被跨度门槛
  误伤，conv2d 仍按稳态跨度超限。


### 5.144　两阶段搜索：把 45 个候选拆成 4 + 11 + 2

5.143 的下一步是缩短跨度。跨度门槛其实已经说明"这一轮太长"和"这个空间太大"
是同一件事：conv2d 45 个候选一轮 2.2～2.5 秒，预热也只能把它从"首轮 3.3 秒"
降到"每轮 2.2 秒"，照样超限。这一节不调测量参数，而是换搜索结构。

#### 一、做法

`tune_staged(space, measure, ...)` 做坐标搜索，每一段都是完整的
``tune_paired``（预热、跨度门槛、两半检查全部继承）：

1. 阶段 1 只扫 ``first``（默认取候选值最少的键，并列取声明顺序第一个），其余
   参数固定为默认值。conv2d 这次自动选到 ``tile_w``，4 个候选；
2. 阶段 2 把 ``first`` 固定为阶段 1 的赢家，再扫剩下的键（``roc × cb``，11 个
   候选）；它的基线是"阶段 1 赢家 + 其余默认"，只回答这一段坐标；
3. 若组合配置与全局默认不同，再做一次**两候选**对照（默认 vs 组合），返回的
   就是这次对照——``speedup`` / ``decided`` 的口径与 ``tune_paired`` 完全一致，
   配置库不必区分搜索方式。

省掉多余步骤的两个条件：``first`` 覆盖全部参数（单键空间直接返回阶段 1）；
组合配置等于默认（直接返回阶段 2，不再跑最终对照）。每段的摘要记在
``PairedResult.stages``（``keys`` / ``winner`` / ``decided`` / ``round_seconds`` /
``long_round`` / ``unstable``），报告能回答"哪一段贡献了收益、哪一段被拦下"。

CLI 增加 ``--staged``（``--serial`` 下显式报错）；``tune_compare`` 报告顶层记录
``staged``，每个用例记录 ``stages``。新增 5 条测试：两轴按序搜索、单键空间退化
为配对、``first`` 自动选最少档、非法 ``first`` 被拒、报告可 JSON 落盘。

#### 二、结果

HIP（RX 7650 GRE / DXG），`--rounds 8 --warmup-rounds 1 --max-round-seconds 1.0
--staged`，与 5.143 同用例：

| 用例 | 阶段划分 | 各段最长轮 | `long-round` | `unstable` | 最终 |
|---|---|---|---|---|---|
| add_broadcast | `block`（单段） | 0.070s | 0 | 1 | `block=256` 1.08×，已判定 |
| relu_1m | `block` | 0.394s | 0 | 3 | 保留默认 |
| dropout_mask_1m | `block` | 0.434s | 0 | 1 | 保留默认 |
| conv2d_tiled_1m | `tile_w` → `roc×cb` → 最终 | 0.210 / 0.646 / 0.646s | 0 / 0 / 0 | 2 / 10 / — | 保留默认 |

三条结论：

1. **跨度问题解决了。** conv2d 的单轮跨度从 5.143 的 2.2～2.5 秒降到阶段 1 的
   0.21 秒、阶段 2 与最终对照的 0.65 秒；1.0 秒门槛全程不再触发（5.143 是
   44/45 个候选 `long-round`）。
2. **赢家仍不可复现。** conv2d 阶段 2 的 11 个候选里有 10 个两半方向相反，最终
   保留默认；逐元素里只有 add_broadcast 判 1.08×，relu 与 dropout 保留默认，分布
   与 5.143 又不同。说明剩下的瓶颈不是轮内跨度，而是整机漂移下的 8 轮样本量。
3. **成本明显下降。** conv2d 从 45 候选 × 8 轮（约 21 秒）降到
   (4 + 11 + 2) 候选 × 8 轮 + 3 轮预热（约 12～13 秒），且每一轮都短到可以
   直接对照。

#### 三、边界

- 坐标搜索会漏掉"两个键必须同时改才更快"的交互；现在接受这个漏检，因为要解决
  的是"测不准"，不是"解析最优"。
- 阶段 2 的 ``speedup`` 只相对"阶段 1 赢家 + 其余默认"，不能单独当收益读；相对
  全局默认的收益只在返回的最终对照里。
- 最终对照只有 2 个候选、8 轮，同样可能两半不稳（本次没有触发，但不保证）。
- ``first=None`` 的"最少档"只是启发式；并列时取声明顺序，换成另一台后端可能选到
  不同键，报告里必须看 ``stages[0]["keys"]`` 而不是假设它一定是 ``tile_w``。
- 只采一次，CUDA 侧未采。

#### 四、下一步

- conv2d 阶段 2 的 10/11 个候选两半不稳，说明 8 轮样本仍不足以压住整机漂移；
  5.145 已试 16 轮：不稳数减半（10/11 → 5/11），但赢家仍未出现，且 8 轮的唯一
  收益（add_broadcast 1.08×）消失（该候选两半差 +13.18% / -12.07%）。因此不
  采用"加轮数"，下一步是跨采集复现或控制环境；
- CUDA 侧采一次 ``--staged``，确认拆段后单轮跨度与 M40 上的冷启动量级；
- 两阶段之后若仍有"必须同时改两个键"的用例被漏掉，再考虑在阶段 2 加一轮
  "邻近组合"的局部搜索，而不是回到全因子。

#### 五、验证

- 本地 `py -3.14 -m unittest discover -s tests`：**1549 tests OK
  (skipped=311)**（5.143 的 1543 + 5 条两阶段用例 + 1 条报告透传）。
- WSL HIP：`tests.test_tuning` + `tests.test_tune_compare` 共 79 条通过。
- CUDA 真机（Tesla M40）：1543 tests OK (skipped=77)。
- HIP 实测 1 次（报告在临时目录，未入库）：conv2d 拆成 4 + 11 + 2 后
  `long-round` 归零，但阶段 2 的 10/11 个候选两半不稳，最终仍保留默认。


### 5.145　把轮数从 8 加到 16：更少的假阳性，仍然没有真结论

5.144 的下一步留了"延长轮数"这一条，用来验证"8 轮样本不足"是不是唯一瓶颈。
这一节只改一个参数（`--rounds 16`），其余命令、用例、预热与跨度门槛与 5.144
完全相同。

#### 一、结果

HIP（RX 7650 GRE / DXG），`--rounds 16 --warmup-rounds 1 --max-round-seconds 1.0
--staged`：

| 用例 | 8 轮不稳数 | 16 轮不稳数 | 8 轮赢家 | 16 轮赢家 |
|---|---|---|---|---|
| add_broadcast | 1 / 4 | 4 / 4 | `block=256` 1.08× | 默认 |
| relu_1m | 3 / 4 | 2 / 4 | 默认 | 默认 |
| dropout_mask_1m | 1 / 4 | 3 / 4 | 默认 | 默认 |
| conv2d 阶段 1（`tile_w`） | 2 / 4 | 1 / 4 | 默认 | 默认 |
| conv2d 阶段 2（`roc×cb`） | 10 / 11 | 5 / 11 | 默认 | 默认 |

三条结论：

1. **轮数确实影响"两半异号"的比例。** conv2d 阶段 2 从 10/11 降到 5/11，
   幅度接近一半；这是"8 轮样本不足"这一半判断的支持证据。
2. **但赢家一个都没留下。** 8 轮里唯一的收益（add_broadcast 的 `block=256`
   1.08×）在 16 轮下消失；它的配对差前半段是 **+13.18%**、后半段是
   **-12.07%**——同一配置在采集期间经历了约 25% 的整机漂移。在这种数据上，
   任何统计量都无法恢复出 5% 的真实差异，加轮数只是让这段漂移被更多样本
   "平均"掉，不会把它变成配置差异。
3. **"不稳数"本身不可当质量指标。** add_broadcast 从 1 升到 4，dropout 从 1
   升到 3，而 conv2d 是降的；两次采集的整机状态不同，不能把变化严格归因于
   轮数。能确定的只有：16 轮没有产生任何可接受的新赢家。

#### 二、决策

- **不把默认轮数改成 16。** 成本翻倍（conv2d 一次采集约 21 → 42 秒），换来的
  是"更少的假阳性"而不是"更多的真结论"；写库决策仍然按 8 轮 + 两半检查 +
  跨度门槛执行。
- 8 轮的假阳性（add_broadcast 1.08×）说明"单次采集的结论"始终不可全信，
  但这已经由 `decided` 与报告的星号标注暴露给读者，不需要再加轮数来兜底。

#### 三、下一步（都比加轮数更接近根因）

- **跨采集复现**：同一命令跑两次，只接受"两次都判同一配置、同一方向"的赢家。
  这与 5.63 最初的诉求一致，但成本是两倍采集，且需要新的结果合并接口
  （5.146 已实现，见下一节）；
- **控制环境**：锁频 / 固定风扇 / 避开 WSL2 + DXG 的整机状态抬升路径。这是
  环境问题，不是测量问题——25% 的漂移一旦出现，任何 8/16 轮的自助法都救不回来；
- 若两条都不做，就把当前口径的边界写清楚：**报告里的收益是"本次采集的方向"，
  不是"跨采集的保证"**。

#### 四、验证

- 本次没有代码改动，测试基线不变：本地 1549 tests OK (skipped=311)；
  WSL HIP 79 条；CUDA 真机 1543 tests OK (skipped=77)。
- HIP 实测 1 次（报告在临时目录，未入库）：16 轮下四个用例全部保留默认，
  add_broadcast 的 8 轮收益消失、两半差达 +13.18% / -12.07%。


### 5.146　跨采集复现：两次都同向才接受

5.145 的下一步留了两条路：跨采集复现与控制环境。这一节做前者——**不改测量
参数**，只改"什么算结论"：同一个候选必须在多次独立采集里都判 `faster`，才允许
替换默认。

#### 一、做法

新增 `tune_replicated(space, measure, *, repeats=2, ...)`。它按顺序把
`tune_paired` 跑 `repeats` 次（第 `i` 次用 `seed + i`），再逐候选合并：

1. 某候选在任一次采集里不可行（测量抛错或结果非有限），合并判 `unreplicated`；
2. 每次的 `verdict` 完全相同才保留该判定，否则判 `unreplicated`；
3. 只有合并后仍是 `faster` 的候选参与选优，取 `diff_median_pct` 最小者；
   没有这样的候选就保留默认。

`unreplicated` 与 `unstable` / `long-round` 是三种不同的拒绝：`unstable` 是
一次采集内前后两半异号，`long-round` 是一轮跨过太多整机状态，`unreplicated`
是"换个时间结论就不在"。三者都不算 `stable`，但报告要能区分，否则无法判断
该缩短轮次、缩短跨度，还是换环境重测。

每次的摘要记在 `PairedResult.replicates`（`best` / `decided` / `speedup` /
`unstable` / `long_round` / `round_seconds`）；返回结果的 `round_seconds` 与
`order` 来自最后一次采集，不合并。CLI 增加 `--repeats N`，与 `--staged` 互斥
——两阶段的候选集可能每次不同，逐候选下标对齐会错位。`--serial` 下同样拒绝。

#### 二、结果

HIP（RX 7650 GRE / DXG），`--rounds 8 --warmup-rounds 1 --max-round-seconds 1.0
--repeats 2`，与 5.143～5.145 同用例：

| 用例 | 第 1 次采集 | 第 2 次采集 | 合并后 |
|---|---|---|---|
| add_broadcast | `block=512` 1.0756×，已判定 | `block=1024` 1.1316×，已判定 | 3 个 `unreplicated`，保留默认 |
| relu_1m | 默认 1.00×，未判定 | `block=128` 1.0346×，已判定 | 1 个 `unreplicated` + 1 个 `unstable`，保留默认 |
| dropout_mask_1m | 默认 1.00× | 默认 1.00× | 4 个 `unreplicated`，保留默认 |
| conv2d_tiled_1m | 默认；44 个 `long-round` | 默认；44 个 `long-round` | 保留默认（44 个 `long-round`） |

合并后的逐候选判定也说明"显著"有多脆：

- add_broadcast 的 `block=512` 与 `1024` 各自在**某一次**里都过了 1% 地板与置信
  区间（`-8.23%` 与 `-13.25%`），但两次的赢家不是同一个，合并后 128 / 512 / 1024
  全部 `unreplicated`，只有 256 是一次内的两半异号（`unstable`）；
- relu 的 `block=128` 两次配对差是 `-2.32%`（区间 `[-3.07, -0.77]`，单次可判
  收益），但另一次是默认，合并后 `unreplicated`；
- dropout 两次都保留默认，四个 `block` 值全部 `unreplicated`——两次里它们的方向
  本来就不一致；
- conv2d 两次都是 44 个 `long-round`（最长轮 2.68 秒，超过 1.0 秒门槛），合并后
  保持 `long-round` 而不是 `unreplicated`：这是**环境一致地拒绝**，不是结论摇摆。

三条结论：

1. **"两次都显著"不等于"复现"。** add_broadcast 两次分别给出 1.0756× 与
   1.1316×，方向一致、量级相近，但赢家是不同配置。若只看任一次，都会写错一行
   配置库；跨采集复现按"逐候选判定完全相同"合并，直接把它拦下。
2. **单次采集的收益必须当成方向而不是结论。** 本轮的 4 个用例在两次采集里共
   出现 3 次 `decided=True`，合并后全部未留下；这与 5.145"16 轮没有新赢家"的
   结论同向——问题不在样本量，而在整机状态的可比性。
3. **区分"环境拒绝"与"结论不一致"有用。** conv2d 的 44 个 `long-round` 两次
   一致，说明跨度门槛稳定地把这个空间判成"测不了"；若合并逻辑把它们也标
   `unreplicated`，读者会误以为换个时间就能测。

#### 三、边界

- 两次采集仍来自同一台机器的连续运行，`seed + i` 只改 bootstrap 与 shuffle 的
  随机流，不改整机状态；这不是严格的独立实验设计，检验的是"换一段时间后结论
  还在不在"。
- 全票通过（`repeats` 次全部同向）在 `repeats` 增大时越来越苛刻；三次以上应该
  考虑多数票而不是全票，本轮未实现。
- 合并后的 `diff_median_pct` / 区间来自最后一次采集，不是各次的平均；要读某次
  的数值必须看 `replicates`。
- 成本是采集时间乘以 `repeats`，且被拦下后不会给出替代结论——它只降低写库的
  假阳性，不提高发现真赢家的能力。
- 与 `--staged` 互斥；CUDA 侧的同一实验见 5.147。

#### 四、下一步

- **控制环境**仍是根因方向：25% 的整机漂移出现时，任何跨采集合并都只能确认
  "结论不稳定"，不能把它变成稳定结论。锁频 / 固定风扇 / 绕开 WSL2 + DXG 的
  抬升路径需要单独一轮；
- 若要把复现纳入写库决策，需要先决定"未复现"是否留档：现在的行为是保留默认，
  报告里留 `replicates` 供复查；
- CUDA 侧采一次 `--repeats 2`，确认 M40 上两次采集的赢家分歧是否比 HIP 小
  （5.147 已采：9 个用例全部保留默认，两次各一次 `decided=True` 均被判
  `unreplicated`；同一套 1.0 秒跨度门槛在 M40 上不触发）。

#### 五、验证

- 本地 `py -3.14 -m unittest discover -s tests`：**1555 tests OK
  (skipped=311)**（5.144 的 1549 + 5 条跨采集复现用例 + 1 条报告透传）。
- WSL HIP：`tests.test_tuning` + `tests.test_tune_compare` 共 85 条通过。
- CUDA 真机（Tesla M40）：1549 tests OK (skipped=77)。
- HIP 实测 1 次 `--repeats 2`（报告在临时目录，未入库）：两次采集各判出不同
  赢家（add_broadcast 的 `512` 与 `1024`、relu 的默认与 `128`），合并后全部
  保留默认；conv2d 两次都是 44 个 `long-round`。


### 5.147　CUDA 侧的跨采集复现：同一套门槛、不同的拒绝理由

5.146 的边界写着"CUDA 侧未采"，下一步第三条是"确认 M40 上两次采集的赢家分歧
是否比 HIP 小"。这一节只做采集，不改代码。

#### 一、做法

CUDA 真机（Tesla M40 24GB，sm_52，Python 3.10）：

```bash
python3 -m benchmarks.tune_compare --backend cuda --rounds 8 --warmup-rounds 1 \
  --max-round-seconds 1.0 --repeats 2 --save repeats2_cuda.json
```

与 5.146 的 HIP 命令同参数，但用例是 CUDA 侧的完整 9 个（HIP 那次只跑 4 个）。

#### 二、结果

| 用例 | 第 1 次采集 | 第 2 次采集 | 合并后 | 最长轮 |
|---|---|---|---|---|
| add_same_1m | 默认 | 默认 | 1 个 `unreplicated` + 2 个 `unstable` | 0.175s |
| add_broadcast | 默认 | 默认 | 2 个 `unreplicated` | 0.018s |
| relu_1m | 默认 | 默认 | 2 个 `unreplicated` + 2 个 `unstable` | 0.123s |
| gelu_1m | 默认 | 默认 | 2 个 `unreplicated` + 2 个 `unstable` | 0.136s |
| sigmoid_1m | `block=512` 1.0144×，已判定 | 默认 | 4 个 `unreplicated` | 0.140s |
| silu_gate | 默认 | `block=128` 1.0139×，已判定 | 4 个 `unreplicated` | 0.054s |
| step_1m | 默认 | 默认 | 2 个 `unreplicated` + 1 个 `unstable` | 0.131s |
| dropout_mask_1m | 默认 | 默认 | 3 个 `unreplicated` + 1 个 `unstable` | 0.895s |
| conv2d_tiled_1m | 默认 | 默认 | 44 个 `unreplicated` | 0.874s |

三条结论：

1. **跨采集复现在 CUDA 上同样把单次收益全部拦下。** 两次采集各出现一次
   `decided=True`（sigmoid 的 `block=512` 1.0144×、silu_gate 的 `block=128`
   1.0139×），合并后都判 `unreplicated`、保留默认。加起来的 4 次"单次显著"
   （HIP 3 次 + M40 2 次，去重后 5 个用例）没有一次通过复现；收益量级本身只有
   1.4%，与 1% 噪声地板只差 0.4 个百分点。
2. **同一套绝对秒数门槛跨机不可移植。** `--max-round-seconds 1.0` 在 M40 上
   全程不触发（conv2d 最长轮 0.874s），而在 HIP + WSL2/DXG 上 44/45 个候选
   触发（2.68s）。门槛拦住的是"整机状态漂移"，但"多长算长"取决于机器与运行时
   栈，同一个 1.0 秒在两台机器上的含义完全不同。
3. **同样是 conv2d，两台机器的拒绝理由相反。** M40 的 44 个候选判
   `unreplicated`：两次都测完了，但方向不一致；HIP 的 44 个判 `long-round`：
   跨度本身就把结论作废。两种判定都正确，但读法不同——`long-round` 说明"这台
   机器上这个空间测不了"，`unreplicated` 说明"测得出来但换一次就不成立"。

#### 三、边界

- 只采 2 次，且仍是同一台机器的连续运行；`seed + i` 只改随机流，不改整机状态。
- 9 个用例里有 7 个两次都保留默认，没有任何"稳定收益"通过复现——因此这组数据
  只能证明"未复现的都被拦下"，不能证明"复现通过时一定可靠"。
- 报告里 `round_seconds` / `order` 只来自最后一次采集；逐次摘要见 `replicates`。
- CUDA 的 9 个用例与 HIP 的 4 个不是同一批，跨机器的"赢家分歧大小"不能直接比
  次数，只能比"两次是否同向"这个二值结论。

#### 四、下一步

- 把 `max_round_seconds` 从绝对秒数改成相对量（例如"默认候选单轮耗时的倍数"），
  或至少在报告里同时给出 `round_seconds` 与候选数，让门槛在不同机器上可比
  （5.148 已实现相对量：`--max-round-factor`，以预热轮平均耗时为标尺）；
- 控制环境（锁频 / 固定风扇 / 绕开 WSL2 + DXG）仍未做，它决定 25% 的漂移是否
  能被消除；
- 三次以上采集的多数票，而不是当前的全票。

#### 五、验证

- 本轮没有代码改动，测试基线不变：本地 1555 tests OK (skipped=311)；
  WSL HIP 85 条；CUDA 真机 1549 tests OK (skipped=77)。
- CUDA 实测 1 次 `--repeats 2`（报告在临时目录，未入库）：9 个用例全部保留
  默认，两次各一次 `decided=True` 均被判 `unreplicated`；conv2d 最长轮 0.874s
  未触发跨度门槛，44 个候选全部 `unreplicated`。


### 5.148　相对跨度门槛：以预热轮为标尺

5.147 的第一条结论是"绝对秒数门槛跨机不可移植"：同一个 1.0 秒在 M40 上从不
触发，在 HIP + WSL2/DXG 上拦掉 44/45 个候选。这一节把门槛改成相对量。

#### 一、做法

新增 `max_round_factor`：上限 = 该倍数 × **预热轮平均耗时**。预热轮本来就是
一次完整的全候选连测，它的耗时天然包含机器、候选数与算子成本三个因素，不需要
额外测量。约束与语义：

- 必须 `warmup_rounds >= 1`，否则显式报错——没有预热就没有标尺，不退回绝对
  秒数，也不静默跳过；
- 与 `max_round_seconds` 互斥：两把尺子同给时无法解释是哪一把拦下了结论；
- 实际算出的秒数记在 `PairedResult.round_budget_seconds`，随 `max_round_factor`
  一起写进报告。判定仍是"任一轮超过上限 → 所有非默认候选 `long-round`、保留
  默认"，与 5.142 相同。

CLI 增加 `--max-round-factor F`（`tune.py` 与 `tune_compare.py` 各一份），并在
`--serial` 下拒绝。打印 tail 从"轮内最长 x.xxs"扩成"轮内最长 x.xxs，上限
x.xxs（F× 预热）"，让读者能直接判断这一轮是异常还是正常。

#### 二、结果

HIP（RX 7650 GRE / DXG），`--rounds 8 --warmup-rounds 1 --max-round-factor 1.5`：

| 用例 | 绝对门槛 1.0s（5.146） | 相对门槛 1.5×（单次） | 相对门槛 + `--repeats 2` |
|---|---|---|---|
| add_same_1m | 未采 | `block=256`，-5.50%，已判定 | 4 个 `unreplicated`，保留默认 |
| add_broadcast | 3 个 `unreplicated` | `block=1024`，-4.13%，已判定 | 2 个 `unreplicated` + 1 个 `unstable`，保留默认 |
| relu_1m | 1 个 `unreplicated` + 1 个 `unstable` | 默认 | 4 个 `unreplicated` |
| dropout_mask_1m | 4 个 `unreplicated` | 默认 | 4 个 `unreplicated` |
| conv2d_tiled_1m | 44 个 `long-round` | `tile_w=32`，-6.66%，已判定；16 个 `unstable` | 65 个 `unreplicated` + 7 个 `unstable`，保留默认 |

三条结论：

1. **相对门槛解决了"误伤"。** 绝对 1.0 秒把 conv2d 的 44 个候选全部判成
   `long-round`，而这台机器上 conv2d 一轮本来就要 2.7 秒；相对门槛取
   `1.5 × 2.7 = 4.0` 秒，正常轮不再触发，`long-round` 归零。门槛的语义从
   "绝对多长"变成"比这台机器的常态长多少"。
2. **放行不等于可信。** 相对门槛下单次采集判出 3 个收益（add_same_1m 5.50%、
   add_broadcast 4.13%、conv2d 6.66%），加上 `--repeats 2` 后全部
   `unreplicated`，一个都没留下。绝对门槛是"一刀切地拒绝"，相对门槛是
   "放行后交给复现筛"——两者都需要，但不能只用前者就宣称"这些配置没收益"。
3. **conv2d 的 `unstable` 数量随门槛放宽而显形。** 单次采集 16 个、两次采集
   7 个候选前后两半异号，而绝对门槛下这些信息被 `long-round` 覆盖了。这说明
   `long-round` 是一种"不解释"的拒绝，只在确实测不了时使用；一旦跨度正常，
   应该让 `unstable` / `noise` 把问题暴露出来。

#### 三、边界

- 标尺是预热轮的**平均**耗时，预热轮本身也可能被冷启动或一次抖动污染；所以
  `warmup_rounds` 仍应至少 1，且报告里保留 `warmup_seconds` 供复查。
- 倍数越小越严格：`1.5` 只是本轮的选择，没有理论最优；它只回答"这一轮是否
  明显比常态长"，不回答"多长算合理"。
- 相对门槛不与 `--staged` 冲突（两阶段每段各自有预热），但与 `--repeats`
  叠加时每次采集的预热基准不同——这正是跨采集复现要检验的。
- M40 上未采相对门槛；绝对门槛在那里不触发，两者预期差异很小。
- 本轮只采 2 次，仍是同一台机器的连续运行。

#### 四、下一步

- 把 `round_budget_seconds` 与 `round_seconds` 一起画进报告，让"门槛是否过紧"
  有直观依据；
- 控制环境（锁频 / 固定风扇 / 绕开 WSL2 + DXG）仍决定 25% 的漂移能否消除；
- 三次以上采集改多数票。

#### 五、验证

- 本地：1561 tests OK (skipped=311)（5.147 的 1555 + 5 条相对门槛用例 +
  1 条报告透传）。
- WSL HIP：`tests.test_tuning` + `tests.test_tune_compare` 共 91 条通过。
- CUDA 真机（Tesla M40）：1555 tests OK (skipped=77)。
- HIP 实测 2 次（报告在临时目录，未入库）：单次 `--max-round-factor 1.5`
  conv2d 不再 `long-round` 并判出 `tile_w=32`；叠加 `--repeats 2` 后 9 个用例
  全部保留默认。


### 5.149　多数票：允许少数次不同

#### 一、做法

5.148 的下一步写着"三次以上采集改多数票"。全票口径的问题不在统计，而在组合
数：`repeats` 越大，"每个候选每次都同向"的概率越低；只要有一两次整机状态换
向，整个候选就被作废，哪怕真实差异稳定存在。5.148 的 4 个单次收益在
`--repeats 2` 下全军覆没，正是这种过严的表现。

改了三处：

1. `tune_replicated` 增加 `vote` 参数：`all`（默认）与 `majority`。
   `majority` 要求 `repeats >= 3`——两次采集下"过半"就是 2/0 或 1/1，与全票
   没有区别，放开只会让参数看起来换了判据。
2. 合并规则分三层。硬约束对两种模式都成立：任一次不可行记 `unreplicated`
   （可行性不是投票能抵消的）；`faster` 与 `slower` 同时出现也记
   `unreplicated`（方向都定不下来，多数没有意义）。第二层是过半：某一判定
   达到 `len // 2 + 1` 票且没有反向票即采纳。第三层是众数：没有任何一方过半
   时取票数最多的判定；平票按保守序取（`faster` / `slower` <
   `noise` / `baseline` < `unstable` / `long-round` < `unreplicated`），
   即平票时选"不换配置"的那一侧。
3. 每个候选记录票数（`votes`，如 `{"faster": 2, "noise": 1}`），结果对象与
   报告都带 `vote`；单次采集的 `vote` 是 `None`，因为那时没有票可数。
   CLI `--vote` 与 `--serial` / `--staged` / `--repeats < 3` 显式冲突，API
   层同样抛 `ValueError`。

默认仍是 `all`：5.146～5.148 的全部采集都按全票口径做，换默认前应先有 A/B
证据；多数票是显式 opt-in。

#### 二、结果

HIP（RX 7650 GRE，`rounds 8`、`warmup-rounds 1`、`max-round-factor 1.5`）
三次采集，两种合并方式跑同一批 9 个用例：

- **全票**：9 个用例全部保留默认，报告的 `decided` 列表为空。conv2d 的 45 个
  候选里 42 个 `unreplicated`、2 个 `noise`、1 个 baseline，没有一个
  `faster`。
- **多数票**：8 个逐元素用例仍全部保留默认；conv2d 的
  `cb=2 / tile_w=64` 以 2/3 票（2 `faster` + 1 `noise`，无 `slower`）转正，
  配对差 -11.75%（CI -14.20～-2.33），相对默认 1.037×。

票数分布比结论更有信息量。多数票下逐元素候选的票型是
`unstable×2 + noise×1`、`unstable×2 + faster×1`、
`faster×1 + noise×1 + unstable×1` 之类，**没有候选拿到两张 `faster`**。也就是
说，逐元素保留默认不是"票数门槛太严"，而是三次采集里连两次同向都很少发生：
整机漂移把 `unstable` 写进了大半个候选集。conv2d 是唯一出现 2/3 同向的用例，
且该候选三次里没有一次判 `slower`。

#### 三、边界

- 多数票放宽的是"次数"，不是"幅度"：每次采集仍要过 bootstrap 区间与噪声
  地板。2/3 票的候选在报告里 CI 仍然很宽（-14.20～-2.33，跨度 11.9 个百分
  点），是弱证据；替换默认前应看区间与两半差，而不是只看 `verdict`。
- 3 次里 2 次同向在纯噪声下的概率不低（二项 3/8），但"每次都要显著"把这个
  概率压低了；本轮 45 个候选里只出现 1 个。单个用例、单台机器的一次采集不足
  以证明多数票不会扩大假阳性。
- `unstable` 与 `long-round` 也参与投票：3 次里 2 次 `unstable` 就判
  `unstable`。这比全票宽松，但仍会拒绝；把它们当普通判定计数是一种工程简
  化——它们本来不是"更快/更慢"的票。
- 平票规则（保守优先）是工程选择，没有统计含义；`votes` 只记判定词，不记
  逐次配对差，复查时仍需打开 `replicates`。
- 本轮仍是同一台机器的连续采集。**CUDA 真机环境退化**：该机 cupy 已不可用
  （全量 1571 tests 里 skipped 从 5.147 的 77 变成 311），多数票在 M40 上未
  采集。

#### 四、下一步

- 恢复 CUDA 真机的 cupy 环境，补 `--repeats 3 --vote majority` 采集，验证
  2/3 票型是否只在 HIP 的 DXG 路径下出现；
- 把 `repeats` 加到 5 或 7，看 conv2d 的 2/3 是变成 3/5、4/5（真差异）还是
  停在 2/3 附近（随机同向）；
- 给 `votes` 附上逐次配对差，排查 2/3 里那张 `noise` 是幅度不足还是方向
  反转。

#### 五、验证

- 本地：1571 tests OK (skipped=311)（5.148 的 1561 + 7 条合并规则用例 +
  3 条报告/参数校验用例）。
- WSL HIP：`tests.test_tuning` + `tests.test_tune_compare` 共 101 条通过
  （5.148 的 91 + 10）。
- CUDA 真机（Tesla M40）：1571 tests，1 error、skipped=311。错误在
  `tests/test_frontend.py:476`——远程 torch 较旧，`scaled_dot_product_attention`
  不认识 `enable_gqa`，与本轮改动无关；跳过数 311 说明该机 cupy 已不可用
  （5.147/5.148 为 77），本轮 CUDA 侧没有复跑 GPU 内核与调优采集。
- HIP 实测两份报告（临时目录，未入库）：`vote3_all.json` 9 个用例全部保留
  默认；`vote3_majority.json` 仅 conv2d 判出 `cb=2 / tile_w=64`。

### 5.150　逐次配对差与 5 次多数票

#### 一、做法

5.149 的下一步留了两个问题：`repeats` 加到 5 时 conv2d 的 2/3 会变成 3/5、
4/5（真差异），还是停在 2/3 附近（随机同向）；以及那张 `noise` 到底是幅度
不足还是方向反转。票数只回答"判了几次更快"，回答不了这两个问题。于是先把
逐次配对差落到结果里，再用它跑 5 次：

1. `PairedCandidate` 增加 `replicate_diffs: Tuple[Optional[float], ...]`：
   每次采集的配对差中位数（百分比），顺序与 `replicates` 对齐；那一次不可行
   （或有候选没有可配对轮次）记 `None`，不补 0、不丢项。
2. `tune_replicated` 合并时从各次采集的 `diff_median_pct` 填入；
   `to_dict` 只在非空时输出该键，因此单次采集的报告形状不变。
3. `tune_compare` 的 `pairs` 透传 `replicate_diffs`；`tune.py` 的候选行在
   `票` 之后追加 `逐次`（如 `逐次 -2.22% / -2.99% / -1.90% / +3.30% / +3.28%`，
   不可行显示 `—`），复查时不必打开 JSON。
4. 选**中位差**而不是均值差：与赢家排序口径一致（5.140 起因中位差选赢家），
   且单轮异常不会把整列数字拖走。

#### 二、结果

HIP（RX 7650 GRE，`rounds 8`、`warmup-rounds 1`、`max-round-factor 1.5`、
`--repeats 5 --vote majority`）一次采集，9 个用例全部保留默认；45 次
"用例 × 采集"里有 8 次单次判出赢家，但**同一用例的赢家从不重复**：

| 用例 | 判出赢家的采集 | 单次赢家与提升 |
| --- | --- | --- |
| add_broadcast | 第 2 次 | `block=128`，1.046× |
| relu_1m | 第 3 次 | `block=256`，1.002× |
| sigmoid_1m | 第 4 次 | `block=1024`，1.048× |
| silu_gate | 第 1 次 | `block=512`，1.042× |
| dropout_mask_1m | 第 4 次 | `block=512`，1.024× |
| conv2d_tiled_1m | 第 2 / 4 / 5 次 | `64/4/1` 1.076×、`64/2/2` 1.098×、`16/auto/1` 1.130× |

合并后 conv2d 的 45 个可行候选里 27 个 `unstable`、13 个 `noise`、4 个
`unreplicated`、1 个 baseline（`unreplicated` 的 4 个是"某次采集不可行、某次
可行"）；另有 35 个候选在最后一次采集里不可行，候选总数 80（即
`tile_w × roc × cb` 的完整笛卡尔积）。逐元素的 8 个用例各 5 个可行候选，
`unstable` 占 3～4 个。

**5.149 那个 2/3 转正候选（`cb=2 / tile_w=64 / roc=auto`）在 5 次里拿到 0 张
`faster`**：票型是 `unstable×4 + slower×1`，逐次差
`+2.95 / +0.27 / +5.34 / -2.20 / -8.05`，中位 `+0.27%`——5.149 的 `-11.75%`
没有复现。它既不是"稳定更快、只是幅度小"，也不是"稳定更慢"：方向本身在采集
之间翻转。

全批 85 个可行候选、425 次判定里只有 14 张 `faster` 票，**没有任何候选拿到
两张以上**。含 `faster` 票的候选逐次差都正负同现（relu 的 `block=256`：
`+4.31 / -2.06 / -3.68 / -3.59 / +5.63`；silu_gate 的 `256`：
`-2.63 / +3.70 / +0.76 / +1.18 / -0.71`）。

**CUDA 侧复跑同一实验**（裸机 Ubuntu 22.04、内核 `6.8.0-138-generic`、
3 × Tesla M40 24GB、驱动 535.309.01 / CUDA 12.2、`cupy-cuda12x 14.2.0`，
固定用 2 号卡；0/1 号卡被 `llama-server` 各占约 20GB）：9 个用例同样全部保留
默认；45 次"用例 × 采集"里 4 次判出赢家（add_broadcast `512` 1.037×、step
`512` 1.015×、dropout `128` 1.007×、conv2d `16/4/auto` 1.042×），合并后
同样互相抵消。425 次判定只有 4 张 `faster` 票，比 HIP 的 14 张还少。

M40 与 HIP 最大的差别是 conv2d 出现了**真实的更慢**：45 个可行候选里 21 个
五次全票 `slower`（`tile_w=64 / roc=2 / cb=2` 的逐次差是
`+15.43 / +14.61 / +19.15 / +13.22 / +16.40`，5/5 票），另有 13 个
`unstable`、10 个 `noise`。5.149 那个转正候选（`tile_w=64 / roc=auto / cb=2`）
在 M40 上逐次差 `+0.49 / +0.04 / +1.71 / +0.42 / +0.92`，5 次全为正、0 张
`faster`。

CUDA 上唯一"方向稳定、只是幅度不够"的是 dropout 的 `block=128`：逐次差
`-0.55 / -1.36 / -1.29 / -1.11 / -0.36`（5 次同号），均值区间
`[-0.59, -0.34]` 整体在 0 以下，但只有 3 次的中位差越过 1% 地板，合并成
`faster×1 + noise×3 + unstable×1` → `noise`。这正是逐次差想区分的第二类：
不是方向翻转，而是"差稳定存在、幅度贴着地板"。

#### 三、结论

- 5.149 的 2/3 转正不是"真差异被全票压住"，而是"每次采集的赢家都在换"的
  一个切片。次数从 3 加到 5，该候选连一次 `faster` 都拿不到。
- 8 次单次判出的赢家两两不同，说明瓶颈仍是整机漂移，而不是投票门槛：
  多数票放宽的是次数，改不了方向本身。
- 逐次差把"幅度不足"与"方向反转"分开了：HIP 上绝大多数候选是方向反转
  （逐次差正负交替），CUDA 上则出现了"5 次同号、幅度贴地板"的 dropout
  `block=128` 和"5 次全票更慢"的 21 个 conv2d 候选。
- 跨两台不同厂商、不同架构的机器（RDNA3 的 RX 7650 GRE / Maxwell 的 M40）
  上，5.149 的转正候选都不成立：HIP 是 0/5 张 `faster` 且方向翻转，M40 是
  5 次全为正、从未更快。M40 还证明了这个空间里确实存在真实更慢的配置
  （21 个 5/5 `slower`），因此"保留默认"不是保守，而是两台机器都支持的结论。

#### 四、边界

- 5 次仍是同一台机器的连续采集（`seed + i`），不是独立实验设计；它检验的是
  "换一段时间后结论还在不在"，不是严格复现性。
- 逐次差是中位差，不提供区间：那一次是否显著仍看对应的 `verdict`。
- 任一次不可行即整候选 `unreplicated`（可行性不能投票），但已可用的那几次
  逐次差仍如实保留，`None` 只出现在失败的那一位。
- 报告格式仍是 `aetherflow-tune-compare/3`：`replicate_diffs` 是附加字段，
  读旧报告不受影响。
- 80 个候选里 35 个在最后一次采集里不可行，`unstable` 数在 5 次间从 7 波动到
  36——整机状态本身在变，这一点没有因为记录逐次差而改善。
- **跨架构不可外推**：M40 是 Maxwell（sm_52），HIP 侧是 RDNA3；两边只跑同一批
  9 个用例，M40 上"21 个候选全票更慢"不能推出 HIP 上也更慢。
- **CUDA 侧运行环境**：远程无 torch，全量测试是 1571 条（本地 1576，差的 5 条
  是 torch 条件定义），与调优无关；远程代码经 SFTP 同步（160 个文件，跳过
  `__pycache__`），与本地一致；采集固定在 2 号卡，0/1 号卡仍被
  `llama-server` 占用约 20GB。

#### 五、下一步

- 把"5 次同号"作为新的筛选器落地：CUDA 的 dropout `128`（5 次同号、幅度
  贴地板）与 HIP 上那些正负交替的候选正好构成两组对照样本；
- 在 M40 上复跑该筛选器，确认它不会把 21 个 5/5 `slower` 的候选误选成赢家，
  也不会把 0/5 `faster` 的候选放进短名单；
- 三张 M40 并存时，观察 `warmup-rounds` 与 `max-round-factor` 在空闲卡上的
  稳定性（本次固定 2 号卡，未测卡间差异）。

#### 六、验证

- 本地：1576 tests OK (skipped=311)（5.149 的 1571 + 4 条逐次差用例 +
  1 条报告透传用例）。
- WSL HIP：`tests.test_tuning` + `tests.test_tune_compare` 共 106 条通过
  （5.149 的 101 + 5）。
- HIP 实测报告（临时目录，未入库）：`vote5_majority.json`，9 个用例全部保留
  默认；`replicate_diffs` 与 `replicates` 长度、顺序一致。
- CUDA 真机：`1571 tests OK (skipped=77)`（远程未装 torch，比本地少 5 条
  torch 条件测试）；`cuda_vote5_majority.json`，9 个用例全部保留默认，
  45 次采集里 4 次判出赢家、425 次判定 4 张 `faster` 票。


### 5.151　隐式 GEMM 卷积：A/B 输给 cuBLAS，删掉自动选择器

#### 一、做法

卷积此前有两条路径：`K <= 32` 走分块直算（5.16，核小、权重复用收益大），
其余走 `im2col + cuBLAS`。第三条路是**隐式 GEMM**：不物化
`(N, C*KH*KW, H*W)` 的中间矩阵，内核沿 K 循环、按 `(kh, kw, ci)` 现算输入
地址再累加。它的卖点是省掉那份中间矩阵，代价是不能借 cuBLAS 的分块 GEMM。

`lower/plan.py` 里曾有一个 `conv_implicit_preferred(k, work)`，想在"K 大且
输出够多"时自动切到隐式。这个启发式此前只有"应该更快"的推理，没有测量。
本轮先在 M40 上做同进程、同输入的三实现 A/B（`conv_impl="im2col" /
"tiled" / "implicit"`，每格 3 次预热 + 10 次取中位，im2col 的结果作为相对
误差的基准），再决定选择器保留与否。

#### 二、结果

Tesla M40 24GB（sm_52，驱动 535.309.01 / CUDA 12.2 / cupy-cuda12x 14.2.0，
固定 2 号卡），全部 3x3 / stride 1 / padding 1，单位 ms：

| 形状 `(N,C,H,W,O,k)` | K | 输出元素 | im2col | tiled | implicit | implicit / im2col |
| --- | --- | --- | --- | --- | --- | --- |
| (1,64,32,32,64,3) | 576 | 65536 | 2.00 | 2.38 | 1.89 | 0.95× |
| (1,128,16,16,128,3) | 1152 | 32768 | 1.99 | 3.17 | 2.36 | 1.18× |
| (2,96,28,28,96,3) | 864 | 150528 | 2.23 | 5.22 | 3.36 | 1.51× |
| (1,32,56,56,64,3) | 288 | 200704 | 2.21 | 2.93 | 2.38 | 1.08× |
| (1,256,14,14,256,3) | 2304 | 50176 | 2.34 | 5.13 | 3.62 | 1.55× |
| (4,3,64,64,32,3) | 27 | 524288 | 2.54 | 2.15 | 2.08 | 0.82× |
| (2,512,7,7,512,3) | 4608 | 50176 | 3.97 | 11.04 | 11.93 | 3.00× |

三者的数值一致：七个形状里 tiled 相对 im2col 的最大偏差 `7.15e-7`、
implicit 的 `1.65e-6`，都在 float 累加的同口径误差内——差别只在求和顺序，
不是语义差异。

延迟上隐式**没有稳定的优势区**：K=27 与 K=576 两格在 ±20% 内（本机漂移
量级，5.63 记过整机 30% 的漂移），K>=288 的六格里另外五格慢
1.08×～3.00×，最慢的 K=4608 慢到 3 倍。分块核在 K=27 仍然最快（2.15 vs 2.54，与 5.16 的
"K <= 32 占优"一致）。因此：

- 删除 `conv_implicit_preferred` 与 `CONV_IMPLICIT_MIN_WORK`，`conv_impl=
  "auto"` 只在 `im2col` 与 `tiled` 之间选；
- `conv_impl="implicit"` 保留为**显式对照路径**（想单独量这一条件时用），
  f16 下显式拒绝——隐式内核只有 f32 实例。

#### 三、边界

- 单机单次采集（每格 3 预热 + 10 次中位），两格 <10% 的差不出结论；1.5× /
  3.0× 这两档超过漂移，方向可信。
- 只测 3x3 / stride 1 / padding 1；其它核形状按比例外推没有依据。
- 本轮只测延迟，不测峰值。隐式"省一份 im2col 中间矩阵"的显存收益未量化；
  但自动选择器要的是"更快"，延迟上的劣势已足以否掉它。
- 删掉的是**判据**，不是实现：显式路径的代码、内核与对拍用例都还在。

#### 四、验证

- 本地 `tests.test_plan`：`TestConvTilePlan` 等 76 条通过；删除
  `test_implicit_only_for_large_k`（它钉的是已删的 `conv_implicit_preferred`）。
- 远程 `TestCudaConvImplicit` 4 条：显式 implicit 与 im2col 对拍、小形状
  可用、f16 显式拒绝。
- 远程 `tests.test_lower_kernel` 逐 token 比对通过（隐式内核本身没改）。

### 5.152　f16 注意力：能力早在 5.121 就有，补的是文档与用例

#### 一、做法

5.121 把融合 attention 的前向内核改成混合精度模板：`q/k/v/mask/out` 按
dtype 声明指针（f16 下是 `unsigned short*`），读取走 `af_ld_h`、写回走
`af_st_h`，而共享内存、行最大值、归一化因子与输出累加器仍是 float。也就
是说 f16 的 attention 在 5.121 就已经可用，但 1.2 的能力表与 5.117 的叙述
把它写成了"待做"，测试集里也没有一条 f16 用例跨过 online softmax 的块边界。

本轮不改内核，只做两件事：

1. `TestCudaAttention._build` 增加 `dtype` / `np_dtype` 参数，新增
   `test_float16_mixed_precision`：`T=300`、f16 输入、与参考语义对拍，
   容差 `rtol=atol=2e-2`；
2. 修正 1.2 表中"attention 走混合精度模板"与 `_F16_OPS` 覆盖面的描述
   （见本轮同步的三处）。

#### 二、结果

`T=300` 跨过多个 online-softmax 块。真按 f16 累加时误差会随 T 线性放大；
混合精度只在写回时舍入一次，与参考后端的差距停在 f16 的 ulp 量级
（1e-2 以内），f32 的同一图仍在 1e-5 级——误差差一个量级，说明累加确实
留在 float。

#### 三、边界

- `2e-2` 是 f16 存储 + float 累加的实测口径，不是形式化误差上界；分数动态
  范围更大或 T 更长时未扫。
- f16 只省存储与带宽，不承诺更快；5.93 / 5.91 的结论（剥离派发后 f16 GEMM
  1.19×～1.33×，但训练步被 cast 吃掉）对 attention 不自动成立。
- mask 与 q 必须同 dtype（内核按同一宽度读），IR 已有契约，运行期再校验
  一次。

#### 四、验证

- 远程 `TestCudaAttention` 9 条全绿，含新增的 f16 用例。
- 远程全量（见 5.155 的验证节）覆盖同一条路径在 f32 下的对照。

### 5.153　归约与归一化的任意轴

#### 一、做法

此前 `reduce_sum` / `reduce_max` / `reduce_mean` 与 `softmax` 只支持最后一
维，非末维在构图期被拒绝；`layernorm` / `rmsnorm` 的 `axes` 也只按末维处理。
本轮补三条路径，末维快路径一行不改：

1. **非末维归约**（`_reduce_any_axis`）：输入先取连续视图，按
   `(outer, reduce_len, inner)` 三段式展开，一个线程负责一个输出元素，沿
   reduce 维串行读。内核是 L2 描述 `reduce_axis(kind)`（sum / max / mean
   共用骨架，只有初值、合并方式与是否除以长度不同），输入统一 cast 到
   float 再累加，与末维内核一致。
2. **非末维 softmax**（`softmax_axis`）：一个线程扫一整行，三遍——求最大
   值、求 `exp` 之和、写回。`exp` 因此算两遍，换来的是不需要 shared、也
   不需要行内同步；`scale` 仍是可选第二输入（融合 pass 塞进来的标量）。
3. **任意轴归一化**（`_norm_layout` / `_norm_param`）：不做第二份数值路径，
   只在**布局**上做等价变换。归一化轴正好是最后若干维时取零拷贝二维视图；
   否则按 `[keep..., norm...]` 置换一次再转连续，算完逆置换回原轴序。
   `gamma` / `beta` 右对齐到 `x` 的秩后展平成 `(cols,)`：非归一化轴必须为
   1（随 batch 变化的参数内核装不下，显式拒绝），归一化轴上等于 `x` 的
   对应维或 1（1 走 `broadcast_to` 展开）。

选择三段式而不是"物化转置"：转置要多一份 `x` 大小的拷贝且读不连续；三段
式只需要一次连续化，读仍是步长访问。非末维归约**不承诺更快**——同一行的
元素跨了 `inner` 步，块内归约也拿不到合并访问，这里的目标只是把"只支持最
后一维"这条能力限制去掉。

#### 二、结果

- 非末维的数值与末维同口径：f16 输入先 cast 到 float 再累加，输出按存储
  dtype 写出。
- `layernorm` / `rmsnorm` 在"归一化轴 = 最后若干维"时仍是零拷贝视图，
  峰值与改造前相同；其它轴集合多一次置换拷贝，`plan_memory` 按 `x` 的元
  素量计一份，不需要新增存储假设。
- 归约内核的 block 由 `plan_reduce(total)` 给出（`total = outer * inner`），
  与非末维的实际并行度一致。

#### 三、边界

- `reduce_*` 一次只支持**一个**轴：`axis` 为 tuple / list 且长度 > 1 时显式
  拒绝（多轴归约需要"两段步长 + 折叠"，没有在本轮展开）。
- `argmax` 仍只有末维（`argmax_last`），不在本轮范围。
- `layernorm` / `rmsnorm` 的 `axes` 可以是任意**集合**（多轴），与
  `reduce_*` 的单轴限制不同：前者是归一化布局，后者是归约算子。
- `gamma` / `beta` 沿非归一化轴变化时拒绝，而不是逐行重算——内核一行一份
  参数，逐行变体会让"一行一个 block"的形态失效。

#### 四、验证

- 本地 `tests.test_plan` 的 76 条通过（归约规划部分未改）。
- 本地 `tests.test_ops_extra` 的多轴 `rmsnorm` 用例通过
  （`axes=(-2, -1)` 与 numpy 广播逐点一致）。
- 参考后端 conformance 85 executed / 0 refused / 0 failed，覆盖 56 个算子。

### 5.154　池化的三个属性：ceil_mode / count_include_pad / indices

#### 一、做法

三件事一起改，因为它们在语义上互相牵连。

1. **`ceil_mode` 从前端分解改成 IR 属性直通。** 旧前端为模拟 ceil 在输入
   右侧补 `-inf`（maxpool）或复制边界（avgpool）再切片，既改图结构，也让
   "窗口里到底有几个真实元素"变得难以数。现在 `_pool_out_dim` 按 PyTorch
   规则两步走：先向上取整，再在 `(out - 1) * stride >= size + padding` 时
   减一（去掉起点越界的多余窗口）。符号维无法表达这个分支，`ceil_mode` 要
   求静态空间维，否则构图期报错。
2. **`count_include_pad` 的分母。** 旧实现把 `1/(kh*kw)` 作为 `scale` 传进
   内核，即分母恒为核面积——这在 `count_include_pad=True` 且无 ceil 越界时
   碰巧对，其余情况全错。正确口径是"窗口落在 **padded 输入** 里的格数"：
   `True` 时区间是 `[-p, H+p)`（padding 的零计入），`False` 时是
   `[p, p+H)`；`ceil_mode` 越过 padded 输入的格子**既不计值也不计分母**。
   参考实现用 `_pool_count_window` 滑动数格数并 `maximum(..., 1)`；L2 内核
   在扫描里 `cnt++`、`acc += 在图像内 ? Load : 0`，`den = cnt > 0 ?
   1/(float)cnt : 1`；反向先数 `cnt`，散射包在 `If(在图像内)` 里。
3. **`maxpool2d_indices`。** 新增 IR 算子：输出 i32，每个窗口里最大元素的
   **平面内**展平下标 `ih * W + iw`（与 PyTorch `max_pool2d(...,
   return_indices=True)` 的第二个输出同定义，不是 `N*C*H*W` 全局下标），
   并列最大值取第一个。参考实现与 CUDA 内核同语义；`ceil_mode` 只改前向输
   出长度，梯度沿同一组窗口散射。

前端同步：`_max_pool2d` 直接把 `ceil_mode` 放进 attrs；`avg_pool2d` 支持
`ceil_mode` 与 `count_include_pad=False`，只对 `divisor_override` 报
`UnsupportedOperatorError`（它绕开"数格数"直接指定分母，内核没有这条
路径，拒绝比静默忽略安全）。`_replicate_last` / `_ceil_mode_maxpool_input`
两个分解辅助函数随之删除。

#### 二、结果

- 前端与 torch eager 对拍暴露了真实偏差：`ceil_mode` 的 avgpool 最大差到
  1.5×，根因正是分母用了 `KH*KW`。改成 padded-边界计数后逐点一致。
- 两个具体判据：8x8 输入、k3 / s2 / p0、`ceil_mode=True`，最后一格的窗口
  只有 4 个格子在 padded 输入里（分母 4，不是 9）；4x4、k3 / s1 / p1 的
  左上角分母是 9（padding 计入），`count_include_pad=False` 时是 4。
- 前端不再改写图：导出图里就是 `maxpool2d` / `avgpool2d` 节点，
  `ceil_mode` 是 attrs，没有 `concat`、没有切片。

#### 三、边界

- `den = cnt > 0 ? ... : 1` 的守卫保留：PyTorch 的窗口规则保证 `cnt >= 1`，
  但内核不靠调用方保证，0 除比"多一个分支"贵得多。
- `divisor_override` 明确拒绝，不是忽略；`count_include_pad` 与
  `ceil_mode` 同时为真时，分母按"padded 输入 ∩ 窗口"算，两个属性不互相
  抵消。
- `maxpool2d_indices` 的下标是**平面内**展平下标（`ih*W + iw`），不是窗口
  内偏移、也不是 `N*C*H*W` 全局下标；它与 `maxpool2d_backward` 的定位规则
  同一份，三者（前向、位置、反向）并列时选中同一个元素。
- `ceil_mode` 与符号空间维不兼容，构图期拒绝；这是 IR 表达能力的限制，
  不是实现偷懒。

#### 四、验证

- `tests.test_frontend`：`test_maxpool_ceil_mode`（节点仍是 `maxpool2d`、
  `ceil_mode` 为真、无 `concat`）、`test_avgpool_ceil_mode`、
  `test_avgpool_count_include_pad_false`、
  `test_avgpool_divisor_override_is_rejected`。
- `tests.test_ops_extra`：`test_ceil_mode_divisor_clips_at_padded_input`
  （8x8 / k3 / s2 最后一格分母 4；4x4 / k3 / s1 / p1 角落分母 9）、
  `test_gradient_matches_numeric_with_ceil_mode`。
- `tests.test_lower_kernel` 逐 token 比对通过：`AVGPOOL2D_SRC` /
  `AVGPOOL2D_BACKWARD_SRC` 两个对照模板与 L2 渲染结果完全相等。
- 远程 `TestCudaPaddingAndMaxPoolBackward` 16 条全绿。

### 5.155　scan 的自适应分块

#### 一、做法

5.128 的分块扫描是显式开关（`chunked_scan=<块长>`），默认仍沿时间串行：
串行的并行度只有 `elems`，长 T + 小尾维时设备填不满，瓶颈是纯依赖链；
但短序列或尾维已经喂饱带宽时，分块多出的两趟 scratch 流量只会更慢。本轮
把"什么时候分块"变成可判定的纯函数。

1. `chunked_scan` 从布尔 / 固定块长改成**三态**：`None`（自动，默认）、
   `0`（强制串行、逐位可复现）、正数（显式块长，且只有 `steps > chunk`
   才生效——块数 < 2 时第一趟只是把整段搬一遍）。构造期拒绝负数、`bool`
   与非整数。
2. `lower/plan.py` 新增 `scan_auto_chunk` / `scan_last_auto_chunk` /
   `scan_chunk`。判据只用两个量：`steps >= SCAN_CHUNK_MIN_STEPS` 且
   `elems <= SCAN_CHUNK_MAX_ELEMS` 时取块长 64。`scan_last_state` 不写整段
   输出、串行那条只有依赖链没有大流量，尾维上限放宽到
   `SCAN_LAST_MAX_ELEMS = 4096`。
3. `_scan` 与 `scan_last_state` 只调用 `scan_chunk(...)`；块长由它解析后
   传给第一趟 scratch 与第二趟内核，两处不会再各算一份。
4. `benchmarks/scan_chunked.py` 的列名改成
   `chunk{auto,serial,64,128,256}_ms`，默认口径就是自动。

阈值是实测出来的，不是猜的。M40 全扫（T ∈ {64…16384} × elems ∈
{16…4096}，块长 {串行, 64, 128, 256, 512}）先给出两个方向：`elems >= 4096`
时串行已经被带宽喂饱，分块只会多付两趟读写；`steps <= 256` 时块数太少，
块前缀摊不开。第一版把下限写成 256，本轮的自动 vs 串行矩阵（`T` × `E` 的
45 个格子）把它推到 512：T=256 的四格里三格持平或略慢，`elems=1024` 那格
反而慢 29%。

#### 二、结果

M40 自动（默认）vs 强制串行，中位，单位 µs（`f` 前向、`b` 反向、`l`
末状态；`chunk` 列是 `scan_auto_chunk` 给出的块长，0 = 串行；`l` 列用
`scan_last_auto_chunk`，尾维上限更宽，所以 `elems=4096` 那几格的 `l`
仍可能分块）：

| T | elems | chunk | f 串行 | f 自动 | b 串行 | b 自动 | l 串行 | l 自动 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 1024 | 0 | 81.5 | 78.9 | 78.8 | 79.9 | 71.5 | 72.4 |
| 256 | 16 | 0 | 126.2 | 125.2 | 131.0 | 131.3 | 120.9 | 157.7 |
| 256 | 1024 | 0 | 138.5 | 178.9 | 175.4 | 192.9 | 134.6 | 105.5 |
| 512 | 16 | 64 | 179.5 | 136.9 | 205.6 | 144.3 | 195.8 | 131.6 |
| 512 | 1024 | 64 | 249.6 | 156.0 | 288.4 | 163.5 | 300.5 | 186.3 |
| 512 | 4096 | 0 | 306.2 | 285.4 | 319.7 | 320.6 | 287.5 | 171.4 |
| 1024 | 1024 | 64 | 451.1 | 238.0 | 511.8 | 202.3 | 490.3 | 129.6 |
| 1024 | 4096 | 0 | 468.5 | 517.5 | 573.5 | 566.0 | 530.1 | 279.7 |
| 4096 | 1024 | 64 | 1324.5 | 548.7 | 1629.0 | 510.7 | 1611.7 | 267.1 |
| 8192 | 16 | 64 | 1490.6 | 155.9 | 1958.0 | 157.7 | 1907.4 | 170.0 |
| 8192 | 1024 | 64 | 2568.5 | 959.3 | 3242.9 | 943.2 | 3177.5 | 437.6 |
| 8192 | 4096 | 0 | 3043.6 | 3050.0 | 3630.5 | 3633.5 | 3242.7 | 1365.3 |
| 16384 | 16 | 64 | 3803.3 | 190.2 | 4766.4 | 187.2 | 3665.7 | 170.3 |
| 16384 | 1024 | 64 | 5075.1 | 2063.1 | 6440.7 | 1976.5 | 6310.6 | 808.3 |

- T=256 的 `l` 两列其实走的是**同一条**串行路径（`chunk` 判定为 0），
  105.5 vs 134.6、120.9 vs 157.7 的差就是同进程先后两次采样的整机漂移
  （20%～30%）——单格差值不下结论，只看方向的依据正在这里。
- 最极端的一格（T=16384、elems=16）前向 20.0×、反向 25.5×；真实形状
  （T=1024、elems=1024）前向 1.90×、反向 2.53×、末状态 3.78×。
- 关闭档（前向 / 反向的 elems=4096）差距在 -7%～+11% 之间且方向不一致
  （同一格自动有时快 6.8%、有时慢 10.5%），落在漂移里，说明阈值把这些格子
  挡在分块之外是安全的；末状态在同一格仍有 1.6×～2.4×，因为
  `scan_last_state` 本来就不物化整段输出。
- T=256 / elems=1024 的 29% 变慢是上限从 256 提到 512 的直接依据。

#### 三、边界

- 分块与串行**不逐位相等**：块间用 `(prod, sum)` 复合，舍入路径不同。要逐位
  一致只有 `chunked_scan=0`（自动档不假装能保住逐位）。
- scratch 固定 float32（不随 dtype 降 f16），大小是
  `2 * chunks * elems * 4` 字节；`elems=2048`、`T=16384` 时 `chunks=256`，
  约 4 MB。选择序列化的判据里没有显存项——这个量级在目标卡上不值得作为
  拒绝条件。
- 阈值全部来自 M40（Maxwell / sm_52）；HIP 上同一判据未扫，跨架构不可外推。
- 只测 2 维 `(T, elems)`。实际尾维的阶数不影响内核（按 `t * elems + i`
  展平），但"三维尾维的访存模式是否与二维一致"没有单独验证。
- 极端小尾维的 20× 是"串行本来就只有 16 个线程"的产物，不代表长尾维也这样。

#### 四、验证

- 本地 `tests.test_plan` 的 `TestScanChunk` 8 条：短序列、长序列、宽尾维、
  边界 511 / 512、空形状、`last_state` 更宽的尾维预算、显式值优先、块数
  < 2 时退回串行。
- 远程 `TestCudaChunkedScan` 13 条：自动 == 显式 64 且 != 串行（用
  `a ≈ 0.9` 的自建图，`a ≈ 0.3` 时初值十几步就被衰减到 ulp 以下，断言会
  空转）、T=256 短序列逐位等于串行、尾维 4096 不分块、
  `scan_last_state` 在 T=512 / 尾维 4096 仍走块长 64、构造期拒绝 `-1` /
  `True` / `2.5`。
- 远程全量 1633 tests OK（skipped=80）；conformance 参考 85/85、
  CUDA 83 executed / 2 refused / 0 failed。


### 5.156　HIP 的 pinned 暂存与设备态训练

#### 一、做法

5.87 把 CUDA 那套运行期语义（`shared_buffers` / `stream` / `run_device` / 自拷贝跳过）
搬到了 HIP，但明确留了两条口子："不接 pinned、不支持设备态训练，都在第一次 step 明确
报错"。本轮把这两条补上，并把容器的运算符面补到优化器实际需要的那一小块。

1. `hip/runtime.py` 新增 pinned 分配：`hipHostMalloc` / `hipHostFree`，`pinned(shape,
   dtype)` 返回一个持有者（`ptr` / `nbytes` / `view`），`view` 是同形同 dtype 的 numpy
   视图。dtype 校验用 `_SUPPORTED`（容器能承载的六种）而不是 `_dtype` 的四种——pinned
   只负责搬运，f64 / bool 也必须能上传，才能让"不支持"发生在 `invoke` 的能力检查里。
2. `HipBackend.pinned_array(key, arr, dtype)` 按 `(key, 形状, dtype)` 缓存一块固定内存，
   每次 `np.copyto(holder.view, arr, casting="unsafe")` 写进去。缓存字段 `_pinned` 惰性
   建立、不写进 `__init__`：`TestHipBackendSplit` 用 AST 对账两份构造的实例字段，只
   允许 `cp` / `codegen` / `sm_count`。
3. 事件与跨流依赖。捕获流与默认流之间没有隐式同步，因此 `replay` 先在当前流上 record
   一个禁用计时的事件，让捕获流 wait 它（仅当两条流不同），再按输入类型入队：numpy 输入
   走 pinned + `copy_async(H2D)`，设备输入走 `copy_async(D2D)`，最后 launch，销毁事件。
   以前 numpy 输入是同步 H2D，每步的第二次 replay 都要等主机把数据搬完。
4. `DeviceArray` 的运算符面补到优化器够用：`__add__` / `__sub__` / `__mul__` /
   `__truediv__` 及反射与原地形式、`__neg__`、`__pow__(2)`、`copy`、`zeros_like`、
   `sqrt`、`sum()`。数组 × 数组走一个带广播步长的逐元素内核；数组 × 标量走 double
   alpha + `reverse`；一元只需 neg / sqrt；`__setitem__` 的数组路径是同形同 dtype 连续
   的 D2D memcpy，标量路径是 fill。全和是两段归约（块内 double 累加 → 分片 → 单块收尾），
   不用原子加：顺序固定、结果可复现。
5. `train/_arrays.array_module` 按类型所在模块名分派，`aetherflow.backends.hip.*` 交给
   `hip.runtime`；`train/loop.py` 的 `_check_optimizable` 同步放开 HIP。

#### 二、结果

WSL ROCm（RX 7650 GRE / gfx1100，`HSA_ENABLE_DXG_DETECTION=1`）：

- `tests.test_hip_backend` 58 条全绿。新增的 `TestHipPinnedArray` 4 条（复用、dtype
  转换、replay 注册 pinned、拒绝 complex64）、`TestHipElementwise` 7 条（二元、标量 /
  反向、原地与填充、广播 / 一元、全和 / copy / zeros_like、`pow` 拒绝、联合折叠回归）、
  `TestHipDeviceStateTraining` 3 条。
- 设备态训练：SGD 10 步与 Adam 5 步的损失和全部参数轨迹都与主机态一致；
  `verify_device_state()` 的三条链（优化器参数 → 驻留参数、捕获缓冲 → 驻留参数、
  槽位与参数同一数组模块）全部通过；`clip_global_norm` 的设备分支（平方和 → 开方 →
  逐元素缩放）正确。
- 全量：本地 1652 通过 / 359 跳过；WSL HIP 1652 通过 / 324 跳过。

#### 三、边界

- pinned 池按 `(key, 形状, dtype)` 缓存，只增不减；同一个 key 换形状会再开一块，旧块
  留到后端生命周期结束。这是"每步零分配"的代价，不是泄漏。
- `copy_async` 只有在源是 pinned 时才是真异步；D2D 不经过主机，两者都不在主机侧等待。
- 逐元素运算不做隐式提升：两侧必须同 dtype，标量必须是 0 维（Python 数会被
  `np.asarray` 成 0 维，进而按 `np.float64` 传给内核）；`pow` 只接受 2；`sum` 只做
  全和；`__setitem__` 不做隐式广播写入。
- 广播索引的维度上限是 4，且两个操作数必须**联合折叠**——这是本轮修掉的缺陷，见下。
- 设备态训练要求后端提供 `capture` / `to_array` / `to_host`；参考后端没有，默认关闭。

#### 四、验证与修掉的缺陷

MIOpen 对照首轮数值对拍失败（3x3_c3 有 93.8% 元素不一致），但把 bias 置零后两路完全
相同（最大误差 5.96e-7），因此问题只可能在 `y.__iadd__(bias)`。最小复现：对连续的
`(2,16,4,4)` 原地加 `(1,16,1,1)`，四个通道都得到 channel 0 的偏置。

根因不在内核索引，而在两套折叠结果被混用：`_ew_binary` 分别对两个操作数调用
`_collapse`，连续目标按自身步长折成 `[2,256,1,1]`，广播视图折成 `[2,16,16,1]`；内核
只有一套 `dims`（取自目标），却同时乘了视图的步长表，channel 的步长因此落到 h\*w 维上，
所有通道都读成 channel 0。修法是 `_collapse_pair(shape, sa, sb)`：用同一个条件同时检查
两套步长，要么两个都可合并、要么都不合并，保证 `dims` 与两张步长表一一对应。回归用例
`test_inplace_broadcast_shares_one_decomposition` 钉住这个形状。


### 5.157　MIOpen 对照：legacy 前向在 ROCm 7.2 已不可用，改走 solution API

#### 一、做法

`benchmarks/miopen_compare.py` 用 ctypes 直调 `libMIOpen.so.1`，与 `cudnn_compare.py`
同一口径：不引 PyTorch，只依赖系统里已有的库，先对拍数值再计时。

1. legacy 的 `miopenFindConvolutionForwardAlgorithm` + `miopenConvolutionForward` 在
   ROCm 7.2 上跑不通：Find 正常返回算法，随后的 Forward 报 `No invoker was registered
   for convolution forward. Was find executed?`（status 7）。改走 solution API：
   `GetSolutionCount` → `GetSolution` → `GetSolutionWorkspaceSize` →
   `CompileSolution` → `Immediate`。
2. `miopenConvSolution_t` 的字段顺序必须照 `miopen.h` 抄：`float time; size_t
   workspace_size; uint64_t solution_id; miopenConvAlgorithm_t algorithm;`。按 int64
   猜 `solution_id` 会读到 `time` 与 `workspace_size` 的拼接值，`Immediate` 报
   `invalid solution id`。
3. `miopenConvolutionForwardImmediate` 的参数顺序是 `(handle, wDesc, w, xDesc, x,
   convDesc, yDesc, y, workspace, size, solution_id)`——wDesc 在 w 之前，与直觉里的
   "输入在前"相反。
4. 选解取库给出的 `time` 最小的那个；`CompileSolution` 放在计时窗口之外，否则第一次
   `Immediate` 会把 JIT 时间算进去。workspace 按 `GetSolutionWorkspaceSize` 的字节数
   一次分配（容器没有 uint8，用 i8，同为 1 字节）。
5. bias 两边都算进窗口：本项目的卷积核把 bias 融在里面，MIOpen 侧在窗口内用容器自己的
   `__iadd__` 补一次 `(1,O,1,1)` 广播——5.156 的折叠缺陷就是这一步暴露的。
6. 三路口径：tiled（分块直算）/ im2col（im2col + hipBLAS）/ MIOpen；某形状拿不到解时
   跳过并记录，不伪造数字。

#### 二、结果

MIOpen 3.5.1，WSL ROCm，1 轮，单位 ms（`tiled/mio`、`im2col/mio` 是相对 MIOpen 的倍数）：

| shape | tiled | im2col | MIOpen | tiled/mio | im2col/mio |
| --- | --- | --- | --- | --- | --- |
| 3x3_c3 | 2.4245 | 2.3971 | 0.3489 | 6.95× | 6.87× |
| 1x1_c32 | 2.8225 | 2.3932 | 0.5298 | 5.33× | 4.52× |
| 3x3_c8 | 5.1478 | 7.9546 | 0.3205 | 16.06× | 24.82× |
| 3x3_c16 | 5.9418 | 8.2295 | 0.4379 | 13.57× | 18.79× |
| 3x3_c24 | 7.6254 | 9.4026 | 0.3303 | 23.09× | 28.47× |
| 3x3_c32 | 11.6293 | 10.7918 | 0.3675 | 31.64× | 29.37× |
| 5x5_c16 | 6.6575 | 10.0689 | 0.3870 | 17.20× | 26.02× |
| 1x1_c256 | 6.1824 | 5.7142 | 0.3665 | 16.87× | 15.59× |
| 3x3_c64 | 11.6676 | 13.0091 | 0.3513 | 33.21× | 37.03× |
| 3x3_c128 | 4.3765 | 4.8111 | 0.3672 | 11.92× | 13.10× |

差距 4.52×～37.03×。MIOpen 在这 10 个形状上都稳定在 0.32～0.53 ms，几乎不随通道数
变化——它用的是预编译 / 汇编内核加 workspace 的隐式 GEMM 路线；自研核里 3x3_c128 反而
比 3x3_c64 快（4.38 vs 11.67 ms），说明分块核的通道维布局在中等通道数上没吃满带宽。
这与 5.69 的 cuDNN 对照（4～12 倍）同一量级，AMD 侧差距更大。

#### 三、边界

- 单轮、默认流、未开 MIOpen 的 find 缓存；选解按库给出的估计时间，不是实测。
- 固定 NCHW / f32 / cross-correlation / stride 1 / dilation 1 / groups 1，只覆盖
  `benchmarks/conv_compare.py` 的 10 个形状；f16、分组、非 1 步长都没有对照。
- 只做测量，不接生产路径——它回答"差多少"，不回答"怎么换"。
- workspace 只在 `prepare` 时按选定的解分配一次；形状或选解变了要重新 `prepare`。

#### 四、验证

- 10 个形状三路数值全部一致（rtol/atol 1e-4；零 bias 时自研 tiled 与 MIOpen 的
  最大差 5.96e-7）。
- 首轮失败定位到 `_collapse_pair` 缺陷（见 5.156 第四节），修完后重跑通过。
- 本地 1652 通过 / 359 跳过；WSL HIP 1652 通过 / 324 跳过。


### 5.158　模糊测试扩到硬件后端

#### 一、做法

1. `run_fuzz(iterations, start, backends=())` 新增可选后端列表。默认是空元组，
   所以没有 GPU 的机器上行为与以前完全一致——不因为缺硬件而改变检查范围。
2. 每张随机图在 `_check_case` 通过之后，把参考执行的输出当基准，对每个后端
   `execute` 一次并逐输出比较：浮点按 `rtol=atol=1e-4`（与 conformance 同口径，
   允许归约顺序不同），整型要求逐位相等；形状与 dtype 也必须一致。
3. `BackendCapabilityError` 记为"拒绝"而不是失败。随机图本来就会拼出后端没
   承接的形态，这正是它该说"不"的地方；其它异常、输出名不一致、数值偏差都算
   缺陷。实测拒绝全部来自 i32：HIP 只承接浮点，`slice` / `transpose` / `spike` /
   `dropout_mask` / `cast` 对 int32 明确报错，符合 5.87 的能力边界。
4. 后端名按 `get_backend` 解析。显式要求了却装不起来（`BackendUnavailableError`）
   或名字没注册（`LookupError`）都记一条失败——把"没测"报成"测过"比不测更糟。
5. `FuzzReport` 增加 `backends` / `compared` / `refused`，文本按后端给出
   "对拍 N 例（拒绝 M 例）"；CLI 增加 `--backend`，可重复也可逗号分隔。
6. `_check_case` 改为同时返回参考输出，对拍时不再重跑一遍参考执行。
7. 另加**语义变异** `_semantic_rejected`：在克隆图上用合法属性名 + 非法值构造
   三类节点，要求构图期抛错——`reshape` 元素数不守恒、`matmul` 内维不匹配、
   `cast` 到不存在的 dtype。它与属性变异互补：那里问"未知属性会不会被放行"，
   这里问"已知属性配上非法值会不会被放行"。构造必须用**克隆图自己的 Value**：
   把原图的 Value 塞进克隆图，`add_node` 会先报"引用了不属于本图的 Value"，
   三类全被拒但语义根本没被检查到——第一版就是这样，60 例"拒绝"全是假阳性。

#### 二、结果

- WSL HIP（RX 7650 GRE / ROCm 7.2）：`--iterations 100 --backend hip` 全绿，
  81 例对拍、19 例按契约拒绝、300 例语义非法节点被拒（3 例/种子），覆盖 31 个
  算子；1 次无可用变异（该种子没有内维不匹配的二维值对）。
- 拒绝原因抽样（40 个种子）：`slice` 5、`transpose` 2、`spike` 1、
  `dropout_mask` 1、`cast` 1，全部是"不支持 int32"。
- `tests/test_fuzz.py` 16 条：本地通过 14 / 跳过 2（CUDA 与 HIP 各一条），
  WSL 通过 15 / 跳过 1（CUDA）。
- 全量：本地 1659 通过 / 362 跳过；WSL HIP 1659 通过 / 325 跳过。

#### 三、边界

- 随机图仍是静态形状、最多 10 个节点，符号维不进生成器（与 5.38 一致）；
  后端对拍只覆盖这些随机图能拼出来的形态。
- 浮点用容差而不是逐位：GPU 与 numpy 的归约顺序不同，逐位相等会把正确实现
  判成失败；整型输出没有这个自由度，仍要求逐位。
- 报告只统计拒绝数量，不细分原因；定位具体拒绝要看异常文本。
- 语义变异只有三类，且都要求"构图期"拒绝；越界下标、除零这类运行期语义
  错误没有构造——它们要喂进执行器才知道，与"构图期显式报错"是两件事。
- CUDA 分支本轮没有真机，测试按可用性跳过——它是否也通过，下次上 M40 才能回答。

#### 四、验证

- `python -m aetherflow.fuzz --iterations 100 --backend hip` → 全部通过。
- 新增六条用例：HIP 随机图对拍、CUDA 随机图对拍、未注册后端记为失败、
  报告文本带后端计数、三类语义非法节点被拒、报告文本带语义计数。
- 语义变异做了负向对照：合法 `reshape`（元素数不变）被接受，非法 `reshape`
  因"元素个数不守恒: 16 -> 17"被拒；`matmul` 报"内维不匹配: 4 vs 8"，
  `cast` 报 `'f8' is not a valid DType`。三者都来自语义检查本身，不是跨图引用。
- 全量两端均为 1659 通过、0 失败（WSL 连跑两次确认）。


### 5.159　捕获窗口内的释放必须推迟

#### 一、做法

本轮跑 HIP 全量时偶发一次 `hipModuleLaunchKernel(conv2d_tiled)` 失败
（`code=901`，`hipErrorStreamCaptureInvalidated`），单独复跑不复现。对照实验：
在 `hipStreamBeginCapture` 与 `hipStreamEndCapture` 之间直接调 `hipFree`，
`hipFree` 返回 `900`，随后 `hipStreamEndCapture` 报 `901`。也就是说，
**`hipFree` / `hipHostFree` 在捕获窗口里是非法操作**。

触发条件是 GC 与捕获窗口没有协调：上一个测试留下的 `DeviceArray` 或 pinned
缓冲可能在下一个测试的捕获期间被引用计数归零，`free` 恰好落在窗口里。捕获期
的 arena 只管图内缓冲，管不到图外对象的析构，所以这是一个与 arena 无关的
时序缺口——它偶发，不是每次都能撞上。

修法：

1. `_Runtime.free` / `free_pinned` 在 `_capturing` 为真时只记账，分别推入
   `_deferred_free` / `_deferred_pinned`，不碰驱动。
2. `_CaptureSession.__exit__` 在 `hipStreamEndCapture` 之后统一
   `drain_deferred()`；放在 `finally` 里，捕获体抛异常、EndCapture 失败也补做。
3. 安全性：被图引用的缓冲由 arena 强引用，不会在窗口内被 GC；被推迟的都是
   图外对象，EndCapture 之后释放不影响图里写死的指针。

#### 二、结果

- 新增 `TestHipCaptureDeferredFree`：捕获窗口里先后析构一块设备内存与一块
  pinned 缓冲，`hipStreamEndCapture` 成功、两个队列都清空。
- WSL 全量连跑两次 1659 通过 / 325 跳过，901 没有再次出现。
- 本地 1659 通过 / 362 跳过。

#### 三、边界

- 只推迟"释放"。捕获期申请显存仍然直接报错（`malloc` 的既有检查）：图里若
  引用一块新分配的缓冲，EndCapture 之后它必须一直活着，语义与"推迟释放"
  完全不同，不能混为一谈。
- `drain_deferred` 在 EndCapture 失败时也会执行。此时流是否仍处于捕获态
  不确定，但继续占着内存没有好处。
- 这是"GC 时机"的修补，不改变捕获计划的依赖语义，也不改变 5.61 的 arena
  容量边界。

#### 四、验证

- 负向对照：窗口内直接 `hipFree` → `hipFree` 返回 900、EndCapture 报 901；
  走容器析构（记账 + drain）则捕获成功。
- 两次全量复跑均全绿，说明偶发失败与"窗口内释放"这条路径一致。

### 5.160　归约类的 block 开放，以及逐轮同号筛选器

#### 一、被排除的前提不成立

5.56 开放调优面时，注释里写的是"归约类的 shared 大小直接由 `blockDim`
决定，改 block 会连带改变 shared 预算与归约树深度"。本轮逐个核对内核后，
这条前提不成立：

- 归约族的暂存区是 `Shared("float", "smem", REDUCE_SCRATCH)`，`REDUCE_SCRATCH`
  是常量 1024（4 KB），正好等于 block 上限，**不随 `blockDim` 变化**；
- 内核沿 `blockDim.x` 跨步读输入（`For("i", Axis("tid"), Var("cols"),
  Axis("bdim"), step)`），block 小于列数只是每个线程多走几步；
- 块内归约树对任意 2 的幂都成立，而合法 block（32～1024）全部是 2 的幂。

于是新增 `_REDUCE_TUNABLE_OPS`（softmax / layernorm / rmsnorm / reduce_sum /
reduce_max / reduce_mean / argmax / batch_norm / batch_norm_stats），与
`_ELEMENTWISE_TUNABLE_OPS` 合并成 `_TUNABLE_OPS`；`tunable_ops()` 现在覆盖
除 conv2d 外的全部可调算子（conv2d 拧的是线程铺法，走 `_conv_tile`）。

launch 侧收口到一个入口，优先级与 `_launch_block` 一致：

```python
def _reduce_block(self, op: str, cols: int) -> int:
    override = self._blocks.get(op)
    if override is not None:
        return override
    if self.block is not None:
        return self.block
    return plan_reduce(cols)
```

九处 `plan_reduce(...)` 调用点全部替换：末维与任意轴的 softmax、layernorm
（两条内核共用）、rmsnorm、reduce_sum / reduce_max / reduce_mean（末维与任意
轴）、argmax、batch_norm 与 batch_norm_stats。构造参数 `self.block` 也因此
对归约类生效——它是"全局固定 block"，不是"只对逐元素生效"。

#### 二、逐轮同号筛选器

配对差的中位数与 bootstrap 区间回答的是"平均而言差多少"，但一个幅度很小、
方向相反的轮次可以被平均掉。5.149 的多数票只在**采集之间**投票，单次采集
内部仍然只看合并后的区间。`min_same_sign` 补的是这一点：

- 统计与中位差同号的轮次数（零差异不算同号），记进 `same_sign`；
- 若原本判出 `faster` / `slower`，而同号数不足门槛，降为 `unstable`；
- 原本是 `noise` 的不升级——它已经是不下结论，保留更准确；
- 门槛与 `rounds` 绑定：`0` 关闭，`1` 无意义，`> rounds` 无法成立，非整数
  一律报错；`--serial` 没有配对差，CLI 直接拒绝。

`PairedResult.min_same_sign` 与逐候选的 `same_sign` 都进 `to_dict()`，报告
能回答"这个结论是 3/5 还是 5/5"。`tune_staged` / `tune_replicated` 透传同一
参数；后者合并时保留 base 的同号计数。

典型用法是 `--rounds 5 --min-same-sign 5`：五轮方向全部一致才换默认配置。

#### 三、实测（WSL HIP，RX 7650 GRE / ROCm 7.2，`--rounds 5 --min-same-sign 5`）

| 用例 | 默认中位 | 最优候选 | 配对差 | 同号 | 判定 |
|---|---|---|---|---|---|
| softmax `[512, 4096]` | 7.894 ms | `block=256` | −0.53% | 3/5 | unstable |
| softmax | 7.894 ms | `block=1024` | −1.09% | 4/5 | unstable |
| reduce_sum `[512, 8192]` | 5.255 ms | `block=128` | −3.36% | 3/5 | unstable |
| reduce_sum | 5.255 ms | `block=512` | +7.20% | 3/5 | noise |
| layernorm `[256, 2048]` | 5.121 ms | `block=128` | +2.38% | 3/5 | unstable |
| layernorm | 5.121 ms | `block=1024` | +4.72% | 4/5 | unstable |

三个用例全部保留默认，没有候选拿到 5/5。连跑三次（同一台机器、同一形状）
的结论一致：**方向会跨采集翻转**——`reduce_sum` 的 `128` 依次是 −0.9%、
+5.3%、−3.36%；`softmax` 的 `512` 从"4/5 同号的 −4.5%"变成"+0.51%"。这与
5.63 / 5.150 的"整机漂移 + 1% 地板"是同一件事。

表里还有一处值得记的现象：layernorm 的 `128` 中位数 5.1156 ms 低于默认的
5.1207 ms，配对差却是 **+2.38%**。中位数各算各的，基线在两轮之间漂移；
配对差是同一轮内的比值，消掉的就是这部分漂移。判据不能只看"谁的中位数
更小"。

#### 四、边界

- 归约的求和顺序随块大小改变，**换 block 不逐位相等**。要逐位必须不覆盖
  （`block=None`）。测试因此对归约用浮点容差，对逐元素仍用 `assert_array_equal`。
- 同号门槛对 `faster` / `slower` 对称：一个 4/5 同号的 +4.7% 也会被降级。
  它只影响"是否下结论 / 是否改默认"，诊断用的 `diff_median_pct` 与
  `same_sign` 仍然写在报告里。
- 它不替代跨采集复现：同号只保证"这一次采集内方向一致"，`--repeats` 保证
  "换一个时间段还成立"。两者可以叠加。
- 开放旋钮不等于"一定更快"。三次采集里三个形状的最优差都在 ±5% 内且同号
  不足，说明这几张形状上归约 block 的收益在地板以下；旋钮的价值是能力面
  一致、以及在别的形状 / 别的卡上确实需要时能显式指定。

#### 五、验证

- `tests/test_cuda_backend.py` 两条：`_reduce_block` 的优先级（离线，含
  "只影响被覆盖的算子"）与 `block=128` 覆盖 4096 列时的数值近似；原先
  "归约类不接受 block"的断言反转成"归约类在调优面里"。
- `tests/test_hip_backend.py` 新增 `TestHipReduceTuning` 三条：调优面包含
  归约类、`_reduce_block` 覆盖优先、`block=128` 跑 `[256, 2048]` 的 softmax
  与默认近似一致。
- `tests/test_tuning.py` 新增 `TestSameSignFilter` 五条：5/5 保留结论、
  6/7 因门槛 7 降为 `unstable` 且 `same_sign=6`、门槛 6 时保留、非法门槛
  （负数 / 1 / 超过轮数 / 非整数）报错、`tune_staged` 与 `tune_replicated`
  透传。
- `benchmarks/tune.py` 的 `TUNE_CASES` 加三个归约用例（softmax / reduce_sum /
  layernorm），CLI 加 `--min-same-sign` 并在 `--serial` 下拒绝。
- 本地 1669 通过 / 367 跳过；WSL HIP 1669 通过 / 327 跳过。

### 5.161　常量折叠的收益量化，以及逐节点 GPU 计时

#### 一、折叠到底省了什么

5.36 交付常量折叠时只验证了"数值一致、图合法"，没有回答"省了多少"。
本轮补的两件事是同一件事的两面：先能测出每个节点的 GPU 时间，再用它
量化折叠前后的整图差异。

量化脚本构造三类图，在 WSL HIP（RX 7650 GRE / ROCm 7.2）上跑
`default_pipeline()` 前后各 30 次取中位：

| 用例 | 折叠链 | 节点数 | wall | GPU |
|---|---|---|---|---|
| mask | `mul(c1,c2) → add(c3) → sub → step`，结果被 `mul(x, mask)` 读到 | 9 → 3 | −11.0% | **+22.4%** |
| all_constant | `mul → add → exp → reduce_sum`，输出标量 | 7 → 1 | −80.4% | −89.2% |
| compress | `exp → mul → reduce_sum`，输出标量 | 5 → 1 | −73.8% | −85.6% |

结论分两层：

1. **折叠产物越小，收益越大**。两条标量输出把 GPU 时间压到原来的
   10%～15%：省掉的是几次 kernel 与它们之间的中间缓冲。
2. **折叠产物大、原计算便宜时，GPU 时间是倒退的**。mask 用例折出一个
   256×256 f32 常量（256 KB），执行期每个 `constant` 节点都要一次主机到
   设备的拷贝；原来那 4 个小 kernel 在 GPU 上比一次 256 KB 非 pinned 拷贝
   更快。wall 仍降 11%（少 4 次下发），但 GPU 时间涨 22%。

第 2 条不是折叠本身的错，而是"常量节点不做设备缓存"的代价：同一份常量
在每次 `execute` 都重新上传。本轮的边界记录后来由 5.172 收掉——常量副本
随执行计划缓存（计划期上传一次），mask 用例从 3.3838 ms 降到 0.3390 ms。

#### 二、折叠只物化存活边界

量化过程顺带暴露了 pass 的一个真实缺陷：旧实现给**每个**可折节点都发一个
`constant`，包括只被其他折叠节点使用的中间结果。于是 mask 用例折完节点数
仍是 9——四个计算节点变成四个常量节点，启动次数一个没少，还多出几次上传，
第一次测出来 wall +161%、GPU +295%。

修法是只物化"存活边界"：

```python
boundary = {value.id for value in graph.outputs}
for node in graph.nodes:
    if node.output.id in folded:
        continue
    for value in node.inputs:
        if value.id in folded:
            boundary.add(value.id)
```

中间结果的使用者本身都在 `folded` 里，链尾的值就是边界值；原始常量节点
则由流水线里的 DCE 收走。改完之后 mask 是 9 → 3（而不是 9 → 6 带三条死
常量），实测从 +161% 转为 −11%。

#### 三、逐节点 GPU 计时

`NodeTiming` 增加 `gpu_ms`：后端提供 `gpu_clock` 时，包装器在每个节点
前后各记一个计时事件（CUDA 用 `cupy.cuda.Event`，HIP 用
`hipEventCreateWithFlags(0)`），整次执行完**只同步一次**，再逐个读
`elapsed`。逐节点插同步会把测量对象换成同步开销本身，事件对不会。

- `InstrumentedBackend(inner, recorder, gpu_timing=True)`：只在内层有
  `gpu_clock` 时启用；`gpu_timing=False` 可关掉，省下每节点两个事件。
- 事件在收尾时统一 `release`（HIP `hipEventDestroy`、cupy `close`）；
  读数失败（驱动不给、事件没记上）只让 `gpu_ms` 保持 `None`，不改变
  执行结果。HIP 侧另加 `__del__` 兜底，异常路径不泄漏事件。
- `RunRecord.node_gpu_totals()` 单独聚合，不和 `node_totals()` 混在
  一起：没有时钟的后端应报"空"，而不是一排 0。`to_dict()` 只在有读数时
  才写 `gpu_ms` / `node_gpu_totals`，旧记录仍能原样解析。
- `RunRecord` 校验负读数会抛 `ValueError`——时钟倒流是驱动或测量
  错误，不能静默写成 0。

参考执行器没有这个时钟，`gpu_ms` 恒为 `None`；这也是测试里最关键的
负向对照："没有观测"和"0 ms"必须可区分。

#### 四、验证

- `tests/test_passes.py` 新增 `test_only_boundary_results_are_materialised`：
  一条三步常量链只留一个 constant（跑完整流水线），并与参考后端逐位对拍。
- `tests/test_telemetry.py` 新增 `TestGpuNodeTiming` 六条：逐节点读数、
  `node_gpu_totals` 与 JSON 往返、无时钟时字段全缺、`gpu_timing=False`、
  读数异常不影响执行且必须释放事件、负读数报错。
- `tests/test_hip_backend.py` 新增 `TestHipGpuTiming`：512 阶 matmul 经
  `InstrumentedBackend` 报出正的 `gpu_ms`，聚合按算子名。
- 本地 1677 通过 / 368 跳过；WSL HIP 1677 通过 / 327 跳过。


### 5.162　IR 跨版本迁移：结构化改写，而不是重建后打补丁

#### 一、5.8 的版本号只解决了"能不能读"

5.8 定的规则是：主版本不同拒绝、次版本高于本实现拒绝、次版本低则接受，
理由是"新增字段一律可选、未知字段忽略"。这条规则对**只加可选字段**的
演进成立，但它默认了一件事：旧图缺的字段，读入后靠算子自己的默认值补齐
就够了。一旦某个默认值在新版本里变了、或者旧字段要拆成两个新字段，这个
默认就不成立——而当前的实现没有任何地方能表达"从 1.0 到 1.1 需要做什么"。

迁移工具补的就是这一段。它不改变 `check_version` 的准入规则：能不能读
仍由主/次版本决定；迁移只负责"在允许读之后，把结构补齐到当前语义"。

#### 二、迁移作用在序列化结构上

关键选择是迁移的输入是 `serialize.to_dict` 的产物，不是 `Graph`：

- 旧图可能含当前实现已经删掉的算子、或当前 `Op` 表不认识的属性；先
  `from_dict` 重建会在构图期直接失败，迁移根本没有机会跑。
- 结构是纯 JSON：迁移只改 `nodes[i].attrs`、`inputs`、`outputs` 这类字段，
  不需要构造 `Value` / `Node`，也不会碰到 `Graph.seal` 之后的不可变约束。
- 迁移完再走一遍正常的 `from_dict`，类型推导、名字唯一性、`verify` 全部
  照旧生效——迁移不能成为绕过校验的旁路。

因此 `from_dict` 的顺序是：`check_version(原始版本)` → `migrate(data)` →
`check_version(迁移后版本)` → 重建。迁移过的图把原始版本记在
`Graph.migrated_from`（进程内构造的图是 `None`）；再次导出写的是迁移后的
`ir_version`，来源只作凭证。

#### 三、注册表：单出边、只升不降、缺一段报一段

```python
register_migration((1, 0), (1, 1), add_default_eps)
migrate(data)                       # 默认目标 = IR_VERSION
migration_path((1, 0), (1, 2))      # ((1, 0), (1, 1))
```

三条约束都是为了让路径**唯一**：

- **每个源版本最多一条出边。** 有分叉时选哪条取决于注册顺序，同一份数据
  在不同进程里可能走不同路径——这类不确定性没有排查手段。
- **目标严格高于源。** 降级需要另一套语义（旧实现不认识新字段），不在
  本工具范围内，直接拒绝。
- **不跳过中间版本。** 注册 `(1.0)→(1.2)` 后请求到 `(1.1)` 会报"越过了
  请求的终点"；缺边时报"缺 1.0 的出边"并列出所有已注册的边。

迁移函数收到一份**深拷贝**（版本相同时直接返回入参，不拷贝——正常路径
没有额外开销），返回值必须是 `dict`。返回 `None` 会被当成"跑完没改"与
"写错了"的混合体，统一按 `TypeError` 拒绝。

#### 四、验证

- `tests/test_ir_migration.py` 17 条：
  - 路径：同版本为空、缺边消息带缺口与已注册列表、降级拒绝、两步链按序、
    越过终点拒绝；
  - 注册：目标不高于源、重复源、非 callable、版本格式错误；
  - 结构：整条链生效、入参不被就地修改、同版本返回同一对象、迁移返回
    非 dict 报错、非 dict 入参报错；
  - 集成：`from_dict` 在模拟"本实现 1.1"时走迁移并记下
    `migrated_from=(1,0)`、当前版本读入不记标记、缺迁移时加载失败。
- 本地全量通过（数字见 §0）。

#### 五、边界

- 主版本迁移仍然拒绝：结构不兼容时，工具不提供"尽力而为"的路径。
- 默认值怎么补由迁移作者决定。工具不知道某个字段在新版本里应该取什么，
  也不提供"按类型猜一个零值"的兜底——猜错的默认值比缺字段更难发现。
- 迁移只处理 `from_dict` 这一条入口；`deploy` 的 bundle 读入最终也走
  `loads`，因此同样受益，但工具不额外钩住其他自定义反序列化路径。

### 5.163　编译期与调优插件契约：从"两个实现"提炼，而不是从一家猜

#### 一、为什么现在才写

`contracts.py` 在 5.x 就明确写着"``CodegenContract`` / ``TuningContract``
现在不写：第二个后端长什么样还不知道，按一个实现猜出来的接口，等真接入时
几乎必然要改"。现在条件满足：

- CUDA 与 HIP 的私有编译入口都是 `_kernel(key, source, func)`，语义一致：
  key 是编译缓存键，source 是厂商源码（字符串或返回字符串的可调用对象），
  func 是入口名；同一个 key 命中缓存返回同一个句柄。
- 调优面由同一个 `DeviceBackend` 提供，两边共用 `tunable_ops` /
  `tuning_space` / `set_tuning` / `clear_tuning` / `tuning_config` 五个方法。

所以提炼的是"两者共同的部分"，不是"先设计一套再让实现去填"。

#### 二、两个协议

```python
@runtime_checkable
class CodegenContract(Protocol):
    def kernel(self, key: Any, source: Any, func: str) -> Any: ...

@runtime_checkable
class TuningContract(Protocol):
    def tunable_ops(self) -> FrozenSet[str]: ...
    def tuning_space(self, op: str) -> TuningSpace: ...
    def set_tuning(self, op: str, config: Mapping[str, Any]) -> None: ...
    def clear_tuning(self, op: Optional[str] = None) -> None: ...
    def tuning_config(self) -> Mapping[str, Mapping[str, Any]]: ...
```

刻意的留白：

- codegen 不约束句柄类型。CUDA 是 `cupy.RawKernel`，HIP 是模块函数；契约
  只承诺"可调用且能按该后端的方式启动"，把句柄形状写死等于要求第三方后端
  复制某一家。
- 调优不约束搜索算法与存储格式。那些在 `aetherflow.tuning` 里，对任何实现
  本契约的后端通用；写进契约就等于要求每个后端自带一套调优器。
- `TuningSpace` 只在 `TYPE_CHECKING` 下导入，避免 `contracts` 在导入期
  把 tuning 的 numpy 依赖带进来。

#### 三、公开入口与探测结果

`DeviceBackend` 新增公开的 `kernel()`，转发到子类的 `_kernel()`。公开这
一层的原因很直接：契约探测只看方法名，工具需要一条不依赖下划线命名的
稳定路径；而 `_kernel` 仍是厂商实现，`DeviceBackend` 的默认实现照旧抛
`NotImplementedError`。

`CONTRACTS` 的声明顺序变成 `device / codegen / tuning / memory / stream`，
因此 `implemented_contracts()` 的输出也变了：

- CUDA 与 HIP：`("device", "codegen", "tuning", "memory", "stream")`；
- `ReferenceBackend`：仍是 `()`——最小准入面没有被抬高；
- 只有 `device_info` 的假后端仍然只报 `("device",)`。

探测语义不变：`isinstance(obj, Protocol)` 只检查方法是否存在，不检查签名，
更不检查返回值含义。"声称具备"要靠实际调用与 conformance 证明。`TuningContract`
要求五个方法齐全，少一个（例如只有 `tuning_space`）就不算。

#### 四、验证

- `tests/test_backend_contracts.py` 27 条：新增 codegen / tuning 相互独立、
  两协议与参考后端零契约、`TuningContract` 五方法缺一不算；CUDA 真机组
  的契约列表更新为五项，并新增 `kernel()` 真编译探针（同一 key 两次取到
  同一句柄）。
- `tests/test_hip_backend.py` 的设备契约测试补 `codegen` / `tuning` 两项。
- 本地全量通过（数字见 §0）；WSL HIP 全量通过。

#### 五、边界

- 契约只描述"能力存在"，不描述性能与正确性；conformance 仍以实际执行
  为准。
- `kernel()` 的 `source` 可以由插件决定方言（CUDA C / HIP C++），契约不
  要求跨厂商可移植；跨后端可移植靠 IR 与 `verify_portable_only`，不是靠
  复用同一段源码。
- 调优配置的键与取值仍由后端定义，`tuning_config()` 只保证"能原样读回"。


### 5.164　多设备：先把 HIP 的 0 号卡解绑，再谈契约

#### 一、缺口在哪

两处写死的 `0`：

- `CudaBackend.__init__` 用 `getDeviceProperties(0)` 取 SM 数，`device_info()`
  也从 0 号卡读名字、版本、显存——**构造时切到别的卡也读不出区别**；
- HIP 更彻底：`hipGetDeviceProperties(..., 0)`、`hipDeviceGetAttribute(..., 16, 0)`，
  `runtime` 根本没绑 `hipSetDevice` / `hipGetDevice`。整个后端只能跑 0 号卡。

`DeviceContract` 回答的是"**这块**设备是什么"，一个方法，单卡后端也能实现。
"N 块卡分别是什么、现在在哪块、切过去"是另一组语义——5.89 的
`StreamContract` 注释里就写着"不写多设备枚举与切换：HIP 侧没有对应实现"。
现在 HIP 侧补上了，两个实现可以互相参照，才轮到提炼。

#### 二、HIP：`runtime` 层补齐，属性按设备缓存

- ctypes 绑上 `hipGetDevice` / `hipSetDevice`；`hipGetDeviceCount` 的结果在
  构造时存成 `device_count`。`current_device` **故意不缓存**：别的库也可能
  调 `hipSetDevice`，缓存会读到一个过期的"当前设备"，后续分配与内核就落在
  错误的卡上。
- `set_device(index)` 先在 Python 侧校验类型与范围，再调驱动。越界时错误
  信息带上"进程可见 N 块（0..N-1）"，而不是把 HIP 的 code 原样抛出去。
- `_parse_props(index)` 从"只解析一次"改成按索引缓存字典——每块卡的
  名字与 gfx 架构各不相同，一份缓存装不下。
- `hipMemGetInfo` 只认当前设备，没有下标参数。`_total_memory(index)` 因此
  临时切过去读、`finally` 切回来；读失败返回 0，不让一次查询失败拖垮整个
  `device_info()`。
- `sm_count` 也按设备缓存。它是**当前设备**的属性，`set_device` 之后必须
  重新取，否则自动 block 会拿上一块卡的 CU 数去铺这一块卡的网格。
- 模块级新增 `device_count` / `current_device` / `set_device` /
  `enumerate_devices`，与既有的 `sm_count` / `device_info` 同一层薄封装。

#### 三、CUDA：去掉 0，构造期选卡

- `__init__` 新增 `device: int = 0`（追加在参数表末尾，位置调用不受影响）。
  先校验类型与范围（`bool` 不算 `int`），再 `cp.cuda.Device(device).use()`，
  最后才问 SM 数——顺序反了就是拿 0 号卡的数值去铺别的卡。
- `device_info()` 改用 `cp.cuda.Device().id`；`enumerate_devices()` 逐卡在
  `with cp.cuda.Device(i):` 里读 `mem_info`（它只认当前设备），`with` 退出
  时还原，枚举本身不改变"现在用哪块卡"。
- `set_device(index)` 校验后 `use()`，并同步 `self.sm_count`。

#### 四、契约：`DeviceManagerContract`，三个方法

```
enumerate_devices() -> Tuple[DeviceInfo, ...]   # 按索引顺序
current_device() -> int
set_device(index) -> None                       # 非法索引当场报错
```

与 `DeviceContract` 分开而不是合并：单卡后端答得了"我用的是什么设备"，
答不了"还有哪几块"；合并会让 `implemented_contracts` 的粒度失真。
`CONTRACTS` 里排在 `device` 之后（同属启动期一次性查询），后面依次是
codegen / tuning / memory / stream。

协议只查方法存在性，查不到"切换后 `device_info` 与 SM 数跟着走"这条语义；
它写进契约 docstring，由真机用例守。

#### 五、验证

- `tests/test_backend_contracts.py`：新增 `DeviceManagerContract` 与
  `DeviceContract` 互不蕴含、三方法缺一不算、两个 GPU 后端类在不实例化的
  情况下都声称具备、参考后端零契约；CUDA 真机组契约列表更新为六项，并新增
  枚举 / 越界拒绝 / 报错后仍在原卡上的用例。
- `tests/test_hip_backend.py`：设备契约补 `device_manager`；新增真机用例
  覆盖枚举长度与 `device_count()` 一致、`set_device(0)` 后可读、越界 /
  负数 / 非 int 三种拒绝、拒绝后当前设备不变；另有一条离线用例只查
  `runtime` 的符号齐全（无 ROCm 的机器也能跑）。
- 本地全量通过（1703 用例 / 371 跳过），WSL HIP 全量通过（1703 / 329），
  HIP 新用例在 RX 7650 GRE 上实跑。**CUDA 真机本轮未复跑**（研究机停机），
  CUDA 侧的多设备用例处于跳过状态。

#### 六、边界

- 只验证了单卡机器：枚举长度恒为 1，"多块卡各自的名字与 SM 数不同"这条
  没有被真机覆盖。它依赖的是驱动按索引返回属性，没有额外假设，但确实没跑。
- `set_device` 不迁移已有状态：捕获计划、arena、常驻缓冲都绑定在原来的卡
  上。切卡前应先同步并丢弃这些对象；跨卡数据拷贝（peer copy）与多卡并行
  没有做——那是分布式执行的问题，不是设备枚举的问题。5.190 补了经主机中转的
  跨卡拷贝与数据并行最小闭环，仍不是 peer-to-peer，也没有真正的并发。
- 枚举会临时切换当前设备读显存，结束后还原；如果另一个线程同时依赖
  "当前设备"这一进程级状态，两者仍需外部加锁。HIP 与 CUDA 的都是进程级
  当前设备，这一点不会因为本文档的封装而改变。


### 5.165　分组 / depthwise 卷积：不再逐组 im2col

#### 一、旧路径的代价

5.34 让 groups>1 走"逐组切片 + im2col + GEMM"：每组物化一份
``(N, CG*KH*KW, HO*WO)`` 的列矩阵，再发一次 GEMM，最后把每组结果写回
``out[:, gi*og:(gi+1)*og]``。三条账都不好看：

- **列矩阵**：depthwise 下 ``CG=1``，列矩阵与输入同大，物化一次读一遍写一遍，
  真正的乘法却只有 ``KH*KW`` 次；
- **GEMM 形状**：``(N*HO*WO, KH*KW) × (KH*KW, 1)``，K 与 N 都极瘦，
  cuBLAS 的效率接近最坏情况；
- **launch 次数**：``groups`` 组各一次 im2col + 一次 GEMM，depthwise 的
  ``groups=C`` 就是 ``2C`` 次下发。

#### 二、分组直算核

``lower.kernels.conv2d_grouped``（L2 描述，生产路径）：

- 一个线程一个 ``(n, o, ho, wo)`` 输出元素，``_flat_loop`` 网格步进；
- 只遍历自己那一组的 ``CG * KH * KW`` 个乘加，输入与权重都直接从全局读；
  同一 block 的相邻线程共享输出通道与权重，复用由 L1 承担，不搭 shared
  staging——分组卷积的 ``CG`` 通常很小（depthwise 是 1），staging 的成本
  比省下的读高；
- 位姿：``g = o / OG``、``cstart = g * CG``、输入基址
  ``x + (n*GROUPS*CG + cstart + cg)*H*W``、权重基址 ``w + (o*CG + cg)*KH*KW``
  （布局 ``(O, CG, KH, KW)``，只用 ``o*CG`` 就落到组内正确位置）；
- bias 在写回时融合，与分块 / 隐式核一致；
- 全程 float，不按 dtype 参数化；f16 仍走 im2col + hgemm。

**实现时踩到的坑**：第一版直接复用池化的 ``_window_scan`` 生成窗口循环，
本地 emit 用例通过、HIP 对拍在 ``dilation=2`` 上出现成片偏差——池化窗口
没有膨胀，``_window_scan`` 里的下标是 ``ho*SH - PH + ki``，少了 ``ki*DH``。
改成内核自己的双层循环后，``ih = ho*SH - PH + ki*DH``、``iw`` 同理；离线
用例现在逐条钉住这两行，以及组起点与权重基址。

#### 三、派发

- ``conv_impl="auto"`` + f32 + groups>1 → 分组直算；f16 在上一个分支已经
  落到 im2col，不受影响；
- ``conv_impl="im2col"`` → 仍是逐组 im2col，保留为强制对照；
- ``conv_impl="tiled" / "implicit"`` → 仍明确报错（它们的 shared staging
  假设"每个输出通道看全部输入通道"）；
- groups == 1 的三条路径（tiled / implicit / im2col / cudnn）逐字不变。

#### 四、HIP 容器的按组写回

强制 im2col 路径在 HIP 上此前根本跑不通：``DeviceArray.__setitem__`` 要求
两侧都连续，而 ``out[:, gi*og:(gi+1)*og] = part`` 的目标在通道维上被夹着。
5.33 起这条路径只在 CUDA 上验证过，所以一直没暴露。

新增 ``_COPY_PAIR_SRC``：连续源 → 带步长目标。索引分解复用
``_collapse_pair``（同一套 dims 同时折叠源与目标步长），逐字节拷贝；
``__setitem__`` 在目标不连续时走它，源不连续时先 ``copy()`` 物化。整块
赋值仍然要求同形同 dtype，隐式广播写入仍拒绝——放开的是布局，不是语义。

#### 五、验证

- ``tests/test_lower_kernel.py``：分组核源码里的 dilation、组起点、输入 /
  权重基址逐 token 断言；f16 下指针仍是 ``const float*``（说明确实不参数化）。
- ``tests/test_cuda_backend.py``：新增分组 + dilation 与参考后端对拍、
  auto 与 ``conv_impl="im2col"`` 对拍（累加顺序不同，用 ``allclose`` 而非逐位）。
- ``tests/test_hip_backend.py``：容器层新增"非连续目标 + 非连续源"两条
  写回用例；conformance 的 grouped / depthwise 形态在 RX 7650 GRE 上实跑。
- WSL 临时脚本实测：同一张 ``groups=2`` 的图，auto 路径 ``_conv2d_grouped``
  被调用 1 次、与参考后端最大差 1.43e-6；强制 im2col 路径同样 1.43e-6，
  说明新写回内核在真实分组路径上工作。
- 本地全量通过（1707 用例 / 374 跳过），WSL HIP 全量通过（1707 / 331）。
  **CUDA 真机本轮未复跑**，CUDA 侧两条新用例处于跳过状态。

#### 六、边界

- 没有 shared staging：``CG`` 很大（groups 很小但又不是 1）时，权重会被
  反复从 L1 读，理论上不如分块核。这个区间与"groups>1 通常伴随小 CG"的
  实际用法不重叠，没有为它加分支，也没有做逐形状的实测对照。
- 没有给分组核加调优旋钮：``tile_w`` / ``roc`` / ``cb`` 是分块核的线程铺法，
  分组核根本没有分块，硬塞进去只会让调优面报出一个不影响结果的参数。
- 没有做 channel-last / NHWC 变换，也没有专门的 depthwise 向量化（一次算
  多个通道）；这两条都要改数据布局，属于另一个量级的改动。
- 只做了语义与正确性的验证，没有做性能基准：旧路径的列矩阵与极瘦 GEMM
  是"应该更快"的动机，不是本轮实测的结论。


### 5.166　分组卷积的性能对照：把"应该更快"量出来

#### 一、为什么单独做一个入口

``benchmarks/conv_compare.py`` 只测 ``groups=1``（分块核 vs im2col），而且
写死 ``CudaBackend``。分组核的收益要能同时回答两件事，才值得写进文档：

1. 在典型 depthwise / 低 ``CG`` 形状上快多少；
2. 在 ``CG`` 较大时会不会被 GEMM 反超——直算核没有 shared staging，
   权重靠 L1 复用，理论上存在一个交叉点。

新入口 ``benchmarks/grouped_conv_compare.py``：形状集固定 9 个，``--backend``
走注册表（``cuda`` / ``hip`` 都可），每条路径先 ``warmup=3``、再
``repeat=10`` 取中位数，每形状 ``--rounds`` 轮取最好。计时前后都同步；
HIP 的同步走 ``hip.runtime.synchronize()``（``benchmarks.run._sync`` 已按
是否存在 ``cp.cuda`` 分派）。数值先对拍（``rtol=1e-4``）再计时——两条路
不一致时对照表没有意义。

形状集（NCHW，f32，权重 ``(O, C/groups, KH, KW)``）：

- depthwise：3×3 / 3×3 stride 2 / 5×5 / 3×3 dilation 2 / 1×1；
- 一般分组：``groups=2/4/8/16``，覆盖 ``CG=64/16/4/16``。

#### 二、HIP 实测（RX 7650 GRE / ROCm 7.2 / WSL DXG 直通）

| 形状 | 直算核 ms | 逐组 im2col ms | 倍数 | im2col 列矩阵 MB |
|---|---:|---:|---:|---:|
| dw3x3_32 | 3.2363 | 29.6335 | 9.16× | 13.78 |
| dw3x3s2_64 | 6.6438 | 61.5672 | 9.27× | 27.56 |
| dw5x5_64 | 1.0061 | 55.2419 | 54.91× | 12.50 |
| dw3x3d2_32 | 0.9376 | 28.6697 | 30.58× | 1.72 |
| dw1x1_64 | 5.1454 | 59.3684 | 11.54× | 3.06 |
| g4_64 | 5.5602 | 13.9575 | 2.51× | 27.56 |
| g8_32 | 1.1791 | 8.1895 | 6.95× | 4.50 |
| g2_128 | 1.7692 | 4.9693 | 2.81× | 6.89 |
| g16_256 | 1.0915 | 14.6061 | 13.38× | 3.45 |

#### 三、结论与边界

- 9 个形状全部更快，最小 2.51×（``groups=4``、``CG=16``），最大 54.91×
  （depthwise 5×5：im2col 每组只有 25 个乘加，列矩阵却要物化一整份）。
- 没有观察到交叉点：``CG`` 最大的 ``g2_128``（``CG=64``）仍是 2.81×。
  这与"逐组 im2col 的固定开销（物化 + 独立 launch）远大于权重复用收益"
  一致，但不能外推成"任意 ``CG`` 都不会反超"——形状集没覆盖
  ``CG > 64`` 且空间维很小的区间。
- 列矩阵 1.72～27.56 MB 全部消失，这是显存峰值的直接下降，与耗时无关。
- 绝对毫秒来自 WSL DXG 直通，不可与裸机 ROCm 或 CUDA 数字横比；同一时刻
  两条路径的相对差有效。
- 只采了一轮（内部 ``rounds=3`` × ``repeats=10``），没有做 5.146 的跨采集
  复现。按 5.140～5.150 的教训，单次采集的相对差在 ±3% 以内不可信；这里
  最小的 2.51× 也远大于该门槛，所以方向可以下结论，具体倍数不可当基线。
- CUDA 侧没有跑同一对照（研究机停机）；HIP 与 CUDA 共用同一份内核与派发，
  但仍不把 HIP 的倍数当成 CUDA 的结论。

复现命令：

```bash
python -m benchmarks.grouped_conv_compare --backend hip
python -m benchmarks.grouped_conv_compare --backend cuda --rounds 5
```

### 5.167　遥测包装器补上契约转发：包一层不许把能力吃掉

**动机**：5.89 修过一次同类缺陷——`InstrumentedBackend` 没转发当时刚上提的
`allocator` / `new_stream`，是真机全量第一次跑才暴露的。此后 5.163 从两份实现里提炼出
`CodegenContract`（`kernel`）与 `TuningContract`（`tunable_ops` / `tuning_space` /
`set_tuning` / `clear_tuning` / `tuning_config`），5.164 又提炼出
`DeviceManagerContract`（`enumerate_devices` / `current_device` / `set_device`），
包装器一处都没跟上。

这不是"少一个方便方法"。`implemented_contracts` 是 `runtime_checkable` 的存在性探测，
基准、调优工具与 `agent_api` 都按它判断后端能力：包装后少认 `device_manager`，一台
多卡机器就会被看成单卡；少认 `tuning`，有旋钮的后端会被当成没有；少认 `codegen`，
"打印内核源码"这条对照路径直接消失。全程不报错、不警告，只在结果里少几项。

**做法**：在 `__init__` 末尾对九个名字做同一件事——内层有就绑定，没有就不绑：

```python
for name in (
    "kernel",
    "tunable_ops", "tuning_space", "set_tuning", "clear_tuning", "tuning_config",
    "enumerate_devices", "current_device", "set_device",
):
    if hasattr(inner, name):
        setattr(self, name, getattr(inner, name))
```

不用 `__getattr__` 兜底：那会让任何拼写错误悄悄落到内层，包装器的"透传面"也就不再是
一份可审阅的名单。显式名单的代价是新增契约要同步这里，这条代价由测试兜住。

**验证**：

- 离线：六项全齐的假执行器包一层后 `implemented_contracts` 必须等于
  `tuple(CONTRACTS)`；`set_device` / `set_tuning` / `clear_tuning` 的调用确实打到内层
  对象（改的是内层状态，而不是包装器自己记一份）；只实现 `device_info` 的假后端包一层
  后仍然只认 `("device",)`，不许被"升格"。
- 真机：HIP 上 `TelemetryRecorder().wrap(backend)` 之后整表与内层相等、`current_device()`
  一致。真机这一条是必须的——5.89 的同类缺陷正是只在真机上出现。
- 本地 1711 个用例 / 375 跳过、WSL HIP 1711 个用例 / 331 跳过，全绿；本轮新增 4 个
  用例（3 个离线 + 1 个 HIP 真机）。

**边界**：

- 只转发 `CONTRACTS` 里的六项；`capture` / `resident_parameters` 不在其中，包装器
  仍然不覆盖，`group_nodes` 也是故意不覆盖的（遥测要的是逐节点派发耗时）。
- 转发的是内层的绑定方法，耗时与显存统计仍由包装器自己记——转发契约不等于把包装层
  变成透明代理。
- 名单是手写的，新增契约仍要同步；整表相等的断言让"下次漏一项"在本地就变红，而不是
  等到真机全量。


### 5.168　`kernel()` 的公开面收口：两家句柄的形态不一样

**动机**：5.163 把 `kernel(source, func)` 提炼成 `CodegenContract` 的唯一入口，契约只写了
"返回一个可用的句柄"，没有写句柄长什么样。两份实现各自不同：

- CUDA：`cupy.RawKernel` 的实例，调用方式是 `handle(grid, block, args)`，**没有** `launch`；
- HIP：`hip/runtime.py::_Module` 交出的是持函数地址的包装对象，只有 `launch(...)`。

这与 5.89 的漏转发、5.164 的单卡假设同一类问题——**直到真机全量第一次跑才暴露**：
`test_kernel_entry_compiles_source` 断言 `handle.launch` 存在，CUDA 侧当场 `AttributeError`。

**做法**：`DeviceBackend.kernel()` 不再把厂商句柄原样交出，而是缓存一层 `_KernelHandle`：

```python
class _KernelHandle:
    __slots__ = ("_fn",)
    def __call__(self, *args, **kwargs): return self._fn(*args, **kwargs)
    def launch(self, *args, **kwargs): return self._fn(*args, **kwargs)
```

两个名字指向同一个底层调用，因此"capture 里的启动方式"与"对照工具里的启动方式"都能用，
不需要在调用侧判后端。缓存键仍是 5.163 的那一个，所以同一 key 拿到的仍是同一对象
（`is` 相等），捕获与源码打印不会因为多包一层而失去幂等。

**验证**：`test_backend_contracts` 的 `kernel()` 用例改成断言
`callable(handle) and callable(handle.launch)`；CUDA 真机全量从 1 FAIL 转为全绿。

**边界**：包装只统一**调用形态**，不隐藏参数语义。CUDA 的 grid/block 是元组、HIP 的也是，
但参数列表里的指针类型仍由厂商决定；`_KernelHandle` 不是跨后端的可移植句柄，换后端仍要
重新 `kernel()`。`__slots__` 是为了让每个内核句柄的额外开销固定为两个槽位。

### 5.169　多卡真机：切卡要清掉绑在旧卡上的东西

**动机**：5.164 把枚举与切换提炼成 `DeviceManagerContract`，但真机只有单卡，"枚举长度 1、
越界拒绝、报错后仍在原卡"之外全是纸面结论。本轮拿到 3 × Tesla M40 24GB 的机器
（驱动 535.309.01 / CUDA 12.2），多卡路径才有实测对象。

**做法**：`set_device` 不只是 `cp.cuda.Device(index).use()`。以下四类缓存分别绑在
**旧 context / 旧设备 / 旧卡显存**上，切卡后继续用就是错的：

| 缓存 | 绑在哪 | 不清的后果 |
|---|---|---|
| `_modules`（NVRTC 编译产物） | CUDA context | 在 1 号卡上加载 0 号卡编译的模块，启动即报 invalid device function |
| `_functions` / `_kernel_handles` | 同上 | 同左 |
| `_blas`（cuBLAS handle） | 当前设备 | 句柄内的工作区绑在旧卡，算出来是错数或直接报错 |
| `_resident`（驻留参数） | 显存 | 参数在 0 号卡上，1 号卡的内核读不到 |
| 捕获计划（`_invalidate_plans`） | 显存 + 流 | replay 在旧卡的图里 |

同时 `sm_count` 从新卡属性重新读（它决定 `auto_block`），`_tiered` arena 先 `close()`
再置空——它的地址来自旧卡，不能带到新卡。HIP 侧做同一套（`hip/runtime.py` 的
`set_device` 也清 `_modules` / `_functions`），两家因此对称。

**验证**（`TestCudaMultiDevice`，3 卡全可见时 5 例全绿，单卡机器整组跳过）：

- `test_enumerates_every_visible_device`：枚举长度等于 runtime 报的卡数，每张的名字与
  显存都非空；
- `test_runs_on_another_device_after_switch`：切到 1 号卡跑同一张图，与 0 号卡逐位一致；
- `test_switch_drops_resident_parameters`：先在 0 号卡常驻参数，切到 1 号卡后 `_resident`
  为空，重新常驻后结果仍对；
- `test_each_device_runs_its_own_graph`：0 号卡与 1 号卡各跑一张不同的图，互不影响；
- `test_rejects_bad_index`：越界 / 负数 / 非 int 当场报错，且 `current_device()` 不变。

`TestCudaMultiDevice` 的 `setUp` 记录 `origin`、`tearDown` 还原，避免多卡用例把后续用例
留在别的卡上——测试顺序不是契约的一部分。

**边界**：`set_device` 不迁移已有捕获计划 / arena / 常驻缓冲，而是**丢弃**它们。这是
显式的代价：跨卡复用意味着让调用方决定"重新上传还是跨卡拷"，框架不替它猜。跨卡拷贝
（peer-to-peer）与多卡并行执行仍未做，也不在本轮范围内。

### 5.170　CUDA 侧的执行期分层内存：复用捕获期的那本账

**动机**：5.131 给 HIP 做了 `TieredAllocator`（小对象分档、中对象 arena、大对象直调），
并新增 `DeviceBackend.memory_mode="container"|"tiered"` 与 `_memory_scope` 钩子；CUDA 侧
当时**显式拒绝**并说明未接。同一台多卡机器上，这成了两家唯一的运行期面差异。

**做法**：不新写分配器——`cuda/memory.py::Arena` 本来就实现了"chunk + best-fit + 相邻
合并"，`malloc(size)` 的返回形状正好满足 cupy 的 `set_allocator` 约定。`_memory_scope`
因此变成：

```python
previous = cp.cuda.get_allocator()          # 保存现场
cp.cuda.set_allocator(self._tiered.malloc)  # 安装
...
cp.cuda.set_allocator(previous)             # 成对还原
self._last_arena = self._tiered             # memory_stats() 读它
```

容器模式（默认）走 `nullcontext()`，一个字节都不碰——默认行为与之前逐位相同。

**顺带修掉的一处真缺陷**：`Executor.run` 只 `to_host` 输出、从不把设备侧的值还回去，
arena 里因此留着"这一轮的输入副本 + 输出块"。第一版
`test_repeated_execute_reuses_arena` 的第二遍 `peak_bytes` 比第一遍高 50%
（3072 vs 2048），根因就是逐轮累积。修复是 `run` 在取完主机副本后 `dispose` 输出，
并 `dispose` 那些"与调用方传入对象不是同一个"的输入设备副本——同一对象是调用方的，
框架不能替它释放。

**验证**（`TestCudaTieredExecution`，真机 4 例）：

- `test_tiered_matches_container`：同一张图两种模式下输出逐位一致；
- `test_allocator_is_restored_after_execute`：执行前后 `cp.cuda.get_allocator()` 是同一个
  对象（成对安装 / 还原，异常路径也走 `finally`）；
- `test_container_mode_does_not_install_allocator`：默认模式不安装分配器；
- `test_repeated_execute_reuses_arena`：连跑两遍，`peak_bytes` 相等。

**边界**：`Arena` 不是 `TieredAllocator`——它没有按 512 B 分档的小对象自由表，小分配会
落进 chunk 里按 best-fit 切，碎片行为不同。这里的目标是"CUDA 也有执行期 arena"，
不是"两家字节级同构"。`memory_stats()` 的口径跟着执行策略走（5.105）：`release=False`
时峰值等于总量。

### 5.171　CUDA 上的分组卷积：5.166 的跨厂商结论被推翻

**动机**：5.166 在 HIP 上量出"直算核比逐组 im2col 快 2.51×～54.91×"，文档里的措辞是
"分组卷积的性能对照"，读起来像跨厂商结论。CUDA 侧没跑过——这正是 5.166 风险行写的
"CUDA 侧同一对照未跑"。

**做法**：`grouped_conv_compare.py --backend cuda`，固定 2 号卡（0/1 号被 llama-server
占约 20 GB）。脚本本身不用改，5.166 之后唯一的改动是显式取 `grouped` 与 `im2col` 两条
实现（`auto` 在 5.171 之后不再代表直算核）。

**数据**（median ms，9 个形状，`--rounds 3`）：

| 形状 | grouped | im2col | 直算核相对 im2col |
|---|---|---|---|
| dw3x3_32 | 0.6948 | 5.5203 | 7.95× |
| dw3x3s2_64 | 2.5777 | 11.0779 | 4.30× |
| dw5x5_64 | 0.4589 | 9.8678 | 21.50× |
| dw3x3d2_32 | 0.3542 | 5.1136 | 14.44× |
| dw1x1_64 | 1.0696 | 10.4541 | 9.77× |
| g4_64 | 2.9842 | 2.3763 | **0.80×** |
| g8_32 | 0.6621 | 1.8579 | 2.81× |
| g2_128 | 2.4473 | 1.0774 | **0.44×** |
| g16_256 | 0.6795 | 2.7546 | 4.05× |

三条结论：

1. **depthwise（groups == C）一律更快**，4.30×～21.50×，与 HIP 同向；
2. **非 depthwise 不是**：`g4` 与 `g2` 两个大 `CG` 形状上，im2col + cuBLAS 反超
   （0.80× / 0.44×）。直算核没有 shared staging，`CG*KH*KW` 大时权重访存是重复的全局读，
   而 im2col 把它变成一次 GEMM；
3. HIP 的 2.51×～54.91× 不能外推到 CUDA——同一份内核源码，两家 BLAS 的相对强度不同。

**据此收窄 `auto`**：`conv_impl="auto"` 只在 `groups == C`（depthwise）时走直算核，
其余分组形状回落到 im2col；`conv_impl="grouped"` 变成显式选项，愿意承担那个风险的
调用方可以强制。`tiled` / `implicit` 对 `groups>1` 改为明确报
`BackendCapabilityError`，而不是悄悄算错或悄悄换实现。

**验证**：

- 新增 `test_auto_uses_grouped_only_for_depthwise`：monkeypatch 两条实现记录 `groups`，
  depthwise 命中 `[4]`、非 depthwise 命中 `[]`；
- 原有 `test_grouped_dilation` / `test_grouped_matches_im2col` 改为显式
  `conv_impl="grouped"`，不再依赖 `auto` 的隐含选择；
- HIP 复测（同一脚本，`--backend hip`）：2.07×（`g4`）～45.77×（`dw5x5`），
  非 depthwise 在 HIP 上仍是正收益——这正是"不能跨厂商外推"的反向证据。

**边界**：分组核没有 shared staging，也没有 NHWC / depthwise 向量化；`CG` 更大
（如 `groups=2` 且 `C=512`）的区间仍未覆盖。想让非 depthwise 也稳赢，需要的是把权重
搬进 shared 再复用，而不是调 launch 参数。


### 5.172　常量节点的设备副本：跟着计划缓存，不跟着每次执行重传

**动机**：5.161 量化常量折叠的收益时留下一处倒退——mask 用例（`c1*c2 + c3 - 0.5` →
`step`，256×256）逐节点 GPU 时间反而 +22.4%。原因不在折叠本身：折叠把中间值变成
`constant` 节点后，每个 `execute` 都会执行一次 `_op_constant`，而它的输入是属性里的
**嵌套 Python 列表**——`np.asarray(65536 个 float 的列表)` 就要 1～3 ms，比省下来的
那几次逐元素核贵得多。折叠产物越大、原计算越便宜，倒退越明显。

**做法**：常量的值只写在属性里、与运行期输入无关，因此可以随执行计划缓存：

- `_GroupPlan` 新增 `constants: Dict[value_id, device_array]`；
- `DeviceBackend._prepare_plan` 在计划期对"单节点且算子是 `constant`、dtype 是
  f32/f16"的组调用一次 `_op_constant`，结果存进去；
- `run_device` 遇到这类组直接取缓存值，不再走 `invoke`；`release=True` 的存活表把
  常量输出加进 `keep`，否则"最后一次被读"会把它还给分配器，缓存当场被打空；
- 切卡 / 关闭 arena 走 `_clear_plan_values()`（清计划缓存，下一次执行按新地址重建）。

**捕获期的坑（真机验证时才暴露）**：第一版缓存直接复用了预热轮的副本，HIP 的设备态
训练 5 步后 loss 是 1221 / 1145 / 1074…，而主机态是 4.37 / 4.23 / 4.10…。根因是
`HipBackend.capture` 在预热后调用 `Arena.recycle()`——它把预热期的地址整体收回并交
给捕获轮重新切分，于是图里写死的常量指针指向了后来被中间值覆盖的内存。修复是在
两家的 `capture` 里于"预热结束、开始捕获"之间调 `_drop_plan(graph)`：计划重建、
常量在捕获轮重新上传，图里记录的才是本轮地址。CUDA 侧没有 `recycle`，但同样走这条
路径，两家因此一致。

**验证**：

- `TestConstantPlanCache`（离线，3 例）：两遍执行只上传一次、常量不被 `release` 回收、
  `_clear_plan_values` 之后重新上传；
- `TestCudaConstantCache`（真机，2 例）：跨 `execute` 只调一次 `_op_constant`，
  清计划后调第二次；
- `TestCudaGraph.test_capture_with_constant_uses_capture_round_copy`（真机）：常量 +
  整图捕获 + 三次 replay 与参考逐位一致；
- WSL HIP 的 `TestHipDeviceStateTraining`（SGD 10 步 / Adam 5 步轨迹）重新全绿——
  它正是被 `recycle` 那条路径打红的用例。

**M40 实测**（`CUDA_VISIBLE_DEVICES=2`，同一张折叠后的 2 节点图，50 轮取中位数）：

| 口径 | 整图 wall | 常量上传次数 |
|---|---|---|
| 每次重传（等价修复前） | 3.3838 ms | 50 |
| 随计划缓存 | 0.3390 ms | 1（预热轮） |

−90.0%（9.98×）。这里的耗时主要是 `np.asarray` 解析嵌套列表与随后的 H2D，不是内核
本身——2 节点图的内核时间在几十微秒量级。

**边界**：只缓存"单节点 `constant` 组"；被融合进大组的常量、由前端折叠进 attrs 的
标量参数仍按原路径走。计划按 `graph.revision` 缓存，因此"每次重新构图"的调用方拿不到
这个收益（那本来也不是热路径）。`InstrumentedBackend` 为了逐节点计时会绕开批量派发，
它观察到的常量上传成本仍是每轮一次——这是遥测口径的必然，不是缓存失效。


### 5.173　`sparse_linear` 对 cuSPARSE SpMM：把 H2D 从计时里拿掉之后反超

**动机**：项目里的稀疏算子没有一个能和现成库直接对照，`sparse_linear` 是例外——
它的 `indices` / `values` / `offsets` 就是 CSR 三件套，"事件驱动"与"稀疏矩阵乘一次
权重"是同一件事。5.135 的端到端结论是"events 比 dense 慢约 15%"，但那是含压缩与
两趟拷贝的整网口径，内核本身快慢一直没测过。

**做法**：`benchmarks/sparse_compare.py` 把同一份 CSR 分别喂给三条路：

1. `sparse_linear`（本项目内核，图里只有这一个算子）；
2. `cupyx.scipy.sparse.csr_matrix.dot(dense)`（cupy 背后就是 cuSPARSE 的 SpMM）；
3. 把 `(S, K)` 脉冲矩阵真的物化再乘一次（物化开销算进计时）。

验证用 `assert_allclose` 对拍第 2 条路。**第一条实测的坑**：初版量的是
`backend.execute(graph, feed)`，结果我们全是 0.17×～0.49×——但 `execute` 每次都会把
numpy 输入拷上设备，而权重是 `K×N`（最大 16 MB），量到的主要是 PCIe 带宽。改成先把
输入放到设备上、只量 `run_device` 之后，结论整体翻转。

**数据**（M40，`CUDA_VISIBLE_DEVICES=2`，median ms，`--rounds 3`）：

| 形状（S×K×N，每行非零比例） | ours | cuSPARSE | 物化稠密乘 | vs cuSPARSE | vs 稠密 | 稀疏 / 稠密缓冲 |
|---|---|---|---|---|---|---|
| 512×512×512，2% | 0.1654 | 0.2863 | 1.0856 | **1.73×** | 6.56× | 0.04 / 1.00 MB |
| 512×1024×1024，2% | 0.2973 | 0.6336 | 1.4411 | **2.13×** | 4.85× | 0.08 / 2.00 MB |
| 256×2048×2048，0.5% | 0.2339 | 0.5278 | 1.4882 | **2.26×** | 6.36× | 0.02 / 2.00 MB |
| 64×4096×1024，10% | 0.5796 | 1.3783 | 1.2663 | **2.38×** | 2.18× | 0.20 / 1.00 MB |

结论：在内核层面，本项目的散射比 cuSPARSE SpMM 快 1.73×～2.38×，比"物化稠密再乘"
快 2.18×～6.56×；稀疏缓冲比稠密小 5～100×。这与 5.135 的整网负结果不矛盾——那里
多出来的是 `spike_offsets` / `spike_csr` 的压缩与两趟额外拷贝，不是这次比较的对象。

**边界**：只在 CUDA 上跑。WSL 的 ROCm 环境没有 cupy（hipSPARSE 的等效入口需要
ctypes 手写绑定），HIP 侧暂缓——这是外部条件，不是跳过。形状覆盖 2%～10% 稀疏、
`S=64～512`；CUDA 上 SpMM 的小矩阵调度开销较大，稠密侧还没到"稀疏必然更慢"的区间，
更大 `N` 的交叉点未测。脚本与 `grouped_conv_compare.py` 一样不属于门禁。


### 5.174　非末维归约：把归约轴的并行度借过来

**动机**：5.153 让归约与 softmax 支持任意单轴，但非末维与末维走的是两套内核：末维是"一个 block 负责
一行、块内归约"，非末维是"一个线程负责一个输出、沿归约轴串行读"。后者有两个隐含前提——输出平面自己
足够大（并行度全由它提供），且相邻线程的 `i` 连续（`inner` 够宽才能合并访存）。`inner=1` 时两条同时
失效：对 `(B, T, 1)` 沿 T 归约只有 B 个线程，每个线程跨 4096 个元素读同一列，线程之间隔着一整行。
5.153 记的"不承诺更快"指的就是这个：能力补上了，但这条路径在窄 `inner` 上近乎纯串行。

**做法**：在同一个 `_reduce_any_axis` 里加第二条铺法，按"输出平面能不能自己把机器填满"选路。

1. 新内核 `lower/kernels.py::reduce_axis_tiled(kind)`：网格 `(outer, cdiv(inner, tx))`、块 `(tx, ty)`。
   `threadIdx.x` 沿 inner（相邻线程读相邻元素，一个 warp 覆盖一整段连续 inner），`threadIdx.y` 切归约轴
   （`r += blockDim.y` 跨步），块内沿 y 做对数步树归约。sum / mean / max / min 共用一份骨架，mean 在
   写回前除以 `reduce_len`；f16 只在写回时舍入一次、读入一律转 float——与 5.50 的末维归约同一口径，
   因此两条路的差别只在归约顺序。
2. prelude 补三个原语 `block_reduce_sum_y` / `block_reduce_max_y` / `block_reduce_min_y`：下标取
   `threadIdx.y * blockDim.x + threadIdx.x`（不被 x 方向的步长带偏），每级一次 `__syncthreads()`。
   与其余 prelude 一样，HIP 与示例方言 novac 引用同一份 strings，因此方言不必各写一份。
3. 计划是纯函数 `lower/plan.py::plan_reduce_tiled(inner, reduce_len)`：
   `tx = pow2_block(inner, REDUCE_TILE_X=32)`、`ty = pow2_block(reduce_len, BLOCK_MAX // tx)`。
   两者都取 2 的幂（树归约按 `>> 1` 折半，非 2 的幂会漏项），且 `tx * ty <= BLOCK_MAX == REDUCE_SCRATCH`。
   例如 `inner=1` → `(1, 1024)`，`inner=1024` → `(32, 32)`，`inner=3` → `(4, 256)`（向上取整，多出的
   x 线程只读不写回）。
4. 判据 `DeviceBackend._use_tiled_reduce(inner, total)`：`inner < REDUCE_TILE_X`（凑不满一个 warp，
   合并访存无从谈起）或 `outer * inner < sm_count * BLOCK_DEFAULT`（输出平面填不满一屏 SM，一维铺法
   同样没有并行度）时走两维，否则留在一维。M40（24 SM）上的门槛是 6144 个输出元素。抽成方法只为让
   基准能强制走某一条（`benchmarks/reduce_any_axis.py::forced` 在实例上盖掉它，`finally` 装回），
   生产路径只经这里分派。

**数据**（3 × Tesla M40 24GB，`CUDA_VISIBLE_DEVICES=2`，median ms，`--rounds 3`，两条路都强制跑过一遍）：

| 用例（形状，轴） | reduce | inner | 输出元素 | 两维 | 一维 | auto | 一维 / 两维 |
|---|---|---|---|---|---|---|---|
| `(1024, 4096)` axis=0 | 1024 | 4096 | 4096 | 0.1660 | 0.2151 | 0.1671 | 1.30× |
| `(1024, 4096, 1)` axis=1 | 4096 | 1 | 1024 | 0.2968 | 2.2971 | 0.2477 | **7.74×** |
| `(64, 4096, 64)` axis=1 | 4096 | 64 | 4096 | 0.4102 | 0.6082 | 0.4095 | 1.48× |
| `(256, 333, 8)` axis=1 | 333 | 8 | 2048 | 0.0987 | 0.1109 | 0.0983 | 1.12× |
| `(32, 2048, 32)` axis=1 | 2048 | 32 | 1024 | 0.1620 | 0.3474 | 0.1577 | **2.14×** |
| `(1, 4096, 1024)` axis=1 | 4096 | 1024 | 1024 | 0.1627 | 0.5308 | 0.1630 | **3.26×** |
| `(128, 333, 3)` axis=1 | 333 | 3 | 384 | 0.0843 | 0.0932 | 0.0852 | 1.11× |
| `(64, 4096, 256)` axis=1 | 4096 | 256 | 16384 | 1.3897 | 1.2880 | 1.2885 | 0.93× |

最后一行是**应当留在一维**的格子：`outer * inner = 16384` 已超过门槛，判据选一维，auto 与一维一致
（1.2885 / 1.2880），强行两维反而慢 8%。其余七行 auto 全部等于两维。

**结论**：窄 `inner` 的收益与"一维铺法损失了多少并行度"成正比。`inner=1` 时 7.7×（另一轮 0.2366 /
2.2792 → 9.63×，两轮差距来自整机漂移）；`inner=32` 且输出平面只有 1024 时 2.1×；`inner=1024`、输出
1024 时 3.3×；`inner>=32` 且输出平面已到 4096 时回落到 1.5×；`inner=3` / `8` 这种极小规模只剩 1.1×，
因为两种铺法都被 launch 开销主导（0.08～0.10 ms 的量级）。判据的价值不在"两维永远更快"，而在它能在
16384 个输出的格子上正确留在旧路径——把 `sm_count` 那一条去掉就会稳定慢 8%。

**边界**：只在 M40（sm_52，24 SM）上标定；`sm_count` 是判据里唯一的机器相关项，HIP 与其它架构未扫。
`tx * ty <= REDUCE_SCRATCH` 意味着一次最多 1024 个线程切归约轴，更长的 reduce 靠 `r += blockDim.y`
循环，所以归约长度不影响正确性、只影响每线程的串行长度（这就是 `inner=3` 收益有限的原因：`ty` 已顶到
256，`reduce_len=333` 仍需每线程串两轮）。f16 容差与末维归约同样放宽到 1e-2（累加顺序不同）。`argmax`
不在这条路径上（仍只有末维）。conformance 补了 `reduce_max_axis0` / `reduce_mean_axis0` 两条用例，
参考 / CUDA / HIP 三处都是 87 个用例、85 executed / 2 refused / 0 failed。


### 5.175　运行期语义错误：让"数据非法"也说清楚

**动机**：5.38 的模糊测试只覆盖构图期——随机图 + 五条不变量 + 非法属性变异（5.96 又补了"合法属性名 +
非法值"）。但有整整一类错误它碰不到：**图本身合法，喂进去的数据非法**。下标与分母都是数据，verifier
看不见它们，只有执行器才知道。这类错误一旦被静默接受，产物是"看起来算完了"的错误结果，比报错危险得多。
本轮把这条路径固定下来时，真的抓到两处静默：

- `embedding` 的正向越界由 numpy 抛 `IndexError`，但**负下标会静默回绕**到倒数几行（`table[-1]` 就是
  最后一行），反向的 `np.add.at` 同样回绕——梯度被加到错的表项上，loss 照常下降；
- 整数 `div` 的分母为 0 时，`np.divide` 先升到 float64 得到 `inf`，参考执行器再把结果转回 i32，
  `inf → INT_MIN` 只留一条被默认配置忽略的 warning，输出是 −2147483648。

**做法**：

1. 参考语义补齐两处检查。`ir/ops.py` 新增 `_check_embedding_ids`，在 `embedding` 与
   `embedding_backward` 里统一要求 `0 <= ids < rows`，违规抛 `ValueError` 并报出实际区间；`div` 单独
   注册一份 `_div_ref`：**浮点**除零按 IEEE 放行（用 `np.errstate` 吞掉 numpy 警告，不再污染宿主日志），
   **整数**分母出现 0 当场拒绝，错误信息说明"要 inf/nan 请先 cast 到浮点"。整数除数非零时仍是截断除法，
   与最终 cast 一致，并有用例钉住。
2. 模糊测试新增一组**固定构造**的运行期用例 `fuzz._runtime_cases()`：`embedding` 正向越界 / 负下标、
   `embedding_backward` 负下标、`sparse_linear` 越界 `indices`、`spike_linear` 越界 `indices`、
   `spike_csr` 溢出 `offsets`、整数 `div` 分母为 0、浮点 `div` 分母为 0，共 8 条。前七条要求抛
   `ValueError`，最后一条要求**不**抛错且结果含 inf——把"除零"的两种语义分开钉住，任何一边被改写都算
   失败。它们不进随机构图：随机生成的前提是"图合法"，这里要的恰恰是"图合法、数据非法"。
3. `fuzz._check_runtime` 统一分类：抛 `ValueError` 算通过；抛其他异常（如 `IndexError`）记失败，因为
   那说明错误没被翻译成约定的出口；该报错却被接受同样记失败，`FuzzReport.runtime` 计数并进报告文本。
   运行期检查放在种子循环之前，这样"数据非法"的缺陷不会被随机图的失败遮住。

**数据**：新增 9 个用例（`test_ops_extra` 里 2 个 embedding 越界 + 3 个除零，`test_fuzz` 里 4 个），三处全量：

| 环境 | 用例 | 跳过 | 结果 |
|---|---|---|---|
| 本地（py3.14，无 GPU） | 1749 | 391 | OK |
| CUDA 真机（3×Tesla M40，`CUDA_VISIBLE_DEVICES=2`） | 1739 | 104 | OK |
| WSL HIP（RX 7650 GRE / ROCm 7.2） | 1749 | 347 | OK |

`test_silently_accepted_invalid_data_is_a_failure` 与 `test_unexpected_exception_is_a_failure` 把失败
路径本身也测了：前者喂一条"该报错但其实成功"的用例，后者抛 `IndexError`，都要求 `_check_runtime` 记成
失败而不是放过。

**结论**：模糊测试现在有两层——构图期（非法属性 / 非法值 / 语义非法节点）与运行期（非法数据）。两层
共用同一个失败通道，报告里分开计数。

**边界**：检查做在**参考语义**这一层，也就是"框架对这批数据应该怎么答"的定义处；CUDA / HIP 的内核
不做逐元素下标范围检查——设备端无法抛异常，做检查要么每步 D2H 一次 `ids`，要么在核心里写越界标志再
同步复查（5.98 的 `err` 标志只覆盖稀疏事件路径，`embedding` 还没接）。因此 GPU 后端的契约仍是"调用方
保证下标合法"，与"GPU 上越界读是未定义行为"这一硬件事实一致；这条差异写在 §7 的风险表里。整数 `div`
的截断语义与浮点除零的 IEEE 语义同样只钉在参考侧，GPU 侧沿用内核的自然行为。


### 5.176　`spike_csr`：把行内串行换成块内扫描

**动机**：5.35 起脉冲压缩的第二趟（`spike_csr`）是"一个线程负责一行、沿列号串行"。这条铺法有一个
硬约束——行内必须**按列号递增**写，否则 `sparse_linear` 逐事件累加的浮点顺序就与稠密矩阵乘不一致，
逐位对拍会失败。约束本身没错，但实现把它当成了"不能并行"，于是并行度只有行数：`steps=1` 的
`(1, 65536)` 只有一个线程在干活，`(64, 8192)` 只有 64 个。

**做法**：把一行摊给一个 block，用块内前缀和保住递增顺序。

1. 内核 `lower/kernels.py::spike_csr`：`blockIdx.x` 取行，`blockDim.x` 沿列号**分段**扫（每段
   `blockDim.x` 个列号），每段调一次 `block_scan_sum`，第 `tid` 个线程拿到"本段排在自己前面的命中数"，
   写位置是 `lo + 已写 + 前缀`。段间串行（`已写` 逐段累加）、段内并行，顺序天然递增。
   段数 `ceil(width / blockDim.x)` 对整块一致，不会有一半线程跳过 `__syncthreads`。
2. prelude 新增 `block_scan_sum(int v, int* smem)`：Hillis-Steele 对数步扫描，每步"读上一轮 → 同步 →
   写 → 同步"，块总数留在 `smem[blockDim.x]`（扫描区之后的第 0 格），调用方在尾部屏障之后读。HIP 与
   novac 引用同一份 strings。
3. 计划是两个纯函数：`plan_csr_scan(width)` 给块大小（`pow2_block(width, 256)`，宽 8 就 8 个线程，
   上限 256），`plan_csr_scan_enabled(steps, sm_count)` 决定用哪条铺法——`steps < sm_count * 32` 时走
   并行，否则留串行。槽位是 `CSR_SCAN_SLOTS = 1026` 个 int（4.1 KB）。
4. 设备端把"命中数 ≠ `hi - lo`"也记进 `err`：此前设备侧只在越界时置位，事件数与起点表不符会被**静默
   丢弃**；参考实现一直是双向检查的。顺带修掉这条不一致。
5. 旧铺法保留为 `spike_csr_serial`（标注"仅对照"），基准用 `_use_scan_csr` 的实例覆盖强制走它——
   与 5.174 的 `_use_tiled_reduce` 同一手法，没有开关就量不出两条路的差。

**数据**（3 × Tesla M40 24GB，`CUDA_VISIBLE_DEVICES=2`，median ms，`--rounds 3`，先逐位对拍再计时）：

| 用例 | steps | width | nnz | 并行 | 串行 | auto | 串行 / 并行 |
|---|---|---|---|---|---|---|---|
| `s8192_k512_2pct` | 8192 | 512 | 83689 | 3.3629 | 3.0380 | 2.9406 | **0.90×** |
| `s4096_k1024_2pct` | 4096 | 1024 | 83689 | 2.7127 | 2.7102 | 2.7112 | 1.00× |
| `s2048_k4096_1pct` | 2048 | 4096 | 83783 | 5.1130 | 5.1141 | 5.1158 | 1.00× |
| `s256_k4096_10pct` | 256 | 4096 | 105028 | 0.7883 | 1.1969 | 0.7898 | 1.52× |
| `s64_k8192_5pct` | 64 | 8192 | 26064 | 0.8541 | 1.6187 | 0.8541 | 1.90× |
| `s1_k65536_2pct` | 1 | 65536 | 1274 | 3.8405 | 8.4782 | 3.8424 | **2.21×** |

`auto` 每一行都取了更快的那个：前三个走串行，后三个走并行。

**结论**：收益与"串行铺法浪费了多少并行度"成正比。`steps=1` 时并行版是 2.2×——这是能力缺口，不是
常数因子优化：原来只有一个线程扫 65536 个列号。`steps=64` / `256` 时 1.9× / 1.5×。而 `steps>=2048`
时串行版本身就有足够多线程，每线程的长循环靠 ILP 藏住访存延迟，并行版反而多付块内同步——所以判据
不能写成"永远并行"。

**边界**：拐点（每 SM 32 个长循环线程）只在 M40 上标定，两轮测量的分界一致（2048 行两轮都是 1.00× 与
0.95×，256 行都是 1.51×～1.52×），但更宽的形状谱没扫。`plan_csr_scan` 的 256 上限来自"一段里线程
再多也只是多层同步"；`width` 超过 256 时靠多段串行，段数增加不影响正确性。块内扫描的 `__syncthreads`
要求网格精确等于行数（不能像一维逐元素那样被 `_grid` 截断到 `max_blocks`），`_op_spike_csr` 里因此
单独算 `grid = steps`。`spike_csr_serial` 只用于基准，不参与任何生产分派。


### 5.177　语义级 L2 的索引层：把线性化收成纯函数（范围评估）

**动机**：5.42 的 L2 是语法级，``Index`` 只认一维下标，多维张量怎么线性化由每个内核手写
``Bin("*")``。5.57 曾逐字比对过一遍，结论是"重复只值得合并一处"——但那次比的是**文本**，
没有回答"这些下标里有多少是同一套规则"。本轮先量化，再决定做多大。

**评估（`lower/kernels.py`，5524 行、78 个内核函数、164 处 `Bin("*")`、77 处 `Bin("+")`）**：

| 形态 | 代表 | 是否同一套规则 |
|---|---|---|
| 行主序线性化 | `_image_slice` 的 `(n * C + c) * H * W`、`_tap` 的 `ih * W + iw` | 是，被 10 个池化 / 卷积内核复用 |
| 解线性化 | `_split_output_index` 的 `wo → ho → c → n` | 是，被同 10 个内核复用 |
| tiling / 跨步 / 共享内存偏移 | 分块卷积、attention、scan | 否：括号习惯、类型、步长来源各不相同 |

前两类是"同一套规则被复制了十次"，后一类是每个内核自己的布局决策。差别正是 5.42 划的线：
线性化属于语法，tiling 属于语义。

**做法**：新增 `lower/indexing.py`，只有三个面：

1. `Layout(dims, sizes)`：行主序、元素为单位；最高维长度可以是 `None`（它只参与解线性化的
   最后一次除法，而那里除的是第二维长度）。
2. `offset(coords)`：折叠成 `((c0 * S1 + c1) * S2 + c2)`；坐标少于维数时末尾乘剩余维的长度
   乘积（`_image_slice` 的形态）；乘积链的**首因子**提升到 64 位（`int * int` 在
   `(long long)H * W` 这种形状上会溢出）；折叠和落到乘法时显式写 `Paren`——打印器不推断优先级。
3. `unravel_steps(flat, names)`：生成 `Let` / `AugAssign`，末维先取模、中间维"取模再整除"、
   第二维只取模、最高维取商。语句顺序与 `_split_output_index` 逐条相同。

三处 helper 改为调用它（`_split_output_index` / `_image_slice` / `_tap`），生产源码
**逐 token 不变**——`TestPooling` 的 7 个内核与 `TestConv2dGrads` 的 2 个内核都拿手写模板
逐字对照，这条测试就是"抽象没有改行为"的证据。

**明确没做的（以及为什么）**：

- `concat_copy` 的下标是 `((o * out_axis) + offset + s) * inner + j`，其中 `o * out_axis`
  有一层手写括号。统一规则不会产生它；要复刻就得给 `offset` 加"哪一段包括号"的参数，抽象
  退化成参数化的 `Bin` 拼装器，比手写更难读。这一类保持原样。
- 卷积 / attention / scan 的索引带 tiling、跨步与共享内存偏移，那是布局决策，不属于线性化；
  这一层刻意不做，边界与 5.42 一致。
- 不追求 78 个内核全覆盖：只有"同一套规则被复制多次"的地方，抽象才付得回成本。

**验证**：`tests/test_indexing.py` 17 条——把生成的表达式**求值**（括号错位打印出来仍然
"看着对"，只有算一遍才知道）、部分坐标、运行期尺寸、`cast=None`、200 组随机 `offset` 与
行主序参考值、50 组 `unravel` × `offset` 互逆、8 条非法输入。三处全量：本地 1786 个用例 /
398 跳过、WSL HIP 1786 / 354 跳过、CUDA 真机 1776 / 104 跳过。

**边界**：`Layout` 不查变量类型——只有 `Num` / `Cast` 会被跳过，别的都当整数处理，所以
"首因子已经是 64 位"要调用方保证；`unravel_steps` 至少两维（一维下标本来就是坐标本身）；
`offset` 不做零坐标消除，现有调用点没有 0 坐标，加规则只会让 token 与手写模板分叉。


### 5.178　跨块 reset：带出的卷积窗口必须按最后一次 reset 清掉

**动机**：5.99 / 5.104 的 `reset` 只遮住了**块内**的卷积抽头与 SSM 递推，没有处理
"块结束时带出的卷积窗口"。反例是 K=4、`[10,20,30,1]` 之后接 `[2]`：第二块的窗口里
仍有 `30`，它属于 reset 之前那一段，于是输出 32 而不是 3。这个问题只在**分块续接**
时出现，整段一次跑完看不出来——5.99 的"分块 == 整段"恒等式当时只在没有跨块 reset 的
形状上成立。

**做法**：把块内累积和 `_reset_prefix` 提到 `new_state` 之前，新增
`_mask_stale_conv_state` 逐行判断。`new_state` 的第 `s` 行对应输入位置
`p = steps - (K-1) + s`；若区间 `(p, steps-1]` 内发生过 reset（等价于
`cs[steps-1] - cs[p] > 0`，`cs` 单调不减），这一帧就整行清零。`p < 0` 的行来自
传进来的 `state`，只在整块没有 reset 时才有效。**reset 位置本身保留**——它是新段
的第一帧，不是上一段的残留。

**判据**：`tests/test_ssm_streaming.py` 新增 `TestResetAcrossChunks` 5 条：块尾 reset
（审计反例，期望 3）、块首 reset、块短于窗口 K=4、批内逐行互不影响、模型级
"分块续接 == 整段"且带 reset。手工探针确认 head state 为 `[0, 1]`，分块与整段都
得 3.0。

**边界**：`kernel == 1` 时不生成掩码（没有历史帧可清）；`steps == 1` 的
`_reset_prefix` 直接返回自身，省掉 decode 热路径上的一个递归节点。清零是**主动切断**
跨段状态，调用方若要把一批当作已有会话的续段，必须清掉第 0 列 reset（见 5.179）。

### 5.179　左填充隔离：pad 不得推进状态，也不得进入真实 token 的卷积窗口

**动机**：5.120 的 `lm_batch` 已经做了 left padding 与 label mask，但 `ids` 里的
`pad_id=0` 仍然作为普通 token 喂进网络：它会推进 Mamba 状态、进入后续真实 token 的
卷积窗口，也会通过 embedding 行影响梯度。mask 只遮损失，遮不住计算。

**做法**：`lm_batch` 改为返回**四个**数组，新增 `reset (B, T)`（每行首个真实 token
位置为 1）；`build_language_model` / `build_language_model_stateful` 增加
`with_reset=True`，图多一张 `reset` 输入并透传给整栈的卷积与 SSM；
`generate` 的 prefill 喂 `reset[0,0]=1`、逐 token decode 喂全 0。这样 pad 位置被
5.178 的两条路径同时隔离：卷积窗口不跨过首个真实 token，SSM 在起点重置。它顺带
覆盖了"块尾 reset 的跨块泄漏"——那是同一张掩码在分块续接时的另一半语义。

**判据**：`TestLeftPaddingIsolation` 三条——带 pad 的一批与逐条单独跑结果逐位一致；
把 pad 位置的 embedding 行改成任意值，真实位置的 logits / 损失 / 梯度都不变；两条
入口（非流式与流式）都含 `reset` 输入。`test_reset_marks_the_first_real_token_of_each_row`
钉住起点位置。两个"改 pad embedding"用例必须用 `tie_weights=False`：绑定时 embed
同时是词表头，改 pad 行必然改 logits 第 0 列，那是参数共享而不是填充泄漏。

**边界**：reset 是逐行的，同 batch 里一个样本重开一段不影响其它样本；`lm_batch`
显式拒绝"序列里出现 `pad_id`"，padding 位与真实 token 必须可区分。

### 5.180　训练状态存档：未提交梯度、优化器超参、scaler 与 RNG 约定

**动机**：5.27 的存档只覆盖参数与优化器槽位。用了梯度累积之后，在窗口未满时存档
会丢掉已累积的那几批梯度，续训轨迹立刻分叉；优化器超参（lr / momentum / betas）
改了也照旧加载，得到的是一条"看似续训、实则换了优化器"的轨迹；`--seed` 与
`rng_rank` 不同则 dropout 掩码不同，同样无法逐位复现。

**做法**：

1. `GradientAccumulator` 新增 `_weight` 与 `add(grads, weight=1.0)`：首个微批乘
   weight、后续 `+= arr * weight`，`flush()` 除以权重总和；`state_dict()` 返回
   `{"steps", "count", "weight", "sum"}`，`load_state_dict` 校验窗口长度、权重、
   键集合与形状后放回各自的数组模块。
2. `Trainer.snapshot()` 的 `aux` 写入 `grad/<参数名>`（未提交的累加和），meta 增加
   `accum_steps` / `accum_pending` / `rng_seed` / `rng_rank`。
3. `Trainer.load_training_state(path, *, state=None)`：拒绝"设备态已建立"的实例；
   用 `load_checkpoint(..., extra=template)` 先校验后原地写；校验 `accum_steps`
   与 RNG 约定；恢复 `steps`；窗口未满则恢复累加和，已满则丢弃（下一次 step 会
   立刻提交）；scaler 状态缺失或不该有时都报错；最后 `_refresh_resident()` 并清
   `_host_stale`。
4. 优化器新增 `hyperparams()`，`state_dict()` 带 `hyper`；`load_state_dict` 逐项
   比较，不一致抛 ValueError 并指明是超参。

**判据**：`TestWeightedAccumulation` 三条（加权梯度等于解析加权均值、默认等权、
非法权重拒绝），以及 4 条续训用例：待提交窗口恢复后与"从未中断"逐位一致；满窗
存档不需要待提交梯度；超参被绑定；窗口长度与 RNG 约定不匹配时明确报错。

**边界**：权重只接受有限正数，它改变的是窗口内的比例，不改变优化器内部的归一；
存档绑定的是"约定"（窗口长度、超参、RNG 种子与秩），不是完整的优化器实现，换
优化器类仍需调用方自己判断；RNG 部分只记录 seed / rank，不序列化 numpy 生成器的
内部状态——可复现的前提是每步都由 `(seed, step, rank)` 派生 key（5.125）。

### 5.181　入口 dtype 规则：浮点→整数必须显式，窄化先验范围，布尔不是下标

**动机**：执行器遇到输入 dtype 与图声明不一致时直接 `astype`。numpy 的转换是静默的：
`0.7` 截成 `0`、`300` 窄化成 `44`（i8）、`True` 当成 `1`。这些都不是"类型协商"，
而是数据损坏，而且设备侧与参考侧会得到同样的错误结果，没有任何一处会喊。

**做法**：`runtime/executor.py` 新增模块级 `_check_input_dtype`，只在
`arr.dtype != want` 时调用一次：

- 目标为整数：拒绝布尔输入（布尔不是下标），拒绝浮点输入（必须由调用方显式
  `astype`，让截断发生在用户看得见的地方）；
- 目标为整数且源整数的字节数更大：先比较 min/max 是否落在目标范围内，越界报
  `ValueError` 并给出实际区间；
- 目标为 bool：拒绝数值输入。

**判据**：`tests/test_ir.py` 新增 5 条：浮点→整数 `TypeError`、窄化越界
`ValueError` 与范围内可窄化、布尔不能当整数下标、整数→浮点无损放行、embedding
越界在参考侧与设备侧的报错口径一致。

**边界**：只作用于**图输入边界**；图内 `cast` 仍按算子语义执行，因为那是构图者
显式写的。参考执行器与设备执行器共用同一个 `Executor`，规则不会分叉。

### 5.182　语言模型的损失与初始化全程 f32

**动机**：`masked_lm_loss` 原来跟随输入 dtype。f16 下 `1e-12` 下溢成 0，一个全
padding 的批（分母为 0）变成 `0/0 = NaN`，梯度也非有限；而损失的精度本来就不该
跟着算子走。同一轮还发现 `LMSpec.init_params` 的 `embed` 用了默认 dtype，混精下
参数表会先落成 f16，再被优化器当成主副本。

**做法**：`masked_lm_loss` 开头把 logits / targets / mask 统一 `g.cast(..., f32)`，
其余按 f32 计算；`LMSpec.init_params` 的 `embed` 显式 `.astype(np.float32)`。

**判据**：`TestMaskedLoss` 增加 f16 全空批：损失是有限的 0、梯度全部有限；
`TestLanguageModelForward` 断言 `init_params` 的每个数组都是 f32，且两个构图入口
（非流式 / 流式）在 `with_reset=True` 时都含 `reset` 输入。

**边界**：f16 前向因此多两次 cast（反向同理），只在图上多两个节点；参数保持 f32
意味着 5.126 的 master weights 仍是"驻留 f16 + 主副本 f32"两套缓冲，省的是带宽
与显存，不是把参数永久降精度。

### 5.183　softplus 的零点导数与 decay_diagnostics 的注入上界

**动机**：5.94 已把 softplus 从 `log(1+exp(x))` 改成
`max(x,0) + log(1+exp(-|x|))` 修掉溢出，但 `max` 是用 `relu` 拼的，而 `relu` 在
0 处的子梯度取 0，整条链在 `x=0` 的导数因此变成 0（解析值是 `sigmoid(0)=0.5`）。
另一边，5.114 的 `decay_diagnostics` 把注入写成 `dt * x_act * B_w`，把
`(d_inner, N)` 当成 `(…, N)` 广播，实际上漏掉了 `d_inner` 上的求和，上界偏小多少
取决于 `d_inner` 与 `B_w` 的取值；并且先过滤非有限值再取
`q = max(a)`，NaN 被丢掉后 `q < 1` 仍然成立，于是"整条递归都收缩"这个结论建在
被丢弃的单元上。

**做法**：

- `_softplus` 的前半段写成 `(x + |x|) / 2`，`|x| = relu(x) + relu(-x)`：在
  `x != 0` 处逐点相等，在 0 处导数是 `(1 + 0) / 2 = 0.5`。不新增算子。
- 注入上界改为先算 `bt = x_act @ B_w`（沿 `d_inner` 求和），再
  `b = (dt * x_act)[..., None] * expand_dims(bt, -2)`。
- `contractive` 要求 `bad_a == 0 and bad_dt == 0 and isfinite(q) and q < 1`；
  非有限时 `reason` 明确写"存在非有限单元"，`state_bound` 保持 `None`。

**判据**：`tests/test_models_extra.py` 三条：softplus 在 0 处导数为 0.5；
`d_inner=2` 的注入上界等于 `log(2) * 2 ≈ 1.3863`（逐单元乘会漏掉求和）；
含 NaN 的单元不得报 `contractive`。

**边界**：这是诊断而不是证明——`q < 1` 只约束递归本身，残差、门控与投影的放大
仍不在上界里；`f16` 的 `a` 舍入成 1 的单元数继续单独报告。

### 5.184　seal 之后常量载荷不可变

**动机**：5.106 的 `Graph.seal()` 把 `attrs` 换成只读映射，但 `constant` 的 `data`
如果是 list，仍然可以被原地追加或改写。`revision` 是执行计划与捕获计划的缓存键，
"图已变、键未变"正是 5.106 想消除的失效形态。

**做法**：新增 `_freeze_payload`，递归把 list / tuple 转成 tuple；`Graph.constant`
写入时 `attrs["data"] = _freeze_payload(data)`。`_freeze_attrs` 的说明改为
"data 在写入时已冻结"，不再依赖 seal 时补做。

**判据**：`test_seal_freezes_constant_payloads` 断言 seal 后对原 list 的修改不影响
节点属性，且 `attrs` 的嵌套值不可变。

**边界**：只冻结容器结构，不冻结 numpy 数组的内容（数组仍按"调用方不写"的约定
共享，避免每次构图都深拷贝）；这条与 `constant` 已有的"产物只读"约定一致。

### 5.185　按 token 加权的梯度累积

**动机**：变长批的微批大小不同，等权平均会让"短句多的批"与"长句多的批"贡献相同，
而损失本身是按 token 数归一的。训练器因此无法表达"这一批有 3 个 token、那一批有
30 个"。

**做法**：`GradientAccumulator.add(grads, weight=1.0)` 与 `Trainer.step(batch,
weight=...)` 全部透传；窗口内按权重和归一，默认等权时逐位回到旧行为；
`state_dict` 带上 `weight`，续训恢复后比例不变。

**判据**：`TestWeightedAccumulation`：两批权重 1 / 3 的累积梯度等于解析加权均值；
不传权重时等于等权平均；权重为 0、负数、非有限时 `ValueError`。

**边界**：传 weight 时不建累加器（`accumulate_steps == 1`）也可以，此时 weight
没有意义；权重只作用于累积窗口，不改变优化器内部对梯度的任何归一。

### 5.186　受约束解码的根形态、生成边界与 HIP 探测

**动机**：四处边界各自都能静默给出错误结果。`JsonGrammar` 允许根是标量，工具
调用却要求一个对象；`Tool.check` 对 `float` 只查 `isinstance`，`inf` / `nan`
因此能通过签名校验；`max_new_tokens=0` 是"只 prefill"的合法请求，旧实现仍会
先采一个 token 再判断；`sample_next_token` 在极小温度下先除后减，`inf - inf`
得到 NaN；HIP 的 `available()` 只探测库能否 `CDLL`，WSL 里缺
`HSA_ENABLE_DXG_DETECTION=1` 时库照样加载、设备数却是 0。

**做法**：

- `JsonGrammar` 增加 `require_object`，栈空时只允许 `{`；`ToolRegistry.grammar()`
  返回 `JsonGrammar(require_object=True)`。
- `Tool.check` 对 `float` 参数额外要求 `math.isfinite`。
- `generate` 在 `limit == 0` 时直接返回 `[]`（受约束且文法未完成则报错）。
- `sample_next_token` 先 `shifted = z - max(z)` 再除以温度；温度非有限时
  `ValueError`。
- `hip.runtime.available()` 改为"库可加载 **且** `hipGetDeviceCount` 返回 0
  且设备数 >= 1"。

**判据**：`test_lm_tools` 两条（非有限 float 被拒、registry 文法不能完成非对象）、
`test_lm` 两条（极小温度仍是确定性的 one-hot、非有限温度被拒）、`test_generate`
一条（`max_new_tokens=0` 返回空）、`test_hip_backend` 的
`TestHipAvailabilityProbe` 4 条（有库无设备 / 枚举失败 / 有设备 / 缺库）。

**边界**：`require_object` 只是文法层面的根约束，不检查字段名与嵌套深度；
HIP 探测仍然只回答"能不能用"，不回答"用哪张卡"（那是 5.164 的多设备面）。

**验证**：本轮新增 37 个用例（1786 → 1823）。三处全量见 §0 速览的实测行。

### 5.187　语言模型的推理路径：固定块预填充、末位置投影与权重常驻

**动机**：`generate` 此前是"整段一次 prefill → 逐 token decode"，四个成本都随
上下文长度线性增长：整段图只能跑一次，prompt 超过 `spec.steps` 直接拒绝；每次
prefill 都物化 `(B,T,V)` logits，32K 上下文、V=32K 时那是 4 GiB；权重每步都由
调用方重新喂进 `execute`，设备后端每步重做一次 H2D；取 logits 行时又直接对设备
数组做 numpy 索引——cupy 禁止隐式转换，CUDA 上这条路径其实跑不通。

**做法**：

- `_head(..., last_only=True)` 只对最后位置做 RMSNorm 与词表投影，返回
  `(B,1,V)`。RMSNorm 逐位置独立，切片后算与整段算逐个相等，不是近似。
- `build_language_model_stateful(..., head=)` 三档：`"all"` 全位置、`"last"`
  末位置、`"none"` 不建词表头（只带状态）。非法值在构图期拒绝。
- `generate(..., chunk_size=, resident=)`：`chunk_size` 给定时按固定块切
  prompt，中间块用 `head="none"`、最后一块用 `head="last"`，同一 `(长度,
  head)` 的图只建一次并在块间复用（固定块的全部意义就在这）；尾块按**实际
  长度**构图，不用 pad 填满——pad 会推进状态，而 reset 又会连带清掉真实历史。
- `resident=True` 时把权重一次性 `resident_parameters` 到设备，要求后端驻留表
  为空、结束时 `release_parameters`；不会悄悄覆盖调用方已有的驻留权重。
- `_host_row` 统一走 `backend.to_host`，每步只取回 `V` 个 float。

**判据**：`test_lm` 新增 10 条——末位置投影与整段投影的最后一行一致（1e-6），
`head="none"` 的状态与 `head="all"` 一致且不产出 logits，非法 `head` 被拒；
分块（2/3/4、整除与非整除）与整段生成**完全相等**；prompt 超过 `steps` 且未给
`chunk_size`、`chunk_size` 为 0 / 负 / 超过单块上限都显式报错；驻留路径与主机
feed 结果一致、结束后驻留表为空、每步 feed 不含权重；非空驻留表与没有
`resident_parameters` 的后端都拒绝。

**基准**（`benchmarks/lm_inference.py`，chunk=256、d_model=64、2 层、
V=1024、贪心 8 token）：

| 机器 | prompt | 块数 | TTFT | prefill tok/s | decode ms/token | decode tok/s | 峰值（prefill / 全程） |
|---|---|---|---|---|---|---|---|
| Tesla M40 | 1K | 4 | 289.2 ms | 3,540 | 54.52 | 18.3 | 7.86 MB / 14.59 MB |
| Tesla M40 | 8K | 32 | 446.2 ms | 18,361 | 53.62 | 18.7 | 7.86 MB / 14.59 MB |
| Tesla M40 | 32K | 128 | 1,091.0 ms | 30,036 | 43.51 | 23.0 | 22.45 MB / 29.18 MB |
| RX 7650 GRE | 1K | 4 | 198.5 ms | 5,159 | 12.78 | 78.3 | 1.63 MB / 1.63 MB |

传输口径（设备常驻）：每 decode 步 H2D 8 B（ids + reset）、D2H 4 KB（一行
logits），**状态跨步 0 B**（设备→设备，不经过主机）；若不驻留，每步还要多传
517,888 B 权重。TTFT 的固定成本（构图 + `graph_mode="auto"` 的首次捕获）约
200～250 ms，所以 1K 的 tok/s 反而最低；8K/32K 的边际预填充吞吐是 18K～30K
token/s。这个数字**不是**跨请求稳态：每次 `generate` 都新建图，跨请求复用图与
常驻会话还没做。

**边界**：batch 仍为 1；尾块长度可变，因此图缓存最多三张（中间块一张、整块末尾
一张、尾块一张）；`head="none"` 仍声明 `final_norm` 输入但不用，是为了三张图共用
同一份驻留表；32K 只在 M40 上跑，HIP 只抽测 1K，阈值与数字都只代表这两台机器。

**验证**：本轮新增 10 个用例（1823 → 1833）。本地 1833（398 跳过）、WSL HIP
1833（354 跳过）、CUDA 真机 1823（104 跳过）全绿；基准在三处各跑通一轮。

**5.189 之后的真机补跑**（2026-11-13，3×Tesla M40，`CUDA_VISIBLE_DEVICES=2`
独占）：全量 1851 个用例 / 104 跳过全绿，基准三档复跑如下。

| prompt | 块数 | TTFT | prefill tok/s | decode ms/token | decode tok/s | 峰值（prefill / 全程） |
|---|---|---|---|---|---|---|
| 1K | 4 | 142.9 ms | 7,165 | 5.83 | 171.6 | 7.86 MB / 14.59 MB |
| 8K | 32 | 262.8 ms | 31,167 | 4.87 | 205.4 | 22.45 MB / 29.18 MB |
| 32K | 128 | 663.1 ms | 49,413 | 3.67 | 272.2 | 37.04 MB / 43.77 MB |

这一轮比上表快得多（32K 的 decode 从 43.51 降到 3.67 ms/token，TTFT 从 1,091
降到 663 ms），而代码与 spec 完全相同——两轮之间这台机器上还跑着别的负载（上一
轮没限定卡位）。上表保留上一轮的原始记录不覆盖：两列数字都是"某一时刻的整机
状态"，不是这台卡的稳态；能跨轮比较的只有同轮内的相对关系（decode 耗时随上下文
基本不涨，正是"状态不物化历史"的直接体现）。

### 5.188　语言模型的整数标签损失

**动机**：`train/lm.py` 的 `targets` 是 one-hot，与 logits 同形，`B*T*V` 个
float32。V=32K、B=8、T=1K 时是 1 GiB 的主机侧标签，每步都要重新上传，光传输
就压过了计算；而这 1 GiB 里只有 `B*T` 个 1 有用。整数标签是同一损失的另一种
编码，尺寸小 V 倍。

**做法**：
- `lm_batch(..., labels="onehot"|"index")`。默认仍是 one-hot（老调用逐位不变）；
  `index` 模式返回 `(B,T)` int32 的**扁平下标** `(b*T + t)*V + y`，无标签位填 0
  ——它一定是合法下标，且被 mask 遮住。`B*T*V` 超过 int32 上限时在**分配之前**
  报错，因为下标回绕会静默指到别的 `(b,t)`，比 one-hot 占内存危险得多。
- `masked_lm_loss_indexed(g, logits, targets, mask)`：把 `(B,T,V)` 的 logits
  拉直成一维"表"，再交给 `embedding` 的查表语义取出目标位的 logit。`embedding`
  本身就是"按整数下标沿第一维取值"，拉直之后正好是"沿最后一维取一格"；反向
  直接是 `embedding_backward` 的散射累加。**没有新增算子**——这条是选它而不是
  新写 `gather_last` 的理由：新算子要同时落下参考语义、VJP、CUDA/HIP 内核、
  conformance 与序列化五处，而这里已经有同构的现成算子。
- `targets` 保持 `(B,T)` 参与查表，**不 reshape**。设备后端的 `reshape` 只承接
  f16/f32（5.42 起逐元素核只认这两种），HIP 真机上第一次跑就撞上
  `reshape 不支持 int32`；`embedding` 的输出形状本来就是 `ids.shape + tail`，
  省掉这一步反而是正解。
- 下标在主机侧算。图内没有可用的整数偏移运算（同上，逐元素核只认 f16/f32），
  靠 cast 到 f32 再算会在 `B*T*V > 2^24` 时丢精度——而那个规模正是整数标签
  唯一有意义的规模。
- 归一化抽成 `_masked_mean` 供两条路径共用；`logp` 的运算顺序刻意与 one-hot
  路径相同（`(logit - max) - log(sum exp)`），前向因此**逐位相等**。

**判据**：`tests/test_lm.py` 新增 8 条——整数下标能解回 one-hot 的 token 与
`(b,t)`、两条路径的损失逐位相等、梯度在 1 ulp 内一致（one-hot 侧多一次
`(B,T,V)` 的广播乘与归约，求和顺序不同）、只改无标签位的下标不改变损失、
全 mask 的一批是有限值 0、`targets` 形状不符 / dtype 是浮点 / 未知 `labels` /
int32 溢出四处显式报错。

**收益与代价**：标签内存与 H2D 从 `4*B*T*V` 降到 `4*B*T` 字节（V 倍）；前向
少两个 `(B,T,V)` 级别的节点（`mul` 与沿最后一维的 `reduce_sum`），也不需要
整份 `logp` 矩阵。反向仍要一份 `(B,T,V)` 的 `dlogits`——那是 softmax 的梯度，
两条路径都省不掉，`embedding_backward` 的内部缓冲与 one-hot 路径的梯度同大。

**边界**：整数标签要求 logits 形状全静态（下标要靠 `B*T*V` 与 `V`）；只接受
i32 / i8，浮点 `targets` 在构图期拒绝；两条路径的**前向**逐位相同、反向差
1 ulp，不承诺跨后端逐位一致。CUDA 侧与 HIP 共用同一份 `embedding` 内核与
`reshape` 的 dtype 检查，两家真机都已对拍（见验证）。

**验证**：本轮新增 8 个用例（1833 → 1841）。本地 1841（398 跳过）、WSL HIP
1841（354 跳过）全绿。HIP 真机（RX 7650 GRE / ROCm 7.2）另跑一次两条路径的
前向 / 反向对拍：损失逐位相等、`dlogits` 最大差 1.49e-8。

**CUDA 真机补跑**（2026-11-13，3×Tesla M40，`CUDA_VISIBLE_DEVICES=2`）：
`tests.test_lm` + `tests.test_lm_tools` 94 条全过，全量 1851 / 104 跳过全绿；
另用独立脚本（V=64、B=2、T=6）在同一张卡上对拍两条路径——损失
`9.991523742675781` 两边逐位相同，`dlogits` 最大绝对差 `1.49e-8`，与 HIP 的
数字一致。`masked_lm_loss_indexed` 里"不 reshape int32 的 targets"这条正是被
HIP 撞出来、CUDA 上同样成立的约束。


### 5.189　工具的 JSON Schema 约束与按注册表合成的受约束文法

**动机**：5.122 的 `JsonGrammar` 只保证"是合法 JSON 对象"。模型仍能写出未注册
的工具名（`"name":"sub"`）、多余的参数键（`{"c":1}`）与类型不符的值
（`{"a":"x"}`）；`Tool` 的参数也只有 `{参数名: Python 类型}` 这一种便捷写法，
表达不了数组、枚举与"可选缺省"。成熟框架在这一层的做法是**把工具的 JSON
Schema 直接当解码约束**，而不是先放开再校验。

**做法**：

- `Tool` 支持两种写法，二选一：`required` / `optional`（便捷，只能表达标量）
  或 `parameters`（JSON Schema 子集：`type` / `enum` / `required` /
  `properties` / `additionalProperties` / `items`）。同时给两种直接拒绝——
  "哪个生效"不该靠猜。`schema()` 把便捷写法翻成等价 schema，输出标准
  `{"name","description","parameters"}`；`ToolRegistry.schema()` 按名字排序，
  可直接作为 function calling 的 `tools` 数组。
- **构造期校验**：根类型必须是 `object`；`required` 里每一项都必须在
  `properties` 里声明；每个 property 过一遍 `_schema_kind`。受约束文法只支持
  标量与"标量元素数组"，嵌套对象、数组元素为对象都在构造期显式拒绝，并提示
  改用 `JsonGrammar`（它接受任意合法 JSON，再由 `Tool.check` 兜底）——不是
  做不了，而是要把 schema 递归展开成动态段，收益远小于复杂度。
- **运行期校验**（`Tool.check` → `_check_against_schema`）：`enum` 先于
  `type` 判定（类型对但取值不在枚举里同样拒绝）；`bool` 不算 `integer`；
  `number` 拒绝非有限值（`1e999` 是合法 JSON，解析成 `inf`，进工具就是 NaN
  扩散的起点）；`additionalProperties` 默认 `False`，多出来的键一律拒绝。
  便捷写法的旧行为逐字不变。
- **`ToolCallGrammar`**：按注册表合成，把五个自由度同时限死——
  1. 工具名：维护"已匹配前缀 → 候选名字集合"，每一步只允许候选名字的下一个
     字符或闭合引号，没注册的名字写不出来；
  2. 参数键名：键名余下的字节由文法器补，模型只输出开引号；
  3. 参数顺序：固定为"先必填、后可选"，状态机因此不需要回溯，"省略可选"
     只剩一个出口（直接写 `}`）；
  4. 值的类型：`integer` 只走 `start → minus → zero/int`，小数点和指数一律
     视为"数字到此为止"、由分隔符处理，`1.5` 写不出来；
  5. required / optional：`_after_allowed(index)` 只给 optional 开 `}`
     这个出口。
- **数字状态机抽成纯函数** `_advance_number(stage, ch, *, integer_only=False)`，
  `JsonGrammar` 与 `ToolCallGrammar` 共用。复制一份会立刻漂移——两处对
  `1e` / `1.` 的接受面必须永远一致。

**判据**：`tests/test_lm_tools.py` 新增 20 条。`TestToolSchema` 9 条：便捷
写法转标准 schema、显式 `parameters` 原样透传、混用报错、`required` 引用未
声明属性报错、嵌套对象构造期拒绝、数组元素为对象拒绝、无映射的 `list` 报错、
`enum` 与数组在 `check` 里生效、`registry.schema()` 排序完整。`TestToolCallGrammar`
11 条：六种合法调用逐字节往返（`add` / `echo` / `pick` 空与非空数组 /
`toggle` 省略与给出 optional）、未注册名字不可写、多余参数键不可写、`integer`
拒绝小数点、required 不能跳过、optional 可以跳过、无参数工具、提前 EOS 被拒、
PAD 被拒、空注册表构造报错、40 次随机游走仍能 `parse_tool_call` + 调用成功。
每条合法用例都逐字节断言在 `allowed()` 内。

**验证**：本轮新增 20 个用例（1841 → 1861）。本地 1861（398 跳过）、WSL HIP
1861（354 跳过）、CUDA 真机 1851（104 跳过）三处全绿。本项全在主机侧（文法与
schema 校验都不碰图、不产生内核），三处跑的是同一份纯 Python 状态机；真机那一轮
同时把 `tests.test_lm` + `tests.test_lm_tools` 单独跑了一遍（94 条）。

**边界**：受约束文法只支持"顶层必填在前 + 可选在后"的固定顺序，想跳过中间
的某个 optional 必须先把它写出来；数组元素只能是标量，嵌套对象走 `JsonGrammar`；
`enum` 只校验取值、不参与解码约束（文法只认类型）；工具名有公共前缀时（如
`add` / `add_all`）靠逐字符枚举区分，名字很长很多时每步的候选集合会变大，
这是主机侧的字典查询开销，不进图。

### 5.190　多卡最小闭环：跨卡拷贝与数据并行

**动机**：5.164 给的是枚举、查询与切换，5.169 的真机验证明确把"跨卡拷贝
（peer-to-peer）与多卡并行执行"排除在范围外，只留下"跨卡复用由调用方决定"。
于是 3 × Tesla M40 上能切卡、能逐卡跑图，却不能把一份参数放到两块卡上训练。
成熟框架在这一层至少要做两件事：数组能在卡之间搬，和在多个卡上跑同一步。

**做法**（`aetherflow/train/parallel.py`，不新增算子、不新增执行机制）：

- **`copy_across_devices(backend, value, dst_device, *, src_device=None)`**：
  唯一的跨卡入口，经主机中转（D2H → H2D），因此不要求两块卡可互访、对
  CUDA 与 HIP 语义一致。`src_device` 缺省取 `current_device()`；显式给出时
  只在需要时才切卡。显式参数的理由是 HIP 的 `DeviceArray` 不记录自己的设备，
  D2H 依赖当前上下文。默认**不**做隐式 peer-to-peer：它要求两块卡可互访，
  失败后的静默回退会掩盖"哪一步降级了"。
- **`allreduce(values, reduction="mean")` / `allreduce_grads(per_device)`**：
  主机侧求和或平均。只接受浮点、要求形状与 dtype 完全一致——两张卡的梯度
  精度不同本身就是配置错误，静默升成 f64 只会让问题更难定位。梯度名集合
  必须逐卡相同，用交集掩盖"某块卡少跑一路反向"是不可接受的。
- **`DataParallelTrainer(graph, store, optimizer, backend=..., devices=...)`**：
  每步的执行顺序固定为"逐卡切卡 → 上传本卡数据切片 → 前向 → 反向 → 梯度
  回主机 → 主机侧 allreduce → `optimizer.step`（更新主机参数）→ 把新参数
  整份写回每卡副本"。参数始终有一份主机副本，所以 `store` 在每步之后都是
  最新的，不需要 `sync_to_host()`。
- **两条显式前提**：图按**每卡** batch 构图（执行器只认静态形状，总 batch
  才是 `world_size` 倍）；图内损失对 batch 求均值（`mse` /
  `softmax_cross_entropy` 都是），否则"逐卡损失取平均"与"合并 batch"不等价。
  batch 不能被设备数整除、分片大小与图的静态维不符、数据输入多传少传、
  设备索引越界 / 重复 / 非 int，全部当场报错，不丢尾样本、不做隐式补齐。

**判据**：`tests/test_parallel.py` 新增 22 条。`TestAllreduce` 7 条（mean /
sum、单数组恒等、非法 reduction、空输入、形状与 dtype 不一致、整数数组、
梯度名分叉）；`TestCopyAcrossDevices` 4 条（值与设备还原、同卡无操作、
非法目标 / 源、缺 `DeviceManagerContract` 的后端）；`TestDataParallelTrainer`
9 条（两块"设备"两步与单卡合并 batch 的损失 / 参数逐位接近、只切指定设备、
batch 不整除、数据名多传少传、第 0 维不一致、设备列表非法、后端与
reduction 非法、非标量目标、图按总 batch 构图的显式拒绝）；
`TestCudaDataParallel` 2 条真机用例。CPU 侧用 numpy 上的伪多设备后端
（数组就是 numpy，"设备"只记录索引），因此切分 / 归约 / 写回逻辑在没有 GPU
的机器上也能逐位验证。

**验证**：本轮新增 22 个用例（1861 → 1883）。本地 1883（400 跳过）、WSL HIP
1883（356 跳过）全绿；CUDA 真机（3×Tesla M40，`CUDA_VISIBLE_DEVICES=2`）
1871（105 跳过）全绿，两块卡的对拍用例在 3 卡全可见时单独跑通：跨卡拷贝
逐位一致、数据并行两步的损失与参数与单卡合并 batch 在 `rtol=1e-4` 内一致。

**边界**：这是**功能闭环，不是性能并行**——同一进程内依次切卡执行，没有
并发重叠。`set_device` 是丢弃而不是迁移旧卡缓存（5.169），所以每切一次卡
旧卡的编译产物与捕获计划都作废，参数副本也是每步整份重传；小模型会被这笔
固定开销主导，它只适合"一步计算量远大于重新编译 / 上传"的规模。
`DataParallelTrainer` 不与 `Trainer` 的 `device_state` / 混合精度 / 分段重算
叠加：需要这些语义时先用 `Trainer` 把单卡跑稳。真正的并行（多进程 / 多流、
NCCL / RCCL 通信）与跨卡 peer-to-peer 仍未做。

### 5.191　语言模型的跨请求推理会话

**动机**：5.187 的固定块预填充把 `(块长, head)` 的图在**一次** `generate` 内
复用（32K / chunk=256 时 128 块共享 3 张图），但每次调用仍从头构图，并在
`graph_mode="auto"` 下重新捕获；TTFT 里约 200～250 ms 是这笔固定成本，导致
1K 的 tok/s 反而低于 8K/32K。真实服务里同一个模型会被反复请求，这笔成本
本应只付一次。5.187 的边界也明确写着"跨请求复用图与常驻会话还没做"。

**做法**（`aetherflow/models/lm.py`，不新增算子、不新增执行机制）：

- **抽出 `_decode`**：把原来的解码循环（预填充分块 → 尾块采样 → 逐 token
  推进 → 停止判定）整体搬进一个内部函数，`generate` 与 `LMRunner` 共用同一
  条数值路径。`generate` 的对外行为与校验顺序保持不变（`limit==0` 与
  `constraint` 的判定仍用 `if/elif`，先校验请求再判是否需要构图）。
- **`LMRunner(backend, params, spec=None, *, resident=False, max_graphs=8)`**：
  持有一个按 `(length, head)` 索引的图缓存，`OrderedDict` 记录插入序，命中时
  `move_to_end`，超限时淘汰最旧的一张。缓存键是"块长 + head"而不是块序号，
  因此不同 prompt 长度只要落在同一块长上就复用。
- **权重驻留**：`resident=True` 时在构建第一张图时调用
  `backend.resident_parameters(graph, params)`，此后每次 `execute` 的 feed
  不再携带权重；`close()` 只释放自己登记的那一份。构造期显式拒绝两种情况：
  后端没有驻留接口、调用方的驻留表已经非空（避免误释放别人的权重）。
- **`close()` 与上下文管理**：幂等，释放驻留、清空图缓存并置 `_closed`；
  之后任何 `generate` 都显式报 `ValueError`，不做静默重建。支持
  `with LMRunner(...) as runner:`。
- **构造期校验**：`spec.batch != 1`、`max_graphs` 非 int 或 `< 1` 都当场报错；
  `max_graphs` 不接受 `bool`（`isinstance(True, int)` 会误导）。

**判据**：`tests/test_lm.py` 新增 `TestLMRunner` 7 条：

- 与 `generate` 的 token 输出逐位一致，且重复请求不增长图数（prompt=5 /
  chunk=2 时缓存 `(2,"none")` 与 `(1,"last")` 两张，decode 复用尾块图；
  换成另一长度后增长到 3），`close()` 后 `graph_count == 0`；
- `max_graphs=2` 时图数始终不超过 2（LRU 淘汰）；
- 驻留跨请求存活、每次 feed 不含 `embed`，退出上下文后后端驻留表清空
  （用 `_ResidentRecorder` 记录每次 `execute` 的 feed）；
- `max_new_tokens=0` 返回空列表且一张图都不建；
- `close()` 幂等，之后 `generate` 报"关闭"；
- 非法配置在构造期报错（batch=2、`max_graphs=0`、`max_graphs=True`、
  后端无驻留接口、驻留表非空）；
- 非法请求在构图前就被拒绝（空 prompt、`chunk_size=0`）。

**验证**：本轮新增 7 个用例（1883 → 1890）。本地 1890（400 跳过）、WSL HIP
1890（356 跳过）全绿；CUDA 真机（3×Tesla M40，`CUDA_VISIBLE_DEVICES=2`）
1878（105 跳过）全绿。真机会话脚本（1K prompt / chunk=256 / 8 token，M40）：
`graphs=3`、`values_equal=True`、`resident_released=True`，
`t_generate_ms=387.34` 对 `t_session_repeat_ms=35.93`，加速 10.78×。

**边界**：会话本身只缓存"图 + 驻留权重"，不**自动**缓存 KV / SSM 状态——每次
`generate` 默认仍从零状态开始，多轮对话的续接要靠调用方显式 `prime` 前缀
（见 5.193），不是隐式增量会话。
decode 的批量入口见 5.192，但仍是串行执行，没有并发会话与请求队列。图与驻留
都绑定构造时的参数表：权重更新后必须新建会话（旧会话不会感知新值）。
`close()` 之外没有自动回收，长生命周期的 runner 需要调用方自己管理。

### 5.192　语言模型的批量解码

**动机**：5.187 / 5.191 的推理路径都只做 `batch=1`。真实服务里一次请求通常带
多条序列，逐条调用虽然正确，却把固定成本（一次预填充的块数 × 图派发）付了
多遍；而把多条序列直接拼成一批又会撞上两个语义问题——prompt 长度不同怎么对
齐、某一行的 EOS 之后其余行怎么办。这两条都不能靠"取最短"糊过去：最短序列
决定整批会让长 prompt 的后半段被截断。

**做法**（`aetherflow/models/lm.py`，不新增算子）：

- **`generate_batch`**：`prompts` 的条数必须等于 `spec.batch`（图按固定 batch
  构建），返回 ragged 的 `List[List[int]]`，第 `i` 条是第 `i` 行的生成部分
  （不含 prompt）。
- **左侧补 `pad_id`**：所有行对齐到最长 prompt 的宽度，每行首个真实 token 的
  位置置 `reset=1`。reset 之前的 pad 会推进状态，但 reset 点把状态清掉，因此
  行与行之间互不影响——这正是训练侧 `train.lm.lm_batch` 的左填充约定，不是
  近似。右侧补 pad 不行：pad 会继续推进 Mamba 状态，末状态对不上。
- **逐行采样与逐行停止**：第 `i` 行用 `seed + i` 派生独立生成器（贪心不消耗
  随机数），EOS 与 `max_new_tokens` 都按行判定。已结束的行在后续 decode 里
  喂 `pad_id` 并丢弃输出，其余行照常推进；全部结束即提前退出，不再派发。
- **共用 `_decode`**：把原来只认单条 prompt 的解码循环推广成 `prompts` 列表，
  `generate` 传 `[prompt]`、`generate_batch` 传全部；因此 `batch=1` 的批量
  入口与单条入口在同样 seed 下**逐 token 一致**（含 temperature / top-k），
  这条恒等式有用例钉住。
- **`chunk_size=None` 取最长那条**：批量里"整段一次算完"意味着块长等于
  `max(len(prompt))`；用第一条的长度会让短 prompt 决定所有行的切块方式。
- **`LMRunner` 按 batch 绑定**：构造期不再要求 `batch=1`；单条 `generate`
  在非 1 的会话上显式报错并指向 `generate_batch`。两种请求命中同一份
  `(块长, head)` 图缓存。
- **`constraint` 显式拒绝**：文法推进状态是逐序列的，共享一个实例会把两条
  序列的写法搅在一起；需要受约束解码时逐条调用。

**判据**：`tests/test_lm.py` 新增 8 条：

- 三条长度不同的 prompt（1 / 5 / 2，chunk=2）批量结果与逐条 `generate` 逐位
  相等——覆盖"整块都是填充"与"块从 prompt 中间开始"两种偏移；
- `batch=1` 的 `generate_batch` 与 `generate` 在 `temperature=0.8, top_k=5`
  下逐 token 一致（随机数消耗顺序不变）；
- 用脚本化后端让两行在不同步遇到 EOS：一行生成 3 个 token、另一行首 token
  即 EOS，验证互不牵连；
- 每行各自遵守 `max_new_tokens`（一行先到上限、另一行同一步 EOS）；
- `max_new_tokens=0` 返回 `[[], []]`；
- 批量 + `resident=True` 与不驻留结果相同、驻留表在结束后清空、每次 feed
  不含权重；
- 非法请求在构图前报错：空 `prompts`、条数不符、空的单条 prompt、批量带
  `constraint`、`chunk_size=0`；
- 批量会话首次结果与无状态入口相同、重复请求图数不增长（2 张），单条入口
  在 batch=2 的会话上报错并提示 `generate_batch`。

**验证**：本轮新增 8 个用例（1890 → 1898）。本地 1898（400 跳过）、WSL HIP
1898（356 跳过）、CUDA 真机（3×Tesla M40，`CUDA_VISIBLE_DEVICES=2`）1886
（105 跳过）全部通过。

**边界**：批量解码解决的是"一次请求内多条序列"，不是服务端的 continuous
batching——没有请求队列、没有跨请求的动态拼批，decode 仍逐步串行派发。
`spec.batch` 在构图期固定，换 batch 要新建 spec / 会话；同一批内
`temperature` / `top_k` 共享，不能逐行设置。左填充意味着短序列要陪跑最长
序列的块数，按长度排序分桶是调用方的事。本轮只验证了参考后端上的语义
（批量路径与单条逐位对拍），没有在 GPU 上标定批量吞吐。

### 5.193　语言模型的会话前缀缓存

**动机**：5.191 的会话把"图 + 驻留权重"移出了单次请求，但 **SSM 状态没有**：
每次 `generate` 仍从零状态开始，system prompt 与历史对话每请求重算一遍。
聊天场景里前缀（system + 历史）远长于本轮新增的 token，重复预填充是纯浪费。
把前缀状态留在会话里，需要三件事同时成立：前缀必须能被预先消费、后续请求只能
是它的**真前缀扩展**（否则缓存的状态是错的）、生成结束后缓存必须失效（decode
已经把状态推进过，再当成前缀状态使用就会算错）。这三条都不能靠"猜"来兜底。

**做法**（`aetherflow/models/lm.py`，不新增算子）：

- **`_decode` 拆成两段**：`_prefill(...)` 只负责把 prompt 推进到末位置并返回
  `(state, logits)`，`_sample(...)` 只负责逐步解码与停止判定。无状态入口、批量
  入口与会话入口因此共用同一条数值路径，`batch=1` 的恒等性不靠第二份实现维持。
- **`_prefill` 新增 `consumed` / `state`**：`consumed[row]` 是每行已消费长度，
  `remain` 必须全部为正，否则报"增量前缀至少要包含一个未消费的 token"；非
  首次调用时禁止左填充（pad 会推进状态、reset 会清历史，两者都与"续接"矛盾）。
  复位标记按**本次调用前是否已有状态**判定（循环前捕获 `fresh = state is None`），
  不能每块重判——首块之后的 `state` 已被赋值，会让后续块里每行首个真实 token
  的 `reset=1` 丢掉。
- **`prime(prompt, *, chunk_size=None) -> int`**：只支持 `batch=1`；新 prompt
  必须让已缓存前缀是它的真前缀（等长直接返回，更长只跑 `[cached:]` 这一段），
  返回前缀长度。结果 `(state, logits)` 与已消费长度存在会话上。
- **`_advance(prompt, chunk)`**：无前缀走全量 `_prefill`；有前缀时校验真前缀、
  等长直接复用缓存 `(state, logits)`、更长按 `consumed=[cached]` 走增量。
  `generate` 拿到 `(state, logits)` 后立刻 `_clear_prefix()` 再 `_sample`——
  生成已经推进了状态，旧前缀不再成立。
- **`reset()` / `_clear_prefix()`**：`reset()` 只清前缀与缓存状态，**保留**图缓存
  与驻留权重；`close()` 顺带清前缀。`max_new_tokens=0` 提前返回、不动缓存。
- **批量不做前缀续接**：有前缀时 `generate_batch` 直接报"请先 `reset()`"。
  批量里每行状态各自推进，共享一份前缀状态没有意义。

**判据**：`tests/test_lm.py` 新增 8 条（`_CountingBackend` 记录每次 `execute` 的
`ids.size`，用来区分"走了全量"与"只走了增量"）：

- 先 `prime` 再 `generate` 与从头 `generate` 逐位一致（贪心）；`temperature=0.5`
  / `top_k=5` 下同样一致——缓存的是状态而不是采样结果；
- 在**已 prime 的同一 prompt** 上 `generate` 只发生 1 次 decode（`head="last"`
  那一块），说明等长请求真的复用了缓存 logits；
- `prime` 只喂前缀长度、随后增量只喂新增的 1 个 token；`reset()` 前后图数不变
  （图缓存没有被前缀清理连累）；
- 变短或分叉的 prompt 报"前缀"错，`reset()` 后恢复；
- `max_new_tokens=0` 不消费也不清空缓存；
- `batch=2` 的会话上 `prime` 报 "batch=1"，有前缀时 `generate_batch` 报
  "reset"；
- `prime` / `reset` 在 `close()` 之后照常可调用（幂等、不抛）。

**验证**：本轮新增 8 个用例（1898 → 1906）。本地 1906（400 跳过）、WSL HIP
1906（356 跳过）、CUDA 真机（3×Tesla M40，`CUDA_VISIBLE_DEVICES=2`）1894
（105 跳过）全部通过。真机数值（1K prompt 切成 768 + 256，chunk=256）：全量
预填充 656.287 ms，`prime` 之后只跑 256 token 的增量 5.691 ms。

**边界**：参考后端（float64）上与全量预填充**逐位一致**，真机只在短序列上
逐位相等（64 token 的 48 + 16 切分）；32 token 的 24 + 8 切分 logits 已有
7.05e-4 的差，随机初始化的大模型上放大到 1.9。原因不在缓存逻辑，而在"分块
边界改变内核形状"本身（`graph_mode="off"` 下 8+8 与 16 的 logits 差 1.2e-2，
`auto` 下 8e-5）叠加模型自身的数值不稳定——Mamba 随机初始化下 256 token 的
全量预填充就会让 `l0_h0` 全 nan，参考后端 float64 则一致。因此长上下文的
"前缀 + 增量"只能承诺与"重新预填充"同量级，不承诺逐位相等；这条与 5.187 记的
"分块边界带来的舍入差"是同一个来源。其余边界：`batch=1` 专属，批量路径不做
前缀续接；生成后前缀自动失效，连续多轮必须重新 `prime`（当前实现是"一次
prime 一次 generate"）；权重更新需重建会话；驻留与前缀都占显存，且没有对
"前缀被重用多少次"做任何统计。

### 5.194　语言模型的采样面：top-p、重复惩罚与束搜索

**动机**：5.119 的采样只有 greedy / temperature / top-k，5.191～5.193 把会话、
批量与前缀缓存都做完了，**选路规则本身**却仍是三个里最小的子集。生产推理的
默认组合是"top-p + 重复惩罚"，而需要"比贪心好、但可复现"的场景要束搜索。
三者都是主机侧逻辑（不新增算子、不改图），但顺序与边界会直接决定分布对不对。

**做法**（`aetherflow/models/lm.py`）：

- **处理顺序固定**：重复惩罚 → 贪心判定（`temperature <= 0`）→ top-k → 温度 →
  归一化 → top-p。先截断再除温度会把被截掉的项重新放大成候选；重复惩罚必须
  在贪心判定之前，否则贪心解码里它根本不生效。
- **`_apply_repetition_penalty`**：CTRL 口径——已出现 token 的正 logit 除以惩罚、
  负 logit 乘惩罚（两种写法都让绝对概率下降），重复 id 只罚一次，越界 id 显式
  报错。`penalty == 1.0` 是严格无操作（不复制数组）。`penalty != 1` 而没给
  `recent` 直接拒绝：没有历史就没有惩罚对象，静默忽略会让调用方以为生效了。
- **`sample_next_token` 的 top-p**：按概率降序累计，保留"自身之前的累计质量
  仍小于 `top_p`"的 token——首次越过阈值的那一个也在内，所以 `top_p` 极小时
  不会把唯一候选切掉；截断后重新归一化，若一个都不剩则报错。top-p 在
  `temperature <= 0` 时不参与（贪心按定义就是取最大项）。
- **历史按行维护**：`_sample` 多收一个 `prompts`，每行历史 = 自己的 prompt +
  自己生成的 token；左填充的 pad 不进历史（它从未被生成）。`generate` /
  `generate_batch` / `LMRunner` 三个入口把 `top_p` / `repetition_penalty`
  透传进来，行与行之间不共享。
- **`generate_beam` / `LMRunner.generate_beam`**：把 `num_beams` 放进**图的
  batch 维**（`spec.with_batch(k)`），每步对 `(k, V)` 做 log-softmax 加前缀分数、
  取全局前 `2k` 个候选（EOS 会进完成池但不占活动槽位，所以要取多一倍），
  活动槽位按父假设下标用主机侧 gather 重排状态，再作为下一步的 feed 上传。
  `length_penalty` 按 `score / len**p` 归一化；分数比较用稳定排序，完全并列
  时取先生成的那条，结果是确定的。`num_beams=1` 退化成贪心，与
  `generate(temperature=0)` 逐位一致。EOS 不进返回值。
- **会话语义的显式拒绝**：`spec.batch != 1` 的会话、已有缓存前缀的会话都
  直接报错（k 条假设会把状态推进到不同末尾，前缀不再成立）；束搜索也不接
  `constraint`——需要约束时用 `generate`。`LMRunner` 的图缓存键从
  `(长度, head)` 变成 `(batch, 长度, head)`，单条与束搜索的图互不覆盖，
  驻留权重仍只有一份。

**判据**：`tests/test_lm.py` 新增 18 条：

- 采样：top-p=0.7 的 256 次抽样全部落在核内；`top_p=1.0` 与不传参在同种子下
  逐位相同；`top_p` 只在"自身之前质量 < p"上截断，概率极低的 `top_p` 仍保留
  argmax；重复惩罚把"4 → 3"翻成"1 → 2"（贪心），负 logit 的乘性分支同样生效；
  `penalty != 1` 缺 `recent`、越界历史、`top_p` / `penalty` 的 0、负数、NaN、
  +inf 全部显式报错。
- 束搜索（`_TableBackend` 按"最后一个 token"查表，结果与调用次数无关）：
  两步时贪心进死胡同、束搜索选出分数更高的另一条路径（贪心 `[2, 0]`、
  束搜索 `[4, 0]`）；`num_beams=1` 在 1/2/3 步上都与 `generate` 逐位一致；
  第一个位置就倾向 EOS 时返回空列表且不进 `max_new_tokens`；`max_new_tokens=0`
  不构图；`num_beams` 的 0 / `True` / 2.5、负 `length_penalty`、0 惩罚、空
  prompt、`spec.batch=2` 全部显式报错。
- **状态重排的真实验收**：宽度 >= 词表、两步时束搜索是穷举——用真实
  参考后端把 256 条候选逐个整段前向打分，束搜索的结果必须与暴力最优**分数
  相等**，且不低于贪心。父下标取错时这条立刻失败，`_TableBackend` 的零状态
  测不出来。
- 入口一致性：`generate` 的重复惩罚改变自己的输出（`[2, 2]` → `[2, 4]`），
  `generate_batch` 两行各自得到同样的历史演化；`LMRunner.generate_beam`
  与模块级逐位一致、第二次调用不再增图、同一会话里 batch 不同的图共存；
  有前缀或 `batch=2` 的会话上束搜索显式拒绝。

**验证**：本轮新增 18 个用例（1906 → 1924）。本地 1924（400 跳过）、WSL HIP
1924（356 跳过）、CUDA 真机（3×Tesla M40，`CUDA_VISIBLE_DEVICES=2`）1912
（105 跳过）全部通过。

**边界**：束搜索的状态重排走主机（`to_host` 后 gather），每步多一趟
D2H/H2D，因此它是"正确性优先"的实现，不承诺吞吐；只支持单条 prompt，不接
`constraint`，不与会话前缀缓存叠加；`length_penalty` 只作用于解码器式
（累计 log 概率）分数，不是 encoder-decoder 的"长度做幂次加在分数上"。
重复惩罚是全历史窗口，没有 `no_repeat_ngram_size` / 滑动窗口；top-p 与
top-k 同时给时先 top-k 后 top-p（与 HuggingFace 的顺序一致）。真机上只验证了
数值正确性，未做束宽与吞吐的标定。

### 5.195　后端承接面：任意单轴 argmax 与多轴 reduce

**动机**：5.153 / 5.174 把归约的**任意单轴**在设备上做完了（末维块内归约、
其余轴两维铺法），但边界还剩两类：`argmax` 只认最后一维，`reduce_sum` /
`reduce_max` / `reduce_mean` 一次只认一个轴。前端与 numpy 语义都允许更多，
于是同一张图在参考后端能跑、在 CUDA / HIP 上抛能力错误——这不是性能问题，
是**同一份 IR 的语义在不同后端不一致**。

**做法**（`backends/device_backend.py`，CUDA 与 HIP 共用同一份 Python 层）：

- **`argmax` 任意单轴**：非末轴先按"其余轴保持原序 + 目标轴放最后"置换，
  `ascontiguousarray` 之后复用同一个末维内核，输出再按原轴序恢复形状；
  `keepdims` 在恢复时把该轴写成 1。下标语义是**该轴上的位置**，置换不改变它。
  多个轴仍然拒绝：numpy 的多轴 argmax 返回的是展平后的下标，语义与单轴不同，
  不该借"多支持一点"悄悄换掉。
- **多轴归约**：归约与轴顺序无关，因此先把轴归一到升序、取 `keep + reduce`
  置换，把归约轴全部挪到末尾并合并成一个轴（`cols` 是它们的乘积），再复用
  同一份单轴内核；`keepdims` 用原形状（归约轴写 1）reshape 回去。轴本来就在
  末尾时 `perm` 是恒等，连转置都不发生。
- **空归约轴**：`cols == 0` 时按 numpy 的单位元返回——`sum` 是 0、`mean` 是
  NaN（numpy 同时发 warning）、`max` 没有单位元因此显式拒绝。这不走内核，
  避免"读 0 个元素"在块内归约里除以 0。
- **重复轴与空轴**：重复轴报能力错误；`axis=()` 报错并说明"要恒等就删掉这个
  节点"，而不是返回一份共享缓冲的假拷贝。

**判据**：conformance 新增 4 条用例（`argmax_axis0`、`argmax_axis1_keepdims`、
`reduce_sum_multi_axis`、`reduce_mean_multi_axis_keepdims`），参考后端
91 executed / 0 refused / 0 failed，CUDA 89 / 2 / 0、HIP 89 / 2 / 0；
`tests/test_cuda_backend.py` 的 `TestCudaReductions` 新增 4 条：

- `argmax` 在 `(4,7)` 的轴 0、`(3,4,5)` 的轴 0/1、`(2,3,4,5)` 的轴 2 上与参考
  后端逐位相等（`rtol=atol=0`，输出 dtype 是 i32）；`axis=1, keepdims=True`
  的形状是 `(3,1,5)`。
- 三个归约 × 四种轴组合（`(0,1)` / `(1,2)` / `(0,2)` / `(-1,-2)`）× `keepdims`
  两态，在 `(3,4,5)` 上与 numpy 一致。
- f16 的多轴 `reduce_mean`：dtype 保持 f16，与参考后端在 2e-3 容差内一致
  （f16 累加在 float 里、只在写回时舍入，与内核的既有口径相同）。

本轮替换掉的两条旧用例是"非末轴 argmax 拒绝"与"多轴 mean 拒绝"——它们
固定的正是这次要消除的边界。

**验证**：本轮新增 6 个用例（1924 → 1926；本机跳过数 400 → 402，因为两条
新用例只在 CUDA 上跑），conformance 87 → 91 条。本地 1926（402 跳过）、
WSL HIP 1926（358 跳过）、CUDA 真机 1914（105 跳过）全部通过。

**边界**：这是**语义补齐**而不是性能优化——非末轴 argmax 与多轴归约都要一次
转置拷贝（输入已连续且轴在末尾时为零），没有为它们单独写内核；多轴 argmax
仍显式拒绝（返回展平下标是另一套语义）；`axis=()` 被拒绝，参考后端的恒等
归约因此不是跨后端可移植的形态。

### 5.196　推理服务层：请求队列与连续批处理

**动机**：5.192 的 `generate_batch` 解决的是"一批请求同时到齐、左填充对齐、
一起跑完"。真实服务不是这个形状：请求随时到达、长度各异、完成时间不同。
把"等这一批全跑完再收下一批"当作批处理，慢请求会拖住所有新请求的首 token；
而 `generate_batch` 本身也没有队列、没有准入、没有槽位复用。这不是性能差
一点，是**服务语义缺失**。

**做法**（`aetherflow/models/lm.py`，不新增算子、不新增执行机制）：调度单位
是**解码步**而不是请求，`ServingEngine` 把一步拆成三段：

- **采样**：上一轮交出 logits 的每个槽位各采一个 token（复用
  `sample_next_token`，因此 top-p / 重复惩罚 / constraint 都是 per-request 的）。
  命中 EOS 或到达 `max_new_tokens` 的请求立即输出并腾槽。
- **准入**：空槽按 FIFO 从等待队列取请求，**只对本次准入的那几条**做一次
  批量预填充（`_prefill`，左填充 + reset，同 5.192 的口径），把每行状态与
  末位置 logits 拆回各自的请求对象。
- **解码**：对需要往里喂 token 的槽位跑一次批量 decode，图的 batch 固定为
  `capacity`；空槽喂 `pad_id`、结果丢弃；本次刚准入的行不取本步 decode 结果
  （它们的下一步 logits 就是预填充交出来的那一份）。

槽位一空就立刻准入新请求，因此慢请求不阻塞新请求的首 token——这就是
continuous batching 与"整批一起跑"的实质差别。图缓存键含 `(batch, length,
head)`（预填充按准入条数建图，条数变化就要另一张图），`max_graphs` 用 LRU
封顶；`resident=True` 时首次建图后把权重常驻（口径同 5.191）。

**公开面**：`submit(prompt, *, max_new_tokens, temperature, top_k, top_p,
repetition_penalty, eos_id, seed, constraint) -> rid`、`step() -> [(rid,
tokens)]`（返回**本步完成**的请求）、`run() -> [(rid, tokens)]`（推进到全部
完成，按完成先后）、`pending` / `active` 计数、`close` / 上下文管理。
`capacity` 必须是 int ≥ 1，`chunk_size` 落在 `[1, steps]`，`spec.batch` 必须
是 1（并发度是调度器自己的概念，不借用 spec 的 batch 维），构造期全部显式
校验。`max_new_tokens=0` 的请求不构图，直接进完成队列。

**判据**（`tests/test_lm.py` 的 `TestServingEngine`，7 条）：

- 同一组请求在 `capacity=1` / `2` / `4` 下的输出与**逐个调用 `generate`
  逐位相同**——这是连续批处理唯一有意义的验收：调度不改变结果。
- 容量上限生效：`capacity=2` 时第三、四条请求在槽位腾出前不开始，且中途
  准入的请求输出仍与单独跑一致。
- `max_new_tokens=0` 不建图（`graph_count` 不变）。
- EOS 只停自己那一路，其它槽位继续。
- `constraint` 按请求生效（`_OnlyToken` 只放行一个 token）。
- `resident=True` 会把权重登记给后端（`_ResidentRecorder` 记录调用）。
- 非法配置（`capacity=0`、非 int、`max_graphs=0`、`spec.batch≠1`、
  `chunk_size` 越界、非法 top_p / 重复惩罚）全部显式报错。

**验证**：本轮新增 7 个用例（1926 → 1933，全部在 `TestServingEngine`），本地
1933（402 跳过）、WSL HIP 1933（358 跳过）、CUDA 真机 1921（105 跳过）全部
通过。

**边界**：状态按槽位在主机侧拼成 batch 再拆回（参考后端是 numpy 拼接，设备
上是一次 D2H/H2D）；decode 图 batch 固定为 `capacity`，空槽让内核白跑，不做
动态扩图；预填充按准入条数建图，条数频繁变化会触发若干次构图（`max_graphs`
封顶后 LRU 复用）；只支持 `spec.batch=1` 的会话构造；`constraint` 对象有内部
状态，必须每条请求各给一份，这一点无法在构造期校验，只能由调用方保证。

### 5.197　中层优化的独立 pass 层：注册表与单独入口

**动机**：`passes/` 里七个 pass 都能单测，但"层"这件事只做了一半：两条管线
各自在函数里写死一串具体类（`__init__.default_pipeline` 一份、
`fusion.fused_pipeline` 一份），想只跑一个 pass、或按配置挑一串，调用方只能
import 具体类自己拼；名字（打印、日志、将来的命令行与 fuzz 选择）没有唯一
来源，加一个 pass 要改两处、漏一处就漂移。差距不是"少个函数"，是**这层没有
对外的统一入口**。

**做法**（新增 `aetherflow/passes/registry.py`，不改任何 pass 的实现）：

- **注册表**：`register_pass(name, factory)` / `unregister_pass` /
  `available_passes` / `get_pass`。三条显式约定：
  - **名字唯一**：重名默认报错，只有 `replace=True` 才覆盖；
  - **工厂每次给新实例**：pass 可以带状态，单例共享会让两次执行互相污染；
  - **注册名必须等于 pass 自己的 `name`**：注册期用一个丢弃的实例校验，
    拼错当场失败，而不是等到日志里出现两个不同的名字。
- **单独入口**：`run_pass(graph, name)` 只跑一个 pass；`run_passes(graph,
  names)` 按序列跑；`build_pipeline(specs)` 构造管线。三者都走
  `PassManager`，因此**单独跑也不能绕开 verifier**。
- **名字与实例混用**：管线元素可以是名字或 `Pass` 实例。带参数的 pass
  （`Autocast` 的 dtype / 算子集）**刻意不进注册表**——名字表达不了参数，
  用一个默认值偷偷构造出来只会让调用方以为配置生效了；它按实例进管线。
- **单一来源**：`DEFAULT_PASS_NAMES` / `FUSED_PASS_NAMES` 是两条管线的
  唯一顺序定义，`default_pipeline()` / `fused_pipeline()` 都从注册表构造；
  `fusion.fused_pipeline` 保留为薄委托，旧的 import 路径不破。

**判据**（`tests/test_passes.py` 的 `TestPassRegistry`，13 条）：

- 内置七个名字都在 `available_passes()` 里且有序；`autocast` **不在**（带参数）。
- `get_pass` 两次返回不同实例；未知名字抛 `ValueError` 并列出可用项；
  非字符串抛 `TypeError`。
- 重名注册抛错且消息含 `replace=True`；`replace=True` 后能覆盖；
  注册名与 `pass.name` 不一致时拒绝，且该名字不出现在可用列表里。
- `run_pass("dce", g)` 与 `DeadCodeElimination().run(g)` 的节点与输出名一致，
  且**原图未被原地修改**（pass 仍是函数式的）。
- `[p.name for p in default_pipeline().passes] == list(DEFAULT_PASS_NAMES)`，
  融合管线同理；`run_passes(g, DEFAULT_PASS_NAMES)` 与
  `default_pipeline().run(g)` 结构一致。
- `build_pipeline([Autocast(), "dce"])` 接受实例；`build_pipeline("cse")`
  （裸名字）报错，提示要写成列表；空管线是恒等（返回原图对象）。
- 单独入口仍然过 verifier：一个返回空图的坏 pass 在 `run_pass` 下抛
  `VerifierError`，`check=False` 时才放行。

**验证**：本轮新增 13 个用例（1933 → 1946），本地 1946（402 跳过）、
WSL HIP 1946（358 跳过）、CUDA 真机 1934（105 跳过）全部通过。

**边界**：注册表是**进程级全局**，插件注册的名字与本进程内所有调用方共享，
冲突只能靠 `replace=True` 显式表态，不做命名空间隔离；注册发生在 import
期，顺序由模块导入顺序决定，因此 `available_passes()` 只保证按字典序返回、
不保证注册顺序；带参数的 pass 必须按实例传入，注册表不解析"名字 + 参数字典"
这种配置形态；注册表不改变任何 pass 的语义，也不提供幂等 / 固定点迭代
——跑两遍是否收敛仍由各 pass 自己的性质决定。

### 5.198　二阶导数：对反向图再求导

**动机**：一阶反向图本身就是一张普通的 PORTABLE 图（5.17），所以 `backward`
可以再作用在它上面——引擎层面"二阶"早就成立，缺的是**入口**与**验收**。
让调用方自己拼 `backward(backward(...))` 有两个真实陷阱：一阶反向图的输出名
是 `d<wrt 名>`、输入是"前向输入 + 一阶种子 `d<输出别名>`"，要再求导必须在
一阶图里按名字找回 `wrt` 对应的那个 `Value`；二阶种子名如果没对齐，得到的
是"对 dy 求导"的图，形状还可能正好对得上，于是静默算错。同时"二阶"的边界
从来没写进判据：曲率为零时引擎报 `NoGradientError` 而不是给零，没有二阶规则
的算子（`conv2d_input_grad` 等）同样报错——这两点不写清楚就会被当成缺陷。

**做法**（新增 `aetherflow/ir/second_order.py`，不新增算子、不新增执行机制）：

- `hessian_vector_product(graph, wrt=None, *, active=None, seeds=None, name=None)`
  先建一阶反向图，按名字把 `wrt` 映射到一阶图对应的输入，再用一阶输出名
  `d<wrt 名>` 作种子键、`dd<wrt 名>` 作二阶种子输入名，建二阶反向图。
- 二阶图的三组输入与输出是**显式契约**：
  输入 = 前向全部输入 + 每个激活输出的种子 `d<别名>` + 每个 `wrt` 的二阶种子
  `dd<wrt 名>`；输出 = 每个 `wrt` 一项 `d<wrt 名>`，即 `H·v` 的对应分量。
  前向重放只需一份输入，一阶节点与二阶节点共用。
- `active` 与一阶同义（未激活输出在构图期被剪掉）；`seeds` 允许改名——
  默认名撞上已有输入时由 `backward` 显式拒绝，这里的 `seeds=` 是那条错误
  信息指向的出口；`wrt` 为空、`seeds` 长度不匹配或有重名都在构造期报错。
- 顶层与 `aetherflow.ir` 都导出，`__all__` 同步。

**判据**（`tests/test_autodiff.py` 的 `TestSecondOrder`，9 条，f64）：

- `f(x)=sum(x**3)`：HVP 与解析值 `6x·v` 逐元素相等（1e-12）。
- **独立判据**：与一阶梯度的中心差分 `(g(x+εv)-g(x-εv))/2ε` 一致（不借用
  二阶规则，1e-6）。
- 不给 `wrt` 时每个图输入各一项输出；`f=sum(x²w²)` 的 `dx` / `dw` 同时含
  对角项与跨输入块的混合项 `4xw`——说明这是完整 Hessian 乘向量，不是逐块独立。
- 收敛判据：用 HVP 逐基向量拼出 Hessian，解 `Hp=g` 后二次目标**一步**落到最优点
  （1e-12），且 `H=2I` 与解析值一致。
- 零曲率（`f=sum(x·w)` 对 `x`）抛 `NoGradientError`：IR 里没有无来源的零。
- 含 `conv2d` 的图抛 `NoGradientError` 且消息含 `conv2d`——"没有定义二阶规则"
  与"曲率为零"必须能被区分。
- 默认种子名 `ddx` 撞上已有输入时 `ValueError` 并提示 `seeds=`；改用
  `seeds=["v"]` 后成功。
- 参数校验：空 `wrt`、`seeds` 个数不符、`seeds` 重名各自报错。
- 二阶图仍是普通 IR：过 verifier、JSON 往返、执行逐位一致。

**验证**：本轮新增 9 个用例（1946 → 1955），本地 1955（402 跳过）、
WSL HIP 1955（358 跳过）、CUDA 真机 1943（105 跳过）全部通过。

**边界**：二阶图把前向重放一遍，节点数约 4 倍（反向 = 前向重放 + 一阶规则，
二阶再套一层）；只对具备 VJP 的算子成立，`conv2d_*_grad` / `maxpool2d_backward`
等会 `NoGradientError`；曲率为零不造零张量；`seeds` 只改名不改形状
（二阶种子必须与对应 `wrt` 同形）；HVP 要求静态形状，符号维在广播反向那一处
显式报错；不做 Hessian 的稀疏 / 分块 / 低秩近似，也不提供 `jvp` 前向模式
（只有反向模式的反向模式）。

### 5.199　趋势的自动采集调度

**动机**：5.38 有了 `benchmarks/trend.py`（跨运行的吞吐 / 显存曲线），5.70 的
CI 里也排了一步 `trend --rounds 3 --save`。但那一版每次都从零开始：GPU 作业
在托管平台上每次是一个干净工作区，`--save` 写出来的文件随 artifact 上传之后
就没人再接回去。也就是说"趋势"这一步每周都在产生**单点**，跨运行的含义为零。
另一半是文件会无限长：真接上历史之后，每天追加、从不裁剪，一年就是几千条。

**做法**（`.github/workflows/ci.yml` + `benchmarks/trend.py`）：

- **定时触发**：`on.schedule` 每天 18:00 UTC（北京次日 02:00）；GPU 作业的
  `if` 从"仅手动"改成"手动或定时"。正确性作业仍然只在 push / PR 上跑。
- **跨 run 接续**：用 `actions/cache/restore` + `actions/cache/save`。cache 是
  不可变的，因此 key 必须每轮唯一（`telemetry-cuda-<run_id>`），靠
  `restore-keys: telemetry-cuda-` 前缀匹配取最近一次；恢复到的
  `telemetry_history.json` 通过 `--compare` 接在本次记录之前，再原地写回。
  这件事用 artifact 做不到（要自己查最近一次 run id），用 git 提交也做不到
  （没有仓库——见客观阻塞项）。
- **按图名裁剪**：新增 `prune(recorder, keep)` 与 `--prune N`。按 `record.graph`
  分别只留最近 N 条，不是全局取最近 N 条——后者会让低频用例被高频用例挤掉，
  曲线直接断掉。裁剪只影响写盘，打印与 `--json` 仍用完整历史。
- **参数校验前置**：`--prune` 不给 `--save` 时在**跑基准之前**拒绝，而不是
  采几分钟再报错；`keep` 必须是 ≥1 的 int（`bool` 不算，`1.5` / `"3"` 不算）。

**判据**（`tests/test_trend.py` 的 `TestPrune`，4 条）：

- 每张图留最近 N 条：交错记录的保留集合与顺序正确，裁剪后 `trend()` 的
  `count` 与之一致。
- `keep` 大于历史长度时是恒等（长度与顺序都不变）。
- `0` / `-1` / `True` / `1.5` / `"3"` 全部 `ValueError`。
- `--prune` 单独出现时 `SystemExit` 且消息提到 `--save`。

**验证**：本轮新增 4 个用例（1955 → 1959），本地 1959（402 跳过）、
WSL HIP 1959（358 跳过）、CUDA 真机 1947（105 跳过）全部通过。CI 文件本身
只做静态改动，**托管平台仍未实跑**（见 §7 的阻塞项）。

**边界**：cache 的淘汰策略由平台决定（长期不活跃会被清掉），因此历史可能
从某一轮重新开始——曲线从"无限长"变成"平台愿意留多久就多久"，这一点不做
补偿（要做就得有外部存储，那是另一件事）；`prune` 只按图名裁剪，不区分
用例名与图名（`record.graph` 记的是 `Graph.name`，同名图本来就会被合并成
一条曲线）；定时任务只在装了 self-hosted GPU runner 的机器上有意义，没有
runner 时 schedule 只是排一个永远 pending 的作业。

### 5.200　系统化边界值测试与长上下文数值契约

**动机**：边界用例一直是散着写的——某处发现一个空轴问题就补一条，某处发现
f16 溢出就补一条。这种增量方式有两个盲区：一是**入口没有穷举过**，不知道
哪一类边界根本没被碰过；二是**极端值的语义没有被钉住**，只是"某条用例顺手
断言了一下"。本轮把它变成一张表：非法值必须在构造期显式报错，极端值必须有
确定的 IEEE 语义，且不能在宿主上打 numpy 的 RuntimeWarning（同一张图在参考
后端与设备后端上不该有"日志差异"）。

**做法**：新增 `tests/test_boundaries.py`，按五组组织；其中两处是**真实缺陷**，
不是补测试就能了事：

- **数学族在宿主上打警告**：`log(0)`、`sqrt(-1)`、`exp(1000)`、f16 的 `add`
  溢出都会往 stderr 打 RuntimeWarning。`div` 在 5.175 已经按 IEEE 静默处理，
  其余没跟上。`ops.py` 新增 `_ieee_math(fn)` 与 `_binary_ref` 的
  `np.errstate(all="ignore")`，把 exp / log / sqrt 与全部二元逐元素算子统一到
  同一条口径；**结果不变，只是不再污染日志**。
- **`axis=None` 抛的是 `TypeError: 'NoneType' object is not iterable`**：
  IR 不支持全归约（没有独立的语义位），但错误信息没告诉调用方该写什么。
  `_norm_axes` 改为显式 `ValueError`，并给出 `axis=tuple(range(rank))`。

五组判据（20 条）：

1. **形状**：0 元素张量逐元素照常流动、秩 0 是一等值；归约轴用一张表讲清
   `0 / -1 / (0,2) / 空轴 / keepdims` 的结果形状，`3 / -4 / (0,0)` 显式报错；
   空轴是恒等而不是"全归约"；`reshape` 元素数守恒；输入名 / 输出名重复拒绝。
2. **dtype**：未知 dtype（`"u8"` / `"float32"` / `""`）拒绝；bf16 在参考后端
   借用 f32 承载（契约写死，`cast` 后类型标签仍是 bf16）；f16 溢出成 `inf`
   是位宽语义、不报错、不告警。
3. **数值**：`1/0`、`-1/0`、`0/0` 分别 `inf` / `-inf` / `nan`；`log`、`sqrt`、
   `exp` 的定义域外边界；softmax 在 `[1000, 0, -1000, 999]` 上仍有限且归一；
   NaN 经 `relu` 传播；空轴 `reduce_max` 显式 `ValueError`（没有单位元）。
4. **序列化**：无输出的图是 `verify` 的 error issue；截断 JSON 抛
   `JSONDecodeError`；`dumps` 不替坏图补输出，往返之后仍然坏。
5. **长上下文数值**：1024 步、1 层 Mamba、d=8 的语言模型，整段前向与切成
   4 / 16 / 32 块的续接，末位 logits 相对误差 < 1e-5，且 `errors[32] ≤
   3·errors[16] + 1e-6`——**误差不随块数增长**；全部中间状态与输出有限。
   实测 4 块 0、16 块 1.4e-7、32 块 9.3e-8。

**验证**：本轮新增 20 个用例（1959 → 1979），本地 1979（402 跳过）、
WSL HIP 1979（358 跳过）、CUDA 真机 1967（105 跳过）全部通过。

**边界**：这张表覆盖的是 IR 语义层与参考后端；设备端的 dtype / 轴支持由
conformance 与 `test_cuda_backend.py` 负责，两边不重复。长上下文只测到 1024 步、
1 层、d=8：它钉的是"分块不放大误差"，不是"32K 上下文可用"——真正的长上下文
还缺真实 checkpoint、tokenizer 与显存口径，那属于产品级验收，不是这一层能
宣称的。`axis=None` 仍然不支持（只是错误更清楚）；bf16 仍只有参考后端的
f32 承载语义，设备端没有实现，构图期由后端能力检查拒绝。

### 5.201　跨图控制流：按数据选择子图与按数据迭代

**动机**：静态图是 DAG，节点序列里没有"这一层走哪支"的语义位，也不允许回边。
此前数据相关的分支只能写在两次 `run` 之间（5.68 的 `DynamicRunner`）：读回中间
值、在 Python 里 `if`、再重新取图。这能跑，但控制流**不在任何可交付物里**——
既不能跟着图一起序列化，也没有一处显式的地方说明"哪些输入决定了走哪支"。

**为什么不做成 IR 算子（把子图塞进节点属性）**：

1. 节点属性必须能完整 JSON 序列化，而 `Graph` 是活对象（值 id、生产者指针、
   seal 状态），塞进 `attrs` 之后 `dumps` 会立刻失效；
2. 设备后端要递归编译嵌套子图，等于给每个后端各写一遍控制流语义，而这正是
   "IR 是唯一契约"要避免的事；
3. 谓词必须回主机读一次真值。这一点绕不过：数据相关的分支本来就不在
   CUDA Graph 可捕获的范围内（捕获要求路径固定），把它做进 IR 不会让任何后端
   跑得更快，只会让语义边界更模糊。

**做法**（新增 `aetherflow/ir/control.py`，不新增算子、不改执行骨架）：
控制流是"子图之间"的关系，每张子图仍是普通 `Graph`——照样过 verifier、照样
`dumps` / `loads`、照样交给任何后端执行。

- `ControlRegion`：统一接口，只有三样东西——`inputs`（名字 + 类型）、`outputs`
  （名字 + 类型）、`run(feeds, backend)`。`GraphRegion` 把一张图包成区域，
  输入名与输出名就是图上的名字，不另造映射表。
- `Conditional`：`predicate` 必须是**恰好一个 0 维标量输出**（bool 或数值都接受，
  取 Python 真值）；`then` / `otherwise` 的输出**名字、顺序、类型**必须完全一致。
  执行时只跑谓词与**被选中的那一支**——未选中分支里的错误不会被触发。
- `WhileLoop`：`carried` 是「状态名 → 循环体输出名」的映射。状态名既是调用方
  给初值的名字，也是循环体的输入名；输出名是这一轮算出的新状态。要求状态名是
  体的输入、输出名是体的输出、两者类型相同；条件区域只能看到"自由输入 + 当前
  状态"，同名必须同类型。`max_steps` **必须显式给出**（没有默认值），用尽仍未
  退出就抛 `ControlFlowError`——默认值会把"本该收敛"的错误变成一次静默截断，
  而截断后的数值看起来完全合法。
- 区域可以互相嵌套：`Conditional` 的分支、`WhileLoop` 的体都可以是另一个控制
  节点，验证与执行不需要分支判断（"if 里套 while"不需要额外机制）。
- `to_spec` / `from_spec`：控制结构导出成 `{"kind": "graph" | "cond" | "while"}`
  的纯 JSON，子图走 `ir.serialize.to_dict`，不重复定义格式；`from_spec` 会把
  构造期的全部契约检查重新跑一遍。

**判据**（新增 `tests/test_control_flow.py`，24 条）：

- 选支：同一张 `Conditional` 在不同输入上走两支；else 分支是整数除零，`x > 0`
  时它不被执行（运行正常），`x < 0` 时错误按预期抛出——这一条直接证明分支是
  惰性的，而不是"两支都算再挑"。
- 接口：`inputs` / `outputs` 取自 `then`，名字与类型逐项断言。
- 分支契约：输出名不同、输出类型不同、谓词有两个输出、谓词不是 0 维、同名输入
  类型冲突、分支不是图也不是区域、`name` 为空——全部构期报错。
- 循环：从 0 数到 5（步长 1）与从 0 数到 4.5（步长 0.5）都停在正确值；对外
  接口只暴露自由输入，状态名不出现在 `inputs` 里；`max_steps=2` 抛
  `ControlFlowError` 且消息含 `max_steps`；缺初始状态、`carried` 不是映射、
  `carried` 为空、状态名与输出名相同、状态名不是体输入、输出名不是体输出、
  输入输出类型不一致、`max_steps` 为 `0` / `-1` / `True` / `1.5` / `"4"`
  全部显式报错。
- 嵌套：`Conditional` 里套 `Conditional`（内外两层谓词看到同一个 `x`）；
  `WhileLoop` 的体是 `Conditional`。
- 序列化：`Conditional` 与 `WhileLoop` 走 `json.dumps` → `json.loads` →
  `from_spec` 后执行结果逐位一致（`max_steps` 与 `carried` 也还原）；未知
  `kind` 抛 `ValueError`、`spec` 不是映射抛 `TypeError`。
- 与参考执行器对拍：循环退出时的状态等于"参考执行器在最后一步输入上再走
  一步"的输出，而不是多走一轮。

**验证**：本轮新增 24 个用例（1979 → 2003），本地 2003（402 跳过）、
WSL HIP 2003（358 跳过）、CUDA 真机 1991（105 跳过）全部通过。

**边界**：谓词每次都要回主机读一次真值（一次同步），因此**长循环的派发开销
在主机侧**——没有 device-side 循环，也没有把循环转成 `scan_linear` 的自动改写；
后者是性能优化，与这里的语义表达是两件事。控制流不是可微结构：要训练带分支的
模型，仍得在 Python 里展开（`for` + `if` 只看 Python 值）或按段构造反向图，
`backward` 不会穿过 `Conditional` / `WhileLoop`。两个分支不并发执行，也没有
"同时算两支再选"的向量化形态（那需要把分支写成 `where` 语义的算子组合）。
序列化覆盖的是控制结构本身，子图仍受各自的 IR 版本治理；区域里的图仍是静态图，
控制区域不引入新的 IR 版本。

### 5.202　代码生成的第三条路：跨方言源码转译

**问题**：1.2 里"复用 / 转译 / 原生生成三模式"一直是 🟡，只有两条——"原生生成"
（L1 → L2 → 方言 → 源码 → NVRTC / hipRTC，全部 56 个算子的生产路径）与"调库"
（cuBLAS / cuDNN / cuBLAS Lt / hipBLAS / MIOpen）。缺的那条是**别人写好的源码**：
论文附录里的 `.cu`、旧项目里的一段 kernel、某家库的示例。此前唯一的处理方式是
"读懂它，再用 L2 重写一遍"——而重写正是最容易悄悄改掉语义的一步。

**做法**：`lower/translate.py`，`translate(source, source_dialect=, target_dialect=)`。
它不是 C 解析器，是一张**可以逐条审计的替换表**加一个词法分层的扫描器：

- 只改**标识符**与 `#include` 里登记过的头文件名。注释、字符串、字符字面量整体
  跳过（注释里的 `cudaMalloc` 是散文，字符串里的它是数据，改了就是另一份程序）；
- 未登记的标识符原样透传，未登记的头文件原样保留；
- 同方言调用是恒等：`translate(s, "cuda", "cuda") == s`，不扫描也不改——"把源码
  复制到同厂商的另一台机器"不该因为里面有 `asm` 而失败；
- 跨方言命中拒收项立刻抛 `TranslationError`，消息带**行号**与原因。

**拒收项就是转译的边界**（这是本小节最重要的部分）。只做"换个名字仍是同一个
语义"的改写，凡是"换名字会变语义"的一律拒绝：

| 源码 | 为什么不能改名 |
|---|---|
| `asm("cvt.f32.f16 …")` | PTX 内联汇编在 AMD clang 上没有对应写法 |
| `__shfl_down_sync(mask, v, d)` | HIP 的 `__shfl_down(v, d)` 少一个参数——这不是改名，是改参数表 |
| `__ballot_sync` / `__any_sync` / `__all_sync` / `__activemask` | 同上；且两边 warp 宽度与掩码语义不同 |
| `__half` / `__half2` / `__half2float` / `__float2half_rn` | CUDA 的半精度是包装类型 + 转换函数，改名会改变舍入语义 |
| `_Float16`（→ CUDA） | 反过来同理：`(_Float16)v` 改名成 `(__half)v` 能编译，但舍入路径不同 |
| `__builtin_bit_cast`（→ CUDA） | Clang 内建，NVRTC 没有 |
| `cooperative_groups` / `cudaLaunchCooperativeKernel` | 两家的协作组接口不同，必须按目标平台重写 |

这张表不是"还没做"，而是**结论**：`backends/hip/codegen.py` 的 f16 辅助函数必须
另写一份（`_Float16` + `__builtin_bit_cast` + CAS 原子加），不能靠转译得到；反过
来 CUDA 的 f16 prelude 用了 PTX，也不能转给 HIP。两条路径在这里互相印证，用例
直接拿这**两份真实的 prelude** 做判据——它们必须被拒绝，而不是被"尽力转换"。

**接进编译缓存**：`compile_translated(backend, source, func, source_dialect=)`。
转译完成后走 `backend.kernel(key, source, func)`，也就是与原生生成同一个编译缓存
（`CodegenContract`）。键是 `("translate", 源方言, 目标方言, sha256(原文), func)`：
原文改一个字符就重新编译，同一份原文转译两次命中同一个句柄。目标方言缺省按后端
的显示标签推断；显式给出却与后端不符时当场报错——把 HIP 源码交给 NVRTC，只会
得到一句来自编译器、看不出根因的失败。

**验证**：本轮新增 23 个用例（`tests/test_translate.py`）：

- 正向：`#include <cuda_runtime.h>` → `<hip/hip_runtime.h>`，`cudaMalloc` /
  `cudaMemcpy` / `cudaMemcpyHostToDevice` / `cudaDeviceSynchronize` /
  `cudaError_t` 等逐个对上；反方向（HIP → CUDA）同样；
- 不该改的：注释、字符串、字符字面量、未登记标识符、未登记头文件全部原样；
- 拒收：PTX `asm`（断言消息含"第 2 行"）、`__shfl_down_sync`（断言消息含
  "掩码"）、两个方向的半精度类型、`__builtin_bit_cast`；
- 与本框架的原生生成互证：f32 的两份方言源码逐字相同（5.40），因此转译必须
  恒等；f16 的两份源码不同，且**两个方向都必须被拒绝**；
- 编译入口：缓存键相同 / 原文改一个字符后不同、键里带 64 位 sha256、后端标签
  与目标方言不符时拒绝且一次 `kernel()` 都没调、无标签后端必须显式给目标、
  没有 `kernel()` 的后端报 `TypeError`。

**边界**：转译不解析 C，因此它不检查源码本身是否合法、不检查目标方言的类型是否
齐全、也不做任何语义等价证明——它只保证"表上的 token 换了、表外的没动、表上
标为不可换的会失败"。跨厂商源码仍要自己过一遍目标编译器（这正是
`compile_translated` 存在的意义：失败发生在真实编译器上，而不是在转译器里）。
不做 C++ 模板展开、不做宏展开（宏体内的标识符同样按 token 处理，但宏参数拼接
`##` 之后的标识符不会被识别）——这两项都属于"写一个真正的 C 前端"，与本节
"让一份现成源码换个厂商跑"的目标不成比例。

### 5.203　cuSPARSE 接成 `sparse_linear` 的可选实现

**动机**：5.173 只做过内核级对照——`benchmarks/sparse_compare.py` 用
`cupyx.scipy.sparse` 在同一批形状上量"自研散射 vs cuSPARSE"，四个形状里自研
核快 1.73×～2.38×，因此生产路径一直没接库。但"当前形状自研更快"不等于"永远
不需要库"：事件稀疏度、行宽与硬件代际都会改变拐点，而缺了这条可选实现，
换一台机器就只能改内核。这一节把库路径接成**可选的第二条实现**，同时保留
默认不动。

**做法**（`aetherflow/backends/cuda/cusparse.py`）：

- ctypes 直接绑 9 个符号：`cusparseCreate` / `cusparseSetStream` /
  `cusparseCreateCsr` / `cusparseCreateDnMat` / `cusparseSpMM_bufferSize` /
  `cusparseSpMM` / `cusparseDestroySpMat` / `cusparseDestroyDnMat` /
  `cusparseDestroy`。加载顺序是 `find_library` → 常见 soname → pip 包
  `nvidia.cusparse` 的包内路径；失败只试一次，错误里列出全部尝试过的名字。
- **CSR 就是三件套**：`offsets` 当 `csrRowOffsets`、`indices` 当
  `csrColInd`、`values` 当 `csrValues`，索引基址固定 0。稠密权重与输出都用
  行主序描述符（`ld = cols`），因为两者在框架里本来就是 C 连续，转成列主序
  只会多一次主机侧心算。
- **覆盖范围刻意收窄**：f32 值 + i32 的 `indices` / `offsets`。f16 与 f64
  在 `data_type()` 里抛 `CusparseUnsupported`，i64 索引在 `index_type()` 里
  同样拒绝。这不是"库不支持"（cuSPARSE 本身都支持），而是"这些形态还没有
  真机对拍数据"；登记表就是覆盖声明，表外的当场失败。
- **workspace 缓存、描述符每次重建**：workspace 字节数只由
  `(步数, 行, 列, nnz, dtype)` 决定，按这个键缓存；描述符里写死的是本次
  五个数组的地址，而同一张图每步都换一批事件，所以每次重建、用完立刻销毁。
  `cusparseCsrSetPointers` 能省掉重建，但那是另一条没有真机验证过的路径，
  正确性优先。
- **三个取值**：`sparse_impl="custom"`（默认）/ `"auto"` / `"cusparse"`。
  `"custom"` 与 `"auto"` 在基类 `DeviceBackend` 里，"cusparse" 由
  `CudaBackend` 扩展；HIP 的 `SPARSE_IMPLS` 不含它，**构造期**就报错，而不是
  等到第一次稀疏乘。默认仍是 custom，理由有两条：
  1. 自研核在设备端带越界检查（`err` 标志 + 主机侧同步复查），库不做逐元素
     检查——默认值不能把一道检查悄悄换掉；
  2. 5.173 的四个形状上自研核更快。
- **回落规则**：`auto` 只在"库可用 + 非捕获期 + 登记过的 dtype"三条同时满足
  时走库；捕获期显式跳过（`cusparseSpMM_bufferSize` 是 host API，与 cuDNN 的
  algo 查询同一条理由）。显式 `"cusparse"` 把这三种情况都变成
  `BackendCapabilityError`，并在错误里说清是哪一条不满足。
- 首次失败的原因记进 `_cusparse_broken`，后续步骤不再反复 `try/except`
  （与 `_cudnn_broken` 同一条策略）；`set_device` 会把 handle 与 workspace
  一起丢掉，因为它们绑在旧卡的 context 上。
- 这仍属 ADR-004：IR 里的 `sparse_linear` 是语义完整的 PORTABLE 算子，
  "用 cuSPARSE 实现它"是后端内部选择，不产生 LIBRARY 节点，conformance 也
  不因它多一条用例。

**验证**：新增 26 个用例（`tests/test_cusparse.py`）。

- 离线 19 条：f32/i32 的映射、f16/f64/i64 的显式拒绝与消息、库缺失时
  `available()` 为 `False` 且建引擎抛 `CusparseUnavailable`、`plan_key` 把
  nnz 与 dtype 算进键、`sparse_impl` 的七条选择分支（custom 不用库 / auto 在
  库缺失、f16、捕获期回落 / auto 库可用时启用 / 显式 cusparse 对 f64、
  i64、捕获期、缺库分别报错）；`_pick_cusparse` 用
  `CudaBackend.__new__` 喂字段，不碰显卡。
- 真机 7 条（`CUDA_AVAILABLE` 且 `available()`）：空步与重复行、nnz 从 0 到
  7 变化、512×256×64 步的大形状、`auto` 与 custom 对拍、f16/f64/捕获期的
  显式拒绝。**本轮全部跳过**：`192.168.1.18` 的 SSH 可达，但 NVIDIA 驱动
  未加载（`nvidia-smi` 无法通信、`lsmod` 里没有 nvidia 模块，只剩陈旧的
  `/dev/nvidiactl`），整个 CUDA 套件都没能复跑。
- 本地 2052 个用例 / 409 跳过、WSL HIP 2052 个用例 / 365 跳过，全绿。

**边界**：

1. **库路径没有真机数值验证。** `cusparseCreateCsr` / `SpMM_bufferSize` /
   `SpMM` 的调用序列是按官方签名写的，参数顺序（尤其 `alpha` → A → B →
   `beta` → C）在离线用例里无法证明；只有恢复真机后跑通那 7 条，这条路径才
   算闭合。这也是默认值不切换、`auto` 必须显式打开的直接原因。
2. **不做下标越界检查。** 自研核越界时置 `err` 并抛错；库路径越界是设备端
   未定义行为。显式选 `"cusparse"` 时，下标合法性回到调用方责任（与 5.175
   记录的 GPU 端边界一致）。
3. **只覆盖 f32 + i32。** 半精度、双精度与 i64 索引都已验证过"拒绝路径"，
   但没有对拍数据就不开放；要放开需要各补一组真机用例。
4. **描述符不缓存。** 每一步多出 6 次 host API 调用（建/毁各三次），小事件
   数下这部分开销可能吃掉库本身的收益——这正是"默认不启用"的另一个理由，
   等真机数据再谈是否值得引入 `SetPointers`。
5. **HIP 没有对应入口。** hipSPARSE 存在，但本轮没有接；HIP 上
   `SPARSE_IMPLS` 只有 custom / auto，`auto` 等价于 custom。

### 5.204　事件幅度的梯度：`sparse_linear` 的 `values` 不再被静默丢弃

5.132 的 `sparse_linear` 前向是

    out[s, :] = Σ_{e ∈ [offset[s], offset[s+1])} values[e] * weights[indices[e], :]

事件表示把稠密的 `S×K` 脉冲矩阵换成了 `(values, indices, offsets)` 三元组，前向
因此只花 `E` 次行向量缩放。但它的 VJP 当时只给 `weights` 梯度，`values` 直接
返回 `None`——注释的理由是"SNN 里事件幅度是 0/1 的发放强度，不训练它"。这个理由
只对了一半：`values` 确实常常来自 `spike`（不可训练），但它也可能来自一个可学习的
发射强度、门控或掩码，而 `None` 让"梯度为零"与"这里是离散量、不求导"两种语义
混在一起，上游再也无法区分。按 `Op.vjp` 的约定，`None` 表示"该输入确实不接收
梯度"，所以这里不是"没实现"，而是**语义上缺了一块**。

补齐后的第 2 个输入梯度就是

    dvalues[e] = dout[s(e), :] · weights[indices[e], :]

`s(e)` 是事件 `e` 所属的步，由 `offsets` 唯一决定。它与权重梯度的方向正好相反：
权重梯度要"把 `values[e] * dout[s]` 散射回 `indices[e]` 那一行"——同一行会被多个
事件命中，必须原子加；而每个事件只属于一步、只出现一次，所以 `dvalues` 是**逐
事件独立**的，直接写一次即可，不需要 `add.at` 式的重复下标处理。

**为什么不散射回稠密 `(S, K)` 的 `dspikes`。** 那需要 `S×K` 的缓冲并原子加，正好
把前向省下来的稠密脉冲矩阵又在反向造回来。事件形态的梯度长度就是 `E`，交给
`spike` 的反向（替代梯度逐元素乘）或任何上游算子按事件处理即可。这也是 5.136
`spike_linear` 的经验：能让梯度留在事件形态，就不要先展开再压缩。

**内核按步并行，不按事件并行。** 按事件并行时每个线程需要先知道"自己在第几步"，
而在设备上把事件号映射回步号只能对 `offsets` 做二分——L2 没有 `while`，二分要展开
成 `ceil(log2(steps+1))` 轮比较。这里改成第 `i` 个线程包干第 `i` 步的全部事件：
`offsets` 顺序读一遍、每个事件算一个长度 `cols` 的点积后写 `dvalues[e]`，一个事件
只被一个线程写一次。代价写在 §7：并行度是 `steps` 而非 `E`，空步的线程直接空转，
单步事件极多时那一个线程要沿 `cols` 串行。

**接口与拒绝面。** 新算子 `sparse_linear_values_grad(dout, weights, indices,
offsets) -> (E,)`：`dout` 必须是 `(S, N)`、`weights` 必须是 `(K, N)`，两者最后一维
相等且同 dtype；`offsets` 长度静态且等于 `S + 1`（与 `sparse_linear_weight_grad`
用 `rows` 属性带过行数是同一条"反向图里没有前向输入"的理由，这里步数可以直接从
`dout` 的形状读出来，所以不需要属性）；`indices` / `offsets` 是整数、无梯度。
输出形状取 `indices` 的形状——事件数可以是符号维，和 `spike` / `spike_offsets`
的符号 `E` 一致。`indices` 越界与 `offsets` 非降/越界在参考实现里 `ValueError`，
设备端走既有的 `err` 标志。

**判据。** 11 条纯 Python 用例：解析式逐事件对拍、f64 中心差分、同一步里两个事件
命中同一行时的互不干扰、空步、符号事件数（`E = 3 / 5` 各跑一次）、f16（f32 累加）、
构图期拒绝步数不符与 dtype 不符、越界下标与 `offsets` 非降、反向图的
`verify` / `verify_portable_only` / 序列化往返。设备路径由 conformance 新增的
`sparse_linear_values_grad` 用例覆盖（CUDA / HIP 共用同一份 L2 描述）。本地
2063 / WSL HIP 2063 全绿，CUDA 真机仍因驱动未加载没有复跑（见 §7 的 5.203 条）。

### 5.205　优化器状态分片（ZeRO-1）与 GPipe 式流水

5.190 的数据并行把一批数据摊到多块卡，但每块卡仍持有**整份参数**与**整份优化器
状态**；而且它解决不了"模型本身一块卡放不下"。这一节补的是训练侧的另一半：
状态按参数名分片（ZeRO 第 1 级），以及把整图按用户给的切点拆段、每段常驻一块卡
的流水骨架。

**分片省的是主机内存，不是显存。** `ShardedOptimizer` 刻意只在主机侧切开槽位：
参数表仍是同一个 `dict`，各片的优化器拿到的是**同一批 ndarray 对象**的子集，
`param -= ...` 原地改的就是共享的那份数据，因此不需要"写回"步骤，也不存在两份
副本不一致的窗口。要省显存必须让每个进程只上传自己那片参数，那属于运行时改造，
与 5.190 的边界（单进程串行切卡、无 peer-to-peer）同源。

**为什么分片必须逐位一致。** Adam / SGD 的更新是逐参数的：某个参数的 `m` / `v`
只与它自己的梯度以及共享的步数有关。分片把"对全部参数循环"换成"对分片循环"，
每个参数内部的运算序列一个字都没变，所以结果**逐位相等**，不是"在一个容差内
相等"。这条性质就是判据——测试拿同一起点、同一梯度序列，跑整份优化器与
`world_size = 2 / 3` 的分片优化器，逐步比对每个参数的原始字节（Adam 与
SGD + nesterov 各一组），一个 ulp 都不放过。

**分片按名字切、不按元素切。** 拆单个张量要改优化器的槽位布局，而收益只在
"一个参数就大到一片装不下"时才出现。片内归属用"每次放进当前最轻的一片"的贪心，
权重取参数字节数：结果只由参数名顺序与形状决定，同样的输入永远得到同样的划分
（可复现比更精确的均衡重要，否则两次运行的槽位归属不同，存档就对不上）。各片
最多相差一个参数的大小；`world_size` 大于参数个数时出现空片，这是**合法状态**，
不报错，但空片仍要参与 `step_count` 的推进，否则各片步数会分叉。

**接口与拒绝面。** `ShardedOptimizer(factory, params, *, world_size)` 不继承
`Optimizer`（它持有的是多份子优化器），但提供相同的 `params` / `assignments` /
`shards` / `step_count` / `lr` / `step` / `hyperparams` / `state_dict` /
`load_state_dict`，5.190 的 `DataParallelTrainer` 因此只把类型检查放宽成
`(Optimizer, ShardedOptimizer)`。构造时逐片核对 `hyperparams()`：分片是存储布局，
不是"每片用不同学习率"，不一致当场报错。`world_size` 非 int / 小于 1、参数表为空、
`factory` 造出来的不是 `Optimizer`、存档里的分片数或归属表与当前配置不符，全部
显式拒绝。存档额外写 `world_size` 与 `assignments`，恢复时按名字把槽位放回原片，
续训与整份优化器逐位一致。

**流水给的是容量，不是吞吐。** `PipelineStage(graph, params, inputs=(), output=None)`
把整图拆成 `段 0 → 段 1 → … → loss`，每段是一张独立的 `Graph`，段间只通过一个
激活张量相连，每段参数常驻该段所在的卡。后端是单进程串行切卡，"前向的
micro-batch 1 与反向的 micro-batch 0 并发"做不到，所以本实现是 **GPipe 的调度
骨架**：micro-batch 逐个走完全部段，再逆序反向，段间 `dout` 用
`copy_across_devices` 经主机中转。如实写在接口上，不假装能重叠。

**种子带权，micro-batch 只等分。** 损失约定为按样本求均值（与 5.190 同一条），
第 `j` 份的种子因此是 `1 / micro_batches`（`reduction="sum"` 时是 `float(chunk)`），
各份梯度直接在主机侧相加后交给 `optimizer.step`，不再二次缩放。只有等分时
"各份梯度求和再除以份数"才等于整批均值；不等分要先按样本数加权，而每份的样本数
在静态图里表达不了，所以 `batch % micro_batches != 0` 在构图期拒绝。

**构图期的整图一致性检查。** 参数必须是本段图的输入且不跨段重复；非首段必须
恰好一个段间输入、名字等于上一段的 `output` 且类型一致；最后一段输出必须是标量；
各段参数并集必须与 `ParameterStore` 的名字集合相等；数据输入允许出现在任意段，
但同名形状必须一致；每段的图输入必须恰好是"本段参数 ∪ 段间输入 ∪ 数据输入"。
每条都对应一个显式错误，不做"猜一个合理布局"。

**数值判据为什么是容差而不是逐位。** 整批一次算与 micro-batch 分别算再相加，
矩阵乘内部的归约顺序不同，浮点结果会有 1e-7 量级差异，所以判据是"两段 / 三段
流水跑 N 步后的参数与单卡整图在 `rtol=1e-5` 内一致"（`micro_batches = 1 / 2` 各
一组）。这与 ZeRO 那条逐位判据的强弱不同，是有意的：分片不改变任何运算序列，
流水会改变归约形状。

**判据。** 35 条纯 Python 用例：ZeRO 侧 18 条（分片的确定性 / 覆盖 / 均衡 / 空片
合法性 / 非法参数；Adam、SGD + nesterov 与整份优化器的**逐位**对拍；空片步数同步；
缺失与多余梯度；`factory` 非 `Optimizer`；超参不一致；`lr` setter；归属表逆映射；
存档往返续训；分片数与归属变化被拒绝），流水侧 17 条（`_FakeDevices` 伪多设备后端；
两段 / 三段 / 单段与单卡整图对拍；`micro_batches = 1 / 2`；设备数不符、输入名不符、
首段有输入、参数跨段重复、store 覆盖不全、损失非标量、段图额外输入、batch 不整除、
`micro_batches` 非法、静态 batch 不符、缺数据、设备状态还原、跨卡切换）。两条路径
都在纯主机侧，设备路径由 5.190 的跨卡拷贝覆盖，没有新增内核。本地 2098 / WSL HIP
2098 全绿（跳过分别 409 / 365）；CUDA 真机仍因驱动未加载没有复跑（见 §7 的 5.203 条）。

### 5.206　批量解码的逐行约束：把"状态逐序列"从拒绝理由变成接口

**动机**：5.196 的 `ServingEngine` 早就支持**按请求**挂约束，但同一份能力在
`generate_batch` 上是关着的（`_check_batch_request` 见到非空 `constraint` 就抛
"批量生成不支持受约束解码"）。拒绝的理由本身是对的——文法实例带推进状态，
共享一份会把两条序列的写法搅在一起——但它只否定了"共享"，没有回答"每个序列
各给一份"这种显然的用法。批量解码与受约束解码因此成了两个不能叠加的功能。

**做法**：`constraint` 参数接受第三种形态。除了 `None` 与 `batch=1` 时的单个
文法，还可以是**长度等于 `spec.batch` 的序列**，元素是该行的文法实例或 `None`
（该行不受约束）。规范化收在一个纯函数 `_row_constraints(constraint, batch)` 里，
`_check_batch_request` 与 `_sample` 共用，因此无状态入口、`LMRunner.generate_batch`
与会话内解码看到的是同一套规则。屏蔽与推进仍是原来的两步：`_masked(row,
held.allowed())` → 采样 → `held.feed(token)`，只是按下标取该行自己的实例。

**为什么不允许"一个实例广播给所有行"。** 这不是省参数，而是静默的错误来源：
`JsonGrammar` 的 `allowed()` 取决于已经写出的前缀，两条序列在同一实例上推进会
互相污染，结果既不是第一条的约束、也不是第二条的。`batch > 1` 传单个实例继续
显式报 `ValueError`，并提示改用序列；`str` / `bytes` 不算序列（它们的 len 是
字符数，容易被误当成"每行一个字符"）。

**拒绝面共有四条，全部在碰图与设备之前判定**：序列长度与 `batch` 不等；`batch > 1`
时传单个实例；序列元素缺少 `allowed` / `feed` / `done` 三者之一（逐行报出下标）；
以及原有的"约束与自定义 `eos_id` 冲突"（逐行复用 `_check_request`，停止符必须
等于 `spec.eos_id`）。补完文法失败也按行报错——`_finish` 逐个检查未完成的实例，
错误信息带行号，而不是笼统的"受约束解码没有走完"。

**判据**：4 条新用例（本地全量 2102 通过 / 409 跳过，此前 2098 / 409）。
`test_constraint_is_per_row` 让第 0 行只允许 token 3、第 1 行不约束，断言第 0 行
吐出 `[3, 3]`、第 1 行与单条 `generate` 逐 token 相等（行间互不影响由 5.192 的
左填充 + 逐行 reset 保证）；`test_row_constraint_sequence_must_match_batch` 覆盖
长度不符与元素不是文法两种拒绝；`test_unfinished_row_constraint_is_an_error` 用
永不完成的玩具文法断言错误信息带"第 1 行"；`test_batch_session_row_constraints_
match_plain_call` 断言 `LMRunner` 的复用路径与无状态入口逐 token 相同。

**边界**：约束的屏蔽与推进都在主机侧（`allowed` / `feed` 是 numpy 与纯 Python），
因此这条能力与后端无关；HIP 全量 2102 / 365 跳过只证明没有回归，不构成"设备侧
约束"的新证据。`generate_beam` 仍不接约束（束搜索的 k 条假设会把文法状态分叉到
k 个不同前缀，与 5.194 的"束搜索只支持单条 prompt"同一条边界）；`ServingEngine`
的 per-request 约束路径不变，两条入口现在语义一致。

### 5.207　反向图专用优化：按求导目标剪枝与交换律 CSE

**动机**：5.17 就把反向图定义成"再构一张普通 IR"。但"普通"只该指它能被任何后端
执行，不该指优化时只能套前向的默认管线。缺口有两个：`backward` 对每条可微路径都
调用 vjp，`wrt` 只取一个参数时另一个分支的梯度节点照样建出来（没有任何输出引用
它们）；反向图里 cotangent 的累加顺序由逆拓扑遍历决定，同一个值的两份贡献会以
`add(a, b)` / `add(b, a)` 两种形态出现，普通 CSE 按有序签名比较，认不出它们同源。

**做法一：按求导目标限定 vjp 的传播范围。** `backward(..., minimize=True)`（默认
`False`，既有构图行为不变）先在前向图上沿**消费者**方向做一次 BFS：从每个 `wrt`
值出发、能沿前向走到的值构成 `downstream`，逆拓扑循环只对 `downstream` 里的节点
调用 vjp。判据不是"梯度为零"而是"不在任何一条通往目标的链上"——被跳过的节点既
不会给任何被求导的值贡献 cotangent，也不改变留下来的节点的处理顺序，因此梯度
**逐位相同**。`minimize` 只接受 bool，`1` / `"yes"` 这类"看起来像真"的值显式报
`TypeError`：它改变的是构图范围而不是数值，隐式转换只会让调用方以为生效了。

**做法二：交换律规范化的 CSE。** `CommutativeCSE`（注册名 `cse-commutative`）把
`add` / `mul` 的签名按输入 id 排序后再比较，合并保留**先出现**的节点（输入顺序
不动）。浮点加法与乘法在 IEEE 754 下对有限值与 inf 逐位可交换，所以合并不改变
任何数值。它不进默认管线：默认管线的契约是"在任何图上都安全且值得做"，而互换
求和的合并只在反向图这类自动生成的图里常见。

**做法三：第三条管线。** `BACKWARD_PASS_NAMES = (simplify-shape-ops,
constant-folding, cse-commutative, dce)` 与 `backward_pipeline()`。不含融合：融合
会改变算子的输入形态、要求目标后端额外承接，而反向图的设计前提是"任何承接前向
的后端都能直接跑它"。

**前向重放不剪。** `minimize` 只减少 vjp 调用与反向节点，前向重放仍是整张 live
图：被剪掉的分支上的值可能正是某个 vjp 要读的输入（`add` 的 `sum_to` 就要读输入
的类型），"读值"与"求它的导数"是两件事。

**判据。** 12 条新用例：`tests/test_autodiff.py` 5 条（无关分支节点数下降且 `dx`
逐位等于 `2x`；中间值作目标时不再向上传播；`wrt` 覆盖全部输入时与默认路径完全
相同；`minimize` 后仍过 verifier；`minimize` 非 bool 的四类拒绝）、
`tests/test_passes.py` 7 条（普通 CSE 不合并互换加法；交换律版本合并且数值逐位；
`mul` 合并而 `sub` 不合并；按名注册；反向管线的名字序列与无融合；优化后逐位
不变；输出仍是合法可移植 IR）。量化：双分支玩具图（`loss = x*x + tanh(y)^2`）
只对 `x` 求导时节点数 14 → 7 → 2（默认 / `minimize` / 再过 `backward_pipeline`）；
MLP 对全部输入求导时 13 → 12（仅 DCE 清掉一个死累加），只对 `x` 求导时
`backward_pipeline` 把它从 13 降到 7。本地全量 2114 通过 / 409 跳过、WSL HIP
2114 / 365（跳过项不变）。HIP 只证明没有回归：这两条优化都在 IR 层，不碰内核。

**边界。** `minimize` 与 `active` 正交（前者管 vjp 范围，后者管种子与前向重放）；
图里存在没有 vjp 的算子且它不在任何目标下游时，`minimize=True` 不会报错——默认
路径会报 `NoGradientError`。这是有意的差别：那条路径本来就不会被计算。

### 5.208　训练态 BatchNorm 的 torch.export 导入

**背景**：5.26 定下 IR 的 `batch_norm` 是训练态（用当前 batch 的统计量，running
stats 由训练循环用 `batch_norm_stats` 维护），5.39 补的是推理态导入：
`aten.batch_norm(training=False)` 展开成 scale/shift 仿射式，训练态则被显式拒绝
（"请换成 eval 态导出"）。这条拒绝一直没有必要：训练态 BN 的**输出**只依赖 batch
统计量、`weight`、`bias`、`eps`，与 IR 的 `batch_norm` 逐项同义；被拒的其实是
running stats 更新，而那本来就不在张量语义里。

**导入口**：`aten.batch_norm(training=True)` → `batch_norm(x, weight, bias,
axis=1, eps=...)`。`momentum` / `running_mean` / `running_var` 不进图；`eps` 取第 7
个参数（缺省 1e-5）；要求通道维静态且 `weight` / `bias` 形状为 `(C,)`，否则显式
报错——这两条是 IR 自身的约束，前端不放宽。

**分解后的同一条路**：`ep.run_decompositions()` 把训练态 BN 拆成
`aten._native_batch_norm_legit_functional`（返回 `out, save_mean, save_invstd`）。
它走同一个翻译函数，只保留第 0 槽；取 1/2 槽的 `getitem` 若真被读到，`a()` 仍按
既有约定报"未支持的多输出槽位"，不静默给零。

**图输出要分类**：`torch.export` 会把 buffer 更新（`num_batches_tracked`）和
`save_mean` / `save_invstd` 挂到图输出上，分解后尤其如此。它们不是张量语义，
而且 `num_batches_tracked` 是 int64——原样翻译会在 placeholder 阶段撞上"不支持的
dtype"。现在入口先按 `graph_signature.user_outputs` 把输出槽位分成用户输出与
其余，其余连同**只服务它们的生产者闭包**一起不翻译；用户输出的可达闭包同时
决定哪些 placeholder 需要。非分解图里 0 使用者、首参是 buffer 的原地写
（`add_(num_batches_tracked, 1)`）与 0 使用者的 `getitem` 仍按"没有可观测效果"
跳过。

**契约**：导入图**不含** running stats 更新。要统计量，导出侧得在训练循环里显式
算 `batch_norm_stats`（5.26）；这不是降级，图里从来就没有这一步，训练态输出也
不依赖它。另外训练态 BN 的 `running_mean` / `running_var` 是图输入但不在输出
依赖路径上，`backward()` 默认 `wrt=全部输入` 会按 5.19x 的契约报
`NoGradientError`——要梯度就显式挑 `wrt`（x、weight、bias），这是既有契约的
正常表现，不是新限制。

**判据**：3 条新用例（训练态前向与 eager 逐位到 1e-6；`run_decompositions()`
之后同一条路也逐位；导入后 `dx` / `dweight` / `dbias` 与 torch autograd 对拍
1e-4。三条都断言 `user_inputs == ["x"]`，即 int64 的 `num_batches_tracked` 没有
漏成用户输入）。本地全量 2117 通过 / 409 跳过、WSL HIP 2117 / 365。torch 版本
2.13.0。

**边界**：`momentum` 被忽略（只影响 running stats 的更新，不影响输出）；目前只开
`aten.batch_norm` 与 `_native_batch_norm_legit_functional` 两个入口，三输出的
`aten._native_batch_norm_legit` 原语仍走显式拒绝，遇到真实模型再加。

### 5.209　索引/布局抽象铺到生产内核

**背景**：5.177 的 `Layout` 已经把「多维坐标 ↔ 扁平下标」收成纯函数，但只有
`_image_slice`（`(n*C+c)*H*W`）、`_tap`（`ih*W+iw`）、`_split_output_index` 三处
在用；其余 19 处仍是手写 `a * stride + b`。重复本身不致命，问题是同一套行主序
规则散在 19 个地方：改一次规则（换列主序、加 padding 步长）必然漏几个，而漏掉
的那几个只有数值测试才发现。

**做法**：五个模块级布局常量，覆盖生产内核里反复出现的二维形态——
`_ROW_COLS`（`(r,n)` / `cols`，5 处）、`_T_ELEMS`（`(t,i)` / `elems`，7 处）、
`_C_ELEMS`（`(c,i)` / `elems`，3 处）、`_S_ROWS`（`(s,r)` / `rows`，2 处）、
`_T_WIDTH`（`(t,k)` / `width`，2 处），共 19 处换成
`Layout.offset((...), cast=None)`。`cast=None` 是关键：默认会做「首因子提升到
64 位」，而这段代码原本不做，打开就多一层 `(long long)`，不再逐 token 相等。

**判据**：先对 57 个内核工厂的 CUDA 源码取 SHA-256 作基线，替换后再算一遍，
**0 处差异**（12 个在 `float32` 下渲染报错的是 int 模板，未受影响，错误类型也
一致）。4 条新用例直接看渲染后的源码：`sparse_linear` 必须出现
`r * cols + n` 且不含 `(long long)r * cols`，`scan_linear` 出现 `t * elems + i`，
`spike_linear` 出现 `s * rows + r`，五个常量确实是 `Layout` 实例。本地全量
2121 通过 / 409 跳过、WSL HIP 2121 / 365。

**边界**：只收「二维、坐标齐全、行主序、无 64 位提升」这一形态。带 `Paren`
的前缀乘积（`(n*C+c)*H*W`）、需要提升的乘积、以及跨步不连续的布局仍手写——
`Layout` 不表达 strides，这次也不加：加了就要处理「步长从哪来、谁来保证连续」，
那是 `plan.py` 与各内核的职责（5.5x 的边界）。分块、向量化、合并访存同理，仍不
进这一层。

### 5.210　复数 dtype 与 FFT

**问题**：IR 只有实数与整数 dtype，`fft` 这类"输入是一组复数、输出还是复数"的
算子无处安放。把复数拆成 `(..., 2)` 的实数张量看似绕开 dtype，但它把"相位"和
"实虚部"混在同一根轴上，形状推导、`norm` 语义与后端映射全都要各自解释一遍，
而且用户会在 `fft` 与 `rfft` 之间反复手写拆合。缺的这一层是类型系统，不是算子。

**做法**：

1. `DType` 增加 `C64` / `C128`（numpy `complex64` / `complex128`），并给出
   `is_complex` / `is_float` 两个判定。dtype 名同时进入"形状符号禁用表"，
   因此 `c64[c64, 4]` 与 `f32[f32, 4]` 一样在构图期被拒。
2. `fft`：复数进、复数出，沿最后一维，`inverse=True` 走逆变换；`rfft`：f32/f64
   进、c64/c128 出，最后一维 `n -> n // 2 + 1`（符号维写成 `T//2+1`）。
   `norm` 只收 `backward` / `ortho` / `forward`，与 `numpy.fft` 同义；非法值、
   rank 0、`inverse` 非 bool 都在构图期显式报错。
3. **不接受隐式升格**：实数进 `fft` 直接 `TypeError`，提示改用 `rfft` 或先
   `cast`。反过来，执行入口拒绝"声明实数却喂复数"——虚部不能悄悄丢掉。
4. 参考语义就是 `numpy.fft`，conformance 新增 3 条（正变换、逆变换、半谱），
   参考后端 95 executed。
5. `backends/cuda/cufft.py`：`cufftPlanMany` 的一维批量变换。覆盖表只登记
   c64/c128（C2C/Z2Z）与 f32/f64（R2C/D2Z），其余 dtype 与 `rfft(inverse=True)`
   在碰驱动前就被 `CufftUnsupported` 拒掉；plan 键只含 `(类型, 长度, batch)`，
   方向是执行参数、不进键，同一 plan 正反共用；输入必须连续，非连续显式拒绝。
6. **归一化是这一层唯一的"语义"**：cuFFT 两个方向都不除 N，而 numpy 的
   `norm` 三档要求不同缩放。`normalization(size, norm, inverse)` 是纯函数表——
   正向 backward 1、ortho `1/√N`、forward `1/N`，逆变换 backward `1/N`、
   ortho `1/√N`、forward 1——结果上原地乘一次。漏掉这步就是差 N 倍的错误，
   所以它必须有独立用例，而不是靠端到端数值测试兜底。
7. CUDA 侧：`invoke` 在 dtype 门禁之前接住 `fft` / `rfft`；`capabilities()` 在
   `cufft.available()` 为真时才声明这两个名字——库缺失时宁可让一致性套件记为
   "未声明算子"，也不声明一个执行时才失败的算子。捕获期显式拒绝
   （`cufftPlanMany` 是 host API）。
8. HIP 侧不新写 rocFFT 绑定。但 HIP 容器原先连带承载都不允许复数，错误会在
   拷贝阶段变成裸的 `HipRuntimeError`；把 `complex64` / `complex128` 加进容器
   承载面后，拒绝发生在 `invoke` 的"未实现算子"分支，是一致性套件认识的
   `BackendCapabilityError`——与 f64 / bool 的处理同一条原则。

**边界**：只做沿最后一维的一维变换。多维 FFT 需要新的轴属性与新的参考语义，
不在这步顺带扩。`fft` 没有 VJP：复数线性算子的伴随是共轭转置，IR 里还没有
`conj`，求导会明确抛 `NoGradientError` 而不是给一个错误的实数梯度。
cuFFT 只有离线验证（本机没有可用 NVIDIA 显卡），真机数值对拍留给 CUDA 恢复时。

**用例**（23 条）：dtype 别名 / numpy 承载 / 实数进复数无损 / 复数进实数被拒；
正变换与三档 `norm`、逆变换往返、`rfft` 的形状与数值、符号维实例化、无 VJP、
序列化往返；5 类构图期拒绝；cuFFT 的归一化表与非法参数、dtype 登记表与拒绝面、
plan 键、`available()` 不抛异常、库缺失时的显式错误。

### 5.211　流水调度与 ZeRO 通信

**问题**：5.205 的流水只有一种排布——每个 micro-batch 走完全部段的前向与
反向，再轮到下一份（GPipe 的朴素形态）。段间激活因此"一份活着、一份算了就
放"，驻留是 1，但反向必须等该份的前向全部结束；而训练里真正被人期待的是
"前向的下一份"与"反向的上一份"交替推进，让**同时驻留的段间激活**压到段数
减一，而不是随 micro-batch 数线性堆积。另一侧，ZeRO-1 的梯度同步仍是
All-Reduce：每个 rank 收整份梯度（通信量 `W·N`），可它真正要用到的只有自己
分到的那一片（`N`）。

**做法**：

1. `PipelineTrainer(schedule=...)` 三档，默认 `sequential`，既有行为逐位不变：
   - `sequential`：一份前向完立刻反向，驻留 1；
   - `gpipe`：全部份前向完再全部反向，驻留 = 份数；
   - `1f1b`：预热 `min(段数-1, 份数)` 份，之后"反向一份、前向下一份"交替，
     驻留 = `min(段数-1, 份数)`。
   非法取值在构造期 `ValueError`，并把三档的含义一起写进报错。
2. 把原 `_micro_step` 拆成 `_forward_micro` / `_backward_micro`：前向留下一份
   `{index, piece, activations, loss}`，反向逆序消费并在结束时清空
   `activations`。三档调度只是以不同顺序调用这两个函数，梯度累加、种子权重、
   `optimizer.step` 与参数写回全部沿用同一条路径——**调度不改变数值定义**。
3. 驻留用 `_ResidentTracker` 记"前向已完成、反向还没做"的份数，每步写进
   `last_resident_peak`。为什么不记字节：同一段激活的形状由图的静态形状决定，
   调度决定的量是**有几份同时活着**；记账要点选成"份数"，测试才能把
   `gpipe == 份数`、`1f1b == min(段数-1, 份数)` 直接钉成断言。
4. `stages == 1` 时 1F1B 没有段间交替，退化成逐份前向 + 反向（否则预热窗口
   为 0、`waiting.pop(0)` 立刻 `KeyError`）。这条退化分支有独立用例。
5. `train/sharded.py` 新增 `reduce_scatter_grads(grads_per_rank, params, ...)`：
   按 `shard_params` 的分片规则，每个 rank 只接收本片的名字，累加顺序与
   `allreduce_grads` **完全相同**（按 rank 序号依次加），因此结果与 All-Reduce
   逐位一致；`reduction="mean"` 时同样在累加完再除。空片不调用 `transport`
   （世界大小大于参数个数是合法状态），`transport` 的返回值必须恰好等于本片
   名字集，否则 `KeyError`。
6. 显式拒绝面：`world_size` 与实际份数不符、`reduction` 非 `mean`/`sum`、
   空序列、某 rank 缺名字或多出名字、形状与 `params` 不一致、dtype 混用，
   全部在进入通信前抛错——这一层不做隐式补齐。

**边界**：后端仍是单进程串行切卡（5.190），所以"重叠"只发生在**顺序**层面：
1F1B 交换的是"哪一份先算"，不是线程并发，墙钟不会有流水线加速。收益是可
验证的两条——数值与 GPipe 在容差内一致（`rtol=1e-6`）、驻留份数按调度下降。
`reduce_scatter_grads` 省的是通信量，不是显存：参数与槽位仍按 5.205 的分片
布局，ZeRO-2/3 的参数分片不在这一层。

**本轮明确不做的一件事**：`spike_linear` 的 `dspikes` 事件化。前向的事件表示
（CSR）只覆盖"发放了的突触"，很自然会想把它推广到反向；但 `spike` 的替代
梯度在**未发放位置同样非零**（只要膜电位落在三角窗内），把 `dspikes` 只散回
非零事件，等于把这些位置的梯度永久丢弃——初始层一次都不发放时完全学不动。
稠密 `(S, K)` 的 `dspikes` 因此是**语义要求**，不是可以省的中间值；
`ir/grad_rules.py` 的 `_spike_linear` 已把这条写成注释，权重梯度
（`spike_linear_weight_grad`）才走事件散射。要改只能先换掉替代梯度的定义，
那是换模型，不是优化。

**用例**（13 条）：三段小图下三档调度各跑 2 步、参数在 `rtol=1e-6` 内一致；
`sequential` 峰值 1、`gpipe` 峰值等于份数、`1f1b` 峰值等于段数减一、份数小于
段数时被份数封顶；单段 1F1B 的退化与参数确有变化；未知调度名被拒。
Reduce-Scatter 侧：3 rank 与 `allreduce_grads` 逐位一致且分片与 `assignments`
一致、`sum` 口径、接收总量从 `W·N` 降到 `N`（transport 只看到本片名字）、
分片 step 与整份优化器逐位一致、空片跳过 transport、7 类拒绝面。

### 5.212　审计缺陷修复：训练状态一致性、语义缝隙与契约边界

**问题**：一次外部审计在 11 处找到"能跑但语义不对"的缝隙，可以分成三类：
（a）训练状态的自洽性——AMP 的缩放 / 跳过顺序、累积窗口的权重与存档事务、
调度器与静态超参的混淆、分片优化器"校验一个提交一个"；（b）梯度与入口的
语义——门控 SiLU 漏乘上游、归一化参数被广播后的梯度形状、整数入口按位宽
而不是按可表示范围判断；（c）外围契约——签名硬件包重启后消失、fuzz 生成
整数除法时可能抽到零除数。它们共同的特征是：**没有异常、没有 NaN，只是数值
定义悄悄变了某一项**，所以只能靠针对性的对拍与"失败时状态不许变"的断言钉住。

**做法**（按修复项编号）：

1. **门控 SiLU**（`ir/grad_rules.py`）：`y = silu(x) * gate` 对 gate 的梯度
   必须是 `g * silu(x)`，原实现返回 `silu(x)`，丢掉了上游权重——全 1 seed
   时两个写法结果相同，所以复现用例额外乘一个权重输入。
2. **AMP 顺序**（`train/loop.py`）：非累积路径改为"反缩放 → 有限性判定 →
   梯度钩子 → 调度 → `scaler.update`"。原先裁剪钩子看到的是 scale 倍的梯度，
   同一个 `max_norm` 会随缩放历史漂移；溢出判定又被推迟到钩子与调度之后，
   被跳过的步白白消耗了一格学习率。
3. **累积窗口作废**（`train/grads.py`）：`discard()` 同时清零 `_weight`。
   只清累加和与计数会让下一个窗口的分母带上已作废的微批，梯度整体缩小。
4. **窗口权重进存档**（`train/loop.py`）：`snapshot()` 写入 `accum_weight`，
   恢复时按它重建窗口。原文只存 `count`，加权累积（`weight != count`）续训
   后分母退回微批数。
5. **恢复事务化**（`train/checkpoint.py` + `train/loop.py`）：新增只读入口
   `inspect_checkpoint(path)`，先把累积长度、随机流、`pending` 范围、
   `accum_weight`（存在且有限正）、附加数组键集合恰好等于
   `grad/<全部参数>`、AMP 三键齐全且取值合法全部校验完，再调用
   `load_checkpoint` 提交。原先校验排在提交之后，失败会留下"权重已换、
   窗口还是旧的"半加载现场。
6. **调度器与静态超参**（`train/optim.py` + `checkpoint.py`）：`load_state_dict`
   新增 `restore_lr`（默认 `False`）。为 `True` 时超参比较剔除 `lr`、校验后
   从存档恢复；`ShardedOptimizer` 与 `load_checkpoint` 转发该参数，`Trainer`
   按"有没有调度器"决定。带调度时 `lr` 是训练走到哪的一部分，不带时是静态
   配置——两种语义不能混成一种。
7. **硬件包信任记录**（`backends/hardware.py`）：安装验过的来源写到安装根目录
   的 `.trust.json`（`{name: {algorithm, public_key}}`），扫描时按记录把公钥
   重新作为信任锚。记录**不能**放进包目录，否则会变成摘要校验里的"未声明
   文件"。卸载与替换同步删除 / 覆盖记录；无签名替换必须清掉旧锚，否则扫描
   会拿旧公钥去要求一个本来就没有签名的包。
8. **分片先全局校验**（`train/optim.py` + `train/sharded.py`）：把 `step` 拆成
   `prepare_step` / `apply_step`，`load_state_dict` 拆成
   `_prepare_state_dict` / `_commit_state`；`ShardedOptimizer` 先对全部分片
   准备，再统一提交。原先逐片提交，后一片失败时前几片的步数与槽位已经换掉。
9. **归一化参数梯度形状**（`ir/grad_rules.py`）：layernorm 的 `dgamma`/`dbeta`
   与 rmsnorm 的 `dgamma` 在"对非归一化轴求和"之后再用 `sum_to` 归约到参数的
   实际形状。`gamma`/`beta` 广播成标量时，原实现给出 `(C,)`，优化器的形状
   校验会直接失败。
10. **整数入口范围**（`runtime/executor.py`）：把"源类型更宽才验范围"改成
    "目标能表示源的全部取值才跳过"，否则运行期验 min/max。`uint32 -> int32`
    位宽相同却有一半取值会包成负数。
11. **fuzz 整数除法**（`fuzz.py`）：整数 dtype 且算子为 `div` 时，除数改用
    显式常量 `2`（同 dtype、同形状）。池子里的整数可能是全零张量，抽中它
    当除数会让"合法图 + 合法输入"抛除零错误。

**边界**：这一轮全部是纯 Python 路径，没有新增算子、没有改后端内核；设备侧
数值不受影响，HIP 全量复跑用于确认这一点。`restore_lr` 默认 `False` 意味着
既有"超参必须完全一致"的契约不变，只有训练器在有调度器时才放宽 `lr` 一项。
信任记录只保存公钥，不保存"已验证"的结论：扫描仍会重新验签与验摘要，改过
内容的包照样消失。事务化的范围是"失败时对象状态不变"，不是"失败可重试"——
调用方仍需重新构造 Trainer 再加载。

**用例**（20 条，全部纯 Python）：门控 SiLU 的非均匀 seed 有限差分（1）；
AMP 钩子看到反缩放后的梯度、溢出不推进调度（2）；`discard` 清零权重、加权
窗口跨存档续训（2）；配置校验失败时参数 / 步数不变、缺 `accum_weight` 与
附加键不符都在提交前被拒、无 scaler 加载 AMP 存档不改参数（3）；`restore_lr`
的严格 / 恢复 / 非法 lr 三条分支、带调度续训后参数逐位一致（2）；签名包安装
后可被重新发现、装后篡改仍隐藏、卸载清记录、无签名替换清旧锚（4）；分片某片
梯度非法不留半步、某片存档非法不留半加载（2）；标量 `gamma`/`beta` 的梯度
形状为 `()`（2）；`uint32 -> int32` 越界报错且范围内正常、可全量表示时不报错
（1）；整数 `div` 用非零除数（1）。

### 5.213　BF16 闭环：真实量化、两字节驻留与 autocast / 训练路径

**问题**：审计报告把 bf16 列成高优先级缺口，理由是"参考表示借用 f32、autocast
不接受 bf16、没有形成存储 / 计算 / 训练闭环——枚举存在不能当作真实支持"。
在此之前 `DType.BF16` 只是一个标签：`np_dtype` 映射到 f32，值一点没变，任何
"bf16 之后精度应下降"的断言都无从写起；`Autocast` 的 dtype 只认 `f16`，训练层
更没有 `mixed_precision="bf16"` 这条路。这一节把标签变成语义。

**做法**（四层，从数值到训练）：

1. **量化**（新增 `runtime/bf16.py`）：`round_bf16` 按位做 RNE——取 f32 位模式
   `bits += 0x7FFF + ((bits >> 16) & 1)` 再 `& 0xFFFF0000`，恰好一半时进位到
   偶数尾数；inf / nan 因指数全 1 且低 16 位被截掉而原样保留（NaN 的位模式
   不参与进位——偏置加到尾数全 1 的 NaN 上会把它抬成 0；截断后尾数为 0 的
   NaN 补一个 quiet 位，否则会被读成 inf），f32 最大有限值
   在 RNE 下进位成 inf（这是位宽决定的语义，不是 bug）。另有两个 2 字节打包
   入口：`pack_bf16` 取高 16 位存成 `uint16`，`unpack_bf16` 左移 16 位还原
   f32 载体，`unpack_bf16` 对非 `uint16` 直接 `TypeError`（避免把 int32 位模式
   静默当成 bf16）。
2. **执行**（`runtime/executor.py`）：新增 `Executor.quantize(value, dtype)`，
   bf16 且是 ndarray 时走 `round_bf16`，其余原样返回。调用点只有两处——
   **输入对齐之后**、**节点输出 `cast` 之后**，正好是 IR 里 bf16 类型的两个边界。
   计划缓存 `_GroupPlan` 增加 `out_dtype`（IR 的 `DType`）与既有的 `dtype`
   （`np.dtype`，即载体 f32）并列：只看后者会把 bf16 和 f32 混为一谈。`cast`
   本身仍按载体做位宽转换，量化交给 `quantize`，两者不重叠。
3. **autocast**（`passes/autocast.py`）：`_FLOAT_ORDER` 从"f16/f32/f64"扩成
   `f16 < bf16 < f32 < f64`，理由是尾数位数（10 / 7 / 23 / 52）；`force_f32`
   分支泛化成"宽于 f32 才保留"，即 f16 与 bf16 的纯低精度子图都回收成 f32，
   f64 不被降级。bf16 与 f16 共用同一张算子白名单，区别只在数值：bf16 的指数
   与 f32 同宽，不需要 `GradScaler`。
4. **训练**（`train/loop.py`）：`Trainer(mixed_precision="bf16")` 把低精度入口
   声明成 bf16；`master_weights=True` 时主机态驻留缓冲是 `uint16`（`pack_bf16`），
   主副本仍是 f32，喂图前 `unpack_bf16` 还原。拒绝面显式：bf16 带 `scaler`
   报错（指数范围同 f32，缩放只会污染数值）、bf16 带 `device_state` 报错
   （设备后端只接受 f32 / f16，不做静默降级）、`master_weights` 没有低精度
   目标报错。存档 meta 增加 `mixed_precision`，`load_training_state` 在提交前
   校验一致，bf16 存档接进 f16 训练器会被拒。

**数值**：240 步线性回归（`Adam(lr=0.05)`，bf16 入口 + 主副本 vs 全 f32）后，
权重相对误差 `w = 2.17e-3`、`b = 1.71e-3`；驻留缓冲 `w` 从 16 B 降到 8 B、
`b` 从 4 B 降到 2 B，正好一半。RNE 的边界值：`1 + 2^-8 → 1`、
`1 + 3·2^-8 → 1.015625`（都进位到偶数尾数）、`1 + 2^-8 + 2^-20 → 1.0078125`。
`1e30` / `1e-30` 在 bf16 下都保持有限（f16 下分别是 inf 与 0），这是选 bf16
而不用 f16 的直接理由。

**用例**（23 条，全部纯 Python）：codec 6 条（RNE 的五个位模式、非有限与溢出、
NaN 位模式不落成 0、2 字节打包往返与字节数、非 `uint16` 拒绝、范围宽于 f16）；执行 4 条（输入边界
量化、`cast` 保持 IR 标签并量化、`matmul` 的输入与输出都量化、图输出恢复 f32）；
autocast 4 条（白名单结构与敏感算子留在 f32、前向在 bf16 精度内对拍、`low_precision_inputs`、
幂等）；训练 6 条（两字节驻留、240 步收敛且在 bf16 精度内、`refresh_resident`、
无 master 时参数保持 f32、存档 meta 与 f32 参数、跨精度恢复被拒）；拒绝面 3 条
（`scaler`、`device_state`、`master_weights` 无低精度目标）。

**负向对照**（做完还原）：把 `quantize` 的 `if dtype is DType.BF16` 改成
`if False`，三条边界用例失败——`matmul` 实际 `-2.4085`、期望 `-2.421875`，
输入量化那条退化成 f32 的全值；把 `_to_resident` 的 bf16 分支改成直接返回 f32，
驻留用例 FAIL、4 条训练用例 ERROR（`unpack_bf16` 拒绝 float32）。说明这两处
不是"写了就过"的装饰。

**检修补记**：`round_bf16` 的第一版把 RNE 偏置无条件加到所有位模式上，尾数
全 1 的 NaN 会进位溢出——`0x7FFFFFFF` 量化成 `-0.0`、`0xFFFFFFFF` 量化成
`+0.0`。训练时一旦出现 NaN，`master_weights` 打包会把它静默抹成零，等于把
"发散了"改成"权重是 0"。现在指数全 1 的值只截断尾数，截断后尾数为 0 的
NaN 再补 quiet 位，`test_nan_payload_survives_quantization` 在旧实现下失败。

**验证**：本地 `Ran 2215 tests / OK (skipped=410)`；WSL HIP（RX 7650 GRE /
ROCm 7.2）`Ran 2215 tests / OK (skipped=369)`。

**边界**：bf16 的 2 字节只覆盖**主机态参数驻留**（master weights），中间值仍是
f32 载体、只是每个 bf16 边界被量化，所以它不是"全流程 bf16 计算"；图输出按声明
类型恢复 f32。设备后端（CUDA / HIP）没有 bf16 内核——M40 是 sm_52、2080 Ti 是
sm_75，PTX 的 bf16 转换要 sm_80+——因此 `device_state=True` 与 bf16 组合直接报错，
而不是退化成 f32 假装支持。autocast 的白名单与 f16 相同，含义是"这些算子在
低精度下可接受"，不是"bf16 有专属算子集"。

### 5.214　完整续训：数据迭代位置与训练配置凭证

**问题**：审计的「还缺什么」里「完整续训」一条指的不是权重接不上——5.27 起
续训轨迹就与未中断逐位一致——而是两处**没有可判等的凭证**：

1. **数据顺序只到轮次边界。** `DataLoader` 只有 `epoch` 与 `reset()`，一轮
   吃到一半存档，恢复只能从轮首重跑，要么重复样本、要么重跑浪费时间；而
   "从第几批接上"需要先知道"这一轮怎么洗的牌"，只存游标不存配置是不够的。
2. **模型图与调度器没有身份证。** `load_checkpoint` 校验参数名、形状与 dtype，
   但把激活从 tanh 换成 sigmoid、或把 `step_size=2` 改成 3，这几项全都对得上，
   存档照接，轨迹却已经换了一条。

**做法**（`train/data.py` + `train/loop.py` + `train/schedule.py`）：

1. **批游标**：`DataLoader` 新增 `_cursor`，记录**当前轮已经产出**的批数。
   `__iter__` 在游标非 0 时用 `epoch - 1` 重建同一份置换、从该游标继续；完整
   走完一轮才归零。生成器被提前 `close` 或调用方拿了两批就不要了，抛的是
   `GeneratorExit`，游标保持原值——这正是"中途存档"需要的语义。
2. **位置协议**：`state_dict()` 存 `epoch / batch / seed / batch_size / size /
   shuffle / drop_last`；`load_state_dict()` 先校验键完整、位置合法
   （`0 <= batch <= len`，且 `batch > 0` 时 `epoch >= 1`），再逐项比配置，
   任何一项不同都在**改动状态之前**报错。拒绝而不是"尽量对上"：同一个游标在
   另一套 seed / batch_size 下指向的是另一批样本，静默恢复会把数据顺序换掉，
   而这种偏差在损失曲线上看不出来。
3. **图指纹**：`Trainer.graph_fingerprint()` 对 `serialize.dumps(graph)` 取
   sha256 前 16 位。参数集合与形状之外再钉住"计算图本身"；autocast 会改图里
   的 dtype，因此指纹也隐含了精度信息（与 `mixed_precision` 的校验互为冗余，
   不是替代）。
4. **调度器凭证**：四个内置调度器各自实现 `hyperparams()`——
   `CosineDecay`（base_lr / total_steps / min_lr）、`StepDecay`（step_size /
   gamma / min_lr）、`ExponentialDecay`（gamma / min_lr）、`Warmup` 递归记录
   内层的 `kind` 与超参。没有 `hyperparams()` 的自定义调度器只比类型。
5. **meta 与校验**：`graph_fingerprint` / `schedule_kind` / `schedule_config`
   三个键（配置是键排序、无空白的 JSON 串）以 `setdefault` 写进快照；
   `load_training_state` 在优化器提交前校验，旧存档缺这些键时放行——与
   `mixed_precision` 同一条兼容策略。

**用例**（14 条，纯 Python）：数据 6 条（中途恢复后剩余批逐位一致、恢复后
下一轮顺序与未中断一致、走完回到轮次边界、提前放弃迭代器保留游标、配置不一致
拒绝且状态不变、缺键与越界位置拒绝）；凭证 8 条（同图同指纹 / 改图不同指纹、
往返记录三项、换图拒绝、换调度器类型拒绝、同类型换超参拒绝、`Warmup` 内层
更换拒绝、有 / 无调度器拒绝、续训五步参数逐位一致）。

**负向对照**（做完还原）：把图指纹校验关掉，`test_graph_change_is_rejected`
从"报错"退化成"静默加载"而失败；把 `__iter__` 的 `resume` 写死成 0，
2 条恢复用例失败（剩余批从轮首重新开始）。说明两处都不是"写了就过"的装饰。

**验证**：本地 `Ran 2215 tests / OK (skipped=410)`；WSL HIP（RX 7650 GRE /
ROCm 7.2）`Ran 2215 tests / OK (skipped=369)`。

**边界**：数据位置由调用方把 `state_dict()` 放进 meta、恢复时自行
`load_state_dict()`——`Trainer` 不持有 loader，框架不替用户决定数据流怎么接；
`hyperparams()` 只做判等、不参与学习率计算，`lr` 刻意不在凭证里（它随调度器
变化，由优化器的 `restore_lr` 负责）；旧存档没有这三个键时放行，这是兼容的
代价，不能当作已验证。

### 5.215　训练态 BN 的 running stats：从训练输出对齐到 eval 状态对齐

**问题**：审计指出的不是"BN 算错了"，而是**两种状态之间的缺口**。IR 的
`batch_norm` 是训练态语义：只用当前 batch 的均值/方差归一化，输出只依赖
`x / weight / bias / eps`——这四样都在图里，所以 5.208 的导入与 5.39 的
端到端对拍都是对的。但 eval 用的是**整段训练累积下来的** running mean/var；
torch.export 留在图里的 `mul_` / `add_` 对 buffer 的原地写在导入口被跳过，
而通用 `Trainer` 也没有任何地方维护这对累积值。于是"训练输出对齐"通过、
"训练后 eval 状态对齐"却没人检查：拿 `bn_infer` 或推理图去跑，用的还是初始
的 `(0, 1)`，缩放系数系统性偏错。这不是数值精度问题，是闭环缺了一段。

**做法**（`train/bn.py` + `train/loop.py`）：不要求模型改构图，也不往图里
塞"更新 buffer"的节点——IR 没有图外状态，BN 的 running stats 属于训练策略。

1. **找站点**：`bn_sites(graph)` 扫描 `batch_norm` 节点，记下通道轴、通道数、
   每个通道的样本数（无偏修正要用）与输入的 Value id。形状必须静态：符号维
   算不出"每个通道多少个样本"，这类图在构造期就报错并指出是哪个节点。
2. **统计子图**：`build_stats_graph(graph, sites)` 按 BN 站点的依赖闭包重建
   一张小图，复制到 BN 输入为止的节点（常量按原 attrs 重建），对每个站点加
   一个 `batch_norm_stats` 输出。不重跑整张网络：`reduce_sum` / 损失那段不在
   闭包里，用例直接钉住这一点。
3. **EMA**：`RunningStats` 每步按 `running = (1-m)·running + m·batch` 更新。
   方差用**无偏**估计（`n/(n-1)`）——归一化用的是有偏方差，但累积到
   `running_var` 的必须是对总体方差的无偏估计，否则 eval 的缩放系数会系统性
   偏小；`n == 1` 时退回有偏值以避免除零，与 torch 一致。
4. **接进 Trainer**：`Trainer(bn_momentum=0.1)`（torch 的默认值）。每步前向
   之后执行一次统计子图——**与参数是否真的更新无关**：累积窗口未满、AMP
   溢出跳过更新时，BN 观察到的数据照样要进 EMA。`bn_momentum=None` 是显式
   关闭（图里没有 BN，或调用方自己维护），此时 `bn_running_stats()` 抛错而
   不是返回初始值——返回 `(0, 1)` 会让 eval 永远算错且看不出来。
5. **接推理**：`Trainer.bn_running_stats()` 交出 `{站点: (mean, var)}` 副本，
   直接喂 `bn_infer` 的 `m` / `v` 两个输入，就得到与 torch eval 同语义的输出。
6. **凭证**：快照的 aux 写 `bn/mean/<站点>` / `bn/var/<站点>`，meta 写
   `bn_momentum`（站点集合由 5.214 的图指纹覆盖）。`load_training_state` 在
   提交前校验三件事：momentum 相同、键集合完全一致、图里没有 BN 时存档也不许
   带这些键；`extra` 模板按"这份存档真的会有的条目"构造（未提交的累积窗口
   不写梯度模板），所以 pending=0 时 BN 键是唯一一组。

**用例**（9 条，纯 Python）：EMA 与手写公式（含无偏方差）逐项对拍；参数
没更新时统计量照进；训练出的 `(mean, var)` 喂 `bn_infer` 与手写仿射逐位一致；
统计子图不含 `reduce_sum`；`bn_momentum=None` 时取用报错；`1.5 / -0.1 / True`
三个非法 momentum 拒绝；设备态 + BN 显式拒绝；符号通道拒绝；存档往返后
续跑一步仍与不中断一致，换 momentum 拒绝。

**负向对照**（做完还原）：把无偏修正改成 `× 1.0`，`test_running_stats_follow_momentum`
失败（方差系统性偏小）；注释掉 `step()` 里的 `_observe_bn` 调用，EMA 与
"训练到推理"两条用例失败。说明这两个环节都不是装饰。

**验证**：本地 `Ran 2224 tests / OK (skipped=410)`；WSL HIP（RX 7650 GRE /
ROCm 7.2）`Ran 2224 tests / OK (skipped=369)`。

**边界**：设备态训练暂不支持——捕获计划要求每步地址固定，而 running stats
是主机侧 EMA；含 BN 的图在 `device_state=True` 时构造期就拒绝，并给出两条出路
（主机态，或显式 `bn_momentum=None`）。`num_batches_tracked` 不维护（只支持
固定 momentum，不支持 torch 的 cumulative 模式）；统计子图每步多一次
`Backend.execute`，代价是"BN 之前那段"的重复计算，而不是整网。旧存档没有
`bn_momentum` 时拒绝加载——那批存档的 eval 状态从未被维护过，静默接上等于
把"从没对齐过"当成"已验证"。

### 5.216　训练开关的组合矩阵：单开都对，叠在一起还得对

**问题**：5.207–5.214 每个开关都有自己的验收（累积、加权、AMP、裁剪、调度、
存档、重算、主副本、设备驻留），但"同时打开"只有少数偶然的组合被测到。把
窗口长度从 2 改成 3 而手算对照照样绿，就说明那条测试只是在复述实现。组合
矩阵要回答的是：累积 → 加权 → 裁剪 → 调度的顺序，在 AMP 缩放与两条执行
路径（整图 / 分段 / 设备捕获）之间是否还是同一套语义。

**做法**（`tests/test_train_combinations.py`，只加测试、不改实现）：

1. **手算对照**：`accumulate_steps=2`、权重 `(3, 5)`、全局范数裁剪 `0.05`、
   `StepDecay`、`GradScaler(8)`。另起一个 trainer，同一初始参数、同样的图与
   输入，钩子只捕获窗口梯度并返回 `None`（不更新）。实测更新必须等于
   `initial - lr_at(0) * clip(captured)`。这条同时钉住三件事：缩放乘在种子上
   又被原样除掉、裁剪作用在**窗口平均后**的梯度上、调度用优化器真实更新次数。
2. **窗口一半存档续训**：全开配置跑 2 个微批（提交一次）存档，换 trainer 接着
   跑 3 个微批；未提交窗口的权重、`GradScaler.scale`、调度步数、裁剪上限必须
   一起接回去，轨迹与不中断逐位一致。
3. **分段重算**：同一组开关、`segments=2` 与整图反向逐位一致——重算只改
   "中间值怎么存"，不该改梯度。
4. **f16 驻留 + f32 主副本**：与全 f32 的轨迹在量化误差内（`rtol=2e-2`），
   覆盖"只更新时量化一次"与累积/裁剪/调度的叠加。
5. **设备驻留**：`_FakeDeviceBackend` 把"捕获缓冲 + replay"如实演一遍，同样
   五开关下与主机态逐位一致。这里踩到一个**既有契约**：设备态的
   `ParameterStore` 在 `sync_to_host()` 之前是旧值，直接拿 store 比较会得到
   两份完全不同的参数（诊断时首步差 10 个量级，而 `_dev_params` 与主机态逐位
   相同）。测试因此显式同步后再比——这也说明"设备态参数什么时候可读"应该在
   用例里写出来，而不是靠读者记得。

**用例**：5 条，全部纯 Python；单次运行约 0.2 秒。断言同时检查
`optimizer.step_count`、`scaler.skipped` 与每条路径的最终参数。

**负向对照**（做完还原）：把 `_all_knobs` 的 `accumulate_steps` 从 2 改成 3，
手算对照与续训对照一起失败（更新次数从 1 变 0/2）；不调用
`device.sync_to_host()` 时设备态对照失败 48/48 个元素。两次都说明这些断言
不是"跟着实现走"的空转。

**验证**：本地 `Ran 2229 tests / OK (skipped=410)`；WSL HIP（RX 7650 GRE /
ROCm 7.2）`Ran 2229 tests / OK (skipped=369)`。

**边界**：这是交互矩阵，不是笛卡尔积——`rng_inputs`、bf16、BN running stats
与 MoE 等开关不在这一组里（bf16 见 5.213，BN 见 5.215）。f16 的容差是量化
上界而不是"应该一样"；设备态替身后端不模拟显存，只在主机内存里如实演一遍
捕获与 replay，所以它验证的是调度与数值，不是容量。

### 5.217 动态形状契约：七个环节各自说清楚"支不支持"

**问题**：符号维在 IR 里是一等形状——`f32["B", 4]` 能构图、能序列化，执行器按
实际输入绑定后能跑不同 batch。但下游各层的支持面并不相同：`attention` 的反向
能带符号批维，`reduce_sum` 的广播与 `reshape` 的目标形状却必须在构图期拿到具体
数字。此前这些边界只散落在三处零散的拒绝里（`Trainer` 构造期拦符号输入、
`reshape` 拒绝 `SymExpr`、`deploy` 要求权重静态），读者要从代码里拼出一张表；
torch 导出侧的 `_shape_of` 更直接 `int(dim)`，动态维会被一次 guard 具体化成
"碰巧这次成立"的数字，导出的图对别的形状不再成立——这是静默降级。

**做法**：

1. 新增 `ir/shape_contract.py`：`StageSupport(stage, label, dynamic, reason)` 与
   只读的 `DYNAMIC_SHAPE_SUPPORT`，七个环节各有一个状态。四档含义固定：
   `full`（有覆盖用例）、`per-op`（由算子决定）、`data-only`（数据输入可符号，
   权重必须静态）、`rejected`（入口统一拒绝）。`support_for` 对未知环节直接
   `ValueError`，不默认放行。
2. `require_static_shapes(graph, *, stage, hint="")` 是 `rejected` 环节的统一守卫：
   先查表——把 `full` / `per-op` 环节传进来会直接报错，避免误拦本来可用的图；
   再列出每个含符号维的输入及其类型，错误消息带环节中文名、契约表里的理由与
   调用方建议。`Trainer` 与 `CheckpointPlan` 的构造期检查都改走它，
   `Trainer` 的重算提示保持原样。
3. `frontend/torch_export.py` 的 `_shape_of` 对动态维显式抛
   `UnsupportedOperatorError`，并提示先用 `torch._dynamo.mark_static` 固定该维。
   框架不 import torch，这条提示只是给调用方的。

**用例**（`tests/test_shape_contract.py`，13 条）：

1. 契约表：七个环节齐全、状态合法、理由与中文标签非空；映射只读；未知环节
   `support_for("serving")` 报错；把 `execute` / `autodiff` 交给守卫会被告知
   "不应调用"（防误用本身就是契约的一部分）。
2. `symbolic_inputs` 只列符号输入：裸符号与 `sym_mul("B", 2)` 都算，静态输入
   不出现。
3. 训练循环：符号图构造 `Trainer` 报错含「训练循环环节」「符号维」「%x」与
   `drop_last` 建议；静态图通过守卫。
4. 分段重算：符号图 `CheckpointPlan` 报「分段重算环节」；静态图照常构造。
5. `full` 的声明可执行：同一张 `x["B",4] → silu` 图在 batch 1/3/7 上各跑一次，
   `execute` 的"一次构图多形状"不是空话。
6. `data-only` 的声明可执行：`x["B",4] * w[4]` 存成 `.afpkg`、加载后喂 `(3,4)`
   得到 `(3,4)`；符号权重 `w["N"]` 被 `BundleError` 拒绝。
7. `per-op` 的边界：`reshape(x, shape=(sym_mul(4, "B"),))` 报「符号表达式」——
   `(B,4) → (4*B,)` 这个目标 IR 表达不了，必须在构图期拒绝而不是算错。
8. torch 导入：假 `SymInt` 触发「动态维」；静态形状照常返回 `(2, 4)`。

**负向对照**（做完还原）：把 `require_static_shapes` 短路成直接返回，训练循环与
分段重算两条用例一起变红——用户会在 `backward` 的 `reduce_sum` 里看到一句与
环节无关的形状错误；把 `_shape_of` 退回 `tuple(int(d) for d in shape)`，动态维
用例抛出的是裸 `TypeError` 而不是契约错误。两次都说明断言钉的是入口行为。

**验证**：本地 `Ran 2242 tests / OK (skipped=410)`；WSL HIP（RX 7650 GRE /
ROCm 7.2）`Ran 2242 tests / OK (skipped=369)`。

**边界**：契约只管"图输入是否静态"——符号维只能从图输入进入，中间值不可能凭空
多出一个符号。卷积 / 池化按 `attrs` 派生出来的空间维（5.53 的 `SymExpr`）不经过
这道守卫，由算子自己的构图期校验负责；部署包的数据输入仍可符号，所以
`deploy` 是 `data-only` 而不是 `rejected`。这张表是"当前支持面"的声明，不是
"未来都要支持"的承诺：哪个环节将来接上动态形状，先补用例再改状态。

## 6. 路线图

### M0　语义 IR + 参考执行 + 优化框架　✅ 已完成

验收：Transformer Block 可表达、可执行、可优化，且优化前后数值一致。

### M1　前端适配器　✅ 已交付

- `frontend/torch_export.py`：读 `torch.export` 的图，降级到本 IR
- 遇到不支持的算子**明确报错退出**，不静默 fallback
- 验收：同一份权重下，"torch eager 输出"与"IR 在参考后端上的输出"数值一致
- 已扩到 7 个真实 torchvision 模型（5.39），其中 resnet18 / squeezenet1_0 在真机 CUDA 上对拍

注意：这一步验证的是**降级过程没有改变语义**，不是跨后端一致。

### M2　CUDA 后端　🟡 第一版已交付（已在 RTX 2080 Ti 上验证）

接口要求：

- `capabilities()` 如实声明承接的算子；支持不了的必须报错，不能假装支持
- `matmul` 走 cuBLAS（表达为 `LIBRARY` 节点，声明 layout 与精度约束）
- 其余算子由自研 CUDA C 源码生成，交给 NVRTC 运行时编译（不依赖 CUDA toolkit）
- 必须实现 `determinism` 与 `precision` 声明，否则一致性测试无法判定

验收标准（三条同时）：

1. Transformer Block 与参考后端数值对齐（rtol 1e-4）
2. 图中不出现"悄悄降级"：每个节点要么被原生承接，要么明确失败
3. `verify_portable_only` 仍然通过（CUDA 后端不得把专有节点写回共享路径）

三条验收标准均已通过。第一版的已知缺口（按优先级排序）：

| 缺口 | 影响 |
|---|---|
| 两条直算路径只覆盖 f32 | 分块直算与隐式 GEMM 的 shared 与累加器都是 float：显式选择且输入为 f16 时以 `BackendCapabilityError` 失败（f16 的 conv2d 走 im2col + hgemm）。其余 57 个算子的 f16 自 5.121 起全覆盖——attention 是混合精度（q/k/v/mask/out 存半精度，点积、在线 softmax 与输出累加在 float） |
| 累积与更新仍在 f32 | f16 只降低存储与带宽，归约与 GEMM 的累加都在 f32 完成；master 权重与优化器状态必须保持 f32（5.28），f16 参数直接更新会被量化吞掉 |
| softmax / layernorm / reduce_sum 只支持最后一维 | 覆盖 Transformer，不覆盖全部模型。**已由 5.153 关闭**：任意单轴可跑，多轴 `reduce_*` 与 `argmax` 仍限末维 |
| 广播折叠后上限 4 维 | 超高维张量的非常规广播会被明确拒绝 |
| 批量的部分广播（`(2,1,…)` 对 `(2,3,…)`） | cuBLAS 一个标量 stride 表达不了，明确拒绝 |
| conv2d 分块核只在 K <= 32 时占优；分组卷积不走分块核 | K 大时仍走 im2col（列矩阵放大 8~9 倍）；groups > 1 时逐组 im2col 并逐组 GEMM，depthwise 可用但没有专门的 depthwise 内核。见 5.16、5.34 |
| 池化仍缺参数开关 | maxpool2d / avgpool2d / adaptive_avg_pool2d 均可用（全局池化 = `output_size=(1,1)`）。**已由 5.154 关闭**：`ceil_mode` / `count_include_pad` 是 IR 属性、前端直通，新增 `maxpool2d_indices`；只有 `divisor_override` 仍明确拒绝 |
| scan 的并行度 | `scan_linear` / `scan_linear_backward` 一个尾元素一个线程沿 T 串行，延迟随 T 线性增长；`chunked_scan=` 打开后走两趟分块（块内复合 + 块前缀/读出），M40 上 `T=16384` 前向 3.11×、反向 3.56×，`scan_last_state` 只写末状态（16 MB → 4 KB）。**5.155 起默认自动**：`steps >= 512` 且尾维 `<= 2048` 时取块长 64，短 T / 宽尾维仍串行；分块与串行不逐位，要逐位显式 `chunked_scan=0`。见 5.35、5.128、5.155 |

"每节点独立 launch"已由运行期计划解决（5.4 M3-3），不再是缺口：默认路径仍逐节点下发，
捕获路径按固定形状整图 replay。"无 IR 版本号"也已由 5.8 解决。

### M3　融合 + 内存规划

- **设备驻留 ✅ 已完成**：`resident_parameters()` 把权重一次性上传并常驻显存，
  之后 `execute` 只传激活值；错名字/错形状在驻留时就失败。收益见 5.4。
  训练侧也补齐了（5.84）：`Trainer(device_state=True)` 让权重、梯度、优化器
  状态整步都不回主机，前向与反向共享同一组权重缓冲。配套的三件事随后落地：
  训练辅助工具全部分派到数组所在设备（5.85）、数据侧走 pinned 暂存的异步
  H2D 且共享缓冲不再自拷贝（5.86）、HIP 对齐同一套 `shared_buffers` /
  `run_device` 语义（5.87）；设备态与激活重算也能叠加（5.88）。
- **IR 层融合 ✅ 三条规则**：`FuseSoftmaxScale` 把 `softmax(mul(x, s))` 变成
  `softmax(x, s)`；`FuseLayerNormResidual` 把 `layernorm(add(a, b), …)` 变成
  `layernorm(a, …, b)`；`FuseSiLU` 把 `mul(a, silu(b))` 变成 `silu(b, a)`。
  三条都扩展已有算子的输入，不新增算子（ADR-010）。收益与局限见 5.4、5.36。
- **运行期计划 ✅ 已完成**：`CudaGraphPlan` 把整图捕获成 CUDA Graph，CPU 派发
  800 μs → 8 μs，小规模端到端 3.3x。为此自研了 cuBLAS 的 ctypes 薄绑定。见 5.4 M3-3。
- **内存规划 ✅ 已完成**：捕获期按"最后一次被使用"回收中间缓冲，`Arena` 负责
  确定性切分与相邻块合并；捕获后可直接读 `peak_bytes`（下界）与 `memory_bytes`
  （实际占用）。显存 medium 70 → 32 MiB，吞吐无变化。见 5.4 M3-4。
  **编译期静态预算**也已补齐：`plan_memory` 直接从 IR 的 liveness 算出峰值与复用
  倍数，不执行、不需要 GPU；与捕获期峰值同量级（真机 8192 / 6656 B，见 5.36）。
- **深层融合 ✅ 已交付**：LayerNorm+残差与 SwiGLU 门控都表达为已有算子的输入扩展，
  CUDA 侧各有一个内核（含 f16）。pre-norm Block 的残差 add fanout == 2，因此
  LayerNorm 残差规则在那里不命中——这是结构性的，已写成测试的否定结论。
  GEMM epilogue 已被实验否定（cuBLAS 必须物化输出），不再列为待办。见 5.36。

### M3.5　CI 与性能回归　🟡 第一版

按 1.3 的推进顺序，这一项排在 CNN 之后、自动微分之前——因为一旦开始改算子语义
（反向传播会动到 conv / pool 的每个内核），没有性能与正确性的门禁，就分不清
"改错了"和"改慢了"。见 5.15。

- **已做**：`benchmarks/` 10 个固定用例 + 中位数计时 + 多轮取最好 + 同机基线判定；两份基线
  （参考后端 / RTX 2080 Ti）已采集，CUDA 那份在 5.92 重采并带上宿主指纹；`ci.yml` 现在有三个作业——`cpu` 跑 Python 3.10 / 3.11
  的全量测试与参考后端 conformance，`fuzz` 每次 push 跑 200 个随机种子的模糊测试，
  `gpu` 手动触发、串起 verify_cuda → 全量测试 → CUDA conformance → 25% 回归检查 →
  调优往返 → 趋势采集并上传产物；`TelemetryRecorder.trend` + `benchmarks/trend.py`
  提供了跨运行的吞吐 / 显存聚合与 `--compare`（5.38）。
- **未做**：GPU job 尚未接上真实的 self-hosted runner（目前是在远程机器上手动跑
  `benchmarks.run`；五个步骤在 5.92 复跑全绿，基线也已重采，缺的只剩仓库侧注册）；
  趋势只有在被显式调用时才产生，没有自动采集调度、没有看板，
  也没有"哪次提交变慢"的自动二分（趋势的斜率只作观察，不作门禁）；GPU 基线只覆盖一张卡。

### M4　第二后端 + 一致性套件　✅ 已完成

这一阶段拆成两组：前三项**不依赖新硬件**，所以先做；最后一项是真正的硬件验证。

**不需要新硬件（先做）**

- **IR 版本化与兼容策略 ✅ 已完成**：图可导出为纯 JSON（`to_dict` / `dumps`），
  自带 `ir_version`；主版本不同、或次版本高于本实现，一律拒绝，verifier 与执行
  入口都会拦。见 5.8、ADR-011。
- **conformance 套件 ✅ 已完成**：`run_conformance(backend) -> Report` 用固定用例
  （IR 构图，可导出为 JSON）+ 5 条契约检查（接口完整、可移植性、序列化往返、
  失败方式、覆盖度）产出结构化报告。用例集在自动微分（5.17）、SNN（5.18）、
  SSM（5.19）、新增算子（5.20）、随机算子（5.23）、批归一化（5.24）、融合注意力
  （5.32）、自适应池化（5.34）与 `erf`（5.39）后从 31 扩到 67；参考后端 67 executed
  / 0 refused / 0 failed。CUDA 侧按能力声明判定：46 个算子对应的用例执行，只有两个
  scan 算子与超出承接范围的形状/精度用例记为 refused（`BackendCapabilityError`）。真机
  （RTX 2080 Ti）复跑全绿：62 executed / 5 refused / 0 failed，另加 32 条边界用例
  全过，见 5.20、5.23、5.24、5.34、5.39。套件本身用四种"坏后端"验证过，不会永远返回 PASS。见 5.9。
- **自动调优 + 智能体接口 ✅ 第一版**：与后端无关的参数空间 / 网格搜索 / 配置库
  （`aetherflow/tuning/`）、结构化运行时上报（`runtime/telemetry.py`）、只读查询面
  （`agent_api.py`）均已交付，见 5.37。CUDA 侧开放逐元素算子的 block 与 conv2d 的
  `tile_w`；调优面与遥测层本身都与后端无关，HIP 靠继承即可用（5.58 补掉了漏写的一个
  字段并把两份构造的字段对账做成了测试），三个基准 CLI 也已在 5.59 改为查注册表选择
  后端；5.60 又在 WSL 的 ROCm（RX 7650 GRE）上实跑通过，两侧已能各自产出对比数据。

**需要硬件（后做）**

- **HIP/ROCm 第二硬件后端 ✅ 已完成**：`backends/hip/` 一个包，容器（ctypes 直连
  `libamdhip64`）、编译（hipRTC）、BLAS（hipBLAS）与方言注入（f16 存取 / 原子加）
  是新写的；算子语义、形状检查、分块策略与 57 个内核模板全部复用
  `DeviceBackend`（5.109 之后 HIP 与 CUDA 是兄弟类，共享的设备无关执行计划在
  基类里，厂商钩子各自实现）。
  准入判据是上面那套 conformance：在 RX 7650 GRE（gfx1100 / ROCm 7.2）上
  **62 executed / 5 refused / 0 failed**，与 CUDA 真机逐项一致，拒绝项相同。见 5.40。
  这条也顺带回答了一个隐含问题：新硬件接入没有改动 IR / verifier / 前端 / 优化任何
  一行代码，说明"新硬件只写插件"不是宣传语而是可执行的事实。
  设备态训练与 MIOpen 对照已由 5.156 / 5.157 补齐：`pinned_array` 按
  `(名字, 形状, dtype)` 复用固定内存、`replay` 走异步 H2D 与跨流事件，
  `Trainer(device_state=True)` 的 SGD / Adam 轨迹与主机态一致；MIOpen 侧用
  solution API 对拍了 10 个形状，自研分块核慢 4.52×～37.03×。
  hipGraph 捕获、捕获期 arena、调优 / 遥测接入、跨厂商性能对比已由 5.61～5.63 补齐。

---

## 7. 已知风险与待决问题

| 风险 | 影响 | 当前对策 |
|---|---|---|
| 跨厂商只各验证了一张卡 | 中 | 同一套 IR 已在 NVIDIA sm_75 与 AMD gfx1100 上跑通（5.40），但结论的边界是"两个厂商各一型号"：gfx9 的波前 64、`__shfl` 语义、驱动版本差异都未触及；5.62 的 HIP 数字来自 WSL2 DXG 直通，与裸机 ROCm 不可直接对比，只能作同一条路径内的相对结论 |
| 多设备已在 3 卡机器验证，跨卡协作只有最小闭环 | 低 | 5.164 的枚举 / 切换在 CUDA 与 HIP 上都已实现并提炼成 `DeviceManagerContract`；5.169 在 3 × Tesla M40 上补齐真机：枚举长度、卡名 / 显存 / SM 数各不相同、切卡后同一张图逐位一致、切卡丢弃驻留参数与捕获计划、越界 / 负数 / 非 int 当场拒绝且状态不变，5 个用例全绿（单卡机器整组跳过）。`set_device` 是**丢弃**而非迁移已有捕获计划 / arena / 常驻缓冲；5.190 补了跨卡拷贝（`copy_across_devices`，经主机中转）与数据并行最小闭环（每卡参数副本 + 主机侧归约 + `DataParallelTrainer`），两块卡上的数值与单卡合并 batch 对齐；但仍是串行切卡执行，peer-to-peer 与真正的并发（多进程 / 多流、NCCL / RCCL）未做 |
| 分组卷积的收益随厂商与形状翻转 | 低 | 5.165 的直算核与 HIP 容器的非连续写回在 RX 7650 GRE 上跑过（含 dilation 与 im2col 对照）。5.166 的 HIP 对照：9 个形状快 2.51×～54.91×、省掉 1.72～27.56 MB 列矩阵。5.171 补 CUDA 真机（3 × M40，2 号卡）：depthwise 4.30×～21.50×，但非 depthwise 的 `g4` / `g2` 出现 **0.80× / 0.44×** 反例（cuBLAS 反超）——`auto` 因此收窄为只对 depthwise 走直算核，`conv_impl="grouped"` 改为显式选项。剩余边界：分组核没有 shared staging，也没有 NHWC / depthwise 向量化；`CG` 更大（如 groups=2 且 C=512）未覆盖 |
| 可选契约要在多处手工同步 | 低 | 5.163 / 5.164 的三组契约要分别落在 `contracts.py`、两个硬件后端的实现与所有包装 / 装饰层上。5.167 修的正是 `InstrumentedBackend` 漏同步：包装后 `implemented_contracts` 静默少三项，不报错、不警告。目前只有 telemetry 这一层有"`implemented_contracts` 与内层整表相等"的测试兜底；将来新增包装层或新增契约仍可能重演。训练与基准里其余 `hasattr` 探测（`capture` / `resident_parameters`）不在契约表内，也没有同类断言 |
| HIP 缺运行期计划与分配器 | 中 | 已由 5.61 补齐（hipGraph 捕获、捕获期 arena、`memory_stats()`）；5.62 又补上捕获与逐节点两条口径的对照入口；5.156 再补 pinned 暂存与设备态训练，运行期面与 CUDA 对齐；5.159 把捕获窗口内的 `hipFree` / `hipHostFree` 推迟到 EndCapture 之后（窗口内只记账，GC 与捕获不再打架，偶发 901 消失）。剩余差异：`reuse_memory=False` 明确拒绝，捕获期不能扩容，大图必须靠预热轮的块装下 |
| MIOpen 对照只覆盖 10 个形状、单轮 | 低 | 5.157 的 4.52×～37.03× 是固定 NCHW / f32 / stride 1 / dilation 1 / groups 1、默认流、单次采集下的相对结论；f16、分组、非 1 步长与裸机 ROCm 都未测，不能外推成"AMD 上的全部差距"。选解按库给出的估计时间，不是实测；没有任何调优结论从这张表得出 |
| 内核模板只有一份、方言靠注入 | 中 | 好处是不分叉，代价是任何一处方言差异都必须能在 `%PRELUDE%` / `%T%` / `%ATOMIC%` 三个占位符里表达。HIP 已经用满两个（f16 存取、无符号原子加）；接入第三家（如 Intel / Apple）前要先确认模板没有被 CUDA 语义绑死 |
| CUDA 库掩盖 IR 硬件泄漏 | 高 | ADR-002 + ADR-003 的机制已就位，M2 是第一次真实考验 |
| IR 格式一旦对外就是兼容义务 | 中 | 版本判定已落地（ADR-011）；新增字段必须保持可选。5.162 补上跨版本迁移工具：迁移作用于**序列化结构**、每个源版本只允许一条出边（路径唯一）、只升不降、缺一段报一段，迁移过的图记 `Graph.migrated_from`。边界仍在：主版本不迁移；默认值怎么补要由迁移作者显式注册，工具不猜 |
| 算子融合收益未量化 | 中 | M3 用 CUDA 后端做 A/B，而不是估算 |
| 跨图控制流每次判定都要回主机 | 低 | 5.201 的 `Conditional` / `WhileLoop` 把控制流表达成子图之间的关系，代价是每次分支/每轮迭代都要读一次标量谓词（一次同步），长循环的派发开销在主机侧：没有 device-side 循环，也没有把循环改写成 `scan_linear`。控制流不可微——`backward` 不穿过控制区域，带分支的训练仍要在 Python 里展开或按段建反向图；两个分支不并发，也没有"两支都算再选"的 `where` 式向量化形态。适用面是"路径由少量标量决定"的图（早退、门控、迭代到收敛），不是把整个模型的控制流搬进 IR |
| 自动微分的二阶能力 | 低 | 已由 5.198 补上并验证（解析二阶导、一阶梯度中心差分、跨输入块的混合偏导、二次目标上一步 Newton、序列化往返）。剩余边界：二阶图重放前向、节点数约 4 倍；零曲率与 `conv2d_*_grad` / `maxpool2d_backward` 这类没有 VJP 的算子显式 `NoGradientError`，不造零张量；默认种子名撞上已有输入时必须显式给 `seeds=` |
| 反向图优化的边界 | 低 | 5.207 的 `minimize` 只剪 vjp 与反向节点，前向重放仍是整张 live 图（vjp 要读输入值，"读值"与"求它的导数"是两件事）；`wrt` 覆盖全部输入时传播范围不减，收益只出现在"只对部分参数求导"的场景。交换律 CSE 只认 `add` / `mul`，其余算子仍按有序签名比较；`backward_pipeline` 不含融合，节点数可能高于 `fused_pipeline`——这是"任何承接前向的后端都能直接跑反向图"的代价 |
| cuFFT 只有离线验证 | 中 | 5.210 的 cuFFT 绑定（归一化表、dtype 登记、plan 键、plan 池）全部是纯函数或离线用例，没有一张真实 GPU 跑过 `cufftExec*`；本机 NVIDIA 驱动未加载，CUDA 的 `capabilities()` 也因此只在 `cufft.available()` 为真时声明 `fft` / `rfft`。剩余边界：只做沿最后一维的一维变换、输入必须连续、`cufftPlanMany` 是 host API 因而捕获期拒绝；HIP 没有 rocFFT，`fft` / `rfft` 在 HIP 上是一致性套件里的显式拒绝项 |
| 常量折叠的收益已量化，产物大时的倒退已收掉 | 低 | 5.161 用三张图对照：mask 9→3 节点、wall −11.0%；all_constant 7→1（wall −80.4%、GPU −89.2%）；compress 5→1（−73.8% / −85.6%）。当时 mask 用例的逐节点 GPU 时间 **+22.4%**（`constant` 每次 execute 都重新 H2D，256 KB 产物比原来的 4 个小 kernel 更贵）。5.172 把常量设备副本挂到执行计划上，mask 用例 3.3838 → 0.3390 ms（−90.0%），并按 5.161 的结论只物化存活边界；捕获轮会重建计划重新上传（HIP 的 `Arena.recycle` 会收回预热期地址）。`max_elements`（默认 65536）仍挡住超大张量的折叠 |
| 调优的旋钮面与实测覆盖仍窄 | 中 | 参数空间、搜索器与配置库已交付（5.37），HIP 侧调优面已补齐并实跑（5.58～5.60、5.63）；5.160 起归约类（softmax / layernorm / rmsnorm / reduce_sum / reduce_max / reduce_mean / argmax / batch_norm / batch_norm_stats）的 block 也开放了，旋钮面覆盖 `tunable_ops()` 全部算子，只剩 conv2d 走自己的线程铺法。配置库按设备名命中，换卡必须重新采集，不能跨机复用 |
| 调优收益未量化、且已证实会被噪声误导 | 中 | 5.63 在 CUDA 与 HIP 上各采两次：同一台机器的"最优 block"两次之间只有 2/9 与 4/9 重合，跨后端 2/9 与 0/9。5.140 已把测量改成交替 A/B + 配对差的 bootstrap 区间，并加"前后两半同向"检查；5.141 又实现每轮随机排列并做了 alternate/shuffle 的 A/B 对照。两次改进都没有让赢家可复现：7 次 8 轮采集里 conv2d 有 4 次判收益、3 次判噪声，4 次收益的赢家各不相同（`64/4/auto`、`auto/2/2`、`64/4/2`、`auto/4/1`）；逐元素 block 基本保留默认，但 7 次里仍有 3 次一次性判收益，且都没有在下一次重复。"批次不稳"的候选数随整机状态在 12～41/45 之间波动，与 order 无关。根因不是排列，而是轮内跨度：conv2d 一轮 45 个候选约 2.3 秒，逐元素一轮 4 个候选约 0.26 秒。5.142 已把跨度接成门槛（`--max-round-seconds`）：HIP 一次采集里 conv2d 8 轮全部 2.2～3.3 秒、44/45 个候选判 `long-round` 保留默认；dropout 只有首轮 1.60 秒超限，却因"任一轮超限即整次降级"被 4/5 个候选一起作废；add_broadcast 与 relu 未超限，但赢家仍与 5.141 不同（`block=512` / 256）。实测同时暴露更强的根因：四个用例的第一轮分别是后续轮次中位数的约 14× / 2.1× / 4.0× / 1.3×，是冷启动而不是随机的位置效应。5.143 已实现预热轮（`--warmup-rounds`，默认 0）：预热耗时与 5.142 的首轮同量级（0.67 / 0.86 / 0.94 / 2.79 秒），正式轮全部回到稳态；dropout 的跨度误伤消失，conv2d 仍按稳态的 2.2～2.5 秒超限；5.142 判出的两个收益（`block=512` 1.03×、`block=256` 1.05×）在预热后都退回默认。5.144 再做两阶段搜索（`--staged`）：conv2d 从 45 个候选拆成 4 + 11 + 2，单轮跨度降到 0.21 / 0.65 秒，`long-round` 归零（5.143 是 44/45）；但阶段 2 的 11 个候选里 10 个两半不稳，最终仍保留默认，逐元素里只有 add_broadcast 判 1.08×。跨度已不是唯一瓶颈，剩下的是整机漂移下的 8 轮样本量。5.145 把轮数加到 16：conv2d 阶段 2 的不稳数减半（10/11 → 5/11），但仍无可接受赢家；8 轮的唯一收益（add_broadcast 1.08×）消失，该候选两半差达 +13.18% / -12.07%，即采集内整机漂移约 25%。加轮数只减少假阳性，不产生真结论，成本翻倍，故不采用。下一步是跨采集复现（两次同向才接受）或控制环境（锁频/温度），而不是继续调测量参数。5.146 已实现前者（`--repeats`）：HIP 上两次采集各判出不同赢家（add_broadcast 的 `512` 1.0756× 与 `1024` 1.1316×、relu 的默认与 `128`，两次都过了 1% 地板），逐候选合并后全部 `unreplicated`、保留默认；conv2d 两次一致地 44 个 `long-round`，说明合并逻辑能区分"环境拒绝"与"结论不一致"。跨采集复现只降低写库假阳性，不提高发现真赢家的能力，根因仍是整机状态不可比。5.147 在 CUDA 真机（M40）补齐同一实验：9 个用例全部保留默认，两次各一次 `decided=True`（sigmoid 1.0144×、silu_gate 1.0139×）合并后都判 `unreplicated`；同一套 1.0 秒跨度门槛在 M40 上不触发（conv2d 最长轮 0.874s），说明绝对秒数门槛跨机不可移植——同一台机器上 conv2d 判 `long-round`（HIP）还是 `unreplicated`（M40）取决于跨度是否超限，而不是结论本身。5.148 用预热轮做相对标尺（`--max-round-factor`）：conv2d 的 `long-round` 归零，但单次判出的 3 个收益叠加 `--repeats 2` 后全部 `unreplicated`——放宽门槛只是把"拒绝"换成"由复现筛"，不能单独使用。5.149 又加多数票（`--vote majority`，需 3 次以上）：同一批 HIP 三次采集按全票合并时 9 个用例全部保留默认（conv2d 42/45 个候选 `unreplicated`），按多数票合并时 conv2d 的 `cb=2 / tile_w=64` 以 2/3 票（2 faster + 1 noise、无 slower）转正——配对差 -11.75%、CI -14.20～-2.33、1.037×。它把"全票过严"换成"少数可通过"，但 2/3 票的区间跨度 11.9 个百分点，属弱证据；逐元素用例没有候选拿到两张 `faster`（票型以 `unstable×2` 为主），说明整机漂移才是主要噪声源，投票门槛不是。CUDA 真机当时 cupy 不可用、多数票未采，本轮已恢复（裸机 Ubuntu 22.04、3 × M40 24GB、驱动 535.309.01 / CUDA 12.2、cupy-cuda12x 14.2.0，固定 2 号卡）：5.150 补了逐次配对差（`replicate_diffs`），并把 `--repeats` 加到 5：HIP 上 9 个用例仍全部保留默认，但 45 次"用例 × 采集"里有 8 次单次判出赢家，且**两两不同**——conv2d 三次三个赢家（`64/4/1` 1.076×、`64/2/2` 1.098×、`16/auto/1` 1.130×），逐元素各只出现过一次。5.149 那个以 2/3 票转正的 `cb=2 / tile_w=64 / roc=auto` 在 5 次里拿到 0 张 `faster`，逐次差 `+2.95 / +0.27 / +5.34 / -2.20 / -8.05`（中位 +0.27%）：方向本身在采集间翻转，说明那次转正只是"赢家每次都在换"的一个切片。全批 85 个可行候选、425 次判定里只有 14 张 `faster` 票，没有任何候选拿到两张以上——瓶颈是整机漂移，不是投票门槛，也不是幅度判别。M40 上 425 次判定只有 4 张 `faster` 票，5.149 的转正候选逐次差 `+0.49 / +0.04 / +1.71 / +0.42 / +0.92`（5 次全为正、0 张 `faster`）；conv2d 里 21 个候选 5/5 全票 `slower`（`64/2/2` 逐次 +13.2%～+19.2%），说明该空间确实存在真实更慢的配置。CUDA 上唯一"方向稳定、幅度贴地板"的是 dropout `block=128`（逐次 -0.55/-1.36/-1.29/-1.11/-0.36、均值 CI -0.59～-0.34，合并仍 `noise`）。两台机器的结论一致：瓶颈是整机漂移与 1% 地板，不是投票门槛。5.160 把归约类纳入调优面，并新增 `--min-same-sign N`（逐轮核对方向，同号不足把 `faster` / `slower` 降为 `unstable`）：HIP 两次采集中 softmax / reduce_sum / layernorm 三个用例的候选全部保留默认，单次最优差 −4.5%～+5.3% 但同号仅 3～4/5（或 5/5 更慢），且 `reduce_sum` 的 `128` 两次方向相反（−0.9% 与 +5.3%）；同号门槛把"4/5 同号 + 4.5%"这类弱证据直接降级，不写进配置库 |
| 派发耗时容易被误读成 GPU 时间 | 低 | `RunRecord.dispatch_ms` 在文档与字段名里都注明是 CPU 派发；整图 `wall_ms` 才含同步。5.161 起提供逐节点 GPU 计时（`NodeTiming.gpu_ms` + `RunRecord.node_gpu_totals()`，CUDA 用 `cupy.cuda.Event`、HIP 用 `hipEventCreateWithFlags(0)`），只在收尾同步一次，不把同步开销算进测量对象。边界：`gpu_ms` 为 `None` 表示"没有观测"（后端无时钟或事件不可用），不是 0；计时事件本身有约微秒级开销，小于 10 µs 的 kernel 读数只能作相对参考 |
| SNN 的稀疏已接到构件层，但收益门槛未达到 | 低 | 事件表示、内存池（5.35）、事件驱动的 `sparse_linear` / 权重梯度（5.132）与 `build_snn_events` 构件（5.133）都已交付，稠密 `S×K` 脉冲矩阵不再物化，事件路径与稠密网络在三组深度上逐位相等；5.136 之后层间压缩也可微（`spike_linear`：脉冲梯度走普通 `matmul`，只有权重梯度走事件散射），事件路径能训练并与稠密路径对拍梯度。剩余边界有两条：**时间展开仍让图随 T 线性增长**（5.18）；**反向的脉冲梯度是稠密 GEMM**——5.204 之后 `sparse_linear` 的 `values` 梯度也留在事件形态（`sparse_linear_values_grad`，长度 `E`、无需原子加），但 `spike_linear` 的 `dspikes` 仍是稠密 GEMM，"事件驱动"在训练里省的是权重梯度与事件幅度梯度，不省 `dspikes`；5.137 实测训练步与稠密路径持平（HIP 1.01×、M40 1.00×）。HIP 实测是负结果：`steps=8 / batch=4 / hidden=(64,64,64)` 下事件路径 0.72×～0.96×，开权重常驻后仍只有 0.96×～1.00×，瓶颈是每层多出的两趟压缩与稀疏乘；压缩的 `spike_offsets` 已由 5.134 并行化（大 `T` 下 19.4×，但小规模只快 6%），`spike_csr` 仍是行内串行；`sparse_linear` 在 2048×1024×1024 上常驻权重时 2% 稀疏才 1.54×、0.2% 1.72×，收益门槛约 10% 稀疏；5.135 在捕获口径下复测同配置（`hidden=(64,64,64)`、`steps=8`、`batch=4`）：dense 2.75 ms / events 3.17 ms，launch 开销被摊掉后 events 仍慢约 15% |
| SSM 的分块扫描按阈值自动切换，阈值只在一台机器上标定 | 低 | 5.155 起默认不再是"一律串行"：`steps >= 512` 且尾维 `<= 2048` 时自动取块长 64，`scan_last_state` 的尾维上限放宽到 4096；M40 上 T=16384 / 尾维 16 前向 3803 → 190 µs（20×），真实形状 T=1024 / 尾维 1024 前向 451 → 238 µs（1.90×）、反向 2.53×。decode 单步（`T=1`）仍走串行，短 T 与宽尾维也仍走串行。风险在于阈值全部来自 sm_52，HIP 与其它架构未扫；分块与串行不逐位相等，要逐位必须显式 `chunked_scan=0`。5.103 的串行边界（T ≤ 2048 时整图由 launch 固定开销主导）仍然成立 |
| CI 未在托管平台实跑 | 中 | `.github/workflows/ci.yml` 已配置 `cpu` / `fuzz` / `gpu` 三作业，GPU 组已在真机手动逐条执行、五步全绿；趋势的跨 run 接续与每日 cron 已就位（5.199），但仍缺 git 仓库与 self-hosted runner 的注册令牌，schedule 只能排出一个永远 pending 的作业。这是**外部前置条件**，不是代码缺口 |
| 模糊测试的后端对拍与语义变异覆盖仍有限 | 低 | 5.158 起随机图逐例与 CUDA / HIP 对拍（浮点 1e-4、整型逐位、按契约拒绝只计数），并补三类语义非法节点（reshape 元素数不等、matmul 内维不匹配、cast 到未知 dtype），要求构图期明确抛错。剩余边界：随机图仍是静态形状、≤10 节点、31 个算子，符号维不进生成器；CUDA 分支本轮真机未复跑；运行期语义错误已由 5.175 构造（8 条固定用例），但仍只覆盖 `embedding` / 事件类算子 / `div` 三族，池化窗口与 `slice` 的边界下标还靠各自的单测 |
| 趋势噪声不能当门禁 | 中 | `trend` 记录首末、极值、中位数与最小二乘斜率，但不判定回归：样本少（默认 5 轮）、跨机噪声大，斜率只作观察。真正的门禁仍是 5.15 的同机中位数 25% 判定（5.38） |
| 小规模下 CUDA 反而更慢 | 中 | 已定位为 CPU 侧固定开销并由运行期计划解决（5.4 M3-3），不是实现缺陷 |
| 运行期非法数据只在参考侧报错 | 中 | 5.175 把"图合法、数据非法"的检查钉在参考语义上：`embedding` / `embedding_backward` 的正负越界下标、整数的分母为 0 都在执行期抛 `ValueError`（此前负下标静默回绕、整数除零静默截断成 INT_MIN），浮点除零按 IEEE 给 inf 且不再打宿主 warning。CUDA / HIP 内核不做逐元素下标检查——设备端无法抛异常，做检查要么每步 D2H 一次 `ids`、要么在核心里写越界标志再同步复查（5.98 的 `err` 标志只接在稀疏事件路径上）。因此 GPU 后端的契约仍是"调用方保证下标合法"，越界读在设备上仍是未定义行为；要闭合这条需要把 `embedding` 也接进 `err` 标志，且只在打开检查时才付同步成本 |
| 非末维归约的性能路径只在一台机器上标定 | 低 | 5.153 起 softmax / layernorm / rmsnorm / reduce_sum 支持任意单轴（多轴 `reduce_*` 仍显式拒绝，`argmax` 仍只有末维）。5.174 补了两维铺法（`blockDim.x` 沿 inner 合并访存、`blockDim.y` 切归约轴 + 块内树归约），窄 `inner` 上 `inner=1` 快 7.7×～9.6×、输出平面 1024 且 `inner>=32` 时 2.1×～3.3×，判据 `_use_tiled_reduce` 在输出平面够大时正确留在一维（16384 个输出的格子上强行两维会慢 8%）。剩余边界：`sm_count` 是判据里唯一的机器相关项，只在 M40（sm_52，24 SM）上标定，HIP 与其它架构未扫；`inner` 极小（3 / 8）时两种铺法都被 launch 开销主导，收益只有 1.1× |
| `spike_csr` 两条路径的拐点只在一台机器上标定 | 低 | 5.176 把压缩的第二趟从"一线程一行、行内串行"换成"一 block 一行 + 段内块内前缀和"（`block_scan_sum`，暂存区固定 `CSR_SCAN_SLOTS = 1026` 个 int，即 `BLOCK_MAX + 2`）。M40 上 `steps=1` 2.21×、64 行 1.90×、256 行 1.52×，而 `steps>=2048` 时串行更快（0.90×～1.00×）：扫描路径每行要按 `width` 分 `width / blockDim.x` 段、每段一次块内同步，行数一多就填不满 24 个 SM。判据 `plan_csr_scan_enabled(steps, sm_count)` 取 `steps < sm_count × 32`，auto 在 6 个形状上都取到更快的那条；但 32 这个系数与拐点位置只在 M40 上标定，HIP 与其它架构未扫，列宽远小于 `blockDim.x` 时扫描路径也仍被 launch 开销主导 |
| 索引层只覆盖线性化，不覆盖布局决策 | 低 | 5.177 只把"坐标 ↔ 扁平下标"收成 `Layout`，分块 / 向量化 / 跨步仍在各内核与 `plan.py` 里。`concat_copy` 这类带手写括号的下标没有迁移（统一规则不会产生那层括号，强行抽象会退化成一个参数化的 `Bin` 拼装器）；卷积 / attention / scan 的 tiling 偏移也不在这层。新增内核应优先用 `Layout`，旧内核保持原样——逐 token 对照是生产源码的硬约束，不能用改模板的方式"迁就"生成器 |
| 训练状态存档绑定的是"约定"，不是完整运行时状态 | 低 | 5.180 把未提交梯度、优化器超参、窗口长度、`rng_seed` / `rng_rank` 与 scaler 写进存档，恢复时逐项比对；但换一个优化器类、换 RNG 实现或改 `(seed, step, rank)` 的派生规则，存档无法判断。RNG 只记录种子与秩，不序列化 numpy 生成器的内部状态——逐位可复现的前提仍是每步由三元组派生 key（5.125）。另外 `aux/grad/*` 只覆盖累积窗口，`grad_hook` 里其他自定义状态需调用方自己存 |
| 入口 dtype 规则只覆盖图输入边界 | 低 | 5.181 在 `Executor` 的入口处拒绝浮点→整数、越界窄化与布尔当整数；图内 `cast` 仍按构图者显式写的语义执行，设备端也不做逐元素下标检查（与 11506 同一条边界：GPU 内核无法抛异常，越界读仍是调用方责任）。要闭合设备侧需要把 `embedding` 接进既有的 `err` 标志，并且只在打开检查时付一次同步 |
| 语言模型的推理图只在一台机器上标定，且会话状态不进图 | 低 | 5.187 的固定块预填充把 `(块长, head)` 的图在**一次** `generate` 内复用，32K / chunk=256 时 128 块共享 3 张图，但每次调用仍重新构图并（在 `graph_mode="auto"` 下）重新捕获，TTFT 里约 200～250 ms 是这笔固定成本，1K 的 tok/s 反而低于 8K/32K。5.191 的 `LMRunner` 把图缓存与权重驻留移出单次请求：同一 `(块长, head)` 的图跨请求命中，`OrderedDict` 按 `max_graphs` 封顶；M40 上 1K prompt / chunk=256 / 8 token 的重复请求 35.93 ms，无状态 `generate` 387.34 ms（10.78×），输出逐位相等、`close()` 后驻留表清空。5.192 的 `generate_batch` 补上同一次请求里的多样本解码：左侧补 pad + 逐行 reset 让不同长度互不影响，每行独立采样与独立停止。剩余边界：decode 仍是串行执行；5.196 的 `ServingEngine` 补上了请求队列与 continuous batching（以解码步为调度单位、图 batch 固定为 `capacity`、空槽白跑、状态在主机侧按槽位拼/拆、只支持 `spec.batch=1` 的会话、`constraint` 必须每请求一份），`constraint` 自 5.206 起在 `generate_batch` / `LMRunner.generate_batch` 里也**按行接受**（每行一份实例或 `None`，`batch=1` 可直接传单个实例；共享一份实例仍显式拒绝），`ServingEngine` 原本就是 per-request；权重更新必须重建会话（图与驻留都绑定当前参数表）；数字原只来自 M40（1K/8K/32K）与 RX 7650 GRE（1K），与 5.205 同期补上了 HIP 的长上下文实测（`benchmarks.lm_inference --backend hip --lengths 1024,8192,32768 --new-tokens 4`，RX 7650 GRE / `memory_mode="tiered"` / `graph_mode="auto"`、chunk=256）：TTFT 181.3 / 239.5 / 449.3 ms（1K / 8K / 32K），decode 27.6 / 31.3 / 33.1 ms/token，显存峰值三个长度都是 1.63 MB 且不随上下文增长——TTFT 的增长远慢于长度（32K 只有 1K 的 2.48×）说明 5.187 的图复用与固定块预填在 HIP 上同样生效，decode 的主项是每步的 launch 与一次 logits 行 D2H，不是模型算力；批量解码（`generate_batch` / `ServingEngine`）仍只在参考后端上对拍过，HIP 未跑。5.193 的 `prime` 把前缀状态（`state` 与末位置 `logits`）移进会话，后续请求只跑增量：1K prompt 的 768 + 256 切分上，全量预填充 656.287 ms、增量 5.691 ms；参考后端与全量逐位一致，真机只在短序列上逐位相等（64 token 的 48 + 16），32 token 的 24 + 8 已经差 7e-4（随机初始化的大模型上 1.9）——分块边界改变内核形状本身就有 1e-5～1e-2 的舍入差，再被 SSM 递归放大，所以长上下文不承诺逐位相等。前缀缓存另有三条边界：只支持 `batch=1`、生成后自动失效（一次 `prime` 一次 `generate`）、权重更新必须重建会话。5.194 的束搜索把 k 条假设放进图的 batch 维：状态按父假设在主机侧重排（`to_host` 后 gather，再作为下一次 feed 上传），因此每步多一趟 D2H/H2D；只支持单条 prompt（`spec.batch=1`）、不与会话前缀缓存叠加、不接 `constraint`；`num_beams` 的收益在参考后端上验收过（与穷举搜索分数相等），真机未计时 |
| 跨块 reset 与左填充的代价随 T 增长 | 低 | 5.178 的掩码用 `scan_linear` 表达块内累积和（不新增算子），`with_reset=True` 因此让每层多一条沿时间的递归链与若干逐元素节点；`steps=1` 的 decode 跳过扫描节点，但 prefill 长序列的图节点数与 T 线性相关。5.179 的隔离依赖调用方**正确填写 reset**：续接已有会话时必须清掉第 0 列，否则会丢掉带进来的状态。这两条都只在参考后端与 CUDA 上固定了用例，HIP 走同一份 L2 模板但未单独建 reset 形状用例 |
| 整数标签损失只有 HIP 真机验证过 | 低 | ~~5.188 的两条损失路径只在 HIP 真机上对拍~~ —— 2026-11-13 已在 3×Tesla M40（`CUDA_VISIBLE_DEVICES=2`）补跑：损失逐位相等、`dlogits` 最大差 1.49e-8，全量 1851 / 104 跳过全绿。剩余边界是"1 ulp"本身：两条路径的反向求和顺序不同，跨后端不承诺逐位一致，只承诺 1 ulp 内；`B*T*V` 超过 int32 上限时在分配前显式报错 |
| 受约束工具文法只覆盖标量与标量数组、且键顺序固定 | 低 | 5.189 的 `ToolCallGrammar` 支持 string / integer / number / boolean / null 与"标量元素数组"。嵌套对象参数、数组元素为对象都在**构造期**显式拒绝并提示改用 `JsonGrammar`（接受任意合法 JSON，再由 `Tool.check` 校验），不是运行期才发现写不出来。参数顺序固定为"必填在前 + 可选在后"，跳不过中间的某个 optional——只能先把它写出来。`enum` 只在校验期生效，解码期只认类型（取值集合不进 `allowed()`），所以 `{"flag":"maybe"}` 这类值文法放行、`Tool.check` 拒绝。工具名逐字符枚举，名字很多且有公共前缀时每步候选集合会变大（主机侧字典开销，不进图） |
| cuSPARSE 快路径没有真机对拍 | 中 | 5.203 的调用序列（`cusparseCreateCsr` → `SpMM_bufferSize` → `SpMM`）只做了静态审查与 19 条离线用例；本轮真机 SSH 可达但 NVIDIA 驱动未加载（`nvidia-smi` 无法通信、`lsmod` 无 nvidia 模块），7 条 GPU 用例整组跳过，CUDA 全量也未复跑。因此覆盖刻意收窄到 f32 值 + i32 索引、默认 `sparse_impl="custom"`、`auto` 只在库可用时启用，且库路径不做下标越界检查（自研核有）。恢复真机后的第一件事是跑 `tests.test_cusparse` 的 7 条用例，把结果补进本节与 §0 |
| 事件幅度梯度的设备路径按步并行，负载不均且无真机计时 | 低 | 5.204 的 `sparse_linear_values_grad` 一个线程包干一步的全部事件：并行度是 `steps` 而不是事件数 `E`，batch 小、步数少时会有大量空转线程（空步直接不做任何事），单步事件极多时该线程沿 `cols` 串行做点积。选择它的理由是避免"事件号 → 步号"的设备端二分（L2 没有 while），而不是实测更快——本轮只验证了数值（解析式、f64 中心差分、f16、符号事件数、越界拒绝）与 CUDA / HIP 的共同 L2 描述，没有任何计时。若真机上 `steps < SM 数` 成为瓶颈，改法是"按事件并行 + 定长展开的二分"，代价是内核里多 `ceil(log2(steps+1))` 轮比较 |
| 分片只省主机内存；流水只有"顺序"层面的重叠 | 中 | 5.205 的 `ShardedOptimizer` 在同一进程里切开槽位，省的是主机内存，不是显存；要省显存必须"每进程只上传自己那片参数"，属于运行时改造。5.211 补齐了两件可验证的事：`PipelineTrainer(schedule=...)` 提供 `sequential` / `gpipe` / `1f1b` 三种调度，1F1B 让同时驻留的段间激活从"份数"降到 `min(段数-1, 份数)`（`last_resident_peak` 记账，三者数值一致）；`reduce_scatter_grads` 用 Reduce-Scatter 取代 All-Reduce，每 rank 只接收本片（接收总量从 W·N 降到 N），与 `allreduce_grads` 逐位相同。但仍然**没有真正的并发**：单进程串行切卡，换的是执行顺序，不是线程或流；段间激活仍经主机中转，参数也不做跨进程 all-gather。流水与整批一次算不逐位相等（归约形状不同），只承诺 `rtol=1e-5`；micro-batch 不支持不等分。两条路径都只在主机侧验证，CUDA 真机未复跑（同 5.203 条） |

### 待决问题

已决（保留记录，供后来者对照）：

1. ~~是否需要 `constant` 算子~~ —— 需要。反向图必须自包含，常数不能靠额外输入投喂；
   已落为 `nullary` 算子（5.17）；折叠 pass 见 5.36。
2. ~~自动微分放在语义 IR 层还是新增一层~~ —— 放语义 IR 层。反向就是再构一张普通图，
   复用 verifier / 序列化 / 全部后端；SNN 替代梯度靠替换算子 `vjp` 实现（5.17）。

仍未决：

3. ~~三个插件契约（编译期 / 运行期 / 调优）何时拆分~~ —— 已完成：运行期最小面
   （`capabilities` + `execute`）与设备无关的执行计划在 5.109 上提到 `DeviceBackend`；
   编译期与调优在 5.163 提炼为 `CodegenContract`（公开 `kernel()`）与
   `TuningContract`（五个调优方法整组判定）。剩余边界是契约只描述"方法存在"：
   句柄类型、旋钮语义与调优收益仍由厂商与工具层各自定义。
4. ~~反向图的显存策略~~ —— 已在框架层回答（5.75）：不把"可重算段"写进 IR，而是
   用普通子图表达段，`CheckpointPlan` 负责切分与逐段前向 / 反向；IR 与执行器都没改。
   剩下的已知代价是"边界值被多段共用时浮点结合顺序可能差末位"。
5. ~~多设备枚举与切换要不要进契约~~ —— 进。5.164 先把 HIP 的 0 号卡解绑（绑上
   `hipGetDeviceCount` / `hipGetDevice` / `hipSetDevice`，属性与 SM 数按设备缓存），
   两家语义对齐后才提炼成 `DeviceManagerContract`（三个方法，非法索引当场拒绝）。
   剩余边界是真机只覆盖单卡（5.164 时）；5.169 已在 3 × Tesla M40 上补齐枚举、切换、
   逐卡跑图与切卡清理，跨卡拷贝 / 多卡并行仍未做。

---

## 8. 运行方式

### 本地（无 GPU）

```powershell
# 全部测试：CUDA / HIP 组自动跳过（需要各自厂商运行时）
py -3.10 -m unittest discover -s tests -t . -v

# 演示：打印 IR、执行、优化前后对比
py -3.10 demo_transformer_block.py

# 后端自证：跑一致性套件并打印结构化报告
py -3.10 -c "from aetherflow import run_conformance, ReferenceBackend; r = run_conformance(ReferenceBackend()); print(r.text()); assert r.ok"

# IR 模糊测试：随机图上的五条不变量 + 非法属性必须被拒绝（5.38）
py -3.10 -m aetherflow.fuzz --iterations 200

# 真实模型导入覆盖：需要 torch + torchvision，否则整组跳过（5.39）
py -3.10 -m unittest tests.test_frontend -v

# 换后端不改代码：按名、按环境变量或自动探测（5.41）
py -3.10 -c "from aetherflow import get_backend, registered_backends; print(registered_backends()); print(get_backend().name)"
```

任何新后端（含第三方插件）的准入命令都是同一句：把 `ReferenceBackend()` 换成
`get_backend("名字")` 或自己的实例，`r.ok` 为真才算承接了这套 IR。

必须使用 `py -3.10`：系统默认的 `python` 是 3.8 且没有 numpy。

### 远程 GPU 机器

代码通过 rsync/scp 同步到远程目录后，在远程执行：

```bash
python3 -m unittest discover -s tests -t .
```

CUDA 后端依赖 `cupy-cuda12x`（提供显存管理、cuBLAS 与 NVRTC）。
目标机器不需要安装 CUDA toolkit，也不需要 nvcc。

```bash
# 性能基线 / 回归检查（5.15）
python3 -m benchmarks.run --backend cuda --save benchmarks/baseline_cuda.json
python3 -m benchmarks.run --backend cuda --check benchmarks/baseline_cuda.json

# 卷积两条实现路径的对照，含 im2col 临时张量大小（5.16）
python3 -m benchmarks.conv_compare --rounds 5

# 单算子调优：搜索 block、写配置库、应用后逐位对拍（5.37）
python3 -m benchmarks.tune --backend cuda --save benchmarks/tuning_cuda.json
python3 -m benchmarks.tune --backend cuda --apply benchmarks/tuning_cuda.json

# 跨运行吞吐 / 显存趋势：采集并落盘，之后可跨文件对比（5.38）
python3 -m benchmarks.trend --backend cuda --rounds 5 --save benchmarks/telemetry_cuda.json
python3 -m benchmarks.trend --backend cuda --compare benchmarks/telemetry_cuda.json
```

`CudaBackend(conv_impl=...)` 可取 `auto`（默认）/ `im2col` / `tiled`；
强制 `tiled` 时绕过 `auto` 的规模门限，只校验 shared memory 预算。

**凭据不入库**：SSH 密码不写入任何项目文件；已改用部署的公钥免密登录。

### WSL2 + ROCm（AMD）

Windows 上经 WSL2 的 DXG 直通使用 AMD 显卡（本文档的实测环境是 Ubuntu 22.04 +
ROCm 7.2.53211 + RX 7650 GRE / gfx1100）：

```bash
# 全量测试（HIP 组在有 ROCm 的环境里跑，CUDA 组跳过）
# HSA_ENABLE_DXG_DETECTION=1 是 DXG 直通的前置：不设时 ROCm 看不到 /dev/dxg 背后的卡
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/g/Cloumi/aetherflow && HSA_ENABLE_DXG_DETECTION=1 /opt/rocm-env/bin/python -m unittest discover -s tests -t ."

# HIP 准入：conformance 全绿 + 容器视图语义（26 个用例）
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/g/Cloumi/aetherflow && HSA_ENABLE_DXG_DETECTION=1 /opt/rocm-env/bin/python -m unittest tests.test_hip_backend -v"

# 直接看结构化报告
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/g/Cloumi/aetherflow && HSA_ENABLE_DXG_DETECTION=1 /opt/rocm-env/bin/python -c 'from aetherflow import run_conformance; from aetherflow.backends.hip import HipBackend; r = run_conformance(HipBackend()); print(r.text()); assert r.ok'"

# 装配层：自动探测在这台机器上应选中 hip；环境变量可强制指定（5.41）
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/g/Cloumi/aetherflow && HSA_ENABLE_DXG_DETECTION=1 AETHERFLOW_BACKEND=hip /opt/rocm-env/bin/python -c 'from aetherflow import get_backend; print(get_backend().name)'"
```

HIP 后端不依赖 cupy / torch：容器是 `backends/hip/runtime.py` 的 ctypes 绑定，编译用
libhiprtc，GEMM 用 libhipblas，目标机只需要 ROCm 运行时。WSL 发行版自带的系统
`python3` 可能没有 numpy，要用自建环境（本例为 `/opt/rocm-env`）。hipRTC 的编译
不需要 GPU，所以"模板能否编译"在没有显卡的机器上也能先验证。
