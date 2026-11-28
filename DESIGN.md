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
| 当前代码 | 80 个算子（IR 里新增 `fft` / `rfft` 两个复数算子、MoE 路由用的 `topk_mask`、分块 SSM 前缀和与因果掩码用的 `cumsum` / `tril`（5.247）、SSM 数值稳定的 `clamp` 与 Engram 哈希用的整数 `mod`（5.248）、weight-only int8 的 `quantize_i8` / `dequantize_i8` / `linear_i8` 与 W8A8 的 `linear_i8a8`（主机侧参考语义见 5.308，设备内核见 5.309：i8 输入、int32 累加，CUDA 走 `__dp4a`、HIP 走标量乘加）、QAT 用的 `fake_quantize_i8`（5.311）、非对称零点的 `quantize_i8_zp` / `dequantize_i8_zp` / `linear_i8_zp` / `linear_i8a8_zp`（5.315），以及交叉熵与任意轴取值用的 `gather` / `gather_backward` / `cross_entropy` / `cross_entropy_backward`（5.286）；CUDA / HIP 各自承接 77 个——73 个自研内核、1 个由反量化与 GEMM 组合的 `linear_i8`、1 个由量化与反量化组合的 `fake_quantize_i8`、1 个前向同源而反向为 LSQ 的 `fake_quantize_i8_lsq`、2 个逐组（per-group）weight-only 反量化 `dequantize_i8_group` / `dequantize_i8_group_zp`（设备内核见 5.321）、2 个非对称量化 / 反量化 `quantize_i8_zp` / `dequantize_i8_zp`（设备内核见 5.322）、2 个非对称融合线性层 `linear_i8_zp`（反量化 + GEMM 组合）与 `linear_i8a8_zp`（int32 累加专用内核，5.323），CUDA 另有 cuFFT 的 fft/rfft；63 个自研内核全部支持 f16——attention 走混合精度模板、存储 f16 而累加在 float；HIP 沿用同一批模板，没有 rocFFT 绑定；`topk_mask` 目前只有参考实现；f8e4m3 / f8e5m2 两个 dtype 及其参考语义见 5.318、fp8 权重的读取入口见 5.319、设备端 cast 的量化内核见 5.325）、9 层模块、3199 个测试（本轮实测：本地 3199 个用例 / 460 跳过；CUDA 真机 A800-40GB 2857 个用例 / 80 跳过——123 条 error 全部落在前端用例，原因是该机 torch 2.5.1 不在兼容矩阵（2.9–2.13）内、被 `UnsupportedTorchVersion` 显式拒绝，与本轮改动无关；CUDA 组连同本轮新增的 13 条全过；WSL HIP（RX 7650 GRE / ROCm 7.2）数字沿用上一轮：2652 个用例 / 373 跳过。上一轮 2×Tesla V100-PCIE-16GB 的 2335 / 120 不再复跑——那台机器已释放，cuDNN 快路径整组跳过与驱动的结论见 5.230）；上一轮新增的 46 条全部是纯 Python 用例（5.204 的 11 条：事件梯度的解析式、有限差分、符号事件数与越界拒绝；5.205 的 35 条：ZeRO 分片与整份逐位对拍、流水两段/三段的单卡对拍与接口拒绝面），本轮新增的 4 条同样是纯 Python（5.206 的逐行约束：按行生效、长度与接口拒绝、未完成按行报错、会话复用与无状态入口一致），5.207 的 12 条也是纯 Python（反向图按目标剪枝 5 条、交换律 CSE 与反向管线 7 条），5.208 的 3 条同样是纯 Python（训练态 BN 与 eager 逐位、分解后同一条路、导入后梯度与 torch autograd 对拍），5.209 的 4 条同样不碰设备（生产内核的线性化由 `Layout` 生成、源码形态与 `cast` 关闭、五个布局常量），5.210 的 23 条同样是纯 Python / 离线（复数 dtype 与 `fft` / `rfft` 的参考语义、拒绝面、序列化，以及 cuFFT 的归一化表与 dtype 登记表——本机没有可用 NVIDIA 显卡，`cufftExec*` 未跑），5.211 的 13 条同样是纯 Python（`sequential`/`gpipe`/`1f1b` 三种流水调度的数值等价与驻留峰值计数、Reduce-Scatter 与 All-Reduce 的逐位一致与接收量、拒绝面），上一轮新增的 20 条同样是纯 Python（5.212 的审计修复：门控 SiLU 反向、AMP 顺序与调度、累积权重与续训事务、分片先校验后提交、硬件包信任记录、归一化标量参数形状、整数入口范围、fuzz 整数除法）；上一轮新增的 23 条同样是纯 Python（5.213 的 bf16 闭环：RNE 量化与两字节打包、NaN 位模式的截断与 quiet 补位、执行器在 bf16 类型边界的量化、autocast 接受 bf16、bf16 训练驻留与拒绝面）；上一轮新增的 14 条同样是纯 Python（5.214 的续训闭环：DataLoader 的轮次/批游标与配置校验、前向图指纹、调度器类型与超参凭证）；上一轮新增的 9 条同样是纯 Python（5.215 的 BN running stats 闭环：EMA 与手写公式/无偏方差对拍、参数不更新时统计量照进、训练统计量接推理图、存档往返与 momentum 拒绝面、设备态与符号通道拒绝）；上一轮新增的 5 条同样是纯 Python（5.216 的组合矩阵：五开关全开的手算对照、窗口内续训、分段重算对整图、f16 主副本、设备驻留与主机态逐位一致），上一轮新增的 13 条同样是纯 Python（5.217 的动态形状契约：契约表与查询、未知环节拒绝、训练循环与分段重算的入口守卫、执行器多形状绑定、部署只要求权重静态、reshape 表达式目标被拒绝、torch 动态维在导入期显式拒绝），上一轮新增的 13 条同样是纯 Python（5.218 的 MoE：top-k 路由与门控归一化、负载均衡损失手算、中心差分梯度、训练损失下降、专家并行拒绝面、序列化往返，以及 `topk_mask` 的掩码语义与两条 conformance 形态），上一轮新增的 7 条同样是纯 Python（5.219 的分块反向：分块图与整块图逐位对齐、广播 mask 跨块求和、逐行 mask 的拼接还原、dropout 不切块、符号序列回退整块、静态峰值下界），本轮新增的 7 条同样是纯 Python（5.220 的服务控制面：排队取消与槽位复用、幂等取消面、上限拒绝与恢复、超时残片与排队计时、非法 timeout / max_pending / clock），本轮新增的 16 条同样是纯 Python（5.221–5.223：集合通信能力声明与 provider 返回值的三类校验、跨进程分片落地与装错片拒绝、torch 兼容矩阵与 dtype 拒绝），本轮新增的 8 条同样是纯 Python（5.224 的 torch 入口补全：int64 token 按 i32 承载并落成 embedding、`F.embedding` 无 padding 路径、padding_idx 与多索引高级索引显式拒绝、提升常量与 `get_attr` 常量落成 `constant`、常量体积上限、越界 i64 在执行器入口报错），本轮新增的 5 条同样是纯 Python（5.225 的续训协议：`Trainer.save_checkpoint` 接管 `DataLoader` 位置、恢复时校验 `data_kind` 与配置、`run_loader` 按 loader 游标推进），本轮新增的 2 条同样是纯 Python（5.226 的审查修订：loader 状态改用专属键 `loader_state`、游标延后到全部提交之后、`run_loader` 委托 `run`、i32 常量值域与矩阵 dtype 在导入入口校验），本轮的 0 条新增与 3 条修订同样是纯 Python（5.227：cuSPARSE 的 32I 枚举值、能力面与内核表的子集关系、CSR 铺法行数按 SM 现算；真机结论见 5.227），本轮新增的 40 条同样是纯 Python 为主（5.228 的 16 条：NCCL 的 dtype / 归约登记表、unique id 的字节往返、communicator 入口与缓冲校验、真机四集合；5.229 的 20 条：跨进程 ZeRO 的分片语义与两步逐位对拍、空片、拒绝面、只写本片的存档、真机两步；5.230 的 4 条：cuDNN 快路径的判定——auto 回落只试一次、显式报错带 cuDNN、f64 与 groups>1 不选、捕获期不切），其中 2 条真机用例在 2×V100 上真跑，设备路径由 conformance 的跨后端用例覆盖；另有 7 个多卡用例（设备切换 5 + 数据并行 2）在 3 卡全可见时单独跑通；本轮新增的 47 条全部是纯 Python（5.231 的 30 条：ZeRO-2 的分段与补零、两步与整份 SGD 逐位对拍、通信元素数、空段、拒绝面、只写本片的存档、loopback 的 ReduceScatter / AllGather；5.232 的 17 条：环境变量优先级与缺省拒绝、unique id 的 TCP 分发与超时、`spawn` 的拉起与退出码、重复 serve 拒绝），一条都不碰设备；本轮再新增 23 条（5.233 的 ZeRO-3：分块计划、拼回与完整参数逐位、一步与两步同整份 SGD 对拍、补零不污染、常驻只占一块、缓存生命周期、拒绝面、分片存档），同样不碰设备；本轮再新增 13 条（5.234 的 ZeRO-3 训练接入：逐段借用的判据、两步与整份 2 段训练逐位对拍、store 与 sync_to_host 的一致性、累积 + 钩子、续训 2+2 与 4 步对拍、七类拒绝面），同样不碰设备；本轮再新增 21 条（5.235 的分片合并与重分片：ZeRO-1/2/3 存档的合并、合并后续训逐位对拍、2→4 与 2→3 的重切落位、全量 checkpoint 切分、十二类拒绝面），同样不碰设备；本轮再新增 19 条（5.236 的主机中转与多进程数据并行：TCP 中转的两 rank 求和与 dtype / count 不一致、`DistributedDataParallel` 与单进程合并 batch 的参数与损失对拍、用户钩子看到归约后的梯度、拒绝面、真两进程 `launch` 对拍），同样不碰设备；本轮再新增 8 条（5.237 的静态执行快路径：与通用路径的输出 / 环境 / dtype / 形状报错逐位一致、符号图与融合组不启用、bf16 仍量化），同样不碰设备；本轮再新增 13 条（5.238 的卷积实现选择：隐式 GEMM 的预算可行性、窗口与权重的预算竞争、超大核拒绝、HIP 的 auto 选隐式 / 1x1 与小 K 不改道、CUDA 的 auto 保持 im2col、显式指定不被改写），同样不碰设备。本轮再新增 23 条（5.239 的 weight-only int8：RNE 的六个平局点与 ±127 端点饱和、非正 scale 的三道拒绝面、逐通道 scale 的广播作用域、量化误差不超过半个 scale、`linear_i8` 与"显式反量化再 matmul"逐位一致、JSON 往返、`dx` / `dbias` 手推对照、量化权重不可导、构建器的权重体积与形状校验），同样不碰设备；三条新 conformance 形态在 WSL HIP 上执行通过（`quantize_i8` / `dequantize_i8` 走同一份 NVRTC / hipRTC 源码，`linear_i8` 走 hipBLAS 反量化 GEMM），CUDA 真机本轮不可用、未执行；本轮再新增 25 条（5.240 的张量并行：切分工具的可整除性 / 越界拒绝与副本语义、列并行拼接等于稠密乘、行并行部分和等于稠密乘、行并行拒绝 bias、头切分的等价、world=1/2/4 的 block 与稠密逐项一致、fused 变体、本地参数形状与体积、两段图过 verifier、切分接口与输入校验的拒绝面），同样不碰设备。本轮再新增 29 条（5.241 的分页 KV cache 与投机解码：页分配的跨页一致 / 回收复用 / 回滚 / 容量失败后原样保留、单层平面数组入口、增量注意力与稠密图逐位对拍（多种页宽 / fused / 单层单头）、四种草稿（完美 / 恒定 / 重复末 token / 停止符）与空草稿弃权下的贪心逐位等价、驻留与关闭幂等），同样不碰设备。本轮再新增 41 条（5.242 的成本模型与组合搜索：`conv_tile_cost` 的派生量与 `plan_conv_tiled` 逐项相等、`cb` / `roc` 的方向性单调、`tile_w` 越界被截断后两个配置的估算完全相同、`roc=3` / 超大 block / 超预算 `cb` / 未知参数四类不可行各带原始原因、逐元素核的 `vec` 与空线程、排序稳定且不可行殿后、预算裁剪后默认配置仍排第一、不裁剪时与 `tune` 平局一致、实测更快的非默认配置照样翻盘、`measure_configs` 的平局与全拒语义），同样不碰设备。本轮再新增 20 条（5.243 的 Chrome Trace 导出：`start_ms` 的采集与 JSON 往返、缺失时刻按结束点累加 / 混合记录只补缺行、GPU 轨道从首个读数起首尾相接且标 `approximate`、多记录 `pid` 递增与 `wall_ms` 错开、空记录仍是合法 trace、重复导出逐字节相同、参数拒绝面、`write_trace` 不留临时文件、CLI 往返，以及包装器实测时刻非负且单调），同样不碰设备。本轮再新增 11 条（5.244 的 RCCL 绑定：`librccl.so.1` / `rccl.dll` 判 HIP 与 `libnccl.so.2` / `nccl.dll` 判 CUDA、候选表覆盖两套 soname、版本整数到三段串的三个边界、库缺失时 `vendor()` 与 `version_string()` 抛 `NcclUnavailable`、HIP `DeviceArray` 的四个属性过 `_pointer`、能力理由按厂商给出 `RCCL x.y.z` / `NCCL x.y.z` / 加载不到三种文本、以及 WSL HIP 上 world=1 的真机四集合），其中 10 条纯离线、1 条真机。本轮再新增 19 条（5.245 的设备端循环：宿主路径与设备路径逐位相等、重复 `run` 的轮数复位、两张图各捕获一次且同流、每轮同步数 = 轮数 + 1、状态零重上传、嵌套区域 / 缺能力 / 缺初值 / 换形状换 dtype / `max_steps` 用尽的拒绝面，以及 WSL HIP 上一条真机），其中 18 条离线、1 条真机。同一批再新增 23 条（5.246 的 NHWC 布局：conv / pool 的 nhwc 与 nchw 逐位一致、NHWC 下权重与 bias 不翻转、非法 layout 与 NHWC 反向规则的拒绝、后端布局声明的三种形态、传播 pass 的单组 / 链 / reshape 切断 / 多组 / 权重来源剔除 / 边界转置计数、后端不声明时的拒绝、pass 不修改输入图，以及 WSL HIP 上一条"设备端显式拒绝 NHWC"），其中 22 条离线、1 条真机。跳过数的差异来自机器缺哪一组硬件） |
| 已闭环的能力 | torch.export → IR → 校验 → 参考执行 → 中层优化 → 融合 → CUDA 后端对齐 → 运行期计划（CUDA Graph）→ 捕获期内存复用 → IR 序列化与版本治理 → 后端一致性（conformance）套件 → 插件契约分层（准入面 + 可选契约）→ 算子注册与调度 → 卷积与池化端到端（IR 语义 + 分块直算 / im2col/cuBLAS + 自研 maxpool kernel）→ CNN 前端导入（LeNet 端到端三方对齐）→ 性能基线与回归判定 + CI 配置 → 自动微分（语义 IR 层反向展开，反向图可序列化、可执行）→ 训练闭环（参数容器 / SGD+Adam / 图内损失，收敛性有测试）→ 六类模型构件（MLP / CNN / RNN(LSTM/GRU) / 多层 Transformer / Mamba block / 多层 SNN）→ 随机算子（seed 化 dropout_mask，跨后端逐位可复现）→ 批归一化（训练态 BN + 独立 stats 算子，反向全部复用已有语义）→ 构件层接入 BN 与 dropout（开关在构图期，同一份代码构建训练图与推理图，5.26）→ 权重存档与优化器状态恢复（续训轨迹与未中断逐位一致，5.27）→ 可复现的 DataLoader → f16 图与梯度缩放（把种子当缩放因子，图不用改，5.28）→ bf16（RNE 量化、2 字节驻留、autocast 与训练闭环，5.213）→ 融合注意力（一个 IR 节点，反向由已有算子组合，5.29）→ 插件扩展面（命名空间护栏的自定义算子注册 + 多实现按代价选择 + 开发指南，5.30）→ 训练辅助（warmup / 余弦与阶梯调度、全局范数裁剪、梯度累积、EMA，5.31）→ CUDA 融合注意力前向（分块 online softmax，分数矩阵不物化，5.32）→ CUDA f16（一套模板 + dtype 占位符展开、hgemm / hgemm_batched、卷积走 im2col+hgemm，5.33）→ 分组 / depthwise 卷积（按组切片复用 im2col，分块核明确拒绝 groups>1）与自适应 / 全局池化（PyTorch 的 floor/ceil 窗口规则、重叠窗口原子累加，5.34）→ CUDA 承接 SSM 线性递归（前向与伴随，一个尾元素一个线程、沿时间串行；5.155 起长 T 自动走分块，短 T 仍串行）→ SNN 稀疏事件表示与事件内存池（CSR 变体、只读事件的发放率、跨前向复用下标缓冲，5.35）→ 常量折叠（`constant` 承载任意张量 + PORTABLE 子图求值，5.36）→ 深层融合（LayerNorm+残差、SwiGLU 都表达为已有算子的输入扩展，5.36）→ 编译期静态显存预算（liveness 直接从 IR 算出峰值与复用倍数，5.36）→ 自动调优（参数空间 + 网格搜索 + 按设备/形状命中的配置库，5.37）→ 运行时数据上报（逐节点派发耗时、整图 wall、显存快照，5.37）→ 只读智能体查询面（后端画像 / 性能摘要 / 配置推荐，不执行不改图，5.37）→ IR 差分模糊测试（随机图 + 五条不变量 + 非法属性变异，5.38）→ CI 的模糊测试作业与完整 GPU 作业（5.38）→ 跨运行吞吐 / 显存趋势（5.38）→ 真实模型导入覆盖（7 个 torchvision 模型端到端对拍，推理态 BN / 精确 GELU 由前端分解承接、ceil_mode 池化改直通 IR（5.154），5.39）→ HIP/ROCm 第二硬件后端（复用同一批内核模板，hipRTC 运行时编译 + hipBLAS + ctypes 容器，conformance 62 executed / 5 refused / 0 failed，与 CUDA 逐项一致，5.40）→ 后端装配层（内置三工厂 + pip entry point，按名 / 环境变量 / 自动探测挑选，退回参考后端必须可见，5.41）→ L2 内核 IR（语法级：25 个表达式 / 语句节点 + C 语法打印器 + 方言数据；逐元素二元 / 一元 / step / 广播 / 归约 / softmax / LayerNorm（含融合残差）/ RMSNorm 九个内核族已接管生产路径，与手写模板逐 token 相等，CUDA 真机 conformance 62 / 5 / 0，5.42）→ zip 硬件包（manifest 校验 + 安全解压 + 目录扫描 + 装完即用，24 个用例，5.43）→ 示例硬件包 novac（一份方言 + 三方法后端类，装上 zip 即可打印内核并跑图，6 个用例，5.44）→ 归一化内核迁到 L2（Widen / Narrow / Let(type_name=None) 三个节点由模板倒推，LayerNorm / 融合残差 / RMSNorm 共 7 个新用例，5.45）→ 门控 SiLU 与批归一化迁到 L2（零新节点，批归一化模板拆成两个内核，6 个新用例，5.46）→ 池化家族迁到 L2（`AtomicAdd` / `Continue` 两个语句节点、方言新增可选的 `atomic` 能力；maxpool2d / avgpool2d / 自适应池化及三者的反向共六个内核，3 个新用例，5.47）→ 拼接 / 随机 / 扫描 / 卷积梯度迁到 L2（`Store(raw=True)` 裸赋值与内核私有 `helpers` 两项能力；concat_copy / dropout_mask / scan_linear 及其反向 / 卷积输入与权重梯度共六个内核，7 个新用例，5.48）→ im2col / 分块卷积 / 融合注意力迁到 L2（`Not`、`Shared(extern=True)` 动态共享内存、`ArrayDecl(values=None)` 三项能力；9 份字符串模板至此全部迁完，L2 成为唯一生产路径，6 个新用例，5.49）→ 归约族补到 max / mean / argmax（`AtomicMin` 语句节点，三者共用一份骨架，argmax 输出 i32 且并列取最小下标，7 个新用例，5.50）→ 转置卷积导入（不新增算子：`ConvTranspose2d` 直接落到 `conv2d_input_grad`，`output_padding` 由 `input_shape` 表达，3 个前端用例 + 1 条 conformance 形态，5.51）→ 语义级 L2（相邻逐元素算子融合成一条链，中间值不物化；f32 链在长度能被 4 整除且指针 16 字节对齐时走 `float4` 四通道，10 + 7 + 5 个新用例，5.52）→ 前端承接新版 `aten.batch_norm` 的 eval 形态（5.52）→ 符号形状表达式（`Dim = int | str | SymExpr`，卷积 / 池化的派生空间维留在 IR 里，一次构图多分辨率执行，7 个 IR + 3 个 GPU 新用例，5.53）→ 执行计划层（`lower/plan.py`：block / grid / vec / shared 的决策与校验变成纯函数，后端只剩 launch，41 个离线新用例，5.54）→ 共享网格步进骨架（`grid_stride` 一处定义，逐元素单算子与融合链共用；归约暂存区改由 `REDUCE_SCRATCH` 供给，源码逐 token 不变，5.55）→ 分块卷积接入调优面（conv2d 的 `tile_w` 成为旋钮：计划层构造期校验 block 上限，调优空间首候选是"自动"，调优图集补上 conv2d 用例，7 个新用例，5.56）→ 多维索引核实（逐字比对后只合并了 `conv2d_input_grad` 的一处真实重复，其余差异是语义差异，5.57）→ HIP 调优面修复（`HipBackend.__init__` 漏掉 conv2d 的 `tile_w` 覆盖字段，使 HIP 上分块卷积在 launch 前抛 `AttributeError`；补一行并用 AST 对账两份 `__init__` 的实例字段，1 个新用例，5.58）→ 基准 CLI 的后端选择改为查注册表（`--backend` 从写死的 `ref/cuda` 变成 `ref` + `registered_backends()`，三个脚本共用一个工厂；明确指定却装不起来时一行拒绝，绝不退回 CPU，4 个新用例，5.59）→ WSL ROCm 实跑（DXG 直通下 `HSA_ENABLE_DXG_DETECTION=1` 是前置；补上 `_Stream.synchronize`——缺它时 HIP 上调优的 9 个用例全部 `AllRejectedError`；conformance 13/13、全量 760 通过，`tune` 搜出 9 条配置并逐位对拍一致，2 个新用例，5.60）→ HIP 运行期图捕获（hipGraph：模块级流栈 + `_Stream` 上下文、捕获期自动切异步拷贝、分配器钩子与 `hip/memory.py` 的 arena、`HipGraphPlan`；`reuse_memory=False` 明确拒绝，含 hipBLAS 的图在 WSL 真机捕获并 replay 通过，8 个新用例，5.61）→ HIP 性能对照（`benchmarks/capture_compare.py` 在同一套 8 个用例上跑"捕获 replay vs 逐节点下发"两条口径，`--backend` 走注册表、先对拍数值再计时；顺带修掉捕获期扩容导致 4 个用例捕获失败的真缺陷——`Arena.recycle()` 收回预热轮的地址，2 个新用例，5.62）→ 跨后端调优对照（`benchmarks/tune_compare.py` 两段式：采集复用 `tune.TUNE_CASES` 与 `tune.search`，报告记录赢家、默认候选与机内提升；对照只比配置是否相同与各自"相对默认"的提升，不比跨机毫秒。9 个单算子用例在 CUDA 与 HIP 上各采两次，赢家两次之间只有 2/9 与 4/9 一致、跨后端 2/9 与 0/9 一致；重复测量与交替 A/B 证明 1M 逐元素算子的 `block` 真实差异 ≤3%，淹没在整机 30% 的漂移里，"最优 block"不可判定；9 个新用例，5.63）→ 硬件包的内容摘要与依赖声明（manifest 新增两个可选字段：`files` 是逐文件 sha256，声明了就要求与包内文件**完全一致**（多一个未声明文件也拒绝，`__pycache__` / `*.pyc` 是运行期产物不计入）；`requires` 声明用到的算子（必须在框架算子表里）与 Python 模块（用 `find_spec` 查，不 import）。安装期与扫描期都校验，坏包按既有策略跳过；摘要只证明"包在打包后没被改过"，不证明"谁发的"，来源签名仍没做；示例 `build.py` 改为现算清单；16 个新用例，5.64）→ 自动分块（`lower/plan.py` 的 `auto_block` 把 block 的取值变成纯函数：不派空线程 + 网格至少铺满一屏 SM 两条规则，`block=None` 成为"自动"这一可调优、可回退的状态，22 处 launch 收敛到一处解析，12 个新用例，5.65）→ 训练循环（`train/loop.py` 的 `Trainer`：反向图建一次，种子名/形状与多输出图的零种子由它负责，符号形状在构造期拦下并解释根因；9 个新用例，5.66）→ 硬件包来源签名（manifest 可选 `signature`：Ed25519 覆盖"除 signature 外的整份清单"，信任锚由安装方的 `trusted_keys` 给出，包内公钥不能自证；带签名不给信任键、给了信任键包未签名、公钥不受信三种情况都拒绝，未签名且未要求来源仍可装，9 个新用例，5.67）→ 动态执行第一版（`DynamicRunner` 按输入形状建图并按 LRU 复用，`build` 只能依赖形状，数据相关分支留在两次 `run` 之间；一次 `run` 内部仍是静态图，10 个新用例，5.68）→ cuDNN 对照（`benchmarks/cudnn_compare.py` 用 ctypes 直调 cuDNN 9 的 legacy API，同形状同 dtype 同时刻对照：自研分块核慢 4～12 倍、im2col+cuBLAS 慢 3.2～8.8 倍，中等通道的 3x3/5x5 差距最大；只做测量，不接生产路径，5.69）→ CI 作业真机逐条实跑（正确性三步 + 调优往返 + 趋势采集全绿；性能门禁为红，基线不重采，5.70）→ 内核源码惰性生成（`_kernel` 只在编译缓存未命中时才渲染源码，31 处 `codegen.render` 改惰性，分块卷积的缓存键补上 dtype；conv2d 单步 1.19 → 0.45 ms、cnn_lenet −61%，门禁差距由 3.4～4.5 倍收到 1.3～1.6 倍，5.71）→ 执行计划缓存（融合分组 / 静态 attrs / 期望类型 / 铺法只在建图后算一次，`_contig` 跳过冗余的连续性转换；host 侧 transformer 2.09 → 1.89 ms、cnn_lenet 端到端 −14%；以 `matmul_256` 为整机漂移标尺归一化后，卷积与 CNN 已追平、只有 transformer_block 还剩 23%，5.72）→ conv2d bias 融合（分块核写回时就地加 bias，省掉每步一次 `(1, O, 1, 1)` 的广播核；三处同改：L2 参数与写回表达式、手写对照模板、launch 传参）与符号图解析备忘（`_GroupPlan` 按本次绑定记住 attrs 与期望类型，形状切换即失效；卷积 −11%～−13%、transformer 1.495 → 1.362 ms）；门禁 8 个用例全绿、基线仍未重采，按 `matmul_256` 折算后只剩 transformer_block 的 8%，5.73）→ conv2d 的两条快路径（cuDNN 接成 `conv_impl="auto"` 的可回落实现，分组与 f64 仍走自研核；分块核的 `roc` / `cb` 也进调优面，5.74）→ 激活重算（`train/recompute.py` 分段前向只留边界、反向逐段重放，链式模型梯度与整图逐位一致且静态峰值下降，5.75）→ 部署包（`deploy.py` 把图 JSON + 权重 + 清单摘要装进一个 `.afpkg`，加载即校验、跨进程逐位一致，5.76）→ eager 前端的即时求导（`loss.backward()` 建一次反向图并按图版本缓存，仍是静态图不是 tape，5.77）→ 热路径上的三个 Python 常数（错误信息惰性化、dtype 在计划期解析、去掉多余小分配；门禁另加 0.5 s 热身以消除空闲降频误报，折算后 3.6%，5.78）→ 常规执行路径接进 CUDA Graph（`graph_mode="auto"` 按"图 + 结构版本 + 输入签名"缓存捕获计划、LRU 4 份、捕获失败回落逐节点；`transformer_block` −56%、`cnn_lenet` −55%，单算子图无稳定收益，默认仍 off，7 个新用例，5.79）→ 基准补"权重常驻"口径（`Case.split` + `prepare_case`，参考后端仍用完整输入；`transformer_block_resident` 1.0206 → 0.2179 ms、`cnn_lenet_resident` 0.5758 → 0.1264 ms，叠加 auto 后分别比基线 −81.6% / −81.2%，2 个新用例，5.80）→ 自动图按节点数开闸（`min_graph_nodes=2`，单算子图不进捕获、多节点图照旧；反向顺序复测证明 5.79 记的 `matmul_batched` +8% 是整机漂移，1 个新用例，5.81）→ CUDA 训练闭环有了门禁（`Trainer + CudaBackend`：线性回归收敛、CNN 一步与参考逐位一致、30 步损失下降），同时修掉换机才暴露的 cuDNN 描述符指针 bug（3 个新用例，5.82）→ 训练 step 进图（前向、反向各一张捕获计划，10 步的损失与全部参数和逐节点执行一致）与视图内存的生命周期（arena 按视图引用计数延迟回收，捕获峰值不涨；2 个新用例，5.83）→ 参数常驻显存的训练步（`Trainer(device_state=True)`：`capture` 新增 `shared_buffers`，前向与反向共用同一组权重缓冲，优化器在显存上原地更新，一步只有损失标量回主机；SGD 10 步与 Adam 5 步的轨迹都与主机态一致，2 个真机用例 + 1 个换表用例，5.84）→ 训练辅助工具跟着参数留在显存（`train/_arrays.py` 的 `array_module` / `to_host` / `host_scalar`，`clip_global_norm`、`GradientAccumulator`、`EMA`、`checkpoint`、`GradScaler.unscale` 全部按数组所在模块分派，设备态下只在取标量时同步一次；`grad_hook` 因此可以直接放裁剪与累积，2 个新用例，5.85）→ 数据侧 pinned 暂存（`pinned_array` 按 `(名字, 形状, dtype)` 复用固定内存，`replay` 里 numpy 输入走异步 H2D，共享缓冲直接跳过；配合"进流前先等一次当前流"的事件依赖，每步最后一次 H2D 的等待消失，5.86）→ HIP 对齐同一套运行期语义（`shared_buffers` / `stream` / `run_device` / 自拷贝跳过；不接 pinned、不支持设备态训练都在第一次 step 明确报错，5.87）→ 设备态叠加激活重算（逐段捕获、逐段 replay，全部共用一条流；边界值与 cotangent 分两组缓冲，数据输入只在首个使用它的段里走 pinned 上传；本地用 numpy 替身后端验收调度，1 个真机用例，5.88）→ 设备管理器与分层分配器的契约上提（`StreamContract` 独立成协议、`AllocatorContract` 只收两个 Arena 共有的五个方法、`MemoryContract.allocator()` 交出**新**分配器；`InstrumentedBackend` 按存在性转发新契约——这条是真机全量第一次跑才暴露的；同时给基准报告加机器指纹，换宿主后整体慢 1.9～2.7 倍因此只提示、不覆盖基线，3 个新用例，5.89）→ 自动混合精度（`passes/autocast.py` 的图重写 pass：matmul / matmul_batched / conv2d 降 f16，softmax / 归一化 / 归约强制回 f32，其余取最宽输入，图输出恢复原 dtype 因而幂等；反向不用特判，梯度天然落回 f32。真机上顺带修掉一条从未被走到的缺陷——`CudaBackend.cast` 到 f16 走 cupy 的 `astype` 会链接不到 `__half` 的构造函数，改为自研 PTX 内核 `ew_cast` + 只给这条内核用的 `CAST_DIALECT`（HIP 侧同一份描述换 `_Float16` 方言）；另把 torch 2.13 导出图里的 `aten.relu_` / `aten.add_` 降到函数式同名版，20 + 1 个新用例，5.90）→ 混精接进训练循环并量化（`Trainer(mixed_precision="f16")` 在构造期改图，反向图与两张捕获计划自动跟随；参数仍是 f32，优化器与存档不用改。真机 7 组口径实测：单算子 f16 GEMM 确实更快，但训练步在主机态慢 6～10%、设备态基本持平——重写把前向节点 11 → 17、反向 39 → 51（其中 12 个是 cast，因为 cast 的 VJP 也是 cast），主机态每多一个 cast 节点约 0.08～0.12 ms，与 GEMM 省下的时间同量级。开关默认关闭，6 个新用例，5.91）→ 换宿主后的基线重采（8 个用例齐平慢 52%～176%，形状确认是宿主差异而非回归；新基线带 machine 指纹、覆盖 10 个用例，把 5.80 的两个 resident 口径一并纳入；GPU 作业五步在真机复跑全绿，门禁由红转绿；托管平台仍未实跑，因为这份工程没有 git 仓库与 runner 令牌，5.92）→ 剥离派发后的 GEMM 对照（单算子方阵图、输入全常驻、`graph_mode="auto"` 只 replay：f16 相对 f32 为 1.19×（batch 128）/ 1.33×（batch 1024）；把 5.91"消掉 cast 就能拿回收益"的说法收窄成"必要条件而非充分条件"，混精在这个规模上的价值是省显存；顺带让 `cast` 的 VJP 在上游梯度已是输入 dtype 时直接透传，2 个新用例，5.93）→ 外部审计的 P0：softplus 溢出（`log(1+exp(x))` 在 f32 上 x>88 就 inf，inf 经 `a = exp(dt*A)` 污染整条 scan、反向变 NaN；改成 `max(x,0) + log(1+exp(-|x|))`，指数入参被压到非正半轴、两头都不溢出，绝对值用 `relu(x) + relu(-x)` 拼、不新增算子；3 个新用例，5.94）→ 外部语义审计的逐条核对与修复（报告基于 `853 通过 / 291 跳过` 的旧快照，先按当前代码逐条复核：AF-02 已在 5.94 修掉、AF-29 由 5.84 的设备态部分解决，其余仍成立的条目里本轮修 13 条可复现的语义红线：交叉熵改稳定 log-softmax、`Value` 走对象身份、自动 SSA 名避让用户输入、序列化显式写节点输出名、中间值可作 `wrt`、分段重算种子与下游 cotangent 相加且只对浮点输入求导、融合链与 CUDA Graph 的缓存键补全、优化器两阶段提交与槽位完整性、checkpoint 先校验后原地写、Adam 拒绝 f16 参数；14 个新用例，5.95）→ 阶段 A 收尾（attention 的加性偏置在广播时按被广播轴求和返回梯度——旧规则只在形状完全相同时给、把可学习的位置偏置当成常量；前端按 ATen schema 归一化参数读取：`add`/`sub` 的 kw-only `alpha`、`layer_norm` 的**位置** `eps`、SDPA 的 kw-only `scale` 与 `enable_gqa` 此前都读错了位置、会被静默换成默认值，另加"不认识的参数就拒绝"；6 个新用例，5.96）→ 阶段 A 续（未激活输出按可达性剪枝，不再靠喂零种子——零种子在 `0/x` 上仍是 NaN；`backward` 新增 `active=`，训练器与分段重算都只给损失那一个种子；符号维与静态非 1 或另一个符号的广播改为构图期拒绝，IR 里没有能表达"这一维可能是 1"的广播表达式；形状符号禁止与 dtype 名重名，否则 `f32[f32, 4]` 与属性里的 `dtype="f32"` 无法区分；输入在执行入口转换到 IR 声明的 dtype，不再"按 f64 算、输出才降 f32"；10 个新用例，5.97）→ 阶段 A 收官（AF-16：捕获期把越界标志登记进计划，replay 同步后复查、报错前清零，普通执行与图重放共用同一条错误消息，HIP 对称实现；AF-20：`dropout_mask` 增加运行期 i32 key 输入与 `offset`，随机源因此放在设备内存里、捕获后 replay 仍能重新采样，`Trainer(rng_inputs=...)` 每步推进，同一 step 的前向 / 分段重算 / 反向共用同一个 key；11 个新用例，5.98）→ 阶段 B 的第一块：SSM 流式推理（`build_selective_scan_stateful` 导出末状态、`reset` 掩码把递归在重置点切断；`build_mamba_block_stateful` 让卷积窗口与 SSM 状态都成为图输入/输出，`steps=1` 的同一份构图就是 decode；三条恒等式——任意分块 == 整段、逐 token == prefill、带 reset == 各段独立跑——在参考后端与 CUDA 上都成立，13 个新用例，5.99）→ 阶段 B 续：多层 Mamba 栈（`_mamba_layer` 抽出单层主体，`build_mamba_model` / `build_mamba_model_stateful` 把 N 层放进**同一张**图，键带 `l{i}_` 前缀；状态出参只差输入名末尾那个 0，`spec.carry_state` 负责翻译，分块续接就是 `feed.update(...)`；`with_steps(1)` 的同一份构图即 decode；单层退化与 `build_mamba_block` 逐位一致，整栈的分块 / 逐 token / reset 三条恒等式在参考后端与 CUDA 上都成立，7 个新用例，5.100）→ 阶段 B 续：Mamba 专属初始化（通用初始化让 `A_raw≈0`、`dt_b=0`，于是 `a≈0.5`、所有通道半衰期都约一个 token，长依赖没有梯度压力；`init_mamba_params` 改成直接指定目标半衰期网格——`(d_inner, d_state)` 每个单元在 1～32768 上对数等距分到一个 `H`，`dt_b` 逐通道在 `[1e-3, 1e-1]` 上对数等距给出名义 Δ，再由 `|A| = log2/(H·Δ)` 反解 `A_raw`，`dt_w` 初始化为零以免激活幅度盖过分层；`effective_half_lives` 用同一把尺子报告训练前后的分布，`MambaModelSpec.init_params` 逐层生成、靠名字前缀区分随机流；4 个新用例，5.101）→ 阶段 B 续：流式状态进存档与 AF-27/AF-28 量化（`save_checkpoint` / `load_checkpoint` 新增可选 `state`：键直接用图输入名 `l{i}_conv0` / `l{i}_h0`，加载后就是 decode 的 feed，模板原地写入并校验形状 / dtype；"传了目标却没存档"报错、"没传目标"忽略，旧格式存档照常可读；判据是 prefill → 存档 → 新进程式重建 → 逐 token decode 与从未中断**逐位**一致。AF-27 用 `plan_memory` 固定下来：节点数恒为 14，T 翻倍时中间值与峰值按每 token 6848 / 3232 B 线性增长；`state_shapes` 的字节数与已处理 token 数无关，"每 chunk 恒定状态"的边界因此可量化。AF-28 在 RTX 2080 Ti 上实测：14 节点整图在 T ≤ 512 时被 14 次 launch 的 ~1.6 ms 固定开销主导，隔离出的单节点 `scan_linear` 在 4096 → 16384 严格 4 倍、65536 时每 token 涨 35%，batch 从 1 到 16 耗时按总元素数增长、不摊薄——只记录口径，不改 kernel、不宣称加速；8 个新用例，5.102～5.103）→ 阶段 B 收官：混合长度与恒定显存（left padding + 有效起点 reset 让 batch 内不同长度的样本等价于各自单独跑，right padding 做不到——pad 会继续推进状态、末状态对不上；reset 是逐样本的，同一 batch 里一个样本重开一段不影响其他样本；真机上预热后连续 200 步 decode，cupy 默认池的 `used_bytes` 恒为 512 B、增量为 0，`total_bytes` 停在 56,320 B——池扩到这么大就不再要，历史既没被物化、临时量也被复用；3 个新用例，5.104）→ 审计 AF-31：预算口径跟着执行策略走（`plan_memory` 新增 `release` 参数，与 `Executor.run_device` 的同名开关一一对应：`False` 时所有值活到图结束、峰值等于总量、`reuse_ratio` 恒为 1；`MemoryBudget.release` 与 `as_dict()` 让引用数字必须带口径。`Executor.run` 固定走 `release=True`——它只读输出、不返回环境，普通执行因此真的回收中间值，而 `run_device` 保留"交出完整环境"的契约。峰值算法从逐时刻扫描改成事件扫描（O(V log V)），与测试里的暴力实现逐项对照。文档把"严格下界"改写为"指定存储假设下的 IR 估计"，并列出三条假设与方向性偏差：视图高估；内核临时区 / 库 workspace / 优化器槽位 / 捕获缓存 / 主机副本低估。真机 6 段 `relu(matmul)` 链：`run` 池总量 20.0 MiB、`run_device` 56.0 MiB，静态估计 8 / 48 MiB，差出来的约 12 MB 正是声明不算的 workspace；3 个新用例，5.105）→ 审计 AF-32 的图 seal（`Graph.seal()` 把"构建后只读"变成代码：`input` / `add_node` / `output` 统一拒绝、`Value` / `Node` 字段不可写、`Node.attrs` 变只读映射且形状类 list 转 tuple；`serialize` 的嵌套属性判断从 `dict` 放宽到 `Mapping`，seal 后的图照常导出；`revision` 因此是真正的不可变凭证，执行计划与捕获计划不会在"图已变、键未变"时静默命中；3 个新用例，5.106）→ 审计 AF-32 的后端能力契约（`BackendCapabilities` 是 `frozenset` 子类，集合语义与既有对账代码完全兼容；为算子附 `OpCapability` 明细：dtype / axis / layout 是排除项、`None` 表示未声明限制，backward / capture / deterministic 是用途承诺；`refined` / `unrefined` 让"只有名字"可以被单独看见。参考后端逐个声明 49 个 PORTABLE 算子的形态：backward 与算子表 VJP 一致、capture=False、deterministic=True，dtype / axis / layout 不设限；`verify` 与 `unsupported_ops` 在名字都命中时继续按输出 dtype、axis/axes、layout 判定，把"算不了这一形态"提前到构图期；conformance 报告区分"带形态明细"与"仅算子名"，只声明名字的旧式后端行为不变、不判失败；8 个新用例，5.107）→ 审计 AF-32 的算子效果描述（`ir/ops.py` 新增 `Effect` 枚举 RNG / STATE / IO 与 `Op.effects` / `Op.pure`，空集即纯函数，STATE / IO 汇总成 `UNREPLAYABLE_EFFECTS`；CSE 不合并非纯节点、DCE 把非纯节点当可达性根（`saved()` 只算纯节点）、常量折叠删掉 `_RANDOM_OPS` 白名单改判 `op.pure`、`CheckpointPlan` 在构造期拒绝会被重放两次的 STATE / IO；内置算子表里只有 `dropout_mask` 声明 `RNG`——它不能折叠也不能合并，但显式 key 下可以在激活重算里重放；11 个新用例，5.108）→ 审计 AF-32 的 HIP / CUDA 大类拆分（`backends/device_backend.py` 新增 `DeviceBackend(Executor)`：110 个成员、49 个算子面、执行计划 / 融合分组 / 调优 / 捕获缓存全部中立，厂商只留 9 个钩子；`CudaBackend` 与 `HipBackend` 变成兄弟类，继承树从 `HIP → CUDA → Executor` 改成两条 `→ DeviceBackend`。顺带拆掉 `hip/runtime.py` 里"假装是 cupy"的 `_CudaNamespace`，`current_stream()` 成为公开函数；拆分过程中修掉两个真 bug——`memory_stats` 被错放在 CUDA 一侧（HIP 上 `AttributeError`）、`transpose` 读视图的 `.base`（HIP 容器没有该属性，conformance 两个用例一直 FAIL）；`TestHipInitParity` 换成 `TestBackendSplit`，断言 MRO 与构造字段集；本地 1298 / WSL HIP 1298 / CUDA 真机 1298 三处全绿，5.109）→ 审计 AF-09 的交错输入序列化（根因已由 5.95 的"节点输出显式写名字"解决，本轮把证据补齐：审计原例现在 `to_dict(loads(dumps(g))) == to_dict(g)`、二次 `dumps` 逐字节相同、`verify` 无 issue、执行输出逐位一致，回读只把值 id 由 `[0,2]` 重排成 `[0,1]`。因此不采纳"input() 在节点出现后拒绝"这条短期方案——它假定格式表达不了交错，而该前提已不成立，加限制反而堵死 eager 的按需声明；需要守的是"名字到值的双射"，两个方向已在 `input()` / `add_node()` 上堵住。同时把"值 id 不跨序列化稳定、契约是名字与类型"写进 `serialize` 模块说明与两处 docstring，并修掉 `models/transformer.py` 里"输入必须在节点之前是硬约束"的过期注释；1 个新用例，5.110）→ 审计 AF-15 的收尾（5.83 的视图引用计数只覆盖 `reshape` 与恒等 `transpose`：`slice` 与退化成恒等的 `broadcast_to` 同样返回零拷贝视图，而 slice 的 `ptr` 带偏移，按视图地址扣账一条也匹配不上；HIP 侧则根本没有 `dispose`，捕获期中间值从不回收、复用收益为零。抽出 `_shared_view` 让四个返视图的算子统一登记，`out is src` 也登记——执行器按两条 Value 各 dispose 一次，第二次靠新增的 `Arena.pending` 认出同一段地址、扣账而不是重复 free；HIP 的 `Arena` 补齐 `alias` / `unalias` / `pending`，`DeviceArray` 加 `_view` 标记，`dispose` / `_alias_view` / `_shares_storage` 三个钩子接上。负向对照：短路登记后 CUDA 新旧两条别名用例、HIP 的 slice 用例全部失败（100% 元素错），恢复后三处全绿；3 个新用例，5.111）→ 审计 AF-19 的收尾（`save_checkpoint` 不再直接 `np.savez(target)`，改成同目录临时文件 + `fsync` + `os.replace`：写到一半被打断时旧存档原样保留、目录里不留半成品；`load_checkpoint` 的提交顺序改成 读取 → 参数校验 → 状态模板校验 → 预转换 → 优化器提交 → 原地写入，设备态的上传也挪到提交之前，坏状态模板不会再留下"优化器已恢复、参数还是旧的"反向半步。负向对照：退回直接 `np.savez` 时旧存档被写成半截，把优化器提交挪回状态校验之前时 `step_count` 变成 1；3 个新用例，5.112）→ 审计 AF-33 的统计口径与失效版本（`tune` 的 `median_ms` 现在真的是中位数，`best_ms` / `p95_ms` / `spread` 分开报；新增 `tie_epsilon=0.02`，差距不显著时保留默认候选。`STORE_VERSION` 与调优对照报告都升到 2，`best ≤ median ≤ p95` 在加载期校验、v1 显式拒绝并说明两个口径的差别。捕获失败从"整实例关掉"改成按图隔离：`_graph_failures` 记原因、`graph_capture_failure(graph)` 可查，同图同版本不再重试，其他图照常捕获。调优覆盖 / conv2d 旋钮 / 驻留权重统一走 `_invalidate_plans`：`_config_epoch` 递增并清铺法、捕获计划与失败记录三层，epoch 也进计划键。负向对照：退回按最小值排名、去掉阈值、让失败原因恒为空，对应用例全部失败；HIP 上用 `HipBackend` 实测三层缓存一起失效。HIP 上实跑调优往返：9 条 v2 记录、`--apply` 命中 9 条且逐位一致，3 个用例在阈值内保留默认；过程中修掉 `benchmarks.run._sync` 只认 `cp.cuda` 的缺陷（5.109 拆掉假命名空间后 HIP 上第一次 `measure` 就 `AttributeError`），补上模块级 `hip.runtime.synchronize()`。本地 1318 / WSL HIP 1318 全绿，CUDA 真机因研究机停机未复跑；13 个新用例，5.113）→ 审计 AF-23 的稳定性诊断（`decay_diagnostics` 把"Δ>0、A<0 ⇒ 稳定"换成可检查条件：`q=max(a)` 必须严格小于 1、`M=max|Δ·B·x|` 必须由输入给出，两者齐了才报 `q^t·|h0|+M/(1-q)`，否则在 `state_bound_reason` 里说明缺哪一条；同时报 Δ / log a 的分位、非有限值与索引、以及 f16 下舍入成 `a=1` 的单元数——初始化里最长半衰期的通道在 f16 下必然退化，报告而不掩盖。负向对照：放宽成 `q<=1`、无输入时把 `M` 填 0，对应用例失败；8 个新用例，5.114）→ 审计 AF-25 的 Mamba 口径与扫描对拍（模块说明明确"受 Mamba 启发的变体"、三处结构差异、不承诺官方权重兼容与性能继承；新增 `mamba_reference.official_selective_scan` 按 mamba-ssm `selective_scan_ref` 的语义逐步递推，与 `build_selective_scan_stateful` 在 `h0` / `reset` / `D` / 门控下逐位对拍；`map_official_params` 只映射 `A_log→A_raw` 与 `D→D_skip`，其余项列入 `unmapped`，`from_official_checkpoint` 因此拒绝整块迁移并解释三处差异。负向对照：翻转参考实现的 A 符号、跳过 `unmapped` 检查，对应用例失败；7 个新用例，5.115）→ 审计 AF-30 的前向 mask（`attention_fwd` 新增 `mask_shared`：`(1,1,S,T)` 的因果 mask 不再被复制成 `(BH,S,T)`，内核按 `(mask_shared ? 0 : bh)*S+s` 寻址，主机只在"最后两维正好 `(S,T)`、其余维全 1"时共用一份，`(1,S,1)` 这类可广播但不是 `(S,T)` 的形态仍走广播复制；HIP 实测同形状共享路径分配 5.0 MB、全量 mask 36.0 MB，差 31 MB 正是 `BH-1` 份。第一次改动落在 `cuda/codegen.py` 的非生产对照模板上，HIP 对拍 82% 元素不一致才定位到生产内核在 `lower/kernels.py`；3 个判断用例 + 1 个 HIP 对拍，5.116）→ 审计 AF-30 的反向量化（先用 `plan_memory` 量化再决定要不要写内核：B=8 / H=4 / S=T=512 / D=64 时一份 `(BH,S,T)` 缓冲 32.0 MB，反向图 26 节点，`release=False` 总量与峰值都是 439.1 MB（≈13.7 份），按"最后一次使用"回收后峰值 108.1 MB（≈3.4 份）。真实路径固定 `release=True`——`Executor.run` 与 CUDA/HIP 的 `capture` 都传它，所以 439 MB 只是"直接调 `run_device` 又不传 release"的上限。朴素下界"输入 + 输出"是 7 份 224 MB，反而高于复用后的 3.4 份，说明组合展开 + liveness 复用已经比手工分块更省；专用反向内核的收益只在计算（省一次 scores/probs 重算），代价是新增仅 CUDA/HIP 可用的算子与两套测试，因此不写。补 `TestAttentionBackwardBudget` 钉住"复用至少省掉三分之二"；1 个新用例，5.117）→ 审计 AF-29 的训练状态凭证与异步快照（前三条"设备端参数 / 槽位 / 捕获缓冲"已由 5.84～5.88 落地，本轮补审计最后那句"验证"：`Trainer.training_state()` 分开报 `steps` 与 `optimizer_step`（`grad_hook` 返回 `None` 的步只前进不更新），`host_in_sync` 说明 `store` 里是不是旧值；`Trainer.verify_device_state()` 只比对象身份与数组模块、不做 D2H，检查"优化器 params → 驻留参数 / 每张捕获计划里的缓冲 → 驻留参数 / 槽位与参数同一数组模块"三条链，任何一处分叉都抛 `RuntimeError` 并列出全部问题，为此 `CudaGraphPlan` 与 `HipGraphPlan` 新增 `buffers` 只读属性；`Trainer.snapshot()` 在 step 边界把参数与槽位拷成主机副本，`save_async` 交给模块级单线程池并返回 `Future`——**异步的是写盘、不是拷贝**，拷贝必须在调用方选定的时刻完成，否则会拿到跨若干步拼出来的权重；`Trainer.save_checkpoint()` 让设备态不必先 `sync_to_host()`（那一版会覆盖 `store`），快照另记 `trainer_steps`。`device_state` 默认仍为 `False`：设备态需要 `capture` / `to_array` / `to_host`，参考后端没有，默认开会把构图期说明变成运行期异常。负向对照三组：关掉优化器引用检查、关掉捕获缓冲检查、去掉快照拷贝，对应用例分别变红；12 个新用例，5.118）→ 审计 AF-34 的第一块，SSM 语言模型图内闭环（`models/lm.py`：`embedding` → N 层 Mamba（新增 `build_mamba_layers` 从已有 Value 叠层）→ `rmsnorm` → 词表头（`tie_weights` 时 `embed^T`）→ `masked_lm_loss`（只除以有效 token 数，全 padding 的一批是 0 而非 NaN）→ 流式 decode（`with_steps(1)` + `carry_state`）→ 主机侧 `sample_next_token`（greedy / temperature / top-k）→ `generate`（EOS 或 `max_new_tokens` 停止，只支持 batch=1）。两条流式恒等式（分块续接 == 整段、逐 token == prefill）与损失/采样/停止各有可执行判据；负向对照两组（去掉 mask 加权、EOS 先入列）分别变红。16 个新用例全部在参考后端上跑，5.119）→ 审计 AF-34 的第二块，词表双射 / chat 模板 / batch mask / 配置绑定（`models/lm_text.py` 的 `ByteTokenizer`：256 字节 + `<pad>=0` / `<eos>=1`，无 BPE（ByteLevel BPE 谱系见 5.314），`decode(encode(s)) == s` 对任意 UTF-8 成立，越界 id 与截断 UTF-8 都拒绝；`ChatTemplate` 纯文本拼接、**不新增特殊 token**（新增会改词表让旧嵌入表整体错位），`add_generation_prompt` 补 assistant 前缀即 prefill 输入；`train/lm.py` 的 `lm_batch` 做 **left padding**（右侧 pad 会继续推进 Mamba 状态、末状态对不上），`targets` 与 logits 同形因而省掉一次切片；`LMSpec.meta()` / `check_meta()` 把形状与词表约定写进存档凭证、缺项即"未绑定"拒绝，布尔先于整数判断以免 `False` 被当成 `0`。负向对照两组：改成右对齐、eos 不停止解码，对应用例分别变红；25 个新用例，5.120）→ 审计 AF-30 的第三块，f16 融合注意力（前向内核改成**混合精度模板**：`q/k/v/mask/out` 按 dtype 声明指针（f16 下是 `unsigned short*`），读取走 `Load` 的 `af_ld_h`、写回走 `Store` 的 `af_st_h`，而共享内存、行最大值、归一化因子与输出累加器仍是 float——f16 只省存储，softmax 不在半精度里做。为了让"指向存储类型的指针"能被描述，`Let.type_name` 支持 `%T%` 占位符（`const %T%* mrow`），打印器按 dtype 展开；`Ptr` 接管 qs / vbase / orow 三处 base+offset 指针，`Index → Load`、`Assign → Store` 让存取跟着 dtype 走。生产侧 `_op_attention` 的输出 dtype 改为跟随 `q`、内核缓存键加上 dtype、渲染时传 `q.dtype`，并在运行期再校验一次 mask 与 q 同 dtype（IR 已有契约，内核按同一宽度读两者，混用会静默读错位宽）；`_F16_OPS` 的默认集从"`_OPS` 减 attention"变成 `_OPS`，49 个算子全部支持 f16。本机没有支持 `enable_gqa` 的 torch 时 `test_frontend` 的那条用例会直接 TypeError——它依赖 torch ≥ 2.5，与本次改动无关。负向对照两组：输出缓冲退回 f32（f16 内核按半字写，结果成垃圾）、mask 指针退回 `const float*`（HIP 编译期报类型不匹配），各自变红后还原；3 个新用例（1 个 emit 级 + 2 个 HIP 真机对拍），5.121）→ 审计 AF-34 的第三块，工具调用的受约束输出与执行回路（`models/lm_tools.py`：`JsonGrammar` 是**字节级** JSON 文法——`ByteTokenizer` 让"允许哪些 token"与"允许哪些字节"成为同一个集合，因此不需要前缀树、不需要 BPE 合并表，也不需要在每步把 token 展开成字节；状态由"标量阶段 + 容器栈 + 逐项 phase"三部分组成，对象区分 `key` / `key_more`（逗号之后不允许直接 `}`）以免放过尾逗号，数组区分 `first` / `value` 同理；数字的结束是隐式的，`complete` 必须按阶段判定（`12` 完整、`12.` 与 `1e` 不完整），否则写完整数不会收尾。`generate(constraint=...)` 每步用 `allowed()` 把其余 logits 置 `-inf`，`sample_next_token` 据此把"拒绝 NaN / +inf"与"允许 -inf（合法的屏蔽）"分开，并显式拒绝"整行被屏蔽"；文法没走完时抛错而不是返回半截 JSON——半截比报错危险得多。执行回路 `Tool` / `ToolRegistry` / `parse_tool_call` 只认 `{"name": ..., "arguments": {...}}` 这一个形状，必填 / 未知 / 类型不符全部拒绝，`bool` 不算 `int`、`float` 接受整数。`BYTE_OFFSET` 由 `lm_text.py` 公开导出，字节到 id 的映射不再各复制一份。负向对照两组：去掉 logits 屏蔽（约束只是摆设，`feed` 立刻拒绝越界 token）、`_after_value` 不区分 key 与值（多键对象的第二个 key 之后期望逗号而非冒号，合法 JSON 被拒），各自变红后还原；29 个新用例（正向闭包 21 条合法 JSON 逐字节 / 反向非法状态 / 随机游走 12 组必须 `json.loads` / 约束下的生成与未完成报错 / 执行回路），5.122）→ 阶段 C 的第一块，把散装的训练辅助串进 `Trainer`（`schedule=` / `accumulate_steps=` / `scaler=` 三个参数，默认值让老代码逐位不变；顺序固定为 unscale（每微批）→ accumulate → grad_hook → apply_lr → optimizer.step：反缩放必须在累积之前、否则窗口里混着不同 scale 的梯度，累积必须在钩子之前、否则裁剪上限对不上有效 batch，调度必须在真正更新之前、否则 warmup 会被微批次数提前用完。`accumulate_steps==1` 不建累加器，因为 `GradientAccumulator` 固定 f32 缓冲、会让 f64 图的梯度被悄悄降精度。scaler 与累积叠加时新增 `GradScaler.update` 接受已反缩放梯度（再调 `step` 会二次相除），溢出时 `GradientAccumulator.discard()` 整窗作废而不是只丢出错那一批（累加和里减不掉单批贡献）。缩放状态三标量以 `amp_*` 进快照 meta、`load_amp_state` 恢复，缺键或 `scale` 非有限正数都拒绝。设备态的种子改写缓冲内容即可推进（捕获写死的是地址不是值），分段路径的 cotangent 读同一个值。负向对照两组：注释掉 `apply_lr`、去掉 `discard()` 各自变红后还原——第一版整窗用例先喂坏批，窗口本来是空的，丢不丢都一样，改成先喂好批才有判别力；13 个新用例，5.123）→ 审计遗留的 f16 反向量化（`Autocast._align` 只对齐浮点操作数：此前把 i32 的 `ids` 也 cast 成 f32，带 embedding 的真实语言模型在构图期就 `TypeError`，5.90 的 20 个用例全是 matmul / softmax 因而没走到；`low_precision` / `force_f32` 的 `or` 改成 `is None`，显式空集不再回落默认集。同一份参数与数据对拍四组口径：混合精度（matmul 降 f16，scan / 归一化 / 归约留 f32）2 层 / 16 步最差梯度相对误差 2.3e-3、余弦 ≥0.999997、无 NaN，扩到 4 层 / 64 步为 4.0e-3，**不随深度与长度放大**，24 步训练轨迹的 loss 偏差 2.9e-4、参数偏差 4.9e-3；白名单扩到全部算子时 `l0_A_raw` 的梯度相对误差 **2.63e-1**、余弦 0.9696，它的梯度范数 6.5e-6 比 `embed` 小六个数量级——A 只通过状态衰减影响 loss，梯度是多时间步**相消**后的残差，各项在 f16 里的量化噪声在相消后占主导，`GradScaler` 同比放大量化噪声、救不回已丢的低位，把 `scan_linear` 单独留 f32 也无改善（26.3%）；纯 f16 的 scan 图（x / dt / A / B / C / D 全 f16）在 T = 4～64 上 A 的梯度误差 1.10e-3～1.67e-3、不随长度放大，scale 从 2^0 到 2^12 误差不变、2^14 时 `D_skip`（梯度范数 24）溢出成 7/16 个非有限值——scaler 的窗口由梯度的动态范围决定，必须保留溢出检测与回退。结论：`mixed_precision="f16"` 可用于训练，把逐元素链也降 f16 不可用（scan 本身是安全的）；3 个新用例，5.124）→ 阶段 C 的 RNG key 派生（`Trainer` 原来把 `steps` 直接写进 `rng_inputs`：单机可复现，但同一 step 的所有 rank 拿到同一个 key、同一张 dropout 掩码——数据分片不同、噪声相同，多卡的梯度噪声完全相关。新增 `_derive_rng_key(seed, step, rank)`，用 SplitMix64 的 finalizer 把三元组混成 i32（取 31 位正数；不用 `step + rank*K`，加法会让相邻 rank 落在等差线上、低位相关性传进掩码），`Trainer` 新增 `rng_seed=0` / `rng_rank=0` / `rng_world_size=1` 并校验 `0 <= rank < world_size`、`0 <= seed < 2^31`；`_rng_key()` 同时供主机的 `_rng_feed` 与设备态的 `_advance_rng`，两条路径的 key 一定一致，key 仍是图输入、同一 `(seed, step, rank)` 逐位可复现、捕获成 CUDA Graph 之后照常推进。负向对照把派生退化成 `return int(step)`，3 条用例变红后还原；4 个新用例，5.125）→ 阶段 C 的 f32 master weights（`Autocast` 新增 `low_precision_inputs`，把指定图输入直接声明成 f16；`Trainer(master_weights=True)` 要求 `mixed_precision="f16"` 且参数全 f32，图输入因此是 f16 驻留缓冲、`store` 仍是 f32 主副本，每次真的更新后把主副本覆盖写回驻留缓冲，设备态下上传 f16 / f32 两组缓冲、每步末尾做一次 D2D cast，`sync_to_host` 与快照都取 f32 主副本（取 f16 会把精度永久降一级）；`_update` 改为返回"参数是否真的动了"，`grad_hook` 返回 `None` / scaler 溢出 / 窗口未满都不写回；`refresh_resident()` 用于 `load_checkpoint` 之后重建副本，未开启时调用报错。结构代价实测（2 层 LM、26 个参数）：cast 30 → 45（down 18 → 9、up 12 → 36），全参数 f16 驻留并没有减少 cast，省的是参数缓冲与读取带宽；设备态参数总占用从 1 份 f32 变成 1.5 份。200 步线性回归与全 f32 参考的权重相对误差 6.07e-5 / 3.27e-4，与"参数 f32、图里 cast"的 5.91 轨迹同量级。负向对照三组（去掉主机态写回、`low_precision_inputs=()`、去掉设备态写回）各自变红后还原；13 个新用例，5.126）→ sm_52 的 f16 原子加（Tesla M40 真机上 26 个 f16 用例在 NVRTC 编译期全部失败：Maxwell 没有 `atomicCAS(unsigned short*)`；`af_cas_h` 在 `__CUDA_ARCH__ < 600` 时把 CAS 展宽到对齐的 32 位字并按半字掩码比较，比较值必须由调用方传入的 `assumed` 构造——第一版用 `*base` 现读整个字，成功时返回的旧值可能不等于 `assumed`，外层重试会二次累加，`test_embedding_backward` 因此得到 32.1 / −513 / 256.5；修正后 CUDA 后端 250 个用例全过（1 跳过）、训练用例 7 个全过、真机全量 1453 个用例跳过 64，5.127）→ 阶段 C 收官：分块 selective scan（两趟——块内复合 `(prod, sum)` 与块前缀/读出，前向与反向各两个内核；末状态专用内核只写 `(tail...)`，`out` 不物化；块内递推逐字不变、块间复合改变舍入，因此是显式开关 `chunked_scan=`，默认路径逐位不变。M40 上 `T=16384, B=16, D=64` 前向 5.48 → 1.76 ms（3.11×，47.9M → 148.9M tokens/s）、反向 3.56×，f32 前向相对误差 1.16e-7、VJP 1.2～1.9e-7、8 步训练轨迹 loss 逐位一致；末状态显存 16 MB → 4 KB、decode 单步 35.56 → 29.76 µs，12 个新用例，5.128）→ HIP 上的自动图实测（`graph_mode="auto"` 这一层 5.79 只在 CUDA 上跑过，本轮在 WSL ROCm 上补齐：同图同输入只捕一个计划、换值不重捕、单节点图不进捕获、`max_graph_plans=1` 是硬上限；`execute` 口径 `transformer_block` 5.510 → 2.935 ms（1.88×）、`cnn_lenet` 3.920 → 1.847 ms（2.12×），4 个新用例，5.129）→ 批量派发（计划期把"算子是否注册 / 输入 dtype 是否受支持"判定一次并绑定内核函数，运行期成批执行不再逐节点校验，未通过判定的组原地回退、报错时机与文案不变；`transformer_block` −6.3%、`cnn_lenet` −5.8%，单算子图无收益；第 21 行的"执行器架构改造"至此清空，8 个新用例，5.130）→ 执行期分层内存（`hip/memory.py` 的 `TieredAllocator`：`<=64KiB` 按 512B 对齐分档自由表、`<=2MiB` 走 Arena best-fit、`>2MiB` 直调 `hipMalloc`；`runtime.set_allocator` 是进程级钩子，`HipBackend._memory_scope` 严格成对安装/还原并把统计写进 `_last_arena`，`memory_stats()` 因此不用新增 API；`DeviceBackend` 新增 `memory_mode="container"｜"tiered"` 参数与 `_memory_scope` 钩子，CUDA 侧显式拒绝并说明未接；小对象块用 `_PooledBlock` 持地址、不能复用 `_Buffer`（后者的 `__del__` 直接 hipFree），6 个新用例，5.131）→ 事件驱动的 SNN 稀疏算子（`sparse_linear`：`weights[K,N]` × CSR 事件 `(indices, values, offsets)` 逐段加权求和成 `[S,N]`，稠密的 `S×K` 脉冲矩阵从不物化；事件数 `E` 是符号维，`offsets` 长度必须静态（它决定输出步数），越界在设备端记 `err` 后由 `_check_ids` 报错、捕获期也能用；反向只对 `weights` 求导，`sparse_linear_weight_grad` 用原子加把 `values[e]·dout[s]` 散射回行，`indices` / `offsets` / `values` 返回 `None`——事件幅度不训练。CUDA 与 HIP 共用同一份 L2 描述，conformance 新增两条用例，9 个新用例，5.132）→ 事件驱动的 SNN 构件（`build_snn_events`：第 0 层仍是稠密 `matmul`，之后每层用 `spike_offsets` + `spike_csr` + `sparse_linear` 按事件累加，与 `build_snn_network` 的输入/参数/输出完全一致；最初把列号与起点表打包进一个 i32 张量再 `slice`，参考后端逐位正确但 HIP 拒绝 `slice` 的 int32，于是拆成两个单输出算子——不扩大后端能力面；三组深度与稠密网络逐位相等，只支持前向，训练仍走稠密路径；HIP 上量化是负结果：小规模下事件路径 0.72×～0.96×，开权重常驻后仍只有 0.96×～1.00×，瓶颈是每层多出的两趟压缩与稀疏乘的访存地板；放大到 2048×1024×1024、常驻权重时 2% 稀疏才 1.54×（收益门槛约 10%），5 个新用例，5.133）→ CSR 压缩从单线程扫描改成块内并行（`spike_offsets` 原先一个 block 一个线程扫完 `T×K`，改为单 block 两阶段：每线程跨步数若干行的非零数、结果暂存在 `offsets[t+1]`，`__syncthreads` 后由 0 号线程做 `T` 步前缀和；输出缓冲当 scratch、仍是单输出单内核，不需要新 L2 能力，也解释了为什么不铺多 block——块间没有 `__syncthreads`。HIP 两趟合计 8192×512 从 205.6 → 10.6 ms（19.4×）、8192×64 26.2 → 5.1 ms，但 32×64 只快 6%，所以 5.133 的端到端负结论不变；conformance 新增 `steps=300` 的跨步用例，参考后端 79 executed / HIP 74 executed / 5 refused / 0 failed，5.134）→ 本轮补齐的九个能力面（隐式 GEMM 卷积的 A/B 否掉了自动选择器，5.151；f16 attention 的用例与文档，5.152；任意轴归约与归一化，5.153；池化的 ceil_mode / count_include_pad / maxpool2d_indices 直通 IR，5.154；scan 按 T 与尾维自适应分块，5.155；HIP 的 pinned 暂存与设备态训练，顺带修掉广播原地加的两套折叠混用，5.156；MIOpen 对照入口，legacy 前向在 ROCm 7.2 已无 invoker 因而改走 solution API，5.157；模糊测试扩到硬件后端，随机图逐例与参考语义对拍、按契约拒绝的不算失败，5.158；捕获窗口内的 `hipFree` / `hipHostFree` 推迟到 EndCapture 之后，修掉偶发 901，5.159；归约类的 block 开放给调优面，5 轮同号门槛把 ±5% 的方向翻转拦在默认之外，5.160；逐节点 GPU 计时把"某个 kernel 变慢"从整图 wall 里拆出来，CUDA/HIP 事件对只在收尾同步一次，5.161；常量折叠改为只物化存活边界，三张图分别 9→3 / 7→1 / 5→1，同时量化出"折叠产物大且原计算便宜时会倒退"这一边界，5.161；IR 跨版本迁移在序列化结构上完成、按单出边路径补齐缺字段并记录来源版本，5.162；编译期与调优面从 CUDA / HIP 两个实现提炼成 `CodegenContract` / `TuningContract`，5.163；多设备枚举与切换（HIP 侧补上 `hipGetDeviceCount` / `hipGetDevice` / `hipSetDevice`，属性与 SM 数按设备缓存；CUDA 侧去掉写死的 0 号卡；两边提炼 `DeviceManagerContract`），5.164；分组 / depthwise 卷积的直算核（一个线程一个输出元素，不物化列矩阵、不按组分次 launch，dilation 与 bias 一并支持，f32 的 auto 路径切换过去、im2col 保留为强制对照；HIP 容器的整块赋值同时放开"连续源 → 带步长目标"，分组路径的按组写回才真正可用；dilation 由内核自己的窗口循环处理，没有复用池化的无膨胀窗口），5.165；分组卷积的性能对照（`benchmarks/grouped_conv_compare.py`，HIP 上 9 个形状直算核快 2.51×～54.91×、列矩阵 1.72～27.56 MB 全部消失），5.166；遥测包装器按存在性转发全部六项可选契约（此前漏掉 5.163 的 `kernel` 与调优五方法、5.164 的多设备三方法，包一层后 `implemented_contracts` 会静默少三项），5.167；`kernel()` 的公开面收口（CUDA 的 cupy 句柄只有 `__call__`、HIP 的只有 `launch`，真机全量第一次跑就撞上；公开入口统一返回带 `launch` 的包装句柄，同一 key 仍是同一对象），5.168；多卡真机验证（3×Tesla M40：切卡时清空 NVRTC 模块、cuBLAS handle、驻留参数与捕获计划——它们分别绑在旧 context / 旧设备 / 旧卡显存上；5 个多卡用例在 3 卡全可见时通过），5.169；CUDA 侧的执行期分层内存（复用捕获期的 `Arena` 作 cupy 分配器，`Executor.run` 显式归还设备侧输入副本与输出，`memory_stats()` 与捕获期同口径，4 个用例），5.170；CUDA 上的分组卷积对照推翻 5.166 的跨厂商结论（depthwise 4.30×～21.50×，非 depthwise 在 cuBLAS 上出现 0.44× / 0.80× 反例；`auto` 收窄为只对 depthwise 走直算核，`conv_impl="grouped"` 变成显式选项，HIP 复测 2.07×～45.77×），5.171；常量节点的设备副本随执行计划缓存（计划期上传一次、运行期直接复用，`release` 不回收、切卡与捕获轮重建；mask 用例 3.384 → 0.339 ms，修掉捕获期复用预热副本导致设备态训练 loss 从 4.37 变成 10^2 量级的缺陷），5.172；`sparse_linear` 与 cuSPARSE SpMM 的内核级对照（把 H2D 从计时里拿掉后，比 cuSPARSE 快 1.73×～2.38×、比物化稠密乘快 2.18×～6.56×），5.173；非末维归约的两维铺法（`blockDim.x` 沿 inner 合并访存、`blockDim.y` 切归约轴，`inner=1` 时 8.6×～9.6×、小输出平面 3.3×，分派判据 `_use_tiled_reduce`），5.174；运行期语义错误进模糊测试（8 条固定用例覆盖 embedding 正/负越界下标、事件类越界 `indices` 与溢出 `offsets`、整数除零必须报错、浮点除零必须给 IEEE inf；顺带修掉两处静默——负下标回绕、整数除零截断成 INT_MIN），5.175）→ 语义级索引层（`lower/indexing.py` 的 `Layout` / `offset` / `unravel_steps`，把三处手写线性化收成纯函数、生产源码逐 token 不变；只覆盖线性化不覆盖布局决策，5.177）→ 语言模型与训练状态的边界（跨块 reset 清掉带出窗口、左填充隔离、训练状态存档绑定未提交梯度 / 超参 / scaler / RNG、入口 dtype 规则、f32 损失与初始化、softplus 零点导数与注入上界、seal 后常量载荷冻结、按 token 加权累积、受约束解码根形态与生成边界、HIP 探测区分"库可加载"与"设备可枚举"，5.178～5.186）→ LM 推理路径（固定块预填充：中间块 `head="none"` 只带状态、末块 `head="last"` 只投影末位置，尾块按实际长度构图避免 pad 污染；权重常驻与 `to_host` 取行让设备上每步只剩 8 B H2D / 4 KB D2H、状态跨步 0 B；`benchmarks/lm_inference.py` 在 M40 上 1K/8K/32K 实测 TTFT 289/446/1091 ms、边际预填充 3.5K/18.4K/30.0K tok/s，5.187）→ 语言模型的整数标签损失（`lm_batch(labels="index")` 返回 `(B,T)` int32 的扁平下标、`masked_lm_loss_indexed` 把 logits 拉直后借 `embedding` 的查表语义取目标位，不新增算子；标签内存与 H2D 少 V 倍，两条路径前向逐位相等，5.188）→ 工具的 JSON Schema 约束与按注册表合成的受约束文法（`Tool` 接受标准 JSON Schema 子集并输出 `{"name","description","parameters"}`，`ToolCallGrammar` 把工具名、参数键名、参数类型与必填顺序都在解码期限死，嵌套对象参数在构造期显式拒绝并要求改用 `JsonGrammar`；20 个新用例，5.189）→ 多卡最小闭环（`train/parallel.py`：`copy_across_devices` 经主机中转、默认不做 peer-to-peer；主机侧 `allreduce` 只接受同形同 dtype 的浮点；`DataParallelTrainer` 按每卡 batch 切分、逐卡前向反向、主机侧平均梯度后更新、参数整份写回；22 个新用例），5.190）→ 语言模型的跨请求推理会话（`models/lm.py` 的 `LMRunner`：图按 `(块长, head)` 缓存并用 `OrderedDict` 封顶、`resident=True` 权重只上传一次、`close()` 只释放自己登记的那一份；解码循环与 `generate` 共用同一份 `_decode`，因此首次请求逐位一致；M40 上 1K prompt / chunk=256 / 8 token 的重复请求 35.93 ms，无状态 `generate` 387.34 ms，10.78×；7 个新用例），5.191）→ 语言模型的批量解码（`generate_batch` / `LMRunner.generate_batch`：不同长度的 prompt 按左侧补 `pad_id` 对齐、每行首个真实 token 处置 `reset=1`，行与行互不影响；每行独立采样（第 `i` 行用 `seed + i`）与独立停止，已结束的行喂 pad 并丢弃输出；批量会话的图缓存与单条共用同一套 `(块长, head)` 键；`constraint` 在批量里显式拒绝；8 个新用例），5.192 → 语言模型的会话前缀缓存（`LMRunner.prime` 预先消费前缀并把 `(state, logits)` 留在会话里，后续 `generate` 只把新增 token 送进图；新 prompt 必须是已缓存前缀的真正前缀，否则显式报错并要求 `reset()`；生成结束后前缀自动失效，`reset()` 只清前缀、保留图缓存；只支持 batch=1，批量路径不做前缀续接；8 个新用例），5.193 → 语言模型的采样面补齐（`sample_next_token` 增加 top-p 核采样与 CTRL 式重复惩罚，处理顺序固定为"重复惩罚 → 贪心判定 → top-k → 温度 → 归一化 → top-p"；`generate` / `generate_batch` / `LMRunner` 三个入口按行维护 `prompt + 已生成 token` 的历史；新增 `generate_beam` / `LMRunner.generate_beam`：k 条假设放进图的 batch 维、按父假设重排状态、EOS 进完成池不占活动槽位、`length_penalty` 归一化分数，`num_beams=1` 与贪心逐位一致，宽度 >= 词表且两步时与穷举搜索分数相等；18 个新用例），5.194 → 后端生态的算子承接面补齐（`backends/device_backend.py`：任意单轴 `argmax` 与非末维归约一样先置换到末维再复用同一个内核，`keepdims` 按原轴序恢复；多轴 `reduce_sum` / `reduce_max` / `reduce_mean` 先置换到 `[keep..., reduce...]` 再合并成一个轴，空归约轴按 numpy 语义给单位元；conformance 新增 4 条用例，参考 91 executed / CUDA 89 / HIP 89；6 个新用例），5.195 → 推理服务层的请求队列与连续批处理（`ServingEngine`：提交即入队、`step()` 以**解码步**为调度单位——先给有 logits 的槽位采样、再把空槽按 FIFO 填入等待请求并只对本次准入的做一次批量预填充、最后对需要喂 token 的槽位跑一次 batch=capacity 的 decode；槽位一空即准入，慢请求不阻塞新请求的首 token；容量 1/2/4 与逐个 `generate` 逐位一致，约束、重复惩罚、上采样面、停止都按请求独立；7 个新用例），5.196 → 中层优化的独立 pass 层（`passes/registry.py`：pass 注册表要求名字唯一且等于 `pass.name`、工厂每次给新实例；`run_pass` / `run_passes` / `build_pipeline` 三个按名入口都走 `PassManager`，单独跑也过 verifier；带参数的 pass 按实例进管线；默认与融合两条管线的名字序列收敛到 `DEFAULT_PASS_NAMES` / `FUSED_PASS_NAMES` 一处；13 个新用例），5.197 → 二阶导数（`ir/second_order.py` 的 `hessian_vector_product`：把"对反向图再求导"的名字映射固定下来，二阶图输入 = 前向输入 + 一阶种子 `d<输出别名>` + 二阶种子 `dd<wrt 名>`，输出 = 每个 `wrt` 的 `d<wrt 名>`；判据是解析二阶导 `6x·v`、一阶梯度的中心差分、跨输入块的完整 Hessian 乘向量、以及用 HVP 拼 Hessian 解 `Hp=g` 在二次目标上一步收敛；零曲率与缺少二阶规则的算子（`conv2d_*_grad`）都显式报错而不是给零；9 个新用例），5.198 → 趋势的自动采集调度（`.github/workflows/ci.yml` 加每日 cron、GPU 作业接受 `schedule` 事件；趋势历史用 `actions/cache/restore` + `cache/save` 跨 run 接续（cache 不可变，key 带 run_id、restore-keys 取最近一次），每轮追加后 `--prune 200` 按**图名**分别裁剪——全局取最近 N 条会让冷门用例的曲线直接断掉；`benchmarks.trend.prune` 的边界是 `keep` 必须是 ≥1 的整数、`--prune` 必须与 `--save` 同用且参数校验在跑基准之前；4 个新用例），5.199 → 系统化边界值测试与长上下文数值契约（`tests/test_boundaries.py` 五组 20 条：形状 / dtype / 数值 / 序列化 / 1024 步长上下文；顺带修掉两处真实缺陷——exp / log / sqrt 与全部二元逐元素算子在定义域外会往宿主打 RuntimeWarning，现统一走 `np.errstate(all="ignore")`（结果不变、日志干净）；`axis=None` 从 `TypeError: 'NoneType' object is not iterable` 改成显式 `ValueError` 并给出 `axis=tuple(range(rank))`。判据包括 f16 溢出成 inf、softmax 大值仍归一、NaN 传播、bf16 借 f32 承载、`dumps` 不修坏图，以及"分块续接的误差不随块数增长"（4/16/32 块实测 0 / 1.4e-7 / 9.3e-8）），5.200 → 跨图控制流（`ir/control.py`：`Conditional` 按数据选支、`WhileLoop` 按数据迭代，子图仍是普通 `Graph`，因此照样过 verifier 与序列化；未选中的分支不执行、循环状态必须封闭、`max_steps` 显式给出、控制区域可互相嵌套；24 个新用例），5.201 → 代码生成的第三条路（`lower/translate.py` 的跨方言源码转译：只改登记过的标识符与 `#include` 头文件，注释 / 字符串 / 未登记标识符原样透传，同方言调用是恒等；PTX 内联汇编、带掩码的 warp 内建、半精度包装类型、Clang 位转换内建一律拒收并带行号——`backends/hip/codegen.py` 的 f16 辅助函数不能靠转译得到，用例直接拿两份真实 prelude 做判据；`compile_translated` 走 `CodegenContract` 的同一个编译缓存，键含原文 sha256 与两个方言名，目标方言与后端标签不符时当场拒绝；23 个新用例），5.202 → cuSPARSE 接成 `sparse_linear` 的可选实现（`backends/cuda/cusparse.py`：ctypes 绑定 CSR×稠密 SpMM，workspace 按 `(步数, 行, 列, nnz, dtype)` 缓存、描述符每次重建；`sparse_impl` 取 `custom` / `auto` / `cusparse`，前两者在基类、第三条由 CUDA 扩展、HIP 构造期拒绝；默认仍是带越界检查的自研核，`auto` 在库缺失 / 捕获期 / f16·f64 / i64 索引时回落，显式 `cusparse` 把这些情况变成明确错误；真机因驱动掉线未验证，覆盖只放 f32 + i32、默认不切换；26 个新用例），5.203 → 事件稀疏线性的幅度梯度（`sparse_linear` 的 VJP 此前让 `values` 返回 `None`、事件幅度永远不训练；新增 `sparse_linear_values_grad`：`dvalues[e] = dout[s(e)] · W[indices[e]]`，由 `offsets` 决定事件所属步，每个事件只写一次、**不需要原子加**——与权重梯度"散射回行、重复下标要累加"正好相反；内核按**步**并行（一个线程包干该步的全部事件，逐事件做长度 `cols` 的点积），省掉"事件号 → 步号"的设备端二分；输出长度就是事件数 `E`，可以是符号维，不把稠密 `(S, K)` 脉冲矩阵造回来；`indices` / `offsets` 仍无梯度，`dout` 与 `weights` 必须同 dtype（f16 用 f32 累加）；判据是解析式、f64 中心差分、同一步里两个事件命中同一行时的互不干扰、f16 与符号维；CUDA / HIP 共用同一份 L2 描述，conformance 新增一条用例，11 个新用例），5.204 → 训练侧的两块并行能力（`ShardedOptimizer` 把 Adam / SGD 的槽位按参数名分片、每片一个独立实例，分片更新与整份优化器**逐位一致**——分片只改存储布局、不改每个参数内部的运算序列，空片合法且步数同步；`PipelineTrainer` 把整图按用户给的切点拆段、每段独占一块卡，段间激活经主机中转、micro-batch 顺序流过全部段再逆序反向，参数梯度在主机侧按 1/份数 加权累加；判据是两段 / 三段流水 N 步的参数与单卡整图在 `rtol=1e-5` 内一致，35 个新用例），5.205，5.206 → 批量解码的逐行约束（`constraint` 在 `generate_batch` / `LMRunner.generate_batch` 里也按行接受：长度等于 `spec.batch` 的序列，每行一份文法实例或 `None`，`batch=1` 时可传单个实例；共享一份、长度不符、元素缺 `allowed`/`feed`/`done` 三面显式拒绝，未完成的文法按行号报错，4 个新用例），5.207 → 反向图专用优化（`backward(minimize=True)` 沿消费者方向限定 vjp 的传播范围，无关分支不再产生反向节点且梯度逐位不变；`cse-commutative` 把 `add` / `mul` 的签名按输入 id 排序后比较，认得出 cotangent 顺序不同的同源累加；`backward_pipeline()` 是第三条管线，不含任何融合 pass——反向图要能被任何承接前向的后端直接执行；12 个新用例），5.208 → 训练态 BatchNorm 的 torch.export 导入（`aten.batch_norm(training=True)` 直接映射为 IR 的 `batch_norm`：训练态输出只依赖 batch 统计量 / weight / bias / eps，与 IR 同义；`momentum` 与 running stats 不进图，`run_decompositions()` 后的 `_native_batch_norm_legit_functional` 走同一条翻译、只留第 0 槽；入口按 `graph_signature.user_outputs` 把 buffer 更新与 save 中间量连同其生产者闭包排除在构图之外，int64 的 `num_batches_tracked` 不会漏成用户输入；3 个新用例），5.209 → 索引/布局抽象铺到生产内核（`lower/indexing.py` 的 `Layout` 从 3 处扩到 22 处：五个模块级布局常量收掉 19 处二维行主序手写线性化，`cast=None` 关掉首因子提升，57 个内核工厂的 CUDA 源码 SHA-256 与改动前**逐字节相同**；带括号前缀乘积、64 位提升与跨步不连续的形态仍手写；4 个新用例），5.210 → 复数 dtype 与 FFT（IR 新增 `c64` / `c128` 两个 dtype 与 `fft`（含 `inverse`）/ `rfft` 两个算子：变换只沿最后一维，`norm` 三档与 numpy 同义，`rfft` 只收 f32/f64 并给出 c64/c128，f16/bf16 显式拒绝；参考语义由 `numpy.fft` 给出，反向不提供 VJP（复数伴随需要共轭转置，IR 里还没有 `conj`）；`backends/cuda/cufft.py` 用 `cufftPlanMany` 做离线绑定——dtype/方向登记表、`plan_key` 与归一化缩放都是纯函数，cuFFT 两个方向都不归一化，逆变换必须显式补 1/N（或 1/√N）；CUDA 的 `capabilities()` 只在 cuFFT 可加载时声明这两个算子，HIP 没有 rocFFT、容器改为能承载复数以便 `invoke` 给出"未实现算子"而不是裸运行时错误；23 个新用例），5.211 → 流水调度与 ZeRO 通信（`PipelineTrainer(schedule=...)` 新增 `sequential` / `gpipe` / `1f1b` 三种调度：前两者是既有顺序与经典批式，1F1B 预热"段数-1"份后交替"反向一份、前向下一份"，同时驻留的段间激活从份数降到 `min(段数-1, 份数)`，用 `last_resident_peak` 记账，三者参数在 `rtol=1e-6` 内一致；`train/sharded.py` 新增 `reduce_scatter_grads`：把整份 All-Reduce 换成 Reduce-Scatter，每 rank 只接收本片（接收总量从 W·N 降到 N），累加顺序与 `allreduce_grads` 相同因而**逐位一致**，`transport` 只看到本片的名字；给空片不调用 transport，世界大小大于参数个数仍是合法状态；13 个新用例），5.212 → 审计缺陷修复（门控 SiLU 的 gate 梯度补乘上游、AMP 先反缩放再判有限性且溢出步不推进调度、累积窗口权重随存档恢复、训练状态与分片状态的"先校验后提交"、调度器 lr 与静态超参的分工、签名硬件包的信任记录落盘、归一化标量参数的梯度归约、整数入口按可表示范围校验、fuzz 整数除数非零；20 个新用例）→ 完整续训的两块凭证（`DataLoader.state_dict` / `load_state_dict` 记录轮次编号、批游标与配置，中途停止后从同一批继续且下一轮顺序与未中断逐位一致，配置不同即拒绝；`Trainer` 的前向图指纹与调度器类型 / 超参进 meta 并在提交前校验，四个内置调度器各自实现 `hyperparams()`、`Warmup` 递归到内层，14 个新用例），5.214 → 训练态 BN 的 running stats 维护（`train/bn.py`：扫描 `batch_norm` 节点、按依赖闭包重建只到 BN 输入的统计子图、每步执行一次并按 EMA 更新；`Trainer(bn_momentum=0.1)` 默认开启，方差按 `n/(n-1)` 无偏修正因而与 torch 的 eval 语义一致，`bn_running_stats()` 直接接 `bn_infer`；存档写 `bn/mean|var/<站点>` 与 `bn_momentum` 凭证并在提交前校验，`None` 是显式关闭而非静默降级；设备态与含符号通道的 BN 显式拒绝；9 个新用例），5.215 → 训练开关的组合矩阵（`tests/test_train_combinations.py`：五开关全开时第一次更新等于手算的 `initial - lr_at(0)·clip(窗口加权梯度)`；窗口一半存档续训把权重、scale、调度步数一起接回去；分段重算与整图逐位一致；f16 驻留 + f32 主副本在量化误差内；设备驻留与主机态逐位一致，并钉住"设备态 store 在 `sync_to_host()` 之前是旧值"这条既有契约；5 个新用例），5.216 → 动态形状契约（`ir/shape_contract.py`：构图 / 执行 / 自动微分 / 训练循环 / 分段重算 / 部署 / torch 导入七个环节各自声明 `full`、`per-op`、`data-only` 或 `rejected`，`support_for` 对未知环节直接报错；`require_static_shapes` 是 rejected 环节的统一守卫，训练循环与分段重算在构造期报错并带环节名、具体输入与建议；契约声明的支持面各有用例：同一张符号图被执行器按多个 batch 跑、部署包只要求权重静态而数据输入可绑定、`reshape` 的 `(B,4) → (4*B,)` 在构图期拒绝、torch 动态维（SymInt）在导入期显式拒绝而不是 `int()` 具体化；13 个新用例），5.217 → MoE 构件（`models/moe.py`：`topk_mask` 选出前 k 个专家、概率乘掩码后按行重新归一化成门控权重，所有 token 流经全部专家再按门控加权组合；负载均衡损失 `E·Σ f_e·P_e` 可微并接进训练闭环；专家并行在 `require_expert_parallel` 处显式拒绝；13 个新用例），5.218 → 分块注意力反向（`ir/grad_rules.py` 的 `_attention` 在序列维超过 128 时按 query 行切块重算：`dq` 与逐行 mask 梯度沿行维 `concat` 回原形状、`dk` / `dv` 跨块累加，`kᵀ` / `vᵀ` 只建一次；dropout 不切块——掩码按块重建会换掉"同 seed 同结果"的采样流；符号序列维退回整块重算。S=T=512 / D=64 / BH=32 的静态峰值 108.1 → 59.0 MB，`tracemalloc` 实测 136.1 → 100.1 MB（剩下的地板是输入、输出与累加器）；7 个新用例），5.219 → 服务控制面（5.220：`ServingEngine` 的 `max_pending` 排队上限与 `QueueFullError`、幂等 `cancel(rid)`（在跑的请求清空槽位即参与下次准入）、`submit(..., timeout=秒)` 墙钟超时（从提交起算、只在 `step` 边界清扫、到点带残片进 `take_timed_out()` 而不进完成列表）、可注入 `clock` 保证测试确定性；7 个新用例），5.220 → 集合通信能力声明（5.221：`collective_support` 如实声明主机中转归约与 `overlap=False`，`collective="device"` 无实现时抛 `DeviceCollectiveUnavailable` 而**不静默降级**，注入 provider 的路径只把归约结果拉回主机一次并对返回值的名字/形状/dtype 逐一校验；6 个新用例），5.221 → 跨进程 ZeRO 分片（5.222：`slot_bytes_by_shard` 量化每片常驻、`save_shard` / `load_shard` 只碰本片且用归属表把"装错片"挡在加载期；4 个新用例），5.222 → torch 兼容矩阵（5.223：`TORCH_MATRIX` 声明版本区间 / dtype / `verified`，`_require_torch` 校验版本、矩阵外版本与 `int64` 显式拒绝、新增 `bfloat16` 映射；6 个新用例），5.223 → torch 入口补全（5.224：int64 / int32 入口 dtype、`table[ids]` 落成 embedding、提升常量与 `get_attr` 常量转 `constant`、padding_idx 与多索引显式拒绝、常量体积上限；8 个新用例），5.224 → 续训接管数据迭代位置（5.225：存档写入 `data_*` 与 `data_kind`、恢复前只读校验 `validate_state_dict`、`run_loader` 按 loader 游标跑、位置与 loader 参数同进同出；5 个新用例），5.225 → 审查修订（5.226：`loader_state` / `loader_kind` 专属键、游标在所有提交之后推进、`run_loader` 委托 `run`、常量 i32 值域检查、`_to_dtype` 接入矩阵行、`padding_idx` 图内值显式拒绝；2 个新用例），5.226 → CUDA 真机复跑（5.227：cuSPARSE 的 32I 枚举改为 2、SpMM 实参按 11 个排、f64 拒绝提示、能力面改成子集关系、CSR 铺法行数按 SM 现算；0 个新用例），5.227 → NCCL 集合通信（5.228：dtype / 归约登记表、unique id 的 128 字节往返、communicator 入口与主机缓冲校验、AllReduce / AllGather / ReduceScatter / Broadcast 在 2×V100 上真跑；16 个新用例），5.228 → 跨进程 ZeRO-1（5.229：只持本片槽位、整份 AllReduce 后本地取片、参数按 owner 广播、两步与主机 ShardedOptimizer 逐位对拍、只存本片的存档与三道校验；20 个新用例），5.229 → cuDNN 快路径的真机边界（5.230：`AttributeError` 归入"这条快路径不可用"、显式模式报可读错、`_cudnn_broken` 记类型名、四条不碰 GPU 的判定用例；cuDNN 9.28 在驱动 550 上执行期 5003 并回落成功，cuDNN 9.1 因镜像限流未完成真机对拍；4 个新用例），5.230 → ZeRO-2（5.231：分段按参数边界切、每段补零到等长 `pad`、ReduceScatter 后梯度只留本片、AllGather 拼回完整参数、`LoopbackWorld` 与 `NcclHostCollective` 各补两个集合面、存档与 ZeRO-1 共用同一份写盘逻辑；30 个新用例），5.231 → 多 rank 启动器（5.232：`AF_*` 环境变量优先、torchrun 的 `RANK` / `WORLD_SIZE` 兜底、128 字节 unique id 走 TCP 分发、用 `python -c` 起进程以免在 `-m unittest` 下重跑 `__main__`、子进程非 0 退出带输出抛出；17 个新用例），5.232 → ZeRO-3 参数驻留分片（5.233：每个参数按元素均分成 W 块、rank 只常驻本块，`materialize` / `release` 拼回即释放，`step` 只 ReduceScatter 出本块梯度并只更本块，存档同时带参数分片与槽位、装错片 / 分片数 / 块长不符都显式拒绝；23 个新用例），5.233 → ZeRO-3 接进训练循环（5.234：`CheckpointPlan` 的段级借用回调，`Trainer(zero3=..., segments=N)` 每段只拼回这一段要读的参数、执行完删引用并 `release`，更新走 `zero3.step`（ReduceScatter + 只更本块），`store` 降级为借出的完整副本、`sync_to_host()` 才拼回、`snapshot()` 先自动同步，BN 统计子图按需借用；与整份 2 段训练逐位对拍、续训 2+2 与 4 步对拍；13 个新用例），5.234 → 分布式存档的合并与重分片（5.235：`merge_shards` 把 N 个分片拼回完整参数与槽位、`ShardedCheckpoint.split/write_shards` 按新 `world_size` 重切、`split_checkpoint` 把一份全量 checkpoint 切成每 rank 一份；names 布局必须给模型权重、elements 布局合并时逐位校验权重一致；输出与 `save_shard` 同构、`save()` 可被 `load_checkpoint` 直接读；21 个新用例），5.236 → 主机中转的多进程数据并行（5.236：`HostRelay` / `HostCollective` 在 TCP 上做跨进程求和、`launch` 一条命令拉起 W 个进程并注入 `AF_RELAY_PORT`、`DistributedDataParallel` 把 `Trainer` 的一步摊到多进程——梯度在 `grad_hook` 里先归约再交给用户钩子，损失跨 rank 取均值，切分校验与 `DataParallelTrainer` 同源；拒绝设备态 / GradScaler / ZeRO-3 与随机流不匹配；19 个新用例），5.236 → 静态执行快路径（5.237：`_fast_step` 把静态单节点组的 attrs / 期望形状 / dtype / 是否量化 / 释放清单在第一次执行时收成一条记录，运行期只做"取输入 → 调内核 → 写回"；形状与 dtype 仍在运行期复核，语义不变，含符号维的组、融合组与常量节点仍走通用路径；本地空内核口径 24 节点骨架 69 → 40 µs，HIP 真机上被驱动提交与 H2D 噪声淹没，剩余的主机侧开销只能靠图重放，8 个新用例），5.237 → 中等通道卷积在 HIP 上改走隐式 GEMM（5.238：`DeviceBackend.CONV_AUTO_IMPLICIT` 默认 False、HIP 覆盖成 True，auto 在"K > 分块阈值且核面积 > 1"时选隐式；`conv_implicit_fits` 用与 `plan_conv_implicit` 同一份 shared 预算（权重 + 窗口）判可行性，1x1 与小 K 不改道，CUDA 保持 5.151 的 im2col；WSL HIP 实测中等通道 3x3/5x5 快 1.32～1.61×、3x3_c128 快 1.48×，数值与 im2col 在 1e-5 内一致，13 个新用例），5.238 → 张量并行（5.240：`train/tensor_parallel.py` 的列并行 / 行并行线性层、attention 头切分与 `TensorParallelBlock`；列切分不需要通信、行切分返回**部分和**必须求和，块被拆成 `attn` / `ff` 两张图、段间由注入的 `all_reduce` 归约，图外只剩两次残差加法；`num_heads` / `d_ff` 必须被 `world_size` 整除、`dropout` 必须为 0（各 rank 头宽不同，同一 seed 采不到一致掩码），不满足即拒绝；判据是 world=1/2/4 的 block 与 `build_transformer_block` 在 1e-5 内一致，行并行带 bias 在构图期直接拒绝（会被乘成 W 倍）；只做前向，反向通信与优化器分片仍属 ZeRO 的范畴；25 个新用例），5.240 → 分页 KV cache 与投机解码（5.241：`runtime/paged_cache.py` 的块表 / 物理页池 / 全有或全无追加 / 按页回滚 / `pad_to` 补零，`models/cached_attention.py` 把"已有 T 个 token 的 K/V、只算当前一个 token"写成单查询增量图（`concat` 拼缓存 + 加性掩码屏蔽补零位与未来位），`models/speculative.py` 的贪心验收 + **状态快照回滚**；判据是增量输出与 `build_transformer` 逐位置一致、任意草稿下输出与 `generate(temperature=0)` 逐位一致；只做贪心与 batch=1，Mamba 主干本身没有增长 KV，分页缓存因此以独立组件交付、未接进 `generate`；29 个新用例），5.241 → 成本模型与组合搜索（5.242：`tuning/cost.py` 的 `MachineModel` / `conv_tile_cost` / `elementwise_cost` / `rank_configs` / `cost_ranked_tune`，把 conv2d 的 80 个 tile 组合先按六项 roofline 折算（`arith` / `read` / `staging` / `sync` / `tail` / `dead`）排序，派生量全部取自 `lower/plan.py` 以免两处推导漂移，计划层拒绝的组合直接变成带原因的不可行项；`search.py` 抽出 `measure_configs`，让「按估算排序后只测前 N 个」复用同一套排名与平局语义；默认配置永远排第一、不可行项殿后、同分保持空间原顺序；判据是 80 个组合里 5 个不可行的剔除、`cb` / `roc` 的单调方向、`tile_w` 被截断后逐项相等、裁剪后仍能测出更快的非默认配置；**不是性能预测器**，也没接进 `benchmarks/tune.py` 的交替 A/B 路径；41 个新用例），5.242 → Chrome Trace 导出（5.243：`NodeTiming.start_ms` 记下相对 `execute` 起点的派发时刻，`runtime/trace.py` 的 `trace_events` / `build_trace` / `write_trace` 把它变成 chrome://tracing 与 Perfetto 能读的时间轴，一份记录一个 `pid`、多条记录按 `wall_ms` 错开；缺时刻的旧记录按顺序累加补齐并在 `otherData.synthesized_nodes` 里计数，"补的"与"测的"必须分得开；GPU 轨道首尾相接、线程名写明"累计占位"、事件带 `approximate`，因为事件对只给时长不给绝对时刻；CLI 是 `python -m aetherflow.runtime.trace telemetry.json [-o trace.json]`；判据是实测时刻进 `ts`、混合记录只补缺行、GPU 起点取首个读数、参数拒绝面、落盘不留临时文件、包装器时间轴单调；**不是采样 profiler**（无调用栈、无符号化），也没接进基准默认路径；20 个新用例），5.243 → NCCL / RCCL 同一份绑定（5.244：`train/nccl.py` 不写第二份 ctypes 声明，RCCL 的 `nccl*` ABI 与 NCCL 相同，候选表同时列 `libnccl.so.2` 与 `librccl.so.1`；厂商由库文件名判定（`_vendor_from_path`，不看 cupy 能否导入），`vendor()` / `version()` / `version_string()` 三面公开、库缺失即抛 `NcclUnavailable`；`synchronize()` 按厂商分叉——HIP 走 `backends.hip.runtime.synchronize()`、CUDA 走 cupy 当前流，不做双路 try/except；错误信息带真实库名；HIP 的 `DeviceArray` 已提供 `data.ptr` / `size` / `dtype` / `flags`，四个集合操作直接可用；能力声明仍是 `device_allreduce=False`（数据并行路径未接线），理由区分"加载不到"与"库在未接线"；WSL HIP 上 RCCL 2.27.7 单卡 world=1 真跑通过，跨卡数值仍只有 CUDA 的结论；11 个新用例），5.244 → 设备端循环（5.245：条件图与循环体各捕获一次、状态常驻显存、每轮只剩一次谓词 D2H；本机实测不比宿主路径快，交付的是执行形态；19 个新用例，其中 1 条真机），5.245 → 布局传播（5.246：conv2d / maxpool2d / avgpool2d 增加 layout 属性（nhwc 参考语义与 nchw 逐位相同，权重不随激活翻转）；反向规则显式拒绝 NHWC；后端用 supported_layouts 声明承接面，设备端在 invoke 与计划快路径两处拒绝 NHWC；LayoutPropagation 把 conv / pool + 一元逐元素连成的区域整段翻成 NHWC、只在区域边界插转置；23 个新用例，其中 1 条真机），5.246 |
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
| 一套工作流换后端 | ✅ | 同一份 conformance 在 CUDA 与 HIP 上都跑过且 0 failed（111 个用例；参考后端本轮 111 executed / 0 refused / 0 failed；HIP 上一轮 90 executed / 2 refused / 0 failed，本轮未复跑，用例集扩到 111 后不能按差值推算；拒绝项是既有的 f64 与分组 depthwise 形态，加上没有 rocFFT 的 `fft` / `rfft`）；装配层让迁移连 import 与类名都不用改：`get_backend()` 自动挑，或 `AETHERFLOW_BACKEND=hip` 指定（5.40、5.41）。HIP 的运行期面也已与 CUDA 对齐：pinned 暂存、跨流事件依赖与 `Trainer(device_state=True)` 的 SGD / Adam 都在真机跑通（5.156） |
| 主流模型架构 | 🟡 | 六类构件都有可训练表达（5.22）：MLP ✅；Transformer ✅（单块符号化 + 多层 decoder-only LM）；CNN 🟡（conv（含分组 / depthwise）/ 转置卷积（5.51）/ max-avg-自适应-全局池化 / BN / dropout / 残差块 / 分类网络；cuDNN 对照已做（5.69 首测自研分块核慢 4～12 倍；5.253 按"稳态 + 驻留输入"重读后小通道已同量级甚至更快（1x1_c32 为 0.72×），真实差距只剩宽通道 3x3/5x5；5.260 / 5.268 的 auto 进一步到 c8～c24 的 0.45～0.75×、c32/c64 的 1.02×/1.36×，c64 是唯一还没消掉的 1.36×）；RNN ✅（RNN / LSTM / GRU 时间展开）；SNN 🟡（多层 LIF + 替代梯度；稀疏事件表示与事件驱动的构件已交付（5.35、5.132、5.133），层间压缩自 5.136 起可微（`spike_linear`：脉冲梯度走普通 `matmul`，只有权重梯度走事件散射），5.204 又把 `sparse_linear` 的事件幅度梯度补成 `sparse_linear_values_grad`；剩余边界是 `dspikes` 仍是稠密 GEMM，且 HIP 上事件路径尚未达到加速门槛）；SSM 🟡（完整 Mamba block 第一版，递归收敛为 `scan_linear`；语言模型侧的跨块 reset 与左填充隔离见 5.178 / 5.179，固定块预填充 / 末位置投影 / 权重常驻的推理路径见 5.187，整数标签损失（one-hot 的 `B*T*V` → `B*T` 下标）见 5.188，跨请求复用的推理会话（图缓存 + 权重驻留）见 5.191，批量解码（左填充对齐 + 逐行独立停止）见 5.192，会话前缀缓存（`prime` 预先消费 system prompt、后续只跑增量，batch=1）见 5.193，采样面（top-p / 重复惩罚）与束搜索 `generate_beam`（k 条假设进 batch 维）见 5.194，请求级并发（请求队列 + 连续批处理，以解码步为调度单位、容量固定）见 5.196） |
| 主流硬件 | 🟡 | NVIDIA（sm_75，RTX 2080 Ti）与 AMD（gfx1100，RX 7650 GRE / ROCm 7.2）各一台，同一套 conformance 全绿（5.40）；两家还各有一次库级对照——cuDNN（5.69 首测；5.260 / 5.268 的 auto 已到 c8～c24 的 0.45～0.75×、c32/c64 的 1.02×/1.36×）与 MIOpen（5.157 测的是自研分块核，慢 4.52×～37.03×；HIP 自 5.238 改走隐式 GEMM 后未对 MIOpen 复测，这个倍数不能当成 HIP 现状）；Intel / Apple / 移动端未接，同厂商多型号未铺开 |

**二、架构**

| 目标 | 现状 | 说明 |
|---|---|---|
| 静态图导入 | ✅ | torch.export → IR，覆盖 Transformer 与 CNN 的 aten 子集（代码里出现 116 个 `aten.*` 名字，含拒绝面的提示名，不等于全部承接）；未知算子明确报错 |
| 动态图导入 | 🟡 | 不做 tape 式 define-by-run 执行：图必须静态，才能做构图期后端能力预检、让反向图自包含可序列化、并满足 CUDA Graph 的固定路径要求。补的是**构图手感**——`EagerGraph` / `Tensor` 用运算符直接写图，产物仍是同一份静态 IR（5.25）；数据相关的分支与循环由 `Conditional` / `WhileLoop` 显式表达，控制流是子图之间的关系、子图仍是普通静态图（5.201）；逐算子即时求值仍未做 |
| 形状推导（含符号维） | ✅ | 编译期推导 + 运行期回校；空间维可以是 `floor((S + 2p - k) / s) + 1` 形式的派生表达式，一次构图多分辨率执行（5.53） |
| 自动微分（含 SNN 非标准反传） | 🟡 | 语义 IR 层反向展开：49 条 VJP 规则，反向图仍是可序列化、可执行的普通 PORTABLE 图（5.17）；eager 前端可直接 `loss.backward()`，但它仍是这张图而不是 tape（5.77）；激活检查点由 `train/recompute.py` 的分段重算补上（5.75）；二阶导数由 `hessian_vector_product` 补上（对反向图再求导，判据是解析值 / 中心差分 / 二次目标一步 Newton，5.198），但只对具备 VJP 的算子成立；反向图专用优化自 5.207 起有两条：`backward(minimize=True)` 按求导目标限定 vjp 范围（无关分支不再建反向节点，梯度逐位不变），`backward_pipeline()` 用交换律 CSE + DCE 且不含融合；SNN 替代梯度只需替换算子 `vjp`，路已铺好 |
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
| CUDA 一等插件 | ✅ | 68 个算子语义对齐（性能见 5.4；卷积见 5.12，池化见 5.13；反向图见 5.17；SNN 见 5.18；scan 见 5.19 / 5.35；新增五算子、两个反向算子、`pad` 与三个卷积/池化反向内核见 5.20；随机算子见 5.23；批归一化见 5.24；融合注意力见 5.32；f16 支持面见 5.33；分组卷积与自适应池化见 5.34；线性递归前向与伴随见 5.35；LayerNorm 残差与 SwiGLU 门控两个内核见 5.36；`erf` 见 5.39；任意轴归约 / softmax / 归一化见 5.153（非末维归约的两维铺法见 5.174）；池化三属性与 maxpool 下标见 5.154；scan 自适应分块见 5.155；隐式卷积仅作显式对照路径见 5.151；`spike_csr` 的块内扫描见 5.176；任意单轴 argmax 与多轴归约见 5.195；事件稀疏线性的两侧梯度见 5.132 / 5.204） |
| 新硬件只写插件 | ✅ | 契约分层 + 算子注册表 + 插件开发指南（5.30）齐备；HIP 后端是第一个真实实例——全部新增代码在 `backends/hip/` 一个包里，IR / verifier / 前端 / 优化 / 契约零改动，内核模板逐字复用（5.40）。第三方硬件包还能用 pip entry point（组 `aetherflow.backends`）注册，装包即被 `get_backend` 发现（5.41）；zip 包还能声明内容摘要与依赖，装包期就拒绝被改动或依赖不满足的包（5.64） |

**三、核心模块**

| 目标 | 现状 | 说明 |
|---|---|---|
| 插件内 codegen | ✅ | NVRTC 运行时编译（CUDA 侧）与 hipRTC 运行时编译（HIP 侧），目标机都不需要厂商 toolkit；公开入口 `DeviceBackend.kernel()` 已提炼为可选契约 `CodegenContract`（5.163） |
| 内存分块 / 指令映射 | 🟡 | conv2d 有真正的分块（权重复用 + 寄存器复用 + 边界快速路径，见 5.16）；其余算子仍只有广播维折叠与 block size 选择。分块与布局的**决策**已在 5.54 收进 `lower/plan.py`（block / grid / vec / shared / OC tile 全部是纯函数并带约束校验），一维网格步进骨架已在 5.55 统一（`kernels.grid_stride`），多维索引在 5.57 逐字比对后确认只值得合并一处，其余按语义差异保持手写（5.177 起 `_split_output_index` / `_image_slice` / `_tap` 三处线性化收进 `lower/indexing.py` 的 `Layout`，生产源码逐 token 不变；布局决策仍留在各内核）；block 的取值在 5.65 变成"按规模推导"的纯函数（只有上限与下限两条规则，不含性能结论） |
| 复用 / 转译 / 原生生成三模式 | ✅ | 三条都通：**原生生成**是 63 个算子的生产路径（L1 → L2 → 方言 → NVRTC / hipRTC），**复用**是 cuBLAS / cuDNN / cuSPARSE / hipBLAS / MIOpen 的库调用（cuSPARSE 自 5.203 起接管 `sparse_linear` 的可选路径），**转译**是 `lower/translate.py` 把别人写好的源码换成另一家方言的源码（5.202）。量化算子另算一类：`quantize_i8` / `dequantize_i8` 是两份直接交给 NVRTC / hipRTC 的源码（不经 L2），`linear_i8` 是组合实现（5.239）。转译只做逐 token 可证的改写：注释 / 字符串 / 未登记标识符原样、命中拒收项（PTX 内联汇编、带掩码的 warp 内建、半精度包装类型、Clang 位转换内建）当场报错并带行号；`compile_translated` 与原生生成共用同一个编译缓存，键含原文 sha256 |
| 标准算子 / 自定义算子规范 | ✅ | `Op` 数据结构 + attrs schema + `register_operator` 命名空间护栏（5.30）；插件算子进 `OPS` 后立刻能被参考后端执行与 conformance 对拍 |
| 语义在 IR 层、内核在各插件 | ✅ | 参考后端承接全部 80 个算子的语义（含 W8A8 的 `linear_i8a8`，5.308 / 5.309；非对称零点的 `*_zp` 族，5.315；逐组的 `dequantize_i8_group` / `dequantize_i8_group_zp`，5.317；QAT 的 `fake_quantize_i8`，5.311；LSQ 的 `fake_quantize_i8_lsq`，5.312；`fft` / `rfft` 走 `numpy.fft`）；CUDA 承接 71 个——68 个自研内核（含 W8A8 的 `linear_i8a8`，5.309）加 1 个由反量化与 GEMM 组合的 `linear_i8`，再加 1 个由量化与反量化组合的 `fake_quantize_i8`（5.311），再加 1 个前向同源、反向为 LSQ 的 `fake_quantize_i8_lsq`（5.312），另有 cuFFT 的 `fft` / `rfft`（conv2d = 分块直算 / im2col + cuBLAS 两条生产路径，隐式 GEMM 只作显式对照、不进自动选择，5.151；分组 / depthwise 自 5.165 起有专用直算核，5.171 按真机数据把 `auto` 收窄为"只有 depthwise 走直算核"——CUDA 上非 depthwise 的 `g4` / `g2` 被 im2col + cuBLAS 反超（0.80× / 0.44×），HIP 上仍是正收益（2.07×），不能跨厂商外推；`conv_impl="grouped"` 可显式强制，5.34 的"按组切片复用 im2col"保留在 `conv_impl="im2col"` 下；maxpool2d = 自研索引 kernel，adaptive_avg_pool2d = 自研窗口 kernel，embedding 与池化/卷积反向的重叠写用原子加，attention = 不物化分数矩阵的分块 online softmax，scan_linear / scan_linear_backward = 一个尾元素一个线程、沿时间串行，长 T（>=512）且尾维 <=2048 时自动分块，见 5.35 / 5.155；归约与 softmax 的任意轴见 5.153；池化的 ceil_mode / count_include_pad / maxpool 下标见 5.154；融合形态也在这里：layernorm 的第 4 个 residual 输入、silu 的第 2 个 gate 输入，见 5.36）。dtype 上 f32 全覆盖，f16 也全覆盖 64 个（attention 自 5.121 起是混合精度：q/k/v/mask/out 存半精度，softmax 的行统计与输出累加仍在 float——f16 只省存储与带宽，不参与累加，5.152 补了跨块边界的用例）。HIP 承接同一批 69 个算子的方式不是复制：内核源码逐字复用上面这些模板，只换 f16 的存取与原子加（5.40） |
| 复用 cuDNN / cuBLAS / cuSPARSE | ✅ | cuBLAS 是 matmul 的生产路径；cuDNN 在 5.74 接成 `conv_impl="auto"` 的可回落快路径（`groups=1` 的 f32/f16，缺库或形状不支持就静默回到自研核），显式 `conv_impl="cudnn"` 可强制并在缺库时报能力错误。本轮真机没有 libcudnn，cuDNN 数值对拍按设计跳过，5.69 的对照数字仍然有效。cuSPARSE 自 5.203 起是 `sparse_linear` 的可选实现（`sparse_impl="custom"｜"auto"｜"cusparse"`，默认 custom；覆盖 f32 值 + i32 索引），本轮因真机驱动掉线没有数值对拍，库路径没有设为默认 |
| 自研底层内核 | ✅ | 逐元素（含 gelu、SwiGLU 门控）、归约（含任意单轴；非末维的两维铺法见 5.174）与 softmax（含任意单轴）、layernorm（含残差形态）/ rmsnorm、batch_norm、im2col、conv2d_tiled、分组 / depthwise 直算、隐式卷积（仅显式对照）、max/avgpool（含 ceil_mode 与 padded-边界分母）、maxpool 下标、adaptive_avg_pool2d、concat / pad copy、embedding gather / scatter、maxpool 与卷积的输入/权重梯度、SplitMix64 掩码、分块 attention（online softmax）、线性递归扫描（前向 + 伴随 + 自适应分块）、脉冲压缩（`spike_offsets` 前缀和 + `spike_csr` 块内扫描，见 5.176）、任意单轴 argmax 与多轴归约（5.195）、事件稀疏线性的两侧梯度（5.132 / 5.204）、int8 对称量化与反量化（一份源码同时喂 NVRTC 与 hipRTC，5.239） |
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
| 跨后端数值一致性基准 | ✅ | `run_conformance`：111 用例 + 5 条检查（5.9，用例集随 5.17 / 5.18 / 5.19 / 5.29 / 5.34 / 5.39 / 5.50 / 5.51 / 5.132～5.134 / 5.153～5.155 / 5.174 / 5.195 / 5.204 / 5.286 扩增）；本轮本地实测参考后端 111 executed / 0 refused / 0 failed；HIP 上一轮 90 / 2 / 0，本轮未复跑（拒绝项不变：f64 与分组 depthwise 的既有形态） |
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
3. **模型架构只覆盖一半。** Transformer 完整；CNN 的 cuDNN 对照已补（5.69 首测 4～12 倍，5.253～5.268 把 auto 收窄到 0.45～1.36×）；SNN / SSM 各有第一版。

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
（5.16）都已跑通，cuDNN 对照也已在 5.69 补上（首测 4～12 倍，5.268 的 auto 已到 1.02×/1.36×）。第 3 项已落地第一版：10 个固定用例 + 两份同机基线 +
25% 判定 + CI 配置（5.15、M3.5）；GPU job 尚未接上真实 self-hosted runner（作业脚本已在真机逐条执行验证，5.70；性能门禁因换宿主变红，5.92 重采基线后已转绿）。
跨运行吞吐 / 显存趋势已可作为 CLI 显式采集（5.38）。
第 4 项已完成第一版：`backward()` 在语义 IR 层展开，49 条 VJP 规则，反向图可序列化、
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

### 1.4 与主流框架的定位对照

这一节只写**结构差异与能力边界**，不写任何一方的性能数字：性能对照分两层，
库对库已有真机数据（5.69 / 5.157 / 5.173 / 5.253～5.268），框架对框架
**尚未实测**（仓库里没有 torch / JAX 的基准脚本）。对其它框架的描述只到
"执行模型与工程形态"这一层，不涉及版本细节，也不替它们下性能结论。

| 维度 | AetherFlow | 其它框架的典型做法 | 可比性 |
|---|---|---|---|
| 执行模型 | 静态语义 IR：构图 → 校验 → 优化 → 后端预检 → 执行；`EagerGraph` 只改构图手感，产物仍是静态图 | PyTorch define-by-run + Dynamo 捕获；JAX 函数式 trace + `jit`；ONNX Runtime 解释 / 编译 ONNX 图；TVM 编译 Relay / Relax 图 | 都是"图或程序 → 后端"，但我们在构图期就把能力不足报掉 |
| 微分 | IR 层展开成普通图（49 条 VJP），反向图可序列化、可执行、可再优化；二阶导只对具备 VJP 的算子成立 | PyTorch autograd 引擎按算子注册反向；JAX `vjp` / `jvp` 可任意阶；ONNX Runtime 以推理为主；TVM 的自动微分生态较小 | 不提供 tape，也不承诺"任意算子可导" |
| 动态形状 | 前端拒绝 SymInt；执行器支持一次构图多形状（5.217）；训练循环与分段重算要求静态 | PyTorch / JAX 原生支持（JAX 按形状重编译） | 我们是"多形状绑定"，不是"动态图" |
| 控制流 | `Conditional` / `WhileLoop` 是子图关系，判定回主机；设备端循环每次读一次谓词（5.245） | PyTorch 用 TorchScript / Dynamo 表达分支；JAX 的 `lax.cond` / `lax.while_loop` 编译进设备端 | 我们的控制流不追求设备端无同步 |
| 后端接入 | 两个必需方法 + 六个可选契约；新硬件写一个包即可（HIP 为证），IR / 前端 / 优化零改动 | PyTorch 需 PrivateUse1 / dispatch key；JAX 需 PJRT plugin；ONNX Runtime 需 Execution Provider；TVM 需 BYOC | 接入面更窄，代价是生态工具要自己写 |
| 算子覆盖 | 78 个 IR 算子；前端代码里出现 116 个 `aten.*` 名字（含拒绝面） | PyTorch 的 aten 是全集量级；JAX 以 Primitive 为基；ONNX Runtime 按 ONNX op 集 | 我们是**子集**，缺的算子显式拒绝，不静默降级 |
| 精度 | f32 / f16 全覆盖（65 个设备算子），bf16 32 个，f64 与复数部分承接 | 各家覆盖更宽，且已有 FP8 / FP4 等低精度格式 | 量化有 weight-only int8 与 W8A8（int8 输入、int32 累加的设备内核，5.309），对称与非对称零点两套主机侧语义（非对称见 5.315、量化 / 反量化内核见 5.322、融合线性层见 5.323；逐组的非对称 `dequantize_i8_group_zp` 见 5.321），逐组（per-group）weight-only 反量化（`dequantize_i8_group` / `_zp`，5.317 语义、5.321 设备内核），以及 QAT 的假量化算子（5.311，scale 不可学）与 LSQ 的可学步长版本（5.312），fp8 的 cast 已落到设备端（5.325，载体仍是 f32、不省带宽），但没有 fp8 GEMM |
| 分布式 | DP（主机中转 / 设备侧 NCCL）、ZeRO-1/2/3、GPipe / 1F1B、张量并行（仅前向） | PyTorch 有 FSDP / DTensor / 专家并行；JAX 有 `pjit` / sharding | 缺专家并行、P2P 与 FSDP 等价物 |
| 推理服务 | OpenAI 兼容 HTTP、连续批处理、设备态解码、分页 KV cache（`PagedGenerator` 的 token→token 闭环与自动前缀缓存，5.310 / 5.335；未接 `LMRunner`）、贪心投机解码 | PyTorch 生态靠 vLLM / TGI 等；ONNX Runtime 有 Serving | 自带服务面；权重能读 safetensors（5.313）、文本能过 ByteLevel BPE（5.314），但没有内置 HF 权重转换表、无 GGUF、无 WordPiece / Unigram |
| 生态 | 能读 safetensors（5.313，按需读取 + 名字 / 布局显式映射）与 HF `tokenizer.json` 的 ByteLevel BPE 谱系（5.314），无预训练权重分发、无 Hub、无第三方算子库 | 各自有完整生态 | 权重与分词入口已通，分发仍然缺位 |
| 失败语义 | 能力未覆盖即显式报错（构图期拒绝或 `BackendCapabilityError`），捕获期登记、replay 后复查 | 常见做法是回落或静默降级 | 这是设计差异，不是能力差异 |

**这张表怎么用**：定位是"可插拔后端 + 可验证跨厂商一致性的训练 / 推理基础设施"，
不是"替代 PyTorch 的训练框架"。对外能说的只有三件事：同一张图换后端不改代码；
跨厂商数值一致性可判定；支持不了的地方明确失败。

**框架级实测（第一组，2026-11-21）**：A800-SXM4-40GB，torch 2.5.1+cu124，
`python -m benchmarks.framework_compare --device cuda`；输入常驻、输出不回传，
两边**交替**测，每轮连续 10 次平摊、取 5 轮最好：

| 用例 | AetherFlow | torch eager | eager / af |
|---|---|---|---|
| matmul_4096 | 20.196 ms | 20.050 ms | 0.99× |
| matmul_2048 | 3.069 ms | 3.004 ms | 0.98× |
| matmul_1024 | 0.594 ms | 0.602 ms | 1.01× |
| conv2d_3x3_pad1 | 0.260 ms | 0.251 ms | 0.97× |
| maxpool2x2 | 0.485 ms | 0.489 ms | 1.01× |

结论只到这一层：**这五个算子的框架开销与 torch eager 无显著差异**
（0.97～1.01×）。它们都落到同一批库（cuBLAS / 自家卷积核 / 自家池化核），
框架本身的派发差异在这一档形状上被内核时间吃掉。`torch.compile` 在 2048³
GEMM 上没有额外收益（3.87 ms 对 eager 3.88 ms，编译 2.8 s），因为最终都调
cuBLAS。方法上有一条必须记下来：先连测 A 再连测 B 时，同一形状两轮差过 40%
（2.25 ms 对 3.88 ms）——这台是共享机，顺序测量会把负载漂移当成框架差异；
交替测量后比值才稳定在 0.97～1.01×。

**训练一步（第二组，2026-11-21）**：batch=128、d=1024、f32、lr=1e-6，
前向（两层 MLP + ReLU）→ 交叉熵 → 反向 → SGD，反向图复用；同机、交替测、
每轮连续 10 次平摊、取 5 轮最好：

| 配置 | AetherFlow | torch eager | torch / af |
|---|---|---|---|
| 逐节点派发（`graph_mode="off"`） | 5.696 ms | 0.739 ms | 0.13× |
| 图重放（`graph_mode="auto"`） | 3.033 ms | 0.738 ms | 0.24× |

**历史结果，尚不能归因**：以上为共享卡、原始反向图下的旧测量，不能代表
当前默认路径或独占成绩。单算子接近不能排除训练图内核、调度或内存访问的
问题，也不能把总差额直接算成 Python 成本。执行路径确认有逐组派发、输入
复制和校验，具体各占多少需分段测量（见 5.287 / 5.288）。

再分开测了一次（`graph_captures` / `graph_replays` 计数 + `--no-update`）：
重放**确实命中**（捕获 1 次、命中 52 次，无失败原因），旧样本关掉参数更新
相差 **0.30 ms**（3.046 → 2.746 ms）。共享负载和路径变化可能影响差值，不能
把它当更新成本或融合收益上限；其余时间也不能在没有分段数据时直接归因。

**稳定性复测（连续独立进程，每个进程自己捕获）**：`auto` 跑 10 次，3 次
0.58 ms（0.98×）、7 次 3.05 ms（0.24×）；`off` 跑 5 次稳定在 1.32～1.50 ms
（0.41～0.47×）。把参数更新改成原地（不再每步分配 8 个临时数组）之后，
`auto` 仍是双峰（0.55 / 3.05），`off` 约 1.35 ms。这说明原地更新没有消除
双峰；不能据此判定“捕获到慢 5 倍的图”。后续事件测试在同一张图重放期间
也观察到状态切换，而且发现另一个训练进程；时钟恒定不能排除共享负载。

图重放默认值保持关闭，但这些共享卡样本不足以给出可靠胜率或期望提速。
需要独占复测和设备时间线，再决定默认值及优化顺序。

**独占复测（2026-11-22，A800 空闲，同一 `framework_compare --training`）**：
同机无其他计算进程，每轮 40 次、9 轮取中位数：

| 配置 | AetherFlow | torch eager | torch / af |
|---|---|---|---|
| 逐节点派发 + 参数更新 | 0.896 ms | 0.552 ms | 0.62× |
| 逐节点派发、不含更新 | 0.751 ms | 0.457 ms | 0.61× |
| 图重放 + 参数更新 | 0.486 ms | 0.532 ms | 1.09× |
| 图重放、不含更新 | 0.387 ms | 0.443 ms | 1.14× |

图外 SGD 占 0.145 ms（AF）对 0.095 ms（torch）。这个形状（两层 MLP、
8.4 MB 参数）的主机态上限只有 0.62×：每步仍有 18 组 Python 派发与校验，
而 torch 走的是常驻参数 + 单进程 autograd。真实训练的参数规模下必须开
设备态，见 5.296。

预置 cuBLAS workspace 已试过：`AETHERFLOW_CUBLAS_WORKSPACE_MB=32`
（`cublasSetWorkspace_v2`）各跑 5 次，开是 2 快 3 慢、不开是 1 快 4 慢，
本轮没有消除双峰。这不支持“设置 workspace 即可解决”的假设，也不能仅凭
这些受干扰样本定位到 cuBLAS 算法。开关保留（默认 0），供后续对照。

训练显存这一轮**没有取到可信数字**：cupy 池在预热后已驻留，`used_bytes`
增量为 0，不能当峰值；torch 侧 allocated 37.8 MiB 只作量级参考。显存对照
需要换一套统计方式（池的 `total_bytes` 或逐分配打点）再补。

**框架级性能对照的缺口**（这三层补齐之前，"比 X 快 / 慢"只能引用库对库数据）：

1. 单算子与训练一步已有（上两表），但**训练显存与整图（`transformer_block`）**
   还没有：显存要同口径打点（`plan_memory` 是静态估计，不能与
   `torch.cuda.max_memory_allocated()` 并列）；整图需要两边各写一条等价模块；
2. A800 上的 torch 2.5.1 不在前端兼容矩阵内，对照只能绕过前端、直接写等价算式；
3. JAX / ONNX Runtime / TVM 未安装，也没有与它们的数值对拍；
4. CI 未在托管平台实跑、wheel 未发布，外部无法复现我们的数字。

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

**当时仍然缺的**：cuDNN 对照（已在 5.69 补上：自研分块核慢 4～12 倍，im2col + cuBLAS 慢 5.5～12 倍；这两组都是 5.69 首测，后续 5.253～5.268 已把 auto 收窄到 0.45～1.36×）。`avg_pool2d` 与自适应 / 全局池化已分别由 5.20 与 5.34
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
  （后续 5.253～5.268 已收窄到 0.45～1.36×，见 1.2；分组卷积与自适应 / 全局池化已在 5.34 补齐，转置卷积已在 5.51 补齐）。
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

> 后续修正（5.253～5.268）：这里的 4～12 倍是"每轮 H2D + 单次延迟"口径下
> 分块直算 / im2col 与 cuDNN 的比值。按"稳态 + 驻留输入"重读、再加上隐式 GEMM
> 与 Winograd 之后，`auto` 在 c8～c24 是 cuDNN 9 的 0.45～0.75×、c32/c64 是
> 1.02×/1.36×。本节数字只作历史记录，当前结论以 1.2 与 5.268 为准。

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

### 5.218　MoE：稠密参考语义下的路由、专家组合与负载均衡

**问题**：混合专家模型的真实瓶颈不在"多几层 MLP"，而在**随数据变化的专家分组**：
每个 token 由门控选 top-k 个专家，被选中与否取决于这一批的数值，因此每个专家
拿到的是**变长的 token 子集**。静态图表达不了这件事——要么引入动态形状 + 变长
gather/scatter，要么固定容量并接受截断/补齐的语义偏差。框架的两条既有约束
（静态形状、专家并行未做）都不允许前者。

**做法**（`models/moe.py` + 一个新算子）：

1. 新增 IR 算子 `topk_mask(x, k, axis)`：沿 `axis` 取最大的 `k` 个位置置 1、其余
   置 0，并列时取下标较小的（稳定排序）。`k` 必须是正整数、不得大于该轴长度，
   非法 `k` / 非法轴在构图期报错。它的 VJP 返回全 `None`——"选第几个"是离散
   结果，与输入数值之间没有可用导数。
2. 走**稠密参考语义**：所有 token 流经全部专家，再按门控加权求和。门控是
   `softmax(logits) * topk_mask` 后按行重新归一化，因此"未被选中的专家权重为 0"，
   数值上等价于 dispatch / combine，但每一步都是稠密矩阵乘，不引入动态形状。
3. 专家为两层 MLP（`w1/b1/w2/b2`，逐专家一组参数），门控用
   `slice(dim=1, start=e, stop=e+1)` 取出单列后再与专家输出相乘，避免额外算子。
4. 辅助负载均衡损失 `E·Σ_e f_e·P_e` 可微（`f_e` 是掩码的列均值，`P_e` 是概率的
   列均值），权重为 0 时整项为 0，可直接与前向损失相加。
5. 专家并行显式拒绝：`EXPERT_PARALLEL_SUPPORTED = False`，
   `require_expert_parallel(devices)` 对 `devices > 1` 抛 `NotImplementedError`
   并说明"专家并行"未实现；`MoESpec` 在构造期就调用它，把问题提前到构图之前。

**用例**（`tests/test_moe.py`，13 条）：IR 可移植且只有一个 `softmax`、含
`topk_mask`；与 numpy 手写稠密 MoE 对拍；专家同参时输出等于单专家（门控归一化
的直接判据）；top-1 时路由到 argmax；负载均衡权重为 0 时 aux 恰好为 0；中心差分
梯度（`w_router` / 两个专家的 `w1` / `w2` / `b2`）；`ParameterStore + Adam` 训练
60 步损失下降；序列化往返；专家并行与非法 spec 的拒绝面；`topk_mask` 自身的
稳定并列、`k == dim` 全 1、非法 attrs。conformance 另加两条 `topk_mask` 形态。

**负向对照**（做完还原）：把门控改成不归一化的 `selected`，`test_gates_sum_to_one`
与 `test_matches_numpy` 一起变红；删掉 `topk_mask` 的 VJP 装饰器，`backward` 在
该节点抛 `NoGradientError`。两次都说明断言钉的是语义而不是形状。

**验证**：本地 `Ran 2255 tests / OK (skipped=410)`；WSL HIP（RX 7650 GRE /
ROCm 7.2）`Ran 2255 tests / OK (skipped=369)`。

**边界**：稠密语义下计算量是"专家数倍"，收益只在**显存与实现简单性**上，不做
稀疏加速；`topk_mask` 目前只有参考实现，CUDA / HIP 没有对应内核，因此含 MoE 的
图在设备后端会被 conformance 判为"未承接"而不是静默算错。专家并行、容量因子、
token 丢弃这些需要动态形状的语义都没有实现，属于显式缺口。

### 5.219　分块注意力反向：把二次中间量从"三份半"压到"一块"

**问题**：`attention` 的反向由已有 PORTABLE 算子组合重算（5.29 / 5.117），不用
专用内核；但整块重算时分数、概率、`dp`、`ds` 四份 `(...,S,T)` 会在一段区间内
同时活着。5.117 量到 S=T=512 时峰值 108.1 MB（≈3.4 份）。长上下文训练的真实
约束是**峰值显存**，不是计算次数。

**做法**（`ir/grad_rules.py`）：

1. 反向重算按 **query 行**切块（`ATTENTION_BACKWARD_CHUNK_ROWS = 128`）。每块
   只算 `(块行数, T)` 的分数与概率，贡献完即被 liveness 回收；`dq` 与逐行 mask
   梯度沿行维 `concat` 回原形状，`dk` / `dv` 跨块累加。`kᵀ` / `vᵀ` 在块外建一次
   共用，避免同一份转置被算 N 遍。
2. 语义不变性：softmax 沿 key 维归一化，各行独立，因此任意切分在同一后端上
   逐位等价；`dk` / `dv` 的累加只是改变求和顺序，末位差在 `1e-12` 量级。
3. 两条显式回退：(a) 序列维是符号 → 退回整块（`slice` 的区间要静态整数，与卷积
   拒绝符号空间维同一条理由）；(b) `dropout > 0` → 退回整块。后者是刻意的：
   掩码由"输入形状 + seed"重建，切块后每块的掩码是另一组采样，会悄悄改掉 5.29
   约定的"同 seed 同结果"。

**用例**（`tests/test_attention.py::TestAttentionChunkedBackward`，7 条）：
块=2 时 `dq/dk/dv` 与中心差分一致；分块图与整块图逐位对齐；广播 mask（序列维
为 1）的梯度跨块求和；逐行 mask（序列维等于 S）的梯度 concat 还原；带 dropout
时反向图里没有 `concat`（钉住"不切块"）；符号序列维退回整块且图仍可校验；
S=T=512 的静态峰值至少省四分之一。

**负向对照**（做完还原）：把广播 mask 的跨块累加删成只取第一块，
`test_chunked_broadcast_mask_gradient_sums_all_blocks` 立刻变红；把每块的 `q`
都取第 0 块（形状对、数值错），四条数值用例一起变红。两次都说明断言钉的是
"每块用的是自己那一段"。

**验证**：本地 `Ran 2262 tests / OK (skipped=410)`；WSL HIP（RX 7650 GRE /
ROCm 7.2）`Ran 2262 tests / OK (skipped=369)`。同一形状下 `plan_memory` 的静态
峰值 108.1 → 59.0 MB，`tracemalloc` 实测执行峰值 136.1 → 100.1 MB；块长继续
减小（256 → 128 → 64）实测峰值不再下降而节点数线性上涨，因此默认取 128。

**边界**：分块只减少**中间量**，输入、输出与 `dk` / `dv` 累加器仍是整份，所以
实测降幅（26%）小于静态估计（45%）；这是 liveness 复用的地板，不是切块没生效。
设备端的收益取决于后端的分配器是否真的在最后一次使用后归还（HIP 的 arena 有
回收，容器模式直调 `hipMalloc` 时池化行为另计）。不切块时行为与 5.117 完全一致。

### 5.220　推理服务控制面：排队上限、取消与请求超时

**问题**：5.196 的 `ServingEngine` 只有"队列 + 连续批处理"，是一个**吞吐**调度器，
没有**控制面**：请求可以无限堆积（显存与延迟都没有上限）、提交后无法撤回、
没有时限（一条长请求能把整批拖到 SLO 之外）。真实服务必然要处理这三件事，
而且必须**显式**——静默丢弃请求、把超时当完成、把取消当错误都是把问题推给调用方。

**做法**（`models/lm.py` 的 `ServingEngine`）：

1. **排队上限** `max_pending`：只约束**等待队列**，在跑的请求不受影响。满了抛
   `QueueFullError`（继承 `RuntimeError` 而非 `ValueError`：这是容量条件，
   调用方该重试或扩容，不是改参数）。`max_new_tokens=0` 的请求不排队、不占上限。
2. **取消** `cancel(rid)`：排队中直接出队；在跑的清空槽位，腾出的槽位立即参与
   下一次准入。返回 `True` / `False` 表示"这次是否真的移除了它"，已完成、未知
   id、重复取消都是 `False`——幂等，不是异常。
3. **请求超时** `submit(..., timeout=秒)`：**墙钟**时限，从 `submit` 起算（排队
   时间也计入 SLO），在每个 `step` 的最前面统一清扫，采样与解码不会在中间状态被
   半路改写。到点的请求带**已生成 token** 进 `take_timed_out()`，**不混进完成
   列表**：完成与超时是两种结果，混在一起调用方就分不清拿到的是结果还是残片。
   时钟可注入（构造参数 `clock`，默认 `time.monotonic`），测试用假时钟即可逐位
   确定，不依赖 `sleep`。
4. 三者的非法参数都在边界显式报错：`timeout` 非有限正数 → `ValueError`、非数值
   → `TypeError`；`max_pending` 非正整数 → `ValueError` / `TypeError`；`clock`
   不可调用 → `TypeError`；关闭后的 `submit` / `step` / `cancel` → `ValueError`。

**用例**（`tests/test_lm.py::TestServingControl`，7 条）：排队取消后不出现在完成
列表且不干扰在跑请求；取消在跑请求腾出的槽位下一步被队列请求用上；未知 id /
已完成 / 非整数 id 的返回与拒绝面；`max_pending` 满了抛 `QueueFullError`、取消后
可再提交、零 token 请求不占上限；超时请求带已生成 token 进超时列表且不出现在完成
列表；排队期间的超时从提交起算且不影响没设时限的请求；非法 timeout /
max_pending / clock 的显式报错。

**负向对照**（做完还原）：删掉 `step` 开头的 `_sweep_timeouts()`，两条超时用例
立刻变红；把上限判断的 `>=` 改成 `>`，`test_queue_cap_rejects_and_recovers_after_cancel`
变红（第 `max_pending+1` 条才被拒）。两次都说明断言钉的是"到点即摘除"和
"第 N 条就拒绝"的边界。

**验证**：本地 `Ran 2269 tests / OK (skipped=410)`；WSL HIP（RX 7650 GRE /
ROCm 7.2）`Ran 2269 tests / OK (skipped=369)`。

**边界**：超时是**协作式**的——只在 `step` 边界生效，一次 `step` 内部（一次
prefill 或一次 decode）不会被打断；要打断单步长内核需要后端支持异步
取消/流中断，那不在参考执行器的能力里。取消不回收已执行的计算，只释放槽位。
队列上限是**条数**而不是字节数：按 KV 状态显存做的准入控制需要
`state_shapes × batch` 的预算模型，属于调度策略层，未做。

### 5.221　集合通信的能力声明与设备侧归约接线

**问题**：5.190 / 5.211 的数据并行只有"主机中转"归约（D2H → 主机求和 → H2D），
没有设备侧集合通信（NCCL / RCCL），也没有计算/通信重叠。缺的不只是性能：调用方
无从得知自己拿到的是哪一级通信，只能从性能曲线反推，而"以为在设备上归约、
其实每卡都往主机拉了一整份梯度"是最难从结果里发现的错误。

**做法**（`train/parallel.py`）：

1. `collective_support(backend, provider=None)` 返回 `CollectiveSupport`
   （`device_allreduce` / `overlap` / `provider` / `reason`）。没有 provider 时
   声明 `device_allreduce=False`，`reason` 点名 NCCL / RCCL 未接入；`overlap`
   恒为 False，理由写进声明：执行器同步、切卡串行，进程内没有重叠窗口。
2. `DataParallelTrainer(collective=...)` 只有 `"host"` / `"device"` 两个取值；
   `"device"` 而没有 provider 时抛 `DeviceCollectiveUnavailable`（继承
   `NotImplementedError`）并带上 `collective_support().reason`——**不静默退回
   主机路径**。注入 provider 却选 `"host"` 直接 `ValueError`：两条归约路径
   只能选一条，不猜。
3. 设备路径的契约是 `provider(per_device, target_device, reduction=...)`：
   输入是"设备号 → 参数名 → **留在设备上的**数组"，返回值必须落在
   `target_device` 上。调用方只把归约结果拉回主机一次，D2H 字节数从
   "每卡一份整梯度"降到"一份整梯度"。
4. provider 的返回值按外部输入校验（`_host_grads`）：不是映射 → `TypeError`；
   名字多/少 → `KeyError`；形状或 dtype 不符 → `ValueError` / `TypeError`。
   断言消息里带 `device_collective`，因为优化器本身也会对缺名字报 `KeyError`
   ——"碰巧同类型"的通过说明不了校验发生过。

**用例**（`tests/test_parallel.py::TestCollectiveSeam`，6 条）：无 provider 时的
声明（设备归约不可用、重叠不可用、原因点名 NCCL）；注入 provider 后声明随之
变化；`collective="device"` 无实现时显式抛能力错误；非法取值 / 混用两条路径 /
provider 不可调用的拒绝面；设备路径与主机路径两步骤的损失与参数逐位相同且
provider 每步只调一次、目标是第一块卡；provider 返回值的三类校验。

**负向对照**（做完还原）：把"没有实现就报错"改成静默 `collective = "host"`
（模拟降级），`test_device_mode_without_provider_is_explicit` 变红；把
`_host_grads` 换成不做校验的字典推导，provider 校验用例的三个子例一起变红
（其中两个在改写前"碰巧也抛了同类异常"——正因如此才把断言收紧到消息内容）。

**验证**：本地 `Ran 2285 tests / OK (skipped=410)`；WSL HIP（RX 7650 GRE /
ROCm 7.2）`Ran 2285 tests / OK (skipped=369)`。

**边界**：没有实现任何真实的 NCCL / RCCL 调用，也没有实现重叠——`overlap`
是**如实为 False 的能力位**，不是一个可以打开的开关。注入 provider 的路径
只验证接线与契约，不验证厂商库行为；多进程 / 多流并行仍属未做（与 5.190
的边界一致）。

### 5.222　真 ZeRO：每进程只持有 / 只落地自己那片槽位

**问题**：5.196 / 5.211 的分片优化器只是**进程内**分片：所有分片仍在同一个
进程里，常驻内存就是整份槽位，"分片省了内存"只是一句账面上的话。跨进程落地
需要两件事：每片能独立写盘、独立装回，且装错片必须报错。

**做法**（`train/sharded.py`）：

1. `slot_bytes_by_shard()` 给出每片槽位的字节数（不含参数本身）：这是"到底省了
   多少"的唯一硬指标——把每一片交给一个进程，每个进程常驻一项，合计仍是
   **一份**整槽位；不分片时每个进程都要整份，合计是 `world_size` 份。
2. `shard_state(rank)` 返回"步数 + 超参 + 只有本片的槽位 + 归属表"。归属表与
   `rank` / `world_size` 一起存档，是为了让"把 rank 1 的存档装进 rank 0"在
   加载时就是显式错误：名字和形状都可能对得上，只有归属不同。
3. `save_shard(path, rank)` / `load_shard(path, rank)` 只碰本片：一份 `.npz`
   里只有本片的槽位数组（键为 `slot::name`）与 `__meta__`（JSON）。加载校验
   顺序是片号 → 文件归属 → 分片数 → 归属表，最后才交给该片的
   `load_state_dict`（它再核对槽位名、形状、dtype 与超参）。
4. 非法 `rank`（bool / 非 int / 越界）在入口报 `TypeError` / `ValueError`；
   缺 `__meta__`、分片数不一致、归属表不一致都抛 `ValueError`。

**用例**（`tests/test_sharded.py::CrossProcessShardTest`，4 条）：各片槽位字节
之和等于不分片的一份（且单片小于整份）；每个存档文件里只出现本片参数名、
合起来覆盖全部参数；各片分别落地再装回后续训一步与不分片优化器逐位相同
（参数本身与槽位分开存，续训从已训练的参数出发）；装错片 / 分片数不同 /
非法 rank 的拒绝面。

**负向对照**（做完还原）：把 `load_shard` 里的片号校验短路掉，加载错误片的
用例变红（此时会落到归属表校验，报错文案不同——说明两道校验各自独立生效）。

**验证**：本地 `Ran 2285 tests / OK (skipped=410)`；WSL HIP `Ran 2285 tests /
OK (skipped=369)`。

**边界**：参数本身仍假定各进程共享（all-gather 未做），省的是**优化器槽位**
的常驻内存与落地体积，不是显存；分片按名字切、不按元素切（见 5.190 的边界）。
`load_state_dict` 对"槽位覆盖不全"的容忍度依赖优化器自身校验，本层额外用
归属表与分片数把"装错片"挡在前面。

### 5.223　torch 兼容矩阵：把"跑过哪个版本"写成可查询的声明

**问题**：前端对齐测试一直只在**一个** torch 版本上跑过（5.152 手工核对过
2.9，2.13 上跑回归），dtype 只有 f16 / f32 / f64。文档里那句"应该也支持"
就是缺口本身：`torch.export` 的 IR 细节（动态维表示、dtype 名、算子拆分）
会随版本变，没验证过就是没验证过。

**做法**（`frontend/compat.py` + `frontend/torch_export.py`）：

1. `TORCH_MATRIX` 每行声明一个版本区间（`min_version`–`max_version`）、该区间内
   允许的 dtype、真正跑过回归的 `verified` 版本与说明。`matrix_row(version)`
   只返回覆盖行，否则抛 `UnsupportedTorchVersion`（继承 `RuntimeError`：这是
   环境不匹配，不是参数写错），消息里带上已验证区间。
2. `parse_torch_version` 接受 `2.13.0+cpu` / `2.9.0a0` / `2.13`；解析不了报
   `ValueError`。`check_torch(module=None)` 做版本 → 矩阵行的校验。
3. `torch_export._require_torch()` 改为返回矩阵行，并在 torch 可用后**校验版本**：
   矩阵之外的版本显式失败，而不是"能 import 就往下跑"。
4. dtype 覆盖到 `bfloat16`（IR 有 `bf16`，前端此前只映射三个浮点）；矩阵外的
   dtype（如 `torch.int64`）报 `UnsupportedOperatorError` 并列出矩阵内 dtype。

**用例**（`tests/test_frontend.py::TestTorchCompatMatrix`，6 条）：矩阵行自洽
（区间有序、`verified` 版本落在区间内、dtype 都能落到 IR dtype）；版本串解析
（含 `+cpu` 与预发布后缀）；矩阵外版本被拒并带区间；`check_torch` 只认版本号
（假模块 1.7.1 / nightly 的两种拒绝）；本机 torch 落在矩阵内；`bfloat16` 可映射、
`int64` 显式拒绝。

**负向对照**（做完还原）：把矩阵上界改成 `99.0`（等价于"什么都支持"），
"矩阵外版本被拒"用例变红。

**验证**：本地 `Ran 2285 tests / OK (skipped=410)`（torch 2.13.0+cpu / py3.14）；
WSL HIP `Ran 2285 tests / OK (skipped=369)`（WSL 内无 torch，矩阵相关用例
按设计跳过——纯声明类的 4 条仍跑）。

**边界**：矩阵是**声明**而不是"自动探测能力"：区间内未标 `verified` 的版本
只是有声明，没有逐版本回归；`verified` 只写真正跑过用例的版本。dtype 覆盖
只到"能映射进 IR"，不代表该 dtype 的数值对齐在设备后端验证过。torch 的
版本升级仍要人工跑回归后把新版本写进 `verified`。

### 5.224　torch 入口补全：整数 token、提升常量与 `get_attr` 常量

**问题**：审计表的 torch 兼容行指出两处入口缺口：`_TORCH_DTYPE` 只有浮点三类，整数 token
输入被拒；`get_attr` 常量节点直接报"暂不支持"。实测 torch 2.13 下 `nn.Embedding` 的
`forward` 导出成 `aten.index.Tensor`（`table[ids]`），常量张量则被提升成 placeholder
（`c_lifted_tensor_0`），包在 `lift_fresh_copy` / `detach_` 里——三者都在入口被拒或被
误当用户输入。

**做法**：
1. 整数 dtype 落成 IR 的 `i32`：`int32` 无损、`int64` 收窄（IR 没有 i64）。收窄不是静默
   截断：执行器入口的 `_check_input_dtype` 先验范围，越界值当场 `ValueError`（5.181）。
2. `aten.index.Tensor` 只接受**单个**整数索引张量，落成 `embedding`；多索引 / 含 `None`
   的通用高级索引显式拒绝（只取第一个索引会把语义悄悄改掉）。
3. `aten.embedding.default` 只在 `padding_idx` 缺省（或 -1）、`scale_grad_by_freq` 与
   `sparse` 为假时落成 `embedding`；`padding_idx >= 0` 显式拒绝——IR 的查表没有"该行梯度
   恒为零"的语义。
4. 图签名登记为 `inputs_to_lifted_tensor_constants` 的 placeholder 落成 `constant` 节点，
   不再进 `user_inputs` / `parameters`；`get_attr` 指向的 GraphModule 属性同样落成
   `constant`。常量写进 attrs（`data`），图仍然自包含、可序列化。
5. 常量有体积上限 `MAX_IMPORTED_CONSTANT_ELEMENTS = 65536`，超限显式拒绝：更大的值应该是
   参数或显式输入，塞进 attrs 会让序列化体积与回读开销失控。
6. `detach_` / `lift_fresh` / `lift_fresh_copy` 与 `clone` / `alias` / `detach` 同属恒等
   包装，直接透传。

**用例**：`tests/test_frontend.py` 新增 8 条——int64 token 的 embedding 往返（并断言输入
声明为 `i32`）、`F.embedding` 无 padding 路径、`padding_idx` 拒绝、多索引高级索引拒绝、
提升常量不改写 `user_inputs` 且图里出现 `constant`、手工构造的 `get_attr` 常量图、
超限常量拒绝、越界 int64 在执行器入口报错。

**负向对照**（做完还原）：关掉提升常量分支 → 常量用例变红（要求调用方喂
`c_lifted_tensor_0`）；把索引长度检查放宽到"只拒绝空列表" → 多索引用例变红；删掉体积上限
→ 超限用例变红。

**验证**：本地 2293 / 410 跳过、WSL HIP 2293 / 369 跳过，全绿。

**边界**：整数只有 i32 一种载体——`int64` 在 `[-2**31, 2**31-1]` 之外的值入口拒绝，不承诺
64 位下标；`padding_idx` 的梯度语义、`sparse=True` / `scale_grad_by_freq=True` 都仍未实现
（显式拒绝）。常量上限是硬编码的，不按图大小自适应；`get_attr` 只接受张量，非张量属性
仍报明确错误。torch 2.13 下 `torch.arange` 之类的字面量生成 `aten.arange` 算子而不是
常量，仍按"未支持算子"拒绝。

### 5.225　训练存档接管数据迭代位置

**问题**：审计表的"完整续训"行最后剩一条：`DataLoader` 自己已经能 `state_dict` /
`load_state_dict`，但 `Trainer` 的存档不认识它——"下一批是哪批"仍要调用方在 meta 里手工
传、手工校验。权重与数据位置是同一份训练检查点的两半，分开管理就会出现"参数接上了、数据
从轮首重来"的静默偏差，而这种偏差在损失曲线上只表现为一个小的跳变。

**做法**：
1. `Trainer.save_checkpoint(..., loader=...)`：把 loader 的迭代位置与配置按
   `data_<键>` 铺平进 meta（存档 meta 只收标量，不能塞 dict），另记 `data_kind`。
   `meta` 里已经有 `data` / `data_*` 时拒绝——两份位置不许互相覆盖。
2. `Trainer.load_training_state(..., loader=...)`：位置与权重**同进同出**。
   - 存档有位置但没传 loader → `KeyError`（不恢复位置等于静默改数据顺序）；
   - 传了 loader 但存档没有位置 → `KeyError`（来源不明的数据流不能续）；
   - `data_kind` 或 loader 配置（seed / batch_size / size / shuffle / drop_last）
     不符 → `ValueError`。
3. `DataLoader.validate_state_dict`：把原来的校验逻辑抽成只读入口，`load_state_dict`
   复用。训练器在**提交权重之前**调用它，避免出现"数据位置已改、权重没恢复"的半现场；
   真正的写入排在 `load_checkpoint` 成功之后。
4. `Trainer.run_loader(loader, epochs, on_step)`：按 loader 自己的位置跑轮次，游标随之
   推进。与 `run` 的唯一差别是数据来源，所以"跑一半存档、恢复后接着跑"不需要手工游标。

**用例**：`tests/test_resume.py` 新增 5 条——`run_loader` 推进位置、存档后恢复的下一批
与原 loader 完全一致、配置变化在写入前拒绝且参数不变、位置的有无必须与 loader 参数一致、
非法参数（非 DataLoader / meta 冲突 / `epochs=0`）显式报错。

**负向对照**（做完还原）：不铺平 `data_*` → 恢复用例报"没有数据迭代位置"；把"存档有
位置但没传 loader"改回静默通过 → 对应用例变红；删掉提交前的 `validate_state_dict` →
配置不符要到 `load_checkpoint` 之后才报错，参数断言捕获到半现场。

**验证**：本地 2298 / 410 跳过、WSL HIP 2298 / 369 跳过，全绿。

**边界**：存档格式仍只收标量，loader 状态必须能 `int()`；`data_kind` 只认精确类名，
子类（自定义 loader）会与基类存档互斥——这是有意的，不同迭代语义不能共用一个游标。
"位置"只到"第几轮第几批"，批次内部的样本偏移（一个 batch 里吃到第几个样本）不在协议内。
流式/无限 loader 没有 `len`，`state_dict` 的 `size` 是构造时的快照，不随流变化。

### 5.226　审查修订：存档命名空间、提交顺序与入口一致性

**问题**：对 5.224 / 5.225 的逐行审查暴露了六个缺陷：

1. `data_*` 前缀把调用方自己的 meta 键（`data_dir`、`data_path` 之类）当成了 loader
   游标，而且 `save_checkpoint` 拒绝裸 `data` 键——这是 5.225 之前不存在的向后兼容破坏。
2. loader 游标在 `load_training_state` 的提交链条**中途**写入：权重已恢复、游标已前进，
   若后面的累积窗口 / BN / AMP 提交失败，会留下半现场。
3. `run_loader` 复制了 `run` 的整个循环，两份 `epochs` 校验与步号逻辑迟早漂移。
4. 常量路径绕过执行器入口：整型常量超过 i32 范围时，`constant` 节点直接落进 attrs，
   没有入口的 `_check_input_dtype` 兜底。
5. 兼容矩阵的 `accepts_dtype` 没有任何生产调用方，`_TORCH_DTYPE` 与矩阵 `dtypes`
   成了两套事实来源。
6. `aten.embedding` 的 `padding_idx` 是图内值时抛裸 `TypeError`，不是显式拒绝。

**做法**：
1. loader 状态改成一个专属标量键：`loader_state`（JSON 字符串）+ `loader_kind`；
   冲突检查只看这两个键，调用方的 `data_*` 完全不受影响。
2. 游标推进移到 `load_training_state` 的**最后一步**（`_refresh_resident()` 之后、
   `return` 之前），保证所有权重与附加状态已经提交。
3. `run_loader` 只做类型校验后委托 `self.run(loader, ...)`。
4. `_constant_node` 对 i32 常量做值域检查，越界在导入时拒绝，并提示"改成显式输入，
   让入口报出具体越界值"。
5. `_to_dtype(dtype, row=None)` 接收矩阵行，`import_exported` 把自己拿到的行传进去：
   声明与入口共用同一份事实来源（当时 `bool` 还在 `_TORCH_DTYPE` 之外；
   5.330 起映射到 `boolean`，5.340 起逻辑组合与 bool cast 也在导入侧接通）。
6. `padding_idx` 是 `Value` 时显式报"只支持静态缺省（None / -1）"。

**用例**：`tests/test_resume.py` 新增 `test_user_meta_is_not_confused_with_loader_state`，
`tests/test_frontend.py` 新增 `test_out_of_range_integer_constant_is_rejected`；原
`test_invalid_arguments_are_rejected` 的冲突键从 `data` 改为 `loader_state`。

**负向对照**（做完还原）：把冲突检查改回 `data_*` 前缀 → 用户 meta 用例报冲突；删掉
末尾的游标写入 → 恢复用例的下一批断言变红；去掉 i32 值域检查 → 越界常量用例变红。

**验证**：本地 2300 / 410 跳过、WSL HIP 2300 / 369 跳过，全绿。

**边界**：`loader_state` 是字符串标量，JSON 解析失败按"存档已损坏"报错，不尝试修复；
矩阵校验只作用于导入入口，已序列化的图不重新校验；整型常量的 i32 检查不改变"IR 只有
i32"这一事实，超界值仍然无法表达；`bool` 常量仍无 IR 语义。

### 5.227　CUDA 真机复跑：cuSPARSE 索引枚举与三条机器相关用例

**问题**：CUDA 侧此前只有离线断言与伪 runtime，`cupy` 装上之后第一次在 2×Tesla
V100-PCIE-16GB（sm_70，CUDA 12.4）上把设备路径真的跑起来，暴露五处问题：

1. `cusparse.py` 的 `_CUSPARSE_INDEX_32I` 写成 1。`cusparseIndexType_t` 里 1 是
   `CUSPARSE_INDEX_16U`，32I 是 2——cuSPARSE 会把 int32 的 `rowOffsets` 按 uint16
   读，越界访问不报参数错误，而是内核 `cudaErrorIllegalAddress`。
2. `cusparseSpMM` 的实参多传了一个 C 描述符（12 个对 11 个形参），参数整体错位一
   位，ctypes 报 `argument 9: TypeError: wrong type`；修好个数之后仍然非法访存，
   根因是第 1 条。
3. `invoke` 的 dtype 拒绝提示只在 f16 时给出"只支持 float32"，f64 没有提示，两条
   用例对同一件事给出不同信息。
4. `test_capabilities_come_from_registry` 断言能力面与内核表**恒等**。5.210 之后
   能力面是"内核表 + 可用的 cuFFT"，在装了 cufft 的机器上这条必然失败。
5. `test_many_rows_stay_on_serial_path` 把行数写死 1024。判据是 `steps < SM×32`
   走并行，M40（24 SM，阈值 768）上 1024 行在串行侧，V100（80 SM，阈值 2560）上
   1024 行仍在并行侧——用例前提在换机器后不成立。

**做法**：

1. `_CUSPARSE_INDEX_32I = 2`，注释写明四个枚举值与"写错会越界读"的后果。
2. `cusparseSpMM` 按 `alpha, A, B, beta, C` 排实参（11 个）。
3. 提示对 f16 / f64 都给，且按 `_f16_ops` 给出可接受集合（`float16 / float32` 或
   `float32`），不再把"只支持 float32"套给其实有 f16 路径的算子。
4. 能力用例改成三条子集关系：内核表 ⊆ 能力面 ⊆ 内核表 ∪ {fft, rfft}，
   `CUDA_OPS` ⊆ 能力面，且 `能力面 − 内核表 == {fft, rfft}` 当且仅当 cuFFT 可用。
5. 行数改成 `max(1024, sm_count * 32 + 8)`，阈值从当前卡现算。

**用例**：`tests/test_cusparse.py` 的 `test_int32_index_maps_to_32i` 断言从 1 改为 2
并补 docstring；`tests/test_kernel_registry.py` 与 `tests/test_cuda_backend.py` 各改
写一条。本轮新增 0 条、修订 3 条，总数不变（本地仍 2300）。

**负向对照**（都在真机上实测到红，随后还原）：

- 枚举改回 1 → `tests.test_cusparse` 7 条 E，且 CUDA context 中毒后把同一次全量里
  后面的 11 条也带成 E（共 17 条）。
- 实参多传一个 `c` → 4 条 E，首行为 `ctypes.ArgumentError: argument 9`。
- 行数写回 1024 → V100 上 1 条 F（"用例前提是 1024 行判给串行铺法"）。
- 能力面改回恒等 → V100 上 1 条 F（多出 `fft` / `rfft`）。

**验证**：本地 2300 / 410 跳过、WSL HIP 2300 / 369 跳过、远端 CUDA 真机
2295 / 119 跳过，三处全绿。真机上 CUDA 组 332 条只跳过 1 条（系统没有 cuDNN），
cuSPARSE 快路径与自研稀疏核在 7 个形状上逐点对齐；`spike_csr` 的两条铺法在
80 SM 的卡上各跑到了一条。

**边界**：cuSPARSE 快路径仍只登记 f32 与 int32 索引（f16 / f64 显式拒绝，不降级）；
V100 只验证了 sm_70 一套指令集，M40（sm_52）的历史结论不重测；远端没有 cuDNN，
cuDNN 卷积的真机结论仍缺（NCCL 已装上，集合通信与跨进程 ZeRO 的结论见 5.228 /
5.229）；远端收集到的用例数比本地少 5（torch 版本差异，历来如此，不是本轮引入）。

### 5.228　NCCL 集合通信：设备侧 AllReduce 与多进程 communicator

**问题**：5.221 把"单进程多卡没有设备侧归约"写成能力声明（`device_allreduce=False`），
原因点名 NCCL 未接入。真机复跑之后要补上这一半，但 NCCL 与现有数据并行的形态不
兼容，暴露五个问题：

1. NCCL 的 communicator 是"一个 rank 一张卡"，`ncclCommInitRank` 阻塞到全部 rank
   到齐；现有 `DataParallelTrainer` 是单进程依次切卡，两者不是同一层。塞进单进程
   需要多线程同时进入初始化，还会把 cupy 的"当前设备"变成线程局部状态——失败时
   分不清是哪一层没设对。
2. `ncclAllGather` 只有 6 个形参（没有归约枚举），`ncclAllReduce` /
   `ncclReduceScatter` 是 7 个。按同一张 argtypes 表绑定时，真机上首个 AllGather
   直接 `TypeError: this function takes at least 7 arguments (6 given)`。
3. `ncclUniqueId` 是 128 字节不透明结构。`bytes(c_char * 128)` 会按第一个 0 字节
   截断——实测返回 `b''`，跨进程分发 unique id 的路径直接失效，而报错现场离根因
   很远。
4. NCCL 只认连续的设备缓冲。主机数组没有 `.data.ptr`，非连续数组的 strides 与线性
   读取不一致；不校验就会在驱动里读到别的内存。
5. bf16 在 NCCL 有枚举，但 numpy 没有 bf16，"主机侧期望值"算不出来；`ncclAvg` 需要
   2.10 以上，版本差异不该藏进开关。

**做法**：

1. 新增 `aetherflow/train/nccl.py`（ctypes 绑定，不引第三方依赖）：
   `NcclComm(rank, world_size, uid)` 构造即初始化，提供 `all_reduce` / `all_gather` /
   `reduce_scatter` / `broadcast` / `group()` / `synchronize()`。库查找先走
   `ctypes.util.find_library`，再试 pip 包里的 `nvidia/nccl/lib/libnccl.so.2`。
2. AllGather 单独绑 6 个形参，AllReduce / ReduceScatter / Broadcast 绑 7 个。
3. `NcclUniqueId.to_bytes` / `from_bytes` 走 `ctypes.string_at` / `memmove`，不经过
   `c_char` 数组的 NUL 截断。
4. `_pointer` 拒绝没有 `.data.ptr` 的主机数组，`_contiguous` 拒绝非 C 连续数组；
   `all_gather` / `reduce_scatter` 在收发缓冲之间交叉校验元素数与 dtype。
5. 登记表只放开 f16 / f32 / f64 与 sum / prod / max / min；bf16 与 avg 显式拒绝，
   消息里给出替代路径。
6. 单进程多卡路径不变（仍是 5.190 的主机中转），5.221 的能力声明也不改：本模块
   服务的是跨进程路径；要把它接进单进程 provider，需要"单进程内多 comm"，那会把
   设备切换变成线程局部状态，不在本轮范围。

**用例**：新增 `tests/test_nccl.py` 16 条：登记表与拒绝面 6 条、unique id 往返 2 条、
communicator 入口校验 5 条、缓冲校验 3 条、真机四集合 1 条。真机用例用
`tests/_nccl_worker.py` 起两个子进程（rank 0 生成 unique id 写文件，其余轮询），
AllReduce / AllGather / ReduceScatter / Broadcast 与主机期望值对拍。

**负向对照**：

- `to_bytes` 改回 `bytes(self.internal)` → `test_round_trip` 1 条 F
  （`b'' != 128 字节`），恢复后全绿。
- AllGather 按 7 个形参绑定 → 真机 1 条 F（上面第 2 条的 `TypeError`）；这是真机
  首次运行实际暴露的失败，修复后四集合全部通过。

**验证**：本地 16 条（15 通过 / 1 跳过，跳过的是"库缺失"那一条）；远端 2×V100
全部通过（真机 1 条不再跳过）。NCCL 版本 23206（2.32.6），从官方 PyPI 安装
`nvidia-nccl-cu12` 后由 `nvidia/nccl/lib/libnccl.so.2` 加载。

**边界**：只验证单机双卡（同一 PCIe 域），不涉及跨节点 IB / RoCE；不提供单进程多
comm；dtype 与归约面同上（bf16 / avg 显式拒绝）；真机判据 1e-5，不承诺逐位
（归约顺序由拓扑决定）。

### 5.229　跨进程 ZeRO-1：只持本片槽位与参数广播

**问题**：5.205 / 5.222 的分片优化器仍把全部片造在同一个进程里——内存账面分片、
常驻仍是整份。要在两个进程上真跑，四个问题：

1. 变长分片与 NCCL `ReduceScatter` 的等长段要求冲突：按参数名分片时每片大小不同。
   ZeRO-1 的通信量本来与 DDP 相同，用整份梯度的一次 AllReduce 更简单，语义不变。
2. 参数更新只发生在片主，其余 rank 的那片参数是旧值；不广播就会在下一步前向里
   各进程使用不同参数——错误表现为"训着训着不一致"，不是异常。
3. 混合 dtype 无法打包成一条缓冲；factory 复制子表会让优化器更新落不回参数。
   这两类必须在构造期 / 入口拒绝，不能靠看损失曲线。
4. 空片（world_size 大于参数数）是合法状态：该 rank 不执行 step，但仍要参与广播，
   否则其余 rank 会卡在集合调用上。

**做法**：

1. 新增 `aetherflow/train/zero.py`：`ZeroRank` 复用 `shard_params` 的确定性划分，
   只对本片参数子表构造优化器，并核对子表与参数表是同一块内存；本片优化器返回的
   名字必须与分片一致。
2. `step`：整份梯度按参数顺序打包 → 一次 `all_reduce(sum)` → 按 `reduction` 除以
   world_size → 只切本片 → 本片 `optimizer.step` → 每个参数按 owner 广播并原地
   写回；空片跳过 step 但仍走广播。
3. `LoopbackWorld` / `LoopbackCollective`：线程屏障 + 调用配对，在进程内模拟多个
   rank 并发；所有集合调用必须按相同顺序发起，与 NCCL 的语义一致。
4. `NcclHostCollective`：H2D → NCCL → 同步 → D2H；广播时非 root 只分配空缓冲。
5. 存档沿用 5.222 的 `slot::name` 格式，但只写本片；装错 rank / world_size /
   归属表都在加载期拒绝。
6. 通信面只依赖四个成员（`rank` / `world_size` / `all_reduce` / `broadcast`），
   真机与离线替身可互换。

**用例**：新增 `tests/test_zero.py` 20 条：分片语义 5 条（两步逐位对拍、槽位只持
本片、优化器名字、归属表、空片）、拒绝面 10 条（factory 复制子表、非 Optimizer、
空参数、混合 dtype、非法 rank/world、attach 缺方法 / rank 不一致 / world 不一致、
未接通信、梯度不齐、reduction 非法）、存档 4 条、真机 1 条。真机用例用
`tests/_zero_worker.py` 起两个子进程跑两步，与 `ShardedOptimizer` 在主机上的
期望值对拍。

**负向对照**：

- 广播后不写回（`array[...] = updated` 改成 `_ = updated`）→ 2 条 F（两步逐位
  对拍、空片一致性），恢复后全绿。

**验证**：本地 2336 个用例 / 412 跳过、WSL HIP 2336 / 371 跳过、远端 2×V100
2331 / 120 跳过，三处全绿。真机上两个进程各占一张卡跑两步，最终参数与
`ShardedOptimizer` 的期望值在 1e-5 内一致；`owned_bytes` 之和等于整份槽位的
字节数，单进程常驻的槽位严格小于整份。

**边界**：ZeRO-1 不是 ZeRO-3——每个进程仍持完整参数，缩小的是优化器槽位；不做
变长 reduce-scatter（那会改变分片布局）；只验证单机双卡；真机判据 1e-5（NCCL
归约顺序不保证逐位），离线 Loopback 判据是逐位。

### 5.230　cuDNN 快路径的真机边界：驱动代际与符号守卫

**问题**：

1. `available()` 只回答"库能不能 dlopen"，不回答"这台机器跑不跑得动"。
   远端装上 cuDNN 9.28.1.4（`version=92801`）后，`conv_impl="cudnn"` 在**第一次
   执行**时才报 `cudnnStatus_t=5003`；而库加载、句柄创建、描述符设置、v7 算法
   选择全部返回 0。5003 经 `cudnnGetErrorString` 翻译是
   `CUDNN_STATUS_EXECUTION_FAILED_CUDART`——cuDNN 9.28 要求 Linux 驱动
   ≥570.26，这台机器是 550.127.05（CUDA 12.4）。
2. `_cudnn_engine()` 只捕获 `CudnnUnavailable` / `OSError`。若某版本的 cuDNN
   连 legacy 符号都没有（`cudnnGetConvolutionForwardAlgorithm_v7` 等），
   `CudnnConv.__init__` 里取符号抛的是 `AttributeError`，它会越过 auto 的
   回落，让默认 `conv_impl="auto"` 在**构造期**整体失败。
3. `_cudnn_broken` 只记 `str(exc)`，异常类型丢失——排障时看不出是"库不在"
   还是"库在但符号没了"。

**做法**：

1. 把 `AttributeError` 归入"这条快路径在这台机器上不可用"，与缺库同路：
   auto 回落自研核，显式 `cudnn` 抛 `BackendCapabilityError`。理由是两者都是
   环境事实，不是调用参数错误；在 auto 下把它冒出去等于让一个可选快路径
   拖垮整个 conv2d。
2. `_cudnn_broken` 改记 `类型名: 消息`，与 `_conv2d_cudnn` 里运行期失败的
   记录格式一致。
3. 判定逻辑（dtype / groups / 捕获期 / 已标记 broken）本来就不碰 GPU，用
   `__new__` 绕过 `__init__` 就能离线验证，不必等真机。

**用例**（4 条，`TestCudnnFallback`）：auto 在符号缺失时回落且**只试一次**
（第二次构造必失败）、显式 `cudnn` 报的错里带 "cuDNN"、f64 与 groups>1 一律
不选、捕获期不切实现。

**负向对照**：把 `except` 里的 `AttributeError` 去掉 → 2 条 error（auto 与
显式各一），恢复后全绿。

**验证**：本地 2340 个用例 / 412 跳过、WSL HIP 2340 / 371 跳过，两处全绿。
远端 2×V100：cuDNN 9.28.1.4 可加载、v7 legacy 符号齐全，执行期返回 5003；
`conv_impl="auto"` 正常回落，数值与 `im2col` 最大差 4.8e-7；`conv_impl="cudnn"`
抛出带状态码的 `RuntimeError`。cuDNN 9.1（仍支持 sm_70、要求驱动 ≥525）的
wheel 下到 121/159 段后被镜像限流，**卷积真机对拍与
`benchmarks/cudnn_compare.py` 的逐形状数据仍是空白**。

**边界**：只在一台 V100 + 驱动 550 上观测；"驱动代际不足"的判据来自 cuDNN
支持矩阵，不是逐版本穷举；`available()` 仍不预检驱动，因为预检要把驱动版本
和 cuDNN 版本配成一张表，那等于把厂商的兼容矩阵抄一遍，且必须随版本更新——
现在的做法是"执行失败就当这条快路径不可用"，代价由 auto 承担、由显式模式的
报错兜底。

### 5.231　ZeRO-2：梯度也只留本片（ReduceScatter + AllGather）

**问题**

1. ZeRO-1（5.229）把**优化器槽位**按参数名分片，但每步仍要把一整份梯度
   AllReduce 一遍：归约结果在每个 rank 上都存在一份，而只有 ``1/world_size``
   会被优化器用到。
2. 参数更新后的同步是逐参数 ``broadcast``，消息数随参数个数线性增长；参数
   一多，小消息的固定开销就盖过带宽。
3. 想直接把分片从槽位推到梯度，卡在 NCCL 的约束上：``ncclReduceScatter``
   要求各段**等长**，而 ZeRO-1 按参数名分出的片大小不一（``b1`` 是 4 个
   元素、``w1`` 是 12 个），凑不出等长段。

**做法**

1. **分段仍按参数边界，但补零到等长。** 一个参数不跨两段（否则优化器要处理
   半张量），段内按全局参数顺序排列；每段补零到最长段 ``pad``，通信缓冲统一
   ``pad × world_size``。代价最多浪费 ``pad - 平均段长`` 的带宽，收益是
   "参数边界"与"等长段"能同时成立。
2. **一步三步走**：打包整份梯度 → ``reduce_scatter`` 只收本段 ``pad`` 个
   元素（``mean`` 在此处除以 ``world_size``）→ 只对本段参数调
   ``optimizer.step`` → ``all_gather`` 本段更新后的参数 → 按段写回完整参数表。
3. **常驻内存下降、通信总量不变。** 梯度峰值从 ``N`` 降到 ``N/W + pad``；
   每步收发各 ``pad × W`` 个元素，与 ZeRO-1 的 AllReduce + Broadcast 同量级。
4. **存档与 ZeRO-1 共用一份写盘逻辑**：``_save_shard`` / ``_load_shard``
   抽到模块级，两类的差别只在归属表。``steps`` 改成
   ``optimizer.step_count`` 的只读视图——另存一份计数器会在 ``load_shard``
   之后与槽位对不上（负向对照第一版的真实症状）。
5. **集合面补齐**：``LoopbackCollective`` 与 ``NcclHostCollective`` 各加
   ``reduce_scatter`` / ``all_gather``，语义与 ``NcclComm`` 逐参对齐
   （``recv`` 原地写入并返回）；长度不符在主机侧就报错。
6. **入口校验**：``attach`` 要求两个方法都在、``rank`` 与 ``world_size``
   一致；每步回读集合结果的形状，偏一个元素就报错而不是把垃圾写进参数。

**用例**（30 条，`tests/test_zero2.py`）

- 分段：每参数恰好属于一段、段内顺序服从全局顺序、段长差不超过最大参数、
  ``pad`` 等于最大段长、``world_size`` 的类型与范围、空参数表拒绝。
- 语义：两步更新与"完整参数 + 梯度均值"的 SGD **逐位**相等；四个 rank 收尾
  参数逐位相同；优化器只看见本段名字；各 rank 槽位字节和等于整份、单片更小；
  2 个参数切 5 段出现空段时仍能跑；通信元素数等于 ``pad × W``；``sum`` 与
  ``mean`` 的更新量相差 ``world_size`` 倍；步数随更新推进。
- 拒绝面：collective 缺一个方法、rank / world 不一致、没接 collective 就
  step、rank 类型与越界、缺梯度 / 多梯度、形状与 dtype 不符、参数混合 dtype、
  factory 复制子表、非法 ``reduction``。
- 存档：只写本段参数名、装错 rank 报错、换初值重建后装回槽位与步数一致。
- 集合面：``reduce_scatter`` 与手算分段和逐位相等、``all_gather`` 按 rank
  顺序拼接、长度不符与非法 ``op`` 拒绝、``Zero2Rank`` 对偏短的集合结果报错。

**负向对照**

- 去掉 ``mean`` 的除法（``/ float(world_size)`` 改成乘 1）：3 条变红——两步
  逐位对拍、空段一致性、``mean`` / ``sum`` 相差 W 倍。
- 把 ``steps`` 改回独立计数器：``round_trip_keeps_slots`` 变红（``0 != 1``），
  正是"两份真相"的症状；随即改成优化器步数的只读视图。

**验证**

- 本地 2387 个用例 / 412 跳过，全绿；WSL HIP 2387 / 371 跳过，全绿。
- **真机未跑**：上一轮那台 2×V100 已释放，``reduce_scatter`` /
  ``all_gather`` 的 NCCL 形参一致性由 ``tests/test_nccl.py`` 的绑定检查覆盖
  （6 形参 / 7 形参），设备路径本身没有本轮的实测数字。

**边界**

- 仍是 ZeRO-2：参数在每个 rank 上都是完整一份，参数驻留拆分是 ZeRO-3，这里
  不做；因此显存省的只有梯度与槽位。
- 各 rank 必须用**同一份参数顺序**构造（同一个模型定义），否则分段对不上；
  这一点不做跨进程核对，只在文档与用例里固定前提。
- 数值不承诺与 ZeRO-1 逐位相等（NCCL 的归约顺序由拓扑决定）；单机替身按
  rank 序号累加，所以离线判据可以是逐位。
- ``pad`` 带来的额外带宽在参数个数远多于 ``world_size`` 时很小；参数很少
  （例如 2 个参数切 5 段）时浪费比例会上去，这是"等长段"换来的代价。

### 5.232　多 rank 启动器：环境变量、unique id 分发与进程拉起

**问题**

1. 5.229 / 5.231 给的是类，没有"把 W 个进程同时拉起来"的入口；每个 rank
   手敲一遍 ``python -m ...``，rank 与端口靠人记，错了要么挂住要么静默跑成
   单进程。
2. ``ncclUniqueId`` 是 128 字节不透明数据，只有 rank 0 能生成，其余 rank
   必须拿到**同一份**。跨机时文件系统不共享；rank 之间也通常没有公网可达的
   端口，不能互相广播。
3. ``multiprocessing`` 的 spawn 会在子进程里重新 import ``__main__``；在
   ``python -m unittest`` 下，那等于把整个测试套件再跑一遍——启动器自己的
   测试会递归爆炸。

**做法**

1. **环境变量三层**：``AF_RANK`` / ``AF_WORLD_SIZE`` / ``AF_LOCAL_RANK`` /
   ``AF_MASTER_ADDR`` / ``AF_MASTER_PORT`` 优先；其次是 torchrun 的
   ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK``；两者都没有时用调用方给的
   默认值，**没有默认值就报错**。猜 0 会让 W 个进程抢同一个 rank，症状是
   NCCL 初始化挂住，而不是一条错误。
2. **uid 走一条 TCP**：``UniqueIdServer`` 在 rank 0 起后台线程 accept，
   协议是"1 字节 rank + 128 字节 uid"；客户端 ``fetch_unique_id`` 带重试直到
   超时（服务端可能晚起几百毫秒）。端口传 0 让系统分配，避免写死端口撞车。
3. **用 ``subprocess`` + ``python -c`` 起进程**：``entrypoint`` 写成
   ``"包.模块:函数"``，参数只接受字符串（跨进程不 pickle）。这既绕开
   ``__main__`` 重入，也把"参数能不能 pickle"这类启动期才暴露的问题提前到
   调用点。
4. **失败不静默**：任何子进程非 0 退出立刻杀其余进程并抛
   ``RuntimeError``（带该 rank 的输出）；正常收尾才检查分发线程的错误——
   顺序反了会让"分发失败"盖掉"退出码 3"这个更直接的诊断。
5. **不鉴权**：与 NCCL 自身的假定一致，master 地址假定在本机或可信内网。

**用例**（17 条，`tests/test_distributed.py`）

- 环境变量：``AF_*`` 优先于 torchrun 名、四个键的回退、缺失且无默认值报错、
  默认值生效、非整数报错。
- uid 分发：128 字节往返且 ``delivered`` 计数正确、uid 长度不符、重复
  ``serve``、rank 0 来拉取、rank 类型、服务端已关时统一报 ``RuntimeError``、
  ``world_size`` 类型与范围。
- 启动：3 个真进程各自写文件，父进程核对 rank 集合、``world_size`` 与三份
  uid 完全一致；子进程 ``SystemExit(3)`` 时父进程抛错且消息含"退出码 3"；
  entrypoint 缺冒号 / 函数名为空、参数非字符串、``world_size`` 类型与范围。

**负向对照**

- 把 ``_int_env`` 的查找顺序反转成"torchrun 名优先"：``prefers_af_prefix``
  变红（读到了 ``RANK=7`` 而不是 ``AF_RANK=2``），还原后全绿。
- 跨平台真缺陷（HIP/Linux 全量第一次跑才暴露）：服务端已关闭后客户端
  ``connect`` 仍能成功（内核的 backlog 还收着 SYN），随后 ``recv`` 抛
  ``ConnectionResetError``。Windows 上同一个场景是静默等到超时，因此本地跑
  不出来。修复是把"连上之后的收发"包进 ``except OSError`` 转成
  ``RuntimeError``，两个平台给同一条可读错误。还原那层 ``except`` 后该用例在
  Linux 上重新变红（``ERROR: ConnectionResetError``），Windows 上仍是绿的——
  这正是"只在一种平台上跑的测试不算验证"的实例。

**验证**

- 本地 2387 个用例 / 412 跳过，全绿；WSL HIP 2387 / 371 跳过，全绿。
- ``spawn`` 的用例是真的起 3 个 Python 进程（``python -c``），不是线程替身：
  退出码、文件内容与 uid 一致性都在父进程侧核对。
- 真机未跑：本轮没有多卡机器，因此"真 NCCL unique id + 真 NCCL 集合"这条
  完整链路仍缺一台机器的实测（上一轮 5.228 / 5.229 的结论不含本模块）。

**边界**

- 只做"单机多进程 + 单 master 分发"，没有 rendezvous 的成员变更与容错：
  某个 rank 挂掉时其余进程会卡在集合调用上，需要外部看门狗。
- 参数只能是字符串，复杂对象要调用方自己序列化到文件或环境变量。
- 不做端口自动探测之外的网络配置（防火墙、NCCL 的 ``NCCL_SOCKET_IFNAME``
  等仍由环境变量决定）。

### 5.233　ZeRO-3：参数本身也只留一块（按参数均分 + 用时拼回）

**问题**

到 5.231 为止，参数在每个 rank 上仍是一整份：ZeRO-1 切了优化器槽位，ZeRO-2
又切了梯度，但前向与反向要读权重，所以每个进程都驻留完整参数。参数一多，
"每张卡一份副本"就是常驻内存的下限——这才是 ZeRO 三级里最贵的那一份。

**做法**

`train/zero3.py` 把分片从"槽位 / 梯度"推到参数：每个参数按元素均分成
`world_size` 块，rank `r` 只常驻第 `r` 块（一维、定长 `chunk = ceil(n/W)`，
尾部补零）。参数的真身从此只是一堆分片，要算的时候临时拼回来：

    materialize(names)  →  每个名字一次 AllGather，拼回完整形状（进缓存）
    前向 / 反向          →  用拼回来的那份算
    release(names)      →  丢掉副本，回到只持分片的状态
    step(grads)         →  每个参数 ReduceScatter 出本块的梯度、只更本块

常驻内存从"参数 + 槽位"各一份降到各 `1/W`；代价是前向 / 反向期间被借用的
那部分参数会短暂存在。

**为什么按参数均分，而不是把所有参数拍成一条大缓冲再切。** 拍平能让一步只发
一条消息（5.231 的补零做法），但参数边界会与块边界错位：某个参数横跨两块时，
优化器要在两块上各算半张量，逐参数的权重衰减与学习率也会被劈开。按参数均分
没有这个问题——每块都属于唯一一个参数，优化器看到的仍是一张完整形状的参数，
只是被切成了 `1/W` 长的片段，数值语义与不分片完全相同。代价是消息数随参数
个数线性增长（与 ZeRO-1 的参数广播同量级），这里刻意接受。

**用例**（23 条，`tests/test_zero3.py`）

- 分块计划：`ceil` 除法、元素数 3 / W=2 的零头、非法 `world_size`（bool /
  浮点 / 0）、空参数表与空张量。
- 语义：`materialize` 与完整参数逐位相等（W=2 与 W=3）、一步与两步都与整份
  SGD 逐位对拍（两步是为了证明拼回来的参数还能继续训）、零头参数的补零位置
  既不影响数值也不会被写回（尾块第二个元素恒为 0）、常驻字节严格小于一整份
  而各 rank 合计不小于一整份、缓存复用直到 `release`、`step` 之后缓存立刻作废、
  优化器里只有本块。
- 拒绝面：未接通信就 `materialize` / `step`、梯度缺失 / 多余 / 形状 / dtype、
  非法 `reduction`、`names` 传单个字符串、不认识的参数名、`attach` 缺方法或
  rank / world 不符、factory 复制子表、混合 dtype、空表、非法 rank。
- 存档：只写本块（`param::` 键的形状等于块长）、往返后参数与步数都对得上、
  装错 rank、`world_size` 不符、块长不符三种情况都显式报错。

**负向对照**

1. 把 `step` 里的 `send[:size] = grads.ravel()` 改成只填前 `chunk` 个元素 →
   3 条失败（一步 / 两步 / 零头参数的对拍），还原后全绿。
2. 删掉 `step` 末尾的 `self._cache.clear()` → `test_step_invalidates_the_cache`
   失败（参数变了却还留着旧副本），还原后全绿。

**验证**：本地 2410 个用例 / 412 跳过，全绿；WSL HIP 2410 / 371 跳过，全绿。
真机未跑（没有多卡机器），NCCL 的 `AllGather` / `ReduceScatter` 仍只有
argtypes 绑定检查与 loopback 语义验证。

**边界**

- 参数表必须在任何 rank 上都以同一份定义顺序构造，否则块与参数对不上；这里
  不跨进程核对，只把"同一模型定义顺序"当前提。
- `materialize` 返回的是**副本**：改它不会回写分片。要改参数只能走 `step`。
- 释放时机由调用方决定：`release` 早了后续算子会读到已释放的副本（`materialize`
  会重新拼，因此不会静默拿到空数组，但会多一次通信）。
- 存档必须同时保存参数分片与槽位，这与 ZeRO-1/2 只存槽位不同；也正因如此，
  `world_size` 一变就必须先合并再重新分片，不能直接装。

### 5.234 ZeRO-3 接进训练循环：逐段借出参数，算完就还

**问题**

5.233 只交付了"分片容器"：`Zero3Rank` 会切块、会拼回、会只更本块，但
`Trainer` 完全不认识它。真要用起来，调用方得在前向之前手动 `materialize()`
整份参数、反向之后手动 `release()`——那等于每次训练都拼一份完整参数，分片
只剩"存档时省点空间"。分片能存不能训，这一层就没闭环。

**做法**

1. `CheckpointPlan.forward` / `backward` 增加 `on_segment` 回调：在**每段**
   的 `_feed_of` 之前进入、执行完退出。放在 `_feed_of` 之前是硬要求——段图
   从 `feed` 里取参数，晚一步就是 `KeyError`。
2. `Trainer(zero3=..., segments=N)`：每段执行前把**这一段要读的参数**
   （`segment.inputs` 里属于参数的子集）`materialize` 进 `feed`，执行完把
   引用从 `feed` 删掉再 `release`。只删 feed 不 release 会留着缓存，只
   release 不删 feed 会让 `feed` 的引用把内存钉住，两者必须成对。
3. 更新走 `zero3.step(grads)`：完整梯度 → ReduceScatter 出本块 → 只更
   本块。因此 `optimizer` 必须是 `zero3.optimizer`，构造期不一致直接拒绝。
4. `store` 从"参数真身"降级为"借出的完整副本"：训练期间它是旧值，
   `sync_to_host()` 拼回一次覆盖它；`snapshot()` 发现它是旧的就先 sync，
   免得 checkpoint 静默写进训练开始前的权重。
5. BN 的 running stats 统计子图也按需借用：只拼这张子图输入里的参数，
   不为一次 EMA 观察把整份借出来。

**用例**（13 条，`tests/test_zero3_train.py`）

- 语义：两个 rank 各喂同一批数据，两步的损失与参数都与"单份参数的 2 段
  训练"**逐位**相等（ReduceScatter 求和再除以 2 是精确运算）；续训 2 + 2
  步与不中断跑 4 步逐位相等，分片存档带回优化器步数。
- 结构性判据：一步之内每次 `materialize` 的 `names` 都非空且严格少于全部
  参数（5 个参数、2 段时峰值 3 个），一步跑完 `resident` 为空；`store`
  在 `sync_to_host()` 之前是旧值、之后等于分片；累积 + `grad_hook` 仍然
  生效（窗口 2 步，钩子每 rank 一次，优化器只前进一步）。
- 拒绝面：没有 `segments`、`device_state`、`master_weights`、`GradScaler`、
  `optimizer` 不是分片优化器、参数集合 / 形状 / dtype 不一致。

**负向对照**

1. 把 `_borrowed_params` 里的 `materialize(names)` 改成 `materialize()`（整份
   拼回）→ `test_each_step_only_borrows_the_segment_params` 失败（记录到
   `None`），还原后全绿。
2. 把更新改成 `zero3.step(grads, reduction="sum")`（漏掉除以 world_size）→
   两步对拍失败（损失 -71.30 vs -31.52），还原后全绿。

**验证**：本地 2423 个用例 / 412 跳过，全绿；WSL HIP 2423 / 371 跳过，全绿。
真机未跑（没有多卡机器）：NCCL 侧仍只有 argtypes 绑定检查。

**边界**

- **必须同时给 `segments`**：整图前向要一次拼回全部参数，"只留一段"就不
  成立，构造期直接拒绝，而不是给一个看起来在分片的假象。
- 省下的是"其它段的参数"，不是"段内参数"：前向与反向各拼一次同一段，
  段与段之间不重叠。段数越多单段越小，消息数也越多（每参数一次 AllGather）。
- 完整梯度在 ReduceScatter 之前仍是主机内存里的一份 `N`——这是本实现的
  边界；按段累积梯度、把峰值也压到 `N/W` 需要另一层改动。
- 与 `device_state` / `master_weights` / `GradScaler` 互斥（各自还要一份完整
  参数，或没有分片版本的缩放逻辑）；`grad_hook` / 梯度累积 / `schedule`
  只看完整梯度，照常可用。
- `Trainer.steps` 不随 `load_shard` 恢复（分片存档带的是优化器步数）；
  要恢复训练器自己的计数器走 `save_checkpoint` 的 meta。

### 5.235 分布式存档的合并与重分片：N 个分片 ↔ 一份完整状态

**问题**

5.229～5.234 的 `save_shard` 都只写"本 rank 那一份"：单卡数续训够用，但两件
事做不了——把 N 个分片合成一份完整权重（推理、发布、单卡续训），以及换
`world_size`（8 卡训到一半改 4 卡）。后者不是"多连几个 rank"能解决的：分片
数、归属表、块长全变了，`load_shard` 会按"分片数不一致"直接拒绝，于是续训能
跑、换卡数接不上。

**做法**

1. 新增 `train/shard_io.py`，三件纯文件操作：`merge_shards(paths, params=?)`
   合并、`ShardedCheckpoint.save/split/write_shards` 输出、`split_checkpoint()`
   把一份完整 checkpoint 按目标 `world_size` 切开。**全程不碰通信**：换卡数靠
   "先拼回完整状态、再按新布局切"，而不是让两个不同规模的 world 直接对话。
2. 布局由 meta 判定，不做猜测：有 `plan` 是元素布局（ZeRO-3，存档里带参数
   分片）；有 `assignments` 是按名布局（ZeRO-1/2，**只带槽位**，参数在模型
   里）；两者都没有就报"不是分片存档"。
3. 合并前先校验整组：`world_size` 一致、rank 0..W-1 不重不漏、步数一致、
   超参逐键一致、布局一致。按名布局还要查"rank 把参数记在了别人名下"这种
   自相矛盾的归属表，以及槽位与归属表对不上的缺片。
4. 元素布局合并：按 rank 顺序拼块、**截掉尾部补零**再 reshape 回参数形状。
   槽位也一样——留在 flatten 后的样子会被 `Optimizer.load_state_dict` 按
   "形状不符"拒绝。切分时再 ravel，等价往返。
5. 按名布局合并：槽位取并集，参数只能由调用方给（`params=`），`save()` 缺
   参数直接报错——不用零填充伪造一份权重。元素布局下 `params=` 反过来是
   oracle：逐位不等说明分片来自另一份权重，直接拒绝。
6. 重切：按名布局用 `shard_params` 重算归属（需要参数形状），元素布局用
   `plan_chunks` 重切块长；同 `world_size` 且没给参数时沿用存档里的归属表。
7. 输出与 `save_shard` 逐键同构（`param::` / `slot::` / `__meta__`），对应类的
   `load_shard` 直接读；`save()` 写 `checkpoint.py` 的同构格式，`load_checkpoint`
   直接读。合并产物的判据就是"能被既有读取路径读回、数值逐位相等"。

**用例**（21 条，`tests/test_shard_io.py`）

- 合并：ZeRO-1 两步存档合并出的槽位逐位等于片主优化器里的那份，`save()` 后
  `load_checkpoint` 读回的参数与步数都对；ZeRO-2 原样写回两份装进新实例，
  再走一步与旧实例逐位相等。
- 换 `world_size`：ZeRO-1 2 → 4 后归属按 `shard_params` 落位、槽位值不变；
  ZeRO-3 2 → 3 后参数逐位拼得回来，再各走一步合并的结果与单份 SGD 逐位相等。
- 全量 → 分片：`split_checkpoint` 的两种 layout 各写 3 份，分别被
  `Zero3Rank` / `ZeroRank` 的 `load_shard` 读回。
- 拒绝面：缺片、重复片、`world_size` 混杂、步数不一致、布局混杂、把普通
  checkpoint 当分片、短块、块布局不一致、拿错权重、无参数时 `save()`、
  换 `world_size` 却不给参数、命名模板缺 `{rank}`。

**负向对照**

1. 元素合并漏掉"截掉尾部补零"（`flat[:size]` 改成 `flat`）→ 4 条用例失败
   （补零后的块长度放不进参数形状），还原后全绿。
2. 重切时把块起点写成 `start = 0`（所有 rank 拿同一块）→ 2 条用例失败
   （2 → 3 后参数重复），还原后全绿。

**验证**：本地 2444 个用例 / 412 跳过，全绿；WSL HIP 2444 / 371 跳过，全绿。

**边界**

- 按名布局的存档里**没有参数**，"合并出权重"必须由调用方给模型权重；本模块
  不替你决定"哪个数组是参数"。
- 重切只保证"参数与槽位重新落位"，不保证跨拓扑逐位：NCCL 的归约顺序仍由
  拓扑决定（ZeRO-1/2 本来就不承诺逐位）。元素布局的 AllGather 只搬数据，
  拼回来的参数是逐位的。
- 换卡数是一次性动作：合并后先落盘再启动新世界，别在两个 world 之间直接
  传文件读。分片文件仍是"每 rank 一份"，任何进程都能独立合并。
- 元素布局要求全部参数同一 dtype（`Zero3Rank` 的构造约束），合并与重切沿用
  这个前提；混合精度请先统一到主权重精度。
- 流式状态（`state/`）与附加数组（`aux/`）不在合并范围内：它们属于推理
  会话，不属于"可切分的训练状态"。

### 5.236 主机中转的多进程数据并行：`launch` 一条命令拉起 W 个进程

**问题**

5.232 的 `spawn` 只是"起进程"，进程之间没有通信通道：集合调用要用 NCCL，
而 NCCL 只在特定 GPU / 驱动上可用。5.225 的 `parallel.py` 是**线程内切卡**的
`DataParallelTrainer`（一份参数、一个进程、按设备切 batch），`Trainer` 走的就是
这条单进程路径；真多进程（各自一份参数、各自的 `Trainer`）则要用户自己写
worker，还得自己搭一套跨进程归约。于是"换到多进程"不是配置项，而是一次重写。

**做法**

1. 新增 `train/host_collective.py`：`HostRelay` 是父进程里的 TCP 服务，接受
   `world_size` 条连接（1 字节 rank 握手，越界 / 重复直接报错），每个 rank
   一条线程。`HostCollective` 是客户端，按环境变量 `AF_MASTER_ADDR` /
   `AF_RELAY_PORT` 连上去，提供 `all_reduce(array, op="sum")`。
2. 帧格式 `struct("!QQ8s")` = `(seq, count, dtype 左对齐 8 字节)` + 裸字节。
   服务端按 `seq` 归组，校验整组 `count` / `dtype` 一致，按 rank 顺序
   `np.frombuffer` 相加后原样发回。**只做 sum**，平均留给调用方——均值是
   "除以 world_size"，放在客户端做不会引入额外的传输。
3. `launch(entrypoint, world_size, *args, ...)` 把三件事串起来：起
   `HostRelay` → `spawn(..., extra_env={"AF_RELAY_PORT": port})` → `finally`
   关 relay 并检查 `relay.error`。`spawn` 新增的 `extra_env` 只补不覆盖，
   `AF_RANK` / `AF_WORLD_SIZE` / `AF_LOCAL_RANK` / `AF_MASTER_ADDR` /
   `AF_MASTER_PORT` 这五个保留键出现即报错。
4. `parallel.DistributedDataParallel(trainer, collective, *, reduction="mean")`
   是 `Trainer` 的多进程外壳：挂到 `trainer.grad_hook` 上做梯度归约（先
   AllReduce、再调用户原来的钩子），`step(batch)` 按 rank 切第 0 维、调
   `trainer.step(local)`，损失再跨 rank 求和除 world_size。
5. 切片的形状要能对上静态图：新增 `Trainer.data_input_shapes()`，返回"非参数、
   非 `rng_inputs`"的输入形状。每卡分片大小必须等于图声明的第 0 维，否则
   在 `step` 里直接报错，而不是等执行器抛形状不符。
6. 构造期拒绝几种"看起来能跑其实语义不对"的组合：`rng_rank /
   rng_world_size` 与 collective 不一致（各 rank 会拿到同一张 dropout 掩码）、
   `device_state`（设备态变量归约没有实现）、`scaler`（unscale 与归约的顺序
   没定）、`zero3`（参数分片与梯度归约是两套逻辑）。
7. 导出面：`aetherflow.train` 里新增 `HostRelay`、`HostCollective`、
   `launch`、`DistributedDataParallel`。

**用例**（19 条，`tests/test_ddp.py`）

- `TestHostRelay`（7 条）：真 TCP 两个 rank 求和；`dtype` 不一致、`count`
  不一致都在服务端报错；重复 `serve()` 报错；`op!="sum"` 与整型输入被拒；
  `from_env` 缺 `AF_RELAY_PORT` / 缺 `AF_MASTER_ADDR` 分别给能看懂的错误。
- `TestDataInputShapes`（1 条）：只列数据输入（参数与 `rng_inputs` 不在内）。
- `TestDistributedDataParallel`（10 条）：两 rank 与"单进程合并 batch"两步
  **逐位**对拍；返回的损失是全局均值；用户钩子看到的是归约后的梯度；`rng`
  不匹配、batch 不整除、分片大小与图不符、数据名缺失 / 多余、设备态 /
  `GradScaler` / ZeRO-3、collective 非法、`reduction` 或 trainer 类型错误。
- `TestLaunch`（1 条）：`launch("tests._ddp_worker:train", 2, ...)` 拉起两个
  真进程，各训两步后与单进程两步对拍（`rtol=1e-5`、`atol=1e-6`），两个退出码
  都是 0。

**负向对照**

1. `_reduce_grads` 里把 `reduction == "mean"` 改成 `"sum"`（跳过平均）→ 3 条
   用例失败（单进程对拍、钩子梯度范数、真多进程对拍），还原后全绿。
2. `step` 末尾改成 `return loss`（返回本片损失而非全局均值）→ 3 条用例失败
   （单进程对拍、全局均值、真多进程对拍），还原后全绿。

**验证**：本地 2463 个用例 / 412 跳过，全绿；WSL HIP 2463 / 371 跳过，全绿。

**边界**

- 只做 `sum`。平均、`max`、`prod` 都没实现；需要别的归约先在调用方组合。
- 走的是主机 TCP，`all_reduce` 的形状越大越吃亏：这条路径是为"没有 NCCL 的
  环境"准备的，不是 GPU 集群的性能路径。
- 图必须静态 batch，且每卡分片大小固定；末批不满要 `drop_last` 或补齐。
- 需要一个常驻的父进程（`launch` 或自己起 `HostRelay`）；纯 `spawn` 起出来的
  进程连不上 relay，`from_env` 会明确报"缺 `AF_RELAY_PORT`"。
- 数据并行只归约梯度，参数各 rank 各存一份；存档仍是 5.235 的分片格式，
  换卡数请先合并再重切。

### 5.237 静态执行快路径：把每步的判断收成一条记录

**问题**

性能清单上 `transformer_block` 与 CUDA 参考实现还差 8%。在 WSL HIP 真机上
（RX 7650 GRE / ROCm 7.2）把这一次执行拆开看：

- 默认 `memory_mode="container"` + `graph_mode="off"`：中位 5.46 ms；
- `memory_mode="tiered"` + graph off：4.64 ms（−27%，分层分配器省掉重复的
  底层分配）；
- `graph_mode="auto"`：0.81～0.92 ms（约 6.7×）；
- `cProfile`：每次执行 32 次 malloc/free、13 次 memcpy（12 次是权重 H2D，
  1 次是 D2H）、12 次 launch、8 次 `sgemm_batched`；`run_device` 自身的
  Python tottime 约 0.27 ms，而主机侧提交合计 6.19 ms——**比 GPU 计算还长**。

也就是说"剩 8%"的主体不是算术，是逐节点的骨架 + 驱动提交。图重放是正解，
但它在 5.79 / 5.80 就已交付、默认 off（随机掩码与主机同步的语义差异需要
使用者知情）。本轮先把**不依赖图捕获**的那部分主机开销砍掉。

本地没有 GPU，所以换了一个能测的口径：空内核（no-op）执行器上的 24 节点
图，每节点约 2.7 µs、整图 66～71 µs。这恰好是把骨架单独抽出来的量。

**做法**

`Executor.run_device` 每节点都要走一遍：构造 `args` → `group.resolve(binding)`
→ `invoke_group`（多层调用）→ 形状比较 → `cast` → `quantize` → 写回 `env`
→ 递减 `release`。前四项对一个**静态单节点组**每次都会得出同一个答案。

于是给 `_GroupPlan` 加一个 `fast` 槽位，第一次执行时由 `_fast_step` 把结论
收成 8 元组 `(kernels[0], node, used, attrs[0], expected.shape, output.id,
dtype, quantize 标志)`；之后的执行只做"取输入 → 调内核 → 复核 → 写回"。

`fast` 是三态：

- `None`：还没试过，第一次执行时才判断（整份计划因此不需要额外标志位）；
- `()`（`_NO_FAST`）：已确认不能直连，永远走通用路径；
- 非空元组：直连记录。

不直连的四类情况都回退通用路径：符号图（`attrs is None`）、融合组
（`len(nodes) != 1`）、常量节点（`output.id in group.constants`）、后端没绑定
kernels（`kernels is None`）。

**语义不变**是这一节的关键：运行期仍然做 `out.dtype != dtype → self.cast` 与
`out.shape != shape → ShapeMismatchError`，bf16 仍然 `quantize`。快路径只省
"反复问同一个问题"，不省"检查本身"。

效果：本地空内核口径，24 节点骨架 69.3 → 40.0 µs（−42%）。48 节点的读数有
GC 抖动，不可靠，不作为结论。

**用例**（8 条，`tests/test_executor_fastpath.py`）

- `FastPathEquivalenceTest`（5 条）：快路径与通用路径的**输出**逐位一致、
  **环境里每个中间值**逐位一致（`run_device` 的环境逐键对拍）、首次执行后
  `group.fast[0]` 已绑定内核、内核返回更宽的 dtype 时仍 cast 回声明类型、
  形状报错文案与通用路径**完全相同**。
- `FastPathBoundaryTest`（3 条）：带符号维的图留在通用路径（`group.fast == ()`）、
  多节点融合组留在通用路径、bf16 输出仍被量化（`group.fast[7]` 为 True）。
  一条都不碰设备。

**负向对照**

1. `if out.shape != shape:` 改成 `if False and out.shape != shape:` → 1 条失败
   （形状报错文案对照），还原后全绿。
2. `if out.dtype != dtype:` 改成 `if False and out.dtype != dtype:` → 1 条失败
   （dtype 回写），还原后全绿。

**验证**：本地 2471 个用例 / 412 跳过，全绿；WSL HIP 2471 / 371 跳过，全绿。
HIP 真机上这一项的量级小于 1%，被驱动提交与 H2D 的噪声淹没，**测不出来也不
声称有收益**；文档里只记本地骨架口径的数字。

**边界**

- 只直连"一个节点、属性静态、非常量"的组。融合组、符号图、常量节点照旧走
  通用路径；要让它们也受益需要图重放，而不是再加一层判断。
- CUDA 侧本轮**没有真机**（2×V100 已释放）。8% 的构成是在 HIP 上同构推断的：
  两家的逐节点提交开销同一量级，结论可以迁移，但具体毫秒数不迁移。
- 默认值一个都没改：`memory_mode` 仍是 `container`（5.131 / 5.170 的决定）、
  `graph_mode` 仍是 `off`。本轮只削骨架，不动语义开关。

### 5.238 中等通道的 3x3/5x5 在 HIP 上改走隐式 GEMM

**问题**

性能清单上还有一项：卷积在中等通道（C = 8～128）的 3x3/5x5 上，比 CUDA 参考
实现慢 4～12 倍（5.69 的记录）。这项的候选解法一直是隐式 GEMM——把卷积直接
展开成"每个输出像素点乘一整块权重"的 GEMM，不落 im2col 的中间矩阵，省掉
`KH*KW*C` 倍的显存流量与一次写读。

但 5.151 在 M40 上做过 A/B：隐式 GEMM 在 K ≥ 288 的 6 个形状上输给 im2col
1.37～2.32×，所以当时 `auto` 明确**不**启用隐式，只保留手动指定。本轮 CUDA
真机不可用，于是在 WSL HIP（RX 7650 GRE / ROCm 7.2）上把同一批中等通道形状
重测了一遍，结论相反：隐式全面赢。auto 与 im2col 交错测量（5 次预热 + 3 轮
交错、每轮 10 次取中位、最后取最小值，规避顺序噪声）：

| 形状 | auto（隐式） | im2col | 加速 |
|---|---|---|---|
| 3x3, C=3 | 1.589 ms | 1.577 ms | 0.99× |
| 3x3, C=8 | 3.957 ms | 6.059 ms | 1.53× |
| 3x3, C=16 | 4.230 ms | 6.677 ms | 1.58× |
| 3x3, C=32 | 6.734 ms | 9.102 ms | 1.35× |
| 3x3, C=64 | 7.242 ms | 9.900 ms | 1.37× |
| 3x3, C=128 | 2.501 ms | 3.332 ms | 1.33× |
| 5x5, C=16 | 4.468 ms | 7.271 ms | 1.63× |
| 5x5, C=32 | 7.169 ms | 10.559 ms | 1.47× |
| 1x1, C=32 | 1.511 ms | 1.693 ms | 1.12×（tiled） |
| 1x1, C=256 | 4.286 ms | 4.157 ms | 0.97×（噪声内） |

两格负向数据决定了改道的边界：C=3 的 3x3（K=27）持平，小 K 上分块直算仍
更划算；1x1 的 C=256 那格 auto 与 im2col 持平，更早的直接对比里隐式比
im2col 慢 6%。所以这条规则不是"大就换"，而是**只在厂商实测支持、核面积
大于 1、K 超过分块阈值时**才换。

**做法**

跨厂商结论相反，改基类默认值就等于把 CUDA 的行为一起改掉，而 CUDA 这轮
没有真机可以复核。于是加一个厂商开关：

- `DeviceBackend.CONV_AUTO_IMPLICIT = False`，注释里写明依据是 5.151；
- `HipBackend.CONV_AUTO_IMPLICIT = True`，依据是本节的表。

`_pick_conv_kernel` 的判定顺序随之变成：显式 `conv_impl` 永远不被改写 →
`auto` 且开关开、`kh*kw > 1`、`k > CONV_TILED_MAX_K`（32）、shared 预算放得
下 → 隐式；否则回到原有的 tiled / im2col 选择。调用点不变：`dims` 里本来
就有每组通道数 `c` 与 `sw` / `dw`。

"预算放得下"抽成纯函数 `conv_implicit_fits`，它与 `plan_conv_implicit`
**共用**同一份算式：shared 里同时放权重 tile（`toc * kh * kw * 4`）与输入
窗口（`kh * span * 4`），48 KB 预算内取最大的 OC tile，再看
`tile_w * (toc // roc)` 是否超过 block 上限。判定与规划同源，避免"选得中
但规划时抛错"。

**用例**（13 条，`tests/test_conv_impl_pick.py`）

- `TestConvImplicitFit`（8 条）：中等通道 3x3 / 5x5 能放下；`fits` 用的
  tile 宽度与 `plan_conv_implicit` 算出的 `tile_w` 一致；窗口与权重**竞争**
  同一份预算（`conv_implicit_oc_tile(32, 48, 32, 79)` 只看权重是 8、计入
  窗口是 4）；120x120 的超大核明确放不下；7 组空/负维度（含 `tile_w=0`、
  `dw=-1`）一律拒绝；默认 roc 下线程数不越 block 上限。
- `TestAutoConvImplPick`（5 条）：厂商开关的取值（基类 / CUDA 为 False、
  HIP 为 True）；HIP 的 auto 在 3x3 K=72/144/288 与 5x5 K=400 上选隐式；
  1x1 保持 im2col；小 K（C=3，K=27）保持 tiled；CUDA 的 auto 在同样的中等
  通道形状上仍选 im2col；显式指定 `im2col` / `tiled` 不会被改写。
  一条都不碰设备。

**负向对照**

1. 把 `HipBackend.CONV_AUTO_IMPLICIT` 改回 `False` → 2 条失败（厂商开关、
   HIP 的 auto 选隐式），还原后全绿。
2. 去掉 `conv_implicit_oc_tile` 里 `+ kh * span * 4` 的窗口项 → 1 条失败
   （窗口与权重的预算竞争），还原后全绿。第一版用例用的是 `kh=kw=40`，
   该形状只看权重也给同一个 tile，改坏后不变红；换成上面那组能区分的形状
   才暴露出差异。

**验证**：本地 2484 个用例 / 412 跳过，全绿；WSL HIP 2484 / 371 跳过，全绿。
HIP 真机上 auto 与 im2col 的输出在 1e-5 内一致（10 个形状里 9 个最大差为 0，
5x5 C=32 为 2.38e-05）。

**边界**

- CUDA 侧本轮**没有真机**，`CONV_AUTO_IMPLICIT` 保持 False，行为与 5.151
  的决定逐字不变；本节不声称 CUDA 上也有收益。
- 1x1 与"核面积等于 1"的形态不改道：直接对比里隐式在 1x1 上慢 6%，而
  tiled 在 C=32 上已经快 12%。
- `benchmarks/cases.py` 里没有中等通道卷积用例，所以没有端到端（整图）数字，
  上面全是单算子口径。
- 与 tiered / graph 模式无关：这里改的是"选哪个内核"，两者的默认值都没动。

### 5.239 weight-only int8 量化：权重驻留从 4 字节压到 1 字节

**问题**

README 的边界清单里，"量化推理"一直是空白：没有 int8 权重的表达方式，也
没有一条能与参考语义逐位对拍的量化路径。缺的不是"除以一个 scale"这个乘法，
而是四条必须钉死的语义——舍入方式、饱和端点、scale 的作用域、以及反量化之后
**计算在什么精度上做**。这四条任意一条含糊，跨后端就不可能对齐：RNE 与
四舍五入在 .5 处差 1，[-128,127] 与 [-127,127] 在极值处差 1，逐通道与逐张量
的 scale 直接决定小通道会不会整片归零，而"int8 累加"与"反量化后 f32 累加"
是两种数值结果。

**做法**

三个 PORTABLE 算子，语义全部写在 `ir/ops.py`，参考实现是唯一真值：

| 算子 | 输入 → 输出 | 用途 |
|---|---|---|
| `quantize_i8` | f32 权重 + f32 scale → i8 | 离线导出 |
| `dequantize_i8` | i8 + f32 scale → f32 | 唯一读出口 |
| `linear_i8` | f32 x + i8 权重 + scale [+ bias] → f32 | 服务路径 |

为什么是三个而不是"一个 linear_i8 加几个 cast"：量化的舍入与饱和没有现成
算子能表达（`cast` 是截断，`clip` 不存在），所以 `quantize_i8` 必须是语义
算子；`dequantize_i8` 单独存在的理由是它让"权重读出"成为可检查、可对拍的
一步，`linear_i8` 的数值就定义为 `matmul(x, dequantize_i8(q, s)) + bias`；
`linear_i8` 收成一个节点，是为了让图里只有一种写法——否则"先 dequantize 再
matmul"和"直接 linear_i8"会同时存在，融合与显存预算无从判断该按哪条走。

语义选择与理由：

- **对称、无零点，端点 `[-127, 127]`。** 不用 -128：反量化后两端量程不对称，
  同一份权重在不同实现里会被舍到不同端点，逐位对拍失去意义。
- **RNE（`np.rint`）。** 四舍五入在 .5 处系统性高估，量化偏差随层数累积。
- **scale 严格为正，参考与设备都拒绝非正。** `scale = 0` 不是"退化成另一种
  量化"，而是除零；负 scale 会翻转权重符号。两者都属于语义未定义。
- **scale 用广播表达作用域。** `(N,)` 对 `(K, N)` 就是逐输出通道；逐张量就是
  `(1, 1)`。符号维只在"等于 1 或两边相同"时放行，与 `_broadcast_shape` 同一
  套保守规则——IR 里没有"这一维可能是 1"的广播维。
- **计算仍是 f32。** `linear_i8` 先反量化再走普通 GEMM，收益是权重占用与
  读带宽，不是算力。int8 tensor-core GEMM 会改变累加精度（累加器宽度、
  饱和、舍入），必须另立一套语义，不在这一版里。

后端侧（`device_backend.py`）没有新增内核模板：`quantize_i8` 用 cupy 的
`rint` / `clip` / `astype`，`dequantize_i8` 是一次广播乘，`linear_i8` 是
"反量化 → 既有 cublas / hipBLAS GEMM → 广播 add"。这是刻意的：量化的收益
在权重驻留（`K*N*4` 字节 → `K*N` 字节 + `N*4`），把它做成自研内核不会让
这一步更快，只会多一份要维护的源码。CUDA 与 HIP 因此同时获得这三项能力，
能力面从 57 涨到 60。

模型层（`models/quantize.py`）只做两件事：`quantize_weight` 从 `(K, N)` 权重
算逐输出通道 scale（全零通道取 1.0，避开"非正 scale"拒绝）并**直接调用算子的
reference**，保证"导出时"与"图里跑"是同一份舍入语义；`build_int8_linear`
按项目约定 `(in, out)` 构图，返回输入表与输出 Value。

**用例**（23 条，`tests/test_quantize.py`）

- `TestQuantizeSemantics`（8 条）：RNE 的六个 .5 平局点；±127.4 / ±127.6 /
  ±1000 的端点饱和；逐通道 scale 的独立作用域；`scale = 0` 与负 scale 被拒；
  `dequantize_i8` 就是广播乘（逐位相等）；f16 权重被拒；scale 秩超过权重被拒；
  符号维与静态非 1 维的广播在构图期被拒。
- `TestRoundTrip`（4 条）：量化误差不超过 `scale/2`；每个非零通道都用到 127；
  全零通道得到 `scale = 1.0` 且量化为 0；非 2 维 / 非浮点 / 含 NaN 的权重被拒。
- `TestLinearI8`（6 条）：与"显式 dequantize 再 matmul"逐位一致（含 3 维批）；
  无 bias 形态；五类 dtype / 秩 / scale 形状拒绝；内维与 bias 长度检查；
  图走 JSON 往返后输出逐位相同；运行时 `scale = 0` 报错。
- `TestLinearI8Gradient`（2 条）：`dx` / `dbias` 与手推公式一致；对量化权重与
  scale 求导抛 `NoGradientError`（不造零张量）。
- `TestQuantizedLinearBuilder`（2 条）：构建器的端到端数值在量化误差量级内、
  权重体积正好是 f32 的四分之一；in/out features 非正被拒。
- `TestConformanceCases`（1 条）：三条新 conformance 用例在参考后端全部通过。

conformance 图集同时新增 `quantize_i8_channel` / `dequantize_i8_channel` /
`linear_i8_bias` 三条。这不是顺手：`test_all_declared_operators_exercised`
要求"声明的算子全部至少成功执行一次"，能力面涨了 3 个而图集不涨，CUDA 用例
会在真机上直接判红。

**负向对照**

1. `_QUANT_MIN` 从 -127 改成 -128 → 端点饱和用例 1 条失败（-128 vs -127），
   还原后全绿。
2. 把参考实现里的非正 scale 检查短路（`if False and ...`）→ 3 条失败
   （量化拒绝面、执行期零 scale、`linear_i8` 拒绝面），还原后全绿。

**验证**：本地 2507 个用例 / 412 跳过，全绿。

**边界**

- 只有 weight-only：激活是 f32，没有 int8 累加、没有 fp8、没有 QAT、没有
  非对称（带零点）量化。
- `linear_i8` 只接受秩 2 的量化权重与 f32 的 x；f16 / bf16 激活入口未开放。
- 卷积权重没有对应的融合算子。用 `quantize_i8` + `im2col` 之外的组合可以
  表达，但 `conv2d` 本身不认识 i8，必须显式反量化——语义成立，性能不成立。
- 设备侧的非正 scale 检查是一次 `min` 归约（会同步一次）。量化是离线步骤，
  不在每步推理里，可以接受；但不要把它放进热路径。
- 本轮 CUDA 真机不可用：三个量化算子的 CUDA 路径未在 NVIDIA 卡上执行过，
  只做了能力声明与离线用例；HIP 侧见验证记录。**不声称任何 CUDA 上的数值
  或性能结论。**

### 5.240 张量并行：列 / 行切分与 attention 头切分

**问题**：一块卡放不下的权重，切开之后还能不能得到同一个结果。上一代只有数据并行
（整份参数复制到每张卡）与 ZeRO（切优化器槽位 / 参数驻留），没有"把同一个矩阵乘
按列或按行切开、各算一部分再拼起来"这条路径。

**做法**：新增 `train/tensor_parallel.py`，不新增算子、不新增内核，也不改执行器。

| 构件 | 切法 | 通信 |
|---|---|---|
| `column_parallel_linear` | 权重 `(K, N)` → `(K, N/W)`，bias 同理 | 无：各 rank 输出的是独立特征段，拼接即完整 |
| `row_parallel_linear` | 权重 `(K, N)` → `(K/W, N)` | 必须求和：返回的是**部分和** |
| `build_tensor_parallel_attention` | q/k/v 按 head 分组切列，`wo` 按 head 分组切行 | 一次求和 |
| `TensorParallelBlock` | 两段图 + 段间 `all_reduce` | 两次求和（attention / 前馈） |

**四条语义选择**

1. **行并行返回部分和，不隐式归约。** IR 里没有 collective 节点（加了它就得为每个
   后端定义设备侧语义，而参考后端没有"卡"的概念）。归约由调用方注入的
   `collective.all_reduce` 完成，与 ZeRO / 数据并行是同一个注入面。命名与注释都写着
   "部分和"：静默把它当完整输出会得到 `1/W` 的结果，而且不会报错。
2. **行并行不接受 bias。** 每个 rank 各加一次，归约之后就是 `W * bias`。构图期直接拒绝，
   由调用方在归约之后加一次。
3. **头切分就是列切分。** q/k/v 的输出维按 `head * head_dim + i` 排列，所以"沿列等切"
   与"按头分组"是同一个切片；`wo` 的行维同理。前提是 `num_heads % world_size == 0`，
   不满足就在构造期报错（`d_ff % world_size == 0` 同理）。
4. **block 里 `dropout` 必须为 0。** 掩码按形状采样，各 rank 的头宽度不同，同一 seed
   采不到一致的掩码；要训练就先关掉，或者先实现"一份掩码 broadcast 给所有 rank"。

**图外只剩两次加法。** LayerNorm、matmul、softmax、SiLU 全部在 IR 里；段间归约只能
在主机侧发生，所以 `TensorParallelBlock.run` 的顺序是：执行 `attn` 图 → `all_reduce`
→ 加残差 → 执行 `ff` 图 → `all_reduce` → 加残差。这是本层的边界，不是为省事。

**切分工具**：`shard_size` 要求整除（不补零、不丢尾——两个 rank 的图形状不同在静态 IR
里不可表达）；`split_axis` 返回**副本**（参数会被优化器原地更新，共享视图会让一个 rank
的更新悄悄改到另一份上）；`merge_axis` 校验非切分维一致。`split_dense` 按完整形状校验
缺失 / 多余 / 形状不符三类错误，复制类参数（LayerNorm 的 `g` / `b`、`scale`）必须拷贝。

**判据**：world = 1 / 2 / 4 的 block 与 `build_transformer_block` 在 `1e-5` 内一致
（Loopback 按 rank 顺序累加部分和，与 BLAS 的求和顺序不同，因此不写"逐位"）；
`fused_attention` 变体同样对齐；两段图各自过 verifier；头切分单独与手算的完整头
attention 对拍；本地参数形状按 `W` 缩小、复制类参数不变，总字节数小于整份。

**负向对照**

1. 在 `run` 里让 attention 段跳过 `all_reduce`（直接用部分和）→ world=2 / 4 与 fused
   共 4 条失败，还原后全绿。
2. 把 `wo` 的切分轴从 0 改成 1 → `split_dense` 的形状校验与全部 block 用例失败，
   还原后全绿。

**验证**：本地 2532 个用例 / 412 跳过，全绿；WSL HIP 2532 / 371 全绿（本层不碰设备，
HIP 只用来确认没有平台相关的导入问题）。

**边界**

- **只做前向。** 反向的集合通信（行并行输入梯度求和、列并行权重梯度按列拼接）与
  优化器状态分片没有接入；不要把它当成可训练的模型并行。训练侧现有的并行手段仍是
  数据并行、流水与 ZeRO-1/2/3。
- 没有接 NCCL 的**设备侧**归约：`collective` 由调用方注入，当前可用的实现是
  `LoopbackWorld`（进程内线程）与 `NcclHostCollective`（主机缓冲）。归约走的是主机
  内存，`overlap` 仍恒为 `False`。
- 段间的两次加法在主机侧，因此每段都要把部分和拉回主机一次；这不是性能路径，
  是"切分后语义仍然正确"的证明路径。
- 序列维与 batch 仍是符号维，但两张图必须由同一个 `spec` 构建；混用不同 spec 的
  参数形状会被 `split_dense` 拦下。



### 5.241 分页 KV cache 与投机解码

**问题**：两件相关但不同的事。一是"多条不等长序列共享一块显存里的 K/V"——按
`max_len` 给每条请求预分配连续缓冲，短请求会浪费、长请求会挤掉别人；二是
"一次前向只出一个 token"——自回归解码的每步都是一次形状相同的矩阵乘，
算力远没喂满。

**做法**：三个新文件，不新增算子、不改执行器。

| 文件 | 内容 |
|---|---|
| `runtime/paged_cache.py` | `PagedKVCache`：块表 + 物理页池，序列只记"页号列表"与长度 |
| `models/cached_attention.py` | 单查询增量图 + `cached_forward` 驱动 |
| `models/speculative.py` | `SpeculativeDecoder`：草稿 → 验收 → 提交 |

分页缓存的语义面：`allocate` / `free`（幂等）/ `reset` / `truncate`（只能缩短）/
`append` / `gather`，外加只读的 `capacity_tokens`、`used_blocks`、`free_blocks`、
`peak_blocks`、`block_table`、`length`、`stats`。所有层共享**一张**块表——
同一序列的各层页号必须逐项相等，否则 `gather` 会把不同层的 K/V 拼到一起。

增量解码的关键是"图里怎么表达缓存"：`build_cached_decoder_step` 把已缓存的
`(B, T, H, D)` 转成 `(B, H, T, D)` 后与当前 token 的 k/v `concat`，再做一次
单查询注意力。掩码是加性的 `(1, 1, 1, T+1)`：前 `valid` 个缓存位与末尾当前位
为 0，其余（补零位）为 `-1e9`。判据是"与 `build_transformer` 在同样权重下逐位置
一致"，不是"看起来像增量解码"。

投机解码的顺序固定为：草稿给出 `k` 个 token → 目标模型一次前向验证这 `k` 个 →
逐位比较 `argmax` 接受前缀 → 从第一个不一致处取目标的预测作为 bonus。

**四条语义选择**

1. **追加是全有或全无。** `append` 先算新长度需要几页，页不够就整体抛
   `PagedCacheFullError` 且**不改任何状态**——半写进去的 K/V 会让下一个人
   拿到"长度对但是内容错"的缓存，这种错误在任何显存检查里都看不出来。
2. **回滚以页为粒度。** `truncate(rid, length)` 只释放"整页都不再需要"的尾页，
   页内剩余位不清零——长度是唯一的真相，`gather` 永远只取 `pad_to` 以内的部分。
3. **验收被拒之后必须回滚状态。** 语言模型主干是 Mamba，状态是递归的、
   不能按 token 截断；接受 `a < k` 个草稿 token 时，只能回到验证前快照再
   把"接受的前缀 + bonus"重新喂一遍。这就是 `_snapshot` / `base_state` 存在的理由。
4. **空草稿不是错误，是弃权。** 目标模型下一步就是停止符时，草稿合法地返回空；
   此时回退到目标的贪心 token。把它当异常会让"完美草稿"这条最该省事的路径失败。

**判据**

- 分页缓存：跨页追加后 `gather` 与一次性写入逐位一致；各层共享块表；
  回收的页被新序列复用时不串数据；`free` 幂等；容量失败后 `stats` 原样；
  `pad_to` 右侧补零且不改已写部分。
- 增量注意力：`steps` ∈ {1,3,5} × `block_size` ∈ {1,2,3,4,8} 全组合下，
  `cached_forward` 与稠密 `build_transformer` 输出在 `1e-6` 内一致；
  `fused_attention` 变体同样对齐。
- 投机解码：四种草稿（完美 / 恒定 0 / 末 token 值 / 重复末 token）与
  空草稿弃权下，输出与 `generate(temperature=0)` **逐位**一致；
  完美草稿下 `accepted == proposed`，且轮数 < 5。

**负向对照**

1. 拒绝分支回滚成 `base_state = verified_state`（相信"被拒的那一步没写进状态"）
   → `tests.test_speculative` **7 条失败**，还原后全绿。
2. `_pages_for` 改成 `(tokens // block_size) + 1`（每段都多算一页）
   → `test_paged_cache` + `test_attention_cache` **2 failures + 7 errors**。
3. `_causal_step_mask` 全置 0（不屏蔽补零位）→ `test_attention_cache` **17 条失败**。

**验证**：本地 2561 个用例 / 412 跳过，全绿；WSL HIP 2561 / 371 全绿
（本层不碰设备，HIP 只用来确认没有平台相关的导入问题）。

**边界**

- **主干是 Mamba，不是 Transformer。** 递归状态与已处理 token 数无关
  （5.102），所以"分页 KV cache"在这里是独立的 runtime 组件 + 一个真实的
  注意力消费者，**没有接进 `generate` / `LMRunner`**：那条路径没有会增长的 KV 可管。
- 增量图只做推理（`dropout=0`）、只吃隐状态（`vocab=0`，位置编码由调用方加）、
  `cached_forward` 只支持 `batch=1` 的静态 batch。
- 投机解码只做贪心验收（`temperature=0`）。采样验收需要
  "拒绝时按修正分布重采样"，那是另一套随机数与另一条判据，本轮没做。
- `greedy_draft` 是**正确性参照**，不是性能路径：每轮都从 history 重新 prefill，
  比直接解码更贵。它的价值是"任意草稿都不改变输出"这条性质有可执行的对照物。
- 不声称任何吞吐收益。没有真机对照数据（CUDA 机已释放），也没有把草稿模型
  做成"小模型跑在设备上"。

### 5.242 成本模型与组合搜索：先算再测，不可行的先剔掉

**问题**

调优面到 5.241 时，conv2d 的三个旋钮（`tile_w` / `roc` / `cb`）合起来是
4 × 4 × 5 = 80 个组合，其中 5 个在**计划层**就跑不起来（`tile_w=64` 被截到
`wo` 之后仍可能让 `tile_w * block_y` 超过 1024）。5.144 之前靠 `tune_staged`
把 45 个候选拆成 4 + 11 + 2 来压单轮跨度，但那是"分段的实测"，回答不了
"这 80 个里哪些根本不必测"。真机上每测一个候选都要付一次同步与计时开销，
而不可行的候选只在 `launch` 前才炸——报错类型（`invalid configuration`、
`roc 必须整除`）与"慢"完全不是一回事，混进排名就会把"没有可行配置"
退化成"有一个很慢的配置"。

**做法**

新增 `aetherflow/tuning/cost.py`，全部纯函数、不 import 后端：

- `MachineModel`：成本模型需要的机器常数（每周期 FMA 数、每周期字节数、
  同步延时、每 SM 的线程 / block / shared 上限）。**这些是量级假设，不是
  测量结论**，只回答方向性问题（"哪个的访存量少一半"），不预测毫秒。
  两个预设 `M40_LIKE` / `V100_LIKE`。
- `conv_tile_cost(...)` / `elementwise_cost(...)`：`config -> CostBreakdown`。
  **派生量全部取自 `lower/plan.py`**（`plan_conv_tiled` / `auto_block`）：
  block 线程数、shared 字节、输出瓦片数不在这里第二次推导——两处各推一遍
  必然漂，而漂的后果是"估算说可行、launch 说不行"。计划层抛的 `ValueError`
  原样变成 `feasible=False` + `reason`，不概括、不改写。
- `CostBreakdown`：六项耗时（`arith` / `read` / `staging` / `sync` / `tail` /
  `dead`）+ `dominant`（最大项）+ `explain()`（一句话结论）。
- `rank_configs(space, cost_fn)`：可行项按总耗时升序、不可行项殿后，
  **排序稳定**（同分保持 `space.configs()` 原顺序），不可行项不丢弃。
- `cost_ranked_tune(space, cost_fn, measure, max_evals=...)`：按估算排序后只测
  前 N 个可行候选。**默认配置永远排第一**（否则 `tie_epsilon` 以内的平局会
  判给"只是估算更低"的候选，`speedup` 的基准也随之漂走）；`max_evals=None`
  时可行项全测，结论与 `tune` 一致。
- `search.py` 把测量与判定抽成 `measure_configs(configs, measure, ...)`，
  `tune` 变成它的薄封装。抽这一层的理由是可测性：裁剪后的候选集通常**不是
  笛卡尔积**（例如 80 个里按估算取前 10），塞回 `TuningSpace` 会展开出没被
  选中的组合，于是只能靠一个收"已展开配置"的入口。

六项成本的算法（`dtype_bytes` 按 f32 = 4）：

| 项 | 公式 | 随旋钮的方向 |
|---|---|---|
| `arith` | `n·ho·wo·o·kh·kw·c` 次 FMA ÷ 每周期 FMA | 与旋钮无关 |
| `read` | `arith 的输入量 / roc` ÷ 每周期字节 | `roc` 越大越小 |
| `staging` | `块数 · ceil(c/cb) · toc·kh·kw` ÷ 每周期字节 | `cb` 越大越小 |
| `sync` | `波次 · ceil(c/cb) · 2 · 同步周期` | `cb` 越大越小 |
| `tail` | 最后一波空槽比例 × 基础耗时 | 块数能否整除驻留容量 |
| `dead` | 最后一个 x 瓦片的空线程比例 × 基础耗时 | `tile_w` 与 `wo` 的余数 |

`sync` 按**波次**而不是按块数折算：栅栏在 block 内串行、同一波内的 block
并行。第一版按块数算，在 80 个组合里让 `sync` 无差别地盖过算术与读
（1×64×56×56 的卷积上 896 us vs 算术 75 us），排序退化成"谁的块少谁赢"；
改成按波次之后同一形状上是 16 us，`read` 才浮上来成为主导项。

`tail` / `dead` 是**占空比**而不是绝对延迟：`total_us` 的含义是"按满占用
折算的耗时"，机器填不满时的浪费记在 `tail` 里，不是"这个 kernel 要跑这么久"。
这一条在文档与 `CostBreakdown` 的注释里都写明，免得被当成性能预测。

**示例（离线，M40_LIKE，1×64×56×56，3×3，o=64）**

同一份 80 个组合的排序：75 个可行、5 个不可行（`tile_w=64` 截到 56 后
× `block_y=32` = 1792 > 1024）。估最低的是默认（自动推导）575.6 us
（`read=385.4` / `arith=75.3` / `dead=57.6` / `tail=34.5` / `staging=6.9` /
`sync=16.0`）；最差可行项是 `tile_w=16, roc=1, cb=1` 2395.1 us（4.2×），
`dead` 从 57.6 涨到 202.1、`read` 从 385.4 涨到 1541.4——估算给出的
"为什么不该这么配"是可核对的，不是一句"实测更慢"。
`cb=None` 与 `cb=8` 同分，因为该形状下计划层推导出的默认 `cb` 本来就是 8：
估算没有凭空造出差异，这一点比"排序好看"更重要。

**判据（41 条新用例，`tests/test_cost_model.py`，全部纯 Python）**

- 计划一致性（2 条）：`threads` / `shared_bytes` / `oc_tiles` / 块数逐项等于
  `plan_conv_tiled` 的输出；六项的顺序固定为 `arith, read, staging, sync,
  tail, dead`。
- 方向性（8 条）：`cb` 变大时 `staging` / `sync` 下降而 `arith` / `read` 不变；
  `roc` 从 1 到 4 时 `read` 恰好降到 1/4；`tile_w` 超过 `wo` 被截断后两个配置
  的 `terms` 与 `detail` 逐项相等；`wo=20` 时 `tile_w=8` 的 `dead` 小于
  `tile_w=16` 且块数更多；`roc=3`（不整除 OC tile）不可行且 `total_us == inf`、
  `dominant` 为空；`tile_w=1024 × block_y=8` 撞 block 上限；`cb=64` 撑爆
  shared 预算；未知参数名不可行。
- 拒绝面与快照（5 条）：非法形状 / 非 `MachineModel` 在构造期拒绝；`explain()`
  写出主导项；不可行的 `explain()` / `to_dict` 保留原始原因；缺省的旋钮等价于
  `None`（自动推导）；`CostBreakdown` 是冻结快照。
- 逐元素（5 条）：`block=None` 与计划层 `auto_block` 一致；`vec=4` 的块数是
  `vec=1` 的 1/4 而读写字节不变；`n=8192`（整除）的 `dead` 为 0、`n=8000`
  （不整除）的 `dead` 为正；非 2 的幂的 block 不可行；非法 `vec` / 负元素数
  在构造期拒绝。
- 排序（4 条）：覆盖数等于空间大小；可行在不可行之前且两次调用逐项相同；
  全不可行时保持空间原顺序；类型错误显式拒绝。
- 组合搜索（9 条）：`max_evals` 只测前 N 个且默认配置排第一；测量顺序等于
  "默认 + 估算升序"；不裁剪时与 `tune` 在平局下给出同一个赢家；实测更快的
  非默认配置仍然赢（估算只决定测谁）；测量异常记为不可行但搜索继续；
  全不可行抛 `AllRejectedError`；非法 `max_evals`（0 / -1 / 1.5 / True）拒绝；
  `to_dict` 同时留下估算与实测；逐元素空间同样能跑通。
- `measure_configs`（4 条）：空列表拒绝；平局保留第一个；显著更快才替换
  （`speedup == 2.0`）；`None` / `NaN` 全不可行时抛 `AllRejectedError`。
- `MachineModel`（4 条）：默认值为正、非法字段拒绝、`occupancy` 同时受
  线程与 shared 约束、线程数为 0 拒绝。

**负向对照**

1. `read` 改成 `macs`（忽略 `roc` 的输入复用）→ `test_cost_model`
   **1 条失败**（`roc` 从 1 到 4 的 `read` 不再是 1/4）。
2. `CostBreakdown.total_us` 去掉"不可行返回 `inf`"→ **2 条失败**：排序把不可行
   项排到了前面，`total_us` 也不再是 `inf`。
3. `cost_ranked_tune` 去掉"默认配置提到最前"→ **3 条失败**：`seen[0]` 变成
   `roc=2` 的自动组合，`tune` 平局对照与 `to_dict` 的赢家一起变。

**验证**：本地 `python -m unittest discover -s tests -t .` → 2602 个用例 /
412 跳过；本层 41 条全绿（另有一条 `test_conformance` 的算子覆盖用例失败，
来自工作区里另一份当时未提交的 `cumsum` / `tril` 工作（已在 5.247 提交），与本层无关）。
WSL HIP 侧不跑设备路径，只用来确认没有平台相关的导入问题。

**边界**

- **不是性能预测器**：机器常数是量级假设，"估算 575 us"不承诺任何墙钟数字。
  真机分胜负仍然必须实测（`tune` / `tune_paired` / `tune_replicated`）。
- **没有接进 `benchmarks/tune.py` 的默认路径**：那条路径的定义是"每轮连测
  全部候选"，裁剪候选集会让跨轮的可比性消失。`cost_ranked_tune` 是串行测量
  + 预算裁剪的入口，适合"候选多到测不完"的场景，两者不互相替代。
- 只覆盖两类旋钮：conv2d 的 `tile_w` / `roc` / `cb` 与逐元素类的 `block`。
  归约类的 block 没有成本函数（它的耗时由列数与归约树决定，常数量级不同），
  因此 `cost_ranked_tune` 不接受"归约类的空间"——传进去会得到一条
  "不认识调优参数"的不可行，而不是一个假装正确的排序。
- `tail` / `dead` 用的是"槽位比例 × 基础耗时"的一阶近似，不建模 L2 命中、
  合并访存与 bank conflict；`tile_w` 与 `wo` 的余数只影响 `dead`，不影响
  `read` 的实际合并程度。

### 5.243 Chrome Trace 导出：把已有的观测换成能看的时间轴

**问题。** 5.161 起每次执行都留下了逐节点的 `dispatch_ms` 与（有 `gpu_clock`
时）`gpu_ms`，`RunRecord` 也能存能读，但两件事回答不了：一是"整图 12ms 里，
这些 0.1ms 的节点是紧挨着、还是中间空着 3ms 在等设备"；二是"我该先优化哪个
算子"。前者缺的是**时刻**——只有时长画不出时间轴，按顺序首尾相接排出来的图
会把所有空闲时间抹平，看起来永远是满的；后者缺的是一个按时间摆放的视图。
浏览器能读的 Chrome Trace Event Format（chrome://tracing / Perfetto）正好是
那个视图，而它是"换一种表示法"，不是新的观测。

**做法。**

- `NodeTiming` 增加可选字段 `start_ms`：包装器在 `invoke` 之前用
  `perf_counter` 记下相对本次 `execute` 起点的时刻，零点与 `wall_ms` 相同。
  因此图上的空白是真实的"CPU 没有在派发"（等设备、Python 里算形状、同步），
  不是画图造出来的。旧记录没有这个字段时按顺序累加补齐，补齐条数写进
  `otherData.synthesized_nodes`——**"补的"和"测的"必须在文件里分得开**。
- `runtime/trace.py` 三个入口：`trace_events(records)` 展开成 `(事件, 附注)`、
  `build_trace(records)` 返回可直接 `json.dump` 的顶层结构、`write_trace(path, …)`
  落盘（临时文件 + `os.replace`，与 `TelemetryRecorder.save` 同一套写法：
  trace 常在 CI 里边生成边被 artifact 上传，半个 JSON 比没有更糟）。
- 一份记录一个 `pid`，进程名取 `label or graph` 加后端（有设备再拼设备名），
  多条记录按 `wall_ms` 顺序往后排：不同时刻的两次 `execute` 不能叠加在一条
  时间轴上。线程 1 是"CPU 派发"（实测），线程 2 是"GPU 内核（累计占位）"。
- **GPU 轨道是累计占位，不是真实时刻。** 事件对只给逐节点时长、不给绝对
  时刻（5.161 的限制），所以这里把它们首尾相接排开，线程名直接写进
  "累计占位"，每个事件的 `args.approximate` 也是 `True`。它能读占比与顺序，
  不能读"哪个 kernel 和 CPU 的哪一段重叠"。要真实时刻得让后端暴露事件时间戳，
  那是另一件事。
- 命令行入口：`python -m aetherflow.runtime.trace telemetry.json [-o trace.json]`，
  输入是 `TelemetryRecorder.save()` 的产物，默认输出 `<输入>.trace.json`。

**判据（20 条，全部纯 Python）**

- CPU 时间轴（4 条）：`ts` / `dur` 等于 `start_ms` / `dispatch_ms` 换算成微秒；
  缺时刻时按"上一个节点的结束"累加；混合记录只补缺的行、实测行保持原值；
  事件名与 `args` 能定位到具体输出。
- GPU 轨道（3 条）：从第一个有读数的节点起首尾相接、时长等于 `gpu_ms`、
  `approximate` 为真；没有读数时**不出现** GPU 事件；两条轨道的名字都写进
  元数据。
- 记录与布局（4 条）：多条记录 `pid` 递增且按 `wall_ms` 错开；设备名进标题；
  空记录仍是合法 trace；同一份记录导出两次逐字节相同。
- 参数校验（4 条）：非 `RunRecord` 抛 `TypeError`；负 `offset_ms`、非法
  `start_pid`（0 / -1 / 1.5 / `True`）、负 `start_ms` 各自显式拒绝。
- 序列化与文件（4 条）：`start_ms` 能 JSON 往返、缺省时不出现在 JSON 里；
  `write_trace` 不留临时文件且落盘内容等于返回值；CLI 能把 `save()` 的产物
  转成 trace（补齐数为 0）。
- 包装器时间轴（1 条）：真跑一个小图，每个节点都有 `start_ms`、非负、
  单调不减且不超过 `wall_ms`。

**负向对照**

1. `_cpu_starts` 忽略实测 `start_ms`、一律累加 → **5 条失败**（实测时刻、
   混合记录、GPU 起点、CLI 补齐计数、包装器时间轴一起变）。
2. `ts` 不乘 1000（毫秒当微秒写）→ **5 条失败**：时间轴上所有位置缩小 1000 倍。
3. GPU 轨道从 0 起而不是从首个读数的实测起点起 → **1 条失败**
   （`test_gpu_events_are_sequential_from_first_reading`）。

**验证**：本地 `python -m unittest discover -s tests -t .` → 2622 个用例 /
412 跳过，本层 20 条全绿；WSL HIP → 2622 个用例 / 371 跳过。两处唯一的失败
都还是 `test_conformance` 的算子覆盖断言（`cumsum` / `tril`），来自工作区里
另一份未提交的工作，与本层无关。

**边界**

- **不是采样 profiler**：没有调用栈采样、没有符号化、没有多线程聚合。
  它只把已有观测换一种表示法；想知道"Python 里哪一行慢"要用 `cProfile`，
  想知道"kernel 内部哪里慢"要用 Nsight / rocprof。
- **只覆盖派发时间轴**：异步派发时一个节点的 `dispatch_ms` 只是"把活发出去"
  的耗时，图上会看到一串极短条与大片空白——那正是"Python 是瓶颈"的证据，
  不是画错了。
- **GPU 轨道没有绝对时刻**（见上）；要读重叠关系必须先让后端给出事件时间戳。
- **没有接进 `benchmarks/run.py` 的默认路径**：导出是显式动作（先
  `TelemetryRecorder` 采集，再 `write_trace` 或 CLI），避免每次基准都多写
  一个大文件。

### 5.244 RCCL 绑定：同一份 ctypes 声明，按库文件名分辨厂商

**问题**

5.228 把 NCCL 的四个集合操作绑到了 CUDA 上，能力声明里写的却是"NCCL / RCCL
未接入"。HIP 侧一直缺这一半：`backends.hip` 能算，但跨卡归约没有任何设备侧
入口，`collective_support` 对 HIP 后端也只能回一句"没有"。RCCL 并不是另一个
API——它把 `nccl*` 的函数名、枚举值与 `ncclUniqueId` 的 128 字节布局都实现了
一遍；真正需要处理的差异只有"怎么找到库"与"怎么同步流"。

**做法**

- **一份签名表**：`_bind()` 不动，加载成功的是什么库就用什么库。RCCL 的
  soname 是 `librccl.so.1`（NCCL 是 `libnccl.so.2`），候选表两组都列；
  `find_library("nccl")` 与 `find_library("rccl")` 的结果放最前，且保持 NCCL
  优先——一台机器同时装了两套时按候选顺序的第一份当事实来源。
- **厂商判定是纯函数**：`_vendor_from_path()` 只看 basename 是否含 `rccl`，
  不看能不能 `import cupy`。理由很直接：设备上装了什么库与 Python 侧装了哪个
  包是两件独立的事，用后者推断前者会在"有 ROCm 但没装 cupy"的机器上判错。
- **公开三个查询**：`vendor()`（`"cuda"` / `"hip"`）、`version()`（整数）、
  `version_string()`（`"2.27.7"`）。库缺失时三者都抛 `NcclUnavailable`，
  不返回 `"unknown"` 或 `0`——调用方需要的是"能不能用"，不是"猜一个默认值"。
- **同步按厂商分叉**：`NcclComm.synchronize()` 在 HIP 上走
  `backends.hip.runtime.synchronize()`（设备级），在 CUDA 上走 cupy 的当前流。
  不做 try/except 双路尝试：那会把"同步失败"降级成"同步被跳过"。
- **错误信息带厂商名**：`_check()` 用 `_label()` 取 `RCCL` 或 `NCCL`，
  与真实加载到的库一致；加载失败时同时提到两者。
- **HIP 数组直接可用**：`DeviceArray` 已有 `data.ptr` / `size` / `dtype` /
  `flags`，正好是绑定层要求的四项，不需要为 RCCL 写数组适配层。
  `tests/_rccl_worker.py` 用它跑与 CUDA 版相同的四个集合操作。
- **能力声明保持 `device_allreduce=False`**，但理由文本改为读实际状态：
  库缺失时说"本机也加载不到"，库在时说"RCCL x.y.z 已可加载并已绑定调用面，
  但数据并行路径还没接它"。这两句话对应两种不同的下一步。

**判据**（11 条）

- 厂商判定（3）：`librccl.so.2` / `librccl.so.1` / 绝对路径 / Windows 的
  `rccl.dll` 都判 HIP；`libnccl.so.2` / `nccl.dll` 判 CUDA；候选表同时包含
  两套 soname。
- 版本与可用性（3）：`_format_version(22707) == "2.27.7"`、`21803` 与
  `10000` 两个边界；库缺失时 `vendor()` 与 `version_string()` 抛
  `NcclUnavailable`；库在时 `vendor()` 属于 `{cuda, hip}` 且版本串是三段。
- HIP 数组形态（1）：只提供 `data.ptr` / `size` / `dtype` / `flags` 的最小
  对象能通过 `_pointer` 与 `_contiguous`。
- 能力理由（3）：`vendor() == "hip"` 时理由含 `RCCL 2.27.7` 与"主机中转"；
  库缺失时含"加载不到"；`vendor() == "cuda"` 时含 `NCCL 2.18.3`。
- 真机（1）：`TestRcclOnDevice` 在 WSL HIP 上以 world=1 启动
  `tests/_rccl_worker.py`，跑通 unique id、comm 初始化与四个集合操作
  （world=1 时期望值等于输入）。
- 既有判据的收紧（0 条新增，2 条改写）：`tests/test_parallel.py` 的两条能力
  断言原本写着 `assertIn("NCCL", reason)`，理由是 5.228 年代理由文本固定叫
  NCCL；5.244 起理由按本机真实加载到的库命名，这两条在 HIP 机器上必红（WSL
  全量首跑就是它们变红，不是数值错）。改写为 `assertRegex(reason,
  "NCCL|RCCL")`——判据要求的是"点名厂商"，不是"点名某一家"。

**负向对照**

1. `_vendor_from_path` 一律返回 `"cuda"` → **1 条失败**
   （`test_rccl_names_are_hip`），HIP 会被当成 CUDA，`synchronize()` 走错分支。
2. 能力理由把 `label` 写死成 `"NCCL"` → **1 条失败**
   （`test_reason_names_the_loaded_library`）。
3. `synchronize()` 的 HIP 分支改成 `if False:`（退回 cupy）→ WSL 上
   `TestRcclOnDevice` **1 条失败**，worker 以 `ModuleNotFoundError: cupy` 退出。
   这条是最有价值的一组：它证明真机用例确实走到了厂商分叉，而不是被跳过。

**验证**：本地 `python -m unittest discover -s tests -t .` → 2633 个用例 /
414 跳过，本层离线 10 条全绿；WSL HIP → 2633 个用例 / 373 跳过，
`TestRcclOnDevice` 真跑通过（RCCL 2.27.7，单卡 world=1）。WSL 全量首跑有 3 条
失败：两条是上面那对写死 `NCCL` 的旧判据（改为接受两家后重跑
`tests.test_parallel` + `tests.test_nccl` 55 条全绿 / 5 跳过），剩下一条仍是
`test_conformance` 的算子覆盖断言（`cumsum` / `tril`），来自工作区里另一份当时
未提交的工作（已在 5.247 提交），与本层无关。WSL 上运行前必须带 `HSA_ENABLE_DXG_DETECTION=1`，
否则 RCCL 在 `ncclGetUniqueId` 处直接 FATAL（`no ROCm-capable device is
detected`）。

**边界**

- **没有接进数据并行路径**：`DeviceDataParallel` 的归约仍是每卡拉回主机求和，
  能力声明据此保持 `device_allreduce=False`。本层交付的是"调用面可用"，
  不是"训练已经用上设备侧通信"。
- **真机只到 world=1**：WSL 只有一块 AMD 卡，跨卡数值（ring / tree 的归约顺序、
  AllGather 的拼接顺序）没有被本层验证；那部分仍只有 CUDA 2×V100 的结论。
- **不支持 bf16**（numpy 没有对应类型，算不出主机侧期望值）、不做变长分片、
  不提供单进程多 communicator；这些限制与 5.228 相同。
- **不提供 RCCL 专属旋钮**：`NCCL_*` / `RCCL_*` 环境变量原样交给库，本层不解释。

### 5.245 设备端循环：条件与体各捕获一次，状态常驻显存

**问题**：`WhileLoop.run`（5.201）每轮把携带量 D2H 回主机、在主机上拼 `feeds`、
再交给 `execute` H2D 上去。数值是对的，但显存里的状态从来没有真正"驻留"过，
循环体里的内核也每轮重新派发一次。对比主流框架的差距清单里，这一条写作
"设备端控制流"。

**做法**（`runtime/device_control.py`，`DeviceWhileLoop`）

- 条件图与循环体各 `capture` 一次，**同一条流**；循环跑多少轮，捕获次数都是 2
  （`captures` 属性把这件事变成可判的数字）。
- 携带量不再回主机：体 `replay` 的输出直接当下一轮的输入，由计划在自己的流上
  做一次 D2D 拷进静态缓冲。计划不重建、不重捕获。
- 体的 `replay` 是**异步**的，不额外同步：它和条件排在同一条流上，下一轮条件的
  `run_device` 的同步自然覆盖它。因此每轮只有一次同步、一次主机往返（谓词标量）。
- 自由输入只在建计划时上传一次；`run(feeds)` 给了新值才重传（`to_array` 一次），
  状态复位是"指针回到初值缓冲"，不产生上传。
- 每次 `run` 都从初值重新开始，可重复调用；`run_device` 返回仍在显存里的状态，
  `run` 只在收尾时对每个状态做一次 D2H。
- 后端前置能力与设备态训练同源：`capture` / `new_stream` / `to_array` / `to_host`；
  计划还必须提供 `replay` 与 `device_outputs`（异步面）。缺任何一项都在构造期
  显式拒绝，不做"偷偷退回每轮两次同步"的降级。
- 放在 `runtime` 而不是 `ir/control.py`：控制流是 IR 语义（IR 不认识 `capture`），
  设备形态是运行期能力，与 `train/loop.py` 的设备态训练同一层。

**判据**（19 条，其中 1 条真机）

- 数值（3）：宿主 `WhileLoop.run` 与设备路径逐位相等（两种步长）；`run_device`
  返回的状态可直接读回；重复 `run` 每次都从初值跑同样的轮数（只比数值抓不到
  "接着上次的终值立刻退出"，所以同时比 `iterations`）。
- 捕获复用（4）：两张图各捕获一次且跑多少轮都不再捕获；两次捕获用的是同一条
  流；每轮同步数 = 轮数 + 1（含退出那次条件）、`to_host` 次数 = 轮数 + 2；
  状态在两次 `run` 之间一次上传都不发生，换自由输入才上传一次。
- 契约（9）：嵌套 `Conditional` 作条件时抛 `DeviceLoopError`；后端缺 `capture`
  在构造期抛 `TypeError`；缺初始状态抛 `ValueError`；缺自由输入、多给名字抛
  `KeyError`；`max_steps` 用尽抛 `ControlFlowError`（消息含 `max_steps`）；
  换形状、换 dtype 抛 `ValueError`；非 `WhileLoop` 抛 `TypeError`。
- 真机（1）：WSL HIP 上 5 轮、2 次捕获、6 次同步，与参考后端逐位一致。

**负向对照**（3 组，全部先变红再还原）

1. 体从异步 `replay` 改成同步 `run_device` → 同步计数判据变红（每轮多一次）。
2. 体改用另一条流捕获 → "两次捕获同一条流"变红（时序前提被破坏）。
3. 嵌套区域的 `DeviceLoopError` 改成 `ValueError` → 该条变红（异常类型是契约）。

**验证**：本地 → 2652 个用例 / 415 跳过；WSL HIP（RX 7650 GRE / ROCm 7.2）→
2652 个用例 / 373 跳过，真机那条通过。两处唯一失败仍是 `test_conformance` 的
算子覆盖断言（`cumsum` / `tril`），来自工作区里另一份当时未提交的工作
（已在 5.247 提交），与本层无关。

**耗时（负结果，如实记）**：WSL HIP，64 轮，每轮对 1 个元素做一次加法。
状态 4 KB 时 host 8.6 ms / device 37 ms；状态 4 MB 时 host 26 ms / device
44–46 ms。原因不是同步次数（两边每轮都是一次判定），而是 HIP 上捕获图 replay
的流同步比"逐节点派发 + 默认流同步"更贵。**本层交付的是执行形态（状态驻留、
捕获复用、可判的同步数），不是加速开关**；要真正更快，需要 device-side 循环
或 Graph 条件节点，那不在第一代范围。

**边界**

- **谓词每轮一次 D2H 同步**：数据相关分支绕不过；没有 device-side 循环，也没有
  CUDA Graph 的条件节点，本层不假装零同步。
- **只支持条件与体都是单张 `Graph`**：嵌套控制区域没有可捕获的 launch 序列，
  明确拒绝并指回 `WhileLoop.run`。
- 分支形态见 5.277（`DeviceConditional`）；模块随它一起更名为
  `runtime/device_control.py`。
- **换形状要新建计划**：一张捕获图只对应一组固定形状/dtype，复用会被显式拒绝。
- **CUDA 真机本轮未验证**（那台机器已释放）；HIP 已验证，CUDA 走的是同一组
  `capture` / `replay` / `device_outputs` 接口。
- **未接进训练/推理主路径**：`Trainer` 的 `device_state` 与 `LMRunner` 都不用它，
  本层是独立可用的执行形态。

### 5.246 通道在后（NHWC）布局传播与后端布局声明

**问题**：对比主流框架的差距清单里，"布局优化"写作一条：框架应当能识别一段
连续的可翻转区域，整段换成通道后在后（NHWC），只在区域边界付两次转置。此前
AetherFlow 是"全 NCHW、无布局传播"——IR 里根本没有 `layout` 这个位，后端也没
有地方声明自己承接哪种布局。差距不只是少一个优化：**"后端承接什么"这件事没有
表达位**，能力协商在布局这一维上是空白的。

**做法**

1. **IR 先有语义位**（`ir/ops.py`）：`conv2d` / `maxpool2d` / `avgpool2d` 增加
   可选 `layout` 属性（缺省 `"nchw"`，允许 `"nhwc"`），进 `attrs_schema`。
   NHWC 参考语义**不复制数学**：把输入从 `(N,H,W,C)` 转成 `(N,C,H,W)` 交给原来
   的 NCHW 参考实现，输出再转回；infer 同理（临时换成 NCHW 推导，输出形状换回）。
   因此两条路径**逐位相同**。权重与 bias 不随激活布局翻转（`(O,C/groups,kh,kw)`
   不变），这与 cuDNN / TensorFlow 的惯例一致。非法 `layout` 在 infer 期拒绝。
2. **反向显式拒绝**（`ir/grad_rules.py`）：三个算子的 VJP 开头检查 `layout`，
   非 NCHW 抛 `NoGradientError`。反向算子（`conv2d_input_grad` 等）的语义与
   内核都是 NCHW 的，若把 NHWC 形状原样喂进去，梯度会沿错误的轴归约——
   看着有值但全错。布局传播是**推理期**优化，训练图保持 NCHW。
3. **后端布局声明**（`backends/base.py`）：`DEFAULT_SUPPORTED_LAYOUTS = {"nchw"}`
   与 `supported_layouts_of(backend)`——没声明的后端按**最小承诺**处理，而不是
   "默认什么都行"。参考后端声明 `{"nchw","nhwc"}`；`DeviceBackend` 声明
   `{"nchw"}`，并在 `invoke` 入口与计划快路径（`_planned_kernel`）两处显式拒绝
   NHWC。两处都要有：只拦 `invoke` 的话，图捕获 / 批量派发的快路径会绕过它。
4. **传播 pass**（`passes/layout.py`，`LayoutPropagation`）：
   - 可翻转算子 = `{conv2d, maxpool2d, avgpool2d}` ∪ 形状保持的一元逐元素算子
     （relu / sigmoid / tanh / gelu / silu(1 输入) / exp / log / sqrt / neg /
     erf / step / spike），且输出类型必须与输入**完全相同**（形状 + dtype +
     秩 4）。
   - 按拓扑求可翻转节点的极大连通区域（union-find），区域里至少有一个 conv /
     pool 才翻——纯逐元素区域翻过去只是白付两次转置。
   - **只有激活参与连通**：卷积的第 0 个输入是激活，weight / bias 是参数。
     若权重来自可翻转节点，先把该卷积踢出候选集（循环到稳定），否则权重会被
     当成 NHWC 激活去转置——语义直接错。
   - 重写用 `GraphRewriter`，但**不能走 `emit`**：翻过的节点类型变了
     （`emit` 会断言类型不变）。区域入口插 `transpose(perm=(0,2,3,1))`，出口插
     `transpose(perm=(0,3,1,2))`；同一来源只转一次（入口缓存）；区域输出同时被
     区域内与区域外消费时，只插一个出口转置。图输出若在区域内，必须转回 NCHW。
   - 目标后端声明里没有 `nhwc` 就抛 `LayoutError`（带声明内容与区域数），
     不静默跳过、也不"翻完再让执行期报错"。
5. 只给 conv / pool 写 `layout` 属性：逐元素算子的语义与维度顺序无关，给它们
   加算子表之外的属性会被 `check_attr_names` 拒绝（这是第一轮跑测试就撞到的，
   `relu` 不接受 `layout`）。

**判据**（23 条，其中 1 条真机）

- 语义（6）：conv / maxpool / avgpool 的 NHWC 与 NCHW 参考结果**逐位相等**
  （`assert_array_equal`，不是 allclose）；NHWC 下权重与 bias 的形状不翻转
  （带 bias 的 stride=2 用例）；非法 `layout="chw"` 被 infer 拒绝；NHWC 前向
  求导抛 `NoGradientError` 且消息含 `nhwc`。
- 声明（3）：参考后端两类都声明；设备后端只声明 NCHW；未声明 `supported_layouts`
  的对象按最小承诺（`{"nchw"}`）处理。
- 传播（13）：单 conv 被翻（节点序列为 transpose / conv2d / transpose，conv 的
  `layout` 为 nhwc 且输出形状变 NHWC）；conv→relu→pool 整段翻且只付两次转置；
  翻转后与翻转前输出逐位一致（conv→relu；conv→gelu→pool）；图输出仍是 NCHW
  形状；纯逐元素链不翻；`reshape` 把区域切断（两侧各付两次转置，共 4 次）；
  区域输出同时被内外消费时出口只转一次；权重来自可翻转节点时该 conv 被剔除、
  图保持原样；两个不相连区域各付两次（共 4 次）；后端只声明 NCHW 时抛
  `LayoutError` 且消息含 `nhwc`；没有可翻区域时不要求声明；pass 不修改输入图。
- 真机（1）：WSL HIP 上执行 NHWC conv 抛 `BackendCapabilityError`，消息同时含
  `NCHW` 与 `NHWC`。

**负向对照**（5 组，全部先变红再还原）

1. 区域内的 conv 漏设 `layout="nhwc"` → 9 条变红（图按 NCHW 解释，形状推导直接
   失败）。
2. 入口转置的 perm 写成 `(0,3,1,2)` → 9 条变红（拓扑还在，语义已错）。
3. 后端声明检查短路（`if False and ...`）→ `test_backend_without_nhwc_is_refused`
   变红，**只有这一条**：说明该判据精确守住了"不静默降级"。
4. 权重也被 `to_nhwc`（去掉"只有激活进 NHWC 域"的判断）→ 6 条失败 + 1 条错误。
5. 设备端不拒绝 NHWC（`invoke` 与 `_planned_kernel` 两处短路）→ WSL HIP 上真机
   判据变红，且报出来的是内核层的误导性错误
   （`conv2d 输入通道 8 与权重通道 3 * groups=1 不一致`）。这正是必须在入口显式
   拒绝、而不是让内核自己撞墙的理由。

**验证**：本地 → 2675 个用例 / 416 跳过；WSL HIP（RX 7650 GRE / ROCm 7.2）→
2675 个用例 / 373 跳过。两处唯一失败仍是 `test_conformance` 的算子覆盖断言
（`cumsum` / `tril`），来自工作区里另一份未提交的工作，与本层无关。

**边界**

- **设备端没有原生 NHWC 内核**：传播出来的图只有参考后端能执行，HIP / CUDA
  在入口显式拒绝。要真正拿到收益，需要按 NHWC 写卷积的分块与向量化内核，
  那是另一层工作，不是给现有内核加一个分支。
- **无 GEMM epilogue**：NHWC 最常见的收益来源是"卷积输出直接是 GEMM 的 N 维
  行主序"，这里没有 epilogue，因此本层交付的是**表达位与优化形态**，不是
  加速开关——没有做任何性能对比，也不声称更快。
- **传播范围只覆盖 conv / pool + 一元逐元素**：reshape、split / concat、归约、
  矩阵乘都不翻。它们要么依赖维度顺序，要么分块策略与布局耦合，翻过去只是把
  决策推给后端。
- **训练图不支持 NHWC 梯度**：前向可翻，反向显式拒绝。布局传播按推理期优化
  设计，接进训练器之前需要补齐三个反向算子的 NHWC 形态。
- **没有接进默认管线**：`LayoutPropagation` 需要目标后端声明，带参数，按约定
  不进 pass 注册表；目前也不在任何 `*_PASS_NAMES` 里，由调用方显式实例化。

### 5.247 IR 补 `cumsum` / `tril`：分块 SSM 的块内权重与因果掩码

**动机**：Helix（Nucleus 侧）的 SSM 用分块扫描，块内需要 log-space 的**前缀和**
（还要乘块内 0/1 因果权重）；窗口注意力需要**下三角掩码**。这两件事用已有算子
拼不出来：`scan_linear` 的 `a_t` 是可变递归，纯累加 `a_t == 1` 套进去会白背串行
骨架；掩码则需要"最后两维逐片"的语义，`mul` + 广播只能表达 1 维。IR 里补这两个
算子，而不是让每个前端各写一遍切片 + 累加。

**语义**

- `cumsum(axis)`：沿**单个轴**的含端点前缀和，形状与 dtype 不变；f16 在 float32
  里累加（前缀和每一步都是最终输出，逐项 f16 累加会先溢出再失真，没有"最后
  cast 一次"能补救）。`axis` 缺省 `-1`，负轴按秩归一化。
- `tril(diagonal)`：最后两维保留下三角（含对角线偏移 `k`），其余置零；前导维
  逐片独立，不做跨片操作。`diagonal` 必须是整数，构图期钉死类型。
- 反向：`cumsum` 的伴随是"总和 − 不含端点的前缀和"（`constant` + `pad` +
  `slice` 平移，全部由已有语义组合，因此反向图仍然可移植）；`tril` 是逐元素
  0/1 掩码的乘法，掩码自伴，反向就是它自己。两者与中心差分对拍通过。

**设备端接线**（`device_backend.py`）

- `GPU_OPS` 增加两项；`_register_kernels` 同步登记——末尾的 `_OPS` 与已注册集合
  对账会让漏一处**在构造期**就失败。`HIP_OPS = GPU_OPS`，HIP 不需要改代码。
- `_op_cumsum`：把形状折成 `(outer, steps, inner)`，`work = outer * inner` 就是
  并行链数，grid 按 `work` 铺（与 `scan_linear` 同构）；参数顺序与内核一致
  `(x, out, work, steps, inner)`。轴越界在设备端也拦一次，快路径绕过 verifier 时
  不会把错误轴当合法轴。
- `_op_tril`：`rows/cols` 取 `shape[-2:]`，`n = x.size`，grid 按元素数铺；参数
  `(x, out, n, rows, cols, diagonal)`。条件外写 0 而不是跳过，输出不带未初始化值。
- 修复 L2 描述里的一个真 bug：`tril()` 的 `grid_stride(limit, *body)` 是**可变
  参数**，原写法把三条语句包成一个 tuple 传进去，渲染器报"未知语句类型: tuple"。
  本地无 GPU 时这条路径不被执行，只有 HIP 真机 conformance 才会暴露。

**验收**

- conformance 新增两个用例：`cumsum` 用 `(2,4,3)` + `axis=1`（**非末轴**才覆盖
  两侧折维），`tril` 用 `(2,5,5)` + `diagonal=-1`（负偏移 + 前导维）。套件的覆盖
  度检查因此重新全绿——声明了却没用例会报 untested。
- `verify_cuda.py` 加两条真机用例：`cumsum_mid_axis`、`tril_negative_diagonal`。
- 本地全量 2675 个用例 / 416 跳过，全绿（此前唯一失败就是这两个算子的覆盖断言，
  现在消失）。WSL HIP（RX 7650 GRE / ROCm 7.2）2675 个用例 / 373 跳过，全绿。
  真机 conformance 报告实测：`cumsum` / `tril` 均为 `status=executed`、
  `max_error=0.000e+00`，`contract` / `portability` / `serialization` /
  `explicit_failure` / `coverage` 五项检查全为 `True`。f16 口径：两者都随
  `DType` 渲染 f16 变体（累加在 float、存回走 `af_st_f`），因此计入"f16 覆盖
  59 个"（62 个设备端算子 − 3 个 int8 语义算子）。
- IR 算子总数 63 → 65；设备端承接数 60 → 62（61 个自研内核 + 由反量化与 GEMM
  组合的 `linear_i8`）。`fft` / `rfft` 仍只有 CUDA 的 cuFFT 路径。

**边界**

- `cumsum` 是一条线程一条链：并行度来自 `outer * inner`，**时间轴本身仍串行**。
  长轴 + 小尾维（典型的分块 SSM 块内 `T=64`、`inner=1`）填不满机器；要真正加速
  需要块内并行前缀和（Blelloch / 分块两趟），那是下一步而不是当前语义缺口。
- 设备内核只承接浮点（与其它自研内核一致），整数 `cumsum` / `tril` 在设备端显式
  拒绝；参考后端按 numpy 语义支持。
- 这两个算子只是**补上表达力**，Helix 迁移还缺的 `clamp` 与整数 `mod` 已在
  5.248 补齐。其余此前被列为缺口的算子可以绕过：`outer` / `softplus` 由广播与
  已有一元算子组合，`argsort` 由 `topk_mask` 覆盖，`repeat_interleave` 由
  `reshape + broadcast_to` 表达，`scatter_add` 复用 `models/moe.py` 的
  `topk_mask + sparse_linear` 图，`arange` / `where` 在序列长度静态时于构图期
  算成常量或加性 mask。

### 5.248 IR 补 `clamp` / `mod`：SSM 的数值稳定与 Engram 的整数哈希

**动机**：Helix 的 SSM 靠截断维持数值稳定——`logA.clamp(-4, 0)` 防止衰减因子
越界，`cum_in.clamp(min=-80)` 防止 `exp` 下溢成 0 后再取对数，`dt.clamp(max=2)`
限制步长，`novelty.clamp(0, 1)` 把门控压回概率区间。Engram 的哈希表下标则是
`slot = (slot*131 + g) % n_slots`，是**整数**运算。两者都拼不出来：`step` 能造
门，但把 `clamp` 拆成三个节点会让每处调用都重复一遍；`mod` 在 IR 里根本不存在，
而浮点 `div` 拿不到余数。

**语义**

- `clamp(x, min=..., max=...)`：逐元素截断，`min` / `max` 至少给一侧，都放在
  **属性**里。它们是前端的 Python 常量，做成输入会凭空多两条边与两次设备驻留；
  真要动态上下界，用 `maximum` / `minimum` 语义自己拼。构图期拒绝 `min > max`
  与两侧都缺省，非浮点 dtype 也拒绝。
- NaN 原样透传：比较为假时返回 `x` 本身。设备端因此用 `cond ? bound : x` 而不是
  `fminf` / `fmaxf`——后两者的语义是"返回非 NaN 的那一侧"，会把 NaN 吃掉。
- NaN 之外，边界点属于保留区间；反向就是区间门的 0/1 乘法，边界取闭区间导数
  （与 PyTorch 一致，与 `relu` 在 0 点取 0 的约定刻意不同）。
- `mod(a, b)`：**i32 only**，结果与除数同号，即 floor 语义，与 Python `%` /
  `torch.remainder` 一致。C 的 `%` 是截断取余，`-7 % 3` 在设备上会得 -1，
  必须显式修正。模数为 0 在参考实现里当场报错（整数没有 inf/nan 可用）。
- 不承接 i8：`slot * 131 + g` 这类累加在 i8 上必然溢出，IR 层就直接拒绝，
  免得运行期得到一个"看起来合理"的错误下标。

**设备端接线**（`device_backend.py`）

- `GPU_OPS` 增加两项；`_register_kernels` 同步登记。`mod` 还要进 `_INDEX_ARGS`
  （两侧都是整数数据），否则会在 `invoke` 的"只支持浮点"检查前被挡下。
- `_op_clamp`：参数 `(x, out, n, has_lo, has_hi, lo, hi)`。上下界走内核参数而不是
  生成期分支：同一张图里 `clamp(min=-80)` 与 `clamp(0, 1)` 很常见，按数值生成
  源码会让编译缓存按边界值增长。
- `_op_mod`：只承接同形与标量广播（Engram 的两侧同形、模数是标量），其它广播
  形态显式拒绝——整数缓冲不能落到"按 f32 解释"的错误路径上。整数内核不经过
  `af_ld_f` / `af_st_h`（那是浮点存取），L2 里用 `type_name="int"` 钉死参数类型，
  读写走裸 `Index` 与 `raw=True` 的 `Store`。
- L2 的 floor 修正只有一句：`r != 0 && (r < 0) != (y < 0)` 时 `r += y`。

**验收**

- conformance 新增四个用例：`clamp_range`（双侧）、`clamp_min_only`、
  `clamp_max_only`（缺省一侧的开关必须真的生效）、`mod_int_floor`（被除数有负值、
  模数正负混合且非零——只测正数发现不了截断取余的偏差）。
- `verify_cuda.py` 加两条真机用例：`clamp_two_sided`、`mod_negative_floor`。
- 本地全量 2675 个用例 / 416 跳过，全绿；WSL HIP（RX 7650 GRE / ROCm 7.2）
  2675 个用例 / 373 跳过，全绿，`clamp` / `mod` 的 conformance 实测
  `err=0.0e+00`。确认渲染源码：f32 / f16 的 clamp 都读成 float 比较、写回
  `af_st_f` / `af_st_h`；mod 为 `const int*` + `b[0]` 的标量路径。
- 真机抓到一个本地看不见的缺陷：`_mod_kernel` 渲染时把 `np.float32`（类）当
  dtype 传进 `codegen.render`，`_dtype_key` 按 `.name` 取类型，于是直接抛
  "只支持 f32/f16"（此前 REMIND 还把它当成正确写法记过一条）。改成
  `np.dtype("float32")` 后通过。整数内核的 dtype 参数只为拿同一份 prelude，
  类型由 `type_name="int"` 钉死。
- IR 算子总数 65 → 67；设备端承接数 62 → 64；VJP 规则 46 → 47（`mod` 是整数，
  没有梯度，刻意不注册规则）；f16 覆盖 59 → 60（64 个设备算子里去掉 3 个 int8
  语义算子与整数 `mod`）。

**边界**

- `clamp` 的上下界只能是标量常量。按张量动态截断需要 `maximum` / `minimum`，
  这两个算子仍然没有；目前用 `step` 的门 + 乘加可以表达，但要多物化两三个中间量。
- `mod` 只有 i32：i64 在 IR 里没有对应 dtype，超出 2^31 的哈希需要前端先拆分或
  换成 i32 的模运算（`n_slots` 远小于 2^31，实际不受影响）。
- `mod` 的符号语义是 floor，不是 C 的截断。任何"照着 CUDA 教程抄 `%`"的后端
  都会在负被除数上错一个模数，这一点写进内核注释而不是只写在文档里。

### 5.249 性能基线：图捕获 7×，与三个真实瓶颈

性能攻坚的第一步不是改代码，而是确认"慢在哪"。本节记录一次完整的 HIP
基线测量与由此定位的三个问题；它们都没有在本节修复，作为 5.250 起的输入。

**测量条件**：WSL Ubuntu-22.04 + ROCm 7.2，AMD RX 7650 GRE（8 GB），
`python -m benchmarks.run --rounds 3 --warmup 0.5`，显卡先压 0.5 秒抬频。

| 用例 | 参考后端 | HIP（graph off） | HIP（graph auto） |
| --- | --- | --- | --- |
| matmul_256 | 0.146 ms | — | 0.697 ms |
| matmul_1024 | 8.09 ms | 8.22 ms | 8.15 ms |
| matmul_batched | 0.352 ms | — | 0.982 ms |
| conv2d_3x3_pad1 | 4.36 ms | 1.32 ms | 1.28 ms（p90 5.69） |
| conv2d_1x1 | 1.62 ms | — | 1.22 ms |
| maxpool2x2 | 6.73 ms | 7.27 ms | 7.67 ms |
| transformer_block | 0.724 ms | 5.90 ms | 0.829 ms |
| transformer_block_resident | 0.707 ms | — | 0.685 ms |
| cnn_lenet | 2.07 ms | — | 0.583 ms |
| cnn_lenet_resident | 2.05 ms | — | 0.511 ms |

未测的格子是这一轮只跑了部分后端组合；不补数字。

**结论一：整图调度被 launch 开销主导，图捕获能拿回 7 倍。**
`transformer_block` 在 `graph off` 下 5.90 ms，`graph auto` 下 0.829 ms。
差异不在内核，而在每个算子一次 `hipLaunchKernel` + 状态查询：这张图有几十
个节点，单次 launch 在 WSL 上要几十微秒。默认保持 `off` 是刻意的——基准的
职责是暴露 launch 成本，不是把它藏起来；真实推理应当开 `auto`（`_min_graph_nodes`
与 `_graph_failure` 已经处理"节点太少不值得捕获"和"捕获失败回退"）。

**结论二：`maxpool2x2` 在大形状上明显异常。**
输入 `(2, 32, 256, 256)`、输出 `(2, 32, 128, 128)`，每输出读 4 个 float、
写 1 个，总访存约 20 MB。7.67 ms 对应 2.6 GB/s，比同规模逐元素内核低约两个
数量级。图捕获对它无效（7.27 → 7.67 ms），所以问题在内核或它的线程铺法，
不在调度。这是当前最大的绝对浪费，也是下一节要查的第一个对象。

**结论三：`matmul_1024` 只有 264 GFLOPS。**
2×1024³ = 2.15 GFLOP / 8.15 ms。这条路径**确实**走 hipBLAS `Sgemm`
（`hip/backend.py::_cublas`），不是回退到自研内核，所以问题是这个库版本在
这张卡上的 GEMM 效率，而非我们的接线。要么换 hipBLASLt / rocWMMA，要么
确认该卡的 FP32 上限后调整预期——在拿到第二组硬件数据前不下结论。

**结论四：这些绝对值不能用来判断内核质量，WSL 环境本身限速。**
把 maxpool 换成一个同规模的纯逐元素 `relu`（4M 元素，读写 64 MB）对照：
9.0 ms，即 7.4 GB/s；maxpool 是 2.9 GB/s。一个只有 4 个元素的 `relu` 图要
0.52 ms，说明这两条里真正花在内核上的时间是 8.5 ms 与 6.8 ms，不是调度开销。
按这张卡约 288 GB/s 的名义带宽，两个数字都低一到两个数量级；hipBLAS 的
264 GFLOPS 也落在同一量级。三条互不相关的路径同时偏慢，指向环境而不是实现。
直接证据：WSL 里 `rocm-smi` 报 `Driver not initialized (amdgpu not found in
modules)`，读不到也调不动时钟——没有 amdgpu 内核模块，GPU 的 DPM 不在我们的
控制范围内。

因此：**WSL 上的绝对值只能用于"同一环境、同一轮测量"的前后对比，不能用来
和 cuDNN / MIOpen 比，也不能据此判定自研内核比库慢。** 要给出对外可引用的
性能数字，必须先在一台原生 Linux（或 Windows 原生 ROCm）机器上复测同一套
`benchmarks/run.py`。在那之前，性能攻坚的目标只能是"相对提速"。

**边界**：单卡、单驱动版本、WSL 环境。上表不能外推到其他 GPU；`conv2d_3x3`
的 p90（5.69 ms）与中位数（1.28 ms）差 4 倍，说明单次测量在这张卡上不稳定，
任何"优化前/后"的对比都必须用同一轮多轮中位数。

### 5.250 性能测量的环境前提：先证明机器可信，再谈优化

5.249 的结论是"WSL 上的绝对值不可引用"。这一节把它变成可操作的验收：
换机器不是目的，**能证明这台机器没有被人为限速**才是。

**最低要求（HIP / ROCm）**

1. 原生 Linux（不是 WSL），且 `lsmod | grep amdgpu` 有输出。WSL 里没有这个
   内核模块，时钟与功耗状态都不受控——这正是 5.249 里所有带宽低两个数量级
   的原因。
2. `rocm-smi --showclocks` 能读到 `sclk`；采集前执行
   `sudo rocm-smi --setperflevel high`，避免 DPM 把频率压到最低档。
3. 卡直连 PCIe x16，不走 USB / Thunderbolt eGPU，也不与显示输出抢占。

NVIDIA 侧对应的是 `nvidia-smi -q -d CLOCK` 可读、`nvidia-smi -lgc` 可用。

**自检：三条命令，任何一条不过就先修环境**

```bash
lsmod | grep amdgpu                  # 必须有输出
rocm-smi --showclocks | grep -i sclk # 必须能读到当前频率
python -m benchmarks.run --backend hip --rounds 5
```

第三条看两个数：`matmul_1024` 若低于 **2 TFLOPS**（即耗时高于约 1.1 ms），
或逐元素类用例折算出低于 **100 GB/s** 的有效带宽，说明机器仍受限——此时
优化任何内核都是在错误的基线上做无用功。

**采集与留档**

```bash
python -m benchmarks.run --backend hip --graph-mode off  --rounds 5 --save benchmarks/hip_off.json
python -m benchmarks.run --backend hip --graph-mode auto --rounds 5 --save benchmarks/hip_auto.json
python -m benchmarks.run --backend ref                  --rounds 5 --save benchmarks/ref.json
```

三份 JSON 与 `rocm-smi --showclocks`、`python -c "import aetherflow;
print(aetherflow.__version__)"` 的输出一起提交。`--check` 需要基线文件，因此
第一次采集同时也是往后回归门禁的起点。

**CUDA 云机（本机只有 AMD 卡时的替代路径）**

```bash
git clone https://github.com/Aether-studio-Modles/AetherFlow.git
cd AetherFlow && python -m venv .venv && . .venv/bin/activate
pip install -e '.[cuda]'                       # cupy-cuda12x
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,power.limit --format=csv
sudo nvidia-smi -lgc "$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits)"
python -m benchmarks.run --backend cuda --graph-mode off  --rounds 5 --save benchmarks/cuda_off.json
python -m benchmarks.run --backend cuda --graph-mode auto --rounds 5 --save benchmarks/cuda_auto.json
python -m benchmarks.run --backend ref                  --rounds 5 --save benchmarks/ref_cloud.json
```

云实例通常不给 sudo，`-lgc` 失败就跳过，但必须把 `nvidia-smi` 那张表原样贴进
记录，否则事后无法判断数字是不是被共享实例拖慢的。CUDA 侧的合格线按卡型另定
（A100 的 `matmul_1024` 应到 8 TFLOPS 以上，4090 更高）；先记录，再谈优化。

**判据**

- 每个用例的 `p90 / median ≤ 1.5` 才算稳定；超过就重跑，不要用单次结果做
  优化前后的对比。
- 只有满足上面自检的机器，才允许把数字写进 README §5 或对外材料；否则只能
  写"同机相对提速 X 倍"，并注明测量环境。

### 5.251 性能口径：把 D2H 从算力里剥出去

5.249 把所有路径都慢归因于"WSL 限速"，这个结论只对了一半。在 A800 上复测后
真相是：**慢的是测量口径，不是内核。**

**机器是达标的。** 只用 cupy、不经过 AetherFlow：

| 项 | 数值 |
| --- | --- |
| `relu`，4M 元素（读写 64 MB） | 0.052 ms，646 GB/s |
| `matmul` 1024³ | 0.191 ms，11.2 TFLOPS |
| `matmul` 4096³ | 9.41 ms，14.6 TFLOPS |

A800 的 f32 上限约 19.5 TFLOPS，14.6 是合理的 75%。

**慢的是每一次 `execute` 的收尾。** 同一个 `relu` 图（输入已驻留在设备上）
单次 `execute` 要 1.98 ms。`cProfile` 跑 50 次共 0.082 s，其中 0.074 s 全在
`cupy.ndarray.get()`——即 `to_host`。4 MB / 1.48 ms ≈ 2.8 GB/s，这是这台云主机
的 D2H 带宽，与内核无关。`Executor.run` 的契约是"返回主机数组"，每个图输出
都要拷一次；基准的 `measure` 每轮调 `execute`，于是每轮付一次。

**修复：`DeviceBackend.execute_device`。** 它走 `run_device(release=True)`
但不在结尾 `to_host`，返回设备数组；基准加 `--device-outputs` 走这条路径，
JSON 里记录 `device_outputs`，`check_regression` 发现基线与当前口径不同时
直接判 `throughput_mode_mismatch`，避免"换了测量方式"被当成回归。
刻意不接 CUDA Graph 计划：捕获计划的返回值是按主机数组组织的，混用会让
"是否回传"隐含在计划里。

**A800 上的三层数字（ms，graph off，同一轮采集）**

| 用例 | 端到端 | 设备侧 | 裸 cupy 常驻 |
| --- | --- | --- | --- |
| matmul_256 | 0.313 | 0.226 | — |
| matmul_1024 | 2.130 | 1.156 | 0.191 |
| matmul_batched | 0.587 | 0.454 | — |
| conv2d_3x3_pad1 | 0.379 | 0.227 | — |
| conv2d_1x1 | 0.393 | 0.258 | — |
| maxpool2x2 | 3.496 | 2.486 | — |
| transformer_block | 1.764 | 1.661 | — |
| transformer_block_resident | 1.374 | 1.288 | — |
| cnn_lenet | 0.850 | 0.801 | — |
| cnn_lenet_resident | 0.641 | 0.592 | — |

端到端与设备侧之差稳定在 0.05～0.13 ms：这台实例的 D2H 不是 5.249 里 WSL 的
那种量级，但仍是每轮一次的固定成本；`matmul_1024` 最明显（2.13 → 1.16）。

设备侧口径下 `matmul_1024` 仍有 1.16 ms，而纯算只要 0.19 ms。差额 ~0.95 ms
与两个 4 MB 输入的 H2D 吻合（实测 8 MB / 0.9 ms ≈ 9 GB/s）。也就是说：
**这台云主机的 PCIe 带宽才是当前最大的固定成本**，不是内核，也不是调度。

**图重放：对多节点图仍然是 3～6 倍，单节点图无收益**

第一版的 `execute_device` 故意绕过了图捕获，于是"设备侧 + auto"测到的完全是
逐节点 Python 开销——这不是流水线的真实吞吐。补齐之后（`execute_device` 现在
与 `execute` 共用同一份捕获计划），A800 上同一批用例（设备侧，ms）：

| 用例 | 逐节点 | 捕获重放 | 倍数 |
| --- | --- | --- | --- |
| transformer_block | 1.661 | 0.404 | 4.1× |
| transformer_block_resident | 1.288 | 0.216 | 6.0× |
| cnn_lenet | 0.801 | 0.239 | 3.4× |
| cnn_lenet_resident | 0.592 | 0.135 | 4.4× |

端到端口径一致（transformer_block 1.764 → 0.437，cnn_lenet 0.850 → 0.270）。
单节点图（`matmul_*`、`conv2d_*`、`maxpool`）没有可省的东西：它们的成本是
输入 H2D 与一次内核，捕获前后差 <5%。所以默认仍保持 `graph_mode="off"`，
需要吞吐的路径显式打开。

两条边界必须说清：

- 捕获计划返回的是**计划持有的输出缓冲**，下一次 replay 会覆盖它。设备侧
  流水线要么当场消费，要么自己 `copy()`。
- `conv_impl="cudnn"` 与捕获互斥：algo 查询是 host API，捕获期不允许。显式
  指定 cudnn 时捕获整图回落逐节点；`conv_impl="auto"` 在捕获期自动改走
  tiled，所以 auto + auto 才是"捕获可用"的组合。

**cuDNN 的 f32 卷积默认会掉到 TF32（已修）**

在 A800 上按形状扫了一遍 cuDNN 快路径与参考语义的差：`c_in=3` 时 1e-6，
`c_in=8/64/128` 时到 **5e-3**。这不是实现差异——cuDNN 在 Ampere 之后默认
允许卷积走 TF32（10 位尾数）。多通道卷积正好是这条默认最容易命中的形状，
而一致性套件里没有这么宽的单层卷积，所以一直没暴露。

修法是在描述符上锁死 `CUDNN_FMA_MATH`（只允许 FP32 FMA 算法），f16 不动。
修后同一组形状的最大绝对误差回到 1e-6～9e-6（纯累加顺序差），回归用例见
`TestCudaConvCudnn.test_wide_channels_keep_f32_precision`。

⚠️ 对外做 cuDNN 对照（`benchmarks/cudnn_compare.py`）时仍要
`NVIDIA_TF32_OVERRIDE=0`：对照脚本直接调库，不受本项目的数学类型约束。

**对 5.249 / 5.250 的修正**

- 5.249 的"环境限速"结论要改成"端到端口径 + WSL 的 D2H 更慢"；WSL 数字仍不
  可引用，但原因不是 GPU 算力。
- 5.250 的自检门槛不变，但它只能筛掉"算力不达标"的机器，筛不掉"PCIe 很慢"
  的实例。采集时必须同时报 H2D / D2H 带宽。
- 此前所有"自研内核比 cuDNN/MIOpen 慢 N 倍"的结论都混入了每轮一次的 D2H。
  已按设备侧口径在 A800 上重测（见下），旧数字作废。

**重测：自研卷积与 cuDNN 9.1 的真实差距（A800，设备侧，`NVIDIA_TF32_OVERRIDE=0`）**

| 形状 | tiled | im2col | cuDNN | tiled/cuDNN | im2col/cuDNN |
| --- | --- | --- | --- | --- | --- |
| 3x3_c3 | 0.264 | 0.321 | 0.077 | 3.43× | 4.16× |
| 1x1_c32 | 0.251 | 0.346 | 0.110 | 2.29× | 3.15× |
| 3x3_c8 | 0.413 | 0.370 | 0.087 | 4.76× | 4.26× |
| 3x3_c16 | 0.634 | 0.427 | 0.086 | 7.35× | 4.95× |
| 3x3_c24 | 0.829 | 0.506 | 0.087 | 9.53× | 5.81× |
| 3x3_c32 | 1.010 | 0.586 | 0.090 | 11.24× | 6.52× |
| 5x5_c16 | 1.007 | 0.579 | 0.189 | 5.32× | 3.06× |
| 1x1_c256 | 0.632 | 0.505 | 0.128 | 4.95× | 3.96× |
| 3x3_c64 | 1.874 | 0.716 | 0.128 | 14.66× | 5.60× |
| 3x3_c128 | 1.323 | 0.603 | 0.102 | 12.99× | 5.92× |

结论没变，但口径干净了：分块直算落后 cuDNN 2.3～14.7×（差距随通道数拉大，
因为 cuDNN 走的是张量核 / Winograd 一类的多算法选择），im2col+cuBLAS 落后
3.1～6.5×，且 im2col 的额外缓冲随通道数线性增长（`cols` 列）。这才是下一步
要做的事：卷积的算法选择与张量核路径，而不是继续抠 launch 开销。

⚠️ 在 Ampere 及之后的卡上跑这个对照必须先 `NVIDIA_TF32_OVERRIDE=0`：cuDNN
默认允许卷积走 TF32，与 f32 差 1e-3，会先被数值一致性检查拦下。

**边界**：`execute_device` 对 numpy 输入仍每轮上传，测"纯算"需要把输入驻留
（`resident_parameters` 或直接传设备数组）；它也不做 `dispose`，输出归调用方。

### 5.252　卷积的 auto 选择：把"哪个实现更快"压成一个形状的函数

5.251 结尾把卷积列为最大差距，但那张表比的是**单个实现**与 cuDNN。框架自己
的 `conv_impl="auto"` 在四个实现里选，选错了差距会被放大——问题不在内核，在
选择。旧判据来自 5.151（M40）：隐式 GEMM 只在 `K > 32` 且特征图不大时进，
`K >= 288` 一律让给 im2col。A800 上这条结论不成立。

**测量方法先固定。** 同一实现按固定顺序测会互相污染：先跑的 tiled 把 L2 与
时钟状态改掉，后面的实现能差出 50%。这里每个实现先 spin 0.10 s、按随机顺序
各测 3 轮取最好（`execute_device` 口径，输入常驻）。

**二维网格（n=8，o=32，3x3；单元格是 implicit/im2col 的耗时比，<1 为隐式快）**

| hw | k=144 | k=216 | k=288 | k=324 | k=360 |
| --- | --- | --- | --- | --- | --- |
| 256 | 0.830 | 0.826 | 0.830 | 0.844 | 0.846 |
| 576 | 0.827 | 0.840 | 0.841 | 0.859 | 0.869 |
| 784 | 0.844 | 0.847 | 0.864 | 0.871 | 0.880 |
| 1024 | 0.847 | 0.865 | 0.878 | 0.889 | 0.917 |
| 1600 | 0.870 | 0.892 | **1.069** | **1.117** | **1.232** |

分界与 k、hw 各自都不是单调的：k=288 在 1024 像素内赢、1600 像素输；k=144
到 1600 像素还在赢。换成 batch 也一样——所以真实变量是第三个数。

**分界是 im2col 要物化的元素数 `n*ho*wo*K`。** 把 n 也扫一遍：

| n | K | hw | work（M 元素） | implicit/im2col |
| --- | --- | --- | --- | --- |
| 8 | 144 | 1600 | 1.84 | 0.903 |
| 8 | 288 | 784 | 1.81 | 0.868 |
| 16 | 144 | 1024 | 2.36 | 0.873 |
| 8 | 288 | 1024 | 2.36 | 0.873 |
| 8 | 144 | 2304 | 2.65 | 0.901 |
| 8 | 216 | 1600 | 2.76 | 0.900 |
| 8 | 360 | 1024 | 2.95 | 0.934 |
| 16 | 288 | 784 | 3.61 | **1.065** |
| 8 | 288 | 1600 | 3.69 | **1.050** |
| 32 | 144 | 1024 | 4.72 | **1.059** |
| 16 | 288 | 1024 | 4.72 | **1.149** |
| 8 | 144 | 4096 | 4.72 | **1.134** |
| 16 | 360 | 1024 | 5.90 | **1.249** |
| 32 | 216 | 1024 | 7.08 | **1.314** |
| 64 | 144 | 1024 | 9.44 | **1.424** |

<= 2.95M 的 15 个点全部 <1（0.87～0.93），>= 3.61M 的 10 个点全部 >1
（1.05～1.42）。3.2M 元素 ≈ 12.8 MB 的 f32 中间缓冲，正好卡在两组之间。
这也解释了 5.151 为什么说"K >= 288 隐式就输"：那批形状都在 12.8 MB 以上。

**K 的上界独立存在。** 物化规模很小也救不了长内层循环：

| n | K | hw | work（M） | implicit/im2col |
| --- | --- | --- | --- | --- |
| 8 | 400（5x5） | 256 | 0.82 | 0.809 |
| 8 | 576 | 576 | 2.65 | 0.907 |
| 8 | 576 | 256 | 1.18 | 1.099 |
| 8 | 800（5x5） | 256 | 1.64 | 1.303 |
| 8 | 864 | 196 | 1.35 | 1.358 |
| 8 | 1152 | 64 | 0.59 | 2.079 |
| 8 | 2304 | 64 | 1.18 | 2.113 |

所以判据收成两条独立排除项 + 一条门槛（`plan.conv_implicit_preferred`）：

```
K <= 32                      -> tiled（分块直算）
kh*kw == 1                   -> 不进隐式（1x1 没有窗口重叠）
K > 400                      -> im2col
n*ho*wo*K > 3.2M             -> im2col
否则                          -> implicit
```

**cuDNN 只在 1 GFLOP 以上启用。** 大形状上 cuDNN 全面领先，但低 GFLOP 时它
付的是描述符、算法查询与独立 bias 核的 host 开销：

| 形状 | GFLOP | cuDNN | im2col | 说明 |
| --- | --- | --- | --- | --- |
| 16x16x16x16 c16 3x3 | 0.02 | 0.255 | — | cuDNN 慢于隐式 0.177（1.4×） |
| 8x64x64x64 c64 3x3 | 2.42 | 0.836 | 1.103 | 交叉点附近 |
| 8x128x56x56 c128 | 7.40 | 1.194 | 1.594 | |
| 8x256x56x56 c256 | 29.6 | 2.549 | 4.042 | |
| 16x512x28x28 c512 | 59.2 | 3.192 | 6.987 | |
| 8x64x224x224 c64 | 29.6 | 9.648 | 12.257 | |

门槛 `CONV_CUDNN_MIN_GFLOPS = 1.0`，显式 `conv_impl="cudnn"` 不受它约束。

**复测（A800，设备侧，auto 与同形状三个实现里的最优之比）**：14 个小/中形状
1.00～1.05（1.05 的三点是与 im2col 同实现的测量噪声），5 个大形状 0.99～1.03
（即 auto 已经等于 cuDNN）。回归：`tests/test_cuda_backend.py` 355 例通过。

**没采纳的两个点，留档**

- 1x1 卷积走隐式在 A800 上快 11%（0.351 vs 0.392 ms），但 HIP 实测慢 6%，
  且 1x1 的 im2col 缓冲本来就小（不随核面积放大），收益与风险不成比例；
- `K=576` + 576 像素（work 2.65M）隐式快 9%，但同 K 的 256 像素慢 10%——
  单一 K 上互有胜负，不为它开口子，K 上界保持 400。

### 5.253　卷积的三层口径：H2D、固定延迟与内核

5.251 的卷积表（"分块直算慢 2.3～14.7×、im2col 慢 3.1～6.5×"）有两个混淆项，
拆开之后结论要改。

**混淆项一：每轮一次的 H2D。** 那张表的自研侧用 `execute_device(g, feed)`，
`feed` 是 numpy——`execute_device` 只免掉 D2H，输入仍每轮上传；cuDNN 侧是
`cp.asarray` 一次后常驻。同形状 A/B（A800，稳态，ms）：

| 形状 | 输入 MB | tiled（numpy） | tiled（驻留） | im2col（numpy） | im2col（驻留） |
| --- | --- | --- | --- | --- | --- |
| 3x3_c16 | 1.02 | 0.432 | 0.326 | 0.436 | 0.162 |
| 3x3_c32 | 2.04 | 0.647 | 0.485 | 0.529 | 0.212 |
| 3x3_c64 | 3.20 | 1.614 | 1.380 | 0.635 | 0.249 |
| 3x3_c128 | 1.33 | 1.088 | 0.970 | 0.472 | 0.266 |

H2D 在这台实例上是 6～9 GB/s，3.2 MB 要 0.4 ms——比卷积本身还大。旧表里
"3x3_c16 慢 7.35×"有 0.25 ms 是 PCIe，不是内核。

**混淆项二：单次调用的固定下限。** 同一批形状换成"单次 + sync"，10 个形状
× 4 个自研实现的 40 个格子里，**每一个**都落在 **0.0753～0.0867 ms**，与形状
和实现都无关（cuDNN 直调更低，0.052～0.122）。这是 host 提交 + sync 往返的
延迟下限（`execute_device` 逐节点派发一次卷积），不是 GPU 时间。低 GFLOP 的
形状（0.007～0.3 GFLOP）全部被它盖住：谁快谁慢在单次口径里测不出来。所以
5.251 的"低 GFLOP 落后 3～5×"是**延迟差距**。

**三层口径的定义**（`benchmarks/cudnn_compare.py` 已按此改）：

| 口径 | 做法 | 回答的问题 |
| --- | --- | --- |
| 单次 | 一次调用 + sync | 调用方眼里的延迟（含 0.08 ms 固定成本） |
| 稳态 | 连续 20 次后 sync 一次，平摊 | 流水线吞吐（host 与 GPU 重叠） |
| 边际 | batch=1 与 batch=4 的斜率 | 纯内核：每多一张图多花多少 |

**稳态与边际（A800，驻留输入，TF32 关闭，ms）**

| 形状 | GFLOP | tiled | implicit | im2col | cuDNN | 最优 TFLOPS | cuDNN TFLOPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3x3_c3 | 0.007 | 0.076 | 0.077 | 0.152 | 0.052 | 0.09 | 0.14 |
| 1x1_c32 | 0.008 | 0.076 | 0.076 | 0.153 | 0.105 | 0.11 | 0.08 |
| 3x3_c8 | 0.075 | 0.135 | 0.088 | 0.154 | 0.076 | 0.86 | 1.00 |
| 3x3_c16 | 0.151 | 0.256 | 0.163 | 0.156 | 0.076 | 0.97 | 2.00 |
| 3x3_c24 | 0.226 | 0.370 | 0.236 | 0.168 | 0.076 | 1.35 | 3.00 |
| 3x3_c32 | 0.302 | 0.484 | 0.312 | 0.213 | 0.077 | 1.42 | 3.95 |
| 5x5_c16 | 0.419 | 0.692 | 0.436 | 0.278 | 0.101 | 1.51 | 4.15 |
| 1x1_c256 | 0.206 | 0.245 | 0.263 | 0.154 | 0.116 | 1.33 | 1.77 |
| 3x3_c64 | 0.925 | 1.375 | 0.884 | 0.249 | 0.078 | 3.71 | 11.92 |
| 3x3_c128 | 0.462 | 0.960 | 0.587 | 0.267 | 0.074 | 1.73 | 6.22 |

边际口径（每多一张图）给出的结论一致，且把固定成本彻底消掉：3x3_c64 的
tiled / implicit / im2col / cuDNN 分别是 1.69 / 1.09 / 0.28 / 0.075 ms。

**结论**

- 低 GFLOP 的形状**自研并不落后**：3x3_c3、1x1_c32、1x1_c256 的稳态耗时与
  cuDNN 同量级甚至更低（1x1_c32 是 cuDNN 的 0.72×）。旧表把它们算成 2～3×，
  是单次口径与 H2D 的假象。
- 真实差距在宽通道的 3x3/5x5，且是**内核效率**：im2col 稳态 3.71 TFLOPS
  （c64）对 cuDNN 的 11.92，差 3.2×；implicit 1.05、tiled 0.67，差 11～18×。
  差距随 K 单调拉大，与 5.251 的方向判断一致，但幅度收窄到 1.4～3.7×。
- cuDNN 的 3x3 系列在 c16 以上维持 0.072～0.080 ms（2～12 TFLOPS）：耗时不随
  K 爆炸，说明走的是减少乘法的算法（Winograd 一类）而不是更快的同构内核。

**分块参数没有余量。** 对 tile_w ∈ {16,32,64} × roc ∈ {2,4} 全扫（tiled 与
implicit，五个代表形状），六个格子的差距都在 10% 以内，且最优组合随形状漂移：

| 形状 | 实现 | 默认（32,4） | 扫描最优 |
| --- | --- | --- | --- |
| 3x3_c16 | tiled | 0.257 | 0.230（16,2） |
| 3x3_c32 | tiled | 0.490 | 0.447（16,2） |
| 3x3_c64 | tiled | 1.368 | 1.234（16,4） |
| 3x3_c64 | implicit | 0.883 | 0.844（64,4） |
| 3x3_c128 | implicit | 0.583 | 0.581（32,4） |
| 5x5_c16 | tiled | 0.688 | 0.610（16,2） |

所以瓶颈不在线程铺法，在内核结构：一个线程只算 **1 个空间位置 × ROC 个通道**，
每 4 次 FMA 要付 1 次输入读 + 4 次 shared 权重读，算术强度约 2 FLOP/B；调参
只能在这条曲线上挪几个百分点。要追上 cuDNN 必须换结构——空间寄存器分块
（每线程 2～4 个输出位置，输入与权重在寄存器里复用）或 Winograd
F(2x2,3x3)（乘法量降到 2.25 分之一，代价是变换的读写与 1e-4 容差内的数值
验证）。这是 5.254 的题目，本轮只把口径与判据钉死。

**口径修正落地**：`benchmarks/cudnn_compare.py` 改为设备驻留输入 + 稳态口径，
`--single` 可切回单次延迟；两列都会算，取最优时用稳态。表里的 `cols MB`
不变，仍是 im2col 的物化上界。

### 5.254　空间寄存器分块：一个否定结果

5.253 把瓶颈定位到"每线程 1 个空间位置 × ROC 个通道"，并列出两条改结构的
路：空间寄存器分块、Winograd。这一节做第一条，结论是**在 A800 上交不出收益，
已回退**；把过程与判据留在这里，免得下次再花同样的时间。

**设计**：每个线程算两个空间位置，`acc` 从 `ROC`（4）个涨到 `2 * ROC`（8，
正好是 `ArrayDecl` 的上限），同一个 `(c, ki, kj)` 的权重从 shared 读一次、
乘两次。两个位置取 `wo0 = base + tx`、`wo1 = base + nx + tx`——相隔一个 tile
宽，而不是相邻一列。理由是访存形状：相隔一列会让两条写指令的地址步长变成
2，合并失效；相隔一个 tile 宽时两条写各自 `tid` 连续，warp 内都是合并的。
输入窗口因此从 `tile_w * sw + (kw-1) * dw` 变成
`2 * tile_w * sw + (kw-1) * dw`，`grid.x` 减半。

**判据**：只有 `kh*kw > 1`、`k > 32`、`wo > tile_w` 才加倍。1x1 的权重本来
只服务一个位置（5.252 已测到 1x1 走隐式还慢 6%）；K ≤ 32 是分块直算的地盘；
输出宽度不足两个 tile 时第二个位置整段越界。窗口翻倍若把 `toc` 挤小一档也
退回单位置（多一轮 staging 通常吃掉了加倍的收益）。

**实测**（A800，驻留输入，burst=20，稳态，ms；只列真正进入 tw=2 的六个形状）

| 形状 | tw=1 | tw=2（x 维不变） | tw=2（x 维减半） |
| --- | --- | --- | --- |
| 3x3_c8 | 0.0882 | 0.0934（+6%） | 0.1004（+14%） |
| 3x3_c16 | 0.1628 | 0.1685（+3%） | 0.1762（+8%） |
| 3x3_c24 | 0.2375 | 0.2475（+4%） | 0.2569（+8%） |
| 3x3_c32 | 0.3108 | 0.3249（+5%） | 0.3349（+8%） |
| 5x5_c16 | 0.4351 | 0.4711（+8%） | 0.4622（+6%） |
| 3x3_c64 | 0.8850 | 1.1094（+25%） | 1.1427（+29%） |

其余四个形状（3x3_c3、1x1_c32、1x1_c256、3x3_c128）按判据仍是 tw=1，两版
差异在 ±2% 噪声内。两份源码的数值都与 cuDNN 在 1e-4 内一致，所以问题不是
正确性，而是**加倍位置在这台机器上就是更慢**。

第一版保持 `blockDim.x = tile_w`，一个 block 覆盖 2 个 tile 宽 ⇒ 块数减半；
第二版把 `blockDim.x` 减半、覆盖列数不变 ⇒ 块数与线程总量都不变，每线程
工作量翻倍。两版都慢，说明不是"块数减半、SM 喂不满"这一条能解释的：

- 权重从 shared 读一次、乘两次，省下的是 shared 带宽；但 5.253 的测量
  （tile_w/roc 全扫只有 ±10%）已经说明 shared 权重读不是瓶颈，省它不产生
  收益，反而多出一次输入窗口读与 4 个累加器；
- `acc` 从 4 涨到 8 让每线程的活跃寄存器变多，`roc=8` 在 5.16 里同样更慢，
  这次是换了个方式撞上同一堵墙；
- 3x3_c64 的 25% 超出了寄存器压力的量级，还叠着 staging 窗口翻倍（shared
  从 12.5 KB 到 15.5 KB）与 `cb` 的摊销边界变化。

**回退**：两次实现提交都用 `git revert` 回到 5.253 的代码，`CONV_IMPLICIT_TW`
这类开关不留在树里——没有实测支撑的形状开关只会变成下一轮的噪声。结论写进
本节，基准脚本的 `--implicit` 对照一并回退。

**下一步**：空间分块在本机上不值得再试相邻列 / 沿 H 方向 / 4 个位置这些
变体（同一堵墙：算术强度上去、占用率与 shared 预算下来，而当前瓶颈不是
shared 带宽）。要真正接近 cuDNN 的 11.9 TFLOPS（c64），方向是**减少乘法量**
而不是提高单位乘法的复用：Winograd F(2x2,3x3) 把 3x3 卷积的乘法降到 2.25
分之一，代价是变换的额外读写与 1e-4 容差内的数值验证。

### 5.255　Winograd F(2x2,3x3)：第一个真正追上 im2col 的结构改动

5.253 把差距归到"乘法量没变"，5.254 证明"提高单位乘法复用"在这台机器上
无效。这一节走另一条路：**减少乘法量**。F(2x2,3x3) 把 3x3 卷积的乘法从
`9 * K` 个降到 `16 * K / 4 = 4 * K` 个，理论 2.25×；代价是输入 / 输出各一次
4x4 变换的读写，以及变换系数放大的舍入误差。

**设计：三次自研 launch + 一次批量 GEMM**

| 步骤 | 布局 | 做什么 |
| --- | --- | --- |
| `winograd_weight` | `(16, O, C)` | `U = G g G^T`，一线程一个 `(o, c)` |
| `winograd_input` | `(16, C, NT)` | `V = B^T d B`，一线程一个 `(c, tile)` |
| cuBLAS `sgemm_batched` | — | 16 个 `(O,C) x (C,NT)`，`batch=16` |
| `winograd_output` | 写回 `(N,O,HO,WO)` | `Y = A^T M A` + bias，一线程一个 `(o, tile)` |

`NT = N * ceil(HO/2) * ceil(WO/2)`。把 16 个变换位置放成**最外维**，每个位置
是一块连续的 `(C, NT)` / `(O, NT)`，正好对上 strided batched GEMM 的 batch
步长（`stride_b = C*NT`）——逐元素乘那一步因此不需要任何重排，也不写内核。

三条中间缓冲（相对 im2col 的 `N*K*HO*WO = 9*N*C*HO*WO`）：
`16*C*NT ≈ 4*N*C*HO*WO`、`16*O*NT ≈ 4*N*O*HO*WO`，比 3x3 的列矩阵小一半多。

`B^T` / `A^T` 的系数只有 `±1`，两次 4x4 乘法全部展开成加减（输入变换 32 次
加减、输出变换 16 次）；只有 `G` 的中间两行带 `1/2`，所以权重变换与给 bias
是仅有的乘法。核只做 f32、stride=1、dilation=1、groups=1；其余形状由派发处
显式拒绝（`plan.conv_winograd_supported`），不静默换实现。

**数值**（对 cuDNN，`NVIDIA_TF32_OVERRIDE=0`，七个 3x3 形状）

| 形状 | `max|wino-cuDNN|` | 相对 max | 1e-4 容差 |
| --- | --- | --- | --- |
| 3x3_c3 | 5.1e-07 | 1.1e-07 | 通过 |
| 3x3_c8 | 1.2e-06 | 1.4e-07 | 通过 |
| 3x3_c16 | 1.5e-06 | 2.0e-07 | 通过 |
| 3x3_c24 | 1.9e-06 | 1.3e-07 | 通过 |
| 3x3_c32 | 2.4e-06 | 1.5e-07 | 通过 |
| 3x3_c64 | 3.4e-06 | 2.5e-07 | 通过 |
| 3x3_c128 | 2.5e-06 | 1.9e-07 | 通过 |

相对误差约 2e-7，比 1e-4 的验收线低三个数量级。F(2x2) 的系数都是小整数
（`B^T`/`A^T` 只有 ±1，`G` 只有 1/2），误差主要来自 16 路累加的次序差异，
没有 F(4x4) 那种放大的系数。**F(4x4,3x3) 没有做**：它的变换系数到 4～5，
理论误差会到 1e-5～1e-4 量级，与验收线同阶；而这一轮的目标是先看
"减少乘法"能否兑现。

**实测**（A800，驻留输入，burst=20，稳态，ms）

| 形状 | winograd | im2col | 分块直算 | cuDNN | wino/im2col | wino/cuDNN |
| --- | --- | --- | --- | --- | --- | --- |
| 3x3_c3 | 0.133 | 0.157 | 0.079 | 0.063 | 0.85× | 2.1× |
| 3x3_c8 | 0.133 | 0.156 | 0.171 | 0.073 | 0.85× | 1.8× |
| 3x3_c16 | 0.134 | 0.159 | 0.326 | 0.072 | 0.84× | 1.9× |
| 3x3_c24 | 0.134 | 0.167 | 0.367 | 0.073 | 0.80× | 1.8× |
| 3x3_c32 | 0.134 | 0.212 | 0.486 | 0.073 | 0.63× | 1.8× |
| 3x3_c64 | 0.135 | 0.249 | 1.369 | 0.080 | 0.54× | 1.7× |
| 3x3_c128 | 0.135 | 0.266 | 0.968 | 0.073 | 0.51× | 1.9× |

**耗时几乎不随通道增长（0.133～0.135）**，而 im2col 从 0.157 涨到 0.266。
这既说明"减少乘法"确实兑现了（c64 上 0.54×、c128 上 0.51×），也说明这条
曲线到 c8 之后就被**固定开销**压平了。

**固定开销的分解**（c3 到 c128，手工调用四段，ms）

| 形状 | 四段 | 去掉权重变换 | 仅 GEMM + 分配 |
| --- | --- | --- | --- |
| 3x3_c3 | 0.078 | 0.069 | 0.034 |
| 3x3_c16 | 0.080 | 0.070 | 0.032 |
| 3x3_c64 | 0.105 | 0.096 | 0.073 |
| 3x3_c128 | 0.095 | 0.082 | 0.046 |

读法：

- 输入 + 输出变换各一次提交共 ≈ 0.035 ms（每次 ≈ 0.017）；权重变换那次
  ≈ 0.009。所以四次提交的 Python 侧成本 ≈ 0.045 ms。
- 走 `execute_device` 比手工四段多约 **0.055 ms**（0.133 对 0.078）——那是
  逐节点派发、`_contig`、dtype / 布局检查等框架成本，与实现无关（im2col 也
  付这一份）。
- 真正落在 GPU 上的只有 `仅 GEMM + 分配` 这一列：c3 的 0.034 里还含一次
  cuBLAS 提交与四次 `cp.empty`；纯计算要到 c64（0.073）才显形。

**CUDA graph 捕获没有帮忙**：同一批形状在 `graph_mode="auto",
min_graph_nodes=1` 下，winograd 在 c3/c8 上降到 0.074（捕获取胜），但 c32
以上反而变慢（c64 0.160、c128 0.144，都慢于未捕获的 0.135）。捕获计划每轮
要搬运可变输入到静态缓冲，且 16 路 batched GEMM 在 replay 里的调度并不比
逐次提交更省。所以"把四个 launch 折起来"这条不能指望捕获层解决，要动就得
在执行计划里合并（例如把权重变换挪到图 / 权重驻留层）。

**接管范围（auto）**：3x3 / stride=1 / dilation=1 且 `k > 32`
（`plan.conv_winograd_preferred`）。K ≤ 32 留给分块直算——3x3_c3 的分块
0.079 仍快于 Winograd 0.133。实测 auto 口径：

| 形状 | auto | 选中 | im2col（旧 auto） | cuDNN |
| --- | --- | --- | --- | --- |
| 3x3_c3 | 0.081 | tiled | 0.157 | 0.051 |
| 3x3_c8 | 0.136 | winograd | 0.156 | 0.073 |
| 3x3_c16 | 0.136 | winograd | 0.159 | 0.072 |
| 3x3_c24 | 0.135 | winograd | 0.167 | 0.074 |
| 3x3_c32 | 0.134 | winograd | 0.212 | 0.075 |
| 3x3_c64 | 0.136 | winograd | 0.249 | 0.075 |
| 3x3_c128 | 0.134 | winograd | 0.266 | 0.073 |

`CONV_AUTO_WINOGRAD` 只在 CUDA 上开（HIP 侧没有实测）。5x5 与 1x1 的路径
不变；cuDNN 的 1 GFLOP 门槛仍在 Winograd 之前——3x3_c64 的 0.925 GFLOP
刚好不到门槛，更大的图仍交给 cuDNN。

**结论与下一步**

- 3x3 宽通道上 Winograd 是第一个把自研路径从 im2col 的 0.16～0.27 ms 拉到
  0.133～0.136 ms 的结构改动（快 15%～48%），数值误差 2e-7，远在容差内。
- 剩下的 1.7～1.9× 差距**不是内核算得慢**：四段手工调用里 GPU 计算只占
  0.03～0.07 ms，其余是四次提交（0.045）与框架逐节点派发（0.055）。
  继续压内核（例如换 F(4x4)）的收益上限是先验的这 0.03～0.07，而且要以
  1e-4 附近的数值风险换。
- 下一步优先级：① 把权重变换挪出每次执行（图级常量折叠 / 权重驻留），
  省 0.009 ms 与一次提交；② 在执行计划里合并输入 + GEMM + 输出三段，
  这才是 0.1 ms 量级的来源；③ 若要继续减乘法再考虑 F(4x4)，但先补
  1e-4 的误差实测。

### 5.256　融合 Winograd：把 4 次提交压成 1 次，小通道追平 cuDNN

5.255 把差距拆成三份：四次提交 0.045、框架逐节点派发 0.055、GPU 计算
0.03～0.07。这一节只动第一份与控制流，做法是**把后三段合进一个内核**：
`V` 不再落全局缓冲，而是留在共享内存 `(16, C, TT)`；`M` 不进全局缓冲，
而是留在寄存器；输出变换与 bias 在同一线程里接着做。权重变换仍是独立
一次 launch（原因见文末）。

换机说明：本轮在 A800-SXM4-40GB + cuDNN 9.1（Ubuntu 22.04）上测，
与 5.255 的 A800 不是同一台，绝对值不可跨节对比；同一节内的比值可比。

**实现**（`lower/kernels.py::winograd_fused`，L2 描述）

| 阶段 | 线程分工 | 数据落点 |
| --- | --- | --- |
| 输入变换 `V = B^T d B` | 一个线程一个 `(c, tile)`，`idx < C*TT` | 写共享 `sv[(p*C+c)*TT+lt]` |
| 逐点乘 + 沿 `C` 求和 | 一个线程一个 `(o, tile)`，共 `TT*O` 个 | 16 个累加器留在寄存器 |
| 输出变换 + bias | 同一线程接着算 `A^T M A` | 写回 `(N,O,HO,WO)` 的 2x2 |

`TT = 8`（`plan.CONV_WINOGRAD_TILE`）：c3～c24 四档上 TT=2/4 的每块线程
数不足，一块只有 1～2 个 warp，拉不满 SM（c8 的 device 时间 0.0364/0.0240
对 TT=8 的 0.0199）。共享内存按 `16*C*TT*4` 算，C=24 时也只有 12 KB。

两个实现上的坑，都记在这里免得重犯：

- **L2 的 `Bin` 打印不加括号**。`(p*C + c)*TT` 写成
  `Bin("*", Bin("+", ...), Var("TT"))`，渲染出来是 `p * C + c * TT`。共享
  索引必须显式 `Paren`，否则读到的和写到的不是同一格。第一版就是这样
  静默算错的（相对误差 1.0 量级）。
- **写回的四条边要各自带 `oy*WO`**。第一版只给 `y10/y11` 加了
  `(th*2+1)*WO`，`y00/y01` 漏了 `th*2*WO`，于是每个 tile 行都往第 0 行写，
  表现为"第 0 行全错、第 1 行全对"。这种半对半错比全错更难看出来。

**数值**：与四段版**逐位相等**（同一组 `B`/`A`/`G` 系数、同样的求和顺序，
`max|fused - 四段| = 0`）；对 cuDNN 的相对误差 ≤ 2.3e-7，七个 3x3 形状
全部通过 1e-4 验收线。

**实测**（新机 A800-40GB，`execute_device`，驻留输入，burst=20 稳态，ms）

| 形状 | 四段 | 融合 | cuDNN | 改前 auto/cuDNN | 改后 auto/cuDNN |
| --- | --- | --- | --- | --- | --- |
| 3x3_c3 | 0.103 | —（tiled 0.060） | 0.045 | 1.33× | 1.33× |
| 3x3_c8 | 0.102 | 0.064 | 0.064 | 1.65× | **1.00×** |
| 3x3_c16 | 0.103 | 0.066 | 0.065 | 1.63× | **1.01×** |
| 3x3_c24 | 0.103 | 0.065 | 0.064 | 1.63× | **1.02×** |
| 3x3_c32 | 0.108 | 0.107 | 0.065 | 1.65× | 1.66× |
| 3x3_c64 | 0.109 | 0.246 | 0.071 | 1.49× | 1.51× |
| 3x3_c128 | 0.105 | 0.222 | 0.064 | 1.65× | 1.65× |

读法：

- **c8/c16/c24 从落后 1.6× 变成持平**。这三档的四段版都是 0.103，融合后
  0.064～0.066，正好落在 cuDNN 的 0.064 上。
- 融合版的 device 时间比端到端更低（c8 0.020、c16 0.037、c24 0.043 ms），
  剩下的 0.02 是框架派发 + 权重变换那次 launch + 一次 `cp.empty`。也就是
  说 c8～c24 已经不是被 GPU 计算限制，而是被"一次执行至少要付的固定成本"
  限制（约 0.046 ms）。
- **c32 是分界**：每线程要读 `16*C = 512` 次 `U`，`U` 的重复全局读
  （c32 约 268 MB，c64 约 822 MB）把乘加压到不如 cuBLAS 的 batched GEMM。
  c32 融合 0.107 与四段 0.108 打平，c64/c128 明显更差。
- **c3 不该用融合**：device 只要 0.0086 ms，但第一阶段只有 `C*TT = 24` 个
  线程干活、其余一百多个空转，端到端 0.063 反而慢于分块直算的 0.058——
  多出的权重变换 launch 抵掉了省下的提交。

**接管范围（auto）**：`plan.conv_winograd_fused_preferred`，3x3 /
stride=1 / dilation=1 / `8 <= C <= 24` / `TT*O <= 1024` / work >= 4096。
不满足时退回 5.255 的四段版（c32～c128）或分块直算（c3）。显式
`conv_impl="winograd"` 仍是四段版，A/B 对照不用改代码。

**结论与下一步**

- 5.255 猜的两个方向里，"合并三段"确实是一条独立的路：在 c8～c24 上把
  1.6× 的差距直接抹平，且没有动数值（逐位相等）。
- 剩下的 0.046 ms 固定成本里，权重变换那次 launch 约占 0.014（launch
  0.008 + 变换 0.006）。把它挪出去（图级常量折叠或权重驻留）是下一步：
  需要先定"权重没变"怎么证明——当前实现仍然每次重算，因为按指针缓存会
  在权重原地更新时静默用旧值。
- 大通道（c32+）若也要融合，得把 `U` 分块进共享内存并做寄存器复用；
  在那之前四段版的 cuBLAS 更划算。

### 5.257　权重驻留：把 `U` 从每轮执行里拿出来

5.256 结尾留了一个不敢做的东西：权重变换那次 launch 占 0.046 ms 固定成本
里的 0.014，挪出去就能让 c8～c24 反超 cuDNN；但"权重没变"当时**没法证明**，
按指针缓存会在权重原地更新时静默用旧值。这一节先把判据定死，再谈收益。

**判据：看 IR，不看内存**

`torch.export` 的 `get_attr` 与 lifted 张量都落成 IR 的 `constant` 节点
（`frontend/torch_export.py::_constant_node`）；显式构图也可以写
`Graph.constant`。常量节点的语义就是"与运行期输入无关、值不可变"，它的设备
副本由执行计划持有（`_GroupPlan.constants`）、与计划同寿命。因此

> 只有当 conv 的权重输入是一个 `constant` 节点的输出时，`U` 才允许驻留。

这不是"指针看起来没变"——那是猜。权重是图输入时，调用方每次都可能在同一个
设备缓冲里换值，任何基于地址或对象身份的缓存都会把新权重算成旧值，而且没有
任何可观测迹象。判据写成 `_holds_constant_weight(node, constant_ids)`，只看
`node.op.name == "conv2d"`、输入 0 是 f32、输入 1 的 id 在常量集合里。

**实现：一个随计划存活的盒子**

`_prepare_plan` 改两遍扫描：先上传全部常量副本（5.161 的行为不变），第二遍
逐节点绑内核。命中上述判据的 conv，内核被包进 `_resident_conv2d`：

```python
box = []
def call(node, args, attrs):
    return fn(node, args, attrs, resident=box)
```

`box` 由 `group.kernels` 持有，因此**与执行计划同寿命**；计划作废
（`_clear_plan_values` / `_drop_plan` / `graph.revision` 变化）时盒子一起
消失，不可能跨图误用。盒子是惰性填的：计划期还不知道 cuDNN 会不会接管这一
层，白算一次 `U` 只是浪费；第一次真正走融合内核时才 `cp.empty` + launch，
之后每轮只跑融合内核。

盒子里的 `U` 形状或 dtype 与本次不符时重算并替换——那说明权重本身变了，
旧 `U` 不能再用。`DeviceBackend.conv_weight_transforms` 记录权重变换内核
实际被调用的次数，它让"驻留有没有生效"变成可观测的整数，而不是靠计时猜。

**两个边界**

- **图输入权重不驻留**。基准脚本的常见形态（`build` 把 `w` 当图输入）因此
  在「输入权重」一列仍然是每轮一次 `U`，与 5.256 完全一致。
- **捕获图里仍有一次 `U`**。`capture` 前的 `_drop_plan` 让常量副本重新落在
  捕获 arena 里，盒子也随之重建为空，于是捕获期那次执行会把权重变换写进图，
  replay 时它跟着重放。这不影响正确性，只是 `graph_mode` 下省不掉这一次
  launch；`capture` 与 `_drop_plan` 的地址语义（见 `_GroupPlan.constants`）
  不能为了这一点微调去动。

**数值**：常量权重与图输入权重两条路径**逐位相等**（同一内核、同一组系数）；
对 cuDNN 的归一化最大误差 ≤ 5.3e-7，六个 3x3 形状全部通过 1e-4 验收线。

**实测**（A800-40GB，`execute_device`，burst=20 稳态，三轮取最好，ms；
`U/20` 是预热后连续执行 20 次里权重变换的调用次数）

| 形状 | tiled | 输入权重 | 常量权重 | cuDNN | 常量/cuDNN | U/20 输入 | U/20 常量 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3x3_c3 | 0.0589 | 0.0636 | 0.0494 | 0.0443 | 1.12× | 20 | 0 |
| 3x3_c8 | 0.1331 | 0.0633 | 0.0505 | 0.0646 | **0.78×** | 20 | 0 |
| 3x3_c16 | 0.2538 | 0.0640 | 0.0509 | 0.0644 | **0.79×** | 20 | 0 |
| 3x3_c24 | 0.3691 | 0.0647 | 0.0513 | 0.0637 | **0.81×** | 20 | 0 |
| 3x3_c32 | 0.4780 | 0.1274 | 0.0908 | 0.0651 | 1.39× | 20 | 0 |
| 3x3_c64 | 1.3653 | 0.3118 | 0.3075 | 0.0805 | 3.82× | 20 | 0 |

读法：

- **c8～c24 从 1.00× 变成 0.78～0.81×**，也就是在同一台机器上反超 cuDNN
  约 20%。省下的 0.013～0.014 ms 与 5.256 对那次 launch 的估算吻合。
- `U/20` 两列是 20 与 0 的对照：常量权重在预热之后一次都没有再算 `U`，
  输入权重每轮都算。表格与计数互相印证，不需要相信任何一个数字。
- **c3 顺带被带动**：常量权重把下限从 8 放到 3（`conv_winograd_fused_preferred`
  的 `min_c` 分档），auto 在 c3 上从分块直算的 0.0589 变成 0.0494，对 cuDNN
  从 1.33× 收到 1.12×。图输入权重仍是 0.0636，所以这条放宽只跟着"权重是
  常量"走，不是无条件放宽。
- **c32/c64 只是略好**：驻留省掉的是固定 launch，而这两档的瓶颈是每线程
  `16*C` 次 `U` 的重复全局读（5.256 已测：c32 约 268 MB、c64 约 822 MB），
  与那次 launch 无关。要动它们得走"`U` 分块进共享内存 + 寄存器复用"。

**接管范围**：auto 的融合判据与 5.256 相同，唯一变化是权重为常量时下限降到
c3；显式 `conv_impl="winograd_fused"` 新增，适合做 A/B 对照，共享内存预算
（`16*C*TT*4 <= 48 KB`，即 C ≤ 96）在派发处显式检查，超了直接报错而不是
启动失败。

**可复现**：`python -m benchmarks.winograd_compare --rounds 3`。四列
（tiled / 输入权重 / 常量权重 / cuDNN）共用同一份数据（生成顺序必须与
`cudnn_compare.build` 一致，否则数值对拍会以"算法错误"的形式失败），末尾两列
直接打印 `U/20`。

### 5.258　c32 以上：`U` 分块进共享内存试过，但没有赢

5.257 结尾留的那条路是"`U` 分块进共享内存 + 寄存器复用"。这一节把它做完，
结论是**不做**：改完的版本在 c24 上慢 1.9 倍，在 c32/c64 上与改之前持平，
所以代码回退，只留下这段记录，避免以后重走。

**要解决的问题是怎么算出来的**

融合版的第二阶段里，每个线程负责一个 `(输出通道, tile)`，沿 `C` 把
`M[p] += U[p][oo][c] * V[p][c][lt]` 累起来。`U` 原先由每个线程各自从全局读，
所以同一个 `U[p][oo][c]` 被同一 warp 里 `TT = 8` 个 tile 线程各读一遍：整块
`16*C*TT*O` 次。c32 约 268 MB、c64 约 822 MB——数字很吓人，于是一定是瓶颈，
这就是 5.256/5.257 里的假设。

**做法**

按 `CB` 个输入通道分块，`V` 与 `U` 都只保留一块在共享内存里，沿 `C` 逐块
累加寄存器里的 16 个 `M`：

```text
for (c0 = 0; c0 < C; c0 += CB) {
  V 分块 → sv[(p*CB + lc)*TT + lt]        // CB*TT 个线程位
  U 分块 → sv[usv + (p*O + oo)*(CB+1) + lc]  // 整块摊 16*O*CB 个元素
  __syncthreads();
  for (lc = 0; lc < CB; ++lc) for (p = 0; p < 16; ++p) acc[p] += U*V;
  __syncthreads();                         // 下一块要覆盖这块
}
```

共享内存是 `16*CB*TT + 16*O*(CB+1)` 个 float，`CB` 由 48 KB 反解成 2 的幂
（`16*CB*(TT+O) + 16*O <= 12288`）：O=24 时 CB=16、O=32 时 CB=16、O=64 时 CB=8。
`U` 每行多一格填充，避免 `CB` 为 2 的幂时整行落在同一个 bank。求和顺序逐字
不变，数值与改之前逐位相等（`U/20` 仍是 0，对拍全部通过）。

**实测**（A800-40GB，`winograd_compare --rounds 3`，常量权重，ms）

| 形状 | 改之前 | 分块版 | 四段版 |
| --- | --- | --- | --- |
| 3x3_c8 | 0.0505 | 0.0536 | 0.103 |
| 3x3_c16 | 0.0509 | 0.0540 | 0.103 |
| 3x3_c24 | 0.0513 | **0.0958** | 0.103 |
| 3x3_c32 | 0.0908 | 0.0972 | 0.105 |
| 3x3_c64 | 0.3075 | 0.3016 | 0.112 |

c24 的 1.9 倍退化最说明问题：那里 `U` 一共只有 16*24*24*4 = 36 KB、本来就能
被 L1 装住；分块反而把每块的共享内存从 12 KB 抬到 34 KB，一个 SM 能同时驻留
的块数上限从十余块掉到 4 块，另外还多了一对同步。c8/c16 的 5% 也是同一个
方向。c32/c64 持平说明的是另一件事：**把 `U` 的重复读砍掉 8 倍之后时间一点
没动**，那就不是访存带宽的问题，`U` 的重复读本来就被 L1/L2 吸收了；再改 `U`
的布局或读取顺序（另一条候选）也不会赢，因为瓶颈不在这里。

**c32 到底差在哪**

同一进程里交替跑 6 轮、每轮 20 次 burst 取中位数（这台机器与另一路任务共用，
绝对值的抖动有 ±25%，所以只看同轮内的相对关系）：

| 实现 | 稳态中位数 |
| --- | --- |
| 融合版（常量权重） | 0.1217 |
| 四段版 | 0.1046 |
| cuDNN | 0.065 |

融合版在 c32 输给四段版，不是输在 `U`：把主机提交与设备执行分开量，两条路
的主机提交都是 ~50 µs，融合版的设备部分 ~65 µs、四段版 ~55 µs。c32 这一档
已经落在"提交 + 派发"的量级上（5.255 的结论），要进 1.1× 得把设备侧压到
20 µs 以下，靠调整访存做不到。

**判据不变**：`conv_winograd_fused_preferred` 的上限仍是 24。中途试过把常量
权重的上限放到 32（依据是一轮里 0.0972 对 0.1053），交替复测后中位数反过来，
所以没有保留——单轮 burst 的 8% 差距在这台机器上不可信。

### 5.259　层 / 网络级：conv-bn-relu 多节点图，捕获没有收益，逐通道 BN 亏 27%

前几节全部是**单个卷积**。这一节换成真实的层结构，同时回答两个此前一直靠
推断的问题：逐元素融合实际把派发压到了几次、CUDA Graph 捕获在这类图上还能
省多少。复现：`python -m benchmarks.conv_graph_compare --blocks 4 --rounds 3`。

**图**：`blocks` 层 `conv2d → mul → add → relu`（推理形态的 BN 就是"逐通道
仿射 + relu"；这份 IR 里的 `batch_norm` 是训练形态，会引入沿空间轴的归约，
不代表推理）。卷积权重、bias、BN 参数全部是 `constant` 节点，所以权重驻留与
融合版 Winograd 都生效（C=16 在 8～24 窗口内）。数据 4×16×56×56。

三个用例只差 BN 参数怎么给：

- `scalar`：`mul`/`add` 的一侧恰好一个元素，可以进融合链；
- `channel4d`：参数 `(1,C,1,1)`，逐通道广播；
- `channel`：参数 `(C,)` 经 `reshape` 成 `(1,C,1,1)`，与前端导入形态一致。

**分组**（`DeviceBackend.group_nodes`，与 `graph_mode` 无关）

`scalar` 在 4 层时是 32 个节点 → 24 组，每个 block 的 `mul+add+relu` 合成一组，
这是融合按设计生效的直接证据（4 组融合链）。`channel4d` 是 32 个节点 →
32 组、`channel` 是 40 个节点 → 40 组，两者都**没有一组融合**：`_chainable`
只认"整块同形"或"一侧恰好一个元素"，`(1,C,1,1)` 不满足，于是 `mul`、`add`
各自一个内核。

**设备侧耗时**（`execute_device`，20 次 burst，3 轮取最好，ms）

| blocks | 用例 | 节点 | 组 | `graph_mode=off` | `auto` | 加速 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | scalar | 8 | 6 | 0.1226 | 0.1227 | 1.00× |
| 1 | channel | 10 | 10 | 0.1642 | 0.1649 | 1.00× |
| 1 | channel4d | 8 | 8 | 0.1500 | 0.1505 | 1.00× |
| 4 | scalar | 32 | 24 | 0.4595 | 0.4545 | 1.01× |
| 4 | channel | 40 | 40 | 0.6256 | 0.6258 | 1.00× |
| 8 | scalar | 64 | 48 | 0.8931 | 0.8853 | 1.01× |
| 8 | channel | 80 | 80 | 1.2368 | 1.2319 | 1.00× |
| 8 | channel4d | 64 | 64 | 1.1306 | 1.1330 | 1.00× |

**结论一：捕获的收益取决于每节点的工作量，这张图上没有。** 5.251 的
"3.4～6×"是在小算子的图上测的：那里 Python 派发比 GPU 慢，去掉派发就等于
提速。这里每个 block 有 0.11 ms 的卷积，8 层 0.89 ms 里派发只占一小部分，
而 `execute_device` 是异步提交、本来就与设备侧重叠。所以 1.00× 不是"捕获
失效"，是"这张图上没有可省的主机时间"——两件事不能混为一谈，README 里
"多节点图收益显著"这句也因此加了限定。

**结论二：逐通道 BN 不进融合链，代价是每层 27%～39%。** 8 层的每层均值：
`scalar` 0.112、`channel4d` 0.141（+27%）、`channel` 0.155（+39%）。差的
就是那两个广播内核（`channel` 再加两个 `reshape` 节点）。而逐通道仿射正是
BN 在推理时的**常见**形态，所以这条判据的缺口不是边角：把"沿某一维广播、
其余维相同"纳入 `_chainable`，并让链内核按输入各自算索引，能把这两次派发
省掉。本轮没有做——它要改的是 `lower/elementwise.build_chain` 的索引生成，
而不是调度决策，属于单独一块工作，先记在这里。

### 5.260　运行时固定开销：把静态 conv 的 launch 在计划期展开

前几节反复出现同一个判断："c16 已经追平 cuDNN，剩下的 50 µs 是主机提交"。
这一节把它拆开，确认这 50 µs 具体由哪几块构成，并改掉其中最大的一块。
复现：`python -m benchmarks.overhead_compare --cases single,block,wide`。

**七层口径**（`benchmarks/overhead_compare.py`）

| 指标 | 含义 |
| --- | --- |
| `burst` | 20 次背靠背提交 + 一次同步，摊薄后的端到端（框架 + 内核） |
| `host` | 同样 20 次提交但不带同步，只量主机走完 `execute_device` 的时间 |
| `kernel` | 跳过执行骨架，直接调计划里已绑定的内核函数 |
| `skel` | `host - kernel`，执行骨架（环境维护、输入校验、回收、计划查找） |
| `empty` | 同形状 `cp.empty` 的单次成本 |
| `pack` | 卷积 launch 那一组 `np.int32` 的构造成本 |
| `cap` | `graph_mode="auto"` 的 `burst`，逼近纯设备时间 |

**改动前**（A800-40GB，ms）

| 用例 | 节点 | 组 | `burst` | `host` | `kernel` | `skel` | `empty` | `pack` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 单 conv c16 | 3 | 3 | 0.0537 | 0.0527 | 0.0321 | 0.0206 | 0.0028 | 0.0052 |
| 8 层 conv-bn-relu | 64 | 48 | 0.9191 | 0.9157 | — | — | 0.0028 | 0.0052 |
| 单 conv c64 | 3 | 3 | 0.1127 | 0.1077 | 0.0859 | 0.0217 | 0.0028 | 0.0052 |

单 conv 的 52.7 µs 主机构成是：骨架 20.6 µs、内核函数 32.1 µs（其中输出分配
2.8 µs、参数打包 5.2 µs、其余是 `_op_conv2d` 的解析与 cupy 的 launch）。
`cProfile` 的排序与此一致：`run_device` 与 `_op_conv2d` 各占约六分之一，
其后是 `_conv2d_winograd_fused`、`dict.get`、`_bind`、`_pair`。

关键判断：**这 32 µs 里真正与"算"有关的部分很少**。形状解包、`_pair`、
`_pick_conv_kernel`、`_kind`、`_kernel` 查表与十几个 `np.int32` 的打包，
在静态图上只取决于 IR，与本次绑定值无关，却每步重做一次。

**改动**

1. `_planned_winograd_fused`（`backends/device_backend.py`）：对"权重来自
   `constant`、`auto` 判据选中融合 Winograd"的静态 3x3，在 `_prepare_plan`
   里把形状、线程铺法、`shared_mem` 与标量参数一次算好，闭包运行期只做
   `cp.empty` + 一次内核调用。首次执行仍走原实现算 `U` 并装进计划持有的
   盒子；形状、dtype 或连续性不符时回落 `_op_conv2d`。实现选择仍只由
   `_pick_conv_kernel` 这一份判据给出，没有复制阈值。
2. `Executor.run_device`：常量组提前 `continue`。它的值在计划期已上传且
   不会被回收，每轮只需要写回环境；此前它仍要过 args 构造、`resolve`、
   形状校验与 cast。

**复测**（同一台机器、同一进程交替）

| 用例 | `burst` 前 → 后 | `host` 前 → 后 | `kernel` 前 → 后 |
| --- | --- | --- | --- |
| 单 conv c16 | 0.0537 → **0.0265** | 0.0527 → 0.0253 | 0.0321 → 0.0140 |
| 8 层 conv-bn-relu | 0.9191 → **0.6816** | 0.9157 → 0.6769 | — |
| 单 conv c64 | 0.1127 → 0.1128 | 0.1077 → 0.1030 | 0.0859 → 0.0849 |

`winograd_compare` 的常量权重列（另一套图与口径，不能与上表混比）：

| 形状 | 改前 | 改后 | cuDNN | 改后 / cuDNN |
| --- | --- | --- | --- | --- |
| 3x3_c3 | 0.0494 | 0.0278 | 0.0443 | 0.63× |
| 3x3_c8 | 0.0505 | 0.0288 | 0.0641 | 0.45× |
| 3x3_c16 | 0.0509 | 0.0391 | 0.0651 | 0.60× |
| 3x3_c24 | 0.0513 | 0.0476 | 0.0631 | 0.75× |
| 3x3_c32 | — | 0.1156 | 0.0641 | 1.80× |
| 3x3_c64 | — | 0.3430 | 0.0803 | 4.27× |

**结论**

- 窄通道 3x3 不再只是"追平"：c8 是 cuDNN 的 0.45×、c16 是 0.60×。原先
  "追平"的表里混着 20 µs 的框架税，去掉之后才看到内核本身的平均水平。
- 8 层图从 0.92 ms 降到 0.68 ms（-26%），且与捕获后的 0.676 ms 基本重合
  ——这张图上已经几乎没有可省的主机时间。
- c32/c64 基本不动（-4%）：它们走**四段版** Winograd，本轮没有展开。
  这两档的差距现在可以确定是设备侧三段内核 + batched GEMM 的结构问题，
  而不是派发。下一步若要继续压，应当先展开四段版的 launch 再改内核，
  否则会把主机成本误判成内核成本。
- 未做：逐通道 BN（`(1,C,1,1)`）进融合链，仍记在 5.259。
- 预解析只覆盖"常量权重 + 融合 Winograd"。图输入权重的 conv 仍每步走
  `_op_conv2d`：那里没有可证的"值没变"，缓存 `U` 不安全，但**解析与选择**
  本身与权重无关，后续可以按 IR 缓存这部分而只让 `U` 每轮重算。

### 5.261　逐元素融合链也在计划期展开

5.260 之后单 conv 已经到设备时间（`gpu 0.0248` ≥ `host 0.0249`），但 8 层
图还有 0.68 ms。用 `overhead_compare` 的 `gpu` 列（CUDA events）一测就分明：

| 用例 | `host` | `gpu` | 每层 |
| --- | --- | --- | --- |
| 8 层纯 conv | 0.155 | 0.155 | 19.4 µs |
| 8 层纯逐元素（标量，融合成 1 内核/层） | 0.493 | 0.494 | 61.7 µs |
| 8 层 conv-bn-relu | 0.675 | 0.670 | 84.4 µs |

`host == gpu` 且远超内核应有的时间，说明 GPU 在空转等主机——每层 60 µs 不是
算出来的，是 `_try_chain` 每步重新推导参数顺序、广播形状、线程铺法并组装
源码键花掉的。一个 200704 元素的融合内核本身只要 2 µs 量级。

**改动**

- `_planned_chain`（`backends/device_backend.py`）：静态逐元素组在
  `_prepare_plan` 里把 entries 顺序、`specs`、`out_shape`、`vec/block/grid`
  与源码键一次算好。运行期只核对每个外部数组的形状 / dtype / 连续 / 16 字节
  对齐，然后 `cp.empty` + 一次 launch；任一不符返回 `None`，回落 `_try_chain`。
  内核仍惰性获取（计划期只建键），首次执行才编译，之后句柄留在闭包里。
- `_GroupPlan.chain`（`runtime/executor.py`）：计划期展开结果的槽位；
  `DeviceBackend.invoke_group` 优先用它，未命中再走 `_try_chain`。

**复测**（A800-40GB，ms）

| 用例 | 5.260 后 | 5.261 后 |
| --- | --- | --- |
| 8 层纯逐元素 | 0.493 | **0.146** |
| 8 层 conv-bn-relu（`scalar`） | 0.682 | **0.303** |
| 8 层 `channel4d`（逐通道 BN） | 0.891 | 0.891 |
| 4 层 `scalar` | 0.346 | **0.157** |

`conv_graph_compare` 的 8 层 `scalar` 从 5.259 的 0.919 降到 0.298（-68%），
每层从 115 µs 降到 37 µs：19 µs 卷积 + 18 µs 融合链。数值与逐节点路径一致
（`test_cuda_backend + test_conformance + test_plan` 448 passed）。

**两个发现**

- `graph_mode="auto"` 在这些图上**从未捕获成功**：`_graph_failure` 返回
  `RuntimeError: the current stream is capturing, so H2D transfers are
  disallowed`。所以 5.259 记的"捕获收益 1.00～1.01×"其实是"回落到了逐节点
  路径"，不是"设备已饱和"。捕获期还在做 H2D 的入口需要单独排查；修好后
  多节点图每层还能再省下这 18 µs 的提交。
- 逐通道 BN（`(1,C,1,1)`）仍然进不了融合链，`channel4d` 8 层 0.891 ms 是
  `scalar` 的 3 倍——每层 3 个独立逐元素内核，每个都要付一次解析与 launch。
  这仍是当前层/网络级最大的单点缺口（5.259 已记为待做；5.262 解决）。

### 5.262　逐通道广播进融合链

5.261 把融合链的提交成本压到 18 µs/层之后，`channel4d` 的 0.905 ms 与
`scalar` 的 0.303 ms 之间那 0.6 ms 就全是"没融合"的代价：`_chainable`
只认同形或单元素，`(1,C,1,1)` 让 `mul` / `add` 各自成组，`relu` 也被
拆在外面——8 层图的组数等于节点数。

**改动**

- `ElemInput.channel = (C, H*W)`（`lower/elementwise.py`）：逐通道输入的
  第 `i` 个展平输出元素取 `(i / (H*W)) % C` 号元素。`channel_broadcast`
  只认"右对齐后只有通道维非 1"的 `(1,C,1,1)` / `(C,1,1)`；`(B,C,1,1)`
  这类随 batch 变化的形状仍然不融。
- `build_chain` 的两条路径：`vec=1` 直接按 `i` 取通道；`vec=4` **不能**
  把逐通道输入一次读成 `float4`（四个连续元素可能跨通道边界），改为每个
  lane 单独取索引。正确性优先，带宽由 L2 兜住。
- 源码键加入 `channel`：同一个链形状在 `H*W` 或 `C` 变化后生成的索引
  表达式不同，只按名字与 `scalar` 复用会拿错内核（这一点在第一次实测时
  直接表现为 73% 元素不匹配）。
- `_chainable` 接受逐通道形状，判据与 `build_chain` 共用同一个
  `channel_broadcast`，没有第二份阈值。

**复测**（A800-40GB，8 层，ms）

| 用例 | 5.261 | 5.262 |
| --- | --- | --- |
| `scalar`（标量 BN） | 0.2986 | 0.2986 |
| `channel4d`（`(1,C,1,1)`） | 0.9055 | **0.3034** |
| `channel`（`(C,)` + 两次 reshape） | 1.0215 | 0.4020 |

`channel4d` 与 `scalar` 已经持平（差 1.6%），组数从 64 回到 48。数值与参考
后端最大绝对差 2.4e-7（f32 的 relu 边界）。全量 `pytest tests` 为
2572 passed / 53 failed，失败全部来自 `frontend/compat.py` 对 torch 2.5.0
的版本拒绝（矩阵是 2.9–2.13），与本轮改动无关。

**剩余缺口**：`channel` 仍比 `channel4d` 多 0.1 ms（每层 2 个 `reshape`
组）。`reshape` 在设备端本可以是零拷贝视图，现在仍按一次独立提交计价，
是下一处值得看的主机成本。

### 5.263　常量上的 reshape 在计划期折成视图

5.262 之后 `channel` 的 0.402 ms 里还剩每层两个 `reshape` 组的提交成本
（16 个节点 / 8 层）。这两个 `reshape` 的输入都是 `constant`——BN 参数在
`(C,)` 上生成，再 reshape 成 `(1,C,1,1)` 以右对齐到 NCHW 的通道维。

`reshape` 不读也不写数据，只换元数据。既然源值是计划期已经上传的设备常量，
"运行期再 reshape 一次"与"计划期 reshape 好、运行期直接取用"逐位相同；区别
只在后者省掉一次派发、一次形状比对和一次写回环境。判据因此只有一条：**源值
已经是计划里的常量，且目标形状全为字面整数**。含符号维的目标随绑定变化，
仍然留给运行期解析。

**改动**（`backends/device_backend.py`）

- `_prepare_plan` 的第一遍改成顺序扫描并维护 `constant_values`（值 id → 设备
  数组），`constant` 与折叠后的 `reshape` 都写进 `group.constants`、同时加入
  `constant_ids`。后者让折叠产物作为 conv 权重时自然也走权重驻留（5.257）。
- `_fold_constant_reshape`：只认单输入、已折叠的源、字面整数目标形状。返回值
  是与 `_op_reshape` 同一条数据路径的视图。不登记 arena 别名——常量在计划存活
  期内不会被回收，源数组由 `constant_values` 与 `group.constants` 双双持有。
- 运行期没有新增分支：折叠后的组就是普通常量组，`run_device` 里既有的
  "常量组提前 `continue`"直接覆盖，`_fast_step` 也因 `output.id in constants`
  返回 `None`。

**复测**（A800-40GB，8 层，ms）

| 用例 | 5.262 | 5.263 |
| --- | --- | --- |
| `scalar`（标量 BN） | 0.2986 | 0.2963 |
| `channel4d`（`(1,C,1,1)`） | 0.3034 | 0.2977 |
| `channel`（`(C,)` + 两次 reshape） | 0.4020 | **0.3020** |

三种形态已经齐平（差 ≤ 2%），`channel` 相对 5.259 的 1.0215 降 70%。2 层
对照参考后端的最大绝对差 9.5e-7（与 `channel4d` 的 9.5e-7 同源，来自卷积），
`channel` 的常量组从 8 增到 12（每层两个 `reshape` 折叠）。全量 `pytest tests`
仍为 2572 passed / 53 failed（torch 2.5.0 版本拒绝，与本轮无关）。

### 5.264　四段版 Winograd 的 launch 也在计划期展开

5.260 只展开了融合版（C ≤ 24）；C ≥ 32 走四段版，每次执行仍要重新解形状、
查三次 `_grid` / `_kernel`、打包二十多个 `np.int32`，并做三块中间缓冲的
`cp.empty`。`overhead_compare` 在 c64（1×64×56×56，3x3 pad1）上把它量化得很清楚：

| 口径 | 5.263 | 含义 |
| --- | --- | --- |
| `burst` | 0.1127 | 端到端（20 次提交 + 同步） |
| `gpu` | 0.1120 | CUDA events 设备时间 |
| `host` | 0.1051 | 只走完主机侧提交 |
| `kernel` | 0.0853 | 直接调已绑定内核函数 |
| 分段合计 | 0.0443 | weight 0.003 + input 0.005 + gemm 0.032 + output 0.005 |

设备侧内核合计只有 0.044，`kernel` 却要 0.085——差的那 40 µs 不在内核里，
而在"每次执行重新解析 + 分配 + 打包"。所以这一轮仍然只动主机，不碰内核。

**改动**（`backends/device_backend.py`）

- `_winograd_static(node, kinds)`：形状校验、cuDNN 让位判据与
  `_pick_conv_kernel` 的调用抽成一份，融合版与四段版共用；`kinds` 是调用方
  愿意展开的实现名，显式 `conv_impl` 直接当实现名，只有 `auto` 才问判据。
- `_planned_winograd(node)`：计划期算好三段 kernel 的 `grid` / `block`、
  静态标量参数与 GEMM 的 `(m, n, k, lda...)`，内核句柄在第一次执行时懒加载
  后缓存在闭包局部（与 `_planned_chain` 同一手法，避免计划构建期触发 NVRTC）。
  运行期只剩：形状 / dtype / 连续性核对 → 三次 `cp.empty` → 两次 launch +
  一次 `sgemm_batched`。不缓存 `v` / `m`：缓冲随执行分配，多图或并发执行时
  互不干扰的语义不变。
- 权重驻留：`U = G g G^T` 只依赖权重，而这条展开只在"权重是常量"的分支里
  被采用（调用点即判据）。首次执行算一次 `U` 后随计划常驻，之后每轮跳过权重
  变换的 launch 与一次分配。`v` / `m` 是写满的内核输出，不驻留——复用会让
  同一张图的并发执行互相覆盖。

**复测**（A800-40GB，c64，ms，同一次 `overhead_compare` 内相对值）

| 口径 | 展开前 | 展开后 |
| --- | --- | --- |
| `burst` | 0.1127 | **0.0942** |
| `host` | 0.1051 | 0.0624 |
| `gpu` | 0.1120 | 0.0878 |
| `kernel` | 0.0853 | 0.0423 |

从 0.0853 到 0.0510 是 launch 展开，到 0.0423 是把 `U` 驻留之后——已经贴近
三段内核 + GEMM 的分段合计 0.0443，剩下的只有分配本身。`burst` 降 17%，
`kernel` 降 50%；`gpu` 也跟着降，因为主机提交不再让设备空等。
数值：c32 对参考后端最大绝对差 1.2e-6、c64 为 2.1e-6（Winograd 变换本身的
舍入，与四段版原路径一致）；同一输入连续执行三轮，输出逐位相同（`U` 驻留
没有把上一轮的状态带进来）。`burst` 对共享机上的其它负载很敏感（同一份代码
复测到过 0.2173），`host` / `gpu` / `kernel` 则稳定复现为 0.062 / 0.088 /
0.042——判断这一层优先看后三列。

**未采信的比值**：测这段时共用机器上有别的负载，`winograd_compare` 的 cuDNN
基准列从 0.080 漂到 0.180，c8/c16 这类未改动路径的比值也一起翻倍，说明整轮
不可比。设备侧仍是主体（`gpu 0.088` vs `host 0.062`），真正的 cuDNN 差距
要等机器空闲后按同进程交替重测。

### 5.265　捕获的静态输入拷贝要与预热流建立依赖

`tests/test_cuda_backend.py` 有一个既存的顺序依赖：整文件跑时
`TestCudaGraph::test_replay_reports_out_of_range_ids` 必失败，单跑必过；切到
5.262 的代码同样复现，所以与本轮的三个性能改动无关，但它是真缺陷。

**定位**：在 `_op_embedding` 的越界分支前打印实际下标，失败时 ids 里是一串
随机浮点（`1058314997` 之类），而不是传入的 `[0,1,2,3,4]`。也就是说
"越界"是内核读到未初始化内存后的误报。

**根因**：`capture` 为静态输入分配缓冲后，用 `buf[...] = src` 从主机数组拷贝
（默认流上的异步 H2D），紧接着在 `self.new_stream()`——一条非阻塞流——上跑
预热。两条流之间没有任何依赖，预热可能先于拷贝完成。单跑时默认流刚创建、
拷贝瞬间完成，所以看不到；整文件跑时前面的用例把默认流排满，竞态就稳定暴露。

**修复**（`backends/cuda/backend.py`）：待填的缓冲先记进 `pending`，在**换
分配器之前**统一拷贝，然后 `cp.cuda.Event().record()` + `stream.wait_event()`
把捕获流挂到这条拷贝之后。放在换分配器之前还有一个副作用是好的：这次 H2D
的临时分配不落进捕获 arena，`capture_reports_memory_budget` /
`static_budget_matches_captured_peak` 的预算只反映图本身。

**结果**：`tests/test_cuda_backend.py` 连跑三次都是 339 passed / 5 skipped；
全量回到 2572 passed / 53 failed（差集仍是 torch 2.5.0 版本拒绝）。

### 5.266　捕获前的计划重建不能落在捕获流上

5.261 记下过一个更根本的问题：`graph_mode="auto"` 在 conv-bn-relu 这类图上
**从未捕获成功**，`graph_capture_failure` 一直是
`RuntimeError: the current stream is capturing, so H2D transfers are disallowed`。
5.259 那张"捕获 1.00×"的表其实是回落逐节点后的结果。

**根因**：`capture` 的预热轮结束后会 `_drop_plan(graph)`，让常量在捕获轮重新
上传到本次捕获的 arena（预热轮的地址会被回收，不能复用）。但重建发生在捕获
开始**之后**——捕获轮第一次 `run_device` 才走到 `_plan` → `_prepare_plan` →
`_op_constant`，那一遍 `cp.array` 是 `cudaMemcpyAsync` 的 H2D，而捕获流上
任何 H2D 都被驱动拒绝，整张图随即被记成失败。

**修复**（`backends/cuda/backend.py`）：`_drop_plan` 之后、`begin_capture` 之前
显式调一次 `self._plan(graph)`，让计划重建与常量上传在默认流上完成；随后
`cp.cuda.Event().record()` + `stream.wait_event()` 把捕获流挂到它后面。图里
读到的因此是**本轮 arena** 的那份常量，捕获流内不再出现任何 H2D。HIP 侧不动：
它的流捕获实现把捕获期 H2D 的主机源记进 `session.host_refs` 并随计划持有，
语义本来就成立。

**结果**（A800-40GB，8 层 conv-bn-relu，C=16，56×56，batch 4）

- `capture_failure` 从那条 RuntimeError 变为 `None`，`_graph_plans` 里留下 1 个
  计划；输出与 `graph_mode="off"` 逐位相同（最大绝对差 0.0）。
- 同进程交替测量（每轮各自同步）：`scalar` 形态 off 0.338 ms → auto 0.201 ms，
  1.68×。`channel4d` / `channel` 当轮不可比：共享机上另一个进程长期占着 91%
  的 SM（13.8 GB），两种模式都被拖到 2.6 ms 上下，属负载而非代码差异，待机器
  空闲后复测（与 5.264 的 cuDNN 基准列同一批待办）。
- 纯图 replay 的主机提交约 0.016 ms，设备侧单次约 0.16～0.39 ms；图的收益
  仍取决于图内 kernel 本身，捕获只消掉逐节点的 Python 与 launch 开销。

**回归**：`tests/test_cuda_backend.py` 339 passed / 5 skipped；本项只改动
CUDA `capture` 的一段流程，不涉及参考后端与 HIP。

### 5.267　把 clamp / step / spike 接进逐元素链

链的成员以前是"L2 里已有一元/二元表达式表的算子"，`clamp`、`step`、`spike`
被挡在外面：它们有一个**标量参数**，而链的 `ElemStep` 只描述"算子 + 张量
操作数"。结果 SSM 的 `clamp(x, min=-80) → mul → add`、SNN 的
`step(x, 0.5) → mul` 都要各写一次内存。

这三个算子的参数都来自节点属性，是**编译期常量**，可以直接内联成表达式：

- `ElemStep` 增加 `params: Tuple[Optional[float], ...]`（默认空）；
- `apply_op(op, args, params)` 新增两个分支。`clamp` 生成
  `Cond(x > hi, hi, x)` 再 `Cond(x < lo, lo, x)`——先上界后下界，与
  `kernels.ew_clamp` 同序，**NaN 因此原样透传**（用 `fminf` / `fmaxf` 会
  把 NaN 换成边界）；`step` / `spike` 生成 `Cond(x > thr, 1.f, 0.f)`，
  默认阈值分别 0.0 / 1.0，与 `ir/ops.py` 的两张参考表一致。
- 参数进源码缓存键（`(out, op, args, params)`）。这不是洁癖：同一张图里
  `clamp(min=-80)` 与 `clamp(0, 1)` 很常见，只按算子名复用内核会静默换掉
  边界；`step` 的旧内核把阈值当运行期标量，所以以前不存在这个问题。
- `_group_elementwise` 不需要改：`clamp` 是一元、输出与输入同形同 dtype，
  已经满足 `_chainable` 的全部条件。`_step_params` 负责从属性取参数。

**验证**（A800-40GB 真机）

- `step(0.5) → clamp(-1,1) → mul` 被分到**一组**（f32 与 f16 相同），与
  numpy 手算逐位一致（最大绝对差 0.0）；
- 单边 `clamp(min=0)` 把 `[-1, 0.5, NaN, 3]` 映射成 `[0, 0.5, NaN, 3]`——
  NaN 透传成立；`spike` 默认阈值把 `[0.5, 1.0, 1.5, NaN]` 映射成
  `[0, 0, 1, 0]`（`>` 而非 `>=`，NaN 比较为假取 0），与参考实现一致。
- 新增 3 条纯 Python 用例（两组源码形态、两条分组判据）；远端
  `tests/test_cuda_backend.py` + `tests/test_lower_kernel.py`
  428 passed / 5 skipped。

**回归**：两个受影响文件全绿；本轮未跑全量（改动的两个模块不在
`frontend/compat.py` 的 53 条失败面里）。

### 5.268　c32/c64 对 cuDNN 的真实比值：auto 早就在赢了

5.257～5.264 的表里一直有一列"c32/c64 对 cuDNN 的 1.80×/4.27×"，并注明
"共享机上有负载、待空闲后同进程交替重测"。这一节把那次重测定下来，结论与
预期相反：**那两个比值不是 auto 的表现**。

**口径错在哪**：`benchmarks.winograd_compare` 的"常量权重"列显式构造
`CudaBackend(conv_impl="winograd_fused")`，而 `conv_winograd_fused_preferred`
的上限是 C=24——c32 以上根本不在 auto 的接管面里。把这一列当成"框架 vs
cuDNN"来读，等于拿一条只用于 A/B 对照的实验路径去代表 `conv_impl="auto"`。

**复测**（A800-40GB 空闲，同进程、每轮把四路交替跑一遍、6 轮取中位数，
每轮 `measure(burst=20)`；ms）

| 形状 | 融合版 | 四段版 | `auto` | cuDNN | auto/cuDNN |
| --- | --- | --- | --- | --- | --- |
| 3x3_c32 | 0.0938 | 0.0654 | 0.0655 | 0.0639 | **1.02×** |
| 3x3_c64 | 0.3071 | 0.0889 | 0.0887 | 0.0654 | **1.36×** |

`auto` 与显式四段版逐位对齐（0.0655 / 0.0887），说明派发判据在 c32 以上
选的就是四段版。把全部实现放进同一轮再测一次（4 轮中位数）：

| 形状 | im2col | tiled | implicit | winograd | winograd_fused | auto | cuDNN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3x3_c32 | 0.2118 | 0.4890 | 0.3107 | 0.0674 | 0.0912 | 0.0681 | 0.0649 |
| 3x3_c64 | 0.2489 | 1.3708 | 0.8809 | 0.0885 | 0.3064 | 0.0885 | 0.0659 |

两个形状上 auto 都是全实现里最快的，没有"判据漏选"的空间。

**融合版与四段版的分界正好在 24/32 之间**：c24 融合版 0.0372、四段版 0.103；
c32 融合版 0.0938、四段版 0.0654。`conv_winograd_fused_preferred` 的上限
24 因此不需要动——5.258 试过放到 32，当时"交替复测中位数反过来"，现在有了
空闲机上的四段版展开后数据（5.264 之前四段版还背着每轮 40 µs 的主机成本），
可以确认那次判断是对的。

**剩余真实差距**只有 c64 的 1.34×（0.0885 对 0.0659）。0.0885 里输入/输出
变换各约 0.005～0.02、GEMM 是主体（411 MFLOP 当量，折 4.6 TFLOPS f32，A800
峰值 19.5）。再压只有两条路：减少乘法量（F(4x4,3x3) 的变换矩阵更大、数值
更差）或走 TF32（与"没有 TF32 路径"的精度承诺冲突）。两条都不在本轮。

**结论**：P1 不需要改代码。需要改的是记录口径——把 `winograd_compare` 的
两列明确标成"显式融合版"，并把 README 里 c32/c64 的比值换成 auto 口径。

### 5.269　bf16 落到设备：`uint16` 位模式 + cuBLAS 的 CUDA_R_16BF

在此之前 bf16 只是一层**数值承诺**：IR 有 `DType.BF16`，参考后端把每个边界
舍入到 bf16 格点（载体仍是 f32），设备端则完全没有 16 位存储。DESIGN 记过
一条"不做 bf16"的理由：PTX 的 bf16 转换指令（`cvt.rn.bf16.f32`）要 sm_80，
而项目承诺不依赖 CUDA toolkit、也不想把内核源码绑死在一种架构上。

这条理由只对"用 `cvt`"成立。bf16 的定义就是 f32 的高 16 位：加宽是左移
16 位，收窄是高半字的 RNE 舍入，两件事都能用整数指令写出来——

```cuda
__device__ __forceinline__ float af_ld_b(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}
```

舍入规则与主机侧 `runtime.bf16.round_bf16` 逐条对齐：尾数 LSB 为 1 时进位；
inf / nan 只截断；截断后尾数清零的 NaN 补 quiet 位（否则 NaN 会变成 inf）。
于是 bf16 在设备上就是 **`uint16` 位模式的数组**，从 Maxwell 起都能编，
也不需要 `cuda_bf16.h`。

**改动分四层**：

1. **codegen**：`_STORAGE["bf16"] = ("unsigned short", "af_ld_b", "af_st_b",
   "af_add_b")`，`_BF16_HELPERS` 给出五个设备函数（含 CAS 版原子加，写读
   半字的 32 位 CAS 与 f16 那份同构）。`_dtype_key` / `_ir_dtype` 认
   `uint16` / `bfloat16`。方言三处数据（类型名、load、store、atomic、
   helpers）各加一行；`CAST_DIALECT` 注入 f32 / f16 / bf16 三套 helper，
   因为 cast 是唯一"两侧存储不同"的内核。
2. **执行骨架**：新增 `Executor.device_dtype(dtype)` 钩子。计划里的期望
   dtype 从此不再等于 numpy 载体——bf16 的载体是 f32，但设备容器是
   `uint16`。设备后端覆盖 `quantize`：边界上 f32 → bf16 直接调 `cast`
   （store helper 就是 RNE），**不落一次 D2H**；`to_host` 把 `uint16`
   解包回 f32 载体。这两处让"每个 bf16 边界都要量化"的既有语义在设备侧
   仍然成立，只是量化的落点从主机挪到了内核。
3. **逐元素链**：`build_chain` 的 `round_steps` 从 `dtype is F16` 扩到
   `dtype in (F16, BF16)`。含义是链里每一步先 `af_st_b` 再 `af_ld_b`：
   不融合的图里每个算子都是一次真实的 16 位往返，融合若一路在 float 里算，
   结果会与参考实现不同。链的分组判据 `_chainable` 也放开 bf16，运行期
   再由 `_try_chain` / `_planned_chain` 核对 `BF16_OPS`。
4. **matmul**：cuBLAS 的 `cublasGemmEx` / `cublasGemmStridedBatchedEx`
   多一对 `bgemm` / `bgemm_batched`，`CUDA_R_16BF = 14` + `COMPUTE_32F`。
   `uint16` 数组的位布局与 `CUDA_R_16BF` 完全一致，指针直接传，不需要
   任何重排。

**支持面**（`backends/cuda/backend.py` 的 `BF16_OPS`，32 个）：逐元素
（17）、搬运与 constant / cast（8）、matmul（1）、softmax / layernorm /
rmsnorm（3）、reduce_sum / max / mean（3）。卷积 / 池化 / scan / sparse
**没开**：它们的内核里有权重布局与分块假设，bf16 版本要逐个验证数值，
不能靠模板替换就宣布支持。表外的 bf16 算子由 `invoke` 明确拒绝（不做隐式
降级），消息里点明"bf16 支持面不含这个算子"。

**数值**（A800，与参考后端逐位比对）：

| 用例 | 结果 |
| --- | --- |
| 逐元素链 `relu(x*w + b)`（bf16 全程） | 逐位一致 |
| `cast` f32 → bf16 → f32 往返 | 逐位一致 |
| constant / reduce_sum / softmax / layernorm | 逐位一致 |
| matmul 32×64×16、批量 2×32×64×16 | 逐位一致 |
| matmul 256×1024×256 | 最大相对误差 7.6e-3（一个 bf16 ulp） |

大矩阵上的差异来自 f32 累加的结合顺序（cuBLAS 的分块与 numpy 的 BLAS
不同），量化到 bf16 后偶尔跨一个格点。这与参考后端"f32 累加、边界量化"
的约定不冲突：差异在**约定允许的累加顺序**里，不在语义上。小 K 上两者
恰好同序，因此可以逐位断言。

**性能**（2048³，设备常驻）：

| 精度 | 时间 | 吞吐 |
| --- | --- | --- |
| f32 (Sgemm) | 1.255 ms | 13.7 TFLOPS |
| bf16 (CUDA_R_16BF) | 0.190 ms | 90.5 TFLOPS |

6.6×，确认走的是 HMMA.bf16 而非"转 f32 再算"。

**回归**：远端 `tests/test_cuda_backend.py` 349 passed / 5 skipped（新增
8 条 `TestCudaBf16`）；本地全量 2236 passed / 51 failed，失败项全是
"torch 2.4.1 不在兼容矩阵"，与本轮无关。

**遗留**：conv / pool / scan / sparse / attention 的 bf16 支持面，需要
逐算子做数值验证后再放开。

### 5.270　conv + BN + 激活融进输出变换的写回点

5.262 之后 `channel4d` 的 BN 链已经是一个内核，但它仍然是"conv 落盘、
链读回来"：`conv` 的 `Y` 每当只为这条链服务时，那次写 + 那次读都是白付的。
这一轮把链内联进 conv 内核的**写回点**，`Y` 不再物化。

**改动**

1. **模板**（`lower/kernels.py`）：`winograd_output_transform(epilogue=None,
   extra=())` 与 `winograd_fused(epilogue=None, extra=())` 在写回前插
   `epi(value, channel)`。`channel` 是**当前输出通道变量**（四段版是 `o`，
   融合版是 `oo`），逐通道参数因此可以直接 `Index(name, channel)`，不需要
   按展平下标反算通道。`extra` 是 `((name, "scalar" | "channel"), ...)` 的
   声明，实际数组走独立的 `extra_args`——把"参数类型声明"与"运行期指针"
   分开，是因为前者进源码缓存键、后者每次执行才拿到。默认 `None` / 空时
   生成的源码与改动前逐字相同。
2. **分组**（`device_backend._merge_conv_epilogue`）：单个 `conv2d` 节点与
   紧随其后的逐元素链并成一组。前提全部是静态的：conv 输出不是图输出、
   只被链首使用、链全部 `_chainable`、dtype 与形状一致、链的组外输入只能
   是标量或逐通道广播（否则 conv 内核里没有可索引的通道变量）。
3. **计划**（`_planned_conv_epilogue`）：用 `apply_op` 把链翻译成表达式，
   逐通道参数取 `Index(name, channel)`、标量取 `Index(name, 0)`；先试
   四段版 `_planned_winograd`，再试融合版 `_planned_winograd_fused`，两者
   都不可用就返回 `None`，整组退回逐节点执行（正确性优先）。
4. **执行**（`_GroupPlan.fused` + `invoke_group`）：conv 组优先调 `fused`，
   返回 `None` 才交给父类逐节点。`fused` 与 `chain` 分开是因为链首不是
   逐元素算子，`_try_chain` 的判据套不上。

**范围**：只对 f32 + Winograd。f16 / bf16 的 conv 输出要先舍入到存储类型，
融合后一路在 float 里算会改变数值；im2col / tiled 内核没有集中的写回点，
没有可内联的位置。

**数值**（3 层 C=16 16×16×16，batch 2）

| 对照 | max\|d\| | 值域 |
| --- | --- | --- |
| 纯 conv，auto vs 参考 | 2.4e-3 | ±6500 |
| 纯 conv，im2col vs 参考 | 1.9e-3 | ±6500 |
| conv + BN + relu，auto vs 参考 | 9.2e-4 | [0, 2418] |
| fused vs 关掉合并 | **0（逐位）** | — |

融合路径与"conv 落盘 + 链"逐位相同，说明表达式拼接与通道索引没有引入
额外误差；与参考后端的 1e-3 量级差异在纯 conv 上就已存在，来自 Winograd
变换与 f32 累加顺序（im2col 对参考也有同样的量级），不是融合造成的。

**性能**（A800-40GB，8 层 conv-bn-relu，4×C×56×56，CUDA events，ms）

| 用例 | 关融合 | 开融合 | 组数 |
| --- | --- | --- | --- |
| c16（`conv_graph_compare` 口径） | 0.306 | **0.185** | 48 → 40 |
| c32 | 0.607 | **0.473** | 48 → 40 |

c16 是 1.65×、c32 是 1.28×。差距来自两档的 conv 本身占比：c16 走融合版
Winograd，本身只有约 20 µs/层，链的 3 次启动与一次物化就显得可观；c32
走四段版，conv 自身 50 µs/层以上，省下的仍是固定的那几微秒。

**顺手修的真问题**：`codegen._dtype_key` 只认 `np.dtype` 与字符串，不认
标量**类型**（`cp.float32` 没有 `.name`，`str()` 是 `<class 'numpy.float32'>`）。
融合版计划展开里传的正是 `cp.float32`，此前一直没暴露，是因为它的缓存键
`("winograd_fused", "f32")` 与非计划路径共享、后者已用 `np.dtype` 编译过；
带 epilogue 的内核换了键，第一次真正编译就抛 `ValueError`。改成"先取
`.name`，不是字符串再用 `np.dtype()` 归一化"。

**回归**：远端 `tests/test_cuda_backend.py` 355 passed / 5 skipped，新增
`TestConvEpiloguePlan`（5 条静态判据）与 `TestCudaConv.test_conv_epilogue_
fusion_is_bitwise_identical`（逐位 + 参考对照）。

**遗留**：im2col / tiled 没有写回点；f16 / bf16 需要内核里先舍入再进链；
多张量 epilogue（如 `layernorm(add(a, b))`）不能只按通道索引。

### 5.271　梯度拼桶与并发提交

`DistributedDataParallel` 的归约一直是**逐张量**调用集合通信：每个参数一次
`all_reduce`。每次调用的固定开销（建消息、握手、relay 转发）与张量大小无关，
所以参数一多，时间几乎全花在往返上——200 个 2 KB 的参数就是 200 次往返，
真正要搬的只有 400 KB。

**改动**（`train/parallel.py`）

- `bucket_bytes`（默认 25 MB，与 DDP 同量级；`0` = 逐参数，保留旧行为）：
  按 `grads` 的相邻顺序与 dtype 切桶，桶内 `concatenate` 成一块只归约一次，
  再按下标切回视图。超过上限的单张量独占一桶——它拼不进任何桶，切块只会
  多付启动开销。
- `threads`（默认 2）：桶与桶之间没有依赖，交给线程池并发提交；拼下一个桶
  的主机时间与上一桶的传输重叠。前提是集合实现等待网络时释放 GIL
  （socket / MPI 都是），否则线程只会轮流跑。

**数值**：每个元素仍只参与一次求和，拼桶与逐参数的归约结果**逐位相同**
（`test_bucketed_result_is_bitwise_identical`）。

**数据**（本机两 rank、真 TCP relay，200 个 8×64 f32 参数，一步）

| 形态 | 调用次数 | 每步 |
| --- | --- | --- |
| 逐参数 | 200 | 65.44 ms |
| 32 KB 桶 | 13 | **28.47 ms**（2.30×） |
| 32 KB 桶 + 4 线程 | 13 | 29.12 ms |
| 单桶（400 KB） | 1 | 35.60 ms |

三条结论：调用次数是主要成本；桶不是越大越好（单线程 relay 转发大包反而
更慢，且要等最慢的桶）；线程并发在单线程 relay 上看不出收益，只有在集合
实现真正并行（MPI / NCCL）时才体现。

**顺带的同步点收窄**：新增 `Executor.to_host_async` / `Trainer.step_async`。
设备态每步原本在"前向之后"用 `cp.asnumpy` 把损失拉回主机，那是一次**全设备
同步**——主机要等整条流空。现在排一次 pinned 的 4 字节 D2H、只记一个事件，
反向的 replay 可以立刻跟着排；`step` 仍是 `float(handle)`，行为不变。单卡
实测连续步 0.577 → 0.584 ms，**收益在噪声内**：这个规模的 GPU 一直是瓶颈，
主机提前排队没有可用的空泡。它保留下来是因为语义正确（同步点从"整条流"
收窄到"这一次拷贝"），而不是因为测出了加速。pinned 指针不能塞进
`cp.ndarray`（`alloc_pinned_memory` 给的是 `PinnedMemoryPointer`），拷贝走
`cudaMemcpyAsync(D2H)`。

**遗留**：真正的"通信与反向计算重叠"要在设备态分段反向里"某参数梯度一就绪
就提交 NCCL"（PyTorch DDP 的桶 + 反向钩子）。本机只有一张卡，NCCL 双进程
同卡不可行，给不出真机带宽数字，因此没有动设备态路径；`collective_support`
的 `overlap` 仍为 `False`——只做了"桶之间并发"，不是"与反向重叠"。

### 5.272　OpenAI 兼容的 HTTP 推理接口

`ServingEngine` 已经是"请求队列 + 连续批处理"，但它只有 Python 入口。把
`ByteTokenizer`、`ChatTemplate`、`ServingEngine` 串成一个 HTTP 服务，只需要
协议与并发两件事——**这一层不碰模型**，三者都可替换（`aetherflow/serve`）。

**为什么是兼容子集而不是全兼容**：`logprobs`、`n>1`、`function calling`、
`response_format`、`logit_bias` 都要求模型或采样面提供框架没有的能力。返回
200 再悄悄忽略参数，调用方会以为它生效了，直到线上才发现输出不对。所以这些
字段**显式 400**，错误类型 `invalid_request_error`——"没有被执行"必须让调用
方知道。

**并发模型**

- 一个后台线程驱动 `engine.step()` 与 `take_timed_out()`；每个 HTTP 请求一个
  线程等自己的 `Event`。真正的解码并发度是 `ServingEngine.capacity`，不随
  HTTP 线程数变化（`ThreadingHTTPServer` 只负责连接）。
- `submit_tokens` 在**持锁的同一临界区**里入队并登记作业：`max_tokens=0` 的
  请求直接进完成队列，驱动线程下一次 `step` 就能取到它；登记晚一步结果就丢了
  （有专门的用例钉住）。
- 作业失败（驱动线程抛异常）会唤醒全部等待者并转成 500/流内 error 帧，线程
  不会静默死掉。

**流式的语义**

- `step` 只在请求**完成**时返回，所以 SSE 每轮读一次 `ServingEngine.progress()`
  的只读快照，把"比已发送文本多出来的部分"推给客户端。增量是文本前缀差，
  不是 token 数差——字节 tokenizer 的 token 边界不等于字符边界。
- 一个多字节字符被切在 token 边界上时 `decode` 会抛错：这时**先不发**，下一轮
  多一个 token 就能解出完整字符（`safe_decode` 返回 `None`）。这一刻不发比发
  半个字符更接近客户端看到的文本。
- 请求结束前用完整 token 再解一次（最后几个 token 可能没来得及进快照）。
  若整串 id 最终仍不是合法 UTF-8（模型输出任意字节时会发生），给出**最长的
  可解码前缀**——不是乱码、也不是断开连接；这与流式一直在做的事同源，只是
  结束时没有下一轮可以等了。
- `stop` 只在文本层截断，**不参与调度**：命中后引擎仍会生成到 `max_tokens`，
  只是不再往客户端发。做成"提前取消"需要调度器支持文本级停止条件，目前没有。
- 客户端断连会取消后台请求，不让它继续占着槽位生成没人要的结果。

**判据**（`tests/test_openai_server.py`，18 条）

- `/v1/models`、`/healthz` 的形状；
- 非流式 completions / chat 的文本等于**另一个独立 `ServingEngine`**
  （capacity=1）给出的结果——HTTP 层没有改变调度语义；
- `stream=true` 的增量拼接**等于**非流式完整文本，且只有最后一帧带
  `finish_reason` 与 `usage`，末尾是 `data: [DONE]`；
- 400 面：空 prompt、`n>1`、`logprobs` / `logit_bias` / `best_of` /
  `response_format` / 非 0 penalty、越界 token、空 messages、未知 role；
- 429 / 504 / 500 的映射各用最小假引擎钉住；404 覆盖未知路径与错误方法；
- `max_tokens=0` 返回空文本且 `completion_tokens=0`；
- "半个 UTF-8 字符先不发"与"最长可解码前缀"各一条。

测试用一层 `_AsciiBackend` 把 logits 限制在小写字母上：随机初始化模型吐出的
任意字节会让"文本层"的判据退化成"能不能解码"，限制成 ASCII 后流式与非流式
才有可比内容。

**遗留**：没有认证、限流、多副本与背压（`max_pending` 只约束等待队列，满了
返回 429，由调用方决定重试）；流式请求各自轮询 `progress()`，是 `O(并发)` 的
快照复制，容量大时应改成"调度器推送增量"；`/v1/chat/completions` 只接受
字符串 `content`，多模态内容数组显式 400。

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
| 两条直算路径只覆盖 f32 | 分块直算与隐式 GEMM 的 shared 与累加器都是 float：显式选择且输入为 f16 时以 `BackendCapabilityError` 失败（f16 的 conv2d 走 im2col + hgemm）。其余 59 个算子的 f16 自 5.121 起全覆盖（62 个设备端算子去掉 3 个 int8 语义算子 `quantize_i8` / `dequantize_i8` / `linear_i8`）——attention 是混合精度（q/k/v/mask/out 存半精度，点积、在线 softmax 与输出累加在 float） |
| 累积与更新仍在 f32 | f16 只降低存储与带宽，归约与 GEMM 的累加都在 f32 完成；master 权重与优化器状态必须保持 f32（5.28），f16 参数直接更新会被量化吞掉 |
| softmax / layernorm / reduce_sum 只支持最后一维 | 覆盖 Transformer，不覆盖全部模型。**已由 5.153 关闭**：任意单轴可跑，多轴 `reduce_*` 与 `argmax` 仍限末维 |
| 广播折叠后上限 4 维 | 超高维张量的非常规广播会被明确拒绝 |
| 批量的部分广播（`(2,1,…)` 对 `(2,3,…)`） | cuBLAS 一个标量 stride 表达不了。**已由 5.324 关闭**：广播侧物化成完整批次后仍走 strided batched，代价是 batch 份副本 |
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

### 5.273　前端补全：只扩 aten 子集，不引入 define-by-run

第一代前端的导入是"能跑通 ResNet / SqueezeNet 的最小集"：卷积与池化、linear、
LayerNorm、softmax、SDPA、reshape/切片/拼接。真实模型还会用到截断、标量运算、
归约、常量构造与 dtype 转换，缺一条就整图导不进来。P7 把这批补齐，仍然守住
**只吃 `torch.export` 的 aten 子集、不做 define-by-run** 的边界。

**三条接地规则**

1. **只落到 IR 已有的算子上。** 这轮没有新增 IR 算子，所以也没有新的后端内核
   与新的对齐面：算子表仍是 67 个，CUDA/HIP 的承接范围不变。能用已有算子
   精确表达的，就展开成多个节点。
2. **展开必须是恒等式，不是近似。** `leaky_relu(x) = relu(x) - slope*relu(-x)`、
   `elu(z) = relu(z) + alpha*(exp(-relu(-z)) - 1)`、`triu = tril(x^T, -k)^T`、
   `abs(x) = relu(x) + relu(-x)` 在全部输入上都成立。`softplus` 用
   `max(z,0) + log(1 + exp(-|z|))`：指数项恒 `<= 1`，任意输入都不溢出，而且
   比 torch 的 threshold 截断（`x > 20` 时直接返回 `x`）更精确——两者相差
   `exp(-20)`。正因如此，**非默认 `threshold` 一律拒绝**：换值会真的改变输出。
3. **不能精确展开就拒绝，并说清原因。** `pow` 只接 0 / 0.5 / `-1` / 整数
   （平方求幂展开）；`x ** 1.5` 走 `exp(1.5*log x)` 在非正输入上不是原式，
   直接报错。`clamp` 的边界必须是构图期常量（IR 把边界写在属性里）；`pad`
   只补零（reflect/replicate 与 `value != 0` 拒绝）；`copy_` 拒绝（IR 的值
   不可变，无法表达"写回后旧读者仍看到旧值"）。

**新增的覆盖**

- 一元数学：`tanh` / `exp` / `log` / `neg` / `sqrt` / `erf` / `abs` / `rsqrt` /
  `reciprocal` / `square` / `log1p` / `expm1`；
- 激活：`leaky_relu` / `elu` / `hardsigmoid` / `hardswish` / `relu6` /
  `softplus` / `log_softmax`；
- 截断与标量：`clamp` / `clamp_min` / `clamp_max`，`add/sub/mul/div/rsub`
  的 `Scalar` 重载与 `Tensor` 重载里的 Python 字面量操作数；
- dtype：`to.dtype` / `_to_copy` / `type_as` / 非原地 `copy`；
  `_assert_tensor_metadata` 与 `contiguous` 按恒等处理（IR 没有步长与设备语义）；
- 归约：`mean` / `sum` / `amax` / `max.dim` / `argmax`。`max.dim` 的两个输出
  槽位都精确给出（values 用 `reduce_max`、indices 用 `argmax`），而不是把
  indices 填 `None`——那样下游一取下标就变成"未支持的多输出槽位"；
- 掩码与前缀和：`tril` / `triu` / `cumsum`；
- 形状与拼接：`expand_as` / `stack` / `split_with_sizes` / `narrow` /
  `unbind` / `squeeze`（无 dim）/ `flatten` / `t`；
- 矩阵乘族：`mm` / `mv` / `dot` / `addmm`。`mv` 要先补成 `(K,1)`，因为 IR 的
  `matmul` 只接受秩 >= 2；
- 常量构造：`zeros` / `ones` / `full` / `zeros_like` / `ones_like` /
  `full_like` / `arange` / `scalar_tensor` 落成 `constant` 节点，复用导入期的
  元素上限（65536）；超过上限请改成参数或显式输入，而不是把大张量塞进图属性。

**明确的拒绝清单**（都走 `UnsupportedOperatorError`，不静默跳过）：
`where` / `masked_fill` / 比较算子（IR 没有 bool 张量）、`topk`、`sign` /
`round` / `floor` / `ceil`、`baddbmm` 等批量矩阵乘变体、
`copy_`、非默认 `softplus.threshold`、非整数 `pow` 指数、图内 `clamp` 边界、
非零/非 constant `pad`。

**判据**（`tests/test_frontend.py`，83 条）

- 每一族至少一条端到端对拍：`import_exported` 后在参考后端算出的结果与 torch
  eager 逐点一致；`abs`/`tril`/`triu`/`pad`/常量构造用 `atol=0` 钉住精确性；
- `softplus` 在 `±50`、`±80` 上不溢出，且与 `F.softplus` 的差在 `1e-6` 内；
- `max.dim` 的 values 与 indices 各一条，并确认图里真的出现 `argmax`；
- 反例逐条：`pow(1.5)`、图内 `clamp` 边界、`pad(mode="reflect")`、
  `pad(value=1.0)`、`softplus(threshold=5)`、`floor`、`sign`、超上限常量；
- `resnet18` 与 `squeezenet1_0` 的整体对拍仍在（未受影响）。

**遗留**：`where` / 比较 / `topk` 需要 IR 新增 bool 张量或选择算子，属于
"扩 IR"而不是"扩前端"，留给后续；`log_softmax` 走 `log(softmax(x))`，在
概率下溢到 0 时会得到 `-inf` 而 torch 的稳定实现对同一输入给有限值——这是
浮点表示差异，不是语义改变，但极端 logits 上会看到差别。

### 5.274　工程化收口：门禁、单一事实来源与发布路径

P8 不碰语义层，只把"别人能不能稳定地用上"补齐。判断标准是：这条改动能否
在流水线里被自动验证；不能验证的（比如容器镜像）就先不做。

**静态检查只开门槛规则**

`ruff` 的规则集在 `pyproject.toml` 里固定为 `E9 + F`：语法错误、未定义名、
未使用导入/变量。风格类规则（行宽、`lambda` 赋值、import 排序）不进门禁——
代码库里有 13 处为了构图糖而写的 `sigmoid = lambda x: ...`，把 `E731` 打开
只会制造与正确性无关的改动。门禁的价值在于"新代码不能带进明确的错误"，
而不是"格式化历史代码"。

首次全量检查的结果不是零：47 个 `F` 类问题里有三个值得记录：

- `backends/hip/runtime.py` 的 `_module()` 引用了一个从未定义的 `_MODULE`，
  而且函数没有任何调用者——纯死代码，删除。这类问题 `flake8` 能发现、
  "看起来能跑"发现不了（注解有 `from __future__` 兜底，死函数没人调用）。
- `ir/grad_rules.py` 的 softplus 反向里 `g` 取了但没用；`ir/ops.py` 的
  sparse 反向里 `cols` 同理——都是重构后的残留，删掉即可。
- `tests/test_hip_backend.py` 的"捕获期释放"用例靠 `dead = None` 触发引用
  计数释放。自动修复把它改成裸表达式会让测试**静默失效**；改成
  `del dead, host`，语义不变且不再是未使用变量。这条说明 lint 的自动修复
  不能无条件信任，测试的副作用必须先读懂。

**版本与兼容矩阵各只有一处事实来源**

`pyproject.toml` 原来同时写死 `version = "0.1.0"` 与 `frontend = ["torch>=2.1"]`，
而 `frontend/compat.py` 的矩阵是 2.9–2.13。两处都在描述"能装什么"，却给出
了不同答案：按 `>=2.1` 装进来的 torch 会在导入时抛 `UnsupportedTorchVersion`。
现在版本改成 `dynamic = ["version"]`（读 `aetherflow.__version__`），torch 约束
改成 `>=2.9,<2.14`，并新增 `tests/test_packaging.py` 把两条对应关系钉死：

- `pyproject` 里不能同时出现静态 `version`；
- `frontend` extra 的下界等于矩阵 `min_version`，上界等于 `max_version + 1`。

这类"文档/配置与代码各说各话"的漂移，只有写成测试才会在提交前暴露。

**CI 补三个作业**

- `lint`：`ruff check .`，规则集来自 `pyproject`；
- `frontend`：在 torch 2.9 与 2.13 两端各跑一次 `tests/test_frontend`。
  此前 CPU 作业不装 torch，前端测试在 CI 上**永远跳过**——本地能跑不等于
  云端在跑。矩阵两端各钉一次，是为了让"声明支持"至少有两处实测；
- `publish`：`v*` tag 触发，先校验 tag 与 `__version__` 一致，再用 PyPI
  trusted publishing 发布。普通 push 与 PR 不持有 `id-token: write`。

CPU 作业改为 `coverage run` 并上传 `.coverage`（只观测、不设阈值）。覆盖率
作为门禁需要先有一段历史来定基线，现在设阈值只会得到拍脑袋的数字。

**没做的**

容器镜像、GitHub Release 自动创建、pre-commit 钩子。前两者在本地无法验证
（没有 docker daemon、没有远端 release 权限），后者与 CI 的 `lint` 作业重复。
工程化的取舍标准是"能否被验证"，而不是"清单上还有几项"。

### 5.275　动态形状：把符号批维打通到训练循环

**问题**：5.217 的契约表把 `graph` / `execute` 记为 `full`，`autodiff` 记为
`per-op`，`trainer` 记为 `rejected`。这张表当时是如实的，但三处具体阻塞把它
钉死在了 `rejected`：

1. `reshape` 的 infer 拒绝 `SymExpr` 目标，`(B,4) -> (4*B,)` 在构图期就报错；
2. 执行器的 `resolve_attrs` 只把 `str` 当符号去绑定，属性里的 `SymExpr`
   原样交给后端，目标形状仍是一个对象而不是整数；
3. 归约 / 广播的 VJP 一律走 `_static`：`reduce_mean` 要常量 `1/count`，
   广播反向 `sum_to` 要用 `_prod` 在构图期算前导乘积。批维是符号时，
   参数梯度必然要沿批维求和，训练循环因此整条断掉。

**做法**：

- `reshape` 只校验"目标里的符号都能在输入类型里找到"，静态可证时再做元素
  数守恒检查；不可证的情况交给执行期——不静默算错，也不提前拒绝本来能跑的图。
- `resolve_attrs` 递归解析 `SymExpr`（`dim_resolve` 按当前绑定求值），
  属性里的 tuple/list 一并处理。执行器因此能把同一张符号图绑到不同 batch。
- `reduce_sum` / `reduce_mean` 的 VJP 改为按原形状（含符号）工作：`keepdims=False`
  时 reshape 到保维形状、再 `broadcast_to` 回输入形状；两者都已接受符号目标。
  均值在归约轴全静态时仍用常量 `1/count`（数值与算子顺序不变），含符号时
  用"标量 1 广播成保维形状 -> 沿归约轴 `reduce_sum`"得到运行期计数张量，
  再与梯度相除。不用 `exp(x-x)` 之类的技巧：`x` 含 `inf` 时它会把计数污染成 NaN。
- `sum_to`（广播反向）先做结构比较：形状相同直接返回，不需要任何具体数字。
  需要求和时，内部广播维用 `transpose + reduce_sum(keepdims=True)`——只依赖
  秩，不依赖各维数值；前导维全静态时保留原来的"合并成二维再一次归约"，
  数值与算子顺序不变；含符号时改成"前导维转置到最后 -> 单轴归约 -> reshape
  去掉那个 1"，reshape 的目标只有本来就已定义的保留维，不引入新符号。
  `sum_axes`（去掉维、必须把若干维合并成二维）仍要求静态，与 conv / embedding
  等规则一起留在 `per-op` 的拒绝侧。
- `Trainer`：主机态不再拒绝符号输入。训练循环本身只做形状无关的参数更新，
  损失若对符号维求平均，反向的计数在每一步按实际形状算。设备态的捕获计划把
  每步的维度与地址写死在图里，仍在构造期用 `symbolic_inputs` 拒绝并给出
  `device_state=False` 的建议；`data_input_shapes()` 遇到符号维也显式报错
  （数据并行按每卡固定 batch 切分，没有"该切多少"的答案）。
- 契约表同步：`autodiff` 的理由去掉"reshape 的目标形状需要具体数字"，改为
  如实描述"归约与广播反向按运行期形状算 + `sum_axes` 一类需要合并维度的规则
  仍要求静态"；`trainer` 从 `rejected` 改为 `per-op`。`recompute` 与
  `torch-import` 不动。

**用例**（`tests/test_shape_contract.py` 增至 19 条）：

1. `reshape` 的目标可以是 `sym_mul(4, "B")`，batch 1/3/7 各跑一次，输出 `(4B,)`
   且元素顺序正确；未定义的符号仍在构图期报错。
2. 符号批维上的 `reduce_mean` 反向：`(B,4)` 沿最后一维求均值，梯度每元素
   0.25（运行期计数路径），batch 1/3/7 一致；`reduce_sum` 反向梯度全 1。
3. 主机态 `Trainer` 用同一张符号图跑 batch 1/3/5 各一步，损失有限且参数真的
   被更新——`per-op` 不是"构造能过、跑起来再说"。
4. 设备态 `Trainer` 在构造期拒绝符号输入，错误含「设备态训练要求静态形状」、
   `%x` 与 `device_state=False`；`data_input_shapes()` 报「含符号维」。
5. `test_autodiff` 里原来钉"符号广播反向必须报错"的用例改成数值对拍：
   `a[B,1] + b[B,4]` 对 `a` 的梯度是每行 4，batch 1/3 都成立。

**验证**：本地 `py -3.14` + torch 2.13.0，`Ran 2773 tests / OK (skipped=427)`；
`ruff check` 改动文件全过。

**边界**：这只是"动态形状全链路"的第一刀。分段重算（`recompute`）、设备态
捕获、`torch.export` 的 SymInt 落成符号维、以及 `sum_axes` 这类需要合并维度的
反向，都仍按契约表拒绝；下一步才是 trace-once 重放与设备侧控制流。原则不变：
静态可证明的检查保留，不可证明的交给执行期报错，绝不静默算错。

### 5.276　trace-once：声明动态轴，一次构图多次重放

**问题**：5.68 的 `DynamicRunner` 缓存键是**完整形状**。5.275 把符号批维打通
之后，同一张图本来就能跑不同 batch，runner 却仍按 `(2,4)`、`(3,4)`、`(7,4)`
各建一张——动态形状的收益在缓存键上被吃掉了。用户要么绕开 runner 自己写符号
图，要么在变长 batch 下反复付构图成本。

**做法**：新增可选参数 `dynamic_axes`，声明「输入名 → {轴: 符号名}」：

```python
runner = DynamicRunner(build, dynamic_axes={"x": {0: "B"}})
y = runner.run({"x": np.ones((3, 4), np.float32)})["y"]   # 用 f32[B, 4] 构图
y = runner.run({"x": np.ones((7, 4), np.float32)})["y"]   # 复用同一张图，B=7
```

- 缓存键只保留**未声明**的维与 dtype；声明过的轴换成符号后，取任意值都命中
  同一张图。`shapes` 里看到的就是 `("B", 4)`——"这张图覆盖了哪些形状"一目了然。
- `build` 收到的是**形状模板**：被声明的轴是符号名，其余轴仍是具体整数。执行器
  在每次 `run` 时按实际数组绑定符号，因此同一条计划能重放到任意 batch。
- 共享符号表示"这两维必须相等"：`{"x": {0: "B"}, "t": {0: "B"}}` 下 x 与 t 的
  批维不一致时，执行器报「符号维 B 绑定冲突」，而不是悄悄挑一个。这也顺带解决
  了"两个输入的批维各自起名、形状推导比较不出来"的问题——符号名就是相等关系的
  显式表达。
- 负轴按本次实际秩换算（`-2` 在 `(2,3)` 上是 `0`），换算后重复声明直接拒绝。
- **构图后立刻自检**：声明为符号的轴如果被 `build` 写成整数（典型是硬编码
  `f32[2,3]`），当场报错并给出图里的实际类型。否则第一次恰巧能跑、换形状才以
  "静态维不匹配"失败，错误离原因太远。同理，声明了某个输入但图里没有它，也在
  构图后拒绝。
- 声明本身在构造期校验：值必须是非空映射、轴必须是整数（`bool` 不算）、符号名
  复用 `TensorType` 的维校验（合法标识符、不与 dtype 名撞车）。空声明、未知
  输入名、轴越界各有各的错误，不共用一条"参数非法"。

**为什么不做自动推断**：不让 runner 猜"哪些轴该动态"（比如"凡是第 0 维都动态"）。
缓存键的正确性依赖声明——把用户没打算复用的轴也吞进同一张图，只会在更晚的时候
以形状错误暴露。也不做"符号构图失败就退回按形状构图"的静默降级：那会让同一次
运行的行为取决于第一次遇到的形状，正是 5.275 边界里说的"绝不静默算错"要避免的。

**用例**（`tests/test_dynamic.py` 10 → 24，新增 14 条）：

1. 同一 `dynamic_axes={"x": {0: "B"}}` 下 batch 1/3/7 只建一次图，输出逐位正确；
   `shapes` 键是 `("B", 4)`。
2. 只声明批维：特征维 3→4 仍重建（`builds == 2`），动态轴不会顺手吞掉静态维。
3. 负轴 `-2` 与正轴 `0` 等价；`0` 与 `-2` 同时声明被拒（换算后重复）。
4. 共享符号：x、y 批维一致时一次构图跑通；不一致时报「符号维 B 绑定冲突」。
5. 与参考后端逐位对拍（batch 1/4）；`matmul` 线性层在 batch 1/3/7 上复用同一张
   图，结果与 `x @ w` 一致。
6. `build` 把模板轴写成整数 → 构图期 `ValueError`，消息含「必须使用传入的形状
   模板」与图里的实际类型；把输入秩写小（声明的轴超出图里的秩）、声明了图里没有
   的输入、声明了却未喂入、轴越界、换算后重复，全部显式报错。
7. 声明本身：非映射、空映射、`bool` 轴、非字符串符号、与 dtype 同名的符号、非
   标识符符号，构造期即拒绝；`dynamic_axes=None` 保持 5.68 的按形状缓存不变。

**验证**：本地 `py -3.14` + torch 2.13.0，`Ran 2787 tests / OK (skipped=427)`；
`ruff check aetherflow/dynamic.py tests/test_dynamic.py` 通过。远端 Ubuntu 22.04 /
Python 3.10.12 拉取同一 commit：`tests.test_dynamic` 24 条通过；全量 2787 条里
非前端用例全通过，`test_frontend` 的 121 条因该机 torch 2.5.1 不在兼容矩阵内报
`UnsupportedTorchVersion`——这是环境问题，不是本轮回归。

**边界**：trace-once 只覆盖"未声明维不变、声明维取任意值"的复用；声明维之外的
形状变化仍按模板分别建图，这正是 `capacity` 的作用。`build` 必须在声明轴上是
形状多态的：依赖具体数字的算子（例如把总元素数算成 `int` 的 `reshape`）在该轴
上会在构图期报错——这是显式边界，不是缺陷；要覆盖这类图就不要把那一维声明为
动态。缓存仍不感知布局等形状之外的输入属性，也不是线程安全的。设备侧分支见 5.277。

### 5.277　设备端分支：三张图各捕获一次，未选分支不执行

**问题**：5.245 只补了循环。`Conditional.run`（5.201）每次判定仍要把谓词拉回
主机、再整图 `execute` 选中分支，分支体内的内核每次重新派发——"设备端控制流"
只完成了半边。循环的对偶不是把 `Conditional` 塞进 `WhileLoop`，而是同样把
三张图捕获下来、只 replay 被选中的那一支。

**做法**（`runtime/device_control.py`，`DeviceConditional`；模块由
`device_loop.py` 提升而来，循环与分支放在一起）

- 谓词、`then`、`otherwise` 各 `capture` 一次，**同一条流**；捕获数恒为 3。
- 每次 `run`：谓词计划 `run_device`（唯一一次同步与谓词 D2H），选中分支
  `replay` 异步；未选中分支**连作业都不提交**——与宿主 `Conditional.run` 的
  惰性逐条对齐。
- `run_device` 不做任何 D2H，分支输出留在显存；`run` 收尾对每个输出做一次 D2H。
- 自由输入建计划时上传一次，`run(feeds)` 给了新值才重传；形状/dtype 变化显式
  `ValueError`（一张捕获图只对应一组固定形状）。
- 契约与 5.245 同源：区域必须是单张 `Graph`（`GraphRegion`），嵌套控制区域抛
  `DeviceLoopError` 并指回 `Conditional.run`；后端缺
  `capture/new_stream/to_array/to_host` 在构造期 `TypeError`；计划缺
  `replay/device_outputs` 在构造期 `DeviceLoopError`；缺输入、多给名字抛
  `KeyError`；谓词输出缺失或运行期非标量抛 `ControlFlowError`（IR 已静态校验
  0 维标量，这里只兜运行期）。
- 异常类型沿用 `DeviceLoopError`：它描述的是"这个控制结构没有设备形态"，循环
  与分支共用同一种失败语义，不为分支新造异常类型。

**判据**（新增 16 条，其中 1 条真机）

- 数值（4）：与宿主 `Conditional.run` 在阈值两侧逐位相等（含 0.999 / 1.0 边界）；
  `run_device` 只读谓词、不 D2H；`run` 只多一次输出 D2H；谓词翻转时选中支跟着
  翻转。
- 捕获复用（4）：三张图各捕获一次、同一条流，重复 `run` 不再捕获；未选分支的
  replay 次数恒为 0（按计划记账）；每次 `run` 一次同步（`syncs == 1`、后端的
  计划同步计数 +1）；新自由输入只上传一次。
- 契约（7）：嵌套区域、缺 `capture`、缺输入、多给名字、换形状、换 dtype、
  非 `Conditional`，全部在构造期或入口显式拒绝。
- 真机（1）：WSL HIP 上 3 次捕获、两支各跑一次与参考后端逐位一致（本机无 HIP，
  跳过）。

**负向对照**（3 组，全部先变红再还原）

1. 未选中的分支也 replay → "未选分支不 replay"变红。
2. 分支改用同步 `run_device` → 同步计数变红（`syncs` 2 ≠ 1）。
3. 嵌套区域改成 `ValueError` → 异常类型判据变红。

**验证**：本地 `py -3.14` + torch 2.13.0，`Ran 2803 tests / OK (skipped=428)`；
`ruff check` 通过。远端 Ubuntu 22.04 / Python 3.10.12 拉取同一 commit：
`tests.test_device_control` 35 条通过（2 条 HIP 用例跳过）；全量 2803 条中非前端
用例全通过，`test_frontend` 的 121 条仍因该机 torch 2.5.1 不在兼容矩阵内报错
（同 5.276，环境问题）。本轮**没有可用的 HIP 设备**（本地与远端都没有 ROCm），
真机用例保留但未执行；CUDA 未验证。不写"已验证"，也不拿 numpy 替身冒充真机。

**边界**

- **谓词仍需一次主机往返**：没有 CUDA Graph 条件节点 / device-side 分支，本层
  不假装零同步——与 5.245 同一原因。
- **只支持单张 `Graph` 的区域**：分支里再嵌套 `Conditional` / `WhileLoop` 只会
  得到"捕获了半个控制结构"，明确拒绝并指回宿主路径。
- **只减少派发，不保证更快**：5.245 的 HIP 实测已说明捕获 replay 的同步开销
  可能更贵；本层交付的是执行形态（捕获复用、未选分支不执行、可判的同步与往返
  次数），不是加速开关。
- **换形状要新建计划**；未接进 `Trainer` / `LMRunner` 主路径。
- **真机覆盖缺口**：HIP 用例本轮未执行（无设备），CUDA 从未验证。

### 5.278　设备端分支接不进解码主路径：判定在采样之后（负结果）

**问题**：5.277 之后剩下的唯一问句是"这套设备形态能不能接进推理主路径"。
`generate` / `LMRunner` 的解码循环是现成的控制流：每步算一步 logits、采样、
决定是否继续。如果能用 `DeviceConditional` 表达，推理侧就顺带拿到了设备端分支。

**结论（负结果）**：接不进去，而且不是"捕获契约差一点"——解码的分支在图外
没有对应物。

三条事实（读代码与一次探针即可复现，不需要设备）：

1. **判定输入是主机采样的产物。** `models/lm.py` 的 `_sample._pick` 先用
   `_host_row` 把 logits 行拉回主机，再由 `sample_next_token`（numpy：温度 /
   top-k / top-p / 重复惩罚 / 约束屏蔽）选出 `int` token；停止判定是
   `token == stop` 与 `len(generated[row]) >= limit`。`Conditional` 的谓词必须
   是**区域的输出**、且恰好一个 0 维标量；decode 图输出的是 `logits` 加全部
   状态张量，采样器不在图里，谓词也就不存在。
2. **分支的动作是改主机循环状态，不是"选一张子图产出"。** `active` /
   `generated` / `history` / 下一轮的 `ids` 都是 Python 侧对象；每步都要执行
   同一张 decode 图，"未选分支不执行"这种可省的作业在解码里没有。
3. **设备端循环的时序不允许主机在中间介入。** `DeviceWhileLoop` 的契约是
   "条件 replay → 体 replay"，两者之间没有回调位；解码的采样恰好必须发生在
   体（产出 logits）与下一次体（写入新 ids）之间。就算谓词能在设备上算出来，
   这一次主机介入也省不掉。

**每步的往返账（探针，参考后端）**：`generate(prompt=[2,3,4], chunk_size=2,
max_new_tokens=4)` 一共 5 次 execute（2 块预填充 + 3 步 decode）；每步 feed 里的
`ids` / `reset` / 状态张量全是主机 numpy。`DeviceBackend.execute` 收尾统一
`to_host`，于是每步是"状态 D2H + 下一步 H2D"，而下一轮 ids 必须从主机写回——
一次 H2D 不可省。真实规模（V=32000、d_model=1024、24 层、d_state=16、
d_conv=4）下状态是 48 张量 / 3.56 MiB：**每个 token 状态往返 7.12 MiB，而采样
只需要 128 KB 的 logits 行**。这笔往返与"分支"无关，它是状态常驻问题。

**CUDA Graph 条件节点可行性（评估，未实现）**

- 前提是整个循环进一张图、谓词是设备侧句柄：采样（argmax、EOS 比较、长度
  计数器、把 token 写回 ids）都得变成图内算子。IR 已有 `argmax` / `reduce_max`
  / `cumsum` / `topk_mask`，但**没有随机数算子**，所以只有贪心能设备化；温度 /
  top-k / top-p 需要 RNG，与主机 numpy 序列无法逐位对拍。
- 批量还要按行掩码 + any-reduce；提前退出与长度上限要从主机计数器变成图内状态。
- 两个后端都没有条件节点封装（全仓没有 `cudaGraphConditionalHandle` 一类的
  符号），HIP 侧同样没有；这不是给 `capture` 加个参数能补的。
- 收益上限也要看清：条件节点省的是"未选分支的派发"，解码里不存在；真正能省
  同步的是"采样搬到设备"，代价是与主机采样不再逐位一致。

**因此本轮不接，只记负结果**：没有代码改动，也没有新增测试——判据就是上面三条
契约与探针账。后续若要做设备态解码，先做的是**状态常驻**（decode 步用
`execute_device` 而不是 `execute`，状态不回主机），它省掉的正是 7.12 MiB/token，
且不需要条件节点；本轮没做，是为了不把"分支"悄悄换成另一个特性。

### 5.279　设备态解码：状态常驻，每步只取回 logits

**问题**：5.278 把问题定位清楚了——分支接不进去，但 decode 每步的
"状态 D2H + 下一步 H2D"（真实规模 7.12 MiB/token）与分支无关，是状态没有常驻。
采样只需要最后一行的 logits（128 KB），状态却每步都往返。

**做法**（`models/lm.py`；不动 `generate` / 束搜索）

1. **一个开关，不静默降级**：`LMRunner(device_state=True)`。它要求
   `resident=True`——否则捕获图必须把全部权重当可变输入，每步重传的字节数比
   省下的更多；还要求后端提供 `capture` / `to_host` / `to_array`。缺任一项在
   构造期 `ValueError`，不偷偷回落到宿主路径。
2. **decode 图只捕获一次**：`steps=1`、`head="last"` 的形状按 batch 固定，
   权重驻留后写死进图；`ids` / `reset` / 全部状态是每步可变的输入。计划与图
   分开持有、不进 `_graphs`——计划记的是捕获时的显存地址，不能被 LRU 淘汰掉
   它依赖的图。
3. **每步只回传 logits**：`_device_step` 把上一步的设备输出直接作为输入
   （HIP 的 `HipGraphPlan.replay` 对同一块缓冲走 D2D），`run_device` 后只
   `to_host(logits)`。状态对象在 Python 侧换了个键名，指针不动。
4. **整块取回**：`_sample` 的 logits 从"逐行 `_host_row`"改成整块一次
   `to_host`。批量 B 行时原来是 B 次 D2H，现在 1 次；这条对宿主路径同样生效。
5. **prefill 不假装设备态**：仍是 `execute`（状态与 logits 回主机一次），第一步
   decode 把状态上传一次（3.56 MiB）。省下的是第 2 步起的每 token 往返。

**顺带修复（HIP 真机暴露，与解码无关但挡路）**

- **容器缺 `cp.uint16`**：`hip/runtime.py` 的 dtype 常量表没有 `uint16`，而
  `device_backend.py` 在 `_kind` / bf16 判定处按 `cp.uint16` 比较。结果是**任何
  图**第一次 `invoke` 就 `AttributeError`（不是"bf16 不支持"的显式拒绝）。此前
  HIP 用例一直跳过，所以从未暴露。补 `uint16 = np.uint16`、加进 `_SUPPORTED`
  与 `_C_TYPES`（`unsigned short`）；HIP 的 bf16 支持面仍是空集，由 `invoke`
  的能力检查拒绝，而不是崩在属性上。
- **分支 replay 的异步时序**：`DeviceConditional.run_device` 的分支 replay 提交
  到捕获流后立即返回，`run` 的收尾 D2H 走默认流。CUDA 的 legacy 默认流会隐式
  同步，HIP 不会——真机症状是 `x=0` 得 10（走 then）、改成 `x=2` 仍得 -10（else
  读到上一次的 0；直接 replay 该计划得 -8）。分支 replay 之后同步捕获流；这不
  是主机往返，仍只读谓词一个标量。

**判据**（新增 9 条，`tests/test_lm_device_state.py`）

- 数值（3）：与宿主会话（`resident=True`）、模块级 `generate` 逐 token 一致
  （含温度 / top-k / seed）；`generate_batch` 两行一致；`prime` 前缀后一致。
- 往返账（2）：同一会话两次请求只捕获 1 次；decode 每步恰好一次 logits D2H、
  状态 D2H 为 0；状态只在首步上传一次（`h2d == 2*steps + states`），此后每步
  每张状态一次 D2D（`d2d == states*(steps-1)`）。
- 形态（2）：束搜索不建计划、仍走宿主；`close` 丢弃计划。
- 契约（2）：缺 `resident`、缺 `capture` / `to_host` 在构造期拒绝。
- 真机（1）：WSL HIP 上与宿主路径逐 token 一致；prefill 之后 decode 每步只取回
  `(1,1,V)`，状态形状的 `to_host` 次数正好等于 prefill 的块数。

**负向对照**（真机，先红后绿）：去掉 `DeviceConditional.run_device` 的流同步，
`test_conditional_matches_reference_on_hip` 变红（`-10.0 != -8.0`）；恢复后绿，
并据此断定是"默认流不等捕获流"而不是数值错误。

**验证**

- 本地（Windows / Python 3.14 + torch 2.13）：`Ran 2812 tests / OK (skipped=429)`
  ——比上一轮多 9 条新用例、多 1 条跳过（HIP 真机用例）；`ruff check` 通过。
  用 Python 3.10 跑全量时 `test_frontend` 的 123 条仍因 torch 2.4.1 不在兼容矩阵内
  报错（环境问题，同 5.276）。
- WSL + ROCm 7.2.4 / RX 7650 GRE（`gfx1102`，DXG 直通）：
  `tests.test_hip_backend` 72 条全过（修复前首次 `execute` 即 `AttributeError`）；
  `tests.test_device_control` + `tests.test_lm_device_state` 共 44 条全过
  （含 2 条 HIP 控制流与 1 条设备态解码真机用例）。
- **环境前提**：WSL 必须 `export HSA_ENABLE_DXG_DETECTION=1`，否则
  `hsa_init` 失败、`hipGetDeviceCount` 返回 0，`available()` 会正确地判为不可用。

**边界**

- **静态形状**：一张捕获图一组固定形状，换 batch 要重建计划（与 5.217 的设备态
  捕获限制同源）。
- **只省状态往返**：采样、停止判定、下一轮 `ids` 写回仍在主机；每步仍有同步与
  128 KB logits D2H，不承诺比宿主路径更快。
- **prefill 与首步**：prefill 仍回主机，状态第一步上传一次；短序列省不下多少。
- **束搜索 / 投机解码**：按父假设 gather 状态或回滚快照，与单条 decode 不是同一
  张图，仍走宿主路径。
- **真机覆盖**：HIP 上验证的是小形状的逐 token 一致与往返次数，没有大模型的吞吐
  数字；CUDA 未验证。

### 5.280　设备态解码进服务化：三处捕获期缺陷与双厂商实测

**问题**

5.279 只覆盖单条会话。连续批处理（`ServingEngine`）每步仍把整块状态 D2H 再
H2D——批越大往返越贵。把 `_device_step` 的后端能力接进 `ServingEngine` 之后，
两家后端各暴露了捕获期缺陷，都不在解码逻辑里：

1. **HIP：D2D 拷贝 `code=1`**（`n=131072`，正好一层 SSM 状态）。`Arena._coalesce`
   把两次独立 `hipMalloc` 但虚拟地址相邻的空闲块合并，best-fit 随后切出跨越分配
   边界的 slice；`hipMemcpyAsync` 按分配校验范围，直接拒绝。
2. **CUDA：图输出与中间值同址**（`logits` 与 `l0_h` 都是 `0x...9e00`）。捕获轮
   `release=True` 时输出是零拷贝视图（reshape / slice），其源按"最后一个读者"
   回收后，地址被后续算子复用；图里两个输出写进同一块。
3. **CUDA：捕获图与逐节点 `execute` 首步分叉**（argmax 4 vs 6）。`_shares_storage`
   只认 `base is source`；当**源本身是视图**（SSM 状态被连续 slice / reshape）时，
   `out.base` 指向最底层数组，判定失败 → 源被当成"没人再读"提前还给 arena →
   地址分给后面的算子，下游读到被覆盖的数据。

**做法**

1. **服务化复用同一张 decode 图**（`ServingEngine(device_state=True)`）：
   `batch=capacity` 的 `steps=1` 图捕获一次，状态整块留在显存（`_state_dev`）；
   准入是低频事件，`_write_state_rows` 整块往返一次，逐 token 只 `to_host(logits)`。
   与 5.279 同一条限制：要求 `resident=True` 与静态形状，缺能力在构造期报错。
2. **arena 只合并同一 chunk 内的相邻块**（两家一致）：`_coalesce` 加
   `_chunk_of(addr)` 判定。HIP 侧有 `recycle`，CUDA 侧没有，但"跨 chunk 不合并"
   是分配器本身的约束，与回收策略无关。
3. **捕获期为图输出物化独立缓冲**：`_capture_outputs` 记下输出 value id，
   `_shared_view` 命中时 `empty` + 拷贝一次。只发生在捕获期，replay 地址仍然固定。
4. **视图记账统一落在底层持有者**：`_storage_root` 追 `base` 链到底，
   `_shares_storage` 比较两侧的根身份，`_alias_view` 按根地址记账。HIP 本就用
   `_buffer`（天然是根），不受影响。

**验证**

- 本地（Windows / Python 3.14）：`Ran 2820 tests / OK (skipped=433)`；
  `ruff check` 通过。新增用例覆盖"不跨 chunk 合并"（HIP 用伪造相邻 chunk 的
  `_FakeArena`，CUDA 只走 `free` 账本）与 Serving 设备态的逐 token 一致。
- WSL + ROCm 7.2.4 / `gfx1102`：`tests.test_hip_backend` 50 条全过；
  中规模 ms/token（宿主 → 设备态）8 层 117.77 → 16.25（7.25×）、12 层
  179.50 → 24.99（7.18×）、24 层 353.64 → 44.86（7.88×），逐 token 输出一致。
- A800-SXM4-40GB（驱动 550.127，CUDA 12.4，`cupy-cuda12x` 14.2.0）：
  真实规模（24 层 / `d_model=1024` / `vocab=32000`）宿主 86.408 →
  设备态 14.162 ms/token（**6.10×**，best of 3，32 步，greedy 逐 token 一致）。
  `tests.test_cuda_backend` + `tests.test_device_control` +
  `tests.test_lm_device_state` 共 406 条 `OK (skipped=5)`。

**边界**

- **Serving 的 batch 固定**：容量在构造期定死，换容量重建计划（与 5.217 同源）。
- **只省状态往返**：采样、停止判定、`ids` 写回与每步同步仍在主机；logits 仍要
  回传。加速比来自"省掉的是整块状态、留下的是 128 KB logits"，不是"设备端全流程"。
- **首步与准入**：prefill 与新增请求仍要一次整块状态上传；短序列收益有限。
- **逐节点 `execute` 仍是诊断基准**：它不经过 arena 回收，永远比捕获图"安全"；
  两者数值必须一致，这次分叉就是靠它定位的。

### 5.281　连续批处理的设备态：容量扫描与实际吞吐

**问题**

5.279 / 5.280 只验证了单条会话与正确性：设备态在 `capacity=1` 时省掉每步
7.12 MiB 的状态往返，批处理路径的收益没有数字。批处理里两边都被放大——
宿主路径每步仍要整块 D2H + H2D（`capacity` 行一起搬，字节数线性涨），设备路径
每步只回传 `capacity × 128 KB` 的 logits。哪一边先饱和，只能测。

**口径**

- 同机（A800-SXM4-40GB，驱动 550.127，`cupy-cuda12x` 14.2.0）空闲运行：
  测量前后 `nvidia-smi` 均为 `0 % / 10 MiB`，SM 1410 MHz。
- 24 层 Mamba，`d_model=1024`、`d_state=16`、`vocab=32000`、`steps=32`、
  32-token prompt、每条请求生成 32 token；`resident=True`，只切
  `device_state` 一个开关，两个引擎各自做一轮预热后再测两轮取最快。
- 计时单位是 `step()`（一次调度步），吞吐 = `capacity × 32 /` 总墙钟
  （含准入那一步的预填充）；每行都是 33 步。
- 逐请求输出与宿主路径**完全一致**（4 档容量、8 条请求逐 token 比对）。

| capacity | 每步状态 | 宿主 ms/步 | 设备态 ms/步 | 解码加速 | 宿主 tok/s | 设备态 tok/s | 吞吐加速 |
|---|---|---|---|---|---|---|---|
| 1 | 3.56 MiB | 77.258 | 5.535 | 13.96× | 12.46 | 114.31 | 9.18× |
| 2 | 7.12 MiB | 95.742 | 8.238 | 11.62× | 20.19 | 169.03 | 8.37× |
| 4 | 14.25 MiB | 101.958 | 9.501 | 10.73× | 37.95 | 300.33 | 7.91× |
| 8 | 28.50 MiB | 117.345 | 12.061 | 9.73× | 66.08 | 491.58 | 7.44× |

**读法**

1. **宿主路径的每步成本几乎与容量无关**：1→8 行只涨 52%（77 → 117 ms）。
   它花在"每步一次 `capacity` 行状态 D2H/H2D"上，搬运字节随行数线性涨，但
   内核与同步的固定部分不变；真正的浪费是"一行也在搬 3.56 MiB"。
2. **设备路径的边际行成本是 0.93 ms**：1→8 行 5.54 → 12.06 ms，八行共享
   一次 replay。摊到每个 token 上是 5.5 / 4.1 / 2.4 / 1.5 ms。
3. 所以**解码加速比随容量收窄**（13.96× → 9.73×），但**吞吐加速比稳定在
   7.4～9.2×**——宿主那边被摊薄的是固定拷贝，设备那边被摊薄的是 replay。
   `capacity=8` 时 66.08 → 491.58 tok/s。
4. 首轮（含建图 / 捕获 / 权重驻留）0.91～1.05 s，与容量无关；表里是稳态。

**同日复测 5.280 的单条数字**

同一进程内用 `LMRunner`（5.280 的口径）复测：宿主 81.185 → 设备态
**8.207 ms/token（9.89×）**。宿主侧与 5.280 的 86.408 差 6%，设备侧与
14.162 差 42%——差别不在代码：设备路径吃 SM 时钟与同卡其它进程。同一段
代码在外部负载 91%～100%（显存被占 6.3～21.8 GiB）时测到设备态
22.8～25.7 ms/token，而宿主侧 87.7～90.2（≤6%）。**结论：设备态数字必须
在空闲卡上取；宿主侧对争用不敏感，可以直接跨时段比较。**

**HIP 同套扫描**（WSL2 / ROCm 7.2 / RX 7650 GRE，24 层，每条 16 token）

| capacity | 宿主 ms/步 | 设备态 ms/步 | 解码加速 | 宿主 tok/s | 设备态 tok/s |
|---|---|---|---|---|---|
| 1 | 354.629 | 38.555 | 9.20× | 2.54 | 13.11 |
| 2 | 440.418 | 43.525 | 10.12× | 4.15 | 23.27 |
| 4 | 448.567 | 51.728 | 8.67× | 8.08 | 40.81 |
| 8 | 484.615 | 68.998 | 7.02× | 14.00 | 51.12 |

`capacity=1` 的宿主 354.6 ms/步与 5.280 记录的 353.64 ms/token 对得上，
逐请求输出同样一致。两家后端趋势一致：宿主被固定拷贝拖住、设备被 replay
摊薄。

**边界**

- 只改了测量，不改代码：`ServingEngine(device_state=True)` 的语义、准入
  写行与捕获期约束见 5.280。
- 空槽仍随 pad 推进并白跑内核；这里四档容量都填满 8 条请求，不测"半空"。
- 设备态省的是**状态**往返；采样、停止判定、`ids` 写回、每步同步仍在主机，
  logits 仍回传（`capacity × 128 KB/步`）。
- 首轮捕获是固定成本；短请求（生成个位数 token）收益有限。

### 5.282　束搜索的设备态：把状态重排留在显存

**问题**

`generate_beam` 每步按 `parents` 重排状态：整块 D2H（`k × 3.56 MiB`）→ numpy
下标 → H2D。5.279～5.281 把单条与批处理的 decode 状态留在显存，束搜索却仍在
这条老路上；而且两边都是**逐数组**做的（24 层 × 2 个状态 = 48 次
`to_host` / `to_array`），束宽越大，往返的字节与调用次数都线性涨。

**做法**

一张重排图：每个状态张量一次 `embedding(state, parents)`——表就是上一步的
`(k, ...)` 状态，下标是父行号，内核按行拷整块。`parents` 只有 `k` 个 int32 要
上传，重排输出直接作为下一步 decode 的输入（D2D），中间不落主机。

- 这张图没有权重输入，捕获与"权重驻留"无关；每个束宽一张，跨请求复用。
- 顺手修一个只在真机上暴露的 bug：`_decode_plan_for` 的样本形状原先取
  `self.spec.state_shapes`，束宽 `k ≠ spec.batch` 时捕获期形状不匹配；改成按
  图自己的批宽取。参考后端与假后端不校验样本形状，所以此前
  `LMRunner(device_state=True)` 只服务过单条生成。
- 每步的 logits 整块取回一次（原来逐行 `to_host`，`k` 次 D2H）。

**口径**

- A800-SXM4-40GB 空闲运行，测量前后 `nvidia-smi` 均为 `0 % / 10 MiB`。
- 24 层 Mamba，`d_model=1024`、`d_state=16`、`vocab=32000`、`steps=32`、
  32-token prompt，每条生成 32 token；两侧都 `resident=True`，只切
  `device_state`；`early_stopping=False` 让每行都走满 32 步。预热一轮后测两轮
  取最快。
- 逐 token 输出两侧**完全一致**（`num_beams=1/2/4`）。

| beams | 每步状态 | 宿主 ms/步 | 设备态 ms/步 | 解码加速 | 宿主 tok/s | 设备态 tok/s |
|---|---|---|---|---|---|---|
| 1 | 3.56 MiB | 12.98 | 11.63 | 1.12× | 77.0 | 86.0 |
| 2 | 7.12 MiB | 22.07 | 18.19 | 1.21× | 45.3 | 55.0 |
| 4 | 14.25 MiB | 36.01 | 28.40 | 1.27× | 27.8 | 35.2 |

**HIP 同套配置**（WSL2 / ROCm 7.2 / RX 7650 GRE，24 层，每条 16 token；每个
格子一个独立进程，原因见"顺带发现"）：

| beams | 宿主 ms/步 | 设备态 ms/步 | 解码加速 | 宿主 tok/s | 设备态 tok/s |
|---|---|---|---|---|---|
| 1 | 46.897 | 42.861 | 1.09× | 21.32 | 23.33 |
| 2 | 54.544 | 50.575 | 1.08× | 18.33 | 19.77 |
| 4 | 69.630 | 70.891 | 0.98× | 14.36 | 14.11 |

**读法**

1. 收益来自"少做 `k` 次整块搬运 + 96 次逐数组调用"，不是带宽：`k=4` 省下
   7.6 ms，`k=1` 只省 1.3 ms（3.56 MiB 在这一代 PCIe 上往返不到 0.6 ms）。
2. 设备态多付一次 replay：`_swap_state` 单次 **2.4 ms**（48 个 `embedding` 节点，
   每个都有自己的下标守卫要读回）。A800 上 `k=4` 还剩 2/3 的收益；HIP 上
   replay 更贵，`k=4` 打平（70.9 vs 69.6）——**HIP 的 `num_beams=4` 目前不该
   开设备态**。
3. 单条 decode 的设备态是 ≈6.0 ms/步，束搜索 `k=1` 是 11.63 ms/步：差值里
   2.4 ms 是重排 replay，≈3.3 ms 是束搜索自己的打分（32k 词表上的
   `log_softmax` + 两次 `argpartition`），这部分两条路径都要付。
4. 所以下一个动作不是"再省一点状态"，而是**把重排并进 decode 图**：
   `(parents, state) → (state', logits)` 一张图、每步一次 replay。
   `_stream_graph` 的状态输入换成一个以 `parents` 为下标的 `embedding`
   就能得到，本轮先留下这把尺子。

**与 5.281 的宿主列不可横比**

同一段 decode 代码今天在 A800 上量到 8.7 ms/步（驻留）与 106.8 ms/步
（`resident=False`），而 5.281 记的宿主是 77.3 ms/步并标注 `resident=True`。
两个量级今天都复现得出来，说明上一轮的宿主列取自"未驻留量级"的配置或机器
状态。本节只做同一口径内的比较（两列都 `resident=True`），不对 5.281 的
绝对值下推论。

**顺带发现（HIP，与本节改动无关）**

1. 同一进程里起第二个 `LMRunner`（即使换一个后端实例），24 层 /
   `d_model=1024` 的设备态会给出 NaN logits，或报"捕获期不能申请新显存：
   arena 容量没覆盖捕获轮"。纯 `generate`（不涉及束搜索）同样复现；一个进程
   一个配置就正常。工作方式是每个配置单独起进程。
2. HIP 在 `d_model=512 / 12 层 / vocab=8192 / batch=2` 上 `generate_batch`
   直接出 NaN（同配置 batch=1 正常，`d_model=1024` 的 batch=2 正常），是另一个
   待查的形状相关内核问题。

**边界**

- 重排图按束宽捕获，与 decode 图同生命周期，`close()` 一并释放；
  `resident=True` 仍是前提。
- prefill 仍是普通执行，第一步 decode 会把状态上传一次；末步状态不回传。
- 死槽仍随 pad 推进（图是固定 batch），父行 0，它们的 logits 被丢弃。
- HIP 只有 8 GB：24 层 / `d_model=1024` 时一张捕获计划里的权重常量副本约
  750 MB，设备态一次只够一个会话。

### 5.283　把重排并进 decode 图：一张计划、每步一次 replay

**问题**

5.282 把束搜索的状态重排搬进显存，但每步仍是两次 replay：先跑一张只做
`embedding(state, parents)` 的 48 节点图，再跑 decode 图。5.282 把这 2.4 ms
记成"replay 的固定开销"，并留下尺子：把重排并进 decode 图
（`(parents, state) → (state', logits)`），每步只剩一次 replay。

**做法**

- `build_mamba_layers` 增加 `state_source(name, shape) -> (value, feed)` 钩子：
  状态不再固定由 `g.input` 产生，调用方可以交回"已有 Value + 要 feed 的图输入"。
  `build_language_model_stateful` 原样透传。
- `_stream_graph(..., state_swap=True)` 多建一张 `parents (B,)` 输入，每个状态
  先 `embedding(g.input(name + "__in"), parents)` 再进层；`inputs` 里状态键仍是
  逻辑名（`l0_conv0`），`inputs[name].name` 是重放要写的 `l0_conv0__in`。
- `LMRunner._beam_plan_for(beam_width)` 捕获这张融合图；`_beam_step(parents, ids,
  state)` 一次 replay 同时完成重排与推进。`_swap_plan_for` / `_swap_state` 删除，
  束搜索与单条生成各占一个计划槽位。
- `_beam_decode` 的设备分支从"`swap_state` + `device_step` 两次调用"收成一个
  `device_step(parents, ids, state)`；主机路径（`device_state=False`）一字未动。

**口径**

- A800-SXM4-40GB 空闲，前后 `nvidia-smi` 均为 `0 % / 10 MiB`。
- 24 层 Mamba，`d_model=1024`、`d_state=16`、`vocab=32000`、`steps=32`、
  32-token prompt，每条生成 32 token；`early_stopping=False`，`resident=True`，
  只切 `device_state`。预热一轮、测两轮取最快；每个格子一个独立进程。
- **同机同口径对照**：把 `920aa2c`（5.282）检出到另一个工作树，跑同一份脚本。
  两版的宿主列互相吻合（91.8 / 111.6 / 122.2 vs 92.9 / 117.4 / 128.6），
  设备列才是本节要比较的差值。
- 逐 token 输出两侧完全一致（`num_beams=1/2/4`，两版都一样）。

| beams | 5.282 设备 ms/步 | 5.283 设备 ms/步 | 变化 | 5.283 tok/s | 同机宿主 ms/步 | 加速 |
|---|---|---|---|---|---|---|
| 1 | 14.818 | 14.010 | −0.81（−5.5%） | 71.38 | 92.905 | 6.63× |
| 2 | 21.947 | 21.032 | −0.92（−4.2%） | 47.55 | 117.420 | 5.58× |
| 4 | 33.164 | 31.662 | −1.50（−4.5%） | 31.58 | 128.622 | 4.06× |

（5.282 与 5.283 两列都是同一台机器、同一脚本、同一时间段；`tok/s` 与
5.282 同口径，按 `1000 / ms_per_step` 计。）

**读法**

1. 合并计划确实更快，但只快 4～5%，不是 5.282 预估的 2.4 ms。原因在成本
   结构：重排的主体不是"一次 replay 的边界"，而是**48 个 `embedding`
   节点各自的越界守卫**——每次 replay 后 `CudaGraphPlan._check_guards` 要
   对每个 guard 做一次 `int(err.item())`。合并前后 guard 总数都是 49
   （重排 48 + 词表 embedding 1），所以只省下第二次 replay 的流同步与输入
   暂存（0.8～1.5 ms）。
2. 顺带纠正 5.282 的拆解：那 2.4 ms 不是纯固定开销，主要是 48 次同步读回。
   下一步若要再省，得把同一张图的越界标志汇总成一个标量（或 replay 前一次
   清零、一次读回），预计再省 ~2 ms；不改这个，继续拆计划没有意义。
3. 同机宿主列是 92.9 / 117.4 / 128.6 ms/步，与 5.282 文档里的
   12.98 / 22.07 / 36.01 差 7×。同机旧版就是 91.8，说明 5.282 的宿主列不是
   这台机器 / 这套环境的同口径结果；**本节只做同机两版之间的比较**。
4. 设备态相对宿主 6.6× / 5.6× / 4.1× 的加速，来自主机路径每步 96 次逐数组
   往返（48 次 `to_array` + 48 次 H2D）；这个量级与 5.282"未驻留量级"的
   106.8 ms/步相符，与 12.98 不符。跨条目横比仍按 5.282 的告诫作废。

**边界**

- 融合图按束宽捕获，与单条生成的 decode 计划分开；`close()` 一并释放。
- `resident=True` 仍是前提；prefill 仍是普通执行，第一步 decode 上传一次状态。
- `parents` 只有 `k` 个 int32 上传；死槽仍随 pad 推进，父行 0。

### 5.284　数据并行接上设备侧归约：`ncclCommInitAll` + 原地 AllReduce

**问题**

`DataParallelTrainer` 的 `collective="device"` 接缝从 5.228 起就在（provider 注入、
`DeviceAllReduce` 契约、`overlap` 声明），但一直没有内置实现：请求设备侧归约却
没注入 provider 时只能抛 `DeviceCollectiveUnavailable`，实际能跑的跨卡归约仍是
"每卡 D2H → 主机求和 → 每卡 H2D"。

**做法**

- `train/nccl.py`：登记 `ncclCommInitAll`（NCCL / RCCL 同名），新增
  `init_all(devices)`——单进程内一次建好 world 个 communicator，不需要 unique id
  分发。`NcclComm._from_handle` 补齐已有 handle 的字段。
- `train/parallel.py`：新增 `NcclDeviceAllReduce` 与工厂
  `nccl_device_collective(backend, devices)`。provider 对每个参数、每张卡发起
  一次原地 `AllReduce(op="sum", stream=0)`；`reduction="mean"` 在设备上补
  `array /= world_size`——绑定刻意不登记 `ncclAvg`，平均在哪一侧除留给调用方。
  结束把当前设备还原成进入时的那张卡。
- 不自动启用：没有库、没有多卡、或后端不是设备后端时，仍然显式报
  `DeviceCollectiveUnavailable`，错误消息点名这个工厂。多进程 DDP
  （`DistributedDataParallel`）仍只接受主机态，跨卡归约仍走主机中转——它每个
  rank 一个进程，用不了 `ncclCommInitAll` 建出来的那组 communicator。

**验证**

| 面 | 结果 |
|---|---|
| 离线（假 comm） | 每名字 × 每卡各一次 `sum`、mean 的原地除、非法 reduction 拒绝、`close` 销毁全部 comm、缺库时抛 `NcclUnavailable` |
| HIP world=1（RX 7650 GRE） | `ncclCommInitAll` 可建；sum / mean 均恒等；设备归位 |
| A800 world=1 | `TestNcclProviderOnCuda` 通过 |
| 两卡数值（sum / mean） | 用例已写（`TestCudaDataParallel`），**当前没有多卡机器，未执行** |

**读法**

- 归约留在设备上，主机侧只剩 `DataParallelTrainer` 统一取回的那一次 D2H；
  相比主机中转省掉"每卡一份 D2H + 每卡一份 H2D"。
- 只支持单进程多卡（`DataParallelTrainer`）。多进程 DDP 若要设备侧归约，得由
  调用方按 `ncclCommInitRank` 自己建 provider 注入，本版不提供。
- 计算/通信重叠仍未实现，`overlap` 恒为 `False`。

**边界**

- `init_all` 的空设备列表由 provider 入口拒绝（`ValueError`）；库缺失与
  `ncclCommInitAll` 符号缺失都是 `NcclUnavailable`。
- provider 不持有训练状态，`close()` 只销毁 communicator；可作上下文管理器。

### 5.285　服务面的三件事：认证、限流、`stop` 参与调度

**问题**

`serve/openai.py` 从落地起就只有协议与并发：任何能连上端口的人都能提交生成，
`max_pending` 只能保护队列，挡不住"不停地提交短请求"；`stop` 只在响应层截断
文本——引擎会继续生成到 `max_tokens`，占着槽位产出一份没人要的结果，排队的
请求跟着一起等。

**做法**

- **认证**：`OpenAIService(..., api_key="sk-...")`。`/v1/*` 要求
  `Authorization: Bearer <key>`，用 `hmac.compare_digest` 比较（`==` 会在
  首个不同字节处提前返回，泄漏密钥长度与公共前缀）。失败 401
  `authentication_error`，带 `WWW-Authenticate: Bearer`。`/healthz` 免认证：
  负载均衡器与容器编排不持有业务凭证，探活不该因此变红。
- **限流**：`rate_limit`（每窗口请求数）+ `rate_window`（秒，默认 60）。
  滑动窗口记录时间戳，而不是固定窗口——后者在窗口边界允许两倍突发。超限
  429 `rate_limit_error` + `Retry-After`（向上取整到秒）。判定在解析请求体
  之前，但**在认证之后**：无效 key 不消耗配额，否则猜密钥也能把正常调用方
  的额度打光。时钟可注入（默认 `time.monotonic`），测试用假时钟逐位确定。
- **`stop` 参与调度**：驱动线程在每个解码步**之前**用
  `ServingEngine.progress()` 的快照求"最长可解码前缀"，命中任一 `stop` 就
  `engine.cancel(rid)`，槽位本轮即可准入新请求；已生成的 token 仍作为结果
  交给调用方，`job.stop_hit` 让 `finish_reason` 是 `"stop"`。
- **流式的 hold-back**：`_sendable` 把"可能是 stop 开头"的尾部压住不发。
  逐 token 推送时前缀已经发出去了，命中 stop 只能少发、不能撤回；压住
  最后 `len(stop)-1` 个字符，命中就永远不发，没命中就随下一帧补发。
- `submit_tokens(..., stops=...)` 拒绝空字符串：它在 `str.find` 里永远命中
  位置 0，客户端会得到一条空回复，请求也会立刻被取消。

**验证**

| 面 | 判据 |
|---|---|
| 401 | 缺头、错 key、`Basic` 方案、无 `Bearer` 前缀四种都 401 + `authentication_error` + `WWW-Authenticate` |
| 免认证面 | `/healthz` 200；`/v1/models` 无 key 401、有 key 200 |
| 429 | 额度 2 时前两次 200、第三次 429 + `rate_limit_error` + `Retry-After ∈ [1, 10]`；时钟拨过窗口后恢复 |
| 不占配额 | 三次无效 key 的 401 之后，仍能完整用掉两次额度 |
| 非流式 stop | 探测出的前两个 token 作 stop：`completion_tokens == 2`、`finish_reason == "stop"` |
| 流式 stop | 帧拼接为空（hold-back 生效）、末帧 `finish_reason == "stop"`、`completion_tokens < 8` |
| 参数 | 空字符串 stop 与含空串的数组都 400 |
| 配置 | `api_key=""`、`rate_limit=0`、`rate_window=0` 构造期报错，`rate_limit=1.5` 报 `TypeError` |

全量 `unittest discover`：2843 条 OK（skipped=437）；`ruff check` 全过。

**读法**

- stop 的检测粒度是 **token**：跨 token 的 stop（"llo" 落在两个 token 里）会
  多生成到完整命中为止，但不会多到 `max_tokens`。取消发生在采样后的下一个
  调度点，因此最多多一个已采样 token。
- 限流是**进程内全局**的：多进程部署时每个进程各有一份额度，别把它当集群
  配额。需要按密钥/路径分别计量时，注入自己的准入逻辑，而不是改这里的窗口。
- 认证是**静态单密钥**：没有轮转、吊销与审计。它是"别让端口裸奔"的下限，
  不是多租户鉴权。

**边界**

- `api_key=None`（默认）时认证关闭；`rate_limit=None` 时限流关闭。两者都不
  改变既有调用方行为。
- 被限流/未认证的请求不读 body、不入队、不占解码槽位。
- `Retry-After` 是建议值（最早一条出窗的时间），调用方可以更早重试，但会
  继续收到 429。

### 5.286　`gather` 与融合交叉熵：训练从 MSE 换成真 CE 的前提

**问题**

Nucleus 的训练入口此前只能拿 MSE 当损失——AetherFlow 里没有"按标签取 logit"的
算子，也没有交叉熵。只补一个 `gather` 也能在 IR 里拼出 CE，但要把
`logsumexp(logits)`、`gather(logits, target)`、`sub` 与 `reduce_mean` 连起来，
`[B, T, V]` 的概率或中间量会被物化一次以上：1B 参数、`V=32k`、`T=2048` 时
一份 `[B,T,V]` 就是 256 MB，反向再重算一遍就是第二份。所以两个一起补，
CE 直接做成**融合算子**。

**做法**

- **`gather(x, idx, dim=-1)`**：语义与 `torch.gather` 一致，但形状约束收紧成
  "`x` 与 `idx` 秩相同、除 `dim` 外每一维相等"，输出形状即 `idx.shape`。
  收紧之后内核只需要三个运行期长度：把 `x` 视作 `(outer, x_dim, inner)`、
  `idx`/`out` 视作 `(outer, idx_dim, inner)`，扁平下标 `p` 解成
  `i = p % inner`、`t = p / inner`、`o = t / idx_dim`，`x` 的偏移是
  `(o * x_dim + v) * inner + i`。这样不必把 shape 表传进设备。
  `dim` 支持负值（构图期加秩再校验），越界直接 `ValueError`，不用 `%` 静默回绕。
- **`gather_backward(dy, idx, {x_shape, dim})`**：反向是 scatter-add，
  重复下标必须累加（`torch.gather` 的反向就是这个语义）。设备端用
  `%ATOMIC%(&dx[...], ...)`，因此这条算子在能力面里标 `deterministic=False`；
  参考端用 `np.add.at` 给出同一份真值。`dx` 先整块清零，没有被索引到的位置是 0。
- **`cross_entropy(logits, target)`**：`logits` 最后一维是词表，`target` 与
  `logits.shape[:-1]` 同形、dtype 为 i32 / i8，输出是**逐行** loss（不是标量）。
  内核一块一行、两遍块归约：第一遍求行最大值，第二遍求 `exp(x - m)` 的和，
  0 号线程写 `m + log(s) - logits[row, target[row]]`。整个过程不物化
  `[rows, vocab]` 的概率矩阵，也不缓存 `logsumexp`。
- **`cross_entropy_backward(logits, target, dloss)`**：同样两遍块归约**重算**
  行统计，逐列写 `(p - onehot) * dloss[row]`，`p = exp(x - m) / s`。重算比缓存
  多一遍读 `logits`，但省掉一份与 `logits` 等大的概率缓冲——反向本来就要读一遍
  `logits`，缓存版本还要再读一遍概率。`dloss` 的 dtype 必须等于 `logits`，
  形状必须是 head 形状，两处都在构图期拒绝。
- **注册面**：四个算子都进 `GPU_OPS`（设备承接 68 个）、`_INDEX_ARGS`
  都是 `frozenset({1})`（第 1 个输入是整数下标）、f16 支持面跟随 `_OPS`
  （新算子默认在 f16 集合里），bf16 **没有**加进 `BF16_OPS`，因此 bf16 图会
  在构图期显式拒绝而不是走默认容量。CUDA 与 HIP 共用
  `backends/cuda/codegen.py` 的四份模板（HIP 直接 import 同一批），越界由内核写
  `err` 标量、`_check_ids` 转成 `BackendCapabilityError`，捕获期登记进
  `_capture_guards`、replay 之后复查。
- **自动微分**：`gather` 的 VJP 只有 `idx` 这一个数据依赖以外的形状信息，
  所以规则里把 `x_shape` 与 `dim` 作为属性传给 `gather_backward`；
  `cross_entropy` 的 VJP 直接返回 `cross_entropy_backward`。两个 backward
  算子刻意**没有** VJP（二阶导报"缺规则"），与 `conv2d_*_grad` 同口径。

**验证**

| 面 | 判据 |
|---|---|
| gather 前向 | 末维 / 非末维与 `np.take_along_axis` 逐位相等；i8 下标可用 |
| gather 反向 | 重复下标与 `np.add.at` 逐位相等，并与中心差分一致 |
| gather 拒绝面 | 正 / 负越界、除 `dim` 外形状不等、浮点下标，构图期或执行期报错 |
| CE 前向 | 与 `-log(softmax(logits))` 逐位相等；`1e4` 量级 logits 不溢出 |
| CE 反向 | 与 `(softmax - onehot) * dloss` 逐位相等；autodiff 对数值梯度一致、行和为 0 |
| CE 拒绝面 | `target` 越界、`dloss` dtype 与 `logits` 不一致 |
| 跨后端 | conformance 新增 5 条形态：`gather_last` / `gather_dim1` / `gather_backward` / `cross_entropy` / `cross_entropy_backward` |
| 内核几何 | `TestIndexMath` 用 numpy 复刻内核公式（三轴偏移、反向 scatter-add），无 GPU 也能跑 |

本地 `unittest discover`：2865 条 OK（skipped=444）；`ruff check` 全过。
A800-40GB 上 `TestIndexMath` 与 `TestCudaGatherCrossEntropy` 共 13 条全部通过
（含 f16 前向 / 反向、i8 下标、重复下标的原子加、越界拒绝），
`Ran 13 tests ... OK`；该机全量 2857 条里 123 条 error 全部来自前端用例
（torch 2.5.1 不在兼容矩阵 2.9–2.13 内，`UnsupportedTorchVersion` 显式拒绝），
与本轮改动无关。HIP 侧本轮未复跑这两组，模板与 CUDA 同源。

**读法**

- 训练里常见写法是 `reduce_mean(cross_entropy(logits, labels))`：CE 输出逐行
  loss，标量化交给已有的 `reduce_mean`，不在 CE 里写死归约方式。
  `masked_lm_loss_indexed` 那条老路仍然保留，它借 `embedding` 的查表语义取目标位，
  与 CE 的区别是后者自带数值稳定的 `logsumexp`。
- `gather` 的 `dim` 是属性而不是输入，因此同一张图不能在不同轴之间切换；
  需要多轴时构图期分别建节点。
- 反向的 atomic 累加意味着**同一张图连跑两次的浮点结果可能最后一位不同**
  （重复下标越多越明显），这与 `embedding_backward` 是同一条口径：能力面标
  `deterministic=False`，要逐位复现就按参考语义在主机侧对拍。

**边界**

- `gather` 不支持广播：`idx` 的每一维（除 `dim`）必须与 `x` 相等。
  `torch.gather` 允许的广播形态在这里被拒绝，理由是内核省掉 shape 表。
- `cross_entropy` 只接受"最后一维是类别"的布局，`[B, T, V]` 与 `[B, V]` 都行；
  `target` 不能是 onehot 或概率分布，也不做 `ignore_index` / label smoothing /
  类别权重。需要这些时在 IR 外用 mask 与 `reduce_sum` 组合。
- 两个 backward 算子没有二阶导；对 CE 求 HVP 会得到"缺少 `cross_entropy_backward`
  的 VJP 规则"的显式错误，而不是零。
- CE 的数值稳定性来自行最大值归约（与 `softmax` 同一条路径），但 `logits` 若已
  含 `inf` / `NaN`，结果按 IEEE 语义透传，不做掩码。

### 5.287　训练步"双峰"与共享卡干扰：归因尚待独占复测

**现象**。`batch=128, d=1024` 的两层 MLP + CE + SGD，`graph_mode="auto"` 下同一个
训练步在 **0.55 ms 与 3.05 ms** 之间**成块跳变**（连续 4 轮 3.07 然后连续 4 轮
0.55），而 `graph_mode="off"` 稳定在 1.43 ms；同机 torch 是 0.69～0.73 ms。
`graph_captures=1 / graph_replays=42`，捕获失败原因为空，所以不是"每步重新捕获"。

**逐段拆解**（每步 10 次平摊，慢态）：

| 段 | 耗时 |
|---|---|
| 输入下拷 + 图 launch（host 侧） | 0.10 ms |
| `capture_stream.synchronize()`（等图） | 2.65 ms |
| 参数更新 8 个原地 kernel（事件计时） | 0.054 ms |
| `cudaDeviceSynchronize`（基准口径） | 2.51 ms |

两项局部交叉验证说明，部分样本中设备计算明显短于整个调用：

- 用 `cudaEvent` 量**图本体**（只 launch、不碰输入缓冲）：**0.272～0.282 ms**，
  30 次无一例外；
- 小 kernel（64 个 float）与 4 MiB D2D 拷贝在默认流 / 捕获流 / 新建流上都是
  **6～16 µs**，与流无关。

但同一条测量里，图 launch 在三条流的每 15 次中各出现**一次 2.2～2.8 ms**——
说明停顿是全设备级的，不归任何一条流。

**已确认的环境干扰**。`nvidia-smi --query-compute-apps` 显示这块 A800 上还跑着
`train_photon.py`（20.7 GB；当时全卡 util 79%、功耗 260 W、clocks.sm 恒定
1410 MHz）。共享资源是调度停顿的合理解释，但没有进程级时间线，不能把每次
2.7 ms 停顿归给某个具体 kernel，也不能据此排除框架自身开销。

**同步结构需要区分**。`wait_event` 提交设备流依赖，不等同于主机阻塞的
`synchronize`，此前把它算成第三个主机同步点不准确。AF 图输出路径会等捕获流，
随后读取 guard 标量，基准再在 burst 尾做设备同步；torch eager 路径在 burst
尾做显式设备同步。两边结构不同，可能对共享负载有不同响应，但没有测出各段
增量成本。此前“0.24× 只是环境伪影、不是框架缺陷”的断言撤回：现有数据不能
区分环境影响和框架开销各占多少。

**处置**。`benchmarks/framework_compare.py` 启动时查一次
`--query-compute-apps`，发现其他计算进程就打印警告。查询覆盖可见 GPU，未按
目标卡过滤；若确实共享目标卡，绝对耗时与比值都可能失真。交替执行只减轻
顺序偏差，不能保证消除干扰。用户选择等待原训练自然结束，不暂停或终止它。

**遗留**。这台 A800 没装 `nsys` / `ncu`，没有 kernel 级时间线；事件和分段
wall clock 只能给局部证据。干净数字等卡空出来后重测，不能把本节当独占成绩。

### 5.288　训练对照审计：冗余反向图、输入复制与同步边界

2026-11-21 的代码审计、CPU 回归与 A800 空闲窗口复测。原占卡训练于远端
日志时间 22:01:16 自然结束；等待日志出现 `CLEAN_DONE`。旧版本复测与随后
两批新基准运行前后均检查计算进程；后续剖析期间又启动了别的训练，污染样本
单独排除。以下结果只覆盖本例，不代表其他模型或 torch.compile。

**发现与修正**：

1. 训练基准直接执行 `backward(graph, wrt=params)`，没有接已有的
   `backward_pipeline()`。`batch=128, d=1024` 时原始执行图 30 节点，经标准
   管线变为 22 节点：matmul 从 6 个降到 5 个，去掉未输出的 x 梯度、无用的
   前向交叉熵与配套广播/归约等。默认启用标准管线；`--raw-backward` 保留原始
   路径作消融。原始数字不能直接与新默认数字拼接为提速结论。
2. CUDA 自动图每步将外部设备输入复制到捕获用静态缓冲。本例全部输入
   8,921,604 字节（约 8.51 MiB），其中参数 8,396,800 字节（约 8.01 MiB）。
   “已经在设备”不等于零复制；参数仍在图外由 Python/CuPy 更新，4 个参数各
   有缩放和减法两次启动。这些是实际工作，耗时待测。
3. 普通执行逐输入校验、绑定，逐组 Python 派发并检查输出 dtype/shape。
   交叉熵前向和反向都检查越界标志并 `err.item()`；旧原始图包含两次校验，
   DCE 后仅保留反向校验。图模式会等捕获流并逐项读取 guard。不能为了提速
   删除输入越界契约，也不能把 `wait_event` 当主机同步。
4. 原对拍只看 dw1 的 4×4。现在计时前校验完整 loss、全部四份参数梯度与
   实际设备路径的一步参数结果，再恢复初值。CPU 参数各自复制，避免
   `torch.as_tensor` 与 numpy 共享存储造成两框架互相修改参数。
5. 训练 `--compile` 以前被静默忽略，现在显式拒绝；当前训练只对照 torch
   eager。计时遵从 `--warmup`，逐轮轮换先后次序，训练主表使用中位数并输出
   全部轮次和最好值。删除 torch allocated 与 CuPy 池步后增量的伪峰值比较。
   设备驻留调用计入 Python、内部复制和同步，不能称为“只计内核”。

**验证范围**：本地 `tests.test_benchmark` 29 条通过，覆盖先后次序、模式拒绝、
原始与优化反向图 × 有无参数更新的完整对拍与输入不被修改。默认尺寸 CPU CLI
跑通，Ruff 通过。CUDA 每次计时前均做完整 loss、主机/设备路径全部梯度及
一步参数结果的对拍；自动图捕获成功，无回落。图节点数与字节数由真实 IR 统计。

**A800 空闲窗口复测**。代码 `08045a0`，形状 `batch=128, d=1024`、f32、
学习率 1e-6。双方输入/输出设备驻留，图外 SGD、每个 burst 尾设备同步；
AF 接口内部同步保留。预热各 10 次，逐轮轮换先后顺序，以下均为 wall clock
每步中位数，单位 ms；比值为 torch/AF，超过 1 表示本例 AF 更快。

| 批次 | 图 / 模式 | 轮数 × burst | AF | torch eager | torch/AF |
|---|---|---:|---:|---:|---:|
| 甲 | 原始 / off | 7 × 30 | 1.2470 | 0.5882 | 0.47 |
| 甲 | 优化 / off | 7 × 30 | 1.0984 | 0.5967 | 0.54 |
| 甲 | 优化 / auto | 7 × 30 | 0.5356 | 0.5822 | 1.09 |
| 甲 | 优化 / auto，不更新 | 7 × 30 | 0.4373 | 0.5223 | 1.19 |
| 乙 | 原始 / auto | 9 × 40 | 0.5548 | 0.5852 | 1.05 |
| 乙 | 优化 / auto | 9 × 40 | 0.5093 | 0.5859 | 1.15 |
| 乙 | 优化 / off | 9 × 40 | 1.0731 | 0.5827 | 0.54 |

两批结束均未列出其他计算进程，但不是全程独占锁，不能排除短时外部负载。
甲批原始/off 与优化/off 的中位数相差约 11.9%，乙批原始/auto 与优化/auto
相差约 8.2%；这是顺序运行的消融，不作严格配对因果估计。不同配置下 torch
自身也有波动，保留逐轮样本与范围判断，不用最好值挑成绩。

原始轮次，按运行顺序记录：

- 甲原始/off：AF `[1.2561,1.2449,1.2451,1.2691,1.2470,1.2466,1.2478]`；
  torch `[0.5936,0.5892,0.5882,0.5859,0.5877,0.5977,0.5834]`。
- 甲优化/off：AF `[1.0887,1.0668,1.0778,1.1149,1.0984,1.1073,1.1353]`；
  torch `[0.5908,0.5892,0.5889,0.6085,0.6036,0.5993,0.5967]`。
- 甲优化/auto：AF `[0.5412,0.5387,0.5314,0.5356,0.5361,0.5316,0.5307]`；
  torch `[0.5845,0.5810,0.5883,0.5815,0.5831,0.5773,0.5822]`。
- 甲优化/auto无更新：AF `[0.4460,0.4534,0.4373,0.4360,0.4398,0.4357,0.4361]`；
  torch `[0.5269,0.5331,0.5282,0.5223,0.5155,0.5160,0.5164]`。
- 乙原始/auto：AF `[0.5548,0.5805,0.5529,0.5558,0.5520,0.5540,0.5512,0.5634,0.5683]`；
  torch `[0.5813,0.5852,0.5817,0.5863,0.5852,0.5843,0.5867,0.5863,0.5805]`。
- 乙优化/auto：AF `[0.5093,0.5078,0.5155,0.5066,0.5258,0.5068,0.5127,0.5067,0.5105]`；
  torch `[0.5867,0.5876,0.5913,0.5863,0.5840,0.5841,0.5859,0.5854,0.5817]`。
- 乙优化/off：AF `[1.1037,1.1155,1.0731,1.0688,1.0773,1.0771,1.0615,1.0665,1.0730]`；
  torch `[0.5833,0.5857,0.5894,0.5827,0.5851,0.5817,0.5827,0.5827,0.5814]`。

**分段与普通路径问题**。甲优化/auto 开启的累计 profile（含预热、校验及计时）
每步 upload 0.097 ms、launch 0.010 ms、sync 0.267 ms、guards 0.031 ms。
upload 包含 CPU 调用与异步 D2D 提交，sync 包含实际 GPU 工作，不能把它们
直接解释为纯复制时间和纯等待损耗，也不能与主表作可加分解。

普通优化图仍有 21 个执行组，其中 5 次 matmul、5 次 transpose、2 次
reduce_sum；转置路径会 `ascontiguousarray` 物化连续副本。首轮主机 cProfile
的 100 次无更新调用有 63,904 次函数调用，自动图约 7,804 次；这些次数能说明
派发结构差异。累计时间中的 `err.item()` 包含前序 GPU 工作，插桩也会扰动
发射速度，**不把其累计时间当校验本身成本**。后续已出现共享负载，不能把
两份 cProfile 总时间当空闲性能对照。

**污染样本排除**。第二次临时消融显示约 2.7 ms；随后确认新 PID 29096 正在
跑 `train_photon_fm.py`，util 99%。该消融与设备事件约 0.47 ms 的样本全部不作
复制开销结论；未暂停或终止该任务。探针最初错误读取 `CudaGraphPlan.inputs`，
实为私有 `_buffers`，按真实定义修正后通过固定输入的完整梯度一致性检查。

**当前结论与下一步**：旧慢态受共享负载影响，但框架普通路径仍比 eager 慢
约 1.8 倍；优化自动图在这个小模型上已接近或略快于 eager。已消除冗余反向
工作；下一项优先验证 GEMM 转置标志能否避免 5 次转置物化、减少逐组派发和
临时分配，并审计保留输入越界语义的异步校验方案。图模式仍有每步输入复制，
但贡献比例未取得干净消融；未改默认图模式，也未声称超过 torch.compile。

### 5.289　CUDA 二维转置并入 GEMM：减少复制与执行组

2026-11-21，代码 `0120563`。仅 CUDA 开启：相邻的一个或两个二维转置，
其输出只有紧随的 matmul 一个消费者且不作为图输出时，合并为执行组。
源矩阵保持连续，直接给 cuBLAS GEMM 设置转置参数；leading dimension 取
物理源矩阵行宽，不能取转置后逻辑宽度。f32 用 Sgemm，f16/bf16 用 GemmEx
并保留 f32 累加。批量、空矩阵、符号维、多消费者、独立图输出和 HIP 不参与。
不满足运行时布局条件时回退逐节点。未将全局 transpose 改成非连续视图。

本例五个转置中消除三个：执行组 21 → 18，余下两个不满足相邻融合边界。
没有改变输入复制、越界守卫、图外 SGD 或默认图模式。
`run_device(release=False)` 沿用已有融合组环境语义：组尾输出可见，组内
未物化值不进环境；没有通过额外删除环境键来优化。

**正确性**：本地 42 项计划、执行快路径和基准测试通过，Ruff 通过；A800 完整
CUDA 后端 372 项测试通过（其中 1 项跳过）。新增测试覆盖左/右/双转置、
f32/f16/bf16、off/auto（显式断言实际捕获）、container/tiered、release 开关、
多消费者与图输出拒绝融合、批量/空矩阵回退、非连续源拒绝快路，以及新增图
输出后计划失效。训练基准保留完整 loss、四份梯度和一步更新与 torch 的校验。

**同进程开关交替消融**：A800，batch=128、d=1024、f32、反向优化图、含 SGD。
分别创建独立参数副本和后端，仅 `GEMM_TRANSPOSE` 不同；每个配置预热20步，
15轮×40步，逐轮轮换测量顺序。测量前后 GPU 进程列表均为空；无独占锁，
不保证测量每一瞬间都无外部负载。中位数如下，耗时单位 ms：

| 路径 | 关闭融合 | 开启融合 | 耗时下降 |
|---|---:|---:|---:|
| off | 1.0481 | 1.0013 | 4.5% |
| auto | 0.4817 | 0.4645 | 3.6% |

逐轮原始数据：

- off关闭：`[1.065,1.0853,1.0626,1.0461,1.0481,1.0512,1.033,1.05,1.0538,1.0476,1.0477,1.0636,1.0421,1.0415,1.0388]`。
- off开启：`[1.0175,1.0053,1.015,1.0037,1.0041,1.0031,0.9892,0.9864,0.9926,1.0026,1.0013,0.9986,0.9772,0.9955,0.9783]`。
- auto关闭：`[0.5222,0.5325,0.5212,0.4837,0.4895,0.4858,0.4923,0.4817,0.4655,0.4611,0.4638,0.4603,0.459,0.4577,0.4622]`。
- auto开启：`[0.5098,0.5025,0.4915,0.4702,0.4691,0.4672,0.4645,0.4681,0.4534,0.4479,0.4501,0.4467,0.4467,0.4436,0.4463]`。

**torch eager对照补充**：独立进程中每配置预热10步、7轮×30步交替。off中位数
AF 1.0220 / torch 0.5899；auto AF 0.5031 / torch 0.5461。off最后一轮AF有
2.6937 ms尖峰，不能隐去；不与上表跨进程直接作加速归因。

- off AF：`[1.0209,1.0214,1.0391,1.022,1.0139,1.0243,2.6937]`；torch：`[0.5615,0.6356,0.6333,0.6144,0.5678,0.5434,0.5899]`。
- auto AF：`[0.504,0.5113,0.5031,0.5071,0.4993,0.4758,0.4501]`；torch：`[0.5461,0.5379,0.5539,0.5392,0.5977,0.5401,0.5759]`。

结论：本轮有可复现的小幅收益，普通路径仍约比 eager 慢1.7倍。不能声称
已经解决全部框架开销或超过 torch.compile。下一步仍是逐组派发、临时分配和
保留异常语义的守卫汇总。临时消融脚本运行完成，结果入本文后删除；未记录凭证。

### 5.290　中间值回收前移到执行计划

2026-11-21，代码 `7d69133`。`run_device(release=True)` 原先每步扫描全图
建立输入引用计数，再在每组结束后逐边减计数。现在在 `_prepare_plan` 完成后
一次性生成各组的 `release` 元组，执行期只摘掉到期的环境键并调用 `dispose`。
仍按原有节点、输入顺序统计，重复输入只在计数归零处列一次；输入、图输出和
后端缓存常量保留，融合组内未物化值仍由 `env.pop(..., None)` 跳过。回收时点、
调用顺序和通用路径断开 `args` 引用的行为不变，不修改 arena 的别名管理。
`release=False` 不执行回收；图 revision 改变时整份计划一起重建。首轮构建
增加一次生命周期预计算，稳态移除逐步统计，不宣称降低首轮延迟。

**正确性**：新增五项测试，覆盖通用/直连路径、重复边、分支释放顺序、多次执行、
关闭回收后的完整环境、融合组未物化值、后加图输出导致计划失效、准备钩子缓存
常量和符号形状切换。相关回归61项全过；本地全量2880项、跳过446项，其余全过，
Ruff和`git diff --check`通过；测试机同步到`1fb7666`后，Python 3.10上的
执行快路径与生命周期测试25项全过（不使用GPU）。本轮没有执行新的GPU回归：两次检查均有其他
python3任务占用26690 MiB，GPU利用率分别75%与100%，未终止或干预该任务。
5.289的GPU回归仅验证旧提交，不冒充本轮结果。

**主机微基准**：Windows / Python 3.14 / NumPy，基线取`fe4c25a`中的执行器，
新版使用相同NumPy内核并预绑定单节点快路径；输入为f32[16]，连续relu，
`run_device(release=True)`，含输入绑定与环境维护，不含GPU、D2H或SGD。
同进程两份执行器各预热30次，21轮×1000次，逐轮交换先后顺序；计时期间关闭GC，
计时前逐项核对环境键和数值。单位ms/次：

| 节点数 | 基线中位数 | 新版中位数 | 配对耗时变化中位数 |
|---|---:|---:|---:|
| 24 | 0.046678 | 0.040061 | -13.75% |
| 96 | 0.192733 | 0.164496 | -14.68% |

每轮配对定义为`新版/基线-1`，不能由两列独立中位数直接反推。两种规模均21轮
新版更快。这只证明该主机微基准的稳态成本下降，不代表A800训练端到端下降14%，
也不代表捕获后replay加速。待设备空闲后补CUDA/HIP回归与训练同进程交替测量。
临时脚本`af_release_ab.py`在结果入文后删除；无新增依赖，未记录凭证。

### 5.291　执行组入口固化与两轮主机优化的GPU复测

2026-11-21，代码 `c2896f1`。后端 `_prepare_plan` 完成后，统一确定每组的
`fast` 直连记录和单节点缓存常量入口 `cached=(id, array)`。执行期不再重复
检查组长、常量字典及快路径是否初始化。静态形状、dtype、bf16量化和释放位置
保持不变；符号组仍走通用路径，缓存清理或图revision变化会重建这些入口。
常量数组仍由原计划持有，没有新增设备副本；只前移选择，不跳过运行期契约。
新增“入口每份计划只选择一次、清缓存后重新选择”测试，增强常量缓存断言。

**未保留的尝试**：曾在输入 `_bind` 开头加形状元组相等检查，静态输入有收益，
但符号图增加比较成本、主机微基准略退化，已完整撤回。临时对照脚本首次把
SymExpr内部操作码写成`*`，构造失败；按实际定义改为`mul`后12项绑定与异常
诊断对照通过。这只是临时脚本问题，未修改IR实现。执行组入口优化的两次主机
交替测量中，无缓存常量的24/96节点静态图下降约5.5%～7.2%，符号图收益较小；
短常量图轮间波动大，不宣称全部形状稳定加速。下列设备数据才用于训练结论。

**回归**：Windows本地全量2881项、跳过446项，其余全过；Ruff通过。测试机
Python 3.10上的执行快路径、生命周期和常量缓存相关29项通过，随后GPU空闲，
完整CUDA回归372项、跳过1项，其余通过，补齐5.290的CUDA验证。HIP设备本轮
未测试，不宣称HIP已验证。没有终止或干预其他GPU任务。

**同进程三版本消融**：A800-SXM4-40GB，基线`fe4c25a`、生命周期版`3a54a82`、
当前`c2896f1`。动态加载历史执行器的`_plan/run_device/_fast_step`，相同当前
CUDA后端、内核与优化反向图，每版本独立参数。batch128/d1024/f32 MLP，前向+
反向+SGD，设备级同步；三版本完整梯度与一步参数更新对拍通过。每版预热20次，
15轮×40步，轮换三者顺序并隔轮反转；每轮查询其他GPU进程，前后均为空，
结束时利用率0%、显存10 MiB。计时包括接口内部复制、主机派发、图外更新。

| 模式 | fe4c25a 中位数ms | 3a54a82 中位数ms | c2896f1 中位数ms |
|---|---:|---:|---:|
| off | 1.005797 | 0.960074 | 0.948206 |
| auto | 0.446723 | 0.447072 | 0.449014 |

逐轮配对`后/前-1`的中位数：off回收计划 **-4.328%**，本轮入口固化
**-0.957%**，两项合计 **-5.726%**；auto分别-0.567%/-0.145%/-0.213%，
视为无明显变化，不宣称replay加速。配对变化不能由独立中位数直接反推。
原始off样本（ms，按轮排列，保留四位）：

- fe4c25a：[1.0333,1.0175,1.0606,0.9885,0.9850,0.9974,1.0074,1.0268,0.9948,1.0021,1.0058,0.9982,1.0147,1.0578,1.0000]
- 3a54a82：[0.9621,0.9548,0.9581,0.9534,0.9635,0.9649,0.9601,1.0072,0.9517,0.9651,0.9849,0.9736,0.9532,0.9515,0.9512]
- c2896f1：[0.9396,0.9517,0.9697,0.9570,0.9364,0.9403,0.9415,0.9509,0.9751,0.9823,0.9482,0.9410,0.9448,0.9424,0.9588]

原始auto样本：

- fe4c25a：[0.5075,0.5037,0.5060,0.4436,0.4453,0.4461,0.4518,0.4503,0.4467,0.4430,0.4435,0.4467,0.4467,0.4486,0.4420]
- 3a54a82：[0.5019,0.5008,0.4985,0.4448,0.4490,0.4483,0.4471,0.4454,0.4438,0.4462,0.4508,0.4497,0.4434,0.4439,0.4446]
- c2896f1：[0.5001,0.5000,0.5096,0.4490,0.4479,0.4447,0.4509,0.4452,0.4515,0.4495,0.4462,0.4441,0.4428,0.4429,0.4559]

**独立进程与PyTorch对照**：当前版本，`framework_compare --training`，
分别off/auto，预热20、15轮×40步，AF与torch逐轮交替，损失、全部梯度、
设备路径及一步参数更新全部对拍。off AF **0.9613** / torch **0.5452** ms，
AF耗时约1.76倍；auto AF **0.4746** / torch **0.5679** ms，AF耗时低约16.4%。
只对比torch eager，不代表torch.compile或其他模型。off有一轮AF 2.2688ms
异常点，保留未剔除；auto有明显随轮次下降，不能把该独立进程数字与前文三版
消融的差直接归因到代码。原始轮次：

- off AF：[0.9608,0.9975,0.9567,0.9730,2.2688,0.9802,0.9613,0.9671,0.9536,0.9607,0.9526,0.9721,0.9486,0.9741,0.9541]
- off torch：[0.5570,0.5394,0.5558,0.5393,0.5595,0.5341,0.5535,0.5344,0.5587,0.5358,0.5431,0.5385,0.5623,0.5452,0.5605]
- auto AF：[0.5093,0.5146,0.5040,0.4962,0.4826,0.4943,0.4884,0.4631,0.4503,0.4746,0.4511,0.4663,0.4506,0.4601,0.4515]
- auto torch：[0.5738,0.5455,0.5814,0.5588,0.5779,0.5679,0.5794,0.5525,0.5778,0.5596,0.5911,0.5689,0.5660,0.5452,0.5649]

结论：最近两项主机优化确实降低普通训练路径耗时，但普通路径仍明显慢于eager；
本轮新增优化的独立收益约1%，不夸大成主机微基准的百分比。下一步应分离图外
参数更新与输入复制成本，再决定捕获边界或派发优化。临时本地与设备消融脚本
在结果入文后删除。GitHub更新主分支时服务端返回`pre-receive hook declined`，
未绕过规则或强推；Gitee和测试机照常同步，无新增依赖，未记录凭证。

### 5.292　attention mask 前导维改用 reshape 降维

2026-11-21，代码 `c4444f5`。设备后端原先用 `raw_mask = raw_mask[0]` 逐个去掉
长度为 1 的多余前导维。numpy / cupy 的整数索引会降维，而天数（Iluvatar）的
`DeviceArray.__getitem__` 只接受切片、拒绝降维索引，传 `(1,1,S,T)` mask 时
直接抛 `TypeError`；一致性套件只覆盖 `(S,T)`，缺陷因此一直没暴露（REMIND 坑 12）。
调用方此前只能自己 squeeze 成 `(S,T)`。

现在把"去掉前导 1"拆成纯形状推导 `_mask_target_shape(mask_shape, keep)`：
前导维含非 1 时返回 `None`，沿用原有 `BackendCapabilityError` 文案；否则给出
目标形状，调用方对数组只做 `cp.ascontiguousarray(raw_mask).reshape(target)`，
全程不出现整数索引。秩不变时不碰数组；前导维连续多个 1 也能一次 reshape 到
`len(batch)+2`。numpy / cupy 的掩码语义与结果逐位不变，天数不再依赖调用方
预 squeeze。非连续 mask 在降维处可能多一次连续化拷贝，之后广播路径本来也要
拷贝，因此不改变输出，只记录这一处可能的额外复制。

**验证**：主机侧新增 2 项形状推导测试（多前导 1、非 1 前导拒绝），
`tests.test_cuda_backend`、`tests.test_attention`、`tests.test_hip_backend`
本地 488 项中跳过 388 项、其余通过，Ruff 通过。测试机（GPU 正被其他任务占用）
上传改动后运行 `TestCudaAttention` 与 `TestAttentionMaskSharing` 共 15 项全过，
包含新增的 `(1,1,1,S,T)` 设备用例；未做性能测量，也未干预占用 GPU 的任务。
天数上机需按 Iluvatar 包流程 `py -3.10 build.py` 后
`install_hardware("dist/iluvatar-0.1.0.zip", replace=True)`，不能直接覆盖
已安装的 `.py`（sha256 校验会失败）。本轮没有天数硬件，未宣称天数实测通过。

### 5.293　硬件包被跳过时保留原因，装配报错附带诊断

`iter_installed()` 的契约是"坏包跳过、不打断别人"：manifest 不合法、文件摘要
不符、依赖不满足，都当作"这个包在这台机器上用不了"。问题在于跳过时把
`HardwarePackageError` 直接吞了，装配层只看到名字不在 `_FACTORIES` 里，于是

    LookupError: 未注册的后端 'iluvatar'；已注册: cuda、hip、reference

"包在目录里但校验没过"与"根本没装"给出同一句话。换机器时直接把源码覆盖进
已安装目录（或 sftp 只替换几个 `.py`）正好命中这个场景：manifest 的 sha256
对不上，包被静默跳过，排查要自己对目录做哈希。

现在把扫描拆成 `_scan()`，产出 `(目录名, record 或 None, 跳过原因)`；
`iter_installed()` 只取 record，新增 `scan_errors()` 返回当前被跳过的
`{目录名: 原因}`。装配层在抛"未注册的后端"时调用它，若同名目录存在就附上
校验失败原因与"重新打包后用 `install_hardware(..., replace=True)` 重装"的
提示。

边界：跳过行为不变，坏包仍不影响其它后端；`scan_errors()` 每次调用重扫一遍
硬件目录，只在**已经是错误路径**上使用，不进入热路径；原因文本直接来自现有
`HardwarePackageError`，不新增异常类型。目录名与包名不一致时（安装目录来自
manifest 的 `name`，通常一致）不猜，只按目录名匹配。

**验证**：新增 2 项测试——`scan_errors()` 能报出"摘要不符"；覆盖已安装
`.py` 后 `get_backend(name)` 的 `LookupError` 里同时含"摘要不符"与
`install_hardware`。`tests.test_hardware_package` 与 `tests.test_backend_discovery`
共 72 项通过，Ruff 通过。未触碰天数硬件，包侧与 `build.py` 无改动。

---

### 5.294　Module / Parameter：把"声明参数 + 构图一次"变成可迁移的一等入口

**动机**：5.25 的 eager 前端解决了"怎么写算子"，模型侧却仍然每写一个都要手工做
四件事：`g.input` 声明每个参数、维护参数名表、拼 `feed`、自己存 `state_dict`。
已有构图函数（如 HelixLM 的 `build_helix_lm`）更尴尬：继续直接喂
`dict[str, ndarray]` 拿不到参数视图，改写成 eager 写法又等于重写模型。

**做法**（`frontend/module.py`，两个类，不引入第二套执行机制）：

- `Parameter(Tensor)`：构图期的 `Value` + 执行期的主机数组。`data` 的 setter 只做
  原地写入，不换数组对象——`ParameterStore` 与优化器持有的是同一个引用。
- `Module`：`param()` 声明图输入与数组（名字用点号前缀 `l0.attn.wq`，初始化种子
  混入完整名，同形状不同层不会拿到同一份初值）；第一次 `__call__` 执行 `forward`
  构图并把返回值登记为图输出，之后只能 `run()` 重放，第二次调用直接报错。
- `adopt(name, value, data)`：接管图上**已经存在**的输入。迁移路径的关键——让
  `build_helix_lm` 照旧自己 `g.input(...)`，构图完成后再把输入包成 `Parameter`。
- 参数视图四个：`arrays()`（引用，交给优化器）、`state_dict()`（深拷贝）、
  `load_state_dict()`、`feed()`（外部输入 + 当前参数，同名冲突直接报错）。
- `Tensor.__getattr__` 把算子表映射成方法：`x.relu()`、`x.reshape(shape=(3, 2))`，
  签名与 `Graph` DSL 逐字一致（`g.op(*inputs, **attrs)`），不另造 PyTorch 风格的
  位置参数版本；实测 IR 文本与手写 `add_node` 完全相同。

**两个必须做对的细节**（都属"单测看不出、训练时静默错"）：

1. `param()` 从 `graph.input()` 拿到的是 `Tensor`，登记 `Parameter` 时必须同时替换
   前端记录（`_replace_tensor`）。漏了它，`backward()` 的缺省求导目标仍是旧
   `Tensor`，梯度能算出来但不会写回参数，表现为"训练一步不动"。
2. `adopt()` 必须**拷贝**外部数组。调用方通常把 torch 的 `state_dict()` 直接
   `.numpy()` 传进来，那是共享内存的视图；不拷贝等于两个框架写同一份权重，双方
   优化器各走一步、参数只前进一次。首次对拍的征兆是第 1 步参数差 2×lr 且方向相反。

**判据要分层**：与 torch 对拍 HelixLM（逐层独立 26 个参数 / 共享权重 15 个）时，
前向 2.9e-8、反向 2.8e-9、resume 逐位 0；但 **100 步 Adam 的逐参数偏差会到 3.9e-4**
——Adam 的更新近似 `sign(grad)`，两侧 1 ulp 的差足以让个别参数的更新方向翻转 ±lr，
再被自适应步长放大。所以短程（20 步）逐参数卡 2e-5，全程（100 步）只看端到端 MSE
的相对偏差（实测 1.6e-4，门限 1e-3）。拿 100 步的逐参数偏差当硬门禁，只会在实现
正确时随 BLAS 版本随机变红。

**验证**：`tests/test_eager_module.py` 17 项（声明、重名、静态形状、构图一次、
adopt 的同图 / 异图 / 未知 / 形状 / 拷贝、state_dict 往返、feed 冲突），
`tests/test_eager_frontend.py` 新增 3 项算子方法用例，两者共 42 项通过，Ruff 通过。
Nucleus 侧 `HelixAFModule` 与 `verify_af_helix_module.py` 按上述判据两档全过。

**明确没做**：没有嵌套子模块（参数名用点号扁平表达，P1 再补）；没有参数共享
（`shared_layers` 由构图函数自己决定）；仍然只构图一次，不是 define-by-run。

### 5.295　训练基础设施在 AMP 与设备态续训下的三个缺口

**动机**：B3 在 HelixLM 上把 DataLoader、梯度累积、checkpoint 续训、AMP 串成
闭环后，三处"单测覆盖不到、但真实训练一定撞上"的问题才暴露出来。三处都没有
新增执行机制，只是把状态搬移与命名规则补完整。

**缺口一：autocast 重写未命名输出**。`Module` 构图会把 `forward` 的返回值
登记成图输出但不给别名，`output_names` 因此退化成值名 `v<id>`。Autocast 重写
图时为保别名，拿"重写后的新值名"与输出名比较：插入 cast 后值重新编号，两者
不再相等，旧值名便被当成显式别名写进新图，与自动名撞车（`名字重复: v245`）。
任何 `Module` 构图 + `mixed_precision` 都会撞上。修法是比较**原值**的 SSA 名
（`name == value.name`），未命名继续不命名。
回归：`tests/test_autocast.py::TestAutocastUnnamedOutput`
（matmul → softmax 之间必插 cast，编号一定错位）。

**缺口二：续训恢复的累加和留在主机**。`load_training_state` 必须在第一次
`step` 之前调用，此时设备计划还没建立，`GradientAccumulator.load_state_dict`
按主机参数表恢复了累加和。设备态续训的第一个微批梯度是 cupy 数组，累加器
却还在主机侧，`xp.asarray(设备数组)` 被 cupy 拒绝（不允许隐式回主机）。
修法：`add()` 发现"已有累加和的 array module ≠ 本微批梯度的 array module"
时，把累加和一次性迁到梯度所在设备。窗口的语义不变——只有设备换了一次。

**缺口三：续训恢复的优化器槽位留在主机**。同类问题但位置不同：
`Optimizer.rebind` 把参数表换成显存数组时只换了 `params`，`m/v` 槽位仍是
恢复出来的主机数组。第一步更新 `m += (1-beta1) * g` 里 `g` 是 cupy、`m` 是
numpy，cupy 的 ufunc 报 `Unsupported type numpy.ndarray`（CScalar 路径）。
修法：`rebind` 在换表后让槽位跟随参数的 array module 迁移；名字与形状校验
保持原样。

**判据**（Nucleus `verify_af_helix_b3.py`，A800 设备态）：

- DataLoader：3 轮 18 步，末轮均值降 27.4%；与 torch 同序轨迹 loss 1.8e-7、
  参数 8.9e-7。
- 梯度累积：`accumulate_steps=2` × 微批 2，窗口均值与整批 torch 的 loss
  1.1e-7、六次更新后参数 1.8e-7；优化器步数恰为 6。
- 续训（中断在第 3 步，窗口里剩 1 个微批）：中断前与续训段 loss 都是 0，
  最终参数 6.0e-8；游标恢复到 epoch=1/batch=3，优化器步数 4 与不中断一致。
- AMP f16（含 GradScaler）：首步 loss 相对偏差 4.7e-6、整段 5.0e-5，
  `skipped=0`，scale 稳定在 16384；逐参数差 7.6e-2 只作参考（Adam 的
  sign 离散性，与 B2 同口径）。
- 回归：主机 `pytest tests --ignore=tests/test_frontend.py` 2379 项通过；
  A800 `tests.test_cuda_backend + test_cuda_train + test_autocast` 420 项通过
  （1 项按条件跳过），跑前跑后 GPU 均为 10 MiB / 0%。

### 5.296　Trainer 默认设备态：训练一步从 0.33× 回到 1.33×

**动机**：框架对比里训练的逐节点派发只有 torch 的 0.62×、真实模型探针更低，
这是"别人凭什么选我们"最前面的那道门槛。这一轮只做一件事：把"训练一步"拆开
归因，定位损失发生在哪一段，然后修掉。

**归因（A800-40GB 独占，跑前跑后 GPU 10 MiB / 0%）**：

1. 两层 MLP 基准（8.4 MB 参数，`benchmarks/framework_compare.py --training`，
   `--rounds 9 --burst 40 --warmup 20`）：逐节点派发 + 图外 SGD 0.896 ms
   （torch 0.552 ms，0.62×）；去掉参数更新 0.751 ms（0.61×），说明差距不在
   优化器；图外 SGD 单独 0.145 ms（torch 0.095 ms）。同一张图开图重放：
   带更新 0.486 ms（1.09×）、不带更新 0.387 ms（1.14×）。结论：主机态下
   每步把 8.4 MB 参数重新 H2D，就把上限压在 0.62× 附近。
2. HelixLM 0.28M 参数 / 254 节点：主机态 40.348 ms（0.325×），设备态
   9.224 ms（1.421×），torch eager 13.103 ms；loss 8.319477 三方逐位相同。
3. HelixLM 4.215M 参数 / 502 节点（`graph_mode=auto`）：主机态 57.025 ms
   （0.526×），设备态 33.738 ms（0.889×），torch 29.998 ms；loss 差 1.14e-07。

即"训练慢"不是反向展开或优化器的问题，而是默认配置走了主机态：参数每步
往返，图重放也救不回来（重放省的是派发，不是 H2D）。

**改动**：`Trainer(device_state=None)` 的默认值从 `False` 改成 `None`，含义是
按后端自动选：后端有 `capture`、`mixed_precision != "bf16"`、`zero3 is None`、
图里没有 BatchNorm 时开设备态，否则留主机态。三类回退都有硬理由——设备后端
只支持 f32/f16 缓冲，ZeRO-3 与常驻参数互斥，设备态 BN 的 running stats 未接。
显式 `True` / `False` 语义不变；原先四个直接读 `store` 断言数值的 CUDA 用例
改成显式 `False` 并在注释里写清原因，避免把"默认值"与"这一步要读主机值"
混在一起。

**复测**：0.28M 从 0.325× 到 1.330×（探针 default 档 9.852 ms，与 device 档
9.224 ms 只差一次显式传参）；4.215M 从 0.526× 到 0.894×（33.543 ms）。默认
配置不再要求用户知道"该开 device_state"，开箱即是最快的那一档。

**剩余差距**：大模型 0.89× 里那 11% 不在训练循环——502 个节点每个一次 kernel
发射，差距应落到算子与 kernel 效率上。下一项应做核时间线的单算子对比，而不是
继续抠主机侧派发开销。

**回归**：主机 `pytest tests --ignore=tests/test_frontend.py` 2379 项通过（该机
torch 2.4.1 不在兼容矩阵，`tests/test_frontend.py` 的 `UnsupportedTorchVersion`
与本改动无关）；A800 `tests.test_cuda_train + tests.test_cuda_backend` 386 项
通过（1 跳过），另 `test_parallel / test_lm_device_state / test_snn /
test_ssm_streaming / test_autocast / test_conformance / test_backend_contracts /
test_conv_impl_pick / test_cusparse / test_kernel_registry / test_events`
320 项通过（5 跳过）。新增
`test_device_state_defaults_on_for_device_backend`，覆盖默认开、参考后端留主机
态、`sync_to_host` 前后主机值三点。

### 5.297　窄末维归约：一个 block 打包多行，大档单步 28.6 → 23.5 ms

**动机**：5.296 把训练一步的损失压到设备态之后，大档剩下的差距只能在 kernel
里找。这一轮先做核时间线的单算子对比，再只修被数据点名的那个算子。

**归因（A800-40GB 独占，跑前跑后 GPU 10 MiB / 0%）**：

1. 图节点普查（本地，纯 CPU 构图）：大档前向 630 节点、反向 2464 节点。布局
   类算子占 57%——`reshape` 649 次、`slice` 508 次、`transpose` 397 次、`pad`
   196 次；`reshape` 是零拷贝视图不发射 kernel，`transpose` 每次都
   `ascontiguousarray` 真实物化。matmul 前向 25 次、反向 75 次。
2. 归约普查（本地）：前向 12 个末维归约里 8 个 `cols<=32`；反向 196 个归约
   **全部** `cols<=32`，主要是 `(8192,8)`、`(1024,8)`、`(131072,16)`、
   `(65536,8)`、`(4096,16)`、`(73728,8)`、`(1048576,8)`、`(65536,32)`。
3. AF 设备态 kernel 表（10 步合计）：`reduce_sum_last` 58.4 ms / 1990 次
   （29.3 μs/次，5.84 ms/步）排第一；`ampere_sgemm_128x128_nn` 57.3 ms /
   1000 次（57.3 μs/次，5.73 ms/步）；`ew_bcast` 31.7 ms / 2220 次、
   `cupy_copy` 28.6 ms / 7820 次、`concat_copy` 15.7 ms / 4840 次、
   `ew_bin` 11.6 ms / 3980 次。
4. torch 同类对照（10 步合计）：GEMM 用小 tile 拼（`32x32_sliced1x4` 约
   19 μs/次、`64x32_sliced1x4` 约 19 μs/次），GEMM 类合计约 2.2 ms/步；
   `reduce_kernel` 24 次/步约 0.33 ms/步；elementwise 三类合计约 2.6 ms/步。
5. eager 前端（非设备态）每步 49 个输入（含全部参数）重新 H2D：CUPTI 里
   `Memcpy HtoD (Pinned→Device)` 490 次 / 9.2 ms、`cudaPointerGetAttributes`
   490 次、`cudaEventQuery` 1166 次。这解释了 eager wall≈21 ms 而纯 kernel
   只有约 6.9 ms——设备态已经把这段消掉，本轮的差距与它无关。

结论：归约的次数与语义冗余是最大单项。`_reduce_last_dim` 在 `cols<=32` 时仍
按"一个 block 一行"发射，`plan_reduce(cols)` 返回 1..32 个线程，于是百万行
就是百万次 block 调度，而每次只做 8~32 个 float 的加法。

**改动**：

- 新增 L2 内核 `reduce_sum_rows`（`lower/kernels/reduce.py`）：`blockDim.x`
  仍是 `plan_reduce(cols)` 个线程沿 cols 求和，`blockDim.y` 打包多行
  （`ny = min(256 // nx, 32)`），grid 按 `ny` 行一组发射。越界行把行号折叠到
  最后一行参与 `__syncthreads`，但只在 `row < rows` 时写回——不能为了"省一次
  `if`"让部分线程提前退出，那会让块内屏障死锁。
- prelude 增加 `block_reduce_sum_x`（`backends/cuda/codegen.py`）：按
  `threadIdx.y` 切片 smem，每行仍是 `block_reduce_sum` 的折半树，因此结果与
  旧内核**逐位一致**，不是"近似相同"。示例方言 `examples/hardware/novac`
  照抄同一份 prelude（该测试要求两份逐字一致）。
- `device_reduce.py` 在 `reduce_sum` 且 `cols<=32` 时分流到新内核；`cols>32`
  与其它归约仍走原路径。
- 新增 `test_reduce_sum_narrow_last_dim`：`(1000, cols)`，cols 取
  1/2/8/16/32/33，同时覆盖新路径与回落边界。

**判据（A800，同一进程内 A/B）**：

- 禁用分流（临时把阈值改为 `cols <= 0`）：设备态单步 **28.649 ms**，
  `reduce_sum_last` 58.4 ms/10 步；torch 28.826 ms，约 1.01×。
- 启用分流：设备态单步 **23.475 / 23.473 ms**（两次重复），torch 28.921 /
  30.251 ms，**1.22~1.28×**。kernel 侧 `reduce_sum_rows` 9.63 ms/10 步
  （137 次/步，7.03 μs）+ `reduce_sum_last` 3.35 ms/10 步（62 次/步，
  5.40 μs），合计 1.30 ms/步，比 5.84 ms/步省 4.54 ms，与 wall 的 5.18 ms
  差在同一量级。
- 正确性：A800 `tests.test_cuda_backend.TestCudaReductions` 18 项通过；
  主机 `pytest tests --ignore=tests/test_frontend.py` **2381 通过、449 跳过、
  277 subtests 通过**（该机 torch 2.4.1 不在兼容矩阵，frontend 与本次无关）。

**剩余差距**：GEMM 仍是 AF 的 5.73 ms/步对 torch 约 2.2 ms/步——AF 固定用
128x128 tile，遇到 `(131072,8)x(8,768)` 这类窄 K 形状时每个 tile 只做很少的
乘加；下一项应做 GEMM 的形状驱动选 tile。之后是布局物化（`transpose` 397 次
`ascontiguousarray`）与 elementwise 链（`ew_bcast` 222 次/步、`cupy_copy`
782 次/步、`concat_copy` 484 次/步），再往后才是多进程 DP/ZeRO-2。

### 5.298　GEMM 形状驱动选 tile：负结果，tile 不是那 2.6× 的来源

**动机**：按 5.297 的结论做「形状驱动选 tile」。假设是 AF 固定用
`ampere_sgemm_128x128_nn`，窄 K 形状上每 tile 只做很少乘加，而 cublasLt 的
启发式能按 `(m, n, k)` 选到 sliced 小 tile，从而吃掉 5.73 → 2.2 ms/步 的差距。

**归因（A800-40GB，跑前跑后各查一次 `nvidia-smi`，全程独占到 10 MiB / 0%）**：

1. **先证伪旧前提**：`torch.mm((131072,8), (8,768))` 在 cublasLt 日志里的框架
   与 AF 完全相同——`Adesc rows=768 cols=8 ld=768`、`Bdesc rows=8 cols=131072
   ld=8`，全部 `CUBLAS_OP_N`；启发式只返回 1 个结果 `MATMUL_TILE_128x64`
   （`maxWorkspaceSizeinBytes=8519680`、`minBytesAlignment=16`）。5.297 里
   「torch 用小 tile 拼、约 19 μs/次」在本机 torch 2.5.1 上不成立。
2. **实现 Lt 路径再测**（ctypes 绑定 `libcublasLt`，只开 f32；
   `CUBLASLT_MATMUL_PREF_SEARCH_MODE` 不设、workspace 上限按预置量给）：
   窄 K 四形状 `(131072,768,8)`、`(65536,4096,16)`、`(32,64,16)`、`(7,5,3)`
   与 `sgemm` **maxdiff=0**；f16/bf16 原样回退旧路径；CUDA Graph 捕获期首次
   调用（含描述符创建与启发式）即可成功，replay maxdiff=0。
3. **逐形状 A/B**（同一进程内交错取样，`batch=20`、`rounds=7`、取最小 μs/次）：

   | 形状 (m,k)x(k,n) | sgemm | Lt | torch | Lt/sgemm |
   |---|---|---|---|---|
   | (131072,8)x(8,768) | 271.87 | 272.69 | 272.44 | 1.00 |
   | (131072,16)x(16,768) | 284.52 | 284.62 | 284.52 | 1.00 |
   | (131072,8)x(8,8) | 16.54 | 16.95 | 16.59 | 0.98 |
   | (131072,32)x(32,64) | 49.51 | 50.12 | 49.56 | 0.99 |
   | (65536,16)x(16,4096) | 704.92 | 704.20 | 705.59 | 1.00 |
   | (8192,1024)x(1024,4096) | 3623.27 | 3627.01 | 3623.99 | 1.00 |
   | (1024,1024)x(1024,1024) | 130.05 | 132.04 | 130.25 | 0.98 |

   三者同速（差距 ≤2%）。预置 32 MB workspace（足以容纳 split-K 所需）后复测，
   七行仍全部 0.98~1.00。
4. **带宽核算**：`(131072,768,8)` 的流量是输出 `131072×768×4 ≈ 402.7 MB` 加
   输入约 4.2 MB，共约 407 MB；271 μs 对应 **1.50 TB/s**，已在 A800 的 HBM
   实测带宽（峰值 2.0 TB/s）饱和区。这类"大 M、窄 N/K"的形状是纯写带宽受限，
   任何 tile 都无法加速。

**结论（负结果）**：`cublasLt` 在这些形状上与 `cublasSgemm_v2` 同速，且两者
都与 torch 同速。5.297 记录的「GEMM 57.3 μs/次 vs torch 约 19~22 μs/次」不能
由 tile 选择解释。改动已**全部回退**（`cublas.py` 无新增、`device_linear.py`
恢复三条 dtype 分支），不保留无用路径。

**真实形状清单（A800，钩子 + `torch.profiler`，同一模型各跑一步）**：

上一版 A/B 的 7 个形状确实不是模型真实清单，本轮补抓。AF 侧挂 `Cublas` 钩子，
torch 侧用 `profiler(record_shapes=True)`，模型取 `embed=256 / layers=8 / seq=8 /
batch=2 / vocab=24`（4.281M 参数，AF 前向 998 节点）：

- AF 一步 **196 次** GEMM，**全部走 `sgemm_batched`**，且前导维是 `(batch=2,
  seq=8)` 而不是合并后的 `16`。次数前四：`(2,8,256)x(256,256)`×72、
  `(2,8,256)x(256,512)`×32、`(2,256,8)x(2,8,256)`×24、`(2,8,256)x(256,288)`×16。
- torch 的 `aten::linear([2,8,256], [512,256])` 会先展平成 **`aten::mm([16,256],
  [256,512])`**，落到 `ampere_sgemm_32x32_sliced1x4_tn`（M=16 的小 tile）、
  `gemmSN_TN_kernel`（small-N）、`magma_sgemmEx_kernel` 与少量 `128x128_nt`。
  即 5.297「torch 用小 tile」的说法**在真实形状上成立**，只是不发生在
  大 M 窄 K 形状上。

**真实形状 A/B**（同进程交错，`N=200`、`ROUNDS=7`，μs/次）：

| 情形 | AF `sgemm`（合并 M=16） | AF `sgemm_batched`（b=2,m=8） | `torch.mm` | 合并/torch | batched/torch |
|---|---|---|---|---|---|
| lin 256→512 | 14.13 | 9.92 | 12.50 | 1.13 | **0.79** |
| lin 256→256 | 13.84 | 9.92 | 12.34 | 1.12 | **0.80** |
| lin 256→288 | 13.88 | 10.02 | 12.34 | 1.12 | **0.81** |
| lm_head 256→24 | 21.33 | 10.01 | 21.33 | 1.00 | **0.47** |
| wgrad 256x16@16x256 | 9.70 | 10.07 | 8.29 | 1.17 | 1.22 |
| x 16x512@512x256 | 13.61 | 9.86 | 11.22 | 1.21 | **0.88** |
| bmm 2x16x16@2x16x16 | — | 10.09 | 8.44 | — | 1.20 |
| bmm 2x16x16@2x16x256 | — | 10.07 | 8.26 | — | 1.22 |
| bmm 64x1x64@64x64x8 | — | 10.15 | 8.33 | — | 1.22 |

**结论比上一版更强**：AF 现在走的 batched 路径在五个主形状上**比 torch 快
12%~53%**，只有反向里的两个小形状（K=16 的权重梯度、batch 内 16x16）慢约
1.2×。按 196 次 × ~10 μs ≈ **1.96 ms/步**，与 torch 一步约 235 个 GEMM kernel
的 2 ms 量级持平。所以 GEMM 既不需要 Lt，也不需要形状驱动选 tile——**5.297 的
「GEMM 5.73 ms/步」与本轮实测的 1.96 ms 也对不上，那 57.3 μs/次的均值同样不可
信**（可能是把单次元素级/布局开销串进了统计，需在布局那一步重测口径）。

**下一步（按证据顺序）**：

1. 布局物化：`transpose` 397 次 `ascontiguousarray`。
2. elementwise 链：`ew_bcast` 222 次/步、`cupy_copy` 782 次/步、`concat_copy`
   484 次/步。
3. GEMM 只剩两处 1.2×：K=16 的权重梯度形状与 batch 内 16x16 小 bmm，合计约
   0.2~0.3 ms/步，优先级低于前两项。

**回归**：本轮无代码改动（回退到 HEAD），无需新增用例。远端 `/root/AetherFlow`
的 `cublas.py` / `device_linear.py` 已同步回主线，`git diff` 为空。

### 5.299　归因布局物化：1174 次拷贝 ≈ 2.4 ms/步（9%），成本在 kernel 数不在带宽

**动机**：5.298 把下一步指向布局物化（`transpose` 的 `ascontiguousarray`）。要回答
三件事：一步里有多少次、值多少毫秒、能不能消除。

**口径**（这一节的结论依赖这三条，先写清楚）：

1. `transpose` 本身只是视图，**零拷贝**；真正物化的是紧随其后的
   `ascontiguousarray`。恒等置换、或置换后仍连续（含 size-1 维）时也不拷贝。
   所以只统计 `ascontiguousarray` 入参**非连续**的那些调用。
2. 设备态一步是 CUDA Graph 捕获：捕获前先**预热跑一轮**再**捕获跑一轮**，
   Python 侧计数会翻倍，必须除以 2（脚本里断言记录数为偶数）。
3. 单次成本不能用 `event` 包住 Python 调用去测——那样把分配和 Python 开销
   一起算进去，会高估 2~3×。改把同形状的 N 次拷贝捕成一张小图，按 replay
   均摊，与整图里一次 kernel 同口径。

**基线**（A800-40GB，跑前后各查一次 `nvidia-smi`，独占、10 MiB / 0%）：
同一模型（`embed=256 / layers=8 / seq=8 / batch=2 / vocab=24`，4.281M 参数，
AF 前向 998 节点、反向 2464 节点），AF **26.07 ms/步**，torch 2.5.1 同模型同数据
**41.68 ms/步** → **AF/torch = 0.63（快 1.60×）**。下面所有占比都以 26.07 为分母。

**一步的布局物化**：

- `transpose` 789 次，其中 **264 次置换后仍连续、零拷贝**；`ascontiguousarray`
  共 1852 次，**真实拷贝 1174 次 / 90.36 MB**，其余 678 次是 no-op。
- 按物化来源拆：`_op_transpose` **525 次 / 73.3 MB**；`_op_slice` **608 次 /
  8.7 MB**；`_op_broadcast_to` **41 次 / 8.4 MB**。

**逐节点归属**（次数/步，取前 20 中的代表；`->` 后是消费者）：

| 次数 | MB | 形状 / 来源 |
|---|---|---|
| 64 | 0.52 | `(4,8,32,2)` transpose`[1,2,3,0]` → `reduce_sum` |
| 64 | 0.26 | `(1,8,32,4)` transpose`[0,2,3,1]` → `reduce_sum` |
| 49 | 0.80 | `(2,256,8)` transpose`(0,2,1)` → `matmul` |
| 48 | 0.39 | `(2,4,8,32)` slice dim=3 `[32:64]` → `mul` |
| 42 | 0.69 | `(256,2,8)` transpose`[2,0,1]` → `reshape` |
| 32×8 | 4.19 | `(2,4,8,64)` slice dim=2 `[k:k+8]`（k=0..7）→ `reshape` |
| 24 | 6.29 | `(256,256)` transpose`(1,0)` → `matmul` |
| 24 | 12.58 | `(256,256,2)` transpose`[1,2,0]` → `reshape` |
| 16 | 16.78 | `(256,512,2)` transpose`[1,2,0]` → `reshape` |
| 16 | 8.39 | `(512,256)` transpose`(1,0)` → `matmul` |
| 8 | 4.72 | `(256,288,2)` transpose`[1,2,0]` → `reshape` |
| 16 | 4.19 | `(8,2,256,16)` transpose`(1,0,2,3)` → `scan_linear` |
| 16 | 4.19 | `(2,8,16,256)` transpose`[0,1,3,2]` → `reduce_sum` |
| 16 | 2.10 | `(2,4,8,8,64)` broadcast → `mul` |

**单次成本标定**（捕获图内 replay 均摊，μs/次）：`(256,512,2)` 6.50、
`(256,256,2)` 4.68、`(512,256)` 4.83、`(256,256)` 3.79、`(2,8,256,16)` 2.62、
`(2,4,8,32)` 2.02、slice`(2,4,8,64)[...,32:64]` 2.02。拟合出
**μs ≈ 1.7 + 字节 / 0.2 TB/s**：转置是 gather/scatter，有效带宽只有约
0.2 TB/s（远低于 HBM 的 1.5~2.0 TB/s），而 1.7 μs 是与字节数无关的固定项。

**合计**（按上表同口径外推）：

| 来源 | 次数/步 | MB/步 | 估计 ms/步 |
|---|---|---|---|
| transpose | 525 | 73.3 | 1.26 |
| slice | 608 | 8.7 | 1.08 |
| broadcast | 41 | 8.4 | 0.08 |
| 合计 | 1174 | 90.4 | **≈2.42（9.3%）** |

**结论**：

1. **这是 kernel 数问题，不是带宽问题**。90.4 MB 的纯流量按峰值也只值 0.12 ms，
   而 1174 × 1.7 μs 的固定项就有 2.0 ms。想让这一项变小，只能减少 kernel 个数
   （把布局并进消费者/生产者），换更快的拷贝内核没有意义。
2. **不值得为此改 strided 访问**。上限就是 2.42 ms/步（9.3%），而代价是全部生成
   内核的寻址重写；且 AF 在这个档位上已经比 torch 快 1.60×，torch 靠 strided
   kernel 省下的这部分正是它 41.68 ms 里的一小项，差距主因不在这里。
3. **唯一现成的「可消除」缺口**：反向图有 **49 个** 二维 `transpose(1,0)` 单消费者
   就是 `matmul`（`(256,256)`×24、`(512,256)`×16、`(288,256)`×8、`(256,24)`×1），
   **全部没有命中** `_merge_transpose_matmul` 的融合（前向候选 0 个）。按标定约
   0.17 ms/步（0.65%），可以顺手做，但不是本轮重点。
4. **slice 608 次是次数第一**，其中 256 次是 `(2,4,8,64)` dim=2 的 8 个滑窗
   `[k:k+8]`（窗长 8）→ `reshape`，80 次是 dim=3 `[32:64]` → `mul`。语义上就是
   「窗口移位 + 展平」，要省掉同样得让下游按 strided 读。

**下一步**：按同一口径（捕获期计数 + 捕获图单次成本）归因 elementwise 链
（5.297 记的 `ew_bcast` 222 次/步、`concat_copy` 484 次/步）。5.297 的
`cupy_copy` 782 次/步与本轮 1174 次同量级（那版模型前向 630 节点、本轮 998），
互相印证：`cupy_copy` 就是这些物化拷贝，而不是别的隐藏开销。

**回归**：本轮无代码改动，无需新增用例；本地与远端仓库均干净。

### 5.300　elementwise 链归因：1.84 ms/步（17.1%），大形状广播慢在 64 位 div/mod

**动机**：5.299 的下一步指定了 elementwise 链。要回答：一步里 `ew_bcast` 与
`concat`/`pad` 各多少次、值多少毫秒、慢在带宽还是寻址。

**口径**（与 5.299 一致，另有两条本轮踩出来的坑）：

1. 计数在**静态图**上做，不在运行期挂钩。设备态一步是 CUDA Graph 捕获，
   `_kernel` 有缓存，第二次执行不重建，monkeypatch 计数会漏；直接扫
   `trainer.graph` 与 `trainer._bwd` 的节点才是"一步一次"的准确口径。
2. `ew_bcast` 的判定与后端 `_binary` 的分派一致：op ∈ `BINARY_OP`、两侧形状
   不同、且不是 size-1 快速路径。
3. **单次成本必须用 `plan.run_device()`，不能用 `plan.run()`**。后者对每个
   输出做 D2H `to_host` 并逐个同步，8192 元素的逐元素被报成 30 μs，高估约
   20×（本轮第一版数据因此全部作废）。
4. 即使 `run_device`，replay 仍有固定的 Python/下发开销，所以用**两点法**：
   同一形状捕 R 次与 R0=4 次两张小图，`(t_hi - t_lo) / (R - R0)` 才是单次
   成本。R 按输出元素数取 2048 / 256 / 32 三档。

**基线**（A800-40GB，跑前 `nvidia-smi` 确认独占 10 MiB / 0%）：
`LMSpec(vocab=24, d_model=256, layers=8, d_state=16, d_conv=4, expand=2,
steps=8, batch=2)` 的 Mamba 语言模型 + `masked_lm_loss`，**3,387,648 参数**，
前向 608 节点、反向 2722 节点。AF **9.92 / 10.78 ms/步**（两次运行；逐形状
标定值两次一致，差距来自 replay 抖动），拆开：fwd replay **1.51 ms**、
bwd replay **5.37 / 5.62 ms**，其余 **3.04 / 3.65 ms**（优化器 + D2H + Python）。

注意：这版模型与 5.296~5.299 的 Transformer（4.281M、fwd 998 / bwd 2464、
26.07 ms）不同，**绝对 ms 不可与那几节直接比**，占比可比。

**计数**（每步）：

| 类别 | 前向 | 反向 | 合计 |
|---|---|---|---|
| `ew_bcast` | 98 | 281 | **379** |
| `concat`/`pad` | 16 段 | 112 段 | **128** |

`concat`/`pad` 共 13 种键：`concat((2,3,512),(2,8,512)) axis=1` 32 段；
`pad (512,1)→(512,4)` 32 次；`pad (2,8,512)→(2,11,512)` 32 次（四种
before/after 各 8）；`pad (2,8,512)→(2,8,1024)` 16 次；
`pad (8,2,512,16)→(9,...)` 与 `pad (1,2,512,16)→(8,...)` 各 8 次。

**逐形状成本**（`run_device` + 两点法，μs/次）：

| 次数 | a × b | 输出元素 | μs | 小计 ms |
|---|---|---|---|---|
| 96 | `(2,8,512)` × `(1,1,512)` | 8192 | 2.18 | 0.209 |
| 48 | `(2,8,512)` × `(512,)` | 8192 | 2.15 | 0.103 |
| 40 | `(8,2,512,16)` × `(8,2,512,1)` | 131072 | 7.55 | 0.302 |
| 32 | `(8,2,512,16)` × `(8,2,1,16)` | 131072 | 7.96 | 0.255 |
| 27 | `(2,8,256)` × `(2,8,1)` | 4096 | 2.24 | 0.060 |
| 24 | `(2,8,512)` × `(2,8,1)` | 8192 | 2.25 | 0.054 |
| 16 | `(1,8,512)` × `(2,8,512)` | 8192 | 2.16 | 0.035 |
| 16 | `(8,2,512,1)` × `(512,16)` | 131072 | 7.96 | 0.127 |
| 16 | `(8,2,512,1)` × `(8,2,1,16)` | 131072 | 7.81 | 0.125 |
| 16 | `(512,)` × `(8,2,512)` | 8192 | 2.14 | 0.034 |
| 9 | `(2,8,256)` × `(256,)` | 4096 | 2.15 | 0.019 |
| 9 | `(2,8,1)` × `(2,8,256)` | 4096 | 2.24 | 0.020 |
| 8 | `(2,8,1)` × `(2,8,512)` | 8192 | 2.23 | 0.018 |
| 8 | `(8,2,512)` × `(512,)` | 8192 | 2.14 | 0.017 |
| 8 | `(8,2,512,16)` × `(512,16)` | 131072 | 7.60 | 0.061 |
| 4 | `(2,8,24)` × `(2,8,1)` | 384 | 2.25 | 0.009 |
| 2 | `(2,8,1)` × `(2,8,24)` | 384 | 1.92 | 0.004 |

**成本**（第二次运行，占比以 10.78 ms 为分母）：

| 项 | 次数/步 | ms/步 | 占 step |
|---|---|---|---|
| `ew_bcast` | 379 | 1.45 | 13.5% |
| `concat`/`pad` | 128 段 | 0.39 | 3.6% |
| 合计 | 507 | **1.84** | **17.1%** |

按图拆：fwd 里 ew 98 次 **0.393 ms（占 fwd replay 26.1%）**、cc 16 段
0.031（2.1%）；bwd 里 ew 281 次 **1.060 ms（占 bwd replay 18.9%）**、
cc 112 段 0.359（6.4%）。

**慢在哪**（同形对照，本轮新增）：

| 形状 | 输出元素 | 路径 | μs | 有效带宽 |
|---|---|---|---|---|
| `(8,2,512,16)` | 131072 | 同形 add | 2.28 | ~691 GB/s |
| `(8,2,512,16)`×`(8,2,512,1)` | 131072 | 广播 add | 7.55 | ~208 GB/s |
| `(2,8,512)` | 8192 | 同形 add | 1.72 | launch-bound |
| `(2,8,512)`×`(512,)` | 8192 | 广播 add | 2.15 | launch-bound |

131072 元素的广播比同形慢 **3.3~3.5×**。根因在模板：`ew_broadcast` 每个元素
都要用最多 4 维的 `%` / `/` 把线性下标还原成坐标，而 `shape` 是运行期参数，
编译器不知道它们都是 2 的幂，只能发 64 位整数除法；同形路径是 `a[i] + b[i]`
的连续访存，没有这一步。

这个低效点很集中：**112 次 131072 元素广播合计 0.87 ms（占 ew 的 60%、
占 step 的 8.1%）**，压到同形水平只需 0.26 ms。8192 及以下的小形状（267 次、
0.58 ms）已经贴着 launch 下限（2.15 vs 同形 1.72 μs），没有多少可挖。

**结论**：

1. elementwise 链合计 **1.84 ms/步（17.1%）**，是目前归因过的最大单项，但
   其中 60% 集中在 5 种大形状广播上，目标很窄。
2. 慢因不是带宽而是**寻址**：131072 元素广播 208 GB/s vs 同形 691 GB/s。
3. `concat`/`pad` 只占 3.6%；`pad` 由 `cp.zeros` + 逐段 `concat_copy` 组成
   （每次至少 2 个 kernel，3.07~4.60 μs），但总量太小，不值得为它单改语义。

**下一步**（按收益/风险排序）：

1. **2 的幂维用移位/掩码替代 div/mod**（曾预计大形状降 2~3×、省 0.4~0.6 ms）。
   已实现并**上机实测否决、方案撤回**（2026-11-27，Tesla PG503-216 / V100 32GB、
   CUDA 12.4、cupy 14.2）：131072 元素广播上 pow2+`sh` 形态比 div/mod **慢
   14~15%**（20.60 vs 17.70 μs、18.40 vs 15.72 μs；where 27.12 vs 23.62 μs；
   两次独立测量：两点法与交替 400 次 × 5 轮取最小）。手写三内核对照
   （同形状、同 block/grid）定位到真凶：多传的 `int sh[4]` 局部数组被循环变量
   `d` 动态下标寻址，落到 local memory —— `div` 7.85 μs / `pow2+sh` 10.87 μs
   （1.38×）/ `pow2+clz`（`rem >> (31-__clz(shape))`，去掉 sh 数组）7.88 μs
   （1.00×）。即去掉数组只能**追平**，`&`/`>>` 的收益被 `__clz` 抵消，没有
   净收益。因此 `ew_broadcast` / `ew_where_bcast` 的 pow2 分支、`sh` 参数与
   后端缓存键里的 pow2 维度全部删除，回到单一 div/mod 形态（本地 2943 用例
   全过；V100 复跑 conformance 121 executed / 8 refused / 0 failed，81/81 覆盖）。
   继续提速只能靠第 2 条（把广播乘融进邻近内核的 epilogue），而不是改寻址。
2. **广播乘法融进 `scan_linear` / `reduce_sum` 的 epilogue**。`A*delta`、
   `B*x` 这类输入本来就是 scan 的入参，整段可消掉 0.87 ms，但要改 scan
   内核，风险高，排在 1 之后。
3. `pad` 精确清零（只写被 pad 的区域）：上限 0.1~0.2 ms，收益最小。

**回归**：本轮无代码改动；标定脚本与静态普查脚本用完即删。本地与远端仓库
均干净。

### 5.301　对标清单核对与补齐路线（阶段 A / B）

外部给了一份"对比 PyTorch / TVM / vLLM / TensorRT-LLM"的缺口清单。逐条对照
代码后，先作废 4 条与现状不符的，再把剩下的按"挡不挡使用"分层。**清单不能
直接当待办**：按它排期会把已经做完的东西再做一遍。

**核对结果（作废项）**

1. **训练配套不是空白**：`train/grads.py` 有 `clip_global_norm` 与
   `GradientAccumulator`，`train/schedule.py` 有 `CosineDecay` / `StepDecay` /
   `ExponentialDecay` / `Warmup`，`train/checkpoint.py` 有 `save_async` /
   `load_checkpoint`，`train/shard_io.py` 有分片 ckpt，`train/amp.py` 有
   `GradScaler` 且 `passes/autocast.py` 是独立 pass。这些是"上层要自己拼装"，
   不是"没有"。
2. **MoE / 投机解码 / 分页 KV 已有**：`models/moe.py` 用稠密参考语义把路由、
   top-k 门控与负载均衡闭环表达完整（专家并行显式拒绝），
   `models/speculative.py` 与 `models/paged_cache.py` 是组件级交付。准确表述是
   "未接进 `LMRunner` 主干"，不是"没有实现"。
3. **49 条 VJP 不是抽样白名单**：IR 一共 71 个算子，可微的那部分正好 49 个，
   等于全覆盖。PyTorch 需要"海量算子兜底"是因为它有 2000+ 算子；算子桶小
   就不存在兜底问题。真正的差距是**算子面窄**，不是反向规则缺兜底。
4. **包发布已自动化**：`v*` tag 触发 PyPI trusted publishing，版本号单点来源
   在 `aetherflow/__init__.py`。"只能源码 `pip install -e .`"已过时。

**真缺口（按挡不挡使用排序）**

| 序 | 缺口 | 挡的是什么 |
| --- | --- | --- |
| 1 | bool 已是一等类型（5.330）且导入侧已接（5.340）；仍缺设备端 bool 归约内核 | 带 mask 的模型能进、参考端能跑逻辑组合；`reduce_all` / `reduce_any` 仍只有参考后端承接 |
| 2 | 设备侧归约已是数据并行默认（5.331），主机态已有桶间并发归约（5.302）；仍缺"设备态与反向重叠"与 P2P | 多卡扩展性（设备态反向是整图 capture / replay，准入判据见 5.338） |
| 3 | 量化只到 weight-only int8 | 推理性能上限（W8A8 的主机侧语义见 5.308、设备内核见 5.309；QAT 的假量化算子见 5.311，可学步长（LSQ）见 5.312；非对称零点的语义与构图入口见 5.315，量化 / 反量化内核见 5.322、融合线性层见 5.323，逐组 weight-only 反量化见 5.317、设备内核见 5.321；量化权重的 safetensors 读取约定见 5.317；fp8 的参考语义见 5.318、设备端 cast 的量化内核见 5.325，但仍无 fp8 GEMM 与存储容器） |
| 4 | 无调度搜索 / GEMM epilogue（loop IR 见 5.332、原生 NHWC 见 5.334；5.333 只把配置库接进执行路径） | 峰值性能上限 |
| 5 | prefix caching / KV swap / 投机未接主路径 | 长上下文服务的显存效率（投机已接 `LMRunner`，见 5.303；分页换出 / 换入与共享前缀见 5.304 / 5.307；token→token 的生成闭环见 5.310，`LMRunner` 主干是 Mamba、没有 KV，故不在其中；前缀缓存的自动匹配见 5.335，跨进程 / 落盘复用见 5.337） |
| 6 | 国产卡后端 | 特定硬件市场（天数 / 沐曦两个硬件包已在各自仓库真机跑通，主仓侧契约验收通过；主仓未跑 conformance，见 5.336） |

第 4 项与"IR 语义简单、数值可验证"的设计约束正面冲突（等于换一套后端），
排在最后且很可能永远不做；第 3、5 项取决于项目定位（卖推理才做）；第 6 项
需要厂商联合优化，单机项目做了也是半成品——5.336 把这句收窄成"只做离线
可验证的方言与装配骨架"，真机部分仍等硬件与 SDK。**只有第 1、2 项是无论
定位如何都该补的。**

**阶段 A：bool 与掩码算子面（本地可完整验证）**

类型系统其实已经有 `DType.BOOL`（`ir/types.py`，numpy 侧映射为 `bool`），缺的是
"谁能产出它"和"后端怎么存"：

- **A1 比较算子**：`eq` / `ne` / `lt` / `le` / `gt` / `ge`。`infer` 复用现有广播
  规则、输出 `bool`；`reference` 直接用 numpy 比较；`codegen` 增加 bool 存储键
  （设备容器 `unsigned char`，存取 helper `af_ld_q` / `af_st_q` 把 0/1 与
  float 互转），设备内核复用 `ew_bcast` 的坐标还原结构（div/mod 单一形态，
  5.300 的 pow2 变体已实测否决并删除）。
  反向规则不注册——比较在数学上没有梯度，`vjp=None` 会让 `backward` 明确报错。
- **A2 `where` 与 `masked_fill`**：`where(cond, a, b)` 三输入广播，设备内核在
  `ew_bcast` 上多一路 stride；`masked_fill` 不新增算子，前端降成
  `where(eq(mask, 0), x, full(value))`。`where` 的 vjp 是
  `g_a = where(cond, g, 0)`、`g_b = where(cond, 0, g)`，不需要 bool→float 的 cast。
- **A3 一元补齐**：`sign` / `round` / `floor` / `ceil`。`round` 必须是
  half-to-even（`nearbyintf` / `rintf`），与 torch 的 `round` 一致；`floor` / `ceil`
  用 `floorf` / `ceilf`。这四个都进 `UNARY_EXPR` 与前端白名单。
- **A4 `topk`**：返回 values + indices，最后一维块内选择。它涉及多输出与 i32
  索引输出，放在 A1~A3 稳定之后单独做；并列时按**下标升序**定序，写进语义。

**验收**：IR verifier 拒绝 bool 参与算术；参考后端数值正确；参考↔CUDA 一致性
套件覆盖新增算子；一个含比较 + `where` 的 `torch.export` 模型端到端跑通；
`pytest` 全绿。

**阶段 B：DDP 多进程与反向重叠（受机器限制）**

- B1 `DistributedDataParallel` 在提供设备集合通信时走 NCCL，不再经主机中转；
- B2 反向按参数分桶做 allreduce，并与计算重叠（桶就绪即发，不等全图结束）；
- B3 rank / world 从环境变量读，配一个最小启动器。

**限制先写清楚**：现只有单卡 A800 与两卡数值用例，`world > 1` 无法真机验证。
因此 B 只做到"代码路径与两卡用例就位"，**不承诺规模数字**；要真数字得先有
多卡机器。

**明确不做**（写进边界，不再单独讨论）：auto-schedule 搜索、loop 级 IR、
TVM 式分层 IR、通用自动微分兜底、国产卡后端的真机 runtime、C++ runtime、
K8s / 监控 / 熔断、CV / 语音专用算子（NMS / ROIAlign）。理由统一：要么与
"语义简单、数值可验证"的核心约束冲突，要么需要团队与厂商资源，单人项目
做了也是半成品（国产卡只保留 5.336 的离线可验证部分）。

**风险**：bool 落地后必须钉死转换方向（只允许 bool→f32 / i32，反向拒绝），
否则 `cast` 会把任意整数当布尔；常量折叠要能处理比较结果，但要避免把
bool 常量写进属性时被当成整数；`topk` 的并列定序必须写进语义，否则参考端与
设备端的下票顺序可能不同。

**实施进展：A1–A3 已落地，A4 暂缓**

A1（比较算子）已完成。`DType.BOOL` 本来就在 `ir/types.py`（numpy 侧是
`bool`），缺的只是"谁产出"和"怎么存"：`eq` / `ne` / `lt` / `le` / `gt` / `ge`
注册进算子表，`infer` 复用既有广播推断、输出 `bool`，`reference` 直接调 numpy；
bool 参与 `+ - * / pow` 等算术在 `infer` 阶段就报 `TypeError`，所以"bool 只能
由比较产出"这条约束是类型系统强制的，不是约定。设备端 bool 是 1 字节存储
（`_STORAGE["bool"] = "unsigned char"`），由 `BOOL_DIALECT` 提供
`af_ld_q` / `af_st_q`（0/1 ↔ float），布尔常量写回走 `Store(raw=True)`，不套
store helper——否则 helper 会把 `__ldg` 的 float 语义带进来。比较无梯度，
不注册 vjp，`backward` 遇到会明确报错而不是静默给 0。

A2（`where` / `masked_fill`）已完成。`where` 没有新写索引代码，`ew_where_same`
与 `ew_where_bcast` 就是逐元素骨架多维护一路 stride；`_collapse_dims` 从
"两输入"泛化成 `*strides`（二元调用方解包 `(dim, step_a, step_b)` 不受影响），
三元调用传 `(cond, a, b)`。`ew_where_bcast` 同样有 5.300 的两套形态：2 的幂
用 `rem & (dim-1)` / `rem >> sh[d]`，否则 `rem % dim` / `rem / dim`。
`masked_fill` 不新增算子，前端三个变体（`.Scalar` / `.Tensor` / `_`）统一降成
`where(mask, value, x)`；标量 value 用既有的 `branch()` 折成与张量同 dtype，
避免把 f32 标量塞进 f16 图。vjp 用 `where` 自己表达：
`g_a = sum_to(where(cond, g, 0))`、`g_b = sum_to(where(cond, 0, g))`，bool 分支
返回 `None`——不需要 bool→float 的 `cast`。

A3（一元补齐）已完成。`sign` / `round` / `floor` / `ceil` 进 `UNARY_EXPR`、
手写对照表与前端白名单，NHWC 逐元素集合同步收录（布局 pass 不为这四者开
分支）。`round` 必须用 `nearbyintf`（默认舍入模式是 round-half-to-even，与
`torch.round` / `np.round` 一致），**不能用 `roundf`**（half-away-from-zero，
差在 x.5 上）；`sign` 用嵌套三目保证 0→0 且 NaN 透传，不能写成
`(x > 0) - (x < 0)`（NaN 会变成 0）。四者梯度都是 `None`，与 `step` 同口径。

前端承接：`aten.where.*` 三种签名都归一到 `condition / input / other`，先定位
张量分支确定 dtype 再校验 cond——`cond` 必须是 bool，两侧 dtype 不一致直接报
`UnsupportedOperatorError`，不做静默 fallback（与"语义精确、数值可校验"的
既定边界一致）。端到端已跑通三类模型：`torch.masked_fill(x, mask, 7.5)`、
`torch.where(x > y, x - y, y - x)`、bool 输入直通 `where`，参考后端输出与
torch 逐位相等。

**规模与回归**：算子表 71 → **84**（+6 比较、+`where`、+4 取整、+`topk_values`
/`topk_indices`），VJP 规则 49 → **56**（`where`、`topk_values`、`topk_indices`
各一条，四取整各一条 `None`；可微算子仍是 49 个）。
`python -m unittest discover -s tests` → **2943 tests, OK**（skipped 452，全部
是需要 GPU 的用例）。

**A4（`topk`）已落地：拆成两个单输出算子，设备内核仍缺**

- `topk_values` / `topk_indices` 的属性只有 `("k", "axis")`，语义是**前 k 大、
  按值降序、并列时小下标优先**：`np.argsort(..., kind="stable")` 的稳定序正好
  给出"并列保持原下标顺序"。NaN 先替换成 `+inf` 再排序，于是被当作最大值——
  这是 `torch.topk` 的口径；numpy 原生 argsort 会把 NaN 丢到末尾，两边结果
  会完全相反。三个算子（含既有的 `topk_mask`）共用一份 `_topk_axis_k_shape`
  校验，避免"mask 拒绝 k=0 而 values 接受"这种边界分叉。
- 前端 `aten.topk.default` 的两个槽位各映射一个节点。export 图里 `getitem`
  是独立节点，所以只取 values 的模型在 DCE 之后不会残留 `topk_indices`——
  这正是"拆成两个单输出算子"相对"一个双输出算子"的收益（IR 的 `Op.infer`
  本来就只能返回一个 `TensorType`）。
- `largest=False` 明确拒绝（提示先 `neg`、跑完再 `neg` 回来），`sorted=False`
  接受（返回有序结果是"顺序不保证"的合法子集），`k` 必须是静态整数（动态 k
  会让输出形状随输入变化，IR 的形状推断表达不了）。
- 梯度：`topk_values` 有完整反向——重算一遍 `topk_indices`，再用
  `gather_backward` 把 dy 散射回被选位置（与 `gather` 同一条约束：要求 x 的
  形状静态，因为 `x_shape` 要写进反向节点）；`topk_indices` 与 `topk_mask`
  返回 `None`。反向重算下标而不是前向保存，代价是排序跑两次，换来反向图不
  依赖前向中间量。
- **设备端仍然没有内核**（当时），与 `topk_mask` 同一状态：CUDA/HIP 上会抛
  `BackendCapabilityError`，不静默降级。补它需要"k 轮块内选择 + 跳过已选
  下标"的专用内核（O(k²n)，`Shared("int", "sel", k)`，cache key 含 k）。
  **已在 5.305 补齐**（三个算子共用一份内核）。
- conformance 新增 4 条（values / indices × 末轴 / 首轴），参考后端
  **84/84 个声明算子**全部至少成功执行一次；回归 **2943 tests OK**。

**验收核对**：IR 拒绝 bool 算术 ✅；参考后端数值 ✅（`conformance.py` 新增
8 条比较、2 条 `where`、4 条取整）；`torch.export` 端到端 ✅；测试全绿 ✅；
参考↔CUDA 一致性 ✅（见下）。

**A1–A3 的 GPU 上机验证（2026-11-27）**

机器：Tesla PG503-216（V100 32GB，sm_70），CUDA 12.4，驱动 535，Python 3.10
+ cupy 14.2；代码上传到 `/root/afv`（独立目录，未碰共用机的 `/root/AetherFlow`），
跑 `verify_cuda.py`：

- conformance：**81 个声明算子 / 121 executed / 8 refused / 0 failed，PASS**。
  比较（eq/ne/lt/le/gt/ge）、`where`、`sign`、`round`、`floor`、`ceil` 全部
  `executed` 且 `err=0`；`topk_mask` / `topk_values` / `topk_indices` 共 6 条
  按契约 `refused`（未声明，符合预期）。
- `verify_cuda.py` 的边界用例新增 5 条：`where_bcast_pow2_shape`（折叠后每维
  都是 2 的幂的规则形状；曾经专走掩码 + 右移内核，该形态已于 5.300 删除，
  现在走 div/mod 路径）、`where_bcast_odd_rank3`（非 2 的幂，走 div 余数
  路径）、`compare_feeds_where`（比较结果直接当条件，bool 数据流端到端）、
  `rounding_boundaries`（`.5` 取偶 / 负数 / `-0`）、`sign_special`（`±0` /
  `±inf` / NaN）。原脚本的 tuning 回环用 `rounds=1`，已被 `search()`
  "至少 2 轮" 的约束拒绝，一并修成 2。
- **已知限制（环境级，不是代码问题）**：这台机器上 GPU 对次正规数
  （subnormal，如 `1e-38`）的比较一律为 false——`cp.sign(1e-38)` 自己返回 0、
  `cp.maximum(1e-38, 0)` 返回 0，即 FTZ 生效。参考后端是 numpy，语义不冲刷。
  因此所有一致性用例的输入都避开次正规数，否则会得到一个与代码无关的假失败；
  真实模型里出现次正规数的概率可忽略。此事记入 REMIND。

### 5.302　阶段 B：设备侧归约、分桶反向重叠与最小启动器

**B1：`DistributedDataParallel` 在设备态直接交给 NCCL。** 新增
`NcclDeviceCollective`（`train/zero.py`），契约与 `NcclHostCollective` 相同，
但少一次往返：

- 输入是设备数组（带 `__cuda_array_interface__`）时把缓冲原地交给 NCCL，
  归约结果留在显存；输入是主机数组（损失标量那条极小的路径）时才做
  H2D → NCCL → D2H。
- `device_side = True` 是显式声明。DDP 只在 collective 带这个属性时才接受
  设备态 Trainer，否则构造期报错，不静默退回主机中转。设备态还要求
  `bucket_bytes=0`（没有通用的设备拼接）与 `threads=1`（同一个 NCCL
  communicator 不能被多线程同时调用）。
- `reduction="mean"` 直接以 `op="mean"` 交给适配器，在设备上补一次原地除法；
  `train/nccl.py` 仍只登记 sum / prod / max / min。
- 归约层有 5 条单测（设备态拒绝无声明 collective、拒绝拼桶 / 多线程、走
  `op="mean"`、原地归约返回同一缓冲、拒绝非浮点梯度）。**端到端未验证**：
  手上只有单卡 A800 与两卡数值用例，`world > 1` 的真机联调要等设备。

**B2：反向按桶拆开，桶就绪即发。** `Trainer.enable_bucket_overlap` 为每个
参数桶建一张独立反向图，桶算完立刻回调；DDP 的 `overlap=True` 把回调接到
后台单线程 `_BucketReducer` 上：

- 单线程而不是线程池：集合调用必须各 rank 同序。桶的提交顺序由反向图执行
  顺序决定（各 rank 图相同），并发提交会让顺序取决于调度——一侧先发 A、
  另一侧先发 B 就会互相等死。
- 权重更新仍在 `grad_hook` 里等全部桶归约完才发生；`reduction="mean"` 与
  形状 / dtype 校验沿用主机路径，数值与"整图反向 + 一次归约"逐位相同
  （单测在 6 参数 / 3 桶上逐步对拍）。
- **代价**：每桶反向都要重算从损失到该桶的共同链式路径，桶越多越慢。因此
  overlap 要求 `bucket_bytes > 0`（禁止逐参数一桶），并限制在主机态、
  `accumulate_steps=1`、无 checkpoint 分段、`threads=1`。
- 为什么不做执行器逐节点回调：整图反向是一次 `backend.execute`，本层拿不到
  "某个参数梯度算完"的时刻；把反向拆成桶是对后端零改动的最小做法。

**B3：rank / world 从环境变量读。** `DistributedDataParallel.from_env` 复用
`env_rank` / `env_world_size`（`AF_RANK` / `AF_WORLD_SIZE`），`collective`
缺省 `HostCollective.from_env()`；最小启动器仍是 `host_collective.launch`，
不新造一套。设备态由调用方显式传 `NcclDeviceCollective(comm)`——
communicator 的 `init_rank` 参数本层不猜。

**规模与回归**：`python -m unittest discover -s tests` → **2958 tests, OK**
（skipped 452，全部需要 GPU）。新增 15 条：DDP 设备态 6 条、分桶重叠 5 条、
`NcclDeviceCollective` 4 条。

### 5.303　投机解码接入 LMRunner 主干：复用图缓存、驻留与前缀

5.301 把"prefix caching / KV swap / 投机解码"列为组件级交付、尚未接进主路径。
这一节补上投机解码的主干入口：`LMRunner.generate_speculative`。

**一份循环，两个入口。** `speculative.py` 的草稿 → 验收 → 提交循环抽成模块级
`_speculative_loop`（连同 `_snapshot_state` / `_feed_state` / `_verify_tokens`
三个状态工具），`SpeculativeDecoder.generate` 与
`LMRunner.generate_speculative` 都调用它。抽出来的动机不是复用几行代码，而是
**避免两条数值路径**：验收只认目标模型的 argmax，任何一处口径分叉（停止符、
奖励 token、提交块长）都会让"投机 = 贪心"的等式在其中一个入口失效。

**会话里真正被复用的三样东西。**

1. 图缓存：验收用的 `(长度, head="all")` 图与 `generate` 的 `(长度, "last")`
   共用同一个 `OrderedDict`，LRU 上限 `max_graphs` 不变；
2. 驻留权重：`resident=True` 时验收与提交的 feed 都不含参数，权重只登记一次；
3. `prime` 前缀：`_advance(prompt, chunk)` 从缓存状态起步，投机循环直接以
   `state` / `logits` 为起点，不再全量预填充。

生成结束后的口径与 `generate` 一致：`_clear_prefix()`，前缀与状态一并丢弃。

**边界照旧，不加新旋钮。** 只支持 `batch == 1`（每行接受长度不同，递归状态
无法整批回滚）；`device_state=True` 时仍走普通执行——验收一次要喂 `k` 个
token，捕获图只覆盖单步 decode，混用不会更快；采样参数一概不提供（投机只对
贪心成立）。`greedy_draft` 仍标注为正确性参照，不是性能方案。

**验证**：`tests.test_speculative` 18 条全过，新增 7 条——与
`generate(temperature=0)` 逐位一致且第二次调用图数不增长、与独立
`SpeculativeDecoder` 逐位一致、四类坏草稿改不了结果、`prime` 前缀可用且
生成后清空、`resident=True` 时 feed 不含权重且关闭后释放、`max_new_tokens=0`
不建图、七类拒绝面（batch、关闭、draft 类型、draft_tokens 零 / 布尔、越界、
chunk_size）。全量 `2965 tests, OK`（skipped 452）。

**未验证 / 不声称**：任何吞吐或显存数字。投机解码的收益取决于草稿命中率与
验收窗口，参考后端上的"一次前向确认几个"不能外推到真机；`world > 1`、设备态
捕获图与投机混用都未上机。GPU 上的验收仍待空闲机器。

### 5.304　分页 KV cache 的换出与换入

5.241 的分页缓存只有"追加 / 取回 / 回滚 / 回收"四个动作，容量不足时只能抛
`PagedCacheFullError`。这一节补上第五、第六个动作：`swap_out` / `swap_in`。

**语义。** `swap_out(rid)` 把整条序列每层的 K/V 拷一份、释放全部物理页、
块表清空，长度不变；`swap_in(rid)` 按长度重新领页、按块表位置写回，页号
可能与换出前不同，数值逐位不变。换出期间只有 `length` / `is_swapped` /
`block_table`（空）可查，`append` / `gather` 显式要求先换入；`truncate` 允许
裁剪主机副本（返回 0 页），`reset` / `free` 直接丢弃副本。换出只覆盖**已分配
过**的层，从未用到的层不会因为换出而突然占一份数组。

**换入失败保持换出态。** 需要的新页一次性领齐；空闲页不足时抛
`PagedCacheFullError`，主机副本与长度原样保留，腾出页后可重试。这条与
`append` 的"全有或全无"同源：半换入既占页又丢副本，比直接失败更难排查。

**不做真正的 D2H。** 缓存本身就在主机内存，"搬到主机"是把页归还空闲链、
把副本挂到序列上。真机上的显存换出由上层按同一语义执行（释放设备页、保留
主机副本）；这一层只保证存储语义与页复用。调度策略（谁该被换出、什么时候
换入）仍未接进服务控制面，属于后续工作。

**验证**：`tests.test_paged_cache` 21 条全过，新增 7 条：换出后释放页、
换入后逐位一致、换出的页被别的序列占用后换入失败且原样保留、腾出后重试
成功、换出期间读写被拒、换出态 `truncate` 裁剪后换入、`reset` / `free`
丢弃副本、重复换出 / 未换出换入的拒绝面、单层路径。全量 `2972 tests, OK`
（skipped 452）。GPU 侧顺带复跑 `tests.test_cuda_backend` +
`tests.test_cuda_train`：**383 条 OK**（V100 32GB、python3.10.12 +
cupy 14.2.0，2.9 s），确认阶段 A 的比较 / where / 取整内核在当前快照上无回归。

### 5.305　topk 的设备内核：k 轮块内选择

5.301 把 `topk_values` / `topk_indices` 落进了 IR 与参考后端，`topk_mask` 更早
就有，但三者都没有设备内核——CUDA/HIP 上会抛 `BackendCapabilityError`。
这一节补齐：三个算子共用一份手写内核，注册进能力面与调优面。

**算法。** 一个 block 处理一行，做 `k` 轮"块内最大值 + 并发取最小下标"：
每轮先让每个线程扫描自己那份列（跳过已选下标），得到局部最大值；`block_reduce_max`
归约出全局最大；胜出线程再用 `atomicMin` 把下标压到最小。这轮的赢家写进
`sel[rank]`，下一轮扫描时跳过它。并列口径与 `argmax_last` 完全一致（值相等时
小下标优先），因此 `k=1` 的结果就是 `argmax`；NaN 按 `+inf` 参与比较（
`torch.topk` 口径），但输出仍是原值（values 是 NaN，indices 指向该位置）。

复杂度是朴素的 O(rows · k · (cols + k))：每轮每线程扫一遍列，再检查 `rank` 个
已选下标。选择"简单可验证"而非"并行前缀和 + 分桶"是因为 k 在 MoE 路由里通常是
4～8，k 轮的块内归约已经远小于一次 GEMM；真要优化应先做"一次扫描出 k 大"
的分桶版，而不是把这个版本改复杂。

**动态共享内存放已选下标。** 最初的写法是用 `bool used[cols]` 标记已选位置，但那
把每行长度钉死在 48 KiB / 1 B = 49152 列；改成 `extern __shared__ int sel_dyn[]`
只存 `k` 个下标，每轮 O(k) 线性检查，长度不再受列数限制，只受 `4·k + 4 KiB`
（静态归约 scratch）≤ 48 KiB 约束。`k` 因此有上限（约 11k），超过时显式报错
而不是凑合跑。

**三个入口，一种输出参数。** values 输出 `%T%*`、indices 输出 `int*`、mask 输出
`%T%*` 的同形矩阵，签名无法统一；内核内部改成 `void* outp`，三个
`extern "C"` 入口各自做一次类型转换再调共用的 `topk_select_row`，省掉了三份
几乎相同的 device 代码。mask 模式不需要额外清零：输出缓冲在算子层用
`cp.zeros` 分配，内核只写 1。

**非末轴。** 与 `argmax` 同一套路：先 `transpose + ascontiguousarray` 把目标轴换到
末维，算完再按逆置换搬回来。代价是一次显存拷贝，换来一份内核覆盖所有单轴情形。

**能力面。** `GPU_OPS` 与 `_REDUCE_TUNABLE_OPS` 同时加入三个名字：f32 直接支持，
f16 由 `_F16_OPS = None` 自动放行，bf16 不加（与 `argmax` 一致，bf16 子集不含
比较类归约）。非浮点输入在 `invoke` 的 dtype 面就被拒绝，不会把 i32 当 f32 读。

**验证。** 本地全量 `2980 tests, OK`（skipped 460，新增 8 条 topk 用例在无 GPU
时整组跳过）。V100 真机（32GB、python3.10.12 + cupy 14.2.0）：
`tests.test_cuda_backend` **382 条 OK**（skipped 2），conformance 的 6 条 topk
（values / indices / mask × 末轴 / 首轴）从 `refused` 变为 **`executed` 且无误差**。
新增用例覆盖：末轴与非末轴四组形状、并列小下标优先、NaN 排最前、
`k == 轴长`、mask 每行恰 k 个 1、f16、i32 输入被拒。

**未声称**：任何 topk 吞吐数字。k 轮的块内归约在这个形状下不是瓶颈，但
O(k²) 的朴素结构在大 k 时一定不是最优；真要给数字得先把分桶版写出来再测。

### 5.306　分页缓存的调度策略：LRU 换出、访问换入与交错解码

5.304 只补了"换出 / 换入"这两个存储动作，谁该被换出、什么时候换入仍然没有
答案——代价是容量不足时调用方只能自己写一遍循环。这一节把策略补成一个可复用
的层，并接到真正的多序列解码上。

**`PagedCacheManager`（`runtime/paged_cache.py`）。** 在 `PagedKVCache` 外面
套一层 LRU：

- 每次 `append` / `gather` / `touch` 把序列移到使用顺序末尾；`_order` 的迭代
  顺序就是"最久未使用在前"。
- `append` / `ensure_resident` 捕获 `PagedCacheFullError`，从最久未使用的一条
  开始换出，再重试。跳过自己（正在等页的序列不能换出自己）、已换出的、pin
  住的，以及长度为 0 的（换出腾不出页）。每轮至少推进一条，重试循环最多
  "序列数 + 1"次，不会因为"换出了一条空序列"而空转。
- 换不进去时把原异常原样抛出。**已换出的序列保持换出态**——数据在主机副本里
  没丢，等页腾出来还能换入；这与 `append` 的全有或全无语义一致，宁可失败也
  不留半条序列。
- `pin` / `unpin` 的语义只是"自动换出时跳过"。正在解码的请求每步都要读 K/V，
  换出它等于下一步立刻换回来，白搬一遍；所以服务层在请求活跃时 pin，挂起时
  unpin。显式 `evict` 不受 pin 约束，那是调用方明确要求。
- `on_swap_out` / `on_swap_in` 两个钩子在页已归还 / 已重新领到时收到
  `(rid, pages)`。这一层仍然不做真正的 D2H，真机上的显存搬运由上层挂这两个
  钩子完成；钩子抛异常会向上传播（例如设备侧释放失败必须暴露）。

**`PagedDecodeScheduler`（`models/cached_attention.py`）。** 策略要落到真实
解码上才有意义。它构造时建一张固定形状的单步图（缓存宽度 `max_steps`，权重
只喂一次），之后：

- `submit(rid)`：分配空序列并 pin；
- `step(rid, x)`：`gather` 自动换入（页不够就先换出挂起的序列），执行单步图，
  `append` 自动写回。返回主机态的 `d_model` 向量；
- `suspend(rid, evict=False)`：解除 pin，成为换出候选；`evict=True` 立刻归还
  页（适合"这条要等很久"的调度决策）；
- `resume(rid)`：先 `ensure_resident`（页不够就按 LRU 换出别人）再 pin 回来；
- `finish(rid)`：解 pin、回收，id 可复用。

**失败的那一步不写缓存。** 所有序列都 active 而池又满时，`step` 抛
`PagedCacheFullError`。此时单步图已经算完 y，但 `append` 是全有或全无，序列
长度与 K/V 都没变——调用方 `suspend` 一条低优先级请求后重试**同一步**即可，
不会重复计数。这是有意的顺序：先算后写，换来"重试幂等"，代价是失败时白算
一步（比丢 token 或留下半条序列便宜）。

**为什么不做成"池满时自动挂起最低优先级请求"。** 优先级属于调度器的业务
决策，缓存层看不到请求的 SLA、队列位置或用户等级；把优先级塞进 LRU 会把
两件事耦合在一起。现在的边界是：缓存层给确定性策略（LRU + pin）与明确的
失败信号，调度器决定挂起谁。

**验证。** 本地全量 `2995 tests, OK`（skipped 460）。新增 15 条，全部纯
Python：

- `tests.test_paged_cache` 10 条：容量不足自动换出后逐位一致、`touch` 改变
  受害者、pin 跳过、全 pin 时抛错且所有序列状态不变、单条序列不能换出自己、
  换入失败保持换出态并在腾页后重试成功、钩子的 `(rid, pages)`、显式 `evict`
  不受 pin 约束、`truncate` / `reset` 转发、十类拒绝面与幂等 `free`；
- `tests.test_attention_cache` 5 条：两条序列交错解码与稠密图逐位置一致、
  池只够一条时挂起 → 换出 → 恢复后继续逐位一致（`swaps_out=2` /
  `swaps_in=1`）、active 序列被 pin 时宁可抛错也不换出、`suspend(evict=True)`
  立刻归还页、构造与状态机的拒绝面 / id 复用。

**未声称 / 未验证**：任何吞吐或显存数字。管理器与调度器都是主机逻辑，没有
真机换出开销的实测；`PagedDecodeScheduler` 仍要求 `batch=1` 与静态
`max_steps`，也没有接进 `LMRunner` 主干（那个主干是 Mamba，没有随上下文增长
的 KV）——它是组件级的服务控制面，与 5.304 的定位相同。

### 5.307　分页 KV 的共享前缀：块级引用计数与写时复制

5.304 / 5.306 解决的是"容量不够怎么办"，没有回答"同一段上下文被多条序列需要
怎么办"。最典型的三种用法——同一提示的多路采样、束搜索的分支、同一个系统提示
下的多条请求——前缀完全相同，各拷一份 K/V 是最贵的浪费：页数翻倍，cache 的
命中率却一分不涨。

**做法是块级引用计数加写时复制。** `PagedKVCache` 新增 `_refs`（物理页 → 持有
它的序列数），并多出一个动作 `fork(parent, child)`：

- `fork` 把父序列的块表整份共享给新子序列，每页引用加一。`used_blocks` 不动，
  `shared_blocks`（引用 > 1 的页数）变大；
- 任何一方之后 `append` **写到已有页**时先拆页：领一个新页、整页拷贝、改自己的
  块表、把旧页引用减一。只拆真正被写到的页，没碰到的共享页继续共用（追加到全
  新的页时连拆都不用）；
- `free` / `reset` / `truncate` / `swap_out` 全部改走 `_give_back`：引用减一，
  **降到 0 才回到空闲链**。因此"放手"和"腾出"是两件事。

**返回值也跟着变了。** `truncate` / `reset` / `swap_out` 原本返回"块表里放手的
页数"，在共享下这个数会骗人（放手 2 页可能一页也没腾出）。现在它们返回**实际
回到空闲链的页数**：一条全共享的序列换出后返回 0，主机副本照样独立保存。这让
"换出腾了多少容量"可以直接读返回值。

**自动换出跳过"换了也白换"的候选。** `PagedKVCache.exclusive_blocks(rid)` 给出
只被这条序列引用的页数，也就是换出它的收益上限。`PagedCacheManager._evict_one`
跳过 `exclusive_blocks == 0` 的序列：否则 LRU 会先搬走一条全共享的父序列、腾出
零页、再搬第二条，白白多一次换出，还会把 `swaps_out` 计数抬高。

**容量检查把拆页算进去。** `append` 在动手前同时算出"新增页"和"要拆的共享页"，
两者之和超过空闲页就抛 `PagedCacheFullError`，长度、块表、引用计数全部原样
保留——与 5.304 的"全有或全无"一致；拆到一半失败会留下引用计数错乱的块表，
比直接失败难查得多。

**只接受驻留序列。** `parent` 处于换出态时 `fork` 抛 `ValueError` 提示先
`swap_in`，不做一个"共享主机副本"的隐式路径：换出态没有物理页可共享，静默复制
一份会让"共享"这个承诺失去意义。

**调度器也接上了 fork。** `PagedDecodeScheduler.fork(parent, child)` 把已有前缀
交给一条新的 active 序列：它转给缓存层做块级共享，再给子序列 `pin`——两条序列
都按 active 管理，各自的 `suspend` / `finish` 互不影响。父序列处于换出态时
显式要求先 `resume`，不做一个隐式的"先换入再共享"。之后两条序列交替 `step`，
被写到的共享尾页各自拆开，隐状态与各自单独解码逐位一致。

**验证。** 本地全量 `3007 tests, OK`（skipped 460）。新增 12 条，全部纯 Python：

- `tests.test_paged_cache` 9 条：分叉不多占页且两侧逐位相同、子序列只拆被写的
  尾页（父序列内容不变）、父序列追加不穿透到子序列、一侧 `free` 后另一侧仍能读、
  `truncate` 只减引用不腾页、拆页空间不够时状态原样保留、全共享序列 `swap_out`
  返回 0 且另一方不受影响（`swap_in` 领新页后不再共享）、链式 fork 共用一组页、
  四类拒绝面（未分配 / 子 id 已分配 / 非 int / 父已换出）；
- `TestManagerFork` 1 条：管理器转发 `fork`，并在子序列写共享尾页需要新页时，
  跳过全共享的父序列、换出独占两页的第三条序列，`swaps_out=1`；
- `TestPagedDecodeScheduler` 2 条：共享 2 个 token 的前缀后分叉、交错解码的
  隐状态与各自整段稠密前向逐位一致（分叉不额外占页、`shared_blocks` 等于分叉前
  的占用）；三类拒绝面（父未提交 / 子 id 已提交 / 父已换出）。

**未声称 / 未验证**：真机 D2H 与拆页的显存代价（这一层仍全在主机内存）；
按内容寻址的前缀哈希（`fork` 是显式调用，共享点由调用方决定，不做前缀匹配）；
调度器仍要求 `batch=1` 与静态 `max_steps`，没有接进 `LMRunner` 主干。5.310 已把
这一层接成 token→token 的生成闭环（`PagedGenerator`）；`LMRunner` 因为主干是
Mamba、没有 KV，本来就不该是它的归属。

### 5.308　W8A8 的参考语义：int32 累加，设备端显式拒绝

5.301 的第 3 项缺口是"量化只到 weight-only int8"：权重压到 1 字节，计算仍在
f32 上做反量化 GEMM。它省的是权重占用与读带宽，**不省算力**——真正的 int8
矩阵乘要把激活也量化，让乘加在整数域完成。这一步在本地能闭环的部分是**语义**，
不是性能：设备内核需要 int8 tensor core 的调度与上机测数，那是另一件事。

**新算子 `linear_i8a8`。** 输入是 ``(x_q, x_scale, w_q, w_scale[, bias])``：

- ``x_q`` 是 int8 激活 ``(..., K)``，``w_q`` 是 int8 权重 ``(K, N)``；
- 前向是 ``out = (int32(x_q) @ int32(w_q)).f32() * x_scale * w_scale [+ bias]``。
  整数累加在 int32 里完成，**不是**先反量化再在 f32 上逐项相加；
- ``x_scale`` 每个 token 一个（形状广播到 ``x_q``，最后一维必须是 1，如
  ``(B, T, 1)`` 或标量）；``w_scale`` 每个输出通道一个（``(N,)`` 或标量）。

**两条"能广播 ≠ 能提出求和号"的约束。** 把 scale 乘到累加结果之外，前提是
它沿 K 维恒定。``x_scale`` 形状 ``(B, K)`` 虽然能广播到 ``(B, K)``，但每列不同，
乘到 ``int32`` 结果上已经不是原来的点积；``w_scale`` 的 ``(K, N)`` 同理。这两种
形态在 `infer` 处显式拒绝，而不是静默地当成 per-token / per-channel。这与
`linear_i8` 不同——后者的 scale 逐元素乘在权重上，允许 ``(K, N)``。

**整数溢出是语义问题，不是实现细节。** int32 累加的上界是
``K * 127 * 127 ≤ 2**31 - 1``，即 ``K ≤ 133144``。超过时 int32 会回绕，而回绕
依赖溢出行为，不能跨后端对拍。构图期（K 静态时）与执行期（按实际数组）各查
一次，超过就拒绝并提示分段累加；不做"让它绕一圈"。

**反向只给 bias。** int8 激活、int8 权重、两个 scale 都是量化域的量，没有连续
梯度，与 `linear_i8` 同口径返回 `None`（该输入不接收梯度），而不是造一张零
张量。要训练量化模型需要 QAT，不在这一版。

**工具与能力面。** `models.quantize` 增加：

- `quantize_activation(x)`：按 token（最后一维）取 amax 得到 `(..., 1)` 的 scale，
  与 `quantize_i8` 共用同一份 RNE 舍入；全零 token 给 scale=1；
- `build_int8a8_linear(g, K, N)`：把上面四个输入 + bias 落成一个 IR 节点。

CUDA / HIP **没有** `linear_i8a8` 内核，`GPU_OPS` 不含它，执行时会收到显式的
"后端不支持该算子"，而不是悄悄退回 f32 累加；`test_declared_ops_match_kernel_methods`
（CUDA_OPS == 已实现方法）不受影响。conformance 新增 `linear_i8a8_bias` 形态，
在参考后端执行、在设备后端记为 refused——这是显式拒绝的正常路径。

**验证。** 本地全量 `3020 tests, OK`（skipped 460）。新增 13 条，全部纯 Python：

- `TestLinearI8A8` 6 条：与手写 int32 公式**逐位相等**；K=512 时与"先反量化再
  f32 matmul"确实不同（证明新语义不是别名）；per-tensor 激活 scale；八类拒绝面
  （dtype / 秩 / 内维 / 沿 K 变化的 scale / 2 维权重 scale / 长度 / 输入个数 /
  K 超上界）；非正 scale 在执行期报错；JSON 往返逐位相同；
- `TestQuantizeActivation` 3 条：per-token scale 形状与"误差 ≤ scale/2"、全零
  token 的 scale=1、三类非法激活；
- `TestQuantizedW8A8Builder` 2 条：端到端与 f32 `x @ w + b` 在量化误差内一致、
  in/out 维度必须为正；
- `TestLinearI8A8Gradient` 2 条：bias 梯度等于 `dy.sum(0)`、对 int8 输入与
  scale 求导抛 `NoGradientError`；
- conformance 集合补 `linear_i8a8_bias`。

**未声称 / 未验证**：没有 int8 tensor core 内核、没有任何真机数字，因此本文
不主张 W8A8 "更快"；没有 QAT、没有 fp8、没有非对称零点、没有 per-token 权重
scale；没有分段累加（K 过大直接拒绝）；CUDA / HIP 的 refused 计数会因新形态
各加一，历史 conformance 数字（62 executed / 5 refused）不再代表当前形态总数。VJP 表从 49 条增到 50 条。
**5.309 已补 CUDA / HIP 的设备内核；本节"设备端显式拒绝"只代表 5.308 当时的状态。**

### 5.309　W8A8 的设备内核：i8 读入、int32 累加，最后才缩放

**动机。** 5.308 给了 `linear_i8a8` 的参考语义，但设备端显式拒绝——这在能力面上
是诚实的，在推理链路上却是断的。本节补内核：把"累加留在整数域"这件事真正落到
设备上，而不是悄悄退回 f32 GEMM 再声称是 W8A8。

**内核 `af_linear_i8a8`。** 一个线程算一个输出元素：沿 K 读 `x_q[row, kk]` 与
`w_q[kk, col]`，用 int32 累加，最后 `(float)acc * x_scale[row] * w_scale[col]`。

- CUDA 走 `__dp4a`（4 路 int8 乘加，sm_61+；V100 是 sm_70、A800 是 sm_80），
  K 不是 4 的倍数时用标量加收尾；
- HIP 的 hipRTC 不带 `hip_runtime.h`，同一份签名用标量乘加；
- 两条路都是**精确整数运算**，累加结果逐位相同，因此跨后端对拍成立。两份源码
  只在 `_LinearMixin._I8A8_DOT4` 上分叉（`CudaBackend` 覆盖为 True），语义不分叉。

**缩放与 bias 的顺序与参考实现一致**：`(float)acc * xs * ws`，bias 交给既有
`add` 内核，对应参考端的 `acc.astype(f32) * row * ws + b`。IR 已守住
`K <= 133144`（`K * 127 * 127 < 2**31`），int32 不会回绕。

**设备端校验。** 两个输入必须是 i8、两个 scale 必须是 f32；激活秩 >= 2、权重秩 2；
`x_scale` 形状必须是激活前导维加 1；内维、`w_scale` 长度、bias 的 dtype 与长度
逐一核对，违反即 `BackendCapabilityError`。设备端是最后一道，不假设 IR 一定被走过。

**能力面。** `GPU_OPS` 增 `linear_i8a8`，dtype 形态表把 0 / 2 号输入标成整数
（1 / 3 / 4 仍是浮点）；`_LinearMixin` 注册 `linear_i8a8`。真机报告的
`caps: 85 个算子`，与 `CUDA_OPS == 已实现方法` 的约束一致。

**验证。**

- 远端 Tesla PG503-216（V100 32GB / sm_70 / CUDA 12.9 / cupy 14.2.0）：
  conformance **128 executed / 2 refused / 0 failed**，`linear_i8a8_bias`
  的 `err = 0.0e+00`（与参考**逐位相等**），coverage 85/85；
- `verify_cuda.py` 新增两条边界：K=13（dp4a 尾处理）+ 秩 3 + bias、K=512 无 bias，
  都 PASS；完整脚本（含调优与内存预算）全部通过；
- 临时脚本另对 7 种形态逐位对拍：K ∈ {1, 2, 7, 8, 13, 512, 133144}、秩 2 / 3、
  有 / 无 bias、N ∈ {2, 3, 4, 5, 6, 32}，全部与参考逐位相等；
- 本地全量 `3020 tests, OK`（skipped 460）——本轮没有新增纯 Python 用例，设备
  路径由 conformance 与 `verify_cuda.py` 覆盖。

**未声称 / 未验证。**

- 这不是 int8 tensor core：`__dp4a` 是一条标量化的 4 路点积指令，吞吐远低于
  IMMA / wmma 的 int8 矩阵乘；没有做共享内存分块、寄存器分块或 warp 级 mma。
- 不给性能卖点。实测 `M = K = N = 512` 一次 `execute` 的中位墙钟是 1.14 ms，
  这是整条执行路径（构图找节点、传输、启动）的耗时，不是内核吞吐，不能拿来
  与 cuBLAS 比；本节的收益是"累加在整数域、权重与激活都只读 1 字节"的语义与带宽。
- HIP 路径没有真机验证（本机无 ROCm），只保证与 CUDA 共用同一份语义与同一套
  逐位对拍条件；WSL 上的 RX 7650 GRE 本轮没有复跑。
- 仍无 QAT、无 fp8、无非对称零点、无分段累加（K 过大直接拒绝）。

### 5.310　分页 KV 的生成闭环：token 进、token 出

5.303 把投机解码接进了 `LMRunner`，5.304 / 5.307 给了换出换入与共享前缀，但
这几节的验收都停在"隐状态"这一层。`PagedDecodeScheduler.step(rid, x)` 吃隐状态、
吐隐状态；真实推理要的是 token 进、token 出。缺的三件事不在分页层，而在分页层
外面：把 token 变成隐状态的嵌入与位置编码、把隐状态变成下一个 token 的词表头、
以及逐 token 的停止条件。

**为什么不是 `LMRunner`。** 5.301 把"未接主路径"挂在 `LMRunner` 名下，实现这一节
时才发现这个说法需要修正：`LMRunner` 的模型是 Mamba（SSM 递归状态 + 卷积状态），
根本没有 K/V，分页 KV 接不进它。分页 KV 的主路径只能是 Transformer 增量解码——
`PagedDecodeScheduler` 已经是那条路径的核心，缺的是它外面的生成闭环。所以新增
`PagedGenerator`（`models/cached_attention.py`），不碰 `LMRunner`。

**`PagedGenerator` 只补三件事**，其余全部转发调度器：

1. 嵌入与位置编码在**主机侧相加**：`x = tok_emb[token] + pos[t]`。增量图的输入
   本来就是隐状态，位置信息不必在设备上再算一遍；"位置表有多长"因此成为一个
   显式参数（`max_steps`）。
2. 词表头是一张 `(1, d) → (1, V)` 的小图（`y @ w_head`），语义等同
   `build_transformer` 的末层。没有把它塞进 `build_cached_decoder_step`：那个
   函数的返回值面是 `(inputs, y, new_kv)`，为一个可选输出改元数会牵动所有调用
   方；一次额外的 `execute` 换来零侵入。
3. 停止条件：贪心 argmax 与 EOS。采样参数一个都不给——温度 / top-k 请走
   `LMRunner` 的采样路径，投机与分页这两条都只对贪心成立。

**调度器一行不改。** 生成器给 `PagedDecodeScheduler` 传一份
`replace(spec, vocab=0)`：分页解码只解隐状态，词表头与嵌入不属于它。
`_check_spec` 仍然拒绝 `vocab != 0`，这条约束没有被放松，只是换了一个不触发它的
spec。

**`fork` 需要 token 历史。** 共享前缀要校验"子序列的 prompt 真的以父序列已解码
的全部 token 开头"，而调度器只记长度、不记 token。生成器自己维护 `_tokens[rid]`：
`fork` 复制一份、`feed` 追加一个。由此定下一条边界：`fork` 之后 prompt 必须
**比父序列长至少一个 token**——生成需要一个尚未解码的位置来取下一个 token，父
序列最后一步的 logits 当时没有保存，这里不做"回滚一位重算"。

**容量与位置的显式边界。** `gather(pad_to=max_steps)` 是既有约束，所以池容量
（`num_blocks × block_size`）必须装得下 `max_steps`；位置表长度也必须
`>= max_steps`，`prompt + max_new_tokens` 超过 `max_steps` 在执行前就报错。都是
构造期 / 入口校验，不靠内核越界。

**验证。** 本地全量 `3026 tests, OK`（skipped 460），新增 7 条纯 Python 用例
（`tests.test_attention_cache.TestPagedGenerator`）：

- 贪心 token 与稠密图逐 token 相等：页宽 1 / 2 / 3 各一遍，判据是每步整段重建
  `build_transformer` 前向、取最后位置 argmax，与分页增量解码完全独立；
- EOS 提前停止（取参照序列中部一个此前没出现过的 token 当 EOS）；
- `fork` 不额外占页（`used_blocks` 不变、`shared_blocks` 等于分叉前占用、
  `forks=1`），子序列续写与独立实例逐 token 相等，父序列长度与 token 历史不被
  写脏；
- `suspend(evict=True)` + `resume` 之后续写的 token 与不换出时完全一致
  （`swaps_out=1 / swaps_in=1`）；
- `prefill` / `feed` 直调与历史副本语义（改返回值不影响内部）；
- 拒绝面覆盖：`vocab=0`、`max_steps` 超位置表、`pos` / `tok_emb` 形状、
  缺 `w_head`、空 prompt、token 越界 / 浮点 / bool / 字符串、`max_new_tokens`
  负数 / bool、超出 `max_steps`、EOS 越界、未提交 / 重复提交 / 自 fork、
  fork 前缀不一致 / 同长、到达 `max_steps`。

**未声称 / 未验证。**

- prefill 仍是逐 token 调 `step`，没有整段 prefill 图；长 prompt 的 prefill 只
  保证语义，不保证吞吐。
- 只有贪心；没有束搜索、投机、采样温度。`fork` 只共享"已经解完的前缀"，不做
  前缀哈希匹配。
- 全部是参考后端 + 纯 Python 用例，没有上真机；设备后端走的是同一张
  `PagedDecodeScheduler` 图与同一个小词表头图，但没有 D2H 次数与显存数字。
- 没有接 HTTP 服务面（`serve`）；连续批处理仍是 `ServingEngine` 自己的那条
  （Mamba）路径。

### 5.311　QAT 的假量化：量化误差进前向，梯度走直通

**动机。** 5.301 的缺口表里量化一行还写着"无 QAT"：weight-only int8 与 W8A8
把权重 / 激活真的落成 int8，能省驻留与带宽，但训练图中的权重始终是 f32——
量化误差只在推理时才出现，模型没有机会在训练中适应它。QAT 的通行做法是
训练时用"假量化"：前向把权重舍到 int8 的格点上再放大回 f32（数值上等同
推理时会看到的误差），反向用直通估计器（STE）把梯度送回去。

**一个算子，不是一张子图。** `fake_quantize_i8(x, scale)` 是 f32 → f32 的
新算子，前向逐位等于 `dequantize_i8(quantize_i8(x, scale), scale)`：
参考实现先真的落成 int8（`np.rint` 半偶舍入、夹到 ±127）再放大，
舍入与端点饱和只有一份定义。把它写成"图上串联两个算子"也能得到同样的
前向，但那条链的中间类型是 i8，`quantize_i8` 没有、也不该有反向规则，
训练图走不过去。**独立成算子的唯一理由在反向**。

**反向是 STE，且不做饱和掩码。** `dx = dy`，梯度原样穿过量化；饱和区也
不额外截断。这与 PyTorch `fake_quantize_per_tensor_affine` 的反向口径一致，
也是"让量化误差只影响前向数值"的最简表达。`scale` 是量化参数而不是连续
变量，与 `linear_i8` / `linear_i8a8` 同口径返回 `None`（该输入不接收梯度，
`backward` 对它显式报 `NoGradientError`）；要学 scale（LSQ）需要另一条规则，
不在这一版。

**设备端零新内核。** `_op_fake_quantize_i8` 直接调用既有的
`_op_quantize_i8` 与 `_op_dequantize_i8`，前向与"显式串联两个算子"同源。
注册进 `device_backend` 与 `GPU_OPS` 之后，CUDA / HIP 的承接数从 69 到 70
（68 个自研内核 + 2 个组合算子）。conformance 增加
`fake_quantize_i8_channel`，与 `quantize_i8` / `dequantize_i8` 的形态并列。

**验证。** 本地全量 `3033 tests, OK`（skipped 460），新增 7 条纯 Python 用例
（`tests.test_quantize.TestFakeQuantizeI8`）：

- 前向与"显式串联 `quantize_i8` → `dequantize_i8`"的图逐位相等；
- RNE 半偶（0.5→0、1.5→2、2.5→2、-0.5→0）与 ±127 饱和（127.6→127、
  -127.6→-127）；零输入对任意 scale 恒为零；
- STE：`y = matmul(x, fake_quantize_i8(w, s))` 的 `dw` 与"不做假量化"时
  逐位相同；对 `s` 求梯度显式报 `NoGradientError`；
- 拒绝面：非 f32 输入 / scale、scale 秩超过输入、符号维与静态非 1 维的
  广播、参数个数、非正 scale（执行期）；
- JSON 往返后执行结果逐位相同。
- 远端 V100 32GB（`sm_70` / CUDA 12.9 / cupy 14.2.0）上 `conformance`
  129 executed / 2 refused / 0 failed，`fake_quantize_i8_channel` 误差 0
  （设备端组合路径与参考逐位相等），同批 `quantize_i8` / `dequantize_i8` /
  `linear_i8a8` 也全过。

**未声称 / 未验证。**

- 只做 weight-only 的假量化算子本身；没有 W8A8 的 QAT 构建器，也没有把
  假量化自动插进 `Trainer` / 构件层——调用方自己在图上插。
- scale 不可学（无 LSQ）、无 per-token 动态 scale、无非对称零点、无 fp8。
- 设备端只验证了数值一致（V100 上误差 0），没有 QAT 训练吞吐 / 收敛性的
  真机数字；7 条新增用例全部在参考后端上跑。


### 5.312　LSQ：量化步长本身也能学

**动机。** 5.311 的假量化把量化误差带进了前向，但 `scale` 仍是外部给定的
超参：逐通道最大绝对值、或推理前的校准结果。步长选小了，大值被夹到 ±127、
误差集中在一小撮权重上；选大了，小值挤在少数几个码点里。LSQ（Esser et al.
2020）的做法是把步长当成可训练参数，让"量化误差往哪边推"直接进梯度。

**另立算子，不改 5.311 的语义。** `fake_quantize_i8_lsq(x, scale)` 的前向与
`fake_quantize_i8` **逐位相同**（共用同一份参考实现，见 `ops.py`），差别只在
反向。给同一个算子加 `learnable_scale=True` 之类的属性也能做，但那样"这个
scale 到底会不会被训练"就藏进了 attrs：同一张图换个属性，`backward(wrt=[s])`
从报错变成返回梯度。两个算子、两条 vjp 规则，语义边界写在名字上。

**反向的两条规则。** 记 `v = x / s`、`v̂ = clamp(round(v), -127, 127)`、
`x̂ = v̂ * s`：

- `dx = g * 1[-127 <= v <= 127]`：舍入处的解析梯度为零，仍用直通；但被夹到
  端点的元素不再接收梯度——它们的输入再怎么变也跨不出量化格点。5.311 的
  STE 版不区分这一段，LSQ 版区分。
- `ds = sum(g * (v̂ - v))`：非饱和区把 `∂v̂/∂s` 记为零，于是 `v̂ - v` 就是
  "量化误差把步长往哪边推"；饱和区 `v̂` 就是端点，同一式子仍然成立。这是一条
  干净的归约：`reduce_sum` 沿广播维求和回 `scale` 的形状。

非饱和判定用 `v == clamp(v)` 而不是 `v̂ == round(v)`：后者在 (-127.5, -127)
这类半格点上会把已经饱和的元素误判成可导。两条规则都用 IR 已有算子表达
（`div` / `round` / `clamp` / `eq` / `cast` / `mul` / `sub` / `reduce_sum`），
反向图本身可序列化、可执行、也落在设备算子面内（测试里直接断言）。

**`scale` 必须是图输入。** 它是这条规则唯一能拿到非零梯度的量化参数；若接成
`constant`，梯度会在常量节点处按"没有输入"被丢掉——前向仍是假量化，训练却
不会更新它。这是 LSQ 与 STE 版在接口上的唯一约束，写进算子 doc 与测试。

**设备端零新内核。** `_op_fake_quantize_i8_lsq` 直接转调
`_op_fake_quantize_i8`，前向与 STE 版同源；CUDA / HIP 的承接数从 70 到 71
（68 个自研内核 + 3 个组合算子）。conformance 增加
`fake_quantize_i8_lsq_channel`，与 5.311 的形态并列。

**验证。** 本地全量 `3040 tests, OK`（skipped 460），新增 7 条纯 Python 用例
（`tests.test_quantize.TestFakeQuantizeI8LSQ`）：

- 前向与 `fake_quantize_i8` 逐位相同（量级跨过饱和端点）；
- `dx`：饱和列（`w=100, s=0.1`）梯度为 0，其余列与 STE 版一样原样穿过；
- `ds` 与手算 `sum(g * (v̂ - v))` 对拍（逐通道，含一个下饱和通道）；
- 沿负梯度走一步，量化误差严格下降（"步长真的能学"的最小证据）；
- 反向图整体落在 `GPU_OPS` 内（否则"设备上能训练"只是口号）；
- 拒绝面：非 f32 输入 / scale、scale 秩超过输入、参数个数、非正 scale；
- JSON 往返后执行结果逐位相同。
- 远端 V100 32GB 上 `conformance` 130 executed / 2 refused / 0 failed，
  `fake_quantize_i8_lsq_channel` 误差 0（与 5.311 形态共用同一对内核）。

**未声称 / 未验证。**

- 只做假量化算子本身；没有 LSQ 的 W8A8 构建器，也没有接入 `Trainer` 的
  参数分组（scale 要作为参数传进去）。
- 固定对称端点 ±127，没有 per-token 动态 scale、没有非对称零点、没有 fp8。
- 不约束 `scale` 的正性：优化器把它推成非正时，执行期按 5.308 的口径报
  `ValueError`——这是有意的（量化的步长为负没有定义），但没有额外的投影。
- 只验证了梯度公式与前向一致性，没有真机训练吞吐 / 收敛性数字。

### 5.313　权重入口：自己解析 safetensors

**为什么这一项排在"内核差距"前面。** 1.4 的对照表把差距分成三类：定位差异、
硬缺口、竞争性缺口。内核数量、融合深度属于第三类（取决于跟谁比）；而"权重装
不进来"属于第二类——它不挑对手，直接挡在所有真实模型前面。此前的权重通路只有
`train/checkpoint.py` 自己的存档格式，能续训，但读不了别人的文件。

**为什么不引入 `safetensors` 包。** 三条理由，按重要性排：

1. 运行期依赖必须只有 numpy（`pyproject.toml` 只声明 `numpy>=1.24`）。为一个
   容器格式加一个二进制依赖，会让"参考后端在任意平台可用"这条约束退化。
2. 格式本身很小：8 字节小端头长 + 一段 JSON 头 + 连续数据区。标准库足够，
   而且能被逐行审。
3. 解析器要按"它是坏的"来写，而这正是第三方库不肯替你决定的部分——它们是通用
   读取器，校验口径与本项目的"显式失败"不完全一致。

**先校验，后读。** 头里的每个张量在拿到内存之前都要过五关：dtype 在白名单内、
`shape` 是 `>= 0` 的整数列表、`data_offsets` 是递增的两个整数、区间不越过数据区、
区间字节数正好等于 `numel * itemsize`。另外头长本身卡在 100 MiB（safetensors 官方
实现的同一上限）——一个声称"头有 400 GB"的文件只可能是在诱导读取方先分配内存。
坏头的报错信息带具体张量名与期望 / 实际值，现场不需要再猜。

**两个数值约定。**

- **bf16 无损展宽成 f32。** numpy 没有 bf16 类型；左移 16 位是精确的，NaN 的
  payload 也跟着保留。返回值是 f32，调用方不需要知道源文件是 bf16。反过来若
  返回 `uint16` 位模式，每个使用方都要自己写一遍移位，且极容易漏。
- **fp8 解码成 f32 载体。** `F8_E4M3` / `F8_E5M2` 是 1 字节位模式，`tensor()`
  用 `unpack_fp8` 解码成 f32（fp8 -> f32 是精确的），与 bf16 的展宽同一策略：
  调用方拿到的是数值，不需要自己拆指数 / 尾数。e4m3 的 `0x7F` 是 NaN、没有
  inf（5.318），e5m2 的 `0x7C` 是 inf；这些特殊值原样保留。返回 `u8` 位模式
  是"能跑但语义错"，所以在读取入口一次解完。

`tensor()` 总是返回拷贝。`np.frombuffer(bytes)` 得到的是只读视图，让它带着
`writeable=False` 流进训练，错误会在第一次参数更新时才炸——这是最贵的一类延迟失败。

**装载的三条"不猜"。** `load_into_store(store, tensors, mapping=None,
transpose=(), strict=True, allow_downcast=False)`：

- **名字不猜。** HF 的 `model.layers.0.self_attn.q_proj.weight` 与本仓库的 `w_q`
  没有可推导关系。默认同名对应，不同名的必须写 `mapping`；`mapping` 里出现 store
  不认识的键直接报错（拼错名字静默失效是最难查的错）。
- **布局不猜。** PyTorch `nn.Linear` 是 `(out, in)`，本仓库 `matmul` 是 `x @ W`
  即 `(in, out)`，差一次转置。要求显式列在 `transpose` 里——猜错不会报错，只会
  训出一个静默错误的模型，这比崩溃更糟。
- **精度不猜。** 目标 dtype 与源 dtype 不是安全转换（`np.can_cast(..., "safe")`
  为假）时，必须 `allow_downcast=True`。f16 → f32 自动放行，f64 → f32 要显式同意。

**原子性。** 先遍历所有参数收集问题、一次性报全（不是遇到第一个就抛），全部
校验通过后才写；写入用 `dst[...] = ...` 原地覆盖，保持优化器持有的数组对象身份
（与 `train/checkpoint.py` 的同一约定）。中途失败不会留下"一半新权重、一半旧
初始化"的参数——那种状态不报错、训练也能跑，但结果永远是错的。

**分片。** `load_safetensors_dir` 按文件名排序读 `*.safetensors` 并合并；同名张量
出现在两个分片里直接报错并指出两个文件（那是打包错了，静默取后者会加载出一半
旧权重）。`*.index.json` 只是清单，不参与解析——排序全读结果相同，少一条信任链。

**验证。** 新增 38 条用例（`tests/test_weights.py`），全量 `3078 tests, OK`
（skipped 460）：

- 往返：f32 / f16 / f64 / i8 / i16 / i32 / bool / 0 维标量逐位相同；bf16 四个
  位模式（±1.0、0.0、NaN）展宽为 f32；
- 坏文件 15 类：短文件、头长超上限、头越过文件尾、非 JSON、非对象、
  `__metadata__` 非字符串映射、未知 dtype、`shape` 非列表、负维、offsets 非二元组、
  逆序、越界、字节数与 shape 不符、张量描述非对象、空名字；
- fp8 两种格式按位模式解码成 f32（e4m3 的 448 与 -0、e5m2 的 inf 与 NaN）；
- 返回值可写、且两次读取互不影响；
- 分片合并、跨片同名报错（信息含两个文件名）、空目录、传入文件而非目录；
- 装载：同名 / 映射改名 / PyTorch 布局转置 / 转置用在非二维时报错 /
  多处错误一次报全 / 失败时 store 一个字节都不变 / 原地写（数组对象身份不变）/
  f16 源自动展宽 / f64 源要求 `allow_downcast` / 缺张量 / 多余张量回报 /
  `mapping` 与 `transpose` 的未知键 / 非 `ParameterStore` 入参。

**未声称 / 未验证。**

- 只支持 safetensors 容器；不读 GGUF、PyTorch `.bin` / `.pt`、TF 检查点。
- 不支持 fp4、稀疏张量、量化元数据（`*.scale` 之类按普通张量处理）。
- 不做名字的模糊匹配或按后缀猜映射；没有内置任何具体模型的转换表。
- 没有拿真实规模权重（GB 级）做实机加载耗时与内存峰值测量；按需读取的接口
  设计是为了让这件事可能，但数字还没有。

### 5.314　文本入口：读 HF 的 ByteLevel BPE

**为什么紧接着权重。** 5.313 解决了"权重进不来"，但真实模型还有第二道门：文本。
自带的 `ByteTokenizer`（`models/lm_text.py`）是字节级词表，双射干净，却没有
合并表——一个汉字 3 个 token，且与任何给定的 checkpoint 词表都对不上。权重装进
去了、分词器不对，模型只会安静地输出错位文本。这一项把 HF `tokenizer.json` 里
的 ByteLevel BPE 读进来，让"权重 + 词表"这对东西能一起落地。

**为什么不引 `tokenizers` / `transformers`。** 同 5.313 的三条：运行期依赖必须
只有 numpy；`tokenizer.json` 是自描述的 JSON，标准库够用；而"拒绝而不是近似"的
判据要自己定——通用实现默认支持一大家子算法，而我们只打算为读进来的那份词表
负责。

**支持面刻意收窄。** 只认 `model.type == "BPE"` 且词表覆盖 ByteLevel 的 256 个
字节映射（GPT-2 / Qwen2 / Qwen3 的形态）。另外承接：

- `merges` 的两种写法（`"a b"` 字符串与 `["a", "b"]` 数组），顺序即 rank；
- `added_tokens`（`special` 标记；`lstrip` / `rstrip` 显式拒绝，因为那会改变
  匹配时的文本边界）；
- `normalizer`：null / NFC / NFD / NFKC / NFKD 及其 `Sequence`；
- `pre_tokenizer`：null 或 `ByteLevel` 及其 `Sequence`，并**真正实现**
  `use_regex`（False 时整段不切）与 `add_prefix_space`（开头补一个空格）；
- `decoder`：null / `ByteLevel` / `Sequence`（`Replace` 不参与解码，ByteLevel
  已经能还原字节）。

拒绝 WordPiece / Unigram、SentencePiece 的 `byte_fallback` /
`end_of_word_suffix` / `continuing_subword_prefix`、`Metaspace`、以及 `Split`
预分词。`Split` 这一条最值得说：它的 `pattern` 会改切分边界（Llama-3 把数字按
1～3 位一组切），忽略 pattern 就是把别的分词器当成自己的用——拼出来的 id 每个
都"合法"，只是不是训练时的那串。

**预分词用 `unicodedata` 手写扫描。** GPT-2 的规则是
`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`。
Python 的 `re` 不支持 `\p{L}`，用 `[A-Za-z]` 替代会把中文、西里尔、希腊字母
全切错位；所以字母 / 数字用 `unicodedata.category` 的 `L*` / `N*` 前缀判断，
扫描器手写。两处容易写错的地方各有一条用例钉住：

- 前导空格并入后面的词：`"a a"` → `["a", " a"]`，第二个片段的字节映射以
  `Ġ` 开头；
- `\s+(?!\S)` 在空白串后面还有内容时只吃"除最后一个以外"的空白：`"  hello"`
  → `[" ", " hello"]`（而不是 `["  ", "hello"]`）。

**编码：特殊 token 不参与合并。** 先按 added token 的原文整体匹配——按首字符
建桶、桶内长的优先，避免对整份 added 表逐个 `startswith`——命中前后的普通文本
才走 BPE。否则 `<|im_start|>` 会被拆成一堆字节，而这些字节恰好也都在词表里，
不会报错。没有 added token 时走一条无桶的短路径。

**解码：先拼字节，再解 UTF-8。** 一个汉字在字节级词表里是 3 个 token，逐个解
会得到 3 个替换字符；实现里连续字节进同一个 `bytearray`，遇到 added token 或
序列结束才解码。added token 按原文输出（含非 `special` 的），不参与字节还原。
顺序上**先判 eos 再判 pad**：两者常常同 id（Qwen 的 `pad = <|endoftext|>`），
此时"停止"比"跳过"更接近原意。

**词表自洽在构造期查完。** 256 个字节字符必须齐全；id 不得重复（vocab 内部、
added 与 vocab 之间、added 与 added 之间都查——两个 added 共用 id 会让编码与
解码各挑一个，静默不对称）；`pad` / `eos` 必须在 `[0, vocab_size)`；
`vocab_size = max(id) + 1`。`spec()` 与 `LMSpec` 对齐，且不允许覆盖
`vocab_size` / `pad_id` / `eos_id`。

**验证。** 新增 53 条用例（`tests/test_lm_bpe.py`），全量 `3131 tests, OK`
（skipped 460）。词表用最小自造件（`id == 字节值` + 一个 `<|endoftext|>` +
若干合并 token），判据是 id 级而不是字符串级：

- 往返：ASCII / 中文 / emoji / 控制字符 / 空串；一个汉字恰好 3 个 id；
  按字节拼回 UTF-8；截断的 UTF-8 与非 ByteLevel token 显式报错；
- 合并：单对合并、rank 最小优先（`bc` 先于 `ab`，结果与从左到右不同）、
  换 rank 顺序结果随之改变、`merges` 数组写法、merges 与 vocab 不自洽时报错；
- 切分：前导空格附着、缩略后缀独立成片段（`it's`）、字母与数字不跨类合并、
  空白串的"留最后一个"与末尾整体、`use_regex=False` 整段不切、`add_prefix_space`、
  NFC 规整先于切分；
- added token：最长优先、整体成 id 不拆字节、解码回原文、`skip_special_tokens`、
  非 special 的 added token 按文本输出、特殊 token 两侧的文本仍走 BPE；
- 词表：`vocab_size` 覆盖最高 added id、`spec()` 与词表一致且拒绝覆盖、
  eos 从候选表推断 / 显式指定、未知 eos 与 pad 报错、pad == eos 时解码停止、
  pad != eos 时跳过；
- 拒绝面：`model.type` 非 BPE、三种 SentencePiece 开关、BPE dropout、
  `merges` 形态错、缺字节字符、空词表、id 重复（vocab / added×vocab / added×added）、
  pad / eos 越界、不支持的 normalizer / pre_tokenizer（含 `Metaspace` 与 `Split`）/
  decoder、`lstrip`、非法 added 项、顶层非对象；
- `from_file`：正常读取与坏 JSON。

**未声称 / 未验证。**

- 不解析 `tokenizer_config.json` 与 `special_tokens_map.json`：`eos` / `pad` 从
  added token 的候选名推断，或由调用方显式传入；chat 模板仍用 `ChatTemplate`
  手写，没有"读模板字符串"的能力。
- 不支持 Llama-3 的 `Split` 预分词、SentencePiece（Llama-2 系）、WordPiece
  （BERT 系）、Unigram（T5 系）、BPE dropout、`lstrip` / `rstrip`。
- 不做 offset mapping（char → token 的 span），因此没有"把生成结果定位回原文"
  的能力。
- 用自造小词表逐 id 对拍，没有与 HF 官方实现在真实 15 万级词表上做全量对拍，
  也没有分词吞吐数字。

### 5.315　非对称零点：把 int8 的另一半格子用起来

**为什么补这一项。** 5.308 / 5.309 的 int8 量化是对称的：``q = clip(rint(x/s),
-127, 127)``，0 恒映射到 0，用 254 个格子；非对称用满 ``-128..127`` 共 256 个，
零点可以不在 0 上。这不是精度游戏——ReLU 之后的激活**全为正**，对称量化会把
负半轴整片格子分给永远不会出现的值，等于白扔一半分辨率；非对称把值域平移到
``[-128, 127]``，同样的 8 位能多出一档有效精度。权重分布通常近对称，收益主要
在激活侧，所以真正值得走这条路的是 W8A8（``linear_i8a8_zp``）；weight-only 的
``linear_i8_zp`` 留给"导出模型里权重零点确实非零"的场景。

**为什么是新增算子而不是给现有算子加一个可选输入。** 与 5.312 把 LSQ 拆成
``fake_quantize_i8_lsq`` 同一条理由：可选的第三输入会让"这份 scale 到底配不配
零点"藏进运行时分支，构图期就无法拒绝；拆开之后，``quantize_i8`` 与
``quantize_i8_zp`` 各自是一份完整契约。零点以 **i8 张量**进图（不是 i32），
取值天然落在 ``[-128, 127]``，不需要"越界即拒绝"的哨兵。

**语义。** 四个算子（``ir/ops.py``，参考实现在同一处）：

- ``quantize_i8_zp(x, scale, zp) -> i8``：``clip(rint(x/s) + zp, -128, 127)``；
- ``dequantize_i8_zp(q, scale, zp) -> f32``：``(q - zp) * s``，减法在 **int32**
  里做（``-128 - 127 = -255``，i8 会回绕）；
- ``linear_i8_zp(x, qw, s, zp[, b])``：先反量化权重再 f32 GEMM；
- ``linear_i8a8_zp(xq, xs, xzp, wq, ws, wzp[, b])``：各自减零点后在 int32 里
  累加，再乘 ``xs * ws`` 并加 bias。

``quantize_i8_zp`` 的舍入顺序是**先舍入再平移**：``rint(x/s) + zp``。写成
``rint(x/s + zp)`` 在 ``x/s`` 不是整数时差一格，两者都"看起来合理"，所以有一条
用例专门钉住它。饱和端点也随语义改变：负端点是 ``-128``，不再是 ``-127``。

**值域必须先扩到包含 0。** 零点按 ``zp = -128 - lo/s`` 反解（``lo`` 落到
``-128``），只有在 ``lo <= 0 <= hi`` 时它才落在 i8 内。全正数据若直接用
``lo > 0``，零点会小于 ``-128`` 被夹死，正端则整片饱和——第一版就是这么写的，
``test_round_trip_error_is_bounded_by_half_a_step`` 直接测出 17 格误差。修正很
简单：先取 ``lo' = min(lo, 0)``、``hi' = max(hi, 0)`` 再算 ``scale =
(hi' - lo') / 255``。对 ReLU 激活 ``lo' = 0``、``zp = -128``，一格不浪费；这也
是"0 必须能被精确表示"这条约束的直接推论。代价是全正且远离 0 的数据要额外
覆盖 0 到最小值之间的一段空白，此时非对称并不比对称更优——这是它的边界，
不是 bug。

常数通道（``hi == lo``，扩展后只剩一个非零点或纯 0）没有值域可用，退化成
``scale = 1``；纯 0 通道仍是 ``zp = -128``、反量化为 0。常数非零通道在扩展后
反而能被**精确**表示（``255 * scale`` 恰好等于该常数），有一条用例钉住。

**W8A8 的 int32 上界变紧了。** 对称版 ``xq`` 与 ``wq`` 各自绝对值不超过 127，
``K * 127 * 127 <= 2**31 - 1``；非对称先减零点，差值最大 255，边界变成
``K * 255 * 255``，``_I32_ACC_LIMIT_ZP = (2**31 - 1) // 65025 = 33025``。超过就
在构图期拒绝，不给"跑到一半溢出"的机会。另一个算法上的硬约束：量化参数必须
能提到求和号外面，因此激活侧参数沿 K 必须恒为 1（每个 token 一组）、权重侧
秩不超过 1（每个输出通道一组）——沿 K 逐元素变化的 scale 无法用 int32 累加
表达，构图期显式拒绝。

**先减再乘，不在 f32 里补偏置。** ``(xq - xzp) @ (wq - wzp)`` 的展开式含
``-xzp*colsum(wq) - wzp*rowsum(xq) + K*xzp*wzp`` 三项；把对称版本的 acc 算完
再在 f32 里补这三项，需要额外两次归约，且与整数累加不逐位等价。参考实现选
前者，设备内核将来也应如此。

**反向。** ``linear_i8_zp`` 对激活与 bias 求导，``dx`` 用的必须是**带零点反
量化**的权重（用 ``q * s`` 会整体偏移一个常量）；``linear_i8a8_zp`` 只有 bias
可导，激活、权重与四组量化参数全部冻结——与 ``linear_i8a8`` 同口径。

**设备端：显式拒绝，暂不写内核。** 四个 ``*_zp`` 算子只进参考后端与非设备
后端；CUDA / HIP 的 ``GPU_OPS`` 不含它们，遇到即 ``BackendCapabilityError``。
这与 ``fft`` / ``rfft`` 只有 CUDA 承接、HIP 显式拒绝是同一套契约：能力声明
必须诚实，"主机侧已定义语义、设备端尚未承接"是合法的中间态，静默回退到 f32
才是错的。补内核的前提是能在真机上逐位对拍，当前没有可用 GPU 会话。

**验证。** 新增 21 条用例（``tests/test_quantize.py`` 的三个类），全量
``3152 tests, OK``（skipped 460）；ruff 通过。conformance 新增 4 个形态
（``quantize_i8_zp_channel`` / ``dequantize_i8_zp_channel`` /
``linear_i8_zp_bias`` / ``linear_i8a8_zp_bias``），总数 132 → 136：

- 语义：零点平移、先舍入再平移（0.5 → 1 而不是 2）、两端点饱和、``q - zp``
  走 int32、非正 scale 执行期拒绝、构图期 dtype / 秩 / 符号维拒绝、JSON 往返；
- 参数：手算两条通道的端点与零点、半步误差界、全零通道、常数通道精确表示、
  零点对跨零 / 全正 / 全负 / 常数都不越界、非对称在正激活上误差显著低于对称
  （实测 < 0.7×）、helper 拒绝面；
- 图：weight-only 与"反量化再 matmul"对拍、W8A8 与 int32 手算展开对拍、
  换零点改变结果、``K`` 超上限拒绝、构图期拒绝、两种反向。

**未声称 / 未验证。**

- 设备端没有非对称内核，也没有真机逐位对拍；本轮只有参考语义。
- 不做逐元素（per-K）零点：那会让 int32 累加失效，属于另一套算法
  （如 per-group 反量化的 weight-only）。
- 非对称参数只由离线工具（``quantize_weight_asymmetric`` /
  ``quantize_activation_asymmetric``）算，没有"运行时校准集搜索 min/max"的
  流程，也没有把 zero_point 写进 safetensors 量化元数据的格式约定（5.313 的
  读取器只认名字与布局映射）。
- 没有非对称的 QAT / LSQ 版本（假量化仍是 5.311 / 5.312 的对称口径）。

### 5.316　框架级对照补上整图推理

**缺口。** ``benchmarks/framework_compare.py`` 的对照面一直只有单算子
（``matmul`` / ``conv2d`` / ``maxpool``）与训练一步。整图推理——多种 kernel
交替、中间张量在设备上流转、整图调度——是实际用量最大的一块，却完全没有
torch 对照；``benchmarks/lm_inference.py`` 只测 AetherFlow 自己，给不出"和
主流框架比差多少"。

**做法。** 默认用例改为 ``matmul_1024`` / ``conv2d_3x3_pad1`` /
``transformer_block`` / ``cnn_lenet``；torch 侧新增
``_torch_transformer_block`` 与 ``_torch_cnn_lenet``，按
``benchmarks/cases.py`` 的结构**逐项重写算式**：

- pre-norm：``x + Attn(LN(x))``、``x1 + FFN(LN(x1))``，``eps = 1e-5``；
- 双向注意力：该用例构建时**没有因果掩码**，torch 侧也不加——加一个会让
  数值"看起来合理"但两边测的不是同一张图；
- ``heads = 4`` 与 ``BlockSpec(d_model=64, num_heads=4, d_ff=128)`` 绑定，
  刻意不用 ``scaled_dot_product_attention``：融合算子会把层间差异一起抹平，
  而这里要检验的正是"同样的层叠、不同的执行栈"；
- CNN 侧是两层 ``conv2d(3x3, pad=1) + relu + maxpool(2,2)`` 接全连接。

同一份权重先对拍再计时；对拍沿用脚本既有的 ``assert_allclose(rtol=atol=1e-4)``，
不通过就直接退出，不出数字。

**守卫。** ``tests/test_benchmark.py`` 增加
``test_torch_matches_whole_graph_cases``：在参考后端上把两条整图算式与 torch
对齐到 1e-4。用例结构（改 BlockSpec、换激活、动布局）一旦漂移，会先在单测
失败，而不是等某天有人拿基准报告下结论。

**本地结果（CPU，只为证明脚本能跑）。** 参考后端解释执行，必然比 torch 慢：

    transformer_block: 纯算 af 0.9605 / eager 0.3440 ms
    cnn_lenet        : 纯算 af 2.0551 / eager 0.4542 ms

这台机器没有可用 GPU 会话，**以上数字不能当性能结论**；它的价值是确认整图
算式在两边数值一致、计时路径可用。有卡时用
``python -m benchmarks.framework_compare --device cuda``（可加 ``--compile``）。

**未声称 / 未验证。**

- 没有 GPU 真机数字，也没有与 ``torch.compile`` 的整图对照结果（``--compile``
  在 CPU 上不具代表性）。
- 不含 KV cache 的逐步 decode：``benchmarks/lm_inference.py`` 测的是 prefill
  与 decode 的单步耗时，但没有 torch 对照版本，留给后续。
- 不覆盖数据加载、主机↔设备拷贝与多流重叠，口径与 5.287 一致。

### 5.317　量化权重的元数据约定与逐组反量化

**缺口。** 5.313 的权重入口只认 f32 张量；5.315 补齐非对称零点之后，
`quantize_weight` / `quantize_weight_asymmetric` 产出的 `(q, scale, zero_point)`
仍然没有"怎么打包进 safetensors、怎么读回来"的约定。两类问题：

1. **打包错误读不出来。** 非对称的包漏了零点、对称的包多带了零点、逐组的
   scale 被按逐通道读——三种都能读成合法数组，然后在推理里安静地给出错的数。
   safetensors 只保证"头里的字节数对得上"，它不知道 `scale` 是几个。
2. **逐组量化没有语义承接。** GPTQ / AWQ 的 `group_size = 128` 是当前最
   常见的 weight-only 形态，`dequantize_i8` 的 scale 只能是 `(1,)` 或 `(N,)`，
   接不了。

**做法：读取入口（主机侧）。** 新增 `QuantizedWeight`（frozen dataclass）、
`build_quantized_weight`（从名字→数组的表组装）与 `load_quantized_linear`
（从文件读）。校验集中在唯一的构造路径 `_check_quantized_arrays`：

- `q` 必须是 **int8**，二维 `(K, N)`。不做有损强转：非 int8 说明打包时漏了
  或错了一次转换，而 `300 → 44` 是静默回绕，不是报错。
- `transpose=True` 处理 PyTorch `nn.Linear` 的 `(out, in)`，与 5.313 的
  `load_into_store` 同一约定（布局不猜）。
- `scale` 必须是浮点，形状 `(1,)` / `(N,)` / `(K/G, N)`，列数对齐 N，组数整除
  K，值必须有限且严格为正；f64 → f32 有损，需要 `allow_downcast=True`。
- `zero_point` 必须是整数（int8 / int32 / int64 都接受，读出后转 i8），形状
  必须与 `scale` 相同，值域先校验再转换——float 零点说明它其实是中间量。
- metadata 的自描述一旦存在就必须与张量一致：
  `quantization ∈ {"int8", "int8_zp"}` 且必须与"有没有零点"对上；
  `quantization.group_size` 若写了，必须等于形状推出的 `K / K_g`。
  **缺失不校验**——别处导出的文件没有这套 metadata，按形状照样能校验。
- `dequantize()` 直接调 IR 的 reference（四选一：`dequantize_i8` /
  `dequantize_i8_zp` / `dequantize_i8_group` / `dequantize_i8_group_zp`），
  不在这里写第二份公式。

**做法：逐组反量化算子。** 新增 `dequantize_i8_group` 与
`dequantize_i8_group_zp`：`q (K, N)` 沿输入维分成 `K/G` 组，`scale` 与
`zero_point` 都是 `(K/G, N)`，组内共用一对参数；非对称仍然先转 int32 减
再乘。**组大小不入图**，由 `G = K / K_g` 唯一确定。

为什么不留 `group_size` 这个 attr：形状与 attr 表达同一个事实，两边一旦漂移
没人知道该信谁。`K` 与 `K_g` 都是静态维，infer 里直接校验整除，这已经是
唯一真值。退化情形也随之自动正确：`K_g = 1` 就是逐通道（单测与
`dequantize_i8` 逐位对拍），`K_g = K` 是逐行缩放。

**为什么不同时加 `linear_i8_group`。** weight-only 的实际用法是"反量化权重
再算"，`dequantize_i8_group` 接普通 `matmul` 已经闭合；把 `linear_i8` 的
`scale` 放宽到 `(K/G, N)` 会牵动所有后端内核与既有 infer 契约，换来的只是
省一次 f32 物化。等真机上有逐组 GEMM 的需求再说，不在没有设备对拍的情况
下改它的语义。

**验证。** 本地 3178 个用例（+25：8 条逐组算子含拒绝面与 JSON 往返、
10 条量化权重入口，其余是 5.316 的整图对照），460 跳过；
conformance 默认用例 136 → 138（`dequantize_i8_group` /
`dequantize_i8_group_zp` 两条新增形态，参考后端逐位通过）；
`K_g = 1` 与 `dequantize_i8` 逐位相同、`QuantizedWeight.dequantize()` 与
`dequantize_weight*` 逐位相同都有独立断言。

**未声称 / 未验证。**

- 设备端当时没有逐组内核：两个算子只有参考实现，`GPU_OPS` 未登记，在 CUDA / HIP
  上显式失败（与 5.315 的 `*_zp` 同一状态）；设备内核已在 5.321 补齐。
- 没有主机侧的逐组量化 helper（`quantize_weight_group` 之类）：本轮只补
  "读与反量化"这半边，写侧由导出器负责。
- 不声称兼容 GPTQ / AWQ 的原始检查点：它们还有 int32 交织打包、`qzeros`
  的 +1 偏移等约定，本轮只定义本仓库自己的命名与形状约定。
- metadata 不是强制格式：没有 metadata 的文件照旧读，语义由形状与显式参数
  决定；`check_metadata=False` 只跳过自描述检查，张量校验不可关闭。

### 5.318　fp8（e4m3 / e5m2）的参考语义

**缺口。** 5.317 之后，int8 的对称 / 非对称 / 逐组三族都有主机侧语义，但 fp8
完全没有：`DType` 里没有 f8，`cast` 造不出 fp8 张量，也没有位模式解码。
fp8 是当下低精度推理与训练最常提到的格式，缺口在语义层：不是"少一个快内核"，
而是"这套 IR 根本描述不了 fp8 张量"。

**做法：两个 DType + f32 载体。** 新增 `DType.F8E4M3` / `DType.F8E5M2`，
`np_dtype` 都是 `"float32"`，与 bf16 的载体方案同构：

- 参考后端借 f32 承接算术，每个 f8 边界由 `Executor.quantize()` 调
  `round_fp8` 落回格式格点；
- 不新增算子：`cast(dtype="f8e4m3")` 已经能表达"进入 fp8"，
  `pack_fp8` / `unpack_fp8` 负责单字节位模式；
- 不把 f8 做成 u8 语义算子：u8 只存位模式，算术前必须反量化，等于把
  "每个边界量化一次"从类型系统搬进每一个算子；bf16 先例已经证明载体方案
  的数值真值是唯一的（`aetherflow/runtime/fp8.py`）。

**两种格式的规则。**

| | 指数位 / 偏置 / 尾数 | 最小正规 | 最大有限值 | 溢出行为 | NaN 码 |
| --- | --- | --- | --- | --- | --- |
| e4m3 | 4 / 7 / 3 | `2^-6` | 448 | 饱和到 ±448 | 指数全 1 且尾数全 1（`0x7F`） |
| e5m2 | 5 / 15 / 2 | `2^-14` | 57344 | 按 RNE 到 ±inf | 指数全 1 且尾数非 0 |

- e4m3 采用 OCP 的 `E4M3FN`：**没有 inf**，只有 NaN，因此超出范围饱和而不是
  溢出；e5m2 遵循 IEEE，`0x7C` 是 inf、`0x7D` 起是 NaN，与 57344 的中点
  61440 按 RNE 走到 inf（无穷的尾数为 0，算偶数）。
- 舍入统一 RNE；次正规 ulp 固定 `2^(e_min - M)`（e4m3 最小 `2^-9`、e5m2
  最小 `2^-16`），更小的一半按 RNE 归零；`-0` 的符号位保留。
- `round_fp8(x) = unpack_fp8(pack_fp8(x))`：量化与编解码只有一份公式，
  返回值必然落在可表示位模式上。`STORAGE_BYTES = 1` 只是存储宽度声明，
  参考语义不承诺字节宽度（载体仍是 f32）。

**设备端显式拒绝。** `np_dtype` 对 f8 返回 f32，设备后端会把 f8 当 f32 算。
`DeviceBackend.invoke` 因此在"未实现算子"判定之后按 **IR 类型** 检查，
遇到 `F8E4M3` / `F8E5M2` 抛 `BackendCapabilityError`（CUDA 上 fp8 需要
`__nv_fp8_e4m3` 类型与专门的存取，HIP 各家不统一）。控制流的谓词白名单
（`ir/control.py`）不含 f8，控制流上同样显式拒绝。

**验证。** 本地 3194 个用例（较 5.317 的 3178 增加 16，集中在
`tests/test_fp8.py`）全绿；conformance 默认用例 138 → 140
（`cast_f8e4m3` / `cast_f8e5m2`，参考后端逐位通过）。独立断言覆盖：
已知编解码表（`1.0 → 0x38`、`448 → 0x7E`、`57344 → 0x7B` 等）、RNE 中点、
e4m3 饱和 / e5m2 溢出到 inf、次正规、±0 与 NaN；256 个位模式里非 NaN 的
全部 `pack ∘ unpack` 往返恒等；`pack` / `unpack` 对未知格式与非 uint8
位模式显式报错；图级 `cast` 的量化、f8 输入的入口量化、`add` 每个边界的
量化、JSON 往返语义不变；控制流谓词拒绝 f8。

**未声称 / 未验证。**

- 设备端没有 fp8 的存储容器与内核，也没有 fp8 GEMM；逐位对拍需要 GPU 真机，
  本轮不做（与 5.315 / 5.317 的设备端状态一致）。
- 没有 fp8 的权重量化 helper（`quantize_fp8_weight` 之类），也没有 per-tensor
  scale 的计算规则：本轮只定义数值语义与位模式编解码。
- 参考后端把 f8 参与的每个节点输出都量化一次，这不是 fp8 硬件的行为
  （硬件在累加器里保持更高精度）；它只作语义基准，性能与逐位对齐以设备端为准。
- 没有把 fp8 接进 `LMRunner` / 模型构建路径：`f8e4m3` / `f8e5m2` 目前需要
  调用方显式 `cast`。

### 5.319　fp8 权重从 safetensors 读入

**缺口。** 5.318 给了 fp8 的数值语义与位模式编解码，但权重读取入口（5.313）
仍把 `F8_E4M3` / `F8_E5M2` 显式拒绝：`tensor()` 直接报错。后果是量化到 fp8
的检查点连"读出来看一眼"都做不到，5.318 的 `unpack_fp8` 也没有出口。

**做法：读取入口解码成 f32 载体。** `SafeTensorsFile.tensor()` 对两种 fp8
dtype 走 `unpack_fp8`，返回 f32（fp8 -> f32 是精确的，与 bf16 的展宽同一
策略）。头里的项宽仍是 1 字节，偏移校验不变，只在取值时多一次解码。特殊值
原样保留：e4m3 的 `0x7F` 是 NaN（它没有 inf），e5m2 的 `0x7C` 是 inf，`-0`
的符号位也保留。

**为什么不返回 u8 位模式。** 位模式是"能跑但语义错"：调用方拿到 `0x38` 会
当成 56，而不是 1.0。需要位模式的调用方本来就该用 `pack_fp8` 自己打包，不必
让读取入口替它决定。返回 f32 也让 `load_into_store` 直接可用——f32 是参数
容器里最常见的 dtype，`can_cast` 不会误报有损。

**不猜 scale。** HF 的 fp8 检查点通常把权重与 `weight_scale` / `input_scale`
分开存。本轮只解权重本身的位模式，scale 张量按普通张量读出，如何应用由调用方
决定；5.317 的 `QuantizedWeight` 是 int8 容器，不承接 fp8 的 scale 约定。

**验证。** 本地 3195 个用例（较 5.318 增加 1：`test_weights.py` 的 fp8 读取，
覆盖 e4m3 的 448 / -0 与 e5m2 的 inf / NaN），460 跳过，全绿。原先断言
"fp8 显式拒绝"的用例改为解码断言。

**未声称。**

- 没有 per-channel 的 fp8 量化 helper，也没有 fp8 版 `QuantizedWeight` /
  `load_quantized_linear`；fp8 检查点的 scale 应用规则同样未定义。
- 设备端仍无 fp8 的存储与内核（5.318），读取入口只到主机侧 f32 载体。

### 5.320　计划期不预绑定 fp8 输出（真机 5.318 的漏洞修复）

**真机发现。** 5.318 声称"设备端遇到 fp8 显式拒绝"，本地用例也全部通过，
但在 Tesla V100-SXM2-32GB（`sm_70`）上跑 `verify_cuda.py` 时，`cast_f8e4m3`
与 `cast_f8e5m2` 是 **failed** 而不是 refused：

```
cases   : 130 executed / 8 refused / 2 failed
[failed  ] cast_f8e4m3  ops=cast  最大绝对误差 inf
[failed  ] cast_f8e5m2  ops=cast  最大绝对误差 inf
```

设备端没有拒绝，而是真的用 f32 的 cast 内核把 f32 载体原样算了过去。

**根因：两条路径只有一条带检查。** 逐节点的 `invoke` 会遍历
`(*node.inputs, node.output)` 判 fp8；但计划构建期的 `_planned_kernel`
（`device_plan`）只扫 `node.inputs` 的声明 dtype 决定是否预绑定内核。`cast`
的输入是 f32 载体、fp8 只出现在**输出**上，于是它被判为"可预判"、绑进
`plan.kernels`，运行期走批量派发快路径，整段绕开了 `invoke` 的类型检查。
本地参考后端与本地用例都只覆盖 `invoke` 与被墙掉的真机路径，因此没暴露。

**做法：计划期也判输出。** `_planned_kernel` 增加一条"节点输出类型是 fp8
即返回 `None`"，失败即退回逐节点 `invoke`，在原时机抛原来的
`BackendCapabilityError` 文案（不提前抛，符合 `_prepare_plan` 既有的
"预判只决定走哪条路，不改变报错时机"约定）。`_FP8_DTYPES` 从
`device_backend` 下沉到 `device_shared`（改名 `FP8_DTYPES`），两条路径共用
同一份集合，不再各写一处。

**验证。** 同一台 V100 上重跑 `verify_cuda.py`：`130 executed / 10 refused /
0 failed`，`result : PASS`，两条 fp8 用例都变成 refused 且文案为
"CUDA 后端还没有 fp8 的存储与内核……见 DESIGN 5.318"。本地新增 2 条纯 Python
用例：`cast` 到 f8e4m3 / f8e5m2 时 `_planned_kernel` 返回 `None`，而 f32 的
`cast` 仍被预绑定（防止一刀切关掉快路径）。本地 3197 个用例 / 460 跳过全绿，
ruff 通过；同机 `tests.test_fp8` 18 条在 3.10.12 + cupy 14.2.0 环境同样通过。

**未声称。** 这只修了"预绑定漏检"这一条路径；设备端仍没有 fp8 的存储容器、
内核与 GEMM（5.318 / 5.319 的限制不变）。

### 5.321　设备端逐组反量化内核（`dequantize_i8_group` / `_zp`）

**起点。** 5.317 给了逐组（GPTQ / AWQ 的 `group_size` 形态）weight-only 反量化
的参考语义与权重入口：q 形状 `(K, N)`、scale 形状 `(K/G, N)`，组大小由
`K / scale.shape[0]` 推出、不设 attr；但 CUDA / HIP 一直没有内核，一执行就
显式拒绝，这条路径在真机上不可用。

**语义复述。** 行 r 属于第 `r / G` 组，`out[r, c] = q[r, c] * s[r / G, c]`；
非对称版先扩到 int32 再减零点、再乘 scale：`(int32(q) - int32(zp)) * s`。
减法不落 int8 是为了避开 `-128 - 127` 的回绕，语义与参考实现逐位一致。

**两个内核、一份源码。** `device_shared` 新增 `_DEQUANTIZE_I8_GROUP_SRC` 与
`_DEQUANTIZE_I8_GROUP_ZP_SRC`：`extern "C" __global__`，扁平索引 `i`，
`row = i / cols`、`col = i - row * cols`、`si = (row / group) * cols + col`；
组大小与形状参数都按 `long long` 走、`np.int64` 传参，与既有 i8 内核同一套
约定。NVRTC 与 hipRTC 编译的是同一份源。

**注册面三处同步。** `GPU_OPS` 补两个名字（HIP 的 `HIP_OPS = GPU_OPS` 自动
继承）；`_INDEX_ARGS` 登记 `dequantize_i8_group -> {0}`、
`dequantize_i8_group_zp -> {0, 2}`（q 与 zero_point 是 i8 **数据**，scale 仍是
f32），`invoke` 的通用 dtype 检查只放行登记过的 i8 位置；`device_backend` 用
`r.register(...)` 绑到同名 `_op_*` 方法。`test_kernel_registry` 的"声明即实现"
反射检查继续保证 `CUDA_OPS` 与 `_op_*` 一一对应，并新增一条断言这三项登记值。

**形状边界。** `_group_shape` 在设备端再校验一次：q / scale 都是二维、列数
一致、组数大于 0 且整除 K；`_zp` 另外要求 `zp.shape == scale.shape`。报的是
`BackendCapabilityError`，不让内核越界读。

**验证。** V100（`sm_70` / 驱动 550.127.05 / cupy 14.2.0）上 `verify_cuda.py`：
`132 executed / 8 refused / 0 failed`，`result : PASS`，两条新形态逐位相同：

```
[executed] dequantize_i8_group     ops=dequantize_i8_group     err=0.0e+00
[executed] dequantize_i8_group_zp  ops=dequantize_i8_group_zp  err=0.0e+00
```

同机 `tests.test_cuda_backend` 382 条 4.5 秒全过（2 跳过）。本地新增一条注册表
用例；`tests.test_kernel_registry` 20 条（2 跳过）与 `tests.test_quantize`
96 条全过，全量 3198 个用例 / 460 跳过全绿，ruff 通过。

**实测注册表口径。** 本次 `len(OPS) == 93`、`len(GPU_OPS) == len(CUDA_OPS) == 87`。
未进 `GPU_OPS` 的 6 个是 `dequantize_i8_zp` / `quantize_i8_zp` /
`linear_i8_zp` / `linear_i8a8_zp`（非对称逐通道，仍显式拒绝）与
`fft` / `rfft`（CUDA 由 cuFFT 承接）。一页速览里的"80 个 IR 算子 / 承接 73 个"
是 5.312 起的增量口径，长期滞后（REMIND 维护者备注已记），本轮只按上一版
+2 顺延，没有统一重数。

**未声称。**

- 只有反量化本身，没有融合 GEMM（`linear_i8_group` 未做）：真机推理仍要先
  反量化成 f32 再走常规 `matmul`，中间张量照旧物化。
- 非对称**逐通道**的 4 个 `*_zp` 算子设备端仍无内核、显式拒绝（5.315）。
- 没有主机侧的逐组量化 helper，写侧仍由导出器负责（5.317）。

### 5.322　设备端非对称量化 / 反量化内核（`quantize_i8_zp` / `dequantize_i8_zp`）

**起点。** 5.315 给了非对称逐通道（zero_point）的主机侧语义与构图入口，
5.321 又补了逐组反量化，但最基础的这两个算子一直没有设备内核：`GPU_OPS`
不登记、一执行就 `BackendCapabilityError`（5.321 的"未声称"里仍列着）。
缺的是"能跑"，不是"能描述"——参考语义与 `linear_i8_zp` 的构图入口都已存在。

**语义复述。** 量化是 `clip(rint(x / scale) + zp, -128, 127)`：零点在**舍入
之后**加到整数格点上，不是值域里的浮点偏移（`x/scale` 非整数时两种写法差一个
舍入）。饱和下界用满 -128（对称版只用 -127）。反量化是
`f32((int32)q - (int32)zp) * scale`，减法留在 int32 里做，避开 `-128 - 127`
的回绕。

**两个内核、一份源码。** `device_shared` 新增 `_QUANTIZE_I8_ZP_SRC` 与
`_DEQUANTIZE_I8_ZP_SRC`：`extern "C" __global__`、扁平索引、`cols` / `per_row`
与既有 i8 内核同一套约定（`np.int64` 传参），`rintf` 也是同一份 RNE 舍入；
NVRTC 与 hipRTC 编译的都是这段源，HIP 侧不需要另写近似。

**参数归一。** 新增 `_zp_args`：scale 与 zero_point 都恰好是"最后一维长度 = N"
的一维张量时走 `s[i % cols]` / `zp[i % cols]` 的复用读法——这是逐通道量化的
常态，避免把参数物化成与数据同大的数组；其余广播形态（逐张量、逐元素、或两者
形态不一致）一次性 `broadcast_to` + `ascontiguousarray` 物化。取"任意广播都
算对"优先于省这一份临时量：量化是离线步骤，反量化的参数通常也只有 N 个。

**注册面三处同步。** `GPU_OPS` 补两个名字（HIP 的 `HIP_OPS = GPU_OPS` 自动
继承）；`_INDEX_ARGS` 登记 `quantize_i8_zp -> {2}`、`dequantize_i8_zp -> {0, 2}`
（q 与 zero_point 是 i8 **数据**，scale 仍是 f32）；`device_backend` 用
`r.register(...)` 绑到同名 `_op_*` 方法。`test_kernel_registry` 的反射检查与
新增断言自动覆盖这三项。

**验证。** V100（`sm_70` / 驱动 550.127.05 / cupy 14.2.0）上 `verify_cuda.py`：
`134 executed / 6 refused / 0 failed`，`result : PASS`，两条新形态逐位相同：

```
[executed] quantize_i8_zp    ops=quantize_i8_zp    err=0.0e+00
[executed] dequantize_i8_zp  ops=dequantize_i8_zp  err=0.0e+00
```

同机 `tests.test_cuda_backend` 382 条全过（2 跳过）。本地 `tests.test_quantize`
96 条与 `tests.test_kernel_registry` 20 条（2 跳过）全过，全量 3198 个用例 /
460 跳过全绿，ruff 通过。

**实测注册表口径。** `len(OPS) == 93`、`len(GPU_OPS) == len(CUDA_OPS) == 89`。
未进 `GPU_OPS` 的 4 个是 `linear_i8_zp` / `linear_i8a8_zp`（非对称的融合
GEMM，仍显式拒绝）与 `fft` / `rfft`（CUDA 由 cuFFT 承接）。

**未声称。**

- 只补量化 / 反量化本身：`linear_i8_zp` / `linear_i8a8_zp` 仍显式拒绝，
  非对称权重的推理要自己先 `dequantize_i8_zp` 再走常规 `matmul`；融合 GEMM
  没有做，中间 f32 张量照旧物化。
- 非逐通道的广播形态会物化 scale 与 zero_point（见"参数归一"的取舍），
  没有把逐张量零点单独做成常量参数。
- 没有新增主机侧 helper：写侧仍沿用 5.315 的入口，本轮只补设备执行。

### 5.323　设备端非对称融合线性层（`linear_i8_zp` / `linear_i8a8_zp`）

**起点。** 5.315 给了非对称零点的四个算子语义，5.322 补齐了量化 / 反量化两个
设备内核，但融合线性层 `linear_i8_zp` / `linear_i8a8_zp` 仍不在 `GPU_OPS` 里，
一执行就 `BackendCapabilityError`（5.322 的"未声称"第一条）。非对称权重的推理
只能自己先 `dequantize_i8_zp` 再走常规 `matmul`，中间 f32 张量照旧物化。

**`linear_i8_zp` 走组合，不写新内核。** 参考语义是
`weight = (q32 - zp32).f32() * scale`、`out = matmul(x, weight) (+ bias)`，
与对称版 `linear_i8` 的结构完全同形，只把反量化换成带零点那一支。所以
`_op_linear_i8_zp` 直接复用 5.322 的 `_op_dequantize_i8_zp`、既有 `_op_matmul`
与 `_binary("add")`：校验 `x` 是 f32、`x` 秩 >= 2、权重秩 2 之后按这条链拼装。
多写一个内核只会让"非对称"和"对称"两条路的数值口径分叉。

**`linear_i8a8_zp` 必须新内核。** 参考语义要求每项先减零点再相乘、在 int32 里
累加，`out = float(acc) * x_scale * w_scale (+ bias)`。这不是"对称 acc 再在 f32
里补一个偏置"能等价替换的：`sum((x-xz)*(w-wz)) = sum(x*w) - xz*sum(w) - wz*sum(x)
+ K*xz*wz`，右边每一项都依赖整行 / 整列的累加结果，而零点减法本身让
`(x-xz)`、`(w-wz)` 超出 i8 值域（`-128-127`），字节打包被打散，`__dp4a` 用不了。
所以 `_LINEAR_I8A8_ZP_SRC` 走标量乘加：一个线程算一个输出元素，`int acc` 逐项
累加后一次转 f32 乘两个 scale。这与 HIP 既有的 `_LINEAR_I8A8_SRC` 同一路线，
NVRTC 与 hipRTC 共用同一份源码，两条后端逐位一致——不像对称版那样 CUDA 走
`__dp4a`、HIP 走标量（两路虽然也逐位相同，但那是精确整数运算的巧合）。

**形状与广播的边界。** `x` 秩 >= 2、`w` 秩 2；`x_scale` / `x_zp` 必须恰为
`xq.shape[:-1] + (1,)`，`w_scale` / `w_zp` 的元素数必须等于 `N`（`(N,)` 或标量），
bias 若是 f32 且长度 `N`。这套严格度沿用既有 `linear_i8a8` 的口径，IR 允许的
逐元素广播形态在这里显式拒绝——把 `m` 行的 scale 物化成 `(m, n)` 是另一件事，
不在本轮。K 的上界由 IR 侧 `_I32_ACC_LIMIT_ZP`（`K*255*255` 不溢出 int32）保证，
内核侧不再重复检查。

**注册面三处同步。** `GPU_OPS` 补两个名字（HIP 的 `HIP_OPS = GPU_OPS` 自动继承）；
`_INDEX_ARGS` 登记 `linear_i8_zp -> {1, 3}`（`qweight` 与 `zp` 是 i8 数据，`x` /
`scale` 仍是浮点）、`linear_i8a8_zp -> {0, 2, 3, 5}`（`xq` / `xzp` / `wq` / `wzp`）；
`device_backend` 用 `r.register(...)` 绑到同名 `_op_*` 方法。
`test_kernel_registry` 的反射检查（`set(CUDA_OPS) == {_op_* 方法名}`）与新增的
两条 `_INDEX_ARGS` 断言一起覆盖这三项。

**验证。** V100（`sm_70` / 驱动 550.127.05 / cupy 14.2.0）上 `verify_cuda.py`：
`136 executed / 4 refused / 0 failed`，`result : PASS`。两条新形态：

```
[executed] linear_i8a8_zp_bias  ops=linear_i8a8_zp  err=0.0e+00
[executed] linear_i8_zp_bias    ops=linear_i8_zp    err=1.5e-05
```

`linear_i8_zp` 的 1.5e-05 来自 f32 GEMM 的累加顺序（参考实现用 numpy 的
`matmul`，cuBLAS 的分块顺序不同），在 1e-4 容差内；`linear_i8a8_zp` 全程整数
累加、只在最后转一次 f32，因此逐位相同。剩余 4 个 refused 是既有的显式拒绝路径
（`matmul_partial_batch`、`cast_f8e4m3`、`cast_f8e5m2`、`f64_input_relu`）。
同机 `tests.test_cuda_backend` 382 条 3.1 秒全过（2 跳过）。本地
`tests.test_quantize` + `tests.test_kernel_registry` 97 条全过（2 跳过），
全量 3198 个用例 / 460 跳过全绿，ruff 通过。

**实测注册表口径。** `len(OPS) == 93`、`len(GPU_OPS) == len(CUDA_OPS) == 91`。
未进 `GPU_OPS` 的只剩 `fft` / `rfft`（CUDA 由 cuFFT 承接，HIP 无 rocFFT 绑定）。

**未声称。**

- 非对称 W8A8 走标量乘加，没有 `__dp4a`、也没有 IMMA / wmma 分块：
  正确性已对齐，"int8 更省算力"仍不成立（与 5.309 对称版同一结论）。
- 只支持逐通道参数布局（`x_scale` / `x_zp` 为前导维 + `(1,)`、`w_scale` / `w_zp`
  为 `(N,)`），不支持 IR 允许的标量 / 逐元素广播。
- 没有反向规则（IR 侧 `linear_i8a8_zp` 只有 bias 可导），也没有 epilogue 融合：
  int32 累加后仍要物化一份 f32 输出。

### 5.324　设备端 matmul 支持批量维的部分广播

**起点。** 5.98 起设备端批量 GEMM 走 cuBLAS 的 strided batched，两个操作数各只有
一个标量 stride，因此只接受两种批量形状：与输出批量维完全一致（密集步长），
或整批只有 1 个 batch（步长 0，复用同一份数据）。`(2,1,4,8) @ (2,3,8,6)` 这种
部分广播一直被 `_batch_stride` 显式拒绝，conformance 里记为一个 `refused`
（5.322 / 5.323 的 `136 executed / 4 refused` 中的第一个）。

**为什么不能靠 stride 表达。** 部分广播的每个 batch 在左操作数上的偏移是
`(i * 0) * (m*k)` 与 `(j) * 0` 这样的组合，取决于哪个轴取 1；strided batched
的接口只给 A/B 各一个标量，没有"逐 batch 偏移表"。要正确表达只能换接口
（cuBLAS 没有）或者改变数据布局。

**方案：物化广播侧。** `_batch_operand` 替代 `_batch_stride`：形状一致 → 密集
步长；整体为 1（包括没有批量维）→ 步长 0；其余 → `broadcast_to` 出完整批次再
`ascontiguousarray`，然后给密集步长，复用同一条已测的批量路径。三条分支之外
不再有拒绝面。刻意不选"逐 batch 循环 launch"：那会把一次 batched GEMM 拆成
batch 次小 GEMM，多出 batch 次 launch 开销，而副本路径至少还留在一条连续的
批量调用里。

**代价。** 物化只在部分广播时发生，内存是 `batch × m × k` 的 f32/f16 副本；
常规批量与整体广播路径的代码与步长完全不变。没有加体积上限——能算出结果优先，
病态形状（例如 `(1,1,4096,4096)` 广播到 64 个 batch）会把选择权留给调用方，
这与其他"显式物化"的取舍一致（见 5.282 的参数广播）。

**验证。** V100 上 `verify_cuda.py`：`137 executed / 3 refused / 0 failed`，
`result : PASS`，`matmul_partial_batch` 首次成功执行：

```
[executed] matmul_partial_batch  ops=matmul  err=4.8e-07
```

4.8e-07 来自 f32 GEMM 的累加顺序，在 1e-4 容差内。剩余 3 个 refused 是
`cast_f8e4m3` / `cast_f8e5m2`（无 fp8 存储容器）与 `f64_input_relu`（f64 部分
承接）。同机 `tests.test_cuda_backend` 382 条全过（2 跳过），本地全量
3198 条 / 460 跳过全绿，ruff 通过。

**未声称。**

- 只补了批量维的广播；逐元素广播仍由 IR 在构图期拒绝（`_broadcast_shape` 的
  符号维规则不变）。
- 没有体积上限，也没有把副本缓存起来复用：每次调用各算各的。
- 只影响批量路径，非批量 GEMM 仍走 `sgemm` / `hgemm` / `bgemm`。

### 5.325　设备端 fp8 的 cast：把值落到格式格点

**缺口。** 5.318 给了 fp8 的参考语义与位模式编解码、5.319 让权重读得进来，
但设备端只有"显式拒绝"（5.320 修好预绑定漏检之后是干净的 `refused`）。
conformance 里 `cast_f8e4m3` / `cast_f8e5m2` 两条一直是 refused。

**载体仍是 f32，不省带宽。** 设备端没有 1 字节容器，`DType.F8E4M3` /
`F8E5M2` 的 `np_dtype` 都是 `float32`。因此"cast 到 fp8"在设备上不是类型
转换，而是**把数值舍入到该格式可表示的格点上**：`af_cast_fp8` 读 f32、写
f32，中间不出现单字节。想省带宽必须另做打包（`pack_fp8` 是主机侧 API），
这里不声称任何带宽收益。

**内核逐位复刻主机算法。** `runtime/fp8.py` 用 numpy float64 做两遍 RNE
（先在所属 binade 上舍入，跨 binade 的进位用新指数重算）再 encode / decode；
设备内核用 double 重写同一流程，`fabs` / `isnan` / `isinf` / `pow2` /
`ilog2` / `rint` 全部靠位操作与 union 实现，不依赖设备 libm。e4m3 饱和到
±448 且没有 inf，e5m2 按 IEEE 溢出到 ±inf，NaN / inf / ±0 都按 5.318 的
位模式处理。这样 `cast` 的 fp8 输出与参考后端逐位相等（真机 err=0.0）。

**只开 `cast` 一个口子。** 量化落在 `quantize` 上而不是新算子：输入装载、
节点输出边界、`_fast_step` 都走同一条路径，与参考的"边界量化"语义一致；
`_op_cast` 把 fp8 加进 `quantize` 分支（重复量化幂等）。与此同时：

- `invoke` 只在 `name == "cast"` 时放行 fp8，其它算子仍抛
  `BackendCapabilityError`；
- `_planned_kernel` 对 `cast` 的 fp8 输入 / 输出放行，其它算子返回 `None`
  （不预绑定，留给 `invoke` 报 refused）；
- `executor._fast_step` 的量化标记从 `BF16` 扩到
  `{BF16, F8E4M3, F8E5M2}`。漏这一步时，图输出是 fp8 的 `cast` 在快路径上
  会跳过最后的 `quantize`，真机表现为 `failed`（值停在 f32）而不是 refused。

**真机踩到的坑。** NVRTC 只接受 ASCII：内核源码里写中文注释会抛
`UnicodeEncodeError: 'ascii' codec can't encode characters`，被 conformance
记成 failed（而不是编译错误）。`_CAST_FP8_SRC` 的注释因此全部用英文，与
`device_shared.py` 里其它 `_SRC` 一致。

**验证。** 本地 `3199` 个用例 / 460 跳过全绿（较 5.324 增加 1：
`test_fp8.py` 把"fp8 输出不预绑定"改成两个方向——`cast` 的 f8e4m3 / f8e5m2
输出必须预绑定，`add(f8, f8)` 仍不预绑定），ruff 通过。V100 上
`verify_cuda.py`：`139 executed / 1 refused / 0 failed`，`result : PASS`，
两条 fp8 用例：

```
[executed] cast_f8e4m3  ops=cast  err=0.0e+00
[executed] cast_f8e5m2  ops=cast  err=0.0e+00
```

唯一剩下的 refused 是 `f64_input_relu`。同机 `tests.test_cuda_backend`
382 条全过（2 跳过）。

**未声称。**

- 不省带宽、不省显存，也没有 fp8 的存储容器；`f8` 张量在设备上仍是 f32。
- 没有 fp8 GEMM / attention 内核，也不支持 fp8 参与算术（`add(f8, f8)`
  仍 refused）。
- 参考后端把"每个节点输出"都量化一次，这不是 fp8 硬件的行为；两边一致
  的只是单节点 `cast` 的边界结果。

### 5.326　fp8 写侧：带 scale 的量化 helper 与权重容器

**缺口。** 5.318 定义了 fp8 的数值与位模式，5.319 让 fp8 的 safetensors
读得进来，5.325 让设备端能 `cast` 到 fp8。但整条链路缺"产生 fp8 量化结果"
的一侧：没有任何 helper 把 f32 权重 / 激活压到格点并给出 scale，也没有一个
容器把 `weight` / `weight_scale` / 可选的 `input_scale` 绑在一起做形状与
一致性校验。缺了它，5.319 的读取入口只能读别人导出好的文件，本仓库自己造
不出能被它接受的权重。

**带 scale 的量化只有一条公式。** `runtime/fp8.py` 新增
`quantize_fp8(x, scale, kind) = round_fp8(clip(x / scale, ±max))`，与 int8 的
`quantize_i8` 同构：

- 先除以 scale 把值域压进 `±max`（e4m3 是 448、e5m2 是 57344）；
- 落到格式格点，产物是**缩放后的 f8 载体**（数量级在 `±max` 内）；
- 反量化 `dequantize_fp8(q, scale) = q * scale`，与 `dequantize_i8` 同一口径。

`clip` 不是冗余：e4m3 在 encode 阶段自己饱和，但 e5m2 溢出会按 IEEE 舍到
`±inf`；量化权重里出现 inf 会让整条推理链变成 inf，所以两种格式统一先夹到
最大有限值。fp8 的 0 精确可表示，非对称量化没有收益，因此这里**不提供
zero_point**（与 `*_zp` 族明确区分）。

**scale 的口径与 int8 逐字对齐。** `models/quantize.py` 新增三个写侧入口：

- `quantize_weight_fp8(w, kind, per_tensor=False)`：`(K, N)` 权重逐输出通道
  算 `amax / max`，返回 `(w_f8, (N,))`；`per_tensor=True` 退化成 `(1,)`，就是
  HF fp8 检查点里的 `weight_scale` 形态。
- `quantize_activation_fp8(x, kind)`：按 token（最后一维）算 scale，返回
  `(x_f8, x.shape[:-1] + (1,))`，与 int8 的逐 token 版本同口径。
- `dequantize_weight_fp8(w, scale)`：`w * scale`。

全零的通道 / token 给 `scale = 1.0`：零在任何 scale 下都量化为零，但
`scale = 0` 会触发"非正 scale"拒绝，这个分支必须显式写出来。反过来，
`quantize_fp8` 自身拒绝非正 / 非有限的 scale，且在 `quantize_weight_fp8`
里先拒绝 NaN / Inf 的输入——量化会把一个坏值扩散到整条输出。

**容器多守一条 int8 不需要的规则：值必须在格点上。**
`weights.FP8Weight` 与 `QuantizedWeight` 的字段对齐（`shape` /
`in_features` / `out_features` / `group_size` / `per_group` / `dequantize`），
但载荷是 f32 载体，所以 dtype 证明不了"这份数组量化过"。一份原始 f32 权重
与一份量化后的 f8 载体在字节层面完全同型：不查，就会把"忘了量化"读成
"量化精度很高"，推理照跑、数值只差一点，没有任何症状。因此
`_check_fp8_arrays` 要求 `round_fp8(w, kind) == w` 逐位成立，否则直接报
"不在格点上，请先 quantize_weight_fp8"。

**格式由文件决定，不让人重复声明。** `load_fp8_linear` 的规则：

- 权重张量的 dtype 是 `F8_E4M3` / `F8_E5M2` 时 `kind` 可缺省，由 dtype 推出；
  显式给了就必须一致，矛盾直接报错。
- 权重存成 `F32`（已解码的载体）时无法从 dtype 推格式，必须显式给 `kind`。
- `weight_scale` 接受 `(1,)` / `(N,)` / `(K/G, N)`，有限且严格为正；逐组时
  要求 `K` 能被组数整除，`dequantize` 沿 K 重复 scale。
- `input_scale` 是可选张量，只接受标量（`(1,)`），对应静态激活 scale；
  动态 per-token 场景没有这个张量，缺省即不校验。
- `transpose=True` 读 PyTorch 的 `(N, K)` 布局，与本仓库 `(K, N)` 对齐。
- metadata 的 `quantization` 若存在，必须是 `fp8`（格式另定）/ `fp8_e4m3` /
  `fp8_e5m2`，与 `kind` 矛盾即报错；`quantization.group_size` 的校验与 int8
  共用同一个函数（同一条自描述必须与形状自洽的规则）。

**验证。** 本地 `3220` 个用例 / 460 跳过全绿（较 5.325 增加 21：
`test_quantize.py` 9 条钉住量化公式、逐通道 / 逐 token scale 与误差界；
`test_weights.py` 12 条钉住格式推断、格点拒绝、transpose、逐组 scale、
metadata 交叉校验与缺张量报错），ruff 通过。5.326 没有设备端改动，
V100 上的 `verify_cuda.py` 与 `tests.test_cuda_backend` 作为回归确认。

**未声称。**

- 不支持逐块 scale（`weight_block_size = [128, 128]` 这类二维块布局）；
  容器只认 `(1,)` / `(N,)` / `(K/G, N)`。
- 不提供 fp8 linear 算子——写侧只产出数组与 scale，怎么参与 GEMM 是 5.327。
- 不改变存储宽度：`w` 仍是 f32 载体，1 字节打包与传输是 5.328。

### 5.327　fp8 线性层（`linear_fp8` / `linear_fp8a8`）

**缺口。** 5.326 让本仓库能造出 fp8 权重，但没有任何算子消费它：设备端对
fp8 的白名单只有 `cast`（5.325），一张 `linear_fp8` 图在真机上会以
"不承接 fp8"被 refused。缺的是**语义定义**——"拿到 f8 载体和 scale 之后，
线性层到底算什么"。

**参考语义只有一条式子。** 新增两个算子，`infer` 负责类型与形状，参考实现
与 int8 族逐字对齐：

- `linear_fp8(x, qweight, weight_scale[, bias])`，
  `y = matmul(x, qweight.astype(f32) * weight_scale) + bias`；
- `linear_fp8a8(x_q, x_scale, w_q, w_scale[, bias])`，
  `y = matmul(x_q * x_scale, w_q * w_scale) + bias`。

因为 fp8 的载荷已经是"缩放后的 f8 载体"，反量化就是一次乘法，不需要解码
位模式，也不需要 zero_point（fp8 的 0 精确可表示）。这与 `linear_i8` 的
"反量化后走 f32 GEMM"同构，区别只在**累加**：`linear_i8a8` 在 int32 里累加、
scale 必须提到求和号外，因此 IR 要求激活 scale 沿 K 恒定；fp8 先落成 f32
张量、累加本来就在浮点，所以 `linear_fp8a8` 的 scale 只要按 numpy 规则
广播到载荷即可（IR 仍把形态收窄为逐 token / 逐输出通道，见下）。

**IR 把口径钉死，而不是留给后端猜。**

- `x` / `x_q` 必须是 `F32`（载荷的载体），秩 >= 2；权重必须是 fp8 且秩 2；
  两个 fp8 输入（`linear_fp8a8` 的两路载荷）**必须是同一种格式**——混合
  e4m3 / e5m2 没有定义好的反量化顺序，允许它只会产出"看起来能跑"的数。
- `weight_scale` 走 `_check_scale_broadcast`（标准 numpy 广播），因此接受
  `(1,)` 与 `(N,)`；`(K/G, N)` 逐组 scale 对 `(K, N)` 不广播，**构图期
  直接拒绝**。逐组是 `linear_i8_group` 那一批的事（5.329）：它需要"沿 K
  重复 scale 再乘"的语义，不能靠广播偷渡。
- 激活 scale 在 `linear_fp8a8` 里要求"沿 K 为 1"（`shape[-1] == 1`），
  外加可广播；这样 `(B, T, 1)` 与 `(1,)` 都合法，而 `(B, T, K)` 被拒——
  逐元素 scale 会让"量化"退化成"带噪声的恒等变换"。
- `bias` 必须是 `F32`、秩 1、长度 `N`。输出恒为 `F32`：fp8 只是输入侧
  的存储格式，结果不落回格点（要落回就显式 `cast`）。

**设备端不新增内核。** 反量化是 `_binary("mul")`（标量 / 尾维 / 逐 token
三种广播面都已支持），GEMM 复用既有 sgemm，bias 是 `_binary("add")`。收益
只在**驻留与读带宽**——fp8 权重按 f32 载体存时并不省显存（1 字节存储是
5.328），更不省算力：fp8 tensor core 需要另一套缩放与累加语义（scale 提到
求和号外、块级缩放、溢出处理），本版明确不承诺。

**白名单要同时覆盖两条执行路径。** 5.320 的教训是 `invoke` 与
`_planned_kernel` 的判定必须逐条对应：只在 `invoke` 里放行，预绑定的快路径
会绕开检查、把 fp8 载体当普通 f32 算掉。因此新增共享常量
`device_shared.FP8_OPS = {cast, linear_fp8, linear_fp8a8}`，两条路径都按它
判定；写成白名单而非"排除几个"，新算子默认仍不承接 fp8。

**设备端不复查格点。** `weights.load_fp8_linear` 已经逐位校验
`round_fp8(w) == w`；设备端再查一次要把权重读回主机，与整条设备路径的
"零同步"直接冲突。这条分工写进了内核注释：写入侧保证值在格点上，设备端
只负责按载体读数。

**验证。** 本地 `3228` 个用例 / 460 跳过全绿（较 5.326 增加 8：
`test_quantize.py::TestLinearFP8` 钉住"等于手写反量化再 matmul"、per-tensor
scale、e5m2 按自己的格点舍入、W8A8 的逐 token scale、非正 scale 拒绝、
7 类构图期拒绝与 JSON 往返），ruff 通过；conformance 新增 4 条
（`linear_fp8_e4m3_bias` / `linear_fp8_e5m2_bias` / `linear_fp8a8_e4m3_bias` /
`linear_fp8a8_e5m2_rank2`），参考后端全通过。

**未声称。**

- 不承诺 fp8 tensor core：算力与等价的 f32 GEMM 相同。
- 不省显存 / 带宽：载荷仍是 f32 载体，1 字节打包是 5.328。
- 不支持逐组 scale；`linear_fp8` 只有 weight-only 与 W8A8 两种形态，
  没有 zero_point 变体。
- 没有反向规则（与 int8 线性层一致，量化路径按推理用）。

### 5.328 fp8 的 1 字节存储：pack / unpack 与融合的 packed 线性层

**缺口。** 5.325 的 `cast(f8e4m3)` 与 5.327 的两个 fp8 线性层都只把数值
落到格式格点上，容器仍是 f32：4 字节/权重。也就是说 fp8 目前只是"数值
精度变低"，显存驻留与读带宽一点没降——而后者才是 fp8 权重的主要动机。

**为什么不能直接让后端换容器。** IR 里 `DType.F8E4M3` / `F8E5M2` 的
`np_dtype` 都是 `float32`（5.318 起就是这么定的），参考语义、序列化、
verifier 全按 f32 载体推理。把 dtype 改成 1 字节会连带改掉整条链路的
语义，且 numpy 没有 fp8 标量类型，参考语义仍得绕回 f32。所以打包是一个
**显式的表示转换**，不是 dtype 的重新定义。

**容器取 i8 位模式。** IR 没有 `u8`，而 `int8` 正好是 8 位、有符号性不影响
位模式。于是新增三个算子：

- `pack_fp8(x, kind) -> i8`：f32 载体 → 1 字节码。参考语义是
  `runtime.fp8.pack_fp8(x, kind).view(np.int8)`——纯粹的位拷贝，返回的
  i8 **不代表整数**，`-128` 就是码 `0x80`。
- `unpack_fp8(q, kind) -> f8 载体`：i8 码 → f32 载体。先把 i8 按位
  `view(np.uint8)`，再 `unpack_fp8`，避免负码在升位时被符号扩展。
- `linear_fp8_packed(x, q_i8, weight_scale[, bias], kind) -> f32`：直接吃
  1 字节权重。参考语义是 `matmul(x, unpack(q) * scale) [+ bias]`，即"先
  解码再走浮点 GEMM"——设备端可以自己决定怎么实现，语义只有这一条。

`kind` 走 attr（`attrs_schema=("kind",)`），取值 `{\"e4m3\", \"e5m2\"}`；
缺省或拼错在构图期报 `ValueError`，不是静默按 e4m3 处理。

**设备端复用同一份编解码。** 5.325 的 `af_cast_fp8` 里那套 RNE + 饱和 +
e4m3 无 inf 的逻辑被抽成 `af_fp8_encode(float, int) -> int` 与
`af_fp8_decode(int, int) -> float`（`_FP8_COMMON_SRC`）。`af_cast_fp8`
改成 `decode(encode(x))`，逐行搬迁、算法未改；`af_pack_fp8` 直接存
`encode` 的字节，`af_unpack_fp8` 读字节后 `decode`。因此 **pack → unpack
恒等于 cast**，三者共享同一份舍入与饱和规则，不会出现"打包路径和 cast
路径在边界值上不一致"。

`af_linear_fp8_packed` 照 `_LINEAR_I8A8_SRC` 的朴素结构：每线程算一个
输出元素，权重按 `unsigned char` 读、在寄存器里解码、乘 weight scale、
f32 顺序累加。`ws_stride=0` 表示逐张量 scale，`1` 表示逐列——两种都只走
`ws[0]` / `ws[col]` 一次，内层循环不重复读 scale。这里刻意不接 cuBLAS：
V100 是 sm_70，没有 fp8 tensor core，也没有能吃 1 字节权重的 GEMM 入口。
**这条路径只承诺存储与读带宽下降，不承诺速度**——解码是每元素几次整数
位运算，算力与 f32 GEMM 持平甚至略慢。

**scale / bias 的边界。** `linear_fp8_packed` 的 scale 只接 `(1,)` 或
`(N,)`（rank ≤ 1），逐组 scale 在构图期显式拒绝并留到 5.329——把它接到
这个内核上要么静默只用第一组、要么退化成逐元素分支，两者都不该发生。
bias 必须是 f32、长度 `N`；非正 scale 在参考语义与设备端都报错（构图期
只能看到形状，看不到值）。设备端还要求 x 是 f32 且秩 ≥ 2、权重是 i8 秩 2。

**白名单与索引参数。** `device_shared.FP8_OPS` 只放 `cast` /
`linear_fp8` / `linear_fp8a8` / `pack_fp8` / `unpack_fp8`——
`linear_fp8_packed` 不在其中，因为它的输出 dtype 是 f32、图里根本没有 fp8
类型，不属于"fp8 dtype 算子"这一族，放进白名单会让"未声明 fp8 却执行"
这条检查失去意义。`_INDEX_ARGS` 新增 `unpack_fp8: {0}` 与
`linear_fp8_packed: {1}`：前者的输入是 1 字节码，后者的权重是 1 字节码，
都不能套用 f32 的字节数推断。

**主机侧入口。** `FP8Weight.pack()` 返回 1 字节副本（`w` 变成 i8 码、
`packed=True`），幂等；`dequantize()` 在 packed 时先解码再乘 scale，所以
打包前后数值逐位相同。`load_fp8_linear(..., packed=True)` 要求权重 dtype
是 `F8_E4M3` / `F8_E5M2` / `I8` / `U8`，字节用新增的
`SafeTensorsFile.raw()` **原样**读出：不能走 `tensor()` 解码再编码——数值
虽等价，但 e4m3 的 NaN 有 `0x7F` / `0xFF` 两个码，往返会归一化成一个，
位模式不再是文件里的那一份。`I8` / `U8` 推不出格式，`kind` 必须显式给；
`F32` 载体用 `packed=True` 读会直接报错并提示改用 `packed=False`。

**验证。** 本地 3241 条通过 / 460 跳过（较 5.327 增 13），ruff 通过。
`test_quantize` 新增 9 条：pack / unpack 的码逐位对拍、pack→unpack 恒等
cast、融合线性层与手写 `matmul(unpack*scale)` 对拍（含 bias、逐张量
e5m2）、8 类形状 / dtype 拒绝、非正 scale、JSON 往返。
`test_weights` 新增 5 条：`pack()` 的字节数与码、逐组 scale 打包后
`dequantize` 不变、`packed=True` 读原始字节与 I8 容器需显式 kind、
F32 载体与三维权重的拒绝、PyTorch 转置布局的打包读取。conformance 新增
6 条（`pack_fp8_e4m3` / `pack_fp8_e5m2` / `unpack_fp8_e4m3` /
`unpack_fp8_e5m2` / `linear_fp8_packed_e4m3_bias` /
`linear_fp8_packed_e5m2`），参考后端全通过。

**未声称。**

- 不省算力：没有 fp8 tensor core，解码 + f32 累加比原生 GEMM 慢。
- 不支持逐组 scale、也不支持 W8A8 的 packed 激活打包（`linear_fp8_packed`
  只打包权重；激活打包留到 5.329）。
- 没有反向规则（量化权重按冻结常量处理，与 int8 / fp8 线性层一致）。
- V100 真机回归（`verify_cuda.py`、`tests.test_cuda_backend`、pack /
  unpack / packed linear 的 conformance）仍待授权执行；设备端只做了源码级
  检查，未经编译验证。

### 5.329 W8A8 的五个缺口：scale 反向、逐组量化、动态 scale、QAT、zp epilogue

**缺口。** 5.309 的 `linear_i8a8` 只定义了前向：`vjp` 缺失，`backward` 直接
报"没有规则"；逐组量化只有主机侧容器（5.317）与反量化算子（5.321），没有
融合 GEMM；激活 scale 必须离线校准后作为图输入；QAT 只有裸的假量化算子，
没有能直接搭层的入口；`linear_i8a8_zp` 的 bias 要在内核之外再发一次 `add`。

**1. W8A8 的反向只对两路 scale。** `out = (x_q * x_s) @ (w_q * w_s) + b` 里
唯一连续的量是 `x_s` / `w_s`。int8 码继续按冻结常量处理：量化是阶跃函数，
几乎处处导数为 0，给码返回一个伪造梯度等于把 STE 偷偷塞进部署路径；训练期
要不要直通，由显式的 QAT 构建器决定（见第 4 点）。梯度推导与手写一致：

```
dact    = dy @ W^T            W = w_q * w_s
dweight = X^T @ dy            X = x_q * x_s
dx_s    = sum_k(dact * x_q)   # 归约到 (…, 1)
dw_s    = sum_m(dweight * w_q) # 归约到 (N,)
db      = sum over leading dims dy
```

实现用 `cast` 把 i8 码升到 f32、`mul` 乘两路 scale、`sum_to` 归约到 scale
的声明形状，再沿前导维求和送 bias。`fake_quantize_i8_lsq`（5.312）之所以能
把梯度送给 scale，是因为它的前向在 f32 上做"乘 scale → 舍入 → 除 scale"，
scale 参与连续运算；`linear_i8a8` 的 scale 只在 epilogue 乘一次，形式不同，
但归约目标相同。

**2. 逐组量化：`linear_i8_group`。** 签名是
`(x: f32 (…, K), q_i8: i8 (K, N), scale: f32 (K/G, N)[, bias]) -> f32`。
参考语义只有一条：

```python
weight = q.astype(np.float32) * np.repeat(scale, K // G, axis=0)
out = np.matmul(x, weight) [+ bias]
```

`infer` 复用 `_check_group_params`（5.321 为 `dequantize_i8_group` 写的
校验）：`K` 与 `K/G` 必须静态且整除、列数等于 `N`。`vjp` 镜像 `linear_i8`，
只是权重换成 `dequantize_i8_group`，所以 `dx` 与 `db` 的口径与逐通道版本
逐位相同。

设备端 `_op_linear_i8_group` 也是镜像：`_op_dequantize_i8_group` →
`_op_matmul` → 可选 `_binary("add")`。**没有专用内核**——逐组 scale 在 K
方向变化，无法提到整数累加外面（这正是 `linear_i8a8` 拒绝逐元素 scale 的
同一条理由），所以它保持 weight-only 形态：反量化后走 f32 GEMM。

主机侧 `quantize_weight_group(weight, group_size)` 每 `G` 行取 abs-max / 127，
`scale` 形状 `(K/G, N)`，全零组给 1.0；组大小必须整除 `K`。`__all__` 与
`build_int8_group_linear` 只是把这条路径接到 IR 上。

**3. per-token 动态 scale：`build_int8a8_linear_dynamic`。** 激活 scale 不再
是图输入，而是在图里现算：

```
absx    = where(x >= 0, x, neg(x))
amax    = reduce_max(absx, axis=-1, keepdims=True)
x_scale = where(amax > 0, amax / 127, 1)
x_q     = quantize_i8(x, x_scale)
out     = linear_i8a8(x_q, x_scale, w_q, w_scale[, bias])
```

全零 token 的 scale 必须是 1 而不是 0：`quantize_i8` 在前向与设备端都拒绝
非正 scale。整条链只由既有算子组成，**没有新增内核**；量化格点与主机侧的
`quantize_activation` 逐位相同（测试用 `assert_array_equal` 对拍）。

**4. QAT：`build_int8a8_linear_qat`。** 输入全是 f32
（`x` / `x_scale` / `w` / `w_scale`[ / `bias`]），前向是
`matmul(fake(x, x_scale), fake(w, w_scale)) + bias`，图里**没有 i8 张量**。
`learn_scale=True` 时换成 `fake_quantize_i8_lsq`，量化步长本身可学；否则是
标准 STE，`x_scale` / `w_scale` 求导显式报 `NoGradientError`（不是返回零）。

"前向格点与部署一致"是有限的承诺：假量化与 `quantize_weight` /
`quantize_activation` 共用同一份 RNE 与饱和，因此导出权重得到的 int8 码与
训练时逐位相同；GEMM 的累加顺序不同（训练走 f32 matmul，部署走 int32
累加），末位可能差 1 ulp。

**5. `linear_i8a8_zp` 的 epilogue 融合，dp4a 仍延后。** bias 直接进内核：
`af_linear_i8a8_zp_bias` 把 `(float)acc * xs * ws + bias` 一次写完，省掉一次
launch 与一块 `m*n` 中间张量。两次乘法用 `__fmul_rn`、加法用 `__fadd_rn`：
不融合时 bias 在另一颗内核里做 f32 加法、各自按 RN 舍入，若让编译器发 FFMA，
末位会差 1 ulp，"融合前后逐位相同"就不成立。无 bias 入口与融合入口共用同一个
`__device__ af_i8a8_zp_acc`，累加体只有一份。

dp4a / IMMA 仍不做。零点相减后的差不是 int8，`__dp4a` 吃不下；要复用整数
点积只能展开成 `P - zx*Sw - zw*Sx + K*zx*zw`（`Sw` / `Sx` 是逐行 / 逐列
的和）。展开在数学上等价，但要额外做两次轴归约、并重新论证 `P` 的 int32
上界，得等有真机能对拍时再动，避免"源码看起来更快的版本无法验证"。

**验证。** 本地 3262 条通过 / 460 跳过（较 5.328 增 21），ruff 通过。
`test_quantize` 新增 21 条：`linear_i8_group` 的数值 / 组数=K 退化 / 反向 /
JSON 往返 / 8 类拒绝；`quantize_weight_group` 的形状、误差界、逐组优于逐通道、
全零组、5 类拒绝；`build_int8_group_linear` 构图对拍与参数拒绝；QAT 前向格点、
STE 无 scale 梯度、LSQ 有 scale 梯度、参数拒绝；动态 scale 与主机口径逐位
对拍、per-token 而非 per-tensor、无 bias + JSON 往返、参数拒绝。
conformance 新增 1 条 `linear_i8_group_bias`，参考后端通过。

**未声称。**

- `linear_i8_group` 在设备端没有专用内核，逐组 scale 只是让**权重更细**，
  不省算力，也不支持逐组 scale 的 W8A8。
- `linear_i8a8_zp` 仍是标量乘加，没有 dp4a / IMMA。
- fp8 的逐组 scale 与 packed 激活（5.328 遗留）仍未做，留作独立阶段。
- 新增 op 的 V100 真机 conformance 仍待授权；设备端只做了源码级检查。

### 5.330 bool 张量算子：逻辑组合、bool 归约与 cast 的双向护栏

**缺口。** bool 在 5.301 已经能存储、能比较、能被 `where` 消费，但三处没闭合：

1. **掩码只能"产生"不能"组合"。** `gt` / `lt` 各自出一张掩码，要表达
   "大于 0 且小于 1" 只能退回浮点算术（`mul(gt(...), lt(...))`），既绕开了
   判定语义，也把 bool 又送进了算术路径。
2. **归约两端不一致。** `_infer_reduce` 原先允许 bool，参考端 `np.sum` 把
   true 当 1 累加，结果是**计数**；设备端却拒绝。"同一个图在两个后端一个能跑
   一个报错"是最不该留下的状态。
3. **设备端 cast 不接 bool。** `cast` 的 `dst_key` 只有 f32 / f16 / bf16，
   bool 与浮点之间没有桥，量化掩码、导出标签都得在主机上倒一手。

**1. `logical_and` / `logical_or` / `logical_not`。** 语义就是 numpy 的那三个：
两侧必须是 bool，输出 bool，前两个走 numpy 广播。`infer` 在浮点输入上直接
报错并指出正确写法（`ne(x, 0)`），不替调用方做"非零即真"的隐式转换——那会
让 `logical_and(x, y)` 与 `logical_and(ne(x,0), ne(y,0))` 看起来等价，实际
在 NaN 与负数上不是一回事。

设备端**没有新增内核**：`logical_and` / `logical_or` 复用比较模板，`COMPARE_OP`
加 `land` / `lor` 两个符号（bool 操作数经 `af_ld_q` 取出就是 0/1 的 float，
C 的 `&&` / `||` 在 0/1 上的真值语义与 numpy 一致），`logical_not` 复用一元
模板加 `lnot` 表达式（`x != 0 ? 0 : 1`）。结果写回必须走 raw store，否则
store helper 会按 float 语义写 1 字节以外的宽度；因此 CUDA / HIP 的 `render`
改成**按 dtype 自动选方言**：bool 用 `BOOL_DIALECT`，其余用原方言。这样
所有调用点都不必各自记着传方言。

**2. `reduce_all` / `reduce_any`。** 判定掩码的正确归约是 all / any，不是
sum。两者只接受 bool、输出仍是 bool（降秩规则、`axis` 支持整数 / 元组 /
负数、`keepdims` 与 `reduce_*` 完全同口径），空轴的恒等值分别是 true / false。
参考语义直接落在 `np.all` / `np.any` 上。

**3. `reduce_sum(bool)` 改为显式拒绝。** 名字承诺的是求和，bool 求和实际是
计数——`reduce_sum(mask)` 在 numpy 里能跑，会让调用方以为拿到了"这些元素
之和"，读代码的人也不会想到它是 `count_nonzero`。所以 `_infer_reduce` 在
入口拒绝 bool，错误信息直接给出两条替代路径：判定用 `reduce_all` /
`reduce_any`，计数先 `cast` 到 f32 再求和。`reduce_max` / `reduce_mean` 共用
同一入口，一并拒绝（max 在 bool 上退化成 OR、mean 退化成比例，同样不是名字
承诺的语义）。

**4. `cast` 的 bool 双向。** 设备端补 `"bool"` 的载体键（1 字节
`unsigned char`）：目标是 bool 时用 `src != 0`，从 bool 出发时统一先升到
f32、再走既有的 `_cast_kernel`。f16 / bf16 必须经 f32 中转——直接和
`__half` 互相构造会触发"Unresolved extern"，而 bf16 的 uint16 载体按数值
读会被当成位模式。判据只有一条：**非零即真**，与 numpy 的 `astype(bool)` 一致。

**5. 反向护栏。** 逻辑算子与 bool 归约没有 vjp（输出是判定值，导数无处可
落），`backward` 走既有的 `NoGradientError`。另外两处原本会**静默**给出
无意义梯度，现在显式报错：`reduce_sum` 的输入是 bool 时抛 `ValueError`
（否则反向会把 bool 梯度原样传回去），`cast` 的源或目标是 bool 时也抛
`ValueError`（掩码不该进损失；要在可导路径上用它，得在 cast 之前把这条
支路摘掉）。

**验证。** 本地 3275 条通过 / 460 跳过（较 5.329 增 13），ruff 通过。算子
99 → 104；CUDA 承接 99 → 102（`GPU_OPS` 97 → 100，`reduce_all` /
`reduce_any` 不在其中），HIP 97 → 100。`test_ops_extra` 新增 13 条：
`TestLogical` 6 条（同形 / 广播对拍 numpy、`logical_not`、浮点输入拒绝、
不可导、`gt`→`logical_and`→`where` 的掩码链对拍）、`TestReduceBool` 5 条
（两种形状 × 两轴 × keepdims 两态对拍、空轴恒等值、`reduce_sum(bool)` 拒绝
且消息含 `reduce_all`、浮点输入拒绝、不可导）、`TestBoolCast` 2 条
（f32↔bool 往返对拍、从 bool 求导抛 `ValueError`）。conformance 新增 3 条
逻辑用例（`logical_and_same` / `logical_or_broadcast` / `logical_not`）与
2 条 bool 归约用例（`reduce_all_last` / `reduce_any_axis0`），参考后端通过；
bool 输出只能精确相等，这五条的 rtol / atol 都是 0。

**未声称。**

- 设备端**没有** bool 归约内核（`af_add_q` 仍是 stub），`reduce_all` /
  `reduce_any` 不在 `GPU_OPS` 里：CUDA / HIP 会显式拒绝，只有参考后端承接。
- 没有 `count_nonzero`；计数必须先 `cast` 到 f32。这是有意的，不把计数
  伪装成 `reduce_sum`。
- 逻辑算子没有导数的替身（不做 STE）：判定值不进损失是使用方的责任。
- 前端 `torch.export` 的 `aten.logical_and` / `aten.logical_or` /
  `aten.logical_not` 与 bool 的 `to(dtype=...)` 尚未映射到这三个算子；
  IR 有语义，导入侧仍显式拒绝。
- 新增设备端路径（bool cast、逻辑组合的方言选择）只有源码级检查与参考
  后端对拍，V100 真机回归仍待授权。

### 5.331 DDP 去掉主机中转：设备侧 communicator 成为默认路径

**缺口。** 设备侧集合通信在 5.228 / 5.244 已经能加载 NCCL / RCCL、能建
communicator，但数据并行这条路仍要求调用方自己把 provider 拼好：

1. `collective_support` 的声明是"库已接入但路径没接"，`device_allreduce`
   在未注入 provider 时恒为 False——能力存在却声称没有。
2. `DataParallelTrainer(collective="device")` 没有注入
   `device_collective` 时直接抛 `DeviceCollectiveUnavailable`，等于把
   "建 communicator" 这件事推回给调用方。
3. `DistributedDataParallel.from_env` 缺省用 `HostCollective.from_env()`：
   设备态的梯度本来就在显存里，却先 D2H、主机求和、再 H2D，是反常的那一侧。

**1. 能力声明的判据改为"库能不能加载"。** `collective_support` 不再假设
"路径没接"：`_device_collective_reason()` 尝试 `nccl.vendor()` /
`version_string()`，成功即声明 `device_allreduce=True`、`provider="nccl"`，
失败才声明只能主机中转。理由文本按真实厂商命名（HIP 上是 RCCL），并写明
两条默认路径各自的构造入口。`overlap` 仍恒为 False（执行器同步、切卡串行，
没有可重叠的窗口），不随这次改动变化。

**2. `DataParallelTrainer` 默认 `collective="auto"`。** 解析顺序：注入
provider ⇒ `device`；否则查 `collective_support(backend).device_allreduce`
⇒ `device` / `host`。解析成 `device` 且没有 provider 时，构造期直接调
`nccl_device_collective(backend, picked)`（`ncclCommInitAll`），不再要求
调用方注入。显式 `collective="device"` 但库缺失仍抛
`DeviceCollectiveUnavailable`，消息带原因并给出 `collective="host"` 的
出口——**不静默降级**，否则调用方会以为通信留在设备上。

**3. `NcclDeviceCollective.from_env`：一进程一卡自建 communicator。**
rank 0 生成 `ncclUniqueId`，用 `UniqueIdServer` 在 `AF_MASTER_ADDR` /
`AF_MASTER_PORT` 上分发，其余 rank `fetch_unique_id` 取回后
`ncclCommInitRank`。协议与 `train.distributed.spawn` /
`host_collective.launch` 同源（`AF_RANK` / `AF_WORLD_SIZE` /
`AF_MASTER_ADDR` / `AF_MASTER_PORT`），所以用这两个启动器或 `torchrun`
拉起的进程直接可用。rank 0 先挂分发线程再初始化自己：`init_rank` 会阻塞
到全员到齐，顺序反过来会让其余 rank 在 fetch 上白等到超时。
`world_size == 1` 返回 `_LocalComm`（归约恒等），单卡调试不需要装 NCCL。

**4. `nccl.init_rank`：切卡与初始化合并。** 先 `cp.cuda.Device(target)`
再 `ncclCommInitRank`，`device` 缺省取 rank。分开写的话，"comm 绑错卡"
要到第一次集合调用才炸，错误现场离配置错误很远。没有 cupy 时抛
`NcclUnavailable`——无法确定 communicator 绑在哪张卡上，不能猜。

**5. `DistributedDataParallel.from_env` 默认跟随 `trainer.device_state`。**
`device=None` 时设备态走 `NcclDeviceCollective.from_env`，主机态走
`HostCollective.from_env`；显式 `device=False` 保留主机中转（跨机且没有
NCCL 时唯一可用的路）。桶与线程数的缺省值也随形态走：设备态
`bucket_bytes=0` / `threads=1`（NCCL 调用必须各 rank 同序，设备态不拼桶），
主机态 `DEFAULT_BUCKET_BYTES` / 2。`NcclUnavailable` 被转成
`DeviceCollectiveUnavailable`，消息提示 `device=False`。

**验证。** 本地 3287 条通过 / 460 跳过（较 5.330 增 12），ruff 通过。
`test_nccl` 新增 `TestInitRank` 3 条（先切卡再初始化、device 缺省取 rank、
负 device 拒绝），并把能力声明用例改为按新事实断言（成功时
`device_allreduce=True`、`provider="nccl"`、理由含库版本与两条例名入口；
缺库时 False + 理由含"加载不到"）。`test_zero` 新增
`NcclDeviceCollectiveFromEnvTest` 4 条（单 rank 直通、rank 0 分发 uid、
worker 取回 uid、rank / world 越界拒绝）。`test_ddp` 新增
`TestDistributedFromEnvBindsDevice` 3 条（设备态默认走
`NcclDeviceCollective.from_env` 且桶 0 / 线程 1、显式 `device=False` 走
`HostCollective.from_env` 且桶 >0 / 线程 2、缺库抛
`DeviceCollectiveUnavailable`）。`test_parallel` 的 `TestCollectiveSeam`
增 2 条（auto 在有库时自建 provider 并落在 device、缺库时落回 host 且不建
communicator），旧的"注入 provider 却选 host"用例改为显式
`collective="host"` 以保留原语义。

**未声称。**

- 本机没有 NCCL / RCCL，"有库"分支全部靠 mock 覆盖；真机上的
  `ncclCommInitAll` 多卡归约与 `from_env` uid 分发仍待授权回归。
- `overlap` 仍恒为 False：设备态强制 `bucket_bytes=0` / `threads=1` /
  `overlap=False`，没有"梯度一就绪就归约"的反向重叠。
- `ncclCommInitAll` 那组 communicator 只服务单进程多卡；每进程一卡的 DDP
  必须走 `ncclCommInitRank`，两组不能混用。
- 跨机、无 NCCL 的场景仍只能 `device=False` 走主机中转；`auto` 不会
  在没有库时偷偷退回。
- `ncclAvg` 与 bf16 仍被绑定拒绝（平均在调用侧补除、bf16 主机侧算不出
  期望值），本次未放开。


### 5.332 循环级 IR：循环不变量外提与单图规范形态

**缺口。** 控制流一直只有"图与图之间"的形态：`WhileLoop` 每轮在主机上拼
feeds（5.201），`DeviceWhileLoop` 把条件与体各捕获一次（5.245）。循环内部
的结构没有被任何一层分析过，所以"体里哪些计算每轮重算、结果却不变"没有
答案，编译器也就没有循环级优化的落点。README 第 6 节因此把"无 loop 级 IR"
列为与主流编译器的差距。

**1. `LoopIR`：循环的单图规范形态。** 新模块 `ir/loop.py` 用
`LoopIR(condition, body, carried=..., max_steps=..., name=...)` 表示一个循环：
条件与体都必须是单张 `Graph`，构造期直接复用 `WhileLoop` 的契约检查
（carried 封闭、谓词单标量、类型一致），不另写一份事实来源。
`LoopIR.from_control` / `to_control` 与 `WhileLoop` 互转；`to_control()`
是它与既有宿主路径（`WhileLoop.run`、`DeviceWhileLoop`、`Trainer`）的桥——
本层不新增执行机制。嵌套控制区域（`Conditional` / 内层 `WhileLoop`）没有
可静态分析的单图形态，显式抛 `LoopIRError` 并指回 `WhileLoop`，不假装能分析。

**2. 依赖分析：`must_run` 与 `invariant_nodes`。** 从 body 中 carried 状态
对应的输入出发，沿生产者传播得到"必须每轮执行"的值；不纯节点（`op.pure`
为假，即带 RNG / STATE / IO 效果）及它们的消费者也进入该集合——把"每轮
重新采样"改成"只采样一次"会改变数值。其余纯节点即 `invariant_nodes()`。
`constant` 不计入不变量：它本身几乎零成本，外提只会多一次每轮传递的输入。

**3. `split_invariants`：preheader + 瘦身循环。** 从"必须留在体内的节点
输入"与图输出出发，沿生产者收集被实际消费的不变值（未被使用的纯节点是
死代码，两边都不保留），复制成 preheader 图；新体保留原自由输入，并把
外提值作为**同名输入**接回来——carried 映射与输出别名因此都不用改。
体内直接引用的常量就地复制，不变成循环输入。重写后两张图都过 verifier：
外提最容易出的事故是"新图少一条边"，而它看起来仍是一张合法 DAG。
`LoopSplit.run` 先跑一次 preheader、把结果并入 feeds，再走
`WhileLoop.run`，所以数值与原循环逐位相同。`optimize_loop(WhileLoop)`
是"提取 + 外提"的一步入口。

**4. 序列化。** `loop_to_spec` / `loop_from_spec` 与 `ir.control` 的
`to_spec` / `from_spec` 并列：普通控制流存档仍是 `kind="while"`，循环级 IR
用 `kind="loop"`，子图复用 `ir.serialize`。两套入口刻意不合并——`control`
不能反向依赖 `loop`（会形成循环导入），而把两个 `kind` 塞进同一个函数只会
让"拿到的是哪种对象"更难判断。

**验证。** `tests/test_loop_ir.py` 30 条：提取 / 往返、嵌套区域拒绝、
`invariant_nodes` 的命名与数量（含 RNG 节点不参与、常量链可外提但常量
本身不出现在结果里）、外提后的节点数（preheader 4 / 体 3）、两张图过
verifier、多轮数值与原 `WhileLoop` 逐位相同（`acc = 21`）、重复执行、
额外输出（`y`）拿到相同的 bias、无可外提时原样返回、缺前置输入 / 缺
carried 初值的显式错误、JSON 往返与三类非法存档。

**未声称。**

- 只支持单图体；嵌套控制区域不进入循环级 IR，也不做"把内层循环展平"
  这类变换。
- 只做循环不变量外提，不做循环融合 / 分块 / 软件流水，也不把循环接到
  schedule 搜索或编译流水线上——调度搜索的接线是后续阶段。
- 条件图不参与外提：它必须每轮求值，否则"何时退出"就不再由数据决定。
- 外提只按"是否依赖状态 + 是否纯"判断，没有成本模型；对极小的不变计算，
  preheader 多一次图执行可能不划算。


### 5.333 调度配置接进执行路径：按形状命中的 `SchedulePolicy`

**缺口。** 配置库（5.37，搜索与配对判定见 5.140～5.160）与执行路径之间
一直没有桥。`tune` / `cost_ranked_tune` 能把"某张卡、某个形状下实测最快"
写进 `ConfigStore`，`agent_api.recommend_config` 能把它查出来给智能体看
——但真正 launch 时，`_launch_block` / `_reduce_block` / `_conv2d_tiled`
只看 `set_tuning` 的覆盖与启发式，库里的答案不会自己生效。结果就是
"测过、存过，跑起来还是老样子"。

**1. 签名必须与写库时同源。** 新增 `tuning/signature.py`：
`block_signature(op, *, work, vec=1, dtype=None)` 与
`conv2d_signature(x_shape, weight_shape, *, stride, padding, dilation, groups,
dtype)`。两者都过 `canonical_signature`（tuple 与 list 统一成 list、dtype 转
字符串），保证"写库脚本算出的键"与"执行期算出的键"是同一个 JSON 字符串。
形状长度不为 2 直接抛 `ValueError`——`_pair` 式的长度校验放在入口，比在
配置库里留一条永远匹配不上的记录好查。

**2. `SchedulePolicy`：查表、缓存、统计。** 新模块 `tuning/policy.py`。
四段键与库一致（`backend + device + op + signature`）；`from_backend` 剥掉
上报包装层（`inner`），设备名取 `device_info().name`，取不到写 `"unknown"`
——与库的写入口径一致，不编造。`lookup` 按 `(op, 规范签名)` 缓存结果（包括
"没有记录"），逐次 launch 不再重复序列化签名；`enabled=False` 时直接返回
`None` 且不计入统计，"没开"与"没命中"要能分开。

`overlay(config, current)` 只覆盖**当前认识的键**：未知键（旧版本记录多带
的参数）忽略并计入 `unknown`；值不是正整数或 `None` 则抛 `ValueError`，
与 `set_tuning` 同一口径。库里一条 `block=96` 的记录会让内核按 3 的倍数
派线程，静默退回自动只会表现为"调优没生效"，所以宁可当场报错。`None`
是合法值，表示该项仍交给计划层按形状推导。

**3. 接通执行路径与优先级。** `DeviceBackend` 增加
`set_schedule(store)` / `clear_schedule()` / `schedule_policy()`，前者按
后端自身的 `name` 与设备名建策略并 `_invalidate_plans()`。优先级固定为：

```
按算子 set_tuning 覆盖 → 构造参数 block → 配置库 → 按规模自动
```

前两者是人当场下的指令，库是过去测出来的建议，自动是没人说话时的兜底。
`_launch_block` 的签名只用 `work` 与 `vec`，不含 dtype：block 是线程铺法，
取决于"有多少活、每线程搬几个"，与元素类型无关；离线写库也必须用同一组
字段，否则永远不命中（文档写在两个函数的 docstring 里）。

conv2d 只在 `_conv_tile` / `_conv_roc` / `_conv_cb` **全为 `None`** 时查库，
命中后用局部变量覆盖三个旋钮传给 `plan_conv_tiled`，不改后端字段：调用方
手动调过其中任意一项，说明正在做 A/B，此时混入库里的值会让"这次改的到底
起没起作用"说不清。几何约束（`tile_w * block_y <= 1024` 等）仍由
`plan_conv_tiled` 在真正执行前判断，坏值同样报错而不是静默回退。

**验证。** `tests/test_schedule_policy.py` 26 条：签名键与 JSON 往返、
tuple/list 等价、形状非法报错；命中 / 未命中 / 禁用不计 / 缓存计数 / 返回值
是副本 / 空签名与非法构造参数；`from_backend` 剥包装层、设备名缺失记
`unknown`、设备名可显式覆盖；`overlay` 的覆盖 / 忽略未知键 / 坏值报错 /
`None` 表自动；后端优先级（显式覆盖 > 构造参数 > 库 > 自动）、库中非法
block 报错、`clear_schedule` 回到自动、`set_schedule` 作废计划缓存。
用例用 `object.__new__(DeviceBackend)` 装上 `_launch_block` 需要的字段，
不启动硬件——这一层要钉的是优先级，不是这张卡快不快。

**未声称。**

- 不做"运行时自动发现在跑什么并触发搜索"。接库只回答"已经测过的形状用
  什么"，没测过的形状仍走启发式与成本模型；在线探索要先有可信测量。
- 配置库在执行期被改写不会自动生效：`SchedulePolicy` 缓存的是查表结果，
  更新库后要重新 `set_schedule`，不做隐式热更新。
- 只接了逐元素 / 归约的 block 与 conv2d 分块核的三个旋钮。im2col、
  implicit GEMM、Winograd 与 matmul 的选型还没有按形状查库的入口。
- 同一份配置的收益仍以真机测量为准；本轮只保证"库里的答案会被用上、用错
  会报错"，不声称任何加速比。

### 5.334 设备端原生 NHWC：conv2d 分块直算与池化前向

**缺口。** 布局传播 pass（5.246）能把 conv / maxpool / avgpool 区域翻成 NHWC
并在区域边界插转置，但设备端声明的是 `supported_layouts = {"nchw"}`——翻好的
图到 `invoke` 入口被整体拒绝，只能在参考后端跑。也就是说"布局决策"在设备上
从来没有兑现过：要么不翻转，要么翻完报错。

**1. 参数化内核描述，而不是复制一套数学。** NHWC 改变的只有**寻址**：
NCHW 的通道偏移每步跳 `H*W`，NHWC 是相邻地址。所以 `lower/kernels/` 里没有
第二套卷积：

- `image.py` 增加 NHWC 三片段：`_split_output_index_nhwc`（把展平下标拆成
  `(n, h, w, c)`，`c` 在最后一段）、`_image_slice_nhwc`（只加
  `n * H*W*C + c`，窗口偏移交给 tap）、`_tap_nhwc`（`(ih*W + iw) * C`）。
- `pool.py` 用 `_layout_fragments(layout)` 在这三片段与 NCHW 版之间选，
  `maxpool2d(layout=...)` / `avgpool2d(layout=...)` 的内核名带 `_nhwc` 后缀。
  窗口、越界判断、`count_include_pad` 的分母逻辑一行未改。
- `conv.py` 的 `conv2d_tiled(layout=...)` 只换 6 处寻址：输入通道偏移
  （`Cast("long long", index)` vs `((long long)index * H) * W`）、行步长
  （`W` vs `W * C`）、列步长（`1` vs `C`）、输出指针、输出通道步长。dst 与
  写回抽成 `_output_writeback(layout)` 供 tiled / implicit 共用；NHWC 版
  implicit 只在下标处调用默认参数，不产生第二份代码。

命名后缀是刻意的：同一个进程里两种布局要能各占一个内核缓存槽
（`("conv2d_tiled", kind)` 与 `("conv2d_tiled_nhwc", kind)`），否则先跑的
那版会被复用去算另一种布局。

**2. 后端只放行真正有内核的覆盖面。** `supported_layouts` 改成
`{"nchw", "nhwc"}`，但 NHWC 的承接面收窄在 `device_conv` / `device_pool`
的入口：

- conv2d：只走分块直算。`groups != 1`、`conv_impl` 显式指定
  im2col / implicit / winograd / grouped / cudnn、非 f32，都抛
  `BackendCapabilityError` 并给出"改用 `layout='nchw'`"的建议。`auto` 在这里
  等价于 `tiled`——NHWC 的收益来自通道连续，而 tiled 的内层正好沿通道走。
- 池化：`maxpool2d` / `avgpool2d` 只加前向分支；反向与 `maxpool2d_indices`
  没有 `layout` 属性（不在 `_LAYOUT_OPS` 里），训练图仍按 NCHW 构造，因此
  不需要也不应该在这里假装支持。
- 输出形状：`(N,HO,WO,O)` 与 `(N,HO,WO,C)`；权重保持 `(O,C,KH,KW)`，与
  参考后端的 NHWC 约定一致（不引入 OHWI 之类的第二套权重布局）。

**3. 闸门从"只认 NCHW"改成"按声明判断"。** `invoke` 与
`device_plan._prepare_plan` 原来是硬编码 `layout != "nchw"` 就拒绝。前者改为
`layout not in self.supported_layouts`，第三方后端只要不改代码就仍是"只接
NCHW"；后者对非 NCHW 直接弃权返回 `None`——快路径是手写的 NCHW 拼接，NHWC
走 `invoke` 的算子实现。

**4. 签名区分布局。** `conv2d_signature(..., layout=...)` 只在非 NCHW 时并入
`layout` 字段：NHWC 的 `(1,8,8,3)` 不能套用 NCHW 同形状测出的 tile 配置，
但 NCHW 的旧签名必须一字不变，否则库里既有记录全部失效。

**验证。** `tests/test_nhwc_native.py` 13 条（无 GPU）：两版内核名与源码的
寻址差异（通道偏移、输出写回、池化 base / tap）、非法 layout 在内核层与签名
层都报错、NCHW 签名不含 `layout`、入口拒绝（im2col / groups=2 / f16 /
通道数不一致）。`tests/test_layout.py` 原有"设备端只声明 NCHW"改为声明两种，
并把"HIP 拒绝 NHWC"换成两条真机一致性用例（NHWC 与 NCHW 转置后逐位相等，
conv 与 maxpool 各一条，无 ROCm 时跳过）。全量 3357 条通过，ruff 通过。

**未声称。**

- 不做声明式转换：图里写 `layout="nchw"` 不会偷偷按 NHWC 执行，反之亦然。
  翻转仍由 `LayoutPropagation` 决定。
- 没有 NHWC 的 im2col / implicit / Winograd / 分组 / cuDNN，也没有池化反向；
  这些请求显式失败，不静默换实现。
- 不声称加速比。NHWC 只保证"翻好的图能在设备端跑出与 NCHW 逐位相同的结果"，
  收益取决于形状与访存模式，要以真机测量为准。
- 配置库没有 NHWC 记录：新签名要重新测过才有命中，本轮不生成。参数化版本
  也没有进 `plan_conv_tiled` 的成本模型（几何仍按 NCHW 同一套预算判定）。

### 5.335 前缀缓存接主路径：`generate` 自动挑选最长前缀

**缺口。** 5.307 给了块级引用计数与写时复制，5.310 给了 token→token 的生成
闭环，但"共享前缀"仍是**显式动作**：调用方必须自己知道父序列的 id，并且保证
prompt 以父序列全部 token 开头。服务侧最想要的那一格——"同一个系统提示、
不同的用户问题"——要么手工维护一张 prompt→rid 的映射，要么每次重解系统提示。
缺口表第 5 项的 KV swap 部分已由 5.304 / 5.310 接上（`suspend(evict=True)` 后
`resume` 续写），本节补的是 prefix caching 的自动匹配。

**做法：命中判断放进 `PagedGenerator.generate`，不进调度器。**

- `PagedGenerator.__init__` 增加 `prefix_cache: bool = True`，并给
  `stats()` 增加三个计数（与调度器统计合并返回）：
  `prefix_hits` / `prefix_reused_tokens` / `prefix_misses`。
- `generate` 在 `fork_from is None` 且开关打开时调用 `_match_prefix(ids)`：
  遍历驻留序列（`self._tokens`），取"token 序列是 prompt 严格前缀"里最长的
  一条，`fork` 它，只对剩余 token 做 prefill。显式 `fork_from` 优先，且不计
  命中/未命中——那是调用方的决策，不是缓存判断。
- 三个排除条件都是正确性而不是优化：**自身**（防止调用顺序变化时自 fork）、
  **换出态**（`fork` 要求驻留，换出的父序列得先 `resume`，而"自动触发换入"
  会把一次生成变成不可预期的搬运）、**等长**（生成需要一个尚未解码的位置，
  父序列最后一步的 logits 没保存，和显式 `fork_from` 的既有报错同一理由）。

**为什么不做哈希索引。** 单进程里驻留序列通常是个位数到几十，线性扫一眼的
成本远低于维护一张"前缀树 / token 序列 → rid"索引——后者要在 `finish`、
`fork`、`suspend`、`resume`、`feed` 五处同步失效，任何一处漏掉就是**用别的
请求的 KV 算出错误 token**（不会报错，只会答错）。这是本仓库最不能接受的
失败模式，所以第一版选可读的线性匹配；等序列数真的上千再谈索引，届时正确性
边界要重新论证。

**与换出 / 换入的关系。** 命中只共享**驻留**序列：换出态被跳过，但换出态本身
仍可通过 `resume` 回到驻留、再被后续请求命中（它的 `_tokens` 一直在
`self._tokens` 里）。`fork` 出来的子序列也进入同一张表，所以链式命中成立
（A → B → C 依次共享越来越长的前缀），用例
`test_longest_prefix_wins` 覆盖了这条。

**验证。** `tests/test_attention_cache.py::TestPrefixCache` 5 条：

1. 命中后的生成结果与同参数集独立实例逐 token 相等（缓存只省 prefill）；
2. 命中共享父序列的物理页，`forks` 加一、`shared_blocks` 非零，`used_blocks`
   至多多一个写时复制的尾页；
3. 多条候选时取最长前缀（复用 token 数 2 + 3）；
4. 换出态与等长序列都不参与匹配，结果仍等于独立生成；
5. `prefix_cache=False` 时 `shared_blocks` 为 0、页数按独立序列增长。

全量 3362 条通过，ruff 通过。

**未声称。**

- 进程内的前缀缓存，跨进程 / 跨实例 / 落盘的那一半见 5.337；也没有 token 级
  哈希（按整条序列比对，命中粒度是"某个请求已经解码到的位置"）。
- 不自动换入：换出态的序列不会被命中触发 `resume`，调用方仍要自己决定
  何时把长上下文请求搬回显存。
- 不做前缀树的 LRU 淘汰：驻留序列的生命周期仍由显式 `finish` 控制，
  `prefix_cache` 只改变"新请求怎么复用"，不改变"旧序列何时释放"。
- 仍是主机态组件：命中省的是 prefill 的 host→device 与算子调用，没有真机
  D2H / 页搬运的端到端时延数据。

---

### 5.336 国产卡后端：接入路径、离线可验证面与真机准入判据

本小节记录**已落地的接入形态**与主仓侧的验收状态。缺口表第 6 项原先写"不做"，
理由是"需要厂商联合优化，单机项目做了也是半成品"；现在的收窄方式是：实现放在
**独立仓库的硬件包**里（`aetherflow_iluvatar` / `aetherflow_metax`，各自独立
git 仓库），主仓只负责契约验收与拒绝面，不在 `aetherflow/backends/` 下再养两套
设备后端。

**两个实现（事实）**

| 包 | 厂商 / 卡 | 工具链 | 状态 |
|---|---|---|---|
| `iluvatar_af` 0.1.0 | Iluvatar CoreX / 智铠 MR-V100 | 天数 Clang `-x ivcore --cubin`，driver API 加载 | 真机跑通，flash attention 512×512×64 / 256 batch·head 约 0.6 ms（约 57 TFLOPS） |
| `metax_af` 0.1.0 | MetaX / 曦云 C500 | `mxcc -O2 -D__CUDACC__`，MACA 3.10.1.11 的同名 CUDA 兼容库 | 真机跑通（含 D=64 / D=128 两套 flash attention） |

两者共同的取舍：不继承 `CudaBackend`（否则把 cupy 容器与 cuDNN 假设一起带过来），
直接继承 `DeviceBackend`；内核模板逐字复用 `cuda.codegen`，只替换 f16 的三个
辅助函数（PTX → `_Float16` + `__builtin_bit_cast`）；`ctypes` 直连厂商运行时与
BLAS，不引 cupy / torch；`CONV_IMPLS` 去掉 `cudnn`。两家 flash attention 的
mma 片段形状不同（天数 16×16×32、C500 16×16×16），这部分各自重写并实测。

**主仓侧已验收（本机无卡，2026-11-28）**

用 `install_hardware` 把两个 zip 装进隔离的 `AETHERFLOW_HARDWARE_DIR`：

- 两个包都被识别成 `api=1` 的硬件包，`name` / `version` / `vendor` / `factory`
  与清单一致，`scan_errors` 为空；未签名（`signed=False`）按契约接受。
- 无厂商库时 `get_backend("iluvatar")` / `get_backend("metax")` 抛
  `BackendUnavailableError`，消息列出尝试过的 `libcudart.so.10.2` /
  `libcudart.so.10`——**不静默回退参考后端**，这正是契约要的行为。
- `requires` 只声明 `numpy` 与算子子集，安装期不执行包作者代码（`find_spec`
  查依赖），`metax` 的清单在构建时按当前 `GPU_OPS` 生成。

**未验收**：主仓内没有跑过这两个包的 conformance——那需要对应硬件。真机结论
（数值、性能、四模式 flash attention）在两包各自的 `REMIND.md` 里，不是主仓
的证据。

**已具备的接入条件（事实，不是设想）**

- 插件契约有两层：pip entry point 与 zip 硬件包（清单 / 摘要 / 可选签名 /
  依赖校验，5.43）。后者正是 5.292 里天数（Iluvatar）现场走的路径：
  `py -3.10 build.py` 打包后 `install_hardware("dist/iluvatar-0.1.0.zip",
  replace=True)`。也就是说"国产卡作为硬件包接入"不是新设计，是既有能力面。
- `DeviceBackend` 已把厂商相关收敛成两个注入点：`self.cp = runtime`
  （容器 / 编译 / 启动 / 同步 / 设备属性）与 `self.codegen`（模板 + 方言 +
  f16 存取）。HIP 是这条路线的实证：同一批模板逐字复用，只换 runtime /
  BLAS / f16 存取，conformance 62 executed / 5 refused / 0 failed（5.40）。
- 纯数据部分完全离线可验证：`lower.emit` 打印 L2 源码，方言是数据，
  novac 示例（5.44）已演示"装上 → 识别 → 出源码 → 跑图"。
- 代码里已按天数 `DeviceArray` 的语义差异做过适配（5.292：拒绝降维整数
  索引，改用 `reshape`），说明这条路线踩过真问题，不是纸面推演。

**三层拆分，各自的可验证性**

1. **方言层**：模板是纯 C++ + `__global__`，不通用的是 f16 的 PTX 内联汇编。
   两家都用编译器原生能力替换：天数 Clang（`-x ivcore`）与沐曦 `mxcc` 都接受
   `_Float16` + `__builtin_bit_cast`，与 HIP 踩过的同一处（5.40 的
   `invalid output constraint`）解法一致；主体与 CUDA 逐字共用。
2. **运行时层**：两包都用 `ctypes` 直连厂商运行时（不引 cupy / torch），编译
   是离线产 `cubin`（天数 Clang、沐曦 `mxcc`），加载与启动走 driver API
   （`cuModuleLoadData` / `cuLaunchKernel`），BLAS 走厂商提供的 CUDA 兼容库。
   主仓不重复实现这一层。
3. **能力声明层**：`capabilities()` 只能来自实际跑通的内核集合，不能照抄 CUDA
   的表。两包都从 `GPU_OPS` 去掉库依赖项（SPARSE / FFT / cuDNN 快路径）起步
   （`metax_af` 参考 100 个算子，`iluvatar_af` 同量级），等 conformance 逐项
   通过再扩。

**真机准入判据（不可谈判）**

- 对应物是 `verify_cuda.py`：`run_conformance` 全绿（executed + refused，
  0 failed）才算"接入"；只跑通 demo 不算。
- 拒绝面要有用例：库缺失、非 f32、`groups > 1`、不支持的 `layout`、
  不支持的图捕获，都显式报错，不静默降级。
- 性能不进准入：没有实测数字就不写数字，也不把"能跑"说成"更快"。

**主仓侧待做与依赖**

- 已做（离线）：安装 / 识别 / 能力声明 / 无卡拒绝面，见上。
- 待做（需天数机器）：`run_conformance` 的 f32 子集，通过后把逐项结果写回
  两包各自的 `REMIND.md`，并更新本节状态列。
- 待做（需沐曦机器）：同上，另加 D=64 / D=128 两条 flash attention 的数值
  对拍（掩码 / dropout / causal 截断四模式）。
- 待做（两机都有后）：`SchedulePolicy` 的设备名入库（5.333 的库要真机签名）。

**明确不做**

- 主仓不重复实现厂商 runtime：两个包的 runtime / BLAS / 编译都在包内，主仓
  只保证契约；厂商 API 以两包现场验证过的调用为准，不另写一套。
- 不在 `aetherflow/backends/` 里加第四套设备后端：国产卡走硬件包，与 novac
  同层；把转译塞进装配层会让"一套 IR 落到不同硬件"变成各家自己解释 IR
  （`hardware.py` 已有的边界）。
- 不做多卡 / 集合通信，先有单卡 conformance。
- 不承诺性能，不写吞吐 / 时延数字。

**未声称**

- 主仓内没有任何国产卡 conformance 结果；本机无卡，只验证了装配与拒绝面。
- 两包的真机数字（MR-V100 的 0.6 ms / 57 TFLOPS、C500 的 flash attention）
  来自包侧 `REMIND.md`，主仓未复跑，也不把它们写成主仓的基线。
- 两包不在主仓的 CI 里，也不随主仓发布；换框架版本时要各自重打包——清单里的
  文件摘要会失配，安装期就拒绝，不会"装上了但跑的是旧代码"。

**继续推进需要的东西**

一台天数 MR 卡与一台沐曦 C500（或远程机器，凭证走环境变量 / 密钥文件，不写进
仓库与日志），在两机上按各自 `REMIND.md` 的命令复跑 conformance，把结果回填到
本节的状态列。

---

### 5.337 落盘 / 跨进程前缀缓存：条目格式、失配拒绝与生成侧接线

5.335 把前缀缓存接进 `generate`，但复用面只在**进程内**：进程退出后 KV 随池
消失，同一个系统提示每天第一次请求仍要重解一遍。本节补缺口表第 5 项剩下的
那一格，做法是"把 token 序列与每层 K/V 落盘，下次进程按同一判据读回来"。

**落盘的是"token 序列 + 每层 K/V"，不是页。**

- 新模块 `aetherflow/runtime/prefix_store.py::DiskPrefixCache`：一个目录对应
  一份配置，`model_id`（模型指纹）、`block_size`、层数、头数、头宽、dtype 全
  写进条目 meta，任一项不同就不命中。
- 条目是单个 `.npz`：`meta`（UTF-8 JSON 字节）、`tokens`（int64）、`keys` /
  `values`（`(L, T, H, D)`）。文件名是 `sha256(model_id + tokens)`，meta 里存
  同一份摘要并与文件名绑定：zip 成员自带 CRC 负责数据段完整性，摘要负责
  "这个文件确实属于这串 token"。
- 写入走**同目录临时文件 + `os.replace`**，`fsync` 在换名之前。任何时刻可见的
  条目都是完整的一份，断电最多丢一条，不会留下半份被读回。
- 查询只接受**严格**前缀里最长的一条，与 5.335 同判据（等长不行：生成需要一个
  尚未解码的位置）。候选按长度降序尝试，最长那条读坏就试次长的。
- 损坏（读不出 zip / 少了成员 / meta 非法 / token 摘要不符 / 形状或 dtype
  不符）与配置失配都**不是异常**，是"没有条目"：扫描跳过并计入 `corrupt`，
  查询返回 `None`，生成侧退回普通 prefill。同一个目录混进别的配置的文件不该
  让进程起不来。
- 淘汰按文件 mtime（命中时 `utime`），先按 `max_entries` 再按 `max_bytes`。
  跨进程没有可靠的访问计数，mtime 是最便宜的"最近使用"。
- 并发不做锁：不同条目是不同文件名 + 原子替换；同 key 并发写只会互相覆盖成
  同一份内容。

**生成侧接线：命中省 prefill，落盘在序列结束。**

- `PagedGenerator.__init__` 增加 `disk_cache=None`，并校验它的 `block_size` /
  `num_layers` / `num_heads` / `head_dim` / `dtype` 与分页池一致，不一致在
  构造时报 `ValueError`——不做"临时转换再导入"，否则配置失配会从显式错误变成
  静默慢路径。换配置请换目录。
- `generate` 的自动分支顺序：显式 `fork_from` > 内存前缀（5.335）> 盘前缀 >
  普通 prefill。盘命中时 `submit` 后直接把条目 KV `append` 进池，`_tokens[rid]`
  设为条目 token，只对剩余 token 做 prefill。装不进池（`PagedCacheFullError`）
  时结束刚提交的序列、退回普通 prefill——"盘上有但池里放不下"是容量条件，不该
  让请求失败。
- `generate` 返回前把这条序列已解码的全长落盘（最后一个生成的 token 还没喂回，
  所以落盘长度 = prompt + 生成数 - 1）；同 key 已存在且长度一致时只刷新 mtime，
  不重写。
- 统计：`disk_hits` / `disk_reused_tokens` / `disk_misses` / `disk_stores`，
  与 5.335 的三个计数一起进 `stats()`。`prefix_cache=False` 同时关闭盘查询与
  落盘——那是对"自动前缀复用"的显式拒绝；`prefix_misses` 现在表示"内存与盘
  都没命中"。

**为什么落盘粒度是整条序列而不是分页。** 页是**进程内**的物理布局，页号换一次
进程就没有意义；跨进程唯一稳定的身份是 token 序列。存成 `(T, H, D)` 连续段，
导入方按自己的 `block_size` 重新分页——5.335 已验证页宽不影响数值，所以换页宽
只是换个目录（配置失配），不会算错。反过来，若存"页 + 块表"，就得在盘上重建
一份分配器，且任何页宽变化都不可读。

**为什么不做 token 级哈希索引。** 与 5.335 同一条理由：索引要在 `feed` /
`finish` / `fork` / 换出换入多处同步失效，漏一处就是用别的请求的 KV 答错字。
落盘条目数本来就被 `max_entries` 限制在几十条，线性扫描加 meta 预筛足够。

**验证。** `tests/test_prefix_store.py` 20 条。

存储层 15 条：

1. 存→查往返，逐层逐位一致；
2. 等长不算命中（`misses` 加一）；
3. 多条候选取最长严格前缀；
4. 新实例扫描同目录（跨实例）；
5. **真子进程**写入、父进程读回（`subprocess` + `PYTHONPATH`），这是"跨进程"
   的直接证据；
6. 五种配置失配（`model_id` / 层数 / 头数 / 头宽 / 页宽）都只跳过并计 `corrupt`；
7. 文件被改成非 zip：新实例扫描拒绝、同实例查询时剔除；
8. tokens 数组被篡改：长度与内容校验拒绝；
9. `max_entries` 淘汰最旧；
10. `max_bytes` 淘汰到装得下（上限 1 字节时清空）；
11. `clear()` 清空目录；
12. 空 tokens / 层数不对 / 长度不对 / 非浮点都拒绝；
13. 同 key 不重写（`stores` 不增）；
14. 原子写不留临时文件；
15. 返回的 payload 是副本，改它不影响下一次读。

生成侧 5 条：

16. 新实例（内存无序列）盘命中，生成结果与稠密图逐 token 相等，
    `disk_reused_tokens` 等于前缀长度；
17. 等长条目不命中，退回 prefill，结果仍相等；
18. `prefix_cache=False` 时不查盘不落盘（四个计数全 0，已有条目还在）；
19. 盘上条目损坏时退回 prefill，结果仍相等；
20. 形状失配的盘缓存在构造时报 `ValueError`，非 `DiskPrefixCache` 报
    `TypeError`。

全量 3382 条通过，ruff 通过。

**未声称 / 剩余**

- 条目里存的是**本层权重算出的 k/v**，不含调用方在图外加的位置信息；换模型
  版本必须换 `model_id`，否则会被当成同一份配置复用（这是调用方的契约）。
- 不做压缩 / 量化：条目大小 = `2 × L × T × H × D × sizeof(dtype)`，长上下文
  会很大，靠 `max_bytes` 兜底而不是编码。
- 不做跨进程锁与自动清理：多进程同目录写不同条目安全，同 key 并发写是"最后
  一份赢"（内容相同）；坏文件不自动删，`clear()` 手动清。
- `generate` 结束与 `finish` 释放前都会落盘；裸 `feed` / `prefill` 之后不
  `finish` 就不会落盘（`finish` 是手工驱动序列的结束信号，5.339）。
- 仍是主机态组件：没有真机 D2H 与落盘时延数据。
- 未接 `LMRunner`（Mamba 主干没有随上下文增长的 KV）。


### 5.338 多卡路径事实回填：设备侧归约默认、主机态桶间并发与设备态反向重叠的准入判据

缺口表第 2 项此前写成一句话："DDP 走主机中转、无反向重叠、无 P2P"。三件事
要么已经变了，要么本来就不是同一件事，继续用这一句会让人误判现状：设备侧
归约从 5.331 起就是默认路径；"桶间并发"从 5.302 起就有；真正缺的是"设备态的
反向计算与集合调用重叠"和 P2P。本节把事实拆开写清，并把
`collective_support().overlap` 恒为 False 的理由收窄到它实际声明的范围。

**事实一：设备侧归约是默认，不再是"只能主机中转"。**

- `collective="auto"`（`DataParallelTrainer` / `DistributedDataParallel` 的缺省）
  在 NCCL / RCCL 可加载时走设备侧：单进程多卡 `ncclCommInitAll`，每进程一卡
  `NcclDeviceCollective.from_env`（rank 0 分发 uid、其余 `ncclCommInitRank`）。
- 库加载不到才经主机中转（D2H → 主机求和 → H2D）；显式
  `collective="device"` 而库缺失抛 `DeviceCollectiveUnavailable`，不静默降级。
- 所以"无 NCCL 时每卡一份整梯度的往返"仍是真实代价，但它现在是**回退路径**，
  不是默认路径。

**事实二：主机态的桶间并发已经落地（5.302）。**

- `DistributedDataParallel(overlap=True)` 在主机态、`threads=1`、
  `bucket_bytes>0` 时，为每个参数桶建一张独立反向图；桶算完立刻把"桶就绪"
  回调和该桶梯度交给后台单线程 FIFO，归约与主线程算下一个桶重叠。200 个
  2 KB 参数实测 65.4 → 28.5 ms/步（2.30×，5.302）。
- 这是"桶与桶之间"的并发：每个桶仍要从损失重算一遍共同链式路径，所以强制
  `bucket_bytes>0`（禁止逐参数一桶）。
- 它**不是**"设备态与反向重叠"，也不是 P2P：主机中转的求和仍发生在主机，
  单线程 FIFO 是为各 rank 同序，不是为并行发多条通信。

**事实三：`overlap` 恒为 False 是准确声明，但要读对它的主语。**

- `CollectiveSupport.overlap` 声明的是"**设备态**集合调用能否与反向计算
  交错"。设备态反向是整图 capture / replay，执行器没有"某个参数（桶）梯度
  就绪"的时刻，`Trainer.enable_bucket_overlap` 在设备态显式抛
  `ValueError("设备态不支持分桶反向：capture / replay 是整图")`。既然没有可
  提交的窗口，声明 False 是对的。
- 早先的 docstring / reason 写成"执行器同步、切卡串行，进程内没有重叠窗口"
  "计算/通信重叠仍未实现"，把"设备态没有"说成了"完全没有"，与 5.302 的事实
  矛盾。本次改为点名主语：设备态没有；主机态桶间并发在
  `DistributedDataParallel(overlap=True)`，不进这份声明。
- `overlap=False` 的返回值不变。`tests/test_parallel.py` 只断言
  `overlap is False` 与 reason 含"重叠"，`tests/test_nccl.py` 断言 reason 含
  库版本与两条例名入口；改后文本仍满足这几条。

**设备态反向重叠的准入判据（要动哪些组件）。** 这件事不是"把 `overlap` 翻成
True"：

1. **执行器要能按参数分段反向并暴露就绪事件。** 设备态现在是整图捕获一次、
   重放一次（5.245）；分段反向只在主机态 / 图级成立（5.219 的分块是主机图
   切块）。设备侧必须先有"按参数桶捕获、按桶重放"，且每段反向在设备上完成、
   不把梯度拉回主机才能提交 NCCL。
2. **NCCL 调用要落在可控的流序里。** 现绑定用 `all_reduce(..., stream=0)`，
   与反向计算流的关系靠默认流顺序；真要重叠，需要显式流与事件，且各 rank 的
   段序一致（否则互相等死，与 5.302 的 FIFO 约束同源）。
3. **至少 2 卡真机**验证"分段重放 + 重叠归约"的数值与整图反向逐位一致，并测出
   不回归的桶大小 / 时延。本机没有可用 GPU 对（2×V100 已释放），这一步现在
   做不了。
4. 需要同步改的组件：`train/loop.py::Trainer`（设备态分段捕获 + 就绪回调）、
   `train/parallel.py::DistributedDataParallel`（设备态放开 `overlap`、把回调
   接到 NCCL 提交）、`train/nccl.py`（流 / 事件参数）。三处都没有"纯主机态
   可验证"的形态：没有真机就只能 mock 出一个假的"桶就绪"，验证不到重叠本身。

**P2P 仍不做（5.169 的决策不变）。** `copy_across_devices` 一律经主机中转，
不提供隐式 peer-to-peer：它要求两块卡可互访，失败回退会掩盖"哪一步降级了"，
且需要 `cudaMemcpyPeer` / `hipMemcpyPeer` 与真机验证。要做的话是一个独立入口
（显式声明"这两块卡能互访"），不是把默认路径改成 P2P。

**验证。** 本次只改对外表述，无逻辑变更：`py -3.14 -m ruff check aetherflow
tests` 通过，`py -3.14 -m unittest discover -s tests -t .` 全量 3382 条通过
（与 5.337 同数，无新增用例）。`test_parallel` 的能力声明用例、`test_nccl` 的
理由文本用例、`test_ddp` 的 overlap 等价性与设备态拒绝面全部保持通过——
`overlap=False` 与设备态拒绝行为均未改动。

**未声称 / 剩余**

- 设备态与反向重叠仍未实现，P2P 仍未实现；两者都需要多卡真机，当前机器不可
  验证。
- 主机态桶间并发的数字（2.30×）是 5.302 在主机内存上的实测，不是设备态或
  跨机网络的数字。
- 真机债不变：`ncclCommInitAll` 多卡归约与 `from_env` uid 分发仍只有 mock
  覆盖（5.331 未声称）。


### 5.339 手工驱动序列的落盘时机：`finish` 释放前落盘与"真写入"计数

5.337 的落盘只挂在 `generate` 返回前，未声称里留了一条："只有 `generate`
结束才落盘：`feed` / `prefill` 手工驱动的序列没有'结束'信号"。这条其实不
准确：`finish` 就是结束信号——它释放页、删掉 `_tokens[rid]`，正是"这条序列
到此为止"的时刻。本节把落盘补在 `finish` 上，并顺带把 `disk_stores` 的计数
改成"真正写入了条目"。

**落盘点：`finish` 释放前。**

- `PagedGenerator.finish` 现在先 `_persist(rid)`，再 `scheduler.finish(rid)`、
  删 `_tokens[rid]`。顺序不能反：`_persist` 要从池里 `gather` 每层 K/V，
  `scheduler.finish` 之后页已经回收。
- 只用驻留态序列落盘。换出态（`suspend(evict=True)` 之后）的 KV 在主机副本
  里，池里 `gather` 取不到；为一次释放先 `resume` 换入再落盘，等于为一个马上
  要释放的序列多付一次换入。所以 `finish` 里先问 `is_swapped(rid)`，是就跳过
  ——序列照常释放，只是不产生条目。
- 空序列（`submit` 之后没喂过 token）不落盘：`_tokens[rid]` 为空，
  `_persist` 直接返回，不留 `tokens=[]` 的废条目。
- 落盘失败（`OSError` 等）时 `finish` 抛错、序列仍驻留，调用方可以清盘后
  重试；不吞异常，与 `generate` 的 `_persist` 一致。

**为什么不在 `feed` / `prefill` 里落盘。** 条目粒度是"一条已解码的完整
前缀 + 每层连续 K/V"（5.337），每喂一个 token 就存一条，会在盘上留下一串
几乎重复的条目（长度 1..T），既浪费空间也把 LRU 冲掉。`finish` 是唯一确定
"不再变长"的点，与 `generate` 结束时落盘同一粒度。

**计数修正：`disk_stores` 是"真写入"。** 5.337 里 `_persist` 每调用一次就让
`disk_stores += 1`，而 `DiskPrefixCache.store` 对"同 key 且等长"的条目只刷新
mtime、不重写。`generate` 之后紧接 `finish` 会连着走两次 `_persist`，旧写法
计成 2，盘上却只有 1 条。现在 `DiskPrefixCache` 暴露只读属性 `stores`
（`_stores`，只在真正新增 / 重写时自增），`_persist` 比较调用前后的值，只有
真写入才计入生成侧统计。`stats()` 里的 `stores` 与它同源，语义一致。

**验证。** `TestGeneratorDiskCache` 新增 5 条：

1. `test_finish_persists_manual_sequence`：`submit` + `prefill` 后 `finish`，
   `disk_stores == 1`、目录里 1 条；新实例以"prompt + 1 个 token"生成，
   `disk_hits == 1`、`disk_reused_tokens == len(prompt)`，且输出与
   `_dense_greedy` 逐 token 相等。
2. `test_finish_after_generate_does_not_rewrite`：`generate` 后 `finish`，
   `disk_stores` 仍为 1（同 key 等长只刷新 mtime），条目数仍为 1。
3. `test_finish_skips_persist_when_swapped`：`suspend(evict=True)` 后 `finish`
   不落盘、不抛，序列正常移除。
4. `test_finish_without_tokens_stores_nothing`：`submit` 后直接 `finish`，
   不产生条目。
5. `test_finish_respects_disabled_prefix_cache`：`prefix_cache=False` 时
   `finish` 不落盘（该开关同时关掉查盘与落盘，语义与 5.337 一致）。

`py -3.14 -m ruff check aetherflow tests` 通过；
`py -3.14 -m unittest discover -s tests -t .` 全量 3387 条通过（5.337 的 3382
加本节 5 条）。

**未声称 / 剩余**

- 换出态序列在 `finish` 时不落盘；要落盘必须显式 `resume` 后再 `finish`。
- 裸 `feed` / `prefill` 之后不调 `finish` 仍不会落盘——落盘需要"结束"这个
  触发点，没有隐式判定"用户不再续写"的办法。
- 仍是主机态：`gather` 与 `np.savez` 的时延没有真机数据；条目大小随
  `L × T` 线性增长，靠 `max_bytes` 兜底（5.337 的约束不变）。

### 5.340 torch.export 前端接上 bool：逻辑组合、`& | ~` 与显式 cast

**缺口。** 5.330 把 IR 侧的 bool 补齐了（`logical_*`、`reduce_all/any`、`cast`
双向、设备端 1 字节存储），但导入侧只把 bool 当"已经存在的掩码输入"搬运：比较
算子已产出 bool，`where` / `masked_fill` 已消费 bool，而**组合**掩码的写法在
导出图里全部落在拒绝面上——`torch.logical_and/or/not` 没有映射，
`mask = (x > 0) & (y > 0)` 这种更常见的写法在 FX 图里是
`aten.__and__.Tensor`，同样没有映射。结果是：带 mask 的模型只要把两个条件
与在一起，就在导入时断掉。5.330 的"未声称"把 bool 的 `to(dtype=...)` 也记成
"尚未映射"；实际入口是通用的 `aten.to.dtype` / `aten._to_copy` /
`aten.type_as`，没有专门拦 bool——它能用，但既没有测试也没有声明。

**实现。** 只改前端，不加 IR 算子、不改设备端。

1. `aten.logical_and` / `aten.logical_or` / `aten.logical_not` 映射到 IR 同名
   算子；`& | ~` 与 `aten.bitwise_and/or/not` 一并归到这三个算子（bool 上位
   运算与逻辑运算逐位等价）。家族名按 `target.rsplit(".", 1)[0]` 取，
   `default` / `Tensor` 两种写法都能接。
2. 输入校验与 5.330 同口径，但**区分两类非 bool 输入**：
   - 显式 `logical_*`：torch 允许浮点 / 整数（"非零即真"），而 IR 在 5.330
     定的口径是"只收 bool"，掩码必须由显式比较落成。前端沿用这条口径，抛
     `UnsupportedOperatorError` 并给出 `ne(x, 0)` 的写法；不替调用方插入
     隐式转换——前端只翻译，语义边界留在模型里可见。
   - `& | ~` / `bitwise_*`：整数上是**位运算**（`2 & 4 == 0`），与逻辑与
     不等价，IR 没有位运算算子，同样显式拒绝；浮点在 torch 侧本就报错，
     到不了前端。
3. bool cast 不改代码：三条入口共用 `_to_dtype`，`bool` 已在 `_TORCH_DTYPE`
   与兼容矩阵的 dtypes 里，落到 IR 的 `cast`。本次把它纳入验收（f32 ↔ bool
   往返、bool 输出序列化），并把 5.330 的"未声称"改正为"已接、当时缺测试"。
4. 边界仍是**导入与参考执行**：设备端 bool 逻辑组合的方言选择是 5.330 的
   源码级交付，本节没有新增真机数据。

**验证。** `tests/test_frontend.py` 新增 6 条：

1. `test_logical_ops_and_operator_sugar`：`(a & b) | ~a` 与
   `logical_or(logical_and(a, b), logical_not(a))` 两种写法导入后与 eager
   逐元素相等（含 `(2,1) × (1,3)` 广播），图里分别是 `logical_and` /
   `logical_or` / `logical_not`。
2. `test_compare_and_combine_then_where`：`mask = (x > 0) & (y > 0)` 后接
   `where`，端到端与 eager 相等，图里出现 `logical_and`。
3. `test_bool_cast_roundtrip`：`x.bool().float()` 与 eager 逐位相等，输出是
   f32，图里恰好两个 `cast`。
4. `test_bool_graph_serializes`：bool 输出图 `dumps` / `loads` 往返后算子序列
   不变，执行结果仍是 `np.bool_` 且与往返前逐位相等。
5. `test_float_logical_is_rejected`：`torch.logical_and(f32, f32)` 在 torch 里
   合法，导入时必须抛 `UnsupportedOperatorError`。
6. `test_integer_bitwise_is_rejected`：`i64 & i64` 导出后导入必须抛
   `UnsupportedOperatorError`。

`py -3.14 -m ruff check aetherflow tests` 通过；前端 96 条通过（2 条 GPU
跳过），全量 3393 条通过 / 461 跳过（5.339 的 3387 加本节 6 条）。

**未声称。**

- 没有 `logical_xor`：IR 没有该算子，torch 会导出
  `aten.logical_xor.default`，仍显式拒绝。
- 整数位运算（`bitwise_and/or/not`、整数上的 `& | ~`）不支持：语义与逻辑
  运算不同，IR 没有位运算算子。
- 浮点 `logical_*` 不自动降级成 `ne(x, 0)`：前端只翻译，不改写语义边界。
- 没有 `count_nonzero`；计数先 `cast` 到 f32（5.330 的约束不变）。
- 设备端 bool 归约仍无内核（`af_add_q` 仍是 stub，`reduce_all/any` 不在
  `GPU_OPS`）；bool 逻辑组合的设备路径仍是 5.330 的源码级交付，V100 未回归。


















