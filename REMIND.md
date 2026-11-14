# REMIND

规范与历史全部在 [DESIGN.md](DESIGN.md)；这里只记"再进来时先要知道的事"。

## 验证命令

```powershell
# 本地全量
py -3.14 -m unittest discover -s tests -t .

# WSL HIP 全量
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/g/Cloumi/aetherflow && HSA_ENABLE_DXG_DETECTION=1 /opt/rocm-env/bin/python -m unittest discover -s tests -t ."
```

当前基线：本地 2336 个用例 / 412 跳过，WSL HIP 2336 个用例 / 371 跳过。

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

## 临时脚本

调试用的临时脚本（`_tmp_*.py` 等）用完即删，关键结论、问题与解法写进
DESIGN 的对应小节，凭证不落盘。
