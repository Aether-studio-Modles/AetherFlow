# REMIND

规范与历史全部在 [DESIGN.md](DESIGN.md)；这里只记"再进来时先要知道的事"。

## 验证命令

```powershell
# 本地全量
py -3.14 -m unittest discover -s tests -t .

# WSL HIP 全量
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/g/Cloumi/aetherflow && HSA_ENABLE_DXG_DETECTION=1 /opt/rocm-env/bin/python -m unittest discover -s tests -t ."
```

当前基线：本地 2293 个用例 / 410 跳过，WSL HIP 2293 个用例 / 369 跳过。

## CUDA 真机

显卡已拆、NVIDIA 驱动未加载（`nvidia-smi` 无法通信、`lsmod` 里没有 nvidia 模块），
CUDA 侧只能做离线 / 伪 runtime 验证；涉及 `cufftExec*`、真机耗时与多卡切换的结论
本轮都没有复跑，恢复后再补。

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
  `DeviceCollectiveUnavailable`。
- 跨进程 ZeRO（5.222）只把优化器槽位按 rank 切分落地（npz + `__meta__`），不省显存、
  不做 all-gather；`load_shard` 靠片号 / 分片数 / 归属表三道校验兜底（优化器自身不查）。
- torch 兼容矩阵（5.223）是声明式区间（2.9–2.13，四个 dtype）；区间内非 verified 版本
  不保证回归覆盖，越界或 dtype 缺席显式抛 `UnsupportedTorchVersion` /
  `UnsupportedOperatorError`。
- torch 入口（5.224）的整数只有 i32：int64 收窄、越界值在执行器入口拒绝；`padding_idx`
  / `sparse` / `scale_grad_by_freq` 仍未实现（显式拒绝）；提升常量与 `get_attr` 常量落成
  `constant`，单常量上限 65536 个元素；`torch.arange` 之类仍是未支持算子。

## 临时脚本

调试用的临时脚本（`_tmp_*.py` 等）用完即删，关键结论、问题与解法写进
DESIGN 的对应小节，凭证不落盘。
