# REMIND

规范与历史全部在 [DESIGN.md](DESIGN.md)；这里只记"再进来时先要知道的事"。

## 验证命令

```powershell
# 本地全量
py -3.14 -m unittest discover -s tests -t .

# WSL HIP 全量
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/g/Cloumi/aetherflow && HSA_ENABLE_DXG_DETECTION=1 /opt/rocm-env/bin/python -m unittest discover -s tests -t ."
```

当前基线：本地 2242 个用例 / 410 跳过，WSL HIP 2242 个用例 / 369 跳过。

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

## 临时脚本

调试用的临时脚本（`_tmp_*.py` 等）用完即删，关键结论、问题与解法写进
DESIGN 的对应小节，凭证不落盘。
