"""自动调优：参数空间 + 网格搜索 + 配置库。

与后端无关。后端要做的是三件事（形状由插件自己决定）：

```python
backend.tuning_space(op)        # -> TuningSpace，op 不可调时明确报错
backend.set_tuning(op, config)  # 应用配置，非法值明确拒绝
backend.clear_tuning()          # 回到默认
```

搜索本身用 ``tune(space, measure)``（串行：每个候选连续测若干次）或
``tune_paired(space, measure)``（交替 A/B：每轮连测全部候选、逐轮反转顺序，
并用配对差的 bootstrap 区间判定差异是否超出噪声；传入 ``max_round_seconds``
时，轮内跨度超限的采集会被判 ``long-round``、保留默认；``warmup_rounds``
先连测若干轮并丢弃样本，用来跑掉首轮冷启动；``tune_staged`` 把大空间按
"先扫一个键、再扫剩下的键"分两段，每段单独判定，最后用一次两候选对照回答
"整条搜索有没有让默认变快"；``tune_replicated`` 把同一空间独立采若干次，再按
``vote`` 合并（``all`` 要求每次同向，``majority`` 允许少数次不同、需 repeats>=3）；
推荐后者，理由见
``paired.py`` 与 DESIGN 5.63 / 5.140～5.146）。配置用 ``ConfigStore`` 落盘，
键是 ``backend + device + op + signature``。这一层**不执行图**、不 import 任何
加速库，因此本地无 GPU 也能完整测试搜索与库的正确性。

不含强化学习。配置库的更新来自"实测更快的配置覆盖旧记录"这一条规则；
没有策略网络、没有奖励信号，也没有在线探索——那些要先有可信的测量与
第二个后端才有意义。
"""

from .cost import (
    M40_LIKE,
    V100_LIKE,
    CostBreakdown,
    MachineModel,
    RankedTuningResult,
    conv_tile_cost,
    cost_ranked_tune,
    elementwise_cost,
    rank_configs,
)
from .paired import (
    ORDER_ALTERNATE,
    ORDER_SHUFFLE,
    VOTE_ALL,
    VOTE_MAJORITY,
    PairedCandidate,
    PairedResult,
    tune_paired,
    tune_replicated,
    tune_staged,
)
from .policy import SchedulePolicy
from .search import (
    AllRejectedError,
    CandidateResult,
    TuningResult,
    measure_configs,
    tune,
)
from .signature import block_signature, conv2d_signature
from .space import TuningSpace, canonical_signature
from .store import STORE_VERSION, ConfigStore, TuningRecord

__all__ = [
    "AllRejectedError",
    "CandidateResult",
    "ConfigStore",
    "CostBreakdown",
    "M40_LIKE",
    "MachineModel",
    "ORDER_ALTERNATE",
    "ORDER_SHUFFLE",
    "PairedCandidate",
    "PairedResult",
    "RankedTuningResult",
    "STORE_VERSION",
    "SchedulePolicy",
    "TuningRecord",
    "TuningResult",
    "TuningSpace",
    "V100_LIKE",
    "VOTE_ALL",
    "VOTE_MAJORITY",
    "block_signature",
    "canonical_signature",
    "conv2d_signature",
    "conv_tile_cost",
    "cost_ranked_tune",
    "elementwise_cost",
    "measure_configs",
    "rank_configs",
    "tune",
    "tune_paired",
    "tune_replicated",
    "tune_staged",
]
