"""AetherFlow：统一深度学习基础设施。

架构约定（写在导入处，因为它是所有模块的公共前提）：

1. 前端与后端之间只有一份契约——IR。任何后端都不得读取前端对象。
2. 每个算子都有一份不依赖加速库的参考语义，它是跨后端对齐的唯一真值。
3. 每个节点都带可移植性标注，跨后端共享的优化路径只允许 PORTABLE 节点。

当前实现范围：高层语义 IR + JSON 序列化与版本治理 + 参考语义执行器 + 参考后端 +
verifier + 中层优化 pass + torch.export 前端适配 + CUDA 后端 + 后端一致性套件 +
插件契约分层（准入面 + 可选契约）+ 算子注册与调度（含多实现按代价选择）+
插件算子扩展面 + 自动微分（语义 IR 层的反向展开）+
训练闭环（参数容器 + SGD/Adam + 图内损失）+ 模型构件（MLP / CNN / RNN(LSTM/GRU) /
多层 Transformer / Mamba block / 多层 SNN）+ eager 构图前端（语法糖，产物仍是静态 IR）
+ SNN 的稀疏事件表示与事件内存池（`runtime/events.py`，数据侧，不进 IR）
+ 自动调优（参数空间 + 网格搜索 + 配置库）+ 运行时数据上报 + 只读智能体查询面
+ zip 硬件包（`backends/hardware.py`，manifest 校验 + 安全解压 + 装完即用 +
  内容摘要 / 依赖声明 / Ed25519 来源签名）+ 动态执行（`dynamic.DynamicRunner`，
  按输入形状建图并复用；声明动态轴后一次构图多次重放，一次 run 内仍是静态图）
  + 部署包（`deploy.py`，图 JSON + 权重 + 清单校验和，加载即校验）
  + 激活重算（`train/recompute.py`，分段前向只留边界，反向逐段重放）。
尚未实现：完整的 define-by-run 执行（逐算子即时求值）、强化学习式的配置更新、
硬件包的公钥分发与撤销链。
"""

import sys as _sys

# 运行期版本守卫：pyproject 的 requires-python 只约束安装器，源码目录直接导入、
# 或旧解释器复用已装好的 site-packages 时都不会经过它。实测 3.8 能导入本包的一部分
# 模块，但行为没有保证——"能导入、半可用"正是本项目明确拒绝的形态，所以在包入口
# 显式拦一次，让错误落在版本上，而不是几十层调用之后。
if _sys.version_info < (3, 10):  # pragma: no cover - 老解释器上无法执行测试
    raise RuntimeError(
        "AetherFlow 需要 Python 3.10 及以上，当前解释器是 "
        f"{_sys.version.split()[0]}。请升级解释器，或用 py -3.14 之类的启动器"
        "选择受支持的版本。"
    )

from .agent_api import (
    AgentQueryError,
    describe_backend,
    describe_backends,
    describe_operator,
    performance_summary,
    recommend_config,
    snapshot,
)
from .backends.base import Backend, BackendCapabilityError, unsupported_ops
from .backends.capabilities import (
    BackendCapabilities,
    OpCapability,
    capability_gaps,
    to_capabilities,
)
from .backends.discovery import (
    BackendUnavailableError,
    get_backend,
    register_backend,
    registered_backends,
    unregister_backend,
)
from .backends.hardware import (
    HardwarePackageError,
    HardwareRecord,
    install_hardware,
    installed_hardware,
    search_roots,
    uninstall_hardware,
)
from .backends.contracts import (
    CONTRACTS,
    AllocatorContract,
    CodegenContract,
    DeviceContract,
    DeviceInfo,
    DeviceManagerContract,
    MemoryContract,
    MemoryStats,
    StreamContract,
    TuningContract,
    implemented_contracts,
    supports,
)
from .backends.reference_backend import ReferenceBackend
from .backends.registry import KernelRegistry
from .conformance import Case, CaseResult, CheckResult, Report, default_cases, run_conformance
from .deploy import BundleError, Deployment, load_bundle, save_bundle
from .dynamic import DynamicRunner
from .ir import (
    IR_VERSION,
    UNREPLAYABLE_EFFECTS,
    Effect,
    GradCtx,
    Graph,
    IRVersionError,
    MemoryBudget,
    NoGradientError,
    Op,
    Portability,
    TensorType,
    backward,
    hessian_vector_product,
    Conditional,
    ControlFlowError,
    ControlRegion,
    GraphRegion,
    WhileLoop,
    from_spec,
    to_spec,
    bf16,
    boolean,
    c128,
    c64,
    dumps,
    f16,
    f32,
    f64,
    from_dict,
    i8,
    i32,
    loads,
    migrate,
    migration_path,
    plan_memory,
    register_migration,
    registered_migrations,
    to_dict,
    unregister_migration,
)
from .ir.verifier import Issue, VerifierError, require, verify, verify_portable_only
from .fuzz import FuzzCase, FuzzFailure, FuzzReport, random_case, run_fuzz
from .plugin import BUILTIN_OPS, plugin_ops, register_operator, unregister_operator
from .runtime.reference import ReferenceExecutor
from .runtime.telemetry import RunRecord, TelemetryRecorder, load_records
from .tuning import (
    AllRejectedError,
    ConfigStore,
    TuningRecord,
    TuningResult,
    TuningSpace,
    tune,
)

__version__ = "0.1.0"
# 后端 / 硬件包契约的主版本。它与包版本分开：包可以频繁发版，但只要
# Backend 协议、IR 序列化格式与方言数据结构没变，aetherflow_api 就保持 1，
# 已安装的 zip 硬件包不需要跟着重发。
__api_version__ = "1"

__all__ = [
    "Graph",
    "DynamicRunner",
    "BundleError",
    "Deployment",
    "load_bundle",
    "save_bundle",
    "IR_VERSION",
    "IRVersionError",
    "migrate",
    "migration_path",
    "register_migration",
    "unregister_migration",
    "registered_migrations",
    "to_dict",
    "from_dict",
    "dumps",
    "loads",
    "TensorType",
    "MemoryBudget",
    "plan_memory",
    "Op",
    "Effect",
    "UNREPLAYABLE_EFFECTS",
    "Portability",
    "backward",
    "hessian_vector_product",
    "ControlFlowError",
    "ControlRegion",
    "GraphRegion",
    "Conditional",
    "WhileLoop",
    "to_spec",
    "from_spec",
    "GradCtx",
    "NoGradientError",
    "f16",
    "bf16",
    "c64",
    "c128",
    "f32",
    "f64",
    "i8",
    "i32",
    "boolean",
    "Backend",
    "BackendCapabilityError",
    "BackendCapabilities",
    "OpCapability",
    "capability_gaps",
    "to_capabilities",
    "BackendUnavailableError",
    "get_backend",
    "register_backend",
    "registered_backends",
    "unregister_backend",
    "HardwarePackageError",
    "HardwareRecord",
    "install_hardware",
    "uninstall_hardware",
    "installed_hardware",
    "search_roots",
    "ReferenceBackend",
    "KernelRegistry",
    "unsupported_ops",
    "CONTRACTS",
    "DeviceContract",
    "DeviceManagerContract",
    "DeviceInfo",
    "CodegenContract",
    "TuningContract",
    "MemoryContract",
    "MemoryStats",
    "StreamContract",
    "AllocatorContract",
    "implemented_contracts",
    "supports",
    "Case",
    "CaseResult",
    "CheckResult",
    "Report",
    "default_cases",
    "run_conformance",
    "Issue",
    "VerifierError",
    "verify",
    "verify_portable_only",
    "require",
    "BUILTIN_OPS",
    "plugin_ops",
    "register_operator",
    "unregister_operator",
    "AllRejectedError",
    "ConfigStore",
    "TuningRecord",
    "TuningResult",
    "TuningSpace",
    "tune",
    "RunRecord",
    "TelemetryRecorder",
    "load_records",
    "ReferenceExecutor",
    "FuzzCase",
    "FuzzFailure",
    "FuzzReport",
    "random_case",
    "run_fuzz",
    "AgentQueryError",
    "describe_backend",
    "describe_backends",
    "describe_operator",
    "performance_summary",
    "recommend_config",
    "snapshot",
    "__version__",
    "__api_version__",
]
