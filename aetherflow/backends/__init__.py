"""后端插件。"""

from .base import Backend, BackendCapabilityError, unsupported_ops
from .capabilities import (
    BackendCapabilities,
    OpCapability,
    capability_gaps,
    to_capabilities,
)
from .discovery import (
    BackendUnavailableError,
    get_backend,
    register_backend,
    registered_backends,
    unregister_backend,
)
from .hardware import (
    HardwarePackageError,
    HardwareRecord,
    install_hardware,
    installed_hardware,
    search_roots,
    uninstall_hardware,
)
from .contracts import (
    CONTRACTS,
    CodegenContract,
    DeviceContract,
    DeviceInfo,
    MemoryContract,
    MemoryStats,
    TuningContract,
    implemented_contracts,
    supports,
)
from .reference_backend import ReferenceBackend
from .registry import KernelRegistry

__all__ = [
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
    "unsupported_ops",
    "ReferenceBackend",
    "KernelRegistry",
    "CONTRACTS",
    "CodegenContract",
    "DeviceContract",
    "DeviceInfo",
    "MemoryContract",
    "MemoryStats",
    "TuningContract",
    "implemented_contracts",
    "supports",
]
