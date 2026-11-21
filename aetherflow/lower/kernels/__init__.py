"""参考内核的 L2 描述。

``ew_binary`` 是最简单的并行映射（网格步进 + 一维索引），``reduce_sum_last``
引入归约、共享内存与提前退出；``ew_unary`` / ``ew_step`` 覆盖三元条件与
一元负号，``ew_broadcast`` 覆盖定长数组与倒序循环，``softmax_last`` 把两段
归约与三段重扫合在同一个内核里。这些合起来正好覆盖 L2 目前需要的全部节点，
也正好对应 ``cuda/codegen.py`` 里的手写模板，便于用"去空白后逐字相等"验收。

matmul 不在这里：它走 cuBLAS / hipBLAS，没有内核模板，也就没有可下降的内核。
线性代数的可移植性由库契约保证，不是 L2 的职责。
"""

from .common import BINARY_OP as BINARY_OP, grid_stride
from .elementwise import (
    ew_binary,
    ew_unary,
    ew_cast as ew_cast,
    ew_step,
    ew_clamp as ew_clamp,
    ew_mod_int as ew_mod_int,
    silu_gate,
    ew_broadcast,
    UNARY_EXPR as UNARY_EXPR,
)
from .reduce import (
    reduce_sum_last,
    reduce_max_last,
    reduce_mean_last,
    reduce_axis,
    reduce_axis_tiled,
    softmax_axis as softmax_axis,
    argmax_last,
    softmax_last,
)
from .norm import (
    layernorm_last,
    layernorm_residual_last,
    rmsnorm_last,
    batch_norm,
    batch_norm_stats,
)
from .misc import concat_copy, dropout_mask
from .sparse import (
    spike_offsets,
    spike_csr_serial,
    spike_csr,
    sparse_linear,
    sparse_linear_weight_grad,
    sparse_linear_values_grad as sparse_linear_values_grad,
    spike_linear,
    spike_linear_weight_grad,
    _ROW_COLS as _ROW_COLS,
    _S_ROWS as _S_ROWS,
    _T_WIDTH as _T_WIDTH,
)
from .scan import (
    scan_linear,
    scan_linear_backward,
    cumsum as cumsum,
    tril as tril,
    scan_linear_chunk_sum,
    scan_linear_chunk,
    scan_linear_chunk_sum_backward,
    scan_linear_chunk_backward,
    scan_linear_last,
    scan_linear_chunk_last,
    _T_ELEMS as _T_ELEMS,
    _C_ELEMS as _C_ELEMS,
)
from .pool import (
    maxpool2d,
    maxpool2d_backward,
    maxpool2d_indices,
    avgpool2d,
    avgpool2d_backward,
    adaptive_avg_pool2d,
    adaptive_avg_pool2d_backward,
)
from .conv import (
    conv2d_input_grad,
    conv2d_weight_grad,
    im2col,
    conv2d_tiled,
    conv2d_grouped,
    conv2d_implicit,
    winograd_input_transform,
    winograd_weight_transform,
    winograd_output_transform,
    winograd_fused,
)
from .attention import attention_fwd

__all__ = [
    "grid_stride",
    "ew_binary",
    "ew_unary",
    "ew_step",
    "ew_broadcast",
    "reduce_sum_last",
    "reduce_max_last",
    "reduce_mean_last",
    "reduce_axis",
    "reduce_axis_tiled",
    "argmax_last",
    "softmax_last",
    "layernorm_last",
    "layernorm_residual_last",
    "rmsnorm_last",
    "silu_gate",
    "batch_norm",
    "batch_norm_stats",
    "concat_copy",
    "dropout_mask",
    "spike_offsets",
    "spike_csr",
    "spike_csr_serial",
    "sparse_linear",
    "sparse_linear_weight_grad",
    "spike_linear",
    "spike_linear_weight_grad",
    "scan_linear",
    "scan_linear_backward",
    "scan_linear_chunk_sum",
    "scan_linear_chunk",
    "scan_linear_chunk_sum_backward",
    "scan_linear_chunk_backward",
    "scan_linear_last",
    "scan_linear_chunk_last",
    "conv2d_input_grad",
    "conv2d_weight_grad",
    "maxpool2d",
    "maxpool2d_backward",
    "maxpool2d_indices",
    "avgpool2d",
    "avgpool2d_backward",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool2d_backward",
    "im2col",
    "conv2d_tiled",
    "conv2d_grouped",
    "conv2d_implicit",
    "winograd_weight_transform",
    "winograd_input_transform",
    "winograd_output_transform",
    "winograd_fused",
    "attention_fwd",
]
