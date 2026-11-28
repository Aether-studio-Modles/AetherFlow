"""模型级 IR 用例。

这里的每个构件都不是"运行时库"，而是对 IR 表达能力的检验：一个真实架构
若只能靠新增算子才能表达，说明 IR 的抽象层次选错了。当前覆盖六类结构：

- 前馈：``build_mlp``（MLP）
- 卷积：``build_cnn`` / ``conv_block`` / ``residual_block``（CNN / ResNet 基本块）
- 循环：``build_rnn``（RNN / LSTM / GRU 的时间展开）
- 注意力：``build_transformer_block``（单块，序列维符号化）、
  ``build_transformer``（多层 decoder-only LM）
- 状态空间：``build_mamba_block`` / ``build_mamba_model``（多层栈）+
  ``build_selective_scan``（``*_stateful`` 变体带状态出参与 reset）
- 脉冲：``build_snn`` / ``build_snn_network``（单层 / 多层 LIF）

共性约定：构件返回 ``(输入表, 输出)``，输入表里同时含数据输入（``x`` /
``ids`` / ``pos`` / ``mask``）与全部参数。训练时把数据输入名传给
``ParameterStore.from_graph(..., exclude=...)``，否则它们会被当成参数。
"""

from .cnn import CNNSpec, bn_conv_block, bn_infer, build_cnn, conv_block, residual_block
from .cached_attention import (
    PagedDecodeScheduler,
    PagedGenerator,
    build_cached_decoder_step,
    cached_forward,
)
from .lm import (
    LMSpec,
    LMRunner,
    QueueFullError,
    ServingEngine,
    build_language_model,
    build_language_model_stateful,
    generate,
    generate_batch,
    generate_beam,
    masked_lm_loss,
    masked_lm_loss_indexed,
    sample_next_token,
)
from .lm_bpe import BpeTokenizer, BpeTokenizerError
from .lm_text import ByteTokenizer, ChatTemplate
from .lm_unigram import UnigramTokenizer, UnigramTokenizerError
from .lm_wordpiece import WordPieceTokenizer, WordPieceTokenizerError
from .lm_tools import JsonGrammar, Tool, ToolRegistry, parse_tool_call
from .llama import (
    LlamaSpec,
    build_llama,
    llama_inv_freq,
    llama_param_shapes,
    llama_param_transposes,
    load_llama_weights,
)
from .mamba import (
    MambaModelSpec,
    MambaSpec,
    build_mamba_block,
    build_mamba_block_stateful,
    build_mamba_layers,
    build_mamba_model,
    build_mamba_model_stateful,
    causal_depthwise_conv1d,
    causal_depthwise_conv1d_stateful,
    decay_diagnostics,
    effective_half_lives,
    init_mamba_params,
)
from .mamba_reference import (
    from_official_checkpoint,
    map_official_params,
    official_selective_scan,
)
from .mlp import MLPSpec, build_mlp
from .quantize import (
    QUANT_MAX,
    QUANT_MIN_ASYMMETRIC,
    build_int8_linear,
    build_int8_linear_zp,
    build_int8a8_linear,
    build_int8a8_linear_zp,
    dequantize_weight,
    dequantize_weight_asymmetric,
    quantize_activation,
    quantize_activation_asymmetric,
    quantize_weight,
    quantize_weight_asymmetric,
)
from .moe import (
    EXPERT_PARALLEL_SUPPORTED,
    MoESpec,
    build_moe,
    require_expert_parallel,
)
from .rnn import RNNSpec, build_rnn
from .snn import (
    LIFSpec,
    SNNSpec,
    build_snn,
    build_snn_events,
    build_snn_network,
    constant_current_encode,
    lif_step,
    mean_rate,
)
from .ssm import build_selective_scan, build_selective_scan_stateful
from .speculative import DraftModel, SpeculativeDecoder, greedy_draft
from .transformer import (
    BlockSpec,
    TransformerSpec,
    build_transformer,
    build_transformer_block,
    causal_mask_bias,
)

__all__ = [
    "BlockSpec",
    "build_transformer_block",
    "TransformerSpec",
    "build_transformer",
    "causal_mask_bias",
    "MLPSpec",
    "build_mlp",
    "QUANT_MAX",
    "QUANT_MIN_ASYMMETRIC",
    "build_int8_linear",
    "build_int8_linear_zp",
    "build_int8a8_linear",
    "build_int8a8_linear_zp",
    "dequantize_weight",
    "dequantize_weight_asymmetric",
    "quantize_activation",
    "quantize_activation_asymmetric",
    "quantize_weight",
    "quantize_weight_asymmetric",
    "MoESpec",
    "build_moe",
    "require_expert_parallel",
    "EXPERT_PARALLEL_SUPPORTED",
    "CNNSpec",
    "build_cnn",
    "conv_block",
    "bn_conv_block",
    "bn_infer",
    "residual_block",
    "RNNSpec",
    "build_rnn",
    "MambaSpec",
    "MambaModelSpec",
    "build_mamba_block",
    "build_mamba_block_stateful",
    "build_mamba_layers",
    "build_mamba_model",
    "build_mamba_model_stateful",
    "causal_depthwise_conv1d",
    "causal_depthwise_conv1d_stateful",
    "init_mamba_params",
    "effective_half_lives",
    "decay_diagnostics",
    "official_selective_scan",
    "map_official_params",
    "from_official_checkpoint",
    "LIFSpec",
    "SNNSpec",
    "lif_step",
    "build_snn",
    "build_snn_events",
    "build_snn_network",
    "constant_current_encode",
    "mean_rate",
    "build_selective_scan",
    "build_selective_scan_stateful",
    "LMSpec",
    "build_language_model",
    "build_language_model_stateful",
    "masked_lm_loss",
    "masked_lm_loss_indexed",
    "sample_next_token",
    "generate",
    "generate_batch",
    "generate_beam",
    "LMRunner",
    "ServingEngine",
    "QueueFullError",
    "ByteTokenizer",
    "ChatTemplate",
    "BpeTokenizer",
    "BpeTokenizerError",
    "WordPieceTokenizer",
    "WordPieceTokenizerError",
    "UnigramTokenizer",
    "UnigramTokenizerError",
    "JsonGrammar",
    "Tool",
    "ToolRegistry",
    "parse_tool_call",
    "build_cached_decoder_step",
    "cached_forward",
    "PagedDecodeScheduler",
    "PagedGenerator",
    "DraftModel",
    "SpeculativeDecoder",
    "greedy_draft",
    "LlamaSpec",
    "build_llama",
    "load_llama_weights",
    "llama_inv_freq",
    "llama_param_shapes",
    "llama_param_transposes",
]
