"""前向示例：把 Llama / Qwen2 结构的模型跑一次，看输出长什么样。

两种用法：

    # 随机初始化的缩小版（默认，CPU 秒级）
    python examples/llm/forward.py

    # 真实 Qwen2.5-0.5B 权重（目录里要有 config.json 与 *.safetensors）
    python examples/llm/forward.py --weights /root/models/Qwen2.5-0.5B --ids 9707,1879

``--weights`` 下**不做文本编码**：真实模型的词表是 15 万级的 BPE，本仓库不带
BPE 分词器，直接给 token id 才能确定"喂进去的是什么"。默认模式用仓库自带的
字节级 ``ByteTokenizer``，输入文本与输出的 top-1 token 都能解回字符串。

``--device cuda`` 走设备后端（需要可用的 CUDA 环境），其余参数不变。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from aetherflow.backends.reference_backend import ReferenceBackend
from aetherflow.models import ByteTokenizer, LlamaSpec, build_llama, load_llama_weights
from aetherflow.train import ParameterStore
from aetherflow.weights import SafeTensorsFile, load_safetensors_dir


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=None, help="HF 权重目录（config.json + *.safetensors）")
    parser.add_argument("--text", default="aetherflow: a small training framework.",
                        help="tiny 模式下要前向的文本（会被截断/补齐到 --seq）")
    parser.add_argument("--ids", default="0", help="--weights 下要前向的 token id，逗号分隔")
    parser.add_argument("--seq", type=int, default=32, help="tiny 模式的上下文长度")
    parser.add_argument("--seed", type=int, default=0, help="tiny 模式的随机初始化种子")
    parser.add_argument("--top-k", type=int, default=5, help="最后一个位置打印几个候选")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="执行后端")
    return parser.parse_args(argv)


def _backend(device: str):
    if device == "cuda":
        from aetherflow.backends.cuda.backend import CudaBackend

        return CudaBackend()
    return ReferenceBackend()


def _has_qkv_bias(folder: Path) -> bool:
    """按权重文件判断 q/k/v 是否带 bias（config 的 ``attention_bias`` 不可信）。"""
    for path in sorted(folder.glob("*.safetensors")):
        with SafeTensorsFile(path) as fh:
            if any(name.endswith("self_attn.q_proj.bias") for name in fh.names):
                return True
    return False


def _load_hf(folder: Path, tokens: np.ndarray) -> tuple[LlamaSpec, dict]:
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    spec = LlamaSpec.from_hf_config(
        config, seq=int(tokens.shape[1]), qkv_bias=_has_qkv_bias(folder)
    )
    return spec, load_safetensors_dir(folder)


def _tiny_tokens(tokenizer: ByteTokenizer, text: str, seq: int) -> np.ndarray:
    ids = tokenizer.encode(text)[:seq]
    ids += [tokenizer.eos_id] * (seq - len(ids))
    return np.asarray(ids, dtype=np.int32)[None, :]


def _show(tokenizer: ByteTokenizer | None, token_id: int) -> str:
    if tokenizer is None:
        return str(token_id)
    try:
        return repr(tokenizer.decode([token_id]))
    except ValueError:
        return f"<byte 0x{token_id - 2:02x}>"


def main(argv=None) -> int:
    args = _parse_args(argv)
    tokenizer: ByteTokenizer | None = None
    tensors: dict | None = None

    if args.weights is None:
        tokenizer = ByteTokenizer()
        tokens = _tiny_tokens(tokenizer, args.text, args.seq)
        spec = LlamaSpec(
            vocab_size=tokenizer.vocab_size,
            d_model=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            d_ff=256,
            seq=int(tokens.shape[1]),
            batch=1,
        )
    else:
        ids = [int(part) for part in args.ids.split(",") if part.strip()]
        tokens = np.asarray(ids, dtype=np.int32)[None, :]
        spec, tensors = _load_hf(Path(args.weights), tokens)

    graph, _ = build_llama(spec)
    store = ParameterStore.from_graph(graph, exclude=("tokens",), seed=args.seed)
    if tensors is not None:
        unused = load_llama_weights(spec, store, tensors)
        if unused:
            print(f"未使用的张量 {len(unused)} 个（例如 {unused[0]}）")

    logits = _backend(args.device).execute(graph, store.feed({"tokens": tokens}))["logits"]

    last = logits[0, -1].astype(np.float64)
    probs = np.exp(last - last.max())
    probs /= probs.sum()
    order = np.argsort(-probs)[: args.top_k]
    print(
        f"logits {logits.shape}  dtype={logits.dtype}  "
        f"tokens={tokens.shape[1]}  device={args.device}"
    )
    for rank, token_id in enumerate(order, start=1):
        print(
            f"  top{rank}  id={int(token_id):<7d} p={probs[token_id]:.4f}  "
            f"{_show(tokenizer, int(token_id))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
