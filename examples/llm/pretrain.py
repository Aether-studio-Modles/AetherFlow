"""字节级预训练示例：从零训练一个 Qwen2 / Llama 结构的语言模型。

验收标准是 **loss 明显下降**，不是"训出可用模型"：数据流、前向、反向、
优化器任何一环错了，小语料上损失都会停在高位或发散。默认配置是
Qwen2.5-0.5B 的缩小版（同一套 RMSNorm / RoPE / GQA / SwiGLU），
CPU 上几十秒跑完。

用法（在仓库根目录）：

    python examples/llm/pretrain.py
    python examples/llm/pretrain.py --steps 30 --lr 3e-3
    python examples/llm/pretrain.py --preset qwen2.5-0.5b --device cuda

``--preset qwen2.5-0.5b`` 是官方超参：参数约 5 亿，主机内存/显存要够。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from aetherflow.backends.reference_backend import ReferenceBackend
from aetherflow.ir.types import f32, i32
from aetherflow.models import ByteTokenizer, LlamaSpec, build_llama, masked_lm_loss_indexed
from aetherflow.train import Adam, ParameterStore, Trainer, save_checkpoint


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=60, help="训练步数")
    parser.add_argument("--lr", type=float, default=2e-3, help="Adam 学习率")
    parser.add_argument("--seq", type=int, default=64, help="上下文长度（静态）")
    parser.add_argument("--batch", type=int, default=4, help="批大小")
    parser.add_argument("--seed", type=int, default=0, help="参数初始化种子")
    parser.add_argument(
        "--preset",
        choices=("tiny", "qwen2.5-0.5b"),
        default="tiny",
        help="tiny = 缩小版（默认，CPU 可跑）；qwen2.5-0.5b = 官方超参",
    )
    parser.add_argument(
        "--corpus",
        default=str(REPO / "DESIGN.md"),
        help="语料文件（默认用本仓库的 DESIGN.md，字节级编码）",
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu", help="执行后端"
    )
    parser.add_argument(
        "--save", default=None, help="训练结束后把权重写成 .npz（可接 sft.py --init）"
    )
    return parser.parse_args(argv)


def _spec(args: argparse.Namespace, vocab_size: int) -> LlamaSpec:
    if args.preset == "qwen2.5-0.5b":
        return LlamaSpec.qwen2_5_0_5b(seq=args.seq, batch=args.batch)
    return LlamaSpec(
        vocab_size=vocab_size,
        d_model=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=256,
        seq=args.seq,
        batch=args.batch,
    )


def _backend(device: str):
    if device == "cuda":
        from aetherflow.backends.cuda.backend import CudaBackend

        return CudaBackend()
    return ReferenceBackend()


def _blocks(data: np.ndarray, spec: LlamaSpec) -> np.ndarray:
    """把连续 token 流切成 ``(blocks, batch, seq+1)``：每行多一个 token 做标签。"""
    span = spec.batch * (spec.seq + 1)
    usable = len(data) // span
    if usable < 1:
        raise SystemExit(
            f"语料只有 {len(data)} 个 token，一个 batch 需要 {span} 个；"
            "请换更长的语料或减小 --seq / --batch"
        )
    return data[: usable * span].reshape(usable, spec.batch, spec.seq + 1)


def _batch(blocks: np.ndarray, step: int, vocab_size: int) -> dict:
    chunk = blocks[step % len(blocks)]
    inputs = chunk[:, :-1]
    labels = chunk[:, 1:]
    batch, steps = inputs.shape
    # 整数标签是扁平的 ``(b*T + t)*V + y``：图里把 (B,T,V) 的 logits 拉直后查表。
    flat = ((np.arange(batch)[:, None] * steps + np.arange(steps)[None, :]) * vocab_size + labels)
    return {
        "tokens": inputs.astype(np.int32),
        "targets": flat.astype(np.int32),
        "mask": np.ones((batch, steps), dtype=np.float32),
    }


def main(argv=None) -> int:
    args = _parse_args(argv)
    tokenizer = ByteTokenizer()
    spec = _spec(args, tokenizer.vocab_size)

    graph, _ = build_llama(spec)
    logits = graph.outputs[0]
    targets = graph.input("targets", i32[spec.batch, spec.seq])
    mask = graph.input("mask", f32[spec.batch, spec.seq])
    graph.output(masked_lm_loss_indexed(graph, logits, targets, mask), "loss")

    store = ParameterStore.from_graph(
        graph, exclude=("tokens", "targets", "mask"), seed=args.seed
    )
    trainer = Trainer(
        graph,
        store,
        Adam(store.arrays(), lr=args.lr),
        backend=_backend(args.device),
        output="loss",
    )

    text = Path(args.corpus).read_text(encoding="utf-8")
    data = np.asarray(tokenizer.encode(text), dtype=np.int64)
    blocks = _blocks(data, spec)
    n_params = sum(int(np.prod(store[name].shape)) for name in store.names)

    print(
        f"preset={args.preset} device={args.device} params={n_params / 1e6:.2f}M "
        f"seq={spec.seq} batch={spec.batch} blocks={len(blocks)} tokens={len(data)}"
    )

    losses = []
    for step in range(args.steps):
        loss = trainer.step(_batch(blocks, step, spec.vocab_size))
        losses.append(loss)
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == args.steps:
            print(f"step {step + 1:4d}/{args.steps}  loss {loss:.4f}")

    window = min(5, len(losses))
    head = sum(losses[:window]) / window
    tail = sum(losses[-window:]) / window
    print(f"loss {head:.4f} -> {tail:.4f}  ({'下降' if tail < head else '未下降'})")

    if args.save:
        # 设备态训练期间 store 是上传时的旧值，先写回再存档。
        trainer.sync_to_host()
        save_checkpoint(
            args.save,
            store,
            meta={"stage": "pretrain", "steps": args.steps, "lr": args.lr},
        )
        print(f"权重已写入 {args.save}")
    return 0 if tail < head else 1


if __name__ == "__main__":
    raise SystemExit(main())
