"""LoRA 示例：冻结基座，只训练低秩旁路。

数据、mask 与损失复用 ``sft.py`` 的口径（同一批指令对、只对回答段算 loss），
差别只在"谁参与训练"：

- 图里基座权重与 adapter 都是输入，前向把两者一起算进去；
- ``Trainer(trainable=adapter)`` 让反向只对 adapter 求导，冻结参数不算梯度；
- 优化器只持 adapter，``prepare_step`` 因此不会把基座当成"缺梯度"；
- ``lora_b`` 显式初始化为 0：训练前的损失与基座逐位相同，任何"初始化就跳"
  都能被第一行日志抓到。

    # 缩小版（默认，CPU 几十秒）：内置指令对 + ByteTokenizer
    python examples/llm/lora.py

    # 接全参 SFT 的存档当基座（sft.py --save sft.npz）
    python examples/llm/lora.py --init sft.npz

    # 真实 Qwen2.5-0.5B + 预编码样本（同 sft.py 的 --data-json 口径）
    python examples/llm/lora.py --weights /root/models/Qwen2.5-0.5B \
        --data-json pairs.json --device cuda

验收标准是**回答段 loss 明显下降，而基座一个 bit 都不变**：最后会把冻结参数
与训练前的副本逐位比较，这一步失败说明 ``trainable`` 没生效（例如反向仍在
对全部参数求导），比"loss 降了"更值得看。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (str(REPO), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from aetherflow.backends.reference_backend import ReferenceBackend
from aetherflow.ir.types import f32, i32
from aetherflow.models import (
    BpeTokenizer,
    ByteTokenizer,
    LlamaSpec,
    LoraSpec,
    build_llama,
    load_llama_weights,
    masked_lm_loss_indexed,
)
from aetherflow.train import Adam, ParameterStore, Trainer, load_checkpoint
from aetherflow.weights import load_safetensors_dir

# 同一份数据管线只维护一处：sft.py 里的编码、补边与 mask 口径。
from sft import _batch, _has_qkv_bias, _loss, _rows, _rows_from_json

_ALL_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20, help="训练步数")
    parser.add_argument(
        "--lr", type=float, default=None,
        help="adapter 的学习率（默认 tiny 2e-2 / 真实权重 2e-4；LoRA 通常比全参 SFT 大）",
    )
    parser.add_argument("--rank", type=int, default=8, help="低秩瓶颈宽度")
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="缩放分子（默认 2*rank，与 HF PEFT 的常见配置一致）",
    )
    parser.add_argument(
        "--targets", default="q_proj,v_proj",
        help=f"逗号分隔的投影名；'all' 表示全部 7 个线性层（{_ALL_TARGETS}）",
    )
    parser.add_argument("--seq", type=int, default=None, help="上下文长度（默认 tiny 256 / 真实 128）")
    parser.add_argument("--batch", type=int, default=1, help="批大小")
    parser.add_argument("--seed", type=int, default=0, help="adapter 初始化种子")
    parser.add_argument("--weights", default=None, help="HF 权重目录（config.json + *.safetensors）")
    parser.add_argument("--init", default=None, help="本仓库的 .npz 存档（pretrain.py / sft.py 的 --save 产物）")
    parser.add_argument(
        "--data-json",
        default=None,
        help="预编码样本：[{\"ids\": [...], \"answer_start\": n}, ...]",
    )
    parser.add_argument("--pad-id", type=int, default=0, help="--data-json 模式下的 pad id")
    parser.add_argument("--save", default=None, help="训练结束后把 adapter 写成 .npz（只含 lora_a / lora_b）")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="执行后端")
    return parser.parse_args(argv)


def _backend(device: str):
    if device == "cuda":
        from aetherflow.backends.cuda.backend import CudaBackend

        return CudaBackend()
    return ReferenceBackend()


def _lora_spec(args) -> LoraSpec:
    targets = (
        tuple(_ALL_TARGETS.split(",")) if args.targets == "all"
        else tuple(part.strip() for part in args.targets.split(",") if part.strip())
    )
    alpha = args.alpha if args.alpha is not None else 2.0 * args.rank
    return LoraSpec(rank=args.rank, alpha=alpha, targets=targets)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.weights and args.init:
        raise SystemExit("--weights 与 --init 只能给一个：它们都是基座权重来源")

    lora = _lora_spec(args)
    chatml = args.weights is not None
    tensors: dict | None = None
    tokenizer = None
    if args.weights:
        folder = Path(args.weights)
        seq = args.seq or 128
        lr = args.lr if args.lr is not None else 2e-4
        if args.data_json is None:
            tokenizer = BpeTokenizer.from_file(folder / "tokenizer.json")
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        spec = LlamaSpec.from_hf_config(
            config, seq=seq, batch=args.batch, qkv_bias=_has_qkv_bias(folder)
        )
        tensors = load_safetensors_dir(folder)
    else:
        seq = args.seq or 256
        lr = args.lr if args.lr is not None else 2e-2
        byte_tokenizer = ByteTokenizer()
        tokenizer = None if args.data_json else byte_tokenizer
        spec = LlamaSpec(
            vocab_size=byte_tokenizer.vocab_size,
            d_model=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            d_ff=256,
            seq=seq,
            batch=args.batch,
        )

    graph, _ = build_llama(spec, lora=lora)
    logits = graph.outputs[0]
    targets = graph.input("targets", i32[spec.batch, spec.seq])
    mask = graph.input("mask", f32[spec.batch, spec.seq])
    graph.output(masked_lm_loss_indexed(graph, logits, targets, mask), "loss")

    store = ParameterStore.from_graph(
        graph, exclude=("tokens", "targets", "mask"), seed=args.seed
    )
    adapter = tuple(name for name in store.names if ".lora_" in name)
    base = tuple(name for name in store.names if ".lora_" not in name)
    # 低秩旁路必须以"零增量"起步，否则训练前的 loss 与基座对不上，后续数字都不可信。
    for name in adapter:
        if name.endswith(".lora_b"):
            store[name][...] = 0.0

    if tensors is not None:
        unused = load_llama_weights(spec, store, tensors, skip=adapter)
        if unused:
            print(f"未使用的张量 {len(unused)} 个（例如 {unused[0]}）")
    if args.init:
        # 存档是基座（全参预训练 / SFT 的产物），参数集合恰好等于 base；
        # 用一个只含基座数组的 store 承接即可——它与主 store 共享同一批数组对象。
        meta = load_checkpoint(
            args.init, ParameterStore({name: store[name] for name in base})
        )
        print(f"基座来自 {args.init}（meta={meta}）")

    if args.data_json:
        rows = _rows_from_json(args.data_json, spec.seq)
        pad_id = args.pad_id
        source = "json"
    else:
        rows = _rows(tokenizer, spec.seq, chatml=chatml)
        pad_id = tokenizer.pad_id
        source = "chatml" if chatml else "byte"
    base_count = sum(int(np.prod(store[name].shape)) for name in base)
    adapter_count = sum(int(np.prod(store[name].shape)) for name in adapter)
    print(
        f"{source} base={base_count / 1e6:.2f}M adapter={adapter_count / 1e6:.2f}M "
        f"({adapter_count / base_count:.2%} 可训练) rank={lora.rank} "
        f"alpha={lora.alpha:g} targets={len(lora.targets)} "
        f"seq={spec.seq} batch={spec.batch} pairs={len(rows)} lr={lr:g} device={args.device}"
    )

    backend = _backend(args.device)
    trainer = Trainer(
        graph,
        store,
        Adam({name: store[name] for name in adapter}, lr=lr),
        backend=backend,
        output="loss",
        trainable=adapter,
    )

    frozen_before = {name: store[name].copy() for name in base}
    feed, valid = _batch(rows, 0, spec, pad_id)
    full = _loss(backend, graph, store, feed, valid)
    answer = _loss(backend, graph, store, feed, feed["mask"])
    print(f"训练前（第 1 批）: 全段 {full:.4f} / 回答段 {answer:.4f}")

    losses = []
    for step in range(args.steps):
        losses.append(trainer.step(_batch(rows, step, spec, pad_id)[0]))
        if step == 0 or (step + 1) % 5 == 0 or step + 1 == args.steps:
            print(f"step {step + 1:4d}/{args.steps}  回答段 loss {losses[-1]:.4f}")

    trainer.sync_to_host()
    full = _loss(backend, graph, store, feed, valid)
    answer = _loss(backend, graph, store, feed, feed["mask"])
    print(f"训练后（第 1 批）: 全段 {full:.4f} / 回答段 {answer:.4f}")

    moved = [name for name in base if not np.array_equal(store[name], frozen_before[name])]
    if moved:
        print(f"错误：{len(moved)} 个冻结参数被改动了（例如 {moved[0]}）")
        return 1
    print(f"冻结检查: {len(base)} 个基座参数逐位不变")

    window = max(1, min(5, len(losses) // 2))
    head = sum(losses[:window]) / window
    tail = sum(losses[-window:]) / window
    print(f"回答段 loss {head:.4f} -> {tail:.4f}  ({'下降' if tail < head else '未下降'})")

    if args.save:
        # 只存 adapter：键就是 IR 参数名，加载时按同一个 LoraSpec 建图再覆盖即可。
        np.savez(args.save, **{name: store[name] for name in adapter})
        print(f"adapter 已写入 {args.save}（{len(adapter)} 个数组）")
    return 0 if tail < head else 1


if __name__ == "__main__":
    raise SystemExit(main())
