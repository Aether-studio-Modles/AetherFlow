"""全参 SFT 示例：指令对训练，只对回答段算 loss。

与 ``pretrain.py`` 共用同一个训练闭环（前向 / 反向 / 优化器 / checkpoint），
只换三处：数据从连续语料换成指令对、损失用 ``mask`` 把 prompt 段置 0、
初始权重可以从预训练存档或 HF 权重接续。

    # 缩小版（默认，CPU 几十秒）：内置指令对 + ByteTokenizer
    python examples/llm/sft.py

    # 接预训练存档（pretrain.py --save out.npz）
    python examples/llm/sft.py --init out.npz

    # 真实 Qwen2.5-0.5B：HF 权重 + 预编码样本。Qwen 的 tokenizer.json 用
    # ``Split`` 预分词，本仓的 BpeTokenizer 明确拒绝它（不读 pattern 就换了
    # 一份分词器），所以 ids 与 answer_start 由外部编好后送进来。
    python examples/llm/sft.py --weights /root/models/Qwen2.5-0.5B \
        --data-json pairs.json --device cuda

验收标准是**回答段 loss 明显下降**，不是"训出可用模型"：mask 写错（例如把
prompt 也算进去）时，回答段 loss 会停在高位，而全段 loss 照样很低。
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
from aetherflow.ir.types import f32, i32
from aetherflow.models import (
    BpeTokenizer,
    ByteTokenizer,
    LlamaSpec,
    build_llama,
    load_llama_weights,
    masked_lm_loss_indexed,
)
from aetherflow.train import Adam, ParameterStore, Trainer, load_checkpoint, save_checkpoint
from aetherflow.weights import SafeTensorsFile, load_safetensors_dir

_SYSTEM = "You are a helpful assistant."

_PAIRS = (
    ("用一句话解释什么是梯度下降。",
     "梯度下降是沿着损失下降最快的方向一步步更新参数，直到损失收敛。"),
    ("把下面这句话翻译成英文：今天天气很好。",
     "The weather is nice today."),
    ("1 加 1 等于几？", "等于 2。"),
    ("写一个 Python 函数，返回两个数的和。",
     "def add(a, b):\n    return a + b"),
    ("什么是过拟合？",
     "过拟合是模型把训练集的噪声也学进去，在新数据上表现变差。"),
    ("列出三种常见的优化器。", "SGD、Adam 和 Adagrad。"),
    ("用一个词概括深度学习的本质。", "表示学习。"),
    ("为什么训练时要打乱数据顺序？",
     "避免模型学到样本顺序带来的虚假规律，让每批数据的分布更接近整体。"),
)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20, help="训练步数")
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Adam 学习率（默认 tiny 2e-3 / 真实权重 2e-5）",
    )
    parser.add_argument("--seq", type=int, default=None, help="上下文长度（默认 tiny 256 / 真实 128）")
    parser.add_argument("--batch", type=int, default=1, help="批大小")
    parser.add_argument("--seed", type=int, default=0, help="参数初始化种子")
    parser.add_argument("--weights", default=None, help="HF 权重目录（config.json + *.safetensors + tokenizer.json）")
    parser.add_argument("--init", default=None, help="本仓库的 .npz 存档（pretrain.py / sft.py 的 --save 产物）")
    parser.add_argument(
        "--data-json",
        default=None,
        help="预编码样本：[{\"ids\": [...], \"answer_start\": n}, ...]；"
             "词表不被本仓 BpeTokenizer 支持时（如 Qwen 的 Split 预分词）走这条",
    )
    parser.add_argument("--pad-id", type=int, default=0, help="--data-json 模式下的 pad id")
    parser.add_argument("--save", default=None, help="训练结束后把权重写成 .npz")
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


def _encode_pair(tokenizer, question: str, answer: str, *, chatml: bool):
    """返回 ``(ids, answer_start)``：``answer_start`` 之后的 token 才计入 loss。"""
    if chatml:
        head = (
            f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        prompt_ids = tokenizer.encode(head)
        answer_ids = tokenizer.encode(f"{answer}<|im_end|>")
        return prompt_ids + answer_ids, len(prompt_ids)
    head = f"### User:\n{question}\n### Assistant:\n"
    prompt_ids = tokenizer.encode(head)
    answer_ids = tokenizer.encode(answer + "\n") + [tokenizer.eos_id]
    return prompt_ids + answer_ids, len(prompt_ids)


def _rows(tokenizer, seq: int, *, chatml: bool):
    rows = []
    for question, answer in _PAIRS:
        ids, start = _encode_pair(tokenizer, question, answer, chatml=chatml)
        rows.append(_check_row(ids, start, seq))
    return rows


def _check_row(ids, start: int, seq: int):
    if not 0 < start <= len(ids):
        raise SystemExit(f"answer_start={start} 不在 (0, {len(ids)}] 里")
    if len(ids) > seq + 1:
        raise SystemExit(
            f"样本有 {len(ids)} 个 token，超过 seq+1={seq + 1}；"
            "请加大 --seq 或缩短指令对"
        )
    return np.asarray(ids, dtype=np.int64), start


def _rows_from_json(path, seq: int):
    """读预编码样本；样本自带 ``answer_start``，不经过任何分词器。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit(f"{path} 必须是非空数组")
    rows = []
    for index, item in enumerate(data):
        if not isinstance(item, dict) or "ids" not in item or "answer_start" not in item:
            raise SystemExit(f"{path} 第 {index} 项缺 ids 或 answer_start")
        rows.append(_check_row(item["ids"], int(item["answer_start"]), seq))
    return rows


def _batch(rows, index: int, spec: LlamaSpec, pad_id: int):
    """组一个 batch，返回 ``(feed, valid)``。

    ``feed`` 里的 ``mask`` 只覆盖回答段（含结束符），``valid`` 是"真实 token"
    的位置——评估全段 loss 时用它，否则 pad 位置的 id=0 会把损失拉下去。
    """
    chosen = [rows[(index + i) % len(rows)] for i in range(spec.batch)]
    tokens = np.full((spec.batch, spec.seq), pad_id, dtype=np.int32)
    targets = np.zeros((spec.batch, spec.seq), dtype=np.int32)
    mask = np.zeros((spec.batch, spec.seq), dtype=np.float32)
    valid = np.zeros((spec.batch, spec.seq), dtype=np.float32)
    for b, (ids, start) in enumerate(chosen):
        labels = ids[1:]
        positions = np.arange(len(labels))
        tokens[b, :len(ids) - 1] = ids[:-1]
        # 整数标签是扁平的 ``(b*T + t)*V + y``：图里把 (B,T,V) 的 logits 拉直后查表。
        targets[b, :len(labels)] = (b * spec.seq + positions) * spec.vocab_size + labels
        mask[b, :len(labels)] = (positions + 1 >= start).astype(np.float32)
        valid[b, :len(labels)] = 1.0
    return {"tokens": tokens, "targets": targets, "mask": mask}, valid


def _loss(backend, graph, store, feed: dict, mask: np.ndarray) -> float:
    return float(backend.execute(graph, store.feed({**feed, "mask": mask}))["loss"])


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.weights and args.init:
        raise SystemExit("--weights 与 --init 只能给一个：它们都是初始权重来源")

    chatml = args.weights is not None
    tensors: dict | None = None
    tokenizer = None
    if args.weights:
        folder = Path(args.weights)
        seq = args.seq or 128
        lr = args.lr if args.lr is not None else 2e-5
        if args.data_json is None:
            # Qwen 的 tokenizer.json 用 Split 预分词，本仓的 BpeTokenizer 明确
            # 拒绝它；这种词表请先用 --data-json 交预编码样本。
            tokenizer = BpeTokenizer.from_file(folder / "tokenizer.json")
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        spec = LlamaSpec.from_hf_config(
            config, seq=seq, batch=args.batch, qkv_bias=_has_qkv_bias(folder)
        )
        tensors = load_safetensors_dir(folder)
    else:
        seq = args.seq or 256
        lr = args.lr if args.lr is not None else 2e-3
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

    graph, _ = build_llama(spec)
    logits = graph.outputs[0]
    targets = graph.input("targets", i32[spec.batch, spec.seq])
    mask = graph.input("mask", f32[spec.batch, spec.seq])
    graph.output(masked_lm_loss_indexed(graph, logits, targets, mask), "loss")

    store = ParameterStore.from_graph(
        graph, exclude=("tokens", "targets", "mask"), seed=args.seed
    )
    if tensors is not None:
        unused = load_llama_weights(spec, store, tensors)
        if unused:
            print(f"未使用的张量 {len(unused)} 个（例如 {unused[0]}）")
    if args.init:
        meta = load_checkpoint(args.init, store)
        print(f"初始权重来自 {args.init}（meta={meta}）")

    if args.data_json:
        rows = _rows_from_json(args.data_json, spec.seq)
        pad_id = args.pad_id
        source = "json"
    else:
        rows = _rows(tokenizer, spec.seq, chatml=chatml)
        pad_id = tokenizer.pad_id
        source = "chatml" if chatml else "byte"
    n_params = sum(int(np.prod(store[name].shape)) for name in store.names)
    print(
        f"{source} params={n_params / 1e6:.2f}M "
        f"seq={spec.seq} batch={spec.batch} pairs={len(rows)} "
        f"lr={lr:g} device={args.device}"
    )

    backend = _backend(args.device)
    trainer = Trainer(
        graph, store, Adam(store.arrays(), lr=lr), backend=backend, output="loss"
    )

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

    # 头尾窗口不能重叠：步数少于 10 时按步数折半，否则"前后对比"是同一段。
    window = max(1, min(5, len(losses) // 2))
    head = sum(losses[:window]) / window
    tail = sum(losses[-window:]) / window
    print(f"回答段 loss {head:.4f} -> {tail:.4f}  ({'下降' if tail < head else '未下降'})")

    if args.save:
        save_checkpoint(
            args.save,
            store,
            meta={"stage": "sft", "steps": args.steps, "lr": lr},
        )
        print(f"权重已写入 {args.save}")
    return 0 if tail < head else 1


if __name__ == "__main__":
    raise SystemExit(main())
