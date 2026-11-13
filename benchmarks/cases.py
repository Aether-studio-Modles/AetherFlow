"""基准用的固定图集。

选图原则与 conformance 相同：**每个用例代表一种真实的执行形态**，
而不是"把所有算子都测一遍"。规模取到单次执行在毫秒量级——
太小会被主机派发开销淹没，太大则一次回归检查要跑很久。

所有用例只依赖 IR（不依赖 torch），因此本地无 GPU 也能跑，只是意义在于
"框架本身没坏"，不是性能数字。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Tuple

import numpy as np

from aetherflow.ir.graph import Graph
from aetherflow.ir.types import f32
from aetherflow.models import BlockSpec, build_transformer_block


def _f32(rng: np.random.Generator, shape: tuple, scale: float = 1.0) -> np.ndarray:
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _matmul(m: int, k: int, n: int):
    def build(rng):
        g = Graph("matmul")
        a = g.input("a", f32[m, k])
        b = g.input("b", f32[k, n])
        g.output(g.matmul(a, b), "y")
        return g, {"a": _f32(rng, (m, k)), "b": _f32(rng, (k, n), 0.1)}

    return build


def _matmul_batched(batch: int, m: int, k: int, n: int):
    def build(rng):
        g = Graph("matmul_batched")
        a = g.input("a", f32[batch, m, k])
        b = g.input("b", f32[batch, k, n])
        g.output(g.matmul(a, b), "y")
        return g, {
            "a": _f32(rng, (batch, m, k)),
            "b": _f32(rng, (batch, k, n), 0.1),
        }

    return build


def _conv2d(x_shape, w_shape, stride=(1, 1), padding=(0, 0)):
    def build(rng):
        g = Graph("conv2d")
        x = g.input("x", f32[x_shape])
        w = g.input("w", f32[w_shape])
        b = g.input("b", f32[w_shape[0]])
        y = g.add_node(
            "conv2d", [x, w, b], {"stride": stride, "padding": padding}
        )
        g.output(y, "y")
        return g, {
            "x": _f32(rng, x_shape),
            "w": _f32(rng, w_shape, 0.1),
            "b": _f32(rng, (w_shape[0],), 0.05),
        }

    return build


def _maxpool2d(x_shape, kernel=2, stride=2):
    def build(rng):
        g = Graph("maxpool2d")
        x = g.input("x", f32[x_shape])
        g.output(g.maxpool2d(x, kernel=kernel, stride=stride), "y")
        return g, {"x": _f32(rng, x_shape)}

    return build


def _transformer_block():
    def build(rng):
        spec = BlockSpec(d_model=64, num_heads=4, d_ff=128)
        g = Graph("block")
        _, out = build_transformer_block(g, spec)
        g.output(out, "y")

        feed = {"x": _f32(rng, (4, 32, spec.d_model), 0.5)}
        for name, shape in spec.param_shapes.items():
            if name == "scale":
                feed[name] = np.array([1.0 / math.sqrt(spec.head_dim)], dtype=np.float32)
            else:
                feed[name] = _f32(rng, shape, 0.1)
        return g, feed

    return build


def _cnn_lenet():
    """手工用 IR 搭一个 LeNet：多算子串联，检验整图调度而不是单个 kernel。"""

    def build(rng):
        g = Graph("cnn")
        x = g.input("x", f32[4, 1, 32, 32])
        w1 = g.input("w1", f32[6, 1, 5, 5])
        b1 = g.input("b1", f32[6])
        w2 = g.input("w2", f32[16, 6, 5, 5])
        b2 = g.input("b2", f32[16])
        fc = g.input("fc", f32[400, 10])
        fcb = g.input("fcb", f32[10])

        h = g.maxpool2d(g.relu(g.conv2d(x, w1, b1)), kernel=2)
        h = g.maxpool2d(g.relu(g.conv2d(h, w2, b2)), kernel=2)
        h = g.reshape(h, shape=(4, 400))
        h = g.relu(g.matmul(h, fc))
        g.output(g.add(h, fcb), "y")

        return g, {
            "x": _f32(rng, (4, 1, 32, 32)),
            "w1": _f32(rng, (6, 1, 5, 5), 0.1),
            "b1": _f32(rng, (6,), 0.05),
            "w2": _f32(rng, (16, 6, 5, 5), 0.1),
            "b2": _f32(rng, (16,), 0.05),
            "fc": _f32(rng, (400, 10), 0.05),
            "fcb": _f32(rng, (10,), 0.05),
        }

    return build


def _split_x(feed: Mapping[str, np.ndarray]):
    """把 ``x`` 留给每步传入，其余输入当作常驻权重。

    真实推理会把权重放在显存里，只传激活；基准的默认用例却每步都把全部权重
    从主机拷一遍。两种口径都要有：前者是"框架能做到多快"，后者是"一个没做
    常驻优化的调用方会看到多少开销"。
    """
    return {k: v for k, v in feed.items() if k != "x"}, {"x": feed["x"]}


@dataclass(frozen=True)
class Case:
    name: str
    build: Callable
    warmup: int
    repeat: int
    note: str
    # 非空时表示这个用例要按"权重常驻"跑：``split(feed) -> (params, activations)``。
    # 只有支持 ``resident_parameters`` 的后端才走常驻；参考后端仍用完整 feed。
    split: Optional[Callable[[Mapping], Tuple[Mapping, Mapping]]] = None


CASES: Tuple[Case, ...] = (
    Case("matmul_256", _matmul(256, 256, 256), 3, 10, "cuBLAS 的基本盘"),
    Case("matmul_1024", _matmul(1024, 1024, 1024), 2, 5, "大 GEMM：算力主导"),
    Case("matmul_batched", _matmul_batched(8, 128, 256, 128), 3, 10, "批量 GEMM"),
    Case("conv2d_3x3_pad1", _conv2d((2, 3, 64, 64), (16, 3, 3, 3), padding=(1, 1)), 3, 5,
         "浅通道 3x3：K=27，auto 选分块直算"),
    Case("conv2d_1x1", _conv2d((2, 32, 32, 32), (64, 32, 1, 1)), 3, 5,
         "1x1 卷积：K=32，auto 选分块直算"),
    # 尺寸取到毫秒量级：早先的 (2,16,64,64) 只有 0.1 ms，同一台机器上重复测量
    # 的中位数能在 0.097~0.186 ms 之间跳（一次 CPU 抢占就能翻倍），
    # 这种用例做门禁只会制造假回归。宁可跑慢一点，也要测得准。
    Case("maxpool2x2", _maxpool2d((2, 32, 256, 256)), 5, 10, "纯索引算子：访存主导"),
    Case("transformer_block", _transformer_block(), warmup=2, repeat=5,
         note="整图调度 + 多种 kernel 混合"),
    Case("transformer_block_resident", _transformer_block(), warmup=2, repeat=5,
         note="同上，权重常驻、只传激活（真实推理口径）", split=_split_x),
    Case("cnn_lenet", _cnn_lenet(), warmup=2, repeat=5, note="CNN 的端到端形态"),
    Case("cnn_lenet_resident", _cnn_lenet(), warmup=2, repeat=5,
         note="同上，权重常驻、只传激活（真实推理口径）", split=_split_x),
)


def build_case(case: Case, seed: int = 0):
    """构建单个用例的 (graph, feed)；固定种子保证各后端看到同一份数据。"""
    return case.build(np.random.default_rng(seed))
