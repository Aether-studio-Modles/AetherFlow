"""AetherFlow 最小闭环演示：构图 -> 打印 IR -> 参考后端执行。

运行：
    py -3.10 demo_transformer_block.py
"""

import numpy as np

from aetherflow import Graph, ReferenceBackend, verify, verify_portable_only
from aetherflow.models import BlockSpec, build_transformer_block
from aetherflow.passes import CommonSubexpressionElimination, default_pipeline

SPEC = BlockSpec(d_model=32, num_heads=4, d_ff=64)


def main() -> None:
    g = Graph("transformer_block")
    inputs, out = build_transformer_block(g, SPEC)
    g.output(out, "y")

    print("=" * 72)
    print("IR")
    print("=" * 72)
    print(g.to_ir())

    issues = verify(g)
    print()
    print("verifier:", "通过" if not issues else issues)
    print("可移植性:", "全部节点 PORTABLE" if not verify_portable_only(g) else "存在不可移植节点")

    rng = np.random.default_rng(0)
    feed = {"x": rng.standard_normal((2, 5, SPEC.d_model)).astype(np.float32)}
    for name, shape in SPEC.param_shapes.items():
        if name == "scale":
            feed[name] = np.array([1.0 / np.sqrt(SPEC.head_dim)], dtype=np.float32)
        else:
            feed[name] = rng.standard_normal(shape).astype(np.float32) * 0.1

    y = ReferenceBackend().execute(g, feed)["y"]
    print()
    print("=" * 72)
    print("执行结果")
    print("=" * 72)
    print(f"输入 x: {feed['x'].shape} {feed['x'].dtype}")
    print(f"输出 y: {y.shape} {y.dtype}")
    print(f"输出统计: mean={y.mean():.6f} std={y.std():.6f} max={y.max():.6f}")
    print(f"节点数: {len(g.nodes)}  图输入数: {len(g.inputs)}  值总数: {len(g.values)}")

    opt = default_pipeline().run(g)
    y_opt = ReferenceBackend().execute(opt, feed)["y"]
    print()
    print("=" * 72)
    print("中层优化")
    print("=" * 72)
    print(f"可合并的重复子表达式: {CommonSubexpressionElimination().duplicates(g)}")
    print(f"节点数: {len(g.nodes)} -> {len(opt.nodes)}")
    print(f"优化前后最大偏差: {np.abs(y - y_opt).max():.3e}")
    print("说明: 该模型图上没有冗余子计算，图级收益要靠算子融合，而非删除。")


if __name__ == "__main__":
    main()
