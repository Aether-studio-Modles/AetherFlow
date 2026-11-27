"""远程 GPU 验证脚本：让 CUDA 后端自证。

本地没有 GPU 时这一步只能跳过，所以凡是"新增了 CUDA 内核"的改动，都必须
在真机上跑一遍再宣称完成。这个脚本就是那一步：

1. 打印设备信息（确认真的拿到了 GPU，而不是回落到别的东西）；
2. 跑完整的 conformance 套件（覆盖全部 46 个算子）；
3. 对新增内核做额外的形状/边界对拍（官方图集里的形状只有一种，
   边界条件必须另外覆盖），包括 f16 形态与编译期静态显存预算的对照；
4. 全部通过才返回 0，否则返回 1。

用法::

    python verify_cuda.py
    python verify_cuda.py --strict   # 把 refused 也视为失败（全算子后端才该用）

需要可用的 CUDA 设备与 cupy；脚本不修改任何文件。
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from aetherflow import Graph, ReferenceBackend, boolean, f32, i8, i32
from aetherflow.backends.cuda import CudaBackend
from aetherflow.conformance import run_conformance


def _device_banner(backend: CudaBackend) -> None:
    info = backend.device_info()
    print(f"device  : {info.name}")
    print(f"runtime : CUDA {info.runtime // 1000}.{(info.runtime % 1000) // 10}")
    print(f"memory  : {info.total_memory_bytes / 2**30:.1f} GiB")
    print(f"caps    : {len(backend.capabilities())} 个算子")


def _rand(rng: np.random.Generator, *shape, scale: float = 1.0) -> np.ndarray:
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _extra_cases() -> list:
    """新增内核的边界用例：(名字, 构图函数, 输入)。

    这些形状刻意与 conformance 图集不同——图集保证"能跑"，这里保证
    "边界对"，两者缺一不可。
    """
    rng = np.random.default_rng(20261031)

    def gelu():
        g = Graph()
        x = g.input("x", f32[3, 11])
        g.output(g.gelu(x), "y")
        # 跨饱和区：值域必须覆盖 tanh 两端的平台
        return g, {"x": np.array(
            [[-40.0, -5.0, -1.0, -0.25, 0.0, 0.25, 1.0, 5.0, 40.0, 0.1, -0.1]] * 3,
            dtype=np.float32,
        )}

    def rmsnorm():
        g = Graph()
        x = g.input("x", f32[2, 3, 16])
        gamma = g.input("gamma", f32[16])
        g.output(g.rmsnorm(x, gamma, axes=(-1,), eps=1e-6), "y")
        return g, {"x": _rand(rng, 2, 3, 16), "gamma": _rand(rng, 16) + 1.0}

    def avgpool_rect():
        g = Graph()
        x = g.input("x", f32[2, 3, 9, 7])
        g.output(g.avgpool2d(x, kernel=(3, 2), stride=(2, 1), padding=(1, 1)), "y")
        return g, {"x": _rand(rng, 2, 3, 9, 7)}

    def avgpool_backward_overlap():
        g = Graph()
        dy = g.input("dy", f32[1, 2, 6, 6])
        g.output(
            g.avgpool2d_backward(
                dy,
                input_shape=[1, 2, 6, 6],
                kernel=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            "y",
        )
        return g, {"dy": _rand(rng, 1, 2, 6, 6)}

    def maxpool_pad_stride():
        """窗口伸出图像边界：越界采样点必须跳过，不能跟真实值争最大值。"""
        g = Graph()
        x = g.input("x", f32[2, 3, 7, 5])
        g.output(g.maxpool2d(x, kernel=(3, 3), stride=(2, 2), padding=(1, 1)), "y")
        return g, {"x": _rand(rng, 2, 3, 7, 5)}

    def maxpool_backward_ties_overlap():
        """步长 1 + padding 1：窗口重叠，量化后的输入里还会出现并列最大值。"""
        g = Graph()
        x = g.input("x", f32[1, 2, 6, 6])
        dy = g.input("dy", f32[1, 2, 6, 6])
        g.output(
            g.maxpool2d_backward(
                x, dy, kernel=(3, 3), stride=(1, 1), padding=(1, 1)
            ),
            "y",
        )
        # 量化到整数格点，保证窗口里出现多个相同最大值，考验"取第一个"的平局规则。
        return g, {
            "x": np.round(_rand(rng, 1, 2, 6, 6) * 2.0),
            "dy": _rand(rng, 1, 2, 6, 6),
        }

    def concat_axis0():
        g = Graph()
        a = g.input("a", f32[2, 3])
        b = g.input("b", f32[5, 3])
        g.output(g.concat(a, b, axis=0), "y")
        return g, {"a": _rand(rng, 2, 3), "b": _rand(rng, 5, 3)}

    def concat_middle():
        g = Graph()
        a = g.input("a", f32[2, 3, 4])
        b = g.input("b", f32[2, 5, 4])
        g.output(g.concat(a, b, axis=1), "y")
        return g, {"a": _rand(rng, 2, 3, 4), "b": _rand(rng, 2, 5, 4)}

    def embedding_i8():
        g = Graph()
        table = g.input("table", f32[100, 2, 3])
        ids = g.input("ids", i8[4, 5])
        g.output(g.embedding(table, ids), "y")
        return g, {
            "table": _rand(rng, 100, 2, 3),
            "ids": rng.integers(0, 100, (4, 5), dtype=np.int8),
        }

    def embedding_backward_duplicates():
        g = Graph()
        dy = g.input("dy", f32[3, 5, 4])
        ids = g.input("ids", i32[3, 5])
        g.output(g.embedding_backward(dy, ids, table_shape=[10, 4]), "y")
        return g, {
            "dy": _rand(rng, 3, 5, 4),
            # 大量重复：反向必须逐条累加，覆盖 atomicAdd 的路径
            "ids": np.array(
                [[0, 0, 0, 1, 1], [1, 1, 2, 2, 2], [0, 3, 3, 3, 9]], dtype=np.int32
            ),
        }

    def chain():
        """端到端小图：查表 -> 归一化 -> 激活，三个新内核首尾相接。"""
        g = Graph()
        table = g.input("table", f32[16, 8])
        ids = g.input("ids", i32[2, 4])
        gamma = g.input("gamma", f32[8])
        h = g.embedding(table, ids)
        h = g.rmsnorm(h, gamma, axes=(-1,), eps=1e-6)
        g.output(g.gelu(h), "y")
        return g, {
            "table": _rand(rng, 16, 8, scale=0.5),
            "ids": rng.integers(0, 16, (2, 4), dtype=np.int32),
            "gamma": _rand(rng, 8) + 1.0,
        }

    def batch_norm_axis0():
        """通道轴不是 1：一般化靠 (outer, C, inner) 三元组，不是四维特判。"""
        g = Graph()
        x = g.input("x", f32[3, 2, 4])
        gamma = g.input("gamma", f32[3])
        beta = g.input("beta", f32[3])
        g.output(g.batch_norm(x, gamma, beta, axis=0), "y")
        return g, {
            "x": _rand(rng, 3, 2, 4),
            "gamma": _rand(rng, 3),
            "beta": _rand(rng, 3),
        }

    def batch_norm_single_sample():
        """N=1 且空间 1x1：方差恰好为 0，只剩 eps，最容易在这里除零。"""
        g = Graph()
        x = g.input("x", f32[1, 5, 1, 1])
        gamma = g.input("gamma", f32[5])
        beta = g.input("beta", f32[5])
        g.output(g.batch_norm(x, gamma, beta, axis=1, eps=1e-5), "y")
        return g, {
            "x": _rand(rng, 1, 5, 1, 1),
            "gamma": _rand(rng, 5),
            "beta": _rand(rng, 5),
        }

    def batch_norm_stats_axis0():
        g = Graph()
        x = g.input("x", f32[4, 3, 5])
        g.output(g.batch_norm_stats(x, axis=0), "y")
        return g, {"x": _rand(rng, 4, 3, 5)}

    def conv2d_grouped():
        """矩形核 + 非对称桶形：分组切片要同时对齐输入与输出通道。"""
        g = Graph()
        x = g.input("x", f32[2, 4, 9, 7])
        w = g.input("w", f32[6, 2, 3, 2])
        b = g.input("b", f32[6])
        g.output(
            g.conv2d(x, w, b, stride=(1, 2), padding=(1, 0), groups=2), "y"
        )
        return g, {
            "x": _rand(rng, 2, 4, 9, 7),
            "w": _rand(rng, 6, 2, 3, 2, scale=0.3),
            "b": _rand(rng, 6),
        }

    def conv2d_depthwise_stride2():
        """depthwise + stride2：每组只有一个通道，最容易在通道偏移上出错。"""
        g = Graph()
        x = g.input("x", f32[1, 4, 11, 11])
        w = g.input("w", f32[4, 1, 3, 3])
        g.output(g.conv2d(x, w, padding=(1, 1), stride=(2, 2), groups=4), "y")
        return g, {
            "x": _rand(rng, 1, 4, 11, 11),
            "w": _rand(rng, 4, 1, 3, 3, scale=0.3),
        }

    def conv2d_input_grad_grouped():
        g = Graph()
        dy = g.input("dy", f32[2, 6, 9, 7])
        w = g.input("w", f32[6, 2, 3, 3])
        g.output(
            g.conv2d_input_grad(
                dy, w, input_shape=[2, 4, 9, 7], padding=(1, 1), groups=2
            ),
            "y",
        )
        return g, {
            "dy": _rand(rng, 2, 6, 9, 7),
            "w": _rand(rng, 6, 2, 3, 3, scale=0.3),
        }

    def conv2d_weight_grad_grouped():
        g = Graph()
        x = g.input("x", f32[2, 4, 9, 7])
        dy = g.input("dy", f32[2, 6, 9, 7])
        g.output(
            g.conv2d_weight_grad(
                x, dy, weight_shape=[6, 2, 3, 3], padding=(1, 1), groups=2
            ),
            "y",
        )
        return g, {"x": _rand(rng, 2, 4, 9, 7), "dy": _rand(rng, 2, 6, 9, 7)}

    def adaptive_pool_non_divisible():
        """7/3 与 5/2 都不整除：窗口端点 floor/ceil 与重叠都要覆盖。"""
        g = Graph()
        x = g.input("x", f32[2, 3, 7, 5])
        g.output(g.adaptive_avg_pool2d(x, output_size=(3, 2)), "y")
        return g, {"x": _rand(rng, 2, 3, 7, 5)}

    def adaptive_pool_backward_overlap():
        g = Graph()
        dy = g.input("dy", f32[2, 3, 3, 2])
        g.output(
            g.adaptive_avg_pool2d_backward(
                dy, input_shape=[2, 3, 7, 5], output_size=(3, 2)
            ),
            "y",
        )
        return g, {"dy": _rand(rng, 2, 3, 3, 2)}

    def scan_linear_3d():
        """(T,B,D) + 三维尾：尾维被压平，索引不能依赖尾维的具体阶数。"""
        g = Graph()
        a = g.input("a", f32[7, 2, 3])
        b = g.input("b", f32[7, 2, 3])
        h0 = g.input("h0", f32[2, 3])
        g.output(g.add_node("scan_linear", [a, b, h0]), "y")
        return g, {
            "a": _rand(rng, 7, 2, 3, scale=0.3),
            "b": _rand(rng, 7, 2, 3),
            "h0": _rand(rng, 2, 3),
        }

    def scan_linear_single_step():
        """T=1：退化成 a0*h0+b0，最容易被"先循环再写"的写法漏掉。"""
        g = Graph()
        a = g.input("a", f32[1, 16])
        b = g.input("b", f32[1, 16])
        h0 = g.input("h0", f32[16])
        g.output(g.add_node("scan_linear", [a, b, h0]), "y")
        return g, {
            "a": _rand(rng, 1, 16, scale=0.9),
            "b": _rand(rng, 1, 16),
            "h0": _rand(rng, 16),
        }

    def scan_linear_wide():
        """长序列 + 宽尾：T 走 64 步，尾 2048 个元素跨多个 block。"""
        g = Graph()
        a = g.input("a", f32[64, 2, 1024])
        b = g.input("b", f32[64, 2, 1024])
        h0 = g.input("h0", f32[2, 1024])
        g.output(g.add_node("scan_linear", [a, b, h0]), "y")
        return g, {
            "a": _rand(rng, 64, 2, 1024, scale=0.3),
            "b": _rand(rng, 64, 2, 1024),
            "h0": _rand(rng, 2, 1024),
        }

    def scan_linear_backward_long():
        """伴随从后往前累积：T=9 且 |a| 不小，任何"少乘一步 a"都会放大。"""
        g = Graph()
        a = g.input("a", f32[9, 4])
        dy = g.input("dy", f32[9, 4])
        g.output(g.add_node("scan_linear_backward", [a, dy]), "y")
        return g, {
            "a": _rand(rng, 9, 4, scale=0.6),
            "dy": _rand(rng, 9, 4),
        }

    def cumsum_mid_axis():
        """中间轴前缀和：折成 (outer, steps, inner) 后两侧折维都不能省。"""
        g = Graph()
        x = g.input("x", f32[2, 5, 3])
        g.output(g.add_node("cumsum", [x], {"axis": 1}), "y")
        return g, {"x": _rand(rng, 2, 5, 3)}

    def tril_negative_diagonal():
        """对角线以下才保留 + 前导维逐片独立：掩码偏移与片号都不能串。"""
        g = Graph()
        x = g.input("x", f32[2, 5, 5])
        g.output(g.add_node("tril", [x], {"diagonal": -1}), "y")
        return g, {"x": _rand(rng, 2, 5, 5)}

    def silu_gate_odd():
        """SwiGLU 门控：元素总数 21 不是 block 维的倍数，跨步循环必须收尾。"""
        g = Graph()
        x = g.input("x", f32[3, 7])
        gate = g.input("gate", f32[3, 7])
        g.output(g.silu(x, gate), "y")
        return g, {"x": _rand(rng, 3, 7), "gate": _rand(rng, 3, 7)}

    def layernorm_residual_odd():
        """残差融合：列数 7 不是 2 的幂，块内归约要处理非整块。"""
        g = Graph()
        a = g.input("a", f32[5, 7])
        r = g.input("r", f32[5, 7])
        gamma = g.input("gamma", f32[7])
        beta = g.input("beta", f32[7])
        g.output(g.layernorm(a, gamma, beta, r, axes=(-1,)), "y")
        return g, {
            "a": _rand(rng, 5, 7),
            "r": _rand(rng, 5, 7),
            "gamma": _rand(rng, 7) + 1.0,
            "beta": _rand(rng, 7),
        }

    def clamp_two_sided():
        """双侧截断：输入跨度覆盖区间两侧，否则只验证了"原样透传"。"""
        g = Graph()
        x = g.input("x", f32[4, 6])
        g.output(g.add_node("clamp", [x], {"min": -0.5, "max": 0.5}), "y")
        return g, {"x": _rand(rng, 4, 6, scale=2.0)}

    def where_bcast_pow2_shape():
        """折叠后每维都是 2 的幂的规则形状。

        曾经有掩码 + 右移的 pow2 专用内核，实测比 div/mod 慢已删除
        （DESIGN 5.300）；用例保留原形状，作为规则形状的回归证据。
        """
        g = Graph("where_pow2_shape")
        c = g.input("c", boolean[8, 1, 8])
        a = g.input("a", f32[8, 4, 8])
        b = g.input("b", f32[8])
        g.output(g.where(c, a, b), "y")
        return g, {
            "c": rng.integers(0, 2, size=(8, 1, 8)).astype(bool),
            "a": _rand(rng, 8, 4, 8),
            "b": _rand(rng, 8),
        }

    def where_bcast_odd_rank3():
        """非 2 的幂 + 中间维广播：div 余数路径，折叠后的组数不能算错。"""
        g = Graph("where_odd")
        c = g.input("c", boolean[3, 1, 5])
        a = g.input("a", f32[3, 7, 5])
        b = g.input("b", f32[5])
        g.output(g.where(c, a, b), "y")
        return g, {
            "c": rng.integers(0, 2, size=(3, 1, 5)).astype(bool),
            "a": _rand(rng, 3, 7, 5),
            "b": _rand(rng, 5),
        }

    def compare_feeds_where():
        """比较结果直接当 where 的条件：bool 数据流必须端到端成立。"""
        g = Graph("cmp_where")
        x = g.input("x", f32[3, 9])
        y = g.input("rhs", f32[3, 9])
        g.output(g.where(g.gt(x, y), x, y), "y")
        return g, {
            # 量化到整数格点，保证 eq/ne 之外还有大量并列（选哪支都合法）
            "x": np.round(_rand(rng, 3, 9) * 2.0),
            "rhs": np.round(_rand(rng, 3, 9) * 2.0),
        }

    def rounding_boundaries():
        """.5 取偶、负数与 -0：round / floor / ceil 三条内核一起对拍。"""
        g = Graph("rounding")
        x = g.input("x", f32[4, 8])
        y = g.add(g.round(x), g.floor(x))
        g.output(g.add(y, g.ceil(x)), "y")
        return g, {
            "x": np.array(
                [[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 0.0, -0.0],
                 [-3.7, -0.1, 0.1, 3.7, 1.4999999, 1.5000001, -1.4999999, -1.5000001],
                 [1e-8, -1e-8, 2.0, -2.0, 7.5, 8.5, -7.5, -8.5],
                 [1e6, -1e6, 0.49999997, -0.49999997, 123.456, -123.456, 4.5, -4.5]],
                dtype=np.float32,
            )
        }

    def sign_special():
        """±0 / ±inf / NaN：sign 写成 (x>0)-(x<0) 会让 NaN 变成 0。

        刻意**不**用 1e-38 这类次正规数：实测这台 V100 上（CUDA 12.4 + cupy
        14.2）GPU 对 subnormal 的比较一律为 false（``cp.sign(1e-38)`` 自己
        也返回 0），是环境级 FTZ 行为，与参考后端的 numpy 语义天然不同。
        拿它当一致性判据只会得到一个与代码无关的假失败。
        """
        g = Graph("sign_special")
        x = g.input("x", f32[3, 6])
        g.output(g.sign(x), "y")
        return g, {
            "x": np.array(
                [[0.0, -0.0, np.inf, -np.inf, np.nan, 3.5],
                 [-3.5, 1e-30, -1e-30, 1e30, -1e30, -1.0],
                 [2.0, -2.0, 0.5, -0.5, np.nan, 0.0]],
                dtype=np.float32,
            )
        }

    def mod_negative_floor():
        """负被除数 / 负模数：C 的截断取余在这里会差一个模数。"""
        g = Graph()
        a = g.input("a", i32[3, 5])
        b = g.input("b", i32[3, 5])
        g.output(g.add_node("mod", [a, b]), "y")
        return g, {
            "a": np.array(
                [[-7, 7, -7, 7, 0], [13, -13, 13, -13, 5], [-50, 50, -1, 1, 0]],
                dtype=np.int32,
            ),
            "b": np.array(
                [[3, 3, -3, -3, 4], [-8, 8, -8, 8, -8], [7, -7, 6, -6, 2]],
                dtype=np.int32,
            ),
        }

    def linear_i8a8_tail():
        """W8A8：K=13 不是 4 的倍数，``__dp4a`` 之后必须收尾；激活秩 3。"""
        g = Graph("w8a8_tail")
        xq = g.input("xq", i8[2, 3, 13])
        xs = g.input("xs", f32[2, 3, 1])
        wq = g.input("wq", i8[13, 5])
        ws = g.input("ws", f32[5])
        b = g.input("b", f32[5])
        g.output(g.add_node("linear_i8a8", [xq, xs, wq, ws, b]), "y")
        return g, {
            "xq": rng.integers(-127, 128, (2, 3, 13), dtype=np.int8),
            "xs": np.abs(_rand(rng, 2, 3, 1)) + 0.02,
            "wq": rng.integers(-127, 128, (13, 5), dtype=np.int8),
            "ws": np.abs(_rand(rng, 5)) + 0.05,
            "b": _rand(rng, 5),
        }

    def linear_i8a8_no_bias():
        """W8A8 的无 bias 形态 + K 恰为 4 的倍数：走纯 dp4a 主循环。"""
        g = Graph("w8a8_nobias")
        xq = g.input("xq", i8[9, 512])
        xs = g.input("xs", f32[9, 1])
        wq = g.input("wq", i8[512, 32])
        ws = g.input("ws", f32[32])
        g.output(g.add_node("linear_i8a8", [xq, xs, wq, ws]), "y")
        return g, {
            "xq": rng.integers(-127, 128, (9, 512), dtype=np.int8),
            "xs": np.abs(_rand(rng, 9, 1)) + 0.005,
            "wq": rng.integers(-127, 128, (512, 32), dtype=np.int8),
            "ws": np.abs(_rand(rng, 32)) + 0.01,
        }

    return [
        ("gelu_saturation", gelu),
        ("rmsnorm_rank3", rmsnorm),
        ("avgpool2d_rect_pad", avgpool_rect),
        ("avgpool2d_backward_overlap", avgpool_backward_overlap),
        ("maxpool2d_pad_stride", maxpool_pad_stride),
        ("maxpool2d_backward_ties_overlap", maxpool_backward_ties_overlap),
        ("concat_axis0", concat_axis0),
        ("concat_middle_rank3", concat_middle),
        ("embedding_i8_trailing", embedding_i8),
        ("embedding_backward_duplicates", embedding_backward_duplicates),
        ("chain_embedding_rmsnorm_gelu", chain),
        ("batch_norm_axis0", batch_norm_axis0),
        ("batch_norm_single_sample", batch_norm_single_sample),
        ("batch_norm_stats_axis0", batch_norm_stats_axis0),
        ("conv2d_grouped", conv2d_grouped),
        ("conv2d_depthwise_stride2", conv2d_depthwise_stride2),
        ("conv2d_input_grad_grouped", conv2d_input_grad_grouped),
        ("conv2d_weight_grad_grouped", conv2d_weight_grad_grouped),
        ("adaptive_pool2d_non_divisible", adaptive_pool_non_divisible),
        ("adaptive_pool2d_backward_overlap", adaptive_pool_backward_overlap),
        ("scan_linear_3d_tail", scan_linear_3d),
        ("scan_linear_single_step", scan_linear_single_step),
        ("scan_linear_wide", scan_linear_wide),
        ("scan_linear_backward_long", scan_linear_backward_long),
        ("cumsum_mid_axis", cumsum_mid_axis),
        ("tril_negative_diagonal", tril_negative_diagonal),
        ("clamp_two_sided", clamp_two_sided),
        ("where_bcast_pow2_shape", where_bcast_pow2_shape),
        ("where_bcast_odd_rank3", where_bcast_odd_rank3),
        ("compare_feeds_where", compare_feeds_where),
        ("rounding_boundaries", rounding_boundaries),
        ("sign_special", sign_special),
        ("mod_negative_floor", mod_negative_floor),
        ("linear_i8a8_tail_rank3", linear_i8a8_tail),
        ("linear_i8a8_no_bias", linear_i8a8_no_bias),
        ("silu_gate_odd_length", silu_gate_odd),
        ("layernorm_residual_odd_cols", layernorm_residual_odd),
    ]


def _run_extra(cuda: CudaBackend, ref: ReferenceBackend) -> list:
    failures = []
    for name, build in _extra_cases():
        graph, feed = build()
        try:
            expected = ref.execute(graph, feed)["y"]
            got = cuda.execute(graph, feed)["y"]
            np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
            print(f"  [PASS] {name}")
        except Exception as exc:  # noqa: BLE001 - 验证脚本要把任何异常都记成失败
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
    return failures


def _run_dropout_exact(cuda: CudaBackend, ref: ReferenceBackend) -> list:
    """dropout 是随机算子，但两边共用 SplitMix64，所以这里要求**逐位相等**。

    随机不是放松校验的理由：只要参考实现与内核用同一套位运算，掩码就该
    一个 bit 都不差。用 allclose 反而会把"某处悄悄换了随机源"放过去。
    """
    rng = np.random.default_rng(2024)
    failures = []

    for p in (0.0, 0.25, 0.9):
        g = Graph(f"drop{p}")
        x = g.input("x", f32[7, 13])
        g.output(g.dropout_mask(x, p=p, seed=987654321), "y")
        feed = {"x": _rand(rng, 7, 13)}
        name = f"dropout_p{p}_bitwise"
        try:
            np.testing.assert_array_equal(
                cuda.execute(g, feed)["y"], ref.execute(g, feed)["y"]
            )
            print(f"  [PASS] {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  [FAIL] {name}: {exc}")

    g = Graph("drop_train")
    x = g.input("x", f32[6, 10])
    mask = g.dropout_mask(x, p=0.3, seed=7)
    g.output(g.mul(x, mask), "y")
    feed = {"x": _rand(rng, 6, 10)}
    try:
        np.testing.assert_array_equal(
            cuda.execute(g, feed)["y"], ref.execute(g, feed)["y"]
        )
        print("  [PASS] dropout_train_bitwise")
    except Exception as exc:  # noqa: BLE001
        failures.append(("dropout_train_bitwise", f"{type(exc).__name__}: {exc}"))
        print(f"  [FAIL] dropout_train_bitwise: {exc}")

    return failures


def _run_f16_extra(cuda: CudaBackend, ref: ReferenceBackend) -> list:
    """融合内核的 f16 形态：存储减半、算术仍在 float 里。

    f16 的参考实现按 numpy 的 float16 算，CUDA 在 float 里算再舍入回 f16，
    两者本来就不是逐位关系，所以这里只要求落在半精度的合理误差内。
    """
    from aetherflow import f16

    rng = np.random.default_rng(31415)
    failures = []

    cases = []

    g = Graph("silu_gate_f16")
    x = g.input("x", f16[3, 7])
    gate = g.input("gate", f16[3, 7])
    g.output(g.silu(x, gate), "y")
    cases.append(
        (
            "silu_gate_f16",
            g,
            {
                "x": _rand(rng, 3, 7, scale=0.5).astype(np.float16),
                "gate": _rand(rng, 3, 7, scale=0.5).astype(np.float16),
            },
        )
    )

    g = Graph("layernorm_residual_f16")
    a = g.input("a", f16[4, 16])
    r = g.input("r", f16[4, 16])
    gamma = g.input("gamma", f16[16])
    beta = g.input("beta", f16[16])
    g.output(g.layernorm(a, gamma, beta, r, axes=(-1,)), "y")
    cases.append(
        (
            "layernorm_residual_f16",
            g,
            {
                "a": _rand(rng, 4, 16, scale=0.5).astype(np.float16),
                "r": _rand(rng, 4, 16, scale=0.5).astype(np.float16),
                "gamma": (_rand(rng, 16, scale=0.1) + 1.0).astype(np.float16),
                "beta": _rand(rng, 16, scale=0.1).astype(np.float16),
            },
        )
    )

    for name, graph, feed in cases:
        try:
            got = cuda.execute(graph, feed)["y"]
            expected = ref.execute(graph, feed)["y"]
            np.testing.assert_allclose(got, expected, rtol=2e-2, atol=2e-2)
            print(f"  [PASS] {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  [FAIL] {name}: {exc}")

    return failures


def _run_budget_check(cuda: CudaBackend) -> list:
    """编译期静态预算与捕获期峰值的对照。

    静态预算（``ir/liveness.py``）在图还没执行时就能报出容量需求；它假设
    每个中间值独占内存，而 CUDA 后端会把连续张量上的 reshape 实现成视图、
    arena 又复用已释放的块，所以实测峰值只会更小。预算若被实测超过，说明
    还有模型没覆盖的分配（工作区之类），必须查清楚。
    """
    from aetherflow.ir.liveness import plan_memory
    from aetherflow.models import BlockSpec, build_transformer_block

    spec = BlockSpec(d_model=32, num_heads=4, d_ff=64)
    g = Graph("budget")
    _, out = build_transformer_block(g, spec)
    g.output(out, "y")

    budget = plan_memory(g, symbols={"B": 2, "S": 5})
    print(
        f"  [info] 静态预算：peak_live={budget.peak_live_bytes} B, "
        f"intermediate_peak={budget.intermediate_peak_bytes} B, "
        f"reuse={budget.reuse_ratio:.2f}x"
    )

    rng = np.random.default_rng(2718)
    feed = {"x": _rand(rng, 2, 5, spec.d_model)}
    for name, shape in spec.param_shapes.items():
        if name == "scale":
            feed[name] = np.array([1.0 / np.sqrt(spec.head_dim)], dtype=np.float32)
        else:
            feed[name] = _rand(rng, *shape, scale=0.1)

    try:
        plan = cuda.capture(g, feed)
        captured = plan.peak_bytes
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] static_budget: {type(exc).__name__}: {exc}")
        return [("static_budget", f"{type(exc).__name__}: {exc}")]

    static = budget.aligned_intermediate_peak_bytes
    print(f"  [info] 捕获峰值：{captured} B（静态预算 {static} B，已按 512B 对齐）")
    # 两者不要求严格相等：对齐与后端小临时让实测偏大，连续张量上的
    # reshape 走视图又会让它偏小。超出 2 倍说明预算模型失真。
    if captured > static * 2 or static > captured * 2:
        message = f"捕获峰值 {captured} B 与静态预算 {static} B 相差超过 2 倍"
        print(f"  [FAIL] static_budget: {message}")
        return [("static_budget", message)]

    print("  [PASS] static_budget_matches_capture")
    return []


def _run_tuning_check(cuda: CudaBackend) -> list:
    """自动调优闭环：搜索一条配置，应用后与默认配置逐位一致。

    本地只能测搜索器；这一步证明插件真的把 block 接进了 launch，
    而且"应用配置"没有引入任何数值差异。
    """
    from benchmarks.tune import TUNE_CASES, apply_and_verify, search

    case = next(c for c in TUNE_CASES if c.name == "relu_1m")
    try:
        space = cuda.tuning_space("relu")
        store = search(cuda, cases=[case], rounds=2)
        hits, problems = apply_and_verify(cuda, store, cases=[case])
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] tuning: {type(exc).__name__}: {exc}")
        return [("tuning", f"{type(exc).__name__}: {exc}")]

    record = store.records()[0]
    print(
        f"  [info] relu 候选 {list(space.values('block'))}，"
        f"最优 {dict(record.config)} @ {record.median_ms:.4f} ms"
    )
    if problems:
        for item in problems:
            print(f"  [FAIL] tuning: {item}")
        return [("tuning", item) for item in problems]
    if hits != 1:
        print(f"  [FAIL] tuning: 期望命中 1 条配置，实际 {hits}")
        return [("tuning", f"命中 {hits} 条配置")]

    print("  [PASS] tuning_roundtrip")
    return []


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description="CUDA 后端远程验证")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="把 refused 的用例也视为失败（只对声称承接全部算子的后端有意义）",
    )
    args = parser.parse_args(argv)

    cuda = CudaBackend()
    _device_banner(cuda)

    print("\n== conformance ==")
    report = run_conformance(cuda)
    print(report.text())
    if not report.ok:
        print("\nconformance 未通过：上面标 FAIL 的检查或用例需要先修")
        return 1
    if args.strict and report.refused:
        print(f"\n--strict：仍有 {len(report.refused)} 个用例被拒绝")
        return 1

    print("\n== 新增内核的边界用例 ==")
    ref = ReferenceBackend()
    failures = _run_extra(cuda, ref)
    failures += _run_dropout_exact(cuda, ref)
    failures += _run_f16_extra(cuda, ref)
    failures += _run_budget_check(cuda)
    failures += _run_tuning_check(cuda)
    if failures:
        print(f"\n{len(failures)} 个边界用例失败")
        return 1

    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
