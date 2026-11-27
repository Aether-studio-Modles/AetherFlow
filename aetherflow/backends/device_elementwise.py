"""逐元素算子：二元 / 一元模板、广播、阶跃、常量与 dropout 掩码。"""

from __future__ import annotations

from typing import Any, List
from ..ir.types import DType
from ..runtime.bf16 import pack_bf16
from .base import BackendCapabilityError
import numpy as np


class _ElementwiseMixin:
    """逐元素算子：二元 / 一元模板、广播、阶跃、常量与 dropout 掩码。"""

    # ------------------------------------------------------------------
    # 逐元素
    # ------------------------------------------------------------------

    def _op_add(self, node, args, attrs):
        return self._binary("add", args)

    def _op_sub(self, node, args, attrs):
        return self._binary("sub", args)

    def _op_mul(self, node, args, attrs):
        return self._binary("mul", args)

    def _op_div(self, node, args, attrs):
        return self._binary("div", args)

    # 比较算子：与二元算术共用广播骨架，只把输出换成 bool、运算符换成比较。
    def _op_eq(self, node, args, attrs):
        return self._binary("eq", args, compare=True)

    def _op_ne(self, node, args, attrs):
        return self._binary("ne", args, compare=True)

    def _op_lt(self, node, args, attrs):
        return self._binary("lt", args, compare=True)

    def _op_le(self, node, args, attrs):
        return self._binary("le", args, compare=True)

    def _op_gt(self, node, args, attrs):
        return self._binary("gt", args, compare=True)

    def _op_ge(self, node, args, attrs):
        return self._binary("ge", args, compare=True)

    def _op_where(self, node, args, attrs):
        """``where(cond, a, b)``：cond 是 bool，a / b 同 dtype（输出跟随它）。"""
        cond, a, b = args
        out_shape = tuple(np.broadcast_shapes(cond.shape, a.shape, b.shape))
        if cond.shape == a.shape == b.shape:
            return self._where_kernel(cond, a, b)
        return self._where_broadcast(cond, a, b, out_shape)

    def _where_kernel(self, cond, a, b):
        cp = self.cp
        out = cp.empty(a.shape, dtype=a.dtype)
        n = int(a.size)
        if n == 0:
            return out
        tensor = a
        dtype_key = self._kind(tensor)
        kernel = self._kernel(
            ("where", dtype_key),
            lambda: self.codegen.where_source(dtype=tensor.dtype),
            "ew_where",
        )
        block, grid = self._grid("where", n)
        kernel((grid,), (block,), (cond, a, b, out, np.int64(n)))
        return out

    def _where_broadcast(self, cond, a, b, out_shape: tuple):
        cp = self.cp
        cond = self._contig(cond)
        a = self._contig(a)
        b = self._contig(b)
        out = cp.empty(out_shape, dtype=a.dtype)
        n = int(out.size)
        if n == 0:
            return out

        groups = self._collapse_dims(
            out_shape,
            self._elem_strides(cond.shape, out_shape),
            self._elem_strides(a.shape, out_shape),
            self._elem_strides(b.shape, out_shape),
        )
        if len(groups) > 4:
            raise BackendCapabilityError(
                f"{self._LABEL}的广播最多支持 4 个不可折叠维度，"
                f"where 收到 {cond.shape} / {a.shape} / {b.shape}"
                f"（折叠后 {len(groups)} 维）"
            )

        dims = [1, 1, 1, 1]
        sc = [0, 0, 0, 0]
        sa = [0, 0, 0, 0]
        sb = [0, 0, 0, 0]
        for i, (dim, step_c, step_a, step_b) in enumerate(groups):
            dims[i], sc[i], sa[i], sb[i] = dim, step_c, step_a, step_b

        dtype_key = self._kind(a)
        kernel = self._kernel(
            ("where_bcast", dtype_key),
            lambda: self.codegen.where_source(broadcast=True, dtype=a.dtype),
            "ew_where_bcast",
        )
        block, grid = self._grid("where", n)
        kernel(
            (grid,),
            (block,),
            (
                cond,
                a,
                b,
                out,
                np.int64(n),
                np.int32(len(groups)),
                *[np.int64(v) for v in dims],
                *[np.int64(v) for v in sc],
                *[np.int64(v) for v in sa],
                *[np.int64(v) for v in sb],
            ),
        )
        return out

    def _op_clamp(self, node, args, attrs):
        """按属性里的标量上下界截断；缺省的一侧由 ``has_lo`` / ``has_hi`` 关掉。

        标量通过内核参数传入而不是写进源码：同一张图里 ``clamp(min=-80)``
        与 ``clamp(0, 1)`` 很常见，按边界值生成源码会让编译缓存按数值增长。
        """
        cp = self.cp
        x = args[0]
        out = cp.empty_like(x)
        n = int(x.size)
        if n == 0:
            return out
        lo = attrs.get("min")
        hi = attrs.get("max")
        kernel = self._kernel(
            ("clamp", self._kind(x)),
            lambda: self.codegen.render(self.codegen.clamp_source(), x.dtype),
            "ew_clamp",
        )
        block, grid = self._grid("clamp", n)
        kernel(
            (grid,),
            (block,),
            (
                x,
                out,
                np.int64(n),
                np.int32(lo is not None),
                np.int32(hi is not None),
                np.float32(0.0 if lo is None else lo),
                np.float32(0.0 if hi is None else hi),
            ),
        )
        return out

    def _op_mod(self, node, args, attrs):
        """整数取模（floor 语义）。

        只承接同形与标量广播：整数逐元素内核没有做多维 stride 广播的版本，
        而 Engram 的哈希两侧同形、模数是标量。其它广播形态显式拒绝，避免
        落到"按 f32 解释整数缓冲"的错误路径上。
        """
        a, b = args
        try:
            out_shape = tuple(np.broadcast_shapes(a.shape, b.shape))
        except ValueError as exc:
            raise BackendCapabilityError(
                f"{self._LABEL}的 mod 无法广播 {a.shape} 与 {b.shape}"
            ) from exc
        if a.shape == b.shape:
            return self._mod_kernel(a, b, mode=0)
        if a.size == 1 and out_shape == b.shape:
            return self._mod_kernel(a, b, mode=1)
        if b.size == 1 and out_shape == a.shape:
            return self._mod_kernel(a, b, mode=2)
        raise BackendCapabilityError(
            f"{self._LABEL}的 mod 只承接同形与标量广播，收到 {a.shape} 与 {b.shape}"
        )

    def _mod_kernel(self, a, b, mode: int):
        cp = self.cp
        tensor = b if mode == 1 else a
        out = cp.empty_like(tensor)
        n = int(tensor.size)
        if n == 0:
            return out
        kernel = self._kernel(
            ("mod", mode),
            lambda: self.codegen.render(
                self.codegen.mod_source(scalar_a=mode == 1, scalar_b=mode == 2),
                # 整数内核把参数类型钉死成 int，dtype 只为拿到方言 prelude；
                # 必须传 np.dtype 实例（codegen 按 ``.name`` 取 f32 / f16），
                # 传 DType 枚举或 ``np.float32`` 类都会被拒绝。
                np.dtype("float32"),
            ),
            "ew_mod",
        )
        lhs = a.reshape(1) if mode == 1 else a
        rhs = b.reshape(1) if mode == 2 else b
        block, grid = self._grid("mod", n)
        kernel((grid,), (block,), (lhs, rhs, out, np.int64(n)))
        return out

    def _op_relu(self, node, args, attrs):
        return self._unary("relu", args)

    def _op_silu(self, node, args, attrs):
        if len(args) == 1:
            return self._unary("silu", args)
        cp = self.cp
        x, gate = args
        if tuple(x.shape) != tuple(gate.shape) or x.dtype != gate.dtype:
            raise BackendCapabilityError(
                f"silu 的 gate 必须与 x 同形状、同 dtype，收到 {x.shape} 与 {gate.shape}"
            )
        out = cp.empty_like(x)
        n = int(x.size)
        if n == 0:
            return out
        kernel = self._kernel(
            ("silu_gate", self._kind(x)),
            lambda: self.codegen.render(self.codegen.silu_gate_source(), x.dtype),
            "silu_gate",
        )
        block, grid = self._grid("silu", n)
        kernel((grid,), (block,), (x, gate, out, np.int64(n)))
        return out

    def _op_exp(self, node, args, attrs):
        return self._unary("exp", args)

    def _op_log(self, node, args, attrs):
        return self._unary("log", args)

    def _op_sqrt(self, node, args, attrs):
        return self._unary("sqrt", args)

    def _op_neg(self, node, args, attrs):
        return self._unary("neg", args)

    def _op_sign(self, node, args, attrs):
        return self._unary("sign", args)

    def _op_round(self, node, args, attrs):
        return self._unary("round", args)

    def _op_floor(self, node, args, attrs):
        return self._unary("floor", args)

    def _op_ceil(self, node, args, attrs):
        return self._unary("ceil", args)

    def _op_tanh(self, node, args, attrs):
        return self._unary("tanh", args)

    def _op_sigmoid(self, node, args, attrs):
        return self._unary("sigmoid", args)

    def _op_erf(self, node, args, attrs):
        """误差函数；精确 GELU 的组成部分，走一元模板。"""
        return self._unary("erf", args)

    def _op_gelu(self, node, args, attrs):
        """tanh 近似 GELU，与 ir/ops.py 的参考式逐位同式，走一元模板。"""
        return self._unary("gelu", args)

    def _op_step(self, node, args, attrs):
        return self._step(args, attrs, "step")

    def _op_spike(self, node, args, attrs):
        """前向与 step 逐位相同（都是"越过阈值发 1"）。

        差异只在反向：`spike` 的 vjp 是替代梯度，而反向图由 IR 层的规则组合，
        后端不需要知道这件事。所以这里直接复用同一个 kernel。
        调优键仍需分开：两者是不同的 IR 算子，各自可以被单独调参。
        """
        return self._step(args, attrs, "spike")

    def _step(self, args, attrs, op: str):
        cp = self.cp
        x = args[0]
        out = cp.empty_like(x)
        n = int(x.size)
        if n == 0:
            return out
        kind = self._kind(x)
        kernel = self._kernel(
            ("step", kind),
            lambda: self.codegen.render(self.codegen.step_source(), x.dtype),
            "ew_step",
        )
        block, grid = self._grid(op, n)
        kernel(
            (grid,),
            (block,),
            (x, out, np.int64(n), np.float32(float(attrs.get("threshold", 0.0)))),
        )
        return out

    def _op_dropout_mask(self, node, args, attrs):
        """生成缩放后的 dropout 掩码（保留 ``1/(1-p)``，丢弃 0）。

        与参考实现共用 SplitMix64 与"取高 24 位"的映射，所以两边可以逐位比较。
        输入 x 的数值不读，只取形状——它在图里仍是真实的边，这样输出的形状
        能跟着符号 batch 维走，而不必把形状写进 attrs。
        """
        cp = self.cp
        x = args[0]
        p = float(attrs.get("p", 0.5))
        if not 0.0 <= p < 1.0:
            raise BackendCapabilityError(f"dropout 的 p 必须在 [0, 1) 内，收到 {p}")
        out = cp.empty(x.shape, dtype=x.dtype)
        n = int(out.size)
        if n == 0:
            return out
        offset = int(attrs.get("offset", 0)) & 0xFFFFFFFFFFFFFFFF
        kind = self._kind(x)
        if len(args) > 1:
            # 运行期 key：随机源在设备内存里。标量参数会被 CUDA Graph 按值
            # 冻结，捕获后的 replay 就再也换不了掩码——所以 key 必须是指针。
            key = cp.ascontiguousarray(args[1]).reshape(1)
            kernel = self._kernel(
                ("dropout_mask_keyed", kind),
                lambda: self.codegen.render(self.codegen.dropout_mask_keyed_template(), x.dtype),
                "dropout_mask_keyed",
            )
            block, grid = self._grid("dropout_mask", n)
            kernel(
                (grid,),
                (block,),
                (
                    out,
                    np.int64(n),
                    np.float32(p),
                    np.float32(1.0 / (1.0 - p)),
                    key,
                    np.uint64(offset),
                ),
            )
            return out
        seed = (int(attrs.get("seed", 0)) + int(offset)) & 0xFFFFFFFFFFFFFFFF
        kernel = self._kernel(
            ("dropout_mask", kind),
            lambda: self.codegen.render(self.codegen.dropout_mask_source(), x.dtype),
            "dropout_mask",
        )
        block, grid = self._grid("dropout_mask", n)
        kernel(
            (grid,),
            (block,),
            (
                out,
                np.int64(n),
                np.float32(p),
                np.float32(1.0 / (1.0 - p)),
                np.uint64(seed),
            ),
        )
        return out

    def _op_constant(self, node, args, attrs):
        """常量节点：值在属性里，直接物化一份设备张量。

        零输入节点绕过了 ``invoke`` 里对输入 dtype 的检查，所以这里自己查一次，
        保持「内核只吃 f32/f16/bf16」这条规则没有例外。

        i32 常量是唯一的非浮点例外：它不进任何算术内核，只作为索引表喂给
        ``embedding`` / ``gather``（HelixLM 的"按通道分组选尺度"就是这样）。
        参考后端本来就支持，GPU 后端拒绝物化会让这类模型连构图都跑不起来。

        bool 常量同样只是搬运：它来自常量折叠（例如整张图里的掩码比较），
        只作 ``where`` 的 cond，不参与算术（算术在 IR 层就被拒绝）。
        """
        dtype = DType(attrs["dtype"])
        if dtype is DType.BF16:
            if not self._bf16_ops:
                raise BackendCapabilityError(f"{self._LABEL}未声明 bf16 支持，constant 收到 bf16")
            # bf16 常量以 uint16 位模式驻留：先在主机把值落到 bf16 格点再打包，
            # 与 ``quantize`` 在节点边界上做的事逐位一致。
            shape = tuple(int(d) for d in attrs["shape"])
            data = attrs.get("data")
            if data is None:
                host = np.full(shape, np.float32(attrs["value"]), dtype=np.float32)
            else:
                host = np.asarray(data, dtype=np.float32).reshape(shape)
            return self.cp.asarray(pack_bf16(host))
        if dtype is not DType.I32 and dtype not in (DType.F32, DType.F16, DType.BOOL):
            raise BackendCapabilityError(
                f"{self._LABEL}只支持 float32/float16/bf16、索引用的 int32 与 bool，"
                f"constant 收到 {dtype.value}"
            )
        np_dtype = {
            DType.F32: np.float32,
            DType.F16: np.float16,
            DType.I32: np.int32,
            DType.BOOL: np.bool_,
        }[dtype]
        shape = tuple(int(d) for d in attrs["shape"])
        data = attrs.get("data")
        if data is None:
            return self.cp.full(shape, np_dtype(attrs["value"]), dtype=np_dtype)
        # 常量折叠的产物是任意张量，不一定是同值填充；先按主机 numpy 构造
        # 再整块拷到设备，避免在 GPU 上逐元素赋值。
        host = np.asarray(data, dtype=np_dtype).reshape(shape)
        return self.cp.array(host, dtype=np_dtype)

    def _op_broadcast_to(self, node, args, attrs):
        """显式广播。物化成连续内存：后续 kernel 都要求输入连续。"""
        target = tuple(int(d) for d in attrs["shape"])
        out = self.cp.ascontiguousarray(self.cp.broadcast_to(args[0], target))
        # 目标形状与输入相同时广播退化成恒等，拿到的是同一个对象或视图
        return self._shared_view(out, args[0], node)

    def _binary(self, kind: str, args: List[Any], *, compare: bool = False):
        a, b = args
        try:
            out_shape = tuple(np.broadcast_shapes(a.shape, b.shape))
        except ValueError as exc:
            raise BackendCapabilityError(
                f"无法广播的二元运算：{kind} 收到 {a.shape} 与 {b.shape}"
            ) from exc

        if a.shape == b.shape:
            return self._binary_kernel(kind, a, b, mode=0, compare=compare)
        # 标量快速路径：只有当广播结果形状恰好等于另一侧形状时才能直接用
        # （例如 (5,) 与 (1,1) 的广播结果是 (1,5)，不能走这条路径）
        if a.size == 1 and out_shape == b.shape:
            return self._binary_kernel(kind, a, b, mode=1, compare=compare)  # a 是标量
        if b.size == 1 and out_shape == a.shape:
            return self._binary_kernel(kind, a, b, mode=2, compare=compare)  # b 是标量
        return self._binary_broadcast(kind, a, b, out_shape, compare=compare)

    def _binary_kernel(self, kind: str, a, b, mode: int, *, compare: bool = False):
        cp = self.cp
        tensor = b if mode == 1 else a
        # 比较的输出容器是 bool：它比输入窄，不能 empty_like 跟随输入。
        out = (
            cp.empty(tensor.shape, dtype=cp.bool_)
            if compare
            else cp.empty_like(tensor)
        )
        n = int(tensor.size)
        if n == 0:
            return out
        scalar_a, scalar_b = mode == 1, mode == 2
        dtype_key = self._kind(tensor)
        if compare:
            kernel = self._kernel(
                ("cmp", kind, dtype_key, mode),
                lambda: self.codegen.compare_source(
                    kind, scalar_a=scalar_a, scalar_b=scalar_b, dtype=tensor.dtype
                ),
                "ew_cmp",
            )
        else:
            kernel = self._kernel(
                ("bin", kind, dtype_key, mode),
                lambda: self.codegen.render(
                    self.codegen.binary_source(kind, scalar_a, scalar_b), tensor.dtype
                ),
                "ew_bin",
            )
        # 标量侧直接传设备指针，不在主机上取值：避免一次强制同步（见 codegen 注释）
        lhs = a.reshape(1) if scalar_a else a
        rhs = b.reshape(1) if scalar_b else b
        block, grid = self._grid(kind, n)
        kernel((grid,), (block,), (lhs, rhs, out, np.int64(n)))
        return out

    def _binary_broadcast(self, kind: str, a, b, out_shape: tuple, *,
                          compare: bool = False):
        """多维广播：把两侧 stride 对齐到输出形状，广播维 stride 置 0。

        广播的维度会被折叠（相邻且两侧 stride 都连续的维合并成一个），
        通常 (B,S,D) 与 (D,) 折叠后只剩 2 维。折叠后仍超过 4 维则明确拒绝。
        """
        cp = self.cp
        a = self._contig(a)
        b = self._contig(b)
        out = (
            cp.empty(out_shape, dtype=cp.bool_)
            if compare
            else cp.empty(out_shape, dtype=a.dtype)
        )
        n = int(out.size)
        if n == 0:
            return out

        groups = self._collapse_dims(
            out_shape,
            self._elem_strides(a.shape, out_shape),
            self._elem_strides(b.shape, out_shape),
        )
        if len(groups) > 4:
            raise BackendCapabilityError(
                f"{self._LABEL}的广播最多支持 4 个不可折叠维度，"
                f"{kind} 收到 {a.shape} 与 {b.shape}（折叠后 {len(groups)} 维）"
            )

        dims = [1, 1, 1, 1]
        sa = [0, 0, 0, 0]
        sb = [0, 0, 0, 0]
        for i, (dim, step_a, step_b) in enumerate(groups):
            dims[i], sa[i], sb[i] = dim, step_a, step_b

        dtype_key = self._kind(a)
        if compare:
            kernel = self._kernel(
                ("cmp_bcast", kind, dtype_key),
                lambda: self.codegen.compare_source(
                    kind, broadcast=True, dtype=a.dtype
                ),
                "ew_cmp_bcast",
            )
        else:
            kernel = self._kernel(
                ("bcast", kind, dtype_key),
                lambda: self.codegen.render(
                    self.codegen.broadcast_binary_source(kind), a.dtype
                ),
                "ew_bcast",
            )
        block, grid = self._grid(kind, n)
        kernel(
            (grid,),
            (block,),
            (
                a,
                b,
                out,
                np.int64(n),
                np.int32(len(groups)),
                *[np.int64(v) for v in dims],
                *[np.int64(v) for v in sa],
                *[np.int64(v) for v in sb],
            ),
        )
        return out

    def _unary(self, kind: str, args: List[Any]):
        cp = self.cp
        x = args[0]
        out = cp.empty_like(x)
        n = int(x.size)
        if n == 0:
            return out
        dtype_key = self._kind(x)
        kernel = self._kernel(
            ("un", kind, dtype_key),
            lambda: self.codegen.render(self.codegen.unary_source(kind), x.dtype),
            "ew_un",
        )
        block, grid = self._grid(kind, n)
        kernel((grid,), (block,), (x, out, np.int64(n)))
        return out
