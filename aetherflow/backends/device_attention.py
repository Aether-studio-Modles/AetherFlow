"""注意力与 softmax：末维 / 任意轴 softmax 与分块 online softmax。"""

from __future__ import annotations

from ..lower.plan import SHARED_LIMIT, attention_block, cdiv
from .base import BackendCapabilityError
import numpy as np
from .device_shared import _mask_shared_over_heads


class _AttentionMixin:
    """注意力与 softmax：末维 / 任意轴 softmax 与分块 online softmax。"""

    # ------------------------------------------------------------------
    # 归约：只支持最后一维
    # ------------------------------------------------------------------

    def _op_softmax(self, node, args, attrs):
        cp = self.cp
        x = args[0]
        # 可选的第二输入是标量缩放（融合 pass 把 softmax(mul(x, s)) 合并成 softmax(x, s)）。
        # 一律以设备指针传给 kernel：不取值到主机，否则每次 execute 都会同步一次。
        scale = args[1].reshape(1).astype(cp.float32, copy=False) if len(args) > 1 else self._one()
        rows, cols = self._last_dim_2d(x, "softmax")
        axis = int(attrs.get("axis", -1)) % x.ndim
        if axis != x.ndim - 1:
            return self._softmax_any_axis(x, axis, scale)
        out = cp.empty_like(x)
        if x.size == 0:
            return out
        dtype_key = self._kind(x)
        kernel = self._kernel(
            ("softmax", dtype_key),
            lambda: self.codegen.render(self.codegen.softmax_source(), x.dtype),
            "softmax_last",
        )
        block = self._reduce_block("softmax", cols)
        kernel(
            (rows,),
            (block,),
            (x, out, np.int64(rows), np.int64(cols), scale),
        )
        return out

    def _softmax_any_axis(self, x, axis: int, scale):
        """非末维 softmax：reshape 成 ``(outer, reduce, inner)``，一线程一行。

        行内元素跨 ``inner`` 步，块内归约拿不到合并访问，所以这里直接串行扫：
        最大值、``exp`` 求和、写回各一遍。``exp`` 因此算两遍——换来的是不需要
        shared 与行内同步。末维仍走 ``softmax_last``。

        沿空轴归一化没有定义（numpy 给整行 nan）：与其返回 nan，不如在边界拒绝。
        """
        cp = self.cp
        x = self._contig(x)
        if x.ndim == 0:
            raise BackendCapabilityError(f"{self._LABEL}的 softmax 需要至少一维")
        shape = tuple(int(dim) for dim in x.shape)
        outer = 1
        for dim in shape[:axis]:
            outer *= dim
        reduce_len = shape[axis]
        inner = 1
        for dim in shape[axis + 1 :]:
            inner *= dim
        if reduce_len == 0:
            raise BackendCapabilityError(
                f"{self._LABEL}的 softmax 不能沿空轴归一化（形状 {shape}，axis={axis}）"
            )
        out = cp.empty_like(x)
        if out.size == 0:
            return out
        total = outer * inner
        kernel = self._kernel(
            ("softmax_axis", self._kind(x)),
            lambda: self.codegen.render(self.codegen.softmax_axis_source(), x.dtype),
            "softmax_axis",
        )
        block = self._reduce_block("softmax", total)
        kernel(
            (cdiv(total, block),),
            (block,),
            (x, out, np.int64(total), np.int64(reduce_len), np.int64(inner), scale),
        )
        return out

    def _op_attention(self, node, args, attrs):
        """融合注意力前向：分块 online softmax，不物化 ``(S, T)`` 分数矩阵。

        与参考语义的唯一区别是实现方式：这里按 key 分块、边算边重缩放，
        分数矩阵从不落盘。mask 分两种：按 ``(b, h)`` 可共享的（因果 mask 的
        ``(1,1,S,T)``）只保留一份 ``(S, T)``，由内核按 ``mask_shared`` 寻址；
        只有真正随 batch/head 变化的 mask 才广播成 ``(BH, S, T)``。广播规则
        因此仍与 numpy 一致，但共享情形不再多出 ``BH-1`` 份二次内存。

        f16 是**混合精度**：q/k/v/mask/out 用 f16 存储，点积、在线 softmax
        的行统计与输出累加器全在 float 里做，只在写回 ``out`` 时舍入一次。
        共享内存布局因此不随 dtype 变化（``smem`` 恒为 float）。
        """
        cp = self.cp
        q, k, v = args[0], args[1], args[2]
        s_len = int(q.shape[-2])
        d = int(q.shape[-1])
        t_len = int(k.shape[-2])
        e = int(v.shape[-1])
        batch = tuple(int(x) for x in q.shape[:-2])
        bh = int(np.prod(batch)) if batch else 1
        out = cp.empty(batch + (s_len, e), dtype=q.dtype)
        if out.size == 0:
            return out
        if t_len == 0:
            # 空序列的 softmax 没有定义；与其返回一行 nan，不如在边界处拒绝。
            raise BackendCapabilityError(
                f"attention 的 k/v 序列长度为 0（节点 %{node.output.name}），softmax 无定义"
            )
        if len(args) > 3 and args[3].dtype != q.dtype:
            # IR 的 infer 已经要求 mask 与 q 同 dtype；这里是运行期的最后一道
            # 边界——内核按同一个存储类型读 q 与 mask，混用会静默读错位宽。
            raise BackendCapabilityError(
                f"attention 的 mask 必须与 q 同 dtype，收到 {args[3].dtype} 与 "
                f"{q.dtype}（节点 %{node.output.name}）"
            )
        q3 = cp.ascontiguousarray(q).reshape(bh, s_len, d)
        k3 = cp.ascontiguousarray(k).reshape(bh, t_len, d)
        v3 = cp.ascontiguousarray(v).reshape(bh, t_len, e)
        if len(args) > 3:
            raw_mask = args[3]
            # 与参考实现一致：先去掉长度 1 的多余前导维（如 (1,1,S,T)）。
            # cupy 的 broadcast_to 不允许目标秩比输入小，直接传会报
            # "more dimensions than allowed"。
            while raw_mask.ndim > len(batch) + 2:
                if raw_mask.shape[0] != 1:
                    raise BackendCapabilityError(
                        f"attention 的 mask 前导维必须是 1，得到 {tuple(raw_mask.shape)}"
                        f"（节点 %{node.output.name}）"
                    )
                raw_mask = raw_mask[0]
            # 共享形态：(S,T) 或前导维全为 1。此时每个 (b,h) 读同一份，复制成
            # (BH,S,T) 只会多出 BH-1 份 S*T 的显存。
            if _mask_shared_over_heads(raw_mask.shape, s_len, t_len):
                m3 = cp.ascontiguousarray(raw_mask).reshape(s_len, t_len)
                mask_shared = 1
            else:
                try:
                    mask = cp.broadcast_to(raw_mask, batch + (s_len, t_len))
                except ValueError as exc:
                    raise BackendCapabilityError(
                        f"attention 的 mask 形状 {tuple(args[3].shape)} 无法广播到 "
                        f"{batch + (s_len, t_len)}（节点 %{node.output.name}）"
                    ) from exc
                m3 = cp.ascontiguousarray(mask).reshape(bh, s_len, t_len)
                mask_shared = 0
            has_mask = 1
        else:
            # has_mask=0 时内核不会解引用这个指针，但也不能传 None。
            m3 = q3
            has_mask = 0
            mask_shared = 0

        scale = np.float32(float(attrs.get("scale", 1.0)))
        p_drop = float(attrs.get("dropout", 0.0))
        # 缩放因子在主机侧算好：内核里每个元素都要用它，没必要每线程重算。
        drop_scale = np.float32(1.0 / (1.0 - p_drop)) if p_drop > 0.0 else np.float32(1.0)
        seed = np.uint64(int(attrs.get("seed", 0)) & 0xFFFFFFFFFFFFFFFF)

        block = attention_block(t_len)
        smem = 2 * block * 4 + e * 4
        if smem > SHARED_LIMIT:
            raise BackendCapabilityError(
                f"attention 的 shared memory 需求 {smem} 字节超过上限 "
                f"{SHARED_LIMIT}（E={e}，block={block}）"
            )
        kernel = self._kernel(
            ("attention", self._kind(q)),
            lambda: self.codegen.render(self.codegen.attention_source(), q.dtype),
            "attention_fwd",
        )
        kernel(
            (s_len, bh),
            (block,),
            (
                q3,
                k3,
                v3,
                m3,
                out.reshape(bh, s_len, e),
                np.int64(s_len),
                np.int64(t_len),
                np.int64(d),
                np.int64(e),
                scale,
                np.float32(p_drop),
                drop_scale,
                seed,
                np.int32(has_mask),
                np.int32(mask_shared),
            ),
            shared_mem=smem,
        )
        return out
