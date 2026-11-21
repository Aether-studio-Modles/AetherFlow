"""设备态解码：decode 的状态留在设备上，每步只回传 logits。

配套 DESIGN 5.279。四条判据：

1. **捕获一次**：同一批次形状的 decode 图只 capture 一次，跨请求复用。
2. **状态不往返**：decode 每步的 state 以设备数组喂回（D2D），不产生 D2H；
   只有 logits 块（``B*V`` 个 float）被取回主机。
3. **数值不变**：与宿主路径（``resident=True`` 的同一会话）和模块级
   ``generate`` 逐 token 一致。
4. **契约**：``device_state=True`` 必须配 ``resident=True``，后端必须提供
   capture / to_host / to_array；缺一项在构造期报错，不静默降级。

假后端把"设备数组"包一层，``to_host`` 因此必须被显式调用——这正是要
记账的动作。HIP 真机用例在没有设备时跳过。
"""

import unittest

import numpy as np

from aetherflow import ReferenceBackend
from aetherflow.models import (
    LMSpec,
    LMRunner,
    generate,
    generate_batch,
    generate_beam,
)

BACKEND = ReferenceBackend()


def _small(**overrides):
    base = dict(
        vocab_size=16,
        d_model=8,
        num_layers=1,
        d_state=4,
        d_conv=3,
        expand=2,
        steps=6,
        batch=2,
        tie_weights=True,
        pad_id=0,
        eos_id=1,
    )
    base.update(overrides)
    return LMSpec(**base)


class _Dev:
    """假后端里的设备数组：只有显式 ``to_host`` 才回主机。"""

    __slots__ = ("arr",)

    def __init__(self, arr):
        self.arr = np.asarray(arr)

    @property
    def shape(self):
        return self.arr.shape

    @property
    def dtype(self):
        return self.arr.dtype


class _Plan:
    """捕获结果的替身：replay 一次就是一次设备执行。"""

    def __init__(self, backend, graph):
        self.backend = backend
        self.graph = graph
        self.outputs = None

    def run_device(self, inputs):
        self.backend.replays += 1
        feed = {}
        for name, value in inputs.items():
            if isinstance(value, _Dev):
                self.backend.d2d += 1
                feed[name] = value.arr
            else:
                self.backend.h2d += 1
                feed[name] = np.asarray(value)
        out = self.backend._execute(self.graph, feed)
        self.outputs = {name: _Dev(value) for name, value in out.items()}
        return dict(self.outputs)

    def replay(self, inputs):
        self.run_device(inputs)

    def device_outputs(self):
        return dict(self.outputs or {})


class _DeviceRecorder:
    """包一层参考后端，模拟"设备数组 + 捕获计划"并记账。"""

    def __init__(self, inner):
        self.inner = inner
        self.resident = {}
        self.captures = 0
        self.replays = 0
        self.h2d = 0
        self.d2d = 0
        self.d2h = []

    def resident_parameters(self, graph, values):
        declared = {value.name for value in graph.inputs}
        unknown = set(values) - declared
        if unknown:
            raise KeyError(sorted(unknown))
        self.resident = dict(values)

    def release_parameters(self):
        self.resident = {}

    def _execute(self, graph, feed):
        merged = dict(self.resident)
        merged.update(feed)
        return self.inner.execute(graph, merged)

    def execute(self, graph, feed):
        """prefill 仍走普通执行，输出是主机 numpy。"""
        return self._execute(graph, feed)

    def capture(self, graph, sample_inputs):
        self.captures += 1
        return _Plan(self, graph)

    def to_array(self, value):
        return value if isinstance(value, _Dev) else _Dev(value)

    def to_host(self, value):
        if isinstance(value, _Dev):
            self.d2h.append((tuple(value.shape), str(value.dtype)))
            return value.arr
        return np.asarray(value)


class TestDeviceStateDecode(unittest.TestCase):
    """假后端：记账 + 数值一致。"""

    def _setup(self, device_state=True):
        spec = _small(batch=1, steps=4)
        params = spec.init_params(seed=21)
        return spec, params, [3, 4, 5, 6, 7]

    def test_matches_host_path_and_captures_once(self):
        spec, params, prompt = self._setup()
        kwargs = dict(max_new_tokens=4, chunk_size=2, temperature=0.7,
                      top_k=5, seed=9)
        want = list(generate(BACKEND, params, prompt, spec, **kwargs))

        rec = _DeviceRecorder(BACKEND)
        with LMRunner(rec, params, spec, resident=True, device_state=True) as runner:
            first = list(runner.generate(prompt, **kwargs))
            second = list(runner.generate(prompt, **kwargs))
            captures = rec.captures
            d2h = list(rec.d2h)
        self.assertEqual(first, want)
        self.assertEqual(second, want)
        self.assertEqual(captures, 1, "decode 图必须只捕获一次")
        self.assertTrue(d2h, "设备态 decode 仍要取回 logits")
        self.assertIsNone(runner._decode_plan, "close 必须丢掉捕获计划")

    def test_state_never_returns_to_host_during_decode(self):
        spec, params, prompt = self._setup()
        limit = 4
        rec = _DeviceRecorder(BACKEND)
        with LMRunner(rec, params, spec, resident=True, device_state=True) as runner:
            runner.generate(prompt, max_new_tokens=limit, chunk_size=2)
            d2h = list(rec.d2h)
            h2d = rec.h2d
            d2d = rec.d2d
            replays = rec.replays

        states = len(spec.state_shapes)
        logits = (1, 1, spec.vocab_size)
        # 首 token 来自 prefill 的主机 logits，decode 只剩 limit-1 步
        steps = limit - 1
        self.assertEqual(replays, steps)
        self.assertEqual(sum(1 for shape, _ in d2h if shape == logits), steps)
        self.assertEqual(
            [shape for shape, _ in d2h if shape != logits], [],
            "decode 期间不允许回传任何状态张量",
        )
        # 状态只在第一步从主机上传一次，此后每步都是 D2D
        self.assertEqual(h2d, 2 * steps + states)
        self.assertEqual(d2d, states * (steps - 1))

    def test_batch_session_matches_plain_call(self):
        spec = _small(batch=2, steps=4)
        params = spec.init_params(seed=22)
        prompts = [[3], [3, 4, 5, 6, 7]]
        want = generate_batch(
            BACKEND, params, prompts, spec, max_new_tokens=3, chunk_size=2
        )
        rec = _DeviceRecorder(BACKEND)
        with LMRunner(rec, params, spec, resident=True, device_state=True) as runner:
            got = runner.generate_batch(prompts, max_new_tokens=3, chunk_size=2)
        self.assertEqual(got, want)
        self.assertEqual(rec.captures, 1)

    def test_prime_prefix_still_decodes_on_device(self):
        spec, params, prompt = self._setup()
        kwargs = dict(max_new_tokens=3, chunk_size=2)
        want = list(generate(BACKEND, params, prompt, spec, **kwargs))
        rec = _DeviceRecorder(BACKEND)
        with LMRunner(rec, params, spec, resident=True, device_state=True) as runner:
            runner.prime(prompt[:3], chunk_size=2)
            got = list(runner.generate(prompt, **kwargs))
        self.assertEqual(got, want)
        self.assertEqual(rec.captures, 1)

    def test_beam_search_keeps_the_host_path(self):
        spec = _small(batch=1, steps=3, vocab_size=6)
        params = spec.init_params(seed=23)
        want = list(generate_beam(BACKEND, params, [3], spec, num_beams=2,
                                  max_new_tokens=2))
        rec = _DeviceRecorder(BACKEND)
        with LMRunner(rec, params, spec, resident=True, device_state=True) as runner:
            got = list(runner.generate_beam([3], num_beams=2, max_new_tokens=2))
        self.assertEqual(got, want)
        self.assertEqual(rec.captures, 0, "束搜索不建设备态计划")


class _ResidentOnly:
    """有驻留接口、没有捕获能力——用来验证构造期的显式拒绝。"""

    def __init__(self):
        self.resident = {}

    def resident_parameters(self, graph, values):
        self.resident = dict(values)

    def release_parameters(self):
        self.resident = {}


class _CaptureOnly(_ResidentOnly):
    """能捕获、但没有把输出取回主机的能力。"""

    def capture(self, graph, sample_inputs):
        return None


class TestDeviceStateContract(unittest.TestCase):
    def test_requires_resident(self):
        spec = _small(batch=1, steps=4)
        rec = _DeviceRecorder(BACKEND)
        with self.assertRaisesRegex(ValueError, "resident=True"):
            LMRunner(rec, spec.init_params(seed=0), spec, device_state=True)

    def test_requires_capture_capability(self):
        spec = _small(batch=1, steps=4)
        with self.assertRaisesRegex(ValueError, "capture"):
            LMRunner(
                _ResidentOnly(), spec.init_params(seed=0), spec,
                resident=True, device_state=True,
            )

    def test_requires_output_transfer(self):
        spec = _small(batch=1, steps=4)
        with self.assertRaisesRegex(ValueError, "to_host"):
            LMRunner(
                _CaptureOnly(), spec.init_params(seed=0), spec,
                resident=True, device_state=True,
            )


try:  # 与 test_device_control 相同的容错导入
    from aetherflow.backends.hip import HIP_AVAILABLE, HipBackend
except Exception:  # pragma: no cover - 没有 ROCm 的环境
    HIP_AVAILABLE, HipBackend = False, None


@unittest.skipUnless(HIP_AVAILABLE, "本机没有 HIP 设备")
class TestDeviceStateOnHip(unittest.TestCase):
    """真机：数值与宿主路径一致，decode 每步只回传 logits。"""

    def test_matches_host_path_and_only_logits_come_back(self):
        spec = _small(batch=1, steps=4, vocab_size=8)
        params = spec.init_params(seed=24)
        prompt = [3, 4, 5]
        limit = 3
        want = list(generate(
            HipBackend(), params, prompt, spec, max_new_tokens=limit, chunk_size=2
        ))

        backend = HipBackend()
        sizes = []
        original = backend.to_host

        def counting(value):
            sizes.append(tuple(np.shape(value)))
            return original(value)

        backend.to_host = counting
        with LMRunner(
            backend, params, spec, resident=True, device_state=True
        ) as runner:
            got = list(runner.generate(prompt, max_new_tokens=limit, chunk_size=2))

        self.assertEqual(got, want)
        logits = (1, 1, spec.vocab_size)
        # prefill 的尾块取回一次 logits，decode 每步一次
        self.assertEqual(sizes.count(logits), limit)
        # 状态形状只出现在 prefill 的两块里；decode 没有再取回它们
        state_like = [shape for shape in sizes if shape != logits]
        self.assertEqual(len(state_like), 2 * len(spec.state_shapes))


if __name__ == "__main__":
    unittest.main()
