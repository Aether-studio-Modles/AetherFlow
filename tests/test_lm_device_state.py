"""设备态解码：decode 的状态留在设备上，每步只回传 logits。

配套 DESIGN 5.279～5.282。判据：

1. **捕获一次**：同一批次形状的 decode 图只 capture 一次，跨请求复用；
   束搜索另有一张按父行号重排状态的 gather 图，同样只捕获一次。
2. **状态不往返**：decode 每步的 state 以设备数组喂回（D2D），不产生 D2H；
   只有 logits 块（``B*V`` 个 float）被取回主机。束搜索的状态重排也在
   设备上做，主机只上传 ``k`` 个父行号。
3. **数值不变**：与宿主路径（``resident=True`` 的同一会话）和模块级
   ``generate`` / ``generate_beam`` 逐 token 一致。
4. **契约**：``device_state=True`` 必须配 ``resident=True``，后端必须提供
   capture / to_host / to_array；缺一项在构造期报错，不静默降级。
"""

import unittest

import numpy as np

from aetherflow import ReferenceBackend
from aetherflow.models import (
    LMSpec,
    LMRunner,
    ServingEngine,
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
        with LMRunner(rec, params, spec, resident=True, device_state=False) as runner:
            got = list(runner.generate_beam([3], num_beams=2, max_new_tokens=2))
        self.assertEqual(got, want)
        self.assertEqual(rec.captures, 0, "宿主态会话不捕获任何计划")


class TestDeviceStateBeam(unittest.TestCase):
    """束搜索的设备态：重排走 gather 图，decode 走同一套捕获计划。"""

    def _setup(self):
        spec = _small(batch=1, steps=4, vocab_size=8)
        return spec, spec.init_params(seed=25), [3, 4, 5]

    def test_matches_host_beam_for_each_width(self):
        for beams in (1, 2, 4):
            with self.subTest(beams=beams):
                spec, params, prompt = self._setup()
                kwargs = dict(max_new_tokens=4, chunk_size=2, length_penalty=0.7,
                              early_stopping=False)
                want = list(generate_beam(
                    BACKEND, params, prompt, spec, num_beams=beams, **kwargs
                ))
                rec = _DeviceRecorder(BACKEND)
                with LMRunner(
                    rec, params, spec, resident=True, device_state=True
                ) as runner:
                    got = list(runner.generate_beam(prompt, num_beams=beams, **kwargs))
                    swap_plan = runner._swap_plan
                self.assertEqual(got, want)
                self.assertIsNotNone(swap_plan, "设备态束搜索必须捕获重排图")
                # 两张图各捕获一次：重排 + decode
                self.assertEqual(rec.captures, 2)

    def test_state_swap_stays_on_device(self):
        spec, params, prompt = self._setup()
        limit, beams = 4, 2
        rec = _DeviceRecorder(BACKEND)
        with LMRunner(rec, params, spec, resident=True, device_state=True) as runner:
            runner.generate_beam(
                prompt, num_beams=beams, max_new_tokens=limit, chunk_size=2,
                early_stopping=False,
            )
            d2h = list(rec.d2h)
            replays = rec.replays

        # 每步两次 replay：重排图 + decode 图
        self.assertEqual(replays, 2 * (limit - 1), "decode 只剩 limit-1 步")
        state_shapes = {
            tuple(shape) for shape in spec.with_batch(beams).state_shapes.values()
        }
        self.assertEqual(
            [shape for shape, _ in d2h if shape in state_shapes], [],
            "束搜索不允许把状态取回主机",
        )
        # 每次 decode 只回传一块 logits，而不是 k 个状态张量
        logits_block = (beams, 1, spec.vocab_size)
        self.assertEqual(
            sum(1 for shape, _ in d2h if shape == logits_block), limit - 1
        )

    def test_alternate_settings_match_host(self):
        spec, params, prompt = self._setup()
        for kwargs in (
            dict(length_penalty=0.0, early_stopping=False),
            dict(repetition_penalty=1.3, max_new_tokens=6),
        ):
            with self.subTest(**kwargs):
                options = dict(max_new_tokens=3, chunk_size=2)
                options.update(kwargs)
                want = list(generate_beam(
                    BACKEND, params, prompt, spec, num_beams=2, **options
                ))
                rec = _DeviceRecorder(BACKEND)
                with LMRunner(
                    rec, params, spec, resident=True, device_state=True
                ) as runner:
                    got = list(runner.generate_beam(prompt, num_beams=2, **options))
                self.assertEqual(got, want)

    def test_eos_on_first_step_matches_host(self):
        spec, params, prompt = self._setup()
        # 把贪心首 token 当 EOS：得分最高的那条假设第一步就完成
        probe = generate(BACKEND, params, prompt, spec, max_new_tokens=1,
                         temperature=0.0)
        eos = int(probe[0]) if probe else spec.eos_id
        options = dict(num_beams=2, max_new_tokens=4, chunk_size=2, eos_id=eos)
        want = list(generate_beam(BACKEND, params, prompt, spec, **options))
        rec = _DeviceRecorder(BACKEND)
        with LMRunner(rec, params, spec, resident=True, device_state=True) as runner:
            got = list(runner.generate_beam(prompt, **options))
        self.assertEqual(got, want)
        self.assertEqual(got, [], "首步 EOS 的那条假设得分最高，应返回空序列")

    def test_close_drops_both_plans(self):
        spec, params, prompt = self._setup()
        rec = _DeviceRecorder(BACKEND)
        with LMRunner(rec, params, spec, resident=True, device_state=True) as runner:
            runner.generate_beam(prompt, num_beams=2, max_new_tokens=3, chunk_size=2)
        self.assertIsNone(runner._swap_plan, "close 必须丢掉重排计划")
        self.assertIsNone(runner._decode_plan, "close 必须丢掉 decode 计划")


class _ResidentOnly:
    """有驻留接口、没有捕获能力——用来验证构造期的显式拒绝。"""

    def __init__(self):
        self.resident = {}

    def resident_parameters(self, graph, values):
        self.resident = dict(values)

    def release_parameters(self):
        self.resident = {}


class TestDeviceStateServing(unittest.TestCase):
    """连续批处理：每步只 decode 一次，状态只在准入时往返。"""

    def _requests(self, engine):
        prompts = [[3, 4, 5], [6, 7], [8, 9, 10, 11], [12]]
        return [
            engine.submit(prompt, max_new_tokens=3, seed=index)
            for index, prompt in enumerate(prompts)
        ]

    def test_continuous_batching_matches_host_and_captures_once(self):
        spec = _small(batch=1, steps=6)
        params = spec.init_params(seed=31)
        with ServingEngine(
            _DeviceRecorder(BACKEND), params, spec, capacity=2, resident=True
        ) as engine:
            self._requests(engine)
            want = engine.run()

        rec = _DeviceRecorder(BACKEND)
        with ServingEngine(
            rec, params, spec, capacity=2, resident=True, device_state=True
        ) as engine:
            self._requests(engine)
            got = engine.run()
            captures = rec.captures
            d2h = list(rec.d2h)
            replays = rec.replays

        self.assertEqual(got, want)
        self.assertEqual(captures, 1, "batch=capacity 的 decode 图只捕获一次")
        logits = (2, 1, spec.vocab_size)
        # 每个 decode 步只回传一次 logits 块
        self.assertEqual(sum(1 for shape, _ in d2h if shape == logits), replays)
        # 状态整块只在准入时往返：4 条请求、capacity=2，分两批准入
        states = len(spec.state_shapes)
        state_d2h = sum(1 for shape, _ in d2h if shape != logits)
        self.assertEqual(state_d2h, 2 * states)
        # 若 decode 每步都回传状态，这个数会是 replays * states
        self.assertLess(state_d2h, replays * states)

    def test_requires_resident(self):
        spec = _small(batch=1, steps=4)
        with self.assertRaisesRegex(ValueError, "resident=True"):
            ServingEngine(
                _DeviceRecorder(BACKEND), spec.init_params(seed=0), spec,
                resident=False, device_state=True,
            )

    def test_requires_capture_capability(self):
        spec = _small(batch=1, steps=4)
        with self.assertRaisesRegex(ValueError, "capture"):
            ServingEngine(
                _ResidentOnly(), spec.init_params(seed=0), spec,
                resident=True, device_state=True,
            )


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

try:  # CUDA 后端：能导入 cupy 且真的有一张可见的卡
    from aetherflow.backends.cuda import CudaBackend
    import cupy

    CUDA_AVAILABLE = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover - 没有 CUDA 的环境
    CudaBackend, CUDA_AVAILABLE = None, False


def _check_device_state_on_device(backend_cls):
    """同一套真机判据：逐 token 一致 + 每步只取回 logits。"""
    spec = _small(batch=1, steps=4, vocab_size=8)
    params = spec.init_params(seed=24)
    prompt = [3, 4, 5]
    limit = 3
    want = list(generate(
        backend_cls(), params, prompt, spec, max_new_tokens=limit, chunk_size=2
    ))

    backend = backend_cls()
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

    assert got == want, (got, want)
    logits = (1, 1, spec.vocab_size)
    # prefill 的尾块取回一次 logits，decode 每步一次
    assert sizes.count(logits) == limit, sizes
    # 状态形状只出现在 prefill 的两块里；decode 没有再取回它们
    state_like = [shape for shape in sizes if shape != logits]
    assert len(state_like) == 2 * len(spec.state_shapes), sizes


def _check_serving_device_state_on_device(backend_cls):
    """连续批处理：capacity 固定、准入写行，输出与宿主路径一致。"""
    spec = _small(batch=1, steps=4, vocab_size=8)
    params = spec.init_params(seed=26)
    prompts = [[3, 4], [5, 6, 7]]
    want = None
    with ServingEngine(
        backend_cls(), params, spec, capacity=2, resident=True
    ) as engine:
        for index, prompt in enumerate(prompts):
            engine.submit(prompt, max_new_tokens=2, seed=index)
        want = engine.run()
    with ServingEngine(
        backend_cls(), params, spec, capacity=2, resident=True, device_state=True
    ) as engine:
        for index, prompt in enumerate(prompts):
            engine.submit(prompt, max_new_tokens=2, seed=index)
        got = engine.run()
    assert got == want, (got, want)


def _check_beam_device_state_on_device(backend_cls):
    """束搜索：重排与 decode 都在设备上，逐 token 与宿主路径一致。"""
    spec = _small(batch=1, steps=4, vocab_size=8)
    params = spec.init_params(seed=28)
    prompt = [3, 4, 5]
    beams, limit = 2, 3
    want = list(generate_beam(
        backend_cls(), params, prompt, spec, num_beams=beams,
        max_new_tokens=limit, chunk_size=2,
    ))

    backend = backend_cls()
    sizes = []
    original = backend.to_host

    def counting(value):
        sizes.append(tuple(np.shape(value)))
        return original(value)

    backend.to_host = counting
    with LMRunner(backend, params, spec, resident=True, device_state=True) as runner:
        got = list(runner.generate_beam(
            prompt, num_beams=beams, max_new_tokens=limit, chunk_size=2,
        ))

    assert got == want, (got, want)
    block = (beams, 1, spec.vocab_size)
    # 预填充尾块一次 + 每次 decode 一次；束搜索不再逐行取 logits
    assert sizes.count(block) == limit, sizes
    state_shapes = {
        tuple(shape) for shape in spec.with_batch(beams).state_shapes.values()
    }
    # 状态只在 prefill 的两块里回传；decode 与重排都没有 D2H
    assert len([s for s in sizes if s in state_shapes]) == 2 * len(state_shapes), sizes


@unittest.skipUnless(HIP_AVAILABLE, "本机没有 HIP 设备")
class TestDeviceStateOnHip(unittest.TestCase):
    def test_runner_matches_host_and_only_logits_come_back(self):
        _check_device_state_on_device(HipBackend)

    def test_serving_matches_host(self):
        _check_serving_device_state_on_device(HipBackend)

    def test_beam_matches_host(self):
        _check_beam_device_state_on_device(HipBackend)


@unittest.skipUnless(CUDA_AVAILABLE, "本机没有 CUDA 设备")
class TestDeviceStateOnCuda(unittest.TestCase):
    def test_runner_matches_host_and_only_logits_come_back(self):
        _check_device_state_on_device(CudaBackend)

    def test_serving_matches_host(self):
        _check_serving_device_state_on_device(CudaBackend)

    def test_beam_matches_host(self):
        _check_beam_device_state_on_device(CudaBackend)


if __name__ == "__main__":
    unittest.main()
