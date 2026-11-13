"""后端装配层的测试。

要钉住三件事：内置后端登记正确；选错名字、或选到装不起来的目标时**明确失败**；
自动探测在没有任何硬件时退回参考后端，并且**发出警告**——退回可以，静默不行。
"""

import os
import unittest
import warnings
from unittest.mock import patch

from aetherflow import (
    BackendUnavailableError,
    ReferenceBackend,
    get_backend,
    register_backend,
    registered_backends,
)


class _Recorder:
    """假后端：只记录构造参数，不执行任何图。"""

    name = "recorder"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def capabilities(self):
        return frozenset()

    def execute(self, graph, inputs):
        return {}


class TestRegistry(unittest.TestCase):
    def test_builtins_are_registered(self):
        names = registered_backends()
        for name in ("reference", "cuda", "hip"):
            self.assertIn(name, names)

    def test_unknown_name_lists_what_is_available(self):
        with self.assertRaises(LookupError) as ctx:
            get_backend("no_such_backend")
        self.assertIn("reference", str(ctx.exception))

    def test_duplicate_registration_is_refused(self):
        register_backend("_test_dup", _Recorder)
        with self.assertRaises(ValueError):
            register_backend("_test_dup", _Recorder)

    def test_replace_is_explicit(self):
        register_backend("_test_replace", _Recorder)
        register_backend("_test_replace", _Recorder, replace=True)
        self.assertIsInstance(get_backend("_test_replace"), _Recorder)

    def test_non_callable_factory_is_refused(self):
        with self.assertRaises(TypeError):
            register_backend("_test_bad", 42)

    def test_empty_name_is_refused(self):
        with self.assertRaises(ValueError):
            register_backend("  ", _Recorder)

    def test_factory_failure_is_wrapped_with_cause(self):
        def boom(**kwargs):
            raise OSError("没有驱动")

        register_backend("_test_boom", boom)
        with self.assertRaises(BackendUnavailableError) as ctx:
            get_backend("_test_boom")
        # 原始异常必须留在 __cause__ 里：装配失败的原因不该被一层包装吃掉
        self.assertIsInstance(ctx.exception.__cause__, OSError)
        self.assertIn("没有驱动", str(ctx.exception))


class TestSelection(unittest.TestCase):
    def test_kwargs_reach_the_factory(self):
        register_backend("_test_kwargs", _Recorder)
        backend = get_backend("_test_kwargs", block=512, conv_impl="tiled")
        self.assertEqual(backend.kwargs, {"block": 512, "conv_impl": "tiled"})

    def test_explicit_reference(self):
        self.assertIsInstance(get_backend("reference"), ReferenceBackend)

    def test_env_var_selects_backend(self):
        register_backend("_test_env", _Recorder)
        with patch.dict(os.environ, {"AETHERFLOW_BACKEND": "_test_env"}):
            self.assertIsInstance(get_backend(), _Recorder)

    def test_explicit_name_beats_env_var(self):
        register_backend("_test_explicit", _Recorder)
        with patch.dict(os.environ, {"AETHERFLOW_BACKEND": "reference"}):
            self.assertIsInstance(get_backend("_test_explicit"), _Recorder)

    def test_name_is_case_and_space_insensitive(self):
        register_backend("_test_case", _Recorder)
        self.assertIsInstance(get_backend("  _TEST_CASE  "), _Recorder)

    def test_explicit_reference_does_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            get_backend("reference")
        self.assertEqual(
            [w for w in caught if issubclass(w.category, RuntimeWarning)], []
        )

    def test_auto_returns_a_real_backend(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            backend = get_backend()
        self.assertIn(backend.name, ("cuda", "hip", "reference"))


class TestFallbackWithoutHardware(unittest.TestCase):
    """没有可用硬件的机器上，退回路径必须可见。"""

    def test_auto_falls_back_with_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            backend = get_backend()
        if backend.name != "reference":
            self.skipTest("这台机器有可用硬件，测不到退回路径")
        self.assertTrue(
            any(issubclass(w.category, RuntimeWarning) for w in caught),
            "退回参考后端必须发 RuntimeWarning",
        )

    def test_fallback_can_be_disabled(self):
        try:
            backend = get_backend(fallback=False)
        except BackendUnavailableError:
            return
        self.skipTest(f"这台机器有可用硬件（{backend.name}），测不到失败路径")


if __name__ == "__main__":
    unittest.main()
