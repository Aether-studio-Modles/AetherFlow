"""zip 硬件包的测试。

钉住四件事：合法包"装完即可用"；不合法包在**安装期**就失败（缺 manifest、
路径穿越、API 不匹配、占用内置名）；一个坏包不影响其它后端；卸载后名字立刻
从可选中消失。
"""

import hashlib
import json
import os
import sys
import tempfile
import unittest
import uuid
import zipfile
from pathlib import Path
from unittest.mock import patch

from aetherflow import (
    BackendUnavailableError,
    HardwarePackageError,
    get_backend,
    install_hardware,
    installed_hardware,
    registered_backends,
    uninstall_hardware,
)
from aetherflow.backends import discovery, hardware

try:  # 签名是可选能力：没有 cryptography 就不跑这组用例
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    _HAVE_CRYPTO = True
except ImportError:  # pragma: no cover - 取决于环境
    _HAVE_CRYPTO = False

_BACKEND_CODE = '''
from aetherflow.backends.base import BackendCapabilityError


class FakeBackend:
    name = "{name}"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def capabilities(self):
        return frozenset()

    def execute(self, graph, inputs):
        raise BackendCapabilityError("假后端不执行图")
'''


def _write_zip(path: Path, entries: dict) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def _manifest(name: str, **overrides) -> dict:
    data = {
        "name": name,
        "version": "0.1.0",
        "aetherflow_api": "1",
        "factory": f"{name}.backend:FakeBackend",
    }
    data.update(overrides)
    return data


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _package(tmp: Path, name: str, **manifest_overrides) -> Path:
    """打一个合法包；默认带正确的 ``files`` 清单。

    ``files=None`` 表示**不声明**清单（模拟手工解压进目录的包），
    而不是写一个 ``"files": null``。
    """
    body = {
        f"{name}/__init__.py": "",
        f"{name}/backend.py": _BACKEND_CODE.format(name=name),
    }
    manifest = _manifest(name, **manifest_overrides)
    if "files" in manifest:
        if manifest["files"] is None:
            del manifest["files"]
    else:
        manifest["files"] = {
            entry: _digest(data) for entry, data in body.items()
        }
    return _write_zip(
        tmp / f"{name}.zip", {"manifest.json": json.dumps(manifest), **body}
    )


class _HardwareTest(unittest.TestCase):
    """每个用例一个独立的硬件目录，互不干扰，也不碰真实的 ~/.aetherflow。"""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.home = self.dir / "hardware"
        self.home.mkdir()
        patcher = patch.dict(
            os.environ, {"AETHERFLOW_HARDWARE_DIR": str(self.home)}
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._cleanup_modules)

    def _cleanup_modules(self):
        root = str(self.dir)
        for module in list(sys.modules.values()):
            origin = getattr(module, "__file__", "") or ""
            if origin.startswith(root):
                sys.modules.pop(module.__name__, None)
        sys.path[:] = [entry for entry in sys.path if not entry.startswith(root)]

    def _name(self) -> str:
        return "fake" + uuid.uuid4().hex[:8]

    def _install(self, name=None, **manifest_overrides):
        target = name or self._name()
        record = install_hardware(_package(self.dir, target, **manifest_overrides))
        self.addCleanup(uninstall_hardware, target)
        return record


class TestInstall(_HardwareTest):
    def test_installed_package_is_immediately_usable(self):
        record = self._install()
        backend = get_backend(record.name, block=256)
        self.assertEqual(backend.name, record.name)
        self.assertEqual(backend.kwargs, {"block": 256})

    def test_metadata_is_recorded(self):
        record = self._install(vendor="假工厂", description="测试用")
        self.assertEqual(record.version, "0.1.0")
        self.assertEqual(record.vendor, "假工厂")
        self.assertEqual(record.description, "测试用")
        self.assertIn(record.name, installed_hardware())

    def test_install_creates_a_real_directory(self):
        record = self._install()
        self.assertTrue((record.root / "manifest.json").is_file())
        self.assertTrue((record.root / record.name / "backend.py").is_file())
        self.assertEqual(record.root.parent, self.home)

    def test_duplicate_install_requires_replace(self):
        name = self._name()
        source = _package(self.dir, name)
        install_hardware(source)
        self.addCleanup(uninstall_hardware, name)
        with self.assertRaises(HardwarePackageError):
            install_hardware(source)
        # 覆盖是显式的；带上 replace 就能换版本
        record = install_hardware(_package(self.dir, name, version="0.2.0"), replace=True)
        self.assertEqual(record.version, "0.2.0")

    def test_builtin_name_is_refused(self):
        source = _package(self.dir, "cuda")
        with self.assertRaises(HardwarePackageError):
            install_hardware(source)
        self.assertFalse((self.home / "cuda").exists())


class TestManifestValidation(_HardwareTest):
    def test_missing_manifest_is_refused(self):
        source = _write_zip(self.dir / "nomani.zip", {"x.py": ""})
        with self.assertRaises(HardwarePackageError) as ctx:
            install_hardware(source)
        self.assertIn("manifest.json", str(ctx.exception))

    def test_broken_json_is_refused(self):
        source = _write_zip(
            self.dir / "badjson.zip", {"manifest.json": "{ not json"}
        )
        with self.assertRaises(HardwarePackageError):
            install_hardware(source)

    def test_name_must_be_a_lowercase_identifier(self):
        for bad in ("MyChip", "my-chip", "1chip", "", "chip name"):
            with self.subTest(name=bad):
                source = _write_zip(
                    self.dir / f"bad{uuid.uuid4().hex[:6]}.zip",
                    {"manifest.json": json.dumps(_manifest(bad))},
                )
                with self.assertRaises(HardwarePackageError):
                    install_hardware(source)

    def test_factory_must_be_module_colon_attr(self):
        name = self._name()
        source = _write_zip(
            self.dir / f"{name}.zip",
            {"manifest.json": json.dumps(_manifest(name, factory="missing_colon"))},
        )
        with self.assertRaises(HardwarePackageError):
            install_hardware(source)

    def test_api_major_mismatch_is_refused(self):
        source = _package(self.dir, self._name(), aetherflow_api="2")
        with self.assertRaises(HardwarePackageError) as ctx:
            install_hardware(source)
        self.assertIn("aetherflow_api", str(ctx.exception))

    def test_api_minor_is_accepted(self):
        record = self._install(aetherflow_api="1.7")
        self.assertEqual(record.api, "1.7")


class TestZipSafety(_HardwareTest):
    def test_path_traversal_is_refused(self):
        name = self._name()
        source = _write_zip(
            self.dir / "evil.zip",
            {
                "manifest.json": json.dumps(_manifest(name)),
                "../evil.py": "x = 1",
            },
        )
        with self.assertRaises(HardwarePackageError):
            install_hardware(source)
        self.assertFalse((self.home / "evil.py").exists())

    def test_absolute_path_is_refused(self):
        name = self._name()
        source = _write_zip(
            self.dir / "abs.zip",
            {
                "manifest.json": json.dumps(_manifest(name)),
                "/tmp/evil.py": "x = 1",
            },
        )
        with self.assertRaises(HardwarePackageError):
            install_hardware(source)

    def test_symlink_entry_is_refused(self):
        name = self._name()
        info = zipfile.ZipInfo("link")
        info.external_attr = 0o120777 << 16
        path = self.dir / "symlink.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("manifest.json", json.dumps(_manifest(name)))
            zf.writestr(info, "target")
        with self.assertRaises(HardwarePackageError):
            install_hardware(path)

    def test_entry_count_limit_is_enforced(self):
        with patch.object(hardware, "MAX_ENTRIES", 2):
            with self.assertRaises(HardwarePackageError):
                install_hardware(_package(self.dir, self._name()))

    def test_not_a_zip_is_refused(self):
        path = self.dir / "plain.zip"
        path.write_text("这不是 zip", encoding="utf-8")
        with self.assertRaises(HardwarePackageError):
            install_hardware(path)


class TestIntegrity(_HardwareTest):
    """``files`` 内容摘要：声明了就完全吻合，不声明照旧。"""

    def _body(self, name: str) -> dict:
        return {
            f"{name}/__init__.py": "",
            f"{name}/backend.py": _BACKEND_CODE.format(name=name),
        }

    def _zip(self, name: str, files: dict, body: dict) -> Path:
        manifest = _manifest(name, files=files)
        return _write_zip(
            self.dir / f"{name}.zip",
            {"manifest.json": json.dumps(manifest), **body},
        )

    def test_matching_digests_install(self):
        name = self._name()
        body = self._body(name)
        record = install_hardware(
            self._zip(name, {k: _digest(v) for k, v in body.items()}, body)
        )
        self.addCleanup(uninstall_hardware, name)
        self.assertEqual(record.name, name)

    def test_tampered_file_is_refused(self):
        name = self._name()
        body = self._body(name)
        files = {k: _digest(v) for k, v in body.items()}
        files[f"{name}/backend.py"] = "0" * 64
        with self.assertRaises(HardwarePackageError) as ctx:
            install_hardware(self._zip(name, files, body))
        self.assertIn("摘要不符", str(ctx.exception))
        # 安装期就失败，目录里不该留下半个包
        self.assertFalse((self.home / name).exists())

    def test_missing_declared_file_is_refused(self):
        name = self._name()
        body = self._body(name)
        files = {k: _digest(v) for k, v in body.items()}
        files[f"{name}/ghost.py"] = "0" * 64
        with self.assertRaises(HardwarePackageError) as ctx:
            install_hardware(self._zip(name, files, body))
        self.assertIn("没有这个文件", str(ctx.exception))

    def test_undeclared_extra_file_is_refused(self):
        """只校验声明过的子集，等于允许偷偷多放一个文件。"""
        name = self._name()
        body = self._body(name)
        body[f"{name}/extra.py"] = "x = 1"
        files = {k: _digest(v) for k, v in body.items() if not k.endswith("extra.py")}
        with self.assertRaises(HardwarePackageError) as ctx:
            install_hardware(self._zip(name, files, body))
        self.assertIn("摘要覆盖必须完整", str(ctx.exception))

    def test_digest_must_be_sha256_hex(self):
        name = self._name()
        body = self._body(name)
        for bad in ("", "abc", "z" * 64, 123):
            with self.subTest(bad=bad):
                files = {k: _digest(v) for k, v in body.items()}
                files[f"{name}/backend.py"] = bad
                with self.assertRaises(HardwarePackageError):
                    install_hardware(self._zip(name, files, body))

    def test_files_must_be_an_object(self):
        name = self._name()
        with self.assertRaises(HardwarePackageError):
            install_hardware(_package(self.dir, name, files=["x", "y"]))

    def test_package_without_files_is_still_accepted(self):
        """手工解压进目录的包没有清单；声明可选，不声明就不校验。"""
        record = self._install(files=None)
        self.assertNotIn("files", json.loads(
            (record.root / "manifest.json").read_text(encoding="utf-8")
        ))
        self.assertTrue(record.root.is_dir())

    def test_runtime_bytecode_is_ignored(self):
        """导入过就会生成 __pycache__，它不该让摘要变成不符。"""
        record = self._install()
        cache = record.root / record.name / "__pycache__"
        cache.mkdir()
        (cache / "backend.cpython-310.pyc").write_bytes(b"\x00")
        self.assertIn(record.name, installed_hardware())

    def test_tampered_installed_package_is_skipped_by_scan(self):
        """扫描期同样校验：目录里的包被改过，就不该再出现在可选中。"""
        record = self._install()
        (record.root / record.name / "backend.py").write_text("# 改过", encoding="utf-8")
        self.assertNotIn(record.name, installed_hardware())


class TestRequires(_HardwareTest):
    """``requires`` 依赖声明：算子得在框架算子表里，模块得能导入。"""

    def test_known_ops_are_accepted(self):
        record = self._install(requires={"ops": ["matmul", "relu"]})
        self.assertTrue(record.root.is_dir())

    def test_missing_op_is_refused(self):
        name = self._name()
        with self.assertRaises(HardwarePackageError) as ctx:
            install_hardware(_package(self.dir, name, requires={"ops": ["no_such_op"]}))
        self.assertIn("no_such_op", str(ctx.exception))
        self.assertFalse((self.home / name).exists())

    def test_missing_python_module_is_refused(self):
        name = self._name()
        with self.assertRaises(HardwarePackageError) as ctx:
            install_hardware(
                _package(self.dir, name, requires={"python": ["aetherflow_no_such_pkg"]})
            )
        self.assertIn("aetherflow_no_such_pkg", str(ctx.exception))

    def test_present_python_module_is_accepted(self):
        record = self._install(requires={"python": ["numpy"]})
        self.assertTrue(record.root.is_dir())

    def test_unknown_key_is_refused(self):
        """拼错的键会让整条声明静默失效，所以不认识的键直接拒绝。"""
        with self.assertRaises(HardwarePackageError) as ctx:
            install_hardware(_package(self.dir, self._name(), requires={"op": ["relu"]}))
        self.assertIn("requires", str(ctx.exception))

    def test_shape_must_be_string_arrays(self):
        for bad in ("relu", {"relu": 1}, 3):
            with self.subTest(bad=bad):
                with self.assertRaises(HardwarePackageError):
                    install_hardware(
                        _package(self.dir, self._name(), requires={"ops": bad})
                    )

    def test_requires_must_be_an_object(self):
        with self.assertRaises(HardwarePackageError):
            install_hardware(_package(self.dir, self._name(), requires=["relu"]))


class TestDiscovery(_HardwareTest):
    def test_scan_skips_broken_directories(self):
        (self.home / "broken").mkdir()
        (self.home / "broken" / "manifest.json").write_text("{ not json")
        (self.home / "empty").mkdir()
        record = self._install()
        found = installed_hardware()
        self.assertIn(record.name, found)
        self.assertNotIn("broken", found)
        self.assertNotIn("empty", found)

    def test_preinstalled_directory_is_discovered(self):
        """模拟"包已经在目录里"（随框架分发或别的进程装好），只靠扫描识别。"""
        name = self._name()
        dest = self.home / name
        dest.mkdir()
        with zipfile.ZipFile(_package(self.dir, name)) as zf:
            zf.extractall(dest)
        with patch.object(discovery, "_discovered", False):
            self.assertIn(name, registered_backends())
            self.assertEqual(get_backend(name).name, name)

    def test_missing_factory_target_is_reported_with_cause(self):
        record = self._install(factory="%s.nope:Missing" % self._name())
        with self.assertRaises(BackendUnavailableError) as ctx:
            get_backend(record.name)
        self.assertIsInstance(ctx.exception.__cause__, HardwarePackageError)

    def test_broken_package_does_not_break_others(self):
        record = self._install()
        (self.home / "broken").mkdir()
        (self.home / "broken" / "manifest.json").write_text("{ not json")
        self.assertEqual(get_backend(record.name).name, record.name)

    def test_scan_errors_explain_why_a_package_is_skipped(self):
        """坏包照旧跳过，但原因要能拿到：静默跳过等于"装没装"无从判断。"""
        record = self._install()
        (record.root / record.name / "backend.py").write_text("# 改过", encoding="utf-8")
        reasons = hardware.scan_errors()
        self.assertIn(record.name, reasons)
        self.assertIn("摘要不符", reasons[record.name])

    def test_tampered_install_is_explained_in_lookup_error(self):
        """覆盖已安装的 .py 后，报错要指明校验失败并给出重装方式。

        这正是换机器时"直接拷源码进包目录"的现象：包还在目录里，但 sha256
        对不上而被跳过；报错若只说"未注册的后端"，很难定位。
        """
        record = self._install()
        (record.root / record.name / "backend.py").write_text("# 改过", encoding="utf-8")
        discovery.unregister_backend(record.name)
        with self.assertRaises(LookupError) as ctx:
            get_backend(record.name)
        message = str(ctx.exception)
        self.assertIn("摘要不符", message)
        self.assertIn("install_hardware", message)


class TestUninstall(_HardwareTest):
    def test_uninstall_removes_directory_and_registration(self):
        record = self._install()
        self.assertTrue(uninstall_hardware(record.name))
        self.assertFalse(record.root.exists())
        self.assertNotIn(record.name, registered_backends())
        with self.assertRaises(LookupError):
            get_backend(record.name)

    def test_uninstall_missing_returns_false(self):
        self.assertFalse(uninstall_hardware("fake_not_installed"))

    def test_uninstall_rejects_bad_name(self):
        with self.assertRaises(HardwarePackageError):
            uninstall_hardware("../etc")

    def test_builtin_cannot_be_unregistered(self):
        with self.assertRaises(ValueError):
            discovery.unregister_backend("cuda")


def _sign(manifest: dict, key) -> dict:
    """对"除 signature 以外的整份清单"签名，与 hardware.py 的载荷定义一致。"""
    payload = json.dumps(
        {k: v for k, v in manifest.items() if k != "signature"},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    public = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    signed = dict(manifest)
    signed["signature"] = {
        "algorithm": "ed25519",
        "public_key": public.hex(),
        "value": key.sign(payload).hex(),
    }
    return signed


@unittest.skipUnless(_HAVE_CRYPTO, "需要 cryptography")
class TestSignature(_HardwareTest):
    """来源签名：信任锚在安装方，签名覆盖整份清单。"""

    def setUp(self):
        super().setUp()
        self.key = Ed25519PrivateKey.generate()
        self.public = self.key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def _signed(self, name: str, key=None, *, tamper=None) -> Path:
        body = {
            f"{name}/__init__.py": "",
            f"{name}/backend.py": _BACKEND_CODE.format(name=name),
        }
        manifest = _manifest(name)
        manifest["files"] = {k: _digest(v) for k, v in body.items()}
        manifest = _sign(manifest, key or self.key)
        if tamper:
            manifest.update(tamper)
        return _write_zip(
            self.dir / f"{name}.zip", {"manifest.json": json.dumps(manifest), **body}
        )

    def test_signed_package_installs_with_a_trusted_key(self):
        name = self._name()
        record = install_hardware(self._signed(name), trusted_keys=[self.public])
        self.addCleanup(uninstall_hardware, name)
        self.assertTrue(record.signed)
        self.assertEqual(record.name, name)

    def test_hex_public_key_is_accepted(self):
        name = self._name()
        record = install_hardware(self._signed(name), trusted_keys=[self.public.hex()])
        self.addCleanup(uninstall_hardware, name)
        self.assertTrue(record.signed)

    def test_untrusted_key_is_refused(self):
        other = Ed25519PrivateKey.generate().public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        name = self._name()
        with self.assertRaises(HardwarePackageError) as ctx:
            install_hardware(self._signed(name), trusted_keys=[other])
        self.assertIn("trusted_keys", str(ctx.exception))
        self.assertFalse((self.home / name).exists())

    def test_signature_without_trusted_keys_is_refused(self):
        """验不了就当没看见，与没签一样；静默跳过是安全反模式。"""
        name = self._name()
        with self.assertRaises(HardwarePackageError) as ctx:
            install_hardware(self._signed(name))
        self.assertIn("trusted_keys", str(ctx.exception))

    def test_trusted_keys_reject_an_unsigned_package(self):
        name = self._name()
        with self.assertRaises(HardwarePackageError) as ctx:
            install_hardware(_package(self.dir, name), trusted_keys=[self.public])
        self.assertIn("没有签名", str(ctx.exception))

    def test_unsigned_package_still_installs_without_trusted_keys(self):
        """向后兼容：不要求来源时，未签名的包照旧可装。"""
        record = self._install()
        self.assertFalse(record.signed)

    def test_manifest_tamper_breaks_the_signature(self):
        name = self._name()
        with self.assertRaises(HardwarePackageError) as ctx:
            install_hardware(
                self._signed(name, tamper={"version": "9.9.9"}),
                trusted_keys=[self.public],
            )
        self.assertIn("签名", str(ctx.exception))

    def test_file_tamper_is_still_refused(self):
        """文件内容改了但 files 没重算：摘要先拦下来。"""
        name = self._name()
        body = {
            f"{name}/__init__.py": "",
            f"{name}/backend.py": _BACKEND_CODE.format(name=name) + "\n# tampered\n",
        }
        manifest = _manifest(name)
        manifest["files"] = {
            k: _digest(_BACKEND_CODE.format(name=name) if k.endswith("backend.py") else "")
            for k in body
        }
        manifest = _sign(manifest, self.key)
        path = _write_zip(
            self.dir / f"{name}.zip", {"manifest.json": json.dumps(manifest), **body}
        )
        with self.assertRaises(HardwarePackageError) as ctx:
            install_hardware(path, trusted_keys=[self.public])
        self.assertIn("摘要不符", str(ctx.exception))

    def test_unknown_algorithm_is_refused(self):
        name = self._name()
        with self.assertRaises(HardwarePackageError) as ctx:
            install_hardware(
                self._signed(name, tamper={"signature": {
                    "algorithm": "rsa",
                    "public_key": self.public.hex(),
                    "value": "00" * 64,
                }}),
                trusted_keys=[self.public],
            )
        self.assertIn("算法", str(ctx.exception))

    def test_wrong_signature_bytes_are_refused(self):
        name = self._name()
        with self.assertRaises(HardwarePackageError) as ctx:
            install_hardware(
                self._signed(name, tamper={"signature": {
                    "algorithm": "ed25519",
                    "public_key": self.public.hex(),
                    "value": "00" * 64,
                }}),
                trusted_keys=[self.public],
            )
        self.assertIn("签名", str(ctx.exception))

    def test_signed_package_is_rediscovered_after_install(self):
        """重启后重新扫描必须还能认出签名包（F07）。

        安装时验过的来源要落到安装根目录；否则 ``iter_installed`` 没有
        ``trusted_keys``，带签名的包会被当成"无法确认来源"而消失。
        """
        name = self._name()
        install_hardware(self._signed(name), trusted_keys=[self.public])
        self.addCleanup(uninstall_hardware, name)

        # 模拟新进程：扫描是只读目录，不依赖安装时的任何内存状态
        records = installed_hardware()
        self.assertIn(name, records)
        self.assertTrue(records[name].signed)

    def test_tampering_after_install_still_hides_the_package(self):
        """信任记录只给锚，包本身仍要重新验签；改过内容就不能被发现。"""
        name = self._name()
        record = install_hardware(self._signed(name), trusted_keys=[self.public])
        self.addCleanup(uninstall_hardware, name)
        (record.root / name / "backend.py").write_text("# tampered\n", encoding="utf-8")
        self.assertNotIn(name, installed_hardware())

    def test_uninstall_clears_the_trust_record(self):
        name = self._name()
        install_hardware(self._signed(name), trusted_keys=[self.public])
        self.assertTrue((self.home / ".trust.json").is_file())

        self.assertTrue(uninstall_hardware(name))
        self.assertFalse((self.home / ".trust.json").exists())

    def test_unsigned_replace_clears_the_old_trust_record(self):
        """换成无签名版本后必须删掉旧锚，否则扫描会拿旧公钥要求签名。"""
        name = self._name()
        install_hardware(self._signed(name), trusted_keys=[self.public])
        self.addCleanup(uninstall_hardware, name)

        record = install_hardware(
            _package(self.dir, name, version="0.2.0"), replace=True
        )
        self.assertFalse(record.signed)
        records = installed_hardware()
        self.assertIn(name, records)
        self.assertFalse(records[name].signed)
