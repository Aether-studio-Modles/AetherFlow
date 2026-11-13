"""把示例硬件包打成 zip。

用法（仓库根目录）::

    python examples/hardware/build.py            # 产物在 examples/hardware/
    python examples/hardware/build.py out.zip    # 指定输出路径

产物可以直接交给 ``aetherflow.install_hardware``。
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "novac"
MANIFEST = "manifest.json"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(dest) -> Path:
    """按固定清单打包。

    先把文件清单算完再写 zip：边遍历边写会把刚生成的 zip 自己打进去，
    上一轮就是这么把包膨胀到几个 GB 的。

    ``manifest.json`` 额外写入 ``files``（除它自己以外的每个文件的 sha256）：
    安装时据此拒绝被改动过的包。摘要必须现算——源目录里改一行代码，
    写死的清单就过期了。
    """
    manifest = json.loads((SOURCE / MANIFEST).read_text(encoding="utf-8"))
    dest = Path(dest)
    entries = {
        path.relative_to(SOURCE).as_posix(): path
        for path in sorted(SOURCE.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path.name != MANIFEST
        and path.resolve() != dest.resolve()
    }
    manifest["files"] = {name: _digest(path) for name, path in entries.items()}
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2))
        for name, path in entries.items():
            zf.write(path, name)
    return dest


def main(argv) -> int:
    manifest = json.loads((SOURCE / MANIFEST).read_text(encoding="utf-8"))
    default = HERE / f"{manifest['name']}-{manifest['version']}.zip"
    dest = Path(argv[1]) if len(argv) > 1 else default
    path = build(dest)
    print(f"已生成 {path}（{path.stat().st_size} 字节）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
