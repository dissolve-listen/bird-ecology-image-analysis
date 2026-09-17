"""File provenance utilities shared by preparation and reproduction commands."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs(root: Path) -> dict:
    manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"复现输入缺失或校验失败: {name}；请恢复提交版本。")
    return manifest
