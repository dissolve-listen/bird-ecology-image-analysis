"""Check the committed Git snapshot before pushing it; needs only Python + Git."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*arguments: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments])


def main():
    if git("status", "--porcelain").strip():
        raise SystemExit("存在尚未提交的修改，请先提交，再检查将要上传的 Git 版本。")
    blobs = {}
    for record in git("ls-tree", "-r", "-l", "-z", "HEAD").decode("utf-8").split("\0"):
        if record:
            metadata, name = record.split("\t", 1)
            fields = metadata.split()
            if fields[1] == "blob":
                blobs[name] = int(fields[3])
    required = {"README.md", "requirements.lock", "data/manifest.json",
                "data/processed/cub_metadata.csv", "data/processed/cub_avonet_traits.csv",
                "data/processed/image_features.csv", "data/source/AVONET_birdtree.csv",
                "resources/cub_to_scientific.csv", "models/extra_trees_full.joblib",
                "models/extra_trees_traits.joblib", "scripts/reproduce.py",
                "reference/metrics.json", ".github/workflows/verify.yml"}
    missing = required - set(blobs)
    oversized = {name: size for name, size in blobs.items() if size >= 100 * 1024 * 1024}
    accidental = [name for name in blobs if name.startswith((".venv/", "data/raw/", "build/")) or name.endswith("CUB_200_2011.tgz")]
    if missing or oversized or accidental:
        raise SystemExit(json.dumps({"missing": sorted(missing), "oversized": oversized,
                                    "unwanted": accidental}, ensure_ascii=False))
    largest = max(blobs, key=blobs.get)
    print(json.dumps({"passed": True, "commit": git("rev-parse", "HEAD").decode().strip(),
                      "tracked_files": len(blobs), "total_bytes": sum(blobs.values()),
                      "largest_file": largest, "largest_bytes": blobs[largest]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
