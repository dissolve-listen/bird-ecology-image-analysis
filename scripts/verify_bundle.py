"""Verify the exact submitted dataset and trained models without the raw archive."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reproducibility import sha256_file, verify_inputs


def verify_environment() -> dict:
    if sys.version_info[:2] != (3, 13):
        raise ValueError(f"需要 Python 3.13，当前为 {sys.version.split()[0]}")
    versions = {}
    for line in (ROOT / "requirements.lock").read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, expected = line.strip().split("==")
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError(f"依赖版本不一致: {name}={actual}，应为 {expected}")
        versions[name] = actual
    return versions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args()
    versions = verify_environment()
    manifest = verify_inputs(ROOT)
    import pandas as pd
    from src.dataset import validate_feature_cache
    from src.regression import regression_data
    features = pd.read_csv(ROOT / "data/processed/image_features.csv")
    metadata = pd.read_csv(ROOT / "data/processed/cub_metadata.csv")
    validate_feature_cache(features, metadata)
    regression_data(features, pd.read_csv(ROOT / "data/processed/cub_avonet_traits.csv"))
    for name, expected in manifest["artifacts"].items():
        if not args.inputs_only or name.startswith("reference/"):
            if sha256_file(ROOT / name) != expected["sha256"]:
                raise ValueError(f"提交模型/参考结果的校验值不一致: {name}")
    if not args.inputs_only:
        from check_results import check_results
        check_results(ROOT, retrained=False)
    print(json.dumps({"passed": True, "images": len(features), "train": 5994,
                      "test": 5794, "classes": 200, "features": 68,
                      "locked_packages": len(versions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
