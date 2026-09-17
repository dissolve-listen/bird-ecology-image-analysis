from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.analysis import BirdAnalyzer
from src.config import PATHS
from src.dataset import read_portable_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="生成用于课堂展示的批量处理结果")
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    metadata = read_portable_metadata(PATHS.metadata_csv, PATHS.cub_root)
    samples = metadata.loc[metadata["split"] == "test"].groupby("class_id", group_keys=False).head(1).head(args.count)
    analyzer = BirdAnalyzer()
    demo_dir = PATHS.metrics_path.parent / "demo_batch_images"
    demo_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in samples.itertuples():
        source = Path(row.image_path)
        target = demo_dir / source.name
        if not target.exists():
            shutil.copy2(source, target)
        result, *_ = analyzer.analyze_path(target)
        rows.append(result.to_csv_row())
    pd.DataFrame(rows).to_csv(PATHS.metrics_path.parent / "demo_batch_results.csv", index=False, encoding="utf-8-sig")
    print(f"已生成 {len(rows)} 条课堂批处理演示结果。")


if __name__ == "__main__":
    main()
