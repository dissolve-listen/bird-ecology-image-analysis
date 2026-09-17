from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PATHS, find_cub_archive
from src.dataset import load_cub_metadata, validate_cub_metadata
from src.ecology import build_crosswalk, download_avonet, load_avonet_table, select_traits
from src.reproducibility import sha256_file, verify_inputs


def download_cub(destination: Path, expected: dict) -> Path:
    import requests
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(destination) == expected["sha256"]:
        return destination
    partial = destination.with_suffix(".tgz.part")
    print(f"下载完整 CUB（约 1.15 GB）: {expected['url']}", flush=True)
    with requests.get(expected["url"], stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(8 * 1024 * 1024):
                handle.write(chunk)
    if sha256_file(partial) != expected["sha256"]:
        raise ValueError("CUB 下载文件 SHA-256 不匹配；拒绝解压，请重新下载。")
    partial.replace(destination)
    return destination


def safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        destination_resolved = destination.resolve()
        for member in tar.getmembers():
            member_path = (destination / member.name).resolve()
            if not member_path.is_relative_to(destination_resolved):
                raise RuntimeError("压缩包包含非法路径，已拒绝解压。")
        tar.extractall(destination, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser(description="全量准备 CUB-200-2011 与 AVONET 数据")
    parser.add_argument("--archive", help="CUB_200_2011.tgz 的路径")
    parser.add_argument("--download", action="store_true", help="本地无压缩包时从 CaltechDATA 下载并校验")
    parser.add_argument("--skip-avonet", action="store_true", help="仅准备 CUB；训练前仍需完成 AVONET 关联")
    parser.add_argument("--avonet-file", help="已下载 AVONET xlsx/csv 的路径")
    args = parser.parse_args()
    manifest = verify_inputs(ROOT)
    PATHS.make_output_dirs()
    if not PATHS.cub_root.is_dir():
        try:
            archive = find_cub_archive(args.archive)
        except FileNotFoundError:
            if args.archive or not args.download:
                raise
            archive = download_cub(ROOT / "data/downloads/CUB_200_2011.tgz", manifest["cub_archive"])
        if sha256_file(archive) != manifest["cub_archive"]["sha256"]:
            raise ValueError("CUB 压缩包 SHA-256 不匹配，拒绝使用不同版本或损坏的数据。")
        print(f"解压完整 CUB 数据集: {archive}")
        safe_extract(archive, PATHS.cub_root.parent)
    metadata = load_cub_metadata(PATHS.cub_root)
    summary = validate_cub_metadata(metadata, PATHS.cub_root)
    PATHS.metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    # The committed CSV contains relative paths; readers rebuild runtime paths.
    import pandas as pd
    expected_metadata = pd.read_csv(PATHS.metadata_csv)
    pd.testing.assert_frame_equal(metadata.drop(columns=["image_path"]), expected_metadata, check_dtype=False)
    for row in metadata.itertuples():
        if sha256_file(Path(row.image_path)) != manifest["cub_images"][str(row.image_id)]:
            raise ValueError(f"CUB 图像内容校验失败: {row.relative_path}")
    print(f"CUB 校验通过: {summary}")
    if args.skip_avonet:
        return
    avonet_path = Path(args.avonet_file) if args.avonet_file else ROOT / "data/source/AVONET_birdtree.csv"
    if sha256_file(avonet_path) != manifest["files"]["data/source/AVONET_birdtree.csv"]["sha256"]:
        raise ValueError("精确复现必须使用随包 AVONET BirdTree 快照，不能替换为其他版本。")
    avonet = load_avonet_table(avonet_path)
    with tempfile.TemporaryDirectory() as temporary:
        crosswalk = Path(temporary) / "cub_to_scientific.csv"
        shutil.copy2(PATHS.crosswalk_csv, crosswalk)
        joined = build_crosswalk(metadata, avonet, crosswalk)
    traits = select_traits(joined)
    if len(traits) != 200:
        raise RuntimeError(f"AVONET 关联数量错误: {len(traits)}，应为 200。")
    pd.testing.assert_frame_equal(traits, pd.read_csv(PATHS.traits_csv), check_dtype=False)
    print(f"AVONET 关联通过: {len(traits)} 个 CUB 鸟种")


if __name__ == "__main__":
    main()
