from __future__ import annotations

from pathlib import Path

import pandas as pd


EXPECTED_IMAGES = 11788
EXPECTED_CLASSES = 200
EXPECTED_TRAIN = 5994
EXPECTED_TEST = 5794


def _read_key_value(path: Path, value_name: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", names=["image_id", value_name], engine="python")


def load_cub_metadata(cub_root: Path) -> pd.DataFrame:
    """Read the official CUB text annotations into one reliable table."""
    required = [
        "images.txt",
        "classes.txt",
        "image_class_labels.txt",
        "train_test_split.txt",
        "bounding_boxes.txt",
    ]
    missing = [name for name in required if not (cub_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"CUB 标注文件缺失: {', '.join(missing)}")

    images = pd.read_csv(cub_root / "images.txt", sep=r"\s+", names=["image_id", "relative_path"], engine="python")
    labels = _read_key_value(cub_root / "image_class_labels.txt", "class_id")
    split = _read_key_value(cub_root / "train_test_split.txt", "is_train")
    boxes = pd.read_csv(
        cub_root / "bounding_boxes.txt",
        sep=r"\s+",
        names=["image_id", "bbox_x", "bbox_y", "bbox_width", "bbox_height"],
        engine="python",
    )
    classes = pd.read_csv(cub_root / "classes.txt", sep=r"\s+", names=["class_id", "class_name"], engine="python")
    metadata = images.merge(labels, on="image_id").merge(split, on="image_id").merge(boxes, on="image_id").merge(classes, on="class_id")
    metadata["split"] = metadata["is_train"].map({1: "train", 0: "test"})
    metadata["image_path"] = metadata["relative_path"].map(lambda value: str(cub_root / "images" / value))
    metadata["display_name"] = metadata["class_name"].str.replace(r"^\d+\.", "", regex=True).str.replace("_", " ")
    return metadata.sort_values("image_id").reset_index(drop=True)


def validate_cub_metadata(metadata: pd.DataFrame, cub_root: Path) -> dict[str, int]:
    """Fail early if the data was reduced, damaged, or pointed at the wrong folder."""
    image_count = len(metadata)
    class_count = metadata["class_id"].nunique()
    train_count = int((metadata["split"] == "train").sum())
    test_count = int((metadata["split"] == "test").sum())
    checks = {
        "images": image_count == EXPECTED_IMAGES,
        "classes": class_count == EXPECTED_CLASSES,
        "train": train_count == EXPECTED_TRAIN,
        "test": test_count == EXPECTED_TEST,
        "image_files": sum(Path(path).is_file() for path in metadata["image_path"]) == EXPECTED_IMAGES,
    }
    if not all(checks.values()):
        raise ValueError(f"CUB 数据校验失败，拒绝继续运行: {checks}; 数据目录: {cub_root}")
    return {"images": image_count, "classes": class_count, "train": train_count, "test": test_count}


def read_portable_metadata(path: Path, cub_root: Path) -> pd.DataFrame:
    """Resolve image paths at runtime; never trust another machine's absolute paths."""
    metadata = pd.read_csv(path)
    metadata["image_path"] = metadata["relative_path"].map(
        lambda value: str(cub_root / "images" / Path(value)))
    return metadata


def validate_feature_cache(features: pd.DataFrame, metadata: pd.DataFrame) -> None:
    """Check every sample ID, label and official split, not just the row count."""
    from .modeling import image_feature_columns
    if len(features) != EXPECTED_IMAGES or features["image_id"].duplicated().any():
        raise ValueError("完整特征表应包含 11,788 个唯一图像 ID。")
    keys = ["image_id", "class_id", "split", "relative_path"]
    left = features[keys].sort_values("image_id").reset_index(drop=True)
    right = metadata[keys].sort_values("image_id").reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("特征缓存的图像 ID、标签、相对路径或官方划分与元数据不一致。")
    if len(image_feature_columns(features)) != 68:
        raise ValueError("完整特征缓存必须包含 68 维视觉特征。")
    if features["class_id"].nunique() != 200 or (features["split"] == "train").sum() != 5994 or (features["split"] == "test").sum() != 5794:
        raise ValueError("特征缓存不符合 CUB 官方 200 类、5994/5794 划分。")
