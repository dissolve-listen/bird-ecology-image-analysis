from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from .config import SEED
from .features import feature_groups


def make_classifier(n_estimators: int = 500) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        class_weight="balanced",
        max_leaf_nodes=256,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=SEED,
    )


def image_feature_columns(features: pd.DataFrame) -> list[str]:
    """Only image-derived predictors; IDs, labels, traits and evaluation IoU are excluded."""
    columns = [column for column in features.columns if column.startswith(("color_", "lab_", "shape_", "hu_", "glcm_", "lbp_", "fractal_", "symmetry_", "segmentation_"))]
    if not columns:
        raise ValueError("未找到图像特征列。")
    if not np.isfinite(features[columns].to_numpy(dtype=float)).all():
        raise ValueError("图像特征包含缺失值或无穷值，请重新提取特征。")
    return columns


def train_model(features: pd.DataFrame, n_estimators: int = 500) -> tuple[ExtraTreesClassifier, list[str]]:
    feature_columns = image_feature_columns(features)
    train = features.loc[features["split"] == "train"]
    model = make_classifier(n_estimators)
    model.fit(train[feature_columns], train["class_id"])
    return model, feature_columns


def save_model(model: ExtraTreesClassifier, feature_columns: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": feature_columns, "seed": SEED}, path, compress=3)


def load_model(path: Path) -> tuple[ExtraTreesClassifier, list[str]]:
    artifact = joblib.load(path)
    return artifact["model"], artifact["feature_columns"]


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ablation_columns(feature_columns: list[str]) -> dict[str, list[str]]:
    groups = feature_groups(feature_columns)
    return {
        "颜色与形状": groups["color_shape"],
        "增加纹理": groups["color_shape"] + groups["texture"],
        "完整特征": groups["color_shape"] + groups["texture"] + groups["complexity_symmetry"],
    }
