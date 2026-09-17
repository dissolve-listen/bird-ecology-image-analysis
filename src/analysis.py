from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import PATHS
from .features import extract_features
from .modeling import load_model
from .regression import TRAIT_NOTE, load_regressor


@dataclass
class AnalysisResult:
    source_path: str
    predicted_class_id: int
    predicted_species: str
    confidence: float
    top5: list[dict]
    segmentation_quality: float
    predicted_bbox: tuple[int, int, int, int]
    image_features: dict[str, float]
    ecological_traits: dict[str, object]
    warning: str | None
    regressed_traits: dict[str, float] = field(default_factory=dict)
    regression_note: str = TRAIT_NOTE

    def to_dict(self) -> dict:
        return asdict(self)

    def to_csv_row(self) -> dict[str, object]:
        """Serialize the stable analysis contract without discarding nested results."""
        return {
            "source_path": self.source_path,
            "predicted_class_id": self.predicted_class_id,
            "predicted_species": self.predicted_species,
            "confidence": self.confidence,
            "top5_json": json.dumps(self.top5, ensure_ascii=False),
            "segmentation_quality": self.segmentation_quality,
            "predicted_bbox_json": json.dumps(self.predicted_bbox),
            "image_features_json": json.dumps(self.image_features, ensure_ascii=False),
            "ecological_traits_json": json.dumps(self.ecological_traits, ensure_ascii=False),
            "regressed_traits_json": json.dumps(self.regressed_traits, ensure_ascii=False),
            **{f"regression_{name}": value for name, value in self.regressed_traits.items()},
            "regression_note": self.regression_note,
            "warning": self.warning or "",
        }


class BirdAnalyzer:
    def __init__(self, model_path: Path = PATHS.model_path, metadata_path: Path = PATHS.metadata_csv, traits_path: Path = PATHS.traits_csv, regression_model_path: Path | None = None):
        if not model_path.is_file():
            raise FileNotFoundError("未找到训练模型，请先执行 scripts/train_and_evaluate.py。")
        self.model, self.feature_columns = load_model(model_path)
        self.metadata = pd.read_csv(metadata_path)
        self.classes = self.metadata[["class_id", "display_name"]].drop_duplicates().set_index("class_id")["display_name"].to_dict()
        self.traits = pd.read_csv(traits_path).set_index("class_id") if traits_path.is_file() else pd.DataFrame()
        regression_model_path = regression_model_path or model_path.with_name("extra_trees_traits.joblib")
        self.regressor = None
        if regression_model_path.is_file():
            self.regressor, self.regression_columns, self.regression_targets = load_regressor(regression_model_path)

    def analyze_path(self, image_path: str | Path) -> tuple[AnalysisResult, np.ndarray, np.ndarray, np.ndarray]:
        path = Path(image_path)
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图片: {path}")
        feature_map, segmentation = extract_features(image)
        vector = pd.DataFrame([[feature_map.get(column, 0.0) for column in self.feature_columns]], columns=self.feature_columns)
        probabilities = self.model.predict_proba(vector)[0]
        order = np.argsort(probabilities)[::-1][:5]
        class_ids = self.model.classes_
        top5 = [{"class_id": int(class_ids[index]), "species": self.classes[int(class_ids[index])], "probability": float(probabilities[index])} for index in order]
        predicted = top5[0]
        traits: dict[str, object] = {}
        if not self.traits.empty and predicted["class_id"] in self.traits.index:
            traits = self.traits.loc[predicted["class_id"]].dropna().to_dict()
        warning = "预测置信度较低，输入图像可能不属于 CUB-200-2011 的 200 个鸟种。" if predicted["probability"] < 0.25 else None
        regressed_traits = {}
        regression_note = TRAIT_NOTE
        if self.regressor is not None:
            # Direct path uses only image features; classifier output and lookup
            # values cannot influence regression predictions.
            regression_vector = pd.DataFrame([{name: feature_map[name] for name in self.regression_columns}])
            values = self.regressor.predict(regression_vector)[0]
            regressed_traits = {name: float(value) for name, value in zip(self.regression_targets, values)}
        else:
            regression_note = "回归模型缺失，请运行 scripts/train_and_evaluate.py --regression-only。"
        result = AnalysisResult(
            source_path=str(path),
            predicted_class_id=predicted["class_id"],
            predicted_species=predicted["species"],
            confidence=float(predicted["probability"]),
            top5=top5,
            segmentation_quality=float(segmentation.quality),
            predicted_bbox=segmentation.bbox,
            image_features={name: round(float(value), 6) for name, value in feature_map.items()},
            ecological_traits=traits,
            warning=warning,
            regressed_traits=regressed_traits,
            regression_note=regression_note,
        )
        return result, image, segmentation.mask, segmentation.filtered
