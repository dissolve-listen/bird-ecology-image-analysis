"""Compare rebuilt numerical results and reloaded model predictions to references."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.modeling import load_model, image_feature_columns
from src.regression import load_regressor, TRAIT_COLUMNS


def compare_json(actual, expected, key="metrics"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"指标字段不一致: {key}")
        for name in expected:
            compare_json(actual[name], expected[name], f"{key}.{name}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"指标条目数不一致: {key}")
        for i, value in enumerate(expected):
            compare_json(actual[i], value, f"{key}[{i}]")
    elif isinstance(expected, (int, float)):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-8, err_msg=key)
    elif actual != expected:
        raise ValueError(f"指标内容不一致: {key}")


def check_results(output: Path, retrained: bool = True) -> dict:
    reference = ROOT / "reference"
    actual_results = output / "results"
    expected_metrics = json.loads((reference / "metrics.json").read_text(encoding="utf-8"))
    actual_metrics = json.loads((actual_results / "metrics.json").read_text(encoding="utf-8"))
    compare_json(actual_metrics, expected_metrics)
    features = pd.read_csv(ROOT / "data/processed/image_features.csv")
    rebuilt_features = output / "data/processed/image_features.csv"
    full_features_checked = rebuilt_features.is_file() and output != ROOT
    if full_features_checked:
        reconstructed = pd.read_csv(rebuilt_features)
        pd.testing.assert_frame_equal(reconstructed, features, check_dtype=False, rtol=1e-9, atol=1e-8)
    test = features.loc[features["split"] == "test"]
    classifier, columns = load_model(output / "models/extra_trees_full.joblib")
    expected_classes = pd.read_csv(reference / "classification_predictions.csv")
    np.testing.assert_array_equal(test["image_id"], expected_classes["image_id"])
    np.testing.assert_array_equal(classifier.predict(test[columns]), expected_classes["predicted_class_id"])
    regressor, regression_columns, targets = load_regressor(output / "models/extra_trees_traits.joblib")
    expected_prediction = pd.read_csv(reference / "regression_test_predictions.csv")
    np.testing.assert_array_equal(test["image_id"], expected_prediction["image_id"])
    if columns != image_feature_columns(features) or regression_columns != columns or targets != TRAIT_COLUMNS:
        raise ValueError("模型输入列或回归目标顺序不符合提交规范")
    predicted_columns = [f"pred_{target}" for target in targets]
    np.testing.assert_allclose(regressor.predict(test[regression_columns]),
        expected_prediction[predicted_columns], rtol=1e-9, atol=1e-8)
    for name in ("regression_test_predictions.csv", "regression_group_predictions.csv",
                 "regression_metrics.csv", "regression_group_metrics.csv",
                 "regression_ablation_results.csv", "ablation_results.csv",
                 "regression_feature_trait_correlations.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(actual_results / name), pd.read_csv(reference / name),
                                      check_dtype=False, rtol=1e-9, atol=1e-8)
    summary = {"passed": True, "retrained": retrained,
               "all_raw_image_features_recomputed": full_features_checked,
               "test_predictions_checked": len(test), "oof_predictions_checked": 5994,
               "feature_count": 68, "regression_targets": len(targets),
               "classification_top1": actual_metrics["classification"]["top1_accuracy"],
               "regression_mean_r2": actual_metrics["regression"]["official_split"]["mean_r2"],
               "grouped_mean_r2": actual_metrics["regression"]["grouped_validation"]["mean_r2"],
               "relative_tolerance": 1e-9, "absolute_tolerance": 1e-8}
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(check_results(args.output_dir.resolve(), args.output_dir.resolve() != ROOT), indent=2))
