from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import BirdAnalyzer
from src.regression import (TRAIT_COLUMNS, evaluate_regression, load_regressor,
                            make_regressor, regression_data, save_regressor)


def sample_data():
    features = pd.DataFrame([
        {"image_id": species * 10 + i, "class_id": species, "split": "train" if i < 3 else "test",
         "relative_path": f"{species}/{i}.jpg", "color_signal": species + i / 10,
         "shape_signal": species * 2 + i, "bbox_iou": 1.0, "predicted_class_id": species}
        for species in range(1, 5) for i in range(5)])
    traits = pd.DataFrame({"class_id": range(1, 5),
        **{name: np.arange(1, 5) * (i + 1.0) for i, name in enumerate(TRAIT_COLUMNS)}})
    return features, traits


class RegressionTests(unittest.TestCase):
    def test_label_join_and_predictor_exclusions(self):
        f, t = sample_data()
        joined, columns = regression_data(f, t.iloc[::-1])
        self.assertEqual(columns, ["color_signal", "shape_signal"])
        np.testing.assert_array_equal(joined[TRAIT_COLUMNS[0]], joined["class_id"])
        for bad in [pd.concat([t, t.iloc[:1]]), t.iloc[:-1], t.assign(**{TRAIT_COLUMNS[0]: np.nan})]:
            with self.assertRaises(ValueError):
                regression_data(f, bad)
        with self.assertRaises(ValueError):
            regression_data(f, t.assign(科学名=["Taxon alpha", "Taxon alpha", "Taxon beta", "Taxon gamma"]))

    def test_holdout_never_changes_fitted_model_and_group_folds_are_disjoint(self):
        f, t = sample_data()
        with tempfile.TemporaryDirectory() as d, patch("src.regression.save_regression_figures"):
            root = Path(d)
            metrics = evaluate_regression(f, t, root / "model.joblib", root, n_estimators=5, group_folds=2, run_ablation=False)
            model, columns, targets = load_regressor(root / "model.joblib")
            data, _ = regression_data(f, t)
            training = data.loc[data["split"] == "train"]
            np.testing.assert_allclose(model.transformer_.mean_, training[targets].mean().to_numpy())
            for fold in metrics["grouped_validation"]["fold_details"]:
                self.assertFalse(set(fold["train_classes"]) & set(fold["validation_classes"]))
            heldout = pd.read_csv(root / "regression_test_predictions.csv")
            oof = pd.read_csv(root / "regression_group_predictions.csv")
            self.assertEqual(set(heldout.image_id), set(f.loc[f.split == "test", "image_id"]))
            self.assertEqual(set(oof.image_id), set(f.loc[f.split == "train", "image_id"]))
            self.assertTrue(oof.image_id.is_unique)
            f2 = f.copy()
            f2.loc[f2.split == "test", columns] = 100000.0
            evaluate_regression(f2, t, root / "model2.joblib", root / "second", n_estimators=5, group_folds=2, run_ablation=False)
            model2, _, _ = load_regressor(root / "model2.joblib")
            np.testing.assert_allclose(model.predict(f[columns]), model2.predict(f[columns]))
            # The baseline must come from train labels, not from test labels.
            np.testing.assert_allclose(heldout[f"baseline_{targets[0]}"], training[targets[0]].mean())

    def test_inference_is_independent_of_classifier_and_lookup_and_csv_keeps_units(self):
        f, t = sample_data()
        data, columns = regression_data(f, t)
        train = data[data.split == "train"]
        model = make_regressor(5).fit(train[columns], train[TRAIT_COLUMNS])
        classifier = SimpleNamespace(classes_=np.array([1, 2]), predict_proba=lambda x: np.array([[0.9, 0.1]]))
        features = {"color_signal": 2.1, "shape_signal": 4.0}
        segmentation = SimpleNamespace(quality=0.8, bbox=(0, 0, 8, 8), mask=np.ones((8, 8), np.uint8), filtered=np.zeros((8, 8, 3), np.uint8))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            classifier_path = root / "extra_trees_full.joblib"
            classifier_path.touch()
            save_regressor(model, columns, root / "extra_trees_traits.joblib")
            pd.DataFrame({"class_id": [1, 2], "display_name": ["bird one", "bird two"]}).to_csv(root / "metadata.csv", index=False)
            t.to_csv(root / "traits.csv", index=False)
            cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))[1].tofile(root / "bird.png")
            with patch("src.analysis.load_model", return_value=(classifier, columns)), patch("src.analysis.extract_features", return_value=(features, segmentation)):
                analyzer = BirdAnalyzer(classifier_path, root / "metadata.csv", root / "traits.csv")
                first, *_ = analyzer.analyze_path(root / "bird.png")
                classifier.predict_proba = lambda x: np.array([[0.1, 0.9]])
                analyzer.traits.loc[:, TRAIT_COLUMNS] = 99999.0
                second, *_ = analyzer.analyze_path(root / "bird.png")
                self.assertNotEqual(first.predicted_species, second.predicted_species)
                self.assertEqual(first.regressed_traits, second.regressed_traits)
                row = first.to_csv_row()
                self.assertEqual(json.loads(row["regressed_traits_json"]), first.regressed_traits)
                self.assertEqual(row["regression_体重_g"], first.regressed_traits["体重_g"])
                np.testing.assert_allclose(list(first.regressed_traits.values()), model.predict(pd.DataFrame([features]))[0])
                missing = BirdAnalyzer(classifier_path, root / "metadata.csv", root / "traits.csv", root / "missing.joblib")
                result, *_ = missing.analyze_path(root / "bird.png")
                self.assertEqual(result.regressed_traits, {})
                self.assertIn("回归模型缺失", result.regression_note)


if __name__ == "__main__":
    unittest.main()
