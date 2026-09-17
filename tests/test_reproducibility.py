from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.dataset import read_portable_metadata, validate_feature_cache
from src.reproducibility import verify_inputs
from src.config import find_cub_archive


class ReproducibilityTests(unittest.TestCase):
    def test_paths_follow_new_checkout_even_if_csv_contains_old_absolute_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = root / "metadata.csv"
            pd.DataFrame({"relative_path": ["001.Species/photo.jpg"],
                          "image_path": ["Z:/unavailable/old/photo.jpg"]}).to_csv(csv, index=False)
            resolved = read_portable_metadata(csv, root / "raw")
            self.assertEqual(resolved.iloc[0]["image_path"], str(root / "raw/images/001.Species/photo.jpg"))

    def test_same_size_corrupt_cache_cannot_pass(self):
        metadata = pd.read_csv(ROOT / "data/processed/cub_metadata.csv")
        features = pd.read_csv(ROOT / "data/processed/image_features.csv")
        validate_feature_cache(features, metadata)
        features.loc[0, "class_id"] = 200
        with self.assertRaises(ValueError):
            validate_feature_cache(features, metadata)

    def test_explicit_missing_archive_does_not_silently_fall_back(self):
        with self.assertRaises(FileNotFoundError):
            find_cub_archive(str(ROOT / "definitely-missing.tgz"))

    def test_supplied_inputs_are_complete_and_unchanged(self):
        manifest = verify_inputs(ROOT)
        self.assertEqual(len(manifest["cub_images"]), 11788)


if __name__ == "__main__":
    unittest.main()
