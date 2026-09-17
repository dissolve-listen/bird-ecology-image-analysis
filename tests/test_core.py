from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.features import extract_features
from src.processing import segment_bird


def test_segmentation_returns_image_sized_mask() -> None:
    image = np.full((120, 160, 3), 210, dtype=np.uint8)
    cv2.ellipse(image, (80, 60), (28, 45), 15, 0, 360, (30, 90, 180), -1)
    result = segment_bird(image)
    assert result.mask.shape == image.shape[:2]
    assert result.mask.dtype == np.uint8
    assert 0 <= result.quality <= 1


def test_features_are_deterministic() -> None:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.circle(image, (64, 64), 36, (30, 150, 220), -1)
    first, _ = extract_features(image)
    second, _ = extract_features(image)
    assert list(first) == list(second)
    assert np.allclose(list(first.values()), list(second.values()))
    assert "fractal_dimension" in first
    assert "symmetry_correlation" in first


if __name__ == "__main__":
    test_segmentation_returns_image_sized_mask()
    test_features_are_deterministic()
    print("核心图像处理单元测试通过")
