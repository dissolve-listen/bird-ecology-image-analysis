from __future__ import annotations

from collections import OrderedDict
import math

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from .processing import SegmentationResult, segment_bird


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 10 or np.std(a) < 1e-7 or np.std(b) < 1e-7:
        return 0.0
    value = np.corrcoef(a, b)[0, 1]
    return float(np.nan_to_num(value))


def _box_counting(binary: np.ndarray) -> float:
    pixels = binary > 0
    h, w = pixels.shape
    sizes = [size for size in (2, 4, 8, 16, 32, 64) if size <= min(h, w)]
    counts: list[int] = []
    valid_sizes: list[int] = []
    for size in sizes:
        padded = np.pad(pixels, ((0, (-h) % size), (0, (-w) % size)), mode="constant")
        blocks = padded.reshape(padded.shape[0] // size, size, padded.shape[1] // size, size)
        count = int(blocks.any(axis=(1, 3)).sum())
        if count > 0:
            valid_sizes.append(size)
            counts.append(count)
    if len(counts) < 2:
        return 0.0
    slope = np.polyfit(np.log(1 / np.array(valid_sizes)), np.log(np.array(counts)), 1)[0]
    return float(slope)


def _feature_map(image_bgr: np.ndarray, segmentation: SegmentationResult) -> OrderedDict[str, float]:
    mask = segmentation.mask
    h, w = mask.shape
    features: OrderedDict[str, float] = OrderedDict()
    hsv = cv2.cvtColor(segmentation.filtered, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(segmentation.filtered, cv2.COLOR_BGR2LAB)
    for channel_index, channel_name in enumerate(("h", "s", "v")):
        hist = cv2.calcHist([hsv], [channel_index], mask, [8], [0, 180 if channel_index == 0 else 256]).flatten()
        hist = hist / max(hist.sum(), 1.0)
        for index, value in enumerate(hist):
            features[f"color_hist_{channel_name}_{index}"] = float(value)
    foreground = mask > 0
    for channel_index, channel_name in enumerate(("l", "a", "b")):
        values = lab[:, :, channel_index][foreground]
        features[f"lab_mean_{channel_name}"] = float(values.mean()) if values.size else 0.0
        features[f"lab_std_{channel_name}"] = float(values.std()) if values.size else 0.0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None
    x, y, bw, bh = segmentation.bbox
    area = float(np.count_nonzero(mask))
    features["shape_area_ratio"] = area / (h * w)
    features["shape_aspect_ratio"] = bw / max(bh, 1)
    features["shape_extent"] = area / max(bw * bh, 1)
    if contour is not None:
        contour_area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        perimeter = cv2.arcLength(contour, True)
        features["shape_solidity"] = contour_area / max(hull_area, 1.0)
        features["shape_compactness"] = 4 * math.pi * contour_area / max(perimeter * perimeter, 1.0)
        moments = cv2.moments(contour)
        hu = cv2.HuMoments(moments).flatten()
    else:
        features["shape_solidity"] = 0.0
        features["shape_compactness"] = 0.0
        hu = np.zeros(7)
    for index, value in enumerate(hu):
        features[f"hu_{index + 1}"] = float(np.sign(value) * np.log10(abs(value) + 1e-12))

    gray = cv2.cvtColor(segmentation.filtered, cv2.COLOR_BGR2GRAY)
    crop_gray = gray[y : y + bh, x : x + bw]
    crop_mask = mask[y : y + bh, x : x + bw]
    if crop_gray.size < 16:
        crop_gray, crop_mask = gray, mask
    texture = cv2.resize(crop_gray, (96, 96), interpolation=cv2.INTER_AREA)
    texture_mask = cv2.resize(crop_mask, (96, 96), interpolation=cv2.INTER_NEAREST)
    quantized = (texture // 32).astype(np.uint8)
    glcm = graycomatrix(quantized, distances=[1, 3], angles=[0, np.pi / 4], levels=8, symmetric=True, normed=True)
    for prop in ("contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"):
        values = graycoprops(glcm, prop).ravel()
        features[f"glcm_{prop}_mean"] = float(np.nanmean(values))
        features[f"glcm_{prop}_std"] = float(np.nanstd(values))
    lbp = local_binary_pattern(texture, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp[texture_mask > 0], bins=np.arange(11), range=(0, 10), density=False)
    lbp_hist = lbp_hist / max(lbp_hist.sum(), 1)
    for index, value in enumerate(lbp_hist):
        features[f"lbp_{index}"] = float(value)
    edges = cv2.Canny(texture, 45, 130)
    features["fractal_dimension"] = _box_counting(cv2.bitwise_and(edges, edges, mask=texture_mask))

    half = texture[:, : texture.shape[1] // 2]
    other = cv2.flip(texture[:, texture.shape[1] - half.shape[1] :], 1)
    half_mask = texture_mask[:, : texture.shape[1] // 2] > 0
    other_mask = cv2.flip(texture_mask[:, texture.shape[1] - half.shape[1] :], 1) > 0
    paired = half_mask & other_mask
    features["symmetry_correlation"] = _safe_corr(half[paired].astype(float), other[paired].astype(float))
    features["symmetry_mask_iou"] = float((half_mask & other_mask).sum() / max((half_mask | other_mask).sum(), 1))
    features["segmentation_quality"] = segmentation.quality
    return features


def extract_features(image_bgr: np.ndarray) -> tuple[OrderedDict[str, float], SegmentationResult]:
    segmentation = segment_bird(image_bgr)
    return _feature_map(image_bgr, segmentation), segmentation


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    return {
        "color_shape": [name for name in columns if name.startswith(("color_", "lab_", "shape_", "hu_", "segmentation_"))],
        "texture": [name for name in columns if name.startswith(("glcm_", "lbp_"))],
        "complexity_symmetry": [name for name in columns if name.startswith(("fractal_", "symmetry_"))],
    }
