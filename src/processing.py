from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SegmentationResult:
    filtered: np.ndarray
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    quality: float


def _border_pixels(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    width = max(2, min(width, h // 5, w // 5))
    return np.concatenate(
        [image[:width].reshape(-1, image.shape[2]), image[-width:].reshape(-1, image.shape[2]), image[:, :width].reshape(-1, image.shape[2]), image[:, -width:].reshape(-1, image.shape[2])]
    )


def _largest_plausible_component(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best_label, best_score = 0, -1.0
    area_total = float(h * w)
    for label in range(1, count):
        x, y, bw, bh, area = stats[label]
        area_ratio = area / area_total
        if area_ratio < 0.003 or area_ratio > 0.88:
            continue
        cx, cy = centroids[label]
        centre_distance = np.hypot((cx - w / 2) / w, (cy - h / 2) / h)
        border_penalty = int(x == 0) + int(y == 0) + int(x + bw >= w) + int(y + bh >= h)
        score = area_ratio * (1.0 - min(centre_distance, 0.7)) - 0.03 * border_penalty
        if score > best_score:
            best_label, best_score = label, score
    if best_label == 0:
        return mask
    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    points = cv2.findNonZero(mask)
    if points is None:
        h, w = mask.shape
        return 0, 0, w, h
    return tuple(int(value) for value in cv2.boundingRect(points))


def _quality(mask: np.ndarray) -> float:
    h, w = mask.shape
    area_ratio = float(np.count_nonzero(mask)) / (h * w)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    area = cv2.contourArea(max(contours, key=cv2.contourArea))
    perimeter = cv2.arcLength(max(contours, key=cv2.contourArea), True)
    compactness = (4 * np.pi * area / (perimeter * perimeter)) if perimeter else 0.0
    area_score = max(0.0, 1.0 - abs(area_ratio - 0.22) / 0.45)
    return float(np.clip(0.65 * area_score + 0.35 * min(compactness * 3, 1.0), 0, 1))


def segment_bird(image_bgr: np.ndarray) -> SegmentationResult:
    """Classical OpenCV foreground segmentation. It never consumes CUB boxes."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("无法读取图像")
    filtered = cv2.bilateralFilter(image_bgr, 7, 55, 55)
    filtered = cv2.GaussianBlur(filtered, (5, 5), 0)
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
    border = _border_pixels(lab, max(4, min(image_bgr.shape[:2]) // 20))
    background = np.median(border, axis=0).astype(np.float32)
    color_distance = np.linalg.norm(lab.astype(np.float32) - background, axis=2)
    distance_u8 = cv2.normalize(color_distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, distance_mask = cv2.threshold(distance_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    saturation = hsv[:, :, 1]
    _, saturation_mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY), 45, 130)
    candidates = cv2.bitwise_or(distance_mask, cv2.bitwise_and(saturation_mask, cv2.dilate(edges, np.ones((5, 5), np.uint8))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = _largest_plausible_component(cleaned)
    if np.count_nonzero(cleaned) < image_bgr.shape[0] * image_bgr.shape[1] * 0.002:
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        _, cleaned = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cleaned = _largest_plausible_component(cleaned)
    return SegmentationResult(filtered=filtered, mask=cleaned, bbox=_mask_bbox(cleaned), quality=_quality(cleaned))


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = image_bgr.copy()
    color[mask > 0] = (0.45 * color[mask > 0] + 0.55 * np.array([0, 190, 255])).astype(np.uint8)
    return color
