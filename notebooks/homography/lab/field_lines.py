"""Field masking, line extraction and manual homography support."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LineSegment:
    """A detected line segment with its orientation and length."""

    x1: float
    y1: float
    x2: float
    y2: float
    angle: float
    length: float
    family: int = -1


def green_mask(
    image: np.ndarray,
    *,
    min_green: int = 35,
    green_margin: int = 12,
) -> np.ndarray:
    """Keep pixels whose green channel dominates red and blue."""
    b, g, r = cv2.split(image)
    mask = (g >= min_green) & (g >= b + green_margin) & (g >= r + green_margin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)


def white_line_mask(
    image: np.ndarray,
    field_mask: np.ndarray,
    *,
    min_value: int = 150,
    max_saturation: int = 100,
) -> np.ndarray:
    """Extract bright, low-saturation pixels inside the green-field region."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)
    mask = (value >= min_value) & (saturation <= max_saturation) & (field_mask > 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def detect_segments(
    line_mask: np.ndarray,
    *,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 35,
    min_line_length: int = 45,
    max_line_gap: int = 15,
) -> list[LineSegment]:
    """Detect line segments with Canny followed by probabilistic Hough."""
    edges = cv2.Canny(line_mask, canny_low, canny_high)
    raw = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if raw is None:
        return []
    segments: list[LineSegment] = []
    for values in raw.reshape(-1, 4):
        x1, y1, x2, y2 = map(float, values)
        angle = float(np.arctan2(y2 - y1, x2 - x1) % np.pi)
        length = float(np.hypot(x2 - x1, y2 - y1))
        segments.append(LineSegment(x1, y1, x2, y2, angle, length))
    return segments


def group_families(
    segments: Iterable[LineSegment],
    *,
    perpendicular_tolerance_deg: float = 25.0,
) -> tuple[list[LineSegment], list[LineSegment]]:
    """Split segments into two dominant orientation families."""
    items = list(segments)
    if len(items) < 2:
        return items, []
    angles = np.asarray([item.angle for item in items])
    reference = float(np.median(angles))
    distance = np.abs(((angles - reference + np.pi / 2) % np.pi) - np.pi / 2)
    first = distance <= np.deg2rad(perpendicular_tolerance_deg)
    if first.all() or (~first).all():
        pivot = int(np.argmax([item.length for item in items]))
        reference = items[pivot].angle
        distance = np.abs(((angles - reference + np.pi / 2) % np.pi) - np.pi / 2)
        first = distance <= np.deg2rad(35)
    family_a = [
        LineSegment(item.x1, item.y1, item.x2, item.y2, item.angle, item.length, 0)
        for item, keep in zip(items, first, strict=True)
        if keep
    ]
    family_b = [
        LineSegment(item.x1, item.y1, item.x2, item.y2, item.angle, item.length, 1)
        for item, keep in zip(items, first, strict=True)
        if not keep
    ]
    return family_a, family_b


def _line_coefficients(segment: LineSegment) -> tuple[float, float, float]:
    a = segment.y1 - segment.y2
    b = segment.x2 - segment.x1
    c = segment.x1 * segment.y2 - segment.x2 * segment.y1
    return a, b, c


def intersections(
    family_a: Iterable[LineSegment],
    family_b: Iterable[LineSegment],
    *,
    max_distance: float = 100.0,
) -> np.ndarray:
    """Return finite intersections near both detected segments."""
    points: list[tuple[float, float]] = []
    for first in family_a:
        a1, b1, c1 = _line_coefficients(first)
        for other in family_b:
            a2, b2, c2 = _line_coefficients(other)
            determinant = a1 * b2 - a2 * b1
            if abs(determinant) < 1e-8:
                continue
            x = (b1 * c2 - b2 * c1) / determinant
            y = (c1 * a2 - c2 * a1) / determinant
            distances = (
                np.hypot(x - first.x1, y - first.y1),
                np.hypot(x - first.x2, y - first.y2),
                np.hypot(x - other.x1, y - other.y1),
                np.hypot(x - other.x2, y - other.y2),
            )
            if (
                min(distances[0], distances[1]) <= max_distance
                and min(distances[2], distances[3]) <= max_distance
            ):
                points.append((x, y))
    if not points:
        return np.empty((0, 2), dtype=float)
    return np.unique(np.round(np.asarray(points), 1), axis=0)


def homography_from_matches(
    image_points: np.ndarray,
    field_points: np.ndarray,
    *,
    threshold_m: float = 1.5,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Estimate image-to-field H from manually associated points."""
    image = np.asarray(image_points, dtype=np.float32)
    field = np.asarray(field_points, dtype=np.float32)
    if image.shape != field.shape or image.ndim != 2 or image.shape[1] != 2:
        raise ValueError("image_points and field_points must both have shape (n, 2)")
    if len(image) < 4:
        raise ValueError("At least four point correspondences are required")
    return cv2.findHomography(
        image,
        field,
        cv2.RANSAC,
        threshold_m,
        maxIters=2_000,
        confidence=0.995,
    )
