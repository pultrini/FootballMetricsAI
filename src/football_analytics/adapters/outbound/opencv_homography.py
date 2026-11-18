"""OpenCV implementation of the homography backend port."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from football_analytics.domain.pitch.backend import FloatArray


class OpenCvHomographyBackend:
    """Delegate perspective operations to OpenCV without exposing it to domain code."""

    def find_homography(
        self,
        image_points: FloatArray,
        pitch_points: FloatArray,
        *,
        reprojection_threshold: float,
        max_iterations: int,
        confidence: float,
    ) -> tuple[FloatArray | None, npt.NDArray[np.uint8] | None]:
        matrix, mask = cv2.findHomography(
            image_points,
            pitch_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=reprojection_threshold,
            maxIters=max_iterations,
            confidence=confidence,
        )
        return (
            None if matrix is None else np.asarray(matrix, dtype=np.float64),
            None if mask is None else np.asarray(mask, dtype=np.uint8),
        )

    def perspective_transform(self, points: FloatArray, matrix: FloatArray) -> FloatArray:
        return cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
