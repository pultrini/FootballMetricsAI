"""Robust image-to-pitch perspective transformation."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from football_analytics.storage import PitchKeypoint

type Point = tuple[float, float]
type FloatArray = npt.NDArray[np.float64]


class HomographyError(RuntimeError):
    """Base error raised for unusable perspective transformations."""


class HomographyEstimationError(HomographyError):
    """Raised when correspondences cannot produce a valid matrix."""


class UnreliableHomographyError(HomographyError):
    """Raised when spatial output is requested from an unreliable estimate."""


class PitchHomographyConfig(BaseModel):
    """RANSAC and reliability thresholds for metric projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_correspondences: int = Field(default=4, ge=4)
    minimum_keypoint_confidence: float = Field(default=0.25, ge=0, le=1)
    ransac_reprojection_threshold_m: float = Field(default=1.5, gt=0)
    minimum_inlier_ratio: float = Field(default=0.6, ge=0, le=1)
    maximum_reprojection_rmse_m: float = Field(default=2.0, gt=0)
    minimum_confidence: float = Field(default=0.5, ge=0, le=1)
    ransac_max_iterations: int = Field(default=2_000, ge=1)
    ransac_confidence: float = Field(default=0.995, gt=0, lt=1)


class PitchHomography:
    """Stateful homography recalculated whenever new keypoints are available."""

    def __init__(self, config: PitchHomographyConfig | None = None) -> None:
        self.config = config or PitchHomographyConfig()
        self.clear()

    @property
    def matrix(self) -> FloatArray | None:
        """Return a defensive copy of the current 3x3 matrix."""
        return None if self._matrix is None else self._matrix.copy()

    @property
    def confidence(self) -> float:
        """Return the estimate quality in the interval [0, 1]."""
        return self._confidence

    @property
    def inlier_mask(self) -> npt.NDArray[np.bool_] | None:
        """Return which filtered correspondences were RANSAC inliers."""
        return None if self._inlier_mask is None else self._inlier_mask.copy()

    @property
    def reprojection_rmse(self) -> float | None:
        """Return inlier reprojection RMSE in pitch metres."""
        return self._reprojection_rmse

    def clear(self) -> None:
        """Discard the previous estimate so it cannot leak across camera cuts."""
        self._matrix: FloatArray | None = None
        self._confidence = 0.0
        self._inlier_mask: npt.NDArray[np.bool_] | None = None
        self._reprojection_rmse: float | None = None

    def estimate(
        self,
        image_points: Sequence[Point] | npt.NDArray[np.number],
        pitch_points: Sequence[Point] | npt.NDArray[np.number],
        confidences: Sequence[float] | npt.NDArray[np.number] | None = None,
    ) -> PitchHomography:
        """Estimate an image-to-pitch homography with RANSAC.

        A failed estimation clears any previous matrix, preventing stale spatial
        coordinates after cuts or poorly observed frames.
        """
        self.clear()
        image = self._point_array(image_points, "image_points")
        pitch = self._point_array(pitch_points, "pitch_points")
        if len(image) != len(pitch):
            msg = "image_points and pitch_points must have equal length"
            raise HomographyEstimationError(msg)

        scores = self._confidence_array(confidences, len(image))
        accepted = scores >= self.config.minimum_keypoint_confidence
        image = image[accepted]
        pitch = pitch[accepted]
        scores = scores[accepted]
        if len(image) < self.config.minimum_correspondences:
            msg = (
                f"At least {self.config.minimum_correspondences} confident "
                f"correspondences are required; received {len(image)}"
            )
            raise HomographyEstimationError(msg)
        self._validate_geometry(image, pitch)

        matrix, raw_mask = cv2.findHomography(
            image,
            pitch,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.ransac_reprojection_threshold_m,
            maxIters=self.config.ransac_max_iterations,
            confidence=self.config.ransac_confidence,
        )
        if matrix is None or raw_mask is None:
            raise HomographyEstimationError("OpenCV could not estimate a homography")
        matrix = np.asarray(matrix, dtype=np.float64)
        if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
            raise HomographyEstimationError("Estimated homography is degenerate")
        scale = matrix[2, 2]
        if abs(scale) <= np.finfo(np.float64).eps:
            raise HomographyEstimationError("Estimated homography has an invalid scale")
        matrix /= scale

        inliers = raw_mask.reshape(-1).astype(bool)
        if int(inliers.sum()) < self.config.minimum_correspondences:
            raise HomographyEstimationError("RANSAC found fewer than four inliers")
        projected = self._project(image, matrix)
        errors = np.linalg.norm(projected - pitch, axis=1)
        rmse = float(np.sqrt(np.mean(np.square(errors[inliers]))))
        inlier_ratio = float(np.mean(inliers))
        mean_keypoint_confidence = float(np.mean(scores[inliers]))
        error_quality = 1.0 / (1.0 + rmse / self.config.maximum_reprojection_rmse_m)

        self._matrix = matrix
        self._inlier_mask = inliers
        self._reprojection_rmse = rmse
        self._confidence = float(
            np.clip(inlier_ratio * mean_keypoint_confidence * error_quality, 0.0, 1.0)
        )
        return self

    def estimate_keypoints(self, keypoints: Sequence[PitchKeypoint]) -> PitchHomography:
        """Estimate directly from validated pitch-keypoint records."""
        if not keypoints:
            return self.estimate([], [], [])
        frame_ids = {keypoint.frame_id for keypoint in keypoints}
        if len(frame_ids) != 1:
            self.clear()
            raise HomographyEstimationError("Keypoints must belong to exactly one frame")
        return self.estimate(
            [(point.image_x, point.image_y) for point in keypoints],
            [(point.pitch_x, point.pitch_y) for point in keypoints],
            [point.confidence for point in keypoints],
        )

    def validate(self) -> bool:
        """Return whether the current estimate is safe for spatial statistics."""
        return bool(
            self._matrix is not None
            and self._inlier_mask is not None
            and int(self._inlier_mask.sum()) >= self.config.minimum_correspondences
            and float(np.mean(self._inlier_mask)) >= self.config.minimum_inlier_ratio
            and self._reprojection_rmse is not None
            and self._reprojection_rmse <= self.config.maximum_reprojection_rmse_m
            and self._confidence >= self.config.minimum_confidence
        )

    def transform_point(self, point: Point) -> Point:
        """Transform one pixel point into pitch metres when the estimate is reliable."""
        return self.transform_points([point])[0]

    def transform_points(self, points: Sequence[Point] | npt.NDArray[np.number]) -> list[Point]:
        """Transform pixel points into pitch metres when the estimate is reliable."""
        if not self.validate() or self._matrix is None:
            raise UnreliableHomographyError(
                "Homography is missing or below the configured reliability threshold"
            )
        values = self._point_array(points, "points")
        projected = self._project(values, self._matrix)
        return [(float(x), float(y)) for x, y in projected]

    @staticmethod
    def _point_array(points: Sequence[Point] | npt.NDArray[np.number], name: str) -> FloatArray:
        values = np.asarray(points, dtype=np.float64)
        if values.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
            raise HomographyEstimationError(f"{name} must be a finite Nx2 array")
        return values

    @staticmethod
    def _confidence_array(
        confidences: Sequence[float] | npt.NDArray[np.number] | None,
        expected_length: int,
    ) -> FloatArray:
        if confidences is None:
            return np.ones(expected_length, dtype=np.float64)
        values = np.asarray(confidences, dtype=np.float64)
        if values.shape != (expected_length,) or not np.isfinite(values).all():
            raise HomographyEstimationError(
                "confidences must be a finite vector matching the point count"
            )
        if np.any((values < 0) | (values > 1)):
            raise HomographyEstimationError("confidences must be between 0 and 1")
        return values

    @staticmethod
    def _validate_geometry(image: FloatArray, pitch: FloatArray) -> None:
        for values, name in ((image, "image_points"), (pitch, "pitch_points")):
            if len(np.unique(values, axis=0)) < 4:
                raise HomographyEstimationError(f"{name} must contain four unique points")
            homogeneous = np.column_stack((values, np.ones(len(values))))
            if np.linalg.matrix_rank(homogeneous) < 3:
                raise HomographyEstimationError(f"{name} points are collinear")

    @staticmethod
    def _project(points: FloatArray, matrix: FloatArray) -> FloatArray:
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float64)
        projected = cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        if not np.isfinite(projected).all():
            raise HomographyEstimationError("Transformation produced non-finite coordinates")
        return projected
