"""Ultralytics pose-model adapter for pitch landmark detection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

from pydantic import BaseModel, ConfigDict, Field
from ultralytics import YOLO

from football_analytics.detection import Frame
from football_analytics.detection.conversion import as_numpy
from football_analytics.detection.detector import validate_frame
from football_analytics.pitch.keypoint_detector import PitchKeypointDetector
from football_analytics.pitch.keypoints import DEFAULT_PITCH_LANDMARKS, PitchLandmark
from football_analytics.storage import PitchKeypoint


class YoloPitchKeypointConfig(BaseModel):
    """Inference and landmark-selection settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    detection_confidence_threshold: float = Field(default=0.25, ge=0, le=1)
    keypoint_confidence_threshold: float = Field(default=0.35, ge=0, le=1)
    instance_index: int = Field(default=0, ge=0)
    device: str | int | None = None


class PosePredictionModel(Protocol):
    """Small interface implemented by Ultralytics pose models and test doubles."""

    def predict(self, source: Frame, **kwargs: object) -> Sequence[object]:
        """Return pose results for one frame."""


class YoloPitchKeypointDetector(PitchKeypointDetector):
    """Map custom YOLO pose keypoint indices to canonical pitch landmarks."""

    def __init__(
        self,
        model_path: str,
        config: YoloPitchKeypointConfig | None = None,
        *,
        landmarks: Sequence[PitchLandmark] = DEFAULT_PITCH_LANDMARKS,
        model: PosePredictionModel | None = None,
    ) -> None:
        if not landmarks:
            msg = "At least one canonical landmark is required"
            raise ValueError(msg)
        self.config = config or YoloPitchKeypointConfig()
        self.landmarks = tuple(landmarks)
        self._model = model or cast("PosePredictionModel", YOLO(model_path))

    def detect(self, frame: Frame, *, frame_id: int = 0) -> list[PitchKeypoint]:
        """Return confident landmark correspondences from a custom pose model."""
        validate_frame(frame)
        if frame_id < 0:
            msg = "frame_id must be non-negative"
            raise ValueError(msg)

        options: dict[str, object] = {
            "conf": self.config.detection_confidence_threshold,
            "verbose": False,
        }
        if self.config.device is not None:
            options["device"] = self.config.device
        results = self._model.predict(source=frame, **options)
        if len(results) != 1:
            msg = f"Expected one result for one frame, received {len(results)}"
            raise RuntimeError(msg)

        keypoints = getattr(results[0], "keypoints", None)
        if keypoints is None:
            return []
        if keypoints.conf is None:
            msg = "Pitch keypoint model must provide per-keypoint confidence"
            raise RuntimeError(msg)

        coordinates = as_numpy(keypoints.xy)
        confidences = as_numpy(keypoints.conf)
        instance_index = self.config.instance_index
        if instance_index >= len(coordinates):
            return []
        instance_coordinates = coordinates[instance_index]
        instance_confidences = confidences[instance_index]
        if len(instance_coordinates) != len(self.landmarks):
            msg = (
                f"Model returned {len(instance_coordinates)} keypoints, "
                f"but {len(self.landmarks)} landmarks were configured"
            )
            raise RuntimeError(msg)

        observations: list[PitchKeypoint] = []
        for landmark, image_point, confidence in zip(
            self.landmarks, instance_coordinates, instance_confidences, strict=True
        ):
            score = float(confidence)
            if score < self.config.keypoint_confidence_threshold:
                continue
            observations.append(
                PitchKeypoint(
                    frame_id=frame_id,
                    id=landmark.id,
                    name=landmark.name,
                    image_x=float(image_point[0]),
                    image_y=float(image_point[1]),
                    pitch_x=landmark.pitch_x,
                    pitch_y=landmark.pitch_y,
                    confidence=score,
                )
            )
        return observations
