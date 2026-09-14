"""Dedicated metric-space tracking for the football."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from football_analytics.analytics.geometry import euclidean_distance
from football_analytics.pitch.pitch_model import DEFAULT_PITCH, Pitch
from football_analytics.storage import BallPosition


class BallTrackerConfig(BaseModel):
    """Association, smoothing, and short-occlusion settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    association_distance_m: float = Field(default=8.0, gt=0)
    smoothing_alpha: float = Field(default=0.65, gt=0, le=1)
    maximum_missed_frames: int = Field(default=5, ge=0)
    prediction_confidence_decay: float = Field(default=0.75, gt=0, lt=1)
    minimum_prediction_confidence: float = Field(default=0.1, ge=0, le=1)
    reacquisition_confidence_threshold: float = Field(default=0.8, ge=0, le=1)


class BallTracker:
    """Associate projected ball candidates and bridge brief detection gaps."""

    def __init__(
        self,
        config: BallTrackerConfig | None = None,
        *,
        pitch: Pitch = DEFAULT_PITCH,
    ) -> None:
        self.config = config or BallTrackerConfig()
        self.pitch = pitch
        self.reset()

    def reset(self) -> None:
        """Clear temporal state before processing another video."""
        self._last: BallPosition | None = None
        self._missed_frames = 0

    def update(
        self,
        candidates: Sequence[BallPosition],
        *,
        frame_id: int,
        timestamp: float,
    ) -> BallPosition | None:
        """Return one smoothed observation or a marked short-term prediction."""
        if frame_id < 0 or timestamp < 0:
            raise ValueError("frame_id and timestamp must be non-negative")
        if any(candidate.frame_id != frame_id for candidate in candidates):
            raise ValueError("All ball candidates must belong to the supplied frame_id")
        if any(candidate.timestamp != timestamp for candidate in candidates):
            raise ValueError("All ball candidates must use the supplied timestamp")
        if self._last is not None and timestamp <= self._last.timestamp:
            raise ValueError("Ball timestamps must increase")

        if self._last is None:
            return self._initialize(candidates)

        predicted = self._predicted_point(timestamp)
        associated = [
            candidate
            for candidate in candidates
            if euclidean_distance((candidate.x, candidate.y), predicted)
            <= self.config.association_distance_m
        ]
        if associated:
            measurement = min(
                associated,
                key=lambda candidate: (
                    euclidean_distance((candidate.x, candidate.y), predicted),
                    -candidate.confidence,
                ),
            )
            return self._accept_measurement(measurement, timestamp)

        if candidates:
            strongest = max(candidates, key=lambda candidate: candidate.confidence)
            if strongest.confidence >= self.config.reacquisition_confidence_threshold:
                self._last = BallPosition(
                    **strongest.model_dump(exclude={"velocity_x", "velocity_y", "speed"}),
                    velocity_x=None,
                    velocity_y=None,
                    speed=None,
                )
                self._missed_frames = 0
                return self._last
        return self._predict(frame_id, timestamp, predicted)

    def _initialize(self, candidates: Sequence[BallPosition]) -> BallPosition | None:
        if not candidates:
            return None
        strongest = max(candidates, key=lambda candidate: candidate.confidence)
        self._last = BallPosition(
            **strongest.model_dump(exclude={"velocity_x", "velocity_y", "speed"}),
            velocity_x=None,
            velocity_y=None,
            speed=None,
        )
        self._missed_frames = 0
        return self._last

    def _predicted_point(self, timestamp: float) -> tuple[float, float]:
        assert self._last is not None
        delta_time = timestamp - self._last.timestamp
        velocity_x = self._last.velocity_x or 0.0
        velocity_y = self._last.velocity_y or 0.0
        return (
            self._last.x + velocity_x * delta_time,
            self._last.y + velocity_y * delta_time,
        )

    def _accept_measurement(self, measurement: BallPosition, timestamp: float) -> BallPosition:
        assert self._last is not None
        alpha = self.config.smoothing_alpha
        predicted_x, predicted_y = self._predicted_point(timestamp)
        x = alpha * measurement.x + (1 - alpha) * predicted_x
        y = alpha * measurement.y + (1 - alpha) * predicted_y
        delta_time = timestamp - self._last.timestamp
        measured_velocity_x = (x - self._last.x) / delta_time
        measured_velocity_y = (y - self._last.y) / delta_time
        previous_velocity_x = self._last.velocity_x
        previous_velocity_y = self._last.velocity_y
        velocity_x = (
            measured_velocity_x
            if previous_velocity_x is None
            else alpha * measured_velocity_x + (1 - alpha) * previous_velocity_x
        )
        velocity_y = (
            measured_velocity_y
            if previous_velocity_y is None
            else alpha * measured_velocity_y + (1 - alpha) * previous_velocity_y
        )
        self._last = BallPosition(
            frame_id=measurement.frame_id,
            timestamp=timestamp,
            x=x,
            y=y,
            confidence=measurement.confidence,
            observed=True,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            speed=math.hypot(velocity_x, velocity_y),
        )
        self._missed_frames = 0
        return self._last

    def _predict(
        self,
        frame_id: int,
        timestamp: float,
        predicted: tuple[float, float],
    ) -> BallPosition | None:
        assert self._last is not None
        self._missed_frames += 1
        confidence = self._last.confidence * self.config.prediction_confidence_decay
        if (
            self._missed_frames > self.config.maximum_missed_frames
            or confidence < self.config.minimum_prediction_confidence
            or not self.pitch.contains(predicted)
        ):
            self.reset()
            return None
        velocity_x = self._last.velocity_x
        velocity_y = self._last.velocity_y
        self._last = BallPosition(
            frame_id=frame_id,
            timestamp=timestamp,
            x=predicted[0],
            y=predicted[1],
            confidence=confidence,
            observed=False,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            speed=(
                None
                if velocity_x is None or velocity_y is None
                else math.hypot(velocity_x, velocity_y)
            ),
        )
        return self._last
