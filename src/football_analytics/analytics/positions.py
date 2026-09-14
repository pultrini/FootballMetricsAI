"""Projection of pixel observations into metric pitch trajectories."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from football_analytics.pitch import DEFAULT_PITCH, Pitch, PitchHomography
from football_analytics.storage import (
    BallPosition,
    Detection,
    PlayerPosition,
    TeamId,
    Track,
)


class PositionEstimatorConfig(BaseModel):
    """Filtering and temporal settings for metric trajectories."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    maximum_velocity_gap_seconds: float = Field(default=1.0, gt=0)
    player_classes: frozenset[str] = frozenset({"player"})
    goalkeeper_classes: frozenset[str] = frozenset({"goalkeeper"})
    referee_classes: frozenset[str] = frozenset({"referee"})
    ball_classes: frozenset[str] = frozenset({"ball"})


class MetricPositionEstimator:
    """Create trusted pitch positions and finite-difference player velocities."""

    def __init__(
        self,
        homography: PitchHomography,
        config: PositionEstimatorConfig | None = None,
        *,
        pitch: Pitch = DEFAULT_PITCH,
    ) -> None:
        self.homography = homography
        self.config = config or PositionEstimatorConfig()
        self.pitch = pitch
        self._previous: dict[int, tuple[float, float, float]] = {}

    def reset(self) -> None:
        """Clear velocity history before a new video or discontinuous segment."""
        self._previous.clear()

    def project_tracks(
        self,
        tracks: Sequence[Track],
        *,
        timestamp: float,
        team_assignments: Mapping[int, TeamId | str] | None = None,
    ) -> list[PlayerPosition]:
        """Project person-like tracks, omitting untrusted or off-pitch results."""
        if timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        if not self.homography.validate():
            return []
        frame_ids = {track.frame_id for track in tracks}
        if len(frame_ids) > 1:
            raise ValueError("All tracks must belong to exactly one frame")
        track_ids = [track.track_id for track in tracks]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("Each track_id may appear only once per frame")

        selected = [track for track in tracks if self._is_person_like(track.class_name)]
        if not selected:
            return []
        metric_points = self.homography.transform_points(
            [track.pixel_position for track in selected]
        )
        positions: list[PlayerPosition] = []
        for track, (x, y) in zip(selected, metric_points, strict=True):
            if not self.pitch.contains((x, y)):
                continue
            velocity_x, velocity_y, speed = self._velocity(track.track_id, timestamp, x, y)
            team_id = self._team_id(track, team_assignments or {})
            positions.append(
                PlayerPosition(
                    frame_id=track.frame_id,
                    timestamp=timestamp,
                    track_id=track.track_id,
                    team_id=team_id,
                    x=x,
                    y=y,
                    velocity_x=velocity_x,
                    velocity_y=velocity_y,
                    speed=speed,
                )
            )
            self._previous[track.track_id] = (timestamp, x, y)
        return positions

    def project_ball(
        self,
        detection: Detection,
        *,
        timestamp: float,
    ) -> BallPosition | None:
        """Project a ball-box centre and combine detection and geometry confidence."""
        if timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        if detection.class_name.casefold() not in self.config.ball_classes:
            msg = f"Detection class {detection.class_name!r} is not configured as ball"
            raise ValueError(msg)
        if not self.homography.validate():
            return None
        x, y = self.homography.transform_point(detection.center)
        if not self.pitch.contains((x, y)):
            return None
        return BallPosition(
            frame_id=detection.frame_id,
            timestamp=timestamp,
            x=x,
            y=y,
            confidence=detection.confidence * self.homography.confidence,
        )

    def project_balls(
        self,
        detections: Sequence[Detection],
        *,
        timestamp: float,
    ) -> list[BallPosition]:
        """Project every valid ball candidate for downstream temporal association."""
        positions: list[BallPosition] = []
        for detection in detections:
            if detection.class_name.casefold() not in self.config.ball_classes:
                continue
            position = self.project_ball(detection, timestamp=timestamp)
            if position is not None:
                positions.append(position)
        return positions

    def _velocity(
        self, track_id: int, timestamp: float, x: float, y: float
    ) -> tuple[float | None, float | None, float | None]:
        previous = self._previous.get(track_id)
        if previous is None:
            return (None, None, None)
        previous_time, previous_x, previous_y = previous
        delta_time = timestamp - previous_time
        if delta_time <= 0:
            raise ValueError("Timestamps must increase for each track")
        if delta_time > self.config.maximum_velocity_gap_seconds:
            return (None, None, None)
        velocity_x = (x - previous_x) / delta_time
        velocity_y = (y - previous_y) / delta_time
        return (velocity_x, velocity_y, math.hypot(velocity_x, velocity_y))

    def _is_person_like(self, class_name: str) -> bool:
        normalized = class_name.casefold()
        return normalized in (
            self.config.player_classes
            | self.config.goalkeeper_classes
            | self.config.referee_classes
        )

    def _team_id(self, track: Track, assignments: Mapping[int, TeamId | str]) -> TeamId | None:
        normalized = track.class_name.casefold()
        if normalized in self.config.goalkeeper_classes:
            return TeamId.GOALKEEPER
        if normalized in self.config.referee_classes:
            return TeamId.REFEREE
        assigned = assignments.get(track.track_id)
        return None if assigned is None else TeamId(assigned)
