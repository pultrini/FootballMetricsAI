"""Conservative shot detection from temporal ball and possession evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from football_analytics.analytics.geometry import shot_distance, visible_goal_angle
from football_analytics.events.detector import EventDetector, EventFrame
from football_analytics.pitch import DEFAULT_PITCH, GoalSide, Pitch
from football_analytics.storage import ShotEvent, TeamId


class ShotDetectorConfig(BaseModel):
    """Thresholds for observable shot-launch evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_ball_speed_mps: float = Field(default=12.0, gt=0)
    minimum_ball_acceleration_mps2: float = Field(default=8.0, ge=0)
    minimum_separation_m: float = Field(default=1.5, gt=0)
    minimum_goal_direction_cosine: float = Field(default=0.7, ge=-1, le=1)
    minimum_control_confidence: float = Field(default=0.05, ge=0, le=1)
    cooldown_frames: int = Field(default=10, ge=0)
    require_observed_ball: bool = True
    team_a_goal_side: GoalSide = GoalSide.RIGHT


@dataclass(frozen=True, slots=True)
class _ShotOrigin:
    player_id: int
    team_id: TeamId
    x: float
    y: float
    confidence: float


class ShotDetector(EventDetector[ShotEvent]):
    """Emit shots only when all configured launch signals agree."""

    def __init__(
        self,
        config: ShotDetectorConfig | None = None,
        *,
        pitch: Pitch = DEFAULT_PITCH,
    ) -> None:
        self.config = config or ShotDetectorConfig()
        self.pitch = pitch
        self.reset()

    def reset(self) -> None:
        """Clear armed player, ball history, and cooldown."""
        self._origin: _ShotOrigin | None = None
        self._last_ball_speed: float | None = None
        self._last_ball_timestamp: float | None = None
        self._last_frame_timestamp: float | None = None
        self._cooldown_until_frame = -1

    def update(self, frame: EventFrame) -> list[ShotEvent]:
        """Evaluate launch evidence and emit at most one shot."""
        if self._last_frame_timestamp is not None and frame.timestamp <= self._last_frame_timestamp:
            raise ValueError("Shot detector timestamps must increase")
        self._last_frame_timestamp = frame.timestamp
        ball = frame.ball
        possession = frame.possession
        player_id = possession.player_id
        team_id = None if possession.team_id is None else TeamId(possession.team_id)

        has_control_evidence = (
            frame.frame_id > self._cooldown_until_frame
            and player_id is not None
            and team_id in (TeamId.TEAM_A, TeamId.TEAM_B)
            and ball is not None
            and possession.confidence >= self.config.minimum_control_confidence
        )
        should_refresh_origin = has_control_evidence and (
            self._origin is None
            or self._origin.player_id != player_id
            or ball.speed is None
            or ball.speed < self.config.minimum_ball_speed_mps
        )
        if should_refresh_origin:
            assert player_id is not None
            assert team_id is not None
            assert ball is not None
            self._origin = _ShotOrigin(
                player_id=player_id,
                team_id=team_id,
                x=ball.x,
                y=ball.y,
                confidence=min(possession.confidence, ball.confidence),
            )

        event = self._detect_launch(frame)
        self._remember_ball(frame)
        if event is None:
            return []
        self._origin = None
        self._cooldown_until_frame = frame.frame_id + self.config.cooldown_frames
        return [event]

    def _detect_launch(self, frame: EventFrame) -> ShotEvent | None:
        ball = frame.ball
        origin = self._origin
        if ball is None or origin is None:
            return None
        if self.config.require_observed_ball and not ball.observed:
            return None
        if ball.speed is None or ball.velocity_x is None or ball.velocity_y is None:
            return None
        if self._last_ball_speed is None or self._last_ball_timestamp is None:
            return None
        delta_time = frame.timestamp - self._last_ball_timestamp
        if delta_time <= 0:
            raise ValueError("Shot detector timestamps must increase")
        acceleration = (ball.speed - self._last_ball_speed) / delta_time
        separation = math.hypot(ball.x - origin.x, ball.y - origin.y)
        goal_side = self._goal_side(origin.team_id)
        direction_cosine = self._goal_direction_cosine(
            (ball.x, ball.y),
            (ball.velocity_x, ball.velocity_y),
            goal_side,
        )
        if (
            ball.speed < self.config.minimum_ball_speed_mps
            or acceleration < self.config.minimum_ball_acceleration_mps2
            or separation < self.config.minimum_separation_m
            or direction_cosine < self.config.minimum_goal_direction_cosine
        ):
            return None

        confidence = min(
            origin.confidence,
            ball.confidence,
            min(1.0, ball.speed / (2 * self.config.minimum_ball_speed_mps)),
            self._acceleration_quality(acceleration),
            (direction_cosine + 1) / 2,
            min(1.0, separation / (2 * self.config.minimum_separation_m)),
        )
        return ShotEvent(
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
            team_id=origin.team_id,
            player_id=origin.player_id,
            x=origin.x,
            y=origin.y,
            distance_to_goal=shot_distance((origin.x, origin.y), goal_side, self.pitch),
            angle_to_goal=visible_goal_angle((origin.x, origin.y), goal_side, self.pitch),
            ball_speed=ball.speed,
            body_part=None,
            set_piece=None,
            pressure=None,
            outcome=None,
            confidence=confidence,
        )

    def _remember_ball(self, frame: EventFrame) -> None:
        if frame.ball is None or frame.ball.speed is None:
            return
        self._last_ball_speed = frame.ball.speed
        self._last_ball_timestamp = frame.timestamp

    def _goal_side(self, team_id: TeamId) -> GoalSide:
        if team_id is TeamId.TEAM_A:
            return self.config.team_a_goal_side
        return GoalSide.LEFT if self.config.team_a_goal_side is GoalSide.RIGHT else GoalSide.RIGHT

    def _acceleration_quality(self, acceleration: float) -> float:
        threshold = self.config.minimum_ball_acceleration_mps2
        if threshold == 0:
            return 1.0
        return min(1.0, acceleration / (2 * threshold))

    def _goal_direction_cosine(
        self,
        ball: tuple[float, float],
        velocity: tuple[float, float],
        goal_side: GoalSide,
    ) -> float:
        goal_x, goal_y = self.pitch.goal_center(goal_side)
        goal_vector = (goal_x - ball[0], goal_y - ball[1])
        goal_norm = math.hypot(*goal_vector)
        velocity_norm = math.hypot(*velocity)
        if goal_norm == 0 or velocity_norm == 0:
            return -1.0
        return (goal_vector[0] * velocity[0] + goal_vector[1] * velocity[1]) / (
            goal_norm * velocity_norm
        )
