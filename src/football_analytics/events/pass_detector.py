"""Pass detection from control, ball travel, and reception sequences."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from football_analytics.analytics.geometry import euclidean_distance
from football_analytics.events.detector import EventDetector, EventFrame
from football_analytics.storage import PassEvent, TeamId


class PassDetectorConfig(BaseModel):
    """Spatial and temporal evidence required for a pass sequence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_distance_m: float = Field(default=1.0, ge=0)
    minimum_duration_seconds: float = Field(default=0.04, ge=0)
    maximum_duration_seconds: float = Field(default=6.0, gt=0)
    minimum_control_confidence: float = Field(default=0.05, ge=0, le=1)
    minimum_departure_distance_m: float = Field(default=1.0, ge=0)
    minimum_flight_observations: int = Field(default=1, ge=1)


@dataclass(slots=True)
class _PassOrigin:
    passer_id: int
    team_id: TeamId
    frame_id: int
    timestamp: float
    x: float
    y: float
    confidence: float
    departed: bool = False
    flight_observations: int = 0


class PassDetector(EventDetector[PassEvent]):
    """Infer passes only after control, departure, travel, and reception."""

    def __init__(self, config: PassDetectorConfig | None = None) -> None:
        self.config = config or PassDetectorConfig()
        self.reset()

    def reset(self) -> None:
        """Clear the current possession and pass candidate."""
        self._last_player_id: int | None = None
        self._last_team_id: TeamId | None = None
        self._last_controlled_ball: tuple[int, int, float, float, float, float] | None = None
        self._pending: _PassOrigin | None = None
        self._last_timestamp: float | None = None

    def update(self, frame: EventFrame) -> list[PassEvent]:
        """Advance the pass-sequence state and emit at most one event."""
        if self._last_timestamp is not None and frame.timestamp <= self._last_timestamp:
            raise ValueError("Pass detector timestamps must increase")
        self._last_timestamp = frame.timestamp
        events: list[PassEvent] = []
        possession = frame.possession
        player_id = possession.player_id
        team_id = None if possession.team_id is None else TeamId(possession.team_id)

        if self._pending is not None:
            self._observe_ball_travel(frame)
            timed_out = (
                frame.timestamp - self._pending.timestamp > self.config.maximum_duration_seconds
            )
            if timed_out:
                event = (
                    self._finish_pending(frame, receiver_id=None, completed=False)
                    if self._has_flight_evidence()
                    else None
                )
                self._pending = None
                if event is not None:
                    events.append(event)

        transitioned = player_id != self._last_player_id
        if transitioned and self._last_player_id is not None and self._pending is None:
            self._pending = self._origin_from_previous()
            self._observe_ball_travel(frame)

        if self._pending is not None and player_id is not None:
            if player_id == self._pending.passer_id:
                self._pending = None
            elif team_id in (TeamId.TEAM_A, TeamId.TEAM_B):
                if self._has_flight_evidence():
                    completed = team_id is self._pending.team_id
                    receiver_id = player_id if completed else None
                    event = self._finish_pending(frame, receiver_id, completed)
                    if event is not None:
                        events.append(event)
                # A different controller without ball travel is an identity/proximity
                # change, not a delayed pass waiting to be counted later.
                self._pending = None

        if (
            player_id is not None
            and team_id is not None
            and frame.ball is not None
            and possession.confidence >= self.config.minimum_control_confidence
            and team_id in (TeamId.TEAM_A, TeamId.TEAM_B)
        ):
            self._last_controlled_ball = (
                player_id,
                frame.frame_id,
                frame.timestamp,
                frame.ball.x,
                frame.ball.y,
                min(possession.confidence, frame.ball.confidence),
            )

        self._last_player_id = player_id
        self._last_team_id = team_id
        return events

    def _observe_ball_travel(self, frame: EventFrame) -> None:
        """Record that the ball genuinely left its controlled origin."""
        if self._pending is None or frame.ball is None:
            return
        displacement = euclidean_distance(
            (self._pending.x, self._pending.y),
            (frame.ball.x, frame.ball.y),
        )
        if displacement < self.config.minimum_departure_distance_m:
            return
        self._pending.departed = True
        self._pending.flight_observations += 1

    def _has_flight_evidence(self) -> bool:
        return (
            self._pending is not None
            and self._pending.departed
            and self._pending.flight_observations >= self.config.minimum_flight_observations
        )

    def _origin_from_previous(self) -> _PassOrigin | None:
        if (
            self._last_player_id is None
            or self._last_team_id not in (TeamId.TEAM_A, TeamId.TEAM_B)
            or self._last_controlled_ball is None
            or self._last_controlled_ball[0] != self._last_player_id
        ):
            return None
        _, frame_id, timestamp, x, y, confidence = self._last_controlled_ball
        return _PassOrigin(
            passer_id=self._last_player_id,
            team_id=self._last_team_id,
            frame_id=frame_id,
            timestamp=timestamp,
            x=x,
            y=y,
            confidence=confidence,
        )

    def _finish_pending(
        self,
        frame: EventFrame,
        receiver_id: int | None,
        completed: bool,
    ) -> PassEvent | None:
        assert self._pending is not None
        origin = self._pending
        if frame.ball is None:
            return None
        duration = frame.timestamp - origin.timestamp
        distance = euclidean_distance((origin.x, origin.y), (frame.ball.x, frame.ball.y))
        if (
            not self._has_flight_evidence()
            or duration < self.config.minimum_duration_seconds
            or distance < self.config.minimum_distance_m
        ):
            return None
        return PassEvent(
            timestamp=origin.timestamp,
            frame_id=origin.frame_id,
            team_id=origin.team_id,
            passer_id=origin.passer_id,
            receiver_id=receiver_id,
            start_x=origin.x,
            start_y=origin.y,
            end_x=frame.ball.x,
            end_y=frame.ball.y,
            distance=distance,
            duration=duration,
            completed=completed,
            confidence=min(origin.confidence, frame.ball.confidence),
        )
