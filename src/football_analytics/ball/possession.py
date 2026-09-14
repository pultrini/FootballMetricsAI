"""Temporally stable player and team possession estimation."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from football_analytics.analytics.geometry import euclidean_distance
from football_analytics.storage import (
    BallPosition,
    PlayerPosition,
    PossessionObservation,
    PossessionSummary,
    TeamId,
)


class PossessionConfig(BaseModel):
    """Distance thresholds and temporal confirmation windows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    acquisition_distance_m: float = Field(default=2.0, gt=0)
    release_distance_m: float = Field(default=3.0, gt=0)
    acquisition_frames: int = Field(default=2, ge=1)
    switch_frames: int = Field(default=2, ge=1)
    release_frames: int = Field(default=3, ge=1)
    minimum_ball_confidence: float = Field(default=0.2, ge=0, le=1)

    @model_validator(mode="after")
    def validate_hysteresis(self) -> PossessionConfig:
        """Ensure release is less sensitive than acquisition."""
        if self.release_distance_m < self.acquisition_distance_m:
            raise ValueError("release_distance_m must be at least acquisition_distance_m")
        return self


class BallPossessionEstimator:
    """Estimate stable possession and accumulate duration-weighted summaries."""

    def __init__(self, config: PossessionConfig | None = None) -> None:
        self.config = config or PossessionConfig()
        self.reset()

    @property
    def current_player_id(self) -> int | None:
        """Return the currently confirmed player in possession."""
        return self._current_player_id

    @property
    def current_team_id(self) -> TeamId | None:
        """Return the semantic team/category for the current player."""
        return self._current_team_id

    def reset(self) -> None:
        """Clear smoothing state and accumulated durations."""
        self._current_player_id: int | None = None
        self._current_team_id: TeamId | None = None
        self._pending_player_id: int | None = None
        self._pending_team_id: TeamId | None = None
        self._pending_frames = 0
        self._release_frames = 0
        self._last_timestamp: float | None = None
        self._player_seconds: dict[int, float] = {}
        self._team_seconds: dict[str, float] = {
            TeamId.TEAM_A.value: 0.0,
            TeamId.TEAM_B.value: 0.0,
        }
        self._unknown_seconds = 0.0

    def update(
        self,
        players: Sequence[PlayerPosition],
        ball: BallPosition | None,
        *,
        frame_id: int,
        timestamp: float,
    ) -> PossessionObservation:
        """Advance possession by one frame using distance and hysteresis."""
        self._validate_inputs(players, ball, frame_id, timestamp)
        self._accrue_elapsed_time(timestamp)

        usable_ball = ball is not None and ball.confidence >= self.config.minimum_ball_confidence
        if not usable_ball:
            self._handle_missing_candidate()
            return self._observation(frame_id, timestamp, None, ball)

        assert ball is not None
        eligible = [
            player
            for player in players
            if player.team_id is None or TeamId(player.team_id) is not TeamId.REFEREE
        ]
        distances = {
            player.track_id: euclidean_distance((player.x, player.y), (ball.x, ball.y))
            for player in eligible
        }
        nearest = min(
            eligible,
            key=lambda player: (distances[player.track_id], player.track_id),
            default=None,
        )
        candidate = (
            nearest
            if nearest is not None
            and distances[nearest.track_id] <= self.config.acquisition_distance_m
            else None
        )

        current = next(
            (player for player in eligible if player.track_id == self._current_player_id),
            None,
        )
        current_distance = None if current is None else distances[current.track_id]
        if (
            current is not None
            and current_distance is not None
            and current_distance <= self.config.release_distance_m
        ):
            self._clear_pending()
            self._release_frames = 0
            return self._observation(frame_id, timestamp, current_distance, ball)

        if candidate is not None:
            required_frames = (
                self.config.acquisition_frames
                if self._current_player_id is None
                else self.config.switch_frames
            )
            self._consider_candidate(candidate, required_frames)
            self._release_frames = 0
        else:
            self._handle_missing_candidate()

        confirmed_distance = distances.get(self._current_player_id)
        return self._observation(frame_id, timestamp, confirmed_distance, ball)

    def summary(self) -> PossessionSummary:
        """Return duration totals and percentages over known team possession."""
        team_a = self._team_seconds[TeamId.TEAM_A.value]
        team_b = self._team_seconds[TeamId.TEAM_B.value]
        known_total = team_a + team_b
        if known_total == 0:
            team_a_percentage = 0.0
            team_b_percentage = 0.0
        else:
            team_a_percentage = 100 * team_a / known_total
            team_b_percentage = 100 * team_b / known_total
        return PossessionSummary(
            player_seconds=dict(self._player_seconds),
            team_seconds=dict(self._team_seconds),
            unknown_seconds=self._unknown_seconds,
            possession_percentage_team_a=team_a_percentage,
            possession_percentage_team_b=team_b_percentage,
        )

    def _consider_candidate(self, candidate: PlayerPosition, required_frames: int) -> None:
        if self._pending_player_id == candidate.track_id:
            self._pending_frames += 1
        else:
            self._pending_player_id = candidate.track_id
            self._pending_team_id = None if candidate.team_id is None else TeamId(candidate.team_id)
            self._pending_frames = 1
        if self._pending_frames >= required_frames:
            self._current_player_id = self._pending_player_id
            self._current_team_id = self._pending_team_id
            self._clear_pending()

    def _handle_missing_candidate(self) -> None:
        self._clear_pending()
        if self._current_player_id is None:
            return
        self._release_frames += 1
        if self._release_frames >= self.config.release_frames:
            self._current_player_id = None
            self._current_team_id = None
            self._release_frames = 0

    def _clear_pending(self) -> None:
        self._pending_player_id = None
        self._pending_team_id = None
        self._pending_frames = 0

    def _accrue_elapsed_time(self, timestamp: float) -> None:
        if self._last_timestamp is None:
            self._last_timestamp = timestamp
            return
        elapsed = timestamp - self._last_timestamp
        if elapsed <= 0:
            raise ValueError("Possession timestamps must increase")
        self._last_timestamp = timestamp
        if self._current_player_id is None:
            self._unknown_seconds += elapsed
            return
        self._player_seconds[self._current_player_id] = (
            self._player_seconds.get(self._current_player_id, 0.0) + elapsed
        )
        if self._current_team_id in (TeamId.TEAM_A, TeamId.TEAM_B):
            assert self._current_team_id is not None
            self._team_seconds[self._current_team_id.value] += elapsed
        else:
            self._unknown_seconds += elapsed

    def _observation(
        self,
        frame_id: int,
        timestamp: float,
        distance: float | None,
        ball: BallPosition | None,
    ) -> PossessionObservation:
        threshold = self.config.release_distance_m
        proximity = 0.0 if distance is None else max(0.0, 1.0 - distance / threshold)
        ball_confidence = 0.0 if ball is None else ball.confidence
        return PossessionObservation(
            frame_id=frame_id,
            timestamp=timestamp,
            player_id=self._current_player_id,
            team_id=self._current_team_id,
            distance_to_ball=distance,
            confidence=ball_confidence * proximity,
        )

    @staticmethod
    def _validate_inputs(
        players: Sequence[PlayerPosition],
        ball: BallPosition | None,
        frame_id: int,
        timestamp: float,
    ) -> None:
        if frame_id < 0 or timestamp < 0:
            raise ValueError("frame_id and timestamp must be non-negative")
        if any(player.frame_id != frame_id for player in players):
            raise ValueError("All players must belong to the supplied frame_id")
        if any(player.timestamp != timestamp for player in players):
            raise ValueError("All players must use the supplied timestamp")
        if ball is not None and ball.frame_id != frame_id:
            raise ValueError("Ball must belong to the supplied frame_id")
        if ball is not None and ball.timestamp != timestamp:
            raise ValueError("Ball must use the supplied timestamp")
