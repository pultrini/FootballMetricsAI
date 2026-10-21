"""Canonical player identities from fragmented metric-space tracklets."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import linear_sum_assignment

from football_analytics.storage import IdentityResolutionSummary, PlayerPosition, TeamId, Track


class IdentityResolverConfig(BaseModel):
    """Spatial, temporal, and motion gates for reconnecting tracklets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    maximum_missing_seconds: float = Field(default=2.0, gt=0)
    base_distance_m: float = Field(default=2.0, ge=0)
    maximum_player_speed_mps: float = Field(default=12.0, gt=0)
    maximum_association_distance_m: float = Field(default=12.0, gt=0)
    unknown_team_penalty_m: float = Field(default=0.5, ge=0)
    velocity_smoothing_alpha: float = Field(default=0.5, gt=0, le=1)


@dataclass(slots=True)
class _IdentityState:
    canonical_id: int
    raw_track_id: int
    frame_id: int
    timestamp: float
    x: float
    y: float
    team_id: TeamId | None
    velocity_x: float | None = None
    velocity_y: float | None = None


class MetricIdentityResolver:
    """Reconnect short ByteTrack fragments using their projected pitch motion."""

    def __init__(self, config: IdentityResolverConfig | None = None) -> None:
        self.config = config or IdentityResolverConfig()
        self.reset()

    def reset(self) -> None:
        """Clear identities before processing a new video."""
        self._states: dict[int, _IdentityState] = {}
        self._raw_to_canonical: dict[int, int] = {}
        self._raw_history: dict[int, int] = {}
        self._last_timestamp: float | None = None
        self._next_canonical_id = 0

    def update(
        self,
        positions: Sequence[PlayerPosition],
        *,
        frame_id: int,
        timestamp: float,
    ) -> list[PlayerPosition]:
        """Return positions relabelled with stable, one-to-one canonical IDs."""
        self._validate_inputs(positions, frame_id, timestamp)
        direct: dict[int, int] = {}
        used_canonical_ids: set[int] = set()
        unmatched: list[PlayerPosition] = []

        for position in sorted(positions, key=lambda item: item.track_id):
            canonical_id = self._raw_to_canonical.get(position.track_id)
            if canonical_id is None or canonical_id in used_canonical_ids:
                unmatched.append(position)
                continue
            direct[position.track_id] = canonical_id
            used_canonical_ids.add(canonical_id)

        available = [
            state
            for state in self._states.values()
            if state.canonical_id not in used_canonical_ids
            and 0 < timestamp - state.timestamp <= self.config.maximum_missing_seconds
        ]
        matched = self._associate(unmatched, available, timestamp)

        assignments = dict(direct)
        for raw_track_id, canonical_id in matched.items():
            previous_raw = self._states[canonical_id].raw_track_id
            self._raw_to_canonical.pop(previous_raw, None)
            self._raw_to_canonical[raw_track_id] = canonical_id
            self._raw_history[raw_track_id] = canonical_id
            assignments[raw_track_id] = canonical_id
            used_canonical_ids.add(canonical_id)

        for position in unmatched:
            if position.track_id in assignments:
                continue
            canonical_id = self._allocate_id(position.track_id)
            self._raw_to_canonical[position.track_id] = canonical_id
            self._raw_history[position.track_id] = canonical_id
            assignments[position.track_id] = canonical_id

        resolved: list[PlayerPosition] = []
        for position in positions:
            canonical_id = assignments[position.track_id]
            item = self._resolved_position(position, canonical_id)
            resolved.append(item)
            self._remember(position.track_id, item)
        self._last_timestamp = timestamp
        return resolved

    def canonical_id(self, raw_track_id: int) -> int | None:
        """Return the canonical identity historically assigned to one raw ID."""
        return self._raw_history.get(raw_track_id)

    def summary(self) -> IdentityResolutionSummary:
        """Return the complete raw-to-canonical mapping for the processed video."""
        mapping = dict(sorted(self._raw_history.items()))
        raw_track_count = len(mapping)
        canonical_track_count = len(set(mapping.values()))
        return IdentityResolutionSummary(
            raw_track_count=raw_track_count,
            canonical_track_count=canonical_track_count,
            merged_tracklet_count=raw_track_count - canonical_track_count,
            raw_to_canonical=mapping,
        )

    def relabel_tracks(self, tracks: Sequence[Track]) -> list[Track]:
        """Relabel currently resolved raw tracks for annotated-video display."""
        result: list[Track] = []
        for track in tracks:
            canonical_id = self.canonical_id(track.track_id)
            if canonical_id is None:
                result.append(track)
                continue
            result.append(Track.model_validate({**track.model_dump(), "track_id": canonical_id}))
        return result

    def resolve_team_assignments(
        self, assignments: Mapping[int, TeamId | str]
    ) -> dict[int, TeamId]:
        """Combine raw-track team labels into canonical-identity majority votes."""
        votes: dict[int, Counter[TeamId]] = {}
        for raw_track_id, value in assignments.items():
            canonical_id = self._raw_history.get(raw_track_id)
            if canonical_id is None:
                continue
            votes.setdefault(canonical_id, Counter())[TeamId(value)] += 1
        for canonical_id, state in self._states.items():
            if state.team_id is not None:
                votes.setdefault(canonical_id, Counter())[state.team_id] += 1
        return {
            canonical_id: min(
                counter,
                key=lambda team: (-counter[team], team.value),
            )
            for canonical_id, counter in votes.items()
        }

    def _associate(
        self,
        positions: Sequence[PlayerPosition],
        states: Sequence[_IdentityState],
        timestamp: float,
    ) -> dict[int, int]:
        if not positions or not states:
            return {}
        impossible = 1e9
        costs = np.full((len(positions), len(states)), impossible, dtype=np.float64)
        for row, position in enumerate(positions):
            for column, state in enumerate(states):
                cost = self._association_cost(position, state, timestamp)
                if cost is not None:
                    costs[row, column] = cost
        rows, columns = linear_sum_assignment(costs)
        return {
            positions[row].track_id: states[column].canonical_id
            for row, column in zip(rows.tolist(), columns.tolist(), strict=True)
            if costs[row, column] < impossible
        }

    def _association_cost(
        self,
        position: PlayerPosition,
        state: _IdentityState,
        timestamp: float,
    ) -> float | None:
        if not self._teams_compatible(position.team_id, state.team_id):
            return None
        elapsed = timestamp - state.timestamp
        velocity_x, velocity_y = self._bounded_velocity(state)
        predicted_x = state.x + velocity_x * elapsed
        predicted_y = state.y + velocity_y * elapsed
        distance = math.hypot(position.x - predicted_x, position.y - predicted_y)
        gate = min(
            self.config.maximum_association_distance_m,
            self.config.base_distance_m + self.config.maximum_player_speed_mps * elapsed,
        )
        if distance > gate:
            return None
        unknown_penalty = (
            self.config.unknown_team_penalty_m
            if position.team_id is None or state.team_id is None
            else 0.0
        )
        return distance + unknown_penalty

    def _resolved_position(self, position: PlayerPosition, canonical_id: int) -> PlayerPosition:
        state = self._states.get(canonical_id)
        team_id = position.team_id
        velocity_x = velocity_y = speed = None
        if state is not None:
            team_id = state.team_id or team_id
            elapsed = position.timestamp - state.timestamp
            if elapsed > 0:
                measured_x = (position.x - state.x) / elapsed
                measured_y = (position.y - state.y) / elapsed
                alpha = self.config.velocity_smoothing_alpha
                velocity_x = (
                    measured_x
                    if state.velocity_x is None
                    else alpha * measured_x + (1 - alpha) * state.velocity_x
                )
                velocity_y = (
                    measured_y
                    if state.velocity_y is None
                    else alpha * measured_y + (1 - alpha) * state.velocity_y
                )
                speed = math.hypot(velocity_x, velocity_y)
        return PlayerPosition(
            frame_id=position.frame_id,
            timestamp=position.timestamp,
            track_id=canonical_id,
            team_id=team_id,
            x=position.x,
            y=position.y,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            speed=speed,
        )

    def _remember(self, raw_track_id: int, position: PlayerPosition) -> None:
        previous = self._states.get(position.track_id)
        observed_team = None if position.team_id is None else TeamId(position.team_id)
        team_id = observed_team if previous is None else previous.team_id or observed_team
        velocity_x = position.velocity_x
        velocity_y = position.velocity_y
        if velocity_x is not None and velocity_y is not None:
            speed = math.hypot(velocity_x, velocity_y)
            if speed > self.config.maximum_player_speed_mps:
                scale = self.config.maximum_player_speed_mps / speed
                velocity_x *= scale
                velocity_y *= scale
        self._states[position.track_id] = _IdentityState(
            canonical_id=position.track_id,
            raw_track_id=raw_track_id,
            frame_id=position.frame_id,
            timestamp=position.timestamp,
            x=position.x,
            y=position.y,
            team_id=team_id,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
        )
        self._raw_to_canonical[raw_track_id] = position.track_id
        self._raw_history[raw_track_id] = position.track_id

    def _bounded_velocity(self, state: _IdentityState) -> tuple[float, float]:
        return (state.velocity_x or 0.0, state.velocity_y or 0.0)

    def _allocate_id(self, preferred: int) -> int:
        if preferred not in self._states:
            self._next_canonical_id = max(self._next_canonical_id, preferred + 1)
            return preferred
        while self._next_canonical_id in self._states:
            self._next_canonical_id += 1
        canonical_id = self._next_canonical_id
        self._next_canonical_id += 1
        return canonical_id

    @staticmethod
    def _teams_compatible(left: TeamId | str | None, right: TeamId | None) -> bool:
        if left is None or right is None:
            return True
        return TeamId(left) is right

    def _validate_inputs(
        self,
        positions: Sequence[PlayerPosition],
        frame_id: int,
        timestamp: float,
    ) -> None:
        if frame_id < 0 or timestamp < 0:
            raise ValueError("frame_id and timestamp must be non-negative")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("Identity resolver timestamps must increase")
        if any(position.frame_id != frame_id for position in positions):
            raise ValueError("All positions must belong to the supplied frame_id")
        if any(position.timestamp != timestamp for position in positions):
            raise ValueError("All positions must use the supplied timestamp")
        raw_ids = [position.track_id for position in positions]
        if len(raw_ids) != len(set(raw_ids)):
            raise ValueError("Each raw track ID may appear only once per frame")
