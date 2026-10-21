"""Evidence-based player and team match statistics."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from itertools import pairwise

from pydantic import BaseModel, ConfigDict, Field

from football_analytics.pitch import DEFAULT_PITCH, Pitch
from football_analytics.storage import (
    ExpectedGoalPrediction,
    MatchStatistics,
    PassEvent,
    PlayerPosition,
    PlayerStatistics,
    PossessionObservation,
    ShotEvent,
    TeamId,
    TeamStatistics,
)


class StatisticsConfig(BaseModel):
    """Definitions and gap limits used by match statistics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    maximum_tracking_gap_seconds: float = Field(default=2.0, gt=0)
    progressive_pass_distance_m: float = Field(default=10.0, ge=0)
    team_a_attacks_right: bool = True


class MatchStatisticsCalculator:
    """Aggregate trajectories and events without filling unknown evidence."""

    def __init__(
        self,
        config: StatisticsConfig | None = None,
        *,
        pitch: Pitch = DEFAULT_PITCH,
    ) -> None:
        self.config = config or StatisticsConfig()
        self.pitch = pitch

    def calculate(
        self,
        *,
        positions: Sequence[PlayerPosition] = (),
        possessions: Sequence[PossessionObservation] = (),
        passes: Sequence[PassEvent] = (),
        shots: Sequence[ShotEvent] = (),
        xg_predictions: Sequence[ExpectedGoalPrediction] = (),
    ) -> MatchStatistics:
        """Return serializable statistics for all observed players and both teams."""
        teams_by_player = self._resolve_player_teams(positions, possessions, passes, shots)
        positions_by_player = self._group_positions(
            [
                position
                for position in positions
                if position.team_id is None or TeamId(position.team_id) is not TeamId.REFEREE
            ]
        )
        possession_seconds, team_seconds, spatial_possession = self._possession_data(
            possessions, positions
        )
        prediction_by_shot = self._index_predictions(xg_predictions, shots)

        player_ids = set(teams_by_player) | set(positions_by_player) | set(possession_seconds)
        player_ids.update(event.passer_id for event in passes)
        player_ids.update(event.receiver_id for event in passes if event.receiver_id is not None)
        player_ids.update(event.player_id for event in shots)

        players = tuple(
            self._player_statistics(
                player_id,
                teams_by_player.get(player_id),
                positions_by_player.get(player_id, ()),
                possession_seconds.get(player_id, 0.0),
                passes,
                shots,
                prediction_by_shot,
                possessions,
            )
            for player_id in sorted(player_ids)
        )

        shape = self._team_shape(positions)
        known_possession = sum(team_seconds.values())
        final_third_totals = {
            team: sum(
                duration
                for sample_team, x, _y, duration in spatial_possession
                if sample_team is team and self._in_attacking_final_third(team, x)
            )
            for team in self._match_teams()
        }
        territorial_totals = {
            team: sum(
                duration
                for sample_team, x, _y, duration in spatial_possession
                if sample_team is team and self._in_opposition_half(team, x)
            )
            for team in self._match_teams()
        }
        total_final_third = sum(final_third_totals.values())
        total_territorial = sum(territorial_totals.values())
        team_records = tuple(
            self._team_statistics(
                team,
                team_seconds.get(team, 0.0),
                known_possession,
                passes,
                shots,
                prediction_by_shot,
                final_third_totals[team],
                total_final_third,
                territorial_totals[team],
                total_territorial,
                shape.get(team),
            )
            for team in self._match_teams()
        )
        return MatchStatistics(players=players, teams=team_records)

    def _player_statistics(
        self,
        player_id: int,
        team: TeamId | None,
        positions: Sequence[PlayerPosition],
        possession_seconds: float,
        all_passes: Sequence[PassEvent],
        all_shots: Sequence[ShotEvent],
        prediction_by_shot: dict[tuple[int, int], float],
        possessions: Sequence[PossessionObservation],
    ) -> PlayerStatistics:
        tracked_seconds, distance, maximum_speed = self._trajectory_metrics(positions)
        player_passes = [event for event in all_passes if event.passer_id == player_id]
        completed_passes = sum(event.completed for event in player_passes)
        player_shots = [event for event in all_shots if event.player_id == player_id]
        shots_on_target, goals = self._shot_outcomes(player_shots)
        xg = self._xg_total(player_shots, prediction_by_shot)
        average_x = self._mean([position.x for position in positions])
        average_y = self._mean([position.y for position in positions])
        return PlayerStatistics(
            player_id=player_id,
            team_id=team,
            tracked_seconds=tracked_seconds,
            tracked_minutes=tracked_seconds / 60,
            distance_m=distance,
            average_speed_mps=(distance / tracked_seconds if tracked_seconds > 0 else None),
            maximum_speed_mps=maximum_speed,
            possession_seconds=possession_seconds,
            passes=len(player_passes),
            completed_passes=completed_passes,
            pass_accuracy=self._percentage(completed_passes, len(player_passes)),
            shots=len(player_shots),
            shots_on_target=shots_on_target,
            goals=goals,
            xg=xg,
            xg_per_shot=(xg / len(player_shots) if xg is not None and player_shots else None),
            average_position_x=average_x,
            average_position_y=average_y,
            progressive_distance_with_ball_m=self._progressive_distance_with_ball(
                player_id, team, positions, possessions
            ),
        )

    def _team_statistics(
        self,
        team: TeamId,
        possession_seconds: float,
        known_possession: float,
        all_passes: Sequence[PassEvent],
        all_shots: Sequence[ShotEvent],
        prediction_by_shot: dict[tuple[int, int], float],
        final_third: float,
        total_final_third: float,
        territorial: float,
        total_territorial: float,
        shape: tuple[float, float, float, float] | None,
    ) -> TeamStatistics:
        team_passes = [event for event in all_passes if TeamId(event.team_id) is team]
        completed = [event for event in team_passes if event.completed]
        team_shots = [event for event in all_shots if TeamId(event.team_id) is team]
        shots_on_target, goals = self._shot_outcomes(team_shots)
        xg = self._xg_total(team_shots, prediction_by_shot)
        width, length, compactness, defensive_line = shape or (None, None, None, None)
        return TeamStatistics(
            team_id=team,
            possession_seconds=possession_seconds,
            possession_percentage=self._percentage(possession_seconds, known_possession) or 0.0,
            passes=len(team_passes),
            completed_passes=len(completed),
            pass_accuracy=self._percentage(len(completed), len(team_passes)),
            shots=len(team_shots),
            shots_on_target=shots_on_target,
            goals=goals,
            xg=xg,
            xg_per_shot=(xg / len(team_shots) if xg is not None and team_shots else None),
            field_tilt=self._percentage(final_third, total_final_third),
            territorial_possession=self._percentage(territorial, total_territorial),
            progressive_passes=sum(self._is_progressive(event, team) for event in completed),
            final_third_entries=sum(self._enters_final_third(event, team) for event in completed),
            penalty_area_entries=sum(self._enters_penalty_area(event, team) for event in completed),
            ppda=None,
            average_width_m=width,
            average_length_m=length,
            compactness_m=compactness,
            average_defensive_line_x=defensive_line,
        )

    def _trajectory_metrics(
        self, positions: Sequence[PlayerPosition]
    ) -> tuple[float, float, float | None]:
        ordered = sorted(positions, key=lambda item: (item.timestamp, item.frame_id))
        tracked_seconds = 0.0
        distance = 0.0
        speeds = [item.speed for item in ordered if item.speed is not None]
        for previous, current in pairwise(ordered):
            delta = current.timestamp - previous.timestamp
            if delta <= 0:
                raise ValueError("Player positions must have unique increasing timestamps")
            if delta > self.config.maximum_tracking_gap_seconds:
                continue
            segment = math.hypot(current.x - previous.x, current.y - previous.y)
            tracked_seconds += delta
            distance += segment
            speeds.append(segment / delta)
        return tracked_seconds, distance, max(speeds, default=None)

    def _progressive_distance_with_ball(
        self,
        player_id: int,
        team: TeamId | None,
        positions: Sequence[PlayerPosition],
        possessions: Sequence[PossessionObservation],
    ) -> float:
        if team not in self._match_teams():
            return 0.0
        position_by_frame = {item.frame_id: item for item in positions}
        ordered_possessions = sorted(possessions, key=lambda value: value.timestamp)
        direction = self._direction(team)
        total = 0.0
        for previous, current in pairwise(ordered_possessions):
            if previous.player_id != player_id or current.player_id != player_id:
                continue
            previous_position = position_by_frame.get(previous.frame_id)
            current_position = position_by_frame.get(current.frame_id)
            if previous_position is None or current_position is None:
                continue
            delta = current.timestamp - previous.timestamp
            if 0 < delta <= self.config.maximum_tracking_gap_seconds:
                total += max(
                    0.0,
                    direction * (current_position.x - previous_position.x),
                )
        return total

    def _possession_data(
        self,
        possessions: Sequence[PossessionObservation],
        positions: Sequence[PlayerPosition],
    ) -> tuple[dict[int, float], dict[TeamId, float], list[tuple[TeamId, float, float, float]]]:
        ordered = sorted(possessions, key=lambda item: (item.timestamp, item.frame_id))
        position_by_identity = {(item.frame_id, item.track_id): item for item in positions}
        player_seconds: dict[int, float] = defaultdict(float)
        team_seconds: dict[TeamId, float] = defaultdict(float)
        spatial: list[tuple[TeamId, float, float, float]] = []
        for current, following in pairwise(ordered):
            duration = following.timestamp - current.timestamp
            if duration <= 0:
                raise ValueError("Possession observations must have increasing timestamps")
            if current.player_id is not None:
                player_seconds[current.player_id] += duration
            team = self._match_team(current.team_id)
            if team is None:
                continue
            team_seconds[team] += duration
            if current.player_id is not None:
                position = position_by_identity.get((current.frame_id, current.player_id))
                if position is not None:
                    spatial.append((team, position.x, position.y, duration))
        return dict(player_seconds), dict(team_seconds), spatial

    def _team_shape(
        self, positions: Sequence[PlayerPosition]
    ) -> dict[TeamId, tuple[float, float, float, float]]:
        by_frame_team: dict[tuple[int, TeamId], list[PlayerPosition]] = defaultdict(list)
        for position in positions:
            team = self._match_team(position.team_id)
            if team is not None:
                by_frame_team[(position.frame_id, team)].append(position)
        samples: dict[TeamId, list[tuple[float, float, float, float]]] = defaultdict(list)
        for (_frame_id, team), players in by_frame_team.items():
            if len(players) < 2:
                continue
            xs = [player.x for player in players]
            ys = [player.y for player in players]
            centre_x = sum(xs) / len(xs)
            centre_y = sum(ys) / len(ys)
            compactness = sum(
                math.hypot(player.x - centre_x, player.y - centre_y) for player in players
            ) / len(players)
            defensive_line = min(xs) if self._direction(team) > 0 else max(xs)
            samples[team].append(
                (max(ys) - min(ys), max(xs) - min(xs), compactness, defensive_line)
            )
        return {
            team: tuple(sum(values) / len(values) for values in zip(*items, strict=True))
            for team, items in samples.items()
        }

    def _resolve_player_teams(
        self,
        positions: Sequence[PlayerPosition],
        possessions: Sequence[PossessionObservation],
        passes: Sequence[PassEvent],
        shots: Sequence[ShotEvent],
    ) -> dict[int, TeamId]:
        result: dict[int, TeamId] = {}

        def assign(player_id: int, value: TeamId | str | None) -> None:
            team = self._match_team(value)
            if team is None:
                return
            previous = result.get(player_id)
            if previous is not None and previous is not team:
                raise ValueError(f"Player {player_id} is assigned to conflicting teams")
            result[player_id] = team

        for position in positions:
            assign(position.track_id, position.team_id)
        for possession in possessions:
            if possession.player_id is not None:
                assign(possession.player_id, possession.team_id)
        for event in passes:
            assign(event.passer_id, event.team_id)
            if event.completed and event.receiver_id is not None:
                assign(event.receiver_id, event.team_id)
        for event in shots:
            assign(event.player_id, event.team_id)
        return result

    @staticmethod
    def _group_positions(
        positions: Sequence[PlayerPosition],
    ) -> dict[int, tuple[PlayerPosition, ...]]:
        grouped: dict[int, list[PlayerPosition]] = defaultdict(list)
        for position in positions:
            grouped[position.track_id].append(position)
        return {key: tuple(value) for key, value in grouped.items()}

    @staticmethod
    def _index_predictions(
        predictions: Sequence[ExpectedGoalPrediction],
        shots: Sequence[ShotEvent],
    ) -> dict[tuple[int, int], float]:
        shots_by_identity: dict[tuple[int, int], ShotEvent] = {}
        for shot in shots:
            key = (shot.frame_id, shot.player_id)
            if key in shots_by_identity:
                raise ValueError(f"Duplicate shot identity {key}")
            shots_by_identity[key] = shot
        result: dict[tuple[int, int], float] = {}
        for prediction in predictions:
            key = (prediction.frame_id, prediction.player_id)
            if key in result:
                raise ValueError(f"Duplicate xG prediction for shot identity {key}")
            shot = shots_by_identity.get(key)
            if shot is None:
                raise ValueError(f"xG prediction has no matching shot identity {key}")
            if TeamId(prediction.team_id) is not TeamId(shot.team_id):
                raise ValueError(f"xG prediction team does not match shot identity {key}")
            result[key] = prediction.xg
        return result

    @staticmethod
    def _xg_total(
        shots: Sequence[ShotEvent], predictions: dict[tuple[int, int], float]
    ) -> float | None:
        if not shots:
            return 0.0
        keys = [(shot.frame_id, shot.player_id) for shot in shots]
        if not all(key in predictions for key in keys):
            return None
        return sum(predictions[key] for key in keys)

    @staticmethod
    def _shot_outcomes(shots: Sequence[ShotEvent]) -> tuple[int | None, int | None]:
        if any(shot.outcome is None for shot in shots):
            return None, None
        outcomes = [shot.outcome.casefold() for shot in shots if shot.outcome is not None]
        goals = sum(outcome in {"goal", "scored"} for outcome in outcomes)
        on_target = sum(outcome in {"goal", "scored", "saved", "on_target"} for outcome in outcomes)
        return on_target, goals

    def _is_progressive(self, event: PassEvent, team: TeamId) -> bool:
        progress = self._direction(team) * (event.end_x - event.start_x)
        return progress >= self.config.progressive_pass_distance_m

    def _enters_final_third(self, event: PassEvent, team: TeamId) -> bool:
        start = self._oriented_x(team, event.start_x)
        end = self._oriented_x(team, event.end_x)
        boundary = 2 * self.pitch.length / 3
        return start < boundary <= end

    def _enters_penalty_area(self, event: PassEvent, team: TeamId) -> bool:
        y_min = (self.pitch.width - 40.32) / 2
        y_max = y_min + 40.32
        boundary = self.pitch.length - 16.5
        start_inside = (
            self._oriented_x(team, event.start_x) >= boundary and y_min <= event.start_y <= y_max
        )
        end_inside = (
            self._oriented_x(team, event.end_x) >= boundary and y_min <= event.end_y <= y_max
        )
        return not start_inside and end_inside

    def _in_attacking_final_third(self, team: TeamId, x: float) -> bool:
        return self._oriented_x(team, x) >= 2 * self.pitch.length / 3

    def _in_opposition_half(self, team: TeamId, x: float) -> bool:
        return self._oriented_x(team, x) >= self.pitch.length / 2

    def _oriented_x(self, team: TeamId, x: float) -> float:
        return x if self._direction(team) > 0 else self.pitch.length - x

    def _direction(self, team: TeamId) -> int:
        team_a_direction = 1 if self.config.team_a_attacks_right else -1
        return team_a_direction if team is TeamId.TEAM_A else -team_a_direction

    @staticmethod
    def _match_team(value: TeamId | str | None) -> TeamId | None:
        if value is None:
            return None
        team = TeamId(value)
        return team if team in MatchStatisticsCalculator._match_teams() else None

    @staticmethod
    def _match_teams() -> tuple[TeamId, TeamId]:
        return TeamId.TEAM_A, TeamId.TEAM_B

    @staticmethod
    def _percentage(numerator: float, denominator: float) -> float | None:
        return None if denominator == 0 else 100 * numerator / denominator

    @staticmethod
    def _mean(values: Sequence[float]) -> float | None:
        return sum(values) / len(values) if values else None
