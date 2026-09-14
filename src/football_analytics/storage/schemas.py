"""Stable, serializable records exchanged by pipeline stages."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

Point = tuple[float, float]
BBox = tuple[float, float, float, float]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
Probability = Annotated[float, Field(ge=0, le=1)]
Percentage = Annotated[float, Field(ge=0, le=100)]
PitchX = Annotated[float, Field(ge=0, le=105)]
PitchY = Annotated[float, Field(ge=0, le=68)]


class TeamId(StrEnum):
    """Supported semantic team assignments."""

    TEAM_A = "TEAM_A"
    TEAM_B = "TEAM_B"
    GOALKEEPER = "GOALKEEPER"
    REFEREE = "REFEREE"


class Record(BaseModel):
    """Base configuration shared by all pipeline records."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)


class Detection(Record):
    """One detector observation in pixel coordinates."""

    frame_id: NonNegativeInt
    class_id: NonNegativeInt
    class_name: str = Field(min_length=1)
    confidence: Probability
    bbox: BBox

    @model_validator(mode="after")
    def validate_bbox(self) -> Detection:
        """Reject inverted bounding boxes."""
        x_min, y_min, x_max, y_max = self.bbox
        if x_max < x_min or y_max < y_min:
            msg = "bbox must use (x_min, y_min, x_max, y_max) ordering"
            raise ValueError(msg)
        return self

    @computed_field
    @property
    def center(self) -> Point:
        """Return the bounding-box centre."""
        x_min, y_min, x_max, y_max = self.bbox
        return ((x_min + x_max) / 2, (y_min + y_max) / 2)


class Track(Record):
    """One tracked object observation in pixel coordinates."""

    frame_id: NonNegativeInt
    track_id: NonNegativeInt
    class_name: str = Field(min_length=1)
    bbox: BBox
    pixel_position: Point
    confidence: Probability

    @model_validator(mode="after")
    def validate_bbox(self) -> Track:
        """Reject inverted bounding boxes."""
        x_min, y_min, x_max, y_max = self.bbox
        if x_max < x_min or y_max < y_min:
            msg = "bbox must use (x_min, y_min, x_max, y_max) ordering"
            raise ValueError(msg)
        return self


class PitchKeypoint(Record):
    """A correspondence between an image landmark and the canonical pitch."""

    frame_id: NonNegativeInt
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    image_x: NonNegativeFloat
    image_y: NonNegativeFloat
    pitch_x: PitchX
    pitch_y: PitchY
    confidence: Probability


class PlayerPosition(Record):
    """A trusted player position transformed to pitch coordinates."""

    frame_id: NonNegativeInt
    timestamp: NonNegativeFloat
    track_id: NonNegativeInt
    team_id: TeamId | None = None
    x: PitchX
    y: PitchY
    velocity_x: float | None = None
    velocity_y: float | None = None
    speed: NonNegativeFloat | None = None


class BallPosition(Record):
    """A trusted ball position transformed to pitch coordinates."""

    frame_id: NonNegativeInt
    timestamp: NonNegativeFloat
    x: PitchX
    y: PitchY
    confidence: Probability
    observed: bool = True
    velocity_x: float | None = None
    velocity_y: float | None = None
    speed: NonNegativeFloat | None = None


class PossessionObservation(Record):
    """Temporally smoothed possession state for one frame."""

    frame_id: NonNegativeInt
    timestamp: NonNegativeFloat
    player_id: NonNegativeInt | None = None
    team_id: TeamId | None = None
    distance_to_ball: NonNegativeFloat | None = None
    confidence: Probability


class PossessionSummary(Record):
    """Accumulated possession durations and known-team percentages."""

    player_seconds: dict[int, NonNegativeFloat]
    team_seconds: dict[str, NonNegativeFloat]
    unknown_seconds: NonNegativeFloat
    possession_percentage_team_a: Percentage
    possession_percentage_team_b: Percentage


class PassEvent(Record):
    """A pass inferred from a transition in ball possession."""

    timestamp: NonNegativeFloat
    frame_id: NonNegativeInt
    team_id: TeamId
    passer_id: NonNegativeInt
    receiver_id: NonNegativeInt | None = None
    start_x: PitchX
    start_y: PitchY
    end_x: PitchX
    end_y: PitchY
    distance: NonNegativeFloat
    duration: NonNegativeFloat
    completed: bool
    confidence: Probability


class ShotEvent(Record):
    """A shot candidate and its measurable evidence; unknown values stay null."""

    timestamp: NonNegativeFloat
    frame_id: NonNegativeInt
    team_id: TeamId
    player_id: NonNegativeInt
    x: PitchX
    y: PitchY
    distance_to_goal: NonNegativeFloat
    angle_to_goal: Annotated[float, Field(ge=0, le=3.141592653589793)]
    ball_speed: NonNegativeFloat | None = None
    body_part: str | None = None
    set_piece: bool | None = None
    pressure: float | None = None
    outcome: str | None = None
    confidence: Probability


class HistoricalShot(BaseModel):
    """One supervised training example from a historical shot dataset."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    shot_x: PitchX
    shot_y: PitchY
    distance: NonNegativeFloat
    angle: Annotated[float, Field(ge=0, le=3.141592653589793)]
    goal: bool
    body_part: str | None = None
    assist_type: str | None = None
    set_piece: bool | None = None
    pressure: float | None = None
    goalkeeper_position: Point | None = None


class ExpectedGoalPrediction(Record):
    """Model output attached to the shot identity used for aggregation."""

    frame_id: NonNegativeInt
    timestamp: NonNegativeFloat
    team_id: TeamId
    player_id: NonNegativeInt
    xg: Probability


class EventRecord(Record):
    """Normalized pass or shot for a unified persistent event table."""

    event_type: Literal["pass", "shot"]
    frame_id: NonNegativeInt
    timestamp: NonNegativeFloat
    team_id: TeamId
    player_id: NonNegativeInt
    related_player_id: NonNegativeInt | None = None
    start_x: PitchX
    start_y: PitchY
    end_x: PitchX | None = None
    end_y: PitchY | None = None
    completed: bool | None = None
    outcome: str | None = None
    xg: Probability | None = None
    confidence: Probability


class XGAggregation(Record):
    """Expected-goals totals grouped by player and team."""

    player_xg: dict[int, NonNegativeFloat]
    team_xg: dict[str, NonNegativeFloat]

    @model_validator(mode="after")
    def validate_equal_totals(self) -> XGAggregation:
        """Ensure player and team totals describe the same predictions."""
        if not math.isclose(
            sum(self.player_xg.values()),
            sum(self.team_xg.values()),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("Sum of player xG must equal sum of team xG")
        return self


class PlayerStatistics(Record):
    """Match statistics attributable to one tracked player."""

    player_id: NonNegativeInt
    team_id: TeamId | None = None
    tracked_seconds: NonNegativeFloat
    tracked_minutes: NonNegativeFloat
    distance_m: NonNegativeFloat
    average_speed_mps: NonNegativeFloat | None = None
    maximum_speed_mps: NonNegativeFloat | None = None
    possession_seconds: NonNegativeFloat
    passes: NonNegativeInt
    completed_passes: NonNegativeInt
    pass_accuracy: Percentage | None = None
    shots: NonNegativeInt
    shots_on_target: NonNegativeInt | None = None
    goals: NonNegativeInt | None = None
    xg: NonNegativeFloat | None = None
    xg_per_shot: NonNegativeFloat | None = None
    average_position_x: PitchX | None = None
    average_position_y: PitchY | None = None
    progressive_distance_with_ball_m: NonNegativeFloat


class TeamStatistics(Record):
    """Match statistics attributable to one of the two competing teams."""

    team_id: TeamId
    possession_seconds: NonNegativeFloat
    possession_percentage: Percentage
    passes: NonNegativeInt
    completed_passes: NonNegativeInt
    pass_accuracy: Percentage | None = None
    shots: NonNegativeInt
    shots_on_target: NonNegativeInt | None = None
    goals: NonNegativeInt | None = None
    xg: NonNegativeFloat | None = None
    xg_per_shot: NonNegativeFloat | None = None
    field_tilt: Percentage | None = None
    territorial_possession: Percentage | None = None
    progressive_passes: NonNegativeInt
    final_third_entries: NonNegativeInt
    penalty_area_entries: NonNegativeInt
    ppda: NonNegativeFloat | None = None
    average_width_m: NonNegativeFloat | None = None
    average_length_m: NonNegativeFloat | None = None
    compactness_m: NonNegativeFloat | None = None
    average_defensive_line_x: PitchX | None = None


class MatchStatistics(Record):
    """Serializable player and team statistics for one analysed match."""

    players: tuple[PlayerStatistics, ...]
    teams: tuple[TeamStatistics, ...]
