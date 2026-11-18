"""Versioned configuration for complete analysis sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from football_analytics.application.errors import AnalysisError, AnalysisErrorCode


class SessionConfig(BaseModel):
    """All CLI-configurable values with the current pipeline defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1"
    config_version: str = "1"
    object_model: Path = Path("models/player_tracker.onnx")
    pitch_model: Path = Path("models/football_field.onnx")
    xg_model: Path | None = None
    device: str | int | None = None
    team_a_color: tuple[int, int, int] | None = None
    team_b_color: tuple[int, int, int] | None = None
    team_a_attacks_right: bool = True
    max_frames: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("max_frames", "maximum_frames"),
    )
    export_formats: tuple[str, ...] = ("parquet",)
    no_annotated_video: bool = False
    fallback_fps: float = Field(default=25.0, gt=0)
    video_codec: str = Field(default="mp4v", min_length=4, max_length=4)
    minimum_homography_confidence: float = Field(default=0.4, ge=0, le=1)
    detector_confidence_threshold: float = Field(default=0.25, ge=0, le=1)
    detector_iou_threshold: float = Field(default=0.7, ge=0, le=1)
    keypoint_detection_confidence_threshold: float = Field(default=0.25, ge=0, le=1)
    keypoint_confidence_threshold: float = Field(default=0.35, ge=0, le=1)
    tracker_lost_track_buffer: int = Field(default=30, ge=0)
    tracker_activation_threshold: float = Field(default=0.7, ge=0, le=1)
    tracker_minimum_consecutive_frames: int = Field(default=2, ge=1)
    tracker_minimum_iou_threshold: float = Field(default=0.1, ge=0, le=1)
    tracker_high_conf_det_threshold: float = Field(default=0.6, ge=0, le=1)
    possession_acquisition_distance_m: float = Field(default=4.0, gt=0)
    possession_release_distance_m: float = Field(default=4.0, gt=0)
    possession_acquisition_frames: int = Field(default=2, ge=1)
    possession_switch_frames: int = Field(default=2, ge=1)
    possession_release_frames: int = Field(default=3, ge=1)
    possession_reference_fps: float = Field(default=25.0, gt=0)
    possession_adapt_temporal_windows_to_fps: bool = True
    possession_minimum_ball_confidence: float = Field(default=0.2, ge=0, le=1)
    shot_minimum_ball_speed_mps: float = Field(default=12.0, gt=0)
    shot_minimum_ball_acceleration_mps2: float = Field(default=8.0, ge=0)
    shot_minimum_separation_m: float = Field(default=1.5, gt=0)
    shot_minimum_goal_direction_cosine: float = Field(default=0.7, ge=-1, le=1)
    shot_minimum_control_confidence: float = Field(default=0.05, ge=0, le=1)
    shot_cooldown_frames: int = Field(default=10, ge=0)
    shot_require_observed_ball: bool = True
    statistics_maximum_tracking_gap_seconds: float = Field(default=2.0, gt=0)
    statistics_progressive_pass_distance_m: float = Field(default=10.0, ge=0)
    maximum_velocity_gap_seconds: float = Field(default=1.0, gt=0)

    @field_validator("team_a_color", "team_b_color")
    @classmethod
    def validate_color(cls, value: tuple[int, int, int] | None) -> tuple[int, int, int] | None:
        if value is not None and (
            len(value) != 3 or any(channel < 0 or channel > 255 for channel in value)
        ):
            raise ValueError("team colours must contain three values from 0 to 255")
        return value

    @field_validator("export_formats")
    @classmethod
    def validate_export_formats(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        supported = {"csv", "json", "parquet"}
        if not value or any(item.casefold().lstrip(".") not in supported for item in value):
            raise ValueError("export_formats must contain only csv, json, or parquet")
        if len(set(value)) != len(value):
            raise ValueError("export_formats must be unique")
        return tuple(item.casefold().lstrip(".") for item in value)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load and validate a session configuration from YAML."""
        source = Path(path)
        try:
            data = yaml.safe_load(source.read_text(encoding="utf-8"))
        except OSError as error:
            raise AnalysisError(
                AnalysisErrorCode.CONFIG_INVALID,
                f"Could not read configuration: {source}",
            ) from error
        except yaml.YAMLError as error:
            raise AnalysisError(
                AnalysisErrorCode.CONFIG_INVALID,
                f"Invalid YAML configuration: {source}",
            ) from error
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise AnalysisError(
                AnalysisErrorCode.CONFIG_INVALID,
                "YAML configuration must contain a mapping at the top level",
            )
        try:
            return cls.model_validate(data)
        except ValueError as error:
            raise AnalysisError(
                AnalysisErrorCode.CONFIG_INVALID,
                "YAML configuration does not match the session schema",
            ) from error

    def with_overrides(self, overrides: dict[str, Any]) -> Self:
        """Return a validated copy with explicitly provided CLI values."""
        values = self.model_dump()
        values.update({key: value for key, value in overrides.items() if value is not None})
        try:
            return type(self).model_validate(values)
        except ValueError as error:
            raise AnalysisError(
                AnalysisErrorCode.CONFIG_INVALID,
                "Session configuration overrides are invalid",
            ) from error
