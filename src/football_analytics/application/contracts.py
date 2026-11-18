"""Versioned public records for incremental analysis integrations."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class WindowMetrics(BaseModel):
    """Metrics belonging to one processed segment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    segment_index: int = Field(ge=0)
    frames_processed: int = Field(ge=0)
    distance_m: float = Field(ge=0)
    possession_seconds_team_a: float = Field(ge=0)
    possession_seconds_team_b: float = Field(ge=0)
    passes: int = Field(ge=0)
    completed_passes: int = Field(ge=0)
    shots: int = Field(ge=0)
    xg: float = Field(ge=0)
    average_speed_mps: float | None = Field(default=None, ge=0)
    average_position_x: float | None = Field(default=None, ge=0, le=105)
    average_position_y: float | None = Field(default=None, ge=0, le=68)
    compactness_m: float | None = Field(default=None, ge=0)
    field_tilt: float | None = Field(default=None, ge=0, le=100)
    defensive_line_x: float | None = Field(default=None, ge=0, le=105)
    coverage: float = Field(ge=0, le=1)
    canonical_track_ids: tuple[int, ...] = ()


class AnalysisEvent(BaseModel):
    """Public event envelope emitted by incremental workers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: UUID
    schema_version: Literal["1"] = "1"
    stream_id: UUID
    job_id: UUID
    segment_index: int = Field(ge=0)
    type: Literal[
        "analysis.started",
        "analysis.progress",
        "analysis.segment_completed",
        "analysis.completed",
        "analysis.failed",
    ]
    occurred_at: datetime
    data: dict[str, Any] = Field(default_factory=dict)


class JobRequest(BaseModel):
    """Queue message identifying an ordered segment to process."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    stream_id: UUID
    job_id: UUID
    segment_index: int = Field(ge=0)
    source: str
    source_offset_seconds: float = Field(ge=0)


class ReportContract(BaseModel):
    """Stable top-level report shape matching report.json."""

    model_config = ConfigDict(extra="allow", frozen=True)

    fps: float = Field(gt=0)
    frame_count: int = Field(ge=0)
    event_count: int = Field(ge=0)
    team_assignments: dict[str, str]
    identity_resolution: dict[str, Any]
    statistics: dict[str, Any]
