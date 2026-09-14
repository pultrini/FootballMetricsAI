"""Static match heatmaps, shot maps, and average-position maps."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from football_analytics.detection import Frame
from football_analytics.storage import (
    ExpectedGoalPrediction,
    PassEvent,
    PlayerPosition,
    PossessionObservation,
    ShotEvent,
    TeamId,
)
from football_analytics.visualization.pitch import Color, PitchRenderer


class HeatmapConfig(BaseModel):
    """Spatial resolution and appearance for density maps."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bins_x: int = Field(default=52, ge=2)
    bins_y: int = Field(default=34, ge=2)
    blur_kernel_size: int = Field(default=9, ge=1)
    opacity: float = Field(default=0.72, ge=0, le=1)


class ShotMapConfig(BaseModel):
    """Marker sizing and BGR colours for shot maps."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_radius: int = Field(default=5, ge=1)
    xg_radius_scale: int = Field(default=18, ge=0)
    team_a_color: Color = (255, 80, 50)
    team_b_color: Color = (50, 50, 255)
    unknown_team_color: Color = (180, 180, 180)
    goal_ring_color: Color = (0, 255, 255)


class HeatmapRenderer:
    """Render density from metric coordinates onto the canonical pitch."""

    def __init__(
        self,
        pitch_renderer: PitchRenderer | None = None,
        config: HeatmapConfig | None = None,
    ) -> None:
        self.pitch_renderer = pitch_renderer or PitchRenderer()
        self.config = config or HeatmapConfig()

    def render_positions(
        self,
        positions: Sequence[PlayerPosition],
        *,
        player_id: int | None = None,
    ) -> Frame:
        """Render all positions or an individual player's heatmap."""
        selected = (
            positions
            if player_id is None
            else [position for position in positions if position.track_id == player_id]
        )
        return self.render_points((position.x, position.y) for position in selected)

    def render_passes(self, passes: Sequence[PassEvent]) -> Frame:
        """Render pass-path density using evenly sampled metric segments."""
        points: list[tuple[float, float]] = []
        for event in passes:
            sample_count = max(2, math.ceil(event.distance) + 1)
            for fraction in np.linspace(0.0, 1.0, sample_count):
                points.append(
                    (
                        event.start_x + float(fraction) * (event.end_x - event.start_x),
                        event.start_y + float(fraction) * (event.end_y - event.start_y),
                    )
                )
        return self.render_points(points)

    def render_shots(self, shots: Sequence[ShotEvent]) -> Frame:
        """Render shot-location density."""
        return self.render_points((shot.x, shot.y) for shot in shots)

    def render_possession(
        self,
        positions: Sequence[PlayerPosition],
        possessions: Sequence[PossessionObservation],
    ) -> Frame:
        """Render locations where a tracked player had confirmed possession."""
        by_identity = {(position.frame_id, position.track_id): position for position in positions}
        points = []
        for possession in possessions:
            if possession.player_id is None:
                continue
            position = by_identity.get((possession.frame_id, possession.player_id))
            if position is not None:
                points.append((position.x, position.y))
        return self.render_points(points)

    def render_points(self, points: Iterable[tuple[float, float]]) -> Frame:
        """Render arbitrary trusted metric points as normalized density."""
        materialized = list(points)
        canvas = self.pitch_renderer.render()
        if not materialized:
            return canvas
        xs = np.asarray([point[0] for point in materialized], dtype=np.float64)
        ys = np.asarray([point[1] for point in materialized], dtype=np.float64)
        if not np.isfinite(xs).all() or not np.isfinite(ys).all():
            raise ValueError("Heatmap points must be finite")
        pitch = self.pitch_renderer.pitch
        if np.any((xs < 0) | (xs > pitch.length) | (ys < 0) | (ys > pitch.width)):
            raise ValueError("Heatmap points must lie inside the pitch")
        density, _, _ = np.histogram2d(
            ys,
            xs,
            bins=(self.config.bins_y, self.config.bins_x),
            range=((0.0, pitch.width), (0.0, pitch.length)),
        )
        kernel = self.config.blur_kernel_size
        if kernel % 2 == 0:
            raise ValueError("blur_kernel_size must be odd")
        density = cv2.GaussianBlur(density.astype(np.float32), (kernel, kernel), 0)
        maximum = float(density.max())
        if maximum <= 0:
            return canvas
        normalized = density / maximum
        pixels_per_metre = self.pitch_renderer.config.pixels_per_metre
        field_width = round(pitch.length * pixels_per_metre)
        field_height = round(pitch.width * pixels_per_metre)
        normalized = cv2.resize(
            normalized,
            (field_width, field_height),
            interpolation=cv2.INTER_CUBIC,
        )
        normalized = np.clip(normalized, 0.0, 1.0)
        colors = cv2.applyColorMap(
            np.uint8(np.clip(normalized * 255, 0, 255)),
            cv2.COLORMAP_TURBO,
        )
        padding = self.pitch_renderer.config.padding
        region = canvas[padding : padding + field_height, padding : padding + field_width]
        alpha = (normalized * self.config.opacity)[..., np.newaxis]
        blended = colors.astype(np.float32) * alpha + region.astype(np.float32) * (1 - alpha)
        region[:] = np.uint8(np.clip(blended, 0, 255))
        return canvas


class ShotMapRenderer:
    """Render shot positions with xG-scaled markers and known goal outcomes."""

    def __init__(
        self,
        pitch_renderer: PitchRenderer | None = None,
        config: ShotMapConfig | None = None,
    ) -> None:
        self.pitch_renderer = pitch_renderer or PitchRenderer()
        self.config = config or ShotMapConfig()

    def render(
        self,
        shots: Sequence[ShotEvent],
        xg_predictions: Sequence[ExpectedGoalPrediction] = (),
    ) -> Frame:
        """Render shots; missing predictions use only the minimum marker size."""
        predictions: dict[tuple[int, int], float] = {}
        for prediction in xg_predictions:
            key = (prediction.frame_id, prediction.player_id)
            if key in predictions:
                raise ValueError(f"Duplicate xG prediction for shot identity {key}")
            predictions[key] = prediction.xg
        canvas = self.pitch_renderer.render()
        for shot in shots:
            xg = predictions.get((shot.frame_id, shot.player_id))
            radius = self.config.minimum_radius + round((xg or 0.0) * self.config.xg_radius_scale)
            color = self._team_color(shot.team_id)
            centre = self.pitch_renderer.metric_to_pixel((shot.x, shot.y))
            cv2.circle(canvas, centre, radius, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, centre, radius, (20, 20, 20), 1, cv2.LINE_AA)
            if shot.outcome is not None and shot.outcome.casefold() in {"goal", "scored"}:
                cv2.circle(
                    canvas,
                    centre,
                    radius + 3,
                    self.config.goal_ring_color,
                    2,
                    cv2.LINE_AA,
                )
        return canvas

    def _team_color(self, value: TeamId | str) -> Color:
        team = TeamId(value)
        if team is TeamId.TEAM_A:
            return self.config.team_a_color
        if team is TeamId.TEAM_B:
            return self.config.team_b_color
        return self.config.unknown_team_color


class AveragePositionRenderer:
    """Render each player's arithmetic mean metric position."""

    def __init__(self, pitch_renderer: PitchRenderer | None = None) -> None:
        self.pitch_renderer = pitch_renderer or PitchRenderer()

    def render(self, positions: Sequence[PlayerPosition]) -> Frame:
        """Render one labeled average marker per track."""
        grouped: dict[int, list[PlayerPosition]] = {}
        for position in positions:
            grouped.setdefault(position.track_id, []).append(position)
        canvas = self.pitch_renderer.render()
        for player_id, samples in sorted(grouped.items()):
            point = (
                sum(sample.x for sample in samples) / len(samples),
                sum(sample.y for sample in samples) / len(samples),
            )
            centre = self.pitch_renderer.metric_to_pixel(point)
            cv2.circle(canvas, centre, 8, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, centre, 8, (20, 20, 20), 1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(player_id),
                (centre[0] + 9, centre[1] - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return canvas
