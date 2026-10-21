"""Per-frame top-down minimap and broadcast-frame overlay."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from football_analytics.detection import Frame
from football_analytics.storage import BallPosition, PlayerPosition, TeamId
from football_analytics.visualization.pitch import Color, PitchRenderer


class MinimapConfig(BaseModel):
    """Entity markers and video-overlay settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    marker_radius: int = Field(default=7, ge=1)
    show_track_ids: bool = True
    overlay_width_fraction: float = Field(default=0.32, gt=0, le=1)
    overlay_opacity: float = Field(default=0.9, ge=0, le=1)
    overlay_margin: int = Field(default=16, ge=0)
    team_a_color: Color = (255, 80, 50)
    team_b_color: Color = (50, 50, 255)
    goalkeeper_color: Color = (0, 220, 255)
    referee_color: Color = (255, 0, 255)
    unassigned_color: Color = (180, 180, 180)
    ball_color: Color = (255, 255, 255)
    predicted_ball_color: Color = (130, 130, 130)
    ball_trail_color: Color = (0, 230, 255)
    predicted_ball_trail_color: Color = (100, 100, 100)
    ball_trail_thickness: int = Field(default=2, ge=1)


class MinimapRenderer:
    """Render trusted positions and optionally overlay them on a video frame."""

    def __init__(
        self,
        pitch_renderer: PitchRenderer | None = None,
        config: MinimapConfig | None = None,
    ) -> None:
        self.pitch_renderer = pitch_renderer or PitchRenderer()
        self.config = config or MinimapConfig()

    def render(
        self,
        players: Sequence[PlayerPosition],
        ball: BallPosition | None = None,
        ball_trail: Sequence[BallPosition] = (),
    ) -> Frame:
        """Render all observations belonging to one frame."""
        frame_ids = {player.frame_id for player in players}
        if ball is not None:
            frame_ids.add(ball.frame_id)
        if len(frame_ids) > 1:
            raise ValueError("Minimap observations must belong to exactly one frame")

        canvas = self.pitch_renderer.render()
        self._draw_ball_trail(canvas, ball_trail)
        for player in players:
            center = self.pitch_renderer.metric_to_pixel((player.x, player.y))
            color = self._player_color(player.team_id)
            cv2.circle(
                canvas,
                center,
                self.config.marker_radius,
                (20, 20, 20),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                canvas,
                center,
                max(1, self.config.marker_radius - 2),
                color,
                -1,
                cv2.LINE_AA,
            )
            if self.config.show_track_ids:
                cv2.putText(
                    canvas,
                    str(player.track_id),
                    (center[0] + self.config.marker_radius, center[1] - self.config.marker_radius),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        if ball is not None:
            center = self.pitch_renderer.metric_to_pixel((ball.x, ball.y))
            cv2.circle(canvas, center, 5, (15, 15, 15), -1, cv2.LINE_AA)
            ball_color = (
                self.config.ball_color if ball.observed else self.config.predicted_ball_color
            )
            cv2.circle(canvas, center, 3, ball_color, -1, cv2.LINE_AA)
        return canvas

    def _draw_ball_trail(self, canvas: Frame, trail: Sequence[BallPosition]) -> None:
        """Draw recent tracked ball positions, splitting observed and predicted segments."""
        for previous, current in pairwise(trail):
            color = (
                self.config.ball_trail_color
                if previous.observed and current.observed
                else self.config.predicted_ball_trail_color
            )
            cv2.line(
                canvas,
                self.pitch_renderer.metric_to_pixel((previous.x, previous.y)),
                self.pitch_renderer.metric_to_pixel((current.x, current.y)),
                color,
                self.config.ball_trail_thickness,
                cv2.LINE_AA,
            )

    def overlay(self, frame: Frame, minimap: Frame) -> Frame:
        """Return a copy of a frame with a scaled minimap in its top-right corner."""
        self._validate_image(frame, "frame")
        self._validate_image(minimap, "minimap")
        margin = self.config.overlay_margin
        available_width = frame.shape[1] - 2 * margin
        available_height = frame.shape[0] - 2 * margin
        if available_width < 1 or available_height < 1:
            raise ValueError("frame is too small for the configured overlay margin")

        target_width = min(
            available_width,
            max(1, round(frame.shape[1] * self.config.overlay_width_fraction)),
        )
        scale = min(
            target_width / minimap.shape[1],
            available_height / minimap.shape[0],
        )
        target_size = (
            max(1, round(minimap.shape[1] * scale)),
            max(1, round(minimap.shape[0] * scale)),
        )
        resized = cv2.resize(minimap, target_size, interpolation=cv2.INTER_AREA)
        x_start = frame.shape[1] - margin - resized.shape[1]
        y_start = margin
        result = frame.copy()
        region = result[
            y_start : y_start + resized.shape[0],
            x_start : x_start + resized.shape[1],
        ]
        cv2.addWeighted(
            resized,
            self.config.overlay_opacity,
            region,
            1 - self.config.overlay_opacity,
            0,
            dst=region,
        )
        cv2.rectangle(
            result,
            (x_start, y_start),
            (x_start + resized.shape[1] - 1, y_start + resized.shape[0] - 1),
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        return result

    def _player_color(self, team_id: TeamId | str | None) -> Color:
        if team_id is None:
            return self.config.unassigned_color
        normalized = TeamId(team_id)
        return {
            TeamId.TEAM_A: self.config.team_a_color,
            TeamId.TEAM_B: self.config.team_b_color,
            TeamId.GOALKEEPER: self.config.goalkeeper_color,
            TeamId.REFEREE: self.config.referee_color,
        }[normalized]

    @staticmethod
    def _validate_image(image: Frame, name: str) -> None:
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] == 0
            or image.shape[1] == 0
        ):
            raise ValueError(f"{name} must be a non-empty HxWx3 uint8 BGR array")
