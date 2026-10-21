"""Broadcast-frame annotations for the end-to-end video pipeline."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence

import cv2
from pydantic import BaseModel, ConfigDict, Field

from football_analytics.detection import Frame, validate_frame
from football_analytics.storage import (
    BallPosition,
    PlayerPosition,
    PossessionObservation,
    TeamId,
    Track,
)
from football_analytics.visualization.minimap import MinimapRenderer
from football_analytics.visualization.pitch import Color


class VideoRendererConfig(BaseModel):
    """Bounding-box and label appearance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    box_thickness: int = Field(default=2, ge=1)
    font_scale: float = Field(default=0.5, gt=0)
    team_a_color: Color = (255, 80, 50)
    team_b_color: Color = (50, 50, 255)
    other_color: Color = (180, 180, 180)
    ball_color: Color = (0, 230, 255)
    predicted_ball_color: Color = (130, 130, 130)
    ball_radius: int = Field(default=12, ge=2)
    ball_trail_seconds: float = Field(default=2.0, gt=0)


class VideoRenderer:
    """Draw tracks, possession, and a metric minimap without mutating input."""

    def __init__(
        self,
        minimap_renderer: MinimapRenderer | None = None,
        config: VideoRendererConfig | None = None,
    ) -> None:
        self.minimap_renderer = minimap_renderer or MinimapRenderer()
        self.config = config or VideoRendererConfig()
        self._ball_trail: deque[BallPosition] = deque()

    def reset(self) -> None:
        """Clear visual history before rendering another video."""
        self._ball_trail.clear()

    def render(
        self,
        frame: Frame,
        tracks: Sequence[Track],
        positions: Sequence[PlayerPosition],
        ball: BallPosition | None,
        possession: PossessionObservation,
        team_assignments: Mapping[int, TeamId | str],
        ball_pixel_position: tuple[float, float] | None = None,
    ) -> Frame:
        """Return one annotated BGR video frame."""
        validate_frame(frame)
        result = frame.copy()
        for track in tracks:
            team = team_assignments.get(track.track_id)
            color = self._color(team)
            x_min, y_min, x_max, y_max = (round(value) for value in track.bbox)
            cv2.rectangle(
                result,
                (x_min, y_min),
                (x_max, y_max),
                color,
                self.config.box_thickness,
                cv2.LINE_AA,
            )
            label = f"{track.class_name} #{track.track_id}"
            cv2.putText(
                result,
                label,
                (x_min, max(12, y_min - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                color,
                1,
                cv2.LINE_AA,
            )
        if possession.player_id is not None:
            cv2.putText(
                result,
                f"Possession: #{possession.player_id}",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        self._update_ball_trail(ball)
        self._draw_ball_marker(result, ball, ball_pixel_position)
        minimap = self.minimap_renderer.render(positions, ball, tuple(self._ball_trail))
        return self.minimap_renderer.overlay(result, minimap)

    def _update_ball_trail(self, ball: BallPosition | None) -> None:
        if ball is None:
            return
        self._ball_trail.append(ball)
        minimum_timestamp = ball.timestamp - self.config.ball_trail_seconds
        while self._ball_trail and self._ball_trail[0].timestamp < minimum_timestamp:
            self._ball_trail.popleft()

    def _draw_ball_marker(
        self,
        frame: Frame,
        ball: BallPosition | None,
        pixel_position: tuple[float, float] | None,
    ) -> None:
        if ball is None or pixel_position is None:
            return
        center = tuple(round(value) for value in pixel_position)
        height, width = frame.shape[:2]
        if not (0 <= center[0] < width and 0 <= center[1] < height):
            return
        color = self.config.ball_color if ball.observed else self.config.predicted_ball_color
        cv2.circle(frame, center, self.config.ball_radius, color, 2, cv2.LINE_AA)
        cv2.circle(frame, center, 2, color, -1, cv2.LINE_AA)
        label = "BALL" if ball.observed else "BALL predicted"
        cv2.putText(
            frame,
            label,
            (center[0] + self.config.ball_radius + 3, max(14, center[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            color,
            1,
            cv2.LINE_AA,
        )

    def _color(self, value: TeamId | str | None) -> Color:
        if value is None:
            return self.config.other_color
        team = TeamId(value)
        if team is TeamId.TEAM_A:
            return self.config.team_a_color
        if team is TeamId.TEAM_B:
            return self.config.team_b_color
        return self.config.other_color
