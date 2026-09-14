"""OpenCV rendering of the canonical football pitch."""

from __future__ import annotations

import cv2
import numpy as np

from football_analytics.detection import Frame
from football_analytics.pitch.pitch_model import DEFAULT_PITCH, Pitch

type Color = tuple[int, int, int]
type PixelPoint = tuple[int, int]


class PitchCanvasConfig:
    """Pixel layout and BGR colours for a top-down pitch."""

    def __init__(
        self,
        *,
        pixels_per_metre: float = 8.0,
        padding: int = 20,
        line_thickness: int = 2,
        grass_color: Color = (45, 120, 45),
        line_color: Color = (240, 240, 240),
    ) -> None:
        if pixels_per_metre <= 0:
            raise ValueError("pixels_per_metre must be positive")
        if padding < 0 or line_thickness < 1:
            raise ValueError("padding must be non-negative and line_thickness positive")
        self.pixels_per_metre = pixels_per_metre
        self.padding = padding
        self.line_thickness = line_thickness
        self.grass_color = grass_color
        self.line_color = line_color


class PitchRenderer:
    """Render pitch markings and convert metres to canvas pixels."""

    def __init__(
        self,
        config: PitchCanvasConfig | None = None,
        *,
        pitch: Pitch = DEFAULT_PITCH,
    ) -> None:
        self.config = config or PitchCanvasConfig()
        self.pitch = pitch

    @property
    def image_size(self) -> tuple[int, int]:
        """Return canvas size as (height, width)."""
        height = round(self.pitch.width * self.config.pixels_per_metre)
        width = round(self.pitch.length * self.config.pixels_per_metre)
        padding = 2 * self.config.padding
        return (height + padding, width + padding)

    def metric_to_pixel(self, point: tuple[float, float]) -> PixelPoint:
        """Convert canonical metres to the corresponding canvas pixel."""
        x, y = point
        return (
            round(self.config.padding + x * self.config.pixels_per_metre),
            round(self.config.padding + y * self.config.pixels_per_metre),
        )

    def render(self) -> Frame:
        """Draw a complete top-down pitch."""
        height, width = self.image_size
        canvas = np.full((height, width, 3), self.config.grass_color, dtype=np.uint8)
        color = self.config.line_color
        thickness = self.config.line_thickness
        top_left = self.metric_to_pixel((0.0, 0.0))
        bottom_right = self.metric_to_pixel((self.pitch.length, self.pitch.width))

        cv2.rectangle(canvas, top_left, bottom_right, color, thickness, cv2.LINE_AA)
        halfway_top = self.metric_to_pixel((self.pitch.length / 2, 0.0))
        halfway_bottom = self.metric_to_pixel((self.pitch.length / 2, self.pitch.width))
        cv2.line(canvas, halfway_top, halfway_bottom, color, thickness, cv2.LINE_AA)
        centre = self.metric_to_pixel(self.pitch.centre)
        circle_radius = round(9.15 * self.config.pixels_per_metre)
        cv2.circle(canvas, centre, circle_radius, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, centre, max(thickness, 2), color, -1, cv2.LINE_AA)

        self._draw_box(canvas, 0.0, 16.5, 40.32)
        self._draw_box(canvas, self.pitch.length - 16.5, self.pitch.length, 40.32)
        self._draw_box(canvas, 0.0, 5.5, 18.32)
        self._draw_box(canvas, self.pitch.length - 5.5, self.pitch.length, 18.32)
        self._draw_penalty_spot(canvas, 11.0)
        self._draw_penalty_spot(canvas, self.pitch.length - 11.0)
        self._draw_goals(canvas)
        return canvas

    def _draw_box(self, canvas: Frame, x_start: float, x_end: float, width: float) -> None:
        y_start = (self.pitch.width - width) / 2
        y_end = y_start + width
        cv2.rectangle(
            canvas,
            self.metric_to_pixel((x_start, y_start)),
            self.metric_to_pixel((x_end, y_end)),
            self.config.line_color,
            self.config.line_thickness,
            cv2.LINE_AA,
        )

    def _draw_penalty_spot(self, canvas: Frame, x: float) -> None:
        cv2.circle(
            canvas,
            self.metric_to_pixel((x, self.pitch.width / 2)),
            max(self.config.line_thickness, 2),
            self.config.line_color,
            -1,
            cv2.LINE_AA,
        )

    def _draw_goals(self, canvas: Frame) -> None:
        y_start = (self.pitch.width - self.pitch.goal_width) / 2
        y_end = y_start + self.pitch.goal_width
        depth = 2.0
        for x_start, x_end in ((-depth, 0.0), (self.pitch.length, self.pitch.length + depth)):
            cv2.rectangle(
                canvas,
                self.metric_to_pixel((x_start, y_start)),
                self.metric_to_pixel((x_end, y_end)),
                self.config.line_color,
                self.config.line_thickness,
                cv2.LINE_AA,
            )
