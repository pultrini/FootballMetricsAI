"""Canonical landmarks used to relate broadcast images to pitch coordinates."""

from __future__ import annotations

from dataclasses import dataclass

from football_analytics.pitch.pitch_model import DEFAULT_PITCH, Pitch


@dataclass(frozen=True, slots=True)
class PitchLandmark:
    """A named, stable point in the canonical pitch coordinate system."""

    id: str
    name: str
    pitch_x: float
    pitch_y: float


def build_pitch_landmarks(pitch: Pitch = DEFAULT_PITCH) -> tuple[PitchLandmark, ...]:
    """Build the Roboflow 32-keypoint model order in canonical pitch metres."""
    centre_x, centre_y = pitch.centre
    penalty_area_half_width = 40.32 / 2
    goal_area_half_width = 18.32 / 2
    centre_circle_radius = 9.15
    penalty_depth = 16.5
    goal_area_depth = 5.5
    penalty_mark_distance = 11.0

    return (
        PitchLandmark("01", "Left goal line at top touchline", 0.0, 0.0),
        PitchLandmark(
            "02",
            "Left goal line at top penalty-area edge",
            0.0,
            centre_y - penalty_area_half_width,
        ),
        PitchLandmark(
            "03",
            "Left goal line at top goal-area edge",
            0.0,
            centre_y - goal_area_half_width,
        ),
        PitchLandmark(
            "04",
            "Left goal line at bottom goal-area edge",
            0.0,
            centre_y + goal_area_half_width,
        ),
        PitchLandmark(
            "05",
            "Left goal line at bottom penalty-area edge",
            0.0,
            centre_y + penalty_area_half_width,
        ),
        PitchLandmark("06", "Left goal line at bottom touchline", 0.0, pitch.width),
        PitchLandmark(
            "07",
            "Left goal-area top corner",
            goal_area_depth,
            centre_y - goal_area_half_width,
        ),
        PitchLandmark(
            "08",
            "Left goal-area bottom corner",
            goal_area_depth,
            centre_y + goal_area_half_width,
        ),
        PitchLandmark("09", "Left penalty spot", penalty_mark_distance, centre_y),
        PitchLandmark(
            "10",
            "Left penalty-area top corner",
            penalty_depth,
            centre_y - penalty_area_half_width,
        ),
        PitchLandmark(
            "11",
            "Left penalty line at top goal-area edge",
            penalty_depth,
            centre_y - goal_area_half_width,
        ),
        PitchLandmark(
            "12",
            "Left penalty line at bottom goal-area edge",
            penalty_depth,
            centre_y + goal_area_half_width,
        ),
        PitchLandmark(
            "13",
            "Left penalty-area bottom corner",
            penalty_depth,
            centre_y + penalty_area_half_width,
        ),
        PitchLandmark("14", "Halfway line at top touchline", centre_x, 0.0),
        PitchLandmark(
            "15",
            "Top of centre circle",
            centre_x,
            centre_y - centre_circle_radius,
        ),
        PitchLandmark(
            "16",
            "Bottom of centre circle",
            centre_x,
            centre_y + centre_circle_radius,
        ),
        PitchLandmark("17", "Halfway line at bottom touchline", centre_x, pitch.width),
        PitchLandmark(
            "18",
            "Right penalty-area top corner",
            pitch.length - penalty_depth,
            centre_y - penalty_area_half_width,
        ),
        PitchLandmark(
            "19",
            "Right penalty line at top goal-area edge",
            pitch.length - penalty_depth,
            centre_y - goal_area_half_width,
        ),
        PitchLandmark(
            "20",
            "Right penalty line at bottom goal-area edge",
            pitch.length - penalty_depth,
            centre_y + goal_area_half_width,
        ),
        PitchLandmark(
            "21",
            "Right penalty-area bottom corner",
            pitch.length - penalty_depth,
            centre_y + penalty_area_half_width,
        ),
        PitchLandmark(
            "22",
            "Right penalty spot",
            pitch.length - penalty_mark_distance,
            centre_y,
        ),
        PitchLandmark(
            "23",
            "Right goal-area top corner",
            pitch.length - goal_area_depth,
            centre_y - goal_area_half_width,
        ),
        PitchLandmark(
            "24",
            "Right goal-area bottom corner",
            pitch.length - goal_area_depth,
            centre_y + goal_area_half_width,
        ),
        PitchLandmark("25", "Right goal line at top touchline", pitch.length, 0.0),
        PitchLandmark(
            "26",
            "Right goal line at top penalty-area edge",
            pitch.length,
            centre_y - penalty_area_half_width,
        ),
        PitchLandmark(
            "27",
            "Right goal line at top goal-area edge",
            pitch.length,
            centre_y - goal_area_half_width,
        ),
        PitchLandmark(
            "28",
            "Right goal line at bottom goal-area edge",
            pitch.length,
            centre_y + goal_area_half_width,
        ),
        PitchLandmark(
            "29",
            "Right goal line at bottom penalty-area edge",
            pitch.length,
            centre_y + penalty_area_half_width,
        ),
        PitchLandmark("30", "Right goal line at bottom touchline", pitch.length, pitch.width),
        PitchLandmark(
            "31",
            "Left of centre circle",
            centre_x - centre_circle_radius,
            centre_y,
        ),
        PitchLandmark(
            "32",
            "Right of centre circle",
            centre_x + centre_circle_radius,
            centre_y,
        ),
    )


DEFAULT_PITCH_LANDMARKS = build_pitch_landmarks()
