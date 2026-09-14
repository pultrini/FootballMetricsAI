import numpy as np
import pytest

from football_analytics.storage import (
    ExpectedGoalPrediction,
    PlayerPosition,
    ShotEvent,
    TeamId,
)
from football_analytics.visualization import (
    AveragePositionRenderer,
    HeatmapConfig,
    HeatmapRenderer,
    PitchCanvasConfig,
    PitchRenderer,
    ShotMapRenderer,
)


def position(frame_id: int, x: float, y: float) -> PlayerPosition:
    return PlayerPosition(
        frame_id=frame_id,
        timestamp=float(frame_id),
        track_id=9,
        team_id=TeamId.TEAM_A,
        x=x,
        y=y,
    )


def shot() -> ShotEvent:
    return ShotEvent(
        frame_id=10,
        timestamp=1.0,
        team_id=TeamId.TEAM_A,
        player_id=9,
        x=94.0,
        y=34.0,
        distance_to_goal=11.0,
        angle_to_goal=0.6,
        outcome="goal",
        confidence=0.9,
    )


def pitch_renderer() -> PitchRenderer:
    return PitchRenderer(PitchCanvasConfig(pixels_per_metre=3.0, padding=10))


def test_individual_heatmap_changes_pitch_without_changing_shape() -> None:
    pitch = pitch_renderer()
    base = pitch.render()
    image = HeatmapRenderer(pitch).render_positions(
        [position(0, 20.0, 20.0), position(1, 21.0, 20.0)],
        player_id=9,
    )

    assert image.shape == base.shape
    assert not np.array_equal(image, base)


def test_empty_heatmap_is_plain_pitch_and_even_kernel_is_rejected() -> None:
    pitch = pitch_renderer()

    assert np.array_equal(HeatmapRenderer(pitch).render_points([]), pitch.render())
    with pytest.raises(ValueError, match="odd"):
        HeatmapRenderer(pitch, HeatmapConfig(blur_kernel_size=4)).render_points([(20.0, 20.0)])


def test_shot_map_uses_xg_marker_and_goal_ring() -> None:
    pitch = pitch_renderer()
    prediction = ExpectedGoalPrediction(
        frame_id=10,
        timestamp=1.0,
        team_id=TeamId.TEAM_A,
        player_id=9,
        xg=0.8,
    )

    without_xg = ShotMapRenderer(pitch).render([shot()])
    with_xg = ShotMapRenderer(pitch).render([shot()], [prediction])

    assert np.count_nonzero(with_xg != pitch.render()) > np.count_nonzero(
        without_xg != pitch.render()
    )


def test_average_position_map_places_marker_at_mean() -> None:
    pitch = pitch_renderer()
    image = AveragePositionRenderer(pitch).render(
        [position(0, 20.0, 20.0), position(1, 40.0, 30.0)]
    )
    x, y = pitch.metric_to_pixel((30.0, 25.0))

    assert tuple(image[y, x]) == (255, 255, 255)
