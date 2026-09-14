import numpy as np
import pytest

from football_analytics.storage import BallPosition, PlayerPosition, TeamId
from football_analytics.visualization import (
    MinimapConfig,
    MinimapRenderer,
    PitchCanvasConfig,
    PitchRenderer,
)


def player(
    track_id: int,
    team_id: TeamId | None,
    x: float,
    y: float,
    *,
    frame_id: int = 4,
) -> PlayerPosition:
    return PlayerPosition(
        frame_id=frame_id,
        timestamp=1.0,
        track_id=track_id,
        team_id=team_id,
        x=x,
        y=y,
    )


def test_pitch_renderer_has_canonical_aspect_and_markings() -> None:
    renderer = PitchRenderer(PitchCanvasConfig(pixels_per_metre=4.0, padding=10))

    image = renderer.render()

    assert image.shape == (292, 440, 3)
    centre = renderer.metric_to_pixel((52.5, 34.0))
    assert tuple(image[centre[1], centre[0]]) != renderer.config.grass_color


def test_minimap_draws_all_semantic_categories_and_ball() -> None:
    pitch_renderer = PitchRenderer(PitchCanvasConfig(pixels_per_metre=4.0, padding=10))
    config = MinimapConfig(marker_radius=6, show_track_ids=False)
    renderer = MinimapRenderer(pitch_renderer, config)
    players = [
        player(1, TeamId.TEAM_A, 20.0, 10.0),
        player(2, TeamId.TEAM_B, 40.0, 20.0),
        player(3, TeamId.GOALKEEPER, 5.0, 34.0),
        player(4, TeamId.REFEREE, 52.5, 40.0),
        player(5, None, 60.0, 50.0),
    ]
    ball = BallPosition(
        frame_id=4,
        timestamp=1.0,
        x=30.0,
        y=30.0,
        confidence=0.8,
    )

    image = renderer.render(players, ball)

    expected = [
        (players[0], config.team_a_color),
        (players[1], config.team_b_color),
        (players[2], config.goalkeeper_color),
        (players[3], config.referee_color),
        (players[4], config.unassigned_color),
    ]
    for position, color in expected:
        x, y = pitch_renderer.metric_to_pixel((position.x, position.y))
        assert tuple(image[y, x]) == color
    ball_x, ball_y = pitch_renderer.metric_to_pixel((ball.x, ball.y))
    assert tuple(image[ball_y, ball_x]) == config.ball_color


def test_predicted_ball_uses_distinct_color() -> None:
    pitch_renderer = PitchRenderer(PitchCanvasConfig(pixels_per_metre=4.0, padding=10))
    config = MinimapConfig(show_track_ids=False)
    renderer = MinimapRenderer(pitch_renderer, config)
    ball = BallPosition(
        frame_id=4,
        timestamp=1.0,
        x=30.0,
        y=30.0,
        confidence=0.4,
        observed=False,
    )

    image = renderer.render([], ball)
    x, y = pitch_renderer.metric_to_pixel((ball.x, ball.y))

    assert tuple(image[y, x]) == config.predicted_ball_color


def test_minimap_rejects_mixed_frames() -> None:
    renderer = MinimapRenderer()

    with pytest.raises(ValueError, match="one frame"):
        renderer.render(
            [
                player(1, TeamId.TEAM_A, 10.0, 10.0, frame_id=1),
                player(2, TeamId.TEAM_B, 20.0, 20.0, frame_id=2),
            ]
        )


def test_overlay_preserves_shape_and_does_not_modify_input() -> None:
    renderer = MinimapRenderer(config=MinimapConfig(overlay_margin=8))
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    original = frame.copy()
    minimap = renderer.render([player(1, TeamId.TEAM_A, 52.5, 34.0)])

    result = renderer.overlay(frame, minimap)

    assert result.shape == frame.shape
    assert np.array_equal(frame, original)
    assert np.count_nonzero(result) > 0
