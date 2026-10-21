import numpy as np

from football_analytics.storage import BallPosition, PossessionObservation
from football_analytics.visualization import VideoRenderer, VideoRendererConfig


def test_video_renderer_marks_tracked_ball_on_broadcast_frame() -> None:
    config = VideoRendererConfig(ball_radius=10)
    renderer = VideoRenderer(config=config)
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    ball = BallPosition(
        frame_id=0,
        timestamp=0.0,
        x=52.5,
        y=34.0,
        confidence=0.9,
    )
    possession = PossessionObservation(
        frame_id=0,
        timestamp=0.0,
        confidence=0.0,
    )

    rendered = renderer.render(
        frame,
        [],
        [],
        ball,
        possession,
        {},
        ball_pixel_position=(100.0, 100.0),
    )

    assert tuple(rendered[100, 100]) == config.ball_color
    assert tuple(rendered[100, 110]) == config.ball_color


def test_video_renderer_distinguishes_predicted_ball() -> None:
    config = VideoRendererConfig(ball_radius=10)
    renderer = VideoRenderer(config=config)
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    ball = BallPosition(
        frame_id=0,
        timestamp=0.0,
        x=52.5,
        y=34.0,
        confidence=0.4,
        observed=False,
    )
    possession = PossessionObservation(frame_id=0, timestamp=0.0, confidence=0.0)

    rendered = renderer.render(frame, [], [], ball, possession, {}, (100.0, 100.0))

    assert tuple(rendered[100, 100]) == config.predicted_ball_color
