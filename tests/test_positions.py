import pytest

from football_analytics.analytics import MetricPositionEstimator, PositionEstimatorConfig
from football_analytics.pitch import PitchHomography, PitchHomographyConfig
from football_analytics.storage import Detection, TeamId, Track

IMAGE_CORNERS = [(0.0, 0.0), (0.0, 100.0), (100.0, 0.0), (100.0, 100.0)]
PITCH_CORNERS = [(0.0, 0.0), (0.0, 68.0), (105.0, 0.0), (105.0, 68.0)]


def reliable_homography() -> PitchHomography:
    return PitchHomography().estimate(IMAGE_CORNERS, PITCH_CORNERS, [0.9] * 4)


def make_track(
    *,
    frame_id: int,
    track_id: int = 1,
    class_name: str = "player",
    pixel_position: tuple[float, float] = (50.0, 50.0),
) -> Track:
    return Track(
        frame_id=frame_id,
        track_id=track_id,
        class_name=class_name,
        bbox=(40.0, 20.0, 60.0, 50.0),
        pixel_position=pixel_position,
        confidence=0.9,
    )


def test_tracks_are_projected_and_velocity_uses_elapsed_time() -> None:
    estimator = MetricPositionEstimator(reliable_homography())

    first = estimator.project_tracks(
        [make_track(frame_id=0)],
        timestamp=1.0,
        team_assignments={1: TeamId.TEAM_A},
    )[0]
    second = estimator.project_tracks(
        [make_track(frame_id=1, pixel_position=(60.0, 50.0))],
        timestamp=1.5,
        team_assignments={1: TeamId.TEAM_A},
    )[0]

    assert (first.x, first.y) == pytest.approx((52.5, 34.0))
    assert first.team_id == TeamId.TEAM_A
    assert first.velocity_x is None
    assert first.velocity_y is None
    assert first.speed is None
    assert second.velocity_x == pytest.approx(21.0)
    assert second.velocity_y == pytest.approx(0.0)
    assert second.speed == pytest.approx(21.0)


def test_long_observation_gap_does_not_invent_velocity() -> None:
    estimator = MetricPositionEstimator(
        reliable_homography(),
        PositionEstimatorConfig(maximum_velocity_gap_seconds=0.5),
    )
    estimator.project_tracks([make_track(frame_id=0)], timestamp=0.0)

    position = estimator.project_tracks(
        [make_track(frame_id=1, pixel_position=(60.0, 50.0))],
        timestamp=1.0,
    )[0]

    assert position.speed is None


def test_semantic_roles_are_preserved_for_minimap() -> None:
    estimator = MetricPositionEstimator(reliable_homography())
    tracks = [
        make_track(frame_id=0, track_id=7, class_name="goalkeeper"),
        make_track(frame_id=0, track_id=8, class_name="referee"),
        make_track(frame_id=0, track_id=9, class_name="ball"),
    ]

    positions = estimator.project_tracks(tracks, timestamp=0.0)

    assert [position.team_id for position in positions] == [
        TeamId.GOALKEEPER,
        TeamId.REFEREE,
    ]


def test_unreliable_homography_produces_no_positions() -> None:
    config = PitchHomographyConfig(minimum_keypoint_confidence=0.0)
    homography = PitchHomography(config).estimate(
        IMAGE_CORNERS,
        PITCH_CORNERS,
        [0.1] * 4,
    )
    estimator = MetricPositionEstimator(homography)

    assert estimator.project_tracks([make_track(frame_id=0)], timestamp=0.0) == []


def test_off_pitch_projection_is_omitted() -> None:
    estimator = MetricPositionEstimator(reliable_homography())

    positions = estimator.project_tracks(
        [make_track(frame_id=0, pixel_position=(120.0, 50.0))],
        timestamp=0.0,
    )

    assert positions == []


def test_ball_center_is_projected_with_combined_confidence() -> None:
    estimator = MetricPositionEstimator(reliable_homography())
    ball = Detection(
        frame_id=2,
        class_id=3,
        class_name="ball",
        confidence=0.8,
        bbox=(49.0, 49.0, 51.0, 51.0),
    )

    position = estimator.project_ball(ball, timestamp=2.0)

    assert position is not None
    assert (position.x, position.y) == pytest.approx((52.5, 34.0))
    assert position.confidence == pytest.approx(0.72)


def test_velocity_rejects_non_increasing_timestamp() -> None:
    estimator = MetricPositionEstimator(reliable_homography())
    estimator.project_tracks([make_track(frame_id=0)], timestamp=1.0)

    with pytest.raises(ValueError, match="increase"):
        estimator.project_tracks([make_track(frame_id=1)], timestamp=1.0)
