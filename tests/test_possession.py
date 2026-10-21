import pytest

from football_analytics.ball import BallPossessionEstimator, PossessionConfig
from football_analytics.storage import BallPosition, PlayerPosition, TeamId


def player(
    frame_id: int,
    track_id: int,
    team_id: TeamId | None,
    x: float,
    y: float,
    timestamp: float,
) -> PlayerPosition:
    return PlayerPosition(
        frame_id=frame_id,
        timestamp=timestamp,
        track_id=track_id,
        team_id=team_id,
        x=x,
        y=y,
    )


def ball(frame_id: int, timestamp: float, x: float, y: float) -> BallPosition:
    return BallPosition(
        frame_id=frame_id,
        timestamp=timestamp,
        x=x,
        y=y,
        confidence=0.9,
    )


def test_possession_requires_temporal_confirmation() -> None:
    estimator = BallPossessionEstimator(PossessionConfig(acquisition_frames=2))

    first = estimator.update(
        [player(0, 10, TeamId.TEAM_A, 20.0, 20.0, 0.0)],
        ball(0, 0.0, 20.5, 20.0),
        frame_id=0,
        timestamp=0.0,
    )
    second = estimator.update(
        [player(1, 10, TeamId.TEAM_A, 20.0, 20.0, 0.1)],
        ball(1, 0.1, 20.5, 20.0),
        frame_id=1,
        timestamp=0.1,
    )

    assert first.player_id is None
    assert second.player_id == 10
    assert second.team_id == TeamId.TEAM_A


def test_temporal_confirmation_scales_down_for_low_fps_video() -> None:
    estimator = BallPossessionEstimator(
        PossessionConfig(acquisition_frames=2, switch_frames=2, release_frames=3)
    )
    estimator.configure_for_fps(5.0)

    acquired = estimator.update(
        [player(0, 10, TeamId.TEAM_A, 20.0, 20.0, 0.0)],
        ball(0, 0.0, 20.5, 20.0),
        frame_id=0,
        timestamp=0.0,
    )

    assert acquired.player_id == 10


def test_fps_adaptation_can_be_disabled() -> None:
    estimator = BallPossessionEstimator(
        PossessionConfig(acquisition_frames=2, adapt_temporal_windows_to_fps=False)
    )
    estimator.configure_for_fps(5.0)

    pending = estimator.update(
        [player(0, 10, TeamId.TEAM_A, 20.0, 20.0, 0.0)],
        ball(0, 0.0, 20.5, 20.0),
        frame_id=0,
        timestamp=0.0,
    )

    assert pending.player_id is None


def test_release_hysteresis_prevents_one_frame_flicker() -> None:
    estimator = BallPossessionEstimator(PossessionConfig(acquisition_frames=1, release_frames=2))
    estimator.update(
        [player(0, 10, TeamId.TEAM_A, 20.0, 20.0, 0.0)],
        ball(0, 0.0, 20.5, 20.0),
        frame_id=0,
        timestamp=0.0,
    )

    transient = estimator.update(
        [player(1, 10, TeamId.TEAM_A, 20.0, 20.0, 0.1)],
        ball(1, 0.1, 30.0, 20.0),
        frame_id=1,
        timestamp=0.1,
    )
    released = estimator.update(
        [player(2, 10, TeamId.TEAM_A, 20.0, 20.0, 0.2)],
        ball(2, 0.2, 30.0, 20.0),
        frame_id=2,
        timestamp=0.2,
    )

    assert transient.player_id == 10
    assert released.player_id is None


def test_switch_requires_repeated_evidence() -> None:
    estimator = BallPossessionEstimator(PossessionConfig(acquisition_frames=1, switch_frames=2))
    estimator.update(
        [
            player(0, 10, TeamId.TEAM_A, 20.0, 20.0, 0.0),
            player(0, 20, TeamId.TEAM_B, 30.0, 20.0, 0.0),
        ],
        ball(0, 0.0, 20.5, 20.0),
        frame_id=0,
        timestamp=0.0,
    )

    pending = estimator.update(
        [
            player(1, 10, TeamId.TEAM_A, 20.0, 20.0, 0.1),
            player(1, 20, TeamId.TEAM_B, 30.0, 20.0, 0.1),
        ],
        ball(1, 0.1, 30.5, 20.0),
        frame_id=1,
        timestamp=0.1,
    )
    switched = estimator.update(
        [
            player(2, 10, TeamId.TEAM_A, 20.0, 20.0, 0.2),
            player(2, 20, TeamId.TEAM_B, 30.0, 20.0, 0.2),
        ],
        ball(2, 0.2, 30.5, 20.0),
        frame_id=2,
        timestamp=0.2,
    )

    assert pending.player_id == 10
    assert switched.player_id == 20
    assert switched.team_id == TeamId.TEAM_B


def test_possession_summary_is_duration_weighted() -> None:
    estimator = BallPossessionEstimator(PossessionConfig(acquisition_frames=1, switch_frames=1))
    estimator.update(
        [
            player(0, 10, TeamId.TEAM_A, 10.0, 10.0, 0.0),
            player(0, 20, TeamId.TEAM_B, 30.0, 10.0, 0.0),
        ],
        ball(0, 0.0, 10.5, 10.0),
        frame_id=0,
        timestamp=0.0,
    )
    estimator.update(
        [
            player(1, 10, TeamId.TEAM_A, 10.0, 10.0, 5.0),
            player(1, 20, TeamId.TEAM_B, 30.0, 10.0, 5.0),
        ],
        ball(1, 5.0, 30.5, 10.0),
        frame_id=1,
        timestamp=5.0,
    )
    estimator.update(
        [player(2, 20, TeamId.TEAM_B, 30.0, 10.0, 9.0)],
        ball(2, 9.0, 30.5, 10.0),
        frame_id=2,
        timestamp=9.0,
    )

    summary = estimator.summary()

    assert summary.player_seconds == {10: 5.0, 20: 4.0}
    assert summary.team_seconds == {"TEAM_A": 5.0, "TEAM_B": 4.0}
    assert summary.possession_percentage_team_a == pytest.approx(55.5555556)
    assert summary.possession_percentage_team_b == pytest.approx(44.4444444)
    assert (
        summary.possession_percentage_team_a + summary.possession_percentage_team_b
        == pytest.approx(100.0)
    )


def test_referee_is_not_a_possession_candidate() -> None:
    estimator = BallPossessionEstimator(PossessionConfig(acquisition_frames=1))

    observation = estimator.update(
        [player(0, 99, TeamId.REFEREE, 20.0, 20.0, 0.0)],
        ball(0, 0.0, 20.1, 20.0),
        frame_id=0,
        timestamp=0.0,
    )

    assert observation.player_id is None
