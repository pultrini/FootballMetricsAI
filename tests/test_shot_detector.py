import pytest

from football_analytics.events import EventFrame, ShotDetector, ShotDetectorConfig
from football_analytics.storage import BallPosition, PossessionObservation, TeamId


def frame(
    frame_id: int,
    timestamp: float,
    player_id: int | None,
    team_id: TeamId | None,
    ball_x: float,
    *,
    speed: float,
    velocity_x: float,
    possession_confidence: float,
    observed: bool = True,
) -> EventFrame:
    possession = PossessionObservation(
        frame_id=frame_id,
        timestamp=timestamp,
        player_id=player_id,
        team_id=team_id,
        distance_to_ball=0.5 if possession_confidence > 0 else 4.0,
        confidence=possession_confidence,
    )
    ball = BallPosition(
        frame_id=frame_id,
        timestamp=timestamp,
        x=ball_x,
        y=34.0,
        confidence=0.9,
        observed=observed,
        velocity_x=velocity_x,
        velocity_y=0.0,
        speed=speed,
    )
    return EventFrame(possession=possession, ball=ball)


def test_shot_requires_acceleration_separation_and_goal_direction() -> None:
    detector = ShotDetector()
    detector.update(
        frame(
            0,
            0.0,
            10,
            TeamId.TEAM_A,
            80.0,
            speed=1.0,
            velocity_x=1.0,
            possession_confidence=0.8,
        )
    )

    events = detector.update(
        frame(
            1,
            0.1,
            10,
            TeamId.TEAM_A,
            83.0,
            speed=20.0,
            velocity_x=20.0,
            possession_confidence=0.0,
        )
    )

    assert len(events) == 1
    shot = events[0]
    assert shot.player_id == 10
    assert shot.team_id == TeamId.TEAM_A
    assert (shot.x, shot.y) == (80.0, 34.0)
    assert shot.distance_to_goal == pytest.approx(25.0)
    assert 0 < shot.angle_to_goal < 3.141592653589793
    assert shot.ball_speed == 20.0
    assert shot.body_part is None
    assert shot.pressure is None
    assert shot.set_piece is None
    assert shot.outcome is None


def test_fast_ball_moving_away_from_goal_is_not_a_shot() -> None:
    detector = ShotDetector()
    detector.update(
        frame(
            0,
            0.0,
            10,
            TeamId.TEAM_A,
            80.0,
            speed=1.0,
            velocity_x=1.0,
            possession_confidence=0.8,
        )
    )

    events = detector.update(
        frame(
            1,
            0.1,
            10,
            TeamId.TEAM_A,
            77.0,
            speed=20.0,
            velocity_x=-20.0,
            possession_confidence=0.0,
        )
    )

    assert events == []


def test_ball_without_acceleration_is_not_a_shot() -> None:
    detector = ShotDetector()
    detector.update(
        frame(
            0,
            0.0,
            10,
            TeamId.TEAM_A,
            80.0,
            speed=15.0,
            velocity_x=15.0,
            possession_confidence=0.8,
        )
    )

    events = detector.update(
        frame(
            1,
            0.1,
            10,
            TeamId.TEAM_A,
            83.0,
            speed=15.0,
            velocity_x=15.0,
            possession_confidence=0.0,
        )
    )

    assert events == []


def test_predicted_ball_cannot_create_shot_by_default() -> None:
    detector = ShotDetector()
    detector.update(
        frame(
            0,
            0.0,
            10,
            TeamId.TEAM_A,
            80.0,
            speed=1.0,
            velocity_x=1.0,
            possession_confidence=0.8,
        )
    )

    events = detector.update(
        frame(
            1,
            0.1,
            10,
            TeamId.TEAM_A,
            83.0,
            speed=20.0,
            velocity_x=20.0,
            possession_confidence=0.0,
            observed=False,
        )
    )

    assert events == []


def test_cooldown_prevents_duplicate_shot_events() -> None:
    detector = ShotDetector(ShotDetectorConfig(cooldown_frames=5))
    detector.update(
        frame(
            0,
            0.0,
            10,
            TeamId.TEAM_A,
            80.0,
            speed=1.0,
            velocity_x=1.0,
            possession_confidence=0.8,
        )
    )
    first = detector.update(
        frame(
            1,
            0.1,
            10,
            TeamId.TEAM_A,
            83.0,
            speed=20.0,
            velocity_x=20.0,
            possession_confidence=0.0,
        )
    )
    duplicate = detector.update(
        frame(
            2,
            0.2,
            10,
            TeamId.TEAM_A,
            86.0,
            speed=25.0,
            velocity_x=25.0,
            possession_confidence=0.0,
        )
    )

    assert len(first) == 1
    assert duplicate == []


def test_team_b_shot_must_travel_toward_left_goal() -> None:
    detector = ShotDetector()
    detector.update(
        frame(
            0,
            0.0,
            20,
            TeamId.TEAM_B,
            25.0,
            speed=1.0,
            velocity_x=-1.0,
            possession_confidence=0.8,
        )
    )

    events = detector.update(
        frame(
            1,
            0.1,
            20,
            TeamId.TEAM_B,
            22.0,
            speed=20.0,
            velocity_x=-20.0,
            possession_confidence=0.0,
        )
    )

    assert len(events) == 1
    assert events[0].distance_to_goal == pytest.approx(25.0)


def test_shot_detector_rejects_out_of_order_sequence() -> None:
    detector = ShotDetector()
    detector.update(
        frame(
            0,
            1.0,
            None,
            None,
            50.0,
            speed=1.0,
            velocity_x=1.0,
            possession_confidence=0.0,
        )
    )

    with pytest.raises(ValueError, match="increase"):
        detector.update(
            frame(
                1,
                1.0,
                None,
                None,
                50.0,
                speed=1.0,
                velocity_x=1.0,
                possession_confidence=0.0,
            )
        )
