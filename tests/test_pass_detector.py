import pytest

from football_analytics.events import EventFrame, PassDetector, PassDetectorConfig
from football_analytics.storage import BallPosition, PossessionObservation, TeamId


def frame(
    frame_id: int,
    timestamp: float,
    player_id: int | None,
    team_id: TeamId | None,
    ball_x: float,
    *,
    confidence: float = 0.9,
) -> EventFrame:
    ball = BallPosition(
        frame_id=frame_id,
        timestamp=timestamp,
        x=ball_x,
        y=34.0,
        confidence=0.9,
    )
    possession = PossessionObservation(
        frame_id=frame_id,
        timestamp=timestamp,
        player_id=player_id,
        team_id=team_id,
        distance_to_ball=0.5 if player_id is not None else None,
        confidence=confidence if player_id is not None else 0.0,
    )
    return EventFrame(possession=possession, ball=ball)


def test_completed_pass_is_emitted_for_same_team_reception() -> None:
    detector = PassDetector()

    assert detector.update(frame(0, 0.0, 10, TeamId.TEAM_A, 20.0)) == []
    assert detector.update(frame(1, 0.5, None, None, 25.0)) == []
    events = detector.update(frame(2, 1.0, 20, TeamId.TEAM_A, 32.0))

    assert len(events) == 1
    event = events[0]
    assert event.passer_id == 10
    assert event.receiver_id == 20
    assert event.team_id == TeamId.TEAM_A
    assert event.completed
    assert event.start_x == 20.0
    assert event.end_x == 32.0
    assert event.distance == pytest.approx(12.0)
    assert event.duration == pytest.approx(1.0)


def test_opponent_reception_is_an_incomplete_pass() -> None:
    detector = PassDetector()
    detector.update(frame(0, 0.0, 10, TeamId.TEAM_A, 20.0))
    detector.update(frame(1, 0.5, None, None, 25.0))

    events = detector.update(frame(2, 1.0, 30, TeamId.TEAM_B, 32.0))

    assert len(events) == 1
    assert not events[0].completed
    assert events[0].receiver_id is None
    assert events[0].team_id == TeamId.TEAM_A


def test_ball_returning_to_same_player_cancels_pass_candidate() -> None:
    detector = PassDetector()
    detector.update(frame(0, 0.0, 10, TeamId.TEAM_A, 20.0))
    detector.update(frame(1, 0.5, None, None, 25.0))

    assert detector.update(frame(2, 1.0, 10, TeamId.TEAM_A, 26.0)) == []


def test_timed_out_pass_is_recorded_as_incomplete_when_ball_is_known() -> None:
    detector = PassDetector(PassDetectorConfig(maximum_duration_seconds=2.0))
    detector.update(frame(0, 0.0, 10, TeamId.TEAM_A, 20.0))
    detector.update(frame(1, 0.5, None, None, 25.0))

    events = detector.update(frame(2, 3.0, None, None, 35.0))

    assert len(events) == 1
    assert not events[0].completed
    assert events[0].receiver_id is None
    assert events[0].duration == pytest.approx(3.0)


def test_short_possession_flicker_is_not_a_pass() -> None:
    detector = PassDetector(PassDetectorConfig(minimum_distance_m=2.0))
    detector.update(frame(0, 0.0, 10, TeamId.TEAM_A, 20.0))

    assert detector.update(frame(1, 0.1, 20, TeamId.TEAM_A, 20.5)) == []


def test_unknown_receiver_team_does_not_invent_incomplete_pass() -> None:
    detector = PassDetector()
    detector.update(frame(0, 0.0, 10, TeamId.TEAM_A, 20.0))
    detector.update(frame(1, 0.5, None, None, 25.0))

    assert detector.update(frame(2, 1.0, 40, None, 30.0)) == []
    events = detector.update(frame(3, 1.5, 20, TeamId.TEAM_A, 35.0))

    assert len(events) == 1
    assert events[0].completed
    assert events[0].receiver_id == 20


def test_pass_detector_rejects_out_of_order_sequence() -> None:
    detector = PassDetector()
    detector.update(frame(0, 1.0, 10, TeamId.TEAM_A, 20.0))

    with pytest.raises(ValueError, match="increase"):
        detector.update(frame(1, 1.0, 10, TeamId.TEAM_A, 21.0))
