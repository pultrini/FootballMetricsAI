import pytest

from football_analytics.ball import BallTracker, BallTrackerConfig
from football_analytics.storage import BallPosition


def ball(
    frame_id: int,
    timestamp: float,
    x: float,
    y: float,
    confidence: float = 0.9,
) -> BallPosition:
    return BallPosition(
        frame_id=frame_id,
        timestamp=timestamp,
        x=x,
        y=y,
        confidence=confidence,
    )


def test_tracker_initializes_from_most_confident_candidate() -> None:
    tracker = BallTracker()

    tracked = tracker.update(
        [ball(0, 0.0, 10.0, 10.0, 0.7), ball(0, 0.0, 20.0, 20.0, 0.9)],
        frame_id=0,
        timestamp=0.0,
    )

    assert tracked is not None
    assert (tracked.x, tracked.y) == (20.0, 20.0)
    assert tracked.observed
    assert tracked.speed is None


def test_tracker_prefers_temporally_associated_candidate() -> None:
    tracker = BallTracker(BallTrackerConfig(smoothing_alpha=1.0))
    tracker.update([ball(0, 0.0, 10.0, 10.0)], frame_id=0, timestamp=0.0)

    tracked = tracker.update(
        [ball(1, 1.0, 12.0, 10.0, 0.6), ball(1, 1.0, 60.0, 10.0, 0.99)],
        frame_id=1,
        timestamp=1.0,
    )

    assert tracked is not None
    assert (tracked.x, tracked.y) == (12.0, 10.0)
    assert tracked.velocity_x == pytest.approx(2.0)
    assert tracked.velocity_y == pytest.approx(0.0)
    assert tracked.speed == pytest.approx(2.0)


def test_tracker_marks_prediction_and_expires_after_missed_frames() -> None:
    tracker = BallTracker(
        BallTrackerConfig(
            smoothing_alpha=1.0,
            maximum_missed_frames=1,
            prediction_confidence_decay=0.5,
        )
    )
    tracker.update([ball(0, 0.0, 10.0, 10.0)], frame_id=0, timestamp=0.0)
    tracker.update([ball(1, 1.0, 12.0, 10.0)], frame_id=1, timestamp=1.0)

    predicted = tracker.update([], frame_id=2, timestamp=2.0)
    expired = tracker.update([], frame_id=3, timestamp=3.0)

    assert predicted is not None
    assert not predicted.observed
    assert (predicted.x, predicted.y) == pytest.approx((14.0, 10.0))
    assert predicted.confidence == pytest.approx(0.45)
    assert expired is None


def test_tracker_reacquires_strong_distant_detection_without_false_velocity() -> None:
    tracker = BallTracker(
        BallTrackerConfig(
            association_distance_m=2.0,
            reacquisition_confidence_threshold=0.8,
        )
    )
    tracker.update([ball(0, 0.0, 10.0, 10.0)], frame_id=0, timestamp=0.0)

    tracked = tracker.update(
        [ball(1, 1.0, 80.0, 50.0, 0.95)],
        frame_id=1,
        timestamp=1.0,
    )

    assert tracked is not None
    assert (tracked.x, tracked.y) == (80.0, 50.0)
    assert tracked.velocity_x is None
    assert tracked.speed is None


def test_tracker_rejects_non_increasing_time() -> None:
    tracker = BallTracker()
    tracker.update([ball(0, 1.0, 10.0, 10.0)], frame_id=0, timestamp=1.0)

    with pytest.raises(ValueError, match="increase"):
        tracker.update([], frame_id=1, timestamp=1.0)


def test_tracker_rejects_candidate_from_wrong_timestamp() -> None:
    tracker = BallTracker()

    with pytest.raises(ValueError, match="supplied timestamp"):
        tracker.update(
            [ball(0, 0.5, 10.0, 10.0)],
            frame_id=0,
            timestamp=1.0,
        )
