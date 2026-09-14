import pytest

from football_analytics.pitch import DEFAULT_PITCH, GoalSide, Pitch


def test_canonical_pitch_geometry() -> None:
    assert DEFAULT_PITCH.centre == (52.5, 34.0)
    assert DEFAULT_PITCH.goal_center(GoalSide.RIGHT) == (105.0, 34.0)
    left_post, right_post = DEFAULT_PITCH.goal_posts(GoalSide.LEFT)
    assert left_post == pytest.approx((0.0, 30.34))
    assert right_post == pytest.approx((0.0, 37.66))


def test_pitch_boundary_is_inclusive() -> None:
    assert DEFAULT_PITCH.contains((0.0, 0.0))
    assert DEFAULT_PITCH.contains((105.0, 68.0))
    assert not DEFAULT_PITCH.contains((105.1, 34.0))


def test_invalid_pitch_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        Pitch(width=5.0, goal_width=7.32)
