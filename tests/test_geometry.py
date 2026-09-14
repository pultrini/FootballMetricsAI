import math

import pytest

from football_analytics.analytics.geometry import (
    euclidean_distance,
    shot_distance,
    visible_goal_angle,
)
from football_analytics.pitch import GoalSide


def test_euclidean_distance_uses_metres() -> None:
    assert euclidean_distance((1.0, 2.0), (4.0, 6.0)) == 5.0


def test_shot_distance_targets_selected_goal_center() -> None:
    assert shot_distance((94.0, 34.0), GoalSide.RIGHT) == 11.0
    assert shot_distance((11.0, 34.0), GoalSide.LEFT) == 11.0


def test_visible_goal_angle_is_angle_between_posts() -> None:
    expected = 2 * math.atan(3.66 / 11.0)
    assert visible_goal_angle((94.0, 34.0)) == pytest.approx(expected)


def test_visible_goal_angle_is_symmetric_about_pitch_centreline() -> None:
    upper = visible_goal_angle((90.0, 25.0))
    lower = visible_goal_angle((90.0, 43.0))
    assert upper == pytest.approx(lower)


def test_visible_goal_angle_is_undefined_on_post() -> None:
    with pytest.raises(ValueError, match="undefined"):
        visible_goal_angle((105.0, 30.34))
