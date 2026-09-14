"""Geometry operations in canonical pitch coordinates measured in metres."""

from __future__ import annotations

import math

from football_analytics.pitch.pitch_model import DEFAULT_PITCH, GoalSide, Pitch

Point = tuple[float, float]


def euclidean_distance(first: Point, second: Point) -> float:
    """Return the straight-line distance between two points."""
    return math.hypot(second[0] - first[0], second[1] - first[1])


def shot_distance(
    shot: Point,
    goal_side: GoalSide = GoalSide.RIGHT,
    pitch: Pitch = DEFAULT_PITCH,
) -> float:
    """Return the distance from a shot to the centre of the selected goal."""
    return euclidean_distance(shot, pitch.goal_center(goal_side))


def visible_goal_angle(
    shot: Point,
    goal_side: GoalSide = GoalSide.RIGHT,
    pitch: Pitch = DEFAULT_PITCH,
) -> float:
    """Return the angle subtended by both goalposts at the shot point, in radians.

    The result is in ``[0, pi]`` and uses ``atan2(cross, dot)`` for numerical
    stability. A shot exactly on a goalpost has no defined viewing angle.
    """
    left_post, right_post = pitch.goal_posts(goal_side)
    left_vector = (left_post[0] - shot[0], left_post[1] - shot[1])
    right_vector = (right_post[0] - shot[0], right_post[1] - shot[1])

    if math.hypot(*left_vector) == 0 or math.hypot(*right_vector) == 0:
        msg = "The visible goal angle is undefined at a goalpost"
        raise ValueError(msg)

    cross = abs(left_vector[0] * right_vector[1] - left_vector[1] * right_vector[0])
    dot = left_vector[0] * right_vector[0] + left_vector[1] * right_vector[1]
    return math.atan2(cross, dot)
