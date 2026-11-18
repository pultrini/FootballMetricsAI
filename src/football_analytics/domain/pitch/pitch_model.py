"""Canonical metric model of a football pitch."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

Point = tuple[float, float]


class GoalSide(StrEnum):
    """Goal located on the left or right goal line in canonical coordinates."""

    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True, slots=True)
class Pitch:
    """Pitch dimensions and goal geometry, expressed in metres.

    The origin is the top-left corner in a top-down view. The x-axis runs along
    the touchline and the y-axis runs across the pitch.
    """

    length: float = 105.0
    width: float = 68.0
    goal_width: float = 7.32

    def __post_init__(self) -> None:
        if self.length <= 0 or self.width <= 0 or self.goal_width <= 0:
            msg = "Pitch dimensions must be positive"
            raise ValueError(msg)
        if self.goal_width > self.width:
            msg = "Goal width cannot exceed pitch width"
            raise ValueError(msg)

    @property
    def centre(self) -> Point:
        """Return the centre spot."""
        return (self.length / 2, self.width / 2)

    def goal_center(self, side: GoalSide) -> Point:
        """Return the centre of a goal line."""
        x = 0.0 if side is GoalSide.LEFT else self.length
        return (x, self.width / 2)

    def goal_posts(self, side: GoalSide) -> tuple[Point, Point]:
        """Return the two posts of a goal in canonical coordinates."""
        goal_x, goal_y = self.goal_center(side)
        half_goal = self.goal_width / 2
        return ((goal_x, goal_y - half_goal), (goal_x, goal_y + half_goal))

    def contains(self, point: Point) -> bool:
        """Return whether a point lies on or within the pitch boundaries."""
        return 0 <= point[0] <= self.length and 0 <= point[1] <= self.width


DEFAULT_PITCH = Pitch()
