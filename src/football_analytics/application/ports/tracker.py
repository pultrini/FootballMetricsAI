"""Tracker abstraction independent of tracking libraries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from football_analytics.application.ports import Frame
from football_analytics.domain.schemas import Detection, Track


class Tracker(ABC):
    """Associate detections over time and assign persistent IDs."""

    @abstractmethod
    def update(
        self,
        detections: Sequence[Detection],
        *,
        frame_id: int,
        frame: Frame | None = None,
    ) -> list[Track]:
        """Advance tracker state by exactly one video frame."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all temporal state before processing another video."""
