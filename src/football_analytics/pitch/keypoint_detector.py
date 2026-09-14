"""Pitch-keypoint detector abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from football_analytics.detection import Frame
from football_analytics.storage import PitchKeypoint


class PitchKeypointDetector(ABC):
    """Locate known pitch landmarks in one video frame."""

    @abstractmethod
    def detect(self, frame: Frame, *, frame_id: int = 0) -> list[PitchKeypoint]:
        """Return visible landmark correspondences for one frame."""
