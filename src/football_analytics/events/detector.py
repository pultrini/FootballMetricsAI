"""Common event-detector contract and synchronized input record."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, model_validator

from football_analytics.storage import BallPosition, PossessionObservation


class EventFrame(BaseModel):
    """Possession and ball evidence synchronized to one video frame."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    possession: PossessionObservation
    ball: BallPosition | None = None

    @model_validator(mode="after")
    def validate_synchronization(self) -> EventFrame:
        """Reject evidence originating from different frames or timestamps."""
        if self.ball is not None and self.ball.frame_id != self.possession.frame_id:
            raise ValueError("Ball and possession must belong to the same frame")
        if self.ball is not None and self.ball.timestamp != self.possession.timestamp:
            raise ValueError("Ball and possession must use the same timestamp")
        return self

    @property
    def frame_id(self) -> int:
        """Return the synchronized frame identifier."""
        return self.possession.frame_id

    @property
    def timestamp(self) -> float:
        """Return the synchronized timestamp."""
        return self.possession.timestamp


class EventDetector[EventT](ABC):
    """Stateful detector that consumes synchronized temporal evidence."""

    @abstractmethod
    def update(self, frame: EventFrame) -> list[EventT]:
        """Advance state by one frame and return newly completed events."""

    @abstractmethod
    def reset(self) -> None:
        """Clear state before processing another match or discontinuity."""
