"""Message ports used by the incremental analysis worker."""

from __future__ import annotations

from typing import Protocol

from football_analytics.application.contracts import AnalysisEvent, JobRequest


class MessageConsumer(Protocol):
    """Receive ordered or out-of-order job requests."""

    def receive(self) -> JobRequest | None: ...


class MessagePublisher(Protocol):
    """Publish public analysis events."""

    def publish(self, event: AnalysisEvent) -> None: ...
