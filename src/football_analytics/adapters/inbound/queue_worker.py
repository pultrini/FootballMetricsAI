"""Small queue worker for ordered incremental analysis segments."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

from football_analytics.application.contracts import AnalysisEvent, JobRequest
from football_analytics.application.errors import AnalysisError, AnalysisErrorCode
from football_analytics.application.ports.frame_source import FrameSource
from football_analytics.application.ports.messaging import MessageConsumer, MessagePublisher
from football_analytics.application.session import AnalysisSession


class QueueWorker:
    """Process one ordered stream at a time while tolerating duplicates."""

    def __init__(
        self,
        consumer: MessageConsumer,
        publisher: MessagePublisher,
        *,
        source_factory: Callable[[JobRequest], FrameSource],
        session_factory: Callable[[JobRequest], AnalysisSession],
    ) -> None:
        self.consumer = consumer
        self.publisher = publisher
        self.source_factory = source_factory
        self.session_factory = session_factory
        self._next_segment: dict[str, int] = {}
        self._pending: dict[tuple[str, int], JobRequest] = {}
        self._sessions: dict[str, AnalysisSession] = {}

    def run_once(self) -> bool:
        """Consume and process one message, if available."""
        request = self.consumer.receive()
        if request is None:
            return False
        stream = str(request.stream_id)
        expected = self._next_segment.get(stream, 0)
        if request.segment_index < expected:
            return True
        if request.segment_index > expected:
            self._pending[(stream, request.segment_index)] = request
            self._publish(
                request,
                "analysis.failed",
                {"error_code": "SEGMENT_GAP", "expected_segment_index": expected},
            )
            return True
        self._process(request)
        self._next_segment[stream] = expected + 1
        while (stream, self._next_segment[stream]) in self._pending:
            next_index = self._next_segment[stream]
            pending = self._pending.pop((stream, next_index))
            self._process(pending)
            self._next_segment[stream] += 1
        return True

    def run(self) -> None:
        """Drain the consumer until it is empty."""
        while self.run_once():
            pass

    def _process(self, request: JobRequest) -> None:
        stream = str(request.stream_id)
        session = self._sessions.setdefault(stream, self.session_factory(request))
        source = self.source_factory(request)
        self._publish(request, "analysis.started", {})
        try:
            metrics = session.process_segment(
                source,
                segment_index=request.segment_index,
                source_offset_seconds=request.source_offset_seconds,
            )
            self._publish(
                request,
                "analysis.segment_completed",
                metrics.model_dump(mode="json"),
            )
            self._publish(
                request,
                "analysis.progress",
                {"frames_processed": metrics.frames_processed},
            )
            self._publish(request, "analysis.completed", session.final_report())
        except AnalysisError as error:
            self._publish(
                request,
                "analysis.failed",
                {"error_code": error.code.value, "message": error.message},
            )
        except Exception as error:
            self._publish(
                request,
                "analysis.failed",
                {"error_code": AnalysisErrorCode.UNKNOWN.value, "message": str(error)},
            )

    def _publish(self, request: JobRequest, event_type: str, data: dict) -> None:
        self.publisher.publish(
            AnalysisEvent(
                event_id=uuid4(),
                stream_id=request.stream_id,
                job_id=request.job_id,
                segment_index=request.segment_index,
                type=event_type,
                occurred_at=datetime.now(UTC),
                data=data,
            )
        )


class InMemoryMessageQueue:
    """Minimal replaceable queue implementation for tests and local runs."""

    def __init__(self, messages: list[JobRequest] | None = None) -> None:
        self.messages = list(messages or [])
        self.events: list[AnalysisEvent] = []

    def receive(self) -> JobRequest | None:
        return self.messages.pop(0) if self.messages else None

    def publish(self, event: AnalysisEvent) -> None:
        self.events.append(event)
