from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4

import numpy as np

from football_analytics.adapters.inbound.queue_worker import InMemoryMessageQueue, QueueWorker
from football_analytics.application.contracts import JobRequest, WindowMetrics
from football_analytics.application.session import AnalysisSession


class SegmentSource:
    fps = 10.0

    def __init__(self, count: int) -> None:
        self.count = count
        self.closed = False

    def frames(self):
        for index in range(self.count):
            yield index, np.zeros((2, 2, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


class SegmentEngine:
    def reset(self) -> None:
        pass

    def process_frames(self, frames: Iterable[np.ndarray], **kwargs):
        frames = list(frames)
        return type("Result", (), {"fps": 10.0, "frame_count": len(frames)})()

    def process_segment(self, frames, **kwargs) -> WindowMetrics:
        count = len(list(frames))
        return WindowMetrics(
            segment_index=kwargs["segment_index"],
            frames_processed=count,
            distance_m=float(count),
            possession_seconds_team_a=0.1 * count,
            possession_seconds_team_b=0.0,
            passes=count,
            completed_passes=count,
            shots=0,
            xg=0.2 * count,
            coverage=1.0,
            canonical_track_ids=(7,),
        )


def test_segments_accumulate_additive_metrics_and_close_sources() -> None:
    session = AnalysisSession(SegmentEngine())
    first = SegmentSource(2)
    second = SegmentSource(3)

    session.process_segment(first, segment_index=0, source_offset_seconds=0)
    session.process_segment(second, segment_index=1, source_offset_seconds=0.2)

    report = session.final_report()
    assert report["frames_processed"] == 5
    assert report["distance_m"] == 5
    assert report["passes"] == 5
    assert report["xg"] == 1.0
    assert report["canonical_track_ids"] == [7]
    assert first.closed and second.closed


def test_queue_worker_orders_segments_and_ignores_duplicates() -> None:
    stream_id = uuid4()
    requests = [
        JobRequest(
            stream_id=stream_id,
            job_id=uuid4(),
            segment_index=1,
            source="segment-1",
            source_offset_seconds=0.1,
        ),
        JobRequest(
            stream_id=stream_id,
            job_id=uuid4(),
            segment_index=0,
            source="segment-0",
            source_offset_seconds=0,
        ),
        JobRequest(
            stream_id=stream_id,
            job_id=uuid4(),
            segment_index=0,
            source="duplicate",
            source_offset_seconds=0,
        ),
    ]
    queue = InMemoryMessageQueue(requests)
    session = AnalysisSession(SegmentEngine())
    worker = QueueWorker(
        queue,
        queue,
        source_factory=lambda request: SegmentSource(1),
        session_factory=lambda request: session,
    )

    worker.run()

    completed = [
        event.segment_index for event in queue.events if event.type == "analysis.segment_completed"
    ]
    assert completed == [0, 1]
    assert len([event for event in queue.events if event.type == "analysis.started"]) == 2
