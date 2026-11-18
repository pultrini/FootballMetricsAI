from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from io import StringIO
from types import SimpleNamespace

import numpy as np

from football_analytics.adapters.outbound.jsonl_progress import JsonLinesProgressReporter
from football_analytics.application.errors import AnalysisError, AnalysisErrorCode
from football_analytics.application.session import AnalysisSession, SessionConfig


class FakeSource:
    fps = 10.0

    def __init__(self) -> None:
        self.closed = False

    def frames(self):
        yield 0, np.zeros((2, 2, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


class FakeEngine:
    def __init__(self) -> None:
        self.reset_count = 0
        self.received_frames: list[np.ndarray] = []

    def reset(self) -> None:
        self.reset_count += 1

    def process_frames(
        self,
        frames: Iterable[np.ndarray],
        *,
        fps: float,
        maximum_frames: int | None = None,
        annotated_frame_callback=None,
    ) -> SimpleNamespace:
        self.received_frames = list(frames)
        return SimpleNamespace(fps=fps, frame_count=len(self.received_frames))


class FakeSink:
    def __init__(self) -> None:
        self.result = None
        self.engine = None

    def write_analysis(self, result, engine) -> None:
        self.result = result
        self.engine = engine


class FakeProgress:
    def __init__(self) -> None:
        self.events: list[tuple[str, Mapping[str, object] | None]] = []

    def started(self, *, total_frames: int | None) -> None:
        self.events.append(("started", None))

    def progress(self, *, processed_frames: int, total_frames: int | None) -> None:
        self.events.append(("progress", None))

    def completed(self, *, summary: Mapping[str, object]) -> None:
        self.events.append(("completed", summary))

    def failed(self, *, error_code: str, message: str) -> None:
        self.events.append(("failed", {"error_code": error_code, "message": message}))


def test_analysis_session_delegates_and_closes_source() -> None:
    engine = FakeEngine()
    source = FakeSource()
    sink = FakeSink()
    progress = FakeProgress()
    session = AnalysisSession(engine, SessionConfig(maximum_frames=1))

    result = session.run(source, sink, progress)

    assert result.fps == 10.0
    assert result.frame_count == 1
    assert sink.result == result
    assert sink.engine is engine
    assert source.closed
    assert [event[0] for event in progress.events] == ["started", "completed"]


def test_jsonl_progress_reporter_emits_lifecycle_events() -> None:
    stream = StringIO()
    reporter = JsonLinesProgressReporter(stream)

    reporter.started(total_frames=3)
    reporter.progress(processed_frames=1, total_frames=3)
    reporter.completed(summary={"frame_count": 3})

    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [event["type"] for event in events] == [
        "analysis.started",
        "analysis.progress",
        "analysis.completed",
    ]
    assert events[1]["data"] == {"processed_frames": 1, "total_frames": 3}


def test_analysis_session_reports_classified_failures_and_closes_source() -> None:
    class FailingEngine(FakeEngine):
        def process_frames(self, frames, **kwargs):
            raise AnalysisError(AnalysisErrorCode.INFERENCE_FAILED, "inference failed")

    source = FakeSource()
    progress = FakeProgress()

    try:
        AnalysisSession(FailingEngine()).run(source, FakeSink(), progress)
    except AnalysisError as error:
        assert error.code == AnalysisErrorCode.INFERENCE_FAILED
    else:
        raise AssertionError("expected AnalysisError")

    assert source.closed
    assert progress.events[-1] == (
        "failed",
        {"error_code": "INFERENCE_FAILED", "message": "inference failed"},
    )
