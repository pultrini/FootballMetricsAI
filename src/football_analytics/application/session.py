"""Application use case for one complete video analysis."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol, TypeVar

import numpy as np

from football_analytics.application.config import SessionConfig
from football_analytics.application.contracts import WindowMetrics
from football_analytics.application.errors import AnalysisError, AnalysisErrorCode
from football_analytics.application.ports import FrameSource, ProgressReporter, ResultSink

ResultT = TypeVar("ResultT")


class AnalysisEngine[ResultT](Protocol):
    """Pipeline behavior required by the application use case."""

    def reset(self) -> None: ...

    def process_frames(
        self,
        frames: Iterable[np.ndarray],
        *,
        fps: float,
        maximum_frames: int | None = None,
        annotated_frame_callback: Callable[[np.ndarray], None] | None = None,
    ) -> ResultT: ...

    def process_segment(
        self,
        frames: Iterable[np.ndarray],
        *,
        fps: float,
        segment_index: int,
        source_offset_seconds: float,
    ) -> WindowMetrics: ...


class AnalysisSession[ResultT]:
    """Run the complete analysis use case with injected boundary adapters."""

    def __init__(
        self,
        pipeline: AnalysisEngine[ResultT],
        config: SessionConfig | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.config = config or SessionConfig()
        self._windows: list[WindowMetrics] = []

    def reset(self) -> None:
        """Clear all state held by the analysis pipeline."""
        self.pipeline.reset()
        self._windows.clear()

    def run(
        self,
        source: FrameSource,
        sink: ResultSink,
        progress: ProgressReporter,
    ) -> ResultT:
        """Process a complete source and persist its result through the sink."""
        self._windows.clear()
        progress.started(total_frames=None)
        try:
            processed_frames = 0

            def frames() -> Iterable[np.ndarray]:
                nonlocal processed_frames
                for _, frame in source.frames():
                    processed_frames += 1
                    if processed_frames % 100 == 0:
                        progress.progress(
                            processed_frames=processed_frames,
                            total_frames=None,
                        )
                    yield frame

            result = self.pipeline.process_frames(
                frames(),
                fps=source.fps,
                maximum_frames=self.config.max_frames,
                annotated_frame_callback=getattr(source, "annotated_frame_callback", None),
            )
            self._write_result(result, sink)
            progress.completed(summary={"frame_count": result.frame_count, "fps": result.fps})
            return result
        except AnalysisError as error:
            progress.failed(error_code=error.code.value, message=error.message)
            raise
        except Exception as error:
            classified = AnalysisError(AnalysisErrorCode.UNKNOWN, str(error))
            progress.failed(error_code=classified.code.value, message=classified.message)
            raise classified from error
        finally:
            source.close()

    def _write_result(self, result: ResultT, sink: ResultSink) -> None:
        sink.write_analysis(result, self.pipeline)

    def process_segment(
        self,
        source: FrameSource,
        *,
        segment_index: int,
        source_offset_seconds: float,
    ) -> WindowMetrics:
        """Process one segment while preserving engine state between calls."""
        processor = getattr(self.pipeline, "process_segment", None)
        if processor is None:
            raise TypeError("Analysis engine does not support incremental segments")
        try:
            metrics = processor(
                (frame for _, frame in source.frames()),
                fps=source.fps,
                segment_index=segment_index,
                source_offset_seconds=source_offset_seconds,
            )
            self._windows.append(metrics)
            return metrics
        finally:
            source.close()

    def final_report(self) -> Mapping[str, Any]:
        """Return additive totals accumulated from processed segments."""
        totals = {
            "schema_version": "1",
            "frames_processed": sum(item.frames_processed for item in self._windows),
            "distance_m": sum(item.distance_m for item in self._windows),
            "possession_seconds_team_a": sum(
                item.possession_seconds_team_a for item in self._windows
            ),
            "possession_seconds_team_b": sum(
                item.possession_seconds_team_b for item in self._windows
            ),
            "passes": sum(item.passes for item in self._windows),
            "completed_passes": sum(item.completed_passes for item in self._windows),
            "shots": sum(item.shots for item in self._windows),
            "xg": sum(item.xg for item in self._windows),
            "coverage": (
                sum(item.coverage * item.frames_processed for item in self._windows)
                / sum(item.frames_processed for item in self._windows)
                if self._windows
                else 0.0
            ),
            "canonical_track_ids": sorted(
                {track_id for item in self._windows for track_id in item.canonical_track_ids}
            ),
            "average_speed_mps": None,
            "average_position_x": None,
            "average_position_y": None,
            "compactness_m": None,
            "field_tilt": None,
            "defensive_line_x": None,
        }
        return totals
