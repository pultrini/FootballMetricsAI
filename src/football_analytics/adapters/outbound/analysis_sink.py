"""Complete-video output adapter backed by the existing exporter."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from football_analytics.pipeline.video import AnalysisResult, OutputManifest, VideoAnalysisPipeline


class ParquetResultSink:
    """Persist a complete analysis using the established output implementation."""

    def __init__(
        self,
        output_directory: str | Path,
        *,
        formats: Sequence[str] = ("parquet",),
        annotated_video: str | Path | None = None,
    ) -> None:
        self.output_directory = Path(output_directory)
        self.formats = tuple(formats)
        self.annotated_video = None if annotated_video is None else Path(annotated_video)
        self.manifest: OutputManifest | None = None

    def write_analysis(self, result: AnalysisResult, pipeline: VideoAnalysisPipeline) -> None:
        self.manifest = pipeline.write_outputs(
            result,
            self.output_directory,
            formats=self.formats,
            annotated_video=self.annotated_video,
        )

    def write_records(self, name: str, records: Sequence[BaseModel]) -> None:
        raise NotImplementedError("Complete-video output must be written through write_analysis")

    def write_report(self, report: dict[str, object]) -> None:
        raise NotImplementedError("Complete-video output must be written through write_analysis")

    def write_image(self, relative_path: str, image: np.ndarray) -> None:
        raise NotImplementedError("Complete-video output must be written through write_analysis")
