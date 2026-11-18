"""OpenCV video-file input adapter."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np

from football_analytics.application.errors import AnalysisErrorCode
from football_analytics.pipeline.video import VideoAnalysisError


class OpenCvFrameSource:
    """Decode a video file into BGR frames."""

    def __init__(
        self,
        path: str | Path,
        *,
        fallback_fps: float = 25.0,
        annotated_video: str | Path | None = None,
        video_codec: str = "mp4v",
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise VideoAnalysisError(
                AnalysisErrorCode.VIDEO_UNREADABLE,
                f"Input video does not exist: {self.path}",
            )
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            self.capture.release()
            raise VideoAnalysisError(
                AnalysisErrorCode.VIDEO_UNREADABLE,
                f"Could not open input video: {self.path}",
            )
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.fps = fps if fps > 0 else fallback_fps
        self._writer: cv2.VideoWriter | None = None
        self.annotated_frame_callback: Callable[[np.ndarray], None] | None = None
        if annotated_video is not None:
            destination = Path(annotated_video)
            destination.parent.mkdir(parents=True, exist_ok=True)
            width = round(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = round(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._writer = cv2.VideoWriter(
                str(destination),
                cv2.VideoWriter_fourcc(*video_codec),
                self.fps,
                (width, height),
            )
            if not self._writer.isOpened():
                self.close()
                raise VideoAnalysisError(
                    AnalysisErrorCode.VIDEO_UNREADABLE,
                    f"Could not open annotated video writer: {destination}",
                )
            self.annotated_frame_callback = self._writer.write

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        frame_id = 0
        while True:
            ok, frame = self.capture.read()
            if not ok:
                return
            yield frame_id, frame
            frame_id += 1

    def close(self) -> None:
        self.capture.release()
        if self._writer is not None:
            self._writer.release()
