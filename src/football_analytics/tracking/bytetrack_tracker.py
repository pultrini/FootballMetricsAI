"""ByteTrack adapter built on Roboflow Trackers and Supervision records."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import supervision as sv
from pydantic import BaseModel, ConfigDict, Field
from trackers import ByteTrackTracker as RoboflowByteTrackTracker

from football_analytics.detection import Frame
from football_analytics.storage import Detection, Track
from football_analytics.tracking.tracker import Tracker

LOGGER = logging.getLogger(__name__)
CLASS_NAME_KEY = "class_name"


class ByteTrackConfig(BaseModel):
    """ByteTrack parameters with validated ranges."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lost_track_buffer: int = Field(default=30, ge=0)
    frame_rate: float = Field(default=30.0, gt=0)
    track_activation_threshold: float = Field(default=0.7, ge=0, le=1)
    minimum_consecutive_frames: int = Field(default=2, ge=1)
    minimum_iou_threshold: float = Field(default=0.1, ge=0, le=1)
    high_conf_det_threshold: float = Field(default=0.6, ge=0, le=1)


class ByteTrackBackend(Protocol):
    """Protocol implemented by the external tracker and test doubles."""

    def update(
        self,
        detections: sv.Detections,
        frame: Frame | None = None,
        timestamp: float | None = None,
    ) -> sv.Detections:
        """Associate one frame of detections."""

    def reset(self) -> None:
        """Clear tracker state."""


class ByteTrackTracker(Tracker):
    """Convert domain detections to ByteTrack input and back to domain tracks."""

    def __init__(
        self,
        config: ByteTrackConfig | None = None,
        *,
        backend: ByteTrackBackend | None = None,
    ) -> None:
        self.config = config or ByteTrackConfig()
        self._backend: ByteTrackBackend = backend or RoboflowByteTrackTracker(
            **self.config.model_dump()
        )

    def update(
        self,
        detections: Sequence[Detection],
        *,
        frame_id: int,
        frame: Frame | None = None,
    ) -> list[Track]:
        """Advance ByteTrack and return observations with persistent IDs."""
        if frame_id < 0:
            msg = "frame_id must be non-negative"
            raise ValueError(msg)
        if any(detection.frame_id != frame_id for detection in detections):
            msg = "All detections must belong to the supplied frame_id"
            raise ValueError(msg)

        supervision_detections = self._to_supervision(detections)
        tracked = self._backend.update(supervision_detections, frame=frame)
        if tracked.tracker_id is None:
            msg = "ByteTrack returned detections without tracker IDs"
            raise RuntimeError(msg)
        if tracked.confidence is None:
            msg = "ByteTrack returned detections without confidence scores"
            raise RuntimeError(msg)
        if CLASS_NAME_KEY not in tracked.data:
            msg = "ByteTrack did not preserve class names"
            raise RuntimeError(msg)

        tracks = [
            Track(
                frame_id=frame_id,
                track_id=int(track_id),
                class_name=str(class_name),
                bbox=tuple(float(value) for value in bbox),
                pixel_position=(float((bbox[0] + bbox[2]) / 2), float(bbox[3])),
                confidence=float(confidence),
            )
            for bbox, confidence, track_id, class_name in zip(
                tracked.xyxy,
                tracked.confidence,
                tracked.tracker_id,
                tracked.data[CLASS_NAME_KEY],
                strict=True,
            )
            if int(track_id) >= 0
        ]
        LOGGER.debug("Tracked %d objects in frame %d", len(tracks), frame_id)
        return tracks

    def reset(self) -> None:
        """Clear ByteTrack state for a new video stream."""
        self._backend.reset()

    @staticmethod
    def _to_supervision(detections: Sequence[Detection]) -> sv.Detections:
        return sv.Detections(
            xyxy=np.asarray([detection.bbox for detection in detections], dtype=np.float32).reshape(
                -1, 4
            ),
            confidence=np.asarray(
                [detection.confidence for detection in detections], dtype=np.float32
            ),
            class_id=np.asarray([detection.class_id for detection in detections], dtype=np.int32),
            data={
                CLASS_NAME_KEY: np.asarray(
                    [detection.class_name for detection in detections], dtype=str
                )
            },
        )
