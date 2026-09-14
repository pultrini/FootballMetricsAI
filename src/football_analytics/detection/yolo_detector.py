"""Ultralytics YOLO adapter."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from pydantic import BaseModel, ConfigDict, Field
from ultralytics import YOLO

from football_analytics.detection.conversion import as_numpy
from football_analytics.detection.detector import Detector, Frame, validate_frame
from football_analytics.storage import Detection

LOGGER = logging.getLogger(__name__)


class YoloDetectorConfig(BaseModel):
    """Inference settings that can be changed without altering the adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    confidence_threshold: float = Field(default=0.25, ge=0, le=1)
    iou_threshold: float = Field(default=0.7, ge=0, le=1)
    device: str | int | None = None
    allowed_classes: frozenset[str] | None = None
    class_name_map: Mapping[str, str] = Field(default_factory=dict)


class PredictionModel(Protocol):
    """Small protocol implemented by Ultralytics models and test doubles."""

    def predict(self, source: Frame, **kwargs: object) -> Sequence[object]:
        """Return prediction results for one frame."""


class YoloDetector(Detector):
    """Translate Ultralytics results into model-independent detections."""

    def __init__(
        self,
        model_path: str,
        config: YoloDetectorConfig | None = None,
        *,
        model: PredictionModel | None = None,
    ) -> None:
        self.config = config or YoloDetectorConfig()
        self._model = model or cast("PredictionModel", YOLO(model_path))

    def detect(self, frame: Frame, *, frame_id: int = 0) -> list[Detection]:
        """Run inference for one frame and return validated domain records."""
        validate_frame(frame)
        if frame_id < 0:
            msg = "frame_id must be non-negative"
            raise ValueError(msg)

        prediction_options: dict[str, object] = {
            "conf": self.config.confidence_threshold,
            "iou": self.config.iou_threshold,
            "verbose": False,
        }
        if self.config.device is not None:
            prediction_options["device"] = self.config.device

        results = self._model.predict(source=frame, **prediction_options)
        if len(results) != 1:
            msg = f"Expected one result for one frame, received {len(results)}"
            raise RuntimeError(msg)

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = as_numpy(boxes.xyxy)
        confidences = as_numpy(boxes.conf)
        class_ids = as_numpy(boxes.cls)
        if not (len(xyxy) == len(confidences) == len(class_ids)):
            msg = "YOLO returned inconsistent box, confidence, and class arrays"
            raise RuntimeError(msg)

        names = result.names
        detections: list[Detection] = []
        for raw_bbox, raw_confidence, raw_class_id in zip(
            xyxy, confidences, class_ids, strict=True
        ):
            class_id = int(raw_class_id)
            model_name = str(names[class_id])
            class_name = self.config.class_name_map.get(model_name, model_name)
            if (
                self.config.allowed_classes is not None
                and class_name not in self.config.allowed_classes
            ):
                continue
            bbox = tuple(float(value) for value in raw_bbox)
            if len(bbox) != 4:
                msg = f"Expected xyxy bounding box, received {len(bbox)} values"
                raise RuntimeError(msg)
            detections.append(
                Detection(
                    frame_id=frame_id,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(raw_confidence),
                    bbox=bbox,
                )
            )

        LOGGER.debug("Detected %d objects in frame %d", len(detections), frame_id)
        return detections
