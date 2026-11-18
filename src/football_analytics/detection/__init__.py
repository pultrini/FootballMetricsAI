"""Object-detection contracts and adapters."""

from football_analytics.adapters.outbound.yolo_detector import YoloDetector, YoloDetectorConfig
from football_analytics.application.ports.detector import Detector, Frame, validate_frame

__all__ = ["Detector", "Frame", "YoloDetector", "YoloDetectorConfig", "validate_frame"]
