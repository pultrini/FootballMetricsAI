"""Object-detection contracts and adapters."""

from football_analytics.detection.detector import Detector, Frame, validate_frame
from football_analytics.detection.yolo_detector import YoloDetector, YoloDetectorConfig

__all__ = ["Detector", "Frame", "YoloDetector", "YoloDetectorConfig", "validate_frame"]
