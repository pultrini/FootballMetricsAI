"""Temporal football event detection."""

from football_analytics.events.detector import EventDetector, EventFrame
from football_analytics.events.pass_detector import PassDetector, PassDetectorConfig
from football_analytics.events.shot_detector import ShotDetector, ShotDetectorConfig

__all__ = [
    "EventDetector",
    "EventFrame",
    "PassDetector",
    "PassDetectorConfig",
    "ShotDetector",
    "ShotDetectorConfig",
]
