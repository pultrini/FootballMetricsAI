"""Temporal football event detection."""

from football_analytics.domain.events.detector import EventDetector, EventFrame
from football_analytics.domain.events.pass_detector import PassDetector, PassDetectorConfig
from football_analytics.domain.events.shot_detector import ShotDetector, ShotDetectorConfig

__all__ = [
    "EventDetector",
    "EventFrame",
    "PassDetector",
    "PassDetectorConfig",
    "ShotDetector",
    "ShotDetectorConfig",
]
