"""Ports used by application use cases."""

from football_analytics.application.ports.detector import Detector, Frame, validate_frame
from football_analytics.application.ports.frame_source import FrameSource
from football_analytics.application.ports.pitch_keypoints import PitchKeypointDetector
from football_analytics.application.ports.progress import ProgressReporter
from football_analytics.application.ports.result_sink import ResultSink
from football_analytics.application.ports.tracker import Tracker

__all__ = [
    "Detector",
    "Frame",
    "FrameSource",
    "PitchKeypointDetector",
    "ProgressReporter",
    "ResultSink",
    "Tracker",
    "validate_frame",
]
