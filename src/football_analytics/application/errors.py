"""Stable application errors exposed at process boundaries."""

from __future__ import annotations

from enum import StrEnum


class AnalysisErrorCode(StrEnum):
    """Stable categories used by CLI and queue integrations."""

    VIDEO_UNREADABLE = "VIDEO_UNREADABLE"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    CONFIG_INVALID = "CONFIG_INVALID"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    HOMOGRAPHY_UNAVAILABLE = "HOMOGRAPHY_UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


class AnalysisError(RuntimeError):
    """Expected analysis failure with a stable public code."""

    def __init__(self, code: AnalysisErrorCode | str, message: str) -> None:
        self.code = AnalysisErrorCode(code)
        self.message = message
        super().__init__(message)
