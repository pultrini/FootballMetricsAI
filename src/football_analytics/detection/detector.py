"""Detector abstraction independent of any model runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from football_analytics.storage import Detection

type Frame = npt.NDArray[np.uint8]


class Detector(ABC):
    """Convert one video frame into domain detection records."""

    @abstractmethod
    def detect(self, frame: Frame, *, frame_id: int = 0) -> list[Detection]:
        """Detect objects in a BGR frame."""


def validate_frame(frame: Frame) -> None:
    """Validate the common in-memory representation used by vision adapters."""
    if not isinstance(frame, np.ndarray):
        msg = "frame must be a NumPy array"
        raise TypeError(msg)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        msg = "frame must be a non-empty HxWx3 uint8 BGR array"
        raise ValueError(msg)
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        msg = "frame must be a non-empty HxWx3 uint8 BGR array"
        raise ValueError(msg)
