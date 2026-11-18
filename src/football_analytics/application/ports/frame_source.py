"""Input port for decoded video frames."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

import numpy as np


class FrameSource(Protocol):
    """Provide BGR frames and video metadata."""

    fps: float

    def frames(self) -> Iterator[tuple[int, np.ndarray]]: ...

    def close(self) -> None: ...
