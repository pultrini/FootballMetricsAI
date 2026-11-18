"""Backend port for perspective operations used by pitch homography."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]


class HomographyBackend(Protocol):
    """Provide perspective operations without coupling the domain to OpenCV."""

    def find_homography(
        self,
        image_points: FloatArray,
        pitch_points: FloatArray,
        *,
        reprojection_threshold: float,
        max_iterations: int,
        confidence: float,
    ) -> tuple[FloatArray | None, npt.NDArray[np.uint8] | None]: ...

    def perspective_transform(self, points: FloatArray, matrix: FloatArray) -> FloatArray: ...
