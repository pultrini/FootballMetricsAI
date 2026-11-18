"""Pure metrics used by the homography notebooks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Project Nx2 points through a 3x3 homography."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    matrix = np.asarray(homography, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("homography must have shape (3, 3)")
    homogeneous = np.column_stack((values, np.ones(len(values))))
    projected = homogeneous @ matrix.T
    denominator = projected[:, 2]
    if np.any(np.isclose(denominator, 0)):
        raise ValueError("homography projects a point to infinity")
    return projected[:, :2] / denominator[:, None]


def grid_jump(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return Euclidean displacement for corresponding projected points."""
    before = np.asarray(previous, dtype=float)
    after = np.asarray(current, dtype=float)
    if before.shape != after.shape or before.ndim != 2 or before.shape[1] != 2:
        raise ValueError("point arrays must have equal shape (n, 2)")
    return np.linalg.norm(after - before, axis=1)


def percentile_summary(values: Sequence[float]) -> dict[str, float | None]:
    """Summarize finite values without turning an empty result into zero."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"median": None, "p95": None, "max": None}
    return {
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def homography_geometry(matrix: np.ndarray) -> dict[str, float | bool]:
    """Return basic conditioning and orientation diagnostics for a homography."""
    value = np.asarray(matrix, dtype=float)
    if value.shape != (3, 3):
        raise ValueError("homography must have shape (3, 3)")
    determinant = float(np.linalg.det(value))
    condition = float(np.linalg.cond(value))
    return {
        "determinant": determinant,
        "condition_number": condition,
        "finite": bool(np.isfinite(value).all()),
        "positive_determinant": determinant > 0,
    }
