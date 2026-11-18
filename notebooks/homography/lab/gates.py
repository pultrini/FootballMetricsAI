"""Geometry and temporal gates for experiments."""

from __future__ import annotations

import numpy as np

from .metrics import homography_geometry


def geometry_gate(
    matrix: np.ndarray,
    *,
    max_condition: float = 1e8,
    require_positive_determinant: bool = True,
) -> tuple[bool, str]:
    """Accept finite, numerically stable homographies."""
    diagnostics = homography_geometry(matrix)
    if not diagnostics["finite"]:
        return False, "non_finite"
    if require_positive_determinant and not diagnostics["positive_determinant"]:
        return False, "orientation"
    if diagnostics["condition_number"] > max_condition:
        return False, "condition"
    return True, "ok"


def temporal_gate(
    current: np.ndarray,
    previous: np.ndarray | None,
    *,
    max_jump: float = 10.0,
) -> tuple[bool, str]:
    """Gate a sequence using the Frobenius change of normalized matrices."""
    if previous is None:
        return True, "initial"
    current_value = np.asarray(current, dtype=float)
    previous_value = np.asarray(previous, dtype=float)
    if current_value.shape != (3, 3) or previous_value.shape != (3, 3):
        raise ValueError("homographies must have shape (3, 3)")
    current_value = current_value / current_value[2, 2]
    previous_value = previous_value / previous_value[2, 2]
    jump = float(np.linalg.norm(current_value - previous_value))
    return (jump <= max_jump, "ok" if jump <= max_jump else "temporal_jump")
