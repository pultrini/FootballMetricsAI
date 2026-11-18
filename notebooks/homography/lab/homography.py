"""Common variant interface for the homography laboratory."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

Variant = Callable[..., Mapping[str, Any]]


def normalize_result(
    *,
    frame_id: int,
    matrix: np.ndarray | None,
    valid: bool,
    reason: str,
    source: str,
) -> dict[str, Any]:
    """Return the stable result shape shared by all laboratory variants."""
    result: dict[str, Any] = {
        "frame_id": frame_id,
        "valid": bool(valid),
        "reason": reason,
        "source": source,
    }
    if matrix is None:
        result.update({f"h{row}{column}": None for row in range(3) for column in range(3)})
        return result
    value = np.asarray(matrix, dtype=float)
    if value.shape != (3, 3):
        raise ValueError("matrix must have shape (3, 3)")
    result.update(
        {f"h{row}{column}": float(value[row, column]) for row in range(3) for column in range(3)}
    )
    return result
