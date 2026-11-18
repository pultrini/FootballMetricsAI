"""Conversion helpers shared by model-runtime adapters."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def as_numpy(values: object) -> npt.NDArray[np.generic]:
    """Move tensor-like output to CPU and expose it as a NumPy array."""
    converted = values
    detach = getattr(converted, "detach", None)
    if callable(detach):
        converted = detach()
    cpu = getattr(converted, "cpu", None)
    if callable(cpu):
        converted = cpu()
    to_numpy = getattr(converted, "numpy", None)
    if callable(to_numpy):
        converted = to_numpy()
    return np.asarray(converted)
