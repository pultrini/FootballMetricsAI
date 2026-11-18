"""Video metadata and cache helpers for notebook experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def video_metadata(path: Path) -> dict[str, Any]:
    """Read basic video metadata without decoding the complete clip."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {path}")
        return {
            "path": str(path),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


def read_json(path: Path) -> dict[str, Any]:
    """Read a cached JSON object with an explicit missing-file error."""
    import json

    if not path.exists():
        raise FileNotFoundError(f"Cache artifact does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value
