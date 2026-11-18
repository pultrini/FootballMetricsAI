"""No-op progress adapter for the current CLI mode."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class NoOpProgressReporter:
    """Ignore lifecycle notifications when progress output is disabled."""

    def started(self, *, total_frames: int | None) -> None:
        pass

    def progress(self, *, processed_frames: int, total_frames: int | None) -> None:
        pass

    def completed(self, *, summary: Mapping[str, Any]) -> None:
        pass

    def failed(self, *, error_code: str, message: str) -> None:
        pass
