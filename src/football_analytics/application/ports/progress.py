"""Output port for coarse-grained analysis progress."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class ProgressReporter(Protocol):
    """Report lifecycle events without being called per detection."""

    def started(self, *, total_frames: int | None) -> None: ...

    def progress(self, *, processed_frames: int, total_frames: int | None) -> None: ...

    def completed(self, *, summary: Mapping[str, Any]) -> None: ...

    def failed(self, *, error_code: str, message: str) -> None: ...
