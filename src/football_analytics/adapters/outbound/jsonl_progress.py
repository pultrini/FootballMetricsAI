"""JSON Lines progress reporter for process integrations."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, TextIO


class JsonLinesProgressReporter:
    """Write one valid JSON lifecycle event per line."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout

    def _write(self, event_type: str, data: Mapping[str, Any]) -> None:
        json.dump(
            {
                "type": event_type,
                "occurred_at": datetime.now(UTC).isoformat(),
                "data": dict(data),
            },
            self.stream,
            separators=(",", ":"),
        )
        self.stream.write("\n")
        self.stream.flush()

    def started(self, *, total_frames: int | None) -> None:
        self._write("analysis.started", {"total_frames": total_frames})

    def progress(self, *, processed_frames: int, total_frames: int | None) -> None:
        self._write(
            "analysis.progress",
            {"processed_frames": processed_frames, "total_frames": total_frames},
        )

    def completed(self, *, summary: Mapping[str, Any]) -> None:
        self._write("analysis.completed", summary)

    def failed(self, *, error_code: str, message: str) -> None:
        self._write("analysis.failed", {"error_code": error_code, "message": message})
