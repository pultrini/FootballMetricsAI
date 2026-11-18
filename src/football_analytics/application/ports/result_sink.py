"""Output port for analysis artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel


class ResultSink(Protocol):
    """Persist records, reports, and rendered images."""

    def write_analysis(self, result: Any, engine: Any) -> None: ...

    def write_records(self, name: str, records: Sequence[BaseModel]) -> None: ...

    def write_report(self, report: Mapping[str, Any]) -> None: ...

    def write_image(self, relative_path: str, image: np.ndarray) -> None: ...
