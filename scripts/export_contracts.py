"""Export versioned JSON Schemas for neutral-language consumers."""

from __future__ import annotations

import json
from pathlib import Path

from football_analytics.application.contracts import (
    AnalysisEvent,
    JobRequest,
    ReportContract,
    WindowMetrics,
)

ROOT = Path(__file__).parents[1]
DESTINATION = ROOT / "contracts" / "v1"
MODELS = {
    "analysis-event.schema.json": AnalysisEvent,
    "window-metrics.schema.json": WindowMetrics,
    "job-request.schema.json": JobRequest,
    "report.schema.json": ReportContract,
}


def export(destination: Path = DESTINATION) -> None:
    """Write schemas deterministically."""
    destination.mkdir(parents=True, exist_ok=True)
    for filename, model in MODELS.items():
        (destination / filename).write_text(
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    export()
