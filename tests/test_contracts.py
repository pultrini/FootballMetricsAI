from __future__ import annotations

import json
from pathlib import Path

from football_analytics.application.contracts import (
    AnalysisEvent,
    JobRequest,
    ReportContract,
    WindowMetrics,
)


def test_versioned_contracts_match_pydantic_models() -> None:
    root = Path(__file__).parents[1] / "contracts" / "v1"
    models = {
        "analysis-event.schema.json": AnalysisEvent,
        "window-metrics.schema.json": WindowMetrics,
        "job-request.schema.json": JobRequest,
        "report.schema.json": ReportContract,
    }
    for filename, model in models.items():
        schema = json.loads((root / filename).read_text(encoding="utf-8"))
        assert schema == model.model_json_schema()
