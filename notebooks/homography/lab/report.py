"""Small reporting helpers shared by notebook cells."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a deterministic UTF-8 JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prerequisite_report(config: Any) -> dict[str, Any]:
    """Build a standard report when external analysis inputs are unavailable."""
    missing = config.missing()
    return {
        "status": "ready" if not missing else "blocked",
        "missing_prerequisites": missing,
        "message": (
            "Todos os artefatos necessários estão disponíveis."
            if not missing
            else "Configure os artefatos externos antes da análise real."
        ),
    }
