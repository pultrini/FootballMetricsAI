"""Golden-output characterization test for the current analyze command."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).parents[2]
GOLDEN_ROOT = Path(__file__).parent / "golden"
TABLE_NAMES = (
    "positions",
    "possessions",
    "passes",
    "shots",
    "player_stats",
    "team_stats",
)
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _required_environment() -> dict[str, Path]:
    names = ("FA_TEST_CLIP", "FA_TEST_OBJECT_MODEL", "FA_TEST_PITCH_MODEL")
    values = {name: os.environ.get(name) for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        pytest.skip(f"characterization inputs not configured: {', '.join(missing)}")
    paths = {name: Path(value) for name, value in values.items() if value is not None}
    missing_files = [name for name, path in paths.items() if not path.is_file()]
    if missing_files:
        pytest.skip(f"characterization files not found: {', '.join(missing_files)}")
    return paths


def _normalise_report(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            name: _normalise_report(item, name)
            for name, item in value.items()
            if name.casefold() not in {"timestamp", "executed_at", "execution_time"}
        }
    if isinstance(value, list):
        return [_normalise_report(item, key) for item in value]
    if isinstance(value, str):
        if key is not None and ("path" in key.casefold() or "timestamp" in key.casefold()):
            return "<volatile>"
        if UUID_PATTERN.fullmatch(value):
            return "<volatile>"
    return value


def _assert_values_equal(expected: Any, actual: Any, location: str) -> None:
    if isinstance(expected, float) and isinstance(actual, (float, int)):
        assert math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-12), location
        return
    assert type(expected) is type(actual), location
    if isinstance(expected, dict):
        assert list(expected) == list(actual), location
        for key in expected:
            _assert_values_equal(actual[key], expected[key], f"{location}.{key}")
        return
    if isinstance(expected, list):
        assert len(expected) == len(actual), location
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            _assert_values_equal(expected_item, actual_item, f"{location}[{index}]")
        return
    assert expected == actual, location


def _assert_table_equal(expected_path: Path, actual_path: Path) -> None:
    expected = pl.read_parquet(expected_path)
    actual = pl.read_parquet(actual_path)
    assert expected.columns == actual.columns
    assert expected.schema == actual.schema
    assert expected.shape == actual.shape
    for row_index, (expected_row, actual_row) in enumerate(
        zip(expected.iter_rows(), actual.iter_rows(), strict=True)
    ):
        for column, expected_value, actual_value in zip(
            expected.columns, expected_row, actual_row, strict=True
        ):
            _assert_values_equal(expected_value, actual_value, f"{column}[{row_index}]")


@pytest.mark.slow
def test_analyze_matches_golden_outputs(tmp_path: Path) -> None:
    paths = _required_environment()
    expected_files = [GOLDEN_ROOT / "report.json"] + [
        GOLDEN_ROOT / f"{name}.parquet" for name in TABLE_NAMES
    ]
    missing_golden = [path.name for path in expected_files if not path.is_file()]
    if missing_golden:
        pytest.skip(f"golden outputs not generated: {', '.join(missing_golden)}")

    output = tmp_path / "output"
    command = [
        "football-analytics",
        "analyze",
        str(paths["FA_TEST_CLIP"]),
        "--object-model",
        str(paths["FA_TEST_OBJECT_MODEL"]),
        "--pitch-model",
        str(paths["FA_TEST_PITCH_MODEL"]),
        "--output",
        str(output),
        "--max-frames",
        "100",
        "--no-annotated-video",
    ]
    xg_model = os.environ.get("FA_TEST_XG_MODEL")
    if xg_model:
        command.extend(["--xg-model", xg_model])
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    expected_report = json.loads((GOLDEN_ROOT / "report.json").read_text(encoding="utf-8"))
    actual_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    _assert_values_equal(
        _normalise_report(expected_report),
        _normalise_report(actual_report),
        "report",
    )
    for name in TABLE_NAMES:
        _assert_table_equal(
            GOLDEN_ROOT / f"{name}.parquet",
            output / "data" / f"{name}.parquet",
        )
