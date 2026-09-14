"""Validated loading of historical shot datasets."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import ValidationError

from football_analytics.storage.schemas import HistoricalShot

FEATURE_COLUMNS = ("distance", "angle")
REQUIRED_COLUMNS = frozenset({"shot_x", "shot_y", "distance", "angle", "goal"})


class ShotDatasetError(ValueError):
    """Raised when a historical dataset cannot safely train an xG model."""


class ShotDataset:
    """Validated historical shots with deterministic baseline feature order."""

    def __init__(self, records: Iterable[HistoricalShot]) -> None:
        self.records = tuple(records)
        if not self.records:
            raise ShotDatasetError("Shot dataset must contain at least one row")

    @classmethod
    def load(cls, path: str | Path) -> ShotDataset:
        """Load CSV, Parquet, JSON, or newline-delimited JSON."""
        source = Path(path)
        if not source.is_file():
            raise ShotDatasetError(f"Shot dataset does not exist: {source}")
        suffix = source.suffix.casefold()
        readers = {
            ".csv": pl.read_csv,
            ".parquet": pl.read_parquet,
            ".json": pl.read_json,
            ".ndjson": pl.read_ndjson,
        }
        reader = readers.get(suffix)
        if reader is None:
            supported = ", ".join(sorted(readers))
            raise ShotDatasetError(f"Unsupported dataset format {suffix!r}; use {supported}")
        return cls.from_frame(reader(source))

    @classmethod
    def from_frame(cls, frame: pl.DataFrame) -> ShotDataset:
        """Validate required columns and every historical shot row."""
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            columns = ", ".join(sorted(missing))
            raise ShotDatasetError(f"Missing required shot columns: {columns}")
        records: list[HistoricalShot] = []
        for row_index, row in enumerate(frame.to_dicts()):
            try:
                records.append(HistoricalShot.model_validate(row))
            except ValidationError as error:
                raise ShotDatasetError(
                    f"Invalid historical shot at row {row_index}: {error}"
                ) from error
        return cls(records)

    def features(self) -> npt.NDArray[np.float64]:
        """Return the baseline feature matrix ordered as distance, angle."""
        return np.asarray(
            [[record.distance, record.angle] for record in self.records],
            dtype=np.float64,
        )

    def targets(self) -> npt.NDArray[np.int64]:
        """Return binary goal labels."""
        return np.asarray([record.goal for record in self.records], dtype=np.int64)

    def to_frame(self) -> pl.DataFrame:
        """Return validated records as a Polars DataFrame."""
        return pl.DataFrame([record.model_dump() for record in self.records])
