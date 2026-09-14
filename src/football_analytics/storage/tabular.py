"""Tabular persistence for serializable analytics records."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl
from pydantic import BaseModel

SUPPORTED_FORMATS = frozenset({"csv", "json", "parquet"})


class TabularExportError(ValueError):
    """Raised when a tabular destination or format is invalid."""


def records_to_frame(records: Sequence[BaseModel]) -> pl.DataFrame:
    """Convert immutable records to a flat-compatible Polars DataFrame."""
    return pl.DataFrame([record.model_dump(mode="json") for record in records])


class TabularExporter:
    """Write records to CSV, JSON, or Parquet based on the file suffix."""

    @staticmethod
    def write(records: Sequence[BaseModel], path: str | Path) -> Path:
        """Write records and return the resolved destination path."""
        destination = Path(path)
        format_name = destination.suffix.casefold().lstrip(".")
        if format_name not in SUPPORTED_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_FORMATS))
            raise TabularExportError(f"Unsupported export format {format_name!r}; use {supported}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame = records_to_frame(records)
        if format_name == "parquet":
            frame.write_parquet(destination)
        elif format_name == "csv":
            frame.write_csv(destination)
        else:
            frame.write_json(destination)
        return destination


class MatchDataExporter:
    """Export named match tables in one or more persistent formats."""

    def __init__(self, output_directory: str | Path) -> None:
        self.output_directory = Path(output_directory)

    def export(
        self,
        name: str,
        records: Sequence[BaseModel],
        *,
        formats: Sequence[str] = ("parquet",),
    ) -> dict[str, Path]:
        """Export one logical table, rejecting unsafe or ambiguous names."""
        if not name or Path(name).name != name or Path(name).suffix:
            raise TabularExportError("Table name must be a plain name without a suffix")
        normalized_formats = tuple(item.casefold().lstrip(".") for item in formats)
        if not normalized_formats or len(set(normalized_formats)) != len(normalized_formats):
            raise TabularExportError("Export formats must be non-empty and unique")
        return {
            format_name: TabularExporter.write(
                records, self.output_directory / f"{name}.{format_name}"
            )
            for format_name in normalized_formats
        }
