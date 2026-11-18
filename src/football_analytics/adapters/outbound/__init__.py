"""Outbound adapters."""

from football_analytics.adapters.outbound.parquet_sink import (
    MatchDataExporter,
    TabularExporter,
    TabularExportError,
    records_to_frame,
)

__all__ = ["MatchDataExporter", "TabularExportError", "TabularExporter", "records_to_frame"]
