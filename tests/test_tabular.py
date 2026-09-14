import polars as pl
import pytest

from football_analytics.storage import (
    MatchDataExporter,
    PlayerStatistics,
    TabularExportError,
)


def player_statistics() -> PlayerStatistics:
    return PlayerStatistics(
        player_id=9,
        team_id="TEAM_A",
        tracked_seconds=90.0,
        tracked_minutes=1.5,
        distance_m=230.0,
        average_speed_mps=2.55,
        maximum_speed_mps=7.2,
        possession_seconds=12.0,
        passes=4,
        completed_passes=3,
        pass_accuracy=75.0,
        shots=1,
        shots_on_target=1,
        goals=0,
        xg=0.32,
        xg_per_shot=0.32,
        average_position_x=67.0,
        average_position_y=31.0,
        progressive_distance_with_ball_m=18.0,
    )


def test_match_data_exports_csv_json_and_parquet(tmp_path) -> None:
    outputs = MatchDataExporter(tmp_path).export(
        "player_stats",
        [player_statistics()],
        formats=("csv", "json", "parquet"),
    )

    assert set(outputs) == {"csv", "json", "parquet"}
    assert pl.read_csv(outputs["csv"])["player_id"].to_list() == [9]
    assert pl.read_json(outputs["json"])["xg"].to_list() == [0.32]
    assert pl.read_parquet(outputs["parquet"])["team_id"].to_list() == ["TEAM_A"]


def test_export_rejects_unsafe_names(tmp_path) -> None:
    with pytest.raises(TabularExportError, match="name"):
        MatchDataExporter(tmp_path).export("../player_stats", [player_statistics()])


def test_export_rejects_unsupported_format(tmp_path) -> None:
    with pytest.raises(TabularExportError, match="Unsupported"):
        MatchDataExporter(tmp_path).export("player_stats", [player_statistics()], formats=("xlsx",))
