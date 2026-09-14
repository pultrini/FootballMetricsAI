import polars as pl
import pytest

from football_analytics.storage import ShotDataset, ShotDatasetError


def historical_shots() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "shot_x": [95.0, 80.0, 90.0, 72.0],
            "shot_y": [34.0, 20.0, 40.0, 50.0],
            "distance": [10.0, 28.0, 16.0, 38.0],
            "angle": [0.70, 0.20, 0.45, 0.10],
            "goal": [True, False, True, False],
            "body_part": ["right_foot", None, "left_foot", None],
            "source_match": ["A", "A", "B", "B"],
        }
    )


def test_dataset_has_deterministic_baseline_features() -> None:
    dataset = ShotDataset.from_frame(historical_shots())

    assert dataset.features().tolist() == [
        [10.0, 0.7],
        [28.0, 0.2],
        [16.0, 0.45],
        [38.0, 0.1],
    ]
    assert dataset.targets().tolist() == [1, 0, 1, 0]
    assert "source_match" not in dataset.to_frame().columns


def test_dataset_loads_parquet(tmp_path) -> None:
    source = tmp_path / "shots.parquet"
    historical_shots().write_parquet(source)

    dataset = ShotDataset.load(source)

    assert len(dataset.records) == 4


def test_dataset_rejects_missing_required_columns() -> None:
    with pytest.raises(ShotDatasetError, match="shot_y"):
        ShotDataset.from_frame(historical_shots().drop("shot_y"))


def test_dataset_rejects_invalid_rows() -> None:
    invalid = historical_shots().with_columns(pl.lit(4.0).alias("angle"))

    with pytest.raises(ShotDatasetError, match="row 0"):
        ShotDataset.from_frame(invalid)


def test_dataset_rejects_unknown_file_type(tmp_path) -> None:
    source = tmp_path / "shots.txt"
    source.write_text("not a supported dataset")

    with pytest.raises(ShotDatasetError, match="Unsupported dataset format"):
        ShotDataset.load(source)
