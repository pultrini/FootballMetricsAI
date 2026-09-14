import polars as pl

from football_analytics.analytics import ExpectedGoalsModel
from football_analytics.cli import ROBOFLOW_FOOTBALL_CLASS_NAME_MAP, build_parser, main


def test_cli_accepts_no_command_for_milestone_one() -> None:
    assert main([]) == 0


def test_cli_trains_xg_model(tmp_path, capsys) -> None:
    source = tmp_path / "shots.parquet"
    output = tmp_path / "models" / "xg.joblib"
    pl.DataFrame(
        {
            "shot_x": [96.0, 93.0, 78.0, 69.0],
            "shot_y": [34.0, 31.0, 20.0, 50.0],
            "distance": [9.0, 12.0, 30.0, 40.0],
            "angle": [0.9, 0.7, 0.2, 0.1],
            "goal": [True, True, False, False],
        }
    ).write_parquet(source)

    result = main(["train-xg", str(source), "--output", str(output)])

    assert result == 0
    assert output.is_file()
    assert "Trained xG model on 4 shots" in capsys.readouterr().out
    assert 0 <= ExpectedGoalsModel.load(output).predict_proba([[15.0, 0.5]])[0] <= 1


def test_analyze_cli_parses_models_colours_and_smoke_limit() -> None:
    args = build_parser().parse_args(
        [
            "analyze",
            "match.mp4",
            "--object-model",
            "objects.pt",
            "--pitch-model",
            "pitch.pt",
            "--team-a-color",
            "220,30,30",
            "--team-b-color",
            "30,30,220",
            "--max-frames",
            "20",
            "--no-annotated-video",
        ]
    )

    assert args.video.name == "match.mp4"
    assert args.team_a_color == (220, 30, 30)
    assert args.team_b_color == (30, 30, 220)
    assert args.max_frames == 20


def test_analyze_cli_defaults_to_numeric_roboflow_detector() -> None:
    args = build_parser().parse_args(["analyze", "match.mp4"])

    assert args.object_model.as_posix() == "models/player_tracker.onnx"
    assert ROBOFLOW_FOOTBALL_CLASS_NAME_MAP == {
        "0": "ball",
        "1": "goalkeeper",
        "2": "player",
        "3": "referee",
    }
