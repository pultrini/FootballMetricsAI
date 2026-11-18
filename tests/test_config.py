from __future__ import annotations

from football_analytics.adapters.inbound.cli import build_parser
from football_analytics.application.config import SessionConfig


def test_session_config_defaults_match_current_cli() -> None:
    config = SessionConfig()

    assert config.object_model.as_posix() == "models/player_tracker.onnx"
    assert config.pitch_model.as_posix() == "models/football_field.onnx"
    assert config.device is None
    assert config.max_frames is None
    assert config.export_formats == ("parquet",)
    assert config.team_a_attacks_right is True
    assert config.fallback_fps == 25.0
    assert config.video_codec == "mp4v"


def test_session_config_loads_yaml_and_explicit_cli_values_override(tmp_path) -> None:
    path = tmp_path / "session.yaml"
    path.write_text(
        "\n".join(
            (
                "device: cpu",
                "max_frames: 50",
                "team_a_attacks_right: false",
                "export_formats: [json]",
                "detector_confidence_threshold: 0.4",
            )
        ),
        encoding="utf-8",
    )

    config = SessionConfig.from_yaml(path).with_overrides({"max_frames": 100, "device": "mps"})

    assert config.max_frames == 100
    assert config.device == "mps"
    assert config.team_a_attacks_right is False
    assert config.export_formats == ("json",)
    assert config.detector_confidence_threshold == 0.4


def test_cli_tracks_only_explicit_values_for_yaml_overrides() -> None:
    args = build_parser().parse_args(
        ["analyze", "match.mp4", "--max-frames", "20", "--no-team-a-attacks-right"]
    )

    assert args.max_frames == 20
    assert args.team_a_attacks_right is False
    assert args._explicit_options == {"max_frames", "team_a_attacks_right"}
