"""Command-line entry point for football-analytics."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from football_analytics import __version__

ROBOFLOW_FOOTBALL_CLASS_NAME_MAP = {
    "0": "ball",
    "1": "goalkeeper",
    "2": "player",
    "3": "referee",
}


def _parse_rgb(value: str) -> tuple[int, int, int]:
    """Parse a CLI RGB triplet such as 220,30,30."""
    try:
        channels = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("colour must use R,G,B integers") from error
    if len(channels) != 3 or any(channel < 0 or channel > 255 for channel in channels):
        raise argparse.ArgumentTypeError("colour must contain three values from 0 to 255")
    return channels


def _train_xg(args: argparse.Namespace) -> int:
    """Train and persist an xG model from historical shots."""
    from football_analytics.analytics import ExpectedGoalsModel
    from football_analytics.storage import ShotDataset

    dataset = ShotDataset.load(args.dataset)
    destination = ExpectedGoalsModel().fit_dataset(dataset).save(args.output)
    print(f"Trained xG model on {len(dataset.records)} shots: {destination}")
    return 0


def _analyze(args: argparse.Namespace) -> int:
    """Construct and execute the complete video-analysis pipeline."""
    import cv2

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from football_analytics.analytics import ExpectedGoalsModel
    from football_analytics.detection import YoloDetector, YoloDetectorConfig
    from football_analytics.pipeline import VideoAnalysisError, VideoAnalysisPipeline
    from football_analytics.pitch import YoloPitchKeypointConfig, YoloPitchKeypointDetector
    from football_analytics.teams import JerseyTeamClassifier, TeamClassifierConfig
    from football_analytics.tracking import ByteTrackConfig, ByteTrackTracker

    for model_name, model_path in (
        ("object", args.object_model),
        ("pitch-keypoint", args.pitch_model),
    ):
        if not model_path.is_file():
            raise VideoAnalysisError(f"{model_name} model does not exist: {model_path}")
    if not args.video.is_file():
        raise VideoAnalysisError(f"Input video does not exist: {args.video}")

    probe = cv2.VideoCapture(str(args.video))
    fps = float(probe.get(cv2.CAP_PROP_FPS)) if probe.isOpened() else 0.0
    probe.release()
    tracker_fps = fps if fps > 0 else 25.0
    object_config = YoloDetectorConfig(
        device=args.device,
        allowed_classes=frozenset({"player", "goalkeeper", "referee", "ball"}),
        class_name_map=ROBOFLOW_FOOTBALL_CLASS_NAME_MAP,
    )
    keypoint_config = YoloPitchKeypointConfig(device=args.device)
    team_config = TeamClassifierConfig(
        team_a_rgb=args.team_a_color,
        team_b_rgb=args.team_b_color,
    )
    xg_model = None
    if args.xg_model is not None:
        xg_model = ExpectedGoalsModel.load(args.xg_model)
    pipeline = VideoAnalysisPipeline(
        detector=YoloDetector(str(args.object_model), object_config),
        tracker=ByteTrackTracker(ByteTrackConfig(frame_rate=tracker_fps)),
        keypoint_detector=YoloPitchKeypointDetector(str(args.pitch_model), keypoint_config),
        team_classifier=JerseyTeamClassifier(team_config),
        xg_model=xg_model,
    )
    output = args.output or Path("outputs") / args.video.stem
    annotated_video = None if args.no_annotated_video else output / "annotated.mp4"
    result = pipeline.analyze_video(
        args.video,
        annotated_video=annotated_video,
        maximum_frames=args.max_frames,
    )
    manifest = pipeline.write_outputs(
        result,
        output,
        formats=args.export_format,
        annotated_video=annotated_video,
    )
    print(
        f"Analysed {result.frame_count} frames at {result.fps:.3f} FPS; report: {manifest.report}"
    )
    if xg_model is None:
        print("xG was skipped; pass --xg-model with a trusted trained artifact to enable it")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without causing side effects."""
    parser = argparse.ArgumentParser(
        prog="football-analytics",
        description="Modular computer-vision and football analytics pipeline.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    train_xg = subparsers.add_parser(
        "train-xg",
        help="train the baseline xG model from historical shots",
    )
    train_xg.add_argument(
        "dataset",
        type=Path,
        help="CSV, Parquet, JSON, or NDJSON historical shot dataset",
    )
    train_xg.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("models/xg.joblib"),
        help="trusted model artifact destination (default: models/xg.joblib)",
    )
    train_xg.set_defaults(handler=_train_xg)
    analyze = subparsers.add_parser(
        "analyze",
        help="run the end-to-end analytics pipeline on a match video",
    )
    analyze.add_argument("video", type=Path, help="input match video")
    analyze.add_argument(
        "--object-model",
        type=Path,
        default=Path("models/player_tracker.onnx"),
        help="football object detector weights (default: models/player_tracker.onnx)",
    )
    analyze.add_argument(
        "--pitch-model",
        type=Path,
        default=Path("models/football_field.onnx"),
        help="pitch keypoint pose weights (default: models/football_field.onnx)",
    )
    analyze.add_argument(
        "--xg-model",
        type=Path,
        help="trusted trained xG artifact; omit to skip xG inference",
    )
    analyze.add_argument("--output", "-o", type=Path, help="output directory")
    analyze.add_argument("--device", help="Ultralytics device such as cpu, mps, or 0")
    analyze.add_argument(
        "--team-a-color",
        type=_parse_rgb,
        metavar="R,G,B",
        help="optional Team A jersey colour anchor",
    )
    analyze.add_argument(
        "--team-b-color",
        type=_parse_rgb,
        metavar="R,G,B",
        help="optional Team B jersey colour anchor",
    )
    analyze.add_argument(
        "--export-format",
        choices=("parquet", "csv", "json"),
        nargs="+",
        default=("parquet",),
        help="one or more table formats (default: parquet)",
    )
    analyze.add_argument(
        "--max-frames",
        type=int,
        help="process only the first N frames for a smoke test",
    )
    analyze.add_argument(
        "--no-annotated-video",
        action="store_true",
        help="skip annotated MP4 generation",
    )
    analyze.add_argument("--verbose", action="store_true", help="enable debug logging")
    analyze.set_defaults(handler=_analyze)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    args = build_parser().parse_args(argv)
    handler = getattr(args, "handler", None)
    return handler(args) if handler is not None else 0
