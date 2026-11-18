"""Command-line entry point for football-analytics."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from football_analytics import __version__
from football_analytics.application.errors import AnalysisError, AnalysisErrorCode

ROBOFLOW_FOOTBALL_CLASS_NAME_MAP = {
    "0": "ball",
    "1": "goalkeeper",
    "2": "player",
    "3": "referee",
}


class _TrackOption(argparse.Action):
    """Record options explicitly supplied by the caller."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        setattr(namespace, self.dest, values)
        explicit = getattr(namespace, "_explicit_options", set())
        explicit.add(self.dest)
        namespace._explicit_options = explicit


class _TrackStoreTrue(argparse.Action):
    """Record explicitly supplied boolean switches."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        setattr(namespace, self.dest, True)
        explicit = getattr(namespace, "_explicit_options", set())
        explicit.add(self.dest)
        namespace._explicit_options = explicit


class _TrackBooleanOptional(argparse.BooleanOptionalAction):
    """Track explicitly supplied boolean optional values."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        super().__call__(parser, namespace, values, option_string)
        explicit = getattr(namespace, "_explicit_options", set())
        explicit.add(self.dest)
        namespace._explicit_options = explicit


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
    from football_analytics.domain.analytics import ExpectedGoalsModel
    from football_analytics.domain.analytics.shot_dataset import ShotDataset

    dataset = ShotDataset.load(args.dataset)
    destination = ExpectedGoalsModel().fit_dataset(dataset).save(args.output)
    print(f"Trained xG model on {len(dataset.records)} shots: {destination}")
    return 0


def _analyze(args: argparse.Namespace) -> int:
    """Parse analysis arguments and invoke the application use case."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    from football_analytics.bootstrap import (
        build_analysis_session,
        build_progress,
        build_sink,
        build_source,
    )

    session = build_analysis_session(args)
    sink = build_sink(args)
    source = build_source(args)
    result = session.run(source, sink, build_progress(args))
    if sink.manifest is None:
        raise AnalysisError(
            AnalysisErrorCode.UNKNOWN,
            "Analysis completed without an output manifest",
        )
    manifest = sink.manifest
    print(
        f"Analysed {result.frame_count} frames at {result.fps:.3f} FPS; report: {manifest.report}"
    )
    if args.xg_model is None:
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
    analyze.add_argument("--config", type=Path, help="optional YAML session configuration")
    analyze.add_argument(
        "--progress",
        choices=("none", "jsonl"),
        default="none",
        help="progress output format (default: none)",
    )
    analyze.add_argument(
        "--team-a-attacks-right",
        action=_TrackBooleanOptional,
        default=True,
        help="set Team A attacking direction (default: right)",
    )
    analyze.add_argument("video", type=Path, help="input match video")
    analyze.add_argument(
        "--object-model",
        type=Path,
        default=Path("models/player_tracker.onnx"),
        action=_TrackOption,
        help="football object detector weights (default: models/player_tracker.onnx)",
    )
    analyze.add_argument(
        "--pitch-model",
        type=Path,
        default=Path("models/football_field.onnx"),
        action=_TrackOption,
        help="pitch keypoint pose weights (default: models/football_field.onnx)",
    )
    analyze.add_argument(
        "--xg-model",
        type=Path,
        help="trusted trained xG artifact; omit to skip xG inference",
        action=_TrackOption,
    )
    analyze.add_argument("--output", "-o", type=Path, help="output directory")
    analyze.add_argument(
        "--device", help="Ultralytics device such as cpu, mps, or 0", action=_TrackOption
    )
    analyze.add_argument(
        "--team-a-color",
        type=_parse_rgb,
        metavar="R,G,B",
        action=_TrackOption,
        help="optional Team A jersey colour anchor",
    )
    analyze.add_argument(
        "--team-b-color",
        type=_parse_rgb,
        metavar="R,G,B",
        action=_TrackOption,
        help="optional Team B jersey colour anchor",
    )
    analyze.add_argument(
        "--export-format",
        choices=("parquet", "csv", "json"),
        nargs="+",
        default=("parquet",),
        action=_TrackOption,
        help="one or more table formats (default: parquet)",
    )
    analyze.add_argument(
        "--max-frames",
        type=int,
        help="process only the first N frames for a smoke test",
        dest="max_frames",
        action=_TrackOption,
    )
    analyze.add_argument(
        "--no-annotated-video",
        action=_TrackStoreTrue,
        nargs=0,
        const=True,
        help="skip annotated MP4 generation",
    )
    analyze.add_argument("--verbose", action="store_true", help="enable debug logging")
    analyze.set_defaults(handler=_analyze)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    try:
        args = build_parser().parse_args(argv)
        handler = getattr(args, "handler", None)
        return handler(args) if handler is not None else 0
    except AnalysisError as error:
        logging.error("%s: %s", error.code.value, error.message)
        return {
            AnalysisErrorCode.VIDEO_UNREADABLE: 2,
            AnalysisErrorCode.MODEL_NOT_FOUND: 3,
            AnalysisErrorCode.CONFIG_INVALID: 4,
            AnalysisErrorCode.INFERENCE_FAILED: 5,
            AnalysisErrorCode.HOMOGRAPHY_UNAVAILABLE: 6,
            AnalysisErrorCode.UNKNOWN: 1,
        }[error.code]
