"""Composition root for the complete-video analysis use case."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from football_analytics.adapters.outbound.analysis_sink import ParquetResultSink
from football_analytics.adapters.outbound.bytetrack_tracker import ByteTrackConfig, ByteTrackTracker
from football_analytics.adapters.outbound.jsonl_progress import JsonLinesProgressReporter
from football_analytics.adapters.outbound.noop_progress import NoOpProgressReporter
from football_analytics.adapters.outbound.opencv_frame_source import OpenCvFrameSource
from football_analytics.adapters.outbound.team_classifier import (
    JerseyTeamClassifier,
    TeamClassifierConfig,
)
from football_analytics.adapters.outbound.yolo_detector import YoloDetector, YoloDetectorConfig
from football_analytics.adapters.outbound.yolo_pitch_keypoints import (
    YoloPitchKeypointConfig,
    YoloPitchKeypointDetector,
)
from football_analytics.application.config import SessionConfig
from football_analytics.application.errors import AnalysisError, AnalysisErrorCode
from football_analytics.application.session import AnalysisSession
from football_analytics.domain.analytics import ExpectedGoalsModel
from football_analytics.domain.analytics.positions import PositionEstimatorConfig
from football_analytics.domain.analytics.statistics import (
    MatchStatisticsCalculator,
    StatisticsConfig,
)
from football_analytics.domain.ball.possession import BallPossessionEstimator, PossessionConfig
from football_analytics.domain.events.shot_detector import ShotDetector, ShotDetectorConfig
from football_analytics.domain.pitch.pitch_model import GoalSide
from football_analytics.pipeline import VideoAnalysisPipeline
from football_analytics.pipeline.video import VideoPipelineConfig

ROBOFLOW_FOOTBALL_CLASS_NAME_MAP = {
    "0": "ball",
    "1": "goalkeeper",
    "2": "player",
    "3": "referee",
}


def build_analysis_session(args: Namespace) -> AnalysisSession:
    """Build the use case and concrete inference adapters from CLI arguments."""
    config = SessionConfig.from_yaml(args.config) if args.config is not None else SessionConfig()
    config = config.with_overrides(
        {
            ("export_formats" if key == "export_format" else key): getattr(args, key)
            for key in getattr(args, "_explicit_options", set())
            if hasattr(args, key)
        }
    )
    args._session_config = config
    for model_name, model_path in (
        ("object", config.object_model),
        ("pitch-keypoint", config.pitch_model),
    ):
        if not model_path.is_file():
            raise AnalysisError(
                AnalysisErrorCode.MODEL_NOT_FOUND,
                f"{model_name} model does not exist: {model_path}",
            )

    from cv2 import CAP_PROP_FPS, VideoCapture

    probe = VideoCapture(str(args.video))
    fps = float(probe.get(CAP_PROP_FPS)) if probe.isOpened() else 0.0
    probe.release()
    tracker_fps = fps if fps > 0 else 25.0
    try:
        xg_model = None if config.xg_model is None else ExpectedGoalsModel.load(config.xg_model)
    except (OSError, ValueError) as error:
        raise AnalysisError(
            AnalysisErrorCode.MODEL_NOT_FOUND,
            "Could not load the configured xG model",
        ) from error
    pipeline = VideoAnalysisPipeline(
        detector=YoloDetector(
            str(config.object_model),
            YoloDetectorConfig(
                device=config.device,
                confidence_threshold=config.detector_confidence_threshold,
                iou_threshold=config.detector_iou_threshold,
                allowed_classes=frozenset({"player", "goalkeeper", "referee", "ball"}),
                class_name_map=ROBOFLOW_FOOTBALL_CLASS_NAME_MAP,
            ),
        ),
        tracker=ByteTrackTracker(
            ByteTrackConfig(
                frame_rate=tracker_fps,
                lost_track_buffer=config.tracker_lost_track_buffer,
                track_activation_threshold=config.tracker_activation_threshold,
                minimum_consecutive_frames=config.tracker_minimum_consecutive_frames,
                minimum_iou_threshold=config.tracker_minimum_iou_threshold,
                high_conf_det_threshold=config.tracker_high_conf_det_threshold,
            )
        ),
        keypoint_detector=YoloPitchKeypointDetector(
            str(config.pitch_model),
            YoloPitchKeypointConfig(
                device=config.device,
                detection_confidence_threshold=config.keypoint_detection_confidence_threshold,
                keypoint_confidence_threshold=config.keypoint_confidence_threshold,
            ),
        ),
        team_classifier=JerseyTeamClassifier(
            TeamClassifierConfig(
                team_a_rgb=config.team_a_color,
                team_b_rgb=config.team_b_color,
            )
        ),
        xg_model=xg_model,
        position_config=PositionEstimatorConfig(
            maximum_velocity_gap_seconds=config.maximum_velocity_gap_seconds
        ),
        possession_estimator=BallPossessionEstimator(
            PossessionConfig(
                acquisition_distance_m=config.possession_acquisition_distance_m,
                release_distance_m=config.possession_release_distance_m,
                acquisition_frames=config.possession_acquisition_frames,
                switch_frames=config.possession_switch_frames,
                release_frames=config.possession_release_frames,
                reference_fps=config.possession_reference_fps,
                adapt_temporal_windows_to_fps=config.possession_adapt_temporal_windows_to_fps,
                minimum_ball_confidence=config.possession_minimum_ball_confidence,
            )
        ),
        shot_detector=ShotDetector(
            ShotDetectorConfig(
                minimum_ball_speed_mps=config.shot_minimum_ball_speed_mps,
                minimum_ball_acceleration_mps2=config.shot_minimum_ball_acceleration_mps2,
                minimum_separation_m=config.shot_minimum_separation_m,
                minimum_goal_direction_cosine=config.shot_minimum_goal_direction_cosine,
                minimum_control_confidence=config.shot_minimum_control_confidence,
                cooldown_frames=config.shot_cooldown_frames,
                require_observed_ball=config.shot_require_observed_ball,
                team_a_goal_side=(GoalSide.RIGHT if config.team_a_attacks_right else GoalSide.LEFT),
            )
        ),
        statistics_calculator=MatchStatisticsCalculator(
            StatisticsConfig(
                maximum_tracking_gap_seconds=config.statistics_maximum_tracking_gap_seconds,
                progressive_pass_distance_m=config.statistics_progressive_pass_distance_m,
                team_a_attacks_right=config.team_a_attacks_right,
            )
        ),
        config=VideoPipelineConfig(
            fallback_fps=config.fallback_fps,
            video_codec=config.video_codec,
            minimum_homography_confidence=config.minimum_homography_confidence,
        ),
    )
    return AnalysisSession(pipeline, config)


def build_source(args: Namespace) -> OpenCvFrameSource:
    """Build the video input adapter."""
    config = getattr(args, "_session_config", SessionConfig())
    output = args.output or Path("outputs") / args.video.stem
    annotated_video = None if config.no_annotated_video else output / "annotated.mp4"
    return OpenCvFrameSource(args.video, annotated_video=annotated_video)


def build_sink(args: Namespace) -> ParquetResultSink:
    """Build the tabular and visualization output adapter."""
    config = getattr(args, "_session_config", SessionConfig())
    output = args.output or Path("outputs") / args.video.stem
    annotated_video = None if config.no_annotated_video else output / "annotated.mp4"
    return ParquetResultSink(
        output,
        formats=config.export_formats,
        annotated_video=annotated_video,
    )


def build_progress(args: Namespace) -> NoOpProgressReporter | JsonLinesProgressReporter:
    """Build the current CLI progress adapter."""
    if args.progress == "jsonl":
        return JsonLinesProgressReporter()
    return NoOpProgressReporter()
