"""End-to-end processing and persistence of one football match video."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
from pydantic import BaseModel, ConfigDict, Field

from football_analytics.analytics import ExpectedGoalsModel, MatchStatisticsCalculator
from football_analytics.ball import BallPossessionEstimator, BallTracker
from football_analytics.detection import Detector, Frame, validate_frame
from football_analytics.events import EventFrame, PassDetector, ShotDetector
from football_analytics.pitch import (
    HomographyEstimationError,
    PitchHomography,
    PitchHomographyConfig,
    PitchKeypointDetector,
)
from football_analytics.storage import (
    BallPosition,
    Detection,
    EventRecord,
    ExpectedGoalPrediction,
    MatchDataExporter,
    MatchStatistics,
    PassEvent,
    PitchKeypoint,
    PlayerPosition,
    PossessionObservation,
    ShotEvent,
    TeamId,
    Track,
)
from football_analytics.teams import TeamClassifier
from football_analytics.tracking import Tracker
from football_analytics.visualization import (
    AveragePositionRenderer,
    HeatmapRenderer,
    ShotMapRenderer,
    VideoRenderer,
)

LOGGER = logging.getLogger(__name__)


class VideoAnalysisError(RuntimeError):
    """Raised when a video cannot be opened, decoded, or written."""


class VideoPipelineConfig(BaseModel):
    """Video I/O defaults independent from model inference settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fallback_fps: float = Field(default=25.0, gt=0)
    video_codec: str = Field(default="mp4v", min_length=4, max_length=4)
    minimum_homography_confidence: float = Field(default=0.4, ge=0, le=1)


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """All structured observations produced for one video."""

    fps: float
    frame_count: int
    detections: tuple[Detection, ...]
    tracks: tuple[Track, ...]
    pitch_keypoints: tuple[PitchKeypoint, ...]
    positions: tuple[PlayerPosition, ...]
    ball_positions: tuple[BallPosition, ...]
    possessions: tuple[PossessionObservation, ...]
    passes: tuple[PassEvent, ...]
    shots: tuple[ShotEvent, ...]
    xg_predictions: tuple[ExpectedGoalPrediction, ...]
    events: tuple[EventRecord, ...]
    statistics: MatchStatistics
    team_assignments: Mapping[int, TeamId]


@dataclass(frozen=True, slots=True)
class OutputManifest:
    """Paths written after an analysis run."""

    tables: Mapping[str, Mapping[str, Path]]
    visualizations: Mapping[str, Path]
    report: Path
    annotated_video: Path | None = None


class VideoAnalysisPipeline:
    """Orchestrate detector, tracking, pitch projection, events, xG, and output."""

    def __init__(
        self,
        *,
        detector: Detector,
        tracker: Tracker,
        keypoint_detector: PitchKeypointDetector,
        team_classifier: TeamClassifier,
        xg_model: ExpectedGoalsModel | None = None,
        homography: PitchHomography | None = None,
        ball_tracker: BallTracker | None = None,
        possession_estimator: BallPossessionEstimator | None = None,
        pass_detector: PassDetector | None = None,
        shot_detector: ShotDetector | None = None,
        statistics_calculator: MatchStatisticsCalculator | None = None,
        video_renderer: VideoRenderer | None = None,
        config: VideoPipelineConfig | None = None,
    ) -> None:
        from football_analytics.analytics import MetricPositionEstimator

        self.detector = detector
        self.tracker = tracker
        self.keypoint_detector = keypoint_detector
        self.team_classifier = team_classifier
        self.xg_model = xg_model
        self.config = config or VideoPipelineConfig()
        self.homography = homography or PitchHomography(
            PitchHomographyConfig(minimum_confidence=self.config.minimum_homography_confidence)
        )
        self.position_estimator = MetricPositionEstimator(self.homography)
        self.ball_tracker = ball_tracker or BallTracker()
        self.possession_estimator = possession_estimator or BallPossessionEstimator()
        self.pass_detector = pass_detector or PassDetector()
        self.shot_detector = shot_detector or ShotDetector()
        self.statistics_calculator = statistics_calculator or MatchStatisticsCalculator()
        self.video_renderer = video_renderer or VideoRenderer()

    def reset(self) -> None:
        """Clear every stateful component before another video."""
        self.tracker.reset()
        self.team_classifier.reset()
        self.homography.clear()
        self.position_estimator.reset()
        self.ball_tracker.reset()
        self.possession_estimator.reset()
        self.pass_detector.reset()
        self.shot_detector.reset()

    def process_frames(
        self,
        frames: Iterable[Frame],
        *,
        fps: float,
        maximum_frames: int | None = None,
        annotated_frame_callback: Callable[[Frame], None] | None = None,
    ) -> AnalysisResult:
        """Process an iterable of decoded frames; useful for video I/O and tests."""
        if fps <= 0:
            raise ValueError("fps must be positive")
        if maximum_frames is not None and maximum_frames < 1:
            raise ValueError("maximum_frames must be positive")
        self.reset()
        detections: list[Detection] = []
        tracks: list[Track] = []
        keypoints: list[PitchKeypoint] = []
        positions: list[PlayerPosition] = []
        balls: list[BallPosition] = []
        possessions: list[PossessionObservation] = []
        passes: list[PassEvent] = []
        shots: list[ShotEvent] = []
        predictions: list[ExpectedGoalPrediction] = []
        frame_count = 0

        for frame_id, frame in enumerate(frames):
            if maximum_frames is not None and frame_id >= maximum_frames:
                break
            validate_frame(frame)
            timestamp = frame_id / fps
            frame_detections = self.detector.detect(frame, frame_id=frame_id)
            frame_tracks = self.tracker.update(
                [item for item in frame_detections if item.class_name.casefold() != "ball"],
                frame_id=frame_id,
                frame=frame,
            )
            assignments = self.team_classifier.classify(frame, frame_tracks)
            frame_keypoints = self.keypoint_detector.detect(frame, frame_id=frame_id)
            self._update_homography(frame_keypoints)
            frame_positions = self.position_estimator.project_tracks(
                frame_tracks,
                timestamp=timestamp,
                team_assignments=assignments,
            )
            if self.homography.validate():
                ball_candidates = self.position_estimator.project_balls(
                    [item for item in frame_detections if item.class_name.casefold() == "ball"],
                    timestamp=timestamp,
                )
                ball = self.ball_tracker.update(
                    ball_candidates,
                    frame_id=frame_id,
                    timestamp=timestamp,
                )
            else:
                self.ball_tracker.reset()
                ball = None
            possession = self.possession_estimator.update(
                frame_positions,
                ball,
                frame_id=frame_id,
                timestamp=timestamp,
            )
            event_frame = EventFrame(possession=possession, ball=ball)
            frame_passes = self.pass_detector.update(event_frame)
            frame_shots = self.shot_detector.update(event_frame)
            frame_predictions = (
                [] if self.xg_model is None else self.xg_model.predict_shots(frame_shots)
            )

            detections.extend(frame_detections)
            tracks.extend(frame_tracks)
            keypoints.extend(frame_keypoints)
            positions.extend(frame_positions)
            if ball is not None:
                balls.append(ball)
            possessions.append(possession)
            passes.extend(frame_passes)
            shots.extend(frame_shots)
            predictions.extend(frame_predictions)
            frame_count += 1

            if annotated_frame_callback is not None:
                annotated_frame_callback(
                    self.video_renderer.render(
                        frame,
                        frame_tracks,
                        frame_positions,
                        ball,
                        possession,
                        assignments,
                    )
                )
            if frame_count % 100 == 0:
                LOGGER.info("Processed %d frames", frame_count)

        if frame_count == 0:
            raise VideoAnalysisError("Video contains no decodable frames")
        final_assignments = dict(self.team_classifier.assignments)
        enriched_positions = tuple(
            self._enrich_position(item, final_assignments) for item in positions
        )
        enriched_possessions = tuple(
            self._enrich_possession(item, final_assignments) for item in possessions
        )
        statistics = self.statistics_calculator.calculate(
            positions=enriched_positions,
            possessions=enriched_possessions,
            passes=passes,
            shots=shots,
            xg_predictions=predictions,
        )
        events = self._event_records(passes, shots, predictions)
        return AnalysisResult(
            fps=fps,
            frame_count=frame_count,
            detections=tuple(detections),
            tracks=tuple(tracks),
            pitch_keypoints=tuple(keypoints),
            positions=enriched_positions,
            ball_positions=tuple(balls),
            possessions=enriched_possessions,
            passes=tuple(passes),
            shots=tuple(shots),
            xg_predictions=tuple(predictions),
            events=events,
            statistics=statistics,
            team_assignments=final_assignments,
        )

    def analyze_video(
        self,
        source: str | Path,
        *,
        annotated_video: str | Path | None = None,
        maximum_frames: int | None = None,
    ) -> AnalysisResult:
        """Decode a video, stream it through the pipeline, and optionally write annotations."""
        source_path = Path(source)
        if not source_path.is_file():
            raise VideoAnalysisError(f"Input video does not exist: {source_path}")
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            capture.release()
            raise VideoAnalysisError(f"Could not open input video: {source_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = self.config.fallback_fps
        width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer: cv2.VideoWriter | None = None
        if annotated_video is not None:
            destination = Path(annotated_video)
            destination.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
            writer = cv2.VideoWriter(str(destination), fourcc, fps, (width, height))
            if not writer.isOpened():
                capture.release()
                writer.release()
                raise VideoAnalysisError(f"Could not open annotated video writer: {destination}")

        def decoded_frames() -> Iterable[Frame]:
            while True:
                ok, frame = capture.read()
                if not ok:
                    return
                yield frame

        try:
            return self.process_frames(
                decoded_frames(),
                fps=fps,
                maximum_frames=maximum_frames,
                annotated_frame_callback=None if writer is None else writer.write,
            )
        finally:
            capture.release()
            if writer is not None:
                writer.release()

    def write_outputs(
        self,
        result: AnalysisResult,
        output_directory: str | Path,
        *,
        formats: Sequence[str] = ("parquet",),
        annotated_video: str | Path | None = None,
    ) -> OutputManifest:
        """Persist tabular records, report JSON, and static match maps."""
        root = Path(output_directory)
        data_root = root / "data"
        exporter = MatchDataExporter(data_root)
        records = {
            "detections": result.detections,
            "tracks": result.tracks,
            "pitch_keypoints": result.pitch_keypoints,
            "positions": result.positions,
            "ball_positions": result.ball_positions,
            "possessions": result.possessions,
            "events": result.events,
            "passes": result.passes,
            "shots": result.shots,
            "xg_predictions": result.xg_predictions,
            "player_stats": result.statistics.players,
            "team_stats": result.statistics.teams,
        }
        tables = {
            name: exporter.export(name, values, formats=formats) for name, values in records.items()
        }
        visualization_root = root / "visualizations"
        visualization_root.mkdir(parents=True, exist_ok=True)
        heatmaps = HeatmapRenderer()
        images = {
            "pass_heatmap": heatmaps.render_passes(result.passes),
            "shot_heatmap": heatmaps.render_shots(result.shots),
            "possession_map": heatmaps.render_possession(result.positions, result.possessions),
            "shot_map": ShotMapRenderer().render(result.shots, result.xg_predictions),
            "average_positions": AveragePositionRenderer().render(result.positions),
        }
        visualizations: dict[str, Path] = {}
        for name, image in images.items():
            destination = visualization_root / f"{name}.png"
            if not cv2.imwrite(str(destination), image):
                raise VideoAnalysisError(f"Could not write visualization: {destination}")
            visualizations[name] = destination
        player_root = visualization_root / "players"
        player_root.mkdir(parents=True, exist_ok=True)
        for player in result.statistics.players:
            destination = player_root / f"player_{player.player_id}_heatmap.png"
            image = heatmaps.render_positions(result.positions, player_id=player.player_id)
            if not cv2.imwrite(str(destination), image):
                raise VideoAnalysisError(f"Could not write visualization: {destination}")
            visualizations[f"player_{player.player_id}_heatmap"] = destination
        report = root / "report.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {
                    "fps": result.fps,
                    "frame_count": result.frame_count,
                    "event_count": len(result.events),
                    "team_assignments": {
                        str(player_id): team.value
                        for player_id, team in result.team_assignments.items()
                    },
                    "statistics": result.statistics.model_dump(mode="json"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return OutputManifest(
            tables=tables,
            visualizations=visualizations,
            report=report,
            annotated_video=None if annotated_video is None else Path(annotated_video),
        )

    def _update_homography(self, keypoints: Sequence[PitchKeypoint]) -> None:
        if len(keypoints) < self.homography.config.minimum_correspondences:
            self.homography.clear()
            return
        try:
            self.homography.estimate_keypoints(keypoints)
        except HomographyEstimationError as error:
            LOGGER.debug("Homography rejected: %s", error)

    @staticmethod
    def _event_records(
        passes: Sequence[PassEvent],
        shots: Sequence[ShotEvent],
        predictions: Sequence[ExpectedGoalPrediction],
    ) -> tuple[EventRecord, ...]:
        xg_by_identity = {
            (prediction.frame_id, prediction.player_id): prediction.xg for prediction in predictions
        }
        records = [
            EventRecord(
                event_type="pass",
                frame_id=event.frame_id,
                timestamp=event.timestamp,
                team_id=event.team_id,
                player_id=event.passer_id,
                related_player_id=event.receiver_id,
                start_x=event.start_x,
                start_y=event.start_y,
                end_x=event.end_x,
                end_y=event.end_y,
                completed=event.completed,
                confidence=event.confidence,
            )
            for event in passes
        ]
        records.extend(
            EventRecord(
                event_type="shot",
                frame_id=event.frame_id,
                timestamp=event.timestamp,
                team_id=event.team_id,
                player_id=event.player_id,
                start_x=event.x,
                start_y=event.y,
                outcome=event.outcome,
                xg=xg_by_identity.get((event.frame_id, event.player_id)),
                confidence=event.confidence,
            )
            for event in shots
        )
        return tuple(sorted(records, key=lambda item: (item.timestamp, item.frame_id)))

    @staticmethod
    def _enrich_position(
        position: PlayerPosition,
        assignments: Mapping[int, TeamId],
    ) -> PlayerPosition:
        if position.team_id is not None or position.track_id not in assignments:
            return position
        return PlayerPosition.model_validate(
            {**position.model_dump(), "team_id": assignments[position.track_id]}
        )

    @staticmethod
    def _enrich_possession(
        possession: PossessionObservation,
        assignments: Mapping[int, TeamId],
    ) -> PossessionObservation:
        if possession.team_id is not None or possession.player_id not in assignments:
            return possession
        return PossessionObservation.model_validate(
            {**possession.model_dump(), "team_id": assignments[possession.player_id]}
        )
