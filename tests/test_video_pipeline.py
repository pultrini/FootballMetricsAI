import json
from collections.abc import Mapping, Sequence

import cv2
import numpy as np
import pytest

from football_analytics.detection import Detector, Frame
from football_analytics.pipeline import VideoAnalysisPipeline, VideoPipelineConfig
from football_analytics.pitch import PitchKeypointDetector
from football_analytics.storage import Detection, PitchKeypoint, TeamId, Track
from football_analytics.teams import TeamClassifier
from football_analytics.tracking import Tracker


class FakeDetector(Detector):
    def detect(self, frame: Frame, *, frame_id: int = 0) -> list[Detection]:
        return [
            Detection(
                frame_id=frame_id,
                class_id=0,
                class_name="player",
                confidence=0.95,
                bbox=(10.0, 10.0, 20.0, 30.0),
            ),
            Detection(
                frame_id=frame_id,
                class_id=1,
                class_name="ball",
                confidence=0.9,
                bbox=(14.0, 29.0, 16.0, 31.0),
            ),
        ]


class FakeTracker(Tracker):
    def update(
        self,
        detections: Sequence[Detection],
        *,
        frame_id: int,
        frame: Frame | None = None,
    ) -> list[Track]:
        assert len(detections) == 1
        return [
            Track(
                frame_id=frame_id,
                track_id=9,
                class_name="player",
                bbox=detections[0].bbox,
                pixel_position=(15.0, 30.0),
                confidence=detections[0].confidence,
            )
        ]

    def reset(self) -> None:
        pass


class FragmentingTracker(FakeTracker):
    def update(
        self,
        detections: Sequence[Detection],
        *,
        frame_id: int,
        frame: Frame | None = None,
    ) -> list[Track]:
        tracks = super().update(detections, frame_id=frame_id, frame=frame)
        raw_id = 9 if frame_id == 0 else 99
        return [Track.model_validate({**tracks[0].model_dump(), "track_id": raw_id})]


class FakeKeypointDetector(PitchKeypointDetector):
    def detect(self, frame: Frame, *, frame_id: int = 0) -> list[PitchKeypoint]:
        image_points = ((0.0, 0.0), (119.0, 0.0), (119.0, 79.0), (0.0, 79.0))
        pitch_points = ((0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0))
        return [
            PitchKeypoint(
                frame_id=frame_id,
                id=str(index),
                name=f"corner_{index}",
                image_x=image[0],
                image_y=image[1],
                pitch_x=pitch[0],
                pitch_y=pitch[1],
                confidence=1.0,
            )
            for index, (image, pitch) in enumerate(zip(image_points, pitch_points, strict=True))
        ]


class FakeTeamClassifier(TeamClassifier):
    def __init__(self) -> None:
        self._assignments: dict[int, TeamId] = {}

    @property
    def assignments(self) -> Mapping[int, TeamId]:
        return dict(self._assignments)

    def classify(self, frame: Frame, tracks: Sequence[Track]) -> Mapping[int, TeamId]:
        self._assignments.update({track.track_id: TeamId.TEAM_A for track in tracks})
        return self.assignments

    def reset(self) -> None:
        self._assignments.clear()


def pipeline(**kwargs) -> VideoAnalysisPipeline:
    return VideoAnalysisPipeline(
        detector=FakeDetector(),
        tracker=FakeTracker(),
        keypoint_detector=FakeKeypointDetector(),
        team_classifier=FakeTeamClassifier(),
        **kwargs,
    )


def frames(count: int = 3) -> list[Frame]:
    return [np.zeros((80, 120, 3), dtype=np.uint8) for _ in range(count)]


def test_pipeline_connects_every_stage_and_writes_outputs(tmp_path) -> None:
    annotated: list[Frame] = []

    result = pipeline().process_frames(
        frames(),
        fps=2.0,
        annotated_frame_callback=annotated.append,
    )
    manifest = pipeline().write_outputs(
        result,
        tmp_path,
        formats=("parquet", "json"),
    )

    assert result.frame_count == 3
    assert len(result.detections) == 6
    assert len(result.tracks) == len(result.positions) == 3
    assert len(result.pitch_keypoints) == 12
    assert len(result.ball_positions) == len(result.possessions) == 3
    assert result.team_assignments == {9: TeamId.TEAM_A}
    assert result.identity_resolution.merged_tracklet_count == 0
    assert all(item.team_id == TeamId.TEAM_A for item in result.positions)
    assert len(annotated) == 3
    assert annotated[0].shape == frames(1)[0].shape
    assert manifest.report.is_file()
    assert manifest.tables["positions"]["parquet"].is_file()
    assert manifest.tables["team_stats"]["json"].is_file()
    assert manifest.visualizations["shot_map"].is_file()
    assert manifest.visualizations["player_9_heatmap"].is_file()
    report = json.loads(manifest.report.read_text())
    assert report["identity_resolution"]["raw_to_canonical"] == {"9": 9}


def test_pipeline_honours_maximum_frames() -> None:
    result = pipeline().process_frames(frames(5), fps=2.0, maximum_frames=2)

    assert result.frame_count == 2
    assert {item.frame_id for item in result.detections} == {0, 1}


def test_pipeline_uses_canonical_identity_after_raw_track_fragmentation() -> None:
    analyzer = VideoAnalysisPipeline(
        detector=FakeDetector(),
        tracker=FragmentingTracker(),
        keypoint_detector=FakeKeypointDetector(),
        team_classifier=FakeTeamClassifier(),
    )

    result = analyzer.process_frames(frames(3), fps=2.0)

    assert {item.track_id for item in result.positions} == {9}
    assert {item.track_id for item in result.tracks} == {9, 99}
    assert result.team_assignments == {9: TeamId.TEAM_A}
    assert result.identity_resolution.model_dump() == {
        "raw_track_count": 2,
        "canonical_track_count": 1,
        "merged_tracklet_count": 1,
        "raw_to_canonical": {9: 9, 99: 9},
    }


def test_pipeline_uses_video_calibrated_homography_confidence() -> None:
    analyzer = pipeline()

    assert analyzer.homography.config.minimum_confidence == 0.4


def test_video_capture_and_annotated_writer(tmp_path) -> None:
    source = tmp_path / "source.avi"
    destination = tmp_path / "annotated.avi"
    writer = cv2.VideoWriter(
        str(source),
        cv2.VideoWriter_fourcc(*"MJPG"),
        2.0,
        (120, 80),
    )
    if not writer.isOpened():
        pytest.skip("MJPG video writer is unavailable")
    for frame in frames():
        writer.write(frame)
    writer.release()

    result = pipeline(config=VideoPipelineConfig(video_codec="MJPG")).analyze_video(
        source,
        annotated_video=destination,
    )

    assert result.frame_count == 3
    assert destination.is_file()
    assert destination.stat().st_size > 0
