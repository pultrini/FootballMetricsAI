from types import SimpleNamespace

import numpy as np
import pytest

from football_analytics.pitch import (
    PitchLandmark,
    YoloPitchKeypointConfig,
    YoloPitchKeypointDetector,
)

LANDMARKS = (
    PitchLandmark("a", "A", 0.0, 0.0),
    PitchLandmark("b", "B", 0.0, 68.0),
    PitchLandmark("c", "C", 105.0, 0.0),
    PitchLandmark("d", "D", 105.0, 68.0),
    PitchLandmark("e", "E", 52.5, 34.0),
)


class FakePoseModel:
    def __init__(self, result: object) -> None:
        self.result = result
        self.options: dict[str, object] = {}

    def predict(self, source: np.ndarray, **kwargs: object) -> list[object]:
        self.source = source
        self.options = kwargs
        return [self.result]


def test_pose_adapter_maps_indices_and_filters_low_confidence_points() -> None:
    keypoints = SimpleNamespace(
        xy=np.asarray([[[10.0, 10.0], [10.0, 90.0], [190.0, 10.0], [190.0, 90.0], [100, 50]]]),
        conf=np.asarray([[0.9, 0.8, 0.95, 0.85, 0.1]]),
    )
    model = FakePoseModel(SimpleNamespace(keypoints=keypoints))
    detector = YoloPitchKeypointDetector(
        "unused.pt",
        YoloPitchKeypointConfig(keypoint_confidence_threshold=0.5),
        landmarks=LANDMARKS,
        model=model,
    )

    observations = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8), frame_id=7)

    assert [point.id for point in observations] == ["a", "b", "c", "d"]
    assert observations[0].model_dump() == {
        "frame_id": 7,
        "id": "a",
        "name": "A",
        "image_x": 10.0,
        "image_y": 10.0,
        "pitch_x": 0.0,
        "pitch_y": 0.0,
        "confidence": 0.9,
    }
    assert model.options == {"conf": 0.25, "verbose": False}


def test_pose_adapter_requires_keypoint_confidence() -> None:
    keypoints = SimpleNamespace(xy=np.zeros((1, 5, 2)), conf=None)
    detector = YoloPitchKeypointDetector(
        "unused.pt",
        landmarks=LANDMARKS,
        model=FakePoseModel(SimpleNamespace(keypoints=keypoints)),
    )

    with pytest.raises(RuntimeError, match="per-keypoint confidence"):
        detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))


def test_pose_adapter_rejects_model_with_different_landmark_order_size() -> None:
    keypoints = SimpleNamespace(
        xy=np.zeros((1, 4, 2)),
        conf=np.ones((1, 4)),
    )
    detector = YoloPitchKeypointDetector(
        "unused.pt",
        landmarks=LANDMARKS,
        model=FakePoseModel(SimpleNamespace(keypoints=keypoints)),
    )

    with pytest.raises(RuntimeError, match="5 landmarks"):
        detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))
