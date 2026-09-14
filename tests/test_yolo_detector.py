from types import SimpleNamespace

import numpy as np
import pytest

from football_analytics.detection import YoloDetector, YoloDetectorConfig


class FakeYoloModel:
    def __init__(self, results: list[object]) -> None:
        self.results = results
        self.options: dict[str, object] = {}

    def predict(self, source: np.ndarray, **kwargs: object) -> list[object]:
        self.source = source
        self.options = kwargs
        return self.results


def make_result() -> object:
    boxes = SimpleNamespace(
        xyxy=np.asarray([[10.0, 20.0, 30.0, 80.0], [40.0, 5.0, 50.0, 15.0]]),
        conf=np.asarray([0.92, 0.88]),
        cls=np.asarray([0.0, 1.0]),
    )
    return SimpleNamespace(boxes=boxes, names={0: "person", 1: "sports ball"})


def test_yolo_adapter_translates_results_and_maps_class_names() -> None:
    model = FakeYoloModel([make_result()])
    config = YoloDetectorConfig(
        confidence_threshold=0.4,
        iou_threshold=0.5,
        class_name_map={"person": "player", "sports ball": "ball"},
        allowed_classes=frozenset({"player", "ball"}),
    )
    detector = YoloDetector("unused.pt", config, model=model)
    frame = np.zeros((100, 160, 3), dtype=np.uint8)

    detections = detector.detect(frame, frame_id=12)

    assert [detection.class_name for detection in detections] == ["player", "ball"]
    assert detections[0].bbox == (10.0, 20.0, 30.0, 80.0)
    assert detections[0].center == (20.0, 50.0)
    assert detections[0].frame_id == 12
    assert model.options == {"conf": 0.4, "iou": 0.5, "verbose": False}


def test_yolo_adapter_filters_after_class_name_mapping() -> None:
    model = FakeYoloModel([make_result()])
    config = YoloDetectorConfig(
        class_name_map={"person": "player", "sports ball": "ball"},
        allowed_classes=frozenset({"ball"}),
    )

    detections = YoloDetector("unused.pt", config, model=model).detect(
        np.zeros((20, 20, 3), dtype=np.uint8)
    )

    assert len(detections) == 1
    assert detections[0].class_name == "ball"


def test_yolo_adapter_handles_result_without_boxes() -> None:
    model = FakeYoloModel([SimpleNamespace(boxes=None)])
    detector = YoloDetector("unused.pt", model=model)

    assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
        np.zeros((0, 10, 3), dtype=np.uint8),
    ],
)
def test_yolo_adapter_rejects_invalid_frames(frame: np.ndarray) -> None:
    detector = YoloDetector("unused.pt", model=FakeYoloModel([]))

    with pytest.raises(ValueError, match="HxWx3 uint8"):
        detector.detect(frame)
