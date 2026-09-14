import numpy as np
import pytest
import supervision as sv

from football_analytics.storage import Detection
from football_analytics.tracking import ByteTrackTracker


class FakeByteTrackBackend:
    def __init__(self) -> None:
        self.calls: list[sv.Detections] = []
        self.was_reset = False

    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> sv.Detections:
        del frame, timestamp
        self.calls.append(detections)
        detections.tracker_id = np.full(len(detections), 42, dtype=int)
        return detections

    def reset(self) -> None:
        self.was_reset = True


class UnconfirmedByteTrackBackend(FakeByteTrackBackend):
    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> sv.Detections:
        del frame, timestamp
        detections.tracker_id = np.full(len(detections), -1, dtype=int)
        return detections


def make_detection(frame_id: int = 3) -> Detection:
    return Detection(
        frame_id=frame_id,
        class_id=0,
        class_name="player",
        confidence=0.93,
        bbox=(10.0, 20.0, 30.0, 80.0),
    )


def test_bytetrack_adapter_assigns_id_and_uses_bottom_center() -> None:
    backend = FakeByteTrackBackend()
    tracker = ByteTrackTracker(backend=backend)

    tracks = tracker.update([make_detection()], frame_id=3)

    assert len(tracks) == 1
    assert tracks[0].track_id == 42
    assert tracks[0].class_name == "player"
    assert tracks[0].pixel_position == (20.0, 80.0)
    assert backend.calls[0].xyxy.tolist() == [[10.0, 20.0, 30.0, 80.0]]


def test_bytetrack_adapter_advances_on_frame_without_detections() -> None:
    backend = FakeByteTrackBackend()
    tracker = ByteTrackTracker(backend=backend)

    assert tracker.update([], frame_id=4) == []
    assert len(backend.calls) == 1
    assert backend.calls[0].xyxy.shape == (0, 4)


def test_bytetrack_adapter_omits_unconfirmed_internal_tracks() -> None:
    tracker = ByteTrackTracker(backend=UnconfirmedByteTrackBackend())

    assert tracker.update([make_detection()], frame_id=3) == []


def test_bytetrack_adapter_rejects_mixed_frame_ids() -> None:
    tracker = ByteTrackTracker(backend=FakeByteTrackBackend())

    with pytest.raises(ValueError, match="supplied frame_id"):
        tracker.update([make_detection(frame_id=2)], frame_id=3)


def test_bytetrack_adapter_resets_backend() -> None:
    backend = FakeByteTrackBackend()
    tracker = ByteTrackTracker(backend=backend)

    tracker.reset()

    assert backend.was_reset
