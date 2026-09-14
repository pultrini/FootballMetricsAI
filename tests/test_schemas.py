import pytest
from pydantic import ValidationError

from football_analytics.storage import Detection, PlayerPosition, TeamId


def test_detection_computes_center_and_serializes_it() -> None:
    detection = Detection(
        frame_id=7,
        class_id=0,
        class_name="player",
        confidence=0.91,
        bbox=(10.0, 20.0, 30.0, 60.0),
    )
    assert detection.center == (20.0, 40.0)
    assert detection.model_dump()["center"] == (20.0, 40.0)


def test_detection_rejects_inverted_bbox() -> None:
    with pytest.raises(ValidationError, match="bbox must use"):
        Detection(
            frame_id=0,
            class_id=0,
            class_name="player",
            confidence=0.9,
            bbox=(20.0, 20.0, 10.0, 60.0),
        )


def test_player_position_must_be_inside_canonical_pitch() -> None:
    with pytest.raises(ValidationError):
        PlayerPosition(
            frame_id=0,
            timestamp=0.0,
            track_id=1,
            team_id=TeamId.TEAM_A,
            x=106.0,
            y=34.0,
        )
