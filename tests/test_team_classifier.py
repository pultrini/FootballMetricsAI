import numpy as np
import pytest

from football_analytics.storage import TeamId, Track
from football_analytics.teams import JerseyTeamClassifier, TeamClassifierConfig


def track(track_id: int, bbox: tuple[float, float, float, float]) -> Track:
    return Track(
        frame_id=0,
        track_id=track_id,
        class_name="player",
        bbox=bbox,
        pixel_position=((bbox[0] + bbox[2]) / 2, bbox[3]),
        confidence=0.9,
    )


def jersey_frame() -> np.ndarray:
    frame = np.full((80, 120, 3), (30, 120, 30), dtype=np.uint8)
    frame[10:50, 10:40] = (0, 0, 255)
    frame[10:50, 70:100] = (255, 0, 0)
    return frame


def test_colour_anchors_assign_teams_immediately() -> None:
    classifier = JerseyTeamClassifier(
        TeamClassifierConfig(team_a_rgb=(255, 0, 0), team_b_rgb=(0, 0, 255))
    )

    assignments = classifier.classify(
        jersey_frame(),
        [track(1, (10, 10, 40, 50)), track(2, (70, 10, 100, 50))],
    )

    assert assignments == {1: TeamId.TEAM_A, 2: TeamId.TEAM_B}


def test_unsupervised_colours_create_two_stable_clusters() -> None:
    classifier = JerseyTeamClassifier(
        TeamClassifierConfig(minimum_samples=4, minimum_unique_tracks=2)
    )
    tracks = [track(1, (10, 10, 40, 50)), track(2, (70, 10, 100, 50))]

    classifier.classify(jersey_frame(), tracks)
    assignments = classifier.classify(jersey_frame(), tracks)

    assert set(assignments) == {1, 2}
    assert set(assignments.values()) == {TeamId.TEAM_A, TeamId.TEAM_B}


def test_team_colour_anchors_must_be_supplied_together() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        TeamClassifierConfig(team_a_rgb=(255, 0, 0))
