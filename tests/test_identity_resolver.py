import pytest

from football_analytics.storage import PlayerPosition, TeamId, Track
from football_analytics.tracking import IdentityResolverConfig, MetricIdentityResolver


def position(
    frame_id: int,
    timestamp: float,
    track_id: int,
    x: float,
    *,
    y: float = 20.0,
    team_id: TeamId | None = TeamId.TEAM_A,
) -> PlayerPosition:
    return PlayerPosition(
        frame_id=frame_id,
        timestamp=timestamp,
        track_id=track_id,
        team_id=team_id,
        x=x,
        y=y,
    )


def test_nearby_new_tracklet_reuses_lost_canonical_identity() -> None:
    resolver = MetricIdentityResolver()

    first = resolver.update([position(0, 0.0, 7, 10.0)], frame_id=0, timestamp=0.0)
    resolver.update([], frame_id=1, timestamp=0.2)
    reappeared = resolver.update([position(2, 0.4, 91, 11.0)], frame_id=2, timestamp=0.4)

    assert first[0].track_id == 7
    assert reappeared[0].track_id == 7
    assert resolver.canonical_id(91) == 7


def test_simultaneously_active_identity_cannot_be_reused() -> None:
    resolver = MetricIdentityResolver()
    resolver.update([position(0, 0.0, 7, 10.0)], frame_id=0, timestamp=0.0)

    current = resolver.update(
        [position(1, 0.2, 7, 10.5), position(1, 0.2, 91, 10.7)],
        frame_id=1,
        timestamp=0.2,
    )

    assert {item.track_id for item in current} == {7, 91}


def test_team_mismatch_prevents_tracklet_merge() -> None:
    resolver = MetricIdentityResolver()
    resolver.update([position(0, 0.0, 7, 10.0)], frame_id=0, timestamp=0.0)

    current = resolver.update(
        [position(1, 0.2, 91, 10.2, team_id=TeamId.TEAM_B)],
        frame_id=1,
        timestamp=0.2,
    )

    assert current[0].track_id == 91


def test_expired_identity_is_not_reused() -> None:
    resolver = MetricIdentityResolver(IdentityResolverConfig(maximum_missing_seconds=1.0))
    resolver.update([position(0, 0.0, 7, 10.0)], frame_id=0, timestamp=0.0)

    current = resolver.update([position(1, 1.2, 91, 10.2)], frame_id=1, timestamp=1.2)

    assert current[0].track_id == 91


def test_global_assignment_is_one_to_one() -> None:
    resolver = MetricIdentityResolver()
    resolver.update(
        [position(0, 0.0, 4, 10.0), position(0, 0.0, 8, 20.0)],
        frame_id=0,
        timestamp=0.0,
    )
    resolver.update([], frame_id=1, timestamp=0.2)

    current = resolver.update(
        [position(2, 0.4, 40, 10.5), position(2, 0.4, 80, 19.5)],
        frame_id=2,
        timestamp=0.4,
    )

    assert {item.track_id for item in current} == {4, 8}


def test_team_assignments_and_render_tracks_use_canonical_ids() -> None:
    resolver = MetricIdentityResolver()
    resolver.update([position(0, 0.0, 7, 10.0)], frame_id=0, timestamp=0.0)
    resolver.update([], frame_id=1, timestamp=0.2)
    resolver.update([position(2, 0.4, 91, 10.5)], frame_id=2, timestamp=0.4)
    track = Track(
        frame_id=2,
        track_id=91,
        class_name="player",
        bbox=(0.0, 0.0, 10.0, 20.0),
        pixel_position=(5.0, 20.0),
        confidence=0.9,
    )

    assert resolver.relabel_tracks([track])[0].track_id == 7
    assert resolver.resolve_team_assignments({7: TeamId.TEAM_A, 91: TeamId.TEAM_A}) == {
        7: TeamId.TEAM_A
    }
    assert resolver.summary().model_dump() == {
        "raw_track_count": 2,
        "canonical_track_count": 1,
        "merged_tracklet_count": 1,
        "raw_to_canonical": {7: 7, 91: 7},
    }


def test_rejects_non_increasing_timestamps() -> None:
    resolver = MetricIdentityResolver()
    resolver.update([], frame_id=0, timestamp=0.0)

    with pytest.raises(ValueError, match="timestamps must increase"):
        resolver.update([], frame_id=1, timestamp=0.0)
