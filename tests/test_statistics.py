import pytest

from football_analytics.analytics import MatchStatisticsCalculator
from football_analytics.storage import (
    ExpectedGoalPrediction,
    PassEvent,
    PlayerPosition,
    PossessionObservation,
    ShotEvent,
    TeamId,
)


def position(
    frame_id: int,
    track_id: int,
    team_id: TeamId,
    x: float,
    y: float,
) -> PlayerPosition:
    return PlayerPosition(
        frame_id=frame_id,
        timestamp=float(frame_id),
        track_id=track_id,
        team_id=team_id,
        x=x,
        y=y,
    )


def possession(
    frame_id: int,
    player_id: int,
    team_id: TeamId,
) -> PossessionObservation:
    return PossessionObservation(
        frame_id=frame_id,
        timestamp=float(frame_id),
        player_id=player_id,
        team_id=team_id,
        distance_to_ball=0.5,
        confidence=0.9,
    )


def pass_event(start_x: float, end_x: float) -> PassEvent:
    return PassEvent(
        timestamp=1.0,
        frame_id=1,
        team_id=TeamId.TEAM_A,
        passer_id=9,
        receiver_id=10,
        start_x=start_x,
        start_y=34.0,
        end_x=end_x,
        end_y=34.0,
        distance=end_x - start_x,
        duration=0.5,
        completed=True,
        confidence=0.9,
    )


def shot_event(
    frame_id: int,
    player_id: int,
    team_id: TeamId,
    outcome: str | None,
) -> ShotEvent:
    return ShotEvent(
        timestamp=float(frame_id),
        frame_id=frame_id,
        team_id=team_id,
        player_id=player_id,
        x=94.0 if team_id is TeamId.TEAM_A else 11.0,
        y=34.0,
        distance_to_goal=11.0,
        angle_to_goal=0.6,
        outcome=outcome,
        confidence=0.9,
    )


def test_match_statistics_aggregate_tracking_events_and_xg() -> None:
    positions = [
        position(0, 9, TeamId.TEAM_A, 60.0, 20.0),
        position(1, 9, TeamId.TEAM_A, 75.0, 20.0),
        position(2, 9, TeamId.TEAM_A, 90.0, 20.0),
        position(0, 10, TeamId.TEAM_A, 55.0, 40.0),
        position(1, 10, TeamId.TEAM_A, 70.0, 42.0),
        position(2, 10, TeamId.TEAM_A, 85.0, 44.0),
        position(0, 11, TeamId.TEAM_B, 45.0, 25.0),
        position(1, 11, TeamId.TEAM_B, 30.0, 25.0),
        position(2, 11, TeamId.TEAM_B, 15.0, 25.0),
        position(0, 12, TeamId.TEAM_B, 50.0, 45.0),
        position(1, 12, TeamId.TEAM_B, 35.0, 45.0),
        position(2, 12, TeamId.TEAM_B, 20.0, 45.0),
    ]
    possessions = [
        possession(0, 9, TeamId.TEAM_A),
        possession(1, 9, TeamId.TEAM_A),
        possession(2, 11, TeamId.TEAM_B),
        possession(3, 11, TeamId.TEAM_B),
    ]
    passes = [pass_event(60.0, 75.0), pass_event(80.0, 92.0)]
    shots = [
        shot_event(2, 9, TeamId.TEAM_A, "goal"),
        shot_event(3, 11, TeamId.TEAM_B, None),
    ]
    predictions = [
        ExpectedGoalPrediction(
            frame_id=2,
            timestamp=2.0,
            team_id=TeamId.TEAM_A,
            player_id=9,
            xg=0.4,
        )
    ]

    result = MatchStatisticsCalculator().calculate(
        positions=positions,
        possessions=possessions,
        passes=passes,
        shots=shots,
        xg_predictions=predictions,
    )

    player = next(item for item in result.players if item.player_id == 9)
    assert player.tracked_seconds == 2.0
    assert player.distance_m == 30.0
    assert player.average_speed_mps == 15.0
    assert player.maximum_speed_mps == 15.0
    assert player.possession_seconds == 2.0
    assert player.progressive_distance_with_ball_m == 15.0
    assert player.passes == player.completed_passes == 2
    assert player.pass_accuracy == 100.0
    assert player.shots_on_target == player.goals == 1
    assert player.xg == 0.4
    assert player.average_position_x == 75.0

    team_a, team_b = result.teams
    assert team_a.team_id == TeamId.TEAM_A
    assert team_a.possession_percentage == pytest.approx(200 / 3)
    assert team_b.possession_percentage == pytest.approx(100 / 3)
    assert team_a.progressive_passes == 2
    assert team_a.final_third_entries == 1
    assert team_a.penalty_area_entries == 1
    assert team_a.xg == 0.4
    assert team_a.field_tilt == 50.0
    assert team_b.field_tilt == 50.0
    assert team_a.territorial_possession == pytest.approx(200 / 3)
    assert team_b.xg is None
    assert team_b.shots_on_target is None
    assert team_a.ppda is None
    assert team_a.average_width_m is not None
    assert team_a.compactness_m is not None


def test_unknown_shot_outcome_is_not_invented() -> None:
    result = MatchStatisticsCalculator().calculate(shots=[shot_event(1, 9, TeamId.TEAM_A, None)])

    player = result.players[0]
    assert player.shots == 1
    assert player.shots_on_target is None
    assert player.goals is None
    assert player.xg is None


def test_conflicting_player_teams_are_rejected() -> None:
    with pytest.raises(ValueError, match="conflicting teams"):
        MatchStatisticsCalculator().calculate(
            positions=[
                position(0, 9, TeamId.TEAM_A, 10.0, 10.0),
                position(1, 9, TeamId.TEAM_B, 11.0, 10.0),
            ]
        )


def test_interrupted_possession_does_not_create_progressive_carry() -> None:
    result = MatchStatisticsCalculator().calculate(
        positions=[
            position(0, 9, TeamId.TEAM_A, 10.0, 10.0),
            position(2, 9, TeamId.TEAM_A, 30.0, 10.0),
        ],
        possessions=[
            possession(0, 9, TeamId.TEAM_A),
            possession(1, 11, TeamId.TEAM_B),
            possession(2, 9, TeamId.TEAM_A),
        ],
    )

    player = next(item for item in result.players if item.player_id == 9)
    assert player.progressive_distance_with_ball_m == 0.0


def test_xg_prediction_must_match_a_shot() -> None:
    prediction = ExpectedGoalPrediction(
        frame_id=99,
        timestamp=1.0,
        team_id=TeamId.TEAM_A,
        player_id=9,
        xg=0.2,
    )

    with pytest.raises(ValueError, match="no matching shot"):
        MatchStatisticsCalculator().calculate(xg_predictions=[prediction])
