import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from football_analytics.analytics import ExpectedGoalsModel, aggregate_expected_goals
from football_analytics.storage import (
    ExpectedGoalPrediction,
    ShotEvent,
    TeamId,
)

FEATURES = np.asarray(
    [
        [6.0, 1.00],
        [9.0, 0.80],
        [13.0, 0.60],
        [22.0, 0.30],
        [30.0, 0.18],
        [40.0, 0.08],
    ]
)
TARGETS = np.asarray([1, 1, 1, 0, 0, 0])


def shot(distance: float = 12.0, angle: float = 0.55) -> ShotEvent:
    return ShotEvent(
        timestamp=2.0,
        frame_id=50,
        team_id=TeamId.TEAM_A,
        player_id=9,
        x=94.0,
        y=34.0,
        distance_to_goal=distance,
        angle_to_goal=angle,
        ball_speed=24.0,
        confidence=0.9,
    )


def test_model_must_be_trained_before_inference() -> None:
    with pytest.raises(NotFittedError):
        ExpectedGoalsModel().predict_proba([[12.0, 0.5]])


def test_model_returns_bounded_probabilities_and_predicts_shot() -> None:
    model = ExpectedGoalsModel().fit(FEATURES, TARGETS)

    probabilities = model.predict_proba(FEATURES)

    assert probabilities.shape == (6,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert model.predict_shot(shot()) == pytest.approx(model.predict_proba([[12.0, 0.55]])[0])


def test_model_save_and_load_preserves_predictions(tmp_path) -> None:
    model = ExpectedGoalsModel().fit(FEATURES, TARGETS)
    destination = tmp_path / "models" / "xg.joblib"

    model.save(destination)
    restored = ExpectedGoalsModel.load(destination)

    assert destination.is_file()
    assert restored.predict_proba(FEATURES) == pytest.approx(model.predict_proba(FEATURES))


def test_model_rejects_invalid_training_data() -> None:
    with pytest.raises(ValueError, match="both binary classes"):
        ExpectedGoalsModel().fit([[10.0, 0.5], [20.0, 0.2]], [0, 0])

    with pytest.raises(ValueError, match="between 0 and pi"):
        ExpectedGoalsModel().fit([[10.0, 4.0], [20.0, 0.2]], [1, 0])


def test_xg_aggregation_has_equal_player_and_team_totals() -> None:
    predictions = [
        ExpectedGoalPrediction(
            frame_id=10,
            timestamp=0.4,
            team_id=TeamId.TEAM_A,
            player_id=9,
            xg=0.25,
        ),
        ExpectedGoalPrediction(
            frame_id=20,
            timestamp=0.8,
            team_id=TeamId.TEAM_A,
            player_id=10,
            xg=0.15,
        ),
        ExpectedGoalPrediction(
            frame_id=30,
            timestamp=1.2,
            team_id=TeamId.TEAM_B,
            player_id=11,
            xg=0.30,
        ),
    ]

    totals = aggregate_expected_goals(predictions)

    assert totals.player_xg == {9: 0.25, 10: 0.15, 11: 0.30}
    assert totals.team_xg == {"TEAM_A": 0.40, "TEAM_B": 0.30}
    assert sum(totals.player_xg.values()) == pytest.approx(sum(totals.team_xg.values()))
