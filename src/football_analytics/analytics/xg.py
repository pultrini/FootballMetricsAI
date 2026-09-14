"""Trainable baseline expected-goals model and aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from football_analytics.storage import (
    ExpectedGoalPrediction,
    ShotDataset,
    ShotEvent,
    XGAggregation,
)

FEATURE_NAMES = ("distance", "angle")
ARTIFACT_VERSION = 1


class XGModelConfig(BaseModel):
    """Hyperparameters for the logistic-regression baseline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    regularization_c: float = Field(default=1.0, gt=0)
    maximum_iterations: int = Field(default=1_000, ge=1)
    class_weight: Literal["balanced"] | None = None
    random_state: int = 42


class XGModelArtifactError(ValueError):
    """Raised when a persisted xG artifact is incompatible or malformed."""


class ExpectedGoalsModel:
    """Estimate goal probability from shot distance and visible goal angle."""

    def __init__(self, config: XGModelConfig | None = None) -> None:
        self.config = config or XGModelConfig()
        self._estimator = self._new_estimator()

    def fit(
        self,
        features: Sequence[Sequence[float]] | npt.NDArray[np.number],
        targets: Sequence[int | bool] | npt.NDArray[np.number],
    ) -> ExpectedGoalsModel:
        """Fit the probabilistic model on binary historical outcomes."""
        matrix = self._feature_matrix(features)
        labels = np.asarray(targets, dtype=np.int64)
        if labels.ndim != 1 or len(labels) != len(matrix):
            raise ValueError("targets must be a vector matching the feature rows")
        unique_labels = set(labels.tolist())
        if unique_labels != {0, 1}:
            raise ValueError("targets must contain both binary classes 0 and 1")
        self._estimator = self._new_estimator()
        self._estimator.fit(matrix, labels)
        return self

    def fit_dataset(self, dataset: ShotDataset) -> ExpectedGoalsModel:
        """Fit from a validated historical shot dataset."""
        return self.fit(dataset.features(), dataset.targets())

    def predict_proba(
        self,
        features: Sequence[Sequence[float]] | Sequence[float] | npt.NDArray[np.number],
    ) -> npt.NDArray[np.float64]:
        """Return goal-class probabilities as a one-dimensional array."""
        check_is_fitted(self._estimator)
        matrix = self._feature_matrix(features)
        probabilities = self._estimator.predict_proba(matrix)[:, 1]
        return np.asarray(np.clip(probabilities, 0.0, 1.0), dtype=np.float64)

    def predict_shot(self, shot: ShotEvent) -> float:
        """Return xG for one detected shot."""
        return float(self.predict_proba([shot.distance_to_goal, shot.angle_to_goal])[0])

    def predict_shots(self, shots: Sequence[ShotEvent]) -> list[ExpectedGoalPrediction]:
        """Attach xG probabilities to shot identities."""
        if not shots:
            return []
        features = [[shot.distance_to_goal, shot.angle_to_goal] for shot in shots]
        probabilities = self.predict_proba(features)
        return [
            ExpectedGoalPrediction(
                frame_id=shot.frame_id,
                timestamp=shot.timestamp,
                team_id=shot.team_id,
                player_id=shot.player_id,
                xg=float(probability),
            )
            for shot, probability in zip(shots, probabilities, strict=True)
        ]

    def save(self, path: str | Path) -> Path:
        """Persist the fitted estimator and versioned feature metadata."""
        check_is_fitted(self._estimator)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "artifact_version": ARTIFACT_VERSION,
                "feature_names": FEATURE_NAMES,
                "config": self.config.model_dump(),
                "estimator": self._estimator,
            },
            destination,
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> ExpectedGoalsModel:
        """Load a trusted artifact created by this project."""
        source = Path(path)
        if not source.is_file():
            raise XGModelArtifactError(f"xG model does not exist: {source}")
        payload = joblib.load(source)
        if not isinstance(payload, dict):
            raise XGModelArtifactError("xG artifact must contain a metadata dictionary")
        if payload.get("artifact_version") != ARTIFACT_VERSION:
            raise XGModelArtifactError("Unsupported xG artifact version")
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise XGModelArtifactError("xG artifact uses incompatible features")
        estimator = payload.get("estimator")
        if not isinstance(estimator, LogisticRegression):
            raise XGModelArtifactError("xG artifact does not contain logistic regression")
        try:
            config = XGModelConfig.model_validate(payload.get("config"))
        except ValidationError as error:
            raise XGModelArtifactError("xG artifact contains invalid configuration") from error
        model = cls(config)
        model._estimator = estimator
        check_is_fitted(model._estimator)
        return model

    def _new_estimator(self) -> LogisticRegression:
        return LogisticRegression(
            C=self.config.regularization_c,
            max_iter=self.config.maximum_iterations,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            solver="lbfgs",
        )

    @staticmethod
    def _feature_matrix(
        features: Sequence[Sequence[float]] | Sequence[float] | npt.NDArray[np.number],
    ) -> npt.NDArray[np.float64]:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
            raise ValueError("features must have exactly two columns: distance, angle")
        if len(matrix) == 0:
            raise ValueError("features must contain at least one row")
        if not np.isfinite(matrix).all():
            raise ValueError("features must contain only finite values")
        if np.any(matrix[:, 0] < 0):
            raise ValueError("shot distance must be non-negative")
        if np.any((matrix[:, 1] < 0) | (matrix[:, 1] > np.pi)):
            raise ValueError("shot angle must be between 0 and pi radians")
        return matrix


def aggregate_expected_goals(
    predictions: Sequence[ExpectedGoalPrediction],
) -> XGAggregation:
    """Sum the same shot probabilities by player and team."""
    player_xg: dict[int, float] = {}
    team_xg: dict[str, float] = {}
    for prediction in predictions:
        player_xg[prediction.player_id] = player_xg.get(prediction.player_id, 0.0) + prediction.xg
        team_key = str(prediction.team_id)
        team_xg[team_key] = team_xg.get(team_key, 0.0) + prediction.xg
    return XGAggregation(player_xg=player_xg, team_xg=team_xg)
