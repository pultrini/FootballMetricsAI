"""Pure analytics functions."""

from football_analytics.domain.analytics.geometry import (
    euclidean_distance,
    shot_distance,
    visible_goal_angle,
)
from football_analytics.domain.analytics.positions import (
    MetricPositionEstimator,
    PositionEstimatorConfig,
)
from football_analytics.domain.analytics.statistics import (
    MatchStatisticsCalculator,
    StatisticsConfig,
)
from football_analytics.domain.analytics.xg import (
    ExpectedGoalsModel,
    XGModelArtifactError,
    XGModelConfig,
    aggregate_expected_goals,
)

__all__ = [
    "ExpectedGoalsModel",
    "MatchStatisticsCalculator",
    "MetricPositionEstimator",
    "PositionEstimatorConfig",
    "StatisticsConfig",
    "XGModelArtifactError",
    "XGModelConfig",
    "aggregate_expected_goals",
    "euclidean_distance",
    "shot_distance",
    "visible_goal_angle",
]
