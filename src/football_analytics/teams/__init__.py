"""Jersey-colour based team assignment."""

from football_analytics.adapters.outbound.team_classifier import (
    JerseyTeamClassifier,
    TeamClassifier,
    TeamClassifierConfig,
)

__all__ = ["JerseyTeamClassifier", "TeamClassifier", "TeamClassifierConfig"]
