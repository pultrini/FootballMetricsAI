"""Multi-object tracking contracts and adapters."""

from football_analytics.adapters.outbound.bytetrack_tracker import (
    ByteTrackConfig,
    ByteTrackTracker,
)
from football_analytics.application.ports.tracker import Tracker
from football_analytics.domain.analytics.identity_resolver import (
    IdentityResolverConfig,
    MetricIdentityResolver,
)

__all__ = [
    "ByteTrackConfig",
    "ByteTrackTracker",
    "IdentityResolverConfig",
    "MetricIdentityResolver",
    "Tracker",
]
