"""Multi-object tracking contracts and adapters."""

from football_analytics.tracking.bytetrack_tracker import (
    ByteTrackConfig,
    ByteTrackTracker,
)
from football_analytics.tracking.identity_resolver import (
    IdentityResolverConfig,
    MetricIdentityResolver,
)
from football_analytics.tracking.tracker import Tracker

__all__ = [
    "ByteTrackConfig",
    "ByteTrackTracker",
    "IdentityResolverConfig",
    "MetricIdentityResolver",
    "Tracker",
]
