"""Compatibility exports for pitch homography."""

from football_analytics.adapters.outbound.opencv_homography import OpenCvHomographyBackend
from football_analytics.domain.pitch.homography import (
    PitchHomography as DomainPitchHomography,
)
from football_analytics.domain.pitch.homography import (
    PitchHomographyConfig,
)


class PitchHomography(DomainPitchHomography):
    """Legacy constructor retaining the original default OpenCV backend."""

    def __init__(self, config: PitchHomographyConfig | None = None) -> None:
        super().__init__(config, backend=OpenCvHomographyBackend())
