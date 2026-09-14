"""Team classification from tracked player jersey colours."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Annotated

import cv2
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.cluster import KMeans

from football_analytics.detection import Frame, validate_frame
from football_analytics.storage import TeamId, Track

ColorChannel = Annotated[int, Field(ge=0, le=255)]
RgbColor = tuple[ColorChannel, ColorChannel, ColorChannel]


class TeamClassifierConfig(BaseModel):
    """Sampling thresholds and optional deterministic team colours."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_samples: int = Field(default=20, ge=2)
    minimum_unique_tracks: int = Field(default=4, ge=2)
    random_state: int = 42
    team_a_rgb: RgbColor | None = None
    team_b_rgb: RgbColor | None = None

    @model_validator(mode="after")
    def validate_anchors(self) -> TeamClassifierConfig:
        """Require both colour anchors or neither."""
        if (self.team_a_rgb is None) != (self.team_b_rgb is None):
            raise ValueError("team_a_rgb and team_b_rgb must be supplied together")
        if self.team_a_rgb is not None and self.team_a_rgb == self.team_b_rgb:
            raise ValueError("team colour anchors must be different")
        return self


class TeamClassifier(ABC):
    """Assign stable semantic teams to tracked players."""

    @property
    @abstractmethod
    def assignments(self) -> Mapping[int, TeamId]:
        """Return every stable assignment known so far."""

    @abstractmethod
    def classify(self, frame: Frame, tracks: Sequence[Track]) -> Mapping[int, TeamId]:
        """Update colour evidence and return every known track assignment."""

    @abstractmethod
    def reset(self) -> None:
        """Clear state for a new match."""


class JerseyTeamClassifier(TeamClassifier):
    """Classify jersey crops with anchored prototypes or deterministic K-means."""

    def __init__(self, config: TeamClassifierConfig | None = None) -> None:
        self.config = config or TeamClassifierConfig()
        self.reset()

    @property
    def assignments(self) -> Mapping[int, TeamId]:
        """Return a defensive copy of current stable assignments."""
        return dict(self._assignments)

    def reset(self) -> None:
        """Clear collected colours and restore optional anchor prototypes."""
        self._samples: list[npt.NDArray[np.float64]] = []
        self._sample_track_ids: list[int] = []
        self._track_features: dict[int, npt.NDArray[np.float64]] = {}
        self._track_counts: dict[int, int] = {}
        self._assignments: dict[int, TeamId] = {}
        self._centres = self._anchor_centres()

    def classify(self, frame: Frame, tracks: Sequence[Track]) -> Mapping[int, TeamId]:
        """Collect one colour sample per visible player and classify known tracks."""
        validate_frame(frame)
        for track in tracks:
            if track.class_name.casefold() != "player":
                continue
            feature = self._jersey_feature(frame, track)
            if feature is None:
                continue
            count = self._track_counts.get(track.track_id, 0)
            previous = self._track_features.get(track.track_id)
            self._track_features[track.track_id] = (
                feature if previous is None else (previous * count + feature) / (count + 1)
            )
            self._track_counts[track.track_id] = count + 1
            if self._centres is None:
                self._samples.append(feature)
                self._sample_track_ids.append(track.track_id)

        if self._centres is None and self._ready_to_fit():
            self._fit_centres()
        if self._centres is not None:
            for track_id, feature in self._track_features.items():
                if track_id not in self._assignments:
                    self._assignments[track_id] = self._nearest_team(feature)
        return dict(self._assignments)

    def _ready_to_fit(self) -> bool:
        return (
            len(self._samples) >= self.config.minimum_samples
            and len(set(self._sample_track_ids)) >= self.config.minimum_unique_tracks
        )

    def _fit_centres(self) -> None:
        model = KMeans(n_clusters=2, n_init=10, random_state=self.config.random_state)
        model.fit(np.vstack(self._samples))
        centres = np.asarray(model.cluster_centers_, dtype=np.float64)
        order = sorted(range(2), key=lambda index: tuple(centres[index].tolist()))
        self._centres = np.vstack((centres[order[0]], centres[order[1]]))

    def _nearest_team(self, feature: npt.NDArray[np.float64]) -> TeamId:
        assert self._centres is not None
        distances = np.linalg.norm(self._centres - feature, axis=1)
        return TeamId.TEAM_A if int(np.argmin(distances)) == 0 else TeamId.TEAM_B

    def _anchor_centres(self) -> npt.NDArray[np.float64] | None:
        if self.config.team_a_rgb is None or self.config.team_b_rgb is None:
            return None
        return np.vstack(
            (
                self._rgb_to_lab(self.config.team_a_rgb),
                self._rgb_to_lab(self.config.team_b_rgb),
            )
        )

    @staticmethod
    def _rgb_to_lab(color: RgbColor) -> npt.NDArray[np.float64]:
        pixel = np.asarray([[color]], dtype=np.uint8)
        return cv2.cvtColor(pixel, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float64)

    @staticmethod
    def _jersey_feature(
        frame: Frame,
        track: Track,
    ) -> npt.NDArray[np.float64] | None:
        height, width = frame.shape[:2]
        x_min, y_min, x_max, y_max = track.bbox
        box_width = x_max - x_min
        box_height = y_max - y_min
        left = max(0, min(width, round(x_min + 0.2 * box_width)))
        right = max(0, min(width, round(x_max - 0.2 * box_width)))
        top = max(0, min(height, round(y_min + 0.1 * box_height)))
        bottom = max(0, min(height, round(y_min + 0.55 * box_height)))
        if right <= left or bottom <= top:
            return None
        crop = frame[top:bottom, left:right]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        grass = (
            (hsv[..., 0] >= 30) & (hsv[..., 0] <= 95) & (hsv[..., 1] >= 45) & (hsv[..., 2] >= 25)
        )
        pixels = crop[~grass]
        if len(pixels) < 8:
            pixels = crop.reshape(-1, 3)
        median_bgr = np.median(pixels, axis=0).round().astype(np.uint8)
        lab = cv2.cvtColor(median_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB)
        return lab[0, 0].astype(np.float64)
