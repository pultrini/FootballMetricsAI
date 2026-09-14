import numpy as np
import pytest

from football_analytics.pitch import (
    HomographyEstimationError,
    PitchHomography,
    PitchHomographyConfig,
    UnreliableHomographyError,
)
from football_analytics.storage import PitchKeypoint

IMAGE_CORNERS = [(100.0, 100.0), (100.0, 500.0), (900.0, 100.0), (900.0, 500.0)]
PITCH_CORNERS = [(0.0, 0.0), (0.0, 68.0), (105.0, 0.0), (105.0, 68.0)]


def test_homography_transforms_pixel_coordinates_to_metres() -> None:
    homography = PitchHomography().estimate(
        IMAGE_CORNERS,
        PITCH_CORNERS,
        [0.95, 0.95, 0.95, 0.95],
    )

    assert homography.validate()
    assert homography.confidence == pytest.approx(0.95)
    assert homography.reprojection_rmse == pytest.approx(0.0, abs=1e-8)
    assert homography.transform_point((500.0, 300.0)) == pytest.approx((52.5, 34.0))


def test_homography_ransac_rejects_an_outlier() -> None:
    image = np.asarray([*IMAGE_CORNERS, (500.0, 300.0), (300.0, 200.0)], dtype=np.float64)
    pitch = np.asarray([*PITCH_CORNERS, (52.5, 34.0), (5.0, 60.0)], dtype=np.float64)

    homography = PitchHomography().estimate(image, pitch, np.full(6, 0.95))

    assert homography.validate()
    assert homography.inlier_mask is not None
    assert homography.inlier_mask.sum() == 5
    assert homography.transform_point((500.0, 300.0)) == pytest.approx((52.5, 34.0), abs=1e-6)


def test_failed_recalculation_clears_previous_matrix() -> None:
    homography = PitchHomography().estimate(IMAGE_CORNERS, PITCH_CORNERS)
    assert homography.validate()

    with pytest.raises(HomographyEstimationError, match="At least 4"):
        homography.estimate(IMAGE_CORNERS[:3], PITCH_CORNERS[:3])

    assert homography.matrix is None
    assert not homography.validate()


def test_collinear_correspondences_are_rejected() -> None:
    line = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]

    with pytest.raises(HomographyEstimationError, match="collinear"):
        PitchHomography().estimate(line, PITCH_CORNERS)


def test_low_confidence_estimate_cannot_generate_spatial_output() -> None:
    config = PitchHomographyConfig(
        minimum_keypoint_confidence=0.0,
        minimum_confidence=0.5,
    )
    homography = PitchHomography(config).estimate(
        IMAGE_CORNERS,
        PITCH_CORNERS,
        [0.2, 0.2, 0.2, 0.2],
    )

    assert homography.matrix is not None
    assert not homography.validate()
    with pytest.raises(UnreliableHomographyError):
        homography.transform_point((500.0, 300.0))


def test_homography_can_be_recalculated_after_camera_zoom() -> None:
    homography = PitchHomography().estimate(
        [(0.0, 0.0), (0.0, 100.0), (100.0, 0.0), (100.0, 100.0)],
        PITCH_CORNERS,
    )
    assert homography.transform_point((50.0, 50.0)) == pytest.approx((52.5, 34.0))

    homography.estimate(
        [(0.0, 0.0), (0.0, 200.0), (200.0, 0.0), (200.0, 200.0)],
        PITCH_CORNERS,
    )
    assert homography.transform_point((50.0, 50.0)) == pytest.approx((26.25, 17.0))


def test_estimate_from_keypoints_rejects_mixed_frames() -> None:
    points = [
        PitchKeypoint(
            frame_id=index % 2,
            id=str(index),
            name=str(index),
            image_x=image_x,
            image_y=image_y,
            pitch_x=pitch_x,
            pitch_y=pitch_y,
            confidence=0.9,
        )
        for index, ((image_x, image_y), (pitch_x, pitch_y)) in enumerate(
            zip(IMAGE_CORNERS, PITCH_CORNERS, strict=True)
        )
    ]

    with pytest.raises(HomographyEstimationError, match="exactly one frame"):
        PitchHomography().estimate_keypoints(points)
