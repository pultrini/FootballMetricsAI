"""Matplotlib visualizations for frames, pitch geometry and homographies."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


def read_frame(video_path: Path, frame_id: int) -> np.ndarray:
    """Read one BGR frame from a video and fail explicitly when unavailable."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, frame = capture.read()
        if not success or frame is None:
            raise RuntimeError(f"Unable to read frame {frame_id} from {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def synthetic_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    """Create a deterministic field-like frame for notebook smoke visualization."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 1] = 92
    frame[:, :, 0] = 28
    frame[:, :, 2] = 28
    return frame


def plot_frame_with_keypoints(
    frame: np.ndarray,
    image_points: Sequence[tuple[float, float]],
    *,
    confidences: Sequence[float] | None = None,
    out_of_frame: Sequence[bool] | None = None,
    title: str = "Frame e pontos-chave",
):
    """Plot an RGB frame with numbered keypoints colored by confidence."""
    import matplotlib.pyplot as plt

    image = np.asarray(frame)
    points = np.asarray(image_points, dtype=float)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("frame must be an RGB image with shape (height, width, 3)")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("image_points must have shape (n, 2)")
    if confidences is None:
        scores = np.ones(len(points))
    else:
        scores = np.asarray(confidences, dtype=float)
        if len(scores) != len(points):
            raise ValueError("confidences must match image_points")
    if out_of_frame is None:
        outside = np.zeros(len(points), dtype=bool)
    else:
        outside = np.asarray(out_of_frame, dtype=bool)
        if len(outside) != len(points):
            raise ValueError("out_of_frame must match image_points")
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.imshow(image)
    scatter = axis.scatter(points[:, 0], points[:, 1], c=scores, cmap="viridis", vmin=0, vmax=1)
    if outside.any():
        axis.scatter(
            points[outside, 0],
            points[outside, 1],
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            label="Fora do quadro",
        )
    for index, (x, y) in enumerate(points):
        axis.annotate(
            str(index),
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            color="red" if outside[index] else "white",
        )
    figure.colorbar(scatter, ax=axis, label="Confiança")
    axis.set(title=title, xlabel="x (px)", ylabel="y (px)")
    if outside.any():
        axis.legend()
    axis.set_xlim(0, image.shape[1])
    axis.set_ylim(image.shape[0], 0)
    figure.tight_layout()
    return figure, axis


def plot_homography_on_pitch(
    matrix: np.ndarray,
    *,
    image_size: tuple[int, int] = (1280, 720),
    pitch_size: tuple[float, float] = (105.0, 68.0),
    title: str = "Projeção da homografia no campo",
):
    """Plot the image rectangle projected into canonical pitch coordinates."""
    import matplotlib.pyplot as plt

    from .metrics import project_points

    width, height = image_size
    length, pitch_width = pitch_size
    image_corners = np.array([[0, 0], [width, 0], [width, height], [0, height], [0, 0]])
    projected = project_points(image_corners, matrix)
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.plot(
        [0, length, length, 0, 0],
        [0, 0, pitch_width, pitch_width, 0],
        color="white",
        linewidth=2,
    )
    axis.plot(projected[:, 0], projected[:, 1], "o-", color="tab:orange", label="Imagem projetada")
    axis.scatter([length / 2], [pitch_width / 2], color="tab:red", label="Centro do campo")
    axis.set(
        title=title,
        xlabel="x (m)",
        ylabel="y (m)",
        xlim=(-5, length + 5),
        ylim=(pitch_width + 5, -5),
        aspect="equal",
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure, axis


def plot_homography_grid(
    matrix: np.ndarray,
    *,
    image_size: tuple[int, int] = (1280, 720),
    title: str = "Grade projetada pela homografia",
):
    """Plot image-space grid lines after projection to the pitch."""
    import matplotlib.pyplot as plt

    from .metrics import project_points

    width, height = image_size
    figure, axis = plt.subplots(figsize=(12, 7))
    for x in np.linspace(0, width, 9):
        points = np.column_stack((np.full(30, x), np.linspace(0, height, 30)))
        projected = project_points(points, matrix)
        axis.plot(projected[:, 0], projected[:, 1], color="tab:blue", alpha=0.55)
    for y in np.linspace(0, height, 6):
        points = np.column_stack((np.linspace(0, width, 30), np.full(30, y)))
        projected = project_points(points, matrix)
        axis.plot(projected[:, 0], projected[:, 1], color="tab:green", alpha=0.55)
    axis.set(title=title, xlabel="x (m)", ylabel="y (m)", aspect="equal")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure, axis
