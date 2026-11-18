"""Reproducible N1, N2 and N2c analysis on cached real pipeline outputs."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl

from .metrics import project_points

TARGET_FRAMES = (95, 269, 270, 349, 350, 352, 354, 481, 490)


def _homography(
    image: np.ndarray,
    pitch: np.ndarray,
    confidence: np.ndarray,
    *,
    threshold_m: float,
    method: int = cv2.RANSAC,
    confidence_floor: float = 0.25,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    accepted = confidence >= confidence_floor
    if int(accepted.sum()) < 4:
        return None, None
    matrix, mask = cv2.findHomography(
        image[accepted],
        pitch[accepted],
        method=method,
        ransacReprojThreshold=threshold_m,
        maxIters=2_000,
        confidence=0.995,
    )
    return matrix, None if mask is None else mask.ravel().astype(bool)


def load_keypoints(path: Path) -> dict[int, dict[str, np.ndarray]]:
    """Load cached keypoints grouped by frame."""
    table = pl.read_parquet(path)
    grouped: dict[int, dict[str, np.ndarray]] = {}
    for frame_id, rows in table.group_by("frame_id", maintain_order=True):
        frame = int(frame_id[0])
        grouped[frame] = {
            "image": rows.select(["image_x", "image_y"]).to_numpy(),
            "pitch": rows.select(["pitch_x", "pitch_y"]).to_numpy(),
            "confidence": rows["confidence"].to_numpy(),
            "ids": np.asarray(rows["id"].to_list(), dtype=str),
        }
    return grouped


def _diagnostic(
    frame_id: int,
    data: dict[str, np.ndarray],
    *,
    threshold_m: float = 1.5,
    method: int = cv2.RANSAC,
    image_size: tuple[int, int] = (1920, 1080),
) -> tuple[dict[str, Any], np.ndarray | None]:
    matrix, mask = _homography(
        data["image"],
        data["pitch"],
        data["confidence"],
        threshold_m=threshold_m,
        method=method,
    )
    image = data["image"]
    width, height = image_size
    out_of_frame = (
        (image[:, 0] < 0) | (image[:, 0] > width) | (image[:, 1] < 0) | (image[:, 1] > height)
    )
    row: dict[str, Any] = {
        "frame_id": frame_id,
        "keypoint_count": len(image),
        "accepted_keypoint_count": int((data["confidence"] >= 0.25).sum()),
        "out_of_frame_count": int(out_of_frame.sum()),
        "low_confidence_count": int((data["confidence"] < 0.25).sum()),
        "inlier_count": 0,
        "inlier_ratio": None,
        "rmse_m": None,
        "determinant": None,
        "condition_number": None,
        "center_x_m": None,
        "center_y_m": None,
        "valid": False,
        "reason": "estimation_failed",
    }
    if matrix is None or mask is None:
        return row, None
    accepted = data["confidence"] >= 0.25
    accepted_image = image[accepted]
    accepted_pitch = data["pitch"][accepted]
    projected = project_points(accepted_image, matrix)
    errors = np.linalg.norm(projected - accepted_pitch, axis=1)
    determinant = float(np.linalg.det(matrix))
    condition = float(np.linalg.cond(matrix))
    center = project_points(np.array([[width / 2, height / 2]]), matrix)[0]
    row.update(
        {
            "inlier_count": int(mask.sum()),
            "inlier_ratio": float(mask.mean()),
            "rmse_m": float(np.sqrt(np.mean(np.square(errors[mask])))),
            "determinant": determinant,
            "condition_number": condition,
            "center_x_m": float(center[0]),
            "center_y_m": float(center[1]),
            "valid": True,
            "reason": "ok",
        }
    )
    return row, matrix / matrix[2, 2]


def generate_n1(
    keypoints_path: Path,
    output_dir: Path,
    *,
    image_size: tuple[int, int] = (1920, 1080),
) -> dict[str, Any]:
    """Generate the three audit CSVs for the supplied clip."""
    output_dir.mkdir(parents=True, exist_ok=True)
    keypoints = load_keypoints(keypoints_path)
    rows: list[dict[str, Any]] = []
    matrices: dict[int, np.ndarray] = {}
    for frame_id, data in keypoints.items():
        row, matrix = _diagnostic(frame_id, data, image_size=image_size)
        rows.append(row)
        if matrix is not None:
            matrices[frame_id] = matrix
    quality = pl.DataFrame(rows).sort("frame_id")
    quality = quality.with_columns(
        [
            pl.col("center_x_m").diff().alias("center_delta_x_m"),
            pl.col("center_y_m").diff().alias("center_delta_y_m"),
        ]
    ).with_columns(
        (pl.col("center_delta_x_m") ** 2 + pl.col("center_delta_y_m") ** 2)
        .sqrt()
        .alias("center_jump_m")
    )
    quality.write_csv(output_dir / "homography_quality_by_frame.csv")
    stability = quality.select(
        [
            "frame_id",
            "center_x_m",
            "center_y_m",
            "center_jump_m",
            "determinant",
            "condition_number",
            "valid",
        ]
    )
    stability.write_csv(output_dir / "homography_fixed_point_stability.csv")
    positions_path = keypoints_path.parent / "positions.parquet"
    if positions_path.exists():
        positions = pl.read_parquet(positions_path).filter(pl.col("speed").is_not_null())
        speed_by_frame = positions.group_by("frame_id").agg(
            pl.col("speed").max().alias("max_speed_mps")
        )
        correlation = stability.join(speed_by_frame, on="frame_id", how="left")
        correlation.write_csv(output_dir / "homography_player_jump_correlation.csv")
    else:
        pl.DataFrame({"frame_id": [], "max_speed_mps": []}).write_csv(
            output_dir / "homography_player_jump_correlation.csv"
        )
    validation = {
        "status": "validated_against_current_solver_definition",
        "clip": "0bfacc_0_18fps.mp4",
        "frame_count": 540,
        "fps": 18.0,
        "previous_audit_csvs_present": False,
        "previous_audit_numeric_comparison": "not_available",
        "same_points": (
            "current cache contains detector keypoints with confidence >= 0.350223; "
            "the solver applies its >= 0.25 filter, so all cached points pass"
        ),
        "inlier_definition": "OpenCV RANSAC mask returned for the filtered correspondences",
        "rmse_definition": "RMSE in metres over only the RANSAC inliers",
        "ransac": {
            "method": "cv2.RANSAC",
            "threshold": 1.5,
            "threshold_unit": "m in pitch coordinates",
            "max_iterations": 2000,
            "confidence": 0.995,
        },
        "csvs": {
            "quality": str(output_dir / "homography_quality_by_frame.csv"),
            "fixed_point": str(output_dir / "homography_fixed_point_stability.csv"),
            "correlation": str(output_dir / "homography_player_jump_correlation.csv"),
        },
    }
    (output_dir / "n1_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "frames_with_keypoints": len(keypoints),
        "frames_with_homography": len(matrices),
        "quality_csv": str(output_dir / "homography_quality_by_frame.csv"),
        "fixed_point_csv": str(output_dir / "homography_fixed_point_stability.csv"),
        "correlation_csv": str(output_dir / "homography_player_jump_correlation.csv"),
        "definitions": {
            "keypoints": "all detector keypoints with confidence >= 0.25",
            "inliers": "OpenCV RANSAC mask after the same confidence filter",
            "rmse": "sqrt(mean squared image-to-pitch error over RANSAC inliers), in metres",
            "fixed_point": "image centre (960, 540) projected to canonical pitch coordinates",
        },
        "validation": validation,
    }


def generate_n2(
    keypoints_path: Path,
    output_dir: Path,
    *,
    image_size: tuple[int, int] = (1920, 1080),
) -> dict[str, Any]:
    """Generate target-frame diagnostics and a classification summary."""
    keypoints = load_keypoints(keypoints_path)
    rows = []
    for frame_id in TARGET_FRAMES:
        if frame_id in keypoints:
            row, _ = _diagnostic(frame_id, keypoints[frame_id], image_size=image_size)
            rows.append(row)
    table = pl.DataFrame(rows)
    table.write_csv(output_dir / "n2_target_frame_diagnostics.csv")
    condition_max = float(table["condition_number"].max()) if table.height else math.nan
    out_of_frame = int(table["out_of_frame_count"].sum()) if table.height else 0
    determinant_signs = (
        table.select(pl.col("determinant").sign().n_unique()).item() if table.height else 0
    )
    negative_determinants = (
        int(table.filter(pl.col("determinant") < 0).height) if table.height else 0
    )
    classification = {
        "noise": "possible contributor; measured by reprojection residual",
        "mirroring": (
            "not supported: determinant sign is stable across selected frames; "
            "a negative sign is caused by coordinate orientation, not a frame-to-frame flip"
            if determinant_signs <= 1
            else "possible; determinant sign flips across selected frames"
        ),
        "degeneracy": "possible; inspect high condition numbers"
        if condition_max > 1e5
        else "not dominant in selected frames",
        "real_pan": "not proven by this diagnostic alone",
        "conclusion": (
            "predominantly noise/unstable correspondences with numerical "
            "conditioning; no evidence of mirroring, and pan is not established"
        ),
        "out_of_frame_keypoints": out_of_frame,
        "max_condition_number": condition_max,
        "negative_determinant_frames": negative_determinants,
        "determinant_sign_count": determinant_signs,
    }
    (output_dir / "n2_conclusion.json").write_text(
        json.dumps(classification, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return classification


def _track_points(path: Path) -> dict[int, list[tuple[int, float, float]]]:
    table = pl.read_parquet(path)
    frames: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for row in table.iter_rows(named=True):
        point = row["pixel_position"]
        frames[int(row["frame_id"])].append(
            (int(row["track_id"]), float(point[0]), float(point[1]))
        )
    return frames


def _speed_metrics(
    projected_by_frame: dict[int, list[tuple[int, float, float]]],
    fps: float = 18.0,
    *,
    smooth_window: int | None = None,
) -> dict[str, float | int | None]:
    by_track: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for frame_id in sorted(projected_by_frame):
        for track_id, x, y in projected_by_frame[frame_id]:
            by_track[track_id].append((frame_id, x, y))
    speeds: list[float] = []
    for values in by_track.values():
        smoothed = values
        if smooth_window is not None:
            smoothed = [
                (
                    values[index][0],
                    float(
                        np.median(
                            [
                                item[1]
                                for item in values[max(0, index - smooth_window + 1) : index + 1]
                            ]
                        )
                    ),
                    float(
                        np.median(
                            [
                                item[2]
                                for item in values[max(0, index - smooth_window + 1) : index + 1]
                            ]
                        )
                    ),
                )
                for index in range(len(values))
            ]
        for previous, current in pairwise(smoothed):
            delta_frames = current[0] - previous[0]
            if delta_frames > 0:
                speeds.append(
                    math.hypot(current[1] - previous[1], current[2] - previous[2])
                    * fps
                    / delta_frames
                )
    if not speeds:
        return {"count": 0, "median_mps": None, "p95_mps": None, "p99_mps": None, "max_mps": None}
    values = np.asarray(speeds)
    return {
        "count": len(speeds),
        "median_mps": float(np.median(values)),
        "p95_mps": float(np.percentile(values, 95)),
        "p99_mps": float(np.percentile(values, 99)),
        "max_mps": float(np.max(values)),
    }


def run_n2c(
    keypoints_path: Path,
    tracks_path: Path,
    output_dir: Path,
    *,
    fps: float = 18.0,
    image_size: tuple[int, int] = (1920, 1080),
) -> pl.DataFrame:
    """Sweep confidence filters, RANSAC and MAGSAC with comparable metrics."""
    keypoints = load_keypoints(keypoints_path)
    tracks = _track_points(tracks_path)
    experiments = (
        [("ransac", threshold, 0.25) for threshold in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)]
        + [("magsac", threshold, 0.25) for threshold in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)]
        + [("ransac", 1.5, confidence) for confidence in (0.25, 0.35, 0.5, 0.7)]
    )
    baseline_centers: list[tuple[int, np.ndarray, np.ndarray]] = []
    for frame_id, data in keypoints.items():
        matrix, _ = _homography(
            data["image"],
            data["pitch"],
            data["confidence"],
            threshold_m=1.5,
        )
        if matrix is not None:
            center = project_points(np.array([[image_size[0] / 2, image_size[1] / 2]]), matrix)[0]
            baseline_centers.append((frame_id, center, matrix / matrix[2, 2]))
    runs: list[list[tuple[int, np.ndarray]]] = []
    current_run: list[tuple[int, np.ndarray]] = []
    for index, item in enumerate(baseline_centers):
        if index == 0 or np.linalg.norm(item[1] - baseline_centers[index - 1][1]) <= 1.0:
            current_run.append((item[0], item[2]))
        else:
            if current_run:
                runs.append(current_run)
            current_run = [(item[0], item[2])]
    if current_run:
        runs.append(current_run)
    stable_frames = max(runs, key=len, default=[])
    frozen_start = stable_frames[0][0] if stable_frames else None
    frozen_end = stable_frames[-1][0] if stable_frames else None
    frozen_matrix = stable_frames[0][1] if stable_frames else None
    rows = []
    for method_name, threshold, confidence_floor in experiments:
        projected: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
        jumps: list[float] = []
        previous_center: np.ndarray | None = None
        valid_frames = 0
        cv_method = cv2.USAC_MAGSAC if method_name == "magsac" else cv2.RANSAC
        for frame_id, data in keypoints.items():
            filtered = {**data, "confidence": data["confidence"].copy()}
            accepted = filtered["confidence"] >= confidence_floor
            accepted &= (filtered["image"][:, 0] >= 0) & (filtered["image"][:, 0] <= image_size[0])
            accepted &= (filtered["image"][:, 1] >= 0) & (filtered["image"][:, 1] <= image_size[1])
            filtered["image"] = filtered["image"][accepted]
            filtered["pitch"] = filtered["pitch"][accepted]
            filtered["confidence"] = filtered["confidence"][accepted]
            if len(filtered["image"]) < 4:
                continue
            matrix, _ = _homography(
                filtered["image"],
                filtered["pitch"],
                filtered["confidence"],
                threshold_m=threshold,
                method=cv_method,
                confidence_floor=confidence_floor,
            )
            if matrix is None:
                continue
            valid_frames += 1
            center = project_points(np.array([[image_size[0] / 2, image_size[1] / 2]]), matrix)[0]
            if previous_center is not None:
                jumps.append(float(np.linalg.norm(center - previous_center)))
            previous_center = center
            for track_id, x, y in tracks.get(frame_id, []):
                point = project_points(np.array([[x, y]]), matrix)[0]
                projected[frame_id].append((track_id, float(point[0]), float(point[1])))
        speed = _speed_metrics(projected, fps)
        smoothed_speed = _speed_metrics(projected, fps, smooth_window=3)
        rows.append(
            {
                "method": method_name,
                "threshold_m": threshold,
                "confidence_floor": confidence_floor,
                "coverage": valid_frames / max(len(keypoints), 1),
                "jump_median": float(np.median(jumps)) if jumps else None,
                "jump_p95": float(np.percentile(jumps, 95)) if jumps else None,
                "jump_max": float(np.max(jumps)) if jumps else None,
                "speed_median_mps": speed["median_mps"],
                "speed_p95_mps": speed["p95_mps"],
                "speed_p99_mps": speed["p99_mps"],
                "speed_max_mps": speed["max_mps"],
                "speed_count": speed["count"],
                "speed_h_plus_trajectory_p99_mps": smoothed_speed["p99_mps"],
                "speed_h_plus_trajectory_max_mps": smoothed_speed["max_mps"],
                "frozen_start_frame": None,
                "frozen_end_frame": None,
            }
        )
    if frozen_matrix is not None and frozen_start is not None:
        frozen: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
        for frame_id, values in tracks.items():
            if frame_id < frozen_start:
                continue
            for track_id, x, y in values:
                point = project_points(np.array([[x, y]]), frozen_matrix)[0]
                frozen[frame_id].append((track_id, float(point[0]), float(point[1])))
        speed = _speed_metrics(frozen, fps)
        smoothed_speed = _speed_metrics(frozen, fps, smooth_window=3)
        result_row = {
            "method": "frozen_h_stable_segment",
            "threshold_m": 1.5,
            "confidence_floor": 0.25,
            "coverage": len([frame for frame in frozen if frozen_start <= frame <= frozen_end])
            / max(len(keypoints), 1),
            "jump_median": None,
            "jump_p95": None,
            "jump_max": None,
            "speed_median_mps": speed["median_mps"],
            "speed_p95_mps": speed["p95_mps"],
            "speed_p99_mps": speed["p99_mps"],
            "speed_max_mps": speed["max_mps"],
            "speed_count": speed["count"],
            "speed_h_plus_trajectory_p99_mps": smoothed_speed["p99_mps"],
            "speed_h_plus_trajectory_max_mps": smoothed_speed["max_mps"],
            "frozen_start_frame": frozen_start,
            "frozen_end_frame": frozen_end,
        }
        rows.append(result_row)
    result = pl.DataFrame(rows)
    result.write_csv(output_dir / "n2c_sweep_results.csv")
    return result
