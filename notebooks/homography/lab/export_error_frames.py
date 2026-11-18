"""Export the worst frames from the homography audit for external labeling."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd


def export_error_frames(
    audit_csv: Path,
    video_path: Path,
    output_dir: Path,
    *,
    limit: int = 50,
) -> pd.DataFrame:
    """Export frames ranked by temporal and geometric homography errors."""
    audit = pd.read_csv(audit_csv)
    required = {
        "frame_id",
        "rmse_m",
        "center_jump_m",
        "condition_number",
        "inlier_ratio",
    }
    missing = required.difference(audit.columns)
    if missing:
        raise ValueError(f"Audit CSV is missing columns: {sorted(missing)}")
    if limit < 1:
        raise ValueError("limit must be positive")

    ranked = audit.dropna(subset=list(required - {"frame_id"})).copy()
    ranked["rank_jump"] = ranked["center_jump_m"].rank(ascending=False, method="min")
    ranked["rank_rmse"] = ranked["rmse_m"].rank(ascending=False, method="min")
    ranked["rank_condition"] = ranked["condition_number"].rank(
        ascending=False, method="min"
    )
    ranked["rank_inlier"] = ranked["inlier_ratio"].rank(ascending=True, method="min")
    ranked["error_rank_score"] = ranked[
        ["rank_jump", "rank_rmse", "rank_condition", "rank_inlier"]
    ].mean(axis=1)
    ranked = ranked.sort_values(
        ["error_rank_score", "frame_id"], ascending=[True, True]
    ).head(limit)

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        exported: list[int] = []
        for frame_id in ranked["frame_id"].astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            success, frame = capture.read()
            if not success or frame is None:
                raise RuntimeError(f"Unable to read frame {frame_id}")
            destination = output_dir / f"frame_{frame_id:04d}.jpg"
            if not cv2.imwrite(str(destination), frame):
                raise RuntimeError(f"Unable to write {destination}")
            exported.append(frame_id)
    finally:
        capture.release()

    if len(exported) != len(ranked):
        raise RuntimeError("The number of exported frames does not match the ranking")
    ranked.insert(1, "filename", [f"frame_{frame_id:04d}.jpg" for frame_id in exported])
    index_path = output_dir / "index.csv"
    ranked.to_csv(index_path, index=False)
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    exported = export_error_frames(
        args.audit_csv,
        args.video,
        args.output_dir,
        limit=args.limit,
    )
    print(f"Exported {len(exported)} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
