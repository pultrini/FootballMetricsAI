"""Generate characterization golden files from the current analyze command."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

TABLE_NAMES = (
    "positions",
    "possessions",
    "passes",
    "shots",
    "player_stats",
    "team_stats",
)
PROJECT_ROOT = Path(__file__).parents[1]
GOLDEN_ROOT = PROJECT_ROOT / "tests" / "characterization" / "golden"


def _path_from_argument_or_environment(argument: Path | None, environment_name: str) -> Path:
    value = argument or (
        Path(os.environ[environment_name]) if os.environ.get(environment_name) else None
    )
    if value is None:
        option = environment_name.removeprefix("FA_").lower().replace("_", "-")
        raise SystemExit(f"provide --{option} or {environment_name}")
    if not value.is_file():
        raise SystemExit(f"file does not exist: {value}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path)
    parser.add_argument("--object-model", type=Path)
    parser.add_argument("--pitch-model", type=Path)
    parser.add_argument("--xg-model", type=Path)
    args = parser.parse_args()
    clip = _path_from_argument_or_environment(args.clip, "FA_TEST_CLIP")
    object_model = _path_from_argument_or_environment(args.object_model, "FA_TEST_OBJECT_MODEL")
    pitch_model = _path_from_argument_or_environment(args.pitch_model, "FA_TEST_PITCH_MODEL")
    xg_model = args.xg_model or (
        Path(os.environ["FA_TEST_XG_MODEL"]) if os.environ.get("FA_TEST_XG_MODEL") else None
    )
    if xg_model is not None and not xg_model.is_file():
        raise SystemExit(f"file does not exist: {xg_model}")

    with tempfile.TemporaryDirectory() as temporary_directory:
        output = Path(temporary_directory) / "output"
        command = [
            "football-analytics",
            "analyze",
            str(clip),
            "--object-model",
            str(object_model),
            "--pitch-model",
            str(pitch_model),
            "--output",
            str(output),
            "--max-frames",
            "100",
            "--no-annotated-video",
        ]
        if xg_model is not None:
            command.extend(["--xg-model", str(xg_model)])
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

        GOLDEN_ROOT.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output / "report.json", GOLDEN_ROOT / "report.json")
        for name in TABLE_NAMES:
            shutil.copy2(
                output / "data" / f"{name}.parquet",
                GOLDEN_ROOT / f"{name}.parquet",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
