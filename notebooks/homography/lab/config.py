"""Configuration and prerequisite reporting for the laboratory."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class LabConfig:
    """Paths and deterministic settings shared by every notebook."""

    clip: Path | None
    object_model: Path | None
    pitch_model: Path | None
    device: str = "cpu"
    seed: int = 7

    @classmethod
    def from_environment(cls) -> LabConfig:
        def existing(name: str) -> Path | None:
            value = os.environ.get(name)
            path = Path(value).expanduser() if value else None
            return path if path and path.exists() else None

        return cls(
            clip=existing("FA_CLIP"),
            object_model=existing("FA_OBJECT_MODEL"),
            pitch_model=existing("FA_PITCH_MODEL"),
        )

    def prerequisites(self) -> dict[str, bool]:
        return {
            "FA_CLIP": self.clip is not None,
            "FA_OBJECT_MODEL": self.object_model is not None,
            "FA_PITCH_MODEL": self.pitch_model is not None,
        }

    def missing(self) -> list[str]:
        return [name for name, present in self.prerequisites().items() if not present]


def file_hash(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a stable SHA-256 hash without loading a large artifact at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_directories() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
