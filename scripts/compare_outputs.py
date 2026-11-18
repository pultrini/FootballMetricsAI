"""Compara duas saídas do `analyze` (antiga x refatorada) e lista TODAS as diferenças.

Uso:
    uv run python scripts/compare_outputs.py <pasta_antiga> <pasta_nova> [--allow-extra-new]

--allow-extra-new: não conta como erro chaves/colunas que existem só na saída nova
                   (útil para acréscimos intencionais, como schema_version).
"""

import json
import math
import sys
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

VOLATILE = {"created_at", "analysis_id", "elapsed_seconds"}  # ajuste ao seu report
MAX_PRINT = 60

errors: list[str] = []
notes: list[str] = []


def walk(old, new, path: str, allow_extra_new: bool) -> None:
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(old.keys() - new.keys()):
            errors.append(f"{path}/{key}: existe só no ANTIGO")
        for key in sorted(new.keys() - old.keys()):
            (notes if allow_extra_new else errors).append(f"{path}/{key}: existe só no NOVO")
        for key in sorted(old.keys() & new.keys()):
            if key not in VOLATILE:
                walk(old[key], new[key], f"{path}/{key}", allow_extra_new)
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            errors.append(f"{path}: tamanho {len(old)} != {len(new)}")
        for index, (a, b) in enumerate(zip(old, new)):
            walk(a, b, f"{path}[{index}]", allow_extra_new)
    elif isinstance(old, (int, float)) and isinstance(new, (int, float)) and not isinstance(old, bool):
        if not math.isclose(old, new, rel_tol=1e-9, abs_tol=1e-12):
            errors.append(f"{path}: {old} != {new}")
    elif old != new:
        errors.append(f"{path}: {old!r} != {new!r}")


def compare_parquet(old_file: Path, new_file: Path, allow_extra_new: bool) -> None:
    old_df, new_df = pl.read_parquet(old_file), pl.read_parquet(new_file)
    name = old_file.name
    only_old = set(old_df.columns) - set(new_df.columns)
    only_new = set(new_df.columns) - set(old_df.columns)
    if only_old:
        errors.append(f"{name}: colunas só no ANTIGO: {sorted(only_old)}")
    if only_new:
        (notes if allow_extra_new else errors).append(f"{name}: colunas só no NOVO: {sorted(only_new)}")
    common = [c for c in old_df.columns if c in new_df.columns]
    if old_df.height != new_df.height:
        errors.append(f"{name}: linhas {old_df.height} != {new_df.height}")
        return
    try:
        assert_frame_equal(old_df.select(common), new_df.select(common), check_exact=False, rtol=1e-9)
    except AssertionError as exc:
        errors.append(f"{name}: valores diferentes -> {str(exc).splitlines()[0]}")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    allow_extra_new = "--allow-extra-new" in sys.argv
    old, new = Path(args[0]), Path(args[1])

    walk(
        json.loads((old / "report.json").read_text()),
        json.loads((new / "report.json").read_text()),
        "report",
        allow_extra_new,
    )

    old_files = {p.name for p in (old / "data").glob("*.parquet")}
    new_files = {p.name for p in (new / "data").glob("*.parquet")}
    for name in sorted(old_files - new_files):
        errors.append(f"data/{name}: arquivo só no ANTIGO")
    for name in sorted(new_files - old_files):
        (notes if allow_extra_new else errors).append(f"data/{name}: arquivo só no NOVO")
    for name in sorted(old_files & new_files):
        compare_parquet(old / "data" / name, new / "data" / name, allow_extra_new)

    for line in notes:
        print("nota  ", line)
    for line in errors[:MAX_PRINT]:
        print("DIFF  ", line)
    if len(errors) > MAX_PRINT:
        print(f"... e mais {len(errors) - MAX_PRINT} diferenças")
    print(f"\n{len(errors)} diferença(s), {len(notes)} nota(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())