from __future__ import annotations

from typing import TYPE_CHECKING

from zerosixty.models import DatasetPaths

if TYPE_CHECKING:
    from pathlib import Path


def discover_dataset(
    input_dir: Path,
    *,
    members_csv: Path | None = None,
    exporter_json: Path | None = None,
) -> DatasetPaths:
    """Resolve the input files for a run.

    If explicit paths are not provided, the newest matching files inside `input_dir`
    are selected.
    """

    resolved_members = members_csv or _latest_match(input_dir, "twitter-ListMembers-*.csv")
    resolved_exporter = exporter_json or _latest_match(input_dir, "twitter-web-exporter-*.json")
    return DatasetPaths(
        members_csv=resolved_members.resolve(),
        exporter_json=resolved_exporter.resolve(),
    )


def _latest_match(input_dir: Path, pattern: str) -> Path:
    matches = sorted(input_dir.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(
            f"No files matching {pattern!r} were found in {input_dir}."
        )
    return matches[-1]
