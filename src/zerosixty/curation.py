from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from zerosixty.analyze import build_dataset_stats
from zerosixty.loaders import (
    load_export_rows,
    load_extra_member_handles,
    load_member_records,
)
from zerosixty.normalize import build_tweet_records

SNAPSHOT_ID_RE = re.compile(r"(\d{3,})(?=\.[^.]+$)")
TOKEN_SANITIZER_RE = re.compile(r"[^A-Za-z0-9]+")
CLEAN_SCHEMA_VERSION = 1


@dataclass(slots=True, frozen=True)
class SourceFile:
    kind: Literal["members", "exporter", "extra_members"]
    path: Path
    snapshot_id: int | None
    order_key: int


@dataclass(slots=True, frozen=True)
class CleanBatchPlan:
    batch_id: str
    exporter_file: SourceFile
    members_file: SourceFile
    extra_members_file: SourceFile | None


@dataclass(slots=True)
class CleanBatchResult:
    batch_id: str
    status: Literal["built", "skipped"]
    output_dir: Path
    exporter_file: Path
    members_file: Path
    extra_members_file: Path | None
    member_count: int
    tweet_count: int
    retweet_count: int
    original_count: int


@dataclass(slots=True, frozen=True)
class ResolvedCleanBatch:
    batch_id: str
    batch_dir: Path
    manifest_path: Path
    members_csv: Path
    tweets_csv: Path
    source_members_file: Path | None
    source_exporter_file: Path | None
    source_extra_members_file: Path | None


def discover_clean_batch_plans(
    raw_dir: Path,
    *,
    extra_members_file: Path | None = None,
) -> list[CleanBatchPlan]:
    """Create a deterministic build plan from raw snapshot files."""

    members = _discover_source_files(raw_dir, kind="members")
    exporters = _discover_source_files(raw_dir, kind="exporter")
    explicit_extra = extra_members_file is not None
    extras = (
        [_source_from_path("extra_members", extra_members_file.resolve())]
        if extra_members_file is not None
        else _discover_source_files(raw_dir, kind="extra_members")
    )

    if not members:
        raise FileNotFoundError(
            "No list-member snapshots found. Expected files like twitter-ListMembers-*.csv/.json."
        )
    if not exporters:
        raise FileNotFoundError(
            "No exporter snapshots found. Expected files like twitter-web-exporter-*.json."
        )

    plans: list[CleanBatchPlan] = []
    for exporter in exporters:
        member_match = _latest_not_newer(members, exporter.order_key) or members[-1]
        extra_match = (
            extras[0]
            if explicit_extra and extras
            else _latest_not_newer(extras, exporter.order_key)
        )
        plans.append(
            CleanBatchPlan(
                batch_id=_build_batch_id(exporter, member_match),
                exporter_file=exporter,
                members_file=member_match,
                extra_members_file=extra_match,
            )
        )
    return plans


def build_clean_batches(
    raw_dir: Path,
    clean_dir: Path,
    *,
    extra_members_file: Path | None = None,
    force: bool = False,
) -> tuple[list[CleanBatchResult], Path]:
    """Build versioned clean batches from raw snapshots and return the index path."""

    plans = discover_clean_batch_plans(raw_dir, extra_members_file=extra_members_file)
    clean_dir.mkdir(parents=True, exist_ok=True)

    results = [
        _materialize_clean_batch(plan, clean_dir=clean_dir, force=force)
        for plan in plans
    ]
    index_path = _write_clean_index(clean_dir, results)
    return results, index_path


def resolve_clean_batch(clean_dir: Path, *, batch_id: str | None = None) -> ResolvedCleanBatch:
    """Resolve one clean batch from `datasets-clean/index.json`."""

    index_path = clean_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"No clean index found at {index_path}. Run `zerosixty build-clean` first."
        )
    payload = json.loads(index_path.read_text())

    resolved_batch_id = batch_id or payload.get("latest_batch_id")
    if not isinstance(resolved_batch_id, str) or not resolved_batch_id:
        raise ValueError(
            "Unable to resolve clean batch id from index. Pass `--batch-id` explicitly."
        )

    batch_rows = payload.get("batches")
    if not isinstance(batch_rows, list):
        raise ValueError(f"Unexpected clean index shape in {index_path}.")

    selected: dict[str, Any] | None = None
    for row in batch_rows:
        if isinstance(row, dict) and row.get("batch_id") == resolved_batch_id:
            selected = row
            break
    if selected is None:
        raise ValueError(
            f"Batch id {resolved_batch_id!r} is not present in {index_path}."
        )

    output_dir = selected.get("output_dir")
    if isinstance(output_dir, str) and output_dir.strip():
        batch_dir = Path(output_dir).resolve()
    else:
        batch_dir = (clean_dir / resolved_batch_id).resolve()
    manifest_path = batch_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest for batch {resolved_batch_id!r} not found at {manifest_path}."
        )

    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files", {})
    members_name = _manifest_file_name(files, "members", default="members.csv")
    tweets_name = _manifest_file_name(files, "tweets", default="tweets.csv")

    members_csv = (batch_dir / members_name).resolve()
    tweets_csv = (batch_dir / tweets_name).resolve()
    if not members_csv.exists():
        raise FileNotFoundError(f"Clean members CSV is missing: {members_csv}")
    if not tweets_csv.exists():
        raise FileNotFoundError(f"Clean tweets CSV is missing: {tweets_csv}")

    sources = manifest.get("sources", {})
    return ResolvedCleanBatch(
        batch_id=resolved_batch_id,
        batch_dir=batch_dir,
        manifest_path=manifest_path,
        members_csv=members_csv,
        tweets_csv=tweets_csv,
        source_members_file=_optional_source_path(sources, "members"),
        source_exporter_file=_optional_source_path(sources, "exporter"),
        source_extra_members_file=_optional_source_path(sources, "extra_members"),
    )


def _materialize_clean_batch(
    plan: CleanBatchPlan,
    *,
    clean_dir: Path,
    force: bool,
) -> CleanBatchResult:
    output_dir = clean_dir / plan.batch_id
    manifest_path = output_dir / "manifest.json"
    source_fingerprints = _build_source_fingerprints(plan)

    if not force and manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text())
        if existing_manifest.get("sources") == source_fingerprints:
            counts = existing_manifest.get("counts", {})
            return CleanBatchResult(
                batch_id=plan.batch_id,
                status="skipped",
                output_dir=output_dir,
                exporter_file=plan.exporter_file.path,
                members_file=plan.members_file.path,
                extra_members_file=(
                    plan.extra_members_file.path
                    if plan.extra_members_file is not None
                    else None
                ),
                member_count=int(counts.get("member_count", 0)),
                tweet_count=int(counts.get("tweet_count", 0)),
                retweet_count=int(counts.get("retweet_count", 0)),
                original_count=int(counts.get("original_count", 0)),
            )

    extra_handles = (
        load_extra_member_handles(plan.extra_members_file.path)
        if plan.extra_members_file is not None
        else []
    )
    members = load_member_records(plan.members_file.path, extra_handles=extra_handles)
    raw_rows = load_export_rows(plan.exporter_file.path)
    tweets = build_tweet_records(raw_rows)
    stats = build_dataset_stats(members, tweets)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_dataclass_csv(members, output_dir / "members.csv")
    _write_dataclass_csv(tweets, output_dir / "tweets.csv")

    manifest_payload = {
        "schema_version": CLEAN_SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "batch_id": plan.batch_id,
        "sources": source_fingerprints,
        "counts": {
            "member_count": stats.member_count,
            "tweet_count": stats.tweet_count,
            "active_account_count": stats.active_account_count,
            "retweet_count": stats.retweet_count,
            "original_count": stats.original_count,
            "date_start": _format_datetime(stats.date_start),
            "date_end": _format_datetime(stats.date_end),
            "missing_member_handles": list(stats.missing_member_handles),
        },
        "files": {
            "members": "members.csv",
            "tweets": "tweets.csv",
        },
    }
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2))

    return CleanBatchResult(
        batch_id=plan.batch_id,
        status="built",
        output_dir=output_dir,
        exporter_file=plan.exporter_file.path,
        members_file=plan.members_file.path,
        extra_members_file=(
            plan.extra_members_file.path
            if plan.extra_members_file is not None
            else None
        ),
        member_count=stats.member_count,
        tweet_count=stats.tweet_count,
        retweet_count=stats.retweet_count,
        original_count=stats.original_count,
    )


def _write_clean_index(clean_dir: Path, results: list[CleanBatchResult]) -> Path:
    batches: list[dict[str, Any]] = []
    for result in results:
        manifest_path = result.output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        sources = manifest.get("sources", {})
        batches.append(
            {
                "batch_id": result.batch_id,
                "status": result.status,
                "output_dir": str(result.output_dir.resolve()),
                "exporter_snapshot_id": _nested_lookup(sources, "exporter", "snapshot_id"),
                "members_snapshot_id": _nested_lookup(sources, "members", "snapshot_id"),
                "extra_members_snapshot_id": _nested_lookup(
                    sources,
                    "extra_members",
                    "snapshot_id",
                ),
                "member_count": result.member_count,
                "tweet_count": result.tweet_count,
                "retweet_count": result.retweet_count,
                "original_count": result.original_count,
                "generated_at": manifest.get("generated_at"),
            }
        )

    payload = {
        "schema_version": CLEAN_SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "batch_count": len(results),
        "built_count": sum(1 for item in results if item.status == "built"),
        "skipped_count": sum(1 for item in results if item.status == "skipped"),
        "latest_batch_id": results[-1].batch_id if results else None,
        "batches": batches,
    }

    index_path = clean_dir / "index.json"
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return index_path


def _discover_source_files(raw_dir: Path, *, kind: Literal["members", "exporter", "extra_members"]) -> list[SourceFile]:
    patterns = {
        "members": ("twitter-ListMembers-*.csv", "twitter-ListMembers-*.json"),
        "exporter": ("twitter-web-exporter-*.json",),
        "extra_members": (
            "manual_member_handles_*.txt",
            "manual_member_handles_*.json",
            "manual_member_handles_*.csv",
        ),
    }[kind]

    discovered: list[SourceFile] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in raw_dir.glob(pattern):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(_source_from_path(kind, resolved))

    discovered.sort(key=lambda item: (item.order_key, item.path.name))
    return discovered


def _source_from_path(
    kind: Literal["members", "exporter", "extra_members"],
    path: Path,
) -> SourceFile:
    snapshot_id = _parse_snapshot_id(path)
    stat = path.stat()
    order_key = snapshot_id if snapshot_id is not None else stat.st_mtime_ns
    return SourceFile(
        kind=kind,
        path=path,
        snapshot_id=snapshot_id,
        order_key=order_key,
    )


def _latest_not_newer(items: list[SourceFile], order_key: int) -> SourceFile | None:
    eligible = [item for item in items if item.order_key <= order_key]
    if not eligible:
        return None
    return max(eligible, key=lambda item: (item.order_key, item.path.name))


def _parse_snapshot_id(path: Path) -> int | None:
    match = SNAPSHOT_ID_RE.search(path.name)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _build_batch_id(exporter_file: SourceFile, members_file: SourceFile) -> str:
    exporter_label = _file_label(exporter_file)
    members_label = _file_label(members_file)
    return f"batch_{exporter_label}__{members_label}"


def _file_label(source_file: SourceFile) -> str:
    if source_file.snapshot_id is not None:
        return f"{source_file.kind}-{source_file.snapshot_id}"
    normalized = TOKEN_SANITIZER_RE.sub("-", source_file.path.stem).strip("-").lower()
    return f"{source_file.kind}-{normalized[:40]}"


def _build_source_fingerprints(plan: CleanBatchPlan) -> dict[str, Any]:
    return {
        "members": _fingerprint_path(plan.members_file.path, plan.members_file.snapshot_id),
        "exporter": _fingerprint_path(plan.exporter_file.path, plan.exporter_file.snapshot_id),
        "extra_members": (
            _fingerprint_path(
                plan.extra_members_file.path,
                plan.extra_members_file.snapshot_id,
            )
            if plan.extra_members_file is not None
            else None
        ),
    }


def _fingerprint_path(path: Path, snapshot_id: int | None) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "snapshot_id": snapshot_id,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_dataclass_csv(rows: list[Any], path: Path) -> None:
    if not rows:
        path.write_text("")
        return

    first = rows[0]
    if not is_dataclass(first):
        raise TypeError("CSV export expects dataclass rows.")

    field_names = [field.name for field in fields(first)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            writer.writerow({name: _to_csv_value(payload.get(name)) for name in field_names})


def _to_csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return json.dumps(list(value), ensure_ascii=False)
    return value


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _format_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _nested_lookup(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _optional_source_path(sources: object, key: str) -> Path | None:
    if not isinstance(sources, dict):
        return None
    value = sources.get(key)
    if not isinstance(value, dict):
        return None
    path_value = value.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    return Path(path_value).resolve()


def _manifest_file_name(files: object, key: str, *, default: str) -> str:
    if not isinstance(files, dict):
        return default
    value = files.get(key)
    if not isinstance(value, str) or not value.strip():
        return default
    return value
