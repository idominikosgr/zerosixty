from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from zerosixty.models import MemberRecord
from zerosixty.normalize import parse_datetime


def load_member_records(path: Path) -> list[MemberRecord]:
    """Load and normalize list-member metadata from the exporter CSV."""

    records: list[MemberRecord] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metadata = _safe_json_loads(row.get("metadata", ""))
            legacy = metadata.get("legacy", {}) if isinstance(metadata, dict) else {}
            records.append(
                MemberRecord(
                    member_id=row.get("id", ""),
                    screen_name=row.get("screen_name", ""),
                    name=_empty_to_none(row.get("name")),
                    description=_empty_to_none(row.get("description")),
                    created_at=_parse_iso_datetime(row.get("created_at")),
                    followers_count=_parse_int(row.get("followers_count")),
                    friends_count=_parse_int(row.get("friends_count")),
                    statuses_count=_parse_int(row.get("statuses_count")),
                    favourites_count=_parse_int(row.get("favourites_count")),
                    listed_count=_parse_int(row.get("listed_count")),
                    location=_empty_to_none(row.get("location")),
                    is_blue_verified=_parse_bool(row.get("is_blue_verified")),
                    protected=_parse_bool(row.get("protected")),
                    default_profile=_bool_or_none(legacy.get("default_profile")),
                    default_profile_image=_bool_or_none(legacy.get("default_profile_image")),
                    profile_description_language=_empty_to_none(
                        metadata.get("profile_description_language")
                        if isinstance(metadata, dict)
                        else None
                    ),
                )
            )
    return records


def load_export_rows(path: Path) -> list[dict[str, Any]]:
    """Load raw tweet rows from the twitter-web-exporter JSON bundle."""

    payload = json.loads(path.read_text())
    tables = payload.get("data", {}).get("data", [])
    if not isinstance(tables, list):
        raise ValueError(f"Unexpected exporter shape in {path}.")

    for table in tables:
        if table.get("tableName") == "tweets":
            rows = table.get("rows", [])
            if not isinstance(rows, list):
                raise ValueError(f"Exporter tweets table in {path} is not a list.")
            return rows

    raise ValueError(f"No tweets table found in {path}.")


def _safe_json_loads(value: str) -> Any:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _parse_int(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def _parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _bool_or_none(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    return None


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return parse_datetime(value)
