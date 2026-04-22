from __future__ import annotations

import csv
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from zerosixty.models import MemberRecord, TweetRecord
from zerosixty.normalize import parse_datetime

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


def load_member_records(
    path: Path,
    *,
    extra_handles: Iterable[str] = (),
) -> list[MemberRecord]:
    """Load and normalize list-member metadata from a CSV or JSON export."""

    if path.suffix.lower() == ".csv":
        records = _load_member_csv_records(path)
    elif path.suffix.lower() == ".json":
        records = _load_member_json_records(path)
    else:
        raise ValueError(f"Unsupported member export format for {path}.")

    return _append_placeholder_members(records, extra_handles)


def load_extra_member_handles(path: Path) -> list[str]:
    """Load extra member handles from a text, JSON, or CSV file."""

    suffix = path.suffix.lower()
    if suffix in {"", ".txt"}:
        return _normalize_handles(path.read_text().splitlines())
    if suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return _normalize_handles(payload)
        if isinstance(payload, dict):
            for key in ("handles", "members", "accounts", "urls"):
                value = payload.get(key)
                if isinstance(value, list):
                    return _normalize_handles(value)
        raise ValueError(f"Unsupported extra-member JSON shape in {path}.")
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []
            for candidate in ("screen_name", "handle", "username", "url"):
                if candidate in reader.fieldnames:
                    return _normalize_handles(row.get(candidate) for row in reader)
            first_field = reader.fieldnames[0]
            return _normalize_handles(row.get(first_field) for row in reader)
    raise ValueError(f"Unsupported extra-member handle format for {path}.")


def _load_member_csv_records(path: Path) -> list[MemberRecord]:
    """Load list-member metadata from the original exporter CSV format."""

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


def _load_member_json_records(path: Path) -> list[MemberRecord]:
    """Load list-member metadata from the newer JSON list export format."""

    payload = json.loads(path.read_text())
    items = _extract_member_items(payload)
    return [_member_record_from_json_item(item) for item in items]


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


def load_clean_member_records(path: Path) -> list[MemberRecord]:
    """Load normalized members from a clean-batch `members.csv`."""

    records: list[MemberRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        records.extend(
            MemberRecord(
                member_id=row.get("member_id", ""),
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
                default_profile=_bool_or_none(row.get("default_profile")),
                default_profile_image=_bool_or_none(row.get("default_profile_image")),
                profile_description_language=_empty_to_none(
                    row.get("profile_description_language")
                ),
            )
            for row in reader
        )
    return records


def load_clean_tweet_records(path: Path) -> list[TweetRecord]:
    """Load normalized tweets from a clean-batch `tweets.csv`."""

    records: list[TweetRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        records.extend(
            TweetRecord(
                tweet_id=row.get("tweet_id", ""),
                author_handle=row.get("author_handle", ""),
                author_name=_empty_to_none(row.get("author_name")),
                posted_at=_parse_iso_datetime(row.get("posted_at")),
                source_client=_empty_to_none(row.get("source_client")),
                text=row.get("text", ""),
                canonical_text=row.get("canonical_text", ""),
                is_retweet=_parse_bool(row.get("is_retweet")),
                is_quote=_parse_bool(row.get("is_quote")),
                content_tweet_id=row.get("content_tweet_id", ""),
                content_author_handle=row.get("content_author_handle", ""),
                content_author_name=_empty_to_none(row.get("content_author_name")),
                content_created_at=_parse_iso_datetime(row.get("content_created_at")),
                content_text=row.get("content_text", ""),
                content_canonical_text=row.get("content_canonical_text", ""),
                hashtags_raw=_parse_tuple_json(row.get("hashtags_raw")),
                hashtags=_parse_tuple_json(row.get("hashtags")),
                mentions_raw=_parse_tuple_json(row.get("mentions_raw")),
                mentions=_parse_tuple_json(row.get("mentions")),
            )
            for row in reader
        )
    return records


def _safe_json_loads(value: str) -> Any:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def _parse_tuple_json(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    text = value.strip()
    if not text:
        return ()
    payload = _safe_json_loads(text)
    if isinstance(payload, list):
        return tuple(str(item) for item in payload if item is not None)
    return ()


def _extract_member_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("users", "members", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError("Unexpected list-member JSON shape.")


def _member_record_from_json_item(item: dict[str, Any]) -> MemberRecord:
    metadata = item.get("metadata", {})
    legacy = metadata.get("legacy", {}) if isinstance(metadata, dict) else {}
    return MemberRecord(
        member_id=str(item.get("id", "")),
        screen_name=str(item.get("screen_name", "")),
        name=_empty_to_none(_string_or_none(item.get("name"))),
        description=_empty_to_none(_string_or_none(item.get("description"))),
        created_at=_parse_iso_datetime(_string_or_none(item.get("created_at"))),
        followers_count=_parse_int(_string_or_none(item.get("followers_count"))),
        friends_count=_parse_int(_string_or_none(item.get("friends_count"))),
        statuses_count=_parse_int(_string_or_none(item.get("statuses_count"))),
        favourites_count=_parse_int(_string_or_none(item.get("favourites_count"))),
        listed_count=_parse_int(_string_or_none(item.get("listed_count"))),
        location=_empty_to_none(_string_or_none(item.get("location"))),
        is_blue_verified=_parse_bool(item.get("is_blue_verified")),
        protected=_parse_bool(item.get("protected")),
        default_profile=_bool_or_none(legacy.get("default_profile")),
        default_profile_image=_bool_or_none(legacy.get("default_profile_image")),
        profile_description_language=_empty_to_none(
            _string_or_none(
                metadata.get("profile_description_language")
                if isinstance(metadata, dict)
                else None
            )
        ),
    )


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


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


def _append_placeholder_members(
    records: list[MemberRecord],
    extra_handles: Iterable[str],
) -> list[MemberRecord]:
    merged = list(records)
    existing = {record.screen_name.casefold() for record in records if record.screen_name}
    for handle in _normalize_handles(extra_handles):
        folded = handle.casefold()
        if folded in existing:
            continue
        merged.append(
            MemberRecord(
                member_id=handle,
                screen_name=handle,
                name=None,
                description=None,
                created_at=None,
                followers_count=0,
                friends_count=0,
                statuses_count=0,
                favourites_count=0,
                listed_count=0,
                location=None,
                is_blue_verified=False,
                protected=False,
                default_profile=None,
                default_profile_image=None,
                profile_description_language=None,
            )
        )
        existing.add(folded)
    return merged


def _normalize_handles(values: Iterable[object]) -> list[str]:
    handles: list[str] = []
    seen: set[str] = set()
    for value in values:
        handle = _extract_handle(value)
        if handle is None:
            continue
        folded = handle.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        handles.append(handle)
    return handles


def _extract_handle(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.removeprefix("@")
    if text.startswith(("http://", "https://")):
        parsed = urlparse(text)
        segments = [segment for segment in parsed.path.split("/") if segment]
        if not segments:
            return None
        text = segments[0]
    text = text.strip().strip("/")
    if not text:
        return None
    return text
