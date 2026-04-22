from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any

from zerosixty.models import TweetRecord

HTML_SOURCE_RE = re.compile(r">([^<]+)<")
HANDLE_FROM_RT_RE = re.compile(r"^RT @([A-Za-z0-9_]{1,32}):")
URL_RE = re.compile(r"https?://\S+")


def build_tweet_records(rows: list[dict[str, Any]]) -> list[TweetRecord]:
    """Flatten raw exporter rows into normalized tweet records."""

    records: list[TweetRecord] = []
    for row in rows:
        author_handle = _nested(
            row,
            "core",
            "user_results",
            "result",
            "core",
            "screen_name",
        )
        if not isinstance(author_handle, str) or not author_handle:
            continue

        author_name = _nested(
            row,
            "core",
            "user_results",
            "result",
            "core",
            "name",
        )
        source_client = strip_source_client(row.get("source"))
        is_retweet = bool(_nested(row, "legacy", "retweeted_status_result"))
        is_quote = bool(_nested(row, "legacy", "is_quote_status")) or bool(
            row.get("quoted_status_result")
        )

        content_payload = extract_retweeted_payload(row) if is_retweet else row
        text = extract_tweet_text(row)
        content_text = extract_tweet_text(content_payload)
        hashtags_raw, hashtags = extract_hashtags(content_payload)
        mentions_raw, mentions = extract_mentions(content_payload)

        content_author_handle = _nested(
            content_payload,
            "core",
            "user_results",
            "result",
            "core",
            "screen_name",
        )
        if not isinstance(content_author_handle, str) or not content_author_handle:
            content_author_handle = extract_retweet_handle_from_text(text) or author_handle

        content_author_name = _nested(
            content_payload,
            "core",
            "user_results",
            "result",
            "core",
            "name",
        )
        content_tweet_id = _nested(content_payload, "rest_id") or _nested(
            content_payload,
            "legacy",
            "id_str",
        )
        if not isinstance(content_tweet_id, str) or not content_tweet_id:
            content_tweet_id = row.get("rest_id", "")

        records.append(
            TweetRecord(
                tweet_id=str(row.get("rest_id", "")),
                author_handle=author_handle,
                author_name=author_name if isinstance(author_name, str) else None,
                posted_at=parse_datetime(_nested(row, "legacy", "created_at")),
                source_client=source_client,
                text=text,
                canonical_text=canonicalize_text(text),
                is_retweet=is_retweet,
                is_quote=is_quote,
                content_tweet_id=str(content_tweet_id),
                content_author_handle=content_author_handle,
                content_author_name=(
                    content_author_name if isinstance(content_author_name, str) else None
                ),
                content_created_at=parse_datetime(_nested(content_payload, "legacy", "created_at")),
                content_text=content_text,
                content_canonical_text=canonicalize_text(content_text),
                hashtags_raw=hashtags_raw,
                hashtags=hashtags,
                mentions_raw=mentions_raw,
                mentions=mentions,
            )
        )
    return records


def parse_datetime(value: object) -> datetime | None:
    """Parse either exporter RFC822 dates or ISO-like strings."""

    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None


def normalize_token(value: str) -> str:
    """Normalize a hashtag or mention token for cross-row grouping."""

    normalized = unicodedata.normalize("NFKD", value)
    stripped = "".join(char for char in normalized if not unicodedata.combining(char))
    return unicodedata.normalize("NFKC", stripped).casefold().replace("\u03c2", "\u03c3")


def canonicalize_text(value: str) -> str:
    """Normalize tweet text for duplicate or pattern checks."""

    no_urls = URL_RE.sub("", value or "")
    normalized = unicodedata.normalize("NFKC", no_urls).casefold()
    collapsed = " ".join(normalized.split())
    return collapsed.strip()


def strip_source_client(value: object) -> str | None:
    """Extract the human-readable source client from the HTML anchor field."""

    if not isinstance(value, str) or not value:
        return None
    match = HTML_SOURCE_RE.search(value)
    if match is None:
        return value
    return match.group(1).strip()


def extract_retweeted_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Resolve the original tweet payload from a retweet row."""

    result = _nested(row, "legacy", "retweeted_status_result", "result")
    if isinstance(result, dict):
        tweet = result.get("tweet")
        if isinstance(tweet, dict):
            return tweet
    if isinstance(result, dict):
        return result
    return row


def extract_retweet_handle_from_text(text: str) -> str | None:
    """Fallback parser for `RT @handle:` wrappers."""

    match = HANDLE_FROM_RT_RE.match(text or "")
    if match is None:
        return None
    return match.group(1)


def extract_tweet_text(payload: dict[str, Any]) -> str:
    """Return note-tweet text when available, otherwise fall back to `legacy.full_text`."""

    note_text = _nested(payload, "note_tweet", "note_tweet_results", "result", "text")
    if isinstance(note_text, str) and note_text.strip():
        return note_text
    full_text = _nested(payload, "legacy", "full_text")
    return full_text if isinstance(full_text, str) else ""


def extract_hashtags(payload: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return raw and normalized hashtags from the best available entity container."""

    entries = _entity_list(payload, "hashtags")
    raw = tuple(item.get("text", "") for item in entries if item.get("text"))
    normalized = tuple(normalize_token(item) for item in raw)
    return raw, normalized


def extract_mentions(payload: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return raw and normalized mention handles from the best available entity container."""

    entries = _entity_list(payload, "user_mentions")
    raw = tuple(item.get("screen_name", "") for item in entries if item.get("screen_name"))
    normalized = tuple(normalize_token(item) for item in raw)
    return raw, normalized


def _entity_list(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    note_entities = _nested(
        payload,
        "note_tweet",
        "note_tweet_results",
        "result",
        "entity_set",
        key,
    )
    if isinstance(note_entities, list):
        return [item for item in note_entities if isinstance(item, dict)]
    legacy_entities = _nested(payload, "legacy", "entities", key)
    if isinstance(legacy_entities, list):
        return [item for item in legacy_entities if isinstance(item, dict)]
    return []


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value
