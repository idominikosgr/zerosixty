from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse

from zerosixty.models import AccountProfile, TweetRecord

HTML_SOURCE_RE = re.compile(r">([^<]+)<")
HANDLE_FROM_RT_RE = re.compile(r"^RT @([A-Za-z0-9_]{1,32}):")
URL_RE = re.compile(r"https?://\S+")
TCO_DOMAINS = frozenset({"t.co", "twitter.com", "x.com", "www.twitter.com", "www.x.com"})


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
        in_reply_to_tweet_id = _string_or_none(_nested(row, "legacy", "in_reply_to_status_id_str"))
        in_reply_to_user_handle = _string_or_none(_nested(row, "legacy", "in_reply_to_screen_name"))
        is_reply = in_reply_to_tweet_id is not None

        content_payload = extract_retweeted_payload(row) if is_retweet else row
        text = extract_tweet_text(row)
        content_text = extract_tweet_text(content_payload)
        hashtags_raw, hashtags = extract_hashtags(content_payload)
        mentions_raw, mentions = extract_mentions(content_payload)
        urls = extract_urls(row)
        if is_retweet:
            urls = urls + extract_urls(content_payload)
        urls = _dedup_preserving(urls)
        url_domains = _dedup_preserving(_domain_of(url) for url in urls if _domain_of(url))

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

        quoted_payload = _resolve_quoted_payload(row)
        quoted_tweet_id = (
            _string_or_none(_nested(quoted_payload, "rest_id"))
            or _string_or_none(_nested(quoted_payload, "legacy", "id_str"))
            if quoted_payload is not None
            else None
        )
        quoted_author_handle = (
            _string_or_none(
                _nested(quoted_payload, "core", "user_results", "result", "core", "screen_name")
            )
            if quoted_payload is not None
            else None
        )

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
                is_reply=is_reply,
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
                urls=urls,
                url_domains=url_domains,
                lang=_string_or_none(_nested(row, "legacy", "lang")),
                favorite_count=_int_or_zero(_nested(row, "legacy", "favorite_count")),
                retweet_count_at_capture=_int_or_zero(_nested(row, "legacy", "retweet_count")),
                quote_count=_int_or_zero(_nested(row, "legacy", "quote_count")),
                reply_count=_int_or_zero(_nested(row, "legacy", "reply_count")),
                view_count=_int_or_none(_nested(row, "views", "count")),
                conversation_id=_string_or_none(_nested(row, "legacy", "conversation_id_str")),
                in_reply_to_tweet_id=in_reply_to_tweet_id,
                in_reply_to_user_handle=in_reply_to_user_handle,
                quoted_tweet_id=quoted_tweet_id,
                quoted_author_handle=quoted_author_handle,
            )
        )
    return records


def build_account_profiles(
    *,
    raw_rows: list[dict[str, Any]],
    users_table_rows: list[dict[str, Any]],
    member_handles: set[str],
) -> list[AccountProfile]:
    """Build a unified per-account profile registry from a single raw exporter.

    Sources, in priority order:
        1. ``users`` table (richest — userscript visited the profile directly)
        2. tweet author payloads (`core.user_results.result`)
        3. retweet content-author payloads (inside `legacy.retweeted_status_result`)
        4. quoted-tweet author payloads (inside `quoted_status_result`)

    Higher-priority sources overwrite lower-priority ones for any given handle.
    """

    folded_member_handles = {handle.casefold() for handle in member_handles}
    profiles: dict[str, AccountProfile] = {}

    def merge_profile(handle: str, payload: dict[str, Any], source: str) -> None:
        if not handle:
            return
        key = handle.casefold()
        existing = profiles.get(key)
        legacy = payload.get("legacy", {}) or {}
        core = payload.get("core", {}) or {}
        location_payload = payload.get("location") if isinstance(payload.get("location"), dict) else None
        location_text = location_payload.get("location") if location_payload else None

        candidate = AccountProfile(
            screen_name=handle,
            user_id=str(payload.get("rest_id") or legacy.get("id_str") or ""),
            display_name=_string_or_none(core.get("name") or legacy.get("name")),
            description=_string_or_none(legacy.get("description")),
            location=_string_or_none(location_text or legacy.get("location")),
            url=_string_or_none(legacy.get("url")),
            followers_count=_int_or_none(legacy.get("followers_count")),
            friends_count=_int_or_none(legacy.get("friends_count")),
            statuses_count=_int_or_none(legacy.get("statuses_count")),
            favourites_count=_int_or_none(legacy.get("favourites_count")),
            listed_count=_int_or_none(legacy.get("listed_count")),
            is_blue_verified=_bool_or_none(payload.get("is_blue_verified")),
            default_profile=_bool_or_none(legacy.get("default_profile")),
            default_profile_image=_bool_or_none(legacy.get("default_profile_image")),
            profile_description_language=_string_or_none(payload.get("profile_description_language")),
            created_at=parse_datetime(core.get("created_at") or legacy.get("created_at")),
            is_suspected_member=key in folded_member_handles,
            has_richest_profile=source == "users_table",
            sources=(source,),
        )
        if existing is None:
            profiles[key] = candidate
            return
        # Merge: keep best-non-None for each field, preserving sources history
        merged_sources = tuple(
            dict.fromkeys((*existing.sources, source)).keys()
        )
        profiles[key] = AccountProfile(
            screen_name=existing.screen_name or candidate.screen_name,
            user_id=existing.user_id or candidate.user_id,
            display_name=existing.display_name or candidate.display_name,
            description=existing.description or candidate.description,
            location=existing.location or candidate.location,
            url=existing.url or candidate.url,
            followers_count=_pick_richer(existing.followers_count, candidate.followers_count),
            friends_count=_pick_richer(existing.friends_count, candidate.friends_count),
            statuses_count=_pick_richer(existing.statuses_count, candidate.statuses_count),
            favourites_count=_pick_richer(existing.favourites_count, candidate.favourites_count),
            listed_count=_pick_richer(existing.listed_count, candidate.listed_count),
            is_blue_verified=existing.is_blue_verified if existing.is_blue_verified is not None else candidate.is_blue_verified,
            default_profile=existing.default_profile if existing.default_profile is not None else candidate.default_profile,
            default_profile_image=existing.default_profile_image if existing.default_profile_image is not None else candidate.default_profile_image,
            profile_description_language=existing.profile_description_language or candidate.profile_description_language,
            created_at=existing.created_at or candidate.created_at,
            is_suspected_member=existing.is_suspected_member or candidate.is_suspected_member,
            has_richest_profile=existing.has_richest_profile or candidate.has_richest_profile,
            sources=merged_sources,
        )

    for user_row in users_table_rows:
        sn = _string_or_none(_nested(user_row, "core", "screen_name"))
        if sn:
            merge_profile(sn, user_row, "users_table")

    for tweet in raw_rows:
        author = _nested(tweet, "core", "user_results", "result")
        if isinstance(author, dict):
            sn = _string_or_none(_nested(author, "core", "screen_name"))
            if sn:
                merge_profile(sn, author, "tweet_author")
        rt_payload = extract_retweeted_payload(tweet)
        if rt_payload is not tweet:
            rt_author = _nested(rt_payload, "core", "user_results", "result")
            if isinstance(rt_author, dict):
                sn = _string_or_none(_nested(rt_author, "core", "screen_name"))
                if sn:
                    merge_profile(sn, rt_author, "rt_target")
        quoted = _resolve_quoted_payload(tweet)
        if quoted is not None:
            q_author = _nested(quoted, "core", "user_results", "result")
            if isinstance(q_author, dict):
                sn = _string_or_none(_nested(q_author, "core", "screen_name"))
                if sn:
                    merge_profile(sn, q_author, "quote_target")

    # Add member handles that weren't captured in any payload (placeholder profiles)
    for handle in member_handles:
        key = handle.casefold()
        if key not in profiles:
            profiles[key] = AccountProfile(
                screen_name=handle,
                user_id="",
                display_name=None,
                description=None,
                location=None,
                url=None,
                followers_count=None,
                friends_count=None,
                statuses_count=None,
                favourites_count=None,
                listed_count=None,
                is_blue_verified=None,
                default_profile=None,
                default_profile_image=None,
                profile_description_language=None,
                created_at=None,
                is_suspected_member=True,
                has_richest_profile=False,
                sources=("member_only",),
            )

    return sorted(
        profiles.values(),
        key=lambda profile: (
            -(profile.followers_count or 0),
            profile.screen_name.casefold(),
        ),
    )


def _pick_richer(existing: int | None, candidate: int | None) -> int | None:
    """Prefer the larger value when both are present (newer captures usually have higher counts)."""

    if existing is None:
        return candidate
    if candidate is None:
        return existing
    return max(existing, candidate)


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


def extract_urls(payload: dict[str, Any]) -> tuple[str, ...]:
    """Return expanded URLs from the best available entity container."""

    entries = _entity_list(payload, "urls")
    out: list[str] = []
    for item in entries:
        expanded = item.get("expanded_url") or item.get("url")
        if isinstance(expanded, str) and expanded:
            out.append(expanded)
    return tuple(out)


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


def _resolve_quoted_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return the inner quoted-tweet payload, normalizing the optional `tweet` wrapper."""

    result = _nested(row, "quoted_status_result", "result")
    if not isinstance(result, dict):
        return None
    inner = result.get("tweet")
    if isinstance(inner, dict):
        return inner
    return result


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_or_zero(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "false"}:
        return text == "true"
    return None


def _dedup_preserving(values: Any) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


def _domain_of(url: str) -> str:
    """Extract a normalized domain from an URL, dropping `www.` and excluding self-redirect hosts."""

    try:
        netloc = urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""
    if not netloc or netloc in TCO_DOMAINS:
        return ""
    return netloc
