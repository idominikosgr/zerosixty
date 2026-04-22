from __future__ import annotations

import re

from zerosixty.models import AccountRole, AccountSummary, MemberRecord

_MEDIA_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bnews\b",
        r"\bmedia\b",
        r"\bpress\b",
        r"\btv\b",
        r"\bradio\b",
        r"\bnewspaper\b",
        r"\bmagazine\b",
        r"\bjournal\b",
        r"\bofficial\b",
        r"\bchannel\b",
        r"\bbroadcast\b",
        r"\bbulletin\b",
        r"\bnewsroom\b",
        r"\bagency\b",
        r"\bgazette\b",
        r"\bpublisher\b",
        r"\btimes\b",
        r"\bherald\b",
        r"\btribune\b",
        r"\bdaily\b",
        r"\breport(er|ers)?\b",
        r"\bεφημερίδα\b",
        r"\bτηλεόραση\b",
        r"\bκανάλι\b",
        r"\bειδήσεις\b",
        r"\bδημοσιογρ",
    )
)

_BUSINESS_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bLtd\b",
        r"\bLLC\b",
        r"\bInc\b",
        r"\bGmbH\b",
        r"\bS\.A\.?\b",
        r"\bBV\b",
        r"\bCompany\b",
        r"\bCorp\b",
        r"\bAE\b",
        r"\bEPE\b",
        r"\bIKE\b",
    )
)

_JOURNALIST_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bjournalist\b",
        r"\breporter\b",
        r"\beditor\b",
        r"\bcorrespondent\b",
        r"\banchor\b",
        r"\bcolumnist\b",
        r"\bwriter\b",
        r"\bauthor\b",
        r"\bδημοσιογράφος\b",
        r"\bσυντάκτης\b",
        r"\bαρθρογράφος\b",
    )
)

_PUBLIC_FIGURE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bMP\b",
        r"\bminister\b",
        r"\bpolitician\b",
        r"\bmayor\b",
        r"\bpresident\b",
        r"\bsenator\b",
        r"\bCEO\b",
        r"\bfounder\b",
        r"\bprofessor\b",
        r"\bβουλευτής\b",
        r"\bυπουργός\b",
        r"\bδήμαρχος\b",
        r"\bπρόεδρος\b",
        r"\bκαθηγητής\b",
    )
)


def build_account_roles(
    members: list[MemberRecord],
    account_summaries: list[AccountSummary],
    cohort_ids_by_account: dict[str, tuple[int, ...]] | None = None,
) -> list[AccountRole]:
    """Classify each captured account into a review-oriented role.

    Roles are analyst navigation labels, not accusations. The classifier is
    deterministic and explainable. Each role carries a confidence score in
    [0, 1] and a list of signal strings that triggered the label.
    """

    member_lookup = {member.screen_name: member for member in members}
    cohort_lookup = cohort_ids_by_account or {}
    roles: list[AccountRole] = []

    for summary in account_summaries:
        member = member_lookup.get(summary.account_handle)
        description = (member.description if member is not None else None) or ""
        description_length = len(description.strip())
        followers = summary.followers_count or (member.followers_count if member else None)
        friends = summary.friends_count or (member.friends_count if member else None)
        listed = member.listed_count if member is not None else None
        followers_friends_ratio = _ratio(followers, friends)
        signals: list[str] = []

        looks_media = _text_matches(description, _MEDIA_PATTERNS)
        looks_business = _text_matches(description, _BUSINESS_PATTERNS)
        looks_journalist = _text_matches(description, _JOURNALIST_PATTERNS)
        looks_public_figure = _text_matches(description, _PUBLIC_FIGURE_PATTERNS)

        if looks_media:
            signals.append("description_media_keyword")
        if looks_business:
            signals.append("description_business_keyword")
        if looks_journalist:
            signals.append("description_journalist_keyword")
        if looks_public_figure:
            signals.append("description_public_figure_keyword")

        if listed is not None and listed >= 50:
            signals.append(f"listed_count>=50({listed})")
        if followers is not None and followers >= 100_000:
            signals.append(f"followers>=100k({followers})")
        if followers_friends_ratio is not None and followers_friends_ratio >= 25.0:
            signals.append(
                f"followers_friends_ratio>=25({followers_friends_ratio:.1f})"
            )
        if summary.is_blue_verified:
            signals.append("blue_verified")
        if summary.default_profile:
            signals.append("default_profile")
        if summary.default_profile_image:
            signals.append("default_profile_image")
        if summary.retweet_ratio >= 0.85 and summary.tweet_count >= 5:
            signals.append(f"retweet_ratio>=0.85({summary.retweet_ratio:.2f})")
        if summary.original_count <= 1 and summary.tweet_count >= 5:
            signals.append("zero_or_single_original")
        if summary.first_retweeter_count >= 3:
            signals.append(f"first_retweeter_count>=3({summary.first_retweeter_count})")
        if summary.top_amplified_share >= 0.6 and summary.retweet_count >= 5:
            signals.append(
                f"top_amplified_share>=0.6({summary.top_amplified_share:.2f})"
            )

        cohort_ids = cohort_lookup.get(summary.account_handle, ())
        if cohort_ids:
            signals.append(f"cohort_member({len(cohort_ids)})")

        role_label, confidence = _pick_role(
            summary=summary,
            member=member,
            followers=followers,
            friends=friends,
            listed=listed,
            followers_friends_ratio=followers_friends_ratio,
            looks_media=looks_media,
            looks_business=looks_business,
            looks_journalist=looks_journalist,
            looks_public_figure=looks_public_figure,
            cohort_ids=cohort_ids,
        )

        roles.append(
            AccountRole(
                account_handle=summary.account_handle,
                role_label=role_label,
                role_confidence=round(confidence, 4),
                is_member=summary.in_member_list,
                signals=tuple(signals),
                followers_count=followers,
                friends_count=friends,
                followers_friends_ratio=(
                    round(followers_friends_ratio, 4)
                    if followers_friends_ratio is not None
                    else None
                ),
                listed_count=listed,
                description_language=(
                    member.profile_description_language if member is not None else None
                ),
                description_length=description_length,
                is_blue_verified=summary.is_blue_verified,
                default_profile=summary.default_profile,
                account_age_days=_age_days(summary),
                retweet_ratio=summary.retweet_ratio,
                original_count=summary.original_count,
                cohort_ids=cohort_ids,
            )
        )

    roles.sort(
        key=lambda item: (
            _role_sort_key(item.role_label),
            -item.role_confidence,
            item.account_handle,
        )
    )
    return roles


_ROLE_PRIORITY: dict[str, int] = {
    "source_hub": 0,
    "media_business": 1,
    "journalist_public_figure": 2,
    "amplifier_suspect": 3,
    "retail_user": 4,
    "mixed_behavior": 5,
    "unknown": 6,
}


def _role_sort_key(label: str) -> int:
    return _ROLE_PRIORITY.get(label, 99)


def _pick_role(
    *,
    summary: AccountSummary,
    member: MemberRecord | None,
    followers: int | None,
    friends: int | None,
    listed: int | None,
    followers_friends_ratio: float | None,
    looks_media: bool,
    looks_business: bool,
    looks_journalist: bool,
    looks_public_figure: bool,
    cohort_ids: tuple[int, ...],
) -> tuple[str, float]:
    """Apply layered deterministic rules to assign one role."""

    if _is_source_hub(summary):
        return "source_hub", 0.9

    if _is_media_business(
        looks_media=looks_media,
        looks_business=looks_business,
        listed=listed,
        followers=followers,
        followers_friends_ratio=followers_friends_ratio,
    ):
        confidence = 0.5
        if looks_media or looks_business:
            confidence += 0.2
        if listed is not None and listed >= 100:
            confidence += 0.15
        if followers is not None and followers >= 250_000:
            confidence += 0.15
        return "media_business", min(confidence, 0.99)

    if _is_journalist_public_figure(
        looks_journalist=looks_journalist,
        looks_public_figure=looks_public_figure,
        summary=summary,
        followers=followers,
        followers_friends_ratio=followers_friends_ratio,
    ):
        confidence = 0.55
        if summary.is_blue_verified:
            confidence += 0.2
        if looks_journalist or looks_public_figure:
            confidence += 0.2
        return "journalist_public_figure", min(confidence, 0.99)

    if _is_amplifier_suspect(summary=summary, cohort_ids=cohort_ids):
        confidence = 0.55
        if summary.retweet_ratio >= 0.95:
            confidence += 0.1
        if summary.original_count == 0:
            confidence += 0.1
        if summary.top_amplified_share >= 0.5:
            confidence += 0.1
        if cohort_ids:
            confidence += 0.1
        if summary.first_retweeter_count >= 3:
            confidence += 0.05
        return "amplifier_suspect", min(confidence, 0.99)

    if _is_retail_user(followers=followers, listed=listed, summary=summary, member=member):
        return "retail_user", 0.6

    if summary.tweet_count >= 3:
        return "mixed_behavior", 0.4

    return "unknown", 0.2


def _is_source_hub(summary: AccountSummary) -> bool:
    return (
        summary.tweet_count >= 3
        and summary.retweet_ratio <= 0.3
        and summary.original_count >= 3
    )


def _is_media_business(
    *,
    looks_media: bool,
    looks_business: bool,
    listed: int | None,
    followers: int | None,
    followers_friends_ratio: float | None,
) -> bool:
    if looks_media or looks_business:
        return True
    if listed is not None and listed >= 50:
        return True
    if (
        followers is not None
        and followers >= 10_000
        and followers_friends_ratio is not None
        and followers_friends_ratio >= 25.0
    ):
        return True
    return False


def _is_journalist_public_figure(
    *,
    looks_journalist: bool,
    looks_public_figure: bool,
    summary: AccountSummary,
    followers: int | None,
    followers_friends_ratio: float | None,
) -> bool:
    if not (looks_journalist or looks_public_figure):
        return False
    if summary.is_blue_verified:
        return True
    if followers is not None and followers >= 5_000:
        return True
    if followers_friends_ratio is not None and followers_friends_ratio >= 3.0:
        return True
    return False


def _is_amplifier_suspect(
    *,
    summary: AccountSummary,
    cohort_ids: tuple[int, ...],
) -> bool:
    if summary.tweet_count < 5:
        return False
    if summary.retweet_ratio < 0.85:
        return False
    if summary.original_count > 1:
        return False
    if cohort_ids:
        return True
    if summary.top_amplified_share >= 0.5:
        return True
    if summary.first_retweeter_count >= 3:
        return True
    return False


def _is_retail_user(
    *,
    followers: int | None,
    listed: int | None,
    summary: AccountSummary,
    member: MemberRecord | None,
) -> bool:
    if member is None:
        return False
    if followers is None or followers >= 5_000:
        return False
    if listed is not None and listed >= 10:
        return False
    if summary.retweet_ratio >= 0.9 and summary.original_count == 0:
        return False
    return True


def _text_matches(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    if not text:
        return False
    return any(pattern.search(text) for pattern in patterns)


def _ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator <= 0:
        if numerator <= 0:
            return None
        return float("inf")
    return numerator / denominator


def _age_days(summary: AccountSummary) -> int | None:
    if summary.account_created_at is None:
        return None
    from datetime import UTC, datetime

    reference = datetime.now(UTC)
    delta = reference - summary.account_created_at
    return max(delta.days, 0)
