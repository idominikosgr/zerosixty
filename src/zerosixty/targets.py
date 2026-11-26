"""Inverted (target-first) analyses: target profiles, source-client fingerprints, tight-window cascades."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta

from zerosixty.models import (
    AccountProfile,
    AccountRole,
    CascadeSummary,
    CascadeTightWindowSummary,
    RetweetTargetProfile,
    SourceClientFingerprint,
    TweetRecord,
)

PYRAMID_TIER_THRESHOLDS = (
    (50, "tier1_high_reach"),
    (10, "tier2_mid_reach"),
    (3, "tier3_low_reach"),
)
TOP_AMP_LIST_SIZE = 10
OBSCURE_CLIENT_THRESHOLD = 100  # source_clients with <100 total tweets are "obscure"
AMPLIFIER_CONCENTRATION_THRESHOLD = 0.80


def build_retweet_target_profiles(
    tweets: list[TweetRecord],
    *,
    member_handles: set[str],
    account_profiles: list[AccountProfile],
    reference_time: datetime | None = None,
) -> list[RetweetTargetProfile]:
    """Build the inverted view: every retweet target with profile data + amplifier rollup.

    This is the pyramid-up perspective the existing pipeline lacked. Each row pairs
    a retweet target's profile (when known) with aggregate amplifier metrics and a
    member-vs-non-member split, so analysts can see *who* is being amplified,
    *by whom*, and at *what intensity* — not just which suspected accounts
    retweet a lot.
    """

    if not tweets:
        return []

    profile_lookup = {profile.screen_name.casefold(): profile for profile in account_profiles}
    folded_member_handles = {handle.casefold() for handle in member_handles}

    target_amps: dict[str, Counter[str]] = defaultdict(Counter)
    target_post_ids: dict[str, set[str]] = defaultdict(set)
    target_total: Counter[str] = Counter()

    for tweet in tweets:
        if not tweet.is_retweet or not tweet.content_author_handle:
            continue
        target_amps[tweet.content_author_handle][tweet.author_handle] += 1
        target_post_ids[tweet.content_author_handle].add(tweet.content_tweet_id)
        target_total[tweet.content_author_handle] += 1

    summaries: list[RetweetTargetProfile] = []
    for target_handle, amps in target_amps.items():
        total = target_total[target_handle]
        unique_amps = len(amps)
        member_amps = sum(1 for amp in amps if amp.casefold() in folded_member_handles)
        member_share = member_amps / unique_amps if unique_amps else 0.0
        top_amp, top_count = amps.most_common(1)[0] if amps else (None, 0)
        top_share = top_count / total if total else 0.0
        hhi = sum((count / total) ** 2 for count in amps.values()) if total else 0.0
        top10 = "|".join(f"{handle}:{count}" for handle, count in amps.most_common(TOP_AMP_LIST_SIZE))

        profile = profile_lookup.get(target_handle.casefold())
        age_days: int | None = None
        if profile is not None and profile.created_at is not None and reference_time is not None:
            age_days = max(0, (reference_time - profile.created_at).days)

        pyramid_position = _pyramid_tier(total)

        summaries.append(
            RetweetTargetProfile(
                target_handle=target_handle,
                target_is_suspected_member=target_handle.casefold() in folded_member_handles,
                target_has_rich_profile=bool(profile and profile.has_richest_profile),
                target_followers_count=profile.followers_count if profile else None,
                target_friends_count=profile.friends_count if profile else None,
                target_listed_count=profile.listed_count if profile else None,
                target_statuses_count=profile.statuses_count if profile else None,
                target_is_blue_verified=profile.is_blue_verified if profile else None,
                target_default_profile=profile.default_profile if profile else None,
                target_account_age_days=age_days,
                target_description=(profile.description or "")[:200] if profile else None,
                total_inbound_retweets=total,
                unique_amplifiers=unique_amps,
                unique_member_amplifiers=member_amps,
                unique_amplified_posts=len(target_post_ids[target_handle]),
                member_amplifier_share=round(member_share, 4),
                top_amplifier_handle=top_amp,
                top_amplifier_count=top_count,
                top1_share=round(top_share, 4),
                amplification_hhi=round(hhi, 4),
                top10_amplifiers=top10,
                target_pyramid_position=pyramid_position,
            )
        )

    summaries.sort(
        key=lambda item: (
            -item.total_inbound_retweets,
            -item.unique_amplifiers,
            item.target_handle,
        )
    )
    return summaries


def build_source_client_fingerprints(
    tweets: list[TweetRecord],
    *,
    member_handles: set[str],
    account_roles: list[AccountRole] | None = None,
) -> list[SourceClientFingerprint]:
    """Per-client usage profile, flagged when concentrated on suspected amplifiers.

    Twitter clients are usually broadly distributed (Web App, iPhone, Android).
    A client that's used by ≥5 accounts AND ≥80% of those accounts are
    `amplifier_suspect` is a strong coordination signal — bot toolkits often
    publish via custom-named clients.
    """

    if not tweets:
        return []

    folded_member_handles = {handle.casefold() for handle in member_handles}
    role_lookup = {role.account_handle: role for role in (account_roles or [])}

    by_client: dict[str, list[TweetRecord]] = defaultdict(list)
    for tweet in tweets:
        client = (tweet.source_client or "").strip()
        if not client:
            client = "(unknown)"
        by_client[client].append(tweet)

    summaries: list[SourceClientFingerprint] = []
    for client, client_tweets in by_client.items():
        total_tweets = len(client_tweets)
        per_account_counter: Counter[str] = Counter(t.author_handle for t in client_tweets)
        unique_accounts = len(per_account_counter)
        member_accounts = sum(
            1 for handle in per_account_counter if handle.casefold() in folded_member_handles
        )
        amplifier_accounts = sum(
            1
            for handle in per_account_counter
            if (role := role_lookup.get(handle)) is not None and role.role_label == "amplifier_suspect"
        )
        member_share = member_accounts / unique_accounts if unique_accounts else 0.0
        amp_share = amplifier_accounts / unique_accounts if unique_accounts else 0.0
        top_handle, top_count = per_account_counter.most_common(1)[0] if per_account_counter else (None, 0)
        top_share = top_count / total_tweets if total_tweets else 0.0
        is_obscure = total_tweets < OBSCURE_CLIENT_THRESHOLD
        is_amp_concentrated = unique_accounts >= 5 and amp_share >= AMPLIFIER_CONCENTRATION_THRESHOLD

        summaries.append(
            SourceClientFingerprint(
                source_client=client,
                total_tweet_count=total_tweets,
                unique_account_count=unique_accounts,
                unique_member_account_count=member_accounts,
                member_account_share=round(member_share, 4),
                top_account_handle=top_handle,
                top_account_count=top_count,
                top_account_share=round(top_share, 4),
                amplifier_account_count=amplifier_accounts,
                amplifier_account_share=round(amp_share, 4),
                is_obscure=is_obscure,
                is_amplifier_concentrated=is_amp_concentrated,
            )
        )

    # Sort: amplifier-concentrated obscure clients first, then by amplifier share, then volume
    summaries.sort(
        key=lambda item: (
            not item.is_amplifier_concentrated,
            not item.is_obscure,
            -item.amplifier_account_share,
            -item.unique_account_count,
            item.source_client,
        )
    )
    return summaries


def build_cascade_tight_windows(
    tweets: list[TweetRecord],
    cascades: list[CascadeSummary],
) -> list[CascadeTightWindowSummary]:
    """Per-cascade timing buckets at sub-minute granularity.

    Adds 5s/30s/60s/5m windows on top of the existing 15m/60m, so analysts can
    distinguish bot-burst patterns (sub-30-second co-RT clusters) from organic
    group reactions (typically 5–15min co-RT clusters).
    """

    if not tweets or not cascades:
        return []

    # Group retweets by content_tweet_id, ordered by posted_at
    retweets_by_target_id: dict[str, list[TweetRecord]] = defaultdict(list)
    for tweet in tweets:
        if not tweet.is_retweet:
            continue
        if not tweet.content_tweet_id:
            continue
        retweets_by_target_id[tweet.content_tweet_id].append(tweet)

    summaries: list[CascadeTightWindowSummary] = []
    for cascade in cascades:
        rts = sorted(
            (rt for rt in retweets_by_target_id.get(cascade.original_tweet_id, []) if rt.posted_at),
            key=lambda rt: rt.posted_at or datetime.min,
        )
        if not rts:
            continue
        first_rt_at = rts[0].posted_at
        if first_rt_at is None:
            continue

        # Count retweets occurring at increasing offsets from the FIRST retweet (not from origin)
        within_5s = within_30s = within_60s = within_5m = within_15m = within_60m = 0
        thresholds = (
            (timedelta(seconds=5), "within_5s"),
            (timedelta(seconds=30), "within_30s"),
            (timedelta(seconds=60), "within_60s"),
            (timedelta(minutes=5), "within_5m"),
            (timedelta(minutes=15), "within_15m"),
            (timedelta(minutes=60), "within_60m"),
        )
        for rt in rts:
            if rt.posted_at is None:
                continue
            delta = rt.posted_at - first_rt_at
            if delta <= thresholds[0][0]:
                within_5s += 1
            if delta <= thresholds[1][0]:
                within_30s += 1
            if delta <= thresholds[2][0]:
                within_60s += 1
            if delta <= thresholds[3][0]:
                within_5m += 1
            if delta <= thresholds[4][0]:
                within_15m += 1
            if delta <= thresholds[5][0]:
                within_60m += 1

        total_rt = len(rts)
        burst_5s = within_5s / total_rt if total_rt else 0.0
        burst_60s = within_60s / total_rt if total_rt else 0.0

        summaries.append(
            CascadeTightWindowSummary(
                original_tweet_id=cascade.original_tweet_id,
                original_author_handle=cascade.original_author_handle,
                retweet_count=total_rt,
                first_retweet_at=first_rt_at,
                retweets_within_5s=within_5s,
                retweets_within_30s=within_30s,
                retweets_within_60s=within_60s,
                retweets_within_5m=within_5m,
                retweets_within_15m=within_15m,
                retweets_within_60m=within_60m,
                burstiness_5s_share=round(burst_5s, 4),
                burstiness_60s_share=round(burst_60s, 4),
            )
        )

    # Rank by ABSOLUTE sub-5s retweet count first (1-RT cascades trivially have
    # share=1.0 and would otherwise dominate the top of the list).
    summaries.sort(
        key=lambda item: (
            -item.retweets_within_5s,
            -item.retweets_within_30s,
            -item.retweets_within_60s,
            -item.burstiness_5s_share,
            -item.retweet_count,
            item.original_tweet_id,
        )
    )
    return summaries


def _pyramid_tier(total_inbound: int) -> str:
    for threshold, label in PYRAMID_TIER_THRESHOLDS:
        if total_inbound >= threshold:
            return label
    return "tier4_minimal"
