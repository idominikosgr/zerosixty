from __future__ import annotations

import math
from collections import Counter, defaultdict

from zerosixty.models import (
    AccountRole,
    CascadePropagationSummary,
    CascadeSpreadPath,
    CascadeSummary,
    FirstRetweeterProfile,
    NetworkNodeSummary,
    TweetRecord,
)


def build_cascade_propagation(
    tweets: list[TweetRecord],
    cascades: list[CascadeSummary],
    member_handles: set[str],
    network_nodes: list[NetworkNodeSummary],
    account_roles: list[AccountRole] | None,
    cohort_ids_by_cascade: dict[str, tuple[int, ...]] | None,
) -> tuple[list[CascadePropagationSummary], list[CascadeSpreadPath]]:
    """Compute detailed propagation metrics and ordered spread paths per cascade."""

    if not cascades:
        return [], []

    tweets_by_original: dict[str, list[TweetRecord]] = defaultdict(list)
    for tweet in tweets:
        if tweet.is_retweet:
            tweets_by_original[tweet.content_tweet_id].append(tweet)

    component_lookup = {node.account_handle: node.component_id for node in network_nodes}
    role_lookup = {role.account_handle: role for role in (account_roles or [])}
    cohort_lookup = cohort_ids_by_cascade or {}

    first_retweeter_counts: Counter[str] = Counter(
        cascade.first_retweeter
        for cascade in cascades
        if cascade.first_retweeter is not None
    )

    propagation_summaries: list[CascadePropagationSummary] = []
    spread_paths: list[CascadeSpreadPath] = []
    for cascade in cascades:
        retweets = sorted(
            tweets_by_original.get(cascade.original_tweet_id, []),
            key=lambda item: (
                item.posted_at is None,
                item.posted_at,
                item.author_handle,
            ),
        )
        client_counter: Counter[str] = Counter()
        steps: list[tuple[str, int, int, int]] = []
        elapsed_seconds_per_step: list[int] = []
        first_time = cascade.first_retweet_at
        for tweet in retweets:
            if tweet.source_client:
                client_counter[tweet.source_client] += 1
            if first_time is not None and tweet.posted_at is not None:
                elapsed = int((tweet.posted_at - first_time).total_seconds())
            else:
                elapsed = 0
            elapsed_seconds_per_step.append(elapsed)
            component_id = component_lookup.get(tweet.author_handle, 0)
            cohort_ids = cohort_lookup.get(cascade.original_tweet_id, ())
            primary_cohort = cohort_ids[0] if cohort_ids else 0
            steps.append((tweet.author_handle, elapsed, component_id, primary_cohort))

        total_retweets = len(retweets)
        time_to_k5 = _time_to_k(elapsed_seconds_per_step, 5)
        time_to_k10 = _time_to_k(elapsed_seconds_per_step, 10)
        time_to_k50 = _time_to_k(elapsed_seconds_per_step, 50)
        burst_kurtosis = _burst_kurtosis(elapsed_seconds_per_step)
        peak_window_start, peak_window_count = _burst_peak_window(
            elapsed_seconds_per_step,
            window_sec=60,
        )
        same_client_share = 0.0
        dominant_client: str | None = None
        if client_counter:
            dominant_client, dominant_count = client_counter.most_common(1)[0]
            same_client_share = dominant_count / max(total_retweets, 1)

        amplifier_components = tuple(
            sorted(
                {
                    component_lookup[tweet.author_handle]
                    for tweet in retweets
                    if tweet.author_handle in component_lookup
                }
            )
        )
        amplifier_cohorts = cohort_lookup.get(cascade.original_tweet_id, ())
        amplifier_counter: Counter[str] = Counter(
            tweet.author_handle for tweet in retweets
        )
        role_mix = _role_mix(amplifier_counter, role_lookup)

        propagation_score = _propagation_score(
            retweet_count=total_retweets,
            time_to_k10=time_to_k10,
            burst_kurtosis=burst_kurtosis,
            same_client_share=same_client_share,
            amplifier_cohort_count=len(amplifier_cohorts),
        )
        first_retweeter_recurrence = (
            first_retweeter_counts.get(cascade.first_retweeter, 0)
            if cascade.first_retweeter is not None
            else 0
        )

        preview_parts: list[str] = []
        for handle, elapsed_sec, component_id, cohort_id in steps[:5]:
            preview_parts.append(
                f"{handle}@+{elapsed_sec}s[c{component_id}/h{cohort_id}]"
            )

        propagation_summaries.append(
            CascadePropagationSummary(
                original_tweet_id=cascade.original_tweet_id,
                original_author_handle=cascade.original_author_handle,
                origin_is_member=cascade.original_author_handle in member_handles,
                retweet_count=total_retweets,
                unique_retweeter_count=cascade.unique_retweeter_count,
                first_retweeter=cascade.first_retweeter,
                first_retweeter_recurrence=first_retweeter_recurrence,
                time_to_k5_sec=time_to_k5,
                time_to_k10_sec=time_to_k10,
                time_to_k50_sec=time_to_k50,
                burst_kurtosis=round(burst_kurtosis, 4),
                burst_peak_window_start_sec=peak_window_start,
                burst_peak_window_count=peak_window_count,
                same_client_share=round(same_client_share, 4),
                dominant_client=dominant_client,
                amplifier_component_ids=amplifier_components,
                amplifier_cohort_ids=amplifier_cohorts,
                amplifier_role_mix=role_mix,
                propagation_score=round(propagation_score, 4),
                spread_path_preview=" -> ".join(preview_parts),
            )
        )
        spread_paths.append(
            CascadeSpreadPath(
                original_tweet_id=cascade.original_tweet_id,
                original_author_handle=cascade.original_author_handle,
                first_retweet_at=cascade.first_retweet_at,
                steps=tuple(steps),
            )
        )

    propagation_summaries.sort(
        key=lambda item: (
            -item.propagation_score,
            -item.retweet_count,
            item.original_tweet_id,
        )
    )
    spread_paths.sort(
        key=lambda item: (
            -(len(item.steps)),
            item.first_retweet_at is None,
            item.first_retweet_at,
            item.original_tweet_id,
        )
    )
    return propagation_summaries, spread_paths


def build_first_retweeter_profiles(
    cascades: list[CascadeSummary],
    account_roles: list[AccountRole] | None,
    cohort_ids_by_account: dict[str, tuple[int, ...]] | None,
    network_nodes: list[NetworkNodeSummary],
) -> list[FirstRetweeterProfile]:
    """Profile each account by how often and how tightly they are first on cascades."""

    if not cascades:
        return []

    role_lookup = {role.account_handle: role for role in (account_roles or [])}
    component_lookup = {node.account_handle: node.component_id for node in network_nodes}
    cohort_lookup = cohort_ids_by_account or {}

    by_first: dict[str, list[CascadeSummary]] = defaultdict(list)
    for cascade in cascades:
        if cascade.first_retweeter is None:
            continue
        by_first[cascade.first_retweeter].append(cascade)

    profiles: list[FirstRetweeterProfile] = []
    for handle, account_cascades in by_first.items():
        target_counter: Counter[str] = Counter(
            cascade.original_author_handle for cascade in account_cascades
        )
        top_target, top_target_count = (
            target_counter.most_common(1)[0] if target_counter else (None, 0)
        )
        target_hhi = _herfindahl(target_counter)
        lags = _first_to_second_lag_seconds(account_cascades)
        median_lag = _median_or_none(lags)
        role = role_lookup.get(handle)
        profiles.append(
            FirstRetweeterProfile(
                account_handle=handle,
                first_retweeter_count=len(account_cascades),
                unique_targets_first=len(target_counter),
                top_target_author=top_target,
                top_target_count=top_target_count,
                target_hhi=round(target_hhi, 4),
                median_lag_to_second_retweet_sec=median_lag,
                role_label=role.role_label if role is not None else "unknown",
                cohort_ids=cohort_lookup.get(handle, ()),
                component_id=component_lookup.get(handle),
            )
        )
    profiles.sort(
        key=lambda item: (
            -item.first_retweeter_count,
            -item.target_hhi,
            item.account_handle,
        )
    )
    return profiles


def _first_to_second_lag_seconds(cascades: list[CascadeSummary]) -> list[int]:
    lags: list[int] = []
    for cascade in cascades:
        if (
            cascade.first_retweet_at is None
            or cascade.last_retweet_at is None
            or cascade.retweet_count < 2
        ):
            continue
        span = int(
            (cascade.last_retweet_at - cascade.first_retweet_at).total_seconds()
        )
        if cascade.retweet_count <= 1:
            continue
        per_step = span // max(1, cascade.retweet_count - 1)
        lags.append(max(0, per_step))
    return lags


def _time_to_k(elapsed_seconds: list[int], k: int) -> int | None:
    if len(elapsed_seconds) < k:
        return None
    return elapsed_seconds[k - 1]


def _burst_kurtosis(elapsed_seconds: list[int]) -> float:
    """Population kurtosis of retweet timestamps; high = tight synchronized burst."""

    if len(elapsed_seconds) < 4:
        return 0.0
    mean = sum(elapsed_seconds) / len(elapsed_seconds)
    variance = sum((value - mean) ** 2 for value in elapsed_seconds) / len(elapsed_seconds)
    if variance == 0:
        return 0.0
    std = math.sqrt(variance)
    fourth_moment = sum(
        ((value - mean) / std) ** 4 for value in elapsed_seconds
    ) / len(elapsed_seconds)
    return fourth_moment - 3.0


def _burst_peak_window(
    elapsed_seconds: list[int],
    *,
    window_sec: int,
) -> tuple[int | None, int]:
    if not elapsed_seconds:
        return None, 0
    ordered = sorted(elapsed_seconds)
    best_start: int | None = None
    best_count = 0
    left = 0
    for right, _ in enumerate(ordered):
        while ordered[right] - ordered[left] > window_sec:
            left += 1
        count = right - left + 1
        if count > best_count:
            best_count = count
            best_start = ordered[left]
    return best_start, best_count


def _propagation_score(
    *,
    retweet_count: int,
    time_to_k10: int | None,
    burst_kurtosis: float,
    same_client_share: float,
    amplifier_cohort_count: int,
) -> float:
    if retweet_count <= 0:
        return 0.0
    reach_term = math.log1p(retweet_count) * 8.0
    speed_term = 0.0
    if time_to_k10 is not None:
        speed_term = max(0.0, 25.0 - min(time_to_k10, 3600) / 180.0)
    burst_term = max(0.0, min(burst_kurtosis, 10.0)) * 1.5
    client_term = same_client_share * 12.0
    cohort_term = math.log1p(amplifier_cohort_count) * 6.0
    return reach_term + speed_term + burst_term + client_term + cohort_term


def _role_mix(
    amplifier_counter: Counter[str],
    role_lookup: dict[str, AccountRole],
) -> str:
    if not amplifier_counter:
        return ""
    total = sum(amplifier_counter.values())
    role_counts: Counter[str] = Counter()
    for handle, count in amplifier_counter.items():
        role = role_lookup.get(handle)
        label = role.role_label if role is not None else "unknown"
        role_counts[label] += count
    ranked = role_counts.most_common(4)
    return "; ".join(
        f"{label}:{count / total:.2f}" for label, count in ranked if count > 0
    )


def _herfindahl(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return sum((count / total) ** 2 for count in counter.values())


def _median_or_none(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2
