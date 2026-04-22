from __future__ import annotations

import math
from collections import Counter, defaultdict

from zerosixty.models import (
    AccountRole,
    AmplifiedTargetSummary,
    CascadeSummary,
    NetworkNodeSummary,
    TweetRecord,
    UserCascadeSummary,
)


def build_amplified_targets(
    tweets: list[TweetRecord],
    cascades: list[CascadeSummary],
    user_cascades: list[UserCascadeSummary],
    member_handles: set[str],
    network_nodes: list[NetworkNodeSummary],
    account_roles: list[AccountRole] | None = None,
) -> list[AmplifiedTargetSummary]:
    """Build a ranking of targets that receive concentrated amplification.

    For each amplified author we compute:
      - total inbound retweets and unique amplifiers
      - HHI over amplifier handles (high = tight crew, low = organic)
      - repeat-crew overlap: Jaccard of amplifier sets across their top cascades
      - median first-retweet lag (how quickly faces show up)
      - cross-component reach (how many overlap-network components are involved)
      - amplifier role mix (share of media/business, retail, amplifier_suspect)
      - a composite `amplification_score` for ranking.
    """

    if not tweets or not cascades:
        return []

    component_lookup = {node.account_handle: node.component_id for node in network_nodes}
    role_lookup = {role.account_handle: role for role in (account_roles or [])}

    by_target: dict[str, list[TweetRecord]] = defaultdict(list)
    for tweet in tweets:
        if not tweet.is_retweet:
            continue
        if not tweet.content_author_handle:
            continue
        by_target[tweet.content_author_handle].append(tweet)

    cascades_by_target: dict[str, list[CascadeSummary]] = defaultdict(list)
    for cascade in cascades:
        cascades_by_target[cascade.original_author_handle].append(cascade)

    user_cascade_lookup = {uc.original_author_handle: uc for uc in user_cascades}

    summaries: list[AmplifiedTargetSummary] = []
    for target_handle, retweets in by_target.items():
        amplifier_counter: Counter[str] = Counter(
            tweet.author_handle for tweet in retweets
        )
        total_inbound = len(retweets)
        unique_amplifiers = len(amplifier_counter)
        top_amplifier, top_amplifier_count = (
            amplifier_counter.most_common(1)[0]
            if amplifier_counter
            else (None, 0)
        )
        top_share = top_amplifier_count / total_inbound if total_inbound else 0.0
        hhi = _herfindahl(amplifier_counter)
        target_cascades = cascades_by_target.get(target_handle, [])
        repeat_overlap = _repeat_crew_overlap(target_cascades)
        median_first_lag = _median_first_retweet_lag(target_cascades)
        fastest_amplifier = _fastest_amplifier(target_cascades)
        component_ids = {
            component_lookup[tweet.author_handle]
            for tweet in retweets
            if tweet.author_handle in component_lookup
        }
        user_cascade = user_cascade_lookup.get(target_handle)
        within_15m = user_cascade.total_retweets_within_15m if user_cascade else 0
        within_60m = user_cascade.total_retweets_within_60m if user_cascade else 0
        first_retweeter_counter: Counter[str] = Counter(
            cascade.first_retweeter
            for cascade in target_cascades
            if cascade.first_retweeter is not None
        )
        top_first_retweeter, top_first_count = (
            first_retweeter_counter.most_common(1)[0]
            if first_retweeter_counter
            else (None, 0)
        )
        role_mix = _role_mix(amplifier_counter, role_lookup)

        amplification_score = _amplification_score(
            total_inbound=total_inbound,
            unique_amplifiers=unique_amplifiers,
            hhi=hhi,
            repeat_overlap=repeat_overlap,
            within_15m=within_15m,
            cross_component=len(component_ids),
        )

        summaries.append(
            AmplifiedTargetSummary(
                amplified_author_handle=target_handle,
                in_member_list=target_handle in member_handles,
                total_inbound_retweets=total_inbound,
                unique_amplifier_count=unique_amplifiers,
                unique_amplified_posts=len(
                    {tweet.content_tweet_id for tweet in retweets}
                ),
                top_amplifier_handle=top_amplifier,
                top_amplifier_count=top_amplifier_count,
                top_amplifier_share=round(top_share, 4),
                amplification_hhi=round(hhi, 4),
                repeat_crew_overlap=round(repeat_overlap, 4),
                median_first_retweet_lag_sec=median_first_lag,
                fastest_amplifier_handle=fastest_amplifier,
                cross_component_reach=len(component_ids),
                within_15m_inbound=within_15m,
                within_60m_inbound=within_60m,
                top_first_retweeter_handle=top_first_retweeter,
                top_first_retweeter_count=top_first_count,
                captured_role_mix=role_mix,
                amplification_score=round(amplification_score, 4),
            )
        )

    summaries.sort(
        key=lambda item: (
            -item.amplification_score,
            -item.total_inbound_retweets,
            -item.amplification_hhi,
            item.amplified_author_handle,
        )
    )
    return summaries


def _repeat_crew_overlap(cascades: list[CascadeSummary]) -> float:
    """Mean pairwise Jaccard similarity over the top cascades' retweeter sets."""

    top = sorted(cascades, key=lambda item: -item.retweet_count)[:5]
    if len(top) < 2:
        return 0.0
    sets = [set(item.retweeters) for item in top]
    scores: list[float] = []
    for left_index in range(len(sets)):
        for right_index in range(left_index + 1, len(sets)):
            left = sets[left_index]
            right = sets[right_index]
            union = left | right
            if not union:
                continue
            scores.append(len(left & right) / len(union))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _median_first_retweet_lag(cascades: list[CascadeSummary]) -> int | None:
    lags: list[int] = []
    for cascade in cascades:
        if cascade.first_retweet_at is None or cascade.original_created_at is None:
            continue
        delta = (cascade.first_retweet_at - cascade.original_created_at).total_seconds()
        if delta < 0:
            continue
        lags.append(int(delta))
    if not lags:
        return None
    ordered = sorted(lags)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2


def _fastest_amplifier(cascades: list[CascadeSummary]) -> str | None:
    best: tuple[float, str] | None = None
    for cascade in cascades:
        if cascade.first_retweeter is None or cascade.first_retweet_at is None:
            continue
        if cascade.original_created_at is None:
            delta = 0.0
        else:
            delta = (
                cascade.first_retweet_at - cascade.original_created_at
            ).total_seconds()
        if delta < 0:
            continue
        candidate = (delta, cascade.first_retweeter)
        if best is None or candidate < best:
            best = candidate
    return best[1] if best is not None else None


def _role_mix(
    amplifier_counter: Counter[str],
    role_lookup: dict[str, AccountRole],
) -> str:
    """Render a compact share-of-roles string across amplifiers, weighted by retweet count."""

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


def _amplification_score(
    *,
    total_inbound: int,
    unique_amplifiers: int,
    hhi: float,
    repeat_overlap: float,
    within_15m: int,
    cross_component: int,
) -> float:
    """Rank targets by raw reach + crew concentration + speed + spread."""

    if total_inbound <= 0:
        return 0.0
    reach_term = math.log1p(total_inbound) * 8.0
    concentration_term = hhi * 25.0
    repeat_term = repeat_overlap * 20.0
    speed_term = math.log1p(within_15m) * 4.0
    spread_penalty = 0.0
    if cross_component >= 1 and unique_amplifiers > 0:
        spread_penalty = -min(cross_component / max(unique_amplifiers, 1), 1.0) * 5.0
    return reach_term + concentration_term + repeat_term + speed_term + spread_penalty
