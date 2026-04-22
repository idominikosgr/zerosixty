from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import TYPE_CHECKING

from zerosixty.models import (
    AccountRole,
    AccountSummary,
    CascadePropagationSummary,
    CascadeSummary,
    CohortSummary,
    DatasetStats,
    FeatureRow,
    MemberRecord,
    NetworkComponentSummary,
    NetworkNodeSummary,
    OverlapSummary,
    RetweetEdgeSummary,
    TokenSummary,
    TweetRecord,
    UserCascadeSummary,
)

if TYPE_CHECKING:
    from datetime import datetime


def build_dataset_stats(
    members: list[MemberRecord],
    tweets: list[TweetRecord],
) -> DatasetStats:
    """Build dataset-level counts and date coverage."""

    member_handles = {member.screen_name for member in members}
    active_handles = {tweet.author_handle for tweet in tweets}
    posted_times = [tweet.posted_at for tweet in tweets if tweet.posted_at is not None]
    missing = tuple(sorted(active_handles - member_handles))
    retweet_count = sum(1 for tweet in tweets if tweet.is_retweet)
    return DatasetStats(
        member_count=len(members),
        tweet_count=len(tweets),
        active_account_count=len(active_handles),
        retweet_count=retweet_count,
        original_count=len(tweets) - retweet_count,
        date_start=min(posted_times) if posted_times else None,
        date_end=max(posted_times) if posted_times else None,
        missing_member_handles=missing,
    )


def build_cascade_summaries(tweets: list[TweetRecord]) -> list[CascadeSummary]:
    """Group retweets by original tweet and summarize the resulting cascades."""

    grouped: dict[str, list[TweetRecord]] = defaultdict(list)
    for tweet in tweets:
        if tweet.is_retweet:
            grouped[tweet.content_tweet_id].append(tweet)

    summaries: list[CascadeSummary] = []
    for content_tweet_id, items in grouped.items():
        ordered = sorted(
            items,
            key=lambda item: (
                item.posted_at is None,
                item.posted_at,
                item.author_handle,
            ),
        )
        first = ordered[0]
        last = ordered[-1]
        first_time = first.posted_at
        retweets_within_15m = 0
        retweets_within_60m = 0
        if first_time is not None:
            for item in ordered:
                if item.posted_at is None:
                    continue
                delta_seconds = (item.posted_at - first_time).total_seconds()
                if delta_seconds <= 900:
                    retweets_within_15m += 1
                if delta_seconds <= 3600:
                    retweets_within_60m += 1

        unique_retweeters = tuple(dict.fromkeys(item.author_handle for item in ordered))
        span_minutes = None
        if first.posted_at is not None and last.posted_at is not None:
            span_minutes = int((last.posted_at - first.posted_at).total_seconds() // 60)

        summaries.append(
            CascadeSummary(
                original_tweet_id=content_tweet_id,
                original_author_handle=first.content_author_handle,
                original_created_at=first.content_created_at,
                retweet_count=len(ordered),
                unique_retweeter_count=len(set(unique_retweeters)),
                first_retweeter=first.author_handle,
                first_retweet_at=first.posted_at,
                last_retweet_at=last.posted_at,
                span_minutes=span_minutes,
                retweets_within_15m=retweets_within_15m,
                retweets_within_60m=retweets_within_60m,
                hashtags=tuple(dict.fromkeys(first.hashtags)),
                mentions=tuple(dict.fromkeys(first.mentions)),
                retweeters=unique_retweeters,
                sample_text=first.content_text.replace("\n", " ").strip(),
            )
        )

    return sorted(
        summaries,
        key=lambda item: (-item.retweet_count, item.first_retweet_at or item.original_created_at),
    )


def build_user_cascade_summaries(cascades: list[CascadeSummary]) -> list[UserCascadeSummary]:
    """Aggregate post-level cascades into per-source-account retweet totals."""

    grouped: dict[str, list[CascadeSummary]] = defaultdict(list)
    for cascade in cascades:
        grouped[cascade.original_author_handle].append(cascade)

    summaries: list[UserCascadeSummary] = []
    for original_author_handle, items in grouped.items():
        top_cascade = max(
            items,
            key=lambda item: (
                item.retweet_count,
                item.first_retweet_at or item.original_created_at,
                item.original_tweet_id,
            ),
        )
        first_retweet_at = min(
            (item.first_retweet_at for item in items if item.first_retweet_at is not None),
            default=None,
        )
        last_retweet_at = max(
            (item.last_retweet_at for item in items if item.last_retweet_at is not None),
            default=None,
        )
        unique_retweeters = {
            retweeter
            for item in items
            for retweeter in item.retweeters
        }
        summaries.append(
            UserCascadeSummary(
                original_author_handle=original_author_handle,
                total_retweet_count=sum(item.retweet_count for item in items),
                unique_retweeted_tweet_count=len(items),
                unique_retweeter_count=len(unique_retweeters),
                first_retweet_at=first_retweet_at,
                last_retweet_at=last_retweet_at,
                total_retweets_within_15m=sum(item.retweets_within_15m for item in items),
                total_retweets_within_60m=sum(item.retweets_within_60m for item in items),
                top_cascade_tweet_id=top_cascade.original_tweet_id,
                top_cascade_retweet_count=top_cascade.retweet_count,
            )
        )

    return sorted(
        summaries,
        key=lambda item: (
            -item.total_retweet_count,
            -item.unique_retweeter_count,
            item.original_author_handle,
        ),
    )


def build_token_summaries(
    tweets: list[TweetRecord],
    *,
    token_type: str,
) -> list[TokenSummary]:
    """Build hashtag or mention frequency tables."""

    if token_type not in {"hashtags", "mentions"}:
        raise ValueError(f"Unsupported token type {token_type!r}.")

    counts: Counter[str] = Counter()
    raw_display: dict[str, str] = {}
    accounts: dict[str, set[str]] = defaultdict(set)

    for tweet in tweets:
        tokens = tweet.hashtags if token_type == "hashtags" else tweet.mentions
        raw_tokens = tweet.hashtags_raw if token_type == "hashtags" else tweet.mentions_raw
        for normalized, raw in zip(tokens, raw_tokens, strict=False):
            counts[normalized] += 1
            accounts[normalized].add(tweet.author_handle)
            raw_display.setdefault(normalized, raw)

    summaries = [
        TokenSummary(
            token=token,
            display_token=raw_display.get(token, token),
            count=count,
            unique_account_count=len(accounts[token]),
            sample_accounts=tuple(sorted(accounts[token])[:10]),
        )
        for token, count in counts.items()
    ]
    return sorted(summaries, key=lambda item: (-item.count, item.token))


def build_retweet_edges(tweets: list[TweetRecord]) -> list[RetweetEdgeSummary]:
    """Aggregate retweeter -> source-author counts."""

    counts: Counter[tuple[str, str]] = Counter()
    for tweet in tweets:
        if not tweet.is_retweet:
            continue
        counts[(tweet.author_handle, tweet.content_author_handle)] += 1

    summaries = [
        RetweetEdgeSummary(retweeter=retweeter, source_author=source_author, count=count)
        for (retweeter, source_author), count in counts.items()
    ]
    return sorted(summaries, key=lambda item: (-item.count, item.retweeter, item.source_author))


def build_overlap_summaries(
    tweets: list[TweetRecord],
    *,
    min_shared: int,
) -> list[OverlapSummary]:
    """Measure how often two accounts retweet the same original."""

    by_original: dict[str, list[TweetRecord]] = defaultdict(list)
    per_user_retweets: dict[str, set[str]] = defaultdict(set)
    pair_shared: dict[tuple[str, str], set[str]] = defaultdict(set)
    pair_within_15m: Counter[tuple[str, str]] = Counter()
    pair_within_60m: Counter[tuple[str, str]] = Counter()

    for tweet in tweets:
        if not tweet.is_retweet:
            continue
        by_original[tweet.content_tweet_id].append(tweet)
        per_user_retweets[tweet.author_handle].add(tweet.content_tweet_id)

    for original_tweet_id, items in by_original.items():
        ordered = sorted(
            items,
            key=lambda item: (
                item.posted_at is None,
                item.posted_at,
                item.author_handle,
            ),
        )
        for left_tweet, right_tweet in combinations(ordered, 2):
            pair = _sorted_pair(left_tweet.author_handle, right_tweet.author_handle)
            pair_shared[pair].add(original_tweet_id)
            if left_tweet.posted_at is None or right_tweet.posted_at is None:
                continue
            lag_seconds = abs((right_tweet.posted_at - left_tweet.posted_at).total_seconds())
            if lag_seconds <= 900:
                pair_within_15m[pair] += 1
            if lag_seconds <= 3600:
                pair_within_60m[pair] += 1

    summaries: list[OverlapSummary] = []
    for pair, shared_ids in pair_shared.items():
        shared_count = len(shared_ids)
        if shared_count < min_shared:
            continue
        account_a, account_b = pair
        union_size = len(per_user_retweets[account_a] | per_user_retweets[account_b])
        jaccard = shared_count / union_size if union_size else 0.0
        summaries.append(
            OverlapSummary(
                account_a=account_a,
                account_b=account_b,
                shared_retweets=shared_count,
                jaccard=round(jaccard, 4),
                shared_within_15m=pair_within_15m[pair],
                shared_within_60m=pair_within_60m[pair],
            )
        )

    return sorted(
        summaries,
        key=lambda item: (
            -item.shared_retweets,
            -item.shared_within_15m,
            item.account_a,
            item.account_b,
        ),
    )


def build_account_summaries(
    members: list[MemberRecord],
    tweets: list[TweetRecord],
    cascades: list[CascadeSummary],
) -> list[AccountSummary]:
    """Build per-account coordination summaries."""

    member_lookup = {member.screen_name: member for member in members}
    member_handles = set(member_lookup)
    first_retweeters = Counter(
        cascade.first_retweeter
        for cascade in cascades
        if cascade.first_retweeter is not None
    )
    per_account: dict[str, list[TweetRecord]] = defaultdict(list)
    for tweet in tweets:
        per_account[tweet.author_handle].append(tweet)

    summaries: list[AccountSummary] = []
    for account_handle, account_tweets in per_account.items():
        retweets = [tweet for tweet in account_tweets if tweet.is_retweet]
        originals = [tweet for tweet in account_tweets if not tweet.is_retweet]
        source_counter = Counter(tweet.content_author_handle for tweet in retweets)
        unique_retweeted_tweet_count = len({tweet.content_tweet_id for tweet in retweets})
        unique_retweeted_author_count = len(source_counter)
        retweets_to_member_count = sum(
            1 for tweet in retweets if tweet.content_author_handle in member_handles
        )
        hashtag_count = sum(len(tweet.hashtags) for tweet in account_tweets)
        mention_count = sum(len(tweet.mentions) for tweet in account_tweets)
        top_amplified_account = None
        top_amplified_count = 0
        top_amplified_share = 0.0
        if source_counter:
            top_amplified_account, top_amplified_count = source_counter.most_common(1)[0]
            top_amplified_share = top_amplified_count / len(retweets) if retweets else 0.0

        member = member_lookup.get(account_handle)
        retweet_ratio = len(retweets) / len(account_tweets)
        retweets_to_member_ratio = (
            retweets_to_member_count / len(retweets) if retweets else 0.0
        )

        summary = AccountSummary(
            account_handle=account_handle,
            in_member_list=member is not None,
            tweet_count=len(account_tweets),
            retweet_count=len(retweets),
            original_count=len(originals),
            quote_count=sum(1 for tweet in account_tweets if tweet.is_quote),
            retweet_ratio=round(retweet_ratio, 4),
            first_retweeter_count=first_retweeters.get(account_handle, 0),
            unique_retweeted_tweet_count=unique_retweeted_tweet_count,
            unique_retweeted_author_count=unique_retweeted_author_count,
            retweets_to_member_count=retweets_to_member_count,
            retweets_to_member_ratio=round(retweets_to_member_ratio, 4),
            top_amplified_account=top_amplified_account,
            top_amplified_count=top_amplified_count,
            top_amplified_share=round(top_amplified_share, 4),
            hashtag_count=hashtag_count,
            mention_count=mention_count,
            followers_count=member.followers_count if member is not None else None,
            friends_count=member.friends_count if member is not None else None,
            statuses_count=member.statuses_count if member is not None else None,
            account_created_at=member.created_at if member is not None else None,
            default_profile=member.default_profile if member is not None else None,
            default_profile_image=member.default_profile_image if member is not None else None,
            is_blue_verified=member.is_blue_verified if member is not None else None,
            profile_description_language=(
                member.profile_description_language if member is not None else None
            ),
            coordination_score=0.0,
        )
        summary.coordination_score = _coordination_score(summary)
        summaries.append(summary)

    return sorted(
        summaries,
        key=lambda item: (-item.coordination_score, -item.tweet_count, item.account_handle),
    )


def build_overlap_network(
    overlaps: list[OverlapSummary],
) -> tuple[list[NetworkNodeSummary], list[NetworkComponentSummary]]:
    """Build connected-component and node metrics from overlap edges."""

    if not overlaps:
        return [], []

    adjacency: dict[str, set[str]] = defaultdict(set)
    edge_lookup: dict[tuple[str, str], OverlapSummary] = {}
    for overlap in overlaps:
        adjacency[overlap.account_a].add(overlap.account_b)
        adjacency[overlap.account_b].add(overlap.account_a)
        edge_lookup[(overlap.account_a, overlap.account_b)] = overlap

    components: list[set[str]] = []
    seen: set[str] = set()
    for account in sorted(adjacency):
        if account in seen:
            continue
        stack = [account]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            stack.extend(sorted(adjacency[current] - seen))
        components.append(component)

    component_records: list[NetworkComponentSummary] = []
    node_records: list[NetworkNodeSummary] = []
    ordered_components = sorted(
        components,
        key=lambda component: (-len(component), sorted(component)[0]),
    )

    for index, component in enumerate(ordered_components, start=1):
        component_edges = [
            overlap
            for overlap in overlaps
            if overlap.account_a in component and overlap.account_b in component
        ]
        edge_count = len(component_edges)
        node_count = len(component)
        density = 0.0
        if node_count > 1:
            density = (2 * edge_count) / (node_count * (node_count - 1))

        weighted_degree: dict[str, int] = Counter()
        within_15m_weight: dict[str, int] = Counter()
        within_60m_weight: dict[str, int] = Counter()
        strongest_neighbor: dict[str, tuple[int, str]] = {}
        for edge in component_edges:
            weighted_degree[edge.account_a] += edge.shared_retweets
            weighted_degree[edge.account_b] += edge.shared_retweets
            within_15m_weight[edge.account_a] += edge.shared_within_15m
            within_15m_weight[edge.account_b] += edge.shared_within_15m
            within_60m_weight[edge.account_a] += edge.shared_within_60m
            within_60m_weight[edge.account_b] += edge.shared_within_60m

            _update_strongest_neighbor(
                strongest_neighbor,
                account=edge.account_a,
                neighbor=edge.account_b,
                shared=edge.shared_retweets,
            )
            _update_strongest_neighbor(
                strongest_neighbor,
                account=edge.account_b,
                neighbor=edge.account_a,
                shared=edge.shared_retweets,
            )

        ranked_accounts = sorted(
            component,
            key=lambda account: (
                -weighted_degree.get(account, 0),
                -len(adjacency[account]),
                account,
            ),
        )
        component_records.append(
            NetworkComponentSummary(
                component_id=index,
                node_count=node_count,
                edge_count=edge_count,
                density=round(density, 4),
                total_shared_retweets=sum(edge.shared_retweets for edge in component_edges),
                total_within_15m=sum(edge.shared_within_15m for edge in component_edges),
                total_within_60m=sum(edge.shared_within_60m for edge in component_edges),
                top_accounts=tuple(ranked_accounts[:10]),
            )
        )

        for account in ranked_accounts:
            strongest = strongest_neighbor.get(account)
            node_records.append(
                NetworkNodeSummary(
                    account_handle=account,
                    component_id=index,
                    component_size=node_count,
                    neighbor_count=len(adjacency[account]),
                    weighted_degree=weighted_degree.get(account, 0),
                    within_15m_weight=within_15m_weight.get(account, 0),
                    within_60m_weight=within_60m_weight.get(account, 0),
                    strongest_neighbor=strongest[1] if strongest is not None else None,
                    max_shared_edge=strongest[0] if strongest is not None else 0,
                )
            )

    node_records.sort(
        key=lambda item: (
            item.component_id,
            -item.weighted_degree,
            -item.neighbor_count,
            item.account_handle,
        )
    )
    return node_records, component_records


def build_feature_rows(
    account_summaries: list[AccountSummary],
    network_nodes: list[NetworkNodeSummary],
    *,
    reference_time: datetime | None,
    cohorts: list[CohortSummary] | None = None,
    account_roles: list[AccountRole] | None = None,
    cascade_propagation: list[CascadePropagationSummary] | None = None,
    total_cascade_count: int | None = None,
) -> list[FeatureRow]:
    """Build a numeric feature matrix with network, cohort, role and propagation features."""

    network_lookup = {node.account_handle: node for node in network_nodes}
    cohort_membership: dict[str, list[CohortSummary]] = {}
    for cohort in cohorts or []:
        for handle in cohort.members:
            cohort_membership.setdefault(handle, []).append(cohort)
    role_lookup = {role.account_handle: role for role in (account_roles or [])}
    propagation_lead_counts: Counter[str] = Counter()
    for propagation in cascade_propagation or []:
        if propagation.first_retweeter is not None:
            propagation_lead_counts[propagation.first_retweeter] += 1
    total_cascades = (
        total_cascade_count
        if total_cascade_count is not None
        else len(cascade_propagation or [])
    )

    rows: list[FeatureRow] = []
    for summary in account_summaries:
        node = network_lookup.get(summary.account_handle)
        account_age_days = _account_age_days(summary.account_created_at, reference_time)
        cohort_list = cohort_membership.get(summary.account_handle, [])
        cohort_count = len(cohort_list)
        cohort_max_size = max((cohort.member_count for cohort in cohort_list), default=0)
        cohort_max_cascade_count = max(
            (cohort.cascade_count for cohort in cohort_list), default=0
        )
        role = role_lookup.get(summary.account_handle)
        role_label = role.role_label if role is not None else "unknown"
        first_retweeter_ratio = (
            summary.first_retweeter_count / total_cascades if total_cascades else 0.0
        )
        propagation_lead_count = propagation_lead_counts.get(summary.account_handle, 0)

        rows.append(
            FeatureRow(
                account_handle=summary.account_handle,
                tweet_count=summary.tweet_count,
                retweet_count=summary.retweet_count,
                original_count=summary.original_count,
                quote_count=summary.quote_count,
                retweet_ratio=summary.retweet_ratio,
                first_retweeter_count=summary.first_retweeter_count,
                unique_retweeted_tweet_count=summary.unique_retweeted_tweet_count,
                unique_retweeted_author_count=summary.unique_retweeted_author_count,
                retweets_to_member_count=summary.retweets_to_member_count,
                retweets_to_member_ratio=summary.retweets_to_member_ratio,
                top_amplified_count=summary.top_amplified_count,
                top_amplified_share=summary.top_amplified_share,
                hashtag_count=summary.hashtag_count,
                mention_count=summary.mention_count,
                followers_count=summary.followers_count,
                friends_count=summary.friends_count,
                statuses_count=summary.statuses_count,
                account_age_days=account_age_days,
                default_profile=_bool_to_int(summary.default_profile),
                default_profile_image=_bool_to_int(summary.default_profile_image),
                is_blue_verified=_bool_to_int(summary.is_blue_verified),
                network_component_size=node.component_size if node is not None else 0,
                network_neighbor_count=node.neighbor_count if node is not None else 0,
                network_weighted_degree=node.weighted_degree if node is not None else 0,
                network_within_15m_weight=node.within_15m_weight if node is not None else 0,
                network_within_60m_weight=node.within_60m_weight if node is not None else 0,
                network_max_shared_edge=node.max_shared_edge if node is not None else 0,
                cohort_count=cohort_count,
                cohort_max_size=cohort_max_size,
                cohort_max_cascade_count=cohort_max_cascade_count,
                role_retail_user=int(role_label == "retail_user"),
                role_amplifier_suspect=int(role_label == "amplifier_suspect"),
                role_media_business=int(role_label == "media_business"),
                role_journalist_public_figure=int(role_label == "journalist_public_figure"),
                role_source_hub=int(role_label == "source_hub"),
                first_retweeter_ratio=round(first_retweeter_ratio, 4),
                propagation_lead_count=propagation_lead_count,
                coordination_score=summary.coordination_score,
                analyst_label="",
            )
        )
    return rows


def _coordination_score(summary: AccountSummary) -> float:
    """Compute a simple review score from deterministic coordination indicators."""

    score = 0.0
    score += min(summary.retweet_ratio, 1.0) * 35.0
    if summary.tweet_count >= 5 and summary.original_count == 0:
        score += 15.0
    score += min(summary.first_retweeter_count / 20.0, 1.0) * 25.0
    score += summary.top_amplified_share * 10.0
    score += summary.retweets_to_member_ratio * 15.0
    return round(score, 2)


def _bool_to_int(value: bool | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _account_age_days(
    created_at: datetime | None,
    reference_time: datetime | None,
) -> int | None:
    if created_at is None or reference_time is None:
        return None
    delta = reference_time - created_at
    return max(delta.days, 0)


def _sorted_pair(left: str, right: str) -> tuple[str, str]:
    if left <= right:
        return (left, right)
    return (right, left)


def _update_strongest_neighbor(
    current: dict[str, tuple[int, str]],
    *,
    account: str,
    neighbor: str,
    shared: int,
) -> None:
    existing = current.get(account)
    candidate = (shared, neighbor)
    if existing is None or candidate > existing:
        current[account] = candidate
