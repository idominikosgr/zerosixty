from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

from zerosixty.models import (
    AccountSummary,
    CascadeSummary,
    DatasetStats,
    FeatureRow,
    MemberRecord,
    NetworkComponentSummary,
    NetworkNodeSummary,
    OverlapSummary,
    RetweetEdgeSummary,
    TokenSummary,
    TweetRecord,
)


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


def build_feature_rows(account_summaries: list[AccountSummary]) -> list[FeatureRow]:
    """Build a numeric feature matrix with blank analyst labels."""

    return [
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
            default_profile=_bool_to_int(summary.default_profile),
            default_profile_image=_bool_to_int(summary.default_profile_image),
            is_blue_verified=_bool_to_int(summary.is_blue_verified),
            coordination_score=summary.coordination_score,
            analyst_label="",
        )
        for summary in account_summaries
    ]


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
