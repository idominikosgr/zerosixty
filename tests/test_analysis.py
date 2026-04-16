from __future__ import annotations

from datetime import UTC, datetime, timedelta

from zerosixty.analyze import (
    build_account_summaries,
    build_cascade_summaries,
    build_feature_rows,
    build_overlap_network,
    build_overlap_summaries,
)
from zerosixty.models import MemberRecord, TweetRecord


def test_build_cascade_summaries_picks_first_retweeter() -> None:
    tweets = [
        _retweet("alice", "orig-1", "source-a", 0),
        _retweet("bob", "orig-1", "source-a", 5),
        _retweet("carol", "orig-1", "source-a", 80),
    ]

    summaries = build_cascade_summaries(tweets)

    assert len(summaries) == 1
    assert summaries[0].first_retweeter == "alice"
    assert summaries[0].retweets_within_15m == 2
    assert summaries[0].retweets_within_60m == 2


def test_build_account_summaries_scores_pure_retweeters_higher() -> None:
    members = [
        _member("alice"),
        _member("bob"),
    ]
    tweets = [
        _retweet("alice", "orig-1", "source-a", 0),
        _retweet("alice", "orig-2", "source-a", 10),
        _retweet("bob", "orig-1", "source-a", 15),
        _original("bob", "own-1", 20),
    ]

    cascades = build_cascade_summaries(tweets)
    summaries = build_account_summaries(members, tweets, cascades)
    summary_by_account = {summary.account_handle: summary for summary in summaries}

    assert summary_by_account["alice"].retweet_ratio == 1.0
    assert (
        summary_by_account["alice"].coordination_score
        > summary_by_account["bob"].coordination_score
    )


def test_build_overlap_summaries_respects_threshold() -> None:
    tweets = [
        _retweet("alice", "orig-1", "source-a", 0),
        _retweet("bob", "orig-1", "source-a", 2),
        _retweet("alice", "orig-2", "source-b", 5),
        _retweet("bob", "orig-2", "source-b", 7),
        _retweet("carol", "orig-2", "source-b", 9),
    ]

    summaries = build_overlap_summaries(tweets, min_shared=2)

    assert len(summaries) == 1
    assert summaries[0].account_a == "alice"
    assert summaries[0].account_b == "bob"
    assert summaries[0].shared_retweets == 2


def test_build_overlap_network_extracts_component_and_node_metrics() -> None:
    tweets = [
        _retweet("alice", "orig-1", "source-a", 0),
        _retweet("bob", "orig-1", "source-a", 2),
        _retweet("alice", "orig-2", "source-b", 5),
        _retweet("bob", "orig-2", "source-b", 7),
        _retweet("carol", "orig-2", "source-b", 9),
    ]

    overlaps = build_overlap_summaries(tweets, min_shared=1)
    nodes, components = build_overlap_network(overlaps)

    assert len(components) == 1
    assert components[0].node_count == 3
    assert components[0].edge_count == 3
    assert components[0].top_accounts[0] == "alice"
    node_by_account = {node.account_handle: node for node in nodes}
    assert node_by_account["alice"].component_id == 1
    assert node_by_account["alice"].neighbor_count == 2
    assert node_by_account["alice"].strongest_neighbor == "bob"


def test_build_feature_rows_enriches_accounts_with_network_metrics() -> None:
    members = [
        _member("alice"),
        _member("bob"),
    ]
    tweets = [
        _retweet("alice", "orig-1", "source-a", 0),
        _retweet("bob", "orig-1", "source-a", 2),
        _retweet("alice", "orig-2", "source-b", 5),
        _retweet("bob", "orig-2", "source-b", 7),
        _retweet("carol", "orig-2", "source-b", 9),
    ]

    cascades = build_cascade_summaries(tweets)
    summaries = build_account_summaries(members, tweets, cascades)
    overlaps = build_overlap_summaries(tweets, min_shared=1)
    nodes, _ = build_overlap_network(overlaps)
    reference_time = datetime(2026, 4, 16, tzinfo=UTC)

    feature_rows = build_feature_rows(
        summaries,
        nodes,
        reference_time=reference_time,
    )
    rows_by_account = {row.account_handle: row for row in feature_rows}

    assert rows_by_account["alice"].account_age_days == (
        reference_time - datetime(2024, 1, 1, tzinfo=UTC)
    ).days
    assert rows_by_account["alice"].network_component_size == 3
    assert rows_by_account["alice"].network_neighbor_count == 2
    assert rows_by_account["alice"].network_weighted_degree == 3
    assert rows_by_account["alice"].network_max_shared_edge == 2
    assert rows_by_account["carol"].account_age_days is None
    assert rows_by_account["carol"].network_component_size == 3


def _member(handle: str) -> MemberRecord:
    return MemberRecord(
        member_id=handle,
        screen_name=handle,
        name=handle.title(),
        description=None,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        followers_count=100,
        friends_count=50,
        statuses_count=10,
        favourites_count=20,
        listed_count=1,
        location=None,
        is_blue_verified=False,
        protected=False,
        default_profile=True,
        default_profile_image=False,
        profile_description_language="el",
    )


def _retweet(
    author_handle: str,
    original_tweet_id: str,
    content_author_handle: str,
    minute_offset: int,
) -> TweetRecord:
    created = datetime(2026, 4, 15, 20, 0, tzinfo=UTC) + timedelta(minutes=minute_offset)
    return TweetRecord(
        tweet_id=f"{author_handle}-{original_tweet_id}",
        author_handle=author_handle,
        author_name=author_handle.title(),
        posted_at=created,
        source_client="Twitter Web App",
        text=f"RT @{content_author_handle}: sample",
        canonical_text="rt sample",
        is_retweet=True,
        is_quote=False,
        content_tweet_id=original_tweet_id,
        content_author_handle=content_author_handle,
        content_author_name=content_author_handle.title(),
        content_created_at=created,
        content_text="sample",
        content_canonical_text="sample",
        hashtags_raw=(),
        hashtags=(),
        mentions_raw=(),
        mentions=(),
    )


def _original(author_handle: str, tweet_id: str, minute_offset: int) -> TweetRecord:
    created = datetime(2026, 4, 15, 20, 0, tzinfo=UTC) + timedelta(minutes=minute_offset)
    return TweetRecord(
        tweet_id=tweet_id,
        author_handle=author_handle,
        author_name=author_handle.title(),
        posted_at=created,
        source_client="Twitter Web App",
        text="original",
        canonical_text="original",
        is_retweet=False,
        is_quote=False,
        content_tweet_id=tweet_id,
        content_author_handle=author_handle,
        content_author_name=author_handle.title(),
        content_created_at=created,
        content_text="original",
        content_canonical_text="original",
        hashtags_raw=(),
        hashtags=(),
        mentions_raw=(),
        mentions=(),
    )
