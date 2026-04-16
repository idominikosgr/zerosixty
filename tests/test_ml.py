from __future__ import annotations

import pytest

from zerosixty.ml import MODEL_FEATURE_NAMES, run_ml_pipeline
from zerosixty.models import FeatureRow


def test_run_ml_pipeline_handles_empty_feature_rows() -> None:
    run_summary, account_summaries, cluster_summaries = run_ml_pipeline([])

    assert run_summary.status == "skipped_no_samples"
    assert run_summary.input_feature_count == len(MODEL_FEATURE_NAMES)
    assert account_summaries == []
    assert cluster_summaries == []


def test_run_ml_pipeline_builds_clusters_and_anomaly_ranks() -> None:
    pytest.importorskip("sklearn")

    feature_rows = [
        _feature_row(
            "acct-a",
            tweet_count=40,
            retweet_count=38,
            original_count=2,
            retweet_ratio=0.95,
            first_retweeter_count=15,
            retweets_to_member_count=20,
            retweets_to_member_ratio=0.8,
            top_amplified_count=18,
            top_amplified_share=0.7,
            hashtag_count=5,
            mention_count=8,
            followers_count=180,
            friends_count=90,
            statuses_count=400,
            account_age_days=900,
            network_component_size=12,
            network_neighbor_count=6,
            network_weighted_degree=24,
            network_within_15m_weight=9,
            network_within_60m_weight=14,
            network_max_shared_edge=7,
            coordination_score=82.0,
        ),
        _feature_row(
            "acct-b",
            tweet_count=36,
            retweet_count=34,
            original_count=2,
            retweet_ratio=0.94,
            first_retweeter_count=14,
            retweets_to_member_count=18,
            retweets_to_member_ratio=0.75,
            top_amplified_count=16,
            top_amplified_share=0.66,
            hashtag_count=4,
            mention_count=7,
            followers_count=170,
            friends_count=85,
            statuses_count=390,
            account_age_days=870,
            network_component_size=12,
            network_neighbor_count=5,
            network_weighted_degree=22,
            network_within_15m_weight=8,
            network_within_60m_weight=13,
            network_max_shared_edge=6,
            coordination_score=78.0,
        ),
        _feature_row(
            "acct-c",
            tweet_count=10,
            retweet_count=2,
            original_count=8,
            retweet_ratio=0.2,
            first_retweeter_count=1,
            retweets_to_member_count=0,
            retweets_to_member_ratio=0.0,
            top_amplified_count=1,
            top_amplified_share=0.1,
            hashtag_count=1,
            mention_count=1,
            followers_count=1200,
            friends_count=300,
            statuses_count=2000,
            account_age_days=2200,
            network_component_size=2,
            network_neighbor_count=1,
            network_weighted_degree=2,
            network_within_15m_weight=0,
            network_within_60m_weight=1,
            network_max_shared_edge=1,
            coordination_score=12.0,
        ),
        _feature_row(
            "acct-d",
            tweet_count=12,
            retweet_count=3,
            original_count=9,
            retweet_ratio=0.25,
            first_retweeter_count=1,
            retweets_to_member_count=1,
            retweets_to_member_ratio=0.08,
            top_amplified_count=2,
            top_amplified_share=0.12,
            hashtag_count=1,
            mention_count=2,
            followers_count=1100,
            friends_count=320,
            statuses_count=2100,
            account_age_days=2100,
            network_component_size=2,
            network_neighbor_count=1,
            network_weighted_degree=2,
            network_within_15m_weight=0,
            network_within_60m_weight=1,
            network_max_shared_edge=1,
            coordination_score=15.0,
        ),
    ]

    run_summary, account_summaries, cluster_summaries = run_ml_pipeline(
        feature_rows,
        requested_clusters=2,
        random_state=7,
    )

    assert run_summary.status == "ready"
    assert run_summary.input_feature_count == len(MODEL_FEATURE_NAMES)
    assert run_summary.cluster_count == 2
    assert run_summary.cluster_selection == "requested"
    assert len(account_summaries) == 4
    assert len(cluster_summaries) == 2
    assert {item.cluster_id for item in account_summaries} == {1, 2}
    assert {item.anomaly_rank for item in account_summaries} == {1, 2, 3, 4}
    assert sorted(item.cluster_size for item in account_summaries) == [2, 2, 2, 2]
    assert sorted(item.account_count for item in cluster_summaries) == [2, 2]
    assert all(isinstance(item.embedding_x, float) for item in account_summaries)
    assert all(isinstance(item.embedding_y, float) for item in account_summaries)


def _feature_row(
    account_handle: str,
    *,
    tweet_count: int = 0,
    retweet_count: int = 0,
    original_count: int = 0,
    quote_count: int = 0,
    retweet_ratio: float = 0.0,
    first_retweeter_count: int = 0,
    unique_retweeted_tweet_count: int = 0,
    unique_retweeted_author_count: int = 0,
    retweets_to_member_count: int = 0,
    retweets_to_member_ratio: float = 0.0,
    top_amplified_count: int = 0,
    top_amplified_share: float = 0.0,
    hashtag_count: int = 0,
    mention_count: int = 0,
    followers_count: int | None = None,
    friends_count: int | None = None,
    statuses_count: int | None = None,
    account_age_days: int | None = None,
    default_profile: int | None = 0,
    default_profile_image: int | None = 0,
    is_blue_verified: int | None = 0,
    network_component_size: int = 0,
    network_neighbor_count: int = 0,
    network_weighted_degree: int = 0,
    network_within_15m_weight: int = 0,
    network_within_60m_weight: int = 0,
    network_max_shared_edge: int = 0,
    coordination_score: float = 0.0,
) -> FeatureRow:
    return FeatureRow(
        account_handle=account_handle,
        tweet_count=tweet_count,
        retweet_count=retweet_count,
        original_count=original_count,
        quote_count=quote_count,
        retweet_ratio=retweet_ratio,
        first_retweeter_count=first_retweeter_count,
        unique_retweeted_tweet_count=unique_retweeted_tweet_count,
        unique_retweeted_author_count=unique_retweeted_author_count,
        retweets_to_member_count=retweets_to_member_count,
        retweets_to_member_ratio=retweets_to_member_ratio,
        top_amplified_count=top_amplified_count,
        top_amplified_share=top_amplified_share,
        hashtag_count=hashtag_count,
        mention_count=mention_count,
        followers_count=followers_count,
        friends_count=friends_count,
        statuses_count=statuses_count,
        account_age_days=account_age_days,
        default_profile=default_profile,
        default_profile_image=default_profile_image,
        is_blue_verified=is_blue_verified,
        network_component_size=network_component_size,
        network_neighbor_count=network_neighbor_count,
        network_weighted_degree=network_weighted_degree,
        network_within_15m_weight=network_within_15m_weight,
        network_within_60m_weight=network_within_60m_weight,
        network_max_shared_edge=network_max_shared_edge,
        coordination_score=coordination_score,
        analyst_label="",
    )
