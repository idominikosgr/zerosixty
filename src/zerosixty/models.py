from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(slots=True)
class DatasetPaths:
    """Resolved input paths for one run."""

    members_csv: Path
    exporter_json: Path


@dataclass(slots=True)
class MemberRecord:
    """Normalized member profile metadata."""

    member_id: str
    screen_name: str
    name: str | None
    description: str | None
    created_at: datetime | None
    followers_count: int
    friends_count: int
    statuses_count: int
    favourites_count: int
    listed_count: int
    location: str | None
    is_blue_verified: bool
    protected: bool
    default_profile: bool | None
    default_profile_image: bool | None
    profile_description_language: str | None


@dataclass(slots=True)
class TweetRecord:
    """Flattened tweet row used by downstream analysis."""

    tweet_id: str
    author_handle: str
    author_name: str | None
    posted_at: datetime | None
    source_client: str | None
    text: str
    canonical_text: str
    is_retweet: bool
    is_quote: bool
    content_tweet_id: str
    content_author_handle: str
    content_author_name: str | None
    content_created_at: datetime | None
    content_text: str
    content_canonical_text: str
    hashtags_raw: tuple[str, ...]
    hashtags: tuple[str, ...]
    mentions_raw: tuple[str, ...]
    mentions: tuple[str, ...]


@dataclass(slots=True)
class DatasetStats:
    """Top-level counts and run metadata."""

    member_count: int
    tweet_count: int
    active_account_count: int
    retweet_count: int
    original_count: int
    date_start: datetime | None
    date_end: datetime | None
    missing_member_handles: tuple[str, ...]


@dataclass(slots=True)
class AccountSummary:
    """Per-account deterministic activity and coordination features."""

    account_handle: str
    in_member_list: bool
    tweet_count: int
    retweet_count: int
    original_count: int
    quote_count: int
    retweet_ratio: float
    first_retweeter_count: int
    unique_retweeted_tweet_count: int
    unique_retweeted_author_count: int
    retweets_to_member_count: int
    retweets_to_member_ratio: float
    top_amplified_account: str | None
    top_amplified_count: int
    top_amplified_share: float
    hashtag_count: int
    mention_count: int
    followers_count: int | None
    friends_count: int | None
    statuses_count: int | None
    account_created_at: datetime | None
    default_profile: bool | None
    default_profile_image: bool | None
    is_blue_verified: bool | None
    profile_description_language: str | None
    coordination_score: float


@dataclass(slots=True)
class CascadeSummary:
    """Summary of one retweeted original across the captured accounts."""

    original_tweet_id: str
    original_author_handle: str
    original_created_at: datetime | None
    retweet_count: int
    unique_retweeter_count: int
    first_retweeter: str | None
    first_retweet_at: datetime | None
    last_retweet_at: datetime | None
    span_minutes: int | None
    retweets_within_15m: int
    retweets_within_60m: int
    hashtags: tuple[str, ...]
    mentions: tuple[str, ...]
    retweeters: tuple[str, ...]
    sample_text: str


@dataclass(slots=True)
class TokenSummary:
    """Frequency table for hashtags or mentions."""

    token: str
    display_token: str
    count: int
    unique_account_count: int
    sample_accounts: tuple[str, ...]


@dataclass(slots=True)
class RetweetEdgeSummary:
    """Retweeter -> source-author edge summary."""

    retweeter: str
    source_author: str
    count: int


@dataclass(slots=True)
class OverlapSummary:
    """Account-pair overlap on retweeted originals."""

    account_a: str
    account_b: str
    shared_retweets: int
    jaccard: float
    shared_within_15m: int
    shared_within_60m: int


@dataclass(slots=True)
class NetworkNodeSummary:
    """Node-level metrics for the shared-retweet overlap graph."""

    account_handle: str
    component_id: int
    component_size: int
    neighbor_count: int
    weighted_degree: int
    within_15m_weight: int
    within_60m_weight: int
    strongest_neighbor: str | None
    max_shared_edge: int


@dataclass(slots=True)
class NetworkComponentSummary:
    """Component-level metrics for the shared-retweet overlap graph."""

    component_id: int
    node_count: int
    edge_count: int
    density: float
    total_shared_retweets: int
    total_within_15m: int
    total_within_60m: int
    top_accounts: tuple[str, ...]


@dataclass(slots=True)
class FeatureRow:
    """Numeric feature row for later analyst labeling and ML."""

    account_handle: str
    tweet_count: int
    retweet_count: int
    original_count: int
    quote_count: int
    retweet_ratio: float
    first_retweeter_count: int
    unique_retweeted_tweet_count: int
    unique_retweeted_author_count: int
    retweets_to_member_count: int
    retweets_to_member_ratio: float
    top_amplified_count: int
    top_amplified_share: float
    hashtag_count: int
    mention_count: int
    followers_count: int | None
    friends_count: int | None
    statuses_count: int | None
    default_profile: int | None
    default_profile_image: int | None
    is_blue_verified: int | None
    coordination_score: float
    analyst_label: str


@dataclass(slots=True)
class AnalysisResults:
    """Container for one completed analysis run."""

    dataset_paths: DatasetPaths
    dataset_stats: DatasetStats
    normalized_tweets: list[TweetRecord]
    account_summaries: list[AccountSummary]
    cascade_summaries: list[CascadeSummary]
    hashtag_summaries: list[TokenSummary]
    mention_summaries: list[TokenSummary]
    retweet_edges: list[RetweetEdgeSummary]
    overlap_summaries: list[OverlapSummary]
    network_nodes: list[NetworkNodeSummary]
    network_components: list[NetworkComponentSummary]
    feature_rows: list[FeatureRow]
