from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path


@dataclass(slots=True)
class DatasetPaths:
    """Resolved input paths for one run."""

    members_file: Path
    exporter_json: Path
    extra_members_file: Path | None = None


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
class UserCascadeSummary:
    """Summary of retweet activity aggregated at the source-account level."""

    original_author_handle: str
    total_retweet_count: int
    unique_retweeted_tweet_count: int
    unique_retweeter_count: int
    first_retweet_at: datetime | None
    last_retweet_at: datetime | None
    total_retweets_within_15m: int
    total_retweets_within_60m: int
    top_cascade_tweet_id: str
    top_cascade_retweet_count: int


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
class AmplifiedTargetSummary:
    """Per amplified-author concentration profile across captured retweets."""

    amplified_author_handle: str
    in_member_list: bool
    total_inbound_retweets: int
    unique_amplifier_count: int
    unique_amplified_posts: int
    top_amplifier_handle: str | None
    top_amplifier_count: int
    top_amplifier_share: float
    amplification_hhi: float
    repeat_crew_overlap: float
    median_first_retweet_lag_sec: int | None
    fastest_amplifier_handle: str | None
    cross_component_reach: int
    within_15m_inbound: int
    within_60m_inbound: int
    top_first_retweeter_handle: str | None
    top_first_retweeter_count: int
    captured_role_mix: str
    amplification_score: float


@dataclass(slots=True)
class CohortSummary:
    """A recurrent group of accounts that co-appear tightly across cascades."""

    cohort_id: int
    member_count: int
    cascade_count: int
    median_time_tightness_sec: int | None
    target_concentration_hhi: float
    top_target_author: str | None
    top_target_count: int
    unique_targets: int
    component_ids: tuple[int, ...]
    members: tuple[str, ...]
    top_cascade_tweet_id: str | None
    top_cascade_member_share: float
    cohort_score: float


@dataclass(slots=True)
class CascadePropagationSummary:
    """Propagation shape and crew participation for one retweeted original."""

    original_tweet_id: str
    original_author_handle: str
    origin_is_member: bool
    retweet_count: int
    unique_retweeter_count: int
    first_retweeter: str | None
    first_retweeter_recurrence: int
    time_to_k5_sec: int | None
    time_to_k10_sec: int | None
    time_to_k50_sec: int | None
    burst_kurtosis: float
    burst_peak_window_start_sec: int | None
    burst_peak_window_count: int
    same_client_share: float
    dominant_client: str | None
    amplifier_component_ids: tuple[int, ...]
    amplifier_cohort_ids: tuple[int, ...]
    amplifier_role_mix: str
    propagation_score: float
    spread_path_preview: str


@dataclass(slots=True)
class CascadeSpreadPath:
    """Ordered retweeter timeline for one cascade, written alongside flat summaries."""

    original_tweet_id: str
    original_author_handle: str
    first_retweet_at: datetime | None
    steps: tuple[tuple[str, int, int, int], ...]


@dataclass(slots=True)
class FirstRetweeterProfile:
    """Per account, a profile of how often they are first on captured cascades."""

    account_handle: str
    first_retweeter_count: int
    unique_targets_first: int
    top_target_author: str | None
    top_target_count: int
    target_hhi: float
    median_lag_to_second_retweet_sec: int | None
    role_label: str
    cohort_ids: tuple[int, ...]
    component_id: int | None


@dataclass(slots=True)
class AccountRole:
    """Deterministic review-oriented role label for one account."""

    account_handle: str
    role_label: str
    role_confidence: float
    is_member: bool
    signals: tuple[str, ...]
    followers_count: int | None
    friends_count: int | None
    followers_friends_ratio: float | None
    listed_count: int | None
    description_language: str | None
    description_length: int
    is_blue_verified: bool | None
    default_profile: bool | None
    account_age_days: int | None
    retweet_ratio: float
    original_count: int
    cohort_ids: tuple[int, ...]


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
    account_age_days: int | None
    default_profile: int | None
    default_profile_image: int | None
    is_blue_verified: int | None
    network_component_size: int
    network_neighbor_count: int
    network_weighted_degree: int
    network_within_15m_weight: int
    network_within_60m_weight: int
    network_max_shared_edge: int
    cohort_count: int
    cohort_max_size: int
    cohort_max_cascade_count: int
    role_retail_user: int
    role_amplifier_suspect: int
    role_media_business: int
    role_journalist_public_figure: int
    role_source_hub: int
    first_retweeter_ratio: float
    propagation_lead_count: int
    coordination_score: float
    analyst_label: str


@dataclass(slots=True)
class MLRunSummary:
    """Top-level metadata for one ML baseline run."""

    status: str
    sample_count: int
    input_feature_count: int
    cluster_count: int
    cluster_selection: str
    cluster_model: str
    anomaly_model: str
    embedding_model: str
    feature_names: tuple[str, ...]
    note: str | None


@dataclass(slots=True)
class MLAccountSummary:
    """Account-level ML outputs derived from the deterministic feature matrix."""

    account_handle: str
    cluster_id: int
    cluster_size: int
    anomaly_score: float
    anomaly_rank: int
    centroid_distance: float
    embedding_x: float
    embedding_y: float
    coordination_score: float
    retweet_ratio: float
    network_weighted_degree: int


@dataclass(slots=True)
class MLClusterSummary:
    """Cluster-level summaries for the unsupervised ML baseline."""

    cluster_id: int
    account_count: int
    mean_coordination_score: float
    mean_retweet_ratio: float
    mean_network_weighted_degree: float
    top_accounts: tuple[str, ...]


@dataclass(slots=True)
class AnalysisResults:
    """Container for one completed analysis run."""

    dataset_paths: DatasetPaths
    dataset_stats: DatasetStats
    normalized_tweets: list[TweetRecord]
    account_summaries: list[AccountSummary]
    cascade_summaries: list[CascadeSummary]
    user_cascade_summaries: list[UserCascadeSummary]
    hashtag_summaries: list[TokenSummary]
    mention_summaries: list[TokenSummary]
    retweet_edges: list[RetweetEdgeSummary]
    overlap_summaries: list[OverlapSummary]
    network_nodes: list[NetworkNodeSummary]
    network_components: list[NetworkComponentSummary]
    account_roles: list[AccountRole]
    amplified_targets: list[AmplifiedTargetSummary]
    cohorts: list[CohortSummary]
    cascade_propagation: list[CascadePropagationSummary]
    cascade_spread_paths: list[CascadeSpreadPath]
    first_retweeter_profiles: list[FirstRetweeterProfile]
    feature_rows: list[FeatureRow]
    ml_run_summary: MLRunSummary
    ml_account_summaries: list[MLAccountSummary]
    ml_cluster_summaries: list[MLClusterSummary]
