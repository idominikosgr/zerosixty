from __future__ import annotations

from typing import TYPE_CHECKING

from zerosixty.amplification import build_amplified_targets
from zerosixty.analyze import (
    build_account_summaries,
    build_cascade_summaries,
    build_dataset_stats,
    build_feature_rows,
    build_overlap_network,
    build_overlap_summaries,
    build_retweet_edges,
    build_token_summaries,
    build_user_cascade_summaries,
)
from zerosixty.crews import (
    build_cohorts,
    cohort_ids_by_account,
    cohort_ids_by_cascade,
)
from zerosixty.discovery import discover_dataset
from zerosixty.loaders import (
    load_clean_account_records,
    load_clean_member_records,
    load_clean_tweet_records,
    load_export_rows,
    load_export_users,
    load_extra_member_handles,
    load_member_records,
)
from zerosixty.ml import run_ml_pipeline
from zerosixty.models import AnalysisResults, DatasetPaths, MLRunSummary
from zerosixty.normalize import build_account_profiles, build_tweet_records
from zerosixty.propagation import (
    build_cascade_propagation,
    build_first_retweeter_profiles,
)
from zerosixty.reporting import write_outputs
from zerosixty.targets import (
    build_cascade_tight_windows,
    build_retweet_target_profiles,
    build_source_client_fingerprints,
)
from zerosixty.typology import build_account_roles

if TYPE_CHECKING:
    from pathlib import Path

    from zerosixty.models import AccountProfile, MemberRecord, TweetRecord


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    *,
    members_file: Path | None = None,
    exporter_json: Path | None = None,
    extra_members_file: Path | None = None,
    min_shared_overlap: int = 2,
    enable_ml: bool = True,
    ml_clusters: int | None = None,
    ml_random_state: int = 42,
) -> tuple[AnalysisResults, dict[str, Path]]:
    """Run the deterministic and ML baseline pipelines."""

    dataset_paths = discover_dataset(
        input_dir,
        members_file=members_file,
        exporter_json=exporter_json,
        extra_members_file=extra_members_file,
    )
    extra_handles = (
        load_extra_member_handles(dataset_paths.extra_members_file)
        if dataset_paths.extra_members_file is not None
        else []
    )
    members = load_member_records(
        dataset_paths.members_file,
        extra_handles=extra_handles,
    )
    raw_rows = load_export_rows(dataset_paths.exporter_json)
    raw_users = load_export_users(dataset_paths.exporter_json)
    tweets = build_tweet_records(raw_rows)
    member_handle_set = {member.screen_name for member in members if member.screen_name}
    account_profiles = build_account_profiles(
        raw_rows=raw_rows,
        users_table_rows=raw_users,
        member_handles=member_handle_set,
    )

    return _run_analysis(
        dataset_paths=dataset_paths,
        members=members,
        tweets=tweets,
        account_profiles=account_profiles,
        output_dir=output_dir,
        min_shared_overlap=min_shared_overlap,
        enable_ml=enable_ml,
        ml_clusters=ml_clusters,
        ml_random_state=ml_random_state,
    )


def run_pipeline_clean(
    *,
    members_csv: Path,
    tweets_csv: Path,
    output_dir: Path,
    source_members_file: Path | None = None,
    source_exporter_file: Path | None = None,
    source_extra_members_file: Path | None = None,
    accounts_csv: Path | None = None,
    min_shared_overlap: int = 2,
    enable_ml: bool = True,
    ml_clusters: int | None = None,
    ml_random_state: int = 42,
) -> tuple[AnalysisResults, dict[str, Path]]:
    """Run analysis from normalized clean-batch CSV files.

    Loads `accounts.csv` from the same batch directory when present (sibling of
    tweets.csv); falls back to an empty registry for older batches.
    """

    members = load_clean_member_records(members_csv)
    tweets = load_clean_tweet_records(tweets_csv)
    if accounts_csv is None:
        accounts_csv = tweets_csv.parent / "accounts.csv"
    account_profiles = load_clean_account_records(accounts_csv)
    dataset_paths = DatasetPaths(
        members_file=(source_members_file or members_csv).resolve(),
        exporter_json=(source_exporter_file or tweets_csv).resolve(),
        extra_members_file=(
            source_extra_members_file.resolve()
            if source_extra_members_file is not None
            else None
        ),
    )

    return _run_analysis(
        dataset_paths=dataset_paths,
        members=members,
        tweets=tweets,
        account_profiles=account_profiles,
        output_dir=output_dir,
        min_shared_overlap=min_shared_overlap,
        enable_ml=enable_ml,
        ml_clusters=ml_clusters,
        ml_random_state=ml_random_state,
    )


def _run_analysis(
    *,
    dataset_paths: DatasetPaths,
    members: list[MemberRecord],
    tweets: list[TweetRecord],
    account_profiles: list[AccountProfile],
    output_dir: Path,
    min_shared_overlap: int,
    enable_ml: bool,
    ml_clusters: int | None,
    ml_random_state: int,
) -> tuple[AnalysisResults, dict[str, Path]]:
    """Shared deterministic+ML analysis core for raw and clean pipelines."""

    dataset_stats = build_dataset_stats(members, tweets)
    cascades = build_cascade_summaries(tweets)
    user_cascades = build_user_cascade_summaries(cascades)
    account_summaries = build_account_summaries(members, tweets, cascades)
    hashtag_summaries = build_token_summaries(tweets, token_type="hashtags")
    mention_summaries = build_token_summaries(tweets, token_type="mentions")
    retweet_edges = build_retweet_edges(tweets)
    overlap_summaries = build_overlap_summaries(tweets, min_shared=min_shared_overlap)
    network_nodes, network_components = build_overlap_network(overlap_summaries)

    cohorts = build_cohorts(
        cascades,
        overlap_summaries,
        network_nodes,
        min_shared_cascades=min_shared_overlap,
        random_state=ml_random_state,
    )
    cohort_by_account = cohort_ids_by_account(cohorts)
    cohort_by_cascade = cohort_ids_by_cascade(cohorts, cascades)

    account_roles = build_account_roles(
        members,
        account_summaries,
        cohort_ids_by_account=cohort_by_account,
    )

    member_handle_set = {member.screen_name for member in members}
    amplified_targets = build_amplified_targets(
        tweets,
        cascades,
        user_cascades,
        member_handle_set,
        network_nodes,
        account_roles=account_roles,
    )

    cascade_propagation, cascade_spread_paths = build_cascade_propagation(
        tweets,
        cascades,
        member_handle_set,
        network_nodes,
        account_roles=account_roles,
        cohort_ids_by_cascade=cohort_by_cascade,
    )

    first_retweeter_profiles = build_first_retweeter_profiles(
        cascades,
        account_roles=account_roles,
        cohort_ids_by_account=cohort_by_account,
        network_nodes=network_nodes,
    )

    feature_rows = build_feature_rows(
        account_summaries,
        network_nodes,
        reference_time=dataset_stats.date_end,
        cohorts=cohorts,
        account_roles=account_roles,
        cascade_propagation=cascade_propagation,
        total_cascade_count=len(cascades),
    )
    if enable_ml:
        ml_run_summary, ml_account_summaries, ml_cluster_summaries = run_ml_pipeline(
            feature_rows,
            requested_clusters=ml_clusters,
            random_state=ml_random_state,
        )
    else:
        ml_run_summary = MLRunSummary(
            status="skipped_disabled",
            sample_count=len(feature_rows),
            input_feature_count=0,
            cluster_count=0,
            cluster_selection="disabled",
            cluster_model="none",
            anomaly_model="none",
            embedding_model="none",
            feature_names=(),
            note="ML baseline disabled by caller.",
        )
        ml_account_summaries = []
        ml_cluster_summaries = []

    retweet_target_profiles = build_retweet_target_profiles(
        tweets,
        member_handles=member_handle_set,
        account_profiles=account_profiles,
        reference_time=dataset_stats.date_end,
    )
    source_client_fingerprints = build_source_client_fingerprints(
        tweets,
        member_handles=member_handle_set,
        account_roles=account_roles,
    )
    cascade_tight_windows = build_cascade_tight_windows(tweets, cascades)

    results = AnalysisResults(
        dataset_paths=dataset_paths,
        dataset_stats=dataset_stats,
        normalized_tweets=tweets,
        account_summaries=account_summaries,
        cascade_summaries=cascades,
        user_cascade_summaries=user_cascades,
        hashtag_summaries=hashtag_summaries,
        mention_summaries=mention_summaries,
        retweet_edges=retweet_edges,
        overlap_summaries=overlap_summaries,
        network_nodes=network_nodes,
        network_components=network_components,
        account_roles=account_roles,
        amplified_targets=amplified_targets,
        cohorts=cohorts,
        cascade_propagation=cascade_propagation,
        cascade_spread_paths=cascade_spread_paths,
        first_retweeter_profiles=first_retweeter_profiles,
        feature_rows=feature_rows,
        ml_run_summary=ml_run_summary,
        ml_account_summaries=ml_account_summaries,
        ml_cluster_summaries=ml_cluster_summaries,
        retweet_target_profiles=retweet_target_profiles,
        source_client_fingerprints=source_client_fingerprints,
        cascade_tight_windows=cascade_tight_windows,
    )
    written = write_outputs(results, output_dir)
    return results, written
