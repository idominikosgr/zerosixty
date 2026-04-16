from __future__ import annotations

from pathlib import Path

from zerosixty.analyze import (
    build_account_summaries,
    build_cascade_summaries,
    build_dataset_stats,
    build_feature_rows,
    build_overlap_network,
    build_overlap_summaries,
    build_retweet_edges,
    build_token_summaries,
)
from zerosixty.discovery import discover_dataset
from zerosixty.loaders import load_export_rows, load_member_records
from zerosixty.models import AnalysisResults
from zerosixty.normalize import build_tweet_records
from zerosixty.reporting import write_outputs


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    *,
    members_csv: Path | None = None,
    exporter_json: Path | None = None,
    min_shared_overlap: int = 2,
) -> tuple[AnalysisResults, dict[str, Path]]:
    """Run the full deterministic analysis pipeline."""

    dataset_paths = discover_dataset(
        input_dir,
        members_csv=members_csv,
        exporter_json=exporter_json,
    )
    members = load_member_records(dataset_paths.members_csv)
    raw_rows = load_export_rows(dataset_paths.exporter_json)
    tweets = build_tweet_records(raw_rows)

    dataset_stats = build_dataset_stats(members, tweets)
    cascades = build_cascade_summaries(tweets)
    account_summaries = build_account_summaries(members, tweets, cascades)
    hashtag_summaries = build_token_summaries(tweets, token_type="hashtags")
    mention_summaries = build_token_summaries(tweets, token_type="mentions")
    retweet_edges = build_retweet_edges(tweets)
    overlap_summaries = build_overlap_summaries(tweets, min_shared=min_shared_overlap)
    network_nodes, network_components = build_overlap_network(overlap_summaries)
    feature_rows = build_feature_rows(account_summaries)

    results = AnalysisResults(
        dataset_paths=dataset_paths,
        dataset_stats=dataset_stats,
        normalized_tweets=tweets,
        account_summaries=account_summaries,
        cascade_summaries=cascades,
        hashtag_summaries=hashtag_summaries,
        mention_summaries=mention_summaries,
        retweet_edges=retweet_edges,
        overlap_summaries=overlap_summaries,
        network_nodes=network_nodes,
        network_components=network_components,
        feature_rows=feature_rows,
    )
    written = write_outputs(results, output_dir)
    return results, written
