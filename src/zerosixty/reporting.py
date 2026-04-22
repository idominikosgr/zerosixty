from __future__ import annotations

import csv
import json
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from zerosixty.models import AnalysisResults


def write_outputs(results: AnalysisResults, output_dir: Path) -> dict[str, Path]:
    """Write all flat-file outputs for one analysis run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "normalized_tweets": output_dir / "normalized_tweets.csv",
        "account_summary": output_dir / "account_summary.csv",
        "retweet_cascades": output_dir / "retweet_cascades.csv",
        "retweet_user_cascades": output_dir / "retweet_user_cascades.csv",
        "retweet_edges": output_dir / "retweet_edges.csv",
        "hashtag_summary": output_dir / "hashtag_summary.csv",
        "mention_summary": output_dir / "mention_summary.csv",
        "account_overlap": output_dir / "account_overlap.csv",
        "network_nodes": output_dir / "network_nodes.csv",
        "network_components": output_dir / "network_components.csv",
        "account_roles": output_dir / "account_roles.csv",
        "amplified_targets": output_dir / "amplified_targets.csv",
        "cohorts": output_dir / "cohorts.csv",
        "cohort_members": output_dir / "cohort_members.csv",
        "cascade_propagation": output_dir / "cascade_propagation.csv",
        "cascade_spread_paths": output_dir / "cascade_spread_paths.jsonl",
        "first_retweeter_profiles": output_dir / "first_retweeter_profiles.csv",
        "ml_feature_matrix": output_dir / "ml_feature_matrix.csv",
        "ml_accounts": output_dir / "ml_accounts.csv",
        "ml_clusters": output_dir / "ml_clusters.csv",
        "summary": output_dir / "summary.json",
        "report": output_dir / "report.md",
    }

    _write_dataclass_csv(results.normalized_tweets, written["normalized_tweets"])
    _write_dataclass_csv(results.account_summaries, written["account_summary"])
    _write_dataclass_csv(results.cascade_summaries, written["retweet_cascades"])
    _write_dataclass_csv(results.user_cascade_summaries, written["retweet_user_cascades"])
    _write_dataclass_csv(results.retweet_edges, written["retweet_edges"])
    _write_dataclass_csv(results.hashtag_summaries, written["hashtag_summary"])
    _write_dataclass_csv(results.mention_summaries, written["mention_summary"])
    _write_dataclass_csv(results.overlap_summaries, written["account_overlap"])
    _write_dataclass_csv(results.network_nodes, written["network_nodes"])
    _write_dataclass_csv(results.network_components, written["network_components"])
    _write_dataclass_csv(results.account_roles, written["account_roles"])
    _write_dataclass_csv(results.amplified_targets, written["amplified_targets"])
    _write_dataclass_csv(results.cohorts, written["cohorts"])
    _write_cohort_members_csv(results.cohorts, written["cohort_members"])
    _write_dataclass_csv(results.cascade_propagation, written["cascade_propagation"])
    _write_spread_paths_jsonl(
        results.cascade_spread_paths, written["cascade_spread_paths"]
    )
    _write_dataclass_csv(
        results.first_retweeter_profiles, written["first_retweeter_profiles"]
    )
    _write_dataclass_csv(results.feature_rows, written["ml_feature_matrix"])
    _write_dataclass_csv(results.ml_account_summaries, written["ml_accounts"])
    _write_dataclass_csv(results.ml_cluster_summaries, written["ml_clusters"])

    written["summary"].write_text(
        json.dumps(build_summary_payload(results), ensure_ascii=False, indent=2)
    )
    written["report"].write_text(render_markdown_report(results))
    return written


def build_summary_payload(results: AnalysisResults) -> dict[str, Any]:
    """Build a compact JSON payload for downstream tools."""

    return {
        "inputs": {
            "members_file": str(results.dataset_paths.members_file),
            "exporter_json": str(results.dataset_paths.exporter_json),
            "extra_members_file": (
                str(results.dataset_paths.extra_members_file)
                if results.dataset_paths.extra_members_file is not None
                else None
            ),
        },
        "stats": _dataclass_to_jsonable(results.dataset_stats),
        "top_accounts": [_dataclass_to_jsonable(item) for item in results.account_summaries[:15]],
        "top_cascades": [_dataclass_to_jsonable(item) for item in results.cascade_summaries[:15]],
        "top_user_cascades": [
            _dataclass_to_jsonable(item) for item in results.user_cascade_summaries[:15]
        ],
        "top_hashtags": [_dataclass_to_jsonable(item) for item in results.hashtag_summaries[:15]],
        "top_mentions": [_dataclass_to_jsonable(item) for item in results.mention_summaries[:15]],
        "top_overlap_pairs": [
            _dataclass_to_jsonable(item) for item in results.overlap_summaries[:15]
        ],
        "network_components": [
            _dataclass_to_jsonable(item) for item in results.network_components[:15]
        ],
        "network_nodes": [
            _dataclass_to_jsonable(item) for item in results.network_nodes[:15]
        ],
        "account_roles": [
            _dataclass_to_jsonable(item) for item in results.account_roles[:25]
        ],
        "amplified_targets": [
            _dataclass_to_jsonable(item) for item in results.amplified_targets[:25]
        ],
        "cohorts": [_dataclass_to_jsonable(item) for item in results.cohorts[:25]],
        "cascade_propagation": [
            _dataclass_to_jsonable(item) for item in results.cascade_propagation[:15]
        ],
        "first_retweeter_profiles": [
            _dataclass_to_jsonable(item) for item in results.first_retweeter_profiles[:25]
        ],
        "ml": {
            "run": _dataclass_to_jsonable(results.ml_run_summary),
            "clusters": [
                _dataclass_to_jsonable(item) for item in results.ml_cluster_summaries[:15]
            ],
            "top_anomalies": [
                _dataclass_to_jsonable(item) for item in results.ml_account_summaries[:15]
            ],
        },
    }


def render_markdown_report(results: AnalysisResults) -> str:
    """Render a factual Markdown report from one analysis run."""

    stats = results.dataset_stats
    top_accounts = results.account_summaries[:10]
    top_cascades = results.cascade_summaries[:10]
    top_user_cascades = results.user_cascade_summaries[:10]
    top_hashtags = results.hashtag_summaries[:10]
    top_mentions = results.mention_summaries[:10]
    top_pairs = results.overlap_summaries[:10]
    top_network_components = results.network_components[:10]
    top_network_nodes = results.network_nodes[:10]
    ml_run_summary = results.ml_run_summary
    top_ml_clusters = results.ml_cluster_summaries[:10]
    top_ml_anomalies = results.ml_account_summaries[:10]

    lines: list[str] = []
    lines.append("# zerosixty analysis report")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- members in list export: {stats.member_count}")
    lines.append(f"- tweet rows in exporter bundle: {stats.tweet_count}")
    lines.append(f"- active author handles in sample: {stats.active_account_count}")
    lines.append(f"- retweets: {stats.retweet_count}")
    lines.append(f"- originals or quotes: {stats.original_count}")
    lines.append(f"- sample start: {_fmt_dt(stats.date_start)}")
    lines.append(f"- sample end: {_fmt_dt(stats.date_end)}")
    if stats.missing_member_handles:
        lines.append(
            "- author handles present in tweets but missing from member export: "
            + ", ".join(stats.missing_member_handles)
        )

    lines.append("")
    lines.append("## Coordination indicators")
    lines.append("")
    lines.extend(
        (
            "- "
            f"`{account.account_handle}` score={account.coordination_score} "
            f"tweets={account.tweet_count} retweet_ratio={account.retweet_ratio:.2f} "
            f"first_retweeter_count={account.first_retweeter_count} "
            f"retweets_to_member_ratio={account.retweets_to_member_ratio:.2f} "
            f"top_amplified={account.top_amplified_account or '-'}"
        )
        for account in top_accounts
    )

    lines.append("")
    lines.append("## Top retweet cascades")
    lines.append("")
    for cascade in top_cascades:
        lines.append(
            "- "
            f"`{cascade.original_author_handle}` retweeted {cascade.retweet_count} times; "
            f"first seen via `{cascade.first_retweeter or '-'}` "
            f"at {_fmt_dt(cascade.first_retweet_at)}; "
            f"within_15m={cascade.retweets_within_15m}; within_60m={cascade.retweets_within_60m}"
        )
        lines.append(f"  text: {cascade.sample_text[:200]}")

    lines.append("")
    lines.append("## Most retweeted source accounts")
    lines.append("")
    lines.extend(
        (
            "- "
            f"`{user.original_author_handle}` total_retweets={user.total_retweet_count} "
            f"retweeted_posts={user.unique_retweeted_tweet_count} "
            f"unique_retweeters={user.unique_retweeter_count} "
            f"largest_single_post_cascade={user.top_cascade_retweet_count} "
            f"within_15m={user.total_retweets_within_15m} "
            f"within_60m={user.total_retweets_within_60m}"
        )
        for user in top_user_cascades
    )

    lines.append("")
    lines.append("## Hashtags")
    lines.append("")
    lines.extend(
        (
            "- "
            f"`{token.display_token}` normalized=`{token.token}` count={token.count} "
            f"unique_accounts={token.unique_account_count}"
        )
        for token in top_hashtags
    )

    lines.append("")
    lines.append("## Mentions")
    lines.append("")
    lines.extend(
        (
            "- "
            f"`@{token.display_token}` normalized=`{token.token}` count={token.count} "
            f"unique_accounts={token.unique_account_count}"
        )
        for token in top_mentions
    )

    lines.append("")
    lines.append("## Shared retweet overlap")
    lines.append("")
    lines.extend(
        (
            "- "
            f"`{pair.account_a}` + `{pair.account_b}` shared_retweets={pair.shared_retweets} "
            f"within_15m={pair.shared_within_15m} within_60m={pair.shared_within_60m} "
            f"jaccard={pair.jaccard:.4f}"
        )
        for pair in top_pairs
    )

    lines.append("")
    lines.append("## Shared-retweet network")
    lines.append("")
    lines.extend(
        (
            "- "
            f"component={component.component_id} nodes={component.node_count} "
            f"edges={component.edge_count} density={component.density:.4f} "
            f"shared_retweets={component.total_shared_retweets} "
            f"top_accounts={', '.join(component.top_accounts[:5])}"
        )
        for component in top_network_components
    )
    if top_network_nodes:
        lines.append("")
        lines.append("Top network nodes:")
        lines.extend(
            (
                "- "
                f"`{node.account_handle}` component={node.component_id} "
                f"weighted_degree={node.weighted_degree} neighbors={node.neighbor_count} "
                f"strongest_neighbor={node.strongest_neighbor or '-'} "
                f"max_shared_edge={node.max_shared_edge}"
            )
            for node in top_network_nodes
        )

    lines.append("")
    lines.append("## ML baseline")
    lines.append("")
    lines.append(
        "- "
        f"status={ml_run_summary.status} "
        f"samples={ml_run_summary.sample_count} "
        f"input_features={ml_run_summary.input_feature_count} "
        f"clusters={ml_run_summary.cluster_count} "
        f"cluster_selection={ml_run_summary.cluster_selection}"
    )
    if ml_run_summary.note:
        lines.append(f"- note: {ml_run_summary.note}")
    if ml_run_summary.status == "ready":
        lines.append(
            "- "
            f"models: clusters={ml_run_summary.cluster_model} "
            f"anomaly={ml_run_summary.anomaly_model} "
            f"embedding={ml_run_summary.embedding_model}"
        )
        lines.append("")
        lines.append("Largest ML clusters:")
        lines.extend(
            (
                "- "
                f"cluster={cluster.cluster_id} accounts={cluster.account_count} "
                f"mean_coordination_score={cluster.mean_coordination_score:.4f} "
                f"mean_retweet_ratio={cluster.mean_retweet_ratio:.4f} "
                f"mean_network_weighted_degree={cluster.mean_network_weighted_degree:.4f} "
                f"top_accounts={', '.join(cluster.top_accounts[:5])}"
            )
            for cluster in top_ml_clusters
        )
        if top_ml_anomalies:
            lines.append("")
            lines.append("Top ML anomalies:")
            lines.extend(
                (
                    "- "
                    f"`{anomaly.account_handle}` cluster={anomaly.cluster_id} "
                    f"anomaly_rank={anomaly.anomaly_rank} "
                    f"anomaly_score={anomaly.anomaly_score:.4f} "
                    f"centroid_distance={anomaly.centroid_distance:.4f} "
                    f"coordination_score={anomaly.coordination_score:.2f}"
                )
                for anomaly in top_ml_anomalies
            )

    lines.append("")
    lines.append("## Amplified targets (top reach + concentration)")
    lines.append("")
    for target in results.amplified_targets[:10]:
        lines.append(
            "- "
            f"`{target.amplified_author_handle}` "
            f"score={target.amplification_score:.2f} "
            f"inbound={target.total_inbound_retweets} "
            f"unique_amplifiers={target.unique_amplifier_count} "
            f"hhi={target.amplification_hhi:.4f} "
            f"repeat_overlap={target.repeat_crew_overlap:.4f} "
            f"within_15m={target.within_15m_inbound} "
            f"top_amplifier={target.top_amplifier_handle or '-'}({target.top_amplifier_share:.2f}) "
            f"roles={target.captured_role_mix or '-'}"
        )

    lines.append("")
    lines.append("## Recurrent cohorts")
    lines.append("")
    if not results.cohorts:
        lines.append("- (none detected above threshold)")
    for cohort in results.cohorts[:10]:
        lines.append(
            "- "
            f"cohort={cohort.cohort_id} score={cohort.cohort_score:.2f} "
            f"members={cohort.member_count} cascades={cohort.cascade_count} "
            f"tightness_sec={cohort.median_time_tightness_sec if cohort.median_time_tightness_sec is not None else '-'} "
            f"target_hhi={cohort.target_concentration_hhi:.4f} "
            f"top_target={cohort.top_target_author or '-'}({cohort.top_target_count}) "
            f"sample={', '.join(cohort.members[:6])}"
        )

    lines.append("")
    lines.append("## Cascade propagation")
    lines.append("")
    for propagation in results.cascade_propagation[:10]:
        lines.append(
            "- "
            f"`{propagation.original_author_handle}` "
            f"score={propagation.propagation_score:.2f} "
            f"retweets={propagation.retweet_count} "
            f"t_k10={propagation.time_to_k10_sec if propagation.time_to_k10_sec is not None else '-'}s "
            f"burst_kurtosis={propagation.burst_kurtosis:.2f} "
            f"same_client={propagation.same_client_share:.2f} "
            f"cohorts={list(propagation.amplifier_cohort_ids)} "
            f"components={list(propagation.amplifier_component_ids)[:4]}"
        )
        if propagation.spread_path_preview:
            lines.append(f"  path: {propagation.spread_path_preview}")

    lines.append("")
    lines.append("## Repeat first-retweeters")
    lines.append("")
    for profile in results.first_retweeter_profiles[:10]:
        lines.append(
            "- "
            f"`{profile.account_handle}` "
            f"first_count={profile.first_retweeter_count} "
            f"unique_targets={profile.unique_targets_first} "
            f"target_hhi={profile.target_hhi:.4f} "
            f"top_target={profile.top_target_author or '-'}({profile.top_target_count}) "
            f"role={profile.role_label} "
            f"cohorts={list(profile.cohort_ids)}"
        )

    lines.append("")
    lines.append("## Account roles (deterministic)")
    lines.append("")
    role_counts: dict[str, int] = {}
    for role in results.account_roles:
        role_counts[role.role_label] = role_counts.get(role.role_label, 0) + 1
    for label, count in sorted(role_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {label}: {count}")
    if results.account_roles:
        lines.append("")
        lines.append("Highest-confidence role assignments:")
        for role in sorted(
            results.account_roles,
            key=lambda item: (-item.role_confidence, item.account_handle),
        )[:10]:
            lines.append(
                "- "
                f"`{role.account_handle}` role={role.role_label} "
                f"confidence={role.role_confidence:.2f} "
                f"signals={', '.join(role.signals[:4]) or '-'}"
            )

    lines.append("")
    lines.append("## Limits")
    lines.append("")
    lines.append("- This report is based on one export window, not long-term behavior.")
    lines.append("- A high score is a review signal. It is not proof of automation or payment.")
    lines.append("- Retweet timing only reflects events present in this sample.")
    lines.append(
        "- The current ML lane is unsupervised and review-oriented. "
        "It does not assign truth labels."
    )
    return "\n".join(lines) + "\n"


def _write_dataclass_csv(items: Iterable[Any], path: Path) -> None:
    rows = list(items)
    if not rows:
        path.write_text("")
        return
    if not is_dataclass(rows[0]):
        raise TypeError(f"Expected dataclass instances for {path}.")

    header = [field.name for field in fields(rows[0])]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for item in rows:
            row = {
                key: _csv_value(value)
                for key, value in asdict(item).items()
            }
            writer.writerow(row)


def _write_cohort_members_csv(items: Iterable[Any], path: Path) -> None:
    """Write one row per (cohort_id, member_handle) for easier joining."""

    cohorts = list(items)
    header = ["cohort_id", "member_handle", "cohort_score", "cohort_size"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for cohort in cohorts:
            for handle_value in cohort.members:
                writer.writerow(
                    {
                        "cohort_id": cohort.cohort_id,
                        "member_handle": handle_value,
                        "cohort_score": cohort.cohort_score,
                        "cohort_size": cohort.member_count,
                    }
                )


def _write_spread_paths_jsonl(items: Iterable[Any], path: Path) -> None:
    """Write cascade spread paths as newline-delimited JSON records."""

    paths = list(items)
    with path.open("w", encoding="utf-8") as handle:
        for spread_path in paths:
            payload = {
                "original_tweet_id": spread_path.original_tweet_id,
                "original_author_handle": spread_path.original_author_handle,
                "first_retweet_at": (
                    spread_path.first_retweet_at.isoformat()
                    if spread_path.first_retweet_at is not None
                    else None
                ),
                "steps": [
                    {
                        "handle": handle_value,
                        "elapsed_sec": elapsed,
                        "component_id": component_id,
                        "cohort_id": cohort_id,
                    }
                    for handle_value, elapsed, component_id, cohort_id in spread_path.steps
                ],
            }
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")


def _csv_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return "|".join(str(item) for item in value)
    return value


def _dataclass_to_jsonable(item: Any) -> dict[str, Any]:
    payload = asdict(item)
    return {key: _json_value(value) for key, value in payload.items()}


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return list(value)
    return value


def _fmt_dt(value: datetime | None) -> str:
    if value is None:
        return "-"
    return value.isoformat()
