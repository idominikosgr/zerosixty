from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from zerosixty.models import AnalysisResults


def write_outputs(results: AnalysisResults, output_dir: Path) -> dict[str, Path]:
    """Write all flat-file outputs for one analysis run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "normalized_tweets": output_dir / "normalized_tweets.csv",
        "account_summary": output_dir / "account_summary.csv",
        "retweet_cascades": output_dir / "retweet_cascades.csv",
        "retweet_edges": output_dir / "retweet_edges.csv",
        "hashtag_summary": output_dir / "hashtag_summary.csv",
        "mention_summary": output_dir / "mention_summary.csv",
        "account_overlap": output_dir / "account_overlap.csv",
        "network_nodes": output_dir / "network_nodes.csv",
        "network_components": output_dir / "network_components.csv",
        "ml_feature_matrix": output_dir / "ml_feature_matrix.csv",
        "summary": output_dir / "summary.json",
        "report": output_dir / "report.md",
    }

    _write_dataclass_csv(results.normalized_tweets, written["normalized_tweets"])
    _write_dataclass_csv(results.account_summaries, written["account_summary"])
    _write_dataclass_csv(results.cascade_summaries, written["retweet_cascades"])
    _write_dataclass_csv(results.retweet_edges, written["retweet_edges"])
    _write_dataclass_csv(results.hashtag_summaries, written["hashtag_summary"])
    _write_dataclass_csv(results.mention_summaries, written["mention_summary"])
    _write_dataclass_csv(results.overlap_summaries, written["account_overlap"])
    _write_dataclass_csv(results.network_nodes, written["network_nodes"])
    _write_dataclass_csv(results.network_components, written["network_components"])
    _write_dataclass_csv(results.feature_rows, written["ml_feature_matrix"])

    written["summary"].write_text(
        json.dumps(build_summary_payload(results), ensure_ascii=False, indent=2)
    )
    written["report"].write_text(render_markdown_report(results))
    return written


def build_summary_payload(results: AnalysisResults) -> dict[str, Any]:
    """Build a compact JSON payload for downstream tools."""

    return {
        "inputs": {
            "members_csv": str(results.dataset_paths.members_csv),
            "exporter_json": str(results.dataset_paths.exporter_json),
        },
        "stats": _dataclass_to_jsonable(results.dataset_stats),
        "top_accounts": [_dataclass_to_jsonable(item) for item in results.account_summaries[:15]],
        "top_cascades": [_dataclass_to_jsonable(item) for item in results.cascade_summaries[:15]],
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
    }


def render_markdown_report(results: AnalysisResults) -> str:
    """Render a factual Markdown report from one analysis run."""

    stats = results.dataset_stats
    top_accounts = results.account_summaries[:10]
    top_cascades = results.cascade_summaries[:10]
    top_hashtags = results.hashtag_summaries[:10]
    top_mentions = results.mention_summaries[:10]
    top_pairs = results.overlap_summaries[:10]
    top_network_components = results.network_components[:10]
    top_network_nodes = results.network_nodes[:10]

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
            "- author handles present in tweets but missing from member CSV: "
            + ", ".join(stats.missing_member_handles)
        )

    lines.append("")
    lines.append("## Coordination indicators")
    lines.append("")
    for account in top_accounts:
        lines.append(
            "- "
            f"`{account.account_handle}` score={account.coordination_score} "
            f"tweets={account.tweet_count} retweet_ratio={account.retweet_ratio:.2f} "
            f"first_retweeter_count={account.first_retweeter_count} "
            f"retweets_to_member_ratio={account.retweets_to_member_ratio:.2f} "
            f"top_amplified={account.top_amplified_account or '-'}"
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
    lines.append("## Hashtags")
    lines.append("")
    for token in top_hashtags:
        lines.append(
            "- "
            f"`{token.display_token}` normalized=`{token.token}` count={token.count} "
            f"unique_accounts={token.unique_account_count}"
        )

    lines.append("")
    lines.append("## Mentions")
    lines.append("")
    for token in top_mentions:
        lines.append(
            "- "
            f"`@{token.display_token}` normalized=`{token.token}` count={token.count} "
            f"unique_accounts={token.unique_account_count}"
        )

    lines.append("")
    lines.append("## Shared retweet overlap")
    lines.append("")
    for pair in top_pairs:
        lines.append(
            "- "
            f"`{pair.account_a}` + `{pair.account_b}` shared_retweets={pair.shared_retweets} "
            f"within_15m={pair.shared_within_15m} within_60m={pair.shared_within_60m} "
            f"jaccard={pair.jaccard:.4f}"
        )

    lines.append("")
    lines.append("## Shared-retweet network")
    lines.append("")
    for component in top_network_components:
        lines.append(
            "- "
            f"component={component.component_id} nodes={component.node_count} "
            f"edges={component.edge_count} density={component.density:.4f} "
            f"shared_retweets={component.total_shared_retweets} "
            f"top_accounts={', '.join(component.top_accounts[:5])}"
        )
    if top_network_nodes:
        lines.append("")
        lines.append("Top network nodes:")
        for node in top_network_nodes:
            lines.append(
                "- "
                f"`{node.account_handle}` component={node.component_id} "
                f"weighted_degree={node.weighted_degree} neighbors={node.neighbor_count} "
                f"strongest_neighbor={node.strongest_neighbor or '-'} "
                f"max_shared_edge={node.max_shared_edge}"
            )

    lines.append("")
    lines.append("## Limits")
    lines.append("")
    lines.append("- This report is based on one export window, not long-term behavior.")
    lines.append("- A high score is a review signal. It is not proof of automation or payment.")
    lines.append("- Retweet timing only reflects events present in this sample.")
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
