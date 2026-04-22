from __future__ import annotations

import json
from typing import TYPE_CHECKING

from zerosixty.curation import (
    build_clean_batches,
    discover_clean_batch_plans,
    resolve_clean_batch,
)
from zerosixty.pipeline import run_pipeline, run_pipeline_clean

if TYPE_CHECKING:
    from pathlib import Path


def test_discover_clean_batch_plans_pairs_member_snapshot_not_newer_than_exporter(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "datasets-raw"
    raw_dir.mkdir()

    _write_members_json(raw_dir / "twitter-ListMembers-100.json", ["alice"])
    _write_members_json(raw_dir / "twitter-ListMembers-200.json", ["alice", "bob"])

    _write_exporter_json(
        raw_dir / "twitter-web-exporter-150.json",
        [_tweet_row("t-1", "alice", "hello")],
    )
    _write_exporter_json(
        raw_dir / "twitter-web-exporter-250.json",
        [_tweet_row("t-2", "bob", "hello")],
    )

    plans = discover_clean_batch_plans(raw_dir)

    assert len(plans) == 2
    assert plans[0].exporter_file.path.name == "twitter-web-exporter-150.json"
    assert plans[0].members_file.path.name == "twitter-ListMembers-100.json"
    assert plans[1].exporter_file.path.name == "twitter-web-exporter-250.json"
    assert plans[1].members_file.path.name == "twitter-ListMembers-200.json"


def test_build_clean_batches_creates_versioned_batches_and_skips_unchanged(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "datasets-raw"
    clean_dir = tmp_path / "datasets-clean"
    raw_dir.mkdir()

    _write_members_json(raw_dir / "twitter-ListMembers-100.json", ["alice"])
    _write_members_json(raw_dir / "twitter-ListMembers-200.json", ["alice", "bob"])

    _write_exporter_json(
        raw_dir / "twitter-web-exporter-150.json",
        [_tweet_row("t-1", "alice", "first sample")],
    )
    _write_exporter_json(
        raw_dir / "twitter-web-exporter-250.json",
        [_tweet_row("t-2", "bob", "second sample")],
    )
    (raw_dir / "manual_member_handles_240.txt").write_text("@carol\n")

    initial_results, index_path = build_clean_batches(raw_dir, clean_dir)

    assert len(initial_results) == 2
    assert all(item.status == "built" for item in initial_results)
    assert index_path.exists()
    assert (initial_results[0].output_dir / "members.csv").exists()
    assert (initial_results[0].output_dir / "tweets.csv").exists()
    assert (initial_results[0].output_dir / "manifest.json").exists()
    assert initial_results[0].member_count == 1
    assert initial_results[1].member_count == 3

    second_results, _ = build_clean_batches(raw_dir, clean_dir)

    assert len(second_results) == 2
    assert all(item.status == "skipped" for item in second_results)


def test_build_clean_batches_builds_only_new_batch_after_new_exporter_added(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "datasets-raw"
    clean_dir = tmp_path / "datasets-clean"
    raw_dir.mkdir()

    _write_members_json(raw_dir / "twitter-ListMembers-100.json", ["alice"])
    _write_exporter_json(
        raw_dir / "twitter-web-exporter-150.json",
        [_tweet_row("t-1", "alice", "first sample")],
    )

    initial_results, _ = build_clean_batches(raw_dir, clean_dir)
    assert len(initial_results) == 1
    assert initial_results[0].status == "built"

    _write_exporter_json(
        raw_dir / "twitter-web-exporter-300.json",
        [
            _tweet_row("t-1", "alice", "first sample"),
            _tweet_row("t-2", "alice", "new sample"),
        ],
    )

    rerun_results, index_path = build_clean_batches(raw_dir, clean_dir)
    statuses = [item.status for item in rerun_results]

    assert len(rerun_results) == 2
    assert statuses.count("built") == 1
    assert statuses.count("skipped") == 1

    payload = json.loads(index_path.read_text())
    assert payload["batch_count"] == 2
    assert payload["latest_batch_id"] == rerun_results[-1].batch_id


def test_resolve_clean_batch_defaults_to_latest(tmp_path: Path) -> None:
    raw_dir = tmp_path / "datasets-raw"
    clean_dir = tmp_path / "datasets-clean"
    raw_dir.mkdir()

    _write_members_json(raw_dir / "twitter-ListMembers-100.json", ["alice"])
    _write_exporter_json(
        raw_dir / "twitter-web-exporter-150.json",
        [_tweet_row("t-1", "alice", "first sample")],
    )
    _write_exporter_json(
        raw_dir / "twitter-web-exporter-200.json",
        [_tweet_row("t-2", "alice", "second sample")],
    )

    build_clean_batches(raw_dir, clean_dir)
    resolved = resolve_clean_batch(clean_dir)

    assert resolved.batch_id == "batch_exporter-200__members-100"
    assert resolved.members_csv.exists()
    assert resolved.tweets_csv.exists()


def test_run_pipeline_clean_matches_raw_pipeline_outputs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "datasets-raw"
    clean_dir = tmp_path / "datasets-clean"
    raw_output_dir = tmp_path / "outputs-raw"
    clean_output_dir = tmp_path / "outputs-clean"
    raw_dir.mkdir()

    members_path = raw_dir / "twitter-ListMembers-100.json"
    exporter_path = raw_dir / "twitter-web-exporter-150.json"
    extra_members_path = raw_dir / "manual_member_handles_120.txt"

    _write_members_json(members_path, ["alice"])
    _write_exporter_json(
        exporter_path,
        [
            _tweet_row("t-1", "alice", "first sample"),
            _retweet_row("t-2", "bob", "source-a", "shared retweet"),
            _retweet_row("t-3", "alice", "source-a", "shared retweet"),
        ],
    )
    extra_members_path.write_text("@bob\n")

    raw_results, raw_written = run_pipeline(
        input_dir=raw_dir,
        output_dir=raw_output_dir,
        members_file=members_path,
        exporter_json=exporter_path,
        extra_members_file=extra_members_path,
    )

    build_clean_batches(raw_dir, clean_dir)
    batch = resolve_clean_batch(clean_dir)
    clean_results, clean_written = run_pipeline_clean(
        members_csv=batch.members_csv,
        tweets_csv=batch.tweets_csv,
        output_dir=clean_output_dir,
        source_members_file=batch.source_members_file,
        source_exporter_file=batch.source_exporter_file,
        source_extra_members_file=batch.source_extra_members_file,
    )

    assert raw_results.dataset_stats == clean_results.dataset_stats
    assert raw_results.account_summaries == clean_results.account_summaries
    assert raw_results.cascade_summaries == clean_results.cascade_summaries
    assert raw_results.user_cascade_summaries == clean_results.user_cascade_summaries
    assert raw_results.ml_run_summary == clean_results.ml_run_summary
    assert raw_results.ml_account_summaries == clean_results.ml_account_summaries
    assert raw_results.ml_cluster_summaries == clean_results.ml_cluster_summaries

    for name in (
        "normalized_tweets",
        "account_summary",
        "retweet_cascades",
        "retweet_user_cascades",
        "retweet_edges",
        "hashtag_summary",
        "mention_summary",
        "account_overlap",
        "network_nodes",
        "network_components",
        "ml_feature_matrix",
        "ml_accounts",
        "ml_clusters",
        "summary",
        "report",
    ):
        assert raw_written[name].read_text() == clean_written[name].read_text()


def _write_members_json(path: Path, handles: list[str]) -> None:
    rows = [
        {
            "id": str(index + 1),
            "screen_name": handle,
            "name": handle.title(),
            "description": "",
            "followers_count": 0,
            "friends_count": 0,
            "statuses_count": 0,
            "favourites_count": 0,
            "listed_count": 0,
            "location": "",
            "is_blue_verified": False,
            "protected": False,
            "created_at": "2026-04-16 10:00:00 +00:00",
            "metadata": {"legacy": {}},
        }
        for index, handle in enumerate(handles)
    ]
    path.write_text(json.dumps(rows))


def _write_exporter_json(path: Path, rows: list[dict[str, object]]) -> None:
    payload = {
        "data": {
            "data": [
                {
                    "tableName": "tweets",
                    "rows": rows,
                }
            ]
        }
    }
    path.write_text(json.dumps(payload))


def _tweet_row(tweet_id: str, handle: str, text: str) -> dict[str, object]:
    return {
        "rest_id": tweet_id,
        "source": '<a href="http://twitter.com" rel="nofollow">Twitter Web App</a>',
        "core": {
            "user_results": {
                "result": {
                    "core": {
                        "screen_name": handle,
                        "name": handle.title(),
                    }
                }
            }
        },
        "legacy": {
            "created_at": "Wed Apr 16 10:55:44 +0000 2026",
            "full_text": text,
            "entities": {
                "hashtags": [],
                "user_mentions": [],
            },
        },
    }


def _retweet_row(tweet_id: str, handle: str, source_handle: str, text: str) -> dict[str, object]:
    created_at = "Wed Apr 16 10:55:44 +0000 2026"
    source_payload = {
        "rest_id": f"{tweet_id}-source",
        "core": {
            "user_results": {
                "result": {
                    "core": {
                        "screen_name": source_handle,
                        "name": source_handle.title(),
                    }
                }
            }
        },
        "legacy": {
            "created_at": created_at,
            "full_text": text,
            "entities": {
                "hashtags": [],
                "user_mentions": [],
            },
        },
    }
    row = _tweet_row(tweet_id, handle, f"RT @{source_handle}: {text}")
    row["legacy"] = {
        "created_at": created_at,
        "full_text": f"RT @{source_handle}: {text}",
        "retweeted_status_result": {
            "result": source_payload,
        },
        "entities": {
            "hashtags": [],
            "user_mentions": [],
        },
    }
    return row
