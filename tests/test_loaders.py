from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

from zerosixty.loaders import (
    load_clean_member_records,
    load_clean_tweet_records,
    load_extra_member_handles,
    load_member_records,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_load_member_records_supports_json_exports(tmp_path: Path) -> None:
    path = tmp_path / "twitter-ListMembers-sample.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "1",
                    "screen_name": "alice",
                    "name": "Alice",
                    "description": "sample",
                    "followers_count": 10,
                    "friends_count": 20,
                    "statuses_count": 30,
                    "favourites_count": 40,
                    "listed_count": 2,
                    "location": "Athens",
                    "is_blue_verified": True,
                    "protected": False,
                    "created_at": "2026-04-16 10:00:00 +00:00",
                    "metadata": {
                        "legacy": {
                            "default_profile": False,
                            "default_profile_image": True,
                        },
                        "profile_description_language": "el",
                    },
                }
            ]
        )
    )

    records = load_member_records(path)

    assert len(records) == 1
    assert records[0].screen_name == "alice"
    assert records[0].followers_count == 10
    assert records[0].default_profile is False
    assert records[0].default_profile_image is True
    assert records[0].profile_description_language == "el"


def test_load_member_records_appends_missing_extra_handles(tmp_path: Path) -> None:
    path = tmp_path / "twitter-ListMembers-sample.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "1",
                    "screen_name": "alice",
                    "name": "Alice",
                    "description": "",
                    "followers_count": 10,
                    "friends_count": 20,
                    "statuses_count": 30,
                    "favourites_count": 40,
                    "listed_count": 2,
                    "location": "",
                    "is_blue_verified": False,
                    "protected": False,
                    "created_at": "2026-04-16 10:00:00 +00:00",
                    "metadata": {"legacy": {}},
                }
            ]
        )
    )

    records = load_member_records(
        path,
        extra_handles=("@alice", "https://x.com/bob"),
    )
    handles = {record.screen_name for record in records}
    bob = next(record for record in records if record.screen_name == "bob")

    assert handles == {"alice", "bob"}
    assert bob.member_id == "bob"
    assert bob.followers_count == 0
    assert bob.created_at is None


def test_load_extra_member_handles_parses_urls_and_handles(tmp_path: Path) -> None:
    path = tmp_path / "extra_members.txt"
    path.write_text(
        "\n".join(
            [
                "https://x.com/alice",
                "@bob",
                "carol",
                "",
                "https://twitter.com/alice",
            ]
        )
    )

    handles = load_extra_member_handles(path)

    assert handles == ["alice", "bob", "carol"]


def test_load_clean_records_reads_members_and_tweets_csv(tmp_path: Path) -> None:
    members_csv = tmp_path / "members.csv"
    with members_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "member_id",
                "screen_name",
                "name",
                "description",
                "created_at",
                "followers_count",
                "friends_count",
                "statuses_count",
                "favourites_count",
                "listed_count",
                "location",
                "is_blue_verified",
                "protected",
                "default_profile",
                "default_profile_image",
                "profile_description_language",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "member_id": "1",
                "screen_name": "alice",
                "name": "Alice",
                "description": "",
                "created_at": "2026-04-16T10:00:00+00:00",
                "followers_count": "10",
                "friends_count": "20",
                "statuses_count": "30",
                "favourites_count": "40",
                "listed_count": "2",
                "location": "Athens",
                "is_blue_verified": "True",
                "protected": "False",
                "default_profile": "False",
                "default_profile_image": "True",
                "profile_description_language": "el",
            }
        )

    tweets_csv = tmp_path / "tweets.csv"
    with tweets_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tweet_id",
                "author_handle",
                "author_name",
                "posted_at",
                "source_client",
                "text",
                "canonical_text",
                "is_retweet",
                "is_quote",
                "content_tweet_id",
                "content_author_handle",
                "content_author_name",
                "content_created_at",
                "content_text",
                "content_canonical_text",
                "hashtags_raw",
                "hashtags",
                "mentions_raw",
                "mentions",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "tweet_id": "t1",
                "author_handle": "alice",
                "author_name": "Alice",
                "posted_at": "2026-04-16T10:55:44+00:00",
                "source_client": "Twitter Web App",
                "text": "hello",
                "canonical_text": "hello",
                "is_retweet": "False",
                "is_quote": "False",
                "content_tweet_id": "t1",
                "content_author_handle": "alice",
                "content_author_name": "Alice",
                "content_created_at": "2026-04-16T10:55:44+00:00",
                "content_text": "hello",
                "content_canonical_text": "hello",
                "hashtags_raw": json.dumps(["Tag"]),
                "hashtags": json.dumps(["tag"]),
                "mentions_raw": json.dumps(["Bob"]),
                "mentions": json.dumps(["bob"]),
            }
        )

    members = load_clean_member_records(members_csv)
    tweets = load_clean_tweet_records(tweets_csv)

    assert len(members) == 1
    assert members[0].screen_name == "alice"
    assert members[0].default_profile is False
    assert members[0].default_profile_image is True
    assert members[0].profile_description_language == "el"

    assert len(tweets) == 1
    assert tweets[0].tweet_id == "t1"
    assert tweets[0].is_retweet is False
    assert tweets[0].hashtags_raw == ("Tag",)
    assert tweets[0].hashtags == ("tag",)
    assert tweets[0].mentions_raw == ("Bob",)
    assert tweets[0].mentions == ("bob",)
