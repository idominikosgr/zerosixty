from __future__ import annotations

from zerosixty.normalize import (
    build_tweet_records,
    canonicalize_text,
    extract_retweet_handle_from_text,
    normalize_token,
)


def test_normalize_token_strips_accents_and_final_sigma() -> None:
    assert normalize_token("Μυλωνάκης") == "μυλωνακησ"


def test_extract_retweet_handle_from_text() -> None:
    assert extract_retweet_handle_from_text("RT @Example_User: hello") == "Example_User"


def test_canonicalize_text_removes_urls_and_collapses_spaces() -> None:
    assert canonicalize_text("A   test https://x.com/demo   text") == "a test text"


def test_build_tweet_records_prefers_note_tweet_text() -> None:
    rows = [
        {
            "rest_id": "1",
            "source": (
                '<a href="http://twitter.com/download/android" rel="nofollow">'
                "Twitter for Android</a>"
            ),
            "core": {
                "user_results": {
                    "result": {
                        "core": {
                            "screen_name": "alice",
                            "name": "Alice",
                        }
                    }
                }
            },
            "legacy": {
                "created_at": "Wed Apr 15 10:55:44 +0000 2026",
                "full_text": "short version",
                "entities": {
                    "hashtags": [],
                    "user_mentions": [],
                },
            },
            "note_tweet": {
                "note_tweet_results": {
                    "result": {
                        "text": "long version",
                        "entity_set": {
                            "hashtags": [{"text": "Μυλωνάκης"}],
                            "user_mentions": [{"screen_name": "Somebody"}],
                        },
                    }
                }
            },
        }
    ]

    records = build_tweet_records(rows)

    assert len(records) == 1
    assert records[0].text == "long version"
    assert records[0].hashtags == ("μυλωνακησ",)
    assert records[0].mentions == ("somebody",)
