from __future__ import annotations

from datetime import UTC, datetime, timedelta

from zerosixty.amplification import build_amplified_targets
from zerosixty.analyze import (
    build_account_summaries,
    build_cascade_summaries,
    build_overlap_network,
    build_overlap_summaries,
    build_user_cascade_summaries,
)
from zerosixty.crews import (
    build_cohorts,
    build_k_core_subgraph,
    cohort_ids_by_account,
    cohort_ids_by_cascade,
)
from zerosixty.models import MemberRecord, TweetRecord
from zerosixty.propagation import (
    build_cascade_propagation,
    build_first_retweeter_profiles,
)
from zerosixty.typology import build_account_roles


def test_build_account_roles_tags_source_hub_and_amplifier() -> None:
    members = [
        _member(
            "hub",
            description="Greek News Channel Official",
            listed=250,
            followers=500_000,
            friends=500,
        ),
        _member("amplifier", description="random user", listed=0, followers=20, friends=400),
        _member("retail", description="πατερας δυο παιδιων", listed=0, followers=80, friends=60),
    ]
    tweets = [
        _original("hub", "hub-1", 0),
        _original("hub", "hub-2", 5),
        _original("hub", "hub-3", 10),
        _retweet("amplifier", "hub-1", "hub", 1),
        _retweet("amplifier", "hub-2", "hub", 6),
        _retweet("amplifier", "hub-3", "hub", 11),
        _retweet("amplifier", "other-1", "hub", 15),
        _retweet("amplifier", "other-2", "hub", 20),
        _retweet("retail", "own-1", "hub", 25),
        _original("retail", "retail-1", 40),
    ]
    cascades = build_cascade_summaries(tweets)
    summaries = build_account_summaries(members, tweets, cascades)

    roles = build_account_roles(members, summaries)
    role_by_handle = {role.account_handle: role for role in roles}

    assert role_by_handle["hub"].role_label == "source_hub"
    assert role_by_handle["amplifier"].role_label == "amplifier_suspect"
    assert role_by_handle["retail"].role_label in {"retail_user", "mixed_behavior"}
    assert role_by_handle["hub"].role_confidence > 0.0


def test_build_cohorts_detects_time_tight_crew() -> None:
    tweets = _tight_crew_tweets()
    cascades = build_cascade_summaries(tweets)
    overlaps = build_overlap_summaries(tweets, min_shared=2)
    nodes, _ = build_overlap_network(overlaps)

    cohorts = build_cohorts(cascades, overlaps, nodes, min_cohort_size=3, random_state=7)

    assert cohorts, "expected at least one cohort from a tight crew"
    top = cohorts[0]
    assert top.member_count >= 3
    assert top.cascade_count >= 2
    assert "crew_a" in set(top.members)
    assert "crew_b" in set(top.members)
    assert "crew_c" in set(top.members)
    assert top.cohort_score > 0.0


def test_cohort_lookups_invert_membership() -> None:
    tweets = _tight_crew_tweets()
    cascades = build_cascade_summaries(tweets)
    overlaps = build_overlap_summaries(tweets, min_shared=2)
    nodes, _ = build_overlap_network(overlaps)

    cohorts = build_cohorts(cascades, overlaps, nodes, min_cohort_size=3, random_state=7)
    assert cohorts

    by_account = cohort_ids_by_account(cohorts)
    by_cascade = cohort_ids_by_cascade(cohorts, cascades)

    assert cohorts[0].cohort_id in by_account["crew_a"]
    assert cohorts[0].cohort_id in by_account["crew_b"]
    # at least one cascade should be tagged with the cohort.
    assert any(cohorts[0].cohort_id in ids for ids in by_cascade.values())


def test_build_k_core_subgraph_ranks_dense_nodes_higher() -> None:
    tweets = _tight_crew_tweets()
    overlaps = build_overlap_summaries(tweets, min_shared=1)

    core_rank = build_k_core_subgraph(overlaps, k=3, weight_key="shared_retweets")
    assert core_rank["crew_a"] >= core_rank.get("outsider", 0)


def test_build_amplified_targets_surfaces_concentration_signals() -> None:
    tweets = _tight_crew_tweets()
    cascades = build_cascade_summaries(tweets)
    user_cascades = build_user_cascade_summaries(cascades)
    overlaps = build_overlap_summaries(tweets, min_shared=2)
    nodes, _ = build_overlap_network(overlaps)
    members = {"crew_a", "crew_b", "crew_c", "outsider", "target"}

    targets = build_amplified_targets(
        tweets, cascades, user_cascades, members, nodes, account_roles=None
    )

    assert targets, "expected at least one amplified target"
    top = targets[0]
    assert top.amplified_author_handle == "target"
    assert top.total_inbound_retweets >= 6
    assert top.unique_amplifier_count >= 3
    assert 0.0 <= top.amplification_hhi <= 1.0
    assert 0.0 <= top.repeat_crew_overlap <= 1.0


def test_build_cascade_propagation_produces_metrics_and_paths() -> None:
    tweets = _tight_crew_tweets()
    cascades = build_cascade_summaries(tweets)
    overlaps = build_overlap_summaries(tweets, min_shared=2)
    nodes, _ = build_overlap_network(overlaps)
    members = {"crew_a", "crew_b", "crew_c", "outsider", "target"}

    propagation_summaries, spread_paths = build_cascade_propagation(
        tweets,
        cascades,
        members,
        nodes,
        account_roles=None,
        cohort_ids_by_cascade=None,
    )

    assert len(propagation_summaries) == len(cascades)
    assert len(spread_paths) == len(cascades)
    for summary, path in zip(propagation_summaries, spread_paths, strict=True):
        assert summary.retweet_count >= 0
        assert 0.0 <= summary.same_client_share <= 1.0
        assert summary.propagation_score >= 0.0
        assert path.original_tweet_id == summary.original_tweet_id


def test_build_first_retweeter_profiles_counts_repeat_leaders() -> None:
    tweets = _tight_crew_tweets()
    cascades = build_cascade_summaries(tweets)

    profiles = build_first_retweeter_profiles(
        cascades,
        account_roles=None,
        cohort_ids_by_account=None,
        network_nodes=[],
    )

    lead_counts = {profile.account_handle: profile.first_retweeter_count for profile in profiles}
    assert lead_counts.get("crew_a", 0) >= 2
    for profile in profiles:
        assert 0.0 <= profile.target_hhi <= 1.0
        assert profile.first_retweeter_count > 0


def _tight_crew_tweets() -> list[TweetRecord]:
    tweets: list[TweetRecord] = []
    # three cascades where crew_a/b/c all retweet `target` within 60s of each other.
    for index, original in enumerate(("target-1", "target-2", "target-3")):
        base_minute = index * 30
        tweets.extend(
            [
                _original("target", original, base_minute),
                _retweet("crew_a", original, "target", base_minute + 1),
                _retweet("crew_b", original, "target", base_minute + 2),
                _retweet("crew_c", original, "target", base_minute + 3),
                _retweet("outsider", original, "target", base_minute + 120),
            ]
        )
    # one cascade from a different target, only crew_a and outsider participate.
    tweets.extend(
        [
            _original("other_source", "extra-1", 200),
            _retweet("crew_a", "extra-1", "other_source", 201),
            _retweet("outsider", "extra-1", "other_source", 260),
        ]
    )
    return tweets


def _member(
    handle: str,
    *,
    description: str | None = None,
    listed: int = 1,
    followers: int = 100,
    friends: int = 50,
) -> MemberRecord:
    return MemberRecord(
        member_id=handle,
        screen_name=handle,
        name=handle.title(),
        description=description,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        followers_count=followers,
        friends_count=friends,
        statuses_count=200,
        favourites_count=20,
        listed_count=listed,
        location=None,
        is_blue_verified=False,
        protected=False,
        default_profile=True,
        default_profile_image=False,
        profile_description_language="el",
    )


def _retweet(
    author_handle: str,
    original_tweet_id: str,
    content_author_handle: str,
    minute_offset: int,
) -> TweetRecord:
    created = datetime(2026, 4, 15, 20, 0, tzinfo=UTC) + timedelta(minutes=minute_offset)
    return TweetRecord(
        tweet_id=f"{author_handle}-{original_tweet_id}",
        author_handle=author_handle,
        author_name=author_handle.title(),
        posted_at=created,
        source_client="Twitter Web App",
        text=f"RT @{content_author_handle}: sample",
        canonical_text="rt sample",
        is_retweet=True,
        is_quote=False,
        content_tweet_id=original_tweet_id,
        content_author_handle=content_author_handle,
        content_author_name=content_author_handle.title(),
        content_created_at=created,
        content_text="sample",
        content_canonical_text="sample",
        hashtags_raw=(),
        hashtags=(),
        mentions_raw=(),
        mentions=(),
    )


def _original(author_handle: str, tweet_id: str, minute_offset: int) -> TweetRecord:
    created = datetime(2026, 4, 15, 20, 0, tzinfo=UTC) + timedelta(minutes=minute_offset)
    return TweetRecord(
        tweet_id=tweet_id,
        author_handle=author_handle,
        author_name=author_handle.title(),
        posted_at=created,
        source_client="Twitter Web App",
        text="original",
        canonical_text="original",
        is_retweet=False,
        is_quote=False,
        content_tweet_id=tweet_id,
        content_author_handle=author_handle,
        content_author_name=author_handle.title(),
        content_created_at=created,
        content_text="original",
        content_canonical_text="original",
        hashtags_raw=(),
        hashtags=(),
        mentions_raw=(),
        mentions=(),
    )
