from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

from zerosixty.models import (
    CascadeSummary,
    CohortSummary,
    NetworkNodeSummary,
    OverlapSummary,
)


def build_cohorts(
    cascades: list[CascadeSummary],
    overlap_summaries: list[OverlapSummary],
    network_nodes: list[NetworkNodeSummary],
    *,
    min_shared_cascades: int = 2,
    min_cohort_size: int = 3,
    min_cohort_cascades: int = 2,
    max_tightness_sec: int | None = None,
    random_state: int = 42,
    min_tight_overlap: int = 2,
) -> list[CohortSummary]:
    """Detect recurrent cohorts of accounts that retweet the same originals together.

    Algorithm:
      1. Build a tight-co-activity graph from overlap pairs whose
         `shared_within_60m >= min_tight_overlap` (default 2). This removes
         bulk "same source but hours apart" edges which otherwise collapse
         label propagation into one giant cluster in dense samples.
      2. If the tight graph is empty, fall back to pairs with
         `shared_retweets >= max(min_shared_cascades, 3)` so some structure
         still surfaces on small, lightly co-active datasets.
      3. Weight each kept edge by
         `shared_within_15m*4 + shared_within_60m*2 + shared_retweets`.
      4. Run label propagation on this filtered graph to produce communities.
      5. For each community, pick cascades where >=30% of members participated
         (bounded by >=2 members) and derive timing + target metrics.

    `max_tightness_sec` filters out loose cohorts when set. By default we keep
    them and let the scoring rank tight cohorts higher.
    """

    if not cascades:
        return []

    cascade_lookup: dict[str, CascadeSummary] = {
        cascade.original_tweet_id: cascade for cascade in cascades
    }
    participation: dict[str, set[str]] = defaultdict(set)
    for cascade in cascades:
        for handle in cascade.retweeters:
            participation[handle].add(cascade.original_tweet_id)

    edge_weights = _build_tight_edge_weights(
        overlap_summaries,
        min_shared_cascades=min_shared_cascades,
        min_tight_overlap=min_tight_overlap,
    )
    if not edge_weights:
        edge_weights = _build_tight_edge_weights(
            overlap_summaries,
            min_shared_cascades=max(min_shared_cascades, 3),
            min_tight_overlap=1,
        )
    if not edge_weights:
        return []

    tight_only = {
        pair: overlap.shared_within_15m * 4 + overlap.shared_within_60m * 2
        for overlap in overlap_summaries
        for pair in [_sorted_pair(overlap.account_a, overlap.account_b)]
        if overlap.shared_within_15m >= 2
        or overlap.shared_within_60m >= max(3, min_tight_overlap + 1)
    }
    communities = _resolve_communities(
        edge_weights,
        tight_only_weights=tight_only,
        random_state=random_state,
        max_community_size=max(min_cohort_size * 6, 18),
    )

    component_lookup: dict[str, int] = {
        node.account_handle: node.component_id for node in network_nodes
    }

    cohorts: list[CohortSummary] = []
    cohort_id = 0
    for members in sorted(communities, key=lambda group: (-len(group), sorted(group)[0])):
        if len(members) < min_cohort_size:
            continue
        cohort_id += 1
        cascade_ids = _cohort_cascade_ids(members, participation, cascade_lookup)
        if len(cascade_ids) < min_cohort_cascades:
            cohort_id -= 1
            continue

        tightness, tightness_per_cascade = _cohort_time_tightness(
            members=members,
            cascade_ids=cascade_ids,
            cascade_lookup=cascade_lookup,
        )
        median_tightness = _median_or_none(tightness_per_cascade)
        if (
            max_tightness_sec is not None
            and median_tightness is not None
            and median_tightness > max_tightness_sec
        ):
            cohort_id -= 1
            continue

        target_counts: Counter[str] = Counter()
        for cascade_id in cascade_ids:
            cascade = cascade_lookup[cascade_id]
            target_counts[cascade.original_author_handle] += 1
        target_hhi = _herfindahl(target_counts)
        top_target = target_counts.most_common(1)[0] if target_counts else (None, 0)

        top_cascade_id, top_cascade_share = _top_cascade_by_participation(
            members=members,
            cascade_ids=cascade_ids,
            cascade_lookup=cascade_lookup,
        )

        component_ids = tuple(
            sorted(
                {
                    component_lookup[handle]
                    for handle in members
                    if handle in component_lookup
                }
            )
        )

        cohort_score = _cohort_score(
            member_count=len(members),
            cascade_count=len(cascade_ids),
            target_hhi=target_hhi,
            median_tightness=median_tightness,
            tightness_ref=tightness,
        )

        cohorts.append(
            CohortSummary(
                cohort_id=cohort_id,
                member_count=len(members),
                cascade_count=len(cascade_ids),
                median_time_tightness_sec=median_tightness,
                target_concentration_hhi=round(target_hhi, 4),
                top_target_author=top_target[0],
                top_target_count=top_target[1],
                unique_targets=len(target_counts),
                component_ids=component_ids,
                members=tuple(sorted(members)),
                top_cascade_tweet_id=top_cascade_id,
                top_cascade_member_share=round(top_cascade_share, 4),
                cohort_score=round(cohort_score, 4),
            )
        )

    cohorts.sort(
        key=lambda item: (
            -item.cohort_score,
            -item.member_count,
            -item.cascade_count,
            item.cohort_id,
        )
    )
    for index, cohort in enumerate(cohorts, start=1):
        cohorts[index - 1] = _with_cohort_id(cohort, index)
    return cohorts


def cohort_ids_by_account(cohorts: list[CohortSummary]) -> dict[str, tuple[int, ...]]:
    """Invert cohort membership: handle -> (cohort_id, ...)."""

    lookup: dict[str, list[int]] = defaultdict(list)
    for cohort in cohorts:
        for handle in cohort.members:
            lookup[handle].append(cohort.cohort_id)
    return {handle: tuple(sorted(ids)) for handle, ids in lookup.items()}


def cohort_ids_by_cascade(
    cohorts: list[CohortSummary],
    cascades: list[CascadeSummary],
) -> dict[str, tuple[int, ...]]:
    """Map each cascade to cohorts that meaningfully participated in it."""

    if not cohorts:
        return {}
    cohort_members = {cohort.cohort_id: set(cohort.members) for cohort in cohorts}
    mapping: dict[str, list[int]] = defaultdict(list)
    for cascade in cascades:
        retweeter_set = set(cascade.retweeters)
        for cohort_id, members in cohort_members.items():
            overlap_size = len(retweeter_set & members)
            if overlap_size >= 2 and overlap_size >= max(2, int(0.2 * len(members))):
                mapping[cascade.original_tweet_id].append(cohort_id)
    return {tweet_id: tuple(sorted(ids)) for tweet_id, ids in mapping.items()}


def _cohort_cascade_ids(
    members: set[str],
    participation: dict[str, set[str]],
    cascade_lookup: dict[str, CascadeSummary],
) -> list[str]:
    """Pick cascades where enough cohort members co-retweeted.

    Uses a log-scaled threshold so small cohorts need near-full participation
    while large cohorts need only a handful of members to signal a shared
    cascade. This prevents giant-community dilution from hiding genuine
    co-activity: for a 50-member cohort we only need ~5 co-retweets to
    count, not 15.
    """

    counts: Counter[str] = Counter()
    for handle in members:
        for cascade_id in participation.get(handle, set()):
            counts[cascade_id] += 1
    log_threshold = math.ceil(math.log(max(len(members), 2)) + 1)
    proportional_threshold = math.ceil(len(members) * 0.3)
    threshold = max(2, min(log_threshold, proportional_threshold, 4))
    return [
        cascade_id
        for cascade_id, count in counts.items()
        if count >= threshold and cascade_id in cascade_lookup
    ]


def _cohort_time_tightness(
    *,
    members: set[str],
    cascade_ids: list[str],
    cascade_lookup: dict[str, CascadeSummary],
) -> tuple[int | None, list[int]]:
    """Per-cascade average inter-retweet lag among cohort members.

    The CascadeSummary does not retain per-retweeter timestamps, only the
    cascade span and retweet count. We approximate the tightness of each
    cohort appearance by dividing the cascade span by the retweet count;
    cohorts that co-participate in tight bursts surface low values.
    """

    tightnesses: list[int] = []
    for cascade_id in cascade_ids:
        cascade = cascade_lookup[cascade_id]
        if cascade.first_retweet_at is None or cascade.last_retweet_at is None:
            continue
        member_in_cascade = sum(1 for handle in cascade.retweeters if handle in members)
        if member_in_cascade < 2:
            continue
        span = int((cascade.last_retweet_at - cascade.first_retweet_at).total_seconds())
        denominator = max(1, cascade.retweet_count - 1)
        tightnesses.append(max(0, span // denominator))
    if not tightnesses:
        return None, []
    return tightnesses[0], tightnesses


def _median_or_none(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2


def _top_cascade_by_participation(
    *,
    members: set[str],
    cascade_ids: list[str],
    cascade_lookup: dict[str, CascadeSummary],
) -> tuple[str | None, float]:
    best: tuple[str | None, float] = (None, 0.0)
    for cascade_id in cascade_ids:
        cascade = cascade_lookup[cascade_id]
        retweeter_set = set(cascade.retweeters)
        shared = len(retweeter_set & members)
        share = shared / len(members) if members else 0.0
        if share > best[1] or (share == best[1] and best[0] is None):
            best = (cascade.original_tweet_id, share)
    return best


def _herfindahl(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return sum((value / total) ** 2 for value in counts.values())


def _cohort_score(
    *,
    member_count: int,
    cascade_count: int,
    target_hhi: float,
    median_tightness: int | None,
    tightness_ref: int | None,
) -> float:
    """Prefer small-but-tight, target-concentrated, cascade-rich cohorts."""

    _ = tightness_ref
    size_term = math.log1p(member_count) * 4.0
    cascade_term = math.log1p(cascade_count) * 6.0
    hhi_term = target_hhi * 25.0
    tightness_term = 0.0
    if median_tightness is not None:
        tightness_term = max(0.0, 20.0 - min(median_tightness, 3600) / 180.0)
    return size_term + cascade_term + hhi_term + tightness_term


def _with_cohort_id(cohort: CohortSummary, new_id: int) -> CohortSummary:
    return CohortSummary(
        cohort_id=new_id,
        member_count=cohort.member_count,
        cascade_count=cohort.cascade_count,
        median_time_tightness_sec=cohort.median_time_tightness_sec,
        target_concentration_hhi=cohort.target_concentration_hhi,
        top_target_author=cohort.top_target_author,
        top_target_count=cohort.top_target_count,
        unique_targets=cohort.unique_targets,
        component_ids=cohort.component_ids,
        members=cohort.members,
        top_cascade_tweet_id=cohort.top_cascade_tweet_id,
        top_cascade_member_share=cohort.top_cascade_member_share,
        cohort_score=cohort.cohort_score,
    )


def _resolve_communities(
    edge_weights: dict[tuple[str, str], int],
    *,
    tight_only_weights: dict[tuple[str, str], int],
    random_state: int,
    max_community_size: int,
) -> list[set[str]]:
    """Run label propagation and recursively split oversized communities.

    Oversized communities are re-run on the tight-only subgraph (edges with
    `shared_within_15m >= 2` or strong within_60m overlap). If that still
    yields a giant community, we keep the partial split and let the size
    filter downstream accept it (cohort_score ranks tighter groups higher).
    """

    communities = _label_propagation(edge_weights, random_state=random_state)
    resolved: list[set[str]] = []
    for community in communities:
        if len(community) <= max_community_size:
            resolved.append(community)
            continue
        sub_edges = {
            pair: weight
            for pair, weight in tight_only_weights.items()
            if pair[0] in community and pair[1] in community
        }
        if not sub_edges:
            resolved.append(community)
            continue
        sub_communities = _label_propagation(sub_edges, random_state=random_state)
        covered: set[str] = set()
        for sub in sub_communities:
            if len(sub) >= 2:
                resolved.append(sub)
                covered.update(sub)
        remainder = community - covered
        if remainder:
            resolved.append(remainder)
    return resolved


def _build_tight_edge_weights(
    overlap_summaries: list[OverlapSummary],
    *,
    min_shared_cascades: int,
    min_tight_overlap: int,
) -> dict[tuple[str, str], int]:
    """Build weighted edges from time-tight overlap pairs only."""

    edge_weights: dict[tuple[str, str], int] = defaultdict(int)
    for overlap in overlap_summaries:
        if overlap.shared_retweets < min_shared_cascades:
            continue
        tight_signal = overlap.shared_within_60m + overlap.shared_within_15m
        if tight_signal < min_tight_overlap:
            continue
        weight = (
            overlap.shared_within_15m * 4
            + overlap.shared_within_60m * 2
            + overlap.shared_retweets
        )
        if weight <= 0:
            continue
        pair = _sorted_pair(overlap.account_a, overlap.account_b)
        edge_weights[pair] += weight
    return edge_weights


def _sorted_pair(left: str, right: str) -> tuple[str, str]:
    if left <= right:
        return (left, right)
    return (right, left)


def _label_propagation(
    edge_weights: dict[tuple[str, str], int],
    *,
    random_state: int,
    max_iterations: int = 40,
) -> list[set[str]]:
    """Asynchronous label propagation on a weighted graph.

    Uses per-iteration random shuffling, weighted neighbour voting, and a
    deterministic tie-break that prefers the node's own current label before
    falling back to the lexicographically smallest candidate. Preserving
    the own-label on ties stabilises tight communities on dense graphs
    where a pure min-tie-break tends to collapse everything into one label.
    """

    adjacency: dict[str, dict[str, int]] = defaultdict(dict)
    for (left, right), weight in edge_weights.items():
        if weight <= 0:
            continue
        adjacency[left][right] = weight
        adjacency[right][left] = weight

    rng = random.Random(random_state)
    labels = {node: node for node in adjacency}
    for _ in range(max_iterations):
        nodes = list(adjacency)
        rng.shuffle(nodes)
        changed = False
        for node in nodes:
            neighbors = adjacency[node]
            if not neighbors:
                continue
            counts: Counter[str] = Counter()
            for neighbor, weight in neighbors.items():
                counts[labels[neighbor]] += weight
            if not counts:
                continue
            top_count = max(counts.values())
            candidates = [label for label, count in counts.items() if count == top_count]
            chosen = labels[node] if labels[node] in candidates else min(candidates)
            if labels[node] != chosen:
                labels[node] = chosen
                changed = True
        if not changed:
            break

    groups: dict[str, set[str]] = defaultdict(set)
    for node, label in labels.items():
        groups[label].add(node)
    return list(groups.values())


def build_k_core_subgraph(
    overlap_summaries: list[OverlapSummary],
    *,
    k: int = 3,
    weight_key: str = "shared_within_15m",
) -> dict[str, int]:
    """Return k-core rank for each account using the chosen overlap weight.

    Accounts are repeatedly pruned if their weighted degree falls below `k`.
    The returned mapping contains the maximum k for which each account
    remains present.
    """

    adjacency: dict[str, dict[str, int]] = defaultdict(dict)
    for overlap in overlap_summaries:
        weight = getattr(overlap, weight_key, 0)
        if weight <= 0:
            continue
        adjacency[overlap.account_a][overlap.account_b] = weight
        adjacency[overlap.account_b][overlap.account_a] = weight

    core_rank: dict[str, int] = {}
    current_k = 1
    while adjacency:
        pruned = True
        while pruned:
            pruned = False
            for node in list(adjacency):
                degree = sum(adjacency[node].values())
                if degree < current_k:
                    for neighbor in list(adjacency[node]):
                        adjacency[neighbor].pop(node, None)
                        if not adjacency[neighbor]:
                            adjacency.pop(neighbor, None)
                    adjacency.pop(node, None)
                    core_rank[node] = max(core_rank.get(node, 0), current_k - 1)
                    pruned = True
        remaining = list(adjacency)
        if not remaining:
            break
        for node in remaining:
            core_rank[node] = max(core_rank.get(node, 0), current_k)
        current_k += 1
        if current_k > k + 1:
            break
    return core_rank
