from __future__ import annotations

from collections import Counter
from typing import Any

from zerosixty.models import (
    FeatureRow,
    MLAccountSummary,
    MLClusterSummary,
    MLRunSummary,
)

MODEL_FEATURE_NAMES = (
    "tweet_count",
    "retweet_count",
    "original_count",
    "quote_count",
    "retweet_ratio",
    "first_retweeter_count",
    "unique_retweeted_tweet_count",
    "unique_retweeted_author_count",
    "retweets_to_member_count",
    "retweets_to_member_ratio",
    "top_amplified_count",
    "top_amplified_share",
    "hashtag_count",
    "mention_count",
    "followers_count",
    "friends_count",
    "statuses_count",
    "account_age_days",
    "default_profile",
    "default_profile_image",
    "is_blue_verified",
    "network_component_size",
    "network_neighbor_count",
    "network_weighted_degree",
    "network_within_15m_weight",
    "network_within_60m_weight",
    "network_max_shared_edge",
)

LOG_SCALE_FEATURES = {
    "tweet_count",
    "retweet_count",
    "original_count",
    "quote_count",
    "first_retweeter_count",
    "unique_retweeted_tweet_count",
    "unique_retweeted_author_count",
    "retweets_to_member_count",
    "top_amplified_count",
    "hashtag_count",
    "mention_count",
    "followers_count",
    "friends_count",
    "statuses_count",
    "account_age_days",
    "network_component_size",
    "network_neighbor_count",
    "network_weighted_degree",
    "network_within_15m_weight",
    "network_within_60m_weight",
    "network_max_shared_edge",
}


def run_ml_pipeline(
    feature_rows: list[FeatureRow],
    *,
    requested_clusters: int | None = None,
    random_state: int = 42,
) -> tuple[MLRunSummary, list[MLAccountSummary], list[MLClusterSummary]]:
    """Run an unsupervised ML baseline alongside the deterministic pipeline."""

    if not feature_rows:
        return (
            MLRunSummary(
                status="skipped_no_samples",
                sample_count=0,
                input_feature_count=len(MODEL_FEATURE_NAMES),
                cluster_count=0,
                cluster_selection="none",
                cluster_model="none",
                anomaly_model="none",
                embedding_model="none",
                feature_names=MODEL_FEATURE_NAMES,
                note="No feature rows were available for ML processing.",
            ),
            [],
            [],
        )

    try:
        import numpy as np
        from sklearn.cluster import KMeans  # type: ignore[import-untyped]
        from sklearn.decomposition import PCA  # type: ignore[import-untyped]
        from sklearn.ensemble import IsolationForest  # type: ignore[import-untyped]
        from sklearn.metrics import silhouette_score  # type: ignore[import-untyped]
        from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
    except ImportError as exc:
        return (
            MLRunSummary(
                status="skipped_missing_dependency",
                sample_count=len(feature_rows),
                input_feature_count=len(MODEL_FEATURE_NAMES),
                cluster_count=0,
                cluster_selection="none",
                cluster_model="none",
                anomaly_model="none",
                embedding_model="none",
                feature_names=MODEL_FEATURE_NAMES,
                note=str(exc),
            ),
            [],
            [],
        )

    matrix = np.array([_feature_vector(row) for row in feature_rows], dtype=float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    cluster_count, cluster_selection = _select_cluster_count(
        scaled,
        requested_clusters=requested_clusters,
        random_state=random_state,
        kmeans_cls=KMeans,
        silhouette_score_fn=silhouette_score,
    )

    if cluster_count <= 1:
        raw_labels = np.zeros(len(feature_rows), dtype=int)
        centers = scaled.mean(axis=0, keepdims=True)
    else:
        kmeans = KMeans(n_clusters=cluster_count, n_init=20, random_state=random_state)
        raw_labels = kmeans.fit_predict(scaled)
        centers = kmeans.cluster_centers_

    remapped_labels = _remap_labels(raw_labels)
    cluster_sizes = Counter(remapped_labels)
    centroid_distances = [
        float(np.linalg.norm(scaled[index] - centers[label]))
        for index, label in enumerate(raw_labels)
    ]

    if len(feature_rows) >= 4:
        anomaly_model = IsolationForest(random_state=random_state, contamination="auto")
        anomaly_model.fit(scaled)
        anomaly_scores = [-float(score) for score in anomaly_model.score_samples(scaled)]
        anomaly_model_name = "IsolationForest"
    else:
        anomaly_scores = [0.0 for _ in feature_rows]
        anomaly_model_name = "fallback_constant"

    coordinates = _build_embedding(scaled, pca_cls=PCA, random_state=random_state)
    anomaly_ranks = _rank_descending(anomaly_scores)

    account_summaries = [
        MLAccountSummary(
            account_handle=row.account_handle,
            cluster_id=remapped_labels[index] + 1,
            cluster_size=cluster_sizes[remapped_labels[index]],
            anomaly_score=round(anomaly_scores[index], 4),
            anomaly_rank=anomaly_ranks[index],
            centroid_distance=round(centroid_distances[index], 4),
            embedding_x=round(coordinates[index][0], 4),
            embedding_y=round(coordinates[index][1], 4),
            coordination_score=row.coordination_score,
            retweet_ratio=row.retweet_ratio,
            network_weighted_degree=row.network_weighted_degree,
        )
        for index, row in enumerate(feature_rows)
    ]
    account_summaries.sort(
        key=lambda item: (
            item.anomaly_rank,
            item.cluster_id,
            -item.coordination_score,
            item.account_handle,
        )
    )

    cluster_summaries = _build_cluster_summaries(
        feature_rows,
        remapped_labels,
        cluster_count=cluster_count,
    )

    run_summary = MLRunSummary(
        status="ready",
        sample_count=len(feature_rows),
        input_feature_count=len(MODEL_FEATURE_NAMES),
        cluster_count=max(cluster_count, 1),
        cluster_selection=cluster_selection,
        cluster_model="KMeans" if cluster_count > 1 else "single_cluster_fallback",
        anomaly_model=anomaly_model_name,
        embedding_model="PCA",
        feature_names=MODEL_FEATURE_NAMES,
        note=None,
    )
    return run_summary, account_summaries, cluster_summaries


def _feature_vector(row: FeatureRow) -> list[float]:
    values: list[float] = []
    for feature_name in MODEL_FEATURE_NAMES:
        raw_value = getattr(row, feature_name)
        value = 0.0 if raw_value is None else float(raw_value)
        if feature_name in LOG_SCALE_FEATURES:
            value = _log1p(value)
        values.append(value)
    return values


def _log1p(value: float) -> float:
    import math

    return math.log1p(max(value, 0.0))


def _select_cluster_count(
    scaled: Any,
    *,
    requested_clusters: int | None,
    random_state: int,
    kmeans_cls: Any,
    silhouette_score_fn: Any,
) -> tuple[int, str]:
    sample_count = int(scaled.shape[0])
    if sample_count < 4:
        return 1, "fallback_small_sample"

    if requested_clusters is not None:
        return max(1, min(requested_clusters, sample_count)), "requested"

    max_clusters = min(8, sample_count - 1)
    best_cluster_count = 1
    best_score = float("-inf")
    for cluster_count in range(2, max_clusters + 1):
        model = kmeans_cls(n_clusters=cluster_count, n_init=20, random_state=random_state)
        labels = model.fit_predict(scaled)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score_fn(scaled, labels))
        if score > best_score:
            best_score = score
            best_cluster_count = cluster_count

    if best_cluster_count == 1:
        return 1, "fallback_single_cluster"
    return best_cluster_count, "silhouette"


def _remap_labels(raw_labels: Any) -> list[int]:
    counts = Counter(int(label) for label in raw_labels)
    ordered = sorted(counts, key=lambda label: (-counts[label], label))
    label_map = {label: index for index, label in enumerate(ordered)}
    return [label_map[int(label)] for label in raw_labels]


def _build_embedding(
    scaled: Any,
    *,
    pca_cls: Any,
    random_state: int,
) -> list[tuple[float, float]]:
    import numpy as np

    sample_count = int(scaled.shape[0])
    feature_count = int(scaled.shape[1])
    if sample_count == 1:
        return [(0.0, 0.0)]

    component_count = 2 if min(sample_count, feature_count) >= 2 else 1
    pca = pca_cls(n_components=component_count, random_state=random_state)
    projected = pca.fit_transform(scaled)
    if component_count == 1:
        projected = np.column_stack([projected[:, 0], np.zeros(sample_count)])
    return [(float(row[0]), float(row[1])) for row in projected]


def _rank_descending(values: list[float]) -> list[int]:
    ranks = [0 for _ in values]
    ordered = sorted(range(len(values)), key=lambda index: (-values[index], index))
    for rank, index in enumerate(ordered, start=1):
        ranks[index] = rank
    return ranks


def _build_cluster_summaries(
    feature_rows: list[FeatureRow],
    remapped_labels: list[int],
    *,
    cluster_count: int,
) -> list[MLClusterSummary]:
    cluster_indices: dict[int, list[int]] = {
        cluster_id: [] for cluster_id in range(max(cluster_count, 1))
    }
    for index, cluster_id in enumerate(remapped_labels):
        cluster_indices[cluster_id].append(index)

    summaries: list[MLClusterSummary] = []
    for cluster_id, indices in sorted(cluster_indices.items()):
        members = [feature_rows[index] for index in indices]
        account_count = len(members)
        top_accounts = tuple(
            item.account_handle
            for item in sorted(
                members,
                key=lambda row: (
                    -row.coordination_score,
                    -row.network_weighted_degree,
                    row.account_handle,
                ),
            )[:10]
        )
        summaries.append(
            MLClusterSummary(
                cluster_id=cluster_id + 1,
                account_count=account_count,
                mean_coordination_score=round(
                    sum(item.coordination_score for item in members) / account_count,
                    4,
                ),
                mean_retweet_ratio=round(
                    sum(item.retweet_ratio for item in members) / account_count,
                    4,
                ),
                mean_network_weighted_degree=round(
                    sum(item.network_weighted_degree for item in members) / account_count,
                    4,
                ),
                top_accounts=top_accounts,
            )
        )
    summaries.sort(key=lambda item: (-item.account_count, item.cluster_id))
    return summaries
