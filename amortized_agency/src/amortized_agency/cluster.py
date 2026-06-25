from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from learn_agents.learn_agents import mi_cluster_variable_labels


def labels_from_affinity(affinity: np.ndarray, num_clusters: int) -> np.ndarray:
    """Agglomerative clustering on a symmetric affinity matrix."""
    n = affinity.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.int64)
    aff = np.clip(affinity.astype(np.float64), 0.0, None)
    np.fill_diagonal(aff, aff.diagonal().max() if aff.diagonal().max() > 0 else 1.0)
    dist = 1.0 - aff / (aff.max() + 1e-12)
    n_clust = min(num_clusters, n)
    return AgglomerativeClustering(
        n_clusters=n_clust, metric="precomputed", linkage="complete"
    ).fit_predict(dist)


def mi_affinity_labels(trace: np.ndarray, num_clusters: int) -> np.ndarray:
    return mi_cluster_variable_labels(trace, num_clusters=num_clusters)
