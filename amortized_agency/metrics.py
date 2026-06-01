from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score


def best_mean_jaccard(labels: np.ndarray, true_ids: np.ndarray) -> float:
    pred_clusters = [np.where(labels == c)[0] for c in sorted(set(labels.tolist())) if c >= 0]
    true_clusters = [np.where(true_ids == a)[0] for a in sorted(set(true_ids.tolist()))]
    if not pred_clusters or not true_clusters:
        return 0.0
    scores = []
    for pc in pred_clusters:
        pcs = set(pc.tolist())
        best = 0.0
        for tc in true_clusters:
            tcs = set(tc.tolist())
            inter = len(pcs & tcs)
            union = len(pcs | tcs)
            if union:
                best = max(best, inter / union)
        scores.append(best)
    return float(np.mean(scores))


def score_clustering(labels: np.ndarray, true_ids: np.ndarray) -> dict[str, float]:
    active = labels >= 0
    if active.sum() < 2:
        return {"ari": 0.0, "mean_jaccard": 0.0}
    ari = float(adjusted_rand_score(true_ids[active], labels[active]))
    jacc = best_mean_jaccard(labels[active], true_ids[active])
    return {"ari": ari, "mean_jaccard": jacc}
