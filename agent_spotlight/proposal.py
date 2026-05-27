"""Select a single MI cluster for the current spotlight pass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np

from learn_agents.learn_agents import (
    MiPartitionResult,
    discretize_trace_columns,
    factorize_background,
    mi_partition_search,
    precursor_cluster_stats,
)

from .config import SpotlightConfig


@dataclass
class ClusterCandidate:
    cluster_id: int
    var_indices: List[int]
    score: float
    precursor_passed: bool
    persistence: float
    contingency: float
    richness: float
    partition: MiPartitionResult


def apply_peel_mask(trace: np.ndarray, peeled: Set[int], mode: str) -> np.ndarray:
    out = trace.copy()
    if mode == "mask_zero" and peeled:
        for j in peeled:
            out[:, j] = 0.0
    return out


def _within_cluster_mi(
    trace: np.ndarray,
    var_indices: List[int],
    *,
    bins: int,
    max_lag: int,
) -> float:
    if len(var_indices) < 2:
        return 0.0
    from agency_detect.detection import build_similarity_matrix

    disc = discretize_trace_columns(trace, bins=bins)[:, var_indices]
    sim_m, _ = build_similarity_matrix(disc.astype(np.float64), max_lag=max_lag)
    n = len(var_indices)
    return float((sim_m.sum() - np.trace(sim_m)) / (n * (n - 1)))


def _score_single_cluster(
    trace: np.ndarray,
    var_indices: List[int],
    cfg: SpotlightConfig,
) -> Tuple[float, bool, float, float, float]:
    labels = np.full(trace.shape[1], -1, dtype=np.int64)
    labels[var_indices] = 0
    stats = precursor_cluster_stats(
        trace,
        labels,
        bins=cfg.mi_bins,
        max_lag=cfg.mi_max_lag,
        persistence_floor=cfg.precursor_persistence_floor,
        contingency_floor=cfg.precursor_contingency_floor,
    )
    if not stats:
        return 0.0, False, 0.0, 0.0, 0.0
    s = stats[0]
    signal = max(float(s.persistence), 0.0) + float(s.contingency)
    within = _within_cluster_mi(
        trace,
        var_indices,
        bins=cfg.mi_bins,
        max_lag=cfg.mi_max_lag,
    )
    score = signal + cfg.within_mi_weight * within
    if cfg.cluster_score == "precursor_x_size":
        score *= np.sqrt(len(var_indices))
    if len(var_indices) <= 2:
        score *= cfg.tiny_cluster_penalty
    return score, bool(s.passed), float(s.persistence), float(s.contingency), float(s.richness)


def rank_cluster_candidates(
    trace: np.ndarray,
    cfg: SpotlightConfig,
    peeled: Set[int],
) -> Tuple[List[ClusterCandidate], MiPartitionResult]:
    """All MI cluster candidates sorted by score (best first)."""
    work = apply_peel_mask(trace, peeled, cfg.peel_mode)
    mi_trace = work
    if cfg.proposal_background_factorize:
        mi_trace, _ = factorize_background(work, n_components=cfg.proposal_background_components)

    part = mi_partition_search(
        mi_trace,
        fixed_k=cfg.proposal_mi_k,
        bins=cfg.mi_bins,
        max_lag=cfg.mi_max_lag,
        k_selection="mdl",
    )

    candidates: List[ClusterCandidate] = []
    for cluster_id in np.unique(part.labels[part.labels >= 0]):
        idxs = np.where(part.labels == cluster_id)[0].tolist()
        idxs = [i for i in idxs if i not in peeled]
        if len(idxs) < cfg.min_cluster_size:
            continue
        score, passed, pers, cont, rich = _score_single_cluster(work, idxs, cfg)
        candidates.append(
            ClusterCandidate(
                cluster_id=int(cluster_id),
                var_indices=idxs,
                score=score,
                precursor_passed=passed,
                persistence=pers,
                contingency=cont,
                richness=rich,
                partition=part,
            )
        )
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates, part


def propose_best_cluster(
    trace: np.ndarray,
    cfg: SpotlightConfig,
    peeled: Set[int],
) -> Tuple[Optional[ClusterCandidate], MiPartitionResult]:
    candidates, part = rank_cluster_candidates(trace, cfg, peeled)
    if not candidates:
        return None, part
    best = candidates[0]
    return best, part
