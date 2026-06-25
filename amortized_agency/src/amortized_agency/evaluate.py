from __future__ import annotations

from typing import Dict, List, Literal, Sequence

import numpy as np
import torch

from amortized_agency.cluster import labels_from_affinity, mi_affinity_labels
from amortized_agency.kinds import Kind
from amortized_agency.metrics import score_clustering
from amortized_agency.context_model import ContextualAffinityModel
from amortized_agency.siamese import SiameseAffinityModel
from amortized_agency.slot_model import SlotAttentionAffinity
from amortized_agency.benchmark import EVAL_T_STEPS
from amortized_agency.worlds import Episode, simulate_episode

MethodName = Literal["mi", "siamese", "slot", "context"]


def evaluate_episode(
    episode: Episode,
    num_clusters: int,
    method: MethodName,
    *,
    siamese: SiameseAffinityModel | None = None,
    slot: SlotAttentionAffinity | None = None,
    context: ContextualAffinityModel | None = None,
    device: torch.device | None = None,
) -> Dict[str, float]:
    trace = episode.window
    true_ids = episode.agent_ids
    if method == "mi":
        labels = mi_affinity_labels(trace, num_clusters)
    elif method == "siamese":
        if siamese is None or device is None:
            raise ValueError("siamese model and device required")
        siamese.eval()
        aff = siamese.affinity_matrix(trace, device)
        labels = labels_from_affinity(aff, num_clusters)
    elif method == "slot":
        if slot is None or device is None:
            raise ValueError("slot model and device required")
        slot.eval()
        aff = slot.affinity_matrix(trace, device)
        labels = labels_from_affinity(aff, num_clusters)
    elif method == "context":
        if context is None or device is None:
            raise ValueError("context model and device required")
        context.eval()
        aff = context.affinity_matrix(trace, device)
        labels = labels_from_affinity(aff, num_clusters)
    else:
        raise ValueError(f"unknown method {method}")
    return score_clustering(labels, true_ids)


def evaluate_kind(
    kind: Kind,
    windows: Sequence[int],
    seeds: Sequence[int],
    methods: Sequence[MethodName],
    *,
    siamese: SiameseAffinityModel | None = None,
    slot: SlotAttentionAffinity | None = None,
    context: ContextualAffinityModel | None = None,
    device: torch.device | None = None,
    t_steps: int | None = None,
) -> List[Dict]:
    rows: List[Dict] = []
    t_max = max(max(windows), EVAL_T_STEPS) if t_steps is None else t_steps
    for window in windows:
        for method in methods:
            aris, jaccs = [], []
            for seed in seeds:
                ep = simulate_episode(kind, window, seed, t_steps=t_max)
                m = evaluate_episode(
                    ep,
                    kind.num_agents,
                    method,
                    siamese=siamese,
                    slot=slot,
                    context=context,
                    device=device,
                )
                aris.append(m["ari"])
                jaccs.append(m["mean_jaccard"])
            rows.append(
                {
                    "kind": kind.name,
                    "window": window,
                    "method": method,
                    "ari_mean": float(np.mean(aris)),
                    "ari_std": float(np.std(aris)),
                    "jaccard_mean": float(np.mean(jaccs)),
                    "jaccard_std": float(np.std(jaccs)),
                    "n_seeds": len(seeds),
                }
            )
    return rows
