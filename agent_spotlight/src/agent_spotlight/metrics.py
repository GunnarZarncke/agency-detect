"""Ground-truth and peel-pass metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def best_agent_match(
    var_indices: Sequence[int],
    agent_clusters: Dict[int, List[int]],
) -> tuple[int, float]:
    best_agent, best_j = -1, 0.0
    for agent_id, truth in agent_clusters.items():
        jj = jaccard(var_indices, truth)
        if jj > best_j:
            best_j, best_agent = jj, int(agent_id)
    return best_agent, float(best_j)


def cumulative_agent_recall(
    admitted_agent_ids: Set[int],
    agent_ids: Sequence[int],
) -> float:
    if not agent_ids:
        return 0.0
    return len(admitted_agent_ids & set(agent_ids)) / len(agent_ids)


@dataclass
class PassMetrics:
    pass_index: int
    proposal_mi_k: int
    n_clusters_scored: int
    selected_cluster_id: int
    cluster_size: int
    cluster_var_indices: List[int]
    cluster_score: float
    precursor_passed: bool
    precursor_persistence: float
    precursor_contingency: float
    precursor_richness: float
    pretrain_final_loss: float
    refine_final_align: float
    candidate_var_count: int
    candidate_mode: str
    uad_valid: Optional[bool] = None
    uad_violation: Optional[float] = None
    best_agent_id: int = -1
    best_jaccard: float = 0.0
    is_hit: bool = False
    admitted: bool = False
    peeled_var_count: int = 0
    cumulative_recall: float = 0.0
    stop_reason: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
