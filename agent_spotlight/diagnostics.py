"""Per-agent miss diagnosis for serial spotlight runs."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from .config import SpotlightConfig
from .metrics import best_agent_match, jaccard
from .validation import agency_signature_for_indices


def _miss_reason(
    *,
    admitted: bool,
    best_proposal_j: float,
    best_trained_j: float,
    hit_threshold: float,
    best_agency_passed: Optional[bool],
    partially_peeled: bool,
    agency_gate_blocked: bool,
) -> str:
    if admitted:
        return "admitted"
    if agency_gate_blocked and best_agency_passed is False:
        return "agency_gate_blocked"
    if partially_peeled:
        return "partial_peel_orphan"
    if best_proposal_j >= hit_threshold and best_trained_j < hit_threshold:
        return "refine_or_train_degraded"
    if 0 < best_proposal_j < hit_threshold:
        return "mi_merge_below_threshold"
    if best_proposal_j <= 0:
        return "never_in_mi_cluster"
    return "unknown"


def compute_agent_diagnostics(
    *,
    agent_clusters: Mapping[int, Sequence[int]],
    admitted_agents: Set[int],
    peeled: Set[int],
    passes: Sequence[Any],
    pass_candidate_logs: Sequence[List[Dict[str, Any]]],
    var_names: Sequence[str],
    trace,
    cfg: SpotlightConfig,
) -> Dict[str, Any]:
    """Summarize why each agent was or was not admitted."""
    agent_ids = sorted(int(k) for k in agent_clusters.keys())
    rows: Dict[int, Dict[str, Any]] = {}

    for agent_id in agent_ids:
        truth = list(agent_clusters[agent_id])
        peeled_overlap = len(set(truth) & peeled) / len(truth) if truth else 0.0

        best_proposal_j = 0.0
        best_proposal_pass = -1
        best_proposal_rank = -1
        best_proposal_agency: Optional[bool] = None
        agency_gate_blocked = False

        best_trained_j = 0.0
        best_trained_pass = -1

        for pass_idx, cand_log in enumerate(pass_candidate_logs):
            for rank, entry in enumerate(cand_log):
                _, jj = best_agent_match(entry["var_indices"], {agent_id: truth})
                if jj > best_proposal_j:
                    best_proposal_j = jj
                    best_proposal_pass = pass_idx
                    best_proposal_rank = rank
                    best_proposal_agency = entry.get("agency_passed")

            if pass_idx < len(passes):
                p = passes[pass_idx]
                if getattr(p, "best_agent_id", -1) == agent_id:
                    best_trained_j = max(best_trained_j, float(getattr(p, "best_jaccard", 0.0)))
                    best_trained_pass = pass_idx
                extra = getattr(p, "extra", {}) or {}
                for skip in extra.get("skipped_agency", []):
                    _, jj = best_agent_match(skip.get("var_indices", []), {agent_id: truth})
                    if jj >= cfg.jaccard_hit_threshold:
                        agency_gate_blocked = True

        admitted = agent_id in admitted_agents
        rows[agent_id] = {
            "admitted": admitted,
            "cluster_size": len(truth),
            "peeled_fraction": float(peeled_overlap),
            "best_proposal_jaccard": float(best_proposal_j),
            "best_proposal_pass": best_proposal_pass,
            "best_proposal_rank": best_proposal_rank,
            "best_proposal_agency_passed": best_proposal_agency,
            "best_trained_jaccard": float(best_trained_j),
            "best_trained_pass": best_trained_pass,
            "miss_reason": _miss_reason(
                admitted=admitted,
                best_proposal_j=best_proposal_j,
                best_trained_j=best_trained_j,
                hit_threshold=cfg.jaccard_hit_threshold,
                best_agency_passed=best_proposal_agency,
                partially_peeled=peeled_overlap > 0 and peeled_overlap < 1.0 and not admitted,
                agency_gate_blocked=agency_gate_blocked,
            ),
        }

    missed = [a for a in agent_ids if a not in admitted_agents]
    return {
        "by_agent": {str(k): v for k, v in rows.items()},
        "missed_agent_ids": missed,
        "n_missed": len(missed),
    }


def mi_partition_agent_overlap(
    *,
    trace,
    cfg: SpotlightConfig,
    peeled: Set[int],
    agent_clusters: Mapping[int, Sequence[int]],
    var_names: Sequence[str],
) -> Dict[str, Any]:
    """MI-only view: best cluster per agent on the current peeled trace."""
    from .proposal import rank_cluster_candidates

    candidates, part = rank_cluster_candidates(trace, cfg, peeled, var_names=var_names)
    per_agent: Dict[int, Dict[str, Any]] = {}
    for agent_id, truth in agent_clusters.items():
        best_j = 0.0
        best_cluster = None
        for cand in candidates:
            _, jj = best_agent_match(cand.var_indices, {int(agent_id): list(truth)})
            if jj > best_j:
                best_j = jj
                sig = agency_signature_for_indices(cand.var_indices, var_names, trace, cfg)
                best_cluster = {
                    "cluster_id": cand.cluster_id,
                    "size": len(cand.var_indices),
                    "score": cand.score,
                    "jaccard": jj,
                    "agency_passed": sig["passed"],
                    "n_sensors": sig["n_sensors"],
                    "n_actions": sig["n_actions"],
                    "n_internals": sig["n_internals"],
                }
        per_agent[int(agent_id)] = {"best_cluster": best_cluster, "best_jaccard": best_j}

    return {
        "n_candidates": len(candidates),
        "proposal_mi_k": cfg.proposal_mi_k,
        "by_agent": {str(k): v for k, v in per_agent.items()},
    }
