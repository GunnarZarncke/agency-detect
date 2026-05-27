"""Agency gate cluster selection for spotlight passes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import SpotlightConfig
from .metrics import best_agent_match
from .proposal import ClusterCandidate
from .validation import agency_gate_passes, agency_signature_for_indices


def build_candidate_log(
    candidates: Sequence[ClusterCandidate],
    agent_clusters: Dict[int, List[int]],
    var_names: Sequence[str],
    trace: np.ndarray,
    cfg: SpotlightConfig,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rank, cand in enumerate(candidates):
        sig = agency_signature_for_indices(cand.var_indices, var_names, trace, cfg)
        best_agent, best_j = best_agent_match(cand.var_indices, agent_clusters)
        rows.append(
            {
                "rank": rank,
                "cluster_id": cand.cluster_id,
                "score": cand.score,
                "size": len(cand.var_indices),
                "var_indices": cand.var_indices,
                "precursor_passed": cand.precursor_passed,
                "best_agent_id": best_agent,
                "best_jaccard": best_j,
                "agency_passed": sig["passed"],
                "n_sensors": sig["n_sensors"],
                "n_actions": sig["n_actions"],
                "n_internals": sig["n_internals"],
            }
        )
    return rows


def select_cluster_for_pass(
    candidates: Sequence[ClusterCandidate],
    var_names: Sequence[str],
    trace: np.ndarray,
    cfg: SpotlightConfig,
) -> Tuple[Optional[ClusterCandidate], List[Dict[str, Any]], Optional[Dict[str, Any]], bool]:
    """
    Pick a cluster given agency gate mode.

    Returns: cluster, skipped_agency entries, agency_sig for selected, peeled_agency_skip flag.
    """
    mode = cfg.effective_agency_gate_mode()
    if not candidates:
        return None, [], None, False

    if mode in ("off", "score_penalty"):
        sig = agency_signature_for_indices(candidates[0].var_indices, var_names, trace, cfg)
        return candidates[0], [], sig, False

    skipped: List[Dict[str, Any]] = []
    peeled_flag = False
    for cand in candidates:
        sig = agency_signature_for_indices(cand.var_indices, var_names, trace, cfg)
        passed = agency_gate_passes(sig, mode)
        if not passed:
            entry = {
                "cluster_id": cand.cluster_id,
                "score": cand.score,
                "size": len(cand.var_indices),
                "var_indices": cand.var_indices,
                "n_sensors": sig["n_sensors"],
                "n_actions": sig["n_actions"],
                "n_internals": sig["n_internals"],
            }
            skipped.append(entry)
            if (
                cfg.peel_on_agency_skip
                and mode == "strict"
                and sig["n_actions"] == 0
                and len(cand.var_indices) >= 6
            ):
                peeled_flag = True
            continue
        return cand, skipped, sig, peeled_flag

    if mode == "soft":
        sig = agency_signature_for_indices(candidates[0].var_indices, var_names, trace, cfg)
        return candidates[0], skipped, sig, peeled_flag

    return None, skipped, None, peeled_flag
