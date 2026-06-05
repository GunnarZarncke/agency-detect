"""Evaluate outcome-influence scores per agent and simulation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from learn_agents.learn_agents import SimulationResult

from intention_detect.defense import defense_odds_ratio
from intention_detect.influence import _series, influence_from_trace
from intention_detect.outcomes import CriticalOutcome, control_indices, parse_critical_outcomes


@dataclass(frozen=True)
class OutcomeInfluenceScore:
    agent: int
    outcome_name: str
    outcome_index: int
    influence: float
    defense_or: float
    defense_or_low: float
    selectivity: float
    combined: float
    flagged: bool


def _action_composite(trace: np.ndarray, action_indices: Sequence[int]) -> np.ndarray:
    if not action_indices:
        return np.zeros(trace.shape[0], dtype=np.float64)
    sub = trace[:, list(action_indices)]
    if sub.ndim == 1:
        return np.abs(sub.astype(np.float64))
    return np.mean(np.abs(sub), axis=1).astype(np.float64)


def score_agent_outcome(
    trace: np.ndarray,
    metadata: Mapping[str, object],
    agent: int,
    outcome: CriticalOutcome,
    *,
    or_threshold: float = 1.40,
    or_low_min: float = 1.08,
    selectivity_min: float = 1.05,
    influence_min: float = 0.25,
    bad_quantile: float = 0.80,
    seed: int = 0,
) -> OutcomeInfluenceScore:
    role_indices = metadata["role_indices"]
    action_idx = list(role_indices.get((agent, "action"), []))
    ctrl_idx = control_indices(metadata)
    # Do not control on the same outcome column.
    ctrl_idx = [i for i in ctrl_idx if i != outcome.index]

    infl = influence_from_trace(
        trace,
        action_idx,
        outcome.index,
        ctrl_idx,
        direction=outcome.direction,
    )

    action = _action_composite(trace, action_idx)
    outcome_series = trace[:, outcome.index].astype(np.float64)
    if ctrl_idx:
        controls = trace[:, ctrl_idx].astype(np.float64)
    else:
        controls = np.zeros((trace.shape[0], 0), dtype=np.float64)

    or_point, or_low, selectivity = defense_odds_ratio(
        action,
        outcome_series,
        controls,
        direction=outcome.direction,
        bad_quantile=bad_quantile,
        seed=seed,
    )

    combined = max(or_point / or_threshold, abs(infl) / influence_min, selectivity / selectivity_min)
    defense_path = (
        (or_point >= or_threshold and or_low >= or_low_min and infl > 0.0)
        or (or_point >= 1.65 and or_low >= 1.15)
    )
    control_path = abs(infl) >= influence_min and selectivity >= min(selectivity_min, 0.78)
    flagged = bool(defense_path or control_path)

    return OutcomeInfluenceScore(
        agent=int(agent),
        outcome_name=outcome.name,
        outcome_index=outcome.index,
        influence=float(infl),
        defense_or=float(or_point),
        defense_or_low=float(or_low),
        selectivity=float(selectivity),
        combined=float(combined),
        flagged=bool(flagged),
    )


def score_simulation(
    result: SimulationResult,
    *,
    or_threshold: float = 1.40,
    or_low_min: float = 1.08,
    selectivity_min: float = 1.05,
    influence_min: float = 0.25,
    bad_quantile: float = 0.80,
    min_T: int = 80,
    seed: int = 0,
) -> Dict[str, object]:
    trace = result.trace
    meta = result.metadata
    if trace.shape[0] < min_T:
        return {"T": int(trace.shape[0]), "skipped": "trace_too_short", "agents": {}}

    outcomes = parse_critical_outcomes(meta)
    if not outcomes:
        return {"T": int(trace.shape[0]), "skipped": "no_critical_outcomes", "agents": {}}

    agent_clusters = meta.get("agent_clusters", {})
    agents = sorted(agent_clusters.keys()) if agent_clusters else sorted(
        {int(k[0]) for k in meta["role_indices"] if k[0] >= 0}
    )

    per_agent: Dict[int, Dict[str, object]] = {}
    flagged: List[int] = []
    for agent in agents:
        scores = [
            score_agent_outcome(
                trace,
                meta,
                int(agent),
                outcome,
                or_threshold=or_threshold,
                or_low_min=or_low_min,
                selectivity_min=selectivity_min,
                influence_min=influence_min,
                bad_quantile=bad_quantile,
                seed=seed + int(agent),
            )
            for outcome in outcomes
        ]
        if not scores:
            continue
        best = max(scores, key=lambda s: s.combined)
        if best.flagged:
            flagged.append(int(agent))
        per_agent[int(agent)] = {
            "max_combined": best.combined,
            "flagged": best.flagged,
            "best_outcome": best.outcome_name,
            "scores": [asdict(s) for s in scores],
        }

    gt = meta.get("outcome_influence_ground_truth") or {}

    return {
        "T": int(trace.shape[0]),
        "n_outcomes": len(outcomes),
        "flagged_agents": flagged,
        "agents": per_agent,
        "ground_truth": {str(k): bool(v) for k, v in gt.items()},
    }


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Trapezoid AUROC for binary labels."""
    if not scores or not labels:
        return float("nan")
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(1 for _, y in pairs if y)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = fp = 0
    prev_fpr = prev_tpr = 0.0
    auc = 0.0
    i = len(pairs) - 1
    while i >= 0:
        thr = pairs[i][0]
        while i >= 0 and pairs[i][0] == thr:
            if pairs[i][1]:
                tp += 1
            else:
                fp += 1
            i -= 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5
        prev_tpr, prev_fpr = tpr, fpr
    return float(auc)


def pooled_agent_rows(results: Sequence[Dict[str, object]]) -> Tuple[List[float], List[bool]]:
    scores: List[float] = []
    labels: List[bool] = []
    for row in results:
        gt = row.get("ground_truth") or {}
        agents = row.get("agents") or {}
        for aid, info in agents.items():
            scores.append(float(info.get("max_combined", 0.0)))
            labels.append(bool(gt.get(str(aid), False)))
    return scores, labels
