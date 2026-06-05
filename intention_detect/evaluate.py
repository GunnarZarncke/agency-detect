"""Evaluate outcome-influence scores per agent and simulation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from learn_agents.learn_agents import SimulationResult

from intention_detect.defense import defense_odds_ratio
from intention_detect.influence import _series, influence_from_trace
from intention_detect.outcomes import CriticalOutcome, control_indices, parse_critical_outcomes
from intention_detect.segmentation import calibrate_segment_params, segment_ranges, should_segment


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
    segment_start: int = 0
    segment_end: int = 0
    n_segments: int = 1


def _apply_flag(
    infl: float,
    or_point: float,
    or_low: float,
    selectivity: float,
    *,
    or_threshold: float,
    or_low_min: float,
    selectivity_min: float,
    influence_min: float,
    influence_strong_min: float,
) -> bool:
    defense_path = (
        (or_point >= or_threshold and or_low >= or_low_min and infl > 0.0)
        or (or_point >= 1.65 and or_low >= 1.15)
    )
    strong = abs(infl) >= influence_strong_min
    control_path = abs(infl) >= influence_min and (
        selectivity >= min(selectivity_min, 0.78) or strong
    )
    return bool(defense_path or control_path)


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
    influence_strong_min: float = 0.30,
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
    flagged = _apply_flag(
        infl,
        or_point,
        or_low,
        selectivity,
        or_threshold=or_threshold,
        or_low_min=or_low_min,
        selectivity_min=selectivity_min,
        influence_min=influence_min,
        influence_strong_min=influence_strong_min,
    )

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
        segment_start=0,
        segment_end=int(trace.shape[0]),
        n_segments=1,
    )


def score_agent_outcome_segmented(
    trace: np.ndarray,
    metadata: Mapping[str, object],
    agent: int,
    outcome: CriticalOutcome,
    *,
    or_threshold: float = 1.40,
    or_low_min: float = 1.08,
    selectivity_min: float = 1.05,
    influence_min: float = 0.25,
    influence_strong_min: float = 0.30,
    bad_quantile: float = 0.80,
    seed: int = 0,
) -> OutcomeInfluenceScore:
    """Score over auto-calibrated windows; flag if full trace or any segment flags."""
    role_indices = metadata["role_indices"]
    action_idx = list(role_indices.get((agent, "action"), []))
    action = _action_composite(trace, action_idx)
    T = trace.shape[0]
    params = calibrate_segment_params(T, action)
    ranges = segment_ranges(T, action, params)

    kw = dict(
        or_threshold=or_threshold,
        or_low_min=or_low_min,
        selectivity_min=selectivity_min,
        influence_min=influence_min,
        influence_strong_min=influence_strong_min,
        bad_quantile=bad_quantile,
    )
    full = score_agent_outcome(trace, metadata, agent, outcome, seed=seed, **kw)
    if len(ranges) <= 1:
        return full

    best = full
    any_flagged = full.flagged
    for seg_i, (start, end) in enumerate(ranges):
        if end - start < 40:
            continue
        seg = score_agent_outcome(
            trace[start:end], metadata, agent, outcome, seed=seed + seg_i, **kw
        )
        seg_boost = seg.flagged and abs(seg.influence) >= max(
            influence_min, abs(full.influence) * 1.1 + 0.02
        ) and (seg.combined >= full.combined or abs(seg.influence) >= influence_strong_min)
        any_flagged = any_flagged or seg_boost
        if seg.combined > best.combined or (seg.flagged and not best.flagged):
            best = OutcomeInfluenceScore(
                agent=seg.agent,
                outcome_name=seg.outcome_name,
                outcome_index=seg.outcome_index,
                influence=seg.influence,
                defense_or=seg.defense_or,
                defense_or_low=seg.defense_or_low,
                selectivity=seg.selectivity,
                combined=seg.combined,
                flagged=seg.flagged,
                segment_start=start,
                segment_end=end,
                n_segments=len(ranges),
            )

    return OutcomeInfluenceScore(
        agent=best.agent,
        outcome_name=best.outcome_name,
        outcome_index=best.outcome_index,
        influence=best.influence,
        defense_or=best.defense_or,
        defense_or_low=best.defense_or_low,
        selectivity=best.selectivity,
        combined=max(best.combined, full.combined),
        flagged=any_flagged,
        segment_start=best.segment_start,
        segment_end=best.segment_end,
        n_segments=len(ranges),
    )


def score_simulation(
    result: SimulationResult,
    *,
    or_threshold: float = 1.40,
    or_low_min: float = 1.08,
    selectivity_min: float = 1.05,
    influence_min: float = 0.25,
    influence_strong_min: float = 0.30,
    bad_quantile: float = 0.80,
    min_T: int = 80,
    segment_mode: str = "auto",
    segment_min_T: int = 250,
    seed: int = 0,
) -> Dict[str, object]:
    trace = result.trace
    meta = result.metadata
    if trace.shape[0] < min_T:
        return {"T": int(trace.shape[0]), "skipped": "trace_too_short", "agents": {}}

    outcomes = parse_critical_outcomes(meta)
    if not outcomes:
        return {"T": int(trace.shape[0]), "skipped": "no_critical_outcomes", "agents": {}}

    use_segments = segment_mode == "segmented" or (
        segment_mode == "auto" and should_segment(trace, meta, min_T=segment_min_T)
    )
    score_fn = score_agent_outcome_segmented if use_segments else score_agent_outcome

    agent_clusters = meta.get("agent_clusters", {})
    agents = sorted(agent_clusters.keys()) if agent_clusters else sorted(
        {int(k[0]) for k in meta["role_indices"] if k[0] >= 0}
    )

    per_agent: Dict[int, Dict[str, object]] = {}
    flagged: List[int] = []
    for agent in agents:
        scores = [
            score_fn(
                trace,
                meta,
                int(agent),
                outcome,
                or_threshold=or_threshold,
                or_low_min=or_low_min,
                selectivity_min=selectivity_min,
                influence_min=influence_min,
                influence_strong_min=influence_strong_min,
                bad_quantile=bad_quantile,
                seed=seed + int(agent),
            )
            for outcome in outcomes
        ]
        if not scores:
            continue
        flagged_scores = [s for s in scores if s.flagged]
        any_flagged = bool(flagged_scores)
        best = max(flagged_scores or scores, key=lambda s: s.combined)
        if any_flagged:
            flagged.append(int(agent))
        per_agent[int(agent)] = {
            "max_combined": max(s.combined for s in scores),
            "flagged": any_flagged,
            "best_outcome": best.outcome_name,
            "segment_mode": "segmented" if use_segments else "full",
            "scores": [asdict(s) for s in scores],
        }

    gt = meta.get("outcome_influence_ground_truth") or {}

    return {
        "T": int(trace.shape[0]),
        "n_outcomes": len(outcomes),
        "segment_mode": "segmented" if use_segments else "full",
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
