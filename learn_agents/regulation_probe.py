"""Option D: data-only homeostatic / setpoint regulation probe.

Detects disturbance rejection (good-regulator signature), not goal-reaching or
reactive dynamics. For each agent cluster with known or inferred S/A/I roles:

  flatness F(v)  = max(0, 1 - Var(v) / Var(s))   controlled var quieter than drive
  compensation K = max(0, -corr(a_{t-1}, s_t))   action opposes disturbance
  regulation R   = F * K

An agent is flagged when max internal R exceeds ``threshold``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from learn_agents.learn_agents import SimulationResult


@dataclass(frozen=True)
class RegulationScore:
    agent: int
    internal_var: str
    internal_idx: int
    sensor_var: str
    sensor_idx: int
    flatness: float
    compensation: float
    regulation: float
    active_ratio: float
    var_controlled: float
    var_disturbance: float


def _series(trace: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    if not indices:
        return np.zeros(trace.shape[0], dtype=np.float32)
    sub = trace[:, list(indices)]
    if sub.ndim == 1 or sub.shape[1] == 1:
        return np.asarray(sub.reshape(-1), dtype=np.float64)
    return np.mean(sub, axis=1).astype(np.float64)


def flatness(controlled: np.ndarray, disturbance: np.ndarray, *, eps: float = 1e-9) -> float:
    vc = float(np.var(controlled))
    vd = float(np.var(disturbance))
    if vd <= eps:
        return 0.0
    return max(0.0, 1.0 - vc / (vd + eps))


def active_internal_ratio(controlled: np.ndarray, disturbance: np.ndarray, *, eps: float = 1e-9) -> float:
    """Var(internal) / Var(sensor drive). High => internal not suppressed."""
    vd = float(np.var(disturbance))
    if vd <= eps:
        return float("inf")
    return float(np.var(controlled) / (vd + eps))


def compensation(
    action: np.ndarray,
    disturbance: np.ndarray,
    *,
    lag: int = 1,
    eps: float = 1e-9,
) -> float:
    if action.shape[0] <= lag or disturbance.shape[0] <= lag:
        return 0.0
    a = action[lag - 1 : -1] - np.mean(action[lag - 1 : -1])
    s = disturbance[lag:] - np.mean(disturbance[lag:])
    denom = float(np.std(a) * np.std(s))
    if denom <= eps:
        return 0.0
    corr = float(np.corrcoef(a, s)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return max(0.0, -corr)


def regulation_index(
    controlled: np.ndarray,
    disturbance: np.ndarray,
    action: np.ndarray,
    *,
    lag: int = 1,
    active_ratio_max: float = 0.012,
) -> Tuple[float, float, float, float]:
    f = flatness(controlled, disturbance)
    k = compensation(action, disturbance, lag=lag)
    ratio = active_internal_ratio(controlled, disturbance)
    if ratio > active_ratio_max:
        return f, k, 0.0, ratio
    return f, k, f * k, ratio


def _pair_disturbance_indices(
    sensor_idx: Sequence[int],
    internal_idx: int,
    internal_indices: Sequence[int],
) -> int:
    if not sensor_idx:
        raise ValueError("sensor_idx must be non-empty")
    if len(sensor_idx) == 1:
        return int(sensor_idx[0])
    if len(sensor_idx) == len(internal_indices):
        pos = internal_indices.index(internal_idx)
        return int(sensor_idx[pos])
    return int(sensor_idx[0])


def _active_slice(trace: np.ndarray, active_mask: Optional[np.ndarray]) -> np.ndarray:
    if active_mask is None:
        return trace
    mask = np.asarray(active_mask, dtype=bool)
    if mask.shape[0] != trace.shape[0]:
        raise ValueError("active_mask length must match trace T")
    if mask.sum() < 8:
        return trace
    return trace[mask]


def score_agent_roles(
    trace: np.ndarray,
    role_indices: Mapping[Tuple[int, str], Sequence[int]],
    var_names: Sequence[str],
    agent: int,
    *,
    active_mask: Optional[np.ndarray] = None,
    lag: int = 1,
    active_ratio_max: float = 0.012,
) -> List[RegulationScore]:
    """Score each internal variable for one agent using its S/A/I role indices."""
    trace_use = _active_slice(trace, active_mask)
    sensor_idx = list(role_indices.get((agent, "sensor"), []))
    action_idx = list(role_indices.get((agent, "action"), []))
    internal_idx = list(role_indices.get((agent, "internal"), []))
    if not internal_idx or not sensor_idx or not action_idx:
        return []

    action = _series(trace_use, action_idx)
    out: List[RegulationScore] = []
    for idx in internal_idx:
        s_idx = _pair_disturbance_indices(sensor_idx, idx, internal_idx)
        disturbance = trace_use[:, s_idx].astype(np.float64)
        controlled = trace_use[:, idx].astype(np.float64)
        f, k, r, ratio = regulation_index(
            controlled,
            disturbance,
            action,
            lag=lag,
            active_ratio_max=active_ratio_max,
        )
        out.append(
            RegulationScore(
                agent=int(agent),
                internal_var=str(var_names[idx]),
                internal_idx=int(idx),
                sensor_var=str(var_names[s_idx]),
                sensor_idx=int(s_idx),
                flatness=f,
                compensation=k,
                regulation=r,
                active_ratio=ratio,
                var_controlled=float(np.var(controlled)),
                var_disturbance=float(np.var(disturbance)),
            )
        )
    return out


def score_simulation(
    result: SimulationResult,
    *,
    threshold: float = 0.15,
    lag: int = 1,
    active_ratio_max: float = 0.012,
    min_T: int = 80,
    active_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Score all agents in a simulation result; flag homeostatic regulation."""
    trace = result.trace
    if trace.shape[0] < min_T:
        return {
            "threshold": threshold,
            "active_ratio_max": active_ratio_max,
            "lag": lag,
            "min_T": min_T,
            "T": int(trace.shape[0]),
            "flagged_agents": [],
            "agents": {},
            "skipped": "trace_too_short",
        }
    meta = result.metadata
    role_indices = meta["role_indices"]
    var_names = list(meta["var_names"])
    agent_clusters = meta.get("agent_clusters", {})
    agents = sorted(agent_clusters.keys()) if agent_clusters else sorted(
        {int(k[0]) for k in role_indices if k[0] >= 0}
    )

    per_agent: Dict[int, Dict[str, object]] = {}
    flagged: List[int] = []
    all_scores: List[RegulationScore] = []

    for agent in agents:
        scores = score_agent_roles(
            trace,
            role_indices,
            var_names,
            int(agent),
            active_mask=active_mask,
            lag=lag,
            active_ratio_max=active_ratio_max,
        )
        all_scores.extend(scores)
        if not scores:
            per_agent[int(agent)] = {
                "max_regulation": 0.0,
                "best_internal": None,
                "scores": [],
                "flagged": False,
            }
            continue
        best = max(scores, key=lambda s: s.regulation)
        is_flagged = best.regulation >= threshold
        if is_flagged:
            flagged.append(int(agent))
        per_agent[int(agent)] = {
            "max_regulation": float(best.regulation),
            "best_internal": best.internal_var,
            "flatness": float(best.flatness),
            "compensation": float(best.compensation),
            "active_ratio": float(best.active_ratio),
            "flagged": is_flagged,
            "scores": [
                {
                    "internal_var": s.internal_var,
                    "sensor_var": s.sensor_var,
                    "flatness": s.flatness,
                    "compensation": s.compensation,
                    "regulation": s.regulation,
                    "active_ratio": s.active_ratio,
                    "var_controlled": s.var_controlled,
                    "var_disturbance": s.var_disturbance,
                }
                for s in scores
            ],
        }

    return {
        "threshold": threshold,
        "active_ratio_max": active_ratio_max,
        "lag": lag,
        "min_T": min_T,
        "T": int(trace.shape[0]),
        "flagged_agents": flagged,
        "agents": per_agent,
    }
