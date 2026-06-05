"""Partialled lagged influence of agent actions on critical outcomes."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    if x.size == 0 or x.shape[1] == 0:
        return y - float(np.mean(y))
    x_ = np.column_stack([np.ones(len(y)), x]) if x.ndim == 1 else np.column_stack([np.ones(len(x)), x])
    beta, _, _, _ = np.linalg.lstsq(x_, y, rcond=None)
    return y - x_ @ beta


def _series(trace: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    if not indices:
        return np.zeros(trace.shape[0], dtype=np.float64)
    sub = trace[:, list(indices)]
    if sub.ndim == 1 or sub.shape[1] == 1:
        return np.asarray(sub.reshape(-1), dtype=np.float64)
    return np.mean(sub, axis=1).astype(np.float64)


def partial_influence(
    action: np.ndarray,
    outcome: np.ndarray,
    controls: np.ndarray,
    *,
    lag: int = 1,
    direction: str = "lower_is_better",
) -> float:
    """Signed partial corr(action_{t-1}, Δoutcome_t | controls).

    For lower_is_better outcomes, positive score means action opposes worsening load.
    """
    if len(outcome) <= lag + 8:
        return 0.0
    delta_o = outcome[lag:] - outcome[lag - 1 : -1]
    a_lag = action[lag - 1 : -1]
    w = controls[lag:] if controls.ndim == 2 else controls[lag:].reshape(-1, 1)
    a_res = _residualize(a_lag, w)
    d_res = _residualize(delta_o, w)
    denom = float(np.std(a_res) * np.std(d_res))
    if denom < 1e-9:
        return 0.0
    corr = float(np.corrcoef(a_res, d_res)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    if direction == "lower_is_better":
        return -corr
    return corr


def influence_from_trace(
    trace: np.ndarray,
    action_indices: Sequence[int],
    outcome_index: int,
    control_indices: Sequence[int],
    *,
    lag: int = 1,
    direction: str = "lower_is_better",
) -> float:
    action = _series(trace, action_indices)
    outcome = trace[:, outcome_index].astype(np.float64)
    if control_indices:
        controls = trace[:, list(control_indices)].astype(np.float64)
    else:
        controls = np.zeros((trace.shape[0], 0), dtype=np.float64)
    return partial_influence(action, outcome, controls, lag=lag, direction=direction)
