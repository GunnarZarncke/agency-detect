"""Automatic window/activity segmentation for episodic outcome-influence scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SegmentParams:
    window: int
    step: int
    min_len: int
    activity_coef: float  # active when action > median + coef * std


def _action_abs(trace: np.ndarray, action_indices: Sequence[int]) -> np.ndarray:
    if not action_indices:
        return np.zeros(trace.shape[0], dtype=np.float64)
    sub = trace[:, list(action_indices)]
    if sub.ndim == 1:
        return np.abs(sub.astype(np.float64))
    return np.mean(np.abs(sub), axis=1).astype(np.float64)


def agent_active_fraction(action: np.ndarray) -> float:
    """Fraction of ticks above an auto activity threshold for this agent."""
    a = np.asarray(action, dtype=np.float64)
    mean_a = float(np.mean(np.abs(a)))
    std_a = float(np.std(a))
    if mean_a < 1e-6:
        return 0.0
    if std_a / (mean_a + 1e-6) < 0.08:
        return 1.0  # constant action → always "on"
    thr = float(np.median(a) + 0.25 * std_a)
    return float(np.mean(a > thr))


def should_segment(
    trace: np.ndarray,
    metadata: Mapping[str, object],
    *,
    min_T: int = 250,
    max_median_active_frac: float = 0.40,
) -> bool:
    """Auto-enable segmentation on long traces with sparse/episodic agents."""
    T = trace.shape[0]
    if T < min_T:
        return False
    if metadata.get("prefer_segment_scoring"):
        return True

    role_indices = metadata["role_indices"]
    agents = sorted({int(k[0]) for k in role_indices if k[0] >= 0})
    if not agents:
        return False

    fracs = [
        agent_active_fraction(_action_abs(trace, role_indices.get((a, "action"), [])))
        for a in agents
    ]
    # Machine-like: at least one clearly sparse agent and low median duty cycle.
    return min(fracs) < 0.22 and float(np.median(fracs)) < max_median_active_frac


def calibrate_segment_params(T: int, action: np.ndarray) -> SegmentParams:
    """Pick window length and activity threshold from trace length and action sparsity."""
    window = int(np.clip(T // 6, 80, 300))
    step = max(window // 2, 20)
    min_len = max(40, min(window // 2, 120))

    a = np.asarray(action, dtype=np.float64)
    mean_a = float(np.mean(np.abs(a)))
    std_a = float(np.std(a))
    # Constant or near-constant action → sliding windows only (no activity mask).
    if mean_a < 1e-6 or std_a / (mean_a + 1e-6) < 0.08:
        activity_coef = float("inf")
    else:
        # Sparser agents get a lower threshold (more inclusive of burst edges).
        active_frac = float(np.mean(a > np.median(a)))
        activity_coef = 0.15 + 0.45 * (1.0 - active_frac)

    return SegmentParams(window=window, step=step, min_len=min_len, activity_coef=activity_coef)


def _active_runs(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        if j - i >= min_len:
            runs.append((i, j))
        i = j
    return runs


def segment_ranges(
    T: int,
    action: np.ndarray,
    params: SegmentParams,
) -> List[Tuple[int, int]]:
    """Return [start, end) slices: sliding windows plus activity-bounded runs."""
    if T < params.min_len:
        return [(0, T)]

    ranges: List[Tuple[int, int]] = []
    for start in range(0, max(1, T - params.window + 1), params.step):
        end = min(T, start + params.window)
        if end - start >= params.min_len:
            ranges.append((start, end))

    if np.isfinite(params.activity_coef):
        thr = float(np.median(action) + params.activity_coef * np.std(action))
        mask = action > thr
        for start, end in _active_runs(mask, params.min_len):
            # Pad short active runs to at least window when possible.
            if end - start < params.window:
                pad = params.window - (end - start)
                start = max(0, start - pad // 2)
                end = min(T, start + params.window)
                start = max(0, end - params.window)
            ranges.append((start, end))

    if not ranges:
        return [(0, T)]

    # Deduplicate exact duplicates only; keep overlapping windows separate so
    # episodic bursts are not collapsed into one full-trace average.
    seen = set()
    out: List[Tuple[int, int]] = []
    for start, end in sorted(ranges):
        key = (start, end)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out

