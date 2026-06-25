"""Threat-conditioned defense score (pipeline-style mean shift / odds ratio)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from intention_detect.influence import _residualize


def defense_odds_ratio(
    action_composite: np.ndarray,
    outcome: np.ndarray,
    controls: np.ndarray,
    *,
    direction: str = "lower_is_better",
    bad_quantile: float = 0.80,
    n_bootstrap: int = 120,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Return (OR point, OR low 90% CI, selectivity ratio).

    OR compares mean protective action when outcome is bad vs good, after
    residualizing on controls. Selectivity = OR_active / OR_passive where
    passive windows are those with low outcome variance (proxy for inactive stages).
    """
    if len(outcome) < 40:
        return 1.0, 1.0, 1.0

    if controls.ndim == 1:
        w = controls.reshape(-1, 1)
    elif controls.size == 0:
        w = np.zeros((len(outcome), 0))
    else:
        w = controls

    y_res = _residualize(action_composite.astype(np.float64), w)
    o_res = _residualize(outcome.astype(np.float64), w)

    if direction == "lower_is_better":
        bad = o_res >= float(np.quantile(o_res, bad_quantile))
    else:
        bad = o_res <= float(np.quantile(o_res, 1.0 - bad_quantile))

    good = ~bad
    if bad.sum() < 5 or good.sum() < 5:
        return 1.0, 1.0, 1.0

    eps = 1e-6
    mean_bad = float(np.mean(np.abs(y_res[bad])))
    mean_good = float(np.mean(np.abs(y_res[good])))
    or_point = (mean_bad + eps) / (mean_good + eps)

    rng = np.random.default_rng(seed)
    boots: list[float] = []
    n = len(y_res)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        b = bad[idx]
        g = ~b
        if b.sum() < 3 or g.sum() < 3:
            continue
        mb = float(np.mean(np.abs(y_res[idx][b])))
        mg = float(np.mean(np.abs(y_res[idx][g])))
        boots.append((mb + eps) / (mg + eps))
    if not boots:
        or_low = or_point
    else:
        or_low = float(np.quantile(boots, 0.10))

    # Selectivity: high |Δoutcome| windows vs low (active vs passive dynamics)
    do = np.abs(np.diff(outcome, prepend=outcome[0]))
    active = do >= float(np.quantile(do, 0.60))
    passive = do <= float(np.quantile(do, 0.40))
    if active.sum() >= 8 and passive.sum() >= 8:
        or_active = _subset_or(y_res, bad, active, eps)
        or_passive = _subset_or(y_res, bad, passive, eps)
        selectivity = or_active / (or_passive + eps)
    else:
        selectivity = 1.0

    return or_point, or_low, float(selectivity)


def _subset_or(y_res: np.ndarray, bad: np.ndarray, mask: np.ndarray, eps: float) -> float:
    m = mask & bad
    g = mask & (~bad)
    if m.sum() < 3 or g.sum() < 3:
        return 1.0
    return (float(np.mean(np.abs(y_res[m]))) + eps) / (float(np.mean(np.abs(y_res[g]))) + eps)
