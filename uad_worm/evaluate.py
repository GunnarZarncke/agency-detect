"""M5 + M6 — simplest plausibility checks first.

M5: random-class-set null — is the candidate's pooled blanket loss below that of
biologically-matched random class sets? (The valid additional blanket null; circular/phase
nulls belong to M7, README §7.2.)

M6: behavior-prediction gain — does the candidate's neurons predict a behavior variable
better than a same-size random neuron set? CePNEM/connectome overlays come later; these two
cheap checks gate whether richer evaluation is worth running.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Sequence

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

from uad_worm.candidates import classes_to_indices, random_class_sets
from uad_worm.preprocess import Processed
from uad_worm.score import pooled_mean_loss


@dataclass(frozen=True)
class RandomClassSetNull:
    observed_loss: float
    null_losses: np.ndarray
    z: float
    pvalue: float       # P(random-class-set loss <= observed)


def class_universe(processed_list: Sequence[Processed]) -> List[str]:
    classes = set()
    for proc in processed_list:
        classes.update(c for c in proc.neuron_class if c)
    return sorted(classes)


def random_class_set_null(
    processed_list: Sequence[Processed],
    class_set: FrozenSet[str],
    *,
    n_sets: int = 40,
    representation: str = "whitened",
    min_members: int = 3,
    seed: int = 0,
    **score_kwargs,
) -> RandomClassSetNull:
    """Compare the candidate's pooled loss to random class sets of the same size."""
    observed = pooled_mean_loss(
        processed_list, class_set, representation=representation,
        min_members=min_members, **score_kwargs,
    )
    universe = class_universe(processed_list)
    rand_sets = random_class_sets(universe, size=len(class_set), n=n_sets, seed=seed)
    losses = []
    for cs in rand_sets:
        val = pooled_mean_loss(
            processed_list, cs, representation=representation,
            min_members=min_members, **score_kwargs,
        )
        if np.isfinite(val):
            losses.append(val)
    null = np.asarray(losses)
    if null.size == 0 or not np.isfinite(observed):
        return RandomClassSetNull(observed, null, float("nan"), float("nan"))
    mu, sd = float(null.mean()), float(null.std())
    z = (observed - mu) / sd if sd > 1e-12 else 0.0
    pvalue = float((np.sum(null <= observed) + 1) / (null.size + 1))
    return RandomClassSetNull(observed, null, z, pvalue)


def _cv_r2(X: np.ndarray, y: np.ndarray, *, seed: int = 0) -> float:
    if X.shape[1] == 0 or np.std(y) < 1e-9:
        return float("nan")
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2")
    return float(np.mean(scores))


def behavior_prediction_gain(
    proc: Processed,
    class_set: FrozenSet[str],
    *,
    feature: str = "velocity",
    representation: str = "raw",
    n_baseline: int = 10,
    min_members: int = 3,
    seed: int = 0,
) -> Optional[float]:
    """CV-R² predicting `feature` from candidate neurons minus same-size random sets."""
    if feature not in proc.behavior:
        return None
    members = classes_to_indices(class_set, proc.neuron_class)
    if len(members) < min_members:
        return None
    trace = proc.representation(representation)
    y = proc.behavior[feature]
    T = min(trace.shape[0], y.shape[0])
    trace, y = trace[:T], y[:T]
    cand_r2 = _cv_r2(trace[:, members], y, seed=seed)
    rng = np.random.default_rng(seed)
    V = trace.shape[1]
    base = []
    for _ in range(n_baseline):
        idx = rng.permutation(V)[: len(members)]
        r2 = _cv_r2(trace[:, idx], y, seed=seed)
        if np.isfinite(r2):
            base.append(r2)
    if not base or not np.isfinite(cand_r2):
        return None
    return float(cand_r2 - np.mean(base))


def pooled_behavior_gain(
    processed_list: Sequence[Processed],
    class_set: FrozenSet[str],
    *,
    feature: str = "velocity",
    **kwargs,
) -> float:
    """Mean behavior-prediction gain across animals."""
    gains = [
        g for proc in processed_list
        if (g := behavior_prediction_gain(proc, class_set, feature=feature, **kwargs)) is not None
    ]
    return float(np.mean(gains)) if gains else float("nan")
