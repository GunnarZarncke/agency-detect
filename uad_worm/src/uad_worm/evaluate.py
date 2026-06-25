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
from uad_worm.cmi import gaussian_cmi
from uad_worm.preprocess import Processed
from uad_worm.score import _external_pcs, pooled_mean_loss


def internal_autonomy(trace: np.ndarray, members, *, ext_dim: int = 6, lag: int = 1) -> float:
    """Internal autonomy: I(C_{t+1}; C_t | E_t) — self-prediction beyond the environment.

    The companion axis to the blanket loss (FINDINGS.md #4). A *naive* self-prediction R²
    fails: it rewards redundancy (a shared-latent block self-predicts better than a true
    controller — verified on synthetic). Conditioning on the (PC-reduced) environment fixes
    this: a true agent's internal state predicts its own future *beyond* what the
    environment explains (synthetic agent ≈0.70 vs redundant block ≈0.01). Agent signature =
    LOW blanket loss AND HIGH internal autonomy.
    """
    members = list(members)
    if len(members) == 0:
        return float("nan")
    ext_idx = [i for i in range(trace.shape[1]) if i not in set(members)]
    epc = _external_pcs(trace, ext_idx, ext_dim)
    if epc.shape[1] == 0:
        return float("nan")
    fut = trace[lag:][:, members]
    past = trace[:-lag][:, members]
    return gaussian_cmi(fut, past, epc[:-lag])


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


@dataclass(frozen=True)
class JointNull:
    n_members: int
    loss: float
    autonomy: float
    loss_p: float       # P(random loss <= obs): LOW = autonomy/encapsulation is special
    autonomy_p: float   # P(random autonomy <= obs): HIGH = internal dynamics are special
    agent_corner: bool  # loss_p < 0.5 and autonomy_p > 0.5 (low leakage + high internal dynamics)


def joint_null(
    trace: np.ndarray,
    members,
    *,
    ext_dim: int = 6,
    lag: int = 1,
    n_perm: int = 100,
    seed: int = 0,
    pool=None,
) -> Optional[JointNull]:
    """Score a set on BOTH axes (blanket loss + internal autonomy) vs random same-size sets."""
    from uad_worm.score import blanket_loss_for_members

    members = list(members)
    V = trace.shape[1]
    if len(members) < 2 or V - len(members) < 1:
        return None
    obs_loss, _ = blanket_loss_for_members(trace, members, ext_dim=ext_dim, lag=lag)
    obs_aut = internal_autonomy(trace, members, ext_dim=ext_dim, lag=lag)
    if not (np.isfinite(obs_loss) and np.isfinite(obs_aut)):
        return None
    rng = np.random.default_rng(seed)
    draw_from = np.asarray(list(pool)) if pool is not None else np.arange(V)
    size = len(members)
    losses, auts = [], []
    for _ in range(n_perm):
        rm = rng.permutation(draw_from)[:size]
        loss_k, _ = blanket_loss_for_members(trace, rm, ext_dim=ext_dim, lag=lag)
        aut_k = internal_autonomy(trace, rm, ext_dim=ext_dim, lag=lag)
        if np.isfinite(loss_k) and np.isfinite(aut_k):
            losses.append(loss_k)
            auts.append(aut_k)
    if not losses:
        return None
    losses, auts = np.asarray(losses), np.asarray(auts)
    loss_p = float((np.sum(losses <= obs_loss) + 1) / (losses.size + 1))
    autonomy_p = float((np.sum(auts <= obs_aut) + 1) / (auts.size + 1))
    return JointNull(
        n_members=size, loss=float(obs_loss), autonomy=float(obs_aut),
        loss_p=loss_p, autonomy_p=autonomy_p,
        agent_corner=bool(loss_p < 0.5 and autonomy_p > 0.5),
    )


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
