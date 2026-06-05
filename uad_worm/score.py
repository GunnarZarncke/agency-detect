"""M4 — blanket scoring with operational roles + pooled leave-one-animal-out.

Per animal: assign S/A/I roles inside a candidate's neurons by lagged influence to/from
the rest of the brain, reduce the external set to a few PCs (the blanket CMI is otherwise
singular at T≈1600), and score the blanket loss against a random-partition contrast. Then
pool the same class-defined candidate across animals and run leave-one-animal-out — the
v1 headline (README §5 "Recommended v1 bet").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, FrozenSet, List, Optional, Sequence

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

from uad_worm.blanket import blanket_loss
from uad_worm.candidates import classes_to_indices
from uad_worm.cmi import gaussian_cmi
from uad_worm.preprocess import Processed


@dataclass(frozen=True)
class RoleAssignment:
    sensors: List[int]
    actions: List[int]
    internal: List[int]


@dataclass(frozen=True)
class CandidateScore:
    n_members: int
    loss: float
    pvalue: float       # one-sided: P(random-partition loss <= observed)
    z: float
    roles: RoleAssignment


def _external_pcs(trace: np.ndarray, ext_idx: Sequence[int], ext_dim: int) -> np.ndarray:
    if len(ext_idx) == 0:
        return np.zeros((trace.shape[0], 0))
    ext = trace[:, list(ext_idx)]
    k = min(ext_dim, ext.shape[1])
    return PCA(n_components=k, svd_solver="full").fit_transform(ext)


def _cross_corr(past: np.ndarray, future: np.ndarray) -> np.ndarray:
    """Correlation matrix C[i,j] = corr(past_i(t), future_j(t)); shape (A, B)."""
    a = (past - past.mean(0)) / (past.std(0) + 1e-12)
    b = (future - future.mean(0)) / (future.std(0) + 1e-12)
    return (a.T @ b) / a.shape[0]


def assign_roles(
    trace: np.ndarray,
    members: Sequence[int],
    epc: np.ndarray,
    *,
    n_sensors: int = 1,
    n_actions: int = 1,
) -> RoleAssignment:
    """S = members most driven by E_t; A = members most driving E_{t+1}; I = the rest."""
    mem = list(members)
    m_trace = trace[:, mem]
    if epc.shape[1] == 0:
        in_score = np.zeros(len(mem))
        out_score = np.zeros(len(mem))
    else:
        # in_score[i] = mean_p |corr(E_pc(t), member_i(t+1))|  (E drives member)
        in_score = np.mean(np.abs(_cross_corr(epc[:-1], m_trace[1:])), axis=0)
        # out_score[i] = mean_p |corr(member_i(t), E_pc(t+1))|  (member drives E)
        out_score = np.mean(np.abs(_cross_corr(m_trace[:-1], epc[1:])), axis=1)
    # Cap role counts so at least one internal node remains.
    n_sensors = min(n_sensors, max(0, len(mem) - 1))
    n_actions = min(n_actions, max(0, len(mem) - 1 - n_sensors))
    sensors = [mem[i] for i in np.argsort(in_score)[::-1][:n_sensors]]
    remaining = [m for m in mem if m not in sensors]
    out_remaining = np.array([out_score[mem.index(m)] for m in remaining])
    actions = [remaining[i] for i in np.argsort(out_remaining)[::-1][:n_actions]]
    internal = [m for m in mem if m not in sensors and m not in actions]
    return RoleAssignment(sensors=sensors, actions=actions, internal=internal)


def blanket_loss_for_members(
    trace: np.ndarray,
    members: Sequence[int],
    *,
    n_sensors: int = 1,
    n_actions: int = 1,
    ext_dim: int = 6,
    lag: int = 1,
) -> tuple[float, RoleAssignment]:
    """Assign roles, reduce E to PCs, return I(I_{t+1}; E_{t+1} | S_t, A_t)."""
    members = list(members)
    ext_idx = [i for i in range(trace.shape[1]) if i not in set(members)]
    epc = _external_pcs(trace, ext_idx, ext_dim)
    roles = assign_roles(trace, members, epc, n_sensors=n_sensors, n_actions=n_actions)
    if not roles.internal or epc.shape[1] == 0:
        return float("nan"), roles
    fut_int = trace[lag:][:, roles.internal]
    fut_ext = epc[lag:]
    cond_cols = roles.sensors + roles.actions
    cond = trace[:-lag][:, cond_cols] if cond_cols else None
    loss = gaussian_cmi(fut_int, fut_ext, cond)
    return loss, roles


def score_members(
    trace: np.ndarray,
    members: Sequence[int],
    *,
    n_sensors: int = 1,
    n_actions: int = 1,
    ext_dim: int = 6,
    lag: int = 1,
    n_perm: int = 100,
    seed: int = 0,
    pool: Optional[Sequence[int]] = None,
) -> Optional[CandidateScore]:
    """Blanket loss of `members` + random-partition contrast (matched size).

    `pool` restricts which neuron indices the random partitions are drawn from (default:
    all neurons). Used by the probe to test a labeled-only null reference.
    """
    members = list(members)
    if len(members) < 2 or trace.shape[1] - len(members) < 1:
        return None
    obs, roles = blanket_loss_for_members(
        trace, members, n_sensors=n_sensors, n_actions=n_actions, ext_dim=ext_dim, lag=lag
    )
    if not np.isfinite(obs):
        return None
    rng = np.random.default_rng(seed)
    draw_from = np.asarray(list(pool)) if pool is not None else np.arange(trace.shape[1])
    size = len(members)
    if draw_from.size < size:
        return None
    null = np.empty(n_perm)
    for k in range(n_perm):
        rand_members = rng.permutation(draw_from)[:size]
        loss_k, _ = blanket_loss_for_members(
            trace, rand_members, n_sensors=n_sensors, n_actions=n_actions,
            ext_dim=ext_dim, lag=lag,
        )
        null[k] = loss_k
    null = null[np.isfinite(null)]
    if null.size == 0:
        return None
    mu, sd = float(null.mean()), float(null.std())
    z = (obs - mu) / sd if sd > 1e-12 else 0.0
    pvalue = float((np.sum(null <= obs) + 1) / (null.size + 1))
    return CandidateScore(n_members=size, loss=float(obs), pvalue=pvalue, z=z, roles=roles)


def score_candidate_per_animal(
    processed: Processed,
    class_set: FrozenSet[str],
    *,
    representation: str = "whitened",
    min_members: int = 3,
    **kwargs,
) -> Optional[CandidateScore]:
    """Map a class set to this animal's neurons and score it; None if too few present."""
    members = classes_to_indices(class_set, processed.neuron_class)
    if len(members) < min_members:
        return None
    return score_members(processed.representation(representation), members, **kwargs)


def pooled_mean_loss(
    processed_list: Sequence[Processed],
    class_set: FrozenSet[str],
    *,
    representation: str = "whitened",
    min_members: int = 3,
    n_sensors: int = 1,
    n_actions: int = 1,
    ext_dim: int = 6,
    lag: int = 1,
) -> float:
    """Mean blanket loss of a class set across animals (no permutation; fast path).

    Used by the random-class-set null (M5), which only needs each set's loss, not its own
    random-partition contrast.
    """
    losses = []
    for proc in processed_list:
        members = classes_to_indices(class_set, proc.neuron_class)
        if len(members) < min_members:
            continue
        loss, _ = blanket_loss_for_members(
            proc.representation(representation), members,
            n_sensors=n_sensors, n_actions=n_actions, ext_dim=ext_dim, lag=lag,
        )
        if np.isfinite(loss):
            losses.append(loss)
    return float(np.mean(losses)) if losses else float("nan")


@dataclass(frozen=True)
class PooledScore:
    n_animals: int
    mean_loss: float
    pass_rate: float        # fraction of animals with p < 0.05
    median_z: float
    combined_p: float       # Fisher's method across animals
    per_animal_p: List[float]


def _fisher(pvals: Sequence[float]) -> float:
    p = np.clip(np.asarray(pvals, dtype=float), 1e-12, 1.0)
    if p.size == 0:
        return float("nan")
    chi2 = -2.0 * np.sum(np.log(p))
    return float(stats.chi2.sf(chi2, df=2 * p.size))


def pooled_score(
    processed_list: Sequence[Processed],
    class_set: FrozenSet[str],
    *,
    representation: str = "whitened",
    min_members: int = 3,
    **kwargs,
) -> PooledScore:
    """Pool one class-defined candidate across animals (README §6 statistic pooling)."""
    losses, zs, ps = [], [], []
    for proc in processed_list:
        res = score_candidate_per_animal(
            proc, class_set, representation=representation, min_members=min_members, **kwargs
        )
        if res is None:
            continue
        losses.append(res.loss)
        zs.append(res.z)
        ps.append(res.pvalue)
    n = len(losses)
    if n == 0:
        return PooledScore(0, float("nan"), float("nan"), float("nan"), float("nan"), [])
    return PooledScore(
        n_animals=n,
        mean_loss=float(np.mean(losses)),
        pass_rate=float(np.mean([p < 0.05 for p in ps])),
        median_z=float(np.median(zs)),
        combined_p=_fisher(ps),
        per_animal_p=ps,
    )


def leave_one_animal_out(
    processed_list: Sequence[Processed],
    candidate_fn: Callable[[Sequence[Processed]], FrozenSet[str]],
    *,
    representation: str = "whitened",
    min_members: int = 3,
    **kwargs,
) -> dict:
    """Define the candidate on N-1 animals, test on the held-out one; repeat.

    `candidate_fn(train)` returns a class set (e.g. a fixed anchor, or a data-driven
    recurrent set). Returns held-out pass rate — the generalization headline.
    """
    held_p: List[float] = []
    held_loss: List[float] = []
    for i in range(len(processed_list)):
        train = [p for j, p in enumerate(processed_list) if j != i]
        test = processed_list[i]
        class_set = candidate_fn(train)
        res = score_candidate_per_animal(
            test, class_set, representation=representation, min_members=min_members, **kwargs
        )
        if res is None:
            continue
        held_p.append(res.pvalue)
        held_loss.append(res.loss)
    if not held_p:
        return {"n_held": 0, "holdout_pass_rate": float("nan"), "mean_holdout_loss": float("nan")}
    return {
        "n_held": len(held_p),
        "holdout_pass_rate": float(np.mean([p < 0.05 for p in held_p])),
        "mean_holdout_loss": float(np.mean(held_loss)),
        "held_p": held_p,
    }
