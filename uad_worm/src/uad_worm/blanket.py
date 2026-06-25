"""Blanket loss and the random-partition contrast.

The discovery criterion is the one-step blanket loss

    L_blanket(C) = I(I^C_{t+1}; E^C_{t+1} | S^C_t, A^C_t)

estimated with the Gaussian CMI (`uad_worm.cmi`). A *low* absolute loss is not enough
(README §7.2): a structureless system can also look independent. The arbiter for the
blanket is therefore the **random-partition contrast** — does *this* assignment of
variables to roles mediate better than chance assignments of the same sizes drawn from
the same variable universe?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from uad_worm.cmi import gaussian_cmi


def _col(trace: np.ndarray, idx: Sequence[int]) -> np.ndarray:
    idx = list(idx)
    if not idx:
        return np.empty((trace.shape[0], 0), dtype=np.float64)
    return np.asarray(trace[:, idx], dtype=np.float64)


def blanket_loss(
    trace: np.ndarray,
    internal: Sequence[int],
    external: Sequence[int],
    interface: Sequence[int],
    *,
    lag: int = 1,
    ridge: float = 1e-6,
) -> float:
    """I(internal_{t+1}; external_{t+1} | interface_t).

    `interface` is the conditioning set at time t (the candidate's sensors + actions).
    """
    if lag < 1:
        raise ValueError("lag must be >= 1")
    if len(internal) == 0 or len(external) == 0:
        raise ValueError("internal and external must be non-empty")
    fut_int = _col(trace[lag:], internal)
    fut_ext = _col(trace[lag:], external)
    cond = _col(trace[:-lag], interface)
    return gaussian_cmi(fut_int, fut_ext, cond if cond.shape[1] else None, ridge=ridge)


@dataclass(frozen=True)
class PartitionNull:
    observed: float
    null: np.ndarray
    z: float
    pvalue: float


def random_partition_null(
    trace: np.ndarray,
    *,
    n_internal: int,
    n_interface: int,
    n_external: int,
    observed: float,
    n_perm: int = 200,
    lag: int = 1,
    ridge: float = 1e-6,
    seed: int = 0,
) -> PartitionNull:
    """Null distribution of blanket loss over random role assignments of matched size.

    Draws disjoint random index sets of the given sizes from the variable universe and
    recomputes the blanket loss. A genuine blanket has `observed` BELOW this null
    (one-sided low p-value); a structureless cut sits inside it.
    """
    rng = np.random.default_rng(seed)
    V = trace.shape[1]
    need = n_internal + n_interface + n_external
    if need > V:
        raise ValueError(f"need {need} variables for a random partition but only {V} exist")
    losses = np.empty(n_perm, dtype=np.float64)
    for k in range(n_perm):
        perm = rng.permutation(V)
        internal = perm[:n_internal]
        interface = perm[n_internal : n_internal + n_interface]
        external = perm[n_internal + n_interface : need]
        losses[k] = blanket_loss(
            trace, internal, external, interface, lag=lag, ridge=ridge
        )
    mu = float(losses.mean())
    sd = float(losses.std())
    z = (observed - mu) / sd if sd > 1e-12 else 0.0
    pvalue = float((np.sum(losses <= observed) + 1) / (n_perm + 1))
    return PartitionNull(observed=observed, null=losses, z=z, pvalue=pvalue)


def blanket_pvalue(
    trace: np.ndarray,
    internal: Sequence[int],
    external: Sequence[int],
    interface: Sequence[int],
    *,
    n_perm: int = 200,
    lag: int = 1,
    ridge: float = 1e-6,
    seed: int = 0,
) -> PartitionNull:
    """Blanket loss of a given assignment + its random-partition contrast."""
    obs = blanket_loss(trace, internal, external, interface, lag=lag, ridge=ridge)
    return random_partition_null(
        trace,
        n_internal=len(internal),
        n_interface=len(interface),
        n_external=len(external),
        observed=obs,
        n_perm=n_perm,
        lag=lag,
        ridge=ridge,
        seed=seed,
    )
