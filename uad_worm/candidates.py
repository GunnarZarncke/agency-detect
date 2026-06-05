"""M3 — candidate subsystem generation (single primary approach first).

Candidates are defined at the **neuron_class** level (frozensets of NeuroPAL class names)
so the same hypothesis can be pooled across animals (README §5, §6). Per the
do-one-thing-first discipline, the *primary* unsupervised generator is lagged-correlation
agglomerative communities; the biologically-grounded command-circuit anchor seed and
random matched class sets round out the candidate pool. Alternative clusterings
(Granger/spectral/connectome-constrained) are deferred unless M4 shows no signal.
"""

from __future__ import annotations

from typing import FrozenSet, List, Optional, Sequence

import numpy as np
from sklearn.cluster import AgglomerativeClustering

# Locomotor command circuit (seed only; accepted/rejected by UAD, never supervision).
COMMAND_CIRCUIT_REVERSE = ("AVA", "AVE", "AVD", "RIM", "AIB")
COMMAND_CIRCUIT_FORWARD = ("AVB", "PVC", "RIB")
ANCHOR_CLASSES: FrozenSet[str] = frozenset(COMMAND_CIRCUIT_REVERSE + COMMAND_CIRCUIT_FORWARD)


def lagged_affinity(trace: np.ndarray, lag: int = 1) -> np.ndarray:
    """Symmetric lagged-correlation affinity A[i,j] = max(|corr(i_t, j_{t+lag})|, swap).

    Captures directed temporal dependence in either direction; diagonal zeroed.
    """
    a = trace[:-lag]
    b = trace[lag:]
    a = (a - a.mean(0)) / (a.std(0) + 1e-12)
    b = (b - b.mean(0)) / (b.std(0) + 1e-12)
    n = a.shape[0]
    c = (a.T @ b) / n  # c[i,j] = corr(x_i(t), x_j(t+lag))
    aff = np.maximum(np.abs(c), np.abs(c.T))
    np.fill_diagonal(aff, 0.0)
    return aff


def agglomerative_communities(
    trace: np.ndarray,
    *,
    lag: int = 1,
    distance_threshold: float = 0.6,
    min_size: int = 3,
    max_size: int = 30,
) -> List[List[int]]:
    """Cluster neurons by lagged-correlation distance (1 - affinity); keep mid-size groups.

    `distance_threshold=0.6` ⇒ groups neurons with lagged |corr| ≳ 0.4; `max_size` blocks
    the trivial giant cluster (README §14).
    """
    aff = lagged_affinity(trace, lag=lag)
    dist = 1.0 - aff
    np.fill_diagonal(dist, 0.0)
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    labels = model.fit_predict(dist)
    out: List[List[int]] = []
    for lab in np.unique(labels):
        members = [int(i) for i in np.where(labels == lab)[0]]
        if min_size <= len(members) <= max_size:
            out.append(members)
    return out


def classes_of(indices: Sequence[int], neuron_class: Sequence[Optional[str]]) -> FrozenSet[str]:
    """Labeled neuron classes present among the given neuron indices."""
    return frozenset(neuron_class[i] for i in indices if neuron_class[i])


def community_class_sets(
    communities: Sequence[Sequence[int]],
    neuron_class: Sequence[Optional[str]],
    *,
    min_classes: int = 3,
) -> List[FrozenSet[str]]:
    """Convert index communities to class sets with at least `min_classes` labeled members."""
    out: List[FrozenSet[str]] = []
    seen = set()
    for members in communities:
        cs = classes_of(members, neuron_class)
        if len(cs) >= min_classes and cs not in seen:
            seen.add(cs)
            out.append(cs)
    return out


def classes_to_indices(
    class_set: FrozenSet[str], neuron_class: Sequence[Optional[str]]
) -> List[int]:
    """Neuron indices in one animal whose class is in `class_set`."""
    return [i for i, c in enumerate(neuron_class) if c in class_set]


def random_class_sets(
    universe: Sequence[str], *, size: int, n: int, seed: int = 0
) -> List[FrozenSet[str]]:
    """`n` random class sets of the given size drawn from the class universe (controls)."""
    rng = np.random.default_rng(seed)
    uni = list(dict.fromkeys(universe))
    size = min(size, len(uni))
    out: List[FrozenSet[str]] = []
    for _ in range(n):
        out.append(frozenset(rng.choice(uni, size=size, replace=False)))
    return out
