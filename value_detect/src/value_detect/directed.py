"""Directed-information estimators (the measuring tools).

Everything is built from one smoothed, discrete estimator of conditional mutual
information, matching the conventions in Gunnar's ``agency_detect/markov_blanket.py``
(Laplace smoothing with alpha = 0.1, natural log / nats, marginal-consistent denominators).
We reimplement it cleanly here with our own tests rather than importing his internals.

The one directed primitive is **transfer entropy**: how much the past of a *source*
variable tells you about the *next* step of a *target*, once the target's own past is
accounted for. Formally ``TE(source -> target) = I(target_{t+1}; source_t | target_t)``,
optionally conditioning on extra variables (used to demonstrate mediation).

The push-in / push-out *value* measures are assembled from these primitives in Chunk 4,
once SJ has chosen the "rest of the system" conventions.
"""

from __future__ import annotations

from math import log
from typing import Optional, Sequence, Tuple

import numpy as np

ALPHA = 0.1  # Laplace smoothing, matching markov_blanket.py.


def _labels(a: np.ndarray) -> np.ndarray:
    """Turn a 1-D or 2-D integer array into 1-D integer labels (state-tupling)."""
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    _, inv = np.unique(a, axis=0, return_inverse=True)
    return inv.astype(np.int64)


def _stack(*cols: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Column-stack the non-None inputs into a 2-D array, or return None if all are None."""
    parts = []
    for c in cols:
        if c is None:
            continue
        c = np.asarray(c)
        parts.append(c.reshape(-1, 1) if c.ndim == 1 else c)
    if not parts:
        return None
    return np.hstack(parts)


def _cmi_labeled(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    alpha: float = ALPHA,
) -> float:
    """CMI from pre-computed dense integer label arrays (vectorized counting).

    Same smoothing scheme as before (alpha on the finest cell, marginals scaled by the
    summed-out cardinalities, shared denominator); the cell sum is vectorized so floors
    can afford thousands of shuffled re-measurements.
    """
    n = len(X)
    card_x = int(X.max()) + 1
    card_y = int(Y.max()) + 1
    card_z = int(Z.max()) + 1

    flat = (X * card_y + Y) * card_z + Z
    counts = np.bincount(flat, minlength=card_x * card_y * card_z).astype(np.float64)
    xyz = counts.reshape(card_x, card_y, card_z)
    xz = xyz.sum(axis=1)[:, None, :]  # broadcast over y
    yz = xyz.sum(axis=0)[None, :, :]  # broadcast over x
    zc = xyz.sum(axis=(0, 1))[None, None, :]

    denom = n + alpha * card_x * card_y * card_z
    p_xyz = (xyz + alpha) / denom
    p_xz = (xz + alpha * card_y) / denom
    p_yz = (yz + alpha * card_x) / denom
    p_z = (zc + alpha * card_x * card_y) / denom
    total = float(np.sum(p_xyz * np.log(p_xyz * p_z / (p_xz * p_yz))))
    return max(0.0, total)


def cmi(x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray] = None, alpha: float = ALPHA) -> float:
    """Smoothed plug-in estimate of I(X; Y | Z) in nats.

    X, Y, Z may each be 1-D or 2-D (multiple columns are tupled into a joint state).
    Z=None gives plain mutual information I(X; Y). Matches the smoothing scheme in
    ``markov_blanket.conditional_mutual_info_discrete``: the pseudocount alpha sits on the
    finest (x,y,z) cell and marginals get alpha scaled by the summed-out cardinalities, so
    every probability shares the denominator ``n + alpha * card_x * card_y * card_z``.
    """
    X = _labels(x)
    Y = _labels(y)
    n = len(X)
    if n == 0 or len(Y) != n:
        raise ValueError("x and y must be non-empty and equal length")
    if z is None:
        Z = np.zeros(n, dtype=np.int64)
    else:
        Z = _labels(z)
        if len(Z) != n:
            raise ValueError("z must match x/y length")
    return _cmi_labeled(X, Y, Z, alpha)


def mi(x: np.ndarray, y: np.ndarray, alpha: float = ALPHA) -> float:
    """Smoothed plug-in estimate of I(X; Y) in nats."""
    return cmi(x, y, None, alpha)


def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    cond: Optional[np.ndarray] = None,
    alpha: float = ALPHA,
    lag: int = 1,
) -> float:
    """Directed flow source -> target: I(target_{t+lag}; source_t | target_t, [cond_t]) in nats.

    The world is lag-1 by construction, so lag=1 is the default. ``cond`` conditions on
    additional present-time variables; conditioning on a *mediator* deliberately screens
    off flow that passes through it (see the mediation test and the Chunk-4 warning about
    not conditioning on the action when measuring the goal's push toward the environment).
    """
    source = np.asarray(source)
    target = np.asarray(target)
    y_next = target[lag:]
    x_prev = source[:-lag]
    y_prev = target[:-lag]
    cond_prev = None if cond is None else np.asarray(cond)[:-lag]
    z = _stack(y_prev, cond_prev)
    return cmi(y_next, x_prev, z, alpha)


# ---------- conservation audit (the built-in self-check) ----------

def _seq_labels(seq: np.ndarray, upto: int) -> Optional[np.ndarray]:
    """Labels for columns 0..upto-1 of a [N, n] sequence array; None if upto == 0."""
    if upto <= 0:
        return None
    return seq[:, :upto]


def directed_information(x_seq: np.ndarray, y_seq: np.ndarray, alpha: float = ALPHA) -> float:
    """Massey directed information I(X^n -> Y^n) = sum_t I(X^{1:t}; Y_t | Y^{1:t-1}), in nats.

    ``x_seq`` and ``y_seq`` are [N_realizations, n] arrays of independent length-n sequences.
    """
    N, n = y_seq.shape
    total = 0.0
    for t in range(n):
        x_hist = x_seq[:, : t + 1]  # X^{1:t+1}
        y_t = y_seq[:, t]
        y_hist = _seq_labels(y_seq, t)  # Y^{1:t}
        total += cmi(y_t, x_hist, y_hist, alpha)
    return total


def total_sequence_mi(x_seq: np.ndarray, y_seq: np.ndarray, alpha: float = ALPHA) -> float:
    """I(X^n; Y^n) treating each whole sequence as one joint symbol, in nats."""
    return mi(x_seq, y_seq, alpha)


def conservation_check(x_seq: np.ndarray, y_seq: np.ndarray, alpha: float = ALPHA) -> dict:
    """Marko-Massey conservation: I(X^n; Y^n) = I(X^n -> Y^n) + I(Y^{n-1} -> X^n).

    The reverse term is delayed by one step: I(Y^{n-1} -> X^n) = sum_t I(Y^{1:t-1}; X_t | X^{1:t-1}).
    Returns the three quantities and the residual, so measurements carry a built-in audit.
    """
    N, n = x_seq.shape
    fwd = directed_information(x_seq, y_seq, alpha)

    # Delayed reverse: at step t condition X_t on X history, using Y history up to t-1.
    rev = 0.0
    for t in range(n):
        y_hist = _seq_labels(y_seq, t)  # Y^{1:t}  (i.e. up to t-1 in 1-indexed terms)
        if y_hist is None:
            continue
        x_t = x_seq[:, t]
        x_hist = _seq_labels(x_seq, t)  # X^{1:t}
        rev += cmi(x_t, y_hist, x_hist, alpha)

    total = total_sequence_mi(x_seq, y_seq, alpha)
    return {
        "total_mi": total,
        "forward_di": fwd,
        "reverse_di_delayed": rev,
        "sum_directed": fwd + rev,
        "residual": total - (fwd + rev),
    }


def binary_entropy(p: float) -> float:
    """H_b(p) in nats."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * log(p) - (1 - p) * log(1 - p)
