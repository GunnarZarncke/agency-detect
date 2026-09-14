"""Level-1 tests: the estimator against cases whose answer is known by hand.

Every target here is analytic, in nats (natural log). One bit = ln 2 = 0.6931 nats.
If any of these fail, nothing measured downstream can be trusted.
"""

from __future__ import annotations

from math import log

import numpy as np

from value_detect.directed import (
    binary_entropy,
    conservation_check,
    mi,
    transfer_entropy,
)

LN2 = log(2.0)


# ---------- analytic process generators (fixed seeds) ----------

def _delayed_copy(seed: int, T: int, p: float = 0.0):
    """X iid fair; Y_{t+1} = X_t XOR Bern(p). Returns (X, Y)."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, T)
    Y = np.empty(T, dtype=int)
    Y[0] = rng.integers(0, 2)
    noise = (rng.random(T - 1) < p).astype(int)
    Y[1:] = X[:-1] ^ noise
    return X, Y


def _independent_pair(seed: int, T: int):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, T), rng.integers(0, 2, T)


# ---------- tests ----------

def test_delayed_copy_is_one_bit_forward_zero_back():
    X, Y = _delayed_copy(seed=1, T=50000, p=0.0)
    forward = transfer_entropy(X, Y)      # I(Y_{t+1}; X_t | Y_t) — X fully determines next Y
    backward = transfer_entropy(Y, X)     # X is iid — nothing flows back
    assert abs(forward - LN2) < 0.02, forward
    assert backward < 0.01, backward


def test_noisy_copy_matches_ln2_minus_binary_entropy():
    for p in (0.05, 0.15, 0.30):
        X, Y = _delayed_copy(seed=2, T=80000, p=p)
        expected = LN2 - binary_entropy(p)
        got = transfer_entropy(X, Y)
        assert abs(got - expected) < 0.02, (p, got, expected)


def test_independent_pair_is_near_zero():
    X, Y = _independent_pair(seed=3, T=50000)
    assert transfer_entropy(X, Y) < 0.01
    assert transfer_entropy(Y, X) < 0.01


def test_mediator_screens_off_the_flow():
    # X -> M (same step) -> Y_next. Measuring X's flow to Y should be one bit;
    # conditioning on the mediator M must collapse it to ~0. This is the analytic
    # version of "do not condition on the action when measuring the goal's push to E".
    rng = np.random.default_rng(4)
    T = 60000
    X = rng.integers(0, 2, T)
    M = X.copy()                      # mediator carries X within the same step
    Y = np.empty(T, dtype=int)
    Y[0] = rng.integers(0, 2)
    Y[1:] = M[:-1]                    # Y_{t+1} = M_t
    unconditioned = transfer_entropy(X, Y)
    through_mediator = transfer_entropy(X, Y, cond=M)
    assert abs(unconditioned - LN2) < 0.02, unconditioned
    assert through_mediator < 0.01, through_mediator


def test_estimator_bias_shrinks_with_sample_size():
    # On a truly independent pair the true answer is 0; the plug-in reads a small positive
    # bias that must shrink as N grows.
    bias_small = transfer_entropy(*_independent_pair(seed=5, T=2000))
    bias_large = transfer_entropy(*_independent_pair(seed=5, T=80000))
    assert bias_large < bias_small
    assert bias_large < 0.01, bias_large


# ---------- conservation audit (Marko-Massey) ----------

def _feedforward_sequences(seed: int, N: int, n: int = 3):
    """Y_t = X_{t-1}; X iid. Flow is purely X -> Y."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(N, n))
    Y = np.empty((N, n), dtype=int)
    Y[:, 0] = rng.integers(0, 2, size=N)
    Y[:, 1:] = X[:, :-1]
    return X, Y


def _feedback_sequences(seed: int, N: int, n: int = 3):
    """X_t = Y_{t-1}, Y_t = X_{t-1}: flow in both directions."""
    rng = np.random.default_rng(seed)
    X = np.empty((N, n), dtype=int)
    Y = np.empty((N, n), dtype=int)
    X[:, 0] = rng.integers(0, 2, size=N)
    Y[:, 0] = rng.integers(0, 2, size=N)
    for t in range(1, n):
        X[:, t] = Y[:, t - 1]
        Y[:, t] = X[:, t - 1]
    return X, Y


def test_conservation_feedforward():
    X, Y = _feedforward_sequences(seed=6, N=200000, n=3)
    r = conservation_check(X, Y, alpha=1e-6)
    assert abs(r["residual"]) < 0.03, r
    assert r["reverse_di_delayed"] < 0.03, r          # nothing flows back
    assert r["forward_di"] > 0.5, r                    # X genuinely drives Y


def test_conservation_feedback():
    X, Y = _feedback_sequences(seed=7, N=200000, n=3)
    r = conservation_check(X, Y, alpha=1e-6)
    assert abs(r["residual"]) < 0.03, r
    assert r["forward_di"] > 0.3, r
    assert r["reverse_di_delayed"] > 0.3, r            # both directions carry flow
