"""Sanity tests for the directional scorers on tiny systems with known structure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from value_detect.scorer import (
    drop_aliases,
    push_in_megastate,
    score_trace,
    score_trace_fused,
    score_trace_fused_bestkey,
)


def _chain(seed: int, T: int) -> pd.DataFrame:
    """X iid; Y copies X one step later (10% noise); Z copies Y one step later."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, T)
    Y = np.empty(T, dtype=int)
    Z = np.empty(T, dtype=int)
    Y[0] = Z[0] = 0
    Y[1:] = X[:-1] ^ (rng.random(T - 1) < 0.1).astype(int)
    Z[1:] = Y[:-1] ^ (rng.random(T - 1) < 0.1).astype(int)
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


def _rare_flipper(seed: int, T: int) -> pd.DataFrame:
    """R flips ~1.5% of steps by its own coin; F noisily copies R one step later."""
    rng = np.random.default_rng(seed)
    R = np.empty(T, dtype=int)
    R[0] = 0
    flips = (rng.random(T - 1) < 0.015).astype(int)
    for t in range(1, T):
        R[t] = R[t - 1] ^ flips[t - 1]
    F = np.empty(T, dtype=int)
    F[0] = 0
    F[1:] = R[:-1] ^ (rng.random(T - 1) < 0.1).astype(int)
    return pd.DataFrame({"R": R, "F": F})


def test_chain_polarities_are_ordered():
    t = score_trace(_chain(1, 20000), env_var="Z")
    # The pure driver X leads the ranking; the pure absorber Z trails it.
    assert list(t.index)[0] == "X"
    assert list(t.index)[-1] == "Z"
    assert t.loc["X", "polarity_sys"] > 0.5
    assert t.loc["Z", "polarity_sys"] < -0.5
    # X is driven by nothing: intake ≈ estimator bias only.
    assert t.loc["X", "push_in"] < 0.01
    # Mediator sits between the poles.
    assert t.loc["Z", "polarity_sys"] < t.loc["Y", "polarity_sys"] < t.loc["X", "polarity_sys"]


def test_rare_flipper_is_all_drive():
    t = score_trace(_rare_flipper(2, 20000), env_var="F")
    # R's flows are small in absolute terms (it rarely changes) but almost purely outbound.
    assert t.loc["R", "polarity_sys"] > 0.7
    assert t.loc["F", "polarity_sys"] < -0.7
    assert t.loc["R", "push_in"] < 0.01
    # env flavour for R exists and is positive; NaN for the environment variable itself.
    assert t.loc["R", "out_env"] > 0.05
    assert np.isnan(t.loc["F", "out_env"])


def test_megastate_crosscheck_tracks_pairwise():
    frame = _chain(3, 20000)
    pair = score_trace(frame, env_var="Z")["push_in"]
    mega = push_in_megastate(frame)
    # Same qualitative story: X near zero, Y and Z clearly positive.
    assert mega["X"] < 0.02
    assert mega["Y"] > 0.1 and mega["Z"] > 0.1
    assert pair["X"] < pair["Y"] and pair["X"] < pair["Z"]


def test_drop_aliases():
    frame = pd.DataFrame({"B": [0, 1], "S_alias": [1, 0], "A_alias": [0, 0], "W": [1, 1]})
    out = drop_aliases(frame)
    assert list(out.columns) == ["B", "W"]


def _xor_loop(seed: int, T: int) -> pd.DataFrame:
    """Minimal cipher loop: A(t+1) = B(t) XOR G(t); E driven by A; B tracks E; G rare-flips."""
    rng = np.random.default_rng(seed)
    E = np.empty(T, dtype=int)
    B = np.empty(T, dtype=int)
    G = np.empty(T, dtype=int)
    A = np.empty(T, dtype=int)
    E[0], B[0], G[0], A[0] = rng.integers(0, 2, 4)
    for t in range(T - 1):
        A[t + 1] = B[t] ^ G[t] ^ (rng.random() < 0.04)
        E[t + 1] = E[t] ^ A[t] ^ (rng.random() < 0.03)
        B[t + 1] = E[t] ^ (rng.random() < 0.08)
        G[t + 1] = G[t] ^ (rng.random() < 0.015)
    return pd.DataFrame({"B": B, "G": G, "A": A, "E": E})


def test_conventions_agree_when_there_is_no_cipher():
    # On the plain chain, the two conventions must tell the same story
    # (fused+best-key may read slightly higher; ordering identical).
    frame = _chain(4, 20000)
    pair = score_trace(frame, env_var="Z")
    ciph = score_trace_fused_bestkey(frame, env_var="Z")
    assert list(pair.index) == list(ciph.index)
    for v in frame.columns:
        assert abs(pair.loc[v, "polarity_sys"] - ciph.loc[v, "polarity_sys"]) < 0.25


def test_fused_bestkey_sees_through_the_xor():
    frame = _xor_loop(5, 20000)
    pair = score_trace(frame, env_var="E")
    ciph = score_trace_fused_bestkey(frame, env_var="E")
    # Pairwise is blind to A's intake; the fused mega-state reads it near its true size (~0.5).
    assert pair.loc["A", "push_in"] < 0.02
    assert ciph.loc["A", "push_in"] > 0.35
    # Pairwise is blind to G's grip on the action; best-key decrypts it (key: B).
    assert pair.loc["G", "out_sys"] < 0.02
    assert ciph.loc["G", "out_sys"] > 0.4
    # In THIS test world the decision lags a tick, so G's influence on E needs two hops
    # (G -> next A -> next E): at horizon 1 there is truly nothing to read, and both
    # conventions correctly report ~0 (in Gunnar's within-tick world it appears at lag 1).
    assert ciph.loc["G", "out_env"] < 0.01
    # G keeps a near-zero intake under both conventions (nothing drives it).
    assert ciph.loc["G", "push_in"] < 0.02
    # With the cipher seen through, A reads as a mediator, not a pure source.
    assert ciph.loc["A", "polarity_sys"] < pair.loc["A", "polarity_sys"]


def test_fused_convention_on_the_xor_loop():
    # 4 variables -> the fused Rest has only 8 states, so 20k steps is ample here.
    frame = _xor_loop(6, 20000)
    fused = score_trace_fused(frame, env_var="E")
    # The fused intake sees the action's true inputs (identical measure to the megastate).
    assert fused.loc["A", "push_in"] > 0.35
    # The fused outbound sees G's grip on the action (B's past is inside the fused
    # conditioning, so the cipher is broken automatically).
    assert fused.loc["G", "out_sys"] > 0.3
    # G intake stays at floor; G lands at the drive pole under this convention too.
    assert fused.loc["G", "push_in"] < 0.02
    assert list(fused.index)[0] == "G"
    # Mediator screening: E_next is fully explained by (E, A)'s past inside the fused
    # conditioning, so B gets no outbound credit for the E channel here — B's fused
    # outbound must be far below its decrypted best-key reading.
    bk = score_trace_fused_bestkey(frame, env_var="E")
    assert fused.loc["B", "out_sys"] < bk.loc["B", "out_sys"]


def test_fused_agrees_with_pairwise_on_the_chain():
    frame = _chain(7, 20000)
    pair = score_trace(frame, env_var="Z")
    fused = score_trace_fused(frame, env_var="Z")
    # No ciphers, no duplicates: same ordering, same poles.
    assert list(pair.index) == list(fused.index)
    assert fused.loc["X", "polarity_sys"] > 0.5
    assert fused.loc["Z", "polarity_sys"] < -0.5


def _xor3_loop(seed: int, T: int) -> pd.DataFrame:
    """Deep-synergy preview (two-beliefs design): A(t+1) = B1(t) XOR B2(t) XOR G(t).

    B1, B2 are independent fast coins (two separate belief streams), G rare-flips, and
    the environment integrates the action. Decrypting G's grip on A needs BOTH beliefs
    as keys — one key leaves a residual XOR mask, so any single-key convention is blind.
    """
    rng = np.random.default_rng(seed)
    T = int(T)
    B1 = rng.integers(0, 2, T)
    B2 = rng.integers(0, 2, T)
    G = np.empty(T, dtype=int)
    A = np.empty(T, dtype=int)
    E = np.empty(T, dtype=int)
    G[0], A[0], E[0] = rng.integers(0, 2, 3)
    for t in range(T - 1):
        A[t + 1] = B1[t] ^ B2[t] ^ G[t] ^ (rng.random() < 0.04)
        E[t + 1] = E[t] ^ A[t] ^ (rng.random() < 0.03)
        G[t + 1] = G[t] ^ (rng.random() < 0.015)
    return pd.DataFrame({"B1": B1, "B2": B2, "G": G, "A": A, "E": E})


def test_pure_parity_defeats_greedy_growth_but_not_fused():
    # DISCOVERED at build time (2026-08-11): in PURE three-way parity every single key
    # yields zero marginal gain, so greedy growth faces a flat plateau and stays as
    # blind as best-key. Only the fused joint conditioning cracks pure parity. This is
    # the pre-registered expectation for the parity half of the deep-synergy world.
    from value_detect.floors import ConventionScorer
    frame = _xor3_loop(11, 20000)
    bk = ConventionScorer(frame, "fused_bestkey", env_var="E").score_all()
    gk = ConventionScorer(frame, "grown_keys", env_var="E", key_depth=2).score_all()
    fu = ConventionScorer(frame, "fused", env_var="E").score_all()
    assert bk.loc["G", "out_sys"] < 0.03
    assert gk.loc["G", "out_sys"] < 0.03          # greedy cannot climb a plateau
    assert fu.loc["G", "out_sys"] > 0.35          # joint conditioning sees it whole
    assert fu.loc["G", "push_in"] < 0.02


def _graded3_loop(seed: int, T: int) -> pd.DataFrame:
    """Majority-rule variant: A(t+1) = majority(B1, B2, G) with noise — graded
    composition leaks marginal signal through single keys, so growth can climb."""
    rng = np.random.default_rng(seed)
    B1 = rng.integers(0, 2, T)
    B2 = rng.integers(0, 2, T)
    G = np.empty(T, dtype=int)
    A = np.empty(T, dtype=int)
    E = np.empty(T, dtype=int)
    G[0], A[0], E[0] = rng.integers(0, 2, 3)
    for t in range(T - 1):
        a = 1 if (B1[t] + B2[t] + G[t]) >= 2 else 0
        A[t + 1] = a ^ (rng.random() < 0.04)
        E[t + 1] = E[t] ^ A[t] ^ (rng.random() < 0.03)
        G[t + 1] = G[t] ^ (rng.random() < 0.015)
    return pd.DataFrame({"B1": B1, "B2": B2, "G": G, "A": A, "E": E})


def test_grown_keys_is_the_strongest_reader_of_graded_composition():
    from value_detect.floors import ConventionScorer
    frame = _graded3_loop(13, 20000)
    bk = ConventionScorer(frame, "fused_bestkey", env_var="E").score_all()
    gk = ConventionScorer(frame, "grown_keys", env_var="E", key_depth=2).score_all()
    assert gk.loc["G", "out_sys"] > bk.loc["G", "out_sys"] + 0.05   # the second key earns its keep
    assert gk.loc["G", "out_sys"] > 0.25
    assert gk.loc["G", "push_in"] < 0.02
    assert gk.loc["G", "polarity_sys"] > 0.9


def test_grown_keys_reduces_to_bestkey_when_one_key_suffices():
    from value_detect.floors import ConventionScorer
    frame = _xor_loop(12, 20000)
    bk = ConventionScorer(frame, "fused_bestkey", env_var="E").score_all()
    gk = ConventionScorer(frame, "grown_keys", env_var="E", key_depth=2).score_all()
    # Where one key decrypts everything, growth must not lose it (>= up to small bias).
    for v in frame.columns:
        assert gk.loc[v, "out_sys"] >= bk.loc[v, "out_sys"] - 1e-9
    assert abs(gk.loc["G", "out_sys"] - bk.loc["G", "out_sys"]) < 0.1
    assert list(gk.index)[0] == "G"
