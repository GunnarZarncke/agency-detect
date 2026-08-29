"""V2-4 worlds (docs/V2_4_WORLD_SPECS.md; all ours, modelled on the handle-world's
mechanics; Gunnar's repo untouched). Every generator is fixed-seed deterministic.

* :func:`colony_frame` — 8 agents × {B,S,A,E,G,D} + global W (~49 vars). Weak ring
  coupling by default (SJ 2026-08-11): with probability ``coupling`` per step, agent i's
  environment patch is also flipped by agent (i−1)'s action line. ``coupling=0`` gives
  the partitioned diagnostic variant.
* :func:`deep_synergy_frame` — two isolated agents, each with two belief channels:
  channel 1 is action-driven, channel 2 is EXOGENOUS ("weather" — fixed 2026-08-12:
  driving both channels with the same action made E1⊕E2 quasi-constant, silently
  collapsing the three-way parity; an exogenous second channel restores it). Agent P
  composes its action by parity (B1 ⊕ B2 ⊕ G), agent M by majority. ~17 vars.
* :func:`slow_meter_frame` — the V1 loop with a slowed goal (0.5%) plus four derived
  meter witnesses at different timescales (fast integrator, 8-streak, 32-streak,
  saturating near-frozen).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from uad_handles.minimal import NO_OP, VAR_NAMES, SyntheticHandleWorld

from .world import BURN_IN

NOISE = {"sensor": 0.05, "belief": 0.03, "action": 0.04, "env": 0.03,
         "obs": 0.06, "distractor": 0.06, "goal_flip": 0.015}


def colony_frame(seed: int = 0, n_steps: int = 20000, n_agents: int = 8,
                 coupling: float = 0.1, disconnect_goals: bool = False,
                 puppet=None) -> pd.DataFrame:
    """Ring of handle-world-style agents; agent i's patch weakly nudged by agent i−1.

    ``puppet=(mode, idx)`` replaces agent idx's goal-slot update with a CAPTURED goal
    (V3): still wired into the action (a_line = B ⊕ G as normal) but environment-driven —
    mode "fast": G(t+1) = own-patch E(t) ⊕ 3%; mode "slow": G flips only after its patch
    holds one value for 6 consecutive steps (~a true goal's tempo, an environmental
    trigger's ancestry). Column names stay uniform (G0..G7): the impostor must not be
    identifiable by name.
    """
    rng = np.random.default_rng(seed)
    T = n_steps + BURN_IN
    bern = lambda p, size=None: np.asarray(rng.random(size) < p, dtype=np.int8)

    B = np.empty((T, n_agents), dtype=np.int8)
    E = np.empty((T, n_agents), dtype=np.int8)
    G = np.empty((T, n_agents), dtype=np.int8)
    S = np.empty((T, n_agents), dtype=np.int8)
    A = np.empty((T, n_agents), dtype=np.int8)
    D = np.empty((T, n_agents), dtype=np.int8)
    W = np.empty(T, dtype=np.int8)
    B[0] = bern(0.5, n_agents); E[0] = bern(0.5, n_agents); G[0] = bern(0.5, n_agents)
    D[0] = bern(0.5, n_agents); W[0] = bern(0.5)
    p_mode, p_idx = (puppet if puppet else (None, -1))
    run_len = 0  # slow-puppet trigger counter (runs of same value on its patch)

    for t in range(T):
        s_line = E[t] ^ bern(NOISE["sensor"], n_agents)
        if disconnect_goals:  # per-world no-core control: goals exist but steer nothing
            a_line = B[t] ^ bern(NOISE["action"], n_agents)
        else:
            a_line = (B[t] ^ G[t]) ^ bern(NOISE["action"], n_agents)
        S[t] = s_line ^ bern(NOISE["obs"], n_agents)
        A[t] = a_line ^ bern(NOISE["obs"], n_agents)
        if t + 1 < T:
            nudge = np.roll(a_line, 1) & bern(coupling, n_agents)  # neighbour i−1
            E[t + 1] = E[t] ^ a_line ^ nudge ^ bern(NOISE["env"], n_agents)
            B[t + 1] = s_line ^ bern(NOISE["belief"], n_agents)
            G[t + 1] = G[t] ^ bern(NOISE["goal_flip"], n_agents)
            if p_mode == "fast":
                G[t + 1, p_idx] = E[t, p_idx] ^ bern(0.03)
            elif p_mode == "slow":
                if t > 0 and E[t, p_idx] == E[t - 1, p_idx]:
                    run_len += 1
                else:
                    run_len = 0
                if run_len >= 6:
                    G[t + 1, p_idx] = 1 - G[t, p_idx]
                    run_len = 0
                else:
                    G[t + 1, p_idx] = G[t, p_idx]
            D[t + 1] = E[t + 1] ^ bern(NOISE["distractor"], n_agents)
            W[t + 1] = bern(0.5)

    cols: Dict[str, np.ndarray] = {}
    for i in range(n_agents):
        for name, arr in (("B", B), ("S", S), ("A", A), ("E", E), ("G", G), ("D", D)):
            cols[f"{name}{i}"] = arr[:, i]
    cols["W"] = W
    return pd.DataFrame(cols).iloc[BURN_IN:].reset_index(drop=True).astype(int)


def _agent_two_channel(rng, T: int, rule: str, disconnect_goal: bool = False):
    """One deep-synergy agent: two env channels driven by its action; two sensors; two
    beliefs; action composed by ``rule`` ('parity' | 'majority') from (B1, B2, G)."""
    bern = lambda p: int(rng.random() < p)
    E1 = np.empty(T, dtype=np.int8); E2 = np.empty(T, dtype=np.int8)
    S1 = np.empty(T, dtype=np.int8); S2 = np.empty(T, dtype=np.int8)
    B1 = np.empty(T, dtype=np.int8); B2 = np.empty(T, dtype=np.int8)
    A = np.empty(T, dtype=np.int8); G = np.empty(T, dtype=np.int8)
    E1[0], E2[0], B1[0], B2[0], G[0] = (rng.integers(0, 2) for _ in range(5))
    for t in range(T):
        s1_line = E1[t] ^ bern(NOISE["sensor"])
        s2_line = E2[t] ^ bern(NOISE["sensor"])
        S1[t] = s1_line ^ bern(NOISE["obs"])
        S2[t] = s2_line ^ bern(NOISE["obs"])
        if disconnect_goal:  # no-core: the goal column persists but leaves the rule
            a = (B1[t] ^ B2[t]) if rule == "parity" else int(int(B1[t]) + int(B2[t]) >= 1)
        elif rule == "parity":
            a = B1[t] ^ B2[t] ^ G[t]
        else:
            a = 1 if (int(B1[t]) + int(B2[t]) + int(G[t])) >= 2 else 0
        a_line = a ^ bern(NOISE["action"])
        A[t] = a_line ^ bern(NOISE["obs"])
        if t + 1 < T:
            E1[t + 1] = E1[t] ^ a_line ^ bern(NOISE["env"])
            E2[t + 1] = bern(0.5)  # exogenous weather channel (independent of the agent)
            B1[t + 1] = s1_line ^ bern(NOISE["belief"])
            B2[t + 1] = s2_line ^ bern(NOISE["belief"])
            G[t + 1] = G[t] ^ bern(NOISE["goal_flip"])
    return {"E1": E1, "E2": E2, "S1": S1, "S2": S2, "B1": B1, "B2": B2, "A": A, "G": G}


def deep_synergy_frame(seed: int = 0, n_steps: int = 20000, disconnect_goals: bool = False) -> pd.DataFrame:
    """Agent P (parity) + agent M (majority), isolated from each other, + global W."""
    rng = np.random.default_rng(seed)
    T = n_steps + BURN_IN
    P = _agent_two_channel(rng, T, "parity", disconnect_goals)
    M = _agent_two_channel(rng, T, "majority", disconnect_goals)
    cols = {f"{k}_P": v for k, v in P.items()}
    cols.update({f"{k}_M": v for k, v in M.items()})
    cols["W"] = (np.random.default_rng(seed + 77).random(T) < 0.5).astype(np.int8)
    return pd.DataFrame(cols).iloc[BURN_IN:].reset_index(drop=True).astype(int)


def slow_meter_frame(seed: int = 0, n_steps: int = 20000, flip_rate: float = 0.005,
                     disconnect_goal: bool = False) -> pd.DataFrame:
    """V1 loop (goal slowed to ``flip_rate``) + four derived meter witnesses."""
    from .variants import NoCoreHandleWorld
    cls = NoCoreHandleWorld if disconnect_goal else SyntheticHandleWorld
    world = cls(seed=seed, goal_flip_rate=flip_rate)
    world.rollout(BURN_IN, [(NO_OP, -1)])
    data = world.rollout(n_steps, [(NO_OP, -1)])
    frame = pd.DataFrame(np.asarray(data.x), columns=list(VAR_NAMES)).astype(int)
    A = frame["A"].to_numpy()
    E = frame["E"].to_numpy()
    n = len(frame)

    m_fast = np.empty(n, dtype=int)
    m_fast[0] = 3
    for t in range(1, n):
        m_fast[t] = min(6, max(0, m_fast[t - 1] + (1 if A[t - 1] == 1 else -1)))
    frame["M_fast"] = m_fast

    m = np.empty(n, dtype=int)
    m[0] = 3
    run1 = run0 = 0
    for t in range(1, n):
        if A[t - 1] == 1:
            run1 += 1; run0 = 0
        else:
            run0 += 1; run1 = 0
        step = 1 if run1 >= 8 else (-1 if run0 >= 8 else 0)
        if step:
            run1 = run0 = 0
        m[t] = min(6, max(0, m[t - 1] + step))
    frame["M8"] = m

    # M32: windowed-majority stepper — every 32 steps, move only on a strong majority
    # (≥20/32 of the window one way). A same-action 32-streak of a fair coin virtually
    # never occurs, so a streak design would be frozen; this one is slow but alive.
    m = np.empty(n, dtype=int)
    m[0] = 3
    for t in range(1, n):
        step = 0
        if t % 32 == 0 and t >= 32:
            frac = A[t - 32:t].mean()
            if frac >= 20 / 32:
                step = 1
            elif frac <= 12 / 32:
                step = -1
        m[t] = min(6, max(0, m[t - 1] + step))
    frame["M32"] = m

    f = np.zeros(n, dtype=int)
    run_e = level = 0
    for t in range(n):
        run_e = run_e + 1 if E[t] == 1 else 0
        if run_e >= 12 and level < 3:
            level += 1
            run_e = 0
        f[t] = level
    frame["F_sat"] = f
    return frame


GROUND_TRUTH = {
    "colony": {"values": [f"G{i}" for i in range(8)], "meters": [], "noise": ["W"]},
    "deep_synergy": {"values": ["G_P", "G_M"], "meters": [], "noise": ["W"]},
    "slow_meter": {"values": ["G"], "meters": ["M_fast", "M8", "M32", "F_sat"], "noise": ["W"]},
}


def alias_colony_frame(seed: int = 0, n_steps: int = 20000, alias_noise: float = 0.01,
                       alias_agents=(2, 5), **colony_kwargs) -> pd.DataFrame:
    """V3 alias-colony: the V2 colony plus causally inert goal twins for two agents.

    G{i}_alias = G{i} ⊕ Bern(alias_noise), appended post hoc (dynamics untouched — the
    twin is a recording; the base columns are bit-identical to the plain colony at the
    same seed)."""
    frame = colony_frame(seed=seed, n_steps=n_steps, **colony_kwargs)
    rng = np.random.default_rng(seed * 7919 + int(round(alias_noise * 10000)) + 17)
    for i in alias_agents:
        g = frame[f"G{i}"].to_numpy()
        frame[f"G{i}_alias"] = g ^ (rng.random(len(g)) < alias_noise).astype(int)
    return frame
