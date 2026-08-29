"""V3 — the goal-flip interventional yardstick (docs/V3_PLAN.md).

The two-direction interventional mirror of the value signature, do-style: on operated
steps (50/50 interleaved, Gunnar's pattern) the candidate's value AS CONSUMED BY THE
DYNAMICS is inverted. Calibration only — never blended into the passive signature.

* Goal-role probe: does flipping X change its agent's behaviour GIVEN the agent's
  policy context? score = I(flag ; A_i(t) | B_i(t), G_i(t)) + I(flag ; A_i(t+1) |
  A_i(t), B_i(t), G_i(t)) — the conditioning mirrors Gunnar's own interventional score
  and is REQUIRED: the flag XORs into the action alongside belief-and-goal, so the
  marginal effect is cipher-masked (the V1 lesson, resurfacing under intervention).
  Term 1 captures within-tick wiring (goal-slots, beliefs); term 2 one-step paths
  (patches). Recordings (S, A, D, W readouts; alias columns) are consumed by nothing,
  so their role-effect is honestly ~0.
* Autonomy probe: does flipping X's own patch change X soon after? Two-scale readout,
  BOTH terms conditioned on the full agent context [X(t), B_i(t), G_i(t), E_i(t)] —
  required for the same reason as the role probe: X's next value XORs its drivers with
  the flag, so the unconditioned effect is cipher-masked (third appearance of the V1
  lesson). Term 1, I(flag ; X(t+1) | ctx), catches fast candidates; term 2,
  I(flag ; X-changes-within-6 | ctx), catches slow ones (a trigger's effect spreads).
  Floors mirror the sum.
* Interventional verdict = role-effect above floor AND autonomy-effect at floor
  ("levers the world, unlevered by the world"). True goal: yes/no. Captured puppet:
  yes/yes → refused. Alias: no/– → refused.

Floors: permutation of the flag column (the operation schedule is exchangeable),
p95 + z=3, procedure-mirroring as everywhere else.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .directed import _cmi_labeled, _labels
from .worlds_v2 import NOISE


def operated_colony_rollout(seed: int, n_steps: int, target: str, direction: str,
                            n_agents: int = 8, coupling: float = 0.1,
                            puppet: Optional[Tuple[str, int]] = None,
                            alias_noise: Optional[float] = None,
                            alias_agents=(2, 5)) -> Dict[str, np.ndarray]:
    """Colony rollout with do-style flips of ``target`` on ~50% of steps.

    direction "role": the target's value as consumed this step is inverted (goal-slots
    enter the action; patches enter sensing/updates; recordings enter nothing).
    direction "autonomy": the target's own patch value as consumed is inverted, and the
    target itself is left alone. Returns the flag plus the columns needed for scoring.
    """
    rng = np.random.default_rng(seed)
    op_rng = np.random.default_rng(seed * 31 + 7)
    bern = lambda p, size=None: np.asarray(rng.random(size) < p, dtype=np.int8)

    kind, idx = _parse_target(target)
    B = bern(0.5, n_agents).astype(np.int8)
    E = bern(0.5, n_agents).astype(np.int8)
    G = bern(0.5, n_agents).astype(np.int8)
    run_len = 0
    p_mode, p_idx = (puppet if puppet else (None, -1))
    agent = idx if idx is not None else 0

    flags = np.empty(n_steps, dtype=np.int8)
    A_now = np.empty(n_steps, dtype=np.int8)
    B_now = np.empty(n_steps, dtype=np.int8)
    G_now = np.empty(n_steps, dtype=np.int8)
    E_now = np.empty(n_steps, dtype=np.int8)
    X_val = np.empty(n_steps, dtype=np.int8)
    alias_state = {}
    if alias_noise is not None:
        arng = np.random.default_rng(seed * 7919 + int(round(alias_noise * 10000)) + 17)

    for t in range(n_steps):
        flag = int(op_rng.random() < 0.5)
        flags[t] = flag

        b_used, e_used, g_used = B.copy(), E.copy(), G.copy()
        if flag:
            if direction == "role":
                if kind == "G":
                    g_used[idx] ^= 1
                elif kind == "B":
                    b_used[idx] ^= 1
                elif kind == "E":
                    e_used[idx] ^= 1
                # readouts/aliases/W: consumed by nothing — no flip lands anywhere.
            else:  # autonomy: perturb the target's own patch
                e_used[agent] ^= 1

        s_line = e_used ^ bern(NOISE["sensor"], n_agents)
        a_line = (b_used ^ g_used) ^ bern(NOISE["action"], n_agents)
        A_now[t] = a_line[agent] ^ bern(NOISE["obs"])
        B_now[t] = B[agent]
        G_now[t] = G[agent]
        E_now[t] = E[agent]

        # record the candidate's own value this step (for autonomy scoring)
        if kind == "G":
            X_val[t] = G[idx]
        elif kind == "B":
            X_val[t] = B[idx]
        elif kind == "E":
            X_val[t] = E[idx]
        elif kind == "S":
            X_val[t] = s_line[idx] ^ bern(NOISE["obs"])
        elif kind == "A":
            X_val[t] = A_now[t] if idx == agent else (a_line[idx] ^ bern(NOISE["obs"]))
        elif kind == "D":
            X_val[t] = E[idx] ^ bern(NOISE["distractor"])
        elif kind == "W":
            X_val[t] = bern(0.5)
        elif kind == "ALIAS":
            X_val[t] = G[idx] ^ (1 if (alias_noise and arng.random() < alias_noise) else 0)

        nudge = np.roll(a_line, 1) & bern(coupling, n_agents)
        E_next = e_used ^ a_line ^ nudge ^ bern(NOISE["env"], n_agents)
        B_next = s_line ^ bern(NOISE["belief"], n_agents)
        G_next = G ^ bern(NOISE["goal_flip"], n_agents)
        if p_mode == "fast":
            G_next[p_idx] = e_used[p_idx] ^ bern(0.03)
        elif p_mode == "slow":
            if t > 0 and E[p_idx] == E_prev_val:
                run_len += 1
            else:
                run_len = 0
            if run_len >= 6:
                G_next[p_idx] = 1 - G[p_idx]
                run_len = 0
            else:
                G_next[p_idx] = G[p_idx]
        E_prev_val = E[p_idx] if p_idx >= 0 else 0
        B, E, G = B_next, E_next, G_next

    return {"flag": flags, "A_now": A_now, "B_now": B_now, "G_now": G_now,
            "E_now": E_now, "X": X_val}


def _parse_target(target: str):
    if target.endswith("_alias"):
        return "ALIAS", int(target[1:target.index("_")])
    if target == "W":
        return "W", None
    return target[0], int(target[1:])


def yardstick_scores(seed: int, target: str, n_steps: int = 2000, n_perms: int = 500,
                     window: int = 6, **world_kwargs) -> Dict[str, float]:
    """Both probes for one candidate: scores, permutation floors (p95, mean, sd)."""
    out: Dict[str, float] = {}
    rng = np.random.default_rng(seed * 613 + 29)

    # role probe (context-conditioned; see module docstring)
    r = operated_colony_rollout(seed, n_steps, target, "role", **world_kwargs)
    ctx = _labels(np.column_stack([r["B_now"][:-1], r["G_now"][:-1], r["E_now"][:-1]]))
    ctx2 = _labels(np.column_stack([r["B_now"][:-1], r["G_now"][:-1], r["E_now"][:-1],
                                    r["A_now"][:-1]]))
    a_now, a_next = r["A_now"][:-1].astype(np.int64), r["A_now"][1:].astype(np.int64)

    def role_score(flag):
        fl = _labels(flag)
        return (_cmi_labeled(a_now, fl, ctx) + _cmi_labeled(a_next, fl, ctx2))

    score = role_score(r["flag"][:-1])
    nulls = np.array([role_score(rng.permutation(r["flag"][:-1])) for _ in range(n_perms)])
    out.update(role=score, role_p95=float(np.percentile(nulls, 95)),
               role_mean=float(nulls.mean()), role_sd=float(nulls.std()))

    # autonomy probe (skipped for the global noise variable — no own patch)
    if target != "W":
        a = operated_colony_rollout(seed, n_steps, target, "autonomy", **world_kwargs)
        x = a["X"].astype(np.int64)
        changed = np.array([int(np.any(x[t + 1:t + 1 + window] != x[t]))
                            for t in range(len(x) - window)], dtype=np.int64)
        m = len(changed)
        flag = a["flag"][:m]
        xctx = _labels(np.column_stack([x[:m], a["B_now"][:m], a["G_now"][:m], a["E_now"][:m]]))
        x_next = x[1:m + 1]

        def auto_score(fl):
            fll = _labels(fl)
            return (_cmi_labeled(x_next, fll, xctx)
                    + _cmi_labeled(changed, fll, xctx))

        score = auto_score(flag)
        nulls = np.array([auto_score(rng.permutation(flag)) for _ in range(n_perms)])
        out.update(autonomy=score, autonomy_p95=float(np.percentile(nulls, 95)),
                   autonomy_mean=float(nulls.mean()), autonomy_sd=float(nulls.std()))
    else:
        out.update(autonomy=float("nan"), autonomy_p95=float("nan"),
                   autonomy_mean=float("nan"), autonomy_sd=float("nan"))
    return out


def yardstick_verdict(s: Dict[str, float], z: float = 3.0) -> Dict[str, object]:
    role_above = (s["role"] > s["role_p95"]
                  and s["role_sd"] > 0 and (s["role"] - s["role_mean"]) / s["role_sd"] >= z)
    if np.isfinite(s.get("autonomy", float("nan"))):
        auto_above = (s["autonomy"] > s["autonomy_p95"]
                      and s["autonomy_sd"] > 0
                      and (s["autonomy"] - s["autonomy_mean"]) / s["autonomy_sd"] >= z)
    else:
        auto_above = False
    return {"role_effect": bool(role_above), "autonomy_effect": bool(auto_above),
            "interventional_value": bool(role_above and not auto_above)}
