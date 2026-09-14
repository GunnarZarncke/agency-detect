"""Narrate a passive trace in plain English so SJ can audit the world without code.

For each timestep we tell the loop as a short story and mark whether each designed
relationship held (a tick) or was nudged by the world's built-in noise (a note). We
also single out every goal-flip: the rare, internally-driven events that make G the
planted value-core.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def _tick(ok: bool, noise_label: str) -> str:
    return "ok" if ok else f"noisy ({noise_label})"


def narrate_window(frame: pd.DataFrame, start: int = 0, n: int = 25) -> str:
    """Narrate transitions t -> t+1 for a window of the trace."""
    B = frame["B"].to_numpy()
    S = frame["S"].to_numpy()
    A = frame["A"].to_numpy()
    E = frame["E"].to_numpy()
    G = frame["G"].to_numpy()
    Sa = frame["S_alias"].to_numpy()
    Aa = frame["A_alias"].to_numpy()

    end = min(start + n, len(frame) - 1)
    out: List[str] = []
    out.append(f"Story of steps {start} to {end} (each block is one tick of the loop):\n")

    for t in range(start, end):
        goal_flipped = G[t + 1] != G[t]
        expected_action = B[t] ^ G[t]
        action_ok = A[t] == expected_action
        env_expected = E[t] ^ Aa[t]
        env_ok = E[t + 1] == env_expected
        belief_ok = B[t + 1] == Sa[t]

        block = [f"Step {t} -> {t + 1}"]
        if goal_flipped:
            block.append(f"  GOAL:   flipped {G[t]} -> {G[t + 1]} entirely on its own (internal coin). *** value-core event ***")
        else:
            block.append(f"  Goal:   held at {G[t]} (its internal coin did not flip).")
        block.append(
            f"  Decide: action A={A[t]}  (belief {B[t]} + goal {G[t]} -> expected {expected_action}) -- {_tick(action_ok, 'action readout noise')}"
        )
        block.append(
            f"  Act:    environment {E[t]} -> {E[t + 1]}  (action line {Aa[t]} pushes it toward {env_expected}) -- {_tick(env_ok, 'environment noise')}"
        )
        block.append(
            f"  Sense:  sensor read environment={E[t]} (clean copy S_alias={Sa[t]}, noisy readout S={S[t]})."
        )
        block.append(
            f"  Update: belief {B[t]} -> {B[t + 1]}  (tracks the sensed environment, expected {Sa[t]}) -- {_tick(belief_ok, 'belief noise')}"
        )
        out.append("\n".join(block))

    return "\n\n".join(out)


def goal_flip_report(frame: pd.DataFrame, max_shown: int = 8) -> str:
    """List the goal-flip events: how often, and what the action did around each one."""
    G = frame["G"].to_numpy()
    A = frame["A"].to_numpy()
    B = frame["B"].to_numpy()
    flips = np.where(G[1:] != G[:-1])[0]  # index t means G changed between t and t+1

    n = len(frame)
    rate = len(flips) / max(n - 1, 1)
    lines: List[str] = []
    lines.append(
        f"Goal-flip events: {len(flips)} in {n} steps ({rate * 100:.2f}% of steps; "
        f"the world's designed rate is 1.50%)."
    )
    lines.append(
        "These are the value-core's defining moments: the goal changes only from its own "
        "internal coin, never because the environment pushed it."
    )
    if len(flips) == 0:
        return "\n".join(lines)

    lines.append("")
    for t in flips[:max_shown]:
        after = f"{B[t + 1] ^ G[t + 1]}" if t + 1 < n else "n/a"
        lines.append(
            f"  * step {t}->{t + 1}: goal {G[t]}->{G[t + 1]}; "
            f"the decision now combines the new goal with belief {B[t + 1]} "
            f"(belief XOR goal -> {after})."
        )
    if len(flips) > max_shown:
        lines.append(f"  ... and {len(flips) - max_shown} more.")
    return "\n".join(lines)
