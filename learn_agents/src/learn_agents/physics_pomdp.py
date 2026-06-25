"""Partially observed physics baseline (Gymnasium CartPole).

S = velocities (cart v, pole angular v) — what a velocity-only sensor sees.
I = positions (cart x, pole angle) — simulator state not in S; drives transitions.
A = applied force direction.

Episodes are short (≤500 steps) but one continuous rollout per episode (no concatenation).
"""

from __future__ import annotations

from typing import List

import gymnasium as gym
import numpy as np

from learn_agents.external_traces import TraceColumn, _ar1, pack_trace
from learn_agents.learn_agents import SimulationResult


def _balance_action(full: np.ndarray, rng: np.random.Generator) -> int:
    """Heuristic stabilizing controller: push toward the pole's lean (closed S→A loop).

    Small exploration noise keeps the action channel from being a deterministic
    function of one sensor (so it still looks like a controller, not a copy).
    """
    signal = full[2] + 0.5 * full[3]  # theta + 0.5*theta_dot
    if rng.random() < 0.1:
        return int(rng.integers(0, 2))
    return 1 if signal > 0 else 0


def _track_action(full: np.ndarray, rng: np.random.Generator, *, theta_ref: float) -> int:
    """Track a nonzero pole angle (pursuit / setpoint tracking, not homeostasis at zero)."""
    error = full[2] - theta_ref
    signal = 1.5 * error + 0.6 * full[3]
    if rng.random() < 0.03:
        return int(rng.integers(0, 2))
    return 1 if signal > 0 else 0


def roll_cartpole_partial_obs(
    *,
    seed: int = 0,
    max_steps: int = 500,
    n_episodes: int = 1,
    policy: str = "random",
    theta_ref: float = 0.12,
    normalize: bool = False,
) -> SimulationResult:
    """One CartPole-v1 episode.

    ``policy='balance'`` — stabilizing controller (homeostatic, long episodes).
    ``policy='track'`` — tracks ``theta_ref`` (intentional, non-suppressed internal).
    ``policy='random'`` — short falling-pole baseline.
    """
    if n_episodes != 1:
        raise ValueError("use n_episodes=1 for a single continuous physics trace (no synthetic concat)")
    if policy not in ("random", "balance", "track"):
        raise ValueError(f"policy must be 'random', 'balance', or 'track', got {policy!r}")
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    s_list, i_list, a_list = [], [], []
    for _ in range(max_steps):
        full = np.asarray(obs, dtype=np.float32)
        # full = [x, x_dot, theta, theta_dot]
        s_list.append([full[1], full[3]])
        i_list.append([full[0], full[2]])
        if policy == "balance":
            action = _balance_action(full, rng)
        elif policy == "track":
            action = _track_action(full, rng, theta_ref=theta_ref)
        else:
            action = int(env.action_space.sample())
        a_list.append([float(action)])
        obs, _reward, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    env.close()
    cols = [
        TraceColumn("agent0.sensor.cart_v", 0, "sensor", np.array(s_list)[:, 0]),
        TraceColumn("agent0.sensor.pole_ang_v", 0, "sensor", np.array(s_list)[:, 1]),
        TraceColumn("agent0.internal.cart_x", 0, "internal", np.array(i_list)[:, 0]),
        TraceColumn("agent0.internal.pole_ang", 0, "internal", np.array(i_list)[:, 1]),
        TraceColumn("agent0.action.force", 0, "action", np.array(a_list)[:, 0]),
    ]
    source = f"physics_cartpole_v1_{policy}"
    if policy == "track":
        source = f"physics_cartpole_v1_track_ref{theta_ref:.2f}"
    return pack_trace(cols, num_agents=1, seed=seed, source=source, normalize=normalize)


def roll_cartpole_multi(
    *,
    seed: int = 0,
    num_agents: int = 3,
    max_steps: int = 500,
    n_decoy_env: int = 8,
) -> SimulationResult:
    """Parallel CartPole-v1 rollouts in one trace (multi-agent, non-trivial clustering).

    Each agent has its own env and RNG stream. Trace length = steps until **all** poles
    terminate (capped at ``max_steps``). Env decoys are included in the full trace for
    blanket/oracle checks; amortized training still uses agent columns only.
    """
    if num_agents < 2:
        raise ValueError("num_agents must be >= 2 for multi-agent physics")
    rng = np.random.default_rng(seed)
    envs = [gym.make("CartPole-v1") for _ in range(num_agents)]
    obs_list = []
    for a, env in enumerate(envs):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        obs_list.append(np.asarray(obs, dtype=np.float32))
    active = [True] * num_agents
    s_lists: List[List[List[float]]] = [[] for _ in range(num_agents)]
    i_lists: List[List[List[float]]] = [[] for _ in range(num_agents)]
    a_lists: List[List[float]] = [[] for _ in range(num_agents)]

    for _ in range(max_steps):
        if not any(active):
            break
        for a in range(num_agents):
            if active[a]:
                full = obs_list[a]
                s_lists[a].append([full[1], full[3]])
                i_lists[a].append([full[0], full[2]])
                action = int(envs[a].action_space.sample())
                a_lists[a].append(float(action))
                obs_list[a], _r, term, trunc, _ = envs[a].step(action)
                obs_list[a] = np.asarray(obs_list[a], dtype=np.float32)
                if term or trunc:
                    active[a] = False
            elif s_lists[a]:
                s_lists[a].append(s_lists[a][-1])
                i_lists[a].append(i_lists[a][-1])
                a_lists[a].append(a_lists[a][-1])
    for env in envs:
        env.close()

    T = max(len(s_lists[0]), 1)
    cols: List[TraceColumn] = []
    for a in range(num_agents):
        s_arr = np.array(s_lists[a], dtype=np.float32)
        i_arr = np.array(i_lists[a], dtype=np.float32)
        agent_drive = _ar1(T, rng)
        for t in range(T):
            bump = 0.65 * agent_drive[t]
            s_arr[t, 0] += bump
            s_arr[t, 1] += 0.85 * bump
            i_arr[t, 0] += 0.45 * agent_drive[t]
        cols.extend(
            [
                TraceColumn(f"agent{a}.sensor.cart_v", a, "sensor", s_arr[:, 0]),
                TraceColumn(f"agent{a}.sensor.cart_v_b", a, "sensor", s_arr[:, 0] + 0.03 * agent_drive),
                TraceColumn(f"agent{a}.sensor.pole_ang_v", a, "sensor", s_arr[:, 1]),
                TraceColumn(f"agent{a}.internal.cart_x", a, "internal", i_arr[:, 0]),
                TraceColumn(f"agent{a}.internal.pole_ang", a, "internal", i_arr[:, 1]),
                TraceColumn(f"agent{a}.action.force", a, "action", np.array(a_lists[a], dtype=np.float32)),
            ]
        )
    return pack_trace(
        cols,
        num_agents=num_agents,
        seed=seed,
        source=f"physics_cartpole_x{num_agents}",
        n_decoy_env=n_decoy_env,
    )
