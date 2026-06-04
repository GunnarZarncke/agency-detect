"""Partially observed physics baseline (Gymnasium CartPole).

S = velocities (cart v, pole angular v) — what a velocity-only sensor sees.
I = positions (cart x, pole angle) — simulator state not in S; drives transitions.
A = applied force direction.

Episodes are short (≤500 steps) but one continuous rollout per episode (no concatenation).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from learn_agents.external_traces import TraceColumn, pack_trace
from learn_agents.learn_agents import SimulationResult


def roll_cartpole_partial_obs(
    *,
    seed: int = 0,
    max_steps: int = 500,
    n_episodes: int = 1,
) -> SimulationResult:
    """One or more CartPole-v1 episodes; if n_episodes>1, episodes are separate runs (not spliced)."""
    if n_episodes != 1:
        raise ValueError("use n_episodes=1 for a single continuous physics trace (no synthetic concat)")
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)
    s_list, i_list, a_list = [], [], []
    for _ in range(max_steps):
        full = np.asarray(obs, dtype=np.float32)
        # full = [x, x_dot, theta, theta_dot]
        s_list.append([full[1], full[3]])
        i_list.append([full[0], full[2]])
        action = env.action_space.sample()
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
    return pack_trace(cols, num_agents=1, seed=seed, source="physics_cartpole_v1")
