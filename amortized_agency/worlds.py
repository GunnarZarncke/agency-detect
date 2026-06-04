from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import numpy as np

from learn_agents.learn_agents import SimulationResult, TraceSimulationConfig, simulate_known_agent_trace

from amortized_agency.benchmark import EVAL_T_STEPS
from amortized_agency.kinds import Kind


@dataclass(frozen=True)
class Episode:
    """One window slice restricted to agent variables."""

    window: np.ndarray  # [W, N]
    agent_ids: np.ndarray  # [N] ground-truth agent id per column
    kind: str
    seed: int
    trace_T: int = 0  # full trace length before slicing (external may be short)


def episode_from_result(
    result: SimulationResult,
    kind: Kind,
    window_len: int,
    seed: int,
) -> Episode:
    trace = result.trace
    var_agent = np.asarray(result.metadata["var_agent"], dtype=np.int64)
    agent_cols = np.where(var_agent >= 0)[0]
    w = min(window_len, trace.shape[0])
    sub = trace[:w, agent_cols]
    return Episode(
        window=sub.astype(np.float32),
        agent_ids=var_agent[agent_cols],
        kind=kind.name,
        seed=seed,
        trace_T=int(trace.shape[0]),
    )


def simulate_episode(
    kind: Kind,
    window_len: int,
    seed: int,
    t_steps: int | None = None,
    overrides: Optional[Dict[str, float]] = None,
) -> Episode:
    # Match E13 MI baseline: long horizon then slice [:window_len] (see benchmark.EVAL_T_STEPS).
    t = t_steps if t_steps is not None else max(window_len, EVAL_T_STEPS)

    if kind.backend == "external":
        from learn_agents.external_registry import build_external_trace

        result = build_external_trace(kind.external_key or "", seed=seed, t_steps=t)
        return episode_from_result(result, kind, window_len, seed)

    cfg = TraceSimulationConfig(
        T=t,
        num_agents=kind.num_agents,
        copies_per_role=kind.copies_per_role,
        decoy_vars=kind.decoy_vars,
        interaction_strength=kind.interaction_strength,
        agent_variant_mode=kind.variant_mode,
        episodic=False,
        seed=seed,
    )
    if overrides:
        cfg = replace(cfg, **overrides)
    result = simulate_known_agent_trace(cfg)
    return episode_from_result(result, kind, window_len, seed)


def generate_pool(
    kinds: List[Kind],
    n_worlds: int,
    window_len: int,
    seed_offset: int = 0,
    window_choices: List[int] | None = None,
    rng: np.random.Generator | None = None,
) -> List[Episode]:
    episodes: List[Episode] = []
    gen = rng if rng is not None else np.random.default_rng(seed_offset)
    for k_idx, kind in enumerate(kinds):
        for i in range(n_worlds):
            seed = seed_offset + k_idx * 10_000 + i
            w = int(gen.choice(window_choices)) if window_choices else window_len
            t_steps = max(w, window_len, EVAL_T_STEPS)
            if kind.backend == "external" and kind.name.startswith("physics_cartpole"):
                # Short physics rollouts: do not require T=2000 simulation.
                t_steps = max(w, 500)
            episodes.append(simulate_episode(kind, w, seed=seed, t_steps=t_steps))
    return episodes


def same_agent_matrix(agent_ids: np.ndarray) -> np.ndarray:
    """Binary [N,N] target: 1 if columns share an agent id."""
    n = len(agent_ids)
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        out[i, i] = 1.0
        for j in range(i + 1, n):
            same = float(agent_ids[i] == agent_ids[j])
            out[i, j] = out[j, i] = same
    return out
