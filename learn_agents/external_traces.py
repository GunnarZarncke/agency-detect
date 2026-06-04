"""Build learn_agents-compatible traces from external POMDP / physics simulators.

Produces SimulationResult with var_agent, var_role, role_indices, and agent_clusters
so MI clustering and oracle_uad_scores work unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Sequence, Tuple

import numpy as np

from learn_agents.learn_agents import SimulationResult, TraceSimulationConfig


@dataclass(frozen=True)
class TraceColumn:
    name: str
    agent: int  # >=0 agent id; -1 environment / decoy
    role: str  # sensor | internal | action | env
    values: np.ndarray  # (T,)


def pack_trace(
    columns: Sequence[TraceColumn],
    *,
    num_agents: int,
    seed: int,
    source: str,
    n_decoy_env: int = 4,
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    """Stack columns, add optional env decoys for blanket tests, z-score per column."""
    rng = rng or np.random.default_rng(seed)
    T = int(columns[0].values.shape[0])
    cols: List[np.ndarray] = []
    var_names: List[str] = []
    var_agent: List[int] = []
    var_role: List[str] = []
    role_indices: Dict[Tuple[int, str], List[int]] = {}

    def add(col: TraceColumn) -> None:
        idx = len(cols)
        cols.append(col.values.astype(np.float32))
        var_names.append(col.name)
        var_agent.append(col.agent)
        var_role.append(col.role)
        if col.agent >= 0:
            role_indices.setdefault((col.agent, col.role), []).append(idx)

    for col in columns:
        if col.values.shape[0] != T:
            raise ValueError(f"column {col.name} length {col.values.shape[0]} != {T}")
        add(col)

    # Environment columns so oracle_uad_scores has external vars (single-agent POMDPs).
    for j in range(n_decoy_env):
        env = _ar1(T, rng)
        add(TraceColumn(f"env.{j}", -1, "env", env))

    trace = np.stack(cols, axis=1).astype(np.float32)
    trace = (trace - trace.mean(axis=0, keepdims=True)) / (trace.std(axis=0, keepdims=True) + 1e-6)

    agent_clusters = {
        k: sorted(
            role_indices.get((k, "sensor"), [])
            + role_indices.get((k, "internal"), [])
            + role_indices.get((k, "action"), [])
        )
        for k in range(num_agents)
    }

    cfg = replace(
        TraceSimulationConfig(T=T, num_agents=num_agents, seed=seed),
        decoy_vars=n_decoy_env,
    )
    metadata: Dict[str, object] = {
        "var_names": var_names,
        "var_agent": np.array(var_agent, dtype=np.int64),
        "var_role": np.array(var_role, dtype=object),
        "role_indices": role_indices,
        "agent_clusters": agent_clusters,
        "direct_adjacency_target_source": np.zeros((num_agents, num_agents), dtype=np.float32),
        "source": source,
        "config": cfg,
    }
    return SimulationResult(trace=trace, metadata=metadata)


def merge_agent_traces(
    parts: Sequence[SimulationResult],
    *,
    seed: int,
    source: str,
    n_decoy_env: int = 8,
    max_T: int | None = None,
    align: str = "pad",
) -> SimulationResult:
    """One trace with disjoint agent sets from heterogeneous rollouts (mixed detection).

    Drops per-part env decoys; re-labels agents 0..N-1.
    ``align='pad'``: length = max(part lengths), hold last value for shorter parts.
    ``align='truncate'``: length = min(part lengths).
    """
    if not parts:
        raise ValueError("merge_agent_traces requires at least one part")
    if align not in ("pad", "truncate"):
        raise ValueError(f"align must be 'pad' or 'truncate', got {align!r}")
    lengths = [int(p.trace.shape[0]) for p in parts]
    T = max(lengths) if align == "pad" else min(lengths)
    if max_T is not None:
        T = min(T, max_T)

    def _slice_col(values: np.ndarray) -> np.ndarray:
        v = values.astype(np.float32)
        if v.shape[0] >= T:
            return v[:T]
        out = np.empty(T, dtype=np.float32)
        out[: v.shape[0]] = v
        out[v.shape[0] :] = v[-1]
        return out

    columns: List[TraceColumn] = []
    agent_offset = 0
    for part in parts:
        va = np.asarray(part.metadata["var_agent"], dtype=np.int64)
        names = list(part.metadata["var_names"])
        roles = part.metadata["var_role"]
        sub_src = str(part.metadata.get("source", "sub"))
        n_agents = int(part.metadata["config"].num_agents)
        for j in range(part.trace.shape[1]):
            if va[j] < 0:
                continue
            role = str(roles[j]) if hasattr(roles, "__getitem__") else roles
            columns.append(
                TraceColumn(
                    f"{sub_src}.{names[j]}",
                    int(va[j]) + agent_offset,
                    role,
                    _slice_col(part.trace[:, j]),
                )
            )
        agent_offset += n_agents
    return pack_trace(
        columns,
        num_agents=agent_offset,
        seed=seed,
        source=source,
        n_decoy_env=n_decoy_env,
    )


def _ar1(T: int, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        x[t] = 0.96 * x[t - 1] + 0.15 * rng.normal()
    return x
