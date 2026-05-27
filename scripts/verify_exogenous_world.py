#!/usr/bin/env python3
"""
Verify exogenous world design: no agent->world drive, small world->sensor read,
clean MI split between agent clusters and world clusters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent_spotlight.config import SpotlightConfig
from agent_spotlight.peel import _sim_config
from agent_spotlight.proposal import _score_single_cluster, propose_best_cluster
from learn_agents.learn_agents import (
    TraceSimulationConfig,
    _discretize_series,
    factorize_background,
    mi_partition_search,
    simulate_known_agent_trace,
)
from agency_detect.detection import lagmax_mi, build_similarity_matrix


def _world_indices(meta) -> set[int]:
    return set(meta.get("world_all_var_indices", meta.get("world_var_indices", [])))


def _agent_indices(meta) -> set[int]:
    out: set[int] = set()
    for idxs in meta["agent_clusters"].values():
        out.update(idxs)
    return out


def _cluster_kind(idxs, names, agent_idx, world_idx):
    na = sum(1 for i in idxs if i in agent_idx)
    nw = sum(1 for i in idxs if i in world_idx)
    if na > 0 and nw == 0:
        return "agent-only"
    if nw > 0 and na == 0:
        return "world-only"
    if na > 0 and nw > 0:
        return "mixed"
    return "other"


def verify(cfg: TraceSimulationConfig) -> dict:
    sim = simulate_known_agent_trace(cfg)
    trace = sim.trace
    meta = sim.metadata
    names = list(meta["var_names"])
    agent_idx = _agent_indices(meta)
    world_idx = _world_indices(meta)
    a = meta["latent_action"]
    s = meta["latent_sensor"]
    local_w = meta.get("latent_local_world")
    shared_w = meta.get("latent_world_shared")

    # 1) No agent control of world (exogenous)
    action_world_mi = []
    if local_w is not None and cfg.env_action_coupling == 0.0:
        for k in range(cfg.num_agents):
            for j in range(local_w.shape[2]):
                mi = lagmax_mi(
                    _discretize_series(a[:, k], 8),
                    _discretize_series(local_w[:, k, j], 8),
                    max_lag=3,
                )
                action_world_mi.append(float(mi))
    if shared_w is not None:
        for j in range(shared_w.shape[1]):
            for k in range(cfg.num_agents):
                mi = lagmax_mi(
                    _discretize_series(a[:, k], 8),
                    _discretize_series(shared_w[:, j], 8),
                    max_lag=3,
                )
                action_world_mi.append(float(mi))

    # 2) Small but nonzero world -> sensor read
    sensor_world_mi = []
    if local_w is not None:
        for k in range(cfg.num_agents):
            mi = lagmax_mi(
                _discretize_series(s[:, k], 8),
                _discretize_series(local_w[:, k].mean(axis=1), 8),
                max_lag=3,
            )
            sensor_world_mi.append(float(mi))
    if shared_w is not None:
        smean = shared_w.mean(axis=1)
        for k in range(cfg.num_agents):
            mi = lagmax_mi(_discretize_series(s[:, k], 8), _discretize_series(smean, 8), max_lag=3)
            sensor_world_mi.append(float(mi))

    # 3) MI partition cluster purity
    work, _ = factorize_background(trace, n_components=1)
    part = mi_partition_search(work, fixed_k=16, bins=8, max_lag=3, k_selection="mdl")
    kinds = {"agent-only": 0, "world-only": 0, "mixed": 0, "other": 0}
    top_rows = []
    sc_cfg = SpotlightConfig(
        env_vars_per_agent=cfg.env_vars_per_agent,
        env_action_coupling=cfg.env_action_coupling,
        env_to_sensor_strength=cfg.env_to_sensor_strength,
        world_vars=cfg.world_vars,
        world_to_sensor_strength=cfg.world_to_sensor_strength,
        interaction_strength=cfg.interaction_strength,
        mixing_strength=cfg.mixing_strength,
        local_env_strength=cfg.local_env_strength,
        leakage_strength=cfg.leakage_strength,
        decoy_vars=cfg.decoy_vars,
    )
    for cid in np.unique(part.labels[part.labels >= 0]):
        idxs = np.where(part.labels == cid)[0].tolist()
        if len(idxs) < 2:
            continue
        kind = _cluster_kind(idxs, names, agent_idx, world_idx)
        kinds[kind] += 1
        score, *_ = _score_single_cluster(work, idxs, sc_cfg)
        top_rows.append((score, len(idxs), kind, [names[i] for i in idxs[:4]]))
    top_rows.sort(reverse=True)

    best, _ = propose_best_cluster(trace, sc_cfg, peeled=set())
    best_kind = _cluster_kind(best.var_indices, names, agent_idx, world_idx)

    return {
        "num_vars": trace.shape[1],
        "n_agent_vars": len(agent_idx),
        "n_world_vars": len(world_idx),
        "max_action_world_mi": max(action_world_mi) if action_world_mi else 0.0,
        "mean_sensor_world_mi": float(np.mean(sensor_world_mi)) if sensor_world_mi else 0.0,
        "max_sensor_world_mi": max(sensor_world_mi) if sensor_world_mi else 0.0,
        "partition_kinds": kinds,
        "top_cluster_kinds": [r[2] for r in top_rows[:8]],
        "proposed_kind": best_kind,
        "proposed_names": [names[i] for i in best.var_indices],
        "mixed_in_top3": sum(1 for r in top_rows[:3] if r[2] == "mixed"),
    }


def main() -> None:
    base = _sim_config(
        SpotlightConfig(
            local_env_strength=1.8,
            interaction_strength=0.10,
            mixing_strength=0.02,
            leakage_strength=0.01,
            env_vars_per_agent=0,
            env_action_coupling=0.0,
            world_vars=12,
            world_to_sensor_strength=0.08,
            decoy_vars=0,
        )
    )

    print("=== Exogenous world verification (shared world, adapt1+2) ===\n")
    r = verify(base)
    print(f"Trace: {r['n_agent_vars']} agent + {r['n_world_vars']} world vars (N={r['num_vars']})")
    print(f"Action→world MI: max={r['max_action_world_mi']:.4f} (expect ~0 when exogenous)")
    print(
        f"Sensor←world MI: mean={r['mean_sensor_world_mi']:.4f} max={r['max_sensor_world_mi']:.4f} "
        f"(expect small >0)"
    )
    print(f"K=16 partition kinds: {r['partition_kinds']} (expect 0 mixed for shared-only)")
    print(f"Top-8 cluster kinds: {r['top_cluster_kinds']}")
    print(f"Proposed pass-1: {r['proposed_kind']} {r['proposed_names']}")
    print(f"Mixed clusters in top-3: {r['mixed_in_top3']}")

    ok = (
        r["max_action_world_mi"] < 0.05
        and 0.005 < r["mean_sensor_world_mi"] < 0.20
        and r["partition_kinds"].get("mixed", 0) == 0
        and r["proposed_kind"] == "agent-only"
    )
    print(f"\n{'PASS' if ok else 'FAIL'}: exogenous shared-world split checks")

    # Local patches: allowed small mixing with own agent only.
    local_cfg = _sim_config(
        SpotlightConfig(
            local_env_strength=1.8,
            interaction_strength=0.10,
            mixing_strength=0.02,
            leakage_strength=0.01,
            env_vars_per_agent=3,
            env_action_coupling=0.0,
            env_to_sensor_strength=0.05,
            world_vars=0,
            decoy_vars=0,
        )
    )
    rl = verify(local_cfg)
    print("\n=== Local exogenous patches (optional, adapt3 add-on) ===")
    print(f"Action→world MI max={rl['max_action_world_mi']:.4f}")
    print(f"Partition kinds: {rl['partition_kinds']} (local may show minor agent+own-local mixing)")
    local_ok = rl["max_action_world_mi"] < 0.05
    print(f"{'PASS' if local_ok else 'FAIL'}: local patches exogenous (no action drive)")

    if not (ok and local_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
