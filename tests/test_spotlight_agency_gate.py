"""Agency-signature gate skips world-only MI clusters."""

from __future__ import annotations

import numpy as np

from agent_spotlight.config import SpotlightConfig
from agent_spotlight.validation import agency_signature_for_indices
from learn_agents.learn_agents import TraceSimulationConfig, simulate_known_agent_trace


def test_world_cluster_fails_agency_signature() -> None:
    sim = simulate_known_agent_trace(
        TraceSimulationConfig(
            seed=1,
            T=2000,
            num_agents=4,
            copies_per_role=2,
            decoy_vars=0,
            interaction_strength=0.10,
            mixing_strength=0.02,
            leakage_strength=0.01,
            local_env_strength=1.8,
            world_vars=8,
            env_action_coupling=0.0,
            episodic=False,
        )
    )
    var_names = list(sim.metadata["var_names"])
    world_idxs = list(sim.metadata["world_all_var_indices"])
    cfg = SpotlightConfig(require_agency_signature=True)

    sig = agency_signature_for_indices(world_idxs, var_names, sim.trace, cfg)
    assert not sig["passed"]

    agent_idxs = list(sim.metadata["agent_clusters"][0])
    sig_agent = agency_signature_for_indices(agent_idxs, var_names, sim.trace, cfg)
    assert sig_agent["passed"]
    assert sig_agent["n_sensors"] >= 1
    assert sig_agent["n_actions"] >= 1
    assert sig_agent["n_internals"] >= 1
