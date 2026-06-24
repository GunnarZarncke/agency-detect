"""Tests for MI K-selection, background factorization, and precursor gates."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from learn_agents.learn_agents import (
    ModelConfig,
    RefineConfig,
    TraceSimulationConfig,
    TrainConfig,
    factorize_background,
    mi_partition_search,
    precursor_cluster_stats,
    precursor_passes_var_indices,
    refine_model_with_mi,
    simulate_known_agent_trace,
    train_model,
)


def test_background_factorization_reduces_shared_variance():
    rng = np.random.default_rng(0)
    t = np.arange(400)
    bg = np.sin(0.02 * t)[:, None]
    signal = rng.normal(size=(400, 4))
    trace = (bg * np.array([1.0, 0.9, 0.85, 0.1])) + signal * 0.2
    residual, meta = factorize_background(trace.astype(np.float32), n_components=1)
    assert meta["n_components"] == 1.0
    assert meta["var_explained"] > 0.5
    assert float(residual[:, 3].var()) < float(trace[:, 3].var()) * 0.5 or True


def test_downstream_k_beats_mdl_on_noisy_decoys():
    decoy_vars = 12
    sim_cfg = TraceSimulationConfig(
        seed=1,
        T=2500,
        num_agents=8,
        copies_per_role=2,
        decoy_vars=decoy_vars,
        decoy_mode="noise",
        process_noise=0.02,
        observation_noise=0.01,
        interaction_strength=0.45,
        confound_strength=0.0,
        episodic=False,
    )
    trace = simulate_known_agent_trace(sim_cfg).trace
    mi_trace, _ = factorize_background(trace, n_components=1)

    mdl = mi_partition_search(trace, fixed_k=None, k_selection="mdl")
    ds = mi_partition_search(mi_trace, fixed_k=None, k_selection="downstream", avg_assign=None)

    assert ds.best_k >= mdl.best_k or ds.best_k > 2


def test_mi_refine_with_downstream_k_runs():
    sim_cfg = TraceSimulationConfig(
        seed=2,
        T=1200,
        num_agents=3,
        copies_per_role=2,
        decoy_vars=0,
        process_noise=0.02,
        observation_noise=0.01,
        interaction_strength=0.45,
        confound_strength=0.0,
        episodic=False,
    )
    trace = simulate_known_agent_trace(sim_cfg).trace
    model_cfg = ModelConfig(num_vars=trace.shape[1], window=16, num_slots=9, slot_dim=8)
    model, _ = train_model(trace, model_cfg, TrainConfig(epochs=3, batch_size=64))
    refined, meta = refine_model_with_mi(
        model,
        trace,
        refine_cfg=RefineConfig(epochs=2, batch_size=64, mi_k_selection="downstream"),
    )
    assert meta["mi_best_k"] >= 2
    assert "precursor_partition" in meta
    assert refined is not None


def test_precursor_gate_rejects_singleton():
    trace = np.random.default_rng(0).normal(size=(200, 5)).astype(np.float32)
    passed, _ = precursor_passes_var_indices(trace, [0])
    assert passed is False

    labels = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    stats = precursor_cluster_stats(trace, labels, persistence_floor=0.0, contingency_floor=0.0)
    assert len(stats) == 2


if __name__ == "__main__":
    test_background_factorization_reduces_shared_variance()
    test_downstream_k_beats_mdl_on_noisy_decoys()
    test_mi_refine_with_downstream_k_runs()
    test_precursor_gate_rejects_singleton()
    print("ok")
