"""Unit tests for E16 extended pool wiring (no meltingpot required)."""

from __future__ import annotations

import pytest

from amortized_agency.kinds import (
    ALL_KINDS,
    EXTENDED_TRAIN_KINDS,
    EXTERNAL_KINDS,
    Kind,
)
from amortized_agency.worlds import simulate_episode
from learn_agents.external_registry import EXTERNAL_BUILDERS, build_external_trace


def test_sim_kinds_unchanged():
    assert len(ALL_KINDS) == 3
    for k in ALL_KINDS:
        assert k.backend == "sim"
        assert k.external_key is None


def test_external_kind_validation():
    with pytest.raises(ValueError):
        Kind("bad", 1, "x", backend="external")
    with pytest.raises(ValueError):
        Kind("bad", 1, "x", backend="sim", external_key="physics_cartpole")


def test_registry_covers_external_kinds():
    for k in EXTERNAL_KINDS:
        assert k.external_key in EXTERNAL_BUILDERS


def test_build_physics_trace():
    r = build_external_trace("physics_cartpole", seed=0, t_steps=500)
    assert r.trace.shape[0] >= 1
    assert r.trace.shape[1] >= 5


def test_simulate_external_grid_episode():
    kind = next(k for k in EXTENDED_TRAIN_KINDS if k.name == "grid_pomdp_3x3")
    ep = simulate_episode(kind, window_len=100, seed=1, t_steps=250)
    assert ep.window.shape[0] == 100
    assert ep.agent_ids.shape[0] == ep.window.shape[1]
