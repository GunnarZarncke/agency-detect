"""Tests for E18 outcome-influence detection."""

from __future__ import annotations

from intention_detect.evaluate import auroc, score_simulation
from intention_detect.outcomes import attach_physics_critical_outcome
from learn_agents.learn_agents import TraceSimulationConfig, simulate_known_agent_trace
from learn_agents.physics_pomdp import roll_cartpole_partial_obs


def test_self_preserving_flags_more_often_than_reactive():
    reactive = score_simulation(
        simulate_known_agent_trace(
            TraceSimulationConfig(
                T=2000,
                num_agents=3,
                resource_vars=2,
                world_vars=4,
                self_preserving_agent=-1,
                episodic=False,
                normalize_trace=False,
                seed=1,
            )
        ),
        seed=1,
    )
    preserving = score_simulation(
        simulate_known_agent_trace(
            TraceSimulationConfig(
                T=2000,
                num_agents=3,
                resource_vars=2,
                world_vars=4,
                self_preserving_agent=0,
                episodic=False,
                normalize_trace=False,
                seed=1,
            )
        ),
        seed=1,
    )
    r0 = reactive["agents"][0]["max_combined"]
    p0 = preserving["agents"][0]["max_combined"]
    assert preserving["agents"][0]["flagged"]
    assert not reactive["agents"][0]["flagged"]
    assert preserving["agents"][0]["max_combined"] >= reactive["agents"][0]["max_combined"]


def test_cartpole_balance_flags_on_pole_outcome():
    result = attach_physics_critical_outcome(
        roll_cartpole_partial_obs(seed=0, policy="balance", normalize=False),
        policy="balance",
    )
    summary = score_simulation(result, seed=0)
    assert summary["agents"][0]["flagged"]


def test_cartpole_random_not_flagged():
    result = attach_physics_critical_outcome(
        roll_cartpole_partial_obs(seed=0, policy="random", normalize=False),
        policy="random",
    )
    summary = score_simulation(result, seed=0)
    if summary.get("skipped"):
        return
    assert not summary["agents"][0]["flagged"]


def test_auroc_perfect():
    assert auroc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
