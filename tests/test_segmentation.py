"""Tests for automatic segmentation and segmented scoring."""

from __future__ import annotations

import numpy as np

from intention_detect.evaluate import score_agent_outcome, score_agent_outcome_segmented, score_simulation
from intention_detect.outcomes import CriticalOutcome
from intention_detect.segmentation import calibrate_segment_params, segment_ranges, should_segment
from learn_agents.external_traces import TraceColumn, pack_trace


def _episodic_trace(*, T: int = 600, burst: bool) -> tuple[np.ndarray, dict]:
    """Agent 0 bursts action every 80 ticks; outcome jumps only during bursts."""
    rng = np.random.default_rng(0)
    action = np.zeros(T, dtype=np.float32)
    outcome = rng.normal(0, 0.5, T).astype(np.float32)
    world = rng.normal(0, 0.3, T).astype(np.float32)
    for t in range(20, T, 80):
        action[t : t + 12] = 1.0
        if burst:
            outcome[t + 1 : t + 13] += 2.0
    cols = [
        TraceColumn("world.w", -1, "env", world),
        TraceColumn("resource.out", -1, "env", outcome),
        TraceColumn("agent0.sensor", 0, "sensor", world),
        TraceColumn("agent0.internal", 0, "internal", np.zeros(T)),
        TraceColumn("agent0.action", 0, "action", action),
    ]
    result = pack_trace(cols, num_agents=1, seed=0, source="test", n_decoy_env=0, normalize=False)
    meta = result.metadata
    out_idx = list(meta["var_names"]).index("resource.out")
    meta["critical_outcomes"] = [
        {"name": "resource.out", "index": out_idx, "direction": "lower_is_better"}
    ]
    meta["world_var_indices"] = [list(meta["var_names"]).index("world.w")]
    meta["outcome_influence_ground_truth"] = {"0": burst}
    meta["prefer_segment_scoring"] = True
    return result.trace, meta


def test_should_segment_auto_threshold():
    T = 600
    action = np.zeros(T)
    action[100:120] = 1.0
    assert not should_segment(np.zeros((200, 1)), {"role_indices": {(0, "action"): [0]}})
    # sparse episodic layout
    cols = [
        TraceColumn("world.w", -1, "env", np.zeros(T)),
        TraceColumn("resource.out", -1, "env", np.zeros(T)),
        TraceColumn("agent0.action", 0, "action", action.astype(np.float32)),
        TraceColumn("agent1.action", 1, "action", np.full(T, 0.2, dtype=np.float32)),
    ]
    from learn_agents.learn_agents import SimulationResult

    tr = pack_trace(cols, num_agents=2, seed=0, source="t", n_decoy_env=0, normalize=False)
    meta = tr.metadata
    meta["prefer_segment_scoring"] = True
    assert should_segment(tr.trace, meta)


def test_constant_action_uses_sliding_only():
    T = 600
    action = np.full(T, 0.2)
    params = calibrate_segment_params(T, action)
    assert params.activity_coef == float("inf")
    ranges = segment_ranges(T, action, params)
    assert len(ranges) > 1
    assert all(e - s >= 40 for s, e in ranges)


def test_episodic_action_adds_active_runs():
    T = 600
    action = np.zeros(T)
    action[100:120] = 1.0
    action[300:330] = 1.0
    params = calibrate_segment_params(T, action)
    assert np.isfinite(params.activity_coef)
    ranges = segment_ranges(T, action, params)
    assert any(s <= 100 and e >= 120 for s, e in ranges)


def test_segmented_finds_burst_influencer_full_trace_may_not():
    trace, meta = _episodic_trace(T=600, burst=True)
    outcome = CriticalOutcome.from_mapping(meta["critical_outcomes"][0])
    meta["prefer_segment_scoring"] = True
    full = score_agent_outcome(trace, meta, 0, outcome, seed=0)
    seg = score_agent_outcome_segmented(trace, meta, 0, outcome, seed=0)
    assert seg.n_segments > 1
    assert abs(seg.influence) >= abs(full.influence) * 0.9


def test_segmented_no_false_flag_on_non_influencer():
    trace, meta = _episodic_trace(T=600, burst=False)
    from learn_agents.learn_agents import SimulationResult

    result = SimulationResult(trace=trace, metadata=meta)
    meta["prefer_segment_scoring"] = True
    summary = score_simulation(result, seed=0, segment_mode="segmented")
    assert not summary["agents"][0]["flagged"]
