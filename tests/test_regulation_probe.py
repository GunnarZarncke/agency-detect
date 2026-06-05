"""Tests for Option D regulation probe."""

from __future__ import annotations

import numpy as np

from learn_agents.regulation_probe import compensation, flatness, regulation_index, score_simulation
from learn_agents.external_traces import TraceColumn, pack_trace


def _synthetic_homeostat(T: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    s = np.zeros(T, dtype=np.float32)
    h = np.zeros(T, dtype=np.float32)
    a = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        s[t] = 0.85 * s[t - 1] + 0.35 * rng.normal()
        h[t] = 0.95 * h[t - 1] + 0.02 * s[t]
        a[t] = float(-0.95 * s[t - 1] + 0.02 * rng.normal())
    cols = [
        TraceColumn("agent0.sensor.drive", 0, "sensor", s),
        TraceColumn("agent0.internal.state", 0, "internal", h),
        TraceColumn("agent0.action.ctrl", 0, "action", a),
    ]
    return pack_trace(cols, num_agents=1, seed=seed, source="synthetic_homeostat", normalize=False)


def _synthetic_reactive(T: int = 400, seed: int = 1):
    rng = np.random.default_rng(seed)
    s = rng.normal(0, 1.0, size=T).astype(np.float32)
    h = np.cumsum(0.5 * s + 0.1 * rng.normal(size=T)).astype(np.float32)
    a = (0.4 * s + 0.1 * rng.normal(size=T)).astype(np.float32)
    cols = [
        TraceColumn("agent0.sensor.drive", 0, "sensor", s),
        TraceColumn("agent0.internal.state", 0, "internal", h),
        TraceColumn("agent0.action.ctrl", 0, "action", a),
    ]
    return pack_trace(cols, num_agents=1, seed=seed, source="synthetic_reactive", normalize=False)


def test_flatness_and_compensation_primitives():
    s = np.sin(np.linspace(0, 20, 200))
    h = 0.05 * s  # much quieter
    a = -s
    f = flatness(h, s)
    k = compensation(a, s, lag=1)
    assert f > 0.9
    assert k > 0.5


def test_regulation_homeostat_beats_reactive():
    homeo = score_simulation(_synthetic_homeostat(), threshold=0.15, active_ratio_max=0.05)
    react = score_simulation(_synthetic_reactive(), threshold=0.15, active_ratio_max=0.05)
    r_homeo = homeo["agents"][0]["max_regulation"]
    r_react = react["agents"][0]["max_regulation"]
    assert r_homeo > r_react
    assert homeo["flagged_agents"] == [0]
    assert react["flagged_agents"] == []
    assert r_homeo >= 0.15


def test_regulation_index_tuple():
    s = np.random.default_rng(0).normal(size=100)
    h = 0.1 * s
    a = -s
    f, k, r, ratio = regulation_index(h, s, a, active_ratio_max=1.0)
    assert r == f * k
    assert ratio < 1.0
