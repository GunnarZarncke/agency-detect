"""Light sanity checks for the Chunk 2 world wrapper.

These are not the Level-1 estimator tests (those arrive in Chunk 3). They only confirm
the wrapper records the world faithfully and reproducibly.
"""

from __future__ import annotations

import numpy as np

import value_detect as vd


def test_world_defaults_match_source():
    report = vd.verify_world_defaults()
    assert report["ok"], report["mismatches"]
    assert report["var_names"][:5] == ["B", "S", "A", "E", "G"]


def test_trace_shape_and_columns():
    trace = vd.passive_trace(seed=0, n_steps=500)
    assert trace.frame.shape == (500, 9)
    assert list(trace.frame.columns) == vd.variable_names()
    # Binary world.
    assert set(np.unique(trace.frame.to_numpy())).issubset({0, 1})


def test_trace_is_deterministic_per_seed():
    a = vd.passive_trace(seed=3, n_steps=400).frame
    b = vd.passive_trace(seed=3, n_steps=400).frame
    assert a.equals(b)
    c = vd.passive_trace(seed=4, n_steps=400).frame
    assert not a.equals(c)


def test_goal_changes_are_rare():
    # ~1.5% designed flip rate; over a long run it should be well under 5%.
    trace = vd.passive_trace(seed=0, n_steps=8000)
    g = trace.frame["G"].to_numpy()
    change_rate = np.mean(g[1:] != g[:-1])
    assert 0.003 < change_rate < 0.05, change_rate


def test_noise_variable_changes_often():
    trace = vd.passive_trace(seed=0, n_steps=4000)
    w = trace.frame["W"].to_numpy()
    change_rate = np.mean(w[1:] != w[:-1])
    assert 0.4 < change_rate < 0.6, change_rate
