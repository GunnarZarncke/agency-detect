"""M3: candidate generation (lagged-corr communities + class-set helpers)."""

import numpy as np

from uad_worm.candidates import (
    agglomerative_communities,
    classes_of,
    classes_to_indices,
    community_class_sets,
    random_class_sets,
)


def _two_block_trace(T=3000, seed=0):
    # Two groups, each driven by its own lagged latent → strong within-group lagged corr.
    rng = np.random.default_rng(seed)
    zA = np.zeros(T); zB = np.zeros(T)
    for t in range(1, T):
        zA[t] = 0.8 * zA[t - 1] + rng.standard_normal()
        zB[t] = 0.8 * zB[t - 1] + rng.standard_normal()
    cols = []
    for _ in range(4):
        cols.append(zA + 0.3 * rng.standard_normal(T))
    for _ in range(4):
        cols.append(zB + 0.3 * rng.standard_normal(T))
    return np.column_stack(cols)


def test_communities_recover_blocks():
    trace = _two_block_trace()
    comms = agglomerative_communities(trace, min_size=2, max_size=6)
    # The two latent-driven blocks should appear as separate communities.
    assert len(comms) >= 2
    blocks = {frozenset(c) for c in comms}
    assert frozenset({0, 1, 2, 3}) in blocks or frozenset({4, 5, 6, 7}) in blocks


def test_class_set_helpers():
    neuron_class = ["AVA", None, "RIM", "AVB", None, "AVA"]
    cs = classes_of([0, 1, 2, 5], neuron_class)
    assert cs == frozenset({"AVA", "RIM"})
    idx = classes_to_indices(frozenset({"AVA"}), neuron_class)
    assert idx == [0, 5]


def test_community_class_sets_min_classes():
    neuron_class = ["AVA", "RIM", "AVB", "AIB"]
    comms = [[0, 1, 2, 3], [0, 1]]
    sets = community_class_sets(comms, neuron_class, min_classes=3)
    assert frozenset({"AVA", "RIM", "AVB", "AIB"}) in sets
    assert all(len(s) >= 3 for s in sets)


def test_random_class_sets_sized_and_varied():
    uni = [f"C{i}" for i in range(20)]
    sets = random_class_sets(uni, size=5, n=10, seed=0)
    assert len(sets) == 10
    assert all(len(s) == 5 for s in sets)
    assert len(set(sets)) > 1
