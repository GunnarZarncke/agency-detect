"""M4: role assignment + blanket scoring + pooled leave-one-animal-out (synthetic)."""

import numpy as np

from uad_worm.preprocess import Processed, whiten_derivative
from uad_worm.score import (
    leave_one_animal_out,
    pooled_score,
    score_members,
)
from uad_worm.synth import controller_with_blanket

AGENT = frozenset({"IA", "IB", "SS", "AA"})
DECOY = frozenset({"BG0", "BG1", "BG2", "BG3"})
CLASSES = ["IA", "IB", "SS", "AA", "E0", "E1"] + [f"BG{i}" for i in range(10)]


def _animal(seed: int) -> Processed:
    # Controller agent embedded in a densely-coupled background (shared slow latent), the
    # regime where the random-partition contrast is meaningful: random cuts through the
    # coupled background leak (high loss), only the true blanket is low (README §7.2).
    rng = np.random.default_rng(seed)
    agent = controller_with_blanket(T=3000, seed=seed).trace  # 6 cols
    T = agent.shape[0]
    z = np.zeros(T)
    for t in range(1, T):
        z[t] = 0.8 * z[t - 1] + rng.standard_normal()
    bg = np.column_stack([z + 0.5 * rng.standard_normal(T) for _ in range(10)])
    trace = np.column_stack([agent, bg])
    return Processed(
        dataset_id=f"synth-{seed}",
        neuron_class=list(CLASSES),
        raw=trace,
        whitened=whiten_derivative(trace),
        behavior={},
    )


def test_true_agent_scores_low_and_passes():
    trace = _animal(0).raw
    res = score_members(trace, [0, 1, 2, 3], ext_dim=6, n_perm=120, seed=0)
    assert res is not None
    assert res.pvalue < 0.05
    assert len(res.roles.internal) >= 1


def test_pooled_agent_beats_decoy():
    animals = [_animal(s) for s in range(4)]
    agent = pooled_score(animals, AGENT, representation="raw", ext_dim=6, n_perm=120, min_members=3)
    decoy = pooled_score(animals, DECOY, representation="raw", ext_dim=6, n_perm=120, min_members=3)
    assert agent.n_animals == 4
    assert agent.pass_rate >= 0.75
    assert agent.combined_p < 0.05
    assert agent.mean_loss < decoy.mean_loss


def test_leave_one_animal_out_generalizes():
    animals = [_animal(s) for s in range(4)]
    out = leave_one_animal_out(
        animals, lambda train: AGENT, representation="raw", ext_dim=6, n_perm=120, min_members=3
    )
    assert out["n_held"] == 4
    assert out["holdout_pass_rate"] >= 0.75
