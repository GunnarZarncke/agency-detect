"""M5 + M6: random-class-set null and behavior-prediction gain (synthetic)."""

import numpy as np

from uad_worm.evaluate import (
    behavior_prediction_gain,
    internal_autonomy,
    joint_null,
    pooled_behavior_gain,
    random_class_set_null,
)
from uad_worm.preprocess import Processed, whiten_derivative
from uad_worm.synth import controller_with_blanket

AGENT = frozenset({"IA", "IB", "SS", "AA"})
CLASSES = ["IA", "IB", "SS", "AA", "E0", "E1"] + [f"BG{i}" for i in range(10)]


def _animal(seed: int) -> Processed:
    rng = np.random.default_rng(seed)
    agent = controller_with_blanket(T=3000, seed=seed).trace  # cols: I0,I1,S,A,E0,E1
    T = agent.shape[0]
    z = np.zeros(T)
    for t in range(1, T):
        z[t] = 0.8 * z[t - 1] + rng.standard_normal()
    bg = np.column_stack([z + 0.5 * rng.standard_normal(T) for _ in range(10)])
    trace = np.column_stack([agent, bg])
    # Behavior driven by the agent's action + internal state (so the agent should predict it).
    velocity = 0.7 * agent[:, 3] + 0.3 * agent[:, 0] + 0.2 * rng.standard_normal(T)
    return Processed(
        dataset_id=f"synth-{seed}",
        neuron_class=list(CLASSES),
        raw=trace,
        whitened=whiten_derivative(trace),
        behavior={"velocity": velocity},
    )


def test_random_class_set_null_flags_agent():
    animals = [_animal(s) for s in range(3)]
    res = random_class_set_null(
        animals, AGENT, n_sets=30, representation="raw", ext_dim=6, min_members=3, seed=0
    )
    assert res.observed_loss < res.null_losses.mean()
    assert res.pvalue < 0.1


def test_behavior_prediction_gain_positive_for_agent():
    proc = _animal(0)
    gain = behavior_prediction_gain(proc, AGENT, feature="velocity", representation="raw")
    assert gain is not None
    assert gain > 0.05


def test_pooled_behavior_gain_positive():
    animals = [_animal(s) for s in range(3)]
    gain = pooled_behavior_gain(animals, AGENT, feature="velocity", representation="raw")
    assert gain > 0.05


def test_internal_autonomy_beats_redundant_block():
    # Conditioned on the environment, the controller agent self-predicts; the shared-latent
    # background block does not (its self-prediction is explained away by the rest of E).
    trace = _animal(0).raw
    agent_aut = internal_autonomy(trace, [0, 1, 2, 3], ext_dim=6)
    bg_aut = internal_autonomy(trace, [6, 7, 8, 9], ext_dim=6)
    assert agent_aut > bg_aut
    assert agent_aut > 0.1


def test_joint_null_agent_corner():
    trace = _animal(0).raw
    agent = joint_null(trace, [0, 1, 2, 3], ext_dim=6, n_perm=120, seed=0)
    decoy = joint_null(trace, [6, 7, 8, 9], ext_dim=6, n_perm=120, seed=0)
    assert agent.agent_corner is True          # low loss_p + high autonomy_p
    assert decoy.agent_corner is False
    assert agent.loss_p < decoy.loss_p
    assert agent.autonomy_p > decoy.autonomy_p
