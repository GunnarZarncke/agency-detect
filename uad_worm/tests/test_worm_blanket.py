"""M0: blanket loss + random-partition contrast on synthetic systems (uad_worm).

The decisive checks: a true Markov blanket beats random partitions; a strongly-correlated
system with no mediating interface does not.
"""

import numpy as np

from uad_worm.blanket import blanket_loss, blanket_pvalue
from uad_worm.synth import controller_with_blanket, correlated_no_blanket


def test_true_blanket_scores_low_and_beats_random():
    sys = controller_with_blanket(T=4000, seed=0)
    res = blanket_pvalue(
        sys.trace,
        internal=sys.roles["internal"],
        external=sys.roles["external"],
        interface=sys.interface,
        n_perm=200,
        seed=0,
    )
    # True cut mediates better than chance cuts of the same sizes. The random-partition
    # null is right-skewed (bounded at 0), so the rank-based p-value is the arbiter; z is
    # reported but secondary.
    assert res.observed < res.null.mean()
    assert res.pvalue < 0.05
    assert res.z < -1.0


def test_correlated_system_has_no_blanket():
    sys = correlated_no_blanket(T=4000, seed=0)
    res = blanket_pvalue(
        sys.trace,
        internal=sys.roles["internal"],
        external=sys.roles["external"],
        interface=sys.interface,
        n_perm=200,
        seed=0,
    )
    # No cut is special: the arbitrary cut sits inside the random-partition null.
    assert res.pvalue > 0.1


def test_blanket_loss_is_nonnegative_scalar():
    sys = controller_with_blanket(T=1000, seed=1)
    loss = blanket_loss(
        sys.trace, sys.roles["internal"], sys.roles["external"], sys.interface
    )
    assert isinstance(loss, float)
    assert loss >= 0.0
