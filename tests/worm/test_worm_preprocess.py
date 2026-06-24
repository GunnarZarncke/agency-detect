"""M2: preprocessing + whitening."""

import numpy as np

from uad_worm.preprocess import (
    lag1_autocorr,
    resample_uniform,
    whiten_derivative,
    zscore,
)


def test_resample_recovers_linear_signal():
    # Irregular but monotonic timestamps; a linear signal must be recovered exactly.
    rng = np.random.default_rng(0)
    t = np.cumsum(0.5 + 0.05 * rng.random(200))
    y = (3.0 * t - 1.0).reshape(-1, 1)
    out = resample_uniform(t, y)
    uniform = np.linspace(t[0], t[-1], len(t))
    assert np.allclose(out[:, 0], 3.0 * uniform - 1.0, atol=1e-6)


def test_zscore_unit_variance():
    rng = np.random.default_rng(1)
    x = 5.0 + 2.0 * rng.standard_normal((1000, 3))
    z = zscore(x)
    assert np.allclose(z.mean(0), 0.0, atol=1e-6)
    assert np.allclose(z.std(0), 1.0, atol=1e-6)


def test_whitening_reduces_lag1_autocorrelation():
    # Slow AR(0.9) ~ GCaMP-like; the derivative must have lower lag-1 autocorrelation.
    rng = np.random.default_rng(2)
    T = 5000
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.9 * x[t - 1] + rng.standard_normal()
    raw = zscore(x.reshape(-1, 1))
    white = whiten_derivative(raw)
    ac_raw = float(lag1_autocorr(raw)[0])
    ac_white = float(lag1_autocorr(white)[0])
    assert ac_raw > 0.8
    assert ac_white < ac_raw
    assert ac_white < 0.3
