"""M0: Gaussian CMI estimator sanity checks (uad_worm.cmi)."""

import numpy as np

from uad_worm.cmi import gaussian_cmi, knn_cmi


def test_independent_variables_near_zero():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4000)
    y = rng.standard_normal(4000)
    assert gaussian_cmi(x, y) < 0.02


def test_correlated_variables_positive():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(4000)
    y = 0.8 * x + 0.6 * rng.standard_normal(4000)
    assert gaussian_cmi(x, y) > 0.2


def test_conditional_independence_detected():
    # Common-cause chain: X <- Z -> Y. Marginally dependent, conditionally independent.
    rng = np.random.default_rng(2)
    z = rng.standard_normal(4000)
    x = z + 0.5 * rng.standard_normal(4000)
    y = z + 0.5 * rng.standard_normal(4000)
    marginal = gaussian_cmi(x, y)
    conditional = gaussian_cmi(x, y, z)
    assert marginal > 0.3
    assert conditional < 0.02
    assert conditional < 0.1 * marginal


def test_multivariate_runs():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((2000, 2))
    Y = rng.standard_normal((2000, 2))
    Z = rng.standard_normal((2000, 3))
    val = gaussian_cmi(X, Y, Z)
    assert val >= 0.0


def test_knn_cmi_independent_and_correlated():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(3000)
    y = rng.standard_normal(3000)
    assert knn_cmi(x, y) < 0.05
    y2 = 0.8 * x + 0.6 * rng.standard_normal(3000)
    assert knn_cmi(x, y2) > 0.15


def test_knn_cmi_conditional_independence():
    # X <- Z -> Y: marginally dependent, conditionally independent.
    rng = np.random.default_rng(1)
    z = rng.standard_normal(3000)
    x = z + 0.5 * rng.standard_normal(3000)
    y = z + 0.5 * rng.standard_normal(3000)
    marginal = knn_cmi(x, y)
    conditional = knn_cmi(x, y, z)
    assert marginal > 0.2
    assert conditional < 0.5 * marginal


def test_knn_cmi_detects_nonmonotone_dependence():
    # Y = X^2 + noise: zero linear/monotone correlation, but strong dependence that the
    # Gaussian estimator misses and kNN should catch.
    rng = np.random.default_rng(2)
    x = rng.standard_normal(4000)
    y = x**2 + 0.3 * rng.standard_normal(4000)
    assert gaussian_cmi(x, y) < 0.05
    assert knn_cmi(x, y) > 0.2
