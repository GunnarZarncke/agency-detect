"""Gaussian (partial-correlation) conditional mutual information.

Primary CMI estimator for E20: closed-form, stable at T≈1600, the right default for
continuous calcium where nonparametric kNN-CMI is underpowered (README §7.1). For
jointly-Gaussian variables,

    I(X;Y|Z) = 0.5 * ( logdet Σ_{X|Z} + logdet Σ_{Y|Z} − logdet Σ_{XY|Z} )

where Σ_{A|Z} = Σ_AA − Σ_AZ Σ_ZZ⁻¹ Σ_ZA is the conditional covariance. Units: nats.
Population value is ≥0; finite-sample estimates are clamped at 0.
"""

from __future__ import annotations

import numpy as np


def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    return a


def _standardize(a: np.ndarray) -> np.ndarray:
    mu = a.mean(axis=0, keepdims=True)
    sd = a.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (a - mu) / sd


def _logdet(cov: np.ndarray) -> float:
    cov = np.atleast_2d(cov)
    sign, val = np.linalg.slogdet(cov)
    if sign <= 0:
        # Ridge already applied by caller; fall back to eigenvalue floor.
        w = np.linalg.eigvalsh(cov)
        w = np.clip(w, 1e-12, None)
        return float(np.sum(np.log(w)))
    return float(val)


def _cond_cov(cov: np.ndarray, a_idx: np.ndarray, z_idx: np.ndarray) -> np.ndarray:
    """Conditional covariance of block a given block z, from a joint covariance."""
    if z_idx.size == 0:
        return cov[np.ix_(a_idx, a_idx)]
    c_aa = cov[np.ix_(a_idx, a_idx)]
    c_az = cov[np.ix_(a_idx, z_idx)]
    c_zz = cov[np.ix_(z_idx, z_idx)]
    return c_aa - c_az @ np.linalg.solve(c_zz, c_az.T)


def gaussian_cmi(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray | None = None,
    *,
    ridge: float = 1e-6,
    standardize: bool = True,
) -> float:
    """Estimate I(X;Y|Z) in nats under a Gaussian model.

    X, Y, Z are (n, d) arrays (1-D allowed). Z=None (or 0 columns) gives I(X;Y).
    `ridge` stabilizes near-singular covariances; `standardize` rescales columns so the
    ridge is scale-appropriate.
    """
    X = _as_2d(X)
    Y = _as_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError("X and Y must have the same number of rows")
    blocks = [X, Y]
    if Z is not None:
        Z = _as_2d(Z)
        if Z.shape[0] != n:
            raise ValueError("Z must have the same number of rows as X, Y")
        if Z.shape[1] == 0:
            Z = None
    if Z is not None:
        blocks.append(Z)

    data = np.concatenate(blocks, axis=1)
    if standardize:
        data = _standardize(data)
    cov = np.cov(data, rowvar=False)
    cov = np.atleast_2d(cov)
    cov = cov + ridge * np.eye(cov.shape[0])

    dx, dy = X.shape[1], Y.shape[1]
    x_idx = np.arange(dx)
    y_idx = np.arange(dx, dx + dy)
    xy_idx = np.arange(dx + dy)
    z_idx = np.arange(dx + dy, cov.shape[0]) if Z is not None else np.array([], dtype=int)

    cmi = 0.5 * (
        _logdet(_cond_cov(cov, x_idx, z_idx))
        + _logdet(_cond_cov(cov, y_idx, z_idx))
        - _logdet(_cond_cov(cov, xy_idx, z_idx))
    )
    return max(0.0, float(cmi))
