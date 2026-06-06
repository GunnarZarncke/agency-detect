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
from scipy.spatial import cKDTree
from scipy.special import digamma


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


def knn_cmi(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray | None = None,
    *,
    k: int = 5,
    standardize: bool = True,
    seed: int = 0,
) -> float:
    """Nonparametric I(X;Y|Z) via the KSG / Frenzel–Pompe kNN estimator (nats).

    For Z given (Frenzel & Pompe 2007):
        I(X;Y|Z) = ψ(k) + < ψ(n_z+1) − ψ(n_xz+1) − ψ(n_yz+1) >
    with neighbour counts taken inside the k-th-neighbour Chebyshev radius from the joint
    (X,Y,Z) space. For Z=None this reduces to the KSG-1 estimator of I(X;Y). Captures
    nonlinear (incl. non-monotone) dependence the Gaussian estimator misses, but degrades in
    high dimension — callers must keep the total dimensionality small (PC-reduce first).
    Estimates can be slightly negative; clamped at 0.
    """
    X = _as_2d(X)
    Y = _as_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError("X and Y must have the same number of rows")
    if k >= n:
        return 0.0
    if standardize:
        X = _standardize(X)
        Y = _standardize(Y)
    if Z is not None:
        Z = _as_2d(Z)
        if Z.shape[1] == 0:
            Z = None
        elif standardize:
            Z = _standardize(Z)

    # Tiny jitter breaks ties/degeneracies in the kNN radii.
    rng = np.random.default_rng(seed)
    def jit(a):
        return a + 1e-10 * rng.standard_normal(a.shape)

    X, Y = jit(X), jit(Y)
    if Z is not None:
        Z = jit(Z)
        joint = np.concatenate([X, Y, Z], axis=1)
        xz = np.concatenate([X, Z], axis=1)
        yz = np.concatenate([Y, Z], axis=1)
    else:
        joint = np.concatenate([X, Y], axis=1)
        xz, yz = X, Y

    # k-th neighbour distance in the joint space (Chebyshev), excluding self.
    eps = cKDTree(joint).query(joint, k=k + 1, p=np.inf)[0][:, k]
    radius = np.maximum(eps - 1e-12, 0.0)

    def counts(space):
        tree = cKDTree(space)
        # strictly-inside counts; subtract the self point.
        return np.array(tree.query_ball_point(space, radius, p=np.inf, return_length=True)) - 1.0

    n_xz = counts(xz)
    n_yz = counts(yz)
    if Z is not None:
        n_z = counts(Z)
        cmi = digamma(k) + np.mean(digamma(n_z + 1) - digamma(n_xz + 1) - digamma(n_yz + 1))
    else:
        cmi = digamma(k) + digamma(n) - np.mean(digamma(n_xz + 1) + digamma(n_yz + 1))
    return max(0.0, float(cmi))
