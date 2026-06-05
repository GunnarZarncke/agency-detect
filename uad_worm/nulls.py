"""Autocorrelation-preserving null surrogates.

These are the arbiter for *higher-is-better* statistics (memory Δ_m, Granger/lagged-
dependence seeds): they preserve each variable's own autocorrelation while destroying
cross-variable timing, so a surviving signal is structure beyond self-decay (README
§7.2). The blanket loss uses the random-partition contrast instead (`uad_worm.blanket`).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def circular_shift(trace: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independently circularly shift each column by a random offset.

    Preserves each column's marginal and autocorrelation/spectrum; destroys cross-column
    temporal alignment.
    """
    out = np.empty_like(trace)
    T = trace.shape[0]
    for j in range(trace.shape[1]):
        out[:, j] = np.roll(trace[:, j], int(rng.integers(1, T)))
    return out


def phase_randomize(trace: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-randomized surrogate per column (preserves power spectrum)."""
    out = np.empty_like(trace, dtype=np.float64)
    T = trace.shape[0]
    freqs = np.fft.rfft(trace, axis=0)
    n_freq = freqs.shape[0]
    # Keep DC (bin 0) and, for even T, the Nyquist bin (last) real so that irfft
    # preserves the power spectrum exactly; randomize the interior phases only.
    hi = n_freq - 1 if T % 2 == 0 else n_freq
    for j in range(trace.shape[1]):
        mag = np.abs(freqs[:, j])
        rand_phases = np.angle(freqs[:, j])
        rand_phases[1:hi] = rng.uniform(-np.pi, np.pi, size=hi - 1)
        new = mag * np.exp(1j * rand_phases)
        out[:, j] = np.fft.irfft(new, n=T)
    return out


def null_distribution(
    stat_fn: Callable[[np.ndarray], float],
    trace: np.ndarray,
    *,
    n_perm: int = 200,
    method: str = "circshift",
    seed: int = 0,
) -> np.ndarray:
    """Null distribution of a scalar statistic under a surrogate method."""
    rng = np.random.default_rng(seed)
    surrogate = {"circshift": circular_shift, "phase": phase_randomize}[method]
    return np.array([stat_fn(surrogate(trace, rng)) for _ in range(n_perm)])
