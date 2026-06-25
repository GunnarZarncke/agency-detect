"""M2 — preprocessing + whitening.

Pipeline order (matters, README §7.3): resample to a uniform clock FIRST, z-score per
animal, then derive the whitened representation. The whitened (temporal-derivative)
representation is the primary input for discovery because it raises the effective sample
size by removing the slow GCaMP autocorrelation (README §7.2); the raw z-scored
representation is kept for reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from uad_worm.data import WormDataset


@dataclass(frozen=True)
class Processed:
    dataset_id: str
    neuron_class: List[Optional[str]]
    raw: np.ndarray        # (T, N) z-scored, uniform clock
    whitened: np.ndarray   # (T-1, N) temporal derivative of raw
    behavior: Dict[str, np.ndarray]  # feature -> (T,) on the uniform clock

    def representation(self, name: str) -> np.ndarray:
        if name == "raw":
            return self.raw
        if name == "whitened":
            return self.whitened
        raise ValueError(f"unknown representation {name!r}")


def resample_uniform(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Linear-interpolate columns onto a uniform grid spanning [t0, t1] with T points.

    Mild jitter (CV≈2.5%) makes linear interpolation sufficient (README §7.3). If `time`
    is missing/degenerate, the input is returned unchanged.
    """
    values = np.atleast_2d(values.T).T if values.ndim == 1 else values
    T = values.shape[0]
    if time is None or np.size(time) != T or T < 3:
        return values.copy()
    uniform = np.linspace(float(time[0]), float(time[-1]), T)
    out = np.empty_like(values, dtype=np.float64)
    for j in range(values.shape[1]):
        out[:, j] = np.interp(uniform, time, values[:, j])
    return out


def zscore(values: np.ndarray) -> np.ndarray:
    mu = values.mean(axis=0, keepdims=True)
    sd = values.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (values - mu) / sd


def whiten_derivative(values: np.ndarray) -> np.ndarray:
    """First temporal difference (high-pass); output length is T-1."""
    return np.diff(values, axis=0)


def preprocess(ds: WormDataset) -> Processed:
    """Resample → z-score → whiten; behavior resampled to the same uniform clock."""
    raw = zscore(resample_uniform(ds.time, ds.activity))
    whitened = whiten_derivative(raw)
    behavior = {}
    for feat, arr in ds.behavior.items():
        res = resample_uniform(ds.time, arr.reshape(-1, 1))[:, 0]
        behavior[feat] = res
    return Processed(
        dataset_id=ds.dataset_id,
        neuron_class=list(ds.neuron_class),
        raw=raw,
        whitened=whitened,
        behavior=behavior,
    )


def lag1_autocorr(values: np.ndarray) -> np.ndarray:
    """Per-column lag-1 autocorrelation (diagnostic for whitening)."""
    x = values - values.mean(axis=0, keepdims=True)
    num = np.sum(x[1:] * x[:-1], axis=0)
    den = np.sum(x * x, axis=0)
    den = np.where(den < 1e-12, 1.0, den)
    return num / den
