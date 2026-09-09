"""Noise floors: circular-shift nulls that mirror each convention's exact procedure.

The one engine both real scores and floors go through is :class:`ConventionScorer`, so a
null is guaranteed to be measured the same way as the score it gates (including the
best-key maximisation). Label arrays are cached per frame; a circular shift of a variable
is a roll of its cached labels, so each shifted re-measurement costs only a recount.

Note on labels: caches are built once from full columns. For multi-valued variables a
sliced window could in principle miss a rare top label (harmless here: the six main tests
are binary, and the C3 contrast uses this same engine for scores AND floors, so the
comparison stays internally consistent).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .directed import ALPHA, _cmi_labeled, _labels

CONVENTIONS = ("pairwise", "fused", "fused_bestkey", "grown_keys")


def _dense_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dense joint labels of two label arrays (matches _labels(column_stack))."""
    m = a.astype(np.int64) * (int(b.max()) + 1) + b
    _, inv = np.unique(m, return_inverse=True)
    return inv.astype(np.int64)


class ConventionScorer:
    """Score any variable of a frame under one convention, with label caches.

    ``score_variable(v, vlab_override=...)`` lets the floor engine pass a rolled copy of
    v's labels while every other variable (targets, keys, fused rests) stays fixed —
    the addendum's "shift the candidate against the fixed remainder".
    """

    def __init__(self, frame: pd.DataFrame, convention: str, env_var: Optional[str] = "E",
                 lag: int = 1, alpha: float = ALPHA, key_depth: int = 2):
        assert convention in CONVENTIONS, convention
        self.cols: List[str] = list(frame.columns)
        self.convention = convention
        self.env_var = env_var if (env_var in self.cols if env_var else False) else None
        self.lag = lag
        self.alpha = alpha
        self.key_depth = key_depth  # grown_keys only: maximum keys grown per flow.
        self.n = len(frame)
        self.labels: Dict[str, np.ndarray] = {c: _labels(frame[c].to_numpy()) for c in self.cols}
        self._pair_cache: Dict[tuple, np.ndarray] = {}
        self._rest_cache: Dict[str, np.ndarray] = {}
        self._chain_cache: Dict[tuple, np.ndarray] = {}

    # ---- cached label helpers ----

    def _pair(self, j: str, k: str) -> np.ndarray:
        key = (j, k)
        if key not in self._pair_cache:
            lag = self.lag
            self._pair_cache[key] = _dense_pair(self.labels[j][:-lag], self.labels[k][:-lag])
        return self._pair_cache[key]

    def _rest(self, v: str) -> np.ndarray:
        if v not in self._rest_cache:
            others = [c for c in self.cols if c != v]
            mat = np.column_stack([self.labels[c] for c in others])
            self._rest_cache[v] = _labels(mat)
        return self._rest_cache[v]

    # ---- directed estimates from labels ----

    def _te(self, xl_full: np.ndarray, yl_full: np.ndarray, zl_prev: Optional[np.ndarray] = None) -> float:
        """I(y_next; x_now | y_now [, z_now]) from full-length label arrays."""
        lag = self.lag
        y_next = yl_full[lag:]
        x_prev = xl_full[:-lag]
        if zl_prev is None:
            z = yl_full[:-lag]
        else:
            z = zl_prev
        return _cmi_labeled(y_next, x_prev, z, self.alpha)

    def _te_keyed(self, xl_full: np.ndarray, j: str, keys: List[str]) -> float:
        """Best over no-key and each single key, target j (mirrors _best_key_te)."""
        best = self._te(xl_full, self.labels[j])
        for k in keys:
            v = self._te(xl_full, self.labels[j], self._pair(j, k))
            if v > best:
                best = v
        return best

    def _chain(self, j: str, chosen: tuple) -> np.ndarray:
        """Dense conditioning labels for (target j's past + the chosen key chain)."""
        key = (j,) + chosen
        if key not in self._chain_cache:
            if len(chosen) == 1:
                self._chain_cache[key] = self._pair(j, chosen[0])
            else:
                prev = self._chain(j, chosen[:-1])
                lag = self.lag
                self._chain_cache[key] = _dense_pair(prev, self.labels[chosen[-1]][:-lag])
        return self._chain_cache[key]

    def _te_grown(self, xl_full: np.ndarray, j: str, keys: List[str]) -> float:
        """Greedy key growth (V2-2): start keyless; repeatedly add the single key with
        the best reading; up to ``key_depth`` keys. Returns the max over all stages, so a
        screening (mediator) key can never lower the result below an earlier stage."""
        best = self._te(xl_full, self.labels[j])
        chosen: tuple = ()
        pool = list(keys)
        for _ in range(self.key_depth):
            stage_best, stage_key = -1.0, None
            for k in pool:
                v = self._te(xl_full, self.labels[j], self._chain(j, chosen + (k,)))
                if v > stage_best:
                    stage_best, stage_key = v, k
            if stage_key is None:
                break
            chosen = chosen + (stage_key,)
            pool.remove(stage_key)
            if stage_best > best:
                best = stage_best
        return best

    # ---- public scoring ----

    def score_variable(self, v: str, vlab_override: Optional[np.ndarray] = None) -> Dict[str, float]:
        vl = self.labels[v] if vlab_override is None else vlab_override
        lag = self.lag
        others = [c for c in self.cols if c != v]

        if self.convention == "pairwise":
            push_in = sum(self._te(self.labels[j], vl) for j in others)
            out_sys = sum(self._te(vl, self.labels[j]) for j in others)
        elif self.convention == "fused":
            rest = self._rest(v)
            push_in = _cmi_labeled(vl[lag:], rest[:-lag], vl[:-lag], self.alpha)
            out_sys = _cmi_labeled(rest[lag:], vl[:-lag], rest[:-lag], self.alpha)
        elif self.convention == "fused_bestkey":
            rest = self._rest(v)
            push_in = _cmi_labeled(vl[lag:], rest[:-lag], vl[:-lag], self.alpha)
            out_sys = sum(self._te_keyed(vl, j, [k for k in self.cols if k not in (v, j)]) for j in others)
        else:  # grown_keys: fused intake; greedily-grown conditioning outbound.
            rest = self._rest(v)
            push_in = _cmi_labeled(vl[lag:], rest[:-lag], vl[:-lag], self.alpha)
            out_sys = sum(self._te_grown(vl, j, [k for k in self.cols if k not in (v, j)]) for j in others)

        if self.env_var is None or v == self.env_var:
            out_env = float("nan")
        elif self.convention == "fused_bestkey":
            out_env = self._te_keyed(vl, self.env_var, [k for k in self.cols if k not in (v, self.env_var)])
        elif self.convention == "grown_keys":
            out_env = self._te_grown(vl, self.env_var, [k for k in self.cols if k not in (v, self.env_var)])
        else:
            out_env = self._te(vl, self.labels[self.env_var])

        return {"push_in": push_in, "out_sys": out_sys, "out_env": out_env}

    def score_all(self) -> pd.DataFrame:
        rows = []
        for v in self.cols:
            s = self.score_variable(v)
            out, inn = s["out_sys"], s["push_in"]
            denom = out + inn
            pol = (out - inn) / denom if denom > 0 else 0.0
            oe = s["out_env"]
            pol_env = ((oe - inn) / (oe + inn)) if np.isfinite(oe) and (oe + inn) > 0 else (
                float("nan") if not np.isfinite(oe) else 0.0)
            rows.append({"variable": v, "push_in": inn, "out_sys": out, "out_env": oe,
                         "raw_sys": out - inn,
                         "raw_env": (oe - inn) if np.isfinite(oe) else float("nan"),
                         "polarity_sys": pol, "polarity_env": pol_env,
                         "total_flow": out + inn})
        return pd.DataFrame(rows).set_index("variable").sort_values("polarity_sys", ascending=False)


def shift_null_samples(
    scorer: ConventionScorer,
    n_shifts: int,
    seed: int = 0,
    variables: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Raw null samples (long form: variable, shift, push_in, out_sys, out_env).

    Uses the identical offset stream as :func:`shift_null_floors` for the same seed, so
    pooled re-analysis reproduces the sweep's own nulls exactly.
    """
    rng = np.random.default_rng(seed)
    variables = variables or scorer.cols
    lo, hi = scorer.lag + 1, scorer.n - scorer.lag - 1
    rows = []
    for v in variables:
        base = scorer.labels[v]
        for i in range(n_shifts):
            k = int(rng.integers(lo, hi))
            s = scorer.score_variable(v, vlab_override=np.roll(base, k))
            rows.append({"variable": v, "shift": i, "push_in": s["push_in"],
                         "out_sys": s["out_sys"], "out_env": s["out_env"],
                         "total_flow": s["push_in"] + s["out_sys"]})
    return pd.DataFrame(rows)


def _roll_sampler(scorer: ConventionScorer, v: str, rng: np.random.Generator):
    """Null candidate columns by circular shift (preserves the sequence exactly)."""
    base = scorer.labels[v]
    lo, hi = scorer.lag + 1, scorer.n - scorer.lag - 1
    while True:
        yield np.roll(base, int(rng.integers(lo, hi)))


def _transition_sampler(scorer: ConventionScorer, v: str, rng: np.random.Generator):
    """Null candidate columns regenerated from the column's own one-step transition
    statistics (Laplace 0.5): preserves the variable's dynamics in distribution while
    varying which and when transitions occur — the second null family (V2-1)."""
    base = scorer.labels[v]
    card = int(base.max()) + 1
    counts = np.full((card, card), 0.5)
    np.add.at(counts, (base[:-1], base[1:]), 1.0)
    probs = counts / counts.sum(axis=1, keepdims=True)
    cum = np.cumsum(probs, axis=1)
    n = scorer.n
    while True:
        out = np.empty(n, dtype=np.int64)
        out[0] = base[0]
        u = rng.random(n)
        for t in range(1, n):
            out[t] = int(np.searchsorted(cum[out[t - 1]], u[t]))
        yield out


SAMPLERS = {"roll": _roll_sampler, "transition": _transition_sampler}


def null_floors(
    scorer: ConventionScorer,
    n_shifts: int,
    seed: int = 0,
    variables: Optional[List[str]] = None,
    percentiles=(95.0, 99.5),
    sampler: str = "roll",
) -> pd.DataFrame:
    """Per-variable null statistics for push_in / out_sys / out_env / total_flow.

    Each null sample replaces the candidate's column (by the chosen sampler) against the
    fixed remainder and re-runs the identical scoring. Emits percentiles AND the null
    mean/sd per score, so calibration-margin (z) gates can be evaluated downstream.
    """
    rng = np.random.default_rng(seed)
    variables = variables or scorer.cols
    make = SAMPLERS[sampler]
    rows = []
    for v in variables:
        gen = make(scorer, v, rng)
        samples = {"push_in": [], "out_sys": [], "out_env": [], "total_flow": []}
        for _ in range(n_shifts):
            s = scorer.score_variable(v, vlab_override=next(gen))
            samples["push_in"].append(s["push_in"])
            samples["out_sys"].append(s["out_sys"])
            samples["out_env"].append(s["out_env"])
            samples["total_flow"].append(s["push_in"] + s["out_sys"])
        row = {"variable": v, "n_shifts": n_shifts, "sampler": sampler}
        for name, vals in samples.items():
            arr = np.asarray(vals, dtype=float)
            finite = np.isfinite(arr).any()
            for p in percentiles:
                tag = f"{name}_p{str(p).replace('.5', '5').replace('.0', '')}"
                row[tag] = float(np.nanpercentile(arr, p)) if finite else float("nan")
            row[f"{name}_mean"] = float(np.nanmean(arr)) if finite else float("nan")
            row[f"{name}_sd"] = float(np.nanstd(arr)) if finite else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("variable")


def shift_null_floors(
    scorer: ConventionScorer,
    n_shifts: int,
    seed: int = 0,
    variables: Optional[List[str]] = None,
    percentiles=(95.0, 99.5),
) -> pd.DataFrame:
    """V1-compatible wrapper: circular-shift nulls (identical RNG stream as v1)."""
    return null_floors(scorer, n_shifts, seed=seed, variables=variables,
                       percentiles=percentiles, sampler="roll")
