"""The directional scorers: push-in, push-out (two flavours), and polarity per variable.

Two conventions run in parallel as co-equal tests (SJ, 2026-08-10, docs/DECISIONS.md) —
their divergences are themselves findings about how the tool works:

* **Pairwise** (:func:`score_trace`): push-in = sum of one-at-a-time flows from each other
  variable into the candidate; push-out (system flavour) = the mirror sum. Structurally
  blind to XOR-style joint determination (the cipher effect, results/chunk4/DIAGNOSIS.md).
* **Fused + best-key** (:func:`score_trace_fused_bestkey`) — a hybrid, named for its parts:
  - push-in = the **fused mega-state** measurement (one joint reading from the fused state
    of all others). This is the ORIGINAL Option-A cross-check from the options memo,
    unchanged and promoted to co-equal; it predates the cipher diagnosis — and is what
    caught it. It cannot be cipher-blinded.
  - push-out = the **best-key** measurement: for each flow, the best reading over "no key"
    and each single decryption key. NEW, designed during the cipher diagnosis (the fused
    version of push-out needs far more data than our runs have). The same automatic rule
    for every variable avoids mediator-conditioning (a mediator key only lowers the
    reading, so the max skips it) and does not resurrect decoy witness-flows.

Shared across conventions:

* The **environment flavour** of push-out targets the designated environment variable,
  conditioning on its own past — never forced through the action (mediation; Chunk 3).
  Not applicable for the environment variable itself.
* **Polarity** = (out − in) / (out + in), in [−1, 1]: the fraction of a variable's traffic
  that is outbound; within-convention ranking statistic, raw difference alongside.
  Chunk 5 adds per-convention noise floors to gate who is rankable at all.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .directed import ALPHA, transfer_entropy


def _polarity(out: float, inn: float) -> float:
    denom = out + inn
    if denom <= 0.0:
        return 0.0
    return (out - inn) / denom


def score_trace(
    frame: pd.DataFrame,
    env_var: str = "E",
    lag: int = 1,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Score every variable in the trace; rows sorted by system-flavour polarity.

    Columns: push_in, out_sys, out_env, polarity_sys, polarity_env, raw_sys, raw_env,
    total_flow. out_env / *_env are NaN for the environment variable itself.
    """
    cols: List[str] = list(frame.columns)
    arrays = {c: frame[c].to_numpy() for c in cols}
    rows = []
    for x in cols:
        others = [c for c in cols if c != x]
        push_in = sum(transfer_entropy(arrays[j], arrays[x], alpha=alpha, lag=lag) for j in others)
        out_sys = sum(transfer_entropy(arrays[x], arrays[j], alpha=alpha, lag=lag) for j in others)
        if x == env_var or env_var not in cols:
            out_env = np.nan
        else:
            out_env = transfer_entropy(arrays[x], arrays[env_var], alpha=alpha, lag=lag)
        rows.append(
            {
                "variable": x,
                "push_in": push_in,
                "out_sys": out_sys,
                "out_env": out_env,
                "raw_sys": out_sys - push_in,
                "raw_env": (out_env - push_in) if np.isfinite(out_env) else np.nan,
                "polarity_sys": _polarity(out_sys, push_in),
                "polarity_env": _polarity(out_env, push_in) if np.isfinite(out_env) else np.nan,
                "total_flow": out_sys + push_in,
            }
        )
    table = pd.DataFrame(rows).set_index("variable")
    return table.sort_values("polarity_sys", ascending=False)


def score_trace_fused(
    frame: pd.DataFrame,
    env_var: str = "E",
    lag: int = 1,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Fused mega-state convention: BOTH directions measured against the fused joint
    state of all other variables.

    * push-in = I(X_next ; Rest_now | X_now) — identical to :func:`push_in_megastate`.
    * push-out (system) = I(Rest_next ; X_now | Rest_now) — the fused outbound. Because it
      conditions on the joint past of everything else, it automatically screens duplicates
      and mediators (a variable gets no credit for influence that something else already
      carries). Statistically hungry: with V variables the conditioning tracks 2^(V-1)
      states, so this convention needs much longer runs than the others (see DECISIONS.md
      2026-08-10) and collapses computationally in large worlds.
    * push-out (environment flavour) = the plain single-target reading I(E_next; X_now |
      E_now), same as the pairwise convention (no fusion is involved for a single target;
      no keys are added — that is the best-key convention's device).

    Same columns and sorting as :func:`score_trace`.
    """
    cols: List[str] = list(frame.columns)
    arrays = {c: frame[c].to_numpy() for c in cols}
    rows = []
    for x in cols:
        others = [c for c in cols if c != x]
        rest = frame[others].to_numpy()
        push_in = transfer_entropy(rest, arrays[x], alpha=alpha, lag=lag)
        out_sys = transfer_entropy(arrays[x], rest, alpha=alpha, lag=lag)
        if x == env_var or env_var not in cols:
            out_env = np.nan
        else:
            out_env = transfer_entropy(arrays[x], arrays[env_var], alpha=alpha, lag=lag)
        rows.append(
            {
                "variable": x,
                "push_in": push_in,
                "out_sys": out_sys,
                "out_env": out_env,
                "raw_sys": out_sys - push_in,
                "raw_env": (out_env - push_in) if np.isfinite(out_env) else np.nan,
                "polarity_sys": _polarity(out_sys, push_in),
                "polarity_env": _polarity(out_env, push_in) if np.isfinite(out_env) else np.nan,
                "total_flow": out_sys + push_in,
            }
        )
    table = pd.DataFrame(rows).set_index("variable")
    return table.sort_values("polarity_sys", ascending=False)


def _best_key_te(
    arrays: dict,
    source: str,
    target: str,
    keys: List[str],
    alpha: float,
    lag: int,
) -> float:
    """Directed flow source -> target, decrypted: best over no-key and each single key."""
    best = transfer_entropy(arrays[source], arrays[target], alpha=alpha, lag=lag)
    for k in keys:
        v = transfer_entropy(arrays[source], arrays[target], cond=arrays[k], alpha=alpha, lag=lag)
        if v > best:
            best = v
    return best


def score_trace_fused_bestkey(
    frame: pd.DataFrame,
    env_var: str = "E",
    lag: int = 1,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Fused + best-key convention: fused mega-state intake (the original cross-check,
    unchanged); best-single-key outbound (post-diagnosis addition). Same columns and
    sorting as :func:`score_trace` so the two conventions compare cell for cell."""
    cols: List[str] = list(frame.columns)
    arrays = {c: frame[c].to_numpy() for c in cols}
    rows = []
    for x in cols:
        others = [c for c in cols if c != x]
        rest = frame[others].to_numpy()
        push_in = transfer_entropy(rest, arrays[x], alpha=alpha, lag=lag)
        out_sys = sum(
            _best_key_te(arrays, x, j, [k for k in cols if k not in (x, j)], alpha, lag)
            for j in others
        )
        if x == env_var or env_var not in cols:
            out_env = np.nan
        else:
            out_env = _best_key_te(arrays, x, env_var, [k for k in cols if k not in (x, env_var)], alpha, lag)
        rows.append(
            {
                "variable": x,
                "push_in": push_in,
                "out_sys": out_sys,
                "out_env": out_env,
                "raw_sys": out_sys - push_in,
                "raw_env": (out_env - push_in) if np.isfinite(out_env) else np.nan,
                "polarity_sys": _polarity(out_sys, push_in),
                "polarity_env": _polarity(out_env, push_in) if np.isfinite(out_env) else np.nan,
                "total_flow": out_sys + push_in,
            }
        )
    table = pd.DataFrame(rows).set_index("variable")
    return table.sort_values("polarity_sys", ascending=False)


def push_in_megastate(
    frame: pd.DataFrame,
    lag: int = 1,
    alpha: float = ALPHA,
) -> pd.Series:
    """Cross-check: push-in with Rest as one fused joint state (bias-heavy; long runs only).

    For each variable X: directed flow from the joint state of all others into X.
    """
    cols: List[str] = list(frame.columns)
    out = {}
    for x in cols:
        others = [c for c in cols if c != x]
        rest = frame[others].to_numpy()
        out[x] = transfer_entropy(rest, frame[x].to_numpy(), alpha=alpha, lag=lag)
    return pd.Series(out, name="push_in_megastate")


def drop_aliases(frame: pd.DataFrame, alias_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Configuration (b): the same trace as recorded by a world with no perfect logs.

    The alias columns are causally inert as observations (nothing downstream reads the
    recorded matrix), so dropping them from the analysis is exactly equivalent to a world
    that never recorded them.
    """
    alias_cols = alias_cols or ["S_alias", "A_alias"]
    return frame.drop(columns=[c for c in alias_cols if c in frame.columns])
