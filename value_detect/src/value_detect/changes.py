"""How often each variable changes, and whether the world obeys its designed loop.

Two plain-English confirmations for SJ, computed from a recorded trace:

* :func:`change_frequencies` — the fraction of steps on which each variable flips.
  The goal G should stand out as barely-changing (~1.5%); the noise variable W ~50%.
* :func:`mechanism_agreement` — for each causal relationship the design claims (§4),
  the observed disagreement rate next to the rate the designed noise predicts.
  Close agreement means the world behaves as described.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


def _xor_noise(*ps: float) -> float:
    """P(the XOR of independent Bernoulli(p_i) flips equals 1)."""
    q = 0.0
    for p in ps:
        q = q * (1.0 - p) + (1.0 - q) * p
    return q


def change_frequencies(frame: pd.DataFrame) -> pd.Series:
    """Fraction of consecutive steps on which each variable's value differs."""
    changed = frame.to_numpy()[1:] != frame.to_numpy()[:-1]
    return pd.Series(changed.mean(axis=0), index=list(frame.columns), name="change_rate")


def mechanism_agreement(frame: pd.DataFrame, noise: Dict[str, float]) -> pd.DataFrame:
    """Observed vs designed disagreement for each relationship the world claims.

    Each row: a plain-English relationship, the disagreement rate the designed noise
    predicts, and the disagreement rate actually observed in the trace.
    """
    B = frame["B"].to_numpy()
    S = frame["S"].to_numpy()
    A = frame["A"].to_numpy()
    E = frame["E"].to_numpy()
    G = frame["G"].to_numpy()
    Sa = frame["S_alias"].to_numpy()
    Aa = frame["A_alias"].to_numpy()
    D = frame["D"].to_numpy()

    sen, bel, act = noise["sensor_noise"], noise["belief_noise"], noise["action_noise"]
    env, ho, dis = noise["env_noise"], noise["handle_obs_noise"], noise["distractor_noise"]

    def rate(mask: np.ndarray) -> float:
        return float(np.mean(mask))

    rows = [
        # Sensing: the sensor reads the environment.
        ("Sensor line reads environment (clean copy S_alias vs E)", _xor_noise(sen), rate(Sa != E)),
        ("Sensor readout reads environment (noisy S vs E)", _xor_noise(sen, ho), rate(S != E)),
        # Deciding: the action combines belief and goal.
        ("Action line = belief XOR goal (clean A_alias)", _xor_noise(act), rate(Aa != (B ^ G))),
        ("Action readout = belief XOR goal (noisy A)", _xor_noise(act, ho), rate(A != (B ^ G))),
        # Acting: the action drives the environment forward one step.
        ("Environment responds to the action (E_next vs E XOR action line)", _xor_noise(env), rate(E[1:] != (E[:-1] ^ Aa[:-1]))),
        # Updating: belief tracks the just-sensed environment.
        ("Belief tracks the sensed environment (B_next vs sensor line)", _xor_noise(bel), rate(B[1:] != Sa[:-1])),
        # Decoys.
        ("Distractor tracks the environment (D vs E)", _xor_noise(dis), rate(D != E)),
    ]
    return pd.DataFrame(rows, columns=["relationship", "predicted_disagreement", "observed_disagreement"])


def plot_change_frequencies(freqs: pd.Series, outpath: Union[str, Path]) -> Optional[str]:
    """Bar chart of per-variable change rates. Returns the path, or None if plotting is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    order = list(freqs.index)
    colors = ["#c0392b" if v == "G" else "#95a5a6" for v in order]

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(order, freqs.values, color=colors)
    ax.set_ylabel("Fraction of steps on which the variable changes")
    ax.set_title("How often each variable changes (goal G, in red, barely moves)")
    ax.set_ylim(0, max(0.55, float(freqs.max()) * 1.1))
    for i, v in enumerate(freqs.values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)
    return str(outpath)


def text_bar_chart(freqs: pd.Series, width: int = 40) -> str:
    """A terminal-friendly bar chart, so the result is legible even without matplotlib."""
    lines: List[str] = []
    hi = max(float(freqs.max()), 1e-9)
    for name, v in freqs.items():
        bar = "#" * int(round(width * v / hi))
        lines.append(f"  {name:>8} | {bar:<{width}} {v:.3f}")
    return "\n".join(lines)
