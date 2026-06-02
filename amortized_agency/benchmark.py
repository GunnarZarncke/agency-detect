"""Benchmark regimes for amortized vs MI comparison.

E13 (`baseline_window_breaking_point.py`) established where MI is a trusted
per-trace reference:

  * W >= 250: ARI ~1.0 on easy/med kinds, ~0.96 on hard8_complex (reliable recovery)
  * W in [125, 250): sharp collapse (~0.73 at W=125)
  * W <= 60: near chance (~0.5)

Primary amortized benchmarks should score against the **reference** regime first
(close the gap to MI where MI is correct), then use **breaking** windows as a
secondary axis (short-duration target band).
"""

from __future__ import annotations

from typing import Dict, List

# MI-trusted observation windows (E13 baseline, 5 seeds).
REFERENCE_WINDOWS: List[int] = [250, 500]

# Short-window band where amortization may beat MI (E13 consequence).
BREAKING_WINDOWS: List[int] = [250, 125, 60]

# E13 baseline_window_breaking_point.py simulates T=2000 then slices [:W].
# Shorter T changes the RNG trajectory — the first W steps are NOT a prefix
# invariant to total horizon (verified: T=500 vs T=2000 differ at W=250).
EVAL_T_STEPS: int = 2000

LEARNED_METHODS: List[str] = ["siamese", "context"]

# Frozen MI ceiling from E13 baseline (T=2000, 5 seeds). Use for gap_to_mi when
# skipping per-trace MI in routine sweeps (~300–1200× cheaper).
MI_REFERENCE_ARI: Dict[tuple[str, int], float] = {
    ("easy3_redundant", 250): 1.0,
    ("easy3_redundant", 500): 1.0,
    ("med5_rich", 250): 1.0,
    ("med5_rich", 500): 1.0,
    ("hard8_complex", 250): 0.964,
    ("hard8_complex", 500): 1.0,
    ("easy3_redundant", 125): 0.769,
    ("med5_rich", 125): 0.732,
    ("hard8_complex", 125): 0.732,
    ("easy3_redundant", 60): 0.472,
    ("med5_rich", 60): 0.600,
    ("hard8_complex", 60): 0.522,
}


def mi_reference_ari(kind: str, window: int) -> float | None:
    return MI_REFERENCE_ARI.get((kind, window))


def gap_to_mi(mi_ari: float, learned_ari: float) -> float:
    """Positive gap = learned still below MI reference."""
    return float(mi_ari - learned_ari)


def reference_row_metrics(
    row: Dict[str, float],
    *,
    kind: str | None = None,
    window: int | None = None,
    mi_reference: float | None = None,
) -> Dict[str, float]:
    """Add gap_to_mi_* fields. MI from row, frozen E13 table, or explicit mi_reference."""
    if "mi_ari" in row:
        mi = float(row["mi_ari"])
    elif mi_reference is not None:
        mi = mi_reference
    elif kind is not None and window is not None:
        ref = mi_reference_ari(kind, window)
        if ref is None:
            return {}
        mi = ref
    else:
        return {}
    out: Dict[str, float] = {"mi_reference_ari": mi}
    for m in LEARNED_METHODS:
        key = f"{m}_ari"
        if key in row:
            out[f"gap_{m}"] = gap_to_mi(mi, float(row[key]))
    return out
