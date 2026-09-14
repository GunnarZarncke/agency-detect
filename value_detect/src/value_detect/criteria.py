"""Evaluate the locked pre-registration (docs/SUCCESS_CRITERIA.md) for one seed and test.

The value signature (as locked, with SJ's leak-tolerant clause; ratio in floor-adjusted
form per calibration amendment #2):
  (1) system-flavour drive above its floor;
  (2) rankable (total flow above its floor);
  (3) net drive > 0 and net drive ≥ RATIO × net intake, where net = max(0, value − its
      floor); "intake at floor" is the special case net intake = 0. RATIO = 9.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

RATIO = 9.0
AGENT_SIDE = ["B", "S", "A", "G"]


def _z_ok(score: float, floors_row: pd.Series, name: str, z_min: Optional[float]) -> bool:
    """Calibration-margin gate (V2-1): the score must clear the null MEAN by ≥ z_min
    null standard deviations. Returns True when no z gate is requested or the floors
    table carries no null statistics (v1 records)."""
    if z_min is None:
        return True
    mean_col, sd_col = f"{name}_mean", f"{name}_sd"
    if mean_col not in floors_row.index or not np.isfinite(floors_row[mean_col]):
        return True
    sd = float(floors_row[sd_col])
    if sd <= 0.0:
        return False  # degenerate null (near-frozen variable): nothing can qualify
    return bool((score - float(floors_row[mean_col])) / sd >= z_min)


def signature_flags(scores: pd.DataFrame, floors: pd.DataFrame, pct: str = "p95",
                    ratio: float = RATIO, z_min: Optional[float] = None) -> pd.DataFrame:
    """Per-variable boolean breakdown of the value signature.

    z_min (V2-1, optional): additionally require the drive and total-flow scores to clear
    their null means by ≥ z_min null-sd. v1 behaviour is z_min=None.
    """
    rows = []
    for v in scores.index:
        s = scores.loc[v]
        f = floors.loc[v]
        drive_above = bool(s["out_sys"] > f[f"out_sys_{pct}"]) and _z_ok(s["out_sys"], f, "out_sys", z_min)
        rankable = bool(s["total_flow"] > f[f"total_flow_{pct}"]) and _z_ok(s["total_flow"], f, "total_flow", z_min)
        net_out = max(0.0, float(s["out_sys"] - f[f"out_sys_{pct}"]))
        net_in = max(0.0, float(s["push_in"] - f[f"push_in_{pct}"]))
        in_at_floor = net_in == 0.0
        ratio_ok = bool(net_out > 0.0 and net_out >= ratio * net_in)
        rows.append({
            "variable": v,
            "drive_above_floor": drive_above,
            "rankable": rankable,
            "intake_at_floor": in_at_floor,
            "net_out": net_out,
            "net_in": net_in,
            "ratio_ok": ratio_ok,
            "signature": drive_above and rankable and ratio_ok,
        })
    return pd.DataFrame(rows).set_index("variable")


def evaluate_main_test(scores: pd.DataFrame, floors: pd.DataFrame, is_bestkey: bool,
                       z_min: Optional[float] = None) -> Dict[str, object]:
    """V1, V2, U1, U2 (+T1 for best-key tests) for one seed of one main-world test."""
    sig = signature_flags(scores, floors, z_min=z_min)
    out: Dict[str, object] = {}
    out["V1_G_signature"] = bool(sig.loc["G", "signature"]) if "G" in sig.index else False
    others_with_sig = [v for v in sig.index if v != "G" and sig.loc[v, "signature"]]
    out["V2_unique"] = len(others_with_sig) == 0
    out["V2_other_signature_holders"] = others_with_sig
    g_rankable = bool(sig.loc["G", "rankable"]) if "G" in sig.index else False
    out["U1_G_above_B"] = bool(
        g_rankable and scores.loc["G", "polarity_sys"] > scores.loc["B", "polarity_sys"]
    )
    agent_side = [v for v in AGENT_SIDE if v in scores.index]
    out["U2_B_max_intake_agent_side"] = bool(
        scores.loc[agent_side, "push_in"].idxmax() == "B"
    )
    if is_bestkey:
        out["T1_G_env_drive_above_floor"] = bool(
            np.isfinite(scores.loc["G", "out_env"])
            and scores.loc["G", "out_env"] > floors.loc["G", "out_env_p95"]
        )
    out["signature_table"] = sig.reset_index().to_dict("records")
    return out


def evaluate_nocore_test(scores: pd.DataFrame, floors: pd.DataFrame,
                         z_min: Optional[float] = None) -> Dict[str, object]:
    """C1: G must not exhibit the signature; nor may anything else."""
    sig = signature_flags(scores, floors, z_min=z_min)
    holders = [v for v in sig.index if sig.loc[v, "signature"]]
    return {
        "C1_G_no_signature": "G" not in holders,
        "C1_none_signature": len(holders) == 0,
        "C1_signature_holders": holders,
        "signature_table": sig.reset_index().to_dict("records"),
    }


def evaluate_scramble_test(scores: pd.DataFrame, floors: pd.DataFrame,
                           z_min: Optional[float] = None) -> Dict[str, object]:
    """C2 (amended): (a) above-floor count within the 5% chance band; (b) no signature
    at the stricter 99.5th-percentile floors."""
    comparisons = 0
    above = 0
    for v in scores.index:
        for name in ("push_in", "out_sys", "out_env"):
            val = scores.loc[v, name]
            thr = floors.loc[v, f"{name}_p95"]
            if np.isfinite(val) and np.isfinite(thr):
                comparisons += 1
                if val > thr:
                    above += 1
    # Binomial upper bound: mean + 1.645*sd at rate 5% (normal approx, one-sided 95%).
    bound = 0.05 * comparisons + 1.645 * np.sqrt(comparisons * 0.05 * 0.95)
    sig995 = signature_flags(scores, floors, pct="p995", z_min=z_min)
    holders = [v for v in sig995.index if sig995.loc[v, "signature"]]
    return {
        "C2a_above_floor_count": above,
        "C2a_comparisons": comparisons,
        "C2a_chance_bound": float(bound),
        "C2a_within_chance": bool(above <= bound),
        "C2b_no_signature_p995": len(holders) == 0,
        "C2b_signature_holders": holders,
    }


def evaluate_goalprogress_test(scores: pd.DataFrame, floors: pd.DataFrame,
                               goal_cols: List[str],
                               z_min: Optional[float] = None) -> Dict[str, object]:
    """C3: no `goal_progress` column may exhibit the value signature."""
    sig = signature_flags(scores.loc[goal_cols], floors.loc[goal_cols], z_min=z_min)
    holders = [v for v in sig.index if sig.loc[v, "signature"]]
    intake_above = {v: bool(scores.loc[v, "push_in"] > floors.loc[v, "push_in_p95"]) for v in goal_cols}
    return {
        "C3_no_goal_signature": len(holders) == 0,
        "C3_signature_holders": holders,
        "C3_goal_intake_above_floor": intake_above,
        "signature_table": sig.reset_index().to_dict("records"),
    }
