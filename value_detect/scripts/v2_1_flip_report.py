#!/usr/bin/env python3
"""V2-1 flip-report: which floor fix passes every acceptance check?

Candidate configurations: null family ∈ {roll, transition} × z_min ∈ {None, 3, 4, 5}.
For each: re-evaluate (from stored scores + recomputed floors — no new measurements)

  * V1 main-world criteria (pairwise + fused+best-key; fused via pooled roll stats),
  * C1 (no-core), C2 (scramble), C3 (`goal_progress`),
  * the calibration world (slow value MUST sign; slow meter and frozen variable MUST NOT).

Acceptance (from DECISIONS.md 2026-08-10): slow-true-value signs ≥80%; meters never sign
(≤5% seeds); scramble calibration holds; C3 ≥80%; no V1 main-world PASS flips to FAIL.
Output: results/v2_1/FLIP_REPORT.md.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from value_detect.criteria import (
    evaluate_goalprogress_test,
    evaluate_main_test,
    evaluate_nocore_test,
    evaluate_scramble_test,
    signature_flags,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNK5 = PROJECT_ROOT / "results" / "chunk5"
V21 = PROJECT_ROOT / "results" / "v2_1"

CONVS = ("pairwise", "fused_bestkey")
CONFIGS = ("asis", "noalias")
Z_GRID = (None, 3.0, 4.0, 5.0)
CALIB_VARS = ["G", "M_slow", "F_frozen", "W", "B"]
THRESH = {"V1": 0.8, "V2": 0.8, "U1": 0.9, "U2": 0.8, "T1": 0.8,
          "C1_G": 0.8, "C1_none": 0.8, "C2a": 0.8, "C2b": 0.8, "C3": 0.8}
# V1 FINAL verdict pass/fail map for flip detection (fused+best-key & fused passed all).
V1_PASSES = {f"{cfg}_{conv}": ["V1", "V2", "U1", "U2"] + (["T1"] if conv == "fused_bestkey" else [])
             for cfg in CONFIGS for conv in ("fused_bestkey",)}


def df_from(records) -> pd.DataFrame:
    return pd.DataFrame(records).set_index("variable")


def load_json(path: Path):
    return json.loads(path.read_text())


def rate(vals):
    vals = list(vals)
    return sum(bool(v) for v in vals) / len(vals) if vals else float("nan")


def pooled_fused_stats():
    """condition -> config -> floors DF with p95/p995/mean/sd from fusednulls samples."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for p in CHUNK5.glob("fusednulls_*_seed*.json"):
        d = load_json(p)
        for config, rows in d["samples"].items():
            for r in rows:
                for name in ("push_in", "out_sys", "out_env", "total_flow"):
                    if r[name] is not None:
                        acc[d["condition"]][config][r["variable"]][name].append(r[name])
    out = {}
    for cond, per_cfg in acc.items():
        out[cond] = {}
        for config, per_var in per_cfg.items():
            rows = []
            for v, per_name in per_var.items():
                row = {"variable": v}
                for name, vals in per_name.items():
                    a = np.asarray(vals, dtype=float)
                    row[f"{name}_p95"] = float(np.nanpercentile(a, 95))
                    row[f"{name}_p995"] = float(np.nanpercentile(a, 99.5))
                    row[f"{name}_mean"] = float(np.nanmean(a))
                    row[f"{name}_sd"] = float(np.nanstd(a))
                rows.append(row)
            out[cond][config] = pd.DataFrame(rows).set_index("variable")
    return out


def main() -> None:
    seeds = sorted(load_json(p)["seed"] for p in CHUNK5.glob("main_seed*.json"))
    n_seeds = len(seeds)
    fused_pooled = pooled_fused_stats()

    # Preload stored scores.
    stored = {cond: {s: load_json(CHUNK5 / f"{cond}_seed{s}.json") for s in seeds}
              for cond in ("main", "nocore", "scramble")}
    stored_c3 = {s: load_json(CHUNK5 / f"c3_seed{s}.json") for s in seeds}

    L = [f"# V2-1 flip-report — {n_seeds} seeds, candidate floor fixes\n"]
    stream_ok = all(
        sum(load_json(V21 / f"floorstats_roll_{cond}_seed{s}.json")["p95_mismatches"].values()) == 0
        for cond in ("main", "nocore", "scramble") for s in seeds
    )
    L.append(f"Stream compatibility (recomputed roll p95 == stored sweep p95, all units): "
             f"**{'OK' if stream_ok else 'MISMATCH — investigate before trusting anything below'}**\n")

    summary_rows = []
    for sampler in ("roll", "transition"):
        floorstats = {cond: {s: load_json(V21 / f"floorstats_{sampler}_{cond}_seed{s}.json")
                             for s in seeds} for cond in ("main", "nocore", "scramble")}
        c3stats = {s: load_json(V21 / f"c3floorstats_{sampler}_seed{s}.json") for s in seeds}
        calib = {s: load_json(V21 / f"calib_{sampler}_seed{s}.json") for s in seeds}

        for z in Z_GRID:
            tag = f"{sampler} / z={'off' if z is None else int(z)}"
            # --- V1 main + controls, pairwise & bestkey ---
            crit_rates = {}
            flips = []
            for config in CONFIGS:
                for conv in CONVS:
                    key = f"{config}_{conv}"
                    evs, c1s, c2s = [], [], []
                    for s in seeds:
                        sc = df_from(stored["main"][s]["tests"][key]["scores"])
                        fl = df_from(floorstats["main"][s]["tests"][key])
                        evs.append(evaluate_main_test(sc, fl, is_bestkey=(conv == "fused_bestkey"), z_min=z))
                        sc = df_from(stored["nocore"][s]["tests"][key]["scores"])
                        fl = df_from(floorstats["nocore"][s]["tests"][key])
                        c1s.append(evaluate_nocore_test(sc, fl, z_min=z))
                        sc = df_from(stored["scramble"][s]["tests"][key]["scores"])
                        fl = df_from(floorstats["scramble"][s]["tests"][key])
                        c2s.append(evaluate_scramble_test(sc, fl, z_min=z))
                    r = {
                        "V1": rate(e["V1_G_signature"] for e in evs),
                        "V2": rate(e["V2_unique"] for e in evs),
                        "U1": rate(e["U1_G_above_B"] for e in evs),
                        "U2": rate(e["U2_B_max_intake_agent_side"] for e in evs),
                        "C1_G": rate(e["C1_G_no_signature"] for e in c1s),
                        "C1_none": rate(e["C1_none_signature"] for e in c1s),
                        "C2a": rate(e["C2a_within_chance"] for e in c2s),
                        "C2b": rate(e["C2b_no_signature_p995"] for e in c2s),
                    }
                    if conv == "fused_bestkey":
                        r["T1"] = rate(e.get("T1_G_env_drive_above_floor") for e in evs)
                    crit_rates[key] = r
                    for c in V1_PASSES.get(key, []):
                        if r.get(c, 1.0) < THRESH[c]:
                            flips.append(f"{key}:{c}={r[c]*100:.0f}%")
            # Fused (roll pooled stats only).
            if sampler == "roll":
                for config in CONFIGS:
                    key = f"{config}_fused"
                    evs = [evaluate_main_test(df_from(stored["main"][s]["tests"][key]["scores"]),
                                              fused_pooled["main"][config], is_bestkey=False, z_min=z)
                           for s in seeds]
                    r = {"V1": rate(e["V1_G_signature"] for e in evs),
                         "V2": rate(e["V2_unique"] for e in evs)}
                    crit_rates[key] = r
                    for c in ("V1", "V2"):
                        if r[c] < THRESH[c]:
                            flips.append(f"{key}:{c}={r[c]*100:.0f}%")

            # --- C3 ---
            c3_rates = {}
            for conv in CONVS:
                oks = []
                for s in seeds:
                    sc = df_from(stored_c3[s]["tests"][conv]["scores"])
                    fl = df_from(c3stats[s]["tests"][conv])
                    ev = evaluate_goalprogress_test(sc, fl, stored_c3[s]["goal_cols"], z_min=z)
                    oks.append(ev["C3_no_goal_signature"])
                c3_rates[conv] = rate(oks)

            # --- calibration world ---
            g_sign, m_steal, f_steal = defaultdict(list), defaultdict(list), defaultdict(list)
            for s in seeds:
                for conv in CONVS:
                    sc = df_from(calib[s]["tests"][conv]["scores"])
                    fl = df_from(calib[s]["tests"][conv]["floors"])
                    sig = signature_flags(sc.loc[CALIB_VARS], fl, z_min=z)
                    g_sign[conv].append(bool(sig.loc["G", "signature"]))
                    m_steal[conv].append(bool(sig.loc["M_slow", "signature"]))
                    f_steal[conv].append(bool(sig.loc["F_frozen", "signature"]))

            # --- acceptance ---
            bk = "fused_bestkey"
            accept = (
                rate(g_sign[bk]) >= 0.8
                and rate(m_steal[bk]) <= 0.05 and rate(f_steal[bk]) <= 0.05
                and rate(m_steal["pairwise"]) <= 0.05 and rate(f_steal["pairwise"]) <= 0.05
                and c3_rates[bk] >= 0.8 and c3_rates["pairwise"] >= 0.8
                and all(crit_rates[f"{cfg}_{bk}"]["C2a"] >= 0.8 and crit_rates[f"{cfg}_{bk}"]["C2b"] >= 0.8
                        for cfg in CONFIGS)
                and not flips
            )
            summary_rows.append({
                "config": tag,
                "calib G signs (bk)": f"{rate(g_sign[bk])*100:.0f}%",
                "meter steals (bk)": f"{max(rate(m_steal[bk]), rate(f_steal[bk]))*100:.0f}%",
                "C3 bk": f"{c3_rates[bk]*100:.0f}%",
                "C3 pair": f"{c3_rates['pairwise']*100:.0f}%",
                "scramble ok (bk)": f"{min(crit_rates[f'{c}_fused_bestkey']['C2a'] for c in CONFIGS)*100:.0f}%",
                "V1 flips": "; ".join(flips) if flips else "none",
                "ACCEPT": "YES" if accept else "no",
            })

            L.append(f"## {tag}\n")
            L.append(f"- Calibration world (fused+best-key): slow value signs {rate(g_sign[bk])*100:.0f}%; "
                     f"slow meter steals {rate(m_steal[bk])*100:.0f}%; frozen steals {rate(f_steal[bk])*100:.0f}%")
            L.append(f"- Calibration world (pairwise): value {rate(g_sign['pairwise'])*100:.0f}%; "
                     f"meter {rate(m_steal['pairwise'])*100:.0f}%; frozen {rate(f_steal['pairwise'])*100:.0f}%")
            L.append(f"- C3: pairwise {c3_rates['pairwise']*100:.0f}%, fused+best-key {c3_rates[bk]*100:.0f}%")
            for key in sorted(crit_rates):
                r = crit_rates[key]
                L.append(f"- {key}: " + ", ".join(f"{k}={v*100:.0f}%" for k, v in r.items()))
            L.append(f"- V1 pass→fail flips: {'; '.join(flips) if flips else 'none'}")
            L.append("")

    L.append("## Summary\n")
    df = pd.DataFrame(summary_rows)
    L.append("| " + " | ".join(df.columns) + " |")
    L.append("|" + "---|" * len(df.columns))
    for _, r in df.iterrows():
        L.append("| " + " | ".join(str(v) for v in r.values) + " |")

    (V21 / "FLIP_REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
