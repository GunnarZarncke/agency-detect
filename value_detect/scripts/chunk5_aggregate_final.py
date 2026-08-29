#!/usr/bin/env python3
"""Chunk 5 — FINAL aggregation: fused floors pooled across seeds (as locked).

Combines:
  * pairwise + fused+best-key criteria from the sweep JSONs (per-seed ≥200-shift floors —
    already spec-compliant);
  * fused-convention criteria RE-EVALUATED with floors pooled from the regenerated null
    samples (fusednulls_*.json; ≥1000 pooled samples per score, as the lock specifies);
  * C3 unchanged.

Writes CRITERIA_VERDICT_FINAL.md and signature_rate_heatmap_final.png.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from value_detect.criteria import (
    evaluate_main_test,
    evaluate_nocore_test,
    evaluate_scramble_test,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = PROJECT_ROOT / "results" / "chunk5"

TESTS = [
    "asis_pairwise", "noalias_pairwise",
    "asis_fused", "noalias_fused",
    "asis_fused_bestkey", "noalias_fused_bestkey",
]
NICE = {
    "asis_pairwise": "(a) pairwise", "noalias_pairwise": "(b) pairwise",
    "asis_fused": "(a) fused", "noalias_fused": "(b) fused",
    "asis_fused_bestkey": "(a) fus+key", "noalias_fused_bestkey": "(b) fus+key",
}
THRESH = {"V1": 0.8, "V2": 0.8, "U1": 0.9, "U2": 0.8, "T1": 0.8,
          "C1_G": 0.8, "C1_none": 0.8, "C2a": 0.8, "C2b": 0.8}


def load(prefix: str):
    out = {}
    for p in sorted(OUTDIR.glob(f"{prefix}_seed*.json")):
        d = json.loads(p.read_text())
        out[d["seed"]] = d
    return out


def rate(vals):
    vals = list(vals)
    return sum(bool(v) for v in vals) / len(vals) if vals else float("nan")


def pooled_fused_floors():
    """condition -> config -> DataFrame(variable x [push_in_p95, ..., total_flow_p995])."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for p in OUTDIR.glob("fusednulls_*_seed*.json"):
        d = json.loads(p.read_text())
        cond = d["condition"]
        for config, rows in d["samples"].items():
            for r in rows:
                v = r["variable"]
                for name in ("push_in", "out_sys", "out_env", "total_flow"):
                    if r[name] is not None:
                        acc[cond][config][v][name].append(r[name])
    floors = {}
    for cond, per_cfg in acc.items():
        floors[cond] = {}
        for config, per_var in per_cfg.items():
            rows = []
            for v, per_name in per_var.items():
                row = {"variable": v, "n_pooled": len(per_name["out_sys"])}
                for name, vals in per_name.items():
                    arr = np.asarray(vals, dtype=float)
                    row[f"{name}_p95"] = float(np.nanpercentile(arr, 95.0)) if arr.size else float("nan")
                    row[f"{name}_p995"] = float(np.nanpercentile(arr, 99.5)) if arr.size else float("nan")
                rows.append(row)
            floors[cond][config] = pd.DataFrame(rows).set_index("variable")
    return floors


def scores_frame(rec_test) -> pd.DataFrame:
    return pd.DataFrame(rec_test["scores"]).set_index("variable")


def main() -> None:
    main_r, nocore_r, scram_r, c3_r = load("main"), load("nocore"), load("scramble"), load("c3")
    seeds = sorted(main_r)
    n_seeds = len(seeds)
    pooled = pooled_fused_floors()

    rows = []
    sig_rates = defaultdict(dict)
    v2_holders = defaultdict(lambda: defaultdict(int))
    calib = {}
    fused_note = {}

    for test in TESTS:
        config = "asis" if test.startswith("asis") else "noalias"
        is_fused = test.endswith("_fused")
        is_bestkey = "bestkey" in test

        per_seed_main, per_seed_c1, per_seed_c2 = [], [], []
        for s in seeds:
            if is_fused:
                sc = scores_frame(main_r[s]["tests"][test])
                ev = evaluate_main_test(sc, pooled["main"][config], is_bestkey=False)
                sc_n = scores_frame(nocore_r[s]["tests"][test])
                ev_c1 = evaluate_nocore_test(sc_n, pooled["nocore"][config])
                sc_s = scores_frame(scram_r[s]["tests"][test])
                ev_c2 = evaluate_scramble_test(sc_s, pooled["scramble"][config])
            else:
                ev = main_r[s]["tests"][test]["criteria"]
                ev_c1 = nocore_r[s]["tests"][test]["criteria"]
                ev_c2 = scram_r[s]["tests"][test]["criteria"]
            per_seed_main.append(ev)
            per_seed_c1.append(ev_c1)
            per_seed_c2.append(ev_c2)
            for e in (ev["signature_table"] if isinstance(ev["signature_table"], list) else ev["signature_table"]):
                pass
        # signature rates + V2 holders
        for s, ev in zip(seeds, per_seed_main):
            table = ev["signature_table"]
            entries = table if isinstance(table, list) else table.to_dict("records")
            for e in entries:
                v = e["variable"]
                sig_rates[v][test] = sig_rates[v].get(test, 0) + (1 if e["signature"] else 0)
            for v in ev["V2_other_signature_holders"]:
                v2_holders[test][v] += 1

        rec = {
            "test": NICE[test],
            "V1": rate(ev["V1_G_signature"] for ev in per_seed_main),
            "V2": rate(ev["V2_unique"] for ev in per_seed_main),
            "U1": rate(ev["U1_G_above_B"] for ev in per_seed_main),
            "U2": rate(ev["U2_B_max_intake_agent_side"] for ev in per_seed_main),
            "C1_G": rate(ev["C1_G_no_signature"] for ev in per_seed_c1),
            "C1_none": rate(ev["C1_none_signature"] for ev in per_seed_c1),
            "C2a": rate(ev["C2a_within_chance"] for ev in per_seed_c2),
            "C2b": rate(ev["C2b_no_signature_p995"] for ev in per_seed_c2),
        }
        if is_bestkey:
            rec["T1"] = rate(ev.get("T1_G_env_drive_above_floor") for ev in per_seed_main)
        rows.append(rec)

        above = sum(ev["C2a_above_floor_count"] for ev in per_seed_c2)
        comps = sum(ev["C2a_comparisons"] for ev in per_seed_c2)
        calib[NICE[test]] = (above, comps, above / comps if comps else float("nan"))
        if is_fused:
            fused_note[NICE[test]] = pooled["main"][config]["n_pooled"].iloc[0] if "n_pooled" in pooled["main"][config] else None

    for v in sig_rates:
        for t in sig_rates[v]:
            sig_rates[v][t] /= n_seeds
    verdict = pd.DataFrame(rows).set_index("test")

    c3_rates = {conv: rate(c3_r[s]["tests"][conv]["criteria"]["C3_no_goal_signature"] for s in sorted(c3_r))
                for conv in ("pairwise", "fused_bestkey")}

    # ---- heatmap ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_vars = ["G", "B", "A", "E", "S", "S_alias", "A_alias", "D", "W"]
    mat = np.full((len(all_vars), len(TESTS)), np.nan)
    for i, v in enumerate(all_vars):
        for j, t in enumerate(TESTS):
            if v in ("S_alias", "A_alias") and "noalias" in t:
                continue
            mat[i, j] = sig_rates.get(v, {}).get(t, 0.0)
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(TESTS)), [NICE[t] for t in TESTS], rotation=30, ha="right")
    ax.set_yticks(range(len(all_vars)), all_vars)
    for i in range(len(all_vars)):
        for j in range(len(TESTS)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if mat[i, j] > 0.6 else "#333333")
    ax.set_title(f"Value-signature rate across {n_seeds} seeds (final; fused floors pooled)")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "signature_rate_heatmap_final.png", dpi=170)
    plt.close(fig)

    # ---- memo ----
    L = [f"# Chunk 5 FINAL verdict — {n_seeds} seeds, locked criteria\n"]
    L.append("Fused-convention floors pooled across seeds per the locked spec "
             f"({int(verdict.shape[0] and 1) and ''}≥1000 samples per score); pairwise and fused+best-key "
             "use their per-seed ≥200-shift floors as locked. PASS/FAIL mechanical.\n")
    cols = ["V1", "V2", "U1", "U2", "T1"]
    L.append("## Main-world criteria\n")
    L.append("| Test | " + " | ".join(cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for t, r in verdict.iterrows():
        cells = []
        for c in cols:
            val = r.get(c, np.nan)
            if np.isfinite(val):
                cells.append(f"{val*100:.0f}% ({'PASS' if val >= THRESH[c] else 'FAIL'})")
            else:
                cells.append("—")
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Controls\n")
    L.append("| Test | C1 G-clean | C1 all-clean | C2a calibration | C2b no-signature |")
    L.append("|---|---|---|---|---|")
    for t, r in verdict.iterrows():
        cells = [f"{r[c]*100:.0f}% ({'PASS' if r[c] >= THRESH[c]else 'FAIL'})" for c in ("C1_G", "C1_none", "C2a", "C2b")]
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L.append("")
    L.append(f"C3: pairwise {c3_rates['pairwise']*100:.0f}% "
             f"({'PASS' if c3_rates['pairwise'] >= 0.8 else 'FAIL'}); fused+best-key "
             f"{c3_rates['fused_bestkey']*100:.0f}% ({'PASS' if c3_rates['fused_bestkey'] >= 0.8 else 'FAIL'}).\n")
    L.append("## V2 violators (variable: seeds held signature)\n")
    for t in TESTS:
        if v2_holders[t]:
            L.append(f"- {NICE[t]}: " + ", ".join(f"{v} ({c}/{n_seeds})" for v, c in sorted(v2_holders[t].items())))
    L.append("")
    L.append("## Scramble calibration (pooled above-floor rate; expected ≈ 5%)\n")
    for t, (above, comps, r) in calib.items():
        L.append(f"- {t}: {above}/{comps} = {r*100:.1f}%")
    L.append("")
    L.append("Final heatmap: `signature_rate_heatmap_final.png`.")
    (OUTDIR / "CRITERIA_VERDICT_FINAL.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
