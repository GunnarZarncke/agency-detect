#!/usr/bin/env python3
"""Chunk 5 — aggregate the sweep against the LOCKED pre-registration.

Reads results/chunk5/*.json (80 files: 20 seeds × {main, nocore, scramble, c3}),
computes per-test criterion pass rates, renders the verdict scorecard, the
signature-rate heatmap, and the stability summary. Pure aggregation — no new
measurements, no thresholds beyond docs/SUCCESS_CRITERIA.md.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

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
          "C1_G": 0.8, "C1_none": 0.8, "C2a": 0.8, "C2b": 0.8, "C3": 0.8}


def load(kind: str):
    out = {}
    for p in sorted(OUTDIR.glob(f"{kind}_seed*.json")):
        d = json.loads(p.read_text())
        out[d["seed"]] = d
    return out


def rate(vals):
    vals = list(vals)
    return sum(bool(v) for v in vals) / len(vals) if vals else float("nan")


def main() -> None:
    main_r, nocore_r, scram_r, c3_r = load("main"), load("nocore"), load("scramble"), load("c3")
    seeds = sorted(main_r)
    n_seeds = len(seeds)

    # ---- per-test criterion rates ----
    rows = []
    sig_rates = defaultdict(dict)   # variable -> test -> fraction of seeds with signature
    v2_holders = defaultdict(lambda: defaultdict(int))
    for test in TESTS:
        crit = lambda key: [main_r[s]["tests"][test]["criteria"][key] for s in seeds]
        rec = {
            "test": NICE[test],
            "V1": rate(crit("V1_G_signature")),
            "V2": rate(crit("V2_unique")),
            "U1": rate(crit("U1_G_above_B")),
            "U2": rate(crit("U2_B_max_intake_agent_side")),
        }
        if "bestkey" in test:
            rec["T1"] = rate(crit("T1_G_env_drive_above_floor"))
        rec["C1_G"] = rate(nocore_r[s]["tests"][test]["criteria"]["C1_G_no_signature"] for s in seeds)
        rec["C1_none"] = rate(nocore_r[s]["tests"][test]["criteria"]["C1_none_signature"] for s in seeds)
        rec["C2a"] = rate(scram_r[s]["tests"][test]["criteria"]["C2a_within_chance"] for s in seeds)
        rec["C2b"] = rate(scram_r[s]["tests"][test]["criteria"]["C2b_no_signature_p995"] for s in seeds)
        rows.append(rec)
        for s in seeds:
            for e in main_r[s]["tests"][test]["criteria"]["signature_table"]:
                v = e["variable"]
                sig_rates[v][test] = sig_rates[v].get(test, 0) + (1 if e["signature"] else 0)
            for v in main_r[s]["tests"][test]["criteria"]["V2_other_signature_holders"]:
                v2_holders[test][v] += 1
    for v in sig_rates:
        for t in sig_rates[v]:
            sig_rates[v][t] /= n_seeds
    verdict = pd.DataFrame(rows).set_index("test")

    # C3 (per convention).
    c3_rows = {}
    for conv in ("pairwise", "fused_bestkey"):
        c3_rows[conv] = rate(c3_r[s]["tests"][conv]["criteria"]["C3_no_goal_signature"] for s in sorted(c3_r))

    # ---- scramble calibration detail ----
    calib = {}
    for test in TESTS:
        above = sum(scram_r[s]["tests"][test]["criteria"]["C2a_above_floor_count"] for s in seeds)
        comps = sum(scram_r[s]["tests"][test]["criteria"]["C2a_comparisons"] for s in seeds)
        calib[NICE[test]] = (above, comps, above / comps if comps else float("nan"))

    # ---- stability: G's system polarity across variants ----
    stab_acc = defaultdict(list)
    for s in seeds:
        stab = main_r[s].get("stability", {})
        for key, table in stab.items():
            for e in table:
                if e["variable"] == "G":
                    stab_acc[key].append(e["polarity_sys"])
    stab_summary = {k: (float(np.mean(v)), float(np.std(v))) for k, v in sorted(stab_acc.items())}

    # ---- signature-rate heatmap ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_vars = ["G", "B", "A", "E", "S", "S_alias", "A_alias", "D", "W"]
    mat = np.full((len(all_vars), len(TESTS)), np.nan)
    for i, v in enumerate(all_vars):
        for j, t in enumerate(TESTS):
            if t in sig_rates.get(v, {}):
                mat[i, j] = sig_rates[v][t]
            elif v in ("S_alias", "A_alias") and "noalias" in t:
                mat[i, j] = np.nan
            else:
                mat[i, j] = 0.0
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(TESTS)), [NICE[t] for t in TESTS], rotation=30, ha="right")
    ax.set_yticks(range(len(all_vars)), all_vars)
    for i in range(len(all_vars)):
        for j in range(len(TESTS)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if mat[i, j] > 0.6 else "#333333")
    ax.set_title(f"Value-signature rate across {n_seeds} seeds (fraction of seeds; blank = variable absent)")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "signature_rate_heatmap.png", dpi=170)
    plt.close(fig)

    # ---- verdict memo ----
    L = [f"# Chunk 5 verdict — {n_seeds} seeds against the locked pre-registration\n"]
    L.append("Thresholds: V1/V2/U2/T1/C1/C2/C3 ≥ 80% of seeds; U1 ≥ 90%. PASS/FAIL is mechanical.\n")
    L.append("## Main-world criteria\n")
    cols = ["V1", "V2", "U1", "U2", "T1"]
    L.append("| Test | " + " | ".join(f"{c} rate (pass?)" for c in cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for t, r in verdict.iterrows():
        cells = []
        for c in cols:
            if c in r and np.isfinite(r.get(c, np.nan)):
                ok = "PASS" if r[c] >= THRESH[c] else "FAIL"
                cells.append(f"{r[c]*100:.0f}% ({ok})")
            else:
                cells.append("—")
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Controls\n")
    L.append("| Test | C1 G-clean | C1 all-clean | C2a calibration | C2b no-signature |")
    L.append("|---|---|---|---|---|")
    for t, r in verdict.iterrows():
        cells = [f"{r[c]*100:.0f}% ({'PASS' if r[c] >= THRESH[c2] else 'FAIL'})"
                 for c, c2 in [("C1_G", "C1_G"), ("C1_none", "C1_none"), ("C2a", "C2a"), ("C2b", "C2b")]]
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L.append("")
    L.append(f"C3 (`goal_progress` must not get the signature): pairwise "
             f"{c3_rows['pairwise']*100:.0f}% ({'PASS' if c3_rows['pairwise'] >= 0.8 else 'FAIL'}); "
             f"fused+best-key {c3_rows['fused_bestkey']*100:.0f}% "
             f"({'PASS' if c3_rows['fused_bestkey'] >= 0.8 else 'FAIL'}).\n")
    L.append("## Who else ever held the signature (V2 violators, seed counts)\n")
    for t in TESTS:
        if v2_holders[t]:
            L.append(f"- {NICE[t]}: " + ", ".join(f"{v} ({c}/{n_seeds})" for v, c in sorted(v2_holders[t].items())))
    L.append("")
    L.append("## Scramble calibration (pooled above-floor rate; expected ≈ 5%)\n")
    for t, (above, comps, r) in calib.items():
        L.append(f"- {t}: {above}/{comps} = {r*100:.1f}%")
    L.append("")
    L.append("## Stability of G's polarity (mean ± sd across seeds)\n")
    for k, (m, sd) in stab_summary.items():
        L.append(f"- {k}: {m:+.3f} ± {sd:.3f}")
    L.append("")
    L.append(f"Heatmap: `signature_rate_heatmap.png`")
    (OUTDIR / "CRITERIA_VERDICT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
