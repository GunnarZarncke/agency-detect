#!/usr/bin/env python3
"""V3 — aggregate against the LOCKED registration (docs/V3_REGISTRATION.md).

Outputs V3_VERDICT.md + v3_zone_map.png in results/v3_5/. Pure evaluation.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from value_detect.criteria import evaluate_scramble_test, signature_flags

PROJECT_ROOT = Path(__file__).resolve().parents[2]
D = PROJECT_ROOT / "results" / "v3_5"
Z = 3.0
SEEDS = list(range(20))
TESTS = ("keyring", "menu")
NICE = {"keyring": "own-block", "menu": "any-block"}
ALIAS = {"alias000": 0.0, "alias005": 0.005, "alias010": 0.01,
         "alias020": 0.02, "alias050": 0.05}
GOALS = [f"G{i}" for i in range(8)]


def df(recs):
    return pd.DataFrame(recs).set_index("variable")


def rate(vals):
    vals = list(vals)
    return sum(bool(v) for v in vals) / len(vals) if vals else float("nan")


def load(world, cond):
    return {s: json.loads((D / f"{world}_{cond}_seed{s}.json").read_text()) for s in SEEDS}


def sig_tables(world, cond, test):
    R = load(world, cond)
    return {s: signature_flags(df(R[s]["tests"][test]["scores"]),
                               df(R[s]["tests"][test]["floors"]), z_min=Z) for s in SEEDS}


def main() -> None:
    L = ["# V3 VERDICT — locked registration, 20 seeds, z=3\n"]

    # ---------- puppet worlds ----------
    for world, puppet_kind in (("puppetfast", "fast"), ("puppetslow", "slow")):
        L.append(f"## {world} (captured goal G3, {puppet_kind})\n")
        L.append(f"| Test | true goals (min rate) | G3 puppet signs | worst other thief | nocore clean | C2a/C2b |")
        L.append("|---|---|---|---|---|---|")
        for test in TESTS:
            sigs = sig_tables(world, "main", test)
            true_goals = [g for g in GOALS if g != "G3"]
            gmin = min(rate(bool(sigs[s].loc[g, "signature"]) for s in SEEDS) for g in true_goals)
            puppet_rate = rate(bool(sigs[s].loc["G3", "signature"]) for s in SEEDS)
            others = [v for v in sigs[SEEDS[0]].index if v not in GOALS]
            worst = max(sum(bool(sigs[s].loc[v, "signature"]) for s in SEEDS) for v in others)
            noc = sig_tables(world, "nocore", test)
            noc_clean = rate(not noc[s]["signature"].any() for s in SEEDS)
            Rs = load(world, "scramble")
            c2 = [evaluate_scramble_test(df(Rs[s]["tests"][test]["scores"]),
                                         df(Rs[s]["tests"][test]["floors"]), z_min=Z) for s in SEEDS]
            c2a, c2b = rate(e["C2a_within_chance"] for e in c2), rate(e["C2b_no_signature_p995"] for e in c2)
            L.append(f"| {NICE[test]} | {gmin*100:.0f}% | **{puppet_rate*100:.0f}%** "
                     f"| {worst}/20 | {noc_clean*100:.0f}% | {c2a*100:.0f}%/{c2b*100:.0f}% |")
        if world == "puppetslow":
            lag_rates = {}
            Rm = load(world, "main")
            for lag in (2, 3):
                cnt = 0
                for s in SEEDS:
                    sc = df(Rm[s][f"menu_lag{lag}"])
                    fl = df(Rm[s]["tests"]["menu"]["floors"])  # descriptive: lag-1 floors
                    cnt += int(sc.loc["G3", "push_in"] > fl.loc["G3", "push_in_p95"])
                lag_rates[lag] = cnt / len(SEEDS)
            L.append(f"\nDescriptive (any-block): G3 intake visible above lag-1 floor at lag 2 in "
                     f"{lag_rates[2]*100:.0f}% of seeds, lag 3 in {lag_rates[3]*100:.0f}% "
                     f"(lag-1 floors reused; indicative only).")
        L.append("")

    # ---------- alias worlds: zone map ----------
    L.append("## Alias-colony zone map (twin agents 2 & 5)\n")
    L.append("| Noise | Test | true goals (min) | zones agent-2 twin (I/D/S/V, seeds) | zones agent-5 twin |")
    L.append("|---|---|---|---|---|")
    zone_curve = {t: {} for t in TESTS}
    for world, nz in ALIAS.items():
        for test in TESTS:
            sigs = sig_tables(world, "main", test)
            gmin = min(rate(bool(sigs[s].loc[g, "signature"]) for s in SEEDS) for g in GOALS)
            cells = []
            sep_counts = []
            for agent in (2, 5):
                zc = Counter()
                for s in SEEDS:
                    g = bool(sigs[s].loc[f"G{agent}", "signature"])
                    tw = bool(sigs[s].loc[f"G{agent}_alias", "signature"])
                    zc["I" if (g and tw) else "D" if (not g and not tw)
                       else "S" if (g and not tw) else "V"] += 1
                cells.append(f"I{zc['I']} D{zc['D']} S{zc['S']} V{zc['V']}")
                sep_counts.append(zc["S"] / len(SEEDS))
            zone_curve[test][nz] = float(np.mean(sep_counts))
            L.append(f"| {nz*100:.1f}% | {NICE[test]} | {gmin*100:.0f}% | {cells[0]} | {cells[1]} |")
    Rn = load("alias010", "nocore")
    for test in TESTS:
        noc = {s: signature_flags(df(Rn[s]["tests"][test]["scores"]),
                                  df(Rn[s]["tests"][test]["floors"]), z_min=Z) for s in SEEDS}
        L.append(f"\n- alias@1% nocore clean ({NICE[test]}): "
                 f"{rate(not noc[s]['signature'].any() for s in SEEDS)*100:.0f}%")
    L.append("")

    # ---------- yardstick ----------
    L.append("## Yardstick (interventional verdicts, fraction of seeds)\n")
    L.append("| World | target | role | autonomy | interventional value |")
    L.append("|---|---|---|---|---|")
    premium_rows = []
    for world in ("puppetfast", "puppetslow", "alias000", "alias050"):
        Y = {s: json.loads((D / f"yardstick_{world}_seed{s}.json").read_text()) for s in SEEDS}
        targets = ["G0", "G3"] + (["G2", "G2_alias"] if world.startswith("alias") else [])
        for tgt in targets:
            role = rate(Y[s]["targets"][tgt]["verdict"]["role_effect"] for s in SEEDS)
            auto = rate(Y[s]["targets"][tgt]["verdict"]["autonomy_effect"] for s in SEEDS)
            iv = rate(Y[s]["targets"][tgt]["verdict"]["interventional_value"] for s in SEEDS)
            L.append(f"| {world} | {tgt} | {role*100:.0f}% | {auto*100:.0f}% | **{iv*100:.0f}%** |")
        # registration check: interventional verdict selects exactly the true goals
        ok = []
        for s in SEEDS:
            t = Y[s]["targets"]
            true_set = {g for g in GOALS if not (world.startswith("puppet") and g == "G3")}
            sel = {k for k, v in t.items() if v["verdict"]["interventional_value"]}
            ok.append(sel == true_set)
        L.append(f"| {world} | *exact true-goal selection* | | | {rate(ok)*100:.0f}% |")
        premium_rows.append((world, rate(ok)))
    L.append("")

    # ---------- zone-map figure ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for test, marker in (("menu", "o"), ("keyring", "s")):
        xs = sorted(zone_curve[test])
        ax.plot([x * 100 for x in xs], [zone_curve[test][x] * 100 for x in xs],
                marker=marker, label=f"{NICE[test]}")
    ax.set_xlabel("alias noise (%)")
    ax.set_ylabel("separation rate (% of twin-seeds)")
    ax.set_title("The passive breaking point for the goal role\n(separation = goal signs, twin refused)")
    ax.set_ylim(-5, 105)
    ax.legend()
    fig.tight_layout()
    fig.savefig(D / "v3_zone_map.png", dpi=170)
    plt.close(fig)
    L.append("Zone-map figure: `v3_zone_map.png`.")

    (D / "V3_VERDICT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
