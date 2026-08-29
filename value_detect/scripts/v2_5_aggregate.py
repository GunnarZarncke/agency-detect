#!/usr/bin/env python3
"""V2-5 — FINAL aggregation against the LOCKED registration (docs/V2_5_PREREGISTRATION.md).

Pooled floors where registered (colony keyed tests; anchor fused), per-seed elsewhere;
z = 3 throughout. Pure evaluation — no new measurements. Outputs V2_VERDICT.md +
v2_heatmap.png in results/v2_5/.
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
    signature_flags,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
D = PROJECT_ROOT / "results" / "v2_5"
Z = 3.0
SEEDS = list(range(20))
TESTS = {"anchor": ["fused", "fused_bestkey", "grown_keys", "keyring", "menu"],
         "colony": ["fused_bestkey", "grown_keys", "keyring", "menu"],
         "deep_synergy": ["fused_bestkey", "grown_keys", "keyring", "menu"],
         "slow_meter": ["fused_bestkey", "grown_keys", "keyring", "menu"]}
NICE = {"fused": "fused", "fused_bestkey": "best-key", "grown_keys": "grown",
        "keyring": "key-ring", "menu": "menu"}
GOALS = {"anchor": ["G"], "colony": [f"G{i}" for i in range(8)],
         "deep_synergy": ["G_P", "G_M"], "slow_meter": ["G"]}
METERS = ["M_fast", "M8", "M32", "F_sat"]


def df(recs):
    return pd.DataFrame(recs).set_index("variable")


def rate(vals):
    vals = list(vals)
    return sum(bool(v) for v in vals) / len(vals) if vals else float("nan")


def load_all():
    out = defaultdict(dict)
    for p in D.glob("*_seed*.json"):
        name = p.stem
        if name.startswith("anchorfusednulls"):
            continue
        world, cond, seedtag = name.rsplit("_", 2)
        out[(world, cond)][int(seedtag[4:])] = json.loads(p.read_text())
    return out


def pooled_floor_table(sample_rows_by_seed):
    acc = defaultdict(lambda: defaultdict(list))
    for rows in sample_rows_by_seed:
        for r in rows:
            for name in ("push_in", "out_sys", "out_env", "total_flow"):
                if r[name] is not None:
                    acc[r["variable"]][name].append(r[name])
    out = []
    for v, per in acc.items():
        row = {"variable": v}
        for name, vals in per.items():
            a = np.asarray(vals, dtype=float)
            row[f"{name}_p95"] = float(np.nanpercentile(a, 95))
            row[f"{name}_p995"] = float(np.nanpercentile(a, 99.5))
            row[f"{name}_mean"] = float(np.nanmean(a))
            row[f"{name}_sd"] = float(np.nanstd(a))
        out.append(row)
    return pd.DataFrame(out).set_index("variable")


def main() -> None:
    R = load_all()

    # ---- pooled floors ----
    pooled = {}
    for cond in ("main", "nocore", "scramble"):
        for test in ("fused_bestkey", "grown_keys"):
            pooled[("colony", cond, test)] = pooled_floor_table(
                [R[("colony", cond)][s]["tests"][test]["null_samples"] for s in SEEDS])
        anchor_nulls = [json.loads((D / f"anchorfusednulls_{cond}_seed{s}.json").read_text())
                        for s in SEEDS]
        assert all(a["p95_mismatches"] == 0 for a in anchor_nulls), "anchor null stream mismatch"
        pooled[("anchor", cond, "fused")] = pooled_floor_table([a["samples"] for a in anchor_nulls])

    def floors_for(world, cond, seed, test):
        if (world, cond, test) in pooled:
            return pooled[(world, cond, test)]
        return df(R[(world, cond)][seed]["tests"][test]["floors"])

    def scores_for(world, cond, seed, test):
        return df(R[(world, cond)][seed]["tests"][test]["scores"])

    def sig(world, cond, seed, test):
        return signature_flags(scores_for(world, cond, seed, test),
                               floors_for(world, cond, seed, test), z_min=Z)

    L = ["# V2 VERDICT — locked registration, 20 seeds, z=3\n"]
    verdict_rows = []
    heat = {}
    contingency = []

    # ================= ANCHOR =================
    L.append("## Anchor (V1 world) — regression baseline\n")
    L.append("| Test | V1 | V2 | U1 | U2 | T1 | C1 | C2a/C2b |")
    L.append("|---|---|---|---|---|---|---|---|")
    for test in TESTS["anchor"]:
        evs = [evaluate_main_test(scores_for("anchor", "main", s, test),
                                  floors_for("anchor", "main", s, test),
                                  is_bestkey=(test == "fused_bestkey"), z_min=Z) for s in SEEDS]
        c1 = [evaluate_nocore_test(scores_for("anchor", "nocore", s, test),
                                   floors_for("anchor", "nocore", s, test), z_min=Z) for s in SEEDS]
        c2 = [evaluate_scramble_test(scores_for("anchor", "scramble", s, test),
                                     floors_for("anchor", "scramble", s, test), z_min=Z) for s in SEEDS]
        r = {"V1": rate(e["V1_G_signature"] for e in evs), "V2": rate(e["V2_unique"] for e in evs),
             "U1": rate(e["U1_G_above_B"] for e in evs), "U2": rate(e["U2_B_max_intake_agent_side"] for e in evs),
             "T1": rate(e.get("T1_G_env_drive_above_floor") for e in evs) if test == "fused_bestkey" else None,
             "C1": min(rate(e["C1_G_no_signature"] for e in c1), rate(e["C1_none_signature"] for e in c1)),
             "C2a": rate(e["C2a_within_chance"] for e in c2), "C2b": rate(e["C2b_no_signature_p995"] for e in c2)}
        heat[("anchor G", NICE[test])] = r["V1"]
        ok = lambda v, t=0.8: "" if v is None else (f"{v*100:.0f}%" + ("" if v >= t else " FAIL"))
        L.append(f"| {NICE[test]} | {ok(r['V1'])} | {ok(r['V2'])} | {ok(r['U1'], 0.9)} | {ok(r['U2'])} "
                 f"| {ok(r['T1']) if r['T1'] is not None else '—'} | {ok(r['C1'])} | {ok(r['C2a'])}/{ok(r['C2b'])} |")
        verdict_rows.append(("anchor", test, all(v is None or v >= (0.9 if k == 'U1' else 0.8)
                                                 for k, v in r.items())))
    L.append("")

    # ================= COLONY =================
    L.append("## Colony (8 agents, ring-coupled)\n")
    L.append("| Test | CV1 (min per-goal) | CV2 (worst thief seeds) | CU1 | CU2 | nocore | C2a/C2b |")
    L.append("|---|---|---|---|---|---|---|")
    for test in TESTS["colony"]:
        if test in ("fused_bestkey", "grown_keys"):
            # Defect B (DECISIONS 2026-08-12): fused-intake saturates at 49 vars; the
            # no-core control caught the collapse. Intake-infeasible at scale — recorded.
            drive_ok = []
            for s in SEEDS:
                sc = scores_for("colony", "main", s, test)
                fl = floors_for("colony", "main", s, test)
                for g in GOALS["colony"]:
                    z_ok = (sc.loc[g, "out_sys"] - fl.loc[g, "out_sys_mean"]) >= Z * fl.loc[g, "out_sys_sd"]
                    drive_ok.append(bool(sc.loc[g, "out_sys"] > fl.loc[g, "out_sys_p95"] and z_ok))
            L.append(f"| {NICE[test]} | INTAKE-INFEASIBLE at scale (recorded; no-core control detected "
                     f"saturation). Goal drive above floor: {rate(drive_ok)*100:.0f}% of goal-seeds | | | | | |")
            verdict_rows.append(("colony", test, None))
            heat[("colony Ḡ", NICE[test])] = float("nan")
            continue
        sigs = {s: sig("colony", "main", s, test) for s in SEEDS}
        goal_rates = {g: rate(bool(sigs[s].loc[g, "signature"]) for s in SEEDS) for g in GOALS["colony"]}
        nonspecial = [v for v in sigs[SEEDS[0]].index if v not in GOALS["colony"]]
        thief_counts = {v: sum(bool(sigs[s].loc[v, "signature"]) for s in SEEDS) for v in nonspecial}
        worst_thief = max(thief_counts.values()) if thief_counts else 0
        cu1 = cu2 = []
        cu1 = []
        cu2 = []
        for s in SEEDS:
            sc = scores_for("colony", "main", s, test)
            for i in range(8):
                g, b = f"G{i}", f"B{i}"
                cu1.append(bool(sigs[s].loc[g, "rankable"]) and
                           sc.loc[g, "polarity_sys"] > sc.loc[b, "polarity_sys"])
                side = [f"{c}{i}" for c in "BSAG"]
                cu2.append(sc.loc[side, "push_in"].idxmax() == b)
        noc = {s: sig("colony", "nocore", s, test) for s in SEEDS}
        noc_goal_worst = max(sum(bool(noc[s].loc[g, "signature"]) for s in SEEDS) for g in GOALS["colony"])
        noc_clean = rate(not noc[s]["signature"].any() for s in SEEDS)
        c2 = [evaluate_scramble_test(scores_for("colony", "scramble", s, test),
                                     floors_for("colony", "scramble", s, test), z_min=Z) for s in SEEDS]
        cv1_min = min(goal_rates.values())
        r = {"CV1": cv1_min, "CV2_worst": worst_thief, "CU1": rate(cu1), "CU2": rate(cu2),
             "noc": min(noc_clean, 1.0 if noc_goal_worst <= 2 else 0.0),
             "C2a": rate(e["C2a_within_chance"] for e in c2), "C2b": rate(e["C2b_no_signature_p995"] for e in c2)}
        heat[("colony Ḡ", NICE[test])] = float(np.mean(list(goal_rates.values())))
        passed = (cv1_min >= 0.8 and worst_thief <= 2 and r["CU1"] >= 0.9 and r["CU2"] >= 0.8
                  and noc_goal_worst <= 2 and noc_clean >= 0.8 and r["C2a"] >= 0.8 and r["C2b"] >= 0.8)
        if not passed:
            contingency.append((test, f"CV1min={cv1_min:.2f}, worst_thief={worst_thief}, "
                                      f"CU1={r['CU1']:.2f}, CU2={r['CU2']:.2f}"))
        L.append(f"| {NICE[test]} | {cv1_min*100:.0f}%{'' if cv1_min>=0.8 else ' FAIL'} "
                 f"| {worst_thief}/20{'' if worst_thief<=2 else ' FAIL'} "
                 f"| {r['CU1']*100:.0f}%{'' if r['CU1']>=0.9 else ' FAIL'} "
                 f"| {r['CU2']*100:.0f}%{'' if r['CU2']>=0.8 else ' FAIL'} "
                 f"| {r['noc']*100:.0f}%{'' if r['noc']>=0.8 else ' FAIL'} "
                 f"| {r['C2a']*100:.0f}%/{r['C2b']*100:.0f}% |")
        verdict_rows.append(("colony", test, passed))
        if test in ("keyring", "menu"):
            thieves_named = {v: c for v, c in thief_counts.items() if c > 2}
            if thieves_named:
                L.append(f"|  ↳ systematic thieves: {thieves_named} | | | | | | |")
    L.append("")

    # ================= DEEP SYNERGY =================
    L.append("## Deep synergy (agent P parity, agent M majority)\n")
    L.append("| Test | G_M | G_P | DS2 worst thief | nocore | C2a/C2b |")
    L.append("|---|---|---|---|---|---|")
    parity_gap = {}
    for test in TESTS["deep_synergy"]:
        sigs = {s: sig("deep_synergy", "main", s, test) for s in SEEDS}
        gm = rate(bool(sigs[s].loc["G_M", "signature"]) for s in SEEDS)
        gp = rate(bool(sigs[s].loc["G_P", "signature"]) for s in SEEDS)
        nong = [v for v in sigs[SEEDS[0]].index if v not in GOALS["deep_synergy"]]
        worst = max(sum(bool(sigs[s].loc[v, "signature"]) for s in SEEDS) for v in nong)
        noc = {s: sig("deep_synergy", "nocore", s, test) for s in SEEDS}
        noc_ok = (max(sum(bool(noc[s].loc[g, "signature"]) for s in SEEDS) for g in GOALS["deep_synergy"]) <= 2
                  and rate(not noc[s]["signature"].any() for s in SEEDS) >= 0.8)
        c2 = [evaluate_scramble_test(scores_for("deep_synergy", "scramble", s, test),
                                     floors_for("deep_synergy", "scramble", s, test), z_min=Z) for s in SEEDS]
        c2a, c2b = rate(e["C2a_within_chance"] for e in c2), rate(e["C2b_no_signature_p995"] for e in c2)
        gp_required = test in ("keyring", "menu")
        passed = (gm >= 0.8 and (gp >= 0.8 if gp_required else True) and worst <= 2
                  and noc_ok and c2a >= 0.8 and c2b >= 0.8)
        gp_note = "" if (gp >= 0.8 or not gp_required) else " FAIL"
        if not gp_required:
            gp_note = " (cliff, registered)" if gp < 0.8 else " (unexpected sight!)"
        L.append(f"| {NICE[test]} | {gm*100:.0f}%{'' if gm>=0.8 else ' FAIL'} | {gp*100:.0f}%{gp_note} "
                 f"| {worst}/20{'' if worst<=2 else ' FAIL'} | {'PASS' if noc_ok else 'FAIL'} "
                 f"| {c2a*100:.0f}%/{c2b*100:.0f}% |")
        heat[("deep-syn G_M", NICE[test])] = gm
        heat[("deep-syn G_P", NICE[test])] = gp
        verdict_rows.append(("deep_synergy", test, passed))
        go = np.mean([scores_for("deep_synergy", "main", s, test).loc["G_P", "out_sys"] for s in SEEDS])
        gm_o = np.mean([scores_for("deep_synergy", "main", s, test).loc["G_M", "out_sys"] for s in SEEDS])
        parity_gap[NICE[test]] = (go, gm_o)
    L.append("")
    blocks_gp = np.mean([parity_gap["key-ring"][0], parity_gap["menu"][0]])
    blocks_gm = np.mean([parity_gap["key-ring"][1], parity_gap["menu"][1]])
    L.append(f"Parity gap (mean drive, blocks − grown): G_P {blocks_gp - parity_gap['grown'][0]:+.3f} nats "
             f"(the measured cliff); G_M {blocks_gm - parity_gap['grown'][1]:+.3f} nats (≈0 registered).\n")

    # ================= SLOW METER =================
    L.append("## Slow-meter (G at 0.5%; four meter witnesses)\n")
    L.append("| Test | SM1 (G) | meter thefts (worst) | per-meter seeds | nocore | C2a/C2b |")
    L.append("|---|---|---|---|---|---|")
    watch = {}
    for test in TESTS["slow_meter"]:
        sigs = {s: sig("slow_meter", "main", s, test) for s in SEEDS}
        g = rate(bool(sigs[s].loc["G", "signature"]) for s in SEEDS)
        counts = {m: sum(bool(sigs[s].loc[m, "signature"]) for s in SEEDS) for m in METERS}
        nong = [v for v in sigs[SEEDS[0]].index if v != "G"]
        worst_any = max(sum(bool(sigs[s].loc[v, "signature"]) for s in SEEDS) for v in nong)
        noc = {s: sig("slow_meter", "nocore", s, test) for s in SEEDS}
        noc_ok = (sum(bool(noc[s].loc["G", "signature"]) for s in SEEDS) <= 2
                  and rate(not noc[s]["signature"].any() for s in SEEDS) >= 0.8)
        c2 = [evaluate_scramble_test(scores_for("slow_meter", "scramble", s, test),
                                     floors_for("slow_meter", "scramble", s, test), z_min=Z) for s in SEEDS]
        c2a, c2b = rate(e["C2a_within_chance"] for e in c2), rate(e["C2b_no_signature_p995"] for e in c2)
        passed = g >= 0.8 and max(counts.values()) <= 2 and worst_any <= 2 and noc_ok and c2a >= 0.8 and c2b >= 0.8
        L.append(f"| {NICE[test]} | {g*100:.0f}%{'' if g>=0.8 else ' FAIL'} "
                 f"| {worst_any}/20{'' if worst_any<=2 else ' FAIL'} | {counts} "
                 f"| {'PASS' if noc_ok else 'FAIL'} | {c2a*100:.0f}%/{c2b*100:.0f}% |")
        heat[("slow G", NICE[test])] = g
        watch[NICE[test]] = counts
        verdict_rows.append(("slow_meter", test, passed))
    L.append("")

    # ---- diagnostics ----
    L.append("## Diagnostics\n")
    dials = defaultdict(list)
    gplace = defaultdict(int)
    for s in SEEDS:
        p = R[("colony", "main")][s]["partition"]
        dials[p["dial"]].append(s)
        for g in GOALS["colony"]:
            where = ("block" if any(g in b for b in p["agents"]) else
                     "env" if g in p["env"] else "orphan")
            gplace[where] += 1
    L.append(f"- Colony detection dials: { {d: len(v) for d, v in sorted(dials.items())} }; "
             f"goal placement across 160 goal-seeds: {dict(gplace)}.")
    lm = max(max((R[("colony", "main")][s]["tests"][t].get("lost_mass") or {"": 0}).values())
             for s in SEEDS for t in ("keyring", "menu"))
    L.append(f"- Max compression lost-mass (colony blocks): {lm*100:.1f}%.")
    L.append(f"- Witness-watch (slow-meter theft seeds/20 by test): {watch} — "
             "SJ's registered expectation: ≈0 and declining with complexity.")
    L.append("")

    # ---- registered attribution (colony0: partitioned rerun, block tests) ----
    L.append("## Registered attribution — partitioned colony (coupling = 0), block tests\n")
    L.append("| Test | CV1 min (coupled → partitioned) | worst thief (coupled → part.) | nocore clean (part.) |")
    L.append("|---|---|---|---|")
    for test in ("keyring", "menu"):
        sigs0 = {s: sig("colony0", "main", s, test) for s in SEEDS}
        gr0 = {g: rate(bool(sigs0[s].loc[g, "signature"]) for s in SEEDS) for g in GOALS["colony"]}
        non0 = [v for v in sigs0[SEEDS[0]].index if v not in GOALS["colony"]]
        thief0 = {v: sum(bool(sigs0[s].loc[v, "signature"]) for s in SEEDS) for v in non0}
        worst0 = max(thief0.values()) if thief0 else 0
        noc0 = {s: sig("colony0", "nocore", s, test) for s in SEEDS}
        noc0_clean = rate(not noc0[s]["signature"].any() for s in SEEDS)
        sigs1 = {s: sig("colony", "main", s, test) for s in SEEDS}
        gr1_min = min(rate(bool(sigs1[s].loc[g, "signature"]) for s in SEEDS) for g in GOALS["colony"])
        non1 = [v for v in sigs1[SEEDS[0]].index if v not in GOALS["colony"]]
        worst1 = max(sum(bool(sigs1[s].loc[v, "signature"]) for s in SEEDS) for v in non1)
        thieves0_named = {v: c for v, c in thief0.items() if c > 2}
        L.append(f"| {NICE[test]} | {gr1_min*100:.0f}% → {min(gr0.values())*100:.0f}% "
                 f"| {worst1}/20 → {worst0}/20 {thieves0_named if thieves0_named else ''} "
                 f"| {noc0_clean*100:.0f}% |")
        heat[("colony Ḡ (part.)", NICE[test])] = float(np.mean(list(gr0.values())))
    L.append("")
    L.append("Attribution rule (locked): persists ⇒ scale; vanishes ⇒ interference.\n")

    # ---- verdict summary + heatmap ----
    L.append("## Verdict summary\n")
    for world in TESTS:
        row = [f"{NICE[t]}: " + ("intake-infeasible" if ok is None else ("PASS" if ok else "FAIL"))
               for (w, t, ok) in verdict_rows if w == world]
        L.append(f"- **{world}**: " + "; ".join(row))
    L.append("")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = ["anchor G", "colony Ḡ", "colony Ḡ (part.)", "deep-syn G_P", "deep-syn G_M", "slow G"]
    cols = ["fused", "best-key", "grown", "key-ring", "menu"]
    mat = np.full((len(rows), len(cols)), np.nan)
    for i, rname in enumerate(rows):
        for j, c in enumerate(cols):
            if (rname, c) in heat:
                mat[i, j] = heat[(rname, c)]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, "n/a" if np.isnan(mat[i, j]) else f"{mat[i, j]*100:.0f}%",
                    ha="center", va="center", fontsize=10,
                    color="#333333")
    ax.set_title("V2 benchmark: goal-recovery rate (fraction of seeds with the signature)")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(D / "v2_heatmap.png", dpi=170)
    plt.close(fig)
    L.append("Heatmap: `v2_heatmap.png`.")

    (D / "V2_VERDICT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
