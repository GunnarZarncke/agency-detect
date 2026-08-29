#!/usr/bin/env python3
"""V3 — world audits + the yardstick validation bench (SJ reviews BEFORE registration).

Outputs in results/v3_0/: AUDIT_puppet_fast.md, AUDIT_puppet_slow.md, AUDIT_alias.md,
BENCH_yardstick.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import value_detect as vd
from value_detect.directed import cmi
from value_detect.worlds_v2 import NOISE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = PROJECT_ROOT / "results" / "v3_0"
SEED, N = 0, 20000


def xor_p(*ps):
    q = 0.0
    for p in ps:
        q = q * (1 - p) + (1 - q) * p
    return q


def cr(x):
    x = np.asarray(x)
    return float(np.mean(x[1:] != x[:-1]))


def audit_puppet(mode: str) -> str:
    f = vd.colony_frame(seed=SEED, n_steps=N, puppet=(mode, 3))
    g3, e3 = f["G3"].to_numpy(), f["E3"].to_numpy()
    a3, b3 = f["A3"].to_numpy(), f["B3"].to_numpy()
    L = [f"# Audit — puppet-colony A{'1' if mode == 'fast' else '2'} ({mode} captured goal, agent 3; seed 0, 20k)\n"]
    L.append("## Tempi (change rates)\n")
    true_rates = [cr(f[f"G{i}"]) for i in range(8) if i != 3]
    L.append(f"- True goals G0..G7 (excl. 3): {min(true_rates)*100:.2f}–{max(true_rates)*100:.2f}% "
             f"(designed ~1.5%).")
    L.append(f"- Captured goal G3: {cr(g3)*100:.2f}%"
             + (" — emergent ~22%, NOT the naive ~50%: the puppet copies the same patch the belief"
                "\n  copies, so the two partially XOR-cancel in the action and the puppet CALMS its"
                "\n  own patch (a puppet that pacifies its agent)." if mode == "fast" else
                " — tempo-matched to a true goal: **indistinguishable by rate alone**."))
    L.append("")
    L.append("## Wiring and strings\n")
    L.append(f"- Wiring intact: A3 vs (B3 ⊕ G3) disagree {np.mean(a3 != (b3 ^ g3))*100:.1f}% "
             f"(designed {xor_p(NOISE['action'], NOISE['obs'])*100:.1f}%) — the captured goal still"
             " steers its agent exactly like a real one.")
    if mode == "fast":
        L.append(f"- The strings: G3(t+1) vs E3(t) disagree {np.mean(g3[1:] != e3[:-1])*100:.1f}% "
                 f"(designed 3.0%) — every move follows its patch.")
    else:
        flips = np.where(g3[1:] != g3[:-1])[0]
        triggered = sum(1 for t in flips if t >= 6 and len(set(e3[t - 6:t])) == 1)
        L.append(f"- The strings: {len(flips)} flips in 20k steps; **{triggered}/{len(flips)} "
                 f"immediately follow a 6-run of its own patch** (the trigger, verified per flip).")
    other = [np.mean(f[f"A{i}"].to_numpy() != (f[f"B{i}"].to_numpy() ^ f[f"G{i}"].to_numpy()))
             for i in range(8) if i != 3]
    L.append(f"- Other agents unaffected: wiring disagreement {np.mean(other)*100:.1f}% (designed "
             f"{xor_p(NOISE['action'], NOISE['obs'])*100:.1f}%).\n")
    L.append("## Pre-registered predictions\n")
    if mode == "fast":
        L.append("- HARD criterion: G3 refused (no signature) in ≥80% of seeds — its intake is large and visible.")
    else:
        L.append("- PROBE (registered prediction): G3 DEFEATS the lag-1 passive tests (earns the "
                 "signature) — the trigger lives in run-history invisible to one-step conditioning; "
                 "lag-2/3 columns reported; the yardstick unmasks it (autonomy probe).")
    L.append("- The seven true goals sign as in V2; controls as registered.")
    return "\n".join(L)


def audit_alias() -> str:
    L = ["# Audit — alias-colony (goal twins for agents 2 & 5; seed 0, 20k)\n"]
    base = vd.colony_frame(seed=SEED, n_steps=N)
    L.append("| Alias noise | G2 vs twin disagreement | G5 vs twin | Base columns identical to plain colony |")
    L.append("|---|---|---|---|")
    for nz in (0.0, 0.005, 0.01, 0.02, 0.05):
        f = vd.alias_colony_frame(seed=SEED, n_steps=N, alias_noise=nz)
        same = all((f[c].to_numpy() == base[c].to_numpy()).all() for c in base.columns)
        d2 = np.mean(f["G2"].to_numpy() != f["G2_alias"].to_numpy())
        d5 = np.mean(f["G5"].to_numpy() != f["G5_alias"].to_numpy())
        L.append(f"| {nz*100:.1f}% | {d2*100:.2f}% | {d5*100:.2f}% | {'yes' if same else 'NO — BUG'} |")
    L.append("")
    L.append("The twins are recordings appended after the fact — the dynamics never see them; at "
             "0% noise the twin is bit-identical to its goal (the theorem case: no passive method "
             "can separate identical sequences).\n")
    L.append("## Pre-registered predictions\n")
    L.append("- Zones per noise level per test: indistinguishable (both sign) at 0%; separation "
             "(goal signs, twin refused) by 5%; where deadlock appears in between is the finding.")
    L.append("- All eight true goals sign at every level; controls at the 1% level as registered.")
    return "\n".join(L)


def bench() -> str:
    L = ["# Yardstick validation bench — Gunnar's embedded goal_flip vs our do-style (anchor)\n"]
    L.append("Verdict pattern must agree: exactly the goal shows a role-effect. His score formula "
             "on his machinery/world; ours on our 1-agent engine (same designed mechanics).\n")
    from uad_handles.minimal import IDX, VAR_NAMES, generate_operation

    L.append("## His embedded operation (goal_flip attempted on every handle; his score: "
             "CMI(active; A | B, E), 3 seeds, n=4000)\n")
    L.append("| Handle | score (mean) | permutation p95 | verdict |")
    L.append("|---|---|---|---|")
    rng = np.random.default_rng(99)
    for hname in VAR_NAMES:
        scores, floors = [], []
        for seed in range(3):
            data = generate_operation(seed, "goal_flip", IDX[hname], 4000)
            active = (data.op_type == data.op_names.index("goal_flip")).astype(np.int64)
            A = data.x[:, IDX["A"]].astype(np.int64)
            ctx = np.column_stack([data.x[:, IDX["B"]], data.x[:, IDX["E"]]])
            scores.append(cmi(active, A, ctx))
            floors.append(float(np.percentile(
                [cmi(rng.permutation(active), A, ctx) for _ in range(60)], 95)))
        m, fp = float(np.mean(scores)), float(np.mean(floors))
        L.append(f"| {hname} | {m:.4f} | {fp:.4f} | {'ROLE-EFFECT' if m > fp else '—'} |")
    L.append("")

    L.append("## Our do-style flip (1-agent engine, both probes, 3 seeds, n=2000)\n")
    L.append("| Target | role verdict | autonomy verdict | interventional verdict |")
    L.append("|---|---|---|---|")
    rows = [("G0", {}), ("B0", {}), ("E0", {}), ("S0", {}), ("A0", {}), ("D0", {}),
            ("G3 slow-puppet", {"puppet": ("slow", 3), "n_agents": 8, "coupling": 0.1,
                                "_target": "G3", "_n": 6000}),
            ("G2_alias(0%)", {"alias_noise": 0.0, "n_agents": 8, "coupling": 0.1,
                              "_target": "G2_alias", "_n": 2000})]
    for target, kw in rows:
        vs = []
        real_target = kw.pop("_target", target)
        n = kw.pop("_n", 2000)
        if "n_agents" not in kw:
            kw.update(n_agents=1, coupling=0.0)
        for seed in range(3):
            s = vd.yardstick_scores(seed=seed, target=real_target, n_steps=n, n_perms=200, **kw)
            vs.append(vd.yardstick_verdict(s))
        role = sum(v["role_effect"] for v in vs)
        auto = sum(v["autonomy_effect"] for v in vs)
        iv = sum(v["interventional_value"] for v in vs)
        L.append(f"| {target} | {role}/3 | {auto}/3 | {iv}/3 |")
    L.append("")
    L.append("Agreement check: his operation and ours must both say the goal-slot alone carries a "
             "role-effect among the handles; our autonomy probe must additionally clear the true "
             "goal (no autonomy effect) while flagging patch-driven variables.")
    return "\n".join(L)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for name, text in (("AUDIT_puppet_fast", audit_puppet("fast")),
                       ("AUDIT_puppet_slow", audit_puppet("slow")),
                       ("AUDIT_alias", audit_alias()),
                       ("BENCH_yardstick", bench())):
        (OUTDIR / f"{name}.md").write_text(text + "\n")
        print(text)
        print("\n" + "=" * 78 + "\n")


if __name__ == "__main__":
    main()
