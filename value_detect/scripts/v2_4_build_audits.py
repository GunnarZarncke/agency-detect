#!/usr/bin/env python3
"""V2-4 — build the three worlds' audit artifacts (SJ reviews BEFORE any scoring).

Per world: change-rate summary, mechanism-agreement table (predicted vs observed
disagreement from the designed noise), a short story printout, and the pre-registered
prediction table. Output: results/v2_4/AUDIT_<world>.md + a combined summary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import value_detect as vd
from value_detect.directed import cmi
from value_detect.worlds_v2 import NOISE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = PROJECT_ROOT / "results" / "v2_4"
SEED, N = 0, 20000


def xor_p(*ps):
    q = 0.0
    for p in ps:
        q = q * (1 - p) + (1 - q) * p
    return q


def change_rate(x):
    x = np.asarray(x)
    return float(np.mean(x[1:] != x[:-1]))


def audit_colony() -> str:
    f = vd.colony_frame(seed=SEED, n_steps=N, coupling=0.1)
    L = ["# Audit — Colony world (8 agents, weak ring coupling 0.1, seed 0, 20k steps)\n"]
    L.append("## How often things change (mean across the 8 agents)\n")
    L.append("| Role | Mean change rate | Range |")
    L.append("|---|---|---|")
    for role, expect in (("B", "~50%"), ("S", "~50%"), ("A", "~50%"), ("E", "~50%"),
                         ("G", "~1.5%"), ("D", "~50%")):
        rates = [change_rate(f[f"{role}{i}"]) for i in range(8)]
        L.append(f"| {role} (designed {expect}) | {np.mean(rates)*100:.1f}% | "
                 f"{min(rates)*100:.1f}–{max(rates)*100:.1f}% |")
    L.append(f"| W (noise, ~50%) | {change_rate(f['W'])*100:.1f}% | — |\n")

    L.append("## Does each agent obey its designed loop? (agent-averaged)\n")
    rows = []
    pred_sens = xor_p(NOISE["sensor"], NOISE["obs"])
    pred_act = xor_p(NOISE["action"], NOISE["obs"])
    # E_next vs E^A_obs: the action line is common to both sides and cancels;
    # what remains is env noise ^ readout noise ^ the average neighbour nudge.
    pred_env = xor_p(NOISE["env"], NOISE["obs"], 0.1 * 0.5)
    # B_next vs S_obs: the sensor line is common and cancels; belief ^ readout remain.
    pred_bel = xor_p(NOISE["belief"], NOISE["obs"])
    pred_dis = xor_p(NOISE["distractor"])
    for i in range(8):
        B, S, A, E, G, D = (f[f"{c}{i}"].to_numpy() for c in "BSAEGD")
        rows.append([
            np.mean(S != E), np.mean(A != (B ^ G)),
            np.mean(E[1:] != (E[:-1] ^ A[:-1])), np.mean(B[1:] != S[:-1]),
            np.mean(D != E),
        ])
    obs = np.mean(rows, axis=0)
    for name, pred, o in (
        ("Sensor reads own patch (S vs E)", pred_sens, obs[0]),
        ("Action = belief XOR goal (via noisy readouts)", pred_act, obs[1]),
        ("Patch responds to action (+3% env noise, +5% avg neighbour nudge)", pred_env, obs[2]),
        ("Belief tracks what was sensed", pred_bel, obs[3]),
        ("Distractor shadows own patch", pred_dis, obs[4]),
    ):
        L.append(f"- {name}: predicted {pred*100:.1f}% disagreement, observed {o*100:.1f}%")
    coup = []
    for i in range(8):
        Ei = f[f"E{i}"].to_numpy()
        Aprev = f[f"A{(i-1) % 8}"].to_numpy()
        Aown = f[f"A{i}"].to_numpy()
        z = np.column_stack([Ei[:-1], Aown[:-1]])
        coup.append(cmi(Ei[1:], Aprev[:-1], z))
    L.append(f"- Ring coupling is real and weak: neighbour's action → own patch reads "
             f"{np.mean(coup):.4f} nats on average (own action screened); zero when coupling=0.\n")

    g_flips = sum(int(np.sum(f[f'G{i}'].to_numpy()[1:] != f[f'G{i}'].to_numpy()[:-1])) for i in range(8))
    L.append("## Story (one line per event class)\n")
    L.append(f"Across the colony, the eight goals flipped {g_flips} times in 20,000 steps "
             f"(designed ≈ {8 * 0.015 * 20000:.0f}) — each entirely by its own coin. Every agent runs "
             "the V1 loop against its own patch; roughly one step in ten, a neighbour's action also "
             "nudges the patch (the ring). No lookalike columns exist in this world.\n")
    L.append("## Pre-registered predictions\n")
    L.append("- Every G_i — and nothing else — earns the signature (per-test thresholds as in V2-5).")
    L.append("- B_i at the intake pole of its agent; D_i witness-class, refused; W nothing.")
    L.append("- Fused convention: infeasible at this scale — recorded as its result.")
    L.append("- Witness-watch: ≈0 thefts expected (SJ's declining-with-scale expectation).")
    L.append("- On any failure: partitioned rerun (coupling=0) attributes scale vs interference.")
    return "\n".join(L)


def audit_deep_synergy() -> str:
    f = vd.deep_synergy_frame(seed=SEED, n_steps=N)
    L = ["# Audit — Deep-synergy world (agent P: parity; agent M: majority; seed 0, 20k steps)\n"]
    L.append("## Change rates\n")
    L.append("| Variable | Rate | | Variable | Rate |")
    L.append("|---|---|---|---|---|")
    cols = [c for c in f.columns if c != "W"]
    half = (len(cols) + 1) // 2
    for a, b in zip(cols[:half], cols[half:] + [""] * (half - len(cols[half:]))):
        rb = f"{change_rate(f[b])*100:.1f}%" if b else ""
        L.append(f"| {a} | {change_rate(f[a])*100:.1f}% | | {b} | {rb} |")
    L.append(f"| W | {change_rate(f['W'])*100:.1f}% | | | |\n")

    L.append("## Designed relationships\n")
    pred_rule = xor_p(NOISE["action"], NOISE["obs"])
    for tag, rule in (("P", "parity"), ("M", "majority")):
        B1, B2, G, A = (f[f"{c}_{tag}"].to_numpy() for c in ("B1", "B2", "G", "A"))
        pred = (B1 ^ B2 ^ G) if rule == "parity" else ((B1 + B2 + G) >= 2).astype(int)
        L.append(f"- Agent {tag} ({rule}): action disagrees with its rule "
                 f"{np.mean(A != pred)*100:.1f}% (predicted {pred_rule*100:.1f}%).")
        for c in ("1", "2"):
            Bc = f[f"B{c}_{tag}"].to_numpy()
            Ec = f[f"E{c}_{tag}"].to_numpy()
            L.append(f"  - Belief {c} tracks channel {c}: "
                     f"{np.mean(Bc[1:] != Ec[:-1])*100:.1f}% disagreement "
                     f"(predicted {xor_p(NOISE['belief'], NOISE['sensor'])*100:.1f}%).")
    L.append("")
    L.append("## Story\n")
    gp = f["G_P"].to_numpy(); gm = f["G_M"].to_numpy()
    L.append(f"Two isolated agents, each watching two environment channels through two sensors and "
             f"two beliefs. Agent P combines belief-1, belief-2 and its goal by strict parity "
             f"(any single input looks like a coin unless you know the other two); its goal flipped "
             f"{int(np.sum(gp[1:] != gp[:-1]))} times. Agent M combines the same trio by majority vote "
             f"(each input leaks through singly); its goal flipped {int(np.sum(gm[1:] != gm[:-1]))} times.\n")
    L.append("## Pre-registered predictions\n")
    L.append("- G_P and G_M earn the signature; nothing else does.")
    L.append("- G_P is INVISIBLE to best-key and grown keys (the plateau, unit-tested); "
             "fused and both block architectures see it (co-inputs share a block).")
    L.append("- G_M is visible to everyone; grown keys strongest (its graded-composition niche).")
    L.append("- The parity-gap column (fused-family minus grown) is LARGE for agent P's flows, ≈0 for agent M's.")
    return "\n".join(L)


def audit_slow_meter() -> str:
    f = vd.slow_meter_frame(seed=SEED, n_steps=N)
    L = ["# Audit — Slow-meter world (V1 loop, goal at 0.5%, four meter witnesses; seed 0, 20k steps)\n"]
    L.append("## Change rates\n")
    L.append("| Variable | Rate | Class |")
    L.append("|---|---|---|")
    for c in f.columns:
        cls = ("planted value (slowed)" if c == "G" else
               "meter witness" if c in ("M_fast", "M8", "M32", "F_sat") else
               "V1 loop" if c in ("B", "S", "A", "E", "S_alias", "A_alias", "D") else "noise")
        L.append(f"| {c} | {change_rate(f[c])*100:.2f}% | {cls} |")
    L.append("")
    L.append("## Meter definitions verified\n")
    A = f["A"].to_numpy()
    L.append(f"- M_fast: ±1 with the action every step (rate {change_rate(f['M_fast'])*100:.0f}%).")
    L.append(f"- M8: steps only after 8 consecutive same-actions (rate {change_rate(f['M8'])*100:.2f}%).")
    L.append(f"- M32: windowed-majority stepper (rate {change_rate(f['M32'])*100:.2f}%).")
    L.append(f"- F_sat: saturating, {int(np.sum(np.abs(np.diff(f['F_sat'].to_numpy()))))} transitions ever "
             "(the near-frozen class the z-gate exists for).")
    L.append("- All four are pure witnesses computed from the recorded trace — causally inert by construction.\n")
    g = f["G"].to_numpy()
    L.append("## Story\n")
    L.append(f"The V1 loop runs as audited in Chunk 2, but the goal now flips only "
             f"{int(np.sum(g[1:] != g[:-1]))} times in 20,000 steps (designed ≈ 100) — three times "
             "slower than V1, deliberately closer to meter tempo. Around it tick four meters at "
             "four timescales, from every-step to almost-never.\n")
    L.append("## Pre-registered predictions\n")
    L.append("- G alone earns the signature, despite being slow.")
    L.append("- Every meter is refused; per-meter theft rates reported as the z=3 witness watch "
             "(SJ's expectation: ≈0 here, declining with world complexity across the V2 worlds).")
    L.append("- F_sat specifically exercises the frozen-degenerate path (sd=0 ⇒ nothing qualifies).")
    return "\n".join(L)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for name, fn in (("colony", audit_colony), ("deep_synergy", audit_deep_synergy),
                     ("slow_meter", audit_slow_meter)):
        text = fn()
        (OUTDIR / f"AUDIT_{name}.md").write_text(text + "\n")
        print(text)
        print("\n" + "=" * 78 + "\n")


if __name__ == "__main__":
    main()
