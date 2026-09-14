#!/usr/bin/env python3
"""Chunk 4b — diagnosing why A's measured intake is tiny (SJ's question).

Three numbered exhibits, all on real or minimal data:

  1. THE CIPHER: on the real trace, each single input to the action reads ~zero
     (pairwise), but the inputs measured JOINTLY read large — and match the value
     predicted by hand from the world's noise rates. XOR-style combination hides
     each input unless the other (the "key") is known.
  2. SJ'S OPTION 1 TESTED: a minimal variant world where the decision lags one tick
     (action at t+1 computed from belief and goal at t). Pairwise intake is STILL
     dark; joint is bright. So re-timing the world would not fix the reading.
  3. THE DECRYPTION REPAIR (option 2): conditioning on one extra variable (the key)
     reveals the hidden flows — belief's and goal's drive on the environment appear
     at their hand-predicted sizes — while conditioning on a mediator (the action
     wire) wrongly kills them, and the distractor's fake drive stays dead. This maps
     exactly which conditioning conventions see what.

Output: results/chunk4/DIAGNOSIS.md
"""

from __future__ import annotations

from math import log
from pathlib import Path

import numpy as np

import value_detect as vd
from value_detect.directed import binary_entropy, transfer_entropy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = PROJECT_ROOT / "results" / "chunk4"
LN2 = log(2.0)


def _xor_p(*ps: float) -> float:
    q = 0.0
    for p in ps:
        q = q * (1 - p) + (1 - q) * p
    return q


def lagged_decision_world(seed: int, T: int):
    """Minimal variant of the loop where the DECISION LAGS one tick (SJ's option 1).

    A(t+1) = B(t) XOR G(t) XOR 4%-noise; E(t+1) = E(t) XOR A(t) XOR 3%;
    B(t+1) = sensor-read-of-E(t) XOR 3%; G flips by its own 1.5% coin.
    Our own tiny simulation — Gunnar's files untouched.
    """
    rng = np.random.default_rng(seed)
    E = np.empty(T, dtype=int)
    B = np.empty(T, dtype=int)
    G = np.empty(T, dtype=int)
    A = np.empty(T, dtype=int)
    E[0], B[0], G[0], A[0] = rng.integers(0, 2, 4)
    for t in range(T - 1):
        s_line = E[t] ^ (rng.random() < 0.05)
        A[t + 1] = B[t] ^ G[t] ^ (rng.random() < 0.04)
        E[t + 1] = E[t] ^ A[t] ^ (rng.random() < 0.03)
        B[t + 1] = s_line ^ (rng.random() < 0.03)
        G[t + 1] = G[t] ^ (rng.random() < 0.015)
    return {"B": B, "G": G, "A": A, "E": E}


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    trace = vd.passive_trace(seed=0, n_steps=20000)
    f = trace.frame
    a = {c: f[c].to_numpy() for c in f.columns}

    L = []
    L.append("# Diagnosis: why the action's measured intake was tiny\n")
    L.append("All numbers in nats (one bit = 0.693). 'Hand-predicted' values are derived from the")
    L.append("world's designed noise rates before measuring.\n")

    # ---- Exhibit 1: the cipher on the real trace ----
    pair_B = transfer_entropy(a["B"], a["A"])
    pair_G = transfer_entropy(a["G"], a["A"])
    pair_Sa = transfer_entropy(a["S_alias"], a["A"])
    joint_src = np.column_stack([a["S_alias"], a["G"]])
    joint = transfer_entropy(joint_src, a["A"])
    # A(t+1) = B(t+1) xor G(t+1) xor 4% xor 6%; B(t+1) = S_alias(t) xor 3%; G(t+1)=G(t) but 1.5%.
    hand = LN2 - binary_entropy(_xor_p(0.03, 0.015, 0.04, 0.06))
    L.append("## Exhibit 1 — the inputs are there, but encrypted (real trace)\n")
    L.append("| Measurement of tomorrow's action | Reading |")
    L.append("|---|---|")
    L.append(f"| From belief alone (pairwise) | {pair_B:.4f} |")
    L.append(f"| From goal alone (pairwise) | {pair_G:.4f} |")
    L.append(f"| From sensor-line alone (pairwise) | {pair_Sa:.4f} |")
    L.append(f"| From sensor-line AND goal **jointly** | **{joint:.4f}** |")
    L.append(f"| Hand-predicted joint value from noise rates | {hand:.4f} |")
    L.append("")
    mega = vd.push_in_megastate(f)
    L.append(f"(The Chunk 4 mega-state cross-check read A's intake as {mega['A']:.4f} — same story; "
             f"G's stayed at the bias floor, {mega['G']:.4f}, alongside pure noise W at {mega['W']:.4f}.)\n")

    # ---- Exhibit 2: SJ's option 1 (re-time the world) tested ----
    v = lagged_decision_world(seed=11, T=20000)
    lag_pair_B = transfer_entropy(v["B"], v["A"])
    lag_pair_G = transfer_entropy(v["G"], v["A"])
    lag_joint = transfer_entropy(np.column_stack([v["B"], v["G"]]), v["A"])
    lag_hand = LN2 - binary_entropy(0.04)
    L.append("## Exhibit 2 — re-timing the world does NOT fix it (variant simulation)\n")
    L.append("Variant loop in which the decision lags a full tick (tomorrow's action is computed")
    L.append("from today's belief and goal), exactly as proposed:\n")
    L.append("| Measurement of tomorrow's action | Reading |")
    L.append("|---|---|")
    L.append(f"| From belief alone (pairwise) | {lag_pair_B:.4f} |")
    L.append(f"| From goal alone (pairwise) | {lag_pair_G:.4f} |")
    L.append(f"| From belief AND goal **jointly** | **{lag_joint:.4f}** |")
    L.append(f"| Hand-predicted joint value | {lag_hand:.4f} |")
    L.append("")
    L.append("Each input alone still reads ~zero even with the lag in place: the hiding is done by")
    L.append("the XOR combination (each input is a cipher key for the other), not by the timing.\n")

    # ---- Exhibit 3: decryption repair for outbound flows ----
    L.append("## Exhibit 3 — one extra conditioning variable decrypts the outbound flows (real trace)\n")
    rows = [
        ("Belief drives environment, measured naively", transfer_entropy(a["B"], a["E"]), None),
        ("... decrypted with the goal as key", transfer_entropy(a["B"], a["E"], cond=a["G"]),
         LN2 - binary_entropy(_xor_p(0.04, 0.03))),
        ("Goal drives environment, measured naively", transfer_entropy(a["G"], a["E"]), None),
        ("... decrypted with the belief as key", transfer_entropy(a["G"], a["E"], cond=a["B"]),
         LN2 - binary_entropy(_xor_p(0.04, 0.03))),
        ("Goal drives environment, wrongly conditioned on the action wire (mediator)",
         transfer_entropy(a["G"], a["E"], cond=a["A_alias"]), None),
        ("... conditioned on the noisy action readout (partial mediator screen)",
         transfer_entropy(a["G"], a["E"], cond=a["A"]), None),
        ("Distractor 'drives' environment, naively", transfer_entropy(a["D"], a["E"]), None),
        ("... with belief as key (fake drive must stay dead)", transfer_entropy(a["D"], a["E"], cond=a["B"]), None),
        ("Goal's grip on tomorrow's action, decrypted by the sensor line",
         transfer_entropy(a["G"], a["A"], cond=a["S_alias"]),
         LN2 - binary_entropy(_xor_p(0.03, 0.015, 0.04, 0.06))),
    ]
    L.append("| Measurement | Reading | Hand-predicted |")
    L.append("|---|---|---|")
    for name, val, pred in rows:
        L.append(f"| {name} | {val:.4f} | {('%.4f' % pred) if pred is not None else '—'} |")
    L.append("")
    L.append("Reading the exhibits together: single-variable (pairwise) conventions under-read every")
    L.append("flow that passes through an XOR-style combination; one well-chosen extra conditioning")
    L.append("variable restores it to its true size; conditioning on a mediator destroys it (as the")
    L.append("design warned); and decoy 'witness' flows stay dead under decryption. The instrument")
    L.append("family is sound — the choice of conditioning convention decides what it can see.")

    memo = "\n".join(L)
    (OUTDIR / "DIAGNOSIS.md").write_text(memo + "\n")
    print(memo)


if __name__ == "__main__":
    main()
