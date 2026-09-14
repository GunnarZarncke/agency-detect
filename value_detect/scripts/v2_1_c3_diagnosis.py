#!/usr/bin/env python3
"""V2-1 Step 0 — verify the C3 failure mechanism from the stored chunk-5 data.

Question: in the seeds where `goal_progress` stole the value signature under
fused+best-key, which condition actually fired?
  Branch A — "floor swallowed the intake": net intake = 0 because the goal column's
              intake floor rose to meet its (real) intake.
  Branch B — "witness drive": net intake > 0 but net drive cleared 9x anyway.

Also: why did pairwise C3 pass in the same seeds, and how slow/autocorrelated are the
goal columns compared to ordinary ones? No new measurements of the world — only reads
of stored JSONs, plus one regenerated trace for autocorrelation context.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_DIR = PROJECT_ROOT / "results" / "chunk5"
OUT_DIR = PROJECT_ROOT / "results" / "v2_1"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = {}
    for p in sorted(IN_DIR.glob("c3_seed*.json")):
        d = json.loads(p.read_text())
        records[d["seed"]] = d
    seeds = sorted(records)
    goal_cols = records[seeds[0]]["goal_cols"]

    rows = []
    for s in seeds:
        for conv in ("pairwise", "fused_bestkey"):
            t = records[s]["tests"][conv]
            floors = {r["variable"]: r for r in t["floors"]}
            scores = {r["variable"]: r for r in t["scores"]}
            sig = {r["variable"]: r for r in t["criteria"]["signature_table"]}
            for v in goal_cols:
                sc, fl, sg = scores[v], floors[v], sig[v]
                rows.append({
                    "seed": s, "convention": conv, "variable": v,
                    "push_in": sc["push_in"], "in_floor95": fl["push_in_p95"],
                    "out_sys": sc["out_sys"], "out_floor95": fl["out_sys_p95"],
                    "net_in": sg["net_in"], "net_out": sg["net_out"],
                    "intake_at_floor": sg["intake_at_floor"], "ratio_ok": sg["ratio_ok"],
                    "drive_above": sg["drive_above_floor"], "rankable": sg["rankable"],
                    "signature": sg["signature"],
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "c3_per_seed_breakdown.csv", index=False)

    L = ["# C3 mechanism verification (from stored chunk-5 data; no new measurements)\n"]
    L.append(f"Seeds: {len(seeds)}; goal columns: {', '.join(goal_cols)}.\n")

    for conv in ("fused_bestkey", "pairwise"):
        sub = df[df.convention == conv]
        stolen = sub[sub.signature]
        L.append(f"## {conv}\n")
        L.append(f"- Seed-variable cases with the signature: **{len(stolen)}** "
                 f"(seeds: {sorted(set(stolen.seed))})")
        if len(stolen):
            branch_a = stolen[stolen.intake_at_floor]
            branch_b = stolen[~stolen.intake_at_floor]
            L.append(f"- **Branch A (floor swallowed intake; net intake = 0): {len(branch_a)} cases**")
            L.append(f"- **Branch B (net intake > 0 but 9x ratio met anyway): {len(branch_b)} cases**")
            for _, r in stolen.iterrows():
                L.append(
                    f"  - seed {r.seed} {r.variable}: intake {r.push_in:.4f} vs floor {r.in_floor95:.4f} "
                    f"(net {r.net_in:.4f}); drive {r.out_sys:.4f} vs floor {r.out_floor95:.4f} "
                    f"(net {r.net_out:.4f}); branch {'A' if r.intake_at_floor else 'B'}"
                )
        m = sub.groupby("variable")[["push_in", "in_floor95", "out_sys", "out_floor95"]].mean()
        L.append("\nMeans across all seeds:\n")
        L.append("| Variable | intake | intake floor | drive | drive floor |")
        L.append("|---|---|---|---|---|")
        for v, r in m.iterrows():
            L.append(f"| {v} | {r.push_in:.4f} | {r.in_floor95:.4f} | {r.out_sys:.4f} | {r.out_floor95:.4f} |")
        L.append("")

    # Context: how slow are the goal columns? One regenerated trace, descriptive only.
    np.random.seed(seeds[0])
    from agency_detect.agents import generate_decoupled_trace
    frame = pd.DataFrame(generate_decoupled_trace(steps=20000, n_solar_panels=1,
                                                  factory_materials=["steel"])).astype(int)
    L.append("## How slow are these variables? (seed 0 trace, descriptive)\n")
    L.append("| Variable | change rate | lag-1 autocorr | lag-10 autocorr | cardinality |")
    L.append("|---|---|---|---|---|")
    goal_like = goal_cols + [c for c in frame.columns if c.endswith("_sensor") or c.endswith("_action")][:2]
    for v in goal_like:
        x = frame[v].to_numpy().astype(float)
        cr = float(np.mean(x[1:] != x[:-1]))
        xc = x - x.mean()
        denom = float(np.sum(xc * xc)) or 1.0
        ac1 = float(np.sum(xc[1:] * xc[:-1]) / denom)
        ac10 = float(np.sum(xc[10:] * xc[:-10]) / denom)
        L.append(f"| {v} | {cr:.3f} | {ac1:.3f} | {ac10:.3f} | {frame[v].nunique()} |")
    L.append("")

    memo = "\n".join(L)
    (OUT_DIR / "C3_MECHANISM.md").write_text(memo + "\n")
    print(memo)


if __name__ == "__main__":
    main()
