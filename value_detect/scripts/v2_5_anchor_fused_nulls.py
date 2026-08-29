#!/usr/bin/env python3
"""V2-5 — regenerate the anchor fused null SAMPLES for registered cross-seed pooling.

Identical RNG streams as the sweep's null_floors calls (same seed formula, same sampler
order), so each unit self-verifies its derived p95s against the stored sweep floors.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import value_detect as vd
from value_detect.floors import ConventionScorer, shift_null_samples

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = PROJECT_ROOT / "results" / "v2_5"
NFUSED = 2_000_000


def run_unit(task) -> str:
    seed, condition = task
    outpath = OUTDIR / f"anchorfusednulls_{condition}_seed{seed}.json"
    if outpath.exists():
        return f"skip {condition} s{seed}"
    t0 = time.time()
    if condition == "main":
        ffu = vd.passive_trace(seed=seed, n_steps=NFUSED).frame
    elif condition == "nocore":
        ffu = vd.passive_trace_nocore(seed=seed, n_steps=NFUSED).frame
    else:
        ffu = vd.scramble_frame(vd.passive_trace(seed=seed, n_steps=NFUSED).frame, seed=seed + 9000)
    scorer = ConventionScorer(ffu, "fused", env_var="E")
    samples = shift_null_samples(scorer, n_shifts=50, seed=seed * 97 + 13)

    stored = json.loads((OUTDIR / f"anchor_{condition}_seed{seed}.json").read_text())
    stored_floors = {r["variable"]: r for r in stored["tests"]["fused"]["floors"]}
    mism = 0
    for v, g in samples.groupby("variable"):
        p95 = float(np.nanpercentile(g["out_sys"].to_numpy(dtype=float), 95))
        if abs(p95 - stored_floors[v]["out_sys_p95"]) > 1e-9:
            mism += 1
    outpath.write_text(json.dumps({"seed": seed, "condition": condition,
                                   "p95_mismatches": mism,
                                   "samples": samples.to_dict("records")}))
    return f"done {condition} s{seed} in {time.time()-t0:.0f}s (mismatches {mism})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    tasks = [(s, c) for s in range(args.seeds) for c in ("main", "nocore", "scramble")]
    print(f"[anchor-fused-nulls] {len(tasks)} units", flush=True)
    t0 = time.time()
    with Pool(processes=args.jobs, maxtasksperchild=1) as pool:
        for i, msg in enumerate(pool.imap_unordered(run_unit, tasks), 1):
            print(f"[{i}/{len(tasks)}] {msg} | {(time.time()-t0)/60:.1f}m", flush=True)
    print(f"ANCHOR FUSED NULLS COMPLETE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
