#!/usr/bin/env python3
"""V2-1 — C3 rerun with the two verified harness fixes.

1. PYTHONHASHSEED pinned to 0 (set below, before the worker pool spawns), making
   Gunnar's simulator reproducible across processes.
2. Burn-in: 22,000 steps generated, the first 2,000 discarded (stationary trace, same
   20k analysis length as before).

Scores and floors are computed together per unit (same process, same trace), for both
null families, goal columns only. Outputs c3v2_{sampler}_seed{N}.json in results/v2_1/.
"""

from __future__ import annotations

import os

os.environ["PYTHONHASHSEED"] = "0"  # must precede Pool spawn; workers inherit it.

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from value_detect.floors import ConventionScorer, null_floors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = PROJECT_ROOT / "results" / "v2_1"

BURN_IN = 2000
CONVS = ("pairwise", "fused_bestkey")


def run_unit(task) -> str:
    seed, sampler, steps, shifts = task
    outpath = OUTDIR / f"c3v2_{sampler}_seed{seed}.json"
    if outpath.exists():
        return f"skip {sampler} s{seed}"
    t0 = time.time()
    assert os.environ.get("PYTHONHASHSEED") == "0", "hash seed not pinned in worker"
    np.random.seed(seed)
    from agency_detect.agents import generate_decoupled_trace

    raw = pd.DataFrame(generate_decoupled_trace(steps=steps + BURN_IN, n_solar_panels=1,
                                                factory_materials=["steel"])).astype(int)
    frame = raw.iloc[BURN_IN:].reset_index(drop=True)
    goal_cols = [c for c in frame.columns if c.endswith("_goal")]
    rec = {"seed": seed, "sampler": sampler, "burn_in": BURN_IN, "steps": steps,
           "goal_cols": goal_cols, "tests": {}}
    for conv in CONVS:
        scorer = ConventionScorer(frame, conv, env_var=None)
        scores = scorer.score_all()
        floors = null_floors(scorer, n_shifts=shifts, seed=seed * 89 + 7,
                             sampler=sampler, variables=goal_cols)
        rec["tests"][conv] = {"scores": scores.reset_index().to_dict("records"),
                              "floors": floors.reset_index().to_dict("records")}
        del scorer
    outpath.write_text(json.dumps(rec))
    return f"done {sampler} s{seed} in {time.time()-t0:.0f}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--shifts", type=int, default=200)
    args = parser.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    tasks = [(seed, sampler, args.steps, args.shifts)
             for seed in range(args.seeds) for sampler in ("roll", "transition")]
    print(f"[c3v2] {len(tasks)} units, jobs={args.jobs}, PYTHONHASHSEED=0, burn-in={BURN_IN}", flush=True)
    t0 = time.time()
    with Pool(processes=args.jobs, maxtasksperchild=1) as pool:
        for i, msg in enumerate(pool.imap_unordered(run_unit, tasks), 1):
            print(f"[c3v2 {i}/{len(tasks)}] {msg} | {(time.time()-t0)/60:.1f}m", flush=True)
    print(f"C3 RERUN COMPLETE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
