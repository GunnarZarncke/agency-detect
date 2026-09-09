#!/usr/bin/env python3
"""V2-1 calibration suite (background workhorse).

Produces the raw material for the flip-report that decides the floor fix:

  A. V1 floors recomputed WITH null statistics (mean/sd), roll sampler, identical RNG
     streams as the chunk-5 sweep — each unit verifies its 95th percentiles match the
     stored sweep values (stream-compatibility check), for pairwise and fused+best-key
     across {main, nocore, scramble} × 20 seeds.
  B. The same grid under the TRANSITION-SURROGATE null family.
  C. C3 (`goal_progress` world) goal-column floors under both families.
  D. The calibration world (slow value + slow meter + frozen variable) under both
     families, decision variables {G, M_slow, F_frozen, W, B}.

Scores are NOT recomputed where the sweep already stored them (A–C reuse chunk-5
scores); D computes its own. All units resumable; outputs in results/v2_1/.
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
from value_detect.floors import ConventionScorer, null_floors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNK5 = PROJECT_ROOT / "results" / "chunk5"
OUTDIR = PROJECT_ROOT / "results" / "v2_1"

CONVS = ("pairwise", "fused_bestkey")
CONFIGS = ("asis", "noalias")
CALIB_VARS = ["G", "M_slow", "F_frozen", "W", "B"]


def _floors_to_records(df: pd.DataFrame):
    return df.reset_index().to_dict("records")


def _check_p95(new: pd.DataFrame, stored_records, tol=1e-9) -> int:
    stored = {r["variable"]: r for r in stored_records}
    mismatches = 0
    for v in new.index:
        for col in ("push_in_p95", "out_sys_p95", "out_env_p95", "total_flow_p95"):
            a, b = new.loc[v, col], stored[v][col]
            if b is None or (isinstance(b, float) and np.isnan(b)):
                continue
            if not np.isfinite(a) or abs(a - b) > tol:
                mismatches += 1
    return mismatches


def _frames_for(condition: str, seed: int, n20: int):
    if condition == "main":
        return vd.passive_trace(seed=seed, n_steps=n20).frame
    if condition == "nocore":
        return vd.passive_trace_nocore(seed=seed, n_steps=n20).frame
    return vd.scramble_frame(vd.passive_trace(seed=seed, n_steps=n20).frame, seed=seed + 9000)


def run_v1floors(task) -> str:
    seed, condition, sampler, n20, shifts = task
    outpath = OUTDIR / f"floorstats_{sampler}_{condition}_seed{seed}.json"
    if outpath.exists():
        return f"skip v1floors {sampler} {condition} s{seed}"
    t0 = time.time()
    frame_a = _frames_for(condition, seed, n20)
    stored = json.loads((CHUNK5 / f"{condition}_seed{seed}.json").read_text())
    rec = {"seed": seed, "condition": condition, "sampler": sampler, "tests": {}, "p95_mismatches": {}}
    for config in CONFIGS:
        frame = frame_a if config == "asis" else vd.drop_aliases(frame_a)
        for conv in CONVS:
            scorer = ConventionScorer(frame, conv, env_var="E")
            fl = null_floors(scorer, n_shifts=shifts, seed=seed * 97 + 13, sampler=sampler)
            key = f"{config}_{conv}"
            rec["tests"][key] = _floors_to_records(fl)
            if sampler == "roll":
                rec["p95_mismatches"][key] = _check_p95(fl, stored["tests"][key]["floors"])
            del scorer
    outpath.write_text(json.dumps(rec))
    mism = sum(rec["p95_mismatches"].values()) if sampler == "roll" else "-"
    return f"done v1floors {sampler} {condition} s{seed} in {time.time()-t0:.0f}s (p95 mismatches: {mism})"


def run_c3floors(task) -> str:
    seed, sampler, steps, shifts = task
    outpath = OUTDIR / f"c3floorstats_{sampler}_seed{seed}.json"
    if outpath.exists():
        return f"skip c3floors {sampler} s{seed}"
    t0 = time.time()
    np.random.seed(seed)
    from agency_detect.agents import generate_decoupled_trace
    frame = pd.DataFrame(generate_decoupled_trace(steps=steps, n_solar_panels=1,
                                                  factory_materials=["steel"])).astype(int)
    goal_cols = [c for c in frame.columns if c.endswith("_goal")]
    stored = json.loads((CHUNK5 / f"c3_seed{seed}.json").read_text())
    rec = {"seed": seed, "sampler": sampler, "goal_cols": goal_cols, "tests": {}, "p95_mismatches": {}}
    for conv in CONVS:
        scorer = ConventionScorer(frame, conv, env_var=None)
        fl = null_floors(scorer, n_shifts=shifts, seed=seed * 89 + 7, sampler=sampler, variables=goal_cols)
        rec["tests"][conv] = _floors_to_records(fl)
        if sampler == "roll":
            rec["p95_mismatches"][conv] = _check_p95(fl, stored["tests"][conv]["floors"])
        del scorer
    outpath.write_text(json.dumps(rec))
    return f"done c3floors {sampler} s{seed} in {time.time()-t0:.0f}s"


def run_calib(task) -> str:
    seed, sampler, steps, shifts = task
    outpath = OUTDIR / f"calib_{sampler}_seed{seed}.json"
    if outpath.exists():
        return f"skip calib {sampler} s{seed}"
    t0 = time.time()
    frame = vd.calibration_frame(seed=seed, n_steps=steps)
    rec = {"seed": seed, "sampler": sampler, "tests": {}}
    for conv in CONVS:
        scorer = ConventionScorer(frame, conv, env_var="E")
        scores = scorer.score_all()
        fl = null_floors(scorer, n_shifts=shifts, seed=seed * 61 + 3, sampler=sampler, variables=CALIB_VARS)
        rec["tests"][conv] = {"scores": scores.reset_index().to_dict("records"),
                              "floors": _floors_to_records(fl)}
        del scorer
    outpath.write_text(json.dumps(rec))
    return f"done calib {sampler} s{seed} in {time.time()-t0:.0f}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--n20", type=int, default=20_000)
    parser.add_argument("--shifts", type=int, default=200)
    args = parser.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for seed in range(args.seeds):
        for sampler in ("roll", "transition"):
            for condition in ("main", "nocore", "scramble"):
                tasks.append(("v1", (seed, condition, sampler, args.n20, args.shifts)))
            tasks.append(("c3", (seed, sampler, args.n20, args.shifts)))
            tasks.append(("cal", (seed, sampler, args.n20, args.shifts)))

    fn = {"v1": run_v1floors, "c3": run_c3floors, "cal": run_calib}
    total = len(tasks)
    print(f"[v2_1] {total} units, jobs={args.jobs}", flush=True)
    t0 = time.time()
    with Pool(processes=args.jobs, maxtasksperchild=1) as pool:
        results = [pool.apply_async(fn[kind], (payload,)) for kind, payload in tasks]
        for i, r in enumerate(results, 1):
            msg = r.get()
            el = time.time() - t0
            print(f"[v2_1 {i}/{total}] {msg} | {el/60:.1f}m elapsed, eta {el/i*(total-i)/60:.0f}m", flush=True)
    print(f"CALIBRATION SUITE COMPLETE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
