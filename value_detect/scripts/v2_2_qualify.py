#!/usr/bin/env python3
"""V2-2 qualification gate — grown keys against the full V1 criteria.

Grown-keys convention (k_max=2, max over stages) on the V1 world: {main, nocore,
scramble} × {as-is, lookalike-free} × 20 seeds, plus C3 with the V2-1 harness hygiene
(2,000-step burn-in; PYTHONHASHSEED=0). Floors: 200 circular shifts with null statistics;
criteria evaluated at z=3 (stored raw for reanalysis). Pre-registered expected failures:
none on the main world; C3 expected to FAIL (witness inheritance, DECISIONS 2026-08-11).
"""

from __future__ import annotations

import os

os.environ["PYTHONHASHSEED"] = "0"

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import value_detect as vd
from value_detect.criteria import (
    evaluate_goalprogress_test,
    evaluate_main_test,
    evaluate_nocore_test,
    evaluate_scramble_test,
)
from value_detect.floors import ConventionScorer, null_floors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = PROJECT_ROOT / "results" / "v2_2"

Z_MIN = 3.0
CONV = "grown_keys"
BURN_IN = 2000


def _frames_for(condition: str, seed: int, n20: int):
    if condition == "main":
        return vd.passive_trace(seed=seed, n_steps=n20).frame
    if condition == "nocore":
        return vd.passive_trace_nocore(seed=seed, n_steps=n20).frame
    return vd.scramble_frame(vd.passive_trace(seed=seed, n_steps=n20).frame, seed=seed + 9000)


def run_world(task) -> str:
    seed, condition, n20, shifts = task
    outpath = Path(OUTDIR) / f"{condition}_seed{seed}.json"
    if outpath.exists():
        return f"skip {condition} s{seed}"
    t0 = time.time()
    frame_a = _frames_for(condition, seed, n20)
    rec = {"seed": seed, "condition": condition, "convention": CONV, "z_min": Z_MIN, "tests": {}}
    for config in ("asis", "noalias"):
        frame = frame_a if config == "asis" else vd.drop_aliases(frame_a)
        scorer = ConventionScorer(frame, CONV, env_var="E", key_depth=2)
        scores = scorer.score_all()
        floors = null_floors(scorer, n_shifts=shifts, seed=seed * 97 + 13)
        if condition == "main":
            ev = evaluate_main_test(scores, floors, is_bestkey=True, z_min=Z_MIN)
        elif condition == "nocore":
            ev = evaluate_nocore_test(scores, floors, z_min=Z_MIN)
        else:
            ev = evaluate_scramble_test(scores, floors, z_min=Z_MIN)
        rec["tests"][config] = {"scores": scores.reset_index().to_dict("records"),
                                "floors": floors.reset_index().to_dict("records"),
                                "criteria": _jsonable(ev)}
        del scorer
    outpath.write_text(json.dumps(rec))
    return f"done {condition} s{seed} in {time.time()-t0:.0f}s"


def run_c3(task) -> str:
    seed, steps, shifts = task
    outpath = Path(OUTDIR) / f"c3_seed{seed}.json"
    if outpath.exists():
        return f"skip c3 s{seed}"
    t0 = time.time()
    assert os.environ.get("PYTHONHASHSEED") == "0"
    np.random.seed(seed)
    from agency_detect.agents import generate_decoupled_trace
    raw = pd.DataFrame(generate_decoupled_trace(steps=steps + BURN_IN, n_solar_panels=1,
                                                factory_materials=["steel"])).astype(int)
    frame = raw.iloc[BURN_IN:].reset_index(drop=True)
    goal_cols = [c for c in frame.columns if c.endswith("_goal")]
    scorer = ConventionScorer(frame, CONV, env_var=None, key_depth=2)
    scores = scorer.score_all()
    floors = null_floors(scorer, n_shifts=shifts, seed=seed * 89 + 7, variables=goal_cols)
    ev = evaluate_goalprogress_test(scores, floors, goal_cols, z_min=Z_MIN)
    rec = {"seed": seed, "convention": CONV, "burn_in": BURN_IN, "goal_cols": goal_cols,
           "scores": scores.reset_index().to_dict("records"),
           "floors": floors.reset_index().to_dict("records"), "criteria": _jsonable(ev)}
    outpath.write_text(json.dumps(rec))
    return f"done c3 s{seed} in {time.time()-t0:.0f}s"


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--n20", type=int, default=20_000)
    parser.add_argument("--shifts", type=int, default=200)
    args = parser.parse_args()
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)

    tasks = []
    for seed in range(args.seeds):
        for condition in ("main", "nocore", "scramble"):
            tasks.append(("w", (seed, condition, args.n20, args.shifts)))
        tasks.append(("c", (seed, args.n20, args.shifts)))
    fn = {"w": run_world, "c": run_c3}
    print(f"[v2_2] {len(tasks)} units, jobs={args.jobs}, convention={CONV}, z={Z_MIN}", flush=True)
    t0 = time.time()
    with Pool(processes=args.jobs, maxtasksperchild=1) as pool:
        results = [pool.apply_async(fn[k], (p,)) for k, p in tasks]
        for i, r in enumerate(results, 1):
            print(f"[v2_2 {i}/{len(tasks)}] {r.get()} | {(time.time()-t0)/60:.1f}m", flush=True)
    print(f"V2-2 QUALIFICATION COMPLETE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
