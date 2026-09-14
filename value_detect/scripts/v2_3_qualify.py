#!/usr/bin/env python3
"""V2-3 qualification gate — fused-agents (both architectures) against the V1 criteria.

Per unit: detect the partition once (swept selector — Gunnar's detector at every dial,
dial chosen by his blanket-validity coverage), then score BOTH architectures (keyring,
menu) with z=3 floors. {main, nocore, scramble} × 20 seeds + C3 (burn-in 2000,
PYTHONHASHSEED=0). Registered expectations: main criteria pass; C3 PASS for keyring
(structural witness screen); scramble partitions are degenerate by construction (honest:
the control tests floors, not detection).
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
from value_detect.agentblocks import BlockScorer, detect_blocks_swept
from value_detect.criteria import (
    evaluate_goalprogress_test,
    evaluate_main_test,
    evaluate_nocore_test,
    evaluate_scramble_test,
)
from value_detect.floors import null_floors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = PROJECT_ROOT / "results" / "v2_3"

Z_MIN = 3.0
ARCHS = ("keyring", "menu")
BURN_IN = 2000


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


def _frames_for(condition: str, seed: int, n20: int):
    if condition == "main":
        return vd.passive_trace(seed=seed, n_steps=n20).frame
    if condition == "nocore":
        return vd.passive_trace_nocore(seed=seed, n_steps=n20).frame
    return vd.scramble_frame(vd.passive_trace(seed=seed, n_steps=n20).frame, seed=seed + 9000)


def run_world(task) -> str:
    seed, condition, n20, shifts = task
    outpath = OUTDIR / f"{condition}_seed{seed}.json"
    if outpath.exists():
        return f"skip {condition} s{seed}"
    t0 = time.time()
    frame = _frames_for(condition, seed, n20)
    part = detect_blocks_swept(frame)
    rec = {"seed": seed, "condition": condition, "z_min": Z_MIN,
           "partition": {k: part[k] for k in ("agents", "env", "orphans", "dial")}, "tests": {}}
    for arch in ARCHS:
        scorer = BlockScorer(frame, arch, env_var="E", partition=part)
        scores = scorer.score_all()
        floors = null_floors(scorer, n_shifts=shifts, seed=seed * 97 + 13)
        if condition == "main":
            ev = evaluate_main_test(scores, floors, is_bestkey=True, z_min=Z_MIN)
        elif condition == "nocore":
            ev = evaluate_nocore_test(scores, floors, z_min=Z_MIN)
        else:
            ev = evaluate_scramble_test(scores, floors, z_min=Z_MIN)
        rec["tests"][arch] = {"scores": scores.reset_index().to_dict("records"),
                              "floors": floors.reset_index().to_dict("records"),
                              "lost_mass": scorer.lost_mass,
                              "criteria": _jsonable(ev)}
        del scorer
    outpath.write_text(json.dumps(_jsonable(rec)))
    return f"done {condition} s{seed} (dial {part['dial']}) in {time.time()-t0:.0f}s"


def run_c3(task) -> str:
    seed, steps, shifts = task
    outpath = OUTDIR / f"c3_seed{seed}.json"
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
    part = detect_blocks_swept(frame)
    rec = {"seed": seed, "burn_in": BURN_IN, "goal_cols": goal_cols,
           "partition": {k: part[k] for k in ("agents", "env", "orphans", "dial")}, "tests": {}}
    for arch in ARCHS:
        scorer = BlockScorer(frame, arch, env_var=None, partition=part)
        scores = scorer.score_all()
        floors = null_floors(scorer, n_shifts=shifts, seed=seed * 89 + 7, variables=goal_cols)
        ev = evaluate_goalprogress_test(scores, floors, goal_cols, z_min=Z_MIN)
        rec["tests"][arch] = {"scores": scores.reset_index().to_dict("records"),
                              "floors": floors.reset_index().to_dict("records"),
                              "lost_mass": scorer.lost_mass,
                              "criteria": _jsonable(ev)}
        del scorer
    outpath.write_text(json.dumps(_jsonable(rec)))
    return f"done c3 s{seed} (dial {part['dial']}) in {time.time()-t0:.0f}s"


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
        for condition in ("main", "nocore", "scramble"):
            tasks.append(("w", (seed, condition, args.n20, args.shifts)))
        tasks.append(("c", (seed, args.n20, args.shifts)))
    fn = {"w": run_world, "c": run_c3}
    print(f"[v2_3] {len(tasks)} units, jobs={args.jobs}, archs={ARCHS}, z={Z_MIN}", flush=True)
    t0 = time.time()
    with Pool(processes=args.jobs, maxtasksperchild=1) as pool:
        results = [pool.apply_async(fn[k], (p,)) for k, p in tasks]
        for i, r in enumerate(results, 1):
            print(f"[v2_3 {i}/{len(tasks)}] {r.get()} | {(time.time()-t0)/60:.1f}m", flush=True)
    print(f"V2-3 QUALIFICATION COMPLETE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
